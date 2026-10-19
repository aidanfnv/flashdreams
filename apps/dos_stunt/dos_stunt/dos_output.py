# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DOS-specific VGA conversion, Ethernet delivery, and emulator lifecycle."""

from __future__ import annotations

import fcntl
import math
import os
import select
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from dos_stunt.ethernet_transport import (
    BROADCAST_MAC,
    CHUNK_SIZE,
    ETHERTYPE,
    FRAME_SIZE,
    GUEST_MAC,
    HOST_MAC,
    PALETTE_SIZE,
    EthernetLink,
    physical_link,
    read_acks,
    send_frame,
    send_stop,
)
from PIL import Image, ImageEnhance

from dos_stunt.keyboard import KEY_BYTES, pressed_keys


def rgb24_to_vga(frame: np.ndarray, *, palette: str = "adaptive") -> bytes:
    """Quantize one RGB frame into VGA indices and an optional six-bit palette."""
    if frame.shape != (200, 320, 3) or frame.dtype != np.uint8:
        raise ValueError(
            f"Expected a 200x320 RGB byte frame, got {frame.shape} {frame.dtype}"
        )
    if palette == "rgb332":
        return (
            (frame[:, :, 0] & 0xE0)
            | ((frame[:, :, 1] & 0xE0) >> 3)
            | (frame[:, :, 2] >> 6)
        ).tobytes()
    if palette != "adaptive":
        raise ValueError("palette must be adaptive or rgb332")
    image = Image.fromarray(frame).quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    entries = np.asarray(image.getpalette(), dtype=np.uint16).reshape(-1, 3)
    colors = np.zeros((256, 3), dtype=np.uint16)
    colors[: len(entries)] = entries
    # Use a deterministic brightness order before matching to a displayed palette.
    order = np.argsort(colors @ [299, 587, 114], kind="stable")
    indices = np.empty(256, dtype=np.uint8)
    indices[order] = np.arange(256, dtype=np.uint8)
    dac = ((colors[order] * 63 + 127) // 255).astype(np.uint8)
    return indices[np.asarray(image)].tobytes() + dac.tobytes()


def stabilize_vga_palette(pixels: bytes, previous: bytes | None) -> bytes:
    """Match adaptive colors to the preceding displayed frame's palette slots.

    Args:
        pixels: Encoded VGA indices followed by an optional six-bit palette.
        previous: Last delivered frame, or ``None`` before the first frame.

    Returns:
        The same quantized image with remapped indices and palette entries.
        Fixed RGB332 frames and the first adaptive frame pass through unchanged.
    """
    size = FRAME_SIZE + PALETTE_SIZE
    if len(pixels) != size or previous is None or len(previous) != size:
        return pixels
    if pixels[FRAME_SIZE:] == previous[FRAME_SIZE:]:
        return pixels
    colors = np.frombuffer(pixels[FRAME_SIZE:], dtype=np.uint8).reshape(256, 3)
    old = np.frombuffer(previous[FRAME_SIZE:], dtype=np.uint8).reshape(256, 3)
    delta = old.astype(np.int32)[:, None, :] - colors.astype(np.int32)[None, :, :]
    distance = (delta * delta).sum(axis=2, dtype=np.int32)
    mapping = np.empty(256, dtype=np.uint8)
    palette = np.empty_like(colors)
    unavailable = np.iinfo(np.int32).max
    # Exact colors pair first, including rare UI colors, before nearby shades.
    # ponytail: Greedy pairing can sacrifice later matches; use global assignment if needed.
    for _ in range(256):
        target, source = divmod(int(distance.argmin()), 256)
        mapping[source] = target
        palette[target] = colors[source]
        distance[target, :] = unavailable
        distance[:, source] = unavailable
    indices = np.frombuffer(pixels[:FRAME_SIZE], dtype=np.uint8)
    return mapping[indices].tobytes() + palette.tobytes()


class DosOutput:
    """Present frames and exchange input with a DOS VGA client."""

    def __init__(
        self,
        *,
        visible: bool = False,
        max_frames: int | None = None,
        contrast: float = 1.0,
        palette: str = "adaptive",
        direct_vga: bool = False,
        packet_window: int = 4,
        interface: str | None = None,
        dosbox_path: Path | None = None,
        client_dir: Path | None = None,
        dosbox_config: Path | None = None,
    ) -> None:
        """Configure a physical link or DOSBox-X without opening sockets yet.

        Args:
            visible: Show DOSBox-X on the current desktop.
            max_frames: Optional number of completed frames before shutdown.
            contrast: RGB contrast factor before quantization; 1.0 is unchanged.
            palette: Per-frame ``adaptive`` colors or the fixed ``rgb332`` cube.
            direct_vga: Draw RGB332 chunks on arrival, permitting partial images.
            packet_window: Maximum chunks awaiting ACKs, bounded by client capacity.
            interface: Dedicated Linux Ethernet interface for a physical client.
            dosbox_path: DOSBox-X executable with the Unix Ethernet backend.
            client_dir: Writable directory containing ``DOSRX.EXE``.
            dosbox_config: Optional emulator settings, including CPU selection.
        """
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        if not math.isfinite(contrast) or contrast < 0:
            raise ValueError("contrast must be finite and nonnegative")
        if palette not in ("adaptive", "rgb332"):
            raise ValueError("palette must be adaptive or rgb332")
        if direct_vga and palette != "rgb332":
            raise ValueError("direct_vga requires palette=rgb332")
        if not 1 <= packet_window <= 4:
            raise ValueError("packet_window must be between 1 and 4")
        self.visible = visible
        self.contrast = contrast
        self.palette = palette
        self.direct_vga = direct_vga
        self.packet_window = packet_window
        self.receiver_window = 1
        self.dosbox_path = dosbox_path.resolve() if dosbox_path is not None else None
        self.client_dir = client_dir.resolve() if client_dir is not None else None
        self.dosbox_config = (
            dosbox_config.resolve() if dosbox_config is not None else None
        )
        self.socket_directory: tempfile.TemporaryDirectory[str] | None = None
        self.host_socket: Path | None = None
        self.guest_socket: Path | None = None
        self.interface = interface
        self.max_frames = max_frames
        self.seen = 0
        self.sent = 0
        self.dropped = 0
        self.retries = 0
        self.last_pixels: bytes | None = None
        self.inflight_pixels: bytes | None = None
        self.acknowledged: set[tuple[int, int]] = set()
        self.condition = threading.Condition()
        self.pending_pixels: bytes | None = None
        self.sender_thread: threading.Thread | None = None
        self.stop_sender = False
        self.sender_error: Exception | None = None
        self.sock: EthernetLink | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file = None
        self.closed = False
        self.last_control_at = 0.0
        self.control_sequence: int | None = None
        self.pressed_keys: set[str] = set()
        self.pending_events: list[dict[str, Any]] = []
        self.control_packets = 0
        self.input_transitions = 0
        self.quit_requested = False

    def get_input_messages(self) -> list[dict[str, Any]]:
        """Drain DOS keyboard changes and report a close or optional frame limit."""
        with self.condition:
            if self.pressed_keys and time.monotonic() - self.last_control_at > 0.5:
                self._set_pressed_keys(set())
            events = self.pending_events
            self.pending_events = []
            should_close = (
                self.quit_requested
                or self.sender_error is not None
                or (self.max_frames is not None and self.sent >= self.max_frames)
            )
        if should_close or (
            self.process is not None and self.process.poll() is not None
        ):
            events.append({"type": "close"})
        return events

    def _on_packet(self, packet: bytes) -> None:
        host_mac = self.sock.host_mac if self.sock is not None else HOST_MAC
        guest_mac = self.sock.guest_mac if self.sock is not None else GUEST_MAC
        if (
            len(packet) < 20 + KEY_BYTES
            or packet[:6] not in (host_mac, BROADCAST_MAC)
            or packet[6:12] == host_mac
            or packet[6] & 1
            or packet[6:12] == bytes(6)
            or (guest_mac is not None and packet[6:12] != guest_mac)
            or packet[12:14] != ETHERTYPE
            or packet[14:18] != b"FD\x01\x06"
        ):
            return
        if self.sock is not None and guest_mac is None:
            self.sock.guest_mac = packet[6:12]
            print(f"DOS receiver: {packet[6:12].hex(':')}", flush=True)
        sequence = int.from_bytes(packet[18:20], "little")
        with self.condition:
            if self.control_sequence is not None:
                distance = (sequence - self.control_sequence) & 0xFFFF
                if not 0 < distance < 0x8000:
                    return
            capacity = packet[52] if len(packet) > 52 else 0
            self.receiver_window = capacity if 1 <= capacity <= 4 else 1
            if self.direct_vga and not (len(packet) > 53 and packet[53] & 1):
                raise RuntimeError(
                    "DOS client lacks direct VGA support; rebuild DOSRX.EXE"
                )
            self.control_sequence = sequence
            self.control_packets += 1
            self.last_control_at = time.monotonic()
            self._set_pressed_keys(pressed_keys(packet[20 : 20 + KEY_BYTES]))

    def _set_pressed_keys(self, keys: set[str]) -> None:
        for pressed, changed in (
            (False, self.pressed_keys - keys),
            (True, keys - self.pressed_keys),
        ):
            for key in sorted(changed):
                self.pending_events.append(
                    {"type": "keyboard", "key": key, "pressed": pressed}
                )
                self.input_transitions += 1
        self.pressed_keys = keys

    def write(self, frame: np.ndarray) -> None:
        """Resize an RGB image and queue all 320x200 pixels for DOS display."""
        with self.condition:
            if (
                self.quit_requested
                or self.sender_error is not None
                or (self.max_frames is not None and self.sent >= self.max_frames)
            ):
                return
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected an RGB byte image")
        self.seen += 1
        if self.sock is None:
            if self.interface is None:
                self._start_emulator()
            else:
                self._start_physical()
            self.sender_thread = threading.Thread(
                target=self._send_frames, name="dos-frame-sender", daemon=True
            )
            self.sender_thread.start()
        if frame.shape[:2] != (200, 320):
            frame = np.asarray(
                Image.fromarray(frame).resize((320, 200), Image.Resampling.BILINEAR)
            )
        if self.contrast != 1.0:
            frame = np.asarray(
                ImageEnhance.Contrast(Image.fromarray(frame)).enhance(self.contrast)
            )
        pixels = rgb24_to_vga(frame, palette=self.palette)
        with self.condition:
            if self.quit_requested or self.sender_error is not None or self.stop_sender:
                return
            if self.pending_pixels is not None:
                self.dropped += 1
            self.pending_pixels = pixels
            self.condition.notify_all()

    def _send_frames(self) -> None:
        assert self.sock is not None
        try:
            while True:
                with self.condition:
                    if self.pending_pixels is None and not self.stop_sender:
                        self.condition.wait(timeout=0.05)
                    if self.pending_pixels is None and self.stop_sender:
                        return
                    pixels = self.pending_pixels
                    self.pending_pixels = None
                    sequence = self.sent & 0xFFFF
                if pixels is None:
                    read_acks(self.sock, self.acknowledged, self._on_packet)
                    continue
                # Match at delivery time so replaced pending frames cannot shift slots.
                pixels = stabilize_vga_palette(pixels, self.last_pixels)
                with self.condition:
                    self.inflight_pixels = pixels
                self.acknowledged.clear()
                retries = send_frame(
                    self.sock,
                    sequence,
                    pixels,
                    self.acknowledged,
                    self._on_packet,
                    max_attempts=None,
                    check_active=self._check_sender,
                    on_retry=self._report_retry,
                    window=min(self.packet_window, self.receiver_window),
                    direct_vga=self.direct_vga,
                )
                with self.condition:
                    self.retries += retries
                    self.last_pixels = pixels
                    self.inflight_pixels = None
                    self.sent += 1
                    at_limit = (
                        self.max_frames is not None and self.sent >= self.max_frames
                    )
                    if at_limit:
                        self.stop_sender = True
                        self.pending_pixels = None
                    sent = self.sent
                if sent % 10 == 0:
                    suffix = (
                        f"/{self.max_frames}" if self.max_frames is not None else ""
                    )
                    print(f"DOS frames: {sent}{suffix}", flush=True)
                if at_limit:
                    return
        except InterruptedError:
            return
        except Exception as error:
            with self.condition:
                self.sender_error = error
                self.stop_sender = True
                self.pending_pixels = None
                self.condition.notify_all()

    def _check_sender(self) -> None:
        if self.process is not None:
            returncode = self.process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"DOSBox-X exited with status {returncode} while sending a frame"
                )
        with self.condition:
            if self.stop_sender:
                raise InterruptedError("DOS frame sender stopped")

    def _report_retry(self, sequence: int, chunk: int, attempt: int) -> None:
        if attempt == 20 or (attempt > 20 and (attempt - 20) % 30 == 0):
            with self.condition:
                heartbeat_age = time.monotonic() - self.last_control_at
            print(
                f"Waiting for DOS ACK: frame {sequence}, chunk {chunk}, "
                f"{attempt} retries; last DOS keyboard packet "
                f"{heartbeat_age:.1f}s ago",
                flush=True,
            )

    def request_stop(self) -> None:
        """Cancel startup or pending Ethernet retries from another thread."""
        with self.condition:
            self.stop_sender = True
            self.pending_pixels = None
            self.condition.notify_all()

    def close(self) -> None:
        """Stop delivery, close DOS cleanly, and verify emulator output bytes."""
        if self.closed:
            return
        self.closed = True
        self.request_stop()
        if self.sender_thread is not None:
            self.sender_thread.join()
        process = self.process
        sock = self.sock
        if sock is None:
            if self.log_file is not None:
                self.log_file.close()
            if self.socket_directory is not None:
                self.socket_directory.cleanup()
            if self.sender_error is not None:
                raise RuntimeError("DOS frame sender failed") from self.sender_error
            return
        try:
            if self.sender_error is not None:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                raise RuntimeError("DOS frame sender failed") from self.sender_error
            if (
                process is None or process.poll() is None
            ) and sock.guest_mac is not None:
                self.acknowledged.clear()
                send_stop(sock, self.sent & 0xFFFF, self.acknowledged, self._on_packet)
            if process is None:
                print(
                    f"DOS server frames seen={self.seen} sent={self.sent} "
                    f"dropped={self.dropped} retries={self.retries} "
                    f"control packets={self.control_packets} "
                    f"input transitions={self.input_transitions}; "
                    "FRAME.RAW and STATS.TXT are on the DOS machine",
                    flush=True,
                )
                return
            if (
                self.visible
                and self.max_frames is not None
                and self.sent
                and not self.quit_requested
            ):
                print(
                    "DOSBox-X is holding the last FlashDreams frame. "
                    "Focus its window and press any key to close it.",
                    flush=True,
                )
            process.wait()
            assert self.client_dir is not None
            stats_path = self.client_dir / "STATS.TXT"
            if not stats_path.exists():
                if self.sent == 0 and self.inflight_pixels is None:
                    print("DOS server closed before displaying a frame", flush=True)
                    return
                raise RuntimeError("DOS client did not write STATS.TXT")
            stats = dict(item.split("=") for item in stats_path.read_text().split())
            received = int(stats["frames"])
            # Shutdown can interrupt a retry after DOS displayed the frame
            # but its final ACK was lost. The saved VGA bytes settle that case.
            if received == self.sent + 1 and self.inflight_pixels is not None:
                screen = (self.client_dir / "FRAME.RAW").read_bytes()
                if (
                    int(stats["last_sequence"]) == (self.sent & 0xFFFF)
                    and screen == self.inflight_pixels[:FRAME_SIZE]
                ):
                    self.last_pixels = self.inflight_pixels
                    self.sent = received
            if received != self.sent:
                raise RuntimeError(
                    f"DOS received {stats['frames']} of {self.sent} sent frames"
                )
            if (
                self.direct_vga
                and self.inflight_pixels is not None
                and int(stats["last_sequence"]) == (self.sent & 0xFFFF)
            ):
                # Direct drawing may expose part of a frame cancelled at shutdown.
                screen = bytearray((self.last_pixels or bytes(FRAME_SIZE))[:FRAME_SIZE])
                mask = bytes.fromhex(stats["chunk_mask"])
                for chunk in range((FRAME_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE):
                    if mask[chunk >> 3] & (1 << (chunk & 7)):
                        offset = chunk * CHUNK_SIZE
                        screen[offset : offset + CHUNK_SIZE] = self.inflight_pixels[
                            offset : offset + CHUNK_SIZE
                        ]
                self.last_pixels = bytes(screen)
                print("Verifying partial direct VGA frame at shutdown", flush=True)
            if self.last_pixels is not None:
                verified = self._verify_screen(self.last_pixels, stats)
            else:
                verified = 0
            print(
                f"DOS server frames seen={self.seen} sent={self.sent} "
                f"dropped={self.dropped} "
                f"retries={self.retries} control packets={self.control_packets} "
                f"input transitions={self.input_transitions} "
                f"world bytes verified={FRAME_SIZE if self.sent else 0} "
                f"VGA bytes verified={verified}",
                flush=True,
            )
        finally:
            sock.close()
            if self.log_file is not None:
                self.log_file.close()
            if self.socket_directory is not None:
                self.socket_directory.cleanup()

    def _start_physical(self) -> None:
        assert self.interface is not None
        self.sock = physical_link(self.interface)
        print(
            f"Waiting on {self.interface}; start DOSRX.EXE /continuous "
            "on the DOS machine",
            flush=True,
        )
        self._wait_for_receiver()

    def _start_emulator(self) -> None:
        if self.dosbox_path is None or self.client_dir is None:
            raise ValueError(
                "DOSBox mode requires a DOSBox-X executable and DOS client directory"
            )
        for name in (
            "FRAME.RAW",
            "PALETTE.RAW",
            "WORLD.RAW",
            "STATS.TXT",
            "DRIVER.TXT",
            "CLIENT.TXT",
        ):
            (self.client_dir / name).unlink(missing_ok=True)
        self.socket_directory = tempfile.TemporaryDirectory(prefix="flashdreams-dos-")
        directory = Path(self.socket_directory.name)
        self.host_socket = directory / "host.sock"
        self.guest_socket = directory / "guest.sock"
        config = directory / "ethernet.conf"
        config.write_text(
            "[dosbox]\nquit warning=false\n"
            + ("[cpu]\ncore=normal\n" if self.dosbox_config is None else "")
            + "[ne2000]\nne2000=true\nnicbase=300\nnicirq=10\n"
            "macaddr=ac:de:48:88:99:ab\nbackend=unix\n"
        )
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        # Model initialization closes transient low-numbered descriptors.
        # Keep this descriptor below select()'s limit and outside that range.
        fd = fcntl.fcntl(sock.fileno(), fcntl.F_DUPFD_CLOEXEC, 512)
        sock.close()
        sock = socket.socket(fileno=fd)
        sock.bind(str(self.host_socket))
        sock.setblocking(False)
        self.sock = EthernetLink(
            sock, destination=str(self.guest_socket), guest_mac=None
        )
        env = os.environ.copy()
        env.update(
            XDG_CONFIG_HOME=str(directory / "config"),
            DOSBOX_X_ETHERNET_LOCAL=str(self.guest_socket),
            DOSBOX_X_ETHERNET_PEER=str(self.host_socket),
        )
        env["SDL_VIDEODRIVER"] = (
            ("x11" if env.get("DISPLAY") else "wayland") if self.visible else "dummy"
        )
        guest_command = "dosrx.exe /continuous > client.txt"
        if self.max_frames is not None:
            guest_command = (
                "dosrx.exe /hold > client.txt"
                if self.visible
                else "dosrx.exe > client.txt"
            )
        command = [str(self.dosbox_path)]
        if not self.visible:
            command.append("-silent")
        command.extend(["-fastlaunch", "-nopromptfolder"])
        if self.dosbox_config is not None:
            command.extend(["-conf", str(self.dosbox_config)])
        command.extend(
            [
                "-conf",
                str(config),
                "-c",
                "mount c .",
                "-c",
                "c:",
                "-c",
                r"z:\system\ne2000.com 0x60 10 0x300 > driver.txt",
                "-c",
                guest_command,
                "-c",
                "exit",
            ]
        )
        self.log_file = (self.client_dir / "flashdreams-dosbox.log").open("w")
        self.process = subprocess.Popen(
            command,
            cwd=self.client_dir,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            self._wait_for_receiver()
        except BaseException:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()
            raise

    def _verify_screen(self, pixels: bytes, stats: dict[str, str]) -> int:
        """Verify received VGA indices and the palette read from the DAC."""
        assert self.client_dir is not None
        screen = (self.client_dir / "FRAME.RAW").read_bytes()
        if len(screen) != FRAME_SIZE or screen != pixels[:FRAME_SIZE]:
            raise RuntimeError("DOS VGA framebuffer differs from the Linux frame sent")
        if len(pixels) == FRAME_SIZE + PALETTE_SIZE:
            palette = (self.client_dir / "PALETTE.RAW").read_bytes()
            if palette != pixels[FRAME_SIZE:]:
                raise RuntimeError(
                    "DOS VGA palette differs from the Linux palette sent"
                )
            print(f"VGA palette bytes verified={PALETTE_SIZE}", flush=True)
        verified = FRAME_SIZE
        return verified

    def _wait_for_receiver(self) -> None:
        assert self.sock is not None
        deadline = time.monotonic() + 10.0 if self.process is not None else None
        while not self.control_packets:
            if self.stop_sender:
                raise InterruptedError("DOS receiver startup cancelled")
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("DOSBox-X exited before the DOS receiver was ready")
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(
                    "DOS receiver did not send its ready keyboard packet"
                )
            read_acks(self.sock, self.acknowledged, self._on_packet)
            if not self.control_packets:
                select.select([self.sock], [], [], 0.05)
