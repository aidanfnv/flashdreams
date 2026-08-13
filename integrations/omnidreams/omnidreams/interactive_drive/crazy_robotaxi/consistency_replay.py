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

"""Deterministic A/B replay for Crazy Robotaxi consistency prompts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omnidreams.interactive_drive.types import TextPromptUpdate
from omnidreams.interactive_drive.world_model.flashdreams_adapter import (
    FlashdreamsWorldModelSession,
)
from omnidreams.interactive_drive.world_model.manifest import (
    WorldModelManifest,
    load_world_model_manifest,
)
from PIL import Image

from flashdreams.infra.runner_io import write_video_tensor


def build_parser() -> argparse.ArgumentParser:
    """Build the consistency-replay command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay a --taxi-alignment-diagnostics trace with and without "
            "PR431 physical-consistency prompt edits."
        )
    )
    parser.add_argument("trace", type=Path, help="Timestamped diagnostic run")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the recorded/model rollout seed.",
    )
    return parser


def _load_trace(
    trace_dir: Path,
) -> tuple[dict[str, Any], list[tuple[int, list[np.ndarray], TextPromptUpdate]]]:
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    rows_by_chunk: dict[int, list[dict[str, str]]] = defaultdict(list)
    with (trace_dir / "telemetry.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_chunk[int(row["chunk_index"])].append(row)

    chunks: list[tuple[int, list[np.ndarray], TextPromptUpdate]] = []
    for chunk_index, rows in sorted(rows_by_chunk.items()):
        frames = [
            _read_rgb(
                trace_dir / "conditioning" / f"frame_{int(row['sequence']):06d}.png"
            )
            for row in rows
        ]
        first = rows[0]
        prompt = first["text_prompt"] or str(metadata["original_prompt"])
        chunks.append(
            (
                chunk_index,
                frames,
                TextPromptUpdate(
                    prompt=prompt,
                    guidance_scale=float(first["text_guidance_scale"] or 3.0),
                    guidance_chunks=int(first["text_guidance_chunks"] or 6),
                    recache_last_chunk=(first["text_recache"].lower() != "false"),
                    active_modifiers=tuple(
                        value for value in first["text_modifiers"].split(",") if value
                    ),
                ),
            )
        )
    if not chunks:
        raise ValueError(f"Trace contains no captured chunks: {trace_dir}")
    return metadata, chunks


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _materialize_model_frame(frame: object) -> np.ndarray:
    if hasattr(frame, "to_numpy"):
        frame = frame.to_numpy()
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _run_variant(
    *,
    manifest: WorldModelManifest,
    initial_rgb: np.ndarray,
    original_prompt: str,
    chunks: list[tuple[int, list[np.ndarray], TextPromptUpdate]],
    edits_enabled: bool,
) -> list[np.ndarray]:
    session = FlashdreamsWorldModelSession(
        manifest,
        text_edits_enabled=edits_enabled,
    )
    session.warmup_model()
    output: list[np.ndarray] = []
    try:
        _, first_conditions, first_update = chunks[0]
        first_prompt = first_update.prompt if edits_enabled else original_prompt
        output.extend(
            _materialize_model_frame(frame)
            for frame in session.start(initial_rgb, first_conditions, first_prompt)
        )
        for _, conditions, update in chunks[1:]:
            output.extend(
                _materialize_model_frame(frame)
                for frame in session.continue_generation(
                    conditions,
                    text_prompt_update=update if edits_enabled else None,
                )
            )
    finally:
        session.close()
    return output


def _video_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    array = np.stack(frames).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(0, 3, 1, 2)


def main(argv: list[str] | None = None) -> None:
    """Run baseline and treatment generation from one diagnostic trace."""
    args = build_parser().parse_args(argv)
    trace_dir = args.trace.expanduser().resolve()
    metadata, chunks = _load_trace(trace_dir)
    recorded_seed = metadata.get("model_seed")
    seed = args.seed if args.seed is not None else recorded_seed
    if seed is None:
        raise SystemExit(
            "A deterministic A/B replay requires --seed or a trace captured "
            "with manifest seed_for_every_rollout."
        )

    manifest = replace(
        load_world_model_manifest(args.manifest.expanduser().resolve()),
        seed_for_every_rollout=int(seed),
    )
    initial_rgb = _read_rgb(trace_dir / "initial_rgb.png")
    original_prompt = str(metadata["original_prompt"])
    baseline = _run_variant(
        manifest=manifest,
        initial_rgb=initial_rgb,
        original_prompt=original_prompt,
        chunks=chunks,
        edits_enabled=False,
    )
    treatment = _run_variant(
        manifest=manifest,
        initial_rgb=initial_rgb,
        original_prompt=original_prompt,
        chunks=chunks,
        edits_enabled=True,
    )

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_tensor = _video_tensor(baseline)
    treatment_tensor = _video_tensor(treatment)
    write_video_tensor(
        baseline_tensor, output_dir / "baseline.mp4", fps=30, layout="tchw"
    )
    write_video_tensor(
        treatment_tensor, output_dir / "treatment.mp4", fps=30, layout="tchw"
    )
    write_video_tensor(
        torch.cat((baseline_tensor, treatment_tensor), dim=3),
        output_dir / "side_by_side.mp4",
        fps=30,
        layout="tchw",
    )
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir()
    for index, (baseline_frame, treatment_frame) in enumerate(
        zip(baseline, treatment, strict=True)
    ):
        comparison = Image.new(
            "RGB",
            (
                baseline_frame.shape[1] + treatment_frame.shape[1],
                baseline_frame.shape[0],
            ),
        )
        comparison.paste(Image.fromarray(baseline_frame, mode="RGB"), (0, 0))
        comparison.paste(
            Image.fromarray(treatment_frame, mode="RGB"),
            (baseline_frame.shape[1], 0),
        )
        comparison.save(contact_dir / f"frame_{index:06d}.png", format="PNG")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_trace": str(trace_dir),
                "manifest": str(args.manifest),
                "seed": int(seed),
                "frames": len(baseline),
                "left": "baseline",
                "right": "PR431 consistency prompts",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
