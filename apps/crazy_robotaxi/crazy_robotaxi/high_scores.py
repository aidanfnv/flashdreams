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

"""Persistent Crazy Robotaxi high scores."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

_CSV_FIELDS = ("name", "score", "achieved_at_utc")
_PLAYER_NAME_RE = re.compile(r"[A-Za-z0-9 _-]{1,12}")


def default_high_scores_path() -> Path:
    """Return the game-owned leaderboard path outside Interactive Drive."""
    configured = os.environ.get("FLASHDREAMS_CACHE_DIR")
    cache_root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "flashdreams"
    )
    return cache_root / "crazy-robotaxi" / "highscores.csv"


def validate_player_name(name: str) -> str:
    """Normalize a player name or raise :class:`ValueError`."""
    normalized = name.strip()
    if _PLAYER_NAME_RE.fullmatch(normalized) is None:
        raise ValueError(
            "Name must be 1-12 characters using letters, numbers, spaces, "
            "hyphens, or underscores."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class HighScoreEntry:
    """One leaderboard result."""

    name: str
    score: int
    achieved_at_utc: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "score": self.score,
            "achieved_at_utc": self.achieved_at_utc,
        }


class HighScoreStore:
    """Read and atomically update a process-safe top-ten CSV leaderboard."""

    def __init__(self, path: Path | None = None, *, limit: int = 10) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._path = Path(path) if path is not None else default_high_scores_path()
        self._limit = limit
        self._lock_path = self._path.with_suffix(f"{self._path.suffix}.lock")

    @property
    def path(self) -> Path:
        """Return the CSV path."""
        return self._path

    def read(self) -> tuple[HighScoreEntry, ...]:
        """Return sorted valid entries while ignoring malformed rows."""
        if not self._path.exists():
            return ()
        with FileLock(self._lock_path):
            return self._read_unlocked()

    def qualifying_rank(self, score: int) -> int | None:
        """Return the prospective rank, excluding non-positive and tied cutoff scores."""
        if score <= 0:
            return None
        entries = self.read()
        if len(entries) >= self._limit and score <= entries[-1].score:
            return None
        return 1 + sum(entry.score >= score for entry in entries)

    def record(
        self,
        name: str,
        score: int,
        *,
        achieved_at_utc: str | None = None,
    ) -> tuple[HighScoreEntry | None, tuple[HighScoreEntry, ...]]:
        """Persist a qualifying result and return it with the resulting board."""
        normalized_name = validate_player_name(name)
        if score <= 0:
            return None, self.read()
        timestamp = achieved_at_utc or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        datetime.fromisoformat(timestamp)
        entry = HighScoreEntry(normalized_name, int(score), timestamp)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._lock_path):
            entries = list(self._read_unlocked())
            inserted: HighScoreEntry | None = entry
            if len(entries) >= self._limit and score <= entries[-1].score:
                inserted = None
            else:
                entries.append(entry)
            board = self._sort(entries)
            self._write_unlocked(board)
        return inserted, board

    def _read_unlocked(self) -> tuple[HighScoreEntry, ...]:
        if not self._path.exists():
            return ()
        entries: list[HighScoreEntry] = []
        try:
            with self._path.open(newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    try:
                        name = validate_player_name(row.get("name", ""))
                        score = int(row.get("score", ""))
                        achieved_at = row.get("achieved_at_utc", "")
                        datetime.fromisoformat(achieved_at)
                    except (TypeError, ValueError):
                        continue
                    if score > 0:
                        entries.append(HighScoreEntry(name, score, achieved_at))
        except (OSError, csv.Error):
            return ()
        return self._sort(entries)

    def _sort(self, entries: list[HighScoreEntry]) -> tuple[HighScoreEntry, ...]:
        return tuple(
            sorted(entries, key=lambda entry: (-entry.score, entry.achieved_at_utc))[
                : self._limit
            ]
        )

    def _write_unlocked(self, entries: tuple[HighScoreEntry, ...]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as csv_file:
                temporary_path = Path(csv_file.name)
                writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(entry.as_dict() for entry in entries)
                csv_file.flush()
                os.fsync(csv_file.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
