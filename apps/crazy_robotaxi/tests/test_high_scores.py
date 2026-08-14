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

from __future__ import annotations

from pathlib import Path

import pytest
from crazy_robotaxi.high_scores import (
    HighScoreStore,
    default_high_scores_path,
    validate_player_name,
)

pytestmark = pytest.mark.ci_cpu


def test_default_path_uses_game_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    assert default_high_scores_path() == tmp_path / "crazy-robotaxi" / "highscores.csv"


def test_store_sorts_limits_and_uses_earlier_timestamp_for_ties(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv", limit=3)
    store.record("LATE", 100, achieved_at_utc="2026-01-02T00:00:00+00:00")
    store.record("TOP", 200, achieved_at_utc="2026-01-03T00:00:00+00:00")
    store.record("EARLY", 100, achieved_at_utc="2026-01-01T00:00:00+00:00")
    inserted, board = store.record(
        "LOW", 50, achieved_at_utc="2026-01-04T00:00:00+00:00"
    )
    assert inserted is None
    assert [entry.name for entry in board] == ["TOP", "EARLY", "LATE"]
    assert store.qualifying_rank(100) is None
    assert store.qualifying_rank(101) == 2


def test_store_tolerates_malformed_rows_and_csv_escaping(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "name,score,achieved_at_utc\nBAD,nope,never\nVALID,42,2026-01-01T00:00:00+00:00\n",
        encoding="utf-8",
    )
    store = HighScoreStore(path)
    store.record("A B", 99, achieved_at_utc="2026-02-01T00:00:00+00:00")
    assert [entry.name for entry in store.read()] == ["A B", "VALID"]


@pytest.mark.parametrize("name", ["", "thirteen_chars", "bad,name", "emoji🚕"])
def test_player_name_validation_rejects_unsupported_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_player_name(name)
