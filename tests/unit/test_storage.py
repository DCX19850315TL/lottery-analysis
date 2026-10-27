from datetime import date

import pandas as pd
import pytest

from lottery.common.types import DrawRecord
from lottery.data.storage import COLUMNS, DrawStorage


def rec(issue, reds=(1, 2, 3, 4, 5, 6), blue=9):
    return DrawRecord(issue, date(2026, 10, 25), reds, blue)


def test_load_missing_file_returns_empty_frame(tmp_path):
    df = DrawStorage(tmp_path / "x.csv").load()
    assert list(df.columns) == COLUMNS and len(df) == 0


def test_append_is_idempotent(tmp_path):
    s = DrawStorage(tmp_path / "x.csv")
    assert s.append([rec("2026122"), rec("2026121")]) == 2
    assert s.append([rec("2026122")]) == 0
    assert len(s.load()) == 2


def test_append_keeps_row_values(tmp_path):
    s = DrawStorage(tmp_path / "x.csv")
    s.append([rec("2026122", (1, 2, 4, 17, 19, 30), 9)])
    row = s.load().iloc[0]
    assert (row["red1"], row["red6"], row["blue"]) == (1, 30, 9)
    assert row["draw_date"] == "2026-10-25"


def test_rewrite_replaces_atomically(tmp_path):
    s = DrawStorage(tmp_path / "x.csv")
    s.append([rec("2026122")])
    s.rewrite(pd.DataFrame([{**{c: 0 for c in COLUMNS}, "issue": "2026120"}]).astype({"issue": str}))
    assert s.load()["issue"].tolist() == ["2026120"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_rewrite_failure_keeps_old_file(tmp_path, monkeypatch):
    s = DrawStorage(tmp_path / "x.csv")
    s.append([rec("2026122")])
    def boom(*args): raise OSError("boom")
    monkeypatch.setattr("lottery.data.storage.os.replace", boom)
    with pytest.raises(OSError):
        s.rewrite(pd.DataFrame([{c: 0 for c in COLUMNS}]))
    assert s.load()["issue"].tolist() == ["2026122"]
    assert list(tmp_path.glob("*.tmp")) == []
