from datetime import date

import pandas as pd
import pytest

from lottery.cli.recommend import main
from lottery.common.types import DrawRecord
from lottery.data.storage import COLUMNS, DrawStorage


@pytest.fixture
def seeded_csv(tmp_path, monkeypatch):
    path = tmp_path / "ssq_history.csv"
    records = [
        DrawRecord(f"2026{i:03d}", date(2026, 1, min(i, 28)), (1, 2, 3, 4, 5, 6), i % 16 + 1)
        for i in range(1, 21)
    ]
    DrawStorage(path).rewrite(_to_frame(records))
    return path


def _to_frame(records):
    rows = [
        {
            "issue": r.issue,
            "draw_date": str(r.draw_date),
            **{f"red{j + 1}": v for j, v in enumerate(r.reds)},
            "blue": r.blue,
        }
        for r in records
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_offline_recommend_prints_banner_and_tickets(seeded_csv, capsys, monkeypatch):
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: seeded_csv)
    code = main(["--offline", "--count", "3", "--window", "20"])
    out = capsys.readouterr().out
    assert code == 0
    assert "风险提示" in out and "本次推荐 3 注" in out
    assert out.count("红: ") == 3


def test_empty_csv_fails(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: missing)
    assert main(["--offline"]) == 1
    assert "没有可用历史数据" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [["--count", "0"], ["--pool-size", "5"], ["--pool-size", "34"], ["--window", "0"]],
)
def test_invalid_args_rejected(argv, capsys):
    assert main(argv) == 2


def test_refresh_and_offline_are_mutually_exclusive(capsys):
    assert main(["--refresh", "--offline"]) == 2
