from datetime import date

import pandas as pd
import pytest

from lottery.cli.recommend import main
from lottery.common.types import DrawRecord
from lottery.data.cleaner import CleanResult
from lottery.data.crawler import CrawlError
from lottery.data.repository import DrawRepository
from lottery.data.storage import COLUMNS, DrawStorage


def _records():
    return [
        DrawRecord(f"2026{i:03d}", date(2026, 1, min(i, 28)), (1, 2, 3, 4, 5, 6), i % 16 + 1)
        for i in range(1, 21)
    ]


@pytest.fixture
def seeded_csv(tmp_path, monkeypatch):
    path = tmp_path / "ssq_history.csv"
    DrawStorage(path).rewrite(_to_frame(_records()))
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


def _fake_refresh(self):
    DrawStorage(self.storage.path).rewrite(_to_frame(_records()))
    return CleanResult(0, 0, [], True, self.storage.load())


def test_offline_recommend_prints_banner_and_tickets(seeded_csv, capsys, monkeypatch):
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: seeded_csv)
    code = main(["--offline", "--count", "3", "--window", "20"])
    out = capsys.readouterr().out
    assert code == 0
    assert "风险提示" in out and "本次推荐 3 注" in out
    assert out.count("红: ") == 3


def test_default_run_refreshes_when_csv_missing(monkeypatch, tmp_path, capsys):
    path = tmp_path / "ssq_history.csv"
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: path)
    monkeypatch.setattr(DrawRepository, "refresh", _fake_refresh)
    code = main(["--count", "3"])
    assert code == 0
    assert path.exists()
    assert "本次推荐 3 注" in capsys.readouterr().out


def test_zero_byte_csv_triggers_refresh(monkeypatch, tmp_path, capsys):
    path = tmp_path / "ssq_history.csv"
    path.write_bytes(b"")
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: path)
    monkeypatch.setattr(DrawRepository, "refresh", _fake_refresh)
    assert main(["--count", "3"]) == 0
    assert "本次推荐 3 注" in capsys.readouterr().out


def test_zero_byte_csv_offline_fails(monkeypatch, tmp_path, capsys):
    path = tmp_path / "ssq_history.csv"
    path.write_bytes(b"")
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: path)

    def fail_refresh(self):
        raise AssertionError("--offline 不应触发刷新")

    monkeypatch.setattr(DrawRepository, "refresh", fail_refresh)
    assert main(["--offline"]) == 1
    err = capsys.readouterr().err
    assert "没有可用历史数据" in err
    assert "Traceback" not in err


def test_refresh_failure_reports_clean_error(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: missing)

    def boom(self):
        raise CrawlError("网络不可用")

    monkeypatch.setattr(DrawRepository, "refresh", boom)
    assert main([]) == 1
    err = capsys.readouterr().err
    assert "数据加载失败" in err
    assert "Traceback" not in err


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
