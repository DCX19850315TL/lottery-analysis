"""大乐透推荐 CLI 冒烟测试：离线读取本地 CSV 并打印注数与号码。"""

import pandas as pd
import pytest

from lottery.cli.dlt_recommend import main
from lottery.data.crawler import CrawlError
from lottery.data.sporttery import SPORTS_COLUMNS, SportsStorage


def _numbers(index: int) -> str:
    fronts = sorted({((index + step * 3) % 35) + 1 for step in range(5)})
    backs = sorted({((index * 2 + step) % 12) + 1 for step in range(2)})
    return " ".join(str(number) for number in [*fronts, *backs])


def _frame() -> pd.DataFrame:
    rows = [
        {
            "issue": f"26{index:03d}",
            "draw_date": f"2026-01-{min(index, 28):02d}",
            "numbers": _numbers(index),
        }
        for index in range(1, 21)
    ]
    return pd.DataFrame(rows, columns=SPORTS_COLUMNS)


@pytest.fixture
def seeded_csv(tmp_path, monkeypatch):
    path = tmp_path / "dlt_history.csv"
    SportsStorage(path).rewrite(_frame())
    monkeypatch.setattr("lottery.cli.dlt_data.storage_path", lambda _cfg: path)
    return path


def test_offline_recommend_prints_count_and_tickets(seeded_csv, capsys):
    assert main(["--offline", "--count", "3", "--window", "20"]) == 0
    out = capsys.readouterr().out
    assert "风险提示" in out
    assert "本次推荐 3 注" in out
    assert out.count("前区: ") == 3
    assert "后区: " in out


def test_missing_csv_offline_fails(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr("lottery.cli.dlt_data.storage_path", lambda _cfg: missing)
    assert main(["--offline"]) == 1
    assert "没有可用历史数据" in capsys.readouterr().err


def test_refresh_failure_reports_clean_error(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr("lottery.cli.dlt_data.storage_path", lambda _cfg: missing)

    class FailingCrawler:
        def __init__(self, _cfg):
            pass

        def fetch_all(self, _game_no):
            raise CrawlError("接口不可用")

    monkeypatch.setattr("lottery.cli.dlt_data.SportteryCrawler", FailingCrawler)
    assert main([]) == 1
    err = capsys.readouterr().err
    assert "数据加载失败" in err
    assert "接口不可用" in err
    assert "Traceback" not in err


def test_corrupt_csv_reports_single_line_error(monkeypatch, tmp_path, capsys):
    path = tmp_path / "dlt_history.csv"
    path.write_text("issue,draw_date,numbers\n26001,2026-01-01,1 2 3\n", encoding="utf-8")
    monkeypatch.setattr("lottery.cli.dlt_data.storage_path", lambda _cfg: path)
    assert main(["--offline"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("数据加载失败: ")
    assert len(err.strip().splitlines()) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--count", "0"],
        ["--front-pool", "4"],
        ["--front-pool", "36"],
        ["--back-pool", "1"],
        ["--back-pool", "13"],
        ["--window", "0"],
        ["--count", "2", "--front-pool", "5", "--back-pool", "2"],
    ],
)
def test_invalid_args_rejected(argv, capsys):
    assert main(argv) == 2


def test_refresh_and_offline_are_mutually_exclusive(capsys):
    assert main(["--refresh", "--offline"]) == 2
