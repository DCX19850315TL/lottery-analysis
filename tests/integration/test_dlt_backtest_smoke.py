"""大乐透回测 CLI 冒烟测试：离线读取本地 CSV 并输出命中指标。"""

import pandas as pd
import pytest

from lottery.cli.dlt_backtest import main
from lottery.data.sporttery import SPORTS_COLUMNS, SportsStorage


def _numbers(index: int) -> str:
    fronts = sorted({((index + step * 3) % 35) + 1 for step in range(5)})
    backs = sorted({((index * 2 + step) % 12) + 1 for step in range(2)})
    return " ".join(str(number) for number in [*fronts, *backs])


def _frame(count: int = 20) -> pd.DataFrame:
    rows = [
        {
            "issue": f"26{index:03d}",
            "draw_date": f"2026-01-{min(index, 28):02d}",
            "numbers": _numbers(index),
        }
        for index in range(1, count + 1)
    ]
    return pd.DataFrame(rows, columns=SPORTS_COLUMNS)


@pytest.fixture
def seeded_csv(tmp_path, monkeypatch):
    path = tmp_path / "dlt_history.csv"
    SportsStorage(path).rewrite(_frame())
    monkeypatch.setattr("lottery.cli.dlt_data.storage_path", lambda _cfg: path)
    return path


_BASE = ["--offline", "--window", "10", "--test-periods", "5", "--count", "3"]


def test_offline_backtest_prints_metrics(seeded_csv, capsys):
    assert main(_BASE) == 0
    out = capsys.readouterr().out
    assert "风险提示" in out
    assert "dlt:teaching" in out
    assert "测试期数: 5" in out
    assert "前区单注期望命中" in out
    assert "总投入: 30.00" in out


def test_insufficient_history_reports_clean_error(seeded_csv, capsys):
    code = main(["--offline", "--window", "15", "--test-periods", "10"])
    err = capsys.readouterr().err
    assert code == 1
    assert "历史数据不足" in err
    assert "Traceback" not in err


def test_missing_csv_offline_fails(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr("lottery.cli.dlt_data.storage_path", lambda _cfg: missing)
    assert main(["--offline"]) == 1
    assert "没有可用历史数据" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--test-periods", "0"],
        ["--window", "0"],
        ["--count", "0"],
        ["--front-pool", "4"],
        ["--back-pool", "1"],
        ["--ticket-price", "0"],
        ["--count", "2", "--front-pool", "5", "--back-pool", "2"],
        ["--refresh", "--offline"],
    ],
)
def test_invalid_args_rejected(argv, capsys):
    assert main(argv) == 2
