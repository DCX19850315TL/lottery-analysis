from datetime import date

import pandas as pd
import pytest

from lottery.cli.backtest import main
from lottery.common.types import DrawRecord
from lottery.data.storage import COLUMNS, DrawStorage


def _records(count=260):
    return [
        DrawRecord(
            f"2026{i:03d}", date(2026, 1, min(i, 28)), (1, 2, 3, 4, 5, 6), i % 16 + 1
        )
        for i in range(1, count + 1)
    ]


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


@pytest.fixture
def seeded_csv(tmp_path, monkeypatch):
    path = tmp_path / "ssq_history.csv"
    DrawStorage(path).rewrite(_to_frame(_records()))
    monkeypatch.setattr("lottery.cli.backtest._default_storage_path", lambda: path)
    return path


_BASE = ["--offline", "--test-periods", "20", "--window", "100", "--trials", "5"]


def test_teaching_route_prints_metrics_and_conclusion(seeded_csv, capsys):
    code = main([*_BASE, "--route", "teaching"])
    out = capsys.readouterr().out
    assert code == 0
    assert "风险提示" in out
    assert "teaching:rule" in out
    assert "结论:" in out
    assert "策略与随机选号" in out


def test_math_route_uses_coverage_plan(seeded_csv, capsys):
    code = main(
        [
            *_BASE,
            "--route",
            "math",
            "--pool-size",
            "8",
            "--guarantee",
            "match4_cover3",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "math:match4_cover3" in out


def test_default_route_runs_both_engines(seeded_csv, capsys):
    code = main(_BASE)
    out = capsys.readouterr().out
    assert code == 0
    assert "teaching:rule" in out and "math:" in out


def test_insufficient_history_reports_clean_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "ssq_history.csv"
    DrawStorage(path).rewrite(_to_frame(_records(80)))
    monkeypatch.setattr("lottery.cli.backtest._default_storage_path", lambda: path)
    code = main(["--offline", "--route", "teaching", "--test-periods", "20", "--window", "100"])
    err = capsys.readouterr().err
    assert code == 1
    assert "历史数据不足" in err
    assert "Traceback" not in err


def test_missing_offline_csv_fails(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "none.csv"
    monkeypatch.setattr("lottery.cli.backtest._default_storage_path", lambda: missing)
    assert main(["--offline", "--test-periods", "5", "--trials", "3"]) == 1
    assert "没有可用历史数据" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--test-periods", "0"],
        ["--window", "0"],
        ["--trials", "0"],
        ["--pool-size", "5"],
        ["--refresh", "--offline"],
    ],
)
def test_invalid_args_rejected(argv, capsys):
    assert main(argv) == 2
