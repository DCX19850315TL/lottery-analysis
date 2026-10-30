import numpy as np
import pandas as pd
import pytest

from lottery.backtest.metrics import BacktestResult, Metrics
from lottery.backtest.significance import SignificanceTester


def _baseline(red_hits: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial": np.zeros(len(red_hits), dtype=int),
            "period": np.arange(len(red_hits)),
            "red_hits": red_hits,
            "payout": np.zeros(len(red_hits)),
        }
    )


def _result(red_hits) -> BacktestResult:
    metrics = Metrics(
        mean_red_hits=float(np.mean(red_hits)) if len(red_hits) else 0.0,
        mean_blue_hits=0.0,
        hit_rate=0.0,
        expected_hits=1.0,
        max_gap_periods=0,
        max_drawdown=0.0,
        total_cost=0.0,
        total_return=0.0,
    )
    return BacktestResult(
        metrics=metrics,
        curve=pd.DataFrame({"red_hits": list(red_hits)}),
        red_hit_counts=tuple(int(value) for value in red_hits),
        label="test",
    )


def test_identical_distribution_is_not_significant():
    rng = np.random.default_rng(11)
    baseline = _baseline(rng.integers(0, 5, size=20_000))
    observed = rng.integers(0, 5, size=2_000)
    outcome = SignificanceTester().compare(_result(observed), baseline)
    assert not outcome.is_significant
    assert outcome.p_value > 0.05
    assert outcome.conclusion.startswith("策略与随机选号无统计显著差异")


def test_shifted_distribution_is_significant():
    baseline = _baseline(np.zeros(2_000, dtype=int))
    outcome = SignificanceTester().compare(_result([6] * 200), baseline)
    assert outcome.is_significant
    assert outcome.p_value < 0.05
    assert "存在统计显著差异" in outcome.conclusion


def test_empty_strategy_rejected():
    baseline = _baseline(np.zeros(100, dtype=int))
    with pytest.raises(ValueError):
        SignificanceTester().compare(_result([]), baseline)
