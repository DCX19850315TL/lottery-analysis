"""大乐透滚动回测：只用被预测期之前的数据，逐期评分前区与后区命中。"""

from datetime import date

import pandas as pd
import pytest

from lottery.backtest.dlt_engine import DltBacktestEngine, DltBacktestOptions
from lottery.common.config import get_config
from lottery.data.sporttery import SportsDraw


def _draws(count: int) -> list[SportsDraw]:
    draws = []
    for i in range(count):
        fronts = tuple(sorted((i + offset) % 35 + 1 for offset in range(5)))
        backs = tuple(sorted((i + offset) % 12 + 1 for offset in range(2)))
        draws.append(SportsDraw(f"26{i:03d}", date(2026, 1, 1), fronts + backs))
    return draws


def _options(**overrides):
    values = {
        "count": 3,
        "front_pool_size": 10,
        "back_pool_size": 5,
        "window": 10,
        "test_periods": 3,
        "seed": 42,
        "ticket_price": 2.0,
    }
    values.update(overrides)
    return DltBacktestOptions(**values)


def test_dlt_backtest_scores_periods_with_expected_values():
    result = DltBacktestEngine(get_config()).run(_draws(20), _options())
    assert len(result.curve) == 3
    assert result.metrics.total_cost == pytest.approx(18.0)
    assert result.metrics.expected_front_hits_per_ticket == pytest.approx(5 / 7)
    assert result.metrics.expected_back_hits_per_ticket == pytest.approx(1 / 3)
    assert 0 <= result.metrics.mean_max_front_hits <= 5
    assert 0 <= result.metrics.mean_front_hits_per_ticket <= 5
    assert 0 <= result.metrics.mean_back_hits_per_ticket <= 2


def test_dlt_backtest_is_reproducible():
    draws = _draws(20)
    first = DltBacktestEngine(get_config()).run(draws, _options())
    second = DltBacktestEngine(get_config()).run(draws, _options())
    pd.testing.assert_frame_equal(first.curve, second.curve)
    assert first.metrics == second.metrics


def test_dlt_backtest_uses_prior_draws_only():
    draws = _draws(20)
    first = DltBacktestEngine(get_config()).run(draws, _options())
    changed = list(draws)
    changed[-1] = SportsDraw("26999", date(2026, 1, 2), (1, 2, 3, 4, 5, 1, 2))
    second = DltBacktestEngine(get_config()).run(changed, _options())
    assert first.curve.iloc[:2].equals(second.curve.iloc[:2])


def test_dlt_backtest_rejects_short_history():
    with pytest.raises(ValueError):
        DltBacktestEngine(get_config()).run(_draws(12), _options(test_periods=5))


@pytest.mark.parametrize(
    "overrides",
    [
        {"window": 0},
        {"test_periods": 0},
        {"count": 0},
        {"front_pool_size": 4},
        {"back_pool_size": 1},
        {"ticket_price": 0.0},
        {"count": 2, "front_pool_size": 5, "back_pool_size": 2},
    ],
)
def test_dlt_backtest_options_reject_invalid_values(overrides):
    with pytest.raises(ValueError):
        _options(**overrides)
