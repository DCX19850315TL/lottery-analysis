import pytest

from lottery.backtest.math_engine import MathBacktestEngine
from lottery.backtest.metrics import ScoreRules
from lottery.backtest.monte_carlo import MonteCarloBaseline
from lottery.backtest.teaching_engine import TeachingBacktestEngine
from lottery.common.config import get_config
from lottery.common.types import Ticket
from lottery.theory.coverage import CoveragePlan, Guarantee
from lottery.theory.prize_table import PrizeLevel

PAYOUTS = {
    PrizeLevel.FIRST: 5_000_000.0,
    PrizeLevel.SECOND: 100_000.0,
    PrizeLevel.THIRD: 3_000.0,
    PrizeLevel.FOURTH: 200.0,
    PrizeLevel.FIFTH: 10.0,
    PrizeLevel.SIXTH: 5.0,
}


def _plan() -> CoveragePlan:
    return CoveragePlan(
        tickets=(Ticket(reds=(1, 2, 3, 4, 5, 6), blue=1),),
        guarantee=Guarantee.MATCH6_COVER5,
        proven_optimal=True,
        lower_bound=1,
        cost=1.0,
    )


def test_math_engine_scores_fixed_plan(make_draws):
    draws = make_draws(5, seed=3, chooser=lambda rng, index: (1, 2, 3, 4, 5, 6))
    order = list(range(5))
    reordered = [draws[i] for i in order]
    engine = MathBacktestEngine(ScoreRules(ticket_price=2.0, payouts=PAYOUTS))
    result = engine.run(_plan(), reordered, test_periods=3)
    assert result.metrics.total_cost == pytest.approx(6.0)
    assert result.metrics.total_return == pytest.approx(15_000_000.0)
    assert result.metrics.mean_red_hits == pytest.approx(6.0)
    assert list(result.curve["issue"]) == [draw.issue for draw in reordered[-3:]]


def test_math_engine_rejects_too_few_draws(make_draws):
    engine = MathBacktestEngine(ScoreRules(ticket_price=2.0, payouts=PAYOUTS))
    with pytest.raises(ValueError):
        engine.run(_plan(), make_draws(2, seed=4), test_periods=3)


def test_teaching_engine_uses_prior_draws_only(make_draws):
    cfg = get_config()
    draws = make_draws(40, seed=5)
    engine = TeachingBacktestEngine(cfg, ScoreRules(ticket_price=2.0, payouts=PAYOUTS))
    first = engine.run(draws, model_name="rule", window=10, test_periods=3)
    changed = list(draws)
    changed[-1] = make_draws(1, seed=99, chooser=lambda rng, index: (10, 20, 30, 31, 32, 33))[0]
    second = engine.run(changed, model_name="rule", window=10, test_periods=3)
    assert first.curve.iloc[:2].equals(second.curve.iloc[:2])
    assert first.metrics.total_cost == pytest.approx(3 * cfg.recommend.ticket_count * 2.0)


def test_teaching_engine_rejects_unknown_model(make_draws):
    cfg = get_config()
    engine = TeachingBacktestEngine(cfg, ScoreRules(ticket_price=2.0, payouts=PAYOUTS))
    with pytest.raises(ValueError):
        engine.run(make_draws(40, seed=6), model_name="xgb", window=10, test_periods=3)


def test_teaching_engine_rejects_short_history(make_draws):
    cfg = get_config()
    engine = TeachingBacktestEngine(cfg, ScoreRules(ticket_price=2.0, payouts=PAYOUTS))
    with pytest.raises(ValueError):
        engine.run(make_draws(15, seed=7), model_name="rule", window=10, test_periods=10)


def test_monte_carlo_is_reproducible(make_draws):
    draws = make_draws(20, seed=8)
    baseline = MonteCarloBaseline(
        n_tickets=2, rules=ScoreRules(ticket_price=2.0, payouts=PAYOUTS)
    )
    first = baseline.run(draws, n_trials=3, seed=42)
    second = baseline.run(draws, n_trials=3, seed=42)
    assert first.equals(second)
    assert len(first) == 3 * 20
    assert set(first.columns) >= {"trial", "period", "red_hits", "payout"}
    assert first["trial"].nunique() == 3
