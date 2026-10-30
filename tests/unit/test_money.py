import math

import pytest

from lottery.money.budget import BudgetPlanner
from lottery.money.ledger import Ledger
from lottery.money.risk import RiskAssessor


def test_budget_plan_splits_evenly():
    plan = BudgetPlanner().plan(total_budget=2000.0, ticket_price=2.0, periods=100)
    assert plan.per_period == pytest.approx(20.0)
    assert plan.n_tickets_per_period == 10
    assert plan.remainder == pytest.approx(0.0)


def test_budget_plan_keeps_remainder_for_indivisible_budget():
    plan = BudgetPlanner().plan(total_budget=100.0, ticket_price=2.0, periods=3)
    assert plan.per_period == pytest.approx(100 / 3)
    assert plan.n_tickets_per_period == 16
    assert plan.remainder == pytest.approx(4.0)


@pytest.mark.parametrize(("budget", "price", "periods"), [(0, 2, 1), (2, 0, 1), (2, 2, 0)])
def test_budget_plan_rejects_invalid_inputs(budget, price, periods):
    with pytest.raises(ValueError):
        BudgetPlanner().plan(budget, price, periods)


def test_budget_plan_rejects_budget_below_one_ticket():
    with pytest.raises(ValueError):
        BudgetPlanner().plan(total_budget=1.0, ticket_price=2.0, periods=1)


def test_extreme_gap_is_geometric_upper_quantile():
    gap = RiskAssessor().extreme_gap(1 / 16)
    assert gap == math.ceil(math.log(0.01) / math.log(15 / 16))


def test_warn_threshold_carries_gap_and_message():
    warning = RiskAssessor().warn_threshold(1 / 16)
    assert warning.hit_prob == pytest.approx(1 / 16)
    assert warning.extreme_gap == RiskAssessor().extreme_gap(1 / 16)
    assert warning.message


@pytest.mark.parametrize("hit_prob", [0.0, 1.0, -0.1, 1.1])
def test_risk_assessor_rejects_invalid_probability(hit_prob):
    with pytest.raises(ValueError):
        RiskAssessor().extreme_gap(hit_prob)


def test_ledger_curve_tracks_cumulative_net():
    ledger = Ledger(total_budget=100.0)
    ledger.record(1, cost=2.0, payout=0.0)
    ledger.record(2, cost=2.0, payout=10.0)
    curve = ledger.curve()
    assert list(curve["net"]) == pytest.approx([-2.0, 8.0])
    assert list(curve["cumulative_net"]) == pytest.approx([-2.0, 6.0])


def test_stop_loss_triggers_on_budget_loss_ratio():
    ledger = Ledger(total_budget=100.0)
    ledger.record(1, cost=30.0, payout=0.0)
    ledger.record(2, cost=30.0, payout=0.0)
    assert ledger.check_stop_loss(0.5).triggered is True


def test_stop_loss_stays_clear_below_threshold():
    ledger = Ledger(total_budget=100.0)
    ledger.record(1, cost=10.0, payout=0.0)
    signal = ledger.check_stop_loss(0.5)
    assert signal.triggered is False
    assert signal.loss_ratio == pytest.approx(0.1)


def test_ledger_rejects_negative_amounts():
    ledger = Ledger(total_budget=100.0)
    with pytest.raises(ValueError):
        ledger.record(1, cost=-1.0, payout=0.0)
