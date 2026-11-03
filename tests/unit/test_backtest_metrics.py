import pytest

from lottery.backtest.metrics import (
    PeriodScore,
    ScoreRules,
    build_period_score,
    compute_metrics,
    prize_level,
    score_ticket,
)
from lottery.common.types import Ticket
from lottery.theory.prize_table import PrizeLevel

PAYOUTS = {
    PrizeLevel.FIRST: 5_000_000.0,
    PrizeLevel.SECOND: 100_000.0,
    PrizeLevel.THIRD: 3_000.0,
    PrizeLevel.FOURTH: 200.0,
    PrizeLevel.FIFTH: 10.0,
    PrizeLevel.SIXTH: 5.0,
}


def test_score_ticket_counts_red_and_blue_hits(make_draws):
    draw = make_draws(1, seed=1, chooser=lambda rng, index: (1, 2, 3, 4, 5, 6))[0]
    ticket = Ticket(reds=(1, 2, 3, 7, 8, 9), blue=1)
    assert score_ticket(ticket, draw) == (3, True)


@pytest.mark.parametrize(
    ("red_hits", "blue_hit", "level"),
    [
        (6, True, PrizeLevel.FIRST),
        (6, False, PrizeLevel.SECOND),
        (5, True, PrizeLevel.THIRD),
        (5, False, PrizeLevel.FOURTH),
        (4, True, PrizeLevel.FOURTH),
        (4, False, PrizeLevel.FIFTH),
        (3, True, PrizeLevel.FIFTH),
        (3, False, None),
        (2, True, PrizeLevel.SIXTH),
        (0, False, None),
    ],
)
def test_prize_level_matches_rules(red_hits, blue_hit, level):
    assert prize_level(red_hits, blue_hit) is level


def test_build_period_score_sums_ticket_payouts(make_draws):
    draw = make_draws(1, seed=2, chooser=lambda rng, index: (1, 2, 3, 4, 5, 6))[0]
    tickets = (
        Ticket(reds=(1, 2, 3, 4, 5, 6), blue=1),
        Ticket(reds=(1, 2, 3, 7, 8, 9), blue=2),
    )
    rules = ScoreRules(ticket_price=2.0, payouts=PAYOUTS)
    score = build_period_score(tickets, draw, rules)
    assert score.red_hits == 6
    assert score.red_hit_total == 9
    assert score.blue_hits == 1
    assert score.cost == pytest.approx(4.0)
    assert score.payout == pytest.approx(5_000_000.0)


def test_compute_metrics_aggregates_periods():
    scores = (
        PeriodScore(issue="1", red_hits=6, red_hit_total=6, blue_hits=1, cost=2.0, payout=5.0),
        PeriodScore(issue="2", red_hits=1, red_hit_total=1, blue_hits=0, cost=2.0, payout=0.0),
        PeriodScore(issue="3", red_hits=2, red_hit_total=2, blue_hits=0, cost=2.0, payout=0.0),
        PeriodScore(issue="4", red_hits=3, red_hit_total=3, blue_hits=1, cost=2.0, payout=10.0),
    )
    metrics = compute_metrics(scores, expected_per_ticket=12 / 11, n_tickets=1)
    assert metrics.mean_max_red_hits == pytest.approx(3.0)
    assert metrics.mean_red_hits_per_ticket == pytest.approx(3.0)
    assert metrics.mean_blue_hits_per_ticket == pytest.approx(0.5)
    assert metrics.hit_rate == pytest.approx(0.5)
    assert metrics.expected_red_hits_per_ticket == pytest.approx(12 / 11)
    assert metrics.expected_red_hits_per_period == pytest.approx(12 / 11)
    assert metrics.max_gap_periods == 2
    assert metrics.total_cost == pytest.approx(8.0)
    assert metrics.total_return == pytest.approx(15.0)
    assert metrics.max_drawdown == pytest.approx(4.0)
    assert metrics.roi == pytest.approx(15.0 / 8.0 - 1.0)
    assert metrics.roi_ci_low <= metrics.roi <= metrics.roi_ci_high


def test_compute_metrics_separates_max_from_per_ticket_mean():
    scores = (
        PeriodScore(issue="1", red_hits=4, red_hit_total=7, blue_hits=0, cost=4.0, payout=0.0),
    )
    metrics = compute_metrics(scores, expected_per_ticket=1.0, n_tickets=2)
    assert metrics.mean_max_red_hits == pytest.approx(4.0)
    assert metrics.mean_red_hits_per_ticket == pytest.approx(3.5)
    assert metrics.expected_red_hits_per_period == pytest.approx(2.0)


def test_compute_metrics_rejects_empty_periods():
    with pytest.raises(ValueError):
        compute_metrics((), expected_per_ticket=12 / 11, n_tickets=1)
