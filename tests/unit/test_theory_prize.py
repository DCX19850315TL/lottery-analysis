import math

import pytest

from lottery.theory.hypergeometric import HypergeometricCalculator
from lottery.theory.prize_table import PrizeLevel

TOTAL = math.comb(33, 6)


def _red_probability(hits):
    return math.comb(6, hits) * math.comb(27, 6 - hits) / TOTAL


def test_blue_hit_probability_is_one_in_sixteen():
    assert HypergeometricCalculator().blue_hit_probability() == pytest.approx(1 / 16)


def test_first_prize_is_one_in_17_721_088():
    prizes = HypergeometricCalculator().prize_probabilities()
    assert prizes[PrizeLevel.FIRST] == pytest.approx(1 / 17_721_088, rel=1e-12)


def test_prize_probabilities_match_closed_form():
    prizes = HypergeometricCalculator().prize_probabilities()
    expected = {
        PrizeLevel.FIRST: _red_probability(6) / 16,
        PrizeLevel.SECOND: _red_probability(6) * 15 / 16,
        PrizeLevel.THIRD: _red_probability(5) / 16,
        PrizeLevel.FOURTH: _red_probability(5) * 15 / 16 + _red_probability(4) / 16,
        PrizeLevel.FIFTH: _red_probability(4) * 15 / 16 + _red_probability(3) / 16,
        PrizeLevel.SIXTH: sum(_red_probability(k) for k in range(3)) / 16,
    }
    assert len(prizes) == 6
    for level, value in expected.items():
        assert prizes[level] == pytest.approx(value, rel=1e-12)


def test_prize_probabilities_exclude_non_winning_outcomes():
    prizes = HypergeometricCalculator().prize_probabilities()
    no_prize = 15 / 16 * sum(_red_probability(k) for k in range(4))
    assert sum(prizes.values()) == pytest.approx(1 - no_prize, rel=1e-12)


def test_expected_hits_scale_with_ticket_count():
    calc = HypergeometricCalculator()
    assert calc.expected_hits(1) == pytest.approx(12 / 11, rel=1e-12)
    assert calc.expected_hits(100) == pytest.approx(1200 / 11, rel=1e-12)


def test_hits_variance_matches_hypergeometric_formula():
    single = 6 * (6 / 33) * (27 / 33) * (27 / 32)
    calc = HypergeometricCalculator()
    assert calc.hits_variance(1) == pytest.approx(single, rel=1e-12)
    assert calc.hits_variance(10) == pytest.approx(10 * single, rel=1e-12)
