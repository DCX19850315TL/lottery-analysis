"""大乐透推荐流程测试：前区/后区打分、选池、抽样与可复现性。"""

from collections import Counter
from datetime import date

import pandas as pd
import pytest

from lottery.data.sporttery import SportsDraw
from lottery.teaching.dlt import (
    BACK_MAX,
    BACK_MIN,
    FRONT_MAX,
    FRONT_MIN,
    PICK_BACK,
    PICK_FRONT,
    DltOptions,
    ScoreWeights,
    ZoneScorer,
    recommend_dlt,
)

DRAWS = [
    SportsDraw("26001", date(2026, 1, 1), (1, 2, 3, 4, 5, 1, 2)),
    SportsDraw("26002", date(2026, 1, 3), (1, 2, 3, 4, 6, 1, 3)),
    SportsDraw("26003", date(2026, 1, 5), (7, 8, 9, 10, 11, 2, 4)),
]


def _manual_scores(draws, window, rule, weights):
    recent = list(draws[-window:])
    numbers = range(rule.minimum, rule.maximum + 1)
    counts = Counter(number for draw in recent for number in draw.numbers[rule.start : rule.stop])
    freq = pd.Series({number: float(counts[number]) for number in numbers}, dtype=float)
    omission = pd.Series(
        {
            number: next(
                (
                    distance
                    for distance, draw in enumerate(reversed(recent))
                    if number in draw.numbers[rule.start : rule.stop]
                ),
                window,
            )
            for number in numbers
        },
        dtype=float,
    )
    return weights.freq * _zscore(freq) + weights.omission * _zscore(omission)


def _zscore(series):
    std = series.std(ddof=0)
    return pd.Series(0.0, index=series.index) if std == 0 else (series - series.mean()) / std


def test_zone_scores_match_manual_frequency_and_omission():
    scorer = ZoneScorer(3, ScoreWeights(1.0, 1.0))
    front, back = scorer.score(DRAWS)
    assert list(front.index) == list(range(FRONT_MIN, FRONT_MAX + 1))
    assert list(back.index) == list(range(BACK_MIN, BACK_MAX + 1))
    pd.testing.assert_series_equal(front, _manual_scores(DRAWS, 3, _front_rule(), ScoreWeights(1, 1)))
    pd.testing.assert_series_equal(back, _manual_scores(DRAWS, 3, _back_rule(), ScoreWeights(1, 1)))


def _front_rule():
    from lottery.teaching.dlt import FRONT_RULE

    return FRONT_RULE


def _back_rule():
    from lottery.teaching.dlt import BACK_RULE

    return BACK_RULE


def test_recommend_dlt_is_valid_and_reproducible():
    options = DltOptions(count=4, front_pool_size=10, back_pool_size=6, window=3, seed=42)
    first = recommend_dlt(DRAWS, options, ScoreWeights(1.0, 1.0))
    second = recommend_dlt(DRAWS, options, ScoreWeights(1.0, 1.0))
    assert first == second
    assert first.latest_issue == "26003"
    assert first.n_draws == 3
    assert len(first.tickets) == 4
    assert len(set(first.tickets)) == 4
    for ticket in first.tickets:
        assert len(ticket.fronts) == PICK_FRONT
        assert len(ticket.backs) == PICK_BACK
        assert list(ticket.fronts) == sorted(ticket.fronts)
        assert list(ticket.backs) == sorted(ticket.backs)
        assert len(set(ticket.fronts)) == PICK_FRONT
        assert len(set(ticket.backs)) == PICK_BACK
        assert all(FRONT_MIN <= number <= FRONT_MAX for number in ticket.fronts)
        assert all(BACK_MIN <= number <= BACK_MAX for number in ticket.backs)
        assert set(ticket.fronts) <= set(first.front_pool)
        assert set(ticket.backs) <= set(first.back_pool)


def test_recommend_dlt_rejects_wrong_number_count():
    broken = SportsDraw("26004", date(2026, 1, 7), (1, 2, 3, 4, 5, 6))
    options = DltOptions(1, 10, 6, 3, 42)
    with pytest.raises(ValueError, match="大乐透"):
        recommend_dlt([broken], options, ScoreWeights(1.0, 1.0))


def test_recommend_dlt_rejects_out_of_range_numbers():
    broken = SportsDraw("26004", date(2026, 1, 7), (1, 2, 3, 4, 36, 1, 2))
    options = DltOptions(1, 10, 6, 3, 42)
    with pytest.raises(ValueError, match="大乐透"):
        recommend_dlt([broken], options, ScoreWeights(1.0, 1.0))
