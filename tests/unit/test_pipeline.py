import numpy as np
import pandas as pd
import pytest
from datetime import date
from lottery.common.config import get_config
from lottery.common.types import DrawRecord, RecommendOptions
from lottery.teaching.features import FeatureBuilder
from lottery.teaching.pipeline import (
    TicketSampler,
    blue_frequencies,
    blue_scores,
    red_scores,
    select_pool,
    recommend,
)

def rec(i, reds, blue=9): return DrawRecord(f"20260{i:02d}", date(2026, 10, i), reds, blue)
DRAWS = [rec(1, (1, 2, 3, 4, 5, 6), 9), rec(2, (1, 2, 3, 4, 5, 7), 10),
         rec(3, (1, 2, 8, 9, 10, 11), 9)]

def feats():
    return FeatureBuilder().build(DRAWS, 3)

def test_red_scores_are_zscore_combination():
    f = feats()
    freq = pd.Series({n: f.loc[0, f"freq_{n}"] for n in range(1, 34)}, dtype=float)
    omission = pd.Series({n: f.loc[0, f"omission_{n}"] for n in range(1, 34)}, dtype=float)
    expected = ((freq - freq.mean()) / freq.std(ddof=0)
                + (omission - omission.mean()) / omission.std(ddof=0))
    s = red_scores(f, 1.0, 1.0)
    assert list(s.index) == list(range(1, 34))
    assert s[1] == pytest.approx(expected[1])

@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0, "pool_size": 12, "window": 3, "seed": 42},
        {"count": 1, "pool_size": 5, "window": 3, "seed": 42},
        {"count": 1, "pool_size": 34, "window": 3, "seed": 42},
        {"count": 1, "pool_size": 12, "window": 0, "seed": 42},
        {"count": 17, "pool_size": 6, "window": 3, "seed": 42},
    ],
)
def test_recommend_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RecommendOptions(**kwargs)


def test_select_pool_breaks_ties_with_seeded_random_order():
    s = pd.Series({i: float(i % 2) for i in range(1, 34)})
    first = select_pool(s, 12, 42)
    assert first == select_pool(s, 12, 42)
    assert first != select_pool(s, 12, 43)
    assert set(first) <= {number for number in range(1, 34) if number % 2 == 1}
    assert first != list(range(1, 24, 2))

def test_tickets_valid_and_counted():
    rng = np.random.default_rng(7)
    s = red_scores(feats(), 1.0, 1.0)
    pool = select_pool(s, 12, 7)
    sampler = TicketSampler(pool, s, pd.Series(1.0, index=range(1, 17)))
    tickets = sampler.sample(5, rng)
    assert len(tickets) == 5
    assert len({ticket.blue for ticket in tickets}) == 5
    for t in tickets:
        assert len(t.reds) == 6 and list(t.reds) == sorted(t.reds)
        assert set(t.reds) <= set(pool) and 1 <= t.blue <= 16

def test_same_seed_reproduces():
    opts = RecommendOptions(count=5, pool_size=12, window=3, seed=42)
    a = recommend(DRAWS, get_config(), opts)
    b = recommend(DRAWS, get_config(), opts)
    assert a == b

def test_different_seed_changes_tickets():
    a = recommend(DRAWS, get_config(), RecommendOptions(5, 12, 3, 42))
    b = recommend(DRAWS, get_config(), RecommendOptions(5, 12, 3, 43))
    assert a.pool.tickets != b.pool.tickets


def test_five_tickets_cover_entire_pool():
    result = recommend(DRAWS, get_config(), RecommendOptions(5, 12, 3, 42))
    tickets = result.pool.tickets
    assert set().union(*(set(ticket.reds) for ticket in tickets)) == set(
        result.pool.pool_numbers
    )

def test_negative_scores_do_not_produce_nan():
    s = pd.Series(np.linspace(-5, -1, 33), index=range(1, 34))
    sampler = TicketSampler(list(range(1, 34)), s,
                            pd.Series(0.0, index=range(1, 17)))
    assert len(sampler.sample(3, np.random.default_rng(1))) == 3


def test_blue_frequencies_counts_only_last_window_draws():
    f = blue_frequencies(DRAWS, 2)
    assert list(f.index) == list(range(1, 17))
    assert f.sum() == 2.0
    assert f[9] == 1.0 and f[10] == 1.0 and f[1] == 0.0
    last = blue_frequencies(DRAWS, 1)
    assert last[9] == 1.0 and last[10] == 0.0


@pytest.mark.parametrize(
    "weights, prefix",
    [((1.0, 0.0), "blue_freq"), ((0.0, 1.0), "blue_omission")],
)
def test_blue_scores_weights_select_matching_feature(weights, prefix):
    f = feats()
    column = pd.Series({n: f.loc[0, f"{prefix}_{n}"] for n in range(1, 17)}, dtype=float)
    expected = (column - column.mean()) / column.std(ddof=0)
    pd.testing.assert_series_equal(blue_scores(f, *weights), expected, check_names=False)


def test_blue_scores_constant_blue_history_is_neutral():
    draws = [
        DrawRecord(f"2026{i:03d}", date(2026, 1, 1), (1, 2, 3, 4, 5, 6), 9)
        for i in range(1, 101)
    ]
    scores = blue_scores(FeatureBuilder().build(draws, 100), 1.0, 1.0)
    assert scores.tolist() == pytest.approx([0.0] * 16)


def test_recommend_allocates_distinct_blues_before_repeating():
    draws = [
        DrawRecord(f"2026{i:03d}", date(2026, 1, 1), (1, 2, 3, 4, 5, 6), 9)
        for i in range(1, 101)
    ]
    result = recommend(draws, get_config(), RecommendOptions(5, 12, 100, 42))
    assert len({ticket.blue for ticket in result.pool.tickets}) == 5


@pytest.mark.parametrize("weights, prefix", [((1.0, 0.0), "freq"), ((0.0, 1.0), "omission")])
def test_red_scores_weights_select_matching_feature(weights, prefix):
    f = feats()
    column = pd.Series({n: f.loc[0, f"{prefix}_{n}"] for n in range(1, 34)}, dtype=float)
    expected = (column - column.mean()) / column.std(ddof=0)
    pd.testing.assert_series_equal(red_scores(f, *weights), expected, check_names=False)


def test_red_scores_constant_feature_contributes_zero():
    columns = {f"freq_{n}": 4.0 for n in range(1, 34)}
    columns.update({f"omission_{n}": float(n) for n in range(1, 34)})
    scores = red_scores(pd.DataFrame([columns]), 1.0, 1.0)
    std = ((33 ** 2 - 1) / 12) ** 0.5  # 1..33 的总体标准差
    assert scores.tolist() == pytest.approx([(n - 17) / std for n in range(1, 34)])


@pytest.mark.parametrize(
    "weights, required, allowed",
    [
        ((1.0, 0.0), set(range(1, 12)), set(range(1, 34))),
        ((0.0, 1.0), set(), set(range(12, 34))),
    ],
)
def test_recommend_uses_configured_weights(weights, required, allowed):
    cfg = get_config()
    rec_cfg = cfg.recommend.model_copy(
        update={"freq_weight": weights[0], "omission_weight": weights[1]}
    )
    options = RecommendOptions(5, 12, 3, 42)
    result = recommend(DRAWS, cfg.model_copy(update={"recommend": rec_cfg}), options)
    pool = set(result.pool.pool_numbers)
    assert required <= pool <= allowed
    assert len(pool) == 12


def test_sample_uses_blue_capacity_then_rejects_overflow():
    blue = pd.Series(0.0, index=range(1, 17))
    sampler = TicketSampler([1, 2, 3, 4, 5, 6], pd.Series(0.0, index=range(1, 34)), blue)
    rng = np.random.default_rng(1)
    assert len(sampler.sample(16, rng)) == 16
    with pytest.raises(ValueError, match="16"):
        sampler.sample(17, rng)
