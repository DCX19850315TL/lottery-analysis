import numpy as np, pandas as pd, pytest
from datetime import date
from lottery.common.config import get_config
from lottery.common.types import DrawRecord, RecommendOptions
from lottery.teaching.features import FeatureBuilder
from lottery.teaching.pipeline import TicketSampler, blue_frequencies, red_scores, select_pool, recommend

def rec(i, reds, blue=9): return DrawRecord(f"20260{i:02d}", date(2026, 10, i), reds, blue)
DRAWS = [rec(1, (1, 2, 3, 4, 5, 6), 9), rec(2, (1, 2, 3, 4, 5, 7), 10),
         rec(3, (1, 2, 8, 9, 10, 11), 9)]

def feats():
    return FeatureBuilder().build(DRAWS, 3)

def test_red_scores_are_zscore_combination():
    f = feats()
    freq = pd.Series({n: f.loc[0, f"freq_{n}"] for n in range(1, 34)}, dtype=float)
    omission = pd.Series({n: f.loc[0, f"omission_{n}"] for n in range(1, 34)}, dtype=float)
    expected = (freq - freq.mean()) / freq.std(ddof=0) + (omission - omission.mean()) / omission.std(ddof=0)
    s = red_scores(f, 1.0, 1.0)
    assert s.index.min() == 1 and s.index.max() == 33
    assert s[1] == pytest.approx(expected[1])

def test_select_pool_is_deterministic_on_ties():
    s = pd.Series({i: float(i % 2) for i in range(1, 34)})
    pool = select_pool(s, 12)
    assert pool[:2] == [1, 3]   # 同分时号码小的优先

def test_tickets_valid_and_counted():
    rng = np.random.default_rng(7)
    s = red_scores(feats(), 1.0, 1.0)
    pool = select_pool(s, 12)
    sampler = TicketSampler(pool, s, pd.Series(1.0, index=range(1, 17)))
    tickets = sampler.sample(5, rng)
    assert len(tickets) == 5
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


def test_recommend_samples_blue_by_window_frequency():
    draws = [rec(i, (1, 2, 3, 4, 5, 6), 9) for i in range(1, 31)]
    result = recommend(draws, get_config(), RecommendOptions(5, 12, 30, 42))
    assert {ticket.blue for ticket in result.pool.tickets} == {9}
