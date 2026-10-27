from datetime import date
from lottery.common.types import DrawRecord
from lottery.teaching.features import FeatureBuilder

def rec(i, reds): return DrawRecord(f"20260{i:02d}", date(2026, 10, i), reds, 9)

DRAWS = [rec(1, (1, 2, 3, 4, 5, 6)),
         rec(2, (1, 2, 3, 4, 5, 7)),
         rec(3, (8, 9, 10, 11, 12, 13))]

def test_frequency_counts_within_window():
    f = FeatureBuilder().build(DRAWS, 10)
    assert f.loc[0, "freq_1"] == 2 and f.loc[0, "freq_7"] == 1 and f.loc[0, "freq_8"] == 1

def test_omission_zero_when_in_latest():
    f = FeatureBuilder().build(DRAWS, 10)
    assert f.loc[0, "omission_8"] == 0

def test_omission_counts_draws_since_last_seen():
    f = FeatureBuilder().build(DRAWS, 10)
    assert f.loc[0, "omission_1"] == 1
    assert f.loc[0, "omission_20"] == 10   # 窗口内从未出现，封顶为 window

def test_window_limits_history():
    f = FeatureBuilder().build(DRAWS, 1)
    assert f.loc[0, "freq_1"] == 0 and f.loc[0, "freq_8"] == 1
