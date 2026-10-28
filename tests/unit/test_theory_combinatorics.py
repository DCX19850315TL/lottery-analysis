import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from lottery.common.constants import PICK_RED, RED_MAX
from lottery.theory.combinatorics import CombinatoricsEngine, FeatureName

TOTAL = math.comb(RED_MAX, PICK_RED)


@pytest.fixture(scope="module")
def engine():
    return CombinatoricsEngine()


@pytest.fixture(scope="module")
def combos(engine):
    return engine.enumerate_all()


def test_enumerate_all_shape_and_dtype(combos):
    assert combos.shape == (TOTAL, PICK_RED)
    assert combos.dtype == np.uint8


def test_enumerate_all_is_strictly_increasing_within_range(combos):
    assert np.all(np.diff(combos, axis=1) > 0)
    assert combos.min() == 1 and combos.max() == RED_MAX


def test_enumerate_all_has_no_duplicates(combos):
    assert len(np.unique(combos, axis=0)) == TOTAL


def test_enumerate_all_is_cached(engine):
    assert engine.enumerate_all() is engine.enumerate_all()


def test_sum_feature_range_and_mean(engine, combos):
    sums = engine.feature_values(combos)[FeatureName.SUM]
    assert sums.min() == 21 and sums.max() == 183
    assert sums.mean() == pytest.approx(102.0)


def test_distribution_matches_feature_counts(engine, combos):
    dist = engine.distribution(FeatureName.SUM)
    counts = np.unique(combos.sum(axis=1), return_counts=True)[1]
    assert dist.sum() == TOTAL
    assert dist.tolist() == counts.tolist()


def test_odd_even_closed_form(engine):
    dist = engine.distribution(FeatureName.ODD_EVEN)
    assert list(dist.index) == [0, 1, 2, 3, 4, 5, 6]
    assert dist[0] == math.comb(16, 6)
    assert dist[6] == math.comb(17, 6)


def test_zone3_closed_form(engine):
    dist = engine.distribution(FeatureName.ZONE3)
    assert dist[222] == 55 ** 3
    assert list(dist.index) == sorted(dist.index)


def test_consecutive_runs_closed_form(engine):
    dist = engine.distribution(FeatureName.CONSECUTIVE_RUNS)
    assert dist[1] == math.comb(RED_MAX - PICK_RED + 1, PICK_RED)
    assert dist.index.max() == PICK_RED


def test_entropy_of_single_zone_is_zero(engine):
    values = engine.feature_values(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint8))
    assert values[FeatureName.DISCRETE_ENTROPY][0] == pytest.approx(0.0)


def test_entropy_of_balanced_zones_is_log3(engine):
    values = engine.feature_values(np.array([[1, 2, 12, 13, 23, 24]], dtype=np.uint8))
    assert values[FeatureName.DISCRETE_ENTROPY][0] == pytest.approx(math.log(3))


@given(st.lists(st.integers(1, RED_MAX), min_size=PICK_RED, max_size=PICK_RED, unique=True))
@settings(max_examples=50, deadline=None)
def test_features_agree_with_brute_force(numbers):
    combo = np.array([sorted(numbers)], dtype=np.uint8)
    values = CombinatoricsEngine().feature_values(combo)
    assert values[FeatureName.SUM][0] == sum(numbers)
    assert values[FeatureName.ODD_EVEN][0] == sum(1 for n in numbers if n % 2)
    assert values[FeatureName.CONSECUTIVE_RUNS][0] == _longest_run(sorted(numbers))
    assert values[FeatureName.DISCRETE_ENTROPY][0] >= 0.0


def _longest_run(numbers):
    best = run = 1
    for left, right in zip(numbers, numbers[1:]):
        run = run + 1 if right - left == 1 else 1
        best = max(best, run)
    return best
