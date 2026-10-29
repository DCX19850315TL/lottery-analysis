import numpy as np
import pytest

from lottery.common.constants import PICK_RED
from lottery.theory.combinatorics import CombinatoricsEngine, FeatureName
from lottery.theory.constraints import ConstraintApplier
from lottery.theory.sampler import PoolSampler


@pytest.fixture(scope="module")
def engine():
    return CombinatoricsEngine()


@pytest.fixture(scope="module")
def sum_target(engine):
    return engine.distribution(FeatureName.SUM)


def test_same_seed_gives_identical_pool():
    first = PoolSampler().sample(12, np.random.default_rng(42))
    second = PoolSampler().sample(12, np.random.default_rng(42))
    assert first == second


def test_pool_numbers_are_union_of_tickets():
    pool = PoolSampler().sample(12, np.random.default_rng(7))
    union = {number for ticket in pool.tickets for number in ticket.reds}
    assert set(pool.pool_numbers) == union
    assert len(pool.tickets) == 12


@pytest.mark.parametrize("pool_size", [0, -2])
def test_pool_size_out_of_range_rejected(pool_size):
    with pytest.raises(ValueError):
        PoolSampler().sample(pool_size, np.random.default_rng(1))


def test_sampling_follows_theoretical_sum_mass(engine):
    pool = PoolSampler().sample(5000, np.random.default_rng(2026))
    sums = np.array([sum(ticket.reds) for ticket in pool.tickets])
    thresholds = np.quantile(
        engine.feature_values(engine.enumerate_all())[FeatureName.SUM],
        np.linspace(0.1, 0.9, PoolSampler.STRATA - 1),
    )
    deciles = np.searchsorted(thresholds, sums, side="right")
    fractions = np.bincount(deciles, minlength=PoolSampler.STRATA) / len(sums)
    assert np.allclose(fractions, 1 / PoolSampler.STRATA, atol=0.03)


def test_apply_same_inputs_returns_same_pool(sum_target):
    pool = PoolSampler().sample(12, np.random.default_rng(3))
    applier = ConstraintApplier()
    assert applier.apply(pool, sum_target, 0.2) == applier.apply(pool, sum_target, 0.2)


def test_apply_keeps_only_central_sum_band(sum_target):
    pool = PoolSampler().sample(400, np.random.default_rng(11))
    constrained = ConstraintApplier().apply(pool, sum_target, 0.2)
    weights = sum_target / sum_target.sum()
    cdf = np.cumsum(weights.to_numpy())
    values = sum_target.index.to_numpy()
    lower = values[np.searchsorted(cdf, 0.1)]
    upper = values[min(np.searchsorted(cdf, 0.9), len(values) - 1)]
    assert constrained.tickets
    for ticket in constrained.tickets:
        assert lower <= sum(ticket.reds) <= upper
    rejected_ratio = 1 - len(constrained.tickets) / len(pool.tickets)
    assert rejected_ratio == pytest.approx(0.2, abs=0.05)


def test_apply_with_tiny_tolerance_keeps_every_ticket(sum_target):
    pool = PoolSampler().sample(12, np.random.default_rng(5))
    constrained = ConstraintApplier().apply(pool, sum_target, 0.01)
    assert constrained.tickets == pool.tickets


def test_apply_result_is_subset_and_keeps_seed(sum_target):
    pool = PoolSampler().sample(12, np.random.default_rng(13))
    constrained = ConstraintApplier().apply(pool, sum_target, 0.5)
    assert set(constrained.tickets) <= set(pool.tickets)
    assert constrained.seed == pool.seed
    assert len(set(constrained.tickets)) == len(constrained.tickets)


@pytest.mark.parametrize("tol", [0.0, -0.1, 1.5])
def test_invalid_tolerance_rejected(sum_target, tol):
    pool = PoolSampler().sample(PICK_RED, np.random.default_rng(2))
    with pytest.raises(ValueError):
        ConstraintApplier().apply(pool, sum_target, tol)
