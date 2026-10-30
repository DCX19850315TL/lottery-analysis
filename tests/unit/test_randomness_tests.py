import numpy as np
import pytest

from lottery.randomness.tests import (
    MIN_DRAWS,
    AutocorrTester,
    ChiSquareTester,
    GeometricTester,
    KSTester,
    PermutationTester,
    RunsTester,
)

LOW_NUMBERS = np.arange(1, 11)


def alternating_tickets(rng, index):
    """奇偶期轮换两组互斥号码，制造强时间依赖。"""
    return (1, 2, 3, 4, 5, 6) if index % 2 == 0 else (28, 29, 30, 31, 32, 33)


def always_ball_one(rng, index):
    """每期都含 1 号，制造频次偏差。"""
    return (1, *rng.choice(np.arange(2, 34), size=5, replace=False))


def low_numbers_only(rng, index):
    """只从 1..10 抽号，制造明显的分布偏斜。"""
    return rng.choice(LOW_NUMBERS, size=6, replace=False)


def test_chi2_accepts_uniform_draws(make_draws):
    _, p_value = ChiSquareTester().run(make_draws(600, seed=11))
    assert p_value > 0.05


def test_chi2_rejects_constant_ball(make_draws):
    _, p_value = ChiSquareTester().run(make_draws(300, seed=12, chooser=always_ball_one))
    assert p_value < 0.001


def test_ks_accepts_uniform_draws(make_draws):
    _, p_value = KSTester().run(make_draws(600, seed=13))
    assert p_value > 0.05


def test_ks_rejects_low_number_bias(make_draws):
    _, p_value = KSTester().run(make_draws(300, seed=14, chooser=low_numbers_only))
    assert p_value < 0.001


def test_runs_accepts_uniform_draws(make_draws):
    _, p_value = RunsTester().run(make_draws(600, seed=15))
    assert p_value > 0.05


def test_runs_rejects_alternating_tickets(make_draws):
    _, p_value = RunsTester().run(make_draws(300, seed=16, chooser=alternating_tickets))
    assert p_value < 0.001


def test_autocorr_accepts_uniform_draws(make_draws):
    _, p_value = AutocorrTester(max_lag=20).run(make_draws(600, seed=17))
    assert p_value > 0.01


def test_autocorr_rejects_alternating_tickets(make_draws):
    _, p_value = AutocorrTester(max_lag=20).run(
        make_draws(300, seed=18, chooser=alternating_tickets)
    )
    assert p_value < 0.001


def test_permutation_accepts_uniform_draws(make_draws):
    _, p_value = PermutationTester(trials=200, seed=19).run(make_draws(300, seed=20))
    assert p_value > 0.05


def test_permutation_rejects_alternating_tickets(make_draws):
    _, p_value = PermutationTester(trials=200, seed=21).run(
        make_draws(300, seed=22, chooser=alternating_tickets)
    )
    assert p_value < 0.01


def test_permutation_is_reproducible(make_draws):
    draws = make_draws(200, seed=23)
    first = PermutationTester(trials=100, seed=24).run(draws)
    second = PermutationTester(trials=100, seed=24).run(draws)
    assert first == second


def test_geometric_accepts_uniform_draws(make_draws):
    _, p_value = GeometricTester().run(make_draws(600, seed=25))
    assert p_value > 0.05


def test_geometric_rejects_alternating_tickets(make_draws):
    _, p_value = GeometricTester().run(make_draws(300, seed=26, chooser=alternating_tickets))
    assert p_value < 0.001


@pytest.mark.parametrize(
    "tester",
    [ChiSquareTester(), KSTester(), RunsTester(), AutocorrTester(20), GeometricTester()],
)
def test_too_few_draws_rejected(make_draws, tester):
    with pytest.raises(ValueError):
        tester.run(make_draws(MIN_DRAWS - 1, seed=27))


def test_permutation_rejects_too_few_draws(make_draws):
    with pytest.raises(ValueError):
        PermutationTester(trials=10, seed=28).run(make_draws(MIN_DRAWS - 1, seed=29))
