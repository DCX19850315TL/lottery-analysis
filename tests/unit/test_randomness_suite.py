import json

import pytest

from lottery.common.config import RandomnessConfig
from lottery.randomness.report import to_json, to_markdown
from lottery.randomness.suite import RandomnessSuite, TestResult

TEST_NAMES = ["chi2", "ks", "runs", "autocorr", "permutation", "geometric"]
NOTE = "随机性检验仅用于验证独立均匀随机前提"


def make_config(**overrides):
    values = {
        "alpha": 0.05,
        "correction": "bonferroni",
        "test_names": list(TEST_NAMES),
        "autocorr_max_lag": 20,
        "permutation_trials": 200,
        "seed": 20260101,
    }
    values.update(overrides)
    return RandomnessConfig(**values)


def always_ball_one(rng, index):
    return (1, *rng.choice(range(2, 34), size=5, replace=False))


def test_run_returns_one_result_per_enabled_test(make_draws):
    report = RandomnessSuite(make_config()).run(make_draws(300, seed=31))
    assert [result.name for result in report.results] == TEST_NAMES
    assert report.n_draws == 300


def test_bonferroni_divides_alpha_by_test_count(make_draws):
    report = RandomnessSuite(make_config()).run(make_draws(100, seed=32))
    assert all(result.adjusted_alpha == pytest.approx(0.05 / len(TEST_NAMES)) for result in report.results)


def test_correction_none_keeps_alpha(make_draws):
    report = RandomnessSuite(make_config(correction="none")).run(make_draws(100, seed=33))
    assert all(result.adjusted_alpha == pytest.approx(0.05) for result in report.results)


def test_uniform_draws_pass_overall(make_draws):
    report = RandomnessSuite(make_config()).run(make_draws(600, seed=34))
    assert report.overall_pass, [result for result in report.results if not result.passed]


def test_biased_draws_fail_overall_on_chi2(make_draws):
    report = RandomnessSuite(make_config()).run(
        make_draws(300, seed=35, chooser=always_ball_one)
    )
    assert not report.overall_pass
    failed = {result.name for result in report.results if not result.passed}
    assert "chi2" in failed


def test_run_is_deterministic(make_draws):
    draws = make_draws(200, seed=36)
    first = RandomnessSuite(make_config()).run(draws)
    second = RandomnessSuite(make_config()).run(draws)
    assert first == second


def test_unknown_test_name_rejected():
    with pytest.raises(ValueError):
        RandomnessSuite(make_config(test_names=["chi2", "nope"]))


@pytest.mark.parametrize("correction", ["holm", ""])
def test_invalid_correction_rejected(correction):
    with pytest.raises(ValueError):
        RandomnessSuite(make_config(correction=correction))


def test_empty_test_list_rejected():
    with pytest.raises(ValueError):
        RandomnessSuite(make_config(test_names=[]))


def test_passed_uses_adjusted_alpha():
    passing = TestResult(name="chi2", statistic=1.0, p_value=0.03, adjusted_alpha=0.02)
    failing = TestResult(name="chi2", statistic=1.0, p_value=0.01, adjusted_alpha=0.02)
    assert passing.passed
    assert not failing.passed


@pytest.fixture
def report(make_draws):
    return RandomnessSuite(make_config()).run(make_draws(300, seed=37))


def test_json_report_carries_results_and_note(report):
    payload = json.loads(to_json(report, NOTE))
    assert payload["n_draws"] == 300
    assert payload["overall_pass"] is True
    assert payload["note"] == NOTE
    assert [item["name"] for item in payload["results"]] == TEST_NAMES
    assert all("p_value" in item and "adjusted_alpha" in item for item in payload["results"])


def test_markdown_report_lists_every_test_and_note(report):
    markdown = to_markdown(report, NOTE)
    assert NOTE in markdown
    for name in TEST_NAMES:
        assert name in markdown
