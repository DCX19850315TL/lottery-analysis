import pytest
from lottery.common.config import get_config, PROJECT_ROOT

def test_loads_project_config():
    cfg = get_config()
    assert cfg.lottery.pick_red == 6
    assert cfg.recommend.window == 100
    assert cfg.recommend.pool_size == 12
    assert cfg.recommend.ticket_count == 5
    assert cfg.crawler.api_url.startswith("https://www.cwl.gov.cn/")

def test_loads_math_section():
    cfg = get_config()
    assert cfg.math.default_pool_size == 12
    assert cfg.math.default_guarantee == "MATCH6_COVER5"
    assert cfg.math.coverage_algorithm == "greedy"
    assert cfg.math.constraint_tolerance == 0.05
    assert cfg.math.solver_timeout_s == 60
    assert cfg.math.blue_joint_cover is False

def test_loads_randomness_section():
    cfg = get_config()
    assert cfg.randomness.alpha == 0.05
    assert cfg.randomness.correction == "bonferroni"
    assert cfg.randomness.test_names == [
        "chi2",
        "ks",
        "runs",
        "autocorr",
        "permutation",
        "geometric",
    ]
    assert cfg.randomness.autocorr_max_lag == 20
    assert cfg.randomness.permutation_trials == 1000
    assert cfg.randomness.seed == 20260101


def test_paths_resolve_under_project_root():
    cfg = get_config()
    assert (PROJECT_ROOT / cfg.paths.raw_csv).parent.name == "raw"

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_config(tmp_path / "nope.yaml")

def test_lottery_rules_must_match_constants(tmp_path):
    data = (PROJECT_ROOT / "config/config.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "config.yaml"
    bad.write_text(data.replace("pick_red: 6", "pick_red: 5"), encoding="utf-8")
    with pytest.raises(ValueError):
        get_config(bad)
