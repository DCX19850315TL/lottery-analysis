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


def test_loads_backtest_section():
    cfg = get_config()
    assert cfg.backtest.window_mode == "rolling"
    assert cfg.backtest.test_periods == 100
    assert cfg.backtest.monte_carlo_trials == 1000


def test_loads_prize_amounts():
    prizes = get_config().lottery.prizes
    assert prizes.first == 5_000_000.0
    assert prizes.second == 100_000.0
    assert prizes.third == 3_000.0
    assert prizes.fourth == 200.0
    assert prizes.fifth == 10.0
    assert prizes.sixth == 5.0


def test_loads_sporttery_section():
    cfg = get_config()
    assert cfg.sporttery.api_url.startswith("https://webapi.sporttery.cn/")
    assert cfg.sporttery.games["dlt"].game_no == "85"
    assert cfg.sporttery.games["pl3"].game_no == "35"
    assert cfg.sporttery.games["pl5"].game_no == "350133"
    assert cfg.sporttery.games["dlt"].raw_csv.endswith("dlt_history.csv")


def test_loads_money_section():
    cfg = get_config()
    assert cfg.money.total_budget == 2000.0
    assert cfg.money.periods == 100
    assert cfg.money.confidence == 0.99
    assert cfg.money.stop_loss_ratio == 0.5


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
