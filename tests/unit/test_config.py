import pytest
from lottery.common.config import get_config, PROJECT_ROOT

def test_loads_project_config():
    cfg = get_config()
    assert cfg.lottery.pick_red == 6
    assert cfg.recommend.window == 100
    assert cfg.recommend.pool_size == 12
    assert cfg.recommend.ticket_count == 5
    assert cfg.crawler.api_url.startswith("https://www.cwl.gov.cn/")

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
