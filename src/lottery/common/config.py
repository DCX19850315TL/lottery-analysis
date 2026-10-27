"""统一配置加载与校验：读取 config/config.yaml，并校验号码规则与常量一致。"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from lottery.common import constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ProjectConfig(BaseModel):
    """项目级元信息：Python 版本约束、随机种子与风险声明。"""

    model_config = ConfigDict(extra="ignore")

    python_requires: str
    random_seed: int
    disclaimers: dict[str, str]


class PathsConfig(BaseModel):
    """项目内数据与产物的相对路径。"""

    model_config = ConfigDict(extra="ignore")

    raw_csv: Path
    cache_dir: Path
    report_dir: Path
    coverage_table: Path


class LotteryConfig(BaseModel):
    """双色球号码范围、每注红球数与单注价格。"""

    model_config = ConfigDict(extra="ignore")

    red_min: int
    red_max: int
    pick_red: int
    blue_min: int
    blue_max: int
    ticket_price: float


class RetryConfig(BaseModel):
    """采集重试策略：最大尝试次数与指数退避基数（秒）。"""

    model_config = ConfigDict(extra="ignore")

    max_attempts: int
    backoff_s: float


class CrawlerConfig(BaseModel):
    """采集器配置：页面地址、开奖 API、限速与重试。"""

    model_config = ConfigDict(extra="ignore")

    base_url: str
    api_url: str
    page_size: int
    timeout_s: float
    max_pages: int
    rate_limit_qps: float
    user_agent: str
    retry: RetryConfig


class RecommendConfig(BaseModel):
    """推荐参数：历史窗口、候选池大小、注数与评分权重。"""

    model_config = ConfigDict(extra="ignore")

    window: int
    pool_size: int
    ticket_count: int
    freq_weight: float
    omission_weight: float


class AppConfig(BaseModel):
    """整份应用配置，各段独立校验。"""

    model_config = ConfigDict(extra="ignore")

    project: ProjectConfig
    paths: PathsConfig
    lottery: LotteryConfig
    crawler: CrawlerConfig
    recommend: RecommendConfig


_NUMBER_RULES = {
    "red_min": constants.RED_MIN,
    "red_max": constants.RED_MAX,
    "pick_red": constants.PICK_RED,
    "blue_min": constants.BLUE_MIN,
    "blue_max": constants.BLUE_MAX,
}


def _validate_lottery_rules(lottery: LotteryConfig) -> None:
    """校验 YAML 号码规则与 constants 完全一致，不一致则抛 ValueError。"""
    mismatches = [
        f"{name}={getattr(lottery, name)} 应为 {expected}"
        for name, expected in _NUMBER_RULES.items()
        if getattr(lottery, name) != expected
    ]
    if mismatches:
        raise ValueError("配置号码规则与 constants 不一致: " + "; ".join(mismatches))


def _load_config(path: Path) -> AppConfig:
    """读取并校验指定 YAML 配置文件；文件不存在时抛 FileNotFoundError。"""
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig(**data)
    _validate_lottery_rules(cfg.lottery)
    return cfg


@lru_cache(maxsize=1)
def _default_config() -> AppConfig:
    """加载并缓存项目默认配置。"""
    return _load_config(CONFIG_PATH)


def get_config(path: Path | None = None) -> AppConfig:
    """返回应用配置；无参调用走缓存，传入路径时每次重新加载并校验。"""
    if path is None:
        return _default_config()
    return _load_config(Path(path))
