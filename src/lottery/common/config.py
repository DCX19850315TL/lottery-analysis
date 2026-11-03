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


class PrizeConfig(BaseModel):
    """六个奖级的名义奖金（元）；一、二等奖为浮动奖金的名义值。"""

    model_config = ConfigDict(extra="ignore")

    first: float
    second: float
    third: float
    fourth: float
    fifth: float
    sixth: float


class LotteryConfig(BaseModel):
    """双色球号码范围、每注红球数、单注价格与奖级奖金。"""

    model_config = ConfigDict(extra="ignore")

    red_min: int
    red_max: int
    pick_red: int
    blue_min: int
    blue_max: int
    ticket_price: float
    prizes: PrizeConfig


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


class SportsGameConfig(BaseModel):
    """单个体彩玩法：名称、官方 gameNo 与本地 CSV 路径。"""

    model_config = ConfigDict(extra="ignore")

    name: str
    game_no: str
    raw_csv: str


class SportteryConfig(BaseModel):
    """体彩官方接口参数：地址、分页、限速、重试与玩法列表。"""

    model_config = ConfigDict(extra="ignore")

    api_url: str
    home_url: str
    page_size: int
    timeout_s: float
    max_pages: int
    rate_limit_qps: float
    user_agent: str
    accept_language: str
    retry: RetryConfig
    games: dict[str, SportsGameConfig]


class RandomnessConfig(BaseModel):
    """随机性检验参数：显著性水平、校正方式、启用检验与随机种子。"""

    model_config = ConfigDict(extra="ignore")

    alpha: float
    correction: str
    test_names: list[str]
    autocorr_max_lag: int
    permutation_trials: int
    seed: int


class MathConfig(BaseModel):
    """数学路线参数：号码池规模、覆盖等级、构造算法与约束容差。"""

    model_config = ConfigDict(extra="ignore")

    default_pool_size: int
    default_guarantee: str
    coverage_algorithm: str
    constraint_tolerance: float
    solver_timeout_s: float
    blue_joint_cover: bool


class BacktestConfig(BaseModel):
    """回测参数：窗口模式、测试期数与蒙特卡洛试验次数。"""

    model_config = ConfigDict(extra="ignore")

    window_mode: str
    test_periods: int
    monte_carlo_trials: int


class MoneyConfig(BaseModel):
    """资金管理参数：总预算、期数、置信度与止损比例。"""

    model_config = ConfigDict(extra="ignore")

    total_budget: float
    periods: int
    confidence: float
    stop_loss_ratio: float


class RecommendConfig(BaseModel):
    """推荐参数：历史窗口、候选池大小、注数与评分权重。"""

    model_config = ConfigDict(extra="ignore")

    window: int
    pool_size: int
    ticket_count: int
    dlt_back_pool_size: int
    freq_weight: float
    omission_weight: float


class AppConfig(BaseModel):
    """整份应用配置，各段独立校验。"""

    model_config = ConfigDict(extra="ignore")

    project: ProjectConfig
    paths: PathsConfig
    lottery: LotteryConfig
    crawler: CrawlerConfig
    sporttery: SportteryConfig
    randomness: RandomnessConfig
    math: MathConfig
    backtest: BacktestConfig
    money: MoneyConfig
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
