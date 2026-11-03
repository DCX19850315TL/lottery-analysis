# 双色球预测分析系统 — 工程化架构设计

<span style="color:red">风险声明：本系统为历史数据统计与算法教学演示系统。双色球每期开奖为独立随机事件，历史结果与未来开奖无因果关系。所有 AI/ML 模型仅做历史数据拟合，无法提升中奖概率。所有基于覆盖设计的投注优化仅能在「已命中若干号码」的前提下改善奖金回收结构，无法提升头奖中奖概率，不改变彩票负期望收益本质。系统所有对外结论必须附带本条声明。</span>

## 0. 前置冲突与假设

| 项 | 现状 | 决策 |
|---|---|---|
| Python 版本 | 现有 `.venv` 为 CPython 3.14.5（uv 管理），`pyproject.toml` 声明 `>=3.14` | <span style="color:red">冲突：torch / xgboost 对 3.14 无稳定 wheel，深度学习模块无法安装</span>。决策：锁定 CPython 3.12.x，重建 venv，`requires-python = ">=3.10,<3.13"` |
| 依赖版本 | 需求要求固定版本号 | `requirements.txt` 中版本为 3.12 环境下的目标基线，落地时以实测为准并回写，禁止照抄后宣称可用 |
| 数据源 | 福彩官网结构可能变更 | 采集层必须做 schema 校验并在失败时显式抛错，禁止静默返回空 DataFrame |

显式假设：

1. 号码规则固定为红球 1–33 选 6、蓝球 1–16 选 1，全空间为 `C(33,6) × 16 = 1,107,568 × 16 = 17,721,088`。
2. 数学最优路线的选号与覆盖构造不接受任何历史数据输入，该约束由 import 契约在 CI 中强制。
3. 历史数据只出现在三处：采集与持久化、教学路线的特征与训练、回测的评分侧。三处均为只读。

## 1. 项目目录树

```text
lottery-analysis/
├── config/
│   ├── config.yaml               # 唯一配置源：路径/常量/超参/风险阈值/覆盖表路径
│   ├── config.schema.json        # JSON Schema，启动时校验 config.yaml
│   └── logging.yaml              # 日志格式与级别
├── data/                         # 运行时数据（.gitignore，除 .gitkeep）
│   ├── raw/ssq_history.csv       # 原始开奖，唯一持久化格式
│   ├── cache/                    # 采集临时页、覆盖设计缓存
│   └── reports/                  # 随机性检验 / 回测报告输出
├── src/lottery/
│   ├── common/                   # 共享内核：无预测语义，两路线都可依赖
│   │   ├── config.py             # YAML 加载 + Schema 校验 + 单例访问
│   │   ├── constants.py          # RED_MIN/MAX, PICK_RED=6, BLUE_MAX=16, 期号规则
│   │   ├── types.py              # DrawRecord / Ticket / TicketPool 等冻结数据类
│   │   ├── disclaimer.py         # 风险文案常量 + 红色 HTML 横幅生成
│   │   ├── logging_conf.py       # 日志初始化
│   │   └── registry.py           # 通用注册表（模型/策略/覆盖算法扩展点）
│   ├── data/                     # 数据采集（只写不预测）
│   │   ├── crawler.py            # 官网抓取，分页 + 重试 + 限速
│   │   ├── parser.py             # HTML → DrawRecord，schema 校验
│   │   ├── storage.py            # CSV 读写（原子写、列序固定）
│   │   ├── cleaner.py            # 去重 / 排序 / 缺失期检测
│   │   └── repository.py         # 门面，只读 API：all_draws() / latest_issue()
│   ├── theory/                   # ===== 数学最优路线（零历史数据）=====
│   │   ├── combinatorics.py      # 全组合枚举 + 五大特征理论分布
│   │   ├── hypergeometric.py     # 各奖级概率、期望命中数、命中数方差
│   │   ├── prize_table.py        # 奖级定义（常量，非训练所得）
│   │   ├── coverage.py           # 覆盖设计构造器（Greedy / ILP / 查表）
│   │   ├── coverage_tables.py    # 预计算最优覆盖数表（下界达成与否标注）
│   │   ├── bounds.py             # Schönheim / 球覆盖下界计算
│   │   ├── sampler.py            # 分层无偏抽样生成候选池
│   │   └── constraints.py        # 软分布约束（方差平滑，不改概率）
│   ├── randomness/               # ===== 随机性检验（前提验证，不参与选号）=====
│   │   ├── tests.py              # 6 个检验的独立实现
│   │   ├── suite.py              # 套件编排 + Bonferroni 校正
│   │   └── report.py             # 报告序列化（JSON + Markdown）
│   ├── teaching/                 # ===== 教学演示路线（历史数据拟合）=====
│   │   ├── features.py           # 频率 / 遗漏 / 分布 / 和值 / 极差 / 连号
│   │   ├── dataset.py            # 滑动窗口 → (X, y)，严格时序切分
│   │   ├── models/
│   │   │   ├── base.py           # BasePredictor 抽象基类
│   │   │   ├── xgb_model.py      # XGBoost 二分类
│   │   │   └── lstm_model.py     # PyTorch LSTM 时序模型
│   │   └── pipeline.py           # 特征→训练→预测 编排，强制挂声明
│   ├── backtest/                 # 回测（两条路线各一个引擎，共享指标层）
│   │   ├── teaching_engine.py    # 滚动窗口，严格 train/test 隔离
│   │   ├── math_engine.py        # 固定方案对历史结果的纯评分，禁止按期重生成
│   │   ├── metrics.py            # 命中数 / 空窗期 / 回撤等指标
│   │   ├── monte_carlo.py        # 随机基准并行回测
│   │   └── significance.py       # 策略 vs 随机基准的差异检验
│   ├── money/
│   │   ├── budget.py             # 预算按期等额拆分
│   │   ├── risk.py               # 几何分布极端空窗期 + 99% 预警线
│   │   └── ledger.py             # 模拟资金曲线、回撤、止损触发
│   └── ui/
│       ├── app.py                # 入口：先渲染风险横幅，再路由
│       ├── pages/
│       │   ├── 1_teaching.py     # 教学演示页
│       │   ├── 2_math.py         # 数学最优页
│       │   ├── 3_data.py         # 数据管理页
│       │   └── 4_randomness.py   # 随机性检验报告页
│       ├── components.py         # 风险横幅 / 免责卡片 / 红色警示块
│       └── charts.py             # plotly 图表工厂
├── cli/
│   ├── fetch.py                  # 增量采集
│   ├── theory_report.py          # 输出理论分布与覆盖方案
│   └── backtest.py               # 命令行回测
├── tests/
│   ├── unit/                     # 每模块独立
│   ├── property/                 # hypothesis：组合枚举与覆盖完全性
│   ├── integration/              # 端到端链路
│   └── contract/test_no_history_leak.py   # 物理隔离契约测试
├── scripts/
│   └── rebuild_venv.ps1          # 重建 3.12 venv
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .importlinter                 # 分层依赖契约
├── README.md
└── CLAUDE.md
```

物理隔离手段（三重）：

1. 目录分离：`theory/ + randomness/` 属于数学最优路线，`teaching/` 属于教学演示路线，两者之间没有任何交叉 import。
2. `.importlinter` 契约：`theory` 与 `money` 禁止依赖 `data`、`teaching`、`backtest`；`randomness` 可以依赖 `data`，但禁止被 `theory` 依赖。
3. `tests/contract/test_no_history_leak.py`：用 AST 扫描 `theory/`、`money/` 内的全部 import 语句，一旦命中 `data.repository` 或 `pandas.read_csv` 就直接 fail。

## 2. 核心类型（common/types.py）

```python
@dataclass(frozen=True, order=True)
class DrawRecord:
    issue: str                    # 期号 "2024001"
    draw_date: date
    reds: tuple[int, ...]         # 6 个，严格升序，1..33
    blue: int                     # 1..16

@dataclass(frozen=True)
class Ticket:
    reds: tuple[int, ...]         # 6 个，升序
    blue: int | None = None       # None = 仅红球方案

@dataclass(frozen=True)
class TicketPool:
    tickets: tuple[Ticket, ...]
    pool_numbers: tuple[int, ...] # 参与覆盖的号码池
    seed: int

class Guarantee(Enum):
    MATCH6_COVER5 = (6, 5)        # 池中含全部 6 红 → 至少一注中 5 红
    MATCH5_COVER4 = (5, 4)        # 池中含 5 红   → 至少一注中 4 红
    MATCH4_COVER3 = (4, 3)        # 池中含 4 红   → 至少一注中 3 红
```

## 3. 配置管理

统一配置见 `config/config.yaml`，其键树如下。访问方式统一为 `get_config() -> AppConfig`，返回 pydantic v2 模型，在启动期一次性校验并冻结；禁止模块内硬编码任何常量或路径。

```yaml
project:
  python_requires: ">=3.10,<3.13"
  random_seed: 20260101
  disclaimers:
    global_banner: "..."          # 强制红色横幅文案，唯一来源
    teaching_note: "..."
    math_note: "..."

paths:
  raw_csv: "data/raw/ssq_history.csv"
  cache_dir: "data/cache"
  report_dir: "data/reports"
  coverage_table: "src/lottery/theory/coverage_tables.py"

lottery:
  red_min: 1
  red_max: 33
  pick_red: 6
  blue_min: 1
  blue_max: 16
  ticket_price: 2.0

crawler:
  base_url: "..."
  page_size: 100
  timeout_s: 15
  retry: {max_attempts: 5, backoff_s: 2.0}
  rate_limit_qps: 1.0
  user_agent: "..."

randomness:
  alpha: 0.05
  correction: bonferroni         # bonferroni | none
  test_names: [chi2, ks, runs, autocorr, permutation, geometric]
  autocorr_max_lag: 20

teaching:
  window_sizes: [10, 30, 50]
  default_window: 30
  train_ratio: 0.8
  xgb: {n_estimators: 300, max_depth: 4, learning_rate: 0.05, subsample: 0.8}
  lstm: {hidden_size: 64, num_layers: 2, epochs: 50, batch_size: 32, lr: 0.001}

math:
  default_pool_size: 12
  default_guarantee: MATCH6_COVER5
  coverage_algorithm: greedy      # greedy | ilp | table
  constraint_tolerance: 0.05      # 软约束偏离上限（仅方差平滑）
  solver_timeout_s: 60

backtest:
  window_mode: rolling
  test_periods: 100
  monte_carlo_trials: 1000

money:
  total_budget: 2000.0
  periods: 100
  confidence: 0.99
  stop_loss_ratio: 0.5            # 累计亏损达预算 50% 触发止损
```

## 4. 模块接口定义

### 4.1 数据采集（src/lottery/data）

```python
class SSQCrawler:
    def __init__(self, cfg: CrawlerConfig) -> None: ...
    def fetch_page(self, page: int) -> list[DrawRecord]: ...
    def fetch_all(self, max_pages: int | None = None) -> Iterator[list[DrawRecord]]: ...
    def fetch_since(self, issue: str) -> list[DrawRecord]: ...
    # 失败语义：重试耗尽 raise CrawlError，禁止返回空列表

class DrawStorage:
    def load(self) -> pd.DataFrame: ...
    def append(self, records: Sequence[DrawRecord]) -> int: ...   # 返回新增行数
    def rewrite(self, df: pd.DataFrame) -> None: ...              # 原子写：tmp + os.replace

class DrawCleaner:
    def clean(self, df: pd.DataFrame) -> CleanResult: ...
    # CleanResult(n_dropped_dup, n_sorted_fixed, missing_issues: list[str], ok: bool)

class DrawRepository:
    def refresh(self) -> CleanResult: ...
    def all_draws(self) -> list[DrawRecord]: ...
    def latest_issue(self) -> str | None: ...
    def slice(self, start_issue: str, end_issue: str) -> list[DrawRecord]: ...
```

关键实现思路：解析层做字段级 schema 校验，期号必须匹配正则、红球必须 6 个且落在 1–33 且升序、蓝球必须落在 1–16，任一不满足即抛错；存储层写入前用期号做幂等去重；缺失期检测按当年期号序列推算，只报告不填补。

### 4.2 理论计算（src/lottery/theory）— 数学最优路线

```python
class CombinatoricsEngine:
    def __init__(self, red_max: int = 33, pick: int = 6) -> None: ...
    def enumerate_all(self) -> np.ndarray: ...            # (1107568, 6) uint8，一次生成后缓存
    def distribution(self, feature: FeatureName) -> pd.Series: ...
    # FeatureName = SUM | ODD_EVEN | ZONE3 | CONSECUTIVE_RUNS | DISCRETE_ENTROPY
    def feature_values(self, combos: np.ndarray) -> dict[FeatureName, np.ndarray]: ...

class HypergeometricCalculator:
    def prize_probabilities(self) -> dict[PrizeLevel, float]: ...      # 6 个奖级
    def expected_hits(self, n_tickets: int) -> float: ...              # 红球期望命中数
    def hits_variance(self, n_tickets: int) -> float: ...
    def blue_hit_probability(self) -> float: ...                       # 1/16

class CoverageDesigner:
    def __init__(self, algorithm: CoverageAlgorithm) -> None: ...
    def build(self, pool: Sequence[int], guarantee: Guarantee) -> CoveragePlan: ...
    # CoveragePlan(tickets, guarantee, proven_optimal: bool, lower_bound: int, cost: float)

class CoverageTable:
    def lookup(self, v: int, k: int, t: int) -> tuple[int, bool] | None: ...
    # 返回 (最优块数, 是否已达理论下界)；未收录返回 None，调用方降级到构造算法

class PoolSampler:
    def sample(self, pool_size: int, rng: np.random.Generator) -> TicketPool: ...
    # 分层无偏：按 SUM 分位数分层，各层按理论质量比例抽取，不做任何历史加权

class ConstraintApplier:
    def apply(self, pool: TicketPool, target: Distribution, tol: float) -> TicketPool: ...
    # 仅接受/拒绝候选以贴合理论分布，降低奖金波动方差；不改变任何单注中奖概率
```

关键实现思路：

- 覆盖设计的本质是 covering number `C(v, k, t)`。中6保5 对应 `t=5`，中5保4 对应 `t=4`，中4保3 对应 `t=3`，统一抽象为 `C(v, 6, t)`。
- 下界取两条界的较大值：球覆盖界 `ceil(C(v,t) / C(k,t))` 与 Schönheim 递归界 `ceil(v/k · L(v-1, k-1, t-1))`。
- 构造用贪心算法，每一步选出覆盖未覆盖 t-子集数量最多的块，再做局部交换优化；`ilp` 算法走 `scipy.optimize.milp`，超时则返回当前最优解并标注 `proven_optimal=False`。禁止把未达下界的方案对外称为最优。
- 全部结果由 seed 决定，同一 seed 必然产生同一输出，该性质作为可复现性测试项。

### 4.3 随机性检验（src/lottery/randomness）

```python
@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    adjusted_alpha: float
    passed: bool                  # p_value >= adjusted_alpha

class RandomnessSuite:
    def __init__(self, cfg: RandomnessConfig) -> None: ...
    def run(self, draws: Sequence[DrawRecord]) -> RandomnessReport: ...
    # RandomnessReport(results: list[TestResult], overall_pass: bool, n_draws: int)

class RandomnessTester(Protocol):
    name: str
    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]: ...  # (statistic, p)
```

六个检验的检验对象：

| 检验 | 统计量对象 |
|---|---|
| 卡方拟合优度 | 33 个红球各自出现频次 vs 理论 `6/33` |
| K-S 均匀性 | 号码值归一化后的经验分布 vs U(0,1) |
| 游程检验 | 和值序列相对中位数的游程数 |
| 多阶自相关 | 各号码出现指示序列的 Ljung-Box 检验（滞后 1..20） |
| 置换检验 | 打乱期序后重算统计量的经验 p 值 |
| 几何分布拟合 | 同一号码相邻两次出现的间隔 vs `Geom(6/33)` |

Bonferroni 校正取 `adjusted_alpha = alpha / n_tests`。<span style="color:red">本模块输出仅用于验证「独立均匀随机」前提是否成立，结论不得以任何形式反馈到选号逻辑或模型调参。</span>

### 4.4 特征工程（src/lottery/teaching）

```python
class FeatureBuilder:
    def __init__(self, window: int) -> None: ...
    def build(self, draws: Sequence[DrawRecord]) -> pd.DataFrame: ...
    # 行 = 期号，列：freq_1..freq_33, omission_1..omission_33,
    #              odd_count, big_count, zone_counts, sum_value,
    #              range_value, consecutive_count

class DatasetBuilder:
    def to_xy(self, features: pd.DataFrame, draws: Sequence[DrawRecord]
             ) -> tuple[np.ndarray, np.ndarray]: ...
    # y[i][j] = 1 若第 i+1 期红球含号码 j+1；严格 i 只用 <= i 期特征
    def split_time_series(self, X, y, train_ratio: float
                         ) -> tuple[Dataset, Dataset]: ...
```

<span style="color:red">特征与开奖结果无因果关系，仅用于算法演示与拟合能力展示。</span>窗口参数的合法性在下游校验，取值必须属于 `config.teaching.window_sizes`。

### 4.5 教学模型（src/lottery/teaching/models）

```python
class BasePredictor(ABC):
    name: ClassVar[str]
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...   # (n_samples, 33)
    @abstractmethod
    def save(self, path: Path) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BasePredictor": ...

@register_predictor("xgb")
class XGBPredictor(BasePredictor):
    def __init__(self, params: XGBParams, seed: int) -> None: ...

@register_predictor("lstm")
class LSTMPredictor(BasePredictor):
    def __init__(self, params: LSTMParams, seed: int) -> None: ...

class TeachingPipeline:
    def run(self, draws: Sequence[DrawRecord], window: int,
            model_name: str) -> PredictionResult: ...
    # PredictionResult(probs: dict[int, float], top_k: list[int], disclaimer: str)
```

预测结果对象强制携带 `disclaimer` 字段，UI 层拿不到该字段就直接渲染失败，无法绕过。

### 4.6 回测（src/lottery/backtest）

```python
@dataclass(frozen=True)
class Metrics:
    mean_max_red_hits: float              # 每期最佳单注红球命中
    mean_red_hits_per_ticket: float       # 单注平均红球命中
    mean_blue_hits_per_ticket: float
    hit_rate: float
    expected_red_hits_per_ticket: float
    expected_red_hits_per_period: float
    max_gap_periods: int
    max_drawdown: float
    total_cost: float
    total_return: float
    roi: float
    roi_ci_low: float
    roi_ci_high: float

class TeachingBacktestEngine:
    def run(self, draws: Sequence[DrawRecord], model_name: str,
            window: int, test_periods: int) -> BacktestResult: ...

class MathBacktestEngine:
    def run(self, plan: CoveragePlan, draws: Sequence[DrawRecord],
            test_periods: int) -> BacktestResult: ...
    # 关键约束：plan 在进入本函数前已生成完毕，函数内不得调用任何 theory 生成器

class DltBacktestEngine:
    def run(self, draws: Sequence[SportsDraw],
            options: DltBacktestOptions) -> DltBacktestResult: ...
    # 大乐透滚动回测：前区/后区命中分别报告，训练窗口严格截止到被预测期之前

class MonteCarloBaseline:
    def run(self, draws: Sequence[DrawRecord], n_trials: int,
            seed: int) -> pd.DataFrame: ...      # 每次 trial 为纯随机购票

class SignificanceTester:
    def compare(self, strategy: BacktestResult,
                baseline: pd.DataFrame) -> SignificanceResult: ...
    # SignificanceResult(chi2, p_value, effect_size, is_significant, conclusion: str)
```

关键实现思路：

- 教学回测采用滚动窗口，第 `i` 期预测只用 `[i-window, i)` 的数据训练，测试期与训练期零重叠；任何特征计算若引用未来期数据，由 `DatasetBuilder.split_time_series` 的断言拦截。
- 数学回测中方案一次性固定，逐期用真实开奖号码做只读评分。
- <span style="color:red">回测结论不得反向修改策略参数；`SignificanceTester` 输出 `is_significant=False` 时必须在报告首行明示「策略与随机选号无统计显著差异」。</span>

### 4.7 资金管理（src/lottery/money）

```python
class BudgetPlanner:
    def plan(self, total_budget: float, ticket_price: float,
             periods: int) -> BudgetPlan: ...
    # BudgetPlan(per_period: float, n_tickets_per_period: int, remainder: float)

class RiskAssessor:
    def extreme_gap(self, hit_prob: float, confidence: float = 0.99) -> int: ...
    # 几何分布上分位：ceil(log(1-confidence) / log(1-hit_prob))
    def warn_threshold(self, hit_prob: float) -> RiskWarning: ...

class Ledger:
    def record(self, period: int, cost: float, payout: float) -> None: ...
    def curve(self) -> pd.DataFrame: ...
    def check_stop_loss(self, stop_loss_ratio: float) -> StopSignal: ...
```

<span style="color:red">固定单期等额投注是负期望游戏下唯一不加速资金衰减的投注法；追加投注、翻倍追号等任何形式的倍投必然放大期望损失。止损线触发后系统只输出停止提示，不提供继续投注建议。</span>

### 4.8 界面（src/lottery/ui）

```python
def render_disclaimer_banner() -> None: ...   # 由 app.py 在所有页面之前调用一次
def route() -> None: ...                      # st.navigation 多页路由
```

`charts.py` 提供 `cold_hot_bar(freq)`、`sum_distribution_plot(dist, observed)`、`backtest_curve(df)`、`prediction_heatmap(probs)`、`coverage_matrix_plot(plan)`。所有图表标题或副标题内嵌路线标签，即 `[教学演示]` 或 `[数学最优]`。

## 5. 两条端到端数据流

### 5.1 教学演示路线

```text
福彩官网
  └─ SSQCrawler.fetch_*            (data/crawler.py)
      └─ parser → DrawRecord[]     (schema 校验，失败抛错)
          └─ DrawCleaner.clean     (去重/排序/缺失期报告)
              └─ DrawStorage.append → data/raw/ssq_history.csv
                  └─ DrawRepository.all_draws()
                      └─ FeatureBuilder.build(window)         [频率/遗漏/和值/连号…]
                          └─ DatasetBuilder.to_xy + split_time_series   (严格时序隔离)
                              └─ TeachingPipeline.run(model_name)
                                  ├─ XGBPredictor.fit / LSTMPredictor.fit
                                  └─ predict_proba → PredictionResult(+disclaimer)
                                      └─ TeachingBacktestEngine.run(rolling)
                                          └─ Metrics + MonteCarloBaseline + SignificanceTester
                                              └─ ui/pages/1_teaching.py
                                                  └─ render_disclaimer_banner() → plotly 图表
```

<span style="color:red">该链路终点产出的任何数字（概率、命中率、回测曲线）均为历史拟合结果，不代表未来预测能力。</span>

### 5.2 数学最优路线

```text
[分支 A：先验理论，不触碰历史数据]
lottery 常量 (33/6/16)
  └─ CombinatoricsEngine.enumerate_all        → 1,107,568 组合
      └─ feature_values → distribution        → 和值/奇偶/三区/连号/熵 理论分布
  └─ HypergeometricCalculator                 → 6 奖级概率/期望命中/方差
  └─ CoverageTable.lookup(v,6,t) ──命中──→ 最优块数(标注是否达下界)
                              └─未命中─→ CoverageDesigner.build (greedy|ilp) + bounds 下界对比
  └─ PoolSampler.sample(seed)                 → 分层无偏候选池
      └─ ConstraintApplier.apply(tol)         → 方差平滑后的 TicketPool（概率不变）
          └─ CoverageDesigner.build(pool, guarantee) → CoveragePlan

[分支 B：前提验证，只读历史，单向不回流]
DrawRepository.all_draws → RandomnessSuite.run → RandomnessReport
   （仅 UI 展示与报告归档；theory/ 不 import randomness/）

[汇合：评分与风控]
CoveragePlan(已冻结)
  └─ MathBacktestEngine.run(plan, draws, test_periods)   # 纯评分，禁止重新生成
      └─ Metrics + MonteCarloBaseline + SignificanceTester
          └─ BudgetPlanner.plan + RiskAssessor.extreme_gap + Ledger.check_stop_loss
              └─ ui/pages/2_math.py
                  └─ render_disclaimer_banner() → 方案表 + 覆盖矩阵图 + 风险预警
```

<span style="color:red">该链路的核心承诺：选号与覆盖构造的所有输入只有 {号码空间常量, 覆盖等级, 池规模, 随机种子, 理论分布}，历史开奖数据只在最后一步作为评分输入，且不回流、不参与任何参数选择。「数学最优」指覆盖效率在给定预算下达到或逼近理论下界，不指中奖概率提升。</span>

## 6. 依赖清单

运行时依赖见 `requirements.txt`，开发依赖见 `requirements-dev.txt`。两个文件均已锁定版本号，且以 Python 3.12.x 为基线。

安装命令：

```text
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

torch 使用 CPU 版，安装时需要指定 PyTorch 官方 CPU wheel 索引。安装完成后必须运行 `tests/property` 与 `tests/contract` 验证，实测版本与基线不符时以实测结果为准回写依赖文件。

## 7. 扩展接口

| 扩展点 | 机制 | 新增成本 |
|---|---|---|
| 教学模型 | 用 `@register_predictor("name")` 装饰 `BasePredictor` 子类，并加 `config.teaching.<name>` 参数节 | 新增 1 文件 + 1 配置节 |
| 覆盖算法 | 用 `@register_coverage("greedy"/"ilp"/"table"/…)` 装饰实现 `build(pool, guarantee) -> CoveragePlan` 的类 | 新增 1 文件 |
| 候选池约束 | 实现同签名的 `ConstraintApplier`，配置 `math.constraints: [name…]` 按顺序执行 | 新增 1 文件 |
| 覆盖等级 | 扩展 `Guarantee` 枚举的 `(required, covered)` 二元组，统一映射到 `C(v,6,t)` | 1 行 + 表格补录 |
| 随机性检验 | 实现 `RandomnessTester` 协议并注册到 `randomness.test_names` | 新增 1 文件 |
| 奖级/票价规则 | 修改 `prize_table.py` 常量与 `config.lottery` | 不涉及算法层 |

注册表实现于 `common/registry.py`，为泛型 `Registry[T]`，提供 `register(key)` 装饰器与 `get(key)` / `keys()`；重复注册直接抛错，防止静默覆盖。

## 8. 测试策略

| 层 | 内容 |
|---|---|
| 单元 | 各模块独立可跑；采集层用本地 HTML fixture，不访问网络 |
| 属性（hypothesis） | ① `enumerate_all()` 行数等于 `math.comb(33,6)`；② 各奖级计数与组合闭式解一致；③ 覆盖方案对池内全部 `C(v,t)` 个 t-子集逐一断言至少被一注命中（覆盖完全性）；④ 同 seed 两次采样结果完全相同 |
| 契约 | `theory/`、`money/` 的 import 图不含 `data` / `teaching`；`import-linter` 在 CI 执行 |
| 集成 | 采集→清洗→存储→读取全链路（用 mock 源）；教学链路小样本端到端；数学链路理论分布→覆盖→评分全链路 |
| 反过拟合 | 数学路线无任何 `fit()` 调用点（AST 断言）；回测参数与策略参数分文件、无共享可变状态 |

## 9. 落地顺序

1. `common` + `config` + `types`，先立骨架与契约。
2. `data`，采集可用后才有数据。
3. `theory`，数学最优路线核心，优先级最高。
4. `randomness`，前置验证。
5. `backtest` + `money`，两侧共用的评分与风控。
6. `teaching`，辅助功能，最后实现。
7. `ui`，横幅组件最先做，保证声明无处不在。

<span style="color:red">最终交付前必须确认：任何界面、日志、报告、注释中均不得出现「预测准确」「提高中奖率」「必中」「稳赚」类表述；所有结论均附风险与局限性说明。</span>
