# v1「本期推荐」设计

日期：2026-10-27
上游设计：`docs/ARCHITECTURE.md`（本文件只描述 v1 切片，不替代它）

## 1. 意图与原因

用户要的第一个可运行成果是「本期推荐」：一条命令输出本次买几注、每注的红球 6 个号码与蓝球 1 个号码。
v1 用历史数据做简单统计选号，先把「采集 → 存储 → 统计 → 选号 → 打印」整条链路跑通；XGBoost、覆盖设计、回测、UI 留到后续版本。

必须说清的限制：开奖为独立随机事件，历史统计不能提高中奖概率。v1 属于架构中的**教学演示路线**，输出必须带风险声明。

## 2. 目标

- 一条命令完成：拉取/读取历史 → 计算频率与遗漏 → 选出红球池 → 生成定量投注 → 打印。
- 首次运行自动采集全量历史；之后只增量更新。
- 同一 seed 输出完全可复现。
- 采集、解析、存储失败必须显式抛错，不允许静默返回空数据。

## 3. 非目标（v1 不做）

- XGBoost / LSTM 模型
- `theory/` 覆盖设计、下界计算、期望值计算
- 回测引擎、蒙特卡洛、资金管理
- Streamlit UI、plotly 图表
- 随机性检验套件（后续版本用于验证前提，不参与选号）
- `.importlinter` 契约与 AST 防泄漏测试（v1 尚无 `theory/`，无泄漏面）

## 4. 反目标（即使能跑也算失败）

- 静默吞掉网络或解析错误后输出「推荐」
- 输出推荐却不显示风险声明
- 把历史数据写进 `theory/`（v1 不建 `theory/`，规则保持）
- 用历史统计结论宣称能提高中奖概率

## 5. 约束

- 严格遵循 `docs/ARCHITECTURE.md` 的目录结构与命名。
- Python 3.12 venv：`D:\2dev\ide\pythonprojects\Lottery analysis\.venv\Scripts\python.exe`
- TDD：先写失败测试，再实现。
- 函数 ≤20 行、≤3 参数、嵌套 ≤3；公开函数写 docstring。
- 所有参数从 `config/config.yaml` 读取，禁止模块内硬编码。

## 6. 已决定的设计

### 6.1 数据源（实测可用）

`GET https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice`
参数：`name=ssq&pageNo=<n>&pageSize=100&systemType=PC`，`User-Agent` 必填。
响应：`{"state":0,"total":2080,"result":[{"code":"2026122","date":"2026-10-25(日)","red":"01,02,04,17,19,30","blue":"09",...}]}`。
失败语义：HTTP 非 200、`state != 0`、字段缺失、重试耗尽 → 抛错。

### 6.2 模块

```
src/lottery/
├── common/
│   ├── config.py        # YAML + pydantic v2 校验，get_config() 单例
│   ├── constants.py     # 红 1..33 选 6，蓝 1..16
│   ├── types.py         # DrawRecord / Recommendation / Ticket（frozen dataclass）
│   └── disclaimer.py    # 风险文案 + 红色 HTML/终端横幅
├── data/
│   ├── crawler.py       # 分页抓取、重试、限速
│   ├── parser.py        # 原始 dict → DrawRecord，字段级校验
│   ├── storage.py       # CSV 原子写、期号幂等
│   ├── cleaner.py       # 去重 / 排序 / 缺期检测（只报告）
│   └── repository.py    # all_draws / latest_issue / refresh
├── teaching/
│   ├── features.py      # 近 N 期频率 + 当前遗漏
│   └── pipeline.py      # 打分 → 红球池 → 生成注（v1 简单统计实现）
└── cli/
    └── recommend.py     # 命令行入口
```

v1 新增 `recommend:` 配置段（架构文档之外的最小扩展）：
```yaml
recommend:
  window: 100
  pool_size: 12
  ticket_count: 5
  freq_weight: 1.0
  omission_weight: 1.0
```

### 6.3 处理规则

- **解析**：期号 `^\d{7}$`；红球 6 个、互不重复、严格升序、1..33；蓝球 1..16；日期去掉 `(日)` 后按 `YYYY-MM-DD` 解析。
- **存储**：CSV 列固定 `issue,draw_date,red1..red6,blue`；按期号幂等追加；重写用临时文件 + `os.replace`。
- **清洗**：按期号去重（保留最后一条）、按期号排序、按当年期号序列检测缺口并只报告不填补。
- **打分**：取最近 `window` 期；每号码 `freq` = 出现次数，`omission` = 距最近一次出现的期数；对 33 个号码分别做 z-score，`score = freq_weight*z(freq) + omission_weight*z(omission)`。
- **选池**：按 score 降序取前 `pool_size` 个红球。蓝球不建池，直接按窗口内频率加权。
- **生成注**：从红球池按 score 归一化权重做无放回抽样 6 个；拒绝重复整注；生成 `ticket_count` 注。蓝球按窗口频率归一化权重抽 1 个。
- **可复现**：种子取 `config.project.random_seed`，`--seed` 可覆盖；同 seed 结果一致。
- **取数策略**：CSV 不存在或为空 → 全量采集；存在 → 默认直接读，`--refresh` 时增量采集。

### 6.4 命令行

```
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --count 5
```

可选参数：`--pool-size`、`--window`、`--seed`、`--refresh`、`--offline`。
输出：顶部风险横幅；随后「本次推荐 N 注」；每注 `红: 01 05 12 18 27 33  蓝: 09`。

### 6.5 测试

- `tests/unit/test_parser.py`：合法样本解析成功；红球重复/越界/非升序、蓝球越界、期号非法均抛错。
- `tests/unit/test_storage.py`：重复期号幂等；原子重写后内容正确。
- `tests/unit/test_cleaner.py`：去重、排序、缺期报告。
- `tests/unit/test_features.py`：频率与遗漏在手工构造的小样本上取值正确。
- `tests/unit/test_pipeline.py`：注数正确；每注 6 红互异升序、蓝球合法；同 seed 两次结果相同，不同 seed 大概率不同。
- `tests/integration/test_recommend_smoke.py`：用固定 fixture CSV 离线跑通整条命令。

## 7. 与上游文档的偏差（显式记录，不静默合并）

| 偏差 | 上游写法 | v1 做法 | 原因 |
|---|---|---|---|
| 落地顺序 | §9：`common` → `data` → `theory` → … → `teaching` 最后 | v1 做 `common` → `data` → `teaching` | 用户要求第一条可运行命令是「本期推荐」；`theory` 仍排在 XGBoost 之前 |
| CLI 入口 | 只有 `fetch` / `theory_report` / `backtest` | 新增 `cli/recommend.py` | 上游未定义推荐命令，v1 需要 |
| 配置 | 无 `recommend` 段 | 新增 `recommend:` 段 | 窗口 100 / 池 12 / 5 注不属于 `teaching` 或 `math` 语义 |
| 配置校验 | `config.schema.json` + pydantic | 仅 pydantic | 双份校验 v1 无收益；JSON Schema 留到配置面变大时补 |
| 其他骨架 | `logging_conf.py` / `registry.py` / `logging.yaml` | 不建 | v1 无扩展点与日志需求，避免空壳代码 |

## 8. 构建者判定（复审后补齐，均为本文档下的实现决定）

1. **默认取数**：CSV 不存在或为空 → 全量采集；存在 → 默认只读，`--refresh` 才增量采集；`--offline` 禁止一切网络，无本地数据即报错退出。每次输出必须打印数据最新期号与总期数，避免静默使用陈旧数据。
2. **遗漏定义**：在最近 `window` 期内计算。号码在最新一期出现 → 遗漏 0；窗口内未出现 → 遗漏 = `window`（封顶）。`freq` = 窗口内出现次数。
3. **权重换算**：池边界按 `(-score, number)` 确定性排序。抽样用 `softmax(score - max_score)`，避免负分与除零；蓝球按频率同样处理，全 0 时退化为均匀分布。
4. **接口对齐**：实现 `ARCHITECTURE.md` §4.1 的签名——`SSQCrawler.fetch_page/fetch_all/fetch_since`、`DrawStorage.load/append/rewrite`、`DrawCleaner.clean`、`DrawRepository.all_draws/latest_issue/slice/refresh`；`FeatureBuilder.build(draws, window)` 返回 DataFrame，v1 只产出 `freq_*` 与 `omission_*` 列。
5. **全量采集原子性**：所有页拉取并解析成功后才一次性写入 CSV；中途失败抛 `CrawlError`，CSV 保持不变。增量采集同样先汇总再原子追加。
6. **端点配置**：新增 `crawler.api_url` 指向实测可用的 JSON 接口，保留 `base_url` 不作删除。

## 9. 留给构建者的其余细节

- pydantic 模型类内部字段命名。
- 重试/退避用 requests 适配器还是显式循环。
- 终端横幅用 ANSI 红色还是纯文本加框。
- 测试 fixture 的具体历史条数与号码。
