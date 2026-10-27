# v1「本期推荐」Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一条离线可测的命令，从历史开奖中用频率与遗漏打分选出红球池，生成 N 注红蓝号码并附风险声明。

**Architecture:** 严格按 `docs/ARCHITECTURE.md` 的目录分层：`common` 提供类型与配置，`data` 负责采集/解析/存储/清洗/只读仓库，`teaching` 用简单统计打分并生成投注，`cli` 只做编排与打印。历史数据只进入 `teaching`，不建 `theory`。

**Tech Stack:** Python 3.12、pydantic 2.8、pandas 2.2、numpy 1.26、requests 2.32、PyYAML、pytest。

**Spec:** `docs/superpowers/specs/2026-10-27-v1-recommend-design.md`

## Global Constraints

- 解释器：`D:\2dev\ide\pythonprojects\Lottery analysis\.venv\Scripts\python.exe`
- 测试路径：`tests/`，`pyproject.toml` 已设 `pythonpath = ["src"]`
- 函数 ≤20 行、≤3 参数、嵌套 ≤3 层；公开函数写 docstring
- 所有参数取自 `config/config.yaml`，不得硬编码
- 号码规则：红 1–33 选 6，蓝 1–16 选 1；期号 `^\d{7}$`
- 任何输出与异常信息不得出现「预测准确」「提高中奖率」「必中」「稳赚」
- 每个任务结束提交一次；提交信息末尾加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`

## Review Focus

- 本地 CSV 为空、缺列或 0 行：必须报错退出，不得输出任何推荐（Task 10）
- 历史期数少于窗口：用全部可用期数计算，不崩、不补零（Task 8）
- CLI 参数越界：`pool_size < 6`、`pool_size > 33`、`count < 1`、`window < 1` 必须明确报错（Task 10）
- 采集或解析中途失败：CSV 必须保持原样，抛 `CrawlError` / `ParseError`（Task 6、Task 7）
- 陈旧数据：输出必须打印最新期号与总期数，让用户看见数据截至哪里（Task 10）

---

### Task 1: 类型与常量

**Files:**
- Create: `src/lottery/__init__.py`
- Create: `src/lottery/common/__init__.py`
- Create: `src/lottery/common/constants.py`
- Create: `src/lottery/common/types.py`
- Test: `tests/unit/test_types.py`

**Interfaces:**
- Produces: `RED_MIN=1, RED_MAX=33, PICK_RED=6, BLUE_MIN=1, BLUE_MAX=16, ISSUE_PATTERN`
- Produces: `DrawRecord(issue: str, draw_date: date, reds: tuple[int, ...], blue: int)`，frozen、order
- Produces: `Ticket(reds: tuple[int, ...], blue: int | None = None)`，frozen
- Produces: `TicketPool(tickets: tuple[Ticket, ...], pool_numbers: tuple[int, ...], seed: int)`，frozen
- Produces: `Recommendation(pool: TicketPool, latest_issue: str, window: int, n_draws: int)`，frozen
- Produces: `RecommendOptions(count: int, pool_size: int, window: int, seed: int)`，frozen

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_types.py
from datetime import date
import pytest
from lottery.common.types import DrawRecord, Ticket, TicketPool

def test_draw_record_accepts_valid():
    d = DrawRecord("2026122", date(2026, 10, 25), (1, 2, 4, 17, 19, 30), 9)
    assert d.reds == (1, 2, 4, 17, 19, 30)

@pytest.mark.parametrize("reds", [
    (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 5),
    (0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 34), (2, 1, 3, 4, 5, 6),
])
def test_draw_record_rejects_bad_reds(reds):
    with pytest.raises(ValueError):
        DrawRecord("2026122", date(2026, 10, 25), reds, 9)

def test_draw_record_rejects_bad_blue_and_issue():
    with pytest.raises(ValueError):
        DrawRecord("2026122", date(2026, 10, 25), (1, 2, 3, 4, 5, 6), 17)
    with pytest.raises(ValueError):
        DrawRecord("202612", date(2026, 10, 25), (1, 2, 3, 4, 5, 6), 1)

def test_ticket_rejects_bad_reds():
    assert Ticket((1, 2, 3, 4, 5, 6), 9).blue == 9
    with pytest.raises(ValueError):
        Ticket((1, 2, 3, 4, 5, 6), 0)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottery'`

- [ ] **Step 3: 实现**

`constants.py` 只放 `RED_MIN/RED_MAX/PICK_RED/BLUE_MIN/BLUE_MAX` 与 `ISSUE_PATTERN = re.compile(r"^\d{7}$")`。
`types.py` 三个 dataclass 用 `__post_init__` 校验：红球长度 6、严格升序、无重复、限 1..33；蓝球 1..16；期号正则。`TicketPool` 校验 `pool_numbers` 非空且都在 1..33。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_types.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/common/constants.py src/lottery/common/types.py src/lottery/__init__.py src/lottery/common/__init__.py tests/unit/test_types.py
git commit -m "feat: 添加双色球核心类型与号码常量

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: 配置加载

**Files:**
- Create: `src/lottery/common/config.py`
- Modify: `config/config.yaml`（`crawler` 增 `api_url`，文件末新增 `recommend` 段）
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: `constants` 的号码规则（用于一致性校验）
- Produces: `AppConfig`，字段 `project: ProjectConfig`、`paths: PathsConfig`、`lottery: LotteryConfig`、`crawler: CrawlerConfig`、`recommend: RecommendConfig`
- Produces: `get_config(path: Path | None = None) -> AppConfig`
- Produces: `PROJECT_ROOT: Path`
- Produces: `CrawlerConfig.retry.max_attempts/backoff_s`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_config.py
from pathlib import Path
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
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -v`
Expected: FAIL — import error / `get_config` 不存在

- [ ] **Step 3: 实现**

`config.yaml` 的 `crawler` 增加：
```yaml
  api_url: "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
```
文件末增加：
```yaml
recommend:
  window: 100
  pool_size: 12
  ticket_count: 5
  freq_weight: 1.0
  omission_weight: 1.0
```
`config.py` 用 pydantic v2：各段独立 `BaseModel`，`model_config = ConfigDict(extra="ignore")`；`PROJECT_ROOT = Path(__file__).resolve().parents[3]`；`get_config` 带缓存，仅无参调用走缓存；加载后断言 `lottery` 五项等于 constants，否则 `ValueError`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add config/config.yaml src/lottery/common/config.py tests/unit/test_config.py
git commit -m "feat: 加载并校验统一配置，新增采集 API 与推荐参数段

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: HTML/JSON 解析

**Files:**
- Create: `src/lottery/data/__init__.py`
- Create: `src/lottery/data/parser.py`
- Test: `tests/unit/test_parser.py`

**Interfaces:**
- Consumes: `DrawRecord`
- Produces: `ParseError(ValueError)`
- Produces: `parse_draw(raw: dict[str, object]) -> DrawRecord`
- Produces: `parse_response(payload: dict[str, object]) -> list[DrawRecord]`（`state != 0` 或 `result` 非列表时抛 `ParseError`）

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_parser.py
import pytest
from lottery.data.parser import parse_draw, parse_response, ParseError

GOOD = {"code": "2026122", "date": "2026-10-25(日)",
        "red": "01,02,04,17,19,30", "blue": "09"}

def test_parse_draw_valid():
    d = parse_draw(GOOD)
    assert d.issue == "2026122"
    assert d.reds == (1, 2, 4, 17, 19, 30)
    assert d.blue == 9
    assert str(d.draw_date) == "2026-10-25"

@pytest.mark.parametrize("patch", [
    {"red": "01,02,04,17,19"}, {"red": "01,02,04,17,19,19"},
    {"red": "01,02,04,17,19,34"}, {"red": "30,19,17,04,02,01"},
    {"blue": "00"}, {"blue": "17"}, {"code": "202612"}, {"date": "not-a-date"},
])
def test_parse_draw_rejects_bad_fields(patch):
    with pytest.raises(ParseError):
        parse_draw({**GOOD, **patch})

def test_parse_response_state_error():
    with pytest.raises(ParseError):
        parse_response({"state": 1, "result": []})

def test_parse_response_returns_records():
    out = parse_response({"state": 0, "result": [GOOD, {**GOOD, "code": "2026121"}]})
    assert [d.issue for d in out] == ["2026122", "2026121"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_parser.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`parse_draw` 拆成小函数：`_parse_issue`、`_parse_date`（去 `(日)`，`date.fromisoformat`）、`_parse_reds`（split、int、tuple）、`_parse_blue`。字段缺失或类型错误统一包成 `ParseError`。`DrawRecord` 的校验异常也在 `parse_draw` 内转成 `ParseError`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_parser.py -v`
Expected: PASS（11 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/data/__init__.py src/lottery/data/parser.py tests/unit/test_parser.py
git commit -m "feat: 解析福彩接口响应为开奖记录并做字段校验

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: CSV 存储

**Files:**
- Create: `src/lottery/data/storage.py`
- Test: `tests/unit/test_storage.py`

**Interfaces:**
- Consumes: `DrawRecord`
- Produces: `COLUMNS = ["issue", "draw_date", "red1", "red2", "red3", "red4", "red5", "red6", "blue"]`
- Produces: `DrawStorage(path: Path)`；`load() -> pd.DataFrame`（文件不存在返回同列空表）；`append(records: Sequence[DrawRecord]) -> int`；`rewrite(df: pd.DataFrame) -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_storage.py
from datetime import date
import pandas as pd
from lottery.common.types import DrawRecord
from lottery.data.storage import DrawStorage, COLUMNS

def rec(issue, reds=(1, 2, 3, 4, 5, 6), blue=9):
    return DrawRecord(issue, date(2026, 10, 25), reds, blue)

def test_load_missing_file_returns_empty_frame(tmp_path):
    df = DrawStorage(tmp_path / "x.csv").load()
    assert list(df.columns) == COLUMNS and len(df) == 0

def test_append_is_idempotent(tmp_path):
    s = DrawStorage(tmp_path / "x.csv")
    assert s.append([rec("2026122"), rec("2026121")]) == 2
    assert s.append([rec("2026122")]) == 0
    assert len(s.load()) == 2

def test_append_keeps_row_values(tmp_path):
    s = DrawStorage(tmp_path / "x.csv")
    s.append([rec("2026122", (1, 2, 4, 17, 19, 30), 9)])
    row = s.load().iloc[0]
    assert (row["red1"], row["red6"], row["blue"]) == (1, 30, 9)
    assert row["draw_date"] == "2026-10-25"

def test_rewrite_replaces_atomically(tmp_path):
    s = DrawStorage(tmp_path / "x.csv")
    s.append([rec("2026122")])
    s.rewrite(pd.DataFrame([{**{c: 0 for c in COLUMNS}, "issue": "2026120"}]).astype({"issue": str}))
    assert s.load()["issue"].tolist() == ["2026120"]
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_storage.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`append` 先 `load`，用已有 `issue` 集合过滤，拼行后按 `issue` 升序 `rewrite`，返回新增行数。`rewrite` 写 `path.with_suffix(".csv.tmp")` 后 `os.replace`；父目录 `mkdir(parents=True, exist_ok=True)`。`draw_date` 存 ISO 字符串。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_storage.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/data/storage.py tests/unit/test_storage.py
git commit -m "feat: CSV 存储支持期号幂等追加与原子重写

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: 清洗与缺期检测

**Files:**
- Create: `src/lottery/data/cleaner.py`
- Test: `tests/unit/test_cleaner.py`

**Interfaces:**
- Produces: `CleanResult`（frozen dataclass：`n_dropped_dup: int`、`n_sorted_fixed: int`、`missing_issues: list[str]`、`ok: bool`、`frame: pd.DataFrame`）
- Produces: `DrawCleaner.clean(df: pd.DataFrame) -> CleanResult`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_cleaner.py
import pandas as pd
from lottery.data.cleaner import DrawCleaner
from lottery.data.storage import COLUMNS

def frame(issues):
    rows = []
    for i in issues:
        rows.append({"issue": i, "draw_date": "2026-10-25", "red1": 1, "red2": 2,
                     "red3": 3, "red4": 4, "red5": 5, "red6": 6, "blue": 9})
    return pd.DataFrame(rows, columns=COLUMNS)

def test_dedupes_keeping_last_and_sorts():
    df = frame(["2026003", "2026001", "2026003", "2026002"])
    df.loc[2, "blue"] = 10
    res = DrawCleaner().clean(df)
    assert res.n_dropped_dup == 1
    assert res.n_sorted_fixed == 1
    assert res.frame["issue"].tolist() == ["2026001", "2026002", "2026003"]
    assert res.frame.iloc[-1]["blue"] == 10

def test_reports_missing_issues_without_filling():
    res = DrawCleaner().clean(frame(["2026001", "2026003"]))
    assert res.missing_issues == ["2026002"]
    assert len(res.frame) == 2

def test_clean_frame_is_ok_when_no_missing():
    res = DrawCleaner().clean(frame(["2026001", "2026002"]))
    assert res.ok is True and res.missing_issues == []

def test_empty_frame_is_ok():
    res = DrawCleaner().clean(frame([]))
    assert res.ok is True and res.frame.empty
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_cleaner.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`n_sorted_fixed` 为「原顺序不是严格升序」时记 1，否则 0。去重用 `drop_duplicates(subset="issue", keep="last")`。缺期按前 4 位年份分组，取该年最早期号与最晚期号的序号区间，报告区间内缺失；格式 `YYYYNNN`。`ok` 定义为无缺期。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_cleaner.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/data/cleaner.py tests/unit/test_cleaner.py
git commit -m "feat: 开奖数据去重排序与缺期报告

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: 采集器

**Files:**
- Create: `src/lottery/data/crawler.py`
- Test: `tests/unit/test_crawler.py`

**Interfaces:**
- Consumes: `CrawlerConfig`、`parse_response`
- Produces: `CrawlError(RuntimeError)`
- Produces: `SSQCrawler(cfg: CrawlerConfig, session: requests.Session | None = None)`
- Produces: `fetch_page(page: int) -> list[DrawRecord]`
- Produces: `fetch_all(max_pages: int | None = None) -> Iterator[list[DrawRecord]]`
- Produces: `fetch_since(issue: str) -> list[DrawRecord]`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_crawler.py
import pytest, requests
from lottery.common.config import get_config
from lottery.data.crawler import SSQCrawler, CrawlError

class FakeResp:
    def __init__(self, status, payload): self.status_code, self._p = status, payload
    def json(self): return self._p
    def raise_for_status(self):
        if self.status_code >= 400: raise requests.HTTPError(str(self.status_code))

class FakeSession:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def get(self, url, **kw):
        self.calls.append((url, kw)); return self.responses.pop(0)

def payload(*codes):
    return {"state": 0, "result": [{"code": c, "date": "2026-10-25(日)",
            "red": "01,02,03,04,05,06", "blue": "09"} for c in codes]}

def test_fetch_page_parses():
    s = FakeSession([FakeResp(200, payload("2026122"))])
    out = SSQCrawler(get_config().crawler, s).fetch_page(1)
    assert out[0].issue == "2026122"
    assert s.calls[0][0].endswith("findDrawNotice")

def test_state_error_raises():
    s = FakeSession([FakeResp(200, {"state": 1, "result": []})])
    with pytest.raises(CrawlError):
        SSQCrawler(get_config().crawler, s).fetch_page(1)

def test_retries_then_raises(monkeypatch):
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.retry.max_attempts = 2; cfg.retry.backoff_s = 0.0
    s = FakeSession([FakeResp(500, {}), FakeResp(500, {})])
    with pytest.raises(CrawlError):
        SSQCrawler(cfg, s).fetch_page(1)
    assert len(s.calls) == 2

def test_fetch_all_stops_on_short_page(monkeypatch):
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.page_size = 2; cfg.rate_limit_qps = 0.0
    s = FakeSession([FakeResp(200, payload("2026122", "2026121")),
                     FakeResp(200, payload("2026120"))])
    pages = list(SSQCrawler(cfg, s).fetch_all())
    assert [len(p) for p in pages] == [2, 1]

def test_fetch_since_stops_at_known_issue():
    cfg = get_config().crawler.model_copy(deep=True)
    cfg.page_size = 2; cfg.rate_limit_qps = 0.0
    s = FakeSession([FakeResp(200, payload("2026122", "2026121"))])
    out = SSQCrawler(cfg, s).fetch_since("2026121")
    assert [d.issue for d in out] == ["2026122"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_crawler.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`_request(page) -> dict`：`session.get(cfg.api_url, params={"name":"ssq","pageNo":page,"pageSize":cfg.page_size,"systemType":"PC"}, headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout_s)`；`raise_for_status` 后 `json()`；异常与非 200 进入重试，重试间 `time.sleep(backoff_s * 2**attempt)`；耗尽抛 `CrawlError`。每次请求后按 `1/rate_limit_qps` 限速，`rate_limit_qps <= 0` 时跳过。`fetch_all` 逐页 yield，页长 `< page_size` 或为空即停，并受 `max_pages` 限制。`fetch_since` 逐页收集，遇到 `issue <= since` 即截断返回。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_crawler.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/data/crawler.py tests/unit/test_crawler.py
git commit -m "feat: 福彩接口分页采集，带重试、限速与显式失败

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: 只读仓库与刷新

**Files:**
- Create: `src/lottery/data/repository.py`
- Test: `tests/unit/test_repository.py`

**Interfaces:**
- Consumes: `DrawStorage`、`DrawCleaner`、`SSQCrawler`
- Produces: `DrawRepository(storage: DrawStorage, cleaner: DrawCleaner, crawler: SSQCrawler)`
- Produces: `refresh() -> CleanResult`
- Produces: `all_draws() -> list[DrawRecord]`
- Produces: `latest_issue() -> str | None`
- Produces: `slice(start_issue: str, end_issue: str) -> list[DrawRecord]`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_repository.py
from datetime import date
import pandas as pd
from lottery.common.types import DrawRecord
from lottery.data.cleaner import DrawCleaner
from lottery.data.repository import DrawRepository
from lottery.data.storage import DrawStorage, COLUMNS

def rec(i): return DrawRecord(i, date(2026, 10, 25), (1, 2, 3, 4, 5, 6), 9)

class FakeCrawler:
    def __init__(self, full=(), since=()): self.full, self.since_out = list(full), list(since)
    def fetch_all(self, max_pages=None): yield self.full
    def fetch_since(self, issue): return self.since_out

def repo(tmp_path, crawler):
    return DrawRepository(DrawStorage(tmp_path / "x.csv"), DrawCleaner(), crawler)

def test_refresh_full_when_empty(tmp_path):
    r = repo(tmp_path, FakeCrawler(full=[rec("2026002"), rec("2026001")]))
    res = r.refresh()
    assert [d.issue for d in r.all_draws()] == ["2026001", "2026002"]
    assert res.ok is True

def test_refresh_incremental_appends(tmp_path):
    storage = DrawStorage(tmp_path / "x.csv")
    DrawRepository(storage, DrawCleaner(), FakeCrawler(full=[rec("2026001")])).refresh()
    r2 = DrawRepository(storage, DrawCleaner(), FakeCrawler(since=[rec("2026002")]))
    r2.refresh()
    assert r2.latest_issue() == "2026002"
    assert len(r2.all_draws()) == 2

def test_failed_full_crawl_leaves_csv_untouched(tmp_path):
    class Boom:
        def fetch_all(self, max_pages=None): raise RuntimeError("boom"); yield
        def fetch_since(self, issue): raise RuntimeError("boom")
    r = repo(tmp_path, Boom())
    try:
        r.refresh()
    except RuntimeError:
        pass
    assert not (tmp_path / "x.csv").exists()

def test_slice_is_inclusive(tmp_path):
    r = repo(tmp_path, FakeCrawler(full=[rec("2026001"), rec("2026002"), rec("2026003")]))
    r.refresh()
    assert [d.issue for d in r.slice("2026002", "2026003")] == ["2026002", "2026003"]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_repository.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`refresh`：先 `storage.load()`；为空则把 `fetch_all()` 全部页收集到 list，全部成功后 `storage.rewrite(cleaner.clean(df).frame)`；非空则 `fetch_since(latest_issue())`，有新记录才 `storage.append`；最后返回对当前全量数据清洗的 `CleanResult`。`all_draws` 从 `load()` 经 `clean` 后逐行构造 `DrawRecord`。`slice` 在 `all_draws` 上按 issue 闭区间过滤。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_repository.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/data/repository.py tests/unit/test_repository.py
git commit -m "feat: 只读仓库支持全量/增量刷新，失败不留半份数据

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: 频率与遗漏特征

**Files:**
- Create: `src/lottery/teaching/__init__.py`
- Create: `src/lottery/teaching/features.py`
- Test: `tests/unit/test_features.py`

**Interfaces:**
- Consumes: `DrawRecord`、`RED_MIN`、`RED_MAX`
- Produces: `FeatureBuilder()`；`build(draws: Sequence[DrawRecord], window: int) -> pd.DataFrame`，单行，列 `freq_1..freq_33`、`omission_1..omission_33`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_features.py
from datetime import date
from lottery.common.types import DrawRecord
from lottery.teaching.features import FeatureBuilder

def rec(i, reds): return DrawRecord(f"20260{i:02d}", date(2026, 10, i), reds, 9)

DRAWS = [rec(1, (1, 2, 3, 4, 5, 6)),
         rec(2, (1, 2, 3, 4, 5, 7)),
         rec(3, (8, 9, 10, 11, 12, 13))]

def test_frequency_counts_within_window():
    f = FeatureBuilder().build(DRAWS, 10)
    assert f.loc[0, "freq_1"] == 2 and f.loc[0, "freq_7"] == 1 and f.loc[0, "freq_8"] == 1

def test_omission_zero_when_in_latest():
    f = FeatureBuilder().build(DRAWS, 10)
    assert f.loc[0, "omission_8"] == 0

def test_omission_counts_draws_since_last_seen():
    f = FeatureBuilder().build(DRAWS, 10)
    assert f.loc[0, "omission_1"] == 1
    assert f.loc[0, "omission_20"] == 10   # 窗口内从未出现，封顶为 window

def test_window_limits_history():
    f = FeatureBuilder().build(DRAWS, 1)
    assert f.loc[0, "freq_1"] == 0 and f.loc[0, "freq_8"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_features.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`build(draws, window)` 取 `draws[-window:]`，按号码统计频次；遗漏从窗口末尾向前扫描，首次出现位置距末尾的期数即遗漏值，未出现取 `window`。列名固定为 `freq_{n}` / `omission_{n}`（n 不加前导零）。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_features.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/teaching/__init__.py src/lottery/teaching/features.py tests/unit/test_features.py
git commit -m "feat: 窗口内红球频率与遗漏特征

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: 打分、选池与生成注

**Files:**
- Create: `src/lottery/teaching/pipeline.py`
- Test: `tests/unit/test_pipeline.py`

**Interfaces:**
- Consumes: `FeatureBuilder`、`Ticket`、`TicketPool`、`Recommendation`、`RecommendOptions`
- Produces: `red_scores(features: pd.DataFrame, freq_weight: float, omission_weight: float) -> pd.Series`（index = 1..33）
- Produces: `select_pool(scores: pd.Series, size: int) -> list[int]`
- Produces: `TicketSampler(pool: Sequence[int], red_scores: pd.Series, blue_weights: pd.Series)`；`sample(count: int, rng: np.random.Generator) -> tuple[Ticket, ...]`
- Produces: `recommend(draws: Sequence[DrawRecord], cfg: AppConfig, options: RecommendOptions) -> Recommendation`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_pipeline.py
import numpy as np, pandas as pd, pytest
from datetime import date
from lottery.common.config import get_config
from lottery.common.types import DrawRecord, RecommendOptions
from lottery.teaching.features import FeatureBuilder
from lottery.teaching.pipeline import TicketSampler, red_scores, select_pool, recommend

def rec(i, reds, blue=9): return DrawRecord(f"20260{i:02d}", date(2026, 10, i), reds, blue)
DRAWS = [rec(1, (1, 2, 3, 4, 5, 6), 9), rec(2, (1, 2, 3, 4, 5, 7), 10),
         rec(3, (1, 2, 8, 9, 10, 11), 9)]

def feats():
    return FeatureBuilder().build(DRAWS, 3)

def test_red_scores_are_zscore_combination():
    s = red_scores(feats(), 1.0, 0.0)
    assert s.index.min() == 1 and s.index.max() == 33
    assert s[1] == pytest.approx((3 - s.mean()) / s.std(ddof=0))

def test_select_pool_is_deterministic_on_ties():
    s = pd.Series({i: float(i % 2) for i in range(1, 34)})
    pool = select_pool(s, 12)
    assert pool[:2] == [1, 3]   # 同分时号码小的优先

def test_tickets_valid_and_counted():
    rng = np.random.default_rng(7)
    s = red_scores(feats(), 1.0, 1.0)
    pool = select_pool(s, 12)
    sampler = TicketSampler(pool, s, pd.Series(1.0, index=range(1, 17)))
    tickets = sampler.sample(5, rng)
    assert len(tickets) == 5
    for t in tickets:
        assert len(t.reds) == 6 and list(t.reds) == sorted(t.reds)
        assert set(t.reds) <= set(pool) and 1 <= t.blue <= 16

def test_same_seed_reproduces():
    opts = RecommendOptions(count=5, pool_size=12, window=3, seed=42)
    a = recommend(DRAWS, get_config(), opts)
    b = recommend(DRAWS, get_config(), opts)
    assert a == b

def test_different_seed_changes_tickets():
    a = recommend(DRAWS, get_config(), RecommendOptions(5, 12, 3, 42))
    b = recommend(DRAWS, get_config(), RecommendOptions(5, 12, 3, 43))
    assert a.pool.tickets != b.pool.tickets

def test_negative_scores_do_not_produce_nan():
    s = pd.Series(np.linspace(-5, -1, 33), index=range(1, 34))
    sampler = TicketSampler(list(range(1, 34)), s,
                            pd.Series(0.0, index=range(1, 17)))
    assert len(sampler.sample(3, np.random.default_rng(1))) == 3
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`red_scores`：`(freq - mean)/std` 与 `(omission - mean)/std`（`ddof=0`；`std == 0` 时该项全 0）按权重相加。
`select_pool`：`sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:size]`。
`TicketSampler`：构造时保存池、红球分数、蓝球权重；`sample` 内权重 `_softmax(scores - max)`，每注 `rng.choice(pool, 6, replace=False, p=w)` 后排序；重复整注则重抽，最多 1000 次，仍重复则抛 `RuntimeError`。蓝球 `_softmax` 后 `rng.choice(1..16, p=...)`；权重全 0 时用均匀分布。
`recommend`：`FeatureBuilder().build(draws, options.window)` → 打分 → 选池 → `TicketSampler` → `TicketPool` → `Recommendation(latest_issue=draws[-1].issue, n_draws=len(draws))`。`rng = np.random.default_rng(options.seed)`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/teaching/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: 统计打分选池并按 seed 可复现生成投注

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: 风险声明与命令行

**Files:**
- Create: `src/lottery/common/disclaimer.py`
- Create: `src/lottery/cli/__init__.py`
- Create: `src/lottery/cli/recommend.py`
- Test: `tests/unit/test_disclaimer.py`
- Test: `tests/integration/test_recommend_smoke.py`

**Interfaces:**
- Consumes: `get_config`、`DrawRepository`、`DrawStorage`、`DrawCleaner`、`SSQCrawler`、`recommend`
- Produces: `render_banner(text: str, stream: TextIO) -> None`
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Produces: `build_parser() -> argparse.ArgumentParser`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_disclaimer.py
import io
from lottery.common.disclaimer import render_banner

def test_banner_contains_text():
    out = io.StringIO()
    render_banner("风险提示", out)
    assert "风险提示" in out.getvalue()
```

```python
# tests/integration/test_recommend_smoke.py
from datetime import date
import pytest
from lottery.cli.recommend import main
from lottery.common.types import DrawRecord
from lottery.data.storage import DrawStorage

@pytest.fixture
def seeded_csv(tmp_path, monkeypatch):
    path = tmp_path / "ssq_history.csv"
    records = [DrawRecord(f"2026{i:03d}", date(2026, 1, min(i, 28)),
                          (1, 2, 3, 4, 5, 6), i % 16 + 1) for i in range(1, 21)]
    DrawStorage(path).rewrite(_to_frame(records))
    return path

def _to_frame(records):
    import pandas as pd
    from lottery.data.storage import COLUMNS
    rows = [{"issue": r.issue, "draw_date": str(r.draw_date),
             **{f"red{j+1}": v for j, v in enumerate(r.reds)}, "blue": r.blue}
            for r in records]
    return pd.DataFrame(rows, columns=COLUMNS)

def test_offline_recommend_prints_banner_and_tickets(seeded_csv, capsys, monkeypatch):
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: seeded_csv)
    code = main(["--offline", "--count", "3", "--window", "20"])
    out = capsys.readouterr().out
    assert code == 0
    assert "风险提示" in out and "本次推荐 3 注" in out
    assert out.count("红: ") == 3

def test_empty_csv_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("lottery.cli.recommend._default_storage_path", lambda: tmp_path / "none.csv")
    assert main(["--offline"]) == 1
    assert "没有可用历史数据" in capsys.readouterr().err

@pytest.mark.parametrize("argv", [["--count", "0"], ["--pool-size", "5"], ["--pool-size", "34"], ["--window", "0"]])
def test_invalid_args_rejected(argv, capsys):
    assert main(argv) == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_disclaimer.py tests/integration/test_recommend_smoke.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现**

`render_banner`：`sys.stdout.isatty()` 为真且非 Windows 无 ANSI 风险时用 `\033[91m` 包裹，否则纯文本上下边框；始终带 `cfg.project.disclaimers.global_banner` 原文。
`build_parser`：`--count`（默认 `recommend.ticket_count`）、`--pool-size`、`--window`、`--seed`、`--refresh`、`--offline`。
`main`：`build_parser()` 后 `try` 包裹 `parse_args` 与参数范围校验，捕获 `SystemExit` 并 `return int(exc.code)`（`parser.error` 抛 2，测试可断言返回值）；`--offline` 且无数据 → stderr `没有可用历史数据` 返回 1；否则按需 `refresh()`，读 `all_draws()`，用 `RecommendOptions(count, pool_size, window, seed)` 调用 `recommend`，打印最新期号、总期数、注数与每注。
`_default_storage_path()` 返回 `PROJECT_ROOT / cfg.paths.raw_csv`，便于测试替换。模块末尾 `if __name__ == "__main__": raise SystemExit(main())`。

- [ ] **Step 4: 运行确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_disclaimer.py tests/integration/test_recommend_smoke.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
git add src/lottery/common/disclaimer.py src/lottery/cli tests/unit/test_disclaimer.py tests/integration/test_recommend_smoke.py
git commit -m "feat: 本期推荐命令行，强制风险横幅与参数校验

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: 全量回归与真机验证

**Files:**
- Modify: `README.md`（若不存在则创建）
- Test: `tests/` 全量

**Interfaces:**
- Consumes: 前 10 个任务的全部接口

- [ ] **Step 1: 跑全量测试**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 2: 真机采集一次（联网）**

Run: `PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --refresh --count 5`
Expected: 退出码 0；输出含最新期号（约 `2026122`）、总期数（约 2080）、5 注号码与风险横幅

- [ ] **Step 3: 断网复跑验证可复现**

Run: `PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --offline --count 5`
Expected: 与上一步注数一致、号码一致（同 seed）

- [ ] **Step 4: README 写入运行方式**

README 只写：安装、运行命令、风险声明原文、`data/raw/ssq_history.csv` 位置。
```bash
git add README.md
git commit -m "docs: 补充本期推荐运行说明

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

- 规格覆盖：数据源、解析、存储、清洗、仓库、特征、打分、选池、生成注、CLI、声明、测试策略均落到任务。
- 偏差：`FeatureBuilder` 只出频率/遗漏、`CleanResult` 增加 `frame` 字段、新增 `Recommendation`，均已在 Task 8/5/1 明确。
- 类型一致：`TicketPool(pool_numbers=...)` 与 `Recommendation.pool` 在 Task 1 定义，Task 9 使用同名。
- 未覆盖：真实网络下的官网结构变更只靠 `ParseError` 暴露，未做 schema 快照；列入后续版本。
