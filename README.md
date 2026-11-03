# 双色球分析系统

历史数据统计与算法教学演示项目。当前版本提供「本期推荐」命令行：按历史窗口统计红球频率与遗漏特征，打分选池，并在固定随机种子下生成投注号码。

## 风险声明

风险提示：本系统为历史数据统计与算法教学演示系统。彩票每期开奖为独立随机事件，历史结果与未来开奖无因果关系。所有 AI/ML 模型仅做历史数据拟合，无法提升中奖概率；覆盖设计仅改善已命中情形下的奖金回收结构，不改变彩票负期望收益本质。

本系统仅用于教学演示，不构成任何投注建议。

## 安装

要求 Python 3.10 至 3.12（不含 3.13）。

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

说明：

- 本项目未以包形式安装，运行命令必须带 `PYTHONPATH=src`。
- 如默认源无法安装 CPU 版 torch，可单独安装 `.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu`，再安装其余依赖。
- 如需运行测试，再安装开发依赖（pytest 等）：`.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt`
- 运行测试：`.venv\Scripts\python.exe -m pytest -q`（在项目根目录执行）。

## 运行

在项目根目录执行：

```bash
# 首次运行：本地无数据时自动全量采集，然后生成推荐
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --count 5

# 强制联网更新历史数据后再生成推荐
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --count 5 --refresh
```

首次运行无需额外参数：本地 CSV 不存在或为空时会自动联网全量采集。本地已有数据时默认直接读取；`--refresh` 强制联网更新（已有数据时只补抓新期号）；`--offline` 禁止一切网络，本地无数据时直接报错退出。

输出首先打印风险横幅，随后是最新期号、历史期数与推荐号码。

频率与遗漏都是对历史窗口的描述统计量，不是下一期号码的出现概率，也不预测未来。推荐器把两者标准化后加权选池，并对同一期内已使用的蓝球与红球降权，使注单之间尽量少重复；这只改善号码覆盖结构，不改变中奖概率。

大乐透推荐（体彩，读取 `data/raw/dlt_history.csv`）：

```bash
# 首次运行：本地无数据时自动全量采集，然后生成推荐
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.dlt_recommend --count 5

# 强制联网更新历史数据后再生成推荐
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.dlt_recommend --count 5 --refresh
```

输出风险横幅、玩法名、最新期号、历史期数、注数，以及每注前区 5 个号码与后区 2 个号码。`--front-pool`、`--back-pool`、`--window`、`--seed` 覆盖候选池大小、统计窗口与随机种子；默认前区池 12、后区池 6，均由 `config/config.yaml` 的 `recommend` 段提供。`--offline` 与双色球推荐含义相同。

大乐透滚动回测：

```bash
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.dlt_backtest --offline --window 100 --test-periods 100 --count 5
```

只用被预测期之前的数据逐期重跑推荐，输出前区最佳/单注平均命中、后区单注平均命中、理论期望与总投入。当前未配置大乐透奖级奖金，因此不报告奖金与收益率。

理论路线（全程不读取历史数据）：

```bash
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.theory_report --pool-size 12 --guarantee match6_cover5
```

输出理论特征分布、六奖级单注概率与「中N保M」覆盖方案；`--algorithm greedy|ilp|table` 选择构造算法，`--seed` 决定号码池。

常用参数：

- `--refresh`：先联网从福彩官网刷新历史数据并写入本地 CSV，再生成推荐；本地已有数据时只补抓新期号，本地为空时全量采集。
- `--offline`：禁止一切网络，只读本地 CSV；本地无数据时直接报错。
- `--count`：推荐注数，默认来自 `config/config.yaml`。
- `--pool-size`、`--window`、`--seed`：覆盖候选池大小、统计窗口期数与随机种子。

随机种子在 `config/config.yaml` 的 `project.random_seed` 中固定。同一份历史数据、同一组参数下，重复运行输出的号码完全一致。

图形界面（Streamlit，包含教学、数学、数据管理与随机性检验四个页面）：

```bash
PYTHONPATH=src .venv\Scripts\python.exe -m streamlit run src/lottery/ui/app.py
```

体彩历史数据（大乐透 / 排列3 / 排列5，来源体彩官网接口）：

```bash
# 拉取全部三个玩法
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.sports_history

# 只拉取一个玩法：dlt | pl3 | pl5
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.sports_history --game dlt
```

每次运行按官方分页抓取全量记录，按期号幂等追加并排序，重复运行不会产生重复行。输出每个玩法的拉取期数、新增期数与最新一期。

## 数据位置

双色球历史开奖数据存放在 `data/raw/ssq_history.csv`，由 `config/config.yaml` 的 `paths.raw_csv` 指定。

体彩历史数据按玩法存放：

- `data/raw/dlt_history.csv`（超级大乐透）
- `data/raw/pl3_history.csv`（排列3）
- `data/raw/pl5_history.csv`（排列5）

体彩 CSV 列为 `issue,draw_date,numbers`，`numbers` 按官方开奖顺序以空格分隔（大乐透 7 个、排列3 3 个、排列5 5 个）。路径与官方 gameNo 由 `config/config.yaml` 的 `sporttery.games` 配置。

双色球与大乐透推荐默认在本地无数据时会自动采集；`--refresh` 会联网更新对应 CSV；`--offline` 只读取本地文件。`data/raw` 已被 `.gitignore` 忽略，仓库不提交任何抓取数据。
