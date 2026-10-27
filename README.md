# 双色球分析系统

历史数据统计与算法教学演示项目。当前版本提供「本期推荐」命令行：按历史窗口统计红球频率与遗漏特征，打分选池，并在固定随机种子下生成投注号码。

## 风险声明

风险提示：本系统为历史数据统计与算法教学演示系统。双色球每期开奖为独立随机事件，历史结果与未来开奖无因果关系。所有 AI/ML 模型仅做历史数据拟合，无法提升中奖概率；覆盖设计仅改善已命中情形下的奖金回收结构，不改变彩票负期望收益本质。

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
# 首次运行：联网刷新历史数据后再生成推荐
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --count 5 --refresh

# 之后本地已有数据，可省略 --refresh
PYTHONPATH=src .venv\Scripts\python.exe -m lottery.cli.recommend --count 5
```

首次运行必须加 `--refresh`；后续运行可省略，或用 `--offline` 强制只读本地。

输出首先打印风险横幅，随后是最新期号、历史期数与推荐号码。

常用参数：

- `--refresh`：先联网从福彩官网刷新历史数据并写入本地 CSV，再生成推荐。
- `--offline`：只读本地 CSV，不访问网络；本地无数据时直接报错。
- `--count`：推荐注数，默认来自 `config/config.yaml`。
- `--pool-size`、`--window`、`--seed`：覆盖候选池大小、统计窗口期数与随机种子。

随机种子在 `config/config.yaml` 的 `project.random_seed` 中固定。同一份历史数据、同一组参数下，重复运行输出的号码完全一致。

## 数据位置

历史开奖数据存放在 `data/raw/ssq_history.csv`。

- 该路径由 `config/config.yaml` 的 `paths.raw_csv` 指定。
- `--refresh` 会联网更新此文件；`--offline` 只读取此文件。
- 该目录已被 `.gitignore` 忽略，仓库不提交任何抓取数据。
