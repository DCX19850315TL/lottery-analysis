"""回测命令行：两条路线共用评分规则，并与随机基准做显著性检验。"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lottery.backtest.math_engine import MathBacktestEngine
from lottery.backtest.metrics import BacktestResult, ScoreRules
from lottery.backtest.monte_carlo import MonteCarloBaseline
from lottery.backtest.significance import SignificanceResult, SignificanceTester
from lottery.backtest.teaching_engine import TeachingBacktestEngine
from lottery.common.config import PROJECT_ROOT, AppConfig, get_config
from lottery.common.constants import PICK_RED, RED_MAX, RED_MIN
from lottery.common.disclaimer import render_banner
from lottery.common.types import DrawRecord
from lottery.data.cleaner import DrawCleaner
from lottery.data.crawler import CrawlError, SSQCrawler
from lottery.data.parser import ParseError
from lottery.data.repository import DrawRepository
from lottery.data.storage import DrawStorage, StorageError
from lottery.theory.coverage import (
    CoverageAlgorithm,
    CoverageDesigner,
    CoveragePlan,
    Guarantee,
)
from lottery.theory.prize_table import PrizeLevel

_ROUTES = ("teaching", "math", "both")
_ALGORITHMS = [algorithm.value for algorithm in CoverageAlgorithm] + ["table"]
_PRIZE_FIELDS = {
    PrizeLevel.FIRST: "first",
    PrizeLevel.SECOND: "second",
    PrizeLevel.THIRD: "third",
    PrizeLevel.FOURTH: "fourth",
    PrizeLevel.FIFTH: "fifth",
    PrizeLevel.SIXTH: "sixth",
}


@dataclass(frozen=True)
class RouteReport:
    """单条路线的回测结果与随机基准显著性结论。"""

    result: BacktestResult
    significance: SignificanceResult


@dataclass(frozen=True)
class _Runner:
    """把配置、命令行参数与历史数据固定后，按路线产出回测报告。"""

    cfg: AppConfig
    args: argparse.Namespace
    draws: Sequence[DrawRecord]

    def teaching(self) -> RouteReport:
        """滚动窗口回测教学路线。"""
        result = TeachingBacktestEngine(self.cfg, _score_rules(self.cfg)).run(
            self.draws, "rule", self.args.window, self.args.test_periods
        )
        return self._compare(result, self.cfg.recommend.ticket_count, self.cfg.recommend.pool_size)

    def math(self) -> RouteReport:
        """对固定覆盖方案回测数学路线。"""
        plan = _build_plan(self.cfg, self.args)
        result = MathBacktestEngine(_score_rules(self.cfg)).run(
            plan, self.draws, self.args.test_periods
        )
        return self._compare(result, len(plan.tickets), self.args.pool_size)

    def _compare(self, result: BacktestResult, n_tickets: int, pool_size: int) -> RouteReport:
        """在相同区间上跑同池随机基准并做卡方检验。"""
        window = list(self.draws[-self.args.test_periods :])
        baseline = MonteCarloBaseline(
            n_tickets, _score_rules(self.cfg), pool_size
        ).run(window, self.args.trials, self.cfg.project.random_seed)
        return RouteReport(result, SignificanceTester().compare(result, baseline))


def _score_rules(cfg: AppConfig) -> ScoreRules:
    """按配置的名义奖金构造评分规则。"""
    payouts = {
        level: getattr(cfg.lottery.prizes, field) for level, field in _PRIZE_FIELDS.items()
    }
    return ScoreRules(ticket_price=cfg.lottery.ticket_price, payouts=payouts)


def _select_pool(pool_size: int, seed: int) -> tuple[int, ...]:
    """按理论均匀原则选取号码池，结果只由 seed 决定。"""
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.arange(RED_MIN, RED_MAX + 1), size=pool_size, replace=False)
    return tuple(sorted(int(number) for number in picked))


def _build_plan(cfg: AppConfig, args: argparse.Namespace) -> CoveragePlan:
    """按配置算法构造覆盖方案；table 只用于查表，构造降级为 greedy。"""
    algorithm = (
        CoverageAlgorithm.ILP if args.algorithm == CoverageAlgorithm.ILP else CoverageAlgorithm.GREEDY
    )
    plan = CoverageDesigner(algorithm, cfg.math.solver_timeout_s)
    return plan.build(_select_pool(args.pool_size, args.seed), Guarantee(args.guarantee))


def build_parser() -> argparse.ArgumentParser:
    """按配置默认值构造回测命令解析器。"""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="lottery-backtest", description="回测教学与数学两条路线并与随机基准比较"
    )
    parser.add_argument("--route", choices=_ROUTES, default="both")
    parser.add_argument("--test-periods", type=int, default=cfg.backtest.test_periods)
    parser.add_argument("--window", type=int, default=cfg.recommend.window)
    parser.add_argument("--pool-size", type=int, default=cfg.math.default_pool_size)
    parser.add_argument(
        "--guarantee",
        default=cfg.math.default_guarantee.lower(),
        choices=[guarantee.value for guarantee in Guarantee],
    )
    parser.add_argument("--algorithm", default=cfg.math.coverage_algorithm, choices=_ALGORITHMS)
    parser.add_argument("--trials", type=int, default=cfg.backtest.monte_carlo_trials)
    parser.add_argument("--seed", type=int, default=cfg.project.random_seed)
    parser.add_argument("--refresh", action="store_true", help="先联网刷新历史数据")
    parser.add_argument("--offline", action="store_true", help="只读本地数据，不访问网络")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """校验参数范围与互斥开关，违规时通过 parser.error 退出码 2。"""
    if args.refresh and args.offline:
        parser.error("--refresh 与 --offline 不能同时使用")
    if args.test_periods < 1:
        parser.error("--test-periods 必须 >= 1")
    if args.window < 1:
        parser.error("--window 必须 >= 1")
    if args.trials < 1:
        parser.error("--trials 必须 >= 1")
    if not PICK_RED <= args.pool_size <= RED_MAX:
        parser.error(f"--pool-size 必须在 {PICK_RED}..{RED_MAX}")


def _exit_code(exc: SystemExit) -> int:
    """把 argparse 抛出的 SystemExit 转成整数退出码。"""
    return exc.code if isinstance(exc.code, int) else 1


def _default_storage_path() -> Path:
    """返回默认历史数据 CSV 的绝对路径。"""
    return PROJECT_ROOT / get_config().paths.raw_csv


def _build_repository(cfg: AppConfig) -> DrawRepository:
    """按配置组装存储、清洗器与采集器。"""
    storage = DrawStorage(_default_storage_path())
    return DrawRepository(storage, DrawCleaner(), SSQCrawler(cfg.crawler))


def _load_draws(args: argparse.Namespace, repository: DrawRepository) -> list[DrawRecord]:
    """按需刷新或只读加载历史数据；--offline 永不联网。"""
    is_empty = repository.storage.load().empty
    if args.refresh or (is_empty and not args.offline):
        repository.refresh()
    return repository.all_draws()


def _load_draws_or_report(
    args: argparse.Namespace, repository: DrawRepository
) -> list[DrawRecord] | None:
    """加载历史数据；采集、解析或存储失败时打印一行错误并返回 None。"""
    try:
        return _load_draws(args, repository)
    except (CrawlError, ParseError, StorageError) as exc:
        print(f"数据加载失败: {exc}", file=sys.stderr)
        return None


def _has_enough_history(args: argparse.Namespace, draws: Sequence[DrawRecord]) -> bool:
    """教学路线需要 window + test_periods 期，数学路线只需 test_periods 期。"""
    if args.route in ("teaching", "both"):
        return len(draws) - args.window >= args.test_periods
    return args.test_periods <= len(draws)


def _print_report(report: RouteReport) -> None:
    """打印单条路线的指标与显著性结论。"""
    metrics = report.result.metrics
    print(f"结论: {report.significance.conclusion}")
    print(f"路线: {report.result.label}")
    print(f"  测试期数: {len(report.result.curve)}")
    print(
        f"  单期最佳红球命中: {metrics.mean_max_red_hits:.3f}"
        f"  单注平均: {metrics.mean_red_hits_per_ticket:.3f}"
    )
    print(
        f"  单注期望命中: {metrics.expected_red_hits_per_ticket:.3f}"
        f"  命中率: {metrics.hit_rate:.2%}"
    )
    print(f"  最大空窗: {metrics.max_gap_periods} 期  最大回撤: {metrics.max_drawdown:.2f}")
    print(f"  总投入: {metrics.total_cost:.2f}  总回收: {metrics.total_return:.2f}")
    print(
        f"  收益率: {metrics.roi:.2%}"
        f"  95% 区间: [{metrics.roi_ci_low:.2%}, {metrics.roi_ci_high:.2%}]"
    )


def _run_routes(args: argparse.Namespace, draws: Sequence[DrawRecord]) -> None:
    """按 --route 依次运行路线并打印报告。"""
    cfg = get_config()
    runner = _Runner(cfg, args, draws)
    if args.route in ("teaching", "both"):
        _print_report(runner.teaching())
    if args.route in ("math", "both"):
        _print_report(runner.math())


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：校验参数、加载历史数据并输出回测报告，返回进程退出码。"""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        _validate_args(parser, args)
    except SystemExit as exc:
        return _exit_code(exc)
    cfg = get_config()
    render_banner(cfg.project.disclaimers["global_banner"], sys.stdout)
    draws = _load_draws_or_report(args, _build_repository(cfg))
    if draws is None:
        return 1
    if not draws:
        print("没有可用历史数据", file=sys.stderr)
        return 1
    if not _has_enough_history(args, draws):
        print("历史数据不足：教学路线需要至少 window + test-periods 期", file=sys.stderr)
        return 1
    _run_routes(args, draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
