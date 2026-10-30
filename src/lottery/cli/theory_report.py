"""理论路线命令行：输出理论分布、奖级概率与覆盖方案，全程不读取历史数据。"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lottery.common.config import get_config
from lottery.common.constants import PICK_RED, RED_MAX, RED_MIN
from lottery.common.disclaimer import render_banner
from lottery.common.types import MathOptions
from lottery.theory.combinatorics import CombinatoricsEngine, FeatureName
from lottery.theory.coverage import (
    CoverageAlgorithm,
    CoverageDesigner,
    CoveragePlan,
    Guarantee,
)
from lottery.theory.coverage_tables import CoverageTable
from lottery.theory.hypergeometric import HypergeometricCalculator
from lottery.theory.prize_table import PrizeLevel

_ALGORITHMS = [algorithm.value for algorithm in CoverageAlgorithm] + ["table"]


@dataclass(frozen=True)
class MathReport:
    """一次理论路线组装的完整结果：候选号码池、覆盖方案与理论分布。"""

    pool_numbers: tuple[int, ...]
    plan: CoveragePlan
    table_reference: tuple[int, bool] | None
    prize_probabilities: dict[PrizeLevel, float]
    distributions: dict[FeatureName, pd.Series]


def _select_pool(pool_size: int, seed: int) -> tuple[int, ...]:
    """按理论均匀原则选取号码池：每个号码等概率，结果只由 seed 决定。"""
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.arange(RED_MIN, RED_MAX + 1), size=pool_size, replace=False)
    return tuple(sorted(int(number) for number in picked))


def _builder_algorithm(algorithm: str) -> CoverageAlgorithm:
    """把配置算法映射到构造器；table 只做查表标注，构造降级为 greedy。"""
    return CoverageAlgorithm.ILP if algorithm == CoverageAlgorithm.ILP else CoverageAlgorithm.GREEDY


def build_report(options: MathOptions) -> MathReport:
    """串联理论路线：选池、查表、超几何概率、理论分布与覆盖构造。"""
    if not PICK_RED <= options.pool_size <= RED_MAX:
        raise ValueError(f"号码池规模必须位于 {PICK_RED}..{RED_MAX}")
    guarantee = Guarantee(options.guarantee.lower())
    pool_numbers = _select_pool(options.pool_size, options.seed)
    engine = CombinatoricsEngine()
    plan = CoverageDesigner(
        _builder_algorithm(options.algorithm), options.time_limit_s
    ).build(pool_numbers, guarantee)
    return MathReport(
        pool_numbers=pool_numbers,
        plan=plan,
        table_reference=CoverageTable().lookup(len(pool_numbers), PICK_RED, guarantee.t),
        prize_probabilities=HypergeometricCalculator().prize_probabilities(),
        distributions={feature: engine.distribution(feature) for feature in FeatureName},
    )


def build_parser() -> argparse.ArgumentParser:
    """按配置默认值构造理论路线命令解析器。"""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="lottery-theory-report", description="输出双色球理论分布与覆盖方案（不使用历史数据）"
    )
    parser.add_argument("--pool-size", type=int, default=cfg.math.default_pool_size)
    parser.add_argument(
        "--guarantee",
        default=cfg.math.default_guarantee.lower(),
        choices=[guarantee.value for guarantee in Guarantee],
    )
    parser.add_argument(
        "--algorithm", default=cfg.math.coverage_algorithm, choices=_ALGORITHMS
    )
    parser.add_argument("--seed", type=int, default=cfg.project.random_seed)
    parser.add_argument("--time-limit", type=float, default=cfg.math.solver_timeout_s)
    return parser


def _exit_code(exc: SystemExit) -> int:
    """把 argparse 抛出的 SystemExit 转成整数退出码。"""
    return exc.code if isinstance(exc.code, int) else 1


def _guarantee_label(guarantee: Guarantee) -> str:
    """把枚举名转换为中N保M显示文案。"""
    match, cover = guarantee.name.split("_")
    return f"中{match[-1]}保{cover[-1]}"


def _print_distributions(distributions: dict[FeatureName, pd.Series]) -> None:
    """打印每个理论特征的取值个数与理论众数。"""
    print("理论分布:")
    for feature, series in distributions.items():
        print(f"  {feature.value}: {len(series)} 个取值, 众数 {series.idxmax()} 计数 {series.max()}")


def _print_prizes(probabilities: dict[PrizeLevel, float]) -> None:
    """打印六个奖级的单注中奖概率。"""
    print("奖级概率（单注）:")
    for level, probability in probabilities.items():
        print(f"  {level.name}: {probability:.10%}")


def _print_plan(report: MathReport) -> None:
    """打印覆盖方案摘要与每注红球。"""
    plan = report.plan
    reference = "无" if report.table_reference is None else str(report.table_reference[0])
    print(f"号码池: {len(report.pool_numbers)} -> {' '.join(f'{n:02d}' for n in report.pool_numbers)}")
    print(
        f"覆盖方案: {_guarantee_label(plan.guarantee)}  注数: {len(plan.tickets)}"
        f"  下界: {plan.lower_bound}  已达下界: {'是' if plan.proven_optimal else '否'}  查表参考: {reference}"
    )
    for ticket in plan.tickets:
        print(f"红: {' '.join(f'{red:02d}' for red in ticket.reds)}")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：校验参数、组装理论路线并打印报告，返回进程退出码。"""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if not PICK_RED <= args.pool_size <= RED_MAX:
            parser.error(f"--pool-size 必须在 {PICK_RED}..{RED_MAX}")
        if args.time_limit <= 0:
            parser.error("--time-limit 必须 > 0")
    except SystemExit as exc:
        return _exit_code(exc)
    cfg = get_config()
    render_banner(cfg.project.disclaimers["global_banner"], sys.stdout)
    options = MathOptions(
        pool_size=args.pool_size,
        guarantee=args.guarantee,
        algorithm=args.algorithm,
        seed=args.seed,
        time_limit_s=args.time_limit,
    )
    report = build_report(options)
    _print_distributions(report.distributions)
    _print_prizes(report.prize_probabilities)
    _print_plan(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
