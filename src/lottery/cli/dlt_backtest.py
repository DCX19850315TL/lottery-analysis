"""大乐透回测命令行：滚动窗口重跑推荐并报告前后区命中。"""

import argparse
import sys
from collections.abc import Sequence
from math import comb

from lottery.backtest.dlt_engine import (
    DltBacktestEngine,
    DltBacktestOptions,
    DltBacktestResult,
)
from lottery.cli.dlt_data import load_draws_or_report
from lottery.common.config import get_config
from lottery.common.disclaimer import render_banner
from lottery.teaching.dlt import (
    BACK_MAX,
    FRONT_MAX,
    PICK_BACK,
    PICK_FRONT,
)


def build_parser() -> argparse.ArgumentParser:
    """按配置默认值构造大乐透回测命令解析器。"""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="dlt-backtest", description="滚动窗口回测大乐透教学推荐（仅历史统计演示）"
    )
    parser.add_argument("--count", type=int, default=cfg.recommend.ticket_count, help="每期注数")
    parser.add_argument(
        "--front-pool", type=int, default=cfg.recommend.pool_size, help="前区候选池大小"
    )
    parser.add_argument(
        "--back-pool", type=int, default=cfg.recommend.dlt_back_pool_size, help="后区候选池大小"
    )
    parser.add_argument("--window", type=int, default=cfg.recommend.window, help="训练窗口期数")
    parser.add_argument(
        "--test-periods", type=int, default=cfg.backtest.test_periods, help="测试期数"
    )
    parser.add_argument("--seed", type=int, default=cfg.project.random_seed, help="随机种子")
    parser.add_argument(
        "--ticket-price", type=float, default=cfg.lottery.ticket_price, help="单注价格"
    )
    parser.add_argument("--refresh", action="store_true", help="先联网刷新历史数据")
    parser.add_argument("--offline", action="store_true", help="只读本地数据，不访问网络")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """校验范围与组合容量；违规时通过 parser.error 返回退出码 2。"""
    if args.refresh and args.offline:
        parser.error("--refresh 与 --offline 不能同时使用")
    if args.count < 1:
        parser.error("--count 必须 >= 1")
    if not PICK_FRONT <= args.front_pool <= FRONT_MAX:
        parser.error(f"--front-pool 必须在 {PICK_FRONT}..{FRONT_MAX}")
    if not PICK_BACK <= args.back_pool <= BACK_MAX:
        parser.error(f"--back-pool 必须在 {PICK_BACK}..{BACK_MAX}")
    capacity = comb(args.front_pool, PICK_FRONT) * comb(args.back_pool, PICK_BACK)
    if args.count > capacity:
        parser.error(f"--count 超过候选池容量 {capacity}")
    if args.window < 1:
        parser.error("--window 必须 >= 1")
    if args.test_periods < 1:
        parser.error("--test-periods 必须 >= 1")
    if args.ticket_price <= 0:
        parser.error("--ticket-price 必须为正")


def _exit_code(exc: SystemExit) -> int:
    """把 argparse 抛出的 SystemExit 转成整数退出码。"""
    return exc.code if isinstance(exc.code, int) else 1


def _print_result(result: DltBacktestResult) -> None:
    """打印回测路线、测试期数、命中指标与成本。"""
    metrics = result.metrics
    print("超级大乐透回测")
    print(f"路线: {result.label}")
    print(f"测试期数: {len(result.curve)}")
    print(
        f"单期最佳前区命中: {metrics.mean_max_front_hits:.3f}"
        f"  单注平均: {metrics.mean_front_hits_per_ticket:.3f}"
    )
    print(
        f"前区单注期望命中: {metrics.expected_front_hits_per_ticket:.3f}"
        f"  后区单注平均: {metrics.mean_back_hits_per_ticket:.3f}"
        f"  后区期望: {metrics.expected_back_hits_per_ticket:.3f}"
    )
    print(f"总投入: {metrics.total_cost:.2f}")
    print("奖金与收益率: 未建模（当前无大乐透奖级奖金配置）")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：校验参数、加载历史并输出回测报告，返回退出码。"""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        _validate_args(parser, args)
    except SystemExit as exc:
        return _exit_code(exc)
    cfg = get_config()
    render_banner(cfg.project.disclaimers["global_banner"], sys.stdout)
    draws = load_draws_or_report(args, cfg)
    if draws is None:
        return 1
    if not draws:
        print("没有可用历史数据", file=sys.stderr)
        return 1
    if len(draws) - args.window < args.test_periods:
        print(
            "历史数据不足：需要至少 window + test-periods 期",
            file=sys.stderr,
        )
        return 1
    options = DltBacktestOptions(
        args.count,
        args.front_pool,
        args.back_pool,
        args.window,
        args.test_periods,
        args.seed,
        args.ticket_price,
    )
    try:
        result = DltBacktestEngine(cfg).run(draws, options)
    except (ValueError, RuntimeError) as exc:
        print(f"回测失败: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
