"""大乐透推荐命令行：读取体彩历史数据并输出注数与号码。"""

import argparse
import sys
from collections.abc import Sequence
from math import comb

from lottery.cli.dlt_data import load_draws_or_report
from lottery.common.config import get_config
from lottery.common.disclaimer import render_banner
from lottery.teaching.dlt import (
    BACK_MAX,
    FRONT_MAX,
    PICK_BACK,
    PICK_FRONT,
    DltOptions,
    DltRecommendation,
    DltTicket,
    ScoreWeights,
    recommend_dlt,
)


def build_parser() -> argparse.ArgumentParser:
    """按配置默认值构造大乐透推荐命令行解析器。"""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="dlt-recommend", description="按历史统计生成大乐透推荐号码（仅教学演示）"
    )
    parser.add_argument("--count", type=int, default=cfg.recommend.ticket_count, help="推荐注数")
    parser.add_argument(
        "--front-pool", type=int, default=cfg.recommend.pool_size, help="前区候选池大小"
    )
    parser.add_argument(
        "--back-pool", type=int, default=cfg.recommend.dlt_back_pool_size, help="后区候选池大小"
    )
    parser.add_argument("--window", type=int, default=cfg.recommend.window, help="统计窗口期数")
    parser.add_argument("--seed", type=int, default=cfg.project.random_seed, help="随机种子")
    parser.add_argument("--refresh", action="store_true", help="先联网刷新历史数据")
    parser.add_argument("--offline", action="store_true", help="只读本地数据，不访问网络")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """校验参数范围与互斥开关，违规时通过 parser.error 退出码 2。"""
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


def _exit_code(exc: SystemExit) -> int:
    """把 argparse 抛出的 SystemExit 转成整数退出码。"""
    return exc.code if isinstance(exc.code, int) else 1


def _format_ticket(ticket: DltTicket) -> str:
    """按两位补零格式输出一注：前区: ... 后区: ...。"""
    fronts = " ".join(f"{number:02d}" for number in ticket.fronts)
    backs = " ".join(f"{number:02d}" for number in ticket.backs)
    return f"前区: {fronts} 后区: {backs}"


def _print_result(result: DltRecommendation) -> None:
    """依次打印最新期号、历史期数、注数与每注号码。"""
    print("超级大乐透")
    print(f"最新期号: {result.latest_issue}")
    print(f"历史期数: {result.n_draws}")
    print(f"本次推荐 {len(result.tickets)} 注")
    for ticket in result.tickets:
        print(_format_ticket(ticket))


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：校验参数、准备数据并打印推荐，返回进程退出码。"""
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
    options = DltOptions(args.count, args.front_pool, args.back_pool, args.window, args.seed)
    try:
        result = recommend_dlt(
            draws,
            options,
            ScoreWeights(cfg.recommend.freq_weight, cfg.recommend.omission_weight),
        )
    except (ValueError, RuntimeError) as exc:
        print(f"推荐失败: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
