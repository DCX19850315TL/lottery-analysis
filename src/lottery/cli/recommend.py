"""双色球推荐命令行：强制风险横幅、参数校验与离线/刷新数据读取。"""

import argparse
import sys
from collections.abc import Sequence
from math import comb
from pathlib import Path

from lottery.common.config import PROJECT_ROOT, AppConfig, get_config
from lottery.common.constants import BLUE_MAX, BLUE_MIN, PICK_RED, RED_MAX
from lottery.common.disclaimer import render_banner
from lottery.common.types import DrawRecord, RecommendOptions, Recommendation, Ticket
from lottery.data.cleaner import DrawCleaner
from lottery.data.crawler import CrawlError, SSQCrawler
from lottery.data.parser import ParseError
from lottery.data.repository import DrawRepository
from lottery.data.storage import DrawStorage, StorageError
from lottery.teaching.pipeline import recommend


def build_parser() -> argparse.ArgumentParser:
    """按配置默认值构造命令行解析器。"""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="lottery-recommend", description="按历史统计生成双色球推荐号码（仅教学演示）"
    )
    parser.add_argument("--count", type=int, default=cfg.recommend.ticket_count, help="推荐注数")
    parser.add_argument(
        "--pool-size", type=int, default=cfg.recommend.pool_size, help="候选池大小"
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
    if not PICK_RED <= args.pool_size <= RED_MAX:
        parser.error(f"--pool-size 必须在 {PICK_RED}..{RED_MAX}")
    capacity = comb(args.pool_size, PICK_RED) * (BLUE_MAX - BLUE_MIN + 1)
    if args.count > capacity:
        parser.error(f"--count 超过候选池容量 {capacity}")
    if args.window < 1:
        parser.error("--window 必须 >= 1")


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
    """按需刷新或只读加载历史数据；--offline 永不联网，本地为空也不采集。"""
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


def _format_ticket(ticket: Ticket) -> str:
    """按两位补零格式输出一注：红: 01 ... 蓝: 09。"""
    reds = " ".join(f"{red:02d}" for red in ticket.reds)
    blue = "--" if ticket.blue is None else f"{ticket.blue:02d}"
    return f"红: {reds} 蓝: {blue}"


def _print_result(result: Recommendation) -> None:
    """依次打印最新期号、历史期数、注数与每注号码。"""
    print(f"最新期号: {result.latest_issue}")
    print(f"历史期数: {result.n_draws}")
    print(f"本次推荐 {len(result.pool.tickets)} 注")
    for ticket in result.pool.tickets:
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
    draws = _load_draws_or_report(args, _build_repository(cfg))
    if draws is None:
        return 1
    if not draws:
        print("没有可用历史数据", file=sys.stderr)
        return 1
    options = RecommendOptions(args.count, args.pool_size, args.window, args.seed)
    try:
        result = recommend(draws, cfg, options)
    except (ValueError, RuntimeError) as exc:
        print(f"推荐失败: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
