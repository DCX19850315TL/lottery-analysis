"""大乐透推荐命令行：读取体彩历史数据并输出注数与号码。"""

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from lottery.common.config import PROJECT_ROOT, AppConfig, get_config
from lottery.common.disclaimer import render_banner
from lottery.data.crawler import CrawlError
from lottery.data.sporttery import SportsDraw, SportsStorage, SportteryCrawler
from lottery.data.storage import StorageError
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
    validate_dlt_draw,
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
    parser.add_argument("--back-pool", type=int, default=6, help="后区候选池大小")
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
    if args.window < 1:
        parser.error("--window 必须 >= 1")


def _exit_code(exc: SystemExit) -> int:
    """把 argparse 抛出的 SystemExit 转成整数退出码。"""
    return exc.code if isinstance(exc.code, int) else 1


def _storage_path(cfg: AppConfig) -> Path:
    """返回大乐透 CSV 的绝对路径。"""
    return PROJECT_ROOT / cfg.sporttery.games["dlt"].raw_csv


def _refresh(cfg: AppConfig, storage: SportsStorage) -> None:
    """联网抓取大乐透全量历史并幂等写入本地 CSV。"""
    game = cfg.sporttery.games["dlt"]
    records = SportteryCrawler(cfg.sporttery).fetch_all(game.game_no)
    if not records:
        raise CrawlError(f"{game.name} 未获取到任何记录")
    storage.append(records)


def _to_draw(row: tuple) -> SportsDraw:
    """把一行 CSV 转成 SportsDraw；格式错误转 ValueError。"""
    try:
        numbers = tuple(int(part) for part in str(row.numbers).split())
        draw = SportsDraw(
            issue=str(row.issue),
            draw_date=date.fromisoformat(str(row.draw_date)[:10]),
            numbers=numbers,
        )
        validate_dlt_draw(draw)
        return draw
    except (TypeError, ValueError) as exc:
        raise ValueError(f"体彩数据行无法解析: {row}") from exc


def _to_draws(df: pd.DataFrame) -> list[SportsDraw]:
    """把 CSV 数据表转换为按文件顺序排列的开奖记录。"""
    return [_to_draw(row) for row in df.itertuples(index=False)]


def _load_draws(args: argparse.Namespace, cfg: AppConfig) -> list[SportsDraw]:
    """按需刷新或只读加载大乐透历史；--offline 永不联网。"""
    storage = SportsStorage(_storage_path(cfg))
    df = storage.load()
    if args.refresh or (df.empty and not args.offline):
        _refresh(cfg, storage)
        df = storage.load()
    return _to_draws(df)


def _load_draws_or_report(args: argparse.Namespace, cfg: AppConfig) -> list[SportsDraw] | None:
    """加载历史数据；失败时打印一行错误并返回 None。"""
    try:
        return _load_draws(args, cfg)
    except (CrawlError, StorageError, ValueError) as exc:
        print(f"数据加载失败: {exc}", file=sys.stderr)
        return None


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
    draws = _load_draws_or_report(args, cfg)
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
