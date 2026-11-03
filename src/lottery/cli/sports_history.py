"""体彩历史数据命令行：拉取大乐透、排列3、排列5并写入本地 CSV。"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lottery.common.config import PROJECT_ROOT, AppConfig, SportsGameConfig, get_config
from lottery.data.crawler import CrawlError
from lottery.data.sporttery import SportteryCrawler, SportsDraw, SportsStorage
from lottery.data.storage import StorageError


def build_parser() -> argparse.ArgumentParser:
    """按配置中的玩法列表构造命令行解析器。"""
    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="sports-history", description="拉取体彩历史开奖数据（大乐透 / 排列3 / 排列5）"
    )
    parser.add_argument(
        "--game",
        choices=[*cfg.sporttery.games, "all"],
        default="all",
        help="玩法代码，默认拉取全部",
    )
    return parser


def _storage_path(game: SportsGameConfig) -> Path:
    """返回玩法配置指定的 CSV 绝对路径。"""
    return PROJECT_ROOT / game.raw_csv


def _run_game(cfg: AppConfig, key: str) -> tuple[SportsGameConfig, int, int, SportsDraw]:
    """采集并存储单个玩法的全部历史，返回配置、拉取数、新增数与最新记录。"""
    game = cfg.sporttery.games[key]
    records = SportteryCrawler(cfg.sporttery).fetch_all(game.game_no)
    if not records:
        raise CrawlError(f"{game.name} 未获取到任何记录")
    added = SportsStorage(_storage_path(game)).append(records)
    return game, len(records), added, records[0]


def _print_game(game: SportsGameConfig, fetched: int, added: int, latest: SportsDraw) -> None:
    """打印单个玩法的拉取结果与最新一期。"""
    print(
        f"{game.name}: 拉取 {fetched} 期，新增 {added} 期，"
        f"最新 {latest.issue} {latest.draw_date}"
    )


def _exit_code(exc: SystemExit) -> int:
    """把 argparse 抛出的 SystemExit 转成整数退出码。"""
    return exc.code if isinstance(exc.code, int) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：逐个玩法采集、存储并打印结果，返回进程退出码。"""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return _exit_code(exc)
    cfg = get_config()
    keys = list(cfg.sporttery.games) if args.game == "all" else [args.game]
    try:
        for key in keys:
            _print_game(*_run_game(cfg, key))
    except (CrawlError, StorageError) as exc:
        print(f"体彩数据拉取失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
