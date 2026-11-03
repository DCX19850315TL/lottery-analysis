"""大乐透命令行共用的历史数据加载：刷新、只读与单行错误处理。"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

from lottery.common.config import PROJECT_ROOT, AppConfig
from lottery.data.crawler import CrawlError
from lottery.data.sporttery import SportsDraw, SportsStorage, SportteryCrawler
from lottery.data.storage import StorageError
from lottery.teaching.dlt import validate_dlt_draw


def storage_path(cfg: AppConfig) -> Path:
    """返回大乐透 CSV 的绝对路径。"""
    return PROJECT_ROOT / cfg.sporttery.games["dlt"].raw_csv


def refresh(cfg: AppConfig, storage: SportsStorage) -> None:
    """联网抓取大乐透全量历史并幂等写入本地 CSV。"""
    game = cfg.sporttery.games["dlt"]
    records = SportteryCrawler(cfg.sporttery).fetch_all(game.game_no)
    if not records:
        raise CrawlError(f"{game.name} 未获取到任何记录")
    storage.append(records)


def to_draws(df: pd.DataFrame) -> list[SportsDraw]:
    """把 CSV 数据表转换为按文件顺序排列的开奖记录。"""
    return [_to_draw(row) for row in df.itertuples(index=False)]


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


def load_draws(args, cfg: AppConfig) -> list[SportsDraw]:
    """按需刷新或只读加载大乐透历史；--offline 永不联网。"""
    storage = SportsStorage(storage_path(cfg))
    df = storage.load()
    if args.refresh or (df.empty and not args.offline):
        refresh(cfg, storage)
        df = storage.load()
    return to_draws(df)


def load_draws_or_report(args, cfg: AppConfig) -> list[SportsDraw] | None:
    """加载历史数据；失败时打印一行错误并返回 None。"""
    try:
        return load_draws(args, cfg)
    except (CrawlError, StorageError, ValueError) as exc:
        print(f"数据加载失败: {exc}", file=sys.stderr)
        return None
