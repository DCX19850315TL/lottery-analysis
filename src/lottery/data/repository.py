"""只读开奖仓库：全量/增量刷新与按期号闭区间读取。"""

from collections.abc import Sequence
from datetime import date

import pandas as pd

from lottery.common.constants import PICK_RED
from lottery.common.types import DrawRecord
from lottery.data.cleaner import CleanResult, DrawCleaner
from lottery.data.crawler import SSQCrawler
from lottery.data.storage import COLUMNS, DrawStorage, record_to_row


class DrawRepository:
    """基于存储、清洗器与采集器的只读开奖仓库。"""

    def __init__(self, storage: DrawStorage, cleaner: DrawCleaner, crawler: SSQCrawler) -> None:
        """注入存储、清洗器与采集器。"""
        self.storage = storage
        self.cleaner = cleaner
        self.crawler = crawler

    def refresh(self) -> CleanResult:
        """全量或增量刷新仓库；采集或写入失败直接抛出，不留下半份数据。"""
        if self.storage.load().empty:
            return self._refresh_full()
        return self._refresh_incremental()

    def all_draws(self) -> list[DrawRecord]:
        """返回清洗后的全部开奖记录，按期号升序。"""
        frame = self.cleaner.clean(self.storage.load()).frame
        return [self._to_record(row) for _, row in frame.iterrows()]

    def latest_issue(self) -> str | None:
        """返回最新期号；没有任何记录时返回 None。"""
        draws = self.all_draws()
        return draws[-1].issue if draws else None

    def slice(self, start_issue: str, end_issue: str) -> list[DrawRecord]:
        """返回期号落在 [start_issue, end_issue] 闭区间内的记录。"""
        return [draw for draw in self.all_draws() if start_issue <= draw.issue <= end_issue]

    def _refresh_full(self) -> CleanResult:
        """空仓库时先收齐全量页，全部成功后再原子重写，返回清洗结果。"""
        records = [record for page in self.crawler.fetch_all() for record in page]
        if not records:
            return self.cleaner.clean(self.storage.load())
        result = self.cleaner.clean(self._to_frame(records))
        self.storage.rewrite(result.frame)
        return result

    def _refresh_incremental(self) -> CleanResult:
        """已有数据时只追加新期号，返回刷新后全量数据的清洗结果。"""
        latest = self.latest_issue()
        fresh = self.crawler.fetch_since(latest) if latest is not None else []
        if fresh:
            self.storage.append(fresh)
        return self.cleaner.clean(self.storage.load())

    @staticmethod
    def _to_frame(records: Sequence[DrawRecord]) -> pd.DataFrame:
        """把记录列表转成列顺序为 COLUMNS 的 DataFrame。"""
        rows = [record_to_row(record) for record in records]
        return pd.DataFrame(rows, columns=COLUMNS)

    @staticmethod
    def _to_record(row: pd.Series) -> DrawRecord:
        """把清洗后的一行转成 DrawRecord，非法数据交由类型校验报错。"""
        reds = tuple(int(row[f"red{index}"]) for index in range(1, PICK_RED + 1))
        return DrawRecord(
            str(row["issue"]), date.fromisoformat(row["draw_date"]), reds, int(row["blue"])
        )
