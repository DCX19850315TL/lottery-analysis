"""CSV 存储：期号幂等追加与原子重写。"""

import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from lottery.common.types import DrawRecord

COLUMNS = ["issue", "draw_date", "red1", "red2", "red3", "red4", "red5", "red6", "blue"]


def record_to_row(record: DrawRecord) -> dict[str, object]:
    """把一条记录转成 CSV 行，日期存 ISO 字符串，红球展开为 red1..red6。"""
    row: dict[str, object] = {"issue": record.issue, "draw_date": record.draw_date.isoformat()}
    row.update({f"red{index}": red for index, red in enumerate(record.reds, start=1)})
    row["blue"] = record.blue
    return row


class DrawStorage:
    """用 CSV 文件持久化开奖记录。"""

    def __init__(self, path: Path) -> None:
        """记录 CSV 文件路径。"""
        self.path = path

    def load(self) -> pd.DataFrame:
        """读取全部记录；文件不存在时返回同列空表。"""
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path, dtype={"issue": str})[COLUMNS]

    def append(self, records: Sequence[DrawRecord]) -> int:
        """幂等追加新期号记录，按期号升序重写，返回新增行数。"""
        existing = self.load()
        known = set(existing["issue"])
        fresh = [record for record in records if not self._seen(record, known)]
        if not fresh:
            return 0
        rows = pd.DataFrame([self._to_row(record) for record in fresh], columns=COLUMNS)
        merged = pd.concat([existing, rows], ignore_index=True)
        self.rewrite(merged.sort_values("issue", kind="stable").reset_index(drop=True))
        return len(fresh)

    def rewrite(self, df: pd.DataFrame) -> None:
        """原子写回整表：先写临时文件，再替换目标文件，失败时清理临时文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".csv.tmp")
        try:
            df.to_csv(tmp_path, index=False, columns=COLUMNS)
            os.replace(tmp_path, self.path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _seen(record: DrawRecord, known: set[str]) -> bool:
        """判断期号是否已存在，同时把新期号记入集合以过滤批内重复。"""
        if record.issue in known:
            return True
        known.add(record.issue)
        return False

    @staticmethod
    def _to_row(record: DrawRecord) -> dict[str, object]:
        """把一条记录转成 CSV 行，委托给 record_to_row。"""
        return record_to_row(record)
