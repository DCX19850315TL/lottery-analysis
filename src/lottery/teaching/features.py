"""窗口内红球频率与遗漏特征。"""

from collections import Counter
from collections.abc import Sequence

import pandas as pd

from lottery.common.constants import RED_MAX, RED_MIN
from lottery.common.types import DrawRecord


class FeatureBuilder:
    """构造单行红球频率与遗漏特征。"""

    def build(self, draws: Sequence[DrawRecord], window: int) -> pd.DataFrame:
        """返回最近 window 期的频率与遗漏特征，单行、列 freq_n / omission_n。"""
        recent = list(draws[-window:])
        counts = Counter(number for draw in recent for number in draw.reds)
        row = {}
        for number in range(RED_MIN, RED_MAX + 1):
            row[f"freq_{number}"] = counts[number]
            row[f"omission_{number}"] = self._omission(recent, number, window)
        return pd.DataFrame([row])

    @staticmethod
    def _omission(draws: list[DrawRecord], number: int, window: int) -> int:
        """返回号码距窗口末尾的期数；窗口内从未出现时取 window。"""
        for distance, draw in enumerate(reversed(draws)):
            if number in draw.reds:
                return distance
        return window
