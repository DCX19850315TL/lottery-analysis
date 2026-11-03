"""窗口内红球、蓝球的频率与遗漏特征。

遗漏只描述「距最近一次出现的期数」，是历史统计量，不是未来出现概率。
"""

from collections import Counter
from collections.abc import Sequence

import pandas as pd

from lottery.common.constants import BLUE_MAX, BLUE_MIN, RED_MAX, RED_MIN
from lottery.common.types import DrawRecord


class FeatureBuilder:
    """构造单行红球、蓝球频率与遗漏特征。"""

    def build(self, draws: Sequence[DrawRecord], window: int) -> pd.DataFrame:
        """返回最近 window 期特征，单行，红蓝分别带 freq/omission 列。"""
        recent = list(draws[-window:])
        red_counts = Counter(number for draw in recent for number in draw.reds)
        blue_counts = Counter(draw.blue for draw in recent)
        red_sets = [draw.reds for draw in recent]
        blue_sets = [(draw.blue,) for draw in recent]
        row = {}
        for number in range(RED_MIN, RED_MAX + 1):
            row[f"freq_{number}"] = red_counts[number]
            row[f"omission_{number}"] = self._omission(red_sets, number, window)
        for number in range(BLUE_MIN, BLUE_MAX + 1):
            row[f"blue_freq_{number}"] = blue_counts[number]
            row[f"blue_omission_{number}"] = self._omission(blue_sets, number, window)
        return pd.DataFrame([row])

    @staticmethod
    def _omission(draws: list[tuple[int, ...]], number: int, window: int) -> int:
        """返回号码距窗口末尾的期数；从未出现时取 window。仅作历史统计量。"""
        for distance, numbers in enumerate(reversed(draws)):
            if number in numbers:
                return distance
        return window
