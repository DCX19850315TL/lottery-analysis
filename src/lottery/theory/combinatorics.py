"""组合枚举与理论特征分布（数学最优路线，不读取任何历史数据）。

各特征的取值口径：
- ``SUM``：6 个红球之和，范围 21..183。
- ``ODD_EVEN``：奇数红球个数，范围 0..6。
- ``ZONE3``：三区（1-11 / 12-22 / 23-33）分布模式的整数编码 ``z1*100 + z2*10 + z3``。
- ``CONSECUTIVE_RUNS``：最长连号段长度，无连号时为 1。
- ``DISCRETE_ENTROPY``：三区归一化分布的香农熵（自然对数），范围 0..ln3。
"""

import itertools
from enum import StrEnum

import numpy as np
import pandas as pd

from lottery.common.constants import PICK_RED, RED_MAX, RED_MIN

ZONE_COUNT = 3
ZONE_WEIGHTS = np.array([100, 10, 1], dtype=np.int16)


class FeatureName(StrEnum):
    """理论特征名称。"""

    SUM = "sum"
    ODD_EVEN = "odd_even"
    ZONE3 = "zone3"
    CONSECUTIVE_RUNS = "consecutive_runs"
    DISCRETE_ENTROPY = "discrete_entropy"


def _zone_counts(combos: np.ndarray, zone_width: int) -> np.ndarray:
    """按三区统计每个组合各区内的号码个数，返回 (n, 3) 数组。"""
    zones = (combos.astype(np.int16) - RED_MIN) // zone_width
    return np.stack([(zones == index).sum(axis=1) for index in range(ZONE_COUNT)], axis=1)


def _entropy(counts: np.ndarray, pick: int) -> np.ndarray:
    """计算三区分布归一化后的香农熵；零概率区贡献 0。"""
    probs = counts / pick
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, probs * np.log(probs), 0.0)
    return -terms.sum(axis=1)


def _longest_run(combos: np.ndarray) -> np.ndarray:
    """返回每个组合的最长连号段长度，逐列累积相邻差为 1 的段长。"""
    runs = np.ones(combos.shape, dtype=np.uint8)
    for column in range(1, combos.shape[1]):
        joined = combos[:, column] - combos[:, column - 1] == 1
        runs[:, column] = np.where(joined, runs[:, column - 1] + 1, 1)
    return runs.max(axis=1)


class CombinatoricsEngine:
    """枚举全部红球组合并计算其理论特征值。"""

    def __init__(self, red_max: int = RED_MAX, pick: int = PICK_RED) -> None:
        """记录号码上限与选号个数，所有组合首次枚举后缓存。"""
        self.red_max = red_max
        self.pick = pick
        self._zone_width = red_max // ZONE_COUNT
        self._cache: np.ndarray | None = None

    def enumerate_all(self) -> np.ndarray:
        """返回全部 C(red_max, pick) 个升序组合，形状 (n, pick)，dtype uint8。"""
        if self._cache is None:
            combos = itertools.combinations(range(RED_MIN, self.red_max + 1), self.pick)
            self._cache = np.array(list(combos), dtype=np.uint8)
        return self._cache

    def feature_values(self, combos: np.ndarray) -> dict[FeatureName, np.ndarray]:
        """返回每个组合在各特征上的取值，键为特征名，值为等长数组。"""
        counts = _zone_counts(combos, self._zone_width)
        return {
            FeatureName.SUM: combos.sum(axis=1, dtype=np.int16),
            FeatureName.ODD_EVEN: (combos % 2).sum(axis=1, dtype=np.int16),
            FeatureName.ZONE3: counts @ ZONE_WEIGHTS,
            FeatureName.CONSECUTIVE_RUNS: _longest_run(combos),
            FeatureName.DISCRETE_ENTROPY: _entropy(counts, self.pick),
        }

    def distribution(self, feature: FeatureName) -> pd.Series:
        """返回全部组合上该特征的取值计数，索引升序，值为组合数。"""
        values = self.feature_values(self.enumerate_all())[feature]
        unique, counts = np.unique(values, return_counts=True)
        return pd.Series(counts, index=unique, name=str(feature))
