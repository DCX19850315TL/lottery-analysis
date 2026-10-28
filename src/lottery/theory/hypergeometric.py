"""超几何计算：红球命中分布与六奖级概率（数学最优路线，不读取历史数据）。"""

from math import comb

from lottery.common.constants import BLUE_MAX, PICK_RED, RED_MAX
from lottery.theory.prize_table import PrizeLevel

RED_HITS = range(PICK_RED + 1)


def _red_hit_probability(hits: int, red_max: int, pick: int) -> float:
    """单注恰好命中 hits 个红球的概率 C(pick,h)·C(N-pick,pick-h)/C(N,pick)。"""
    return (
        comb(pick, hits)
        * comb(red_max - pick, pick - hits)
        / comb(red_max, pick)
    )


class HypergeometricCalculator:
    """按超几何分布给出各奖级的中奖概率与命中数的期望、方差。"""

    def __init__(self, red_max: int = RED_MAX, pick: int = PICK_RED, blue_max: int = BLUE_MAX) -> None:
        """记录红球上限、选号个数与蓝球上限。"""
        self.red_max = red_max
        self.pick = pick
        self.blue_max = blue_max

    def prize_probabilities(self) -> dict[PrizeLevel, float]:
        """返回六个奖级各自的单注中奖概率。"""
        red = {hits: _red_hit_probability(hits, self.red_max, self.pick) for hits in RED_HITS}
        blue = 1.0 / self.blue_max
        miss = 1.0 - blue
        return {
            PrizeLevel.FIRST: red[6] * blue,
            PrizeLevel.SECOND: red[6] * miss,
            PrizeLevel.THIRD: red[5] * blue,
            PrizeLevel.FOURTH: red[5] * miss + red[4] * blue,
            PrizeLevel.FIFTH: red[4] * miss + red[3] * blue,
            PrizeLevel.SIXTH: (red[2] + red[1] + red[0]) * blue,
        }

    def expected_hits(self, n_tickets: int) -> float:
        """返回 n_tickets 注独立彩票的红球期望命中数。"""
        return n_tickets * self.pick * self.pick / self.red_max

    def hits_variance(self, n_tickets: int) -> float:
        """返回 n_tickets 注独立彩票红球命中数的方差（超几何方差 × 注数）。"""
        share = self.pick / self.red_max
        single = self.pick * share * (1 - share) * (self.red_max - self.pick) / (self.red_max - 1)
        return n_tickets * single

    def blue_hit_probability(self) -> float:
        """返回单注蓝球命中概率 1/blue_max。"""
        return 1.0 / self.blue_max
