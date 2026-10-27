"""打分、选池与按 seed 可复现地生成投注。"""

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from lottery.common.config import AppConfig
from lottery.common.constants import BLUE_MAX, BLUE_MIN, PICK_RED, RED_MAX, RED_MIN
from lottery.common.types import (
    DrawRecord,
    RecommendOptions,
    Recommendation,
    Ticket,
    TicketPool,
)
from lottery.teaching.features import FeatureBuilder

_MAX_RETRIES = 1000


def _zscore(series: pd.Series) -> pd.Series:
    """返回 z 分数；标准差为 0 时返回全 0。"""
    std = series.std(ddof=0)
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def red_scores(features: pd.DataFrame, freq_weight: float, omission_weight: float) -> pd.Series:
    """按权重合并频率与遗漏的 z 分数，返回 index 为 1..33 的分数。"""
    numbers = range(RED_MIN, RED_MAX + 1)
    freq = pd.Series({n: features.loc[0, f"freq_{n}"] for n in numbers}, dtype=float)
    omission = pd.Series({n: features.loc[0, f"omission_{n}"] for n in numbers}, dtype=float)
    return freq_weight * _zscore(freq) + omission_weight * _zscore(omission)


def select_pool(scores: pd.Series, size: int) -> list[int]:
    """按分数降序选取前 size 个号码，同分时号码小者优先。"""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [int(number) for number, _ in ordered[:size]]


def blue_frequencies(draws: Sequence[DrawRecord], window: int) -> pd.Series:
    """统计最近 window 期蓝球 1..16 的出现次数，返回浮点频数序列。"""
    recent = list(draws[-window:])
    counts = Counter(draw.blue for draw in recent)
    return pd.Series(
        {number: float(counts[number]) for number in range(BLUE_MIN, BLUE_MAX + 1)},
        dtype=float,
    )


def _softmax(weights: pd.Series) -> np.ndarray:
    """把权重转成和为 1 的概率向量；权重全为 0 时结果即均匀分布。"""
    values = np.asarray(weights, dtype=float)
    exp = np.exp(values - values.max())
    return exp / exp.sum()


class TicketSampler:
    """在候选池内按分数概率生成互不重复的号码。"""

    def __init__(
        self, pool: Sequence[int], red_scores: pd.Series, blue_weights: pd.Series
    ) -> None:
        """保存候选池，并按池顺序预计算红球、蓝球的抽样概率。"""
        self._pool = [int(number) for number in pool]
        self._red_weights = _softmax(red_scores.loc[self._pool])
        self._blue_weights = _softmax(blue_weights.loc[range(BLUE_MIN, BLUE_MAX + 1)])

    def sample(self, count: int, rng: np.random.Generator) -> tuple[Ticket, ...]:
        """生成 count 注互不重复的号码；重抽 1000 次仍重复则抛 RuntimeError。"""
        tickets: list[Ticket] = []
        seen: set[Ticket] = set()
        while len(tickets) < count:
            ticket = self._draw_unique(rng, seen)
            seen.add(ticket)
            tickets.append(ticket)
        return tuple(tickets)

    def _draw_unique(self, rng: np.random.Generator, seen: set[Ticket]) -> Ticket:
        """抽出一注未出现过的号码，达到重抽上限仍重复时抛 RuntimeError。"""
        for _ in range(_MAX_RETRIES):
            ticket = self._draw_one(rng)
            if ticket not in seen:
                return ticket
        raise RuntimeError(f"重抽 {_MAX_RETRIES} 次仍未得到不重复的号码")

    def _draw_one(self, rng: np.random.Generator) -> Ticket:
        """按权重抽 6 个红球与 1 个蓝球，红球升序排列。"""
        reds = rng.choice(self._pool, size=PICK_RED, replace=False, p=self._red_weights)
        blue_numbers = np.arange(BLUE_MIN, BLUE_MAX + 1)
        blue = int(rng.choice(blue_numbers, p=self._blue_weights))
        return Ticket(reds=tuple(sorted(int(red) for red in reds)), blue=blue)


def recommend(
    draws: Sequence[DrawRecord], cfg: AppConfig, options: RecommendOptions
) -> Recommendation:
    """按配置打分、选池并生成可复现的推荐结果。"""
    features = FeatureBuilder().build(draws, options.window)
    scores = red_scores(features, cfg.recommend.freq_weight, cfg.recommend.omission_weight)
    pool = select_pool(scores, options.pool_size)
    blue_weights = blue_frequencies(draws, options.window)
    tickets = TicketSampler(pool, scores, blue_weights).sample(
        options.count, np.random.default_rng(options.seed)
    )
    return Recommendation(
        pool=TicketPool(tickets=tuple(tickets), pool_numbers=tuple(pool), seed=options.seed),
        latest_issue=draws[-1].issue,
        window=options.window,
        n_draws=len(draws),
    )
