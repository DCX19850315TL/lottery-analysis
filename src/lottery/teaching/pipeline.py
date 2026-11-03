"""打分、选池与按 seed 可复现地生成投注。"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import comb

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
from lottery.teaching.scoring import select_pool, softmax, zscore

_MAX_RETRIES = 1000


def red_scores(features: pd.DataFrame, freq_weight: float, omission_weight: float) -> pd.Series:
    """按权重合并频率与遗漏的 z 分数，返回 index 为 1..33 的分数。"""
    numbers = range(RED_MIN, RED_MAX + 1)
    freq = pd.Series({n: features.loc[0, f"freq_{n}"] for n in numbers}, dtype=float)
    omission = pd.Series({n: features.loc[0, f"omission_{n}"] for n in numbers}, dtype=float)
    return freq_weight * zscore(freq) + omission_weight * zscore(omission)


def blue_scores(features: pd.DataFrame, freq_weight: float, omission_weight: float) -> pd.Series:
    """按权重合并蓝球频率与遗漏的 z 分数，返回 index 为 1..16 的分数。"""
    numbers = range(BLUE_MIN, BLUE_MAX + 1)
    freq = pd.Series({n: features.loc[0, f"blue_freq_{n}"] for n in numbers}, dtype=float)
    omission = pd.Series(
        {n: features.loc[0, f"blue_omission_{n}"] for n in numbers}, dtype=float
    )
    return freq_weight * zscore(freq) + omission_weight * zscore(omission)


def blue_frequencies(draws: Sequence[DrawRecord], window: int) -> pd.Series:
    """返回最近 window 期蓝球频数；这是历史计数，不是未来出现概率。"""
    recent = list(draws[-window:])
    counts = Counter(draw.blue for draw in recent)
    return pd.Series(
        {number: float(counts[number]) for number in range(BLUE_MIN, BLUE_MAX + 1)},
        dtype=float,
    )


@dataclass
class _SampleState:
    """抽样过程中已接受票、蓝球使用情况与红球使用次数。"""

    tickets: list[Ticket] = field(default_factory=list)
    seen: set[Ticket] = field(default_factory=set)
    used_blues: set[int] = field(default_factory=set)
    usage: Counter[int] = field(default_factory=Counter)


class TicketSampler:
    """在候选池内按分数概率生成互不重复且覆盖更散的号码。"""

    def __init__(
        self, pool: Sequence[int], red_scores: pd.Series, blue_scores: pd.Series
    ) -> None:
        """保存候选池，按池顺序预计算红球概率，并按蓝球分数预计算概率。"""
        self._pool = [int(number) for number in pool]
        self._red_weights = softmax(red_scores.loc[self._pool])
        self._blue_weights = softmax(blue_scores.loc[range(BLUE_MIN, BLUE_MAX + 1)])
        self._blue_numbers = list(range(BLUE_MIN, BLUE_MAX + 1))
        self._capacity = comb(len(self._pool), PICK_RED) * len(self._blue_numbers)
        self._max_retries = max(1, min(_MAX_RETRIES, self._capacity))

    def sample(self, count: int, rng: np.random.Generator) -> tuple[Ticket, ...]:
        """生成 count 注互不重复的号码；容量不足时立即抛 ValueError。"""
        if count > self._capacity:
            raise ValueError(f"候选池最多支持 {self._capacity} 注，实际请求 {count} 注")
        state = _SampleState()
        while len(state.tickets) < count:
            ticket = self._draw_unique(rng, state)
            state.seen.add(ticket)
            state.tickets.append(ticket)
            state.usage.update(ticket.reds)
            self._remember_blue(state, ticket.blue)
        return tuple(state.tickets)

    @staticmethod
    def _remember_blue(state: _SampleState, blue: int) -> None:
        """记录已用蓝球；16 个用满后清空，允许下一轮重复。"""
        state.used_blues.add(blue)
        if len(state.used_blues) == BLUE_MAX - BLUE_MIN + 1:
            state.used_blues.clear()

    def _draw_unique(self, rng: np.random.Generator, state: _SampleState) -> Ticket:
        """抽出一注未出现过的号码，达到重抽上限仍重复时抛 RuntimeError。"""
        for _ in range(self._max_retries):
            ticket = self._draw_one(rng, state)
            if ticket not in state.seen:
                return ticket
        raise RuntimeError(f"重抽 {self._max_retries} 次仍未得到不重复的号码")

    def _draw_one(self, rng: np.random.Generator, state: _SampleState) -> Ticket:
        """按权重抽 6 个红球与 1 个蓝球；红球对已用号码降权以扩大覆盖。"""
        usage = np.array([1 + state.usage[number] for number in self._pool], dtype=float)
        red_weights = self._red_weights / usage
        red_weights = red_weights / red_weights.sum()
        reds = rng.choice(self._pool, size=PICK_RED, replace=False, p=red_weights)
        blue = self._draw_blue(rng, state.used_blues)
        return Ticket(reds=tuple(sorted(int(red) for red in reds)), blue=blue)

    def _draw_blue(self, rng: np.random.Generator, used: set[int]) -> int:
        """从尚未使用的蓝球中按分数概率抽取；全部用完时退回全范围。"""
        available = [number for number in self._blue_numbers if number not in used]
        if not available:
            available = self._blue_numbers
        indices = [number - BLUE_MIN for number in available]
        weights = self._blue_weights[indices]
        total = weights.sum()
        if total == 0:
            weights = np.full(len(available), 1.0 / len(available))
        else:
            weights = weights / total
        return int(rng.choice(available, p=weights))


def recommend(
    draws: Sequence[DrawRecord], cfg: AppConfig, options: RecommendOptions
) -> Recommendation:
    """按配置打分、选池并生成可复现的推荐结果。"""
    features = FeatureBuilder().build(draws, options.window)
    scores = red_scores(features, cfg.recommend.freq_weight, cfg.recommend.omission_weight)
    pool = select_pool(scores, options.pool_size, options.seed)
    blues = blue_scores(features, cfg.recommend.freq_weight, cfg.recommend.omission_weight)
    tickets = TicketSampler(pool, scores, blues).sample(
        options.count, np.random.default_rng(options.seed)
    )
    return Recommendation(
        pool=TicketPool(tickets=tuple(tickets), pool_numbers=tuple(pool), seed=options.seed),
        latest_issue=draws[-1].issue,
        window=options.window,
        n_draws=len(draws),
    )
