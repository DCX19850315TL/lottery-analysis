"""大乐透前区/后区打分、选池与可复现抽样。"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from lottery.data.sporttery import SportsDraw
from lottery.teaching.scoring import select_pool, softmax, zscore

FRONT_MIN, FRONT_MAX, PICK_FRONT = 1, 35, 5
BACK_MIN, BACK_MAX, PICK_BACK = 1, 12, 2
_MAX_RETRIES = 1000


@dataclass(frozen=True)
class ZoneRule:
    """一个号码区的切片位置、选号数与号码范围。"""

    start: int
    stop: int
    pick: int
    minimum: int
    maximum: int


FRONT_RULE = ZoneRule(0, PICK_FRONT, PICK_FRONT, FRONT_MIN, FRONT_MAX)
BACK_RULE = ZoneRule(PICK_FRONT, PICK_FRONT + PICK_BACK, PICK_BACK, BACK_MIN, BACK_MAX)


@dataclass(frozen=True)
class ScoreWeights:
    """频率与遗漏 z 分数的组合权重。"""

    freq: float
    omission: float


@dataclass(frozen=True)
class DltOptions:
    """大乐透推荐参数：注数、前后区池大小、窗口与随机种子。"""

    count: int
    front_pool_size: int
    back_pool_size: int
    window: int
    seed: int

    def __post_init__(self) -> None:
        """构造时复用范围与组合容量校验。"""
        _validate_options(self)


@dataclass(frozen=True)
class DltTicket:
    """一注大乐透号码，前后区均升序排列。"""

    fronts: tuple[int, ...]
    backs: tuple[int, ...]

    def __post_init__(self) -> None:
        """校验前后区数量、范围、升序与去重。"""
        _validate_zone(self.fronts, FRONT_RULE)
        _validate_zone(self.backs, BACK_RULE)


@dataclass(frozen=True)
class DltRecommendation:
    """一次大乐透推荐结果及其号码池。"""

    tickets: tuple[DltTicket, ...]
    latest_issue: str
    n_draws: int
    window: int
    front_pool: tuple[int, ...]
    back_pool: tuple[int, ...]


@dataclass(frozen=True)
class _DltPlan:
    """抽样所需的号码池与对应分数。"""

    front_pool: tuple[int, ...]
    back_pool: tuple[int, ...]
    front_scores: pd.Series
    back_scores: pd.Series


def _validate_zone(values: tuple[int, ...], rule: ZoneRule) -> None:
    """校验一个号码区：数量、范围、严格升序且不重复。"""
    if len(values) != rule.pick:
        raise ValueError(f"大乐透号码区必须恰好 {rule.pick} 个，实际 {len(values)} 个")
    if any(not rule.minimum <= number <= rule.maximum for number in values):
        raise ValueError(f"大乐透号码必须位于 {rule.minimum}..{rule.maximum}")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("大乐透号码必须严格升序且不重复")


def validate_dlt_draw(draw: SportsDraw) -> None:
    """校验每期恰好 7 个号码，且前后区均合法。"""
    if len(draw.numbers) != PICK_FRONT + PICK_BACK:
        raise ValueError(
            f"大乐透每期必须 {PICK_FRONT + PICK_BACK} 个号码，实际 {len(draw.numbers)} 个"
        )
    _validate_zone(draw.numbers[FRONT_RULE.start : FRONT_RULE.stop], FRONT_RULE)
    _validate_zone(draw.numbers[BACK_RULE.start : BACK_RULE.stop], BACK_RULE)




class ZoneScorer:
    """按窗口统计前后区频率与遗漏，并合并为号码分数。"""

    def __init__(self, window: int, weights: ScoreWeights) -> None:
        """保存窗口与权重；窗口必须为正数。"""
        if window < 1:
            raise ValueError("统计窗口必须 >= 1")
        self._window = window
        self._weights = weights

    def score(self, draws: Sequence[SportsDraw]) -> tuple[pd.Series, pd.Series]:
        """返回前区 1..35 与后区 1..12 的分数序列。"""
        if not draws:
            raise ValueError("大乐透历史数据不能为空")
        recent = list(draws[-self._window :])
        for draw in recent:
            validate_dlt_draw(draw)
        return self._score(recent, FRONT_RULE), self._score(recent, BACK_RULE)

    def _score(self, draws: list[SportsDraw], rule: ZoneRule) -> pd.Series:
        """计算一个号码区的频率与遗漏加权 z 分数。"""
        counts = Counter(
            number for draw in draws for number in draw.numbers[rule.start : rule.stop]
        )
        numbers = range(rule.minimum, rule.maximum + 1)
        freq = pd.Series({number: float(counts[number]) for number in numbers}, dtype=float)
        omission = pd.Series(
            {number: float(self._omission(draws, number, rule)) for number in numbers},
            dtype=float,
        )
        return self._weights.freq * zscore(freq) + self._weights.omission * zscore(omission)

    def _omission(self, draws: list[SportsDraw], number: int, rule: ZoneRule) -> int:
        """返回号码距窗口末尾的期数；窗口内从未出现时取窗口长度。"""
        for distance, draw in enumerate(reversed(draws)):
            if number in draw.numbers[rule.start : rule.stop]:
                return distance
        return self._window


class DltSampler:
    """在前后区候选池内按分数概率生成互不重复的投注。"""

    def __init__(self, plan: _DltPlan) -> None:
        """保存候选池并预计算前后区抽样概率。"""
        self._fronts = np.asarray(plan.front_pool, dtype=int)
        self._backs = np.asarray(plan.back_pool, dtype=int)
        self._front_weights = softmax(plan.front_scores.loc[list(plan.front_pool)])
        self._back_weights = softmax(plan.back_scores.loc[list(plan.back_pool)])
        self._capacity = comb(len(self._fronts), PICK_FRONT) * comb(len(self._backs), PICK_BACK)
        self._max_retries = max(1, min(_MAX_RETRIES, self._capacity))

    def sample(self, count: int, rng: np.random.Generator) -> tuple[DltTicket, ...]:
        """生成 count 注互不重复的号码；超出组合容量时直接报错。"""
        if count > self._capacity:
            raise ValueError(f"候选池最多支持 {self._capacity} 注，实际请求 {count} 注")
        tickets: list[DltTicket] = []
        seen: set[DltTicket] = set()
        while len(tickets) < count:
            ticket = self._draw_unique(rng, seen)
            seen.add(ticket)
            tickets.append(ticket)
        return tuple(tickets)

    def _draw_unique(self, rng: np.random.Generator, seen: set[DltTicket]) -> DltTicket:
        """抽出一注未出现过的号码，达到容量感知的重抽上限时抛 RuntimeError。"""
        for _ in range(self._max_retries):
            ticket = self._draw_one(rng)
            if ticket not in seen:
                return ticket
        raise RuntimeError(f"重抽 {self._max_retries} 次仍未得到不重复的号码")

    def _draw_one(self, rng: np.random.Generator) -> DltTicket:
        """按权重抽 5 个前区号码与 2 个后区号码，分别升序排列。"""
        fronts = rng.choice(self._fronts, size=PICK_FRONT, replace=False, p=self._front_weights)
        backs = rng.choice(self._backs, size=PICK_BACK, replace=False, p=self._back_weights)
        return DltTicket(
            fronts=tuple(sorted(int(number) for number in fronts)),
            backs=tuple(sorted(int(number) for number in backs)),
        )


def _validate_options(options: DltOptions) -> None:
    """校验注数、前后区池大小与窗口范围。"""
    if options.count < 1:
        raise ValueError("推荐注数必须 >= 1")
    if not PICK_FRONT <= options.front_pool_size <= FRONT_MAX:
        raise ValueError(f"前区池大小必须在 {PICK_FRONT}..{FRONT_MAX}")
    if not PICK_BACK <= options.back_pool_size <= BACK_MAX:
        raise ValueError(f"后区池大小必须在 {PICK_BACK}..{BACK_MAX}")
    if options.window < 1:
        raise ValueError("统计窗口必须 >= 1")
    capacity = comb(options.front_pool_size, PICK_FRONT) * comb(
        options.back_pool_size, PICK_BACK
    )
    if options.count > capacity:
        raise ValueError(f"候选池最多支持 {capacity} 注，实际请求 {options.count} 注")


def recommend_dlt(
    draws: Sequence[SportsDraw], options: DltOptions, weights: ScoreWeights
) -> DltRecommendation:
    """按历史窗口打分、选池并生成可复现的大乐透推荐。"""
    _validate_options(options)
    front_scores, back_scores = ZoneScorer(options.window, weights).score(draws)
    front_pool = tuple(select_pool(front_scores, options.front_pool_size, options.seed))
    back_pool = tuple(
        select_pool(back_scores, options.back_pool_size, options.seed + 1)
    )
    plan = _DltPlan(front_pool, back_pool, front_scores, back_scores)
    tickets = DltSampler(plan).sample(options.count, np.random.default_rng(options.seed))
    return DltRecommendation(
        tickets=tickets,
        latest_issue=draws[-1].issue,
        n_draws=len(draws),
        window=options.window,
        front_pool=front_pool,
        back_pool=back_pool,
    )
