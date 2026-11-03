"""大乐透教学路线滚动回测：训练窗口严格截止到被预测期的前一期。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lottery.common.config import AppConfig
from lottery.data.sporttery import SportsDraw
from lottery.teaching.dlt import (
    BACK_MAX,
    BACK_MIN,
    FRONT_MAX,
    FRONT_MIN,
    PICK_BACK,
    PICK_FRONT,
    DltOptions,
    DltTicket,
    ScoreWeights,
    recommend_dlt,
)

_FRONT_COUNT = FRONT_MAX - FRONT_MIN + 1
_BACK_COUNT = BACK_MAX - BACK_MIN + 1


@dataclass(frozen=True)
class DltBacktestOptions:
    """大乐透回测参数：窗口、测试期数、候选池、注数、种子与单注价格。"""

    count: int
    front_pool_size: int
    back_pool_size: int
    window: int
    test_periods: int
    seed: int
    ticket_price: float

    def __post_init__(self) -> None:
        """复用推荐参数校验，并校验测试期数与单注价格。"""
        DltOptions(
            self.count,
            self.front_pool_size,
            self.back_pool_size,
            self.window,
            self.seed,
        )
        if self.test_periods < 1:
            raise ValueError("回测期数必须 >= 1")
        if self.ticket_price <= 0:
            raise ValueError(f"单注价格必须为正，实际 {self.ticket_price}")


@dataclass(frozen=True)
class DltPeriodScore:
    """单期评分：前区最佳/全部命中、后区命中总数与成本。"""

    issue: str
    max_front_hits: int
    front_hit_total: int
    back_hit_total: int
    cost: float


@dataclass(frozen=True)
class DltMetrics:
    """大乐透回测汇总；命中按单注平均报告，另给单期最佳前区命中。"""

    mean_max_front_hits: float
    mean_front_hits_per_ticket: float
    mean_back_hits_per_ticket: float
    expected_front_hits_per_ticket: float
    expected_back_hits_per_ticket: float
    total_cost: float


@dataclass(frozen=True)
class DltBacktestResult:
    """一次大乐透回测的指标与逐期曲线。"""

    metrics: DltMetrics
    curve: pd.DataFrame
    label: str


class DltBacktestEngine:
    """按滚动窗口逐期重跑大乐透推荐，并对下一期真实开奖评分。"""

    def __init__(self, cfg: AppConfig) -> None:
        """保存应用配置，评分权重来自 recommend 段。"""
        self._cfg = cfg

    def run(
        self, draws: Sequence[SportsDraw], options: DltBacktestOptions
    ) -> DltBacktestResult:
        """只使用 [i-window, i) 的数据预测第 i 期；窗口不足时报错。"""
        maximum = len(draws) - options.window
        if not 1 <= options.test_periods <= maximum:
            raise ValueError(f"回测期数必须在 1..{maximum}，实际 {options.test_periods}")
        start = len(draws) - options.test_periods
        scores = tuple(
            self._score_period(draws, index, options) for index in range(start, len(draws))
        )
        return DltBacktestResult(
            metrics=_metrics(scores, options.count),
            curve=_curve_frame(scores),
            label="dlt:teaching",
        )

    def _score_period(
        self, draws: Sequence[SportsDraw], index: int, options: DltBacktestOptions
    ) -> DltPeriodScore:
        """用截止到 index 之前的历史推荐，再对 draws[index] 评分。"""
        history = draws[index - options.window : index]
        rec_options = DltOptions(
            options.count,
            options.front_pool_size,
            options.back_pool_size,
            options.window,
            options.seed + index,
        )
        recommendation = recommend_dlt(
            history,
            rec_options,
            ScoreWeights(
                self._cfg.recommend.freq_weight,
                self._cfg.recommend.omission_weight,
            ),
        )
        return _score_tickets(recommendation.tickets, draws[index], options.ticket_price)


def _score_tickets(
    tickets: Sequence[DltTicket], draw: SportsDraw, ticket_price: float
) -> DltPeriodScore:
    """统计当期全部注单的前后区命中数与成本。"""
    fronts = set(draw.numbers[:PICK_FRONT])
    backs = set(draw.numbers[PICK_FRONT:])
    hits = [(len(set(ticket.fronts) & fronts), len(set(ticket.backs) & backs)) for ticket in tickets]
    return DltPeriodScore(
        issue=draw.issue,
        max_front_hits=max(front for front, _ in hits),
        front_hit_total=sum(front for front, _ in hits),
        back_hit_total=sum(back for _, back in hits),
        cost=len(tickets) * ticket_price,
    )


def _metrics(scores: Sequence[DltPeriodScore], n_tickets: int) -> DltMetrics:
    """聚合逐期评分；期望命中按单注超几何理论值计算。"""
    if not scores:
        raise ValueError("回测期数不能为空")
    periods = len(scores)
    return DltMetrics(
        mean_max_front_hits=float(np.mean([score.max_front_hits for score in scores])),
        mean_front_hits_per_ticket=float(
            sum(score.front_hit_total for score in scores) / (periods * n_tickets)
        ),
        mean_back_hits_per_ticket=float(
            sum(score.back_hit_total for score in scores) / (periods * n_tickets)
        ),
        expected_front_hits_per_ticket=PICK_FRONT * PICK_FRONT / _FRONT_COUNT,
        expected_back_hits_per_ticket=PICK_BACK * PICK_BACK / _BACK_COUNT,
        total_cost=float(sum(score.cost for score in scores)),
    )


def _curve_frame(scores: Sequence[DltPeriodScore]) -> pd.DataFrame:
    """把逐期评分整理成回测曲线表。"""
    return pd.DataFrame(
        {
            "issue": [score.issue for score in scores],
            "max_front_hits": [score.max_front_hits for score in scores],
            "front_hit_total": [score.front_hit_total for score in scores],
            "back_hit_total": [score.back_hit_total for score in scores],
            "cost": [score.cost for score in scores],
        }
    )
