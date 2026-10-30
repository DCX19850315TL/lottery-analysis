"""回测指标层：单注评分、奖级判定与跨期指标聚合。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lottery.common.types import DrawRecord, Ticket
from lottery.theory.hypergeometric import HypergeometricCalculator
from lottery.theory.prize_table import PrizeLevel


@dataclass(frozen=True)
class ScoreRules:
    """评分规则：单注价格与各奖级名义奖金。"""

    ticket_price: float
    payouts: dict[PrizeLevel, float]

    def __post_init__(self) -> None:
        """校验价格为正值且六个奖级奖金齐全。"""
        if self.ticket_price <= 0:
            raise ValueError(f"单注价格必须为正，实际 {self.ticket_price}")
        missing = [level for level in PrizeLevel if level not in self.payouts]
        if missing:
            raise ValueError(f"缺少奖级奖金: {missing}")


@dataclass(frozen=True)
class PeriodScore:
    """单期评分结果：最佳红球命中数、蓝球命中注数与当期收支。"""

    issue: str
    red_hits: int
    blue_hits: int
    cost: float
    payout: float


@dataclass(frozen=True)
class Metrics:
    """回测汇总指标。"""

    mean_red_hits: float
    mean_blue_hits: float
    hit_rate: float
    expected_hits: float
    max_gap_periods: int
    max_drawdown: float
    total_cost: float
    total_return: float


@dataclass(frozen=True)
class BacktestResult:
    """一次回测的指标、逐期曲线与红球命中分布。"""

    metrics: Metrics
    curve: pd.DataFrame
    red_hit_counts: tuple[int, ...]
    label: str


def score_ticket(ticket: Ticket, draw: DrawRecord) -> tuple[int, bool]:
    """返回单注对单期的红球命中数与蓝球是否命中。"""
    red_hits = len(set(ticket.reds) & set(draw.reds))
    return red_hits, ticket.blue == draw.blue


def prize_level(red_hits: int, blue_hit: bool) -> PrizeLevel | None:
    """按双色球规则把红球命中数与蓝球命中映射到奖级。"""
    if red_hits == 6:
        return PrizeLevel.FIRST if blue_hit else PrizeLevel.SECOND
    if red_hits == 5:
        return PrizeLevel.THIRD if blue_hit else PrizeLevel.FOURTH
    if red_hits == 4:
        return PrizeLevel.FOURTH if blue_hit else PrizeLevel.FIFTH
    if red_hits == 3:
        return PrizeLevel.FIFTH if blue_hit else None
    return PrizeLevel.SIXTH if blue_hit else None


def _ticket_result(ticket: Ticket, draw: DrawRecord) -> tuple[int, bool, PrizeLevel | None]:
    """返回单注的红球命中数、蓝球命中与奖级。"""
    red_hits, blue_hit = score_ticket(ticket, draw)
    return red_hits, blue_hit, prize_level(red_hits, blue_hit)


def _sum_payouts(tickets: Sequence[Ticket], draw: DrawRecord, rules: ScoreRules) -> float:
    """累加当期全部注单的奖金。"""
    return float(
        sum(
            rules.payouts[level] if level is not None else 0.0
            for _, _, level in (_ticket_result(ticket, draw) for ticket in tickets)
        )
    )


def build_period_score(
    tickets: Sequence[Ticket], draw: DrawRecord, rules: ScoreRules
) -> PeriodScore:
    """评分一期的整组注单：红球取最佳单注，蓝球取命中注数。"""
    if not tickets:
        raise ValueError("当期注单不能为空")
    results = [_ticket_result(ticket, draw) for ticket in tickets]
    return PeriodScore(
        issue=draw.issue,
        red_hits=max(red_hits for red_hits, _, _ in results),
        blue_hits=sum(blue_hit for _, blue_hit, _ in results),
        cost=len(tickets) * rules.ticket_price,
        payout=_sum_payouts(tickets, draw, rules),
    )


def _max_drawdown(scores: Sequence[PeriodScore]) -> float:
    """以 0 为初始峰值计算累计净收益的最大回撤。"""
    net = np.array([score.payout - score.cost for score in scores])
    cumulative = np.concatenate(([0.0], np.cumsum(net)))
    peak = np.maximum.accumulate(cumulative)
    return float(np.max(peak - cumulative))


def _max_gap(scores: Sequence[PeriodScore]) -> int:
    """返回连续无奖期数的最长记录。"""
    longest = current = 0
    for score in scores:
        current = current + 1 if score.payout <= 0 else 0
        longest = max(longest, current)
    return longest


def compute_metrics(
    scores: Sequence[PeriodScore], expected_per_ticket: float, n_tickets: int
) -> Metrics:
    """把逐期评分聚合为 8 项指标；空序列直接报错。"""
    if not scores:
        raise ValueError("回测期数不能为空")
    payouts = np.array([score.payout for score in scores])
    return Metrics(
        mean_red_hits=float(np.mean([score.red_hits for score in scores])),
        mean_blue_hits=float(np.mean([score.blue_hits for score in scores])),
        hit_rate=float(np.mean(payouts > 0)),
        expected_hits=expected_per_ticket * n_tickets,
        max_gap_periods=_max_gap(scores),
        max_drawdown=_max_drawdown(scores),
        total_cost=float(sum(score.cost for score in scores)),
        total_return=float(payouts.sum()),
    )


def _curve_frame(scores: Sequence[PeriodScore]) -> pd.DataFrame:
    """把逐期评分整理成回测曲线表。"""
    frame = pd.DataFrame(
        {
            "issue": [score.issue for score in scores],
            "red_hits": [score.red_hits for score in scores],
            "blue_hits": [score.blue_hits for score in scores],
            "cost": [score.cost for score in scores],
            "payout": [score.payout for score in scores],
        }
    )
    frame["net"] = frame["payout"] - frame["cost"]
    frame["cumulative_return"] = frame["net"].cumsum()
    return frame


def build_result(
    scores: Sequence[PeriodScore], n_tickets: int, label: str
) -> BacktestResult:
    """构造回测结果；期望命中数按超几何理论值计算。"""
    expected_per_ticket = HypergeometricCalculator().expected_hits(1)
    return BacktestResult(
        metrics=compute_metrics(scores, expected_per_ticket, n_tickets),
        curve=_curve_frame(scores),
        red_hit_counts=tuple(score.red_hits for score in scores),
        label=label,
    )
