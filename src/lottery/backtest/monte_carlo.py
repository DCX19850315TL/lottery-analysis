"""随机基准回测：每次 trial 均为纯随机购票，不读取任何历史模式。"""

from collections.abc import Sequence
from math import comb

import numpy as np
import pandas as pd

from lottery.backtest.metrics import ScoreRules, build_period_score
from lottery.common.constants import BLUE_MAX, BLUE_MIN, PICK_RED, RED_MAX, RED_MIN
from lottery.common.types import DrawRecord, Ticket

_REDS = np.arange(RED_MIN, RED_MAX + 1)
_BLUES = np.arange(BLUE_MIN, BLUE_MAX + 1)


def _random_ticket(rng: np.random.Generator) -> Ticket:
    """均匀随机生成一注号码。"""
    reds = rng.choice(_REDS, size=PICK_RED, replace=False)
    blue = int(rng.choice(_BLUES))
    return Ticket(reds=tuple(sorted(int(red) for red in reds)), blue=blue)


class MonteCarloBaseline:
    """对同一历史区间反复做纯随机购票，产出逐期基准分布。"""

    def __init__(
        self, n_tickets: int, rules: ScoreRules, pool_size: int | None = None
    ) -> None:
        """保存每期注数、评分规则与可选候选池规模。"""
        if n_tickets < 1:
            raise ValueError(f"每期注数必须为正，实际 {n_tickets}")
        if pool_size is not None and not PICK_RED <= pool_size <= RED_MAX:
            raise ValueError(f"候选池必须在 {PICK_RED}..{RED_MAX}，实际 {pool_size}")
        capacity = comb(pool_size, PICK_RED) if pool_size is not None else n_tickets
        if n_tickets > capacity:
            raise ValueError(f"候选池最多支持 {capacity} 注，实际请求 {n_tickets} 注")
        self._n_tickets = n_tickets
        self._rules = rules
        self._pool_size = pool_size

    def run(
        self, draws: Sequence[DrawRecord], n_trials: int, seed: int
    ) -> pd.DataFrame:
        """返回 (trial, period, issue, red_hits, blue_hits, payout) 长表。"""
        if n_trials < 1:
            raise ValueError(f"trial 次数必须为正，实际 {n_trials}")
        if not draws:
            raise ValueError("回测期数不能为空")
        rng = np.random.default_rng(seed)
        rows = [
            self._score_trial(rng, draws, trial)
            for trial in range(n_trials)
        ]
        return pd.DataFrame([row for trial_rows in rows for row in trial_rows])

    def _score_trial(
        self, rng: np.random.Generator, draws: Sequence[DrawRecord], trial: int
    ) -> list[dict]:
        """完成一次 trial 的全部期数评分。"""
        rows = []
        for period, draw in enumerate(draws):
            tickets = self._draw_tickets(rng)
            score = build_period_score(tickets, draw, self._rules)
            rows.append(
                {
                    "trial": trial,
                    "period": period,
                    "issue": score.issue,
                    "red_hits": score.red_hits,
                    "blue_hits": score.blue_hits,
                    "payout": score.payout,
                }
            )
        return rows

    def _draw_tickets(self, rng: np.random.Generator) -> tuple[Ticket, ...]:
        """生成每期票：固定池模式从同一池抽不重复票，默认独立随机。"""
        if self._pool_size is None:
            return tuple(_random_ticket(rng) for _ in range(self._n_tickets))
        pool = rng.choice(_REDS, size=self._pool_size, replace=False)
        tickets: set[Ticket] = set()
        while len(tickets) < self._n_tickets:
            tickets.add(_random_ticket_from_pool(rng, pool))
        return tuple(tickets)


def _random_ticket_from_pool(rng: np.random.Generator, pool: np.ndarray) -> Ticket:
    """从给定候选池均匀抽一注，蓝球在全范围均匀抽取。"""
    reds = rng.choice(pool, size=PICK_RED, replace=False)
    blue = int(rng.choice(_BLUES))
    return Ticket(reds=tuple(sorted(int(red) for red in reds)), blue=blue)
