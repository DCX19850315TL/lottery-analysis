"""分层无偏候选池抽样：按理论 SUM 分布分层，不使用任何历史加权。"""

import numpy as np
import pandas as pd

from lottery.common.types import Ticket, TicketPool
from lottery.theory.combinatorics import CombinatoricsEngine, FeatureName


def _allocate(strata: np.ndarray, pool_size: int) -> list[int]:
    """按各层理论质量比例分配注数，余数按小数部分从大到小补齐。"""
    masses = np.bincount(strata)
    exact = masses / masses.sum() * pool_size
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    for position in order[: pool_size - counts.sum()]:
        counts[position] += 1
    return counts.tolist()


class PoolSampler:
    """在全部组合上按 SUM 分位数分层抽样，层内均匀抽取候选注单。"""

    STRATA = 10

    def __init__(self, engine: CombinatoricsEngine | None = None) -> None:
        """复用组合引擎的缓存；分层的层号首次使用时计算并缓存。"""
        self._engine = engine or CombinatoricsEngine()
        self._strata: np.ndarray | None = None

    def _combo_strata(self) -> np.ndarray:
        """返回每个组合所属的 SUM 分位层号，层数不足时自动减少。"""
        if self._strata is None:
            sums = self._engine.feature_values(self._engine.enumerate_all())[FeatureName.SUM]
            self._strata = np.asarray(pd.qcut(sums, self.STRATA, labels=False, duplicates="drop"))
        return self._strata

    def sample(self, pool_size: int, rng: np.random.Generator) -> TicketPool:
        """抽取 pool_size 注候选号码，返回含号码池与随机种子的 TicketPool。"""
        if pool_size < 1:
            raise ValueError(f"注数必须为正，实际 {pool_size}")
        seed = int(rng.integers(0, 2**31 - 1))
        strata = self._combo_strata()
        combos = self._engine.enumerate_all()
        rows = np.concatenate(
            [
                rng.choice(np.flatnonzero(strata == stratum), size=count, replace=False)
                for stratum, count in enumerate(_allocate(strata, pool_size))
            ]
        )
        rng.shuffle(rows)
        tickets = tuple(Ticket(reds=tuple(int(number) for number in combos[row])) for row in rows)
        pool_numbers = tuple(sorted({number for ticket in tickets for number in ticket.reds}))
        return TicketPool(tickets=tickets, pool_numbers=pool_numbers, seed=seed)
