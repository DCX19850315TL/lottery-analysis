"""软分布约束：只对候选注单做接受/拒绝，不改变任何单注中奖概率。"""

import numpy as np
import pandas as pd

from lottery.common.types import TicketPool
from lottery.theory.combinatorics import CombinatoricsEngine, FeatureName


class ConstraintApplier:
    """按理论分布的中心区间过滤候选，压缩极端和值以平滑奖金方差。"""

    def __init__(self, engine: CombinatoricsEngine | None = None) -> None:
        """复用组合引擎，以便按 target 名称计算候选注单的特征值。"""
        self._engine = engine or CombinatoricsEngine()

    def apply(self, pool: TicketPool, target: pd.Series, tol: float) -> TicketPool:
        """保留特征值落在 target 中心 (1-tol) 质量内的注单，其余剔除。"""
        if not 0 < tol <= 1:
            raise ValueError(f"tol 必须位于 (0, 1]，实际 {tol}")
        if not pool.tickets:
            return pool
        lower, upper = self._central_band(target, tol)
        values = self._engine.feature_values(self._as_array(pool))[FeatureName(target.name)]
        kept = tuple(ticket for ticket, value in zip(pool.tickets, values) if lower <= value <= upper)
        if not kept:
            raise ValueError(f"约束过滤后没有剩余注单，请放宽 tol={tol}")
        numbers = tuple(sorted({number for ticket in kept for number in ticket.reds}))
        return TicketPool(tickets=kept, pool_numbers=numbers, seed=pool.seed)

    @staticmethod
    def _central_band(target: pd.Series, tol: float) -> tuple[float, float]:
        """用累积质量定位区间上下界，tol=1 时覆盖全部取值。"""
        weights = target.to_numpy(dtype=float) / float(target.sum())
        cdf = np.cumsum(weights)
        values = target.index.to_numpy()
        lower = values[np.searchsorted(cdf, tol / 2)]
        upper = values[min(np.searchsorted(cdf, 1 - tol / 2), len(values) - 1)]
        return float(lower), float(upper)

    @staticmethod
    def _as_array(pool: TicketPool) -> np.ndarray:
        """把注单转成特征计算所需的 uint8 数组。"""
        return np.array([ticket.reds for ticket in pool.tickets], dtype=np.uint8)
