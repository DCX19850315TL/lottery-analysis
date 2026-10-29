"""覆盖数查表：仅收录可验证的条目，并标注是否已达理论下界。"""

import math
from collections.abc import Mapping

from lottery.theory.bounds import covering_lower_bound, validate_covering_params


class CoverageTable:
    """预计算最优覆盖数表；未收录的参数组合返回 None，由构造算法兜底。"""

    def __init__(self, entries: Mapping[tuple[int, int, int], int] | None = None) -> None:
        """接收 (v, k, t) -> 块数 的条目，低于下界的条目视为错误直接拒绝。"""
        self._entries: dict[tuple[int, int, int], int] = {}
        for (v, k, t), blocks in (entries or {}).items():
            validate_covering_params(v, k, t)
            bound = covering_lower_bound(v, k, t)
            if blocks < bound:
                raise ValueError(f"条目 {(v, k, t)}={blocks} 低于可证明下界 {bound}")
            self._entries[(v, k, t)] = blocks

    def lookup(self, v: int, k: int, t: int) -> tuple[int, bool] | None:
        """返回 (最优块数, 是否已达下界)；未收录返回 None。"""
        validate_covering_params(v, k, t)
        blocks = self._entries.get((v, k, t))
        if blocks is not None:
            return blocks, blocks == covering_lower_bound(v, k, t)
        if t == 0:
            return 1, True
        if t == k:
            return math.comb(v, k), True
        if t == 1:
            return math.ceil(v / k), True
        return None
