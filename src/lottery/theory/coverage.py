"""覆盖设计构造：把号码池转换成满足“中 N 保 M”的注单集合。

``pool`` 为候选号码池，块大小固定为 6；保证等级决定需要覆盖的 t-子集大小。
构造结果只由号码池、保证等级与算法决定，不读取任何历史数据。
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from lottery.common.constants import PICK_RED, RED_MAX, RED_MIN
from lottery.common.types import Ticket
from lottery.theory.bounds import covering_lower_bound


class Guarantee(StrEnum):
    """覆盖保证：中 6 保 5 / 中 5 保 4 / 中 4 保 3，t 取名称中的覆盖个数。"""

    MATCH6_COVER5 = "match6_cover5"
    MATCH5_COVER4 = "match5_cover4"
    MATCH4_COVER3 = "match4_cover3"

    @property
    def t(self) -> int:
        """返回必须被覆盖的号码子集大小。"""
        return int(self.name[-1])


class CoverageAlgorithm(StrEnum):
    """覆盖构造算法；配置中的 ``table`` 由 CoverageTable 单独处理。"""

    GREEDY = "greedy"
    ILP = "ilp"


@dataclass(frozen=True)
class CoveragePlan:
    """一次覆盖设计的结果，cost 以块数计价，金额换算留给 money 层。"""

    tickets: tuple[Ticket, ...]
    guarantee: Guarantee
    proven_optimal: bool
    lower_bound: int
    cost: float


def _normalize_pool(pool: Sequence[int]) -> tuple[int, ...]:
    """校验号码池：去重后至少 6 个、位于 1..33、不接受重复号码。"""
    numbers = tuple(pool)
    if len(set(numbers)) != len(numbers):
        raise ValueError("号码池不允许重复号码")
    if len(numbers) < PICK_RED:
        raise ValueError(f"号码池至少需要 {PICK_RED} 个号码，实际 {len(numbers)} 个")
    if any(not RED_MIN <= number <= RED_MAX for number in numbers):
        raise ValueError(f"号码池必须位于 {RED_MIN}..{RED_MAX}")
    return tuple(sorted(numbers))


def _target_masks(blocks: Sequence[tuple[int, ...]], t: int) -> tuple[dict[tuple[int, ...], int], list[int]]:
    """为每个块建立“覆盖的 t-子集”位图，返回子集索引与位图列表。"""
    targets = list(dict.fromkeys(sub for block in blocks for sub in itertools.combinations(block, t)))
    index = {target: position for position, target in enumerate(targets)}
    masks = [sum(1 << index[sub] for sub in itertools.combinations(block, t)) for block in blocks]
    return index, masks


def _bits(mask: int) -> list[int]:
    """把位图展开成置位下标列表。"""
    return [position for position in range(mask.bit_length()) if mask >> position & 1]


def _drop_redundant(chosen: list[int], masks: Sequence[int], total: int) -> list[int]:
    """按逆序删除可被其余块完全覆盖的冗余块，减少注数。"""
    counts = [0] * total
    for position in chosen:
        for target in _bits(masks[position]):
            counts[target] += 1
    kept = list(chosen)
    for position in reversed(chosen):
        if all(counts[target] >= 2 for target in _bits(masks[position])):
            for target in _bits(masks[position]):
                counts[target] -= 1
            kept.remove(position)
    return kept


def _greedy_cover(blocks: Sequence[tuple[int, ...]], t: int) -> list[int]:
    """贪心：每步选覆盖未覆盖 t-子集最多的块，再做冗余块删除。"""
    index, masks = _target_masks(blocks, t)
    remaining = (1 << len(index)) - 1
    chosen: list[int] = []
    while remaining:
        best = max(range(len(blocks)), key=lambda position: (masks[position] & remaining).bit_count())
        if masks[best] & remaining == 0:
            raise RuntimeError("贪心无法继续覆盖，号码池数据异常")
        chosen.append(best)
        remaining &= ~masks[best]
    return _drop_redundant(chosen, masks, len(index))


def _ilp_cover(blocks: Sequence[tuple[int, ...]], t: int, time_limit_s: float) -> list[int]:
    """整数规划：最小化块数并强制覆盖每个 t-子集，超时返回当前最优解。"""
    index, _ = _target_masks(blocks, t)
    matrix = np.zeros((len(index), len(blocks)), dtype=np.uint8)
    for column, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            matrix[index[sub], column] = 1
    result = milp(
        c=np.ones(len(blocks)),
        constraints=LinearConstraint(matrix, lb=np.ones(len(index)), ub=np.inf),
        integrality=np.ones(len(blocks)),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit_s},
    )
    if result.x is None:
        raise RuntimeError(f"ILP 未产出可行解: {result.message}")
    return [position for position, value in enumerate(result.x) if value > 0.5]


class CoverageDesigner:
    """按指定算法构造覆盖注单，并给出可证明下界与是否已达下界的标注。"""

    def __init__(self, algorithm: CoverageAlgorithm = CoverageAlgorithm.GREEDY, time_limit_s: float = 60.0) -> None:
        """记录算法与 ILP 求解超时；greedy 不受超时影响。"""
        self.algorithm = algorithm
        self.time_limit_s = time_limit_s

    def build(self, pool: Sequence[int], guarantee: Guarantee) -> CoveragePlan:
        """对号码池构造覆盖方案；块数等于下界时才标记 proven_optimal。"""
        numbers = _normalize_pool(pool)
        blocks = list(itertools.combinations(numbers, PICK_RED))
        chosen = self._select(blocks, guarantee.t)
        lower_bound = covering_lower_bound(len(numbers), PICK_RED, guarantee.t)
        tickets = tuple(Ticket(reds=blocks[position]) for position in sorted(chosen))
        return CoveragePlan(
            tickets=tickets,
            guarantee=guarantee,
            proven_optimal=len(tickets) == lower_bound,
            lower_bound=lower_bound,
            cost=float(len(tickets)),
        )

    def _select(self, blocks: list[tuple[int, ...]], t: int) -> list[int]:
        """按算法选择块下标，两种算法实现同一覆盖契约。"""
        if self.algorithm is CoverageAlgorithm.ILP:
            return _ilp_cover(blocks, t, self.time_limit_s)
        return _greedy_cover(blocks, t)
