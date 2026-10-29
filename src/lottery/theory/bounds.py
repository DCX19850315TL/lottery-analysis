"""覆盖数下界：球覆盖界与 Schönheim 递归界。

下界只依赖组合恒等式，不引用任何历史数据或未经验证的文献表值。
"""

import math


def validate_covering_params(v: int, k: int, t: int) -> None:
    """校验覆盖参数：块大小为正、号码数不少于块大小、0 <= t <= k。"""
    if k <= 0:
        raise ValueError(f"块大小 k 必须为正，实际 {k}")
    if v < k:
        raise ValueError(f"号码数 v={v} 不能小于块大小 k={k}")
    if not 0 <= t <= k:
        raise ValueError(f"t 必须位于 0..{k}，实际 {t}")


def sphere_covering_bound(v: int, k: int, t: int) -> int:
    """球覆盖界 ceil(C(v,t) / C(k,t))：每个块最多覆盖 C(k,t) 个 t-子集。"""
    validate_covering_params(v, k, t)
    return math.ceil(math.comb(v, t) / math.comb(k, t))


def schoenheim_bound(v: int, k: int, t: int) -> int:
    """Schönheim 递归界 ceil(v/k * L(v-1,k-1,t-1))，边界 L(v,k,1)=ceil(v/k)。

    t=k 时每个 k-子集必须独占一块，故 L(v,k,k)=C(v,k) 为精确值而非 1。
    """
    validate_covering_params(v, k, t)
    if t == 0:
        return 1
    if t == k:
        return math.comb(v, k)
    if t == 1:
        return math.ceil(v / k)
    inner = schoenheim_bound(v - 1, k - 1, t - 1)
    return math.ceil(v * inner / k)


def covering_lower_bound(v: int, k: int, t: int) -> int:
    """两条下界取较大值，作为覆盖数 C(v,k,t) 的可证明下界。"""
    return max(sphere_covering_bound(v, k, t), schoenheim_bound(v, k, t))
