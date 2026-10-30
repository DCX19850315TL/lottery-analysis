"""六项随机性检验的独立实现：仅验证前提，结论不得反馈到选号逻辑。"""

from collections.abc import Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from lottery.common.constants import PICK_RED, RED_MAX, RED_MIN
from lottery.common.types import DrawRecord

RED_COUNT = RED_MAX - RED_MIN + 1
MIN_DRAWS = 30  # 卡方 33 类期望频数 >= 5 的最低期数


def _require_draws(draws: Sequence[DrawRecord]) -> None:
    """期数不足时直接报错，避免输出无统计意义的 p 值。"""
    if len(draws) < MIN_DRAWS:
        raise ValueError(f"随机性检验至少需要 {MIN_DRAWS} 期，实际 {len(draws)} 期")


def _presence(draws: Sequence[DrawRecord]) -> np.ndarray:
    """返回 (n_draws, 33) 布尔矩阵：第 j 列表示号码 j+1 是否出现。"""
    matrix = np.zeros((len(draws), RED_COUNT), dtype=bool)
    for row, draw in enumerate(draws):
        matrix[row, np.asarray(draw.reds) - RED_MIN] = True
    return matrix


def _sums(draws: Sequence[DrawRecord]) -> np.ndarray:
    """返回每期红球和值序列。"""
    return np.array([sum(draw.reds) for draw in draws], dtype=float)


class ChiSquareTester:
    """卡方拟合优度：33 个红球各自出现频次 vs 理论 6/33。"""

    name = "chi2"

    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]:
        """返回卡方统计量与 p 值。"""
        _require_draws(draws)
        counts = _presence(draws).sum(axis=0)
        expected = len(draws) * PICK_RED / RED_COUNT
        result = stats.chisquare(counts, f_exp=np.full(RED_COUNT, expected))
        return float(result.statistic), float(result.pvalue)


class KSTester:
    """K-S 均匀性：33 个号码的经验分布 vs 离散均匀分布。

    离散取值不能直接套用连续 K-S，这里按唯一值计算经验 CDF 的上下界，
    p 值取连续 K-S 的保守近似（同号球在一期内负相关，实际更保守）。
    """

    name = "ks"

    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]:
        """返回离散 K-S 统计量与 p 值。"""
        _require_draws(draws)
        flat = np.concatenate([np.asarray(draw.reds) for draw in draws])
        counts = np.bincount(flat, minlength=RED_MAX + 1)[1: RED_MAX + 1]
        cumulative = np.cumsum(counts)
        empirical = cumulative / flat.size
        before_jump = (cumulative - counts) / flat.size
        theoretical = np.arange(1, RED_COUNT + 1) / RED_COUNT
        statistic = max(
            float(np.max(empirical - theoretical)),
            float(np.max(theoretical - 1 / RED_COUNT - before_jump)),
        )
        return statistic, float(stats.kstwo.sf(statistic, flat.size))


class RunsTester:
    """游程检验：和值序列相对中位数的游程数。"""

    name = "runs"

    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]:
        """返回游程数与正态近似 p 值；恰等于中位数的期不参与计号。"""
        _require_draws(draws)
        sums = _sums(draws)
        median = np.median(sums)
        signs = sums[sums != median] > median
        above = int(signs.sum())
        below = len(signs) - above
        if above == 0 or below == 0:
            return 0.0, 1.0
        runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
        total = above + below
        mean = 1 + 2 * above * below / total
        variance = 2 * above * below * (2 * above * below - total) / (total**2 * (total - 1))
        z_score = (runs - mean) / float(np.sqrt(variance))
        return float(runs), float(2 * stats.norm.sf(abs(z_score)))


class AutocorrTester:
    """多阶自相关：每个号码的出现指示序列做 Ljung-Box（滞后 1..max_lag）。"""

    name = "autocorr"

    def __init__(self, max_lag: int) -> None:
        """记录最大滞后阶数。"""
        if max_lag < 1:
            raise ValueError(f"max_lag 必须 >= 1，实际 {max_lag}")
        self._max_lag = max_lag

    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]:
        """返回最显著号码的 Ljung-Box 统计量，p 值按 33 个号码做内部校正。"""
        _require_draws(draws)
        matrix = _presence(draws).astype(float)
        lags = list(range(1, self._max_lag + 1))
        best_statistic, best_p = 0.0, 1.0
        for column in matrix.T:
            if column.std() == 0:
                continue  # 恒出现/恒不出现的号码无自相关可言，交给卡方检验
            table = acorr_ljungbox(column, lags=lags, return_df=True)
            statistic = float(table["lb_stat"].iloc[-1])
            p_value = float(table["lb_pvalue"].iloc[-1])
            if p_value < best_p:
                best_statistic, best_p = statistic, p_value
        return best_statistic, min(1.0, best_p * RED_COUNT)


class PermutationTester:
    """置换检验：打乱期序后比较和值一阶自相关的绝对值。"""

    name = "permutation"

    def __init__(self, trials: int, seed: int) -> None:
        """记录置换次数与随机种子；同一 seed 必然复现同一 p 值。"""
        if trials < 1:
            raise ValueError(f"置换次数必须 >= 1，实际 {trials}")
        self._trials = trials
        self._seed = seed

    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]:
        """返回观测统计量与经验 p 值（含 +1 平滑，避免 p=0）。"""
        _require_draws(draws)
        sums = _sums(draws)
        observed = _lag1_abs(sums)
        rng = np.random.default_rng(self._seed)
        shuffled = sums.copy()
        extreme = 0
        for _ in range(self._trials):
            rng.shuffle(shuffled)
            if _lag1_abs(shuffled) >= observed:
                extreme += 1
        return observed, (extreme + 1) / (self._trials + 1)


def _lag1_abs(values: np.ndarray) -> float:
    """一阶自相关的绝对值；序列无波动时视为 0。"""
    if values.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(values[:-1], values[1:])[0, 1]))


class GeometricTester:
    """几何分布拟合：同一号码相邻两次出现的间隔 vs Geom(6/33)。"""

    name = "geometric"

    def run(self, draws: Sequence[DrawRecord]) -> tuple[float, float]:
        """返回卡方统计量与 p 值；尾部期望频数不足 5 的箱合并。"""
        _require_draws(draws)
        gaps = [
            np.diff(np.flatnonzero(column)) - 1
            for column in _presence(draws).T
            if np.count_nonzero(column) >= 2
        ]
        if not gaps:
            return 0.0, 1.0
        failures = np.concatenate(gaps)
        return _chi_square_geometric(failures)


def _chi_square_geometric(failures: np.ndarray) -> tuple[float, float]:
    """对零起始的几何等待失败数做卡方拟合，尾部箱合并到期望 >= 5。"""
    q = PICK_RED / RED_COUNT
    max_gap = int(failures.max())
    indices = np.arange(max_gap + 1)
    expected = (1 - q) ** indices * q * failures.size
    observed = np.bincount(failures, minlength=max_gap + 1)
    cut = len(expected)
    while cut > 1 and expected[cut - 1] < 5:
        cut -= 1
    if cut < len(expected):
        observed = np.concatenate([observed[:cut], [observed[cut:].sum()]])
        expected = np.concatenate([expected[:cut], [expected[cut:].sum()]])
    if len(expected) < 2:
        return 0.0, 1.0
    expected *= observed.sum() / expected.sum()  # 截断的尾部质量按比例补回
    result = stats.chisquare(observed, f_exp=expected)
    return float(result.statistic), float(result.pvalue)
