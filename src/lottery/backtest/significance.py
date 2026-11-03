"""策略与随机基准的差异检验：只报告显著性，不回写任何参数。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from lottery.backtest.metrics import BacktestResult

_HIT_RANGE = range(7)
_MIN_EXPECTED = 5.0
_NOT_SIGNIFICANT = "策略与随机选号无统计显著差异"
_SIGNIFICANT = "策略与随机选号存在统计显著差异"


@dataclass(frozen=True)
class SignificanceResult:
    """检验结果；is_significant 为 False 时结论必须以无差异开头。"""

    chi2: float
    p_value: float
    effect_size: float
    is_significant: bool
    conclusion: str


def _merge_small_bins(observed: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """反复合并期望频数过小的相邻箱，直到每箱期望 >= 5。"""
    observed = observed.astype(float).tolist()
    expected = expected.astype(float).tolist()
    while len(expected) > 2 and min(expected) < _MIN_EXPECTED:
        position = int(np.argmin(expected))
        neighbour = position - 1 if position == len(expected) - 1 else position + 1
        observed[neighbour] += observed[position]
        expected[neighbour] += expected[position]
        del observed[position]
        del expected[position]
    return np.array(observed), np.array(expected)


def _expected_counts(observed: tuple[int, ...], baseline: pd.Series) -> np.ndarray:
    """按基准分布的红球命中频率换算策略期数的期望频数。"""
    distribution = baseline.value_counts(normalize=True).reindex(_HIT_RANGE, fill_value=0.0)
    return distribution.to_numpy() * len(observed)


class SignificanceTester:
    """卡方检验：策略逐期最佳红球命中分布 vs 随机基准分布。"""

    def __init__(self, alpha: float = 0.05) -> None:
        """保存显著性水平。"""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha 必须在 (0, 1)，实际 {alpha}")
        self._alpha = alpha

    def compare(
        self, strategy: BacktestResult, baseline: pd.DataFrame
    ) -> SignificanceResult:
        """返回卡方统计量、p 值与明示结论。"""
        if not strategy.red_hit_counts:
            raise ValueError("策略回测结果为空")
        if baseline.empty or "red_hits" not in baseline:
            raise ValueError("随机基准必须包含 red_hits 列且非空")
        observed = np.bincount(np.asarray(strategy.red_hit_counts), minlength=len(_HIT_RANGE))
        expected = _expected_counts(strategy.red_hit_counts, baseline["red_hits"])
        observed, expected = _merge_small_bins(observed, expected)
        chi2, p_value = self._chi_square(observed, expected)
        effect_size = self._cramers_v(chi2, observed)
        significant = bool(p_value < self._alpha)
        prefix = _SIGNIFICANT if significant else _NOT_SIGNIFICANT
        return SignificanceResult(
            chi2=chi2,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=significant,
            conclusion=f"{prefix}（p={p_value:.4f}，效应量 V={effect_size:.3f}）",
        )

    @staticmethod
    def _cramers_v(chi2: float, observed: np.ndarray) -> float:
        """返回 Cramér's V；样本为空或只有一箱时返回 0。"""
        total = float(observed.sum())
        bins = len(observed) - 1
        if total <= 0 or bins <= 0:
            return 0.0
        if np.isinf(chi2):
            return 1.0
        return float(np.sqrt(chi2 / (total * bins)))

    @staticmethod
    def _chi_square(observed: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
        """期望总和为 0 或只剩一箱时视为不可检验，返回无差异。"""
        if len(expected) < 2 or expected.sum() == 0:
            return 0.0, 1.0
        if np.any((expected <= 0) & (observed > 0)):
            return float("inf"), 0.0
        result = stats.chisquare(observed, f_exp=expected)
        return float(result.statistic), float(result.pvalue)
