"""随机性检验套件：编排六个检验并做 Bonferroni 校正。"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from lottery.common.config import RandomnessConfig
from lottery.common.types import DrawRecord
from lottery.randomness.tests import (
    AutocorrTester,
    ChiSquareTester,
    GeometricTester,
    KSTester,
    PermutationTester,
    RunsTester,
)

_CORRECTIONS = ("bonferroni", "none")
_TESTERS = {
    "chi2": lambda cfg: ChiSquareTester(),
    "ks": lambda cfg: KSTester(),
    "runs": lambda cfg: RunsTester(),
    "autocorr": lambda cfg: AutocorrTester(cfg.autocorr_max_lag),
    "permutation": lambda cfg: PermutationTester(cfg.permutation_trials, cfg.seed),
    "geometric": lambda cfg: GeometricTester(),
}


@dataclass(frozen=True)
class TestResult:
    """单个检验结果；passed 由 p 值与校正后 alpha 派生，不单独赋值。"""

    __test__ = False  # 防止 pytest 把测试结果类型当作测试类收集

    name: str
    statistic: float
    p_value: float
    adjusted_alpha: float
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        """按 Bonferroni 校正后的显著性水平判定是否通过。"""
        object.__setattr__(self, "passed", self.p_value >= self.adjusted_alpha)


@dataclass(frozen=True)
class RandomnessReport:
    """一次随机性检验的完整结果。"""

    results: tuple[TestResult, ...]
    overall_pass: bool
    n_draws: int


class RandomnessSuite:
    """按配置顺序运行启用的检验，并统一换算校正后 alpha。"""

    def __init__(self, cfg: RandomnessConfig) -> None:
        """校验配置并实例化检验；未知名称或非法校正方式立即报错。"""
        if not cfg.test_names:
            raise ValueError("randomness.test_names 不能为空")
        if cfg.correction not in _CORRECTIONS:
            raise ValueError(f"未知校正方式: {cfg.correction}，可选 {_CORRECTIONS}")
        unknown = [name for name in cfg.test_names if name not in _TESTERS]
        if unknown:
            raise ValueError(f"未知检验名称: {unknown}，可选 {sorted(_TESTERS)}")
        self._cfg = cfg
        self._testers = [_TESTERS[name](cfg) for name in cfg.test_names]

    def _adjusted_alpha(self) -> float:
        """按校正方式换算显著性水平：Bonferroni 为 alpha / 检验数。"""
        if self._cfg.correction == "bonferroni":
            return self._cfg.alpha / len(self._testers)
        return self._cfg.alpha

    def run(self, draws: Sequence[DrawRecord]) -> RandomnessReport:
        """运行全部检验，p 值不低于校正后 alpha 即视为通过。"""
        alpha = self._adjusted_alpha()
        results = tuple(self._run_one(tester, draws, alpha) for tester in self._testers)
        return RandomnessReport(
            results=results,
            overall_pass=all(result.passed for result in results),
            n_draws=len(draws),
        )

    @staticmethod
    def _run_one(tester, draws: Sequence[DrawRecord], alpha: float) -> TestResult:
        """运行单个检验并封装为 TestResult。"""
        statistic, p_value = tester.run(draws)
        return TestResult(
            name=tester.name,
            statistic=float(statistic),
            p_value=float(p_value),
            adjusted_alpha=alpha,
        )
