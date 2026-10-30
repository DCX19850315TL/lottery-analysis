"""数学路线回测：方案进入本模块前已固定，逐期只读评分。"""

from collections.abc import Sequence

from lottery.backtest.metrics import (
    BacktestResult,
    ScoreRules,
    build_period_score,
    build_result,
)
from lottery.common.types import DrawRecord
from lottery.theory.coverage import CoveragePlan


class MathBacktestEngine:
    """对固定覆盖方案做逐期评分，不调用任何理论生成器。"""

    def __init__(self, rules: ScoreRules) -> None:
        """保存评分规则。"""
        self._rules = rules

    def run(
        self, plan: CoveragePlan, draws: Sequence[DrawRecord], test_periods: int
    ) -> BacktestResult:
        """用最后 test_periods 期数据对方案评分。"""
        if not 1 <= test_periods <= len(draws):
            raise ValueError(f"回测期数必须在 1..{len(draws)}，实际 {test_periods}")
        window = draws[-test_periods:]
        scores = tuple(
            build_period_score(plan.tickets, draw, self._rules) for draw in window
        )
        return build_result(scores, len(plan.tickets), f"math:{plan.guarantee}")
