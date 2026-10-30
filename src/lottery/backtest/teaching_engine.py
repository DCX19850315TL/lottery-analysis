"""教学路线回测：滚动窗口，训练数据严格截止到被预测期的前一期。"""

from collections.abc import Sequence

from lottery.backtest.metrics import (
    BacktestResult,
    PeriodScore,
    ScoreRules,
    build_period_score,
    build_result,
)
from lottery.common.config import AppConfig
from lottery.common.types import DrawRecord, RecommendOptions
from lottery.teaching.pipeline import recommend

_MODELS = ("rule",)


class TeachingBacktestEngine:
    """按滚动窗口逐期重算推荐，并对下一期真实开奖评分。"""

    def __init__(self, cfg: AppConfig, rules: ScoreRules) -> None:
        """保存应用配置与评分规则。"""
        self._cfg = cfg
        self._rules = rules

    def run(
        self,
        draws: Sequence[DrawRecord],
        model_name: str,
        window: int,
        test_periods: int,
    ) -> BacktestResult:
        """只使用 [i-window, i) 的数据预测第 i 期，逐期评分。"""
        if model_name not in _MODELS:
            raise ValueError(f"未知教学模型: {model_name}，当前可选 {_MODELS}")
        if not 1 <= test_periods <= len(draws) - window:
            raise ValueError(
                f"回测期数必须在 1..{len(draws) - window}，实际 {test_periods}"
            )
        start = len(draws) - test_periods
        scores = tuple(
            self._score_period(draws, index, window) for index in range(start, len(draws))
        )
        count = self._cfg.recommend.ticket_count
        return build_result(scores, count, f"teaching:{model_name}")

    def _score_period(
        self, draws: Sequence[DrawRecord], index: int, window: int
    ) -> PeriodScore:
        """用截止到 index 之前的历史生成推荐，再对 draws[index] 评分。"""
        options = RecommendOptions(
            count=self._cfg.recommend.ticket_count,
            pool_size=self._cfg.recommend.pool_size,
            window=window,
            seed=self._cfg.project.random_seed + index,
        )
        history = draws[index - window : index]
        recommendation = recommend(history, self._cfg, options)
        return build_period_score(recommendation.pool.tickets, draws[index], self._rules)
