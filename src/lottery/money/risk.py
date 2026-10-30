"""风险提示：几何分布极端空窗期，仅用于预期管理，不改变投注策略。"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskWarning:
    """命中概率对应的极端空窗期与预警文案。"""

    hit_prob: float
    extreme_gap: int
    message: str


class RiskAssessor:
    """按几何分布上分位计算极端空窗期。"""

    def __init__(self, confidence: float = 0.99) -> None:
        """保存默认置信度。"""
        if not 0 < confidence < 1:
            raise ValueError(f"置信度必须在 (0, 1)，实际 {confidence}")
        self._confidence = confidence

    def extreme_gap(self, hit_prob: float, confidence: float = 0.99) -> int:
        """返回连续未命中期数的上分位：ceil(log(1-conf) / log(1-p))。"""
        if not 0 < hit_prob < 1:
            raise ValueError(f"命中概率必须在 (0, 1)，实际 {hit_prob}")
        if not 0 < confidence < 1:
            raise ValueError(f"置信度必须在 (0, 1)，实际 {confidence}")
        return math.ceil(math.log(1 - confidence) / math.log(1 - hit_prob))

    def warn_threshold(self, hit_prob: float) -> RiskWarning:
        """按实例置信度生成极端空窗期预警。"""
        gap = self.extreme_gap(hit_prob, self._confidence)
        message = (
            f"单期命中概率 {hit_prob:.4f} 时，连续 {gap} 期未命中仍属正常随机波动；"
            "空窗期不等于趋势反转，不得据此追加投注。"
        )
        return RiskWarning(hit_prob=hit_prob, extreme_gap=gap, message=message)
