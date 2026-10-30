"""模拟资金曲线与止损信号：触发后只输出停止提示。"""

from dataclasses import dataclass

import pandas as pd

_COLUMNS = ["period", "cost", "payout"]


@dataclass(frozen=True)
class StopSignal:
    """止损检查结果：累计亏损占预算比例与提示文案。"""

    triggered: bool
    loss_ratio: float
    message: str


class Ledger:
    """逐期记录成本与奖金，维护累计净收益曲线。"""

    def __init__(self, total_budget: float) -> None:
        """保存总预算，用于换算亏损比例。"""
        if total_budget <= 0:
            raise ValueError(f"总预算必须为正，实际 {total_budget}")
        self._total_budget = total_budget
        self._rows: list[dict[str, float | int]] = []

    def record(self, period: int, cost: float, payout: float) -> None:
        """追加一期记录；成本与奖金不允许为负。"""
        if period < 1:
            raise ValueError(f"期号必须为正，实际 {period}")
        if cost < 0 or payout < 0:
            raise ValueError(f"成本与奖金不能为负，实际 cost={cost} payout={payout}")
        self._rows.append({"period": period, "cost": cost, "payout": payout})

    def curve(self) -> pd.DataFrame:
        """返回逐期曲线：net 为单期净收益，cumulative_net 为累计净收益。"""
        frame = pd.DataFrame(self._rows, columns=_COLUMNS)
        frame["net"] = frame["payout"] - frame["cost"]
        frame["cumulative_net"] = frame["net"].cumsum()
        return frame

    def check_stop_loss(self, stop_loss_ratio: float) -> StopSignal:
        """累计亏损达到预算指定比例时触发止损，只返回停止提示。"""
        if not 0 < stop_loss_ratio <= 1:
            raise ValueError(f"止损比例必须在 (0, 1]，实际 {stop_loss_ratio}")
        curve = self.curve()
        cumulative = float(curve["cumulative_net"].iloc[-1]) if not curve.empty else 0.0
        loss_ratio = max(0.0, -cumulative) / self._total_budget
        triggered = loss_ratio >= stop_loss_ratio
        if triggered:
            message = (
                f"已触发止损：累计亏损占预算 {loss_ratio:.1%}（阈值 {stop_loss_ratio:.1%}）。"
                "系统仅输出停止提示，不提供继续投注建议。"
            )
        else:
            message = (
                f"未触发止损：累计亏损占预算 {loss_ratio:.1%}（阈值 {stop_loss_ratio:.1%}）。"
            )
        return StopSignal(triggered=triggered, loss_ratio=loss_ratio, message=message)
