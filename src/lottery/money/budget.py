"""预算按期等额拆分：固定单期等额投注，余数留作未使用资金。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetPlan:
    """预算拆分结果：每期金额、每期注数与不可整除余数。"""

    per_period: float
    n_tickets_per_period: int
    remainder: float


class BudgetPlanner:
    """把总预算平均拆到 periods 期，并按单注价格向下取整注数。"""

    def plan(self, total_budget: float, ticket_price: float, periods: int) -> BudgetPlan:
        """返回等额投注方案；预算不足一注时直接报错。"""
        if total_budget <= 0:
            raise ValueError(f"总预算必须为正，实际 {total_budget}")
        if ticket_price <= 0:
            raise ValueError(f"单注价格必须为正，实际 {ticket_price}")
        if periods < 1:
            raise ValueError(f"期数必须为正，实际 {periods}")
        per_period = total_budget / periods
        n_tickets = int(per_period // ticket_price)
        if n_tickets < 1:
            raise ValueError(
                f"每期预算 {per_period:.2f} 不足以购买一注（单注 {ticket_price:.2f}）"
            )
        remainder = total_budget - n_tickets * ticket_price * periods
        return BudgetPlan(
            per_period=per_period,
            n_tickets_per_period=n_tickets,
            remainder=remainder,
        )
