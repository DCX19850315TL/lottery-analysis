"""数学最优页：先验覆盖方案、奖级概率与预算风控。全程不读取历史数据。"""

import pandas as pd
import streamlit as st

from lottery.common.config import get_config
from lottery.money.budget import BudgetPlanner
from lottery.money.risk import RiskAssessor
from lottery.theory.coverage import CoverageAlgorithm, CoverageDesigner, Guarantee
from lottery.theory.hypergeometric import HypergeometricCalculator
from lottery.ui.charts import coverage_matrix_plot
from lottery.ui.components import render_risk_warning, render_route_note
from lottery.ui.data import select_pool

cfg = get_config()
st.title("数学最优路线")
render_route_note("math_note")

guarantee_values = [guarantee.value for guarantee in Guarantee]
algorithm_values = [algorithm.value for algorithm in CoverageAlgorithm]
pool_size = st.sidebar.number_input(
    "号码池规模", min_value=6, max_value=18, value=cfg.math.default_pool_size
)
guarantee = st.sidebar.selectbox(
    "覆盖等级",
    guarantee_values,
    index=guarantee_values.index(cfg.math.default_guarantee.lower()),
)
algorithm = st.sidebar.selectbox(
    "构造算法",
    algorithm_values,
    index=algorithm_values.index(cfg.math.coverage_algorithm),
)
seed = st.sidebar.number_input("随机种子", min_value=0, value=cfg.project.random_seed)

pool = select_pool(int(pool_size), int(seed))
designer = CoverageDesigner(CoverageAlgorithm(algorithm), cfg.math.solver_timeout_s)
plan = designer.build(pool, Guarantee(guarantee))

columns = st.columns(4)
columns[0].metric("方案注数", len(plan.tickets))
columns[1].metric("理论下界", plan.lower_bound)
columns[2].metric("是否达界", "是" if plan.proven_optimal else "否")
columns[3].metric("单注价格", f"{cfg.lottery.ticket_price:.2f} 元")

st.subheader("覆盖方案")
rows = [
    {"注号": index, "红球": " ".join(f"{number:02d}" for number in ticket.reds)}
    for index, ticket in enumerate(plan.tickets, start=1)
]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.plotly_chart(coverage_matrix_plot(plan), use_container_width=True)

st.subheader("奖级概率与预算预警")
probabilities = HypergeometricCalculator().prize_probabilities()
prize_rows = [
    {
        "奖级": f"{level.value}等奖",
        "单注概率": f"{probability:.2e}",
        "约每多少注中一次": f"{1 / probability:,.0f}",
    }
    for level, probability in probabilities.items()
]
st.dataframe(pd.DataFrame(prize_rows), hide_index=True, use_container_width=True)

budget = BudgetPlanner().plan(
    cfg.money.total_budget, cfg.lottery.ticket_price, cfg.money.periods
)
st.write(
    f"总预算 {cfg.money.total_budget:.2f} 元，分 {cfg.money.periods} 期，"
    f"每期 {budget.per_period:.2f} 元，可投 {budget.n_tickets_per_period} 注，"
    f"余数 {budget.remainder:.2f} 元。"
)
warning = RiskAssessor(cfg.money.confidence).warn_threshold(sum(probabilities.values()))
render_risk_warning(warning)
st.caption(
    f"止损规则：累计亏损达到预算的 {cfg.money.stop_loss_ratio:.0%} 时，"
    "系统只输出停止提示，不提供继续投注建议。"
)
