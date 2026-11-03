"""教学演示页：历史拟合推荐、候选池与冷热分布。"""

import pandas as pd
import streamlit as st

from lottery.common.config import get_config
from lottery.common.constants import RED_MAX, RED_MIN
from lottery.common.types import RecommendOptions
from lottery.teaching.features import FeatureBuilder
from lottery.teaching.pipeline import recommend
from lottery.ui.charts import cold_hot_bar
from lottery.ui.components import render_route_note
from lottery.ui.data import load_draws

cfg = get_config()
st.title("教学演示路线")
render_route_note("teaching_note")

draws = load_draws()
if not draws:
    st.error("没有可用历史数据，请先到「数据管理」页刷新。")
    st.stop()

window = st.sidebar.number_input(
    "历史窗口（期）",
    min_value=1,
    max_value=len(draws),
    value=min(cfg.recommend.window, len(draws)),
)
pool_size = st.sidebar.number_input(
    "候选池大小", min_value=6, max_value=RED_MAX - RED_MIN + 1, value=cfg.recommend.pool_size
)
count = st.sidebar.number_input("推荐注数", min_value=1, max_value=50, value=cfg.recommend.ticket_count)
seed = st.sidebar.number_input("随机种子", min_value=0, value=cfg.project.random_seed)

options = RecommendOptions(
    count=int(count), pool_size=int(pool_size), window=int(window), seed=int(seed)
)
result = recommend(draws, cfg, options)
st.caption(
    f"最新期号 {result.latest_issue} · 历史 {result.n_draws} 期 · 统计窗口 {result.window} 期"
)

rows = [
    {
        "注号": index,
        "红球": " ".join(f"{number:02d}" for number in ticket.reds),
        "蓝球": f"{ticket.blue:02d}" if ticket.blue is not None else "-",
    }
    for index, ticket in enumerate(result.pool.tickets, start=1)
]
st.subheader("推荐号码")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

features = FeatureBuilder().build(draws, int(window))
frequency = pd.Series(
    {number: features.loc[0, f"freq_{number}"] for number in range(RED_MIN, RED_MAX + 1)}
)
st.plotly_chart(cold_hot_bar(frequency), use_container_width=True)
