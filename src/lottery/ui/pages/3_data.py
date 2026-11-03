"""数据管理页：历史数据概览、联网刷新、冷热与和值分布。"""

import pandas as pd
import streamlit as st

from lottery.common.config import get_config
from lottery.common.constants import RED_MAX, RED_MIN
from lottery.data.crawler import CrawlError
from lottery.data.parser import ParseError
from lottery.data.storage import StorageError
from lottery.teaching.features import FeatureBuilder
from lottery.ui.charts import cold_hot_bar, sum_distribution_plot
from lottery.ui.data import load_draws, refresh_draws, sum_distribution

cfg = get_config()
st.title("数据管理")
draws = load_draws()
if not draws:
    st.error("没有可用历史数据，请点击下方按钮联网刷新。")

if st.button("联网刷新历史数据"):
    try:
        result = refresh_draws()
        draws = load_draws()
        if result.ok:
            st.success(f"刷新完成：共 {len(draws)} 期，未发现缺期。")
        else:
            st.warning(
                f"刷新完成：共 {len(draws)} 期，发现 {len(result.missing_issues)} 个缺期"
                f"（示例 {result.missing_issues[:10]}）。"
            )
    except (CrawlError, ParseError, StorageError) as exc:
        st.error(f"刷新失败：{exc}")

if not draws:
    st.stop()

window = st.sidebar.number_input(
    "统计窗口（期）",
    min_value=1,
    max_value=len(draws),
    value=min(cfg.recommend.window, len(draws)),
)
limit = st.sidebar.number_input("展示最近期数", min_value=5, max_value=200, value=20)

columns = st.columns(3)
columns[0].metric("最新期号", draws[-1].issue)
columns[1].metric("历史总期数", len(draws))
columns[2].metric("时间范围", f"{draws[0].draw_date} ~ {draws[-1].draw_date}")

st.subheader(f"最近 {int(limit)} 期")
recent = draws[-int(limit) :][::-1]
rows = [
    {
        "期号": record.issue,
        "开奖日期": str(record.draw_date),
        "红球": " ".join(f"{number:02d}" for number in record.reds),
        "蓝球": f"{record.blue:02d}",
    }
    for record in recent
]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

features = FeatureBuilder().build(draws, int(window))
frequency = pd.Series(
    {number: features.loc[0, f"freq_{number}"] for number in range(RED_MIN, RED_MAX + 1)}
)
st.plotly_chart(cold_hot_bar(frequency), use_container_width=True)

observed_sums = pd.Series([sum(record.reds) for record in draws[-int(window) :]])
observed_counts = observed_sums.value_counts().sort_index()
st.plotly_chart(
    sum_distribution_plot(sum_distribution(), observed_counts), use_container_width=True
)
