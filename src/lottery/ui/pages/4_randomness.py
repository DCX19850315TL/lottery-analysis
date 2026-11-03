"""随机性检验页：六项检验、Bonferroni 校正结论与报告下载。"""

import pandas as pd
import streamlit as st

from lottery.common.config import get_config
from lottery.common.types import DrawRecord
from lottery.randomness.report import to_json, to_markdown
from lottery.randomness.suite import RandomnessReport, RandomnessSuite
from lottery.ui.data import load_draws

cfg = get_config()
st.title("随机性检验")


@st.cache_data(show_spinner=False)
def _run_suite(draws: tuple[DrawRecord, ...]) -> RandomnessReport:
    """运行配置启用的全部检验；相同数据直接复用结果。"""
    return RandomnessSuite(cfg.randomness).run(draws)


draws = load_draws()
if not draws:
    st.error("没有可用历史数据，请先到「数据管理」页刷新。")
    st.stop()

report = _run_suite(tuple(draws))
st.metric("总体结论", "通过" if report.overall_pass else "未通过")
st.caption(f"检验样本 {report.n_draws} 期 · 校正方式 {cfg.randomness.correction}")

rows = [
    {
        "检验": result.name,
        "统计量": f"{result.statistic:.4f}",
        "p 值": f"{result.p_value:.4f}",
        "校正后 alpha": f"{result.adjusted_alpha:.4f}",
        "通过": "是" if result.passed else "否",
    }
    for result in report.results
]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

note = cfg.project.disclaimers["global_banner"]
st.warning(f"风险声明：{note}", icon="⚠️")
st.download_button(
    "下载 Markdown 报告", to_markdown(report, note), file_name="randomness_report.md"
)
st.download_button(
    "下载 JSON 报告", to_json(report, note), file_name="randomness_report.json"
)
