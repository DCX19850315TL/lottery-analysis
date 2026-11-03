"""Streamlit 共用组件：全局风险横幅、路线声明与红色警示块。"""

import streamlit as st

from lottery.common.config import get_config
from lottery.money.risk import RiskWarning

_ROUTE_LABELS = {"teaching": "[教学演示]", "math": "[数学最优]"}


def route_label(route: str) -> str:
    """返回路线标签；未知路线直接报错，避免图表绕过路线声明。"""
    if route not in _ROUTE_LABELS:
        raise ValueError(f"未知路线: {route}，可选 {sorted(_ROUTE_LABELS)}")
    return _ROUTE_LABELS[route]


def render_disclaimer_banner() -> None:
    """渲染全局风险横幅；app.py 在任何页面之前调用一次。"""
    st.error(get_config().project.disclaimers["global_banner"], icon="⚠️")


def render_route_note(key: str) -> None:
    """渲染指定路线的免责声明，未知键直接报错。"""
    notes = get_config().project.disclaimers
    if key not in notes:
        raise ValueError(f"未知声明: {key}，可选 {sorted(notes)}")
    st.warning(notes[key], icon="⚠️")


def render_risk_warning(warning: RiskWarning) -> None:
    """渲染红色风险预警块，只用于预期管理。"""
    st.error(warning.message, icon="🛑")
