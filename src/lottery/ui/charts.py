"""Plotly 图表工厂；标题内嵌路线标签，防止数字脱离路线语境被误读。"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lottery.common.constants import RED_MAX, RED_MIN
from lottery.theory.coverage import CoveragePlan
from lottery.ui.components import route_label

_HEAT_ON = "#d62728"
_HEAT_OFF = "#f0f0f0"


def _probabilities(series: pd.Series) -> pd.Series:
    """把计数序列归一化为概率；总和非正时直接报错。"""
    total = float(series.sum())
    if total <= 0:
        raise ValueError("分布总和必须为正")
    return series.astype(float) / total


def cold_hot_bar(freq: pd.Series) -> go.Figure:
    """绘制红球冷热分布柱状图，标题标注教学演示路线。"""
    figure = go.Figure(
        go.Bar(
            x=[int(number) for number in freq.index],
            y=[float(value) for value in freq.values],
            name="出现次数",
        )
    )
    figure.update_layout(
        title=f"{route_label('teaching')} 红球冷热分布",
        xaxis_title="红球号码",
        yaxis_title="出现次数",
        xaxis={"dtick": 1},
        bargap=0.15,
    )
    return figure


def sum_distribution_plot(dist: pd.Series, observed: pd.Series | None = None) -> go.Figure:
    """绘制和值理论分布，可叠加历史观测；两条曲线均换算为概率。"""
    theory = _probabilities(dist)
    figure = go.Figure()
    figure.add_bar(x=theory.index, y=theory.values, name="理论概率")
    if observed is not None:
        history = _probabilities(observed)
        figure.add_scatter(
            x=history.index, y=history.values, mode="lines+markers", name="历史观测"
        )
    figure.update_layout(
        title=f"{route_label('math')} 和值理论分布与历史观测",
        xaxis_title="红球和值",
        yaxis_title="概率",
    )
    return figure


def backtest_curve(df: pd.DataFrame, route: str = "teaching") -> go.Figure:
    """绘制回测累计收益曲线；缺少必要列时直接报错。"""
    missing = {"issue", "cumulative_return"} - set(df.columns)
    if missing:
        raise ValueError(f"回测曲线缺少列: {sorted(missing)}")
    figure = go.Figure(
        go.Scatter(
            x=df["issue"],
            y=df["cumulative_return"],
            mode="lines+markers",
            name="累计净收益",
        )
    )
    figure.add_hline(y=0, line_dash="dash", line_color="gray")
    figure.update_layout(
        title=f"{route_label(route)} 回测累计收益曲线",
        xaxis_title="期号",
        yaxis_title="累计净收益（元）",
    )
    return figure


def prediction_heatmap(probs: pd.DataFrame) -> go.Figure:
    """绘制号码概率热力图，标题标注教学演示路线。"""
    if probs.empty:
        raise ValueError("概率矩阵不能为空")
    figure = go.Figure(
        go.Heatmap(
            z=probs.to_numpy(dtype=float),
            x=[str(column) for column in probs.columns],
            y=[str(index) for index in probs.index],
            colorscale="Blues",
        )
    )
    figure.update_layout(
        title=f"{route_label('teaching')} 预测概率热力图",
        xaxis_title="号码",
        yaxis_title="注单",
    )
    return figure


def coverage_matrix_plot(plan: CoveragePlan) -> go.Figure:
    """绘制覆盖矩阵：行为注单，列为 1..33 号码，命中格为红色。"""
    numbers = list(range(RED_MIN, RED_MAX + 1))
    matrix = np.zeros((len(plan.tickets), len(numbers)), dtype=int)
    for row, ticket in enumerate(plan.tickets):
        for number in ticket.reds:
            matrix[row, number - RED_MIN] = 1
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=numbers,
            y=[f"第{row}注" for row in range(1, len(plan.tickets) + 1)],
            colorscale=[[0, _HEAT_OFF], [1, _HEAT_ON]],
            showscale=False,
        )
    )
    figure.update_layout(
        title=f"{route_label('math')} 覆盖矩阵",
        xaxis_title="红球号码",
        yaxis_title="注单",
        xaxis={"dtick": 1},
    )
    return figure
