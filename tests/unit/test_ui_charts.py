import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lottery.common.types import Ticket
from lottery.theory.coverage import CoveragePlan, Guarantee
from lottery.ui.charts import (
    backtest_curve,
    cold_hot_bar,
    coverage_matrix_plot,
    prediction_heatmap,
    sum_distribution_plot,
)


def _frequency() -> pd.Series:
    return pd.Series({number: float(number) for number in range(1, 34)})


def test_cold_hot_bar_labels_teaching_route():
    figure = cold_hot_bar(_frequency())
    assert isinstance(figure, go.Figure)
    assert "[教学演示]" in figure.layout.title.text
    assert list(figure.data[0].x) == list(range(1, 34))


def test_sum_distribution_normalizes_both_series():
    theory = pd.Series({21: 1.0, 22: 3.0})
    observed = pd.Series({21: 2, 22: 6})
    figure = sum_distribution_plot(theory, observed)
    assert "[数学最优]" in figure.layout.title.text
    assert len(figure.data) == 2
    assert sum(figure.data[0].y) == pytest.approx(1.0)
    assert sum(figure.data[1].y) == pytest.approx(1.0)


def test_backtest_curve_plots_cumulative_return():
    curve = pd.DataFrame({"issue": ["1", "2"], "cumulative_return": [0.0, -2.0]})
    figure = backtest_curve(curve)
    assert "[教学演示]" in figure.layout.title.text
    assert list(figure.data[0].y) == [0.0, -2.0]


def test_backtest_curve_requires_cumulative_return():
    with pytest.raises(ValueError):
        backtest_curve(pd.DataFrame({"issue": ["1"]}))


def test_prediction_heatmap_keeps_input_shape():
    probs = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=["t1", "t2"], columns=[1, 2])
    figure = prediction_heatmap(probs)
    assert "[教学演示]" in figure.layout.title.text
    assert np.asarray(figure.data[0].z).shape == (2, 2)


def test_coverage_matrix_marks_six_numbers_per_ticket():
    plan = CoveragePlan(
        tickets=(Ticket((1, 2, 3, 4, 5, 6)), Ticket((2, 3, 4, 5, 6, 7))),
        guarantee=Guarantee.MATCH6_COVER5,
        proven_optimal=False,
        lower_bound=10,
        cost=2.0,
    )
    figure = coverage_matrix_plot(plan)
    matrix = np.asarray(figure.data[0].z)
    assert "[数学最优]" in figure.layout.title.text
    assert matrix.shape == (2, 33)
    assert matrix.sum(axis=1).tolist() == [6, 6]
