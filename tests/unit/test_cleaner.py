import pandas as pd

from lottery.data.cleaner import DrawCleaner
from lottery.data.storage import COLUMNS


def frame(issues):
    rows = []
    for i in issues:
        rows.append({"issue": i, "draw_date": "2026-10-25", "red1": 1, "red2": 2,
                     "red3": 3, "red4": 4, "red5": 5, "red6": 6, "blue": 9})
    return pd.DataFrame(rows, columns=COLUMNS)


def test_dedupes_keeping_last_and_sorts():
    df = frame(["2026003", "2026001", "2026003", "2026002"])
    df.loc[2, "blue"] = 10
    res = DrawCleaner().clean(df)
    assert res.n_dropped_dup == 1
    assert res.n_sorted_fixed == 1
    assert res.frame["issue"].tolist() == ["2026001", "2026002", "2026003"]
    assert res.frame.iloc[-1]["blue"] == 10


def test_reports_missing_issues_without_filling():
    res = DrawCleaner().clean(frame(["2026001", "2026003"]))
    assert res.missing_issues == ["2026002"]
    assert len(res.frame) == 2


def test_clean_frame_is_ok_when_no_missing():
    res = DrawCleaner().clean(frame(["2026001", "2026002"]))
    assert res.ok is True and res.missing_issues == []


def test_empty_frame_is_ok():
    res = DrawCleaner().clean(frame([]))
    assert res.ok is True and res.frame.empty


def test_sorted_fixed_uses_input_order_before_dedupe():
    res = DrawCleaner().clean(frame(["2026001", "2026002", "2026001"]))
    assert res.n_sorted_fixed == 1
    assert res.frame["issue"].tolist() == ["2026001", "2026002"]


def test_missing_does_not_bridge_years_and_zero_pads():
    res = DrawCleaner().clean(frame(["2025005", "2025006", "2026009", "2026011"]))
    assert res.missing_issues == ["2026010"]
    assert res.ok is False


def test_empty_frame_keeps_columns():
    res = DrawCleaner().clean(frame([]))
    assert list(res.frame.columns) == COLUMNS
