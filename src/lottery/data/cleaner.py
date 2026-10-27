"""清洗开奖数据：去重、排序与缺期报告。"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleanResult:
    """清洗结果：重复数、排序修正标记、缺期列表、健康标记与清洗后数据。"""

    n_dropped_dup: int
    n_sorted_fixed: int
    missing_issues: list[str]
    ok: bool
    frame: pd.DataFrame


def _is_strictly_ascending(issues: pd.Series) -> bool:
    """判断期号序列是否严格升序。"""
    values = issues.tolist()
    return all(prev < cur for prev, cur in zip(values, values[1:]))


def _missing_in_year(year: str, issues: list[str]) -> list[str]:
    """返回某年份最早期号与最晚期号之间缺失的期号，格式 YYYYNNN。"""
    seen = {int(issue[4:]) for issue in issues}
    return [f"{year}{n:03d}" for n in range(min(seen), max(seen) + 1) if n not in seen]


def _missing_issues(issues: pd.Series) -> list[str]:
    """按年份分组报告区间内缺失的期号，不跨年补期。"""
    missing: list[str] = []
    for year in sorted({issue[:4] for issue in issues}):
        group = [issue for issue in issues if issue.startswith(year)]
        missing.extend(_missing_in_year(year, group))
    return missing


class DrawCleaner:
    """对已加载的开奖 DataFrame 做去重、排序与缺期检测。"""

    def clean(self, df: pd.DataFrame) -> CleanResult:
        """返回去重排序后的数据及缺期报告；缺期只报告，不补行。"""
        n_dropped = int(df.duplicated(subset="issue", keep="last").sum())
        n_sorted = 0 if _is_strictly_ascending(df["issue"]) else 1
        cleaned = df.drop_duplicates(subset="issue", keep="last")
        cleaned = cleaned.sort_values("issue", kind="stable").reset_index(drop=True)
        missing = _missing_issues(cleaned["issue"])
        return CleanResult(n_dropped, n_sorted, missing, not missing, cleaned)
