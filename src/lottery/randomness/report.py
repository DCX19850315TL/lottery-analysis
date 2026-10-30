"""随机性检验报告序列化：JSON 与 Markdown。"""

import json

from lottery.randomness.suite import RandomnessReport


def to_dict(report: RandomnessReport, note: str) -> dict:
    """把报告转成可序列化字典，note 为风险声明文案。"""
    return {
        "n_draws": report.n_draws,
        "overall_pass": report.overall_pass,
        "note": note,
        "results": [
            {
                "name": result.name,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "adjusted_alpha": result.adjusted_alpha,
                "passed": result.passed,
            }
            for result in report.results
        ],
    }


def to_json(report: RandomnessReport, note: str) -> str:
    """输出 UTF-8 友好的 JSON 文本（中文不转义）。"""
    return json.dumps(to_dict(report, note), ensure_ascii=False, indent=2)


def to_markdown(report: RandomnessReport, note: str) -> str:
    """输出 Markdown 报告：总体结论、逐项结果表与风险声明。"""
    verdict = "通过" if report.overall_pass else "未通过"
    lines = [
        f"# 随机性检验报告（{report.n_draws} 期）",
        "",
        f"总体结论: {verdict}",
        f"风险声明: {note}",
        "",
        "| 检验 | 统计量 | p 值 | 校正后 alpha | 通过 |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {result.name} | {result.statistic:.4f} | {result.p_value:.4f} "
        f"| {result.adjusted_alpha:.4f} | {'是' if result.passed else '否'} |"
        for result in report.results
    )
    return "\n".join(lines)
