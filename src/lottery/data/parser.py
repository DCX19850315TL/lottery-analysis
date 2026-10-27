"""解析福彩接口响应为已校验的开奖记录。"""

from datetime import date

from lottery.common.types import DrawRecord


class ParseError(ValueError):
    """接口响应无法解析为开奖记录。"""


def _field(raw: dict[str, object], name: str) -> str:
    """取出必填字符串字段，缺失或类型错误时抛 ParseError。"""
    value = raw.get(name)
    if not isinstance(value, str):
        raise ParseError(f"字段 {name!r} 缺失或不是字符串: {value!r}")
    return value


def _parse_issue(raw: dict[str, object]) -> str:
    """提取期号，格式由 DrawRecord 校验。"""
    return _field(raw, "code")


def _parse_date(raw: dict[str, object]) -> date:
    """提取开奖日期，去掉星期后缀后按 ISO 格式解析。"""
    text = _field(raw, "date")
    try:
        return date.fromisoformat(text.split("(", 1)[0].strip())
    except ValueError as exc:
        raise ParseError(f"日期无法解析: {text!r}") from exc


def _parse_reds(raw: dict[str, object]) -> tuple[int, ...]:
    """把逗号分隔的红球字符串转成整数元组。"""
    text = _field(raw, "red")
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise ParseError(f"红球无法解析: {text!r}") from exc


def _parse_blue(raw: dict[str, object]) -> int:
    """把蓝球字符串转成整数。"""
    text = _field(raw, "blue")
    try:
        return int(text)
    except ValueError as exc:
        raise ParseError(f"蓝球无法解析: {text!r}") from exc


def parse_draw(raw: dict[str, object]) -> DrawRecord:
    """把单条接口记录解析为 DrawRecord，失败时抛 ParseError。"""
    if not isinstance(raw, dict):
        raise ParseError(f"开奖记录必须是对象: {type(raw).__name__}")
    try:
        return DrawRecord(
            issue=_parse_issue(raw),
            draw_date=_parse_date(raw),
            reds=_parse_reds(raw),
            blue=_parse_blue(raw),
        )
    except ParseError:
        raise
    except ValueError as exc:
        raise ParseError(f"开奖记录无效: {exc}") from exc


def parse_response(payload: dict[str, object]) -> list[DrawRecord]:
    """解析接口响应中 result 列表的全部开奖记录。"""
    state = payload.get("state")
    if state != 0:
        raise ParseError(f"接口返回错误状态: {state!r}")
    result = payload.get("result")
    if not isinstance(result, list):
        raise ParseError(f"result 不是列表: {type(result).__name__}")
    return [parse_draw(item) for item in result]
