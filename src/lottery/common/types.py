"""双色球核心不可变数据类型。"""

from dataclasses import dataclass
from datetime import date

from lottery.common.constants import (
    BLUE_MAX,
    BLUE_MIN,
    ISSUE_PATTERN,
    PICK_RED,
    RED_MAX,
    RED_MIN,
)


def _validate_reds(reds: tuple[int, ...]) -> None:
    """校验红球：恰好 6 个、位于 1..33、严格升序且无重复。"""
    if len(reds) != PICK_RED:
        raise ValueError(f"红球必须恰好 {PICK_RED} 个，实际 {len(reds)} 个")
    if any(not RED_MIN <= red <= RED_MAX for red in reds):
        raise ValueError(f"红球必须位于 {RED_MIN}..{RED_MAX}")
    if any(left >= right for left, right in zip(reds, reds[1:])):
        raise ValueError("红球必须严格升序且不重复")


def _validate_blue(blue: int) -> None:
    """校验蓝球：位于 1..16。"""
    if not BLUE_MIN <= blue <= BLUE_MAX:
        raise ValueError(f"蓝球必须位于 {BLUE_MIN}..{BLUE_MAX}")


@dataclass(frozen=True, order=True)
class DrawRecord:
    """一期开奖记录。"""

    issue: str
    draw_date: date
    reds: tuple[int, ...]
    blue: int

    def __post_init__(self) -> None:
        """校验期号、红球与蓝球。"""
        if ISSUE_PATTERN.fullmatch(self.issue) is None:
            raise ValueError(f"期号必须为 7 位数字: {self.issue!r}")
        _validate_reds(self.reds)
        _validate_blue(self.blue)


@dataclass(frozen=True)
class Ticket:
    """一注号码，蓝球可选。"""

    reds: tuple[int, ...]
    blue: int | None = None

    def __post_init__(self) -> None:
        """校验红球，蓝球存在时校验范围。"""
        _validate_reds(self.reds)
        if self.blue is not None:
            _validate_blue(self.blue)


@dataclass(frozen=True)
class TicketPool:
    """候选号码池及其随机种子。"""

    tickets: tuple[Ticket, ...]
    pool_numbers: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        """校验号码池非空且号码位于 1..33。"""
        if not self.pool_numbers:
            raise ValueError("pool_numbers 不能为空")
        if any(not RED_MIN <= number <= RED_MAX for number in self.pool_numbers):
            raise ValueError(f"pool_numbers 必须位于 {RED_MIN}..{RED_MAX}")


@dataclass(frozen=True)
class Recommendation:
    """一次推荐结果。"""

    pool: TicketPool
    latest_issue: str
    window: int
    n_draws: int


@dataclass(frozen=True)
class RecommendOptions:
    """推荐参数配置。"""

    count: int
    pool_size: int
    window: int
    seed: int
