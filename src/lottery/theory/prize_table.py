"""奖级定义：六个中奖等级及其红蓝球命中条件。"""

from enum import IntEnum


class PrizeLevel(IntEnum):
    """双色球奖级，取值 1..6 对应一等奖到六等奖。"""

    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5
    SIXTH = 6
