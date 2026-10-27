"""双色球号码范围与期号校验常量。"""

import re

RED_MIN = 1
RED_MAX = 33
PICK_RED = 6
BLUE_MIN = 1
BLUE_MAX = 16
ISSUE_PATTERN = re.compile(r"^\d{7}$")
