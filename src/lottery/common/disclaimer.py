"""终端风险横幅渲染。"""

import os
from typing import TextIO

_BANNER_BORDER = "-" * 60
_RED = "\033[91m"
_RESET = "\033[0m"


def _use_color(stream: TextIO) -> bool:
    """仅在非 Windows 且传入流为交互式终端时启用 ANSI 颜色。"""
    return os.name != "nt" and stream.isatty()


def render_banner(text: str, stream: TextIO) -> None:
    """把给定风险文案写成带上下边框的横幅；交互式终端额外加红色。"""
    lines = [_BANNER_BORDER, text, _BANNER_BORDER]
    if _use_color(stream):
        lines = [f"{_RED}{line}{_RESET}" for line in lines]
    print("\n".join(lines), file=stream)
