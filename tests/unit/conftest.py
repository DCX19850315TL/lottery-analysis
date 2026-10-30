from datetime import date

import numpy as np
import pytest

from lottery.common.types import DrawRecord

ALL_REDS = np.arange(1, 34)


def _record(index: int, reds) -> DrawRecord:
    return DrawRecord(
        issue=f"{2024001 + index:07d}",
        draw_date=date(2024, 1, 1),
        reds=tuple(int(red) for red in sorted(reds)),
        blue=1,
    )


@pytest.fixture
def make_draws():
    """按期生成开奖记录；chooser(rng, index) 可替换默认的均匀抽样。"""

    def _make(n: int, seed: int = 0, chooser=None) -> list[DrawRecord]:
        rng = np.random.default_rng(seed)
        draws = []
        for index in range(n):
            reds = chooser(rng, index) if chooser else rng.choice(ALL_REDS, size=6, replace=False)
            draws.append(_record(index, reds))
        return draws

    return _make
