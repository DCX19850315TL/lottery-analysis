from datetime import date
import pytest
from lottery.common.types import DrawRecord, Ticket

def test_draw_record_accepts_valid():
    d = DrawRecord("2026122", date(2026, 10, 25), (1, 2, 4, 17, 19, 30), 9)
    assert d.reds == (1, 2, 4, 17, 19, 30)

@pytest.mark.parametrize("reds", [
    (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 5),
    (0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 34), (2, 1, 3, 4, 5, 6),
])
def test_draw_record_rejects_bad_reds(reds):
    with pytest.raises(ValueError):
        DrawRecord("2026122", date(2026, 10, 25), reds, 9)

def test_draw_record_rejects_bad_blue_and_issue():
    with pytest.raises(ValueError):
        DrawRecord("2026122", date(2026, 10, 25), (1, 2, 3, 4, 5, 6), 17)
    with pytest.raises(ValueError):
        DrawRecord("202612", date(2026, 10, 25), (1, 2, 3, 4, 5, 6), 1)

def test_ticket_rejects_bad_reds():
    assert Ticket((1, 2, 3, 4, 5, 6), 9).blue == 9
    with pytest.raises(ValueError):
        Ticket((1, 2, 3, 4, 5, 6), 0)
