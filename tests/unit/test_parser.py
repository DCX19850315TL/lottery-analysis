import pytest
from lottery.data.parser import parse_draw, parse_response, ParseError

GOOD = {"code": "2026122", "date": "2026-10-25(日)",
        "red": "01,02,04,17,19,30", "blue": "09"}

def test_parse_draw_valid():
    d = parse_draw(GOOD)
    assert d.issue == "2026122"
    assert d.reds == (1, 2, 4, 17, 19, 30)
    assert d.blue == 9
    assert str(d.draw_date) == "2026-10-25"

@pytest.mark.parametrize("patch", [
    {"red": "01,02,04,17,19"}, {"red": "01,02,04,17,19,19"},
    {"red": "01,02,04,17,19,34"}, {"red": "30,19,17,04,02,01"},
    {"blue": "00"}, {"blue": "17"}, {"code": "202612"}, {"date": "not-a-date"},
])
def test_parse_draw_rejects_bad_fields(patch):
    with pytest.raises(ParseError):
        parse_draw({**GOOD, **patch})

def test_parse_response_state_error():
    with pytest.raises(ParseError):
        parse_response({"state": 1, "result": []})

def test_parse_response_returns_records():
    out = parse_response({"state": 0, "result": [GOOD, {**GOOD, "code": "2026121"}]})
    assert [d.issue for d in out] == ["2026122", "2026121"]
