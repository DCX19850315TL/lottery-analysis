import io

from lottery.common.disclaimer import render_banner


def test_banner_contains_text():
    out = io.StringIO()
    render_banner("风险提示", out)
    assert "风险提示" in out.getvalue()
