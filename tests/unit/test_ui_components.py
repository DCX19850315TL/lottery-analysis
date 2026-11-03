import pytest
from streamlit.testing.v1 import AppTest

from lottery.ui.components import route_label


def test_route_label_known_routes():
    assert route_label("teaching") == "[教学演示]"
    assert route_label("math") == "[数学最优]"


def test_route_label_rejects_unknown_route():
    with pytest.raises(ValueError):
        route_label("randomness")


def test_banner_renders_global_disclaimer(tmp_path):
    # from_function 在 GBK 环境下写临时脚本会因 emoji 失败，改用 UTF-8 脚本文件
    script = tmp_path / "banner_app.py"
    script.write_text(
        "from lottery.ui.components import render_disclaimer_banner\n"
        "render_disclaimer_banner()\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=10).run()
    assert not app.exception
    assert any("风险提示" in element.value for element in app.error)
