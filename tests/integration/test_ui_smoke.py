from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI_ROOT = Path(__file__).resolve().parents[2] / "src" / "lottery" / "ui"


@pytest.mark.parametrize(
    "page",
    ["pages/1_teaching.py", "pages/2_math.py", "pages/3_data.py", "pages/4_randomness.py"],
)
def test_page_renders_without_exception(page):
    app = AppTest.from_file(str(UI_ROOT / page), default_timeout=120).run()
    assert not app.exception


def test_app_shell_renders_banner_before_default_page():
    app = AppTest.from_file(str(UI_ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert any("风险提示" in element.value for element in app.error)
