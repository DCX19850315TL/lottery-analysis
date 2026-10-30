import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "lottery"
ROUTES = {
    "theory": ("lottery.data", "lottery.teaching", "lottery.randomness"),
    "teaching": ("lottery.theory", "lottery.randomness"),
    "randomness": ("lottery.theory", "lottery.teaching"),
    "money": ("lottery.data", "lottery.teaching", "lottery.backtest"),
}


def _absolute_names(path: Path, package: str):
    """产出文件中所有 import 的绝对模块名，相对导入按所在包展开。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = f"lottery.{package}" if node.level else ""
            yield ".".join(filter(None, [prefix, node.module or ""]))


@pytest.mark.parametrize("package, banned", sorted(ROUTES.items()))
def test_route_does_not_import_other_routes(package, banned):
    root = SRC / package
    if not root.exists():
        pytest.skip(f"{package} 尚未创建")
    offenders = [
        f"{path.relative_to(SRC)} -> {name}"
        for path in root.rglob("*.py")
        for name in _absolute_names(path, package)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in banned)
    ]
    assert not offenders, f"{package} 不得依赖 {banned}: {offenders}"
