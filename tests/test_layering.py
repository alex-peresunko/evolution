"""The package layering in CLAUDE.md: an import may only point down the chain."""
import ast
from pathlib import Path

import pytest

PACKAGE = "evolution"
ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# Bottom to top; modules in one layer may import each other.
LAYERS = [
    {"config", "genetics", "species", "diet"},
    {"brain"},
    {"world"},
    {"sim"},
    {"persistence"},
    {"app"},
    {"ui"},
]
RANK = {name: rank for rank, names in enumerate(LAYERS) for name in names}
PYGAME_LAYER = "ui"


def _modules():
    for path in sorted(ROOT.rglob("*.py")):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        if parts:
            yield path, parts, is_package


def _imports(path, parts, is_package):
    """Dotted names of everything the module imports, relative imports resolved."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                yield node.module
                continue
            keep = len(parts) - node.level + (1 if is_package else 0)
            base = [PACKAGE, *parts[:keep]]
            if node.module:
                yield ".".join([*base, node.module])
            else:
                yield from (".".join([*base, alias.name]) for alias in node.names)


MODULES = list(_modules())
IDS = [".".join(parts) for _, parts, _ in MODULES]


def test_every_top_level_module_has_a_layer():
    top_level = {parts[0] for _, parts, _ in MODULES}
    assert top_level == set(RANK)


@pytest.mark.parametrize("path, parts, is_package", MODULES, ids=IDS)
def test_imports_point_down_the_chain(path, parts, is_package):
    own = parts[0]
    upward = []
    for name in _imports(path, parts, is_package):
        target = name.split(".")
        if target[0] == PACKAGE and len(target) > 1 and RANK[target[1]] > RANK[own]:
            upward.append(name)
    assert not upward, f"{own}/ may not import from a higher layer: {upward}"


@pytest.mark.parametrize("path, parts, is_package", MODULES, ids=IDS)
def test_only_the_ui_imports_pygame(path, parts, is_package):
    if parts[0] == PYGAME_LAYER:
        return
    assert not [name for name in _imports(path, parts, is_package)
                if name.split(".")[0] == "pygame"]
