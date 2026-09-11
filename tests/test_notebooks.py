"""Checks on the notebooks themselves.

The notebooks are the project's entry point, so a broken one costs a Colab
session to discover. These verify the parts that have actually gone wrong:
setup cells drifting apart from each other when only one gets regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def code_sources(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def setup_source(path: Path) -> str:
    """The environment cell: the one that clones and installs the package."""
    for source in code_sources(path):
        if "REPO_DIR" in source:
            return source
    raise AssertionError(f"{path.name} has no environment cell")


@pytest.fixture(params=notebook_paths(), ids=lambda p: p.name)
def notebook(request) -> Path:
    return request.param


def test_notebook_is_valid_json(notebook: Path):
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["cells"]
    assert payload["nbformat"] == 4


def test_environment_cell_installs_the_dev_extra(notebook: Path):
    """Without it `pytest` is missing and the verification cell cannot run."""
    assert '".[dev]"' in setup_source(notebook)


def test_environment_cell_makes_the_package_importable(notebook: Path):
    """A running kernel does not pick up an editable install on its own.

    pip records it through a ``.pth`` file, and those are only read when the
    interpreter starts, so every notebook has to put ``src`` on ``sys.path``
    itself. Leaving it out fails on the *next* cell with a bare
    ``ModuleNotFoundError``, which is a confusing way to find out.
    """
    source = setup_source(notebook)
    assert "sys.path.insert" in source
    assert "importlib.invalidate_caches()" in source


def test_environment_cell_does_not_hide_a_failed_pull(notebook: Path):
    """A silent `git pull` leaves the notebook running against stale code.

    With ``check=False`` a pull blocked by local edits prints nothing and the
    cell reports success, so the session carries on with whatever was there
    before -- the worst way for this to fail.
    """
    source = setup_source(notebook)
    assert 'check=False' not in source


def test_notebooks_do_not_use_google_drive(notebook: Path):
    """Everything durable lives on the Hub; nothing should mount a drive."""
    for source in code_sources(notebook):
        assert "drive.mount" not in source
        assert "/content/drive" not in source


def test_shell_cells_use_the_kernel_interpreter(notebook: Path):
    """`!python` resolves through PATH, which need not be the kernel's Python.

    ``!{sys.executable}`` runs the same interpreter the notebook is using, which
    is the one the package was installed into.
    """
    for source in code_sources(notebook):
        assert "!python " not in source
