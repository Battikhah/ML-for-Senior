"""Validate the two supported Windows GPU and professor-review notebooks."""
import ast
from pathlib import Path

import nbformat


def test_supported_notebooks_have_valid_python():
    """Keep both published entry points syntactically runnable."""
    root = Path.cwd()
    paths = [root / "TreeVibes_Windows_GPU_All_In_One.ipynb",
             root / "TreeVibes_Final_Professor_Review.ipynb"]
    assert all(path.is_file() for path in paths)
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
