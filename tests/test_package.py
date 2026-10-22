"""Check the source archive used by the Windows GPU notebook."""
import zipfile


def test_source_archive_excludes_local_data_and_credentials(tmp_path):
    """Packaging must not sweep up models, caches or hidden account files."""
    from treevibes.packaging import build_colab_archive

    root = tmp_path / "project"
    (root / "treevibes").mkdir(parents=True)
    (root / "treevibes" / "__init__.py").write_text('"""Example package."""')
    (root / ".env").write_text("SECRET=do-not-copy")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "model.pt").write_bytes(b"not-public")
    (root / "README.md").write_text("Example")

    archive = build_colab_archive(root, tmp_path / "source.zip")
    with zipfile.ZipFile(archive) as stream:
        assert set(stream.namelist()) == {"treevibes/__init__.py", "README.md"}


def test_all_in_one_template_unpacks_embedded_project(tmp_path, monkeypatch):
    """The Windows notebook template must expose source before setup runs."""
    import ast
    import nbformat
    import sys
    from treevibes.packaging import build_colab_archive, build_all_in_one

    root = tmp_path / "project"
    (root / "treevibes").mkdir(parents=True)
    (root / "treevibes" / "__init__.py").write_text('"""Embedded project."""')
    archive = build_colab_archive(root, tmp_path / "source.zip")
    path = build_all_in_one(root, archive, tmp_path / "TreeVibes_template.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)

    remote = tmp_path / "remote"
    remote.mkdir()
    monkeypatch.chdir(remote)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("TREEVIBES_SMOKE", "1")
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(cell.source, namespace)
            if "ROOT" in namespace:
                break
    assert (namespace["ROOT"] / "treevibes" / "__init__.py").read_text() == '"""Embedded project."""'
    assert namespace["MODE"] == "smoke"
