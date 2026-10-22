"""Check ZIP recovery preserves experiments and rejects unsafe extraction paths."""
import zipfile
from pathlib import Path
import pytest
from scripts.zip_recovery import extract_snapshot, download_zip


def make_snapshot(path, extra=None):
    """Create a small realistic wrapped Drive archive."""
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr('TreeVibesExperiments/notebook_settings.json', '{}')
        bundle.writestr('TreeVibesExperiments/runs/model/checkpoint.bin', b'weights')
        bundle.writestr('TreeVibesExperiments/cache/features.npy', b'features')
        if extra:
            bundle.writestr(extra, b'unsafe')


def test_zip_download_extract_and_reuse(tmp_path, monkeypatch):
    """One transfer must support repeated recovery without network access."""
    archive=tmp_path/'fixture.zip';make_snapshot(archive)
    calls=[]
    def download(identifier, target):
        """Replace the public network boundary with fixed ZIP bytes."""
        calls.append(identifier)
        target.write_bytes(archive.read_bytes())
    monkeypatch.setattr('scripts.public_drive_download.direct_download',download)
    root=download_zip(tmp_path/'work','zip-id')
    assert (root/'runs/model/checkpoint.bin').read_bytes()==b'weights'
    assert download_zip(tmp_path/'work','zip-id')==root and calls==['zip-id']
    with pytest.raises(ValueError,match='another ZIP'):
        download_zip(tmp_path/'work','other-id')


@pytest.mark.parametrize('entry',['../escape','/absolute','C:/escape','folder\\escape','TREEVIBESEXPERIMENTS/notebook_settings.json'])
def test_zip_rejects_unsafe_paths_before_extracting(tmp_path,entry):
    """Windows and POSIX traversal and case collisions must fail before writing payloads."""
    archive=tmp_path/'unsafe.zip';make_snapshot(archive,entry)
    destination=tmp_path/'snapshot'
    with pytest.raises(ValueError,match='Unsafe'):
        extract_snapshot(archive,destination,'id')
    assert not destination.exists() and not (tmp_path/'escape').exists()


def test_gpu_notebook_downloads_zip_instead_of_folder(tmp_path):
    """The delivered notebook defaults to the user's single shared archive."""
    import nbformat
    from scripts.package_windows_gpu import build_gpu_notebook
    root=Path(__file__).resolve().parents[1]
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w'):
        pass
    notebook=build_gpu_notebook(root,archive,tmp_path/'gpu.ipynb')
    source='\n'.join(c.source for c in nbformat.read(notebook,as_version=4).cells if c.cell_type=='code')
    assert '1MsF2OEAHqL4uOhhZ8nJEiFCEMVoEn1ZT' in source
    assert 'SOURCE = download_zip(WORK, DRIVE_ZIP_ID)' in source
    assert 'SOURCE = download_public_drive' not in source
