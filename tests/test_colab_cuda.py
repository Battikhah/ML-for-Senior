"""Protect PyTorch's CUDA installation when supplying TensorFlow's older cuDNN."""
import importlib.util
import subprocess
import sys
from types import SimpleNamespace


def test_colab_cuda_links_private_cudnn_without_replacing_existing_libraries(tmp_path, monkeypatch):
    """Repeated setup must retain cuDNN 9 and avoid downloading cuDNN 8 again."""
    from treevibes.notebook_setup import setup_colab_cuda
    site = tmp_path / 'site'
    tensorflow = site / 'tensorflow'
    tensorflow.mkdir(parents=True)
    vendor = site / 'nvidia' / 'cudnn' / 'lib'
    vendor.mkdir(parents=True)
    (vendor / 'libcudnn.so.9').write_bytes(b'pytorch-library')
    (tensorflow / 'libkeep.so.1').write_bytes(b'existing-library')
    (vendor / 'libkeep.so.1').write_bytes(b'do-not-replace')
    cache = tmp_path / 'private-cudnn8'
    calls = []

    def install(command):
        """Represent pip's private target without downloading GPU binaries in tests."""
        calls.append(command)
        library = cache / 'nvidia' / 'cudnn' / 'lib' / 'libcudnn.so.8'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'tensorflow-library')

    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(tensorflow / '__init__.py')))
    monkeypatch.setattr(subprocess, 'check_call', install)
    monkeypatch.setattr(sys, 'path', [str(site)])
    assert setup_colab_cuda(cache) == 2
    assert (tensorflow / 'libcudnn.so.8').read_bytes() == b'tensorflow-library'
    assert (vendor / 'libcudnn.so.9').read_bytes() == b'pytorch-library'
    assert (tensorflow / 'libkeep.so.1').read_bytes() == b'existing-library'
    assert '--no-deps' in calls[0] and '--target' in calls[0]
    assert 'nvidia-cudnn-cu12==8.9.7.29' in calls[0]
    assert setup_colab_cuda(cache) == 0 and len(calls) == 1


def test_tensorflow_only_environment_uses_its_installed_cudnn8(tmp_path, monkeypatch):
    """The official CUDA extra must not trigger another cuDNN download."""
    from treevibes.notebook_setup import setup_colab_cuda
    site = tmp_path/'site'
    tensorflow = site/'tensorflow'
    tensorflow.mkdir(parents=True)
    library = site/'nvidia/cudnn/lib/libcudnn.so.8'
    library.parent.mkdir(parents=True)
    library.write_bytes(b'installed-cudnn8')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(tensorflow/'__init__.py')))
    monkeypatch.setattr(sys, 'path', [str(site)])

    def unexpected_download(command):
        """Fail if the already-installed official dependency is unnecessarily replaced."""
        raise AssertionError('cuDNN 8 is already installed')

    monkeypatch.setattr(subprocess, 'check_call', unexpected_download)
    setup_colab_cuda(tmp_path/'unused-cache')
    assert (tensorflow/'libcudnn.so.8').read_bytes() == b'installed-cudnn8'
