"""Share persistent-path setup across independently runnable Colab notebooks."""
import os
from pathlib import Path
from .data import load_dataset, write_json


def setup_colab_cuda(cuda_cache: Path = Path('/content/treevibes-cudnn8')) -> int:
    """Expose pip CUDA libraries to TensorFlow without replacing PyTorch's cuDNN 9."""
    import importlib.util
    import subprocess
    import sys
    spec = importlib.util.find_spec('tensorflow')
    if spec is None or spec.origin is None:
        raise RuntimeError('Install the pinned training requirements before configuring CUDA')
    tensorflow = Path(spec.origin).parent
    cuda_cache = Path(cuda_cache)
    installed_cudnn = tensorflow.parent / 'nvidia/cudnn/lib/libcudnn.so.8'
    if not installed_cudnn.is_file() and not (cuda_cache / 'nvidia/cudnn/lib/libcudnn.so.8').is_file():
        # Installing globally would remove cuDNN 9, which the PyTorch wheel needs.
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--upgrade',
                               '--target', str(cuda_cache), 'nvidia-cudnn-cu12==8.9.7.29'])
    count = 0
    # TensorFlow documents links beside its package for pip libraries absent from ldconfig.
    for root in [cuda_cache, *(Path(path) for path in sys.path if path)]:
        for library in sorted(root.glob('nvidia/*/lib/*.so*')):
            destination = tensorflow / library.name
            if library.is_file() and not (destination.exists() or destination.is_symlink()):
                destination.symlink_to(library.resolve())
                count += 1
    cudnn_location = installed_cudnn if installed_cudnn.is_file() else cuda_cache/'nvidia/cudnn/lib/libcudnn.so.8'
    print(f'TensorFlow CUDA library links created: {count}; cuDNN 8: {cudnn_location}')
    return count


def setup_output(smoke: bool = False) -> tuple[Path, Path | None]:
    """Mount user-approved Drive in Colab and keep active SQLite on the runtime disk."""
    in_colab='COLAB_RELEASE_TAG' in os.environ and not smoke
    if in_colab:
        from google.colab import drive
        drive.mount('/content/drive')
        tracking=Path(os.environ.get('TREEVIBES_TRACKING_DIR','/content/treevibes-tracking'))
        if tracking.resolve().is_relative_to(Path('/content/drive')):
            raise ValueError('Active SQLite must stay outside Drive; use /content/treevibes-tracking')
        os.environ['TREEVIBES_TRACKING_DIR']=str(tracking)
    if 'TREEVIBES_OUTPUT' in os.environ:
        output=Path(os.environ['TREEVIBES_OUTPUT'])
    elif in_colab:
        output=Path('/content/drive/MyDrive/TreeVibesExperiments')
    else:
        output=Path('artifacts/synthetic_notebooks' if smoke else 'artifacts')
    output.mkdir(parents=True,exist_ok=True)
    pointer=output/'active_cache.json'
    cache=None
    if pointer.exists():
        import json
        candidate=Path(json.loads(pointer.read_text())['path'])
        if candidate.exists():
            data=load_dataset(candidate)
            if data['contract'].get('synthetic',False)==smoke:
                cache=candidate
    if smoke and cache is None:
        from .synthetic import save_synthetic
        cache=save_synthetic(output/'synthetic_cache',40)
        write_json(pointer,{'path':str(cache.resolve())})
    print(f'Persistent outputs: {output.resolve()}')
    print(f'Feature cache: {cache or "Run notebook 01 first"}')
    return output,cache
