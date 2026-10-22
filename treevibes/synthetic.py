"""Provide clearly labeled synthetic data solely for integration checks."""
from pathlib import Path
import numpy as np
from .data import DEFAULT_FEATURES, fingerprint, normalization_stats, write_json


def synthetic_dataset(count: int = 80, seed: int = 42) -> dict:
    """Exercise model plumbing without pretending synthetic accuracy measures infestations."""
    if count < 20:
        raise ValueError('Use at least 20 synthetic examples')
    rng = np.random.default_rng(seed)
    data = {}; split = {}
    for part, size in [('train', count), ('val', max(8, count // 4)), ('test', max(8, count // 4))]:
        y = np.arange(size) % 2
        x = rng.normal(0, .3, (size, 30, 13)).astype('float32')
        x[:, :, 0] += y[:, None] * 2 - 1
        data[f'X_{part}'], data[f'y_{part}'] = x, y
        data[f'ids_{part}'] = np.array([f'synthetic-{part}-{i}' for i in range(size)])
        split[part] = data[f'ids_{part}'].tolist()
    data['contract'] = dict(dataset_id='synthetic-' + fingerprint(dict(count=count, seed=seed)),
                            features=DEFAULT_FEATURES, split={'partitions': split}, synthetic=True,
                            normalization=normalization_stats(data['X_train']))
    return data


def save_synthetic(cache: Path, count: int = 80) -> Path:
    """Persist a normal cache contract so smoke checks follow the real runner path."""
    data = synthetic_dataset(count)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / 'features.npz', **{k: v for k, v in data.items() if k != 'contract'})
    write_json(cache / 'contract.json', data['contract'])
    return cache
