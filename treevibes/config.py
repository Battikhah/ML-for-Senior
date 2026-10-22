"""Resolve explicit experiment settings into portable, fingerprintable contracts."""
import copy
import json
import math
from pathlib import Path
from .data import DEFAULT_FEATURES

FAMILIES = {'logistic', 'svm', 'forest', 'histogram', 'mlp', 'temporal', 'cnn',
            'residual', 'depthwise', 'reference_a', 'reference_b'}
DEFAULTS = dict(seed=42, target='remote', normalization='coefficient', width=16,
                dropout=0.3, learning_rate=0.001, batch_size=16, epochs=120,
                patience=15, class_weight=False, noise=0., time_mask=0,
                temperature=3., alpha=0.5, suite='quality')


def load_config(path: Path) -> dict:
    """Reject invalid comparisons before spending GPU time or touching artifacts."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw,dict) or not isinstance(raw.get('features',{}),dict):
        raise ValueError('Configuration and features must be JSON objects')
    unknown = set(raw) - set(DEFAULTS) - {'id', 'family', 'features', 'teacher', 'method'}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = copy.deepcopy(DEFAULTS)
    config.update(raw)
    config['features'] = DEFAULT_FEATURES | raw.get('features', {})
    if not config.get('id') or not all(c.isalnum() or c in '-_' for c in config['id']):
        raise ValueError('id must contain only letters, digits, underscores and hyphens')
    if config.get('family') not in FAMILIES:
        raise ValueError('Unknown model family')
    if config['target'] not in {'local', 'remote'}:
        raise ValueError('target must be local or remote')
    f = config['features']
    if set(f) != set(DEFAULT_FEATURES):
        raise ValueError('Unknown feature setting')
    if config['target'] == 'remote' and (f['frames'], f['coefficients']) != (30, 13):
        raise ValueError('Remote inputs must remain 30 by 13')
    if config['normalization'] not in {'none', 'coefficient'}:
        raise ValueError('Unknown normalization')
    if config.get('method','supervised') not in {'supervised','kd'}:
        raise ValueError('Unknown training method')
    for key in ['width','batch_size','epochs','patience','time_mask','seed']:
        if type(config[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if not 0<=config['seed']<2**31:
        raise ValueError('seed must be between 0 and 2**31-1')
    for key in ['frames','coefficients','sample_rate','filter_order','n_mels','fft_size','hop']:
        if type(f[key]) is not int or f[key]<=0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['width', 'batch_size', 'epochs', 'patience', 'temperature', 'learning_rate']:
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= config['alpha'] <= 1 or not 0 <= config['dropout'] < 1:
        raise ValueError('Invalid distillation weight or dropout')
    if config['noise'] < 0 or not 0 <= config['time_mask'] < f['frames']:
        raise ValueError('Invalid augmentation')
    if f['frames'] < 8 or not 1 <= f['coefficients'] <= f['n_mels']:
        raise ValueError('Invalid feature dimensions')
    if not 0 < f['low_hz'] < f['high_hz'] < f['sample_rate'] / 2:
        raise ValueError('Bandpass frequencies must lie below Nyquist')
    return config
