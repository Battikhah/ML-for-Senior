"""Materialize readable experiment recipes without copying training notebooks."""
from pathlib import Path
from .data import write_json


def generate_catalog(destination: Path) -> list[Path]:
    """Keep a broad staged catalog deterministic and individually inspectable."""
    configs = []

    def add(suite, family, name, **settings):
        """Use explicit names so exported leaderboards remain understandable."""
        configs.append(dict(id=name, suite=suite, family=family, **settings))

    for family in ['reference_a', 'reference_b']:
        add('baseline', family, family, normalization='none', patience=120)
    for family in ['logistic', 'svm', 'forest', 'histogram']:
        for norm in ['none', 'coefficient']:
            add('baseline', family, f'{family}_{norm}', normalization=norm)
    for family in ['mlp', 'temporal', 'cnn', 'residual', 'depthwise']:
        for width in [8, 16, 32, 64]:
            add('quality', family, f'{family}_w{width}', width=width)
        for width in [4, 8, 12]:
            add('compact', family, f'tiny_{family}_w{width}', width=width, target='local', dropout=.1)
        variants = [dict(learning_rate=.0003), dict(learning_rate=.003), dict(dropout=.1),
                    dict(dropout=.5), dict(class_weight=True), dict(noise=.05),
                    dict(time_mask=3), dict(normalization='none'),
                    dict(features={'snr_gate': None}), dict(features={'variant': 'causal'})]
        for index, variant in enumerate(variants):
            add('ablation', family, f'{family}_ablation_{index:02}', **variant)
    for family in ['mlp', 'temporal', 'depthwise']:
        for frames, coefficients in [(60, 13), (60, 20)]:
            add('local_features', family, f'{family}_{frames}x{coefficients}', target='local',
                width=8, features={'frames': frames, 'coefficients': coefficients})
    for family in ['mlp', 'cnn', 'depthwise']:
        for width in [4, 8]:
            for temperature in [2., 4.]:
                for alpha in [.25, .75]:
                    add('distillation', family, f'kd_{family}_w{width}_t{int(temperature)}_a{int(alpha*100)}',
                        target='local', width=width, dropout=.1, method='kd',
                        temperature=temperature, alpha=alpha)
    for seed in [43, 44]:
        for family in ['cnn', 'residual', 'depthwise']:
            add('repeat', family, f'{family}_w16_s{seed}', seed=seed)
    paths = []
    for config in configs:
        path = Path(destination) / config['suite'] / (config['id'] + '.json')
        write_json(path, config); paths.append(path)
    return paths
