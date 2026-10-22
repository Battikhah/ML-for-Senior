"""Prepare auditable MFCC caches while keeping related recordings together."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_FEATURES = dict(sample_rate=8000, seconds=10, low_hz=100, high_hz=2500,
                        filter_order=4, frames=30, coefficients=13, snr_gate=4.,
                        snr_window_ms=100, variant='reference', n_mels=128,
                        fmin=0., fmax=4000., window='hann', power=2.,
                        dct_type=2, dct_norm='ortho', mel_norm='slaney',
                        log_amin=1e-10, log_top_db=80., fft_size=512, hop=128)


def fingerprint(value) -> str:
    """Make cache and run identities sensitive to every resolved setting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash large recordings incrementally using APIs available on Python 3.10."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    """Replace metadata atomically so an interrupted write cannot look complete."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def environment() -> dict:
    """Capture installed versions without importing heavyweight training libraries."""
    versions = {}
    for name in ['numpy', 'scipy', 'librosa', 'scikit-learn', 'tensorflow', 'tf-keras',
                 'torch', 'mlflow', 'pandas', 'openpyxl', 'tensorflow-model-optimization']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def discover_recordings(root: Path, groups_csv: Path | None = None) -> list[dict]:
    """Use explicit source identities when supplied; filenames alone prove no grouping."""
    root = Path(root)
    if (root / 'field/field/train').is_dir():
        root = root / 'field/field/train'
    mapping = {}
    if groups_csv:
        with Path(groups_csv).open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['filepath'] in mapping:
                    raise ValueError('Duplicate filepath in group metadata')
                mapping[row['filepath']] = row['group']
    records = []
    for folder, label in [('clean', 0), ('infested', 1)]:
        if not (root / folder).is_dir():
            raise ValueError(f'Missing dataset class directory: {root / folder}')
        for path in sorted((root / folder).rglob('*')):
            if path.suffix.lower() != '.wav':
                continue
            relative = path.relative_to(root).as_posix()
            if groups_csv and not mapping.get(relative):
                raise ValueError(f'Missing group for {relative}')
            digest=sha256_file(path)
            records.append(dict(id=relative, filepath=str(path), label=label,
                                sha256=digest, group=mapping.get(relative, relative),
                                grouping='source' if groups_csv else 'file_and_duplicate_only'))
            if len(records) % 500 == 0:
                print(f'Hashed {len(records)} WAV recordings...', flush=True)
    if not records or {r['label'] for r in records} != {0, 1}:
        raise ValueError('Both classes must contain WAV files')
    return records


def make_split(manifest: list[dict], seed: int, exclude_conflicting: bool = False) -> dict:
    """Join all source connections, then optionally exclude ambiguous content from splits."""
    parent = list(range(len(manifest)))

    def find(index):
        """Resolve connected aliases without assigning a recording twice."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    labels_by_hash = {}
    for index, row in enumerate(manifest):
        labels_by_hash.setdefault(row['sha256'], set()).add(row['label'])
        for key in [('hash', row['sha256']), ('group', row['group'])]:
            if key in seen:
                parent[find(index)] = find(seen[key])
            seen[key] = index
    conflicting = {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}
    if conflicting and not exclude_conflicting:
        raise ValueError('Duplicate recording has conflicting class labels')
    # Exclude every copy without guessing its label; retain the connections built above.
    eligible = [i for i, row in enumerate(manifest) if row['sha256'] not in conflicting]
    groups = np.array([find(i) for i in eligible])
    y = np.array([manifest[i]['label'] for i in eligible])
    if len(set(groups)) < 6:
        raise ValueError('Need at least six independent groups for a three-way split')
    best = None
    # Search only labels/groups, never features or model performance.
    for train, held in GroupShuffleSplit(n_splits=200, test_size=.30, random_state=seed).split(y, y, groups):
        if len(set(groups[held])) < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed)
        va, te = next(inner.split(held, y[held], groups[held]))
        parts = [train, held[va], held[te]]
        if any(len(set(y[p])) != 2 for p in parts):
            continue
        score = sum(abs(len(p) / len(y) - ratio) + abs(float(y[p].mean() - y.mean()))
                    for p, ratio in zip(parts, [.7, .15, .15]))
        if best is None or score < best[0]:
            best = score, parts
    if best is None:
        raise ValueError('Cannot split independent groups with both classes; supply more groups')
    return dict(seed=seed, grouping=sorted({r.get('grouping', 'explicit') for r in manifest}),
                conflict_policy='exclude_all_conflicting_content' if exclude_conflicting else 'error',
                excluded_ids=[row['id'] for row in manifest if row['sha256'] in conflicting],
                partitions={name: [manifest[eligible[i]]['id'] for i in p]
                            for name, p in zip(['train', 'val', 'test'], best[1])})


def extract_features(path: Path, feature_config: dict) -> tuple[np.ndarray, dict]:
    """Reproduce the host baseline or an explicitly separate causal feature variant."""
    import librosa
    c = DEFAULT_FEATURES | feature_config
    signal, _ = librosa.load(path, sr=c['sample_rate'], mono=True)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError('Empty or nonfinite recording')
    length = int(c['sample_rate'] * c['seconds'])
    signal = np.pad(signal[:length], (0, max(0, length - len(signal))))
    frequencies = [c['low_hz'], c['high_hz']]
    if c['variant'] == 'reference':
        b, a = butter(c['filter_order'], frequencies, btype='bandpass', fs=c['sample_rate'])
        filtered = filtfilt(b, a, signal)
        fft = hop = length // c['frames']
    elif c['variant'] == 'causal':
        sos = butter(c['filter_order'], frequencies, btype='bandpass', fs=c['sample_rate'], output='sos')
        filtered = sosfilt(sos, signal)
        fft, hop = c['fft_size'], c['hop']
    else:
        raise ValueError('Feature variant must be reference or causal')
    width = max(1, int(c['sample_rate'] * c['snr_window_ms'] / 1000))
    envelope = np.convolve(filtered ** 2, np.ones(width) / width, mode='same')
    snr = float(10 * np.log10(max(float(envelope.max()), 1e-12) / (float(envelope.min()) + 1e-12)))
    mel = librosa.feature.melspectrogram(y=filtered, sr=c['sample_rate'], n_fft=fft,
              hop_length=hop, win_length=fft, window=c['window'], center=False,
              power=c['power'], n_mels=c['n_mels'], fmin=c['fmin'], fmax=c['fmax'],
              htk=False, norm=c['mel_norm'])
    log_mel = librosa.power_to_db(mel, ref=1., amin=c['log_amin'], top_db=c['log_top_db'])
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=c['coefficients'], dct_type=c['dct_type'],
                                norm=c['dct_norm'], lifter=0).T
    if c['variant'] == 'causal':
        mfcc = np.stack([chunk.mean(axis=0) for chunk in np.array_split(mfcc, c['frames'])])
    if mfcc.shape != (c['frames'], c['coefficients']) or not np.isfinite(mfcc).all():
        raise ValueError(f'Invalid MFCC matrix: {mfcc.shape}')
    return mfcc.astype('float32'), dict(snr_db=snr, accepted=c['snr_gate'] is None or snr > c['snr_gate'])


def normalization_stats(training: np.ndarray) -> dict:
    """Fit coefficient scaling on training frames while retaining constant coefficients."""
    mean = training.mean(axis=(0, 1)); scale = training.std(axis=(0, 1))
    return dict(mean=mean.tolist(), scale=np.where(scale < 1e-6, 1., scale).tolist())


def prepare_dataset(root: Path, output: Path, feature_config: dict,
                    groups_csv: Path | None = None) -> Path:
    """Persist split and exclusion evidence, reusing only a fully matching cache."""
    manifest = discover_recordings(root, groups_csv)
    features = DEFAULT_FEATURES | feature_config
    split = make_split(manifest, 42, exclude_conflicting=True)
    inventory = [{k: r[k] for k in ['id', 'sha256', 'label', 'group']} for r in manifest]
    extractor_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = fingerprint(dict(inventory=inventory, features=features, split=split, versions=environment(),
                                extractor_source=extractor_source))
    destination = Path(output) / identity[:16]
    if (destination / 'contract.json').exists():
        load_dataset(destination)
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    excluded_ids = set(split['excluded_ids'])
    conflicts = {}
    for row in manifest:
        if row['id'] in excluded_ids:
            row.update(accepted=False, reason='conflicting_labels', partition='excluded')
            conflicts.setdefault(row['sha256'], []).append(row)
    audit = dict(policy=split['conflict_policy'], total_recordings=len(manifest),
                 eligible_recordings=len(manifest)-len(excluded_ids),
                 conflicting_content_groups=len(conflicts), excluded_recordings=len(excluded_ids),
                 excluded_by_class={str(label):sum(row['label']==label for rows in conflicts.values()
                                                  for row in rows) for label in [0,1]})
    # Persist the audit before expensive feature extraction or a possible gate failure.
    write_json(destination / 'data_audit.json', audit)
    write_json(destination / 'conflicting_recordings.json', list(conflicts.values()))
    print(f'Dataset audit: {len(excluded_ids)} conflicting-label files excluded; '
          f'{audit["eligible_recordings"]} recordings eligible. Report: {destination / "data_audit.json"}')
    arrays = {}; duplicates = set()
    by_id = {r['id']: r for r in manifest}
    for name, ids in split['partitions'].items():
        xs, ys, kept = [], [], []
        for index, row_id in enumerate(ids, 1):
            if index == 1 or index % 250 == 0 or index == len(ids):
                print(f'MFCC preparation — {name}: {index}/{len(ids)} recordings', flush=True)
            row = by_id[row_id]; row['partition'] = name
            if row['sha256'] in duplicates:
                row.update(accepted=False, reason='duplicate_content'); continue
            duplicates.add(row['sha256'])
            try:
                matrix, meta = extract_features(Path(row['filepath']), features)
                row.update(meta, reason='accepted' if meta['accepted'] else 'snr_gate')
                if meta['accepted']:
                    xs.append(matrix); ys.append(row['label']); kept.append(row_id)
            except Exception as error:
                row.update(accepted=False, reason='extraction_error', error=str(error))
        if set(ys) != {0, 1}:
            write_json(destination / 'manifest.json', manifest)
            raise ValueError(f'{name} lacks both classes after preprocessing '
                             f'(clean={ys.count(0)}, infested={ys.count(1)}); '
                             f'inspect {destination / "manifest.json"}')
        arrays[f'X_{name}'] = np.stack(xs)
        arrays[f'y_{name}'] = np.array(ys, dtype='int64')
        arrays[f'ids_{name}'] = np.array(kept)
    stats = normalization_stats(arrays['X_train'])
    coverage={}
    for name in ['train','val','test']:
        rows=[r for r in manifest if r['partition']==name]
        accepted=sum(bool(r['accepted']) for r in rows)
        coverage[name]=dict(total=len(rows),accepted=accepted,accepted_fraction=accepted/len(rows),
            exclusions={reason:sum(r['reason']==reason for r in rows)
                        for reason in ['snr_gate','duplicate_content','extraction_error']},
            by_class={str(label):dict(total=sum(r['label']==label for r in rows),
                      accepted=sum(r['label']==label and bool(r['accepted']) for r in rows))
                      for label in [0,1]})
    contract = dict(dataset_id=identity, features=features, split=split, normalization=stats, data_audit=audit,
                    synthetic=False, versions=environment(),coverage=coverage,
                    feature_parity='host_only_unverified_on_device')
    temporary = destination / 'features.tmp.npz'
    np.savez_compressed(temporary, **arrays); temporary.replace(destination / 'features.npz')
    contract['payload_sha256']=sha256_file(destination/'features.npz')
    contract['extractor_source']=extractor_source
    write_json(destination / 'manifest.json', manifest)
    write_json(destination / 'contract.json', contract)
    return destination


def load_dataset(cache: Path) -> dict:
    """Load numeric caches without pickle and check contracts before model training."""
    cache = Path(cache)
    contract = json.loads((cache / 'contract.json').read_text())
    if contract.get('payload_sha256'):
        if sha256_file(cache/'features.npz')!=contract['payload_sha256']:
            raise ValueError('Feature cache payload changed after preparation')
    with np.load(cache / 'features.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    shape = (contract['features']['frames'], contract['features']['coefficients'])
    for part in ['train', 'val', 'test']:
        x, y = data[f'X_{part}'], data[f'y_{part}']
        if x.shape != (len(y), *shape) or set(y.tolist()) != {0, 1} or not np.isfinite(x).all():
            raise ValueError(f'Invalid {part} cache')
    data['contract'] = contract
    return data
