"""Protect feature contracts and recording isolation from accidental leakage."""
import json
import numpy as np
import pytest
from scipy.io import wavfile


def test_features_have_exact_shape_and_silence_is_gated(tmp_path):
    """Padding and gate diagnostics must remain finite for short/silent recordings."""
    from treevibes.data import extract_features, DEFAULT_FEATURES
    for size in [80, 80000, 90000]:
        path = tmp_path / f'{size}.wav'
        wavfile.write(path, 8000, np.zeros(size, dtype=np.int16))
        features, meta = extract_features(path, DEFAULT_FEATURES)
        assert features.shape == (30, 13)
        assert np.isfinite(features).all() and np.isfinite(meta['snr_db'])
        assert not meta['accepted']


def test_duplicate_connected_groups_never_cross_splits():
    """Content aliases connecting sessions must be kept in the same component."""
    from treevibes.data import make_split
    records = [dict(id=str(i), label=i % 2, sha256=str(i), group=f'g{i}') for i in range(40)]
    records[0]['sha256'] = records[2]['sha256'] = 'shared'
    records[2]['group'] = records[4]['group'] = 'linked'
    split = make_split(records, 42)
    assigned = {item: part for part, ids in split['partitions'].items() for item in ids}
    assert assigned['0'] == assigned['2'] == assigned['4']
    assert split == make_split(records, 42)
    for ids in split['partitions'].values():
        assert {records[int(i)]['label'] for i in ids} == {0, 1}


def test_impossible_split_and_conflicting_duplicate_labels_fail():
    """A fallback file split would hide an invalid evaluation design."""
    from treevibes.data import make_split
    with pytest.raises(ValueError):
        make_split([dict(id=str(i), label=i % 2, sha256=str(i), group='one') for i in range(8)], 42)
    with pytest.raises(ValueError, match='conflicting'):
        make_split([dict(id='a', label=0, sha256='same', group='a'),
                    dict(id='b', label=1, sha256='same', group='b')], 42)


def test_normalization_is_training_only_and_cache_contract_changes():
    """Extreme held-out values must not change training-fitted statistics."""
    from treevibes.data import normalization_stats, fingerprint
    stats = normalization_stats(np.array([[[1., 2.]], [[3., 6.]]]))
    np.testing.assert_allclose(stats['mean'], [2, 4])
    np.testing.assert_allclose(stats['scale'], [1, 2])
    assert fingerprint({'frames': 30}) != fingerprint({'frames': 60})


def test_config_rejects_remote_richer_features(tmp_path):
    """Local resolution changes cannot slip into the transmission-limited track."""
    from treevibes.config import load_config
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'id': 'x', 'family': 'mlp', 'target': 'remote', 'features': {'frames': 60, 'coefficients': 20}}))
    with pytest.raises(ValueError, match='30'):
        load_config(path)
    config = json.loads(path.read_text()); config['target'] = 'local'
    path.write_text(json.dumps(config))
    assert load_config(path)['features']['frames'] == 60


@pytest.mark.parametrize('patch',[{'width':1.5},{'seed':-1},{'method':'pretend'},
                                {'features':{'frames':30.5}}])
def test_invalid_recipe_fails_before_gpu_work(tmp_path,patch):
    """Fractional dimensions and unknown training modes must not fail halfway through a run."""
    from treevibes.config import load_config
    p=tmp_path/'invalid.json';p.write_text(json.dumps(dict(id='bad',family='mlp')|patch))
    with pytest.raises(ValueError):
        load_config(p)


def test_actual_audio_preparation_and_cache_integrity(tmp_path):
    """Audit real WAV decoding/extraction and reject a modified persisted feature payload."""
    from treevibes.data import prepare_dataset,load_dataset,DEFAULT_FEATURES
    rng=np.random.default_rng(4)
    root=tmp_path/'audio'
    for label in ['clean','infested']:
        (root/label).mkdir(parents=True)
        for i in range(12):
            wavfile.write(root/label/f'{i}.wav',8000,rng.normal(0,.05,8000).astype('float32'))
    destination=prepare_dataset(root,tmp_path/'cache',DEFAULT_FEATURES)
    assert load_dataset(destination)['X_train'].shape[1:]==(30,13)
    contract=load_dataset(destination)['contract']
    assert contract['coverage']['train']['accepted']==len(load_dataset(destination)['y_train'])
    assert sum(c['total'] for c in contract['coverage'].values())==24
    manifest=json.loads((destination/'manifest.json').read_text())
    assert len(manifest)==24 and all(r['accepted'] for r in manifest)
    assert prepare_dataset(root,tmp_path/'cache',DEFAULT_FEATURES)==destination
    with (destination/'features.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'):
        load_dataset(destination)


def test_conflict_exclusion_preserves_source_connections():
    """Removing ambiguous recordings must not let connected sessions cross partitions."""
    from treevibes.data import make_split
    records = [dict(id=str(i), label=i % 2, sha256=str(i), group=f'g{i}') for i in range(40)]
    records += [dict(id='ambiguous-clean', label=0, sha256='ambiguous', group='g0'),
                dict(id='ambiguous-infested', label=1, sha256='ambiguous', group='g2')]
    split = make_split(records, 42, exclude_conflicting=True)
    assigned = {item: part for part, ids in split['partitions'].items() for item in ids}
    assert set(assigned) == {str(i) for i in range(40)}
    assert assigned['0'] == assigned['2']
    assert split['excluded_ids'] == ['ambiguous-clean', 'ambiguous-infested']


def test_preparation_excludes_both_conflicting_copies_and_records_evidence(tmp_path):
    """A cross-label copy must be audited and excluded without deleting either source."""
    from treevibes.data import prepare_dataset, load_dataset, DEFAULT_FEATURES
    rng = np.random.default_rng(9)
    root = tmp_path / 'audio'
    for label in ['clean', 'infested']:
        (root / label).mkdir(parents=True)
        for i in range(12):
            wavfile.write(root / label / f'{i}.wav', 8000, rng.normal(0, .05, 8000).astype('float32'))
    first = root / 'clean' / 'ambiguous.wav'
    second = root / 'infested' / 'ambiguous.wav'
    wavfile.write(first, 8000, rng.normal(0, .05, 8000).astype('float32'))
    original = first.read_bytes()
    second.write_bytes(original)
    destination = prepare_dataset(root, tmp_path / 'cache', DEFAULT_FEATURES)
    data = load_dataset(destination)
    audit = data['contract']['data_audit']
    assert audit['conflicting_content_groups'] == 1 and audit['excluded_recordings'] == 2
    assert audit['total_recordings'] == 26 and audit['eligible_recordings'] == 24
    assert audit['policy'] == 'exclude_all_conflicting_content'
    manifest = json.loads((destination / 'manifest.json').read_text())
    excluded = [r for r in manifest if r['reason'] == 'conflicting_labels']
    assert {r['id'] for r in excluded} == {'clean/ambiguous.wav', 'infested/ambiguous.wav'}
    assert all(not r['accepted'] and r['partition'] == 'excluded' for r in excluded)
    ids = {item for part in ['train', 'val', 'test'] for item in data[f'ids_{part}'].tolist()}
    assert len(ids) == 24 and not ({r['id'] for r in excluded} & ids)
    report = json.loads((destination / 'conflicting_recordings.json').read_text())
    assert len(report) == 1 and len(report[0]) == 2
    assert first.read_bytes() == original and second.read_bytes() == original
    assert prepare_dataset(root, tmp_path / 'cache', DEFAULT_FEATURES) == destination
