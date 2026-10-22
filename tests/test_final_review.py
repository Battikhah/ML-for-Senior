"""Protect cohort matching, feature identity and train-only calibration in the review campaign."""
import numpy as np
import pytest
from scripts.review_campaign import cohort_digest, fit_scaling, resolve_audio, review_metrics


def test_cohort_identity_keeps_order_and_never_needs_test_labels():
    """A changed evaluation order must not silently reuse predictions."""
    assert cohort_digest(['a','b']) != cohort_digest(['b','a'])
    assert cohort_digest(['a','b']) == cohort_digest(np.array(['a','b']))


def test_scaling_uses_training_only_and_constant_channels_are_safe():
    """Extreme validation values must not alter fitted normalization."""
    x=np.array([[[1.,4.],[3.,4.]],[[5.,4.],[7.,4.]]],dtype='float32')
    mean,scale=fit_scaling(x)
    np.testing.assert_allclose(mean,[4,4]);assert scale[1]==1
    assert np.isfinite((np.array([999,4])-mean)/scale).all()


def test_audio_resolution_requires_exact_recording_identity(tmp_path):
    """Never substitute a same-named WAV from a different class or folder."""
    p=tmp_path/'field/field/train/infested/folder_1/x.wav';p.parent.mkdir(parents=True);p.touch()
    assert resolve_audio(tmp_path,'infested/folder_1/x.wav')==p
    with pytest.raises(FileNotFoundError):resolve_audio(tmp_path,'clean/folder_1/x.wav')


def test_recorded_metrics_retain_false_negatives():
    """RPW mistakes must remain recoverable from the canonical table."""
    m=review_metrics([0,0,1,1],[.1,.8,.4,.9],.5)
    assert m['false_negatives']==1 and m['false_positives']==1
    assert m['recall']==.5 and m['fpr']==.5


def test_cache_reader_leaves_test_members_unopened(tmp_path):
    """A malformed test member must not prevent train/validation review preparation."""
    import hashlib,json
    from scripts.review_campaign import read_cache
    folder=tmp_path/'cache/a';folder.mkdir(parents=True)
    values={key:np.array([1]) for key in ['X_train','y_train','ids_train','X_val','y_val','ids_val']}
    values['y_test']=np.array([object()],dtype=object)
    np.savez(folder/'features.npz',**values)
    digest=hashlib.sha256((folder/'features.npz').read_bytes()).hexdigest()
    (folder/'contract.json').write_text(json.dumps(dict(dataset_id='a',payload_sha256=digest)))
    data=read_cache(tmp_path,'a')
    assert 'y_test' not in data
    (folder/'features.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):read_cache(tmp_path,'a')


def test_notebook_preserves_original_and_avoids_duplicate_workers():
    """Inspect executable cells for the required isolation/reattachment contract."""
    import ast,json
    from pathlib import Path
    path=Path(__file__).parents[1]/'TreeVibes_Final_Professor_Review.ipynb'
    nb=json.loads(path.read_text())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    ast.parse(code)
    assert 'if ACTIVE:' in code and 'No duplicate worker' in code
    assert 'final_review_2026-10-20' in code
    assert 'unrestricted_inception_resnet_v2' in code
    assert 'y_test' not in code


def test_recall_operating_point_uses_highest_threshold_meeting_target():
    """A sensitivity request must expose its false-alarm cost, with no test selection."""
    from scripts.review_results import recall_operating_point
    m=recall_operating_point(np.array([0,0,1,1]),np.array([.1,.8,.4,.9]),1.)
    assert m['threshold']==.4 and m['recall']==1. and m['false_positives']==1
