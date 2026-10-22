"""Prove explicit selection and test-set locking through public entry points."""
import json
import pytest


def test_dry_run_never_starts_training(capsys):
    """Previewing a suite must not require a dataset, GPU or teacher checkpoint."""
    from treevibes.cli import main
    assert main(['train','--suite','screening','--dry-run'])==0
    output=capsys.readouterr().out
    assert 'cnn_w16' in output and 'logistic_coefficient' in output


def test_test_evaluation_requires_unchanged_locked_artifacts(tmp_path):
    """Selected weights and validation threshold cannot change after finalist locking."""
    from treevibes.finalize import lock_finalists, evaluate_final
    from treevibes.tracking import run_experiment
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json,load_dataset
    cache=save_synthetic(tmp_path/'cache',40)
    config=tmp_path/'config.json';write_json(config,dict(id='lock_control',family='logistic'))
    result=run_experiment(config,cache,tmp_path/'out')
    from pathlib import Path
    run=Path(result['run_dir']);lock=tmp_path/'finalists.json'
    lock_finalists([run],[],lock)
    scores=evaluate_final(lock,load_dataset(cache),tmp_path/'final')
    assert scores['results'][0]['test']['count']==10
    assert evaluate_final(lock,load_dataset(cache),tmp_path/'final')==scores
    model=run/'best.joblib';model.write_bytes(model.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):
        evaluate_final(lock,load_dataset(cache),tmp_path/'another_final')
