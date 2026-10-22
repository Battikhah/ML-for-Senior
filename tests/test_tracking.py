"""Exercise persisted attempts and reports through actual runner side effects."""
import json
import sqlite3
import pytest
from pathlib import Path


def test_runner_skips_only_valid_completed_runs_and_keeps_failures(tmp_path):
    """A lost checkpoint or failed retry must remain visible rather than becoming success."""
    from treevibes.tracking import run_experiment, export_leaderboard
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    cache=save_synthetic(tmp_path/'cache', 40)
    contract=json.loads((cache/'contract.json').read_text())
    audit=dict(policy='exclude_all_conflicting_content',conflicting_content_groups=1,excluded_recordings=2)
    write_json(cache/'contract.json',contract|{'data_audit':audit})
    path=tmp_path/'logistic.json'
    write_json(path, dict(id='control', family='logistic'))
    result=run_experiment(path, cache, tmp_path/'out')
    assert result['data_audit']==audit
    import mlflow
    artifact_uri=mlflow.tracking.MlflowClient().get_run(result['mlflow_run_id']).info.artifact_uri
    assert artifact_uri.startswith((tmp_path/'out').resolve().as_uri())
    again=run_experiment(path, cache, tmp_path/'out')
    assert again['run_id']==result['run_id'] and again['skipped']
    directory=Path(result['run_dir'])
    (directory/'best.joblib').unlink()
    retry=run_experiment(path, cache, tmp_path/'out')
    assert retry['run_id'] != result['run_id']
    assert json.loads((directory/'status.json').read_text())['state']=='invalidated'
    write_json(path, dict(id='bad_teacher', family='mlp', method='kd', epochs=1))
    with pytest.raises(ValueError):
        run_experiment(path, cache, tmp_path/'out')
    import pandas as pd
    book=export_leaderboard(tmp_path/'out', tmp_path/'leaderboard.xlsx')
    sheets=pd.read_excel(book, sheet_name=None)
    assert set(sheets)=={'Runs','Experiments','Failures','Board measurements'}
    assert 'failed' in set(sheets['Failures']['state'])
    assert retry['run_id'] in set(sheets['Runs']['run_id'])
    row=sheets['Runs'].set_index('run_id').loc[retry['run_id']]
    assert row['label_conflict_groups']==1 and row['label_conflict_files_excluded']==2
    assert sheets['Board measurements'].empty


def test_online_backup_restores_committed_data(tmp_path):
    """Copying a live SQLite file directly can lose WAL-backed records."""
    from treevibes.tracking import backup_tracking
    source=tmp_path/'live.db'
    with sqlite3.connect(source) as connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE runs (id TEXT)')
        connection.execute("INSERT INTO runs VALUES ('committed')")
        connection.commit()
        target=backup_tracking(source,tmp_path/'backup.db')
        with sqlite3.connect(target) as restored:
            assert restored.execute('SELECT id FROM runs').fetchall()==[('committed',)]


def test_tracking_schema_failure_does_not_discard_training(tmp_path, monkeypatch):
    """An optional tracker must never turn a fitted model into a failed attempt."""
    import mlflow
    from treevibes.tracking import run_experiment, valid_completed
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    def broken_tracker(*args, **kwargs):
        """Represent the incompatible database reported by Colab."""
        raise RuntimeError('found version dc11669786a5, expected b7e4c1a90f23')
    monkeypatch.setattr(mlflow, 'get_experiment_by_name', broken_tracker)
    config=tmp_path/'control.json'
    write_json(config, dict(id='control',family='logistic'))
    output=tmp_path/'out';output.mkdir()
    legacy=output/'tracking_backup.db';legacy.write_bytes(b'legacy database must remain untouched')
    result=run_experiment(config,save_synthetic(tmp_path/'cache',40),output)
    assert valid_completed(Path(result['run_dir']))
    assert 'dc11669786a5' in result['tracking_warning']
    assert result['mlflow_run_id'] is None
    assert legacy.read_bytes()==b'legacy database must remain untouched'
    assert run_experiment(config,tmp_path/'cache',output)['skipped']


@pytest.mark.parametrize("legacy_source", [
    "cc7d49dcd4bdaa8e04cb3ba6af742833b218607ad9eb4cc9fc1198b3363cda87",
    "d6c6838950a09a4357c7194cf8b66e0dcc0634c76f547a93be9fc52d6bd0e8b8"])
def test_infrastructure_patch_reuses_legacy_run(tmp_path, monkeypatch, legacy_source):
    """Only the explicitly compatible release may reuse runs across source hashes."""
    import treevibes.tracking as tracking
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    path=tmp_path/'control.json';write_json(path,dict(id='control',family='logistic'))
    cache=save_synthetic(tmp_path/'cache',40)
    with monkeypatch.context() as old:
        old.setattr(tracking,'source_fingerprint',lambda:legacy_source)
        result=tracking.run_experiment(path,cache,tmp_path/'out')
    again=tracking.run_experiment(path,cache,tmp_path/'out')
    assert again['run_id']==result['run_id'] and again['skipped']


def test_failed_legacy_attempt_restores_committed_checkpoint(tmp_path, monkeypatch):
    """Infrastructure recovery keeps committed epoch pointers paired with their model bytes."""
    import zipfile
    import treevibes.tracking as tracking
    import treevibes.train as training
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    path=tmp_path/'control.json';write_json(path,dict(id='control',family='logistic'))
    cache=save_synthetic(tmp_path/'cache',40)
    actual_train=training.train_run
    def interrupted(config, dataset, directory, teacher):
        """Persist a checkpoint at the real training boundary, then emulate disconnect."""
        with zipfile.ZipFile(directory/'epoch.keras','w') as archive:
            archive.writestr('committed','epoch bytes')
        write_json(directory/'epoch_state.json',{'checkpoint':'epoch.keras','best_checkpoint':'epoch.keras'})
        raise RuntimeError('disconnected')
    with monkeypatch.context() as old:
        old.setattr(tracking,'source_fingerprint',lambda:'cc7d49dcd4bdaa8e04cb3ba6af742833b218607ad9eb4cc9fc1198b3363cda87')
        old.setattr(training,'train_run',interrupted)
        with pytest.raises(RuntimeError,match='disconnected'):
            tracking.run_experiment(path,cache,tmp_path/'out')
    def recovered(config, dataset, directory, teacher):
        """Check recovery before using the actual inexpensive classifier fit."""
        assert json.loads((directory/'epoch_state.json').read_text())['checkpoint']=='epoch.keras'
        with zipfile.ZipFile(directory/'epoch.keras') as archive:
            assert archive.read('committed')==b'epoch bytes'
        return actual_train(config,dataset,directory,teacher)
    monkeypatch.setattr(training,'train_run',recovered)
    result=tracking.run_experiment(path,cache,tmp_path/'out')
    assert tracking.valid_completed(Path(result['run_dir']))
    assert (Path(result['run_dir'])/'recovery.json').is_file()
