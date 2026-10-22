"""Reproduce persistence and artifact-integrity failures found during final review."""
import json
import sys
import types
from pathlib import Path
import pytest


def test_custom_colab_output_still_mounts_drive_and_keeps_sqlite_local(tmp_path,monkeypatch):
    """Overriding the results folder must not bypass Colab's persistence setup."""
    from treevibes.notebook_setup import setup_output
    mounted=[]
    def mount(path):
        """Represent the external authorization boundary without accessing an account."""
        mounted.append(path)
    module=types.ModuleType('google.colab');module.drive=types.SimpleNamespace(mount=mount)
    monkeypatch.setitem(sys.modules,'google.colab',module)
    monkeypatch.setenv('COLAB_RELEASE_TAG','release')
    monkeypatch.setenv('TREEVIBES_OUTPUT',str(tmp_path/'custom'))
    # Register both transitions so pytest also undoes setup_output's environment mutation.
    monkeypatch.setenv('TREEVIBES_TRACKING_DIR','test-restore-marker')
    monkeypatch.delenv('TREEVIBES_TRACKING_DIR')
    output,_=setup_output(False)
    import os
    assert mounted==['/content/drive'] and output==tmp_path/'custom'
    assert os.environ['TREEVIBES_TRACKING_DIR']=='/content/treevibes-tracking'


def test_incomplete_attempt_cannot_block_retry_or_report(tmp_path):
    """An interrupted recovery copy may leave an orphan directory with no metadata."""
    from treevibes.tracking import run_experiment,export_leaderboard
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    cache=save_synthetic(tmp_path/'cache',20)
    recipe=tmp_path/'control.json';write_json(recipe,dict(id='recover',family='logistic'))
    result=run_experiment(recipe,cache,tmp_path/'out')
    attempt=Path(result['run_dir'])
    write_json(attempt/'status.json',dict(state='failed',reason='runtime interrupted'))
    orphan=attempt.parent/'attempt_002';orphan.mkdir()
    # A second malformed initialization has a status file but no config/provenance.
    malformed=attempt.parent/'attempt_004';malformed.mkdir()
    write_json(malformed/'status.json',dict(state='initializing'))
    retry=run_experiment(recipe,cache,tmp_path/'out')
    assert Path(retry['run_dir']).name=='attempt_005'
    book=export_leaderboard(tmp_path/'out',tmp_path/'report.xlsx')
    import pandas as pd
    failures=pd.read_excel(book,sheet_name='Failures')
    assert len(failures)>=3


def test_tampered_teacher_and_export_parent_are_rejected(tmp_path):
    """A completed status cannot override a mismatch in recorded checkpoint bytes."""
    from treevibes.tracking import run_experiment,valid_completed
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json,load_dataset
    from treevibes.train import _teacher
    from treevibes.export import export_model
    cache=save_synthetic(tmp_path/'cache',20)
    recipe=tmp_path/'teacher.json';write_json(recipe,dict(id='teacher',family='mlp',width=4,epochs=1))
    result=run_experiment(recipe,cache,tmp_path/'out');parent=Path(result['run_dir'])
    # Mutate the recorded digest so the real, still-loadable model fails integrity checks.
    record=json.loads((parent/'result.json').read_text());record['artifact_hashes']['best.keras']='0'*64
    write_json(parent/'result.json',record)
    assert not valid_completed(parent)
    contract=json.loads((parent/'input_contract.json').read_text())
    with pytest.raises(ValueError,match='valid|integrity'):
        _teacher(parent,contract)
    with pytest.raises(ValueError,match='valid|integrity'):
        export_model(parent,load_dataset(cache),'ptq',tmp_path/'export')


def test_sha256_does_not_require_python_311_api(tmp_path,monkeypatch):
    """The advertised Python 3.10 path must use APIs present in that version."""
    import hashlib
    from treevibes.tracking import file_hash
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    p=tmp_path/'abc';p.write_bytes(b'abc')
    assert file_hash(p)=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
