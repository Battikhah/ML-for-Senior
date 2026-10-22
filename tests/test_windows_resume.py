"""Check cross-machine recovery without requiring the user's private Drive or Windows PC."""
import json
from pathlib import Path
import pytest


def test_import_preserves_verified_models_and_reuses_campaign(tmp_path,monkeypatch):
    """Path relocation may change metadata but must not retrain verified imported models."""
    from scripts.windows_support import import_experiments, seed_campaign
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    from treevibes.tracking import run_experiment,valid_completed,file_hash
    import treevibes.campaign as campaign
    original=tmp_path/'original';cache=save_synthetic(original/'cache/features',40)
    contract=json.loads((cache/'contract.json').read_text());contract['synthetic']=False
    write_json(cache/'contract.json',contract)
    config=tmp_path/'control.json';write_json(config,dict(id='control',family='logistic',suite='baseline',epochs=1,patience=1))
    result=run_experiment(config,cache,original)
    model_hash=file_hash(Path(result['run_dir'])/'best.joblib')
    write_json(original/'notebook_settings.json',dict(output=str(original)))
    write_json(original/'active_cache.json',dict(path=str(cache)))
    (original/'tracking_backup-mlflow-3.14.0.db').write_bytes(b'original tracking history')
    imported=tmp_path/'imported'
    receipt=import_experiments(original,imported)
    assert receipt['verified_completed_runs']==1
    assert (imported/'imported_tracking/tracking_backup-mlflow-3.14.0.db').read_bytes()==b'original tracking history'
    assert not (imported/'tracking_backup-mlflow-3.14.0.db').exists()
    run=next(imported.glob('runs/*/*/attempt_*'))
    assert valid_completed(run) and file_hash(run/'best.joblib')==model_hash
    assert json.loads((run/'result.json').read_text())['run_dir']==str(run)
    assert valid_completed(Path(result['run_dir']))
    assert import_experiments(original,imported)==receipt
    project=tmp_path/'project';(project/'experiments/baseline').mkdir(parents=True)
    (project/'experiments/baseline/control.json').write_bytes(config.read_bytes())
    settings=dict(mode='full',output=str(imported),epochs=1,export_top_n=1,qat_epochs=1)
    seed_campaign(settings,project)
    def unexpected(*args,**kwargs):
        """Any download or training job would mean imported recovery was bypassed."""
        raise AssertionError('Saved work was not reused')
    monkeypatch.setattr(campaign,'_job',unexpected)
    monkeypatch.setattr(campaign,'_raw_dataset',unexpected)
    summary=campaign.run_campaign(settings,project)
    assert summary['completed_runs']==1 and not summary['failures']


def test_import_does_not_bless_damaged_artifacts(tmp_path):
    """A downloaded completed flag cannot override a model checksum mismatch."""
    from scripts.windows_support import import_experiments
    from treevibes.data import write_json
    source=tmp_path/'original';run=source/'runs/bad/identity/attempt_001';run.mkdir(parents=True)
    (run/'best.keras').write_bytes(b'corrupt')
    write_json(run/'status.json',dict(state='completed'))
    write_json(run/'result.json',dict(artifact_hashes={'best.keras':'wrong'},run_dir=str(run)))
    write_json(source/'notebook_settings.json',dict(output=str(source)))
    receipt=import_experiments(source,tmp_path/'output')
    assert receipt['verified_completed_runs']==0 and len(receipt['invalid_completed_runs'])==1
    assert json.loads((tmp_path/'output/runs/bad/identity/attempt_001/status.json').read_text())['state']=='invalidated'
    assert json.loads((run/'status.json').read_text())['state']=='completed'


def test_windows_python_paths_and_private_drive_command(tmp_path):
    """Use Windows venv executables and readonly OAuth without printing saved tokens."""
    from scripts.windows_support import environment_python, drive_auth_command
    assert str(environment_python(tmp_path,windows=True)).endswith('Scripts/python.exe')
    command=drive_auth_command(Path('rclone.exe'),tmp_path/'auth.conf','folder-id')
    assert 'drive.readonly' in command and '--no-output' in command and 'folder-id' in command


def test_windows_logging_handles_unicode_and_child_failures(tmp_path):
    """Remote Windows code pages must not turn a successful worker into a decoding error."""
    import sys
    from scripts.windows_support import run_logged
    path=tmp_path/'worker.log'
    run_logged([sys.executable,'-c',"print('MFCC — 测试')"],path)
    assert path.read_text(encoding='utf-8').strip()=='MFCC — 测试'
    with pytest.raises(RuntimeError,match='actual failure'):
        run_logged([sys.executable,'-c',"raise RuntimeError('actual failure')"],path)


def test_windows_notebook_keeps_private_recovery_before_training(tmp_path):
    """The standalone file must carry Windows paths, private import and GPU verification."""
    import ast
    import nbformat
    from scripts.package_windows import build_windows_notebook
    from treevibes.packaging import build_colab_archive
    root=Path.cwd();archive=build_colab_archive(root,tmp_path/'source.zip')
    path=build_windows_notebook(root,archive,tmp_path/'windows.ipynb')
    notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    for source in cells:ast.parse(source)
    assert len(cells)==6
    assert 'D:/ML/TreeVibesWindows' in cells[0]
    assert 'download_drive' in cells[2]
    assert 'prepare_windows_workers' in cells[3]
    assert 'import_experiments' in cells[4] and 'seed_campaign' in cells[4]
    assert 'torch.cuda.is_available' in cells[3]
    assert 'google.colab' not in '\n'.join(cells)


def test_private_download_checks_binary_and_only_copies_readonly(tmp_path,monkeypatch):
    """Exercise bootstrap and OAuth command assembly without real credentials or network."""
    import hashlib,io,zipfile
    import scripts.windows_support as windows
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        archive.writestr('rclone-v1.99.0-windows-amd64/rclone.exe',b'fake executable')
    payload=buffer.getvalue();checksum=hashlib.sha256(payload).hexdigest()
    def response(url,**kwargs):
        """Supply verified release metadata at the network boundary."""
        return io.BytesIO(b'rclone v1.99.0\n' if url.endswith('version.txt') else
                           f'{checksum}  rclone-v1.99.0-windows-amd64.zip\n'.encode())
    def retrieve(url,path):
        """Represent the official archive transfer."""
        Path(path).write_bytes(payload)
    commands=[]
    def logged(command,log,**kwargs):
        """Authorize locally and create only a fake downloaded source folder."""
        commands.append(command)
        if command[1:3]==['config','create']:
            config=Path(command[command.index('--config')+1])
            config.write_text('[treevibes]\ntoken = dummy-test-token\n')
        if command[1]=='copy':
            Path(command[3]).mkdir(exist_ok=True)
    monkeypatch.setattr(windows.urllib.request,'urlopen',response)
    monkeypatch.setattr(windows.urllib.request,'urlretrieve',retrieve)
    monkeypatch.setattr(windows,'run_logged',logged)
    assert windows.download_drive(tmp_path,'folder-id')==tmp_path/'colab_original'
    windows.download_drive(tmp_path,'folder-id')
    assert sum(c[1:3]==['config','create'] for c in commands)==1
    assert sum(c[1]=='copy' for c in commands)==2
    assert all('--immutable' in c for c in commands if c[1]=='copy')


def test_imported_interrupted_keras_run_continues_committed_epoch(tmp_path,monkeypatch):
    """The active Colab model must continue its committed epoch after path migration."""
    from scripts.windows_support import import_experiments
    from treevibes.synthetic import save_synthetic
    from treevibes.tracking import run_experiment,valid_completed
    from treevibes.data import write_json
    import treevibes.train as training
    source=tmp_path/'original';cache=save_synthetic(source/'cache/features',40)
    config=tmp_path/'model.json'
    write_json(config,dict(id='partial',family='mlp',width=4,epochs=2,patience=2))
    actual_write=training.write_json
    def interrupted(path,value):
        """Disconnect only after a real model and its epoch pointer are committed."""
        actual_write(path,value)
        if Path(path).name=='epoch_state.json':
            raise RuntimeError('simulated disconnect after epoch commit')
    with monkeypatch.context() as old:
        old.setattr(training,'write_json',interrupted)
        with pytest.raises(RuntimeError,match='simulated disconnect'):
            run_experiment(config,cache,source)
    before=next(source.glob('runs/*/*/attempt_*/epoch_state.json'))
    first=json.loads(before.read_text())['history'][0]
    write_json(source/'notebook_settings.json',dict(output=str(source)))
    destination=tmp_path/'imported'
    import_experiments(source,destination)
    result=run_experiment(config,destination/'cache/features',destination)
    assert result['epochs_completed']==2 and result['history'][0]==first
    assert len(result['history'])==2 and valid_completed(Path(result['run_dir']))
    assert json.loads(before.read_text())['epoch']==1


def test_import_rejects_completed_manifest_without_its_model(tmp_path):
    """A valid checksum for an unrelated file must never certify a missing trained model."""
    from scripts.windows_support import import_experiments
    from treevibes.data import write_json
    from treevibes.tracking import file_hash
    source=tmp_path/'original';run=source/'runs/incomplete/identity/attempt_001';run.mkdir(parents=True)
    write_json(run/'config.json',dict(id='incomplete',family='logistic'))
    (run/'unrelated.txt').write_text('intact')
    write_json(run/'status.json',dict(state='completed'))
    write_json(run/'result.json',dict(artifact_hashes={'unrelated.txt':file_hash(run/'unrelated.txt')}))
    write_json(source/'notebook_settings.json',dict(output=str(source)))
    receipt=import_experiments(source,tmp_path/'output')
    assert receipt['verified_completed_runs']==0 and len(receipt['invalid_completed_runs'])==1


def test_public_download_needs_no_login_and_reuses_complete_copy(tmp_path,monkeypatch):
    """A public folder must download without OAuth/cookies or repeated complete transfers."""
    import os,sys
    import scripts.windows_support as windows
    from scripts.windows_support import download_public_drive
    fake=tmp_path/'fake';fake.mkdir()
    counter=tmp_path/'calls.txt'
    package=fake/'gdown';package.mkdir()
    (package/'__init__.py').write_text(
        'from pathlib import Path\n'
        'from types import SimpleNamespace\n'
        'def download(**kwargs):\n'
        '    assert kwargs["id"] == "settings", "Disposable job log must never be requested"\n'
        '    assert kwargs["use_cookies"] is False and kwargs["resume"] is True\n'
        '    target=Path(kwargs["output"]);target.write_text("{}")\n'
        '    return str(target)\n')
    (package/'download_folder.py').write_text(
        'from pathlib import Path\n'
        'class Session:\n'
        '    def close(self): pass\n'
        'def _get_session(**kwargs):\n'
        '    assert kwargs["use_cookies"] is False\n'
        '    return Session(), None\n'
        'def _parse_embedded_folder_view(**kwargs):\n'
        '    assert kwargs["folder_id"] == "public-folder", "Must prune unwanted folders"\n'
        f'    Path({str(counter)!r}).write_text("called")\n'
        '    return "root", [("settings", "notebook_settings.json", "file"),\n'
        '                    ("logs", "campaign_jobs", "application/vnd.google-apps.folder")]\n')
    work=tmp_path/'work';tools=work/'tools/gdown-6.4.2';tools.mkdir(parents=True)
    (tools/'ready.json').write_text('{}')
    monkeypatch.setattr(windows,'environment_python',lambda *args,**kwargs:Path(sys.executable))
    monkeypatch.setenv('PYTHONPATH',str(fake))
    source=download_public_drive(work,'public-folder')
    assert (source/'notebook_settings.json').exists() and counter.exists()
    counter.unlink()
    assert download_public_drive(work,'public-folder')==source
    assert not counter.exists() and not (work/'private_auth').exists()


def test_public_snapshot_preserves_existing_files_but_rejects_missing_models(tmp_path, monkeypatch):
    """Partial recovery cannot be marked complete when a required model is inaccessible."""
    import sys
    from types import SimpleNamespace
    import pytest
    from scripts.public_drive_download import download_snapshot
    source=tmp_path/'snapshot';source.mkdir()
    (source/'notebook_settings.json').write_text('{}')
    requested=[]
    def listing(**kwargs):
        """Expose both an already downloaded settings file and an unavailable model."""
        return [SimpleNamespace(id='settings',local_path=str(source/'notebook_settings.json')),
                SimpleNamespace(id='model',local_path=str(source/'runs/best.keras'))]
    def missing(**kwargs):
        """Simulate a blocked model without replacing successful downloads."""
        requested.append(kwargs['id'])
        return None
    monkeypatch.setattr('scripts.public_drive_download.list_recovery_entries',lambda *args: listing())
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=missing))
    receipt=tmp_path/'receipt.json'
    with pytest.raises(RuntimeError,match='Required recovery artifact'):
        download_snapshot(source,'folder',receipt)
    assert requested==['model'] and not receipt.exists()
    assert (source/'notebook_settings.json').read_text()=='{}'


def test_public_snapshot_keeps_final_locks_and_names_blocked_artifacts(tmp_path, monkeypatch):
    """Final-test locks must reach the importer, while failures identify their local path."""
    import sys
    from types import SimpleNamespace
    import pytest
    from scripts.public_drive_download import download_snapshot
    source=tmp_path/'source';source.mkdir()
    (source/'notebook_settings.json').write_text('{}')
    lock=source/'campaigns/old/finalists.json'
    def listing(**kwargs):
        """Expose a scientific lock that must not be treated as a disposable journal."""
        return [SimpleNamespace(id='lock-id',local_path=str(lock))]
    def blocked(**kwargs):
        """Simulate the exact class of public download exception seen remotely."""
        raise ValueError('Cannot retrieve the public link')
    monkeypatch.setattr('scripts.public_drive_download.list_recovery_entries',lambda *args: listing())
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=blocked))
    monkeypatch.setattr('scripts.public_drive_download.direct_download',lambda *args: blocked())
    with pytest.raises(RuntimeError,match=r'Required recovery artifact blocked: .*finalists.json'):
        download_snapshot(source,'folder',tmp_path/'receipt.json')


def test_public_listing_retries_and_caches_without_visiting_disposable_folders(tmp_path, monkeypatch):
    """A transient HTTP 500 must not discard earlier listings or enter excluded trees."""
    import sys
    from types import SimpleNamespace
    from scripts.public_drive_download import list_recovery_entries
    calls=[];sleeps=[]
    folder='application/vnd.google-apps.folder'
    def parse(**kwargs):
        """Fail the required runs listing once; disposable folders always fail if visited."""
        identifier=kwargs['folder_id'];calls.append(identifier)
        if identifier=='root':
            return 'root',[('logs','campaign_jobs',folder),('runs','runs',folder),
                           ('settings','notebook_settings.json','file')]
        assert identifier=='runs'
        if calls.count('runs')==1:
            raise RuntimeError('HTTP 500')
        return 'runs',[('model','best.keras','file')]
    module=SimpleNamespace(_get_session=lambda **kwargs:(SimpleNamespace(close=lambda:None),None),
                           _parse_embedded_folder_view=parse)
    monkeypatch.setitem(sys.modules,'gdown.download_folder',module)
    monkeypatch.setattr('time.sleep',sleeps.append)
    source=tmp_path/'snapshot'
    first=list(list_recovery_entries(source,'root'))
    assert [entry.id for entry in first]==['model','settings']
    assert calls==['root','runs','runs'] and sleeps==[1]
    calls.clear()
    assert len(list(list_recovery_entries(source,'root')))==2 and not calls


def test_explicit_public_download_rejects_html_and_commits_only_complete_bytes(tmp_path, monkeypatch):
    """A Drive permission page or truncated file must never become a trusted local artifact."""
    import io
    import pytest
    from scripts.public_drive_download import direct_download
    class Response(io.BytesIO):
        """Represent a bounded HTTP attachment without accessing Google."""
        status=200
        def __init__(self, body, headers):
            super().__init__(body)
            self.headers=headers
    responses=iter([Response(b'<html>login</html>',{'Content-Type':'text/html'}),
                    Response(b'{"family":"cnn"}',{'Content-Type':'application/octet-stream',
                        'Content-Disposition':'attachment; filename="config.json"','Content-Length':'16'})])
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:next(responses))
    target=tmp_path/'config.json'
    assert direct_download('file-id',target)==str(target)
    assert target.read_bytes()==b'{"family":"cnn"}'
    target.unlink()
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:Response(
        b'partial',{'Content-Disposition':'attachment','Content-Length':'100'}))
    with pytest.raises(RuntimeError,match='Both explicit'):
        direct_download('file-id',target)
    assert not target.exists()
