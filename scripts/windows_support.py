"""Prepare native Windows workers and relocate verified experiment copies safely."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath,PureWindowsPath
import shutil
import subprocess
import sys
import urllib.request
import zipfile


def run_logged(command,log,env=None,stream=False,cwd=None):
    """Decode worker output explicitly as UTF-8 even when Windows uses another code page."""
    environment=dict(os.environ if env is None else env)
    environment.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as output:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',env=environment,cwd=cwd)
        try:
            for line in process.stdout:
                output.write(line);output.flush()
                if stream:
                    print(line,end='',flush=True)
            status=process.wait()
        except BaseException:
            process.terminate();process.wait()
            raise
    if status:
        raise RuntimeError(f'Worker failed (exit {status}). Full log: {log}\n'+log.read_text(encoding='utf-8')[-6000:])


def environment_python(directory, windows=None):
    """Choose the platform's real venv executable rather than assuming Unix paths."""
    windows = os.name == 'nt' if windows is None else windows
    return Path(directory)/('Scripts/python.exe' if windows else 'bin/python')


def drive_auth_command(executable, config, folder_id):
    """Authorize readonly Drive access in a browser without dumping OAuth tokens."""
    return [str(executable),'config','create','treevibes','drive','scope','drive.readonly',
            'root_folder_id',folder_id,'config_is_local','true','--config',str(config),'--no-output']


def download_drive(work, folder_id):
    """Copy the private folder to an untouched local source snapshot; never sync back."""
    work=Path(work);tools=work/'tools';tools.mkdir(parents=True,exist_ok=True)
    executable=tools/'rclone.exe';config=work/'private_auth/rclone.conf'
    if not executable.exists():
        with urllib.request.urlopen('https://downloads.rclone.org/version.txt',timeout=60) as response:
            version=response.read().decode().strip().split()[-1]
        if not version.startswith('v') or not all(c.isdigit() or c=='.' for c in version[1:]):
            raise ValueError('Unexpected rclone release identifier')
        name=f'rclone-{version}-windows-amd64.zip'
        base=f'https://downloads.rclone.org/{version}/'
        archive=tools/name;urllib.request.urlretrieve(base+name,archive)
        with urllib.request.urlopen(base+'SHA256SUMS',timeout=60) as response:
            sums=response.read().decode().splitlines()
        expected=next(line.split()[0] for line in sums if line.split()[-1].lstrip('*')==name)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
            raise ValueError('rclone download checksum mismatch')
        with zipfile.ZipFile(archive) as stream:
            member=next(n for n in stream.namelist() if n.endswith('/rclone.exe'))
            executable.write_bytes(stream.read(member))
    config.parent.mkdir(parents=True,exist_ok=True)
    import configparser
    saved=configparser.ConfigParser(interpolation=None)
    saved.read(config)
    if not saved.has_option('treevibes','token'):
        print('Authorize readonly Drive access in the browser on the WINDOWS PC. '
              'Sign into the Google account owning the experiment folder.',flush=True)
        run_logged(drive_auth_command(executable,config,folder_id),work/'drive_auth.log',stream=True)
    source=work/'colab_original'
    # copy leaves remote files intact; immutable prevents overwriting an earlier snapshot.
    run_logged([str(executable),'copy','treevibes:',str(source),'--config',str(config),
                '--immutable','--transfers','2','--checkers','2','--stats','30s','--log-level','INFO'],
               work/'drive_download.log',stream=True)
    return source


def download_public_drive(work,folder_id):
    """Download a link-shared folder without cookies or login, retaining completed transfers."""
    work=Path(work);source=work/'colab_original';receipt=work/'public_download.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text(encoding='utf-8'))
        if saved['folder_id']!=folder_id:
            raise ValueError('This download snapshot belongs to another folder; choose a different WORK directory.')
        if saved['files'] and all((source/name).is_file() for name in saved['files']):
            print('Reusing the complete public Drive download.',flush=True)
            return source
    tools=work/'tools/gdown-6.4.2';python=environment_python(tools)
    if not python.exists():
        run_logged([sys.executable,'-m','venv',str(tools)],work/'public_downloader_venv.log')
    if not (tools/'ready.json').exists():
        run_logged([str(python),'-m','pip','install','--disable-pip-version-check','gdown==6.4.2'],
                   work/'public_downloader_install.log')
        (tools/'ready.json').write_text(json.dumps(dict(gdown='6.4.2')),encoding='utf-8')
    # Isolated tooling leaves the Jupyter environment unchanged and never reads browser cookies.
    print('Downloading the public experiment folder without Google login. '
          'The folder must allow Anyone with the link: Viewer.',flush=True)
    run_logged([str(python),str(Path(__file__).with_name('public_drive_download.py')),
                str(source),folder_id,str(receipt)],work/'public_drive_download.log',stream=True)
    return source


def prepare_windows_workers(project,work):
    """Keep legacy TensorFlow CPU separate from Blackwell-compatible PyTorch CUDA."""
    from treevibes.runtime import worker_environment
    if os.name!='nt' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Select the existing native Windows Python 3.12 kernel.')
    project,work=Path(project),Path(work)
    workers={};env=worker_environment(project)
    for framework in ['tensorflow','pytorch']:
        directory=work/'runtimes'/('tf216-cpu' if framework=='tensorflow' else 'torch271-cu128')
        python=environment_python(directory);ready=directory/'ready.json'
        lock=directory/'requirements.txt'
        if not python.exists():
            run_logged([sys.executable,'-m','venv',str(directory)],work/f'{framework}_venv.log',env=env)
        if not ready.exists():
            command=[str(python),'-m','pip','install','--disable-pip-version-check']
            if framework=='pytorch':
                command+=['--extra-index-url','https://download.pytorch.org/whl/cu128']
            if lock.exists():
                command+=['-r',str(lock)]
            else:
                packaged=project/'requirements'/f'windows-{framework}-py312.txt'
                if packaged.exists():
                    command+=['-r',str(packaged)]
            command+=['-e',f'{project}[keras]' if framework=='tensorflow' else str(project),
                      'mlflow==3.14.0','scikit-learn==1.9.1']
            run_logged(command,work/f'{framework}_install.log',env=env,cwd=project)
            if framework=='pytorch':
                run_logged([str(python),'-m','pip','install','torch==2.7.1',
                            '--index-url','https://download.pytorch.org/whl/cu128'],
                           work/'pytorch_cuda_install.log',env=env)
            frozen=subprocess.run([str(python),'-m','pip','freeze','--exclude-editable'],
                                  check=True,capture_output=True,text=True,env=env)
            lock.write_text(frozen.stdout)
            ready.write_text(json.dumps(dict(framework=framework,python=sys.version)))
        if framework=='tensorflow':
            run_logged([str(python),'-c','import json; from treevibes.runtime import repair_keras_random_bounds; '
                        'print(json.dumps(repair_keras_random_bounds()))'],work/'keras_seed_compatibility.log',env=env)
        workers[framework]=str(python)
    return workers


def _relocate(value,old_root,new_root):
    """Change only strings rooted in the old output folder; retain raw-data provenance."""
    if isinstance(value,str) and (value==old_root or value.startswith(old_root+'/')):
        relative=value[len(old_root):].lstrip('/')
        return str(Path(new_root)/Path(relative)) if relative else str(new_root)
    if isinstance(value,list):
        return [_relocate(v,old_root,new_root) for v in value]
    if isinstance(value,dict):
        return {k:_relocate(v,old_root,new_root) for k,v in value.items()}
    return value


def valid_imported_run(directory):
    """Require the family's actual model and complete, contained artifact metadata."""
    from treevibes.tracking import valid_completed
    directory=Path(directory)
    try:
        config=json.loads((directory/'config.json').read_text())
        result=json.loads((directory/'result.json').read_text())
        hashes=result['artifact_hashes']
        if not isinstance(hashes,dict):
            return False
        family=config['family']
        model=('best.pt' if family.startswith('reference_') else 'best.joblib'
               if family in {'logistic','svm','forest','histogram'} else 'best.keras')
        required={model,'config.json','provenance.json','input_contract.json',
                  'training_result.json','validation_predictions.npz'}
        if not required.issubset(hashes):
            return False
        for name in hashes:
            portable=PurePosixPath(name.replace('\\','/'))
            if (portable.is_absolute() or '..' in portable.parts or PureWindowsPath(name).drive
                    or not (directory/name).resolve().is_relative_to(directory.resolve())):
                return False
        return valid_completed(directory)
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return False


def import_experiments(source,destination):
    """Validate before relocation, retain originals, and commit the imported copy atomically."""
    from treevibes.tracking import file_hash
    from treevibes.data import write_json
    source,destination=Path(source).resolve(),Path(destination).resolve()
    receipt=destination/'windows_import.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    if destination.exists():
        raise ValueError('Destination already exists without an import receipt; choose an empty output folder.')
    if list(source.glob('campaigns/**/finalists*.json')):
        raise ValueError('Final-test locks exist. Preserve them; this importer requires a separate lock migration.')
    settings=json.loads((source/'notebook_settings.json').read_text())
    old_root=settings['output'].rstrip('/')
    verified=[];invalid=[]
    for status in source.glob('runs/*/*/attempt_*/status.json'):
        if json.loads(status.read_text()).get('state')=='completed':
            (verified if valid_imported_run(status.parent) else invalid).append(status.parent.relative_to(source))
    staging=destination.with_name(destination.name+'.importing')
    if staging.exists():
        raise ValueError(f'An unfinished import exists at {staging}; preserve it and use another destination.')
    shutil.copytree(source,staging)
    if any(not valid_imported_run(staging/relative) for relative in verified):
        raise ValueError('Copied artifacts changed during import; the staged copy was preserved for inspection.')
    for path in staging.rglob('*.json'):
        try:
            original=json.loads(path.read_text())
        except (ValueError,UnicodeError):
            continue
        changed=_relocate(original,old_root,destination)
        if changed!=original:
            write_json(path,changed)
    for relative in verified:
        directory=staging/relative
        result=json.loads((directory/'result.json').read_text())
        # Only previously verified runs may receive updated metadata checksums.
        result['artifact_hashes']={name:file_hash(directory/name) for name in result['artifact_hashes']}
        write_json(directory/'result.json',result)
    for relative in invalid:
        write_json(staging/relative/'status.json',dict(state='invalidated',reason='Source artifacts failed import integrity checks'))
    for status in staging.glob('runs/*/*/attempt_*/status.json'):
        value=json.loads(status.read_text())
        if value.get('state') in {'running','initializing'}:
            write_json(status,value|dict(state='interrupted',reason='Recovered on another machine'))
    preserved_backups=[]
    for database in staging.glob('tracking_backup*.db'):
        # Old MLflow databases contain Linux artifact URIs; start fresh without deleting history.
        archived=staging/'imported_tracking'/database.name
        archived.parent.mkdir(exist_ok=True)
        if archived.exists():
            raise ValueError('Imported tracking backup already exists; refusing to overwrite history.')
        database.replace(archived)
        preserved_backups.append(str(archived.relative_to(staging)))
    report=dict(source=str(source),original_output=old_root,destination=str(destination),
                verified_completed_runs=len(verified),invalid_completed_runs=[str(p) for p in invalid],
                preserved_tracking_backups=preserved_backups)
    write_json(staging/'windows_import.json',report)
    staging.replace(destination)
    return report


def seed_campaign(settings,project):
    """Bridge machine-dependent settings to verified imported caches and current result metadata."""
    from treevibes.data import load_dataset,fingerprint,write_json
    from treevibes.tracking import file_hash,source_fingerprint,compatible_sources
    output=Path(settings['output']);receipt=json.loads((output/'windows_import.json').read_text())
    caches={}
    for contract in output.glob('cache/**/contract.json'):
        try:
            data=load_dataset(contract.parent)
            caches[data['contract']['dataset_id']]=str(contract.parent)
        except (ValueError,OSError,KeyError):
            continue
    if not caches:
        raise ValueError('No intact feature caches were imported; copy the complete Drive folder before continuing.')
    runs=[];sources=set(compatible_sources())
    for result_path in sorted(output.glob('runs/*/*/attempt_*/result.json')):
        directory=result_path.parent
        if not valid_imported_run(directory):
            continue
        provenance=json.loads((directory/'provenance.json').read_text())
        if provenance['source_hash'] in sources:
            runs.append(json.loads(result_path.read_text()))
    identity=fingerprint(dict(settings={k:v for k,v in settings.items() if k!='final_test'},
        groups_hash=file_hash(Path(settings['groups_csv'])) if settings.get('groups_csv') else None,
        source=source_fingerprint()))[:16]
    progress=output/'campaigns'/identity/'progress.json'
    if not progress.exists():
        write_json(progress,dict(campaign_id=identity,state='imported',runs=runs,caches=caches,
                                recovered_campaign_id='windows-import'))
    report=receipt|dict(compatible_completed_runs=len(runs),verified_caches=len(caches),campaign_id=identity)
    write_json(output/'windows_resume_audit.json',report)
    return report
