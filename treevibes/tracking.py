"""Track attempts durably and rebuild reviewable reports without a live server."""
import hashlib
import json
import os
import shutil
import sqlite3
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from .config import load_config
from .data import load_dataset, write_json, fingerprint, environment,sha256_file


def file_hash(path: Path) -> str:
    """Identify actual artifacts rather than trusting a completed flag alone."""
    return sha256_file(path)


def _read_json(path: Path,default: dict) -> dict:
    """Treat missing initialization metadata as an interrupted attempt, not a fatal retry error."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError,ValueError):
        return default


def _restore_checkpoint(attempts: list[Path],destination: Path) -> None:
    """Publish copied checkpoint pointers last so a disconnect leaves no half-committed epoch."""
    for previous in reversed(attempts):
        status=_read_json(previous/'status.json',{'state':'interrupted'})
        if status.get('state') not in {'running','initializing','interrupted','failed'}:
            continue
        state=_read_json(previous/'epoch_state.json',{})
        if state:
            names=list(dict.fromkeys([state.get('checkpoint',''),state.get('best_checkpoint','')]))
            names.append('epoch_state.json')
        else:
            names=['best.pt','last.pt']
        if any(not name or not (previous/name).is_file() for name in names):
            continue
        if any(not zipfile.is_zipfile(previous/name) for name in names if name!='epoch_state.json'):
            continue
        for name in names:
            source=previous/name;target=destination/name
            if not source.resolve().is_relative_to(previous.resolve()):
                raise ValueError('Checkpoint path escapes its attempt directory')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix(target.suffix+'.copying')
            shutil.copy2(source,temporary)
            if file_hash(source)!=file_hash(temporary):
                raise ValueError('Checkpoint changed during recovery')
            temporary.replace(target)
        if status.get('state') in {'running','initializing'}:
            _status(previous,'interrupted',reason='Recovered by a later attempt')
        return


def source_fingerprint() -> str:
    """Include uncommitted implementation changes in reproducibility identities."""
    return fingerprint({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def compatible_sources() -> list[str]:
    """Recover the preceding notebook only while its scientific implementation is unchanged."""
    sources = [source_fingerprint()]
    names = ['config.py', 'data.py', 'models.py', 'reference.py', 'train.py', 'evaluate.py']
    scientific = fingerprint({n:file_hash(Path(__file__).parent/n) for n in names})
    if scientific == '92d9bdb0ebafbcf6a64d42524d652f239283327af23af36d3bcd0a88a47c2324':
        sources.extend(['d6c6838950a09a4357c7194cf8b66e0dcc0634c76f547a93be9fc52d6bd0e8b8',
                        'cc7d49dcd4bdaa8e04cb3ba6af742833b218607ad9eb4cc9fc1198b3363cda87'])
    return list(dict.fromkeys(sources))


def backup_tracking(database: Path, destination: Path) -> Path:
    """Use SQLite's online backup API so committed WAL records survive reconnects."""
    database, destination = Path(database), Path(destination)
    if database.resolve() == destination.resolve():
        raise ValueError('Backup must differ from the active database')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.db')
    with sqlite3.connect(database) as source, sqlite3.connect(temporary) as target:
        source.backup(target)
    temporary.replace(destination)
    return destination


def valid_completed(directory: Path) -> bool:
    """Require matching artifact bytes before skipping a supposedly finished attempt."""
    try:
        if json.loads((directory / 'status.json').read_text())['state'] != 'completed':
            return False
        result = json.loads((directory / 'result.json').read_text())
        hashes = result['artifact_hashes']
        return bool(hashes) and all((directory / p).is_file() and file_hash(directory / p) == h
                                    for p, h in hashes.items())
    except (OSError, ValueError, KeyError):
        return False


def _status(directory, state, **details):
    """Retain prior transitions when a runtime is interrupted or retried."""
    value = dict(state=state, time=datetime.now(timezone.utc).isoformat(), **details)
    history_path = directory / 'events.jsonl'
    with history_path.open('a') as stream:
        stream.write(json.dumps(value) + '\n')
    write_json(directory / 'status.json', value)


def run_experiment(config_path: Path, cache: Path, output: Path, resume: bool = True,
                   teacher_dir: Path | None = None) -> dict:
    """Isolate retries, record failures, and log only real completed-run results."""
    from .train import train_run
    import mlflow
    config = load_config(config_path); dataset = load_dataset(cache)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    teacher_id = None
    if teacher_dir:
        teacher_dir = Path(teacher_dir).resolve()
        if not valid_completed(teacher_dir):
            raise ValueError('Teacher artifact integrity is invalid')
        teacher_id = file_hash(teacher_dir / 'best.keras')
    identity = fingerprint(dict(config=config, dataset=dataset['contract'],
                                source=source_fingerprint(), teacher=teacher_id))
    base = output / 'runs' / config['id'] / identity[:16]
    attempts = sorted(base.glob('attempt_*'))
    recovery = []
    if resume:
        for source in compatible_sources()[1:]:
            legacy = fingerprint(dict(config=config, dataset=dataset['contract'], source=source, teacher=teacher_id))
            recovery.extend(sorted((output/'runs'/config['id']/legacy[:16]).glob('attempt_*')))
    if resume:
        for attempt in reversed(recovery + attempts):
            if valid_completed(attempt):
                return json.loads((attempt / 'result.json').read_text()) | {'skipped': True}
    for attempt in attempts:
        status = _read_json(attempt/'status.json',{})
        if not status:
            _status(attempt,'interrupted',reason='Attempt initialization did not finish')
            status={'state':'interrupted'}
        if status['state'] == 'completed' and not valid_completed(attempt):
            _status(attempt, 'invalidated', reason='Required artifact missing or modified')
    number=max([int(p.name.rsplit('_',1)[-1]) for p in attempts]+[0])+1
    directory = base / f'attempt_{number:03}'
    directory.mkdir(parents=True)
    run_id = f'{config["id"]}-{identity[:12]}-{number:03}'
    write_json(directory / 'config.json', config)
    write_json(directory / 'provenance.json', dict(dataset_id=dataset['contract']['dataset_id'],
        source_hash=source_fingerprint(), versions=environment(), teacher_model_hash=teacher_id,
        teacher_run_dir=str(teacher_dir) if teacher_dir else None,
        synthetic=dataset['contract'].get('synthetic', False), run_id=run_id))
    _status(directory,'initializing',run_id=run_id)
    if resume:
        try:
            _restore_checkpoint(recovery + attempts,directory)
            if recovery:
                write_json(directory/'recovery.json', {'compatible_prior_attempts':[str(p) for p in recovery]})
        except BaseException as error:
            _status(directory,'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                    run_id=run_id,error=str(error),stage='checkpoint_recovery')
            raise
    _status(directory,'running',run_id=run_id)
    tracking = Path(os.environ.get('TREEVIBES_TRACKING_DIR', str(output / '.tracking'))).resolve()
    tracking.mkdir(parents=True, exist_ok=True)
    # Keep databases from other MLflow versions intact; schema downgrades can lose records.
    database = tracking / f'mlflow-{mlflow.__version__}.db'
    snapshot = output / f'tracking_backup-mlflow-{mlflow.__version__}.db'
    if not database.exists() and snapshot.exists():
        shutil.copy2(snapshot, database)
    mlflow.set_tracking_uri(f'sqlite:///{database}')
    try:
        # Artifacts and JSON are authoritative; optional MLflow cannot block a fit.
        result = train_run(config, dataset, directory, teacher_dir)
        result['mlflow_run_id'] = None
        try:
            _log_mlflow(mlflow, config, dataset, directory, output, run_id, result)
        except Exception as error:
            result['tracking_warning'] = str(error)
            write_json(directory/'tracking_warning.json', {'error':str(error)})
        artifacts = [p for p in directory.iterdir() if p.is_file() and p.name not in
                     {'status.json', 'events.jsonl', 'result.json'}]
        result.update(run_id=run_id, run_dir=str(directory),
                      config_id=config['id'], dataset_id=dataset['contract']['dataset_id'],
                      synthetic=dataset['contract'].get('synthetic',False),
                      artifact_hashes={p.name:file_hash(p) for p in artifacts})
        write_json(directory / 'result.json', result)
        _status(directory, 'completed', run_id=run_id)
        return result
    except BaseException as error:
        _status(directory, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                run_id=run_id, error=str(error), error_type=type(error).__name__)
        (directory / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if database.exists():
            try:
                backup_tracking(database, snapshot)
            except Exception as error:
                # A backup failure is visible without invalidating completed model bytes.
                write_json(output/'tracking_backup_warning.json', {'error':str(error)})


def _log_mlflow(mlflow, config, dataset, directory, output, run_id, result):
    """Mirror an already fitted run into the optional version-specific tracker."""
    experiment_name='TreeVibes synthetic checks' if dataset['contract'].get('synthetic') else 'TreeVibes'
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name,artifact_location=(output/'mlflow_artifacts').as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_id) as active:
        mlflow.set_tag('mlflow.source.name', 'treevibes')
        mlflow.log_params({k: json.dumps(v) if isinstance(v, dict) else v for k,v in config.items()})
        mlflow.set_tags(dict(dataset_id=dataset['contract']['dataset_id'], attempt_id=run_id,
                             synthetic=str(dataset['contract'].get('synthetic', False))))
        mlflow.log_metrics({f'val_{k}':v for k,v in result['validation'].items()
                            if isinstance(v, (int, float))})
        for epoch in result.get('history', []):
            mlflow.log_metrics({k:v for k,v in epoch.items() if k != 'epoch'}, step=epoch['epoch'])
        for name in ['config.json', 'provenance.json', 'input_contract.json',
                     'training_result.json', 'validation_predictions.npz',
                     'best.keras', 'best.pt', 'best.joblib']:
            if (directory / name).exists():
                mlflow.log_artifact(str(directory / name))
        result["mlflow_run_id"] = active.info.run_id


def export_leaderboard(output: Path, workbook: Path) -> Path:
    """Keep incomplete attempts and pending hardware observations visible in Excel."""
    import pandas as pd
    from openpyxl.styles import Font, PatternFill
    output=Path(output); rows=[]; experiments=[]; failures=[]; hardware=[]
    for path in sorted(output.glob('runs/*/*/attempt_*/status.json')):
        directory=path.parent
        status=_read_json(path,{'state':'interrupted'})
        config=_read_json(directory/'config.json',dict(id=directory.parent.parent.name,family='unknown',
                                                      target='unknown',seed=None))
        result=_read_json(directory/'result.json',{})
        provenance=_read_json(directory/'provenance.json',dict(run_id=f'{config["id"]}-{directory.name}',
                               synthetic=None,dataset_id=None))
        row=dict(run_id=provenance['run_id'], config_id=config['id'], family=config['family'],
                 target=config['target'], seed=config['seed'], state=status['state'],
                 synthetic=provenance['synthetic'], dataset_id=provenance['dataset_id'],
                 run_dir=str(directory), parameters=result.get('parameters'))
        row.update({f'val_{k}':v for k,v in result.get('validation',{}).items() if not isinstance(v,list)})
        row.update({f'{part}_accepted_fraction':value['accepted_fraction']
                    for part,value in (result.get('coverage') or {}).items()})
        audit=result.get('data_audit') or {}
        row.update(label_conflict_groups=audit.get('conflicting_content_groups'),
                   label_conflict_files_excluded=audit.get('excluded_recordings'))
        if status['state'] == 'completed' and not valid_completed(directory):
            row['state']='invalidated'
        rows.append(row)
        experiments.append(dict(run_id=provenance['run_id'], **{k:json.dumps(v) if isinstance(v,dict) else v
                                                               for k,v in config.items()}))
        if row['state'] != 'completed':
            failures.append(row | {'error':status.get('error', status.get('reason',''))})
    for path in output.glob('**/board_measurement.json'):
        hardware.append(json.loads(path.read_text()) | {'source':str(path)})
    seen_exports=set()
    for path in sorted(output.glob('exports/**/export.json')):
        meta=json.loads(path.read_text())
        # Firmware bundles copy metadata; count only actual export roots.
        key=(meta['model_sha256'],meta['parent_run_id'],meta['method'])
        if key in seen_exports:
            continue
        seen_exports.add(key)
        row=dict(run_id=meta['parent_run_id']+'-'+meta['method'],config_id=path.parent.name,
                 family='export',target='deployment_candidate',state=meta['state'],
                 synthetic=meta['synthetic'],dataset_id=meta['dataset_id'],run_dir=str(path.parent),
                 method=meta['method'],artifact_bytes=meta['bytes'],board_status=meta['board_status'],
                 model_sha256=meta['model_sha256'])
        row.update({f'val_{k}':v for k,v in meta['validation'].items() if not isinstance(v,list)})
        rows.append(row)
        experiments.append(dict(run_id=row['run_id'],parent_run=meta['parent_run'],
                                method=meta['method'],export_config=json.dumps(meta,sort_keys=True)))
    for path in sorted(output.glob('exports/**/failure.json')):
        meta=json.loads(path.read_text())
        failures.append(dict(run_id=path.parent.name,state='failed',error=meta['error'],
                             method=meta['method'],run_dir=str(path.parent)))
    for path in sorted(output.glob('preflight_failures/*.json')):
        failures.append(json.loads(path.read_text()))
    workbook=Path(workbook); workbook.parent.mkdir(parents=True,exist_ok=True)
    temporary=workbook.with_name(workbook.stem+'.tmp.xlsx')
    with pd.ExcelWriter(temporary,engine='openpyxl') as writer:
        for name, values in [('Runs',rows),('Experiments',experiments),('Failures',failures),
                             ('Board measurements',hardware)]:
            frame=pd.DataFrame(values)
            if frame.empty:
                frame=pd.DataFrame(columns=['run_id','state','source'])
            frame.to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name]; sheet.freeze_panes='A2'; sheet.auto_filter.ref=sheet.dimensions
            for cell in sheet[1]:
                cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='17365D')
            for column in sheet.columns:
                letter=column[0].column_letter
                sheet.column_dimensions[letter].width=min(48,max(14,max(len(str(c.value or '')) for c in column)+2))
    temporary.replace(workbook)
    return workbook
