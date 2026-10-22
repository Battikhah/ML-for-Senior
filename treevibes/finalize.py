"""Lock finalists before releasing held-out test labels to evaluation code."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .data import write_json, fingerprint
from .tracking import file_hash, valid_completed
from .train import predict_run, normalized
from .evaluate import evaluate_probabilities


def lock_finalists(runs: list[Path], exports: list[Path], destination: Path) -> Path:
    """Snapshot exact checkpoints and validation-selected thresholds before test evaluation."""
    destination=Path(destination)
    if destination.exists():
        raise ValueError('Finalist lock already exists; preserve the original selection')
    entries=[]
    for directory in map(Path,runs):
        if not valid_completed(directory):
            raise ValueError(f'Finalist is not a valid completed run: {directory}')
        result=json.loads((directory/'result.json').read_text())
        contract=json.loads((directory/'input_contract.json').read_text())
        names=['config.json','input_contract.json','training_result.json']
        names += [name for name in ['best.keras','best.pt','best.joblib'] if (directory/name).exists()]
        entries.append(dict(kind='run',path=str(directory.resolve()),dataset_id=contract['dataset_id'],
            run_id=result['run_id'],threshold=result['validation']['threshold'],
            hashes={name:file_hash(directory/name) for name in names}))
    for directory in map(Path,exports):
        meta=json.loads((directory/'export.json').read_text())
        if meta['state']!='exported' or file_hash(directory/'model.tflite')!=meta['model_sha256']:
            raise ValueError('Export changed or conversion was incomplete')
        entries.append(dict(kind='export',path=str(directory.resolve()),dataset_id=meta['dataset_id'],
            run_id=meta['parent_run_id']+'-'+meta['method'],threshold=meta['threshold'],
            hashes={name:file_hash(directory/name) for name in ['model.tflite','export.json','input_contract.json']}))
    if not entries:
        raise ValueError('Choose at least one finalist')
    write_json(destination,dict(created_at=datetime.now(timezone.utc).isoformat(),entries=entries))
    return destination


def evaluate_final(lock: Path,dataset: dict,output: Path) -> dict:
    """Check locked artifact hashes before any prediction on the held-out test split."""
    lock=Path(lock);output=Path(output)
    manifest=json.loads(lock.read_text());lock_hash=file_hash(lock)
    selected=[e for e in manifest['entries'] if e['dataset_id']==dataset['contract']['dataset_id']]
    if not selected:
        raise ValueError('No locked finalists match this dataset cache')
    for entry in selected:
        directory=Path(entry['path'])
        if any(not (directory/name).exists() or file_hash(directory/name)!=digest
               for name,digest in entry['hashes'].items()):
            raise ValueError('Locked finalist artifact changed')
    result_path=output/'final_results.json'
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous['lock_sha256']!=lock_hash or previous['dataset_id']!=dataset['contract']['dataset_id']:
            raise ValueError('Output already contains a different final evaluation')
        return previous
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for index,entry in enumerate(selected):
        directory=Path(entry['path'])
        if entry['kind']=='run':
            probabilities=predict_run(directory,dataset['X_test'])
        else:
            from .export import tflite_predict
            contract=json.loads((directory/'input_contract.json').read_text())
            probabilities=tflite_predict(directory/'model.tflite',normalized(dataset['X_test'],contract))
        metrics=evaluate_probabilities(dataset['y_test'],probabilities,entry['threshold'])
        np.savez_compressed(output/f'predictions_{index:03}.npz',labels=dataset['y_test'],
                            probabilities=probabilities,ids=dataset['ids_test'])
        results.append(dict(run_id=entry['run_id'],kind=entry['kind'],test=metrics))
    result=dict(lock_sha256=lock_hash,dataset_id=dataset['contract']['dataset_id'],
                synthetic=dataset['contract'].get('synthetic',False),results=results,
                warning='Further tuning informed by these test results is exploratory.')
    write_json(result_path,result)
    return result
