"""Evaluate the previously selected local INT8 operating point without retuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
    """Reject accidental changes to the exported model, normalization, or source WAVs."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(review, output):
    """Compare the frozen local route on the same recording split as the PCA audit."""
    import librosa
    from scipy.signal import butter, filtfilt
    from scipy.stats import beta
    sys.path.insert(0, str(review))
    import window_campaign as f
    previous = Path(f.read(review/'config.json')['previous'])
    settings = f.read(previous/'settings.json')
    sys.path.insert(0, settings['project'])
    from treevibes.export import tflite_predict
    point = next(r for r in f.read(review/'operating_points.json')
                 if r['id'] == 'local_raw_int8' and r['target_clean_recall'] == 1.)
    model = review/'runs/local_raw/model.tflite'
    assert digest(model) == '33d97cb39d6dcb6de0af0c4b70621ca85d9a4dd4f165ce2795a44dac78c1e309'
    norm = f.read(review/'cache/raw/contract.json')
    output.mkdir(parents=True, exist_ok=True)
    lock = dict(model_sha256=digest(model), model_bytes=model.stat().st_size,
                normalization_sha256=digest(review/'cache/raw/contract.json'),
                source_sha256=digest(__file__), threshold=point['threshold'],
                threshold_source='pre-existing validation operating_points.json, target recall 1.0',
                aggregation='maximum score across original 10s/5s-stride windows plus tail',
                model_retrained=False, test_threshold_retuned=False)
    lockfile = output/'locked_local.json'
    if lockfile.exists():
        assert f.read(lockfile) == lock, 'Frozen local audit changed'
    else:
        f.save(lockfile, lock)
    if (output/'audit_results.json').exists():
        print('REUSING_FROZEN_LOCAL_AUDIT', flush=True)
        return
    primary = f.read(previous/'audit/selection.json')['primary_dataset_id']
    for path in (Path(settings['original'])/'cache').glob('*/contract.json'):
        if f.read(path)['dataset_id'] == primary:
            manifest = f.read(path.parent/'manifest.json')
            with np.load(path.parent/'features.npz', allow_pickle=False) as original:
                ids, labels, first = original['ids_test'], original['y_test'], original['X_test']
            break
    hashes = {r['id']:r['sha256'] for r in manifest}
    train_ids = np.load(review/'cache/raw/ids_train.npy')
    val_ids = np.load(review/'cache/raw/ids_val.npy')
    assert not set(map(str,ids)) & set(map(str,train_ids))
    assert not set(map(str,ids)) & set(map(str,val_ids))
    assert not {hashes[str(i)] for i in ids} & {hashes[str(i)] for i in np.concatenate([train_ids,val_ids])}
    root = Path(f.read(previous/'audit/audio_source.json')['root'])
    b,a = butter(4,[100,2500],btype='bandpass',fs=8000)
    features, groups, starts = [], [], []
    for i, identifier in enumerate(ids):
        path = next((base/str(identifier) for base in [root,root/'field/field/train'] if (base/str(identifier)).is_file()), None)
        if path is None or digest(path) != hashes[str(identifier)]:
            raise ValueError('Held-out recording identity changed')
        signal,_ = librosa.load(path,sr=8000,mono=True)
        for start in f.window_starts(len(signal)):
            clip = signal[start:start+80000]
            clip = np.pad(clip,(0,max(0,80000-len(clip))))
            matrix = f.mfcc(filtfilt(b,a,clip))
            if start == 0:
                np.testing.assert_allclose(matrix,first[i],atol=1e-4,rtol=0)
            features.append(matrix);groups.append(i);starts.append(start)
        if (i+1)%100 == 0:
            print(f'LOCAL_TEST_FEATURES {i+1}/{len(ids)}',flush=True)
    x = ((np.stack(features)-np.array(norm['mean']))/np.array(norm['scale'])).astype('float32')
    window_probs = tflite_predict(model,x)
    p = f.aggregate(window_probs,np.array(groups),len(ids))
    # The same exported bytes must reproduce the validation row before test results are accepted.
    xv,gv,yv,idv = f.arrays(review,'raw','val')
    pv = f.aggregate(tflite_predict(model,xv),gv,len(yv))
    saved = np.load(review/'predictions/local_raw_int8.npz')
    assert np.array_equal(saved['ids'],idv)
    np.testing.assert_allclose(pv,saved['probabilities'],atol=1e-12,rtol=1e-12)
    val = f.metrics(yv,pv,lock['threshold'])
    assert val['false_negatives'] == 0 and val['false_positives'] == 26
    test = f.metrics(labels,p,lock['threshold'])
    tp = test['confusion'][1][1]; n = int(sum(labels))
    test['rpw_recall_exact_95ci'] = [float(beta.ppf(.025,tp,n-tp+1)) if tp else 0.,float(beta.ppf(.975,tp+1,n-tp)) if tp<n else 1.]
    np.savez_compressed(output/'test_predictions.npz',ids=ids,labels=labels,probabilities=p,groups=groups,starts_samples=starts,window_probabilities=window_probs)
    result = dict(lock=lock,validation=val,test=test,actual_tflite_bytes_evaluated=True,
                  validation_predictions_reproduced=True,exact_audio_overlap=False,
                  grouping='file_and_duplicate_only; not verified independent trees/sessions',
                  test_set_previously_observed_for_pca=True,board_verified=False,
                  missed_rpw_ids=[str(i) for i,y,pred in zip(ids,labels,p) if y==1 and pred<lock['threshold']])
    f.save(output/'audit_results.json',result)
    print('LOCAL_INT8_TEST_RESULT',json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.review,args.output)
