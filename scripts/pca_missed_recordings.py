"""Diagnose the four locked-test misses without fitting models or changing thresholds.

Paired interventions remove byte quantization and then PCA itself while keeping
weights, waveform coverage, normalization, and the original threshold fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def fingerprint(path):
    """Verify saved model and audio identities before comparing new diagnostic scores."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(review, audit, output):
    """Attribute individual errors to representation changes and describe audio context."""
    import tensorflow as tf
    import librosa
    from scipy.signal import butter, filtfilt, welch
    from scipy.special import expit
    from scipy.stats import percentileofscore
    from sklearn.metrics import pairwise_distances
    sys.path.insert(0, str(review))
    import window_campaign as f
    previous = Path(f.read(review / 'config.json')['previous'])
    settings = f.read(previous / 'settings.json')
    sys.path.insert(0, settings['project'])
    lock = f.read(audit / 'locked_finalist.json')
    for key in ['model', 'pca', 'normalization', 'window_source']:
        assert fingerprint(lock['assets'][key]['path']) == lock['assets'][key]['sha256'], key
    model = tf.keras.models.load_model(lock['assets']['model']['path'], compile=False)
    codec = np.load(lock['assets']['pca']['path'])
    norm = f.read(lock['assets']['normalization']['path'])
    threshold = lock['threshold']
    original = np.load(audit / 'test_predictions.npz')
    ids, labels = original['ids'], original['labels']
    misses = set(map(str, ids[(labels == 1) & (original['probabilities'] < threshold)]))
    assert len(misses) == 4
    primary = f.read(previous / 'audit/selection.json')['primary_dataset_id']
    for path in (Path(settings['original']) / 'cache').glob('*/contract.json'):
        if f.read(path)['dataset_id'] == primary:
            hashes = {r['id']: r['sha256'] for r in f.read(path.parent / 'manifest.json')}
            break
    audio_root = Path(f.read(previous / 'audit/audio_source.json')['root'])
    b, a = butter(4, [100, 2500], btype='bandpass', fs=8000)
    features, groups, starts, audio = [], [], [], []
    signals = {}
    for i, identifier in enumerate(ids):
        identifier = str(identifier)
        path = next((base / identifier for base in [audio_root, audio_root / 'field/field/train'] if (base / identifier).is_file()), None)
        if path is None or fingerprint(path) != hashes[identifier]:
            raise ValueError('WAV identity changed: ' + identifier)
        signal, _ = librosa.load(path, sr=8000, mono=True)
        filtered_all = filtfilt(b, a, signal)
        raw_rms = float(np.sqrt(np.mean(signal.astype('float64') ** 2)))
        rms = float(np.sqrt(np.mean(filtered_all ** 2)))
        hz, power = welch(filtered_all, fs=8000, nperseg=2048)
        power = np.maximum(power, 1e-30)
        audio.append(dict(id=identifier, label=int(labels[i]), duration=len(signal)/8000,
                          rms=raw_rms, filtered_rms=rms,
                          peak=float(np.max(np.abs(signal))),
                          clip_fraction=float(np.mean(np.abs(signal) >= .999)),
                          spectral_centroid_hz=float(np.sum(hz * power)/np.sum(power)),
                          spectral_flatness=float(np.exp(np.mean(np.log(power)))/np.mean(power)),
                          zero_fraction=float(np.mean(signal == 0)),
                          sha256=hashes[identifier]))
        for start in f.window_starts(len(signal)):
            clip = signal[start:start+80000]
            clip = np.pad(clip, (0, max(0, 80000-len(clip))))
            features.append(f.mfcc(filtfilt(b, a, clip)))
            groups.append(i)
            starts.append(start/8000)
        if identifier in misses:
            signals[identifier] = signal
        if (i+1) % 100 == 0:
            print(f'ERROR_DIAGNOSTIC_FEATURES {i+1}/{len(ids)}', flush=True)
    x = ((np.stack(features)-np.array(norm['mean']))/np.array(norm['scale'])).astype('float32')
    groups, starts = np.array(groups), np.array(starts)
    flat = x.reshape(len(x), -1)
    pc = (flat-codec['mean']) @ codec['components'].T
    codes = np.clip(np.rint((pc-codec['lo'])/codec['scale']), 0, 255).astype('uint8')
    restored_pc = codes.astype('float32')*codec['scale']+codec['lo']
    continuous = (pc @ codec['components']+codec['mean']).reshape(x.shape)
    restored = (restored_pc @ codec['components']+codec['mean']).reshape(x.shape)

    def score(values):
        """Use the unchanged server model and original max-window aggregation."""
        return np.concatenate([model(values[i:i+128, ..., None], training=False).numpy().ravel() for i in range(0, len(values), 128)])

    logits = {name: score(values) for name, values in [('full_mfcc', x), ('pca16_float', continuous), ('pca16_uint8', restored)]}
    probs = {name: expit(f.aggregate(z, groups, len(ids))) for name, z in logits.items()}
    np.testing.assert_allclose(probs['pca16_uint8'], original['probabilities'], atol=2e-6, rtol=1e-5)
    reconstruction = np.mean((flat-restored.reshape(len(x), -1))**2, axis=1)
    pca_only_mse = np.mean((flat-continuous.reshape(len(x), -1))**2, axis=1)
    quant_only_mse = np.mean((restored.reshape(len(x), -1)-continuous.reshape(len(x), -1))**2, axis=1)
    clipped = np.mean((pc < codec['lo']) | (pc > codec['lo']+255*codec['scale']), axis=1)
    train, train_groups, train_labels, train_ids = f.arrays(review, 'raw', 'train')
    train_flat = train.reshape(len(train), -1)
    window_labels = train_labels[train_groups]
    records = []
    for i, row in enumerate(audio):
        ix = np.flatnonzero(groups == i)
        row.update(scores={name:float(p[i]) for name,p in probs.items()},
                   window_count=len(ix), padded_seconds=max(0,10-row['duration']),
                   pca_mse_mean=float(pca_only_mse[ix].mean()),
                   reconstruction_mse_mean=float(reconstruction[ix].mean()),
                   quantization_mse_mean=float(quant_only_mse[ix].mean()),
                   clipped_coordinate_fraction=float(clipped[ix].mean()),
                   missed_locked=bool(row['id'] in misses))
        if row['id'] not in misses:
            continue
        row['windows'] = [dict(start_seconds=float(starts[k]), **{name:float(expit(z[k])) for name,z in logits.items()},
                               reconstruction_mse=float(reconstruction[k]),quantization_mse=float(quant_only_mse[k]),clipped_fraction=float(clipped[k])) for k in ix]
        distances = pairwise_distances(flat[ix], train_flat)
        neighbors = []
        for j,k in enumerate(ix):
            by_class = {}
            for label,name in [(0,'healthy'),(1,'RPW')]:
                allowed = np.flatnonzero(window_labels == label)
                nearest = allowed[np.argmin(distances[j,allowed])]
                by_class[name] = dict(id=str(train_ids[train_groups[nearest]]), distance=float(distances[j,nearest]))
            neighbors.append(dict(start_seconds=float(starts[k]), nearest_training=by_class))
        row['nearest_training_windows'] = neighbors
        # These are sensitivity probes, not proposed corrections or a retuned test score.
        changed = []
        for gain in [.5, 2.]:
            values = []
            for start in f.window_starts(len(signals[row['id']])):
                clip = signals[row['id']][start:start+80000]*gain
                clip = np.pad(clip,(0,max(0,80000-len(clip))))
                values.append(f.mfcc(filtfilt(b,a,clip)))
            values = ((np.stack(values)-np.array(norm['mean']))/np.array(norm['scale'])).astype('float32')
            projected = (values.reshape(len(values),-1)-codec['mean'])@codec['components'].T
            encoded = np.clip(np.rint((projected-codec['lo'])/codec['scale']),0,255).astype('uint8')
            decoded = ((encoded.astype('float32')*codec['scale']+codec['lo'])@codec['components']+codec['mean']).reshape(values.shape)
            changed.append(dict(gain=gain,pca16_uint8_score=float(expit(np.max(score(decoded)))),full_mfcc_score=float(expit(np.max(score(values))))))
        row['gain_sensitivity_not_retrained'] = changed
        records.append(row)
    positive = [row for row in audio if row['label'] == 1]
    negative = [row for row in audio if row['label'] == 0]
    summaries = {}
    keys=['duration','rms','filtered_rms','spectral_centroid_hz','spectral_flatness','reconstruction_mse_mean','quantization_mse_mean','clipped_coordinate_fraction']
    for name, population in [('RPW',positive),('healthy',negative)]:
        summaries[name] = {key:dict(median=float(np.median([r[key] for r in population])),min=float(min(r[key] for r in population)),max=float(max(r[key] for r in population))) for key in keys}
    for row in records:
        row['RPW_percentiles'] = {key:float(percentileofscore([r[key] for r in positive],row[key],kind='mean')) for key in keys}
        row['threshold_gap'] = float(threshold-row['scores']['pca16_uint8'])
    metrics = {name:f.metrics(labels,p,threshold) for name,p in probs.items()}
    # One fixed diagnostic threshold shows the tradeoff of recovering these misses; never release it as a newly validated setting.
    exploratory_threshold = min(row['scores']['pca16_uint8'] for row in records)
    tradeoff = dict(threshold=exploratory_threshold,metrics=f.metrics(labels,probs['pca16_uint8'],exploratory_threshold),test_selected=True,deployable_evidence=False)
    result=dict(threshold=threshold,misses=records,metrics_same_frozen_threshold=metrics,group_summaries=summaries,
                exploratory_recovery_tradeoff=tradeoff,test_used_for_diagnosis_not_selection=True,model_or_threshold_changed=False,
                original_predictions_reproduced=True,recording_count=len(ids),window_count=len(x),
                source_lock=lock,method='Paired same-weights same-threshold representation removal; all test WAVs hashed; no training or model selection.')
    output.mkdir(parents=True,exist_ok=True)
    f.save(output/'diagnostic_results.json',result)
    f.save(output/'all_test_recordings.json',audio)
    np.savez_compressed(output/'diagnostic_predictions.npz',ids=ids,labels=labels,groups=groups,starts=starts,**{name+'_window_logits':z for name,z in logits.items()},**{name+'_probabilities':p for name,p in probs.items()})
    print('ERROR_DIAGNOSTIC_DONE',json.dumps(dict(misses=[dict(id=r['id'],scores=r['scores'],percentiles=r['RPW_percentiles'],gain=r['gain_sensitivity_not_retrained']) for r in records],metrics=metrics,tradeoff=tradeoff)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.review,args.audit,args.output)
