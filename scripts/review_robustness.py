"""One bounded audio-volume check on fixed validation finalists; no training or threshold tuning."""
import argparse
from pathlib import Path
import numpy as np
from scripts.review_campaign import read_json,save_json,resolve_audio,review_metrics


def mfcc_with_gain(path,gain,features):
    """Re-extract the baseline MFCC after gain, retaining real log floors and dynamic-range clipping."""
    import librosa
    from scipy.signal import butter,filtfilt
    c=features;signal,_=librosa.load(path,sr=c['sample_rate'],mono=True)
    length=int(c['seconds']*c['sample_rate']);signal=np.pad(signal[:length],(0,max(0,length-len(signal))))
    b,a=butter(c['filter_order'],[c['low_hz'],c['high_hz']],btype='bandpass',fs=c['sample_rate'])
    filtered=filtfilt(b,a,signal)*gain;fft=length//c['frames']
    mel=librosa.feature.melspectrogram(y=filtered,sr=c['sample_rate'],n_fft=fft,hop_length=fft,
        win_length=fft,window=c['window'],center=False,power=c['power'],n_mels=c['n_mels'],
        fmin=c['fmin'],fmax=c['fmax'],htk=False,norm=c['mel_norm'])
    log=librosa.power_to_db(mel,ref=1.,amin=c['log_amin'],top_db=c['log_top_db'])
    return librosa.feature.mfcc(S=log,n_mfcc=c['coefficients'],dct_type=c['dct_type'],norm=c['dct_norm'],lifter=0).T.astype('float32')


def run(settings):
    """Compare half-volume with unmodified audio on the existing accepted validation cohort."""
    import tensorflow as tf
    import pandas as pd
    from scipy.special import expit
    from treevibes.export import tflite_predict
    from treevibes.train import normalized
    out=Path(settings['output']);selection=read_json(out/'audit/selection.json')
    c=read_json(out/'cache/mfcc13/contract.json');root=Path(read_json(out/'audit/audio_source.json')['root'])
    ids=np.load(out/'cache/mfcc13/ids_val.npy');labels=np.load(out/'cache/mfcc13/y_val.npy')
    original=np.load(out/'cache/mfcc13/X_val.npy')
    # Verify the independent extraction path before assigning any robustness result.
    for i in range(3):np.testing.assert_allclose(mfcc_with_gain(resolve_audio(root,str(ids[i])),1.,c['features']),original[i],rtol=1e-5,atol=1e-4)
    quieter=np.stack([mfcc_with_gain(resolve_audio(root,str(identifier)),.5,c['features']) for identifier in ids])
    previous=selection['server'];parent=Path(previous['run_dir']);contract=read_json(parent/'input_contract.json')
    model=tf.keras.models.load_model(parent/'best.keras',compile=False)
    exports=read_json(out/'audit/original_exports.json')
    local=max([r for r in exports if r['method']=='ptq' and 'tiny_temporal_w12' in r['parent_run']],key=lambda r:r['validation']['f1'])
    local_parent=Path(local['parent_run']);local_contract=read_json(local_parent/'input_contract.json')
    rows=[]
    for name,raw in [('original_gain_1',original),('audio_gain_0_5',quieter)]:
        x=normalized(raw,contract)
        probabilities=expit(np.concatenate([model(x[i:i+64,...,None],training=False).numpy().ravel() for i in range(0,len(x),64)]))
        rows.append(dict(model=previous['id'],perturbation=name,**review_metrics(labels,probabilities,previous['threshold'])))
        local_p=tflite_predict(Path(local['directory'])/'model.tflite',normalized(raw,local_contract))
        rows.append(dict(model='tiny_temporal_w12_ptq',perturbation=name,**review_metrics(labels,local_p,local['threshold'])))
    pd.DataFrame(rows).to_csv(out/'volume_check.csv',index=False)
    save_json(out/'volume_check.json',dict(results=rows,thresholds_retuned=False,gating_reapplied=False,
        validation_ids_hash=selection['cohort']['val'],test_set_evaluated=False,
        scope='one synthetic waveform gain perturbation; not a microphone/noise/field validation'))
    print('VOLUME CHECK',[(r['model'],r['perturbation'],r['recall'],r['f1']) for r in rows],flush=True)


def main():
    """Run one optional CPU diagnostic without consuming GPU training memory."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--settings',type=Path,required=True)
    args=parser.parse_args();run(read_json(args.settings))


if __name__=='__main__':main()
