"""Run the isolated professor-review study without modifying the original campaign.

Only train/validation arrays are loaded. The existing allocation and accepted recording
IDs define every comparison; final-test labels remain outside this campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np


def read_json(path):
    """Read UTF-8 artifacts consistently on native Windows and the local review host."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    """Commit evidence atomically so interrupted workers cannot appear completed."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def cohort_digest(ids):
    """Fingerprint ordered recording IDs rather than comparing only sample counts."""
    return hashlib.sha256(json.dumps(list(map(str,ids)),ensure_ascii=True).encode()).hexdigest()


def fit_scaling(x):
    """Fit channel scaling on training frames; keep constant channels numerically safe."""
    mean=x.mean(axis=(0,1),dtype=np.float64).astype('float32')
    scale=x.std(axis=(0,1),dtype=np.float64).astype('float32')
    return mean,np.where(scale<1e-6,1.,scale).astype('float32')


def review_metrics(labels, probabilities, threshold):
    """Retain RPW false negatives and healthy false alarms alongside headline metrics."""
    from treevibes.evaluate import evaluate_probabilities
    m=evaluate_probabilities(np.asarray(labels),np.asarray(probabilities),threshold)
    tn,fp=m['confusion'][0];fn,tp=m['confusion'][1]
    return m|dict(false_negatives=fn,false_positives=fp,fpr=fp/max(tn+fp,1))


def resolve_audio(root, identifier):
    """Resolve an exact relative recording identity, never an ambiguous filename match."""
    root=Path(root)
    for prefix in [root,root/'field/field/train']:
        path=prefix/identifier
        if path.is_file():return path
    raise FileNotFoundError(f'Original WAV is missing: {identifier}')



def read_cache(snapshot, identifier):
    """Verify existing cache bytes while opening only training/validation arrays."""
    for path in (Path(snapshot)/'cache').glob('*/contract.json'):
        contract=read_json(path)
        if contract['dataset_id']!=identifier:continue
        payload=path.parent/'features.npz'
        digest=hashlib.sha256()
        with payload.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=contract['payload_sha256']:raise ValueError('Original feature cache changed')
        with np.load(payload,allow_pickle=False) as archive:
            data={key:archive[key] for key in ['X_train','y_train','ids_train','X_val','y_val','ids_val']}
        return data|dict(contract=contract)
    raise ValueError(f'No matching cache: {identifier}')

def audit(settings):
    """Freeze the canonical 131-recipe results, verifying checkpoint and prediction integrity."""
    import pandas as pd
    from scripts.windows_support import valid_imported_run
    old=Path(settings['original']);out=Path(settings['output']);out.mkdir(parents=True,exist_ok=True)
    summary=read_json(old/'campaign_summary.json')
    if summary['state'] not in {'completed','completed_with_failures'}:
        raise RuntimeError('Original campaign is active or incomplete; audit only, do not start workers.')
    save_json(out/'audit/original_campaign_summary.json',summary)
    primary=summary['primary_dataset_id'];data=read_cache(old,primary)
    ids={part:cohort_digest(data[f'ids_{part}']) for part in ['train','val']}
    rows=[]
    for entry in summary['runs']:
        directory=Path(entry['run_dir']);valid=valid_imported_run(directory)
        c=read_json(directory/'config.json');r=read_json(directory/'training_result.json')
        contract=read_json(directory/'input_contract.json')
        history=r.get('history',[])
        with np.load(directory/'validation_predictions.npz',allow_pickle=False) as p:
            labels=p['labels'];probs=p['probabilities']
        metrics=review_metrics(labels,probs,r['validation']['threshold'])
        same=contract['dataset_id']==primary
        # Different feature caches may still share exactly the same recordings/order.
        if not same:
            other=read_cache(old,contract['dataset_id'])
            same=all(cohort_digest(other[f'ids_{part}'])==ids[part] for part in ['train','val'])
        row=dict(id=c['id'],status='completed' if valid else 'invalid_artifacts',run_dir=str(directory),
                 family=c['family'],target=c['target'],dataset_id=contract['dataset_id'],
                 feature_shape=f"{c['features']['frames']}x{c['features']['coefficients']}",
                 seed=c['seed'],epochs_completed=r.get('epochs_completed',0),
                 best_epoch=min(history,key=lambda x:x['val_loss'])['epoch'] if history else None,
                 same_evaluation_cohort=same,validation_ids_hash=ids['val'] if same else None,**metrics)
        rows.append(row)
        target=out/'audit/models'/c['id'];target.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','input_contract.json','training_result.json','validation_predictions.npz','provenance.json','epoch_state.json']:
            if (directory/name).exists():shutil.copy2(directory/name,target/name)
    if len({r['id'] for r in rows})!=summary['planned_runs']:
        raise ValueError('Canonical campaign recipe count does not match recorded rows')
    pd.DataFrame(rows).to_csv(out/'previous_results.csv',index=False)
    save_json(out/'audit/previous_results.json',rows)
    candidates=[r for r in rows if r['status']=='completed' and r['same_evaluation_cohort'] and (Path(r['run_dir'])/'best.keras').exists()]
    rank=lambda r:(r['f1'],r['recall'],-r['false_positives'])
    server=max([r for r in candidates if r['target']=='remote'],key=rank)
    local=max([r for r in candidates if r['target']=='local' and r['feature_shape']=='30x13'],key=rank)
    selection=dict(server=server,local=local,primary_dataset_id=primary,cohort=ids,
                   validation_count=len(data['y_val']),train_count=len(data['y_train']),
                   classes=['healthy','RPW'],test_set_evaluated=False,
                   grouping=data['contract']['split']['grouping'])
    save_json(out/'audit/selection.json',selection)
    # Preserve the allocation, including held-out identities, without opening test arrays.
    save_json(out/'audit/original_split.json',data['contract']['split'])
    save_json(out/'audit/original_input_contract.json',data['contract'])
    save_json(out/'audit/original_exports.json',summary['exports'])
    print('AUDIT',json.dumps({k:v['id'] if isinstance(v,dict) and 'id' in v else v for k,v in selection.items()}),flush=True)
    return selection


def cache_baseline(settings):
    """Reuse cached MFCC bytes and exact training statistics for model continuation."""
    out=Path(settings['output']);selection=read_json(out/'audit/selection.json')
    data=read_cache(Path(settings['original']),selection['primary_dataset_id'])
    destination=out/'cache/mfcc13';destination.mkdir(parents=True,exist_ok=True)
    c=data['contract'];stats=c['normalization']
    for part in ['train','val']:
        np.save(destination/f'X_{part}.npy',data[f'X_{part}'])
        np.save(destination/f'y_{part}.npy',data[f'y_{part}'])
        np.save(destination/f'ids_{part}.npy',data[f'ids_{part}'])
    save_json(destination/'contract.json',dict(representation='mfcc13',shape=[30,13],
        features=c['features'],mean=stats['mean'],scale=stats['scale'],
        train_ids_hash=selection['cohort']['train'],validation_ids_hash=selection['cohort']['val'],
        original_dataset_id=c['dataset_id'],source='original_feature_cache'))
    return destination


def prepare_features(settings):
    """Extract comparable features and high-information images from the same accepted WAVs."""
    import librosa
    from scipy.signal import butter,filtfilt
    from scipy.ndimage import zoom
    out=Path(settings['output']);selection=read_json(out/'audit/selection.json')
    base=out/'cache/mfcc13';c=read_json(base/'contract.json')['features']
    identifiers={p:np.load(base/f'ids_{p}.npy',allow_pickle=False) for p in ['train','val']}
    names=dict(mfcc20=(30,20),mfcc40=(30,40),mel32=(30,32),mel64=(30,64),pcen64=(30,64),stft=(30,1334),image_mel=(299,299))
    complete=out/'cache/features_ready.json'
    if complete.exists():return read_json(complete)
    if settings.get('dataset_root'):
        root=Path(settings['dataset_root'])
    else:
        import kagglehub
        root=Path(kagglehub.dataset_download('potamitis/treevibes'))
    save_json(out/'audit/audio_source.json',dict(root=str(root),sample_rate=8000,screen_seconds=10,
                                               unrestricted_seconds_cap=30))
    manifest=read_json(next(p for p in (Path(settings['original'])/'cache').glob('*/manifest.json')
                            if read_json(p.parent/'contract.json')['dataset_id']==selection['primary_dataset_id']))
    expected={r['id']:r['sha256'] for r in manifest}
    arrays={}
    for name,shape in names.items():
        dest=out/'cache'/name;dest.mkdir(exist_ok=True)
        arrays[name]={}
        for part in ['train','val']:
            arrays[name][part]=np.lib.format.open_memmap(dest/f'X_{part}.npy',mode='w+',dtype='float16' if name=='image_mel' else 'float32',shape=(len(identifiers[part]),*shape))
            for prefix in ['y','ids']:shutil.copy2(base/f'{prefix}_{part}.npy',dest/f'{prefix}_{part}.npy')
    b,a=butter(4,[100,2500],btype='bandpass',fs=8000)
    first=True;done=0;start=time.monotonic()
    for part in ['train','val']:
        for index,identifier in enumerate(identifiers[part]):
            path=resolve_audio(root,str(identifier))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[str(identifier)]:
                raise ValueError(f'WAV identity changed: {identifier}')
            signal,_=librosa.load(path,sr=8000,mono=True)
            clip=np.pad(signal[:80000],(0,max(0,80000-len(signal))))
            filtered=filtfilt(b,a,clip)
            fft=80000//30
            power=abs(librosa.stft(filtered,n_fft=fft,hop_length=fft,win_length=fft,center=False))**2
            mel={n:librosa.feature.melspectrogram(S=power,sr=8000,n_fft=fft,n_mels=n,fmin=0,fmax=4000,norm='slaney') for n in [32,64,128]}
            logs={n:librosa.power_to_db(mel[n],ref=1.,amin=1e-10,top_db=80) for n in mel}
            for n in [20,40]:arrays[f'mfcc{n}'][part][index]=librosa.feature.mfcc(S=logs[128],n_mfcc=n).T
            for n in [32,64]:arrays[f'mel{n}'][part][index]=logs[n].T
            pcen=librosa.pcen(np.sqrt(mel[64])*32768,sr=8000,hop_length=fft,gain=.98,bias=2,power=.5,time_constant=.4,eps=1e-6)
            arrays['pcen64'][part][index]=pcen.T
            arrays['stft'][part][index]=librosa.power_to_db(power,ref=1.,top_db=80).T
            # Full original audio (up to 30 s), retaining short-time events before resizing.
            full=filtfilt(b,a,signal[:240000])
            high=librosa.feature.melspectrogram(y=full,sr=8000,n_fft=512,hop_length=128,n_mels=128,fmin=0,fmax=4000)
            image=librosa.power_to_db(high,ref=np.max,top_db=80)
            image=zoom(image,(299/image.shape[0],299/image.shape[1]),order=1)
            arrays['image_mel'][part][index]=image.astype('float16')
            if first:
                np.savez_compressed(out/'plots/same_recording_features.npz',mfcc=np.load(base/'X_train.npy',mmap_mode='r')[0],mel=logs[64].T,pcen=pcen.T,recording=str(identifier))
                first=False
            done+=1
            if done%100==0:print(f'FEATURES {done}/2070, {(time.monotonic()-start)/60:.1f} min',flush=True)
    for name,parts in arrays.items():
        for array in parts.values():array.flush()
        if name=='image_mel':mean,scale=0.,1.
        else:
            mean,scale=fit_scaling(parts['train']);mean=mean.tolist();scale=scale.tolist()
        save_json(out/'cache'/name/'contract.json',dict(representation=name,shape=list(names[name]),mean=mean,scale=scale,
            train_ids_hash=selection['cohort']['train'],validation_ids_hash=selection['cohort']['val'],
            features=dict(sample_rate=8000,seconds=30 if name=='image_mel' else 10,bandpass=[100,2500],filter='filtfilt_order4',
                fft_size=512 if name=='image_mel' else fft,hop=128 if name=='image_mel' else fft,center=False if name!='image_mel' else True,
                power=2,mel_norm='slaney',pcen_amplitude_multiplier=32768 if name=='pcen64' else None,
                pcen=dict(gain=.98,bias=2,power=.5,time_constant=.4,eps=1e-6) if name=='pcen64' else None,
                image_resize='bilinear frequency/time to 299x299; replicate one channel to RGB; dB [-80,0] to [-1,1]' if name=='image_mel' else None)))
    save_json(complete,dict(representations=list(names),recordings=done,elapsed_seconds=time.monotonic()-start))
    return read_json(complete)


def bridge_job(request):
    """Serialize Keras graphs and INT8 models in the CPU worker, away from CUDA allocations."""
    import tensorflow as tf
    from scripts.torch_bridge import dump_model,apply_weights
    from treevibes.models import build_model
    directory=Path(request['directory']);directory.mkdir(parents=True,exist_ok=True)
    if request['action']=='bridge_dump':
        c=request['config'];tf.keras.utils.set_random_seed(c['seed'])
        model=tf.keras.models.load_model(request['model'],compile=False) if request.get('model') else build_model(c)
        dump_model(model,directory);model.save(directory/'template.keras')
    elif request['action']=='bridge_restore':
        model=tf.keras.models.load_model(directory/'template.keras',compile=False);apply_weights(model,request['weights'])
        with np.load(directory.parent/'bridge_parity.npz') as p:
            np.testing.assert_allclose(model(p['features'][...,None],training=False).numpy().ravel(),p['logits'],rtol=2e-4,atol=5e-5)
        model.save(request['model'])
    elif request['action']=='int8':
        from treevibes.export import tflite_predict,_operators,_details
        from treevibes.evaluate import select_threshold
        model=tf.keras.models.load_model(request['model'],compile=False)
        cache=Path(request['cache']);contract=read_json(cache/'contract.json')
        x=(np.load(cache/'X_train.npy')-np.array(contract['mean']))/np.array(contract['scale'])
        xv=(np.load(cache/'X_val.npy')-np.array(contract['mean']))/np.array(contract['scale'])
        x=x.astype('float32');xv=xv.astype('float32');yv=np.load(cache/'y_val.npy')
        converter=tf.lite.TFLiteConverter.from_keras_model(model);converter.optimizations=[tf.lite.Optimize.DEFAULT]
        indices=np.random.default_rng(42).permutation(len(x))[:128]
        def representative():
            """Calibrate only on training observations."""
            for i in indices:yield [x[i:i+1,...,None]]
        converter.representative_dataset=representative
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
        contents=converter.convert();target=directory/'model.tflite';target.write_bytes(contents)
        interpreter=tf.lite.Interpreter(model_content=contents);interpreter.allocate_tensors()
        if any(np.issubdtype(t['dtype'],np.floating) for t in interpreter.get_tensor_details()):raise ValueError('Float tensors remain in INT8 export')
        p=tflite_predict(target,xv);threshold=select_threshold(yv,p)
        np.savez_compressed(directory/'validation_predictions.npz',labels=yv,probabilities=p,ids=np.load(cache/'ids_val.npy'))
        meta=dict(model_bytes=len(contents),model_sha256=hashlib.sha256(contents).hexdigest(),board_verified=False,
                  operators=_operators(contents),input=_details(interpreter.get_input_details()[0]),
                  output=_details(interpreter.get_output_details()[0]),**review_metrics(yv,p,threshold))
        save_json(directory/'export.json',meta)
    else:raise ValueError('Unknown CPU bridge action')


def worker(settings, request, framework='torch'):
    """Run one bounded worker, retaining logs and releasing framework memory afterward."""
    out=Path(settings['output']);name=request.get('id',request['action'])
    request_path=out/'configs'/f'{name}_{request["action"]}.json';save_json(request_path,request)
    env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([settings['tools_parent'],settings['project'],str(out/'dependencies')]),
        PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',TF_USE_LEGACY_KERAS='1',
        TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        HF_HOME=str(out/'pretrained_cache'))
    if framework=='tf':env['CUDA_VISIBLE_DEVICES']='-1'
    script=Path(__file__).with_name('review_torch.py') if framework=='torch' else Path(__file__)
    command=[settings[f'{framework}_python'],str(script),'--request',str(request_path)]
    from scripts.windows_support import run_logged
    run_logged(command,out/'logs'/f'{name}_{request["action"]}.log',env=env,stream=True,cwd=settings['project'])


def run_campaign(settings):
    """Prioritize finished evidence: extensions, core features, unrestricted model, then follow-ups."""
    out=Path(settings['output']);out.mkdir(parents=True,exist_ok=True)
    selection=audit(settings);cache_baseline(settings)
    failures=[]
    def train(spec):
        """Skip intact completed review runs and record failures without discarding other evidence."""
        directory=out/'checkpoints'/spec['id'];spec=spec|dict(directory=str(directory),output=str(out))
        if (directory/'result.json').exists():print('REUSE',spec['id'],flush=True);return
        save_json(out/'progress.json',dict(state='running',active=spec['id'],failures=failures))
        try:
            if spec.get('family')!='inception_resnet_v2':
                graph=directory/'graph';graph.mkdir(parents=True,exist_ok=True)
                worker(settings,dict(action='bridge_dump',id=spec['id'],directory=str(graph),config=spec['model_config'],
                                     model=str(Path(spec['origin'])/'best.keras') if spec.get('origin') else None),framework='tf')
            worker(settings,spec|dict(action='train'))
        except Exception as error:
            failures.append(dict(id=spec['id'],error=str(error)));save_json(directory/'failure.json',failures[-1])
            print('RECORDED FAILURE',spec['id'],str(error)[-1000:],flush=True)
    for route in ['server','local']:
        previous=selection[route];c=read_json(Path(previous['run_dir'])/'config.json')
        train(dict(id=f'extend_{route}',kind='extension',family=c['family'],model_config=c,
                   origin=previous['run_dir'],cache=str(out/'cache/mfcc13'),epochs=200,patience=20,route=route,
                   learning_rate_factor=1.,seed=c['seed'],batch_size=c['batch_size']))
    prepare_features(settings)
    baseline=next(r for r in read_json(out/'audit/previous_results.json') if r['id']=='cnn_w16')
    c=read_json(Path(baseline['run_dir'])/'config.json')
    def screen(name,epochs=60,width=16,kind='feature_screen'):
        """Use identical CNN channels, seed, optimizer and audio coverage across screens."""
        contract=read_json(out/'cache'/name/'contract.json');model=c|dict(width=width)
        model['features']=model['features']|dict(frames=contract['shape'][0],coefficients=contract['shape'][1])
        train(dict(id=f'{kind}_{name}',kind=kind,family='cnn',model_config=model,
                   cache=str(out/'cache'/name),epochs=epochs,patience=15 if epochs==60 else 20,
                   route='server',seed=42,batch_size=16))
    for name in ['mfcc13','mel64','pcen64']:screen(name)
    train(dict(id='unrestricted_inception_resnet_v2',kind='unrestricted',family='inception_resnet_v2',
               cache=str(out/'cache/image_mel'),epochs=40,head_epochs=5,patience=10,seed=42,
               batch_size=2,accumulation=8,route='unrestricted',learning_rate=1e-4,
               backbone='inception_resnet_v2.tf_in1k',pretrained=True))
    if settings.get('additional_features',True):
        for name in ['mfcc20','mfcc40','mel32','stft']:screen(name)
    screens=[read_json(p) for p in (out/'checkpoints').glob('feature_screen_*/result.json') if 'mfcc13' not in str(p)]
    # Advance two non-baseline features using the declared recall/F1/false-alarm ranking.
    for r in sorted(screens,key=lambda r:(r['recall'],r['f1'],-r['fpr']),reverse=True)[:2]:
        screen(r['representation'],epochs=120,width=32,kind='feature_full')
    save_json(out/'progress.json',dict(state='completed' if not failures else 'completed_with_failures',failures=failures))
    print('REVIEW TRAINING COMPLETE',json.dumps(failures),flush=True)


def main():
    """Separate CPU conversion jobs from the campaign controller and CUDA fitting workers."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path);parser.add_argument('--request',type=Path)
    args=parser.parse_args()
    if args.request:bridge_job(read_json(args.request))
    elif args.settings:run_campaign(read_json(args.settings))
    else:parser.error('Provide --settings or --request')


if __name__=='__main__':main()
