"""Lock one PCA16 pipeline, check fit/leakage, and evaluate the reserved test set once."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
 """Fingerprint locked scientific artifacts, including trained parameters and normalization."""
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path,assets,threshold):
 """Persist the selected operating point before reading final-test arrays; reject later changes."""
 value=dict(assets={name:dict(path=str(p),sha256=digest(p)) for name,p in assets.items()},threshold=float(threshold),aggregation='max logit over original 10s/5s-stride windows plus tail',selection='previously recorded validation F1, no test tuning')
 if path.exists():
  if json.loads(path.read_text())!=value:raise ValueError('Locked finalist changed; test results cannot be retuned')
 else:path.write_text(json.dumps(value,indent=2),encoding='utf-8')
 return value


def audit(review,output):
 """Separate train-fitted PCA reconstruction from classifier fitting and threshold selection."""
 import tensorflow as tf
 import librosa
 from scipy.signal import butter,filtfilt
 from scipy.special import expit
 from sklearn.decomposition import PCA
 from scipy.stats import beta
 sys.path.insert(0,str(review));import window_campaign as frontend
 previous=Path(frontend.read(review/'config.json')['previous']);settings=frontend.read(previous/'settings.json');sys.path.insert(0,settings['project'])
 output.mkdir(parents=True,exist_ok=True)
 row=next(r for r in frontend.read(review/'comparison.json') if r['id']=='server_raw_pca16')
 lock=freeze(output/'locked_finalist.json',dict(model=review/'runs/server_raw/best.keras',pca=review/'runs/server_raw/pca16_codec.npz',normalization=review/'cache/raw/contract.json',window_source=review/'window_campaign.py',audit_source=Path(__file__)),row['threshold'])
 if (output/'audit_results.json').exists():print('REUSING_LOCKED_TEST_AUDIT');return
 profile=np.load(review/'runs/server_raw/pca16_codec.npz');model=tf.keras.models.load_model(review/'runs/server_raw/best.keras',compile=False)
 parts={p:frontend.arrays(review,'raw',p) for p in ['train','val']}
 train=parts['train'][0].reshape(len(parts['train'][0]),-1)
 raw_training=np.load(review/'cache/raw/X_train.npy');norm=frontend.read(review/'cache/raw/contract.json')
 np.testing.assert_allclose(raw_training.mean(axis=(0,1)),norm['mean'],atol=1e-6,rtol=1e-6)
 np.testing.assert_allclose(np.maximum(raw_training.std(axis=(0,1)),1e-6),norm['scale'],atol=1e-6,rtol=1e-6)
 fitted=PCA(n_components=16,svd_solver='full').fit(train)
 np.testing.assert_allclose(fitted.mean_,profile['mean'],atol=1e-6,rtol=1e-6)
 np.testing.assert_allclose(fitted.components_,profile['components'],atol=1e-6,rtol=1e-6)
 projected=fitted.transform(train);lo=projected.min(axis=0);step=np.maximum(np.ptp(projected,axis=0)/255,1e-8)
 np.testing.assert_allclose(lo,profile['lo'],atol=1e-6,rtol=1e-6);np.testing.assert_allclose(step,profile['scale'],atol=1e-6,rtol=1e-6)
 # The lock and train-only reconstruction checks above precede this first final-test access.
 primary=frontend.read(previous/'audit/selection.json')['primary_dataset_id'];manifest=None
 for path in (Path(settings['original'])/'cache').glob('*/contract.json'):
  if frontend.read(path)['dataset_id']==primary:
   contract=frontend.read(path);manifest=frontend.read(path.parent/'manifest.json')
   with np.load(path.parent/'features.npz',allow_pickle=False) as d:
    test_ids=d['ids_test'];test_y=d['y_test'];test_first=d['X_test']
   break
 if manifest is None:raise ValueError('Original split/cache not found')
 assert not set(map(str,test_ids))&set(map(str,parts['train'][3]))
 assert not set(map(str,test_ids))&set(map(str,parts['val'][3]))
 hashes={r['id']:r['sha256'] for r in manifest};by_part={p:{hashes[str(i)] for i in ids} for p,ids in [('train',parts['train'][3]),('val',parts['val'][3]),('test',test_ids)]}
 assert not by_part['train']&by_part['val'] and not by_part['train']&by_part['test'] and not by_part['val']&by_part['test']
 root=Path(frontend.read(previous/'audit/audio_source.json')['root']);b,a=butter(4,[100,2500],btype='bandpass',fs=8000)
 test_features=[];groups=[]
 for i,identifier in enumerate(test_ids):
  path=next((p/str(identifier) for p in [root,root/'field/field/train'] if (p/str(identifier)).is_file()),None)
  if path is None or digest(path)!=hashes[str(identifier)]:raise ValueError('Held-out WAV identity changed')
  signal,_=librosa.load(path,sr=8000,mono=True)
  for start in frontend.window_starts(len(signal)):
   clip=signal[start:start+80000];clip=np.pad(clip,(0,max(0,80000-len(clip))));matrix=frontend.mfcc(filtfilt(b,a,clip))
   if start==0:np.testing.assert_allclose(matrix,test_first[i],atol=1e-4,rtol=0)
   test_features.append(matrix);groups.append(i)
  if (i+1)%100==0:print(f'LOCKED_TEST_FEATURES {i+1}/{len(test_ids)}',flush=True)
 c=frontend.read(review/'cache/raw/contract.json');x=((np.stack(test_features)-np.array(c['mean']))/np.array(c['scale'])).astype('float32')
 parts['test']=(x,np.array(groups),test_y,test_ids)
 results={};reconstruction={}
 for name,(x,groups,y,ids) in parts.items():
  flat=x.reshape(len(x),-1);pc=(flat-profile['mean'])@profile['components'].T;codes=np.clip(np.rint((pc-profile['lo'])/profile['scale']),0,255).astype('uint8')
  decoded=np.frombuffer(codes.tobytes(),dtype='uint8').reshape(codes.shape).astype('float32')*profile['scale']+profile['lo']
  recovered=(decoded@profile['components']+profile['mean']).reshape(x.shape)
  latent=(pc@profile['components']+profile['mean']).reshape(x.shape)
  reconstruction[name]=dict(pca_mse=float(np.mean((latent-x)**2)),encoded_mse=float(np.mean((recovered-x)**2)),feature_variance=float(np.var(x)),clipped_fraction=float(np.mean((pc<profile['lo'])|(pc>profile['lo']+255*profile['scale']))))
  z=np.concatenate([model(recovered[i:i+128,...,None],training=False).numpy().ravel() for i in range(0,len(x),128)])
  p=expit(frontend.aggregate(z,groups,len(y)));m=frontend.metrics(y,p,lock['threshold']);fn=m['false_negatives'];tp=int(sum(y))-fn;n=int(sum(y))
  m['rpw_recall_exact_95ci']=[float(beta.ppf(.025,tp,n-tp+1)) if tp else 0.,float(beta.ppf(.975,tp+1,n-tp)) if tp<n else 1.]
  results[name]=m;np.savez_compressed(output/f'{name}_predictions.npz',ids=ids,labels=y,probabilities=p)
  print('LOCKED_RESULT',name,json.dumps(m),flush=True)
 np.testing.assert_allclose(results['val']['recall'],row['recall'],atol=0,rtol=0)
 assert results['val']['false_positives']==row['false_positives'] and results['val']['false_negatives']==row['false_negatives']
 folder_groups={p:{str(i).rsplit('/',1)[0] for i in d[3]} for p,d in parts.items()}
 folder_overlap=len(folder_groups['train']&folder_groups['test'])
 report=dict(locked_finalist=lock,threshold=lock['threshold'],pca_components=16,original_dimensions=390,pca_variance_explained=float(fitted.explained_variance_ratio_.sum()),train_only_refit_matches_saved_pca=True,normalization_recomputed_from_train_only=True,exact_recording_and_sha_overlap=False,train_test_class_folder_overlap=folder_overlap,grouping=contract['split']['grouping'],metrics=results,reconstruction=reconstruction,train_validation_f1_gap=results['train']['f1']-results['val']['f1'],validation_test_recall_gap=results['val']['recall']-results['test']['recall'],test_set_evaluated=True,test_threshold_retuned=False,train_tree_session_independence_verified=False)
 frontend.save(output/'audit_results.json',report)
 lines=['# Locked PCA16 fit and final-test audit','','The model, PCA profile, normalization, windowing, audit source and validation-selected threshold were hashed and locked before opening the final-test arrays. No model/threshold adjustment used final-test data.','', '| Partition | Recordings | RPW recall | Accuracy | F1 | False alarms | RPW misses |','|---|---:|---:|---:|---:|---:|---:|']
 for name,m in results.items():lines.append(f"| {name} | {m['count']} | {m['recall']:.2%} | {m['accuracy']:.2%} | {m['f1']:.2%} | {m['false_positives']} | {m['false_negatives']} |")
 lines+=['',f"PCA16 retains {report['pca_variance_explained']:.2%} of training-window variance. Re-fitting PCA on training windows exactly reproduced the saved components, mean and quantization ranges. It does not use class labels. The classifier and its threshold determine recall.",'',f"Train–validation F1 gap: {100*report['train_validation_f1_gap']:.2f} percentage points. Reconstruction diagnostics: {reconstruction}.",'',f"No identical recording IDs or audio SHA256 values cross partitions. However, grouping is {report['grouping']}; {folder_overlap} class/folder groups appear in both training and test. Folder groups are not verified tree/session IDs. This is held-out-recording performance, not proven new-tree or farm generalization.",'', '100% validation recall was one operating point across only 109 positives and was accompanied by 25 healthy false alarms. Max pooling gives each recording several chances to trigger. It does not imply 100% accuracy, perfect reconstruction, or absence of fitting/selection bias.','', 'This test set is now evaluated for the locked PCA16 finalist. Preserve this result; future changes need new independent data for a fresh confirmatory test.']
 (output/'audit_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('AUDIT',json.dumps(report),flush=True)


def heatmap(review,output):
 """Replace the padded six-second example with a verified full ten-second crop from training."""
 import librosa
 import soundfile as sf
 from scipy.signal import butter,filtfilt
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 sys.path.insert(0,str(review));import window_campaign as frontend
 previous=Path(frontend.read(review/'config.json')['previous']);source=Path(frontend.read(previous/'audit/audio_source.json')['root'])
 ids=np.load(previous/'cache/mfcc13/ids_train.npy');b,a=butter(4,[100,2500],btype='bandpass',fs=8000)
 for identifier in ids:
  if not str(identifier).startswith('infested/'):continue
  path=next((p/str(identifier) for p in [source,source/'field/field/train'] if (p/str(identifier)).is_file()),None)
  info=sf.info(path)
  if info.duration>=10:break
 signal,_=librosa.load(path,sr=8000,mono=True);filtered=filtfilt(b,a,signal[:80000]);fft=2666
 power=abs(librosa.stft(filtered,n_fft=fft,hop_length=fft,win_length=fft,center=False))**2
 mel=librosa.feature.melspectrogram(S=power,sr=8000,n_fft=fft,n_mels=64,fmin=0,fmax=4000,norm='slaney')
 logs=librosa.power_to_db(mel,ref=1.,amin=1e-10,top_db=80)
 pcen=librosa.pcen(np.sqrt(mel)*32768,sr=8000,hop_length=fft,gain=.98,bias=2,power=.5,time_constant=.4,eps=1e-6)
 mfcc=frontend.mfcc(filtered);stats=frontend.read(previous/'audit/original_input_contract.json')['normalization'];scaled=(mfcc-np.array(stats['mean']))/np.array(stats['scale'])
 assert mfcc.shape==(30,13) and logs.shape==(64,30) and pcen.shape==(64,30)
 plots=output/'plots';plots.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(plots/'same_recording_features.npz',mfcc=mfcc,mfcc_scaled=scaled,mel=logs.T,pcen=pcen.T,recording=str(identifier))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13})
 fig,axes=plt.subplots(1,3,figsize=(15.5,3.2),dpi=170)
 for ax,values,title,label in zip(axes,[scaled.T,logs,pcen],['MFCC (training-scaled)','Log-Mel64','PCEN64'],['Coefficient','Mel band','Mel band']):
  image=ax.imshow(values,aspect='auto',origin='lower',cmap='magma',extent=[0,10,-.5,values.shape[0]-.5]);ax.set_title(title);ax.set_xlabel('Time (seconds)');ax.set_ylabel(label);ax.set_xticks([0,2,4,6,8,10])
 fig.tight_layout();fig.savefig(plots/'feature_heatmaps.png',bbox_inches='tight');plt.close(fig)
 meta=dict(recording=str(identifier),sha256=digest(path),duration_seconds=info.duration,displayed_seconds=10,padding_seconds=0,partition='train',sample_rate=8000,mfcc_display='Original training mean/std scaling; raw values retained in NPZ',source_dimensions=dict(mfcc=[30,13],mel=[30,64],pcen=[30,64]))
 frontend.save(output/'feature_example.json',meta);print('HEATMAP',json.dumps(meta),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();audit(args.review,args.output);heatmap(args.review,args.output)
