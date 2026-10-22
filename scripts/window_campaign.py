"""Compare full-recording MIL and gain normalization without touching prior campaigns.

Every window stays with its original recording. Max-logit pooling lets a positive
recording contain quiet windows without labeling every window as RPW activity.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def save(path,value):
 """Write progress atomically so interruption cannot masquerade as completion."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def read(path):
 """Load recorded settings rather than assuming notebook interpreter paths."""
 return json.loads(Path(path).read_text(encoding='utf-8'))


def window_starts(length,width=80000,stride=40000):
 """Cover the full file with overlapping windows, retaining a unique final tail."""
 if length<=0:raise ValueError('Empty recording')
 last=max(0,length-width)
 return sorted(set(list(range(0,last+1,stride))+[last]))


def gain_normalize(signal,target=.01):
 """Remove scalar gain per window while keeping silence finite and unamplified."""
 signal=np.asarray(signal)
 rms=float(np.sqrt(np.mean(signal.astype('float64')**2)))
 return (signal*(target/max(rms,1e-8))).astype('float32')


def aggregate(logits,record_indices,count):
 """Pool actual windows only; no padded window may create a positive prediction."""
 result=np.full(count,-np.inf,dtype='float64')
 np.maximum.at(result,record_indices,np.asarray(logits).ravel())
 if not np.isfinite(result).all():raise ValueError('Missing or nonfinite recording windows')
 return result


def check_cohort(train,val):
 """Reject duplicate identities and recording leakage before extracting any crops."""
 if len(set(train))!=len(train) or len(set(val))!=len(val) or set(train)&set(val):
  raise ValueError('Duplicate recording or train/validation leakage')


def metrics(labels,probabilities,threshold):
 """Expose missed RPW and healthy false alarms alongside threshold-dependent F1."""
 from treevibes.evaluate import evaluate_probabilities
 result=evaluate_probabilities(labels,probabilities,float(threshold))
 result.update(false_positives=int(result['confusion'][0][1]),false_negatives=int(result['confusion'][1][0]))
 return result


def mfcc(filtered):
 """Keep the original reference frontend so only coverage and gain handling change."""
 import librosa
 mel=librosa.feature.melspectrogram(y=filtered,sr=8000,n_fft=2666,hop_length=2666,win_length=2666,window='hann',center=False,power=2.,n_mels=128,fmin=0.,fmax=4000.,htk=False,norm='slaney')
 return librosa.feature.mfcc(S=librosa.power_to_db(mel,ref=1.,amin=1e-10,top_db=80.),n_mfcc=13,dct_type=2,norm='ortho',lifter=0).T.astype('float32')


def prepare(previous,output):
 """Extract raw/gain-normalized crops for train and validation only, with WAV hash checks."""
 import librosa
 from scipy.signal import butter,filtfilt
 if (output/'cache/ready.json').exists():return
 c=read(previous/'cache/mfcc13/contract.json');selection=read(previous/'audit/selection.json')
 settings=read(previous/'settings.json');source=Path(read(previous/'audit/audio_source.json')['root'])
 ids={p:np.load(previous/f'cache/mfcc13/ids_{p}.npy') for p in ['train','val']}
 check_cohort(list(ids['train']),list(ids['val']))
 expected={}
 for p in (Path(settings['original'])/'cache').glob('*/contract.json'):
  if read(p)['dataset_id']==selection['primary_dataset_id']:
   expected={r['id']:r['sha256'] for r in read(p.parent/'manifest.json')};break
 if not expected:raise ValueError('Original audio manifest missing')
 b,a=butter(4,[100,2500],btype='bandpass',fs=8000);started=time.monotonic();counts={}
 for part in ['train','val']:
  arrays={'raw':[],'gain':[],'raw_half':[],'gain_half':[]};record_indices=[];starts=[]
  for i,identifier in enumerate(ids[part]):
   relative=str(identifier);path=next((p/relative for p in [source,source/'field/field/train'] if (p/relative).is_file()),None)
   if path is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[relative]:raise ValueError(f'WAV identity mismatch: {relative}')
   signal,_=librosa.load(path,sr=8000,mono=True)
   if not np.isfinite(signal).all():raise ValueError(relative)
   for start in window_starts(len(signal)):
    clip=signal[start:start+80000];clip=np.pad(clip,(0,max(0,80000-len(clip))))
    filtered=filtfilt(b,a,clip)
    raw=mfcc(filtered);gain=mfcc(gain_normalize(filtered))
    arrays['raw'].append(raw);arrays['gain'].append(gain)
    if part=='val':
     arrays['raw_half'].append(mfcc(filtered*.5));arrays['gain_half'].append(mfcc(gain_normalize(filtered*.5)))
    if start==0:
     original=np.load(previous/f'cache/mfcc13/X_{part}.npy',mmap_mode='r')[i]
     np.testing.assert_allclose(raw,original,atol=1e-4,rtol=0)
    record_indices.append(i);starts.append(start)
   if (i+1)%200==0:print(f'WINDOW_FEATURES {part} {i+1}/{len(ids[part])} {time.monotonic()-started:.0f}s',flush=True)
  counts[part]=dict(recordings=len(ids[part]),windows=len(starts))
  for variant,values in arrays.items():
   if not values:continue
   dest=output/'cache'/variant;dest.mkdir(parents=True,exist_ok=True)
   np.save(dest/f'X_{part}.npy',np.stack(values));np.save(dest/f'record_{part}.npy',record_indices);np.save(dest/f'starts_{part}.npy',starts)
   np.save(dest/f'ids_{part}.npy',ids[part]);np.save(dest/f'y_{part}.npy',np.load(previous/f'cache/mfcc13/y_{part}.npy'))
 for variant in ['raw','gain']:
  train=np.load(output/f'cache/{variant}/X_train.npy');mean=train.mean(axis=(0,1));scale=train.std(axis=(0,1));scale=np.maximum(scale,1e-6)
  save(output/f'cache/{variant}/contract.json',dict(mean=mean.tolist(),scale=scale.tolist(),representation='mfcc13',shape=[30,13],gain_normalization='filtered per-window RMS to 0.01' if variant=='gain' else 'none',original_cohort=selection['cohort'],feature_settings=c['features']))
 import tensorflow as tf
 from scripts.torch_bridge import dump_model
 exports=read(previous/'audit/original_exports.json')
 local=next(r for r in exports if Path(r['parent_run']).parent.parent.name=='tiny_residual_w12' and r['method']=='ptq')
 sources={'server':previous/'checkpoints/extend_server/best.keras','local':Path(local['parent_run'])/'best.keras'}
 for route,path in sources.items():
  graph=output/'graphs'/route;model=tf.keras.models.load_model(path,compile=False);dump_model(model,graph);model.save(graph/'template.keras')
 save(output/'cache/ready.json',dict(counts=counts,elapsed_seconds=time.monotonic()-started,original_first_windows_match=True,test_set_evaluated=False,sources={k:str(v) for k,v in sources.items()}))


def arrays(output,variant,part,half=False,original_stats=None):
 """Load window features and keep labels at recording grain rather than duplicating them."""
 cache=output/'cache'/variant;c=original_stats or read(cache/'contract.json')
 source=output/'cache'/(variant+'_half') if half else cache
 x=np.load(source/f'X_{part}.npy');x=((x-np.array(c['mean']))/np.array(c['scale'])).astype('float32')
 return x,np.load(cache/f'record_{part}.npy'),np.load(cache/f'y_{part}.npy'),np.load(cache/f'ids_{part}.npy')


def train(output,route,variant):
 """Fine-tune frozen-architecture weights with a recording-level multiple-instance loss on CUDA."""
 import torch
 from scripts.torch_adapter import GraphModel,save_keras_weights
 from scipy.special import expit
 from treevibes.evaluate import select_threshold
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training fallback')
 directory=output/'runs'/f'{route}_{variant}';directory.mkdir(parents=True,exist_ok=True)
 if (directory/'result.json').exists():return
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
 model=GraphModel(output/'graphs'/route).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-7)
 x,groups,y,ids=arrays(output,variant,'train');xv,gv,yv,idv=arrays(output,variant,'val')
 train_groups=[np.flatnonzero(groups==i) for i in range(len(y))]
 xgpu=torch.tensor(x,device='cuda');vgpu=torch.tensor(xv,device='cuda')
 def infer(tensor,group,n):
  """Aggregate bounded inference batches into one max-logit score per recording."""
  model.eval()
  with torch.no_grad():z=np.concatenate([model(tensor[i:i+128]).cpu().numpy().ravel() for i in range(0,len(tensor),128)])
  return aggregate(z,group,n)
 history=[];stale=0;best_loss=float('inf');started=time.monotonic();start_epoch=0
 if (directory/'last.pt').exists():
  state=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];stale=state['stale'];best_loss=state['best_loss'];start_epoch=len(history)
  for state in optimizer.state.values():
   for k,v in state.items():
    if torch.is_tensor(v):state[k]=v.cuda()
 for epoch in range(start_epoch,40):
  if stale>=10:break
  rng=np.random.default_rng(42+epoch);torch.manual_seed(42+epoch);order=rng.permutation(len(y));model.train();total=0.;clock=time.monotonic()
  for begin in range(0,len(order),16):
   selected=order[begin:begin+16];indices=np.concatenate([train_groups[i] for i in selected]);lengths=[len(train_groups[i]) for i in selected]
   optimizer.zero_grad(set_to_none=True);z=model(xgpu[indices]).ravel()
   pooled=torch.stack([a.max() for a in torch.split(z,lengths)])
   loss=torch.nn.functional.binary_cross_entropy_with_logits(pooled,torch.tensor(y[selected],device='cuda',dtype=torch.float32))
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   loss.backward();optimizer.step();total+=float(loss.detach())*len(selected)
  z=infer(vgpu,gv,len(yv));val_loss=float(np.mean(np.logaddexp(0,z)-yv*z));improved=val_loss<best_loss-1e-6
  stale=0 if improved else stale+1
  if improved:
   best_loss=val_loss;torch.save({k:v.cpu().clone() for k,v in model.state_dict().items()},directory/'best.pt')
  history.append(dict(epoch=epoch+1,train_loss=total/len(y),val_loss=val_loss,seconds=time.monotonic()-clock,learning_rate=optimizer.param_groups[0]['lr']))
  if stale and stale%4==0:
   for g in optimizer.param_groups:g['lr']=max(g['lr']*.5,1e-6)
  payload=dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),history=history,stale=stale,best_loss=best_loss)
  tmp=directory/'last.tmp.pt';torch.save(payload,tmp);tmp.replace(directory/'last.pt');save(directory/'history.json',history)
  print(f'WINDOW_TRAIN {route}_{variant} CUDA {epoch+1}/40 val_loss={val_loss:.5f} {history[-1]["seconds"]:.1f}s',flush=True)
 model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=True));z=infer(vgpu,gv,len(yv));p=expit(z);threshold=select_threshold(yv,p)
 save_keras_weights(model,directory/'best_weights.npz');np.savez_compressed(directory/'validation_predictions.npz',labels=yv,ids=idv,probabilities=p)
 fixture=xv[:8];model.eval()
 with torch.no_grad():probe=model(torch.tensor(fixture,device='cuda')).cpu().numpy().ravel()
 np.savez(directory/'parity.npz',features=fixture,logits=probe)
 result=dict(id=f'{route}_{variant}',route=route,variant=variant,initialization='same original best architecture weights in both gain variants; fresh Adam',aggregation='max logit over all windows',max_epochs=40,patience=10,best_epoch=min(history,key=lambda r:r['val_loss'])['epoch'],epochs_completed=len(history),training_seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(0),test_set_evaluated=False,**metrics(yv,p,threshold))
 save(directory/'result.json',result);print('RESULT',json.dumps(result),flush=True)


def evaluate(previous,output):
 """Measure float, actual INT8, PCA16 transmission, and half-volume behavior on all validation files."""
 import tensorflow as tf
 from scipy.special import expit
 from sklearn.decomposition import PCA
 from treevibes.evaluate import select_threshold
 from treevibes.export import tflite_predict
 from scripts.torch_bridge import apply_weights
 rows=[];original=read(previous/'cache/mfcc13/contract.json');old_norm=read(previous/'audit/original_input_contract.json')['normalization']
 def score(model,x):
  """Keep CPU inference allocations small and reproducible."""
  return np.concatenate([model(x[i:i+128,...,None],training=False).numpy().ravel() for i in range(0,len(x),128)])
 def record(name,z,y,ids,groups,threshold=None,**extra):
  """Save prediction evidence at recording level for every comparison row."""
  p=expit(aggregate(z,groups,len(y)));threshold=select_threshold(y,p) if threshold is None else threshold
  result=dict(id=name,threshold_policy='validation F1' if extra.pop('selected_threshold',False) else 'fixed from corresponding clean validation',test_set_evaluated=False,**metrics(y,p,threshold),**extra)
  rows.append(result);np.savez_compressed(output/'predictions'/f'{name}.npz',ids=ids,labels=y,probabilities=p);return result
 (output/'predictions').mkdir(exist_ok=True)
 for route in ['server','local']:
  model=tf.keras.models.load_model(output/'graphs'/route/'template.keras',compile=False)
  x,g,y,ids=arrays(output,'raw','val',original_stats=old_norm);z=score(model,x)
  for first_only in [True,False]:
   keep=np.array([np.flatnonzero(g==i)[0] for i in range(len(y))]) if first_only else np.arange(len(g))
   record(f'frozen_{route}_'+('first10' if first_only else 'all_windows'),z[keep],y,ids,g[keep],selected_threshold=True)
 for route in ['server','local']:
  for variant in ['raw','gain']:
   name=f'{route}_{variant}';directory=output/'runs'/name;model=tf.keras.models.load_model(output/'graphs'/route/'template.keras',compile=False);apply_weights(model,directory/'best_weights.npz')
   probe=np.load(directory/'parity.npz');np.testing.assert_allclose(score(model,probe['features']),probe['logits'],rtol=2e-4,atol=5e-5)
   model.save(directory/'best.keras');x,g,y,ids=arrays(output,variant,'val');xh,_,_,_=arrays(output,variant,'val',half=True);xt,_,_,_=arrays(output,variant,'train')
   result=record(name+'_float',score(model,x),y,ids,g,selected_threshold=True,model_bytes=(directory/'best.keras').stat().st_size)
   record(name+'_float_half',score(model,xh),y,ids,g,threshold=result['threshold'])
   if route=='server':
    pca=PCA(n_components=16,svd_solver='full').fit(xt.reshape(len(xt),-1));projection=pca.transform(xt.reshape(len(xt),-1));lo=projection.min(axis=0);step=np.maximum(np.ptp(projection,axis=0)/255,1e-8)
    np.savez_compressed(directory/'pca16_codec.npz',mean=pca.mean_,components=pca.components_,lo=lo,scale=step)
    def decode(value):
     """Actually encode bytes and reconstruct MFCCs using only training-fitted PCA ranges."""
     codes=np.clip(np.rint((pca.transform(value.reshape(len(value),-1))-lo)/step),0,255).astype('uint8')
     restored=np.frombuffer(codes.tobytes(),dtype='uint8').reshape(codes.shape)*step+lo
     return pca.inverse_transform(restored).reshape(value.shape).astype('float32')
    result=record(name+'_pca16',score(model,decode(x)),y,ids,g,selected_threshold=True,feature_bytes_per_window=16)
    record(name+'_pca16_half',score(model,decode(xh)),y,ids,g,threshold=result['threshold'],feature_bytes_per_window=16)
   else:
    converter=tf.lite.TFLiteConverter.from_keras_model(model);converter.optimizations=[tf.lite.Optimize.DEFAULT]
    calibration=xt[np.linspace(0,len(xt)-1,min(512,len(xt)),dtype=int)]
    def representative():
     """Calibrate INT8 ranges using training windows exclusively."""
     for value in calibration:yield [value[None,...,None].astype('float32')]
    converter.representative_dataset=representative;converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    contents=converter.convert();target=directory/'model.tflite';target.write_bytes(contents)
    interpreter=tf.lite.Interpreter(model_path=str(target));interpreter.allocate_tensors()
    if any(np.issubdtype(t['dtype'],np.floating) for t in interpreter.get_tensor_details()):raise ValueError('Float tensors remain in local INT8 model')
    def local_logits(value):
     """Convert exported probabilities back to logits only for recording-level max pooling."""
     p=tflite_predict(target,value);p=np.clip(p,1e-15,1-1e-15);return np.log(p/(1-p))
    result=record(name+'_int8',local_logits(x),y,ids,g,selected_threshold=True,model_bytes=len(contents),daily_summary_bytes_proposed=24,board_verified=False)
    record(name+'_int8_half',local_logits(xh),y,ids,g,threshold=result['threshold'],model_bytes=len(contents))
 for r in rows:
  path=output/'predictions'/f"{r['id']}.npz";d=np.load(path);r['missed_rpw_ids']=[str(i) for i,l,p in zip(d['ids'],d['labels'],d['probabilities']) if l==1 and p<r['threshold']]
 save(output/'comparison.json',rows)
 columns=['id','accuracy','recall','f1','false_positives','false_negatives','threshold','model_bytes','feature_bytes_per_window']
 with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 lines=['# Full-recording and gain-normalization experiment','','Validation only, same original 361 recordings. Original 131 runs and prior review unchanged. Test set remains held out.','', 'Training uses recording-level max-logit multiple-instance loss, with all crops of a file kept together. Gain normalization uses filtered waveform RMS 0.01 before MFCC extraction. Four controlled warm-start fine-tunes use matching original architecture weights, seed 42, fresh Adam 1e-4, max 40 epochs, patience 10.','', '| Model | Recall | F1 | False alarms | RPW misses |','|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['id']} | {r['recall']:.2%} | {r['f1']:.2%} | {r['false_positives']} | {r['false_negatives']} |")
 lines+=['','All thresholds are validation-selected unless reusing the matching clean threshold for the half-volume check. Multiple model selection on validation does not establish final-test performance.','', 'PCA16 now sends 16 bytes per window, not automatically 16 bytes per recording. Multiply by the saved number of windows; node-side aggregation applies only to local classification. The 24-byte daily local summary is proposed. Windowing/gain frontend RAM, latency and energy require ESP32 measurement.','', 'Per-recording inference max-pools only actual windows. No noise removal or RPW event annotation is claimed. RMS normalization removes scalar gain, not background noise.']
 (output/'conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('COMPARISON',json.dumps([{k:r[k] for k in ['id','recall','f1','false_positives','false_negatives']} for r in rows]),flush=True)


def run(previous,output):
 """Execute isolated CPU preparation, four sequential GPU fits, and CPU export/evaluation."""
 settings=read(previous/'settings.json');output.mkdir(parents=True,exist_ok=True)
 save(output/'config.json',dict(previous=str(previous),output=str(output),stride_seconds=5,window_seconds=10,aggregation='max logit',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_set_evaluated=False))
 env=os.environ.copy();env.update(TF_USE_LEGACY_KERAS='1',MPLBACKEND='Agg',TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1')
 stages=[('prepare',None,None)]+[('train',r,v) for r in ['server','local'] for v in ['raw','gain']]+[('evaluate',None,None)]
 try:
  for stage,route,variant in stages:
   save(output/'progress.json',dict(state='running',stage=stage,route=route,variant=variant))
   child=env.copy()
   if stage!='train':child['CUDA_VISIBLE_DEVICES']='-1'
   command=[settings['torch_python' if stage=='train' else 'tf_python'],str(Path(__file__).resolve()),'--previous',str(previous),'--output',str(output),'--stage',stage]
   if route:command+=['--route',route,'--variant',variant]
   subprocess.run(command,check=True,env=child,cwd=settings['project'])
  save(output/'progress.json',dict(state='completed',runs=4,test_set_evaluated=False))
 except BaseException as error:
  save(output/'progress.json',dict(state='failed',stage=stage,route=route,variant=variant,error=str(error)));raise


def main():
 """Expose a replayable stage CLI so notebook disconnects cannot erase completed work."""
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--previous',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=['run','prepare','train','evaluate'],default='run');parser.add_argument('--route',choices=['server','local']);parser.add_argument('--variant',choices=['raw','gain']);args=parser.parse_args()
 settings=read(args.previous/'settings.json');sys.path[:0]=[settings['project'],settings['tools_parent']]
 if args.stage=='run':run(args.previous,args.output)
 elif args.stage=='prepare':prepare(args.previous,args.output)
 elif args.stage=='train':train(args.output,args.route,args.variant)
 else:evaluate(args.previous,args.output)


if __name__=='__main__':main()
