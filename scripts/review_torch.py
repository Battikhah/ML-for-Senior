"""CUDA-only review training with durable epochs, fixed cohorts and measured resource use."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from scripts.review_campaign import read_json,save_json,review_metrics,cohort_digest


def load_arrays(cache, image=False):
    """Memory-map image inputs and never open final-test labels or features."""
    cache=Path(cache);contract=read_json(cache/'contract.json')
    data={name:np.load(cache/f'{name}.npy',mmap_mode='r',allow_pickle=False)
          for name in ['X_train','y_train','ids_train','X_val','y_val','ids_val']}
    for part in ['train','val']:
        key='train_ids_hash' if part=='train' else 'validation_ids_hash'
        if cohort_digest(data[f'ids_{part}'])!=contract[key]:raise ValueError('Recording cohort changed')
        if not image:
            data[f'X_{part}']=((data[f'X_{part}']-np.array(contract['mean'],dtype='float32'))/
                               np.array(contract['scale'],dtype='float32')).astype('float32')
    return data,contract


def input_batch(x, indices, image):
    """Replicate the spectrogram channel and apply the pretrained model's [-1,1] input scale."""
    value=torch.as_tensor(np.array(x[indices],copy=True),dtype=torch.float32,device='cuda')
    return ((value[:,None]+80)/40-1).repeat(1,3,1,1) if image else value


def predictions(model,x,batch,image):
    """Bound inference allocations independently of training accumulation or dataset size."""
    values=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),batch):
            xb=input_batch(x,slice(start,start+batch),image)
            with torch.autocast('cuda',enabled=image):z=model(xb).float()
            values.append(z.cpu().numpy().ravel())
    return np.concatenate(values)


def cpu_state(model):
    """Detach checkpoints from CUDA so best-state bookkeeping cannot double GPU allocation."""
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train(spec):
    """Continue original optimizer state or train controlled features/pretrained InceptionResNet-V2."""
    from scipy.special import expit
    from treevibes.evaluate import select_threshold
    from scripts.torch_adapter import GraphModel,save_keras_weights
    from scripts.review_campaign import worker
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no silent CPU fallback.')
    torch.set_num_threads(2);random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    directory=Path(spec['directory']);directory.mkdir(parents=True,exist_ok=True)
    save_json(directory/'config.json',spec)
    image=spec['family']=='inception_resnet_v2';data,contract=load_arrays(spec['cache'],image)
    if image:
        import timm
        model=timm.create_model(spec['backbone'],pretrained=spec['pretrained'],num_classes=1)
        # Verify the selected pretrained input normalization, rather than assuming ImageNet defaults.
        cfg=model.pretrained_cfg
        if tuple(cfg.get('mean',()))!=(.5,.5,.5) or tuple(cfg.get('std',()))!=(.5,.5,.5):
            raise ValueError('Unexpected pretrained input scale; update the explicit image transform.')
        save_json(directory/'pretrained_config.json',{k:v for k,v in cfg.items() if isinstance(v,(str,int,float,list,tuple,bool,type(None)))})
    else:model=GraphModel(directory/'graph')
    model=model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=spec.get('learning_rate',spec.get('model_config',{}).get('learning_rate',1e-3)),eps=1e-7)
    scaler=torch.amp.GradScaler('cuda',enabled=image)
    history=[];start_epoch=0;best_loss=1e30;best_weights=cpu_state(model);stale=0
    original_history=[];origin_note=None
    last=directory/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['spec']!=spec:raise ValueError('Resume specification changed; use a separate run ID.')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];start_epoch=saved['epoch'];best_loss=saved['best_loss'];stale=saved['stale']
        best_weights=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    elif spec.get('origin'):
        origin=Path(spec['origin']);state=read_json(origin/'epoch_state.json')
        if state.get('backend')!='torch_cuda_v1':raise ValueError('Continuation needs a CUDA optimizer checkpoint.')
        saved=torch.load(origin/state['checkpoint'],map_location='cpu',weights_only=False)
        best=torch.load(origin/state['best_checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        # Optimizer moments/rate persist; only the explicit LR ablation changes the rate.
        for group in optimizer.param_groups:group['lr']*=spec.get('learning_rate_factor',1.)
        best_weights=best['model'];best_loss=state['best_loss'];start_epoch=state['epoch'];history=state['history'].copy()
        original_history=history.copy();origin_note=dict(epoch=start_epoch,optimizer_restored=True,
            previous_stale=state['stale'],new_patience=spec['patience'],stale_reset_for_continuation=True)
    # Adam state tensors must follow parameters when checkpoints were read on CPU.
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value):state[key]=value.cuda()
    torch.save(best_weights,directory/'best.pt')
    classes=np.bincount(data['y_train'],minlength=2)
    weighted=bool(spec.get('model_config',{}).get('class_weight',False))
    class_weights=torch.tensor(len(data['y_train'])/(2*classes) if weighted else [1,1],device='cuda',dtype=torch.float32)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();head_epochs=spec.get('head_epochs',0)
    switched=False
    for epoch in range(start_epoch,spec['epochs']):
        if stale>=spec['patience']:break
        torch.manual_seed(spec['seed']+epoch);rng=np.random.default_rng(spec['seed']+epoch)
        if image:
            fine=epoch>=head_epochs
            for name,p in model.named_parameters():
                p.requires_grad=name.startswith('classif') or (fine and name.startswith(('repeat_2.','block8.','conv2d_7b.')))
            if epoch==head_epochs and not switched:
                # Fine-tuning has its own low learning rate, never an epoch-limit ablation.
                for group in optimizer.param_groups:group['lr']=1e-5
                stale=0;switched=True
        model.train()
        if image:
            # Frozen BatchNorm statistics must not drift during head training or upper-block tuning.
            for module in model.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        order=rng.permutation(len(data['y_train']));loss_total=0.;epoch_start=time.monotonic();rate=optimizer.param_groups[0]['lr']
        batch=spec['batch_size'];accum=spec.get('accumulation',1);optimizer.zero_grad(set_to_none=True)
        steps=list(range(0,len(order),batch))
        for step,start in enumerate(steps):
            indices=order[start:start+batch];xb=input_batch(data['X_train'],indices,image)
            yb=torch.as_tensor(np.array(data['y_train'][indices],copy=True),device='cuda',dtype=torch.float32).reshape(-1,1)
            c=spec.get('model_config',{})
            if c.get('noise'):xb+=torch.as_tensor(rng.normal(0,c['noise'],xb.shape),device='cuda',dtype=torch.float32)
            if c.get('time_mask'):
                for row in xb:
                    begin=rng.integers(0,row.shape[0]-c['time_mask']+1);row[begin:begin+c['time_mask']]=0
            # Weight each microbatch by observations in its accumulation window, including the last short window.
            window_start=(step//accum)*accum;window_end=min(window_start+accum,len(steps))
            window_count=sum(min(batch,len(order)-steps[j]) for j in range(window_start,window_end))
            with torch.autocast('cuda',enabled=image):
                logits=model(xb);loss=nn.functional.binary_cross_entropy_with_logits(logits,yb,reduction='none')
                loss=(loss*class_weights[yb.long()]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            scaler.scale(loss*len(indices)/window_count).backward();loss_total+=float(loss.detach())*len(indices)
            if step+1==window_end:
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        z=predictions(model,data['X_val'],batch if image else 64,image)
        val_loss=float(np.mean(np.logaddexp(0,z)-data['y_val']*z))
        if not np.isfinite(val_loss):raise ValueError('Nonfinite validation loss')
        p=expit(z);m=review_metrics(data['y_val'],p,.5)
        improved=val_loss<best_loss;stale=0 if improved else stale+1
        if improved:
            best_loss=val_loss;best_weights=cpu_state(model)
            temp=directory/'best.tmp.pt';torch.save(best_weights,temp);temp.replace(directory/'best.pt')
        row=dict(epoch=epoch+1,train_loss=loss_total/len(order),val_loss=val_loss,learning_rate=rate,
                 recall_at_05=m['recall'],f1_at_05=m['f1'],seconds=time.monotonic()-epoch_start,
                 stage='fine_tune' if image and epoch>=head_epochs else 'head' if image else 'supervised')
        history.append(row)
        if stale and stale%5==0:
            for group in optimizer.param_groups:group['lr']=max(group['lr']*.5,1e-6)
        payload=dict(spec=spec,model=cpu_state(model),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                     epoch=epoch+1,history=history,best_loss=best_loss,stale=stale)
        temp=directory/'last.tmp.pt';torch.save(payload,temp);temp.replace(last)
        save_json(directory/'history.json',history)
        print(f"REVIEW {spec['id']} CUDA {epoch+1}/{spec['epochs']} val_loss={val_loss:.5f} recall@.5={m['recall']:.3f} {row['seconds']:.1f}s",flush=True)
    model.load_state_dict(best_weights);z=predictions(model,data['X_val'],batch if image else 64,image);p=expit(z)
    threshold=select_threshold(data['y_val'],p);metrics=review_metrics(data['y_val'],p,threshold)
    np.savez_compressed(directory/'validation_predictions.npz',labels=data['y_val'],probabilities=p,ids=data['ids_val'])
    result=dict(id=spec['id'],kind=spec['kind'],family=spec['family'],route=spec['route'],representation=contract['representation'],
        feature_shape=contract['shape'],validation_ids_hash=contract['validation_ids_hash'],train_ids_hash=contract['train_ids_hash'],
        seed=spec['seed'],best_epoch=min(history,key=lambda h:h['val_loss'])['epoch'],epochs_completed=history[-1]['epoch'],
        history=history,elapsed_new_training_seconds=time.monotonic()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,model_bytes=(directory/'best.pt').stat().st_size,
        parameters=sum(p.numel() for p in model.parameters()),origin=spec.get('origin'),continuation=origin_note,
        test_set_evaluated=False,**metrics)
    if not image:
        save_keras_weights(model,directory/'best_weights.npz')
        fixture=np.asarray(data['X_val'][:4]);np.savez(directory/'bridge_parity.npz',features=fixture,logits=predictions(model,fixture,4,False))
        settings=read_json(Path(spec['output'])/'settings.json')
        worker(settings,dict(action='bridge_restore',id=spec['id'],directory=str(directory/'graph'),
                             weights=str(directory/'best_weights.npz'),model=str(directory/'best.keras')),framework='tf')
    save_json(directory/'result.json',result)
    print('RESULT',spec['id'],json.dumps(metrics),flush=True)


def main():
    """Accept only recorded JSON requests, keeping notebook kernels free of GPU allocations."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();train(read_json(args.request))


if __name__=='__main__':main()
