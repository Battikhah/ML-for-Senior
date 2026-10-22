"""CUDA training for the existing Keras graphs, preserving their input and export contracts."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn


class GraphModel(nn.Module):
    """Interpret only the small audited layer set used by the TreeVibes architectures."""

    def __init__(self,directory):
        """Rebuild the graph from numeric weights without importing TensorFlow on the GPU worker."""
        super().__init__()
        self.graph=json.loads((Path(directory)/'graph.json').read_text())
        self.layers=nn.ModuleDict()
        with np.load(Path(directory)/'weights.npz',allow_pickle=False) as arrays:
            for row in self.graph['layers']:
                kind=row['kind'];config=row['config'];name=row['name']
                if kind in {'InputLayer','Flatten','GlobalAveragePooling2D','Add'}:
                    continue
                weights=[arrays[f'{name}__{i}'] for i in range(row['weights'])]
                if kind=='Dense':
                    module=nn.Linear(weights[0].shape[0],weights[0].shape[1],bias=len(weights)>1)
                    kernel=weights[0].T
                elif kind in {'Conv2D','DepthwiseConv2D'}:
                    shape=weights[0].shape
                    depth=kind=='DepthwiseConv2D';channels=shape[2]
                    module=nn.Conv2d(channels,channels*shape[3] if depth else shape[3],shape[:2],
                        stride=tuple(config.get('strides',[1,1])),
                        padding=tuple(s//2 for s in shape[:2]) if config['padding']=='same' else 0,
                        groups=channels if depth else 1,bias=len(weights)>1)
                    kernel=weights[0].transpose(2,3,0,1).reshape(channels*shape[3],1,*shape[:2]) if depth else weights[0].transpose(3,2,0,1)
                elif kind=='Dropout':
                    module=nn.Dropout(config['rate'])
                elif kind=='MaxPooling2D':
                    if config['padding']!='valid':
                        raise ValueError('Unsupported pooling padding')
                    module=nn.MaxPool2d(tuple(config['pool_size']),stride=tuple(config['strides']))
                elif kind=='ReLU':
                    module=nn.ReLU()
                else:
                    raise ValueError(f'Unsupported converted layer: {kind}')
                if weights:
                    with torch.no_grad():
                        module.weight.copy_(torch.tensor(kernel.copy()))
                        if len(weights)>1:
                            module.bias.copy_(torch.tensor(weights[1]))
                self.layers[name]=module

    def forward(self,x):
        """Keep convolutions NCHW internally and reproduce Keras NHWC flatten ordering."""
        values={}
        for row in self.graph['layers']:
            kind=row['kind'];name=row['name']
            if kind=='InputLayer':
                values[name]=x[:,None] if x.ndim==3 else x
                continue
            inputs=[values[n] for n in row['parents']]
            value=inputs[0]
            if kind=='Flatten':
                value=value.permute(0,2,3,1).reshape(len(value),-1)
            elif kind=='GlobalAveragePooling2D':
                value=value.mean(dim=(2,3))
            elif kind=='Add':
                value=sum(inputs)
            else:
                value=self.layers[name](value)
                activation=row['config'].get('activation','linear')
                if activation=='relu':
                    value=torch.relu(value)
                elif activation!='linear':
                    raise ValueError(f'Unsupported activation: {activation}')
            values[name]=value
        return values[self.graph['output']]


def save_keras_weights(model,path):
    """Invert each kernel transpose exactly for Keras/TFLite serialization."""
    arrays={}
    for row in model.graph['layers']:
        if not row['weights']:
            continue
        module=model.layers[row['name']];kernel=module.weight.detach().cpu().numpy()
        if row['kind']=='Dense':
            kernel=kernel.T
        elif row['kind']=='Conv2D':
            kernel=kernel.transpose(2,3,1,0)
        else:
            channels=module.in_channels;multiplier=module.out_channels//channels
            kernel=kernel.reshape(channels,multiplier,*kernel.shape[2:]).transpose(2,3,0,1)
        arrays[f'{row["name"]}__0']=kernel
        if row['weights']>1:
            arrays[f'{row["name"]}__1']=module.bias.detach().cpu().numpy()
    np.savez(path,**arrays)


def bridge(action,config,directory,model=None,weights=None):
    """Run CPU-only Keras conversion in its own interpreter and retain the complete log."""
    from scripts.windows_support import run_logged
    from treevibes.runtime import worker_environment
    python=os.environ['TREEVIBES_TF_PYTHON']
    command=[python,'-m','scripts.torch_bridge',action,'--config',str(config),'--directory',str(directory)]
    if model is not None:
        command+=['--model',str(model)]
    if weights is not None:
        command+=['--weights',str(weights)]
    environment=worker_environment();environment['CUDA_VISIBLE_DEVICES']='-1'
    run_logged(command,Path(directory)/f'bridge_{action}.log',env=environment)


def logits(model,x,device):
    """Bound validation/teacher memory independently of corpus size."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(x[i:i+64],device=device)).cpu().numpy().ravel()
                               for i in range(0,len(x),64)])


def verify_bridge(model,bundle,device):
    """Fail before fitting if actual converted logits differ from the Keras CPU probe."""
    with np.load(Path(bundle)/'probe.npz',allow_pickle=False) as probe:
        np.testing.assert_allclose(logits(model,probe['features'],device),probe['logits'],rtol=2e-4,atol=5e-5)


def train_cuda(c,x,y,xv,yv,directory,teacher_dir,contract):
    """Warm-start trusted Keras weights, then checkpoint PyTorch optimizer state each epoch."""
    from treevibes.data import fingerprint,write_json
    from treevibes.tracking import file_hash
    from scipy.special import expit
    directory=Path(directory);state_path=directory/'epoch_state.json'
    if c.get('method')=='kd' and teacher_dir is None:
        raise ValueError('Distillation requires a verified teacher')
    config=directory/'config.json';bundle=directory/'torch_graph'
    state=json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state.get('config_hash')!=fingerprint(c):
        raise ValueError('Converted checkpoint configuration differs')
    source=directory/state['checkpoint'] if state and state.get('backend')!='torch_cuda_v1' else None
    if state and state.get('backend')=='torch_cuda_v1':
        saved=torch.load(directory/state['checkpoint'],map_location='cpu',weights_only=False)
        bundle.mkdir(parents=True,exist_ok=True)
        source=bundle/'saved_template.keras';source.write_bytes(saved['template_keras'])
    # A fresh graph is cheap; copied checkpoint files carry the actual committed weights.
    bridge('dump',config,bundle,model=source)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda' and os.environ.get('TREEVIBES_ALLOW_CPU_TEST')!='1':
        raise RuntimeError('GPU training requested but CUDA is unavailable; no CPU fallback.')
    model=GraphModel(bundle).to(device)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    verify_bridge(model,bundle,device)
    template=(bundle/'template.keras').read_bytes()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-7)
    checkpoints=directory/'checkpoints';checkpoints.mkdir(exist_ok=True)
    if state and state.get('backend')=='torch_cuda_v1':
        saved=torch.load(directory/state['checkpoint'],map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=GraphModel(bundle).to(device)
        best.load_state_dict(torch.load(directory/state['best_checkpoint'],map_location=device,weights_only=False)['model'])
    else:
        origin_epoch=state['epoch'] if state else 0
        best=GraphModel(bundle).to(device)
        if state:
            best_bundle=directory/'keras_best_graph'
            bridge('dump',config,best_bundle,model=directory/state['best_checkpoint'])
            best.load_state_dict(GraphModel(best_bundle).state_dict())
            history=state.get('history',[])
            rate=history[-1].get('learning_rate',c['learning_rate']) if history else c['learning_rate']
            if state['stale'] and state['stale']%5==0:
                rate=max(rate*.5,1e-6)
            optimizer.param_groups[0]['lr']=rate
        state=dict(epoch=origin_epoch,history=state.get('history',[]) if state else [],
                   best_loss=state['best_loss'] if state else 1e30,stale=state['stale'] if state else 0,
                   config_hash=fingerprint(c),backend='torch_cuda_v1',migrated_keras_epochs=origin_epoch,
                   checkpoint='checkpoints/gpu_initial.pt',
                   best_checkpoint='checkpoints/gpu_best_initial.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),template_keras=template),directory/state['checkpoint'])
        torch.save(dict(model=best.state_dict()),directory/state['best_checkpoint'])
        write_json(state_path,state)
    write_json(directory/'backend.json',dict(backend='pytorch_cuda',torch=torch.__version__,device=device,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        migrated_keras_epochs=state.get('migrated_keras_epochs',0),keras_optimizer_restored=False,
        note='Keras weights and epoch history retained; Adam optimizer reset once at framework migration.'))
    teacher=None
    if c.get('method')=='kd':
        teacher_dir=Path(teacher_dir)
        if json.loads((teacher_dir/'input_contract.json').read_text())!=contract:
            raise ValueError('Teacher input contract differs')
        teacher_bundle=directory/'teacher_graph'
        bridge('dump',config,teacher_bundle,model=teacher_dir/'best.keras')
        teacher=GraphModel(teacher_bundle).to(device).eval()
        verify_bridge(teacher,teacher_bundle,device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    counts=np.bincount(y,minlength=2)
    class_weights=len(y)/(2*counts) if c['class_weight'] else np.ones(2)
    for epoch in range(state['epoch'],c['epochs']):
        if state['stale']>=c['patience']:
            break
        torch.manual_seed(c['seed']+epoch);model.train()
        rng=np.random.default_rng(c['seed']+epoch);indices=rng.permutation(len(y));losses=[]
        rate=optimizer.param_groups[0]['lr']
        for start in range(0,len(y),c['batch_size']):
            index=indices[start:start+c['batch_size']];batch=x[index].copy()
            if c['noise']:
                batch+=rng.normal(0,c['noise'],batch.shape).astype('float32')
            if c['time_mask']:
                for sample in batch:
                    begin=rng.integers(0,len(sample)-c['time_mask']+1);sample[begin:begin+c['time_mask']]=0
            xb=torch.as_tensor(batch,device=device);yb=torch.as_tensor(y[index,None],dtype=torch.float32,device=device)
            predicted=model(xb)
            loss=nn.functional.binary_cross_entropy_with_logits(predicted,yb,reduction='none')
            if teacher is not None:
                with torch.no_grad():
                    targets=torch.sigmoid(teacher(xb)/c['temperature'])
                soft=nn.functional.binary_cross_entropy_with_logits(predicted/c['temperature'],targets,reduction='none')
                loss=(1-c['alpha'])*loss+c['alpha']*c['temperature']**2*soft
            weights=torch.as_tensor(class_weights[y[index],None],dtype=torch.float32,device=device)
            loss=(loss*weights).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            losses.append((float(loss.detach()),len(index)))
        predicted=logits(model,xv,device)
        loss=float(np.mean(np.logaddexp(0,predicted)-yv*predicted))
        if not np.isfinite(loss):
            raise ValueError('Nonfinite CUDA validation loss')
        improved=loss<state['best_loss'];state.update(epoch=epoch+1,stale=0 if improved else state['stale']+1)
        state['history'].append(dict(epoch=epoch+1,train_loss=sum(a*b for a,b in losses)/len(y),
                                     val_loss=loss,learning_rate=rate,backend='pytorch_cuda'))
        if improved:
            state['best_loss']=loss;best.load_state_dict(model.state_dict())
        if state['stale'] and state['stale']%5==0:
            optimizer.param_groups[0]['lr']=max(rate*.5,1e-6)
        checkpoint=f'checkpoints/gpu_epoch_{epoch+1:04}.pt'
        temporary=directory/(checkpoint+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),template_keras=template),temporary)
        temporary.replace(directory/checkpoint)
        state['checkpoint']=checkpoint
        if improved:
            state['best_checkpoint']=checkpoint
        write_json(state_path,state)
        for old in checkpoints.glob('gpu_*.pt'):
            if str(old.relative_to(directory)).replace('\\','/') not in {state['checkpoint'],state['best_checkpoint']}:
                old.unlink()
        if epoch==0 or (epoch+1)%5==0 or epoch+1==c['epochs']:
            print(f'TV_PROGRESS {c["id"]} CUDA epoch {epoch+1}/{c["epochs"]}: validation loss={loss:.5f}',flush=True)
    probabilities=expit(logits(best,xv,device))
    weights=directory/'torch_best_weights.npz';save_keras_weights(best,weights)
    np.savez(directory/'bridge_parity.npz',features=xv[:8],logits=logits(best,xv[:8],device))
    bridge('restore',config,bundle,model=directory/'best.keras',weights=weights)
    return probabilities,dict(epochs_completed=state['epoch'],history=state['history'],
                             parameters=sum(p.numel() for p in model.parameters()),backend='pytorch_cuda')
