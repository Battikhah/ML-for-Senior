"""Generate review tables, deployment measurements and five-slide evidence from saved runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.review_campaign import read_json,save_json,review_metrics,cohort_digest,worker



def recall_operating_point(labels,probabilities,target):
    """Select the highest validation threshold meeting a declared RPW recall target."""
    labels=np.asarray(labels);probabilities=np.asarray(probabilities)
    if not 0<target<=1:raise ValueError('Recall target must be in (0,1]')
    positives=np.sort(probabilities[labels==1])[::-1]
    if not len(positives):raise ValueError('RPW observations are required')
    required=int(np.ceil(target*len(positives)))
    return review_metrics(labels,probabilities,float(positives[required-1]))

def append_codec(rows, output, name, model, train, val, labels, ids, threshold, model_bytes, representation):
    """Measure real uint8 and PCA round trips with training-fitted profiles and frozen thresholds."""
    from sklearn.decomposition import PCA
    from scipy.special import expit
    def evaluate(codec,recovered,size,profile):
        """Evaluate only frozen validation observations after decoding actual integer codes."""
        logits=np.concatenate([model(recovered[i:i+32,...,None],training=False).numpy().ravel() for i in range(0,len(recovered),32)])
        p=expit(logits);metrics=review_metrics(labels,p,threshold)
        identifier=name+'_'+codec
        np.savez_compressed(output/'plots'/f'{identifier}_predictions.npz',labels=labels,probabilities=p,ids=ids)
        np.savez_compressed(output/'checkpoints'/f'{identifier}_codec.npz',**profile)
        row=dict(id=identifier,model=name,route='server',representation=representation,codec=codec,
                 model_bytes=model_bytes,feature_bytes_per_measurement=size,daily_report_bytes=None,
                 scenario_measurements_per_day=12,scenario_feature_bytes_per_day=size*12,
                 validation_ids_hash=cohort_digest(ids),board_verified=False,**metrics)
        rows.append(row)
    lo=train.min(axis=(0,1));hi=train.max(axis=(0,1));scale=np.maximum((hi-lo)/255,1e-8)
    codes=np.clip(np.rint((val-lo)/scale),0,255).astype('uint8')
    recovered=np.frombuffer(codes.tobytes(),dtype='uint8').reshape(val.shape).astype('float32')*scale+lo
    evaluate('uint8',recovered,int(np.prod(val.shape[1:])),dict(lo=lo,scale=scale))
    for size in [16,32]:
        pca=PCA(n_components=size,svd_solver='full').fit(train.reshape(len(train),-1))
        projected=pca.transform(train.reshape(len(train),-1));low=projected.min(axis=0);high=projected.max(axis=0)
        scale=np.maximum((high-low)/255,1e-8)
        code=np.clip(np.rint((pca.transform(val.reshape(len(val),-1))-low)/scale),0,255).astype('uint8')
        decoded=np.frombuffer(code.tobytes(),dtype='uint8').reshape(code.shape).astype('float32')*scale+low
        restored=pca.inverse_transform(decoded).reshape(val.shape).astype('float32')
        evaluate(f'pca{size}',restored,size,dict(mean=pca.mean_,components=pca.components_,lo=low,scale=scale))


def deployment(settings,results):
    """Compare existing and newly selected local INT8/server models on identical evaluation IDs."""
    import tensorflow as tf
    import pandas as pd
    from treevibes.export import tflite_predict
    out=Path(settings['output']);selection=read_json(out/'audit/selection.json');rows=[]
    valid_hash=selection['cohort']['val'];cache=out/'cache/mfcc13';contract=read_json(cache/'contract.json')
    train=((np.load(cache/'X_train.npy')-np.array(contract['mean']))/np.array(contract['scale'])).astype('float32')
    val=((np.load(cache/'X_val.npy')-np.array(contract['mean']))/np.array(contract['scale'])).astype('float32')
    labels=np.load(cache/'y_val.npy');ids=np.load(cache/'ids_val.npy')
    previous=selection['server'];source=Path(previous['run_dir'])/'best.keras'
    model=tf.keras.models.load_model(source,compile=False)
    append_codec(rows,out,'previous_'+previous['id'],model,train,val,labels,ids,previous['threshold'],source.stat().st_size,'mfcc13')
    tf.keras.backend.clear_session()
    strongest=sorted([r for r in results if r['kind'] in ['feature_screen','feature_full'] and r['representation']!='mfcc13'],key=lambda r:(r['f1'],r['recall']),reverse=True)
    if strongest:
        r=strongest[0];cache=Path(r['cache']);c=read_json(cache/'contract.json')
        train=((np.load(cache/'X_train.npy')-np.array(c['mean']))/np.array(c['scale'])).astype('float32')
        val=((np.load(cache/'X_val.npy')-np.array(c['mean']))/np.array(c['scale'])).astype('float32')
        source=Path(r['directory'])/'best.keras';model=tf.keras.models.load_model(source,compile=False)
        append_codec(rows,out,r['id'],model,train,val,labels,ids,r['threshold'],source.stat().st_size,r['representation'])
        tf.keras.backend.clear_session()
    # Existing INT8 exports are measured again from their actual bytes, including QAT.
    candidates=[]
    for meta in read_json(out/'audit/original_exports.json'):
        if meta['method']=='float':continue
        parent=Path(meta['parent_run']);cfg=read_json(parent/'config.json')
        if cfg['target']!='local' or cfg['features']!=contract['features']:continue
        location=Path(meta['directory']);model_path=location/'model.tflite'
        if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest()!=meta['model_sha256']:continue
        raw=np.load(out/'cache/mfcc13/X_val.npy');input_c=read_json(parent/'input_contract.json')
        from treevibes.train import normalized
        p=tflite_predict(model_path,normalized(raw,input_c));m=review_metrics(labels,p,meta['threshold'])
        candidates.append((cfg['id']+'_'+meta['method'],model_path,m,p))
    extended=out/'checkpoints/extend_local'
    if (extended/'result.json').exists():
        destination=extended/'int8'
        if not (destination/'export.json').exists():
            worker(settings,dict(action='int8',id='extend_local_int8',directory=str(destination),model=str(extended/'best.keras'),cache=str(out/'cache/mfcc13')),framework='tf')
        meta=read_json(destination/'export.json')
        with np.load(destination/'validation_predictions.npz') as p:probs=p['probabilities']
        metrics={k:meta[k] for k in ['accuracy','precision','recall','f1','pr_auc','roc_auc','confusion','threshold','count','false_negatives','false_positives','fpr']}
        candidates.append(('extend_local_int8',destination/'model.tflite',metrics,probs))
    for name,source,m,p in candidates:
        np.savez_compressed(out/'plots'/f'{name}_predictions.npz',labels=labels,probabilities=p,ids=ids)
        rows.append(dict(id=name,model=name,route='ESP32 candidate',representation='mfcc13',codec='local_int8',
                         model_bytes=source.stat().st_size,feature_bytes_per_measurement=0,daily_report_bytes=24,
                         scenario_measurements_per_day=12,scenario_feature_bytes_per_day=24,
                         validation_ids_hash=valid_hash,board_verified=False,**m))
    for r in results:
        if r['kind']=='compressed_server':
            import shutil
            shutil.copy2(Path(r['directory'])/'validation_predictions.npz',out/'plots'/f"{r['id']}_predictions.npz")
            rows.append(dict(id=r['id'],model=r['id'],route='server',representation=r['representation'],codec='pca16_direct',
                             model_bytes=r['model_bytes'],feature_bytes_per_measurement=16,daily_report_bytes=None,
                             scenario_measurements_per_day=12,scenario_feature_bytes_per_day=192,
                             validation_ids_hash=valid_hash,board_verified=False,
                             **{k:r[k] for k in ['accuracy','recall','f1','precision','fpr','false_negatives','false_positives','confusion','threshold','count']}))
    for r in results:
        if r['kind']=='unrestricted':
            # The cached single-channel resized image is a measured float16 transport option.
            # A smaller uint8 image is not assigned this score without a separate round-trip evaluation.
            rows.append(dict(id=r['id'],model=r['id'],route='unrestricted server',representation=r['representation'],codec='float16_spectrogram',
                             model_bytes=r['model_bytes'],feature_bytes_per_measurement=299*299*2,
                             daily_report_bytes=None,scenario_measurements_per_day=12,scenario_feature_bytes_per_day=299*299*2*12,
                             validation_ids_hash=valid_hash,board_verified=False,
                             **{k:r[k] for k in ['accuracy','recall','f1','precision','fpr','false_negatives','false_positives','confusion','threshold','count']}))
    pd.DataFrame(rows).to_csv(out/'deployment_comparison.csv',index=False);save_json(out/'deployment_comparison.json',rows)
    return rows


def render_plots(out, old, results,deploy):
    """Write scientific charts and exact plotted data; retain actual healthy/RPW counts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    plots=out/'plots'
    examples=plots/'same_recording_features.npz'
    if examples.exists():
        with np.load(examples) as source:
            fig,axes=plt.subplots(1,3,figsize=(11,2.6))
            for ax,key,title in zip(axes,['mfcc','mel','pcen'],['MFCC','Log-Mel64','PCEN64']):
                ax.imshow(source[key].T,aspect='auto',origin='lower',cmap='magma');ax.set_title(title);ax.set_xlabel('Time frame')
            fig.tight_layout();fig.savefig(plots/'feature_heatmaps.png');plt.close(fig)
    for route in ['server','local']:
        new=next((r for r in results if r['id']==f'extend_{route}'),None)
        if new:
            previous=next(r for r in old if r['run_dir']==new['origin'])
            history=read_json(out/'audit/models'/previous['id']/'training_result.json').get('history',[])
            fig,ax=plt.subplots(figsize=(6.2,3.2))
            for key,label,color in [('train_loss','Train','#64748b'),('val_loss','Validation','#007f73')]:
                ax.plot([h['epoch'] for h in new['history']],[h[key] for h in new['history']],label=label,color=color)
            ax.axvline(max(h['epoch'] for h in history),ls='--',color='#d97706',label='Previous stop')
            ax.set(xlabel='Epoch',ylabel='Binary cross-entropy',title=route.capitalize()+' continuation');ax.legend()
            fig.tight_layout();fig.savefig(plots/f'extension_{route}.png');plt.close(fig)
    local=max([r for r in deploy if r['route']=='ESP32 candidate'],key=lambda r:(r['f1'],r['recall']))
    server=max([r for r in deploy if r['route']!='ESP32 candidate'],key=lambda r:(r['f1'],r['recall']))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,r,title in zip(axes,[local,server],['Local INT8','Best server']):
        counts=np.array(r['confusion']);norm=counts/counts.sum(axis=1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{counts[i,j]}\n{norm[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.6 else '#102030',fontsize=14)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['healthy','RPW'],yticklabels=['healthy','RPW'],xlabel='Predicted',ylabel='Actual',title=title)
    fig.tight_layout();fig.savefig(plots/'finalist_confusion.png');plt.close(fig)
    from sklearn.metrics import precision_recall_curve
    fig,ax=plt.subplots(figsize=(6,3.5))
    for r in [local,server]:
        path=plots/f"{r['id']}_predictions.npz"
        if not path.exists():path=Path(next(x for x in results if x['id']==r['id'])['directory'])/'validation_predictions.npz'
        with np.load(path) as p:precision,recall,thresholds=precision_recall_curve(p['labels'],p['probabilities'])
        ax.plot(recall,precision,label=r['id'])
    ax.set(xlabel='RPW recall',ylabel='RPW precision',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'finalist_precision_recall.png');plt.close(fig)
    return local,server


def summarize(settings,measure_deployment=True):
    """Assemble tables and slide data exclusively from completed, cohort-matched evidence."""
    import pandas as pd
    out=Path(settings['output']);selection=read_json(out/'audit/selection.json');old=read_json(out/'audit/previous_results.json')
    results=[]
    for path in (out/'checkpoints').glob('*/result.json'):
        r=read_json(path);spec=read_json(path.parent/'config.json')
        if r['validation_ids_hash']!=selection['cohort']['val']:raise ValueError('Finalist evaluation cohorts differ')
        r.update(directory=str(path.parent),cache=spec['cache'],model_width=spec.get('model_config',{}).get('width'),split='validation')
        with np.load(path.parent/'validation_predictions.npz',allow_pickle=False) as pred:
            expected_ids=np.load(out/'cache/mfcc13/ids_val.npy')
            expected_labels=np.load(out/'cache/mfcc13/y_val.npy')
            np.testing.assert_array_equal(pred['ids'],expected_ids)
            np.testing.assert_array_equal(pred['labels'],expected_labels)
            measured=review_metrics(pred['labels'],pred['probabilities'],r['threshold'])
            for key in ['accuracy','recall','f1','fpr']:
                if abs(measured[key]-r[key])>1e-10:raise ValueError('Recorded metric differs from saved predictions')
        results.append(r)
    flat=[{k:v for k,v in r.items() if k not in ['history','continuation','confusion']} for r in results]
    pd.DataFrame(flat).to_csv(out/'runs.csv',index=False)
    features=[r for r in results if r['kind'] in ['feature_screen','feature_full']]
    columns=['id','family','model_width','kind','representation','feature_shape','split','seed','accuracy','recall','f1','fpr','false_negatives','false_positives','best_epoch','validation_ids_hash']
    pd.DataFrame(features).reindex(columns=columns).to_csv(out/'feature_comparison.csv',index=False)
    deploy=deployment(settings,results) if measure_deployment else read_json(out/'deployment_comparison.json')
    local,server=render_plots(out,old,results,deploy)
    edge_high_recall=max([r for r in deploy if r['route']=='ESP32 candidate'],key=lambda r:(r['recall'],r['f1'],-r['model_bytes']))
    operating=[]
    for finalist in [local,server]:
        prediction=out/'plots'/f"{finalist['id']}_predictions.npz"
        if not prediction.exists():prediction=Path(next(r for r in results if r['id']==finalist['id'])['directory'])/'validation_predictions.npz'
        with np.load(prediction) as p:
            for target in [.95,.98,1.]:
                operating.append(dict(id=finalist['id'],target_recall=target,**recall_operating_point(p['labels'],p['probabilities'],target)))
    pd.DataFrame(operating).to_csv(out/'operating_points.csv',index=False)
    extensions=[]
    for route in ['server','local']:
        new=next((r for r in results if r['id']==f'extend_{route}'),None)
        if new:extensions.append(dict(route=route,old=selection[route],new=new))
    screen=[r for r in features if r['kind']=='feature_screen'];full=[r for r in features if r['kind']=='feature_full']
    feature_best=max(features,key=lambda r:(r['f1'],r['recall'])) if features else None
    unrestricted=next((r for r in results if r['kind']=='unrestricted'),None)
    compressed=max([r for r in deploy if r['codec'].startswith('pca16')],key=lambda r:(r['f1'],r['recall']))
    full_mfcc=next(r for r in deploy if r['representation']=='mfcc13' and r['codec']=='uint8')
    delta=max((e['new']['f1']-e['old']['f1'] for e in extensions),default=0)
    lines=['# Final model decisions','',f"Original campaign: {len(old)} canonical recipes; {len(results)} completed review runs.",
        'All scores below are validation results on the same 361 accepted recordings (109 RPW, 252 healthy).',
        'The final test set was not evaluated. Thresholds selected on validation; file/duplicate grouping, not tree/session isolation.','',
        f"- Extra training: maximum F1 change {delta*100:+.2f} percentage points; select the minimum-validation-loss checkpoint.",
        f"- Best balanced edge candidate: {local['id']}, {local['model_bytes']} bytes, recall {local['recall']:.2%}, F1 {local['f1']:.2%}.",
        f"- Best measured deployment-server route: {server['id']}, recall {server['recall']:.2%}, F1 {server['f1']:.2%}.",
        f"- Compact server payload: {compressed['id']}, 16 bytes/measurement, recall {compressed['recall']:.2%}, F1 {compressed['f1']:.2%}.",
        '- Every node sends at least one daily report. Local 24-byte summary is a proposed prediction/health report; remote features retain each reading.',
        '- ESP32 end-to-end RAM, latency, firmware size and power remain unmeasured. Radio framing, security and retries are extra.',
        '- Image-based unrestricted inference is an accuracy experiment; its large input is not a LoRa deployment recommendation.','',
        '## Uncompleted/omitted menu items','',
        '- Separate lower-initial-learning-rate ablation, raw-waveform CNN, extra seeds, cross-feature knowledge distillation and acoustic-noise tests were not automatically launched; one volume perturbation was completed separately.',
        '- Previous training histories contain losses; newly recorded epochs also contain recall/F1 at threshold 0.5. Missing old epoch metrics are not invented.']
    if feature_best:lines.insert(8,f"- Best feature trial by F1: {feature_best['id']}, F1 {feature_best['f1']:.2%}, recall {feature_best['recall']:.2%}.")
    if unrestricted:lines.insert(9,f"- Unrestricted InceptionResNet-V2: accuracy {unrestricted['accuracy']:.2%}, recall {unrestricted['recall']:.2%}, F1 {unrestricted['f1']:.2%}.")
    (out/'final_conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    previous_accuracy=max([r for r in old if r['status']=='completed' and r['same_evaluation_cohort']],key=lambda r:(r['accuracy'],r['recall']))
    feature_accuracy=max(features,key=lambda r:(r['accuracy'],r['recall'])) if features else None
    ceiling=max([previous_accuracy]+results,key=lambda r:(r['accuracy'],r['recall']))
    lines.insert(7,f"- Highest measured accuracy on this cohort: {ceiling['id']}, {ceiling['accuracy']:.2%}, RPW recall {ceiling['recall']:.2%}.")
    (out/'final_conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    volume=read_json(out/'volume_check.json') if (out/'volume_check.json').exists() else None
    if volume:
        for row in volume['results']:
            if row['perturbation']=='audio_gain_0_5':
                lines.insert(8,f"- Half-volume waveform check, fixed thresholds: {row['model']} produced {row['false_positives']} healthy false alarms ({row['fpr']:.2%}); RPW recall {row['recall']:.2%}.")
        lines.insert(8,'- Gain sensitivity needs mitigation and re-evaluation before field deployment; the volume check does not establish field performance.')
        (out/'final_conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines.insert(12,f"- High-recall edge alternative: {edge_high_recall['id']}, {edge_high_recall['model_bytes']} bytes, RPW recall {edge_high_recall['recall']:.2%}, F1 {edge_high_recall['f1']:.2%}, {edge_high_recall['false_positives']} healthy false alarms.")
    (out/'final_conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    payload=dict(edge_high_recall=edge_high_recall,volume_check=volume,evaluation='validation',count=selection['validation_count'],rpw=109,healthy=252,cohort=selection['cohort']['val'],
                 extensions=extensions,features=screen,feature_full=full,feature_best=feature_best,
                 previous_server=selection['server'],previous_accuracy=previous_accuracy,feature_accuracy=feature_accuracy,accuracy_ceiling=ceiling,unrestricted=unrestricted,local=local,server=server,
                 operating_points=operating,deployment=[local,full_mfcc,compressed]+([next(r for r in deploy if r['route']=='unrestricted server')] if unrestricted else []))
    save_json(out/'slide_data.json',payload)
    save_json(out/'report_status.json',dict(completed_review_runs=len(results),test_set_evaluated=False,slide_data_ready=bool(extensions and screen and unrestricted)))
    print('REPORT READY',out,flush=True)
    return payload


def main():
    """Regenerate recorded results without training or changing checkpoints."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--reuse-deployment',action='store_true');args=parser.parse_args()
    summarize(read_json(args.settings),not args.reuse_deployment)


if __name__=='__main__':main()
