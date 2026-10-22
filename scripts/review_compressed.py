"""One direct classifier on actual 16-byte PCA codes, rather than reconstructing MFCC first."""
import argparse
from pathlib import Path
import time
import numpy as np
from scripts.review_campaign import read_json,save_json,cohort_digest,review_metrics


def run(settings):
    """Fit a fixed histogram booster on train-only PCA16 uint8 codes and measure held-out validation."""
    import joblib
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from treevibes.evaluate import select_threshold
    out=Path(settings['output']);directory=out/'checkpoints/server_pca16_histogram';directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return
    selection=read_json(out/'audit/selection.json');cache=out/'cache/mfcc13';contract=read_json(cache/'contract.json')
    profile=out/'checkpoints'/('previous_'+selection['server']['id']+'_pca16_codec.npz')
    with np.load(profile) as p:parameters={k:p[k] for k in p.files}
    codes={};ids={};labels={}
    for part in ['train','val']:
        raw=np.load(cache/f'X_{part}.npy');ids[part]=np.load(cache/f'ids_{part}.npy');labels[part]=np.load(cache/f'y_{part}.npy')
        x=((raw-np.array(contract['mean']))/np.array(contract['scale'])).astype('float32').reshape(len(raw),-1)
        projected=(x-parameters['mean'])@parameters['components'].T
        q=np.clip(np.rint((projected-parameters['lo'])/parameters['scale']),0,255).astype('uint8')
        codes[part]=np.stack([np.frombuffer(row.tobytes(),dtype='uint8') for row in q])
        assert codes[part].shape==(len(raw),16)
    config=dict(id='server_pca16_histogram',kind='compressed_server',family='histogram',seed=42,
                cache=str(cache),codec=str(profile),feature_bytes=16,max_iter=200,max_leaf_nodes=31,
                learning_rate=.1,l2_regularization=0.,early_stopping=True,internal_validation_fraction=.1,
                sklearn_version=sklearn.__version__)
    save_json(directory/'config.json',config);started=time.monotonic()
    model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.1,random_state=42,
        early_stopping=True,validation_fraction=.1,n_iter_no_change=15)
    model.fit(codes['train'],labels['train']);p=model.predict_proba(codes['val'])[:,1]
    threshold=select_threshold(labels['val'],p);metrics=review_metrics(labels['val'],p,threshold)
    joblib.dump(model,directory/'best.joblib')
    np.testing.assert_allclose(joblib.load(directory/'best.joblib').predict_proba(codes['val'])[:,1],p,rtol=0,atol=0)
    np.savez_compressed(directory/'codec.npz',**parameters)
    np.savez_compressed(directory/'validation_predictions.npz',labels=labels['val'],probabilities=p,ids=ids['val'])
    history=[dict(iteration=i,train_loss=float(-a),internal_train_holdout_loss=float(-b))
             for i,(a,b) in enumerate(zip(model.train_score_,model.validation_score_))]
    save_json(directory/'history.json',history)
    result=dict(id=config['id'],kind=config['kind'],family=config['family'],route='server',representation='pca16_uint8',
                feature_shape=[16],validation_ids_hash=cohort_digest(ids['val']),train_ids_hash=cohort_digest(ids['train']),
                seed=42,best_epoch=None,epochs_completed=model.n_iter_,history=history,
                elapsed_new_training_seconds=time.monotonic()-started,model_bytes=(directory/'best.joblib').stat().st_size,
                parameters=None,codec=str(directory/'codec.npz'),test_set_evaluated=False,**metrics)
    save_json(directory/'result.json',result);print('DIRECT PCA16 SERVER',metrics,flush=True)


def main():
    """Run this small CPU experiment separately from the active GPU controller."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--settings',type=Path,required=True)
    args=parser.parse_args();run(read_json(args.settings))


if __name__=='__main__':main()
