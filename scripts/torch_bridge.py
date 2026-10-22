"""Move inference-equivalent weights between trusted Keras files and PyTorch workers."""
import argparse
import json
from pathlib import Path
import numpy as np


def dump_model(model,directory):
    """Record graph connectivity explicitly, including residual branch order and NHWC layout."""
    import tensorflow as tf
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    layers=[];arrays={}
    for layer in model.layers:
        kind=type(layer).__name__
        parents=[] if kind=='InputLayer' else [t._keras_history[0].name for t in tf.nest.flatten(layer.input)]
        weights=layer.get_weights()
        for index,value in enumerate(weights):
            arrays[f'{layer.name}__{index}']=value
        layers.append(dict(name=layer.name,kind=kind,config=layer.get_config(),parents=parents,
                           weights=len(weights)))
    np.savez(directory/'weights.npz',**arrays)
    (directory/'graph.json').write_text(json.dumps(dict(layers=layers,output=model.layers[-1].name),
                                                  default=lambda value:value.item()),encoding='utf-8')
    probe=np.random.default_rng(791).normal(size=(4,*model.input_shape[1:3])).astype('float32')
    np.savez(directory/'probe.npz',features=probe,logits=model(probe[...,None],training=False).numpy().ravel())


def apply_weights(model,path):
    """Restore the original Keras tensor layout before using the existing TFLite exporters."""
    with np.load(path,allow_pickle=False) as arrays:
        for layer in model.layers:
            if layer.get_weights():
                layer.set_weights([arrays[f'{layer.name}__{i}'] for i in range(len(layer.get_weights()))])


def main():
    """Keep TensorFlow imports and CPU serialization out of the CUDA training process."""
    import tensorflow as tf
    from treevibes.config import load_config
    from treevibes.models import build_model
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['dump','restore'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--weights',type=Path)
    args=parser.parse_args()
    tf.keras.utils.set_random_seed(load_config(args.config)['seed'])
    if args.action=='dump':
        model=tf.keras.models.load_model(args.model,compile=False) if args.model else build_model(load_config(args.config))
        dump_model(model,args.directory)
        model.save(args.directory/'template.keras')
    else:
        model=tf.keras.models.load_model(args.directory/'template.keras',compile=False)
        apply_weights(model,args.weights)
        parity=args.directory.parent/'bridge_parity.npz'
        with np.load(parity,allow_pickle=False) as probe:
            np.testing.assert_allclose(model(probe['features'][...,None],training=False).numpy().ravel(),
                                       probe['logits'],rtol=2e-4,atol=5e-5)
        model.save(args.model)


if __name__=='__main__':
    main()
