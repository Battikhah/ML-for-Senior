"""Convert selected Keras students and preserve the evidence needed by firmware."""
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from .data import write_json, fingerprint
from .train import normalized, input_contract
from .evaluate import evaluate_probabilities, select_threshold
from .tracking import file_hash,valid_completed

RESOLVER = {'CONV_2D':'AddConv2D','DEPTHWISE_CONV_2D':'AddDepthwiseConv2D',
            'FULLY_CONNECTED':'AddFullyConnected','MAX_POOL_2D':'AddMaxPool2D',
            'AVERAGE_POOL_2D':'AddAveragePool2D','MEAN':'AddMean','RESHAPE':'AddReshape',
            'ADD':'AddAdd','MUL':'AddMul','QUANTIZE':'AddQuantize','DEQUANTIZE':'AddDequantize',
            'RELU':'AddRelu','RELU6':'AddRelu6','LOGISTIC':'AddLogistic','SOFTMAX':'AddSoftmax',
            'PAD':'AddPad','STRIDED_SLICE':'AddStridedSlice'}


def encode_input(features: np.ndarray, tensor_details: dict) -> np.ndarray:
    """Round and saturate using per-tensor input quantization, never guessed ranges."""
    dtype=np.dtype(tensor_details['dtype'])
    if np.issubdtype(dtype,np.floating):
        return np.asarray(features,dtype=dtype)
    scale,zero=tensor_details['quantization']
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Quantization scale must be positive')
    bounds=np.iinfo(dtype)
    return np.clip(np.rint(np.asarray(features)/scale+zero),bounds.min,bounds.max).astype(dtype)


def decode_output(values: np.ndarray, tensor_details: dict) -> np.ndarray:
    """Restore logits before applying the classifier's sigmoid and threshold."""
    if np.issubdtype(np.dtype(tensor_details['dtype']),np.floating):
        return np.asarray(values,dtype='float32')
    scale,zero=tensor_details['quantization']
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Quantization scale must be positive')
    return (np.asarray(values,dtype='float32')-zero)*scale


def tflite_predict(model_path: Path, x: np.ndarray) -> np.ndarray:
    """Run the exported bytes rather than substituting the pre-conversion network."""
    import tensorflow as tf
    interpreter=tf.lite.Interpreter(model_path=str(model_path),num_threads=1)
    interpreter.allocate_tensors()
    i=interpreter.get_input_details()[0];o=interpreter.get_output_details()[0]
    logits=[]
    for row in x:
        encoded=encode_input(row[None,...,None],i)
        interpreter.set_tensor(i['index'],encoded);interpreter.invoke()
        logits.append(float(decode_output(interpreter.get_tensor(o['index']),o).ravel()[0]))
    return expit(np.array(logits))


def _details(value):
    """Serialize only runtime-relevant tensor metadata for a portable deployment contract."""
    scale,zero=value['quantization']
    return dict(dtype=np.dtype(value['dtype']).name,shape=value['shape'].tolist(),
                scale=float(scale),zero_point=int(zero))


def _operators(contents):
    """Read declared FlatBuffer operators without including host delegate pseudo-ops."""
    from tensorflow.lite.python import schema_py_generated as schema
    model=schema.Model.GetRootAsModel(contents,0)
    names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    return [dict(name=names.get(model.OperatorCodes(i).BuiltinCode(),'CUSTOM'),
                 version=model.OperatorCodes(i).Version()) for i in range(model.OperatorCodesLength())]


def _qat(model, x, y, xv, yv, epochs, seed, destination):
    """Fine-tune compatible students under fake quantization using hard-label BCE."""
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot
    tf.keras.utils.set_random_seed(seed)
    quantized=tfmot.quantization.keras.quantize_model(model)
    optimizer=tf.keras.optimizers.Adam(1e-4)
    best=1e30;best_weights=None;history=[]
    for epoch in range(epochs):
        indices=np.random.default_rng(seed+epoch).permutation(len(y))
        for start in range(0,len(y),16):
            selected=indices[start:start+16]
            with tf.GradientTape() as tape:
                logits=quantized(x[selected,...,None],training=True)
                loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y[selected,None].astype('float32'),logits=logits))
            optimizer.apply_gradients(zip(tape.gradient(loss,quantized.trainable_variables),quantized.trainable_variables))
        logits=np.concatenate([quantized(xv[i:i+64,...,None],training=False).numpy().ravel()
                               for i in range(0,len(xv),64)])
        val_loss=float(np.mean(np.logaddexp(0,logits)-yv*logits))
        history.append(dict(epoch=epoch+1,val_loss=val_loss))
        if val_loss<best:
            best=val_loss;best_weights=quantized.get_weights()
    if best_weights is None:
        raise ValueError('QAT requires positive epochs and finite validation loss')
    quantized.set_weights(best_weights)
    write_json(destination/'qat_history.json',history)
    quantized.save(destination/'qat_best.keras')
    return quantized


def export_model(run_dir: Path,dataset: dict,method: str,destination: Path,qat_epochs: int=3) -> dict:
    """Measure post-export quality and leave unmeasured board requirements pending."""
    import tensorflow as tf
    run_dir=Path(run_dir);destination=Path(destination)
    if method not in {'float','ptq','qat'}:
        raise ValueError('Export method must be float, ptq or qat')
    if (destination/'export.json').exists():
        raise ValueError('Export destination already exists; use a new directory for an immutable variant')
    destination.mkdir(parents=True,exist_ok=True)
    try:
        if not valid_completed(run_dir):
            raise ValueError('Only valid completed runs with intact artifact integrity may be exported')
        config=json.loads((run_dir/'config.json').read_text())
        contract=json.loads((run_dir/'input_contract.json').read_text())
        if input_contract(config,dataset)!=contract:
            raise ValueError('Export cache differs from training contract')
        model=tf.keras.models.load_model(run_dir/'best.keras',compile=False)
        train=normalized(dataset['X_train'],contract);val=normalized(dataset['X_val'],contract)
        parent_logits=np.concatenate([model(val[i:i+64,...,None],training=False).numpy().ravel()
                                      for i in range(0,len(val),64)])
        parent_threshold=json.loads((run_dir/'training_result.json').read_text())['validation']['threshold']
        if method=='qat':
            model=_qat(model,train,dataset['y_train'],val,dataset['y_val'],qat_epochs,config['seed'],destination)
        converter=tf.lite.TFLiteConverter.from_keras_model(model)
        calibration_indices=np.random.default_rng(config['seed']).permutation(len(train))[:128]
        if method!='float':
            converter.optimizations=[tf.lite.Optimize.DEFAULT]

            def representative():
                """Expose only training inputs to activation-range calibration."""
                for index in calibration_indices:
                    yield [train[index:index+1,...,None].astype('float32')]

            converter.representative_dataset=representative
            converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
        contents=converter.convert()
        model_path=destination/'model.tflite';model_path.write_bytes(contents)
        interpreter=tf.lite.Interpreter(model_content=contents,num_threads=1);interpreter.allocate_tensors()
        i=interpreter.get_input_details()[0];o=interpreter.get_output_details()[0]
        if method!='float' and (i['dtype']!=np.int8 or o['dtype']!=np.int8 or
                any(np.issubdtype(t['dtype'],np.floating) for t in interpreter.get_tensor_details())):
            raise ValueError('Requested full INT8 export contains float tensors')
        probs=tflite_predict(model_path,val)
        threshold=select_threshold(dataset['y_val'],probs)
        operators=_operators(contents)
        missing=[op['name'] for op in operators if op['name'] not in RESOLVER]
        encoded=encode_input(val[:1,...,None],i)
        interpreter.set_tensor(i['index'],encoded);interpreter.invoke()
        expected=interpreter.get_tensor(o['index']).copy()
        np.savez_compressed(destination/'fixture.npz',raw=dataset['X_val'][:1],encoded=encoded,expected=expected)
        np.savez_compressed(destination/'validation_predictions.npz',labels=dataset['y_val'],probabilities=probs)
        write_json(destination/'input_contract.json',contract)
        meta=dict(state='exported',board_verified=False,board_status='pending',method=method,
            parent_run=str(run_dir.resolve()),parent_model_sha256=file_hash(run_dir/'best.keras'),
            parent_run_id=json.loads((run_dir/'provenance.json').read_text())['run_id']
                if (run_dir/'provenance.json').exists() else 'untracked-test',
            dataset_id=dataset['contract']['dataset_id'],synthetic=dataset['contract'].get('synthetic',False),
            model_sha256=file_hash(model_path),bytes=len(contents),input=_details(i),output=_details(o),
            operators=operators,unresolved_resolver_ops=missing,
            runtime_compatibility='pending_micro_build_and_board',
            calibration_partition='train' if method!='float' else None,
            calibration_ids=dataset['ids_train'][calibration_indices].tolist() if method!='float' else [],
            threshold=threshold,threshold_rule='probability >= threshold',
            validation=evaluate_probabilities(dataset['y_val'],probs,threshold),
            validation_at_parent_threshold=evaluate_probabilities(dataset['y_val'],probs,parent_threshold),
            parent_float_validation=evaluate_probabilities(dataset['y_val'],expit(parent_logits),parent_threshold),
            mean_probability_change=float(np.mean(np.abs(probs-expit(parent_logits)))),
            qat_epochs=qat_epochs if method=='qat' else 0,qat_objective='hard_label_BCE' if method=='qat' else None,
            input_transform='normalize_coefficients_then_quantize',
            feature_parity=dataset['contract'].get('feature_parity','synthetic_only'))
        write_json(destination/'export.json',meta)
        return meta
    except Exception as error:
        write_json(destination/'failure.json',dict(state='failed',method=method,error=str(error),
                                                   error_type=type(error).__name__))
        raise


def create_firmware_bundle(export_dir: Path,destination: Path) -> Path:
    """Emit model, input transform, resolver entries and reference outputs as one bundle."""
    export_dir=Path(export_dir);destination=Path(destination)
    meta=json.loads((export_dir/'export.json').read_text())
    if meta['input']['dtype']!='int8' or meta['output']['dtype']!='int8':
        raise ValueError('ESP32 harness accepts full INT8 exports only')
    if meta['unresolved_resolver_ops']:
        raise ValueError(f'Unresolved micro operators: {meta["unresolved_resolver_ops"]}')
    if file_hash(export_dir/'model.tflite')!=meta['model_sha256']:
        raise ValueError('Exported model changed after validation')
    destination.mkdir(parents=True,exist_ok=True)
    contract=json.loads((export_dir/'input_contract.json').read_text())
    with np.load(export_dir/'fixture.npz',allow_pickle=False) as fixture:
        raw=fixture['raw'].ravel();expected=fixture['expected'].ravel()
    contents=(export_dir/'model.tflite').read_bytes()
    body=',\n'.join('  '+','.join(str(b) for b in contents[start:start+24]) for start in range(0,len(contents),24))
    (destination/'model_data.cc').write_text('#include "model_data.h"\n'
        'alignas(16) const unsigned char g_model[] = {\n'+body+'\n};\n'
        'const unsigned int g_model_len = sizeof(g_model);\n')
    (destination/'model_data.h').write_text('#pragma once\n// Generated immutable model bytes live in flash.\n'
        'extern const unsigned char g_model[];\nextern const unsigned int g_model_len;\n')
    features=contract['features'];coeff=features['coefficients']
    stats=contract['statistics'] or dict(mean=[0.]*coeff,scale=[1.]*coeff)

    def c_array(values):
        """Keep generated floating constants valid even for integral-valued features."""
        return ','.join(f'{float(v):.9e}f' for v in values)

    header=('#pragma once\n#include <cstdint>\n#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
        f'constexpr int kInputCount={len(raw)};\nconstexpr int kCoefficients={coeff};\n'
        f'constexpr float kInputScale={meta["input"]["scale"]:.12e}f;\n'
        f'constexpr int kInputZero={meta["input"]["zero_point"]};\n'
        f'constexpr float kOutputScale={meta["output"]["scale"]:.12e}f;\n'
        f'constexpr int kOutputZero={meta["output"]["zero_point"]};\n'
        f'constexpr float kThreshold={meta["threshold"]:.12e}f;\n'
        f'constexpr int kExpectedOutput={int(expected[0])};\n'
        f'constexpr float kMeans[]={{{c_array(stats["mean"])}}};\n'
        f'constexpr float kScales[]={{{c_array(stats["scale"])}}};\n'
        f'constexpr float kRawFixture[]={{{c_array(raw)}}};\n'
        f'constexpr char kModelHash[]="{meta["model_sha256"]}";\n'
        '// Register only operations present in this exact exported FlatBuffer.\n'
        'inline TfLiteStatus RegisterModelOps(tflite::MicroMutableOpResolver<32>& resolver) {\n')
    for op in sorted({op['name'] for op in meta['operators']}):
        header+=f'  if (resolver.{RESOLVER[op]}()!=kTfLiteOk) return kTfLiteError;\n'
    header+='  return kTfLiteOk;\n}\n'
    (destination/'model_config.h').write_text(header)
    write_json(destination/'bundle.json',meta | {'input_contract':contract})
    shutil.copy2(export_dir/'model.tflite',destination/'model.tflite')
    return destination


def import_board_measurement(path: Path,output: Path) -> dict:
    """Attach observations to exact model bytes without inferring whole-application fit."""
    value=json.loads(Path(path).read_text())
    required={'model_sha256','arena_used_bytes','inference_us','fixture_pass','free_internal_heap',
              'minimum_free_heap','largest_free_block','stack_high_water_bytes','firmware_bytes',
              'app_partition_bytes','scope'}
    if not required<=value.keys() or value['scope'] not in {'inference_harness','complete_application'}:
        raise ValueError('Incomplete hardware measurement or missing scope')
    if len(value['model_sha256'])!=64 or any(not isinstance(value[k],(int,float)) or value[k]<0
            for k in required-{'model_sha256','scope','fixture_pass'}):
        raise ValueError('Invalid hardware measurement values')
    if not isinstance(value['fixture_pass'],bool):
        raise ValueError('fixture_pass must be boolean')
    value['state']='board_measured' if value['fixture_pass'] else 'board_failed'
    write_json(Path(output)/value['model_sha256']/'board_measurement.json',value)
    return value
