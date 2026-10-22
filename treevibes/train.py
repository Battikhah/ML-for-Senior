"""Shared supervised and teacher/student training with durable epoch checkpoints."""
import json
import random
import shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from .data import fingerprint, write_json
from .evaluate import evaluate_probabilities, select_threshold

CLASSICAL = {'logistic', 'svm', 'forest', 'histogram'}


def input_contract(config: dict, dataset: dict) -> dict:
    """Tie teachers, students and exported transforms to the same feature semantics."""
    contract = dataset['contract']
    if config['features'] != contract['features']:
        raise ValueError('Feature contract differs from cache; prepare the matching cache')
    return dict(dataset_id=contract['dataset_id'], features=contract['features'],
                split=contract['split'], normalization=config['normalization'],
                statistics=contract['normalization'] if config['normalization'] != 'none' else None)


def normalized(x: np.ndarray, contract: dict) -> np.ndarray:
    """Apply the exact persisted training transform to held-out or device features."""
    x = np.asarray(x, dtype='float32')
    if contract['normalization'] == 'none':
        return x
    stats = contract['statistics']
    return ((x - np.array(stats['mean'], dtype='float32')) /
            np.array(stats['scale'], dtype='float32')).astype('float32')


def distillation_loss(labels, student_logits, teacher_logits, temperature: float, alpha: float,
                      weights=None):
    """Transfer Bernoulli soft targets while preserving the hard-label objective."""
    import tensorflow as tf
    labels = tf.cast(labels, tf.float32)
    soft_targets = tf.math.sigmoid(tf.stop_gradient(teacher_logits) / temperature)
    hard = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=student_logits)
    soft = tf.nn.sigmoid_cross_entropy_with_logits(labels=soft_targets, logits=student_logits / temperature)
    loss = (1 - alpha) * hard + alpha * temperature ** 2 * soft
    if weights is not None:
        loss *= tf.cast(weights, tf.float32)
    return tf.reduce_mean(loss)


def _teacher(path: Path, contract: dict):
    """Reject missing, unfinished or incompatible teacher artifacts before training."""
    import tensorflow as tf
    from .tracking import valid_completed
    path = Path(path)
    if not valid_completed(path):
        raise ValueError('Teacher must be a valid completed run with intact artifact integrity')
    if json.loads((path / 'input_contract.json').read_text()) != contract:
        raise ValueError('Teacher input contract differs from student')
    model = tf.keras.models.load_model(path / 'best.keras', compile=False)
    model.trainable = False
    return model


def train_run(config: dict, dataset: dict, run_dir: Path, teacher_dir: Path | None = None) -> dict:
    """Search only validation performance; test arrays are never accessed here."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    contract = input_contract(config, dataset)
    if (run_dir/'input_contract.json').exists() and json.loads((run_dir/'input_contract.json').read_text())!=contract:
        raise ValueError('Resume input contract differs from the saved checkpoint')
    write_json(run_dir / 'input_contract.json', contract)
    write_json(run_dir / 'config.json', config)
    random.seed(config['seed']); np.random.seed(config['seed'])
    x = normalized(dataset['X_train'], contract); y = dataset['y_train']
    xv = normalized(dataset['X_val'], contract); yv = dataset['y_val']
    family = config['family']; extra = {}
    if family in CLASSICAL:
        import joblib
        from .models import build_classical
        model = build_classical(config); model.fit(x.reshape(len(x), -1), y)
        joblib.dump(model, run_dir / 'best.joblib')
        probs = model.predict_proba(xv.reshape(len(xv), -1))[:, 1]
        extra['epochs_completed'] = 0
    elif family.startswith('reference_'):
        probs, extra = _train_reference(config, x, y, xv, yv, run_dir)
    else:
        probs, extra = _train_keras(config, x, y, xv, yv, run_dir, teacher_dir, contract)
    threshold = select_threshold(yv, probs)
    result = dict(validation=evaluate_probabilities(yv, probs, threshold),
                  coverage=dataset['contract'].get('coverage'),
                  data_audit=dataset['contract'].get('data_audit'), **extra)
    np.savez_compressed(run_dir / 'validation_predictions.npz', labels=yv, probabilities=probs)
    write_json(run_dir / 'training_result.json', result)
    return result


def _train_keras(c, x, y, xv, yv, directory, teacher_dir, contract):
    """Resume committed epochs; partially written checkpoint files are never selected."""
    import tensorflow as tf
    from .models import build_model
    tf.keras.utils.set_random_seed(c['seed'])
    teacher = None
    if c.get('method') == 'kd':
        if teacher_dir is None:
            raise ValueError('Distillation requires --teacher pointing to a completed Keras run')
        teacher = _teacher(teacher_dir, contract)
    state_path = directory / 'epoch_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['config_hash'] != fingerprint(c):
            raise ValueError('Resume configuration differs from checkpoint')
        model = tf.keras.models.load_model(directory / state['checkpoint'])
    else:
        model = build_model(c)
        model.compile(optimizer=tf.keras.optimizers.Adam(c['learning_rate']),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
        state = dict(epoch=0, best_loss=1e30, stale=0, history=[],
                     config_hash=fingerprint(c), best_checkpoint=None)
    counts = np.bincount(y, minlength=2)
    class_weights = len(y) / (2 * counts) if c['class_weight'] else np.ones(2)
    checkpoints = directory / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    for epoch in range(state['epoch'], c['epochs']):
        if state['stale'] >= c['patience']:
            break
        tf.random.set_seed(c['seed'] + epoch)
        epoch_rate=float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        rng = np.random.default_rng(c['seed'] + epoch)
        indices = rng.permutation(len(y)); losses = []
        for start in range(0, len(y), c['batch_size']):
            index = indices[start:start + c['batch_size']]
            xb = x[index].copy(); yb = y[index, None].astype('float32')
            if c['noise']:
                xb += rng.normal(0, c['noise'], xb.shape).astype('float32')
            if c['time_mask']:
                for sample in xb:
                    begin = rng.integers(0, sample.shape[0] - c['time_mask'] + 1)
                    sample[begin:begin + c['time_mask']] = 0
            with tf.GradientTape() as tape:
                logits = model(xb[..., None], training=True)
                weights = class_weights[y[index], None].astype('float32')
                if teacher is None:
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yb, logits=logits) * weights)
                else:
                    targets = teacher(xb[..., None], training=False)
                    loss = distillation_loss(yb, logits, targets, c['temperature'], c['alpha'], weights)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append((float(loss), len(index)))
        val_logits = _keras_logits(model, xv)
        val_loss = float(np.mean(np.logaddexp(0, val_logits) - yv * val_logits))
        if not np.isfinite(val_loss):
            raise ValueError('Training produced nonfinite validation loss')
        checkpoint = f'checkpoints/epoch_{epoch + 1:04}.keras'
        improved = val_loss < state['best_loss']
        state.update(epoch=epoch + 1, checkpoint=checkpoint,
                     stale=0 if improved else state['stale'] + 1)
        if improved:
            state.update(best_loss=val_loss, best_checkpoint=checkpoint)
        state['history'].append(dict(epoch=epoch + 1,
            train_loss=sum(a*b for a,b in losses) / len(y), val_loss=val_loss,
            learning_rate=epoch_rate))
        if state['stale'] and state['stale'] % 5 == 0:
            model.optimizer.learning_rate.assign(tf.maximum(model.optimizer.learning_rate * .5, 1e-6))
        model.save(directory / checkpoint)
        # The pointer is committed after its checkpoint is fully saved.
        write_json(state_path, state)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == c['epochs']:
            print(f'TV_PROGRESS {c["id"]} epoch {epoch+1}/{c["epochs"]}: '
                  f'validation loss={val_loss:.5f}', flush=True)
        for old in checkpoints.glob('*.keras'):
            if str(old.relative_to(directory)) not in {checkpoint, state['best_checkpoint']}:
                old.unlink()
    shutil.copy2(directory / state['checkpoint'], directory / 'last.keras')
    shutil.copy2(directory / state['best_checkpoint'], directory / 'best.keras')
    model = tf.keras.models.load_model(directory / 'best.keras', compile=False)
    return expit(_keras_logits(model, xv)), dict(epochs_completed=state['epoch'],
            parameters=model.count_params(), history=state['history'])


def _keras_logits(model, x):
    """Bound prediction memory without creating a background tf.data thread pool."""
    return np.concatenate([model(x[start:start + 64, ..., None], training=False).numpy().ravel()
                           for start in range(0, len(x), 64)])


def _train_reference(c, x, y, xv, yv, directory):
    """Train the original PyTorch control with checkpointed optimizer and best weights."""
    import torch
    from .reference import build_reference
    torch.manual_seed(c['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_reference(c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    state = dict(epoch=0, best_loss=1e30, stale=0, history=[])
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); state = saved['state']
    for epoch in range(state['epoch'], c['epochs']):
        if state['stale'] >= c['patience']:
            break
        torch.manual_seed(c['seed'] + epoch); model.train()
        indices = np.random.default_rng(c['seed'] + epoch).permutation(len(y)); losses=[]
        for start in range(0, len(y), c['batch_size']):
            index = indices[start:start+c['batch_size']]
            xb = torch.tensor(x[index, None], device=device)
            yb = torch.tensor(y[index, None], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append((float(loss.detach()), len(index)))
        model.eval()
        with torch.no_grad():
            logits = np.concatenate([model(torch.tensor(xv[i:i+64, None], device=device)).cpu().numpy().ravel()
                                     for i in range(0, len(xv), 64)])
        val_loss=float(np.mean(np.logaddexp(0, logits)-yv*logits))
        improved=val_loss < state['best_loss']
        state.update(epoch=epoch+1, stale=0 if improved else state['stale']+1)
        if improved:
            state['best_loss']=val_loss
            torch.save(model.state_dict(), directory / 'best.pt')
        state['history'].append(dict(epoch=epoch+1, val_loss=val_loss,
                                    train_loss=sum(a*b for a,b in losses)/len(y)))
        temporary=directory / 'last.tmp.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), state=state), temporary)
        temporary.replace(directory / 'last.pt')
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == c['epochs']:
            print(f'TV_PROGRESS {c["id"]} epoch {epoch+1}/{c["epochs"]}: '
                  f'validation loss={val_loss:.5f}', flush=True)
    model.load_state_dict(torch.load(directory / 'best.pt', map_location=device, weights_only=True)); model.eval()
    with torch.no_grad():
        probs=np.concatenate([torch.sigmoid(model(torch.tensor(xv[i:i+64,None],device=device))).cpu().numpy().ravel()
                              for i in range(0,len(xv),64)])
    return probs, dict(epochs_completed=state['epoch'], history=state['history'],
                       parameters=sum(p.numel() for p in model.parameters()))


def predict_run(run_dir: Path, raw_features: np.ndarray) -> np.ndarray:
    """Reuse the recorded model and preprocessing for final or robustness evaluation."""
    directory = Path(run_dir)
    config = json.loads((directory / 'config.json').read_text())
    contract = json.loads((directory / 'input_contract.json').read_text())
    x = normalized(raw_features, contract)
    if config['family'] in CLASSICAL:
        import joblib
        return joblib.load(directory / 'best.joblib').predict_proba(x.reshape(len(x), -1))[:, 1]
    if config['family'].startswith('reference_'):
        import torch
        from .reference import build_reference
        model=build_reference(config)
        model.load_state_dict(torch.load(directory/'best.pt', map_location='cpu', weights_only=True)); model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(torch.tensor(x[i:i+64,None]))).numpy().ravel()
                                   for i in range(0,len(x),64)])
    import tensorflow as tf
    model=tf.keras.models.load_model(directory / 'best.keras', compile=False)
    return expit(_keras_logits(model, x))
