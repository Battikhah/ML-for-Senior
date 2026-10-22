"""Exercise real forward passes rather than checking architecture source text."""
import numpy as np
import pytest


@pytest.mark.parametrize('family', ['mlp', 'temporal', 'cnn', 'residual', 'depthwise'])
def test_neural_models_produce_logits_and_gradients(family):
    """Every neural family must train on the declared input contract."""
    tf = pytest.importorskip('tensorflow')
    from treevibes.models import build_model
    config = dict(family=family, features=dict(frames=30, coefficients=13), width=8, dropout=.1)
    model = build_model(config)
    x = tf.ones((2, 30, 13, 1))
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.reduce_sum(logits)
    assert logits.shape == (2, 1)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in tape.gradient(loss, model.trainable_variables))
    assert build_model(config | {'width': 4}).count_params() < model.count_params()


@pytest.mark.parametrize('family', ['reference_a', 'reference_b'])
def test_reference_pooling_and_forward(family):
    """The two original pooling choices must remain usable at 30 by 13."""
    torch = pytest.importorskip('torch')
    from treevibes.reference import build_reference
    config = dict(family=family, features=dict(frames=30, coefficients=13), dropout=.3)
    model = build_reference(config)
    assert model(torch.ones(2, 1, 30, 13)).shape == (2, 1)
    assert model.fc1.in_features == (192 if family == 'reference_a' else 1344)


def test_catalog_is_reproducible_and_diverse(tmp_path):
    """Regeneration must not change the meaning or identity of named runs."""
    from treevibes.catalog import generate_catalog
    from treevibes.config import load_config
    paths = generate_catalog(tmp_path)
    configs = [load_config(p) for p in paths]
    assert len(configs) >= 60
    assert len({c['id'] for c in configs}) == len(configs)
    assert len({c['family'] for c in configs}) >= 10
    assert {c['target'] for c in configs} == {'local', 'remote'}
    before = {p.name: p.read_bytes() for p in paths}
    assert before == {p.name: p.read_bytes() for p in generate_catalog(tmp_path)}
