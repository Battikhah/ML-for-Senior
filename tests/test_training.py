"""Check metric selection and real teacher/student updates on synthetic inputs."""
import numpy as np
import pytest


def test_metrics_and_threshold_have_independent_expected_values():
    """A favorable validation threshold must not change the underlying probabilities."""
    from treevibes.evaluate import evaluate_probabilities, select_threshold
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .4, .35, .8])
    result = evaluate_probabilities(y, p, .5)
    assert result['precision'] == 1.
    assert result['recall'] == .5
    assert result['f1'] == pytest.approx(2 / 3)
    assert result['confusion'] == [[2, 0], [1, 1]]
    assert evaluate_probabilities(y, p, select_threshold(y, p))['f1'] == .8
    assert evaluate_probabilities(np.zeros(4), p, .5)['roc_auc'] is None


def test_distillation_has_student_gradients_and_freezes_teacher():
    """Soft targets influence the student without ever training the teacher."""
    import treevibes
    tf = pytest.importorskip('tensorflow')
    from treevibes.train import distillation_loss
    student = tf.Variable([[.1], [-.3]])
    teacher = tf.Variable([[2.], [-2.]])
    labels = tf.constant([[1.], [0.]])
    with tf.GradientTape(persistent=True) as tape:
        loss = distillation_loss(labels, student, teacher, 3., .75)
    assert np.any(tape.gradient(loss, student).numpy() != 0)
    assert tape.gradient(loss, teacher) is None
    hard = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=student))
    assert float(distillation_loss(labels, student, teacher, 3., 0)) == pytest.approx(float(hard))


def test_supervised_and_kd_checkpoint_resume_and_contract(tmp_path,complete_training_run):
    """Persist optimizer/epoch state and reject a teacher trained on different data."""
    import treevibes
    pytest.importorskip('tensorflow')
    from treevibes.synthetic import synthetic_dataset
    from treevibes.train import train_run, predict_run
    from treevibes.config import DEFAULTS
    from treevibes.data import DEFAULT_FEATURES, write_json
    dataset = synthetic_dataset(40)
    config = DEFAULTS | dict(id='smoke', family='mlp', width=4, epochs=2,
                             features=DEFAULT_FEATURES, dropout=0.)
    teacher_dir = tmp_path / 'teacher'; teacher_dir.mkdir()
    write_json(teacher_dir / 'config.json', config)
    result = train_run(config, dataset, teacher_dir)
    assert result['epochs_completed'] == 2
    assert (teacher_dir / 'last.keras').exists() and (teacher_dir / 'best.keras').exists()
    before = predict_run(teacher_dir, dataset['X_val'])
    result2 = train_run(config, dataset, teacher_dir)
    np.testing.assert_allclose(before, predict_run(teacher_dir, dataset['X_val']))
    assert result2['epochs_completed'] == 2
    student_dir = tmp_path / 'student'; student_dir.mkdir()
    student = config | {'id': 'student', 'method': 'kd', 'epochs': 1}
    write_json(student_dir / 'config.json', student)
    from treevibes.data import write_json
    complete_training_run(teacher_dir)
    train_run(student, dataset, student_dir, teacher_dir)
    incompatible = dict(dataset, contract=dataset['contract'] | {'dataset_id': 'other'})
    with pytest.raises(ValueError, match='contract'):
        train_run(student, incompatible, tmp_path / 'bad', teacher_dir)


@pytest.mark.parametrize('family',['svm','forest','histogram','reference_b','temporal','residual','depthwise'])
def test_other_families_fit_real_tiny_data(tmp_path,family):
    """Adapters must execute actual fits, not just instantiate their underlying estimator."""
    import treevibes
    from treevibes.synthetic import synthetic_dataset
    from treevibes.config import DEFAULTS
    from treevibes.data import DEFAULT_FEATURES
    from treevibes.train import train_run,predict_run
    config=DEFAULTS | dict(id='fit_'+family,family=family,width=4,epochs=1,features=DEFAULT_FEATURES)
    data=synthetic_dataset(20)
    result=train_run(config,data,tmp_path)
    probs=predict_run(tmp_path,data['X_val'])
    assert probs.shape==(8,) and np.isfinite(probs).all()
    assert result['validation']['count']==8


def test_resume_rejects_different_training_data(tmp_path):
    """A public trainer call must not attach an old checkpoint to a new dataset."""
    from treevibes.synthetic import synthetic_dataset
    from treevibes.config import DEFAULTS
    from treevibes.data import DEFAULT_FEATURES
    from treevibes.train import train_run
    c=DEFAULTS|dict(id='resume',family='mlp',width=4,epochs=1,features=DEFAULT_FEATURES)
    train_run(c,synthetic_dataset(20),tmp_path)
    with pytest.raises(ValueError,match='contract'):
        train_run(c,synthetic_dataset(22),tmp_path)
