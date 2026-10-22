"""Verify GPU adapter inference parity and checkpoint continuity on a CPU test host."""
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('family',['mlp','cnn','depthwise','residual','temporal'])
def test_converted_graph_matches_keras_and_roundtrips(family,tmp_path):
    """Every family must preserve logits before CUDA training or ESP32 conversion."""
    import tensorflow as tf
    import torch
    from treevibes.models import build_model
    from scripts.torch_bridge import dump_model,apply_weights
    from scripts.torch_adapter import GraphModel,save_keras_weights
    config=dict(family=family,width=4,dropout=.1,features=dict(frames=30,coefficients=13))
    model=build_model(config);directory=tmp_path/family;directory.mkdir()
    dump_model(model,directory)
    converted=GraphModel(directory).eval()
    x=np.random.default_rng(8).normal(size=(3,30,13)).astype('float32')
    expected=model(x[...,None],training=False).numpy()
    actual=converted(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
    converted.train();converted(torch.tensor(x)).sum().backward();converted.eval()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in converted.parameters())
    save_keras_weights(converted,directory/'updated.npz')
    apply_weights(model,directory/'updated.npz')
    np.testing.assert_allclose(model(x[...,None],training=False).numpy(),actual,rtol=2e-4,atol=2e-5)


def test_gpu_backend_keeps_keras_epochs_and_resumes_torch_checkpoint(tmp_path,monkeypatch):
    """A disconnect across framework migration must keep both old and new committed epochs."""
    import os,subprocess,sys
    from treevibes.synthetic import save_synthetic
    from treevibes.data import write_json
    from treevibes.tracking import run_experiment,valid_completed
    from treevibes.runtime import worker_environment
    import treevibes.train as training
    output=tmp_path/'out';cache=save_synthetic(tmp_path/'cache',40)
    config=tmp_path/'model.json';write_json(config,dict(id='migrate',family='mlp',width=4,epochs=3,patience=3))
    original=training.write_json
    def disconnect(path,value):
        """Interrupt only after a real Keras checkpoint and pointer are durable."""
        original(path,value)
        if Path(path).name=='epoch_state.json':raise RuntimeError('Keras disconnected')
    with monkeypatch.context() as patch:
        patch.setattr(training,'write_json',disconnect)
        with pytest.raises(RuntimeError,match='Keras disconnected'):
            run_experiment(config,cache,output)
    old_state=next(output.glob('runs/*/*/attempt_*/epoch_state.json'))
    old_history=json.loads(old_state.read_text())['history'][0]
    request=dict(config=str(config),cache=str(cache),output=str(output),reply=str(tmp_path/'reply.json'))
    path=tmp_path/'request.json';write_json(path,request)
    env=worker_environment();env.update(TREEVIBES_TF_PYTHON=sys.executable,TREEVIBES_ALLOW_CPU_TEST='1')
    code='''import json
+from pathlib import Path
+import treevibes.data as data
+actual=data.write_json
+def stop(path,value):
+    actual(path,value)
+    if Path(path).name=='epoch_state.json' and value.get('backend')=='torch_cuda_v1' and value['epoch']==2:
+        raise RuntimeError('Torch disconnected after commit')
+data.write_json=stop
+from scripts.gpu_campaign import execute_training
+execute_training(json.loads(Path(REQUEST).read_text()))
+'''.replace('\n+','\n').replace('REQUEST',repr(str(path)))
    first=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert first.returncode and 'Torch disconnected after commit' in first.stderr,first.stdout+first.stderr
    completed=subprocess.run([sys.executable,'-m','scripts.gpu_campaign','--job',str(path)],env=env,capture_output=True,text=True)
    assert not completed.returncode,completed.stdout+completed.stderr
    result=json.loads((tmp_path/'reply.json').read_text())
    assert result['epochs_completed']==3 and len(result['history'])==3
    assert result['history'][0]==old_history
    assert result['history'][1]['backend']=='pytorch_cuda'
    assert result['backend']=='pytorch_cuda' and valid_completed(Path(result['run_dir']))
    assert (Path(result['run_dir'])/'best.keras').exists()


def test_gpu_notebook_routes_fits_to_cuda_adapter(tmp_path):
    """The delivered file must select the GPU controller while retaining CPU export tools."""
    import ast,nbformat
    from treevibes.packaging import build_colab_archive
    from scripts.package_windows_gpu import build_gpu_notebook
    root=Path.cwd();archive=build_colab_archive(root,tmp_path/'source.zip')
    notebook=nbformat.read(build_gpu_notebook(root,archive,tmp_path/'gpu.ipynb'),as_version=4)
    nbformat.validate(notebook)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    for source in cells:ast.parse(source)
    run=next(s for s in cells if s.startswith('SETTINGS ='))
    assert '"-m", "scripts.gpu_campaign"' in run
    assert 'environment["TREEVIBES_TF_PYTHON"]' in run
    assert 'TREEVIBES_ALLOW_CPU_TEST' in run
