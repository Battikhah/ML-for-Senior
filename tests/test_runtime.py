"""Verify worker isolation and actionable failures without an authenticated Colab UI."""
import json
import subprocess
import sys


def test_runtime_can_import_without_loading_numeric_frameworks():
    """Setup must remain safe even when the notebook kernel has incompatible packages."""
    code = ('import sys; import treevibes.runtime; '
            'assert not {"tensorflow", "torch", "numpy"}.intersection(sys.modules)')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_preflight_executes_both_frameworks_in_fresh_cpu_workers(tmp_path):
    """CPU smoke mode must exercise real convolution and backpropagation, not imports."""
    from treevibes.runtime import check_frameworks
    report = check_frameworks({'tensorflow':sys.executable, 'pytorch':sys.executable},
                              tmp_path, require_gpu=False)
    assert all(v['passed'] for v in report.values())
    saved = json.loads((tmp_path/'gpu_check.json').read_text())
    assert set(saved) == {'tensorflow', 'pytorch'}


def test_logged_failure_includes_the_actual_child_error(tmp_path):
    """A generic wrapper must never hide the exception that requires attention."""
    import pytest
    from treevibes.runtime import run_logged
    log = tmp_path/'failure.log'
    with pytest.raises(RuntimeError, match='specific diagnostic failure'):
        run_logged([sys.executable,'-c','raise ValueError("specific diagnostic failure")'], log)
    assert 'specific diagnostic failure' in log.read_text()


def test_python313_kernel_selects_managed_python312(tmp_path, monkeypatch):
    """A newer Colab kernel must provision supported workers instead of failing its guard."""
    from types import SimpleNamespace
    import shutil
    import treevibes.runtime as runtime
    commands=[]
    monkeypatch.setattr(runtime, 'sys', SimpleNamespace(version_info=(3,13,0), executable=sys.executable))
    monkeypatch.setattr(shutil, 'which', lambda name: '/tools/uv')

    def logged(command, log, **kwargs):
        """Substitute only the external interpreter download boundary."""
        commands.append(command)

    def find(command, **kwargs):
        """Represent uv reporting its installed interpreter location."""
        return subprocess.CompletedProcess(command,0,stdout='/managed/bin/python3.12\n',stderr='')

    monkeypatch.setattr(runtime, 'run_logged', logged)
    monkeypatch.setattr(runtime.subprocess, 'run', find)
    selected=runtime.select_worker_python(tmp_path/'runtime',tmp_path/'output')
    assert selected == '/managed/bin/python3.12'
    assert commands == [['/tools/uv','python','install','3.12']]


def test_python312_kernel_reuses_its_interpreter(tmp_path, monkeypatch):
    """A compatible kernel must not need another interpreter download."""
    from types import SimpleNamespace
    import treevibes.runtime as runtime
    monkeypatch.setattr(runtime, 'sys', SimpleNamespace(version_info=(3,12,13),executable='/python3.12'))
    assert runtime.select_worker_python(tmp_path/'runtime',tmp_path/'output') == '/python3.12'


def test_runtime_creation_uses_worker_version_instead_of_kernel_version(tmp_path, monkeypatch):
    """A 3.13 supervisor must create both venvs with the selected 3.12 interpreter."""
    from types import SimpleNamespace
    from pathlib import Path
    import treevibes.runtime as runtime
    monkeypatch.setattr(runtime,'sys',SimpleNamespace(version_info=(3,13,0),executable='/kernel/python3.13'))
    monkeypatch.setattr(runtime,'select_worker_python',lambda *args:'/managed/python3.12')
    commands=[]

    def logged(command, log, **kwargs):
        """Replace external installs while retaining the venv creation side effect."""
        commands.append(command)
        if command[1:3]==['-m','venv']:
            destination=Path(command[-1]);(destination/'bin').mkdir(parents=True)
            (destination/'bin/python').touch()

    def completed(command, **kwargs):
        """Return interpreter metadata or an empty freeze from the external process."""
        return subprocess.CompletedProcess(command,0,stdout='[3,12,13]' if '-c' in command else '',stderr='')

    monkeypatch.setattr(runtime,'run_logged',logged)
    monkeypatch.setattr(runtime.subprocess,'run',completed)
    project=tmp_path/'project';project.mkdir();(project/'pyproject.toml').write_text('[project]\nname="test"\n')
    workers=runtime.prepare_runtimes(project,tmp_path/'out',tmp_path/'runtimes')
    assert set(workers)=={'tensorflow','pytorch'}
    creates=[c for c in commands if c[1:3]==['-m','venv']]
    assert len(creates)==2 and all(c[0]=='/managed/python3.12' for c in creates)
    runtime.prepare_runtimes(project,tmp_path/'out',tmp_path/'runtimes')
    repairs=[c for c in commands if '-c' in c and 'repair_keras_random_bounds' in c[-1]]
    assert len(repairs)==2  # Existing ready environments need the fix as well.


def test_worker_discards_notebook_only_plot_backend(monkeypatch):
    """Isolated environments do not contain Colab's inline plotting plugin."""
    from treevibes.runtime import worker_environment
    monkeypatch.setenv('MPLBACKEND','module://matplotlib_inline.backend_inline')
    assert worker_environment()['MPLBACKEND']=='Agg'


def test_keras_seed_repair_handles_seeded_and_unseeded_calls(tmp_path):
    """Python 3.12 rejects the float bound still shipped in tf-keras 2.16."""
    import ast
    import importlib.util
    import operator
    import random
    from pathlib import Path
    from types import SimpleNamespace
    from treevibes.runtime import repair_keras_random_bounds
    installed=Path(importlib.util.find_spec('tf_keras').origin).parent
    package=tmp_path/'tf_keras'
    for name in ['src/backend.py','src/utils/tf_utils.py']:
        target=package/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((installed/name).read_bytes())
    records=repair_keras_random_bounds(package)
    assert sum(r['replacements'] for r in records)==3
    class StrictRandom(random.Random):
        """Enforce Python 3.12 integer arguments while testing on an older host."""
        def randrange(self, start, stop=None, step=1):
            """Preserve the normal generator after validating the public integer contract."""
            operator.index(start)
            if stop is not None: operator.index(stop)
            return super().randrange(start,stop,step)
    source=ast.parse((package/'src/utils/tf_utils.py').read_text())
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='get_random_seed')
    backend=SimpleNamespace(_SEED_GENERATOR=SimpleNamespace(generator=StrictRandom(42)))
    scope=dict(backend=backend,random=StrictRandom(42))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual tf-keras seed function>','exec'),scope)
    expected=StrictRandom(42)
    assert [scope['get_random_seed']() for _ in range(4)]==[expected.randint(1,10**9) for _ in range(4)]
    backend._SEED_GENERATOR.generator=None
    assert isinstance(scope['get_random_seed'](),int)
    assert all(r['replacements']==0 for r in repair_keras_random_bounds(package))
