"""Keep notebook setup independent of Colab's preloaded Python and CUDA packages."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def worker_environment(project: Path | None = None) -> dict:
    """Prevent notebook imports and GPU allocations from leaking into training workers."""
    env = os.environ.copy()
    env.update(TF_USE_LEGACY_KERAS='1', TF_FORCE_GPU_ALLOW_GROWTH='true',
               TF_CPP_MIN_LOG_LEVEL='2', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               MPLBACKEND='Agg')
    # Colab may inject its global site-packages through PYTHONPATH even into a venv.
    env['PYTHONPATH'] = str(Path(project or Path(__file__).parent.parent).resolve())
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS']:
        env.setdefault(name, '2')
    return env


def run_logged(command: list[str], log: Path, env: dict | None = None,
               stream: bool = False, cwd: Path | None = None, prefix: str | None = None) -> None:
    """Save complete diagnostics and show the actual child error when a stage fails."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env or worker_environment(), cwd=cwd)
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                if stream and (prefix is None or line.startswith(prefix)):
                    print(line, end='', flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status:
        raise RuntimeError(f'Worker failed (exit {status}). Full log: {log}\n{log.read_text()[-6000:]}')


def select_worker_python(runtime_root: Path, output: Path) -> str:
    """Provision Python 3.12 without requiring the notebook kernel to use that version."""
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    runtime_root, output = Path(runtime_root).resolve(), Path(output)
    uv = shutil.which('uv')
    env = worker_environment()
    env.update(UV_PYTHON_INSTALL_DIR=str(runtime_root/'python'),
               UV_PYTHON_BIN_DIR=str(runtime_root/'bin'), UV_CACHE_DIR=str(runtime_root/'uv-cache'))
    if not uv:
        tools = runtime_root/'bootstrap'
        uv = str(tools/'bin/uv')
        if not Path(uv).is_file():
            run_logged([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--target', str(tools), 'uv==0.12.18'], output/'python_bootstrap.log', env=env)
    print('Provisioning Python 3.12 for training workers; keeping the notebook kernel unchanged.', flush=True)
    run_logged([uv, 'python', 'install', '3.12'], output/'python_install.log', env=env)
    found = subprocess.run([uv, 'python', 'find', '--managed-python', '3.12'],
                           capture_output=True, text=True, env=env)
    if found.returncode or not found.stdout.strip():
        raise RuntimeError(f'Cannot locate the installed Python 3.12 interpreter:\n{found.stderr}')
    return found.stdout.strip()


def repair_keras_random_bounds(package: Path | None = None) -> list[dict]:
    """Correct tf-keras 2.16's integral float seed bounds without changing their values."""
    if package is None:
        import importlib.util
        package = Path(importlib.util.find_spec('tf_keras').origin).parent
    records = []
    for name in ['src/backend.py', 'src/utils/tf_utils.py']:
        path = Path(package)/name
        original = path.read_text()
        # Python 3.12 removed the implicit conversion accepted by older randint().
        repaired = original.replace('randint(1, 1e9)', 'randint(1, 1000000000)')
        record = dict(file=str(path), replacements=original.count('randint(1, 1e9)'),
                      before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                      after_sha256=hashlib.sha256(repaired.encode()).hexdigest())
        if repaired != original:
            temporary = path.with_suffix('.compat.tmp')
            temporary.write_text(repaired)
            temporary.replace(path)
        records.append(record)
    return records


def prepare_runtimes(project: Path, output: Path, runtime_root: Path = Path('/content/treevibes-runtimes')) -> dict:
    """Install each framework with its own exact CUDA dependencies, leaving Colab untouched."""
    project, output, runtime_root = Path(project).resolve(), Path(output), Path(runtime_root)
    base_python = select_worker_python(runtime_root, output)
    checked = subprocess.run([base_python, '-c', 'import json,sys; print(json.dumps(list(sys.version_info[:3])))'],
                             check=True, capture_output=True, text=True, env=worker_environment(project))
    base_version = json.loads(checked.stdout)
    if base_version[:2] != [3, 12]:
        raise RuntimeError(f'Training workers require Python 3.12; selected {base_version}')
    packaged_locks = sorted((project/'requirements').glob('colab-*-py312.txt'))
    dependency_source = (project/'pyproject.toml').read_bytes() + b''.join(p.read_bytes() for p in packaged_locks)
    key = hashlib.sha256(dependency_source +
                         f'{base_version}-{platform.system()}-{platform.machine()}'.encode()).hexdigest()[:12]
    workers = {}
    for framework, extra in [('tensorflow', 'keras'), ('pytorch', 'reference')]:
        destination = runtime_root/key/framework
        python = destination/'bin/python'
        ready = destination/'ready.json'
        lock_dir = output/'runtime_locks'/key/framework
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock = lock_dir/'requirements.txt'
        env = worker_environment(project)
        if not python.exists():
            print(f'Creating isolated {framework} environment...', flush=True)
            run_logged([base_python, '-m', 'venv', str(destination)], lock_dir/'venv.log', env=env)
        if not ready.exists():
            print(f'Installing {framework}; full setup log: {lock_dir / "install.log"}', flush=True)
            command = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check']
            if lock.exists():
                command += ['-r', str(lock)]
            elif platform.system() == 'Linux' and platform.machine() == 'x86_64':
                packaged_lock = project/'requirements'/f'colab-{framework}-py312.txt'
                if packaged_lock.exists():
                    command += ['-r', str(packaged_lock)]
            command += ['-e', f'{project}[{extra}]']
            if framework == 'tensorflow':
                command.append('tensorflow[and-cuda]==2.16.2')
            run_logged(command, lock_dir/'install.log', env=env, cwd=project)
            if framework == 'tensorflow':
                # The CUDA extra supplies cuDNN 8 here; no PyTorch wheel can replace it.
                run_logged([str(python), '-c',
                            'from treevibes.notebook_setup import setup_colab_cuda; setup_colab_cuda()'],
                           lock_dir/'cuda_links.log', env=env, cwd=project)
            freeze = subprocess.run([str(python), '-m', 'pip', 'freeze', '--exclude-editable'],
                                    check=True, capture_output=True, text=True, env=env)
            lock.write_text(freeze.stdout)
            ready.write_text(json.dumps(dict(framework=framework, requirements_key=key)))
        if framework == 'tensorflow':
            # Apply on ready environments too, so recovery requires no reinstall.
            run_logged([str(python), '-c',
                        'import json; from treevibes.runtime import repair_keras_random_bounds; '
                        'print(json.dumps(repair_keras_random_bounds(), indent=2))'],
                       lock_dir/'keras_seed_compatibility.log', env=env, cwd=project)
        workers[framework] = str(python)
        print(f'{framework} environment ready.', flush=True)
    return workers


def check_frameworks(workers: dict, output: Path, require_gpu: bool = True) -> dict:
    """Require real convolution and finite gradients before any dataset download or fit."""
    probes = {
        'tensorflow': '''import json, tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
assert gpus or not REQUIRE_GPU, 'No TensorFlow GPU is available in this runtime.'
for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.utils.set_random_seed(42)
with tf.device('/GPU:0' if REQUIRE_GPU else '/CPU:0'):
    x = tf.Variable(tf.ones((1, 30, 13, 1)))
    layer = tf.keras.layers.Conv2D(4, 3, padding='same')
    dropout = tf.keras.layers.Dropout(0.1)
    with tf.GradientTape() as tape:
        y = dropout(layer(x), training=True)
        loss = tf.reduce_sum(y)
    gradients = tape.gradient(loss, [x] + layer.trainable_variables)
assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
assert not REQUIRE_GPU or 'GPU' in y.device
print(json.dumps(dict(version=tf.__version__, device=y.device)))
''',
        'pytorch': '''import json, torch
assert torch.cuda.is_available() or not REQUIRE_GPU, 'No PyTorch GPU is available in this runtime.'
device = 'cuda' if REQUIRE_GPU else 'cpu'
x = torch.ones(1, 1, 30, 13, device=device, requires_grad=True)
kernel = torch.ones(4, 1, 3, 3, device=device, requires_grad=True)
y = torch.nn.functional.conv2d(x, kernel)
y.sum().backward()
assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in [x, kernel])
print(json.dumps(dict(version=torch.__version__, device=str(y.device))))
'''}
    report = {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for framework, source in probes.items():
        completed = subprocess.run([workers[framework], '-c', f'REQUIRE_GPU={require_gpu!r}\n'+source],
                                   capture_output=True, text=True, env=worker_environment())
        report[framework] = dict(passed=completed.returncode == 0, stdout=completed.stdout, stderr=completed.stderr)
        print(framework, completed.stdout.strip() or 'preflight failed; see gpu_check.json', flush=True)
    (output/'gpu_check.json').write_text(json.dumps(report, indent=2))
    failures = [name+':\n'+value['stderr'][-6000:] for name, value in report.items() if not value['passed']]
    if failures:
        raise RuntimeError('Framework preflight failed before dataset work:\n'+'\n'.join(failures))
    return report
