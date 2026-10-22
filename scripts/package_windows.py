"""Build a standalone native Windows notebook with private Drive checkpoint recovery."""
from pathlib import Path
import tempfile
from treevibes.packaging import build_all_in_one,build_colab_archive


def build_windows_notebook(root,archive,destination):
    """Keep Windows orchestration outside the unchanged scientific training modules."""
    import nbformat as nbf
    with tempfile.TemporaryDirectory() as temporary:
        template=build_all_in_one(root,archive,Path(temporary)/'template.ipynb')
        notebook=nbf.read(template,as_version=4)
    notebook.metadata.pop('colab',None)
    notebook.cells[0].source=(
        '# TreeVibes — Windows training and Colab recovery\n\n'
        'Select the existing **Windows Python 3.12 kernel** and run cells in order. '
        'No restart, WSL or changes to the Jupyter environment are required. '
        '**TensorFlow trains on CPU. PyTorch uses the RTX 5060 Ti** through a separate CUDA 12.8 environment. '
        'Most unfinished models use TensorFlow, so the Colab T4 timing estimate does not apply.\n\n'
        'Everything is stored under `D:/ML/TreeVibesWindows`. '
        'Default `DRIVE_ACCESS="public"` downloads the link-shared experiment folder without login or cookies. '
        'Share only that folder as **Anyone with the link → Viewer** before running the download cell. '
        'Optional `DRIVE_ACCESS="private"` uses browser authorization on the Windows PC instead. '
        'Nothing is uploaded back to Drive. '
        'Downloaded originals are retained under `colab_original`; path migration occurs in a separate working copy. '
        'Keep the PC awake and use one campaign per output directory.')
    settings=notebook.cells[2].source
    settings='\n'.join(line for line in settings.splitlines() if not line.startswith('OUTPUT = '))
    notebook.cells[2].source=settings+'''
WORK = Path(os.environ.get("TREEVIBES_WINDOWS_WORK", "D:/ML/TreeVibesWindows")).resolve()
OUTPUT = WORK / "experiments"
DRIVE_FOLDER_ID = "13t5KHIi6-ZmSwMWlIAL64shcu3RcRQkz"
DRIVE_ACCESS = "public"  # Public link: no login. "private" requires browser OAuth on the PC.
SOURCE_FOLDER = None  # Optional existing complete downloaded experiment folder.
'''
    notebook.cells[3].source='## 2. Load the embedded project\n\nThe notebook carries all source files and configurations.'
    embedded=notebook.cells[4].source.splitlines()
    notebook.cells[4].source='\n'.join('ROOT = WORK / "project"' if l.startswith('ROOT = ') else l for l in embedded)+'\n'
    notebook.cells[4].source=notebook.cells[4].source.replace(
        'if name == "treevibes" or name.startswith("treevibes."):',
        'if name == "treevibes" or name.startswith("treevibes.") or name == "scripts" or name.startswith("scripts."):')
    setup='''import json, subprocess
from treevibes.runtime import check_frameworks, worker_environment
from scripts.windows_support import prepare_windows_workers, run_logged

os.environ["KAGGLEHUB_CACHE"] = str(WORK / "downloads")
# A previous CPU-only notebook may have hidden the GPU in this kernel's environment.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
if SMOKE:
    python = os.environ.get("TREEVIBES_WORKER_PYTHON", sys.executable)
    WORKERS = {"tensorflow": python, "pytorch": python}
else:
    WORKERS = prepare_windows_workers(ROOT, WORK)
check_frameworks(WORKERS, WORK / "preflight", require_gpu=False)
if not SMOKE:
    probe = """import json, torch
assert torch.cuda.is_available(), 'RTX GPU is unavailable; see pytorch_gpu.log.'
x = torch.ones(2, 1, 30, 13, device='cuda', requires_grad=True)
w = torch.ones(4, 1, 3, 3, device='cuda', requires_grad=True)
y = torch.nn.functional.conv2d(x, w)
y.sum().backward()
assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in [x,w])
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}))
"""
    run_logged([WORKERS["pytorch"], "-c", probe], WORK / "pytorch_gpu.log",
               env=worker_environment(ROOT), stream=True)
print("Workers ready: TensorFlow CPU; PyTorch", "CPU check" if SMOKE else "RTX GPU")
'''
    download='''from scripts.windows_support import download_drive, download_public_drive

if SMOKE:
    SOURCE = None
elif SOURCE_FOLDER:
    SOURCE = Path(SOURCE_FOLDER).expanduser().resolve()
elif DRIVE_ACCESS == "public":
    SOURCE = download_public_drive(WORK, DRIVE_FOLDER_ID)
elif DRIVE_ACCESS == "private":
    # Opens Google authorization on the Windows PC, not on the Mac viewing this notebook.
    SOURCE = download_drive(WORK, DRIVE_FOLDER_ID)
else:
    raise ValueError('DRIVE_ACCESS must be "public" or "private".')
if SOURCE is not None and not (SOURCE / "notebook_settings.json").is_file():
    raise FileNotFoundError("Expected the complete TreeVibesExperiments folder, including notebook_settings.json.")
print("Original snapshot:", SOURCE)
'''
    notebook.cells[5].source=setup
    notebook.cells[5:5]=[
        nbf.v4.new_markdown_cell('## 3. Download the experiment folder\n\n'
            'Default public mode needs no browser or sign-in: share the experiment folder as '
            '**Anyone with the link → Viewer**. Anyone holding that link can access its contents. '
            'A completed local download is reused, so access can be made restricted again afterward. '
            'Optional private mode requires browser OAuth on the Windows PC; its tokens stay in '
            '`private_auth/rclone.conf` and must not be shared. '
            'If you already copied the complete folder, set `SOURCE_FOLDER` in Settings to skip OAuth/download.'),
        nbf.v4.new_code_cell(download),
        nbf.v4.new_markdown_cell('## 4. Create isolated training environments\n\n'
            'First setup may download several GB of packages. Subsequent runs reuse the environments. '
            'Keras seed compatibility is repaired, and actual CPU/GPU convolution gradients are checked.')]
    run=next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('SETTINGS ='))
    run.source=run.source.replace('settings_path = OUTPUT / "notebook_settings.json"',
                                  'settings_path = WORK / "windows_notebook_settings.json"')
    insertion='''
if not SMOKE:
    migration = (
        "import json; from pathlib import Path; "
        "from scripts.windows_support import import_experiments, seed_campaign; "
        f"import_experiments(Path({str(SOURCE)!r}), Path({str(OUTPUT)!r})); "
        f"settings=json.loads(Path({str(settings_path)!r}).read_text()); "
        f"print(json.dumps(seed_campaign(settings, Path({str(ROOT)!r})), indent=2))"
    )
    run_logged([WORKERS["tensorflow"], "-c", migration], WORK / "windows_resume.log",
               env=worker_environment(ROOT), stream=True, cwd=ROOT)
    # Preserve the original downloaded settings separately from current controller settings.
    (OUTPUT / "notebook_settings.json").write_text(json.dumps(SETTINGS, indent=2))
'''
    run.source=run.source.replace('environment = worker_environment(ROOT)',insertion+'\nenvironment = worker_environment(ROOT)')
    run.source=run.source.replace('# Active SQLite stays on runtime storage; consistent backups are saved to Drive.',
                                  '# Working tracking database and backups stay on this PC.')
    for c in notebook.cells:
        if c.cell_type=='markdown' and c.source.startswith('## 3. Prepare'):
            c.source=c.source.replace('## 3.','## 5.')+(
                '\n\nRecovery prints verified imported run/cache counts first. Recipes still appear '
                'from 1/131, but verified completed models print `Reusing verified completed run.` '
                'Intact interrupted epochs are restored. The raw Kaggle archive is downloaded only '
                'when an unfinished feature cache needs audio. New hardware/framework versions are '
                'recorded in new attempt provenance; continuation is not promised to be bitwise identical.')
        elif c.cell_type=='markdown' and c.source.startswith('## 4. Results'):
            c.source=c.source.replace('## 4.','## 6.')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,destination)
    return destination


if __name__=='__main__':
    root=Path.cwd()
    archive=build_colab_archive(root,root/'artifacts/release/treevibes_windows.zip')
    notebook=build_windows_notebook(root,archive,root/'TreeVibes_Windows_All_In_One.ipynb')
    (root/'artifacts/release/TreeVibes_Windows_All_In_One.ipynb').write_bytes(notebook.read_bytes())
    print(notebook)
