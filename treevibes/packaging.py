"""Build a portable Colab source archive from an explicit source-only allowlist."""
from pathlib import Path
import zipfile
import base64


def build_colab_archive(root: Path,destination: Path) -> Path:
    """Exclude credentials, trained artifacts, Git metadata and generated model bundles."""
    root=Path(root);destination=Path(destination)
    patterns=['treevibes/*.py','scripts/*.py','experiments/*/*.json','notebooks/*.ipynb','requirements/*.txt',
              'tests/*.py','docs/superpowers/specs/*.md','docs/superpowers/plans/*.md']
    files={path for pattern in patterns for path in root.glob(pattern)}
    files.update(root/name for name in ['README.md','pyproject.toml','requirements-colab.txt',
        'TreeVibes_CNN_Training.ipynb','deployment/esp32/CMakeLists.txt','deployment/esp32/sdkconfig.defaults',
        'deployment/esp32/README.md','deployment/esp32/main/main.cc',
        'deployment/esp32/main/CMakeLists.txt','deployment/esp32/main/idf_component.yml'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.is_file() and not path.is_symlink():
                archive.write(path,path.relative_to(root))
    return destination


def build_colab_launcher(root: Path,archive: Path,destination: Path) -> Path:
    """Embed the reviewed source ZIP so VS Code can bootstrap a remote Colab kernel."""
    import nbformat
    payload=base64.b64encode(Path(archive).read_bytes()).decode('ascii')
    notebook=nbformat.read(Path(root)/'notebooks/01_prepare_data.ipynb',as_version=4)
    notebook.cells[0].source=('# TreeVibes — VS Code / Colab launcher\n\n'
        'Select your Colab T4 kernel in VS Code, then run cells top to bottom. '
        'The first cell sends the reviewed source archive to the remote runtime. '
        'No browser upload widget or access to the laptop filesystem is required. '
        'Real data preparation remains disabled until you enable RUN_PREPARATION. '
        'Use the notebooks extracted under `/content/treevibes_project/notebooks` for subsequent stages.')
    source=('from pathlib import Path\nimport os, base64, io, zipfile\n'
        '# This payload contains source/configuration only, without credentials or trained data.\n'
        f'archive_data = base64.b64decode({payload!r})\n'
        'destination = Path("/content/treevibes_project") if "COLAB_RELEASE_TAG" in os.environ else Path.cwd() / "treevibes_project"\n'
        'destination.mkdir(parents=True, exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:\n'
        '    for member in archive.infolist():\n'
        '        if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):\n'
        '            raise ValueError("Unsafe archive path")\n'
        '    archive.extractall(destination)\n'
        'os.chdir(destination)\nprint("Project transferred:", destination)\n')
    notebook.cells.insert(1,nbformat.v4.new_code_cell(source))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,destination)
    return destination


def build_all_in_one(root: Path, archive: Path, destination: Path) -> Path:
    """Deliver one Run All notebook with its complete source embedded and no manual handoffs."""
    import nbformat as nbf
    payload = base64.b64encode(Path(archive).read_bytes()).decode('ascii')
    settings = '''from pathlib import Path
import os, sys

# Full runs all 131 recipes; screening runs a short search plus distillation.
SMOKE = os.environ.get("TREEVIBES_SMOKE", "0") == "1"
MODE = "smoke" if SMOKE else "full"
EPOCHS = 1 if SMOKE else 60
EXPORT_TOP_N = 1 if SMOKE else 3
QAT_EPOCHS = 1 if SMOKE else 3
RUN_FINAL_TEST = False  # Enable only when ready to lock finalists and release test results.
GROUPS_CSV = None  # Optional tree/session metadata CSV: filepath,group.
DATASET_ROOT = None  # None automatically reuses/downloads the Kaggle dataset.
OUTPUT = Path(os.environ.get("TREEVIBES_OUTPUT", "/content/drive/MyDrive/TreeVibesExperiments" if not SMOKE else "artifacts/one_notebook_smoke")).resolve()
'''
    embedded = ('#@title Load the embedded project\nfrom pathlib import Path\n'
                'import base64, io, os, sys, zipfile\n'
                f'payload = base64.b64decode({payload!r})\n'
                'ROOT = Path("/content/treevibes_one_notebook") if "COLAB_RELEASE_TAG" in os.environ else Path.cwd()/"treevibes_one_notebook"\n'
                'ROOT.mkdir(parents=True, exist_ok=True)\n'
                'with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n'
                '    for member in archive.infolist():\n'
                '        if not (ROOT/member.filename).resolve().is_relative_to(ROOT.resolve()):\n'
                '            raise ValueError("Unsafe embedded archive path")\n'
                '    archive.extractall(ROOT)\n'
                'sys.path.insert(0, str(ROOT))\n'
                'for name in list(sys.modules):\n'
                '    if name == "treevibes" or name.startswith("treevibes."):\n'
                '        del sys.modules[name]\n'
                'os.chdir(ROOT)\nprint("Embedded project ready.")\n')
    setup = '''from treevibes.runtime import prepare_runtimes, check_frameworks, run_logged, worker_environment
import json

if SMOKE:
    # Local verification is explicit and never downloads the field dataset.
    python = os.environ.get("TREEVIBES_WORKER_PYTHON", sys.executable)
    WORKERS = {"tensorflow": python, "pytorch": python}
else:
    if "COLAB_RELEASE_TAG" not in os.environ:
        raise RuntimeError("Select a Colab T4 kernel to run the real study. Local verification uses TREEVIBES_SMOKE=1.")
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORKERS = prepare_runtimes(ROOT, OUTPUT)

OUTPUT.mkdir(parents=True, exist_ok=True)
check_frameworks(WORKERS, OUTPUT, require_gpu=not SMOKE)
print("Environment verified. Starting the experiment workflow.")
'''
    run = '''SETTINGS = dict(mode=MODE, output=str(OUTPUT), epochs=EPOCHS,
                export_top_n=EXPORT_TOP_N, qat_epochs=QAT_EPOCHS,
                final_test=RUN_FINAL_TEST, groups_csv=str(GROUPS_CSV) if GROUPS_CSV else None,
                dataset_root=str(DATASET_ROOT) if DATASET_ROOT else None,
                reference_python=WORKERS["pytorch"])
settings_path = OUTPUT / "notebook_settings.json"
settings_path.write_text(json.dumps(SETTINGS, indent=2))
environment = worker_environment(ROOT)
# Active SQLite stays on runtime storage; consistent backups are saved to Drive.
environment["TREEVIBES_TRACKING_DIR"] = str(ROOT / "tracking")
run_logged([WORKERS["tensorflow"], "-m", "treevibes.campaign", "--settings", str(settings_path)],
           OUTPUT / "campaign.log", env=environment, stream=True, cwd=ROOT)
'''
    results = '''from IPython.display import display, HTML, Image
SUMMARY = json.loads((OUTPUT / "campaign_summary.json").read_text())
print("Status:", SUMMARY["state"])
print("Training recipes:", SUMMARY["completed_runs"], "/", SUMMARY["planned_runs"])
print("Exports:", len(SUMMARY["exports"]), "| Recorded failures:", len(SUMMARY["failures"]))
print("Excel tracker:", OUTPUT / "leaderboard.xlsx")
print("ESP32 model bundles:", OUTPUT / "esp32_candidates.zip")
print("Progress and complete logs:", OUTPUT)
if SUMMARY.get("report_error"):
    print("Results are saved; report needs retry:", SUMMARY["report_error"])
else:
    if (OUTPUT / "comparison.png").exists():
        display(Image(filename=str(OUTPUT / "comparison.png")))
    if (OUTPUT / "campaign_report.html").exists():
        report = (OUTPUT / "campaign_report.html").read_text()
        # The chart is displayed separately so relative HTML image paths are unnecessary.
        display(HTML(report.replace('<img src="comparison.png" alt="Validation F1 versus exported model size">', '')))
'''
    notebook = nbf.v4.new_notebook()
    notebook.metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                             language_info=dict(name='python'), colab=dict(name='TreeVibes_All_In_One.ipynb'))
    notebook.cells = [
        nbf.v4.new_markdown_cell('# TreeVibes — all experiments in one notebook\n\n'
            'Select your **Colab T4** kernel and choose **Run All**. Authorize Google Drive when prompted. '
            'This file contains the complete project: no ZIP upload, notebook switching, repair cells, '
            'manual teacher paths or kernel restarts. Existing Kaggle downloads and completed runs are reused. '
            'A full campaign spans 131 recipes and may require multiple Colab sessions; rerun this same file to resume. '
            'To recover the preceding notebook after MLflow/plotting errors, keep the same settings and Drive output folder. '
            'Verified completed runs and feature caches are retained; failed recipes are retried.'),
        nbf.v4.new_markdown_cell('## 1. Settings\n\n'
            'Defaults run the full study. `MODE="screening"` provides a shorter run. '
            'Keep final-test evaluation disabled while searching; enable it here when ready to lock finalists. '
            'Remote inputs stay **30 × 13 MFCCs**. Larger local inputs are separate experiments.'),
        nbf.v4.new_code_cell(settings),
        nbf.v4.new_markdown_cell('## 2. Automatic setup\n\n'
            'The embedded source is loaded below. TensorFlow and PyTorch receive separate Python/CUDA '
            'environments using Python 3.12, even when the notebook kernel uses Python 3.13 or newer. '
            'The notebook kernel stays untouched. Installation logs and exact resolved '
            'dependency locks are saved to Drive. Both frameworks must pass GPU convolution and gradient checks.'),
        nbf.v4.new_code_cell(embedded, metadata={'jupyter':{'source_hidden':True}, 'cellView':'form'}),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell('## 3. Prepare data, train, distill and quantize\n\n'
            'Byte-identical recordings with opposing labels are excluded and audited; source WAV files stay unchanged. '
            'Preprocessing reports progress. The workflow trains supervised recipes first, selects a compatible teacher '
            'using validation data, runs distillation, then compares float, full INT8 PTQ and QAT exports. '
            'Each fit/conversion runs in a fresh process. Failures are recorded and compatible recipes continue. '
            'Checkpoints, the Excel tracker and the campaign journal are saved incrementally.'),
        nbf.v4.new_code_cell(run),
        nbf.v4.new_markdown_cell('## 4. Results and ESP32 candidates\n\n'
            'Validation comparisons and exported model sizes appear below; all runs and failures are in the Excel tracker. '
            'The ESP32-WROOM-32 budget is **2 MB total flash and approximately 0.5 MB RAM**. '
            'Board inference, full application memory, latency and waveform-to-MFCC parity still require hardware. '
            'Without source metadata, file-level splits do not prove performance on unseen trees.'),
        nbf.v4.new_code_cell(results)]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    return destination
