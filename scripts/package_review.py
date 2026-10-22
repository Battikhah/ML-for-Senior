"""Package a restart-safe notebook for the independent Windows professor-review campaign."""
from pathlib import Path
import json
import zipfile
import nbformat as nbf


def build(root):
    """Embed only new review tools; use the existing GPU and CPU environments unchanged."""
    names=['review_campaign.py','review_torch.py','review_results.py','review_compressed.py','review_robustness.py']
    sources={name:(root/'scripts'/name).read_text() for name in names}
    setup='''from pathlib import Path
import json, os, sys, subprocess
WORK = Path(os.environ.get("TREEVIBES_WINDOWS_WORK", "D:/ML/TreeVibesWindows")).resolve()
OUTPUT = WORK / "final_review_2026-10-20"
ROOT = WORK / "project"
TF_PYTHON = WORK / "runtimes/tf216-cpu/Scripts/python.exe"
GPU_PYTHON = WORK / "runtimes/torch271-cu128/Scripts/python.exe"
DATASET_ROOT = None  # Reuse/download the Kaggle WAV dataset automatically.
ADDITIONAL_FEATURES = True
assert TF_PYTHON.is_file() and GPU_PYTHON.is_file(), "Use the existing Windows training setup."
original = json.loads((WORK / "experiments/campaign_summary.json").read_text())
assert original["state"] in {"completed", "completed_with_failures"}, "Do not interrupt the original campaign."
OUTPUT.mkdir(parents=True, exist_ok=True)
for name in ["tools/scripts", "configs", "checkpoints", "plots", "logs", "audit", "cache"]:
    (OUTPUT / name).mkdir(parents=True, exist_ok=True)
settings = dict(original=str(WORK/"experiments"), output=str(OUTPUT), project=str(ROOT),
    tools_parent=str(OUTPUT/"tools"), tf_python=str(TF_PYTHON), torch_python=str(GPU_PYTHON),
    dataset_root=DATASET_ROOT, additional_features=ADDITIONAL_FEATURES)
progress_path = OUTPUT / "progress.json"
progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
ACTIVE = False
controller = OUTPUT / "controller.json"
if controller.exists() and progress.get("state") == "running":
    pid = json.loads(controller.read_text())["pid"]
    probe = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True)
    ACTIVE = f'"{pid}"' in probe.stdout
print("Existing campaign:", original["completed_runs"], "/", original["planned_runs"])
print("Review:", progress, "controller active:", ACTIVE)
'''
    install=f'sources = {sources!r}\n'+'''if not ACTIVE:
    (OUTPUT / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    package = OUTPUT / "tools/scripts"
    (package / "__init__.py").write_text("from pkgutil import extend_path\\n__path__=extend_path(__path__,__name__)\\n")
    for name, source in sources.items():
        (package / name).write_text(source, encoding="utf-8")
ENV = os.environ.copy()
ENV.update(PYTHONPATH=os.pathsep.join([str(OUTPUT/"tools"), str(ROOT), str(OUTPUT/"dependencies")]),
    PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg", TF_USE_LEGACY_KERAS="1",
    TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1", HF_HOME=str(OUTPUT/"pretrained_cache"))
sys.path.insert(0, str(ROOT))
from scripts.windows_support import run_logged
if not ACTIVE:
    extras = OUTPUT / "dependencies"
    probe = subprocess.run([str(GPU_PYTHON), "-c", "import torch,torchvision,timm,huggingface_hub,safetensors; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, timm.__version__)"], env=ENV, capture_output=True, text=True)
    if probe.returncode:
        run_logged([str(GPU_PYTHON), "-m", "pip", "install", "--target", str(extras), "--no-deps",
            "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cu128"], OUTPUT/"logs/torchvision_install.log", stream=True)
        run_logged([str(GPU_PYTHON), "-m", "pip", "install", "--target", str(extras), "--no-deps",
            "timm==1.0.15", "huggingface-hub==0.30.2", "safetensors==0.5.3"], OUTPUT/"logs/review_dependencies.log", stream=True)
print("Review tools ready; original project files unchanged.")
'''
    launch='''if ACTIVE:
    print("Attached to the running review. No duplicate worker was started.")
elif progress.get("state") in {"completed", "completed_with_failures"}:
    print("Review training already finished; continue to the report cell.")
else:
    log = (OUTPUT / "campaign.log").open("a", encoding="utf-8")
    process = subprocess.Popen([str(TF_PYTHON), str(OUTPUT/"tools/scripts/review_campaign.py"),
        "--settings", str(OUTPUT/"settings.json")], cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    (OUTPUT/"controller.json").write_text(json.dumps(dict(pid=process.pid)))
    log.close()
    print("Started review controller PID", process.pid)
print("Rerun the next cell to view progress. Training survives closing this notebook.")
'''
    monitor='''progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
print(progress)
log = OUTPUT / "campaign.log"
if log.exists(): print(log.read_text(encoding="utf-8", errors="replace")[-5000:])
print("Saved results:", len(list((OUTPUT/"checkpoints").glob("*/result.json"))))
'''
    report='''progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
if progress.get("state") not in {"completed", "completed_with_failures"}:
    print("Training is still running. Rerun this cell after completion.")
else:
    if not list((OUTPUT/"checkpoints").glob("previous_*_pca16_codec.npz")):
        run_logged([str(TF_PYTHON), str(OUTPUT/"tools/scripts/review_results.py"),
            "--settings", str(OUTPUT/"settings.json")], OUTPUT/"logs/initial_report.log", env=ENV, cwd=ROOT)
    for tool, receipt in [("review_compressed.py", "checkpoints/server_pca16_histogram/result.json"),
                          ("review_robustness.py", "volume_check.json")]:
        if not (OUTPUT/receipt).exists():
            run_logged([str(TF_PYTHON), str(OUTPUT/"tools/scripts"/tool), "--settings", str(OUTPUT/"settings.json")],
                OUTPUT/"logs"/(tool+".log"), env=ENV, cwd=ROOT)
    run_logged([str(TF_PYTHON), str(OUTPUT/"tools/scripts/review_results.py"),
        "--settings", str(OUTPUT/"settings.json")], OUTPUT/"logs/report.log", env=ENV, cwd=ROOT)
    from IPython.display import Markdown, display, Image
    display(Markdown((OUTPUT/"final_conclusions.md").read_text(encoding="utf-8")))
    display(Image(filename=str(OUTPUT/"plots/finalist_confusion.png")))
    print("Tables, source predictions, slide_data.json and plots:", OUTPUT)
    print("Editable PowerPoint/PDF are exported from these measured outputs by the accompanying deck builder.")
'''
    cells=[nbf.v4.new_markdown_cell('# TreeVibes final review — 20 October 2026\n\n'
        'Independent GPU experiments; original 131-recipe campaign remains unchanged. '
        'RPW is the positive class and healthy the negative class. Validation selection only; final test remains held out. '
        'Run setup once, then monitor the detached controller. Existing completed review runs are reused.'),
        nbf.v4.new_markdown_cell('## 1. Locate the completed campaign'),nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell('## 2. Prepare isolated review tools\n\nUses the existing Torch CUDA and TensorFlow CPU workers; new packages go into the review folder.'),nbf.v4.new_code_cell(install),
        nbf.v4.new_markdown_cell('## 3. Start or attach\n\nTwo optimizer continuations → core feature screens → unrestricted InceptionResNet-V2 → remaining screens → two larger feature trials.'),nbf.v4.new_code_cell(launch),
        nbf.v4.new_markdown_cell('## 4. Monitor durable checkpoints'),nbf.v4.new_code_cell(monitor),
        nbf.v4.new_markdown_cell('## 5. Measure deployment routes and prepare slide evidence\n\nRun after training finishes. Reports preserve exact RPW misses and healthy false alarms. A 24-byte local daily report is proposed; server features are required for every measurement. ESP32 board memory/latency remain unmeasured.'),nbf.v4.new_code_cell(report)]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    destination=root/'TreeVibes_Final_Professor_Review.ipynb';nbf.write(nb,destination);nbf.validate(nb)
    release=root/'artifacts/release';release.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(release/'treevibes_final_review_tools.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination,destination.name)
        for name in names+['package_review.py','build_professor_review_verified_local.mjs','finalize_professor_review_verified_local.mjs','build_pca_encoder.py']:
            archive.write(root/'scripts'/name,'scripts/'+name)
        archive.write(root/'docs/final-review-campaign.md','docs/final-review-campaign.md')
    return destination


if __name__=='__main__':print(build(Path.cwd()))
