"""Package CUDA training for unfinished models while preserving the existing Windows setup."""
from pathlib import Path
import tempfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from treevibes.packaging import build_colab_archive
from scripts.package_windows import build_windows_notebook


def build_gpu_notebook(root,archive,destination):
    """Change the training controller while retaining tested download, migration and exports."""
    import nbformat as nbf
    with tempfile.TemporaryDirectory() as temporary:
        template=build_windows_notebook(root,archive,Path(temporary)/'windows.ipynb')
        notebook=nbf.read(template,as_version=4)
    notebook.cells[0].source=(
        '# TreeVibes — Windows CUDA training and Colab recovery\n\n'
        'Select your brother\'s existing **Windows Python 3.12 kernel** and **Run All**. '
        'No reboot or WSL is required. Remaining neural fits and distillation use **PyTorch CUDA** '
        'on the RTX 5060 Ti; completed TensorFlow results are reused. '
        'Trusted Keras weights are converted through an audited layer graph, with numerical '
        'inference checks before training and before saving the Keras export.\n\n'
        '**Framework migration:** committed Keras epochs, history and weights are retained. '
        'Adam state resets once when switching to PyTorch; subsequent PyTorch optimizer states '
        'resume normally. Dropout/numerical trajectories can differ. Backend fingerprints and '
        '`backend.json` identify converted runs. Original models and Drive files remain untouched.\n\n'
        '**CPU stages:** feature preparation, weight serialization, TFLite PTQ/conversion and '
        'the existing short TensorFlow QAT export stage still run on CPU. '
        'The full supervised/student search uses CUDA. First-time dependency installation is '
        'reused from the standard Windows notebook. Default storage remains '
        '`D:/ML/TreeVibesWindows`; keep the PC awake.\n\n'
        'The public experiment ZIP downloads once and is safely extracted before recovery. '
        'Share the ZIP as **Anyone with the link → Viewer**; '
        'the notebook does not require a browser login. After download succeeds, access can be '
        'restricted again. Completed local downloads and experiments are reused.')
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'DRIVE_FOLDER_ID =' in cell.source:
            cell.source = '\n'.join(line for line in cell.source.splitlines()
                if not line.startswith(('DRIVE_FOLDER_ID =', 'DRIVE_ACCESS =')))
            cell.source += '\nDRIVE_ZIP_ID = "1MsF2OEAHqL4uOhhZ8nJEiFCEMVoEn1ZT"\n'
        if cell.cell_type == 'code' and cell.source.startswith('from scripts.windows_support import download_drive'):
            cell.source = """from scripts.zip_recovery import download_zip

if SMOKE:
    SOURCE = None
elif SOURCE_FOLDER:
    SOURCE = Path(SOURCE_FOLDER).expanduser().resolve()
else:
    SOURCE = download_zip(WORK, DRIVE_ZIP_ID)
if SOURCE is not None and not (SOURCE / "notebook_settings.json").is_file():
    raise FileNotFoundError("ZIP must contain the complete TreeVibesExperiments folder.")
print("Original snapshot:", SOURCE)
"""
        if cell.cell_type == 'markdown' and cell.source.startswith('## 3. Download'):
            cell.source = ('## 3. Download and extract the experiment ZIP\n\n'
                'Downloads the shared ZIP without login, validates extraction paths and CRCs, '
                'and locates the experiments folder automatically. Completed downloads/extractions '
                'are reused. Original ZIP contents remain separate from the working experiments. '
                'Keep the PC awake; do not delete the Python/Jupyter environment.')
    for cell in notebook.cells:
        if cell.cell_type=='code' and cell.source.startswith('SETTINGS ='):
            cell.source=cell.source.replace('environment = worker_environment(ROOT)',
                'environment = worker_environment(ROOT)\n'
                'environment["TREEVIBES_TF_PYTHON"] = WORKERS["tensorflow"]\n'
                'if SMOKE:\n'
                '    environment["TREEVIBES_ALLOW_CPU_TEST"] = "1"')
            cell.source=cell.source.replace('"-m", "treevibes.campaign"','"-m", "scripts.gpu_campaign"')
        if cell.cell_type=='markdown' and cell.source.startswith('## 5.'):
            cell.source+='\n\nUnfinished neural fits now run on CUDA, including converted teachers for distillation. '
            cell.source+='CUDA is required for real training; CPU fallback is allowed only in explicit synthetic verification.'
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,destination)
    return destination


if __name__=='__main__':
    root=Path.cwd();archive=build_colab_archive(root,root/'artifacts/release/treevibes_windows_gpu.zip')
    notebook=build_gpu_notebook(root,archive,root/'TreeVibes_Windows_GPU_All_In_One.ipynb')
    (root/'artifacts/release/TreeVibes_Windows_GPU_All_In_One.ipynb').write_bytes(notebook.read_bytes())
    print(notebook)
