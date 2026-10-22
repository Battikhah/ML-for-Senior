"""Keep completed study results while routing unfinished neural fits to PyTorch CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def adapter_source():
    """Give converted training a separate reproducibility identity from TensorFlow fits."""
    from treevibes.tracking import source_fingerprint
    from treevibes.data import fingerprint
    names=['torch_adapter.py','torch_bridge.py','gpu_campaign.py']
    return fingerprint(dict(scientific_source=source_fingerprint(),backend='torch_cuda_v1',
        adapter={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}))


def execute_training(request):
    """Use the existing artifact/MLflow runner with a CUDA-only training implementation."""
    import treevibes.train as training
    import treevibes.tracking as tracking
    from scripts.torch_adapter import train_cuda
    original_sources=tracking.compatible_sources()
    source=adapter_source()
    tracking.source_fingerprint=lambda:source
    tracking.compatible_sources=lambda:[source]+original_sources
    training._train_keras=train_cuda
    result=tracking.run_experiment(Path(request['config']),Path(request['cache']),Path(request['output']),
                                   teacher_dir=Path(request['teacher']) if request.get('teacher') else None)
    from treevibes.data import write_json
    write_json(Path(request['reply']),result)


def run(settings):
    """Preserve teacher selection, validation, exports and cached TensorFlow results."""
    import sys
    from datetime import datetime,timezone
    import treevibes.campaign as campaign
    from treevibes.config import load_config
    from treevibes.data import write_json
    from treevibes.runtime import worker_environment
    from scripts.windows_support import run_logged
    original=campaign._job
    def job(request,output,python):
        """Only unfinished non-classical neural fits switch interpreter/backend."""
        if request['action']!='train':
            return original(request,output,python)
        config=load_config(Path(request['config']))
        if config['family'] in {'logistic','svm','forest','histogram'} or config['family'].startswith('reference_'):
            return original(request,output,python)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        directory=Path(output)/'campaign_jobs'/f'torch_train_{stamp}';directory.mkdir(parents=True)
        request=request|dict(reply=str(directory/'reply.json'))
        path=directory/'request.json';write_json(path,request)
        run_logged([settings['reference_python'],'-m','scripts.gpu_campaign','--job',str(path)],
                   directory/'worker.log',env=worker_environment(),stream=True)
        return json.loads((directory/'reply.json').read_text())
    campaign._job=job
    return campaign.run_campaign(settings)


def main():
    """Run the controller under TensorFlow CPU and fit workers under PyTorch CUDA."""
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--settings',type=Path)
    group.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.job:
        execute_training(json.loads(args.job.read_text()))
    else:
        run(json.loads(args.settings.read_text()))


if __name__=='__main__':
    main()
