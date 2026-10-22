"""Exercise the single-notebook campaign through real training and export workers."""
import json
from pathlib import Path


def test_full_plan_orders_teacher_search_before_all_distillation_recipes():
    """The consolidated notebook must retain the complete approved experiment catalog."""
    from treevibes.campaign import campaign_plan
    plan = campaign_plan(Path('experiments'), 'full')
    assert len(plan) == 131
    assert sum(c['config'].get('method') == 'kd' for c in plan) == 24
    assert all(c['config'].get('method') != 'kd' for c in plan[:107])
    assert all(c['config'].get('method') == 'kd' for c in plan[107:])
    assert {c['config']['features']['frames'] for c in plan} == {30, 60}


def test_campaign_runs_all_stages_and_resumes_without_retraining(tmp_path, monkeypatch):
    """One call must train references, select a teacher, distill, quantize and report."""
    from treevibes.campaign import run_campaign
    monkeypatch.setenv('MPLBACKEND','module://matplotlib_inline.backend_inline')
    settings = dict(mode='smoke', output=str(tmp_path/'out'), epochs=1,
                    export_top_n=1, qat_epochs=1, final_test=True)
    summary = run_campaign(settings, Path.cwd())
    assert summary['state'] == 'completed', summary.get('failures')
    assert summary['completed_runs'] == 6
    assert summary['synthetic'] is True
    assert {e['method'] for e in summary['exports']} == {'float', 'ptq', 'qat'}
    assert summary['teacher'] and summary['final_results']
    assert (tmp_path/'out'/'leaderboard.xlsx').is_file()
    assert (tmp_path/'out'/'comparison.png').is_file()
    assert (tmp_path/'out'/'esp32_candidates.zip').is_file()
    assert (tmp_path/'out'/'campaign_report.html').is_file()
    kd = next(r for r in summary['runs'] if r['config_id'].startswith('kd_'))
    provenance = json.loads((Path(kd['run_dir'])/'provenance.json').read_text())
    assert provenance['teacher_run_dir'] == summary['teacher']
    attempts = sorted((tmp_path/'out'/'runs').glob('*/*/attempt_*'))
    logs = sorted((tmp_path/'out'/'campaign_jobs').rglob('*.log'))
    again = run_campaign(settings, Path.cwd())
    assert again['state'] == 'completed'
    assert sorted((tmp_path/'out'/'runs').glob('*/*/attempt_*')) == attempts
    assert sorted((tmp_path/'out'/'campaign_jobs').rglob('*.log')) == logs
    assert again['exports'] == summary['exports']


def test_real_cache_resume_does_not_download_raw_data_again(tmp_path, monkeypatch):
    """A new session with complete feature caches must not require the raw archive."""
    import sys
    import types
    import numpy as np
    from scipy.io import wavfile
    from treevibes.campaign import run_campaign
    import treevibes.campaign as campaign
    project=tmp_path/'project';(project/'experiments/baseline').mkdir(parents=True)
    (project/'experiments/baseline/control.json').write_text(json.dumps(dict(id='control',family='logistic',suite='baseline')))
    root=tmp_path/'audio';rng=np.random.default_rng(7)
    for label in ['clean','infested']:
        (root/label).mkdir(parents=True)
        for i in range(12):
            wavfile.write(root/label/f'{i}.wav',8000,rng.normal(0,.05,8000).astype('float32'))
    downloads=[]
    def download(handle):
        """Replace the external download with a real local WAV corpus."""
        downloads.append(handle)
        if len(downloads)>1:
            raise AssertionError('Downloaded despite a complete feature cache')
        return str(root)
    monkeypatch.setitem(sys.modules,'kagglehub',types.SimpleNamespace(dataset_download=download))
    settings=dict(mode='full',output=str(tmp_path/'out'),epochs=1,export_top_n=1,qat_epochs=1)
    with monkeypatch.context() as old:
        old.setattr(campaign,'source_fingerprint',lambda:'d6c6838950a09a4357c7194cf8b66e0dcc0634c76f547a93be9fc52d6bd0e8b8')
        first=run_campaign(settings,project)
    assert first['state']=='completed'
    attempts=sorted((tmp_path/'out/runs').glob('*/*/attempt_*'))
    assert run_campaign(settings,project)['state']=='completed'
    assert sorted((tmp_path/'out/runs').glob('*/*/attempt_*'))==attempts
    assert len(downloads)==1
    def broken_report(*args):
        """Represent a reporting crash after successful training."""
        raise ValueError('report dependency failed')
    monkeypatch.setattr(campaign,'_report',broken_report)
    summary=run_campaign(settings,project)
    saved=json.loads((tmp_path/'out/campaign_summary.json').read_text())
    assert saved['completed_runs']==1 and saved['report_error']=='report dependency failed'
    assert saved==summary
