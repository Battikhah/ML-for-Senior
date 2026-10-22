"""Run the full TreeVibes study from one notebook with isolated, resumable workers."""
import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import zipfile
from .config import load_config
from .data import DEFAULT_FEATURES, fingerprint, load_dataset, prepare_dataset, write_json
from .runtime import run_logged, worker_environment
from .tracking import export_leaderboard, file_hash, source_fingerprint, compatible_sources, valid_completed

STAGES = ['baseline', 'quality', 'compact', 'ablation', 'local_features', 'repeat', 'distillation']


def campaign_plan(catalog: Path, mode: str) -> list[dict]:
    """Retain the complete catalog while making a short end-to-end check selectable."""
    presets = {
        'smoke': {'reference_a', 'reference_b', 'logistic_coefficient', 'mlp_w16',
                  'tiny_mlp_w4', 'kd_mlp_w4_t2_a25'},
        'screening': {'logistic_coefficient', 'cnn_w16', 'tiny_depthwise_w8', 'kd_depthwise_w8_t2_a25'}}
    if mode not in {'full', *presets}:
        raise ValueError('mode must be full, screening or smoke')
    plan = [dict(path=str(p.resolve()), config=load_config(p)) for p in Path(catalog).glob('*/*.json')]
    if mode != 'full':
        plan = [entry for entry in plan if entry['config']['id'] in presets[mode]]
    if not plan:
        raise ValueError('The packaged experiment catalog is missing')
    return sorted(plan, key=lambda e: (STAGES.index(e['config']['suite']), e['config']['id']))


def _job(request: dict, output: Path, python: str) -> dict:
    """Release all framework allocations after one fit or conversion, retaining full logs."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    directory = output/'campaign_jobs'/f'{request["action"]}_{stamp}'
    directory.mkdir(parents=True)
    request = request | {'reply':str(directory/'reply.json')}
    path = directory/'request.json'
    write_json(path, request)
    run_logged([str(python), '-m', 'treevibes.campaign', '--job', str(path)],
               directory/'worker.log', env=worker_environment(),
               stream=request['action']=='train', prefix='TV_PROGRESS ')
    return json.loads((directory/'reply.json').read_text())


def _execute_job(path: Path) -> None:
    """Use the selected interpreter so PyTorch never shares TensorFlow's CUDA process."""
    request = json.loads(Path(path).read_text())
    if request['action'] == 'train':
        from .tracking import run_experiment
        result = run_experiment(Path(request['config']), Path(request['cache']), Path(request['output']),
                                teacher_dir=Path(request['teacher']) if request.get('teacher') else None)
    elif request['action'] == 'export':
        from .export import export_model, create_firmware_bundle
        result = export_model(Path(request['run']), load_dataset(Path(request['cache'])),
                              request['method'], Path(request['destination']), request['qat_epochs'])
        if request['method'] != 'float' and not result['unresolved_resolver_ops']:
            create_firmware_bundle(Path(request['destination']), Path(request['destination'])/'firmware_bundle')
    elif request['action'] == 'final_test':
        from .finalize import evaluate_final
        result = evaluate_final(Path(request['lock']), load_dataset(Path(request['cache'])),
                                Path(request['destination']))
    else:
        raise ValueError('Unknown campaign worker action')
    write_json(Path(request['reply']), result)


def _rank(result: dict) -> tuple:
    """Use validation evidence only, with stable ties favoring fewer parameters."""
    metrics = result['validation']
    return (-metrics['f1'], -(metrics.get('pr_auc') or 0), -metrics['recall'],
            result.get('parameters') or result.get('bytes') or 0, result.get('run_id', ''))


def _raw_dataset(settings: dict) -> Path:
    """Fetch the fixed dataset revision only when a feature cache still needs raw audio."""
    if settings.get('dataset_root'):
        return Path(settings['dataset_root'])
    import kagglehub
    print('Locating TreeVibes v1; reusing its runtime download when available.', flush=True)
    return Path(kagglehub.dataset_download('potamitis/treevibes/versions/1'))


def _diverse(runs: list[dict], top_n: int) -> list[dict]:
    """Compare several student families before filling remaining slots by validation rank."""
    ranked = sorted(runs, key=_rank)
    chosen, families = [], set()
    for run in ranked:
        family = json.loads((Path(run['run_dir'])/'config.json').read_text())['family']
        if family not in families and len(chosen) < top_n:
            chosen.append(run)
            families.add(family)
    for run in ranked:
        if len(chosen) < top_n and run not in chosen:
            chosen.append(run)
    return chosen


def _report(summary: dict, output: Path, project: Path) -> None:
    """Publish comparisons and board candidates without claiming unmeasured ESP32 fit."""
    # Matplotlib validates MPLBACKEND during import, before use() can correct it.
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    export_leaderboard(output, output/'leaderboard.xlsx')
    figure, ax = plt.subplots(figsize=(9, 5))
    for method, marker in [('float', 'o'), ('ptq', 's'), ('qat', '^')]:
        rows = [e for e in summary['exports'] if e['method'] == method
                and e['dataset_id'] == summary['primary_dataset_id']]
        if rows:
            ax.scatter([e['bytes']/1024 for e in rows], [e['validation']['f1'] for e in rows],
                       marker=marker, label=method.upper(), alpha=.85)
    ax.set(xlabel='Exported model size (KiB)', ylabel='Validation F1',
           title=('Synthetic workflow check' if summary['synthetic'] else 'TreeVibes model compression'))
    ax.grid(alpha=.2)
    if summary['exports']:
        ax.legend()
    figure.tight_layout()
    figure.savefig(output/'comparison.png', dpi=150)
    plt.close(figure)
    rows = ''.join('<tr><td>'+html.escape(r['config_id'])+'</td><td>'+f'{r["validation"]["f1"]:.3f}'+
                   '</td><td>'+str(r.get('parameters', '—'))+'</td></tr>'
                   for r in sorted(summary['runs'], key=_rank)[:25])
    failures = ''.join('<li>'+html.escape(f['stage']+': '+f['id']+' — '+f['error'][-400:])+'</li>'
                       for f in summary['failures'])
    (output/'campaign_report.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>TreeVibes results</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}'
        'img{max-width:100%}</style><h1>TreeVibes experiment results</h1>'
        f'<p>Status: <b>{html.escape(summary["state"])}</b>. '
        f'{summary["completed_runs"]}/{summary["planned_runs"]} training recipes completed; '
        f'{len(summary["exports"])} exports; {len(summary["failures"])} recorded failures.</p>'
        + ('<p><b>Synthetic check: these scores do not measure infestation detection.</b></p>' if summary['synthetic'] else '')
        + '<p>Primary comparisons and teacher selection use the same 30 × 13 feature cache. '
        'Other feature/gate cohorts remain separate experiments in the workbook. '
        'Without tree/session metadata, results establish file-level performance only.</p>'
        '<img src="comparison.png" alt="Validation F1 versus exported model size">'
        '<h2>Validation results</h2><p>Entries from differing feature cohorts are exploratory comparisons.</p>'
        '<table><tr><th>Recipe</th><th>F1</th><th>Parameters</th></tr>'+rows+'</table>'
        '<h2>ESP32-WROOM-32</h2><p>Budget: 2 MB total flash and approximately 0.5 MB RAM. '
        'Model sizes are measured; complete firmware size, tensor arena, latency and embedded MFCC parity '
        'require target hardware. Host exports do not certify board fit.</p>'
        '<h2>Recorded failures</h2><ul>'+failures+'</ul>')
    bundles = [Path(e['directory'])/'firmware_bundle' for e in summary['exports']
               if (Path(e['directory'])/'firmware_bundle').is_dir()]
    with zipfile.ZipFile(output/'esp32_candidates.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for bundle in bundles:
            for path in bundle.iterdir():
                if path.is_file():
                    archive.write(path, f'candidates/{bundle.parent.parent.name}_{bundle.parent.name}/{path.name}')
        for path in (project/'deployment/esp32').rglob('*'):
            if path.is_file() and 'generated' not in path.parts:
                archive.write(path, 'esp32_harness/'+path.relative_to(project/'deployment/esp32').as_posix())


def run_campaign(settings: dict, project: Path | None = None) -> dict:
    """Prepare, search, distill and compress in order, resuming every completed artifact."""
    project = Path(project or Path(__file__).parent.parent).resolve()
    output = Path(settings['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode = settings.get('mode', 'full')
    epochs = settings.get('epochs', 60)
    top_n = settings.get('export_top_n', 3)
    qat_epochs = settings.get('qat_epochs', 3)
    if any(type(n) is not int or n < 1 for n in [epochs, top_n, qat_epochs]):
        raise ValueError('epochs, export_top_n and qat_epochs must be positive integers')
    plan = campaign_plan(project/'experiments', mode)
    # A final-test toggle does not invalidate the completed search or its checkpoints.
    campaign_contract = dict(settings={k:v for k,v in settings.items() if k!='final_test'},
                             groups_hash=file_hash(Path(settings['groups_csv'])) if settings.get('groups_csv') else None)
    identity = fingerprint(campaign_contract | dict(source=source_fingerprint()))[:16]
    progress = output/'campaigns'/identity/'progress.json'
    previous = json.loads(progress.read_text()) if progress.exists() else {}
    if not previous:
        for source in compatible_sources()[1:]:
            prior_id = fingerprint(campaign_contract | dict(source=source))[:16]
            prior = output/'campaigns'/prior_id/'progress.json'
            if prior.exists():
                previous = json.loads(prior.read_text())
                print('Recovering the preceding notebook campaign; completed artifacts stay in place.', flush=True)
                break
    previous_runs = {r['config_id']:r for r in previous.get('runs', [])}
    summary = dict(state='preparing', campaign_id=identity, synthetic=mode=='smoke',
                   planned_runs=len(plan), completed_runs=0, runs=[], exports=[], failures=[],
                   teacher=None, final_results=None, caches={})
    summary['recovered_campaign_id'] = previous.get('campaign_id')
    write_json(progress, summary)
    caches = {}
    # These snapshots belong to this exact source/settings identity and verify payload hashes.
    if mode != 'smoke' and not settings.get('dataset_root'):
        for dataset_id, path in previous.get('caches', {}).items():
            try:
                data = load_dataset(Path(path))
                if data['contract']['dataset_id']==dataset_id and not data['contract'].get('synthetic'):
                    caches[fingerprint(data['contract']['features'])] = Path(path)
            except (OSError, ValueError, KeyError):
                continue
    if mode == 'smoke':
        from .synthetic import save_synthetic
        default_cache = output/'synthetic_cache'
        if not (default_cache/'contract.json').exists():
            save_synthetic(default_cache, 40)
        dataset_root = None
    else:
        dataset_root = settings.get('dataset_root')
        default_cache = caches.get(fingerprint(DEFAULT_FEATURES))
        if default_cache is None:
            dataset_root = _raw_dataset(settings)
            print('Preparing the shared 30 × 13 feature cache...', flush=True)
            default_cache = prepare_dataset(dataset_root, output/'cache', DEFAULT_FEATURES,
                                            Path(settings['groups_csv']) if settings.get('groups_csv') else None)
        else:
            print('Reusing the verified feature cache; no raw download required.', flush=True)
    default_data = load_dataset(default_cache)
    default_id = default_data['contract']['dataset_id']
    summary['primary_dataset_id'] = default_id
    caches[fingerprint(DEFAULT_FEATURES)] = default_cache
    summary['data_audit'] = default_data['contract'].get('data_audit')
    write_json(output/'active_cache.json', {'path':str(default_cache)})
    metadata = {}
    for index, entry in enumerate(plan, 1):
        config = entry['config'] | {'epochs':epochs}
        config['patience'] = min(config['patience'], epochs)
        key = fingerprint(config['features'])
        print(f'[{index}/{len(plan)}] {config["id"]}', flush=True)
        try:
            if key not in caches:
                if dataset_root is None:
                    dataset_root = _raw_dataset(settings)
                caches[key] = prepare_dataset(dataset_root, output/'cache', config['features'],
                                              Path(settings['groups_csv']) if settings.get('groups_csv') else None)
            cache = caches[key]
            data_id = load_dataset(cache)['contract']['dataset_id']
            summary['caches'][data_id] = str(cache)
            teacher = None
            if config.get('method') == 'kd':
                teachers = [r for r in summary['runs'] if r['dataset_id'] == data_id
                            and metadata[r['config_id']]['normalization'] == config['normalization']
                            and (Path(r['run_dir'])/'best.keras').is_file()
                            and metadata[r['config_id']].get('method') != 'kd']
                if not teachers:
                    raise ValueError('No completed teacher matches this feature/normalization contract')
                teacher = Path(min(teachers, key=_rank)['run_dir'])
                summary['teacher'] = str(teacher)
            resolved = output/'resolved_configs'/f'{config["id"]}.json'
            write_json(resolved, config)
            old = previous_runs.get(config['id'])
            if (old and old['dataset_id'] == data_id and valid_completed(Path(old['run_dir']))
                    and json.loads((Path(old['run_dir'])/'config.json').read_text()) == config
                    and json.loads((Path(old['run_dir'])/'provenance.json').read_text()).get('teacher_model_hash')
                    == (file_hash(teacher/'best.keras') if teacher else None)):
                result = old
                print('  Reusing verified completed run.', flush=True)
            else:
                python = settings.get('reference_python', sys.executable) if config['family'].startswith('reference_') else sys.executable
                result = _job(dict(action='train', config=str(resolved), cache=str(cache), output=str(output),
                                   teacher=str(teacher) if teacher else None), output, python)
            summary['runs'].append(result)
            metadata[config['id']] = config
            summary['completed_runs'] += 1
            print(f'  validation F1={result["validation"]["f1"]:.4f}', flush=True)
        except Exception as error:
            summary['failures'].append(dict(stage='train', id=config['id'], error=str(error)))
            print(f'  recorded failure: {str(error)[-300:]}', flush=True)
        summary['state'] = 'training'
        write_json(progress, summary)
        export_leaderboard(output, output/'leaderboard.xlsx')
    keras = [r for r in summary['runs'] if (Path(r['run_dir'])/'best.keras').is_file()]
    primary = [r for r in keras if r['dataset_id'] == default_id]
    direct = [r for r in primary if metadata[r['config_id']]['suite'] == 'compact']
    distilled = [r for r in primary if metadata[r['config_id']].get('method') == 'kd']
    quality = [r for r in primary if metadata[r['config_id']].get('method') != 'kd']
    candidates = _diverse(direct, top_n) + _diverse(distilled, top_n)
    if quality:
        candidates += [min(quality, key=_rank)]
    # Richer local features are separate cohorts; include their best model independently.
    for dataset_id in sorted({r['dataset_id'] for r in keras if metadata[r['config_id']]['suite']=='local_features'}):
        candidates += [min([r for r in keras if r['dataset_id']==dataset_id], key=_rank)]
    candidates = list({r['run_id']:r for r in candidates}.values())
    for candidate in candidates:
        for method in ['float', 'ptq', 'qat']:
            export_id = candidate['run_id']+'_'+method+'_'+fingerprint(dict(qat_epochs=qat_epochs if method=='qat' else 0))[:8]
            base = output/'exports'/export_id
            attempts = sorted(base.glob('attempt_*'))
            try:
                existing = None
                for attempt in reversed(attempts):
                    if (attempt/'export.json').is_file() and (attempt/'model.tflite').is_file():
                        try:
                            meta = json.loads((attempt/'export.json').read_text())
                        except ValueError:
                            continue
                        if (meta['model_sha256']==file_hash(attempt/'model.tflite') and
                                meta['parent_model_sha256']==file_hash(Path(candidate['run_dir'])/'best.keras')):
                            existing = meta | {'directory':str(attempt)}
                            break
                if existing:
                    result = existing
                else:
                    number = max([int(p.name.rsplit('_', 1)[-1]) for p in attempts]+[0])+1
                    destination = base/f'attempt_{number:03}'
                    print(f'Export {candidate["config_id"]}: {method}', flush=True)
                    result = _job(dict(action='export', run=candidate['run_dir'], method=method,
                                       cache=summary['caches'][candidate['dataset_id']], destination=str(destination),
                                       qat_epochs=qat_epochs), output, sys.executable) | {'directory':str(destination)}
                summary['exports'].append(result)
            except Exception as error:
                summary['failures'].append(dict(stage='export', id=export_id, error=str(error)))
                print(f'  recorded export failure: {str(error)[-300:]}', flush=True)
            summary['state'] = 'exporting'
            write_json(progress, summary)
    if settings.get('final_test'):
        from .finalize import lock_finalists, evaluate_final
        finalists = [r for r in summary['runs'] if r['dataset_id']==default_id]
        exports = [e for e in summary['exports'] if e['dataset_id']==default_id and e['method']!='float']
        summary['final_results'] = []
        # Each lock contains one framework so a PyTorch winner can use its own worker.
        for kind, entries in [('run', finalists), ('export', exports)]:
            if not entries:
                continue
            try:
                winner = min(entries, key=_rank)
                directory = Path(winner['run_dir'] if kind=='run' else winner['directory'])
                lock = progress.parent/f'finalists_{kind}.json'
                destination = progress.parent/f'final_test_{kind}'
                if not lock.exists():
                    lock_finalists([directory] if kind=='run' else [], [directory] if kind=='export' else [], lock)
                if (destination/'final_results.json').exists():
                    result = evaluate_final(lock, default_data, destination)
                else:
                    reference = kind=='run' and metadata[winner['config_id']]['family'].startswith('reference_')
                    python = settings.get('reference_python', sys.executable) if reference else sys.executable
                    result = _job(dict(action='final_test', lock=str(lock), cache=str(default_cache),
                                       destination=str(destination)), output, python)
                summary['final_results'].append(result)
            except Exception as error:
                summary['failures'].append(dict(stage='final_test', id=kind, error=str(error)))
    summary['state'] = 'completed_with_failures' if summary['failures'] else 'completed'
    if not summary['runs']:
        summary['state'] = 'failed'
    # Commit results before plotting so a report dependency cannot hide completed work.
    write_json(progress, summary)
    write_json(output/'campaign_summary.json', summary)
    try:
        _report(summary, output, project)
    except Exception as error:
        summary['report_error'] = str(error)
        print(f'Results saved; report needs retry: {error}', flush=True)
        write_json(progress, summary)
        write_json(output/'campaign_summary.json', summary)
    print(f'{summary["state"]}: {summary["completed_runs"]}/{len(plan)} training recipes; '
          f'{len(summary["exports"])} exports; {len(summary["failures"])} failures. '
          f'Report: {output / "leaderboard.xlsx"}', flush=True)
    return summary


def main(argv=None) -> int:
    """Support the notebook controller and its private per-framework worker jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--settings', type=Path)
    group.add_argument('--job', type=Path)
    args = parser.parse_args(argv)
    if args.job:
        _execute_job(args.job)
    else:
        run_campaign(json.loads(args.settings.read_text()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
