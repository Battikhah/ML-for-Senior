"""Command-line entry points shared by Colab notebooks and local workflows."""
import argparse
import json
from pathlib import Path
from .config import load_config
from .data import DEFAULT_FEATURES, load_dataset, prepare_dataset, write_json

SCREENING=['logistic_coefficient','cnn_w16','tiny_depthwise_w8']


def selected_configs(catalog: Path,suite: str,config_paths: list[str] | None=None) -> list[Path]:
    """Keep broad catalogs selectable without silently launching every recipe."""
    if config_paths:
        return [Path(path) for path in config_paths]
    paths=sorted(Path(catalog).glob('*/*.json'))
    if suite=='screening':
        paths=[p for p in paths if p.stem in SCREENING]
    elif suite!='all':
        paths=[p for p in paths if load_config(p)['suite']==suite]
    if not paths:
        raise ValueError('No selected configurations; check the catalog path and suite')
    return paths


def main(argv=None) -> int:
    """Expose explicit stages and require a conscious action before test-set access."""
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--dataset',type=Path)
    prepare.add_argument('--download',action='store_true');prepare.add_argument('--output',type=Path,default=Path('data/cache'))
    prepare.add_argument('--config',type=Path);prepare.add_argument('--groups',type=Path)
    synthetic=sub.add_parser('synthetic');synthetic.add_argument('--output',type=Path,required=True)
    synthetic.add_argument('--count',type=int,default=80)
    catalog=sub.add_parser('catalog');catalog.add_argument('--output',type=Path,default=Path('experiments'))
    train=sub.add_parser('train');train.add_argument('--suite',default='screening')
    train.add_argument('--catalog',type=Path,default=Path('experiments'));train.add_argument('--config',action='append')
    train.add_argument('--cache',type=Path);train.add_argument('--dataset',type=Path);train.add_argument('--groups',type=Path)
    train.add_argument('--cache-dir',type=Path,default=Path('data/cache'))
    train.add_argument('--output',type=Path,default=Path('artifacts'));train.add_argument('--teacher',type=Path)
    train.add_argument('--max-runs',type=int,default=3,help='0 selects all recipes in this suite')
    train.add_argument('--epochs',type=int);train.add_argument('--seeds',type=int,nargs='+')
    train.add_argument('--dry-run',action='store_true');train.add_argument('--no-resume',action='store_true')
    report=sub.add_parser('report');report.add_argument('--output',type=Path,default=Path('artifacts'))
    report.add_argument('--workbook',type=Path,default=Path('artifacts/leaderboard.xlsx'))
    export=sub.add_parser('export');export.add_argument('--run',type=Path,required=True)
    export.add_argument('--cache',type=Path,required=True);export.add_argument('--output',type=Path,required=True)
    export.add_argument('--method',choices=['float','ptq','qat'],default='ptq');export.add_argument('--qat-epochs',type=int,default=3)
    bundle=sub.add_parser('bundle');bundle.add_argument('--export',dest='export_dir',type=Path,required=True)
    bundle.add_argument('--output',type=Path,default=Path('deployment/esp32/main/generated'))
    lock=sub.add_parser('lock');lock.add_argument('--run',action='append',default=[],type=Path)
    lock.add_argument('--export',dest='exports',action='append',default=[],type=Path)
    lock.add_argument('--output',type=Path,required=True)
    final=sub.add_parser('evaluate');final.add_argument('--lock',type=Path,required=True)
    final.add_argument('--cache',type=Path,required=True);final.add_argument('--output',type=Path,required=True)
    robust=sub.add_parser('robustness');robust.add_argument('--run',type=Path,required=True)
    robust.add_argument('--cache',type=Path,required=True);robust.add_argument('--output',type=Path,required=True)
    board=sub.add_parser('board');board.add_argument('--measurement',type=Path,required=True)
    board.add_argument('--output',type=Path,default=Path('artifacts/board'))
    args=parser.parse_args(argv)
    if args.command=='prepare':
        root=args.dataset
        if args.download:
            import kagglehub
            root=Path(kagglehub.dataset_download('potamitis/treevibes'))
        if root is None:
            parser.error('prepare requires --dataset or the explicit --download flag')
        features=load_config(args.config)['features'] if args.config else DEFAULT_FEATURES
        print(prepare_dataset(root,args.output,features,args.groups))
    elif args.command=='synthetic':
        from .synthetic import save_synthetic
        print(save_synthetic(args.output,args.count))
    elif args.command=='catalog':
        from .catalog import generate_catalog
        print(f'Generated {len(generate_catalog(args.output))} configurations')
    elif args.command=='train':
        from .tracking import run_experiment,export_leaderboard
        paths=selected_configs(args.catalog,args.suite,args.config)
        if args.max_runs<0:
            parser.error('--max-runs must be nonnegative')
        resolved=[]
        for path in paths:
            config=load_config(path)
            for seed in args.seeds or [config['seed']]:
                current=config | {'seed':seed}
                if args.seeds:
                    current['id']+=f'_seed{seed}'
                if args.epochs is not None:
                    current['epochs']=args.epochs
                resolved.append((path,current))
        if args.max_runs:
            resolved=resolved[:args.max_runs]
        for _,c in resolved:
            print(f'{c["id"]}: {c["family"]}, {c["features"]["frames"]}x{c["features"]["coefficients"]}, '
                  f'{c["epochs"]} epochs, method={c.get("method","supervised")}, seed={c["seed"]}')
        if args.dry_run:
            return 0
        if args.cache is None and args.dataset is None:
            parser.error('train requires --cache or --dataset; use --dry-run to preview')
        failures=0
        for _,c in resolved:
            effective=args.output/'resolved_configs'/(c['id']+'.json')
            write_json(effective,c)
            try:
                cache=args.cache
                if cache is None or load_dataset(cache)['contract']['features']!=c['features']:
                    if args.dataset is None:
                        raise ValueError('Recipe needs a different feature cache; supply --dataset and --groups if available')
                    cache=prepare_dataset(args.dataset,args.cache_dir,c['features'],args.groups)
                result=run_experiment(effective,cache,args.output,not args.no_resume,args.teacher)
                print(json.dumps(dict(run_id=result['run_id'],run_dir=result['run_dir'],validation=result['validation'])))
            except Exception as error:
                failures+=1;print(f'FAILED {c["id"]}: {type(error).__name__}: {error}')
                from datetime import datetime,timezone
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                write_json(args.output/'preflight_failures'/f'{c["id"]}_{stamp}.json',
                           dict(run_id=c['id'],state='failed',stage='selection_or_execution',
                                error=str(error),config=c))
        export_leaderboard(args.output,args.output/'leaderboard.xlsx')
        return 1 if failures else 0
    elif args.command=='report':
        from .tracking import export_leaderboard
        print(export_leaderboard(args.output,args.workbook))
    elif args.command=='export':
        from .export import export_model
        print(json.dumps(export_model(args.run,load_dataset(args.cache),args.method,args.output,args.qat_epochs),indent=2))
    elif args.command=='bundle':
        from .export import create_firmware_bundle
        print(create_firmware_bundle(args.export_dir,args.output))
    elif args.command=='lock':
        from .finalize import lock_finalists
        print(lock_finalists(args.run,args.exports,args.output))
    elif args.command=='evaluate':
        from .finalize import evaluate_final
        print(json.dumps(evaluate_final(args.lock,load_dataset(args.cache),args.output),indent=2))
    elif args.command=='robustness':
        from .evaluate import robustness_inputs,evaluate_probabilities
        from .train import predict_run,input_contract
        data=load_dataset(args.cache)
        config=json.loads((args.run/'config.json').read_text())
        contract=input_contract(config,data)
        if contract!=json.loads((args.run/'input_contract.json').read_text()):
            raise ValueError('Robustness cache differs from training data')
        threshold=json.loads((args.run/'training_result.json').read_text())['validation']['threshold']
        results={name:evaluate_probabilities(data['y_val'],predict_run(args.run,x),threshold)
                 for name,x in robustness_inputs(data['X_val'],config['seed']).items()}
        write_json(args.output,dict(scope='validation_feature_perturbations',results=results))
    elif args.command=='board':
        from .export import import_board_measurement
        print(json.dumps(import_board_measurement(args.measurement,args.output),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
