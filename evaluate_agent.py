"""Evaluate an agent through the same configured backend as HGM initialization."""
import argparse
import signal
from datetime import datetime
from pathlib import Path

from config import load_config
from pipeline import ROOT, require_preflight, run_baseline


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    from network_readiness import add_cli, configure_cli
    add_cli(parser)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--agent_path', default=str(ROOT))
    parser.add_argument('--results_dir')
    parser.add_argument('--llm', default=None, help='Legacy API model only')
    parser.add_argument('--hours_per_task', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--n_tasks', type=int, default=1)
    parser.add_argument('--instance_id', action='append')
    parser.add_argument('--polyglot', action='store_true', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config, **{
        'llm.downstream_llm': args.llm,
        'execution.evaluation_timeout': args.hours_per_task * 3600 if args.hours_per_task is not None else None,
        'execution.max_workers': args.num_workers,
        'evaluation.polyglot': args.polyglot,
    })
    configure_cli(config, args)
    for field in ('dataset_path', 'polyglot_dataset_path'):
        path = Path(getattr(config.evaluation, field))
        if not path.is_absolute():
            setattr(config.evaluation, field, str(ROOT / path))
    output = args.results_dir or str(ROOT / 'results' / ('evaluate-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    require_preflight(config, output_root=output)
    run_baseline(config, output, n_tasks=args.n_tasks, instance_ids=args.instance_id,
                 agent_path=args.agent_path, resume=args.resume)


if __name__ == '__main__':
    main()
