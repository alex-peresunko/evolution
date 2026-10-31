from evolution.app import cli
from evolution.app.headless import HeadlessRunner
from evolution.app.metrics import GameMetrics
from evolution.persistence.brain_history import start_source
from evolution.persistence.brain_store import SAVE_DIR


def main():
    args = cli.parse(cli.parser(worker_option=True))
    metrics = GameMetrics()
    if args.worker_id is None:
        metrics.serve(8000)
        source = start_source(args.new, args.load)
    else:
        source = SAVE_DIR
    HeadlessRunner(source, new=args.new, worker_id=args.worker_id, metrics=metrics).run()


if __name__ == "__main__":
    main()
