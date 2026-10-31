from prometheus_client import start_http_server

from evolution.app import cli
from evolution.app.headless import HeadlessRunner
from evolution.app.metrics import GameMetrics
from evolution.persistence.brain_history import start_source
from evolution.persistence.brain_store import SAVE_DIR


def main():
    args = cli.parse(cli.parser(worker_option=True))
    if args.worker_id is None:
        start_http_server(8000)
        source = start_source(args.new, args.load)
    else:
        source = SAVE_DIR
    HeadlessRunner(source, new=args.new, worker_id=args.worker_id, metrics=GameMetrics()).run()


if __name__ == "__main__":
    main()
