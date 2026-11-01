from prometheus_client import start_http_server

from evolution.app import cli
from evolution.persistence.brain_history import start_source
from evolution.ui.app import GameApp


def main():
    args = cli.parse(cli.parser(watch_option=True))
    # Watch mode runs beside the headless simulation, which already serves port 8000.
    if not args.watch:
        start_http_server(8000)
    GameApp(start_source(args.new, args.load), new=args.new, read_only=args.watch).run()


if __name__ == "__main__":
    main()
