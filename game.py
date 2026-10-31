from prometheus_client import start_http_server

from evolution.app import cli
from evolution.persistence.brain_history import start_source
from evolution.ui.app import GameApp


def main():
    args = cli.parse(cli.parser())
    start_http_server(8000)
    GameApp(start_source(args.new, args.load), new=args.new).run()


if __name__ == "__main__":
    main()
