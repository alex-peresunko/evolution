"""Command-line options shared by game.py and game_headless.py."""
import argparse

from ..persistence.brain_history import format_listing, resolve
from ..persistence.brain_store import SAVE_DIR


def parser(description=None, worker_option=False, watch_option=False,
           workers_help="island worker processes (one core each)"):
    p = argparse.ArgumentParser(description=description)
    if worker_option:
        p.add_argument("--worker-id", type=int, default=None)
    p.add_argument("--workers", type=int, metavar="N", default=None,
                   help=workers_help)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--new", action="store_true",
                       help="start from random brains (the current best brains go to history)")
    group.add_argument("--load", metavar="SNAPSHOT",
                       help="start from a history snapshot: index (1 = newest) or folder name")
    group.add_argument("--list-brains", action="store_true",
                       help="list the history snapshots and exit")
    if watch_option:
        group.add_argument("--watch", action="store_true",
                           help="read-only: show the saved best brains and swap new ones into the "
                                "living creatures whenever a running headless simulation "
                                "saves them")
    return p


def parse(p, argv=None, save_dir=SAVE_DIR):
    """Parsed arguments; prints the snapshot listing and exits for --list-brains, and exits with
    a usage error when --load names no snapshot."""
    args = p.parse_args(argv)
    if args.workers is not None:
        if args.workers < 0:
            p.error("--workers must be 0 or more")
        if getattr(args, "worker_id", None) is not None:
            p.error("--workers cannot be combined with --worker-id")
        if getattr(args, "watch", False):
            p.error("--workers cannot be combined with --watch")
    if args.list_brains:
        print(format_listing(save_dir))
        p.exit()
    if args.load is not None:
        try:
            resolve(args.load, save_dir)
        except ValueError as e:
            p.error(str(e))
    return args
