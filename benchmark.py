"""Compare brain sources in the same seeded worlds: how long their creatures live, how often
they eat and how many litters they parent. Writes nothing.

    python benchmark.py best random          # saved best brains against random ones
    python benchmark.py best 2               # ... against history snapshot #2
"""
import argparse
import contextlib
import io

from evolution.app.benchmark import average, format_report, run, source_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+",
                        help="best, random, or a history snapshot (index or folder name)")
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3, help="worlds per source")
    args = parser.parse_args()

    rows = []
    for spec in args.sources:
        source = source_dir(spec)
        results = []
        for seed in range(args.seeds):
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(run(source, args.ticks, seed))
            print(f"{spec}: seed {seed} done", flush=True)
        rows.append((spec, average(results)))
    print(format_report(rows))


if __name__ == "__main__":
    main()
