"""Cross-platform helpers for the Makefile (macOS/Linux sh and Windows cmd.exe alike).

    python scripts/make_tasks.py not-running       exit 1 if a simulation already runs
    python scripts/make_tasks.py wait-url URL [S]  poll URL until it answers (default 120 s)
    python scripts/make_tasks.py open URL          open URL in the browser (or print it)

Standard library only, so it runs before `poetry install` finishes too.
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser

# game.py or game_headless.py. Watch mode never writes saved_brains/, so it does not count.
SIM_PATTERN = re.compile(r"game(_headless)?\.py")
CONTAINER_FILTERS = [
    "--filter", "label=io.podman.compose.project=evolution",
    "--filter", "label=io.podman.compose.service=server",
]


def _process_lines() -> list[str]:
    """One "pid command line" string per running process."""
    if sys.platform == "win32":
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process | ForEach-Object "
            '{ "$($_.ProcessId) $($_.CommandLine)" }',
        ]
    else:
        cmd = ["ps", "-Ao", "pid=,command="]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _local_sims() -> list[str]:
    own = str(os.getpid())
    return [
        line for line in _process_lines()
        if SIM_PATTERN.search(line) and "--watch" not in line and line.split()[0] != own
    ]


def _container_running() -> bool:
    try:
        out = subprocess.run(
            ["podman", "ps", "-q", *CONTAINER_FILTERS],
            capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:  # no podman installed: no container either
        return False
    return bool(out.strip())


def not_running() -> int:
    """Both kinds write saved_brains/best_brain_*.json, so only one simulation may run."""
    local = _local_sims()
    if local:
        print("a simulation is already running locally:")
        print("\n".join(local))
        return 1
    if _container_running():
        print("the headless simulation container is running (make stop)")
        return 1
    return 0


def wait_url(url: str, seconds: int = 120) -> int:
    for _ in range(seconds):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 400:
                    return 0
        except OSError:
            pass
        time.sleep(1)
    print(f"{url} is not ready after {seconds}s")
    return 1


def open_url(url: str) -> int:
    if not webbrowser.open(url):
        print(url)
    return 0


def main(argv: list[str]) -> int:
    match argv:
        case ["not-running"]:
            return not_running()
        case ["wait-url", url]:
            return wait_url(url)
        case ["wait-url", url, seconds]:
            return wait_url(url, int(seconds))
        case ["open", url]:
            return open_url(url)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
