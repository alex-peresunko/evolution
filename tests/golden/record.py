"""Record the golden trace of the current code: poetry run python -m tests.golden.record

Only re-record when the simulation's behaviour is meant to change."""
import json
import os
import platform
import tempfile

from tests.golden import scenarios
from tests.golden.fingerprint import digest, fingerprint

TRACE = os.path.join(os.path.dirname(__file__), "trace.json")


def render_platform():
    import pygame
    return f"{platform.system()} {platform.machine()} pygame {pygame.version.ver} SDL {pygame.get_sdl_version()}"


def capture_evolution():
    with tempfile.TemporaryDirectory() as workdir:
        return [[label, fingerprint(obs)] for label, obs in scenarios.evolution_run(workdir)]


def capture_imitation():
    return fingerprint(scenarios.imitation_run())


def capture_render():
    return [[label, fingerprint(obs)] for label, obs in scenarios.render_run()]


def main():
    trace = {
        "evolution": capture_evolution(),
        "imitation": capture_imitation(),
        "render": {"platform": render_platform(), "frames": capture_render()},
    }
    trace["digest"] = digest(trace["evolution"])
    with open(TRACE, "w") as f:
        json.dump(trace, f, indent=1)
    print(f"Wrote {TRACE}")


if __name__ == "__main__":
    main()
