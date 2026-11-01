"""Record the golden trace of the current code (make golden-record)."""
import contextlib
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


@contextlib.contextmanager
def _scratch_cwd():
    """The scenarios autosave brains into ./saved_brains; keep them out of the real one."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        os.chdir(workdir)
        try:
            yield
        finally:
            os.chdir(cwd)


def capture_imitation():
    with _scratch_cwd():
        return fingerprint(scenarios.imitation_run())


def capture_render():
    with _scratch_cwd():
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
