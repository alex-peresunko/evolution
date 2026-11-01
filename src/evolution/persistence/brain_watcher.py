"""Watch mode: reports best brain files rewritten by another process (the headless run)."""
import json
import os

from ..config import WATCH_POLL_SECONDS
from ..species import ALL_SPECIES
from .brain_history import VOLATILE_BRAIN_KEYS
from .brain_store import SAVE_DIR


def _identity(data):
    return {k: v for k, v in data.items() if k not in VOLATILE_BRAIN_KEYS}


class BrainWatcher:
    """Polls the best brain files. A file counts as changed only when its brain differs from the
    last one seen; history-only rewrites (every headless exit) are ignored. The first scan in
    __init__ sets the baseline, so the files present at startup are not reported."""

    def __init__(self, save_dir=SAVE_DIR, interval=WATCH_POLL_SECONDS):
        self.save_dir = save_dir
        self.interval = interval
        self._next_poll = None
        self._signature = {}  # species -> (st_mtime_ns, st_size) of the last file read
        self._identity = {}   # species -> brain identity of the last file read
        for species in ALL_SPECIES:
            self._scan(species)

    def poll(self, now):
        """[(species, data)] for the brain files that changed since the last poll, where data is
        the parsed file. Returns [] until `interval` seconds have passed since the last poll."""
        if self._next_poll is not None and now < self._next_poll:
            return []
        self._next_poll = now + self.interval
        return [found for species in ALL_SPECIES if (found := self._scan(species))]

    def _scan(self, species):
        """(species, data) if the species' file now holds a different brain, else None. A missing
        or half-written file is retried on the next poll."""
        path = os.path.join(self.save_dir, species.brain_file)
        try:
            st = os.stat(path)
        except OSError:
            return None
        signature = (st.st_mtime_ns, st.st_size)
        if signature == self._signature.get(species):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        self._signature[species] = signature
        identity = _identity(data)
        if identity == self._identity.get(species):
            return None
        self._identity[species] = identity
        return species, data
