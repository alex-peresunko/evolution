from ..persistence.autosave import log_prefix
from ..sim.events import SimulationListener


class ExtinctionReporter(SimulationListener):
    def __init__(self, worker_id=None):
        self.prefix = log_prefix(worker_id)

    def extinction(self, sim, species):
        print(f"\n{self.prefix}--- {species.title} EXTINCTION tick={sim.tick} "
              f"gen={sim.generation} ---")
