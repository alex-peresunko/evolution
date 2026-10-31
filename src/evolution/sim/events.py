class SimulationListener:
    """Hooks a Simulation calls at fixed points of its tick; override the ones you need."""

    def record_set(self, sim, species):
        """A retired creature just became the species' new Hall of Fame best."""

    def deaths_processed(self, sim):
        """This tick's dead were removed and retired."""

    def extinction(self, sim, species):
        """The species died out; it is repopulated right after this call."""
