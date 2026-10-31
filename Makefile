# `make run ui` / `make run headless`: `run` only makes the pair read naturally.
.PHONY: run ui headless stop logs test lint not-running

GRAFANA := http://localhost:3000
# Brackets keep pgrep from matching the recipe shell's own command line.
LOCAL_SIM := [g]ame(_headless)?\.py
CONTAINER_SIM := --filter label=io.podman.compose.project=evolution \
	--filter label=io.podman.compose.service=server

run:
ifeq ($(filter-out run,$(MAKECMDGOALS)),)
	@echo "usage: make run ui | make run headless"; exit 2
else
	@:
endif

ui: not-running
	poetry run python game.py $(ARGS)

headless: not-running
	podman compose up --build -d
	@echo "waiting for Grafana..."
	@until curl -sf $(GRAFANA)/api/health >/dev/null; do sleep 1; done
	@open $(GRAFANA) 2>/dev/null || echo "Grafana: $(GRAFANA)"
	@echo "make logs: follow the simulation · make stop: save and stop"

# Both write saved_brains/best_brain_*.json, so only one simulation may run.
not-running:
	@if pgrep -f '$(LOCAL_SIM)' >/dev/null; then \
		echo "a simulation is already running locally:"; pgrep -fl '$(LOCAL_SIM)'; exit 1; fi
	@if [ -n "$$(podman ps -q $(CONTAINER_SIM) 2>/dev/null)" ]; then \
		echo "the headless simulation container is running (make stop)"; exit 1; fi

stop:
	podman compose down

logs:
	podman compose logs -f server

test:
	poetry run pytest

lint:
	poetry run ruff check .
