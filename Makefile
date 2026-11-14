# `make run ui` / `make run headless`: `run` only makes the pair read naturally.
.PHONY: run ui watch headless restart stop clean-data logs test golden golden-record lint not-running

GRAFANA := http://localhost:3000
# make run headless WORKERS=4: island workers in the container, one core each.
# Exported, so compose sees it in both sh (macOS/Linux) and cmd.exe (Windows).
WORKERS ?= 0
export WORKERS
# Must match DASHBOARD_UID in monitoring/dashboard.py.
DASHBOARD := $(GRAFANA)/api/dashboards/uid/6a418c28-cbeb-46f9-a1f4-af86d6e6d051
# Shell logic lives in Python: on Windows, make runs recipes in cmd.exe, not sh.
TASKS := poetry run python scripts/make_tasks.py

run:
ifeq ($(filter-out run,$(MAKECMDGOALS)),)
	@echo usage: make run ui, make run watch or make run headless
	@exit 2
else
	@:
endif

ui: not-running
	poetry run python game.py $(ARGS)

# Read-only: runs alongside a headless simulation, so no not-running guard.
watch:
	poetry run python game.py --watch $(ARGS)

headless: not-running
	podman compose up --build -d
	@echo waiting for Grafana...
# /api/health answers before the data source and dashboard are provisioned.
	@$(TASKS) wait-url $(DASHBOARD) 120
	@$(TASKS) open $(GRAFANA)
	@echo make logs: follow the simulation - make stop: save and stop

# Both write saved_brains/best_brain_*.json, so only one simulation may run.
not-running:
	@$(TASKS) not-running

stop:
	podman compose down

# Rebuilds the image, so code changes take effect; metrics history is kept.
restart:
	$(MAKE) stop
	$(MAKE) headless

# Stops the stack (the simulation saves its brains) and deletes the Prometheus and Grafana
# volumes, so the next start has empty metrics history and a fresh Grafana database.
clean-data:
	podman compose down --volumes

logs:
	podman compose logs -f server

test:
	poetry run pytest

golden:
	poetry run pytest -m golden

golden-record:
	poetry run python -m tests.golden.record

lint:
	poetry run ruff check .
