### Building and running

Containers run with [Podman](https://podman.io). On macOS, start its VM once with
`podman machine init && podman machine start`; `podman compose` also needs
`podman-compose` (`brew install podman-compose`).

Start the headless simulation together with Prometheus and Grafana:
`podman compose up --build -d`
(or `make run headless`, which also waits for Grafana and opens it; `make stop` stops it).

Then open http://localhost:3000. Grafana opens straight on the evolution dashboard, with no
login. The other ports are:
- http://localhost:9090: Prometheus, which scrapes the simulation every 5 s and keeps its
  data in the `prometheus-data` volume between runs;
- http://localhost:8000: the raw metrics.

`podman compose logs -f server` follows the simulation's console. `podman compose down` stops
everything; on SIGTERM the simulation saves its Hall of Fame before exiting. Add `-v` to
`down` to also delete the stored Prometheus history.

The monitoring config lives in `monitoring/`: the Prometheus scrape job, and Grafana's
provisioned data source and dashboard. The dashboard itself is `grafana_dashboard.json`.

`saved_brains/` is mounted into the container, so the simulation starts from your
`best_brain_*.json` and overwrites them with new Hall of Fame brains as it runs (at most once
per `AUTOSAVE_INTERVAL` ticks, and on stop). Like a local run, it saves a brain that beats the
*faded* score of the previous one, so a file's score can go down. Each start snapshots the
previous pair into `saved_brains/history/`. Don't run `game.py` on the same folder at the
same time: both write the best brain files.

### Deploying to the cloud

Build the image, e.g.: `podman build -t evolution .`.
If your cloud uses a different CPU architecture than your development machine (e.g. you
are on Apple Silicon and your cloud provider is amd64), build for that platform:
`podman build --platform=linux/amd64 -t evolution .`.

Then push it to your registry, e.g. `podman push evolution myregistry.com/evolution`.

### References
* [Podman documentation](https://docs.podman.io/)
* [podman-compose](https://github.com/containers/podman-compose)
