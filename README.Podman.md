### Building and running

Containers run with [Podman](https://podman.io). On macOS, start its VM once with
`podman machine init && podman machine start`; `podman compose` also needs
`podman-compose` (`brew install podman-compose`).

Start the headless simulation with:
`podman compose up --build`

Prometheus metrics are then served at http://localhost:8000 (import
`grafana_dashboard.json` into Grafana). `podman compose down` stops it; on SIGTERM the
simulation saves its Hall of Fame before exiting.

The container works on the brains baked into the image at build time, and what it saves
stays inside the container.

### Deploying to the cloud

Build the image, e.g.: `podman build -t evolution .`.
If your cloud uses a different CPU architecture than your development machine (e.g. you
are on Apple Silicon and your cloud provider is amd64), build for that platform:
`podman build --platform=linux/amd64 -t evolution .`.

Then push it to your registry, e.g. `podman push evolution myregistry.com/evolution`.

### References
* [Podman documentation](https://docs.podman.io/)
* [podman-compose](https://github.com/containers/podman-compose)
