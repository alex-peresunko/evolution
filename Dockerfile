# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements-headless.txt,target=requirements-headless.txt \
    python -m pip install -r requirements-headless.txt

COPY . .

# --no-deps: the headless deps are installed above. As root: appuser has no home for pip.
RUN pip install -e . --no-deps --quiet && chown -R appuser saved_brains

USER appuser

EXPOSE 8000

# Exec form, so python is PID 1 and gets SIGTERM on stop: it saves the Hall of Fame first.
CMD ["python", "game_headless.py"]
