# Manimations

Animations built with [Manim](https://www.manim.community/).

## Projects

- **system_design** — client/server retries, concurrency, queueing
- **distributions** — statistical distributions

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

Render a scene:

```sh
uv run manim src/system_design/scenes.py ClientServerTest
```

Run tests:

```sh
uv run pytest
```

Lint:

```sh
uv run ruff check src/ tests/
```
