# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install/sync dependencies
uv sync

# Render a manim scene (outputs to media/)
uv run manim src/<project>/scenes.py <SceneName>

# Render low-quality preview (faster iteration)
uv run manim -ql src/<project>/scenes.py <SceneName>

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/system_design/test_processor.py

# Run a single test
uv run pytest tests/system_design/test_processor.py::test_function_name

# Lint
uv run ruff check src/ tests/
```

## Architecture

The codebase contains independent animation projects under `src/`, with shared utilities in `src/shared/`.

### Projects

- **system_design** — Client/server simulation demonstrating retries, concurrency limits, queueing, and failure handling. The core loop: `Processor` generates requests (Poisson-distributed), sends `Message` objects along a `Connection`, the server `Processor` handles them with simulated latency/failures, and responses travel back. Visual components (bars, sparklines, labels) track metrics in real time.
- **distributions** — Statistical distribution visualisations. Currently has a Poisson PMF scene showing how variance changes with lambda.

### Shared utilities (`src/shared/`)

- **components/** — Reusable Manim visual elements: `StackedBar` (multi-colour bar chart), `Sparkline` (time-series trace with dissipation), `create_label()` (auto-updating `Variable` labels).
- **aggregators/** — Data tracking: `MovingAverageTracker` (extends `ValueTracker` with bounded deque), `MovingSum` (time-windowed sum).

### Key patterns

- **Updater-driven animation**: Most dynamic behaviour uses Manim updaters (lambdas added via `.add_updater()`). Processors start/stop by adding/removing their update function.
- **Message pooling**: `Processor` recycles `Message` objects from a free list rather than creating new ones each frame.
- **`always_redraw()`**: Used for visuals that recompute every frame (e.g. Poisson distribution bars).
- **`ValueTracker`**: Animatable parameters (RPS rate, failure rate) that drive updaters.

## Manim gotchas

- **Caching hashes closures**: Manim's animation caching inspects closure vars and globals of updater functions recursively. `from scipy.stats import poisson` imports a class instance (not a module), which gets traversed and crashes on `numpy.vectorize` internals. Fix: use `import scipy.stats as stats` so the module reference gets filtered out.
