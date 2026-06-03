# Development

All commands are run from the repository root (the folder with
`pyproject.toml`):

```bash
cd rag-toolkit
```

## Environment

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`uv sync` installs the runtime dependencies plus the default `dev` group defined
in `pyproject.toml`. The `docs` group (which provides `mkdocstrings`) is **not**
part of the default sync and must be requested explicitly:

```bash
uv sync --group dev      # explicit dev tooling (already part of uv sync)
uv sync --group docs     # add mkdocstrings for the documentation site
```

## Documentation

Build and preview the documentation locally with mkdocs-material and
mkdocstrings. Pass `--group docs` so the `mkdocstrings` plugin is available:

```bash
uv sync --group docs
uv run --group docs mkdocs serve     # http://127.0.0.1:8000/
uv run --group docs mkdocs build     # static site → site/
```

API pages under `docs/api/` are generated from source docstrings. If you
add a new module, update the matching `docs/api/<layer>.md` to include it.

## Tests

```bash
uv run pytest
```

## Code style

The project follows standard Python conventions:

- Type hints on public functions and methods.
- Frozen dataclasses for configuration objects.
- Streaming pipelines where possible (generators, no full-corpus loads).
- Short docstrings; parameter/return sections only when they add value.

## Adding a new layer or module

1. Create the module under the appropriate `rag/<layer>/` package.
2. Re-export public names from `rag/<layer>/__init__.py` if relevant.
3. Add a `::: rag.<layer>.<module>` entry to the matching `docs/api/<layer>.md`.
4. Cover the new behaviour with tests under `tests/`.

## Releasing

Bump the version in `pyproject.toml` and tag the commit. The docs build is
self-contained and does not require a release step.
