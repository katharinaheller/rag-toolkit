.PHONY: install docs docs-serve docs-build test clean

install:
	uv sync

docs: docs-serve

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

test:
	uv run pytest

clean:
	rm -rf site/ .pytest_cache/ .ruff_cache/ **/__pycache__
