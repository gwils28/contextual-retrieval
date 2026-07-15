.PHONY: test lint

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
