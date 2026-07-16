.PHONY: test lint notebooks

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

notebooks:
	uv run jupyter nbconvert --to notebook --execute --inplace blocks/*/notebooks/*.ipynb
