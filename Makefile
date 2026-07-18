.PHONY: test lint notebooks

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

notebooks:
	uv run jupyter nbconvert --to notebook --execute --inplace blocks/*/notebooks/*.ipynb

.PHONY: b2-data b2-plain b2-contextualize b2-contextual b2-report b2-analysis b2
b2-data:
	uv run python -m ragbench.b2 data

b2-plain: b2-data
	uv run python -m ragbench.b2 run --variant plain

# The last step unloads the LLM (same as `ollama stop`): on an 8 GB GPU, the embedder of
# b2-contextual runs out of memory while Ollama keeps the model resident.
b2-contextualize: b2-data
	uv run python -m ragbench.b2 contextualize --strategy naive
	uv run python -m ragbench.b2 contextualize --strategy table_aware
	curl -sf http://127.0.0.1:11434/api/generate -d '{"model": "qwen2.5:7b-instruct", "keep_alive": 0}' >/dev/null

b2-contextual: b2-data
	uv run python -m ragbench.b2 run --variant contextual

b2-report:
	uv run python -m ragbench.b2 report

b2-analysis:
	uv run python -m ragbench.b2 analyze

# Full Block 2: needs a running Ollama server for the contextualization step.
b2: b2-plain b2-contextualize b2-contextual b2-report b2-analysis
