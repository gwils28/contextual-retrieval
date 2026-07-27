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

.PHONY: b3-questions
b3-questions:
	uv run python -m ragbench.b3 questions

# Freezes the cascade threshold from the development questions; refuses to overwrite it.
.PHONY: b3-threshold
b3-threshold:
	uv run python -m ragbench.b3 threshold

# Development questions only: prompt design (configs/b3_system_protocol.yaml, data.development).
# Retrieval runs first and frees the GPU before the LLM is loaded (8 GB GPU).
.PHONY: b3-dev
b3-dev:
	uv run python -m ragbench.b3 retrieve --role development
	uv run python -m ragbench.b3 generate --system oracle_context --role development
	uv run python -m ragbench.b3 generate --system fixed --role development
	uv run python -m ragbench.b3 generate --system agentic --role development
	uv run python -m ragbench.b3 cascade --role development
	uv run python -m ragbench.b3 summary --role development

# Evaluation questions: run ONCE, after the freeze (configs/b3_system_protocol.yaml).
.PHONY: b3-eval b3-report b3
b3-eval:
	uv run python -m ragbench.b3 retrieve --role evaluation
	uv run python -m ragbench.b3 generate --system oracle_context --role evaluation
	uv run python -m ragbench.b3 generate --system fixed --role evaluation
	uv run python -m ragbench.b3 generate --system agentic --role evaluation
	uv run python -m ragbench.b3 cascade --role evaluation

# Protocol tables (CI, paired tests, strata, layer economics) from the saved per-question rows.
b3-report:
	uv run python -m ragbench.b3 report --role evaluation

# Full Block 3: needs a running Ollama server.
b3: b3-eval b3-report
