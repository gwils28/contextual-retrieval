# CLAUDE.md

Learning project on advanced & agentic RAG. Decisions and scope: `docs/charter.md` (read it first).

## Commands

- `uv sync` — install (Python 3.12, pinned in `uv.lock`)
- `make test` — pytest; `make lint` — ruff
- Every published number must come from a `make` target, never from an ad-hoc cell.

## Rules (with why)

- **Commits:** author = committer = `wilson goma <10559068+gwils28@users.noreply.github.com>`, backdated
  (`GIT_AUTHOR_DATE` = `GIT_COMMITTER_DATE`, after the previous commit's date). Never add
  `Co-Authored-By` or any tool trailer — the owner must be the sole GitHub contributor.
- **Gates:** submit plan + protocol YAML before coding a block, results + diff before commit.
  The owner decides; never commit or push without approval.
- **Protocol first:** eval protocols live in `configs/*.yaml` and are frozen before results
  are seen — otherwise metric choices get tuned to the outcome.
- **Retrieval eval never looks at generated answers** — Pass@k/recall only judge whether the
  golden chunk was retrieved. Mixing stages hides where failures come from.
- **Golden chunk mapping** must survive re-chunking: goldens are defined at the FinQA fact
  level (`gold_inds`) and mapped to chunks per strategy.
- **Notebooks call `ragbench`**; no logic lives only in a notebook (it would be untested).
- `data/raw/` is read-only once downloaded; its `sha256` is recorded in `data/MANIFEST.json`.
- Secrets in `.env` (gitignored); never print or read it.
- Code, docs and commit messages in English.
