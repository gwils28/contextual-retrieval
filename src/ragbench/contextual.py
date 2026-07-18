"""Contextual Retrieval: ask an LLM to situate each chunk within its document, then prepend it.

Prompt from Anthropic, "Introducing Contextual Retrieval" (2024). FinQA gives pages, not whole
reports, so the "document" is the page, headed by the company name (it stands in for the
report's cover page). Contexts are cached in a JSONL file keyed by chunk id, so an interrupted
run resumes where it stopped and results are reproducible from the cache.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from ragbench.chunking import Chunk, linearize_page
from ragbench.finqa import Page

PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the \
purposes of improving search retrieval of the chunk. Answer only with the succinct context and \
nothing else."""


def page_document(page: Page, company: str) -> str:
    text, _ = linearize_page(page)
    return f"Company: {company}\n{text}"


def contextualized_text(context: str, chunk_text: str) -> str:
    """What gets indexed: context first, then the original chunk."""
    return f"{context.strip()}\n\n{chunk_text}"


def ollama_generate(
    model: str, host: str = "http://127.0.0.1:11434", num_ctx: int = 4096, timeout: int = 300
) -> Callable[[str], str]:
    """Deterministic (temperature 0) completion function backed by a local Ollama server."""

    def generate(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": 160},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()

    return generate


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["chunk_id"]: r["context"] for r in map(json.loads, f)}


def contextualize(
    chunks: Iterable[Chunk],
    pages: dict[str, Page],
    company_of: Callable[[str], str],
    generate: Callable[[str], str],
    cache_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """Generate (or reuse) one context per chunk. Chunks of a page are processed consecutively,
    so servers with prefix caching (Ollama, Anthropic prompt caching) reuse the document prefix."""
    chunks = list(chunks)
    cache = load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [c for c in chunks if c.chunk_id not in cache]
    with cache_path.open("a") as out:
        for n, chunk in enumerate(todo, start=1):
            page = pages[chunk.page_id]
            prompt = PROMPT.format(
                document=page_document(page, company_of(chunk.page_id)), chunk=chunk.text
            )
            context = generate(prompt)
            cache[chunk.chunk_id] = context
            out.write(json.dumps({"chunk_id": chunk.chunk_id, "context": context}) + "\n")
            out.flush()
            if progress:
                progress(n, len(todo))
    return {c.chunk_id: cache[c.chunk_id] for c in chunks}
