from ragbench.chunking import chunk_table_aware
from ragbench.contextual import PROMPT, contextualize, contextualized_text, load_cache
from ragbench.finqa import Page

PAGE = Page(
    "ACME/2019/page_1.pdf", "ACME", ("revenue grew .",), (("", "2019"), ("sales", "$ 5")), ()
)


def test_prompt_follows_anthropic_template():
    assert "<document>" in PROMPT and "<chunk>" in PROMPT
    assert "Answer only with the succinct context" in PROMPT


def test_contextualize_calls_llm_once_per_chunk_and_resumes_from_cache(tmp_path):
    corpus = chunk_table_aware([PAGE], lambda s: len(s.split()))
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return f"context {len(calls)}"

    cache = tmp_path / "ctx.jsonl"
    names = lambda page_id: "Acme Corp"  # noqa: E731
    first = contextualize(corpus.chunks, {PAGE.page_id: PAGE}, names, fake_llm, cache)
    assert len(calls) == len(corpus.chunks) == 2
    assert "Company: Acme Corp" in calls[0]
    again = contextualize(corpus.chunks, {PAGE.page_id: PAGE}, names, fake_llm, cache)
    assert len(calls) == 2  # nothing regenerated
    assert again == first == load_cache(cache)


def test_context_is_prepended():
    assert contextualized_text(" From ACME's 2019 report. ", "sales | 2019: $ 5") == (
        "From ACME's 2019 report.\n\nsales | 2019: $ 5"
    )
