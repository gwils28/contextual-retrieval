"""Two chunking strategies over FinQA pages, each with its gold-fact -> chunk mapping.

- ``naive``: linearize the page (pre_text, table rows as "a | b | c", post_text) and cut it into
  fixed windows of ``window_tokens`` tokens. Windows ignore structure, so a table can be split
  in the middle and a row can end up far from its column headers.
- ``table_aware``: sentences are grouped up to ``window_tokens`` without crossing the table;
  each table data row becomes its own chunk, serialized with its column headers.

A gold fact (``text_i`` / ``table_i``) maps to a *set* of acceptable chunks. The set has one
element except for a gold header row (``table_0``) under ``table_aware``: the header is repeated
in every row chunk, so any row chunk of that table carries it. Tables whose first row is data
(no header, see ``is_headerless``) get one chunk per row, row 0 included.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ragbench.finqa import Page

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_id: str
    ticker: str
    text: str
    kind: str  # "text", "table" or "mixed"


@dataclass
class ChunkedCorpus:
    strategy: str
    chunks: list[Chunk] = field(default_factory=list)
    # page_id -> gold key -> indices (into ``chunks``) of acceptable chunks
    gold_map: dict[str, dict[str, frozenset[int]]] = field(default_factory=dict)

    def gold_sets(self, page_id: str, gold_keys: Iterable[str]) -> list[frozenset[int]]:
        """Acceptable chunk sets for a question's gold facts, duplicates merged."""
        sets = {self.gold_map[page_id][k] for k in gold_keys if k in self.gold_map[page_id]}
        return sorted(sets, key=min)


_NUMERIC = re.compile(r"^[\s$()%\-.,0-9]+$")
_YEAR = re.compile(r"^\s*(19|20)\d\d\s*$")


def is_headerless(table: tuple[tuple[str, ...], ...]) -> bool:
    """True when row 0 is data, not a header: all its value cells are numbers other than years.

    FinQA keeps the first table row at index 0 whatever it is. Year headers ("2015 | 2014") are
    real headers; "cash | $ 108" is a data row.
    """
    cells = [c for c in table[0][1:] if c.strip()]
    return bool(cells) and all(_NUMERIC.match(c) and not _YEAR.match(c) for c in cells)


def serialize_row(header: tuple[str, ...], row: tuple[str, ...]) -> str:
    """'<row label> | <header>: <value> | ...'; header cells that are empty are omitted."""
    parts = [row[0].strip()] if row and row[0].strip() else []
    for h, v in zip(header[1:], row[1:], strict=False):
        h, v = h.strip(), v.strip()
        if v:
            parts.append(f"{h}: {v}" if h else v)
    return " | ".join(parts)


def linearize_page(page: Page) -> tuple[str, list[tuple[str, int, int]]]:
    """Page text and the (gold_key, start, end) character span of every unit."""
    pieces: list[str] = []
    spans: list[tuple[str, int, int]] = []
    pos = 0

    def add(key: str, text: str) -> None:
        nonlocal pos
        if pieces:
            pos += 1  # the "\n" separator
        spans.append((key, pos, pos + len(text)))
        pieces.append(text)
        pos += len(text)

    n_pre = len(page.pre_text)
    for i, s in enumerate(page.pre_text):
        add(f"text_{i}", s)
    for r, row in enumerate(page.table):
        add(f"table_{r}", " | ".join(c.strip() for c in row))
    for j, s in enumerate(page.post_text):
        add(f"text_{n_pre + j}", s)
    return "\n".join(pieces), spans


def chunk_naive(pages: Iterable[Page], tokenizer, window_tokens: int = 200) -> ChunkedCorpus:
    """Fixed token windows over the linearized page (no overlap)."""
    corpus = ChunkedCorpus("naive")
    for page in pages:
        text, spans = linearize_page(page)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        windows = []
        for start in range(0, len(offsets), window_tokens):
            window = offsets[start : start + window_tokens]
            windows.append((window[0][0], window[-1][1]))
        if not windows:
            continue
        base = len(corpus.chunks)
        table_span = [(s, e) for k, s, e in spans if k.startswith("table_")]
        for n, (s, e) in enumerate(windows):
            in_table = any(s < te and e > ts for ts, te in table_span)
            only_table = bool(table_span) and s >= table_span[0][0] and e <= table_span[-1][1]
            kind = "table" if only_table else ("mixed" if in_table else "text")
            corpus.chunks.append(
                Chunk(f"{page.page_id}#naive#{n}", page.page_id, page.ticker, text[s:e], kind)
            )
        mapping = {}
        for key, us, ue in spans:
            overlaps = [max(0, min(ue, e) - max(us, s)) for s, e in windows]
            best = max(range(len(windows)), key=lambda i: (overlaps[i], -i))
            if overlaps[best] == 0:  # empty unit: attach to the window that contains its position
                best = max(i for i, (s, _) in enumerate(windows) if s <= us) if us else 0
            mapping[key] = frozenset({base + best})
        corpus.gold_map[page.page_id] = mapping
    return corpus


def _group_sentences(
    sentences: list[tuple[str, str]], count: TokenCounter, window_tokens: int
) -> list[list[tuple[str, str]]]:
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for key, s in sentences:
        n = count(s)
        if current and size + n > window_tokens:
            groups.append(current)
            current, size = [], 0
        current.append((key, s))
        size += n
    if current:
        groups.append(current)
    return groups


def chunk_table_aware(
    pages: Iterable[Page], count: TokenCounter, window_tokens: int = 200
) -> ChunkedCorpus:
    """Sentence groups that never cross the table + one chunk per table data row."""
    corpus = ChunkedCorpus("table_aware")
    for page in pages:
        corpus.gold_map[page.page_id] = _chunk_page_table_aware(page, corpus, count, window_tokens)
    return corpus


def _chunk_page_table_aware(
    page: Page, corpus: ChunkedCorpus, count: TokenCounter, window_tokens: int
) -> dict[str, frozenset[int]]:
    mapping: dict[str, frozenset[int]] = {}
    n = 0

    def emit(text: str, kind: str) -> int:
        nonlocal n
        corpus.chunks.append(
            Chunk(f"{page.page_id}#table_aware#{n}", page.page_id, page.ticker, text, kind)
        )
        n += 1
        return len(corpus.chunks) - 1

    n_pre = len(page.pre_text)
    pre = [(f"text_{i}", s) for i, s in enumerate(page.pre_text)]
    post = [(f"text_{n_pre + j}", s) for j, s in enumerate(page.post_text)]

    for group in _group_sentences(pre, count, window_tokens):
        idx = emit(" ".join(s for _, s in group), "text")
        mapping.update({k: frozenset({idx}) for k, _ in group})

    if page.table:
        header = page.table[0]
        if len(page.table) == 1 or is_headerless(page.table):
            for r, row in enumerate(page.table):
                mapping[f"table_{r}"] = frozenset(
                    {emit(" | ".join(c.strip() for c in row), "table")}
                )
        else:
            rows = []
            for r, row in enumerate(page.table[1:], start=1):
                idx = emit(serialize_row(header, row), "table")
                mapping[f"table_{r}"] = frozenset({idx})
                rows.append(idx)
            mapping["table_0"] = frozenset(rows)

    for group in _group_sentences(post, count, window_tokens):
        idx = emit(" ".join(s for _, s in group), "text")
        mapping.update({k: frozenset({idx}) for k, _ in group})

    return mapping
