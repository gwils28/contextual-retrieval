"""FinQA loading: download with integrity check, then expose pages and questions.

FinQA ships one record per *question*; each record embeds the full content of one PDF *page*
(``pre_text`` sentences, a ``table``, ``post_text`` sentences). Several questions share a page.
We de-duplicate pages so that the retrieval corpus holds each page once.

Gold evidence (``qa.gold_inds``) uses two kinds of keys:

- ``text_i``: the i-th sentence of ``pre_text + post_text`` (pre first, then post);
- ``table_i``: the i-th row of ``table`` (row 0 is usually the header row).
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/{commit}/dataset/{split}.json"
FINQA_SHA256 = {
    "train": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    "dev": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
}


@dataclass(frozen=True)
class Page:
    page_id: str  # FinQA filename, e.g. "ETR/2016/page_23.pdf"
    ticker: str
    pre_text: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]
    post_text: tuple[str, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        """Sentences in FinQA ``text_i`` order: pre_text then post_text."""
        return self.pre_text + self.post_text


@dataclass(frozen=True)
class Question:
    qid: str
    page_id: str
    ticker: str
    question: str
    gold_keys: tuple[str, ...]  # e.g. ("table_1", "text_3")
    program: str
    answer: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(raw_dir: Path, splits: tuple[str, ...] = ("test",)) -> dict[str, str]:
    """Download FinQA splits (pinned commit) and verify their sha256. Returns the manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split in splits:
        path = raw_dir / f"{split}.json"
        if not path.exists():
            urllib.request.urlretrieve(FINQA_URL.format(commit=FINQA_COMMIT, split=split), path)
        digest = sha256_file(path)
        if digest != FINQA_SHA256[split]:
            raise ValueError(f"sha256 mismatch for {path}: {digest}")
        manifest[f"finqa/{split}.json"] = digest
    return manifest


def load(path: Path) -> tuple[dict[str, Page], list[Question]]:
    """Parse one FinQA split into unique pages (by filename) and questions."""
    records = json.loads(path.read_text())
    pages: dict[str, Page] = {}
    questions: list[Question] = []
    for r in records:
        page_id = r["filename"]
        ticker = page_id.split("/")[0]
        if page_id not in pages:
            pages[page_id] = Page(
                page_id=page_id,
                ticker=ticker,
                pre_text=tuple(r["pre_text"]),
                table=tuple(tuple(row) for row in r["table"]),
                post_text=tuple(r["post_text"]),
            )
        qa = r["qa"]
        questions.append(
            Question(
                qid=r["id"],
                page_id=page_id,
                ticker=ticker,
                question=qa["question"],
                gold_keys=tuple(sorted(qa["gold_inds"])),
                program=qa.get("program", ""),
                answer=str(qa.get("exe_ans", qa.get("answer", ""))),
            )
        )
    return pages, questions


def load_company_names(path: Path) -> dict[str, list[tuple[int | None, str]]]:
    """Ticker -> [(until_year, name), ...]. A company renamed over time has several rows."""
    names: dict[str, list[tuple[int | None, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            until = int(row["until_year"]) if row["until_year"] else None
            names.setdefault(row["ticker"], []).append((until, row["company_name"]))
    return names


def company_name(names: dict[str, list[tuple[int | None, str]]], page_id: str) -> str:
    """Name in use for the page's folder year: first row whose until_year covers that year."""
    ticker, year = page_id.split("/")[0], int(page_id.split("/")[1])
    for until, name in sorted(names[ticker], key=lambda r: r[0] if r[0] is not None else 10**4):
        if until is None or year <= until:
            return name
    raise KeyError(page_id)
