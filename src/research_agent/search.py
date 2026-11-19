"""Local-corpus 'web search' stub using BM25 ranking (no network access)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from research_agent.memory import tokenize

DEFAULT_CORPUS_DIR = Path(__file__).parent / "data" / "corpus"


@dataclass(frozen=True)
class Document:
    """A corpus document. The first line of the file is its title."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    """A ranked search result."""

    doc_id: str
    title: str
    score: float
    snippet: str


class LocalCorpus:
    """BM25 search over ``*.txt`` / ``*.md`` files in a directory."""

    def __init__(self, directory: Path | str = DEFAULT_CORPUS_DIR, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.documents: list[Document] = []
        for path in sorted(Path(directory).glob("*")):
            if path.suffix.lower() in {".txt", ".md"} and path.is_file():
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    title = lines[0].lstrip("# ").strip()
                    self.documents.append(Document(path.stem, title, " ".join(lines[1:]).strip()))
        self._tokens = [tokenize(d.title + " " + d.text) for d in self.documents]
        self._avg_len = sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        self._df: dict[str, int] = {}
        for toks in self._tokens:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        """Return up to ``k`` documents with positive BM25 score, best first."""
        terms = tokenize(query)
        n = len(self.documents)
        hits: list[SearchHit] = []
        for doc, toks in zip(self.documents, self._tokens):
            score = 0.0
            for term in set(terms):
                tf = toks.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * len(toks) / (self._avg_len or 1))
                score += idf * tf * (self.k1 + 1) / norm
            if score > 0:
                hits.append(SearchHit(doc.doc_id, doc.title, round(score, 4), _snippet(doc, terms)))
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:k]


def _snippet(doc: Document, terms: list[str], n_sentences: int = 2) -> str:
    """Join the ``n_sentences`` best-matching sentences (kept in document order)."""
    sentences = re.split(r"(?<=[.!?])\s+", doc.text)
    wanted = set(terms)
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: (-len(wanted & set(tokenize(sentences[i]))), i),
    )
    keep = sorted(ranked[:n_sentences])
    return " ".join(sentences[i].strip() for i in keep)
