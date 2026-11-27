"""Recall's lexical lane: BM25 over the words of each entry, an identifier boost, and how both fuse with similarity.

Similarity finds what an entry is about; it misses the exact name a question turns on (``deploy_api.sh``,
``MemoryService``, ``OTARI-1234``), which an embedding model reads as a few common sub-words. This lane matches
words, and identifiers whole, and its rank is fused with the dense lane's by Reciprocal Rank Fusion.

Everything here is pure and deterministic, so it runs on a worker thread and in a memory service alike. Its input
is opened text, which no database index ever sees: entries are sealed at rest, so the index is built in the process
that opened them, from the candidates a member may recall, and its statistics are over those candidates alone.
"""

import math
import re
import sys
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# BM25's term-frequency saturation and length normalization: the Robertson and Lucene defaults.
BM25_K1 = 1.2
BM25_B = 0.75
# Reciprocal Rank Fusion's constant (Cormack, Clarke and Buettcher, 2009): a lane's first place adds 1/61 and its
# tenth 1/70, so agreement between lanes outweighs a lone first place in one.
RRF_K = 60
# An identifier the query and an entry share adds at most what first place in a third lane would.
IDENTIFIER_BOOST = 1 / (RRF_K + 1)
# The popularity penalty: an identifier ``n`` candidates hold weighs 1 / (1 + alpha (n - 1)^2). Held by one it
# weighs 1, by two 0.67, by three 0.33, by five 0.11 and by ten 0.02, so a name every entry repeats (the repo's,
# the product's) stops telling entries apart, as it should.
IDENTIFIER_POPULARITY = 0.5
# An entry whose shared identifiers weigh this much is recalled however unlike its embedding is: one held by at most
# two candidates, or several rarer ones.
IDENTIFIER_ADMIT = 0.5
# What the fused score is divided by to read 0 to 1: first in both lanes, with a full identifier boost.
FUSED_MAX = 2 / (RRF_K + 1) + IDENTIFIER_BOOST

# Words too common to tell entries apart, English and Spanish, which the demo's memory holds.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by can could did do does for from had has have how i if in into is it its me my no
    not of on or our shall should so than that the their them then there these they this those to too us was we
    were what when where which who whom why will with would you your
    al con de del el en es la las lo los para por que se su un una y
    """.split()
)
# Runs of letters and digits, of any script; underscores split them.
_WORD = re.compile(r"[^\W_]+")
# Where a camelCase or PascalCase word turns: aB, or ABc (the last capital of an acronym starts the next word).
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
# A run of what a code-ish name is written with, before it is told apart from prose.
_CHUNK = re.compile(r"[\w./#:@-]+")
_SNAKE = re.compile(r"[^\W_]*_[^\W_]+(?:_[^\W_]*)*")
_KEBAB = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
_DOTTED = re.compile(r"[A-Za-z_][\w-]*(?:\.[\w-]+)+")
_PATH = re.compile(r"[\w.@-]*/[\w./@-]+")
# Letters and digits together, such as a ticket (OTARI-1234 also reads as kebab), a version or a model.
_WITH_DIGITS = re.compile(r"(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+")
_ISSUE = re.compile(r"#\d+")
# What of a turn the lane reads, from its end, where a question follows what it pastes; and the most distinct terms
# it scores, so a turn the length of a file costs no more than a long question.
QUERY_MAX_CHARS = 8_000
QUERY_MAX_TERMS = 64
# Shorter than this, a match is more likely chance than a name: ``v2``, ``e.g``.
_MIN_IDENTIFIER = 4
_EDGE = ".,;:!?\"'`()[]{}<>"


@dataclass(frozen=True, slots=True)
class EntryTerms:
    """What the lexical lane keeps of one entry: how often each term occurs, and the identifiers it names."""

    counts: Mapping[str, int]
    length: int
    identifiers: frozenset[str]


@dataclass(frozen=True)
class LexicalScores:
    """Each candidate's BM25 score and identifier weight, kept only where above zero."""

    bm25: dict[uuid.UUID, float]
    identifier: dict[uuid.UUID, float]


def identifiers(text: str) -> set[str]:
    """The names ``text`` writes as code would: snake_case, camelCase, kebab-case, dotted, paths, tickets, #123.

    Case-folded, with the punctuation around them stripped; a path also gives its last part, and a dotted name
    each part that is a name too, so ``src/app/_service.py`` matches a question about ``_service.py``.
    """
    found: set[str] = set()
    for raw in _CHUNK.findall(text):
        chunk = raw.strip(_EDGE)
        if chunk.isalpha() and not _mixed(chunk):
            # Most of a text is plain words, which name nothing.
            continue
        if _ISSUE.fullmatch(chunk):
            found.add(chunk)
            continue
        names = [chunk]
        if "/" in chunk and _PATH.fullmatch(chunk):
            # ``./a/b``, ``/a/b`` and ``a/b/`` name the same file as far as a match goes.
            chunk = names[0] = chunk.lstrip("./").rstrip("/")
            names.append(chunk.rsplit("/", 1)[-1])
        elif _DOTTED.fullmatch(chunk):
            names.extend(chunk.split("."))
        for name in names:
            if len(name) >= _MIN_IDENTIFIER and _is_identifier(name):
                found.add(name.casefold())
    return found


def _is_identifier(name: str) -> bool:
    return bool(
        ("/" in name and _PATH.fullmatch(name))
        or _SNAKE.fullmatch(name)
        or (name.isalnum() and _mixed(name))
        or _KEBAB.fullmatch(name)
        or _DOTTED.fullmatch(name)
        or _WITH_DIGITS.fullmatch(name)
    )


def terms(text: str, named: set[str] | None = None) -> list[str]:
    """The terms BM25 counts in ``text``: its words, camelCase and snake_case split, then each identifier whole.

    So ``deploy_api`` matches an entry saying "deploy the api" on two words, and one naming ``deploy_api`` on three.
    ``named`` is ``identifiers(text)`` where the caller has it already. Terms are interned: a workspace's entries
    share most of their words, and each is then held once.
    """
    words = [
        part.casefold() for word in _WORD.findall(text) for part in (_CAMEL.findall(word) if _mixed(word) else (word,))
    ]
    kept = [word for word in words if len(word) > 1 and word not in _STOPWORDS]
    return [sys.intern(term) for term in (*kept, *sorted(identifiers(text) if named is None else named))]


def _mixed(word: str) -> bool:
    return not word.islower() and not word.isupper() and any(char.isupper() for char in word[1:])


def entry_terms(text: str) -> EntryTerms:
    """One entry's terms, as the lexical lane holds them."""
    named = identifiers(text)
    found = terms(text, named)
    return EntryTerms(Counter(found), len(found), frozenset(sys.intern(name) for name in named))


def index_texts(texts: Mapping[uuid.UUID, str]) -> dict[uuid.UUID, EntryTerms]:
    """Each opened text's terms, by entry id. Pure, for a thread; the texts are not kept."""
    return {entry_id: entry_terms(text) for entry_id, text in texts.items()}


def score(entries: Mapping[uuid.UUID, EntryTerms], queries: Sequence[str]) -> LexicalScores:
    """Each entry's best BM25 score and identifier weight over ``queries``. Pure, for a thread.

    Statistics (how many entries hold a term, their mean length, how many share an identifier) are over
    ``entries`` alone: the candidates the member may recall, so an entry they may not recall moves no score.
    """
    bm25: dict[uuid.UUID, float] = {}
    weight: dict[uuid.UUID, float] = {}
    if not entries:
        return LexicalScores(bm25, weight)
    for text in queries:
        query = text[-QUERY_MAX_CHARS:]
        named = sorted(identifiers(query))[:QUERY_MAX_TERMS]
        # Identifiers first, so a long turn's cap drops common words before the names it asks about.
        wanted = set(list(dict.fromkeys([*named, *terms(query)]))[:QUERY_MAX_TERMS])
        for entry_id, value in _bm25(entries, wanted).items():
            bm25[entry_id] = max(value, bm25.get(entry_id, 0.0))
        for entry_id, value in _identifier_weights(entries, set(named)).items():
            weight[entry_id] = max(value, weight.get(entry_id, 0.0))
    return LexicalScores(bm25, weight)


def _bm25(entries: Mapping[uuid.UUID, EntryTerms], query: set[str]) -> dict[uuid.UUID, float]:
    if not query:
        return {}
    total = len(entries)
    mean_length = max(1.0, sum(entry.length for entry in entries.values()) / total)
    holding: dict[str, list[tuple[uuid.UUID, int, int]]] = {term: [] for term in query}
    for entry_id, entry in entries.items():
        for term in query:
            count = entry.counts.get(term)
            if count:
                holding[term].append((entry_id, count, entry.length))
    scores: dict[uuid.UUID, float] = {}
    for held in holding.values():
        if not held:
            continue
        # Lucene's IDF, never negative: a term most entries hold still counts a little.
        idf = math.log(1 + (total - len(held) + 0.5) / (len(held) + 0.5))
        for entry_id, count, length in held:
            norm = BM25_K1 * (1 - BM25_B + BM25_B * length / mean_length)
            scores[entry_id] = scores.get(entry_id, 0.0) + idf * count * (BM25_K1 + 1) / (count + norm)
    return scores


def _identifier_weights(entries: Mapping[uuid.UUID, EntryTerms], query: set[str]) -> dict[uuid.UUID, float]:
    if not query:
        return {}
    holders: dict[str, list[uuid.UUID]] = {name: [] for name in query}
    for entry_id, entry in entries.items():
        for name in query & entry.identifiers:
            holders[name].append(entry_id)
    weights: dict[uuid.UUID, float] = {}
    for held in holders.values():
        damped = popularity_weight(len(held))
        for entry_id in held:
            weights[entry_id] = min(1.0, weights.get(entry_id, 0.0) + damped)
    return weights


def popularity_weight(holders: int) -> float:
    """What one shared identifier weighs when ``holders`` candidates name it."""
    return 1 / (1 + IDENTIFIER_POPULARITY * (holders - 1) ** 2) if holders > 0 else 0.0


def fused_scores(
    ids: Sequence[uuid.UUID], similarities: npt.NDArray[np.float32], lexical: LexicalScores
) -> npt.NDArray[np.float64]:
    """Each of ``ids``' fused score, 0 to 1: its two lanes' reciprocal ranks and its identifier boost.

    ``similarities`` are in the order of ``ids``, and so is what is returned. Every candidate has a dense rank; only
    one with a BM25 score above zero has a lexical one. Ties keep the order of ``ids``.
    """
    count = len(ids)
    dense_rank = np.empty(count, dtype=np.float64)
    dense_rank[np.argsort(-similarities, kind="stable")] = np.arange(1, count + 1)
    fused = 1 / (RRF_K + dense_rank)
    matched = [(index, lexical.bm25[entry_id]) for index, entry_id in enumerate(ids) if entry_id in lexical.bm25]
    # Python's sort is stable, so equal scores keep the candidates' order.
    for rank, (index, _) in enumerate(sorted(matched, key=lambda pair: -pair[1]), start=1):
        fused[index] += 1 / (RRF_K + rank)
    boost = np.fromiter((lexical.identifier.get(entry_id, 0.0) for entry_id in ids), dtype=np.float64, count=count)
    return (fused + IDENTIFIER_BOOST * boost) / FUSED_MAX


def identifier_weights(ids: Iterable[uuid.UUID], lexical: LexicalScores) -> list[float]:
    """Each of ``ids``' identifier weight, 0 where it shares none with the query."""
    return [lexical.identifier.get(entry_id, 0.0) for entry_id in ids]
