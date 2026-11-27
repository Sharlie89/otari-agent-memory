"""Recall's lexical lane: what counts as a term and an identifier, BM25, the popularity penalty, and the fusion."""

import asyncio
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Self, cast

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from gateway.adapters.memory_ranking_adapter import RANK_PATH, HttpMemoryRanking
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.memory_ranking_port import RankCandidate, RankedCandidate
from gateway.schemas.memory import EngineRankRequest, EngineRankResponse, SearchArguments
from gateway.services.memory import RecallVectors, _ranking
from gateway.services.memory._lexical import (
    FUSED_MAX,
    IDENTIFIER_ADMIT,
    IDENTIFIER_BOOST,
    QUERY_MAX_TERMS,
    RRF_K,
    LexicalScores,
    entry_terms,
    fused_scores,
    identifiers,
    index_texts,
    popularity_weight,
    score,
    terms,
)
from gateway.services.memory._ranking import LEXICAL_FALLBACKS, LocalMemoryRanking, rank_vectors
from gateway.services.memory._recall_vectors import RecallTerms
from gateway.services.memory._store import EntryStore

WORKSPACE = uuid.UUID(int=99)


def _ids(ranked: Sequence[RankedCandidate]) -> list[uuid.UUID]:
    return [entry.id for entry in ranked]


# -- What counts as an identifier, and as a term ------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Run deploy_api.sh after the build", {"deploy_api.sh", "deploy_api"}),
        ("MemoryService.recall() reads it", {"memoryservice.recall", "memoryservice"}),
        ("HTTPServer and getUserById", {"httpserver", "getuserbyid"}),
        ("see ./src/gateway/services/memory/_service.py.", {"src/gateway/services/memory/_service.py", "_service.py"}),
        ("Fixed in OTARI-1234, see #456", {"otari-1234", "#456"}),
        ("pin v2.13.0 of gpt4o-mini", {"v2.13.0", "gpt4o-mini"}),
        ("make deploy-api from main", {"deploy-api"}),
        # Prose names nothing: plain words, sentence ends, e.g., short versions.
        ("Deploy the API from Main, e.g. on Fridays. Lucía said v2.", set()),
    ],
)
def test_identifiers_are_the_names_a_text_writes_as_code_would(text: str, expected: set[str]) -> None:
    assert identifiers(text) == expected


def test_terms_split_code_names_into_words_and_keep_each_name_whole() -> None:
    assert terms("The deploy_api script calls getUserById") == [
        "deploy",
        "api",
        "script",
        "calls",
        "get",
        "user",
        "id",
        "deploy_api",
        "getuserbyid",
    ]
    assert terms("the of and a") == [], "stopwords tell nothing apart"


# -- BM25 and the identifier weight -------------------------------------------------


A, B, C, D, E = (uuid.UUID(int=n) for n in range(1, 6))


def test_bm25_prefers_the_rarer_term_and_counts_only_the_candidates() -> None:
    entries = index_texts(
        {
            A: "the billing worker retries payments",
            B: "the billing worker sends invoices",
            C: "the search worker indexes documents",
        }
    )

    scores = score(entries, ["billing invoices"])

    assert scores.bm25[B] > scores.bm25[A] > 0, "invoices is rarer than billing"
    assert C not in scores.bm25, "no term in common, no score"
    more = {**entries, D: entry_terms("invoices invoices invoices everywhere")}
    assert score(more, ["billing invoices"]).bm25[B] < scores.bm25[B], "statistics are over the entries given"


def test_the_popularity_penalty_damps_an_identifier_many_entries_share() -> None:
    assert [round(popularity_weight(n), 3) for n in (1, 2, 3, 5, 10)] == [1.0, 0.667, 0.333, 0.111, 0.024]
    assert popularity_weight(0) == 0.0

    entries = index_texts(
        {
            A: "svc_billing restarts nightly",
            B: "svc_billing owns the ledger_sync job",
            C: "svc_billing pages the on-call",
            D: "nothing in common",
        }
    )
    weights = score(entries, ["is svc_billing up?"]).identifier
    assert weights == {A: pytest.approx(1 / 3), B: pytest.approx(1 / 3), C: pytest.approx(1 / 3)}
    rare = score(entries, ["what runs ledger_sync for svc_billing?"]).identifier
    assert rare[B] == pytest.approx(1.0), "the rare name counts fully, capped at one"
    assert rare[A] == pytest.approx(1 / 3)


def test_a_long_turn_scores_its_names_before_its_words() -> None:
    entries = index_texts({A: "rotate key_rotation_job weekly"})
    padding = " ".join(f"word{n}x" for n in range(QUERY_MAX_TERMS * 2))

    assert A in score(entries, [f"{padding} key_rotation_job"]).bm25


def test_scoring_is_deterministic() -> None:
    entries = index_texts({A: "svc_a restart", B: "svc_b restart", C: "restart all services"})
    assert score(entries, ["restart svc_a"]) == score(entries, ["restart svc_a"])


# -- Fusion ----------------------------------------------------------------------


def test_fusion_adds_reciprocal_ranks_and_the_boost_and_reads_zero_to_one() -> None:
    ids = [A, B, C]
    similarities = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    lexical = LexicalScores(bm25={A: 3.0, C: 1.0}, identifier={A: 1.0})

    fused = fused_scores(ids, similarities, lexical)

    assert fused[0] == pytest.approx(1.0), "first in both lanes with a full boost is the most there is"
    assert fused[1] == pytest.approx((1 / (RRF_K + 2)) / FUSED_MAX)
    assert fused[2] == pytest.approx((1 / (RRF_K + 3) + 1 / (RRF_K + 2)) / FUSED_MAX)
    assert IDENTIFIER_BOOST == pytest.approx(1 / (RRF_K + 1))


def test_an_identifier_the_dense_lane_misses_is_found_first() -> None:
    """An embedding reads ``svc_ledger_rotation`` as a few common sub-words; the entry naming it is not alike."""
    texts = {
        A: "Restart a service with make restart after a deploy",
        B: "The worker service restarts itself after a deploy",
        C: "svc_ledger_rotation runs from cron at 02:00 UTC",
        D: "Deploy services from main",
    }
    candidates = [RankCandidate(entry_id, 0.5) for entry_id in texts]
    # The turn's embedding is about restarting services; the entry it names is about something else.
    vectors = {A: [1.0, 0.1], B: [0.9, 0.2], C: [0.1, 1.0], D: [0.7, 0.3]}
    turn = "when does svc_ledger_rotation restart?"
    lexical = score(index_texts(texts), [turn])

    dense = rank_vectors(candidates, vectors, [[1.0, 0.0]], order="similarity", keep_top=2, keep_above=0.8)
    hybrid = rank_vectors(
        candidates, vectors, [[1.0, 0.0]], order="similarity", keep_top=2, keep_above=0.8, lexical=lexical
    )

    assert C not in _ids(dense)
    assert _ids(hybrid)[0] == C
    assert hybrid[0].identifier == pytest.approx(1.0) and hybrid[0].identifier >= IDENTIFIER_ADMIT
    assert hybrid[0].similarity < 0.2, "similarity still reports the embedding's own"
    assert all(entry.score is not None for entry in hybrid)
    assert all(entry.score is None for entry in dense)


def test_an_identifier_match_is_kept_past_the_top_and_relevance_weighs_confidence() -> None:
    candidates = [RankCandidate(A, 0.9), RankCandidate(B, 0.9), RankCandidate(C, 0.1)]
    vectors = {A: [1.0, 0.0], B: [0.9, 0.1], C: [0.0, 1.0]}
    lexical = LexicalScores(bm25={}, identifier={C: 0.6})

    kept = rank_vectors(
        candidates, vectors, [[1.0, 0.0]], order="relevance", keep_top=1, keep_above=None, lexical=lexical
    )

    assert _ids(kept) == [A, C], "C shares a name with the turn, so it is kept beside the top one"


# -- The lane in a process: held terms, and falling back to similarity ---------------


class _Uow:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _Store:
    """Opens what a test hands it, counting what it was asked."""

    def __init__(self, vectors: Mapping[uuid.UUID, list[float]], texts: Mapping[uuid.UUID, str]) -> None:
        self.vectors = vectors
        self.texts = texts
        self.texts_opened: list[uuid.UUID] = []
        self.fail_texts = False

    async def open_embeddings(self, workspace_id: uuid.UUID, ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, Any]:
        return {entry_id: np.array(self.vectors[entry_id], dtype=np.float32) for entry_id in ids}

    async def open_texts(self, workspace_id: uuid.UUID, ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, str]:
        if self.fail_texts:
            raise OSError("the database went away")
        self.texts_opened.extend(ids)
        return {entry_id: self.texts[entry_id] for entry_id in ids}


def _local(store: _Store, held: RecallVectors | None) -> LocalMemoryRanking:
    return LocalMemoryRanking(cast(UnitOfWork, _Uow()), cast(EntryStore, store), held)


TEXTS = {A: "restart the worker after a deploy", B: "svc_ledger_rotation runs nightly"}
VECTORS = {A: [1.0, 0.0], B: [0.0, 1.0]}
CANDIDATES = [RankCandidate(A, 0.5), RankCandidate(B, 0.5)]


def test_terms_are_opened_once_and_held_beside_the_embeddings_until_erased() -> None:
    store, held = _Store(VECTORS, TEXTS), RecallVectors()
    ranking = _local(store, held)

    async def rank() -> list[RankedCandidate]:
        return await ranking.rank(
            WORKSPACE, CANDIDATES, [[1.0, 0.0]], order="similarity", keep_top=1, texts=["svc_ledger_rotation?"]
        )

    first = asyncio.run(rank())
    second = asyncio.run(rank())

    assert _ids(first) == _ids(second) == [B]
    assert sorted(store.texts_opened) == [A, B], "the second turn opened no text"
    assert len(held.terms) == 2
    held.forget(WORKSPACE)
    assert len(held.terms) == 0


def test_without_a_cache_the_lane_indexes_for_the_turn_alone() -> None:
    store = _Store(VECTORS, TEXTS)
    ranked = asyncio.run(
        _local(store, None).rank(
            WORKSPACE, CANDIDATES, [[1.0, 0.0]], order="similarity", keep_top=1, texts=["svc_ledger_rotation?"]
        )
    )
    assert _ids(ranked) == [B]


def test_a_lexical_lane_that_fails_ranks_by_similarity_alone() -> None:
    store = _Store(VECTORS, TEXTS)
    store.fail_texts = True
    before = LEXICAL_FALLBACKS._value.get()

    ranked = asyncio.run(
        _local(store, RecallVectors()).rank(
            WORKSPACE, CANDIDATES, [[1.0, 0.0]], order="similarity", keep_top=1, texts=["svc_ledger_rotation?"]
        )
    )

    assert _ids(ranked) == [A] and ranked[0].score is None
    assert LEXICAL_FALLBACKS._value.get() == before + 1


def test_a_slow_index_ranks_by_similarity_now_and_is_held_for_the_next_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    def slow(texts: Mapping[uuid.UUID, str]) -> dict[uuid.UUID, Any]:
        time.sleep(0.2)
        return index_texts(texts)

    monkeypatch.setattr(_ranking, "LEXICAL_BUDGET_S", 0.05)
    monkeypatch.setattr(_ranking, "index_texts", slow)
    store, held = _Store(VECTORS, TEXTS), RecallVectors()
    ranking = _local(store, held)

    async def two_turns() -> tuple[list[RankedCandidate], list[RankedCandidate]]:
        args: dict[str, Any] = {"order": "similarity", "keep_top": 1, "texts": ["svc_ledger_rotation?"]}
        first = await ranking.rank(WORKSPACE, CANDIDATES, [[1.0, 0.0]], **args)
        await asyncio.sleep(0.3)
        second = await ranking.rank(WORKSPACE, CANDIDATES, [[1.0, 0.0]], **args)
        return first, second

    first, second = asyncio.run(two_turns())

    assert _ids(first) == [A], "out of time, similarity ranks"
    assert _ids(second) == [B], "what the first turn indexed is held"
    assert sorted(store.texts_opened) == [A, B]


def test_held_terms_expire_and_the_least_recently_used_workspace_goes_first() -> None:
    now = [0.0]
    terms_held = RecallTerms(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    other = uuid.UUID(int=98)
    terms_held.keep(WORKSPACE, {A: entry_terms("a"), B: entry_terms("b")})
    now[0] = 5
    assert terms_held.missing(WORKSPACE, [A, B, C]) == [C]

    terms_held.keep(other, {C: entry_terms("c")})
    assert terms_held.view(WORKSPACE, [A, B]) == {}, "past the cap, the workspace used longest ago went"
    now[0] = 16
    assert terms_held.missing(other, [C]) == [C], "expired"
    assert len(terms_held) == 0


# -- Handed to a memory service ----------------------------------------------------


def _service(handler: Any) -> HttpMemoryRanking:
    return HttpMemoryRanking("http://memory-service/", "t" * 40, transport=httpx.MockTransport(handler))


def test_the_gateway_sends_the_turns_words_and_reads_the_fused_score_back() -> None:
    seen: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == RANK_PATH
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"ranked": [[str(B), 0.5, 0.1, 0.9, 1.0], [str(A), 0.5, 0.9, None, 0.0]]})

    ranked = asyncio.run(
        _service(answer).rank(WORKSPACE, CANDIDATES, [[1.0, 0.0]], order="similarity", keep_top=2, texts=["q"])
    )

    assert seen[0]["texts"] == ["q"]
    assert ranked == [RankedCandidate(B, 0.5, 0.1, 0.9, 1.0), RankedCandidate(A, 0.5, 0.9, None, 0.0)]


def test_a_memory_service_older_than_the_lane_is_asked_again_without_it() -> None:
    seen: list[dict[str, Any]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "texts" in body:
            return httpx.Response(422, json={"detail": "extra fields not permitted"})
        return httpx.Response(200, json={"ranked": [[str(A), 0.5, 0.9]]})

    ranked = asyncio.run(
        _service(answer).rank(WORKSPACE, CANDIDATES, [[1.0, 0.0]], order="similarity", keep_top=2, texts=["q"])
    )

    assert ranked == [RankedCandidate(A, 0.5, 0.9)]
    assert ["texts" in body for body in seen] == [True, False]


def test_the_engine_answers_a_gateway_in_the_shape_it_asked_for() -> None:
    three = EngineRankResponse(ranked=[(A, 0.5, 0.9)])
    five = EngineRankResponse(ranked=[(A, 0.5, 0.9, None, 0.0)])
    assert len(three.model_dump(mode="json")["ranked"][0]) == 3
    assert len(five.model_dump(mode="json")["ranked"][0]) == 5
    with pytest.raises(ValidationError):
        EngineRankRequest.model_validate(
            {
                "workspace_id": str(WORKSPACE),
                "candidates": [],
                "queries": [[1.0]],
                "order": "similarity",
                "keep_top": 1,
                "texts": ["x" * 20_000],
            }
        )


# -- memory_search's filters ------------------------------------------------------------


def test_search_takes_one_kind_or_several_and_a_window_read_as_utc() -> None:
    one = SearchArguments.model_validate({"query": "q", "kind": "fact"})
    several = SearchArguments.model_validate(
        {
            "query": "q",
            "kind": ["procedure", "fact", "fact"],
            "since": "2026-01-31",
            "until": "2026-02-01T10:00:00+02:00",
        }
    )

    assert one.kinds == ["fact"]
    assert several.kinds == ["procedure", "fact"]
    assert several.since is not None and several.since.utcoffset() is not None
    assert several.since.isoformat() == "2026-01-31T00:00:00+00:00"
    assert SearchArguments.model_validate({"query": "q"}).kinds is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "q", "kind": []},
        {"query": "q", "kind": "lesson"},
        {"query": "q", "since": "2026-02-02", "until": "2026-02-01"},
        {"query": "q", "since": "yesterday"},
        {"query": "q", "tag": "billing"},
    ],
    ids=["no-kind", "unknown-kind", "window-backwards", "not-a-time", "unknown-field"],
)
def test_search_refuses_a_filter_it_cannot_apply(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SearchArguments.model_validate(arguments)
