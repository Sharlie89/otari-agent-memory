"""Recall's ranking in this process: open the candidates' embeddings, compare them with the turn, best first.

The candidates' embeddings are compared with the turn as one matrix product, on a worker thread, so a workspace
with thousands of entries holds up no other request the process serves. The opened embeddings are kept in
``held`` where the process keeps any; otherwise they are opened for the turn and dropped after it.

Given the turn's words, the lexical lane (``_lexical``) runs beside it: the candidates' texts are opened once,
reduced to their terms, and held with the embeddings; the two lanes' ranks are then fused. Any failure of that
lane, or its taking longer than ``LEXICAL_BUDGET_S``, ranks by similarity alone, as before it existed.

A memory service runs this same ranking for the gateways that hand theirs off.
"""

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Counter
from gateway.ports.memory_ranking_port import MemoryRankingPort, RankCandidate, RankedCandidate, RankOrder
from gateway.services.memory._lexical import (
    IDENTIFIER_ADMIT,
    EntryTerms,
    LexicalScores,
    fused_scores,
    identifier_weights,
    index_texts,
    score,
)
from gateway.services.memory._recall_vectors import HeldView, Matrix, RecallTerms, RecallVectors, unit_rows
from gateway.services.memory._store import EntryStore
from gateway.services.memory._timing import timed

# The longest the lexical lane may take, a cold workspace's texts opened and indexed included, before ranking goes
# by similarity alone: well inside automatic recall's three seconds and a memory service's two.
LEXICAL_BUDGET_S = 1.0

LEXICAL_FALLBACKS = Counter(
    "gateway_memory_lexical_fallbacks",
    "Rankings that went by similarity alone because the lexical lane failed or ran out of time",
    registry=REGISTRY,
)


class LocalMemoryRanking(MemoryRankingPort):
    """Ranks on the caller's Unit of Work, opening embeddings, and texts for the lexical lane, through its store."""

    def __init__(self, uow: UnitOfWork, store: EntryStore, held: RecallVectors | None) -> None:
        self._uow = uow
        self._store = store
        self._held = held
        self._terms: RecallTerms | None = held.terms if held is not None else None

    async def rank(
        self,
        workspace_id: uuid.UUID,
        candidates: Sequence[RankCandidate],
        queries: Sequence[Sequence[float]],
        *,
        order: RankOrder,
        keep_top: int,
        keep_above: float | None = None,
        texts: Sequence[str] | None = None,
    ) -> list[RankedCandidate]:
        if not candidates or not queries:
            return []
        ids = [candidate.id for candidate in candidates]
        with timed("open"):
            missing = self._held.missing(workspace_id, ids) if self._held is not None else ids
            opened = {}
            if missing:
                async with self._uow:
                    opened = await self._store.open_embeddings(workspace_id, missing)
        lexical = await self._lexical(workspace_id, ids, texts) if texts else None
        with timed("score"):
            if self._held is None:
                return await asyncio.to_thread(
                    rank_vectors,
                    candidates,
                    opened,
                    queries,
                    order=order,
                    keep_top=keep_top,
                    keep_above=keep_above,
                    lexical=lexical,
                )
            self._held.keep(workspace_id, opened)
            view = self._held.view(workspace_id, ids)
            return await asyncio.to_thread(
                rank_view,
                candidates,
                view,
                queries,
                order=order,
                keep_top=keep_top,
                keep_above=keep_above,
                lexical=lexical,
            )

    async def forget(self, workspace_id: uuid.UUID) -> None:
        if self._held is not None:
            self._held.forget(workspace_id)

    async def _lexical(
        self, workspace_id: uuid.UUID, ids: Sequence[uuid.UUID], texts: Sequence[str]
    ) -> LexicalScores | None:
        """The lexical lane's scores for the candidates; None where it failed or ran out of time."""
        try:
            with timed("lexical"):
                async with asyncio.timeout(LEXICAL_BUDGET_S):
                    entries = await self._entry_terms(workspace_id, ids)
                    return await asyncio.to_thread(score, entries, texts)
        except Exception as exc:  # noqa: BLE001 - whatever stops the lexical lane, similarity still ranks
            LEXICAL_FALLBACKS.inc()
            logger.warning("Memory ranked by similarity alone; its lexical lane failed (%s)", type(exc).__name__)
            return None

    async def _entry_terms(self, workspace_id: uuid.UUID, ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, EntryTerms]:
        """The candidates' terms: those held, and those of the rest opened and indexed now."""
        missing = self._terms.missing(workspace_id, ids) if self._terms is not None else list(ids)
        made: dict[uuid.UUID, EntryTerms] = {}
        if missing:
            async with self._uow:
                texts = await self._store.open_texts(workspace_id, missing)
            indexing = asyncio.ensure_future(asyncio.to_thread(index_texts, texts))
            if self._terms is not None:
                # Kept once made, even where this turn stopped waiting, so the next turn finds them.
                indexing.add_done_callback(partial(_keep_terms, self._terms, workspace_id))
            made = await asyncio.shield(indexing)
        if self._terms is None:
            return made
        return self._terms.view(workspace_id, ids)


def _keep_terms(
    held: RecallTerms, workspace_id: uuid.UUID, done: "asyncio.Future[dict[uuid.UUID, EntryTerms]]"
) -> None:
    if not done.cancelled() and done.exception() is None:
        held.keep(workspace_id, done.result())


def rank_vectors(
    candidates: Sequence[RankCandidate],
    vectors: Mapping[uuid.UUID, Sequence[float] | npt.NDArray[np.float32]],
    queries: Sequence[Sequence[float]],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
    lexical: LexicalScores | None = None,
) -> list[RankedCandidate]:
    """What :meth:`LocalMemoryRanking.rank` returns, from embeddings opened for this turn. Pure, for a thread.

    A candidate with no vector in ``vectors`` is left out, and one of another size than the turn's is as unlike
    it as can be (similarity 0).
    """
    present = [candidate.id for candidate in candidates if candidate.id in vectors]
    if not present:
        return []
    sizes = {len(vectors[entry_id]) for entry_id in present}
    if len(sizes) == 1:
        view = HeldView(unit_rows([vectors[entry_id] for entry_id in present]), np.arange(len(present)), present)
        return rank_view(
            candidates, view, queries, order=order, keep_top=keep_top, keep_above=keep_above, lexical=lexical
        )
    # Embeddings of more than one size, as while a deployment changes embedding model: each size apart.
    similarity: dict[uuid.UUID, float] = {}
    for size in sizes:
        alike = [entry_id for entry_id in present if len(vectors[entry_id]) == size]
        matrix = unit_rows([vectors[entry_id] for entry_id in alike])
        similarity.update(zip(alike, _similarities(matrix, np.arange(len(alike)), queries).tolist(), strict=True))
    similarities = np.array([similarity[entry_id] for entry_id in present], dtype=np.float32)
    return _ordered(
        candidates, present, similarities, order=order, keep_top=keep_top, keep_above=keep_above, lexical=lexical
    )


def rank_view(
    candidates: Sequence[RankCandidate],
    view: HeldView | None,
    queries: Sequence[Sequence[float]],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
    lexical: LexicalScores | None = None,
) -> list[RankedCandidate]:
    """What :meth:`LocalMemoryRanking.rank` returns, from the embeddings a process holds. Pure, for a thread."""
    if view is None:
        return []
    similarities = _similarities(view.matrix, view.rows, queries)
    return _ordered(
        candidates, view.ids, similarities, order=order, keep_top=keep_top, keep_above=keep_above, lexical=lexical
    )


def _similarities(
    matrix: Matrix, rows: npt.NDArray[np.intp], queries: Sequence[Sequence[float]]
) -> npt.NDArray[np.float32]:
    """Each of ``rows``' best cosine similarity to any of ``queries``; a query of another size counts as 0."""
    dimensions = matrix.shape[1]
    fitting = [query for query in queries if len(query) == dimensions]
    if not fitting:
        return np.zeros(len(rows), dtype=np.float32)
    # The whole matrix by the queries, then the rows asked: no copy of the rows themselves.
    best = (matrix @ unit_rows(fitting).T)[rows].max(axis=1)
    if len(fitting) < len(queries):
        best = np.maximum(best, 0.0)
    return best


def _ordered(
    candidates: Sequence[RankCandidate],
    ids: Sequence[uuid.UUID],
    similarities: npt.NDArray[np.float32],
    *,
    order: RankOrder,
    keep_top: int,
    keep_above: float | None,
    lexical: LexicalScores | None = None,
) -> list[RankedCandidate]:
    """``ids``, with their similarities, best first by ``order``: the first ``keep_top``, and any alike enough.

    ``ids`` are in the order of ``candidates``, and ties keep it. Only what is kept becomes a result. With
    ``lexical``, the fused score orders instead of similarity, and an entry sharing identifiers enough with the
    turn is kept as one alike enough is.
    """
    confidence = {candidate.id: candidate.confidence for candidate in candidates}
    confidences = np.fromiter((confidence[entry_id] for entry_id in ids), dtype=np.float32, count=len(ids))
    fused = fused_scores(ids, similarities, lexical) if lexical is not None else None
    weights = np.array(identifier_weights(ids, lexical)) if lexical is not None else np.zeros(len(ids))
    key: npt.NDArray[np.floating[Any]]
    if fused is not None:
        key = fused if order == "similarity" else fused * confidences
    else:
        key = similarities if order == "similarity" else np.maximum(similarities, 0.0) * confidences
    best_first = np.argsort(-key, kind="stable")
    kept = np.arange(len(best_first)) < keep_top
    if keep_above is not None:
        kept |= similarities[best_first] >= keep_above
    if fused is not None:
        kept |= weights[best_first] >= IDENTIFIER_ADMIT
    return [
        RankedCandidate(
            ids[index],
            confidence[ids[index]],
            float(similarities[index]),
            score=float(fused[index]) if fused is not None else None,
            identifier=float(weights[index]),
        )
        for index in best_first[kept].tolist()
    ]
