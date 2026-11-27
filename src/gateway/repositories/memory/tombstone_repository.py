"""Data access for ``memory_tombstones``. Flushes, never commits."""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryScope, MemoryTombstone
from gateway.repositories.memory.fenced_repository import FencedRepository


@dataclass(frozen=True)
class NewTombstone:
    """What a forgotten entry version leaves behind. Its embedding arrives sealed."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    scope: MemoryScope
    owner_user_id: uuid.UUID | None
    key_id: uuid.UUID
    embedding_sealed: str
    embedding_model: str
    forgotten_at: datetime
    expires_at: datetime


class TombstoneRepository(FencedRepository[MemoryTombstone]):
    """Keep what forgotten entries leave behind until it expires."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryTombstone)

    async def add_all(self, tombstones: Sequence[NewTombstone]) -> None:
        """Stage tombstones, every one of one workspace."""
        if not tombstones:
            return
        await self._fence(tombstones[0].workspace_id)
        self.db.add_all(
            MemoryTombstone(
                id=tombstone.id,
                workspace_id=tombstone.workspace_id,
                scope=tombstone.scope,
                owner_user_id=tombstone.owner_user_id,
                key_id=tombstone.key_id,
                embedding_sealed=tombstone.embedding_sealed,
                embedding_model=tombstone.embedding_model,
                forgotten_at=tombstone.forgotten_at,
                expires_at=tombstone.expires_at,
            )
            for tombstone in tombstones
        )
        await self.db.flush()

    async def live(
        self,
        workspace_id: uuid.UUID,
        *,
        scope: MemoryScope,
        owner_user_id: uuid.UUID | None,
        embedding_model: str,
        now: datetime,
    ) -> Sequence[MemoryTombstone]:
        """The unexpired tombstones a proposal of ``scope`` is compared with: same scope, same owner, same model."""
        await self._fence(workspace_id)
        query = select(MemoryTombstone).where(
            MemoryTombstone.workspace_id == workspace_id,
            MemoryTombstone.scope == scope,
            MemoryTombstone.embedding_model == embedding_model,
            MemoryTombstone.expires_at > now,
        )
        if owner_user_id is not None:
            query = query.where(MemoryTombstone.owner_user_id == owner_user_id)
        result = await self.db.execute(query.order_by(MemoryTombstone.forgotten_at, MemoryTombstone.id))
        return result.scalars().all()

    async def lift(self, workspace_id: uuid.UUID, tombstone_ids: Collection[uuid.UUID]) -> int:
        """Delete these tombstones, as an admin let what they kept out back in; how many the workspace had."""
        if not tombstone_ids:
            return 0
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryTombstone)
            .where(MemoryTombstone.workspace_id == workspace_id, MemoryTombstone.id.in_(list(tombstone_ids)))
            .returning(MemoryTombstone.id)
        )
        return len(result.scalars().all())

    async def delete_expired(self, workspace_id: uuid.UUID, *, now: datetime) -> int:
        """Delete the workspace's tombstones whose expiry has passed."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryTombstone)
            .where(MemoryTombstone.workspace_id == workspace_id, MemoryTombstone.expires_at <= now)
            .returning(MemoryTombstone.id)
        )
        return len(result.scalars().all())

    async def key_ids_in_use(self, workspace_id: uuid.UUID) -> set[uuid.UUID]:
        """The data keys some tombstone of the workspace is sealed under."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            select(MemoryTombstone.key_id).where(MemoryTombstone.workspace_id == workspace_id).distinct()
        )
        return set(result.scalars().all())

    async def delete_all(self, workspace_id: uuid.UUID) -> int:
        """Delete every tombstone of the workspace."""
        await self._fence(workspace_id)
        result = await self.db.execute(
            delete(MemoryTombstone).where(MemoryTombstone.workspace_id == workspace_id).returning(MemoryTombstone.id)
        )
        return len(result.scalars().all())
