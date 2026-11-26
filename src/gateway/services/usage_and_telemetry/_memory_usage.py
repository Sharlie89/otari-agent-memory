"""Agent memory's own model calls as usage rows, and the usage its savings experiment reads.

Memory names what it needs as ``MemoryMeter`` and ``UsageLedger``; the usage rows are this domain's, so these
implement both on its repository.
"""

import uuid
from datetime import datetime

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.usage_and_telemetry import UsageLogRepository
from gateway.services.memory import MemoryCallUsage, MemoryModels, ModelUse, SessionUsage


class MemoryUsageMeter:
    """Memory's model calls as usage rows of their workspace, staged in the memory service's own block."""

    def __init__(self, uow: UnitOfWork) -> None:
        self._usage = UsageLogRepository(uow)

    async def record(self, workspace_id: uuid.UUID, api_key_id: str | None, use: ModelUse) -> None:
        await self._usage.record_side_call(
            workspace_id=workspace_id,
            api_key_id=api_key_id,
            model=use.model,
            provider=use.provider,
            endpoint=use.endpoint,
            succeeded=use.succeeded,
            prompt_tokens=use.prompt_tokens,
            completion_tokens=use.completion_tokens,
            latency_ms=use.latency_ms,
        )


class MemoryUsageLedger:
    """A workspace's usage rows, as memory's savings experiment reads them, each read in a block of its own."""

    def __init__(self, uow: UnitOfWork) -> None:
        self._uow = uow
        self._usage = UsageLogRepository(uow)

    async def sessions(self, workspace_id: uuid.UUID, since: datetime, until: datetime) -> dict[str, SessionUsage]:
        async with self._uow:
            totals = await self._usage.session_totals(
                workspace_id, since=since, until=until, excluding_endpoints=MemoryModels.ENDPOINT_PREFIX
            )
        return {
            row.session: SessionUsage(
                requests=row.requests,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cost=float(row.cost) if row.cost is not None else None,
            )
            for row in totals
        }

    async def memory_calls(
        self, workspace_id: uuid.UUID, since: datetime, until: datetime
    ) -> dict[str, MemoryCallUsage]:
        async with self._uow:
            totals = await self._usage.endpoint_totals(
                workspace_id, since=since, until=until, endpoints=MemoryModels.ENDPOINT_PREFIX
            )
        return {row.endpoint: MemoryCallUsage(row.calls, row.input_tokens, row.output_tokens) for row in totals}
