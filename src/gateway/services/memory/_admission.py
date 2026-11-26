"""What memory one request through the gateway gets: its tools, automatic recall, and its experiment arm."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Protocol

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions import TenancyError
from gateway.exceptions.memory_exceptions import MEMORY_OFF, NO_READ_SCOPE, NOT_A_MEMBER, MemoryToolsRefusedError
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.services.memory._experiment import ExperimentArm
from gateway.services.memory._service import MemoryCaller, MemoryService
from gateway.services.memory._tool_backend import MemoryToolBackend
from gateway.services.memory._tools import MemoryTools, ToolOutcome
from gateway.types.key_scope import KeyScope


class CallerLookup(Protocol):
    """The member a key speaks for, in the key's own workspace; None for a key that names no active member."""

    async def __call__(self, api_key: APIKey, *, session_id: str | None, model: str | None) -> MemoryCaller | None:
        """Resolve the caller on the request's own reads."""
        ...


@dataclass(frozen=True)
class AdmittedMemory:
    """What memory one request through the gateway gets.

    ``tools`` are memory's tools for its loop, ``recall`` what memory holds on its newest turn, given that turn
    and the user's turns before it (the text to add, and how many entries), and ``arm`` the experiment arm of
    its session.
    """

    tools: MemoryToolBackend | None
    recall: Callable[[str, str | None], Awaitable[tuple[str | None, int]]] | None
    arm: ExperimentArm | None


class MemoryToolAdmission:
    """Decides whether a request through the gateway gets memory's tools, and builds them for its caller.

    A request asks for them by declaring ``{"type": "otari_memory"}`` or sending ``Otari-Memory: on``, and
    gets them unasked where its workspace offers memory to every request, unless it sends ``Otari-Memory: off``.
    Asking where memory cannot be had is refused; not asking never is.

    ``build_service`` builds the memory service on a Unit of Work, and ``caller_for`` resolves a key's caller.
    """

    # Longest automatic recall may hold a request up, embedding included.
    RECALL_BUDGET_S: ClassVar[float] = 3.0

    def __init__(self, build_service: Callable[[UnitOfWork], MemoryService], caller_for: CallerLookup) -> None:
        self._build_service = build_service
        self._caller_for = caller_for

    async def admit(
        self,
        api_key: APIKey | None,
        *,
        asked: bool,
        declined: bool,
        session_label: str | None = None,
        model: str | None = None,
    ) -> AdmittedMemory | None:
        """What memory this request gets, or None when it gets none.

        ``session_label`` is the trace id of the request's session: its experiment arm, and the session its
        proposals name. A session in the experiment's control arm gets no memory, even where it asked.

        Raises:
            MemoryToolsRefusedError: the request asked for memory's tools and its caller cannot have them.
        """
        if declined:
            return None
        caller = await self._caller_for(api_key, session_id=session_label, model=model) if api_key else None
        if caller is None:
            if asked:
                raise MemoryToolsRefusedError(NOT_A_MEMBER)
            return None
        if api_key is not None and KeyScope.MEMORY_READ not in api_key.scopes:
            # A key without memory:read gets neither memory's tools nor what automatic recall would add.
            if asked:
                raise MemoryToolsRefusedError(NO_READ_SCOPE)
            return None
        async with create_unit_of_work() as uow:
            offer = await self._build_service(uow).offer(caller, session_label)
        if not offer.on:
            if asked:
                raise MemoryToolsRefusedError(MEMORY_OFF)
            return None
        if offer.arm == "control":
            return AdmittedMemory(tools=None, recall=None, arm="control")
        tools = MemoryToolBackend(caller, self._run) if asked or offer.every_request else None
        recall = partial(self._recall, caller) if offer.auto_recall else None
        if tools is None and recall is None and offer.arm is None:
            return None
        return AdmittedMemory(tools=tools, recall=recall, arm=offer.arm)

    async def _run(self, caller: MemoryCaller, name: str, arguments: dict[str, Any]) -> ToolOutcome | None:
        # A Unit of Work per call: the request's own session is released before the provider is called.
        async with create_unit_of_work() as uow:
            return await MemoryTools(self._build_service(uow)).call(caller, name, arguments)

    async def _recall(self, caller: MemoryCaller, query: str, context: str | None) -> tuple[str | None, int]:
        # Best effort: a request is served without what memory holds rather than failed or held up for it.
        try:
            async with asyncio.timeout(self.RECALL_BUDGET_S), create_unit_of_work() as uow:
                return await MemoryTools(self._build_service(uow)).recall_for_turn(caller, query, context)
        except TimeoutError:
            logger.warning("Automatic recall took longer than %ss; the request goes without it", self.RECALL_BUDGET_S)
        except TenancyError as exc:
            logger.info("Automatic recall found nothing to hand over: %s", exc.message)
        except DATABASE_ERRORS as exc:
            logger.warning("Automatic recall failed on the database: %s", describe_database_error(exc))
        return None, 0
