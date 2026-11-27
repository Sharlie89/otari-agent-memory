"""Where agent memory is kept: Otari's own memory, or another company's memory service a deployment connects.

Otari stays the control plane whichever backend keeps the memory. It resolves who
the caller is and what every level allows (deployment, organization, workspace,
member, key and request), screens what is written for credentials and personal
data, runs the organization's guardrails, holds writes for review where the
workspace asks it to and the backend cannot, meters every call and bounds how long
recall may take. A backend is handed only calls Otari already authorized, for the
workspace and member :class:`MemoryContext` names; it never decides who may see
what, and an agent never names a workspace or member to it.

Item ids are opaque strings the backend chose. Otari passes them back as it got
them and never builds or parses one.

What a backend can do beyond the required calls it declares as
:class:`MemoryCapability`; asked for one it did not declare, it raises
:class:`MemoryCapabilityError`. Any other failure crosses as
:class:`MemoryBackendError`: automatic recall then goes on without memory, and a
tool call answers that memory is unavailable.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Literal, Protocol

MemoryItemKind = Literal["fact", "decision", "procedure"]
MemoryItemScope = Literal["workspace", "member"]
MemoryFlagReason = Literal["wrong", "stale", "duplicate", "unsafe"]
WriteState = Literal["accepted", "pending", "held"]


class MemoryCapability(StrEnum):
    """What a backend can do beyond recall, search, get, add, forget and erase, which every backend must."""

    MEMBER_SCOPE = "member_scope"
    """Keeps a member's own memory apart from the workspace's, recalled by that member alone."""
    FEEDBACK = "feedback"
    """Takes confirm and flag, and weighs what it returns by them."""
    HISTORY = "history"
    """``get`` returns the versions an item replaced."""
    SEARCH_FILTERS = "search_filters"
    """``search`` narrows by kind and by when an item was recorded."""
    REVIEW = "review"
    """Holds a write for an admin itself. Without it, Otari holds writes the workspace's policy reviews."""
    TOMBSTONES = "tombstones"
    """Keeps a forgotten fact from coming back when an agent writes it again."""
    CONSOLIDATION = "consolidation"
    """Decides whether a write adds, updates or changes nothing, rather than appending every write."""


class MemoryBackendError(Exception):
    """The backend could not answer: unreachable, too slow, or it refused the call."""


class MemoryCapabilityError(MemoryBackendError):
    """The call needs a capability the backend did not declare."""

    def __init__(self, capability: MemoryCapability) -> None:
        super().__init__(f"The memory backend does not support {capability.value}")
        self.capability = capability


class MemoryItemNotFoundError(MemoryBackendError):
    """No item with that id that the context's member may see."""


@dataclass(frozen=True)
class MemoryContext:
    """Who a call is for, already authorized by Otari, and what the workspace chose for its backend."""

    organization_id: uuid.UUID | None
    workspace_id: uuid.UUID
    member_id: uuid.UUID | None
    """The member the call runs as; None only for a call about the whole workspace, such as erase."""
    api_key_id: str | None = None
    session_id: str | None = None
    """The trace id of the request's session, where it names one."""
    model: str | None = None
    """The model the request runs on, where Otari knows it, kept as provenance of a write."""
    can_write: bool = False
    shares: bool = True
    """Whether the workspace scope is on: off, a backend serves and keeps the member's own items only."""
    settings: Mapping[str, object] = field(default_factory=dict)
    """The backend's own settings as the workspace chose them, for what only this backend understands."""


@dataclass(frozen=True)
class MemoryItem:
    """One item of memory as a backend returns it. Fields a backend does not keep are None."""

    id: str
    text: str
    kind: MemoryItemKind
    scope: MemoryItemScope
    observed_on: date | None = None
    """The UTC date the item's evidence was observed, against which relative wording is read."""
    confidence: float | None = None
    relevance: float | None = None
    """How the item ranked for this query, 0 to 1, relative to the other items returned."""
    trust: str | None = None
    version: int | None = None


@dataclass(frozen=True)
class MemoryItemWithHistory:
    """An item with the versions it replaced, newest first; empty where the backend keeps no history."""

    item: MemoryItem
    versions: tuple[MemoryItem, ...] = ()


@dataclass(frozen=True)
class MemoryWrite:
    """What an agent offers memory, already screened by Otari."""

    text: str
    kind: MemoryItemKind
    scope: MemoryItemScope
    external: bool
    """Whether the agent said it came from an outside source (a fetched page, a tool result)."""
    observed_at: datetime


@dataclass(frozen=True)
class WriteReceipt:
    """What became of a write: kept, waiting to be decided, or held for an admin."""

    state: WriteState
    id: str | None = None


@dataclass(frozen=True)
class SearchFilters:
    """What a search is narrowed to; honored only by a backend declaring ``SEARCH_FILTERS``."""

    kinds: frozenset[MemoryItemKind] | None = None
    since: datetime | None = None
    until: datetime | None = None


class MemoryBackend(Protocol):
    """Keeps a workspace's memory: stores what agents offer, finds it again, and forgets it."""

    @property
    def name(self) -> str:
        """The backend's name as usage rows and logs show it, such as ``otari`` or ``everos``."""
        ...

    @property
    def capabilities(self) -> frozenset[MemoryCapability]:
        """What the backend can do beyond the required calls."""
        ...

    async def recall(self, ctx: MemoryContext, turn: str, context: Sequence[str], *, limit: int) -> list[MemoryItem]:
        """The items worth handing the model for this turn, best first, at most ``limit``; empty is a fine answer.

        ``context`` is the turns before it, newest first, for a backend that reads them.

        Raises:
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def search(
        self, ctx: MemoryContext, query: str, *, limit: int, filters: SearchFilters | None = None
    ) -> list[MemoryItem]:
        """The items that best match ``query``, best first, at most ``limit``.

        Raises:
            MemoryCapabilityError: ``filters`` were given and the backend does not declare ``SEARCH_FILTERS``.
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def get(self, ctx: MemoryContext, item_id: str) -> MemoryItemWithHistory:
        """One item the member may see, with the versions it replaced where the backend keeps them.

        Raises:
            MemoryItemNotFoundError: no such item the member may see.
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def add(self, ctx: MemoryContext, write: MemoryWrite) -> WriteReceipt:
        """Offer memory one item.

        Raises:
            MemoryCapabilityError: a member-scope write to a backend without ``MEMBER_SCOPE``.
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def forget(self, ctx: MemoryContext, item_id: str) -> int:
        """Delete an item and every version it replaced; how many versions went.

        Raises:
            MemoryItemNotFoundError: no such item the member may see.
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def confirm(self, ctx: MemoryContext, item_id: str) -> None:
        """Say an item held up in use.

        Raises:
            MemoryCapabilityError: the backend does not declare ``FEEDBACK``.
            MemoryItemNotFoundError: no such item the member may see.
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def flag(self, ctx: MemoryContext, item_id: str, reason: MemoryFlagReason) -> None:
        """Say an item is wrong, stale, a duplicate or unsafe.

        Raises:
            MemoryCapabilityError: the backend does not declare ``FEEDBACK``.
            MemoryItemNotFoundError: no such item the member may see.
            MemoryBackendError: the backend could not answer.
        """
        ...

    async def erase(self, workspace_id: uuid.UUID) -> None:
        """Delete everything the backend keeps for the workspace.

        Raises:
            MemoryBackendError: the backend could not answer; Otari keeps asking until it does.
        """
        ...


__all__ = [
    "MemoryBackend",
    "MemoryBackendError",
    "MemoryCapability",
    "MemoryCapabilityError",
    "MemoryContext",
    "MemoryFlagReason",
    "MemoryItem",
    "MemoryItemKind",
    "MemoryItemNotFoundError",
    "MemoryItemScope",
    "MemoryItemWithHistory",
    "MemoryWrite",
    "SearchFilters",
    "WriteReceipt",
    "WriteState",
]
