"""How a request reaches agent memory: its service, its caller, and what memory a request through the gateway gets."""

import secrets
from datetime import timedelta
from functools import partial
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import ContainerDep, get_config, get_container, get_unit_of_work
from gateway.api.deps.auth import extract_credential_token, verify_api_key
from gateway.api.deps.traces import _person, is_workspace_member
from gateway.container import Container
from gateway.core.config import GatewayConfig
from gateway.core.database import get_db
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.memory_exceptions import NOT_A_MEMBER
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import MANAGEMENT_ROLES
from gateway.ports.content_key_port import ContentKeyPort
from gateway.ports.memory_ranking_port import MemoryRankingPort
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.repositories.memory import MemoryRepositories
from gateway.repositories.tenancy import OrganizationMemberRepository, WorkspaceMemberRepository, WorkspaceRepository
from gateway.services.memory import (
    MemoryCaller,
    MemoryModels,
    MemoryService,
    MemoryToolAdmission,
    RecallVectors,
    WorkspaceKeys,
)
from gateway.services.tenancy import ManagerChecks, OrganizationService
from gateway.services.tenancy.organization_guardrail_service import MemoryMandates
from gateway.services.usage_and_telemetry import MemoryUsageLedger, MemoryUsageMeter
from gateway.types.key_scope import KeyScope


def verify_memory_service_token(request: Request, config: Annotated[GatewayConfig, Depends(get_config)]) -> None:
    """Admit only a gateway of this deployment to the memory service, by ``memory_service_token``.

    Neither an API key nor the master key opens it: the service answers about any workspace, and only a gateway,
    which has already decided what each member may recall, is trusted to ask.
    """
    expected = config.memory_service_token
    token = extract_credential_token(request)
    if not config.memory_engine or not expected or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a gateway of this deployment.")


def memory_keys_for(config: GatewayConfig, keys: ContentKeyPort | None) -> WorkspaceKeys:
    """The workspaces' memory data keys over ``keys``, rotated as the deployment says."""
    return WorkspaceKeys(keys, rotation=timedelta(days=config.memory_key_rotation_days))


def build_memory_service(
    uow: UnitOfWork,
    config: GatewayConfig,
    keys: WorkspaceKeys,
    container: Container,
    db: AsyncSession | None = None,
    *,
    recall_vectors: RecallVectors | None = None,
    ranking: MemoryRankingPort | None = None,
) -> MemoryService:
    """Build the memory service on a request's Unit of Work, or the consolidator's own.

    ``container`` serves the hosted providers memory's models may run on, read on the same Unit of Work.
    ``db`` is the request's session, for the management checks; the consolidator has no caller to check.
    ``recall_vectors`` is the process's opened embeddings, for automatic recall and erasing what it holds.
    ``ranking`` is the memory service recall and search rank on; None ranks in this process.
    """
    managers = ManagerChecks(db, OrganizationService(db, membership_listener=None)) if db is not None else None
    return MemoryService(
        uow,
        MemoryRepositories.on(uow),
        keys,
        MemoryModels(config, container.resolve(ModelProviderPort, db, uow=uow)),
        MemoryUsageMeter(uow),
        config,
        managers,
        ledger=MemoryUsageLedger(uow),
        recall_vectors=recall_vectors,
        ranking=ranking,
        guardrails=MemoryMandates(config),
    )


def get_memory_keys(request: Request) -> WorkspaceKeys:
    """This app's memory data keys, built on first use and held for every request, so opened keys are shared."""
    keys: WorkspaceKeys | None = getattr(request.app.state, "memory_keys", None)
    if keys is None:
        config: GatewayConfig = request.app.state.config
        keys = memory_keys_for(config, get_container(request).resolve(ContentKeyPort, None))
        request.app.state.memory_keys = keys
    return keys


def get_recall_vectors(request: Request) -> RecallVectors | None:
    """This app's opened recall embeddings, held for every request; None where the deployment holds none.

    A gateway that hands ranking to a memory service holds none: the service does.
    """
    config: GatewayConfig = request.app.state.config
    if not config.memory_recall_cache or config.memory_service_url:
        return None
    held: RecallVectors | None = getattr(request.app.state, "memory_recall_vectors", None)
    if held is None:
        held = RecallVectors()
        request.app.state.memory_recall_vectors = held
    return held


def get_memory_ranking(request: Request) -> MemoryRankingPort | None:
    """The memory service this app hands recall's ranking to; None where it ranks in its own process."""
    config: GatewayConfig = request.app.state.config
    if not config.memory_service_url:
        return None
    ranking: MemoryRankingPort = get_container(request).resolve(MemoryRankingPort, None)
    return ranking


def get_memory_service(
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    keys: Annotated[WorkspaceKeys, Depends(get_memory_keys)],
    container: ContainerDep,
    recall_vectors: Annotated[RecallVectors | None, Depends(get_recall_vectors)],
    ranking: Annotated[MemoryRankingPort | None, Depends(get_memory_ranking)],
) -> MemoryService:
    """The request's memory service. The consolidator builds its own through ``build_memory_service``."""
    return build_memory_service(uow, config, keys, container, db, recall_vectors=recall_vectors, ranking=ranking)


async def memory_caller_for(
    db: AsyncSession, api_key: APIKey, *, session_id: str | None = None, model: str | None = None
) -> MemoryCaller | None:
    """The member a key speaks for, in the key's own workspace; None for a key that names no active member.

    A service key, or a billing user no person stands behind, names none. Memory never takes the workspace
    or the member from a request body.
    """
    workspace = await WorkspaceRepository(db).get(api_key.workspace_id)
    person = _person(api_key.user_id)
    if (
        workspace is None
        or person is None
        or not await is_workspace_member(db, api_key.workspace_id, api_key.user_id, workspace)
    ):
        return None
    membership = await OrganizationMemberRepository(db).get_active_by_organization_and_user(
        workspace.organization_id, person
    )
    seat = await WorkspaceMemberRepository(db).get_by_workspace_and_user(api_key.workspace_id, person)
    organization_manager = membership is not None and membership.role in MANAGEMENT_ROLES
    workspace_role = seat.role if seat is not None and seat.status == "active" else None
    return MemoryCaller(
        workspace_id=api_key.workspace_id,
        member_id=person,
        is_manager=organization_manager or workspace_role in MANAGEMENT_ROLES,
        # A workspace viewer recalls what the workspace knows and writes nothing to it.
        can_write=organization_manager or workspace_role != "viewer",
        key_may_write=KeyScope.MEMORY_WRITE in api_key.scopes,
        api_key_id=api_key.id,
        session_id=session_id[:255] if session_id else None,
        model=model[:255] if model else None,
    )


async def get_memory_caller(
    request: Request,
    api_key: Annotated[APIKey, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MemoryCaller:
    """The member a key speaks for on the MCP endpoint.

    Raises:
        HTTPException: 403 when the key names no active member of its workspace.
    """
    session_id = request.headers.get("Otari-Session-Id") or request.headers.get("Mcp-Session-Id")
    caller = await memory_caller_for(db, api_key, session_id=session_id)
    if caller is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NOT_A_MEMBER)
    return caller


def get_memory_tool_admission(
    request: Request, db: Annotated[AsyncSession, Depends(get_db)]
) -> MemoryToolAdmission | None:
    """What admits memory's tools to a request, or None where the deployment does not offer memory.

    The caller is resolved on ``db``, the request's own session.
    """
    config: GatewayConfig = request.app.state.config
    if not config.memory_enabled:
        return None
    keys = get_memory_keys(request)
    container = get_container(request)
    recall_vectors = get_recall_vectors(request)
    ranking = get_memory_ranking(request)
    return MemoryToolAdmission(
        lambda uow: build_memory_service(uow, config, keys, container, recall_vectors=recall_vectors, ranking=ranking),
        partial(memory_caller_for, db),
    )


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
MemoryCallerDep = Annotated[MemoryCaller, Depends(get_memory_caller)]
