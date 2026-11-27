"""Recall and search by exact names as well as by similarity, and memory_search's filters."""

from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import GatewayConfig
from gateway.services.memory import MemoryModels, RecallVectors
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client
from .test_memory_mcp import DEPLOY, _call, _consolidate, _embed, _key
from .test_memory_mcp import memory_client as memory_client  # noqa: F401 - the fixture, used by name
from .test_memory_recall import _last_user_text
from .test_memory_service_split import engine_app as engine_app  # noqa: F401 - the fixture, used by name
from .test_memory_service_split import split_client as split_client  # noqa: F401 - the fixture, used by name
from .test_memory_tool_loop import _chat, _completion, _provider, _switches

# None of the test embedding's words: its vector is unlike any question about deploying the api.
NAMED = "svc_ledger_rotation is paused until the audit ends"
ASK_BY_NAME = "should the api deploy wait for svc_ledger_rotation?"
FACT = "The api listens on port 8010 on main"


@pytest.fixture
def dense_client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A gateway with the lexical lane off: it ranks by similarity alone."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(MemoryModels, "embed", _embed)
    config = test_config.model_copy(
        update={"memory_enabled": True, "memory_embedding_model": "openai:embed", "memory_lexical_search": False}
    )
    yield from build_test_client(config)


def _remember(client: TestClient, headers: dict[str, str], *entries: tuple[str, str]) -> None:
    for text, kind in entries:
        assert _call(client, headers, "memory_propose", text=text, kind=kind)["isError"] is False
    assert _consolidate(client) == len(entries)


def _recall_by_name(client: TestClient, headers: dict[str, str]) -> tuple[Any, str]:
    cast(FastAPI, client.app).state.memory_recall_vectors = RecallVectors()
    with _provider(_completion(None)) as sent:
        response = _chat(client, headers, messages=[{"role": "user", "content": ASK_BY_NAME}])
    assert response.status_code == 200, response.text
    return response, _last_user_text(sent[0])


def test_recall_hands_over_an_entry_named_in_the_turn_that_similarity_alone_misses(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remember(memory_client, headers, (DEPLOY, "procedure"), (NAMED, "decision"))
    _switches(memory_client, db_session_factory, master_key_header, auto_recall=True)

    response, text = _recall_by_name(memory_client, headers)

    assert response.headers["Otari-Memory-Recalled"] == "2"
    assert NAMED in text and DEPLOY in text


def test_with_the_lexical_lane_off_recall_goes_by_similarity_alone(
    dense_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(dense_client, master_key_header, db_session_factory)
    _remember(dense_client, headers, (DEPLOY, "procedure"), (NAMED, "decision"))
    _switches(dense_client, db_session_factory, master_key_header, auto_recall=True)

    response, text = _recall_by_name(dense_client, headers)

    assert response.headers["Otari-Memory-Recalled"] == "1"
    assert DEPLOY in text and NAMED not in text


def test_search_puts_the_entry_a_query_names_first(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remember(memory_client, headers, (DEPLOY, "procedure"), (NAMED, "decision"))

    found = _call(memory_client, headers, "memory_search", query=ASK_BY_NAME)

    results = found["structuredContent"]["results"]
    assert [entry["text"] for entry in results] == [NAMED, DEPLOY]
    assert 0 < results[1]["relevance"] < results[0]["relevance"] <= 1


def test_recall_through_the_memory_service_runs_the_lexical_lane_there(
    split_client: TestClient,
    engine_app: FastAPI,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    engine_app.state.memory_recall_vectors = RecallVectors()
    headers = _key(split_client, master_key_header, db_session_factory)
    _remember(split_client, headers, (DEPLOY, "procedure"), (NAMED, "decision"))
    _switches(split_client, db_session_factory, master_key_header, auto_recall=True)

    with _provider(_completion(None)) as sent:
        response = _chat(split_client, headers, messages=[{"role": "user", "content": ASK_BY_NAME}])

    assert response.headers["Otari-Memory-Recalled"] == "2"
    assert NAMED in _last_user_text(sent[0])
    assert len(engine_app.state.memory_recall_vectors.terms) == 2, "the service holds the terms, not the gateway"


# -- memory_search's filters ---------------------------------------------------------


def _texts(found: dict[str, Any]) -> list[str]:
    assert found["isError"] is False, found
    return sorted(entry["text"] for entry in found["structuredContent"]["results"])


def test_search_narrows_by_kind_and_by_when_an_entry_was_recorded(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)
    _remember(memory_client, headers, (DEPLOY, "procedure"), (FACT, "fact"))
    now = datetime.now(UTC)
    yesterday, tomorrow = (now - timedelta(days=1)).isoformat(), (now + timedelta(days=1)).isoformat()

    def search(**filters: Any) -> list[str]:
        return _texts(_call(memory_client, headers, "memory_search", query="deploy the api on main", **filters))

    assert search() == sorted([DEPLOY, FACT])
    assert search(kind="fact") == [FACT]
    assert search(kind=["procedure", "decision"]) == [DEPLOY]
    assert search(since=yesterday, until=tomorrow) == sorted([DEPLOY, FACT])
    assert search(since=tomorrow) == []
    assert search(until=yesterday) == []
    assert search(kind="fact", since=now.date().isoformat()) == [FACT], "a date alone is its midnight, UTC"


def test_a_filter_search_cannot_apply_is_a_tool_error(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    headers = _key(memory_client, master_key_header, db_session_factory)

    backwards = _call(memory_client, headers, "memory_search", query="q", since="2026-02-02", until="2026-02-01")
    unknown = _call(memory_client, headers, "memory_search", query="q", kind="gossip", tag="billing")

    assert backwards["isError"] is True and "since" in backwards["content"][0]["text"]
    assert unknown["isError"] is True
    assert "tag" in unknown["content"][0]["text"] and "gossip" not in unknown["content"][0]["text"]
