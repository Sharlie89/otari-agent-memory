"""Memory's pure rules: how it scores, what it refuses to keep, and how it reads a consolidation answer."""

import json
import uuid
from datetime import date

import pytest

from gateway.core.settings.memory import MemorySettings
from gateway.services.memory._models import DECIDE_MAX_TOKENS, Decision, Note, decision_request, parse_decision
from gateway.services.memory._scoring import (
    CONFIDENCE_CEILING,
    CONFIDENCE_FLOOR,
    INITIAL_CONFIDENCE,
    confidence,
    cosine,
    relevance,
)
from gateway.services.memory._screening import screen
from gateway.services.memory._service import REPLACE_FLOOR, _forgotten, guarded_decision
from gateway.services.memory._store import OpenedTombstone


def test_confidence_follows_cqs_rules() -> None:
    assert confidence(0, 0) == INITIAL_CONFIDENCE
    assert confidence(1, 0) == pytest.approx(0.6)
    assert confidence(0, 1) == pytest.approx(0.35)
    assert confidence(2, 1) == pytest.approx(0.55)


def test_confidence_stays_between_its_floor_and_ceiling() -> None:
    assert confidence(50, 0) == CONFIDENCE_CEILING
    assert confidence(0, 50) == CONFIDENCE_FLOOR


def test_cosine_of_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize(("a", "b"), [([1.0], [1.0, 2.0]), ([], []), ([0.0, 0.0], [1.0, 1.0])])
def test_cosine_of_vectors_it_cannot_compare_is_zero(a: list[float], b: list[float]) -> None:
    assert cosine(a, b) == 0.0


def test_a_dissimilar_entry_ranks_low_however_trusted() -> None:
    assert relevance(-0.5, 1.0) == 0.0
    assert relevance(0.9, 0.5) > relevance(0.2, 1.0)


@pytest.mark.parametrize(
    ("text", "finding"),
    [
        ("Use the key sk-proj-abcdefghijklmnop1234 for staging", "credential"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload", "credential"),
        ("set password=hunter2hunter2 in the env file", "credential"),
        ('{"api_key": "abcdef123456"}', "credential"),
        ("AKIAIOSFODNN7EXAMPLE is the deploy user's key", "credential"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nabc", "private_key"),
        ("Ask maria.lopez@example.com before deploying", "email_address"),
        ("The test card 4111 1111 1111 1111 always passes", "card_number"),
    ],
)
def test_what_memory_does_not_keep_is_found(text: str, finding: str) -> None:
    assert screen(text) == finding


@pytest.mark.parametrize(
    "text",
    [
        "Run migrations with make migrate before the integration tests.",
        "The staging database is reached through the bastion host, never directly.",
        "Order 1234 5678 9012 3456 is not a card number because it fails Luhn.",
        "Key rotation happens on the first Monday of each month.",
        "password rotation is handled by the platform team",
    ],
)
def test_ordinary_knowledge_passes(text: str) -> None:
    assert screen(text) is None


@pytest.mark.parametrize(
    ("answer", "decision"),
    [
        ('{"action": "add", "target": null}', Decision("add")),
        ('{"action": "update", "target": 2}', Decision("update", 1)),
        ('Sure: {"action": "no_change", "target": 1} done', Decision("no_change", 0)),
        ('{"action": "conflict", "target": 3}', Decision("conflict", 2)),
    ],
)
def test_a_consolidation_answer_is_read(answer: str, decision: Decision) -> None:
    assert parse_decision(answer, existing=3) == decision


@pytest.mark.parametrize(
    "answer",
    [
        '{"action": "update"}',
        '{"action": "update", "target": 0}',
        '{"action": "update", "target": 9}',
        '{"action": "update", "target": true}',
        '{"action": "update", "target": "1"}',
    ],
)
def test_an_answer_this_cannot_act_on_waits_for_an_admin(answer: str) -> None:
    assert parse_decision(answer, existing=3) == Decision("conflict")


@pytest.mark.parametrize("answer", ["", "add it", '{"action": "delete_everything"}', "[1, 2]"])
def test_an_answer_with_no_decision_decides_nothing(answer: str) -> None:
    # The model is asked again on a later pass, rather than its silence holding the proposal for an admin.
    assert parse_decision(answer, existing=3) is None


def test_the_model_is_given_room_to_think_and_asked_for_json() -> None:
    messages, options = decision_request(Note("deploy with make"), [Note("deploy with ci")])
    assert messages[-1]["content"] == 'Existing notes:\n1. "deploy with ci"\n\nProposal: "deploy with make"'
    assert options == {"max_tokens": DECIDE_MAX_TOKENS, "temperature": 0, "response_format": {"type": "json_object"}}
    assert DECIDE_MAX_TOKENS >= 1024


def test_the_model_is_told_when_each_note_and_the_proposal_were_observed() -> None:
    messages, _ = decision_request(
        Note("we moved the standup to 10:00 yesterday", date(2026, 11, 26)),
        [Note("the standup is at 9:30", date(2026, 11, 2)), Note("standups are on Zoom")],
    )

    system, user = messages[0]["content"], messages[-1]["content"]
    assert "observed" in system and "yesterday" in system
    notes, proposal = user.split("\n\nProposal: ")
    assert notes.splitlines()[1:] == [
        '1. {"observed": "2026-11-02", "text": "the standup is at 9:30"}',
        '2. "standups are on Zoom"',
    ]
    assert json.loads(proposal) == {"observed": "2026-11-26", "text": "we moved the standup to 10:00 yesterday"}


def test_a_proposal_is_the_forgotten_fact_only_where_it_is_alike_enough_to_its_tombstone() -> None:
    same, near, apart = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    tombstones = [
        OpenedTombstone(same, (1.0, 0.0)),
        # 0.89 alike: a neighbor that shares most of its words, and still not the same fact.
        OpenedTombstone(near, (1.0, 0.5)),
        OpenedTombstone(apart, (0.0, 1.0)),
    ]

    shared = MemorySettings().memory_forgotten_similarity
    member = MemorySettings().memory_forgotten_member_similarity
    assert (shared, member) == (0.9, 0.95)
    assert _forgotten([2.0, 0.0], tombstones, shared) == [same]
    assert _forgotten([1.0, 0.3], tombstones, shared) == [same, near]
    # 0.996 alike to the first and 0.93 to its neighbor: a shared proposal waits for an admin on both, while a
    # member's own, dropped with nobody to review it, is the forgotten fact only at the stricter floor.
    assert _forgotten([1.0, 0.0875], tombstones, shared) == [same, near]
    assert _forgotten([1.0, 0.0875], tombstones, member) == [same]


@pytest.mark.parametrize(
    ("decision", "alike", "kept"),
    [
        (Decision("update", 1), [0.95, 0.80], Decision("add")),
        (Decision("no_change", 0), [0.70], Decision("add")),
        (Decision("update", 0), [0.90, 0.80], Decision("update", 0)),
        (Decision("no_change", 0), [REPLACE_FLOOR], Decision("no_change", 0)),
        (Decision("conflict", 0), [0.70], Decision("conflict", 0)),
        (Decision("add"), [0.99], Decision("add")),
    ],
)
def test_a_model_never_replaces_or_drops_for_an_entry_too_unlike_the_proposal(
    decision: Decision, alike: list[float], kept: Decision
) -> None:
    assert guarded_decision(decision, alike) == kept
