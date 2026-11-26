"""Rejoin main's migration head and agent memory's into one.

Agent memory's revisions were authored on a branch off main, and main went on
meanwhile: ``7c4b45a55970`` (the per-member setup guide state) is main's head and
``69080982e58d`` (the memory rerank skip) is memory's. They touch different
tables, but leaving both as heads makes ``alembic upgrade head`` refuse to run
at all ("Multiple head revisions are present").

This revision has no schema of its own. It only names both parents, so ``head``
resolves to a single revision again and the two branches apply in either order.
As at every merge point, ``alembic downgrade -1`` from here is ambiguous;
downgrade to a named revision instead.

Revision ID: 62f1e2f1da02
Revises: 7c4b45a55970, 69080982e58d
Create Date: 2026-11-26
"""

from collections.abc import Sequence

revision: str = "62f1e2f1da02"
down_revision: str | Sequence[str] | None = ("7c4b45a55970", "69080982e58d")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Nothing to apply: this revision only rejoins the two branches."""


def downgrade() -> None:
    """Nothing to undo: this revision only rejoins the two branches."""
