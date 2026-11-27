"""Add memory tombstones, when evidence was observed, and how long a forgotten fact stays forgotten.

- ``memory_tombstones``: what is kept of a forgotten entry version until it
  expires, its embedding sealed and never its text, so a proposal of the same
  fact does not come straight back. Fenced by workspace under forced row-level
  security, as the other memory tables are.
- ``memory_entries.observed_at`` / ``memory_proposals.observed_at``: when the
  agent proposed what a row says, so older evidence never replaces newer.
  Backfilled from ``created_at``.
- ``memory_workspace_settings.tombstone_days``: days a tombstone is kept, 1-3650,
  90 by default.

Revision ID: 74ee86cd94e7
Revises: c9d1e3f5a7b2
Create Date: 2026-11-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "74ee86cd94e7"
down_revision: str | Sequence[str] | None = "c9d1e3f5a7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEMORY_TOMBSTONES = "memory_tombstones"
MEMORY_ENTRIES = "memory_entries"
MEMORY_PROPOSALS = "memory_proposals"
MEMORY_WORKSPACE_SETTINGS = "memory_workspace_settings"

_CURRENT_WORKSPACE = "NULLIF(current_setting('otari.memory_workspace', true), '')::uuid"


def upgrade() -> None:
    op.create_table(
        MEMORY_TOMBSTONES,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=True),
        sa.Column("key_id", sa.Uuid(), nullable=False),
        sa.Column("embedding_sealed", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("forgotten_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("(scope = 'member') = (owner_user_id IS NOT NULL)", name="check_memory_tombstones_owner"),
        sa.CheckConstraint("scope IN ('workspace', 'member')", name="check_memory_tombstones_scope"),
        sa.ForeignKeyConstraint(["key_id"], ["memory_keys.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspace.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_tombstones_key_id", MEMORY_TOMBSTONES, ["key_id"])
    op.create_index("ix_memory_tombstones_owner_user_id", MEMORY_TOMBSTONES, ["owner_user_id"])
    op.create_index("ix_memory_tombstones_workspace_scope", MEMORY_TOMBSTONES, ["workspace_id", "scope"])
    op.create_index("ix_memory_tombstones_workspace_expires_at", MEMORY_TOMBSTONES, ["workspace_id", "expires_at"])
    if op.get_bind().dialect.name == "postgresql":
        op.execute(f"ALTER TABLE {MEMORY_TOMBSTONES} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {MEMORY_TOMBSTONES} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {MEMORY_TOMBSTONES}_workspace ON {MEMORY_TOMBSTONES} "
            f"USING (workspace_id = {_CURRENT_WORKSPACE}) WITH CHECK (workspace_id = {_CURRENT_WORKSPACE})"
        )

    # Added nullable and backfilled first: neither engine fills a new column from another one.
    for table in (MEMORY_ENTRIES, MEMORY_PROPOSALS):
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True))
        # The row-level policy hides every row from a role it binds; the migration runs as the tables' owner,
        # which ``FORCE`` binds too, so it names no workspace and is let through by turning the policy off.
        _unfenced(table, f"UPDATE {table} SET observed_at = created_at")
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                "observed_at",
                existing_type=sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )

    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.add_column(sa.Column("tombstone_days", sa.Integer(), server_default="90", nullable=False))
        batch.create_check_constraint("check_memory_settings_tombstone_days", "tombstone_days BETWEEN 1 AND 3650")


def downgrade() -> None:
    with op.batch_alter_table(MEMORY_WORKSPACE_SETTINGS) as batch:
        batch.drop_constraint("check_memory_settings_tombstone_days", type_="check")
        batch.drop_column("tombstone_days")
    for table in (MEMORY_PROPOSALS, MEMORY_ENTRIES):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("observed_at")
    # Dropping the table drops its policy with it.
    op.drop_index("ix_memory_tombstones_workspace_expires_at", table_name=MEMORY_TOMBSTONES)
    op.drop_index("ix_memory_tombstones_workspace_scope", table_name=MEMORY_TOMBSTONES)
    op.drop_index("ix_memory_tombstones_owner_user_id", table_name=MEMORY_TOMBSTONES)
    op.drop_index("ix_memory_tombstones_key_id", table_name=MEMORY_TOMBSTONES)
    op.drop_table(MEMORY_TOMBSTONES)


def _unfenced(table: str, statement: str) -> None:
    """Run ``statement`` over every row of a fenced table, whatever workspace the session names."""
    if op.get_bind().dialect.name != "postgresql":
        op.execute(statement)
        return
    op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
    op.execute(statement)
    op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
