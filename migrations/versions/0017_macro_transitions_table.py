"""Macro transitions table for offline RL logging.

Records (state, action, reward, next_state, done) at each plan lifecycle event.

Revision ID: 0017
Revises: 0016
"""
from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "macro_transitions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("plan_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("kc_id", sa.String(255), nullable=False),
        sa.Column("state", JSONB(), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("action_payload", JSONB(), nullable=True),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("next_state", JSONB(), nullable=True),
        sa.Column("done", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("reason", sa.String(255), nullable=True),
        sa.Column("diagnosis_reason", sa.String(64), nullable=True),
        sa.Column("diagnosis_confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("macro_transitions")
