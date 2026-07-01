"""Add macro step outcomes table for estimator datasets.

Revision ID: 0023
Revises: 0022
"""
from typing import Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0023"
down_revision: Union[str, None] = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "macro_step_outcomes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", UUID(as_uuid=True), nullable=False),
        sa.Column("plan_id", UUID(as_uuid=True), nullable=False),
        sa.Column("plan_step_id", UUID(as_uuid=True), nullable=True),
        sa.Column("kc_id", sa.String(), nullable=False),
        sa.Column("outcome_type", sa.String(64), nullable=False),
        sa.Column("tasks_spent", sa.Integer(), nullable=True),
        sa.Column("tasks_budget", sa.Integer(), nullable=True),
        sa.Column("difficulty_mode", sa.String(), nullable=True),
        sa.Column("mastery_current", sa.Float(), nullable=True),
        sa.Column("recent_accuracy", sa.Float(), nullable=True),
        sa.Column("velocity", sa.Float(), nullable=True),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column("profile_snapshot", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_macro_step_outcomes_student_created_at",
        "macro_step_outcomes",
        ["student_id", "created_at"],
    )
    op.create_index(
        "ix_macro_step_outcomes_outcome_type",
        "macro_step_outcomes",
        ["outcome_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_macro_step_outcomes_outcome_type", table_name="macro_step_outcomes")
    op.drop_index("ix_macro_step_outcomes_student_created_at", table_name="macro_step_outcomes")
    op.drop_table("macro_step_outcomes")
