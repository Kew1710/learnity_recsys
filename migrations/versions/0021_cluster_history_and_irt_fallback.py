"""Add cluster history and IRT fallback observability.

Revision ID: 0021
Revises: 0020
"""
from typing import Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "bandit_log",
        sa.Column("irt_fallback_occurred", sa.Boolean(), nullable=True, server_default="false"),
    )
    op.create_table(
        "student_cluster_history",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", UUID(as_uuid=True), nullable=False),
        sa.Column("from_cluster_id", sa.Integer(), nullable=True),
        sa.Column("to_cluster_id", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("task_count", sa.Integer(), nullable=True),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_student_cluster_history_student_changed_at",
        "student_cluster_history",
        ["student_id", "changed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_student_cluster_history_student_changed_at", table_name="student_cluster_history")
    op.drop_table("student_cluster_history")
    op.drop_column("bandit_log", "irt_fallback_occurred")
