"""Add bandit_log, student_clusters, cluster_task_stats, bandit_model

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY, FLOAT, BYTEA

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bandit_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", UUID(as_uuid=True), nullable=False),
        sa.Column("kc_id", sa.String(), nullable=False),
        sa.Column("context_vector", ARRAY(sa.Float()), nullable=False),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("recommended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_bandit_log_student_task", "bandit_log", ["student_id", "task_id"])

    op.create_table(
        "student_clusters",
        sa.Column("student_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "cluster_task_stats",
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("task_id", UUID(as_uuid=True), nullable=False),
        sa.Column("avg_reward", sa.Float(), nullable=False),
        sa.Column("interaction_count", sa.Integer(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("cluster_id", "task_id"),
    )

    op.create_table(
        "bandit_model",
        sa.Column("kc_id", sa.String(), primary_key=True),
        sa.Column("a_matrix", BYTEA(), nullable=False),
        sa.Column("b_vector", BYTEA(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("bandit_model")
    op.drop_table("cluster_task_stats")
    op.drop_table("student_clusters")
    op.drop_index("ix_bandit_log_student_task", "bandit_log")
    op.drop_table("bandit_log")
