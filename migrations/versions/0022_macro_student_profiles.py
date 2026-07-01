"""Add macro student profiles and snapshots.

Revision ID: 0022
Revises: 0021
"""
from typing import Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels = None
depends_on = None


def _profile_columns(*, student_primary_key: bool) -> list[sa.Column]:
    return [
        sa.Column("student_id", UUID(as_uuid=True), primary_key=student_primary_key),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_kc_id", sa.String(), nullable=True),
        sa.Column("uncertainty_level", sa.Float(), nullable=False),
        sa.Column("mastery_confidence_mean", sa.Float(), nullable=False),
        sa.Column("weak_prereq_fraction", sa.Float(), nullable=False),
        sa.Column("target_subgraph_mastery_mean", sa.Float(), nullable=False),
        sa.Column("learning_speed_global", sa.Float(), nullable=False),
        sa.Column("learning_speed_recent", sa.Float(), nullable=False),
        sa.Column("tasks_to_gain_01_mastery", sa.Float(), nullable=False),
        sa.Column("recovery_after_error", sa.Float(), nullable=False),
        sa.Column("frustration_risk", sa.Float(), nullable=False),
        sa.Column("stall_risk_baseline", sa.Float(), nullable=False),
        sa.Column("regression_risk_baseline", sa.Float(), nullable=False),
        sa.Column("pacing_mode", sa.String(), nullable=False),
        sa.Column("budget_multiplier", sa.Float(), nullable=False),
        sa.Column("prereq_strictness", sa.Float(), nullable=False),
        sa.Column("test_readiness_bias", sa.Float(), nullable=False),
        sa.Column("step_granularity", sa.Float(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table("macro_student_profiles", *_profile_columns(student_primary_key=True))
    op.create_table(
        "macro_student_profile_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        *_profile_columns(student_primary_key=False),
        sa.Column("snapshot_created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_macro_student_profile_snapshots_student_updated_at",
        "macro_student_profile_snapshots",
        ["student_id", "snapshot_created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_macro_student_profile_snapshots_student_updated_at",
        table_name="macro_student_profile_snapshots",
    )
    op.drop_table("macro_student_profile_snapshots")
    op.drop_table("macro_student_profiles")
