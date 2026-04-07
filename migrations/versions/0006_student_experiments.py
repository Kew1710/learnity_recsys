"""Add student_experiments table for A/B testing

Revision ID: 0006
Revises: 0005
Create Date: 2026-03-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "student_experiments",
        sa.Column("student_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", sa.String(), nullable=False),
        sa.Column("variant", sa.String(), nullable=False),   # 'control' | 'treatment'
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_student_experiments_experiment_id",
        "student_experiments",
        ["experiment_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_student_experiments_experiment_id", "student_experiments")
    op.drop_table("student_experiments")
