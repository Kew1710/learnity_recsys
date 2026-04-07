"""Add student_bandit_model table for per-student LinUCB models

При первом обращении к KC копируется из кластерной модели (cold start из кластера).
Далее обновляется только для этого студента, накапливая личный опыт.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, BYTEA

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "student_bandit_model",
        sa.Column("student_id", UUID(as_uuid=True), nullable=False),
        sa.Column("kc_id", sa.String(), nullable=False),
        sa.Column("a_matrix", BYTEA(), nullable=False),
        sa.Column("b_vector", BYTEA(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("student_id", "kc_id"),
    )
    op.create_index("ix_student_bandit_model_student", "student_bandit_model", ["student_id"])


def downgrade() -> None:
    op.drop_index("ix_student_bandit_model_student", "student_bandit_model")
    op.drop_table("student_bandit_model")
