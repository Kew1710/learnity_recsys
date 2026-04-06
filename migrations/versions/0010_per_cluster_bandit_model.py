"""Per-cluster bandit_model: replace PK(kc_id) with PK(cluster_id, kc_id)

cluster_id = -1 используется как глобальная запасная модель для студентов
без назначенного кластера.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import BYTEA

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("bandit_model")
    op.create_table(
        "bandit_model",
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("kc_id", sa.String(), nullable=False),
        sa.Column("a_matrix", BYTEA(), nullable=False),
        sa.Column("b_vector", BYTEA(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("cluster_id", "kc_id"),
    )


def downgrade() -> None:
    op.drop_table("bandit_model")
    op.create_table(
        "bandit_model",
        sa.Column("kc_id", sa.String(), primary_key=True),
        sa.Column("a_matrix", BYTEA(), nullable=False),
        sa.Column("b_vector", BYTEA(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
