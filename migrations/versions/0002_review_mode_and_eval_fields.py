"""Add review_mode to students; recommendation_source to interactions

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "students",
        sa.Column("review_mode", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "interactions",
        sa.Column("recommendation_source", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("interactions", "recommendation_source")
    op.drop_column("students", "review_mode")
