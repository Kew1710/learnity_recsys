"""Add recent_accuracy to mastery table for confidence computation.

Revision ID: 0014
Revises: 0013
"""
from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "mastery",
        sa.Column("recent_accuracy", sa.Float(), nullable=False, server_default="0.0"),
    )


def downgrade() -> None:
    op.drop_column("mastery", "recent_accuracy")
