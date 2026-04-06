"""Add estimated_lr to students

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "students",
        sa.Column("estimated_lr", sa.Float(), nullable=False, server_default="0.15"),
    )


def downgrade() -> None:
    op.drop_column("students", "estimated_lr")
