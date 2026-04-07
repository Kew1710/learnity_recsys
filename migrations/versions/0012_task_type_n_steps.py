"""Add task_type and n_steps to parts

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "parts",
        sa.Column("task_type", sa.String(), nullable=False, server_default="procedural"),
    )
    op.add_column(
        "parts",
        sa.Column("n_steps", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("parts", "n_steps")
    op.drop_column("parts", "task_type")
