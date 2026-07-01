"""Store raw learning signals in bandit_log separately from reward.

Revision ID: 0019
Revises: 0018
"""
from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("bandit_log", sa.Column("raw_score", sa.Float(), nullable=True))
    op.add_column("bandit_log", sa.Column("hints_used", sa.Integer(), nullable=True))
    op.add_column("bandit_log", sa.Column("time_spent_seconds", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("bandit_log", "time_spent_seconds")
    op.drop_column("bandit_log", "hints_used")
    op.drop_column("bandit_log", "raw_score")
