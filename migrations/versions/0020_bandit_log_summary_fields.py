"""Add mastery delta and irt difficulty to bandit_log for micro summaries.

Revision ID: 0020
Revises: 0019
"""
from typing import Union

from alembic import op
import sqlalchemy as sa

revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("bandit_log", sa.Column("mastery_delta", sa.Float(), nullable=True))
    op.add_column("bandit_log", sa.Column("irt_difficulty", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("bandit_log", "irt_difficulty")
    op.drop_column("bandit_log", "mastery_delta")
