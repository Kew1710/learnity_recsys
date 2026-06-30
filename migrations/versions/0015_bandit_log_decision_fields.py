"""Add decision log fields to bandit_log.

Revision ID: 0015
Revises: 0014
"""
from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("bandit_log", sa.Column("selection_reason", sa.String(), nullable=True))
    op.add_column("bandit_log", sa.Column("exploration_type", sa.String(), nullable=True))
    op.add_column("bandit_log", sa.Column("zpd_candidates_count", sa.Integer(), nullable=True))
    op.add_column("bandit_log", sa.Column("plan_step_id", UUID(as_uuid=True), nullable=True))
    op.add_column("bandit_log", sa.Column("difficulty_mode", sa.String(), nullable=True))
    op.add_column("bandit_log", sa.Column("fallback_occurred", sa.Boolean(), nullable=True, server_default="false"))


def downgrade() -> None:
    op.drop_column("bandit_log", "fallback_occurred")
    op.drop_column("bandit_log", "difficulty_mode")
    op.drop_column("bandit_log", "plan_step_id")
    op.drop_column("bandit_log", "zpd_candidates_count")
    op.drop_column("bandit_log", "exploration_type")
    op.drop_column("bandit_log", "selection_reason")
