"""Macro-Micro interface: extend learning_plans and plan_steps

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # learning_plans: macro-level plan metadata
    op.add_column("learning_plans", sa.Column("goal_type", sa.String(), nullable=True))
    op.add_column("learning_plans", sa.Column("mastery_threshold", sa.Float(), server_default="0.80", nullable=False))
    op.add_column("learning_plans", sa.Column("require_test", sa.Boolean(), server_default="false", nullable=False))
    op.add_column("learning_plans", sa.Column("coverage_variant", sa.String(), nullable=True))
    op.add_column("learning_plans", sa.Column("task_budget", sa.Integer(), nullable=True))

    # plan_steps: per-step micro directives
    op.add_column("plan_steps", sa.Column("difficulty_mode", sa.String(), server_default="build", nullable=False))
    op.add_column("plan_steps", sa.Column("tasks_budget", sa.Integer(), server_default="20", nullable=False))
    op.add_column("plan_steps", sa.Column("tasks_spent", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    op.drop_column("plan_steps", "tasks_spent")
    op.drop_column("plan_steps", "tasks_budget")
    op.drop_column("plan_steps", "difficulty_mode")
    op.drop_column("learning_plans", "task_budget")
    op.drop_column("learning_plans", "coverage_variant")
    op.drop_column("learning_plans", "require_test")
    op.drop_column("learning_plans", "mastery_threshold")
    op.drop_column("learning_plans", "goal_type")
