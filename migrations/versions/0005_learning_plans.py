"""Add learning_plans and plan_steps tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "learning_plans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_learning_plans_student_id", "learning_plans", ["student_id"])

    op.create_table(
        "plan_steps",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("plan_id", UUID(as_uuid=True), nullable=False),
        sa.Column("kc_id", sa.String(), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("inserted_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["plan_id"], ["learning_plans.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_plan_steps_plan_id", "plan_steps", ["plan_id"])
    op.create_index("ix_plan_steps_kc_id", "plan_steps", ["kc_id"])


def downgrade() -> None:
    op.drop_index("ix_plan_steps_kc_id", "plan_steps")
    op.drop_index("ix_plan_steps_plan_id", "plan_steps")
    op.drop_table("plan_steps")
    op.drop_index("ix_learning_plans_student_id", "learning_plans")
    op.drop_table("learning_plans")
