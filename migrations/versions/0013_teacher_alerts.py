"""teacher_alerts table

Revision ID: 0013
Revises: 0012
"""
from typing import Union
from alembic import op
import sqlalchemy as sa

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teacher_alerts",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plan_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kc_id", sa.String, nullable=False),
        sa.Column("alert_type", sa.String, nullable=False),  # "plateau" | "budget_exceeded" | "plan_exhausted"
        sa.Column("mastery_at_alert", sa.Float, nullable=False),
        sa.Column("tasks_spent", sa.Integer, nullable=False),
        sa.Column("message", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_teacher_alerts_student_id", "teacher_alerts", ["student_id"])
    op.create_index("ix_teacher_alerts_created_at", "teacher_alerts", ["created_at"])


def downgrade() -> None:
    op.drop_table("teacher_alerts")
