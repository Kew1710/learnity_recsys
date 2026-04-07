"""Initial schema: students, mastery, interactions, tasks, parts

Revision ID: 0001
Revises:
Create Date: 2026-03-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- students ---
    op.create_table(
        "students",
        sa.Column("student_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("grade", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("guessing_rate", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("hint_dependence", sa.Float(), nullable=False, server_default="0.0"),
    )

    # --- mastery ---
    op.create_table(
        "mastery",
        sa.Column("student_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("students.student_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("kc_id", sa.String(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("last_practiced", sa.DateTime(), nullable=False),
        sa.Column("attempts_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("p_transit", sa.Float(), nullable=False, server_default="0.1"),
        sa.Column("p_slip", sa.Float(), nullable=False, server_default="0.1"),
        sa.Column("p_guess", sa.Float(), nullable=False, server_default="0.2"),
        sa.PrimaryKeyConstraint("student_id", "kc_id"),
    )
    op.create_index("ix_mastery_student_id", "mastery", ["student_id"])
    op.create_index("ix_mastery_kc_id", "mastery", ["kc_id"])

    # --- interactions ---
    op.create_table(
        "interactions",
        sa.Column("interaction_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("students.student_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("part_id", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("hints_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("time_spent_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("misconception_triggered", sa.String(), nullable=True),
    )
    op.create_index("ix_interactions_student_id", "interactions", ["student_id"])
    op.create_index("ix_interactions_task_id", "interactions", ["task_id"])

    # --- tasks ---
    op.create_table(
        "tasks",
        sa.Column("task_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("grade_min", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="bank"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # --- parts ---
    op.create_table(
        "parts",
        sa.Column("task_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tasks.task_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("part_id", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("primary_kcs", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("secondary_kcs", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("answer_type", sa.String(), nullable=False),
        sa.Column("correct_answer", postgresql.JSONB(), nullable=True),
        sa.Column("tolerance", sa.Float(), nullable=True),
        sa.Column("irt_difficulty", sa.Float(), nullable=True),
        sa.Column("irt_discrimination", sa.Float(), nullable=True),
        sa.Column("irt_guessing", sa.Float(), nullable=True),
        sa.Column("scaffolding_steps", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("distractors_map", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("task_id", "part_id"),
    )
    op.create_index("ix_parts_task_id", "parts", ["task_id"])


def downgrade() -> None:
    op.drop_table("parts")
    op.drop_table("tasks")
    op.drop_table("interactions")
    op.drop_table("mastery")
    op.drop_table("students")
