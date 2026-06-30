"""Diagnostic CAT state for cold start calibration.

Revision ID: 0018
Revises: 0017
"""
from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "diagnostic_cat_state",
        sa.Column("student_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("kc_theta", JSONB(), nullable=False),
        sa.Column("kc_n", JSONB(), nullable=False),
        sa.Column("tasks_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("diagnostic_cat_state")
