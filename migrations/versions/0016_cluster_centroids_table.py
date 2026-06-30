"""Store cluster centroids in PostgreSQL instead of /tmp.

Revision ID: 0016
Revises: 0015
"""
from typing import Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA

revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cluster_centroids",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("n_clusters", sa.Integer(), nullable=False),
        sa.Column("kc_order", ARRAY(sa.String()), nullable=False),
        sa.Column("centroids_blob", BYTEA(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("cluster_centroids")
