"""initial schema

Revision ID: 20260412_0001
Revises:
Create Date: 2026-04-12 23:59:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260412_0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "patients",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=False),
        sa.Column("mrn", sa.String(length=32), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("sex", sa.String(length=16), nullable=False),
        sa.Column("dob", sa.Date(), nullable=False),
        sa.Column("latest_visit", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_patients_full_name", "patients", ["full_name"], unique=False)
    op.create_index("ix_patients_latest_visit", "patients", ["latest_visit"], unique=False)
    op.create_index("ix_patients_mrn", "patients", ["mrn"], unique=True)

    op.create_table(
        "heart_measurement_sessions",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("patient_id", sa.String(length=32), nullable=False),
        sa.Column("area_id", sa.String(length=32), nullable=False),
        sa.Column("area_label", sa.String(length=64), nullable=False),
        sa.Column("area_short", sa.String(length=120), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("is_locked", sa.Boolean(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stream_status", sa.String(length=64), nullable=False),
        sa.Column("signal_quality", sa.String(length=32), nullable=False),
        sa.Column("waveform_seed", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_heart_measurement_sessions_area_id",
        "heart_measurement_sessions",
        ["area_id"],
        unique=False,
    )
    op.create_index(
        "ix_heart_measurement_sessions_patient_id",
        "heart_measurement_sessions",
        ["patient_id"],
        unique=False,
    )
    op.create_index(
        "ix_heart_measurement_sessions_started_at",
        "heart_measurement_sessions",
        ["started_at"],
        unique=False,
    )
    op.create_index(
        "ix_heart_measurement_sessions_state",
        "heart_measurement_sessions",
        ["state"],
        unique=False,
    )

    op.create_table(
        "heart_recordings",
        sa.Column("id", sa.String(length=40), nullable=False),
        sa.Column("patient_id", sa.String(length=32), nullable=False),
        sa.Column("area_id", sa.String(length=32), nullable=False),
        sa.Column("area_label", sa.String(length=64), nullable=False),
        sa.Column("area_short", sa.String(length=120), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("audio_url", sa.String(length=255), nullable=False),
        sa.Column("waveform_summary", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_heart_recordings_area_id", "heart_recordings", ["area_id"], unique=False)
    op.create_index(
        "ix_heart_recordings_patient_id",
        "heart_recordings",
        ["patient_id"],
        unique=False,
    )
    op.create_index(
        "ix_heart_recordings_stopped_at",
        "heart_recordings",
        ["stopped_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_heart_recordings_stopped_at", table_name="heart_recordings")
    op.drop_index("ix_heart_recordings_patient_id", table_name="heart_recordings")
    op.drop_index("ix_heart_recordings_area_id", table_name="heart_recordings")
    op.drop_table("heart_recordings")

    op.drop_index("ix_heart_measurement_sessions_state", table_name="heart_measurement_sessions")
    op.drop_index("ix_heart_measurement_sessions_started_at", table_name="heart_measurement_sessions")
    op.drop_index("ix_heart_measurement_sessions_patient_id", table_name="heart_measurement_sessions")
    op.drop_index("ix_heart_measurement_sessions_area_id", table_name="heart_measurement_sessions")
    op.drop_table("heart_measurement_sessions")

    op.drop_index("ix_patients_mrn", table_name="patients")
    op.drop_index("ix_patients_latest_visit", table_name="patients")
    op.drop_index("ix_patients_full_name", table_name="patients")
    op.drop_table("patients")
