from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class HeartMeasurementSession(Base):
    __tablename__ = "heart_measurement_sessions"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    area_id: Mapped[str] = mapped_column(String(32), index=True)
    area_label: Mapped[str] = mapped_column(String(64))
    area_short: Mapped[str] = mapped_column(String(120))
    state: Mapped[str] = mapped_column(String(24), index=True)
    is_locked: Mapped[bool] = mapped_column(Boolean, default=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stream_status: Mapped[str] = mapped_column(String(64))
    signal_quality: Mapped[str] = mapped_column(String(32))
    waveform_seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    patient: Mapped["Patient"] = relationship(back_populates="sessions")
