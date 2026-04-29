from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class HeartRecording(Base):
    __tablename__ = "heart_recordings"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    area_id: Mapped[str] = mapped_column(String(32), index=True)
    area_label: Mapped[str] = mapped_column(String(64))
    area_short: Mapped[str] = mapped_column(String(120))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(64))
    audio_url: Mapped[str] = mapped_column(String(255))
    waveform_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="recordings")
