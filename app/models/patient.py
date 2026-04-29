from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(120), index=True)
    mrn: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(16))
    dob: Mapped[date] = mapped_column(Date)
    latest_visit: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    recordings: Mapped[list["HeartRecording"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["HeartMeasurementSession"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
