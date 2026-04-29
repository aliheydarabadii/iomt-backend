from datetime import date, datetime

from pydantic import Field, field_validator, model_validator
from typing import Self

from app.schemas.common import APIModel

SUPPORTED_SEX_VALUES = {"Male", "Female", "Other", "Unknown"}


class PatientSearchQuery(APIModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must not be empty")
        return cleaned


class RecordingResponse(APIModel):
    id: str
    areaId: str
    areaLabel: str
    areaShort: str
    capturedAt: datetime
    durationMs: int
    status: str
    audioUrl: str


class CreatePatientRequest(APIModel):
    fullName: str = Field(min_length=1, max_length=120)
    mrn: str = Field(min_length=1, max_length=32)
    sex: str = Field(min_length=1, max_length=16)
    dob: date
    latestVisit: datetime | None = None

    @field_validator("fullName", "mrn", "sex")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in SUPPORTED_SEX_VALUES:
            raise ValueError(
                f"sex must be one of: {', '.join(sorted(SUPPORTED_SEX_VALUES))}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if self.latestVisit and self.latestVisit.date() < self.dob:
            raise ValueError("latestVisit cannot be earlier than dob")
        return self


class PatientResponse(APIModel):
    id: str
    fullName: str
    mrn: str
    age: int
    sex: str
    dob: date
    latestVisit: datetime
    recordings: list[RecordingResponse]


class PatientSearchResponse(APIModel):
    patients: list[PatientResponse]
