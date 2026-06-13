from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from app.core.constants import AUSCULTATION_AREAS
from app.schemas.common import APIModel
from app.schemas.patients import RecordingResponse


class CurrentMeasurementQuery(APIModel):
    patientId: str | None = Field(default=None, min_length=1, max_length=32)
    patientName: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("patientId", "patientName")
    @classmethod
    def trim_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if not self.patientId and not self.patientName:
            raise ValueError("patientId or patientName is required")
        return self


class CurrentSessionResponse(APIModel):
    isRecording: bool
    runtimeMs: int
    streamStatus: str
    signalQuality: str
    activeAreaId: str | None = None
    waveform: list[float]


class MeasurementControlsResponse(APIModel):
    canRecord: bool
    recordUrl: str
    recordMethod: str


class CurrentMeasurementResponse(APIModel):
    sourceLabel: str
    updatedAt: datetime
    currentSession: CurrentSessionResponse
    controls: MeasurementControlsResponse
    records: list[RecordingResponse]


class RecordActionRequest(APIModel):
    areaId: str

    @field_validator("areaId")
    @classmethod
    def validate_area_id(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned not in AUSCULTATION_AREAS:
            raise ValueError(f"areaId must be one of: {', '.join(AUSCULTATION_AREAS)}")
        return cleaned


class ActionResponse(APIModel):
    success: bool
    message: str
