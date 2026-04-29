from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class RecordingAnalysisQuery(APIModel):
    includeSignals: bool = True
    saveFilteredWav: bool = False
    maxPoints: int = Field(default=3000, ge=100, le=10000)


class RecordingPlotResponse(APIModel):
    sampleRateHz: int
    durationS: float
    pointCount: int
    maxPoints: int
    downsampled: bool
    timeAxisS: list[float]
    amplitude: list[float]
    envelope: list[float]
    peakTimesS: list[float] = Field(default_factory=list)
    s1TimesS: list[float] = Field(default_factory=list)
    s2TimesS: list[float] = Field(default_factory=list)


class RecordingAnalysisResponse(APIModel):
    recordingId: str
    audioUrl: str
    generatedAt: datetime
    plot: RecordingPlotResponse | None = None
    analysis: dict[str, Any]
