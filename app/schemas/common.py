from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ErrorDetail(APIModel):
    code: str
    message: str
    details: dict[str, Any] | list[Any] = Field(default_factory=dict)


class ErrorResponse(APIModel):
    error: ErrorDetail


class BLEStatusResponse(APIModel):
    enabled: bool
    connectionState: str
    connectionLabel: str
    isRunning: bool
    deviceName: str | None = None
    serviceUuid: str | None = None
    characteristicUuid: str | None = None
    activeCapturePatientId: str | None = None
    recentSampleCount: int
    capturedSampleCount: int
    notificationCount: int
    lastNotificationAt: datetime | None = None
    lastError: str | None = None


class BLEDeviceResponse(APIModel):
    name: str | None = None
    address: str
    localName: str | None = None
    serviceUuids: list[str] = Field(default_factory=list)
    rssi: int | None = None
    manufacturerDataKeys: list[int] = Field(default_factory=list)
    isTargetMatch: bool
    matchedBy: list[str] = Field(default_factory=list)


class BLEDeviceScanResponse(APIModel):
    timeoutSeconds: float
    targetDeviceName: str | None = None
    targetServiceUuid: str | None = None
    devices: list[BLEDeviceResponse] = Field(default_factory=list)
