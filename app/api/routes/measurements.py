from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.common import ErrorResponse
from app.schemas.measurements import (
    ActionResponse,
    CurrentMeasurementQuery,
    CurrentMeasurementResponse,
    RecordActionRequest,
    RecordingCommandRequest,
)
from app.services.measurement_service import MeasurementService

router = APIRouter(prefix="/heart-measurements", tags=["heart-measurements"])


@router.get(
    "/current",
    response_model=CurrentMeasurementResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Get the current measurement state for a patient",
)
def get_current_measurement(
    query: Annotated[CurrentMeasurementQuery, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> CurrentMeasurementResponse:
    return MeasurementService(db).get_current_measurement(
        patient_id=query.patientId,
        patient_name=query.patientName,
    )


@router.post(
    "/{patient_id}/record",
    response_model=ActionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Start a new heart sound recording session",
)
def start_recording(
    patient_id: str,
    payload: RecordActionRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ActionResponse:
    return MeasurementService(db).start_recording(patient_id=patient_id, area_id=payload.areaId)


@router.post(
    "/{patient_id}/stop",
    response_model=ActionResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Stop the active heart sound recording session",
)
def stop_recording(
    patient_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> ActionResponse:
    return MeasurementService(db).stop_recording(patient_id=patient_id)


@router.post(
    "/{patient_id}/recording",
    response_model=ActionResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Start or stop recording through a single action endpoint",
)
def control_recording(
    patient_id: str,
    payload: RecordingCommandRequest,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> ActionResponse:
    service = MeasurementService(db)
    if payload.action == "start":
        response.status_code = status.HTTP_201_CREATED
        return service.start_recording(patient_id=patient_id, area_id=payload.areaId)
    return service.stop_recording(patient_id=patient_id)
