from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.common import ErrorResponse
from app.schemas.measurements import (
    ActionResponse,
    CurrentMeasurementQuery,
    CurrentMeasurementResponse,
    RecordActionRequest,
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
    summary="Record a heart sound session for the configured duration and store it",
)
def record(
    patient_id: str,
    payload: RecordActionRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ActionResponse:
    return MeasurementService(db).record(patient_id=patient_id, area_id=payload.areaId)
