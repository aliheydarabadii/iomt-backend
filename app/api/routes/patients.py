from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.common import ErrorResponse
from app.schemas.patients import CreatePatientRequest, PatientResponse, PatientSearchQuery, PatientSearchResponse
from app.services.patient_service import PatientService

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Create a new patient",
)
def create_patient(
    payload: CreatePatientRequest,
    db: Annotated[Session, Depends(get_db)],
) -> PatientResponse:
    return PatientService(db).create_patient(payload)


@router.get(
    "/search",
    response_model=PatientSearchResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Search patients by partial name",
)
def search_patients(
    query: Annotated[PatientSearchQuery, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> PatientSearchResponse:
    return PatientService(db).search_patients(query.name)
