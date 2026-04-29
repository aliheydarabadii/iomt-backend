from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.common import ErrorResponse
from app.schemas.recordings import RecordingAnalysisQuery, RecordingAnalysisResponse
from app.services.recording_service import RecordingService

router = APIRouter(prefix="/heart-recordings", tags=["heart-recordings"])


@router.get(
    "/{recording_id}/audio",
    responses={404: {"model": ErrorResponse}},
    summary="Get recorded audio for a heart recording",
)
def get_recording_audio(
    recording_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    return RecordingService(db).get_audio_response(recording_id)


@router.get(
    "/{recording_id}/analysis",
    response_model=RecordingAnalysisResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Analyze a recorded WAV file and return chart-ready PCG results",
)
def get_recording_analysis(
    recording_id: str,
    query: Annotated[RecordingAnalysisQuery, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> RecordingAnalysisResponse:
    return RecordingService(db).get_analysis(
        recording_id=recording_id,
        include_signals=query.includeSignals,
        save_filtered_wav=query.saveFilteredWav,
        max_points=query.maxPoints,
    )
