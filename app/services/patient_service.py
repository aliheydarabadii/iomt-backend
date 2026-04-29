import re
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import ConflictError
from app.core.time import ensure_utc
from app.models import HeartRecording, Patient
from app.schemas.patients import (
    CreatePatientRequest,
    PatientResponse,
    PatientSearchResponse,
    RecordingResponse,
)


def compute_match_rank(full_name: str, query: str) -> tuple[int, int]:
    normalized_name = full_name.strip().lower()
    normalized_query = query.strip().lower()
    if normalized_name == normalized_query:
        return (0, len(normalized_name))
    if normalized_name.startswith(normalized_query):
        return (1, len(normalized_name))
    tokens = normalized_name.split()
    if any(token.startswith(normalized_query) for token in tokens):
        return (2, len(normalized_name))
    return (3, len(normalized_name))


class PatientService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_patient(self, payload: CreatePatientRequest) -> PatientResponse:
        existing = self.db.scalar(select(Patient).where(Patient.mrn == payload.mrn))
        if existing:
            raise ConflictError(
                "patient_mrn_exists",
                "A patient with this MRN already exists",
                {"mrn": payload.mrn},
            )

        latest_visit = ensure_utc(payload.latestVisit) if payload.latestVisit else ensure_utc(datetime.utcnow())
        patient = Patient(
            id=self._generate_patient_id(),
            full_name=payload.fullName,
            mrn=payload.mrn,
            age=self._calculate_age(payload.dob, latest_visit.date()),
            sex=payload.sex,
            dob=payload.dob,
            latest_visit=latest_visit,
        )
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        patient.recordings = []
        return self._serialize_patient(patient)

    def search_patients(self, name: str) -> PatientSearchResponse:
        normalized = name.strip().lower()
        stmt = (
            select(Patient)
            .where(func.lower(Patient.full_name).contains(normalized))
            .options(selectinload(Patient.recordings))
        )
        patients = self.db.scalars(stmt).unique().all()
        patients.sort(
            key=lambda patient: (
                compute_match_rank(patient.full_name, normalized),
                -patient.latest_visit.timestamp(),
            )
        )
        return PatientSearchResponse(
            patients=[self._serialize_patient(patient) for patient in patients]
        )

    def resolve_patient(self, patient_id: str | None, patient_name: str | None) -> Patient | None:
        if patient_id:
            patient = self.db.get(Patient, patient_id)
            if patient:
                return patient
        if not patient_name:
            return None
        normalized = patient_name.strip().lower()
        stmt = (
            select(Patient)
            .where(func.lower(Patient.full_name).contains(normalized))
            .options(selectinload(Patient.recordings))
        )
        candidates = self.db.scalars(stmt).unique().all()
        if not candidates:
            return None
        candidates.sort(
            key=lambda patient: (
                compute_match_rank(patient.full_name, normalized),
                -patient.latest_visit.timestamp(),
            )
        )
        return candidates[0]

    def _serialize_patient(self, patient: Patient) -> PatientResponse:
        return PatientResponse(
            id=patient.id,
            fullName=patient.full_name,
            mrn=patient.mrn,
            age=patient.age,
            sex=patient.sex,
            dob=patient.dob,
            latestVisit=ensure_utc(patient.latest_visit),
            recordings=[
                self.serialize_recording(recording)
                for recording in sorted(
                    patient.recordings,
                    key=lambda item: item.stopped_at,
                    reverse=True,
                )
            ],
        )

    @staticmethod
    def serialize_recording(recording: HeartRecording) -> RecordingResponse:
        return RecordingResponse(
            id=recording.id,
            areaId=recording.area_id,
            areaLabel=recording.area_label,
            areaShort=recording.area_short,
            capturedAt=ensure_utc(recording.stopped_at),
            durationMs=recording.duration_ms,
            status=recording.status,
            audioUrl=recording.audio_url,
        )

    def _generate_patient_id(self) -> str:
        stmt = select(Patient.id)
        ids = self.db.scalars(stmt).all()
        highest = 0
        for value in ids:
            match = re.fullmatch(r"patient_(\d+)", value)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"patient_{highest + 1:03d}"

    @staticmethod
    def _calculate_age(dob: date, reference_date: date) -> int:
        years = reference_date.year - dob.year
        if (reference_date.month, reference_date.day) < (dob.month, dob.day):
            years -= 1
        return years
