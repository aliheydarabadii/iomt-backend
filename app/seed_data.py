from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.models import HeartRecording, Patient


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


PATIENTS = [
    {
        "id": "patient_001",
        "full_name": "John Smith",
        "mrn": "MRN-10021",
        "age": 58,
        "sex": "Male",
        "dob": date(1967, 3, 14),
        "latest_visit": dt("2026-04-12T10:20:00+00:00"),
    },
    {
        "id": "patient_002",
        "full_name": "Joanna Smith",
        "mrn": "MRN-10022",
        "age": 46,
        "sex": "Female",
        "dob": date(1980, 7, 8),
        "latest_visit": dt("2026-04-11T14:10:00+00:00"),
    },
    {
        "id": "patient_003",
        "full_name": "Alice Johnson",
        "mrn": "MRN-10023",
        "age": 64,
        "sex": "Female",
        "dob": date(1961, 1, 22),
        "latest_visit": dt("2026-04-10T08:45:00+00:00"),
    },
    {
        "id": "patient_004",
        "full_name": "Michael Brown",
        "mrn": "MRN-10024",
        "age": 71,
        "sex": "Male",
        "dob": date(1954, 11, 2),
        "latest_visit": dt("2026-04-09T16:30:00+00:00"),
    },
    {
        "id": "patient_005",
        "full_name": "Sarah Lee",
        "mrn": "MRN-10025",
        "age": 52,
        "sex": "Female",
        "dob": date(1974, 9, 19),
        "latest_visit": dt("2026-04-08T12:00:00+00:00"),
    },
]

RECORDINGS = [
    {
        "id": "rec_001",
        "patient_id": "patient_001",
        "area_id": "aortic",
        "area_label": "Aortic",
        "area_short": "2nd ICS · right sternal border",
        "started_at": dt("2026-04-10T08:59:47+00:00"),
        "stopped_at": dt("2026-04-10T09:00:00+00:00"),
        "duration_ms": 12800,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_001.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.06, "max": 0.92, "avg": 0.48},
    },
    {
        "id": "rec_002",
        "patient_id": "patient_001",
        "area_id": "mitral",
        "area_label": "Mitral / Apex",
        "area_short": "5th ICS · apex / left midclavicular line",
        "started_at": dt("2026-03-29T15:22:30+00:00"),
        "stopped_at": dt("2026-03-29T15:22:42+00:00"),
        "duration_ms": 12100,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_002.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.07, "max": 0.94, "avg": 0.5},
    },
    {
        "id": "rec_003",
        "patient_id": "patient_002",
        "area_id": "pulmonic",
        "area_label": "Pulmonic",
        "area_short": "2nd ICS · left sternal border",
        "started_at": dt("2026-04-11T14:09:45+00:00"),
        "stopped_at": dt("2026-04-11T14:10:00+00:00"),
        "duration_ms": 14750,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_003.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.08, "max": 0.95, "avg": 0.49},
    },
    {
        "id": "rec_004",
        "patient_id": "patient_002",
        "area_id": "tricuspid",
        "area_label": "Tricuspid",
        "area_short": "4th ICS · lower left sternal border",
        "started_at": dt("2026-04-02T10:14:10+00:00"),
        "stopped_at": dt("2026-04-02T10:14:23+00:00"),
        "duration_ms": 13320,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_004.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.09, "max": 0.91, "avg": 0.47},
    },
    {
        "id": "rec_005",
        "patient_id": "patient_003",
        "area_id": "erbs-point",
        "area_label": "Erb's Point",
        "area_short": "3rd ICS · left sternal border",
        "started_at": dt("2026-04-10T08:44:45+00:00"),
        "stopped_at": dt("2026-04-10T08:45:00+00:00"),
        "duration_ms": 15200,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_005.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.05, "max": 0.89, "avg": 0.45},
    },
    {
        "id": "rec_006",
        "patient_id": "patient_003",
        "area_id": "mitral",
        "area_label": "Mitral / Apex",
        "area_short": "5th ICS · apex / left midclavicular line",
        "started_at": dt("2026-03-18T11:05:00+00:00"),
        "stopped_at": dt("2026-03-18T11:05:11+00:00"),
        "duration_ms": 10980,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_006.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.1, "max": 0.96, "avg": 0.52},
    },
    {
        "id": "rec_007",
        "patient_id": "patient_004",
        "area_id": "aortic",
        "area_label": "Aortic",
        "area_short": "2nd ICS · right sternal border",
        "started_at": dt("2026-04-09T16:29:43+00:00"),
        "stopped_at": dt("2026-04-09T16:30:00+00:00"),
        "duration_ms": 17100,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_007.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.11, "max": 0.97, "avg": 0.54},
    },
    {
        "id": "rec_008",
        "patient_id": "patient_004",
        "area_id": "tricuspid",
        "area_label": "Tricuspid",
        "area_short": "4th ICS · lower left sternal border",
        "started_at": dt("2026-03-28T09:40:02+00:00"),
        "stopped_at": dt("2026-03-28T09:40:16+00:00"),
        "duration_ms": 13950,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_008.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.06, "max": 0.9, "avg": 0.49},
    },
    {
        "id": "rec_009",
        "patient_id": "patient_005",
        "area_id": "pulmonic",
        "area_label": "Pulmonic",
        "area_short": "2nd ICS · left sternal border",
        "started_at": dt("2026-04-08T11:59:50+00:00"),
        "stopped_at": dt("2026-04-08T12:00:00+00:00"),
        "duration_ms": 9800,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_009.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.08, "max": 0.88, "avg": 0.46},
    },
    {
        "id": "rec_010",
        "patient_id": "patient_005",
        "area_id": "mitral",
        "area_label": "Mitral / Apex",
        "area_short": "5th ICS · apex / left midclavicular line",
        "started_at": dt("2026-04-12T11:54:50+00:00"),
        "stopped_at": dt("2026-04-12T11:55:00+00:00"),
        "duration_ms": 9600,
        "status": "Stored on server",
        "audio_url": "https://example.com/audio/rec_010.wav",
        "waveform_summary": {"sampleCount": 180, "min": 0.07, "max": 0.93, "avg": 0.5},
    },
]


def seed_database(db: Session) -> None:
    for payload in PATIENTS:
        patient = db.get(Patient, payload["id"])
        if not patient:
            patient = Patient(id=payload["id"])
            db.add(patient)
        patient.full_name = payload["full_name"]
        patient.mrn = payload["mrn"]
        patient.age = payload["age"]
        patient.sex = payload["sex"]
        patient.dob = payload["dob"]
        patient.latest_visit = payload["latest_visit"]

    db.flush()

    for payload in RECORDINGS:
        recording = db.get(HeartRecording, payload["id"])
        if not recording:
            recording = HeartRecording(id=payload["id"])
            db.add(recording)
        recording.patient_id = payload["patient_id"]
        recording.area_id = payload["area_id"]
        recording.area_label = payload["area_label"]
        recording.area_short = payload["area_short"]
        recording.started_at = payload["started_at"]
        recording.stopped_at = payload["stopped_at"]
        recording.duration_ms = payload["duration_ms"]
        recording.status = payload["status"]
        recording.audio_url = payload["audio_url"]
        recording.waveform_summary = payload["waveform_summary"]

    db.commit()
