import random
import time
from uuid import uuid4

from sqlalchemy import desc, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings, get_settings
from app.core.constants import (
    AUSCULTATION_AREAS,
    DEFAULT_RECORDING_STATUS,
    DEFAULT_SIGNAL_QUALITY,
    IDLE_SIGNAL_QUALITY,
    IDLE_STREAM_STATUS,
    LIVE_STREAM_STATUS,
)
from app.core.errors import ConflictError, NotFoundError
from app.core.time import ensure_utc, utcnow
from app.models import HeartMeasurementSession, HeartRecording, Patient
from app.schemas.measurements import (
    ActionResponse,
    CurrentMeasurementResponse,
    CurrentSessionResponse,
    MeasurementControlsResponse,
)
from app.services.audio_storage import AudioStorage
from app.services.ble_sensor import BLEIngestionService, get_ble_ingestion_service
from app.services.patient_service import PatientService
from app.services.recording_service import RecordingService
from app.services.waveform import (
    generate_idle_waveform,
    generate_live_waveform,
    normalize_waveform,
    summarize_waveform,
)


class MeasurementService:
    def __init__(
        self,
        db: Session,
        settings: Settings | None = None,
        ble_service: BLEIngestionService | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.patient_service = PatientService(db)
        self.audio_storage = AudioStorage(
            self.settings.audio_base_url,
            self.settings.audio_storage_dir,
        )
        self.recording_service = RecordingService(db, self.settings)
        self.ble_service = ble_service or get_ble_ingestion_service(self.settings)

    def get_current_measurement(
        self,
        *,
        patient_id: str | None,
        patient_name: str | None,
    ) -> CurrentMeasurementResponse:
        patient = self._get_patient_or_404(patient_id, patient_name)
        active_session = self._get_active_session(patient.id)
        if active_session:
            started_at = ensure_utc(active_session.started_at)
            runtime_ms = max(
                int((utcnow() - started_at).total_seconds() * 1000),
                0,
            )
            ble_snapshot = self.ble_service.get_live_snapshot(
                size=self.settings.live_waveform_size,
                idle_seed=sum(ord(char) for char in patient.id),
            )
            if ble_snapshot:
                waveform = ble_snapshot.waveform
                stream_status = ble_snapshot.stream_status
                signal_quality = ble_snapshot.signal_quality
            else:
                waveform = generate_live_waveform(
                    size=self.settings.live_waveform_size,
                    runtime_ms=runtime_ms,
                    area_id=active_session.area_id,
                    seed=active_session.waveform_seed,
                )
                stream_status = active_session.stream_status
                signal_quality = active_session.signal_quality
            current_session = CurrentSessionResponse(
                isRecording=True,
                runtimeMs=runtime_ms,
                streamStatus=stream_status,
                signalQuality=signal_quality,
                activeAreaId=active_session.area_id,
                waveform=waveform,
            )
        else:
            current_session = CurrentSessionResponse(
                isRecording=False,
                runtimeMs=0,
                streamStatus=IDLE_STREAM_STATUS,
                signalQuality=IDLE_SIGNAL_QUALITY,
                activeAreaId=None,
                waveform=generate_idle_waveform(
                    size=self.settings.idle_waveform_size,
                    seed=sum(ord(char) for char in patient.id),
                ),
            )

        recordings = self._list_recordings(patient.id)
        controls = MeasurementControlsResponse(
            canRecord=not bool(active_session),
            recordUrl=f"{self.settings.api_prefix}/heart-measurements/{patient.id}/record",
            recordMethod="POST",
        )
        return CurrentMeasurementResponse(
            sourceLabel="REST endpoint",
            updatedAt=utcnow(),
            currentSession=current_session,
            controls=controls,
            records=recordings,
        )

    def record(self, *, patient_id: str, area_id: str) -> ActionResponse:
        patient = self._get_patient_or_404(patient_id, None)
        if area_id not in AUSCULTATION_AREAS:
            from app.core.errors import BadRequestError

            raise BadRequestError(
                "invalid_area_id",
                "areaId is not supported",
                {"supportedAreaIds": list(AUSCULTATION_AREAS)},
            )
        if self._get_active_session(patient.id):
            raise ConflictError(
                "recording_already_running",
                "Recording is already in progress for this patient",
            )
        if not self.ble_service.begin_capture(patient.id, patient.full_name):
            raise ConflictError(
                "ble_device_busy",
                "BLE sensor is already streaming to another active patient session",
            )

        area_meta = AUSCULTATION_AREAS[area_id]
        started_at = utcnow()
        session = HeartMeasurementSession(
            id=f"session_{uuid4().hex[:12]}",
            patient_id=patient.id,
            area_id=area_id,
            area_label=area_meta["label"],
            area_short=area_meta["short"],
            state="recording",
            is_locked=True,
            started_at=started_at,
            stopped_at=None,
            stream_status=LIVE_STREAM_STATUS,
            signal_quality=DEFAULT_SIGNAL_QUALITY,
            waveform_seed=random.randint(10_000, 99_999),
            created_at=started_at,
            updated_at=started_at,
        )
        patient.latest_visit = started_at
        self.db.add(session)
        self.db.commit()

        # PCGClient.analyze() runs for the configured capture window and yields
        # BLE batches into the capture buffer.
        captured_samples = self._await_capture(patient.id)

        now = utcnow()
        duration_ms = max(int((now - started_at).total_seconds() * 1000), 1000)
        if self.settings.ble_enabled and not captured_samples:
            self._mark_session_stopped(session, now)
            self.db.commit()
            raise ConflictError(
                "no_ble_audio_captured",
                "No BLE samples were captured for this recording session",
                {
                    "durationMs": duration_ms,
                    "expectedSampleCount": int(
                        duration_ms * self.settings.ble_sample_rate / 1000
                    ),
                },
            )
        if captured_samples:
            waveform = normalize_waveform(captured_samples)
        else:
            waveform = generate_live_waveform(
                size=min(self.settings.live_waveform_size, 180),
                runtime_ms=duration_ms,
                area_id=session.area_id,
                seed=session.waveform_seed,
            )
        recording_id = f"rec_{uuid4().hex[:10]}"
        self.audio_storage.save_wav(
            recording_id=recording_id,
            raw_samples=captured_samples,
            waveform=waveform,
            source_sample_rate=self.settings.ble_sample_rate,
            audio_sample_rate=self.settings.audio_sample_rate,
            gain=self.settings.audio_gain,
        )
        recording = HeartRecording(
            id=recording_id,
            patient_id=patient.id,
            area_id=session.area_id,
            area_label=session.area_label,
            area_short=session.area_short,
            started_at=started_at,
            stopped_at=now,
            duration_ms=duration_ms,
            status=DEFAULT_RECORDING_STATUS,
            audio_url=self.recording_service.build_audio_url(recording_id),
            waveform_summary=self._build_waveform_summary(
                waveform=waveform,
                used_ble=bool(captured_samples),
                captured_sample_count=len(captured_samples),
            ),
        )
        self._mark_session_stopped(session, now)
        patient.latest_visit = now
        self.db.add(recording)
        self.db.commit()
        return ActionResponse(success=True, message="Recording stored")

    def _await_capture(self, patient_id: str) -> list[int]:
        """Block until the fixed-duration capture window finishes, then drain it.

        Only waits when BLE is enabled; otherwise there is nothing streaming and
        we fall back to a synthetic waveform.
        """
        if self.settings.ble_enabled:
            deadline = time.monotonic() + self.settings.ble_analysis_time_seconds
            while time.monotonic() < deadline:
                time.sleep(0.05)
        return self.ble_service.end_capture(patient_id)

    @staticmethod
    def _mark_session_stopped(session: HeartMeasurementSession, now) -> None:
        session.state = "stopped"
        session.stopped_at = now
        session.stream_status = "Stopped"
        session.updated_at = now

    def _get_patient_or_404(self, patient_id: str | None, patient_name: str | None) -> Patient:
        patient = self.patient_service.resolve_patient(patient_id, patient_name)
        if not patient:
            raise NotFoundError("patient_not_found", "Patient was not found")

        if "recordings" not in patient.__dict__:
            patient = self.db.scalar(
                select(Patient)
                .where(Patient.id == patient.id)
                .options(selectinload(Patient.recordings))
            )
        return patient

    def _get_active_session(self, patient_id: str) -> HeartMeasurementSession | None:
        stmt = (
            select(HeartMeasurementSession)
            .where(
                HeartMeasurementSession.patient_id == patient_id,
                HeartMeasurementSession.state == "recording",
            )
            .order_by(desc(HeartMeasurementSession.started_at))
        )
        return self.db.scalar(stmt)

    def _list_recordings(self, patient_id: str):
        stmt = (
            select(HeartRecording)
            .where(HeartRecording.patient_id == patient_id)
            .order_by(desc(HeartRecording.stopped_at))
        )
        return [
            self.patient_service.serialize_recording(recording)
            for recording in self.db.scalars(stmt).all()
        ]

    def _build_waveform_summary(
        self,
        *,
        waveform: list[float],
        used_ble: bool,
        captured_sample_count: int,
    ) -> dict:
        summary = summarize_waveform(waveform)
        summary.update(
            {
                "source": "ble" if used_ble else "synthetic",
                "rawSampleCount": captured_sample_count,
                "sampleRate": self.settings.ble_sample_rate,
                "batchSize": self.settings.ble_batch_size,
                "audioSampleRate": self.settings.audio_sample_rate,
                "audioGain": self.settings.audio_gain,
                "timerIntervalMs": self.settings.ble_timer_interval_ms,
                "samplesPerTick": self.settings.ble_samples_per_tick,
            }
        )
        return summary
