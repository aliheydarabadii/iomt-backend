from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.errors import ConflictError
from app.core.time import utcnow
from app.models import HeartMeasurementSession
from app.services.measurement_service import MeasurementService
from app.services.ble_sensor import LiveWaveformSnapshot


class FakeBLEService:
    def __init__(
        self,
        *,
        begin_allowed: bool = True,
        live_snapshot: LiveWaveformSnapshot | None = None,
        captured_samples: list[int] | None = None,
        enabled: bool = False,
    ) -> None:
        self.begin_allowed = begin_allowed
        self.live_snapshot = live_snapshot
        self.captured_samples = captured_samples or []
        self.is_enabled = enabled
        self.active_capture_patient_id: str | None = None

    def begin_capture(self, _patient_id: str, _patient_name: str | None = None) -> bool:
        if not self.begin_allowed:
            return False
        self.active_capture_patient_id = _patient_id
        self.captured_samples = []
        return True

    def ensure_capture_binding(self, patient_id: str) -> bool:
        if self.active_capture_patient_id is None:
            self.active_capture_patient_id = patient_id
            return True
        return self.active_capture_patient_id == patient_id

    def get_capture_sample_count(self, patient_id: str) -> int:
        if self.active_capture_patient_id != patient_id:
            return 0
        return len(self.captured_samples)

    def is_capture_active(self, patient_id: str) -> bool:
        return self.active_capture_patient_id == patient_id

    def end_capture(self, _patient_id: str) -> list[int]:
        if self.active_capture_patient_id != _patient_id:
            return []
        samples = list(self.captured_samples)
        self.active_capture_patient_id = None
        self.captured_samples = []
        return samples

    def get_live_snapshot(self, *, size: int, idle_seed: int = 0) -> LiveWaveformSnapshot | None:
        return self.live_snapshot


def _insert_active_session(db_session, patient_id: str, area_id: str) -> None:
    now = utcnow()
    db_session.add(
        HeartMeasurementSession(
            id=f"session_test_{patient_id}",
            patient_id=patient_id,
            area_id=area_id,
            area_label="Aortic",
            area_short="2nd ICS",
            state="recording",
            is_locked=True,
            started_at=now,
            stopped_at=None,
            stream_status="Receiving waveform",
            signal_quality="Good",
            waveform_seed=12345,
            created_at=now,
            updated_at=now,
        )
    )
    db_session.commit()


def test_record_persists_new_history(db_session, test_settings) -> None:
    service = MeasurementService(
        db_session,
        settings=test_settings,
        ble_service=FakeBLEService(),
    )

    before = service.get_current_measurement(patient_id="patient_001", patient_name=None)
    before_count = len(before.records)

    stored = service.record(patient_id="patient_001", area_id="aortic")
    assert stored.success is True
    assert stored.message == "Recording stored"

    after = service.get_current_measurement(patient_id="patient_001", patient_name=None)
    assert after.currentSession.isRecording is False
    assert len(after.records) == before_count + 1
    assert after.records[0].areaId == "aortic"
    assert after.records[0].audioUrl.startswith("/api/heart-recordings/")
    saved_path = Path(test_settings.audio_storage_dir) / f"{after.records[0].id}.wav"
    assert saved_path.exists()
    assert saved_path.stat().st_size > 44


def test_current_measurement_uses_ble_waveform_when_available(db_session, test_settings) -> None:
    live_snapshot = LiveWaveformSnapshot(
        waveform=[0.1, 0.2, 0.3, 0.4],
        stream_status="Receiving waveform",
        signal_quality="Good",
        has_live_data=True,
    )
    service = MeasurementService(
        db_session,
        settings=test_settings,
        ble_service=FakeBLEService(live_snapshot=live_snapshot),
    )

    _insert_active_session(db_session, "patient_001", "aortic")
    current = service.get_current_measurement(patient_id="patient_001", patient_name=None)

    assert current.currentSession.isRecording is True
    assert current.currentSession.waveform == [0.1, 0.2, 0.3, 0.4]
    assert current.currentSession.streamStatus == "Receiving waveform"
    assert current.currentSession.signalQuality == "Good"


def test_record_raises_conflict_when_ble_device_is_busy(db_session, test_settings) -> None:
    service = MeasurementService(
        db_session,
        settings=test_settings,
        ble_service=FakeBLEService(begin_allowed=False),
    )

    with pytest.raises(ConflictError):
        service.record(patient_id="patient_001", area_id="aortic")


def test_record_in_ble_mode_fails_when_no_samples_were_captured(
    db_session,
    test_settings,
) -> None:
    fake_ble_service = FakeBLEService(captured_samples=[])
    ble_settings = test_settings.model_copy(
        update={
            "ble_enabled": True,
            "ble_analysis_time_seconds": 0,
            "ble_capture_grace_seconds": 0.0,
        }
    )
    service = MeasurementService(
        db_session,
        settings=ble_settings,
        ble_service=fake_ble_service,
    )

    with pytest.raises(ConflictError):
        service.record(patient_id="patient_001", area_id="aortic")

    session = db_session.scalar(
        select(HeartMeasurementSession)
        .where(HeartMeasurementSession.patient_id == "patient_001")
        .order_by(HeartMeasurementSession.started_at.desc())
    )
    assert session is not None
    assert session.state == "failed"
    assert session.is_locked is False
    assert fake_ble_service.active_capture_patient_id is None


def test_current_measurement_recovers_session_left_by_backend_restart(
    db_session,
    test_settings,
) -> None:
    ble_settings = test_settings.model_copy(update={"ble_enabled": True})
    fake_ble_service = FakeBLEService(enabled=True)
    service = MeasurementService(
        db_session,
        settings=ble_settings,
        ble_service=fake_ble_service,
    )
    _insert_active_session(db_session, "patient_001", "aortic")

    current = service.get_current_measurement(
        patient_id="patient_001",
        patient_name=None,
    )

    assert current.currentSession.isRecording is False
    assert current.controls.canRecord is True
    session = db_session.get(
        HeartMeasurementSession,
        "session_test_patient_001",
    )
    assert session is not None
    assert session.state == "failed"
    assert session.is_locked is False


def test_record_exception_releases_capture_and_marks_session_failed(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    fake_ble_service = FakeBLEService(enabled=True)
    ble_settings = test_settings.model_copy(update={"ble_enabled": True})
    service = MeasurementService(
        db_session,
        settings=ble_settings,
        ble_service=fake_ble_service,
    )

    def fail_capture(_patient_id: str):
        raise RuntimeError("sensor disconnected")

    monkeypatch.setattr(service, "_await_capture", fail_capture)

    with pytest.raises(RuntimeError, match="sensor disconnected"):
        service.record(patient_id="patient_001", area_id="aortic")

    session = db_session.scalar(
        select(HeartMeasurementSession)
        .where(HeartMeasurementSession.patient_id == "patient_001")
        .order_by(HeartMeasurementSession.started_at.desc())
    )
    assert session is not None
    assert session.state == "failed"
    assert session.is_locked is False
    assert fake_ble_service.active_capture_patient_id is None
