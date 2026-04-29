from pathlib import Path

import pytest

from app.core.errors import ConflictError
from app.services.measurement_service import MeasurementService
from app.services.ble_sensor import LiveWaveformSnapshot


class FakeBLEService:
    def __init__(
        self,
        *,
        begin_allowed: bool = True,
        live_snapshot: LiveWaveformSnapshot | None = None,
        captured_samples: list[int] | None = None,
    ) -> None:
        self.begin_allowed = begin_allowed
        self.live_snapshot = live_snapshot
        self.captured_samples = captured_samples or []
        self.active_capture_patient_id: str | None = None

    def begin_capture(self, _patient_id: str) -> bool:
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

    def end_capture(self, _patient_id: str) -> list[int]:
        if self.active_capture_patient_id != _patient_id:
            return []
        samples = list(self.captured_samples)
        self.active_capture_patient_id = None
        self.captured_samples = []
        return samples

    def get_live_snapshot(self, *, size: int, idle_seed: int = 0) -> LiveWaveformSnapshot | None:
        return self.live_snapshot


def test_start_and_stop_recording_persists_new_history(db_session, test_settings) -> None:
    service = MeasurementService(
        db_session,
        settings=test_settings,
        ble_service=FakeBLEService(),
    )

    before = service.get_current_measurement(patient_id="patient_001", patient_name=None)
    before_count = len(before.records)

    started = service.start_recording(patient_id="patient_001", area_id="aortic")
    assert started.success is True

    during = service.get_current_measurement(patient_id="patient_001", patient_name=None)
    assert during.currentSession.isRecording is True
    assert during.currentSession.activeAreaId == "aortic"
    assert during.controls.canStop is True

    stopped = service.stop_recording(patient_id="patient_001")
    assert stopped.success is True

    after = service.get_current_measurement(patient_id="patient_001", patient_name=None)
    assert after.currentSession.isRecording is False
    assert len(after.records) == before_count + 1
    assert after.records[0].areaId == "aortic"
    assert after.records[0].audioUrl.startswith("/api/heart-recordings/")
    saved_path = Path(test_settings.audio_storage_dir) / f"{after.records[0].id}.wav"
    assert saved_path.exists()
    assert saved_path.stat().st_size > 44


def test_stop_without_active_session_raises_conflict(db_session, test_settings) -> None:
    service = MeasurementService(
        db_session,
        settings=test_settings,
        ble_service=FakeBLEService(),
    )

    with pytest.raises(ConflictError):
        service.stop_recording(patient_id="patient_002")


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

    service.start_recording(patient_id="patient_001", area_id="aortic")
    current = service.get_current_measurement(patient_id="patient_001", patient_name=None)

    assert current.currentSession.waveform == [0.1, 0.2, 0.3, 0.4]
    assert current.currentSession.streamStatus == "Receiving waveform"
    assert current.currentSession.signalQuality == "Good"
    assert service.ble_service.active_capture_patient_id == "patient_001"


def test_start_recording_raises_conflict_when_ble_device_is_busy(db_session, test_settings) -> None:
    service = MeasurementService(
        db_session,
        settings=test_settings,
        ble_service=FakeBLEService(begin_allowed=False),
    )

    with pytest.raises(ConflictError):
        service.start_recording(patient_id="patient_001", area_id="aortic")


def test_stop_recording_in_ble_mode_fails_when_no_samples_were_captured(
    db_session,
    test_settings,
) -> None:
    fake_ble_service = FakeBLEService(captured_samples=[])
    ble_settings = test_settings.model_copy(update={"ble_enabled": True})
    service = MeasurementService(
        db_session,
        settings=ble_settings,
        ble_service=fake_ble_service,
    )

    service.start_recording(patient_id="patient_001", area_id="aortic")

    with pytest.raises(ConflictError):
        service.stop_recording(patient_id="patient_001")
    assert fake_ble_service.active_capture_patient_id == "patient_001"
