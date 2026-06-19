import asyncio
import time
from contextlib import suppress
from types import SimpleNamespace

import app.services.ble_sensor as ble_sensor
import app.services.pcg_ble_client as pcg_ble_client
from app.services.ble_sensor import BLEIngestionService


class FakePCGClient:
    instances = []

    def __init__(self, device_name: str, **kwargs) -> None:
        self.device_name = device_name
        self.kwargs = kwargs
        self.connected = False
        self.analyze_calls = []
        FakePCGClient.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def analyze(self, **kwargs):
        self.analyze_calls.append(kwargs)
        yield [100, 120, 140]


class FlakyCapturePCGClient(FakePCGClient):
    instances = []

    def __init__(self, device_name: str, **kwargs) -> None:
        super().__init__(device_name, **kwargs)
        FlakyCapturePCGClient.instances.append(self)

    async def analyze(self, **kwargs):
        self.analyze_calls.append(kwargs)
        if len(FlakyCapturePCGClient.instances) == 1:
            self.connected = False
            raise pcg_ble_client.BLEConnectionError("lost during capture")
        yield [200, 220, 240]


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


async def _stop_service(service: BLEIngestionService) -> None:
    future = service._future
    service.stop()
    if future:
        with suppress(asyncio.CancelledError):
            await asyncio.wrap_future(future)


def test_stale_receiving_snapshot_reports_connected(test_settings, monkeypatch) -> None:
    monkeypatch.setattr(ble_sensor, "BleakClient", object())
    monkeypatch.setattr(ble_sensor, "BleakScanner", object())
    settings = test_settings.model_copy(
        update={
            "ble_enabled": True,
            "ble_stale_after_seconds": 0.5,
        }
    )
    service = BLEIngestionService(settings)

    service._consume_samples([100, 120, 140])
    with service._lock:
        service._connection_state = "receiving"
        service._last_batch_monotonic = time.monotonic() - 1.0

    snapshot = service.get_live_snapshot(size=3)
    status = service.get_status()

    assert snapshot is not None
    assert snapshot.has_live_data is False
    assert snapshot.stream_status == "Connected to sensor"
    assert status.connection_label == "Connected to sensor"


def test_recent_receiving_snapshot_reports_live_waveform(test_settings, monkeypatch) -> None:
    monkeypatch.setattr(ble_sensor, "BleakClient", object())
    monkeypatch.setattr(ble_sensor, "BleakScanner", object())
    settings = test_settings.model_copy(update={"ble_enabled": True})
    service = BLEIngestionService(settings)

    with service._lock:
        service._active_capture_patient_id = "patient_001"
        service._connection_state = "receiving"
    service._consume_samples([100, 120, 140])

    snapshot = service.get_live_snapshot(size=3)
    status = service.get_status()

    assert snapshot is not None
    assert snapshot.has_live_data is True
    assert snapshot.stream_status == "Receiving waveform"
    assert status.connection_label == "Receiving waveform"


def test_begin_capture_reports_receiving(test_settings, monkeypatch) -> None:
    monkeypatch.setattr(ble_sensor, "BleakClient", object())
    monkeypatch.setattr(ble_sensor, "BleakScanner", object())
    settings = test_settings.model_copy(update={"ble_enabled": True})
    service = BLEIngestionService(settings)

    with service._lock:
        service._running = True
        service._future = SimpleNamespace(done=lambda: False)
        service._connection_state = "connected"

    assert service.begin_capture("patient_001", "John") is True

    snapshot = service.get_live_snapshot(size=3)
    status = service.get_status()

    assert snapshot is not None
    assert snapshot.stream_status == "Receiving waveform"
    assert status.connection_label == "Receiving waveform"


def test_capture_buffer_keeps_configured_recording_window(test_settings, monkeypatch) -> None:
    monkeypatch.setattr(ble_sensor, "BleakClient", object())
    monkeypatch.setattr(ble_sensor, "BleakScanner", object())
    settings = test_settings.model_copy(
        update={
            "ble_enabled": True,
            "ble_sample_rate": 10,
            "ble_analysis_time_seconds": 2,
            "ble_capture_grace_seconds": 1.0,
            "ble_capture_buffer_size": 5,
        }
    )
    service = BLEIngestionService(settings)

    with service._lock:
        service._running = True
        service._future = SimpleNamespace(done=lambda: False)
        service._connection_state = "connected"

    assert service.begin_capture("patient_001", "John") is True
    samples = list(range(25))
    service._consume_samples(samples, patient_id="patient_001")

    assert service.get_capture_sample_count("patient_001") == len(samples)
    progress = service.get_capture_progress_snapshot("patient_001", size=10)
    assert progress.sample_count == len(samples)
    assert len(progress.waveform) == 10
    assert all(0 <= sample <= 1 for sample in progress.waveform)
    assert service.end_capture("patient_001") == samples


def test_connected_worker_does_not_analyze_until_capture_request(
    test_settings,
    monkeypatch,
) -> None:
    async def run_scenario() -> None:
        monkeypatch.setattr(ble_sensor, "BleakClient", object())
        monkeypatch.setattr(ble_sensor, "BleakScanner", object())
        monkeypatch.setattr(pcg_ble_client, "PCGClient", FakePCGClient)
        FakePCGClient.instances = []
        settings = test_settings.model_copy(update={"ble_enabled": True})
        service = BLEIngestionService(settings)
        service.bind_event_loop(asyncio.get_running_loop())

        assert service.start() is True
        await _wait_until(lambda: bool(FakePCGClient.instances))
        await asyncio.sleep(0.05)

        client = FakePCGClient.instances[0]
        assert client.analyze_calls == []
        assert service.get_status().connection_label == "Connected to sensor"

        await _stop_service(service)

    asyncio.run(run_scenario())


def test_capture_request_uses_pcg_analyze_api(test_settings, monkeypatch) -> None:
    async def run_scenario() -> None:
        monkeypatch.setattr(ble_sensor, "BleakClient", object())
        monkeypatch.setattr(ble_sensor, "BleakScanner", object())
        monkeypatch.setattr(pcg_ble_client, "PCGClient", FakePCGClient)
        FakePCGClient.instances = []
        settings = test_settings.model_copy(
            update={
                "ble_enabled": True,
                "ble_batch_size": 6,
                "ble_analysis_time_seconds": 60,
                "ble_retry_delay_seconds": 0.01,
            }
        )
        service = BLEIngestionService(settings)
        service.bind_event_loop(asyncio.get_running_loop())

        assert service.start() is True
        await _wait_until(lambda: bool(FakePCGClient.instances))
        assert service.begin_capture("patient_001", "John") is True
        await _wait_until(
            lambda: FakePCGClient.instances[0].analyze_calls
            and service.get_capture_sample_count("patient_001") == 3
        )

        call = FakePCGClient.instances[0].analyze_calls[0]
        assert call == {
            "sample_rate": 500,
            "oversample_count": 8,
            "batch_size": 6,
            "patient_name": "John",
            "analysis_time_seconds": 60,
        }
        assert service.get_status().connection_label == "Receiving waveform"
        assert service.end_capture("patient_001") == [100, 120, 140]
        await _wait_until(
            lambda: service.get_status().connection_label == "Connected to sensor"
        )
        assert service.get_status().connection_label == "Connected to sensor"

        await _stop_service(service)

    asyncio.run(run_scenario())


def test_two_consecutive_captures_use_fresh_ble_connections(
    test_settings,
    monkeypatch,
) -> None:
    async def run_scenario() -> None:
        monkeypatch.setattr(ble_sensor, "BleakClient", object())
        monkeypatch.setattr(ble_sensor, "BleakScanner", object())
        monkeypatch.setattr(pcg_ble_client, "PCGClient", FakePCGClient)
        FakePCGClient.instances = []
        settings = test_settings.model_copy(
            update={
                "ble_enabled": True,
                "ble_retry_delay_seconds": 0.01,
                "ble_poll_interval_seconds": 0.01,
            }
        )
        service = BLEIngestionService(settings)
        service.bind_event_loop(asyncio.get_running_loop())

        assert service.start() is True
        await _wait_until(lambda: bool(FakePCGClient.instances))

        assert service.begin_capture("patient_001", "First") is True
        await _wait_until(
            lambda: FakePCGClient.instances[0].analyze_calls
            and service.get_capture_sample_count("patient_001") == 3
        )
        assert service.end_capture("patient_001") == [100, 120, 140]

        await _wait_until(lambda: len(FakePCGClient.instances) >= 2)
        assert service.begin_capture("patient_001", "Second") is True
        await _wait_until(
            lambda: FakePCGClient.instances[1].analyze_calls
            and service.get_capture_sample_count("patient_001") == 3
        )
        assert service.end_capture("patient_001") == [100, 120, 140]

        assert len(FakePCGClient.instances[0].analyze_calls) == 1
        assert len(FakePCGClient.instances[1].analyze_calls) == 1
        await _stop_service(service)

    asyncio.run(run_scenario())


def test_active_capture_is_retried_after_ble_reconnect(test_settings, monkeypatch) -> None:
    async def run_scenario() -> None:
        monkeypatch.setattr(ble_sensor, "BleakClient", object())
        monkeypatch.setattr(ble_sensor, "BleakScanner", object())
        monkeypatch.setattr(pcg_ble_client, "PCGClient", FlakyCapturePCGClient)
        FlakyCapturePCGClient.instances = []
        settings = test_settings.model_copy(
            update={
                "ble_enabled": True,
                "ble_retry_delay_seconds": 0.01,
                "ble_poll_interval_seconds": 0.01,
            }
        )
        service = BLEIngestionService(settings)
        service.bind_event_loop(asyncio.get_running_loop())

        assert service.start() is True
        await _wait_until(lambda: bool(FlakyCapturePCGClient.instances))
        assert service.begin_capture("patient_001", "John") is True
        await _wait_until(
            lambda: len(FlakyCapturePCGClient.instances) >= 2
            and FlakyCapturePCGClient.instances[1].analyze_calls
            and service.get_capture_sample_count("patient_001") == 3
        )

        assert FlakyCapturePCGClient.instances[0].analyze_calls
        assert FlakyCapturePCGClient.instances[1].analyze_calls
        assert service.end_capture("patient_001") == [200, 220, 240]

        await _stop_service(service)

    asyncio.run(run_scenario())
