import asyncio
import concurrent.futures
import logging
import math
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings, get_settings
from app.core.constants import DEFAULT_SIGNAL_QUALITY, IDLE_SIGNAL_QUALITY, LIVE_STREAM_STATUS
from app.services.waveform import generate_idle_waveform, normalize_waveform

try:
    from bleak import BleakClient, BleakScanner
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    BleakClient = None
    BleakScanner = None


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveWaveformSnapshot:
    waveform: list[float]
    stream_status: str
    signal_quality: str
    has_live_data: bool


@dataclass(slots=True)
class CaptureProgressSnapshot:
    waveform: list[float]
    sample_count: int


@dataclass(slots=True)
class BLEStatusSnapshot:
    enabled: bool
    connection_state: str
    connection_label: str
    is_running: bool
    device_name: str | None
    service_uuid: str | None
    characteristic_uuid: str | None
    active_capture_patient_id: str | None
    recent_sample_count: int
    captured_sample_count: int
    notification_count: int
    last_notification_at: datetime | None
    last_error: str | None


@dataclass(slots=True)
class BLEDeviceSnapshot:
    name: str | None
    address: str
    local_name: str | None
    service_uuids: list[str]
    rssi: int | None
    manufacturer_data_keys: list[int]
    is_target_match: bool
    matched_by: list[str]


class BLEIngestionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._recent_samples: deque[int] = deque(
            maxlen=max(settings.live_waveform_size, settings.ble_recent_buffer_size)
        )
        self._capture_samples: deque[int] = deque(maxlen=self._capture_buffer_size())
        self._active_capture_patient_id: str | None = None
        self._last_batch_monotonic: float | None = None
        self._last_batch_at: datetime | None = None
        self._connection_state = "disabled"
        self._last_error: str | None = None
        self._notification_count = 0
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._future: concurrent.futures.Future[None] | None = None
        self._run_id = 0
        self._capture_requested = threading.Event()
        self._active_capture_patient_name: str | None = None

        if not settings.ble_enabled:
            self._connection_state = "disabled"
        elif (
            not settings.ble_device_address
            and not settings.ble_device_name
            and not settings.ble_service_uuid
        ) or not settings.ble_characteristic_uuid:
            self._connection_state = "not_configured"
        elif BleakClient is None or BleakScanner is None:
            self._connection_state = "unavailable"
        else:
            self._connection_state = "disconnected"

    @property
    def is_enabled(self) -> bool:
        return (
            self.settings.ble_enabled
            and bool(
                self.settings.ble_device_address
                or self.settings.ble_device_name
                or self.settings.ble_service_uuid
            )
            and bool(self.settings.ble_characteristic_uuid)
            and BleakClient is not None
            and BleakScanner is not None
        )

    def bind_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def start(self) -> bool:
        if not self.is_enabled:
            return False

        with self._lock:
            if self._running and self._future and not self._future.done():
                return True

        loop = self._resolve_event_loop()
        if loop is None:
            message = "BLE worker needs a running asyncio event loop"
            self._set_connection_state(f"error:{message}")
            with self._lock:
                self._last_error = message
            logger.error(message)
            return False

        with self._lock:
            self._running = True
            self._run_id += 1
            run_id = self._run_id

        future = asyncio.run_coroutine_threadsafe(self._ble_loop(run_id), loop)
        future.add_done_callback(lambda completed: self._handle_worker_done(completed, run_id))
        with self._lock:
            self._future = future
        logger.info("BLE ingestion worker started")
        return True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._run_id += 1
            future = self._future
            self._future = None
            self._active_capture_patient_id = None
            self._active_capture_patient_name = None
            self._capture_samples.clear()
            self._capture_requested.clear()
        if future and not future.done():
            future.cancel()
        if self.is_enabled:
            self._set_connection_state("disconnected")
        logger.info("BLE ingestion worker stopped")

    def begin_capture(self, patient_id: str, patient_name: str | None = None) -> bool:
        if not self.is_enabled:
            return True
        if not self.start():
            return False
        with self._lock:
            if self._active_capture_patient_id and self._active_capture_patient_id != patient_id:
                return False
            self._active_capture_patient_id = patient_id
            self._active_capture_patient_name = patient_name or patient_id
            self._capture_samples.clear()
            self._recent_samples.clear()
            self._last_batch_monotonic = None
            if self._connection_state in {"connected", "receiving"}:
                self._connection_state = "receiving"
            self._capture_requested.set()
        return True

    def ensure_capture_binding(self, patient_id: str) -> bool:
        if not self.is_enabled:
            return True
        with self._lock:
            return self._active_capture_patient_id == patient_id

    def is_capture_active(self, patient_id: str) -> bool:
        if not self.is_enabled:
            return False
        with self._lock:
            return self._active_capture_patient_id == patient_id

    def get_capture_sample_count(self, patient_id: str) -> int:
        if not self.is_enabled:
            return 0
        with self._lock:
            if self._active_capture_patient_id != patient_id:
                return 0
            return len(self._capture_samples)

    def get_capture_progress_snapshot(
        self,
        patient_id: str,
        *,
        size: int,
    ) -> CaptureProgressSnapshot:
        if not self.is_enabled:
            return CaptureProgressSnapshot(waveform=[], sample_count=0)
        with self._lock:
            if self._active_capture_patient_id != patient_id:
                return CaptureProgressSnapshot(waveform=[], sample_count=0)
            samples = list(self._capture_samples)

        return CaptureProgressSnapshot(
            waveform=self._downsample_capture_waveform(samples, size=max(1, size)),
            sample_count=len(samples),
        )

    def end_capture(self, patient_id: str) -> list[int]:
        if not self.is_enabled:
            return []
        with self._lock:
            if self._active_capture_patient_id != patient_id:
                return []
            samples = list(self._capture_samples)
            self._active_capture_patient_id = None
            self._active_capture_patient_name = None
            self._capture_samples.clear()
            self._recent_samples.clear()
            self._last_batch_monotonic = None
            self._capture_requested.clear()
            if self._connection_state == "receiving":
                self._connection_state = "connected"
            return samples

    def get_live_snapshot(self, *, size: int, idle_seed: int = 0) -> LiveWaveformSnapshot | None:
        if not self.is_enabled:
            return None

        with self._lock:
            recent_samples = list(self._recent_samples)[-size:]
            last_batch_monotonic = self._last_batch_monotonic
            connection_state = self._connection_state
            capture_is_active = self._active_capture_patient_id is not None

        has_recent_signal = (
            capture_is_active
            and bool(recent_samples)
            and self._has_recent_signal(last_batch_monotonic)
        )
        if has_recent_signal:
            return LiveWaveformSnapshot(
                waveform=normalize_waveform(recent_samples),
                stream_status=LIVE_STREAM_STATUS,
                signal_quality=DEFAULT_SIGNAL_QUALITY,
                has_live_data=True,
            )

        return LiveWaveformSnapshot(
            waveform=generate_idle_waveform(size=size, seed=idle_seed),
            stream_status=(
                LIVE_STREAM_STATUS if capture_is_active else self._idle_status_label(connection_state)
            ),
            signal_quality=IDLE_SIGNAL_QUALITY,
            has_live_data=False,
        )

    def get_status(self) -> BLEStatusSnapshot:
        lock_acquired = self._lock.acquire(timeout=0.05)
        if not lock_acquired:
            return BLEStatusSnapshot(
                enabled=self.is_enabled,
                connection_state="status_busy",
                connection_label="BLE status busy",
                is_running=self._running,
                device_name=self.settings.ble_device_name,
                service_uuid=self.settings.ble_service_uuid,
                characteristic_uuid=self.settings.ble_characteristic_uuid,
                active_capture_patient_id=None,
                recent_sample_count=0,
                captured_sample_count=0,
                notification_count=0,
                last_notification_at=None,
                last_error="Timed out while reading BLE service status",
            )
        try:
            connection_state = self._connection_state
            active_capture_patient_id = self._active_capture_patient_id
            recent_sample_count = len(self._recent_samples)
            captured_sample_count = len(self._capture_samples)
            notification_count = self._notification_count
            last_notification_at = self._last_batch_at
            last_error = self._last_error
            last_batch_monotonic = self._last_batch_monotonic
        finally:
            self._lock.release()

        capture_is_active = active_capture_patient_id is not None
        return BLEStatusSnapshot(
            enabled=self.is_enabled,
            connection_state=connection_state,
            connection_label=self._status_label_for_signal(
                connection_state,
                self._has_recent_signal(last_batch_monotonic),
                capture_is_active,
            ),
            is_running=self._running,
            device_name=self.settings.ble_device_name,
            service_uuid=self.settings.ble_service_uuid,
            characteristic_uuid=self.settings.ble_characteristic_uuid,
            active_capture_patient_id=active_capture_patient_id,
            recent_sample_count=recent_sample_count,
            captured_sample_count=captured_sample_count,
            notification_count=notification_count,
            last_notification_at=last_notification_at,
            last_error=last_error,
        )

    async def scan_devices(self, timeout_seconds: float | None = None) -> list[BLEDeviceSnapshot]:
        if BleakScanner is None:
            return []

        timeout = timeout_seconds or self.settings.ble_scan_timeout_seconds
        try:
            discovered = await asyncio.wait_for(
                BleakScanner.discover(timeout=timeout, return_adv=True),
                timeout=timeout + 2.0,
            )
        except TypeError:
            devices = await asyncio.wait_for(
                BleakScanner.discover(timeout=timeout),
                timeout=timeout + 2.0,
            )
            return [self._build_device_snapshot(device, None) for device in devices]

        if isinstance(discovered, dict):
            snapshots = [
                self._build_device_snapshot(device, advertisement_data)
                for device, advertisement_data in discovered.values()
            ]
        else:
            snapshots = [self._build_device_snapshot(device, None) for device in discovered]

        return sorted(
            snapshots,
            key=lambda device: (
                not device.is_target_match,
                device.name or device.local_name or "",
                device.address,
            ),
        )

    async def _ble_loop(self, run_id: int) -> None:
        from app.services.pcg_ble_client import (
            BLEConnectionError,
            PCGClient,
            PCGClientConfig,
        )

        while self._is_current_run(run_id):
            client = PCGClient(
                device_name=self.settings.ble_device_name or PCGClientConfig.patient_name,
                device_address=self.settings.ble_device_address,
                service_uuid=self.settings.ble_service_uuid,
                characteristic_uuid=self.settings.ble_characteristic_uuid,
                scan_timeout_seconds=self.settings.ble_scan_timeout_seconds,
                connect_timeout_seconds=self.settings.ble_connect_timeout_seconds,
            )
            try:
                self._set_connection_state("scanning")
                await client.connect()
                self._mark_connected()
                logger.info("[BLE] Connected: %s", self._client_is_connected(client))

                while self._is_current_run(run_id) and self._client_is_connected(client):
                    request = await self._wait_for_capture_request(run_id)
                    if request is None:
                        break
                    patient_id, patient_name = request
                    await self._run_capture(client, patient_id, patient_name)
                    # Arduino BLE stacks can retain stale notification/GATT state
                    # after a completed capture. Reconnect before accepting the
                    # next request so every recording starts from a clean link.
                    break

                self._set_connection_state("disconnected")
                logger.info("[BLE] Disconnected")
            except BLEConnectionError as exc:
                self._set_connection_state(f"error:{exc}")
                with self._lock:
                    self._last_error = str(exc)
                self._requeue_active_capture_request()
                logger.warning("[BLE] %s", exc)
            except Exception as exc:  # pragma: no cover - depends on hardware/runtime
                self._set_connection_state(f"error:{exc}")
                with self._lock:
                    self._last_error = str(exc)
                self._requeue_active_capture_request()
                logger.exception("BLE worker error", exc_info=exc)
            finally:
                with suppress(Exception):
                    await client.disconnect()

            if self._is_current_run(run_id):
                await asyncio.sleep(self.settings.ble_retry_delay_seconds)

    async def _wait_for_capture_request(
        self,
        run_id: int,
    ) -> tuple[str, str] | None:
        poll_interval = min(max(self.settings.ble_poll_interval_seconds, 0.01), 0.25)
        while self._is_current_run(run_id):
            if self._capture_requested.is_set():
                with self._lock:
                    patient_id = self._active_capture_patient_id
                    patient_name = self._active_capture_patient_name or patient_id
                    self._capture_requested.clear()
                if patient_id and patient_name:
                    return patient_id, patient_name
            await asyncio.sleep(poll_interval)
        return None

    async def _run_capture(self, client, patient_id: str, patient_name: str) -> None:
        self._set_connection_state("receiving")
        logger.info(
            "[BLE] Starting %ss analysis capture for %s",
            self.settings.ble_analysis_time_seconds,
            patient_name,
        )
        async for batch in client.analyze(
            sample_rate=self.settings.ble_sample_rate,
            oversample_count=self.settings.ble_oversample_count,
            batch_size=self.settings.ble_batch_size,
            patient_name=patient_name,
            analysis_time_seconds=self.settings.ble_analysis_time_seconds,
        ):
            if not self._is_capture_active(patient_id):
                break
            self._consume_samples([int(sample) for sample in batch], patient_id=patient_id)
        self._mark_connected()

    def _resolve_event_loop(self) -> asyncio.AbstractEventLoop | None:
        if self._loop and self._loop.is_running():
            return self._loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        self._loop = loop
        return loop

    @staticmethod
    def _client_is_connected(client) -> bool:
        is_connected = getattr(client, "is_connected", False)
        if callable(is_connected):
            return bool(is_connected())
        return bool(is_connected)

    def _is_current_run(self, run_id: int) -> bool:
        with self._lock:
            return self._running and self._run_id == run_id

    def _handle_worker_done(
        self,
        completed: concurrent.futures.Future[None],
        run_id: int,
    ) -> None:
        if completed.cancelled():
            return
        try:
            completed.result()
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            return
        except Exception as exc:  # pragma: no cover - defensive around hardware callbacks
            self._set_connection_state(f"error:{exc}")
            with self._lock:
                self._last_error = str(exc)
            logger.exception("BLE worker exited unexpectedly", exc_info=exc)
            return
        with self._lock:
            if self._run_id == run_id:
                self._running = False
                self._future = None

    def _is_capture_active(self, patient_id: str) -> bool:
        with self._lock:
            return self._active_capture_patient_id == patient_id

    def _requeue_active_capture_request(self) -> None:
        with self._lock:
            if self._active_capture_patient_id:
                self._capture_requested.set()

    def _consume_samples(self, values, *, patient_id: str | None = None) -> None:
        n_samples = len(values)
        if n_samples <= 0:
            return
        with self._lock:
            if patient_id is not None and self._active_capture_patient_id != patient_id:
                return
            self._recent_samples.extend(values)
            if self._active_capture_patient_id:
                self._capture_samples.extend(values)
                self._connection_state = "receiving"
            elif self._connection_state == "receiving":
                self._connection_state = "connected"
            self._last_batch_monotonic = time.monotonic()
            self._last_batch_at = datetime.now(UTC)
            self._last_error = None
            self._notification_count += 1
            notification_count = self._notification_count
        if notification_count <= 5 or notification_count % 100 == 0:
            logger.info(
                "[BLE] Notification #%s: %s samples, first=%s",
                notification_count,
                n_samples,
                values[0],
            )

    def _capture_buffer_size(self) -> int:
        expected_capture_samples = math.ceil(
            self.settings.ble_sample_rate
            * (self.settings.ble_analysis_time_seconds + self.settings.ble_capture_grace_seconds)
        )
        return max(self.settings.ble_capture_buffer_size, expected_capture_samples, 1)

    @staticmethod
    def _downsample_capture_waveform(samples: list[int], *, size: int) -> list[float]:
        if not samples:
            return []
        normalized = normalize_waveform(samples)
        if len(normalized) <= size:
            return normalized

        bucket_size = len(normalized) / size
        result: list[float] = []
        for bucket in range(size):
            start = int(bucket * bucket_size)
            end = max(start + 1, int((bucket + 1) * bucket_size))
            values = normalized[start:min(end, len(normalized))]
            peak = max(values, key=lambda value: abs(value - 0.5))
            result.append(round(peak, 4))
        return result

    def _matches_advertised_device(self, device, advertisement_data) -> bool:
        return bool(self._device_match_reasons(device, advertisement_data))

    def _build_device_snapshot(self, device, advertisement_data) -> BLEDeviceSnapshot:
        service_uuids = self._advertised_service_uuids(advertisement_data)
        manufacturer_data = getattr(advertisement_data, "manufacturer_data", {}) or {}
        matched_by = self._device_match_reasons(device, advertisement_data)
        return BLEDeviceSnapshot(
            name=getattr(device, "name", None),
            address=str(getattr(device, "address", "")),
            local_name=getattr(advertisement_data, "local_name", None),
            service_uuids=service_uuids,
            rssi=getattr(advertisement_data, "rssi", getattr(device, "rssi", None)),
            manufacturer_data_keys=[int(key) for key in manufacturer_data.keys()],
            is_target_match=bool(matched_by),
            matched_by=matched_by,
        )

    def _device_match_reasons(self, device, advertisement_data) -> list[str]:
        device_name = self.settings.ble_device_name
        service_uuid = self.settings.ble_service_uuid
        reasons: list[str] = []
        if device_name and getattr(device, "name", None) == device_name:
            reasons.append("device.name")
        if (
            device_name
            and advertisement_data is not None
            and getattr(advertisement_data, "local_name", None) == device_name
        ):
            reasons.append("advertisement.local_name")
        advertised_service_uuids = self._advertised_service_uuids(advertisement_data)
        if service_uuid and service_uuid.lower() in [uuid.lower() for uuid in advertised_service_uuids]:
            reasons.append("advertisement.service_uuid")
        return reasons

    @staticmethod
    def _advertised_service_uuids(advertisement_data) -> list[str]:
        if advertisement_data is None:
            return []
        return [str(uuid) for uuid in (advertisement_data.service_uuids or [])]

    def _set_connection_state(self, value: str) -> None:
        with self._lock:
            self._connection_state = value

    def _mark_connected(self) -> None:
        with self._lock:
            self._connection_state = "connected"
            self._last_batch_monotonic = None

    def _has_recent_signal(self, last_batch_monotonic: float | None) -> bool:
        return (
            last_batch_monotonic is not None
            and (time.monotonic() - last_batch_monotonic) <= self.settings.ble_stale_after_seconds
        )

    def _idle_status_label(self, connection_state: str) -> str:
        return self._status_label_for_signal(connection_state, has_recent_signal=False)

    @classmethod
    def _status_label_for_signal(
        cls,
        connection_state: str,
        has_recent_signal: bool,
        capture_is_active: bool = False,
    ) -> str:
        if capture_is_active:
            return LIVE_STREAM_STATUS
        if connection_state == "receiving" and not has_recent_signal:
            return cls._status_label("connected")
        return cls._status_label(connection_state)

    @staticmethod
    def _status_label(connection_state: str) -> str:
        if connection_state == "scanning":
            return "Scanning BLE sensor"
        if connection_state == "connecting":
            return "Connecting to sensor"
        if connection_state == "connected":
            return "Connected to sensor"
        if connection_state == "receiving":
            return LIVE_STREAM_STATUS
        if connection_state == "disconnected":
            return "Sensor disconnected"
        if connection_state == "not_configured":
            return "BLE not configured"
        if connection_state == "unavailable":
            return "BLE library unavailable"
        if connection_state == "disabled":
            return "BLE disabled"
        if connection_state == "status_busy":
            return "BLE status busy"
        if connection_state.startswith("error:"):
            return "BLE error"
        return "Waiting for sensor"


_ble_service: BLEIngestionService | None = None
_ble_service_lock = threading.Lock()


def get_ble_ingestion_service(settings: Settings | None = None) -> BLEIngestionService:
    global _ble_service
    current_settings = settings or get_settings()
    with _ble_service_lock:
        if _ble_service is None or _ble_service.settings is not current_settings:
            _ble_service = BLEIngestionService(current_settings)
        return _ble_service
