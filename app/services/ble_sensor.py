import asyncio
import concurrent.futures
import logging
import struct
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
        self._capture_samples: deque[int] = deque(maxlen=settings.ble_capture_buffer_size)
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

        loop = self._resolve_event_loop()
        if loop is None:
            message = "BLE worker needs a running asyncio event loop"
            self._set_connection_state(f"error:{message}")
            with self._lock:
                self._last_error = message
            logger.error(message)
            return False

        with self._lock:
            if self._running and self._future and not self._future.done():
                return True
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
        if future and not future.done():
            future.cancel()
        if self.is_enabled:
            self._set_connection_state("disconnected")
        logger.info("BLE ingestion worker stopped")

    def begin_capture(self, patient_id: str) -> bool:
        if not self.is_enabled:
            return True
        self.start()
        with self._lock:
            if self._active_capture_patient_id and self._active_capture_patient_id != patient_id:
                return False
            self._active_capture_patient_id = patient_id
            self._capture_samples.clear()
            return True

    def ensure_capture_binding(self, patient_id: str) -> bool:
        if not self.is_enabled:
            return True
        self.start()
        with self._lock:
            if self._active_capture_patient_id is None:
                self._active_capture_patient_id = patient_id
                self._capture_samples.clear()
                return True
            return self._active_capture_patient_id == patient_id

    def get_capture_sample_count(self, patient_id: str) -> int:
        if not self.is_enabled:
            return 0
        with self._lock:
            if self._active_capture_patient_id != patient_id:
                return 0
            return len(self._capture_samples)

    def end_capture(self, patient_id: str) -> list[int]:
        if not self.is_enabled:
            return []
        with self._lock:
            if self._active_capture_patient_id != patient_id:
                return []
            samples = list(self._capture_samples)
            self._active_capture_patient_id = None
            self._capture_samples.clear()
            return samples

    def get_live_snapshot(self, *, size: int, idle_seed: int = 0) -> LiveWaveformSnapshot | None:
        if not self.is_enabled:
            return None

        with self._lock:
            recent_samples = list(self._recent_samples)[-size:]
            last_batch_monotonic = self._last_batch_monotonic
            connection_state = self._connection_state

        has_recent_signal = bool(recent_samples) and (
            last_batch_monotonic is not None
            and (time.monotonic() - last_batch_monotonic) <= self.settings.ble_stale_after_seconds
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
            stream_status=self._status_label(connection_state),
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
        finally:
            self._lock.release()

        return BLEStatusSnapshot(
            enabled=self.is_enabled,
            connection_state=connection_state,
            connection_label=self._status_label(connection_state),
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
        while self._is_current_run(run_id):
            try:
                address = self.settings.ble_device_address
                device_name = self.settings.ble_device_name
                if not address:
                    self._set_connection_state("scanning")
                    logger.info("[BLE] Scanning for device...")
                    target = None
                    while target is None and self._is_current_run(run_id):
                        target = await BleakScanner.find_device_by_filter(
                            self._matches_advertised_device,
                            timeout=self.settings.ble_scan_timeout_seconds,
                        )
                        if target is None:
                            message = f"'{device_name}' not found, retrying..."
                            with self._lock:
                                self._last_error = message
                            logger.info("[BLE] %s", message)
                            await asyncio.sleep(self.settings.ble_retry_delay_seconds)

                    if not self._is_current_run(run_id) or target is None:
                        return

                    address = target.address
                    display_name = target.name
                    logger.info("[BLE] Found device: %s [%s]", target.name, target.address)
                else:
                    display_name = address

                self._set_connection_state("connecting")
                logger.info("[BLE] Connecting to %s (%s)", display_name, address)

                async with BleakClient(address) as client:
                    self._set_connection_state("connected")
                    self._last_batch_monotonic = None
                    logger.info("[BLE] Connected: %s", client.is_connected)

                    def on_notify(_handle: int, data: bytearray) -> None:
                        self._consume_notification(data)

                    await client.start_notify(self.settings.ble_characteristic_uuid, on_notify)
                    logger.info("[BLE] Receiving notifications...")

                    while self._is_current_run(run_id) and client.is_connected:
                        await asyncio.sleep(self.settings.ble_poll_interval_seconds)

                    with suppress(Exception):
                        await client.stop_notify(self.settings.ble_characteristic_uuid)

                self._set_connection_state("disconnected")
                logger.info("[BLE] Disconnected")
            except Exception as exc:  # pragma: no cover - depends on hardware/runtime
                self._set_connection_state(f"error:{exc}")
                with self._lock:
                    self._last_error = str(exc)
                logger.exception("BLE worker error", exc_info=exc)
                await asyncio.sleep(self.settings.ble_retry_delay_seconds)

    def _resolve_event_loop(self) -> asyncio.AbstractEventLoop | None:
        if self._loop and self._loop.is_running():
            return self._loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        self._loop = loop
        return loop

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

    def _consume_notification(self, data: bytearray) -> None:
        values = self._decode_notification(data)
        if not values:
            return
        self._consume_samples(values)

    def _decode_notification(self, data: bytearray):
        n_samples = len(data) // 2
        if n_samples <= 0:
            return []
        return struct.unpack(f"<{n_samples}H", data[: n_samples * 2])

    def _consume_samples(self, values) -> None:
        n_samples = len(values)
        if n_samples <= 0:
            return
        with self._lock:
            self._recent_samples.extend(values)
            if self._active_capture_patient_id:
                self._capture_samples.extend(values)
            self._last_batch_monotonic = time.monotonic()
            self._last_batch_at = datetime.now(UTC)
            self._connection_state = "receiving"
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
