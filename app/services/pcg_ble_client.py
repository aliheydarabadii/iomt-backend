import asyncio
import struct
import time
import numpy as np
from bleak import BleakClient, BleakScanner
from typing import Generator

class BLEConnectionError(Exception):
    """Raised when BLE connection fails or drops."""
    pass

class PCGClientConfig:
    sample_rate: int = 500
    oversample_count: int = 8
    batch_size: int = 6
    patient_name: str = "Test_Patient"
    analysis_time_seconds: int = 60
    device_address: str | None = None
    service_uuid: str = "12345678-1234-1234-1234-123456789abc"
    characteristic_uuid: str = "abcd1234-ab12-cd34-ef56-123456789abc"
    scan_timeout_seconds: float = 10.0
    connect_timeout_seconds: float = 10.0

class PCGClient:
    """
    Client for controlling Arduino PCG data collection via BLE.
    Sends analysis requests and receives phonocardiogram signal batches.
    """

    SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
    CHARACTERISTIC_UUID = "abcd1234-ab12-cd34-ef56-123456789abc"
    NOTIFICATION_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        device_name="PCG_Monitor_Raw",
        device_address: str | None = None,
        service_uuid: str | None = None,
        characteristic_uuid: str | None = None,
        scan_timeout_seconds: float = 10.0,
        connect_timeout_seconds: float = 10.0,
    ):
        self.device_name = device_name
        self.device_address = device_address
        self.service_uuid = service_uuid or self.SERVICE_UUID
        self.characteristic_uuid = characteristic_uuid or self.CHARACTERISTIC_UUID
        self.scan_timeout_seconds = scan_timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.client = None
        self._sample_rate = 0
        self._analysis_time_seconds = 0
        self._accumulated_data = []
        self._batch_queue = asyncio.Queue()

    async def connect(self):
        """Establish BLE connection to Arduino."""
        print(f"Searching for device: {self.device_name or self.device_address or self.service_uuid}")

        target_device = await self._find_target_device()
        if target_device is None:
            raise BLEConnectionError("Configured BLE device not found while advertising")

        print(f"Found device: {getattr(target_device, 'address', target_device)}")

        try:
            try:
                self.client = BleakClient(target_device, timeout=self.connect_timeout_seconds)
            except TypeError:
                self.client = BleakClient(target_device)
            await self.client.connect()
            print("Connected to device")
        except Exception as e:
            await self._disconnect_safely()
            raise BLEConnectionError(f"Failed to connect: {e}")

        if not self.is_connected():
            await self._disconnect_safely()
            raise BLEConnectionError("Failed to connect: client did not report an active BLE link")

        # Give connection time to stabilize. Service/MTU inspection is useful
        # for diagnostics, but some Bleak backends expose it lazily or not at
        # all. Do not turn a live BLE link into a reconnect loop just because
        # optional metadata could not be read.
        await asyncio.sleep(1.0)
        try:
            services = self.client.services
            mtu_size = getattr(self.client, "mtu_size", "unknown")
            print(f"Services discovered: {bool(services)}, MTU: {mtu_size}")
        except Exception as e:
            print(f"Connected, but service metadata was unavailable: {e}")

    async def disconnect(self):
        """Close BLE connection."""
        await self._disconnect_safely()

    async def _disconnect_safely(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("Disconnected from device")

    def is_connected(self) -> bool:
        """Return True if BLE connection is active."""
        if self.client is None:
            return False
        try:
            return bool(self.client.is_connected)
        except:
            return False

    async def analyze(self, sample_rate: int, oversample_count: int, batch_size: int,
                     patient_name: str, analysis_time_seconds: int):
        """
        Send analysis request to Arduino and yield batches as they arrive.

        Args:
            sample_rate: Samples per second (e.g., 500)
            oversample_count: ADC reads per sample (e.g., 8)
            batch_size: Samples per BLE packet (e.g., 6)
            patient_name: Patient identifier
            analysis_time_seconds: Duration to collect (e.g., 60)

        Yields:
            np.ndarray of samples (length batch_size), dtype uint16

        Raises:
            BLEConnectionError: If connection drops
        """
        if not self.is_connected():
            raise BLEConnectionError("Not connected to device")

        # Store for later validation
        self._sample_rate = sample_rate
        self._analysis_time_seconds = analysis_time_seconds

        # Reset accumulation
        self._accumulated_data = []
        self._batch_queue = asyncio.Queue()
        self._first_notif_seen = False

        # Encode and send START packet
        packet = self._encode_start_packet(sample_rate, oversample_count, batch_size,
                                           analysis_time_seconds, patient_name)

        # Subscribe to notifications BEFORE starting the analysis. The Arduino
        # begins sampling the instant it receives START and calls notify()
        # immediately; any notifications sent before we subscribe are silently
        # dropped by the BLE stack, so we'd lose the first batches.
        try:
            print(f"Starting notifications on {self.characteristic_uuid}...")
            await self.client.start_notify(
                self.characteristic_uuid,
                self._notification_handler
            )
            print("Notifications started successfully")
        except Exception as e:
            if not self.is_connected():
                raise BLEConnectionError("Connection lost before analysis. Check Arduino serial output.")
            # One retry after a short settle delay
            try:
                await asyncio.sleep(1.0)
                print(f"Retrying notifications after error: {e}")
                await self.client.start_notify(
                    self.characteristic_uuid,
                    self._notification_handler
                )
                print("Notifications started on retry")
            except Exception as e2:
                raise BLEConnectionError(f"Failed to start notifications: {e2}")

        # Now send START. We are already listening, so no batches are missed.
        try:
            await self.client.write_gatt_char(
                self.characteristic_uuid,
                packet,
                response=False
            )
            print(f"Sent START command for {analysis_time_seconds}s analysis")
        except Exception as e:
            raise BLEConnectionError(f"Failed to send command: {e}")

        # Yield batches until analysis time expires
        expected_total_samples = sample_rate * analysis_time_seconds
        expected_num_batches = (expected_total_samples + batch_size - 1) // batch_size
        batches_yielded = 0

        try:
            while batches_yielded < expected_num_batches:
                try:
                    # START should produce notifications almost immediately.
                    # Waiting for the full recording duration when none arrive
                    # prevents the ingestion worker from reconnecting and retrying.
                    timeout_seconds = self.NOTIFICATION_TIMEOUT_SECONDS
                    batch = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=timeout_seconds
                    )
                    yield batch
                    batches_yielded += 1
                except asyncio.TimeoutError:
                    if batches_yielded == 0:
                        raise BLEConnectionError(
                            "No BLE notifications received after START command"
                        )
                    print(
                        "Warning: BLE notification stream stalled after "
                        f"{batches_yielded} batches"
                    )
                    break
        finally:
            # Stop notifications
            try:
                await self.client.stop_notify(self.characteristic_uuid)
            except:
                pass

    async def _find_target_device(self):
        if self.device_address:
            find_by_address = getattr(BleakScanner, "find_device_by_address", None)
            if callable(find_by_address):
                try:
                    device = await find_by_address(
                        self.device_address,
                        timeout=self.scan_timeout_seconds,
                    )
                    if device is not None:
                        return device
                except Exception:
                    pass

        try:
            discovered = await BleakScanner.discover(
                timeout=self.scan_timeout_seconds,
                return_adv=True,
            )
        except TypeError:
            discovered = await BleakScanner.discover(timeout=self.scan_timeout_seconds)

        if isinstance(discovered, dict):
            for device, advertisement_data in discovered.values():
                if self._matches_device(device, advertisement_data):
                    return device
        else:
            for device in discovered:
                if self._matches_device(device, None):
                    return device

        if self.device_address:
            return self.device_address
        return None

    def _matches_device(self, device, advertisement_data) -> bool:
        if self.device_address and getattr(device, "address", None) == self.device_address:
            return True
        if self.device_name and getattr(device, "name", None) == self.device_name:
            return True
        if (
            self.device_name
            and advertisement_data is not None
            and getattr(advertisement_data, "local_name", None) == self.device_name
        ):
            return True
        advertised_service_uuids = self._advertised_service_uuids(advertisement_data)
        return bool(
            self.service_uuid
            and self.service_uuid.lower() in [uuid.lower() for uuid in advertised_service_uuids]
        )

    @staticmethod
    def _advertised_service_uuids(advertisement_data) -> list[str]:
        if advertisement_data is None:
            return []
        return [str(uuid) for uuid in (advertisement_data.service_uuids or [])]

    def get_full_signal(self) -> np.ndarray:
        """
        Return all accumulated samples from analyze().
        Validates sample count and trims/waits as needed.

        Expected: sample_rate * analysis_time_seconds samples

        Returns:
            np.ndarray of shape (expected_samples,), dtype uint16

        Raises:
            BLEConnectionError: If validation times out
        """
        expected_samples = self._sample_rate * self._analysis_time_seconds

        print(f"Waiting for {expected_samples} samples...")

        # Wait up to 2 seconds for any lagging batches
        timeout = time.time() + 2.0
        while len(self._accumulated_data) < expected_samples and time.time() < timeout:
            time.sleep(0.01)

        actual_samples = len(self._accumulated_data)

        if actual_samples < expected_samples:
            print(f"Warning: Expected {expected_samples} samples, got {actual_samples}")

        if actual_samples > expected_samples:
            print(f"Trimming from {actual_samples} to {expected_samples} samples")

        # Return exactly expected_samples
        result = np.array(self._accumulated_data[:expected_samples], dtype=np.uint16)

        return result

    def _encode_start_packet(self, sample_rate: int, oversample_count: int, batch_size: int,
                            analysis_time_seconds: int, patient_name: str) -> bytes:
        """
        Encode binary START packet:
        Byte 0:        Command type (0x01)
        Bytes 1-4:     SAMPLE_RATE (uint32_t, little-endian)
        Bytes 5-6:     OVERSAMPLE_COUNT (uint16_t, little-endian)
        Bytes 7-8:     BATCH_SIZE (uint16_t, little-endian)
        Bytes 9-12:    ANALYSIS_TIME_SECONDS (uint32_t, little-endian)
        Bytes 13-28:   Patient name (null-terminated, max 16 bytes)
        """
        # Truncate patient name to 15 chars (16 bytes with null terminator)
        truncated_name = patient_name[:15].encode('utf-8')

        # Build packet
        packet = bytearray(29)  # Fixed size: 1 + 4 + 2 + 2 + 4 + 16

        packet[0] = 0x01  # START command
        struct.pack_into('<I', packet, 1, sample_rate)
        struct.pack_into('<H', packet, 5, oversample_count)
        struct.pack_into('<H', packet, 7, batch_size)
        struct.pack_into('<I', packet, 9, analysis_time_seconds)

        # Copy patient name (null-padded)
        packet[13:13+len(truncated_name)] = truncated_name
        # Rest is zeros (null padding)

        return bytes(packet)

    def _notification_handler(self, sender, data: bytearray):
        """
        BLE notification callback: parse batch and queue it.
        Expects data to be uint16_t values (2 bytes per sample).

        Synchronous on purpose: an async handler makes bleak schedule a new
        asyncio Task for every packet, which piles up at high sample rates.
        The queue is unbounded, so put_nowait never blocks.
        """
        try:
            # Convert bytearray to uint16 samples
            num_samples = len(data) // 2
            samples = np.frombuffer(data, dtype=np.uint16)[:num_samples]

            if not getattr(self, "_first_notif_seen", False):
                self._first_notif_seen = True
                print(f"[notif] FIRST packet received: {len(data)} bytes, {num_samples} samples")

            self._accumulated_data.extend(samples.tolist())

            # Queue for generator (runs in the event-loop thread, so this is safe)
            self._batch_queue.put_nowait(samples)
        except Exception as e:
            # bleak swallows callback exceptions silently; surface it.
            print(f"[notif] handler error: {e!r}")
