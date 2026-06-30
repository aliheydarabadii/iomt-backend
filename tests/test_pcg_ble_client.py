import asyncio
from types import SimpleNamespace

import pytest

import app.services.pcg_ble_client as pcg_ble_client
from app.services.pcg_ble_client import BLEConnectionError, PCGClient


@pytest.mark.parametrize(
    ("advertisement_data", "expected_address"),
    [
        (SimpleNamespace(local_name="PCG_Monitor", service_uuids=[]), "AA:BB"),
        (
            SimpleNamespace(
                local_name=None,
                service_uuids=["12345678-1234-1234-1234-123456789abc"],
            ),
            "CC:DD",
        ),
    ],
)
def test_find_target_device_matches_advertising_identity(
    advertisement_data,
    expected_address,
    monkeypatch,
) -> None:
    class FakeScanner:
        @staticmethod
        async def discover(timeout: float, return_adv: bool = False):
            assert return_adv is True
            return {
                "ignored": (
                    SimpleNamespace(name="Other", address="00:11"),
                    SimpleNamespace(local_name="Other", service_uuids=[]),
                ),
                "target": (
                    SimpleNamespace(name=None, address=expected_address),
                    advertisement_data,
                ),
            }

    monkeypatch.setattr(pcg_ble_client, "BleakScanner", FakeScanner)
    client = PCGClient(
        device_name="PCG_Monitor",
        service_uuid="12345678-1234-1234-1234-123456789abc",
        scan_timeout_seconds=0.01,
    )

    async def run() -> None:
        device = await client._find_target_device()
        assert device.address == expected_address

    asyncio.run(run())


def test_connect_stays_connected_when_service_metadata_is_unavailable(monkeypatch) -> None:
    device = SimpleNamespace(name="PCG_Monitor", address="AA:BB")

    class FakeScanner:
        @staticmethod
        async def discover(timeout: float, return_adv: bool = False):
            return [device]

    class FakeBleakClient:
        def __init__(self, target_device, timeout: float = 10.0) -> None:
            self.target_device = target_device
            self.timeout = timeout
            self.is_connected = False
            self.disconnect_calls = 0

        async def connect(self) -> None:
            self.is_connected = True

        async def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.is_connected = False

        @property
        def services(self):
            raise RuntimeError("service discovery cache not ready")

    monkeypatch.setattr(pcg_ble_client, "BleakScanner", FakeScanner)
    monkeypatch.setattr(pcg_ble_client, "BleakClient", FakeBleakClient)
    client = PCGClient(device_name="PCG_Monitor", scan_timeout_seconds=0.01)

    asyncio.run(client.connect())

    assert client.is_connected() is True
    assert client.client.disconnect_calls == 0


def test_analyze_raises_when_start_produces_no_notifications() -> None:
    class SilentBleakClient:
        def __init__(self) -> None:
            self.is_connected = True
            self.start_notify_calls = 0
            self.stop_notify_calls = 0
            self.write_calls = 0

        async def start_notify(self, _characteristic_uuid, _handler) -> None:
            self.start_notify_calls += 1

        async def stop_notify(self, _characteristic_uuid) -> None:
            self.stop_notify_calls += 1

        async def write_gatt_char(
            self,
            _characteristic_uuid,
            _packet,
            response: bool,
        ) -> None:
            assert response is False
            self.write_calls += 1

    client = PCGClient()
    client.client = SilentBleakClient()
    client.NOTIFICATION_TIMEOUT_SECONDS = 0.01

    async def run() -> None:
        with pytest.raises(
            BLEConnectionError,
            match="No BLE notifications received",
        ):
            async for _batch in client.analyze(
                sample_rate=500,
                oversample_count=8,
                batch_size=6,
                patient_name="Second",
                analysis_time_seconds=60,
            ):
                pass

    asyncio.run(run())

    assert client.client.start_notify_calls == 1
    assert client.client.write_calls == 1
    assert client.client.stop_notify_calls == 1
