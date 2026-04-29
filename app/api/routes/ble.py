from fastapi import APIRouter, Query

from app.core.errors import BadRequestError
from app.schemas.measurements import ActionResponse
from app.schemas.common import BLEDeviceResponse, BLEDeviceScanResponse, BLEStatusResponse
from app.services.ble_sensor import get_ble_ingestion_service

router = APIRouter(prefix="/ble", tags=["ble"])


@router.get(
    "/status",
    response_model=BLEStatusResponse,
    summary="Inspect BLE connection and capture status",
)
def get_ble_status() -> BLEStatusResponse:
    snapshot = get_ble_ingestion_service().get_status()
    return _serialize_status(snapshot)


@router.get(
    "/devices",
    response_model=BLEDeviceScanResponse,
    summary="List nearby BLE devices visible to the backend",
)
async def list_ble_devices(
    timeout_seconds: float = Query(default=10.0, ge=1.0, le=30.0, alias="timeoutSeconds"),
) -> BLEDeviceScanResponse:
    service = get_ble_ingestion_service()
    if service.get_status().connection_state == "unavailable":
        raise BadRequestError(
            "ble_library_unavailable",
            "BLE library is unavailable in this Python environment",
        )
    try:
        devices = await service.scan_devices(timeout_seconds)
    except TimeoutError as exc:
        raise BadRequestError(
            "ble_scan_timeout",
            "BLE scan timed out before returning devices",
            {"timeoutSeconds": timeout_seconds},
        ) from exc
    except Exception as exc:
        raise BadRequestError(
            "ble_scan_failed",
            "BLE scan failed",
            {"error": str(exc)},
        ) from exc

    return BLEDeviceScanResponse(
        timeoutSeconds=timeout_seconds,
        targetDeviceName=service.settings.ble_device_name,
        targetServiceUuid=service.settings.ble_service_uuid,
        devices=[
            BLEDeviceResponse(
                name=device.name,
                address=device.address,
                localName=device.local_name,
                serviceUuids=device.service_uuids,
                rssi=device.rssi,
                manufacturerDataKeys=device.manufacturer_data_keys,
                isTargetMatch=device.is_target_match,
                matchedBy=device.matched_by,
            )
            for device in devices
        ],
    )


@router.post(
    "/start",
    response_model=ActionResponse,
    summary="Start BLE ingestion without starting a recording",
)
def start_ble() -> ActionResponse:
    service = get_ble_ingestion_service()
    started = service.start()
    if started:
        return ActionResponse(success=True, message="BLE worker started")
    return ActionResponse(success=False, message="BLE worker is disabled or unavailable")


@router.post(
    "/stop",
    response_model=ActionResponse,
    summary="Stop BLE ingestion without affecting patient records",
)
def stop_ble() -> ActionResponse:
    get_ble_ingestion_service().stop()
    return ActionResponse(success=True, message="BLE worker stopped")


def _serialize_status(snapshot) -> BLEStatusResponse:
    return BLEStatusResponse(
        enabled=snapshot.enabled,
        connectionState=snapshot.connection_state,
        connectionLabel=snapshot.connection_label,
        isRunning=snapshot.is_running,
        deviceName=snapshot.device_name,
        serviceUuid=snapshot.service_uuid,
        characteristicUuid=snapshot.characteristic_uuid,
        activeCapturePatientId=snapshot.active_capture_patient_id,
        recentSampleCount=snapshot.recent_sample_count,
        capturedSampleCount=snapshot.captured_sample_count,
        notificationCount=snapshot.notification_count,
        lastNotificationAt=snapshot.last_notification_at,
        lastError=snapshot.last_error,
    )
