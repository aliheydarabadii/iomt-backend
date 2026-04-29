import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import create_app


def test_create_patient_endpoint_creates_patient_with_generated_id(client) -> None:
    response = client.post(
        "/api/patients",
        json={
            "fullName": "Daniel Carter",
            "mrn": "MRN-10999",
            "sex": "Male",
            "dob": "1989-05-02",
            "latestVisit": "2026-04-13T09:00:00Z",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == "patient_006"
    assert payload["fullName"] == "Daniel Carter"
    assert payload["mrn"] == "MRN-10999"
    assert payload["age"] == 36
    assert payload["recordings"] == []


def test_ble_status_endpoint_returns_diagnostics(client) -> None:
    response = client.get("/api/ble/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["connectionState"] in {"disabled", "unavailable"}
    assert payload["connectionLabel"] in {"BLE disabled", "BLE library unavailable"}
    assert payload["notificationCount"] == 0


def test_ble_control_endpoints_are_independent_from_recording(client) -> None:
    start = client.post("/api/ble/start")
    assert start.status_code == 200
    assert start.json() == {
        "success": False,
        "message": "BLE worker is disabled or unavailable",
    }

    stop = client.post("/api/ble/stop")
    assert stop.status_code == 200
    assert stop.json() == {"success": True, "message": "BLE worker stopped"}


def test_patient_search_works_when_ble_is_enabled_but_not_autostarted(
    session_factory,
    test_settings,
) -> None:
    settings = test_settings.model_copy(update={"ble_enabled": True, "ble_autostart": False})
    app = create_app(settings)

    def override_get_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        response = test_client.get("/api/patients/search", params={"name": "smith"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert [patient["id"] for patient in payload["patients"]] == ["patient_001", "patient_002"]


def test_recording_audio_endpoint_redirects_for_seeded_external_audio(client) -> None:
    response = client.get("/api/heart-recordings/rec_001/audio", follow_redirects=False)

    assert response.status_code in {307, 302}
    assert response.headers["location"] == "https://example.com/audio/rec_001.wav"


def test_create_patient_endpoint_rejects_duplicate_mrn(client) -> None:
    response = client.post(
        "/api/patients",
        json={
            "fullName": "Duplicate MRN",
            "mrn": "MRN-10021",
            "sex": "Male",
            "dob": "1990-01-01",
        },
    )

    assert response.status_code == 409
    payload = response.json()
    assert payload["error"]["code"] == "patient_mrn_exists"


def test_patient_search_endpoint_returns_matches_and_history(client) -> None:
    response = client.get("/api/patients/search", params={"name": "smith"})

    assert response.status_code == 200
    payload = response.json()
    assert [patient["id"] for patient in payload["patients"]] == ["patient_001", "patient_002"]
    assert payload["patients"][0]["recordings"][0]["id"] == "rec_001"


def test_patient_search_allows_secondary_vite_origin(client) -> None:
    response = client.get(
        "/api/patients/search",
        params={"name": "a"},
        headers={"Origin": "http://localhost:5174"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"


def test_patient_search_preflight_allows_secondary_vite_origin(client) -> None:
    response = client.options(
        "/api/patients/search?name=a",
        headers={
            "Origin": "http://localhost:5174",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5174"


def test_current_measurement_endpoint_returns_idle_session(client) -> None:
    response = client.get(
        "/api/heart-measurements/current",
        params={"patientId": "patient_001", "patientName": "John Smith"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sourceLabel"] == "REST endpoint"
    assert payload["currentSession"]["isRecording"] is False
    assert payload["currentSession"]["streamStatus"] == "Idle"
    assert len(payload["currentSession"]["waveform"]) == 180
    assert payload["controls"]["recordUrl"] == "/api/heart-measurements/patient_001/record"


def test_record_endpoint_starts_recording(client) -> None:
    response = client.post(
        "/api/heart-measurements/patient_003/record",
        json={"areaId": "mitral"},
    )

    assert response.status_code == 201
    assert response.json() == {"success": True, "message": "Recording started"}

    current = client.get(
        "/api/heart-measurements/current",
        params={"patientId": "patient_003"},
    )
    current_payload = current.json()
    assert current_payload["currentSession"]["isRecording"] is True
    assert current_payload["currentSession"]["activeAreaId"] == "mitral"
    assert current_payload["controls"]["canStop"] is True


def test_recording_command_endpoint_starts_and_stops_recording(client) -> None:
    start = client.post(
        "/api/heart-measurements/patient_005/recording",
        json={"action": "start", "areaId": "aortic"},
    )
    assert start.status_code == 201
    assert start.json() == {"success": True, "message": "Recording started"}

    current = client.get(
        "/api/heart-measurements/current",
        params={"patientId": "patient_005"},
    )
    current_payload = current.json()
    assert current_payload["currentSession"]["isRecording"] is True
    assert current_payload["currentSession"]["activeAreaId"] == "aortic"

    stop = client.post(
        "/api/heart-measurements/patient_005/recording",
        json={"action": "stop"},
    )
    assert stop.status_code == 200
    assert stop.json() == {"success": True, "message": "Recording stopped"}


def test_stop_endpoint_stops_recording_and_persists_history(client) -> None:
    start = client.post(
        "/api/heart-measurements/patient_004/record",
        json={"areaId": "tricuspid"},
    )
    assert start.status_code == 201

    stop = client.post("/api/heart-measurements/patient_004/stop")
    assert stop.status_code == 200
    assert stop.json() == {"success": True, "message": "Recording stopped"}

    current = client.get(
        "/api/heart-measurements/current",
        params={"patientId": "patient_004"},
    )
    payload = current.json()
    assert payload["currentSession"]["isRecording"] is False
    assert payload["records"][0]["areaId"] == "tricuspid"
    assert payload["records"][0]["audioUrl"].startswith("/api/heart-recordings/")
    audio = client.get(payload["records"][0]["audioUrl"])
    assert audio.status_code == 200
    assert len(audio.content) > 44


def test_recording_analysis_endpoint_returns_plotting_json(client) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pywt")
    pytest.importorskip("sklearn")

    start = client.post(
        "/api/heart-measurements/patient_003/record",
        json={"areaId": "aortic"},
    )
    assert start.status_code == 201

    stop = client.post("/api/heart-measurements/patient_003/stop")
    assert stop.status_code == 200

    current = client.get(
        "/api/heart-measurements/current",
        params={"patientId": "patient_003"},
    )
    recording_id = current.json()["records"][0]["id"]

    response = client.get(
        f"/api/heart-recordings/{recording_id}/analysis",
        params={"includeSignals": "true", "maxPoints": 250},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recordingId"] == recording_id
    assert payload["audioUrl"] == f"/api/heart-recordings/{recording_id}/audio"
    assert payload["plot"]["sampleRateHz"] == 500
    assert payload["plot"]["pointCount"] == len(payload["plot"]["timeAxisS"])
    assert payload["plot"]["pointCount"] == len(payload["plot"]["amplitude"])
    assert payload["plot"]["pointCount"] == len(payload["plot"]["envelope"])
    assert payload["analysis"]["file_info"]["sample_rate_hz"] == 500
    assert "segmentation" in payload["analysis"]
    assert "signals" in payload["analysis"]


def test_recording_analysis_endpoint_rejects_non_local_seed_audio(client) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("pywt")
    pytest.importorskip("sklearn")

    response = client.get("/api/heart-recordings/rec_001/analysis")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "recording_analysis_audio_not_found"
