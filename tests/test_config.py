from app.core.config import Settings


def test_settings_accepts_plain_string_cors_origins_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:4173")

    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://localhost:5173", "http://localhost:4173"]


def test_settings_accepts_json_array_cors_origins_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:5173"]')

    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://localhost:5173"]


def test_settings_ble_defaults_match_device_profile() -> None:
    settings = Settings(_env_file=None)

    assert settings.ble_enabled is True
    assert settings.ble_device_address is None
    assert settings.ble_device_name == "PCG_Monitor"
    assert settings.ble_service_uuid == "12345678-1234-1234-1234-123456789abc"
    assert settings.ble_characteristic_uuid == "abcd1234-ab12-cd34-ef56-123456789abc"
    assert settings.ble_payload_format == "uint16-le"
    assert settings.ble_autostart is False
    assert settings.ble_sample_rate == 500
    assert settings.ble_batch_size == 20
    assert settings.audio_sample_rate == 500
    assert settings.audio_gain == 2.0
    assert settings.ble_timer_interval_ms == 20
    assert settings.ble_scan_timeout_seconds == 10.0
    assert settings.ble_retry_delay_seconds == 1.0
    assert settings.ble_poll_interval_seconds == 1.0
    assert settings.ble_samples_per_tick == 10
