import json
from functools import lru_cache
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "Heart Sound Measurement Backend"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/iomt_backend"
    )
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://localhost:5174",
    ]
    audio_base_url: str = "/media/audio"
    audio_storage_dir: str = "storage/audio"
    idle_waveform_size: int = 180
    live_waveform_size: int = 180
    log_level: str = "INFO"
    ble_enabled: bool = True
    ble_autostart: bool = False
    ble_device_address: str | None = None
    ble_device_name: str | None = "PCG_Monitor"
    ble_service_uuid: str | None = "12345678-1234-1234-1234-123456789abc"
    ble_characteristic_uuid: str | None = "abcd1234-ab12-cd34-ef56-123456789abc"
    ble_payload_format: str = "uint16-le"
    ble_sample_rate: int = 500
    ble_batch_size: int = 20
    audio_sample_rate: int = 500
    audio_gain: float = 2.0
    ble_timer_interval_ms: int = 20
    ble_scan_timeout_seconds: float = 10.0
    ble_connect_timeout_seconds: float = 10.0
    ble_retry_delay_seconds: float = 1.0
    ble_poll_interval_seconds: float = 1.0
    ble_stale_after_seconds: float = 1.5
    ble_recent_buffer_size: int = 2048
    ble_capture_buffer_size: int = 120000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str] | tuple[str, ...] | Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                decoded = json.loads(stripped)
                return [str(origin).strip() for origin in decoded if str(origin).strip()]
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        if isinstance(value, tuple):
            return [origin.strip() for origin in value if origin.strip()]
        if isinstance(value, list):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return value

    @field_validator(
        "audio_base_url",
        "audio_storage_dir",
        "ble_device_address",
        "ble_device_name",
        "ble_service_uuid",
        "ble_characteristic_uuid",
        "ble_payload_format",
        mode="before",
    )
    @classmethod
    def normalize_optional_ble_strings(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        return cleaned or None

    @property
    def ble_samples_per_tick(self) -> int:
        return max(1, self.ble_sample_rate * self.ble_timer_interval_ms // 1000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
