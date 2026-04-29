from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.core.database import Base, get_db
from app.main import create_app
from app.seed_data import seed_database


@pytest.fixture()
def test_settings(tmp_path, monkeypatch) -> Generator[Settings, None, None]:
    audio_storage_dir = tmp_path / "audio"
    monkeypatch.setenv("BLE_ENABLED", "false")
    monkeypatch.setenv("AUDIO_BASE_URL", "/media/audio")
    monkeypatch.setenv("AUDIO_STORAGE_DIR", str(audio_storage_dir))
    get_settings.cache_clear()
    settings = Settings(_env_file=None)
    yield settings
    get_settings.cache_clear()


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(engine)
    with TestingSessionLocal() as session:
        seed_database(session)
    yield TestingSessionLocal
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(session_factory, test_settings) -> Generator[Session, None, None]:
    with session_factory() as session:
        yield session


@pytest.fixture()
def client(session_factory, test_settings) -> Generator[TestClient, None, None]:
    app = create_app(test_settings)

    def override_get_db() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
