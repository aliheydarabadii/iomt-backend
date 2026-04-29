import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.schemas.common import ErrorDetail, ErrorResponse
from app.services.ble_sensor import get_ble_ingestion_service


logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    current_settings = settings or get_settings()
    setup_logging(current_settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("starting application", extra={"environment": current_settings.environment})
        ble_service = get_ble_ingestion_service(current_settings)
        ble_service.bind_event_loop(asyncio.get_running_loop())
        if current_settings.ble_autostart:
            ble_service.start()
        yield
        ble_service.stop()
        logger.info("stopping application")

    app = FastAPI(
        title=current_settings.app_name,
        version="0.1.0",
        docs_url=current_settings.docs_url,
        redoc_url=current_settings.redoc_url,
        openapi_url=current_settings.openapi_url,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "ble", "description": "BLE ingestion and diagnostics"},
            {"name": "patients", "description": "Patient search and lookup"},
            {"name": "heart-measurements", "description": "Live measurement control"},
            {"name": "heart-recordings", "description": "Recorded audio and analysis access"},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=current_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if current_settings.audio_base_url.startswith("/"):
        storage_dir = Path(current_settings.audio_storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            current_settings.audio_base_url,
            StaticFiles(directory=storage_dir),
            name="audio-media",
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("request started", extra={"path": request.url.path, "method": request.method})
        response = await call_next(request)
        logger.info(
            "request completed",
            extra={"path": request.url.path, "method": request.method, "status": response.status_code},
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details)
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(
                code="validation_error",
                message="Invalid request input",
                details=exc.errors(),
            )
        )
        return JSONResponse(status_code=400, content=payload.model_dump())

    @app.exception_handler(SQLAlchemyError)
    async def handle_sqlalchemy_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("database error", exc_info=exc)
        payload = ErrorResponse(
            error=ErrorDetail(
                code="database_error",
                message="A database error occurred",
                details={},
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected error", exc_info=exc)
        payload = ErrorResponse(
            error=ErrorDetail(
                code="internal_server_error",
                message="An unexpected error occurred",
                details={},
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router, prefix=current_settings.api_prefix)
    return app


app = create_app()
