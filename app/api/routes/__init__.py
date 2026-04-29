from fastapi import APIRouter

from app.api.routes.ble import router as ble_router
from app.api.routes.measurements import router as measurements_router
from app.api.routes.patients import router as patients_router
from app.api.routes.recordings import router as recordings_router

api_router = APIRouter()
api_router.include_router(ble_router)
api_router.include_router(patients_router)
api_router.include_router(measurements_router)
api_router.include_router(recordings_router)
