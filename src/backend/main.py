from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.routes.meetings import router as meetings_router
from src.backend.routes.processing import router as processing_router
from src.backend.routes.processing_status import router as processing_status_router
from src.backend.routes.transcript import router as transcript_router
from src.backend.routes.transcript_download import (
    router as transcript_download_router,
)
# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[2]


# ============================================================
# API CONFIGURATION
# ============================================================

API_TITLE = "SWBS Meeting Transcription API"

API_DESCRIPTION = (
    "Backend API for the SWBS meeting transcription system."
)

API_VERSION = "0.1.0"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    meetings_router
)
app.include_router(
    processing_router
)
app.include_router(
    processing_status_router
)
app.include_router(
    transcript_router
)
app.include_router(
    transcript_download_router
)
# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, str]:

    return {
        "status": "healthy",
        "service": "swbs-meeting-transcription-api",
        "version": API_VERSION,
    }


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:

    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )