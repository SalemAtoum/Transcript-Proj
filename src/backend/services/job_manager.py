from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


class JobManager:
    """
    Temporary in-memory job manager for the backend MVP.

    This will later be replaced/ backed by the project database.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create_job(self, meeting_id: str) -> dict[str, Any]:
        job_id = f"job_{uuid4().hex[:12]}"

        job = {
            "job_id": job_id,
            "meeting_id": meeting_id,
            "status": "queued",
            "stage": None,
            "progress": 0,
            "error": None,
            "created_at": datetime.now(timezone.utc),
            "started_at": None,
            "completed_at": None,
        }

        with self._lock:
            self._jobs[job_id] = job

        return job.copy()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            return job.copy()

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:

        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return None

            if status is not None:
                job["status"] = status

                if status == "processing" and job["started_at"] is None:
                    job["started_at"] = datetime.now(timezone.utc)

                if status in {"completed", "failed"}:
                    job["completed_at"] = datetime.now(timezone.utc)

            if stage is not None:
                job["stage"] = stage

            if progress is not None:
                job["progress"] = max(0, min(100, progress))

            if error is not None:
                job["error"] = error

            return job.copy()


# One shared manager for the running backend process.
job_manager = JobManager()