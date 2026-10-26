from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Any

from src.backend.services.job_manager import job_manager


class BackgroundWorker:
    """
    Background worker for meeting-processing jobs.

    The worker runs jobs outside the FastAPI request/response cycle.

    For the current MVP:
    - ThreadPoolExecutor is used.
    - The actual AI pipeline will be connected in the next step.
    - Job status is updated through the shared JobManager.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="swbs-worker",
        )

    def submit(
        self,
        job_id: str,
        task: Callable[[], Any],
    ) -> Future:
        """
        Submit a job to the background worker.

        The task itself is responsible for the actual processing.
        """

        future = self._executor.submit(
            self._run_job,
            job_id,
            task,
        )

        return future

    def _run_job(
        self,
        job_id: str,
        task: Callable[[], Any],
    ) -> Any:
        """
        Execute one background job and update its lifecycle status.
        """

        job = job_manager.get_job(job_id)

        if job is None:
            raise ValueError(
                f"Job '{job_id}' was not found."
            )

        job_manager.update_job(
            job_id,
            status="processing",
            progress=0,
        )

        try:
            result = task()

            job_manager.update_job(
                job_id,
                status="completed",
                progress=100,
            )

            return result

        except Exception as exc:
            job_manager.update_job(
                job_id,
                status="failed",
                error=str(exc),
            )

            raise

    def shutdown(self) -> None:
        """
        Gracefully shut down the worker.
        """

        self._executor.shutdown(
            wait=True,
        )


background_worker = BackgroundWorker(
    max_workers=1,
)