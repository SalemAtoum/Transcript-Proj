from fastapi import APIRouter, HTTPException, status

from src.backend.schemas import ProcessingJobResponse
from src.backend.services.job_manager import job_manager


router = APIRouter(
    prefix="/api/jobs",
    tags=["Processing"],
)


@router.get(
    "/{job_id}",
    response_model=ProcessingJobResponse,
    status_code=status.HTTP_200_OK,
)
def get_job_status(job_id: str) -> ProcessingJobResponse:
    """
    Return the current status of a processing job.

    This endpoint exposes a clean API response for the frontend
    instead of returning the internal JobManager dictionary.
    """

    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' was not found.",
        )

    return ProcessingJobResponse(
        job_id=job["job_id"],
        meeting_id=job["meeting_id"],
        status=job["status"],
        progress=job["progress"],
        current_stage=job.get("stage"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
    )
