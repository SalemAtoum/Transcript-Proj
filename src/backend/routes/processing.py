from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from src.backend.schemas import StartProcessingResponse
from src.backend.services.job_manager import job_manager
from src.backend.services.background_worker import background_worker
from src.backend.services.ai_pipeline_service import ai_pipeline_service


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/api/meetings",
    tags=["Processing"],
)


# ============================================================
# Paths
# ============================================================

ROUTES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROUTES_DIR.parents[2]

UPLOADS_DIRECTORY = PROJECT_ROOT / "data" / "uploads"


# ============================================================
# AI Pipeline Task
# ============================================================

def run_ai_pipeline_task(
    input_file: Path,
    job_id: str,
    meeting_id: str,
):
    """
    Run the real SWBS AI pipeline for an uploaded meeting.

    Progress events emitted by the orchestrator are forwarded
    to JobManager so the backend can expose real processing
    progress to the frontend.

    The meeting_id is passed all the way to the AI Pipeline
    Service so every meeting gets its own isolated workspace.
    """

    print("=" * 60)
    print("SWBS AI PIPELINE TASK")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Job ID: {job_id}")
    print(f"Meeting ID: {meeting_id}")
    print()

    # ---------------------------------------------------------
    # Progress callback
    # ---------------------------------------------------------

    def progress_callback(
        stage: str,
        progress: int,
    ) -> None:
        """
        Receive progress events from AIPipelineService
        and update the backend job state.
        """

        print(
            f"JOB PROGRESS | {stage} | {progress}%",
            flush=True,
        )

        job_manager.update_job(
            job_id,
            stage=stage,
            progress=progress,
        )

    # ---------------------------------------------------------
    # Run AI pipeline
    # ---------------------------------------------------------

    result = ai_pipeline_service.run(
        input_file,
        meeting_id=meeting_id,
        progress_callback=progress_callback,
    )

    print()
    print("SWBS AI PIPELINE TASK COMPLETED")
    print("=" * 60)

    return result


# ============================================================
# Start Processing Endpoint
# ============================================================

@router.post(
    "/{meeting_id}/process",
    response_model=StartProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_processing(
    meeting_id: str,
) -> StartProcessingResponse:
    """
    Create and queue a real AI processing job for an uploaded meeting.

    The job is executed asynchronously by the background worker.
    """

    # ---------------------------------------------------------
    # 1. Locate meeting directory
    # ---------------------------------------------------------

    meeting_directory = UPLOADS_DIRECTORY / meeting_id

    if not meeting_directory.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' was not found.",
        )

    if not meeting_directory.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' was not found.",
        )

    # ---------------------------------------------------------
    # 2. Find uploaded meeting file
    # ---------------------------------------------------------

    uploaded_files = [
        file
        for file in meeting_directory.iterdir()
        if file.is_file()
    ]

    if not uploaded_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Meeting '{meeting_id}' has no uploaded file.",
        )

    # The upload API currently stores one meeting file
    # inside each meeting directory.
    input_file = uploaded_files[0]

    # ---------------------------------------------------------
    # 3. Validate input file
    # ---------------------------------------------------------

    if input_file.stat().st_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Meeting '{meeting_id}' has an empty uploaded file.",
        )

    # ---------------------------------------------------------
    # 4. Create processing job
    # ---------------------------------------------------------

    job = job_manager.create_job(
        meeting_id=meeting_id
    )

    job_id = job["job_id"]

    # ---------------------------------------------------------
    # 5. Submit REAL AI pipeline to background worker
    # ---------------------------------------------------------

    background_worker.submit(
        job_id,
        lambda: run_ai_pipeline_task(
            input_file=input_file,
            job_id=job_id,
            meeting_id=meeting_id,
        ),
    )

    # ---------------------------------------------------------
    # 6. Return immediately
    # ---------------------------------------------------------

    return StartProcessingResponse(
        job_id=job_id,
        meeting_id=meeting_id,
        status="queued",
    )