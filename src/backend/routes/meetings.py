from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.backend.schemas import (
    MeetingStatus,
    UploadMeetingResponse,
)


# ============================================================
# PATHS
# ============================================================

ROUTES_DIR = Path(__file__).resolve().parent

# routes/
#   -> backend/
#   -> src/
#   -> SWBS Proj/
PROJECT_ROOT = ROUTES_DIR.parents[2]

UPLOADS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".wav",
    ".mp3",
    ".m4a",
}

MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/meetings",
    tags=["Meetings"],
)


# ============================================================
# UPLOAD MEETING
# ============================================================

@router.post(
    "/upload",
    response_model=UploadMeetingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_meeting(
    file: UploadFile = File(...),
) -> UploadMeetingResponse:

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    original_filename = Path(
        file.filename
    ).name

    extension = Path(
        original_filename
    ).suffix.lower()

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:

        allowed = ", ".join(
            sorted(ALLOWED_EXTENSIONS)
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type. "
                f"Allowed formats: {allowed}"
            ),
        )

    # --------------------------------------------------------
    # Generate meeting ID
    # --------------------------------------------------------

    meeting_id = (
        f"meeting_{uuid.uuid4().hex[:12]}"
    )

    # --------------------------------------------------------
    # Create meeting directory
    # --------------------------------------------------------

    meeting_directory = (
        UPLOADS_DIRECTORY
        / meeting_id
    )

    meeting_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    output_file = (
        meeting_directory
        / original_filename
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    total_bytes = 0

    try:

        with output_file.open(
            "wb"
        ) as destination:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                # ------------------------------------------------
                # File size protection
                # ------------------------------------------------

                if total_bytes > MAX_FILE_SIZE_BYTES:

                    destination.close()

                    output_file.unlink(
                        missing_ok=True
                    )

                    meeting_directory.rmdir()

                    raise HTTPException(
                        status_code=(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail=(
                            "File is too large. "
                            "Maximum allowed size is 500 MB."
                        ),
                    )

                destination.write(
                    chunk
                )

    except HTTPException:
        raise

    except Exception as exc:

        output_file.unlink(
            missing_ok=True
        )

        if meeting_directory.exists():

            try:
                meeting_directory.rmdir()
            except OSError:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    finally:

        await file.close()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not output_file.exists():

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uploaded file was not saved.",
        )

    if output_file.stat().st_size == 0:

        output_file.unlink(
            missing_ok=True
        )

        try:
            meeting_directory.rmdir()
        except OSError:
            pass

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return UploadMeetingResponse(
        meeting_id=meeting_id,
        filename=original_filename,
        status=MeetingStatus.uploaded,
    )