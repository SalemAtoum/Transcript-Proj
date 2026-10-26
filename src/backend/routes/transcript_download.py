from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/api/meetings",
    tags=["Transcript"],
)


# ============================================================
# Paths
# ============================================================

ROUTES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROUTES_DIR.parents[2]

MEETINGS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "meetings"
)


# ============================================================
# Helpers
# ============================================================

def get_meeting_directory(
    meeting_id: str,
) -> Path:
    """
    Return the isolated workspace directory for a meeting.
    """

    return MEETINGS_DIRECTORY / meeting_id


def get_markdown_file(
    meeting_id: str,
) -> Path:
    """
    Return the Markdown transcript belonging to one meeting.
    """

    return (
        get_meeting_directory(meeting_id)
        / "output"
        / "meeting_transcript.md"
    )


# ============================================================
# DOWNLOAD MARKDOWN
# ============================================================

@router.get(
    "/{meeting_id}/transcript/download",
    status_code=status.HTTP_200_OK,
)
def download_transcript(
    meeting_id: str,
) -> FileResponse:
    """
    Download the completed meeting transcript as Markdown.

    Every meeting has its own Markdown artifact:

        data/meetings/<meeting_id>/output/
            meeting_transcript.md
    """

    # --------------------------------------------------------
    # 1. Validate meeting
    # --------------------------------------------------------

    meeting_directory = get_meeting_directory(
        meeting_id
    )

    if not meeting_directory.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Meeting '{meeting_id}' "
                "was not found."
            ),
        )

    if not meeting_directory.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Meeting '{meeting_id}' "
                "was not found."
            ),
        )

    # --------------------------------------------------------
    # 2. Locate meeting-specific Markdown
    # --------------------------------------------------------

    markdown_file = get_markdown_file(
        meeting_id
    )

    if not markdown_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Transcript Markdown is not available yet. "
                "The meeting may still be processing."
            ),
        )

    if not markdown_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Transcript Markdown is not available."
            ),
        )

    if markdown_file.stat().st_size == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Transcript Markdown is empty."
            ),
        )

    # --------------------------------------------------------
    # 3. Return meeting-specific file
    # --------------------------------------------------------

    return FileResponse(
        path=str(markdown_file),
        media_type="text/markdown; charset=utf-8",
        filename="meeting_transcript.md",
    )