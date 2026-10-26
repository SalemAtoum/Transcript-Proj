from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# PIPELINE PATHS
# ============================================================

@dataclass
class PipelinePaths:
    """
    All filesystem paths used by one meeting pipeline run.

    Every meeting gets its own isolated workspace under:

        data/meetings/<meeting_id>/

    This prevents different meetings from overwriting each
    other's intermediate and final artifacts.
    """

    project_root: Path

    meeting_id: str
    meeting_directory: Path

    input_file: Path

    temporary_directory: Path
    output_directory: Path

    processed_audio: Path
    whisper_result: Path
    diarization_result: Path
    identification_result: Path
    alignment_result: Path
    final_transcript: Path

    markdown_output: Path


# ============================================================
# MEETING ID VALIDATION
# ============================================================

def _validate_meeting_id(meeting_id: str) -> str:
    """
    Validate and normalize meeting ID.

    Meeting IDs are used as directory names, so only safe
    filesystem characters are allowed.
    """

    if not isinstance(meeting_id, str):
        raise TypeError(
            "meeting_id must be a string."
        )

    meeting_id = meeting_id.strip()

    if not meeting_id:
        raise ValueError(
            "meeting_id cannot be empty."
        )

    # Prevent path traversal and nested paths.
    if (
        "/" in meeting_id
        or "\\" in meeting_id
        or ".." in meeting_id
    ):
        raise ValueError(
            f"Invalid meeting_id: {meeting_id}"
        )

    # Keep IDs predictable and filesystem-safe.
    allowed_characters = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-"
    )

    if any(
        character not in allowed_characters
        for character in meeting_id
    ):
        raise ValueError(
            f"Invalid meeting_id: {meeting_id}"
        )

    return meeting_id


# ============================================================
# CREATE PIPELINE PATHS
# ============================================================

def create_pipeline_paths(
    input_file: Path,
    meeting_id: str,
) -> PipelinePaths:
    """
    Create an isolated workspace for one meeting.

    Structure:

        data/
        └── meetings/
            └── <meeting_id>/
                ├── temporary/
                └── output/
    """

    input_file = Path(input_file)

    meeting_id = _validate_meeting_id(
        meeting_id
    )

    # --------------------------------------------------------
    # Meeting root
    # --------------------------------------------------------

    meeting_directory = (
        PROJECT_ROOT
        / "data"
        / "meetings"
        / meeting_id
    )

    # --------------------------------------------------------
    # Meeting temporary directory
    # --------------------------------------------------------

    temporary_directory = (
        meeting_directory
        / "temporary"
    )

    # --------------------------------------------------------
    # Meeting output directory
    # --------------------------------------------------------

    output_directory = (
        meeting_directory
        / "output"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Return all paths
    # --------------------------------------------------------

    return PipelinePaths(

        project_root=PROJECT_ROOT,

        meeting_id=meeting_id,

        meeting_directory=meeting_directory,

        input_file=input_file,

        temporary_directory=temporary_directory,

        output_directory=output_directory,

        processed_audio=(
            temporary_directory
            / "meeting_processed.wav"
        ),

        whisper_result=(
            temporary_directory
            / "whisper_result.json"
        ),

        diarization_result=(
            temporary_directory
            / "diarization_result.json"
        ),

        identification_result=(
            temporary_directory
            / "identification_result.json"
        ),

        alignment_result=(
            temporary_directory
            / "alignment_result.json"
        ),

        final_transcript=(
            temporary_directory
            / "final_transcript.json"
        ),

        markdown_output=(
            output_directory
            / "meeting_transcript.md"
        ),
    )