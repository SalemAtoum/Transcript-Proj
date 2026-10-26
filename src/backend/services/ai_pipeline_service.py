from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional


# ============================================================
# Paths
# ============================================================

SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_DIR.parents[2]

ORCHESTRATOR = (
    PROJECT_ROOT
    / "src"
    / "core"
    / "pipeline"
    / "orchestrator.py"
)


# ============================================================
# Types
# ============================================================

ProgressCallback = Callable[[str, int], None]


# ============================================================
# AI Pipeline Service
# ============================================================

class AIPipelineService:
    """
    Runs the production AI pipeline as a subprocess.

    Pipeline:
        Audio Preprocessing
        -> Whisper Transcription
        -> Speaker Diarization
        -> Speaker Identification
        -> Alignment
        -> Final Transcript
        -> Markdown Generation

    Progress lines are expected from the orchestrator in this format:

        SWBS_PROGRESS|stage_name|progress

    Example:

        SWBS_PROGRESS|audio_preprocessing|10
        SWBS_PROGRESS|transcription|25
        SWBS_PROGRESS|diarization|45
        SWBS_PROGRESS|speaker_identification|60
        SWBS_PROGRESS|alignment|75
        SWBS_PROGRESS|final_transcript|90
        SWBS_PROGRESS|markdown_generation|100

    IMPORTANT:
        "completed" is NOT forwarded to the backend as a processing stage.
        The background worker is responsible for marking the job completed.
    """

    PROGRESS_PATTERN = re.compile(
        r"SWBS_PROGRESS\|([^|]+)\|(\d+)"
    )

    def __init__(self) -> None:
        self.orchestrator = ORCHESTRATOR

    # ========================================================
    # Validation
    # ========================================================

    def _validate_paths(self) -> None:
        """Validate required project paths."""

        if not PROJECT_ROOT.exists():
            raise FileNotFoundError(
                f"Project root does not exist: {PROJECT_ROOT}"
            )

        if not self.orchestrator.exists():
            raise FileNotFoundError(
                f"Orchestrator not found: {self.orchestrator}"
            )

    def _validate_input(self, input_file: Path) -> None:
        """Validate input meeting file."""

        if not input_file.exists():
            raise FileNotFoundError(
                f"Input file does not exist: {input_file}"
            )

        if not input_file.is_file():
            raise ValueError(
                f"Input path is not a file: {input_file}"
            )

        if input_file.stat().st_size == 0:
            raise ValueError(
                f"Input file is empty: {input_file}"
            )

    def _validate_meeting_id(self, meeting_id: str) -> None:
        """Validate the meeting ID before passing it to the pipeline."""

        if not isinstance(meeting_id, str):
            raise TypeError(
                "meeting_id must be a string."
            )

        meeting_id = meeting_id.strip()

        if not meeting_id:
            raise ValueError(
                "meeting_id cannot be empty."
            )

        if "/" in meeting_id or "\\" in meeting_id:
            raise ValueError(
                f"Invalid meeting_id: {meeting_id}"
            )

        if ".." in meeting_id:
            raise ValueError(
                f"Invalid meeting_id: {meeting_id}"
            )

    # ========================================================
    # Progress Parsing
    # ========================================================

    def _handle_progress_line(
        self,
        line: str,
        progress_callback: Optional[ProgressCallback],
    ) -> None:
        """
        Parse a pipeline progress line and notify the backend.

        The special "completed" stage is intentionally ignored because
        it is a job status, not a ProcessingStage enum value.
        """

        match = self.PROGRESS_PATTERN.search(line)

        if not match:
            return

        stage = match.group(1).strip()
        progress = int(match.group(2))

        progress = max(0, min(100, progress))

        # ----------------------------------------------------
        # IMPORTANT:
        # "completed" belongs to JobStatus, not ProcessingStage.
        # Do NOT send it to ProcessingJobResponse.current_stage.
        # ----------------------------------------------------

        if stage.lower() == "completed":
            return

        if progress_callback is None:
            return

        try:
            progress_callback(stage, progress)

        except Exception as callback_error:
            # A callback problem must NOT kill the AI pipeline.
            print(
                f"WARNING: Progress callback failed: "
                f"{callback_error}",
                flush=True,
            )

    # ========================================================
    # Main Run Method
    # ========================================================

    def run(
        self,
        input_file: Path,
        meeting_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> bool:
        """
        Run the complete AI pipeline.

        Args:
            input_file:
                Path to the uploaded meeting audio/video file.

            meeting_id:
                Unique meeting ID used to isolate all pipeline artifacts.

            progress_callback:
                Optional callback used to report pipeline progress
                back to the backend job manager.

        Returns:
            True  -> pipeline completed successfully
            False -> pipeline failed
        """

        input_file = Path(input_file)

        print("=" * 70, flush=True)
        print("SWBS AI PIPELINE SERVICE", flush=True)
        print("=" * 70, flush=True)

        print(
            f"Project root: {PROJECT_ROOT}",
            flush=True,
        )

        print(
            f"Orchestrator: {self.orchestrator}",
            flush=True,
        )

        print(
            f"Meeting ID: {meeting_id}",
            flush=True,
        )

        print(
            f"Input file: {input_file}",
            flush=True,
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        self._validate_paths()
        self._validate_input(input_file)
        self._validate_meeting_id(meeting_id)

        # ----------------------------------------------------
        # Build command
        # ----------------------------------------------------

        command = [
            sys.executable,
            str(self.orchestrator),
            "--input-file",
            str(input_file),
            "--meeting-id",
            meeting_id,
        ]

        print(
            "COMMAND:",
            " ".join(
                f'"{arg}"' if " " in arg else arg
                for arg in command
            ),
            flush=True,
        )

        print("-" * 70, flush=True)

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------

        environment = os.environ.copy()

        # Force UTF-8 on Windows.
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"

        # ----------------------------------------------------
        # Start subprocess
        # ----------------------------------------------------

        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )

        except Exception as error:
            print(
                f"ERROR: Failed to start AI pipeline: {error}",
                flush=True,
            )
            return False

        # ----------------------------------------------------
        # Stream output
        # ----------------------------------------------------

        try:

            if process.stdout is not None:

                for raw_line in process.stdout:

                    line = raw_line.rstrip("\r\n")

                    if not line:
                        continue

                    # Always show pipeline output.
                    print(
                        line,
                        flush=True,
                    )

                    # Parse progress.
                    self._handle_progress_line(
                        line=line,
                        progress_callback=progress_callback,
                    )

        except Exception as stream_error:

            print(
                f"ERROR: Failed while reading pipeline output: "
                f"{stream_error}",
                flush=True,
            )

            # Make sure the process is not left running.
            try:
                process.kill()
            except Exception:
                pass

            try:
                process.wait(timeout=10)
            except Exception:
                pass

            return False

        # ----------------------------------------------------
        # Wait for process
        # ----------------------------------------------------

        try:
            return_code = process.wait()

        except Exception as wait_error:

            print(
                f"ERROR: Failed waiting for AI pipeline: "
                f"{wait_error}",
                flush=True,
            )

            return False

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        success = return_code == 0

        print("-" * 70, flush=True)

        print(
            f"AI pipeline exit code: {return_code}",
            flush=True,
        )

        if success:

            print(
                "AI PIPELINE COMPLETED SUCCESSFULLY",
                flush=True,
            )

        else:

            print(
                "AI PIPELINE FAILED",
                flush=True,
            )

        print("=" * 70, flush=True)

        return success


# ============================================================
# Singleton
# ============================================================

ai_pipeline_service = AIPipelineService()