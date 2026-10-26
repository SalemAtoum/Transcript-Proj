from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


# ============================================================
# CONFIG IMPORT
# ============================================================

if __package__:
    # Package execution:
    # python -m src.core.pipeline.orchestrator
    from .config import create_pipeline_paths
else:
    # Direct script execution:
    # python src\core\pipeline\orchestrator.py

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    from config import create_pipeline_paths


# ============================================================
# PIPELINE
# ============================================================

class MeetingPipeline:

    def __init__(
        self,
        input_file: Path,
        meeting_id: str,
    ) -> None:

        self.paths = create_pipeline_paths(
            input_file=input_file,
            meeting_id=meeting_id,
        )

        # ----------------------------------------------------
        # Speaker enrollment profiles
        #
        # These are GLOBAL registered speaker profiles.
        # They are intentionally NOT stored inside the
        # meeting workspace.
        # ----------------------------------------------------

        self.profile_directory = (
            PROJECT_ROOT
            / "data"
            / "temporary"
            / "phase9_enrollment_test"
        )

    # ========================================================
    # LOG
    # ========================================================

    @staticmethod
    def log(message: str) -> None:

        print(
            message,
            flush=True,
        )

    # ========================================================
    # PROGRESS
    # ========================================================

    @staticmethod
    def report_progress(
        stage: str,
        progress: int,
    ) -> None:
        """
        Emit a machine-readable progress event.

        Format:
            SWBS_PROGRESS|<stage>|<progress>

        The backend service parses these events while the
        orchestrator is running.
        """

        progress = max(
            0,
            min(
                100,
                progress,
            ),
        )

        print(
            f"SWBS_PROGRESS|{stage}|{progress}",
            flush=True,
        )

    # ========================================================
    # RUN COMPONENT
    # ========================================================

    def run_component(
        self,
        script: str,
        environment: dict[str, str],
    ) -> bool:

        script_path = (
            PROJECT_ROOT
            / script
        )

        if not script_path.exists():

            self.log(
                "ERROR: Component not found:"
            )

            self.log(
                f"  {script_path}"
            )

            return False

        env = os.environ.copy()
        env.update(environment)

        self.log("")

        self.log(
            f"Running component: {script}"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )

        if result.returncode != 0:

            self.log("")

            self.log(
                f"FAILED: {script}"
            )

            self.log(
                f"Exit code: {result.returncode}"
            )

            return False

        self.log(
            f"PASS: {script}"
        )

        return True

    # ========================================================
    # CHECK FILE
    # ========================================================

    def require_file(
        self,
        path: Path,
        name: str,
    ) -> bool:

        if not path.exists():

            self.log(
                f"ERROR: Missing {name}:"
            )

            self.log(
                f"  {path}"
            )

            return False

        if path.stat().st_size == 0:

            self.log(
                f"ERROR: {name} is empty:"
            )

            self.log(
                f"  {path}"
            )

            return False

        return True

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_input(self) -> bool:

        self.log("")

        self.log(
            "Validating input..."
        )

        if not self.paths.input_file.exists():

            self.log(
                "ERROR: Input file does not exist:"
            )

            self.log(
                f"  {self.paths.input_file}"
            )

            return False

        if self.paths.input_file.stat().st_size == 0:

            self.log(
                "ERROR: Input file is empty."
            )

            return False

        self.log(
            f"Input: {self.paths.input_file}"
        )

        self.log(
            f"Size: "
            f"{self.paths.input_file.stat().st_size:,} bytes"
        )

        self.log(
            f"Meeting ID: {self.paths.meeting_id}"
        )

        self.log(
            f"Meeting workspace: "
            f"{self.paths.meeting_directory}"
        )

        return True

    # ========================================================
    # SPEAKER PROFILE VALIDATION
    # ========================================================

    def validate_profiles(self) -> bool:

        self.log("")

        self.log(
            "Validating speaker profiles..."
        )

        if not self.profile_directory.exists():

            self.log(
                "ERROR: Speaker profile directory "
                "does not exist:"
            )

            self.log(
                f"  {self.profile_directory}"
            )

            return False

        required_profiles = [
            self.profile_directory
            / "salameh_profile.npy",

            self.profile_directory
            / "sami_profile.npy",
        ]

        for profile in required_profiles:

            if not profile.exists():

                self.log(
                    "ERROR: Missing speaker profile:"
                )

                self.log(
                    f"  {profile}"
                )

                return False

            if profile.stat().st_size == 0:

                self.log(
                    "ERROR: Speaker profile is empty:"
                )

                self.log(
                    f"  {profile}"
                )

                return False

        self.log(
            f"Profile directory: "
            f"{self.profile_directory}"
        )

        self.log(
            f"Profiles found: "
            f"{len(required_profiles)}"
        )

        return True

    # ========================================================
    # STAGE 1
    # ========================================================

    def audio_preprocessing(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 1 — AUDIO PREPROCESSING"
        )

        self.log("=" * 70)

        self.report_progress(
            "audio_preprocessing",
            10,
        )

        success = self.run_component(
            "src/core/audio/preprocessor.py",
            {
                "SWBS_INPUT_FILE": str(
                    self.paths.input_file
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.processed_audio
                ),

                "SWBS_METADATA_FILE": str(
                    self.paths.temporary_directory
                    / "audio_metadata.json"
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.processed_audio,
            "processed audio",
        )

    # ========================================================
    # STAGE 2
    # ========================================================

    def transcription(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 2 — WHISPER TRANSCRIPTION"
        )

        self.log("=" * 70)

        self.report_progress(
            "transcription",
            30,
        )

        success = self.run_component(
            "src/core/transcription/whisper_engine.py",
            {
                "SWBS_INPUT_FILE": str(
                    self.paths.processed_audio
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.whisper_result
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.whisper_result,
            "Whisper result",
        )

    # ========================================================
    # STAGE 3
    # ========================================================

    def diarization(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 3 — SPEAKER DIARIZATION"
        )

        self.log("=" * 70)

        self.report_progress(
            "diarization",
            50,
        )

        success = self.run_component(
            "src/core/diarization/pyannote_engine.py",
            {
                "SWBS_INPUT_FILE": str(
                    self.paths.processed_audio
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.diarization_result
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.diarization_result,
            "diarization result",
        )

    # ========================================================
    # STAGE 4
    # ========================================================

    def speaker_identification(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 4 — SPEAKER IDENTIFICATION"
        )

        self.log("=" * 70)

        self.report_progress(
            "speaker_identification",
            70,
        )

        success = self.run_component(
            "src/core/speaker_id/identifier.py",
            {
                "SWBS_INPUT_FILE": str(
                    self.paths.processed_audio
                ),

                "SWBS_DIARIZATION_FILE": str(
                    self.paths.diarization_result
                ),

                "SWBS_PROFILE_DIRECTORY": str(
                    self.profile_directory
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.identification_result
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.identification_result,
            "speaker identification result",
        )

    # ========================================================
    # STAGE 5
    # ========================================================

    def alignment(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 5 — SPEAKER ALIGNMENT"
        )

        self.log("=" * 70)

        self.report_progress(
            "alignment",
            85,
        )

        success = self.run_component(
            "src/core/alignment/alignment_engine.py",
            {
                "SWBS_WHISPER_FILE": str(
                    self.paths.whisper_result
                ),

                "SWBS_DIARIZATION_FILE": str(
                    self.paths.diarization_result
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.alignment_result
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.alignment_result,
            "alignment result",
        )

    # ========================================================
    # STAGE 6
    # ========================================================

    def identity_discovery(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 6 — CONVERSATIONAL IDENTITY DISCOVERY"
        )

        self.log("=" * 70)

        self.report_progress(
            "identity_discovery",
            90,
        )

        # Identity discovery is a meeting-specific artifact.
        # It stays inside the existing meeting temporary
        # workspace.
        identity_discovery_result = (
            self.paths.temporary_directory
            / "identity_discovery_result.json"
        )

        success = self.run_component(
            "src/core/speaker_id/production_identity_discovery.py",
            {
                "SWBS_ALIGNMENT_FILE": str(
                    self.paths.alignment_result
                ),

                "SWBS_OUTPUT_FILE": str(
                    identity_discovery_result
                ),

                "SWBS_MEETING_ID": self.paths.meeting_id,
            },
        )

        if not success:
            return False

        return self.require_file(
            identity_discovery_result,
            "identity discovery result",
        )

    # ========================================================
    # STAGE 7
    # ========================================================

    def final_transcript(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 7 — FINAL TRANSCRIPT"
        )

        self.log("=" * 70)

        self.report_progress(
            "final_transcript",
            92,
        )

        success = self.run_component(
            "src/core/output/final_transcript.py",
            {
                "SWBS_ALIGNMENT_FILE": str(
                    self.paths.alignment_result
                ),

                "SWBS_IDENTIFICATION_FILE": str(
                    self.paths.identification_result
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.final_transcript
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.final_transcript,
            "final transcript",
        )

    # ========================================================
    # STAGE 8
    # ========================================================

    def markdown(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "STAGE 8 — MARKDOWN OUTPUT"
        )

        self.log("=" * 70)

        self.report_progress(
            "markdown_generation",
            97,
        )

        success = self.run_component(
            "src/core/output/markdown_generator.py",
            {
                "SWBS_INPUT_FILE": str(
                    self.paths.final_transcript
                ),

                "SWBS_OUTPUT_FILE": str(
                    self.paths.markdown_output
                ),
            },
        )

        if not success:
            return False

        return self.require_file(
            self.paths.markdown_output,
            "Markdown transcript",
        )

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    def run(self) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(
            "SWBS MEETING TRANSCRIPTION PIPELINE"
        )

        self.log("=" * 70)

        self.log("")

        self.log(
            "Dynamic pipeline mode: ENABLED"
        )

        self.log("")

        self.log(
            f"Meeting ID: {self.paths.meeting_id}"
        )

        self.log(
            f"Workspace: "
            f"{self.paths.meeting_directory}"
        )

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        if not self.validate_input():
            return False

        # ----------------------------------------------------
        # PROFILES
        # ----------------------------------------------------

        if not self.validate_profiles():
            return False

        # ----------------------------------------------------
        # STAGE 1
        # ----------------------------------------------------

        if not self.audio_preprocessing():
            return False

        # ----------------------------------------------------
        # STAGE 2
        # ----------------------------------------------------

        if not self.transcription():
            return False

        # ----------------------------------------------------
        # STAGE 3
        # ----------------------------------------------------

        if not self.diarization():
            return False

        # ----------------------------------------------------
        # STAGE 4
        # ----------------------------------------------------

        if not self.speaker_identification():
            return False

        # ----------------------------------------------------
        # STAGE 5
        # ----------------------------------------------------

        if not self.alignment():
            return False

        # ----------------------------------------------------
        # STAGE 6
        # ----------------------------------------------------

        if not self.identity_discovery():
            return False

        # ----------------------------------------------------
        # STAGE 7
        # ----------------------------------------------------

        if not self.final_transcript():
            return False

        # ----------------------------------------------------
        # STAGE 8
        # ----------------------------------------------------

        if not self.markdown():
            return False

        # ----------------------------------------------------
        # COMPLETED
        # ----------------------------------------------------

        self.report_progress(
            "completed",
            100,
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        self.log("")
        self.log("=" * 70)

        self.log(
            "PIPELINE COMPLETED SUCCESSFULLY"
        )

        self.log("=" * 70)

        self.log("")

        self.log("Meeting workspace:")
        self.log(
            f"  {self.paths.meeting_directory}"
        )

        self.log("")

        self.log("Processed audio:")
        self.log(
            f"  {self.paths.processed_audio}"
        )

        self.log("")

        self.log("Whisper:")
        self.log(
            f"  {self.paths.whisper_result}"
        )

        self.log("")

        self.log("Diarization:")
        self.log(
            f"  {self.paths.diarization_result}"
        )

        self.log("")

        self.log("Speaker identification:")
        self.log(
            f"  {self.paths.identification_result}"
        )

        self.log("")

        self.log("Alignment:")
        self.log(
            f"  {self.paths.alignment_result}"
        )

        self.log("")

        self.log("Identity discovery:")
        self.log(
            f"  {self.paths.temporary_directory / 'identity_discovery_result.json'}"
        )

        self.log("")

        self.log("Final transcript:")
        self.log(
            f"  {self.paths.final_transcript}"
        )

        self.log("")

        self.log("Markdown:")
        self.log(
            f"  {self.paths.markdown_output}"
        )

        return True


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "SWBS Meeting Transcription Pipeline"
        )
    )

    # --------------------------------------------------------
    # Input file
    # --------------------------------------------------------

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Path to the meeting audio/video file."
        ),
    )

    # --------------------------------------------------------
    # Meeting ID
    # --------------------------------------------------------

    parser.add_argument(
        "--meeting-id",
        type=str,
        default="standalone_meeting",
        help=(
            "Unique meeting ID used to isolate "
            "pipeline artifacts."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Input resolution
    # --------------------------------------------------------

    if args.input_file:

        input_file = (
            Path(
                args.input_file
            )
            .expanduser()
            .resolve()
        )

    else:

        # Controlled development fallback.
        #
        # This keeps standalone execution available
        # when no input argument is supplied.

        input_file = (
            PROJECT_ROOT
            / "data"
            / "input"
            / "meeting.wav"
        )

    # --------------------------------------------------------
    # Input information
    # --------------------------------------------------------

    print("")

    print("=" * 70)

    print(
        "SWBS PIPELINE INPUT"
    )

    print("=" * 70)

    print("")

    print(
        "Input file:"
    )

    print(
        f"  {input_file}"
    )

    print("")

    print(
        "Meeting ID:"
    )

    print(
        f"  {args.meeting_id}"
    )

    print("")

    # --------------------------------------------------------
    # Create pipeline
    # --------------------------------------------------------

    pipeline = MeetingPipeline(
        input_file=input_file,
        meeting_id=args.meeting_id,
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    success = pipeline.run()

    print("")

    print(
        f"SUCCESS: {success}"
    )

    return 0 if success else 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )