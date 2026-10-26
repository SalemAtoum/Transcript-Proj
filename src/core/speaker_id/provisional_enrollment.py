from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .enrollment import SpeakerEnrollment


# ======================================================================
# CONFIGURATION
# ======================================================================

TARGET_SAMPLE_RATE = 16000

MIN_SEGMENT_SECONDS = 1.0
MAX_SEGMENTS = 10

# Minimum amount of clean speech required before creating a profile.
MIN_TOTAL_SPEECH_SECONDS = 3.0


# ======================================================================
# DATA MODEL
# ======================================================================

@dataclass
class EnrollmentSegment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProvisionalEnrollmentResult:
    meeting_id: str
    speaker_id: str
    employee_name: str

    segments_found: int
    segments_used: int

    speech_seconds_available: float
    speech_seconds_used: float

    enrollment_audio_file: str
    profile_file: str
    metadata_file: str

    status: str


# ======================================================================
# PROVISIONAL ENROLLMENT
# ======================================================================

class ProvisionalSpeakerEnrollment:
    """
    Create a provisional speaker profile after conversational identity
    confirmation.

    This component:

        1. Loads the meeting audio.
        2. Loads PyAnnote diarization.
        3. Finds segments belonging to the confirmed speaker.
        4. Selects the longest usable segments.
        5. Extracts only that speaker's audio.
        6. Saves a clean enrollment WAV.
        7. Reuses the existing SpeakerEnrollment component.
        8. Creates a profile compatible with SpeakerIdentifier.

    This component does NOT:

        - discover identity
        - confirm identity
        - modify transcript text
        - access the database
        - call an LLM
    """

    def __init__(
        self,
        enrollment: SpeakerEnrollment | None = None,
    ):
        self.enrollment = (
            enrollment
            if enrollment is not None
            else SpeakerEnrollment()
        )

    # ==================================================================
    # HELPERS
    # ==================================================================

    @staticmethod
    def _safe_name(name: str) -> str:
        """
        Convert an employee name into a filesystem-safe identifier.
        """

        name = name.strip().lower()

        name = re.sub(
            r"\s+",
            "_",
            name,
        )

        name = re.sub(
            r"[^a-zA-Z0-9_\-\u0600-\u06FF]",
            "",
            name,
        )

        if not name:
            raise ValueError(
                "Employee name produced an empty safe filename."
            )

        return name

    # ==================================================================
    # LOAD DIARIZATION
    # ==================================================================

    @staticmethod
    def load_diarization(
        diarization_file: str | Path,
    ) -> list[EnrollmentSegment]:

        path = Path(
            diarization_file
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                "Diarization result does not exist:\n"
                f"{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        raw_segments = data.get(
            "segments",
            [],
        )

        if not raw_segments:
            raise ValueError(
                "Diarization result contains no segments."
            )

        segments: list[EnrollmentSegment] = []

        for item in raw_segments:

            speaker = item.get("speaker")
            start = item.get("start")
            end = item.get("end")

            if (
                speaker is None
                or start is None
                or end is None
            ):
                continue

            start = float(start)
            end = float(end)

            if end <= start:
                continue

            segment = EnrollmentSegment(
                speaker=str(speaker),
                start=start,
                end=end,
            )

            if (
                segment.duration
                >= MIN_SEGMENT_SECONDS
            ):
                segments.append(segment)

        if not segments:
            raise ValueError(
                "No usable diarization segments found."
            )

        return segments

    # ==================================================================
    # SELECT SPEAKER SEGMENTS
    # ==================================================================

    @staticmethod
    def select_speaker_segments(
        segments: list[EnrollmentSegment],
        speaker_id: str,
    ) -> list[EnrollmentSegment]:

        speaker_segments = [
            segment
            for segment in segments
            if segment.speaker == speaker_id
        ]

        if not speaker_segments:
            return []

        # Longest segments first.
        speaker_segments.sort(
            key=lambda segment: segment.duration,
            reverse=True,
        )

        selected = speaker_segments[
            :MAX_SEGMENTS
        ]

        # Restore chronological order.
        selected.sort(
            key=lambda segment: segment.start
        )

        return selected

    # ==================================================================
    # LOAD MEETING AUDIO
    # ==================================================================

    @staticmethod
    def load_meeting_audio(
        audio_file: str | Path,
    ) -> tuple[np.ndarray, int]:

        path = Path(
            audio_file
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                "Meeting audio does not exist:\n"
                f"{path}"
            )

        waveform, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )

        # Convert stereo/multi-channel → mono.
        if waveform.ndim == 2:
            waveform = waveform.mean(
                axis=1
            )

        if waveform.ndim != 1:
            raise ValueError(
                "Expected mono or stereo audio. "
                f"Got shape: {waveform.shape}"
            )

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                "Production meeting audio must be "
                f"{TARGET_SAMPLE_RATE} Hz. "
                f"Got {sample_rate} Hz."
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        return waveform, sample_rate

    # ==================================================================
    # EXTRACT SPEAKER AUDIO
    # ==================================================================

    @staticmethod
    def extract_speaker_audio(
        waveform: np.ndarray,
        sample_rate: int,
        segments: list[EnrollmentSegment],
    ) -> np.ndarray:

        if not segments:
            raise ValueError(
                "Cannot extract audio from empty segment list."
            )

        chunks: list[np.ndarray] = []

        total_samples = len(waveform)

        for segment in segments:

            start_sample = int(
                segment.start
                * sample_rate
            )

            end_sample = int(
                segment.end
                * sample_rate
            )

            start_sample = max(
                0,
                start_sample,
            )

            end_sample = min(
                total_samples,
                end_sample,
            )

            if end_sample <= start_sample:
                continue

            chunk = waveform[
                start_sample:end_sample
            ]

            if len(chunk) == 0:
                continue

            chunks.append(chunk)

        if not chunks:
            raise ValueError(
                "No valid audio could be extracted "
                "from the selected speaker segments."
            )

        # Concatenate speaker-only chunks.
        speaker_audio = np.concatenate(
            chunks,
            axis=0,
        ).astype(np.float32)

        return speaker_audio

    # ==================================================================
    # SAVE EXTRACTED AUDIO
    # ==================================================================

    @staticmethod
    def save_enrollment_audio(
        waveform: np.ndarray,
        sample_rate: int,
        output_file: str | Path,
    ) -> Path:

        output_path = Path(
            output_file
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            str(output_path),
            waveform,
            sample_rate,
            subtype="PCM_16",
        )

        return output_path

    # ==================================================================
    # CREATE PROVISIONAL PROFILE
    # ==================================================================

    def create_provisional_profile(
        self,
        *,
        meeting_id: str,
        speaker_id: str,
        employee_name: str,
        meeting_audio_file: str | Path,
        diarization_file: str | Path,
        profile_output_directory: str | Path,
        temporary_output_directory: str | Path,
    ) -> ProvisionalEnrollmentResult:

        meeting_id = meeting_id.strip()
        speaker_id = speaker_id.strip()
        employee_name = employee_name.strip()

        if not meeting_id:
            raise ValueError(
                "meeting_id cannot be empty."
            )

        if not speaker_id:
            raise ValueError(
                "speaker_id cannot be empty."
            )

        if not employee_name:
            raise ValueError(
                "employee_name cannot be empty."
            )

        print()
        print("=" * 70)
        print("PROVISIONAL SPEAKER ENROLLMENT")
        print("=" * 70)

        print(
            f"Meeting ID:     {meeting_id}"
        )

        print(
            f"Speaker ID:     {speaker_id}"
        )

        print(
            f"Employee Name:  {employee_name}"
        )

        print(
            f"Meeting Audio:  "
            f"{Path(meeting_audio_file).resolve()}"
        )

        print(
            f"Diarization:    "
            f"{Path(diarization_file).resolve()}"
        )

        # --------------------------------------------------------------
        # Load diarization.
        # --------------------------------------------------------------

        all_segments = self.load_diarization(
            diarization_file
        )

        speaker_segments = [
            segment
            for segment in all_segments
            if segment.speaker == speaker_id
        ]

        speech_seconds_available = sum(
            segment.duration
            for segment in speaker_segments
        )

        print()
        print(
            f"Segments found: "
            f"{len(speaker_segments)}"
        )

        print(
            f"Speech available: "
            f"{speech_seconds_available:.2f}s"
        )

        # --------------------------------------------------------------
        # Select usable segments.
        # --------------------------------------------------------------

        selected_segments = (
            self.select_speaker_segments(
                all_segments,
                speaker_id,
            )
        )

        speech_seconds_used = sum(
            segment.duration
            for segment in selected_segments
        )

        print(
            f"Segments selected: "
            f"{len(selected_segments)}"
        )

        print(
            f"Speech selected: "
            f"{speech_seconds_used:.2f}s"
        )

        if (
            speech_seconds_used
            < MIN_TOTAL_SPEECH_SECONDS
        ):
            raise ValueError(
                "Insufficient speaker audio for "
                "provisional enrollment. "
                f"Required at least "
                f"{MIN_TOTAL_SPEECH_SECONDS:.2f}s, "
                f"got {speech_seconds_used:.2f}s."
            )

        # --------------------------------------------------------------
        # Load meeting audio.
        # --------------------------------------------------------------

        waveform, sample_rate = (
            self.load_meeting_audio(
                meeting_audio_file
            )
        )

        # --------------------------------------------------------------
        # Extract speaker-only audio.
        # --------------------------------------------------------------

        speaker_audio = (
            self.extract_speaker_audio(
                waveform=waveform,
                sample_rate=sample_rate,
                segments=selected_segments,
            )
        )

        extracted_duration = (
            len(speaker_audio)
            / sample_rate
        )

        print()
        print(
            f"Extracted speaker audio: "
            f"{extracted_duration:.2f}s"
        )

        # --------------------------------------------------------------
        # Save extracted enrollment WAV.
        # --------------------------------------------------------------

        temporary_dir = Path(
            temporary_output_directory
        ).resolve()

        safe_name = self._safe_name(
            employee_name
        )

        enrollment_audio_file = (
            temporary_dir
            / "speaker_enrollment"
            / (
                f"{safe_name}_"
                f"{speaker_id}_"
                f"provisional.wav"
            )
        )

        enrollment_audio_file = (
            self.save_enrollment_audio(
                waveform=speaker_audio,
                sample_rate=sample_rate,
                output_file=enrollment_audio_file,
            )
        )

        print(
            f"Enrollment WAV: "
            f"{enrollment_audio_file}"
        )

        # --------------------------------------------------------------
        # Reuse validated production enrollment.
        # --------------------------------------------------------------

        profile = (
            self.enrollment.create_profile(
                employee_name=employee_name,
                audio_file=enrollment_audio_file,
                output_directory=profile_output_directory,
            )
        )

        # --------------------------------------------------------------
        # Add lifecycle metadata.
        # --------------------------------------------------------------

        metadata_file = Path(
            profile.metadata_file
        ).resolve()

        if metadata_file.exists():

            with metadata_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        else:
            metadata = {}

        metadata.update(
            {
                "profile_status": "PROVISIONAL",
                "enrollment_source": (
                    "conversational_identity_discovery"
                ),
                "meeting_id": meeting_id,
                "source_speaker_id": speaker_id,
                "source_segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.duration,
                    }
                    for segment in selected_segments
                ],
                "speech_seconds_available": (
                    speech_seconds_available
                ),
                "speech_seconds_used": (
                    speech_seconds_used
                ),
                "confirmation_required": False,
                "confirmation_status": "CONFIRMED",
            }
        )

        with metadata_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # --------------------------------------------------------------
        # Result.
        # --------------------------------------------------------------

        result = ProvisionalEnrollmentResult(
            meeting_id=meeting_id,
            speaker_id=speaker_id,
            employee_name=employee_name,
            segments_found=len(
                speaker_segments
            ),
            segments_used=len(
                selected_segments
            ),
            speech_seconds_available=(
                speech_seconds_available
            ),
            speech_seconds_used=(
                speech_seconds_used
            ),
            enrollment_audio_file=str(
                enrollment_audio_file
            ),
            profile_file=str(
                profile.embedding_file
            ),
            metadata_file=str(
                profile.metadata_file
            ),
            status="PROVISIONAL_PROFILE_CREATED",
        )

        print()
        print("=" * 70)
        print("PROVISIONAL ENROLLMENT COMPLETED")
        print("=" * 70)

        print(
            f"Employee:        "
            f"{result.employee_name}"
        )

        print(
            f"Speaker:         "
            f"{result.speaker_id}"
        )

        print(
            f"Segments used:   "
            f"{result.segments_used}"
        )

        print(
            f"Speech used:     "
            f"{result.speech_seconds_used:.2f}s"
        )

        print(
            f"Profile:         "
            f"{result.profile_file}"
        )

        print(
            f"Metadata:        "
            f"{result.metadata_file}"
        )

        print(
            f"Status:          "
            f"{result.status}"
        )

        print("=" * 70)

        return result


# ======================================================================
# STANDALONE VALIDATION
# ======================================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    # --------------------------------------------------------------
    # Controlled Phase 9 test files.
    # --------------------------------------------------------------

    meeting_audio = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_test_60s.wav"
    )

    diarization_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_result.json"
    )

    profile_directory = (
        project_root
        / "data"
        / "temporary"
        / "phase9_enrollment_test"
    )

    temporary_directory = (
        project_root
        / "data"
        / "temporary"
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # This is a controlled validation only.
    #
    # It does NOT automatically enroll a real unknown speaker.
    # It simply validates extraction + profile creation using
    # the existing Phase 9 test artifacts.
    # --------------------------------------------------------------

    speaker_id = "SPEAKER_01"
    employee_name = "Yousef"

    print()
    print("=" * 70)
    print("PHASE 9.9.8A — PROVISIONAL ENROLLMENT TEST")
    print("=" * 70)

    component = (
        ProvisionalSpeakerEnrollment()
    )

    result = (
        component.create_provisional_profile(
            meeting_id="provisional_enrollment_test",
            speaker_id=speaker_id,
            employee_name=employee_name,
            meeting_audio_file=meeting_audio,
            diarization_file=diarization_file,
            profile_output_directory=profile_directory,
            temporary_output_directory=temporary_directory,
        )
    )

    print()
    print("=" * 70)
    print("PHASE 9.9.8A RESULT")
    print("=" * 70)

    print(
        f"Status: {result.status}"
    )

    print(
        f"Profile: {result.profile_file}"
    )

    print(
        f"Metadata: {result.metadata_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()