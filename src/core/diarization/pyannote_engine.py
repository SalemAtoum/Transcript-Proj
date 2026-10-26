from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# ============================================================
# PYANNOTE CONFIGURATION
# ============================================================

MODEL_NAME = "pyannote/speaker-diarization-3.1"

TARGET_SAMPLE_RATE = 16000


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float
    duration: float


@dataclass
class DiarizationResult:
    audio_file: str
    model: str
    device: str
    duration_seconds: float
    processing_seconds: float
    speaker_count: int
    speakers: list[str]
    segment_count: int
    segments: list[DiarizationSegment]


# ============================================================
# PYANNOTE ENGINE
# ============================================================

class PyannoteEngine:
    """
    Production speaker diarization engine.

    Responsibilities:
        - Load pyannote speaker diarization model
        - Load audio using SoundFile
        - Convert waveform to torch tensor
        - Detect anonymous speakers
        - Return speaker/time segments

    This class does NOT:
        - identify speaker names
        - perform Whisper transcription
        - align words
        - generate final Markdown
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ):
        self.model_name = model_name

        # ----------------------------------------------------
        # Load environment variables
        # ----------------------------------------------------

        load_dotenv()

        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN was not found in the environment.\n"
                "Make sure your .env file contains:\n"
                "HF_TOKEN=YOUR_TOKEN"
            )

        # ----------------------------------------------------
        # Automatic device selection
        # ----------------------------------------------------

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        # ----------------------------------------------------
        # Load PyAnnote model
        # ----------------------------------------------------

        print("=" * 70)
        print("Loading PyAnnote diarization model")
        print("=" * 70)

        print(f"Model:  {self.model_name}")
        print(f"Device: {self.device}")

        self.pipeline = Pipeline.from_pretrained(
            self.model_name,
            token=token,
        )

        self.pipeline.to(
            torch.device(self.device)
        )

        print("PyAnnote model loaded.")
        print("=" * 70)

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    @staticmethod
    def _load_audio(
        audio_path: Path,
    ) -> tuple[torch.Tensor, int, float]:

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

        # ----------------------------------------------------
        # Convert stereo → mono if necessary
        # ----------------------------------------------------

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        if waveform.ndim != 1:
            raise ValueError(
                f"Expected mono waveform after conversion. "
                f"Shape: {waveform.shape}"
            )

        # ----------------------------------------------------
        # Validate sample rate
        # ----------------------------------------------------

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Expected {TARGET_SAMPLE_RATE} Hz audio, "
                f"got {sample_rate} Hz."
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Convert NumPy → Torch
        #
        # Shape:
        #     (1, samples)
        #
        # This avoids the TorchCodec/FFmpeg loading issue
        # encountered during Phase 4.
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            waveform
        ).unsqueeze(0)

        duration = (
            len(waveform) / sample_rate
        )

        return (
            tensor,
            sample_rate,
            duration,
        )

    # ========================================================
    # DIARIZE
    # ========================================================

    def diarize(
        self,
        audio_file: str | Path,
    ) -> DiarizationResult:

        audio_path = Path(
            audio_file
        ).resolve()

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist:\n"
                f"{audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Audio path is not a file:\n"
                f"{audio_path}"
            )

        print()
        print("=" * 70)
        print("PYANNOTE DIARIZATION")
        print("=" * 70)

        print(f"Audio:  {audio_path}")
        print(f"Model:  {self.model_name}")
        print(f"Device: {self.device}")

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        (
            waveform,
            sample_rate,
            duration,
        ) = self._load_audio(
            audio_path
        )

        print(
            f"Sample rate: {sample_rate} Hz"
        )

        print(
            f"Duration: {duration:.2f} seconds"
        )

        # ----------------------------------------------------
        # PyAnnote input
        # ----------------------------------------------------

        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        # ----------------------------------------------------
        # Run diarization
        # ----------------------------------------------------

        start_time = time.perf_counter()

        output = self.pipeline(
            audio_input
        )

        processing_seconds = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # PyAnnote 4.x output
        #
        # We explicitly use speaker_diarization.
        # ----------------------------------------------------

        diarization = (
            output.speaker_diarization
        )

        segments: list[
            DiarizationSegment
        ] = []

        speakers: set[str] = set()

        # ----------------------------------------------------
        # Extract speaker/time segments
        # ----------------------------------------------------

        for turn, _, speaker in (
            diarization.itertracks(
                yield_label=True
            )
        ):

            start = float(
                turn.start
            )

            end = float(
                turn.end
            )

            if end <= start:
                continue

            segment = DiarizationSegment(
                speaker=str(speaker),
                start=start,
                end=end,
                duration=end - start,
            )

            segments.append(
                segment
            )

            speakers.add(
                str(speaker)
            )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        segments.sort(
            key=lambda item: (
                item.start,
                item.end,
            )
        )

        speaker_list = sorted(
            speakers
        )

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = DiarizationResult(
            audio_file=str(audio_path),
            model=self.model_name,
            device=self.device,
            duration_seconds=duration,
            processing_seconds=processing_seconds,
            speaker_count=len(speaker_list),
            speakers=speaker_list,
            segment_count=len(segments),
            segments=segments,
        )

        # ----------------------------------------------------
        # Console summary
        # ----------------------------------------------------

        print()
        print("Diarization completed.")

        print(
            f"Speakers:  {result.speaker_count}"
        )

        print(
            f"Segments:  {result.segment_count}"
        )

        print(
            f"Runtime:   "
            f"{result.processing_seconds:.2f}s"
        )

        return result


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    result: DiarizationResult,
    output_file: str | Path,
) -> None:

    output_path = Path(
        output_file
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        result
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nJSON result saved to:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Phase 9.3 validation run.

    Supports two modes:

    1. STANDALONE TEST
       Uses the existing 60-second diarization test audio.

    2. PIPELINE MODE
       Uses environment variables supplied by the
       production orchestrator.

    Environment variables:

        SWBS_INPUT_FILE
        SWBS_OUTPUT_FILE
    """

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__).resolve().parents[3]
    )

    # --------------------------------------------------------
    # Standalone defaults
    # --------------------------------------------------------

    default_audio_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_test_60s.wav"
    )

    default_output_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_result.json"
    )

    # --------------------------------------------------------
    # Pipeline environment variables
    # --------------------------------------------------------

    env_input_file = os.getenv(
        "SWBS_INPUT_FILE"
    )

    env_output_file = os.getenv(
        "SWBS_OUTPUT_FILE"
    )

    pipeline_mode = (
        env_input_file is not None
        or env_output_file is not None
    )

    # --------------------------------------------------------
    # Resolve input/output
    # --------------------------------------------------------

    audio_file = Path(
        env_input_file
        if env_input_file
        else default_audio_file
    )

    output_file = Path(
        env_output_file
        if env_output_file
        else default_output_file
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 9.3 — PRODUCTION PYANNOTE ENGINE"
    )
    print("=" * 70)

    if pipeline_mode:
        print("Mode: PIPELINE")
    else:
        print("Mode: STANDALONE TEST")

    print()
    print("Input:")
    print(
        f"  {audio_file.resolve()}"
    )

    print()
    print("Output:")
    print(
        f"  {output_file.resolve()}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Create engine
    # --------------------------------------------------------

    engine = PyannoteEngine()

    # --------------------------------------------------------
    # Run diarization
    # --------------------------------------------------------

    result = engine.diarize(
        audio_file
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    save_result(
        result,
        output_file,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 9.3 COMPLETE")
    print("=" * 70)

    print(
        f"Duration: "
        f"{result.duration_seconds:.2f}s"
    )

    print(
        f"Speakers: "
        f"{result.speakers}"
    )

    print(
        f"Segments: "
        f"{result.segment_count}"
    )

    print(
        f"Runtime: "
        f"{result.processing_seconds:.2f}s"
    )

    print(
        f"Result: "
        f"{output_file.resolve()}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()