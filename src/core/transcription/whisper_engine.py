from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import whisper


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

MODEL_NAME = "large-v3"

LANGUAGE = "ar"
TASK = "transcribe"

BEAM_SIZE = 5
TEMPERATURE = 0

CONDITION_ON_PREVIOUS_TEXT = True

# IMPORTANT:
# No initial prompt.
# Testing showed that prompts reduced Jordanian/Levantine
# transcription quality instead of improving it.
INITIAL_PROMPT = None

COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WordResult:
    word: str
    start: float
    end: float
    probability: float | None


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None
    compression_ratio: float | None
    no_speech_prob: float | None
    words: list[WordResult]


@dataclass
class TranscriptionResult:
    audio_file: str
    model: str
    device: str
    language: str
    task: str
    duration_seconds: float
    processing_seconds: float
    segment_count: int
    word_count: int
    segments: list[SegmentResult]


# ============================================================
# WHISPER ENGINE
# ============================================================

class WhisperEngine:
    """
    Production Whisper transcription engine.

    Responsibilities:
        - Load Whisper large-v3
        - Select CPU/GPU automatically
        - Transcribe Arabic audio
        - Return structured segments
        - Return word-level timestamps

    This class does NOT perform:
        - speaker diarization
        - speaker identification
        - alignment
        - Markdown generation
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ):
        self.model_name = model_name

        # Automatic device selection.
        # Development:
        #   CPU
        #
        # Production:
        #   CUDA GPU when available.
        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        print("=" * 70)
        print("Loading Whisper model")
        print("=" * 70)

        print(f"Model:  {self.model_name}")
        print(f"Device: {self.device}")

        self.model = whisper.load_model(
            self.model_name,
            device=self.device,
        )

        print("Whisper model loaded.")
        print("=" * 70)

    # ========================================================
    # SAFE FLOAT CONVERSION
    # ========================================================

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ========================================================
    # AUDIO DURATION
    # ========================================================

    @staticmethod
    def _get_audio_duration(
        audio_path: Path,
    ) -> float:

        import wave

        with wave.open(
            str(audio_path),
            "rb",
        ) as wav:

            sample_rate = wav.getframerate()
            frames = wav.getnframes()

        if sample_rate <= 0:
            return 0.0

        return frames / sample_rate

    # ========================================================
    # WORD CONVERSION
    # ========================================================

    @staticmethod
    def _convert_words(
        raw_words: list[dict[str, Any]] | None,
    ) -> list[WordResult]:

        if not raw_words:
            return []

        words: list[WordResult] = []

        for item in raw_words:

            word = str(
                item.get("word", "")
            )

            start = WhisperEngine._safe_float(
                item.get("start")
            )

            end = WhisperEngine._safe_float(
                item.get("end")
            )

            if start is None or end is None:
                continue

            probability = WhisperEngine._safe_float(
                item.get("probability")
            )

            words.append(
                WordResult(
                    word=word,
                    start=start,
                    end=end,
                    probability=probability,
                )
            )

        return words

    # ========================================================
    # SEGMENT CONVERSION
    # ========================================================

    @staticmethod
    def _convert_segments(
        raw_segments: list[dict[str, Any]],
    ) -> list[SegmentResult]:

        segments: list[SegmentResult] = []

        for index, segment in enumerate(raw_segments):

            start = WhisperEngine._safe_float(
                segment.get("start")
            )

            end = WhisperEngine._safe_float(
                segment.get("end")
            )

            if start is None or end is None:
                continue

            words = WhisperEngine._convert_words(
                segment.get("words")
            )

            segments.append(
                SegmentResult(
                    id=int(
                        segment.get(
                            "id",
                            index,
                        )
                    ),
                    start=start,
                    end=end,
                    text=str(
                        segment.get(
                            "text",
                            "",
                        )
                    ).strip(),
                    avg_logprob=WhisperEngine._safe_float(
                        segment.get("avg_logprob")
                    ),
                    compression_ratio=WhisperEngine._safe_float(
                        segment.get("compression_ratio")
                    ),
                    no_speech_prob=WhisperEngine._safe_float(
                        segment.get("no_speech_prob")
                    ),
                    words=words,
                )
            )

        return segments

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    def transcribe(
        self,
        audio_file: str | Path,
    ) -> TranscriptionResult:

        audio_path = Path(
            audio_file
        ).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist:\n{audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Audio path is not a file:\n{audio_path}"
            )

        print()
        print("=" * 70)
        print("WHISPER TRANSCRIPTION")
        print("=" * 70)

        print(f"Audio:    {audio_path}")
        print(f"Model:    {self.model_name}")
        print(f"Device:   {self.device}")
        print(f"Language: {LANGUAGE}")
        print(f"Beam:     {BEAM_SIZE}")
        print(f"Temp:     {TEMPERATURE}")

        print(
            "Previous context:",
            CONDITION_ON_PREVIOUS_TEXT,
        )

        duration = self._get_audio_duration(
            audio_path
        )

        print(
            f"Duration: {duration:.2f} seconds"
        )

        start_time = time.perf_counter()

        # ====================================================
        # WHISPER
        # ====================================================

        raw_result = self.model.transcribe(
            str(audio_path),

            language=LANGUAGE,
            task=TASK,

            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,

            condition_on_previous_text=(
                CONDITION_ON_PREVIOUS_TEXT
            ),

            initial_prompt=INITIAL_PROMPT,

            compression_ratio_threshold=(
                COMPRESSION_RATIO_THRESHOLD
            ),

            logprob_threshold=(
                LOGPROB_THRESHOLD
            ),

            no_speech_threshold=(
                NO_SPEECH_THRESHOLD
            ),

            # FP16 only when CUDA is available.
            fp16=(
                self.device == "cuda"
            ),

            # IMPORTANT:
            # Alignment depends on word timestamps.
            word_timestamps=True,

            verbose=False,
        )

        processing_seconds = (
            time.perf_counter()
            - start_time
        )

        segments = self._convert_segments(
            raw_result.get(
                "segments",
                [],
            )
        )

        word_count = sum(
            len(segment.words)
            for segment in segments
        )

        result = TranscriptionResult(
            audio_file=str(audio_path),
            model=self.model_name,
            device=self.device,
            language=str(
                raw_result.get(
                    "language",
                    LANGUAGE,
                )
            ),
            task=TASK,
            duration_seconds=duration,
            processing_seconds=processing_seconds,
            segment_count=len(segments),
            word_count=word_count,
            segments=segments,
        )

        print()
        print("Transcription completed.")

        print(
            f"Segments: {result.segment_count}"
        )

        print(
            f"Words:    {result.word_count}"
        )

        print(
            f"Runtime:  "
            f"{result.processing_seconds:.2f} seconds"
        )

        return result


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    result: TranscriptionResult,
    output_file: str | Path,
) -> None:

    output_path = Path(
        output_file
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(result)

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
    Phase 9.2 validation run.

    The engine supports two modes:

    1. STANDALONE TEST
       Uses the default 60-second test audio.

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

    project_root = Path(
        __file__
    ).resolve().parents[3]

    # --------------------------------------------------------
    # Standalone defaults
    # --------------------------------------------------------

    default_audio_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_whisper_test_60s.wav"
    )

    default_output_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_whisper_result.json"
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
    print("PHASE 9.2 — PRODUCTION WHISPER ENGINE")
    print("=" * 70)

    if pipeline_mode:
        print("Mode: PIPELINE")
    else:
        print("Mode: STANDALONE TEST")

    print()
    print(f"Input:")
    print(f"  {audio_file.resolve()}")

    print()
    print(f"Output:")
    print(f"  {output_file.resolve()}")

    print("=" * 70)

    # --------------------------------------------------------
    # Create engine
    # --------------------------------------------------------

    engine = WhisperEngine()

    # --------------------------------------------------------
    # Transcribe
    # --------------------------------------------------------

    result = engine.transcribe(
        audio_file
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_result(
        result,
        output_file,
    )

    # --------------------------------------------------------
    # Final validation output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 9.2 COMPLETE")
    print("=" * 70)

    print(
        f"Audio duration: "
        f"{result.duration_seconds:.2f}s"
    )

    print(
        f"Segments: "
        f"{result.segment_count}"
    )

    print(
        f"Words: "
        f"{result.word_count}"
    )

    print(
        f"Processing time: "
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