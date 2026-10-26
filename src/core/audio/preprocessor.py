from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"


# ============================================================
# AUDIO METADATA
# ============================================================

@dataclass
class AudioMetadata:
    input_file: str
    output_file: str
    sample_rate: int
    channels: int
    codec: str
    duration_seconds: float
    frames: int


# ============================================================
# AUDIO PREPROCESSOR
# ============================================================

class AudioPreprocessor:
    """
    Production audio preprocessing layer.

    Converts an input audio/video file into:
        - mono
        - 16 kHz
        - PCM signed 16-bit WAV

    The original input file is never modified.
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
    ):
        self.ffmpeg_path = ffmpeg_path

    # ========================================================
    # FFMPEG
    # ========================================================

    def _run_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
    ) -> None:

        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            TARGET_CODEC,
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg audio preprocessing failed.\n\n"
                f"Input: {input_path}\n"
                f"Output: {output_path}\n\n"
                f"FFmpeg error:\n{result.stderr}"
            )

    # ========================================================
    # READ WAV METADATA
    # ========================================================

    @staticmethod
    def _read_wav_metadata(
        wav_path: Path,
    ) -> tuple[int, int, int, int, float]:

        import wave

        with wave.open(
            str(wav_path),
            "rb",
        ) as wav:

            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()

            duration = (
                frames / sample_rate
                if sample_rate
                else 0.0
            )

        return (
            channels,
            sample_rate,
            sample_width,
            frames,
            duration,
        )

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    def _validate_output(
        self,
        output_path: Path,
    ) -> AudioMetadata:

        if not output_path.exists():

            raise FileNotFoundError(
                "Expected processed audio was "
                f"not created: {output_path}"
            )

        if output_path.stat().st_size == 0:

            raise RuntimeError(
                f"Processed audio file is empty: "
                f"{output_path}"
            )

        (
            channels,
            sample_rate,
            sample_width,
            frames,
            duration,
        ) = self._read_wav_metadata(
            output_path
        )

        if channels != TARGET_CHANNELS:

            raise ValueError(
                f"Invalid channel count: {channels}. "
                f"Expected {TARGET_CHANNELS}."
            )

        if sample_rate != TARGET_SAMPLE_RATE:

            raise ValueError(
                f"Invalid sample rate: {sample_rate}. "
                f"Expected {TARGET_SAMPLE_RATE} Hz."
            )

        if sample_width != 2:

            raise ValueError(
                f"Invalid sample width: "
                f"{sample_width} bytes. "
                "Expected 2 bytes (PCM 16-bit)."
            )

        return AudioMetadata(
            input_file="",
            output_file=str(output_path),
            sample_rate=sample_rate,
            channels=channels,
            codec=TARGET_CODEC,
            duration_seconds=duration,
            frames=frames,
        )

    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        input_file: str | Path,
        output_file: str | Path,
        metadata_file: str | Path | None = None,
    ) -> AudioMetadata:

        input_path = Path(
            input_file
        ).resolve()

        output_path = Path(
            output_file
        ).resolve()

        if not input_path.exists():

            raise FileNotFoundError(
                f"Input file does not exist: "
                f"{input_path}"
            )

        if input_path.is_dir():

            raise ValueError(
                f"Input path is a directory, "
                f"not a file: {input_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if input_path == output_path:

            raise ValueError(
                "Input and output files must be "
                "different. The original input "
                "must remain unchanged."
            )

        self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
        )

        metadata = self._validate_output(
            output_path
        )

        metadata.input_file = str(
            input_path
        )

        if metadata_file is not None:

            metadata_path = Path(
                metadata_file
            ).resolve()

            metadata_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with metadata_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    asdict(metadata),
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        return metadata


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Standalone validation run and pipeline-compatible entry point.

    The component supports two modes:

    1. Standalone mode
       Uses the default development paths.

    2. Pipeline mode
       Uses environment variables supplied by the orchestrator.
    """

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__).resolve().parents[3]
    )

    # --------------------------------------------------------
    # Dynamic pipeline paths
    #
    # If the orchestrator provides these variables,
    # they are used.
    #
    # Otherwise, the original development paths
    # remain the defaults.
    # --------------------------------------------------------

    input_file = Path(
        os.getenv(
            "SWBS_INPUT_FILE",
            str(
                project_root
                / "data"
                / "input"
                / "meeting.wav"
            ),
        )
    )

    output_file = Path(
        os.getenv(
            "SWBS_OUTPUT_FILE",
            str(
                project_root
                / "data"
                / "temporary"
                / "phase9_test_processed.wav"
            ),
        )
    )

    metadata_file = Path(
        os.getenv(
            "SWBS_METADATA_FILE",
            str(
                project_root
                / "data"
                / "temporary"
                / "phase9_audio_metadata.json"
            ),
        )
    )

    # --------------------------------------------------------
    # Detect mode
    # --------------------------------------------------------

    pipeline_mode = (
        "SWBS_INPUT_FILE" in os.environ
        or "SWBS_OUTPUT_FILE" in os.environ
    )

    # --------------------------------------------------------
    # Create processor
    # --------------------------------------------------------

    processor = AudioPreprocessor()

    print("=" * 70)

    if pipeline_mode:

        print(
            "PHASE 9.1 — PRODUCTION AUDIO PREPROCESSOR"
        )

        print(
            "Mode: PIPELINE"
        )

    else:

        print(
            "PHASE 9.1 — PRODUCTION AUDIO PREPROCESSOR"
        )

        print(
            "Mode: STANDALONE TEST"
        )

    print("=" * 70)

    print()
    print(
        "Input:"
    )
    print(
        f"  {input_file}"
    )

    print()
    print(
        "Output:"
    )
    print(
        f"  {output_file}"
    )

    print()
    print(
        "Metadata:"
    )
    print(
        f"  {metadata_file}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    metadata = processor.process(
        input_file=input_file,
        output_file=output_file,
        metadata_file=metadata_file,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print(
        "PASS — Audio preprocessing completed."
    )

    print()
    print(
        f"Input:        {metadata.input_file}"
    )

    print(
        f"Output:       {metadata.output_file}"
    )

    print(
        f"Sample rate:  {metadata.sample_rate} Hz"
    )

    print(
        f"Channels:     {metadata.channels}"
    )

    print(
        f"Codec:        {metadata.codec}"
    )

    print(
        f"Frames:       {metadata.frames}"
    )

    print(
        f"Duration:     "
        f"{metadata.duration_seconds:.2f} seconds"
    )

    print(
        f"Metadata:     {metadata_file}"
    )

    print()
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()