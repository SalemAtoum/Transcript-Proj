import json
from pathlib import Path

import soundfile as sf
import whisper


# ============================================================
# ROUND 10A
# Realistic Meeting Quality Validation
# Long-form transcription sampling
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_FILE = BASE_DIR / "meeting_processed.wav"

OUTPUT_DIR = BASE_DIR / "round10a_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "phase3_round10a_results.json"
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "large-v3"

DEVICE = "cpu"

LANGUAGE = "ar"

TASK = "transcribe"

FP16 = False

BEAM_SIZE = 5

TEMPERATURE = 0.0

CONDITION_ON_PREVIOUS_TEXT = True

COMPRESSION_RATIO_THRESHOLD = 2.4

LOGPROB_THRESHOLD = -1.0

NO_SPEECH_THRESHOLD = 0.6


# ============================================================
# Long samples
# ============================================================

# We deliberately choose different parts of the meeting.
#
# Sample A:
# Beginning of meeting
#
# Sample B:
# Middle of meeting
#
# Sample C:
# Later part of meeting
#
# Each sample = 120 seconds.

SAMPLES = {

    "sample_A_early": {
        "start_seconds": 0,
        "duration_seconds": 120,
    },

    "sample_B_middle": {
        "start_seconds": 1200,
        "duration_seconds": 120,
    },

    "sample_C_late": {
        "start_seconds": 1800,
        "duration_seconds": 120,
    },
}


# ============================================================
# Extract audio
# ============================================================

def extract_sample(
    audio,
    start_seconds,
    duration_seconds,
):

    start_frame = int(
        start_seconds * 16000
    )

    end_frame = int(
        (start_seconds + duration_seconds)
        * 16000
    )

    return audio[start_frame:end_frame]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 10A")
    print("Realistic Meeting Quality Validation")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate audio
    # --------------------------------------------------------

    if not AUDIO_FILE.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_FILE}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print()
    print("[1/4] Loading audio...")

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
        always_2d=False,
    )

    if sample_rate != 16000:

        raise ValueError(
            f"Expected 16000 Hz audio, "
            f"got {sample_rate} Hz"
        )

    print(
        f"Sample rate: {sample_rate} Hz"
    )

    print(
        f"Total audio samples: {len(audio)}"
    )

    print(
        f"Total duration: "
        f"{len(audio) / sample_rate:.2f} seconds"
    )

    # --------------------------------------------------------
    # Load Whisper
    # --------------------------------------------------------

    print()
    print("[2/4] Loading Whisper large-v3...")

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Run transcription
    # --------------------------------------------------------

    print()
    print("[3/4] Running long-form samples...")

    results = {

        "experiment": "Round 10A",

        "description": (
            "Realistic long-form transcription "
            "validation using three 120-second "
            "meeting samples."
        ),

        "configuration": {

            "model": MODEL_NAME,

            "device": DEVICE,

            "language": LANGUAGE,

            "task": TASK,

            "fp16": FP16,

            "beam_size": BEAM_SIZE,

            "temperature": TEMPERATURE,

            "condition_on_previous_text":
                CONDITION_ON_PREVIOUS_TEXT,

            "compression_ratio_threshold":
                COMPRESSION_RATIO_THRESHOLD,

            "logprob_threshold":
                LOGPROB_THRESHOLD,

            "no_speech_threshold":
                NO_SPEECH_THRESHOLD,

            "initial_prompt": None,
        },

        "samples": {},
    }

    for sample_name, sample_info in SAMPLES.items():

        print()
        print("-" * 70)

        print(
            f"Running {sample_name}"
        )

        print(
            f"Start: "
            f"{sample_info['start_seconds']} sec"
        )

        print(
            f"Duration: "
            f"{sample_info['duration_seconds']} sec"
        )

        print("-" * 70)

        sample_audio = extract_sample(
            audio=audio,
            start_seconds=
                sample_info["start_seconds"],
            duration_seconds=
                sample_info["duration_seconds"],
        )

        result = model.transcribe(

            sample_audio,

            language=LANGUAGE,

            task=TASK,

            fp16=FP16,

            temperature=TEMPERATURE,

            condition_on_previous_text=
                CONDITION_ON_PREVIOUS_TEXT,

            compression_ratio_threshold=
                COMPRESSION_RATIO_THRESHOLD,

            logprob_threshold=
                LOGPROB_THRESHOLD,

            no_speech_threshold=
                NO_SPEECH_THRESHOLD,

            beam_size=BEAM_SIZE,

            initial_prompt=None,

            verbose=False,
        )

        segments = result.get(
            "segments",
            [],
        )

        transcript = (
            result["text"]
            .strip()
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        avg_logprob = None

        avg_compression_ratio = None

        avg_no_speech_prob = None

        if segments:

            logprobs = [

                s.get(
                    "avg_logprob",
                    0.0,
                )

                for s in segments
            ]

            compression_ratios = [

                s.get(
                    "compression_ratio",
                    0.0,
                )

                for s in segments
            ]

            no_speech_probs = [

                s.get(
                    "no_speech_prob",
                    0.0,
                )

                for s in segments
            ]

            avg_logprob = (
                sum(logprobs)
                / len(logprobs)
            )

            avg_compression_ratio = (
                sum(compression_ratios)
                / len(compression_ratios)
            )

            avg_no_speech_prob = (
                sum(no_speech_probs)
                / len(no_speech_probs)
            )

        # ----------------------------------------------------
        # Save detailed segments
        # ----------------------------------------------------

        detailed_segments = []

        for segment in segments:

            detailed_segments.append({

                "start":
                    segment["start"],

                "end":
                    segment["end"],

                "text":
                    segment["text"].strip(),

                "avg_logprob":
                    segment.get(
                        "avg_logprob"
                    ),

                "compression_ratio":
                    segment.get(
                        "compression_ratio"
                    ),

                "no_speech_prob":
                    segment.get(
                        "no_speech_prob"
                    ),
            })

        results["samples"][
            sample_name
        ] = {

            "start_seconds":
                sample_info[
                    "start_seconds"
                ],

            "duration_seconds":
                sample_info[
                    "duration_seconds"
                ],

            "transcript":
                transcript,

            "segment_count":
                len(segments),

            "avg_logprob":
                avg_logprob,

            "avg_compression_ratio":
                avg_compression_ratio,

            "avg_no_speech_prob":
                avg_no_speech_prob,

            "segments":
                detailed_segments,
        }

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print()

        print("TRANSCRIPT:")

        print(transcript)

        print()

        print(
            f"Segments: "
            f"{len(segments)}"
        )

        print(
            f"Avg logprob: "
            f"{avg_logprob}"
        )

        print(
            f"Avg compression ratio: "
            f"{avg_compression_ratio}"
        )

        print(
            f"Avg no-speech probability: "
            f"{avg_no_speech_prob}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    print()
    print("[4/4] Saving results...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("ROUND 10A COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()