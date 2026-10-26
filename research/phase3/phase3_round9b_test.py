import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from faster_whisper import WhisperModel


# ============================================================
# ROUND 9B
# Levantine Arabic ASR Model Validation
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_PATH = PROJECT_DIR / "meeting_processed.wav"
GROUND_TRUTH_PATH = PROJECT_DIR / "round8_ground_truth.json"

OUTPUT_DIR = PROJECT_DIR / "round9b_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "phase3_round9b_results.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "HebArabNlpProject/WhisperLevantine"

DEVICE = "cpu"

# CPU-only project
COMPUTE_TYPE = "int8"

LANGUAGE = "ar"
TASK = "transcribe"


# ============================================================
# Decoding
# ============================================================

# We use beam_size=5 because Round 9A showed that beam search
# was the best tested strategy for our baseline large-v3.
BEAM_SIZE = 5

TEMPERATURE = 0.0

CONDITION_ON_PREVIOUS_TEXT = True

WORD_TIMESTAMPS = False

VAD_FILTER = False


# ============================================================
# Ground Truth
# ============================================================

# These values are read from round8_ground_truth.json.
# They are intentionally not hard-coded here.
# This prevents sample timing mistakes.
# ============================================================


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text,
    )

    # Remove tatweel
    text = text.replace("ـ", "")

    # Normalize Arabic character variants
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# WER
# ============================================================

def levenshtein_words(
    reference_words,
    hypothesis_words,
):

    n = len(reference_words)
    m = len(hypothesis_words)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    operations = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):

        dp[i][0] = i
        operations[i][0] = "D"

    for j in range(1, m + 1):

        dp[0][j] = j
        operations[0][j] = "I"

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if (
                reference_words[i - 1]
                == hypothesis_words[j - 1]
            ):

                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = "C"

            else:

                substitution = (
                    dp[i - 1][j - 1] + 1
                )

                deletion = (
                    dp[i - 1][j] + 1
                )

                insertion = (
                    dp[i][j - 1] + 1
                )

                best = min(
                    substitution,
                    deletion,
                    insertion,
                )

                dp[i][j] = best

                if best == substitution:

                    operations[i][j] = "S"

                elif best == deletion:

                    operations[i][j] = "D"

                else:

                    operations[i][j] = "I"

    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        operation = operations[i][j]

        if operation == "C":

            i -= 1
            j -= 1

        elif operation == "S":

            substitutions += 1
            i -= 1
            j -= 1

        elif operation == "D":

            deletions += 1
            i -= 1

        elif operation == "I":

            insertions += 1
            j -= 1

        else:

            break

    total_errors = (
        substitutions
        + deletions
        + insertions
    )

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
    }


def calculate_wer(
    reference,
    hypothesis,
):

    reference_normalized = normalize_text(
        reference
    )

    hypothesis_normalized = normalize_text(
        hypothesis
    )

    reference_words = (
        reference_normalized.split()
    )

    hypothesis_words = (
        hypothesis_normalized.split()
    )

    errors = levenshtein_words(
        reference_words,
        hypothesis_words,
    )

    reference_word_count = len(
        reference_words
    )

    if reference_word_count == 0:

        wer = None

    else:

        wer = (
            errors["total_errors"]
            / reference_word_count
        )

    return {
        "reference_normalized":
            reference_normalized,

        "hypothesis_normalized":
            hypothesis_normalized,

        "reference_word_count":
            reference_word_count,

        "hypothesis_word_count":
            len(hypothesis_words),

        "wer":
            wer,

        **errors,
    }


# ============================================================
# Ground Truth
# ============================================================

def load_ground_truth():

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground Truth not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return data


# ============================================================
# Audio extraction
# ============================================================

def extract_audio_segment(
    audio,
    sample_rate,
    start_seconds,
    duration_seconds,
):

    start_frame = int(
        start_seconds * sample_rate
    )

    end_frame = int(
        (start_seconds + duration_seconds)
        * sample_rate
    )

    segment = audio[
        start_frame:end_frame
    ]

    if len(segment) == 0:

        raise ValueError(
            "Extracted audio segment is empty."
        )

    return segment.astype(
        np.float32
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 9B — LEVANTINE ASR MODEL VALIDATION")
    print("=" * 70)

    print()

    print("Model:")
    print(MODEL_NAME)

    print("Device:")
    print(DEVICE)

    print("Compute type:")
    print(COMPUTE_TYPE)

    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(
            f"Audio not found:\n{AUDIO_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground Truth not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("Loading audio...")

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
    )

    if sample_rate != 16000:

        raise ValueError(
            f"Expected 16000 Hz audio, "
            f"got {sample_rate} Hz"
        )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        ).astype(np.float32)

    print(
        f"Audio duration: "
        f"{len(audio) / sample_rate:.2f} sec"
    )

    print()

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading Levantine model...")
    print()
    print(
        "The first run may download the model "
        "from Hugging Face."
    )
    print()

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print()
    print("Model loaded successfully.")
    print()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "experiment":
            "Round 9B",

        "description":
            (
                "Validation of a Levantine "
                "Arabic fine-tuned Whisper model "
                "against the SWBS Jordanian "
                "Arabic Ground Truth."
            ),

        "model":
            MODEL_NAME,

        "device":
            DEVICE,

        "compute_type":
            COMPUTE_TYPE,

        "audio": {
            "path":
                AUDIO_PATH.name,

            "sample_rate":
                sample_rate,
        },

        "decoding": {

            "language":
                LANGUAGE,

            "task":
                TASK,

            "beam_size":
                BEAM_SIZE,

            "temperature":
                TEMPERATURE,

            "condition_on_previous_text":
                CONDITION_ON_PREVIOUS_TEXT,

            "word_timestamps":
                WORD_TIMESTAMPS,

            "vad_filter":
                VAD_FILTER,
        },

        "samples": {},

    }

    # --------------------------------------------------------
    # Run samples
    # --------------------------------------------------------

    sample_names = [
        "sample_00",
        "sample_20",
        "sample_30",
    ]

    for index, sample_name in enumerate(
        sample_names,
        start=1,
    ):

        print("=" * 70)

        print(
            f"[{index}/3] {sample_name}"
        )

        print("=" * 70)

        sample_info = ground_truth.get(
            sample_name
        )

        if sample_info is None:

            raise KeyError(
                f"{sample_name} not found "
                f"in Ground Truth."
            )

        start_seconds = float(
            sample_info["start_seconds"]
        )

        duration_seconds = float(
            sample_info["duration_seconds"]
        )

        reference = sample_info[
            "reference_text"
        ]

        print(
            f"Start: {start_seconds}s"
        )

        print(
            f"Duration: {duration_seconds}s"
        )

        # ----------------------------------------------------
        # Extract audio
        # ----------------------------------------------------

        audio_segment = (
            extract_audio_segment(
                audio=audio,
                sample_rate=sample_rate,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
        )

        # ----------------------------------------------------
        # Transcribe
        # ----------------------------------------------------

        print()
        print("Transcribing...")

        segments, info = model.transcribe(

            audio_segment,

            language=LANGUAGE,

            task=TASK,

            beam_size=BEAM_SIZE,

            temperature=TEMPERATURE,

            condition_on_previous_text=(
                CONDITION_ON_PREVIOUS_TEXT
            ),

            word_timestamps=WORD_TIMESTAMPS,

            vad_filter=VAD_FILTER,
        )

        # Important:
        # faster-whisper returns a generator.
        # Converting to list forces actual inference.
        segments = list(segments)

        hypothesis = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_wer(
            reference=reference,
            hypothesis=hypothesis,
        )

        # ----------------------------------------------------
        # Segment statistics
        # ----------------------------------------------------

        segment_records = []

        for segment in segments:

            segment_records.append({

                "start":
                    segment.start,

                "end":
                    segment.end,

                "text":
                    segment.text.strip(),

            })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        print("REFERENCE:")
        print(reference)

        print()

        print("HYPOTHESIS:")
        print(hypothesis)

        print()

        print(
            f"WER: "
            f"{metrics['wer']:.4f}"
        )

        print(
            f"Errors: "
            f"{metrics['total_errors']}"
        )

        print(
            f"Substitutions: "
            f"{metrics['substitutions']}"
        )

        print(
            f"Deletions: "
            f"{metrics['deletions']}"
        )

        print(
            f"Insertions: "
            f"{metrics['insertions']}"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results["samples"][
            sample_name
        ] = {

            "sample": {

                "start_seconds":
                    start_seconds,

                "duration_seconds":
                    duration_seconds,
            },

            "reference":
                reference,

            "hypothesis":
                hypothesis,

            "metrics":
                metrics,

            "model_info": {

                "language":
                    getattr(
                        info,
                        "language",
                        None,
                    ),

                "language_probability":
                    getattr(
                        info,
                        "language_probability",
                        None,
                    ),

                "duration":
                    getattr(
                        info,
                        "duration",
                        None,
                    ),

                "duration_after_vad":
                    getattr(
                        info,
                        "duration_after_vad",
                        None,
                    ),
            },

            "segments":
                segment_records,
        }

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    total_reference_words = 0
    total_errors = 0

    sample_wers = []

    for sample_name, sample_result in (
        results["samples"].items()
    ):

        metrics = sample_result[
            "metrics"
        ]

        total_reference_words += (
            metrics[
                "reference_word_count"
            ]
        )

        total_errors += (
            metrics[
                "total_errors"
            ]
        )

        sample_wers.append(
            metrics["wer"]
        )

    corpus_wer = (
        total_errors
        / total_reference_words
    )

    average_sample_wer = float(
        np.mean(sample_wers)
    )

    results["summary"] = {

        "average_sample_wer":
            average_sample_wer,

        "corpus_wer":
            corpus_wer,

        "total_reference_words":
            total_reference_words,

        "total_errors":
            total_errors,

    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROUND 9B COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Average sample WER: "
        f"{average_sample_wer:.4f}"
    )

    print(
        f"Corpus WER: "
        f"{corpus_wer:.4f}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()