import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper


# ============================================================
# ROUND 9A
# Whisper Decoding Strategy Experiment
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_PATH = PROJECT_DIR / "meeting_processed.wav"
GROUND_TRUTH_PATH = PROJECT_DIR / "round8_ground_truth.json"

OUTPUT_DIR = PROJECT_DIR / "round9a_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "phase3_round9a_results.json"


# ============================================================
# Whisper configuration
# ============================================================

MODEL_NAME = "large-v3"
DEVICE = "cpu"

LANGUAGE = "ar"
TASK = "transcribe"

FP16 = False

CONDITION_ON_PREVIOUS_TEXT = True

COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


# ============================================================
# Samples
# Same samples used in Round 8
# ============================================================

SAMPLES = {
    "sample_00": {
        "start_sec": 0,
        "duration_sec": 20,
    },
    "sample_20": {
        "start_sec": 1220,
        "duration_sec": 20,
    },
    "sample_30": {
        "start_sec": 1814,
        "duration_sec": 20,
    },
}


# ============================================================
# Decoding strategies
# ============================================================

STRATEGIES = {
    "A_baseline": {
        "temperature": 0,
        "beam_size": None,
        "best_of": None,
    },

    "B_beam_search": {
        "temperature": 0,
        "beam_size": 5,
        "best_of": None,
    },

    "C_temperature_fallback": {
        "temperature": (
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ),
        "beam_size": None,
        "best_of": None,
    },
}


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Normalize Arabic/Latin text for WER comparison.

    Important:
    This does NOT attempt to rewrite the dialect.
    It only removes orthographic noise.
    """

    text = unicodedata.normalize("NFKC", text)

    # Lowercase Latin
    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Remove tatweel
    text = text.replace("ـ", "")

    # Normalize common Arabic character variants
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
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Word-level WER
# ============================================================

def levenshtein_words(reference_words, hypothesis_words):
    """
    Returns:
        substitutions
        deletions
        insertions
        total_errors
    """

    n = len(reference_words)
    m = len(hypothesis_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    operations = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        operations[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        operations[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if reference_words[i - 1] == hypothesis_words[j - 1]:

                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = "C"

            else:

                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1

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

        op = operations[i][j]

        if op == "C":
            i -= 1
            j -= 1

        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1

        elif op == "D":
            deletions += 1
            i -= 1

        elif op == "I":
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


def calculate_wer(reference, hypothesis):

    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)

    reference_words = reference_normalized.split()
    hypothesis_words = hypothesis_normalized.split()

    errors = levenshtein_words(
        reference_words,
        hypothesis_words,
    )

    reference_count = len(reference_words)

    if reference_count == 0:
        wer = None
    else:
        wer = errors["total_errors"] / reference_count

    return {
        "reference_normalized": reference_normalized,
        "hypothesis_normalized": hypothesis_normalized,
        "reference_word_count": reference_count,
        "hypothesis_word_count": len(hypothesis_words),
        "wer": wer,
        **errors,
    }


# ============================================================
# Ground Truth loader
# ============================================================

def load_ground_truth():

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground Truth file not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return data


def extract_reference(ground_truth_data, sample_name):

    """
    Supports common structures from the Round 8 ground-truth file.
    """

    sample = ground_truth_data.get(sample_name)

    if sample is None:
        raise KeyError(
            f"Could not find '{sample_name}' "
            f"in round8_ground_truth.json"
        )

    if isinstance(sample, str):
        return sample

    if isinstance(sample, dict):

        for key in [
    "reference_text",
    "reference",
    "ground_truth",
    "text",
    "transcript",
]:

            if key in sample:
                return sample[key]

    raise ValueError(
        f"Could not determine reference text "
        f"for {sample_name}"
    )


# ============================================================
# Audio extraction
# ============================================================

def extract_audio_segment(
    audio,
    sample_name,
    start_sec,
    duration_sec,
):

    sample_rate = 16000

    start_frame = int(start_sec * sample_rate)
    end_frame = int(
        (start_sec + duration_sec)
        * sample_rate
    )

    segment = audio[
        start_frame:end_frame
    ]

    if len(segment) == 0:

        raise ValueError(
            f"No audio extracted for {sample_name}"
        )

    return segment.astype(np.float32)


# ============================================================
# Whisper transcription
# ============================================================

def transcribe_sample(
    model,
    audio_segment,
    strategy_config,
):

    result = model.transcribe(
        audio_segment,

        language=LANGUAGE,
        task=TASK,

        fp16=FP16,

        temperature=strategy_config[
            "temperature"
        ],

        beam_size=strategy_config[
            "beam_size"
        ],

        best_of=strategy_config[
            "best_of"
        ],

        condition_on_previous_text=(
            CONDITION_ON_PREVIOUS_TEXT
        ),

        compression_ratio_threshold=(
            COMPRESSION_RATIO_THRESHOLD
        ),

        logprob_threshold=(
            LOGPROB_THRESHOLD
        ),

        no_speech_threshold=(
            NO_SPEECH_THRESHOLD
        ),

        verbose=False,
    )

    return result


# ============================================================
# Segment statistics
# ============================================================

def collect_segment_statistics(result):

    segments = result.get("segments", [])

    if not segments:

        return {
            "segment_count": 0,
            "average_logprob": None,
            "average_no_speech_prob": None,
            "average_compression_ratio": None,
            "segments": [],
        }

    logprobs = []
    no_speech_probs = []
    compression_ratios = []

    clean_segments = []

    for segment in segments:

        avg_logprob = segment.get(
            "avg_logprob"
        )

        no_speech_prob = segment.get(
            "no_speech_prob"
        )

        compression_ratio = segment.get(
            "compression_ratio"
        )

        if avg_logprob is not None:
            logprobs.append(avg_logprob)

        if no_speech_prob is not None:
            no_speech_probs.append(
                no_speech_prob
            )

        if compression_ratio is not None:
            compression_ratios.append(
                compression_ratio
            )

        clean_segments.append({
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": segment.get("text", "").strip(),
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio,
        })

    return {
        "segment_count": len(segments),

        "average_logprob": (
            float(np.mean(logprobs))
            if logprobs else None
        ),

        "average_no_speech_prob": (
            float(np.mean(no_speech_probs))
            if no_speech_probs else None
        ),

        "average_compression_ratio": (
            float(np.mean(compression_ratios))
            if compression_ratios else None
        ),

        "segments": clean_segments,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 9A — WHISPER DECODING STRATEGY EXPERIMENT")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    print("Python / Torch")
    print("-" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)

    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground Truth file not found:\n"
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
            f"Expected 16kHz audio, "
            f"got {sample_rate}Hz"
        )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        ).astype(np.float32)

    print(
        f"Audio loaded: "
        f"{len(audio) / sample_rate:.2f} sec"
    )

    print()

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    ground_truth_data = load_ground_truth()

    # --------------------------------------------------------
    # Load Whisper
    # --------------------------------------------------------

    print("Loading Whisper model...")
    print(f"Model: {MODEL_NAME}")
    print("This may take some time on CPU...")

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
    )

    print("Whisper model loaded.")

    print()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "experiment": "Round 9A",

        "description": (
            "Controlled comparison of Whisper "
            "large-v3 decoding strategies"
        ),

        "model": MODEL_NAME,

        "device": DEVICE,

        "audio": {
            "path": str(AUDIO_PATH.name),
            "sample_rate": sample_rate,
        },

        "common_config": {
            "language": LANGUAGE,
            "task": TASK,
            "fp16": FP16,
            "condition_on_previous_text": (
                CONDITION_ON_PREVIOUS_TEXT
            ),
            "compression_ratio_threshold": (
                COMPRESSION_RATIO_THRESHOLD
            ),
            "logprob_threshold": (
                LOGPROB_THRESHOLD
            ),
            "no_speech_threshold": (
                NO_SPEECH_THRESHOLD
            ),
        },

        "strategies": {},
    }

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    total_runs = (
        len(STRATEGIES)
        * len(SAMPLES)
    )

    current_run = 0

    for strategy_name, strategy_config in STRATEGIES.items():

        print()
        print("=" * 70)
        print(strategy_name)
        print("=" * 70)

        results["strategies"][
            strategy_name
        ] = {
            "config": {
                "temperature": str(
                    strategy_config["temperature"]
                ),
                "beam_size": strategy_config[
                    "beam_size"
                ],
                "best_of": strategy_config[
                    "best_of"
                ],
            },

            "samples": {},
        }

        for sample_name, sample_config in SAMPLES.items():

            current_run += 1

            print()
            print(
                f"[{current_run}/{total_runs}] "
                f"{strategy_name} → {sample_name}"
            )

            # --------------------------------------------
            # Extract sample
            # --------------------------------------------

            audio_segment = extract_audio_segment(
                audio=audio,
                sample_name=sample_name,
                start_sec=sample_config[
                    "start_sec"
                ],
                duration_sec=sample_config[
                    "duration_sec"
                ],
            )

            # --------------------------------------------
            # Ground Truth
            # --------------------------------------------

            reference = extract_reference(
                ground_truth_data,
                sample_name,
            )

            # --------------------------------------------
            # Whisper
            # --------------------------------------------

            print("Transcribing...")

            result = transcribe_sample(
                model=model,
                audio_segment=audio_segment,
                strategy_config=strategy_config,
            )

            hypothesis = result.get(
                "text",
                "",
            ).strip()

            print()
            print("REFERENCE:")
            print(reference)

            print()
            print("HYPOTHESIS:")
            print(hypothesis)

            # --------------------------------------------
            # WER
            # --------------------------------------------

            metrics = calculate_wer(
                reference=reference,
                hypothesis=hypothesis,
            )

            segment_stats = (
                collect_segment_statistics(
                    result
                )
            )

            print()
            print(
                f"WER: "
                f"{metrics['wer']:.4f}"
            )

            print(
                f"Segments: "
                f"{segment_stats['segment_count']}"
            )

            print(
                f"Avg logprob: "
                f"{segment_stats['average_logprob']}"
            )

            print(
                f"Avg no-speech prob: "
                f"{segment_stats['average_no_speech_prob']}"
            )

            print(
                f"Avg compression ratio: "
                f"{segment_stats['average_compression_ratio']}"
            )

            # --------------------------------------------
            # Save
            # --------------------------------------------

            results["strategies"][
                strategy_name
            ]["samples"][sample_name] = {

                "sample": {
                    "start_sec": sample_config[
                        "start_sec"
                    ],
                    "duration_sec": sample_config[
                        "duration_sec"
                    ],
                },

                "reference": reference,

                "hypothesis": hypothesis,

                "metrics": metrics,

                "segment_statistics": (
                    segment_stats
                ),
            }

    # --------------------------------------------------------
    # Calculate strategy summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STRATEGY SUMMARY")
    print("=" * 70)

    for strategy_name in STRATEGIES:

        sample_results = (
            results["strategies"][
                strategy_name
            ]["samples"]
        )

        wers = []

        total_reference_words = 0
        total_errors = 0

        for sample_name, sample_result in (
            sample_results.items()
        ):

            wer = sample_result[
                "metrics"
            ]["wer"]

            wers.append(wer)

            total_reference_words += (
                sample_result["metrics"][
                    "reference_word_count"
                ]
            )

            total_errors += (
                sample_result["metrics"][
                    "total_errors"
                ]
            )

        average_wer = (
            float(np.mean(wers))
            if wers else None
        )

        corpus_wer = (
            total_errors
            / total_reference_words
            if total_reference_words
            else None
        )

        results["strategies"][
            strategy_name
        ]["summary"] = {

            "average_sample_wer": (
                average_wer
            ),

            "corpus_wer": corpus_wer,

            "total_reference_words": (
                total_reference_words
            ),

            "total_errors": total_errors,
        }

        print()
        print(strategy_name)

        print(
            f"Average sample WER: "
            f"{average_wer:.4f}"
        )

        print(
            f"Corpus WER: "
            f"{corpus_wer:.4f}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
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
    print("ROUND 9A COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results saved to:\n{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()