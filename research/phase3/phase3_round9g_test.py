import json
import re
import unicodedata
from pathlib import Path

import soundfile as sf
import whisper
from faster_whisper import WhisperModel


# ============================================================
# ROUND 9G
# OpenAI Whisper vs Faster-Whisper
# Controlled ASR Engine Comparison
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_FILE = BASE_DIR / "meeting_processed.wav"
GROUND_TRUTH_FILE = BASE_DIR / "round8_ground_truth.json"

OUTPUT_DIR = BASE_DIR / "round9g_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "phase3_round9g_results.json"


# ============================================================
# Common configuration
# ============================================================

MODEL_NAME = "large-v3"

LANGUAGE = "ar"
TASK = "transcribe"

BEAM_SIZE = 5
TEMPERATURE = 0.0

CONDITION_ON_PREVIOUS_TEXT = True

COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    # Remove tatweel
    text = text.replace("ـ", "")

    # Normalize Arabic letters
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text,
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

def calculate_wer(reference: str, hypothesis: str):

    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            substitution_cost = (
                0
                if ref_words[i - 1] == hyp_words[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    # --------------------------------------------------------
    # Recover edit operations
    # --------------------------------------------------------

    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        if i > 0 and j > 0:

            substitution_cost = (
                0
                if ref_words[i - 1] == hyp_words[j - 1]
                else 1
            )

            if dp[i][j] == (
                dp[i - 1][j - 1] + substitution_cost
            ):

                if substitution_cost == 1:
                    substitutions += 1

                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:

            deletions += 1
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:

            insertions += 1
            j -= 1
            continue

        break

    total_errors = (
        substitutions
        + deletions
        + insertions
    )

    wer = (
        total_errors / n
        if n
        else None
    )

    return {
        "reference_word_count": n,
        "hypothesis_word_count": m,
        "WER": wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
    }


# ============================================================
# Ground Truth
# ============================================================

def load_ground_truth():

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    required_samples = [
        "sample_00",
        "sample_20",
        "sample_30",
    ]

    for sample_name in required_samples:

        if sample_name not in data:
            raise ValueError(
                f"Missing {sample_name}"
            )

        sample = data[sample_name]

        required_fields = [
            "start_seconds",
            "duration_seconds",
            "reference_text",
        ]

        for field in required_fields:

            if field not in sample:
                raise ValueError(
                    f"Missing {field} "
                    f"for {sample_name}"
                )

    return data


# ============================================================
# Audio extraction
# ============================================================

def extract_sample(audio, sample_info):

    start_seconds = int(
        sample_info["start_seconds"]
    )

    duration_seconds = int(
        sample_info["duration_seconds"]
    )

    start_frame = int(
        start_seconds * 16000
    )

    end_frame = int(
        (start_seconds + duration_seconds)
        * 16000
    )

    return audio[start_frame:end_frame]


# ============================================================
# OpenAI Whisper
# ============================================================

def run_openai_whisper(
    model,
    sample_audio,
):

    result = model.transcribe(
        sample_audio,
        language=LANGUAGE,
        task=TASK,
        fp16=False,
        temperature=TEMPERATURE,
        condition_on_previous_text=(
            CONDITION_ON_PREVIOUS_TEXT
        ),
        compression_ratio_threshold=(
            COMPRESSION_RATIO_THRESHOLD
        ),
        logprob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        beam_size=BEAM_SIZE,
        initial_prompt=None,
        verbose=False,
    )

    segments = result.get(
        "segments",
        [],
    )

    return {
        "text": result["text"].strip(),
        "segments": segments,
    }


# ============================================================
# Faster-Whisper
# ============================================================

def run_faster_whisper(
    model,
    sample_audio,
):

    segments_generator, info = model.transcribe(
        sample_audio,
        language=LANGUAGE,
        task=TASK,
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURE,
        condition_on_previous_text=(
            CONDITION_ON_PREVIOUS_TEXT
        ),
        compression_ratio_threshold=(
            COMPRESSION_RATIO_THRESHOLD
        ),
        log_prob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        initial_prompt=None,
        word_timestamps=False,
        vad_filter=False,
    )

    segments = list(segments_generator)

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return {
        "text": text,
        "segments": segments,
    }


# ============================================================
# Segment metrics
# ============================================================

def calculate_segment_metrics(
    segments,
):

    if not segments:

        return {
            "segment_count": 0,
            "avg_logprob": None,
            "avg_compression_ratio": None,
            "avg_no_speech_prob": None,
        }

    logprobs = []
    compression_ratios = []
    no_speech_probs = []

    for segment in segments:

        # OpenAI Whisper objects are dictionaries
        if isinstance(segment, dict):

            avg_logprob = segment.get(
                "avg_logprob"
            )

            compression_ratio = segment.get(
                "compression_ratio"
            )

            no_speech_prob = segment.get(
                "no_speech_prob"
            )

        # Faster-Whisper Segment objects
        else:

            avg_logprob = getattr(
                segment,
                "avg_logprob",
                None,
            )

            compression_ratio = getattr(
                segment,
                "compression_ratio",
                None,
            )

            no_speech_prob = getattr(
                segment,
                "no_speech_prob",
                None,
            )

        if avg_logprob is not None:
            logprobs.append(avg_logprob)

        if compression_ratio is not None:
            compression_ratios.append(
                compression_ratio
            )

        if no_speech_prob is not None:
            no_speech_probs.append(
                no_speech_prob
            )

    return {
        "segment_count": len(segments),

        "avg_logprob": (
            sum(logprobs) / len(logprobs)
            if logprobs
            else None
        ),

        "avg_compression_ratio": (
            sum(compression_ratios)
            / len(compression_ratios)
            if compression_ratios
            else None
        ),

        "avg_no_speech_prob": (
            sum(no_speech_probs)
            / len(no_speech_probs)
            if no_speech_probs
            else None
        ),
    }


# ============================================================
# Evaluate engine
# ============================================================

def evaluate_engine(
    engine_name,
    engine_model,
    ground_truth,
    audio,
    runner,
):

    print()
    print("=" * 70)
    print(f"ENGINE: {engine_name}")
    print("=" * 70)

    engine_result = {
        "engine": engine_name,
        "samples": {},
    }

    for sample_name in [
        "sample_00",
        "sample_20",
        "sample_30",
    ]:

        sample_info = ground_truth[
            sample_name
        ]

        print()
        print(
            f"Running {sample_name}..."
        )

        sample_audio = extract_sample(
            audio,
            sample_info,
        )

        result = runner(
            engine_model,
            sample_audio,
        )

        hypothesis = result["text"]

        reference = sample_info[
            "reference_text"
        ]

        metrics = calculate_wer(
            reference,
            hypothesis,
        )

        segment_metrics = (
            calculate_segment_metrics(
                result["segments"]
            )
        )

        engine_result["samples"][
            sample_name
        ] = {
            "start_seconds": sample_info[
                "start_seconds"
            ],
            "duration_seconds": sample_info[
                "duration_seconds"
            ],
            "reference": reference,
            "hypothesis": hypothesis,
            "normalized_reference": (
                normalize_text(reference)
            ),
            "normalized_hypothesis": (
                normalize_text(hypothesis)
            ),
            "metrics": metrics,
            **segment_metrics,
        }

        print()
        print("REFERENCE:")
        print(reference)

        print()
        print("HYPOTHESIS:")
        print(hypothesis)

        print()
        print(
            f"WER: "
            f"{metrics['WER'] * 100:.2f}%"
        )

        print(
            f"Errors: "
            f"{metrics['total_errors']}"
        )

    # --------------------------------------------------------
    # Corpus summary
    # --------------------------------------------------------

    total_reference_words = 0
    total_errors = 0

    sample_wers = []

    for sample in engine_result[
        "samples"
    ].values():

        metrics = sample[
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
            metrics["WER"]
        )

    corpus_wer = (
        total_errors
        / total_reference_words
        if total_reference_words
        else None
    )

    average_sample_wer = (
        sum(sample_wers)
        / len(sample_wers)
        if sample_wers
        else None
    )

    engine_result["summary"] = {
        "corpus_WER": corpus_wer,
        "average_sample_WER": (
            average_sample_wer
        ),
        "reference_word_count": (
            total_reference_words
        ),
        "total_errors": total_errors,
    }

    print()
    print("-" * 70)
    print(f"{engine_name} SUMMARY")
    print("-" * 70)

    print(
        f"Corpus WER: "
        f"{corpus_wer * 100:.2f}%"
    )

    print(
        f"Average sample WER: "
        f"{average_sample_wer * 100:.2f}%"
    )

    print(
        f"Total errors: "
        f"{total_errors}"
    )

    return engine_result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 9G")
    print("OpenAI Whisper vs Faster-Whisper")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_FILE}"
        )

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            "Ground truth file not found:\n"
            f"{GROUND_TRUTH_FILE}"
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print()
    print("[1/6] Loading Ground Truth...")

    ground_truth = load_ground_truth()

    print("Ground Truth loaded.")

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    print()
    print("[2/6] Loading audio...")

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
        always_2d=False,
    )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz, "
            f"got {sample_rate} Hz"
        )

    print(
        f"Sample rate: {sample_rate} Hz"
    )

    print(
        f"Audio samples: {len(audio)}"
    )

    # --------------------------------------------------------
    # OpenAI Whisper
    # --------------------------------------------------------

    print()
    print(
        "[3/6] Loading OpenAI Whisper..."
    )

    openai_model = whisper.load_model(
        MODEL_NAME,
        device="cpu",
    )

    print(
        "OpenAI Whisper loaded."
    )

    # --------------------------------------------------------
    # Faster-Whisper
    # --------------------------------------------------------

    print()
    print(
        "[4/6] Loading Faster-Whisper..."
    )

    faster_model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
    )

    print(
        "Faster-Whisper loaded."
    )

    # --------------------------------------------------------
    # Run OpenAI Whisper
    # --------------------------------------------------------

    print()
    print(
        "[5/6] Running controlled comparison..."
    )

    openai_results = evaluate_engine(
        "openai_whisper",
        openai_model,
        ground_truth,
        audio,
        run_openai_whisper,
    )

    faster_results = evaluate_engine(
        "faster_whisper",
        faster_model,
        ground_truth,
        audio,
        run_faster_whisper,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    openai_wer = openai_results[
        "summary"
    ]["corpus_WER"]

    faster_wer = faster_results[
        "summary"
    ]["corpus_WER"]

    if faster_wer < openai_wer:

        winner = "faster_whisper"

        improvement = (
            (
                openai_wer
                - faster_wer
            )
            / openai_wer
            * 100
        )

    else:

        winner = "openai_whisper"

        improvement = (
            (
                faster_wer
                - openai_wer
            )
            / faster_wer
            * 100
        )

    results = {
        "experiment": "Round 9G",

        "description": (
            "Controlled comparison of "
            "OpenAI Whisper and Faster-Whisper "
            "using the same model and decoding "
            "configuration."
        ),

        "common_configuration": {
            "model": MODEL_NAME,
            "language": LANGUAGE,
            "task": TASK,
            "beam_size": BEAM_SIZE,
            "temperature": TEMPERATURE,
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
            "initial_prompt": None,
            "audio_sample_rate": 16000,
        },

        "engines": {
            "openai_whisper": openai_results,
            "faster_whisper": faster_results,
        },

        "comparison": {
            "openai_corpus_WER": openai_wer,
            "faster_corpus_WER": faster_wer,
            "winner": winner,
            "relative_improvement_percent": (
                improvement
            ),
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print(
        "[6/6] Saving results..."
    )

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

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROUND 9G COMPLETE")
    print("=" * 70)

    print()

    print(
        f"OpenAI Whisper WER: "
        f"{openai_wer * 100:.2f}%"
    )

    print(
        f"Faster-Whisper WER: "
        f"{faster_wer * 100:.2f}%"
    )

    print(
        f"Winner: {winner}"
    )

    print(
        f"Relative difference: "
        f"{improvement:.2f}%"
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()