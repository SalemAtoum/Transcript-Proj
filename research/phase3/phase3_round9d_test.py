
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions


# ============================================================
# ROUND 9D — CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_PATH = PROJECT_DIR / "meeting_processed.wav"
GROUND_TRUTH_PATH = PROJECT_DIR / "round8_ground_truth.json"

OUTPUT_DIR = PROJECT_DIR / "round9d_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "phase3_round9d_results.json"

MODEL_NAME = "large-v3"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"

LANGUAGE = "ar"
TASK = "transcribe"

BEAM_SIZE = 5
TEMPERATURE = 0.0

CONDITION_ON_PREVIOUS_TEXT = True

# Fixed VAD configuration
VAD_OPTIONS = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)

EXPERIMENTS = {
    "A_baseline": False,
    "B_vad": True,
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_arabic(text):

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    # Remove Tatweel
    text = text.replace("ـ", "")

    # Normalize Arabic letters
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# WORD ERROR RATE
# ============================================================

def calculate_wer(reference, hypothesis):

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    n = len(ref_words)
    m = len(hyp_words)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            cost = (
                0
                if ref_words[i - 1] == hyp_words[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i = n
    j = m

    substitutions = 0
    insertions = 0
    deletions = 0

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and ref_words[i - 1] == hyp_words[j - 1]
        ):
            i -= 1
            j -= 1
            continue

        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            substitutions += 1
            i -= 1
            j -= 1
            continue

        if (
            i > 0
            and dp[i][j] == dp[i - 1][j] + 1
        ):
            deletions += 1
            i -= 1
            continue

        if (
            j > 0
            and dp[i][j] == dp[i][j - 1] + 1
        ):
            insertions += 1
            j -= 1
            continue

        raise RuntimeError(
            "WER backtracking failed."
        )

    total_errors = (
        substitutions
        + insertions
        + deletions
    )

    wer = (
        total_errors / n
        if n > 0
        else (0.0 if m == 0 else 1.0)
    )

    return {
        "reference_word_count": n,
        "hypothesis_word_count": m,
        "wer": wer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "total_errors": total_errors,
    }


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    required_samples = [
        "sample_00",
        "sample_20",
        "sample_30",
    ]

    samples = {}

    for name in required_samples:

        if name not in data:
            raise ValueError(
                f"Missing sample: {name}"
            )

        item = data[name]

        samples[name] = {
            "start_seconds": float(
                item["start_seconds"]
            ),
            "duration_seconds": float(
                item["duration_seconds"]
            ),
            "reference_text": item["reference_text"],
        }

        if not samples[name]["reference_text"].strip():
            raise ValueError(
                f"Empty reference: {name}"
            )

    return samples


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_sample(
    audio,
    sample_rate,
    start_seconds,
    duration_seconds,
):

    start = round(
        start_seconds * sample_rate
    )

    end = round(
        (start_seconds + duration_seconds)
        * sample_rate
    )

    if start < 0 or end > len(audio):
        raise ValueError(
            "Sample is outside audio boundaries."
        )

    sample = audio[start:end]

    if len(sample) == 0:
        raise ValueError(
            "Extracted sample is empty."
        )

    return np.asarray(
        sample,
        dtype=np.float32,
    )


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_sample(
    model,
    audio,
    vad_enabled,
):

    kwargs = {
        "language": LANGUAGE,
        "task": TASK,
        "beam_size": BEAM_SIZE,
        "temperature": TEMPERATURE,
        "condition_on_previous_text":
            CONDITION_ON_PREVIOUS_TEXT,
        "word_timestamps": False,
        "vad_filter": vad_enabled,
    }

    if vad_enabled:
        kwargs["vad_parameters"] = VAD_OPTIONS

    segments_generator, info = model.transcribe(
        audio,
        **kwargs,
    )

    segments = []

    texts = []

    for segment in segments_generator:

        text = segment.text.strip()

        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })

        texts.append(text)

    hypothesis = " ".join(texts).strip()

    return {
        "hypothesis": hypothesis,
        "segments": segments,
        "model_info": {
            "language": info.language,
            "language_probability":
                info.language_probability,
            "duration": info.duration,
            "duration_after_vad":
                info.duration_after_vad,
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3 — ROUND 9D")
    print("VAD-Controlled Transcription")
    print("=" * 65)

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Missing audio: {AUDIO_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Missing ground truth: {GROUND_TRUTH_PATH}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("\nLoading audio...")

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
    )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz, got {sample_rate}"
        )

    if audio.ndim != 1:
        raise ValueError(
            "Audio must be mono."
        )

    print(
        f"Audio duration: "
        f"{len(audio) / sample_rate:.2f}s"
    )

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    print("\nGround Truth loaded.")

    for name, item in ground_truth.items():

        print(
            f"{name}: "
            f"{item['start_seconds']}s "
            f"for {item['duration_seconds']}s"
        )

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    print("\nLoading Whisper model...")

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Results structure
    # --------------------------------------------------------

    results = {
        "experiment": "Round 9D",
        "description":
            "VAD OFF vs VAD ON comparison",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "decoding": {
            "language": LANGUAGE,
            "task": TASK,
            "beam_size": BEAM_SIZE,
            "temperature": TEMPERATURE,
            "condition_on_previous_text":
                CONDITION_ON_PREVIOUS_TEXT,
        },
        "vad_options": {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        "experiments": {},
    }

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for experiment_name, vad_enabled in (
        EXPERIMENTS.items()
    ):

        print("\n" + "=" * 65)

        print(
            f"EXPERIMENT: {experiment_name}"
        )

        print(
            f"VAD: {'ON' if vad_enabled else 'OFF'}"
        )

        print("=" * 65)

        experiment_results = {
            "vad_enabled": vad_enabled,
            "samples": {},
        }

        total_errors = 0
        total_reference_words = 0

        sample_wers = []

        for sample_name, item in (
            ground_truth.items()
        ):

            print(
                f"\nRunning {sample_name}..."
            )

            sample_audio = extract_sample(
                audio,
                sample_rate,
                item["start_seconds"],
                item["duration_seconds"],
            )

            transcription = transcribe_sample(
                model,
                sample_audio,
                vad_enabled,
            )

            reference = item["reference_text"]

            hypothesis = transcription[
                "hypothesis"
            ]

            reference_normalized = (
                normalize_arabic(reference)
            )

            hypothesis_normalized = (
                normalize_arabic(hypothesis)
            )

            metrics = calculate_wer(
                reference_normalized,
                hypothesis_normalized,
            )

            total_errors += metrics[
                "total_errors"
            ]

            total_reference_words += metrics[
                "reference_word_count"
            ]

            sample_wers.append(
                metrics["wer"]
            )

            experiment_results[
                "samples"
            ][sample_name] = {
                "start_seconds":
                    item["start_seconds"],
                "duration_seconds":
                    item["duration_seconds"],
                "reference": reference,
                "hypothesis": hypothesis,
                "reference_normalized":
                    reference_normalized,
                "hypothesis_normalized":
                    hypothesis_normalized,
                "metrics": metrics,
                "model_info":
                    transcription["model_info"],
                "segments":
                    transcription["segments"],
            }

            print("\nREFERENCE:")
            print(reference)

            print("\nHYPOTHESIS:")
            print(hypothesis)

            print(
                f"\nWER: "
                f"{metrics['wer'] * 100:.2f}%"
            )

            print(
                "Errors: "
                f"S={metrics['substitutions']} "
                f"I={metrics['insertions']} "
                f"D={metrics['deletions']}"
            )

        corpus_wer = (
            total_errors / total_reference_words
            if total_reference_words
            else 0.0
        )

        average_wer = (
            sum(sample_wers) / len(sample_wers)
        )

        experiment_results["summary"] = {
            "corpus_wer": corpus_wer,
            "average_sample_wer":
                average_wer,
            "total_reference_words":
                total_reference_words,
            "total_errors":
                total_errors,
        }

        results["experiments"][
            experiment_name
        ] = experiment_results

        print("\n" + "-" * 65)

        print(
            f"{experiment_name} "
            f"CORPUS WER: "
            f"{corpus_wer * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    baseline = results[
        "experiments"
    ]["A_baseline"]["summary"]["corpus_wer"]

    vad = results[
        "experiments"
    ]["B_vad"]["summary"]["corpus_wer"]

    improvement = (
        baseline - vad
    ) * 100

    if vad < baseline:
        verdict = "VAD_IMPROVED"

    elif vad > baseline:
        verdict = "VAD_WORSE"

    else:
        verdict = "NO_CHANGE"

    results["comparison"] = {
        "baseline_wer": baseline,
        "vad_wer": vad,
        "improvement_percentage_points":
            improvement,
        "verdict": verdict,
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
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
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ROUND 9D FINAL COMPARISON")
    print("=" * 65)

    print(
        f"Baseline WER: "
        f"{baseline * 100:.2f}%"
    )

    print(
        f"VAD WER: "
        f"{vad * 100:.2f}%"
    )

    print(
        f"Improvement: "
        f"{improvement:+.2f} "
        "percentage points"
    )

    print(
        f"Verdict: {verdict}"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
