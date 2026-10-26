import json
import re
import unicodedata
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel


# ============================================================
# ROUND 9C CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_PATH = PROJECT_DIR / "meeting_processed.wav"
GROUND_TRUTH_PATH = PROJECT_DIR / "round8_ground_truth.json"

OUTPUT_DIR = PROJECT_DIR / "round9c_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "phase3_round9c_results.json"


# Candidate model for Round 9C
MODEL_NAME = "dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"

LANGUAGE = "ar"
TASK = "transcribe"

# Keep decoding comparable with Round 9A
BEAM_SIZE = 5
TEMPERATURE = 0.0
CONDITION_ON_PREVIOUS_TEXT = True

WORD_TIMESTAMPS = False
VAD_FILTER = False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for WER comparison.

    This intentionally keeps the normalization conservative.
    """

    text = unicodedata.normalize("NFKC", text)

    # Lowercase Latin characters
    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)

    # Remove Tatweel
    text = text.replace("ـ", "")

    # Normalize common Arabic letter variants
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# WER IMPLEMENTATION
# ============================================================

def calculate_wer(reference: str, hypothesis: str):
    """
    Word-level Levenshtein distance.

    Returns:
        wer
        substitutions
        insertions
        deletions
        total_errors
    """

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    n = len(ref_words)
    m = len(hyp_words)

    # DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1

            substitution = dp[i - 1][j - 1] + cost
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1

            dp[i][j] = min(
                substitution,
                deletion,
                insertion
            )

    # Backtrack to count S/I/D
    i = n
    j = m

    substitutions = 0
    insertions = 0
    deletions = 0

    while i > 0 or j > 0:

        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
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

        raise RuntimeError("WER backtracking failed.")

    total_errors = substitutions + insertions + deletions

    if n == 0:
        wer = 0.0 if m == 0 else 1.0
    else:
        wer = total_errors / n

    return {
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
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n{GROUND_TRUTH_PATH}"
        )

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_samples = [
        "sample_00",
        "sample_20",
        "sample_30",
    ]

    samples = {}

    for sample_name in required_samples:

        if sample_name not in data:
            raise KeyError(
                f"Missing {sample_name} in ground truth."
            )

        item = data[sample_name]

        start_seconds = item.get("start_seconds")
        duration_seconds = item.get("duration_seconds")
        reference_text = item.get("reference_text")

        if start_seconds is None:
            raise ValueError(
                f"{sample_name}: missing start_seconds"
            )

        if duration_seconds is None:
            raise ValueError(
                f"{sample_name}: missing duration_seconds"
            )

        if not reference_text:
            raise ValueError(
                f"{sample_name}: missing reference_text"
            )

        samples[sample_name] = {
            "start_seconds": float(start_seconds),
            "duration_seconds": float(duration_seconds),
            "reference_text": reference_text,
        }

    return samples


# ============================================================
# EXTRACT AUDIO SAMPLE
# ============================================================

def extract_audio_sample(
    audio_data,
    sample_rate,
    start_seconds,
    duration_seconds,
):
    start_frame = int(start_seconds * sample_rate)
    end_frame = int(
        (start_seconds + duration_seconds) * sample_rate
    )

    if start_frame < 0:
        start_frame = 0

    if end_frame > len(audio_data):
        end_frame = len(audio_data)

    sample = audio_data[start_frame:end_frame]

    return sample


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 — ROUND 9C")
    print("Arabic Fine-Tuned Whisper Validation")
    print("=" * 70)

    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Compute type: {COMPUTE_TYPE}")
    print(f"Beam size: {BEAM_SIZE}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n{GROUND_TRUTH_PATH}"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("Loading audio...")

    audio_data, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32"
    )

    if audio_data.ndim != 1:
        raise ValueError(
            "Expected mono audio, but audio has multiple channels."
        )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, got {sample_rate} Hz."
        )

    print(f"Audio sample rate: {sample_rate} Hz")
    print(f"Audio duration: {len(audio_data) / sample_rate:.2f} sec")

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    print()
    print("Ground truth samples:")

    for name, item in ground_truth.items():
        print(
            f"  {name}: "
            f"{item['start_seconds']}s → "
            f"{item['start_seconds'] + item['duration_seconds']}s"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")
    print("This may take some time on CPU.")

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        "experiment": "Round 9C",

        "description": (
            "Validation of an Arabic fine-tuned Whisper model "
            "against the SWBS Jordanian Arabic Ground Truth."
        ),

        "model": MODEL_NAME,

        "device": DEVICE,

        "compute_type": COMPUTE_TYPE,

        "audio": {
            "path": AUDIO_PATH.name,
            "sample_rate": sample_rate,
        },

        "decoding": {
            "language": LANGUAGE,
            "task": TASK,
            "beam_size": BEAM_SIZE,
            "temperature": TEMPERATURE,
            "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
            "word_timestamps": WORD_TIMESTAMPS,
            "vad_filter": VAD_FILTER,
        },

        "samples": {},
    }

    # --------------------------------------------------------
    # Run each sample
    # --------------------------------------------------------

    for sample_name, item in ground_truth.items():

        print()
        print("-" * 70)
        print(f"Running {sample_name}")
        print("-" * 70)

        start_seconds = item["start_seconds"]
        duration_seconds = item["duration_seconds"]
        reference_text = item["reference_text"]

        sample_audio = extract_audio_sample(
            audio_data,
            sample_rate,
            start_seconds,
            duration_seconds,
        )

        print(
            f"Start: {start_seconds:.2f}s"
        )

        print(
            f"Duration: {len(sample_audio) / sample_rate:.2f}s"
        )

        # ----------------------------------------------------
        # Transcribe
        # ----------------------------------------------------

        segments, info = model.transcribe(
            sample_audio,
            language=LANGUAGE,
            task=TASK,
            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            word_timestamps=WORD_TIMESTAMPS,
            vad_filter=VAD_FILTER,
        )

        segments_list = []

        hypothesis_parts = []

        for segment in segments:

            text = segment.text.strip()

            hypothesis_parts.append(text)

            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })

        hypothesis = " ".join(hypothesis_parts).strip()

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        reference_normalized = normalize_arabic(
            reference_text
        )

        hypothesis_normalized = normalize_arabic(
            hypothesis
        )

        # ----------------------------------------------------
        # WER
        # ----------------------------------------------------

        metrics = calculate_wer(
            reference_normalized,
            hypothesis_normalized,
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()
        print("REFERENCE:")
        print(reference_text)

        print()
        print("HYPOTHESIS:")
        print(hypothesis)

        print()
        print("METRICS:")
        print(
            f"WER: {metrics['wer'] * 100:.2f}%"
        )
        print(
            f"Substitutions: {metrics['substitutions']}"
        )
        print(
            f"Deletions: {metrics['deletions']}"
        )
        print(
            f"Insertions: {metrics['insertions']}"
        )
        print(
            f"Total errors: {metrics['total_errors']}"
        )

        # ----------------------------------------------------
        # Save sample
        # ----------------------------------------------------

        results["samples"][sample_name] = {

            "sample": {
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
            },

            "reference": reference_text,

            "hypothesis": hypothesis,

            "metrics": {
                "reference_normalized": reference_normalized,
                "hypothesis_normalized": hypothesis_normalized,

                "reference_word_count": len(
                    reference_normalized.split()
                ),

                "hypothesis_word_count": len(
                    hypothesis_normalized.split()
                ),

                **metrics,
            },

            "model_info": {
                "language": getattr(
                    info,
                    "language",
                    None
                ),

                "language_probability": getattr(
                    info,
                    "language_probability",
                    None
                ),

                "duration": getattr(
                    info,
                    "duration",
                    None
                ),

                "duration_after_vad": getattr(
                    info,
                    "duration_after_vad",
                    None
                ),
            },

            "segments": segments_list,
        }

    # --------------------------------------------------------
    # Corpus-level metrics
    # --------------------------------------------------------

    total_reference_words = 0
    total_errors = 0

    for sample in results["samples"].values():

        total_reference_words += sample["metrics"][
            "reference_word_count"
        ]

        total_errors += sample["metrics"][
            "total_errors"
        ]

    corpus_wer = (
        total_errors / total_reference_words
        if total_reference_words > 0
        else 0.0
    )

    average_sample_wer = sum(
        sample["metrics"]["wer"]
        for sample in results["samples"].values()
    ) / len(results["samples"])

    results["summary"] = {

        "average_sample_wer": average_sample_wer,

        "corpus_wer": corpus_wer,

        "total_reference_words": total_reference_words,

        "total_errors": total_errors,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROUND 9C SUMMARY")
    print("=" * 70)

    print(
        f"Average sample WER: "
        f"{average_sample_wer * 100:.2f}%"
    )

    print(
        f"Corpus WER: "
        f"{corpus_wer * 100:.2f}%"
    )

    print(
        f"Total reference words: "
        f"{total_reference_words}"
    )

    print(
        f"Total errors: "
        f"{total_errors}"
    )

    print()
    print(f"Results saved to:")
    print(OUTPUT_JSON)

    print()
    print("=" * 70)
    print("ROUND 9C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()