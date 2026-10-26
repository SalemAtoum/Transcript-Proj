import json
import re
import subprocess
import unicodedata
from pathlib import Path

import whisper


# ============================================================
# PHASE 3 - ROUND 8
# FINAL CONTROLLED ACCURACY EVALUATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_AUDIO = PROJECT_DIR / "meeting_processed.wav"
GROUND_TRUTH_FILE = PROJECT_DIR / "round8_ground_truth.json"

OUTPUT_DIR = PROJECT_DIR / "round8_evaluation"
CLIPS_DIR = OUTPUT_DIR / "clips"

MODEL_NAME = "large-v3"
LANGUAGE = "ar"
TASK = "transcribe"

TEMPERATURE = 0
CONDITION_ON_PREVIOUS_TEXT = True

COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6

FP16 = False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> list[str]:
    """
    Normalize text for WER calculation.

    Important:
    - We do NOT translate dialect.
    - We do NOT convert dialect to Fusha.
    - We only normalize orthographic noise.
    """

    text = unicodedata.normalize("NFKC", text)

    # Lowercase Latin characters
    text = text.lower()

    # Remove Arabic diacritics and tatweel
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)

    # Normalize common Alef variants
    text = text.translate(
        str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ى": "ي",
            }
        )
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return text.split()


# ============================================================
# WER
# ============================================================

def calculate_wer(reference: str, hypothesis: str):
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    n = len(ref)
    m = len(hyp)

    # DP matrix
    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    # Backtrace matrix
    backtrace = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            substitution_cost = (
                0
                if ref[i - 1] == hyp[j - 1]
                else 1
            )

            substitution = (
                dp[i - 1][j - 1]
                + substitution_cost
            )

            deletion = (
                dp[i - 1][j]
                + 1
            )

            insertion = (
                dp[i][j - 1]
                + 1
            )

            best = min(
                substitution,
                deletion,
                insertion,
            )

            dp[i][j] = best

            if best == substitution:
                backtrace[i][j] = (
                    "equal"
                    if substitution_cost == 0
                    else "substitution"
                )

            elif best == deletion:
                backtrace[i][j] = "deletion"

            else:
                backtrace[i][j] = "insertion"

    # Recover operations
    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        operation = backtrace[i][j]

        if operation == "equal":
            i -= 1
            j -= 1

        elif operation == "substitution":
            substitutions += 1
            i -= 1
            j -= 1

        elif operation == "deletion":
            deletions += 1
            i -= 1

        elif operation == "insertion":
            insertions += 1
            j -= 1

        else:
            raise RuntimeError(
                "WER backtrace error."
            )

    errors = (
        substitutions
        + deletions
        + insertions
    )

    wer = (
        errors / n
        if n > 0
        else 0
    )

    return {
        "wer": wer,
        "reference_word_count": n,
        "hypothesis_word_count": m,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": errors,
    }


# ============================================================
# FFMPEG
# ============================================================

def extract_clip(
    start_seconds: int,
    duration_seconds: int,
    output_path: Path,
):

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(INPUT_AUDIO),
        "-t",
        str(duration_seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    print(
        f"\nExtracting clip "
        f"{start_seconds}s -> "
        f"{start_seconds + duration_seconds}s"
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"FFmpeg failed for {output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 - ROUND 8")
    print("FINAL CONTROLLED ACCURACY EVALUATION")
    print("=" * 70)

    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(
            f"Missing:\n{INPUT_AUDIO}"
        )

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{GROUND_TRUTH_FILE}"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        ground_truth = json.load(f)

    print("\nLoading Whisper model...")
    print(f"Model: {MODEL_NAME}")
    print("Device: CPU")
    print("This may take some time.")

    model = whisper.load_model(
        MODEL_NAME,
        device="cpu",
    )

    results = {
        "phase": 3,
        "round": 8,
        "model": MODEL_NAME,
        "device": "cpu",
        "language": LANGUAGE,
        "task": TASK,
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
        "samples": {},
    }

    for sample_id, sample in ground_truth.items():

        print("\n" + "=" * 70)
        print(f"Processing: {sample_id}")
        print("=" * 70)

        start = sample["start_seconds"]
        duration = sample["duration_seconds"]

        reference = sample["reference_text"]

        clip_path = (
            CLIPS_DIR
            / f"{sample_id}.wav"
        )

        extract_clip(
            start,
            duration,
            clip_path,
        )

        print("Running Whisper...")

        result = model.transcribe(
            str(clip_path),

            language=LANGUAGE,
            task=TASK,

            temperature=TEMPERATURE,

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

            fp16=FP16,

            verbose=False,
        )

        hypothesis = result["text"].strip()

        wer_result = calculate_wer(
            reference,
            hypothesis,
        )

        results["samples"][sample_id] = {
            "start_seconds": start,
            "duration_seconds": duration,

            "reference_text": reference,

            "hypothesis_text": hypothesis,

            "detected_language": (
                result.get("language")
            ),

            "wer": wer_result,
        }

        print("\nREFERENCE:")
        print(reference)

        print("\nWHISPER:")
        print(hypothesis)

        print("\nWER:")
        print(
            f"{wer_result['wer'] * 100:.2f}%"
        )

        print(
            "Substitutions:",
            wer_result["substitutions"],
        )

        print(
            "Deletions:",
            wer_result["deletions"],
        )

        print(
            "Insertions:",
            wer_result["insertions"],
        )

    # ========================================================
    # OVERALL
    # ========================================================

    all_reference = []
    all_hypothesis = []

    for sample in results["samples"].values():

        all_reference.append(
            sample["reference_text"]
        )

        all_hypothesis.append(
            sample["hypothesis_text"]
        )

    overall_reference = " ".join(
        all_reference
    )

    overall_hypothesis = " ".join(
        all_hypothesis
    )

    overall_wer = calculate_wer(
        overall_reference,
        overall_hypothesis,
    )

    results["overall"] = {
        "wer": overall_wer["wer"],
        "reference_word_count": (
            overall_wer[
                "reference_word_count"
            ]
        ),
        "hypothesis_word_count": (
            overall_wer[
                "hypothesis_word_count"
            ]
        ),
        "substitutions": (
            overall_wer["substitutions"]
        ),
        "deletions": (
            overall_wer["deletions"]
        ),
        "insertions": (
            overall_wer["insertions"]
        ),
        "total_errors": (
            overall_wer["total_errors"]
        ),
    }

    output_file = (
        OUTPUT_DIR
        / "phase3_round8_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("ROUND 8 COMPLETE")
    print("=" * 70)

    print(
        f"\nOVERALL WER: "
        f"{overall_wer['wer'] * 100:.2f}%"
    )

    print(
        f"Reference words: "
        f"{overall_wer['reference_word_count']}"
    )

    print(
        f"Total errors: "
        f"{overall_wer['total_errors']}"
    )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()