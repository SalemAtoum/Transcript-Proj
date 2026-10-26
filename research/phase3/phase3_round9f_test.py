import json
import re
import unicodedata
from pathlib import Path

import soundfile as sf
import whisper


# ============================================================
# ROUND 9F
# Jordanian / Levantine Lexical Context Experiment
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_FILE = BASE_DIR / "meeting_processed.wav"
GROUND_TRUTH_FILE = BASE_DIR / "round8_ground_truth.json"

OUTPUT_DIR = BASE_DIR / "round9f_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "phase3_round9f_results.json"


# ============================================================
# Whisper configuration
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
# Experiment prompts
# ============================================================

PROMPTS = {

    # --------------------------------------------------------
    # A - No prompt
    # Exact Round 9A baseline
    # --------------------------------------------------------
    "A_no_prompt": None,

    # --------------------------------------------------------
    # B - Very light Jordanian lexical context
    # Only common spoken Jordanian/Levantine expressions.
    # --------------------------------------------------------
    "B_light_jordanian": (
        "كلام باللهجة الأردنية والشامية العامية. "
        "حافظ على الكلام كما نُطق باللهجة العامية، "
        "ولا تحوله إلى العربية الفصحى."
    ),

    # --------------------------------------------------------
    # C - Jordanian lexical hints
    # Adds examples of dialect vocabulary without domain words.
    # --------------------------------------------------------
    "C_lexical_hints": (
        "لهجة أردنية وشامية عامية. "
        "قد تظهر كلمات مثل: "
        "هاي، إحنا، إنتا، شو، ليش، كديش، "
        "ما بتقدر، مش، فش، وفيش، "
        "بشتغل، بعيش، بتحب، بدك، "
        "الزلمة، البنت، أخي، "
        "الله يسهل عليها. "
        "اكتب الكلام كما نُطق ولا تحوله للفصحى."
    ),

    # --------------------------------------------------------
    # D - Conversational context
    # Focuses on preserving spoken conversation.
    # --------------------------------------------------------
    "D_conversational": (
        "محادثة كلامية طبيعية باللهجة الأردنية والشامية. "
        "المتحدثون يستخدمون العامية والكلام اليومي. "
        "حافظ على الكلمات العامية كما نُطقت، "
        "ولا تستبدلها بكلمات عربية فصحى."
    ),
}


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize Arabic text for WER comparison.

    This is intentionally the same normalization approach
    used in Round 8 / Round 9A.
    """

    text = unicodedata.normalize("NFKC", text)

    # Lowercase Latin characters
    text = text.lower()

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    # Remove tatweel
    text = text.replace("ـ", "")

    # Normalize common Arabic letter variants
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# WER implementation
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
                0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    # --------------------------------------------------------
    # Recover operation counts
    # --------------------------------------------------------

    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        if i > 0 and j > 0:

            substitution_cost = (
                0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            )

            if dp[i][j] == dp[i - 1][j - 1] + substitution_cost:

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

    total_errors = substitutions + deletions + insertions

    wer = total_errors / n if n else None

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
# Ground truth loader
# ============================================================

def load_ground_truth():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["sample_00", "sample_20", "sample_30"]

    for sample_name in required:

        if sample_name not in data:
            raise ValueError(
                f"Missing {sample_name} in {GROUND_TRUTH_FILE}"
            )

        sample = data[sample_name]

        if "start_seconds" not in sample:
            raise ValueError(
                f"Missing start_seconds for {sample_name}"
            )

        if "duration_seconds" not in sample:
            raise ValueError(
                f"Missing duration_seconds for {sample_name}"
            )

        if "reference_text" not in sample:
            raise ValueError(
                f"Missing reference_text for {sample_name}"
            )

    return data


# ============================================================
# Extract exact audio samples
# ============================================================

def extract_sample(audio, sample_info):

    start = int(sample_info["start_seconds"])
    duration = int(sample_info["duration_seconds"])

    start_frame = int(start * 16000)
    end_frame = int((start + duration) * 16000)

    return audio[start_frame:end_frame]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 9F")
    print("Jordanian / Levantine Lexical Context Experiment")
    print("=" * 70)

    print()

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_FILE}"
        )

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n{GROUND_TRUTH_FILE}"
        )

    print("[1/5] Loading Ground Truth...")

    ground_truth = load_ground_truth()

    print("Ground Truth loaded.")

    print()

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print("[2/5] Loading meeting audio...")

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
        always_2d=False,
    )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, got {sample_rate} Hz"
        )

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio samples: {len(audio)}")

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("[3/5] Loading Whisper model...")

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
    )

    print("Model loaded.")

    print()

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    print("[4/5] Running experiments...")
    print()

    results = {
        "experiment": "Round 9F",
        "description": (
            "Controlled lexical-context experiment for "
            "Jordanian/Levantine Arabic"
        ),
        "model": MODEL_NAME,
        "device": DEVICE,
        "language": LANGUAGE,
        "task": TASK,
        "fp16": FP16,
        "beam_size": BEAM_SIZE,
        "temperature": TEMPERATURE,
        "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
        "logprob_threshold": LOGPROB_THRESHOLD,
        "no_speech_threshold": NO_SPEECH_THRESHOLD,
        "samples": {},
    }

    for prompt_name, prompt in PROMPTS.items():

        print("-" * 70)
        print(f"CONFIG: {prompt_name}")
        print("-" * 70)

        config_result = {
            "prompt": prompt,
            "samples": {},
        }

        for sample_name in [
            "sample_00",
            "sample_20",
            "sample_30",
        ]:

            sample_info = ground_truth[sample_name]

            print()
            print(f"Running {sample_name}...")

            sample_audio = extract_sample(
                audio,
                sample_info,
            )

            result = model.transcribe(
                sample_audio,
                language=LANGUAGE,
                task=TASK,
                fp16=FP16,
                temperature=TEMPERATURE,
                condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                logprob_threshold=LOGPROB_THRESHOLD,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                beam_size=BEAM_SIZE,
                initial_prompt=prompt,
                verbose=False,
            )

            hypothesis = result["text"].strip()

            reference = sample_info["reference_text"]

            wer_result = calculate_wer(
                reference,
                hypothesis,
            )

            sample_result = {
                "start_seconds": sample_info["start_seconds"],
                "duration_seconds": sample_info["duration_seconds"],
                "reference": reference,
                "hypothesis": hypothesis,
                "normalized_reference": normalize_text(reference),
                "normalized_hypothesis": normalize_text(hypothesis),
                "metrics": wer_result,
                "segment_count": len(result.get("segments", [])),
                "avg_logprob": (
                    sum(
                        s.get("avg_logprob", 0.0)
                        for s in result.get("segments", [])
                    )
                    / len(result.get("segments", []))
                    if result.get("segments")
                    else None
                ),
                "avg_compression_ratio": (
                    sum(
                        s.get("compression_ratio", 0.0)
                        for s in result.get("segments", [])
                    )
                    / len(result.get("segments", []))
                    if result.get("segments")
                    else None
                ),
                "avg_no_speech_prob": (
                    sum(
                        s.get("no_speech_prob", 0.0)
                        for s in result.get("segments", [])
                    )
                    / len(result.get("segments", []))
                    if result.get("segments")
                    else None
                ),
            }

            config_result["samples"][sample_name] = sample_result

            print()
            print("REFERENCE:")
            print(reference)

            print()
            print("HYPOTHESIS:")
            print(hypothesis)

            print()
            print(
                f"WER: {wer_result['WER'] * 100:.2f}%"
            )

            print(
                f"Errors: {wer_result['total_errors']}"
            )

        # ----------------------------------------------------
        # Corpus-level metrics
        # ----------------------------------------------------

        total_ref_words = 0
        total_errors = 0

        sample_wers = []

        for sample_result in config_result["samples"].values():

            metrics = sample_result["metrics"]

            total_ref_words += metrics["reference_word_count"]
            total_errors += metrics["total_errors"]

            sample_wers.append(metrics["WER"])

        corpus_wer = (
            total_errors / total_ref_words
            if total_ref_words
            else None
        )

        average_sample_wer = (
            sum(sample_wers) / len(sample_wers)
            if sample_wers
            else None
        )

        config_result["summary"] = {
            "corpus_WER": corpus_wer,
            "average_sample_WER": average_sample_wer,
            "reference_word_count": total_ref_words,
            "total_errors": total_errors,
        }

        results["samples"][prompt_name] = config_result

        print()
        print("=" * 70)
        print(f"{prompt_name} SUMMARY")
        print("=" * 70)

        print(
            f"Corpus WER: {corpus_wer * 100:.2f}%"
        )

        print(
            f"Average sample WER: "
            f"{average_sample_wer * 100:.2f}%"
        )

        print(
            f"Total errors: {total_errors}"
        )

        print()

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranking = []

    for config_name, config_result in results["samples"].items():

        summary = config_result["summary"]

        ranking.append({
            "config": config_name,
            "corpus_WER": summary["corpus_WER"],
            "average_sample_WER": summary["average_sample_WER"],
            "total_errors": summary["total_errors"],
        })

    ranking.sort(
        key=lambda x: (
            x["corpus_WER"],
            x["average_sample_WER"],
            x["total_errors"],
        )
    )

    results["ranking"] = ranking

    results["best_config"] = ranking[0]["config"]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("[5/5] Saving results...")

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
    print("ROUND 9F COMPLETE")
    print("=" * 70)

    print()

    print("Ranking:")

    for index, item in enumerate(ranking, start=1):

        print(
            f"{index}. "
            f"{item['config']} | "
            f"Corpus WER: {item['corpus_WER'] * 100:.2f}% | "
            f"Average WER: {item['average_sample_WER'] * 100:.2f}% | "
            f"Errors: {item['total_errors']}"
        )

    print()

    print(
        f"Best configuration: {results['best_config']}"
    )

    print()

    print(
        f"Results saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()