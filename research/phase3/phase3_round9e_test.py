import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions


# ============================================================
# ROUND 9E
# VAD PARAMETER VALIDATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_PATH = PROJECT_DIR / "meeting_processed.wav"
GROUND_TRUTH_PATH = PROJECT_DIR / "round8_ground_truth.json"

OUTPUT_DIR = PROJECT_DIR / "round9e_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "phase3_round9e_results.json"


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "large-v3"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"

LANGUAGE = "ar"
TASK = "transcribe"

BEAM_SIZE = 5
TEMPERATURE = 0.0

CONDITION_ON_PREVIOUS_TEXT = True


# ============================================================
# VAD CONFIGURATIONS
# ============================================================

VAD_CONFIGS = {

    "A_current_baseline": {
        "threshold": 0.50,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    },

    "B_permissive": {
        "threshold": 0.35,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    },

    "C_strict": {
        "threshold": 0.65,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    },

    "D_short_silence": {
        "threshold": 0.50,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 200,
    },

    "E_long_silence": {
        "threshold": 0.50,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 700,
        "speech_pad_ms": 200,
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_arabic(text):

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    # Arabic diacritics
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    # Tatweel
    text = text.replace("ـ", "")

    # Arabic letter normalization
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ى", "ي")

    # Punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    # Whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# WER
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
            and dp[i][j]
            == dp[i - 1][j - 1] + 1
        ):
            substitutions += 1
            i -= 1
            j -= 1
            continue

        if (
            i > 0
            and dp[i][j]
            == dp[i - 1][j] + 1
        ):
            deletions += 1
            i -= 1
            continue

        if (
            j > 0
            and dp[i][j]
            == dp[i][j - 1] + 1
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
# GROUND TRUTH
# ============================================================

def load_ground_truth():

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    required = [
        "sample_00",
        "sample_20",
        "sample_30",
    ]

    samples = {}

    for name in required:

        if name not in data:
            raise ValueError(
                f"Missing {name} "
                "from ground truth."
            )

        item = data[name]

        samples[name] = {
            "start_seconds":
                float(item["start_seconds"]),

            "duration_seconds":
                float(item["duration_seconds"]),

            "reference_text":
                item["reference_text"],
        }

    return samples


# ============================================================
# AUDIO SAMPLE
# ============================================================

def extract_sample(
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

    if start_frame < 0:
        start_frame = 0

    if end_frame > len(audio):
        end_frame = len(audio)

    sample = audio[
        start_frame:end_frame
    ]

    if len(sample) == 0:
        raise ValueError(
            "Extracted audio sample is empty."
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
    vad_config,
):

    vad_options = VadOptions(
        threshold=vad_config["threshold"],
        min_speech_duration_ms=
            vad_config[
                "min_speech_duration_ms"
            ],
        min_silence_duration_ms=
            vad_config[
                "min_silence_duration_ms"
            ],
        speech_pad_ms=
            vad_config[
                "speech_pad_ms"
            ],
    )

    segments_generator, info = (
        model.transcribe(
            audio,
            language=LANGUAGE,
            task=TASK,
            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,
            condition_on_previous_text=
                CONDITION_ON_PREVIOUS_TEXT,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=vad_options,
        )
    )

    segments = []
    text_parts = []

    for segment in segments_generator:

        text = segment.text.strip()

        text_parts.append(text)

        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })

    hypothesis = " ".join(
        text_parts
    ).strip()

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

    print("=" * 70)
    print("PHASE 3 — ROUND 9E")
    print("VAD PARAMETER VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Missing audio:\n{AUDIO_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Missing ground truth:\n"
            f"{GROUND_TRUTH_PATH}"
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
            f"Expected 16000 Hz, "
            f"got {sample_rate} Hz"
        )

    if audio.ndim != 1:
        raise ValueError(
            "Expected mono audio."
        )

    print(
        f"Sample rate: {sample_rate} Hz"
    )

    print(
        f"Duration: "
        f"{len(audio) / sample_rate:.2f}s"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    print("\nGround Truth:")

    for name, item in ground_truth.items():

        print(
            f"  {name}: "
            f"{item['start_seconds']}s → "
            f"{item['start_seconds'] + item['duration_seconds']}s"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading large-v3...")

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

        "experiment":
            "Round 9E",

        "description":
            "Controlled VAD parameter comparison",

        "model":
            MODEL_NAME,

        "device":
            DEVICE,

        "compute_type":
            COMPUTE_TYPE,

        "decoding": {
            "language": LANGUAGE,
            "task": TASK,
            "beam_size": BEAM_SIZE,
            "temperature": TEMPERATURE,
            "condition_on_previous_text":
                CONDITION_ON_PREVIOUS_TEXT,
        },

        "vad_configurations":
            VAD_CONFIGS,

        "experiments": {},
    }

    # --------------------------------------------------------
    # Run configurations
    # --------------------------------------------------------

    for config_name, vad_config in (
        VAD_CONFIGS.items()
    ):

        print("\n" + "=" * 70)

        print(
            f"CONFIGURATION: "
            f"{config_name}"
        )

        print(
            f"threshold="
            f"{vad_config['threshold']}"
        )

        print(
            f"min_silence="
            f"{vad_config['min_silence_duration_ms']}ms"
        )

        print("=" * 70)

        config_results = {
            "vad_config":
                vad_config,

            "samples": {},
        }

        total_errors = 0
        total_reference_words = 0

        sample_wers = []

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

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

            transcription = (
                transcribe_sample(
                    model,
                    sample_audio,
                    vad_config,
                )
            )

            reference = item[
                "reference_text"
            ]

            hypothesis = transcription[
                "hypothesis"
            ]

            reference_normalized = (
                normalize_arabic(
                    reference
                )
            )

            hypothesis_normalized = (
                normalize_arabic(
                    hypothesis
                )
            )

            metrics = calculate_wer(
                reference_normalized,
                hypothesis_normalized,
            )

            total_errors += metrics[
                "total_errors"
            ]

            total_reference_words += (
                metrics[
                    "reference_word_count"
                ]
            )

            sample_wers.append(
                metrics["wer"]
            )

            config_results[
                "samples"
            ][sample_name] = {

                "start_seconds":
                    item["start_seconds"],

                "duration_seconds":
                    item["duration_seconds"],

                "reference":
                    reference,

                "hypothesis":
                    hypothesis,

                "reference_normalized":
                    reference_normalized,

                "hypothesis_normalized":
                    hypothesis_normalized,

                "metrics":
                    metrics,

                "model_info":
                    transcription[
                        "model_info"
                    ],

                "segments":
                    transcription[
                        "segments"
                    ],
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

        # ----------------------------------------------------
        # Configuration summary
        # ----------------------------------------------------

        corpus_wer = (
            total_errors
            / total_reference_words
            if total_reference_words > 0
            else 0.0
        )

        average_wer = (
            sum(sample_wers)
            / len(sample_wers)
        )

        config_results[
            "summary"
        ] = {

            "corpus_wer":
                corpus_wer,

            "average_sample_wer":
                average_wer,

            "total_reference_words":
                total_reference_words,

            "total_errors":
                total_errors,
        }

        results[
            "experiments"
        ][config_name] = config_results

        print(
            "\nCONFIGURATION RESULT:"
        )

        print(
            f"Corpus WER: "
            f"{corpus_wer * 100:.2f}%"
        )

        print(
            f"Total errors: "
            f"{total_errors}"
        )

    # ========================================================
    # RANK CONFIGURATIONS
    # ========================================================

    ranking = []

    for name, data in (
        results["experiments"].items()
    ):

        ranking.append({
            "configuration": name,
            "corpus_wer":
                data["summary"][
                    "corpus_wer"
                ],
            "average_sample_wer":
                data["summary"][
                    "average_sample_wer"
                ],
            "total_errors":
                data["summary"][
                    "total_errors"
                ],
        })

    ranking.sort(
        key=lambda x: x["corpus_wer"]
    )

    results["ranking"] = ranking

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = ranking[0]

    results["best_configuration"] = best

    # ========================================================
    # SAVE
    # ========================================================

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

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ROUND 9E FINAL RANKING")
    print("=" * 70)

    for index, item in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{index}. "
            f"{item['configuration']} "
            f"→ "
            f"{item['corpus_wer'] * 100:.2f}% "
            f"WER "
            f"({item['total_errors']} errors)"
        )

    print("\nBEST CONFIGURATION:")

    print(
        best["configuration"]
    )

    print(
        f"WER: "
        f"{best['corpus_wer'] * 100:.2f}%"
    )

    print(
        f"Errors: "
        f"{best['total_errors']}"
    )

    print("\nResults saved to:")

    print(OUTPUT_PATH)

    print("\n")
    print("=" * 70)
    print("ROUND 9E COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()