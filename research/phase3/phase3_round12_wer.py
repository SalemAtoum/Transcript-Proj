import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import whisper
import webrtcvad


# ============================================================
# ROUND 12
# Hybrid VAD + Less Fragmentation + No Overlap
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

AUDIO_PATH = Path("meeting_processed.wav")

OUTPUT_DIR = Path("round12_results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "phase3_round12_wer_results.json"


# ------------------------------------------------------------
# Whisper configuration
# ------------------------------------------------------------

MODEL_NAME = "large-v3"
DEVICE = "cpu"

LANGUAGE = "ar"
TASK = "transcribe"

FP16 = False

BEAM_SIZE = 5
TEMPERATURE = 0

CONDITION_ON_PREVIOUS_TEXT = False

COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


# ------------------------------------------------------------
# Round 12 VAD configuration
# ------------------------------------------------------------

VAD_MODE = 2

FRAME_MS = 30

PADDING_MS = 300

MIN_SPEECH_MS = 300

MERGE_GAP_MS = 1200

MAX_CHUNK_SEC = 20.0

CHUNK_OVERLAP_SEC = 0.0


# ------------------------------------------------------------
# Ground Truth
# EXACT same evaluation samples
# ------------------------------------------------------------

GROUND_TRUTH = [
    {
        "sample": "sample_00",
        "start": 0.0,
        "duration": 20.0,
        "reference": (
            "هي اخر ظرف انحبست سنة , اخذت براءة وروحت  , "
            "فش اشي بتحقق شو بده يتحقق , لو ما كنت حر شو بدك تكون , "
            "الزلمة بشتغل وبعيش عشان ابنه , بتوصل مرحلة انتا ما تبقدر تحب "
            "مش ما بدك تحب , العلاقات اخي دايما في نهاية زي ما في بداية  , "
            "البنت احسن بنت بالدنيا وفش نصيب , الله يسهل عليها"
        ),
    },
    {
        "sample": "sample_20",
        "start": 1220.0,
        "duration": 20.0,
        "reference": (
            "تخيل لليوم اخي  كانت متجوزة من ابوي حوالي 15 سنة على ما اعتقد "
            "مرض  هي الي تغيرله الي تطعميه الي تشربهعالتخت بقدرش يتحرك 6 سنين اخي"
        ),
    },
    {
        "sample": "sample_30",
        "start": 1814.0,
        "duration": 20.0,
        "reference": (
            "بقلك الفنانيين المشهورين بالاردن انا بس اطلع عليهم بشوف حالي فيهم "
            "مش اني بكبر فيهم لا بشوف نفسي زمان فيهم همه متطبعين مني فا مش قصة "
            "انه الناس بتقدي فيي اخي انا ورجيت الناس انه في امل  وهي الطريقة سهلة"
        ),
    },
]


# ============================================================
# Utility: Arabic normalization
# ============================================================

def normalize_arabic(text: str) -> str:
    """
    Conservative normalization.

    We intentionally do NOT aggressively normalize dialect words,
    because the goal is to measure actual transcription quality.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Remove tashkeel
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize Arabic variants
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    # Normalize alef maqsura
    text = text.replace("ى", "ي")

    # Normalize taa marbuta
    text = text.replace("ة", "ه")

    # Remove tatweel
    text = text.replace("ـ", "")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Remove underscores
    text = text.replace("_", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str):
    normalized = normalize_arabic(text)

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# WER
# ============================================================

def calculate_wer(reference: str, hypothesis: str):
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)

    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1

            substitution = dp[i - 1][j - 1] + cost
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1

            dp[i][j] = min(
                substitution,
                deletion,
                insertion,
            )

    # Backtrace
    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        if i > 0 and j > 0:

            if ref[i - 1] == hyp[j - 1]:

                i -= 1
                j -= 1
                continue

            if dp[i][j] == dp[i - 1][j - 1] + 1:
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

    wer = (
        total_errors / n
        if n > 0
        else 0.0
    )

    return {
        "normalized_reference": " ".join(ref),
        "normalized_hypothesis": " ".join(hyp),
        "reference_word_count": n,
        "hypothesis_word_count": m,
        "WER": wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
    }


# ============================================================
# Audio helpers
# ============================================================

def ensure_mono_int16(audio: np.ndarray) -> np.ndarray:

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio)

    if np.issubdtype(audio.dtype, np.integer):

        if audio.dtype == np.int16:
            return audio

        info = np.iinfo(audio.dtype)

        audio_float = audio.astype(np.float32) / max(
            abs(info.min),
            info.max,
        )

        audio_float = np.clip(
            audio_float,
            -1.0,
            1.0,
        )

        return (audio_float * 32767).astype(np.int16)

    audio = audio.astype(np.float32)

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    return (audio * 32767).astype(np.int16)


# ============================================================
# WebRTC VAD
# ============================================================

def detect_speech_regions(
    audio_int16: np.ndarray,
    sample_rate: int,
):
    """
    Detect speech using WebRTC VAD.

    Round 12 changes:
      - min speech = 300 ms
      - merge gap = 1200 ms
    """

    vad = webrtcvad.Vad(VAD_MODE)

    frame_samples = int(
        sample_rate * FRAME_MS / 1000
    )

    frame_bytes = frame_samples * 2

    audio_bytes = audio_int16.tobytes()

    frames = []

    timestamp = 0.0
    offset = 0

    while offset + frame_bytes <= len(audio_bytes):

        frame = audio_bytes[
            offset: offset + frame_bytes
        ]

        is_speech = vad.is_speech(
            frame,
            sample_rate,
        )

        frames.append(
            {
                "start": timestamp,
                "end": timestamp + FRAME_MS / 1000.0,
                "speech": bool(is_speech),
            }
        )

        offset += frame_bytes
        timestamp += FRAME_MS / 1000.0

    if not frames:
        return []

    # --------------------------------------------------------
    # Initial speech regions
    # --------------------------------------------------------

    regions = []

    current_start = None
    current_end = None

    for frame in frames:

        if frame["speech"]:

            if current_start is None:
                current_start = frame["start"]

            current_end = frame["end"]

        else:

            if current_start is not None:

                duration = (
                    current_end - current_start
                )

                if duration * 1000 >= MIN_SPEECH_MS:

                    regions.append(
                        [
                            current_start,
                            current_end,
                        ]
                    )

                current_start = None
                current_end = None

    if current_start is not None:

        duration = (
            current_end - current_start
        )

        if duration * 1000 >= MIN_SPEECH_MS:

            regions.append(
                [
                    current_start,
                    current_end,
                ]
            )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padding_sec = PADDING_MS / 1000.0

    padded = []

    for start, end in regions:

        start = max(
            0.0,
            start - padding_sec,
        )

        end = end + padding_sec

        padded.append(
            [start, end]
        )

    # --------------------------------------------------------
    # Merge nearby speech regions
    # --------------------------------------------------------

    merge_gap_sec = MERGE_GAP_MS / 1000.0

    merged = []

    for start, end in padded:

        if not merged:

            merged.append(
                [start, end]
            )

            continue

        previous_start, previous_end = merged[-1]

        gap = start - previous_end

        if gap <= merge_gap_sec:

            merged[-1][1] = max(
                previous_end,
                end,
            )

        else:

            merged.append(
                [start, end]
            )

    return merged


# ============================================================
# Limit region to maximum chunk size
# NO OVERLAP
# ============================================================

def split_into_chunks(
    regions,
    max_chunk_sec: float,
):
    chunks = []

    for start, end in regions:

        duration = end - start

        if duration <= max_chunk_sec:

            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            )

            continue

        current = start

        while current < end:

            chunk_end = min(
                current + max_chunk_sec,
                end,
            )

            chunks.append(
                {
                    "start": current,
                    "end": chunk_end,
                    "duration": chunk_end - current,
                }
            )

            current = chunk_end

    return chunks


# ============================================================
# Extract audio
# ============================================================

def extract_audio_segment(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
):
    start_sample = max(
        0,
        int(round(start * sample_rate)),
    )

    end_sample = min(
        len(audio),
        int(round(end * sample_rate)),
    )

    return audio[
        start_sample:end_sample
    ]


# ============================================================
# Suspicious output detection
# ============================================================

def is_repeated_character_segment(text: str):
    cleaned = re.sub(
        r"\s+",
        "",
        text,
    )

    if len(cleaned) < 10:
        return False

    unique_chars = set(cleaned)

    return len(unique_chars) <= 2


def is_suspicious_segment(
    text: str,
    compression_ratio: float,
    avg_logprob: float,
):
    if is_repeated_character_segment(text):
        return True

    if compression_ratio > COMPRESSION_RATIO_THRESHOLD:
        return True

    if avg_logprob < LOGPROB_THRESHOLD:
        return True

    return False


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 12 — Hybrid VAD + Less Fragmentation")
    print("=" * 70)

    print()
    print("Configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Language: {LANGUAGE}")
    print(f"Beam size: {BEAM_SIZE}")
    print(f"Temperature: {TEMPERATURE}")
    print(
        f"Condition on previous text: "
        f"{CONDITION_ON_PREVIOUS_TEXT}"
    )

    print()
    print("VAD:")
    print(f"Mode: {VAD_MODE}")
    print(f"Frame: {FRAME_MS} ms")
    print(f"Padding: {PADDING_MS} ms")
    print(f"Min speech: {MIN_SPEECH_MS} ms")
    print(f"Merge gap: {MERGE_GAP_MS} ms")
    print(f"Max chunk: {MAX_CHUNK_SEC} sec")
    print(f"Overlap: {CHUNK_OVERLAP_SEC} sec")

    # --------------------------------------------------------
    # Check audio
    # --------------------------------------------------------

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    print()
    print("Loading audio...")

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
    )

    audio_int16 = ensure_mono_int16(
        audio
    )

    print(
        f"Sample rate: {sample_rate}"
    )

    print(
        f"Duration: "
        f"{len(audio_int16) / sample_rate:.2f} sec"
    )

    # --------------------------------------------------------
    # Load Whisper
    # --------------------------------------------------------

    print()
    print("Loading Whisper large-v3...")
    print("CPU mode. This may take some time.")

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
    )

    print("Whisper loaded.")

    all_results = []

    # Corpus counters
    total_reference_words = 0
    total_errors = 0

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    for item in GROUND_TRUTH:

        sample_name = item["sample"]
        start = item["start"]
        duration = item["duration"]

        print()
        print("=" * 70)
        print(f"Processing {sample_name}")
        print(
            f"Start: {start}s | "
            f"Duration: {duration}s"
        )
        print("=" * 70)

        end = start + duration

        sample_audio = extract_audio_segment(
            audio_int16,
            sample_rate,
            start,
            end,
        )

        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        speech_regions = detect_speech_regions(
            sample_audio,
            sample_rate,
        )

        # Restrict to sample duration
        sample_duration = len(
            sample_audio
        ) / sample_rate

        cleaned_regions = []

        for region_start, region_end in speech_regions:

            region_start = max(
                0.0,
                min(
                    region_start,
                    sample_duration,
                ),
            )

            region_end = max(
                0.0,
                min(
                    region_end,
                    sample_duration,
                ),
            )

            if region_end > region_start:

                cleaned_regions.append(
                    [
                        region_start,
                        region_end,
                    ]
                )

        speech_regions = cleaned_regions

        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        chunks = split_into_chunks(
            speech_regions,
            MAX_CHUNK_SEC,
        )

        print()
        print(
            f"Speech regions: "
            f"{len(speech_regions)}"
        )

        for index, region in enumerate(
            speech_regions,
            start=1,
        ):

            print(
                f"  Region {index}: "
                f"{region[0]:.2f}s → "
                f"{region[1]:.2f}s"
            )

        print(
            f"Chunks: {len(chunks)}"
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"  Chunk {index}: "
                f"{chunk['start']:.2f}s → "
                f"{chunk['end']:.2f}s"
            )

        # ----------------------------------------------------
        # Whisper transcription
        # ----------------------------------------------------

        segment_results = []

        for chunk_index, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk_audio = extract_audio_segment(
                sample_audio,
                sample_rate,
                chunk["start"],
                chunk["end"],
            )

            if len(chunk_audio) == 0:
                continue

            # Whisper expects float32 [-1, 1]
            chunk_float = (
                chunk_audio.astype(
                    np.float32
                ) / 32768.0
            )

            print()
            print(
                f"Transcribing chunk "
                f"{chunk_index}/{len(chunks)}..."
            )

            result = model.transcribe(
                chunk_float,
                language=LANGUAGE,
                task=TASK,
                fp16=FP16,
                beam_size=BEAM_SIZE,
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
                verbose=False,
            )

            for seg in result.get(
                "segments",
                [],
            ):

                text = seg.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    continue

                local_start = float(
                    seg.get("start", 0.0)
                )

                local_end = float(
                    seg.get("end", 0.0)
                )

                absolute_start = (
                    chunk["start"]
                    + local_start
                )

                absolute_end = (
                    chunk["start"]
                    + local_end
                )

                avg_logprob = float(
                    seg.get(
                        "avg_logprob",
                        0.0,
                    )
                )

                compression_ratio = float(
                    seg.get(
                        "compression_ratio",
                        0.0,
                    )
                )

                no_speech_prob = float(
                    seg.get(
                        "no_speech_prob",
                        0.0,
                    )
                )

                segment_results.append(
                    {
                        "start": absolute_start,
                        "end": absolute_end,
                        "text": text,
                        "avg_logprob": avg_logprob,
                        "compression_ratio": compression_ratio,
                        "no_speech_prob": no_speech_prob,
                    }
                )

        # ----------------------------------------------------
        # Combine hypothesis
        # ----------------------------------------------------

        segment_results.sort(
            key=lambda x: x["start"]
        )

        hypothesis_parts = [
            x["text"]
            for x in segment_results
        ]

        hypothesis = " ".join(
            hypothesis_parts
        ).strip()

        # ----------------------------------------------------
        # WER
        # ----------------------------------------------------

        wer_metrics = calculate_wer(
            item["reference"],
            hypothesis,
        )

        # ----------------------------------------------------
        # Quality metrics
        # ----------------------------------------------------

        if segment_results:

            average_compression = float(
                np.mean(
                    [
                        x["compression_ratio"]
                        for x in segment_results
                    ]
                )
            )

            maximum_compression = float(
                max(
                    x["compression_ratio"]
                    for x in segment_results
                )
            )

            average_logprob = float(
                np.mean(
                    [
                        x["avg_logprob"]
                        for x in segment_results
                    ]
                )
            )

            average_no_speech = float(
                np.mean(
                    [
                        x["no_speech_prob"]
                        for x in segment_results
                    ]
                )
            )

        else:

            average_compression = 0.0
            maximum_compression = 0.0
            average_logprob = 0.0
            average_no_speech = 0.0

        suspicious_segments = []

        repeated_character_segments = []

        for seg in segment_results:

            if is_suspicious_segment(
                seg["text"],
                seg["compression_ratio"],
                seg["avg_logprob"],
            ):

                suspicious_segments.append(
                    seg
                )

            if is_repeated_character_segment(
                seg["text"]
            ):

                repeated_character_segments.append(
                    seg
                )

        sample_result = {

            "sample": sample_name,

            "audio": {
                "source": str(AUDIO_PATH),
                "start_seconds": start,
                "duration_seconds": duration,
                "sample_rate": sample_rate,
            },

            "configuration": {
                "model": MODEL_NAME,
                "device": DEVICE,
                "language": LANGUAGE,
                "task": TASK,
                "fp16": FP16,
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
            },

            "vad_configuration": {
                "vad_mode": VAD_MODE,
                "frame_ms": FRAME_MS,
                "padding_ms": PADDING_MS,
                "min_speech_ms": MIN_SPEECH_MS,
                "merge_gap_ms": MERGE_GAP_MS,
                "max_chunk_sec": MAX_CHUNK_SEC,
                "chunk_overlap_sec": CHUNK_OVERLAP_SEC,
            },

            "speech_regions": [
                {
                    "start": r[0],
                    "end": r[1],
                    "duration": r[1] - r[0],
                }
                for r in speech_regions
            ],

            "chunks": chunks,

            "reference": item["reference"],

            "hypothesis": hypothesis,

            "wer_metrics": wer_metrics,

            "quality_metrics": {
                "segment_count": len(
                    segment_results
                ),
                "chunk_count": len(chunks),
                "speech_region_count": len(
                    speech_regions
                ),
                "average_compression_ratio": (
                    average_compression
                ),
                "maximum_compression_ratio": (
                    maximum_compression
                ),
                "average_logprob": (
                    average_logprob
                ),
                "average_no_speech_probability": (
                    average_no_speech
                ),
                "suspicious_segment_count": len(
                    suspicious_segments
                ),
                "repeated_character_segment_count": len(
                    repeated_character_segments
                ),
            },

            "segments": segment_results,

            "suspicious_segments": (
                suspicious_segments
            ),

            "repeated_character_segments": (
                repeated_character_segments
            ),
        }

        all_results.append(
            sample_result
        )

        total_reference_words += (
            wer_metrics["reference_word_count"]
        )

        total_errors += (
            wer_metrics["total_errors"]
        )

        print()
        print("-" * 70)
        print(f"{sample_name} RESULT")
        print("-" * 70)

        print(
            f"WER: "
            f"{wer_metrics['WER'] * 100:.2f}%"
        )

        print(
            f"Errors: "
            f"{wer_metrics['total_errors']}"
        )

        print(
            f"Substitutions: "
            f"{wer_metrics['substitutions']}"
        )

        print(
            f"Deletions: "
            f"{wer_metrics['deletions']}"
        )

        print(
            f"Insertions: "
            f"{wer_metrics['insertions']}"
        )

        print()
        print("Hypothesis:")
        print(hypothesis)

        print()
        print(
            f"Suspicious segments: "
            f"{len(suspicious_segments)}"
        )

        print(
            f"Repeated-character segments: "
            f"{len(repeated_character_segments)}"
        )

    # ========================================================
    # Corpus results
    # ========================================================

    corpus_wer = (
        total_errors / total_reference_words
        if total_reference_words > 0
        else 0.0
    )

    sample_wers = [
        result["wer_metrics"]["WER"]
        for result in all_results
    ]

    average_sample_wer = (
        float(np.mean(sample_wers))
        if sample_wers
        else 0.0
    )

    previous_best_wer = 0.475

    difference = (
        previous_best_wer
        - corpus_wer
    )

    improvement_percentage_points = (
        difference * 100
    )

    if corpus_wer < previous_best_wer:

        verdict = (
            "PASS — Round 12 beats the previous "
            "best WER."
        )

    elif abs(
        corpus_wer - previous_best_wer
    ) < 1e-12:

        verdict = (
            "TIE — Round 12 matches the "
            "previous best WER."
        )

    else:

        verdict = (
            "NO WER IMPROVEMENT — Round 12 "
            "is worse than the previous best WER."
        )

    # Stability summary
    total_suspicious = sum(
        result["quality_metrics"][
            "suspicious_segment_count"
        ]
        for result in all_results
    )

    total_repeated = sum(
        result["quality_metrics"][
            "repeated_character_segment_count"
        ]
        for result in all_results
    )

    stability_pass = (
        total_suspicious == 0
        and total_repeated == 0
    )

    final_result = {

        "experiment": "Round 12",

        "description": (
            "Hybrid VAD experiment using less "
            "fragmentation, larger maximum chunks, "
            "and zero chunk overlap."
        ),

        "previous_best": {
            "experiment": "Round 9A",
            "corpus_WER": 0.475,
            "reference_word_count": 120,
            "total_errors": 57,
        },

        "configuration": {
            "model": MODEL_NAME,
            "device": DEVICE,
            "language": LANGUAGE,
            "task": TASK,
            "fp16": FP16,
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
        },

        "vad_configuration": {
            "vad_mode": VAD_MODE,
            "frame_ms": FRAME_MS,
            "padding_ms": PADDING_MS,
            "min_speech_ms": MIN_SPEECH_MS,
            "merge_gap_ms": MERGE_GAP_MS,
            "max_chunk_sec": MAX_CHUNK_SEC,
            "chunk_overlap_sec": CHUNK_OVERLAP_SEC,
        },

        "samples": all_results,

        "summary": {

            "corpus_WER": corpus_wer,

            "corpus_WER_percent": (
                corpus_wer * 100
            ),

            "average_sample_WER": (
                average_sample_wer
            ),

            "average_sample_WER_percent": (
                average_sample_wer * 100
            ),

            "reference_word_count": (
                total_reference_words
            ),

            "total_errors": total_errors,

            "previous_best_WER": (
                previous_best_wer
            ),

            "previous_best_WER_percent": 47.5,

            "improvement_percentage_points": (
                improvement_percentage_points
            ),

            "total_suspicious_segments": (
                total_suspicious
            ),

            "total_repeated_character_segments": (
                total_repeated
            ),

            "stability_pass": stability_pass,

            "verdict": verdict,
        },
    }

    # ========================================================
    # Save JSON
    # ========================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final console summary
    # ========================================================

    print()
    print("=" * 70)
    print("ROUND 12 FINAL RESULT")
    print("=" * 70)

    print(
        f"Corpus WER: "
        f"{corpus_wer * 100:.2f}%"
    )

    print(
        f"Reference words: "
        f"{total_reference_words}"
    )

    print(
        f"Total errors: "
        f"{total_errors}"
    )

    print(
        f"Previous best WER: "
        f"{previous_best_wer * 100:.2f}%"
    )

    print(
        f"Difference: "
        f"{improvement_percentage_points:+.2f} "
        f"percentage points"
    )

    print()
    print(
        f"Suspicious segments: "
        f"{total_suspicious}"
    )

    print(
        f"Repeated-character segments: "
        f"{total_repeated}"
    )

    print(
        f"Stability PASS: "
        f"{stability_pass}"
    )

    print()
    print(f"VERDICT: {verdict}")

    print()
    print(
        f"Saved to: {OUTPUT_JSON}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()