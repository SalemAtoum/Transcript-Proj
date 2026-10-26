import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import whisper
import webrtcvad


# ============================================================
# ROUND 11
# WER VALIDATION
# ============================================================

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
# Same VAD configuration used in Round 10B
# ------------------------------------------------------------

VAD_MODE = 2
FRAME_MS = 30
PADDING_MS = 300
MIN_SPEECH_MS = 500
MERGE_GAP_MS = 600

MAX_CHUNK_SEC = 15.0
CHUNK_OVERLAP_SEC = 0.3


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

AUDIO_FILE = BASE_DIR / "meeting_processed.wav"

OUTPUT_DIR = BASE_DIR / "round11_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "phase3_round11_wer_results.json"


# ============================================================
# GROUND TRUTH
# ============================================================
#
# IMPORTANT:
# These are the exact reference texts used in the previous
# controlled WER evaluation.
#
# Sample 00:
# original meeting: 0:00 -> 0:20
#
# Sample 20:
# original meeting: 20:20 -> 20:40
# because Sample B starts at 1200 sec.
#
# Sample 30:
# original meeting: 30:14 -> 30:34
# because Sample C starts at 1800 sec.
# ============================================================

GROUND_TRUTH = [
    {
        "name": "sample_00",
        "start_seconds": 0.0,
        "duration_seconds": 20.0,
        "reference": (
            "هي اخر ظرف انحبست سنة , اخذت براءة وروحت , "
            "فش اشي بتحقق شو بده يتحقق , لو ما كنت حر شو بدك تكون , "
            "الزلمة بشتغل وبعيش عشان ابنه , بتوصل مرحلة انتا ما تبقدر تحب "
            "مش ما بدك تحب , العلاقات اخي دايما في نهاية زي ما في بداية , "
            "البنت احسن بنت بالدنيا وفش نصيب , الله يسهل عليها"
        ),
    },
    {
        "name": "sample_20",
        "start_seconds": 1220.0,
        "duration_seconds": 20.0,
        "reference": (
            "تخيل لليوم اخي كانت متجوزة من ابوي حوالي 15 سنة "
            "على ما اعتقد مرض هي الي تغيرله الي تطعميه الي تشربهعالتخت "
            "بقدرش يتحرك 6 سنين اخي"
        ),
    },
    {
        "name": "sample_30",
        "start_seconds": 1814.0,
        "duration_seconds": 20.0,
        "reference": (
            "بقلك الفنانيين المشهورين بالاردن انا بس اطلع عليهم "
            "بشوف حالي فيهم مش اني بكبر فيهم لا بشوف نفسي زمان فيهم "
            "همه متطبعين مني فا مش قصة انه الناس بتقدي فيي اخي "
            "انا ورجيت الناس انه في امل وهي الطريقة سهلة"
        ),
    },
]


# ============================================================
# ARABIC NORMALIZATION
# ============================================================

def normalize_arabic(text: str) -> str:
    """
    Normalization kept intentionally conservative.

    Goal:
    - remove punctuation
    - normalize common Arabic letter variants
    - normalize whitespace
    - keep actual word differences

    We do NOT convert:
        6 <-> ست
        هي <-> هاي
        اللي <-> الي
        بقدرش <-> بيقدرش

    because those are real recognition differences and should
    remain visible in WER.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic diacritics / combining marks
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # Tatweel
    text = text.replace("ـ", "")

    # Common Arabic letter normalization
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation/symbols
    cleaned = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith("P"):
            cleaned.append(" ")
        elif category.startswith("S"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)

    text = "".join(cleaned)

    # Remove duplicated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# WER / EDIT DISTANCE
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

    dp = [
        [
            {
                "cost": 0,
                "sub": 0,
                "del": 0,
                "ins": 0,
            }
            for _ in range(m + 1)
        ]
        for _ in range(n + 1)
    ]

    # Reference -> empty hypothesis = deletions
    for i in range(1, n + 1):
        dp[i][0] = {
            "cost": i,
            "sub": 0,
            "del": i,
            "ins": 0,
        }

    # Empty reference -> hypothesis = insertions
    for j in range(1, m + 1):
        dp[0][j] = {
            "cost": j,
            "sub": 0,
            "del": 0,
            "ins": j,
        }

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            # Match
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                best = dp[i - 1][j - 1].copy()

            else:
                # Substitution
                previous = dp[i - 1][j - 1]

                substitution = {
                    "cost": previous["cost"] + 1,
                    "sub": previous["sub"] + 1,
                    "del": previous["del"],
                    "ins": previous["ins"],
                }

                # Deletion
                previous = dp[i - 1][j]

                deletion = {
                    "cost": previous["cost"] + 1,
                    "sub": previous["sub"],
                    "del": previous["del"] + 1,
                    "ins": previous["ins"],
                }

                # Insertion
                previous = dp[i][j - 1]

                insertion = {
                    "cost": previous["cost"] + 1,
                    "sub": previous["sub"],
                    "del": previous["del"],
                    "ins": previous["ins"] + 1,
                }

                candidates = [
                    substitution,
                    deletion,
                    insertion,
                ]

                best = min(
                    candidates,
                    key=lambda x: (
                        x["cost"],
                        x["ins"],
                        x["del"],
                    ),
                )

            dp[i][j] = best

    result = dp[n][m]

    return (
        result["sub"],
        result["del"],
        result["ins"],
        result["cost"],
    )


def calculate_wer(reference: str, hypothesis: str):
    normalized_reference = normalize_arabic(reference)
    normalized_hypothesis = normalize_arabic(hypothesis)

    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()

    substitutions, deletions, insertions, total_errors = (
        levenshtein_words(
            reference_words,
            hypothesis_words
        )
    )

    reference_count = len(reference_words)

    if reference_count == 0:
        wer = 0.0 if len(hypothesis_words) == 0 else 1.0
    else:
        wer = total_errors / reference_count

    return {
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
        "reference_word_count": reference_count,
        "hypothesis_word_count": len(hypothesis_words),
        "WER": wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
    }


# ============================================================
# AUDIO
# ============================================================

def load_audio_segment(
    audio_file: Path,
    start_seconds: float,
    duration_seconds: float,
):
    audio, sample_rate = sf.read(
        audio_file,
        dtype="float32",
        always_2d=False,
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    start_sample = int(start_seconds * sample_rate)
    end_sample = int(
        (start_seconds + duration_seconds) * sample_rate
    )

    clip = audio[start_sample:end_sample]

    actual_duration = len(clip) / sample_rate

    if len(clip) == 0:
        raise RuntimeError(
            f"Empty audio clip: "
            f"{start_seconds} -> "
            f"{start_seconds + duration_seconds}"
        )

    return clip, sample_rate, actual_duration


# ============================================================
# VAD
# ============================================================

def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)

    pcm16 = (
        audio * 32767.0
    ).astype(np.int16)

    return pcm16.tobytes()


def detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
):
    if sample_rate != 16000:
        raise RuntimeError(
            f"Round 11 expects 16 kHz audio. "
            f"Got {sample_rate} Hz."
        )

    vad = webrtcvad.Vad(VAD_MODE)

    frame_samples = int(
        sample_rate * FRAME_MS / 1000
    )

    frame_bytes = frame_samples * 2

    pcm_bytes = float_to_pcm16(audio)

    total_frames = len(pcm_bytes) // frame_bytes

    speech_flags = []

    for index in range(total_frames):

        start = index * frame_bytes
        end = start + frame_bytes

        frame = pcm_bytes[start:end]

        if len(frame) != frame_bytes:
            continue

        is_speech = vad.is_speech(
            frame,
            sample_rate,
        )

        speech_flags.append(is_speech)

    # --------------------------------------------------------
    # Build raw speech runs
    # --------------------------------------------------------

    raw_regions = []

    region_start = None

    for index, is_speech in enumerate(speech_flags):

        if is_speech and region_start is None:
            region_start = index

        elif not is_speech and region_start is not None:

            region_end = index

            raw_regions.append(
                (
                    region_start,
                    region_end,
                )
            )

            region_start = None

    if region_start is not None:
        raw_regions.append(
            (
                region_start,
                len(speech_flags),
            )
        )

    # --------------------------------------------------------
    # Merge nearby speech regions
    # --------------------------------------------------------

    merge_gap_frames = max(
        1,
        int(
            MERGE_GAP_MS / FRAME_MS
        )
    )

    merged_regions = []

    for start_frame, end_frame in raw_regions:

        if not merged_regions:
            merged_regions.append(
                [
                    start_frame,
                    end_frame,
                ]
            )
            continue

        previous_start, previous_end = (
            merged_regions[-1]
        )

        gap = start_frame - previous_end

        if gap <= merge_gap_frames:

            merged_regions[-1][1] = end_frame

        else:

            merged_regions.append(
                [
                    start_frame,
                    end_frame,
                ]
            )

    # --------------------------------------------------------
    # Convert to seconds + padding
    # --------------------------------------------------------

    padding_sec = PADDING_MS / 1000.0
    min_speech_sec = MIN_SPEECH_MS / 1000.0

    regions = []

    total_duration = len(audio) / sample_rate

    for start_frame, end_frame in merged_regions:

        start_sec = (
            start_frame * FRAME_MS / 1000.0
        )

        end_sec = (
            end_frame * FRAME_MS / 1000.0
        )

        duration = end_sec - start_sec

        if duration < min_speech_sec:
            continue

        start_sec = max(
            0.0,
            start_sec - padding_sec,
        )

        end_sec = min(
            total_duration,
            end_sec + padding_sec,
        )

        regions.append(
            {
                "start": start_sec,
                "end": end_sec,
                "duration": end_sec - start_sec,
            }
        )

    return regions


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(speech_regions):
    chunks = []

    for region in speech_regions:

        start = region["start"]
        end = region["end"]

        duration = end - start

        if duration <= MAX_CHUNK_SEC:

            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            )

            continue

        # Long speech region:
        # split into <= 15 sec chunks
        cursor = start

        while cursor < end:

            chunk_end = min(
                cursor + MAX_CHUNK_SEC,
                end,
            )

            chunks.append(
                {
                    "start": cursor,
                    "end": chunk_end,
                    "duration": chunk_end - cursor,
                }
            )

            if chunk_end >= end:
                break

            cursor = (
                chunk_end -
                CHUNK_OVERLAP_SEC
            )

    return chunks


# ============================================================
# REPETITION DETECTION
# ============================================================

def has_repeated_characters(text: str):
    """
    Detects obvious pathological output such as:
        ههههههههههههههههه
        مممممممممممممم
    """

    if not text:
        return False

    compact = re.sub(
        r"\s+",
        "",
        text,
    )

    if len(compact) < 12:
        return False

    max_run = 1
    current_run = 1

    for i in range(1, len(compact)):

        if compact[i] == compact[i - 1]:
            current_run += 1
            max_run = max(
                max_run,
                current_run,
            )
        else:
            current_run = 1

    return max_run >= 12


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_chunk(
    model,
    audio,
    sample_rate,
    chunk_start,
    chunk_end,
):
    start_sample = int(
        chunk_start * sample_rate
    )

    end_sample = int(
        chunk_end * sample_rate
    )

    chunk_audio = audio[
        start_sample:end_sample
    ]

    if len(chunk_audio) == 0:
        return None

    result = model.transcribe(
        chunk_audio,
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
        logprob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        initial_prompt=None,
        verbose=False,
    )

    segments = []

    for segment in result.get("segments", []):

        text = segment.get("text", "").strip()

        if not text:
            continue

        global_start = (
            chunk_start +
            float(segment["start"])
        )

        global_end = (
            chunk_start +
            float(segment["end"])
        )

        segments.append(
            {
                "start": global_start,
                "end": global_end,
                "text": text,
                "avg_logprob": float(
                    segment.get(
                        "avg_logprob",
                        0.0,
                    )
                ),
                "compression_ratio": float(
                    segment.get(
                        "compression_ratio",
                        0.0,
                    )
                ),
                "no_speech_prob": float(
                    segment.get(
                        "no_speech_prob",
                        0.0,
                    )
                ),
            }
        )

    return segments


# ============================================================
# SAMPLE EVALUATION
# ============================================================

def evaluate_sample(
    model,
    sample,
):
    name = sample["name"]

    print()
    print("=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)

    audio, sample_rate, actual_duration = (
        load_audio_segment(
            AUDIO_FILE,
            sample["start_seconds"],
            sample["duration_seconds"],
        )
    )

    print(
        f"Audio duration : "
        f"{actual_duration:.2f}s"
    )

    print("Running VAD...")

    speech_regions = detect_speech_regions(
        audio,
        sample_rate,
    )

    print(
        f"Speech regions : "
        f"{len(speech_regions)}"
    )

    chunks = create_chunks(
        speech_regions
    )

    print(
        f"Chunks         : "
        f"{len(chunks)}"
    )

    all_segments = []

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"  Chunk {index:02d}: "
            f"{chunk['start']:.2f}s -> "
            f"{chunk['end']:.2f}s"
        )

        segments = transcribe_chunk(
            model,
            audio,
            sample_rate,
            chunk["start"],
            chunk["end"],
        )

        if segments:
            all_segments.extend(
                segments
            )

    # --------------------------------------------------------
    # Sort by timestamp
    # --------------------------------------------------------

    all_segments.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    # --------------------------------------------------------
    # Build transcript
    # --------------------------------------------------------

    transcript_parts = [
        segment["text"]
        for segment in all_segments
        if segment["text"].strip()
    ]

    transcript = " ".join(
        transcript_parts
    ).strip()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    compression_ratios = [
        s["compression_ratio"]
        for s in all_segments
    ]

    logprobs = [
        s["avg_logprob"]
        for s in all_segments
    ]

    no_speech_probs = [
        s["no_speech_prob"]
        for s in all_segments
    ]

    suspicious_segments = [
        s
        for s in all_segments
        if s["compression_ratio"]
        > COMPRESSION_RATIO_THRESHOLD
    ]

    repeated_segments = [
        s
        for s in all_segments
        if has_repeated_characters(
            s["text"]
        )
    ]

    # --------------------------------------------------------
    # WER
    # --------------------------------------------------------

    wer_metrics = calculate_wer(
        sample["reference"],
        transcript,
    )

    result = {
        "sample": name,
        "audio": {
            "source": str(AUDIO_FILE.name),
            "start_seconds": sample[
                "start_seconds"
            ],
            "duration_seconds": actual_duration,
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
            "chunk_overlap_sec": (
                CHUNK_OVERLAP_SEC
            ),
        },
        "speech_regions": speech_regions,
        "chunks": chunks,
        "reference": sample[
            "reference"
        ],
        "hypothesis": transcript,
        "wer_metrics": wer_metrics,
        "quality_metrics": {
            "segment_count": len(
                all_segments
            ),
            "chunk_count": len(chunks),
            "speech_region_count": len(
                speech_regions
            ),
            "average_compression_ratio": (
                float(
                    np.mean(
                        compression_ratios
                    )
                )
                if compression_ratios
                else 0.0
            ),
            "maximum_compression_ratio": (
                float(
                    np.max(
                        compression_ratios
                    )
                )
                if compression_ratios
                else 0.0
            ),
            "average_logprob": (
                float(
                    np.mean(logprobs)
                )
                if logprobs
                else 0.0
            ),
            "average_no_speech_probability": (
                float(
                    np.mean(
                        no_speech_probs
                    )
                )
                if no_speech_probs
                else 0.0
            ),
            "suspicious_segment_count": len(
                suspicious_segments
            ),
            "repeated_character_segment_count": (
                len(repeated_segments)
            ),
        },
        "segments": all_segments,
        "suspicious_segments": (
            suspicious_segments
        ),
        "repeated_character_segments": (
            repeated_segments
        ),
    }

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(f"RESULT: {name}")
    print("-" * 70)

    print(
        f"Reference words : "
        f"{wer_metrics['reference_word_count']}"
    )

    print(
        f"Hypothesis words: "
        f"{wer_metrics['hypothesis_word_count']}"
    )

    print(
        f"WER             : "
        f"{wer_metrics['WER'] * 100:.2f}%"
    )

    print(
        f"Substitutions   : "
        f"{wer_metrics['substitutions']}"
    )

    print(
        f"Deletions       : "
        f"{wer_metrics['deletions']}"
    )

    print(
        f"Insertions      : "
        f"{wer_metrics['insertions']}"
    )

    print(
        f"Total errors    : "
        f"{wer_metrics['total_errors']}"
    )

    print(
        f"Avg compression : "
        f"{result['quality_metrics']['average_compression_ratio']:.3f}"
    )

    print(
        f"Max compression : "
        f"{result['quality_metrics']['maximum_compression_ratio']:.3f}"
    )

    print(
        f"Suspicious      : "
        f"{result['quality_metrics']['suspicious_segment_count']}"
    )

    print(
        f"Repeated chars  : "
        f"{result['quality_metrics']['repeated_character_segment_count']}"
    )

    print()
    print("REFERENCE:")
    print(sample["reference"])

    print()
    print("HYPOTHESIS:")
    print(transcript)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 11 — VAD + CHUNKING WER VALIDATION")
    print("=" * 70)

    print()
    print(f"Audio file : {AUDIO_FILE}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Device     : {DEVICE}")
    print(f"Language   : {LANGUAGE}")
    print(f"Beam size  : {BEAM_SIZE}")
    print(f"VAD mode   : {VAD_MODE}")
    print(
        f"Max chunk  : {MAX_CHUNK_SEC}s"
    )

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{AUDIO_FILE}"
        )

    # --------------------------------------------------------
    # Load Whisper once
    # --------------------------------------------------------

    print()
    print("Loading Whisper large-v3...")
    print(
        "CPU mode. This may take some time."
    )

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
    )

    print("Whisper loaded successfully.")

    # --------------------------------------------------------
    # Run all samples
    # --------------------------------------------------------

    results = []

    for sample in GROUND_TRUTH:

        result = evaluate_sample(
            model,
            sample,
        )

        results.append(result)

    # --------------------------------------------------------
    # Corpus metrics
    # --------------------------------------------------------

    total_reference_words = sum(
        result["wer_metrics"][
            "reference_word_count"
        ]
        for result in results
    )

    total_errors = sum(
        result["wer_metrics"][
            "total_errors"
        ]
        for result in results
    )

    corpus_wer = (
        total_errors /
        total_reference_words
        if total_reference_words
        else 0.0
    )

    average_sample_wer = (
        np.mean(
            [
                result["wer_metrics"]["WER"]
                for result in results
            ]
        )
        if results
        else 0.0
    )

    # --------------------------------------------------------
    # Compare to previous best
    # --------------------------------------------------------

    PREVIOUS_BEST_WER = 0.475

    improvement = (
        PREVIOUS_BEST_WER -
        corpus_wer
    )

    improvement_percentage_points = (
        improvement * 100
    )

    if corpus_wer < PREVIOUS_BEST_WER:
        verdict = (
            "PASS — Round 11 beats "
            "the previous best WER."
        )
    elif abs(corpus_wer - PREVIOUS_BEST_WER) < 1e-9:
        verdict = (
            "TIE — Round 11 matches "
            "the previous best WER."
        )
    else:
        verdict = (
            "NO WER IMPROVEMENT — "
            "Round 11 is worse than "
            "the previous best WER."
        )

    # --------------------------------------------------------
    # Final result object
    # --------------------------------------------------------

    output = {
        "experiment": "Round 11",
        "description": (
            "WER validation of the exact "
            "Round 10B VAD + chunking "
            "architecture against the "
            "same Ground Truth used in "
            "the previous best evaluation."
        ),
        "previous_best": {
            "experiment": "Round 9A",
            "corpus_WER": PREVIOUS_BEST_WER,
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
            "chunk_overlap_sec": (
                CHUNK_OVERLAP_SEC
            ),
        },
        "samples": results,
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
                PREVIOUS_BEST_WER
            ),
            "previous_best_WER_percent": (
                PREVIOUS_BEST_WER * 100
            ),
            "improvement_percentage_points": (
                improvement_percentage_points
            ),
            "verdict": verdict,
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROUND 11 FINAL RESULT")
    print("=" * 70)

    print()
    print(
        f"Previous best WER : "
        f"{PREVIOUS_BEST_WER * 100:.2f}%"
    )

    print(
        f"Round 11 WER      : "
        f"{corpus_wer * 100:.2f}%"
    )

    print(
        f"Difference        : "
        f"{improvement_percentage_points:+.2f} "
        f"percentage points"
    )

    print(
        f"Reference words   : "
        f"{total_reference_words}"
    )

    print(
        f"Total errors      : "
        f"{total_errors}"
    )

    print()
    print(verdict)

    print()
    print(
        f"Saved results to:\n"
        f"{OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()