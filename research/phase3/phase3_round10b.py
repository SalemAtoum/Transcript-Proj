import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import whisper
import webrtcvad


# ============================================================
# ROUND 10B
# Speech-aware chunking for long-form Whisper stability
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
# VAD configuration
# ------------------------------------------------------------

VAD_MODE = 2

# WebRTC VAD supports 10, 20, or 30 ms frames.
FRAME_MS = 30

# Add a little audio before/after detected speech.
PADDING_MS = 300

# Ignore extremely tiny speech fragments.
MIN_SPEECH_MS = 500

# Merge speech regions separated by short silence.
MERGE_GAP_MS = 600

# Maximum Whisper chunk duration.
MAX_CHUNK_SEC = 15.0

# Small overlap when a long speech region has to be split.
CHUNK_OVERLAP_SEC = 0.30


# ============================================================
# INPUT SAMPLES
# ============================================================

SAMPLES = {
    "sample_A_early": {
        "file": "sample_A_early.wav",
        "start": 0.0,
        "end": 120.0,
    },
    "sample_B_middle": {
        "file": "sample_B_middle.wav",
        "start": 1200.0,
        "end": 1320.0,
    },
    "sample_C_late": {
        "file": "sample_C_late.wav",
        "start": 1800.0,
        "end": 1920.0,
    },
}


OUTPUT_DIR = Path("round10b_results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "phase3_round10b_results.json"


# ============================================================
# HELPERS
# ============================================================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def repeated_character_ratio(text: str) -> float:
    """
    Measures how much of the text consists of repeated
    copies of the same character.

    Example:
        ههههههههههههه
    """

    text = normalize_spaces(text)

    if not text:
        return 0.0

    counts = {}
    for ch in text:
        if not ch.isspace():
            counts[ch] = counts.get(ch, 0) + 1

    if not counts:
        return 0.0

    largest = max(counts.values())
    total = sum(counts.values())

    return largest / total


def is_suspicious_segment(segment: dict) -> bool:
    compression = segment.get("compression_ratio")
    avg_logprob = segment.get("avg_logprob")
    no_speech = segment.get("no_speech_prob")

    if compression is not None and compression > COMPRESSION_RATIO_THRESHOLD:
        return True

    if avg_logprob is not None and avg_logprob < LOGPROB_THRESHOLD:
        return True

    if no_speech is not None and no_speech > NO_SPEECH_THRESHOLD:
        return True

    return False


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path: Path):
    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sample_rate != 16000:
        raise RuntimeError(
            f"{path} has sample rate {sample_rate}, expected 16000 Hz."
        )

    audio = np.asarray(audio, dtype=np.float32)

    return audio, sample_rate


# ============================================================
# WEBRTC VAD
# ============================================================

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767.0).astype(np.int16)

    return pcm.tobytes()


def detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
):
    """
    Detect speech regions using WebRTC VAD.

    This is NOT Whisper.
    It only determines where speech-like activity exists.
    """

    if sample_rate != 16000:
        raise ValueError("WebRTC VAD requires 16 kHz audio.")

    frame_samples = int(sample_rate * FRAME_MS / 1000)

    pcm_bytes = float32_to_pcm16(audio)

    vad = webrtcvad.Vad(VAD_MODE)

    frames = []

    total_samples = len(audio)

    frame_index = 0

    while True:
        start_sample = frame_index * frame_samples
        end_sample = start_sample + frame_samples

        if end_sample > total_samples:
            break

        frame_audio = audio[start_sample:end_sample]

        frame_bytes = float32_to_pcm16(frame_audio)

        speech = vad.is_speech(frame_bytes, sample_rate)

        frames.append({
            "index": frame_index,
            "start": start_sample / sample_rate,
            "end": end_sample / sample_rate,
            "speech": bool(speech),
        })

        frame_index += 1

    if not frames:
        return []

    # --------------------------------------------------------
    # Convert speech frames into raw speech runs
    # --------------------------------------------------------

    raw_regions = []

    current_start = None
    current_end = None

    for frame in frames:

        if frame["speech"]:

            if current_start is None:
                current_start = frame["start"]

            current_end = frame["end"]

        else:

            if current_start is not None:

                raw_regions.append(
                    (
                        current_start,
                        current_end,
                    )
                )

                current_start = None
                current_end = None

    if current_start is not None:
        raw_regions.append(
            (
                current_start,
                current_end,
            )
        )

    # --------------------------------------------------------
    # Remove tiny speech regions
    # --------------------------------------------------------

    min_speech_sec = MIN_SPEECH_MS / 1000.0

    filtered_regions = [
        (start, end)
        for start, end in raw_regions
        if (end - start) >= min_speech_sec
    ]

    # --------------------------------------------------------
    # Merge nearby speech regions
    # --------------------------------------------------------

    merge_gap_sec = MERGE_GAP_MS / 1000.0

    merged = []

    for start, end in filtered_regions:

        if not merged:
            merged.append([start, end])
            continue

        previous_start, previous_end = merged[-1]

        gap = start - previous_end

        if gap <= merge_gap_sec:

            merged[-1][1] = end

        else:

            merged.append([start, end])

    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    padding_sec = PADDING_MS / 1000.0

    padded = []

    for start, end in merged:

        start = max(0.0, start - padding_sec)
        end = min(len(audio) / sample_rate, end + padding_sec)

        padded.append([start, end])

    return padded


# ============================================================
# SPLIT LONG SPEECH REGIONS
# ============================================================

def split_long_regions(
    regions,
    max_chunk_sec,
):
    chunks = []

    overlap = CHUNK_OVERLAP_SEC

    for start, end in regions:

        duration = end - start

        if duration <= max_chunk_sec:

            chunks.append(
                {
                    "start": start,
                    "end": end,
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
                }
            )

            if chunk_end >= end:
                break

            next_start = chunk_end - overlap

            if next_start <= current:
                next_start = chunk_end

            current = next_start

    return chunks


# ============================================================
# TRANSCRIBE ONE CHUNK
# ============================================================

def transcribe_chunk(
    model,
    audio: np.ndarray,
    sample_rate: int,
    chunk_start: float,
    chunk_end: float,
):

    start_sample = int(chunk_start * sample_rate)
    end_sample = int(chunk_end * sample_rate)

    chunk_audio = audio[start_sample:end_sample]

    if len(chunk_audio) == 0:
        return {
            "text": "",
            "segments": [],
        }

    result = model.transcribe(
        chunk_audio,
        language=LANGUAGE,
        task=TASK,
        fp16=FP16,
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURE,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=LOGPROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        initial_prompt=None,
        verbose=False,
    )

    corrected_segments = []

    for segment in result.get("segments", []):

        corrected = dict(segment)

        corrected["start"] = (
            float(segment["start"]) + chunk_start
        )

        corrected["end"] = (
            float(segment["end"]) + chunk_start
        )

        corrected_segments.append(corrected)

    return {
        "text": normalize_spaces(result.get("text", "")),
        "segments": corrected_segments,
    }


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

def process_sample(
    model,
    sample_name: str,
    sample_info: dict,
):

    print()
    print("=" * 70)
    print(f"PROCESSING: {sample_name}")
    print("=" * 70)

    path = Path(sample_info["file"])

    if not path.exists():
        raise FileNotFoundError(
            f"Missing sample file: {path}"
        )

    audio, sample_rate = load_audio(path)

    print(
        f"Audio duration: {len(audio) / sample_rate:.2f} sec"
    )

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    speech_regions = detect_speech_regions(
        audio,
        sample_rate,
    )

    print(
        f"Detected speech regions: {len(speech_regions)}"
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunks = split_long_regions(
        speech_regions,
        MAX_CHUNK_SEC,
    )

    print(
        f"Whisper chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    all_segments = []

    chunk_results = []

    for index, chunk in enumerate(chunks, start=1):

        start = chunk["start"]
        end = chunk["end"]

        duration = end - start

        print(
            f"\nChunk {index}/{len(chunks)} "
            f"[{start:.2f} -> {end:.2f}] "
            f"({duration:.2f}s)"
        )

        result = transcribe_chunk(
            model=model,
            audio=audio,
            sample_rate=sample_rate,
            chunk_start=start,
            chunk_end=end,
        )

        chunk_segments = result["segments"]

        all_segments.extend(chunk_segments)

        suspicious_count = sum(
            1
            for segment in chunk_segments
            if is_suspicious_segment(segment)
        )

        chunk_results.append(
            {
                "chunk_index": index,
                "start": start,
                "end": end,
                "duration": duration,
                "text": result["text"],
                "segment_count": len(chunk_segments),
                "suspicious_segments": suspicious_count,
            }
        )

        print(
            f"Segments: {len(chunk_segments)}"
        )

        print(
            f"Suspicious: {suspicious_count}"
        )

        print(
            f"Text: {result['text']}"
        )

    # --------------------------------------------------------
    # Sort by global timestamp
    # --------------------------------------------------------

    all_segments.sort(
        key=lambda x: x.get("start", 0.0)
    )

    # --------------------------------------------------------
    # Build transcript
    # --------------------------------------------------------

    transcript_parts = []

    for segment in all_segments:

        text = normalize_spaces(
            segment.get("text", "")
        )

        if text:
            transcript_parts.append(text)

    transcript = normalize_spaces(
        " ".join(transcript_parts)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    compression_values = [
        float(s["compression_ratio"])
        for s in all_segments
        if s.get("compression_ratio") is not None
    ]

    logprob_values = [
        float(s["avg_logprob"])
        for s in all_segments
        if s.get("avg_logprob") is not None
    ]

    no_speech_values = [
        float(s["no_speech_prob"])
        for s in all_segments
        if s.get("no_speech_prob") is not None
    ]

    suspicious_segments = [
        s
        for s in all_segments
        if is_suspicious_segment(s)
    ]

    repeated_character_segments = []

    for segment in all_segments:

        text = normalize_spaces(
            segment.get("text", "")
        )

        ratio = repeated_character_ratio(text)

        if ratio >= 0.80 and len(text) >= 5:

            repeated_character_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": text,
                    "repeated_character_ratio": ratio,
                }
            )

    metrics = {
        "segment_count": len(all_segments),

        "chunk_count": len(chunks),

        "speech_region_count": len(
            speech_regions
        ),

        "average_compression_ratio": (
            float(np.mean(compression_values))
            if compression_values
            else None
        ),

        "maximum_compression_ratio": (
            float(np.max(compression_values))
            if compression_values
            else None
        ),

        "average_logprob": (
            float(np.mean(logprob_values))
            if logprob_values
            else None
        ),

        "average_no_speech_probability": (
            float(np.mean(no_speech_values))
            if no_speech_values
            else None
        ),

        "suspicious_segment_count": len(
            suspicious_segments
        ),

        "repeated_character_segment_count": len(
            repeated_character_segments
        ),
    }

    print()
    print("-" * 70)
    print(f"RESULT: {sample_name}")
    print("-" * 70)

    print(
        f"Chunks: {metrics['chunk_count']}"
    )

    print(
        f"Segments: {metrics['segment_count']}"
    )

    print(
        f"Average compression: "
        f"{metrics['average_compression_ratio']:.4f}"
        if metrics["average_compression_ratio"] is not None
        else "Average compression: N/A"
    )

    print(
        f"Maximum compression: "
        f"{metrics['maximum_compression_ratio']:.4f}"
        if metrics["maximum_compression_ratio"] is not None
        else "Maximum compression: N/A"
    )

    print(
        f"Suspicious segments: "
        f"{metrics['suspicious_segment_count']}"
    )

    print(
        f"Repeated-character segments: "
        f"{metrics['repeated_character_segment_count']}"
    )

    return {
        "sample": sample_name,
        "input_file": str(path),
        "requested_range": {
            "start": sample_info["start"],
            "end": sample_info["end"],
        },
        "configuration": {
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
                "start": start,
                "end": end,
                "duration": end - start,
            }
            for start, end in speech_regions
        ],
        "chunks": chunk_results,
        "segments": all_segments,
        "suspicious_segments": suspicious_segments,
        "repeated_character_segments": repeated_character_segments,
        "metrics": metrics,
        "transcript": transcript,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROUND 10B")
    print("Speech-aware chunking + independent Whisper decoding")
    print("=" * 70)

    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Beam size: {BEAM_SIZE}")
    print(f"Temperature: {TEMPERATURE}")
    print(
        f"Condition previous text: "
        f"{CONDITION_ON_PREVIOUS_TEXT}"
    )

    print()
    print("Loading Whisper model...")

    model = whisper.load_model(
        MODEL_NAME,
        device=DEVICE,
    )

    print("Whisper model loaded.")

    results = {
        "round": "10B",
        "description": (
            "Speech-aware segmentation using WebRTC VAD "
            "followed by independent Whisper chunks."
        ),
        "samples": {},
    }

    for sample_name, sample_info in SAMPLES.items():

        result = process_sample(
            model=model,
            sample_name=sample_name,
            sample_info=sample_info,
        )

        results["samples"][sample_name] = result

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

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("ROUND 10B SUMMARY")
    print("=" * 70)

    for sample_name, result in results["samples"].items():

        metrics = result["metrics"]

        print()
        print(sample_name)

        print(
            f"  Chunks: "
            f"{metrics['chunk_count']}"
        )

        print(
            f"  Segments: "
            f"{metrics['segment_count']}"
        )

        print(
            f"  Avg compression: "
            f"{metrics['average_compression_ratio']:.4f}"
            if metrics["average_compression_ratio"] is not None
            else "  Avg compression: N/A"
        )

        print(
            f"  Max compression: "
            f"{metrics['maximum_compression_ratio']:.4f}"
            if metrics["maximum_compression_ratio"] is not None
            else "  Max compression: N/A"
        )

        print(
            f"  Suspicious segments: "
            f"{metrics['suspicious_segment_count']}"
        )

        print(
            f"  Repeated-character segments: "
            f"{metrics['repeated_character_segment_count']}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        f"Results saved to: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()