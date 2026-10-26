from pathlib import Path
import json
import time
import math

import numpy as np
import soundfile as sf
import whisper


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

AUDIO_PATH = PROJECT_ROOT / "meeting_processed.wav"
OUTPUT_PATH = SCRIPT_DIR / "canonical_transcript.json"


# ============================================================
# WHISPER CONFIGURATION
# ============================================================

MODEL_NAME = "large-v3"
DEVICE = "cpu"

LANGUAGE = "ar"
TASK = "transcribe"

BEAM_SIZE = 5
TEMPERATURE = 0

CONDITION_ON_PREVIOUS_TEXT = False

COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6

WORD_TIMESTAMPS = True

# Maximum chunk length
MAX_CHUNK_SECONDS = 15.0


# ============================================================
# HELPERS
# ============================================================

def format_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm
    """

    milliseconds = int(round((seconds % 1) * 1000))

    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# LOAD AUDIO
# ============================================================

print("=" * 70)
print("PHASE 6.1 — CANONICAL TRANSCRIPT GENERATION")
print("=" * 70)

print("\n[1] Checking audio...")

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_PATH}"
    )

audio_info = sf.info(str(AUDIO_PATH))

sample_rate = audio_info.samplerate
channels = audio_info.channels
frames = audio_info.frames
duration = frames / sample_rate

print(f"Audio: {AUDIO_PATH}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Channels: {channels}")
print(f"Frames: {frames:,}")
print(f"Duration: {duration:.2f} sec")
print(f"Duration: {duration / 60:.2f} min")


if sample_rate != 16000:
    raise ValueError(
        f"Expected 16000 Hz audio, got {sample_rate} Hz."
    )

if channels != 1:
    raise ValueError(
        f"Expected mono audio, got {channels} channels."
    )


# ============================================================
# LOAD WHISPER
# ============================================================

print("\n[2] Loading Whisper model...")

model_start = time.time()

model = whisper.load_model(
    MODEL_NAME,
    device=DEVICE
)

model_load_time = time.time() - model_start

print(
    f"Whisper model loaded in "
    f"{model_load_time:.2f} sec"
)


# ============================================================
# LOAD AUDIO INTO MEMORY
# ============================================================

print("\n[3] Loading audio into memory...")

audio, sr = sf.read(
    str(AUDIO_PATH),
    dtype="float32"
)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

if sr != 16000:
    raise ValueError(
        f"Unexpected sample rate after loading: {sr}"
    )

print(f"Loaded samples: {len(audio):,}")


# ============================================================
# CREATE CHUNKS
# ============================================================

print("\n[4] Creating transcription chunks...")

chunk_size = int(MAX_CHUNK_SECONDS * sr)

chunks = []

start_sample = 0
chunk_index = 0

while start_sample < len(audio):

    end_sample = min(
        start_sample + chunk_size,
        len(audio)
    )

    chunk = audio[start_sample:end_sample]

    start_time = start_sample / sr
    end_time = end_sample / sr

    chunks.append(
        {
            "chunk_index": chunk_index,
            "start": start_time,
            "end": end_time,
            "audio": chunk,
        }
    )

    chunk_index += 1
    start_sample = end_sample


print(f"Total chunks: {len(chunks)}")
print(
    f"Maximum chunk duration: "
    f"{MAX_CHUNK_SECONDS:.1f} sec"
)


# ============================================================
# TRANSCRIPTION
# ============================================================

print("\n[5] Starting transcription...")
print("This can take a long time on CPU.")
print()

transcription_start = time.time()

all_segments = []
all_words = []

successful_chunks = 0
failed_chunks = 0

suspicious_segments = 0


for chunk_number, chunk_data in enumerate(chunks, start=1):

    chunk_index = chunk_data["chunk_index"]
    chunk_start = chunk_data["start"]
    chunk_end = chunk_data["end"]
    chunk_audio = chunk_data["audio"]

    print(
        f"[Chunk {chunk_number}/{len(chunks)}] "
        f"{format_time(chunk_start)} → "
        f"{format_time(chunk_end)}"
    )

    try:

        result = model.transcribe(
            chunk_audio,

            language=LANGUAGE,
            task=TASK,

            fp16=False,

            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,

            condition_on_previous_text=
                CONDITION_ON_PREVIOUS_TEXT,

            compression_ratio_threshold=
                COMPRESSION_RATIO_THRESHOLD,

            logprob_threshold=
                LOGPROB_THRESHOLD,

            no_speech_threshold=
                NO_SPEECH_THRESHOLD,

            word_timestamps=WORD_TIMESTAMPS,

            verbose=False
        )

        successful_chunks += 1

        # ----------------------------------------------------
        # SEGMENTS
        # ----------------------------------------------------

        for segment in result.get("segments", []):

            local_start = safe_float(
                segment.get("start")
            )

            local_end = safe_float(
                segment.get("end")
            )

            if local_start is None or local_end is None:
                continue

            global_start = chunk_start + local_start
            global_end = chunk_start + local_end

            text = (
                segment.get("text") or ""
            ).strip()

            if not text:
                continue

            compression_ratio = safe_float(
                segment.get("compression_ratio")
            )

            avg_logprob = safe_float(
                segment.get("avg_logprob")
            )

            no_speech_prob = safe_float(
                segment.get("no_speech_prob")
            )

            if (
                compression_ratio is not None
                and compression_ratio >
                    COMPRESSION_RATIO_THRESHOLD
            ):
                suspicious_segments += 1

            all_segments.append(
                {
                    "segment_id": len(all_segments),

                    "chunk_index": chunk_index,

                    "start": round(
                        global_start,
                        3
                    ),

                    "end": round(
                        global_end,
                        3
                    ),

                    "start_time":
                        format_time(global_start),

                    "end_time":
                        format_time(global_end),

                    "text": text,

                    "avg_logprob":
                        avg_logprob,

                    "compression_ratio":
                        compression_ratio,

                    "no_speech_prob":
                        no_speech_prob,
                }
            )

            # ------------------------------------------------
            # WORD TIMESTAMPS
            # ------------------------------------------------

            for word in segment.get(
                "words",
                []
            ):

                word_start = safe_float(
                    word.get("start")
                )

                word_end = safe_float(
                    word.get("end")
                )

                word_text = (
                    word.get("word") or ""
                ).strip()

                if (
                    word_start is None
                    or word_end is None
                    or not word_text
                ):
                    continue

                global_word_start = (
                    chunk_start + word_start
                )

                global_word_end = (
                    chunk_start + word_end
                )

                all_words.append(
                    {
                        "word_id": len(all_words),

                        "chunk_index":
                            chunk_index,

                        "start": round(
                            global_word_start,
                            3
                        ),

                        "end": round(
                            global_word_end,
                            3
                        ),

                        "start_time":
                            format_time(
                                global_word_start
                            ),

                        "end_time":
                            format_time(
                                global_word_end
                            ),

                        "word":
                            word_text,
                    }
                )

        print(
            f"    segments added: "
            f"{len(result.get('segments', []))}"
        )

    except Exception as exc:

        failed_chunks += 1

        print(
            f"    ERROR: {exc}"
        )


# ============================================================
# FINAL STATISTICS
# ============================================================

transcription_time = (
    time.time() - transcription_start
)

print("\n[6] Transcription completed.")

print(
    f"Successful chunks: {successful_chunks}"
)

print(
    f"Failed chunks: {failed_chunks}"
)

print(
    f"Total segments: {len(all_segments)}"
)

print(
    f"Total words: {len(all_words)}"
)

print(
    f"Suspicious segments: "
    f"{suspicious_segments}"
)

print(
    f"Processing time: "
    f"{transcription_time / 60:.2f} min"
)


# ============================================================
# BUILD OUTPUT
# ============================================================

output = {
    "phase": "6.1",
    "component": "canonical_transcript",

    "audio": str(AUDIO_PATH),

    "audio_metadata": {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_seconds":
            round(duration, 3),
    },

    "whisper_configuration": {
        "model": MODEL_NAME,
        "device": DEVICE,
        "language": LANGUAGE,
        "task": TASK,

        "fp16": False,

        "beam_size":
            BEAM_SIZE,

        "temperature":
            TEMPERATURE,

        "condition_on_previous_text":
            CONDITION_ON_PREVIOUS_TEXT,

        "compression_ratio_threshold":
            COMPRESSION_RATIO_THRESHOLD,

        "logprob_threshold":
            LOGPROB_THRESHOLD,

        "no_speech_threshold":
            NO_SPEECH_THRESHOLD,

        "word_timestamps":
            WORD_TIMESTAMPS,

        "max_chunk_seconds":
            MAX_CHUNK_SECONDS,
    },

    "processing": {
        "total_chunks":
            len(chunks),

        "successful_chunks":
            successful_chunks,

        "failed_chunks":
            failed_chunks,

        "processing_time_seconds":
            round(
                transcription_time,
                2
            ),
    },

    "quality_statistics": {
        "segment_count":
            len(all_segments),

        "word_count":
            len(all_words),

        "suspicious_segments":
            suspicious_segments,
    },

    "segments":
        all_segments,

    "words":
        all_words,
}


# ============================================================
# SAVE
# ============================================================

print("\n[7] Saving canonical transcript...")

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"\nSaved successfully:\n"
    f"{OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("PHASE 6.1 COMPLETE")
print("=" * 70)