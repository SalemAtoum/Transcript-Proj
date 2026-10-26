from pathlib import Path
import json
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SPEAKER_DIRS = {
    "sami": SCRIPT_DIR / "sami",
    "salameh": SCRIPT_DIR / "salameh",
}

OUTPUT_DIR = SCRIPT_DIR / "embeddings"

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

DEVICE = "cpu"

TARGET_SAMPLE_RATE = 16000

SUPPORTED_EXTENSIONS = {".wav"}

# Save normalized embeddings
NORMALIZE_EMBEDDINGS = True


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 70)
print("PHASE 5.2 — SPEAKER EMBEDDING EXTRACTION")
print("=" * 70)

print()
print(f"Device: {DEVICE}")
print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")
print(f"Model: {MODEL_NAME}")
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading SpeechBrain ECAPA-TDNN model...")
print("The first run may download the pretrained model.")
print()

model_start = time.time()

encoder = EncoderClassifier.from_hparams(
    source=MODEL_NAME,
    savedir=str(
        SCRIPT_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
    ),
    run_opts={
        "device": DEVICE,
    },   
    local_strategy=LocalStrategy.COPY,


)

model_load_time = time.time() - model_start

print(
    f"Model loaded successfully in "
    f"{model_load_time:.2f} seconds."
)

print()


# ============================================================
# HELPERS
# ============================================================

def load_and_prepare_audio(audio_path: Path):
    """
    Load audio using SoundFile and standardize it to:
        - mono
        - float32
        - 16 kHz
    """

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=False,
    )

    # --------------------------------------------------------
    # Convert stereo -> mono
    # --------------------------------------------------------

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    # --------------------------------------------------------
    # NumPy -> Torch
    # --------------------------------------------------------

    waveform = torch.from_numpy(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )

    # SpeechBrain expects:
    # [batch, time]

    waveform = waveform.unsqueeze(0)

    # --------------------------------------------------------
    # Resample if necessary
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )

        sample_rate = TARGET_SAMPLE_RATE

    return waveform, sample_rate


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding vector.
    """

    norm = np.linalg.norm(vector)

    if norm <= 1e-12:
        return vector

    return vector / norm


def extract_embedding(audio_path: Path):
    """
    Extract one speaker embedding from one enrollment recording.
    """

    waveform, sample_rate = load_and_prepare_audio(
        audio_path
    )

    duration_seconds = waveform.shape[1] / sample_rate

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if duration_seconds < 1.0:

        raise ValueError(
            f"Audio is too short for embedding extraction: "
            f"{duration_seconds:.2f}s"
        )

    # --------------------------------------------------------
    # Move to CPU explicitly
    # --------------------------------------------------------

    waveform = waveform.to(DEVICE)

    # --------------------------------------------------------
    # Extract embedding
    # --------------------------------------------------------

    with torch.no_grad():

        embedding = encoder.encode_batch(
            waveform,
            normalize=NORMALIZE_EMBEDDINGS,
        )

    # --------------------------------------------------------
    # Convert tensor -> NumPy
    # --------------------------------------------------------

    embedding = embedding.squeeze()

    embedding = embedding.detach().cpu().numpy()

    embedding = embedding.astype(
        np.float32
    )

    # Extra normalization for stable cosine similarity
    embedding = l2_normalize(
        embedding
    )

    return (
        embedding,
        sample_rate,
        duration_seconds,
    )


# ============================================================
# PROCESS ALL RECORDINGS
# ============================================================

all_results = []

total_start = time.time()

for speaker_name, speaker_dir in SPEAKER_DIRS.items():

    print("=" * 70)
    print(f"SPEAKER: {speaker_name.upper()}")
    print("=" * 70)

    if not speaker_dir.exists():

        print(
            f"ERROR: Folder does not exist: "
            f"{speaker_dir}"
        )

        continue

    audio_files = sorted(
        file
        for file in speaker_dir.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not audio_files:

        print(
            f"ERROR: No WAV files found in "
            f"{speaker_dir}"
        )

        continue

    for audio_path in audio_files:

        print()
        print(
            f"Processing: {audio_path.name}"
        )

        start_time = time.time()

        try:

            (
                embedding,
                sample_rate,
                duration_seconds,
            ) = extract_embedding(
                audio_path
            )

            elapsed = time.time() - start_time

            # ------------------------------------------------
            # Output file
            # ------------------------------------------------

            output_name = (
                f"{speaker_name}_"
                f"{audio_path.stem}_embedding.npy"
            )

            output_path = (
                OUTPUT_DIR / output_name
            )

            np.save(
                output_path,
                embedding,
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {
                "speaker": speaker_name,
                "source_file": audio_path.name,
                "source_path": str(audio_path),
                "model": MODEL_NAME,
                "device": DEVICE,
                "sample_rate_hz": sample_rate,
                "duration_seconds": round(
                    duration_seconds,
                    3,
                ),
                "embedding_dimension": int(
                    embedding.shape[0]
                ),
                "normalized": True,
                "embedding_file": str(
                    output_path
                ),
                "extraction_time_seconds": round(
                    elapsed,
                    3,
                ),
            }

            metadata_path = (
                OUTPUT_DIR
                / (
                    f"{speaker_name}_"
                    f"{audio_path.stem}_metadata.json"
                )
            )

            with open(
                metadata_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            all_results.append(
                {
                    **metadata,
                    "status": "PASS",
                }
            )

            print(
                f"  Duration: "
                f"{duration_seconds:.2f}s"
            )

            print(
                f"  Sample rate: "
                f"{sample_rate} Hz"
            )

            print(
                f"  Embedding dimension: "
                f"{embedding.shape[0]}"
            )

            print(
                f"  Saved: "
                f"{output_path.name}"
            )

            print(
                f"  Time: "
                f"{elapsed:.2f}s"
            )

            print(
                "  Status: PASS"
            )

        except Exception as e:

            elapsed = time.time() - start_time

            error_message = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"  ERROR: {error_message}"
            )

            all_results.append(
                {
                    "speaker": speaker_name,
                    "source_file": audio_path.name,
                    "source_path": str(
                        audio_path
                    ),
                    "status": "FAIL",
                    "error": error_message,
                    "extraction_time_seconds": round(
                        elapsed,
                        3,
                    ),
                }
            )


# ============================================================
# SAVE REPORT
# ============================================================

total_elapsed = (
    time.time() - total_start
)

passed = sum(
    1
    for result in all_results
    if result["status"] == "PASS"
)

failed = sum(
    1
    for result in all_results
    if result["status"] == "FAIL"
)

report = {
    "phase": "5.2",
    "name": "Speaker Embedding Extraction",
    "model": MODEL_NAME,
    "device": DEVICE,
    "target_sample_rate_hz": TARGET_SAMPLE_RATE,
    "normalized_embeddings": True,
    "summary": {
        "total_files": len(all_results),
        "passed_files": passed,
        "failed_files": failed,
        "total_processing_time_seconds": round(
            total_elapsed,
            3,
        ),
    },
    "results": all_results,
}


REPORT_PATH = (
    SCRIPT_DIR
    / "embedding_extraction_report.json"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PHASE 5.2 SUMMARY")
print("=" * 70)

print(
    f"Total files: {len(all_results)}"
)

print(
    f"Successful: {passed}"
)

print(
    f"Failed: {failed}"
)

print(
    f"Total processing time: "
    f"{total_elapsed:.2f}s"
)

print()

if failed == 0 and passed == 6:

    print(
        "FINAL STATUS: PASS"
    )

else:

    print(
        "FINAL STATUS: FAIL"
    )

print()
print(
    f"Report saved to:\n"
    f"{REPORT_PATH}"
)

print(
    f"Embeddings saved to:\n"
    f"{OUTPUT_DIR}"
)

print("=" * 70)