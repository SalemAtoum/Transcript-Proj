from pathlib import Path
import os
import json

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_PATH = PROJECT_ROOT / "meeting_processed.wav"

MODEL_NAME = "pyannote/speaker-diarization-3.1"

DEVICE = torch.device("cpu")

FFMPEG_SHARED_BIN = (
    Path(os.environ["LOCALAPPDATA"])
    / "Microsoft"
    / "WinGet"
    / "Packages"
    / "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    / "ffmpeg-9.0.2-full_build-shared"
    / "bin"
)

# Output is saved inside the phase4_diarization folder.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "diarization_result.json"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4 — SPEAKER DIARIZATION")
    print("Baseline Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN was not found in .env"
        )

    print()
    print("HF_TOKEN loaded: True")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print()
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            "WARNING: CUDA is available, "
            "but Phase 4 is configured for CPU."
        )

    # --------------------------------------------------------
    # Audio file
    # --------------------------------------------------------

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    print()
    print(f"Audio: {AUDIO_PATH}")

    # --------------------------------------------------------
    # FFmpeg Shared DLL directory
    # --------------------------------------------------------

    print()
    print("Checking FFmpeg Shared DLL directory...")

    if not FFMPEG_SHARED_BIN.exists():
        raise FileNotFoundError(
            f"FFmpeg Shared bin directory not found:\n"
            f"{FFMPEG_SHARED_BIN}"
        )

    required_dlls = [
        "avcodec-63.dll",
        "avformat-63.dll",
        "avutil-61.dll",
    ]

    for dll_name in required_dlls:

        dll_path = FFMPEG_SHARED_BIN / dll_name

        if not dll_path.exists():
            raise FileNotFoundError(
                f"Required FFmpeg DLL not found:\n{dll_path}"
            )

    print("FFmpeg Shared DLLs: PASS")

    os.add_dll_directory(str(FFMPEG_SHARED_BIN))

    print("FFmpeg DLL directory registered.")

    # --------------------------------------------------------
    # Load audio with SoundFile
    # --------------------------------------------------------

    print()
    print("Loading audio with SoundFile...")
    print("This bypasses TorchCodec file decoding.")

    waveform, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
        always_2d=False,
    )

    print()
    print("SoundFile audio loading: PASS")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio shape: {waveform.shape}")
    print(
        f"Duration: "
        f"{len(waveform) / sample_rate:.2f} seconds"
    )

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz, got {sample_rate} Hz"
        )

    # --------------------------------------------------------
    # Convert waveform to PyTorch tensor
    # --------------------------------------------------------

    waveform_tensor = torch.from_numpy(waveform)

    # SoundFile should already give mono audio,
    # but normalize shape explicitly.
    if waveform_tensor.ndim == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    elif waveform_tensor.ndim == 2:

        # If shape is (samples, channels), transpose.
        if waveform_tensor.shape[0] > waveform_tensor.shape[1]:
            waveform_tensor = waveform_tensor.transpose(0, 1)

    waveform_tensor = waveform_tensor.to(
        device=DEVICE,
        dtype=torch.float32,
    )

    print()
    print("Waveform tensor prepared.")
    print(f"Tensor shape: {tuple(waveform_tensor.shape)}")
    print(f"Tensor dtype: {waveform_tensor.dtype}")
    print(f"Tensor device: {waveform_tensor.device}")

    # --------------------------------------------------------
    # Load PyAnnote
    # --------------------------------------------------------

    print()
    print(f"Loading PyAnnote model:\n{MODEL_NAME}")

    pipeline = Pipeline.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )

    if pipeline is None:
        raise RuntimeError(
            "Failed to load PyAnnote pipeline."
        )

    pipeline.to(DEVICE)

    print()
    print("PyAnnote pipeline loaded successfully.")

    # --------------------------------------------------------
    # Build in-memory audio input
    # --------------------------------------------------------

    audio_input = {
        "waveform": waveform_tensor,
        "sample_rate": sample_rate,
    }

    # --------------------------------------------------------
    # Run diarization
    # --------------------------------------------------------

    print()
    print("Starting diarization...")
    print("CPU processing may take some time.")
    print("Please do not interrupt the process.")

    diarization_output = pipeline(audio_input)

    print()
    print("Diarization completed.")

    # PyAnnote Audio 4.x returns DiarizeOutput.
    # The actual Annotation objects are inside these fields.
    diarization = diarization_output.speaker_diarization
    exclusive_diarization = (
        diarization_output.exclusive_speaker_diarization
    )

    # --------------------------------------------------------
    # SAVE DIARIZATION RESULT IMMEDIATELY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING DIARIZATION RESULT")
    print("=" * 70)

    # Serialize immediately after the expensive pipeline step.
    # This prevents losing the one-hour computation because of
    # an error in later display/post-processing code.
    serialized_result = diarization_output.serialize()

    result_payload = {
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "audio": str(AUDIO_PATH),
        "sample_rate": sample_rate,
        "duration_seconds": len(waveform) / sample_rate,
        "speaker_labels": sorted(diarization.labels()),
        "num_speakers": len(diarization.labels()),
        "diarization": serialized_result.get(
            "diarization",
            serialized_result
        ),
        "exclusive_diarization": serialized_result.get(
            "exclusive_diarization",
            None
        ),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            result_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Result saved to: {OUTPUT_JSON}")

    # --------------------------------------------------------
    # Display segments
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIARIZATION SEGMENTS")
    print("=" * 70)

    speaker_set = set()
    segment_count = 0

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):

        speaker_set.add(speaker)
        segment_count += 1

        print(
            f"{turn.start:10.2f}s → "
            f"{turn.end:10.2f}s | "
            f"{speaker}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIARIZATION SUMMARY")
    print("=" * 70)

    print(f"Total segments: {segment_count}")
    print(f"Detected speakers: {len(speaker_set)}")

    print()
    print("Speaker labels:")

    for speaker in sorted(speaker_set):
        print(f"  - {speaker}")

    print()
    print(f"Saved JSON: {OUTPUT_JSON}")

    print()
    print("=" * 70)
    print("PHASE 4 BASELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
