import os
import subprocess
import json

import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_AUDIO = "meeting.wav"
OUTPUT_AUDIO = "meeting_processed.wav"
METADATA_FILE = "audio_metadata.json"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


# ============================================================
# FFmpeg CHECK
# ============================================================

def check_ffmpeg():
    """
    Check whether FFmpeg is installed and available in PATH.
    """

    print("=" * 60)
    print("FFMPEG CHECK")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        first_line = result.stdout.splitlines()[0]

        print(first_line)
        print("FFmpeg is available.")
        print()

        return True

    except FileNotFoundError:
        print("ERROR: FFmpeg was not found in PATH.")
        return False

    except subprocess.CalledProcessError:
        print("ERROR: FFmpeg could not be executed.")
        return False


# ============================================================
# INPUT AUDIO CHECK
# ============================================================

def validate_input_audio(audio_path):
    """
    Check that the original input audio exists
    and can be read by SoundFile.
    """

    print("=" * 60)
    print("INPUT AUDIO CHECK")
    print("=" * 60)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Input audio was not found: {audio_path}"
        )

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    print(f"File      : {audio_path}")
    print(f"Size      : {file_size_mb:.2f} MB")

    try:
        info = sf.info(audio_path)

        print(f"Format    : {info.format}")
        print(f"Subtype   : {info.subtype}")
        print(f"Sample rate: {info.samplerate} Hz")
        print(f"Channels  : {info.channels}")
        print(f"Duration  : {info.duration:.2f} seconds")

    except Exception as error:
        raise RuntimeError(
            f"Input audio could not be read: {error}"
        )

    print("Input audio is readable.")
    print()


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio(input_path, output_path):
    """
    Convert the input audio into the standard format used
    by the rest of the pipeline.

    Output:
        WAV
        Mono
        16 kHz
        PCM 16-bit
    """

    print("=" * 60)
    print("AUDIO STANDARDIZATION")
    print("=" * 60)

    command = [
        "ffmpeg",

        # Overwrite output if it already exists
        "-y",

        # Input
        "-i",
        input_path,

        # Audio only
        "-vn",

        # Mono
        "-ac",
        str(TARGET_CHANNELS),

        # 16 kHz
        "-ar",
        str(TARGET_SAMPLE_RATE),

        # PCM 16-bit WAV
        "-c:a",
        "pcm_s16le",

        output_path
    ]

    print("Converting audio...")
    print()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)

        raise RuntimeError(
            "FFmpeg failed during audio standardization."
        )

    print("Audio standardization completed.")
    print()


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_output_audio(audio_path):
    """
    Validate that the processed audio has the exact
    properties required by the next stages.
    """

    print("=" * 60)
    print("OUTPUT AUDIO VALIDATION")
    print("=" * 60)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Processed audio was not created: {audio_path}"
        )

    try:
        info = sf.info(audio_path)

    except Exception as error:
        raise RuntimeError(
            f"Processed audio could not be read: {error}"
        )

    sample_rate = info.samplerate
    channels = info.channels
    frames = info.frames
    duration = info.duration
    audio_format = info.format
    subtype = info.subtype

    print(f"File          : {audio_path}")
    print(f"Format        : {audio_format}")
    print(f"Subtype       : {subtype}")
    print(f"Sample rate   : {sample_rate} Hz")
    print(f"Channels      : {channels}")
    print(f"Frames        : {frames}")
    print(f"Duration      : {duration:.2f} seconds")
    print()

    # Sample rate validation
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Invalid sample rate: {sample_rate} Hz. "
            f"Expected {TARGET_SAMPLE_RATE} Hz."
        )

    # Channel validation
    if channels != TARGET_CHANNELS:
        raise ValueError(
            f"Invalid number of channels: {channels}. "
            f"Expected {TARGET_CHANNELS}."
        )

    # Duration validation
    if duration <= 0:
        raise ValueError(
            "Audio duration is zero or invalid."
        )

    # Frame validation
    if frames <= 0:
        raise ValueError(
            "Audio contains no frames."
        )

    # Format validation
    if audio_format != "WAV":
        raise ValueError(
            f"Invalid audio format: {audio_format}. "
            "Expected WAV."
        )

    # PCM validation
    if subtype != "PCM_16":
        raise ValueError(
            f"Invalid audio subtype: {subtype}. "
            "Expected PCM_16."
        )

    print("Audio validation: PASS")
    print()


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(audio_path):
    """
    Save information about the processed audio.
    """

    info = sf.info(audio_path)

    metadata = {
        "source_file": INPUT_AUDIO,
        "processed_file": OUTPUT_AUDIO,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": round(info.duration, 3)
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Metadata saved to: {METADATA_FILE}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SWBS MEETING TRANSCRIPTION")
    print("PHASE 2 - AUDIO PREPROCESSING")
    print("=" * 60)
    print()

    # 1. Check FFmpeg
    if not check_ffmpeg():
        return

    # 2. Check original audio
    validate_input_audio(
        INPUT_AUDIO
    )

    # 3. Standardize audio
    convert_audio(
        INPUT_AUDIO,
        OUTPUT_AUDIO
    )

    # 4. Validate processed audio
    validate_output_audio(
        OUTPUT_AUDIO
    )

    # 5. Save metadata
    save_metadata(
        OUTPUT_AUDIO
    )

    print("=" * 60)
    print("PHASE 2 - AUDIO PREPROCESSING COMPLETE")
    print("=" * 60)
    print()
    print(f"Input     : {INPUT_AUDIO}")
    print(f"Output    : {OUTPUT_AUDIO}")
    print(f"Metadata  : {METADATA_FILE}")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()