import os
import sys
import subprocess

import numpy as np
import torch
import soundfile as sf
from dotenv import load_dotenv


def check_python():
    print("=" * 60)
    print("PYTHON")
    print("=" * 60)

    print(f"Python version : {sys.version}")
    print(f"Python path    : {sys.executable}")

    if sys.version_info[:2] != (3, 11):
        print("WARNING: This project is recommended to run on Python 3.11.")
    else:
        print("OK: Python 3.11 detected.")


def check_pytorch():
    print("\n" + "=" * 60)
    print("PYTORCH")
    print("=" * 60)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")


def check_numpy():
    print("\n" + "=" * 60)
    print("NUMPY")
    print("=" * 60)

    print(f"NumPy version : {np.__version__}")
    print("OK: NumPy imported successfully.")


def check_soundfile():
    print("\n" + "=" * 60)
    print("SOUNDFILE")
    print("=" * 60)

    print(f"SoundFile version : {sf.__version__}")
    print("OK: SoundFile imported successfully.")


def check_ffmpeg():
    print("\n" + "=" * 60)
    print("FFMPEG")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )

        first_line = result.stdout.splitlines()[0]
        print(first_line)
        print("OK: FFmpeg is available.")

    except FileNotFoundError:
        print("ERROR: FFmpeg was not found in PATH.")

    except subprocess.CalledProcessError:
        print("ERROR: FFmpeg exists but could not be executed.")


def check_huggingface_token():
    print("\n" + "=" * 60)
    print("HUGGING FACE TOKEN")
    print("=" * 60)

    load_dotenv()

    token = os.getenv("HF_TOKEN")

    if token:
        print("OK: HF_TOKEN is loaded from .env.")
        print("Token is intentionally not displayed.")
    else:
        print("WARNING: HF_TOKEN was not found.")


def main():
    print("\n")
    print("=" * 60)
    print("SWBS MEETING TRANSCRIPTION")
    print("PHASE 1 - ENVIRONMENT CHECK")
    print("=" * 60)

    check_python()
    check_pytorch()
    check_numpy()
    check_soundfile()
    check_ffmpeg()
    check_huggingface_token()

    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()