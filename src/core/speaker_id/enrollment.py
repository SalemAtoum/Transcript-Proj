from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# ECAPA CONFIGURATION
# ============================================================

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

TARGET_SAMPLE_RATE = 16000

EXPECTED_EMBEDDING_DIMENSION = 192


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VoiceProfile:
    employee_name: str
    model: str
    embedding_dimension: int
    sample_rate: int
    duration_seconds: float
    created_at: str
    embedding_file: str
    metadata_file: str


# ============================================================
# SPEAKER ENROLLMENT
# ============================================================

class SpeakerEnrollment:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        model_cache_dir: str | Path | None = None,
    ):

        self.model_name = model_name

        # ----------------------------------------------------
        # Automatic device selection
        # ----------------------------------------------------

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        # ----------------------------------------------------
        # Project root
        # ----------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        # ----------------------------------------------------
        # Default model cache
        # ----------------------------------------------------

        if model_cache_dir is None:
            model_cache_dir = (
                project_root
                / "models"
                / "speaker_embedding"
                / "spkrec-ecapa-voxceleb"
            )

        self.model_cache_dir = Path(
            model_cache_dir
        ).resolve()

        self.model_cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load ECAPA model
        # ----------------------------------------------------

        print("=" * 70)
        print("Loading ECAPA speaker embedding model")
        print("=" * 70)

        print(
            f"Model:  {self.model_name}"
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Cache:  {self.model_cache_dir}"
        )

        self.encoder = (
            EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=str(
                    self.model_cache_dir
                ),
                run_opts={
                    "device": self.device
                },
                local_strategy=LocalStrategy.COPY,
            )
        )

        print("ECAPA model loaded.")
        print("=" * 70)

    # ========================================================
    # LOAD + STANDARDIZE AUDIO
    # ========================================================

    @staticmethod
    def _load_audio(
        audio_path: Path,
    ) -> tuple[np.ndarray, int, float]:

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

        # ----------------------------------------------------
        # Convert stereo/multi-channel → mono
        # ----------------------------------------------------

        if waveform.ndim == 2:
            waveform = waveform.mean(
                axis=1
            )

        if waveform.ndim != 1:
            raise ValueError(
                f"Expected mono waveform. "
                f"Shape: {waveform.shape}"
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        original_sample_rate = sample_rate

        # ----------------------------------------------------
        # Resample to 16 kHz when necessary
        #
        # Enrollment accepts recordings from different
        # devices/sample rates.
        # ----------------------------------------------------

        if sample_rate != TARGET_SAMPLE_RATE:

            print(
                f"Resampling: "
                f"{sample_rate} Hz → "
                f"{TARGET_SAMPLE_RATE} Hz"
            )

            waveform_tensor = (
                torch.from_numpy(
                    waveform
                )
                .unsqueeze(0)
            )

            waveform_tensor = (
                torchaudio.functional.resample(
                    waveform_tensor,
                    orig_freq=sample_rate,
                    new_freq=TARGET_SAMPLE_RATE,
                )
            )

            waveform = (
                waveform_tensor
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )

            sample_rate = (
                TARGET_SAMPLE_RATE
            )

        duration = (
            len(waveform)
            / sample_rate
        )

        print(
            f"Original sample rate: "
            f"{original_sample_rate} Hz"
        )

        print(
            f"Final sample rate: "
            f"{sample_rate} Hz"
        )

        print(
            f"Duration: "
            f"{duration:.2f}s"
        )

        return (
            waveform,
            sample_rate,
            duration,
        )

    # ========================================================
    # CREATE VOICE PROFILE
    # ========================================================

    def create_profile(
        self,
        employee_name: str,
        audio_file: str | Path,
        output_directory: str | Path,
    ) -> VoiceProfile:

        employee_name = (
            employee_name.strip()
        )

        if not employee_name:
            raise ValueError(
                "employee_name cannot be empty."
            )

        # ----------------------------------------------------
        # Validate audio
        # ----------------------------------------------------

        audio_path = Path(
            audio_file
        ).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Enrollment audio does not exist:\n"
                f"{audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Enrollment path is not a file:\n"
                f"{audio_path}"
            )

        # ----------------------------------------------------
        # Prepare output directory
        # ----------------------------------------------------

        output_dir = Path(
            output_directory
        ).resolve()

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("SPEAKER ENROLLMENT")
        print("=" * 70)

        print(
            f"Employee: {employee_name}"
        )

        print(
            f"Audio:    {audio_path}"
        )

        print(
            f"Device:   {self.device}"
        )

        # ----------------------------------------------------
        # Load and standardize audio
        # ----------------------------------------------------

        (
            waveform,
            sample_rate,
            duration,
        ) = self._load_audio(
            audio_path
        )

        # ----------------------------------------------------
        # Convert NumPy → Torch
        # ----------------------------------------------------

        audio_tensor = (
            torch.from_numpy(
                waveform
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # ----------------------------------------------------
        # Generate ECAPA embedding
        # ----------------------------------------------------

        start_time = time.perf_counter()

        with torch.no_grad():

            embedding = (
                self.encoder.encode_batch(
                    audio_tensor
                )
            )

        processing_seconds = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Convert embedding to NumPy
        # ----------------------------------------------------

        embedding = (
            embedding
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if embedding.ndim != 1:
            embedding = (
                embedding.reshape(-1)
            )

        dimension = int(
            embedding.shape[0]
        )

        # ----------------------------------------------------
        # Validate embedding dimension
        # ----------------------------------------------------

        if (
            dimension
            != EXPECTED_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                "Unexpected ECAPA embedding dimension: "
                f"{dimension}. Expected "
                f"{EXPECTED_EMBEDDING_DIMENSION}."
            )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            raise ValueError(
                "Generated embedding has zero norm."
            )

        embedding = (
            embedding / norm
        ).astype(np.float32)

        # ----------------------------------------------------
        # Safe employee filename
        # ----------------------------------------------------

        safe_name = (
            employee_name
            .lower()
            .replace(" ", "_")
        )

        embedding_file = (
            output_dir
            / f"{safe_name}_profile.npy"
        )

        metadata_file = (
            output_dir
            / f"{safe_name}_profile.json"
        )

        # ----------------------------------------------------
        # Save embedding
        # ----------------------------------------------------

        np.save(
            embedding_file,
            embedding,
        )

        # ----------------------------------------------------
        # Create profile metadata
        # ----------------------------------------------------

        profile = VoiceProfile(
            employee_name=employee_name,
            model=self.model_name,
            embedding_dimension=dimension,
            sample_rate=sample_rate,
            duration_seconds=duration,
            created_at=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            embedding_file=str(
                embedding_file
            ),
            metadata_file=str(
                metadata_file
            ),
        )

        metadata = asdict(
            profile
        )

        metadata[
            "processing_seconds"
        ] = processing_seconds

        metadata[
            "embedding_norm"
        ] = float(
            np.linalg.norm(
                embedding
            )
        )

        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        with metadata_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("Enrollment completed.")
        print("=" * 70)

        print(
            f"Embedding dimension: "
            f"{dimension}"
        )

        print(
            f"Embedding norm: "
            f"{np.linalg.norm(embedding):.6f}"
        )

        print(
            f"Processing time: "
            f"{processing_seconds:.2f}s"
        )

        print(
            f"Embedding: "
            f"{embedding_file}"
        )

        print(
            f"Metadata:  "
            f"{metadata_file}"
        )

        return profile


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Phase 9.4 validation run.

    Supports two modes:

    1. STANDALONE TEST
       Uses the existing validated Salameh recording.

    2. ENROLLMENT MODE
       Uses environment variables supplied by the application
       or future registration workflow.

    Environment variables:

        SWBS_ENROLLMENT_AUDIO
        SWBS_EMPLOYEE_NAME
        SWBS_ENROLLMENT_OUTPUT_DIR
    """

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    # --------------------------------------------------------
    # Standalone defaults
    # --------------------------------------------------------

    default_audio_file = (
        project_root
        / "research"
        / "phase5"
        / "salameh"
        / "salameh_01.wav"
    )

    default_employee_name = (
        "Salameh"
    )

    default_output_directory = (
        project_root
        / "data"
        / "temporary"
        / "phase9_enrollment_test"
    )

    # --------------------------------------------------------
    # Environment variables
    # --------------------------------------------------------

    env_audio_file = os.getenv(
        "SWBS_ENROLLMENT_AUDIO"
    )

    env_employee_name = os.getenv(
        "SWBS_EMPLOYEE_NAME"
    )

    env_output_directory = os.getenv(
        "SWBS_ENROLLMENT_OUTPUT_DIR"
    )

    enrollment_mode = (
        env_audio_file is not None
        or env_employee_name is not None
        or env_output_directory is not None
    )

    # --------------------------------------------------------
    # Resolve configuration
    # --------------------------------------------------------

    audio_file = Path(
        env_audio_file
        if env_audio_file
        else default_audio_file
    )

    employee_name = (
        env_employee_name
        if env_employee_name
        else default_employee_name
    )

    output_directory = Path(
        env_output_directory
        if env_output_directory
        else default_output_directory
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 9.4 — PRODUCTION SPEAKER ENROLLMENT"
    )
    print("=" * 70)

    if enrollment_mode:
        print("Mode: ENROLLMENT")
    else:
        print("Mode: STANDALONE TEST")

    print()
    print("Employee:")
    print(
        f"  {employee_name}"
    )

    print()
    print("Audio:")
    print(
        f"  {audio_file.resolve()}"
    )

    print()
    print("Output directory:")
    print(
        f"  {output_directory.resolve()}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Create enrollment engine
    # --------------------------------------------------------

    enrollment = SpeakerEnrollment()

    # --------------------------------------------------------
    # Create profile
    # --------------------------------------------------------

    profile = enrollment.create_profile(
        employee_name=employee_name,
        audio_file=audio_file,
        output_directory=output_directory,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 9.4 RESULT")
    print("=" * 70)

    print(
        f"Employee: "
        f"{profile.employee_name}"
    )

    print(
        f"Embedding dimension: "
        f"{profile.embedding_dimension}"
    )

    print(
        f"Duration: "
        f"{profile.duration_seconds:.2f}s"
    )

    print(
        f"Profile: "
        f"{profile.embedding_file}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()