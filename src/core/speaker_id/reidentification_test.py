from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16000


class ReIdentificationTester:
    """
    Validate whether an enrolled speaker profile can
    recognize the same speaker from separate audio.

    This is a validation component only.

    It does NOT:
    - create a new profile
    - modify the existing profile
    - perform identity discovery
    - confirm an identity
    """

    def __init__(self):
        print("=" * 70)
        print("Loading ECAPA speaker embedding model")
        print("=" * 70)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("Model:", MODEL_NAME)
        print("Device:", self.device)

        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        cache_dir = (
            project_root
            / "models"
            / "speaker_embedding"
            / "spkrec-ecapa-voxceleb"
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.encoder = EncoderClassifier.from_hparams(
            source=MODEL_NAME,
            savedir=str(cache_dir),
            run_opts={
                "device": str(self.device),
            },
            local_strategy=LocalStrategy.COPY,
        )

        self.encoder.eval()

        print("ECAPA model loaded.")
        print("=" * 70)

    # ------------------------------------------------------------------
    # AUDIO LOADING
    # ------------------------------------------------------------------

    def load_audio(
        self,
        audio_file: str | Path,
    ) -> torch.Tensor:

        audio_path = Path(
            audio_file
        ).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Stereo -> mono
        # --------------------------------------------------------------

        if waveform.ndim == 2:

            # soundfile returns:
            # samples x channels

            waveform = waveform.mean(
                axis=1
            )

        waveform_tensor = torch.from_numpy(
            waveform
        ).float()

        # --------------------------------------------------------------
        # Resample to 16 kHz
        # --------------------------------------------------------------

        if sample_rate != TARGET_SAMPLE_RATE:

            waveform_tensor = (
                torchaudio.functional.resample(
                    waveform_tensor,
                    sample_rate,
                    TARGET_SAMPLE_RATE,
                )
            )

            sample_rate = TARGET_SAMPLE_RATE

        waveform_tensor = waveform_tensor.unsqueeze(
            0
        )

        return waveform_tensor

    # ------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------

    def extract_embedding(
        self,
        audio_file: str | Path,
    ) -> np.ndarray:

        waveform = self.load_audio(
            audio_file
        )

        waveform = waveform.to(
            self.device
        )

        with torch.no_grad():

            embedding = (
                self.encoder
                .encode_batch(waveform)
            )

        embedding = embedding.squeeze()

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=0,
        )

        return (
            embedding
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # ------------------------------------------------------------------
    # LOAD PROFILE
    # ------------------------------------------------------------------

    @staticmethod
    def load_profile(
        profile_file: str | Path,
    ) -> np.ndarray:

        profile_path = Path(
            profile_file
        ).resolve()

        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile not found: {profile_path}"
            )

        embedding = np.load(
            profile_path
        ).astype(np.float32)

        embedding = embedding.reshape(-1)

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            raise ValueError(
                "Profile embedding has zero norm."
            )

        embedding = (
            embedding / norm
        )

        return embedding

    # ------------------------------------------------------------------
    # COSINE SIMILARITY
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:

        a = embedding_a.reshape(-1)
        b = embedding_b.reshape(-1)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError(
                "Cannot calculate similarity "
                "with zero-norm embedding."
            )

        similarity = (
            np.dot(a, b)
            / (norm_a * norm_b)
        )

        return float(
            similarity
        )

    # ------------------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------------------

    def validate(
        self,
        profile_file: str | Path,
        test_audio_file: str | Path,
    ) -> dict:

        print()
        print("=" * 70)
        print("RE-IDENTIFICATION VALIDATION")
        print("=" * 70)

        print(
            "Profile:",
            Path(profile_file).resolve(),
        )

        print(
            "Test audio:",
            Path(test_audio_file).resolve(),
        )

        # --------------------------------------------------------------
        # Load enrolled profile
        # --------------------------------------------------------------

        enrolled_embedding = (
            self.load_profile(
                profile_file
            )
        )

        print()
        print(
            "Enrolled embedding dimension:",
            enrolled_embedding.shape[0],
        )

        # --------------------------------------------------------------
        # Extract embedding from NEW audio
        # --------------------------------------------------------------

        print()
        print(
            "Extracting embedding from test audio..."
        )

        test_embedding = (
            self.extract_embedding(
                test_audio_file
            )
        )

        print(
            "Test embedding dimension:",
            test_embedding.shape[0],
        )

        # --------------------------------------------------------------
        # Dimension validation
        # --------------------------------------------------------------

        if (
            enrolled_embedding.shape
            != test_embedding.shape
        ):
            raise ValueError(
                "Embedding dimensions do not match."
            )

        # --------------------------------------------------------------
        # Similarity
        # --------------------------------------------------------------

        similarity = self.cosine_similarity(
            enrolled_embedding,
            test_embedding,
        )

        print()
        print(
            f"Cosine similarity: {similarity:.4f}"
        )

        # --------------------------------------------------------------
        # Interpretation
        #
        # IMPORTANT:
        # This test does NOT establish a production threshold.
        # We use the existing identifier threshold only as
        # a reference point.
        # --------------------------------------------------------------

        if similarity >= 0.75:
            interpretation = "HIGH_REFERENCE"

        elif similarity >= 0.60:
            interpretation = "MEDIUM_REFERENCE"

        elif similarity >= 0.55:
            interpretation = "LOW_REFERENCE"

        else:
            interpretation = "BELOW_REFERENCE"

        print(
            "Reference interpretation:",
            interpretation,
        )

        result = {
            "status": "completed",
            "profile_file": str(
                Path(profile_file).resolve()
            ),
            "test_audio_file": str(
                Path(test_audio_file).resolve()
            ),
            "embedding_dimension": int(
                enrolled_embedding.shape[0]
            ),
            "cosine_similarity": similarity,
            "reference_interpretation": (
                interpretation
            ),
            "same_speaker_ground_truth": True,
            "production_threshold_validated": False,
        }

        return result

    # ------------------------------------------------------------------
    # SAVE RESULT
    # ------------------------------------------------------------------

    @staticmethod
    def save_result(
        result: dict,
        output_file: str | Path,
    ) -> Path:

        output_path = Path(
            output_file
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path


# ======================================================================
# CONTROLLED TEST
# ======================================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    profile_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_enrollment_test"
        / "yousef_profile.npy"
    )

    test_audio_file = (
        project_root
        / "data"
        / "temporary"
        / "speaker_enrollment"
        / "yousef_SPEAKER_01_provisional.wav"
    )

    output_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_reidentification_result.json"
    )

    print()
    print("=" * 70)
    print(
        "PHASE 9.9.9A — RE-IDENTIFICATION VALIDATION"
    )
    print("=" * 70)

    # --------------------------------------------------------------
    # Verify required artifacts
    # --------------------------------------------------------------

    if not profile_file.exists():

        raise FileNotFoundError(
            "Yousef profile does not exist:\n"
            f"{profile_file}"
        )

    if not test_audio_file.exists():

        raise FileNotFoundError(
            "Yousef test audio does not exist:\n"
            f"{test_audio_file}"
        )

    tester = ReIdentificationTester()

    result = tester.validate(
        profile_file=profile_file,
        test_audio_file=test_audio_file,
    )

    tester.save_result(
        result=result,
        output_file=output_file,
    )

    # --------------------------------------------------------------
    # Controlled validation
    # --------------------------------------------------------------

    similarity = (
        result["cosine_similarity"]
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Cosine similarity: {similarity:.4f}"
    )

    print(
        "Same-speaker ground truth: TRUE"
    )

    # --------------------------------------------------------------
    # For this controlled test we expect the
    # newly enrolled speaker to match strongly
    # against the same source audio.
    #
    # We intentionally do NOT convert this into
    # a production threshold.
    # --------------------------------------------------------------

    if similarity < 0.75:

        print()
        print(
            "TEST A: FAIL"
        )

        print(
            "The same-speaker similarity is below "
            "the HIGH reference range."
        )

        print(
            "Do NOT change the production threshold "
            "based on this test."
        )

        raise SystemExit(1)

    print()
    print(
        "TEST A: PASS"
    )

    print()
    print(
        "Result saved:"
    )

    print(
        output_file
    )

    print()
    print("=" * 70)
    print(
        "PHASE 9.9.9A: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()