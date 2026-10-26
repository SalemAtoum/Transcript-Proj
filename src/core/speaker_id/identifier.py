from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

TARGET_SAMPLE_RATE = 16000
EXPECTED_EMBEDDING_DIMENSION = 192

# Minimum useful diarization segment.
MIN_SEGMENT_SECONDS = 1.0

# Maximum number of segments used for each anonymous speaker.
MAX_SEGMENTS_PER_SPEAKER = 10

# Do NOT lower this just to force identification.
IDENTIFICATION_THRESHOLD = 0.55


# ======================================================================
# DATA CLASSES
# ======================================================================

@dataclass
class SpeakerProfile:
    employee_name: str
    embedding: np.ndarray


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class IdentificationResult:
    anonymous_speaker: str
    identified_name: str
    similarity: float
    confidence: str
    status: str
    segments_used: int


# ======================================================================
# SPEAKER IDENTIFIER
# ======================================================================

class SpeakerIdentifier:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        model_cache_dir: str | Path | None = None,
    ):

        self.model_name = model_name

        # --------------------------------------------------------------
        # Automatic CPU/GPU selection
        # --------------------------------------------------------------

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        # --------------------------------------------------------------
        # Project root
        # --------------------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        # --------------------------------------------------------------
        # Default ECAPA model cache
        # --------------------------------------------------------------

        if model_cache_dir is None:
            model_cache_dir = (
                project_root
                / "models"
                / "speaker_embedding"
                / "spkrec-ecapa-voxceleb"
            )

        self.model_cache_dir = (
            Path(model_cache_dir)
            .resolve()
        )

        self.model_cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Load ECAPA
        # --------------------------------------------------------------

        print("=" * 70)
        print(
            "Loading ECAPA speaker identification model"
        )
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

        print(
            "ECAPA model loaded."
        )

        print("=" * 70)

    # ==================================================================
    # PROFILE LOADING
    # ==================================================================

    @staticmethod
    def load_profile(
        profile_file: str | Path,
        employee_name: str | None = None,
    ) -> SpeakerProfile:

        profile_path = (
            Path(profile_file)
            .resolve()
        )

        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile does not exist:\n"
                f"{profile_path}"
            )

        embedding = (
            np.load(profile_path)
            .astype(np.float32)
        )

        if embedding.ndim != 1:
            embedding = (
                embedding.reshape(-1)
            )

        if (
            embedding.shape[0]
            != EXPECTED_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                "Invalid embedding dimension: "
                f"{embedding.shape[0]}"
            )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            raise ValueError(
                "Profile embedding has zero norm."
            )

        embedding = (
            embedding / norm
        ).astype(np.float32)

        # --------------------------------------------------------------
        # If name wasn't explicitly supplied, infer it from filename.
        # Example:
        #
        # sami_profile.npy
        #      ↓
        # Sami
        # --------------------------------------------------------------

        if employee_name is None:

            employee_name = (
                profile_path.stem
                .replace(
                    "_profile",
                    "",
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

        return SpeakerProfile(
            employee_name=employee_name,
            embedding=embedding,
        )

    # ==================================================================
    # AUDIO LOADING
    # ==================================================================

    @staticmethod
    def load_audio(
        audio_file: str | Path,
    ) -> tuple[np.ndarray, int]:

        audio_path = (
            Path(audio_file)
            .resolve()
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist:\n"
                f"{audio_path}"
            )

        waveform, sample_rate = (
            sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=False,
            )
        )

        # --------------------------------------------------------------
        # Stereo → mono
        # --------------------------------------------------------------

        if waveform.ndim == 2:
            waveform = waveform.mean(
                axis=1
            )

        if waveform.ndim != 1:
            raise ValueError(
                f"Expected mono waveform. "
                f"Shape: {waveform.shape}"
            )

        # --------------------------------------------------------------
        # Identification receives already processed 16k audio.
        # --------------------------------------------------------------

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Expected "
                f"{TARGET_SAMPLE_RATE} Hz audio, "
                f"got {sample_rate} Hz."
            )

        return (
            np.asarray(
                waveform,
                dtype=np.float32,
            ),
            sample_rate,
        )

    # ==================================================================
    # ECAPA EMBEDDING
    # ==================================================================

    def extract_embedding(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:

        if len(waveform) == 0:
            raise ValueError(
                "Cannot extract embedding from empty audio."
            )

        audio_tensor = (
            torch.from_numpy(
                waveform
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            embedding = (
                self.encoder.encode_batch(
                    audio_tensor
                )
            )

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

        if (
            embedding.shape[0]
            != EXPECTED_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                "Unexpected embedding dimension: "
                f"{embedding.shape[0]}"
            )

        # --------------------------------------------------------------
        # L2 normalization
        # --------------------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            raise ValueError(
                "Embedding has zero norm."
            )

        embedding = (
            embedding / norm
        ).astype(np.float32)

        return embedding

    # ==================================================================
    # DIARIZATION JSON
    # ==================================================================

    @staticmethod
    def load_diarization(
        diarization_file: str | Path,
    ) -> list[SpeakerSegment]:

        path = (
            Path(diarization_file)
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Diarization result does not exist:\n"
                f"{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        raw_segments = data.get(
            "segments",
            [],
        )

        if not raw_segments:
            raise ValueError(
                "No diarization segments found."
            )

        segments = []

        for item in raw_segments:

            speaker = item.get(
                "speaker"
            )

            start = item.get(
                "start"
            )

            end = item.get(
                "end"
            )

            if (
                speaker is None
                or start is None
                or end is None
            ):
                continue

            start = float(start)
            end = float(end)

            if end <= start:
                continue

            segments.append(
                SpeakerSegment(
                    speaker=str(
                        speaker
                    ),
                    start=start,
                    end=end,
                )
            )

        if not segments:
            raise ValueError(
                "No valid diarization segments found."
            )

        return segments

    # ==================================================================
    # SELECT USABLE SEGMENTS
    # ==================================================================

    @staticmethod
    def select_segments(
        segments: list[SpeakerSegment],
    ) -> dict[str, list[SpeakerSegment]]:

        by_speaker: dict[
            str,
            list[SpeakerSegment]
        ] = {}

        # --------------------------------------------------------------
        # Group by anonymous speaker
        # --------------------------------------------------------------

        for segment in segments:

            if (
                segment.duration
                < MIN_SEGMENT_SECONDS
            ):
                continue

            by_speaker.setdefault(
                segment.speaker,
                [],
            ).append(
                segment
            )

        selected = {}

        # --------------------------------------------------------------
        # Select longest useful segments
        # --------------------------------------------------------------

        for speaker, speaker_segments in (
            by_speaker.items()
        ):

            speaker_segments = sorted(
                speaker_segments,
                key=lambda x: x.duration,
                reverse=True,
            )

            selected_segments = (
                speaker_segments[
                    :MAX_SEGMENTS_PER_SPEAKER
                ]
            )

            # Restore chronological order.
            selected_segments.sort(
                key=lambda x: x.start
            )

            selected[speaker] = (
                selected_segments
            )

        return selected

    # ==================================================================
    # BUILD AGGREGATED SPEAKER EMBEDDING
    # ==================================================================

    def build_speaker_embedding(
        self,
        waveform: np.ndarray,
        segments: list[SpeakerSegment],
    ) -> np.ndarray:

        embeddings = []

        for segment in segments:

            start_sample = int(
                segment.start
                * TARGET_SAMPLE_RATE
            )

            end_sample = int(
                segment.end
                * TARGET_SAMPLE_RATE
            )

            start_sample = max(
                0,
                start_sample,
            )

            end_sample = min(
                len(waveform),
                end_sample,
            )

            if end_sample <= start_sample:
                continue

            chunk = waveform[
                start_sample:end_sample
            ]

            if len(chunk) == 0:
                continue

            embedding = (
                self.extract_embedding(
                    chunk
                )
            )

            embeddings.append(
                embedding
            )

        if not embeddings:
            raise ValueError(
                "No valid embeddings generated "
                "for speaker."
            )

        # --------------------------------------------------------------
        # Average segment embeddings
        # --------------------------------------------------------------

        stacked = np.stack(
            embeddings,
            axis=0,
        )

        speaker_embedding = (
            np.mean(
                stacked,
                axis=0,
            )
        )

        # --------------------------------------------------------------
        # Normalize aggregated embedding
        # --------------------------------------------------------------

        norm = np.linalg.norm(
            speaker_embedding
        )

        if norm == 0:
            raise ValueError(
                "Aggregated speaker embedding "
                "has zero norm."
            )

        speaker_embedding = (
            speaker_embedding / norm
        ).astype(np.float32)

        return speaker_embedding

    # ==================================================================
    # COSINE SIMILARITY
    # ==================================================================

    @staticmethod
    def cosine_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:

        norm_a = np.linalg.norm(
            embedding_a
        )

        norm_b = np.linalg.norm(
            embedding_b
        )

        if (
            norm_a == 0
            or norm_b == 0
        ):
            return 0.0

        return float(
            np.dot(
                embedding_a,
                embedding_b,
            )
            / (
                norm_a
                * norm_b
            )
        )

    # ==================================================================
    # COMPARE AGAINST ALL PROFILES
    # ==================================================================

    def compare_profiles(
        self,
        embedding: np.ndarray,
        profiles: list[SpeakerProfile],
    ) -> list[tuple[str, float]]:

        scores = []

        for profile in profiles:

            score = (
                self.cosine_similarity(
                    embedding,
                    profile.embedding,
                )
            )

            scores.append(
                (
                    profile.employee_name,
                    score,
                )
            )

        return sorted(
            scores,
            key=lambda x: x[1],
            reverse=True,
        )

    # ==================================================================
    # GLOBAL ONE-TO-ONE ASSIGNMENT
    # ==================================================================

    def global_assignment(
        self,
        speaker_embeddings: dict[
            str,
            np.ndarray,
        ],
        profiles: list[SpeakerProfile],
    ) -> list[IdentificationResult]:

        speakers = list(
            speaker_embeddings.keys()
        )

        if not speakers:
            return []

        if not profiles:

            return [
                IdentificationResult(
                    anonymous_speaker=speaker,
                    identified_name="UNKNOWN",
                    similarity=0.0,
                    confidence="none",
                    status="NO_PROFILES",
                    segments_used=0,
                )
                for speaker in speakers
            ]

        # --------------------------------------------------------------
        # Calculate every speaker → every employee score.
        # --------------------------------------------------------------

        score_matrix = {}

        for speaker in speakers:

            scores = (
                self.compare_profiles(
                    speaker_embeddings[
                        speaker
                    ],
                    profiles,
                )
            )

            score_matrix[speaker] = {
                name: score
                for name, score in scores
            }

        # --------------------------------------------------------------
        # Current SWBS test:
        #
        # 2 anonymous speakers
        # 2 registered employees
        #
        # Try all one-to-one assignments.
        # --------------------------------------------------------------

        if (
            len(speakers) == 2
            and len(profiles) >= 2
        ):

            speaker_a = speakers[0]
            speaker_b = speakers[1]

            best_assignment = None
            best_total = -float("inf")

            for i, profile_a in enumerate(
                profiles
            ):

                for j, profile_b in enumerate(
                    profiles
                ):

                    if i == j:
                        continue

                    name_a = (
                        profile_a.employee_name
                    )

                    name_b = (
                        profile_b.employee_name
                    )

                    total = (
                        score_matrix[
                            speaker_a
                        ][name_a]
                        +
                        score_matrix[
                            speaker_b
                        ][name_b]
                    )

                    if total > best_total:

                        best_total = total

                        best_assignment = (
                            name_a,
                            name_b,
                        )

            results = []

            if best_assignment is None:
                raise RuntimeError(
                    "Could not determine a valid "
                    "one-to-one speaker assignment."
                )

            assigned_names = (
                best_assignment
            )

            for speaker, name in zip(
                [
                    speaker_a,
                    speaker_b,
                ],
                assigned_names,
            ):

                similarity = (
                    score_matrix[
                        speaker
                    ][name]
                )

                if similarity >= 0.75:

                    confidence = "high"

                elif similarity >= 0.60:

                    confidence = "medium"

                elif (
                    similarity
                    >= IDENTIFICATION_THRESHOLD
                ):

                    confidence = "low"

                else:

                    confidence = "none"
                    name = "UNKNOWN"

                results.append(
                    IdentificationResult(
                        anonymous_speaker=speaker,
                        identified_name=name,
                        similarity=similarity,
                        confidence=confidence,
                        status=(
                            "IDENTIFIED"
                            if name != "UNKNOWN"
                            else "BELOW_THRESHOLD"
                        ),
                        segments_used=0,
                    )
                )

            return results

        # --------------------------------------------------------------
        # General fallback
        # --------------------------------------------------------------

        results = []

        used_names = set()

        for speaker in speakers:

            candidates = (
                self.compare_profiles(
                    speaker_embeddings[
                        speaker
                    ],
                    profiles,
                )
            )

            selected_name = "UNKNOWN"
            selected_score = 0.0

            for name, score in candidates:

                if name in used_names:
                    continue

                selected_name = name
                selected_score = score
                break

            if (
                selected_score
                < IDENTIFICATION_THRESHOLD
            ):

                selected_name = "UNKNOWN"
                confidence = "none"
                status = "BELOW_THRESHOLD"

            elif selected_score >= 0.75:

                confidence = "high"
                status = "IDENTIFIED"

            elif selected_score >= 0.60:

                confidence = "medium"
                status = "IDENTIFIED"

            else:

                confidence = "low"
                status = "IDENTIFIED"

            if selected_name != "UNKNOWN":

                used_names.add(
                    selected_name
                )

            results.append(
                IdentificationResult(
                    anonymous_speaker=speaker,
                    identified_name=selected_name,
                    similarity=selected_score,
                    confidence=confidence,
                    status=status,
                    segments_used=0,
                )
            )

        return results


# ======================================================================
# PROFILE DISCOVERY
# ======================================================================

def discover_profiles(
    profile_directory: Path,
) -> list[SpeakerProfile]:
    """
    Discover all *_profile.npy files in the
    configured profile directory.

    The employee name is inferred from the
    profile filename unless an explicit
    metadata JSON file is available.
    """

    if not profile_directory.exists():
        raise FileNotFoundError(
            "Profile directory does not exist:\n"
            f"{profile_directory}"
        )

    profile_files = sorted(
        profile_directory.glob(
            "*_profile.npy"
        )
    )

    if not profile_files:
        raise ValueError(
            "No speaker profiles found in:\n"
            f"{profile_directory}"
        )

    profiles = []

    for profile_file in profile_files:

        metadata_file = (
            profile_file.with_suffix(
                ".json"
            )
        )

        employee_name = None

        # --------------------------------------------------------------
        # Prefer metadata name when available.
        # --------------------------------------------------------------

        if metadata_file.exists():

            try:

                with metadata_file.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    metadata = json.load(
                        file
                    )

                employee_name = (
                    metadata.get(
                        "employee_name"
                    )
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                employee_name = None

        profiles.append(
            SpeakerIdentifier.load_profile(
                profile_file,
                employee_name=employee_name,
            )
        )

    return profiles


# ======================================================================
# MAIN TEST
# ======================================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    # ------------------------------------------------------------------
    # Standalone defaults
    # ------------------------------------------------------------------

    default_audio_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_test_60s.wav"
    )

    default_diarization_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_result.json"
    )

    default_profile_directory = (
        project_root
        / "data"
        / "temporary"
        / "phase9_enrollment_test"
    )

    default_result_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_identification_result.json"
    )

    # ------------------------------------------------------------------
    # Environment variables
    # ------------------------------------------------------------------

    env_audio_file = os.getenv(
        "SWBS_INPUT_FILE"
    )

    env_diarization_file = os.getenv(
        "SWBS_DIARIZATION_FILE"
    )

    env_profile_directory = os.getenv(
        "SWBS_PROFILE_DIRECTORY"
    )

    env_result_file = os.getenv(
        "SWBS_OUTPUT_FILE"
    )

    pipeline_mode = (
        env_audio_file is not None
        or env_diarization_file is not None
        or env_profile_directory is not None
        or env_result_file is not None
    )

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------

    audio_file = Path(
        env_audio_file
        if env_audio_file
        else default_audio_file
    )

    diarization_file = Path(
        env_diarization_file
        if env_diarization_file
        else default_diarization_file
    )

    profile_directory = Path(
        env_profile_directory
        if env_profile_directory
        else default_profile_directory
    )

    result_file = Path(
        env_result_file
        if env_result_file
        else default_result_file
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 9.5 — REAL SPEAKER IDENTIFICATION"
    )
    print("=" * 70)

    if pipeline_mode:
        print("Mode: PIPELINE")
    else:
        print("Mode: STANDALONE TEST")

    print()
    print("Audio:")
    print(
        f"  {audio_file.resolve()}"
    )

    print()
    print("Diarization:")
    print(
        f"  {diarization_file.resolve()}"
    )

    print()
    print("Profiles:")
    print(
        f"  {profile_directory.resolve()}"
    )

    print()
    print("Output:")
    print(
        f"  {result_file.resolve()}"
    )

    print("=" * 70)

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Initialize ECAPA
    # ------------------------------------------------------------------

    identifier = SpeakerIdentifier()

    # ------------------------------------------------------------------
    # Load registered profiles
    # ------------------------------------------------------------------

    print()
    print(
        "Loading registered voice profiles..."
    )

    profiles = discover_profiles(
        profile_directory
    )

    print()

    print(
        "Registered profiles:"
    )

    for profile in profiles:

        print(
            f"  - "
            f"{profile.employee_name} "
            f"({len(profile.embedding)}-d)"
        )

    # ------------------------------------------------------------------
    # Load meeting audio
    # ------------------------------------------------------------------

    waveform, sample_rate = (
        identifier.load_audio(
            audio_file
        )
    )

    duration = (
        len(waveform)
        / sample_rate
    )

    print()
    print(
        f"Meeting duration: "
        f"{duration:.2f}s"
    )

    # ------------------------------------------------------------------
    # Load diarization
    # ------------------------------------------------------------------

    diarization_segments = (
        identifier.load_diarization(
            diarization_file
        )
    )

    print(
        f"Diarization segments: "
        f"{len(diarization_segments)}"
    )

    # ------------------------------------------------------------------
    # Detected speakers
    # ------------------------------------------------------------------

    detected_speakers = sorted(
        set(
            segment.speaker
            for segment in diarization_segments
        )
    )

    print()
    print(
        "Detected speakers:"
    )

    for speaker in detected_speakers:

        count = sum(
            1
            for segment in diarization_segments
            if segment.speaker == speaker
        )

        total_duration = sum(
            segment.duration
            for segment in diarization_segments
            if segment.speaker == speaker
        )

        print(
            f"  {speaker}: "
            f"{count} segments, "
            f"{total_duration:.2f}s total"
        )

    # ------------------------------------------------------------------
    # Select useful segments
    # ------------------------------------------------------------------

    selected = (
        identifier.select_segments(
            diarization_segments
        )
    )

    print()
    print(
        "Selected segments:"
    )

    for speaker in detected_speakers:

        segments = selected.get(
            speaker,
            [],
        )

        total_duration = sum(
            segment.duration
            for segment in segments
        )

        print(
            f"  {speaker}: "
            f"{len(segments)} segments, "
            f"{total_duration:.2f}s speech"
        )

        for segment in segments:

            print(
                f"      "
                f"{segment.start:.2f}"
                f" → "
                f"{segment.end:.2f}"
                f" "
                f"({segment.duration:.2f}s)"
            )

    # ------------------------------------------------------------------
    # Build speaker embeddings
    # ------------------------------------------------------------------

    speaker_embeddings = {}

    segment_counts = {}

    for speaker, segments in (
        selected.items()
    ):

        if not segments:
            continue

        print()
        print(
            f"Building embedding for "
            f"{speaker}..."
        )

        embedding = (
            identifier.build_speaker_embedding(
                waveform,
                segments,
            )
        )

        speaker_embeddings[
            speaker
        ] = embedding

        segment_counts[
            speaker
        ] = len(segments)

        print(
            f"Embedding dimension: "
            f"{len(embedding)}"
        )

    # ------------------------------------------------------------------
    # Global identification
    # ------------------------------------------------------------------

    results = (
        identifier.global_assignment(
            speaker_embeddings,
            profiles,
        )
    )

    # ------------------------------------------------------------------
    # Add actual segment counts
    # ------------------------------------------------------------------

    final_results = []

    for result in results:

        result.segments_used = (
            segment_counts.get(
                result.anonymous_speaker,
                0,
            )
        )

        final_results.append(
            result
        )

    processing_seconds = (
        time.perf_counter()
        - start_time
    )

    # ------------------------------------------------------------------
    # Print final results
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SPEAKER IDENTIFICATION RESULTS"
    )
    print("=" * 70)

    for result in final_results:

        print()
        print(
            result.anonymous_speaker
        )

        print(
            f"  → "
            f"{result.identified_name}"
        )

        print(
            f"  Similarity: "
            f"{result.similarity:.4f}"
        )

        print(
            f"  Confidence: "
            f"{result.confidence}"
        )

        print(
            f"  Status: "
            f"{result.status}"
        )

        print(
            f"  Segments used: "
            f"{result.segments_used}"
        )

    print()
    print(
        f"Processing time: "
        f"{processing_seconds:.2f}s"
    )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    result_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {

        "model": MODEL_NAME,

        "device": identifier.device,

        "audio_file": str(
            audio_file.resolve()
        ),

        "diarization_file": str(
            diarization_file.resolve()
        ),

        "profile_directory": str(
            profile_directory.resolve()
        ),

        "duration_seconds": duration,

        "profiles": [
            profile.employee_name
            for profile in profiles
        ],

        "configuration": {

            "min_segment_seconds":
                MIN_SEGMENT_SECONDS,

            "max_segments_per_speaker":
                MAX_SEGMENTS_PER_SPEAKER,

            "identification_threshold":
                IDENTIFICATION_THRESHOLD,
        },

        "detected_speakers":
            detected_speakers,

        "selected_segments": {

            speaker: [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.duration,
                }
                for segment in selected.get(
                    speaker,
                    [],
                )
            ]
            for speaker in detected_speakers
        },

        "results": [
            asdict(result)
            for result in final_results
        ],

        "processing_seconds":
            processing_seconds,
    }

    with result_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Result JSON: "
        f"{result_file.resolve()}"
    )

    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()