from pathlib import Path
import json

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

EMBEDDINGS_DIR = SCRIPT_DIR / "embeddings"

OUTPUT_DIR = SCRIPT_DIR / "voice_profiles"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SPEAKERS = {
    "sami": "sami_*_embedding.npy",
    "salameh": "salameh_*_embedding.npy",
}

EMBEDDING_DIMENSION = 192


# ============================================================
# HELPERS
# ============================================================

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding to unit length.
    """

    vector = vector.astype(np.float32)

    norm = np.linalg.norm(vector)

    if norm <= 1e-12:
        raise ValueError(
            "Cannot normalize zero-length embedding."
        )

    return vector / norm


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Cosine similarity between two normalized vectors.
    """

    a = l2_normalize(a)
    b = l2_normalize(b)

    return float(np.dot(a, b))


def load_embeddings(
    speaker_name: str,
    pattern: str,
):
    """
    Load all embeddings belonging to one known speaker.
    """

    files = sorted(
        EMBEDDINGS_DIR.glob(pattern)
    )

    if not files:
        raise FileNotFoundError(
            f"No embeddings found for {speaker_name}. "
            f"Pattern: {pattern}"
        )

    embeddings = []

    for file_path in files:

        embedding = np.load(
            file_path
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.shape[0] != EMBEDDING_DIMENSION:

            raise ValueError(
                f"{file_path.name} has dimension "
                f"{embedding.shape[0]}, expected "
                f"{EMBEDDING_DIMENSION}."
            )

        embedding = l2_normalize(
            embedding
        )

        embeddings.append(
            embedding
        )

    return files, embeddings


def build_centroid(
    embeddings,
):
    """
    Build a speaker centroid from multiple
    normalized embeddings.

    The centroid is normalized again after averaging.
    """

    matrix = np.vstack(
        embeddings
    )

    centroid = np.mean(
        matrix,
        axis=0,
    )

    centroid = l2_normalize(
        centroid
    )

    return centroid


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 5.3 — BUILDING SPEAKER VOICE PROFILES")
    print("=" * 70)

    profiles = {}

    overall_pass = True

    for speaker_name, pattern in SPEAKERS.items():

        print()
        print("=" * 70)
        print(
            f"BUILDING PROFILE: "
            f"{speaker_name.upper()}"
        )
        print("=" * 70)

        try:

            files, embeddings = load_embeddings(
                speaker_name,
                pattern,
            )

            print()
            print(
                f"Found {len(files)} embeddings:"
            )

            for file_path in files:
                print(
                    f"  - {file_path.name}"
                )

            # ------------------------------------------------
            # Build centroid
            # ------------------------------------------------

            centroid = build_centroid(
                embeddings
            )

            # ------------------------------------------------
            # Internal consistency
            #
            # Compare every enrollment embedding
            # against the resulting profile.
            # ------------------------------------------------

            similarities = []

            for file_path, embedding in zip(
                files,
                embeddings,
            ):

                similarity = cosine_similarity(
                    embedding,
                    centroid,
                )

                similarities.append(
                    similarity
                )

                print(
                    f"  {file_path.name}: "
                    f"{similarity:.4f}"
                )

            mean_similarity = float(
                np.mean(similarities)
            )

            min_similarity = float(
                np.min(similarities)
            )

            max_similarity = float(
                np.max(similarities)
            )

            std_similarity = float(
                np.std(similarities)
            )

            # ------------------------------------------------
            # Save centroid
            # ------------------------------------------------

            profile_file = (
                OUTPUT_DIR
                / f"{speaker_name}_profile.npy"
            )

            np.save(
                profile_file,
                centroid,
            )

            # ------------------------------------------------
            # Profile metadata
            # ------------------------------------------------

            metadata = {
                "speaker": speaker_name,
                "model": (
                    "speechbrain/"
                    "spkrec-ecapa-voxceleb"
                ),
                "embedding_dimension": int(
                    centroid.shape[0]
                ),
                "num_enrollment_embeddings": len(
                    embeddings
                ),
                "enrollment_files": [
                    file.name
                    for file in files
                ],
                "profile_type": "normalized_centroid",
                "normalized": True,
                "internal_similarity": {
                    "mean": round(
                        mean_similarity,
                        6,
                    ),
                    "min": round(
                        min_similarity,
                        6,
                    ),
                    "max": round(
                        max_similarity,
                        6,
                    ),
                    "std": round(
                        std_similarity,
                        6,
                    ),
                },
                "profile_file": str(
                    profile_file
                ),
            }

            metadata_file = (
                OUTPUT_DIR
                / f"{speaker_name}_profile.json"
            )

            with open(
                metadata_file,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            profiles[speaker_name] = metadata

            print()
            print(
                f"Mean similarity: "
                f"{mean_similarity:.4f}"
            )

            print(
                f"Minimum similarity: "
                f"{min_similarity:.4f}"
            )

            print(
                f"Maximum similarity: "
                f"{max_similarity:.4f}"
            )

            print(
                f"Std similarity: "
                f"{std_similarity:.4f}"
            )

            print()
            print(
                f"Profile saved: "
                f"{profile_file.name}"
            )

            print(
                f"Metadata saved: "
                f"{metadata_file.name}"
            )

            print(
                "Status: PASS"
            )

        except Exception as e:

            overall_pass = False

            print()
            print(
                f"ERROR: "
                f"{type(e).__name__}: {e}"
            )

    # ========================================================
    # CROSS-SPEAKER PROFILE SEPARATION
    # ========================================================

    print()
    print("=" * 70)
    print("CROSS-SPEAKER PROFILE SEPARATION")
    print("=" * 70)

    cross_similarity = None

    if (
        "sami" in profiles
        and "salameh" in profiles
    ):

        sami_profile = np.load(
            profiles["sami"]["profile_file"]
        )

        salameh_profile = np.load(
            profiles["salameh"]["profile_file"]
        )

        cross_similarity = cosine_similarity(
            sami_profile,
            salameh_profile,
        )

        print(
            f"Sami ↔ Salameh cosine similarity: "
            f"{cross_similarity:.4f}"
        )

        print()
        print(
            "Note: A lower cross-speaker similarity "
            "means the two profiles are more separated."
        )

    else:

        overall_pass = False

        print(
            "Could not calculate cross-speaker "
            "similarity because one or more profiles "
            "were not created."
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {
        "phase": "5.3",
        "name": "Building Speaker Voice Profiles",
        "model": (
            "speechbrain/"
            "spkrec-ecapa-voxceleb"
        ),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "profile_type": "normalized_centroid",
        "speakers": profiles,
        "cross_speaker_similarity": (
            round(
                cross_similarity,
                6,
            )
            if cross_similarity is not None
            else None
        ),
        "status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    }

    report_path = (
        SCRIPT_DIR
        / "voice_profile_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 5.3 SUMMARY")
    print("=" * 70)

    print(
        f"Sami profile: "
        f"{'PASS' if 'sami' in profiles else 'FAIL'}"
    )

    print(
        f"Salameh profile: "
        f"{'PASS' if 'salameh' in profiles else 'FAIL'}"
    )

    if cross_similarity is not None:

        print(
            f"Sami ↔ Salameh similarity: "
            f"{cross_similarity:.4f}"
        )

    print()
    print(
        f"FINAL STATUS: "
        f"{report['status']}"
    )

    print()
    print(
        f"Report saved to:\n"
        f"{report_path}"
    )

    print(
        f"Profiles saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()