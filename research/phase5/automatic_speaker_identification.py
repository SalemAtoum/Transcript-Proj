import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# PHASE 5.4
# AUTOMATIC SPEAKER IDENTIFICATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

AUDIO_PATH = PROJECT_ROOT / "meeting_processed.wav"

DIARIZATION_PATH = (
    PROJECT_ROOT
    / "phase4_diarization"
    / "diarization_result.json"
)

VOICE_PROFILES_DIR = SCRIPT_DIR / "voice_profiles"

OUTPUT_DIR = SCRIPT_DIR / "identification_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "speaker_identification_result.json"

# SpeechBrain model directory created during Phase 5.2
MODEL_DIR = (
    SCRIPT_DIR
    / "pretrained_models"
    / "spkrec-ecapa-voxceleb"
)

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"

DEVICE = "cpu"

# ------------------------------------------------------------
# Speaker configuration
# ------------------------------------------------------------

KNOWN_SPEAKERS = {
    "Sami": VOICE_PROFILES_DIR / "sami_profile.npy",
    "Salameh": VOICE_PROFILES_DIR / "salameh_profile.npy",
}

# ------------------------------------------------------------
# Segment selection configuration
# ------------------------------------------------------------

# Ignore very short diarization segments.
MIN_SEGMENT_DURATION = 4.0

# Maximum audio duration used for one ECAPA embedding.
MAX_SEGMENT_DURATION = 8.0

# Number of clean segments selected per anonymous speaker.
SEGMENTS_PER_SPEAKER = 10

# Minimum RMS level to reject nearly silent segments.
MIN_RMS_DB = -45.0


# ============================================================
# Utility functions
# ============================================================

def l2_normalize(vector):
    """
    L2-normalize a vector.
    """
    vector = np.asarray(vector, dtype=np.float32)

    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return vector / norm


def cosine_similarity(a, b):
    """
    Cosine similarity between two vectors.
    """
    a = l2_normalize(a)
    b = l2_normalize(b)

    return float(np.dot(a, b))


def dbfs(audio):
    """
    Calculate RMS level in dBFS.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0:
        return -100.0

    rms = np.sqrt(np.mean(np.square(audio)))

    if rms <= 1e-12:
        return -100.0

    return float(20.0 * np.log10(rms))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Load diarization
# ============================================================

def load_exclusive_diarization():
    """
    Load exclusive diarization segments.

    The project diarization_result.json contains:
        diarization
        exclusive_diarization

    We use exclusive_diarization for speaker embedding extraction
    because it avoids using overlapping speech as enrollment evidence.
    """

    print("=" * 70)
    print("LOADING DIARIZATION")
    print("=" * 70)

    payload = load_json(DIARIZATION_PATH)

    if "exclusive_diarization" not in payload:
        raise KeyError(
            "exclusive_diarization was not found in diarization_result.json"
        )

    segments = payload["exclusive_diarization"]

    if not segments:
        raise ValueError("exclusive_diarization is empty.")

    speakers = sorted(
        {
            segment["speaker"]
            for segment in segments
            if "speaker" in segment
        }
    )

    print(f"Diarization file: {DIARIZATION_PATH}")
    print(f"Total exclusive segments: {len(segments)}")
    print(f"Detected speakers: {speakers}")

    return payload, segments


# ============================================================
# Load audio
# ============================================================

def load_audio():
    """
    Load the already standardized project audio.

    Expected:
        16 kHz
        mono
        PCM_16
    """

    print()
    print("=" * 70)
    print("LOADING AUDIO")
    print("=" * 70)

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    duration = len(audio) / sample_rate

    print(f"Audio: {AUDIO_PATH}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(audio):,}")
    print(f"Duration: {duration:.2f} sec")

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16 kHz audio, got {sample_rate} Hz."
        )

    return audio, sample_rate


# ============================================================
# Prepare candidate segments
# ============================================================

def prepare_candidates(segments, audio_length, sample_rate):
    """
    Prepare clean candidate segments.

    Rules:
    - minimum duration = 4 sec
    - maximum embedding window = 8 sec
    - reject very quiet segments
    - use only exclusive diarization
    """

    candidates = {}

    for speaker in sorted(
        {segment["speaker"] for segment in segments}
    ):
        speaker_segments = []

        for segment in segments:

            if segment["speaker"] != speaker:
                continue

            start = float(segment["start"])
            end = float(segment["end"])

            duration = end - start

            if duration < MIN_SEGMENT_DURATION:
                continue

            # Clip to audio boundaries.
            start = max(0.0, start)
            end = min(audio_length / sample_rate, end)

            if end <= start:
                continue

            # ------------------------------------------------
            # If segment is longer than MAX_SEGMENT_DURATION,
            # split it into several possible windows.
            # ------------------------------------------------

            current = start

            while current < end:

                window_end = min(
                    current + MAX_SEGMENT_DURATION,
                    end,
                )

                window_duration = window_end - current

                if window_duration >= MIN_SEGMENT_DURATION:

                    start_sample = int(current * sample_rate)
                    end_sample = int(window_end * sample_rate)

                    chunk = audio_global[
                        start_sample:end_sample
                    ]

                    rms = dbfs(chunk)

                    if rms >= MIN_RMS_DB:
                        speaker_segments.append(
                            {
                                "start": current,
                                "end": window_end,
                                "duration": window_duration,
                                "rms_db": rms,
                            }
                        )

                current += MAX_SEGMENT_DURATION

        candidates[speaker] = speaker_segments

    return candidates


# Global audio used by candidate preparation.
audio_global = None


# ============================================================
# Select diverse segments
# ============================================================

def select_diverse_segments(candidates):
    """
    Select a limited number of clean segments distributed
    across the entire recording.

    We do not simply take the first 10 segments.

    This gives the speaker representation evidence from
    different parts of the meeting.
    """

    selected = {}

    for speaker, speaker_candidates in candidates.items():

        if len(speaker_candidates) == 0:
            raise RuntimeError(
                f"No usable segments found for {speaker}."
            )

        # Sort chronologically.
        speaker_candidates = sorted(
            speaker_candidates,
            key=lambda x: x["start"],
        )

        if len(speaker_candidates) <= SEGMENTS_PER_SPEAKER:
            chosen = speaker_candidates
        else:
            # Evenly distribute selection over the meeting.
            indices = np.linspace(
                0,
                len(speaker_candidates) - 1,
                SEGMENTS_PER_SPEAKER,
                dtype=int,
            )

            chosen = [
                speaker_candidates[int(i)]
                for i in indices
            ]

        selected[speaker] = chosen

    return selected


# ============================================================
# Load ECAPA model
# ============================================================

def load_encoder():
    print()
    print("=" * 70)
    print("LOADING ECAPA-TDNN")
    print("=" * 70)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Local model directory: {MODEL_DIR}")

    start_time = time.time()

    encoder = EncoderClassifier.from_hparams(
        source=MODEL_NAME,
        savedir=str(MODEL_DIR),
        run_opts={"device": DEVICE},
        local_strategy=LocalStrategy.COPY,
    )

    elapsed = time.time() - start_time

    print(f"Model loaded in {elapsed:.2f} sec")

    return encoder


# ============================================================
# Extract one embedding
# ============================================================

def extract_embedding(
    encoder,
    audio,
    sample_rate,
    start,
    end,
):
    """
    Extract ECAPA embedding from one audio segment.
    """

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    chunk = audio[start_sample:end_sample]

    if len(chunk) == 0:
        raise ValueError("Empty audio chunk.")

    waveform = torch.from_numpy(chunk).float()

    # SpeechBrain expects:
    # [batch, time]
    waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        embedding = encoder.encode_batch(waveform)

    embedding = embedding.squeeze().cpu().numpy()

    embedding = l2_normalize(embedding)

    return embedding


# ============================================================
# Build anonymous speaker profiles
# ============================================================

def build_anonymous_profiles(
    encoder,
    audio,
    sample_rate,
    selected_segments,
):
    """
    Extract multiple ECAPA embeddings for each anonymous speaker,
    then create a centroid profile.
    """

    print()
    print("=" * 70)
    print("EXTRACTING ANONYMOUS SPEAKER EMBEDDINGS")
    print("=" * 70)

    anonymous_profiles = {}
    extraction_report = {}

    for speaker, segments in selected_segments.items():

        print()
        print(f"Speaker: {speaker}")
        print("-" * 50)

        embeddings = []
        segment_results = []

        for index, segment in enumerate(segments, start=1):

            start = segment["start"]
            end = segment["end"]

            print(
                f"[{index:02d}/{len(segments)}] "
                f"{start:.2f}s -> {end:.2f}s "
                f"({segment['duration']:.2f}s)"
            )

            start_time = time.time()

            embedding = extract_embedding(
                encoder=encoder,
                audio=audio,
                sample_rate=sample_rate,
                start=start,
                end=end,
            )

            elapsed = time.time() - start_time

            embeddings.append(embedding)

            segment_results.append(
                {
                    "start": start,
                    "end": end,
                    "duration": segment["duration"],
                    "rms_db": segment["rms_db"],
                    "embedding_dimension": int(
                        len(embedding)
                    ),
                    "processing_seconds": elapsed,
                }
            )

            print(
                f"    embedding_dim={len(embedding)} "
                f"time={elapsed:.2f}s"
            )

        embeddings = np.vstack(embeddings)

        # ----------------------------------------------------
        # Normalize every embedding before centroid.
        # ----------------------------------------------------

        embeddings = np.array(
            [
                l2_normalize(embedding)
                for embedding in embeddings
            ],
            dtype=np.float32,
        )

        centroid = np.mean(
            embeddings,
            axis=0,
        )

        centroid = l2_normalize(centroid)

        # ----------------------------------------------------
        # Internal consistency:
        # similarity of every segment to centroid.
        # ----------------------------------------------------

        similarities = [
            cosine_similarity(
                embedding,
                centroid,
            )
            for embedding in embeddings
        ]

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

        anonymous_profiles[speaker] = centroid

        extraction_report[speaker] = {
            "num_segments": len(segments),
            "embedding_dimension": int(
                len(centroid)
            ),
            "mean_similarity_to_centroid": mean_similarity,
            "min_similarity_to_centroid": min_similarity,
            "max_similarity_to_centroid": max_similarity,
            "std_similarity_to_centroid": std_similarity,
            "segments": segment_results,
        }

        print()
        print(
            f"{speaker} internal consistency:"
        )
        print(
            f"  Mean similarity: {mean_similarity:.4f}"
        )
        print(
            f"  Min similarity:  {min_similarity:.4f}"
        )
        print(
            f"  Max similarity:  {max_similarity:.4f}"
        )
        print(
            f"  Std similarity:  {std_similarity:.4f}"
        )

    return anonymous_profiles, extraction_report


# ============================================================
# Load known voice profiles
# ============================================================

def load_known_profiles():
    print()
    print("=" * 70)
    print("LOADING KNOWN VOICE PROFILES")
    print("=" * 70)

    profiles = {}

    for name, path in KNOWN_SPEAKERS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Voice profile not found: {path}"
            )

        profile = np.load(path)

        profile = l2_normalize(profile)

        profiles[name] = profile

        print(
            f"{name}: "
            f"dimension={len(profile)} "
            f"path={path}"
        )

    return profiles


# ============================================================
# Compare anonymous speakers with known speakers
# ============================================================

def calculate_similarity_matrix(
    anonymous_profiles,
    known_profiles,
):
    print()
    print("=" * 70)
    print("SIMILARITY MATRIX")
    print("=" * 70)

    matrix = {}

    for anonymous_speaker, anonymous_profile in (
        anonymous_profiles.items()
    ):

        matrix[anonymous_speaker] = {}

        for known_name, known_profile in (
            known_profiles.items()
        ):

            similarity = cosine_similarity(
                anonymous_profile,
                known_profile,
            )

            matrix[anonymous_speaker][known_name] = similarity

            print(
                f"{anonymous_speaker} -> "
                f"{known_name}: "
                f"{similarity:.4f}"
            )

    return matrix


# ============================================================
# Global assignment
# ============================================================

def assign_speakers(similarity_matrix):
    """
    For two anonymous speakers and two known speakers,
    evaluate both possible one-to-one assignments.

    Mapping A:
        SPEAKER_00 -> Sami
        SPEAKER_01 -> Salameh

    Mapping B:
        SPEAKER_00 -> Salameh
        SPEAKER_01 -> Sami

    Select the assignment with the highest total similarity.
    """

    anonymous_speakers = sorted(
        similarity_matrix.keys()
    )

    if len(anonymous_speakers) != 2:
        raise RuntimeError(
            "This Phase 5.4 implementation expects exactly "
            "two anonymous speakers."
        )

    known_names = list(KNOWN_SPEAKERS.keys())

    if len(known_names) != 2:
        raise RuntimeError(
            "This Phase 5.4 implementation expects exactly "
            "two known speakers."
        )

    speaker_a = anonymous_speakers[0]
    speaker_b = anonymous_speakers[1]

    name_a = known_names[0]
    name_b = known_names[1]

    mapping_a = {
        speaker_a: name_a,
        speaker_b: name_b,
    }

    mapping_b = {
        speaker_a: name_b,
        speaker_b: name_a,
    }

    score_a = (
        similarity_matrix[speaker_a][name_a]
        + similarity_matrix[speaker_b][name_b]
    )

    score_b = (
        similarity_matrix[speaker_a][name_b]
        + similarity_matrix[speaker_b][name_a]
    )

    if score_a >= score_b:
        best_mapping = mapping_a
        best_score = score_a
        alternative_score = score_b
        selected_assignment = "A"
    else:
        best_mapping = mapping_b
        best_score = score_b
        alternative_score = score_a
        selected_assignment = "B"

    margin = best_score - alternative_score

    return {
        "mapping": best_mapping,
        "selected_assignment": selected_assignment,
        "best_total_similarity": float(best_score),
        "alternative_total_similarity": float(
            alternative_score
        ),
        "assignment_margin": float(margin),
        "all_candidate_assignments": {
            "A": {
                "mapping": mapping_a,
                "total_similarity": float(score_a),
            },
            "B": {
                "mapping": mapping_b,
                "total_similarity": float(score_b),
            },
        },
    }


# ============================================================
# Save results
# ============================================================

def save_results(
    diarization_payload,
    selected_segments,
    extraction_report,
    similarity_matrix,
    assignment_result,
):
    result = {
        "phase": "5.4",
        "title": "Automatic Speaker Identification",
        "method": {
            "embedding_model": MODEL_NAME,
            "device": DEVICE,
            "profile_comparison": "cosine_similarity",
            "segment_source": "exclusive_diarization",
            "global_assignment": True,
        },
        "input": {
            "audio": str(AUDIO_PATH),
            "diarization_file": str(DIARIZATION_PATH),
        },
        "configuration": {
            "min_segment_duration_seconds": MIN_SEGMENT_DURATION,
            "max_segment_duration_seconds": MAX_SEGMENT_DURATION,
            "segments_per_speaker": SEGMENTS_PER_SPEAKER,
            "min_rms_db": MIN_RMS_DB,
        },
        "selected_segments": selected_segments,
        "anonymous_speaker_analysis": extraction_report,
        "similarity_matrix": similarity_matrix,
        "assignment": assignment_result,
        "phase4_metadata": {
            "num_speakers": diarization_payload.get(
                "num_speakers"
            ),
            "speaker_labels": diarization_payload.get(
                "speaker_labels"
            ),
            "duration_seconds": diarization_payload.get(
                "duration_seconds"
            ),
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Result saved to:\n{OUTPUT_JSON}"
    )


# ============================================================
# Main
# ============================================================

def main():

    global audio_global

    overall_start = time.time()

    print()
    print("=" * 70)
    print("PHASE 5.4 — AUTOMATIC SPEAKER IDENTIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load diarization
    # --------------------------------------------------------

    diarization_payload, diarization_segments = (
        load_exclusive_diarization()
    )

    # --------------------------------------------------------
    # 2. Load audio
    # --------------------------------------------------------

    audio, sample_rate = load_audio()

    audio_global = audio

    # --------------------------------------------------------
    # 3. Prepare candidate segments
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARING CLEAN SPEAKER SEGMENTS")
    print("=" * 70)

    candidates = prepare_candidates(
        diarization_segments,
        len(audio),
        sample_rate,
    )

    for speaker, speaker_candidates in candidates.items():
        print(
            f"{speaker}: "
            f"{len(speaker_candidates)} usable candidates"
        )

    # --------------------------------------------------------
    # 4. Select diverse segments
    # --------------------------------------------------------

    selected_segments = select_diverse_segments(
        candidates
    )

    print()
    print("Selected segments:")

    for speaker, segments in selected_segments.items():

        print()
        print(speaker)

        for index, segment in enumerate(
            segments,
            start=1,
        ):
            print(
                f"  {index:02d}. "
                f"{segment['start']:.2f}s -> "
                f"{segment['end']:.2f}s "
                f"({segment['duration']:.2f}s)"
            )

    # --------------------------------------------------------
    # 5. Load ECAPA
    # --------------------------------------------------------

    encoder = load_encoder()

    # --------------------------------------------------------
    # 6. Extract anonymous speaker profiles
    # --------------------------------------------------------

    (
        anonymous_profiles,
        extraction_report,
    ) = build_anonymous_profiles(
        encoder=encoder,
        audio=audio,
        sample_rate=sample_rate,
        selected_segments=selected_segments,
    )

    # --------------------------------------------------------
    # 7. Load known profiles
    # --------------------------------------------------------

    known_profiles = load_known_profiles()

    # --------------------------------------------------------
    # 8. Similarity matrix
    # --------------------------------------------------------

    similarity_matrix = calculate_similarity_matrix(
        anonymous_profiles=anonymous_profiles,
        known_profiles=known_profiles,
    )

    # --------------------------------------------------------
    # 9. Global one-to-one assignment
    # --------------------------------------------------------

    assignment_result = assign_speakers(
        similarity_matrix
    )

    # --------------------------------------------------------
    # 10. Print final mapping
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL AUTOMATIC IDENTIFICATION")
    print("=" * 70)

    for anonymous_speaker, real_name in (
        assignment_result["mapping"].items()
    ):
        print(
            f"{anonymous_speaker} -> {real_name}"
        )

    print()
    print(
        "Best total similarity: "
        f"{assignment_result['best_total_similarity']:.4f}"
    )

    print(
        "Alternative total similarity: "
        f"{assignment_result['alternative_total_similarity']:.4f}"
    )

    print(
        "Assignment margin: "
        f"{assignment_result['assignment_margin']:.4f}"
    )

    # --------------------------------------------------------
    # 11. Save
    # --------------------------------------------------------

    save_results(
        diarization_payload=diarization_payload,
        selected_segments=selected_segments,
        extraction_report=extraction_report,
        similarity_matrix=similarity_matrix,
        assignment_result=assignment_result,
    )

    total_time = time.time() - overall_start

    print()
    print("=" * 70)
    print("PHASE 5.4 COMPLETE")
    print("=" * 70)

    print(
        f"Total processing time: "
        f"{total_time:.2f} sec"
    )


if __name__ == "__main__":
    main()