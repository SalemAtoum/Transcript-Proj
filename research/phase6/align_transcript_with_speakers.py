from pathlib import Path
import json
import time
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TRANSCRIPT_PATH = (
    SCRIPT_DIR / "canonical_transcript.json"
)

DIARIZATION_PATH = (
    PROJECT_ROOT
    / "phase4_diarization"
    / "diarization_result.json"
)

OUTPUT_DIR = (
    SCRIPT_DIR / "aligned_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "aligned_transcript.json"
)


# ============================================================
# SPEAKER IDENTITY MAPPING
# ============================================================
#
# This mapping was established and validated
# during Phase 5.4.
#
# SPEAKER_00 -> Sami
# SPEAKER_01 -> Salameh
#
# ============================================================

DEFAULT_SPEAKER_MAPPING = {
    "SPEAKER_00": "Sami",
    "SPEAKER_01": "Salameh",
}


# ============================================================
# CONFIGURATION
# ============================================================

MIN_OVERLAP_RATIO = 0.20

# If a word has zero duration, use its timestamp
# as a point and search for the speaker containing it.

NEAREST_SPEAKER_MAX_DISTANCE = 1.0


# ============================================================
# HELPERS
# ============================================================

def overlap_duration(
    start_a,
    end_a,
    start_b,
    end_b
):
    """
    Calculate temporal overlap between two intervals.
    """

    start = max(
        start_a,
        start_b
    )

    end = min(
        end_a,
        end_b
    )

    return max(
        0.0,
        end - start
    )


def format_time(seconds):
    """
    Convert seconds to HH:MM:SS.mmm
    """

    milliseconds = int(
        round((seconds % 1) * 1000)
    )

    total_seconds = int(seconds)

    hours = total_seconds // 3600

    minutes = (
        total_seconds % 3600
    ) // 60

    secs = (
        total_seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{milliseconds:03d}"
    )


def find_best_speaker(
    word_start,
    word_end,
    diarization_segments
):
    """
    Find the speaker whose diarization interval
    has the greatest temporal overlap with a word.

    Returns:
        speaker
        overlap_ratio
        overlap_seconds
        method
    """

    word_duration = max(
        0.0,
        word_end - word_start
    )

    candidates = []

    # --------------------------------------------------------
    # Normal duration word
    # --------------------------------------------------------

    if word_duration > 0:

        for segment in diarization_segments:

            seg_start = segment["start"]
            seg_end = segment["end"]

            overlap = overlap_duration(
                word_start,
                word_end,
                seg_start,
                seg_end
            )

            if overlap <= 0:
                continue

            ratio = (
                overlap / word_duration
            )

            candidates.append(
                {
                    "speaker":
                        segment["speaker"],

                    "overlap":
                        overlap,

                    "ratio":
                        ratio,

                    "distance":
                        0.0,
                }
            )

        if candidates:

            candidates.sort(
                key=lambda x: (
                    x["ratio"],
                    x["overlap"]
                ),
                reverse=True
            )

            best = candidates[0]

            return (
                best["speaker"],
                best["ratio"],
                best["overlap"],
                "overlap"
            )

    # --------------------------------------------------------
    # Zero-duration / extremely short word
    # --------------------------------------------------------

    point = (
        word_start + word_end
    ) / 2.0

    nearest = None

    for segment in diarization_segments:

        seg_start = segment["start"]
        seg_end = segment["end"]

        if (
            seg_start
            <= point
            <= seg_end
        ):

            return (
                segment["speaker"],
                1.0,
                0.0,
                "point_inside"
            )

        if point < seg_start:

            distance = (
                seg_start - point
            )

        else:

            distance = (
                point - seg_end
            )

        if nearest is None or distance < nearest["distance"]:

            nearest = {
                "speaker":
                    segment["speaker"],

                "distance":
                    distance
            }

    # --------------------------------------------------------
    # Nearest speaker fallback
    # --------------------------------------------------------

    if (
        nearest is not None
        and nearest["distance"]
        <= NEAREST_SPEAKER_MAX_DISTANCE
    ):

        return (
            nearest["speaker"],
            0.0,
            0.0,
            "nearest"
        )

    return (
        None,
        0.0,
        0.0,
        "unassigned"
    )


# ============================================================
# LOAD TRANSCRIPT
# ============================================================

print("=" * 70)
print("PHASE 6.2 — SPEAKER / TRANSCRIPT ALIGNMENT")
print("=" * 70)

print("\n[1] Loading canonical transcript...")

if not TRANSCRIPT_PATH.exists():

    raise FileNotFoundError(
        f"Transcript not found:\n"
        f"{TRANSCRIPT_PATH}"
    )

with open(
    TRANSCRIPT_PATH,
    "r",
    encoding="utf-8"
) as f:

    transcript_data = json.load(f)


words = transcript_data.get(
    "words",
    []
)

segments = transcript_data.get(
    "segments",
    []
)

print(
    f"Transcript segments: "
    f"{len(segments)}"
)

print(
    f"Transcript words: "
    f"{len(words)}"
)


# ============================================================
# LOAD DIARIZATION
# ============================================================

print("\n[2] Loading diarization result...")

if not DIARIZATION_PATH.exists():

    raise FileNotFoundError(
        f"Diarization file not found:\n"
        f"{DIARIZATION_PATH}"
    )

with open(
    DIARIZATION_PATH,
    "r",
    encoding="utf-8"
) as f:

    diarization_data = json.load(f)


# IMPORTANT:
# Use exclusive diarization.
#
# This avoids assigning one word to two speakers
# when an overlap exists.

diarization_segments = (
    diarization_data
    .get("exclusive_diarization", [])
)


if not diarization_segments:

    raise ValueError(
        "No exclusive diarization segments found."
    )


print(
    f"Exclusive diarization segments: "
    f"{len(diarization_segments)}"
)

detected_speakers = sorted(
    set(
        x["speaker"]
        for x in diarization_segments
    )
)

print(
    f"Detected speakers: "
    f"{detected_speakers}"
)


# ============================================================
# SORT DIARIZATION
# ============================================================

diarization_segments.sort(
    key=lambda x: x["start"]
)


# ============================================================
# ALIGN WORDS
# ============================================================

print("\n[3] Aligning words with speakers...")

start_time = time.time()

aligned_words = []

statistics = {

    "total_words":
        len(words),

    "assigned_words":
        0,

    "unassigned_words":
        0,

    "overlap_method":
        0,

    "point_inside_method":
        0,

    "nearest_method":
        0,

    "unassigned_method":
        0,

    "low_confidence_words":
        0,
}


for index, word in enumerate(words):

    word_start = float(
        word["start"]
    )

    word_end = float(
        word["end"]
    )

    speaker, ratio, overlap, method = (
        find_best_speaker(
            word_start,
            word_end,
            diarization_segments
        )
    )

    if speaker is not None:

        statistics[
            "assigned_words"
        ] += 1

    else:

        statistics[
            "unassigned_words"
        ] += 1

    if method == "overlap":

        statistics[
            "overlap_method"
        ] += 1

    elif method == "point_inside":

        statistics[
            "point_inside_method"
        ] += 1

    elif method == "nearest":

        statistics[
            "nearest_method"
        ] += 1

    else:

        statistics[
            "unassigned_method"
        ] += 1


    if (
        speaker is not None
        and method == "overlap"
        and ratio < MIN_OVERLAP_RATIO
    ):

        statistics[
            "low_confidence_words"
        ] += 1

        alignment_confidence = (
            "low"
        )

    elif (
        speaker is not None
        and ratio < 0.50
    ):

        alignment_confidence = (
            "medium"
        )

    elif speaker is not None:

        alignment_confidence = (
            "high"
        )

    else:

        alignment_confidence = (
            "none"
        )


    aligned_words.append(
        {
            "word_id":
                word.get(
                    "word_id",
                    index
                ),

            "start":
                word_start,

            "end":
                word_end,

            "start_time":
                word.get(
                    "start_time",
                    format_time(
                        word_start
                    )
                ),

            "end_time":
                word.get(
                    "end_time",
                    format_time(
                        word_end
                    )
                ),

            "word":
                word["word"],

            "speaker":
                speaker,

            "overlap_seconds":
                round(
                    overlap,
                    4
                ),

            "overlap_ratio":
                round(
                    ratio,
                    4
                ),

            "alignment_method":
                method,

            "alignment_confidence":
                alignment_confidence,
        }
    )


# ============================================================
# GROUP WORDS INTO SPEAKER TURNS
# ============================================================

print("\n[4] Building speaker turns...")

turns = []

current_turn = None


for word in aligned_words:

    speaker = word["speaker"]

    if speaker is None:

        # Keep unassigned words separate.
        speaker_key = "UNKNOWN"

    else:

        speaker_key = speaker


    # --------------------------------------------------------
    # Start new turn
    # --------------------------------------------------------

    if current_turn is None:

        current_turn = {

            "turn_id":
                len(turns),

            "speaker":
                speaker,

            "start":
                word["start"],

            "end":
                word["end"],

            "words": [
                word
            ],

            "text":
                word["word"],
        }

        continue


    # --------------------------------------------------------
    # Same speaker
    # --------------------------------------------------------

    current_speaker_key = (
        "UNKNOWN"
        if current_turn["speaker"] is None
        else current_turn["speaker"]
    )

    if (
        speaker_key
        == current_speaker_key
    ):

        current_turn["words"].append(
            word
        )

        current_turn["end"] = (
            word["end"]
        )

        current_turn["text"] += (
            " "
            + word["word"]
        )

    else:

        # Finish previous turn
        turns.append(
            current_turn
        )

        # Start new turn
        current_turn = {

            "turn_id":
                len(turns),

            "speaker":
                speaker,

            "start":
                word["start"],

            "end":
                word["end"],

            "words": [
                word
            ],

            "text":
                word["word"],
        }


# Finish final turn

if current_turn is not None:

    turns.append(
        current_turn
    )


# ============================================================
# APPLY IDENTITY MAPPING
# ============================================================

print("\n[5] Applying speaker identities...")

speaker_mapping = (
    DEFAULT_SPEAKER_MAPPING.copy()
)


for turn in turns:

    anonymous_speaker = (
        turn["speaker"]
    )

    if anonymous_speaker is None:

        turn["speaker_name"] = (
            "UNKNOWN"
        )

    else:

        turn["speaker_name"] = (
            speaker_mapping.get(
                anonymous_speaker,
                anonymous_speaker
            )
        )


# ============================================================
# TURN STATISTICS
# ============================================================

speaker_turn_counts = (
    defaultdict(int)
)

speaker_word_counts = (
    defaultdict(int)
)


for turn in turns:

    speaker = turn["speaker_name"]

    speaker_turn_counts[
        speaker
    ] += 1

    speaker_word_counts[
        speaker
    ] += len(
        turn["words"]
    )


# ============================================================
# PROCESSING TIME
# ============================================================

processing_time = (
    time.time() - start_time
)


# ============================================================
# BUILD OUTPUT
# ============================================================

output = {

    "phase":
        "6.2",

    "component":
        "speaker_transcript_alignment",

    "audio":
        transcript_data.get(
            "audio"
        ),

    "speaker_mapping":
        speaker_mapping,

    "input_statistics": {

        "transcript_segments":
            len(segments),

        "transcript_words":
            len(words),

        "diarization_segments":
            len(
                diarization_segments
            ),

        "detected_speakers":
            detected_speakers,
    },

    "alignment_statistics":
        statistics,

    "speaker_statistics": {

        "turn_counts":
            dict(
                speaker_turn_counts
            ),

        "word_counts":
            dict(
                speaker_word_counts
            ),
    },

    "processing_time_seconds":
        round(
            processing_time,
            3
        ),

    "aligned_words":
        aligned_words,

    "speaker_turns":
        turns,
}


# ============================================================
# SAVE
# ============================================================

print("\n[6] Saving aligned transcript...")

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


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6.2 COMPLETE")
print("=" * 70)

print(
    f"\nTotal words: "
    f"{statistics['total_words']}"
)

print(
    f"Assigned words: "
    f"{statistics['assigned_words']}"
)

print(
    f"Unassigned words: "
    f"{statistics['unassigned_words']}"
)

print(
    f"Overlap-based assignments: "
    f"{statistics['overlap_method']}"
)

print(
    f"Point-inside assignments: "
    f"{statistics['point_inside_method']}"
)

print(
    f"Nearest-speaker assignments: "
    f"{statistics['nearest_method']}"
)

print(
    f"Low-confidence words: "
    f"{statistics['low_confidence_words']}"
)

print(
    f"\nSpeaker turns: "
    f"{len(turns)}"
)

print("\nSpeaker mapping:")

for anonymous, name in speaker_mapping.items():

    print(
        f"  {anonymous} -> {name}"
    )

print("\nSpeaker statistics:")

for speaker, count in (
    speaker_word_counts.items()
):

    print(
        f"  {speaker}: "
        f"{count} words / "
        f"{speaker_turn_counts[speaker]} turns"
    )

print(
    f"\nProcessing time: "
    f"{processing_time:.2f} sec"
)

print(
    f"\nOutput:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 70)