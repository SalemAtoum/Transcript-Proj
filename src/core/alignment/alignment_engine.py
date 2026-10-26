from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


# ------------------------------------------------------------
# Environment-variable aware paths
# ------------------------------------------------------------

DEFAULT_WHISPER_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_whisper_result.json"
)

DEFAULT_DIARIZATION_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_diarization_result.json"
)

DEFAULT_OUTPUT_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_alignment_result.json"
)


def resolve_path(env_name: str, default_path: Path) -> Path:
    """
    Resolve a path from an environment variable.

    If the environment variable is not set, use the
    standalone-test default path.
    """

    value = os.getenv(env_name)

    if value:
        path = Path(value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    return default_path


WHISPER_RESULT = resolve_path(
    "SWBS_WHISPER_FILE",
    DEFAULT_WHISPER_RESULT,
)

DIARIZATION_RESULT = resolve_path(
    "SWBS_DIARIZATION_FILE",
    DEFAULT_DIARIZATION_RESULT,
)

OUTPUT_RESULT = resolve_path(
    "SWBS_OUTPUT_FILE",
    DEFAULT_OUTPUT_RESULT,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum time gap allowed when a word does not overlap
# directly with a speaker segment.
MAX_NEAREST_DISTANCE = 1.0

# Minimum overlap ratio used to consider a speaker assignment
# reliable.
MIN_OVERLAP_RATIO = 0.10


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Save JSON using UTF-8."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def overlap_duration(
    word_start: float,
    word_end: float,
    speaker_start: float,
    speaker_end: float,
) -> float:
    """Return temporal overlap between a word and speaker segment."""

    start = max(
        word_start,
        speaker_start,
    )

    end = min(
        word_end,
        speaker_end,
    )

    return max(
        0.0,
        end - start,
    )


def distance_to_segment(
    word_start: float,
    word_end: float,
    speaker_start: float,
    speaker_end: float,
) -> float:
    """
    Calculate temporal distance between a word and speaker segment.

    Returns:
        0.0 if the word overlaps the segment.
        Positive distance otherwise.
    """

    if word_end < speaker_start:
        return speaker_start - word_end

    if word_start > speaker_end:
        return word_start - speaker_end

    return 0.0


# ============================================================
# EXTRACT WHISPER WORDS
# ============================================================

def extract_whisper_words(
    whisper_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract word-level timestamps from Whisper output.

    Expected structure:

    segments:
      [
        {
          "start": ...,
          "end": ...,
          "words": [
            {
              "word": "...",
              "start": ...,
              "end": ...
            }
          ]
        }
      ]
    """

    words: list[dict[str, Any]] = []

    segments = whisper_data.get(
        "segments",
        [],
    )

    for segment_index, segment in enumerate(segments):

        segment_words = segment.get(
            "words",
            [],
        )

        for word_index, word in enumerate(segment_words):

            text = str(
                word.get(
                    "word",
                    "",
                )
            ).strip()

            if not text:
                continue

            start = word.get("start")
            end = word.get("end")

            if start is None or end is None:
                continue

            start = float(start)
            end = float(end)

            if end < start:
                continue

            words.append(
                {
                    "word_id": len(words),
                    "whisper_segment_id": segment_index,
                    "word_index": word_index,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    return words


# ============================================================
# EXTRACT DIARIZATION SEGMENTS
# ============================================================

def extract_diarization_segments(
    diarization_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract anonymous speaker segments."""

    segments = diarization_data.get(
        "segments",
        [],
    )

    result: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        speaker = str(
            segment["speaker"]
        )

        if end < start:
            continue

        result.append(
            {
                "segment_id": index,
                "start": start,
                "end": end,
                "speaker": speaker,
            }
        )

    result.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    return result


# ============================================================
# ASSIGN WORD TO SPEAKER
# ============================================================

def assign_word_to_speaker(
    word: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
) -> tuple[str | None, str, float, float | None]:
    """
    Assign one Whisper word to a diarized speaker.

    Priority:

    1. Maximum temporal overlap
    2. Point-inside fallback
    3. Nearest speaker segment fallback
    4. UNKNOWN
    """

    word_start = float(
        word["start"]
    )

    word_end = float(
        word["end"]
    )

    word_duration = max(
        0.0,
        word_end - word_start,
    )

    overlap_candidates: list[
        tuple[float, dict[str, Any]]
    ] = []

    for diar_segment in diarization_segments:

        overlap = overlap_duration(
            word_start,
            word_end,
            diar_segment["start"],
            diar_segment["end"],
        )

        if overlap > 0:

            overlap_candidates.append(
                (
                    overlap,
                    diar_segment,
                )
            )

    # --------------------------------------------------------
    # CASE 1: Direct temporal overlap
    # --------------------------------------------------------

    if overlap_candidates:

        overlap_candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        best_overlap, best_segment = (
            overlap_candidates[0]
        )

        overlap_ratio = (
            best_overlap / word_duration
            if word_duration > 0
            else 0.0
        )

        if overlap_ratio >= MIN_OVERLAP_RATIO:

            return (
                best_segment["speaker"],
                "overlap",
                best_overlap,
                overlap_ratio,
            )

        # If the word is extremely short,
        # direct overlap is still useful.
        if word_duration <= 0.10:

            return (
                best_segment["speaker"],
                "overlap",
                best_overlap,
                overlap_ratio,
            )

    # --------------------------------------------------------
    # CASE 2: Point-inside fallback
    # --------------------------------------------------------

    midpoint = (
        (word_start + word_end) / 2.0
    )

    point_candidates = [
        segment
        for segment in diarization_segments
        if (
            segment["start"]
            <= midpoint
            <= segment["end"]
        )
    ]

    if point_candidates:

        point_candidates.sort(
            key=lambda x: (
                x["end"] - x["start"]
            ),
        )

        best_segment = point_candidates[0]

        return (
            best_segment["speaker"],
            "point_inside",
            0.0,
            0.0,
        )

    # --------------------------------------------------------
    # CASE 3: Nearest speaker segment
    # --------------------------------------------------------

    nearest_candidates: list[
        tuple[float, dict[str, Any]]
    ] = []

    for diar_segment in diarization_segments:

        distance = distance_to_segment(
            word_start,
            word_end,
            diar_segment["start"],
            diar_segment["end"],
        )

        nearest_candidates.append(
            (
                distance,
                diar_segment,
            )
        )

    if nearest_candidates:

        nearest_candidates.sort(
            key=lambda x: x[0]
        )

        distance, nearest_segment = (
            nearest_candidates[0]
        )

        if distance <= MAX_NEAREST_DISTANCE:

            return (
                nearest_segment["speaker"],
                "nearest",
                0.0,
                0.0,
            )

    # --------------------------------------------------------
    # CASE 4: UNKNOWN
    # --------------------------------------------------------

    return (
        None,
        "unassigned",
        0.0,
        0.0,
    )


# ============================================================
# BUILD SPEAKER TURNS
# ============================================================

def build_speaker_turns(
    assigned_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Group consecutive words belonging to the same speaker
    into speaker turns.
    """

    turns: list[dict[str, Any]] = []

    current: dict[str, Any] | None = None

    for word in assigned_words:

        speaker = word["speaker"]

        if current is None:

            current = {
                "turn_id": 0,
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "words": [word],
            }

            continue

        same_speaker = (
            current["speaker"]
            == speaker
        )

        if same_speaker:

            current["end"] = word["end"]

            current["words"].append(
                word
            )

        else:

            turns.append(
                current
            )

            current = {
                "turn_id": len(turns),
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "words": [word],
            }

    if current is not None:

        turns.append(
            current
        )

    # --------------------------------------------------------
    # Convert words into final text
    # --------------------------------------------------------

    for turn in turns:

        turn["text"] = " ".join(
            word["text"]
            for word in turn["words"]
        ).strip()

        turn["word_count"] = len(
            turn["words"]
        )

        # Keep detailed words internally.
        # This is useful for later confidence analysis.
        turn.pop(
            "words",
            None,
        )

    return turns


# ============================================================
# VALIDATION
# ============================================================

def validate_alignment(
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate alignment integrity."""

    errors: list[str] = []
    warnings: list[str] = []

    # --------------------------------------------------------
    # Word ordering
    # --------------------------------------------------------

    for index in range(
        1,
        len(words),
    ):

        previous_start = (
            words[index - 1]["start"]
        )

        current_start = (
            words[index]["start"]
        )

        if current_start < previous_start:

            errors.append(
                f"Word timestamps are not monotonic at index {index}."
            )

    # --------------------------------------------------------
    # Word intervals
    # --------------------------------------------------------

    for word in words:

        if word["end"] < word["start"]:

            errors.append(
                f"Invalid word interval: {word}"
            )

    # --------------------------------------------------------
    # Turn intervals
    # --------------------------------------------------------

    for turn in turns:

        if turn["end"] < turn["start"]:

            errors.append(
                f"Invalid turn interval: {turn}"
            )

        if not turn["text"]:

            warnings.append(
                f"Turn {turn['turn_id']} has empty text."
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    assigned = sum(
        1
        for word in words
        if word["speaker"] is not None
    )

    unassigned = (
        len(words)
        - assigned
    )

    coverage = (
        assigned / len(words)
        if words
        else 0.0
    )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_words": len(words),
        "assigned_words": assigned,
        "unassigned_words": unassigned,
        "coverage": coverage,
        "total_turns": len(turns),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    pipeline_mode = any(
        os.getenv(name)
        for name in (
            "SWBS_WHISPER_FILE",
            "SWBS_DIARIZATION_FILE",
            "SWBS_OUTPUT_FILE",
        )
    )

    print("=" * 70)
    print("PHASE 9.6 — PRODUCTION SPEAKER ALIGNMENT")
    print("=" * 70)

    print()
    print(
        f"Mode: "
        f"{'PIPELINE' if pipeline_mode else 'STANDALONE TEST'}"
    )

    print()
    print("Whisper result:")
    print(
        f"  {WHISPER_RESULT}"
    )

    print()
    print("Diarization result:")
    print(
        f"  {DIARIZATION_RESULT}"
    )

    print()
    print("Output:")
    print(
        f"  {OUTPUT_RESULT}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading Whisper result...")

    whisper_data = load_json(
        WHISPER_RESULT
    )

    print(
        "Whisper JSON loaded."
    )

    print()
    print("Loading diarization result...")

    diarization_data = load_json(
        DIARIZATION_RESULT
    )

    print(
        "Diarization JSON loaded."
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print()
    print(
        "Extracting Whisper word timestamps..."
    )

    words = extract_whisper_words(
        whisper_data
    )

    print(
        f"Whisper words: {len(words)}"
    )

    print()
    print(
        "Extracting diarization segments..."
    )

    diarization_segments = (
        extract_diarization_segments(
            diarization_data
        )
    )

    print(
        "Diarization segments: "
        f"{len(diarization_segments)}"
    )

    if not words:

        print()
        print(
            "ERROR: No Whisper words found."
        )

        return 1

    if not diarization_segments:

        print()
        print(
            "ERROR: No diarization segments found."
        )

        return 1

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    print()
    print(
        "Aligning words to speakers..."
    )

    print()

    assigned_words: list[
        dict[str, Any]
    ] = []

    method_counts = {
        "overlap": 0,
        "point_inside": 0,
        "nearest": 0,
        "unassigned": 0,
    }

    speaker_counts: dict[
        str,
        int,
    ] = {}

    for word in words:

        (
            speaker,
            method,
            overlap,
            overlap_ratio,
        ) = assign_word_to_speaker(
            word,
            diarization_segments,
        )

        aligned_word = {
            **word,
            "speaker": speaker,
            "assignment_method": method,
            "overlap_seconds": round(
                overlap,
                4,
            ),
            "overlap_ratio": round(
                overlap_ratio,
                4,
            ),
        }

        assigned_words.append(
            aligned_word
        )

        method_counts[method] += 1

        if speaker is not None:

            speaker_counts[speaker] = (
                speaker_counts.get(
                    speaker,
                    0,
                )
                + 1
            )

    # --------------------------------------------------------
    # Build turns
    # --------------------------------------------------------

    print(
        "Building speaker turns..."
    )

    turns = build_speaker_turns(
        assigned_words
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = validate_alignment(
        assigned_words,
        turns,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "schema_version": "1.0",
        "phase": "9.6",
        "audio_duration": diarization_data.get(
            "duration"
        ),
        "whisper_model": whisper_data.get(
            "model"
        ),
        "diarization_model": diarization_data.get(
            "model"
        ),
        "statistics": {
            "total_whisper_words": len(
                assigned_words
            ),
            "total_diarization_segments": len(
                diarization_segments
            ),
            "total_speaker_turns": len(
                turns
            ),
            "speaker_word_counts": speaker_counts,
            "assignment_methods": method_counts,
        },
        "validation": validation,
        "words": assigned_words,
        "speaker_turns": turns,
    }

    save_json(
        OUTPUT_RESULT,
        output,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALIGNMENT RESULTS")
    print("=" * 70)

    print()

    print(
        "Total Whisper words: "
        f"{len(assigned_words)}"
    )

    print(
        "Assigned words: "
        f"{validation['assigned_words']}"
    )

    print(
        "Unassigned words: "
        f"{validation['unassigned_words']}"
    )

    print(
        "Coverage: "
        f"{validation['coverage'] * 100:.2f}%"
    )

    print(
        "Speaker turns: "
        f"{len(turns)}"
    )

    print()
    print(
        "Assignment methods:"
    )

    for method, count in (
        method_counts.items()
    ):

        print(
            f"  {method}: {count}"
        )

    print()
    print(
        "Speaker word counts:"
    )

    for speaker, count in sorted(
        speaker_counts.items()
    ):

        print(
            f"  {speaker}: {count}"
        )

    print()
    print(
        "Validation:"
    )

    if validation["passed"]:

        print(
            "  STATUS: PASS"
        )

    else:

        print(
            "  STATUS: FAIL"
        )

        for error in validation[
            "errors"
        ]:

            print(
                f"  ERROR: {error}"
            )

    if validation["warnings"]:

        print()

        for warning in validation[
            "warnings"
        ]:

            print(
                f"  WARNING: {warning}"
            )

    print()
    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_RESULT}"
    )

    print()
    print(
        "=" * 70
    )

    return (
        0
        if validation["passed"]
        else 1
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    sys.exit(main())