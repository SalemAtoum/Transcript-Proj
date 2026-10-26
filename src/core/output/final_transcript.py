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
# Default standalone-test paths
# ------------------------------------------------------------

DEFAULT_ALIGNMENT_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_alignment_result.json"
)

DEFAULT_IDENTIFICATION_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_identification_result.json"
)

DEFAULT_OUTPUT_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_final_transcript.json"
)


def resolve_path(
    env_name: str,
    default_path: Path,
) -> Path:
    """
    Resolve a path from an environment variable.

    If the environment variable is not set,
    use the standalone-test default path.
    """

    value = os.getenv(env_name)

    if value:

        path = Path(value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    return default_path


ALIGNMENT_RESULT = resolve_path(
    "SWBS_ALIGNMENT_FILE",
    DEFAULT_ALIGNMENT_RESULT,
)

IDENTIFICATION_RESULT = resolve_path(
    "SWBS_IDENTIFICATION_FILE",
    DEFAULT_IDENTIFICATION_RESULT,
)

OUTPUT_RESULT = resolve_path(
    "SWBS_OUTPUT_FILE",
    DEFAULT_OUTPUT_RESULT,
)


# ============================================================
# DEFAULT SPEAKER MAPPING
# ============================================================

# This mapping comes from the controlled Phase 9.5 test.
#
# IMPORTANT:
# 9.7 does not perform speaker identification.
# It only applies the identification result to the
# already-aligned transcript.
#
# IMPORTANT PRODUCTION RULE:
# Default mappings are NEVER allowed to introduce
# speakers that are not present in the current alignment.

DEFAULT_MAPPING = {
    "SPEAKER_00": "UNKNOWN",
    "SPEAKER_01": "Salameh",
}


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load a UTF-8 JSON file.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON root must be an object:\n{path}"
        )

    return data


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Save JSON as UTF-8.
    """

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


def format_timestamp(
    seconds: float,
) -> str:
    """
    Convert seconds to HH:MM:SS.mmm.
    """

    seconds = max(
        0.0,
        float(seconds),
    )

    total_milliseconds = round(
        seconds * 1000
    )

    hours = (
        total_milliseconds
        // 3_600_000
    )

    remaining = (
        total_milliseconds
        % 3_600_000
    )

    minutes = (
        remaining
        // 60_000
    )

    remaining %= 60_000

    secs = (
        remaining
        // 1000
    )

    milliseconds = (
        remaining
        % 1000
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{milliseconds:03d}"
    )


# ============================================================
# DETECT SPEAKERS FROM ALIGNMENT
# ============================================================

def get_detected_speakers(
    alignment_data: dict[str, Any],
) -> set[str]:
    """
    Return the anonymous speaker IDs that actually occur
    in the current alignment result.

    This is the source of truth for which speakers belong
    to the current meeting.
    """

    alignment_turns = alignment_data.get(
        "speaker_turns",
        [],
    )

    if not isinstance(
        alignment_turns,
        list,
    ):
        return set()

    detected_speakers: set[str] = set()

    for turn in alignment_turns:

        if not isinstance(
            turn,
            dict,
        ):
            continue

        speaker = turn.get(
            "speaker"
        )

        if speaker is None:
            continue

        speaker = str(
            speaker
        ).strip()

        if speaker:
            detected_speakers.add(
                speaker
            )

    return detected_speakers


# ============================================================
# LOAD SPEAKER IDENTIFICATION
# ============================================================

def load_speaker_mapping(
    identification_data: dict[str, Any],
) -> dict[str, str]:
    """
    Extract anonymous-speaker -> employee-name mapping.

    Supported identification formats:

    1. Production/current format:

        {
            "results": [
                {
                    "anonymous_speaker": "SPEAKER_00",
                    "identified_name": "Salameh",
                    "similarity": 0.746,
                    "confidence": "medium",
                    "status": "IDENTIFIED"
                }
            ]
        }

    2. Dictionary format:

        {
            "results": {
                "SPEAKER_00": {
                    "identity": "Salameh"
                }
            }
        }

    3. Alternative explicit mapping:

        {
            "speaker_mapping": {
                "SPEAKER_00": "Salameh"
            }
        }

    UNKNOWN / rejected / empty identities are ignored.
    """

    mapping: dict[str, str] = {}

    results = identification_data.get(
        "results",
        [],
    )

    # --------------------------------------------------------
    # Format 1: results is a LIST
    # --------------------------------------------------------

    if isinstance(
        results,
        list,
    ):

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            speaker = (
                result.get("anonymous_speaker")
                or result.get("speaker")
                or result.get("speaker_id")
            )

            identity = (
                result.get("identified_name")
                or result.get("identity")
                or result.get("name")
                or result.get("employee_name")
            )

            status = str(
                result.get(
                    "status",
                    "",
                )
            ).upper()

            if not speaker or not identity:
                continue

            identity = str(
                identity
            ).strip()

            if identity in {
                "UNKNOWN",
                "Guest",
                "guest",
                "",
            }:
                continue

            if status in {
                "REJECTED",
                "UNKNOWN",
                "UNIDENTIFIED",
            }:
                continue

            mapping[
                str(speaker)
            ] = identity

    # --------------------------------------------------------
    # Format 2: results is a DICTIONARY
    # --------------------------------------------------------

    elif isinstance(
        results,
        dict,
    ):

        for speaker, result in results.items():

            if isinstance(
                result,
                dict,
            ):

                identity = (
                    result.get("identified_name")
                    or result.get("identity")
                    or result.get("name")
                    or result.get("employee_name")
                )

            else:

                identity = result

            if not identity:
                continue

            identity = str(
                identity
            ).strip()

            if identity in {
                "UNKNOWN",
                "Guest",
                "guest",
                "",
            }:
                continue

            mapping[
                str(speaker)
            ] = identity

    # --------------------------------------------------------
    # Format 3: explicit speaker_mapping
    # --------------------------------------------------------

    identification_mapping = (
        identification_data.get(
            "speaker_mapping"
        )
    )

    if isinstance(
        identification_mapping,
        dict,
    ):

        for speaker, identity in (
            identification_mapping.items()
        ):

            if not identity:
                continue

            identity = str(
                identity
            ).strip()

            if identity in {
                "UNKNOWN",
                "Guest",
                "guest",
                "",
            }:
                continue

            mapping[
                str(speaker)
            ] = identity

    return mapping


# ============================================================
# BUILD CURRENT MEETING MAPPING
# ============================================================

def build_speaker_mapping(
    alignment_data: dict[str, Any],
    identification_data: dict[str, Any] | None,
) -> tuple[dict[str, str], set[str]]:
    """
    Build the final speaker mapping for the current meeting.

    Critical production rule:

        Only speakers detected in the current alignment
        are allowed to appear in the final mapping.

    Identification results have priority.

    Defaults are used only for detected speakers that do not
    have an identification result.
    """

    detected_speakers = get_detected_speakers(
        alignment_data
    )

    mapping: dict[str, str] = {}

    # --------------------------------------------------------
    # 1. Apply controlled defaults ONLY to detected speakers.
    # --------------------------------------------------------

    for speaker in sorted(
        detected_speakers
    ):

        mapping[speaker] = (
            DEFAULT_MAPPING.get(
                speaker,
                "UNKNOWN",
            )
        )

    # --------------------------------------------------------
    # 2. Apply real identification results.
    # --------------------------------------------------------

    if identification_data is not None:

        detected_mapping = (
            load_speaker_mapping(
                identification_data
            )
        )

        for speaker, identity in (
            detected_mapping.items()
        ):

            # NEVER introduce a speaker that does not
            # exist in the current meeting alignment.
            if speaker not in detected_speakers:
                continue

            mapping[speaker] = identity

    # --------------------------------------------------------
    # 3. Final safety filter.
    # --------------------------------------------------------

    mapping = {
        speaker: identity
        for speaker, identity in mapping.items()
        if speaker in detected_speakers
    }

    return (
        mapping,
        detected_speakers,
    )


# ============================================================
# BUILD FINAL TURNS
# ============================================================

def build_final_turns(
    alignment_data: dict[str, Any],
    speaker_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Convert aligned anonymous speaker turns into final
    human-readable speaker turns.
    """

    alignment_turns = alignment_data.get(
        "speaker_turns",
        [],
    )

    if not isinstance(
        alignment_turns,
        list,
    ):
        return []

    final_turns: list[
        dict[str, Any]
    ] = []

    for index, turn in enumerate(
        alignment_turns
    ):

        if not isinstance(
            turn,
            dict,
        ):
            continue

        anonymous_speaker = turn.get(
            "speaker"
        )

        speaker_name = (
            speaker_mapping.get(
                anonymous_speaker,
                "UNKNOWN",
            )
            if anonymous_speaker
            else "UNKNOWN"
        )

        start = float(
            turn.get(
                "start",
                0.0,
            )
        )

        end = float(
            turn.get(
                "end",
                start,
            )
        )

        text = str(
            turn.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        final_turns.append(
            {
                "turn_id": index,
                "start": round(
                    start,
                    3,
                ),
                "end": round(
                    end,
                    3,
                ),
                "start_time": format_timestamp(
                    start
                ),
                "end_time": format_timestamp(
                    end
                ),
                "speaker_id": anonymous_speaker,
                "speaker": speaker_name,
                "text": text,
                "word_count": int(
                    turn.get(
                        "word_count",
                        0,
                    )
                ),
            }
        )

    return final_turns


# ============================================================
# VALIDATION
# ============================================================

def validate_final_transcript(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate final transcript structure.
    """

    errors: list[str] = []
    warnings: list[str] = []

    previous_start = -1.0

    total_words = 0

    speaker_word_counts: dict[
        str,
        int,
    ] = {}

    unknown_turns = 0

    for index, turn in enumerate(
        turns
    ):

        start = float(
            turn["start"]
        )

        end = float(
            turn["end"]
        )

        speaker = str(
            turn["speaker"]
        )

        text = str(
            turn["text"]
        ).strip()

        word_count = int(
            turn.get(
                "word_count",
                0,
            )
        )

        # ----------------------------------------------------
        # Timestamp validation
        # ----------------------------------------------------

        if start < previous_start:

            errors.append(
                f"Turn {index} starts before "
                f"the previous turn."
            )

        if end < start:

            errors.append(
                f"Turn {index} has invalid "
                f"time interval."
            )

        previous_start = start

        # ----------------------------------------------------
        # Text validation
        # ----------------------------------------------------

        if not text:

            errors.append(
                f"Turn {index} has empty text."
            )

        # ----------------------------------------------------
        # Speaker validation
        # ----------------------------------------------------

        if not speaker:

            errors.append(
                f"Turn {index} has no speaker."
            )

        if speaker == "UNKNOWN":

            unknown_turns += 1

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_words += word_count

        speaker_word_counts[
            speaker
        ] = (
            speaker_word_counts.get(
                speaker,
                0,
            )
            + word_count
        )

    # --------------------------------------------------------
    # Warning for UNKNOWN
    # --------------------------------------------------------

    if unknown_turns > 0:

        warnings.append(
            f"{unknown_turns} turn(s) have "
            f"UNKNOWN speaker identity."
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_turns": len(turns),
        "total_words": total_words,
        "unknown_turns": unknown_turns,
        "speaker_word_counts": (
            speaker_word_counts
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    pipeline_mode = any(
        os.getenv(name)
        for name in (
            "SWBS_ALIGNMENT_FILE",
            "SWBS_IDENTIFICATION_FILE",
            "SWBS_OUTPUT_FILE",
        )
    )

    print("=" * 70)

    print(
        "PHASE 9.7 — FINAL SPEAKER TRANSCRIPT"
    )

    print("=" * 70)

    print()

    print(
        "Mode: "
        f"{'PIPELINE' if pipeline_mode else 'STANDALONE TEST'}"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print()
    print(
        "Alignment result:"
    )

    print(
        f"  {ALIGNMENT_RESULT}"
    )

    print()
    print(
        "Identification result:"
    )

    print(
        f"  {IDENTIFICATION_RESULT}"
    )

    print()
    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_RESULT}"
    )

    # --------------------------------------------------------
    # Load alignment
    # --------------------------------------------------------

    print()
    print(
        "Loading alignment result..."
    )

    alignment_data = load_json(
        ALIGNMENT_RESULT
    )

    print(
        "Alignment JSON loaded."
    )

    # --------------------------------------------------------
    # Detect speakers
    # --------------------------------------------------------

    detected_speakers = (
        get_detected_speakers(
            alignment_data
        )
    )

    print()
    print(
        "Detected speakers in current meeting:"
    )

    if detected_speakers:

        for speaker in sorted(
            detected_speakers
        ):

            print(
                f"  {speaker}"
            )

    else:

        print(
            "  NONE"
        )

    # --------------------------------------------------------
    # Load identification
    # --------------------------------------------------------

    identification_data = None

    if IDENTIFICATION_RESULT.exists():

        print()
        print(
            "Loading speaker identification..."
        )

        identification_data = load_json(
            IDENTIFICATION_RESULT
        )

        print(
            "Identification JSON loaded."
        )

    else:

        print()
        print(
            "WARNING: Identification result "
            "not found."
        )

        print(
            "Using controlled defaults only "
            "for detected speakers."
        )

    # --------------------------------------------------------
    # Build mapping
    # --------------------------------------------------------

    (
        speaker_mapping,
        detected_speakers,
    ) = build_speaker_mapping(
        alignment_data=alignment_data,
        identification_data=identification_data,
    )

    # --------------------------------------------------------
    # Print mapping
    # --------------------------------------------------------

    print()
    print(
        "Final speaker mapping:"
    )

    if speaker_mapping:

        for speaker, identity in sorted(
            speaker_mapping.items()
        ):

            print(
                f"  {speaker} → {identity}"
            )

    else:

        print(
            "  NONE"
        )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    stale_speakers = (
        set(speaker_mapping)
        - detected_speakers
    )

    if stale_speakers:

        print()
        print(
            "ERROR: Stale speaker mappings detected:"
        )

        for speaker in sorted(
            stale_speakers
        ):

            print(
                f"  {speaker}"
            )

        return 1

    # --------------------------------------------------------
    # Build final transcript
    # --------------------------------------------------------

    print()
    print(
        "Building final transcript..."
    )

    final_turns = build_final_turns(
        alignment_data,
        speaker_mapping,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation = (
        validate_final_transcript(
            final_turns
        )
    )

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    output = {
        "schema_version": "1.0",
        "phase": "9.7",
        "language": "ar",
        "audio_duration": (
            alignment_data.get(
                "audio_duration"
            )
        ),
        "models": {
            "transcription": (
                alignment_data.get(
                    "whisper_model"
                )
            ),
            "diarization": (
                alignment_data.get(
                    "diarization_model"
                )
            ),
            "speaker_embedding": (
                "speechbrain/"
                "spkrec-ecapa-voxceleb"
            ),
        },
        "detected_speakers": sorted(
            detected_speakers
        ),
        "speaker_mapping": speaker_mapping,
        "statistics": {
            "total_turns": len(
                final_turns
            ),
            "total_words": sum(
                turn["word_count"]
                for turn in final_turns
            ),
            "speaker_word_counts": (
                validation[
                    "speaker_word_counts"
                ]
            ),
        },
        "validation": validation,
        "turns": final_turns,
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

    print(
        "FINAL TRANSCRIPT RESULTS"
    )

    print("=" * 70)

    print()

    print(
        f"Detected speakers: "
        f"{len(detected_speakers)}"
    )

    print(
        f"Total turns: "
        f"{len(final_turns)}"
    )

    print(
        f"Total words: "
        f"{validation['total_words']}"
    )

    print(
        f"UNKNOWN turns: "
        f"{validation['unknown_turns']}"
    )

    print()

    print(
        "Speaker word counts:"
    )

    for speaker, count in sorted(
        validation[
            "speaker_word_counts"
        ].items()
    ):

        print(
            f"  {speaker}: {count}"
        )

    print()

    print(
        "First turns:"
    )

    for turn in final_turns[:5]:

        print()

        print(
            f"[{turn['start_time']} → "
            f"{turn['end_time']}] "
            f"{turn['speaker']}:"
        )

        print(
            f"  {turn['text']}"
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

    print("=" * 70)

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