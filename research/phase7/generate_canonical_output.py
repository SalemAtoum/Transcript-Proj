import json
import time
from pathlib import Path


# ============================================================
# PHASE 7 — CANONICAL OUTPUT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

INPUT_FILE = (
    PROJECT_DIR
    / "phase6_alignment"
    / "final_transcript"
    / "final_speaker_transcript.json"
)

OUTPUT_JSON = SCRIPT_DIR / "meeting.json"
OUTPUT_MD = SCRIPT_DIR / "meeting.md"
OUTPUT_TXT = SCRIPT_DIR / "meeting.txt"


# ============================================================
# CONFIGURATION
# ============================================================

MEETING_ID = "001"

EXPECTED_MAPPING = {
    "SPEAKER_00": "Sami",
    "SPEAKER_01": "Salameh",
}


# ============================================================
# HELPERS
# ============================================================

def format_timestamp(seconds):
    """
    Convert seconds to HH:MM:SS.mmm
    """
    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds_part = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds_part:02d}."
        f"{milliseconds:03d}"
    )


def calculate_turn_confidence(turn):
    """
    Convert the detailed turn confidence statistics
    into a simple canonical confidence score.

    High = 1.0
    Medium = 0.75
    Low = 0.50
    None = 0.0
    """

    confidence = turn.get("confidence", {})

    word_count = confidence.get("word_count", 0)

    if word_count <= 0:
        return 0.0

    high = confidence.get("high_confidence", 0)
    medium = confidence.get("medium_confidence", 0)
    low = confidence.get("low_confidence", 0)

    score = (
        high * 1.0
        + medium * 0.75
        + low * 0.50
    ) / word_count

    return round(score, 4)


def clean_text(text):
    """
    Normalize whitespace without changing the actual wording.
    """

    if not isinstance(text, str):
        return ""

    return " ".join(text.split()).strip()


# ============================================================
# VALIDATION
# ============================================================

def validate_source(data):

    required_fields = [
        "phase",
        "component",
        "speaker_mapping",
        "final_statistics",
        "speaker_turns",
    ]

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:
        raise ValueError(
            f"Missing required fields in source file: {missing}"
        )

    if data["phase"] != "6.3":
        raise ValueError(
            f"Expected Phase 6.3 source, got {data['phase']}"
        )

    if data["speaker_mapping"] != EXPECTED_MAPPING:
        raise ValueError(
            "Speaker mapping does not match the validated Phase 5.4 mapping."
        )

    turns = data["speaker_turns"]

    if not isinstance(turns, list) or not turns:
        raise ValueError(
            "No speaker turns found in source file."
        )

    return True


# ============================================================
# BUILD CANONICAL JSON
# ============================================================

def build_canonical_output(data):

    source_turns = data["speaker_turns"]

    participants = []

    for turn in source_turns:

        speaker_name = turn.get("speaker_name")

        if (
            speaker_name
            and speaker_name != "UNKNOWN"
            and speaker_name not in participants
        ):
            participants.append(speaker_name)

    segments = []

    for index, turn in enumerate(source_turns):

        speaker_name = turn.get("speaker_name", "UNKNOWN")

        start = float(turn.get("start", 0.0))
        end = float(turn.get("end", start))

        text = clean_text(
            turn.get("text", "")
        )

        confidence = calculate_turn_confidence(turn)

        segment = {
            "segment_id": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "start_time": format_timestamp(start),
            "end_time": format_timestamp(end),
            "speaker": speaker_name,
            "confidence": confidence,
            "text": text,
        }

        segments.append(segment)

    total_duration = 0.0

    if segments:
        total_duration = segments[-1]["end"]

    canonical_output = {
        "schema_version": "1.0",
        "meeting_id": MEETING_ID,
        "language": "ar",
        "participants": participants,
        "duration_seconds": round(total_duration, 3),
        "duration_formatted": format_timestamp(total_duration),
        "total_segments": len(segments),
        "total_words": data["final_statistics"]["total_words"],
        "speaker_mapping": EXPECTED_MAPPING,
        "segments": segments,
    }

    return canonical_output


# ============================================================
# VALIDATE CANONICAL OUTPUT
# ============================================================

def validate_canonical_output(output):

    required_fields = [
        "schema_version",
        "meeting_id",
        "language",
        "participants",
        "duration_seconds",
        "total_segments",
        "total_words",
        "speaker_mapping",
        "segments",
    ]

    for field in required_fields:

        if field not in output:
            raise ValueError(
                f"Canonical output missing field: {field}"
            )

    segments = output["segments"]

    if len(segments) != output["total_segments"]:
        raise ValueError(
            "Segment count mismatch."
        )

    previous_end = -1.0

    for index, segment in enumerate(segments):

        start = segment["start"]
        end = segment["end"]

        if end < start:
            raise ValueError(
                f"Segment {index}: end < start."
            )

        if start < previous_end - 0.001:
            raise ValueError(
                f"Segment {index}: timestamp order violation."
            )

        if not segment["text"].strip():
            raise ValueError(
                f"Segment {index}: empty text."
            )

        if not segment["speaker"]:
            raise ValueError(
                f"Segment {index}: missing speaker."
            )

        confidence = segment["confidence"]

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Segment {index}: invalid confidence {confidence}."
            )

        previous_end = end

    return True


# ============================================================
# MARKDOWN OUTPUT
# ============================================================

def build_markdown(output):

    lines = []

    lines.append("# Meeting Transcript")
    lines.append("")
    lines.append(f"**Meeting ID:** {output['meeting_id']}")
    lines.append(f"**Language:** {output['language']}")
    lines.append(
        f"**Duration:** {output['duration_formatted']}"
    )
    lines.append(
        f"**Segments:** {output['total_segments']}"
    )
    lines.append(
        f"**Words:** {output['total_words']}"
    )
    lines.append("")

    lines.append("## Participants")
    lines.append("")

    for participant in output["participants"]:
        lines.append(f"- {participant}")

    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    for segment in output["segments"]:

        lines.append(
            f"### {segment['speaker']} "
            f"({segment['start_time']} → {segment['end_time']})"
        )

        lines.append("")
        lines.append(segment["text"])
        lines.append("")
        lines.append(
            f"*Confidence: {segment['confidence']:.2f}*"
        )
        lines.append("")

    return "\n".join(lines)


# ============================================================
# TXT OUTPUT
# ============================================================

def build_text(output):

    lines = []

    lines.append(
        f"Meeting ID: {output['meeting_id']}"
    )

    lines.append(
        f"Language: {output['language']}"
    )

    lines.append(
        f"Duration: {output['duration_formatted']}"
    )

    lines.append(
        f"Participants: {', '.join(output['participants'])}"
    )

    lines.append("")

    lines.append("=" * 70)
    lines.append("TRANSCRIPT")
    lines.append("=" * 70)
    lines.append("")

    for segment in output["segments"]:

        lines.append(
            f"[{segment['start_time']} - "
            f"{segment['end_time']}] "
            f"{segment['speaker']}:"
        )

        lines.append(
            segment["text"]
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("PHASE 7 — CANONICAL OUTPUT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load Phase 6.3
    # --------------------------------------------------------

    print("\n[1] Loading Phase 6.3 final transcript...")

    print(f"Input file:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        source_data = json.load(f)

    print("Source JSON loaded successfully.")

    # --------------------------------------------------------
    # 2. Validate source
    # --------------------------------------------------------

    print("\n[2] Validating Phase 6.3 source...")

    validate_source(source_data)

    print("Source validation: PASS")

    # --------------------------------------------------------
    # 3. Build canonical output
    # --------------------------------------------------------

    print("\n[3] Building canonical representation...")

    canonical_output = build_canonical_output(
        source_data
    )

    print(
        f"Canonical segments: "
        f"{len(canonical_output['segments'])}"
    )

    print(
        f"Canonical words: "
        f"{canonical_output['total_words']}"
    )

    print(
        f"Participants: "
        f"{canonical_output['participants']}"
    )

    # --------------------------------------------------------
    # 4. Validate canonical output
    # --------------------------------------------------------

    print("\n[4] Validating canonical output...")

    validate_canonical_output(
        canonical_output
    )

    print("Canonical validation: PASS")

    # --------------------------------------------------------
    # 5. Save JSON
    # --------------------------------------------------------

    print("\n[5] Writing meeting.json...")

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            canonical_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Created: {OUTPUT_JSON}"
    )

    # --------------------------------------------------------
    # 6. Save Markdown
    # --------------------------------------------------------

    print("\n[6] Writing meeting.md...")

    markdown_output = build_markdown(
        canonical_output
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(markdown_output)

    print(
        f"Created: {OUTPUT_MD}"
    )

    # --------------------------------------------------------
    # 7. Save TXT
    # --------------------------------------------------------

    print("\n[7] Writing meeting.txt...")

    text_output = build_text(
        canonical_output
    )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text_output)

    print(
        f"Created: {OUTPUT_TXT}"
    )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    processing_time = (
        time.time() - start_time
    )

    print("\n" + "=" * 70)
    print("PHASE 7 — SUMMARY")
    print("=" * 70)

    print(
        f"\nMeeting ID       : "
        f"{canonical_output['meeting_id']}"
    )

    print(
        f"Participants     : "
        f"{', '.join(canonical_output['participants'])}"
    )

    print(
        f"Duration         : "
        f"{canonical_output['duration_formatted']}"
    )

    print(
        f"Segments         : "
        f"{canonical_output['total_segments']}"
    )

    print(
        f"Words            : "
        f"{canonical_output['total_words']}"
    )

    print("\nOutputs:")

    print(
        f"  JSON : {OUTPUT_JSON}"
    )

    print(
        f"  MD   : {OUTPUT_MD}"
    )

    print(
        f"  TXT  : {OUTPUT_TXT}"
    )

    print("\nPhase 7 validation: PASS")

    print(
        f"\nProcessing time: "
        f"{processing_time:.2f} sec"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()