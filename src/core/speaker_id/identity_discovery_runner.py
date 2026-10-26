from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
import sys

# ============================================================
# Project root
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.speaker_id.identity_discovery import (
    SpeakerTurn,
    discover_identities,
)


# ============================================================
# Paths
# ============================================================

ALIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "alignment_result.json"
)

REAL_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "identity_discovery_result.json"
)

TEST_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "identity_discovery_synthetic_test.json"
)


# ============================================================
# Serialization helpers
# ============================================================

def serialize_value(value):
    """
    Convert dataclasses/enums/dicts/lists into JSON-safe values.
    """

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {
            key: serialize_value(val)
            for key, val in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): serialize_value(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]

    return value


def serialize_candidate(candidate):
    return serialize_value(candidate)


# ============================================================
# Alignment loader
# ============================================================

def load_alignment_turns() -> list[SpeakerTurn]:
    print("Loading alignment result...")

    if not ALIGNMENT_FILE.exists():
        raise FileNotFoundError(
            f"Alignment file not found:\n{ALIGNMENT_FILE}"
        )

    with ALIGNMENT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    print("Alignment JSON loaded.")
    print()
    print("Converting alignment turns...")

    raw_turns = data.get("speaker_turns", [])

    turns: list[SpeakerTurn] = []

    for item in raw_turns:
        turns.append(
            SpeakerTurn(
                speaker_id=str(item["speaker"]),
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
            )
        )

    return turns


# ============================================================
# Real alignment test
# ============================================================

def run_real_alignment_test():
    print("=" * 70)
    print("REAL ALIGNMENT TEST")
    print("=" * 70)
    print()

    turns = load_alignment_turns()

    print(f"Speaker turns: {len(turns)}")

    speakers = sorted(
        {turn.speaker_id for turn in turns}
    )

    print("Detected speakers:")

    for speaker in speakers:
        print(f"  {speaker}")

    print()
    print("Running conversational identity discovery...")

    result = discover_identities(
        turns,
        meeting_id="real_alignment_test",
    )

    print()
    print("-" * 70)
    print("REAL ALIGNMENT RESULTS")
    print("-" * 70)

    print()
    print(f"Candidates: {len(result.candidates)}")

    for candidate in result.candidates:
        print(
            f"  {candidate.speaker_id} -> "
            f"{candidate.candidate_name} | "
            f"confidence={candidate.confidence} | "
            f"level={candidate.confidence_level.value} | "
            f"status={candidate.status.value}"
        )

    output = {
        "test_type": "real_alignment",
        "meeting_id": result.meeting_id,
        "candidate_count": len(result.candidates),
        "candidates": [
            serialize_candidate(candidate)
            for candidate in result.candidates
        ],
    }

    REAL_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REAL_OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Output:")
    print(f"  {REAL_OUTPUT_FILE}")

    return result


# ============================================================
# Synthetic conversational test
# ============================================================

def run_synthetic_test():
    print()
    print("=" * 70)
    print("SYNTHETIC CONVERSATIONAL IDENTITY TEST")
    print("=" * 70)
    print()

    print("Scenario:")
    print()
    print("  SPEAKER_00: أهلا يوسف، كيفك؟")
    print("  SPEAKER_01: الحمد لله تمام.")
    print("  SPEAKER_00: طيب يوسف شو بتنصحنا؟")
    print("  SPEAKER_01: بنصحكم نبدأ بالخطة الجديدة.")
    print()

    synthetic_turns = [
        SpeakerTurn(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=2.0,
            text="أهلا يوسف، كيفك؟",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_01",
            start=2.2,
            end=4.0,
            text="الحمد لله تمام.",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_00",
            start=4.3,
            end=6.5,
            text="طيب يوسف شو بتنصحنا؟",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_01",
            start=6.7,
            end=9.0,
            text="بنصحكم نبدأ بالخطة الجديدة.",
        ),
    ]

    result = discover_identities(
        synthetic_turns,
        meeting_id="synthetic_identity_test",
    )

    print("-" * 70)
    print("SYNTHETIC RESULTS")
    print("-" * 70)

    print()
    print(f"Candidates: {len(result.candidates)}")

    for candidate in result.candidates:
        print()
        print(
            f"Speaker:      {candidate.speaker_id}"
        )
        print(
            f"Candidate:    {candidate.candidate_name}"
        )
        print(
            f"Confidence:   {candidate.confidence}"
        )
        print(
            f"Level:        {candidate.confidence_level.value}"
        )
        print(
            f"Status:       {candidate.status.value}"
        )
        print(
            f"Confirmation: "
            f"{candidate.confirmation_status.value}"
        )
        print(
            f"Evidence:     {candidate.evidence_count}"
        )

        print("Evidence types:")

        for evidence in candidate.evidence:
            print(
                f"  - {evidence.evidence_type.value}"
            )

    output = {
        "test_type": "synthetic_conversational_identity",
        "meeting_id": result.meeting_id,
        "candidate_count": len(result.candidates),
        "candidates": [
            serialize_candidate(candidate)
            for candidate in result.candidates
        ],
    }

    TEST_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TEST_OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Output:")
    print(f"  {TEST_OUTPUT_FILE}")

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert len(result.candidates) >= 1, (
        "FAIL: No identity candidate discovered."
    )

    yousef_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.candidate_name == "Yousef"
    ]

    assert len(yousef_candidates) == 1, (
        "FAIL: Expected exactly one Yousef candidate."
    )

    candidate = yousef_candidates[0]

    assert candidate.speaker_id == "SPEAKER_01", (
        "FAIL: Yousef should map to SPEAKER_01."
    )

    assert candidate.confidence >= 0.60, (
        "FAIL: Repeated direct address should reach "
        "at least MEDIUM confidence."
    )

    evidence_types = {
        evidence.evidence_type.value
        for evidence in candidate.evidence
    }

    required_evidence = {
        "direct_address",
        "immediate_response",
        "conversation_continuity",
        "repeated_address",
    }

    missing = required_evidence - evidence_types

    assert not missing, (
        f"FAIL: Missing evidence types: {sorted(missing)}"
    )

    print()
    print("=" * 70)
    print("SYNTHETIC TEST: PASS")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 9.9.4C — REALISTIC IDENTITY INTEGRATION TEST")
    print("=" * 70)
    print()

    run_real_alignment_test()

    run_synthetic_test()

    print()
    print("=" * 70)
    print("PHASE 9.9.4C: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()