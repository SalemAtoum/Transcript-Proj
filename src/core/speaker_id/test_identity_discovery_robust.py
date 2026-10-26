from __future__ import annotations

from src.core.speaker_id.identity_discovery import (
    SpeakerTurn,
    discover_identities,
)


def find_candidate(result, speaker_id, candidate_name):
    return next(
        (
            candidate
            for candidate in result.candidates
            if candidate.speaker_id == speaker_id
            and candidate.candidate_name == candidate_name
        ),
        None,
    )


def print_result(test_name, result):
    print(f"\n[{test_name}]")

    if not result.candidates:
        print("  No identity candidates.")
        return

    for candidate in result.candidates:
        print(
            f"  {candidate.speaker_id} -> "
            f"{candidate.candidate_name} | "
            f"confidence={candidate.confidence:.2f} | "
            f"level={candidate.confidence_level.value} | "
            f"status={candidate.status.value}"
        )

        for evidence in candidate.evidence:
            print(
                f"    - {evidence.evidence_type.value}"
            )


# ============================================================
# TEST 1 — DIRECT ADDRESS
# ============================================================

def test_direct_address():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            3,
            "أهلا يوسف كيفك؟",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            3.5,
            6,
            "الحمد لله تمام.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_direct_address",
    )

    print_result(
        "TEST 1 — DIRECT ADDRESS",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_01",
        "Yousef",
    )

    assert candidate is not None
    assert candidate.confidence > 0.0
    assert candidate.evidence_count >= 3
    

    print("  PASS")


# ============================================================
# TEST 2 — SELF IDENTIFICATION
# ============================================================

def test_self_identification():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            4,
            "أنا يوسف، تشرفت فيكم.",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            4.5,
            7,
            "أهلا وسهلا يوسف.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_self_identification",
    )

    print_result(
        "TEST 2 — SELF IDENTIFICATION",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_00",
        "Yousef",
    )

    assert candidate is not None

    # Self-identification is the strongest individual evidence.
    assert candidate.confidence >= 0.40

    evidence_types = {
        evidence.evidence_type.value
        for evidence in candidate.evidence
    }

    assert "self_identification" in evidence_types

    print("  PASS")


# ============================================================
# TEST 3 — NAME MENTION ONLY
# ============================================================

def test_name_mention_only():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            4,
            "يوسف اليوم مش راح يحضر.",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            4.5,
            7,
            "آه تمام.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_name_mention_only",
    )

    print_result(
        "TEST 3 — NAME MENTION ONLY",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_01",
        "Yousef",
    )

    assert candidate is None

    print("  PASS")


# ============================================================
# TEST 4 — CONTEXTUAL REFERENCE
# ============================================================

def test_contextual_reference():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            4,
            "أنا حكيت مع يوسف مبارح.",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            4.5,
            7,
            "شو صار معه؟",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_context_reference",
    )

    print_result(
        "TEST 4 — CONTEXTUAL REFERENCE",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_01",
        "Yousef",
    )

    assert candidate is None

    print("  PASS")


# ============================================================
# TEST 5 — REPEATED ADDRESS
# ============================================================

def test_repeated_address():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            3,
            "يوسف شو رأيك؟",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            3.5,
            6,
            "ممتاز.",
        ),
        SpeakerTurn(
            "SPEAKER_00",
            7,
            10,
            "طيب يوسف شو بتنصحنا؟",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            10.5,
            14,
            "نكمل بالخطة.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_repeated_address",
    )

    print_result(
        "TEST 5 — REPEATED ADDRESS",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_01",
        "Yousef",
    )

    assert candidate is not None
    assert candidate.confidence >= 0.60

    evidence_types = [
        evidence.evidence_type.value
        for evidence in candidate.evidence
    ]

    assert "repeated_address" in evidence_types

    print("  PASS")


# ============================================================
# TEST 6 — MULTIPLE SPEAKERS
# ============================================================

def test_multiple_speakers():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            3,
            "يا يوسف شو رأيك؟",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            3.5,
            6,
            "ممتاز.",
        ),
        SpeakerTurn(
            "SPEAKER_02",
            6.5,
            9,
            "وأنا كمان موافق.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_multiple_speakers",
    )

    print_result(
        "TEST 6 — MULTIPLE SPEAKERS",
        result,
    )

    yousef_candidate = find_candidate(
        result,
        "SPEAKER_01",
        "Yousef",
    )

    other_candidate = find_candidate(
        result,
        "SPEAKER_02",
        "Yousef",
    )

    assert yousef_candidate is not None
    assert other_candidate is None

    print("  PASS")


# ============================================================
# TEST 7 — SAME SPEAKER MENTIONING NAME
# ============================================================

def test_same_speaker_name_reference():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            4,
            "يوسف اليوم مش راح يحضر.",
        ),
        SpeakerTurn(
            "SPEAKER_00",
            4.5,
            8,
            "يمكن نحكي معه بكرا.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_same_speaker_reference",
    )

    print_result(
        "TEST 7 — SAME SPEAKER NAME REFERENCE",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_00",
        "Yousef",
    )

    assert candidate is None

    print("  PASS")


# ============================================================
# TEST 8 — NO FALSE IDENTITY FROM GENERIC RESPONSE
# ============================================================

def test_generic_response():
    turns = [
        SpeakerTurn(
            "SPEAKER_00",
            0,
            4,
            "يوسف موجود معنا اليوم.",
        ),
        SpeakerTurn(
            "SPEAKER_01",
            4.5,
            7,
            "تمام.",
        ),
    ]

    result = discover_identities(
        turns,
        meeting_id="robust_generic_response",
    )

    print_result(
        "TEST 8 — GENERIC RESPONSE",
        result,
    )

    candidate = find_candidate(
        result,
        "SPEAKER_01",
        "Yousef",
    )

    assert candidate is None

    print("  PASS")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("IDENTITY DISCOVERY — ROBUST TEST SUITE")
    print("=" * 60)

    tests = [
        test_direct_address,
        test_self_identification,
        test_name_mention_only,
        test_contextual_reference,
        test_repeated_address,
        test_multiple_speakers,
        test_same_speaker_name_reference,
        test_generic_response,
    ]

    passed = 0

    for test in tests:
        test()
        passed += 1

    print()
    print("=" * 60)
    print(
        f"ROBUST IDENTITY DISCOVERY: "
        f"{passed}/{len(tests)} TESTS PASSED"
    )
    print("=" * 60)

    assert passed == len(tests)

    print("PHASE 9.9.3: PASS")


if __name__ == "__main__":
    main()