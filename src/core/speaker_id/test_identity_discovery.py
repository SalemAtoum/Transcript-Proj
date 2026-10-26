from src.core.speaker_id.identity_discovery import (
    SpeakerTurn,
    discover_identities,
)


def print_candidates(result):
    for candidate in result.candidates:
        print(
            f"{candidate.speaker_id} -> "
            f"{candidate.candidate_name} | "
            f"confidence={candidate.confidence:.2f} | "
            f"level={candidate.confidence_level.value} | "
            f"status={candidate.status.value}"
        )

        for evidence in candidate.evidence:
            print(
                f"  - {evidence.evidence_type.value}: "
                f"{evidence.description}"
            )


def main():

    print("IDENTITY DISCOVERY TEST")
    print("-----------------------")

    # ========================================================
    # TEST 1 — Positive conversational identity discovery
    # ========================================================

    positive_turns = [
        SpeakerTurn(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=3.0,
            text="أهلا يوسف كيفك؟",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_01",
            start=3.5,
            end=6.0,
            text="الحمد لله تمام.",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_00",
            start=7.0,
            end=10.0,
            text="شو رأيك يا يوسف بالمشروع؟",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_01",
            start=10.5,
            end=14.0,
            text="ممتاز أنا شايف إنه ممتاز.",
        ),
    ]

    positive_result = discover_identities(
        positive_turns,
        meeting_id="positive_test",
    )

    print_candidates(positive_result)

    yousef = next(
        (
            candidate
            for candidate in positive_result.candidates
            if candidate.candidate_name == "Yousef"
            and candidate.speaker_id == "SPEAKER_01"
        ),
        None,
    )

    assert yousef is not None, (
        "Positive test failed: "
        "SPEAKER_01 was not identified as Yousef."
    )

    assert yousef.confidence >= 0.60, (
        f"Positive test failed: confidence "
        f"{yousef.confidence:.2f} < 0.60"
    )

    assert yousef.evidence_count >= 2, (
        "Positive test failed: insufficient evidence."
    )

    print()
    print("Positive scenario: PASS")

    # ========================================================
    # TEST 2 — Negative name mention
    # ========================================================

    negative_turns = [
        SpeakerTurn(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=4.0,
            text="يوسف اليوم مش راح يحضر.",
        ),
        SpeakerTurn(
            speaker_id="SPEAKER_01",
            start=4.5,
            end=7.0,
            text="آه تمام.",
        ),
    ]

    negative_result = discover_identities(
        negative_turns,
        meeting_id="negative_test",
    )

    false_yousef = next(
        (
            candidate
            for candidate in negative_result.candidates
            if candidate.candidate_name == "Yousef"
            and candidate.speaker_id == "SPEAKER_01"
        ),
        None,
    )

    assert false_yousef is None, (
        "Negative test failed: "
        "SPEAKER_01 was incorrectly identified as Yousef."
    )

    print("Negative scenario: PASS")

    # ========================================================
    # Final
    # ========================================================

    print()
    print("IDENTITY DISCOVERY: PASS")


if __name__ == "__main__":
    main()