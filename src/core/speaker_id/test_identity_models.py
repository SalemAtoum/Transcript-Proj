from identity_models import (
    ConfirmationStatus,
    EvidenceType,
    IdentityCandidate,
    IdentityEvidence,
    IdentityStatus,
    ProfileStatus,
    SpeakerIdentity,
    confidence_level_from_score,
    ConfidenceLevel,
)


def main() -> None:
    # ---------------------------------------------------------
    # Test 1: Confidence conversion
    # ---------------------------------------------------------

    assert confidence_level_from_score(0.94) == ConfidenceLevel.HIGH
    assert confidence_level_from_score(0.72) == ConfidenceLevel.MEDIUM
    assert confidence_level_from_score(0.41) == ConfidenceLevel.LOW

    # ---------------------------------------------------------
    # Test 2: Evidence
    # ---------------------------------------------------------

    evidence = IdentityEvidence(
        evidence_type=EvidenceType.DIRECT_ADDRESS,
        speaker_id="SPEAKER_01",
        candidate_name="Yousef",
        confidence=0.20,
        description="Speaker was directly addressed as Yousef.",
        source_start=10.5,
        source_end=12.2,
        related_speaker_id="SPEAKER_00",
    )

    assert evidence.confidence == 0.20
    assert evidence.speaker_id == "SPEAKER_01"
    assert evidence.candidate_name == "Yousef"

    # ---------------------------------------------------------
    # Test 3: Candidate
    # ---------------------------------------------------------

    candidate = IdentityCandidate(
        speaker_id="SPEAKER_01",
        candidate_name="Yousef",
        confidence=0.94,
        confidence_level=ConfidenceLevel.HIGH,
        status=IdentityStatus.PENDING_CONFIRMATION,
        confirmation_status=ConfirmationStatus.PENDING,
        evidence=[evidence],
    )

    assert candidate.speaker_id == "SPEAKER_01"
    assert candidate.candidate_name == "Yousef"
    assert candidate.confidence == 0.94
    assert candidate.evidence_count == 1

    # ---------------------------------------------------------
    # Test 4: Add another evidence
    # ---------------------------------------------------------

    second_evidence = IdentityEvidence(
        evidence_type=EvidenceType.IMMEDIATE_RESPONSE,
        speaker_id="SPEAKER_01",
        candidate_name="Yousef",
        confidence=0.15,
        description="Speaker responded immediately after being addressed.",
        source_start=12.3,
        source_end=14.0,
        related_speaker_id="SPEAKER_00",
    )

    candidate.add_evidence(second_evidence)

    assert candidate.evidence_count == 2
    assert len(candidate.evidence) == 2

    # ---------------------------------------------------------
    # Test 5: Final identity
    # ---------------------------------------------------------

    identity = SpeakerIdentity(
        speaker_id="SPEAKER_01",
        name="Yousef",
        status=IdentityStatus.CONFIRMED,
        confidence=0.94,
        profile_status=ProfileStatus.PROVISIONAL,
        confirmed=True,
        source="conversational_identity_confirmation",
    )

    assert identity.speaker_id == "SPEAKER_01"
    assert identity.name == "Yousef"
    assert identity.confirmed is True
    assert identity.profile_status == ProfileStatus.PROVISIONAL

    # ---------------------------------------------------------
    # PASS
    # ---------------------------------------------------------

    print("IDENTITY MODELS TEST")
    print("--------------------")
    print("Confidence mapping: PASS")
    print("Evidence model: PASS")
    print("Candidate model: PASS")
    print("Evidence accumulation: PASS")
    print("Final identity model: PASS")
    print()
    print("IDENTITY MODELS: PASS")


if __name__ == "__main__":
    main()