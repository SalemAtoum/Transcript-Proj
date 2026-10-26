"""
SWBS Meeting Intelligence
Conversational Identity Discovery - Data Models

Phase 9.9.1

This module contains the core data structures used by the
Conversational Identity Discovery system.

Important:
- This module contains NO AI inference.
- This module contains NO database logic.
- This module contains NO speaker matching.
- This module contains NO transcript modification.

It only defines structured representations for:
    - identity evidence
    - confidence
    - identity candidates
    - discovery results
    - user confirmation
    - profile status
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    """
    Types of evidence that can support a speaker identity.

    The values are intentionally stable because they may later be
    persisted in JSON/database records.
    """

    SELF_IDENTIFICATION = "self_identification"
    DIRECT_ADDRESS = "direct_address"
    IMMEDIATE_RESPONSE = "immediate_response"
    REPEATED_ADDRESS = "repeated_address"
    CONVERSATION_CONTINUITY = "conversation_continuity"
    VOICE_CONSISTENCY = "voice_consistency"
    CONTEXT_REFERENCE = "context_reference"


class ConfidenceLevel(str, Enum):
    """
    Human-readable confidence bands.

    These bands are separate from the numerical confidence score.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IdentityStatus(str, Enum):
    """
    Lifecycle state of an identity discovery result.
    """

    CANDIDATE = "candidate"
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNIDENTIFIED = "unidentified"


class ConfirmationStatus(str, Enum):
    """
    Status of human confirmation for a discovered identity.
    """

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ProfileStatus(str, Enum):
    """
    Status of a speaker voice profile.

    PROVISIONAL:
        Created after a conversational identity discovery has been
        confirmed, but has not yet accumulated enough independent
        successful matches.

    VERIFIED:
        Has accumulated sufficient evidence across meetings.
    """

    PROVISIONAL = "provisional"
    VERIFIED = "verified"


@dataclass
class IdentityEvidence:
    """
    One piece of evidence supporting a possible identity.

    Example:

        evidence_type = DIRECT_ADDRESS
        speaker_id = SPEAKER_01
        candidate_name = Yousef
        confidence = 0.20
        description = "Another speaker directly addressed this speaker as Yousef."
    """

    evidence_type: EvidenceType
    speaker_id: str
    candidate_name: str

    confidence: float

    description: str = ""

    source_start: float | None = None
    source_end: float | None = None

    related_speaker_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        if self.source_start is not None:
            self.source_start = max(0.0, float(self.source_start))

        if self.source_end is not None:
            self.source_end = max(0.0, float(self.source_end))


@dataclass
class IdentityCandidate:
    """
    A candidate identity proposed for a diarized speaker.

    Example:

        speaker_id:
            SPEAKER_01

        candidate_name:
            Yousef

        confidence:
            0.94

        confidence_level:
            HIGH

        status:
            PENDING_CONFIRMATION
    """

    speaker_id: str
    candidate_name: str

    confidence: float
    confidence_level: ConfidenceLevel

    status: IdentityStatus = IdentityStatus.CANDIDATE

    evidence: list[IdentityEvidence] = field(default_factory=list)

    confirmation_status: ConfirmationStatus = (
        ConfirmationStatus.NOT_REQUIRED
    )

    profile_status: ProfileStatus | None = None

    evidence_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        self.evidence_count = len(self.evidence)

    def add_evidence(self, evidence: IdentityEvidence) -> None:
        """
        Add one evidence item and keep the count synchronized.
        """

        self.evidence.append(evidence)
        self.evidence_count = len(self.evidence)


@dataclass
class IdentityDiscoveryResult:
    """
    Complete result produced by the future identity discovery engine.

    One meeting may contain multiple candidates, for example:

        SPEAKER_00 -> Sami
        SPEAKER_01 -> Yousef
        SPEAKER_02 -> Ahmad
    """

    meeting_id: str

    candidates: list[IdentityCandidate] = field(default_factory=list)

    discovered_at: str | None = None

    total_speakers: int = 0

    speakers_with_candidates: int = 0

    speakers_requiring_confirmation: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.speakers_with_candidates = len(self.candidates)

        self.speakers_requiring_confirmation = sum(
            1
            for candidate in self.candidates
            if candidate.confirmation_status
            == ConfirmationStatus.PENDING
        )


@dataclass
class IdentityConfirmation:
    """
    Represents the employee's answer to an identity confirmation prompt.

    Example:

        speaker_id = SPEAKER_01
        candidate_name = Yousef
        confirmed = True
    """

    meeting_id: str

    speaker_id: str
    candidate_name: str

    confirmed: bool

    confirmed_by: str | None = None

    confirmation_note: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerIdentity:
    """
    Final identity state for a speaker.

    This is intentionally separate from IdentityCandidate.

    Candidate:
        "We think this may be Yousef."

    SpeakerIdentity:
        "This speaker is Yousef."

    The future pipeline will use SpeakerIdentity when generating
    the final named transcript.
    """

    speaker_id: str
    name: str

    status: IdentityStatus

    confidence: float

    profile_status: ProfileStatus | None = None

    confirmed: bool = False

    source: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


def confidence_level_from_score(score: float) -> ConfidenceLevel:
    """
    Convert a numerical confidence score into a confidence band.

    Current specification:

        0.80 - 1.00 -> HIGH
        0.60 - 0.79 -> MEDIUM
        0.00 - 0.59 -> LOW

    These thresholds are initial engineering values.
    They must be validated against real meeting data later.
    """

    score = max(0.0, min(1.0, float(score)))

    if score >= 0.80:
        return ConfidenceLevel.HIGH

    if score >= 0.60:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW