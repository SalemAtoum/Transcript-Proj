from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .identity_models import (
    ConfirmationStatus,
    EvidenceType,
    IdentityCandidate,
    IdentityDiscoveryResult,
    IdentityEvidence,
    IdentityStatus,
    confidence_level_from_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SELF_IDENTIFICATION_WEIGHT = 0.40
DIRECT_ADDRESS_WEIGHT = 0.20
IMMEDIATE_RESPONSE_WEIGHT = 0.15
REPEATED_ADDRESS_WEIGHT = 0.10
CONVERSATION_CONTINUITY_WEIGHT = 0.10
CONTEXT_REFERENCE_WEIGHT = 0.05
VOICE_CONSISTENCY_WEIGHT = 0.15

HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.60

IMMEDIATE_RESPONSE_MAX_GAP = 8.0
CONVERSATION_CONTINUITY_MAX_GAP = 20.0

INTERACTION_BONUS_PER_EVENT = 0.05
MAX_INTERACTION_BONUS = 0.20


# ============================================================
# EVIDENCE WEIGHTS
# ============================================================

EVIDENCE_WEIGHTS = {
    EvidenceType.SELF_IDENTIFICATION: SELF_IDENTIFICATION_WEIGHT,
    EvidenceType.DIRECT_ADDRESS: DIRECT_ADDRESS_WEIGHT,
    EvidenceType.IMMEDIATE_RESPONSE: IMMEDIATE_RESPONSE_WEIGHT,
    EvidenceType.REPEATED_ADDRESS: REPEATED_ADDRESS_WEIGHT,
    EvidenceType.CONVERSATION_CONTINUITY: CONVERSATION_CONTINUITY_WEIGHT,
    EvidenceType.CONTEXT_REFERENCE: CONTEXT_REFERENCE_WEIGHT,
    EvidenceType.VOICE_CONSISTENCY: VOICE_CONSISTENCY_WEIGHT,
}


# ============================================================
# SPEAKER TURN
# ============================================================

@dataclass
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float
    text: str


# ============================================================
# KNOWN NAMES
# ============================================================

ARABIC_NAME_ALIASES = {
    "يوسف": "Yousef",
    "يو سف": "Yousef",
    "سالم": "Salem",
    "سلامة": "Salameh",
    "سلامه": "Salameh",
    "أحمد": "Ahmad",
    "احمد": "Ahmad",
    "محمد": "Mohammad",
    "محمود": "Mahmoud",
    "عمر": "Omar",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    # Normalize common Arabic variants.
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ـ", "")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_name(name: str) -> str:
    return normalize_text(name)


# ============================================================
# CANONICAL NAME
# ============================================================

def canonical_name(name: str) -> str | None:
    normalized = normalize_name(name)

    if not normalized:
        return None

    # Already canonical English name.
    for canonical in ARABIC_NAME_ALIASES.values():
        if normalize_name(canonical) == normalized:
            return canonical

    # Arabic alias.
    for alias, canonical in ARABIC_NAME_ALIASES.items():
        if normalize_name(alias) == normalized:
            return canonical

    return None


def aliases_for_canonical_name(
    canonical: str,
) -> list[str]:
    resolved = canonical_name(canonical)

    if resolved is None:
        return []

    return [
        alias
        for alias, mapped_name in ARABIC_NAME_ALIASES.items()
        if mapped_name == resolved
    ]


# ============================================================
# NAME DETECTION
# ============================================================

def extract_known_name_mentions(
    text: str,
) -> list[str]:
    normalized = normalize_text(text)

    if not normalized:
        return []

    found: list[str] = []

    aliases = sorted(
        ARABIC_NAME_ALIASES.items(),
        key=lambda item: len(normalize_name(item[0])),
        reverse=True,
    )

    for alias, canonical in aliases:
        alias_normalized = normalize_name(alias)

        pattern = (
            rf"(?<!\w)"
            rf"{re.escape(alias_normalized)}"
            rf"(?!\w)"
        )

        if re.search(pattern, normalized):
            if canonical not in found:
                found.append(canonical)

    return found


# ============================================================
# DIRECT ADDRESS
# ============================================================

def looks_like_direct_address(
    text: str,
    candidate_name: str,
) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return False

    canonical = canonical_name(candidate_name)

    if canonical is None:
        return False

    aliases = aliases_for_canonical_name(canonical)

    for alias in aliases:
        name = normalize_name(alias)

        # ----------------------------------------------------
        # يا يوسف
        # ----------------------------------------------------

        if re.search(
            rf"\bيا\s+{re.escape(name)}\b",
            normalized,
        ):
            return True

        # ----------------------------------------------------
        # أهلا يوسف
        # مرحبا يوسف
        # سلام يوسف
        # يعطيك العافية يوسف
        # ----------------------------------------------------

        greeting_pattern = (
            rf"(اهلا|مرحبا|سلام|يعطيك العافية)"
            rf"\s+{re.escape(name)}"
            rf"(?:\s|[،,.!?؟]|$)"
        )

        if re.search(
            greeting_pattern,
            normalized,
        ):
            return True

        # ----------------------------------------------------
        # يوسف شو رأيك؟
        # يوسف كيفك؟
        # يوسف شو بتنصحنا؟
        # ----------------------------------------------------

        beginning_pattern = (
            rf"^{re.escape(name)}"
            rf"\s+"
            rf"(كيفك|شو|ايش|ليش|وين|متى|هل|بتقدر|ممكن|"
            rf"بتوافق|بتشوف|بتنصح|رأيك|شو بتنصح)"
        )

        if re.search(
            beginning_pattern,
            normalized,
        ):
            return True

        # ----------------------------------------------------
        # طيب يوسف شو بتنصحنا؟
        # طيب يوسف كيفك؟
        # يعني يوسف شو رأيك؟
        #
        # The name can appear after conversational fillers.
        # ----------------------------------------------------

        conversational_prefix_pattern = (
            rf"^(?:طيب|يعني|هسا|هلأ|بس|طب|تمام|"
            rf"اوكي|اوكيه|اسمع|شوف)"
            rf"\s+"
            rf"{re.escape(name)}"
            rf"\s+"
            rf"(?:كيفك|شو|ايش|ليش|وين|متى|هل|"
            rf"بتقدر|ممكن|بتوافق|بتشوف|بتنصح|"
            rf"رأيك|شو بتنصح)"
        )

        if re.search(
            conversational_prefix_pattern,
            normalized,
        ):
            return True

        # ----------------------------------------------------
        # Generic question containing the candidate name.
        #
        # Example:
        # "طيب يوسف شو بتنصحنا؟"
        # "يوسف شو رأيك؟"
        # ----------------------------------------------------

        question_address_pattern = (
            rf"(?<!\w)"
            rf"{re.escape(name)}"
            rf"(?:\s+.*)?"
            rf"[؟?]"
        )

        if re.search(
            question_address_pattern,
            normalized,
        ):
            return True

        # ----------------------------------------------------
        # كيفك يوسف؟
        # شو رأيك يوسف؟
        # ----------------------------------------------------

        ending_pattern = (
            rf"(كيفك|شو رأيك|ايش رأيك|"
            rf"بتوافق|بتشوف|ممكن|بتنصح)"
            rf".*"
            rf"{re.escape(name)}"
            rf"\s*[؟?]?$"
        )

        if re.search(
            ending_pattern,
            normalized,
        ):
            return True

    return False


# ============================================================
# TIMING
# ============================================================

def is_immediate_response(
    previous_turn: SpeakerTurn,
    response_turn: SpeakerTurn,
) -> bool:
    if previous_turn.speaker_id == response_turn.speaker_id:
        return False

    gap = response_turn.start - previous_turn.end

    return (
        0 <= gap <= IMMEDIATE_RESPONSE_MAX_GAP
    )


def is_continuous_response(
    previous_turn: SpeakerTurn,
    response_turn: SpeakerTurn,
) -> bool:
    if previous_turn.speaker_id == response_turn.speaker_id:
        return False

    gap = response_turn.start - previous_turn.end

    return (
        0 <= gap <= CONVERSATION_CONTINUITY_MAX_GAP
    )


# ============================================================
# SELF IDENTIFICATION
# ============================================================

def looks_like_self_identification(
    text: str,
) -> str | None:
    normalized = normalize_text(text)

    patterns = [
        r"انا\s+([^\s،,.!?؟]+)",
        r"اسمي\s+([^\s،,.!?؟]+)",
        r"معك\s+([^\s،,.!?؟]+)",
        r"معكم\s+([^\s،,.!?؟]+)",
        r"معاكم\s+([^\s،,.!?؟]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)

        if not match:
            continue

        possible_name = match.group(1)

        canonical = canonical_name(possible_name)

        if canonical:
            return canonical

    return None


# ============================================================
# IDENTITY DISCOVERY ENGINE
# ============================================================

class IdentityDiscoveryEngine:

    def __init__(
        self,
        high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    # ========================================================
    # MAIN DISCOVERY
    # ========================================================

    def discover(
        self,
        turns: Iterable[SpeakerTurn],
        meeting_id: str = "identity_discovery_test",
    ) -> IdentityDiscoveryResult:

        ordered_turns = sorted(
            list(turns),
            key=lambda turn: (
                turn.start,
                turn.end,
            ),
        )

        candidates: dict[
            tuple[str, str],
            IdentityCandidate,
        ] = {}

        # ====================================================
        # Candidate helper
        # ====================================================

        def get_candidate(
            speaker_id: str,
            candidate_name: str,
        ) -> IdentityCandidate:

            key = (
                speaker_id,
                candidate_name,
            )

            if key not in candidates:
                candidates[key] = IdentityCandidate(
                    speaker_id=speaker_id,
                    candidate_name=candidate_name,
                    confidence=0.0,
                    confidence_level=(
                        confidence_level_from_score(0.0)
                    ),
                )

            return candidates[key]

        # ====================================================
        # Analyze turns
        # ====================================================

        for index, turn in enumerate(ordered_turns):

            # ------------------------------------------------
            # SELF IDENTIFICATION
            # ------------------------------------------------

            self_name = looks_like_self_identification(
                turn.text
            )

            if self_name:
                candidate = get_candidate(
                    turn.speaker_id,
                    self_name,
                )

                candidate.add_evidence(
                    IdentityEvidence(
                        evidence_type=(
                            EvidenceType.SELF_IDENTIFICATION
                        ),
                        speaker_id=turn.speaker_id,
                        candidate_name=self_name,
                        confidence=SELF_IDENTIFICATION_WEIGHT,
                        description=(
                            "The speaker explicitly "
                            "identified themselves by name."
                        ),
                        source_start=turn.start,
                        source_end=turn.end,
                    )
                )

            # ------------------------------------------------
            # NAME MENTIONS
            # ------------------------------------------------

            mentioned_names = extract_known_name_mentions(
                turn.text
            )

            if not mentioned_names:
                continue

            # Need another turn.
            if index + 1 >= len(ordered_turns):
                continue

            next_turn = ordered_turns[index + 1]

            # Same speaker cannot be the response.
            if next_turn.speaker_id == turn.speaker_id:
                continue

            # ------------------------------------------------
            # DIRECT ADDRESS
            # ------------------------------------------------

            for name in mentioned_names:

                if not looks_like_direct_address(
                    turn.text,
                    name,
                ):
                    continue

                candidate = get_candidate(
                    next_turn.speaker_id,
                    name,
                )

                candidate.add_evidence(
                    IdentityEvidence(
                        evidence_type=(
                            EvidenceType.DIRECT_ADDRESS
                        ),
                        speaker_id=next_turn.speaker_id,
                        candidate_name=name,
                        confidence=DIRECT_ADDRESS_WEIGHT,
                        description=(
                            "The candidate name "
                            "was used as a direct "
                            "conversational address."
                        ),
                        source_start=turn.start,
                        source_end=turn.end,
                        related_speaker_id=turn.speaker_id,
                    )
                )

                # --------------------------------------------
                # IMMEDIATE RESPONSE
                # --------------------------------------------

                if is_immediate_response(
                    turn,
                    next_turn,
                ):
                    candidate.add_evidence(
                        IdentityEvidence(
                            evidence_type=(
                                EvidenceType.IMMEDIATE_RESPONSE
                            ),
                            speaker_id=next_turn.speaker_id,
                            candidate_name=name,
                            confidence=IMMEDIATE_RESPONSE_WEIGHT,
                            description=(
                                "The candidate speaker "
                                "responded immediately "
                                "after being addressed."
                            ),
                            source_start=next_turn.start,
                            source_end=next_turn.end,
                            related_speaker_id=turn.speaker_id,
                        )
                    )

                # --------------------------------------------
                # CONVERSATION CONTINUITY
                # --------------------------------------------

                if is_continuous_response(
                    turn,
                    next_turn,
                ):
                    candidate.add_evidence(
                        IdentityEvidence(
                            evidence_type=(
                                EvidenceType.CONVERSATION_CONTINUITY
                            ),
                            speaker_id=next_turn.speaker_id,
                            candidate_name=name,
                            confidence=(
                                CONVERSATION_CONTINUITY_WEIGHT
                            ),
                            description=(
                                "The candidate speaker "
                                "participated in the "
                                "immediate conversational "
                                "exchange."
                            ),
                            source_start=next_turn.start,
                            source_end=next_turn.end,
                            related_speaker_id=turn.speaker_id,
                        )
                    )

        # ====================================================
        # REPEATED ADDRESS
        # ====================================================

        self._add_repeated_address_evidence(
            ordered_turns,
            candidates,
        )

        # ====================================================
        # FINAL SCORING
        # ====================================================

        finalized = self._finalize_candidates(
            candidates
        )

        return IdentityDiscoveryResult(
            meeting_id=meeting_id,
            candidates=finalized,
        )

    # ========================================================
    # REPEATED ADDRESS
    # ========================================================

    def _add_repeated_address_evidence(
        self,
        turns: list[SpeakerTurn],
        candidates: dict[
            tuple[str, str],
            IdentityCandidate,
        ],
    ) -> None:

        address_events: dict[
            tuple[str, str],
            int,
        ] = {}

        for index, turn in enumerate(turns):

            if index + 1 >= len(turns):
                continue

            next_turn = turns[index + 1]

            if turn.speaker_id == next_turn.speaker_id:
                continue

            names = extract_known_name_mentions(
                turn.text
            )

            for name in names:

                if not looks_like_direct_address(
                    turn.text,
                    name,
                ):
                    continue

                key = (
                    next_turn.speaker_id,
                    name,
                )

                address_events[key] = (
                    address_events.get(
                        key,
                        0,
                    )
                    + 1
                )

        for key, count in address_events.items():

            if count < 2:
                continue

            candidate = candidates.get(key)

            if candidate is None:
                continue

            candidate.add_evidence(
                IdentityEvidence(
                    evidence_type=(
                        EvidenceType.REPEATED_ADDRESS
                    ),
                    speaker_id=candidate.speaker_id,
                    candidate_name=candidate.candidate_name,
                    confidence=REPEATED_ADDRESS_WEIGHT,
                    description=(
                        "The candidate was repeatedly "
                        "addressed across separate "
                        "conversational turns."
                    ),
                    metadata={
                        "address_count": count,
                    },
                )
            )

    # ========================================================
    # FINALIZE
    # ========================================================

    def _finalize_candidates(
        self,
        candidates: dict[
            tuple[str, str],
            IdentityCandidate,
        ],
    ) -> list[IdentityCandidate]:

        finalized: list[IdentityCandidate] = []

        for candidate in candidates.values():

            # ------------------------------------------------
            # One contribution per evidence type.
            # ------------------------------------------------

            base_score = 0.0

            evidence_types_seen: set[
                EvidenceType
            ] = set()

            for evidence in candidate.evidence:

                if evidence.evidence_type in evidence_types_seen:
                    continue

                evidence_types_seen.add(
                    evidence.evidence_type
                )

                base_score += EVIDENCE_WEIGHTS.get(
                    evidence.evidence_type,
                    0.0,
                )

            # ------------------------------------------------
            # Interaction bonus.
            # ------------------------------------------------

            interaction_events = sum(
                1
                for evidence in candidate.evidence
                if evidence.evidence_type
                in {
                    EvidenceType.DIRECT_ADDRESS,
                    EvidenceType.IMMEDIATE_RESPONSE,
                }
            )

            interaction_bonus = min(
                interaction_events
                * INTERACTION_BONUS_PER_EVENT,
                MAX_INTERACTION_BONUS,
            )

            score = min(
                1.0,
                base_score + interaction_bonus,
            )

            # ------------------------------------------------
            # Update confidence.
            # ------------------------------------------------

            candidate.confidence = round(
                score,
                4,
            )

            candidate.confidence_level = (
                confidence_level_from_score(score)
            )

            candidate.evidence_count = len(
                candidate.evidence
            )

            # ------------------------------------------------
            # Status.
            # ------------------------------------------------

            if score >= self.high_threshold:
                candidate.status = (
                    IdentityStatus.PENDING_CONFIRMATION
                )

                candidate.confirmation_status = (
                    ConfirmationStatus.PENDING
                )

            elif score >= self.medium_threshold:
                candidate.status = (
                    IdentityStatus.PENDING_CONFIRMATION
                )

                candidate.confirmation_status = (
                    ConfirmationStatus.PENDING
                )

            else:
                candidate.status = (
                    IdentityStatus.CANDIDATE
                )

                candidate.confirmation_status = (
                    ConfirmationStatus.NOT_REQUIRED
                )

            finalized.append(candidate)

        # Strongest candidates first.
        finalized.sort(
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )

        return finalized


# ============================================================
# CONVENIENCE API
# ============================================================

def discover_identities(
    turns: Iterable[SpeakerTurn],
    meeting_id: str = "identity_discovery_test",
) -> IdentityDiscoveryResult:

    engine = IdentityDiscoveryEngine()

    return engine.discover(
        turns,
        meeting_id=meeting_id,
    )