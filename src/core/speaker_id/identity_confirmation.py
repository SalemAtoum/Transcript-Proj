from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .identity_models import (
    ConfirmationStatus,
    IdentityCandidate,
    IdentityConfirmation,
    IdentityStatus,
)


# ============================================================
# IDENTITY CONFIRMATION MANAGER
# ============================================================


class IdentityConfirmationManager:
    """
    Manage human confirmation/rejection of identity candidates.

    Responsibilities:
        - Create confirmation requests
        - Confirm a candidate
        - Reject a candidate
        - Apply the decision to IdentityCandidate
        - Keep confirmation metadata

    This class does NOT:
        - discover identities
        - modify transcript text
        - perform speaker diarization
        - perform voice identification
        - access the database
    """

    def __init__(self, meeting_id: str):
        self.meeting_id = meeting_id

    # ========================================================
    # CREATE CONFIRMATION
    # ========================================================

    def create_confirmation(
        self,
        candidate: IdentityCandidate,
    ) -> IdentityConfirmation:
        """
        Create a pending confirmation request.

        The IdentityConfirmation model uses:
            confirmed: bool

        Therefore a newly created confirmation request
        starts with confirmed=False.
        """

        return IdentityConfirmation(
            meeting_id=self.meeting_id,
            speaker_id=candidate.speaker_id,
            candidate_name=candidate.candidate_name,
            confirmed=False,
            confirmed_by=None,
            confirmation_note=None,
            metadata={
                "confidence": candidate.confidence,
                "confidence_level": candidate.confidence_level,
                "candidate_status": (
                    candidate.status.value
                    if hasattr(candidate.status, "value")
                    else str(candidate.status)
                ),
            },
        )

    # ========================================================
    # CONFIRM
    # ========================================================

    def confirm(
        self,
        confirmation: IdentityConfirmation,
        confirmed_by: str = "user",
        note: str | None = None,
    ) -> IdentityConfirmation:
        """
        Mark a confirmation request as confirmed.
        """

        confirmation.confirmed = True
        confirmation.confirmed_by = confirmed_by
        confirmation.confirmation_note = note

        return confirmation

    # ========================================================
    # REJECT
    # ========================================================

    def reject(
        self,
        confirmation: IdentityConfirmation,
        rejected_by: str = "user",
        note: str | None = None,
    ) -> IdentityConfirmation:
        """
        Mark a confirmation request as rejected.

        The IdentityConfirmation model represents rejection
        using confirmed=False plus confirmation metadata.
        """

        confirmation.confirmed = False
        confirmation.confirmed_by = rejected_by
        confirmation.confirmation_note = note

        return confirmation

    # ========================================================
    # APPLY DECISION
    # ========================================================

    def apply_decision(
        self,
        candidate: IdentityCandidate,
        confirmation: IdentityConfirmation,
    ) -> IdentityCandidate:
        """
        Apply the confirmation decision to the candidate.

        Confirmed:
            IdentityStatus.CONFIRMED
            ConfirmationStatus.CONFIRMED

        Rejected:
            IdentityStatus.REJECTED
            ConfirmationStatus.REJECTED
        """

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if confirmation.meeting_id != self.meeting_id:
            raise ValueError(
                "Confirmation meeting_id does not match "
                "the manager meeting_id."
            )

        if confirmation.speaker_id != candidate.speaker_id:
            raise ValueError(
                "Confirmation speaker_id does not match "
                "the candidate speaker_id."
            )

        if (
            confirmation.candidate_name
            != candidate.candidate_name
        ):
            raise ValueError(
                "Confirmation candidate_name does not match "
                "the candidate."
            )

        # ----------------------------------------------------
        # CONFIRMED
        # ----------------------------------------------------

        if confirmation.confirmed:

            candidate.status = (
                IdentityStatus.CONFIRMED
            )

            candidate.confirmation_status = (
                ConfirmationStatus.CONFIRMED
            )

        # ----------------------------------------------------
        # REJECTED
        # ----------------------------------------------------

        else:

            candidate.status = (
                IdentityStatus.REJECTED
            )

            candidate.confirmation_status = (
                ConfirmationStatus.REJECTED
            )

        # ----------------------------------------------------
        # Store confirmation metadata when supported
        # ----------------------------------------------------

        if hasattr(candidate, "metadata"):

            candidate.metadata.update(
                {
                    "confirmation": asdict(
                        confirmation
                    )
                }
            )

        return candidate

    # ========================================================
    # SERIALIZATION
    # ========================================================

    @staticmethod
    def to_dict(
        confirmation: IdentityConfirmation,
    ) -> dict[str, Any]:
        """
        Convert IdentityConfirmation into a JSON-safe dict.
        """

        return asdict(confirmation)