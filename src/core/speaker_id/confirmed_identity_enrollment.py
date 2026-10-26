from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .identity_models import (
    IdentityCandidate,
    IdentityConfirmation,
    IdentityStatus,
)
from .identity_confirmation import (
    IdentityConfirmationManager,
)
from .provisional_enrollment import (
    ProvisionalSpeakerEnrollment,
    ProvisionalEnrollmentResult,
)


class ConfirmedIdentityEnrollment:
    """
    Connect human identity confirmation with provisional
    speaker enrollment.

    Flow:

        IdentityCandidate
            ↓
        IdentityConfirmation
            ↓
        confirmed == True
            ↓
        ProvisionalSpeakerEnrollment
            ↓
        Voice profile created

    If confirmation is rejected, no voice profile is created.
    """

    def __init__(
        self,
        meeting_id: str,
        confirmation_manager: IdentityConfirmationManager | None = None,
        provisional_enrollment: ProvisionalSpeakerEnrollment | None = None,
    ):
        self.meeting_id = meeting_id

        self.confirmation_manager = (
            confirmation_manager
            if confirmation_manager is not None
            else IdentityConfirmationManager(
                meeting_id=meeting_id
            )
        )

        self.provisional_enrollment = (
            provisional_enrollment
            if provisional_enrollment is not None
            else ProvisionalSpeakerEnrollment()
        )

    # ==================================================================
    # CONFIRMATION VALIDATION
    # ==================================================================

    def validate_confirmation(
        self,
        candidate: IdentityCandidate,
        confirmation: IdentityConfirmation,
    ) -> None:
        """
        Validate that the confirmation belongs to the
        current meeting and candidate.
        """

        if confirmation.meeting_id != self.meeting_id:
            raise ValueError(
                "Confirmation meeting_id does not match "
                "the enrollment meeting_id."
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

    # ==================================================================
    # CREATE PROFILE AFTER CONFIRMATION
    # ==================================================================

    def enroll_confirmed_identity(
        self,
        candidate: IdentityCandidate,
        confirmation: IdentityConfirmation,
        *,
        meeting_audio_file: str | Path,
        diarization_file: str | Path,
        profile_output_directory: str | Path,
        temporary_output_directory: str | Path,
    ) -> ProvisionalEnrollmentResult | None:
        """
        Create a provisional voice profile only when the
        identity has been explicitly confirmed.

        Returns:
            ProvisionalEnrollmentResult
                when confirmed.

            None
                when rejected.

        Raises:
            ValueError
                when confirmation does not match candidate/meeting.
        """

        # --------------------------------------------------------------
        # Validate relationship between objects.
        # --------------------------------------------------------------

        self.validate_confirmation(
            candidate=candidate,
            confirmation=confirmation,
        )

        # --------------------------------------------------------------
        # Apply confirmation decision to candidate.
        # --------------------------------------------------------------

        self.confirmation_manager.apply_decision(
            candidate=candidate,
            confirmation=confirmation,
        )

        print()
        print("=" * 70)
        print("CONFIRMED IDENTITY ENROLLMENT")
        print("=" * 70)

        print(
            f"Meeting:       {self.meeting_id}"
        )

        print(
            f"Speaker:       {candidate.speaker_id}"
        )

        print(
            f"Candidate:     {candidate.candidate_name}"
        )

        print(
            f"Confirmed:     {confirmation.confirmed}"
        )

        print(
            f"Candidate status: "
            f"{candidate.status}"
        )

        # --------------------------------------------------------------
        # REJECTED
        # --------------------------------------------------------------

        if not confirmation.confirmed:

            print()
            print(
                "Identity was rejected."
            )

            print(
                "No voice profile will be created."
            )

            print("=" * 70)

            return None

        # --------------------------------------------------------------
        # CONFIRMED
        # --------------------------------------------------------------

        if candidate.status != IdentityStatus.CONFIRMED:
            raise ValueError(
                "Confirmation was marked as confirmed, "
                "but candidate status is not CONFIRMED."
            )

        print()
        print(
            "Identity confirmed."
        )

        print(
            "Starting provisional voice enrollment..."
        )

        # --------------------------------------------------------------
        # Create voice profile.
        # --------------------------------------------------------------

        result = (
            self.provisional_enrollment
            .create_provisional_profile(
                meeting_id=self.meeting_id,
                speaker_id=candidate.speaker_id,
                employee_name=candidate.candidate_name,
                meeting_audio_file=meeting_audio_file,
                diarization_file=diarization_file,
                profile_output_directory=(
                    profile_output_directory
                ),
                temporary_output_directory=(
                    temporary_output_directory
                ),
            )
        )

        print()
        print(
            "Confirmed identity successfully enrolled."
        )

        print(
            f"Profile: {result.profile_file}"
        )

        print("=" * 70)

        return result

    # ==================================================================
    # SAVE RESULT
    # ==================================================================

    @staticmethod
    def save_result(
        result: ProvisionalEnrollmentResult,
        output_file: str | Path,
    ) -> Path:
        """
        Save the enrollment result as JSON.
        """

        output_path = Path(
            output_file
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload: dict[str, Any] = asdict(
            result
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path


# ======================================================================
# CONTROLLED FUNCTIONAL TEST
# ======================================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    # --------------------------------------------------------------
    # Existing controlled test artifacts.
    # --------------------------------------------------------------

    meeting_audio = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_test_60s.wav"
    )

    diarization_file = (
        project_root
        / "data"
        / "temporary"
        / "phase9_diarization_result.json"
    )

    profile_directory = (
        project_root
        / "data"
        / "temporary"
        / "phase9_enrollment_test"
    )

    temporary_directory = (
        project_root
        / "data"
        / "temporary"
    )

    print()
    print("=" * 70)
    print(
        "PHASE 9.9.8B — CONFIRMED IDENTITY ENROLLMENT"
    )
    print("=" * 70)

    # ==============================================================
    # TEST A — CONFIRMED
    # ==============================================================

    print()
    print("=" * 70)
    print("TEST A — CONFIRMED IDENTITY")
    print("=" * 70)

    from .identity_models import (
        ConfirmationStatus,
    )

    candidate = IdentityCandidate(
        speaker_id="SPEAKER_01",
        candidate_name="Yousef",
        confidence=0.75,
        confidence_level="medium",
        status=IdentityStatus.PENDING_CONFIRMATION,
        confirmation_status=ConfirmationStatus.PENDING,
    )

    manager = IdentityConfirmationManager(
        meeting_id="confirmed_enrollment_test"
    )

    confirmation = (
        manager.create_confirmation(
            candidate
        )
    )

    manager.confirm(
        confirmation,
        confirmed_by="test_user",
        note="Controlled functional test.",
    )

    workflow = ConfirmedIdentityEnrollment(
        meeting_id="confirmed_enrollment_test",
        confirmation_manager=manager,
    )

    result = (
        workflow.enroll_confirmed_identity(
            candidate=candidate,
            confirmation=confirmation,
            meeting_audio_file=meeting_audio,
            diarization_file=diarization_file,
            profile_output_directory=(
                profile_directory
            ),
            temporary_output_directory=(
                temporary_directory
            ),
        )
    )

    if result is None:
        raise AssertionError(
            "TEST A failed: confirmed identity "
            "did not create a profile."
        )

    print()
    print(
        "TEST A: PASS"
    )

    # ==============================================================
    # TEST B — REJECTED
    # ==============================================================

    print()
    print("=" * 70)
    print("TEST B — REJECTED IDENTITY")
    print("=" * 70)

    candidate_rejected = IdentityCandidate(
        speaker_id="SPEAKER_02",
        candidate_name="Ahmad",
        confidence=0.70,
        confidence_level="medium",
        status=IdentityStatus.PENDING_CONFIRMATION,
        confirmation_status=ConfirmationStatus.PENDING,
    )

    manager_rejected = (
        IdentityConfirmationManager(
            meeting_id="rejected_enrollment_test"
        )
    )

    confirmation_rejected = (
        manager_rejected.create_confirmation(
            candidate_rejected
        )
    )

    manager_rejected.reject(
        confirmation_rejected,
        rejected_by="test_user",
        note="Controlled rejection test.",
    )

    workflow_rejected = (
        ConfirmedIdentityEnrollment(
            meeting_id="rejected_enrollment_test",
            confirmation_manager=manager_rejected,
        )
    )

    result_rejected = (
        workflow_rejected
        .enroll_confirmed_identity(
            candidate=candidate_rejected,
            confirmation=confirmation_rejected,
            meeting_audio_file=meeting_audio,
            diarization_file=diarization_file,
            profile_output_directory=(
                profile_directory
            ),
            temporary_output_directory=(
                temporary_directory
            ),
        )
    )

    if result_rejected is not None:
        raise AssertionError(
            "TEST B failed: rejected identity "
            "created a profile."
        )

    if (
        candidate_rejected.status
        != IdentityStatus.REJECTED
    ):
        raise AssertionError(
            "TEST B failed: candidate status "
            "is not REJECTED."
        )

    print()
    print(
        "TEST B: PASS"
    )

    # ==============================================================
    # FINAL
    # ==============================================================

    print()
    print("=" * 70)
    print(
        "PHASE 9.9.8B: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()