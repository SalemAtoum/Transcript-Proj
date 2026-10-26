from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Required when this file is executed directly by subprocess.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.speaker_id.identity_discovery import (
    SpeakerTurn,
    discover_identities,
)


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


# ============================================================
# PRODUCTION IDENTITY DISCOVERY
# ============================================================

class ProductionIdentityDiscovery:
    """
    Production adapter for conversational identity discovery.

    Responsibilities:
        1. Load the meeting-specific alignment result.
        2. Convert aligned speaker turns into SpeakerTurn objects.
        3. Run conversational identity discovery.
        4. Save the discovery result for downstream systems.

    IMPORTANT:
        This component does NOT modify the final transcript.

        Identity candidates remain candidates until a later
        confirmation workflow explicitly confirms them.
    """

    def __init__(
        self,
        alignment_file: Path,
        output_file: Path,
        meeting_id: str,
    ):
        self.alignment_file = Path(
            alignment_file
        ).resolve()

        self.output_file = Path(
            output_file
        ).resolve()

        self.meeting_id = meeting_id

    # ========================================================
    # JSON SERIALIZATION
    # ========================================================

    @staticmethod
    def _serialize(value: Any) -> Any:

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return {
                key: ProductionIdentityDiscovery._serialize(
                    val
                )
                for key, val in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                str(key): ProductionIdentityDiscovery._serialize(
                    val
                )
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                ProductionIdentityDiscovery._serialize(
                    item
                )
                for item in value
            ]

        return value

    # ========================================================
    # LOAD ALIGNMENT
    # ========================================================

    def _load_alignment_turns(
        self,
    ) -> list[SpeakerTurn]:

        if not self.alignment_file.exists():

            raise FileNotFoundError(
                "Alignment result not found:\n"
                f"{self.alignment_file}"
            )

        with self.alignment_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        raw_turns = data.get(
            "speaker_turns",
            [],
        )

        if not isinstance(
            raw_turns,
            list,
        ):
            raise ValueError(
                "Invalid alignment result: "
                "'speaker_turns' must be a list."
            )

        turns: list[SpeakerTurn] = []

        for index, item in enumerate(
            raw_turns
        ):

            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    "Invalid alignment turn at index "
                    f"{index}: expected an object."
                )

            # ------------------------------------------------
            # Speaker
            # ------------------------------------------------

            speaker_id = item.get(
                "speaker"
            )

            if speaker_id is None:

                speaker_id = item.get(
                    "speaker_id"
                )

            if speaker_id is None:

                raise ValueError(
                    "Alignment turn is missing "
                    f"speaker ID at index {index}."
                )

            # ------------------------------------------------
            # Start
            # ------------------------------------------------

            start = item.get(
                "start"
            )

            if start is None:

                start = item.get(
                    "start_seconds"
                )

            # ------------------------------------------------
            # End
            # ------------------------------------------------

            end = item.get(
                "end"
            )

            if end is None:

                end = item.get(
                    "end_seconds"
                )

            if start is None or end is None:

                raise ValueError(
                    "Alignment turn is missing "
                    f"start/end time at index {index}."
                )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

            # Empty turns are ignored safely.
            if not text:
                continue

            turns.append(
                SpeakerTurn(
                    speaker_id=str(
                        speaker_id
                    ),
                    start=float(
                        start
                    ),
                    end=float(
                        end
                    ),
                    text=text,
                )
            )

        return turns

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
    ) -> dict[str, Any]:

        turns = self._load_alignment_turns()

        detected_speakers = sorted(
            {
                turn.speaker_id
                for turn in turns
            }
        )

        result = discover_identities(
            turns,
            meeting_id=self.meeting_id,
        )

        serialized_candidates = [
            self._serialize(
                candidate
            )
            for candidate in result.candidates
        ]

        output = {
            "schema_version": "1.0",

            "phase": "9.9.5",

            "meeting_id": self.meeting_id,

            "status": "completed",

            "speaker_turn_count": len(
                turns
            ),

            "detected_speakers": (
                detected_speakers
            ),

            "candidate_count": len(
                serialized_candidates
            ),

            "candidates": (
                serialized_candidates
            ),

            # Critical safety flag:
            # Identity discovery never changes
            # the transcript at this stage.
            "transcript_modified": False,
        }

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 70)

    print(
        "PHASE 9.9.5C — PRODUCTION IDENTITY DISCOVERY"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    alignment_file = Path(
        get_required_env(
            "SWBS_ALIGNMENT_FILE"
        )
    )

    output_file = Path(
        get_required_env(
            "SWBS_OUTPUT_FILE"
        )
    )

    meeting_id = get_required_env(
        "SWBS_MEETING_ID"
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("Meeting ID:")
    print(
        f"  {meeting_id}"
    )

    print()

    print("Alignment:")
    print(
        f"  {alignment_file}"
    )

    print()

    print("Output:")
    print(
        f"  {output_file}"
    )

    print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    service = ProductionIdentityDiscovery(
        alignment_file=alignment_file,
        output_file=output_file,
        meeting_id=meeting_id,
    )

    result = service.run()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()

    print("Identity discovery completed.")

    print()

    print("Speaker turns:")
    print(
        f"  {result['speaker_turn_count']}"
    )

    print()

    print("Detected speakers:")

    for speaker in result[
        "detected_speakers"
    ]:

        print(
            f"  {speaker}"
        )

    print()

    print("Candidates:")
    print(
        f"  {result['candidate_count']}"
    )

    print()

    print(
        "Transcript modified:"
        f" {result['transcript_modified']}"
    )

    print()

    print("Output written:")
    print(
        f"  {output_file}"
    )

    print()

    print("=" * 70)

    print(
        "PHASE 9.9.5C: PASS"
    )

    print("=" * 70)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )