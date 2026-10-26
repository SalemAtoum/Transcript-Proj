from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]


# ------------------------------------------------------------
# Default standalone-test paths
# ------------------------------------------------------------

DEFAULT_INPUT_RESULT = (
    PROJECT_ROOT
    / "data"
    / "temporary"
    / "phase9_final_transcript.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "output"
)

DEFAULT_OUTPUT_MARKDOWN = (
    DEFAULT_OUTPUT_DIRECTORY
    / "meeting_transcript.md"
)


def resolve_path(
    env_name: str,
    default_path: Path,
) -> Path:
    """
    Resolve a path from an environment variable.

    If the environment variable is not set,
    use the standalone-test default path.
    """

    value = os.getenv(env_name)

    if value:

        path = Path(value)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path

    return default_path


INPUT_RESULT = resolve_path(
    "SWBS_INPUT_FILE",
    DEFAULT_INPUT_RESULT,
)

OUTPUT_MARKDOWN = resolve_path(
    "SWBS_OUTPUT_FILE",
    DEFAULT_OUTPUT_MARKDOWN,
)

OUTPUT_DIRECTORY = OUTPUT_MARKDOWN.parent


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load UTF-8 JSON."""

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def format_duration(
    seconds: float | None,
) -> str:
    """Convert seconds to HH:MM:SS."""

    if seconds is None:
        return "Unknown"

    seconds = max(
        0,
        int(
            round(
                float(seconds)
            )
        ),
    )

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = seconds % 60

    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def escape_markdown_text(
    text: str,
) -> str:
    """
    Minimal Markdown escaping.

    We intentionally do NOT alter Arabic text,
    punctuation, or wording.
    """

    return (
        text.replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )


# ============================================================
# BUILD MARKDOWN
# ============================================================

def build_markdown(
    data: dict[str, Any],
) -> str:
    """Build the final human-readable Markdown transcript."""

    duration = format_duration(
        data.get("audio_duration")
    )

    language = data.get(
        "language",
        "ar",
    )

    statistics = data.get(
        "statistics",
        {},
    )

    total_turns = statistics.get(
        "total_turns",
        0,
    )

    total_words = statistics.get(
        "total_words",
        0,
    )

    speaker_mapping = data.get(
        "speaker_mapping",
        {},
    )

    turns = data.get(
        "turns",
        [],
    )

    lines: list[str] = []

    # ========================================================
    # HEADER
    # ========================================================

    lines.append(
        "# Meeting Transcript"
    )

    lines.append("")

    lines.append(
        f"**Duration:** `{duration}`"
    )

    lines.append(
        f"**Language:** `{language}`"
    )

    lines.append(
        f"**Speaker Turns:** `{total_turns}`"
    )

    lines.append(
        f"**Total Words:** `{total_words}`"
    )

    lines.append("")

    lines.append(
        "---"
    )

    lines.append("")

    # ========================================================
    # SPEAKER MAPPING
    # ========================================================

    if speaker_mapping:

        lines.append(
            "## Speaker Identification"
        )

        lines.append("")

        for speaker_id, identity in (
            speaker_mapping.items()
        ):

            lines.append(
                f"- `{speaker_id}` → **{identity}**"
            )

        lines.append("")

        lines.append(
            "---"
        )

        lines.append("")

    # ========================================================
    # TRANSCRIPT
    # ========================================================

    lines.append(
        "## Transcript"
    )

    lines.append("")

    for index, turn in enumerate(
        turns
    ):

        speaker = str(
            turn.get(
                "speaker",
                "UNKNOWN",
            )
        )

        start_time = str(
            turn.get(
                "start_time",
                "00:00:00.000",
            )
        )

        end_time = str(
            turn.get(
                "end_time",
                "00:00:00.000",
            )
        )

        text = escape_markdown_text(
            str(
                turn.get(
                    "text",
                    "",
                )
            )
        )

        if not text:
            continue

        lines.append(
            f"### {speaker}"
        )

        lines.append("")

        lines.append(
            f"**{start_time} → {end_time}**"
        )

        lines.append("")

        lines.append(text)

        lines.append("")

        # Separator between turns,
        # but not after the final turn.

        if index < len(turns) - 1:

            lines.append(
                "---"
            )

            lines.append("")

    # ========================================================
    # FOOTER
    # ========================================================

    lines.append(
        "---"
    )

    lines.append("")

    lines.append(
        "*Generated by SWBS Meeting Transcription System.*"
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# VALIDATION
# ============================================================

def validate_markdown(
    markdown: str,
    data: dict[str, Any],
) -> dict[str, Any]:

    errors: list[str] = []
    warnings: list[str] = []

    turns = data.get(
        "turns",
        [],
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if not markdown.strip():

        errors.append(
            "Generated Markdown is empty."
        )

    if "# Meeting Transcript" not in markdown:

        errors.append(
            "Missing transcript title."
        )

    if "## Transcript" not in markdown:

        errors.append(
            "Missing Transcript section."
        )

    # --------------------------------------------------------
    # Check every speaker turn
    # --------------------------------------------------------

    missing_turns = 0

    for turn in turns:

        speaker = str(
            turn.get(
                "speaker",
                "UNKNOWN",
            )
        )

        text = str(
            turn.get(
                "text",
                "",
            )
        ).strip()

        if speaker not in markdown:

            missing_turns += 1

            continue

        if (
            text
            and text not in markdown
        ):

            missing_turns += 1

    if missing_turns > 0:

        warnings.append(
            f"{missing_turns} turn(s) "
            "could not be located exactly "
            "in the generated Markdown."
        )

    # --------------------------------------------------------
    # Count transcript headings
    # --------------------------------------------------------

    heading_count = markdown.count(
        "### "
    )

    if heading_count != len(turns):

        warnings.append(
            f"Expected {len(turns)} speaker "
            f"turn headings but found "
            f"{heading_count}."
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "turns": len(turns),
        "markdown_bytes": len(
            markdown.encode("utf-8")
        ),
        "speaker_turn_headings": (
            heading_count
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    pipeline_mode = any(
        os.getenv(name)
        for name in (
            "SWBS_INPUT_FILE",
            "SWBS_OUTPUT_FILE",
        )
    )

    print("=" * 70)

    print(
        "PHASE 9.8 — MARKDOWN OUTPUT GENERATOR"
    )

    print("=" * 70)

    print()

    print(
        "Mode: "
        f"{'PIPELINE' if pipeline_mode else 'STANDALONE TEST'}"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    print()
    print("Input:")

    print(
        f"  {INPUT_RESULT}"
    )

    print()
    print(
        "Loading final transcript..."
    )

    data = load_json(
        INPUT_RESULT
    )

    print(
        "Final transcript loaded."
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print(
        "Generating Markdown..."
    )

    markdown = build_markdown(
        data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MARKDOWN.write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        "Markdown generated."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation = validate_markdown(
        markdown,
        data,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "MARKDOWN OUTPUT RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Transcript turns: "
        f"{validation['turns']}"
    )

    print(
        f"Markdown size: "
        f"{validation['markdown_bytes']:,} bytes"
    )

    print(
        f"Speaker headings: "
        f"{validation['speaker_turn_headings']}"
    )

    print()
    print(
        "Validation:"
    )

    if validation["passed"]:

        print(
            "  STATUS: PASS"
        )

    else:

        print(
            "  STATUS: FAIL"
        )

        for error in validation[
            "errors"
        ]:

            print(
                f"  ERROR: {error}"
            )

    if validation["warnings"]:

        print()

        for warning in validation[
            "warnings"
        ]:

            print(
                f"  WARNING: {warning}"
            )

    print()
    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_MARKDOWN}"
    )

    print()
    print(
        "=" * 70
    )

    return (
        0
        if validation["passed"]
        else 1
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    sys.exit(main())