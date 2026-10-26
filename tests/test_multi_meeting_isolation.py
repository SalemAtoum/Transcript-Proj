from __future__ import annotations

import json
import shutil
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEETINGS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "meetings"
)

MEETING_A = "isolation_test_A"
MEETING_B = "isolation_test_B"


# ============================================================
# Test Data
# ============================================================

TRANSCRIPT_A = {
    "schema_version": "1.0",
    "phase": "isolation_test",
    "language": "ar",
    "audio_duration": 12.5,
    "statistics": {
        "total_words": 5,
        "speaker_word_counts": {
            "Salameh": 5
        },
    },
    "speaker_mapping": {
        "SPEAKER_00": "Salameh"
    },
    "turns": [
        {
            "turn_id": 0,
            "start": 0.0,
            "end": 12.5,
            "speaker_id": "SPEAKER_00",
            "speaker": "Salameh",
            "text": "هذا نص اختبار الاجتماع الأول",
            "word_count": 5,
        }
    ],
}


TRANSCRIPT_B = {
    "schema_version": "1.0",
    "phase": "isolation_test",
    "language": "ar",
    "audio_duration": 25.0,
    "statistics": {
        "total_words": 6,
        "speaker_word_counts": {
            "Sami": 6
        },
    },
    "speaker_mapping": {
        "SPEAKER_00": "Sami"
    },
    "turns": [
        {
            "turn_id": 0,
            "start": 0.0,
            "end": 25.0,
            "speaker_id": "SPEAKER_00",
            "speaker": "Sami",
            "text": "هذا نص اختبار الاجتماع الثاني",
            "word_count": 6,
        }
    ],
}


MARKDOWN_A = """# Meeting Transcript

## Salameh

**00:00 → 00:12**

هذا نص اختبار الاجتماع الأول
"""


MARKDOWN_B = """# Meeting Transcript

## Sami

**00:00 → 00:25**

هذا نص اختبار الاجتماع الثاني
"""


# ============================================================
# Helpers
# ============================================================

def create_meeting_workspace(
    meeting_id: str,
    transcript_data: dict,
    markdown_content: str,
) -> None:

    meeting_directory = (
        MEETINGS_DIRECTORY
        / meeting_id
    )

    temporary_directory = (
        meeting_directory
        / "temporary"
    )

    output_directory = (
        meeting_directory
        / "output"
    )

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    transcript_file = (
        temporary_directory
        / "final_transcript.json"
    )

    transcript_file.write_text(
        json.dumps(
            transcript_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_file = (
        output_directory
        / "meeting_transcript.md"
    )

    markdown_file.write_text(
        markdown_content,
        encoding="utf-8",
    )


def cleanup() -> None:

    for meeting_id in (
        MEETING_A,
        MEETING_B,
    ):

        meeting_directory = (
            MEETINGS_DIRECTORY
            / meeting_id
        )

        if meeting_directory.exists():
            shutil.rmtree(
                meeting_directory
            )


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:

    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


# ============================================================
# Tests
# ============================================================

def test_transcript_isolation() -> None:

    print("=" * 70)
    print("TEST 1 — TRANSCRIPT ISOLATION")
    print("=" * 70)

    response_a = requests.get(
        f"{BASE_URL}/api/meetings/{MEETING_A}/transcript",
        timeout=10,
    )

    assert_equal(
        response_a.status_code,
        200,
        "Meeting A transcript request failed.",
    )

    data_a = response_a.json()

    assert_equal(
        data_a["meeting_id"],
        MEETING_A,
        "Meeting A returned the wrong meeting ID.",
    )

    assert_equal(
        data_a["duration_seconds"],
        12.5,
        "Meeting A returned the wrong duration.",
    )

    assert_equal(
        data_a["speakers"][0]["name"],
        "Salameh",
        "Meeting A returned the wrong speaker.",
    )

    assert_equal(
        data_a["turns"][0]["text"],
        "هذا نص اختبار الاجتماع الأول",
        "Meeting A returned the wrong transcript.",
    )

    print("Meeting A transcript: PASS")

    response_b = requests.get(
        f"{BASE_URL}/api/meetings/{MEETING_B}/transcript",
        timeout=10,
    )

    assert_equal(
        response_b.status_code,
        200,
        "Meeting B transcript request failed.",
    )

    data_b = response_b.json()

    assert_equal(
        data_b["meeting_id"],
        MEETING_B,
        "Meeting B returned the wrong meeting ID.",
    )

    assert_equal(
        data_b["duration_seconds"],
        25.0,
        "Meeting B returned the wrong duration.",
    )

    assert_equal(
        data_b["speakers"][0]["name"],
        "Sami",
        "Meeting B returned the wrong speaker.",
    )

    assert_equal(
        data_b["turns"][0]["text"],
        "هذا نص اختبار الاجتماع الثاني",
        "Meeting B returned the wrong transcript.",
    )

    print("Meeting B transcript: PASS")

    # Cross-check: A must still be A after reading B.
    response_a_again = requests.get(
        f"{BASE_URL}/api/meetings/{MEETING_A}/transcript",
        timeout=10,
    )

    assert_equal(
        response_a_again.status_code,
        200,
        "Meeting A failed after reading Meeting B.",
    )

    data_a_again = response_a_again.json()

    assert_equal(
        data_a_again["turns"][0]["text"],
        "هذا نص اختبار الاجتماع الأول",
        "Meeting A was overwritten by Meeting B.",
    )

    print("Cross-meeting transcript isolation: PASS")


def test_markdown_isolation() -> None:

    print()
    print("=" * 70)
    print("TEST 2 — MARKDOWN ISOLATION")
    print("=" * 70)

    response_a = requests.get(
        f"{BASE_URL}/api/meetings/{MEETING_A}/transcript/download",
        timeout=10,
    )

    assert_equal(
        response_a.status_code,
        200,
        "Meeting A Markdown download failed.",
    )

    markdown_a = response_a.text

    if "الاجتماع الأول" not in markdown_a:
        raise AssertionError(
            "Meeting A Markdown contains incorrect content."
        )

    if "الاجتماع الثاني" in markdown_a:
        raise AssertionError(
            "Meeting A Markdown contains Meeting B content."
        )

    print("Meeting A Markdown: PASS")

    response_b = requests.get(
        f"{BASE_URL}/api/meetings/{MEETING_B}/transcript/download",
        timeout=10,
    )

    assert_equal(
        response_b.status_code,
        200,
        "Meeting B Markdown download failed.",
    )

    markdown_b = response_b.text

    if "الاجتماع الثاني" not in markdown_b:
        raise AssertionError(
            "Meeting B Markdown contains incorrect content."
        )

    if "الاجتماع الأول" in markdown_b:
        raise AssertionError(
            "Meeting B Markdown contains Meeting A content."
        )

    print("Meeting B Markdown: PASS")

    print("Cross-meeting Markdown isolation: PASS")


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("SWBS MULTI-MEETING ISOLATION TEST")
    print("=" * 70)
    print()

    try:

        # ----------------------------------------------------
        # Create isolated test workspaces
        # ----------------------------------------------------

        cleanup()

        create_meeting_workspace(
            meeting_id=MEETING_A,
            transcript_data=TRANSCRIPT_A,
            markdown_content=MARKDOWN_A,
        )

        create_meeting_workspace(
            meeting_id=MEETING_B,
            transcript_data=TRANSCRIPT_B,
            markdown_content=MARKDOWN_B,
        )

        print("Test workspaces created.")
        print()

        # ----------------------------------------------------
        # Run tests
        # ----------------------------------------------------

        test_transcript_isolation()

        test_markdown_isolation()

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("MULTI-MEETING ISOLATION TEST: PASS")
        print("=" * 70)
        print()
        print("Meeting A and Meeting B remained completely isolated.")
        print("Transcript API isolation: PASS")
        print("Markdown API isolation: PASS")

    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        cleanup()

        print()
        print("Test workspaces cleaned up.")


if __name__ == "__main__":
    main()