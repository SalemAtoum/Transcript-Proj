from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.backend.main import app


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_AUDIO_FILE = (
    PROJECT_ROOT
    / "research"
    / "phase4"
    / "phase4_test_10s.wav"
)

POLL_INTERVAL_SECONDS = 1.0

# Whisper is running on CPU.
# The previous 1-second test took ~56 seconds just for Whisper,
# so 15 seconds is not enough for a real AI pipeline E2E test.
POLL_TIMEOUT_SECONDS = 600.0


client = TestClient(app)


# ============================================================
# HELPERS
# ============================================================

def validate_test_audio() -> None:
    """
    Validate that the real test audio file exists and is non-empty.
    """

    assert TEST_AUDIO_FILE.exists(), (
        f"Test audio file does not exist:\n{TEST_AUDIO_FILE}"
    )

    assert TEST_AUDIO_FILE.is_file(), (
        f"Test audio path is not a file:\n{TEST_AUDIO_FILE}"
    )

    file_size = TEST_AUDIO_FILE.stat().st_size

    assert file_size > 44, (
        f"Test audio file is too small to be a valid WAV file: "
        f"{file_size} bytes"
    )

    print()
    print("Test audio:")
    print(f"  {TEST_AUDIO_FILE}")
    print(f"  Size: {file_size:,} bytes")


def wait_for_job_completion(job_id: str) -> dict:
    """
    Poll the backend until the background processing job reaches
    completed or failed status.

    This function also prints progress changes so we can observe
    the real AI pipeline stages during the E2E test.
    """

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS

    last_status: dict | None = None

    last_progress = None
    last_stage = None

    print()
    print("-" * 70)
    print("WAITING FOR AI PIPELINE")
    print("-" * 70)

    while time.monotonic() < deadline:

        response = client.get(
            f"/api/jobs/{job_id}"
        )

        assert response.status_code == 200, (
            f"Job status API failed: "
            f"{response.status_code} - {response.text}"
        )

        last_status = response.json()

        current_status = last_status.get("status")
        current_progress = last_status.get("progress")
        current_stage = (
            last_status.get("current_stage")
            or last_status.get("stage")
        )

        # Print only when something changes.
        if (
            current_progress != last_progress
            or current_stage != last_stage
        ):
            print(
                f"JOB STATUS | "
                f"status={current_status} | "
                f"stage={current_stage} | "
                f"progress={current_progress}%"
            )

            last_progress = current_progress
            last_stage = current_stage

        # Terminal states.
        if current_status in {"completed", "failed"}:
            print("-" * 70)
            return last_status

        time.sleep(POLL_INTERVAL_SECONDS)

    raise AssertionError(
        "Job did not finish within "
        f"{POLL_TIMEOUT_SECONDS:.0f} seconds.\n"
        f"Last status: {last_status}"
    )


# ============================================================
# END-TO-END TEST
# ============================================================

def test_backend_end_to_end():

    print()
    print("=" * 70)
    print("SWBS BACKEND END-TO-END TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 0. VALIDATE REAL TEST AUDIO
    # --------------------------------------------------------

    validate_test_audio()

    # --------------------------------------------------------
    # 1. HEALTH API
    # --------------------------------------------------------

    response = client.get("/health")

    assert response.status_code == 200, (
        f"Health API failed: "
        f"{response.status_code} - {response.text}"
    )

    health_data = response.json()

    assert health_data.get("status") == "healthy", (
        f"Unexpected health response: {health_data}"
    )

    print()
    print("1. Health API: PASS")

    # --------------------------------------------------------
    # 2. UPLOAD API
    # --------------------------------------------------------

    with TEST_AUDIO_FILE.open("rb") as audio_file:

        response = client.post(
            "/api/meetings/upload",
            files={
                "file": (
                    TEST_AUDIO_FILE.name,
                    audio_file,
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 201, (
        f"Upload API failed: "
        f"{response.status_code} - {response.text}"
    )

    upload_data = response.json()

    meeting_id = upload_data["meeting_id"]

    assert meeting_id.startswith("meeting_"), (
        f"Unexpected meeting ID: {meeting_id}"
    )

    assert upload_data["filename"] == TEST_AUDIO_FILE.name, (
        f"Unexpected filename: {upload_data['filename']}"
    )

    assert upload_data["status"] == "uploaded", (
        f"Unexpected upload status: {upload_data['status']}"
    )

    print(
        f"2. Upload API: PASS ({meeting_id})"
    )

    # --------------------------------------------------------
    # 3. START PROCESSING API
    # --------------------------------------------------------

    response = client.post(
        f"/api/meetings/{meeting_id}/process"
    )

    assert response.status_code == 202, (
        f"Start Processing API failed: "
        f"{response.status_code} - {response.text}"
    )

    processing_data = response.json()

    job_id = processing_data["job_id"]

    assert job_id.startswith("job_"), (
        f"Unexpected job ID: {job_id}"
    )

    assert processing_data["meeting_id"] == meeting_id, (
        "Processing response contains wrong meeting ID."
    )

    assert processing_data["status"] == "queued", (
        f"Unexpected initial job status: "
        f"{processing_data['status']}"
    )

    print(
        f"3. Start Processing API: PASS ({job_id})"
    )

    # --------------------------------------------------------
    # 4. REAL BACKGROUND AI PIPELINE
    # --------------------------------------------------------

    final_job = wait_for_job_completion(job_id)

    assert final_job["status"] == "completed", (
        f"Background AI pipeline failed:\n{final_job}"
    )

    assert final_job["progress"] == 100, (
        f"Expected progress=100, "
        f"got {final_job['progress']}"
    )

    print("4. Background AI Pipeline: PASS")

    # --------------------------------------------------------
    # 5. TRANSCRIPT API
    # --------------------------------------------------------

    response = client.get(
        f"/api/meetings/{meeting_id}/transcript"
    )

    assert response.status_code == 200, (
        f"Transcript API failed: "
        f"{response.status_code} - {response.text}"
    )

    transcript_data = response.json()

    assert transcript_data["meeting_id"] == meeting_id, (
        "Transcript contains the wrong meeting ID."
    )

    assert "language" in transcript_data
    assert "duration_seconds" in transcript_data
    assert "word_count" in transcript_data
    assert "speakers" in transcript_data
    assert "turns" in transcript_data

    assert isinstance(
        transcript_data["speakers"],
        list,
    )

    assert isinstance(
        transcript_data["turns"],
        list,
    )

    print("5. Transcript API: PASS")

    # --------------------------------------------------------
    # 6. BASIC TRANSCRIPT VALIDATION
    # --------------------------------------------------------

    duration = transcript_data["duration_seconds"]
    word_count = transcript_data["word_count"]
    speakers = transcript_data["speakers"]
    turns = transcript_data["turns"]

    assert duration >= 0, (
        f"Invalid transcript duration: {duration}"
    )

    assert word_count >= 0, (
        f"Invalid transcript word count: {word_count}"
    )

    print()
    print("Transcript validation:")
    print(f"  Duration : {duration:.2f} seconds")
    print(f"  Words    : {word_count}")
    print(f"  Speakers : {len(speakers)}")
    print(f"  Turns    : {len(turns)}")

    # --------------------------------------------------------
    # 7. MARKDOWN DOWNLOAD API
    # --------------------------------------------------------

    response = client.get(
        f"/api/meetings/{meeting_id}/transcript/download"
    )

    assert response.status_code == 200, (
        f"Markdown download failed: "
        f"{response.status_code} - {response.text}"
    )

    content_disposition = response.headers.get(
        "content-disposition",
        "",
    )

    assert "meeting_transcript.md" in content_disposition, (
        "Expected meeting_transcript.md in "
        "Content-Disposition header."
    )

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert content_type.startswith("text/markdown"), (
        f"Unexpected content type: {content_type}"
    )

    assert len(response.content) > 0, (
        "Downloaded Markdown file is empty."
    )

    print("6. Markdown Download API: PASS")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BACKEND END-TO-END TEST: PASS")
    print("=" * 70)

    print()
    print(f"Meeting ID : {meeting_id}")
    print(f"Job ID     : {job_id}")
    print(f"Job Status : {final_job['status']}")
    print(f"Progress   : {final_job['progress']}%")

    print()
    print("Validated flow:")
    print("  Health API")
    print("      ↓")
    print("  Upload API")
    print("      ↓")
    print("  Start Processing API")
    print("      ↓")
    print("  Background Worker")
    print("      ↓")
    print("  Audio Preprocessing")
    print("      ↓")
    print("  Whisper Transcription")
    print("      ↓")
    print("  PyAnnote Diarization")
    print("      ↓")
    print("  Speaker Identification")
    print("      ↓")
    print("  Transcript Alignment")
    print("      ↓")
    print("  Final Transcript")
    print("      ↓")
    print("  Markdown Generation")
    print("      ↓")
    print("  Transcript API")
    print("      ↓")
    print("  Markdown Download")

    print()
    print("=" * 70)


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    test_backend_end_to_end()