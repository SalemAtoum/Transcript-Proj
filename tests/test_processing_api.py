from pathlib import Path
import time

import src.backend.routes.processing as processing_module
from src.backend.services.job_manager import job_manager


def main():
    print("=" * 60)
    print("PROCESSING API WIRING TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create fake meeting directory
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    meeting_id = "meeting_processing_test"

    meeting_directory = (
        project_root
        / "data"
        / "uploads"
        / meeting_id
    )

    meeting_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_input = meeting_directory / "test_meeting.wav"

    if not test_input.exists():
        test_input.write_bytes(b"FAKE_AUDIO_DATA")

    print(f"Meeting ID: {meeting_id}")
    print(f"Input file: {test_input}")

    # ---------------------------------------------------------
    # 2. Replace the REAL AI call temporarily
    # ---------------------------------------------------------

    original_run = processing_module.ai_pipeline_service.run

    def fake_ai_pipeline_run(input_file: Path):
        print()
        print("FAKE AIPIPELINE SERVICE")
        print(f"Received input: {input_file}")

        time.sleep(1)

        print("Fake AI pipeline completed.")

        return "FAKE_AI_PIPELINE_SUCCESS"

    processing_module.ai_pipeline_service.run = fake_ai_pipeline_run

    try:
        # -----------------------------------------------------
        # 3. Call the REAL processing endpoint function
        # -----------------------------------------------------

        response = processing_module.start_processing(
            meeting_id=meeting_id
        )

        print()
        print("PROCESSING RESPONSE")
        print("-" * 60)
        print(f"Job ID: {response.job_id}")
        print(f"Meeting ID: {response.meeting_id}")
        print(f"Status: {response.status}")

        job_id = response.job_id

        # -----------------------------------------------------
        # 4. Wait for background worker
        # -----------------------------------------------------

        timeout_seconds = 10
        start_time = time.time()

        while True:
            job = job_manager.get_job(job_id)

            if job is None:
                raise RuntimeError(
                    "Job was not found in JobManager."
                )

            print(
                f"\rJob status: {job['status']}",
                end="",
                flush=True,
            )

            if job["status"] in {"completed", "failed"}:
                break

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    "Processing API wiring test timed out."
                )

            time.sleep(0.2)

        print()

        # -----------------------------------------------------
        # 5. Validate result
        # -----------------------------------------------------

        final_job = job_manager.get_job(job_id)

        print()
        print("FINAL JOB")
        print("-" * 60)
        print(f"Status: {final_job['status']}")
        print(f"Progress: {final_job['progress']}")
        print(f"Stage: {final_job['stage']}")
        print(f"Error: {final_job['error']}")

        if response.meeting_id != meeting_id:
            raise AssertionError(
                "Returned meeting ID does not match."
            )

        if final_job["status"] != "completed":
            raise AssertionError(
                f"Expected completed job, "
                f"got {final_job['status']}"
            )

        if final_job["progress"] != 100:
            raise AssertionError(
                f"Expected progress 100, "
                f"got {final_job['progress']}"
            )

        if final_job["error"] is not None:
            raise AssertionError(
                f"Unexpected job error: "
                f"{final_job['error']}"
            )

        print()
        print("PROCESSING API WIRING TEST: PASS")

    finally:
        # -----------------------------------------------------
        # 6. Restore the real AI service
        # -----------------------------------------------------

        processing_module.ai_pipeline_service.run = original_run


if __name__ == "__main__":
    main()