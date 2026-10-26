from pathlib import Path
import time

from src.backend.services.job_manager import job_manager
from src.backend.services.background_worker import background_worker


class FakePipelineService:
    """
    Fake pipeline used only for testing the background worker.

    It does NOT run Whisper, PyAnnote, or any real AI model.
    """

    def run(self, input_file: Path):
        print("\nFAKE AI PIPELINE")
        print(f"Input file: {input_file}")

        time.sleep(1)

        print("Fake pipeline completed.")

        return "FAKE_PIPELINE_SUCCESS"


def main():
    print("=" * 60)
    print("BACKGROUND WORKER TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create fake meeting/input file
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    test_input = (
        project_root
        / "data"
        / "uploads"
        / "worker_test.wav"
    )

    test_input.parent.mkdir(parents=True, exist_ok=True)

    if not test_input.exists():
        test_input.write_bytes(b"FAKE_AUDIO_DATA")

    print(f"Test input: {test_input}")

    # ---------------------------------------------------------
    # 2. Create a job
    # ---------------------------------------------------------

    meeting_id = "meeting_worker_test"

    job = job_manager.create_job(
        meeting_id=meeting_id
    )

    job_id = job["job_id"]

    print(f"Created job: {job_id}")

    # ---------------------------------------------------------
    # 3. Create fake pipeline
    # ---------------------------------------------------------

    fake_pipeline = FakePipelineService()

    # ---------------------------------------------------------
    # 4. Define background task
    # ---------------------------------------------------------

    def pipeline_task():
        result = fake_pipeline.run(test_input)

        print(f"Pipeline result: {result}")

        return result

    # ---------------------------------------------------------
    # 5. Submit task to the REAL background worker
    # ---------------------------------------------------------

    background_worker.submit(
        job_id,
        pipeline_task
    )

    print("Worker submitted.")

    # ---------------------------------------------------------
    # 6. Wait for completion
    # ---------------------------------------------------------

    timeout_seconds = 10
    start_time = time.time()

    while True:
        current_job = job_manager.get_job(job_id)

        if current_job is None:
            raise RuntimeError("Job disappeared from JobManager.")

        status_value = current_job["status"]

        print(
            f"\rJob status: {status_value}",
            end="",
            flush=True
        )

        if status_value in {"completed", "failed"}:
            break

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(
                "Background worker test timed out."
            )

        time.sleep(0.2)

    print()

    # ---------------------------------------------------------
    # 7. Validate result
    # ---------------------------------------------------------

    final_job = job_manager.get_job(job_id)

    print("\nWORKER TEST RESULTS")
    print("-" * 60)
    print(f"Status: {final_job['status']}")
    print(f"Progress: {final_job['progress']}")
    print(f"Stage: {final_job['stage']}")
    print(f"Error: {final_job['error']}")

    if final_job["status"] != "completed":
        raise AssertionError(
            f"Expected completed job, got: {final_job['status']}"
        )

    if final_job["progress"] != 100:
        raise AssertionError(
            f"Expected progress 100, got: {final_job['progress']}"
        )

    if final_job["error"] is not None:
        raise AssertionError(
            f"Unexpected job error: {final_job['error']}"
        )

    print("\nBACKGROUND WORKER TEST: PASS")


if __name__ == "__main__":
    main()