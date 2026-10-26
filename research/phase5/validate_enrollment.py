from pathlib import Path
import json
import math

import numpy as np
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SPEAKER_DIRS = {
    "sami": SCRIPT_DIR / "sami",
    "salameh": SCRIPT_DIR / "salameh",
}

OUTPUT_JSON = SCRIPT_DIR / "enrollment_validation.json"

SUPPORTED_EXTENSIONS = {".wav"}

# Recommended ranges
MIN_DURATION_SEC = 15.0
RECOMMENDED_MIN_DURATION_SEC = 30.0
RECOMMENDED_MAX_DURATION_SEC = 120.0

# Audio quality thresholds
VERY_LOW_RMS_DB = -40.0
LOW_RMS_DB = -30.0

# Small frames used to estimate silence
FRAME_SIZE = 1600  # 100 ms at 16 kHz


# ============================================================
# HELPERS
# ============================================================

def calculate_rms_db(audio: np.ndarray) -> float:
    """
    Calculate RMS level in dBFS.
    """
    audio = audio.astype(np.float64)

    rms = np.sqrt(np.mean(audio ** 2))

    if rms <= 1e-12:
        return -100.0

    return 20.0 * math.log10(rms)


def calculate_silence_ratio(
    audio: np.ndarray,
    threshold_db: float = -45.0,
) -> float:
    """
    Estimate the percentage of near-silent frames.
    """
    if len(audio) == 0:
        return 1.0

    threshold_linear = 10 ** (threshold_db / 20.0)

    total_frames = 0
    silent_frames = 0

    for start in range(0, len(audio), FRAME_SIZE):
        frame = audio[start:start + FRAME_SIZE]

        if len(frame) == 0:
            continue

        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))

        total_frames += 1

        if rms < threshold_linear:
            silent_frames += 1

    if total_frames == 0:
        return 1.0

    return silent_frames / total_frames


def calculate_clipping_ratio(audio: np.ndarray) -> float:
    """
    Percentage of samples very close to digital clipping.
    """
    if len(audio) == 0:
        return 0.0

    clipped = np.abs(audio) >= 0.999

    return float(np.mean(clipped))


def validate_file(
    file_path: Path,
    speaker_name: str,
) -> dict:

    result = {
        "file": file_path.name,
        "speaker": speaker_name,
        "path": str(file_path),
        "status": "PASS",
        "warnings": [],
        "errors": [],
    }

    # --------------------------------------------------------
    # Read audio
    # --------------------------------------------------------

    try:
        audio, sample_rate = sf.read(
            file_path,
            dtype="float32",
            always_2d=False,
        )
    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(
            f"Could not read audio: {type(e).__name__}: {e}"
        )
        return result

    # --------------------------------------------------------
    # Detect channels
    # --------------------------------------------------------

    if audio.ndim == 1:
        channels = 1
        mono_audio = audio
    else:
        channels = audio.shape[1]

        # Convert to mono only for analysis.
        mono_audio = np.mean(audio, axis=1)

    duration_sec = len(mono_audio) / sample_rate

    rms_db = calculate_rms_db(mono_audio)

    silence_ratio = calculate_silence_ratio(mono_audio)

    clipping_ratio = calculate_clipping_ratio(mono_audio)

    # --------------------------------------------------------
    # Store metadata
    # --------------------------------------------------------

    result["sample_rate_hz"] = int(sample_rate)
    result["channels"] = int(channels)
    result["duration_seconds"] = round(duration_sec, 3)
    result["rms_dbfs"] = round(rms_db, 2)
    result["silence_ratio"] = round(silence_ratio, 4)
    result["silence_percentage"] = round(silence_ratio * 100, 2)
    result["clipping_ratio"] = round(clipping_ratio, 6)
    result["clipping_percentage"] = round(
        clipping_ratio * 100,
        4,
    )

    # --------------------------------------------------------
    # Duration checks
    # --------------------------------------------------------

    if duration_sec < MIN_DURATION_SEC:
        result["status"] = "FAIL"

        result["errors"].append(
            f"Recording is too short: {duration_sec:.1f}s "
            f"(minimum {MIN_DURATION_SEC:.0f}s)."
        )

    elif duration_sec < RECOMMENDED_MIN_DURATION_SEC:

        result["warnings"].append(
            f"Recording is shorter than recommended: "
            f"{duration_sec:.1f}s. "
            f"Recommended: {RECOMMENDED_MIN_DURATION_SEC:.0f}+ seconds."
        )

    if duration_sec > RECOMMENDED_MAX_DURATION_SEC:

        result["warnings"].append(
            f"Recording is longer than necessary: "
            f"{duration_sec:.1f}s. "
            f"This is allowed, but around "
            f"{RECOMMENDED_MIN_DURATION_SEC:.0f}-"
            f"{RECOMMENDED_MAX_DURATION_SEC:.0f}s "
            f"is usually sufficient."
        )

    # --------------------------------------------------------
    # Sample rate
    # --------------------------------------------------------

    if sample_rate != 16000:

        result["warnings"].append(
            f"Sample rate is {sample_rate} Hz. "
            f"Phase 5 preprocessing will standardize it to 16 kHz."
        )

    # --------------------------------------------------------
    # Channels
    # --------------------------------------------------------

    if channels != 1:

        result["warnings"].append(
            f"Audio has {channels} channels. "
            f"Phase 5 preprocessing will convert it to mono."
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if rms_db <= VERY_LOW_RMS_DB:

        result["status"] = "FAIL"

        result["errors"].append(
            f"Audio level is extremely low: {rms_db:.1f} dBFS."
        )

    elif rms_db <= LOW_RMS_DB:

        result["warnings"].append(
            f"Audio level is relatively low: {rms_db:.1f} dBFS."
        )

    # --------------------------------------------------------
    # Silence
    # --------------------------------------------------------

    if silence_ratio > 0.50:

        result["warnings"].append(
            f"More than 50% of the recording appears silent."
        )

    elif silence_ratio > 0.30:

        result["warnings"].append(
            f"High silence ratio: "
            f"{silence_ratio * 100:.1f}%."
        )

    # --------------------------------------------------------
    # Clipping
    # --------------------------------------------------------

    if clipping_ratio > 0.01:

        result["warnings"].append(
            f"Possible clipping detected in "
            f"{clipping_ratio * 100:.2f}% of samples."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if result["errors"]:
        result["status"] = "FAIL"

    elif result["warnings"]:
        result["status"] = "PASS_WITH_WARNINGS"

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 5.1 — SPEAKER ENROLLMENT VALIDATION")
    print("=" * 70)

    all_results = []

    for speaker_name, speaker_dir in SPEAKER_DIRS.items():

        print()
        print(f"[{speaker_name.upper()}]")

        if not speaker_dir.exists():

            print(f"ERROR: Folder not found: {speaker_dir}")

            all_results.append({
                "speaker": speaker_name,
                "status": "FAIL",
                "errors": [
                    f"Folder does not exist: {speaker_dir}"
                ],
                "warnings": [],
            })

            continue

        files = sorted(
            file
            for file in speaker_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:

            print("ERROR: No WAV files found.")

            all_results.append({
                "speaker": speaker_name,
                "status": "FAIL",
                "errors": [
                    "No WAV enrollment files found."
                ],
                "warnings": [],
            })

            continue

        for file_path in files:

            result = validate_file(
                file_path,
                speaker_name,
            )

            all_results.append(result)

            status = result["status"]

            print()
            print(f"File: {file_path.name}")
            print(f"Status: {status}")

            if "duration_seconds" in result:
                print(
                    f"Duration: "
                    f"{result['duration_seconds']:.2f} sec"
                )

                print(
                    f"Sample rate: "
                    f"{result['sample_rate_hz']} Hz"
                )

                print(
                    f"Channels: "
                    f"{result['channels']}"
                )

                print(
                    f"RMS: "
                    f"{result['rms_dbfs']:.2f} dBFS"
                )

                print(
                    f"Silence: "
                    f"{result['silence_percentage']:.2f}%"
                )

                print(
                    f"Clipping: "
                    f"{result['clipping_percentage']:.4f}%"
                )

            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}")

            for error in result.get("errors", []):
                print(f"ERROR: {error}")

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(all_results)

    passed = sum(
        1
        for r in all_results
        if r.get("status") in {
            "PASS",
            "PASS_WITH_WARNINGS",
        }
    )

    failed = sum(
        1
        for r in all_results
        if r.get("status") == "FAIL"
    )

    warnings = sum(
        1
        for r in all_results
        if r.get("status") == "PASS_WITH_WARNINGS"
    )

    speaker_counts = {}

    for result in all_results:

        speaker = result.get("speaker", "unknown")

        speaker_counts.setdefault(
            speaker,
            0,
        )

        speaker_counts[speaker] += 1

    final_status = (
        "PASS"
        if failed == 0 and total > 0
        else "FAIL"
    )

    output = {
        "phase": "5.1",
        "name": "Speaker Enrollment Validation",
        "status": final_status,
        "summary": {
            "total_files": total,
            "passed_files": passed,
            "failed_files": failed,
            "files_with_warnings": warnings,
            "speaker_counts": speaker_counts,
        },
        "files": all_results,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 5.1 SUMMARY")
    print("=" * 70)

    print(f"Total files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"With warnings: {warnings}")

    print()
    print(f"Final status: {final_status}")

    print()
    print(f"Report saved to:")
    print(OUTPUT_JSON)

    print("=" * 70)


if __name__ == "__main__":
    main()