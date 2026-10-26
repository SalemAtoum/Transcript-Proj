import json
import math
import time
from pathlib import Path
from collections import Counter


# ============================================================
# PHASE 6.4 — FINAL VALIDATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE6_DIR = SCRIPT_DIR.parent

INPUT_FILE = (
    PHASE6_DIR
    / "final_transcript"
    / "final_speaker_transcript.json"
)

OUTPUT_FILE = (
    SCRIPT_DIR
    / "phase6_final_validation_report.json"
)


EXPECTED_MAPPING = {
    "SPEAKER_00": "Sami",
    "SPEAKER_01": "Salameh",
}

EXPECTED_SPEAKERS = {
    "SPEAKER_00",
    "SPEAKER_01",
}

EXPECTED_NAMES = {
    "Sami",
    "Salameh",
    "UNKNOWN",
}


# ============================================================
# HELPERS
# ============================================================

def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def add_check(checks, name, passed, details):
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("PHASE 6.4 — FINAL VALIDATION")
    print("=" * 70)

    checks = []
    warnings = []

    # --------------------------------------------------------
    # 1. Load file
    # --------------------------------------------------------

    print("\n[1] Loading final transcript...")

    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print("\nERROR: Input file does not exist.")
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("JSON loaded successfully.")

    # --------------------------------------------------------
    # 2. Top-level structure
    # --------------------------------------------------------

    print("\n[2] Validating top-level structure...")

    required_top_level = [
        "phase",
        "component",
        "source_phase",
        "audio",
        "speaker_mapping",
        "input_statistics",
        "alignment_statistics",
        "final_statistics",
        "notes",
        "speaker_turns",
    ]

    missing_fields = [
        field
        for field in required_top_level
        if field not in data
    ]

    add_check(
        checks,
        "Top-level JSON structure",
        len(missing_fields) == 0,
        (
            "All required fields are present."
            if not missing_fields
            else f"Missing fields: {missing_fields}"
        ),
    )

    # --------------------------------------------------------
    # 3. Phase metadata
    # --------------------------------------------------------

    print("\n[3] Validating phase metadata...")

    phase_ok = data.get("phase") == "6.3"
    component_ok = (
        data.get("component")
        == "final_speaker_attributed_transcript"
    )
    source_ok = data.get("source_phase") == "6.2"

    add_check(
        checks,
        "Phase metadata",
        phase_ok and component_ok and source_ok,
        (
            f"phase={data.get('phase')}, "
            f"component={data.get('component')}, "
            f"source_phase={data.get('source_phase')}"
        ),
    )

    # --------------------------------------------------------
    # 4. Speaker mapping
    # --------------------------------------------------------

    print("\n[4] Validating speaker mapping...")

    speaker_mapping = data.get("speaker_mapping", {})

    mapping_ok = speaker_mapping == EXPECTED_MAPPING

    add_check(
        checks,
        "Speaker mapping",
        mapping_ok,
        f"Mapping: {speaker_mapping}",
    )

    if not mapping_ok:
        warnings.append(
            "Speaker mapping differs from the expected Phase 5.4 mapping."
        )

    # --------------------------------------------------------
    # 5. Speaker turns
    # --------------------------------------------------------

    print("\n[5] Inspecting speaker turns...")

    turns = data.get("speaker_turns", [])

    turns_ok = isinstance(turns, list) and len(turns) > 0

    add_check(
        checks,
        "Speaker turns exist",
        turns_ok,
        f"Total turns: {len(turns)}",
    )

    # --------------------------------------------------------
    # 6. Turn IDs
    # --------------------------------------------------------

    print("\n[6] Validating turn IDs...")

    turn_ids = []

    for turn in turns:
        turn_ids.append(turn.get("turn_id"))

    expected_turn_ids = list(range(len(turns)))

    turn_ids_ok = turn_ids == expected_turn_ids

    add_check(
        checks,
        "Turn IDs",
        turn_ids_ok,
        (
            "Turn IDs are sequential."
            if turn_ids_ok
            else "Turn IDs are missing, duplicated, or out of order."
        ),
    )

    # --------------------------------------------------------
    # 7. Turn timestamps
    # --------------------------------------------------------

    print("\n[7] Validating turn timestamps...")

    turn_timestamp_errors = []
    previous_end = -1.0

    for index, turn in enumerate(turns):

        start = turn.get("start")
        end = turn.get("end")

        if not is_finite_number(start):
            turn_timestamp_errors.append(
                f"Turn {index}: invalid start={start}"
            )
            continue

        if not is_finite_number(end):
            turn_timestamp_errors.append(
                f"Turn {index}: invalid end={end}"
            )
            continue

        if end < start:
            turn_timestamp_errors.append(
                f"Turn {index}: end < start"
            )

        if start < previous_end - 1e-6:
            turn_timestamp_errors.append(
                f"Turn {index}: timestamp order violation"
            )

        previous_end = end

    add_check(
        checks,
        "Turn timestamps",
        len(turn_timestamp_errors) == 0,
        (
            "All turn timestamps are valid and chronological."
            if not turn_timestamp_errors
            else f"Errors: {turn_timestamp_errors[:10]}"
        ),
    )

    # --------------------------------------------------------
    # 8. Turn speaker validity
    # --------------------------------------------------------

    print("\n[8] Validating turn speakers...")

    invalid_turn_speakers = []

    for index, turn in enumerate(turns):

        speaker = turn.get("speaker")
        speaker_name = turn.get("speaker_name")

        if speaker_name not in EXPECTED_NAMES:
            invalid_turn_speakers.append(
                f"Turn {index}: invalid speaker_name={speaker_name}"
            )

        if speaker not in (None, "UNKNOWN"):
            expected_name = speaker_mapping.get(speaker)

            if expected_name != speaker_name:
                invalid_turn_speakers.append(
                    f"Turn {index}: {speaker} -> {speaker_name}"
                )

    add_check(
        checks,
        "Turn speaker consistency",
        len(invalid_turn_speakers) == 0,
        (
            "All turn speakers match the speaker mapping."
            if not invalid_turn_speakers
            else f"Errors: {invalid_turn_speakers[:10]}"
        ),
    )

    # --------------------------------------------------------
    # 9. Word-level validation
    # --------------------------------------------------------

    print("\n[9] Validating words...")

    all_words = []

    for turn in turns:
        words = turn.get("words", [])

        if not isinstance(words, list):
            warnings.append(
                f"Turn {turn.get('turn_id')} has invalid words field."
            )
            continue

        all_words.extend(words)

    total_words = len(all_words)

    print(f"Total words found: {total_words}")

    word_id_errors = []
    word_timestamp_errors = []
    word_speaker_errors = []

    previous_word_id = None
    previous_word_end = -1.0

    for index, word in enumerate(all_words):

        word_id = word.get("word_id")

        start = word.get("start")
        end = word.get("end")

        speaker = word.get("speaker")

        # Word ID
        if previous_word_id is not None:
            if word_id != previous_word_id + 1:
                word_id_errors.append(
                    f"Expected {previous_word_id + 1}, got {word_id}"
                )

        previous_word_id = word_id

        # Timestamp
        if not is_finite_number(start):
            word_timestamp_errors.append(
                f"Word {word_id}: invalid start"
            )
            continue

        if not is_finite_number(end):
            word_timestamp_errors.append(
                f"Word {word_id}: invalid end"
            )
            continue

        if end < start:
            word_timestamp_errors.append(
                f"Word {word_id}: end < start"
            )

        if start < previous_word_end - 1e-6:
            word_timestamp_errors.append(
                f"Word {word_id}: chronological violation"
            )

        previous_word_end = max(previous_word_end, end)

        # Speaker
        if speaker not in (None, "UNKNOWN"):
            if speaker not in EXPECTED_SPEAKERS:
                word_speaker_errors.append(
            f"Word {word_id}: invalid speaker={speaker}"
        )

    add_check(
        checks,
        "Word IDs",
        len(word_id_errors) == 0,
        (
            "All word IDs are sequential."
            if not word_id_errors
            else f"Errors: {word_id_errors[:10]}"
        ),
    )

    add_check(
        checks,
        "Word timestamps",
        len(word_timestamp_errors) == 0,
        (
            "All word timestamps are valid."
            if not word_timestamp_errors
            else f"Errors: {word_timestamp_errors[:10]}"
        ),
    )

    add_check(
        checks,
        "Word speakers",
        len(word_speaker_errors) == 0,
        (
            "All word speaker labels are valid."
            if not word_speaker_errors
            else f"Errors: {word_speaker_errors[:10]}"
        ),
    )

    # --------------------------------------------------------
    # 10. Word count consistency
    # --------------------------------------------------------

    print("\n[10] Validating word-count consistency...")

    reported_total_words = (
        data.get("final_statistics", {})
        .get("total_words")
    )

    word_count_ok = reported_total_words == total_words

    add_check(
        checks,
        "Word count consistency",
        word_count_ok,
        (
            f"Reported={reported_total_words}, "
            f"actual={total_words}"
        ),
    )

    # --------------------------------------------------------
    # 11. Speaker word counts
    # --------------------------------------------------------

    print("\n[11] Validating speaker word counts...")

    actual_speaker_word_counts = Counter()

    for word in all_words:
        speaker = word.get("speaker")

        if speaker == "SPEAKER_00":
            actual_speaker_word_counts["Sami"] += 1

        elif speaker == "SPEAKER_01":
            actual_speaker_word_counts["Salameh"] += 1

        else:
            actual_speaker_word_counts["UNKNOWN"] += 1

    reported_speaker_word_counts = (
        data.get("final_statistics", {})
        .get("speaker_word_counts", {})
    )

    speaker_word_counts_ok = (
        dict(actual_speaker_word_counts)
        == reported_speaker_word_counts
    )

    add_check(
        checks,
        "Speaker word counts",
        speaker_word_counts_ok,
        (
            f"Actual={dict(actual_speaker_word_counts)}, "
            f"Reported={reported_speaker_word_counts}"
        ),
    )

    # --------------------------------------------------------
    # 12. Confidence validation
    # --------------------------------------------------------

    print("\n[12] Validating confidence labels...")

    actual_confidence_counts = Counter()

    for word in all_words:

        confidence = word.get("alignment_confidence")

        if confidence is None:
            confidence = "none"

        actual_confidence_counts[confidence] += 1

    reported_confidence_counts = (
        data.get("final_statistics", {})
        .get("alignment_confidence_counts", {})
    )

    confidence_ok = (
        dict(actual_confidence_counts)
        == reported_confidence_counts
    )

    add_check(
        checks,
        "Confidence counts",
        confidence_ok,
        (
            f"Actual={dict(actual_confidence_counts)}, "
            f"Reported={reported_confidence_counts}"
        ),
    )

    # --------------------------------------------------------
    # 13. UNKNOWN validation
    # --------------------------------------------------------

    print("\n[13] Validating UNKNOWN handling...")

    unknown_words = [
    word
    for word in all_words
    if word.get("speaker") in (None, "UNKNOWN")
]

    unknown_count = len(unknown_words)

    reported_unknown = (
        data.get("final_statistics", {})
        .get("speaker_word_counts", {})
        .get("UNKNOWN", 0)
    )

    unknown_ok = unknown_count == reported_unknown

    add_check(
        checks,
        "UNKNOWN handling",
        unknown_ok,
        (
            f"UNKNOWN words: {unknown_count}, "
            f"reported: {reported_unknown}"
        ),
    )

    if unknown_count > 0:
        warnings.append(
            f"{unknown_count} words remain UNKNOWN because speaker attribution "
            "was unavailable."
        )

    # --------------------------------------------------------
    # 14. Empty text validation
    # --------------------------------------------------------

    print("\n[14] Checking empty turns...")

    empty_turns = []

    for turn in turns:

        text = turn.get("text", "")

        if not isinstance(text, str) or not text.strip():
            empty_turns.append(
                turn.get("turn_id")
            )

    add_check(
        checks,
        "Non-empty speaker turns",
        len(empty_turns) == 0,
        (
            "All speaker turns contain text."
            if not empty_turns
            else f"Empty turns: {empty_turns}"
        ),
    )

    # --------------------------------------------------------
    # 15. Duplicate turn validation
    # --------------------------------------------------------

    print("\n[15] Checking duplicate turns...")

    duplicate_turns = []

    previous_signature = None

    for turn in turns:

        signature = (
            turn.get("speaker_name"),
            turn.get("start"),
            turn.get("end"),
            turn.get("text"),
        )

        if signature == previous_signature:
            duplicate_turns.append(
                turn.get("turn_id")
            )

        previous_signature = signature

    add_check(
        checks,
        "Duplicate consecutive turns",
        len(duplicate_turns) == 0,
        (
            "No consecutive duplicate turns found."
            if not duplicate_turns
            else f"Duplicate turn IDs: {duplicate_turns}"
        ),
    )

    # --------------------------------------------------------
    # 16. Confidence ratio validation
    # --------------------------------------------------------

    print("\n[16] Validating confidence ratios...")

    confidence_ratio_errors = []

    for turn in turns:

        confidence = turn.get("confidence", {})

        word_count = confidence.get("word_count", 0)
        high = confidence.get("high_confidence", 0)
        medium = confidence.get("medium_confidence", 0)
        low = confidence.get("low_confidence", 0)
        none = confidence.get("none_confidence", 0)

        total_confidence_words = (
            high + medium + low + none
        )

        if total_confidence_words != word_count:
            confidence_ratio_errors.append(
                f"Turn {turn.get('turn_id')}: "
                f"word_count={word_count}, "
                f"confidence_sum={total_confidence_words}"
            )

    add_check(
        checks,
        "Turn confidence consistency",
        len(confidence_ratio_errors) == 0,
        (
            "Confidence counts match turn word counts."
            if not confidence_ratio_errors
            else f"Errors: {confidence_ratio_errors[:10]}"
        ),
    )

    # --------------------------------------------------------
    # 17. Coverage
    # --------------------------------------------------------

    print("\n[17] Calculating assignment coverage...")

    assigned_words = total_words - unknown_count

    coverage = (
        assigned_words / total_words
        if total_words > 0
        else 0.0
    )

    print(
        f"Speaker-assigned coverage: "
        f"{coverage * 100:.2f}%"
    )

    coverage_ok = coverage >= 0.99

    add_check(
        checks,
        "Speaker assignment coverage",
        coverage_ok,
        f"Coverage: {coverage * 100:.2f}%",
    )

    # --------------------------------------------------------
    # 18. Alignment method distribution
    # --------------------------------------------------------

    print("\n[18] Validating alignment methods...")

    method_counts = Counter()

    for word in all_words:

        method = word.get("alignment_method")

        if method:
            method_counts[method] += 1

    valid_methods = {
        "overlap",
        "point_inside",
        "nearest",
        "unassigned",
    }

    invalid_methods = {
        method
        for method in method_counts
        if method not in valid_methods
    }

    add_check(
        checks,
        "Alignment method validity",
        len(invalid_methods) == 0,
        (
            f"Methods: {dict(method_counts)}"
            if not invalid_methods
            else f"Invalid methods: {invalid_methods}"
        ),
    )

    # --------------------------------------------------------
    # 19. Turn word totals
    # --------------------------------------------------------

    print("\n[19] Validating turn/word relationship...")

    turn_word_sum = sum(
        len(turn.get("words", []))
        for turn in turns
    )

    turn_word_ok = turn_word_sum == total_words

    add_check(
        checks,
        "Turn-word relationship",
        turn_word_ok,
        (
            f"Turn word sum={turn_word_sum}, "
            f"total words={total_words}"
        ),
    )

    # --------------------------------------------------------
    # 20. Final verdict
    # --------------------------------------------------------

    failed_checks = [
        check
        for check in checks
        if check["status"] == "FAIL"
    ]

    passed_checks = [
        check
        for check in checks
        if check["status"] == "PASS"
    ]

    validation_pass = len(failed_checks) == 0

    processing_time = time.time() - start_time

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = {
        "phase": "6.4",
        "component": "final_validation",

        "input_file": str(INPUT_FILE),

        "validation_summary": {
            "status": "PASS" if validation_pass else "FAIL",
            "total_checks": len(checks),
            "passed_checks": len(passed_checks),
            "failed_checks": len(failed_checks),
            "warning_count": len(warnings),
        },

        "core_statistics": {
            "total_turns": len(turns),
            "total_words": total_words,
            "assigned_words": assigned_words,
            "unknown_words": unknown_count,
            "speaker_assignment_coverage": round(
                coverage,
                6
            ),
        },

        "speaker_mapping": speaker_mapping,

        "speaker_word_counts": dict(
            actual_speaker_word_counts
        ),

        "confidence_counts": dict(
            actual_confidence_counts
        ),

        "alignment_method_counts": dict(
            method_counts
        ),

        "checks": checks,

        "warnings": warnings,

        "failed_checks": failed_checks,

        "processing_time_seconds": round(
            processing_time,
            3
        ),

        "final_verdict": (
            "PASS"
            if validation_pass
            else "FAIL"
        ),

        "notes": [
            "This phase validates the final Phase 6.3 speaker-attributed transcript.",
            "No transcription, diarization, or speaker identification was rerun.",
            "Validation checks structural and attribution integrity only.",
            "Validation does not imply perfect ASR word accuracy.",
            "UNKNOWN words are preserved rather than force-assigned.",
        ],
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    print("\n[20] Saving validation report...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 6.4 — VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\nChecks passed : {len(passed_checks)}"
    )

    print(
        f"Checks failed : {len(failed_checks)}"
    )

    print(
        f"Warnings      : {len(warnings)}"
    )

    print(
        f"Turns         : {len(turns)}"
    )

    print(
        f"Words         : {total_words}"
    )

    print(
        f"Assigned      : {assigned_words}"
    )

    print(
        f"UNKNOWN       : {unknown_count}"
    )

    print(
        f"Coverage      : {coverage * 100:.2f}%"
    )

    print("\nSpeaker word counts:")

    for speaker, count in actual_speaker_word_counts.items():
        print(
            f"  {speaker}: {count}"
        )

    print("\nConfidence:")

    for confidence, count in actual_confidence_counts.items():
        print(
            f"  {confidence}: {count}"
        )

    print("\nAlignment methods:")

    for method, count in method_counts.items():
        print(
            f"  {method}: {count}"
        )

    print("\nFinal validation:")

    if validation_pass:
        print("  PASS")
    else:
        print("  FAIL")

    if warnings:
        print("\nWarnings:")

        for warning in warnings:
            print(
                f"  - {warning}"
            )

    print("\nReport:")
    print(OUTPUT_FILE)

    print(
        f"\nProcessing time: "
        f"{processing_time:.2f} sec"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()