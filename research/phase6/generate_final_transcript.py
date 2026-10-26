import json
import time
from pathlib import Path
from collections import Counter


# ============================================================
# PHASE 6.3 — FINAL SPEAKER-ATTRIBUTED TRANSCRIPT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    SCRIPT_DIR
    / "aligned_results"
    / "aligned_transcript.json"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "final_transcript"
)

OUTPUT_JSON = OUTPUT_DIR / "final_speaker_transcript.json"
OUTPUT_TXT = OUTPUT_DIR / "final_speaker_transcript.txt"


# ============================================================
# HELPERS
# ============================================================

def format_timestamp(seconds):
    """
    Convert seconds to HH:MM:SS.mmm
    """
    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def clean_text(text):
    """
    Minimal text cleanup.

    IMPORTANT:
    We do NOT rewrite, correct, normalize, or paraphrase
    Whisper's transcription.
    """
    if not text:
        return ""

    return " ".join(text.split())


def calculate_turn_confidence(words):
    """
    Calculate confidence statistics for a speaker turn.
    """

    if not words:
        return {
            "word_count": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "none_confidence": 0,
            "confidence_ratio": 0.0,
        }

    confidence_counts = Counter(
        word.get("alignment_confidence", "none")
        for word in words
    )

    total = len(words)

    high = confidence_counts.get("high", 0)
    medium = confidence_counts.get("medium", 0)
    low = confidence_counts.get("low", 0)
    none = confidence_counts.get("none", 0)

    confidence_ratio = high / total if total > 0 else 0.0

    return {
        "word_count": total,
        "high_confidence": high,
        "medium_confidence": medium,
        "low_confidence": low,
        "none_confidence": none,
        "confidence_ratio": round(confidence_ratio, 4),
    }


def build_turn_text(words):
    """
    Build readable text from the aligned words.

    We preserve the actual Whisper words.
    """

    texts = []

    for word in words:
        text = word.get("word", "").strip()

        if text:
            texts.append(text)

    return clean_text(" ".join(texts))


# ============================================================
# LOAD INPUT
# ============================================================

def load_alignment():
    print()
    print("[1] Loading aligned transcript...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Input file: {INPUT_FILE}")

    return data


# ============================================================
# BUILD FINAL TURNS
# ============================================================

def build_final_turns(data):

    print()
    print("[2] Building final speaker turns...")

    speaker_turns = data.get("speaker_turns", [])

    if not speaker_turns:
        raise ValueError(
            "No speaker_turns found in aligned transcript."
        )

    final_turns = []

    for index, turn in enumerate(speaker_turns):

        words = turn.get("words", [])

        speaker_name = turn.get(
            "speaker_name",
            "UNKNOWN"
        )

        speaker_id = turn.get(
            "speaker"
        )

        start = float(
            turn.get(
                "start",
                words[0].get("start", 0.0) if words else 0.0
            )
        )

        end = float(
            turn.get(
                "end",
                words[-1].get("end", start) if words else start
            )
        )

        text = build_turn_text(words)

        confidence_stats = calculate_turn_confidence(
            words
        )

        final_turn = {
            "turn_id": index,

            "speaker": speaker_id,

            "speaker_name": speaker_name,

            "start": start,

            "end": end,

            "start_time": format_timestamp(start),

            "end_time": format_timestamp(end),

            "duration_seconds": round(
                max(0.0, end - start),
                3
            ),

            "text": text,

            "confidence": confidence_stats,

            "words": words,
        }

        final_turns.append(final_turn)

    return final_turns


# ============================================================
# VALIDATION
# ============================================================

def validate_turns(final_turns):

    print()
    print("[3] Validating final speaker turns...")

    problems = []

    previous_start = -1.0
    previous_end = -1.0

    for turn in final_turns:

        start = turn["start"]
        end = turn["end"]

        if end < start:
            problems.append(
                f"Invalid interval in turn {turn['turn_id']}"
            )

        if start < previous_start:
            problems.append(
                f"Non-monotonic start time in turn {turn['turn_id']}"
            )

        if end < previous_end:
            problems.append(
                f"Non-monotonic end time in turn {turn['turn_id']}"
            )

        if not turn["text"]:
            problems.append(
                f"Empty text in turn {turn['turn_id']}"
            )

        previous_start = start
        previous_end = end

    if problems:
        print()
        print("WARNING: Validation problems found:")

        for problem in problems[:20]:
            print(f"  - {problem}")

        if len(problems) > 20:
            print(
                f"  ... and {len(problems) - 20} more"
            )

        return False

    print("Validation: PASS")

    return True


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(final_turns):

    speaker_turn_counts = Counter()
    speaker_word_counts = Counter()

    total_words = 0
    total_turns = len(final_turns)

    confidence_counts = Counter()

    for turn in final_turns:

        speaker = turn["speaker_name"]

        speaker_turn_counts[speaker] += 1

        word_count = turn["confidence"]["word_count"]

        speaker_word_counts[speaker] += word_count

        total_words += word_count

        confidence_counts["high"] += (
            turn["confidence"]["high_confidence"]
        )

        confidence_counts["medium"] += (
            turn["confidence"]["medium_confidence"]
        )

        confidence_counts["low"] += (
            turn["confidence"]["low_confidence"]
        )

        confidence_counts["none"] += (
            turn["confidence"]["none_confidence"]
        )

    return {
        "total_turns": total_turns,

        "total_words": total_words,

        "speaker_turn_counts": dict(
            speaker_turn_counts
        ),

        "speaker_word_counts": dict(
            speaker_word_counts
        ),

        "alignment_confidence_counts": dict(
            confidence_counts
        ),
    }


# ============================================================
# JSON OUTPUT
# ============================================================

def save_json(
    input_data,
    final_turns,
    statistics,
):

    print()
    print("[4] Saving final JSON...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_output = {
        "phase": "6.3",

        "component": (
            "final_speaker_attributed_transcript"
        ),

        "source_phase": "6.2",

        "audio": input_data.get("audio"),

        "speaker_mapping": input_data.get(
            "speaker_mapping",
            {}
        ),

        "input_statistics": {
            "transcript_segments": input_data.get(
                "input_statistics",
                {}
            ).get(
                "transcript_segments"
            ),

            "transcript_words": input_data.get(
                "input_statistics",
                {}
            ).get(
                "transcript_words"
            ),

            "diarization_segments": input_data.get(
                "input_statistics",
                {}
            ).get(
                "diarization_segments"
            ),

            "detected_speakers": input_data.get(
                "input_statistics",
                {}
            ).get(
                "detected_speakers",
                []
            ),
        },

        "alignment_statistics": input_data.get(
            "alignment_statistics",
            {}
        ),

        "final_statistics": statistics,

        "notes": [
            (
                "Transcript words are preserved from "
                "the Phase 6.2 aligned transcript."
            ),
            (
                "No linguistic correction, "
                "paraphrasing, or ASR rewriting "
                "was performed."
            ),
            (
                "Speaker attribution is based on "
                "Phase 5.4 speaker identification "
                "and Phase 6.2 temporal alignment."
            ),
            (
                "UNKNOWN turns are preserved when "
                "speaker attribution is unavailable."
            ),
        ],

        "speaker_turns": final_turns,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {OUTPUT_JSON}")


# ============================================================
# TXT OUTPUT
# ============================================================

def save_txt(
    input_data,
    final_turns,
    statistics,
):

    print()
    print("[5] Saving human-readable TXT...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SWBS MEETING — FINAL SPEAKER-ATTRIBUTED TRANSCRIPT\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            "Speaker Mapping\n"
        )

        f.write("-" * 70 + "\n")

        mapping = input_data.get(
            "speaker_mapping",
            {}
        )

        for speaker_id, name in mapping.items():
            f.write(
                f"{speaker_id} -> {name}\n"
            )

        f.write("\n")

        f.write(
            f"Total Turns: {statistics['total_turns']}\n"
        )

        f.write(
            f"Total Words: {statistics['total_words']}\n"
        )

        f.write("\n")

        f.write(
            "Speaker Statistics\n"
        )

        f.write("-" * 70 + "\n")

        for speaker, count in (
            statistics[
                "speaker_word_counts"
            ].items()
        ):

            turns = statistics[
                "speaker_turn_counts"
            ].get(
                speaker,
                0
            )

            f.write(
                f"{speaker}: "
                f"{count} words / "
                f"{turns} turns\n"
            )

        f.write("\n")

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "TRANSCRIPT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        for turn in final_turns:

            speaker = turn[
                "speaker_name"
            ]

            start_time = turn[
                "start_time"
            ]

            end_time = turn[
                "end_time"
            ]

            text = turn[
                "text"
            ]

            f.write(
                f"[{start_time} - {end_time}]\n"
            )

            f.write(
                f"{speaker}:\n"
            )

            f.write(
                f"{text}\n\n"
            )

    print(f"Saved: {OUTPUT_TXT}")


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print(
        "PHASE 6.3 — FINAL SPEAKER-ATTRIBUTED TRANSCRIPT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    input_data = load_alignment()

    # --------------------------------------------------------
    # 2. Build
    # --------------------------------------------------------

    final_turns = build_final_turns(
        input_data
    )

    # --------------------------------------------------------
    # 3. Validate
    # --------------------------------------------------------

    validation_pass = validate_turns(
        final_turns
    )

    # --------------------------------------------------------
    # 4. Statistics
    # --------------------------------------------------------

    statistics = calculate_statistics(
        final_turns
    )

    # --------------------------------------------------------
    # 5. Save JSON
    # --------------------------------------------------------

    save_json(
        input_data,
        final_turns,
        statistics
    )

    # --------------------------------------------------------
    # 6. Save TXT
    # --------------------------------------------------------

    save_txt(
        input_data,
        final_turns,
        statistics
    )

    # --------------------------------------------------------
    # 7. Final report
    # --------------------------------------------------------

    processing_time = (
        time.time() - start_time
    )

    print()
    print("=" * 70)
    print("PHASE 6.3 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total turns: {statistics['total_turns']}"
    )

    print(
        f"Total words: {statistics['total_words']}"
    )

    print()
    print("Speaker statistics:")

    for speaker, count in (
        statistics[
            "speaker_word_counts"
        ].items()
    ):

        turns = statistics[
            "speaker_turn_counts"
        ].get(
            speaker,
            0
        )

        print(
            f"  {speaker}: "
            f"{count} words / "
            f"{turns} turns"
        )

    print()
    print("Alignment confidence:")

    confidence = statistics[
        "alignment_confidence_counts"
    ]

    for level in [
        "high",
        "medium",
        "low",
        "none",
    ]:

        print(
            f"  {level}: "
            f"{confidence.get(level, 0)}"
        )

    print()
    print(
        f"Validation: "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )

    print(
        f"Processing time: "
        f"{processing_time:.2f} sec"
    )

    print()
    print("Outputs:")

    print(
        f"  JSON: {OUTPUT_JSON}"
    )

    print(
        f"  TXT:  {OUTPUT_TXT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()