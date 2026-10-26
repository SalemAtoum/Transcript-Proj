# Transcript-Proj

## AI-Powered Meeting Transcription, Speaker Diarization, and Transcript Generation

Transcript-Proj is a Python-based meeting-transcription prototype
developed as part of a technical training experience. It combines
speech-to-text transcription, speaker diarization,
speaker-identification research, transcript alignment, and automated
transcript generation in one workflow.

> **Status:** Local-development prototype. It is not production-ready
> without additional security, privacy, reliability, and deployment
> work.

## Project goals

The goal is to turn an audio/video meeting recording into a structured
transcript with timestamps and speaker turns, then make the result
available through an API and a web workspace.

The intended workflow is:

1.  Upload an audio/video recording.
2.  Prepare audio for downstream processing.
3.  Generate speech-to-text output.
4.  Detect speaker turns and their time intervals.
5.  Explore mapping diarization labels to known speakers through
    enrollment and identification.
6.  Align transcript text, timing, and speaker information.
7.  Generate structured transcript data and a Markdown transcript.
8.  Expose processing status, transcript retrieval, and download
    endpoints.

Results depend on recording quality, background noise, overlapping
speech, dialect, model behavior, and the quality of any
speaker-enrollment material.

## How it works

``` text
Audio/video
   |
   v
FastAPI upload endpoint
   |
   v
Meeting-specific upload folder
   |
   v
Background job + AI pipeline service
   |
   v
Orchestrator
   +--> Audio preprocessing
   +--> Whisper speech recognition
   +--> pyannote-based speaker diarization
   +--> Speaker-identification/enrollment workflow
   +--> Transcript and speaker alignment
   +--> Final transcript generation
   +--> Markdown output generation
   |
   v
Job-status API + transcript API + Markdown download
   |
   v
React frontend workspace
```

The backend runs the orchestrator as a Python subprocess and reads
progress messages in the format `SWBS_PROGRESS|stage_name|progress`. The
current job manager stores jobs in memory and the background worker uses
one worker thread; this is appropriate for a local prototype but does
not provide durable jobs across restarts.

## Technology stack

  -----------------------------------------------------------------------
  Area                                Technologies
  ----------------------------------- -----------------------------------
  Backend                             Python, FastAPI, Uvicorn, Pydantic

  Speech and ML                       PyTorch, torchaudio, OpenAI
                                      Whisper, pyannote.audio,
                                      SpeechBrain

  Audio tooling                       soundfile, FFmpeg in the local
                                      environment

  Frontend                            React, TypeScript, Vite

  Frontend package manager            pnpm

  Output formats                      JSON transcript data, Markdown

  Tests                               pytest-based tests and
                                      project-specific validation scripts

  Configuration                       Environment variables,
                                      python-dotenv
  -----------------------------------------------------------------------

The exact compatible versions depend on the local environment. Check
`requirements.txt`, `frontend/package.json`, and
`frontend/pnpm-lock.yaml`.

## Development and research journey

The repository keeps the research scripts and summaries to document
iterative development. The phase descriptions below are based on the
names and files present in the repository; exact experiment findings and
numerical results should be taken from the phase summaries and
experiment outputs.

### Phase 2 --- Audio preprocessing

**Files:** `research/phase2/preprocess_audio.py`,
`src/core/audio/preprocessor.py`

This stage prepares input audio for downstream recognition and speaker
analysis. Keeping preprocessing separate makes it easier to investigate
how the input format and recording characteristics affect later stages.

### Phase 3 --- Transcription experiments and WER rounds

**Files:** `research/phase3/PHASE3_SUMMARY.md`,
`phase3_round8_evaluate.py`, `phase3_round9a_test.py` through
`phase3_round9g_test.py`, `phase3_round10a_test.py`,
`phase3_round10b.py`, `phase3_round11_wer.py`, `phase3_round12_wer.py`

This phase contains multiple experiment rounds and scripts focused on
transcription testing and Word Error Rate (WER). WER compares a
generated transcript with a reference transcript using substitutions,
deletions, and insertions; lower WER generally means closer word-level
agreement.

Multiple rounds help investigate behavior under different test
conditions rather than relying on one run. For a fair comparison, the
reference text, test audio, normalization, and evaluation rules should
remain consistent. This README does not claim a particular WER or
accuracy figure because the supporting values and their test conditions
should be verified from the actual experiment outputs and
`PHASE3_SUMMARY.md`.

### Phase 4 --- Speaker diarization baseline

**Files:** `research/phase4/PHASE4_SUMMARY.md`,
`research/phase4/diarization_baseline.py`,
`src/core/diarization/pyannote_engine.py`

This stage focuses on estimating who spoke when. Diarization labels such
as `SPEAKER_00` identify detected speaker clusters; they do not
automatically establish a real-world identity. Diarization quality must
be considered separately from transcription quality.

### Phase 5 --- Speaker identification and enrollment

**Files:** `research/phase5/PHASE5_SUMMARY.md`,
`automatic_speaker_identification.py`, `build_voice_profiles.py`,
`extract_embeddings.py`, `validate_enrollment.py`, and modules under
`src/core/speaker_id/`

This phase explores connecting diarized speaker labels to known speakers
through enrollment, embeddings, identity discovery, provisional
enrollment, and confirmation workflows. Identification is distinct from
diarization and may be affected by short samples, noise, overlapping
speech, and recording-channel differences. A similarity match should not
be treated as certain identity without appropriate validation.

Voice profiles and embeddings may be sensitive biometric-related data.
They should only be created or shared with appropriate permission and
should never be committed to a public repository.

### Phase 6 --- Alignment and transcript finalization

**Files:** `research/phase6/align_transcript_with_speakers.py`,
`generate_canonical_transcript.py`, `generate_final_transcript.py`,
`research/phase6/validation/phase6_final_validation.py`

This phase focuses on aligning text with speaker and timing information,
generating canonical/final transcript representations, and validating
the generated output. Successful script execution alone does not prove
that every word, timestamp, or speaker assignment is correct.

### Phase 7 --- Canonical output

**File:** `research/phase7/generate_canonical_output.py`

This stage focuses on producing a consistent canonical output. Core
output modules also handle final transcript representation and Markdown
generation for reading, download, and API integration.

### Application integration

The FastAPI backend connects upload, processing jobs, status reporting,
transcript retrieval, and Markdown download. The React frontend provides
a workspace and API client services. Synthetic mock-meeting data is
included for frontend demonstration.

## Evaluation and testing

The research folder contains experiment scripts, phase summaries, and
validation code. The repository also includes:

-   `tests/test_backend_e2e.py`
-   `tests/test_background_worker.py`
-   `tests/test_multi_meeting_isolation.py`
-   `tests/test_processing_api.py`

Other identity-discovery and phase-validation scripts are kept near the
relevant research code.

WER is only meaningful when the reference transcript, dataset, text
normalization, and evaluation procedure are documented. Scores from
different rounds should not be compared unless these conditions are
comparable. Report metrics with their test setup and scope; avoid
presenting one accuracy figure without supporting evidence.

The frontend build and TypeScript type-check were completed successfully
during project preparation. This verifies frontend
compilation/type-checking in that environment, not full end-to-end
operation of every audio-processing dependency.

Run the Python test suite from the project root after installing
dependencies:

``` powershell
python -m pytest
```

Some research scripts may require local test audio, model access,
environment variables, or other artifacts that are intentionally not
included in the repository.

## Repository structure

``` text
Transcript-Proj/
├── frontend/                    # React + TypeScript UI
│   └── src/
│       ├── components/          # UI components
│       ├── data/                # Synthetic demo meetings
│       ├── pages/               # Workspace pages
│       └── services/             # Backend API clients
├── research/
│   ├── phase2/                  # Audio preprocessing
│   ├── phase3/                  # Transcription and WER rounds
│   ├── phase4/                  # Diarization baseline
│   ├── phase5/                  # Speaker identification/enrollment
│   ├── phase6/                  # Alignment and validation
│   └── phase7/                  # Canonical output
├── scripts/                     # Environment checks
├── src/
│   ├── backend/                 # FastAPI routes and services
│   └── core/                    # Audio, transcription, diarization, alignment, output
├── tests/                       # API, worker, isolation, integration tests
├── .env.example
├── .gitignore
└── requirements.txt
```

## Local setup

### Prerequisites

-   Python compatible with the dependencies in `requirements.txt`.
-   Node.js and pnpm for the frontend.
-   FFmpeg installed and available on PATH where required by the audio
    stack.
-   Access to model resources required by the installed AI components.
-   A Hugging Face token may be required for gated models, depending on
    configuration. Keep real credentials in a local `.env` file and
    never commit them.

### 1. Create a virtual environment

From the project root in Windows PowerShell:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch, torchaudio, pyannote.audio, and TorchCodec compatibility can
depend on exact versions and platform configuration. Resolve
compatibility based on the actual error rather than blindly upgrading
every package.

### 2. Configure environment variables

Review `.env.example`. If a model token is required, create a local
`.env` and enter the value privately. Do not print tokens in logs, paste
them into issues, or commit them.

### 3. Install frontend dependencies

``` powershell
cd frontend
pnpm install --frozen-lockfile
```

### 4. Start the backend

From the project root:

``` powershell
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Local endpoints:

-   `http://127.0.0.1:8000/`
-   `http://127.0.0.1:8000/docs`
-   `http://127.0.0.1:8000/health`

### 5. Start the frontend

Open a second terminal and run this from the repository's `frontend`
folder:

``` powershell
pnpm dev
```

Use the local URL printed by Vite. The API client defaults to
`http://127.0.0.1:8000`; set `VITE_API_BASE_URL` if your backend runs
elsewhere.

## Current limitations and next steps

Before production deployment, address at least the following:

1.  **Authentication and authorization:** protect uploads, job status,
    transcript retrieval, and downloads so users can only access their
    own meetings.
2.  **Input validation:** validate media content rather than relying
    only on filename extensions; set request-size limits at the
    server/proxy layer.
3.  **Error handling:** avoid exposing internal paths and raw exception
    details to API clients.
4.  **Durable jobs:** replace in-memory job storage if jobs must survive
    restarts or support multiple backend processes.
5.  **Resource management:** define queue limits, processing timeouts,
    cancellation, and cleanup of incomplete uploads.
6.  **Reproducible evaluation:** document datasets, reference
    transcripts, normalization, per-round metrics, and experimental
    conclusions.
7.  **Speaker identification:** validate identity assignments and avoid
    overstating confidence.
8.  **Deployment:** configure production CORS, logging, monitoring,
    model access, and resource limits before exposing the service
    publicly.
9.  **End-to-end testing:** test the complete upload-to-transcript flow
    in the target environment.
10. **Dependency documentation:** document exact tested versions and
    required model resources.

## Privacy and responsible use

Meeting audio, transcripts, speaker identities, and voice embeddings may
contain confidential or personal information. Obtain appropriate
permission before processing or sharing recordings. Do not publish
private audio, real meeting transcripts, voice profiles, embeddings,
credentials, or other sensitive artifacts.

The public repository should contain code, documentation, tests, and
synthetic demonstration data only. Keep private data and local model
artifacts excluded through `.gitignore`, and inspect staged files before
each public push.

## Summary

Transcript-Proj demonstrates an iterative engineering workflow for
speaker-aware meeting transcription: preparing audio, testing
transcription, implementing diarization, researching speaker
identification, aligning text and speaker turns, generating final
outputs, and integrating the pipeline into a backend/frontend
application. The included research scripts and tests document the
development process; performance claims should be supported by
reproducible experiment results.
