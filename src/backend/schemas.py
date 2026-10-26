from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================
# MEETING STATUS
# ============================================================

class MeetingStatus(str, Enum):

    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# ============================================================
# JOB STATUS
# ============================================================

class JobStatus(str, Enum):

    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# ============================================================
# PROCESSING STAGE
# ============================================================

class ProcessingStage(str, Enum):

    audio_preprocessing = "audio_preprocessing"
    transcription = "transcription"
    diarization = "diarization"
    speaker_identification = "speaker_identification"
    alignment = "alignment"
    final_transcript = "final_transcript"
    markdown_generation = "markdown_generation"


# ============================================================
# MEETING RESPONSE
# ============================================================

class MeetingResponse(BaseModel):

    id: str

    title: str

    filename: str

    status: MeetingStatus

    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
    )

    speaker_count: int = Field(
        default=0,
        ge=0,
    )

    word_count: int = Field(
        default=0,
        ge=0,
    )

    created_at: datetime


# ============================================================
# MEETING LIST RESPONSE
# ============================================================

class MeetingListResponse(BaseModel):

    meetings: list[MeetingResponse]

    total: int = Field(
        default=0,
        ge=0,
    )


# ============================================================
# PROCESSING JOB RESPONSE
# ============================================================

class ProcessingJobResponse(BaseModel):

    job_id: str

    meeting_id: str

    status: JobStatus

    progress: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    current_stage: ProcessingStage | None = None

    created_at: datetime

    completed_at: datetime | None = None

    error: str | None = None


# ============================================================
# SPEAKER
# ============================================================

class SpeakerResponse(BaseModel):

    id: str

    name: str

    identified: bool

    confidence: str | None = None

    word_count: int = Field(
        default=0,
        ge=0,
    )


# ============================================================
# TRANSCRIPT TURN
# ============================================================

class TranscriptTurn(BaseModel):

    speaker: str

    start_seconds: float = Field(
        ge=0.0,
    )

    end_seconds: float = Field(
        ge=0.0,
    )

    text: str


# ============================================================
# TRANSCRIPT RESPONSE
# ============================================================

class TranscriptResponse(BaseModel):

    meeting_id: str

    language: str

    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
    )

    word_count: int = Field(
        default=0,
        ge=0,
    )

    speakers: list[SpeakerResponse]

    turns: list[TranscriptTurn]


# ============================================================
# UPLOAD RESPONSE
# ============================================================

class UploadMeetingResponse(BaseModel):

    meeting_id: str

    filename: str

    status: MeetingStatus


# ============================================================
# START PROCESSING RESPONSE
# ============================================================

class StartProcessingResponse(BaseModel):

    job_id: str

    meeting_id: str

    status: JobStatus