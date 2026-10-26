import {
  getMeetingTranscript,
  getJobStatus,
  saveMeetingMarkdown,
  startMeetingProcessing,
  uploadMeeting,
  type ProcessingJobResponse,
} from "./backendApi";

import type {
  Meeting,
  MeetingRecord,
  ProcessingJob,
  Speaker,
  TranscriptTurn,
} from "../types";

/* ============================================================
   HELPERS
   ============================================================ */

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "—";
  }

  const totalSeconds = Math.round(seconds);

  const hours = Math.floor(
    totalSeconds / 3600,
  );

  const minutes = Math.floor(
    (totalSeconds % 3600) / 60,
  );

  const remainingSeconds =
    totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(
      2,
      "0",
    )}:${String(remainingSeconds).padStart(
      2,
      "0",
    )}`;
  }

  return `${minutes}:${String(
    remainingSeconds,
  ).padStart(2, "0")}`;
}


/* ============================================================
   JOB STATUS MAPPING
   ============================================================ */

function mapJobStatus(
  status: ProcessingJobResponse["status"],
): ProcessingJob["status"] {
  if (status === "completed") {
    return "completed";
  }

  if (status === "failed") {
    return "failed";
  }

  return "processing";
}


/* ============================================================
   PIPELINE STAGES
   ============================================================ */

function buildStages(
  job: ProcessingJobResponse,
): ProcessingJob["stages"] {
  const stages = [
    {
      id: "audio_preprocessing",
      name: "Prepare audio",
      progress: 10,
    },
    {
      id: "transcription",
      name: "Transcribe speech",
      progress: 30,
    },
    {
      id: "diarization",
      name: "Detect speakers",
      progress: 50,
    },
    {
      id: "speaker_identification",
      name: "Identify speakers",
      progress: 70,
    },
    {
      id: "alignment",
      name: "Align the transcript",
      progress: 85,
    },
    {
      id: "final_transcript",
      name: "Generate final transcript",
      progress: 92,
    },
  ];

  return stages.map(
    (stage, index) => {
      let status:
        | "pending"
        | "processing"
        | "completed" =
        "pending";

      /*
       * Completed job:
       * every frontend stage is displayed
       * as completed.
       */
      if (job.status === "completed") {
        status = "completed";
      }

      /*
       * Current backend stage:
       * show it as processing.
       */
      else if (
        job.current_stage === stage.id
      ) {
        status = "processing";
      }

      /*
       * Previous stages:
       * mark them completed based on
       * the backend progress value.
       */
      else if (
        job.progress >= stage.progress
      ) {
        status = "completed";
      }

      /*
       * If the backend has passed the
       * previous stage but has not yet
       * reached this one, keep it pending.
       */
      else if (
        index === 0 &&
        job.progress > 0
      ) {
        status = "processing";
      }

      return {
        id: stage.id,
        name: stage.name,
        status,
      };
    },
  );
}


/* ============================================================
   MAP BACKEND JOB → FRONTEND JOB
   ============================================================ */

function mapJob(
  job: ProcessingJobResponse,
): ProcessingJob {
return {
  id: job.job_id,
  meetingId: job.meeting_id,
  uploadedAt: job.created_at,
  status: mapJobStatus(job.status),
  progress: job.progress,
  stages: buildStages(job),
  fileName: undefined,
};
}


/* ============================================================
   MAP BACKEND SPEAKER → FRONTEND SPEAKER
   ============================================================ */

function mapSpeaker(
  speaker: {
    id: string;
    name: string;
    identified: boolean;
    confidence?: number | null;
    word_count: number;
  },
): Speaker {
  return {
    id: speaker.id,
    name: speaker.name,
    identified: speaker.identified,
    wordCount: speaker.word_count,
  };
}


/* ============================================================
   MAP BACKEND TURN → FRONTEND TURN
   ============================================================ */

function mapTranscriptTurn(
  turn: {
    speaker: string;
    start_seconds: number;
    end_seconds: number;
    text: string;
  },
  index: number,
): TranscriptTurn {
  return {
    id: `turn-${index}`,
    speakerId: turn.speaker,
    startTime: turn.start_seconds,
    endTime: turn.end_seconds,
    text: turn.text,
  };
}


/* ============================================================
   CREATE MEETING
   ============================================================ */

export async function createMeeting(
  file: File,
  _duration?: number,
): Promise<Meeting> {
  /*
   * STEP 1
   * Upload the actual file to FastAPI.
   */
  const upload = await uploadMeeting(
    file,
  );

  /*
   * STEP 2
   * Tell FastAPI to start the real
   * background AI pipeline.
   */
  const processing =
    await startMeetingProcessing(
      upload.meeting_id,
    );

  /*
   * IMPORTANT
   *
   * The backend has two IDs:
   *
   * meeting_id
   * job_id
   *
   * We keep the job ID in sessionStorage
   * so the Processing page can poll it.
   */
  sessionStorage.setItem(
    `swbs-job-${upload.meeting_id}`,
    processing.job_id,
  );

  /*
   * Keep basic meeting metadata in the
   * current browser session.
   *
   * This is NOT the old mock processing.
   * The actual processing happens in FastAPI.
   */
  const meeting: Meeting = {
    id: upload.meeting_id,
    title: file.name,
    date: new Date().toISOString(),
    duration: 0,
    speakerCount: 0,
    wordCount: 0,
    status: "processing",
  };

  sessionStorage.setItem(
    `swbs-meeting-${upload.meeting_id}`,
    JSON.stringify(meeting),
  );

  return meeting;
}


/* ============================================================
   GET JOB
   ============================================================ */

export async function getJob(
  meetingId: string,
): Promise<ProcessingJob | undefined> {
  try {
    /*
     * Retrieve the real backend job ID
     * associated with this meeting.
     */
    const jobId =
      sessionStorage.getItem(
        `swbs-job-${meetingId}`,
      );

    if (!jobId) {
      return undefined;
    }

    /*
     * Ask FastAPI for the REAL current
     * processing status.
     */
    const job =
      await getJobStatus(jobId);

    return mapJob(job);
  } catch {
    return undefined;
  }
}


/* ============================================================
   GET MEETING TRANSCRIPT
   ============================================================ */

export async function getMeeting(
  meetingId: string,
): Promise<MeetingRecord | undefined> {
  try {
    /*
     * The transcript endpoint only becomes
     * available after processing completes.
     */
    const transcript =
      await getMeetingTranscript(
        meetingId,
      );

    const speakers =
      transcript.speakers.map(
        mapSpeaker,
      );

    const turns =
      transcript.turns.map(
        mapTranscriptTurn,
      );

    /*
     * Try to preserve the original
     * browser-side meeting metadata.
     */
    let storedMeeting:
      | Meeting
      | undefined;

    const stored =
      sessionStorage.getItem(
        `swbs-meeting-${meetingId}`,
      );

    if (stored) {
      try {
        storedMeeting =
          JSON.parse(
            stored,
          ) as Meeting;
      } catch {
        storedMeeting =
          undefined;
      }
    }

    const meeting: Meeting = {
      id: transcript.meeting_id,

      title:
        storedMeeting?.title ??
        transcript.meeting_id,

      date:
        storedMeeting?.date ??
        new Date().toISOString(),

      duration:
        transcript.duration_seconds,

      speakerCount:
        speakers.length,

      wordCount:
        transcript.word_count,

      status: "completed",
    };

    return {
      meeting,
      speakers,
      turns,
    };
  } catch {
    return undefined;
  }
}


/* ============================================================
   GET MEETINGS
   ============================================================ */

export async function getMeetings(): Promise<
  Meeting[]
> {
  /*
   * The current FastAPI backend does not
   * expose a GET /meetings endpoint yet.
   *
   * Therefore, for this integration stage,
   * we use sessionStorage only for the
   * meeting LIST.
   *
   * The actual transcript/job data still
   * comes from FastAPI.
   */

  const meetings: Meeting[] = [];

  for (
    let index = 0;
    index < sessionStorage.length;
    index += 1
  ) {
    const key =
      sessionStorage.key(index);

    if (
      !key ||
      !key.startsWith(
        "swbs-meeting-",
      )
    ) {
      continue;
    }

    const raw =
      sessionStorage.getItem(key);

    if (!raw) {
      continue;
    }

    try {
      const meeting =
        JSON.parse(raw) as Meeting;

      meetings.push(meeting);
    } catch {
      /*
       * Ignore malformed session entries.
       */
    }
  }

  return meetings.sort(
    (a, b) =>
      b.date.localeCompare(
        a.date,
      ),
  );
}


/* ============================================================
   ADVANCE JOB
   ============================================================ */

export async function advanceJob(
  _meetingId: string,
  _step?: number,
): Promise<
  ProcessingJob | undefined
> {
  /*
   * IMPORTANT:
   *
   * This function existed only because
   * the old frontend simulated processing.
   *
   * The real backend now controls processing.
   *
   * We keep the function temporarily so
   * Workspace.tsx still compiles.
   *
   * The next step will remove this simulated
   * polling from ProcessingPage and replace
   * it with direct backend polling.
   */

  return undefined;
}


/* ============================================================
   DOWNLOAD MARKDOWN
   ============================================================ */

export async function downloadMarkdown(
  record: MeetingRecord,
): Promise<void> {
  await saveMeetingMarkdown(
    record.meeting.id,
    `${record.meeting.title || record.meeting.id}.md`,
  );
}