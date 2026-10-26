// src/services/backendApi.ts

// ============================================================
// SWBS REAL BACKEND API CLIENT
// ============================================================

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

// ============================================================
// TYPES
// ============================================================

export type MeetingStatus =
  | "uploaded"
  | "queued"
  | "processing"
  | "completed"
  | "failed";

export type JobStatus =
  | "queued"
  | "processing"
  | "completed"
  | "failed";

export interface UploadMeetingResponse {
  meeting_id: string;
  filename: string;
  status: MeetingStatus;
}

export interface StartProcessingResponse {
  job_id: string;
  meeting_id: string;
  status: JobStatus;
}

export interface ProcessingJobResponse {
  job_id: string;
  meeting_id: string;
  status: JobStatus;
  progress: number;
  current_stage?: string | null;
  created_at: string;
  completed_at?: string | null;
  error?: string | null;
}

export interface SpeakerResponse {
  id: string;
  name: string;
  identified: boolean;
  confidence?: number | null;
  word_count: number;
}

export interface TranscriptTurn {
  speaker: string;
  start_seconds: number;
  end_seconds: number;
  text: string;
}

export interface TranscriptResponse {
  meeting_id: string;
  language: string;
  duration_seconds: number;
  word_count: number;
  speakers: SpeakerResponse[];
  turns: TranscriptTurn[];
}

// ============================================================
// ERROR HANDLING
// ============================================================

export class BackendApiError extends Error {
  status: number;
  detail?: string;

  constructor(
    message: string,
    status: number,
    detail?: string,
  ) {
    super(message);

    this.name = "BackendApiError";
    this.status = status;
    this.detail = detail;
  }
}

// ============================================================
// RESPONSE HELPER
// ============================================================

async function parseErrorResponse(
  response: Response,
): Promise<never> {
  let detail = "";

  try {
    const data = await response.json();

    if (typeof data?.detail === "string") {
      detail = data.detail;
    } else if (data?.detail) {
      detail = JSON.stringify(data.detail);
    }
  } catch {
    // Response may not contain JSON.
  }

  throw new BackendApiError(
    detail ||
      `Backend request failed with status ${response.status}.`,
    response.status,
    detail || undefined,
  );
}

async function ensureOk(
  response: Response,
): Promise<Response> {
  if (!response.ok) {
    await parseErrorResponse(response);
  }

  return response;
}

// ============================================================
// HEALTH
// ============================================================

export async function checkBackendHealth(): Promise<{
  status: string;
  service: string;
  version: string;
}> {
  const response = await fetch(
    `${API_BASE_URL}/health`,
    {
      method: "GET",
    },
  );

  await ensureOk(response);

  return response.json();
}

// ============================================================
// UPLOAD MEETING
// ============================================================

export async function uploadMeeting(
  file: File,
): Promise<UploadMeetingResponse> {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/api/meetings/upload`,
    {
      method: "POST",
      body: formData,
    },
  );

  await ensureOk(response);

  return response.json();
}

// ============================================================
// START PROCESSING
// ============================================================

export async function startMeetingProcessing(
  meetingId: string,
): Promise<StartProcessingResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/meetings/${encodeURIComponent(
      meetingId,
    )}/process`,
    {
      method: "POST",
    },
  );

  await ensureOk(response);

  return response.json();
}

// ============================================================
// GET JOB STATUS
// ============================================================

export async function getJobStatus(
  jobId: string,
): Promise<ProcessingJobResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/jobs/${encodeURIComponent(
      jobId,
    )}`,
    {
      method: "GET",
    },
  );

  await ensureOk(response);

  return response.json();
}

// ============================================================
// GET TRANSCRIPT
// ============================================================

export async function getMeetingTranscript(
  meetingId: string,
): Promise<TranscriptResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/meetings/${encodeURIComponent(
      meetingId,
    )}/transcript`,
    {
      method: "GET",
    },
  );

  await ensureOk(response);

  return response.json();
}

// ============================================================
// DOWNLOAD MARKDOWN
// ============================================================

export async function downloadMeetingMarkdown(
  meetingId: string,
): Promise<Blob> {
  const response = await fetch(
    `${API_BASE_URL}/api/meetings/${encodeURIComponent(
      meetingId,
    )}/transcript/download`,
    {
      method: "GET",
    },
  );

  await ensureOk(response);

  return response.blob();
}

// ============================================================
// DOWNLOAD HELPER
// ============================================================

export async function saveMeetingMarkdown(
  meetingId: string,
  filename = "meeting_transcript.md",
): Promise<void> {
  const blob =
    await downloadMeetingMarkdown(meetingId);

  const url = URL.createObjectURL(blob);

  const anchor =
    document.createElement("a");

  anchor.href = url;
  anchor.download = filename;

  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  URL.revokeObjectURL(url);
}

// ============================================================
// POLLING HELPER
// ============================================================

export async function waitForJobCompletion(
  jobId: string,
  options?: {
    intervalMs?: number;
    timeoutMs?: number;
    onProgress?: (
      job: ProcessingJobResponse,
    ) => void;
  },
): Promise<ProcessingJobResponse> {
  const intervalMs =
    options?.intervalMs ?? 1500;

  const timeoutMs =
    options?.timeoutMs ?? 30 * 60 * 1000;

  const startedAt = Date.now();

  while (true) {
    const job =
      await getJobStatus(jobId);

    options?.onProgress?.(job);

    if (job.status === "completed") {
      return job;
    }

    if (job.status === "failed") {
      throw new BackendApiError(
        job.error ||
          "Meeting processing failed.",
        500,
        job.error || undefined,
      );
    }

    if (
      Date.now() - startedAt >
      timeoutMs
    ) {
      throw new BackendApiError(
        "Meeting processing timed out.",
        408,
      );
    }

    await new Promise<void>(
      (resolve) =>
        setTimeout(resolve, intervalMs),
    );
  }
}

// ============================================================
// API BASE URL
// ============================================================

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}