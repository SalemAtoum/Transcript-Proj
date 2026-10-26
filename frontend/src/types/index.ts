export type MeetingStatus = 'completed' | 'processing' | 'failed';
export type JobStatus = 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
export type StageStatus = 'pending' | 'processing' | 'completed' | 'failed';
export interface Meeting { id: string; title: string; date: string; duration: number; speakerCount: number; wordCount: number; status: MeetingStatus; }
export interface Speaker { id: string; name: string; wordCount: number; identified: boolean; confidence?: number; }
export interface TranscriptTurn { id: string; speakerId: string; startTime: number; endTime: number; text: string; }
export interface ProcessingStage { id: string; name: string; status: StageStatus; }
export interface ProcessingJob { id: string; meetingId: string; status: JobStatus; progress: number; stages: ProcessingStage[]; uploadedAt: string; fileName?: string; }
export interface MeetingRecord { meeting: Meeting; speakers: Speaker[]; turns: TranscriptTurn[]; job?: ProcessingJob; }
