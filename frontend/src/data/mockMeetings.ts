
import type { Meeting, MeetingRecord, Speaker, TranscriptTurn } from '../types';

export const demoStages = [
  ['audio_preprocessing', 'Audio preprocessing'],
  ['transcription', 'Speech transcription'],
  ['diarization', 'Speaker diarization'],
  ['speaker_identification', 'Speaker identification'],
  ['alignment', 'Transcript alignment'],
  ['final_transcript', 'Final transcript'],
  ['markdown_generation', 'Markdown generation'],
] as const;

const titles = [
  'Project Planning Demo',
  'Product Review Demo',
  'Team Progress Demo',
  'Customer Support Demo',
  'Sprint Planning Demo',
  'Operations Demo',
  'Research Review Demo',
  'Design Feedback Demo',
  'Feature Prioritization Demo',
  'Quality Review Demo',
  'Release Planning Demo',
  'Technical Discussion Demo',
  'Workflow Review Demo',
  'Requirements Demo',
  'Engineering Demo',
  'Testing Review Demo',
  'Prototype Feedback Demo',
  'Project Status Demo',
  'Issue Tracking Demo',
  'Development Planning Demo',
  'Demo Meeting 21',
  'Demo Meeting 22',
  'Demo Meeting 23',
  'Demo Meeting 24',
];

export const seedMeetings: Meeting[] = titles.map((title, i) => ({
  id: `demo-meeting-${i + 1}`,
  title,
  date: new Date(Date.UTC(2026, 9, 7 - i)).toISOString(),
  duration: [2327, 3133, 1868, 2740, 3540, 2215, 2894, 1678][i % 8],
  speakerCount: [4, 6, 3, 5, 7, 4, 5, 3][i % 8],
  wordCount: [1039, 1820, 912, 1374, 2011, 1207, 1662, 845][i % 8],
  status: i === 2 ? 'processing' : i === 13 || i === 20 ? 'failed' : 'completed',
}));

export const sampleSpeakers: Speaker[] = [
  { id: 'speaker-1', name: 'Speaker 1', wordCount: 136, identified: true, confidence: 0.96 },
  { id: 'speaker-2', name: 'Speaker 2', wordCount: 885, identified: true, confidence: 0.99 },
  { id: 'speaker-unknown', name: 'Unknown speaker', wordCount: 18, identified: false },
  { id: 'speaker-3', name: 'Speaker 3', wordCount: 274, identified: true, confidence: 0.93 },
];

export const sampleTurns: TranscriptTurn[] = [
  {
    id: 'turn-1',
    speakerId: 'speaker-1',
    startTime: 0,
    endTime: 12,
    text: 'Welcome to the demo. Today we will review the project status and the next steps.',
  },
  {
    id: 'turn-2',
    speakerId: 'speaker-unknown',
    startTime: 12,
    endTime: 20,
    text: 'Could we also review the testing timeline before we finish?',
  },
  {
    id: 'turn-3',
    speakerId: 'speaker-1',
    startTime: 20,
    endTime: 32,
    text: 'Yes. We will include the testing timeline and confirm the remaining tasks.',
  },
  {
    id: 'turn-4',
    speakerId: 'speaker-2',
    startTime: 32,
    endTime: 48,
    text: 'The prototype is ready for review. The next step is to validate the main workflow.',
  },
  {
    id: 'turn-5',
    speakerId: 'speaker-3',
    startTime: 48,
    endTime: 62,
    text: 'I will review the workflow and share feedback on the user experience.',
  },
  {
    id: 'turn-6',
    speakerId: 'speaker-2',
    startTime: 62,
    endTime: 78,
    text: 'I will document the feedback and prepare the next iteration.',
  },
  {
    id: 'turn-7',
    speakerId: 'speaker-1',
    startTime: 78,
    endTime: 91,
    text: 'Great. Let us record the action items and review progress at the next demo.',
  },
];

export function createRecord(meeting: Meeting): MeetingRecord {
  return {
    meeting,
    speakers: sampleSpeakers,
    turns: sampleTurns,
  };
}
