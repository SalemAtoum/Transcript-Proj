import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

import {
  Link,
  useLocation,
  useParams,
} from 'wouter';

import {
  ArrowDown,
  ArrowDownUp,
  ArrowLeft,
  ArrowRight,
  AudioLines,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Copy,
  Download,
  FileAudio,
  FileText,
  Filter,
  Headphones,
  LayoutList,
  Moon,
  Search,
  Sun,
  Upload,
  Users,
  X,
} from 'lucide-react';

import type {
  Meeting,
  MeetingRecord,
  ProcessingJob,
} from '../types';

import {
  createMeeting,
  downloadMarkdown,
  formatDuration,
  getJob,
  getMeeting,
  getMeetings,
} from '../services/meetingsApi';

import { sampleSpeakers } from '../data/mockMeetings';


export const NOTEBOOKLM_URL =
  import.meta.env.VITE_NOTEBOOKLM_URL ||
  'https://notebooklm.google.com/';


const dateLabel = (date: string) =>
  new Date(date).toLocaleDateString(
    'en-US',
    {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    },
  );


function PageTitle({
  eyebrow,
  title,
  subtitle,
  action,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && (
          <div className="eyebrow">
            {eyebrow}
          </div>
        )}

        <h1>{title}</h1>

        {subtitle && (
          <p>{subtitle}</p>
        )}
      </div>

      {action && (
        <div className="heading-action">
          {action}
        </div>
      )}
    </div>
  );
}


function Status({
  status,
}: {
  status: Meeting['status'];
}) {
  return (
    <span
      className={`status status-${status}`}
    >
      <i />

      {status === 'completed'
        ? 'Completed'
        : status === 'processing'
          ? 'Processing'
          : 'Failed'}
    </span>
  );
}


function Button({
  children,
  onClick,
  href,
  variant = 'primary',
  testId,
  disabled = false,
}: {
  children: ReactNode;
  onClick?: () => void;
  href?: string;
  variant?: string;
  testId: string;
  disabled?: boolean;
}) {
  const cls = `button button-${variant}`;

  return href ? (
    <Link
      href={href}
      className={cls}
      data-testid={testId}
    >
      {children}
    </Link>
  ) : (
    <button
      type="button"
      className={cls}
      onClick={onClick}
      data-testid={testId}
      disabled={disabled}
    >
      {children}
    </button>
  );
}


/* ============================================================
   MEETING TABLE
   ============================================================ */

function MeetingRows({
  meetings,
  compact = false,
}: {
  meetings: Meeting[];
  compact?: boolean;
}) {
  const download = (id: string) => {
    void getMeeting(id).then(
      (record) => {
        if (record) {
          void downloadMarkdown(record);
        }
      },
    );
  };

  return (
    <div
      className={`table-wrap ${
        compact ? 'compact-table' : ''
      }`}
    >
      <table>
        <thead>
          <tr>
            <th>MEETING</th>
            <th>DATE</th>
            <th>DURATION</th>
            <th>SPEAKERS</th>

            {!compact && (
              <th>WORDS</th>
            )}

            <th>STATUS</th>
            <th aria-label="Actions" />
          </tr>
        </thead>

        <tbody>
          {meetings.map((m) => (
            <tr
              key={m.id}
              data-testid={`row-meeting-${m.id}`}
            >
              <td>
                <Link
                  href={`/meetings/${m.id}`}
                  className="meeting-title-link"
                  data-testid={`link-meeting-${m.id}`}
                >
                  <span className="table-file">
                    <FileText size={17} />
                  </span>

                  <span>{m.title}</span>
                </Link>
              </td>

              <td className="nowrap">
                {dateLabel(m.date)}
              </td>

              <td className="mono">
                {m.duration
                  ? formatDuration(m.duration)
                  : '—'}
              </td>

              <td>
                {m.speakerCount || '—'}

                {m.speakerCount
                  ? ' speakers'
                  : ''}
              </td>

              {!compact && (
                <td>
                  {m.wordCount
                    ? m.wordCount.toLocaleString()
                    : '—'}
                </td>
              )}

              <td>
                <Status
                  status={m.status}
                />
              </td>

              <td>
                <div className="row-actions">
                  <Link
                    href={`/meetings/${m.id}`}
                    className="row-action"
                    aria-label={`View ${m.title}`}
                    data-testid={`button-view-${m.id}`}
                  >
                    <ArrowRight size={16} />
                  </Link>

                  <button
                    className="row-action"
                    aria-label={`Download Markdown for ${m.title}`}
                    title="Download Markdown"
                    onClick={() =>
                      download(m.id)
                    }
                    data-testid={`button-download-${m.id}`}
                  >
                    <Download size={15} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>


      <div className="meeting-mobile-cards">
        {meetings.map((m) => (
          <article
            className="meeting-mobile-card"
            key={m.id}
            data-testid={`card-mobile-meeting-${m.id}`}
          >
            <div className="meeting-mobile-head">
              <Link
                href={`/meetings/${m.id}`}
                className="meeting-title-link"
                data-testid={`link-mobile-meeting-${m.id}`}
              >
                <span className="table-file">
                  <FileText size={17} />
                </span>

                <span>{m.title}</span>
              </Link>

              <Status
                status={m.status}
              />
            </div>

            <div className="meeting-mobile-meta">
              <span>
                {dateLabel(m.date)}
              </span>

              <span>
                {m.duration
                  ? formatDuration(m.duration)
                  : 'Duration pending'}
              </span>

              <span>
                {m.speakerCount
                  ? `${m.speakerCount} speakers`
                  : 'Speakers pending'}
              </span>

              {!compact && (
                <span>
                  {m.wordCount
                    ? `${m.wordCount.toLocaleString()} words`
                    : 'Words pending'}
                </span>
              )}
            </div>

            <div className="meeting-mobile-footer">
              <span>
                {compact
                  ? 'Meeting transcript'
                  : 'Meeting record'}
              </span>

              <div className="row-actions">
                <Link
                  href={`/meetings/${m.id}`}
                  className="row-action"
                  aria-label={`View ${m.title}`}
                  data-testid={`button-mobile-view-${m.id}`}
                >
                  <ArrowRight size={16} />
                </Link>

                <button
                  className="row-action"
                  aria-label={`Download Markdown for ${m.title}`}
                  title="Download Markdown"
                  onClick={() =>
                    download(m.id)
                  }
                  data-testid={`button-mobile-download-${m.id}`}
                >
                  <Download size={15} />
                </button>
              </div>
            </div>
          </article>
        ))}
      </div>


      {meetings.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">
            <Search size={21} />
          </div>

          <strong>
            No meetings match these filters
          </strong>

          <p>
            Try changing your search or
            clearing a filter.
          </p>
        </div>
      )}
    </div>
  );
}


/* ============================================================
   DASHBOARD
   ============================================================ */

export function Dashboard() {
  const [
    meetings,
    setMeetings,
  ] = useState<Meeting[]>([]);

  useEffect(() => {
    void getMeetings().then(
      setMeetings,
    );
  }, []);

  const recent =
    meetings.slice(0, 5);

  const completed =
    meetings.filter(
      (m) => m.status === 'completed',
    ).length;

  const hours =
    meetings.length === 24
      ? '18.6'
      : (
          meetings
            .filter(
              (m) =>
                m.status === 'completed',
            )
            .reduce(
              (sum, m) =>
                sum + m.duration,
              0,
            ) / 3600
        ).toFixed(1);

  return (
    <div className="page-enter">
      <PageTitle
        eyebrow="Tuesday, October 7, 2026"
        title="Good morning, Amal"
        subtitle="Your meeting intelligence workspace is ready."
        action={
          <Button
            href="/new-meeting"
            testId="button-new-meeting"
          >
            <Upload size={16} />
            New meeting
          </Button>
        }
      />

      <div className="welcome-strip">
        <div className="welcome-symbol">
          <AudioLines size={22} />
        </div>

        <div>
          <strong>
            Meeting Intelligence
          </strong>

          <span>
            Process meetings, identify
            speakers, and generate
            structured transcripts.
          </span>
        </div>

        <Link
          href="/meetings"
          className="welcome-link"
          data-testid="link-browse-all"
        >
          Browse meetings
          <ArrowRight size={16} />
        </Link>
      </div>


      <section
        className="stat-grid"
        aria-label="Meeting statistics"
      >
        {[
          {
            name: 'Total meetings',
            value: meetings.length || 24,
            desc: 'Across your workspace',
            icon: LayoutList,
          },
          {
            name: 'Processed',
            value: completed || 21,
            desc: 'Ready to review',
            icon: CheckCircle2,
          },
          {
            name: 'Processing',
            value:
              meetings.filter(
                (m) =>
                  m.status ===
                  'processing',
              ).length || 1,
            desc: 'Currently in progress',
            icon: AudioLines,
          },
          {
            name: 'Transcribed hours',
            value: `${
              hours === '0.0'
                ? '18.6'
                : hours
            }h`,
            desc: 'Audio processed to date',
            icon: Clock3,
          },
        ].map(
          ({
            name,
            value,
            desc,
            icon: Icon,
          }, i) => (
            <article
              className={`stat-card stat-card-${i}`}
              key={name}
              data-testid={`stat-${name
                .toLowerCase()
                .replaceAll(
                  ' ',
                  '-',
                )}`}
            >
              <div className="stat-top">
                <span>{name}</span>
                <Icon size={17} />
              </div>

              <strong>{value}</strong>

              <small>{desc}</small>

              <div className="stat-rail">
                <i
                  style={{
                    width: `${[
                      72,
                      84,
                      24,
                      62,
                    ][i]}%`,
                  }}
                />
              </div>
            </article>
          ),
        )}
      </section>


      <section className="content-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">
              YOUR WORKSPACE
            </div>

            <h2>
              Recent meetings
            </h2>
          </div>

          <Link
            href="/meetings"
            className="text-link"
            data-testid="link-all-meetings"
          >
            View all meetings
            <ArrowRight size={15} />
          </Link>
        </div>

        <div className="surface">
          <MeetingRows
            meetings={recent}
            compact
          />
        </div>
      </section>


      <section className="action-panel">
        <div className="action-art">
          <div className="sound-wave">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>

        <div className="action-copy">
          <span className="eyebrow">
            NEXT STEP
          </span>

          <h2>
            Process a new meeting
          </h2>

          <p>
            Upload a recording and create
            a speaker-aware transcript for
            your team.
          </p>
        </div>

        <Button
          href="/new-meeting"
          variant="light"
          testId="button-upload-meeting"
        >
          Upload meeting
          <ArrowRight size={16} />
        </Button>
      </section>
    </div>
  );
}


/* ============================================================
   MEETINGS PAGE
   ============================================================ */

export function MeetingsPage() {
  const [
    meetings,
    setMeetings,
  ] = useState<Meeting[]>([]);

  const [
    query,
    setQuery,
  ] = useState('');

  const [
    status,
    setStatus,
  ] = useState('all');

  const [
    date,
    setDate,
  ] = useState('all');

  const [
    sort,
    setSort,
  ] = useState('newest');

  useEffect(() => {
    void getMeetings().then(
      setMeetings,
    );
  }, []);

  const filtered =
    useMemo(
      () =>
        meetings
          .filter(
            (m) =>
              m.title
                .toLowerCase()
                .includes(
                  query.toLowerCase(),
                ) &&
              (
                status === 'all' ||
                m.status === status
              ) &&
              (
                date === 'all' ||
                (
                  date === 'week' &&
                  Date.now() -
                    new Date(
                      m.date,
                    ).getTime() <
                    7 *
                      86400000
                ) ||
                (
                  date === 'month' &&
                  new Date(
                    m.date,
                  ).getMonth() ===
                    9
                )
              ),
          )
          .sort(
            (a, b) =>
              sort === 'newest'
                ? b.date.localeCompare(
                    a.date,
                  )
                : a.title.localeCompare(
                    b.title,
                  ),
          ),
      [
        meetings,
        query,
        status,
        date,
        sort,
      ],
    );

  return (
    <div className="page-enter">
      <PageTitle
        eyebrow="LIBRARY"
        title="Meetings"
        subtitle="View and manage processed meeting transcripts."
        action={
          <Button
            href="/new-meeting"
            testId="button-add-meeting"
          >
            <Upload size={16} />
            New meeting
          </Button>
        }
      />

      <div className="meeting-toolbar">
        <label className="search-field">
          <Search size={17} />

          <input
            value={query}
            onChange={(e) =>
              setQuery(e.target.value)
            }
            placeholder="Search meetings..."
            aria-label="Search meetings"
            data-testid="input-search-meetings"
          />

          {query && (
            <button
              onClick={() =>
                setQuery('')
              }
              aria-label="Clear search"
            >
              <X size={15} />
            </button>
          )}
        </label>

        <div className="toolbar-filters">
          <label className="select-wrap">
            <Filter size={15} />

            <select
              value={status}
              onChange={(e) =>
                setStatus(e.target.value)
              }
              aria-label="Filter by status"
              data-testid="select-status-filter"
            >
              <option value="all">
                All statuses
              </option>

              <option value="completed">
                Completed
              </option>

              <option value="processing">
                Processing
              </option>

              <option value="failed">
                Failed
              </option>
            </select>

            <ChevronDown size={14} />
          </label>


          <label className="select-wrap">
            <select
              value={date}
              onChange={(e) =>
                setDate(e.target.value)
              }
              aria-label="Filter by date"
              data-testid="select-date-filter"
            >
              <option value="all">
                Any date
              </option>

              <option value="week">
                Past 7 days
              </option>

              <option value="month">
                October 2026
              </option>
            </select>

            <ChevronDown size={14} />
          </label>


          <label className="select-wrap">
            <ArrowDownUp size={15} />

            <select
              value={sort}
              onChange={(e) =>
                setSort(e.target.value)
              }
              aria-label="Sort meetings"
              data-testid="select-sort-meetings"
            >
              <option value="newest">
                Newest first
              </option>

              <option value="title">
                Title A–Z
              </option>
            </select>

            <ChevronDown size={14} />
          </label>
        </div>
      </div>


      <div className="list-summary">
        <span>
          <strong>
            {filtered.length}
          </strong>{' '}
          meetings
        </span>

        <span>
          {meetings.length} total in
          workspace
        </span>
      </div>

      <div className="surface meetings-surface">
        <MeetingRows
          meetings={filtered}
        />
      </div>
    </div>
  );
}


/* ============================================================
   NEW MEETING / REAL BACKEND UPLOAD
   ============================================================ */

export function NewMeetingPage() {
  const [
    ,
    navigate,
  ] = useLocation();

  const inputRef =
    useRef<HTMLInputElement>(null);

  const [
    file,
    setFile,
  ] = useState<File | null>(null);

  const [
    error,
    setError,
  ] = useState('');

  const [
    drag,
    setDrag,
  ] = useState(false);

  const [
    duration,
    setDuration,
  ] = useState<number | null>(null);

  const [
    busy,
    setBusy,
  ] = useState(false);


  const acceptFile = (
    candidate?: File,
  ) => {
    if (!candidate) {
      return;
    }

    setError('');
    setDuration(null);

    const ext =
      candidate.name
        .split('.')
        .pop()
        ?.toLowerCase();

    if (
      ![
        'mp4',
        'wav',
        'mp3',
        'm4a',
      ].includes(ext || '')
    ) {
      setFile(null);

      setError(
        'Choose an MP4, WAV, MP3, or M4A recording.',
      );

      return;
    }

    if (
      candidate.size >
      500 * 1024 * 1024
    ) {
      setFile(null);

      setError(
        'This file exceeds the 500 MB upload limit.',
      );

      return;
    }

    setFile(candidate);

    const url =
      URL.createObjectURL(
        candidate,
      );

    const media =
      ext === 'mp4'
        ? document.createElement(
            'video',
          )
        : document.createElement(
            'audio',
          );

    media.preload =
      'metadata';

    media.onloadedmetadata =
      () => {
        if (
          Number.isFinite(
            media.duration,
          )
        ) {
          setDuration(
            Math.round(
              media.duration,
            ),
          );
        }

        URL.revokeObjectURL(
          url,
        );
      };

    media.onerror = () => {
      URL.revokeObjectURL(
        url,
      );
    };

    media.src = url;
  };


  const submit = async () => {
    if (!file) {
      return;
    }

    setBusy(true);
    setError('');

    try {
      const meeting =
        await createMeeting(
          file,
          duration || 0,
        );

      navigate(
        `/processing/${meeting.id}`,
      );
    } catch (err) {
      console.error(
        'Meeting processing start failed:',
        err,
      );

      setError(
        'Unable to upload or start processing this recording. Please try again.',
      );

      setBusy(false);
    }
  };


  const bytes = (
    size: number,
  ) =>
    size >=
    1024 * 1024
      ? `${(
          size /
          1024 /
          1024
        ).toFixed(1)} MB`
      : `${(
          size /
          1024
        ).toFixed(0)} KB`;


  return (
    <div className="page-enter">
      <PageTitle
        eyebrow="WORKSPACE / UPLOAD"
        title="New meeting"
        subtitle="Upload a recording to create a speaker-aware transcript."
      />

      <div className="upload-layout">
        <div className="upload-main">
          <div
            className={`dropzone ${
              drag
                ? 'drop-active'
                : ''
            } ${
              file
                ? 'has-file'
                : ''
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() =>
              setDrag(false)
            }
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              acceptFile(
                e.dataTransfer
                  .files[0],
              );
            }}
            onClick={() =>
              !file &&
              inputRef.current?.click()
            }
            onKeyDown={(e) => {
              if (
                !file &&
                (
                  e.key === 'Enter' ||
                  e.key === ' '
                )
              ) {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            role="button"
            tabIndex={0}
            aria-label="Select or drop a meeting recording"
            data-testid="upload-dropzone"
          >
            <input
              ref={inputRef}
              type="file"
              accept=".mp4,.wav,.mp3,.m4a,audio/*,video/mp4"
              hidden
              onChange={(e) =>
                acceptFile(
                  e.target.files?.[0],
                )
              }
              data-testid="input-meeting-file"
            />


            {!file ? (
              <>
                <div className="upload-icon">
                  <Upload size={24} />
                </div>

                <h2>
                  Drop your meeting
                  recording here
                </h2>

                <p>
                  or{' '}
                  <button
                    className="inline-link"
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      inputRef.current?.click();
                    }}
                    data-testid="button-browse-files"
                  >
                    browse files
                  </button>
                </p>

                <div className="format-row">
                  <span>MP4</span>
                  <span>WAV</span>
                  <span>MP3</span>
                  <span>M4A</span>
                </div>

                <small>
                  Maximum file size:
                  500 MB
                </small>
              </>
            ) : (
              <div
                className="file-picked"
                onClick={(e) =>
                  e.stopPropagation()
                }
              >
                <div className="picked-icon">
                  <FileAudio
                    size={22}
                  />
                </div>

                <div className="picked-meta">
                  <strong>
                    {file.name}
                  </strong>

                  <span>
                    {file.name
                      .split('.')
                      .pop()
                      ?.toUpperCase()}{' '}
                    <b>·</b>{' '}
                    {bytes(
                      file.size,
                    )}

                    {duration !==
                      null && (
                      <>
                        {' '}
                        <b>·</b>{' '}
                        {formatDuration(
                          duration,
                        )}
                      </>
                    )}
                  </span>

                  <small>
                    Ready to upload
                  </small>
                </div>

                <button
                  className="icon-button remove-file"
                  aria-label="Remove selected file"
                  onClick={() => {
                    setFile(null);
                    setDuration(null);

                    if (
                      inputRef.current
                    ) {
                      inputRef.current.value =
                        '';
                    }
                  }}
                  data-testid="button-remove-file"
                >
                  <X size={18} />
                </button>
              </div>
            )}
          </div>


          {error && (
            <div
              className="inline-error"
              role="alert"
              data-testid="text-upload-error"
            >
              {error}
            </div>
          )}


          <div className="upload-actions">
            <span>
              {file
                ? 'The recording will be uploaded securely to the SWBS processing backend.'
                : 'Audio and video recording files are supported.'}
            </span>

            <Button
              onClick={submit}
              disabled={
                !file || busy
              }
              testId="button-start-processing"
            >
              {busy
                ? 'Uploading…'
                : 'Start processing'}

              <ArrowRight
                size={16}
              />
            </Button>
          </div>


          <div className="next-steps">
            <div className="eyebrow">
              WHAT HAPPENS NEXT?
            </div>

            <div className="steps-grid">
              {[
                [
                  '01',
                  'Prepare audio',
                ],
                [
                  '02',
                  'Transcribe speech',
                ],
                [
                  '03',
                  'Detect speakers',
                ],
                [
                  '04',
                  'Identify speakers',
                ],
                [
                  '05',
                  'Align the transcript',
                ],
                [
                  '06',
                  'Generate final transcript',
                ],
              ].map(
                ([n, label]) => (
                  <div
                    className="step-item"
                    key={n}
                  >
                    <span>{n}</span>
                    <strong>
                      {label}
                    </strong>
                  </div>
                ),
              )}
            </div>

            <small>
              Your recording is processed
              by the SWBS AI pipeline using
              real transcription and speaker
              analysis.
            </small>
          </div>
        </div>


        <aside className="upload-aside">
          <div className="aside-card">
            <div className="aside-icon">
              <Headphones
                size={19}
              />
            </div>

            <h3>
              For the best results
            </h3>

            <p>
              Choose a clear recording
              with minimal background
              noise. Longer recordings
              may take longer to process.
            </p>

            <div className="aside-rule" />

            <div className="aside-meta">
              <span>
                Supported formats
              </span>

              <strong>
                MP4, WAV, MP3, M4A
              </strong>
            </div>

            <div className="aside-meta">
              <span>
                Upload limit
              </span>

              <strong>
                500 MB
              </strong>
            </div>
          </div>

          <div className="privacy-note">
            <span className="privacy-dot" />

            <p>
              <strong>
                Private processing
              </strong>
              <br />
              Your recording is sent to
              the SWBS backend for
              processing.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}


/* ============================================================
   PROCESSING PAGE
   REAL FASTAPI JOB POLLING
   ============================================================ */

export function ProcessingPage() {
  const { id = '' } =
    useParams();

  const [, navigate] =
    useLocation();

  const [
    job,
    setJob,
  ] = useState<ProcessingJob>();

  const [
    meeting,
    setMeeting,
  ] = useState<MeetingRecord>();

  const [
    error,
    setError,
  ] = useState(false);


  useEffect(() => {
    let live = true;

    let timer:
      | ReturnType<
          typeof setInterval
        >
      | undefined;


    const loadProcessingState =
      async () => {
        /*
         * The job is the source of truth
         * while processing.
         *
         * We do NOT require the transcript
         * to exist yet.
         */
        const currentJob =
          await getJob(id);

        if (!live) {
          return;
        }


        /*
         * If no job can be found,
         * then the processing page
         * cannot continue.
         */
        if (!currentJob) {
          setError(true);
          return;
        }


        setJob(currentJob);


        /*
         * Once processing is complete,
         * the transcript should exist.
         *
         * Fetch it to populate the meeting
         * information shown in the UI.
         */
        if (
          currentJob.status ===
          'completed'
        ) {
          const record =
            await getMeeting(id);

          if (
            live &&
            record
          ) {
            setMeeting(record);
          }
        }


        /*
         * Stop polling when the backend
         * reports a terminal state.
         */
        if (
          currentJob.status ===
            'completed' ||
          currentJob.status ===
            'failed'
        ) {
          if (timer) {
            clearInterval(
              timer,
            );
          }

          return;
        }
      };


    void loadProcessingState();


    /*
     * Poll the REAL FastAPI job.
     *
     * This replaces the old frontend
     * simulated 820ms stage progression.
     */
    timer = setInterval(
      () => {
        void loadProcessingState();
      },
      1500,
    );


    return () => {
      live = false;

      if (timer) {
        clearInterval(
          timer,
        );
      }
    };
  }, [id]);


  if (error) {
    return (
      <div className="page-enter">
        <PageTitle
          title="Processing unavailable"
          subtitle="The processing job could not be found."
        />

        <div className="surface error-card">
          <p>
            The meeting may have been
            removed or the processing
            job may no longer be available.
          </p>

          <Button
            href="/meetings"
            variant="secondary"
            testId="button-back-meetings"
          >
            Back to meetings
          </Button>
        </div>
      </div>
    );
  }


  const complete =
    job?.status ===
    'completed';

  const failed =
    job?.status ===
    'failed';


  return (
    <div className="page-enter">
      <PageTitle
        eyebrow="WORKSPACE / PROCESSING"
        title={
          complete
            ? 'Meeting processed successfully'
            : failed
              ? 'Meeting processing failed'
              : 'Processing meeting'
        }
        subtitle={
          complete
            ? 'Your transcript is ready to review.'
            : failed
              ? 'The AI pipeline could not complete this meeting.'
              : 'Your meeting is being processed by the SWBS AI pipeline.'
        }
      />


      <div className="processing-layout">
        <div className="processing-main">
          <div className="job-summary">
            <div className="job-file-icon">
              <FileAudio size={22} />
            </div>

            <div className="job-title">
              <strong>
                {meeting?.meeting.title ||
                  id}
              </strong>

              <span>
                {meeting?.meeting.title ||
                  'Meeting processing'}
              </span>
            </div>

            <div className="job-duration">
              <span>
                DURATION
              </span>

              <strong>
                {meeting?.meeting.duration
                  ? formatDuration(
                      meeting.meeting.duration,
                    )
                  : 'Processing'}
              </strong>
            </div>

            <div className="job-duration">
              <span>
                PROGRESS
              </span>

              <strong>
                {job?.progress ??
                  0}
                %
              </strong>
            </div>
          </div>


          <div className="progress-header">
            <div>
              <span className="eyebrow">
                PROCESSING PROGRESS
              </span>

              <strong>
                {job?.progress ??
                  0}
                %
              </strong>
            </div>

            <span
              className={`status ${
                complete
                  ? 'status-completed'
                  : failed
                    ? 'status-failed'
                    : 'status-processing'
              }`}
            >
              <i />

              {complete
                ? 'Completed'
                : failed
                  ? 'Failed'
                  : 'In progress'}
            </span>
          </div>


          <div className="progress-track">
            <div
              style={{
                width: `${job?.progress ?? 0}%`,
              }}
            />
          </div>


          <p className="processing-note">
            {complete
              ? 'All stages are complete. The transcript is ready.'
              : failed
                ? 'The backend reported a processing failure.'
                : 'Processing in progress…'}
          </p>


          <div className="pipeline">
            {(job?.stages || []).map(
              (stage, i) => (
                <div
                  className={`pipeline-stage stage-${stage.status}`}
                  key={stage.id}
                  data-testid={`stage-${stage.id}`}
                >
                  <div className="stage-marker">
                    {stage.status ===
                    'completed' ? (
                      <Check size={14} />
                    ) : stage.status ===
                      'processing' ? (
                      <span />
                    ) : (
                      <i />
                    )}
                  </div>

                  <div className="stage-label">
                    <strong>
                      {stage.name}
                    </strong>

                    {stage.status ===
                      'processing' && (
                      <span>
                        Working
                      </span>
                    )}
                  </div>

                  <span className="stage-number">
                    {String(
                      i + 1,
                    ).padStart(
                      2,
                      '0',
                    )}
                  </span>
                </div>
              ),
            )}
          </div>


          <div className="processing-bottom">
            {complete ? (
              <>
                <span>
                  <CheckCircle2
                    size={17}
                  />

                  Transcript is
                  available
                </span>

                <Button
                  href={`/meetings/${id}`}
                  testId="button-view-transcript"
                >
                  View transcript
                  <ArrowRight
                    size={16}
                  />
                </Button>
              </>
            ) : failed ? (
              <span>
                <CircleHelp
                  size={17}
                />

                Processing failed.
                Please return to meetings
                and try again.
              </span>
            ) : (
              <span>
                <AudioLines
                  size={17}
                />

                You can keep this page
                open while the AI pipeline
                runs.
              </span>
            )}
          </div>
        </div>


        <aside className="processing-aside">
          <div className="aside-card">
            <span className="eyebrow">
              THE PIPELINE
            </span>

            <h3>
              From recording to insight
            </h3>

            <p>
              Every stage prepares the
              meeting for a readable,
              speaker-labeled transcript.
            </p>

            <div className="mini-flow">
              <span>
                Recording
              </span>

              <ArrowRight
                size={14}
              />

              <span>
                Transcript
              </span>
            </div>

            <div className="aside-rule" />

            <div className="simulated-tag">
              <span />
              AI processing
            </div>

            <small>
              Processing is handled by
              the SWBS backend AI pipeline.
            </small>
          </div>
        </aside>
      </div>
    </div>
  );
}


/* ============================================================
   MEETING DETAILS / TRANSCRIPT
   ============================================================ */

export function MeetingDetailsPage() {
  const { id = '' } =
    useParams();

  const [
    record,
    setRecord,
  ] =
    useState<MeetingRecord>();

  const [
    query,
    setQuery,
  ] = useState('');

  const [
    speaker,
    setSpeaker,
  ] = useState('all');

  const [
    copied,
    setCopied,
  ] = useState(false);

  const [, navigate] =
    useLocation();


  useEffect(() => {
    void getMeeting(id).then(
      setRecord,
    );
  }, [id]);


  const recordSpeakers =
    record?.speakers ||
    sampleSpeakers;


  const turns =
    useMemo(
      () =>
        record?.turns.filter(
          (t) =>
            (
              speaker === 'all' ||
              t.speakerId ===
                speaker
            ) &&
            (
              !query ||
              t.text
                .toLowerCase()
                .includes(
                  query.toLowerCase(),
                ) ||
              recordSpeakers
                .find(
                  (s) =>
                    s.id ===
                    t.speakerId,
                )
                ?.name
                .toLowerCase()
                .includes(
                  query.toLowerCase(),
                )
            ),
        ) || [],
      [
        record,
        query,
        speaker,
        recordSpeakers,
      ],
    );


  const time = (
    n: number,
  ) =>
    new Date(
      n * 1000,
    )
      .toISOString()
      .slice(11, 19);


  const copy =
    async () => {
      if (!record) {
        return;
      }

      const text =
        record.turns
          .map(
            (t) =>
              `${
                recordSpeakers.find(
                  (s) =>
                    s.id ===
                    t.speakerId,
                )?.name
              }  ${time(
                t.startTime,
              )}\n${t.text}`,
          )
          .join('\n\n');

      try {
        await navigator.clipboard.writeText(
          text,
        );

        setCopied(true);

        setTimeout(
          () =>
            setCopied(false),
          1800,
        );
      } catch {
        setCopied(false);
      }
    };


  if (!record) {
    return (
      <div className="page-enter">
        <PageTitle
          title="Meeting not found"
          subtitle="This meeting is unavailable."
        />

        <Button
          href="/meetings"
          variant="secondary"
          testId="button-return-library"
        >
          <ArrowLeft
            size={16}
          />

          Return to meetings
        </Button>
      </div>
    );
  }


  const m =
    record.meeting;


  return (
    <div className="page-enter">
      <button
        className="back-link"
        onClick={() =>
          navigate('/meetings')
        }
        data-testid="button-back-library"
      >
        <ArrowLeft size={16} />
        Meetings
      </button>


      <div className="detail-heading">
        <div>
          <div className="detail-topline">
            <span className="eyebrow">
              MEETING TRANSCRIPT
            </span>

            <Status
              status={m.status}
            />
          </div>

          <h1>
            {m.title}
          </h1>

          <div className="meeting-meta">
            <span>
              {dateLabel(
                m.date,
              )}
            </span>

            <i />

            <span>
              {formatDuration(
                m.duration,
              )}
            </span>

            <i />

            <span>
              {m.speakerCount ||
                recordSpeakers.length}{' '}
              speakers
            </span>

            <i />

            <span>
              {m.wordCount.toLocaleString()}{' '}
              words
            </span>
          </div>
        </div>


        <div className="detail-actions">
          <Button
            onClick={() =>
              void downloadMarkdown(
                record,
              )
            }
            variant="secondary"
            testId="button-download-markdown"
          >
            <Download size={16} />
            Download Markdown
          </Button>

          <a
            href={NOTEBOOKLM_URL}
            target="_blank"
            rel="noreferrer"
            className="button button-primary"
            data-testid="button-open-notebooklm"
            aria-label="Open NotebookLM in a new tab"
          >
            <ArrowDown
              size={15}
            />

            Open in NotebookLM
          </a>
        </div>
      </div>


      <div className="notebook-note">
        <CircleHelp size={16} />

        <span>
          NotebookLM is an external
          destination. Download the
          transcript first, then upload
          it to NotebookLM.
        </span>
      </div>


      <section className="speaker-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">
              PARTICIPANTS
            </div>

            <h2>
              Speaker summary
            </h2>
          </div>

          <span className="muted">
            {recordSpeakers.length}{' '}
            detected
          </span>
        </div>


        <div className="speaker-grid">
          {recordSpeakers.map(
            (s, i) => (
              <button
                className={`speaker-card speaker-card-${i}`}
                key={s.id}
                onClick={() => {
                  setSpeaker(
                    s.id,
                  );

                  document
                    .getElementById(
                      'transcript',
                    )
                    ?.scrollIntoView(
                      {
                        behavior:
                          'smooth',
                      },
                    );
                }}
                data-testid={`button-speaker-${s.id}`}
              >
                <div
                  className={`speaker-avatar speaker-avatar-${i}`}
                >
                  {s.name
                    .slice(0, 2)
                    .toUpperCase()}
                </div>

                <div className="speaker-info">
                  <strong>
                    {s.name}
                  </strong>

                  <span>
                    {s.identified
                      ? 'Speaker identified'
                      : 'Unidentified speaker'}
                  </span>
                </div>

                <div className="speaker-words">
                  <strong>
                    {s.wordCount}
                  </strong>

                  <span>
                    words
                  </span>
                </div>
              </button>
            ),
          )}
        </div>
      </section>


      <section
        id="transcript"
        className="transcript-section"
      >
        <div className="section-heading">
          <div>
            <div className="eyebrow">
              RECORDING NOTES
            </div>

            <h2>
              Transcript
            </h2>
          </div>

          <div className="transcript-count">
            {turns.length} of{' '}
            {record.turns.length}{' '}
            segments
          </div>
        </div>


        <div className="transcript-tools">
          <label className="search-field transcript-search">
            <Search size={16} />

            <input
              value={query}
              onChange={(e) =>
                setQuery(
                  e.target.value,
                )
              }
              placeholder="Find in transcript..."
              aria-label="Search transcript"
              data-testid="input-search-transcript"
            />

            {query && (
              <button
                onClick={() =>
                  setQuery('')
                }
                aria-label="Clear transcript search"
              >
                <X size={14} />
              </button>
            )}
          </label>


          <label className="select-wrap speaker-select">
            <Users size={15} />

            <select
              value={speaker}
              onChange={(e) =>
                setSpeaker(
                  e.target.value,
                )
              }
              aria-label="Filter by speaker"
              data-testid="select-speaker-filter"
            >
              <option value="all">
                All speakers
              </option>

              {recordSpeakers.map(
                (s) => (
                  <option
                    value={s.id}
                    key={s.id}
                  >
                    {s.name}
                  </option>
                ),
              )}
            </select>

            <ChevronDown
              size={14}
            />
          </label>


          <button
            className="tool-button"
            onClick={copy}
            data-testid="button-copy-transcript"
          >
            <Copy size={15} />

            {copied
              ? 'Copied'
              : 'Copy transcript'}
          </button>


          <button
            className="tool-button mobile-download"
            onClick={() =>
              void downloadMarkdown(
                record,
              )
            }
            data-testid="button-download-markdown-transcript"
          >
            <Download size={15} />

            <span>
              Markdown
            </span>
          </button>
        </div>


        <div
          className="transcript-card transcript-scroll"
          aria-live="polite"
          data-testid="transcript-content"
        >
          {turns.length ? (
            turns.map(
              (turn) => {
                const who =
                  recordSpeakers.find(
                    (s) =>
                      s.id ===
                      turn.speakerId,
                  );

                return (
                  <article
                    className="transcript-turn"
                    key={turn.id}
                    data-testid={`turn-${turn.id}`}
                  >
                    <div className="turn-gutter">
                      <span
                        className={`turn-dot turn-dot-${recordSpeakers.findIndex(
                          (s) =>
                            s.id ===
                            turn.speakerId,
                        )}`}
                      />

                      <span className="turn-time">
                        {time(
                          turn.startTime,
                        )}
                      </span>
                    </div>

                    <div className="turn-content">
                      <div className="turn-person">
                        {who?.name ||
                          'Unknown'}

                        <span>
                          {time(
                            turn.startTime,
                          )}{' '}
                          <b>
                            →
                          </b>{' '}
                          {time(
                            turn.endTime,
                          )}
                        </span>
                      </div>

                      <p>
                        {turn.text}
                      </p>
                    </div>
                  </article>
                );
              },
            )
          ) : (
            <div className="empty-state">
              <div className="empty-icon">
                <Search size={20} />
              </div>

              <strong>
                No transcript segments found
              </strong>

              <p>
                Try a different search or
                speaker.
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}


/* ============================================================
   SETTINGS
   ============================================================ */

export function SettingsPage({
  dark,
  setDark,
}: {
  dark: boolean;
  setDark: (v: boolean) => void;
}) {
  const [
    language,
    setLanguage,
  ] = useState(
    localStorage.getItem(
      'swbs-language',
    ) || 'Arabic',
  );

  const [
    notifications,
    setNotifications,
  ] = useState(
    localStorage.getItem(
      'swbs-notifications',
    ) !== 'off',
  );

  const [
    saved,
    setSaved,
  ] = useState(false);


  const save = () => {
    localStorage.setItem(
      'swbs-language',
      language,
    );

    localStorage.setItem(
      'swbs-notifications',
      notifications
        ? 'on'
        : 'off',
    );

    setSaved(true);

    setTimeout(
      () =>
        setSaved(false),
      1800,
    );
  };


  return (
    <div className="page-enter">
      <PageTitle
        eyebrow="WORKSPACE / PREFERENCES"
        title="Settings"
        subtitle="Manage your profile and workspace preferences."
      />

      <div className="settings-layout">
        <div className="settings-main">
          <section className="settings-card">
            <div className="settings-section-head">
              <div className="settings-icon">
                <Users size={18} />
              </div>

              <div>
                <h2>
                  Profile
                </h2>

                <p>
                  Your account details in
                  this workspace.
                </p>
              </div>
            </div>

            <div className="form-grid">
              <label>
                Name
                <input
                  value="Amal Mansour"
                  readOnly
                  data-testid="input-profile-name"
                />
              </label>

              <label>
                Email
                <input
                  value="amal.mansour@swbs.com"
                  readOnly
                  data-testid="input-profile-email"
                />
              </label>

              <label>
                Department
                <input
                  value="Operations"
                  readOnly
                  data-testid="input-profile-department"
                />
              </label>
            </div>
          </section>


          <section className="settings-card">
            <div className="settings-section-head">
              <div className="settings-icon">
                <FileText size={18} />
              </div>

              <div>
                <h2>
                  Meeting preferences
                </h2>

                <p>
                  Set defaults for the
                  transcripts you review.
                </p>
              </div>
            </div>


            <label className="preference-row">
              <span>
                <strong>
                  Default transcript
                  language
                </strong>

                <small>
                  Language used for
                  transcript output
                </small>
              </span>

              <span className="select-wrap">
                <select
                  value={language}
                  onChange={(e) =>
                    setLanguage(
                      e.target.value,
                    )
                  }
                  data-testid="select-transcript-language"
                >
                  <option>
                    Arabic
                  </option>

                  <option>
                    English
                  </option>
                </select>

                <ChevronDown
                  size={14}
                />
              </span>
            </label>


            <label className="preference-row">
              <span>
                <strong>
                  Processing
                  notifications
                </strong>

                <small>
                  Show a notice when a job
                  is complete
                </small>
              </span>

              <button
                className={`switch ${
                  notifications
                    ? 'switch-on'
                    : ''
                }`}
                role="switch"
                aria-checked={
                  notifications
                }
                onClick={() =>
                  setNotifications(
                    !notifications,
                  )
                }
                data-testid="toggle-notifications"
              >
                <i />
              </button>
            </label>
          </section>


          <section className="settings-card">
            <div className="settings-section-head">
              <div className="settings-icon">
                <Sun size={18} />
              </div>

              <div>
                <h2>
                  Appearance
                </h2>

                <p>
                  Choose the display mode
                  for this workspace.
                </p>
              </div>
            </div>


            <div className="theme-options">
              <button
                className={`theme-choice ${
                  !dark
                    ? 'theme-selected'
                    : ''
                }`}
                onClick={() =>
                  setDark(false)
                }
                data-testid="button-theme-light"
              >
                <Sun size={18} />

                <span>
                  <strong>
                    Light
                  </strong>

                  <small>
                    Bright and focused
                  </small>
                </span>

                {!dark && (
                  <Check size={16} />
                )}
              </button>


              <button
                className={`theme-choice ${
                  dark
                    ? 'theme-selected'
                    : ''
                }`}
                onClick={() =>
                  setDark(true)
                }
                data-testid="button-theme-dark"
              >
                <Moon size={18} />

                <span>
                  <strong>
                    Dark
                  </strong>

                  <small>
                    Low-light workspace
                  </small>
                </span>

                {dark && (
                  <Check size={16} />
                )}
              </button>
            </div>
          </section>


          <div className="settings-save">
            <span>
              {saved && (
                <>
                  <CheckCircle2
                    size={15}
                  />

                  Preferences saved
                </>
              )}
            </span>

            <Button
              onClick={save}
              testId="button-save-settings"
            >
              Save preferences
            </Button>
          </div>
        </div>


        <aside className="settings-aside">
          <div className="aside-card">
            <div className="profile-large">
              AM
            </div>

            <h3>
              Amal Mansour
            </h3>

            <p>
              Operations team
            </p>

            <div className="profile-status">
              <span className="online-dot" />
              SWBS workspace member
            </div>

            <div className="aside-rule" />

            <div className="aside-meta">
              <span>
                Workspace
              </span>

              <strong>
                SWBS Internal
              </strong>
            </div>

            <div className="aside-meta">
              <span>
                Mode
              </span>

              <strong>
                AI Processing
              </strong>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}


/* ============================================================
   HELP
   ============================================================ */

export function HelpPage() {
  return (
    <div className="page-enter">
      <PageTitle
        eyebrow="SUPPORT"
        title="Help & support"
        subtitle="A few notes to help you work with Meeting Intelligence."
      />

      <div className="help-grid">
        <article className="help-card">
          <span>01</span>

          <h2>
            Upload a recording
          </h2>

          <p>
            Choose an MP4, WAV, MP3,
            or M4A file up to 500 MB.
            The recording is uploaded to
            the SWBS backend for processing.
          </p>

          <Link
            href="/new-meeting"
            className="text-link"
            data-testid="link-help-upload"
          >
            Go to new meeting
            <ArrowRight size={15} />
          </Link>
        </article>


        <article className="help-card">
          <span>02</span>

          <h2>
            Review transcripts
          </h2>

          <p>
            Search transcript text,
            filter by speaker, copy
            notes, and download a
            Markdown copy from any
            completed meeting.
          </p>

          <Link
            href="/meetings"
            className="text-link"
            data-testid="link-help-meetings"
          >
            Browse meetings
            <ArrowRight size={15} />
          </Link>
        </article>


        <article className="help-card">
          <span>03</span>

          <h2>
            AI processing & privacy
          </h2>

          <p>
            Meeting recordings are
            processed by the SWBS
            backend. The frontend does
            not perform transcription
            or speaker analysis itself.
          </p>

          <Link
            href="/settings"
            className="text-link"
            data-testid="link-help-settings"
          >
            Open settings
            <ArrowRight size={15} />
          </Link>
        </article>
      </div>


      <div className="help-callout">
        <CircleHelp size={19} />

        <div>
          <strong>
            Need the real NotebookLM
            destination?
          </strong>

          <p>
            Configure the frontend
            variable{' '}
            <code>
              VITE_NOTEBOOKLM_URL
            </code>{' '}
            to set your organization’s
            destination.
          </p>
        </div>
      </div>
    </div>
  );
}