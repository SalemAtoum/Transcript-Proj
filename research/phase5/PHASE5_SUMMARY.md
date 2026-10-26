The automatic speaker identification module successfully mapped the two diarized anonymous speakers to their enrolled identities using ECAPA-TDNN embeddings and cosine similarity. The resulting mapping was SPEAKER_00 → Sami and SPEAKER_01 → Salameh, with cosine similarities of 0.7418 and 0.8615 respectively. The alternative global assignment had a substantially lower total similarity (0.4919 vs. 1.6033), yielding an assignment margin of 1.1114.


                  meeting_processed.wav
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
        Whisper ASR                 PyAnnote
             │                           │
             │                    SPEAKER_00
             │                    SPEAKER_01
             │                           │
             │                           ▼
             │                    ECAPA-TDNN
             │                           │
             │                     Voice Profiles
             │                           │
             │                    Sami / Salameh
             │                           │
             └──────────────┬────────────┘
                            ▼
                     PHASE 6
                Transcript Alignment