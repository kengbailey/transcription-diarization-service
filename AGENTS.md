# AGENTS.md

Guide for AI coding assistants working on this repository.

## Project Overview

**Speaker Diarization Service** - A complete solution for:
- **Speaker Diarization**: Detect who spoke when in audio
- **Speaker Identification**: Match voices to registered profiles
- **Transcription**: Generate speaker-attributed transcripts

## Architecture

```
├── api/                    # FastAPI backend (Python)
│   ├── main.py            # API endpoints
│   ├── config.py          # Settings (env vars)
│   ├── api_models/        # Pydantic schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services/          # Core ML services
│       ├── diarization.py # pyannote speaker diarization
│       ├── embedding.py   # wespeaker embeddings
│       ├── speaker_db.py  # Qdrant vector storage
│       ├── whisper.py     # Whisper API client
│       └── transcript_merger.py
├── ui/                     # React frontend (TypeScript)
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── TranscriptionTab.tsx
│       │   ├── speakers/SpeakersTab.tsx
│       │   ├── SettingsTab.tsx
│       │   └── ui/        # Reusable UI components
│       └── lib/
│           ├── api.ts     # API client
│           └── utils.ts
├── pipeline/               # Batch meeting-processing scripts (currently deferred; see PLAN.md)
├── docker-compose.yml      # GPU deployment
├── docker-compose.debug.yml # CUDA debug env overrides (opt-in)
└── docker-compose.cpu.yml  # CPU deployment
```

**Current focus (see PLAN.md):** hardening the `api/` service for multi-client LAN use. All compute runs on this host's RTX 3090; former remote-GPU boxes (192.168.8.116/.147) are gone — never point config at them.

## Tech Stack

### Backend
- **FastAPI** - REST API framework
- **pyannote.audio 4.0** - Speaker diarization (GPU compose: `pyannote/speaker-diarization-community-1`; CPU compose: `pyannote/speaker-diarization-3.1`)
- **wespeaker ResNet221-LM (ONNX)** - Speaker-ID embeddings (`data/models/wespeaker-resnet221/`). The diarization pipeline internally uses ResNet34 — same 256 dims, DIFFERENT embedding space; never mix vectors from the two in one collection.
- **Qdrant** - Vector database for speaker embeddings
- **PyTorch 2.8** + onnxruntime-gpu 1.22 (last CUDA 12.x build)
- **ASR: Parakeet** (`nvidia/parakeet-tdt-0.6b-v3` via the parakeet.cpp compose service, port 8081). WAV-only and crashes on long inputs, so the api transcodes and chunks (WHISPER_SEND_WAV / WHISPER_CHUNK_SECONDS). Fallback: speaches container on host port 8000 with `Systran/faster-whisper-large-v3`.

### Frontend
- **React 19** + **TypeScript 5**
- **Vite** - Build tool
- **TailwindCSS 4** - Styling
- **WaveSurfer.js** - Audio visualization

## API Endpoints

### Health & Stats
- `GET /health` - Service health status
- `GET /stats` - Database statistics

### Speaker Management
- `POST /speakers/register` - Register new speaker with audio
- `GET /speakers` - List all speakers
- `GET /speakers/{id}` - Get speaker info
- `PATCH /speakers/{id}` - Update speaker name
- `DELETE /speakers/{id}` - Delete speaker
- `POST /speakers/add-sample/{id}` - Add voice sample
- `GET /speakers/{id}/samples` - List samples
- `DELETE /speakers/{id}/samples/{sample_id}` - Delete sample

### Diarization & Transcription
- `POST /diarize` - Speaker diarization only
- `POST /identify` - Diarize + identify speakers
- `POST /transcribe-diarized` - Transcribe with diarization
- `POST /transcribe-identified` - Full pipeline (transcribe + diarize + identify)

## Key Services

### DiarizationService (`api/services/diarization.py`)
- Runs pyannote pipeline on audio
- Returns segments with speaker labels and timestamps
- Supports exclusive mode (non-overlapping segments)

### EmbeddingService (`api/services/embedding.py`)
- Extracts 256-dim speaker embeddings
- Methods for whole file, segments, or sliding window
- Cosine similarity computation

### SpeakerDBService (`api/services/speaker_db.py`)
- Manages Qdrant vector storage
- Speaker registration/deletion
- Voting-based speaker identification
- Sample management

### TranscriptMerger (`api/services/transcript_merger.py`)
- Aligns Whisper transcription with diarization
- Word-level timestamp alignment
- Groups words into speaker turns

## Environment Variables

**Required:**
- `HUGGINGFACE_TOKEN` - For model downloads

**Optional:**
- `DEVICE` - "auto", "cuda", or "cpu" (default: auto)
- `SIMILARITY_THRESHOLD` - Speaker matching threshold (default: 0.7)
- `QDRANT_HOST` / `QDRANT_PORT` - Vector DB connection
- `WHISPER_API_URL` / `WHISPER_API_KEY` - Whisper API

## Development

### Running with Docker (recommended)
```bash
# GPU
docker compose up -d

# CPU only
docker compose -f docker-compose.cpu.yml up -d
```

### Building
```bash
# Rebuild API
docker compose build api

# Rebuild UI
docker compose build ui
```

### Ports
- **API**: 8000 (internal), mapped to 8008 (external)
- **UI**: 5173
- **Qdrant**: 6333

## Data Storage

- `/data/models/` - Cached ML models (~1GB)
- `/data/uploads/` - Temporary audio uploads
- `/data/qdrant/` - Vector database persistence

## Important Notes

1. **First request is slow** - Models load lazily on first use (~30s)
2. **GPU recommended** - 10x faster than CPU
3. **Audio formats**: WAV, MP3, FLAC, OGG, M4A, WEBM
4. **Max upload**: 500MB
5. **Embedding dimension**: 256 (cosine similarity)
6. **Speaker samples**: 3-5 samples of 10-30s recommended per speaker

## Common Tasks

### Adding a new API endpoint
1. Add Pydantic schema in `api/api_models/schemas.py`
2. Export in `api/api_models/__init__.py`
3. Add endpoint in `api/main.py`
4. Add API function in `ui/src/lib/api.ts`
5. Update UI component as needed

### Adding a new UI component
1. Create component in `ui/src/components/`
2. Use existing UI primitives from `ui/src/components/ui/`
3. Import API functions from `ui/src/lib/api.ts`

### Modifying speaker database logic
- All Qdrant operations in `api/services/speaker_db.py`
- Collection: `speakers_resnet221` (set via `COLLECTION_NAME` in docker-compose.yml). Older collections `speaker_embeddings` / `work_speaker_embeddings` hold ResNet34-space vectors and are unused — do not point the api at them with the ResNet221 embedding model.
- Re-enrollment after wiping: `python scripts/enroll_reference_speakers.py`
- Payload fields: `speaker_id`, `speaker_name`, `created_at`, `audio_source`
