# Speaker Diarization API

A production-ready REST API for speaker diarization, speaker identification, and transcription using:

- **[pyannote.audio](https://github.com/pyannote/pyannote-audio)** - State-of-the-art speaker diarization
- **[Qdrant](https://qdrant.tech/)** - Vector database for speaker embeddings
- **[Whisper](https://openai.com/whisper)** - Speech-to-text (via OpenAI-compatible API)

## Features

| Feature | Description |
|---------|-------------|
| 🎯 **Speaker Diarization** | Detect who spoke when in audio files |
| 🔍 **Speaker Identification** | Match detected speakers to registered voices |
| 📝 **Transcription** | Full speech-to-text with speaker attribution |
| 💾 **Speaker Database** | Register and manage speaker voice profiles |
| 🐳 **Docker Ready** | CPU and GPU compose configs included |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- [HuggingFace Account](https://huggingface.co/) with accepted pyannote model agreements
- (Optional) OpenAI-compatible Whisper API for transcription

### 1. Accept Model Licenses

Visit and accept the terms for both models:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your HuggingFace token:
```env
HUGGINGFACE_TOKEN=hf_your_token_here

# Optional: Whisper API settings
WHISPER_API_URL=http://your-whisper-server:8000/v1
WHISPER_API_KEY=your_api_key
WHISPER_MODEL=Systran/faster-distil-whisper-large-v3
```

### 3. Start Services

**For CPU (macOS/ARM):**
```bash
docker compose -f docker-compose.cpu.yml up -d
```

**For GPU (NVIDIA):**
```bash
docker compose up -d
```

### 4. Verify Installation

```bash
curl http://localhost:8000/health
```

API documentation available at: http://localhost:8000/docs

---

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Database & system stats |

### Diarization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/diarize` | Diarize audio (who spoke when) |

```bash
curl -X POST "http://localhost:8000/diarize" \
  -F "file=@audio.mp3" \
  -F "num_speakers=2"
```

**Response:**
```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2, "duration": 2.7},
    {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.1, "duration": 3.6}
  ],
  "num_speakers": 2,
  "audio_duration": 10.5,
  "processing_time": 2.3
}
```

### Speaker Recognition

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/speakers/register` | Register a new speaker |
| `GET` | `/speakers` | List all registered speakers |
| `GET` | `/speakers/{id}` | Get speaker details |
| `DELETE` | `/speakers/{id}` | Delete a speaker |
| `POST` | `/speakers/add-sample/{id}` | Add more voice samples |
| `POST` | `/identify` | Diarize + identify speakers |

**Register a speaker:**
```bash
curl -X POST "http://localhost:8000/speakers/register" \
  -F "file=@john_speaking.mp3" \
  -F "speaker_name=John"
```

**Identify speakers in audio:**
```bash
curl -X POST "http://localhost:8000/identify" \
  -F "file=@meeting.mp3" \
  -F "num_speakers=3"
```

**Response:**
```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "identified_as": "John", "confidence": 0.92, "start": 0.5, "end": 3.2}
  ],
  "speaker_mapping": {"SPEAKER_00": "John", "SPEAKER_01": null},
  "num_speakers": 2,
  "num_identified": 1
}
```

### Transcription (requires Whisper API)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcribe-diarized` | Transcribe + diarize |
| `POST` | `/transcribe-identified` | Transcribe + diarize + identify |

**Full transcription with speaker identification:**
```bash
curl -X POST "http://localhost:8000/transcribe-identified" \
  -F "file=@podcast.mp3" \
  -F "num_speakers=2"
```

**Response:**
```json
{
  "text": "Hey everyone, welcome to the show. Thanks for having me...",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "identified_as": "Host",
      "confidence": 0.85,
      "start": 0.0,
      "end": 3.2,
      "text": "Hey everyone, welcome to the show."
    },
    {
      "speaker": "SPEAKER_01",
      "identified_as": "Guest",
      "confidence": 0.78,
      "start": 3.5,
      "end": 5.0,
      "text": "Thanks for having me."
    }
  ],
  "speaker_mapping": {"SPEAKER_00": "Host", "SPEAKER_01": "Guest"},
  "num_speakers": 2,
  "num_identified": 2,
  "language": "en"
}
```

---

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | - | **Required.** HuggingFace API token |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model |
| `EMBEDDING_MODEL` | `pyannote/wespeaker-voxceleb-resnet34-LM` | Speaker embedding model |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `QDRANT_HOST` | `qdrant` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `SIMILARITY_THRESHOLD` | `0.7` | Min similarity for speaker matching |
| `WHISPER_API_URL` | - | Whisper API base URL |
| `WHISPER_API_KEY` | - | Whisper API key |
| `WHISPER_MODEL` | - | Whisper model name |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Speaker Diarization API                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │   FastAPI   │  │   pyannote  │  │   Whisper Service    │ │
│  │   Server    │──│ Diarization │──│  (External API)      │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│         │                │                    │              │
│         │         ┌──────┴──────┐            │              │
│         │         │   wespeaker │            │              │
│         │         │  Embeddings │            │              │
│         │         └──────┬──────┘            │              │
│         │                │                    │              │
│         │         ┌──────▼──────┐            │              │
│         └────────▶│   Qdrant    │◀───────────┘              │
│                   │   Vector DB │                            │
│                   └─────────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **FastAPI Server** - REST API handling requests
2. **pyannote.audio** - Speaker diarization (who spoke when)
3. **wespeaker** - Speaker embedding extraction (voice fingerprints)
4. **Qdrant** - Vector database for speaker identification
5. **Whisper** - Speech-to-text transcription (optional, external)

---

## Project Structure

```
speaker_diarization/
├── api/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Settings management
│   ├── Dockerfile              # Container definition
│   ├── requirements.txt        # Python dependencies
│   ├── api_models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── diarization.py      # Diarization service
│       ├── embedding.py        # Speaker embedding service
│       ├── speaker_db.py       # Qdrant speaker database
│       ├── whisper.py          # Whisper API client
│       └── transcript_merger.py # Transcript + diarization merger
├── data/
│   ├── models/                 # Cached ML models
│   ├── qdrant/                 # Qdrant data storage
│   └── uploads/                # Temporary upload storage
├── docker-compose.yml          # GPU deployment
├── docker-compose.cpu.yml      # CPU deployment
├── .env.example                # Environment template
└── README.md
```

---

## Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run API
cd api && uvicorn main:app --reload
```

### Running Tests

```bash
# Health check
curl http://localhost:8000/health

# Register a speaker
curl -X POST "http://localhost:8000/speakers/register" \
  -F "file=@test_audio.mp3" \
  -F "speaker_name=TestUser"

# Test diarization
curl -X POST "http://localhost:8000/diarize" \
  -F "file=@test_audio.mp3"
```

---

## Supported Audio Formats

- WAV
- MP3
- FLAC
- OGG
- M4A
- WEBM

---

## Performance Notes

- **First request** will be slower due to model loading (~30s)
- **CPU inference** is slower than GPU (~10x)
- **Speaker registration** benefits from clear, single-speaker audio
- **Transcription** requires external Whisper API

### Recommended Settings

| Audio Type | Suggested Parameters |
|------------|---------------------|
| 2-person interview | `num_speakers=2` |
| Meeting recording | `min_speakers=2, max_speakers=10` |
| Podcast | `num_speakers=2` or `3` |
| Unknown speakers | Leave parameters empty (auto-detect) |

---

## Troubleshooting

### "Token required" error
Make sure you've:
1. Set `HUGGINGFACE_TOKEN` in `.env`
2. Accepted model licenses on HuggingFace

### Slow first request
Models are loaded on first use. Subsequent requests will be faster.

### Low speaker identification confidence
- Register more voice samples with `/speakers/add-sample/{id}`
- Use clear audio with minimal background noise
- Longer audio samples (10-30 seconds) work best

### Whisper API connection errors
- Verify `WHISPER_API_URL` is correct
- Check the Whisper service is running
- Ensure network connectivity between containers

---

## License

This project uses:
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [Qdrant](https://github.com/qdrant/qdrant) - Apache 2.0 License

Note: pyannote models require accepting HuggingFace model agreements.
