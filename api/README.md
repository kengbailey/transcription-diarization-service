# Speaker Diarization API

A production-ready REST API for speaker diarization, speaker identification, and transcription powered by state-of-the-art ML models.

## Core Technology

| Component | Purpose |
|-----------|---------|
| **pyannote.audio 3.1** | Speaker diarization - identifies who spoke when |
| **wespeaker** | Speaker embeddings - creates voice fingerprints |
| **Qdrant** | Vector database - stores and matches speaker profiles |
| **Whisper** | Speech-to-text transcription (external API) |
| **FastAPI** | High-performance REST API framework |

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check with service status |
| `GET` | `/stats` | Database and system statistics |

### Diarization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/diarize` | Analyze audio to identify who spoke when |

### Speaker Recognition

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/speakers/register` | Register a new speaker with voice sample |
| `POST` | `/speakers/add-sample/{id}` | Add more voice samples to existing speaker |
| `GET` | `/speakers` | List all registered speakers |
| `GET` | `/speakers/{id}` | Get specific speaker details |
| `DELETE` | `/speakers/{id}` | Remove a registered speaker |
| `POST` | `/identify` | Diarize and identify speakers against database |

### Transcription

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcribe-diarized` | Transcribe audio with speaker labels |
| `POST` | `/transcribe-identified` | Full pipeline: transcribe + identify speakers |

## Response Examples

### Diarization Result
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

### Transcription with Identification
```json
{
  "text": "Hey what's up guys, MKBHD here. That's awesome!",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "identified_as": "MKBHD",
      "confidence": 0.85,
      "start": 0.0,
      "end": 3.2,
      "text": "Hey what's up guys, MKBHD here"
    },
    {
      "speaker": "SPEAKER_01",
      "identified_as": "Guest",
      "confidence": 0.78,
      "start": 3.5,
      "end": 5.0,
      "text": "That's awesome!"
    }
  ],
  "speaker_mapping": {"SPEAKER_00": "MKBHD", "SPEAKER_01": "Guest"},
  "num_speakers": 2,
  "num_identified": 2
}
```

## Project Structure

```
api/
├── main.py                 # FastAPI application & endpoints
├── config.py               # Settings management
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── api_models/
│   ├── __init__.py
│   └── schemas.py          # Pydantic request/response models
└── services/
    ├── __init__.py
    ├── diarization.py      # pyannote diarization service
    ├── embedding.py        # wespeaker embedding extraction
    ├── speaker_db.py       # Qdrant speaker database
    ├── whisper.py          # Whisper API client
    └── transcript_merger.py # Merge transcription with diarization
```

## Configuration

Environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | **Required** | HuggingFace API token |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model |
| `EMBEDDING_MODEL` | `pyannote/wespeaker-voxceleb-resnet34-LM` | Embedding model |
| `DEVICE` | `auto` | Compute device: `auto`, `cuda`, or `cpu` |
| `QDRANT_HOST` | `qdrant` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `SIMILARITY_THRESHOLD` | `0.7` | Min similarity for speaker matching |
| `WHISPER_API_URL` | - | Whisper API base URL |
| `WHISPER_API_KEY` | - | Whisper API key |
| `WHISPER_MODEL` | - | Whisper model name |

## Running Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (required)
docker run -p 6333:6333 qdrant/qdrant

# Set environment variables
export HUGGINGFACE_TOKEN=hf_your_token_here

# Run the API
uvicorn main:app --reload --port 8000
```

API will be available at **http://localhost:8000**  
Interactive docs at **http://localhost:8000/docs**

## Docker Deployment

See the root directory for `docker-compose.yml` files:

```bash
# CPU deployment
docker compose -f docker-compose.cpu.yml up -d

# GPU deployment
docker compose up -d
```

## Supported Audio Formats

- WAV
- MP3
- FLAC
- OGG
- M4A
- WEBM

## Performance Tips

- **First request is slow** - Models load on first use (~30 seconds)
- **GPU is 10x faster** - Use CUDA if available
- **More samples = better recognition** - Register 3-5 samples per speaker
- **Longer samples work best** - 10-30 seconds of clear speech

## Model Licenses

Before running, accept the HuggingFace model licenses:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
