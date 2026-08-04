# Speaker Diarization

A complete solution for **speaker diarization**, **speaker identification**, and **transcription**. Identify who said what in your audio files with confidence.

## 🎯 What It Does

Upload an audio file and get back:
- **Who spoke** - Speaker labels or identified names
- **When they spoke** - Timestamps for each speaker turn
- **What they said** - Full transcription with speaker attribution

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Speaker Diarization** | Detect who spoke when in any audio file |
| 🔍 **Speaker Identification** | Match voices to registered speaker profiles |
| 📝 **Transcription** | Full speech-to-text with speaker labels |
| 👥 **Speaker Database** | Register and manage speaker voice profiles |
| 🖥️ **Modern Web UI** | Beautiful interface for all operations |
| 🐳 **Docker Ready** | One-command deployment with compose |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Speaker Diarization                       │
├─────────────────┬─────────────────────┬─────────────────────┤
│     Web UI      │     REST API        │     Services        │
│   (React/TS)    │    (FastAPI)        │                     │
├─────────────────┼─────────────────────┼─────────────────────┤
│ • Transcribe    │ • /diarize          │ • pyannote.audio    │
│ • Speakers      │ • /transcribe-*     │ • wespeaker         │
│ • Settings      │ • /speakers/*       │ • Qdrant            │
│                 │ • /identify         │ • Whisper API       │
└─────────────────┴─────────────────────┴─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- [HuggingFace Account](https://huggingface.co/) with accepted model agreements

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your HuggingFace token
```

### 2. Start Services

```bash
# CPU (macOS/ARM)
docker compose -f docker-compose.cpu.yml up -d

# GPU (NVIDIA)
docker compose up -d
```

### 3. Access

| Service | URL |
|---------|-----|
| **Web UI** | http://localhost:5173 |
| **API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |

## 📁 Project Structure

```
speaker_diarization/
├── api/                    # REST API (FastAPI + Python)
│   ├── main.py             # API endpoints
│   ├── services/           # ML services
│   └── README.md           # API documentation
│
├── ui/                     # Web UI (React + TypeScript)
│   ├── src/                # React components
│   └── README.md           # UI documentation
│
├── pipeline/               # Batch meeting-processing scripts (see PLAN.md)
│
├── data/                   # Persistent storage
│   ├── models/             # Cached ML models
│   ├── qdrant/             # Vector database
│   └── uploads/            # Temporary uploads
│
├── docker-compose.yml      # GPU deployment
└── docker-compose.cpu.yml  # CPU deployment
```

## 📖 Documentation

| Component | Description |
|-----------|-------------|
| [**API Documentation**](./api/README.md) | REST endpoints, configuration, running locally |
| [**UI Documentation**](./ui/README.md) | Features, usage guide, development |

## 🛠️ Tech Stack

### API
- **pyannote.audio 4.0** - State-of-the-art speaker diarization (`community-1` model on GPU, `3.1` on the CPU compose)
- **wespeaker** - Speaker embedding extraction
- **Qdrant** - Vector database for speaker matching
- **Whisper** - Speech-to-text transcription
- **FastAPI** - High-performance REST API

### UI
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe development
- **TailwindCSS 4** - Utility-first styling
- **Vite** - Lightning-fast builds

## 💡 Usage Examples

### Register a Speaker
```bash
curl -X POST "http://localhost:8000/speakers/register" \
  -F "file=@john_speaking.mp3" \
  -F "speaker_name=John"
```

### Transcribe with Speaker Identification
```bash
curl -X POST "http://localhost:8000/transcribe-identified" \
  -F "file=@meeting.mp3" \
  -F "num_speakers=2"
```

### Response
```json
{
  "text": "Hey everyone, welcome to the show...",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "identified_as": "John",
      "confidence": 0.85,
      "start": 0.0,
      "end": 3.2,
      "text": "Hey everyone, welcome to the show."
    }
  ],
  "speaker_mapping": {"SPEAKER_00": "John"},
  "num_speakers": 2,
  "num_identified": 1
}
```

## ⚙️ Configuration

Key environment variables:

| Variable | Description |
|----------|-------------|
| `HUGGINGFACE_TOKEN` | **Required** - HuggingFace API token |
| `WHISPER_API_URL` | Whisper API endpoint |
| `SIMILARITY_THRESHOLD` | Speaker matching threshold (default: 0.7) |

See [API README](./api/README.md) for full configuration options.

## 📋 Model Licenses

Accept these HuggingFace model agreements before running:
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (CPU compose)
- [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

## 🤝 Contributing

Contributions welcome! See the component READMEs for development setup.

## 📄 License

This project uses:
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [Qdrant](https://github.com/qdrant/qdrant) - Apache 2.0 License
