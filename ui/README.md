# Speaker Diarization Studio

A modern, sleek web UI for the Speaker Diarization API. Manage speakers, transcribe audio, and identify who said what—all from your browser.

![Built with React](https://img.shields.io/badge/React-18-blue) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-4-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)

## Features

### 🎤 Transcription
- **Drag & drop audio upload** - Supports MP3, WAV, FLAC, OGG, M4A, WEBM
- **Audio preview player** - Listen before transcribing
- **Speaker configuration** - Auto-detect or specify number of speakers
- **Interactive transcript** - Color-coded by speaker with timestamps
- **Click-to-seek** - Click any segment to jump to that position
- **Visual timeline** - See speaker distribution across the audio
- **Copy transcript** - One-click copy with speaker labels

### 👥 Speaker Management
- **Register new speakers** - Upload voice samples to create speaker profiles
- **Add more samples** - Improve recognition by adding additional voice samples
- **Speaker cards** - View all registered speakers with sample counts
- **Delete speakers** - Remove speakers you no longer need

### ⚙️ Settings & Status
- **System health monitoring** - API status, model loading, Qdrant connection
- **Database statistics** - Total speakers and embeddings count
- **Quick links** - Direct access to API documentation

## Quick Start

### Prerequisites
- Node.js 18+
- Speaker Diarization API running on `localhost:8000`

### Installation

```bash
# Navigate to UI directory
cd ui

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at **http://localhost:5173**

### Production Build

```bash
npm run build
npm run preview
```

## Project Structure

```
ui/
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   └── SettingsTab.tsx       # Settings & status page
│   │   ├── speakers/
│   │   │   └── SpeakersTab.tsx       # Speaker management
│   │   ├── transcription/
│   │   │   └── TranscriptionTab.tsx  # Audio transcription
│   │   └── ui/                       # Reusable UI components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── dialog.tsx
│   │       ├── file-upload.tsx
│   │       ├── input.tsx
│   │       ├── spinner.tsx
│   │       └── tabs.tsx
│   ├── lib/
│   │   ├── api.ts                    # API client
│   │   └── utils.ts                  # Utility functions
│   ├── App.tsx                       # Main application
│   ├── main.tsx                      # Entry point
│   └── index.css                     # Global styles
├── index.html
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## Tech Stack

- **React 18** - UI framework with hooks
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **TailwindCSS 4** - Utility-first CSS
- **Lucide React** - Beautiful icons
- **Custom UI Components** - shadcn-inspired components

## API Integration

The UI proxies requests to the backend API:

| UI Path | Backend |
|---------|---------|
| `/api/*` | `http://localhost:8000/*` |

Ensure the Speaker Diarization API is running before using the UI.

## Usage Guide

### Registering a Speaker

1. Go to the **Speakers** tab
2. Click **Add Speaker**
3. Enter a name for the speaker
4. Upload a clear audio sample (10-30 seconds of speech works best)
5. Click **Add Speaker**

### Adding More Voice Samples

1. Go to the **Speakers** tab
2. Find the speaker card
3. Click **Add Sample**
4. Upload another audio file
5. More samples = better recognition accuracy

### Transcribing Audio

1. Go to the **Transcribe** tab
2. Drag & drop or browse for an audio file
3. (Optional) Specify number of speakers
4. Click **Transcribe Audio**
5. View the color-coded transcript with speaker labels
6. Click any segment to jump to that position in the audio

## Configuration

The Vite config includes a proxy setup for the API:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
},
```

To change the API endpoint, modify the `target` URL.

## Development

```bash
# Start dev server with hot reload
npm run dev

# Type checking
npm run build

# Lint
npm run lint
```

## License

Part of the Speaker Diarization API project.
