// API client for Speaker Diarization API

const API_BASE = '/api'

// Optional shared secret, matching the server's API_KEY setting. Stored in
// localStorage (set via the Settings tab); sent as X-API-Key when present.
const API_KEY_STORAGE = 'diarization-api-key'

export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE) || ''
  } catch {
    return ''
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) {
      localStorage.setItem(API_KEY_STORAGE, key)
    } else {
      localStorage.removeItem(API_KEY_STORAGE)
    }
  } catch {
    // storage unavailable (private mode etc.) — key just won't persist
  }
}

function authHeaders(): Record<string, string> {
  const key = getApiKey()
  return key ? { 'X-API-Key': key } : {}
}

// Types matching the backend schemas
export interface Speaker {
  speaker_id: string
  speaker_name: string
  embeddings_count: number
  created_at: string
}

export interface SpeakerListResponse {
  speakers: Speaker[]
  total_count: number
}

export interface RegisterSpeakerResponse {
  speaker_id: string
  speaker_name: string
  embeddings_count: number
  message: string
}

export interface SpeakerSample {
  sample_id: string
  audio_source: string
  created_at: string
}

export interface SpeakerSamplesResponse {
  speaker_id: string
  speaker_name: string
  samples: SpeakerSample[]
  total_count: number
}

export interface TranscriptSegment {
  speaker: string
  identified_as: string | null
  confidence: number | null
  start: number
  end: number
  duration: number
  text: string
}

export interface TranscriptionResult {
  text: string
  segments: TranscriptSegment[]
  speaker_mapping: Record<string, string | null>
  num_speakers: number
  num_identified: number
  duration: number
  language: string | null
  processing_time: number
}

export interface HealthResponse {
  status: string
  version: string
  models_loaded: boolean
  qdrant_connected: boolean
  device: string
  gpu_memory_used_mb: number | null
  gpu_memory_total_mb: number | null
  jobs_queued: number | null
  jobs_running: number | null
}

export interface StatsResponse {
  database: {
    collection_name?: string
    points_count?: number
    status?: string
    error?: string
  }
  speakers: {
    total_count: number
    total_embeddings: number
  }
  system: {
    device: string
    diarization_model: string
    embedding_model: string
    gpu_memory_used_mb: number | null
    gpu_memory_total_mb: number | null
  }
  jobs?: {
    queued: number
    running: number
  }
}

class ApiError extends Error {
  status: number
  
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }))
    // FastAPI validation errors put an array of objects in `detail`
    let message = error.detail || error.message || 'Request failed'
    if (typeof message !== 'string') {
      message = JSON.stringify(message)
    }
    throw new ApiError(response.status, message)
  }
  return response.json()
}

// Health & Stats
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { headers: authHeaders() })
  return handleResponse(response)
}

export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE}/stats`, { headers: authHeaders() })
  return handleResponse(response)
}

// Speaker Management
export async function getSpeakers(): Promise<SpeakerListResponse> {
  const response = await fetch(`${API_BASE}/speakers`, { headers: authHeaders() })
  return handleResponse(response)
}

export async function registerSpeaker(name: string, audioFile: File): Promise<RegisterSpeakerResponse> {
  const formData = new FormData()
  formData.append('speaker_name', name)
  formData.append('file', audioFile)
  
  const response = await fetch(`${API_BASE}/speakers/register`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })
  return handleResponse(response)
}

export async function addSpeakerSample(speakerId: string, audioFile: File): Promise<RegisterSpeakerResponse> {
  const formData = new FormData()
  formData.append('file', audioFile)
  
  const response = await fetch(`${API_BASE}/speakers/add-sample/${speakerId}`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  })
  return handleResponse(response)
}

export async function deleteSpeaker(speakerId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/speakers/${speakerId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || error.message || 'Delete failed')
  }
}

export async function updateSpeakerName(speakerId: string, speakerName: string): Promise<Speaker> {
  const response = await fetch(`${API_BASE}/speakers/${speakerId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify({ speaker_name: speakerName }),
  })
  return handleResponse(response)
}

export async function getSpeakerSamples(speakerId: string): Promise<SpeakerSamplesResponse> {
  const response = await fetch(`${API_BASE}/speakers/${speakerId}/samples`, { headers: authHeaders() })
  return handleResponse(response)
}

export async function deleteSpeakerSample(speakerId: string, sampleId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/speakers/${speakerId}/samples/${sampleId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || error.message || 'Delete failed')
  }
}

// Transcription
export async function transcribeIdentified(
  audioFile: File,
  numSpeakers?: number,
  signal?: AbortSignal
): Promise<TranscriptionResult> {
  const formData = new FormData()
  formData.append('file', audioFile)
  if (numSpeakers !== undefined) {
    formData.append('num_speakers', numSpeakers.toString())
  }

  const response = await fetch(`${API_BASE}/transcribe-identified`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
    signal,
  })
  return handleResponse(response)
}

