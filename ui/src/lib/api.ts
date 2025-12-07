// API client for Speaker Diarization API

const API_BASE = '/api'

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
}

export interface StatsResponse {
  database: {
    points_count: number
    vectors_count: number
  }
  speakers: {
    total_count: number
    total_embeddings: number
  }
  system: {
    device: string
    diarization_model: string
    embedding_model: string
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
    throw new ApiError(response.status, error.detail || error.message || 'Request failed')
  }
  return response.json()
}

// Health & Stats
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`)
  return handleResponse(response)
}

export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE}/stats`)
  return handleResponse(response)
}

// Speaker Management
export async function getSpeakers(): Promise<SpeakerListResponse> {
  const response = await fetch(`${API_BASE}/speakers`)
  return handleResponse(response)
}

export async function registerSpeaker(name: string, audioFile: File): Promise<RegisterSpeakerResponse> {
  const formData = new FormData()
  formData.append('speaker_name', name)
  formData.append('file', audioFile)
  
  const response = await fetch(`${API_BASE}/speakers/register`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(response)
}

export async function addSpeakerSample(speakerId: string, audioFile: File): Promise<RegisterSpeakerResponse> {
  const formData = new FormData()
  formData.append('file', audioFile)
  
  const response = await fetch(`${API_BASE}/speakers/add-sample/${speakerId}`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(response)
}

export async function deleteSpeaker(speakerId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/speakers/${speakerId}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || error.message || 'Delete failed')
  }
}

// Transcription
export async function transcribeIdentified(
  audioFile: File,
  numSpeakers?: number
): Promise<TranscriptionResult> {
  const formData = new FormData()
  formData.append('file', audioFile)
  if (numSpeakers !== undefined) {
    formData.append('num_speakers', numSpeakers.toString())
  }
  
  const response = await fetch(`${API_BASE}/transcribe-identified`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(response)
}

export async function transcribeDiarized(
  audioFile: File,
  numSpeakers?: number
): Promise<TranscriptionResult> {
  const formData = new FormData()
  formData.append('file', audioFile)
  if (numSpeakers !== undefined) {
    formData.append('num_speakers', numSpeakers.toString())
  }
  
  const response = await fetch(`${API_BASE}/transcribe-diarized`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(response)
}

export { ApiError }
