"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class SpeakerSegment(BaseModel):
    """A single speaker segment from diarization."""
    
    speaker: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00')")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "SPEAKER_00",
                "start": 0.5,
                "end": 3.2,
                "duration": 2.7
            }
        }


class DiarizationResult(BaseModel):
    """Result of speaker diarization."""
    
    segments: list[SpeakerSegment] = Field(..., description="List of speaker segments")
    num_speakers: int = Field(..., description="Number of detected speakers")
    audio_duration: float = Field(..., description="Total audio duration in seconds")
    processing_time: float = Field(..., description="Processing time in seconds")
    exclusive: bool = Field(default=False, description="Whether exclusive diarization was used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "segments": [
                    {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2, "duration": 2.7},
                    {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.1, "duration": 3.6}
                ],
                "num_speakers": 2,
                "audio_duration": 10.5,
                "processing_time": 2.3,
                "exclusive": False
            }
        }


class RegisterSpeakerRequest(BaseModel):
    """Request to register a new speaker."""
    
    speaker_name: str = Field(..., description="Name/identifier for the speaker", min_length=1, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker_name": "John Doe"
            }
        }


class RegisterSpeakerResponse(BaseModel):
    """Response after registering a speaker."""
    
    speaker_id: str = Field(..., description="Unique ID for the registered speaker")
    speaker_name: str = Field(..., description="Name of the registered speaker")
    embeddings_count: int = Field(..., description="Number of embeddings stored for this speaker")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker_id": "abc123",
                "speaker_name": "John Doe",
                "embeddings_count": 3,
                "message": "Speaker registered successfully with 3 embeddings"
            }
        }


class Speaker(BaseModel):
    """Information about a registered speaker."""
    
    speaker_id: str = Field(..., description="Unique ID for the speaker")
    speaker_name: str = Field(..., description="Name of the speaker")
    embeddings_count: int = Field(..., description="Number of embeddings stored")
    created_at: datetime = Field(..., description="When the speaker was first registered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker_id": "abc123",
                "speaker_name": "John Doe",
                "embeddings_count": 5,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class SpeakerListResponse(BaseModel):
    """Response listing all registered speakers."""
    
    speakers: list[Speaker] = Field(..., description="List of registered speakers")
    total_count: int = Field(..., description="Total number of registered speakers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "speakers": [
                    {
                        "speaker_id": "abc123",
                        "speaker_name": "John Doe",
                        "embeddings_count": 5,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                ],
                "total_count": 1
            }
        }


class IdentifiedSegment(BaseModel):
    """A speaker segment with identification information."""
    
    speaker: str = Field(..., description="Original speaker label from diarization")
    identified_as: Optional[str] = Field(None, description="Matched speaker name from database")
    confidence: Optional[float] = Field(None, description="Confidence score of the match (0-1)")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "SPEAKER_00",
                "identified_as": "John Doe",
                "confidence": 0.92,
                "start": 0.5,
                "end": 3.2,
                "duration": 2.7
            }
        }


class IdentifyResult(BaseModel):
    """Result of speaker diarization with identification."""
    
    segments: list[IdentifiedSegment] = Field(..., description="List of identified speaker segments")
    speaker_mapping: dict[str, Optional[str]] = Field(
        ..., 
        description="Mapping from diarization labels to identified names"
    )
    num_speakers: int = Field(..., description="Number of detected speakers")
    num_identified: int = Field(..., description="Number of speakers matched to known identities")
    audio_duration: float = Field(..., description="Total audio duration in seconds")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "identified_as": "John Doe",
                        "confidence": 0.92,
                        "start": 0.5,
                        "end": 3.2,
                        "duration": 2.7
                    }
                ],
                "speaker_mapping": {"SPEAKER_00": "John Doe", "SPEAKER_01": None},
                "num_speakers": 2,
                "num_identified": 1,
                "audio_duration": 10.5,
                "processing_time": 3.1
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    qdrant_connected: bool = Field(..., description="Whether Qdrant is connected")
    device: str = Field(..., description="Compute device being used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": True,
                "qdrant_connected": True,
                "device": "cuda"
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid audio file format",
                "detail": "Supported formats: WAV, MP3, FLAC, OGG"
            }
        }


# ============== Transcription Schemas ==============

class TranscriptSegment(BaseModel):
    """A transcript segment with speaker attribution."""
    
    speaker: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00')")
    identified_as: Optional[str] = Field(None, description="Matched speaker name from database")
    confidence: Optional[float] = Field(None, description="Speaker identification confidence (0-1)")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "SPEAKER_00",
                "identified_as": "MKBHD",
                "confidence": 0.85,
                "start": 0.0,
                "end": 5.2,
                "duration": 5.2,
                "text": "Hey what's up guys, MKBHD here"
            }
        }


class TranscriptionResult(BaseModel):
    """Result of transcription with speaker diarization."""
    
    text: str = Field(..., description="Full transcript text")
    segments: list[TranscriptSegment] = Field(..., description="List of speaker-attributed segments")
    num_speakers: int = Field(..., description="Number of detected speakers")
    duration: float = Field(..., description="Total audio duration in seconds")
    language: Optional[str] = Field(None, description="Detected language code")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hey what's up guys, MKBHD here. Today we're talking about...",
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "identified_as": "MKBHD",
                        "confidence": 0.85,
                        "start": 0.0,
                        "end": 5.2,
                        "duration": 5.2,
                        "text": "Hey what's up guys, MKBHD here"
                    }
                ],
                "num_speakers": 2,
                "duration": 120.5,
                "language": "en",
                "processing_time": 15.3
            }
        }


class TranscriptionIdentifiedResult(BaseModel):
    """Result of transcription with speaker diarization and identification."""
    
    text: str = Field(..., description="Full transcript text")
    segments: list[TranscriptSegment] = Field(..., description="List of identified speaker segments")
    speaker_mapping: dict[str, Optional[str]] = Field(
        ..., 
        description="Mapping from diarization labels to identified names"
    )
    num_speakers: int = Field(..., description="Number of detected speakers")
    num_identified: int = Field(..., description="Number of speakers matched to known identities")
    duration: float = Field(..., description="Total audio duration in seconds")
    language: Optional[str] = Field(None, description="Detected language code")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hey what's up guys, MKBHD here. That's awesome!",
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "identified_as": "MKBHD",
                        "confidence": 0.85,
                        "start": 0.0,
                        "end": 3.2,
                        "duration": 3.2,
                        "text": "Hey what's up guys, MKBHD here"
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "identified_as": "AE",
                        "confidence": 0.78,
                        "start": 3.5,
                        "end": 5.0,
                        "duration": 1.5,
                        "text": "That's awesome!"
                    }
                ],
                "speaker_mapping": {"SPEAKER_00": "MKBHD", "SPEAKER_01": "AE"},
                "num_speakers": 2,
                "num_identified": 2,
                "duration": 120.5,
                "language": "en",
                "processing_time": 18.5
            }
        }
