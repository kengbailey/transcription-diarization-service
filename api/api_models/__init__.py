"""Pydantic models for the speaker diarization API."""

from .schemas import (
    SpeakerSegment,
    DiarizationResult,
    RegisterSpeakerRequest,
    RegisterSpeakerResponse,
    Speaker,
    SpeakerListResponse,
    IdentifyResult,
    IdentifiedSegment,
    HealthResponse,
    ErrorResponse,
    TranscriptSegment,
    TranscriptionResult,
    TranscriptionIdentifiedResult,
)

__all__ = [
    "SpeakerSegment",
    "DiarizationResult",
    "RegisterSpeakerRequest",
    "RegisterSpeakerResponse",
    "Speaker",
    "SpeakerListResponse",
    "IdentifyResult",
    "IdentifiedSegment",
    "HealthResponse",
    "ErrorResponse",
    "TranscriptSegment",
    "TranscriptionResult",
    "TranscriptionIdentifiedResult",
]
