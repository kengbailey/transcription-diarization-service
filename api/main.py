"""FastAPI application for speaker diarization with speaker recognition."""

import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Settings, get_settings
from api_models import (
    DiarizationResult,
    ErrorResponse,
    HealthResponse,
    IdentifiedSegment,
    IdentifyResult,
    RegisterSpeakerResponse,
    Speaker,
    SpeakerListResponse,
    SpeakerSegment,
    TranscriptSegment,
    TranscriptionResult,
    TranscriptionIdentifiedResult,
)
from services import DiarizationService, EmbeddingService, SpeakerDBService, WhisperService, TranscriptMerger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global service instances
settings: Settings = None
diarization_service: DiarizationService = None
embedding_service: EmbeddingService = None
speaker_db_service: SpeakerDBService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global settings, diarization_service, embedding_service, speaker_db_service
    
    logger.info("Starting speaker diarization API...")
    
    # Load settings
    settings = get_settings()
    
    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize services
    diarization_service = DiarizationService(settings)
    embedding_service = EmbeddingService(settings)
    speaker_db_service = SpeakerDBService(settings)
    
    # Pre-initialize models (optional, can be done lazily)
    try:
        logger.info("Pre-loading models...")
        diarization_service.initialize()
        embedding_service.initialize()
        speaker_db_service.initialize()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.warning(f"Delayed model loading due to: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down speaker diarization API...")


# Create FastAPI app
app = FastAPI(
    title="Speaker Diarization API",
    description="API for speaker diarization using pyannote community-1 model with speaker recognition via Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )


async def save_upload_file(file: UploadFile) -> str:
    """Save uploaded file to temporary directory."""
    ext = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{ext}"
    filepath = Path(settings.upload_dir) / filename
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return str(filepath)


def cleanup_file(filepath: str) -> None:
    """Remove temporary file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {e}")


# ============== Health Endpoints ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API and its dependencies."""
    models_loaded = (
        diarization_service.is_initialized and 
        embedding_service.is_initialized
    )
    qdrant_connected = speaker_db_service.is_connected()
    
    device = diarization_service.get_device() if diarization_service else "not initialized"
    
    status = "healthy" if (models_loaded and qdrant_connected) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        models_loaded=models_loaded,
        qdrant_connected=qdrant_connected,
        device=device
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Speaker Diarization API",
        "version": "1.0.0",
        "description": "Speaker diarization using pyannote community-1 with speaker recognition",
        "docs_url": "/docs",
        "health_url": "/health"
    }


# ============== Diarization Endpoints ==============

@app.post("/diarize", response_model=DiarizationResult, tags=["Diarization"])
async def diarize_audio(
    file: UploadFile = File(..., description="Audio file to diarize"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers (if known)"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    exclusive: bool = Form(False, description="Return exclusive diarization (no overlapping segments)")
):
    """
    Perform speaker diarization on an uploaded audio file.
    
    Returns segments with speaker labels and timing information.
    
    - **file**: Audio file (WAV, MP3, FLAC, OGG, M4A, WEBM)
    - **num_speakers**: Optional exact number of speakers if known
    - **min_speakers**: Optional minimum number of speakers
    - **max_speakers**: Optional maximum number of speakers  
    - **exclusive**: If true, returns non-overlapping segments (useful for transcript alignment)
    """
    validate_audio_file(file)
    filepath = None
    
    try:
        # Save uploaded file
        filepath = await save_upload_file(file)
        
        # Run diarization
        result = diarization_service.diarize(
            audio_path=filepath,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=exclusive
        )
        
        # Convert to response model
        segments = [SpeakerSegment(**seg) for seg in result["segments"]]
        
        return DiarizationResult(
            segments=segments,
            num_speakers=result["num_speakers"],
            audio_duration=result["audio_duration"],
            processing_time=result["processing_time"],
            exclusive=result["exclusive"]
        )
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if filepath:
            cleanup_file(filepath)


# ============== Speaker Registration Endpoints ==============

@app.post("/speakers/register", response_model=RegisterSpeakerResponse, tags=["Speaker Recognition"])
async def register_speaker(
    file: UploadFile = File(..., description="Audio file containing the speaker's voice"),
    speaker_name: str = Form(..., description="Name/identifier for the speaker"),
    extract_segments: bool = Form(False, description="Extract embeddings from multiple segments")
):
    """
    Register a new speaker with audio sample(s) for later identification.
    
    The audio file should contain speech from the speaker you want to register.
    For best results, provide at least 10-30 seconds of clear speech.
    
    - **file**: Audio file containing the speaker's voice
    - **speaker_name**: Name or identifier for this speaker
    - **extract_segments**: If true, performs diarization and extracts multiple embeddings
    """
    validate_audio_file(file)
    filepath = None
    
    try:
        # Save uploaded file
        filepath = await save_upload_file(file)
        
        if extract_segments:
            # Run diarization to find speech segments
            diarization_result = diarization_service.diarize(
                audio_path=filepath,
                num_speakers=1  # Assume single speaker for registration
            )
            
            # Extract embeddings from each segment
            segment_embeddings = embedding_service.extract_embeddings_for_segments(
                audio_path=filepath,
                segments=diarization_result["segments"],
                min_duration=1.0  # Minimum 1 second for good embedding
            )
            
            if not segment_embeddings:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract any valid speech segments from the audio"
                )
            
            # Add all embeddings to database
            embeddings = [emb for _, emb in segment_embeddings]
            speaker_id = speaker_db_service.add_speaker_embeddings_batch(
                speaker_name=speaker_name,
                embeddings=embeddings,
                audio_source=file.filename
            )
            
            embeddings_count = len(embeddings)
        else:
            # Extract single embedding from whole file
            embedding = embedding_service.extract_embedding(filepath)
            
            speaker_id = speaker_db_service.add_speaker_embedding(
                speaker_name=speaker_name,
                embedding=embedding,
                audio_source=file.filename
            )
            
            embeddings_count = 1
        
        return RegisterSpeakerResponse(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            embeddings_count=embeddings_count,
            message=f"Speaker registered successfully with {embeddings_count} embedding(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speaker registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if filepath:
            cleanup_file(filepath)


@app.get("/speakers", response_model=SpeakerListResponse, tags=["Speaker Recognition"])
async def list_speakers():
    """
    List all registered speakers in the database.
    
    Returns speaker information including number of stored embeddings.
    """
    try:
        speakers_data = speaker_db_service.get_all_speakers()
        
        speakers = []
        for s in speakers_data:
            speakers.append(Speaker(
                speaker_id=s["speaker_id"],
                speaker_name=s["speaker_name"],
                embeddings_count=s["embeddings_count"],
                created_at=datetime.fromisoformat(s["created_at"]) if s.get("created_at") else datetime.utcnow()
            ))
        
        return SpeakerListResponse(
            speakers=speakers,
            total_count=len(speakers)
        )
        
    except Exception as e:
        logger.error(f"Failed to list speakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers/{speaker_id}", response_model=Speaker, tags=["Speaker Recognition"])
async def get_speaker(speaker_id: str):
    """
    Get information about a specific registered speaker.
    """
    try:
        speaker_data = speaker_db_service.get_speaker_by_id(speaker_id)
        
        if not speaker_data:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        return Speaker(
            speaker_id=speaker_data["speaker_id"],
            speaker_name=speaker_data["speaker_name"],
            embeddings_count=speaker_data["embeddings_count"],
            created_at=datetime.fromisoformat(speaker_data["created_at"]) if speaker_data.get("created_at") else datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/speakers/{speaker_id}", tags=["Speaker Recognition"])
async def delete_speaker(speaker_id: str):
    """
    Delete a registered speaker and all their embeddings.
    """
    try:
        deleted = speaker_db_service.delete_speaker(speaker_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        return {"message": f"Speaker {speaker_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Speaker Identification Endpoints ==============

@app.post("/identify", response_model=IdentifyResult, tags=["Speaker Recognition"])
async def identify_speakers(
    file: UploadFile = File(..., description="Audio file to diarize and identify"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers (if known)"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    similarity_threshold: Optional[float] = Form(None, description="Minimum similarity for speaker matching (0-1)")
):
    """
    Perform speaker diarization and identify speakers against registered voices.
    
    This combines diarization with speaker recognition to label detected speakers
    with their registered names when possible.
    
    - **file**: Audio file to process
    - **num_speakers**: Optional exact number of speakers
    - **min_speakers/max_speakers**: Optional speaker count bounds
    - **similarity_threshold**: Minimum similarity score for matching (default: 0.7)
    """
    validate_audio_file(file)
    filepath = None
    
    try:
        start_time = time.time()
        
        # Save uploaded file
        filepath = await save_upload_file(file)
        
        # Run diarization
        diarization_result = diarization_service.diarize(
            audio_path=filepath,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=True  # Use exclusive for cleaner speaker identification
        )
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in diarization_result["segments"]:
            speaker = segment["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # Identify each speaker
        speaker_mapping = {}
        speaker_confidences = {}
        
        for speaker, segments in speaker_segments.items():
            # Extract embeddings for this speaker's segments
            segment_embeddings = embedding_service.extract_embeddings_for_segments(
                audio_path=filepath,
                segments=segments,
                min_duration=0.5
            )
            
            if segment_embeddings:
                embeddings = [emb for _, emb in segment_embeddings]
                
                # Try to identify using voting
                match = speaker_db_service.identify_speaker_by_voting(
                    embeddings=embeddings,
                    score_threshold=similarity_threshold
                )
                
                if match:
                    speaker_mapping[speaker] = match["speaker_name"]
                    speaker_confidences[speaker] = match["score"]
                else:
                    speaker_mapping[speaker] = None
                    speaker_confidences[speaker] = None
            else:
                speaker_mapping[speaker] = None
                speaker_confidences[speaker] = None
        
        # Build result segments
        identified_segments = []
        for segment in diarization_result["segments"]:
            speaker = segment["speaker"]
            identified_segments.append(IdentifiedSegment(
                speaker=speaker,
                identified_as=speaker_mapping.get(speaker),
                confidence=speaker_confidences.get(speaker),
                start=segment["start"],
                end=segment["end"],
                duration=segment["duration"]
            ))
        
        processing_time = time.time() - start_time
        num_identified = sum(1 for v in speaker_mapping.values() if v is not None)
        
        return IdentifyResult(
            segments=identified_segments,
            speaker_mapping=speaker_mapping,
            num_speakers=diarization_result["num_speakers"],
            num_identified=num_identified,
            audio_duration=diarization_result["audio_duration"],
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Speaker identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if filepath:
            cleanup_file(filepath)


@app.post("/speakers/add-sample/{speaker_id}", response_model=RegisterSpeakerResponse, tags=["Speaker Recognition"])
async def add_speaker_sample(
    speaker_id: str,
    file: UploadFile = File(..., description="Additional audio sample for the speaker"),
    extract_segments: bool = Form(False, description="Extract embeddings from multiple segments")
):
    """
    Add additional audio sample(s) for an existing registered speaker.
    
    More samples improve speaker recognition accuracy.
    """
    validate_audio_file(file)
    filepath = None
    
    try:
        # Check if speaker exists
        speaker = speaker_db_service.get_speaker_by_id(speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
        # Save uploaded file
        filepath = await save_upload_file(file)
        
        if extract_segments:
            # Run diarization to find speech segments
            diarization_result = diarization_service.diarize(
                audio_path=filepath,
                num_speakers=1
            )
            
            segment_embeddings = embedding_service.extract_embeddings_for_segments(
                audio_path=filepath,
                segments=diarization_result["segments"],
                min_duration=1.0
            )
            
            if not segment_embeddings:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract any valid speech segments"
                )
            
            embeddings = [emb for _, emb in segment_embeddings]
            speaker_db_service.add_speaker_embeddings_batch(
                speaker_name=speaker["speaker_name"],
                embeddings=embeddings,
                speaker_id=speaker_id,
                audio_source=file.filename
            )
            
            new_embeddings = len(embeddings)
        else:
            embedding = embedding_service.extract_embedding(filepath)
            
            speaker_db_service.add_speaker_embedding(
                speaker_name=speaker["speaker_name"],
                embedding=embedding,
                speaker_id=speaker_id,
                audio_source=file.filename
            )
            
            new_embeddings = 1
        
        # Get updated speaker info
        updated_speaker = speaker_db_service.get_speaker_by_id(speaker_id)
        
        return RegisterSpeakerResponse(
            speaker_id=speaker_id,
            speaker_name=updated_speaker["speaker_name"],
            embeddings_count=updated_speaker["embeddings_count"],
            message=f"Added {new_embeddings} new embedding(s) to speaker"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add speaker sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if filepath:
            cleanup_file(filepath)


# ============== Transcription Endpoints ==============

@app.post("/transcribe-diarized", response_model=TranscriptionResult, tags=["Transcription"])
async def transcribe_diarized(
    file: UploadFile = File(..., description="Audio file to transcribe and diarize"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers (if known)"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en'). Auto-detect if not specified")
):
    """
    Transcribe audio with speaker diarization.
    
    Combines Whisper transcription with pyannote diarization to produce
    speaker-attributed transcripts.
    
    - **file**: Audio file to process
    - **num_speakers**: Optional exact number of speakers if known
    - **language**: Optional language code (auto-detect if not specified)
    """
    validate_audio_file(file)
    filepath = None
    
    try:
        start_time = time.time()
        
        # Save uploaded file
        filepath = await save_upload_file(file)
        
        # Initialize services (lazily)
        whisper_service = WhisperService(settings)
        whisper_service.initialize()
        merger = TranscriptMerger()
        
        # Run diarization and transcription in parallel conceptually
        # (but sequentially here for simplicity)
        
        # 1. Run diarization
        logger.info("Running diarization...")
        diarization_result = diarization_service.diarize(
            audio_path=filepath,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=True
        )
        
        # 2. Run Whisper transcription
        logger.info("Running Whisper transcription...")
        whisper_result = whisper_service.transcribe_with_words(
            audio_path=filepath,
            language=language
        )
        
        # 3. Merge results
        logger.info("Merging transcription with diarization...")
        merged = merger.merge_transcription_with_diarization(
            whisper_result=whisper_result,
            diarization_result=diarization_result
        )
        
        processing_time = time.time() - start_time
        
        # Build response segments
        segments = [
            TranscriptSegment(
                speaker=seg["speaker"],
                identified_as=seg.get("identified_as"),
                confidence=seg.get("confidence"),
                start=seg["start"],
                end=seg["end"],
                duration=seg.get("duration", seg["end"] - seg["start"]),
                text=seg["text"]
            )
            for seg in merged["segments"]
        ]
        
        return TranscriptionResult(
            text=merged["text"],
            segments=segments,
            num_speakers=merged["num_speakers"],
            duration=merged["duration"],
            language=merged.get("language"),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Transcription with diarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if filepath:
            cleanup_file(filepath)


@app.post("/transcribe-identified", response_model=TranscriptionIdentifiedResult, tags=["Transcription"])
async def transcribe_identified(
    file: UploadFile = File(..., description="Audio file to transcribe, diarize, and identify"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers (if known)"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en'). Auto-detect if not specified"),
    similarity_threshold: Optional[float] = Form(None, description="Minimum similarity for speaker matching (0-1)")
):
    """
    Transcribe audio with speaker diarization and identification.
    
    Combines Whisper transcription with pyannote diarization and matches
    speakers against registered voices in Qdrant.
    
    - **file**: Audio file to process
    - **num_speakers**: Optional exact number of speakers if known
    - **language**: Optional language code (auto-detect if not specified)
    - **similarity_threshold**: Minimum similarity for speaker matching (default: 0.7)
    """
    validate_audio_file(file)
    filepath = None
    
    try:
        start_time = time.time()
        
        # Save uploaded file
        filepath = await save_upload_file(file)
        
        # Initialize services (lazily)
        whisper_service = WhisperService(settings)
        whisper_service.initialize()
        merger = TranscriptMerger()
        
        # 1. Run diarization
        logger.info("Running diarization...")
        diarization_result = diarization_service.diarize(
            audio_path=filepath,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=True
        )
        
        # 2. Identify speakers
        logger.info("Identifying speakers...")
        speaker_segments = {}
        for segment in diarization_result["segments"]:
            speaker = segment["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        speaker_mapping = {}
        speaker_confidences = {}
        
        for speaker, segments in speaker_segments.items():
            segment_embeddings = embedding_service.extract_embeddings_for_segments(
                audio_path=filepath,
                segments=segments,
                min_duration=0.5
            )
            
            if segment_embeddings:
                embeddings = [emb for _, emb in segment_embeddings]
                match = speaker_db_service.identify_speaker_by_voting(
                    embeddings=embeddings,
                    score_threshold=similarity_threshold
                )
                
                if match:
                    speaker_mapping[speaker] = match["speaker_name"]
                    speaker_confidences[speaker] = match["score"]
                else:
                    speaker_mapping[speaker] = None
                    speaker_confidences[speaker] = None
            else:
                speaker_mapping[speaker] = None
                speaker_confidences[speaker] = None
        
        # 3. Run Whisper transcription
        logger.info("Running Whisper transcription...")
        whisper_result = whisper_service.transcribe_with_words(
            audio_path=filepath,
            language=language
        )
        
        # 4. Merge results with speaker identification
        logger.info("Merging transcription with diarization and identification...")
        merged = merger.merge_transcription_with_diarization(
            whisper_result=whisper_result,
            diarization_result=diarization_result,
            speaker_mapping=speaker_mapping,
            speaker_confidences=speaker_confidences
        )
        
        processing_time = time.time() - start_time
        
        # Build response segments
        segments = [
            TranscriptSegment(
                speaker=seg["speaker"],
                identified_as=seg.get("identified_as"),
                confidence=seg.get("confidence"),
                start=seg["start"],
                end=seg["end"],
                duration=seg.get("duration", seg["end"] - seg["start"]),
                text=seg["text"]
            )
            for seg in merged["segments"]
        ]
        
        num_identified = sum(1 for v in speaker_mapping.values() if v is not None)
        
        return TranscriptionIdentifiedResult(
            text=merged["text"],
            segments=segments,
            speaker_mapping=speaker_mapping,
            num_speakers=merged["num_speakers"],
            num_identified=num_identified,
            duration=merged["duration"],
            language=merged.get("language"),
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Transcription with identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if filepath:
            cleanup_file(filepath)


# ============== Statistics Endpoints ==============

@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get statistics about the speaker database and system."""
    try:
        collection_stats = speaker_db_service.get_collection_stats()
        speakers = speaker_db_service.get_all_speakers()
        
        return {
            "database": collection_stats,
            "speakers": {
                "total_count": len(speakers),
                "total_embeddings": sum(s["embeddings_count"] for s in speakers)
            },
            "system": {
                "device": diarization_service.get_device(),
                "diarization_model": settings.diarization_model,
                "embedding_model": settings.embedding_model
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            detail=None
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc)
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
