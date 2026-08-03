"""FastAPI application for speaker diarization with speaker recognition."""

import asyncio
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
    IdentifyResult,
    RegisterSpeakerResponse,
    Speaker,
    SpeakerListResponse,
    SpeakerSample,
    SpeakerSamplesResponse,
    TranscriptionResult,
    TranscriptionIdentifiedResult,
    UpdateSpeakerRequest,
)
from services import DiarizationService, EmbeddingService, SpeakerDBService
from services import processing
from services.job_queue import Job, JobQueue
from services.processing import Services


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global state (populated in lifespan)
settings: Settings = None
services: Services = None
job_queue: JobQueue = None

# One GPU job at a time. Both the sync endpoints and the job-queue worker
# funnel through this, so heavy work is serialized explicitly instead of
# accidentally (by blocking the event loop, as before).
gpu_semaphore = asyncio.Semaphore(1)


async def run_gpu_task(fn, /, *args, **kwargs):
    """Run a blocking GPU-bound task in a worker thread, serialized."""
    async with gpu_semaphore:
        return await asyncio.to_thread(fn, *args, **kwargs)


JOB_KINDS = {"diarize", "identify", "transcribe-diarized", "transcribe-identified"}


async def _execute_job(job: Job) -> dict:
    """Dispatch a queued job to the matching processing function."""
    p = job.params
    if job.kind == "diarize":
        return await run_gpu_task(
            processing.run_diarization, services, job.filepath,
            num_speakers=p.get("num_speakers"),
            min_speakers=p.get("min_speakers"),
            max_speakers=p.get("max_speakers"),
            exclusive=bool(p.get("exclusive", False)),
        )
    if job.kind == "identify":
        return await run_gpu_task(
            processing.run_identify, services, job.filepath,
            num_speakers=p.get("num_speakers"),
            min_speakers=p.get("min_speakers"),
            max_speakers=p.get("max_speakers"),
            similarity_threshold=p.get("similarity_threshold"),
        )
    return await run_gpu_task(
        processing.run_transcription, services, job.filepath,
        identify=(job.kind == "transcribe-identified"),
        num_speakers=p.get("num_speakers"),
        min_speakers=p.get("min_speakers"),
        max_speakers=p.get("max_speakers"),
        language=p.get("language"),
        similarity_threshold=p.get("similarity_threshold"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global settings, services, job_queue

    logger.info("Starting speaker diarization API...")

    # Load settings
    settings = get_settings()

    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # Initialize services
    services = Services(
        settings=settings,
        diarization=DiarizationService(settings),
        embedding=EmbeddingService(settings),
        speaker_db=SpeakerDBService(settings),
    )

    # Pre-initialize models (optional, can be done lazily)
    try:
        logger.info("Pre-loading models...")
        services.diarization.initialize()
        services.embedding.initialize()
        services.speaker_db.initialize()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.warning(f"Delayed model loading due to: {e}")

    job_queue = JobQueue(executor=_execute_job, cleanup=cleanup_file)
    await job_queue.start()

    yield

    # Cleanup
    logger.info("Shutting down speaker diarization API...")
    await job_queue.stop()


# Create FastAPI app
app = FastAPI(
    title="Speaker Diarization API",
    description="API for speaker diarization using pyannote community-1 model with speaker recognition via Qdrant",
    version="1.1.0",
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
        services.diarization.is_initialized and
        services.embedding.is_initialized
    )
    qdrant_connected = await asyncio.to_thread(services.speaker_db.is_connected)

    device = services.diarization.get_device()

    status = "healthy" if (models_loaded and qdrant_connected) else "degraded"

    return HealthResponse(
        status=status,
        version="1.1.0",
        models_loaded=models_loaded,
        qdrant_connected=qdrant_connected,
        device=device
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Speaker Diarization API",
        "version": "1.1.0",
        "description": "Speaker diarization using pyannote community-1 with speaker recognition",
        "docs_url": "/docs",
        "health_url": "/health",
        "jobs_url": "/jobs"
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
        filepath = await save_upload_file(file)

        result = await run_gpu_task(
            processing.run_diarization, services, filepath,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=exclusive,
        )

        return DiarizationResult(**result)

    except HTTPException:
        raise
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
        filepath = await save_upload_file(file)

        registration = await run_gpu_task(
            processing.run_register_speaker, services, filepath,
            speaker_name=speaker_name,
            extract_segments=extract_segments,
            audio_source=file.filename,
        )

        return RegisterSpeakerResponse(
            speaker_id=registration["speaker_id"],
            speaker_name=speaker_name,
            embeddings_count=registration["embeddings_count"],
            message=f"Speaker registered successfully with {registration['embeddings_count']} embedding(s)"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speaker registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if filepath:
            cleanup_file(filepath)


@app.get("/speakers", response_model=SpeakerListResponse, tags=["Speaker Recognition"])
def list_speakers():
    """
    List all registered speakers in the database.

    Returns speaker information including number of stored embeddings.
    """
    try:
        speakers_data = services.speaker_db.get_all_speakers()

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
def get_speaker(speaker_id: str):
    """
    Get information about a specific registered speaker.
    """
    try:
        speaker_data = services.speaker_db.get_speaker_by_id(speaker_id)

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
def delete_speaker(speaker_id: str):
    """
    Delete a registered speaker and all their embeddings.
    """
    try:
        deleted = services.speaker_db.delete_speaker(speaker_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Speaker not found")

        return {"message": f"Speaker {speaker_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/speakers/{speaker_id}", response_model=Speaker, tags=["Speaker Recognition"])
def update_speaker(speaker_id: str, request: UpdateSpeakerRequest):
    """
    Update a speaker's name.
    """
    try:
        updated = services.speaker_db.update_speaker_name(speaker_id, request.speaker_name)

        if not updated:
            raise HTTPException(status_code=404, detail="Speaker not found")

        # Fetch updated speaker info
        speaker_info = services.speaker_db.get_speaker_by_id(speaker_id)
        return Speaker(**speaker_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers/{speaker_id}/samples", response_model=SpeakerSamplesResponse, tags=["Speaker Recognition"])
def get_speaker_samples(speaker_id: str):
    """
    Get all voice samples for a specific speaker.
    """
    try:
        # Get speaker info
        speaker_info = services.speaker_db.get_speaker_by_id(speaker_id)
        if not speaker_info:
            raise HTTPException(status_code=404, detail="Speaker not found")

        # Get samples
        samples = services.speaker_db.get_speaker_samples(speaker_id)

        return SpeakerSamplesResponse(
            speaker_id=speaker_id,
            speaker_name=speaker_info["speaker_name"],
            samples=[SpeakerSample(**s) for s in samples],
            total_count=len(samples)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get speaker samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/speakers/{speaker_id}/samples/{sample_id}", tags=["Speaker Recognition"])
def delete_speaker_sample(speaker_id: str, sample_id: str):
    """
    Delete a specific voice sample from a speaker.

    Note: You cannot delete the last sample from a speaker. Delete the speaker instead.
    """
    try:
        # Check speaker exists and has more than one sample
        speaker_info = services.speaker_db.get_speaker_by_id(speaker_id)
        if not speaker_info:
            raise HTTPException(status_code=404, detail="Speaker not found")

        if speaker_info["embeddings_count"] <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the last sample. Delete the speaker instead."
            )

        deleted = services.speaker_db.delete_speaker_sample(speaker_id, sample_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Sample not found")

        return {"message": f"Sample {sample_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete sample: {e}")
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
        filepath = await save_upload_file(file)

        result = await run_gpu_task(
            processing.run_identify, services, filepath,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            similarity_threshold=similarity_threshold,
        )

        return IdentifyResult(**result)

    except HTTPException:
        raise
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
        speaker = await asyncio.to_thread(services.speaker_db.get_speaker_by_id, speaker_id)
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")

        filepath = await save_upload_file(file)

        registration = await run_gpu_task(
            processing.run_register_speaker, services, filepath,
            speaker_name=speaker["speaker_name"],
            extract_segments=extract_segments,
            audio_source=file.filename,
            speaker_id=speaker_id,
        )

        # Get updated speaker info
        updated_speaker = await asyncio.to_thread(services.speaker_db.get_speaker_by_id, speaker_id)

        return RegisterSpeakerResponse(
            speaker_id=speaker_id,
            speaker_name=updated_speaker["speaker_name"],
            embeddings_count=updated_speaker["embeddings_count"],
            message=f"Added {registration['embeddings_count']} new embedding(s) to speaker"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        filepath = await save_upload_file(file)

        result = await run_gpu_task(
            processing.run_transcription, services, filepath,
            identify=False,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
        )

        return TranscriptionResult(**result)

    except HTTPException:
        raise
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
        filepath = await save_upload_file(file)

        result = await run_gpu_task(
            processing.run_transcription, services, filepath,
            identify=True,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
            similarity_threshold=similarity_threshold,
        )

        return TranscriptionIdentifiedResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription with identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if filepath:
            cleanup_file(filepath)


# ============== Job Queue Endpoints ==============

@app.post("/jobs/{kind}", status_code=202, tags=["Jobs"])
async def submit_job(
    kind: str,
    file: UploadFile = File(..., description="Audio file to process"),
    num_speakers: Optional[int] = Form(None, description="Exact number of speakers (if known)"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers"),
    exclusive: bool = Form(False, description="Exclusive diarization (diarize jobs only)"),
    language: Optional[str] = Form(None, description="Language code (transcribe jobs only)"),
    similarity_threshold: Optional[float] = Form(None, description="Speaker matching threshold (identify/transcribe-identified)")
):
    """
    Submit audio for asynchronous processing and return immediately.

    Job kinds: `diarize`, `identify`, `transcribe-diarized`,
    `transcribe-identified`. Jobs run one at a time in submission order —
    the right choice for long recordings, which would otherwise hold an HTTP
    connection open for many minutes. Poll `GET /jobs/{job_id}` for status
    and the result.
    """
    if kind not in JOB_KINDS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown job kind '{kind}'. Valid kinds: {', '.join(sorted(JOB_KINDS))}"
        )

    validate_audio_file(file)
    filepath = await save_upload_file(file)

    params = {
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "exclusive": exclusive,
        "language": language,
        "similarity_threshold": similarity_threshold,
    }
    params = {k: v for k, v in params.items() if v not in (None, False)}

    job = job_queue.submit(kind, filepath, file.filename, params)

    return job.serialize(
        queue_position=job_queue.queue_position(job),
        include_result=False,
    )


@app.get("/jobs", tags=["Jobs"])
async def list_jobs():
    """List recent jobs (newest first), without result payloads."""
    jobs = job_queue.list_jobs()
    return {
        "jobs": [
            j.serialize(queue_position=job_queue.queue_position(j), include_result=False)
            for j in jobs
        ],
        **job_queue.counts(),
    }


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job(job_id: str):
    """Get a job's status; includes the result once completed."""
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.serialize(queue_position=job_queue.queue_position(job))


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Cancel a queued job, or remove a finished job's record."""
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "queued":
        job_queue.cancel(job_id)
        return {"message": f"Job {job_id} cancelled"}
    if job.status == "running":
        raise HTTPException(status_code=409, detail="Job is already running and cannot be cancelled")

    job_queue.remove(job_id)
    return {"message": f"Job {job_id} removed"}


# ============== Statistics Endpoints ==============

@app.get("/stats", tags=["Statistics"])
def get_statistics():
    """Get statistics about the speaker database and system."""
    try:
        collection_stats = services.speaker_db.get_collection_stats()
        speakers = services.speaker_db.get_all_speakers()

        return {
            "database": collection_stats,
            "speakers": {
                "total_count": len(speakers),
                "total_embeddings": sum(s["embeddings_count"] for s in speakers)
            },
            "system": {
                "device": services.diarization.get_device(),
                "diarization_model": settings.diarization_model,
                "embedding_model": settings.embedding_model
            },
            "jobs": job_queue.counts() if job_queue else {}
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
