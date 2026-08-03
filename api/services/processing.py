"""Blocking task orchestration shared by the sync endpoints and the job queue.

Everything in this module runs inside a worker thread (via asyncio.to_thread)
while the caller holds the GPU semaphore — nothing here may touch the event
loop.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from config import Settings
from services.diarization import DiarizationService
from services.embedding import EmbeddingService
from services.speaker_db import SpeakerDBService
from services.transcript_merger import TranscriptMerger
from services.whisper import WhisperService


logger = logging.getLogger(__name__)


@dataclass
class Services:
    """The singleton services the tasks operate on."""

    settings: Settings
    diarization: DiarizationService
    embedding: EmbeddingService
    speaker_db: SpeakerDBService


_INTERNAL_DIARIZATION_KEYS = {"speaker_embeddings", "waveform", "sample_rate"}

# Cap on per-speaker audio used to build the identification embedding
MAX_IDENTIFY_SECONDS = 60.0
# Segments shorter than this contribute nothing useful to an embedding
MIN_IDENTIFY_SEGMENT_SECONDS = 0.5


def _public_diarization(diarization_result: dict) -> dict:
    """Drop non-JSON-serializable internals (numpy/tensors) from a result."""
    return {
        k: v for k, v in diarization_result.items()
        if k not in _INTERNAL_DIARIZATION_KEYS
    }


def identify_diarized_speakers(
    services: Services,
    filepath: str,
    diarization_result: dict,
    similarity_threshold: Optional[float] = None,
) -> tuple[dict, dict]:
    """Match each diarized speaker against registered speakers.

    Builds one identification embedding per speaker by concatenating their
    longest segments (up to MAX_IDENTIFY_SECONDS) and matching it against
    Qdrant. The diarization pipeline's internal centroids are deliberately
    NOT used: they live in the pipeline's own embedding space (ResNet34),
    while enrollment uses the standalone embedding model — the spaces are
    incompatible even though the dimensions match.

    Returns:
        (speaker_mapping, speaker_confidences) keyed by diarized speaker label
    """
    import torch

    waveform = diarization_result.get("waveform")
    sample_rate = diarization_result.get("sample_rate")
    if waveform is None:
        # Not produced by DiarizationService.diarize — decode once ourselves
        import torchaudio
        waveform, sample_rate = torchaudio.load(filepath)

    speaker_segments = {}
    for segment in diarization_result["segments"]:
        speaker_segments.setdefault(segment["speaker"], []).append(segment)

    speaker_mapping = {}
    speaker_confidences = {}

    for speaker, segments in speaker_segments.items():
        pieces = []
        total = 0.0
        for seg in sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True):
            duration = seg["end"] - seg["start"]
            if duration < MIN_IDENTIFY_SEGMENT_SECONDS:
                break  # sorted by duration — the rest are shorter still
            take = min(duration, MAX_IDENTIFY_SECONDS - total)
            if take <= 0:
                break
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int((seg["start"] + take) * sample_rate)
            pieces.append(waveform[:, start_sample:end_sample])
            total += take

        match = None
        if pieces:
            speaker_waveform = torch.cat(pieces, dim=1)
            embedding = services.embedding.extract_embedding_from_waveform(
                speaker_waveform, sample_rate
            )
            if not _has_nan(embedding):
                match = services.speaker_db.identify_speaker(
                    embedding=embedding,
                    score_threshold=similarity_threshold,
                )

        speaker_mapping[speaker] = match["speaker_name"] if match else None
        speaker_confidences[speaker] = match["score"] if match else None

    return speaker_mapping, speaker_confidences


def _has_nan(embedding) -> bool:
    import numpy as np
    return bool(np.any(np.isnan(np.asarray(embedding))))


def run_diarization(
    services: Services,
    filepath: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    exclusive: bool = False,
) -> dict:
    """Diarize a file. Returns a DiarizationResult-shaped dict."""
    result = services.diarization.diarize(
        audio_path=filepath,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=exclusive,
    )
    return _public_diarization(result)


def run_identify(
    services: Services,
    filepath: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
) -> dict:
    """Diarize + identify speakers. Returns an IdentifyResult-shaped dict."""
    start_time = time.time()

    diarization_result = services.diarization.diarize(
        audio_path=filepath,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=True,  # cleaner speaker identification
    )

    speaker_mapping, speaker_confidences = identify_diarized_speakers(
        services, filepath, diarization_result, similarity_threshold
    )

    segments = [
        {
            "speaker": seg["speaker"],
            "identified_as": speaker_mapping.get(seg["speaker"]),
            "confidence": speaker_confidences.get(seg["speaker"]),
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
        }
        for seg in diarization_result["segments"]
    ]

    return {
        "segments": segments,
        "speaker_mapping": speaker_mapping,
        "num_speakers": diarization_result["num_speakers"],
        "num_identified": sum(1 for v in speaker_mapping.values() if v is not None),
        "audio_duration": diarization_result["audio_duration"],
        "processing_time": round(time.time() - start_time, 3),
    }


def run_transcription(
    services: Services,
    filepath: str,
    identify: bool,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    language: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> dict:
    """Diarize (+identify) + transcribe + merge.

    Returns a TranscriptionResult-shaped dict, plus speaker_mapping and
    num_identified when identify=True.
    """
    start_time = time.time()

    whisper_service = WhisperService(services.settings)
    whisper_service.initialize()
    merger = TranscriptMerger()

    logger.info("Running diarization...")
    diarization_result = services.diarization.diarize(
        audio_path=filepath,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=True,
    )

    speaker_mapping: dict = {}
    speaker_confidences: dict = {}
    if identify:
        logger.info("Identifying speakers...")
        speaker_mapping, speaker_confidences = identify_diarized_speakers(
            services, filepath, diarization_result, similarity_threshold
        )

    logger.info("Running Whisper transcription...")
    whisper_result = whisper_service.transcribe_with_words(
        audio_path=filepath,
        language=language,
    )

    logger.info("Merging transcription with diarization...")
    merged = merger.merge_transcription_with_diarization(
        whisper_result=whisper_result,
        diarization_result=diarization_result,
        speaker_mapping=speaker_mapping if identify else None,
        speaker_confidences=speaker_confidences if identify else None,
    )

    segments = [
        {
            "speaker": seg["speaker"],
            "identified_as": seg.get("identified_as"),
            "confidence": seg.get("confidence"),
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg.get("duration", round(seg["end"] - seg["start"], 3)),
            "text": seg["text"],
        }
        for seg in merged["segments"]
    ]

    result = {
        "text": merged["text"],
        "segments": segments,
        "num_speakers": merged["num_speakers"],
        "duration": merged["duration"],
        "language": merged.get("language"),
        "processing_time": round(time.time() - start_time, 3),
    }
    if identify:
        result["speaker_mapping"] = speaker_mapping
        result["num_identified"] = sum(
            1 for v in speaker_mapping.values() if v is not None
        )
    return result


def run_register_speaker(
    services: Services,
    filepath: str,
    speaker_name: str,
    extract_segments: bool,
    audio_source: Optional[str],
    speaker_id: Optional[str] = None,
) -> dict:
    """Register a new speaker, or add a sample when speaker_id is given.

    Raises:
        ValueError: if no usable speech segments could be extracted
    """
    if extract_segments:
        diarization_result = services.diarization.diarize(
            audio_path=filepath,
            num_speakers=1,  # assume single speaker for registration
        )

        segment_embeddings = services.embedding.extract_embeddings_for_segments(
            audio_path=filepath,
            segments=diarization_result["segments"],
            min_duration=1.0,  # minimum 1 second for good embedding
        )

        if not segment_embeddings:
            raise ValueError("Could not extract any valid speech segments from the audio")

        embeddings = [emb for _, emb in segment_embeddings]
        result_speaker_id = services.speaker_db.add_speaker_embeddings_batch(
            speaker_name=speaker_name,
            embeddings=embeddings,
            speaker_id=speaker_id,
            audio_source=audio_source,
        )
        embeddings_count = len(embeddings)
    else:
        embedding = services.embedding.extract_embedding(filepath)
        result_speaker_id = services.speaker_db.add_speaker_embedding(
            speaker_name=speaker_name,
            embedding=embedding,
            speaker_id=speaker_id,
            audio_source=audio_source,
        )
        embeddings_count = 1

    return {"speaker_id": result_speaker_id, "embeddings_count": embeddings_count}
