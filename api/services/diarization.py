"""Speaker diarization service using pyannote.audio."""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.serialization import add_safe_globals
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.core.task import Problem, Resolution, Specifications, Task

from config import Settings
from services.cuda_recovery import with_cuda_retry


logger = logging.getLogger(__name__)


class DiarizationService:
    """Service for speaker diarization using pyannote community-1 model."""

    def __init__(self, settings: Settings):
        """Initialize the diarization service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.pipeline: Optional[Pipeline] = None
        self.device: Optional[torch.device] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the diarization pipeline."""
        if self._initialized:
            return

        logger.info("Initializing diarization pipeline...")

        # Determine device
        if self.settings.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.settings.device)

        logger.info(f"Using device: {self.device}")

        # Set up HuggingFace cache directory and authentication
        os.environ["HF_HOME"] = self.settings.model_cache_dir
        os.environ["TORCH_HOME"] = self.settings.model_cache_dir

        # Torch 2.6+ sets weights_only=True by default; allow pyannote classes needed to unpickle
        add_safe_globals([Specifications, Problem, Resolution, Task])

        # Set HF token via environment variable (works with all HF versions)
        if self.settings.huggingface_token:
            os.environ["HF_TOKEN"] = self.settings.huggingface_token

        # Load the pipeline
        try:
            # Try loading from local path first (for offline use)
            local_model_path = Path(self.settings.model_cache_dir) / "pyannote-speaker-diarization-community-1"

            if local_model_path.exists():
                logger.info(f"Loading model from local path: {local_model_path}")
                self.pipeline = Pipeline.from_pretrained(str(local_model_path))
            else:
                logger.info(f"Loading model from HuggingFace: {self.settings.diarization_model}")
                self.pipeline = Pipeline.from_pretrained(self.settings.diarization_model)

            # community-1 ships embedding_batch_size=32; the final partial batch
            # can trigger a multi-GB cuDNN workspace spike on long files
            # (pyannote-audio#1963). Lower it to bound VRAM; 0 keeps the model
            # default.
            if self.settings.embedding_batch_size > 0 and hasattr(self.pipeline, "embedding_batch_size"):
                self.pipeline.embedding_batch_size = self.settings.embedding_batch_size
                logger.info(f"Set embedding_batch_size={self.settings.embedding_batch_size}")

            # Move to device
            self.pipeline.to(self.device)

            self._initialized = True
            logger.info("Diarization pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            raise

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    @with_cuda_retry("_reinitialize_pipeline")
    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        exclusive: bool = False,
        use_progress_hook: bool = False
    ) -> dict:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            exclusive: If True, return exclusive diarization (no overlapping
                segments; pyannote produces this specifically for transcript
                alignment)
            use_progress_hook: If True, use progress hook for logging

        Returns:
            Dictionary with diarization results. "speaker_embeddings" maps each
            speaker label to its centroid embedding (raw wespeaker space, same
            space as individually extracted embeddings) when the pipeline
            provides one.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Get audio duration
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate

        logger.info(f"Processing audio: {audio_path} (duration: {audio_duration:.2f}s)")

        # Build kwargs for pipeline
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            elif self.settings.min_speakers is not None:
                kwargs["min_speakers"] = self.settings.min_speakers

            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers
            elif self.settings.max_speakers is not None:
                kwargs["max_speakers"] = self.settings.max_speakers

        # Run diarization using in-memory audio to bypass torchcodec chunk issues
        # Pass waveform dict instead of file path to avoid sample count mismatches
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        if use_progress_hook:
            with ProgressHook() as hook:
                output = self.pipeline(audio_input, hook=hook, **kwargs)
        else:
            output = self.pipeline(audio_input, **kwargs)

        processing_time = time.time() - start_time

        # pyannote 4.x returns DiarizeOutput; legacy checkpoints may still
        # return a bare Annotation
        annotation = getattr(output, "speaker_diarization", output)
        exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)

        if exclusive:
            if exclusive_annotation is not None:
                chosen = exclusive_annotation
            else:
                logger.warning(
                    "Exclusive diarization requested but pipeline output has no "
                    "exclusive_speaker_diarization; returning overlapping segments"
                )
                chosen = annotation
        else:
            chosen = annotation

        segments = []
        speakers = set()
        for turn, _, speaker in chosen.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3)
            })
            speakers.add(speaker)

        # Per-speaker centroid embeddings, rows aligned with
        # annotation.labels(). Zero rows are padding pyannote adds when it
        # finds fewer centroids than labels — skip them.
        speaker_embeddings: dict[str, np.ndarray] = {}
        centroids = getattr(output, "speaker_embeddings", None)
        if centroids is not None:
            for label, row in zip(annotation.labels(), centroids):
                if np.linalg.norm(row) > 0:
                    speaker_embeddings[label] = np.asarray(row)

        logger.info(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments, {processing_time:.2f}s")

        return {
            "segments": segments,
            "num_speakers": len(speakers),
            "audio_duration": round(audio_duration, 3),
            "processing_time": round(processing_time, 3),
            "exclusive": exclusive,
            "speaker_embeddings": speaker_embeddings
        }

    def _reinitialize_pipeline(self) -> None:
        """Reinitialize the pipeline after a CUDA error."""
        logger.warning("Reinitializing diarization pipeline after CUDA error...")
        previous_device = self.device
        self._initialized = False
        self.pipeline = None
        self.initialize()
        if previous_device is not None and self.device != previous_device:
            logger.critical(
                f"Device changed from {previous_device} to {self.device} after "
                "reinitialization — the GPU may have dropped off the bus"
            )

    def get_device(self) -> str:
        """Get the current device being used."""
        if self.device is not None:
            return str(self.device)
        return "not initialized"
