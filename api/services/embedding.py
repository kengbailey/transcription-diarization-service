"""Speaker embedding extraction service using pyannote/wespeaker."""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.serialization import add_safe_globals
import torchaudio
from pyannote.audio import Model, Inference
from torch.torch_version import TorchVersion

from config import Settings
from services.cuda_recovery import with_cuda_retry


logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for extracting speaker embeddings using wespeaker model."""

    def __init__(self, settings: Settings):
        """Initialize the embedding service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model: Optional[Model] = None
        self.inference: Optional[Inference] = None
        self.device: Optional[torch.device] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return

        logger.info("Initializing speaker embedding model...")

        # Determine device
        if self.settings.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.settings.device)

        logger.info(f"Using device: {self.device}")

        # Set up HuggingFace cache directory and authentication
        os.environ["HF_HOME"] = self.settings.model_cache_dir
        os.environ["TORCH_HOME"] = self.settings.model_cache_dir

        # Torch 2.6+ sets weights_only=True by default; allow torch version class used in checkpoints
        add_safe_globals([TorchVersion])

        # Set HF token via environment variable (works with all HF versions)
        if self.settings.huggingface_token:
            os.environ["HF_TOKEN"] = self.settings.huggingface_token

        # Load the model
        try:
            # Try loading from local path first (for offline use)
            local_model_path = Path(self.settings.model_cache_dir) / "pyannote-wespeaker-voxceleb-resnet34-LM"

            if local_model_path.exists():
                logger.info(f"Loading embedding model from local path: {local_model_path}")
                self.model = Model.from_pretrained(str(local_model_path))
            else:
                logger.info(f"Loading embedding model from HuggingFace: {self.settings.embedding_model}")
                self.model = Model.from_pretrained(self.settings.embedding_model)

            # Create inference object with whole audio window
            self.inference = Inference(self.model, window="whole")

            # Move to device
            self.inference.to(self.device)

            self._initialized = True
            logger.info("Speaker embedding model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    @with_cuda_retry("_reinitialize")
    def _infer(self, audio_input: dict) -> np.ndarray:
        """Run the embedding model on an in-memory waveform."""
        if not self._initialized:
            self.initialize()
        return self.inference(audio_input)

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract a single embedding from an entire audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Embedding vector as numpy array (shape: 1 x embedding_dim)
        """
        logger.info(f"Extracting embedding from: {audio_path}")

        # Preload audio with torchaudio to bypass torchcodec chunk issues
        waveform, sample_rate = torchaudio.load(audio_path)
        embedding = self._infer({"waveform": waveform, "sample_rate": sample_rate})

        logger.info(f"Embedding extracted, shape: {embedding.shape}")

        return embedding

    def extract_embeddings_for_segments(
        self,
        audio_path: str,
        segments: list[dict],
        min_duration: float = 0.5
    ) -> list[tuple[dict, np.ndarray]]:
        """Extract embeddings for a list of speaker segments.

        The audio file is decoded once and sliced in memory per segment.

        Args:
            audio_path: Path to the audio file
            segments: List of segment dictionaries with 'start', 'end', 'speaker' keys
            min_duration: Minimum segment duration to extract embedding (in seconds)

        Returns:
            List of (segment, embedding) tuples
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        results = []
        for segment in segments:
            duration = segment["end"] - segment["start"]

            if duration < min_duration:
                logger.debug(f"Skipping short segment: {duration:.2f}s < {min_duration}s")
                continue

            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            try:
                embedding = self._infer({"waveform": segment_waveform, "sample_rate": sample_rate})
                results.append((segment, embedding))
            except Exception as e:
                logger.warning(f"Failed to extract embedding for segment: {e}")
                continue

        logger.info(f"Extracted embeddings for {len(results)}/{len(segments)} segments")

        return results

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1, higher means more similar)
        """
        # Flatten if needed
        e1 = embedding1.flatten()
        e2 = embedding2.flatten()

        # Compute cosine similarity
        similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

        # Convert from [-1, 1] to [0, 1]
        return float((similarity + 1) / 2)

    def _reinitialize(self) -> None:
        """Reinitialize the embedding model after a CUDA error."""
        logger.warning("Reinitializing embedding model after CUDA error...")
        previous_device = self.device
        self._initialized = False
        self.model = None
        self.inference = None
        self.initialize()
        if previous_device is not None and self.device != previous_device:
            logger.critical(
                f"Device changed from {previous_device} to {self.device} after "
                "reinitialization — the GPU may have dropped off the bus"
            )

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.settings.embedding_dimension

    def get_device(self) -> str:
        """Get the current device being used."""
        if self.device is not None:
            return str(self.device)
        return "not initialized"
