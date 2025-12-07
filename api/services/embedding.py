"""Speaker embedding extraction service using pyannote/wespeaker."""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Inference
from pyannote.core import Segment

from config import Settings


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
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extract a single embedding from an entire audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Embedding vector as numpy array (shape: 1 x embedding_dim)
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Extracting embedding from: {audio_path}")
        
        embedding = self.inference(audio_path)
        
        logger.info(f"Embedding extracted, shape: {embedding.shape}")
        
        return embedding
    
    def extract_embedding_from_segment(
        self,
        audio_path: str,
        start: float,
        end: float
    ) -> np.ndarray:
        """Extract embedding from a specific segment of an audio file.
        
        Args:
            audio_path: Path to the audio file
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Embedding vector as numpy array (shape: 1 x embedding_dim)
        """
        if not self._initialized:
            self.initialize()
        
        segment = Segment(start, end)
        
        logger.info(f"Extracting embedding from segment [{start:.2f}s - {end:.2f}s]")
        
        embedding = self.inference.crop(audio_path, segment)
        
        return embedding
    
    def extract_sliding_embeddings(
        self,
        audio_path: str,
        duration: float = 3.0,
        step: float = 1.0
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Extract embeddings using a sliding window.
        
        Args:
            audio_path: Path to the audio file
            duration: Window duration in seconds
            step: Step size in seconds
            
        Returns:
            Tuple of (embeddings array, list of (start, end) tuples for each embedding)
        """
        if not self._initialized:
            self.initialize()
        
        # Create a new inference with sliding window
        sliding_inference = Inference(self.model, window="sliding", duration=duration, step=step)
        sliding_inference.to(self.device)
        
        logger.info(f"Extracting sliding embeddings (window={duration}s, step={step}s)")
        
        embeddings = sliding_inference(audio_path)
        
        # Get the time ranges for each embedding
        sliding_window = embeddings.sliding_window
        time_ranges = []
        for i in range(len(embeddings)):
            start = i * sliding_window.step
            end = start + sliding_window.duration
            time_ranges.append((start, end))
        
        logger.info(f"Extracted {len(embeddings)} embeddings")
        
        return np.array(embeddings.data), time_ranges
    
    def extract_embeddings_for_segments(
        self,
        audio_path: str,
        segments: list[dict],
        min_duration: float = 0.5
    ) -> list[tuple[dict, np.ndarray]]:
        """Extract embeddings for a list of speaker segments.
        
        Args:
            audio_path: Path to the audio file
            segments: List of segment dictionaries with 'start', 'end', 'speaker' keys
            min_duration: Minimum segment duration to extract embedding (in seconds)
            
        Returns:
            List of (segment, embedding) tuples
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        
        for segment in segments:
            duration = segment["end"] - segment["start"]
            
            if duration < min_duration:
                logger.debug(f"Skipping short segment: {duration:.2f}s < {min_duration}s")
                continue
            
            try:
                embedding = self.extract_embedding_from_segment(
                    audio_path,
                    segment["start"],
                    segment["end"]
                )
                results.append((segment, embedding))
            except Exception as e:
                logger.warning(f"Failed to extract embedding for segment: {e}")
                continue
        
        logger.info(f"Extracted embeddings for {len(results)}/{len(segments)} segments")
        
        return results
    
    def extract_embedding_from_memory(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> np.ndarray:
        """Extract embedding from audio loaded in memory.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Embedding vector as numpy array
        """
        if not self._initialized:
            self.initialize()
        
        embedding = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        
        return embedding
    
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
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.settings.embedding_dimension
    
    def get_device(self) -> str:
        """Get the current device being used."""
        if self.device is not None:
            return str(self.device)
        return "not initialized"
