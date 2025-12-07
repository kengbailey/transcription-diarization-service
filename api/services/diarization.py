"""Speaker diarization service using pyannote.audio."""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from config import Settings


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
            exclusive: If True, return exclusive diarization (no overlapping segments)
            use_progress_hook: If True, use progress hook for logging
            
        Returns:
            Dictionary with diarization results
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
        
        # Run diarization
        if use_progress_hook:
            with ProgressHook() as hook:
                output = self.pipeline(audio_path, hook=hook, **kwargs)
        else:
            output = self.pipeline(audio_path, **kwargs)
        
        processing_time = time.time() - start_time
        
        # Extract segments
        segments = []
        speakers = set()
        
        for turn, _, speaker in output.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3)
            })
            speakers.add(speaker)
        
        # If exclusive mode requested, merge overlapping segments by keeping dominant speaker
        if exclusive:
            segments = self._make_exclusive(segments)
        
        logger.info(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments, {processing_time:.2f}s")
        
        return {
            "segments": segments,
            "num_speakers": len(speakers),
            "audio_duration": round(audio_duration, 3),
            "processing_time": round(processing_time, 3),
            "exclusive": exclusive
        }
    
    def diarize_from_memory(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        **kwargs
    ) -> dict:
        """Perform speaker diarization on audio loaded in memory.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            **kwargs: Additional arguments passed to diarize method
            
        Returns:
            Dictionary with diarization results
        """
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Calculate duration
        audio_duration = waveform.shape[-1] / sample_rate
        
        logger.info(f"Processing audio from memory (duration: {audio_duration:.2f}s)")
        
        # Build pipeline kwargs
        pipeline_kwargs = {}
        num_speakers = kwargs.get("num_speakers")
        min_speakers = kwargs.get("min_speakers")
        max_speakers = kwargs.get("max_speakers")
        exclusive = kwargs.get("exclusive", False)
        
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                pipeline_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                pipeline_kwargs["max_speakers"] = max_speakers
        
        # Run diarization with in-memory audio
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, **pipeline_kwargs)
        
        processing_time = time.time() - start_time
        
        # Extract segments
        if exclusive:
            segments = []
            speakers = set()
            
            for turn, speaker in output.exclusive_speaker_diarization:
                segments.append({
                    "speaker": f"SPEAKER_{speaker:02d}" if isinstance(speaker, int) else speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "duration": round(turn.end - turn.start, 3)
                })
                speakers.add(speaker)
        else:
            segments = []
            speakers = set()
            
            for turn, _, speaker in output.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "duration": round(turn.end - turn.start, 3)
                })
                speakers.add(speaker)
        
        logger.info(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments, {processing_time:.2f}s")
        
        return {
            "segments": segments,
            "num_speakers": len(speakers),
            "audio_duration": round(audio_duration, 3),
            "processing_time": round(processing_time, 3),
            "exclusive": exclusive
        }
    
    def _make_exclusive(self, segments: list[dict]) -> list[dict]:
        """Convert overlapping segments to exclusive (non-overlapping) segments.
        
        For overlapping regions, assigns to the speaker with more total duration.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of non-overlapping segments
        """
        if not segments:
            return segments
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        # Simple approach: for overlapping segments, truncate the earlier one
        exclusive = []
        for seg in sorted_segments:
            if not exclusive:
                exclusive.append(seg.copy())
            else:
                last = exclusive[-1]
                if seg["start"] < last["end"]:
                    # Overlap detected - truncate the previous segment
                    last["end"] = seg["start"]
                    last["duration"] = round(last["end"] - last["start"], 3)
                    if last["duration"] <= 0:
                        exclusive.pop()
                exclusive.append(seg.copy())
        
        return exclusive
    
    def get_device(self) -> str:
        """Get the current device being used."""
        if self.device is not None:
            return str(self.device)
        return "not initialized"
