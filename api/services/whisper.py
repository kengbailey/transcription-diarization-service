"""Whisper STT service client for OpenAI-compatible API."""

import logging
from pathlib import Path
from typing import Optional

import httpx

from config import Settings


logger = logging.getLogger(__name__)


class WhisperService:
    """Service for calling OpenAI-compatible Whisper API for transcription."""
    
    def __init__(self, settings: Settings):
        """Initialize the Whisper service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the Whisper service."""
        if self._initialized:
            return
            
        logger.info(f"Whisper service configured for: {self.settings.whisper_api_url}")
        self._initialized = True
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        response_format: str = "verbose_json",
        timestamp_granularities: list[str] = None
    ) -> dict:
        """Transcribe audio file using the Whisper API.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'es'). None for auto-detect
            response_format: 'json', 'text', 'srt', 'verbose_json', 'vtt'
            timestamp_granularities: List of granularities: ['word', 'segment']
            
        Returns:
            Transcription result from Whisper API
        """
        if not self._initialized:
            self.initialize()
        
        if timestamp_granularities is None:
            timestamp_granularities = ["word", "segment"]
        
        url = f"{self.settings.whisper_api_url}/audio/transcriptions"
        
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Prepare the multipart form data
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (Path(audio_path).name, audio_file, "audio/mpeg")
            }
            
            data = {
                "model": self.settings.whisper_model,
                "response_format": response_format,
            }
            
            # Add language if specified
            if language:
                data["language"] = language
            elif self.settings.whisper_language:
                data["language"] = self.settings.whisper_language
            
            # Add timestamp granularities for verbose_json
            if response_format == "verbose_json":
                data["timestamp_granularities[]"] = timestamp_granularities
            
            headers = {}
            if self.settings.whisper_api_key:
                headers["Authorization"] = f"Bearer {self.settings.whisper_api_key}"
            
            try:
                with httpx.Client(timeout=self.settings.whisper_timeout) as client:
                    response = client.post(
                        url,
                        files=files,
                        data=data,
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    logger.info(f"Transcription complete: {len(result.get('text', ''))} chars")
                    
                    return result
                    
            except httpx.TimeoutException:
                logger.error("Whisper API request timed out")
                raise RuntimeError(f"Whisper API timeout after {self.settings.whisper_timeout}s")
            except httpx.HTTPStatusError as e:
                logger.error(f"Whisper API error: {e.response.status_code} - {e.response.text}")
                raise RuntimeError(f"Whisper API error: {e.response.status_code}")
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                raise
    
    def transcribe_with_words(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> dict:
        """Transcribe audio and return word-level timestamps.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en'). None for auto-detect
            
        Returns:
            Dictionary with 'text', 'segments', and 'words' keys
        """
        result = self.transcribe(
            audio_path=audio_path,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
        
        # Ensure we have the expected structure
        if "words" not in result and "segments" in result:
            # Extract words from segments if words not at top level
            words = []
            for segment in result.get("segments", []):
                if "words" in segment:
                    words.extend(segment["words"])
            result["words"] = words
        
        return result
    
    def is_available(self) -> bool:
        """Check if the Whisper API is available.
        
        Returns:
            True if API is reachable
        """
        try:
            url = f"{self.settings.whisper_api_url}/models"
            with httpx.Client(timeout=5) as client:
                response = client.get(url)
                return response.status_code == 200
        except Exception:
            return False
