"""STT service client for OpenAI-compatible ASR APIs (speaches, parakeet.cpp)."""

import logging
import os
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

        # Some ASR servers (parakeet.cpp) accept only WAV uploads, and
        # parakeet.cpp crashes on very long inputs (TDT full attention tops
        # out around 24 minutes) — so the WAV path also chunks long audio.
        if self.settings.whisper_send_wav:
            chunks = self._transcode_to_wav_chunks(audio_path)
            try:
                results = []
                for chunk_path, offset in chunks:
                    if len(chunks) > 1:
                        logger.info(
                            f"Transcribing chunk {len(results) + 1}/{len(chunks)} "
                            f"(offset {offset:.0f}s)"
                        )
                    result = self._post_transcription(
                        url, chunk_path, "audio/wav", language,
                        response_format, timestamp_granularities,
                    )
                    results.append((result, offset))
                return self._merge_chunked_results(results)
            finally:
                for chunk_path, _ in chunks:
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

        return self._post_transcription(
            url, audio_path, "application/octet-stream", language,
            response_format, timestamp_granularities,
        )

    def _transcode_to_wav_chunks(self, audio_path: str) -> list[tuple[str, float]]:
        """Convert audio to 16 kHz mono 16-bit WAV chunk files.

        Returns a list of (wav_path, offset_seconds). A single chunk when the
        audio fits within whisper_chunk_seconds (or chunking is disabled).
        """
        import torch
        import torchaudio
        from scipy.io import wavfile

        logger.info("Transcoding to 16 kHz mono WAV for ASR upload...")
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        pcm = (waveform.squeeze(0).clamp(-1, 1) * 32767).to(torch.int16).numpy()

        chunk_seconds = self.settings.whisper_chunk_seconds
        chunk_samples = int(chunk_seconds * 16000) if chunk_seconds > 0 else 0
        if chunk_samples <= 0 or len(pcm) <= chunk_samples:
            wav_path = f"{audio_path}.16k.wav"
            wavfile.write(wav_path, 16000, pcm)
            return [(wav_path, 0.0)]

        chunks = []
        for i, start in enumerate(range(0, len(pcm), chunk_samples)):
            wav_path = f"{audio_path}.chunk{i}.wav"
            wavfile.write(wav_path, 16000, pcm[start:start + chunk_samples])
            chunks.append((wav_path, start / 16000.0))
        logger.info(f"Split into {len(chunks)} chunks of <= {chunk_seconds}s")
        return chunks

    @staticmethod
    def _merge_chunked_results(results: list[tuple[dict, float]]) -> dict:
        """Merge per-chunk transcription results, re-offsetting timestamps."""
        if len(results) == 1:
            return results[0][0]

        merged = {"text": "", "words": [], "segments": [], "duration": 0.0, "language": None}
        texts = []
        for result, offset in results:
            texts.append((result.get("text") or "").strip())
            for word in result.get("words") or []:
                merged["words"].append({
                    **word,
                    "start": word.get("start", 0) + offset,
                    "end": word.get("end", 0) + offset,
                })
            for seg in result.get("segments") or []:
                seg = {
                    **seg,
                    "start": seg.get("start", 0) + offset,
                    "end": seg.get("end", 0) + offset,
                }
                if seg.get("words"):
                    seg["words"] = [
                        {**w, "start": w.get("start", 0) + offset, "end": w.get("end", 0) + offset}
                        for w in seg["words"]
                    ]
                merged["segments"].append(seg)
            merged["duration"] = max(
                merged["duration"], offset + float(result.get("duration") or 0)
            )
            if merged["language"] is None:
                merged["language"] = result.get("language")
        merged["text"] = " ".join(t for t in texts if t)
        return merged

    def _post_transcription(
        self,
        url: str,
        audio_path: str,
        content_type: str,
        language: Optional[str],
        response_format: str,
        timestamp_granularities: list[str],
    ) -> dict:
        # Prepare the multipart form data
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (Path(audio_path).name, audio_file, content_type)
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
