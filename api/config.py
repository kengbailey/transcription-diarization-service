"""Configuration management for the speaker diarization API."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Hugging Face
    huggingface_token: str = ""
    
    # Model settings (using 3.1 for ARM compatibility)
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    model_cache_dir: str = "/app/models"
    
    # Qdrant settings
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    collection_name: str = "speaker_embeddings"
    embedding_dimension: int = 256  # wespeaker embedding size
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_size: int = 500 * 1024 * 1024  # 500MB
    upload_dir: str = "/app/uploads"
    # Shared secret for LAN clients. Empty = auth disabled. When set, every
    # endpoint except / and /health requires it via "Authorization: Bearer"
    # or "X-API-Key".
    api_key: str = ""
    
    # Processing settings
    device: str = "auto"  # auto, cuda, or cpu
    min_speakers: int | None = None
    max_speakers: int | None = None
    # community-1 ships embedding_batch_size=32; the final partial batch can
    # trigger a multi-GB cuDNN workspace VRAM spike on long files
    # (pyannote-audio#1963). 0 keeps the model default.
    embedding_batch_size: int = 16
    # Unload the diarization/embedding models after this many seconds of GPU
    # inactivity, freeing VRAM for co-hosted services (e.g. llama-swap LLMs).
    # Models reload lazily on the next request (~10-20s). 0 = keep loaded.
    model_idle_timeout: int = 0
    
    # Speaker recognition settings
    similarity_threshold: float = 0.7  # cosine similarity threshold for speaker matching
    
    # Whisper STT settings (speaches server on this host; in Docker the
    # compose file overrides this with host.docker.internal)
    whisper_api_url: str = "http://localhost:8000/v1"
    whisper_api_key: str = "dummy"
    # Full large-v3, not distil: distillation degrades the cross-attention
    # that word timestamps derive from, and distil is English-only
    whisper_model: str = "Systran/faster-whisper-large-v3"
    whisper_language: str | None = None  # None for auto-detect
    whisper_timeout: int = 300  # 5 minutes timeout for long audio
    # Transcode uploads to 16 kHz mono WAV before sending — required for ASR
    # servers that only accept WAV (parakeet.cpp)
    whisper_send_wav: bool = False
    # Split WAV uploads into chunks of this many seconds (0 = never split).
    # parakeet.cpp's attention memory grows ~quadratically with input length
    # and it hard-crashes on CUDA OOM (measured: a 10-min chunk wants ~14 GB);
    # timestamps are re-offset and merged after transcription.
    whisper_chunk_seconds: int = 180
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
