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
    
    # Processing settings
    device: str = "auto"  # auto, cuda, or cpu
    min_speakers: int | None = None
    max_speakers: int | None = None
    
    # Speaker recognition settings
    similarity_threshold: float = 0.7  # cosine similarity threshold for speaker matching
    
    # Whisper STT settings
    whisper_api_url: str = "http://192.168.8.116:8000/v1"
    whisper_api_key: str = "dummy"
    whisper_model: str = "Systran/faster-distil-whisper-large-v3"
    whisper_language: str | None = None  # None for auto-detect
    whisper_timeout: int = 300  # 5 minutes timeout for long audio
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
