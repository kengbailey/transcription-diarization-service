"""Speaker database service using Qdrant for vector storage."""

import logging
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import Settings


logger = logging.getLogger(__name__)


class SpeakerDBService:
    """Service for managing speaker embeddings in Qdrant vector database."""
    
    def __init__(self, settings: Settings):
        """Initialize the speaker database service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client: Optional[QdrantClient] = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize connection to Qdrant and ensure collection exists."""
        if self._initialized:
            return
            
        logger.info(f"Connecting to Qdrant at {self.settings.qdrant_host}:{self.settings.qdrant_port}")
        
        try:
            self.client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                timeout=30
            )
            
            # Check if collection exists, create if not
            self._ensure_collection_exists()
            
            self._initialized = True
            logger.info("Connected to Qdrant successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the speaker embeddings collection exists."""
        collection_name = self.settings.collection_name
        
        try:
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if collection_name not in existing_names:
                logger.info(f"Creating collection: {collection_name}")
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                
                # Create payload index for efficient filtering
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="speaker_name",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="speaker_id",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"Collection {collection_name} created successfully")
            else:
                logger.info(f"Collection {collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized
    
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        if not self._initialized or self.client is None:
            return False
        
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def add_speaker_embedding(
        self,
        speaker_name: str,
        embedding: np.ndarray,
        speaker_id: Optional[str] = None,
        audio_source: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """Add a speaker embedding to the database.
        
        Args:
            speaker_name: Name/identifier for the speaker
            embedding: Embedding vector as numpy array
            speaker_id: Optional existing speaker ID (for adding more embeddings)
            audio_source: Optional source file name
            metadata: Optional additional metadata
            
        Returns:
            Point ID for the added embedding
        """
        if not self._initialized:
            self.initialize()
        
        # Generate IDs
        point_id = str(uuid.uuid4())
        if speaker_id is None:
            speaker_id = str(uuid.uuid4())
        
        # Build payload
        payload = {
            "speaker_name": speaker_name,
            "speaker_id": speaker_id,
            "created_at": datetime.utcnow().isoformat(),
            "audio_source": audio_source or "unknown"
        }
        
        if metadata:
            payload["metadata"] = metadata
        
        # Flatten embedding if needed
        vector = embedding.flatten().tolist()
        
        # Add point to collection
        self.client.upsert(
            collection_name=self.settings.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        
        logger.info(f"Added embedding for speaker '{speaker_name}' (id: {speaker_id})")
        
        return speaker_id
    
    def add_speaker_embeddings_batch(
        self,
        speaker_name: str,
        embeddings: list[np.ndarray],
        speaker_id: Optional[str] = None,
        audio_source: Optional[str] = None
    ) -> str:
        """Add multiple embeddings for a speaker in batch.
        
        Args:
            speaker_name: Name/identifier for the speaker
            embeddings: List of embedding vectors
            speaker_id: Optional existing speaker ID
            audio_source: Optional source file name
            
        Returns:
            Speaker ID for the added embeddings
        """
        if not self._initialized:
            self.initialize()
        
        if speaker_id is None:
            speaker_id = str(uuid.uuid4())
        
        points = []
        for embedding in embeddings:
            point_id = str(uuid.uuid4())
            
            payload = {
                "speaker_name": speaker_name,
                "speaker_id": speaker_id,
                "created_at": datetime.utcnow().isoformat(),
                "audio_source": audio_source or "unknown"
            }
            
            vector = embedding.flatten().tolist()
            
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.settings.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(embeddings)} embeddings for speaker '{speaker_name}' (id: {speaker_id})")
        
        return speaker_id
    
    def search_similar_speakers(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> list[dict]:
        """Search for similar speakers based on embedding.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matches with speaker info and scores
        """
        if not self._initialized:
            self.initialize()
        
        vector = embedding.flatten().tolist()
        
        threshold = score_threshold or self.settings.similarity_threshold
        
        # Use query_points (new API) instead of search
        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            limit=top_k,
            score_threshold=threshold,
            with_payload=True
        )
        
        matches = []
        for point in results.points:
            matches.append({
                "speaker_name": point.payload.get("speaker_name", "unknown"),
                "speaker_id": point.payload.get("speaker_id"),
                "score": point.score,
                "audio_source": point.payload.get("audio_source"),
                "created_at": point.payload.get("created_at")
            })
        
        return matches
    
    def identify_speaker(
        self,
        embedding: np.ndarray,
        score_threshold: Optional[float] = None
    ) -> Optional[dict]:
        """Identify a speaker from their embedding.
        
        Args:
            embedding: Speaker embedding vector
            score_threshold: Minimum similarity score for a match
            
        Returns:
            Best matching speaker info or None if no match
        """
        matches = self.search_similar_speakers(
            embedding,
            top_k=1,
            score_threshold=score_threshold
        )
        
        if matches:
            return matches[0]
        return None
    
    def identify_speaker_by_voting(
        self,
        embeddings: list[np.ndarray],
        score_threshold: Optional[float] = None
    ) -> Optional[dict]:
        """Identify a speaker using multiple embeddings with voting.
        
        This aggregates results from multiple embeddings to make
        a more robust identification.
        
        Args:
            embeddings: List of speaker embedding vectors
            score_threshold: Minimum similarity score for a match
            
        Returns:
            Best matching speaker info with aggregated confidence
        """
        if not embeddings:
            return None
        
        # Collect all matches
        speaker_scores = {}  # speaker_id -> list of scores
        speaker_info = {}    # speaker_id -> speaker info
        
        for embedding in embeddings:
            matches = self.search_similar_speakers(
                embedding,
                top_k=3,
                score_threshold=score_threshold
            )
            
            for match in matches:
                speaker_id = match["speaker_id"]
                
                if speaker_id not in speaker_scores:
                    speaker_scores[speaker_id] = []
                    speaker_info[speaker_id] = match
                
                speaker_scores[speaker_id].append(match["score"])
        
        if not speaker_scores:
            return None
        
        # Find best speaker by average score
        best_speaker_id = None
        best_avg_score = 0
        
        for speaker_id, scores in speaker_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_speaker_id = speaker_id
        
        if best_speaker_id:
            result = speaker_info[best_speaker_id].copy()
            result["score"] = best_avg_score
            result["num_matches"] = len(speaker_scores[best_speaker_id])
            return result
        
        return None
    
    def get_all_speakers(self) -> list[dict]:
        """Get list of all registered speakers.
        
        Returns:
            List of speaker info dictionaries
        """
        if not self._initialized:
            self.initialize()
        
        # Get all unique speaker_ids and their info
        speakers = {}
        
        # Scroll through all points
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.settings.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            for point in results:
                speaker_id = point.payload.get("speaker_id")
                if speaker_id and speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "speaker_id": speaker_id,
                        "speaker_name": point.payload.get("speaker_name", "unknown"),
                        "embeddings_count": 0,
                        "created_at": point.payload.get("created_at")
                    }
                
                if speaker_id:
                    speakers[speaker_id]["embeddings_count"] += 1
            
            if offset is None:
                break
        
        return list(speakers.values())
    
    def get_speaker_by_id(self, speaker_id: str) -> Optional[dict]:
        """Get speaker info by ID.
        
        Args:
            speaker_id: Speaker ID to look up
            
        Returns:
            Speaker info or None if not found
        """
        if not self._initialized:
            self.initialize()
        
        # Search for points with this speaker_id
        results, _ = self.client.scroll(
            collection_name=self.settings.collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="speaker_id",
                        match=qdrant_models.MatchValue(value=speaker_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if results:
            # Count total embeddings for this speaker
            count_result = self.client.count(
                collection_name=self.settings.collection_name,
                count_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="speaker_id",
                            match=qdrant_models.MatchValue(value=speaker_id)
                        )
                    ]
                )
            )
            
            return {
                "speaker_id": speaker_id,
                "speaker_name": results[0].payload.get("speaker_name", "unknown"),
                "embeddings_count": count_result.count,
                "created_at": results[0].payload.get("created_at")
            }
        
        return None
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker and all their embeddings.
        
        Args:
            speaker_id: Speaker ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            self.initialize()
        
        # Check if speaker exists
        speaker = self.get_speaker_by_id(speaker_id)
        if not speaker:
            return False
        
        # Delete all points with this speaker_id
        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="speaker_id",
                            match=qdrant_models.MatchValue(value=speaker_id)
                        )
                    ]
                )
            )
        )
        
        logger.info(f"Deleted speaker: {speaker['speaker_name']} (id: {speaker_id})")
        
        return True
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the speaker embeddings collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self._initialized:
            self.initialize()
        
        try:
            info = self.client.get_collection(self.settings.collection_name)
            
            return {
                "collection_name": self.settings.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "collection_name": self.settings.collection_name,
                "error": str(e)
            }
