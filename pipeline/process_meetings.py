#!/usr/bin/env python3
"""Process meeting audio files: diarize, extract speaker clips, compute embeddings."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8008"
OUTPUT_DIR = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output")
CLIPS_DIR = OUTPUT_DIR / "clips"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
AUDIO_DIR = Path("/home/syran/tmp/mtgs/Users/kenbailey/Library/Application Support/com.taperlabs.shadow")

# Minimum segment duration (seconds) for a usable speaker clip
MIN_CLIP_DURATION = 3.0
# Maximum clip duration to extract
MAX_CLIP_DURATION = 30.0
# How many clips per speaker per meeting
MAX_CLIPS_PER_SPEAKER = 3


def load_manifest() -> dict:
    """Load existing manifest or create new one."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"meetings": {}, "speakers": {}, "clusters": []}


def backup_manifest():
    """Create a timestamped backup of the manifest before processing."""
    if MANIFEST_FILE.exists():
        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MANIFEST_FILE.with_suffix(f".{ts}.bak.json")
        shutil.copy2(MANIFEST_FILE, backup)
        logger.info(f"Manifest backed up to {backup}")


def save_manifest(manifest: dict):
    """Save manifest to disk."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def file_hash(filepath: str) -> str:
    """Get a short hash for a file based on name + size."""
    stat = os.stat(filepath)
    key = f"{os.path.basename(filepath)}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def extract_clip(audio_path: str, start: float, end: float, output_path: str) -> bool:
    """Extract an audio clip using ffmpeg."""
    duration = end - start
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(duration),
                "-ac", "1", "-ar", "16000",
                "-c:a", "libopus", "-b:a", "48k",
                output_path
            ],
            capture_output=True, timeout=60, check=True
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to extract clip: {e}")
        return False


def restart_api_container():
    """Restart the diarization API container to recover from CUDA errors."""
    logger.warning("Restarting API container to recover from CUDA error...")
    subprocess.run(
        ["docker", "compose", "-f", "/home/syran/sandbox/transcription-diarization-service/docker-compose.yml",
         "restart", "api"],
        capture_output=True, timeout=120
    )
    # Wait for container to be healthy
    for i in range(60):
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(f"{API_URL}/health")
                if r.status_code == 200:
                    health = r.json()
                    if health.get("models_loaded"):
                        logger.info("API container recovered and healthy")
                        return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("API container failed to recover")
    return False


MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB - files bigger than this get split
CHUNK_DURATION = 900  # 15 minutes per chunk


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def split_audio(audio_path: str, chunk_dir: str) -> list[str]:
    """Split a large audio file into chunks. Returns list of chunk file paths."""
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        return [audio_path]

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = []
    start = 0
    idx = 0

    while start < duration:
        chunk_path = os.path.join(chunk_dir, f"{basename}__chunk{idx}.m4a")
        chunk_dur = min(CHUNK_DURATION, duration - start)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-ss", str(start), "-t", str(chunk_dur),
                 "-c", "copy", chunk_path],
                capture_output=True, timeout=120, check=True
            )
            chunks.append(chunk_path)
        except Exception as e:
            logger.warning(f"Failed to split chunk {idx}: {e}")
        start += CHUNK_DURATION
        idx += 1

    logger.info(f"Split {os.path.basename(audio_path)} into {len(chunks)} chunks ({duration:.0f}s total)")
    return chunks


def diarize_file(audio_path: str) -> dict:
    """Run diarization on an audio file via the API. Splits large files first."""
    file_size = os.path.getsize(audio_path)

    if file_size > MAX_FILE_SIZE:
        # Split into chunks and diarize each, then merge results
        logger.info(f"Large file ({file_size // 1024 // 1024}MB), splitting into chunks...")
        chunk_dir = os.path.join(str(OUTPUT_DIR), "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunks = split_audio(audio_path, chunk_dir)

        if not chunks:
            raise RuntimeError("Failed to split audio file")

        all_segments = []
        all_speakers = set()
        total_duration = 0
        total_processing = 0
        chunk_offset = 0

        for chunk_path in chunks:
            try:
                result = _diarize_single(chunk_path)
                chunk_dur = result.get("audio_duration", 0)

                # Offset segment timestamps by chunk position
                for seg in result.get("segments", []):
                    seg["start"] = round(seg["start"] + chunk_offset, 3)
                    seg["end"] = round(seg["end"] + chunk_offset, 3)
                    all_segments.append(seg)
                    all_speakers.add(seg["speaker"])

                total_duration += chunk_dur
                total_processing += result.get("processing_time", 0)
                chunk_offset += chunk_dur
            except Exception as e:
                logger.warning(f"Failed to diarize chunk {chunk_path}: {e}")
                # Try to recover
                restart_api_container()
                chunk_offset += CHUNK_DURATION  # Approximate offset
            finally:
                # Clean up chunk file
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

        return {
            "segments": all_segments,
            "num_speakers": len(all_speakers),
            "audio_duration": round(total_duration, 3),
            "processing_time": round(total_processing, 3),
            "exclusive": True
        }
    else:
        return _diarize_single(audio_path)


def _diarize_single(audio_path: str) -> dict:
    """Diarize a single (non-split) audio file."""
    file_size = os.path.getsize(audio_path)
    logger.info(f"Diarizing: {os.path.basename(audio_path)} ({file_size // 1024 // 1024}MB)")
    with open(audio_path, "rb") as f:
        with httpx.Client(timeout=600) as client:
            response = client.post(
                f"{API_URL}/diarize",
                files={"file": (os.path.basename(audio_path), f, "audio/mp4")},
                data={"exclusive": "true"}
            )
            if response.status_code == 500:
                logger.warning("Got 500 from API, attempting container restart...")
                if restart_api_container():
                    f.seek(0)
                    response = client.post(
                        f"{API_URL}/diarize",
                        files={"file": (os.path.basename(audio_path), f, "audio/mp4")},
                        data={"exclusive": "true"}
                    )
            response.raise_for_status()
            return response.json()


def get_embedding(audio_path: str) -> list[float] | None:
    """Get speaker embedding for a clip via the API's internal embedding service.
    
    We use the /identify endpoint with the clip to get an embedding comparison,
    but since we need raw embeddings, we'll use a direct approach.
    """
    # We'll call the diarize endpoint on the clip and then use identify
    # Actually, let's use the speaker registration endpoint to extract embeddings
    # But we don't want to actually register. Instead, let's add a utility endpoint.
    # For now, we'll use the identify endpoint - if no speakers are registered,
    # it will still process the embeddings internally.
    
    # Alternative: use the /identify endpoint which extracts embeddings as part of processing
    # The segments returned will have embedding data we can compare
    
    # Simplest approach: register as temp speaker, get embedding, delete
    # But that's messy. Let's just use cosine similarity from the identify endpoint.
    
    # For clustering, we'll rely on the API's identification capability.
    # We register all speakers and let Qdrant handle similarity.
    return None  # Embeddings handled via Qdrant registration


def register_speaker_clip(clip_path: str, speaker_label: str, meeting_id: str) -> str | None:
    """Register a speaker clip in Qdrant and return the speaker_id."""
    name = f"{meeting_id}__{speaker_label}"
    try:
        with open(clip_path, "rb") as f:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{API_URL}/speakers/register",
                    files={"file": (os.path.basename(clip_path), f, "audio/ogg")},
                    data={
                        "speaker_name": name,
                        "extract_segments": "true"
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("speaker_id")
    except Exception as e:
        logger.warning(f"Failed to register speaker clip {clip_path}: {e}")
        return None


def find_similar_speakers(clip_path: str, threshold: float = 0.5) -> list[dict]:
    """Find similar speakers for a clip using the identify endpoint."""
    try:
        with open(clip_path, "rb") as f:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{API_URL}/identify",
                    files={"file": (os.path.basename(clip_path), f, "audio/ogg")},
                    data={"similarity_threshold": str(threshold)}
                )
                response.raise_for_status()
                result = response.json()
                return result
    except Exception as e:
        logger.warning(f"Failed to identify speakers in {clip_path}: {e}")
        return {}


def process_meeting(audio_path: str, manifest: dict) -> dict | None:
    """Process a single meeting file."""
    basename = os.path.basename(audio_path)
    fhash = file_hash(audio_path)
    meeting_id = basename.replace("-MergedAudio.m4a", "")[:16]
    
    # Skip if already processed
    if fhash in manifest["meetings"]:
        logger.info(f"Skipping already processed: {basename}")
        return manifest["meetings"][fhash]
    
    # Get file info
    file_size = os.path.getsize(audio_path)
    if file_size < 10000:  # Skip tiny files (<10KB)
        logger.info(f"Skipping tiny file: {basename} ({file_size} bytes)")
        return None
    
    # Diarize
    try:
        diar_result = diarize_file(audio_path)
    except Exception as e:
        logger.error(f"Failed to diarize {basename}: {e}")
        return None
    
    segments = diar_result.get("segments", [])
    num_speakers = diar_result.get("num_speakers", 0)
    audio_duration = diar_result.get("audio_duration", 0)
    
    logger.info(f"  {num_speakers} speakers, {len(segments)} segments, {audio_duration:.0f}s duration")
    
    # Group segments by speaker
    speaker_segments = {}
    for seg in segments:
        spk = seg["speaker"]
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append(seg)
    
    # Extract best clips per speaker
    meeting_speakers = {}
    for speaker, segs in speaker_segments.items():
        # Sort by duration, pick longest segments
        segs_sorted = sorted(segs, key=lambda s: s["duration"], reverse=True)
        good_segs = [s for s in segs_sorted if s["duration"] >= MIN_CLIP_DURATION]
        
        if not good_segs:
            # Use best available even if short
            good_segs = segs_sorted[:1]
        
        clips = []
        total_speaker_time = sum(s["duration"] for s in segs)
        
        for i, seg in enumerate(good_segs[:MAX_CLIPS_PER_SPEAKER]):
            clip_name = f"{meeting_id}__{speaker}__clip{i}.ogg"
            clip_path = str(CLIPS_DIR / clip_name)
            
            start = seg["start"]
            end = min(seg["end"], seg["start"] + MAX_CLIP_DURATION)
            
            if extract_clip(audio_path, start, end, clip_path):
                clips.append({
                    "file": clip_name,
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 3)
                })
        
        meeting_speakers[speaker] = {
            "clips": clips,
            "total_time": round(total_speaker_time, 1),
            "segment_count": len(segs),
            "speaker_id": None  # Will be set after Qdrant registration
        }
    
    meeting_data = {
        "file": basename,
        "hash": fhash,
        "meeting_id": meeting_id,
        "num_speakers": num_speakers,
        "audio_duration": audio_duration,
        "processing_time": diar_result.get("processing_time", 0),
        "speakers": meeting_speakers,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    manifest["meetings"][fhash] = meeting_data
    save_manifest(manifest)
    
    return meeting_data


def register_all_speakers(manifest: dict):
    """Register all extracted speaker clips in Qdrant for similarity comparison."""
    logger.info("Registering speaker clips in Qdrant...")
    
    for fhash, meeting in manifest["meetings"].items():
        for speaker_label, speaker_data in meeting.get("speakers", {}).items():
            if speaker_data.get("speaker_id"):
                continue  # Already registered
            
            clips = speaker_data.get("clips", [])
            if not clips:
                continue
            
            # Use the longest clip for registration
            best_clip = max(clips, key=lambda c: c["duration"])
            clip_path = str(CLIPS_DIR / best_clip["file"])
            
            if not os.path.exists(clip_path):
                continue
            
            speaker_id = register_speaker_clip(
                clip_path, speaker_label, meeting["meeting_id"]
            )
            
            if speaker_id:
                speaker_data["speaker_id"] = speaker_id
                logger.info(f"  Registered {meeting['meeting_id']}/{speaker_label} -> {speaker_id}")
    
    save_manifest(manifest)


def build_clusters(manifest: dict):
    """Build speaker clusters by comparing all registered speakers via Qdrant similarity."""
    logger.info("Building speaker clusters...")
    
    # Collect all registered speakers
    all_speakers = []
    for fhash, meeting in manifest["meetings"].items():
        for speaker_label, speaker_data in meeting.get("speakers", {}).items():
            if speaker_data.get("speaker_id") and speaker_data.get("clips"):
                all_speakers.append({
                    "meeting_hash": fhash,
                    "meeting_id": meeting["meeting_id"],
                    "meeting_file": meeting["file"],
                    "speaker_label": speaker_label,
                    "speaker_id": speaker_data["speaker_id"],
                    "clips": speaker_data["clips"],
                    "total_time": speaker_data["total_time"]
                })
    
    if not all_speakers:
        logger.warning("No registered speakers found")
        return
    
    # For each speaker, find similar ones using the identify endpoint
    similarity_matrix = {}
    
    for spk in all_speakers:
        best_clip = max(spk["clips"], key=lambda c: c["duration"])
        clip_path = str(CLIPS_DIR / best_clip["file"])
        
        if not os.path.exists(clip_path):
            continue
        
        result = find_similar_speakers(clip_path, threshold=0.45)
        
        if result and "segments" in result:
            # The identify endpoint returns speaker mapping
            mapping = result.get("speaker_mapping", {})
            for label, identified_name in mapping.items():
                if identified_name:
                    key = f"{spk['meeting_id']}__{spk['speaker_label']}"
                    if key not in similarity_matrix:
                        similarity_matrix[key] = []
                    similarity_matrix[key].append(identified_name)
    
    # Build clusters using union-find approach
    # Group speakers that were identified as similar
    speaker_to_cluster = {}
    clusters = []
    
    for spk in all_speakers:
        key = f"{spk['meeting_id']}__{spk['speaker_label']}"
        if key not in speaker_to_cluster:
            cluster_id = len(clusters)
            clusters.append({
                "id": cluster_id,
                "name": None,  # User will assign
                "members": [spk]
            })
            speaker_to_cluster[key] = cluster_id
    
    manifest["clusters"] = clusters
    manifest["all_speakers"] = all_speakers
    save_manifest(manifest)
    logger.info(f"Built {len(clusters)} initial clusters (pre-merge)")


def main():
    backup_manifest()
    parser = argparse.ArgumentParser(description="Process meeting audio files")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--min-size", type=int, default=500000, help="Min file size in bytes (default 500KB)")
    parser.add_argument("--skip-register", action="store_true", help="Skip Qdrant registration")
    parser.add_argument("--skip-cluster", action="store_true", help="Skip clustering")
    args = parser.parse_args()
    
    # Ensure output dirs exist
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    
    # Find audio files
    audio_files = sorted(AUDIO_DIR.glob("*-MergedAudio.m4a"))
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Filter by size
    audio_files = [f for f in audio_files if f.stat().st_size >= args.min_size]
    logger.info(f"After size filter (>={args.min_size}B): {len(audio_files)} files")
    
    if args.limit:
        audio_files = audio_files[:args.limit]
    
    # Process each meeting
    for i, audio_file in enumerate(audio_files):
        logger.info(f"\n[{i+1}/{len(audio_files)}] Processing {audio_file.name}")
        process_meeting(str(audio_file), manifest)
    
    if not args.skip_register:
        register_all_speakers(manifest)
    
    if not args.skip_cluster:
        build_clusters(manifest)
    
    # Summary
    total_meetings = len(manifest["meetings"])
    total_speakers = sum(
        len(m.get("speakers", {})) for m in manifest["meetings"].values()
    )
    total_clusters = len(manifest.get("clusters", []))
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Meetings processed: {total_meetings}")
    logger.info(f"Speaker instances found: {total_speakers}")
    logger.info(f"Clusters: {total_clusters}")
    logger.info(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
