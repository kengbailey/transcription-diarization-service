#!/usr/bin/env python3
"""Cross-meeting speaker clustering using Qdrant embedding similarity."""

import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Must match the api's COLLECTION_NAME (docker-compose.yml); the pre-rename
# collection "speaker_embeddings" still exists on disk with older enrollments.
COLLECTION = os.environ.get("COLLECTION_NAME", "work_speaker_embeddings")
MANIFEST_FILE = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output/manifest.json")
SIMILARITY_THRESHOLD = 0.55  # Cosine similarity threshold for merging


def load_manifest():
    with open(MANIFEST_FILE) as f:
        return json.load(f)


def backup_manifest():
    if MANIFEST_FILE.exists():
        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MANIFEST_FILE.with_suffix(f".{ts}.bak.json")
        shutil.copy2(MANIFEST_FILE, backup)
        print(f"Manifest backed up to {backup}")


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def get_all_points():
    """Fetch all points from Qdrant with their vectors and payloads."""
    client = httpx.Client(timeout=30)
    
    # Get collection info
    info = client.get(f"{QDRANT_URL}/collections/{COLLECTION}").json()
    total = info["result"]["points_count"]
    logger.info(f"Total points in Qdrant: {total}")
    
    # Scroll through all points
    points = []
    offset = None
    while True:
        body = {"limit": 100, "with_vector": True, "with_payload": True}
        if offset is not None:
            body["offset"] = offset
        
        resp = client.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=body).json()
        batch = resp["result"]["points"]
        if not batch:
            break
        points.extend(batch)
        offset = resp["result"].get("next_page_offset")
        if offset is None:
            break
    
    logger.info(f"Fetched {len(points)} points")
    return points


def build_speaker_embeddings(points):
    """Group points by speaker_name and compute average embedding per speaker."""
    speaker_points = defaultdict(list)
    speaker_meta = {}
    
    for p in points:
        name = p["payload"].get("speaker_name", "unknown")
        speaker_points[name].append(np.array(p["vector"]))
        if name not in speaker_meta:
            speaker_meta[name] = p["payload"]
    
    # Average embedding per speaker
    speaker_embeddings = {}
    for name, vectors in speaker_points.items():
        avg = np.mean(vectors, axis=0)
        avg = avg / np.linalg.norm(avg)  # L2 normalize
        speaker_embeddings[name] = avg
    
    logger.info(f"Unique speakers: {len(speaker_embeddings)}")
    return speaker_embeddings, speaker_meta


def cluster_by_similarity(speaker_embeddings, threshold):
    """Union-find clustering based on cosine similarity."""
    names = list(speaker_embeddings.keys())
    n = len(names)
    
    # Union-Find
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Compare all pairs
    logger.info(f"Comparing {n} speakers ({n*(n-1)//2} pairs)...")
    merge_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(speaker_embeddings[names[i]], speaker_embeddings[names[j]])
            if sim >= threshold:
                union(i, j)
                merge_count += 1
    
    logger.info(f"Merges: {merge_count}")
    
    # Group by cluster
    clusters_map = defaultdict(list)
    for i in range(n):
        clusters_map[find(i)].append(names[i])
    
    clusters = []
    for members in clusters_map.values():
        clusters.append(sorted(members))
    
    # Sort by size descending
    clusters.sort(key=len, reverse=True)
    return clusters


def main():
    backup_manifest()
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else SIMILARITY_THRESHOLD
    logger.info(f"Clustering with similarity threshold: {threshold}")
    
    # Get all embeddings from Qdrant
    points = get_all_points()
    speaker_embeddings, speaker_meta = build_speaker_embeddings(points)
    
    # Cluster
    clusters = cluster_by_similarity(speaker_embeddings, threshold)
    
    # Stats
    sizes = [len(c) for c in clusters]
    singletons = sum(1 for s in sizes if s == 1)
    multi = sum(1 for s in sizes if s > 1)
    
    logger.info(f"\nResults:")
    logger.info(f"  Total clusters: {len(clusters)}")
    logger.info(f"  Multi-member: {multi}")
    logger.info(f"  Singletons: {singletons}")
    
    if multi:
        logger.info(f"  Largest clusters: {sizes[:10]}")
    
    # Print multi-member clusters
    logger.info(f"\nMulti-member clusters:")
    for i, members in enumerate(clusters):
        if len(members) > 1:
            logger.info(f"  Cluster {i} ({len(members)} members): {members[:5]}{'...' if len(members) > 5 else ''}")
    
    # Update manifest
    manifest = load_manifest()
    manifest["clusters"] = []
    for i, members in enumerate(clusters):
        # Check if any member was human-named
        human_name = None
        for m in members:
            if "__SPEAKER_" not in m and not m.startswith("SPEAKER_"):
                human_name = m
                break
        
        cluster_members = []
        for m in members:
            meta = speaker_meta.get(m, {})
            cluster_members.append({
                "speaker_name": m,
                "audio_source": meta.get("audio_source", ""),
                "speaker_id": meta.get("speaker_id", ""),
            })
        
        manifest["clusters"].append({
            "id": i,
            "name": human_name,
            "size": len(members),
            "members": cluster_members
        })
    
    save_manifest(manifest)
    logger.info(f"\nManifest updated with {len(manifest['clusters'])} clusters")


if __name__ == "__main__":
    main()
