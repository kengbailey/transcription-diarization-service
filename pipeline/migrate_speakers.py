#!/usr/bin/env python3
"""One-time migration: extract speaker reference clips from manifest clusters.

Reads the existing manifest.json, finds named clusters (skipping "Mix" clusters
and unnamed ones), and copies the longest clip per speaker to
output/speaker_references/{name}/.

These reference clips are then used by pipeline_speaches.py for known-speaker
diarization via Speaches AI.
"""

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
CLIPS_DIR = OUTPUT_DIR / "clips"
SPEAKER_REFS_DIR = OUTPUT_DIR / "speaker_references"


def main():
    if not MANIFEST_FILE.exists():
        logger.error(f"Manifest not found: {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    clusters = manifest.get("clusters", [])
    if not clusters:
        logger.error("No clusters found in manifest")
        return

    logger.info(f"Found {len(clusters)} clusters")

    migrated = 0
    skipped = 0

    for cluster in clusters:
        name = cluster.get("name")

        # Skip unnamed clusters
        if not name:
            skipped += 1
            continue

        # Skip "Mix" clusters (multi-speaker)
        if name.startswith("Mix"):
            logger.info(f"  Skipping mix cluster: {name}")
            skipped += 1
            continue

        members = cluster.get("members", [])
        if not members:
            skipped += 1
            continue

        # Find the longest clip across all members of this cluster
        best_clip_file = None
        best_duration = 0.0

        for member in members:
            for clip in member.get("clips", []):
                dur = clip.get("duration", 0)
                if dur > best_duration:
                    clip_path = CLIPS_DIR / clip["file"]
                    if clip_path.exists():
                        best_duration = dur
                        best_clip_file = clip_path

        if not best_clip_file:
            logger.warning(f"  No clips found for cluster '{name}'")
            skipped += 1
            continue

        # Copy to speaker_references/{name}/
        dest_dir = SPEAKER_REFS_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / best_clip_file.name
        shutil.copy2(best_clip_file, dest_file)
        logger.info(f"  {name}: {best_clip_file.name} ({best_duration:.1f}s)")
        migrated += 1

    logger.info(f"\nDone! Migrated: {migrated}, Skipped: {skipped}")
    logger.info(f"Speaker references saved to: {SPEAKER_REFS_DIR}")


if __name__ == "__main__":
    main()
