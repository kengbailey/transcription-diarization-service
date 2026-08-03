#!/usr/bin/env python3
"""Enroll reference speaker clips into the api's speaker database.

Loops pipeline/output/speaker_references/{Name}/ clips through
POST /speakers/register (first clip) and POST /speakers/add-sample (rest).
Run this after wiping/renaming the Qdrant collection — e.g. after an
embedding-model change, which invalidates every stored vector.

Usage:
    python scripts/enroll_reference_speakers.py [--api URL] [--api-key KEY]
"""

import argparse
import sys
from pathlib import Path

import httpx


AUDIO_EXTENSIONS = (".ogg", ".wav", ".mp3", ".m4a", ".flac")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8008", help="API base URL")
    parser.add_argument(
        "--refs",
        default=str(Path(__file__).resolve().parent.parent / "pipeline/output/speaker_references"),
        help="Directory of {speaker_name}/clip files",
    )
    parser.add_argument("--api-key", default="", help="API key if auth is enabled")
    args = parser.parse_args()

    refs = Path(args.refs)
    if not refs.is_dir():
        print(f"Reference directory not found: {refs}", file=sys.stderr)
        return 1

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    enrolled = 0

    with httpx.Client(base_url=args.api, timeout=300, headers=headers) as client:
        existing = {
            s["speaker_name"]
            for s in client.get("/speakers").raise_for_status().json()["speakers"]
        }

        for speaker_dir in sorted(p for p in refs.iterdir() if p.is_dir()):
            name = speaker_dir.name
            clips = sorted(
                p for p in speaker_dir.iterdir()
                if p.suffix.lower() in AUDIO_EXTENSIONS
            )
            if not clips:
                print(f"{name}: no clips, skipping")
                continue
            if name in existing:
                print(f"{name}: already enrolled, skipping")
                continue

            response = client.post(
                "/speakers/register",
                data={"speaker_name": name},
                files={"file": (clips[0].name, clips[0].read_bytes())},
            ).raise_for_status()
            speaker_id = response.json()["speaker_id"]

            for clip in clips[1:]:
                client.post(
                    f"/speakers/add-sample/{speaker_id}",
                    files={"file": (clip.name, clip.read_bytes())},
                ).raise_for_status()

            print(f"{name}: {len(clips)} clip(s) enrolled ({speaker_id})")
            enrolled += 1

    print(f"\nDone: {enrolled} speakers enrolled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
