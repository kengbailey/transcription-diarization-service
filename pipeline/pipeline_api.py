#!/usr/bin/env python3
"""Meeting pipeline backed by the local diarization API.

Successor to pipeline_speaches.py (which targeted a now-retired remote
Speaches box). Submits each meeting to the api's job queue
(POST /jobs/transcribe-identified), polls until done, and writes
manifest/transcript/summary files in the same layout meeting_server.py
already reads. Speaker names come from the api's Qdrant roster — enroll new
speakers via the UI or scripts/enroll_reference_speakers.py.

Configuration via environment variables (defaults in CONFIG below):
    MEETINGS_API_URL   api base (default http://localhost:8008)
    MEETINGS_API_KEY   api key, if auth is enabled
    AUDIO_DIR          directory of *-MergedAudio.m4a recordings
    OUTPUT_DIR         pipeline output root (default: <this dir>/output)
    LLM_API_URL        OpenAI-compatible chat endpoint for summaries
                       (unset = skip summarization)
    LLM_MODEL          model name for summaries
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("MEETINGS_API_URL", "http://localhost:8008")
API_KEY = os.environ.get("MEETINGS_API_KEY", "")
AUDIO_DIR = Path(os.environ.get(
    "AUDIO_DIR",
    "/home/syran/tmp/mtgs/Users/kenbailey/Library/Application Support/com.taperlabs.shadow",
))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(Path(__file__).resolve().parent / "output")))
# llama-swap on this host; it loads the model on demand and unloads it after
# its own TTL. Set LLM_API_URL="" to disable summarization entirely.
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:9292/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b")

TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

MIN_FILE_SIZE = 500_000       # skip tiny junk recordings (<500KB)
POLL_INTERVAL = 10.0          # seconds between job status polls
JOB_TIMEOUT = 3 * 3600        # give up on a single meeting after 3h
MANIFEST_BACKUPS_KEPT = 10


# ---------------------------------------------------------------------------
# Manifest helpers (layout shared with the older pipeline scripts)
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"meetings": {}, "speakers": {}, "clusters": []}


def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def backup_manifest():
    if not MANIFEST_FILE.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MANIFEST_FILE.with_suffix(f".{ts}.bak.json")
    shutil.copy2(MANIFEST_FILE, backup)
    logger.info(f"Manifest backed up to {backup.name}")
    # Rotate: keep the newest N backups
    backups = sorted(MANIFEST_FILE.parent.glob("manifest.*.bak.json"))
    for old in backups[:-MANIFEST_BACKUPS_KEPT]:
        old.unlink()
        logger.info(f"Pruned old backup {old.name}")


def file_hash(filepath: str) -> str:
    stat = os.stat(filepath)
    key = f"{os.path.basename(filepath)}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def api_client() -> httpx.Client:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    return httpx.Client(base_url=API_URL, timeout=120, headers=headers)


def check_api(client: httpx.Client) -> bool:
    try:
        health = client.get("/health").raise_for_status().json()
        logger.info(
            f"API healthy (device={health.get('device')}, "
            f"queue={health.get('jobs_queued')}/{health.get('jobs_running')})"
        )
        return health.get("status") == "healthy"
    except Exception as e:
        logger.error(f"API unreachable at {API_URL}: {e}")
        return False


def transcribe_meeting(client: httpx.Client, audio_path: Path) -> dict | None:
    """Submit a meeting to the job queue and wait for the result."""
    with open(audio_path, "rb") as f:
        response = client.post(
            "/jobs/transcribe-identified",
            files={"file": (audio_path.name, f)},
        )
    response.raise_for_status()
    job = response.json()
    job_id = job["job_id"]
    logger.info(f"  Submitted job {job_id} (queue position {job.get('queue_position')})")

    deadline = time.time() + JOB_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        job = client.get(f"/jobs/{job_id}").raise_for_status().json()
        status = job["status"]
        if status == "completed":
            return job["result"]
        if status in ("failed", "cancelled"):
            logger.error(f"  Job {job_id} {status}: {job.get('error')}")
            return None

    logger.error(f"  Job {job_id} timed out after {JOB_TIMEOUT}s")
    return None


# ---------------------------------------------------------------------------
# Transcript formatting (same layout as pipeline_speaches.py)
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def result_to_segments(result: dict) -> list[dict]:
    """Convert api transcript segments to the pipeline's merged-segment shape.

    Speaker labels become enrolled names where identified; unidentified
    speakers keep their SPEAKER_NN label.
    """
    merged = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        merged.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "speaker": seg.get("identified_as") or seg["speaker"],
            "confidence": seg.get("confidence"),
            "text": text,
        })
    return merged


def format_transcript_md(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        ts = f"{_fmt_time(seg['start'])} - {_fmt_time(seg['end'])}"
        lines.append(f"**{seg['speaker']}** ({ts}):\n{seg['text']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LLM summarization (optional)
# ---------------------------------------------------------------------------
def unload_llm() -> None:
    """Ask llama-swap to unload its models before GPU-heavy transcription.

    Diarization peaks at 10-16 GB; a resident LLM would collide. Best-effort —
    other OpenAI-compatible servers simply 404 this.
    """
    if not LLM_API_URL:
        return
    base = LLM_API_URL.rsplit("/v1", 1)[0]
    try:
        httpx.get(f"{base}/unload", timeout=10)
        logger.info("Asked llama-swap to unload LLM models")
    except Exception:
        pass


def summarize_transcript(transcript_text: str) -> str | None:
    max_chars = 80_000
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED]"

    prompt = f"""You are a meeting summarizer. Analyze this meeting transcript and provide a structured summary.

TRANSCRIPT:
{transcript_text}

Please provide:
1. **Meeting Title** - A brief descriptive title
2. **Participants** - List of identified speakers
3. **Summary** - 2-3 paragraph overview of what was discussed
4. **Key Topics** - Bullet list of main topics covered
5. **Action Items** - Any tasks, follow-ups, or commitments mentioned (with who owns them if clear)
6. **Decisions Made** - Any decisions or agreements reached
7. **Notable Quotes** - Any particularly important or notable statements (max 3)

Keep it concise but comprehensive. Use the speaker names as provided in the transcript."""

    try:
        with httpx.Client(timeout=600) as client:
            resp = client.post(
                f"{LLM_API_URL}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                },
            )
            if resp.status_code != 200:
                logger.error(f"  LLM error ({resp.status_code}): {resp.text[:200]}")
                return None
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"  LLM summarization failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-meeting processing
# ---------------------------------------------------------------------------
def process_meeting(
    client: httpx.Client,
    audio_path: Path,
    manifest: dict,
) -> bool:
    fhash = file_hash(str(audio_path))
    meeting_id = audio_path.name.replace("-MergedAudio.m4a", "")[:16]
    existing = manifest["meetings"].get(fhash, {})

    logger.info(f"Processing {audio_path.name} (id={meeting_id})")

    result = transcribe_meeting(client, audio_path)
    if not result:
        return False

    merged = result_to_segments(result)
    if not merged:
        logger.warning("  No transcript segments produced (silent recording?)")
        manifest["meetings"][fhash] = {
            **existing,
            "file": audio_path.name,
            "hash": fhash,
            "meeting_id": meeting_id,
            "num_speakers": result.get("num_speakers", 0),
            "audio_duration": result.get("duration", 0),
            "speakers": {},
            "processed_at": datetime.now().isoformat(),
            "transcript_segments": 0,
        }
        save_manifest(manifest)
        return True

    # Save transcript (md + json + legacy hash.txt for meeting_server.py)
    transcript_md = format_transcript_md(merged)
    md_path = TRANSCRIPTS_DIR / f"{meeting_id}_transcript.md"
    md_path.write_text(transcript_md)
    json_path = TRANSCRIPTS_DIR / f"{meeting_id}_transcript.json"
    with open(json_path, "w") as f:
        json.dump({"segments": merged}, f, indent=2)
    (TRANSCRIPTS_DIR / f"{fhash}.txt").write_text(transcript_md)

    logger.info(
        f"  Saved transcript: {len(merged)} segments, "
        f"{result.get('num_identified', 0)}/{result.get('num_speakers', 0)} speakers identified"
    )

    # Update manifest
    speakers_in_meeting: dict[str, dict] = {}
    for seg in merged:
        spk = seg["speaker"]
        entry = speakers_in_meeting.setdefault(spk, {"total_time": 0.0, "segment_count": 0})
        entry["total_time"] += seg["end"] - seg["start"]
        entry["segment_count"] += 1
    for entry in speakers_in_meeting.values():
        entry["total_time"] = round(entry["total_time"], 1)

    meeting_data = existing.copy()
    meeting_data.update({
        "file": audio_path.name,
        "hash": fhash,
        "meeting_id": meeting_id,
        "num_speakers": result.get("num_speakers", len(speakers_in_meeting)),
        "num_identified": result.get("num_identified", 0),
        "audio_duration": result.get("duration", 0),
        "speakers": speakers_in_meeting,
        "processed_at": datetime.now().isoformat(),
        "transcript_file": str(md_path),
        "transcript_json": str(json_path),
        "transcript_segments": len(merged),
        "transcribed_at": datetime.now().isoformat(),
    })

    manifest["meetings"][fhash] = meeting_data
    save_manifest(manifest)
    return True


def summarize_meeting(manifest: dict, fhash: str) -> bool:
    """Summarize an already-transcribed meeting from its transcript file."""
    entry = manifest["meetings"][fhash]
    meeting_id = entry.get("meeting_id", fhash)

    transcript_ref = entry.get("transcript_file")
    if not transcript_ref:
        return False
    transcript_path = Path(transcript_ref)
    if not transcript_path.is_absolute():
        # Older manifest entries stored paths relative to the pipeline dir
        transcript_path = OUTPUT_DIR.parent / transcript_path
    if not transcript_path.exists():
        logger.warning(f"  Transcript file missing: {transcript_path}")
        return False

    transcript_md = transcript_path.read_text()
    if len(transcript_md.strip()) < 100:
        logger.info(f"  {meeting_id}: transcript too short for summarization")
        entry["summary_file"] = "too_short"
        save_manifest(manifest)
        return True

    logger.info(f"  {meeting_id}: summarizing with {LLM_MODEL}...")
    summary_text = summarize_transcript(transcript_md)
    if not summary_text:
        return False

    summary_path = SUMMARIES_DIR / f"{meeting_id}_summary.md"
    summary_path.write_text(summary_text)
    entry["summary_file"] = str(summary_path)
    entry["summarized_at"] = datetime.now().isoformat()
    save_manifest(manifest)
    logger.info(f"  {meeting_id}: summary saved ({len(summary_text)} chars)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Meeting pipeline (local diarization API)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N meetings (0=all)")
    parser.add_argument("--meeting", type=str, help="Process a specific meeting (substring of filename)")
    parser.add_argument("--force", action="store_true", help="Reprocess meetings already in the manifest")
    parser.add_argument("--since", type=str, metavar="YYYY-MM-DD",
                        help="Only meetings whose recording file date is on/after this day")
    parser.add_argument("--until", type=str, metavar="YYYY-MM-DD",
                        help="Only meetings whose recording file date is on/before this day")
    parser.add_argument("--summarize", action="store_true",
                        help="Also produce LLM summaries (requires LLM_API_URL)")
    args = parser.parse_args()

    summarize = args.summarize
    if summarize and not LLM_API_URL:
        logger.warning("--summarize requested but LLM_API_URL is not set; summaries will be skipped")
        summarize = False

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    client = api_client()
    if not check_api(client):
        return 1

    backup_manifest()
    manifest = load_manifest()

    audio_files = sorted(AUDIO_DIR.glob("*-MergedAudio.m4a"))
    logger.info(f"Found {len(audio_files)} audio files")

    audio_files = [f for f in audio_files if f.stat().st_size >= MIN_FILE_SIZE]
    logger.info(f"After min-size filter: {len(audio_files)} files")

    if args.since:
        cutoff = datetime.strptime(args.since, "%Y-%m-%d").timestamp()
        audio_files = [f for f in audio_files if os.path.getmtime(f) >= cutoff]
        logger.info(f"After --since {args.since}: {len(audio_files)} files")

    if args.until:
        end = datetime.strptime(args.until, "%Y-%m-%d").timestamp() + 86400  # inclusive day
        audio_files = [f for f in audio_files if os.path.getmtime(f) < end]
        logger.info(f"After --until {args.until}: {len(audio_files)} files")

    if args.meeting:
        audio_files = [f for f in audio_files if args.meeting in f.name]
        if not audio_files:
            logger.error(f"Meeting not found: {args.meeting}")
            return 1

    # -----------------------------------------------------------------
    # Phase 1: transcribe. The LLM is unloaded first — diarization peaks
    # at 10-16 GB of VRAM and would collide with a resident model.
    # -----------------------------------------------------------------
    unload_llm()

    processed = failed = skipped = 0
    retranscribed: set[str] = set()
    for audio_file in audio_files:
        if args.limit and processed + failed >= args.limit:
            break

        fhash = file_hash(str(audio_file))
        existing = manifest["meetings"].get(fhash, {})
        if existing.get("transcript_file") and not args.force:
            skipped += 1
            continue

        logger.info(f"\n[{processed + failed + 1}] {audio_file.name}")
        try:
            ok = process_meeting(client, audio_file, manifest)
        except Exception:
            logger.exception("  Unexpected failure")
            ok = False
        if ok:
            processed += 1
            retranscribed.add(fhash)
        else:
            failed += 1

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Transcription: {processed} processed, {skipped} already done, {failed} failed")

    # -----------------------------------------------------------------
    # Phase 2: summarize everything that has a transcript but no summary.
    # Running as a separate pass means the LLM loads once instead of
    # swapping in and out per meeting; meanwhile the api's idle timeout
    # frees the diarization models.
    # -----------------------------------------------------------------
    summarized = sum_failed = 0
    if summarize:
        for audio_file in audio_files:
            fhash = file_hash(str(audio_file))
            entry = manifest["meetings"].get(fhash, {})
            if not entry.get("transcript_file"):
                continue
            # Keep an existing summary only if it's newer than the transcript —
            # a fresh transcript deserves a fresh summary. Timestamp-based so it
            # also heals meetings from interrupted --force runs.
            fresh_summary = (
                entry.get("summary_file")
                and fhash not in retranscribed
                and (entry.get("summarized_at") or "") >= (entry.get("transcribed_at") or "")
            )
            if fresh_summary:
                continue
            try:
                ok = summarize_meeting(manifest, fhash)
            except Exception:
                logger.exception("  Summarization failure")
                ok = False
            if ok:
                summarized += 1
            else:
                sum_failed += 1
        logger.info(f"Summaries: {summarized} written, {sum_failed} failed")

    return 0 if failed == 0 and sum_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
