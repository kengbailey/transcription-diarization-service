#!/usr/bin/env python3
"""Unified meeting pipeline using Speaches AI for diarization and transcription."""

import argparse
import base64
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
SPEACHES_API = "http://192.168.8.147:8000"
LLM_API = "http://192.168.8.147:9292/v1"
LLM_MODEL = "gpt-oss-20b"
AUDIO_DIR = Path(
    "/home/syran/tmp/mtgs/Users/kenbailey/Library/Application Support/com.taperlabs.shadow"
)
OUTPUT_DIR = Path("output")
SPEAKER_REFS_DIR = OUTPUT_DIR / "speaker_references"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

MIN_FILE_SIZE = 500_000  # 500KB — skip tiny files
MAX_FILE_SIZE = 15_000_000  # 15MB — skip large files to avoid GPU OOM
REQUEST_TIMEOUT = 600.0  # seconds — generous timeout
INTER_STEP_DELAY = 2.0  # seconds — must exceed model TTL (30s) so models fully unload
INTER_MEETING_DELAY = 2.0  # seconds between meetings
MAX_SPEAKER_REFS = 10     # limit speaker references to reduce VRAM pressure during diarization


# ---------------------------------------------------------------------------
# Manifest helpers
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
    if MANIFEST_FILE.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MANIFEST_FILE.with_suffix(f".{ts}.bak.json")
        shutil.copy2(MANIFEST_FILE, backup)
        logger.info(f"Manifest backed up to {backup}")


def file_hash(filepath: str) -> str:
    stat = os.stat(filepath)
    key = f"{os.path.basename(filepath)}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Speaker references
# ---------------------------------------------------------------------------
def load_speaker_references() -> dict[str, Path]:
    """Load known speaker references — returns {name: path_to_longest_clip}."""
    refs: dict[str, Path] = {}
    if not SPEAKER_REFS_DIR.is_dir():
        logger.info("No speaker references directory found, running without known speakers")
        return refs

    for speaker_dir in sorted(SPEAKER_REFS_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue
        name = speaker_dir.name
        clips = list(speaker_dir.glob("*.ogg"))
        if not clips:
            continue
        longest = max(clips, key=lambda p: p.stat().st_size)
        refs[name] = longest
        logger.info(f"  Speaker ref: {name} -> {longest.name}")

    logger.info(f"Loaded {len(refs)} speaker references")
    return refs


def build_speaker_form_data(refs: dict[str, Path], max_refs: int = MAX_SPEAKER_REFS) -> dict[str, list]:
    """Build form fields for the diarization endpoint's known-speaker parameters.
    
    Limits to max_refs speakers (sorted by clip size descending — larger clips = better refs).
    """
    # Sort by clip size descending (bigger = more audio = better embedding)
    sorted_refs = sorted(refs.items(), key=lambda x: x[1].stat().st_size, reverse=True)
    if len(sorted_refs) > max_refs:
        logger.info(f"  Limiting speaker refs from {len(sorted_refs)} to {max_refs}")
        sorted_refs = sorted_refs[:max_refs]

    names: list[str] = []
    data_urls: list[str] = []
    for name, clip_path in sorted_refs:
        raw = clip_path.read_bytes()
        b64 = base64.b64encode(raw).decode()
        names.append(name)
        data_urls.append(f"data:audio/ogg;base64,{b64}")
    return {"names": names, "data_urls": data_urls}


# ---------------------------------------------------------------------------
# Speaches API calls
# ---------------------------------------------------------------------------
def _wait_for_speaches(timeout: int = 120) -> bool:
    """Wait for speaches to be healthy. Returns True if healthy."""
    for i in range(timeout // 5):
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(f"{SPEACHES_API}/health")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        if i % 6 == 0 and i > 0:
            logger.warning(f"  Speaches not ready, waited {i * 5}s...")
        time.sleep(5)
    return False


def diarize(audio_path: Path, speaker_refs: dict[str, Path] | None) -> dict | None:
    """Call Speaches diarization endpoint. Returns JSON with duration + segments."""
    size_mb = audio_path.stat().st_size // 1024 // 1024
    logger.info(f"  Diarizing {audio_path.name} ({size_mb}MB)...")

    data: dict[str, str | list[str]] = {}
    if speaker_refs:
        form = build_speaker_form_data(speaker_refs)
        data["known_speaker_names[]"] = form["names"]
        data["known_speaker_references[]"] = form["data_urls"]
        logger.info(f"  Sending {len(form['names'])} speaker refs")

    for attempt in range(2):
        try:
            with open(audio_path, "rb") as f:
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    resp = client.post(
                        f"{SPEACHES_API}/v1/audio/diarization",
                        files={"file": (audio_path.name, f, "audio/mp4")},
                        data=data,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    n_seg = len(result.get("segments", []))
                    dur = result.get("duration", 0)
                    logger.info(f"  Diarization done: {n_seg} segments, {dur:.0f}s duration")
                    return result
        except Exception as e:
            logger.error(f"  Diarization failed (attempt {attempt + 1}): {e}")
            if attempt == 0:
                # Try without speaker refs on retry (less VRAM)
                logger.info("  Retrying diarization without speaker refs...")
                data = {}
                if not _wait_for_speaches():
                    logger.error("  Speaches unresponsive, giving up on diarization")
                    return None
    return None


def transcribe(audio_path: Path) -> dict | None:
    """Call Speaches transcription endpoint. Returns JSON with segments."""
    logger.info(f"  Transcribing {audio_path.name}...")
    for attempt in range(2):
        try:
            with open(audio_path, "rb") as f:
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    resp = client.post(
                        f"{SPEACHES_API}/v1/audio/transcriptions",
                        files={"file": (audio_path.name, f, "audio/mp4")},
                        data={
                            "model": "Systran/faster-distil-whisper-large-v3",
                            "response_format": "verbose_json",
                        },
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    n_seg = len(result.get("segments", []))
                    logger.info(f"  Transcription done: {n_seg} segments")
                    return result
        except Exception as e:
            logger.error(f"  Transcription failed (attempt {attempt + 1}): {e}")
            if attempt == 0:
                if not _wait_for_speaches():
                    logger.error("  Speaches unresponsive, giving up on transcription")
                    return None
    return None


# ---------------------------------------------------------------------------
# Merging diarization + transcription
# ---------------------------------------------------------------------------
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute overlap duration between two intervals."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def merge_diarization_transcription(
    diar_segments: list[dict],
    whisper_segments: list[dict],
) -> list[dict]:
    """Align whisper text segments with diarization speaker labels by timestamp overlap.

    Returns list of {"start", "end", "speaker", "text"} dicts, grouped by
    consecutive same-speaker runs.
    """
    # Assign speaker to each whisper segment
    labeled: list[dict] = []
    for wseg in whisper_segments:
        w_start = wseg.get("start", 0.0)
        w_end = wseg.get("end", w_start)
        text = wseg.get("text", "").strip()
        if not text:
            continue

        best_speaker = "Unknown"
        best_overlap = 0.0
        for dseg in diar_segments:
            ov = _overlap(w_start, w_end, dseg["start"], dseg["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dseg["speaker"]

        labeled.append({
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "speaker": best_speaker,
            "text": text,
        })

    # Group consecutive segments by same speaker
    if not labeled:
        return []

    grouped: list[dict] = [labeled[0].copy()]
    for seg in labeled[1:]:
        prev = grouped[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = seg["end"]
            prev["text"] += " " + seg["text"]
        else:
            grouped.append(seg.copy())

    return grouped


def merge_transcription_only(whisper_segments: list[dict]) -> list[dict]:
    """When diarization is unavailable, just return whisper segments as-is."""
    result: list[dict] = []
    for wseg in whisper_segments:
        text = wseg.get("text", "").strip()
        if not text:
            continue
        result.append({
            "start": round(wseg.get("start", 0.0), 3),
            "end": round(wseg.get("end", 0.0), 3),
            "speaker": "Unknown",
            "text": text,
        })
    return result


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------
def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_transcript_md(segments: list[dict]) -> str:
    lines: list[str] = []
    for seg in segments:
        ts = f"{_fmt_time(seg['start'])} - {_fmt_time(seg['end'])}"
        lines.append(f"**{seg['speaker']}** ({ts}):\n{seg['text']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# LLM summarization (same approach as transcribe_and_summarize.py)
# ---------------------------------------------------------------------------
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
                f"{LLM_API}/chat/completions",
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
    audio_path: Path,
    manifest: dict,
    speaker_refs: dict[str, Path] | None,
    skip_summary: bool = False,
) -> bool:
    """Process a single meeting: diarize, transcribe, merge, summarize, save."""
    fhash = file_hash(str(audio_path))
    meeting_id = audio_path.name.replace("-MergedAudio.m4a", "")[:16]

    existing = manifest["meetings"].get(fhash, {})
    if existing.get("transcript_file") and (existing.get("summary_file") or skip_summary):
        logger.info(f"Skipping already-processed: {audio_path.name}")
        return True

    logger.info(f"Processing {audio_path.name} (id={meeting_id})")

    # 1. Diarize
    diar_result = diarize(audio_path, speaker_refs)
    diar_segments = diar_result.get("segments", []) if diar_result else []

    # Wait between GPU-heavy operations to let models unload
    logger.info(f"  Waiting {INTER_STEP_DELAY}s for GPU cooldown...")
    time.sleep(INTER_STEP_DELAY)

    # 2. Transcribe
    whisper_result = transcribe(audio_path)
    if not whisper_result:
        logger.error(f"  Transcription failed, skipping meeting")
        return False

    whisper_segments = whisper_result.get("segments", [])

    # 3. Merge
    if diar_segments:
        merged = merge_diarization_transcription(diar_segments, whisper_segments)
    else:
        logger.warning("  No diarization segments, using transcription only")
        merged = merge_transcription_only(whisper_segments)

    if not merged:
        logger.warning("  No merged segments produced")
        return False

    # 4. Save transcript
    transcript_md = format_transcript_md(merged)
    md_path = TRANSCRIPTS_DIR / f"{meeting_id}_transcript.md"
    md_path.write_text(transcript_md)

    json_path = TRANSCRIPTS_DIR / f"{meeting_id}_transcript.json"
    with open(json_path, "w") as f:
        json.dump({"segments": merged}, f, indent=2)

    # Also save a plain .txt for backward compat with meeting_server.py
    txt_path = TRANSCRIPTS_DIR / f"{fhash}.txt"
    txt_path.write_text(transcript_md)

    logger.info(f"  Saved transcript: {len(merged)} segments")

    # 5. Summarize
    summary_text = None
    summary_path = SUMMARIES_DIR / f"{meeting_id}_summary.md"
    if skip_summary:
        logger.info("  Skipping summarization (--skip-summary)")
    elif len(transcript_md.strip()) < 100:
        logger.info("  Transcript too short for summarization")
    else:
        logger.info(f"  Summarizing with {LLM_MODEL}...")
        summary_text = summarize_transcript(transcript_md)
        if summary_text:
            summary_path.write_text(summary_text)
            logger.info(f"  Summary saved ({len(summary_text)} chars)")
        else:
            logger.warning("  Summarization failed")

    # 6. Update manifest
    speakers_in_meeting = {}
    for seg in merged:
        spk = seg["speaker"]
        if spk not in speakers_in_meeting:
            speakers_in_meeting[spk] = {"total_time": 0.0, "segment_count": 0}
        speakers_in_meeting[spk]["total_time"] += seg["end"] - seg["start"]
        speakers_in_meeting[spk]["segment_count"] += 1

    for spk in speakers_in_meeting:
        speakers_in_meeting[spk]["total_time"] = round(speakers_in_meeting[spk]["total_time"], 1)

    meeting_data = existing.copy()
    meeting_data.update({
        "file": audio_path.name,
        "hash": fhash,
        "meeting_id": meeting_id,
        "num_speakers": len(speakers_in_meeting),
        "audio_duration": diar_result.get("duration", 0) if diar_result else 0,
        "speakers": speakers_in_meeting,
        "processed_at": datetime.now().isoformat(),
        "transcript_file": str(md_path),
        "transcript_json": str(json_path),
        "transcript_segments": len(merged),
        "transcribed_at": datetime.now().isoformat(),
    })

    if summary_text:
        meeting_data["summary_file"] = str(summary_path)
        meeting_data["summarized_at"] = datetime.now().isoformat()
    elif skip_summary:
        pass  # leave summary fields as-is
    elif len(transcript_md.strip()) < 100:
        meeting_data["summary_file"] = "too_short"

    manifest["meetings"][fhash] = meeting_data
    save_manifest(manifest)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Meeting pipeline (Speaches AI)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N meetings (0=all)")
    parser.add_argument("--skip-summary", action="store_true", help="Only transcribe, skip LLM summarization")
    parser.add_argument("--meeting", type=str, help="Process a specific meeting ID")
    parser.add_argument("--no-speakers", action="store_true", help="Skip known speaker identification")
    args = parser.parse_args()

    # Ensure output directories exist
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    SPEAKER_REFS_DIR.mkdir(parents=True, exist_ok=True)

    backup_manifest()
    manifest = load_manifest()

    # Load speaker references
    speaker_refs = None
    if not args.no_speakers:
        speaker_refs = load_speaker_references()
        if not speaker_refs:
            speaker_refs = None

    # Find audio files
    audio_files = sorted(AUDIO_DIR.glob("*-MergedAudio.m4a"))
    logger.info(f"Found {len(audio_files)} audio files")

    # Filter by size
    too_small = [f for f in audio_files if f.stat().st_size < MIN_FILE_SIZE]
    too_large = [f for f in audio_files if f.stat().st_size > MAX_FILE_SIZE]
    audio_files = [f for f in audio_files if MIN_FILE_SIZE <= f.stat().st_size <= MAX_FILE_SIZE]
    logger.info(f"After size filter: {len(audio_files)} files ({len(too_small)} too small, {len(too_large)} too large)")
    if too_large:
        logger.info(f"  Skipped large files (>{MAX_FILE_SIZE // 1_000_000}MB): {[f.name for f in too_large[:5]]}...")

    # Filter to specific meeting if requested
    if args.meeting:
        audio_files = [f for f in audio_files if args.meeting in f.name]
        if not audio_files:
            logger.error(f"Meeting not found: {args.meeting}")
            return

    if args.limit:
        audio_files = audio_files[: args.limit]

    # Process
    processed = 0
    failed = 0
    skipped = 0

    for i, audio_file in enumerate(audio_files):
        logger.info(f"\n[{i + 1}/{len(audio_files)}] {audio_file.name}")
        fhash = file_hash(str(audio_file))
        existing = manifest["meetings"].get(fhash, {})
        if existing.get("transcript_file") and (existing.get("summary_file") or args.skip_summary):
            skipped += 1
            logger.info(f"  Already processed, skipping")
            continue

        # Health check before processing
        if not _wait_for_speaches(60):
            logger.error("  Speaches unresponsive, stopping pipeline.")
            break

        ok = process_meeting(audio_file, manifest, speaker_refs, skip_summary=args.skip_summary)
        if ok:
            processed += 1
        else:
            failed += 1

        # Delay between meetings to avoid GPU overload
        if i < len(audio_files) - 1:
            time.sleep(INTER_MEETING_DELAY)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Done! Processed: {processed}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
