#!/usr/bin/env python3
"""Phase 3: Transcribe meetings with speaker identification, then summarize with LLM."""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DIARIZATION_API = "http://localhost:8008"
WHISPER_API = "http://localhost:8000/v1"
LLM_API = "http://192.168.8.147:9292/v1"
LLM_MODEL = "gpt-oss-120b"

MANIFEST_FILE = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output/manifest.json")
TRANSCRIPTS_DIR = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output/transcripts")
SUMMARIES_DIR = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output/summaries")
AUDIO_DIR = Path("/home/syran/tmp/mtgs/Users/kenbailey/Library/Application Support/com.taperlabs.shadow")


def load_manifest():
    with open(MANIFEST_FILE) as f:
        return json.load(f)


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def transcribe_meeting(audio_path: str, meeting_id: str) -> dict | None:
    """Transcribe a meeting with speaker identification using the diarization API."""
    client = httpx.Client(timeout=1800)  # 10 min timeout for large files
    
    file_size = os.path.getsize(audio_path)
    logger.info(f"Transcribing {meeting_id} ({file_size // 1024 // 1024}MB)")
    
    try:
        with open(audio_path, "rb") as f:
            resp = client.post(
                f"{DIARIZATION_API}/transcribe-identified",
                files={"file": (os.path.basename(audio_path), f, "audio/mp4")},
                data={"max_speakers": 10},
            )
        
        if resp.status_code != 200:
            logger.error(f"Transcription failed ({resp.status_code}): {resp.text[:200]}")
            return None
        
        return resp.json()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


def format_transcript(result: dict) -> str:
    """Format transcription result into readable text with speaker labels."""
    lines = []
    current_speaker = None
    current_text = []
    
    for seg in result.get("segments", []):
        speaker = seg.get("identified_as") or seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        
        if not text:
            continue
        
        if speaker != current_speaker:
            if current_speaker and current_text:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)
    
    # Don't forget the last segment
    if current_speaker and current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return "\n\n".join(lines)


def summarize_meeting(transcript: str, meeting_id: str) -> str | None:
    """Summarize a meeting transcript using the LLM."""
    client = httpx.Client(timeout=600)  # 5 min timeout
    
    # Truncate very long transcripts to fit context
    max_chars = 80000  # ~20k tokens
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED]"
    
    prompt = f"""You are a meeting summarizer. Analyze this meeting transcript and provide a structured summary.

TRANSCRIPT:
{transcript}

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
            logger.error(f"LLM summarization failed ({resp.status_code}): {resp.text[:200]}")
            return None
        
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


def process_meeting(meeting_id: str, meeting: dict, manifest: dict) -> bool:
    """Process a single meeting: transcribe then summarize."""
    audio_file = meeting["file"]
    audio_path = AUDIO_DIR / audio_file
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return False
    
    # Skip if already transcribed and summarized
    if meeting.get("transcript_file") and meeting.get("summary_file"):
        logger.info(f"Skipping {meeting_id} (already done)")
        return True
    
    # Step 1: Transcribe
    transcript_file = TRANSCRIPTS_DIR / f"{meeting_id}.txt"
    if transcript_file.exists() and meeting.get("transcript_file"):
        logger.info(f"Transcript exists, loading: {transcript_file}")
        transcript_text = transcript_file.read_text()
        result = None
    else:
        result = transcribe_meeting(str(audio_path), meeting_id)
        if not result:
            return False
        
        transcript_text = format_transcript(result)
        
        # Save transcript
        transcript_file.write_text(transcript_text)
        
        # Also save raw JSON
        raw_file = TRANSCRIPTS_DIR / f"{meeting_id}.json"
        with open(raw_file, "w") as f:
            json.dump(result, f, indent=2)
        
        meeting["transcript_file"] = str(transcript_file)
        meeting["transcript_segments"] = len(result.get("segments", []))
        meeting["transcript_speakers"] = result.get("num_speakers", 0)
        meeting["transcript_identified"] = result.get("num_identified", 0)
        meeting["transcribed_at"] = datetime.now().isoformat()
        save_manifest(manifest)
        
        logger.info(f"  Transcribed: {len(result.get('segments', []))} segments, "
                     f"{result.get('num_identified', 0)}/{result.get('num_speakers', 0)} speakers identified")
    
    # Skip summarization for very short transcripts
    if len(transcript_text.strip()) < 100:
        logger.info(f"  Transcript too short for summarization, skipping")
        meeting["summary_file"] = "too_short"
        save_manifest(manifest)
        return True
    
    # Step 2: Summarize
    summary_file = SUMMARIES_DIR / f"{meeting_id}_summary.md"
    if summary_file.exists() and meeting.get("summary_file"):
        logger.info(f"Summary exists, skipping: {summary_file}")
        return True
    
    logger.info(f"  Summarizing with {LLM_MODEL}...")
    summary = summarize_meeting(transcript_text, meeting_id)
    if not summary:
        logger.warning(f"  Summarization failed, transcript saved")
        return True  # Still a partial success
    
    summary_file.write_text(summary)
    meeting["summary_file"] = str(summary_file)
    meeting["summarized_at"] = datetime.now().isoformat()
    save_manifest(manifest)
    
    logger.info(f"  Summary saved ({len(summary)} chars)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize meetings")
    parser.add_argument("--limit", type=int, default=0, help="Max meetings to process (0=all)")
    parser.add_argument("--skip-summary", action="store_true", help="Only transcribe, skip summarization")
    parser.add_argument("--meeting", type=str, help="Process a specific meeting ID")
    args = parser.parse_args()
    
    # Create output dirs
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = load_manifest()
    meetings = manifest["meetings"]
    
    if args.meeting:
        if args.meeting in meetings:
            process_meeting(args.meeting, meetings[args.meeting], manifest)
        else:
            logger.error(f"Meeting {args.meeting} not found")
        return
    
    # Process all meetings
    total = len(meetings)
    processed = 0
    skipped = 0
    failed = 0
    
    for i, (mid, meeting) in enumerate(meetings.items()):
        if args.limit and processed >= args.limit:
            break
        
        logger.info(f"\n[{i+1}/{total}] {meeting['file']}")
        
        # Skip tiny files
        audio_path = AUDIO_DIR / meeting["file"]
        if audio_path.exists() and audio_path.stat().st_size < 500000:
            logger.info(f"  Skipping (too small)")
            skipped += 1
            continue
        
        success = process_meeting(mid, meeting, manifest)
        if success:
            processed += 1
        else:
            failed += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Done! Processed: {processed}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
