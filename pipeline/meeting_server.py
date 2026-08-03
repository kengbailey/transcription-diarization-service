"""Meeting Review UI - Browse meetings, play audio, view summaries, ask questions."""

import json
import os
from pathlib import Path
from datetime import datetime as dt_cls
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Meeting Review")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output")
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
AUDIO_DIR = Path("/home/syran/tmp/mtgs/Users/kenbailey/Library/Application Support/com.taperlabs.shadow")
STATIC_DIR = Path(__file__).parent / "static"

LLM_API = "http://192.168.8.147:9292/v1"
LLM_MODEL = "gpt-oss-120b"


def _get_file_date(path, fmt="date"):
    try:
        mtime = os.path.getmtime(path)
        d = dt_cls.fromtimestamp(mtime)
        if fmt == "date": return d.strftime("%Y-%m-%d")
        if fmt == "time": return d.strftime("%H:%M")
        return d.isoformat()
    except Exception:
        return None


def load_manifest():
    with open(MANIFEST_FILE) as f:
        return json.load(f)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "meetings.html")


@app.get("/api/meetings")
async def list_meetings():
    manifest = load_manifest()
    meetings = []
    for mid, m in manifest["meetings"].items():
        audio_path = AUDIO_DIR / m["file"]
        meetings.append({
            "id": mid,
            "file": m["file"],
            "num_speakers": m.get("num_speakers", 0),
            "audio_duration": m.get("audio_duration", 0),
            "has_transcript": bool(m.get("transcript_file")),
            "has_summary": bool(m.get("summary_file") and m["summary_file"] != "too_short"),
            "transcript_identified": m.get("transcript_identified", 0),
            "transcribed_at": m.get("transcribed_at"),
            "summarized_at": m.get("summarized_at"),
            "audio_size_mb": round(audio_path.stat().st_size / 1024 / 1024, 1) if audio_path.exists() else 0,
            "meeting_date": _get_file_date(audio_path, "date"),
            "meeting_time": _get_file_date(audio_path, "time"),
            "meeting_datetime": _get_file_date(audio_path, "iso"),
        })
    meetings.sort(key=lambda x: x.get("meeting_datetime") or "", reverse=True)
    return {"meetings": meetings, "total": len(meetings)}


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    manifest = load_manifest()
    m = manifest["meetings"].get(meeting_id)
    if not m:
        raise HTTPException(404, "Meeting not found")

    # Load transcript — try new format first, then legacy
    transcript = None
    mid_short = m.get("meeting_id", "")
    for candidate in [
        TRANSCRIPTS_DIR / f"{mid_short}_transcript.md",
        TRANSCRIPTS_DIR / f"{meeting_id}.txt",
    ]:
        if candidate.exists():
            transcript = candidate.read_text()
            break

    # Load raw transcript JSON for segments — try new format first, then legacy
    segments = None
    for candidate in [
        TRANSCRIPTS_DIR / f"{mid_short}_transcript.json",
        TRANSCRIPTS_DIR / f"{meeting_id}.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                raw = json.load(f)
                segments = raw.get("segments", [])
            break

    # Load summary — try new format (short meeting_id) first, then legacy (hash-based)
    summary = None
    for candidate in [
        SUMMARIES_DIR / f"{mid_short}_summary.md",
        SUMMARIES_DIR / f"{meeting_id}_summary.md",
    ]:
        if candidate.exists():
            summary = candidate.read_text()
            break

    # Get attendees from transcript segments (handles both old and new format)
    attendees = []
    if segments:
        seen = {}
        for seg in segments:
            name = seg.get("identified_as") or seg.get("speaker", "Unknown")
            if name not in seen:
                seen[name] = 0
            # New format: compute duration from start/end; old format: use duration field
            dur = seg.get("duration", 0)
            if not dur and "start" in seg and "end" in seg:
                dur = seg["end"] - seg["start"]
            seen[name] += dur
        attendees = [{"name": n, "talk_time": round(t, 1)} for n, t in sorted(seen.items(), key=lambda x: -x[1])]

    return {
        "id": meeting_id,
        "file": m["file"],
        "num_speakers": m.get("num_speakers", 0),
        "audio_duration": m.get("audio_duration", 0),
        "transcript": transcript,
        "summary": summary,
        "attendees": attendees,
        "has_audio": (AUDIO_DIR / m["file"]).exists(),
        "meeting_date": _get_file_date(AUDIO_DIR / m["file"], "date"),
        "meeting_time": _get_file_date(AUDIO_DIR / m["file"], "time"),
    }


@app.get("/api/meetings/{meeting_id}/audio")
async def stream_audio(meeting_id: str):
    manifest = load_manifest()
    m = manifest["meetings"].get(meeting_id)
    if not m:
        raise HTTPException(404, "Meeting not found")

    audio_path = AUDIO_DIR / m["file"]
    if not audio_path.exists():
        raise HTTPException(404, "Audio file not found")

    return FileResponse(audio_path, media_type="audio/mp4", filename=m["file"])


class QuestionRequest(BaseModel):
    question: str
    meeting_id: str


@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    manifest = load_manifest()
    m = manifest["meetings"].get(req.meeting_id)
    if not m:
        raise HTTPException(404, "Meeting not found")

    # Load transcript — try new format first, then legacy
    mid_short = m.get("meeting_id", "")
    transcript = None
    for candidate in [
        TRANSCRIPTS_DIR / f"{mid_short}_transcript.md",
        TRANSCRIPTS_DIR / f"{req.meeting_id}.txt",
    ]:
        if candidate.exists():
            transcript = candidate.read_text()
            break
    if not transcript:
        raise HTTPException(400, "No transcript available for this meeting")

    # Load summary if available — try both naming conventions
    summary = ""
    for candidate in [
        SUMMARIES_DIR / f"{mid_short}_summary.md",
        SUMMARIES_DIR / f"{req.meeting_id}_summary.md",
    ]:
        if candidate.exists():
            summary = f"\n\nMEETING SUMMARY:\n{candidate.read_text()}"
            break

    # Truncate if too long
    max_chars = 80000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n[TRUNCATED]"

    prompt = f"""You are a helpful meeting assistant. Answer the user's question based on this meeting transcript and summary.

TRANSCRIPT:
{transcript}
{summary}

USER QUESTION: {req.question}

Answer concisely based only on what's in the transcript. If the answer isn't in the transcript, say so."""

    try:
        client = httpx.Client(timeout=120)
        resp = client.post(
            f"{LLM_API}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"LLM error: {resp.status_code}")

        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer}
    except httpx.TimeoutException:
        raise HTTPException(504, "LLM request timed out")
    except Exception as e:
        raise HTTPException(502, f"LLM error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
