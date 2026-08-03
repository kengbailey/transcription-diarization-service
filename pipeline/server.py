#!/usr/bin/env python3
"""Simple server for the speaker evaluation UI."""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Speaker Evaluation UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/home/syran/sandbox/transcription-diarization-service/pipeline/output")
CLIPS_DIR = OUTPUT_DIR / "clips"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
STATIC_DIR = Path(__file__).parent / "static"


def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {"meetings": {}, "speakers": {}, "clusters": []}


def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/manifest")
async def get_manifest():
    return load_manifest()


@app.get("/api/clips/{clip_name}")
async def get_clip(clip_name: str):
    clip_path = CLIPS_DIR / clip_name
    if not clip_path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(clip_path, media_type="audio/ogg")


class MergeRequest(BaseModel):
    cluster_ids: list[int]
    name: str


@app.post("/api/clusters/merge")
async def merge_clusters(req: MergeRequest):
    """Merge multiple clusters into one with a name."""
    manifest = load_manifest()
    clusters = manifest.get("clusters", [])
    
    # Find clusters to merge
    to_merge = [c for c in clusters if c["id"] in req.cluster_ids]
    if len(to_merge) < 2:
        raise HTTPException(400, "Need at least 2 clusters to merge")
    
    # Merge members into first cluster
    merged = to_merge[0]
    merged["name"] = req.name
    for c in to_merge[1:]:
        merged["members"].extend(c["members"])
    
    # Remove merged clusters (keep the first)
    new_clusters = [c for c in clusters if c["id"] not in req.cluster_ids or c["id"] == merged["id"]]
    manifest["clusters"] = new_clusters
    save_manifest(manifest)
    
    return {"status": "ok", "cluster": merged}


class NameRequest(BaseModel):
    name: str


@app.post("/api/clusters/{cluster_id}/name")
async def name_cluster(cluster_id: int, req: NameRequest):
    """Name a speaker cluster."""
    manifest = load_manifest()
    for c in manifest.get("clusters", []):
        if c["id"] == cluster_id:
            c["name"] = req.name
            save_manifest(manifest)
            return {"status": "ok"}
    raise HTTPException(404, "Cluster not found")


@app.post("/api/clusters/{cluster_id}/split/{member_index}")
async def split_member(cluster_id: int, member_index: int):
    """Split a member out of a cluster into its own cluster."""
    manifest = load_manifest()
    clusters = manifest.get("clusters", [])
    
    for c in clusters:
        if c["id"] == cluster_id:
            if member_index >= len(c["members"]):
                raise HTTPException(400, "Invalid member index")
            member = c["members"].pop(member_index)
            new_id = max(cl["id"] for cl in clusters) + 1
            clusters.append({
                "id": new_id,
                "name": None,
                "members": [member]
            })
            save_manifest(manifest)
            return {"status": "ok", "new_cluster_id": new_id}
    
    raise HTTPException(404, "Cluster not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
