# Project Plan — LAN Transcription & Diarization Service

*Written 2026-08-03, after a four-agent audit (API backend, pipeline scripts, UI, and a web-verified model/package research pass). Context: development stalled on what looked like CUDA errors; the 3090 has since been undervolted and runs reliably.*

> **Direction update (2026-08-03):** Focus is the **`api/` service only** until it's solid — it will be consumed from other machines, so it comes first. The pipeline/meetings work (Phase 4) is deferred until then. **All compute runs on this box's 3090**; the 192.168.8.147 box is down and out of the picture. VRAM budget confirmed: Parakeet (~2–3 GB) + pyannote with #1963 mitigations (~4 GB peak) fits with ~16 GB headroom — enough to later co-host a summarization LLM. The local `speaches` container (0.8.3-cuda, port 8000, STT_MODEL_TTL=15) becomes the interim ASR endpoint: repoint `whisper_api_url` from `192.168.8.116` to `localhost:8000/v1`. Note 0.8.3 predates Speaches' Parakeet support — the Parakeet path adds a `parakeet.cpp-server` container (preferred) or a Speaches 0.9-RC upgrade. Pipeline audio in the environment can be used for testing the API.

## Where things stand

Two parallel architectures exist in this repo:

1. **`api/` + `ui/` (dockerized)** — the LAN-accessible service: FastAPI + pyannote 4.0.2 (`community-1` model) + wespeaker embeddings + Qdrant + external Whisper. This is the stated goal of the project.
2. **`pipeline/` (untracked)** — batch meeting-processing scripts. The newest, `pipeline_speaches.py`, moved all GPU compute to a Speaches + LLM server on another LAN box (192.168.8.147) and has processed 50/136 meetings. `meeting_server.py` (port 8502) is the day-to-day review/Q&A product.

**Recommended direction:** make `api/` on this box (the 3090) the production LAN service, and make `pipeline/` a first-class client of it. The Speaches box remains available as an ASR backend, but speaker identity (Qdrant) and diarization should have one owner: `api/`.

## Headline discovery: the CUDA story isn't over

The undervolt fixed real hardware load spikes, but two *software* contributors were found that plausibly compounded the instability:

- **pyannote-audio issue #1963 (open, affects 4.0.x):** the final partial batch of the embedding stage triggers a cuDNN workspace selection that transiently allocates **10–12 GB VRAM** on long files (vs ~1.6 GB in 3.x). On a 24 GB card sharing space with an ASR model, this alone produces intermittent CUDA failures. Mitigations: tune `embedding_batch_size` (ships as 32) so typical chunk counts divide evenly / lower it, `torch.backends.cudnn.benchmark = False`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`extract_embedding_from_segment()` re-decodes the entire audio file from disk once per diarized segment** (`api/services/embedding.py:128-168` via the loop in `extract_embeddings_for_segments`). A 150-turn meeting = 150 full decodes — bursty I/O + host-to-GPU transfer, and a major latency hit.

Treat both as part of Phase 1, and don't fully credit the undervolt until they're addressed.

---

## Phase 0 — Housekeeping & commit hygiene (half a day)

Get the working tree clean and truthful before building on it.

1. **`.gitignore`:** add `pipeline/output/` (audio clips, transcripts, manifests, 18 unpruned `manifest.*.bak.json`). Root patterns already catch `.venv/` and `__pycache__/`. Then commit the `pipeline/` scripts themselves — they're real source.
2. **Strip debug flags from `docker-compose.yml`:** `CUDA_LAUNCH_BLOCKING=1` is a permanent throughput tax now that the hardware is fixed; `TORCH_USE_CUDA_DSA=1` is a no-op on stock wheels. Move both to a `docker-compose.debug.yml` override.
3. **Add `PYANNOTE_METRICS_ENABLED=false`** — pyannote 4.x phones home (OTLP to `otel.pyannote.ai`) by default.
4. **Resolve the Qdrant collection rename.** `docker-compose.yml` now says `work_speaker_embeddings` but `config.py` defaults to `speaker_embeddings` and **both collections exist on disk** — previously registered speakers are invisible after the rename. Decide (migrate old vectors or revert the rename), and fix `pipeline/cluster_speakers.py:17` which still hardcodes the old name.
5. **Commit the CUDA-recovery work — but slimmed (see Phase 1.4).**
6. **Fix doc drift:** README says pyannote 3.1; AGENTS.md says 4.0 + `speaker-diarization-3.1`; reality is 4.0.2 + `community-1` on GPU and `3.1` on the CPU compose. Make the docs match reality and align the two compose files.

## Phase 1 — Correctness & stability (2–4 days) — ✅ done 2026-08-03 (commits 23680ed, 518b205, d3cf9a5)

> Verified against real meeting audio: exclusive diarization has zero overlaps, centroid-based identification matched 5/5 speakers and adds ~5s to a 16-minute meeting (was one GPU pass per segment), full transcribe-identified works via the local speaches container. VRAM peaked at 10.2 GB during a 16-min job with EMBEDDING_BATCH_SIZE=16.
>
> **Soak test passed:** 2.55-hour meeting (9172s) through /transcribe-identified in 19m18s (RTF 0.126) — 3 speakers, 3/3 identified, 514 transcript segments. VRAM peaked at 12.5 GB with speaches' whisper model co-resident (11.5 GB headroom). Zero CUDA errors, zero retries triggered. Single-box operation on the 3090 is confirmed stable with the #1963 mitigations.

The biggest wins are *deletions*: pyannote 4.x now provides outputs that replace two chunks of custom code, each of which currently has a live bug.

1. **Adopt `output.exclusive_speaker_diarization`.** Replaces `DiarizationService._make_exclusive()` (`diarization.py:269-301`), which has a real bug: when speaker B interjects inside A's turn, A's remainder is silently dropped — affects `/identify` and both transcribe endpoints. It also fixes `transcript_merger.py`'s nondeterministic first-match-wins speaker assignment in overlap regions. pyannote built this output specifically for STT reconciliation.
2. **Adopt `output.speaker_embeddings`.** The pipeline already returns per-speaker centroids in the raw 256-dim wespeaker space — directly compatible with existing Qdrant vectors. This deletes the second GPU forward pass in `embedding.py` for the identify path. Gotcha: pyannote zero-pads when labels > centroids; filter zero-norm rows before upserting.
3. **Fix the per-segment full-file decode** for the remaining embedding paths (speaker registration): load the waveform once per request, slice tensors per segment.
4. **Slim the CUDA-recovery code** (keep as defense-in-depth): consolidate the thrice-copy-pasted retry block into one decorator in `cuda_recovery.py`; match `torch.cuda.OutOfMemoryError` by type plus a narrower keyword set; add a short backoff; **fail fast on context-corrupting errors** ("device-side assert", "illegal memory access") and rely on `restart: unless-stopped` — in-process recovery from a corrupted CUDA context can return silently wrong results. Log loudly if reinit falls back to CPU.
5. **Apply the #1963 mitigations** (above) and verify with a long (>1 hr) meeting file while watching `nvidia-smi`.
6. **Version bumps:** `pyannote.audio` 4.0.2 → **4.0.7** (patch-level, no API break), `torchcodec>=0.7.0` (4.0.7 requires it), Qdrant image → `v1.18.3`, keep `torch==2.8.0` (satisfies pyannote; sm_86 has a long runway — even torch 2.13 supports Ampere). Pin the currently-floating deps (`fastapi`, `qdrant-client`, `numpy`, …) for reproducible builds.

## Phase 2 — Make it a real multi-client LAN service (1–2 weeks) — ✅ done 2026-08-03 (commits de69cfb, 5a1148e, 3d80aab)

> All items delivered and verified live: /health answers in 4–8ms during a running transcription (was blocked for the whole job); job queue processes FIFO with queue positions (`POST /jobs/{kind}` → 202 + job id, `GET /jobs/{id}` to poll); 401/413 paths verified in an isolated container; startup sweep removed the January crash orphan; final regression on the shipped images passed (5/5 speakers, 98s for a 16-min meeting). Auth is OFF by default — set API_KEY in .env to enable.

The audit's most important architectural finding: **every endpoint is `async def` but runs blocking GPU inference inline, so one job freezes the entire server** — including `/health` — for its duration (up to the 300s Whisper timeout). For multi-machine LAN use this is the core problem.

1. **Immediate fix:** wrap inference in `await asyncio.to_thread(...)` with an `asyncio.Semaphore(1)` guarding the GPU. Event loop stays responsive; GPU work is explicitly serialized instead of accidentally.
2. **Right fix: a job queue for long audio.** `POST /transcribe-identified` returns a `job_id` immediately; a single background worker processes the queue; `GET /jobs/{id}` for polling. Gives LAN clients timeout-immunity, reconnect-ability, queue-position feedback, and a natural place for concurrency limits. (This also lets `pipeline_speaches.py`-style batch runs just enqueue.)
3. **Enforce `max_upload_size`** in the API itself (`config.py:28` defines it; nothing checks it) — port 8008 is exposed directly, bypassing nginx's 500M limit. Check Content-Length and stream-abort past the limit.
4. **Lightweight auth:** shared bearer token via a FastAPI dependency. Destructive endpoints (`DELETE /speakers/*`) are currently open to any LAN device.
5. **Tighten CORS** (drop `allow_credentials=True` with wildcard origins) and stop returning raw exception text in 500s.
6. **Ops hygiene:** periodic sweep of orphaned uploads (one already sits in `data/uploads/` from a crash); `/health` reports GPU memory + queue depth; request-ID/timing middleware.
7. **UI LAN fixes:**
   - Settings quick links hardcode `http://localhost:8000` — wrong host from any LAN client and wrong port (8008) even locally. Use relative links.
   - Transcript copy uses `navigator.clipboard`, which is unavailable on plain `http://<lan-ip>` (non-secure context) — add a fallback or serve HTTPS.
   - Add a cancel (`AbortController`) for in-flight transcriptions.

## Phase 3 — Model upgrades (evaluate, ~1 week elapsed) — ✅ done 2026-08-03 (commits 347a652, 6fe6ecd)

> **ASR step 1 done:** speaches switched to full `faster-whisper-large-v3` (136s for the 16-min meeting, +9% vs distil, better word alignment, multilingual). Kept as the documented fallback.
>
> **ASR step 2 done and promoted to default:** `parakeet-tdt-0.6b-v3` via a `parakeet` compose service (parakeet.cpp CUDA). Measured on the 3090: 16.2 min of audio transcribed in **2.7s** (~360× real-time) with native word timestamps and properly punctuated text. End-to-end transcribe-identified: **91s** (was 136s whisper-large-v3 / 125s distil), identical 5/5 speaker identification. WAV-only limitation handled by an opt-in transcode step (`WHISPER_SEND_WAV`); model gguf cached in `data/models/parakeet/`.
>
> **Embedding upgrade done after all (commits b24b6e2, 1648b23):** speaker-ID moved to wespeaker ResNet221-LM (ONNX) with a fresh `speakers_resnet221` collection; 31 speakers re-enrolled from reference clips via `scripts/enroll_reference_speakers.py`. Identification now builds one embedding per speaker from ≤60s of their longest segments (waveform passed through from diarization — no re-decode, ~2s overhead). Threshold recalibrated to 0.65 from measured scores (worst impostor 0.583, genuine 0.72–0.93). Meetings now come back with real names: Andy/Davide/Jason/Ken/Mat at 0.72–0.93 confidence. Enrollment depth is thin (1 clip for most speakers; Ken's two clips only score 0.43 against each other) — adding more samples per speaker via the UI is the cheapest future accuracy win.
>
> Also fixed en route: parakeet.cpp hard-crashes on long inputs (attention memory ~quadratic; a 10-min chunk wants ~14 GB) — the api now chunks WAV uploads at 180s and re-offsets timestamps on merge.

From the web-verified research (all claims checked against PyPI/GitHub/HF primary sources):

| Component | Current | Verdict |
|---|---|---|
| Diarization | pyannote `community-1` | **Keep** — current best open model; no community-2 exists. |
| ASR model | `faster-distil-whisper-large-v3` | **Replace.** Distillation degrades the cross-attention that word timestamps come from, it's English-only, and Whisper-family models score 12–14% WER on meeting audio (AMI) vs 7–9% for newer models. |
| ASR runtime | faster-whisper via Speaches | **Pin now** (`faster-whisper==1.2.1`, `ctranslate2==4.8.1`) — the wrapper is orphaned (SYSTRAN team left); CTranslate2 itself was rescued and is healthy. Plan migration. |
| Speaker embedding | `wespeaker-voxceleb-resnet34-LM` | **Keep by default.** Upgrade candidate: `Wespeaker/wespeaker-voxceleb-resnet221-LM` — same 256 dims, same fbank frontend, ONNX drop-in, 1.43× lower EER, Apache-2.0. Requires re-enrolling all Qdrant vectors from stored samples. Only do this if speaker-ID confusion is an observed problem. Do **not** swap the embedding inside the diarization pipeline (PLDA is trained for ResNet34's space). |
| Qdrant | older image | Bump to `v1.18.3`. No other changes needed at this scale. |

**ASR replacement path (ordered):**
1. *Cheap, same-runtime:* switch Speaches to full `Systran/faster-whisper-large-v3` — better alignment and accuracy than distil, zero integration work.
2. *Recommended target:* **`nvidia/parakeet-tdt-0.6b-v3`** (CC-BY-4.0, ~2 GB VRAM, RTFx ~6000, native word timestamps, 25 languages) served via **`parakeet.cpp`** (`ghcr.io/mudler/parakeet.cpp-server`, MIT, active, OpenAI-compatible, sm_75+). Dodges both the faster-whisper and NeMo-packaging maintenance messes. Caveats: single-request-at-a-time (front with the Phase 2 queue), WAV only, perf claims not yet independently verified on Ampere.
3. *Watch list:* `ibm-granite/granite-speech-4.1-2b-plus` (best-in-class 38.8 ms word timing + built-in speaker attribution, Apache-2.0, but 3.5-min chunking for timestamp mode); MOSS-Transcribe-Diarize (joint ASR+diarization, but turn-level timestamps only). Avoid: Sortformer (4-speaker ceiling), CrisperWhisper 2.0 (non-commercial weights), ReDimNet2 (no compatible ONNX yet).

**On Speaches:** it does have pyannote-4.x diarization + known-speaker matching built in (undocumented in its README), which duplicates our Qdrant speaker-ID layer — but it does *not* fuse ASR+diarization, doesn't use `exclusive_speaker_diarization`, and hasn't cut a stable release in ~8 months. Keep it as an ASR endpoint for now; keep speaker identity ownership in `api/`.

## CUDA 13 migration — attempted 2026-08-04, reverted

> Host-side there is no blocker: driver 580.x already advertises CUDA 13.0, and the stability fix is a 300W power cap (`nvidia-power-limit.service`) independent of container CUDA versions. The container migration (base `13.0.1-cudnn-runtime`, torch 2.11.0+cu130, torchcodec 0.15, onnxruntime-gpu 1.28) was functionally perfect — bit-identical diarization/identification output — but **~6× slower at diarization** (alternating A/B on the same file: CUDA 12 image 14–20s vs CUDA 13 image 105–120s). Ruled out: missing sm_86 kernels (present), cuDNN (micro-benchmarks identical), audio decode (equal). Root cause somewhere in pyannote's pipeline execution on torch 2.11+cu130; not worth chasing now. Reverted to torch 2.8.0+cu128 / onnxruntime-gpu 1.22. Revisit when pyannote or torch move on, and re-benchmark first.
>
> **Open finding from the A/B:** the same CUDA 12 image diarizes the 16-min file in **14–20s** in a minimal `docker run` container but **85–105s** as the compose `api` service — same code, same model, same GPU. Something in the compose deployment costs ~5×. Worth investigating; a large diarization speedup may be available for free.

## Phase 4 — Rationalize `pipeline/` (after Phase 2)

`pipeline_speaches.py` is the active batch pipeline; `meeting_server.py` is the daily-use product. Superseded once-only tools (`process_meetings.py`, `cluster_speakers.py`, `server.py`, `migrate_speakers.py`) stay in-tree but frozen.

1. **Fix the large-file regression:** `pipeline_speaches.py` silently *skips* files >15 MB (`MAX_FILE_SIZE`, lines 36, 493-497); the old pipeline chunked them into 15-min pieces and stitched timestamps. Port that chunking over.
2. **Restore a speaker-roster growth path.** The 30-name reference-clip roster is a frozen one-time export; new speakers come back as `SPEAKER_NN` forever. Once `api/` is hardened, enroll new speakers through it (Qdrant) rather than per-request base64 reference clips.
3. **Extract config:** hardcoded `192.168.8.147`, Shadow audio path, and relative-cwd `Path("output")` footguns → a small shared config module / `.env`.
4. Resolve the unexplained LLM downgrade (`gpt-oss-120b` → `gpt-oss-20b` in the newest script) — if it was VRAM pressure on the Speaches box, document it; local alternatives on the 3090: `Qwen3-30B-A3B-Instruct` (MoE, fast) or `gpt-oss-20b` via Ollama/llama.cpp.
5. Prune the 18 manifest backups; add rotation (keep last N).

## Phase 5 — Audio trimmer & UI polish (parallel, low risk)

Before committing `audio-trimmer.tsx`:
1. Use wavesurfer regions' native `maxLength: 30` instead of the manual clamp — fixes the right-edge-jumps-back bug when dragging the left handle, and enforces during drag rather than on mouse-up.
2. `audioContext.close()` in a `finally` (currently leaks on decode failure; Chrome caps 6 contexts/page).
3. Add a loading state to "Use Selection" (async full-file decode; double-clicks currently race).
4. Known limitation to fix or accept: the whole source file is decoded to raw PCM in browser memory to cut a ≤30 s clip — a 2-hour recording can crash the tab. Acceptable if inputs are short samples; document or guard with a file-size warning.
5. Later: expose `min_speakers`/`max_speakers`, `language`, `similarity_threshold` in the UI; add a diarize-only quick check; fix the 422-error `[object Object]` rendering; keyboard accessibility for file upload/dialogs/tabs.

---

## Suggested order of attack

1. Phase 0 entirely (unblocks clean commits).
2. Phase 1 items 1–3 (delete-code-fix-bugs) + 6 (bumps), then run a long-file soak test with `nvidia-smi` logging to confirm stability with the undervolt + #1963 mitigations.
3. Phase 2 items 1, 3, 4 (to_thread + upload limit + auth) — at this point the service is safely usable from other LAN machines.
4. Phase 3 ASR step 1 (large-v3 swap) immediately; Parakeet evaluation as a side experiment.
5. Phase 2 item 2 (job queue), then Phase 4 pipeline work on top of it.
6. Phase 5 trimmer fixes whenever convenient; commit it after items 1–3.
