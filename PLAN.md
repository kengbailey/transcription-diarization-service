# Project Plan — Transcription & Diarization Service

*Refocused 2026-08-04: this repo is the **transcription + diarization service only** — the `api/` backend and the `ui/` frontend (Transcription, Speakers, Settings tabs). The batch meeting pipeline, meeting-review UI, and summarization tooling were removed from repo and disk; they live in git history before the `refocus-cleanup` branch if ever needed. Anything meeting-domain should be built *outside* this repo as a client of the API. All compute runs on this host's RTX 3090 (300W power cap via `nvidia-power-limit.service`); the former remote-GPU boxes 192.168.8.116/.147 are gone — never point config at them.*

## Current state (all verified live)

The hardening phases from the 2026-08-03 audit are complete:

- **Correctness:** pyannote 4.0.7 with `exclusive_speaker_diarization` (overlap bug gone) and pipeline-provided speaker centroids; per-segment full-file re-decode eliminated; CUDA-recovery consolidated into a `with_cuda_retry` decorator that fails fast on context-corrupting errors. Issue #1963 VRAM-spike mitigations in place (`EMBEDDING_BATCH_SIZE=16`, `expandable_segments`). Soak-tested on a 2.55-hour meeting: zero CUDA errors.
- **Multi-client LAN service:** GPU work off the event loop behind a `Semaphore(1)`; job queue (`POST /jobs/{kind}` → 202 + poll/cancel) for timeout-immune long audio; optional `API_KEY` auth; streaming upload cap (413); orphan-upload sweeper; request-ID logging; `/health` reports GPU memory + queue depth in 4–8 ms even mid-job.
- **Models:** ASR is `parakeet-tdt-0.6b-v3` via parakeet.cpp (~360× real-time, native word timestamps), launched **on demand** by the host's llama-swap (port 9292, 300s TTL — entry in `~/sandbox/llama-swap/config.yaml`; its health endpoint is `/health`, not `/v1/models`). parakeet.cpp is WAV-only and its attention memory is ~quadratic, so the api transcodes and chunks at 180s (`WHISPER_SEND_WAV`/`WHISPER_CHUNK_SECONDS`). Fallback ASR: speaches with `faster-whisper-large-v3`. Speaker-ID embeddings are wespeaker ResNet221-LM (ONNX, `onnxruntime-gpu==1.22` — last CUDA 12.x build), collection `speakers_resnet221`, threshold 0.65 calibrated from measured scores (worst impostor 0.583, genuine 0.72–0.93). 31 speakers enrolled.
- **VRAM choreography:** `MODEL_IDLE_TIMEOUT` (300s in compose) unloads diarization/embedding models after idle — idle GPU footprint of the whole service is ~0.6 GB, with no measured reload penalty.
- **UI:** comprehensive review done — nginx re-resolves the api container's IP (no more stale-DNS 502s), dialogs reset state on every close path, API-key storage, cancel/abort for in-flight jobs, accessibility (dialogs/tabs/dropzone), audio-trimmer hardening.

## Operational findings worth remembering

- **GPU thermal throttling (unresolved, hardware-side):** the card diarizes a 16-min file in ~15s when thermally healthy but ~90–120s under sustained load. Telemetry shows `SW Thermal Slowdown: Active` with core at only 75–79°C, fan 100%, power below the cap — the classic 3090 GDDR6X memory-junction signature (sensor not exposed on GeForce). The fix is physical (repad/repaste/airflow). A 25-min deep-cooldown A/B to confirm memory-heat hysteresis was designed but not run — do it before any hardware work. Until fixed, treat all sustained-load timings as ~5× pessimistic.
- **CUDA 13 migration: attempted and reverted (2026-08-04).** torch 2.11.0+cu130 produced bit-identical output but ~6× slower diarization (A/B on the same file; kernels, cuDNN, and decode all ruled out). Staying on torch 2.8.0+cu128. Re-benchmark before retrying when pyannote or torch move on.
- **Embedding spaces don't mix:** the diarization pipeline internally uses ResNet34; speaker-ID uses ResNet221. Same 256 dims, different spaces — never put both in one Qdrant collection, and never swap the embedding inside the pyannote pipeline (PLDA is trained for ResNet34's space). Old collections `speaker_embeddings`/`work_speaker_embeddings` (ResNet34-era) still exist on disk, unused.

## Roadmap

UI (deferred from the review, roughly by value):

1. **Job-queue support in the UI** — the biggest gap: long transcriptions should go through `POST /jobs/*` with progress polling instead of one long-lived request.
2. Transcribe-only and diarize-only modes (the API already has the endpoints).
3. Expose `min_speakers`/`max_speakers`, `language`, `similarity_threshold` as form controls.
4. Speaker list search/pagination (31+ speakers and growing).
5. Expose the `extract_segments` toggle.

Accuracy:

- **Enrollment depth is thin** — most speakers have a single reference clip (Ken's two clips only score 0.43 against each other). Adding 2–3 samples per speaker via the UI is the cheapest identification-accuracy win.

Housekeeping:

- Drop the unused ResNet34-era Qdrant collections once confident nothing references them.
- GPU thermal fix (hardware) + the deep-cooldown A/B above.
