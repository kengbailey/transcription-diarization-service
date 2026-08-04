"""In-memory async job queue with a single background worker.

Jobs are processed one at a time (the executor acquires the GPU semaphore),
so long audio never blocks the event loop and LAN clients get immediate
job-id responses instead of multi-minute HTTP waits. Job records live in
memory only — a restart clears history.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional


logger = logging.getLogger(__name__)

# Completed/failed job records retained before the oldest are pruned
JOB_RETENTION = 100


@dataclass
class Job:
    job_id: str
    kind: str
    filename: str
    params: dict
    filepath: str  # internal server path, not serialized
    status: str = "queued"  # queued | running | completed | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def serialize(self, queue_position: Optional[int] = None, include_result: bool = True) -> dict:
        data = {
            "job_id": self.job_id,
            "kind": self.kind,
            "filename": self.filename,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
        if queue_position is not None:
            data["queue_position"] = queue_position
        if include_result:
            data["result"] = self.result
        return data


class JobQueue:
    """Single-worker FIFO queue for GPU jobs."""

    def __init__(
        self,
        executor: Callable[[Job], Awaitable[dict]],
        cleanup: Callable[[str], None],
    ):
        """
        Args:
            executor: async callable that runs the job and returns its result
                dict (expected to hold the GPU semaphore internally)
            cleanup: called with the job's filepath once the job leaves the
                queue (finished, failed, or cancelled)
        """
        self._executor = executor
        self._cleanup = cleanup
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("Job queue worker started")

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def submit(self, kind: str, filepath: str, filename: str, params: dict) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            kind=kind,
            filename=filename,
            params=params,
            filepath=filepath,
        )
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job)
        logger.info(f"Job {job.job_id} ({kind}) queued: {filename}")
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def queue_position(self, job: Job) -> Optional[int]:
        """1-based position among queued jobs; None if not queued."""
        if job.status != "queued":
            return None
        queued = [j for j in self._jobs.values() if j.status == "queued"]
        queued.sort(key=lambda j: j.created_at)
        try:
            return queued.index(job) + 1
        except ValueError:
            return None

    def counts(self) -> dict:
        queued = sum(1 for j in self._jobs.values() if j.status == "queued")
        running = sum(1 for j in self._jobs.values() if j.status == "running")
        return {"queued": queued, "running": running}

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Running/finished jobs cannot be cancelled."""
        job = self._jobs.get(job_id)
        if job is None or job.status != "queued":
            return False
        job.status = "cancelled"
        job.finished_at = time.time()
        return True

    def remove(self, job_id: str) -> bool:
        """Remove a finished job record."""
        job = self._jobs.get(job_id)
        if job is None or job.status in ("queued", "running"):
            return False
        del self._jobs[job_id]
        return True

    def active_filepaths(self) -> set[str]:
        """Upload paths still owned by queued/running jobs (for the sweeper)."""
        return {
            j.filepath
            for j in self._jobs.values()
            if j.status in ("queued", "running")
        }

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()

            if job.status == "cancelled":
                self._cleanup(job.filepath)
                continue

            job.status = "running"
            job.started_at = time.time()
            logger.info(f"Job {job.job_id} ({job.kind}) started")

            try:
                job.result = await self._executor(job)
                job.status = "completed"
                logger.info(
                    f"Job {job.job_id} completed in "
                    f"{time.time() - job.started_at:.1f}s"
                )
            except asyncio.CancelledError:
                job.status = "failed"
                job.error = "Server shutdown"
                job.finished_at = time.time()
                self._cleanup(job.filepath)
                raise
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                logger.exception(f"Job {job.job_id} failed")
            finally:
                if job.finished_at is None:
                    job.finished_at = time.time()
                self._cleanup(job.filepath)
                self._prune()

    def _prune(self) -> None:
        finished = [
            j for j in self._jobs.values()
            if j.status in ("completed", "failed", "cancelled")
        ]
        if len(finished) <= JOB_RETENTION:
            return
        finished.sort(key=lambda j: j.finished_at or j.created_at)
        for job in finished[: len(finished) - JOB_RETENTION]:
            del self._jobs[job.job_id]
