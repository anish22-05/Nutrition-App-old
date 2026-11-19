"""Background upload jobs: a big PDF takes a minute to embed, far too long to hold a request open.

POST /upload/async returns a job id at once; the work runs on a small thread pool and
GET /jobs/{id} reports {status, stage, progress, result | error}. Jobs live in memory (per
process) and are forgotten an hour after they finish. Like the rate limiter, a multi-instance
deploy would move this to a real queue (Redis/RQ, Celery, a managed queue).
"""
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from app import usage

KEEP_FINISHED_S = 3600


class JobStore:
    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, owner_id: str | None, fn: Callable[[Callable[[str, float], None]], dict],
               to_error: Callable[[Exception], str]) -> str:
        """Run fn(progress) in the background. to_error turns an exception into a user message."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._gc()
            self._jobs[job_id] = {"job_id": job_id, "owner_id": owner_id, "status": "queued", "stage": "queued",
                                  "progress": 0.0, "result": None, "error": None, "created": time.time(), "finished": None}

        def progress(stage: str, frac: float) -> None:
            self._update(job_id, stage=stage, progress=round(min(max(frac, 0.0), 1.0), 3))

        def work() -> None:
            usage.start()
            self._update(job_id, status="running")
            try:
                result = fn(progress)
                self._update(job_id, status="done", stage="done", progress=1.0, result=result, finished=time.time())
            except Exception as e:  # noqa: BLE001 - becomes the job's error message
                self._update(job_id, status="failed", error=to_error(e), finished=time.time())

        self._pool.submit(work)
        return job_id

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get(self, job_id: str, owner_id: str | None) -> dict | None:
        """Only the owner who started a job can see it (same 404 otherwise)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["owner_id"] != owner_id:
                return None
            return {k: v for k, v in job.items() if k != "owner_id"}

    def _gc(self) -> None:
        now = time.time()
        for jid in [j for j, v in self._jobs.items() if v["finished"] and now - v["finished"] > KEEP_FINISHED_S]:
            del self._jobs[jid]
