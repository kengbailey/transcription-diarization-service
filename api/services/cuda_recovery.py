"""CUDA error recovery utilities."""

import functools
import logging
import time

import torch

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0

# Errors that corrupt the CUDA context. No in-process recovery is possible and
# a "successful" retry can return silently wrong results — fail fast and rely
# on the container restart policy.
_FATAL_PATTERNS = [
    "device-side assert",
    "illegal memory access",
    "unspecified launch failure",
]

# Errors where clearing the cache and reloading the model may genuinely help
# (fragmentation OOM, transient cuBLAS/cuDNN failures).
_TRANSIENT_PATTERNS = [
    "cuda error",
    "cuda runtime error",
    "cublas",
    "cudnn",
    "out of memory",
]


def classify_cuda_error(exc: Exception) -> str:
    """Classify an exception as 'fatal', 'transient', or 'other' (non-CUDA)."""
    msg = str(exc).lower()
    if any(p in msg for p in _FATAL_PATTERNS):
        return "fatal"
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "transient"
    if any(p in msg for p in _TRANSIENT_PATTERNS):
        return "transient"
    return "other"


def recover_cuda() -> None:
    """Best-effort CUDA cleanup between retries."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("CUDA recovery: cache cleared and synchronized")
    except Exception as e:
        logger.error(f"CUDA recovery failed: {e}")


def with_cuda_retry(reinit_method: str):
    """Decorate an instance method to retry on transient CUDA errors.

    On a transient error: clear the CUDA cache, back off, call
    ``self.<reinit_method>()`` to reload the model, and retry (up to
    MAX_RETRIES). Context-corrupting errors and non-CUDA errors are re-raised
    immediately.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            for attempt in range(MAX_RETRIES + 1):
                try:
                    return fn(self, *args, **kwargs)
                except Exception as e:
                    kind = classify_cuda_error(e)
                    if kind == "fatal":
                        logger.critical(
                            f"Unrecoverable CUDA error in {fn.__name__}: {e}. "
                            "The CUDA context is likely corrupted; the container "
                            "must be restarted."
                        )
                        raise
                    if kind != "transient" or attempt >= MAX_RETRIES:
                        raise
                    logger.warning(
                        f"Transient CUDA error in {fn.__name__} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
                    )
                    recover_cuda()
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                    getattr(self, reinit_method)()

        return wrapper

    return decorator
