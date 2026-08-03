"""CUDA error recovery utilities."""

import logging
import torch

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def recover_cuda():
    """Attempt to recover from a CUDA error by clearing cache and resetting state."""
    logger.warning("Attempting CUDA recovery...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("CUDA recovery: cache cleared and synchronized")
    except Exception as e:
        logger.error(f"CUDA recovery failed: {e}")


def is_cuda_error(exc: Exception) -> bool:
    """Check if an exception is a CUDA-related error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "cuda error",
        "cuda runtime error",
        "cublas",
        "cudnn",
        "out of memory",
        "launch failure",
        "device-side assert",
    ])
