"""One process-wide GPU execution slot, shared by batch and stream workers."""
from functools import wraps
from threading import RLock

GPU_EXECUTION_LOCK = RLock()

def serialized_inference(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with GPU_EXECUTION_LOCK:
            return function(*args, **kwargs)
    return guarded
