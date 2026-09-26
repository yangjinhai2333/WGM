"""Runtime-only cancellation evidence; never participates in search identity/RNG."""
import re
import threading
import time

_lock = threading.RLock()


def secondary(error):
    return isinstance(error, KeyboardInterrupt) or getattr(error, 'code', None) in (
        'NETWORK_READINESS_CANCELLED', 'DGM_CANCELLED')


def prefer(first, later):
    return later if first is None or secondary(first) and not secondary(later) else first


def safe_error(error):
    code = getattr(error, 'code', None)
    return {'type': type(error).__name__, 'code': code if isinstance(code, str)
            and re.fullmatch(r'[A-Z][A-Z0-9_]{0,100}', code) else None}


def request(cancel, error=None, *, signum=None):
    with _lock:
        if not hasattr(cancel, '_wgm_cancel_origin'):
            cancel._wgm_cancel_origin = {'time_ns': time.time_ns(), 'signal': signum,
                'error': safe_error(error) if error is not None else None}
        if error is not None:
            cancel._wgm_primary_error = prefer(getattr(cancel, '_wgm_primary_error', None), error)
        cancel.set()


def primary(cancel, error=None):
    saved = getattr(cancel, '_wgm_primary_error', None)
    return prefer(saved, error) if error is not None else saved
