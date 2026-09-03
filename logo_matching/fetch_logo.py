"""Fetch a single logo image from its crustdata S3 URL, entirely in memory.

No temp file is used - confirmed by direct testing that PaddleOCR's
predict() accepts a decoded numpy array (via cv2.imdecode) exactly like it
accepts a file path, so the downloaded bytes never touch disk at all
(stronger than the task's "short-lived temp file is fine" allowance, not
just meeting it).

Four distinct outcomes, matched to the three real-world access cases this
was built for plus one added during testing:
  - "ok"            - HTTP 200, bytes decoded successfully by OpenCV
  - "no_logo"        - logoS3Path missing/null/empty in ES (or no ES doc at
                        all) - no HTTP request is made at all in this case
  - "access_denied"  - HTTP 403
  - "fetch_failed"    - any other failure: non-200/403 status, timeout,
                        connection error, or (added after live testing)
                        a 200 response whose body OpenCV can't decode as an
                        image at all

IMPORTANT finding from live testing against the real bucket: requesting a
KNOWN-NONEXISTENT key on crustdata-media.s3.us-east-2.amazonaws.com also
returns 403 (AccessDenied), not 404 - this bucket doesn't distinguish
"you can't read this" from "this key doesn't exist" for an unauthenticated
GET. So "access_denied" in the case-C sense may just as often mean a
stale/wrong logoS3Path value as a real permissions problem - exactly the
ambiguity the task called out as "worth investigating later". Logged as
its own category regardless, per the task's requirement to keep it
separate from "fetch_failed"; the summary counts alone won't tell you
which sub-case you're looking at, only that it's worth sampling.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import requests

_TIMEOUT_SECONDS = 20

FetchOutcome = Literal["ok", "no_logo", "access_denied", "fetch_failed"]

# One requests.Session PER FETCH-WORKER THREAD (threading.local), not a
# bare requests.get() call - every logo lives behind the same S3 host
# (crustdata-media.s3.us-east-2.amazonaws.com), so a plain requests.get()
# was paying a fresh TCP+TLS handshake on every single one of 800K
# fetches instead of reusing a keep-alive connection. That handshake is
# real CPU work (key exchange, cipher setup), and with 16 fetch threads
# doing it concurrently on this box's 2 vCPUs, it was contending for CPU
# with the single OCR thread that needs a full core to hit its
# benchmarked 0.27s/image (confirmed against run_logo_ocr_batch.log:
# OCR-inclusive throughput was ~0.52s/image, ~2x the pure predict()
# benchmark). Thread-local rather than a shared module-level Session
# since requests.Session isn't documented thread-safe for concurrent use
# from multiple threads - same rationale as ocr.py's per-thread
# PaddleOCR instance.
_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


@dataclass
class FetchResult:
    outcome: FetchOutcome
    image: np.ndarray | None = None  # BGR image array, only set when outcome == "ok"
    detail: str | None = None  # short human-readable reason, mainly for fetch_failed/access_denied


def fetch_logo_image(logo_url: str | None) -> FetchResult:
    if not logo_url:
        return FetchResult(outcome="no_logo", detail="logoS3Path missing/null/empty in ES")

    try:
        resp = _get_session().get(logo_url, timeout=_TIMEOUT_SECONDS)
    except requests.Timeout:
        return FetchResult(outcome="fetch_failed", detail="timeout")
    except requests.RequestException as exc:
        return FetchResult(outcome="fetch_failed", detail=f"connection error: {exc}")

    if resp.status_code == 403:
        return FetchResult(outcome="access_denied", detail="HTTP 403")
    if resp.status_code != 200:
        return FetchResult(outcome="fetch_failed", detail=f"HTTP {resp.status_code}")

    array = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return FetchResult(outcome="fetch_failed", detail="HTTP 200 but body was not a decodable image")

    return FetchResult(outcome="ok", image=image)
