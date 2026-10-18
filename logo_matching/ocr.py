"""PaddleOCR wrapper for logo images.

Chosen over Tesseract per task requirement: these are real-world,
low-quality/compressed images scraped from LinkedIn (not clean scanned
documents), and PaddleOCR's detection+recognition pipeline is built for
noisy real-world text in that regime.

enable_mkldnn=False is required, not optional, on this machine's CPU/paddle
build - with mkldnn on, text detection raises
"NotImplementedError: ... ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]" on every image (confirmed via
direct testing against the sample logos in `data/logo_samples/`), a known
oneDNN/PIR-executor incompatibility in this paddlepaddle build, not an
image-specific failure.

cpu_threads=1 is likewise required, not optional, on this machine: with the
default multi-threaded matmul, every predict() call segfaults inside
OpenBLAS's sgemm_itcopy (confirmed via gdb backtrace into libphi_core.so) -
a known OpenBLAS runtime-CPU-dispatch bug on AWS Graviton2/Neoverse-N1.
Standard OPENBLAS_NUM_THREADS/OPENBLAS_CORETYPE env vars don't reach this
build (OpenBLAS is statically linked into libphi_core without honoring
them); only PaddleOCR's own cpu_threads config plumbs through and avoids
the crash. This does mean OCR runs single-threaded per instance - fine here
since each worker thread already gets its own PaddleOCR instance (see
below), so parallelism comes from the thread pool, not from within a
single predict() call.

use_doc_orientation_classify/use_doc_unwarping are disabled: those stages
exist for scanned-document photos (skewed page detection, dewarping) and
are irrelevant to small flat logo graphics - leaving them on only adds
startup cost (extra model downloads) for no benefit here.

predict() accepts a decoded numpy array (BGR, same as cv2.imread/imdecode
would produce) exactly like it accepts a file path - confirmed by direct
testing - so a downloaded logo never needs to touch disk as a temp file;
fetch_logo.py decodes straight from the HTTP response bytes and hands the
array here directly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR

# Below this per-line OCR confidence, text is kept in the printed output
# (for manual accuracy spot-checking) but excluded from what the keyword
# patterns scan - garbled low-quality reads otherwise become their own
# false-positive source on top of the patterns' own ambiguity. 0.7 chosen
# per task guidance; real sample scores cluster either near 1.0 (clean
# wordmark text) or well below 0.7 (single stray letterform / decorative
# glyph misread as text), so this threshold cuts cleanly between them
# rather than splitting a dense middle.
DEFAULT_MIN_CONFIDENCE = 0.7

# One PaddleOCR instance PER THREAD (threading.local), not a single shared
# module-level singleton - the earlier version used one shared instance,
# which was never verified safe for concurrent calls from multiple threads
# (PaddleOCR/paddlex's internal predictor state is not documented as
# thread-safe). Rather than gamble on that when this module is about to be
# called from a real worker pool (see combined_matching/logo_detection.py),
# each thread gets its own fully isolated instance - costs ~414MB RSS per
# thread (measured directly against this project's models), which is what
# actually caps how many OCR worker threads a run can afford, but it
# guarantees no cross-thread interference in OCR output regardless of
# paddle's internal thread-safety.
_thread_local = threading.local()


def _get_ocr() -> PaddleOCR:
    if not hasattr(_thread_local, "ocr"):
        _thread_local.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            # Disabled after a direct A/B benchmark against 99 real fetched
            # logos (incl. the 7 with a known ground-truth acquirer): 19.1%
            # faster (0.561s -> 0.454s/image) with ZERO detection-outcome
            # changes across all 99 images and identical corroborated/
            # conflict/no_signal counts on the known-acquirer set. Text
            # differed on 5/99 images, but only in discarded low-confidence
            # fragments (a stray character or two) that never fed into
            # extract_acquirer_from_logo_text either way - profiling (py-spy,
            # live production process) showed this stage running a full
            # per-detected-text-region classification pass, on top of
            # detection+recognition, for no measurable benefit on this
            # project's logos.
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
            cpu_threads=1,
            # Swapped from the default PP-OCRv6_medium det/rec models for the
            # ~800K-image batch run - PP-OCRv6_tiny was tried first (~7.2x
            # faster, ~0.12s/image) but real-world accuracy on this project's
            # logos wasn't acceptable, so this uses PP-OCRv5_mobile instead:
            # benchmarked at ~0.27s/image on this machine vs ~0.89s/image for
            # v6_medium (~3.3x faster - roughly a ~9-day to ~2.6-day single-
            # process run), with confidence much closer to the v6_medium
            # baseline (0.965 vs 0.970 on the one sample benchmarked, vs
            # v6_tiny's 0.941). Detection logic itself (detection.py) is
            # unchanged - this only changes which OCR weights produce the
            # text detection.py scans.
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        )
    return _thread_local.ocr


@dataclass
class OcrResult:
    lines: list[tuple[str, float]]  # (text, confidence) per detected text line, ALL of them

    @property
    def text(self) -> str:
        """Every detected line joined - for printing/spot-checking OCR
        accuracy, not for feeding into keyword detection (see
        high_confidence_text)."""
        return " ".join(t for t, _ in self.lines)

    def high_confidence_text(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> str:
        """Only lines at/above `min_confidence`, joined - this is what
        detection.extract_acquirer_from_logo_text should actually scan."""
        return " ".join(t for t, score in self.lines if score >= min_confidence)


def _run_ocr(source: str | np.ndarray) -> OcrResult:
    ocr = _get_ocr()
    results = ocr.predict(source)

    lines: list[tuple[str, float]] = []
    for res in results:
        data = res.json["res"]
        for text, score in zip(data["rec_texts"], data["rec_scores"]):
            lines.append((text, score))

    return OcrResult(lines=lines)


def run_ocr(image_path: str | Path) -> OcrResult:
    """Run OCR on a local image file (used by the earlier local-sample
    testing path; kept for standalone debugging)."""
    return _run_ocr(str(image_path))


def run_ocr_array(image: np.ndarray) -> OcrResult:
    """Run OCR on an already-decoded in-memory image (BGR numpy array) -
    the path used for logos fetched over HTTP, so the bytes never touch
    disk."""
    return _run_ocr(image)