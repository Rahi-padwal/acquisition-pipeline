"""Run OCR on ONE local image file - no ES, no S3, no network at all.

Purpose: isolate the OCR engine itself from everything else in the
pipeline, for debugging (e.g. confirming whether a crash is in PaddleOCR's
model execution specifically, independent of the fetch/lookup steps).

Usage:
    python test_single_image.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ocr import run_ocr

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_single_image.py <image_path>")
        sys.exit(1)

    result = run_ocr(sys.argv[1])
    print(f"OCR text: {result.text!r}")
    for text, score in result.lines:
        print(f"  line={text!r} score={score:.2f}")
