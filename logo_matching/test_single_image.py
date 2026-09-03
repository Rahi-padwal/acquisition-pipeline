"""Run OCR on ONE local image file - no ES, no S3, no network at all.

Purpose: isolate the OCR engine itself from everything else in the
pipeline, for debugging (e.g. confirming whether a crash is in PaddleOCR's
model execution specifically, independent of the fetch/lookup steps).

Usage:
    python test_single_image.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from detection import extract_acquirer_from_logo_text
from ocr import run_ocr

if __name__ == "__main__":
    img= "/home/ubuntu/acquired_company/logo_matching/sample.jpg"
    company_name = sys.argv[1] if len(sys.argv) > 1 else None

    result = run_ocr(img)
    print(f"OCR text: {result.text!r}")
    for text, score in result.lines:
        print(f"  line={text!r} score={score:.2f}")

    detection = extract_acquirer_from_logo_text(result.high_confidence_text(), company_name)
    print(f"is_acquired: {detection is not None}")
    if detection is not None:
        print(f"  acquirerCompany={detection['acquirerCompany']!r}")
        print(f"  matchedPattern={detection['matchedPattern']!r}")
        print(f"  confidenceScore={detection['confidenceScore']}")
