#!/usr/bin/env python
"""
A command-line tool to probe OCR performance on a PDF file.

This script runs OCR on a given PDF and prints a page-by-page analysis
of the results, including character count, ink ratio, and other metadata.

Usage:
  python bin/ocr_probe.py path/to/your.pdf [OPTIONS]

Example:
  python bin/ocr_probe.py docs/sample.pdf --uniform --psm 6
"""
import argparse
import sys
from pathlib import Path

# Ensure the project root is on the Python path to allow importing 'utils'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from utils.pdf_utils import extract_text_from_pdf
except ImportError as e:
    print(f"Error: Could not import required modules. Make sure you are running from the project root.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)


def main():
    """Main function to parse arguments and run the OCR probe."""
    parser = argparse.ArgumentParser(description="Profile OCR performance on a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to process.")
    parser.add_argument("--uniform", action="store_true", help="Assume a single uniform block of text (sets PSM to 6 unless overridden).")
    parser.add_argument("--psm", type=int, choices=[3, 4, 6], default=None, help="Override Tesseract's Page Segmentation Mode.")
    parser.add_argument("--oem", type=int, default=1, help="Specify Tesseract's OCR Engine Mode.")
    parser.add_argument("--extra", type=str, default=None, help="Pass extra configuration options to Tesseract.")
    parser.add_argument("--no-preproc", action="store_true", help="Disable OCR preprocessing (deskew, binarization).")

    args = parser.parse_args()

    if not args.pdf_path.is_file():
        print(f"Error: File not found at '{args.pdf_path}'", file=sys.stderr)
        sys.exit(0) # Exit 0 as it's a probe tool

    try:
        print(f"Probing '{args.pdf_path.name}' with OCR...")
        page_texts, metas = extract_text_from_pdf(
            str(args.pdf_path),
            use_ocr=True,
            return_meta=True,
            ocr_assume_uniform_block=args.uniform,
            ocr_psm=args.psm,
            ocr_oem=args.oem,
            ocr_extra_config=args.extra,
            enable_preproc=not args.no_preproc,
        )

        # Print header
        print("-" * 60)
        print(f"{'Page':<6} | {'Chars':<8} | {'Ink Ratio':<12} | {'Noisy?':<8} | {'PSM Used':<10}")
        print("-" * 60)

        # Print data rows
        for i, meta in enumerate(metas):
            chars = len(page_texts[i])
            print(
                f"{meta.page_index:<6} | {chars:<8} | {meta.ink_ratio:<12.3f} | "
                f"{str(meta.noisy):<8} | {meta.psm_used:<10}"
            )

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}", file=sys.stderr)
        print("This might be due to missing dependencies like 'tesseract' or 'poppler'.", file=sys.stderr)
    
    sys.exit(0)


if __name__ == "__main__":
    main()