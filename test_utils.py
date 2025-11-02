import os
from utils.pdf_utils import extract_text_from_pdf

# --- Define Test File Paths ---
# (Assumes this script is in the root and files are also in the root)
digital_pdf = "digital_test.pdf"
scanned_pdf = "scanned_test.pdf"
empty_pdf = "empty_test.pdf"

print("Starting PDF utility tests...\n")

# --- Test 1: Digital PDF (Should use PyMuPDF) ---
print("--- TEST 1: Digital PDF (Fast Extraction) ---")
try:
    texts = extract_text_from_pdf(digital_pdf)
    print(f"Result: {texts}")
    if "digital" in texts[0]:
        print("STATUS: ✅ PASSED\n")
    else:
        print("STATUS: ⚠️  FAILED (Text not found)\n")
except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")


# --- Test 2: Scanned PDF (Forcing OCR) ---
print("--- TEST 2: Scanned PDF (Forcing OCR) ---")
try:
    texts = extract_text_from_pdf(scanned_pdf, use_ocr=True)
    print(f"Result: {texts}")
    if "scanned" in texts[0].lower():
        print("STATUS: ✅ PASSED\n")
    else:
        print("STATUS: ⚠️  FAILED (OCR text not found)\n")
except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")


# --- Test 3: Scanned PDF (Testing Auto-Fallback) ---
print("--- TEST 3: Scanned PDF (Auto-Fallback) ---")
try:
    # We call it WITHOUT use_ocr=True to test if it
    # correctly finds no text and falls back to OCR.
    texts = extract_text_from_pdf(scanned_pdf)
    print(f"Result: {texts}")
    if "scanned" in texts[0].lower():
        print("STATUS: ✅ PASSED (Auto-fallback worked)\n")
    else:
        print("STATUS: ⚠️  FAILED (Auto-fallback failed)\n")
except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")


# --- Test 4: Empty Page (Edge Case) ---
print("--- TEST 4: Empty Page (Edge Case) ---")
try:
    texts = extract_text_from_pdf(empty_pdf)
    print(f"Result: {texts}")
    # An empty page should return a list with one empty string
    if texts == ['']:
        print("STATUS: ✅ PASSED\n")
    else:
        print(f"STATUS: ⚠️  FAILED (Expected [''] but got {texts})\n")
except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")

### print("...Tests finished.")

# (Add this to the end of test_utils.py)
from utils.pdf_utils import split_text

print("\n--- TEST 5: Text Splitting (Overlap) ---")
try:
    long_text = "This is the first sentence. " * 200  # A long string
    chunks = split_text(long_text, max_length=100, overlap=10)
    
    print(f"Original length: {len(long_text)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
    print(f"Second chunk: {chunks[1]}")
    
    # Check if the overlap is correct
    # The end of chunk 0 should overlap with the start of chunk 1
    overlap_area = chunks[0][-10:]
    if chunks[1].startswith(overlap_area):
        print("STATUS: ✅ PASSED (Overlap confirmed)\n")
    else:
        print("STATUS: ⚠️  FAILED (Overlap not correct)\n")

except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")

print("--- TEST 6: Text Splitting (Short Text) ---")
try:
    short_text = "This is a short text."
    chunks = split_text(short_text, max_length=100, overlap=10)
    
    if len(chunks) == 1 and chunks[0] == short_text:
        print(f"Result: {chunks}")
        print("STATUS: ✅ PASSED (Handled short text correctly)\n")
    else:
        print(f"STATUS: ⚠️  FAILED (Incorrect short text handling)\n")
except Exception as e:
    print(f"STATUS: ❌ ERROR: {e}\n")

print("...Tests finished.")