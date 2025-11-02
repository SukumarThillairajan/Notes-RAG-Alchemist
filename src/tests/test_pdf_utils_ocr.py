import io
import pytest
from typing import Optional, List, Tuple

# Attempt to import optional dependencies for PDF/image creation
# We set flags here, but will import inside the function to satisfy linters.
try:
    import reportlab  # type: ignore[import]
    _reportlab_available = True
except ImportError:
    _reportlab_available = False

try:
    import PIL
    _pil_available = True
except ImportError:
    _pil_available = False

try:
    import img2pdf  # type: ignore[import]
    _img2pdf_available = True
except ImportError:
    _img2pdf_available = False

try:
    import pytesseract
    _tesseract_available = True
except ImportError:
    _tesseract_available = False

# Import the functions and classes to be tested
from utils.pdf_utils import extract_text_from_pdf, normalize_ocr_text, OCRPageMeta


def _create_dummy_pdf() -> Optional[bytes]:
    """Create a simple, single-page PDF in-memory with text."""
    if _reportlab_available:
        from reportlab.pdfgen import canvas  # type: ignore[import]
        from reportlab.lib.pagesizes import letter  # type: ignore[import]

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(100, 750, "NOZZLE A/A* ~")
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer.getvalue()

    if _pil_available and _img2pdf_available:
        import img2pdf  # type: ignore[import]
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (600, 800), color='white')
        d = ImageDraw.Draw(img)
        try:
            # Use a basic font if available
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        d.text((10, 10), "NOZZLE A/A* ~", fill='black', font=font)
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        pdf_bytes = img2pdf.convert(img_buffer.read())
        return pdf_bytes

    return None


@pytest.mark.skipif(not _tesseract_available, reason="pytesseract is not installed")
def test_extract_text_with_ocr_and_meta():
    """
    Test that OCR extraction returns correct metadata.
    """
    pdf_bytes = _create_dummy_pdf()
    if not pdf_bytes:
        pytest.skip("Could not create dummy PDF (reportlab or PIL+img2pdf missing)")

    page_texts, metas = extract_text_from_pdf(
        io.BytesIO(pdf_bytes),
        use_ocr=True,
        ocr_assume_uniform_block=True,
        return_meta=True
    )

    assert isinstance(page_texts, list)
    assert isinstance(metas, list)
    assert len(page_texts) == len(metas) >= 1

    meta = metas[0]
    assert isinstance(meta, OCRPageMeta)
    assert meta.used_ocr is True
    assert meta.psm_used in (3, 6)  # Should try 6, might fall back to 3
    assert isinstance(meta.ink_ratio, float)
    assert 0.0 <= meta.ink_ratio <= 1.0


@pytest.mark.skipif(not _tesseract_available, reason="pytesseract is not installed")
def test_extract_text_without_meta():
    """
    Test that calling extract_text_from_pdf without return_meta=True
    returns only a list of strings, ensuring backward compatibility.
    """
    pdf_bytes = _create_dummy_pdf()
    if not pdf_bytes:
        pytest.skip("Could not create dummy PDF (reportlab or PIL+img2pdf missing)")

    result = extract_text_from_pdf(io.BytesIO(pdf_bytes), use_ocr=True)

    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], str)


@pytest.mark.parametrize("sample, expected", [
    ("This is a word-\nwrap test.", "This is a wordwrap test."),
    ("Remove stray @ and ® tokens.", "Remove stray and tokens."),
    ("Unify “quotes” and ‘apostrophes’.", 'Unify "quotes" and \'apostrophes\'.'),
    ("Collapse hard\nline breaks.", "Collapse hard line breaks."),
    ("And   squeeze    multiple spaces.", "And squeeze multiple spaces."),
    ("\n\n  leading and trailing whitespace\t\n\n", "leading and trailing whitespace"),
])
def test_normalize_ocr_text(sample, expected):
    """Test various text normalization cases for the normalize_ocr_text function."""
    assert normalize_ocr_text(sample) == expected
