from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Literal, overload

import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageFilter, ImageOps
import fitz  # PyMuPDF for direct text extraction

cv2: Any
np: Any

try:
    import cv2 as _cv2  # type: ignore[import]
    cv2 = _cv2
except Exception:
    cv2 = None

try:
    import numpy as _np  # type: ignore[import]
    np = _np
except Exception:
    np = None

HAS_CV2 = cv2 is not None
HAS_NP = np is not None

if TYPE_CHECKING:
    try:
        from numpy.typing import NDArray
    except Exception:  # pragma: no cover - typing fallback
        NDArray = Any  # type: ignore[assignment]
else:
    NDArray = Any  # type: ignore[assignment]


_SENTENCE_BOUNDARY_PATTERN = r"(?<=[.!?])\s+"
_DEFAULT_OCR_PSM = 3


@dataclass
class OCRPageMeta:
    """Container for metadata about a single page's extraction process."""
    page_index: int
    psm_used: int
    ink_ratio: float
    noisy: bool
    used_ocr: bool


def normalize_ocr_text(s: str) -> str:
    """Clean up common OCR artifacts from a string."""
    if not s:
        return s
    # join hyphenated linebreaks: "trap-\n ezoidal" -> "trapezoidal"
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    # collapse hard linebreaks that aren't paragraph boundaries
    s = re.sub(r"[ \t]*\n(?!\n)", " ", s)
    # normalize stray OCR tokens
    s = s.replace("@", "").replace("\u00ae", "")  # add more if needed
    # unify quotes/dashes
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    # squeeze spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _resolve_psm(psm: int | None, assume_uniform_block: bool, default_psm: int = _DEFAULT_OCR_PSM) -> int:
    """Resolve the effective PSM based on overrides and heuristics."""
    if assume_uniform_block and psm is None:
        return 6
    if psm is not None:
        return psm
    return default_psm


def _looks_sparse(text: str, min_chars: int = 120) -> bool:
    """Return True when OCR output appears too short to be reliable."""
    return len((text or "").strip()) < min_chars


def _estimate_ink_ratio(
    np_gray: NDArray | None,
    thresh: NDArray | None,
    fallback_img: Image.Image | None = None,
) -> float:
    """Estimate the ratio of ink (black pixels) to total pixels."""
    if not (HAS_CV2 and HAS_NP) or np_gray is None or thresh is None:
        # Fallback for PIL-only path
        target_img = fallback_img
        if target_img is not None:
            gs_img = target_img.convert("L")
            dark_pixels = sum(1 for pixel in gs_img.getdata() if pixel < 200)
            total_pixels = gs_img.width * gs_img.height
            return dark_pixels / total_pixels if total_pixels > 0 else 0.0
        return 0.0

    # OpenCV path
    total_pixels = thresh.size
    black_pixels = total_pixels - cv2.countNonZero(thresh)
    return black_pixels / total_pixels if total_pixels > 0 else 0.0


def _deskew_cv(img_gray: NDArray) -> NDArray:
    """Deskew a grayscale image using OpenCV."""
    if not (HAS_CV2 and HAS_NP):
        return img_gray
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img_gray  # No content to deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _morphology(binary: NDArray, *, dilate_iter: int = 0, erode_iter: int = 0) -> NDArray:
    """Apply dilation then erosion."""
    if not HAS_NP:
        return binary
    kernel = np.ones((3, 3), np.uint8)
    if dilate_iter > 0:
        binary = cv2.dilate(binary, kernel, iterations=dilate_iter)
    if erode_iter > 0:
        binary = cv2.erode(binary, kernel, iterations=erode_iter)
    return binary


def _preprocess_for_ocr(
    img: Image.Image, *, dilate_iter: int = 1, erode_iter: int = 1
) -> Tuple[Image.Image, NDArray | None, NDArray | None]:
    """Preprocess image for OCR, returning intermediate arrays for metadata."""
    gs_img = img.convert("L")
    if HAS_CV2 and HAS_NP:
        np_gray = np.array(gs_img)
        deskewed_gray = _deskew_cv(np_gray)
        _, binary = cv2.threshold(deskewed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morphed = _morphology(binary, dilate_iter=dilate_iter, erode_iter=erode_iter)
        return Image.fromarray(morphed), deskewed_gray, morphed
    # Fallback path without OpenCV
    processed = ImageOps.autocontrast(gs_img)
    processed = ImageOps.invert(processed)
    processed = processed.filter(ImageFilter.MedianFilter(3))
    processed = processed.convert("1", dither=Image.Dither.NONE).convert("L")
    return processed, None, None


def ocr_page(
    img: Image.Image,
    *,
    lang: str = "eng",
    oem: int = 1,
    psm: int | None = None,
    assume_uniform_block: bool = False,
    extra_config: str | None = None,
) -> str:
    """Run Tesseract OCR on a PIL image with configurable segmentation mode."""
    psm_final = _resolve_psm(psm, assume_uniform_block)
    config_parts = [f"--oem {oem}", f"--psm {psm_final}", "-c preserve_interword_spaces=1"]
    if extra_config:
        config_parts.append(extra_config.strip())
    config = " ".join(config_parts)
    return pytesseract.image_to_string(img, lang=lang, config=config).strip()


@overload
def extract_text_from_pdf(
    pdf_file: str | io.BytesIO,
    use_ocr: bool = False,
    *,
    ocr_assume_uniform_block: bool = False,
    return_meta: Literal[True],
    ocr_psm: int | None = None,
    ocr_oem: int = 1,
    ocr_extra_config: str | None = None,
    ocr_dilate_iter: int = 1,
    ocr_erode_iter: int = 1,
    enable_preproc: bool = True,
) -> Tuple[List[str], List[OCRPageMeta]]: ...


@overload
def extract_text_from_pdf(
    pdf_file: str | io.BytesIO,
    use_ocr: bool = False,
    *,
    ocr_assume_uniform_block: bool = False,
    return_meta: Literal[False] = False,
    ocr_psm: int | None = None,
    ocr_oem: int = 1,
    ocr_extra_config: str | None = None,
    ocr_dilate_iter: int = 1,
    ocr_erode_iter: int = 1,
    enable_preproc: bool = True,
) -> List[str]: ...


def extract_text_from_pdf(
    pdf_file: str | io.BytesIO,
    use_ocr: bool = False,
    *,
    ocr_assume_uniform_block: bool = False,
    return_meta: bool = False,
    ocr_psm: int | None = None,
    ocr_oem: int = 1,
    ocr_extra_config: str | None = None,
    ocr_dilate_iter: int = 1,
    ocr_erode_iter: int = 1,
    enable_preproc: bool = True,
) -> List[str] | Tuple[List[str], List[OCRPageMeta]]:
    """
    Extract text from a PDF, optionally forcing OCR for every page.

    Args:
        pdf_file: Path to a PDF file or a BytesIO containing PDF bytes.
        use_ocr: If True, skip direct extraction and use OCR for every page.
        return_meta: If True, return a tuple of (page_texts, page_metadata).
        ocr_assume_uniform_block: Hint that each page is a single uniform block (runs PSM 6 first).
        ocr_psm: Explicit PSM override for Tesseract (None preserves the default heuristic).
        ocr_oem: OCR engine mode forwarded to Tesseract.
        ocr_extra_config: Additional config string appended to the Tesseract call.
        ocr_dilate_iter: Number of dilation iterations for morphology.
        ocr_erode_iter: Number of erosion iterations for morphology.
        enable_preproc: Whether to apply preprocessing before running OCR.

    Returns:
        - List[str]: Text per page, preserving page order.
        - (List[str], List[OCRPageMeta]): If `return_meta` is True.

    PSM quick reference: 3=Fully automatic, 4=Single column, 6=Uniform block, 11/12=Sparse text.
    """
    if isinstance(pdf_file, io.BytesIO):
        doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
    else:
        doc = fitz.open(pdf_file)

    page_texts: List[str] = []
    page_metas: List[OCRPageMeta] = []
    ocr_page_indices: List[int] = []

    with doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            extracted = "" if use_ocr else str(page.get_text()).strip()
            if extracted:
                page_texts.append(extracted)
                if return_meta:
                    page_metas.append(
                        OCRPageMeta(
                            page_index=page_index,
                            psm_used=ocr_psm or _resolve_psm(None, ocr_assume_uniform_block),
                            ink_ratio=1.0 if len(extracted) > 300 else 0.15,  # crude proxy
                            noisy=len(extracted.strip()) < 120,
                            used_ocr=False,
                        )
                    )
            else:
                page_texts.append("")
                ocr_page_indices.append(page_index)

    if not ocr_page_indices:
        final_texts = [normalize_ocr_text(text) for text in page_texts]
        if return_meta:
            # Update noisy flag based on final normalized text
            for i, meta in enumerate(page_metas):
                meta.noisy = len(final_texts[i].strip()) < 120
            return final_texts, page_metas
        return final_texts

    # Convert pages to images for OCR when direct extraction failed or was skipped.
    if isinstance(pdf_file, io.BytesIO):
        images = convert_from_bytes(pdf_file.getvalue(), first_page=1, last_page=len(page_texts))
    else:
        images = convert_from_path(pdf_file, first_page=1, last_page=len(page_texts))

    for index in ocr_page_indices:
        img = images[index]
        if enable_preproc:
            image_for_ocr, np_gray, np_binary = _preprocess_for_ocr(
                img, dilate_iter=ocr_dilate_iter, erode_iter=ocr_erode_iter
            )
        else:
            grayscale = img.convert("L")
            image_for_ocr, np_gray, np_binary = grayscale, None, None

        psm_final = _resolve_psm(ocr_psm, ocr_assume_uniform_block)
        primary_text = ocr_page(
            image_for_ocr,
            oem=ocr_oem,
            psm=psm_final,
            assume_uniform_block=False, # Already handled by psm_final
            extra_config=ocr_extra_config,
        )

        if ocr_assume_uniform_block:
            initial_psm = _resolve_psm(ocr_psm, True)
            if initial_psm == 6 and _looks_sparse(primary_text):
                fallback_text = ocr_page(
                    image_for_ocr,
                    oem=ocr_oem,
                    psm=3,
                    assume_uniform_block=False,
                    extra_config=ocr_extra_config,
                )
                if len(fallback_text.strip()) > len(primary_text.strip()):
                    primary_text = fallback_text
                    psm_final = 3

        page_texts[index] = primary_text
        if return_meta:
            ink_ratio = _estimate_ink_ratio(np_gray, np_binary, image_for_ocr)
            page_metas.append(
                OCRPageMeta(
                    page_index=index,
                    psm_used=psm_final,
                    ink_ratio=ink_ratio,
                    noisy=(ink_ratio < 0.02) or (len(primary_text.strip()) < 120),
                    used_ocr=True,
                )
            )

    final_texts = [normalize_ocr_text(text) for text in page_texts]
    if return_meta:
        return final_texts, page_metas
    return final_texts


def split_text(
    text: str,
    max_length: int = 1000,
    overlap: int = 120,
    hard_max: bool = True,
    min_chunk_len: int = 200,
) -> List[str]:
    """
    Split a long text into smaller, slightly overlapping chunks suitable for embeddings.

    Goals:
      - Prefer splitting on natural boundaries (blank lines, paragraph ends, sentence ends).
      - Preserve semantic coherence of derivations/equations where possible.
      - Add a small character overlap between consecutive chunks to avoid cutting formulas mid-derivation.

    Parameters:
      text         : The full input text (may include line breaks, equations).
      max_length   : Maximum characters per chunk (post-trim, excluding overlap).
      overlap      : Characters of backward overlap to prepend to the next chunk.
      hard_max     : If True, never exceed max_length; if False, allow slight overflow to honor boundary.
      min_chunk_len: Minimum characters for a chunk; merge tiny trailing segments into previous when possible.

    Returns:
      List[str]    : List of chunk strings in original order. Chunks are trimmed of leading/trailing whitespace.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= max_length:
        return [stripped]

    normalized = _normalize(text)
    paragraphs = _split_paragraphs(normalized)

    limit = max_length
    overflow_limit = max_length if hard_max else int(max_length * 1.1)

    base_chunks: List[str] = []
    buffer = ""

    for paragraph in paragraphs:
        para = paragraph.strip()
        if not para:
            continue

        if not buffer:
            candidate = para
        else:
            candidate = f"{buffer}\n{para}"

        if len(candidate) <= limit or (not hard_max and len(candidate) <= overflow_limit):
            buffer = candidate
            continue

        if buffer:
            base_chunks.append(buffer.strip())
            buffer = ""

        if len(para) <= limit or (not hard_max and len(para) <= overflow_limit):
            buffer = para
        else:
            subdivided = _subdivide(para, max_length, hard_max)
            for part in subdivided[:-1]:
                base_chunks.append(part.strip())
            buffer = subdivided[-1].strip()

    if buffer:
        base_chunks.append(buffer.strip())

    # Apply overlaps
    if overlap > 0 and len(base_chunks) > 1:
        overlapped: List[str] = []
        for idx, chunk in enumerate(base_chunks):
            ###chunk = chunk.strip()
            if idx == 0:
                overlapped.append(chunk)
                continue

            tail_source = overlapped[-1]
            tail = tail_source[-overlap:] if overlap < len(tail_source) else tail_source

            if tail and not tail.endswith(("\n", " ")) and not chunk.startswith((" ", "\n")):
                merged = f"{tail} {chunk}"
            elif tail and not tail.endswith("\n") and not chunk.startswith("\n"):
                merged = f"{tail}\n{chunk}"
            else:
                merged = f"{tail}{chunk}"

            ###merged = merged.strip()

            if hard_max and len(merged) > max_length:
                merged = merged[-max_length:]
            elif not hard_max and len(merged) > overflow_limit:
                merged = merged[-overflow_limit:]

            overlapped.append(merged)

        base_chunks = overlapped

    # Merge trailing chunk if it is too small.
    if len(base_chunks) >= 2 and len(base_chunks[-1]) < min_chunk_len:
        base_chunks[-2] = f"{base_chunks[-2]}\n{base_chunks[-1]}".strip()
        base_chunks.pop()

    return [chunk for chunk in base_chunks if chunk]


def _normalize(text: str) -> str:
    """Collapse excessive blank lines and trim trailing spaces per line."""
    collapsed = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in collapsed.splitlines()]
    return "\n".join(lines).strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split text on blank lines into paragraphs."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p for p in paragraphs if p.strip()]


def _subdivide(paragraph: str, max_len: int, hard_max: bool) -> List[str]:
    """Recursively split a paragraph using progressively weaker boundaries."""
    paragraph = paragraph.strip()
    if not paragraph:
        return []

    limit = max_len
    overflow_limit = max_len if hard_max else int(max_len * 1.1)
    effective_limit = overflow_limit if not hard_max else limit

    if len(paragraph) <= effective_limit:
        return [paragraph]

    splitters = [
        _split_on_newline,
        _split_on_sentence,
        _split_on_space,
    ]

    for splitter in splitters:
        parts = _attempt_split(paragraph, max_len, hard_max, splitter)
        if parts:
            result: List[str] = []
            for part in parts:
                result.extend(_subdivide(part, max_len, hard_max))
            return result

    # Final fallback: fixed width slicing to avoid infinite recursion.
    slices: List[str] = []
    idx = 0
    while idx < len(paragraph):
        end = idx + max_len
        slices.append(paragraph[idx:end])
        idx = end
    return slices

'''
def _attempt_split(
    text: str,
    max_len: int,
    hard_max: bool,
    splitter: Callable[[str, int], List[str]],
) -> List[str]:
    """Try splitting text using the provided splitter respecting limits."""
    limit = max_len
    overflow_limit = max_len if hard_max else int(max_len * 1.1)
    effective_limit = overflow_limit if not hard_max else limit

    if len(text) <= effective_limit:
        return [text]

    left, right = splitter(text, effective_limit)
    if not left or not right:
        return []

    return [left.strip(), right.strip()]
'''

def _attempt_split(
    text: str,
    max_len: int,
    hard_max: bool,
    splitter: Callable[[str, int], List[str]],
) -> List[str]:
    """Try splitting text using the provided splitter respecting limits."""
    limit = max_len
    overflow_limit = max_len if hard_max else int(max_len * 1.1)
    effective_limit = overflow_limit if not hard_max else limit

    if len(text) <= effective_limit:
        return [text]

    # --- THIS IS THE FIX ---
    # Call the splitter and store the result in a variable
    split_result = splitter(text, effective_limit)

    # Check if the result is valid *before* unpacking
    if not split_result:
        return []

    # Now it's safe to unpack
    left, right = split_result
    # --- END OF FIX ---

    if not left or not right:
        return []

    return [left.strip(), right.strip()]

def _split_on_newline(text: str, limit: int) -> List[str]:
    """Split text at the last newline before the limit."""
    idx = text.rfind("\n", 0, limit + 1)
    while idx != -1:
        head = text[:idx]
        tail = text[idx + 1 :]
        if head.strip() and not _contains_unclosed_math(head):
            return [head, tail]
        idx = text.rfind("\n", 0, idx)
    return []


def _split_on_sentence(text: str, limit: int) -> List[str]:
    """Split text at the last sentence boundary before the limit."""
    matches = list(re.finditer(_SENTENCE_BOUNDARY_PATTERN, text))
    for match in reversed(matches):
        if match.start() > limit:
            continue
        idx = match.end()
        head = text[:idx]
        tail = text[idx:]
        if head.strip() and tail.strip() and not _contains_unclosed_math(head):
            return [head, tail]
    return []


def _split_on_space(text: str, limit: int) -> List[str]:
    """Split text on the last whitespace before the limit, or fall back to slicing."""
    idx = -1
    for i in range(min(len(text) - 1, limit), -1, -1):
        if text[i].isspace():
            idx = i
            break
    if idx == -1:
        idx = min(limit, len(text) - 1)
    head = text[: idx + 1].rstrip()
    tail = text[idx + 1 :].lstrip()
    if not head or not tail or _contains_unclosed_math(head):
        return []
    return [head, tail]


def _contains_unclosed_math(fragment: str) -> bool:
    """Return True if the fragment ends inside an unfinished math block."""
    block_count = fragment.count("$$")
    if block_count % 2:
        return True

    singles = re.findall(r"(?<!\$)\$(?!\$)", fragment)
    if len(singles) % 2:
        return True

    open_brackets = fragment.count(r"\[")
    close_brackets = fragment.count(r"\]")
    if open_brackets > close_brackets:
        return True

    return False
