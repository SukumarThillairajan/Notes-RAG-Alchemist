"""Compiler worker that assembles planner results and excerpts into a PDF report."""

from __future__ import annotations

import os
from pathlib import Path
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from fpdf import FPDF

FontStyle = Literal["", "B", "I", "U", "BI", "BU", "IU", "BIU"]

LINE_H = 6.5          # base line height
PARA_SPACING = 2.0    # space after paragraph
INDENT_BULLET = 6.0   # hanging indent for bullets (mm)
INDENT_BLOCK = 8.0    # indent for quoted/context blocks (mm)
BULLET_GAP = 2.5      # gap between bullet glyph and text (mm)
PAGE_BREAK_GUARD = 40 # mm; break before headings if less space


def _safe_text(s: str) -> str:
    """
    A light final-pass normalizer to ensure no odd OCR artifacts make it into the PDF.
    This is a safety net, as primary normalization should happen upstream.
    """
    if not s:
        return ""
    # join hyphenated linebreaks: "trap-\n ezoidal" -> "trapezoidal"
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    # collapse hard linebreaks that aren’t paragraph boundaries
    s = re.sub(r"[ \t]*\n(?!\n)", " ", s)
    # normalize stray OCR tokens and unify quotes/dashes
    s = s.replace("@", "").replace("®", "").replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


_STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with", "by", "at", "from",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "about", "into", "over", "under", "as", "it", "its", "their", "our", "your", "my",
}


def _clean_text_for_title(text: str) -> str:
    """Normalize whitespace and strip common unicode quotes for title generation."""
    t = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _top_words(text: str, k: int = 6) -> List[str]:
    """Return the top-k keyword candidates from a text."""
    t = _clean_text_for_title(text.lower())
    words = re.findall(r"[a-z][a-z\-]+", t)
    keep = [w for w in words if w not in _STOPWORDS and len(w) >= 3]
    freq: Dict[str, int] = {}
    for w in keep:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))][:k]


def _short_title_from_segment(segment_text: str, max_words: int = 5) -> str:
    """
    Produce a concise 3–5 word title from a segment_text, heuristic only (no LLM).
    Prefer top keywords; if empty, fall back to the first clause trimmed.
    """
    candidates = _top_words(segment_text, k=8)
    if candidates:
        title_words = candidates[:max_words]
        if len(title_words) < 3:
            fallback_words = _clean_text_for_title(segment_text).split()[:max_words]
            for word in fallback_words:
                if len(title_words) >= max_words:
                    break
                lower_word = word.lower()
                if lower_word not in title_words:
                    title_words.append(lower_word)
        return " ".join(w.capitalize() for w in title_words if w)

    first_line = _clean_text_for_title(segment_text).split(".")[0]
    words = first_line.split()[:max_words]
    if not words:
        return "Untitled Segment"
    return " ".join(words)


def _derive_lecture_title(all_segments: List[str], fallback: str = "Lecture Notes - Matched Excerpts") -> str:
    """Build a 3–5 word lecture title from the union of segment keywords."""
    bag: Dict[str, int] = {}
    for seg in all_segments:
        for w in _top_words(seg, k=10):
            bag[w] = bag.get(w, 0) + 1
    if not bag:
        fallback_source = all_segments[0] if all_segments else fallback
        return _short_title_from_segment(fallback_source, 5)
    best = [w for w, _ in sorted(bag.items(), key=lambda x: (-x[1], x[0]))][:5]
    if len(best) < 3 and all_segments:
        fallback_words = _clean_text_for_title(all_segments[0]).split()[:5]
        for word in fallback_words:
            if len(best) >= 3:
                break
            lower_word = word.lower()
            if lower_word not in best:
                best.append(lower_word)
    return " ".join(w.capitalize() for w in best) if best else fallback


def _infer_title(segments: list[str], fallback: str = "Lecture Notes - Matched Excerpts") -> str:
    """Backward-compatible wrapper for legacy title inference."""
    return _derive_lecture_title(segments, fallback=fallback)

def _draw_toc_header(pdf: FPDF, text: str = "Table of Contents") -> None:
    _set_font(pdf, "LiberationSans", "B", 14)
    pdf.ln(2)
    pdf.cell(0, 8, text=text, ln=1)
    pdf.ln(2)


def _draw_toc_line(
    pdf: FPDF,
    title: str,
    page_no: int,
    *,
    left_margin: float | None = None,
    right_margin: float | None = None,
) -> None:
    """Print a single ToC line with dot leaders and right-aligned page number."""
    _set_font(pdf, "LiberationSans", "", 11)
    if left_margin is None:
        left_margin = pdf.l_margin
    if right_margin is None:
        right_margin = pdf.r_margin

    usable_w = pdf.w - left_margin - right_margin
    page_str = str(page_no)
    page_w = pdf.get_string_width(page_str)
    dot = "."
    dot_w = max(pdf.get_string_width(dot), 0.5)

    max_title_w = usable_w - (page_w + 6)
    display_title = title
    while pdf.get_string_width(display_title) > max_title_w and len(display_title) > 4:
        display_title = display_title[:-1]

    title_w = pdf.get_string_width(display_title)
    dots_needed = max(2, int((usable_w - title_w - page_w) // dot_w))
    leaders = dot * dots_needed

    y = pdf.get_y()
    pdf.set_x(left_margin)
    pdf.cell(0, 6, text=f"{display_title} {leaders} {page_str}", ln=1)
    pdf.set_y(y + 6)



def _find_font_path(fname: str, extra_dir_envs: list[str]) -> str | None:
    """Return absolute path to a font file if found; else None."""
    # Env var overrides
    for env_name in extra_dir_envs:
        d = os.getenv(env_name)
        if d:
            p = Path(d) / fname
            if p.is_file():
                return str(p)

    here = Path(__file__).resolve().parent
    candidates = [
        here / "fonts" / "liberation" / fname,  # workers/fonts/liberation/...
        here / "fonts" / fname,                 # workers/fonts/...
        here.parent / "fonts" / fname,          # <project>/fonts/...
        Path.cwd() / fname,                     # cwd
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _try_add_family(pdf: FPDF, family_name: str, files: dict[str, str]) -> bool:
    """
    Attempt to add a font family. `files` is a dict with keys "", "B", "I", "BI".
    Returns True on success, False otherwise.
    """
    try:
        # fpdf2 supports uni=True; PyFPDF ignores it gracefully
        pdf.add_font(family_name, "",  files[""],  uni=True)
        pdf.add_font(family_name, "B", files["B"], uni=True)
        pdf.add_font(family_name, "I", files["I"], uni=True)
        pdf.add_font(family_name, "BI", files["BI"], uni=True)
        return True
    except Exception:
        return False


def _set_font(pdf: FPDF, family: str = "LiberationSans", style: FontStyle = "", size: int = 12) -> None:
    """
    Set a Unicode font if available (LiberationSans preferred, then DejaVu).
    Fall back to core Helvetica if neither is bundled.
    """
    chosen_family = family
    if not hasattr(pdf, "_unicode_fonts_added"):
        # Try Liberation Sans first
        lib_files = {
            "":  _find_font_path("LiberationSans-Regular.ttf", ["LIBERATION_FONTS_DIR"]),
            "B": _find_font_path("LiberationSans-Bold.ttf",    ["LIBERATION_FONTS_DIR"]),
            "I": _find_font_path("LiberationSans-Italic.ttf",  ["LIBERATION_FONTS_DIR"]),
            "BI":_find_font_path("LiberationSans-BoldItalic.ttf", ["LIBERATION_FONTS_DIR"]),
        }
        liberation_ok = False
        if all(lib_files.values()):
            # all() check ensures no values are None, so we can safely cast
            liberation_ok = _try_add_family(pdf, "LiberationSans", lib_files)  # type: ignore

        # If Liberation missing, try DejaVu
        if not liberation_ok:
            djv_files = {
                "":  _find_font_path("DejaVuSans.ttf",             ["DEJAVU_FONTS_DIR"]),
                "B": _find_font_path("DejaVuSans-Bold.ttf",        ["DEJAVU_FONTS_DIR"]),
                "I": _find_font_path("DejaVuSans-Oblique.ttf",     ["DEJAVU_FONTS_DIR"]),
                "BI":_find_font_path("DejaVuSans-BoldOblique.ttf", ["DEJAVU_FONTS_DIR"]),
            }
            dejavu_ok = False
            if all(djv_files.values()):
                dejavu_ok = _try_add_family(pdf, "DejaVu", djv_files)  # type: ignore
            if dejavu_ok:
                chosen_family = "DejaVu"
            else:
                chosen_family = "Helvetica"
        else:
            chosen_family = "LiberationSans"

        setattr(pdf, "_unicode_fonts_added", True)
        setattr(pdf, "_unicode_family", chosen_family)

    # Attempt to use the chosen family; otherwise hard-fallback to Helvetica
    try:
        pdf.set_font(getattr(pdf, "_unicode_family", chosen_family), style=style, size=size)
    except Exception:
        pdf.set_font("Helvetica", style=style, size=size)


def _norm_ocr_text(text: str) -> str:
    """
    Normalize OCR line breaks:
    - Replace special Unicode characters with ASCII equivalents.
    - Collapse 3+ newlines -> 2 (paragraph break)
    - Convert single newlines that look like hard wraps into spaces
    - Trim leading/trailing whitespace
    """
    if not text:
        return ""
    # Character replacement for common unsupported glyphs
    replacements = {
        "’": "'", "‘": "'", "”": '"', "“": '"',
        "–": "-", "—": "-",
        "𝑛": "n",  # Mathematical Italic Small N -> n
        # Add other replacements here as needed
    }
    text = "".join(replacements.get(char, char) for char in text)
    t = text.replace("\r", "")
    # Keep paragraph breaks (double newline), soften single newlines
    # Heuristic: single newline not preceded/followed by empty lines -> space
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Hyphenation fix: "-\nword" -> "word"
    t = re.sub(r"-\n(\w)", r"\1", t)
    # Soft wrap: non-empty line followed by non-empty -> space
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # Collapse whitespace runs
    t = re.sub(r"[ \t]{2,}", " ", t)
    # Trim each paragraph
    t = "\n\n".join(p.strip() for p in t.split("\n\n"))
    return t.strip()


def _split_paragraphs(text: str) -> list[str]:
    # split on blank lines OR headings/bullets; keep bullets as their own paragraphs
    text = text.replace('\r', '')
    paras = re.split(r'\n\s*\n', text)  # paragraph breaks on blank lines
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # further split if many bullets in one line
        bullets = re.split(r'(?:^|\n)\s*(?:[-•]|\d+\.)\s+', p)
        if len(bullets) > 1:
            # re-add markers for each bullet item
            for b in bullets:
                b = b.strip()
                if b:
                    out.append(f"• {b}")
        else:
            out.append(p)
    return out


def _write_paragraph(
    pdf: FPDF, text: str, *, indent: float = 0.0, style: FontStyle = "", size: int = 12, lh: float = LINE_H
) -> None:
    """Write a normalized paragraph with optional left indent."""
    _set_font(pdf, "LiberationSans", style, size)
    txt = _safe_text(text) # _norm_ocr_text is too aggressive for short lines like citations

    if indent > 0:
        pdf.set_x(pdf.l_margin + indent)
        width = getattr(pdf, "epw", pdf.w - pdf.l_margin - pdf.r_margin) - indent
    else:
        width = 0 # extend to right margin
        pdf.set_x(pdf.l_margin) # Explicitly reset X position
    pdf.multi_cell(width, lh, txt)


def _write_bullet(
    pdf: FPDF, text: str, *, bullet: str = "•", indent: float = INDENT_BULLET, gap: float = BULLET_GAP, size: int = 11, lh: float = LINE_H
) -> None:
    """
    Draw a bullet with hanging indent:
    [bullet]  First line...
              wrapped line continues here...
    """
    if pdf.get_y() > pdf.h - PAGE_BREAK_GUARD:
        pdf.add_page()
    _set_font(pdf, "LiberationSans", "", size)
    # Measure bullet cell width; we’ll treat it as fixed gap
    left_x = pdf.l_margin
    full_w = getattr(pdf, "epw", pdf.w - pdf.l_margin - pdf.r_margin)
    # First line: draw bullet cell, then a multicell for the text with reduced width
    bullet_cell_w = indent  # use indent as the bullet box width
    # Bullet glyph
    pdf.set_x(left_x)
    pdf.cell(bullet_cell_w, lh, bullet, align="C")
    # Text box width & position
    text_x = left_x + bullet_cell_w + gap
    text_w = full_w - bullet_cell_w - gap
    txt = _safe_text(_norm_ocr_text(text))
    # FPDF multi_cell always resets X; we need to re-set X before each line.
    # Strategy: temporarily adjust L-margin by (bullet_cell_w + gap), then restore.
    old_lm = pdf.l_margin
    pdf.set_left_margin(text_x)
    pdf.set_x(text_x)
    pdf.multi_cell(text_w, lh, txt)
    pdf.set_left_margin(old_lm)
    pdf.ln(PARA_SPACING)


def _write_block(pdf: FPDF, raw_text: str, *, style: FontStyle = "", size: int = 12, lh: float = LINE_H, indent: float = 0.0):
    """Writes a block of text, splitting it into paragraphs and handling bullets."""
    for para in _split_paragraphs(raw_text):
        _write_paragraph(pdf, para, style=style, size=size, lh=lh, indent=indent)
        pdf.ln(1)


def _ensure_space_for_heading(pdf: FPDF, min_space_mm: float = PAGE_BREAK_GUARD) -> None:
    _prebreak_check(pdf, needed_mm=min_space_mm)


def _prebreak_check(pdf: FPDF, needed_mm: float = PAGE_BREAK_GUARD) -> None:
    if pdf.get_y() > pdf.h - needed_mm:
        pdf.add_page()


def _normalize_page_fields(
    page: int | str | None,
    pages: Sequence[int | str] | None,
) -> tuple[int | None, list[int]]:
    """Coerce page/page-list metadata into integers, dropping invalid values."""
    page_int: int | None = None
    if page is not None:
        try:
            page_int = int(str(page).strip())
        except (TypeError, ValueError):
            page_int = None

    normalized_pages: list[int] = []
    if pages:
        for item in pages:
            try:
                value = int(str(item).strip())
            except (TypeError, ValueError):
                continue
            if value not in normalized_pages:
                normalized_pages.append(value)
        normalized_pages.sort()

    return page_int, normalized_pages


def _compact_page_list(pages: Sequence[int]) -> str:
    """Summarise pages into comma-separated entries with ranges for consecutive numbers."""
    if not pages:
        return ""
    nums = sorted(pages)
    ranges: list[tuple[int, int]] = []
    start = nums[0]
    prev = nums[0]

    for num in nums[1:]:
        if num == prev + 1:
            prev = num
            continue
        ranges.append((start, prev))
        start = num
        prev = num
    ranges.append((start, prev))

    parts: list[str] = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}–{end}")
    return ", ".join(parts)


def _format_page_fragment(page_int: int | None, pages_list: list[int]) -> str:
    """Return the citation fragment (e.g., 'p. 12' or 'pp. 12–13')."""
    if pages_list:
        unique_pages = sorted(set(pages_list))
        if len(unique_pages) >= 2:
            return f"pp. {unique_pages[0]}–{unique_pages[-1]}"
        return f"p. {unique_pages[0]}"
    if page_int is not None:
        return f"p. {page_int}"
    return ""


def format_citation(
    book: str,
    page: int | str | None = None,
    pages: Sequence[int | str] | None = None,
) -> str:
    """Format a Source+page citation string for excerpts and appendix entries."""
    page_int, pages_list = _normalize_page_fields(page, pages)
    fragment = _format_page_fragment(page_int, pages_list)
    if fragment:
        return f"(Source: {book}, {fragment})"
    return f"(Source: {book})"


FIG_PATTERN = re.compile(r"(?i)\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)*)(?:[a-z])?")


def _score_value(value: Any) -> float:
    """Best-effort float conversion used for sorting excerpts by score."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _natural_sort_key(text: str) -> list[Any]:
    """Produce a natural sort key that splits digit runs into integers."""
    tokens = re.split(r"(\d+)", text)
    key: list[Any] = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token.lower())
    return key


def _gather_fig_refs(
    text: str,
    book: str,
    page: int | str | None,
    pages: Sequence[int | str] | None,
    store: set[tuple[str, str, tuple[int, ...], int | None]],
) -> None:
    """Extract figure references from excerpt text and cache them for the appendix."""
    if not text:
        return
    page_int, pages_list = _normalize_page_fields(page, pages)
    for match in FIG_PATTERN.finditer(text):
        figure_id = match.group(1)
        store.add((figure_id, book, tuple(pages_list), page_int))


def _trim_excerpt(text: str, limit: int = 4000) -> str:
    """Trim overly long excerpts so the PDF remains readable."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _write_plan_table(pdf: FPDF, segment_plans: Sequence[dict[str, Any]]) -> None:
    _set_font(pdf, "LiberationSans", "B", 12)
    pdf.cell(0, 8, "Planner Decisions", ln=1)
    _set_font(pdf, "LiberationSans", "", 10)
    # Column widths
    full_w = getattr(pdf, "epw", pdf.w - pdf.l_margin - pdf.r_margin)
    w1, w2, w3 = 20, 60, full_w - 20 - 60  # Segment, Decision, Trace/Score
    pdf.set_x(pdf.l_margin)
    pdf.cell(w1, 6, "Seg", border=0)
    pdf.cell(w2, 6, "Decision", border=0)
    pdf.cell(w3, 6, "Trace | Best", border=0, ln=1)
    for idx, plan in enumerate(segment_plans, start=1):
        dec = plan.get("decision") or ""
        trace = plan.get("trace") or ""
        score = plan.get("best_score")
        score_str = "-" if score is None else f"{float(score):.2f}"
        pdf.set_x(pdf.l_margin)
        pdf.cell(w1, 6, str(idx), border=0)
        pdf.cell(w2, 6, dec, border=0)
        pdf.cell(w3, 6, f"{trace} | {score_str}", border=0, ln=1)
    pdf.ln(4)


def _render_toc(
    pdf: FPDF,
    entries: list[tuple[str, int]],
    toc_page: int | None,
    return_page: int | None,
) -> int | None:
    """Render the table of contents either at the reserved page or append at the end."""
    if not entries:
        return toc_page

    set_page_func = getattr(pdf, "set_page", None)
    if callable(set_page_func) and toc_page:
        try:
            pdf.set_page(toc_page)  # type: ignore[attr-defined]
            top_margin = getattr(pdf, "t_margin", pdf.l_margin)
            pdf.set_y(top_margin)
            _draw_toc_header(pdf)
            for idx, (title, page_no) in enumerate(entries, start=1):
                _draw_toc_line(pdf, f"{idx}. {title}", page_no)
            if return_page:
                pdf.set_page(return_page)  # type: ignore[attr-defined]
            return toc_page
        except Exception:
            if return_page:
                pdf.set_page(return_page)  # type: ignore[attr-defined]

    pdf.add_page()
    _draw_toc_header(pdf, "Table of Contents (moved to end)")
    for idx, (title, page_no) in enumerate(entries, start=1):
        _draw_toc_line(pdf, f"{idx}. {title}", page_no)
    return pdf.page_no()


def _write_segment(
    pdf: FPDF,
    index: int,
    plan: dict[str, Any],
    figure_refs: set[tuple[str, str, tuple[int, ...], int | None]],
    *,
    toc: list[tuple[str, int]] | None,
    segment_title: str,
    show_hidden_note: bool,
    max_excerpts_per_segment: int,
) -> None:
    """Render a single segment section with excerpts and citations."""
    segment_raw = plan.get("segment_text") or ""
    segment_text = _safe_text(segment_raw)

    _prebreak_check(pdf, needed_mm=40)
    # In Pass 1, toc is an empty list that we populate.
    # In Pass 2, toc is an empty list so we don't re-record.
    if toc is not None:
        start_page = pdf.page_no()
        toc.append((segment_title, start_page))

    _set_font(pdf, "LiberationSans", "B", 12)
    pdf.cell(0, 8, f"Segment {index}: {segment_title}", ln=1)

    _write_block(pdf, f"Lecture segment: {segment_text}", style="I", size=11, indent=INDENT_BLOCK)
    pdf.ln(2)

    excerpts = plan.get("excerpts") or []
    sorted_excerpts = sorted(excerpts, key=lambda ex: _score_value(ex.get("score")), reverse=True)
    if max_excerpts_per_segment >= 0:
        displayed_excerpts = sorted_excerpts[:max_excerpts_per_segment]
    else:
        displayed_excerpts = sorted_excerpts

    if displayed_excerpts:
        glue = "Below are closely-matching textbook excerpts supporting this segment."
        _write_paragraph(pdf, glue, style="I", size=11)
        pdf.ln(2)

        for excerpt in displayed_excerpts:
            trimmed = _trim_excerpt(_safe_text(excerpt.get("text", "")))
            _write_block(pdf, trimmed, size=11, indent=INDENT_BULLET)

            raw_page = excerpt.get("page")
            raw_pages = excerpt.get("pages")
            try:
                page_int = int(str(raw_page).strip()) if raw_page is not None else None
            except (TypeError, ValueError):
                page_int = None
            if isinstance(raw_pages, (list, tuple)):
                cleaned_pages: List[int] = []
                for item in raw_pages:
                    text_val = str(item).strip()
                    if text_val.isdigit():
                        cleaned_pages.append(int(text_val))
                pages_list = sorted(set(cleaned_pages)) if cleaned_pages else None
            else:
                pages_list = None

            citation = format_citation(
                excerpt.get("book", "Unknown Book"),
                page_int,
                pages_list,
            )
            _write_paragraph(pdf, citation, indent=INDENT_BLOCK, style="I", size=10)

            score = excerpt.get("score")
            value = _score_value(score)
            if value != float("-inf"):
                pdf.set_text_color(120, 120, 120)
                _write_paragraph(pdf, f"(score={value:.2f})", indent=INDENT_BLOCK, size=9)
                pdf.set_text_color(0, 0, 0)

            _gather_fig_refs(
                excerpt.get("text") or "",
                excerpt.get("book", "Unknown Book"),
                page_int,
                pages_list,
                figure_refs,
            )
            pdf.ln(1)

        hidden_count = max(0, len(sorted_excerpts) - len(displayed_excerpts))
        if hidden_count and show_hidden_note:
            _write_paragraph(pdf, f"(+{hidden_count} more hidden)", style="I", size=10, indent=INDENT_BLOCK, lh=6)
    else:
        if plan.get("decision") == "EXTERNAL" and plan.get("suggested_book"):
            message = "[No excerpt found in provided references; suggested external source: "
            message += str(plan.get("suggested_book"))
            message += "]"
            _write_paragraph(pdf, _safe_text(message), style="I", size=11, indent=INDENT_BLOCK)
        elif plan.get("decision") != "ERROR":
            _write_paragraph(pdf, "[No relevant excerpt found in provided references]", style="I", size=11, indent=INDENT_BLOCK)
    pdf.ln(4)



def _figure_pointer_line(
    figure_id: str,
    book: str,
    page_int: int | None,
    pages_tuple: tuple[int, ...],
) -> str:
    """Format a single line for the Figure Pointers appendix."""
    pages_list = list(pages_tuple)
    fragment = _format_page_fragment(page_int, pages_list)
    if fragment:
        return f"Figure {figure_id} - see {book}, {fragment}"
    return f"Figure {figure_id} - see {book}"


class PDFReport(FPDF):
    """Custom PDF with header/footer and sensible defaults."""

    def __init__(self, title: str, author: str) -> None:
        super().__init__()
        self.title = title
        self.author = author
        self.set_title(self.title)
        self.set_author(self.author)
        self.set_margins(15, 15, 15)        # left, top, right
        self.set_auto_page_break(auto=True, margin=15)
        # Use set_cell_margin if available (fpdf2), otherwise ignore
        if hasattr(self, "set_cell_margin"):
            self.set_cell_margin(1.5)  # type: ignore[attr-defined]
        elif hasattr(self, "set_c_margin"):  # older fpdf2
            self.set_c_margin(1.5)  # type: ignore[attr-defined]


    def header(self) -> None:  # type: ignore[override]
        _set_font(self, "LiberationSans", "B", 12)
        self.cell(0, 8, self.title or "", border=0, ln=1, align="C")
        self.ln(2)

    def footer(self) -> None:  # type: ignore[override]
        self.set_y(-15)
        _set_font(self, "LiberationSans", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _add_title_page(pdf: FPDF, title: str) -> None:
    """Write the title page."""
    pdf.add_page()
    _set_font(pdf, "LiberationSans", "B", 18)
    pdf.cell(0, 12, title, ln=1, align="C")
    pdf.ln(10)
    _set_font(pdf, "LiberationSans", "", 12)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 8, f"Generated on: {now}", ln=1, align="C")
    pdf.ln(20)


def _write_figure_appendix(
    pdf: FPDF,
    figure_refs: set[tuple[str, str, tuple[int, ...], int | None]],
) -> None:
    """Append the figure pointer section."""
    pdf.add_page()
    _set_font(pdf, "LiberationSans", "B", 12)
    pdf.cell(0, 8, "Figure Pointers", ln=1)
    pdf.ln(2)

    if not figure_refs:
        _write_paragraph(pdf, "(No figure references found in excerpts)", style="I", size=11)
        return

    entries = sorted(
        figure_refs,
        key=lambda item: (item[1].lower(), _natural_sort_key(item[0])),
    )
    for figure_id, book, pages_tuple, page_int in entries:
        line = _figure_pointer_line(figure_id, book, page_int, pages_tuple)
        _write_paragraph(pdf, line, size=11)
    pdf.ln(2)


def _build_pdf(
    segment_plans: list[dict[str, Any]],
    *,
    title: str,
    author: str,
    include_plan_table: bool,
    show_hidden_note: bool,
    max_excerpts_per_segment: int,
    toc_page_num: int = 3,
    prompt_logger: Optional[Callable[[str], None]] = None,
) -> PDFReport:
    """Create the PDF report in-memory and return the populated FPDF object."""
    segment_texts = [plan.get("segment_text", "") for plan in segment_plans]
    lecture_title = _derive_lecture_title(segment_texts, fallback=title)

    pdf = PDFReport(title=lecture_title, author=author)
    setattr(pdf, "_lecture_title", lecture_title)
    _add_title_page(pdf, lecture_title)

    pdf.add_page()
    if include_plan_table and segment_plans:
        _write_plan_table(pdf, segment_plans)
        pdf.ln(4)
    else:
        _set_font(pdf, "LiberationSans", "I", 11)
        pdf.cell(0, 6, "Planner decisions omitted in this report.", ln=1)
        pdf.ln(4)

    while pdf.page_no() < toc_page_num:
        pdf.add_page()
    toc_page = pdf.page_no()

    toc_entries: list[tuple[str, int]] = []
    figure_refs: set[tuple[str, str, tuple[int, ...], int | None]] = set()

    for idx, plan in enumerate(segment_plans, start=1):
        short_title = _short_title_from_segment(plan.get("segment_text", ""), max_words=5)
        _write_segment(
            pdf,
            idx,
            plan,
            figure_refs,
            toc=toc_entries,
            segment_title=short_title,
            show_hidden_note=show_hidden_note,
            max_excerpts_per_segment=max_excerpts_per_segment,
        )

    _write_figure_appendix(pdf, figure_refs)
    last_page = pdf.page_no()
    actual_toc_page = _render_toc(pdf, toc_entries, toc_page, last_page)

    setattr(pdf, "_toc_entries", toc_entries)
    setattr(pdf, "_toc_page", actual_toc_page)

    return pdf

def compile_report_bytes(
    segment_plans: list[dict[str, Any]],
    *,
    title: str = "Lecture Notes - Matched Excerpts",
    author: str = "Lecture Notes Assistant",
    show_hidden_note: bool = True,
    include_plan_table: bool = True,
    max_excerpts_per_segment: int = 4,
    toc_page: int = 3,
    prompt_logger: Optional[Callable[[str], None]] = None,
) -> bytes:
    """Compile segment plans into a PDF and return the binary content."""
    pdf = _build_pdf(
        segment_plans,
        title=title,
        author=author,
        include_plan_table=include_plan_table,
        show_hidden_note=show_hidden_note,
        max_excerpts_per_segment=max_excerpts_per_segment,
        toc_page_num=toc_page,
        prompt_logger=prompt_logger,
    )
    buf = pdf.output(dest="S")
    if isinstance(buf, bytearray):
        return bytes(buf)
    if isinstance(buf, bytes):
        return buf
    return buf.encode("latin-1", "replace")


def compile_report(
    segment_plans: list[dict[str, Any]],
    output_pdf_path: str,
    *,
    title: str = "Lecture Notes - Matched Excerpts",
    author: str = "Lecture Notes Assistant",
    show_hidden_note: bool = True,
    include_plan_table: bool = True,
    max_excerpts_per_segment: int = 4,
    toc_page: int = 3,
    prompt_logger: Optional[Callable[[str], None]] = None,
) -> None:
    """Compile segment plans into a PDF saved to disk."""
    pdf = _build_pdf(
        segment_plans,
        title=title,
        author=author,
        include_plan_table=include_plan_table,
        show_hidden_note=show_hidden_note,
        max_excerpts_per_segment=max_excerpts_per_segment,
        toc_page_num=toc_page,
        prompt_logger=prompt_logger,
    )
    pdf.output(output_pdf_path)

if __name__ == "__main__":
    long_paragraph = (
        "The attention mechanism provides an overview of how different parts of the input sequence are weighted. "
        "This is a very long excerpt designed to test the multi_cell function in FPDF and ensure that text wraps correctly "
        "to the next line instead of overflowing off the page. We will repeat this sentence a few times to make sure it is long enough. "
    )

    smoke_plans: List[dict[str, Any]] = [
        {
            "segment_text": "Transformer encoders rely on scaled dot-product attention and residual connections." * 2,
            "decision": "NORMAL",
            "trace": "Normal search",
            "best_score": 0.88,
            "excerpts": [
                {
                    "text": long_paragraph + "See Fig. 3.2(a).",
                    "book": "Deep Learning 101",
                    "page": 10,
                    "pages": ["10", "11"],
                    "score": 0.92,
                },
                {
                    "text": "Detailed derivation with Fig. 3.2(b) illustrating alignment.",
                    "book": "Deep Learning 101",
                    "page": 12,
                    "pages": ["12"],
                    "score": 0.85,
                },
            ],
        },
        {
            "segment_text": "Layer normalization complements the attention block while Fig. 4.1 depicts residual paths." * 2,
            "decision": "NORMAL",
            "trace": "Normal search",
            "best_score": 0.83,
            "excerpts": [
                {
                    "text": "Layer normalization stabilizes training by re-centering the activations.",
                    "book": "Deep Learning 101",
                    "page": 13,
                    "pages": ["13"],
                    "score": 0.83,
                }
            ],
        },
        {
            "segment_text": "A segment about a topic with no good matches in the provided references.",
            "decision": "EXTERNAL",
            "suggested_book": "Attention Is All You Need",
            "trace": "External book suggested",
            "best_score": None,
            "excerpts": [],
        },
    ]

    print("Running compiler smoke test...")

    pdf = _build_pdf(
        smoke_plans,
        title="Lecture Notes - Matched Excerpts",
        author="Lecture Notes Assistant",
        include_plan_table=True,
        show_hidden_note=True,
        max_excerpts_per_segment=3,
    )
    pdf.output("tmp_compiler_toc_test.pdf")

    toc_entries = getattr(pdf, "_toc_entries", [])
    toc_page = getattr(pdf, "_toc_page", None)
    lecture_title = getattr(pdf, "_lecture_title", "<none>")

    print("Title detected:", lecture_title)
    print("ToC page recorded:", toc_page)
    print("First three ToC lines:", [entry[0] for entry in toc_entries[:3]])

    compile_report(
        smoke_plans,
        "compiler_demo.pdf",
        max_excerpts_per_segment=3,
    )

    pdf_bytes = compile_report_bytes(
        smoke_plans,
        max_excerpts_per_segment=3,
    )

    print("Generated demo PDF bytes:", len(pdf_bytes))
    print("Wrote compiler_demo.pdf")
