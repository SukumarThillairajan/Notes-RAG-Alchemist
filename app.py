from dotenv import load_dotenv

load_dotenv()

import hashlib
import io
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, Literal, overload

import streamlit as st

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
else:
    # Fallback for older Streamlit versions or when type checking is not running
    UploadedFile = Any
# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import utilities and workers
from utils.pdf_utils import OCRPageMeta, extract_text_from_pdf, split_text  # noqa: E402

try:  # Prefer top-level module name if available
    from embedding_indexer import init_pinecone_index, index_chunks  # type: ignore  # noqa: E402
except ImportError:  # Fallback to package path
    from utils.embedding_indexer import init_pinecone_index, index_chunks  # type: ignore  # noqa: E402 #

from planner.planner_agent import DEFAULT_MODELS, PlannerConfig, plan_segment  # noqa: E402
from workers.retriever import RetrieverWorker  # noqa: E402
from workers.compiler import compile_report_bytes  # noqa: E402


FIG_PATTERN = re.compile(r"(?i)\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)*)(?:[a-z])?")

def file_sha1(uploaded_file: UploadedFile) -> str:
    """Return SHA-1 hash of uploaded file bytes."""
    return hashlib.sha1(uploaded_file.getvalue()).hexdigest()  # noqa: S324


@overload
def cached_extract_pages(
    pdf_bytes: bytes,
    force_ocr: bool,
    *,
    assume_uniform: bool = False,
    return_meta: Literal[True],
    psm_override: int | None = None,
    enable_preproc: bool = True,
    ocr_oem: int = 1,
    ocr_dilate: int = 1,
    ocr_erode: int = 1,
    ocr_extra_config: str | None = None,
) -> Tuple[List[str], List[OCRPageMeta]]: ...


@overload
def cached_extract_pages(
    pdf_bytes: bytes,
    force_ocr: bool,
    *,
    assume_uniform: bool = False,
    return_meta: Literal[False] = False,
    psm_override: int | None = None,
    enable_preproc: bool = True,
    ocr_oem: int = 1,
    ocr_dilate: int = 1,
    ocr_erode: int = 1,
    ocr_extra_config: str | None = None,
) -> List[str]: ...


@st.cache_data(show_spinner=False)
def cached_extract_pages(
    pdf_bytes: bytes,
    force_ocr: bool,
    *,
    assume_uniform: bool = False,
    return_meta: bool = False,
    psm_override: int | None = None,
    enable_preproc: bool = True,
    ocr_oem: int = 1,
    ocr_dilate: int = 1,
    ocr_erode: int = 1,
    ocr_extra_config: str | None = None,
) -> List[str] | Tuple[List[str], List[OCRPageMeta]]:
    """Extract text from PDF bytes, optionally forcing OCR with tuned settings."""
    buffer = io.BytesIO(pdf_bytes)
    return extract_text_from_pdf(
        buffer,
        use_ocr=force_ocr,
        return_meta=return_meta,
        ocr_assume_uniform_block=assume_uniform,
        ocr_psm=psm_override,
        ocr_oem=ocr_oem,
        ocr_extra_config=ocr_extra_config,
        ocr_dilate_iter=ocr_dilate,
        ocr_erode_iter=ocr_erode,
        enable_preproc=enable_preproc,
    )


@st.cache_resource(show_spinner=False)
def cached_init_index(index_name: str) -> None:
    """Ensure Pinecone index exists and is cached."""
    init_pinecone_index(index_name=index_name)


def is_sparse(pages: List[str]) -> bool:
    """Heuristic to detect sparse extraction results and trigger OCR."""
    if not pages:
        return True
    total_chars = sum(len(page.strip()) for page in pages)
    return total_chars < 100 * len(pages)


def ensure_environment(provider_choice: str) -> str:
    """Validate environment keys and confirm provider selection."""
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not pinecone_key:
        st.error("Missing PINECONE_API_KEY. Please add it to your environment.")
        st.stop()

    if provider_choice == "openai" and not openai_key:
        st.error("Missing OPENAI_API_KEY. Please add it to your environment.")
        st.stop()

    if provider_choice == "anthropic":
        if not anthropic_key:
            st.warning("ANTHROPIC_API_KEY not found. Falling back to OpenAI.")
            return "openai"
        try:
            __import__("anthropic")
        except Exception:
            st.warning("Anthropic client not installed. Falling back to OpenAI.")
            return "openai"

    return provider_choice


def segment_noise_score(segment: str, pages: list[str], metas: list[OCRPageMeta]) -> float:
    """
    Heuristic to estimate segment noise by averaging the 'noisy' flag of source pages.
    For now, this is a global approximation.
    """
    if not metas:
        return 0.0
    return sum(1 for m in metas if m.noisy) / max(1, len(metas))

def main() -> None:
    st.set_page_config(page_title="Notes RAG Alchemist", page_icon="📚", layout="wide")

    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {"refs": 0, "segments": 0, "matches": 0}
    if "namespace" not in st.session_state:
        st.session_state["namespace"] = "default"
    if "lecture_noisy_ratio" not in st.session_state:
        st.session_state["lecture_noisy_ratio"] = 0.0

    anthropic_present = bool(os.getenv("ANTHROPIC_API_KEY"))
    env_provider = (os.getenv("PLANNER_PROVIDER") or "").lower()
    provider_options = ["openai", "anthropic"]
    default_provider_index = 1 if anthropic_present else 0
    if env_provider in provider_options:
        default_provider_index = provider_options.index(env_provider)

    header_left, header_right = st.columns([3, 2])
    with header_left:
        st.title("📚 Notes RAG Alchemist")
        st.caption("Upload lecture notes and reference PDFs to generate a citation-rich study report.")
    metrics_cols = header_right.columns(3)
    metric_placeholders = [col.empty() for col in metrics_cols]
    metrics_snapshot = st.session_state.get("metrics", {"refs": 0, "segments": 0, "matches": 0})
    for placeholder, (label, key) in zip(
        metric_placeholders,
        [("Refs", "refs"), ("Segments", "segments"), ("Matches", "matches")],
    ):
        placeholder.metric(label, metrics_snapshot.get(key, 0))

    # --- Sidebar Configuration ---
    st.sidebar.markdown("## Data & Keys")
    planner_provider_choice = st.sidebar.selectbox(
        "Planner provider",
        provider_options,
        index=default_provider_index,
        help="Choose which LLM provider should guide the planner.",
    )
    index_prefix = st.sidebar.text_input("Index name prefix", value="lecture-notes-index")
    st.sidebar.caption(
        "Set API keys via environment variables: PINECONE_API_KEY, OPENAI_API_KEY, and ANTHROPIC_API_KEY."
    )

    st.sidebar.markdown("## OCR Settings")
    enhance_ocr = st.sidebar.checkbox("Enhance OCR (deskew & binarize)", value=True)
    assume_uniform = st.sidebar.checkbox("Assume single uniform block (PSM 6)", value=True)
    psm_choice = st.sidebar.selectbox(
        "PSM (override)",
        ["(auto)", "3 - Fully automatic", "4 - Single column", "6 - Uniform block"],
        index=0,
    )
    psm_map = {
        "(auto)": None,
        "3 - Fully automatic": 3,
        "4 - Single column": 4,
        "6 - Uniform block": 6,
    }
    psm_override = psm_map[psm_choice]
    morph_dilate = st.sidebar.slider("Morphological dilation (iters)", 0, 3, 1)
    morph_erode = st.sidebar.slider("Morphological erosion (iters)", 0, 3, 1)
    tesseract_extra = st.sidebar.text_input("Tesseract extra config", value="-l eng")

    st.sidebar.markdown("## Planner / Retriever Settings")
    st.sidebar.caption("Planner uses the selected provider above for LLM calls.")
    include_top_snippets = st.sidebar.checkbox("Include top snippets in prompt", value=True)
    top_snippet_limit = st.sidebar.slider("Max top snippets", min_value=1, max_value=3, value=3, step=1)
    noisy_hint = st.session_state.get("lecture_noisy_ratio", 0.0)
    default_min_score = 0.45 if noisy_hint >= 0.35 else 0.55
    min_score = st.sidebar.slider(
        "Retriever min similarity",
        0.0,
        1.0,
        value=default_min_score,
        step=0.01,
        help="Lower this for scanned/OCR-heavy content.",
    )
    retriever_top_k = st.sidebar.slider("Top-K per query", 1, 15, 7)
    st.session_state["retriever_min_score"] = min_score
    st.session_state["retriever_top_k"] = retriever_top_k
    max_steps = st.sidebar.slider("Planner max steps", 1, 10, 4, help="Max planner iterations per segment")
    max_refines = st.sidebar.slider("Planner max refines", 0, 5, 2, help="Max query refinements per segment")
    with st.sidebar.expander("Advanced", expanded=False):
        model_override = st.text_input("Planner model override (optional)", "")
        st.caption("If left blank, the default model for the selected provider is used.")

    st.sidebar.markdown("## Report Settings")
    max_excerpts_slider = st.sidebar.slider(
        "Max excerpts per segment",
        min_value=1, max_value=10, value=4, step=1,
        help="Controls how many matched excerpts are shown for each segment in the final PDF."
    )

    chosen_provider = ensure_environment(planner_provider_choice)

    left, right = st.columns([1.2, 1], gap="large")

    with right:
        status_panel = st.container()
        with status_panel:
            index_status = st.empty()
            index_progress_placeholder = st.empty()
            segment_status = st.empty()
            planner_status = st.empty()
            retriever_status = st.empty()
            compile_status = st.empty()

        indexed_refs_expander = st.expander("Indexed references", expanded=False)
        with indexed_refs_expander:
            indexed_details_placeholder = st.empty()

        planner_decisions_expander = st.expander("Planner decisions", expanded=False)
        with planner_decisions_expander:
            planner_details_placeholder = st.empty()

        prompt_expander = st.expander("Generated planner prompt (if dynamic)", expanded=False)
        with prompt_expander:
            prompt_placeholder = st.empty()

        retrieval_expander = st.expander("Retrieval excerpts sample", expanded=False)
        with retrieval_expander:
            retrieval_placeholder = st.empty()

        retriever_status.info(
            f"Retriever min similarity: {min_score:.2f} | Top-K per query: {retriever_top_k}"
        )

    with left:
        upload_expander = st.expander("Upload Panel", expanded=True)
        with upload_expander:
            notes_file = st.file_uploader(
                "Upload Lecture Notes PDF (single)",
                type=["pdf"],
                accept_multiple_files=False,
                help="Upload the primary lecture notes document.",
            )
            refs_files_input = st.file_uploader(
                "Upload Reference Books PDFs (up to 10)",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload supporting reference PDFs used for retrieval.",
            )
            refs_files_list = list(refs_files_input or [])
            if len(refs_files_list) > 10:
                st.warning("You selected more than 10 reference PDFs. Only the first 10 will be processed.")
                refs_files_list = refs_files_list[:10]

            file_rows = []
            if notes_file:
                size_bytes = getattr(notes_file, "size", None)
                if size_bytes is None:
                    size_bytes = len(notes_file.getvalue())
                file_rows.append(
                    {
                        "Type": "Lecture",
                        "Name": notes_file.name,
                        "Size (KB)": f"{size_bytes / 1024:.1f}",
                    }
                )
            for ref_file in refs_files_list:
                size_bytes = getattr(ref_file, "size", None)
                if size_bytes is None:
                    size_bytes = len(ref_file.getvalue())
                file_rows.append(
                    {
                        "Type": "Reference",
                        "Name": ref_file.name,
                        "Size (KB)": f"{size_bytes / 1024:.1f}",
                    }
                )

            if file_rows:
                st.dataframe(file_rows, width='stretch')
            else:
                st.caption("Selected files will appear here.")

        run_panel = st.container()
        with run_panel:
            st.markdown("### Run Panel")
            run_button = st.button("Generate Report", type="primary", use_container_width=True) # `width` not supported on st.button yet
            inputs_summary_placeholder = st.empty()

    result_container = st.container()

    if run_button:
        result_container.empty()
        index_status.empty()
        segment_status.empty()
        planner_status.empty()
        retriever_status.empty()
        compile_status.empty()
        indexed_details_placeholder.empty()
        planner_details_placeholder.empty()
        prompt_placeholder.empty()
        retrieval_placeholder.empty()

        inputs_summary_placeholder.markdown(
            f"**Lecture:** {notes_file.name if notes_file else 'missing'} | "
            f"**References:** {len(refs_files_list)} file(s)"
        )

        if not notes_file:
            index_status.error("Lecture notes PDF required before running.")
            st.error("Please upload a lecture notes PDF.")
            st.stop()

        if not refs_files_list:
            index_status.error("At least one reference PDF is required.")
            st.error("Please upload at least one reference PDF.")
            st.stop()

        try:
            ref_hashes = sorted([file_sha1(f) for f in refs_files_list])
            # Use a fixed index name and a dynamic namespace based on file content.
            index_prefix_clean = (index_prefix or "lecture-notes-index").strip() or "lecture-notes-index"
            index_name = index_prefix_clean
            combined_hash_str = "".join(ref_hashes)
            namespace = hashlib.sha1(combined_hash_str.encode("utf-8")).hexdigest()[:16]
            st.session_state["namespace"] = namespace

            index_status.info(f"Initializing vector store ({index_name})...")
            cached_init_index(index_name)
            index_status.success(f"Vector store ready: {index_name}")

            index_log_rows: List[dict[str, Any]] = []
            total_refs = len(refs_files_list)
            index_progress_placeholder.progress(0.0, text="Indexing references...")
            for i, ref_file in enumerate(refs_files_list):
                index_progress_placeholder.progress((i + 1) / total_refs, text=f"Indexing: {ref_file.name}")
                ref_bytes = ref_file.getvalue()
                pages_direct = cached_extract_pages(
                    ref_bytes,
                    force_ocr=False,
                    assume_uniform=assume_uniform,
                    psm_override=psm_override,
                    enable_preproc=enhance_ocr,
                    ocr_extra_config=tesseract_extra,
                    ocr_dilate=morph_dilate,
                    ocr_erode=morph_erode,
                )
                use_ocr = is_sparse(pages_direct)
                if use_ocr:
                    index_status.info(f"OCR fallback for {ref_file.name}")
                    pages = cached_extract_pages(
                        ref_bytes,
                        force_ocr=True,
                        assume_uniform=assume_uniform,
                        psm_override=psm_override,
                        enable_preproc=enhance_ocr,
                        ocr_extra_config=tesseract_extra,
                        ocr_dilate=morph_dilate,
                        ocr_erode=morph_erode,
                    )
                else:
                    pages = pages_direct

                total_chars = sum(len((page or "").strip()) for page in pages)
                if total_chars == 0:
                    st.warning(f"No text extracted from {ref_file.name}. Skipping indexing.")
                    continue

                docs: List[str] = []
                metas: List[dict[str, Any]] = []
                file_hash = file_sha1(ref_file)

                for page_no, page_text in enumerate(pages):
                    page_text = page_text or ""
                    if not page_text.strip():
                        continue
                    figure_ids = list(set(FIG_PATTERN.findall(page_text)))
                    chunks = split_text(page_text, max_length=1500, overlap=180, min_chunk_len=100)
                    for chunk in chunks:
                        if not chunk or not chunk.strip():
                            continue
                        docs.append(chunk)
                        metas.append(
                            {
                                "book": ref_file.name,
                                "page": page_no + 1,
                                "pages": [str(page_no + 1)],
                                "text": chunk,
                                "figure_nums": figure_ids,
                            }
                        )

                note_parts = []
                if use_ocr:
                    note_parts.append("OCR fallback")
                if not docs:
                    st.info(f"{ref_file.name}: 0 chunks indexed (OCR may have failed).")
                    note_parts.append("0 chunks")
                    note_parts.append("No text extracted")
                chunks_added = len(docs)
                index_log_rows.append(
                    {
                        "Reference": ref_file.name,
                        "Pages": len(pages),
                        "Chunks": chunks_added,
                        "OCR": "Yes" if use_ocr else "No",
                        "Note": " | ".join(note_parts),
                    }
                )

                if docs:
                    index_chunks(
                        docs,
                        metas,
                        index_name=index_name,
                        namespace=namespace,
                        id_prefix=file_hash,
                    )

            index_progress_placeholder.empty()
            indexed_details_placeholder.dataframe(index_log_rows, width='stretch')
            index_status.success("Reference books indexed successfully!")

            segment_status.info("Extracting & segmenting lecture notes...")
            lecture_bytes = notes_file.getvalue()
            # Unpack pages and metadata
            lecture_pages, lecture_meta = cached_extract_pages(
                lecture_bytes,
                force_ocr=True,
                return_meta=True,
                assume_uniform=assume_uniform,
                psm_override=psm_override,
                enable_preproc=enhance_ocr,
                ocr_extra_config=tesseract_extra,
                ocr_dilate=morph_dilate,
                ocr_erode=morph_erode,
            )

            # Compute a global noise ratio for the lecture notes
            lecture_noisy_ratio = segment_noise_score("", lecture_pages, lecture_meta)
            st.session_state["lecture_noisy_ratio"] = lecture_noisy_ratio
            noisy_bias = lecture_noisy_ratio >= 0.35
            retriever_status.info(
                f"Retriever min similarity: {min_score:.2f} | Top-K per query: {retriever_top_k} | Noise ratio: {lecture_noisy_ratio:.2f}"
            )

            lecture_text = "\n\n".join(lecture_pages)
            segments = split_text(lecture_text, max_length=480, overlap=40, min_chunk_len=200)
            segment_status.success(f"Prepared {len(segments)} segments.")

            prompt_path = ROOT_DIR / "prompts" / "planner_prompt.txt"
            if not prompt_path.is_file():
                planner_status.error(
                    f"Planner prompt file not found at '{prompt_path}'. Cannot continue without it."
                )
                st.error(f"FATAL: Planner prompt file not found at '{prompt_path}'. Cannot proceed.")
                st.stop()

            try:
                prompt_preview_text = prompt_path.read_text(encoding="utf-8")
                prompt_placeholder.code(prompt_preview_text[:1500], language="markdown")
            except Exception as prompt_exc:
                prompt_placeholder.warning(f"Unable to load planner prompt preview: {prompt_exc}")

            planner_cfg = PlannerConfig(
                index_name=index_name,
                top_k=5,
                llm_enabled=True,
                provider=chosen_provider,
                model=model_override or None,
                prompt_path=str(prompt_path),
                include_top_snippets=include_top_snippets,
                top_snippet_limit=top_snippet_limit,
                noise_ratio=lecture_noisy_ratio,
                noisy_bias=noisy_bias,
                max_steps=max_steps,
                max_refines=max_refines,
            )
            retriever = RetrieverWorker(
                index_name=index_name,
                namespace=namespace,
                top_k=retriever_top_k,
                min_score=min_score,
                dedup=True,
            )

            segment_plans: List[dict[str, Any]] = []
            planner_records: List[dict[str, Any]] = []
            first_planner_error: str | None = None
            error_count = 0
            total_segments = len(segments)

            planner_status.info(f"Planning {total_segments} lecture segment(s)...")
            for idx, segment in enumerate(segments, start=1):
                planner_status.info(f"Planning segment {idx}/{total_segments}...")
                try:
                    plan = plan_segment(segment, retriever, planner_cfg)
                    plan["noisy_bias"] = noisy_bias
                    segment_plans.append(plan)
                except Exception as exc:
                    error_count += 1
                    error_message = f"Failed to process Segment {idx}: {exc}"
                    if not first_planner_error:
                        first_planner_error = error_message
                    planner_status.warning(error_message)
                    print(f"ERROR: Segment {idx} failed:\n{traceback.format_exc()}", file=sys.stderr)
                    plan = {
                        "segment_text": segment,
                        "decision": "ERROR",
                        "trace": f"Failed: {exc}",
                        "best_score": None,
                        "excerpts": [],
                        "top_snippets": "",
                    }
                    plan["noisy_bias"] = noisy_bias
                    segment_plans.append(plan)
                best_score = plan.get("best_score")
                best_display = "" if best_score is None else f"{best_score:.2f}"
                trace_text = str(plan.get("trace") or "")
                planner_records.append(
                    {
                        "Segment": idx,
                        "Decision": plan.get("decision"),
                        "Trace": trace_text + (" (noisy)" if noisy_bias else ""),
                        "Best": best_display,
                    }
                )

            if first_planner_error:
                planner_status.error(
                    f"Encountered {error_count} error(s) during planning. First error: {first_planner_error}"
                )

            segment_plans = retriever.retrieve_for_segments(segment_plans, attempt_on_external=False, top_k=retriever_top_k)
            matches_count = sum(len(plan.get("excerpts", [])) for plan in segment_plans)
            planner_details_placeholder.dataframe(planner_records, width='stretch')
            planner_status.success("Planning & retrieval complete.")

            with retrieval_expander:
                if noisy_bias:
                    st.info("Lecture OCR looks noisy; planner is biased toward REFINE/EXTERNAL.")

                has_excerpts = any(plan.get("excerpts") for plan in segment_plans)
                if has_excerpts:
                    for i, plan in enumerate(segment_plans):
                        excerpts = plan.get("excerpts") or []
                        if not excerpts:
                            continue
                        
                        first_excerpt = excerpts[0]
                        excerpt_text = first_excerpt.get("text", "")
                        segment_text_preview = (plan.get("segment_text", "") or "").strip().split('\n')[0][:80]
                        
                        with st.expander(f"Segment {i+1}: {segment_text_preview}..."):
                            st.markdown(excerpt_text.replace('\n', '  \n'))
                else:
                    st.info("No retrieval matches to display.")

            compile_status.info("Compiling final PDF report...")
            pdf_bytes = compile_report_bytes(
                segment_plans,
                max_excerpts_per_segment=max_excerpts_slider
            )
            compile_status.success("Report compiled successfully.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            assert isinstance(pdf_bytes, bytes), f"Expected bytes for Streamlit download, got {type(pdf_bytes)}"

            st.session_state["metrics"] = {
                "refs": len(refs_files_list),
                "segments": len(segments),
                "matches": matches_count,
            }
            for placeholder, (label, key) in zip(
                metric_placeholders,
                [("Refs", "refs"), ("Segments", "segments"), ("Matches", "matches")],
            ):
                placeholder.metric(label, st.session_state["metrics"].get(key, 0))

            with result_container:
                st.success("Report generated successfully!")
                st.download_button(
                    "Download Report PDF",
                    data=pdf_bytes,
                    file_name=f"Notes_RAG_Alchemist_Report_{timestamp}.pdf",
                    mime="application/pdf",
                )

        except Exception as exc:
            index_progress_placeholder.empty()
            compile_status.error("Pipeline terminated with an error.")
            result_container.empty()
            with result_container:
                st.error("Something went wrong while generating the report.")
            st.exception(exc)
            st.stop()


if __name__ == "__main__":
    main()
