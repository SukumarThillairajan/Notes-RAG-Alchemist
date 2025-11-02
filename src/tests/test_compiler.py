import re
from fpdf import FPDF, XPos, YPos

# Import the functions and patterns we need to test from compiler.py
# This assumes 'compiler.py' is in the same directory or on the PYTHONPATH
try:
    from workers.compiler import (
        compile_report, 
        format_citation, 
        FIG_PATTERN, 
        _gather_fig_refs,
        _figure_pointer_line
    )
except ImportError:
    print("Error: Could not import from 'compiler.py'.")
    print("Please make sure 'compiler.py' is in the same directory.")
    exit()

def test_hello_world():
    """
    Test 1: Generate a minimal 'Hello World' PDF to ensure FPDF is working.
    """
    print("--- Running Test 1: Hello World PDF ---")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", size=16)
        pdf.cell(0, 10, "Hello World! FPDF is working.", ln=1, align="C")
        
        output_filename = "compiler_test_01_hello.pdf"
        pdf.output(output_filename)
        
        print(f"✅ Success: Created '{output_filename}'")
    except Exception as e:
        print(f"❌ FAILED: 'Hello World' test failed: {e}")
    print("-" * 40)


def test_figure_regex():
    """
    Test 2: Test the Figure Regex (FIG_PATTERN) from compiler.py
    """
    print("--- Running Test 2: Figure Regex ---")
    
    test_strings = [
        "This is a test of Figure 1.1.",
        "See fig. 2.3(a) for details.",
        "What about Fig 4?",
        "This is not a figure, but a fig tree.",
        "See Table 1.1, not Figure1.1 (no space)."
    ]
    
    # This set will store the results, just like in the real compiler
    # Format: (figure_id, book, pages_tuple, page_int)
    figure_refs: set[tuple[str, str, tuple[int, ...], int | None]] = set()

    print("Testing regex pattern on sample text...")
    for text in test_strings:
        # We call _gather_fig_refs to simulate the real logic
        # We pass dummy book/page info
        _gather_fig_refs(
            text=text,
            book="Sample Book",
            page=1,
            pages=[1],
            store=figure_refs
        )

    if not figure_refs:
        print("❌ FAILED: Regex did not capture any figures.")
        return

    print("✅ Success: Regex captured the following:")
    for ref in sorted(figure_refs):
        # ref[0] is the figure_id
        print(f"  - Captured ID: '{ref[0]}'") 
        
    print("\nTesting appendix formatting for these refs:")
    for entry in figure_refs:
        # Test the line formatting
        line = _figure_pointer_line(entry[0], entry[1], entry[3], entry[2])
        print(f"  - Formatted line: '{line}'")
        
    print("-" * 40)


def test_full_report():
    """
    Test 3: Generate a full report using the sample data.
    This tests text wrapping, multi-page, citations, and appendix.
    """
    print("--- Running Test 3: Full Report Generation ---")
    
    # This is the same sample data from the bottom of compiler.py
    sample_plans = [
        {
            "segment_text": "Explains core transformer attention flow. See Fig. 3.2(a) for heads.",
            "decision": "NORMAL",
            "trace": "Normal search",
            "best_score": 0.88,
            "excerpts": [
                {
                    "text": "Attention mechanism overview referencing Fig. 3.2(a)." * 20, # Long text to test wrapping
                    "book": "Deep Learning 101",
                    "page": "10",
                    "pages": ["10", "11", "NaN"], # Test bad page data
                    "score": 0.92,
                },
                {
                    "text": "Detailed derivation with Fig. 3.2(b) illustrating alignment.",
                    "book": "Deep Learning 101",
                    "page": 12,
                    "pages": ["12"],
                    "score": 0.85,
                },
                {
                    "text": "Layer normalization complements the attention block. This is a very long excerpt designed to test the multi_cell function in FPDF and ensure that text wraps correctly to the next line instead of overflowing off the page. We will repeat this sentence a few times to make sure it is long enough. This is a very long excerpt designed to test the multi_cell function in FPDF.",
                    "book": "Deep Learning 101",
                    "page": 13,
                    "pages": [13],
                    "score": 0.83,
                },
                {
                    "text": "Residual connections shown near Fig. 3.4.",
                    "book": "Deep Learning 101",
                    "page": "14",
                    "pages": ["14"],
                    "score": 0.80,
                },
            ],
        },
        {
            "segment_text": "Further reading recommended from foundational transformer papers.",
            "decision": "EXTERNAL",
            "suggested_book": "Attention Is All You Need",
            "trace": "External book suggested",
            "best_score": None,
            "excerpts": [], # Test "no excerpt found" logic
        },
    ]

    try:
        output_filename = "compiler_test_02_full_report.pdf"
        compile_report(
            sample_plans, 
            output_filename, 
            max_excerpts_per_segment=3
        )
        print(f"✅ Success: Created '{output_filename}'")
        print("\n--- PLEASE MANUALLY CHECK THE PDFs ---")
        print(f"1. Open '{output_filename}'")
        print("2. Check for text wrapping in the long excerpts.")
        print("3. Check the 'Figure Pointers' appendix for correct formatting.")
        print("4. Check the 'Planner Decisions' table.")
        
    except Exception as e:
        print(f"❌ FAILED: Full report test failed: {e}")
    print("-" * 40)


if __name__ == "__main__":
    test_hello_world()
    test_figure_regex()
    test_full_report()
