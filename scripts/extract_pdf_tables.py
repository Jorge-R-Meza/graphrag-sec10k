"""Export a PDF to Markdown with proper table extraction.

Uses pdfplumber for layout-aware PDF parsing:
  - Tables are extracted as proper | col | col | Markdown tables
  - Headings are detected from font size and marked with #/##/###
  - Text and tables are interleaved in their original document order

This produces significantly better output than MarkItDown for PDFs that
contain structured tables, while matching MarkItDown's prose quality.

Install once:
    uv add pdfplumber

Usage (run from the repo root):
    uv run python scripts/extract_pdf_tables.py <input.pdf> [output.md | -]

Examples:
    uv run python scripts/extract_pdf_tables.py "ragtest/input/Methodology For Rating U.S. Governments.pdf"
    uv run python scripts/extract_pdf_tables.py report.pdf report_tables.md
    uv run python scripts/extract_pdf_tables.py report.pdf -   # print to stdout
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not installed.  Run:  uv add pdfplumber", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body_size(words: list[dict]) -> float:
    """Return the most common font size (proxy for body text) from a word list."""
    sizes = [round(w.get("size", 12.0), 1) for w in words if w.get("text", "").strip()]
    if not sizes:
        return 12.0
    return Counter(sizes).most_common(1)[0][0]


def _words_to_markdown(words: list[dict], body_sz: float) -> str:
    """Convert pdfplumber word dicts to markdown text, detecting headings by font size."""
    if not words:
        return ""

    # Group into lines by rounded top-of-word coordinate
    line_map: dict[int, list[dict]] = {}
    for w in words:
        key = round(w["top"])
        line_map.setdefault(key, []).append(w)

    result_lines: list[str] = []
    prev_bottom: float = -1.0

    for y in sorted(line_map):
        line_words = sorted(line_map[y], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        if not text:
            continue

        # Insert a blank line on large vertical gaps (paragraph break heuristic)
        if prev_bottom >= 0:
            gap = y - prev_bottom
            if gap > body_sz * 1.2 and result_lines and result_lines[-1]:
                result_lines.append("")

        prev_bottom = max(w.get("bottom", y) for w in line_words)

        # Heading detection by font-size ratio vs body size
        sizes = [w.get("size", body_sz) for w in line_words]
        avg_sz = sum(sizes) / len(sizes)
        ratio = avg_sz / body_sz if body_sz > 0 else 1.0

        if ratio >= 1.5:
            text = f"# {text}"
        elif ratio >= 1.3:
            text = f"## {text}"
        elif ratio >= 1.15:
            text = f"### {text}"

        result_lines.append(text)

    return "\n".join(result_lines)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber 2-D list to a Markdown | col | table."""
    if not table:
        return ""

    # Sanitise cells: replace None, collapse whitespace
    clean: list[list[str]] = [
        [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
        for row in table
    ]
    # Drop entirely empty rows
    clean = [r for r in clean if any(c for c in r)]
    if not clean:
        return ""

    # Pad ragged rows to uniform width
    n = max(len(r) for r in clean)
    clean = [r + [""] * (n - len(r)) for r in clean]

    header, *body = clean
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * n) + " |",
        *("| " + " | ".join(r) + " |" for r in body),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

# Table detection settings — tweak these to improve accuracy:
# - "lines" uses PDF drawn lines/rectangles to detect table borders (good for bordered tables)
# - "text" uses word alignment to detect columns (good for borderless tables)
# Mixing strategies (e.g. horizontal="lines", vertical="text") works well for
# tables that have horizontal rules but no vertical lines.
_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
    "intersection_tolerance": 3,
}

# Minimum number of columns for a detection to be treated as a real table.
# Single-column detections are almost always sidebar text boxes, not tables.
_MIN_TABLE_COLS = 2


def _extract_page(page) -> str:
    """Return the full markdown string for one pdfplumber page."""
    table_objs = [
        t for t in page.find_tables(table_settings=_TABLE_SETTINGS)
        if t.extract() and max((len(r) for r in t.extract() if r), default=0) >= _MIN_TABLE_COLS
    ]

    if not table_objs:
        # Fast path: no tables — extract words and format with heading detection
        words = page.extract_words(extra_attrs=["size"])
        return _words_to_markdown(words, _body_size(words))

    # Sort tables top-to-bottom (pdfplumber bbox = (x0, top, x1, bottom))
    table_objs_sorted = sorted(table_objs, key=lambda t: t.bbox[1])

    segments: list[str] = []
    current_top = 0.0

    for tbl in table_objs_sorted:
        t_top, t_bottom = tbl.bbox[1], tbl.bbox[3]

        # Prose above this table
        if current_top < t_top - 1:
            crop = page.crop((0, current_top, page.width, t_top))
            words = crop.extract_words(extra_attrs=["size"])
            prose = _words_to_markdown(words, _body_size(words))
            if prose.strip():
                segments.append(prose.strip())

        # The table itself
        md_table = _table_to_markdown(tbl.extract())
        if md_table:
            segments.append(md_table)

        current_top = t_bottom

    # Prose below the last table
    if current_top < page.height - 1:
        crop = page.crop((0, current_top, page.width, page.height))
        words = crop.extract_words(extra_attrs=["size"])
        prose = _words_to_markdown(words, _body_size(words))
        if prose.strip():
            segments.append(prose.strip())

    return "\n\n".join(segments)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(src: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(src) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            print(f"\r  page {i}/{total}", end="", flush=True, file=sys.stderr)
            text = _extract_page(page)
            if text.strip():
                pages.append(text)
    print(file=sys.stderr)
    return "\n\n---\n\n".join(pages)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    print(f"Extracting: {src.name}", file=sys.stderr)
    markdown = convert(src)

    dest_arg = sys.argv[2] if len(sys.argv) >= 3 else str(src.with_suffix(".md"))

    if dest_arg == "-":
        print(markdown)
    else:
        dest = Path(dest_arg)
        dest.write_text(markdown, encoding="utf-8")
        print(
            f"Written: {dest}  ({dest.stat().st_size:,} bytes, {len(markdown):,} chars)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
