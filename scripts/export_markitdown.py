"""Export a PDF (or any MarkItDown-supported file) to Markdown.

Replicates exactly what GraphRAG does in-memory via MarkItDownFileReader,
so the output is byte-for-byte identical to what gets chunked and sent to
the LLM during indexing.

Usage (run from the repo root):
    uv run python scripts/export_markitdown.py <input_file> [output_file]

Examples:
    uv run python scripts/export_markitdown.py ragtest/input/report.pdf
    uv run python scripts/export_markitdown.py ragtest/input/report.pdf ragtest/input/report.md
    uv run python scripts/export_markitdown.py ragtest/input/report.pdf -   # print to stdout
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from markitdown import MarkItDown, StreamInfo


def convert(src: Path) -> str:
    md = MarkItDown()
    result = md.convert_stream(
        BytesIO(src.read_bytes()),
        stream_info=StreamInfo(extension=src.suffix),
    )
    return result.markdown


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        return 1

    markdown = convert(src)

    # Determine output destination
    if len(sys.argv) >= 3:
        dest_arg = sys.argv[2]
    else:
        dest_arg = str(src.with_suffix(".md"))

    if dest_arg == "-":
        print(markdown)
    else:
        dest = Path(dest_arg)
        dest.write_text(markdown, encoding="utf-8")
        print(f"Written: {dest}  ({dest.stat().st_size:,} bytes, {len(markdown):,} chars)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
