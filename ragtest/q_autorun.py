"""Run a question suite in both local and global search modes.

Reads questions from a text file (default: ragtest/questions.txt), runs each one
through GraphRAG local and global search, and writes a JSON results file plus a
timestamped markdown report.

    uv run python ragtest/q_autorun.py [--suite FILE] [--output-dir DIR] [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Workspace paths.
RAGTEST_DIR = Path(__file__).resolve().parent
ROOT_DIR = RAGTEST_DIR
# Defaults; both can be overridden on the command line.
SUITE_MD = RAGTEST_DIR / "questions.txt"
OUTPUT_DIR = RAGTEST_DIR / "responses"

# Output files — timestamped MD in OUTPUT_DIR, JSON kept in ragtest.
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_JSON = RAGTEST_DIR / "q_autorun_output.json"
# Incremental writes go to local disk first; the final MD is copied to OUTPUT_DIR.
RESULTS_MD = RAGTEST_DIR / f"responses_{_RUN_TS}.md"

METHODS = ["local", "global"]
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"

def _safe_write(path: Path, text: str) -> None:
    """Write to a local path safely using a .tmp swap. Path must be on local disk."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)



def parse_suite(md_path: Path) -> list[dict]:
    """Parse a question file.

    Accepts numbered lines ('1. Question text') or plain one-question-per-line
    text; blank lines and lines starting with '#' are ignored.
    """
    text = md_path.read_text(encoding="utf-8")
    parsed: list[dict] = []
    next_num = 1

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            qnum, question = int(m.group(1)), m.group(2).strip()
        else:
            qnum, question = next_num, line
        parsed.append({"qnum": qnum, "question": question})
        next_num = qnum + 1

    parsed.sort(key=lambda row: row["qnum"])
    return parsed


def summarize_context(context_data: dict) -> dict:
    """Return a compact JSON-safe context summary."""
    summary: dict = {}
    if not isinstance(context_data, dict):
        return summary

    for key, value in context_data.items():
        if isinstance(value, pd.DataFrame):
            entry: dict = {"rows": int(len(value))}
            for col in ("title", "name", "short_id", "id"):
                if col in value.columns and len(value):
                    entry["sample"] = [str(x) for x in value[col].head(8).tolist()]
                    break
            summary[key] = entry
        elif isinstance(value, list):
            summary[key] = {"rows": len(value)}

    return summary


async def run_one(method: str, query: str, tables: dict, config) -> tuple[str, dict]:
    """Execute one GraphRAG query by method."""
    from graphrag import api

    if method == "global":
        return await api.global_search(
            config=config,
            entities=tables["entities"],
            communities=tables["communities"],
            community_reports=tables["community_reports"],
            community_level=COMMUNITY_LEVEL,
            dynamic_community_selection=True,
            response_type=RESPONSE_TYPE,
            query=query,
        )

    if method == "local":
        return await api.local_search(
            config=config,
            entities=tables["entities"],
            communities=tables["communities"],
            community_reports=tables["community_reports"],
            text_units=tables["text_units"],
            relationships=tables["relationships"],
            covariates=tables.get("covariates"),
            community_level=COMMUNITY_LEVEL,
            response_type=RESPONSE_TYPE,
            query=query,
        )

    raise ValueError(f"Unknown method: {method}")


def render_markdown(results: list[dict], suite: list[dict]) -> str:
    """Render a readable markdown report with question then local and global responses."""
    by_key = {(r["qnum"], r["method"]): r for r in results}

    lines = [
        "# GraphRAG Autorun Results",
        "",
        f"Suite: {SUITE_MD}",
        f"Community level: {COMMUNITY_LEVEL}",
        f"Response type: {RESPONSE_TYPE}",
        "",
    ]

    for q in suite:
        lines.extend(
            [
                "---",
                "",
                f"## Q{q['qnum']}",
                "",
                f"**Question:** {q['question']}",
                "",
            ]
        )

        for method in METHODS:
            r = by_key.get((q["qnum"], method))
            lines.extend([f"### {method.capitalize()} Response", ""])
            if r is None:
                lines.extend(["_not run_", ""])
                continue
            if not r["ok"]:
                lines.extend([f"[ERROR] {r['error']}", ""])
                continue
            lines.extend([r["response"], ""])

    return "\n".join(lines)


async def main() -> None:
    global SUITE_MD
    parser = argparse.ArgumentParser(description="Run GraphRAG question suite")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing results from RESULTS_JSON and only rerun empty/failed responses.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=SUITE_MD,
        help="Question file: numbered or one-per-line (default: ragtest/questions.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the timestamped markdown report (default: ragtest/responses).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output MD file path (default: timestamped file in --output-dir).",
    )
    args = parser.parse_args()

    SUITE_MD = args.suite
    output_md = args.output if args.output else args.output_dir / f"responses_{_RUN_TS}.md"

    if not SUITE_MD.exists():
        raise FileNotFoundError(f"Question file not found: {SUITE_MD}")

    output_md.parent.mkdir(parents=True, exist_ok=True)

    suite = parse_suite(SUITE_MD)
    if not suite:
        raise ValueError(f"No questions parsed from: {SUITE_MD}")

    # Canned "I do not know" response GraphRAG emits when all map scores are 0.
    _CANNED = "I am sorry but I am unable to answer this question given the provided data."

    # Resume mode: load existing results, skip already-filled entries.
    existing: dict[tuple[int, str], dict] = {}
    if args.resume and RESULTS_JSON.exists():
        prior = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        for r in prior.get("results", []):
            resp = r.get("response", "").strip()
            if resp and resp != _CANNED:
                existing[(r["qnum"], r["method"])] = r
        print(f"Resume: {len(existing)} existing responses loaded, skipping those.")

    from graphrag.cli.query import _resolve_output_files
    from graphrag.config.load_config import load_config

    config = load_config(root_dir=ROOT_DIR)
    data_frames = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
        optional_list=["covariates"],
    )

    tables = {
        "communities": data_frames.get("communities"),
        "community_reports": data_frames.get("community_reports"),
        "text_units": data_frames.get("text_units"),
        "relationships": data_frames.get("relationships"),
        "entities": data_frames.get("entities"),
        "covariates": data_frames.get("covariates"),
    }

    # Start results with already-filled entries from a prior run.
    results: list[dict] = list(existing.values())
    sem_local = asyncio.Semaphore(9)
    sem_global = asyncio.Semaphore(1)
    lock = asyncio.Lock()

    total = len(suite) * len(METHODS)

    async def run_task(q: dict, method: str) -> None:
        sem = sem_global if method == "global" else sem_local
        async with sem:
            start = time.time()
            try:
                response, context = await run_one(method, q["question"], tables, config)
                ok = True
                error = None
            except Exception as exc:  # noqa: BLE001
                response = ""
                context = {}
                ok = False
                error = f"{type(exc).__name__}: {exc}"

            result = {
                "qnum": q["qnum"],
                "method": method,
                "question": q["question"],
                "response": response if isinstance(response, str) else str(response),
                "context": summarize_context(context),
                "ok": ok,
                "error": error,
                "seconds": round(time.time() - start, 1),
            }

            async with lock:
                results.append(result)
                md_text = render_markdown(results, suite)
                _safe_write(
                    RESULTS_JSON,
                    json.dumps(
                        {
                            "suite": str(SUITE_MD),
                            "methods": METHODS,
                            "community_level": COMMUNITY_LEVEL,
                            "response_type": RESPONSE_TYPE,
                            "results": results,
                        },
                        indent=2,
                    ),
                )
                # Write MD incrementally to a local staging file; copied to output_md at the end.
                _safe_write(RESULTS_MD, md_text)
                print(f"  [{len(results)}/{total}] Q{q['qnum']} {method} ({result['seconds']}s)")

    def pending(method: str) -> list:
        return [
            run_task(q, method)
            for q in suite
            if (q["qnum"], method) not in existing
        ]

    local_tasks = pending("local")
    global_tasks = pending("global")
    skipped = len(existing)
    print(f"Running {len(local_tasks)} local then {len(global_tasks)} global ({skipped} skipped from prior run)")

    print("--- Phase 1: local (9 concurrent) ---")
    await asyncio.gather(*local_tasks)

    print("--- Phase 2: global (1 concurrent) ---")
    await asyncio.gather(*global_tasks)

    # Copy final MD to destination once, now that all incremental writes are done.
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(RESULTS_MD.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Completed {len(results)} total responses")
    print(f"Questions source: {SUITE_MD}")
    print(f"Results JSON: {RESULTS_JSON}")
    print(f"Results MD (local):   {RESULTS_MD}")
    print(f"Results MD (copy):    {output_md}")


if __name__ == "__main__":
    asyncio.run(main())
