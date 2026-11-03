# ragtest: sample 10-K project

A ready-to-run GraphRAG project over six SEC 10-K filings.

## Layout

| Path | Purpose |
| --- | --- |
| `settings.yaml` | Pipeline configuration: OpenRouter completion model, local embeddings, two-phase extraction, scoped facts, document-scoped communities |
| `.env.example` | Template for `.env`, which holds `OPENROUTER_API_KEY` and is gitignored |
| `input/` | Plain-text 10-K filings named `<TICKER>_10k_<FY>.txt`; the name drives document grouping and scope labels |
| `prompts/` | Custom prompts for extraction, gleaning, summarisation, community reports and search |
| `filer_aliases.json` | Known entity aliases merged during indexing |
| `questions.txt` | Default question suite for the evaluation harness, one question per line |
| `q_autorun.py` | Evaluation harness: runs each question through local and global search |
| `run_graphrag.py` | CLI wrapper that loads `.env`, injects the OS trust store, and registers the fastembed provider |
| `extensions/` | Optional fastembed embedding provider (`type: local_embeddings`) |
| `tools/` | One-off inspection and visualisation scripts, see below |

`output/`, `cache/`, `logs/` and `responses/` are created by runs and ignored by git.

## Running

```bash
uv run graphrag index --root ragtest
uv run graphrag index --root ragtest --deadline 45m     # pause after 45 minutes, resume on re-run
uv run graphrag query --root ragtest --method local --query "..."
uv run python ragtest/q_autorun.py                       # whole suite, local + global
uv run python ragtest/q_autorun.py --suite my_questions.txt --output-dir ./runs
```

`run_graphrag.py` accepts the same arguments as the `graphrag` CLI and is only needed on
networks that intercept TLS or when using the fastembed provider.

## Tools

Run these from inside `ragtest/` after an index has been built; they read from `output/`.

| Script | What it does |
| --- | --- |
| `tools/edge_claims.py` | Explodes `raw_relationships.parquet` into per-edge claims (`edge_claims.parquet`); needs `snapshots.raw_graph: true` |
| `tools/enrich_graph.py` | Adds entity and relationship attributes to `graph.graphml` for Gephi |
| `tools/regenerate_graphml.py` | Rebuilds `graph.graphml` from `relationships.parquet` |
| `tools/visualize_graph.py` | Renders an interactive pyvis HTML view of the graph |
| `tools/token_count.py` | Token statistics per document and text unit |
| `tools/audit_chunks.py`, `tools/audit_pickles.py`, `tools/audit_pickles2.py` | Cross-checks between text units, documents and entities |
