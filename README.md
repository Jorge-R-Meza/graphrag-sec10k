# GraphRAG 10-K

A fork of [microsoft/graphrag](https://github.com/microsoft/graphrag) (v3.0.9 base) tuned for
building knowledge graphs from SEC 10-K filings and querying them with local, global and
DRIFT search. It talks to language models through [LiteLLM](https://docs.litellm.ai/), so any
provider LiteLLM supports works; the sample project is configured for
[OpenRouter](https://openrouter.ai/).

Upstream documentation applies to everything not listed below:
[microsoft.github.io/graphrag](https://microsoft.github.io/graphrag).

## What this fork changes

Indexing pipeline:

- **Two-phase graph extraction.** Phase 1 extracts entities only across the corpus. Phase 2
  extracts relationships per chunk with the accumulated entity list injected as salience
  context, with optional relationship gleaning passes and inline classification of
  relationship endpoints that were never declared as entities. Configured under
  `extract_graph` (`two_phase`, `relationship_passes`, `relationship_gleanings`,
  `inline_endpoint_classification`, `variant`, `filer_alias_file`).
- **Scoped facts.** Entity descriptions are summarised per source document, labelled by a
  regex over the document title (for example `AAPL FY2025`), and optionally emitted as
  claims. Configured under `scoped_facts`.
- **Document-scoped communities.** Community detection runs per document group (for example
  per filer) rather than over the merged graph, and community reports can be stamped with
  their source document. Configured under `document_scoped_communities`.
- **Entity alias resolution.** An optional post-extraction workflow merges entities that
  are known aliases of one another, driven by a YAML mapping file. Configured under
  `resolve_entity_aliases`; off in the sample project. Separately, `filer_alias_file` under
  `extract_graph` maps each document title to its filer, so self-references such as "the
  Company" resolve to the actual filer during extraction.
- **Custom prompts** for entity-only extraction, gleaning, relationship extraction,
  description summarisation and community reports live in `ragtest/prompts/`.

Runtime:

- **Circuit breaker and `--deadline`.** Authentication failures or repeated connection
  errors trip a process-wide breaker that drains in-flight work and pauses the run instead
  of producing empty results. `graphrag index --deadline 50m` pauses gracefully after a time
  budget; re-running resumes from the saved progress.
- **Local embeddings.** A `sentence-transformers` embedding provider (default model
  `BAAI/bge-m3`) runs on CPU with no API calls, alongside the LiteLLM providers.
- **Retry classification** matches parent exception classes, so provider-specific
  subclasses of non-retriable errors are not retried.

Evaluation:

- `ragtest/q_autorun.py` runs a question suite through local and global search and writes a
  JSON plus markdown report, with resume support.

## Quickstart

Requirements: Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/). uv downloads a
matching interpreter if your system Python is outside that range.

```bash
uv sync --all-packages
cp ragtest/.env.example ragtest/.env   # then paste your OpenRouter key into it
uv run graphrag index --root ragtest
uv run graphrag query --root ragtest --method global --query "Which filers report supply concentration risk?"
```

Six 10-K filings (AAPL, AVGO, CRUS, QCOM, QRVO, SWKS) are included under `ragtest/input/`
as plain text. Indexing them end to end makes several thousand model calls; set
`rate_limit` and `concurrent_requests` in `ragtest/settings.yaml` to match your OpenRouter
account and the model you choose.

To use another provider, change `model_provider` and `model` on the completion model in
`ragtest/settings.yaml`. LiteLLM sends the request as `model_provider/model`.

See [ragtest/README.md](ragtest/README.md) for the sample project layout and tools.

## Repository layout

| Path | Contents |
| --- | --- |
| `packages/graphrag` | The GraphRAG library and CLI, including the pipeline changes above |
| `packages/graphrag-llm` | Model layer: LiteLLM completion and embedding providers, retry, rate limiting, cache, circuit breaker |
| `packages/graphrag-*` | Upstream support packages (storage, vectors, cache, chunking, input, common) |
| `ragtest/` | Sample 10-K project: settings, prompts, inputs, evaluation harness, helper tools |
| `scripts/` | PDF table extraction and MarkItDown export helpers used to prepare inputs |
| `docs/` | Upstream documentation site sources |
| `tests/` | Unit, integration and smoke tests from upstream plus tests for the new operations |

## Development

```bash
uv run poe check        # ruff format check, ruff lint, pyright
uv run pytest tests/unit
```

Some golden-file regression tests for clustering and community creation predate the
pipeline changes and currently fail; see the test output for the list.

## License and attribution

MIT, as upstream. Microsoft's copyright notice is retained in [LICENSE](LICENSE); changes in
this fork are copyright their respective authors under the same license.
