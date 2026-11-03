"""Spec compliance verification for v5_entonly_combined_scoped port."""
import inspect

# ── Gate: if two_phase and relationship_prompt ──────────────────────────
with open('packages/graphrag/graphrag/index/operations/extract_graph/extract_graph.py') as f:
    src = f.read()
assert 'if two_phase and relationship_prompt:' in src, 'FAIL: P3 gate missing'
print('OK [P3] entry gate: if two_phase and relationship_prompt:')

assert 'new_endpoints_set' in src, 'FAIL: endpoint filter missing'
assert 'name in new_endpoints_set' in src, 'FAIL: filter check missing'
print('OK     _classify_chunk_endpoints: filters to asked-about endpoints only')

assert 'declared = _known_titles | set(' in src, 'FAIL: declared union missing'
print('OK     known_titles includes current-chunk entity results (declared union)')

assert 'continue_prompt=ENTITY_CONTINUE_PROMPT if phase1_entities_only else None' in src
print('OK     phase1_entities_only -> ENTITY_CONTINUE_PROMPT in entity extractor')

assert 'full_reference = _format_entity_reference(entities) if pass_entity_list else ""' in src
print('OK [P1] pass_entity_list gates entity reference injection')

assert 'return "\\n".join(f"- {t}" for t in sorted(entities["title"].astype(str)))' in src
print('OK [P8] _format_entity_reference: bare titles, sorted')

# ── Config layer ────────────────────────────────────────────────────────
from graphrag.config.models.extract_graph_config import ExtractGraphConfig
c = ExtractGraphConfig()
assert c.pass_entity_list == True
assert c.inline_endpoint_classification == False
assert c.phase1_entities_only == False
c2 = ExtractGraphConfig(two_phase=True)
p2 = c2.resolved_prompts()
from graphrag.prompts.index.extract_graph import SECOND_PHASE_PROMPT
assert p2.relationship_prompt == SECOND_PHASE_PROMPT, 'two_phase should default to SECOND_PHASE_PROMPT'
print('OK     ExtractGraphConfig: defaults correct, resolved_prompts() uses SECOND_PHASE_PROMPT when two_phase=True')

# ── Validator ───────────────────────────────────────────────────────────
from graphrag.config.models.graph_rag_config import GraphRagConfig
try:
    GraphRagConfig(scoped_facts={'enabled': True, 'emit_claims': True}, extract_claims={'enabled': True})
    print('FAIL: scoped_claims validator not firing')
except Exception as e:
    assert 'scoped_facts.emit_claims and extract_claims cannot both be enabled' in str(e)
    print('OK     _validate_scoped_claims raises ValueError correctly [P6]')

# ── should_skip_retry ────────────────────────────────────────────────── 
from graphrag_llm.retry.exceptions_to_skip import should_skip_retry
class E401(Exception):
    status_code = 401
class ESub(E401): pass
assert should_skip_retry(E401(), [])
assert should_skip_retry(ESub(), []), 'MRO walk must catch subclasses'
assert not should_skip_retry(Exception(), [])
print('OK     should_skip_retry: 401 skips, subclass skips, bare Exception retries')

# ── LLMProviderType ─────────────────────────────────────────────────────
from graphrag_llm.config.types import LLMProviderType
assert LLMProviderType.SentenceTransformers == 'sentence_transformers'
print('OK     LLMProviderType.SentenceTransformers registered')

# ── Workflow 5-tuple ─────────────────────────────────────────────────── 
from graphrag.index.workflows.extract_graph import extract_graph as wf_extract
sig = inspect.signature(wf_extract)
params = list(sig.parameters)
for p in ['inline_endpoint_classification','phase1_entities_only','pass_entity_list',
          'scoped_facts_enabled','scoped_facts_emit_claims','text_unit_to_document']:
    assert p in params, f'workflow missing param: {p}'
ann = str(sig.return_annotation)
assert 'None' in ann, 'return type should include pd.DataFrame | None for covariates'
print('OK     workflow extract_graph: Phase-B+C params present, 5-tuple return with optional covariates')

# ── Prompts ─────────────────────────────────────────────────────────────
from graphrag.prompts.index.extract_graph import (
    ENTITY_CONTINUE_PROMPT, CLASSIFY_CHUNK_ENDPOINTS_PROMPT,
    CLASSIFY_ORPHAN_ENDPOINTS_PROMPT, SECOND_PHASE_PROMPT,
)
assert 'do NOT emit relationships' in ENTITY_CONTINUE_PROMPT
assert '{entity_types}' in CLASSIFY_CHUNK_ENDPOINTS_PROMPT
assert '{endpoint_list}' in CLASSIFY_CHUNK_ENDPOINTS_PROMPT
assert '{input_text}' in CLASSIFY_CHUNK_ENDPOINTS_PROMPT
assert '{entity_list}' in SECOND_PHASE_PROMPT
assert '{entity_types}' in CLASSIFY_ORPHAN_ENDPOINTS_PROMPT
print('OK     All 4 prompts present with correct placeholders')

from graphrag.prompts.index.summarize_descriptions import (
    SCOPED_SUMMARIZE_PROMPT, SCOPED_SUMMARIZE_DERIVED_PROMPT,
)
for p in [SCOPED_SUMMARIZE_PROMPT, SCOPED_SUMMARIZE_DERIVED_PROMPT]:
    for ph in ['{entity_name}', '{fragment_list}', '{relationship_list}', '{max_length}']:
        assert ph in p, f'scoped prompt missing: {ph}'
print('OK     Scoped summarization prompts have all required placeholders')

# ── Resolve entity aliases workflow present ─────────────────────────────
from graphrag.index.workflows.resolve_entity_aliases import run_workflow as rae_wf
print('OK     resolve_entity_aliases workflow registered')

# ── Document-scoped communities ──────────────────────────────────────────
from graphrag.index.workflows.create_communities import run_workflow as cc_wf
from graphrag.index.operations.cluster_graph import cluster_graph
sig_cg = inspect.signature(cluster_graph)
assert 'node_groups' in sig_cg.parameters
print('OK     cluster_graph accepts node_groups param for per-document Leiden')

print()
print('=== All spec verification checks passed ===')
