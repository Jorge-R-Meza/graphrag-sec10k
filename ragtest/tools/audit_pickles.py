import os, tempfile, pickle, pandas as pd

tus = pd.read_parquet('output/text_units.parquet')
docs = pd.read_parquet('output/documents.parquet')
doc_title = docs.set_index('id')['title'].to_dict()
tus['doc'] = tus['document_id'].map(doc_title)

doc_sets = {d: set(tus[tus['doc'].str.contains(d)]['id']) for d in ['AAPL', 'ADI', 'AVGO']}

ckpt_base = os.path.join(tempfile.gettempdir(), 'graphrag_p2', '1b91a848a308')
p1_files = [p for p in os.listdir(ckpt_base) if 'pass_-1' in p]

# Per doc: list of (chunk_id, n_entities)
per_doc = {d: [] for d in ['AAPL', 'ADI', 'AVGO', 'unknown']}
for p in p1_files:
    with open(os.path.join(ckpt_base, p), 'rb') as f:
        cid, result = pickle.load(f)
    n_ents = len(result) if isinstance(result, pd.DataFrame) else 0
    doc = next((d for d, ids in doc_sets.items() if cid in ids), 'unknown')
    per_doc[doc].append((cid, n_ents))

print('=== Phase-1 pickle entity coverage ===')
for doc, items in per_doc.items():
    if not items:
        continue
    empty = [(c, n) for c, n in items if n == 0]
    non_empty = [(c, n) for c, n in items if n > 0]
    avg = sum(n for _, n in non_empty) / len(non_empty) if non_empty else 0
    print(f'{doc}: {len(non_empty)}/{len(items)} non-empty  ({len(empty)} empty)  avg_ents_when_non_empty={avg:.1f}')

print()
# Phase-2 pass 0 and pass 1
for pass_idx, pass_name in [(0, 'pass-0 (phase2 pass1)'), (1, 'pass-1 (phase2 pass2)')]:
    prefix = f'pass_{pass_idx}_'
    p2_files = [p for p in os.listdir(ckpt_base) if p.startswith(prefix)]
    per_doc2 = {d: [] for d in ['AAPL', 'ADI', 'AVGO', 'unknown']}
    for p in p2_files:
        with open(os.path.join(ckpt_base, p), 'rb') as f:
            cid, result = pickle.load(f)
        n = len(result) if isinstance(result, pd.DataFrame) else 0
        doc = next((d for d, ids in doc_sets.items() if cid in ids), 'unknown')
        per_doc2[doc].append((cid, n))
    print(f'=== {pass_name} ===')
    for doc, items in per_doc2.items():
        if not items:
            continue
        empty = sum(1 for _, n in items if n == 0)
        non_empty = [(c, n) for c, n in items if n > 0]
        avg = sum(n for _, n in non_empty) / len(non_empty) if non_empty else 0
        print(f'  {doc}: {len(non_empty)}/{len(items)} non-empty  ({empty} empty)  avg_rels_when_non_empty={avg:.1f}')
    print()
