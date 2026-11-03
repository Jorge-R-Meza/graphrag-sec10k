import os, tempfile, pickle, pandas as pd

tus = pd.read_parquet('output/text_units.parquet')
docs = pd.read_parquet('output/documents.parquet')
doc_title = docs.set_index('id')['title'].to_dict()
tus['doc'] = tus['document_id'].map(doc_title)
tus_by_id = tus.set_index('id')

doc_sets = {d: set(tus[tus['doc'].str.contains(d)]['id']) for d in ['AAPL', 'ADI', 'AVGO']}

ckpt_base = os.path.join(tempfile.gettempdir(), 'graphrag_p2', '1b91a848a308')

print('=== Phase-1 entity coverage ===')
p1_files = [p for p in os.listdir(ckpt_base) if p.startswith('pass_-1_')]
per_doc_p1 = {d: [] for d in ['AAPL', 'ADI', 'AVGO', 'unknown']}
for p in p1_files:
    with open(os.path.join(ckpt_base, p), 'rb') as f:
        cid, result = pickle.load(f)
    n = len(result) if isinstance(result, pd.DataFrame) else 0
    doc = next((d for d, ids in doc_sets.items() if cid in ids), 'unknown')
    per_doc_p1[doc].append((cid, n))

for doc, items in per_doc_p1.items():
    if not items:
        continue
    empty = [(c, n) for c, n in items if n == 0]
    non_empty = [(c, n) for c, n in items if n > 0]
    avg = sum(n for _, n in non_empty) / len(non_empty) if non_empty else 0
    print(f'{doc}: {len(non_empty)}/{len(items)} non-empty  ({len(empty)} empty)  avg_ents={avg:.1f}')

print()
print('=== Phase-2 pass-0 relationship coverage ===')
p2_files = [p for p in os.listdir(ckpt_base) if p.startswith('pass_0_')]
per_doc_p2 = {d: [] for d in ['AAPL', 'ADI', 'AVGO', 'unknown']}
for p in p2_files:
    with open(os.path.join(ckpt_base, p), 'rb') as f:
        cid, result = pickle.load(f)
    # result is (new_ents_df, rels_df, known_set)
    n_rels = len(result[1]) if isinstance(result, tuple) and len(result) >= 2 else 0
    doc = next((d for d, ids in doc_sets.items() if cid in ids), 'unknown')
    per_doc_p2[doc].append((cid, n_rels))

for doc, items in per_doc_p2.items():
    if not items:
        continue
    empty = sum(1 for _, n in items if n == 0)
    non_empty = [(c, n) for c, n in items if n > 0]
    avg = sum(n for _, n in non_empty) / len(non_empty) if non_empty else 0
    print(f'{doc}: {len(non_empty)}/{len(items)} non-empty  ({empty} empty)  avg_rels={avg:.1f}')

print()
print('=== ADI empty phase-1 chunks — text preview ===')
adi_empty = [(c, n) for c, n in per_doc_p1['ADI'] if n == 0]
for cid, _ in adi_empty:
    if cid in tus_by_id.index:
        text = tus_by_id.loc[cid, 'text'][:200].replace('\n', ' ').strip()
        print(f'  {cid[:12]}... | {text}')
    else:
        print(f'  {cid[:12]}... | (not in current TUs)')
