import pandas as pd

tus = pd.read_parquet('output/text_units.parquet')
docs = pd.read_parquet('output/documents.parquet')
ents = pd.read_parquet('output/entities.parquet')
rels = pd.read_parquet('output/relationships.parquet')

doc_title = docs.set_index('id')['title'].to_dict()
tus['doc'] = tus['document_id'].map(doc_title)

ents_exploded = ents.explode('text_unit_ids').dropna(subset=['text_unit_ids'])
rels_exploded = rels.explode('text_unit_ids').dropna(subset=['text_unit_ids'])

chunks_with_ents = set(ents_exploded['text_unit_ids'])
chunks_with_rels = set(rels_exploded['text_unit_ids'])

tus['has_entities'] = tus['id'].isin(chunks_with_ents)
tus['has_rels'] = tus['id'].isin(chunks_with_rels)
tus['entity_count'] = tus['id'].map(ents_exploded.groupby('text_unit_ids').size()).fillna(0).astype(int)
tus['rel_count'] = tus['id'].map(rels_exploded.groupby('text_unit_ids').size()).fillna(0).astype(int)

print('=== ENTITY COVERAGE BY DOC ===')
g = tus.groupby('doc').agg(
    total=('id','count'),
    with_entities=('has_entities','sum'),
    with_rels=('has_rels','sum'),
    avg_entities=('entity_count','mean'),
    avg_rels=('rel_count','mean'),
)
print(g.to_string())
print()

print('=== CHUNK CATEGORY BREAKDOWN ===')
def cat(row):
    if row['has_entities'] and row['has_rels']:
        return 'ent+rel'
    elif row['has_entities']:
        return 'ent only'
    elif row['has_rels']:
        return 'rel only'
    else:
        return 'empty'
tus['category'] = tus.apply(cat, axis=1)
print(tus['category'].value_counts())
print()

print('=== SAMPLE EMPTY CHUNKS (first 15) ===')
empty = tus[tus['category'] == 'empty']
for _, row in empty.head(15).iterrows():
    doc = row['doc']
    preview = row['text'][:250].replace('\n', ' ').strip()
    print(f"[{doc}] {preview}")
    print()
