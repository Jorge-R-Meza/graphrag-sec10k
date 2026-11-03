import pandas as pd

tus = pd.read_parquet('output/text_units.parquet')
docs = pd.read_parquet('output/documents.parquet')

print('=== TOKEN COUNTS ===')
print(f'Text units: {len(tus)}')
if 'n_tokens' in tus.columns:
    total = int(tus['n_tokens'].sum())
    print(f'Total tokens (text_units.n_tokens): {total:,}')
    print(f'Mean tokens/chunk: {tus["n_tokens"].mean():.0f}  Max: {int(tus["n_tokens"].max())}')
    col = 'n_tokens'
else:
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        tus['_tok'] = tus['text'].apply(lambda t: len(enc.encode(t)))
        col = '_tok'
        print(f'Total tokens (cl100k estimate): {int(tus[col].sum()):,}')
        print(f'Mean tokens/chunk: {tus[col].mean():.0f}  Max: {int(tus[col].max())}')
    except ImportError:
        tus['_tok'] = tus['text'].apply(lambda t: len(t.split()))
        col = '_tok'
        print(f'Total words (approx): {int(tus[col].sum()):,}')

print()
print('=== PER DOCUMENT ===')
doc_title = docs.set_index('id')['title'].to_dict()
tus['doc'] = tus['document_id'].map(doc_title)
grp = tus.groupby('doc')[col].agg(['sum', 'count', 'mean'])
grp.columns = ['total_tokens', 'chunks', 'avg_tokens_per_chunk']
print(grp.to_string())
