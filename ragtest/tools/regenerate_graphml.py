"""Regenerate graph.graphml from relationships.parquet using a simple Graph."""

from pathlib import Path

import networkx as nx
import pandas as pd

OUTPUT_DIR = Path("output")

rels = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
graph = nx.from_pandas_edgelist(rels, edge_attr=["weight"])
graphml = "\n".join(nx.generate_graphml(graph))
(OUTPUT_DIR / "graph.graphml").write_text(graphml, encoding="utf-8")

print(f"Nodes : {graph.number_of_nodes()}")
print(f"Edges : {graph.number_of_edges()}")
print(f"Written: {OUTPUT_DIR / 'graph.graphml'}")
