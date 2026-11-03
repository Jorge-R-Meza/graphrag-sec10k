"""
Enrich graph.graphml with attributes from entities.parquet and relationships.parquet
for visualization in Gephi.

Output: output/graph_enriched.graphml
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
GRAPHML_IN  = OUTPUT_DIR / "graph.graphml"
GRAPHML_OUT = OUTPUT_DIR / "graph_enriched.graphml"

# ── load parquet data ──────────────────────────────────────────────────────────
try:
    ent = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
except FileNotFoundError:
    raise FileNotFoundError(f"entities.parquet not found at {OUTPUT_DIR / 'entities.parquet'}")

try:
    rel = pd.read_parquet(OUTPUT_DIR / "relationships.parquet")
except FileNotFoundError:
    raise FileNotFoundError(f"relationships.parquet not found at {OUTPUT_DIR / 'relationships.parquet'}")

# build lookup dicts keyed by title (uppercase, matches graphml node ids)
entity_attrs = {
    row["title"]: {
        "type":        str(row["type"]),
        "description": str(row["description"]).replace("\n", " "),
        "degree":      int(row["degree"]),
        "frequency":   int(row["frequency"]),
    }
    for _, row in ent.iterrows()
    if row["title"] and str(row["title"]).strip()  # skip blank titles
}

# relationships: key = (source, target) — graphml edges use title as ids
rel_attrs = {
    (row["source"], row["target"]): {
        "description":    str(row["description"]).replace("\n", " "),
        "combined_degree": int(row["combined_degree"]),
    }
    for _, row in rel.iterrows()
}

# ── parse graphml ──────────────────────────────────────────────────────────────
NS = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

tree = ET.parse(GRAPHML_IN)
root = tree.getroot()

def tag(local: str) -> str:
    return f"{{{NS}}}{local}"

# ── define new attribute keys ──────────────────────────────────────────────────
NEW_NODE_KEYS = [
    ("type",        "string"),
    ("description", "string"),
    ("degree",      "int"),
    ("frequency",   "int"),
]
NEW_EDGE_KEYS = [
    ("description",     "string"),
    ("combined_degree", "int"),
]

# find insertion point (before first <graph> element)
graph_el = root.find(tag("graph"))
if graph_el is None:
    raise ValueError(f"No <graph> element found in {GRAPHML_IN}")
graph_idx = list(root).index(graph_el)

# assign stable key ids: d0 is taken by weight, so start at d1
existing_keys = {el.get("attr.name"): el.get("id") for el in root.findall(tag("key"))}

key_id_counter = max(
    (int(re.sub(r"\D", "", k)) for k in existing_keys.values() if re.sub(r"\D", "", k).isdigit()),
    default=-1,
) + 1

node_key_ids = {}
edge_key_ids = {}

def add_key(attr_name, attr_type, for_elem, dest_dict):
    global key_id_counter, graph_idx
    if attr_name in existing_keys:
        dest_dict[attr_name] = existing_keys[attr_name]
        return
    kid = f"d{key_id_counter}"
    key_id_counter += 1
    el = ET.Element(tag("key"), {
        "id": kid,
        "for": for_elem,
        "attr.name": attr_name,
        "attr.type": attr_type,
    })
    root.insert(graph_idx, el)
    graph_idx += 1  # update graph_idx after insertion
    dest_dict[attr_name] = kid

for name, atype in NEW_NODE_KEYS:
    add_key(name, atype, "node", node_key_ids)
for name, atype in NEW_EDGE_KEYS:
    add_key(name, atype, "edge", edge_key_ids)

# recompute graph_el position after key insertions
graph_el = root.find(tag("graph"))

# ── enrich existing nodes ─────────────────────────────────────────────────────
matched_nodes = 0
seen_node_ids = set()
for node_el in list(graph_el.findall(tag("node"))):
    node_id = node_el.get("id")
    if not node_id:                      # drop blank-id nodes — Gephi rejects them
        graph_el.remove(node_el)
        continue
    seen_node_ids.add(node_id)
    attrs = entity_attrs.get(node_id)
    if attrs is None:
        continue
    matched_nodes += 1
    for attr_name, kid in node_key_ids.items():
        data_el = ET.SubElement(node_el, tag("data"), {"key": kid})
        data_el.text = str(attrs[attr_name])

# ── add isolated nodes (extracted but no edges) ────────────────────────────────
added_isolated = 0
for title, attrs in entity_attrs.items():
    if title in seen_node_ids:
        continue
    node_el = ET.SubElement(graph_el, tag("node"), {"id": title})
    for attr_name, kid in node_key_ids.items():
        data_el = ET.SubElement(node_el, tag("data"), {"key": kid})
        data_el.text = str(attrs[attr_name])
    added_isolated += 1

# ── enrich edges ──────────────────────────────────────────────────────────────
matched_edges = 0
edge_id_counter = 0
for edge_el in list(graph_el.findall(tag("edge"))):
    src = edge_el.get("source", "")
    tgt = edge_el.get("target", "")
    if not src or not tgt:               # drop edges with blank endpoints
        graph_el.remove(edge_el)
        continue
    # Gephi requires every edge to have an explicit id attribute
    if not edge_el.get("id"):
        edge_el.set("id", f"e{edge_id_counter}")
    edge_id_counter += 1
    # Enrich with attributes from rel_attrs
    attrs = rel_attrs.get((src, tgt)) or rel_attrs.get((tgt, src))
    if attrs is None:
        continue
    matched_edges += 1
    for attr_name, kid in edge_key_ids.items():
        data_el = ET.SubElement(edge_el, tag("data"), {"key": kid})
        data_el.text = str(attrs[attr_name])

# ── write output ───────────────────────────────────────────────────────────────
ET.indent(tree, space="  ")
tree.write(GRAPHML_OUT, encoding="unicode", xml_declaration=True)

print(f"Nodes enriched  : {matched_nodes}")
print(f"Isolated added  : {added_isolated}")
print(f"Total nodes     : {matched_nodes + added_isolated}")
print(f"Edges enriched  : {matched_edges}")
print(f"Output written  : {GRAPHML_OUT}")
