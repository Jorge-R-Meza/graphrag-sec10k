"""Visualize the GraphRAG knowledge graph using pyvis.

Reads graph.graphml from the index output and renders an interactive
force-directed HTML visualization.

Usage:
    uv run python ragtest/tools/visualize_graph.py
    uv run python ragtest/tools/visualize_graph.py --graphml ragtest/output/graph.graphml --out graph.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError:
    print(
        "Missing dependencies. Install with:\n  uv pip install pyvis networkx",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize GraphRAG knowledge graph")
    parser.add_argument(
        "--graphml",
        type=Path,
        default=Path("ragtest/output/graph.graphml"),
        help="Path to graph.graphml file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ragtest/graph.html"),
        help="Output HTML file",
    )
    parser.add_argument(
        "--height", type=str, default="900px", help="Canvas height"
    )
    parser.add_argument(
        "--width", type=str, default="100%%", help="Canvas width"
    )
    args = parser.parse_args()

    if not args.graphml.exists():
        print(
            f"Error: {args.graphml} not found.\n"
            "Ensure snapshots.graphml is enabled in settings.yaml and re-index.",
            file=sys.stderr,
        )
        return 1

    G = nx.read_graphml(str(args.graphml))
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    net = Network(
        height=args.height,
        width=args.width,
        notebook=False,
        directed=False,
        bgcolor="#222222",
        font_color="white",
    )

    # Color nodes by entity type if available
    TYPE_COLORS = {
        "organization": "#4e79a7",
        "person": "#f28e2b",
        "geo": "#59a14f",
        "event": "#e15759",
    }
    DEFAULT_COLOR = "#76b7b2"

    for node_id, data in G.nodes(data=True):
        label = data.get("label", data.get("name", node_id))
        entity_type = str(data.get("type", data.get("entity_type", ""))).lower()
        color = TYPE_COLORS.get(entity_type, DEFAULT_COLOR)
        title = f"<b>{label}</b><br>Type: {entity_type}"
        if "description" in data:
            desc = str(data["description"])[:300]
            title += f"<br>{desc}"
        net.add_node(node_id, label=str(label), color=color, title=title)

    for src, dst, data in G.edges(data=True):
        weight = float(data.get("weight", 1.0))
        label = str(data.get("label", data.get("description", "")))[:60]
        net.add_edge(src, dst, value=weight, title=label)

    # Physics settings for readable layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "iterations": 150
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true
        }
    }
    """)

    net.write_html(str(args.out))
    print(f"Graph visualization saved to {args.out}")
    print(f"Open in browser: start {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
