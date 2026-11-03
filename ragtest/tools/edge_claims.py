"""edge_claims.py — Post-hoc deterministic edge-claims exploder.

Reads raw_relationships.parquet + text_units.parquet + documents.parquet from
the index output directory and produces edge_claims.parquet.

Usage (from repo root):
    uv run python ragtest/tools/edge_claims.py [output_dir]

    output_dir defaults to ragtest/output

Requirements:
    - snapshots.raw_graph: true  in settings.yaml so raw_relationships.parquet
      is written by the pipeline.
    - The phase-2 relationship prompt must emit [PREDICATE]-prefixed descriptions
      (relationship_extraction.txt / second_pass.txt already do this).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ── Ordered, first-match-wins predicate rules ─────────────────────────────────
# Seeded from the phase-2 prompt taxonomy and masked-Leiden discovery.
# Highest-value commercial predicates come first so a fact that could match
# two lands on the more specific one. AFFECTS is sign-resolved downstream.
PREDICATE_RULES: list[tuple[str, str]] = [
    ("COMPETITOR_OF", r"\bcompet(e|es|ing|itor|ition)\b|competes? (with|for|against)|rival"),
    # Customer concentration needs a counterparty/concentration signal — NOT a
    # bare "% of revenue" (that also matches expense/margin ratios, which
    # should fall through to REPORTS_VALUE).
    ("CUSTOMER_OF", r"\bcustomer\b|largest customer|end customer|purchases from|"
                    r"distributor|accounted for (approximately )?[\d.]+|"
                    r"(constitut|represent|compris|accounted for).{0,40}"
                    r"(ten percent|10 ?%|10 ?percent|more than (ten|10)|greater than (ten|10))|"
                    r"(more than|greater than|over) (ten|10) ?(percent|%)"),
    ("SUPPLIER_OF", r"\bsuppl(y|ier|ies|ied)\b|sole[- ]source|single source|foundry|"
                    r"outsourc|third[- ]party manufactur|contract manufactur|wafer"),
    ("PARTNER_OF", r"\bpartner(s|ed|ship|ing)?\b|joint venture|collaborat(e|ed|ion|ive)|"
                   r"\balliance\b|strategic (agreement|relationship)|teamed with|co[- ]develop"),
    ("OFFICER_OF", r"(served|serves|serving) as .{0,30}(chief|president|officer|director|"
                   r"chair)|appointed .{0,30}(chief|president|officer|ceo|cfo|coo)|"
                   r"(is|was) (our|the company'?s) (chief|president|ceo|cfo|coo|chair)"),
    ("MANUFACTURES_IN", r"manufactur(e|es|ed|ing) (at|in)|fabricat(e|es|ed|ion)|"
                        r"factor(y|ies)|assembl(e|ed|y) (at|in|and test)|produced (at|in)|"
                        r"wafer fab"),
    ("OPERATES_IN", r"operations in|facilit(y|ies) in|headquarter|located in|based in|"
                    r"plants? in|offices? in|presence in|operates? in"),
    ("PART_OF", r"\bsegment\b|\bsubsidiar(y|ies)\b|\bdivision\b|business unit|"
                r"wholly[- ]owned|part of|contributed \d.*%.*(revenue|sales)|"
                r"reportable segment"),
    ("MEMBER_OF", r"member of|belongs to|trade (association|group)|industry (association|"
                  r"consortium|group)|is a member"),
    ("PROVIDES", r"\bprovides?\b|\boffers?\b|\bdelivers?\b|develops? and (markets|sells)|"
                 r"designs? and (markets|sells|manufactures)|markets? a (range|line|portfolio)"),
    ("SUBJECT_TO", r"subject to|must comply|comply with|complian|regulat(e|ed|es|ion|ory)|"
                   r"covenant|tariff|export control|antidumping|sanction|"
                   r"prohibited|restrict(ion|ed|s)?|NDAA|CHIPS Act|governed by"),
    ("PARTY_TO", r"acqui(re|red|res|sition)|merger|merged|lawsuit|litigation|"
                 r"investigation|complaint|settl(e|ed|ement)|court|antitrust|"
                 r"class action|patent infringement|allege"),
    ("DEPENDS_ON", r"depend(s|ent|ence|ing)?\b|rel(y|ies|iance|ied) (on|upon)|"
                   r"concentrat(e|ed|ion)|exposure to|exposed to|reliant"),
    ("AFFECTS", r"\bdr(o|i)ve[sn]?\b|driven by|increase in|decrease in|"
                r"adversely affect|materially (affect|reduce|impact)|"
                r"\baffect(s|ed|ing)?\b|\bimpact(s|ed|ing)?\b|contribut(e|ed|ing)"),
    # Value-bearing residue no relational rule caught (e.g. "Net income was
    # $2,510 million") — lifted last so relational predicates win first.
    ("REPORTS_VALUE", r"(was|were|totaled|totalled|amounted to|reached|reported|"
                      r"increased to|decreased to|of)\s+\$?[\d.,]+\s*"
                      r"(million|billion|thousand|%|percent)"),
]

# Valence lexicons for signing an AFFECTS match.
NEG_VALENCE = re.compile(
    r"advers|negativ|reduc|declin|\bharm|hurt|weaken|lower|decreas|"
    r"\bloss|disrupt|shortage|unfavor|impair|erod|pressure|slowdown|deterior|"
    r"headwind|softness|constrain"
)
POS_VALENCE = re.compile(
    r"grow(th|s|ing)?\b|increas|benefit|improv|boost|favor|higher|\bgain|"
    r"expand|strengthen|accelerat|tailwind|robust|\brise|rising|\bstrong"
)

VALUE_RE = re.compile(
    r"\$[\d.,]+\s*(?:million|billion|thousand)?|"
    r"(?:MYR|EUR|CNY|RMB|€)\s*[\d.,]+\s*(?:million|billion)?|"
    r"\d+(?:\.\d+)?\s*%"
)

# ── Scope regex ───────────────────────────────────────────────────────────────
# Must produce the SAME string as settings.yaml scoped_facts.label_pattern /
# label_replacement so edge-claim scopes align with entity-claim scopes.
SCOPE_RE = re.compile(r"^([A-Z.-]+)_10k_(\d{4})", re.I)

# ── Emitted-prefix schema (MUST byte-match the prompt's label set) ────────────
EMITTED_SCHEMA: frozenset[str] = frozenset({
    "CUSTOMER_OF", "SUPPLIER_OF", "COMPETITOR_OF", "PARTNER_OF", "REVENUE_SPLIT",
    "OPERATES_IN", "PRODUCES", "SUBJECT_TO", "PARTY_TO", "DEPENDS_ON",
    "IMPACTS", "RELATED_TO",
})

# The extractor's _strip_placeholder_brackets strips the LEADING '[' from
# descriptions, so "[LABEL] text" arrives as "LABEL] text". Accept both.
_EMITTED_PREFIX = re.compile(r"^\s*\[?([A-Z_]+)\]\s*(.*)", re.S)


def split_emitted(text: str) -> tuple[str | None, str]:
    """Return (schema predicate, statement-without-prefix) if the fact starts
    with a valid emitted label, else (None, original text)."""
    m = _EMITTED_PREFIX.match(text)
    if m and m.group(1) in EMITTED_SCHEMA:
        return m.group(1), m.group(2).strip()
    return None, text


def type_fact(text: str) -> str:
    """Apply ordered regex rules; sign AFFECTS into NEGATIVELY/POSITIVELY_IMPACTS."""
    t = text.lower()
    for predicate, pattern in PREDICATE_RULES:
        if re.search(pattern, t):
            if predicate == "AFFECTS":
                neg, pos = NEG_VALENCE.search(t), POS_VALENCE.search(t)
                if neg and not pos:
                    return "NEGATIVELY_IMPACTS"
                if pos and not neg:
                    return "POSITIVELY_IMPACTS"
                return "AFFECTS"
            return predicate
    return "UNTYPED"


def extract_value(text: str) -> str | None:
    m = VALUE_RE.search(text)
    return m.group(0).strip() if m else None


def _scope_from_title(title: str) -> str:
    m = SCOPE_RE.match(str(title))
    return f"{m.group(1).upper()} FY{m.group(2)}" if m else str(title)


def main(output_dir: Path) -> None:
    print(f"edge_claims: reading from {output_dir}")

    raw_rel_path = output_dir / "raw_relationships.parquet"
    if not raw_rel_path.exists():
        raise FileNotFoundError(
            f"{raw_rel_path} not found. "
            "Ensure snapshots.raw_graph: true in settings.yaml and re-run the pipeline."
        )

    raw = pd.read_parquet(raw_rel_path)
    tu = pd.read_parquet(output_dir / "text_units.parquet")[["id", "document_id"]]
    docs = pd.read_parquet(output_dir / "documents.parquet")[["id", "title"]]

    # text_unit_id -> scope label (TICKER FYxxxx)
    doc_scope: dict = {}
    for _, d in docs.iterrows():
        doc_scope[d["id"]] = _scope_from_title(str(d["title"]))
    tu_scope = {row["id"]: doc_scope.get(row["document_id"], "?") for _, row in tu.iterrows()}

    # Explode description + text_unit_ids together (parallel per-fact arrays).
    needed = {"source", "target", "description", "text_unit_ids", "weight"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"raw_relationships.parquet is missing columns: {missing}")

    ex = raw[["source", "target", "description", "text_unit_ids", "weight"]].copy()
    ex = ex.explode(["description", "text_unit_ids"]).dropna(subset=["description"])
    ex = ex.reset_index(drop=True)
    ex = ex.rename(columns={"description": "statement", "text_unit_ids": "text_unit_id"})

    # Emitted-prefix routing first; regex rules are the fallback.
    emitted = ex["statement"].map(split_emitted)
    ex["emitted_pred"] = [p for p, _ in emitted]
    ex["statement"] = [s for _, s in emitted]
    ex["predicate"] = [
        p if p is not None else type_fact(s)
        for p, s in zip(ex["emitted_pred"], ex["statement"])
    ]
    ex["predicate_src"] = ["emitted" if p is not None else "regex"
                           for p in ex["emitted_pred"]]
    ex.drop(columns=["emitted_pred"], inplace=True)

    ex["value"] = ex["statement"].map(extract_value)
    ex["scope"] = ex["text_unit_id"].map(tu_scope).fillna("?")

    n_emit = (ex["predicate_src"] == "emitted").sum()
    if n_emit:
        print(f"(emitted-prefix routing: {n_emit:,}/{len(ex):,} facts "
              f"({n_emit / len(ex):.1%}) typed by prompt label; "
              f"rest fell back to regex rules)")

    out = ex[["source", "target", "predicate", "predicate_src", "value",
              "scope", "statement", "text_unit_id", "weight"]]
    out_path = output_dir / "edge_claims.parquet"
    out.to_parquet(out_path, index=False)

    # ── Coverage report ───────────────────────────────────────────────────────
    n = len(out)
    typed = (out["predicate"] != "UNTYPED").sum()
    with_value = out["value"].notna().sum()
    print(f"\n{'=' * 62}\nEDGE-CLAIMS COVERAGE\n{'=' * 62}")
    print(f"per-chunk facts (exploded):  {n:,}")
    print(f"unique edges:                {out.groupby(['source','target']).ngroups:,}")
    print(f"avg facts/edge:              {n / max(1, out.groupby(['source','target']).ngroups):.2f}")
    print(f"typed by rules:              {typed:,}  ({typed / n:.1%})")
    print(f"UNTYPED (retrievable text):  {n - typed:,}  ({(n - typed) / n:.1%})")
    print(f"carry a $/% value:           {with_value:,}  ({with_value / n:.1%})")
    print(f"scope resolved:              {(out['scope'] != '?').sum():,}")

    print(f"\n{'predicate':<22}{'count':>8}{'share':>8}{'w/value':>9}")
    print("-" * 47)
    counts = out["predicate"].value_counts()
    for pred, c in counts.items():
        wv = out[(out["predicate"] == pred) & out["value"].notna()].shape[0]
        print(f"{pred:<22}{c:>8,}{c / n:>7.1%}{wv:>9,}")

    print("\n--- 12 UNTYPED samples (what the rules miss) ---")
    unt = out[out["predicate"] == "UNTYPED"]["statement"]
    for s in unt.sample(min(12, len(unt)), random_state=7):
        print("  -", str(s)[:130])

    print("\n--- CUSTOMER_OF facts carrying a value ---")
    cust = out[(out["predicate"] == "CUSTOMER_OF") & out["value"].notna()]
    for _, row in cust.sample(min(6, len(cust)), random_state=5).iterrows():
        print(f"  [{row['scope']}] {row['source']} -> {row['target']} | value={row['value']!r}")
        print(f"      {str(row['statement'])[:120]}")

    print(f"\nwrote {out_path}  ({n:,} claims)")


if __name__ == "__main__":
    _output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "output"
    main(_output_dir)

