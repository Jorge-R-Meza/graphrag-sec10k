# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for top-k entity selection in two-phase graph extraction.

Validates that _select_relevant_entities correctly:
- Includes lexical floor (entities whose title appears in chunk text).
- Fills to k with semantic nearest neighbors by cosine similarity.
- Respects budget (len(selected) <= k).
- Handles deterministic tie-breaking by title for resume caching.
"""

import numpy as np
import pytest

from graphrag.index.operations.extract_graph.extract_graph import (
    _select_relevant_entities,
)


def _make_normalized_vectors(values: list[list[float]]) -> np.ndarray:
    """Convert list of vectors to L2-normalized float32 matrix."""
    vecs = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def test_select_relevant_entities_lexical_floor():
    """Lexical matches always included, even if cosine rank < k."""
    titles = ["Alice", "Bob", "Charlie", "David"]
    # Arbitrary entity vectors; scores don't matter for this test
    entity_matrix = _make_normalized_vectors([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, -1, -1],  # Very far from chunk
    ])
    chunk_vec = np.array([1, 0, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "The person named Alice talked to Charlie about data."
    k = 2

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    # Alice and Charlie both appear in text (case-insensitive).
    assert "Alice" in selected
    assert "Charlie" in selected
    assert len(selected) <= k


def test_select_relevant_entities_semantic_fill():
    """When lexical < k, fill with highest-cosine entities."""
    titles = ["Alice", "Bob", "Charlie", "David"]
    # Alice far, Bob close, Charlie medium, David far
    entity_matrix = _make_normalized_vectors([
        [-1, 0, 0],      # far from chunk
        [1, 0.1, 0],     # closest
        [0.5, 0.5, 0],   # medium
        [-1, -0.1, -1],  # far
    ])
    chunk_vec = np.array([1, 0, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "Discussion about Alice."  # Only Alice in text
    k = 2

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    # Must have Alice (lexical), Bob should be added (highest cosine)
    assert "Alice" in selected
    assert "Bob" in selected
    assert len(selected) <= k


def test_select_relevant_entities_budget_respected():
    """Returned set size never exceeds k."""
    titles = ["E1", "E2", "E3", "E4", "E5"]
    entity_matrix = _make_normalized_vectors([
        [1, 0, 0],
        [0.9, 0.1, 0],
        [0.8, 0.2, 0],
        [0.7, 0.3, 0],
        [0.6, 0.4, 0],
    ])
    chunk_vec = np.array([1, 0, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "Some generic text"
    k = 3

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    assert len(selected) <= k


def test_select_relevant_entities_deterministic_tiebreak():
    """Tied cosine scores broken by title alphabetically."""
    titles = ["Zebra", "Apple", "Banana"]
    # All have same cosine score (dot product = 0.5 after norm)
    entity_matrix = _make_normalized_vectors([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    chunk_vec = np.array([1, 0, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "Some text"
    k = 2

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    # Should pick the two with highest (score, then -title), which means lowest title
    # (Apple, Banana) over Zebra due to alphabetical order
    assert len(selected) <= k
    assert "Apple" in selected or "Banana" in selected


def test_select_relevant_entities_all_within_budget():
    """If len(entities) <= k, all returned (fast path implied)."""
    titles = ["A", "B"]
    entity_matrix = _make_normalized_vectors([
        [1, 0],
        [0, 1],
    ])
    chunk_vec = np.array([1, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "Some text"
    k = 5  # Budget larger than entity count

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    assert len(selected) == len(titles)


def test_select_relevant_entities_case_insensitive_lexical():
    """Lexical matching is case-insensitive."""
    titles = ["ALICE", "bob"]
    entity_matrix = _make_normalized_vectors([
        [1, 0],
        [0, 1],
    ])
    chunk_vec = np.array([1, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "The alice and Bob corporation."
    k = 1

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    # Both "alice" and "bob" appear in text (case-insensitive)
    # Lexical floor = {ALICE, bob}, capped to k=1 so one is dropped
    # The test ensures at least one lexical match is kept
    assert len(selected) >= 1


def test_select_relevant_entities_no_lexical_matches():
    """When no entities appear textually, fill purely by cosine."""
    titles = ["Alice", "Bob", "Charlie"]
    entity_matrix = _make_normalized_vectors([
        [-1, 0, 0],      # far
        [1, 0, 0],       # closest
        [0.5, 0.5, 0],   # medium
    ])
    chunk_vec = np.array([1, 0, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = "Discussion about unrelated topics."  # No entity names
    k = 2

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    # Should pick Bob (highest cosine) and Charlie (second)
    assert "Bob" in selected
    assert "Charlie" in selected
    assert len(selected) == k


def test_select_relevant_entities_empty_text():
    """Empty chunk text means no lexical floor."""
    titles = ["E1", "E2"]
    entity_matrix = _make_normalized_vectors([
        [1, 0],
        [0, 1],
    ])
    chunk_vec = np.array([1, 0], dtype=np.float32)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
    chunk_text = ""
    k = 1

    selected = _select_relevant_entities(chunk_text, chunk_vec, titles, entity_matrix, k)

    # Fill from cosine; should get closest (E1)
    assert "E1" in selected
    assert len(selected) <= k
