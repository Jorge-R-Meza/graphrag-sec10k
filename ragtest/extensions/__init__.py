# Copyright (c) 2026.
# Local extensions for the ragtest GraphRAG project.
#
# Importing this package triggers registration of custom embedding providers
# with the graphrag-llm factories.

from ragtest.extensions import local_embed  # noqa: F401  (registers via side-effect)
