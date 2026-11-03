# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

SUMMARIZE_PROMPT = """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.
Limit the final description length to {max_length} words.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

SCOPED_SUMMARIZE_PROMPT = """You are a precise knowledge-base author. Write a factual portrait of the entity below from the provided source fragments, then record scope-specific claims.

Entity: {entity_name}

Source fragments (each tagged [DOCUMENT_TITLE]):
{fragment_list}

Adjacent relationships (for context only — do not invent new ones):
{relationship_list}

Instructions:
1. Write a concise general portrait in third person that integrates what is consistent across ALL fragments. Do NOT include document-specific numbers or thresholds in the portrait — those belong in CLAIM lines.
2. For each statement that differs between documents (different thresholds, ratings, scores, time horizons, etc.) emit a CLAIM line in this exact format:
   CLAIM | <document_title> | <statement_about_entity> | <fragment_refs e.g. F1 F3>
   - <document_title> must match one of the [DOCUMENT_TITLE] tags exactly.
   - <statement_about_entity> must be grounded in the fragment(s) cited.
   - Only emit CLAIM lines for genuine scope differences — not for synonyms, acronym expansions, or rephrasing.
3. End with <|COMPLETE|>.

Max description length: {max_length} words.

Output (portrait first, then CLAIM lines):"""

SCOPED_SUMMARIZE_DERIVED_PROMPT = """You are an expert analyst skilled at decoding intricate regulatory and methodological frameworks.

You are given: one entity (or entity pair), a numbered list of description fragments —
each tagged with the source document it came from — and a list of the entity's strongest
relationships, also tagged.

Write a single comprehensive description in English, in third person, including the
entity names so we have full context.

Rules:
1. Include information from all fragments. If any fragment contains a formula, equation,
   ratio definition, numeric scale, enumeration, or threshold, reproduce it VERBATIM —
   never paraphrase an equation or drop an operator, coefficient, or threshold value.
2. Keep every conditional qualifier attached to its fact ("for investment-grade
   issuers...", "when owned by a financial sponsor..."). Never turn a conditional fact
   into an unconditional one.
3. THE DESCRIPTION IS THE GENERAL PORTRAIT. Write what is true of the entity across
   all fragments — its role, definition, and shared facts. When scoped claims are
   emitted (rule 5), do not repeat the conflicting values in the description; state
   the shared core and note that the specifics differ by document or condition.
4. A difference between fragments is SCOPE-DRIVEN if and only if the conflicting
   fragments come from different documents (different tags), or carry different
   stated conditions ("for investment-grade issuers...", "when owned by a financial
   sponsor..."). Never merge or resolve a scope-driven difference.
5. CLAIMS carry the scoped facts. When fragments assign INCOMPATIBLE values to the
   SAME attribute and the difference is scope-driven, emit one line per scoped value,
   after the description, in exactly this format:
   CLAIM | <condition> | <one-sentence factual statement> | <supporting fragments, e.g. F2,F5>
   - Do NOT name the source document. The system assigns it from the fragments you cite,
     so cite each fact's fragments accurately — that citation is what fixes its scope.
   - <condition> is ONLY a stated within-document qualifier ("investment-grade issuers",
     "when owned by a financial sponsor"). Leave it EMPTY when the values differ purely
     by source document. Always keep the empty field and both its pipes:
     CLAIM |  | <statement> | F#
   - Most entities need NO claims.
6. A scope-driven conflict too diffuse for a claim (no crisp attribute and value)
   stays in the description as clearly separated, attributed threads:
   [<document label or condition>] <that account>. Never blend the threads into one
   statement.
7. A contradiction you CANNOT attribute to a document or a stated condition is source
   noise, not a scope: mention both readings neutrally in one sentence, do not resolve
   it, do not invent a scope for it, and do not emit a claim.
8. Do NOT expand an acronym to a full name unless that full name appears verbatim in the
   fragments. If the fragments give only the acronym, use only the acronym.
9. If the fragments are consistent, output only the description — no threads, no claims.

Limit the description to {max_length} words.

######################
-Example-
Entities: "ANNUAL LEAVE"
Fragments:
F1 [HR Policy 2019] Annual leave is fixed at 15 days for all employees.
F2 [HR Policy 2019] Annual leave requests require two weeks notice.
F3 [Union Agreement] Annual leave is 20 days, rising to 25 after five years of service.
Relationships:
- [HR Policy 2019] ANNUAL LEAVE -> HR DEPARTMENT: leave requests are approved by the HR department
Output:
ANNUAL LEAVE is the employees' paid vacation entitlement. Requests require two weeks
notice and are approved by the HR department. Its duration is set differently by the two
agreements.
CLAIM |  | Annual leave is 15 days for all employees. | F1
CLAIM |  | Annual leave is 20 days, rising to 25 after five years of service. | F3
######################
-Data-
Entities: {entity_name}
Fragments:
{fragment_list}
Relationships:
{relationship_list}
Output:"""
