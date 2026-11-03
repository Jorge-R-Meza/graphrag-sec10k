# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A file containing prompts definition."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.
 
4. When finished, output <|COMPLETE|>
 
######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
<|COMPLETE|>

######################
Example 2:
Entity_types: ORGANIZATION
Text:
TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
######################
Output:
("entity"<|>TECHGLOBAL<|>ORGANIZATION<|>TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones)
##
("entity"<|>VISION HOLDINGS<|>ORGANIZATION<|>Vision Holdings is a firm that previously owned TechGlobal)
##
("relationship"<|>TECHGLOBAL<|>VISION HOLDINGS<|>Vision Holdings formerly owned TechGlobal from 2014 until present<|>5)
<|COMPLETE|>

######################
Example 3:
Entity_types: ORGANIZATION,GEO,PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
######################
Output:
("entity"<|>FIRUZABAD<|>GEO<|>Firuzabad held Aurelians as hostages)
##
("entity"<|>AURELIA<|>GEO<|>Country seeking to release hostages)
##
("entity"<|>QUINTARA<|>GEO<|>Country that negotiated a swap of money in exchange for hostages)
##
##
("entity"<|>TIRUZIA<|>GEO<|>Capital of Firuzabad where the Aurelians were being held)
##
("entity"<|>KROHAARA<|>GEO<|>Capital city in Quintara)
##
("entity"<|>CASHION<|>GEO<|>Capital city in Aurelia)
##
("entity"<|>SAMUEL NAMARA<|>PERSON<|>Aurelian who spent time in Tiruzia's Alhamia Prison)
##
("entity"<|>ALHAMIA PRISON<|>GEO<|>Prison in Tiruzia)
##
("entity"<|>DURKE BATAGLANI<|>PERSON<|>Aurelian journalist who was held hostage)
##
("entity"<|>MEGGIE TAZBAH<|>PERSON<|>Bratinas national and environmentalist who was held hostage)
##
("relationship"<|>FIRUZABAD<|>AURELIA<|>Firuzabad negotiated a hostage exchange with Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>AURELIA<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>QUINTARA<|>FIRUZABAD<|>Quintara brokered the hostage exchange between Firuzabad and Aurelia<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>ALHAMIA PRISON<|>Samuel Namara was a prisoner at Alhamia prison<|>8)
##
("relationship"<|>SAMUEL NAMARA<|>MEGGIE TAZBAH<|>Samuel Namara and Meggie Tazbah were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>DURKE BATAGLANI<|>Samuel Namara and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>DURKE BATAGLANI<|>Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release<|>2)
##
("relationship"<|>SAMUEL NAMARA<|>FIRUZABAD<|>Samuel Namara was a hostage in Firuzabad<|>2)
##
("relationship"<|>MEGGIE TAZBAH<|>FIRUZABAD<|>Meggie Tazbah was a hostage in Firuzabad<|>2)
##
("relationship"<|>DURKE BATAGLANI<|>FIRUZABAD<|>Durke Bataglani was a hostage in Firuzabad<|>2)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Do NOT repeat what was already extracted. Do NOT explain what you will do. Output ONLY the missing entities and relationships using the exact same format — (\"entity\"<|>NAME<|>TYPE<|>DESCRIPTION) and (\"relationship\"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH) — separated by ##. When done, output <|COMPLETE|>.\n"
ENTITY_CONTINUE_PROMPT = "MANY entities were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types, and do NOT emit relationships. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"

CLASSIFY_CHUNK_ENDPOINTS_PROMPT = """-Task-
Some relationship endpoints extracted from the passage below have not yet been classified as entities. Classify each one that fits a valid entity type.

-Valid entity types-
{entity_types}

-Unclassified endpoints (classify each that fits a valid type)-
{endpoint_list}

-Passage they were found in-
{input_text}

-Instructions-
For each endpoint you can confidently classify into one of the valid types above, output:
("entity"<|>NAME<|>TYPE<|>DESCRIPTION)
where NAME matches the endpoint exactly (upper-case), TYPE is one of the valid types, and DESCRIPTION is a one-sentence characterization supported by the passage.
Output ONLY endpoints you can classify. Skip any that do not fit any valid type.
Use ## to separate entries. When done, output <|COMPLETE|>.

output:"""

ENTITY_GLEANING_PROMPT = """-Task-
Review the text below and identify any entities that were MISSED in a previous extraction pass.

-Valid entity types-
{entity_types}

-Already extracted entities (do NOT re-extract these)-
{extracted_entities}

-Text to re-examine-
{input_text}

-Instructions-
Identify ONLY entities of the valid types above that appear in the text but are NOT in the already-extracted list above.
Do NOT repeat any entity already listed. Do NOT explain what you will do. Output ONLY new missing entities.
Format each entity as ("entity"<|>NAME<|>TYPE<|>DESCRIPTION)
Separate entries with ##. When done, output <|COMPLETE|>.

output:"""

RELATIONSHIP_EXTRACTION_PROMPT = """-Task-
Given a text and a complete list of entities, extract ALL relationships between those entities.

-Entities-
{entity_list}

-Text-
{input_text}

-Instructions-
For each pair of entities from the list above that are related in the text, output:
("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH)

Strength scale:
1-3 = entities co-occur or are tangentially related
4-6 = stated but indirect relationship
7-10 = direct, explicit relationship in the text
Err on the side of including weak relationships at low strength rather than omitting them.

Use ## to separate entries. When done, output <|COMPLETE|>.

output:"""

SECOND_PHASE_PROMPT = """-Goal-
You are given a list of KNOWN ENTITIES already identified across a document, and ONE passage from that document. Identify every relationship that THIS passage supports.
-Use the known-entity list for salience-
Some known entities appear in this passage in NON-OBVIOUS ways — a defined term, an acronym, or a descriptive phrase that does not read as a named entity on its own. Recognize them and connect them, reusing their EXACT names. For example, "the entire unfunded liability is refinanced annually and never paid down" describes an open amortization method even if it is never named — if a matching known entity exists, link it.
-Steps-
1. Identify relationships EXHAUSTIVELY. For every pair of entities this passage connects in ANY way, extract a relationship — do not restrict yourself to the few strongest links; a passage that mentions N entities usually supports many more than N relationships. For each, extract:
  - source_entity, target_entity: prefer a known entity's EXACT name; if a clearly-named entity in this passage is NOT in the list, you may still use it as an endpoint (name it exactly as the passage does — it will be added as a new entity).
  - relationship_description: why they are related, grounded in this passage
  - relationship_strength: integer 1-10 (1-3 tangential/co-occurring; 4-6 stated but indirect; 7-10 direct/explicit). Report a weak link rather than omitting it.
  Format each as ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>STRENGTH)
  Actively look for these connection kinds, not just the explicit ones:
  - DEFINES / DESCRIBES: an entity is defined or characterized in terms of another.
  - COMPONENT-OF / PART-OF: an entity is a term in a formula, a subfactor of a factor, a step in a process, or a member of a category.
  - TRIGGERS / CONSTRAINS / CAPS: an entity causes, gates, limits, or sets a condition on another.
  - IMPROVES / WORSENS / AFFECTS: an entity raises, lowers, or otherwise influences another's value or assessment.
  - THRESHOLD / BAND: a numeric value, band, or category maps to a score, outcome, or classification.
  - EXAMPLE-OF / INSTANCE-OF: an entity is a specific instance or member of a general one.
  - USES / DERIVED-FROM / REFERENCES: an entity is computed from, sourced from, or refers to another.
2. DENSITY CHECK: the entities in a passage form a connected web, not a flat list. Your relationships should OUTNUMBER the known entities that appear in this passage. If you have fewer, you have under-connected — re-read for the structural, conditional, co-occurrence, and component links you skipped, and add them at strength 3-6.
3. Output ONLY relationships — do NOT output entity records. Use **##** as the delimiter.
4. When finished, output <|COMPLETE|>.
######################
-Example-
Known entities:
- OPEN AMORTIZATION METHOD
- UNFUNDED PENSION LIABILITY
- DEBT AND LIABILITIES ASSESSMENT
text:
Under this approach the entire unfunded liability is reset and refinanced each year, so it is projected to never be fully paid down. We view such approaches as weak when assessing debt and liabilities, which can worsen the pension funding discipline score.
------------------------
output:
("relationship"<|>OPEN AMORTIZATION METHOD<|>UNFUNDED PENSION LIABILITY<|>The open amortization method resets and refinances the entire unfunded pension liability each year instead of paying it down<|>9)
##
("relationship"<|>OPEN AMORTIZATION METHOD<|>DEBT AND LIABILITIES ASSESSMENT<|>Open amortization methods are viewed as weak when assessing debt and liabilities<|>7)
##
("relationship"<|>OPEN AMORTIZATION METHOD<|>PENSION FUNDING DISCIPLINE SCORE<|>A weak (open) amortization approach can worsen the pension funding discipline score<|>6)
<|COMPLETE|>
######################
-Known entities (recognize and reuse these where they appear; you may also name new endpoints)-
{entity_list}
######################
-Text-
text: {input_text}
######"""

CLASSIFY_ORPHAN_ENDPOINTS_PROMPT = """-Task-
You are given entity names that appeared as relationship endpoints but were never classified. Use the relationship snippets shown under each name to assign each one a type and write a brief description.

-Valid entity types-
{entity_types}

-Entities to classify (each shown with snippets that reference it)-
{entity_list}

-Instructions-
For each entity you can confidently classify into one of the valid types above, output:
("entity"<|>NAME<|>TYPE<|>DESCRIPTION)
where DESCRIPTION is a one-sentence characterization inferred from the snippets.
Output ONLY entities you can classify. Skip any that do not fit any valid type.
Use NAME exactly as given (upper-case). Use ## to separate entries. When done, output <|COMPLETE|>.

output:"""
