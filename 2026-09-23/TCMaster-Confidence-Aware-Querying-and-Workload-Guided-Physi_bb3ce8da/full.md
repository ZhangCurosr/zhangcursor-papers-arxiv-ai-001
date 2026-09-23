# TCMaster: Confidence-Aware Querying and Workload-Guided Physical Design for Multi-Source Traditional Chinese Medicine Knowledge Graphs

Zheng Chen<sup>∗§</sup>, Yuzhu Li<sup>†§</sup>, Haoxuan Li<sup>∗</sup>, Zhongde Zhang<sup>‡</sup>, Lianshun Jin<sup>‡</sup>, Peiwu Qin<sup>‡¶</sup>

<sup>∗</sup>Tsinghua University, China <sup>†</sup>Beijing University of Chinese Medicine, China

<sup>‡</sup>Guangdong Provincial Laboratory of Traditional Chinese Medicine, China

<sup>§</sup>Equal contribution. <sup>¶</sup>Corresponding author.

{chen-z24, li-hx24}@mails.tsinghua.edu.cn 20220121029@bucm.edu.cn

{zdzhang26, lsjin26}@outlook.com, pwqin1979@gmail.com

Abstract—Multi-source knowledge graphs (KGs) need query mechanisms that expose reliability and exploit domain structure. This paper presents TCMaster, a property-graph query substrate for confidence-aware traversal and workload-guided physical design over Traditional Chinese Medicine KGs. TCMaster integrates pharmacopoeias, prescriptions, molecular databases, and LLM-extracted micro-semantics into a KG with approximately 221K entities and 723K base edges. It annotates edges with provenance-level confidence, rewrites Cypher queries with confidence predicates, ranks multi-hop paths under PRODUCT, MIN, or weighted-average policies, and uses ontology skew through direction selection, herb-attribute bitmaps, and materialized shortcut edges. On Neo4j, direction selection improves attribute lookup by a factor of 1.47, shortcuts accelerate high-fanout target counting by a factor of 4.42, confidence filtering removes 39.3 percent of low-quality heterogeneous paths, and KG retrieval improves TCMbench QA accuracy by 20.0 percentage points.

Index Terms—Traditional Chinese Medicine Knowledge Graph, Confidence-Aware Query Processing, Workload-Guided Physical Design, Knowledge Graph Embedding, Retrieval-Augmented Generation

## I. INTRODUCTION

Traditional Chinese Medicine (TCM) contains heterogeneous knowledge across herbs, molecular ingredients, therapeutic targets, and classical prescriptions [1]–[5]. Existing TCM knowledge bases are valuable repositories, but they are not query processing systems: they rarely expose edge reliability, provenance, or physical designs for ontology-skewed graph workloads.

Making a TCM knowledge graph (KG) queryable raises three data-management challenges. First, evidence reliability varies sharply across pharmacopoeias, curated prescription sources, molecular databases, LLM-extracted microsemantics, and predicted links. Second, TCM ontologies are highly skewed: a few Nature, Flavor, Meridian, and Toxicity values connect to more than 15K Herb nodes, creating predictable traversal asymmetry. Third, clinical knowledge graph retrieval-augmented generation (KG-RAG), prescription safety inspection, and drug-target discovery require interactive multi-

## TABLE I

COMPARISON WITH REPRESENTATIVE TCM RESOURCES. M, RX, SYN,AND MICRO DENOTE MOLECULAR, PRESCRIPTION,SYNDROME/SYMPTOM, AND MICRO-SEMANTIC KNOWLEDGE.

<table><tr><td>System</td><td>Scope</td><td></td><td>Edge conf. Path query</td><td>Opt.</td><td>Artifact</td></tr><tr><td>TCMSP</td><td>M</td><td>No</td><td>No</td><td>No</td><td>Web</td></tr><tr><td>TCMID</td><td>M/Disease</td><td>No</td><td>No</td><td>No</td><td>Web</td></tr><tr><td>SymMap</td><td>Syn</td><td>No</td><td>No</td><td>No</td><td>Web</td></tr><tr><td>HERB 2.0</td><td>M</td><td>No</td><td>No</td><td>No</td><td>Web</td></tr><tr><td>ETCM</td><td>Rx</td><td>No</td><td>No</td><td>No</td><td>Web</td></tr><tr><td>OpenTCM</td><td>RAG/Diag.</td><td>No</td><td>No</td><td>No</td><td>Paper</td></tr><tr><td>TCMaster</td><td>M/Rx/Micro</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Code+Data</td></tr></table>

hop retrieval rather than static browsing. Existing resources such as TCMSP [1], TCMID [2]/TCMID 2.0 [6], SymMap [3], HERB 2.0 [4], and ETCM [5] do not jointly address these requirements. Table I summarizes the gap.

We present TCMaster, a property-graph substrate for confidence-aware traversal and workload-guided physical design over multi-source TCM KGs. TCMaster integrates 221K entities and 723K base edges, annotates every base edge with provenance-level confidence, rewrites Cypher queries with confidence predicates, ranks paths under PRODUCT, MIN, or weighted-average policies, and exploits ontology skew through direction selection, herb-attribute bitmaps, and materialized cross-layer shortcuts. It is not a new clinical LLM, diagnosis protocol, probabilistic database, or universal graph optimizer; it is a data-management substrate that makes reliability and ontology-induced asymmetry visible to query execution.

Experiments on Neo4j 4.4 show 5.2–12.0 ms latency across 1–4 hop workloads. Direction selection improves attribute lookup by 1.47×, materialized shortcuts accelerate high-fanout target counting by 4.42×, confidence filtering removes 39.3% of low-quality heterogeneous paths at θ = 0.18, and downstream KG retrieval improves TCMbench QA accuracy by 20.0 percentage points (pp) over the LLM baseline. Crosssource validation reaches 93.7% precision on herb-ingredient pairs and 95.5% audited precision on LLM-extracted micro-

semantics.

This paper makes the following contributions:

1) A confidence-bounded path query processor. We define PRODUCT, MIN, and weighted-average operational scoring policies, implement transparent Cypher rewriting for confidence constraints, and support incremental edge-confidence updates at 4.7–5.4 ms per edge without global recomputation.

2) A workload characterization of ontology-skew physical design. We exploit TCM-specific cardinality skew through traversal direction selection, compact attribute bitmaps, and materialized cross-layer shortcuts. The evaluation reports both gains and boundaries: 1.47× for attribute direction selection and 4.42× for high-fanout target counting, but limited benefit for bitmap-only and top-k lookup workloads under Neo4j’s native planner.

3) A provenance-aware KG substrate and reproducible artifact. We construct TCMaster-KG, integrating 221K entities and 723K base edges from four source types across a unified five-layer ontology, plus 28.75M materialized shortcut edges for reachability workloads. Every base edge carries a provenance-tagged confidence score, data validation reaches 93.7% precision on herbingredient pairs and 95.5% audited precision on LLMextracted micro-semantics, and the submitted artifact provides code, query templates, result files, and a Neo4j dump snapshot for reproduction.

## II. PRELIMINARIES AND PROBLEM DEFINITION

## A. Multi-Source TCM Knowledge Graph

We model TCMaster-KG as a directed labeled multi-graph $G \ = \ ( V , E , R , s r c , d s t , r e l , L )$ , following standard graph database practices [9], [11]. V is the entity set, E the edge identifier set, R the relation type set, and L assigns labels and properties. Each edge $e \in E$ has a source node src(e), a destination node $d s t ( e )$ , and a relation type $r e l ( e ) \in R$ Node and edge properties store canonical names, identifiers, provenance fields, confidence values, and derived physicaldesign attributes. TCMaster-KG currently contains 221,225 entities and 722,671 base edges across 27 entity types and 23 relation types, organized in five knowledge layers (Table II).

TABLE II  
FIVE-LAYER ONTOLOGY OF TCMASTER-KG. EDGE COUNTS SUMMARIZE PRIMARY RELATION GROUPS AND ARE ROUNDED.
<table><tr><td>Layer</td><td>Content</td><td>Primary Edges</td></tr><tr><td></td><td>L1: Molecular Herb-Ingredient-Target</td><td>265K</td></tr><tr><td>tributes</td><td>L2: TCM At- Nature, Flavor, Meridian, Toxicity</td><td>45K</td></tr><tr><td>L3: Prescription</td><td>Prescription composition</td><td>71K</td></tr><tr><td>L4: semantics</td><td>Micro- Processing, botany, efficacy, etc.</td><td>236K</td></tr><tr><td>L5: Clinical</td><td>Prescription efficacy, indication</td><td>20K</td></tr></table>

TCMaster models frequently queried TCM attributes as first-class graph nodes rather than repeated Herb properties. Thus attribute constraints become graph patterns, and Herb serves as the bridge across molecular, prescription, microsemantic, and clinical layers. The rounded layer counts in Table II cover the main relation groups; the full 722,671-edge graph also includes cross-layer mapping, normalization, and auxiliary relation types.

Source heterogeneity is preserved at the edge level. A relation imported from a molecular database and a relation extracted from text may share the same endpoint labels, but they remain distinguishable by source category, provenance pointer, and confidence score. This representation lets the query processor filter or rank evidence without duplicating the ontology for each data source.

## B. Confidence-Annotated Knowledge Graph

A confidence-annotated KG extends G with two functions: $\sigma : E  [ 0 , 1 ]$ assigns a reliability score to each edge, and $\phi : E \to S$ traces each edge to its source. The source set S contains the five levels listed in Table III; PREDICTED denotes outputs of knowledge graph embedding (KGE) models.

AUTHORITATIVE edges come from pharmacopoeias and experimentally supported molecular databases; CURATED edges come from manually organized formulary sources; LLM EXTRACTED edges cover 11 relation categories with 95.5% audited precision but receive a conservative 0.70 score to reflect hallucination risk [8]; and PREDICTED edges are generated by KGE models [16]. Unlike probabilistic databases or semiring provenance systems [17], [18], [37], TCMaster uses deterministic scores as execution annotations, exposing reliability ordering to threshold filters, path ranking, and provenance inspection.

## C. Confidence-Bounded Path Query

Given a query pattern Q, an aggregation function $A :$ $[ 0 , 1 ] ^ { * }  [ 0 , 1 ]$ , and a threshold $\theta \in [ 0 , 1 ]$ , TCMaster returns all paths matching $Q$ whose aggregated confidence satisfies:

$$
\begin{array} { r } { R e s u l t ( Q , A , \theta ) = \{ p \in P a t h s _ { G _ { c } } ( Q ) ~ | } \\ { A ( \{ \sigma ( e ) : e \in p \} ) \geq \theta \} . } \end{array}\tag{1}
$$

Three aggregation strategies are supported: PRODUCT $( \prod \sigma ( e _ { i } ) )$ , penalizing long paths with any low-confidence edge; MIN (min $\sigma ( e _ { i } ) )$ , evaluating by the weakest link; and weighted average (WAVG) with $w _ { i } = 1 / ( 1 + \log i )$ , giving higher weight to edges closer to the query start.

This definition separates candidate generation from result interpretation. Neo4j evaluates the graph pattern, TCMaster scores returned edge sequences, and θ denotes a query-time reliability threshold rather than a learned global cutoff. The system problem is to support these confidence-bounded path queries with predictable latency and auditable evidence: TC-Master must preserve provenance semantics, rewrite queries without changing logical answers except for the requested confidence or top-k policy, and exploit stable ontology skew when it reduces traversal cost.

TABLE III  
SOURCE-LEVEL CONFIDENCE ASSIGNMENT
<table><tr><td>Source Level</td><td>Conf.</td><td>Example Relations</td></tr><tr><td>AUTHORITATIVE</td><td>0.95</td><td>HAS_INGREDIENT, TARGETS</td></tr><tr><td>AUTHORITATIVE_PHARMA</td><td>0.90</td><td>HAS_PROPERTY, HAS_MERIDIAN</td></tr><tr><td>CURATED</td><td>0.85</td><td>HAS_COMPONENT, HAS_RX_EFFICACY</td></tr><tr><td>LLM_EXTRACTED</td><td>0.70</td><td>HAS_BOTANY, PROCESSED_BY</td></tr><tr><td>PREDICTED (KGE)</td><td>0.30-0.60</td><td>Link prediction outputs</td></tr></table>

## III. SYSTEM ARCHITECTURE

The formal model in §2 defines what TCMaster-KG represents; Figure 1 shows how the system realizes it. TCMaster ingests and cleans multi-source data, stores it with confidence metadata and precomputed physical structures, processes confidence-aware queries with workload-guided rewrites, and serves application-facing modes. Two invariants guide the design: every base edge remains provenance complete, and physical structures are derived views that can be rebuilt or disabled without changing the logical ontology.

![](images/0e4140443be8ce733485358f77a0adb51773149455f84cbc3b638b6e0756c94a.jpg)  
Fig. 1. TCMaster system architecture from data ingestion and cleaning to confidence-annotated Neo4j storage, workload-guided query processing, and downstream application modes.

## A. Data Integration and Quality Layer

TCMaster ingests four source categories: HERB 2.0 molecular relations [4], SymMap TCM-Western medicine links [3], classical prescription compositions (11,845 prescriptions), and LLM-extracted micro-semantics over 11 relation categories. External resources are treated as versioned snapshots with recorded dates, schema mappings, and checksums. Before ingestion, seven cleaning rules check identifier patterns, synonyms, mandatory keys, and numerical anomalies [12]–[14]. Therapeutic category triples drop from 5,095 to 724 (14.2% pass rate), while audited extraction precision rises from 91.2% to 95.5%. The layer outputs normalized edge tuples with canonical endpoints, relation type, source category, provenance pointer, validation status, and inherited confidence; discarded tuples remain in logs but do not enter the executable graph.

## B. Knowledge Graph Layer

All integrated data is stored in Neo4j 4.4 Community Edition using the property graph model. The unified ontology spans molecular interactions, TCM attributes, prescriptions, micro-semantics, and clinical prescription-level relations, with Herb entities connecting all five layers. TCMaster-KG augments the base schema with three derived structures: per-edge confidence and provenance tags (Section IV), per-herb 28-bit attribute bitmaps (Section V-C), and materialized REACHES TARGET shortcut edges for frequent Prescription→Herb→Ingredient→Target traversals (Section V-D). Confidence annotations define semantics, bitmaps summarize low-cardinality attributes, and shortcuts summarize high-fanout reachability.

These structures are kept separate because they answer different database questions. Confidence controls which evidence is admissible, bitmaps reduce local attribute filtering cost, and shortcuts reduce repeated cross-layer traversal cost. Disabling a bitmap or shortcut therefore changes performance experiments, not the logical ontology or provenance attached to base edges.

## C. Query Processing Layer

The query processing layer provides confidence rewriting, workload-guided physical rewrites, and an optional naturallanguage-to-Cypher (NL2Cypher) template interface. At runtime, a Cypher template is bound to parameters, rewritten with edge-level predicates or path-level scoring code, optionally mapped to a reversed traversal, bitmap lookup, or shortcut relation, and executed by Neo4j. TCMaster returns ranked paths with the rewrite decision and provenance-bearing edge sequence. When a shortcut is used, the corresponding basepath pattern remains defined so that the result can be expanded or audited.

## D. Application Layer

TCMaster supports clinical KG-RAG, prescription safety auditing, and drug-target discovery. All three modes consume the same result object: matched nodes, ordered edges, confidence values, source labels, and optional shortcut expansion. Applications therefore differ in templates and thresholds, not in hidden downstream schemas.

## IV. CONFIDENCE-AWARE QUERY PROCESSING

## A. Confidence Annotation Model

TCMaster-KG assigns a confidence score $\sigma ( e ) ~ \in ~ [ 0 , 1 ]$ and a source provenance label $\phi ( e ) ~ \in ~ S$ to every edge $\begin{array} { r l r } { e } & { { } \in } & { E . } \end{array}$ The source set S contains five levels: AU-THORITATIVE, AUTHORITATIVE PHARMA, CURATED, LLM EXTRACTED, and PREDICTED. The assignment follows a two-pass procedure implemented as Cypher transactions.

Pass 1: Per-edge inheritance. Some edges imported from upstream resources carry pre-existing confidence values. For instance, HERB 2.0 provides multi-source evidence weights for molecular interaction edges, typically in the range [0.10, 0.45]. Pass 1 preserves these values by setting r.conf\_score = toFloat(r.confidence) for all edges where r.confidence IS NOT NULL, and assigns the source level from the type-level mapping as a fallback via COALESCE:

MATCH ()-[r:{rtype}]->()   
WHERE r.confidence IS NOT NULL   
SET r.conf\_score = toFloat(r.confidence),   
r.source\_level = COALESCE(r.source\_level, \$source)

Pass 2: Type-level defaults. Edges without per-edge confidence values are assigned the type-level constant from the assignment scheme in Table III:

MATCH ()-[r:{rtype}]->()   
WHERE r.conf\_score IS NULL   
SET r.conf\_score = \$conf,   
r.source\_level = COALESCE(r.source\_level, \$source)

After both passes, per-type statistics are computed to verify annotation quality: min(r.conf\_score), max(r.conf\_score), avg(r.conf\_score) for each relationship type. The procedure processes all annotated relationship types sequentially, producing 722,671 annotated edges.

AUTHORITATIVE edges (0.95) derive from pharmacopoeias [34] and experimentally supported molecular databases such as DrugBank 6.0 and TTD [35], [36]. AU-THORITATIVE PHARMA edges (0.90) cover TCM pharmacopoeia attributes including Nature, Flavor, Meridian, and Toxicity. CURATED edges (0.85) come from manually organized formulary and prescription data sources. LLM EXTRACTED edges (0.70) cover the 11 micro-semantic categories; the automated cleaning pipeline (§3.1) requires these edges to pass anchor-field validation before reaching the annotator. Although audit precision reaches 95.5%, we conservatively assign 0.70 to reflect the non-trivial hallucination risk of LLM-generated content [8]. PREDICTED edges (0.30–0.60) are reserved for KGE link prediction outputs, where model-specific confidence calibration determines the exact value.

The numeric values in Table III should be read as sourcecategory reliability scores, not calibrated probabilities of clinical truth. They encode a stable ordering among evidence sources and make that ordering available to the execution layer. Per-edge inherited scores take precedence when upstream resources provide edge-level evidence weights; type-level constants are used only for relations without such scores. This avoids overwriting heterogeneous molecular evidence with a single type default while still ensuring that every edge has a query-visible reliability annotation. The threshold analysis in Section VII evaluates how query results change across a range of θ values, rather than relying on a single calibrated cutoff.

## B. Confidence Semantics and Query Guarantees

TCMaster treats confidence as an execution-visible edge annotation, not as a hidden data-cleaning score. For an edge e, σ(e) summarizes source reliability and validation status, while ϕ(e) records the provenance category that explains why the score was assigned. A path confidence value is derived only after a query returns candidate paths; it is not stored as a global truth value for all possible paths. This distinction matters because edge confidence controls predicate pushdown, whereas path confidence controls ranking and post-filtering.

The thresholded query semantics are monotone with respect to θ: if $\theta _ { 1 } ~ \leq ~ \theta _ { 2 } ,$ every path returned under $\theta _ { 2 }$ is also a candidate under $\theta _ { 1 }$ before top-k truncation. This property gives application developers a predictable reliability knob. Raising θ cannot introduce lower-confidence edges; it can only reduce or preserve the feasible path set. For PRODUCT and MIN aggregation, the path score is also monotone with respect to each edge score, so expert feedback that increases a validated edge confidence cannot lower the confidence of any path containing that edge.

TCMaster also preserves provenance continuity: each returned path includes edge sequence, confidence scores, and source labels, letting users distinguish pharmacopoeia-only paths from mixed LLM-extracted or predicted evidence. This is operational rather than statistical; PRODUCT is a conservative path score, not a possible-world probability, while MIN provides a weakest-link policy and WAVG preserves exploratory recall. This distinction is important for reviewer interpretation: TCMaster does not claim to solve probabilistic query evaluation, but it does expose reliability metadata at the same point where practitioners inspect graph evidence. The design therefore favors transparent execution behavior over hidden calibration assumptions.

## C. Confidence Query Rewriting

Given a Cypher query Q and threshold θ, the ConfidenceQueryRewriter transparently injects confidence constraints. The rewriter first extracts all relationship variables from the query pattern using a regular expression that matches named Cypher relationships such as [r] and [r:HAS\_INGREDIENT]. For each extracted variable $r _ { i } ,$ it generates the constraint r .conf\_score ≥ θ. All constraints are conjoined with AND and injected into the query.

If the original query contains a WHERE clause, constraints are prepended to it:

MATCH (rx:Prescription)-[r1:HAS\_COMPONENT]->(h:Herb)   
-[r2:HAS\_INGREDIENT]->(i:Ingredient)   
WHERE r1.conf\_score >= 0.8 AND r2.conf\_score >= 0.8   
RETURN rx, h, i

If no WHERE clause exists, the rewriter locates the first occurrence of RETURN, WITH, ORDER, or LIMIT and inserts a WHERE clause before it:

MATCH ...   
WHERE r1.conf\_score >= 0.8   
AND r2.conf\_score >= 0.8   
RETURN ...

This rewriting is transparent and deliberately narrow: it targets Cypher templates with explicit relationship variables in MATCH clauses, preserves existing predicates by conjunction, and leaves projection, grouping, ordering, and limits unchanged. For anonymous relationships, TCMaster first names the relationships. Each injected predicate maps to one relationship variable, keeping the transformation deterministic and executable by Neo4j. We intentionally avoid semantic rewrites that change return variables or aggregation clauses; this makes the transformation auditable and allows reviewers to compare original and rewritten queries in the artifact.

## D. Path Confidence Aggregation

For multi-hop path queries, TCMaster extracts per-edge confidence scores and computes a single path-level confidence through the user-selected aggregation function. The PathRanker module appends a confidence extraction clause to the user’s Cypher:

```sql
{user_pattern}
RETURN p, [r IN relationships(p) | r.conf_score] AS scores RETURN p, [r IN relationships(p) | r.conf_score] AS scores
LIMIT 200 LIMIT 200
```

Three aggregation functions are supported. (1) PRODUCT: $\begin{array} { l l l } { { P a t h C o n f ( p ) } } & { { = } } & { { \prod _ { e \in p } \sigma ( e ) } } \end{array}$ . This penalizes long paths containing any low-confidence edge; a single LLM EXTRACTED edge at 0.70 reduces the aggregate to at most $0 . 9 5 ~ \times ~ 0 . 7 0 ~ = ~ 0 . 6 6 5$ . PRODUCT is recommended for safety-critical scenarios. (2) MIN: $\begin{array} { r l r } { P a t h C o n f ( p ) } & { { } = } & { \operatorname* { m i n } _ { e \in p } \sigma ( e ) } \end{array}$ This evaluates the path by its weakest constituent edge, requiring every edge to individually satisfy the threshold. (3) WAVG: $\begin{array} { r } { \bar { P a t h C o n f ( p ) } ~ = ~ \frac { \sum _ { i = 1 } ^ { \lceil p \rceil } w _ { i } \sigma ( e _ { i } ) } { \sum _ { i = 1 } ^ { \lfloor p \rfloor } w _ { i } } } \end{array}$ , where $w _ { i } \ = \ 1 / ( 1 + \log i )$ This weighted average gives higher weight to edges closer to the query start, providing the highest recall and supporting exploratory analysis.

Paths below θ are discarded; the rest are ranked by descending PathConf(p) and truncated to top-k (default k = 20). For m paths of maximum length ℓ, extraction and aggregation cost O(mℓ) plus O(m log m) ranking. Edge-level predicates provide early pruning, while path aggregation is the final policy check; exploratory retrieval can set the edge threshold to zero and rank only after execution.

The design is intentionally conservative: instead of full probabilistic query semantics, TCMaster separates edge thresholding before execution from path ranking after execution. The same path can be inspected under PRODUCT, MIN, or WAVG without changing the graph or rebuilding indexes. This separation also supports different application modes: safety inspection can use strict edge filtering and PRODUCT ranking, while exploratory target discovery can retain more candidates and rank them post hoc.

## E. Incremental Confidence Updates

TCMaster supports interactive confidence refinement through batch edge updates. When a domain expert validates or rejects a specific edge, the update sets both the new confidence score and marks the edge as source\_level = USER\_VERIFIED:

MATCH (a)-[r:{rel\_type}]->(b)   
WHERE id(a) = \$sid AND id(b) = \$eid   
SET r.conf\_score = \$conf, r.source\_level = ’USER\_VERIFIED’

Updates are executed as Neo4j write transactions, so a feedback batch either commits all confidence and provenance changes or leaves the previous annotation state intact. Since confidence is stored on edges, update cost depends on the number of matched relationships rather than graph size. Batch updates of 100 or more edges achieve stable per-edge latency of 4.7–5.4 ms, with 1,000-edge batches completing within 5 seconds. The update does not trigger global recomputation; only the directly modified edges are affected. For materialized shortcuts that depend on updated edges, a staleness flag is set via the check\_shortcut\_staleness method, and affected shortcuts are lazily recomputed on the next materialization cycle.

## V. WORKLOAD-GUIDED PHYSICAL DESIGN FOR ONTOLOGY-SKEWED GRAPH QUERIES

Clinical workloads in TCM are dominated by three query patterns: (i) attribute lookup—given a herb, retrieve its Nature, Flavor, Meridian, and Toxicity values; (ii) attributeconstrained search—find herbs matching specific attribute combinations (e.g., Cold and Bitter and targeting Liver); and (iii) cross-layer traversal—follow multi-hop paths such as Prescription→Herb→Ingredient→Target for drug discovery or Prescription→Herb→Warning for safety auditing. These patterns share a structural bottleneck: each involves traversing from high-cardinality entity nodes (15K+ herbs) to lowcardinality attribute nodes (5–12 values) or across three knowledge layers. General-purpose graph databases use costbased optimizers that treat all nodes as having comparable cardinalities, missing the optimization opportunities created by TCM’s extreme ontology skew. We exploit this skew through three techniques: cardinality-based direction selection (Section V-B), bitmap-based attribute pruning (Section V-C), and materialized cross-layer shortcuts (Section V-D).

## A. Cardinality Skew in TCM Ontology

TCM ontologies exhibit orders-of-magnitude cardinality skew between entity types. Attribute nodes have very small, fixed cardinalities: TCMMeridian (12 nodes), TCMProperty (5 nodes for Cold, Hot, Warm, Cool, Neutral), TCMFlavor (7 nodes for Sour, Bitter, Sweet, Pungent, Salty, Bland, Astringent), TCMToxicity (4 nodes for Non-toxic, Slightlytoxic, Toxic, Highly-toxic). In contrast, Herb nodes number 15,092, Ingredient nodes 44,595, and Target nodes 15,515. This cardinality skew—attribute nodes being three orders of magnitude smaller than entity nodes—creates three optimization opportunities that general graph engines do not exploit by default.

## B. Cardinality-Based Direction Selection

When a Cypher query constrains TCM attributes (e.g., “find herbs that are Cold in Nature and target the Liver meridian”), the default traversal from the Herb side starts from 15K candidate nodes and fans out to attribute nodes. Our optimizer detects such Herb-to-attribute patterns and reverses the traversal direction:

Original query:   
MATCH (h:Herb)-[:HAS\_MERIDIAN]->   
(m:TCMMeridian {name:’GanJing’})   
Rewritten query:   
MATCH (m:TCMMeridian {name:’GanJing’})   
<-[:HAS\_MERIDIAN]-(h:Herb)

This transformation reduces the initial search space from 15K+ Herb nodes to a single Meridian node. The optimizer applies this rewrite statically before query execution; the Cypher query planner then uses the low-cardinality starting point to bound the search. For queries involving multiple attribute constraints, all direction reversals are applied simultaneously. In our experiments, this optimization is transparent—users write queries in the natural Herb→Attribute direction, and the system automatically selects the efficient Attribute→Herb traversal.

## C. Bitmap-Based Attribute Pruning

The TCM attribute space is small and closed in the current ontology snapshot: 5 properties, 7 flavors, 12 meridians, and 4 toxicity levels, yielding 28 distinct values in total. We exploit this by precomputing a 28-bit attribute bitmap for every Herb node, encoding the presence of each attribute value as a single bit.

Bit layout. The 28-bit integer uses four contiguous blocks: Nature (bits 0–4), Flavor (5–11), Meridian (12–23), and Toxicity (24–27). The layout follows the ontology rather than observed frequency, which keeps the encoding stable across graph snapshots and makes each bit interpretable during debugging. If later releases add controlled values, the representation can allocate additional bits or a second integer word without changing query semantics.

Encoding. For each Herb node h, B(h) = $\mathsf { V } _ { a t t r \in A t t r s ( h ) } ( 1 \ll { i n d e x } ( a t t r ) )$ . The bitmap is computed once during ingestion, permits multiple values per dimension, and is stored directly on Herb nodes. In the current snapshot, 11,014 Herb nodes receive at least one bitmap property, and the builder performs 31,426 herb-dimension bitmap updates across the four attribute dimensions. The non-mutualexclusive design matters for TCM: a herb may have several flavors or meridians, so a categorical one-hot representation would either lose information or require repeated property values. The bitmap keeps multi-label attributes compact while leaving the original graph edges intact for provenance and explanation.

Query-time pruning. A query requiring a 2-hop traversal (Herb→Meridian→filter by name) is reduced to a 1-hop lookup plus a constant-time bitwise AND:

$$
\mathtt { W H E R E } \mathtt { h . m e r i d i a n \_ b i t m a p } \mathtt { d . ( 1 \_ < < \mathtt { k } ) } > 0
$$

where k is the target meridian index. For multi-attribute queries, several such constant-time checks replace repeated 2- hop traversals. This does not make bitmap filtering universally faster than graph traversal; rather, it creates a cheap local test once a candidate herb set has already been produced. Section VII therefore reports both useful and limited cases instead of presenting bitmaps as a general-purpose index.

## D. Materialized Cross-Layer Shortcuts

The most frequent clinical query pattern traverses three knowledge layers: Prescription → Herb → Ingredient → Target. This 3-hop pattern is evaluated repeatedly in clinical QA and drug discovery workloads. TCMaster materializes this pattern as a direct shortcut edge REACHES\_TARGET Prescription → Target.

Construction. Shortcuts are created by batched Cypher processing over 200 prescriptions per batch: expand the 3-hop path, deduplicate at the (Prescription, Target) level, skip existing shortcuts, and create REACHES\_TARGET with metadata. Each shortcut stores conf\_score=0.8075, the PRODUCT of CURATED and two AUTHORITATIVE edges, and source\_level=MATERIALIZED. The edge is not a new biomedical assertion; it is a physical design artifact summarizing reachability through existing evidence paths. We keep the source level explicit so applications can distinguish materialized reachability from base pharmacological facts.

Maintenance. Shortcuts are timestamped, lazily invalidated after constituent-edge updates, and recomputed in the next materialization cycle. This matches snapshot-oriented TCM sources; high-update deployments should rebuild shortcuts offline or disable them. Creating 28.75M shortcut edges takes about 74 minutes for the complete graph; the longer target-reachability stage also includes validation, indexing, and export checks. This cost is acceptable for offline release preparation but not for per-query construction, which is why TCMaster treats shortcuts as a workload-guided physical structure rather than a dynamic query rewrite alone.

## E. Rewrite Applicability and Cost Intuition

The optimizer is rule-based, but the rules are not applied blindly. Each rewrite is tied to a structural precondition and a cost intuition. Direction reversal is applicable when a query binds one or more low-cardinality ontology values and asks for matching high-cardinality entities. Its benefit comes from reducing the starting frontier before expansion. It is therefore useful for Herb-Attribute and Prescription-Herb-Attribute workloads, but it provides little benefit when the query already starts from a selective entity identifier.

Bitmap pruning applies to the closed TCM attribute vocabulary, replacing repeated edge expansion with integer predicates. It helps multi-attribute filtering and repeated screening, but is less useful for single top-k lookups where Neo4j already exploits selective node access. This boundary is important: a property predicate can reduce arithmetic cost while also changing the planner’s available access path. TCMaster therefore enables bitmap predicates only when the query shape suggests repeated local attribute checks, not when a selective ontology node is already available as the starting point.

Shortcut materialization applies when a long traversal is reused as a reachability primitive. TCMaster materializes Prescription→Target paths for drug discovery, clinical QA, and target-count aggregation; the benefit is largest for high-fanout counting and grouping. The cost is storage and maintenance for 28.75M shortcut edges, so infrequent exploratory traversals remain on the native graph engine. In this sense, the optimizer is closer to a conservative physical-design policy than a universal graph query optimizer: it exposes a small number of domain-stable structures and uses them only when the workload makes the trade-off favorable.

## VI. IMPLEMENTATION

TCMaster is implemented in Python 3.9 with approximately 12K lines of code across data pipeline, confidence annotation, query processing, and optimization modules. TCMaster-KG runs on Neo4j 4.4 Community Edition and stores three added structures: per-edge confidence/provenance properties, 28-bit Herb attribute bitmaps, and 28.75M materialized REACHES TARGET shortcuts. Neo4j is configured with 8 GB heap and 16 GB page cache on NVMe storage.

System organization. The implementation follows four modules that match the logical architecture in Fig. 1. The Extract-Transform-Load (ETL) module normalizes heterogeneous source files into typed entity and relation tables. The confidence module annotates each edge, writes provenance properties, and exports per-relation statistics for audit. The query module rewrites Cypher templates and ranks paths. The optimization module builds bitmap and shortcut structures offline, then applies rewrite rules only when a query matches a supported pattern.

Query pipeline. TCMaster processes each query in three stages. The ConfidenceQueryRewriter names relationship variables and injects r.conf\_score >= θ predicates. The OntologyOptimizer applies direction reversal, bitmap filtering, and shortcut substitution when the query pattern matches the corresponding workload. Neo4j executes the rewritten query, and the PathRanker computes aggregate path confidence, filters by θ, and returns top-k paths.

Extraction and validation. LLM extraction uses DeepSeek-V3 through SiliconFlow with batched processing, retries, and deterministic MD5-based subsampling (seed 42). The audit model checks triples against source text using temperature 0.1 and JSON output. KGE validation uses PyKEEN 1.10/PyTorch 2.0, six embedding models over S1–S4, 3 seeds, dimension 200, and filtered rank-based evaluation; Relational Graph Convolutional Network (R-GCN) is an auxiliary baseline. We report KGE as structural validation rather than as a deployed inference module, because link prediction scores are sensitive to relation heterogeneity and negative sampling choices.

Benchmarking and ETL. Query latency is measured on Intel Xeon Gold 6342, 256 GB RAM, and NVMe SSD with 100 sampled herbs/prescriptions, 3 warmup iterations, and 10 measured iterations per sample. The full ETL pipeline takes about 36 hours: 8 h ingestion, 2 h cleaning, 10 h confidence annotation, 12 h target-reachability preprocessing, and 4 h bitmap construction. These construction costs are separated from online latency measurements. The artifact contains scripts, templates, configurations, results, and reproduction instructions so reviewers can inspect short-running verification scripts without rebuilding the full KG.

Reproducibility controls. All experiments use fixed random seeds, pinned query templates, and materialized result files. Expensive construction steps, including shortcut generation and KGE training, are separated from short verification scripts so reviewers can inspect reported tables without rerunning the full pipeline. For large KG snapshots, the artifact records schema definitions, loader commands, checksum files, and sampled data slices, while the public release will provide downloadable snapshots for full reconstruction.

## VII. EXPERIMENTAL EVALUATION

We organize the evaluation around five research questions, each tied to a database-system claim. RQ1 evaluates graphquery latency under workload-guided rewrites: when do direction selection, bitmap predicates, and shortcut substitution reduce execution time relative to Neo4j’s native planner? RQ2 evaluates execution behavior under sampled KG scale-up on the core workload layers. RQ3 evaluates whether extraction, cleaning, and confidence annotation produce auditable graph facts with explicit provenance. RQ4 treats link prediction as an auxiliary KG structural diagnostic rather than as the main system contribution. RQ5 evaluates downstream retrieval as an application-level validation of the graph substrate, not as evidence for a new clinical model.

## A. Experimental Setup

TCMaster-KG contains 221,225 entities, 722,671 base edges, and 28.75M materialized shortcut edges. KGE ablations use S1–S4 layer settings, six embedding baselines (TransE, RotatE, ComplEx, DistMult, TuckER, MuRE), and auxiliary R-GCN where available, with dimension 200 and seeds 42, 123, and 456. Query benchmarks run on Neo4j 4.4 with 8 GB heap. Each workload samples 100 start entities and repeats three runs after warmup. Unless stated otherwise, Base is Neo4j 4.4’s native planner on the same graph, constraints, and indexes, with TCMaster rewrites, bitmaps, and shortcut substitution disabled.

## B. Workload Characterization

The benchmark suite is organized by operator stress rather than hop count alone: attribute-cardinality skew, cross-layer traversal, grouping, high-fanout matching, and safety audit. Table IV summarizes the workloads.

This workload design is intentionally mixed. Simple 1- hop and 2-hop queries test whether confidence predicates and ontology rewrites add overhead to already selective traversals. Cross-layer 3-hop and 4-hop queries test the setting where provenance and path confidence matter most, because each answer may combine prescription, ingredient, target, disease, and predicted evidence. Aggregation and high-fanout workloads test whether the physical structures are useful beyond top-k lookup. The combination prevents the evaluation from reporting only favorable cases: the same benchmark suite includes workloads where TCMaster should help and workloads where Neo4j’s native planner is already competitive.

indexes, with the corresponding TCMaster transformation disabled. High-fanout workloads use 20 prescriptions with 5K–8K reachable targets and 5 repeated runs.

TABLE IV  
CHARACTERIZATION OF THE QUERY WORKLOAD. RX AND INGR. DENOTE PRESCRIPTION AND INGREDIENT.
<table><tr><td>Query Pattern</td><td></td><td>Operator stress Avg. out</td><td></td></tr><tr><td>Q1</td><td>Herb-Attr</td><td>1-hop lookup</td><td>3.03</td></tr><tr><td>Q2</td><td>Rx-Herb-Attr</td><td>2-hop join</td><td>5.27</td></tr><tr><td>Q3</td><td>Rx-Herb-Ingr.-Target</td><td>shortcut path</td><td>91.10</td></tr><tr><td>Q4</td><td>Rx-Herb-Ingr.-Disease</td><td>long traversal</td><td>91.02</td></tr><tr><td>Q5</td><td>Rx-Herb-Meridian</td><td>aggregation</td><td>6.28</td></tr><tr><td>Q6</td><td>Rx-Herb-Rx</td><td>pattern match</td><td>22.00</td></tr><tr><td>Q7</td><td>Rx-Herb-Safety</td><td>safety audit</td><td>5.27</td></tr></table>

## C. Query Performance

Table V summarizes latency for seven representative clinical workloads.

TABLE V  
QUERY LATENCY FOR REPRESENTATIVE CLINICAL WORKLOADS
<table><tr><td>Query</td><td>Hops</td><td>Mean (ms)</td><td>P95 (ms)</td></tr><tr><td>Q1: Herb attribute lookup</td><td>1</td><td>8.41</td><td>9.64</td></tr><tr><td>Q2: Prescription property</td><td>2</td><td>5.23</td><td>6.49</td></tr><tr><td>Q3: Prescription→Target</td><td>3</td><td>6.70</td><td>8.89</td></tr><tr><td>Q4: Disease association</td><td>4</td><td>7.12</td><td>9.77</td></tr><tr><td>Q5: Aggregation</td><td>2</td><td>5.27</td><td>6.54</td></tr><tr><td>Q6: Pattern matching</td><td>var.</td><td>12.04</td><td>17.52</td></tr><tr><td>Q7: Safety audit</td><td>2</td><td>5.40</td><td>6.56</td></tr></table>

Key findings: (1) Latency is not monotonic with hop count: the 4-hop Q4 (7.12 ms) is faster than the 1-hop Q1 (8.41 ms), because latency is dominated by index lookup and intermediate cardinality rather than path length alone. (2) Cold/warm cache comparison for Q3 shows negligible difference (6.74 ms vs. 6.73 ms), indicating the graph is effectively memory-resident. (3) Q6 (pattern matching with high fanout) shows the highest latency, identifying this as the remaining bottleneck.

The P95 values refine the same conclusion. Six of seven workloads remain below 10 ms at P95, while Q6 reaches 17.52 ms because variable-length pattern matching must explore a broader candidate frontier before returning bounded results. This spread matters for interpreting TCMaster as a query substrate: the system is interactive for the tested clinical lookup, safety, and target-discovery templates, but the latency profile is still workload dependent. The results also show why we report representative templates rather than only aggregate averages. Q2, Q5, and Q7 have similar mean latency despite different semantics because their traversals are anchored by selective prescription or safety nodes; Q1 is slower because attribute lookup can start from higher-degree Herb neighborhoods unless direction selection is applied. Thus the queryperformance result supports a bounded claim: confidenceaware graph retrieval can be served at interactive latency for the evaluated workloads, while high-fanout pattern matching remains the main optimization target.

TABLE VI  
WORKLOAD-GUIDED PHYSICAL-DESIGN RESULTS
<table><tr><td>Workload</td><td>Base</td><td></td><td>Opt. Speedup</td><td>Avg. out</td></tr><tr><td>Direction selection</td><td>9.40</td><td>6.41</td><td>1.47×</td><td>100</td></tr><tr><td>Shortcut top-k lookup</td><td>13.62</td><td>11.64</td><td>1.17×</td><td>100</td></tr><tr><td>Bitmap single attribute</td><td>15.99</td><td>15.53</td><td>1.03×</td><td>200</td></tr><tr><td>Combined top-k lookup</td><td>10.73</td><td>16.45</td><td>0.65×</td><td>100</td></tr><tr><td>High-fanout target count</td><td>51.19</td><td>11.57</td><td>4.42×</td><td>6555</td></tr><tr><td>High-fanout target enum.</td><td>336.08</td><td>283.43</td><td>1.19×</td><td>6555</td></tr></table>

## D. Optimization Evaluation

We evaluate the three workload-guided physical-design choices on two classes of graph-query workloads. The first class mirrors interactive top-k lookup, where Neo4j can often terminate early after finding 100 results. The second class targets high-fanout complete expansion and aggregation, where the execution layer must avoid repeatedly materializing large intermediate paths. Table VI reports the measured results.

Direction selection. Reversing traversal from Herb→Attribute to Attribute→Herb improves property lookup from 9.40 ms to 6.41 ms (1.47×), confirming that low-cardinality ontology nodes can bound the search before expansion to Herb nodes.

Materialized shortcuts. For top-k target lookup, shortcuts improve latency only modestly (13.62 ms to 11.64 ms, 1.17×) because both plans stop after 100 targets. The benefit is larger for complete high-fanout expansion: target counting over prescriptions with 5K–8K reachable targets improves from 51.19 ms to 11.57 ms (4.42×), while complete enumeration improves from 336.08 ms to 283.43 ms (1.19×).

Bitmap pruning. Single-attribute bitmap filtering provides little benefit over Neo4j’s native traversal (1.03×), and multiattribute probes are mixed because scanning Herb nodes with bitmap predicates can be more expensive than starting from selective attribute nodes. We therefore treat bitmaps as a compact representation for repeated local checks rather than the main accelerator.

Combined optimization. Shortcut lookup plus a confidence predicate is slower than unoptimized top-k traversal (0.65×) because REACHES\_TARGET edges already carry high confidence, so the predicate adds cost without pruning many edges. TCMaster therefore applies rewrites selectively instead of assuming that transformations compose beneficially.

Overall, the optimizer results support a narrower but defensible claim: TCM ontology skew can be exploited effectively when the workload exposes the corresponding bottleneck. Direction selection benefits low-cardinality attribute lookup, and materialized shortcuts substantially improve high-fanout complete expansion. For top-k lookup workloads already optimized by Neo4j, the gains are limited.

Selective rewrite policy. TCMaster applies transformations only when their workload assumptions hold: direction reversal for Herb–attribute patterns with ontology cardinality below 32, shortcut substitution for aggregate counts or complete enumeration over Prescription→Herb→Ingredient→Target paths, bitmap predicates for repeated local herb-attribute checks, and no shortcut-plus-confidence composition for already highconfidence shortcuts unless pruning is expected. This is not a full cost model, but an auditable policy derived from the ablation; the artifact includes Base/Opt. query templates.

Two design implications follow from the ablation. First, materialization should be justified by output cardinality, not merely by path length. The shortcut path is only modestly faster for top-k lookup because both plans can stop early, but it is much faster when the query must count all reachable targets. Second, reliability predicates are not free: when a shortcut relation already has nearly uniform high confidence, adding a threshold predicate can reduce planner flexibility without removing many edges. These observations motivate keeping the rewrite policy explicit and auditable rather than hiding it behind an opaque cost heuristic.

## E. Core-Layer Scalability

We evaluate query latency at four scales of the core L1–L3 subgraph (25%, 50%, 75%, 100%) by sampling herb subsets and re-importing subgraphs into a clean Neo4j instance. Results are shown in Table VII.

TABLE VII  
QUERY LATENCY (MS) VS. CORE-LAYER GRAPH SCALE
<table><tr><td>Scale</td><td>Nodes</td><td>Edges</td><td>Q1</td><td>Q3</td><td>Q5</td></tr><tr><td>25%</td><td>26,575</td><td>55,263</td><td>7.98</td><td>6.50</td><td>11.73</td></tr><tr><td>50%</td><td>41,755</td><td>113,000</td><td>11.67</td><td>6.43</td><td>10.84</td></tr><tr><td>75%</td><td>53,698</td><td>165,745</td><td>8.08</td><td>2.83</td><td>5.59</td></tr><tr><td>100%</td><td>64,254</td><td>223,696</td><td>6.32</td><td>1.99</td><td>5.80</td></tr></table>

In the controlled L1–L3 scale benchmark, latency remains bounded but not monotonic. Q3 decreases from 6.50 ms at 25% to 1.99 ms at 100%, consistent with favorable index utilization in the sampled subgraphs. The temporary Q1 increase at 50% (11.67 ms vs. 7.98 ms at 25%) reflects over-sampling herbs with dense attribute out-degree. This is a stability check for latency-critical paths, not a claim about arbitrary graph scaling; extending controlled scale-up to L4–L5 remains future work.

The non-monotonicity is worth reporting because it prevents a misleading scalability narrative. In Neo4j, sampled subgraphs can change both data volume and selectivity; a larger sample may expose more index-friendly anchors or more selective attribute values. We therefore use this experiment as a bounded robustness check for the core query layer, while the full 221K-node graph remains the primary evaluation setting for reported workload latency.

## F. Data Quality and Extraction

Pipeline effectiveness. Batch parallelization reduces cleaning time by about 60%. After cleaning, anchor-field completeness increases from 0.72 to 0.91, LLM extraction precision improves from 91.2% to 95.5%, and hallucination drops from 8.3% to 4.5% (Fig. 3).

![](images/cc857dde557c0afaf5696b89761fae87be22c9a81496e850cc65b6ba9ef68ee2.jpg)

![](images/cab8b84fc2b90d383cfa6b9bf358f38fca14f34df3af69a992e3478e8baa1500.jpg)  
Fig. 2. Core-layer scalability over 25–100% sampled L1–L3 graph scales, showing bounded latency variation for Q1, Q3, and Q5.

Cross-source validation. Validation against external references yields: herb-ingredient fuzzy precision 93.7%, ingredient-target precision 100.0%, and LLM micro-semantics audited precision 95.5%. Prescription composition recall is 76.0% (F1 = 72.4%), reflecting reference formulary incompleteness relative to our multi-source coverage.

These results support two separate claims. First, the graph is clean enough to serve as a retrieval substrate: the most latency-critical molecular and micro-semantic relations pass high-precision validation. Second, recall is not uniformly high because TCM formulary sources disagree in naming, granularity, and composition variants. TCMaster therefore preserves provenance and confidence rather than forcing all sources into a single unqualified truth table.

The cleaning results also clarify the role of LLM extraction. We use LLMs to recover micro-semantic relations that are absent from structured databases, but the graph never treats these triples as first-class authoritative facts without checks. Anchor-field validation removes triples that cannot be linked back to core entities; audit scoring estimates whether the extracted relation is supported by the source text; and the confidence model keeps LLM EXTRACTED edges below curated or pharmacopoeia-derived relations. This pipeline turns LLM extraction into a controlled data-ingestion channel rather than an unverified data source.

## G. Structural Validation via Link Prediction

We use link prediction as a structural validation probe rather than as a primary system contribution. Figure 4 reports incremental settings S1–S4 using mean reciprocal rank (MRR). S1 (molecular only) MRR = 0.185; S2 (+TCM attributes) MRR = 0.157; S3 (+prescriptions) MRR = 0.131; S4 (+LLM micro-KG) MRR = 0.136.

Ablation interpretation. Adding TCM attributes lowers RotatE MRR from 0.185 to 0.157 because many herbs map to a few controlled attribute nodes, although MuRE improves from 0.142 to 0.149 (+8.0%), consistent with the layer’s tree-like topology. Adding prescriptions further increases candidate-space and relation heterogeneity, reducing RotatE MRR to 0.131. Adding LLM micro-semantics recovers a modest RotatE gain to 0.136 (+3.7%). We treat KGE as a diagnostic stress test; extraction audit and cross-source validation remain the primary quality evidence.

![](images/ec39c2e043521b62c6ce8d61551282eee11a4005a1e91be4abe9f0086be9be22.jpg)

![](images/755331cac55189ae3f4a96865d7a0891b36d5cefad315dda239dfc15f14eacb1.jpg)

![](images/4e793fffc31aa8aaa5ff4adf18f5b5f6585375a88faeae0c1c360b8af0650c67.jpg)  
Fig. 3. Data quality and cleaning effectiveness by validation layer and before/after cleaning metric.  
Fig. 4. KGE structural-validation ablation across S1–S4, with MRR baselines and RotatE Hits@1/3/10.

The main lesson is that structural predictability is not monotonic with graph size. Adding useful domain layers can make the prediction task harder because the candidate space expands and relation semantics diversify. We therefore do not interpret lower MRR as evidence that a layer is harmful. Instead, KGE complements the query benchmarks by showing how each layer changes relational geometry under standard embedding objectives.

## H. Downstream Clinical QA (TCMbench)

We evaluate a downstream KG-RAG application built on TCMaster using TCMbench v3 (2,000 clinical questions across Clinical Prescription, Prescription Logic, and Safety Audit dimensions) and DeepSeek-V3 as the reasoning LLM. TCMbench v3 is our clinical evaluation split built from the submitted artifact’s benchmark data, while external TCM QA benchmarks are cited for comparison context [30], [33]. Results are shown in Table VIII.

TABLE VIII  
TCMBENCH ACCURACY BY MODE AND DIMENSION
<table><tr><td>Dimension</td><td>Baseline</td><td>Vector-RAG</td><td>KG-RAG</td></tr><tr><td>Clinical Prescription</td><td>17.9%</td><td>14.1%</td><td>45.3%</td></tr><tr><td>Prescription Logic</td><td>99.7%</td><td>99.4%</td><td>100.0%</td></tr><tr><td>Safety Audit</td><td>40.2%</td><td>44.1%</td><td>72.2%</td></tr><tr><td>Overall</td><td>52.5%</td><td>52.5%</td><td>72.5%</td></tr></table>

KG-RAG improves overall accuracy by 20.0 pp over Baseline and Vector-RAG (72.5% vs. 52.5%); Wilson intervals are 70.5–74.4% for KG-RAG and 50.3–54.7% for the baseline. Clinical Prescription improves by 27.4 pp and Safety Audit by 32.0 pp, while Vector-RAG underperforms Baseline on Clinical Prescription (14.1% vs. 17.9%) due to context dilution. This is downstream validation of structured KG retrieval, not a new LLM architecture or a direct causal test of the confidence model. We use full-recall retrieval (θ = 0); adaptive confidence thresholds remain future work.

The dimension-level results are consistent with the graph structure. Prescription Logic is already near saturated for the baseline, leaving little headroom. Clinical Prescription and Safety Audit require retrieving specific herbs, properties, warnings, and toxicity nodes, where structured paths are more reliable than unstructured passages. The Vector-RAG failure case reinforces the same point: text similarity can retrieve plausible but incomplete passages, while KG retrieval constrains context to typed relations.

We deliberately keep this experiment downstream rather than central. KG-RAG accuracy is not used to tune the optimizer, and the retrieval setting uses full recall with θ = 0 so that the result does not conflate confidence filtering with answer generation. Its purpose is to show that the query substrate exposes useful structured context to an application layer. The stronger system claim remains the database claim evaluated in the query benchmarks: confidence and provenance are available at query time, and workload-specific physical structures change latency only when the workload matches their assumptions.

## I. Confidence Query Analysis

Threshold filtering effect. EXP-11a evaluates PRODUCT filtering over five workloads and thresholds 0.10–0.45. Filtering is selective: Q1, Q2, Q5, and Q7 retain all results because they traverse mostly AUTHORITATIVE, AUTHORI-TATIVE PHARMA, or CURATED relations, while Q3 mixes curated, molecular, and predicted evidence. At $\theta \ = \ 0 . 1 8 ,$ Q3 filters 39.3%; at $\theta \ : = \ : 0 . 3 0$ , it filters 69.5%. Thresholds above 0.35 remove all Q3 paths, useful for conservative safety filtering but too strict for exploratory target discovery.

![](images/a4588503078ff13ff09d29dd0f2abd02d5cc58e01caeb718430b92491b444685.jpg)

![](images/1d52aa6d976e567bd5cb442855f58bd2594523f1220b08835fb18b8851a60bcb.jpg)  
Fig. 5. Downstream TCMbench validation: accuracy by mode and KG-RAG gains by dimension.

TABLE IX  
FILTERING RATIO (%) UNDER CONFIDENCE THRESHOLDS
<table><tr><td colspan="5">Query  $\theta = 0 . 1 0$   $\theta = 0 . 1 8$   $\theta = 0 . 3 0$   $\theta = 0 . 3 5$   $\theta = 0 . 4 5$ </td></tr><tr><td>Q1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Q2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0 0.0 0.0</td></tr><tr><td>Q3</td><td>0.0</td><td>39.3</td><td>69.5</td><td>100.0 100.0</td></tr><tr><td>Q5</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0 0.0</td></tr><tr><td>Q7</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0 0.0</td></tr></table>

This behavior is desirable for interactive analysis. Raising θ does not uniformly shrink every query; it primarily affects paths that combine heterogeneous evidence. In practice, a curator can keep authoritative attribute and prescription queries unchanged while tightening mixed molecular paths. The threshold is therefore a workload-specific reliability knob rather than a global quality switch.

![](images/b01a63501d63f3904dff406d22b64491d162316ccf150ee054b760204fa2416f.jpg)

![](images/b3adfefb8f5d6eb5cd7c08951eb8ce67f2833214f4f9751eec1e63aa40c764ad.jpg)  
Fig. 6. Effect of confidence threshold θ on filtering ratio and recall/precision trade-off.

Aggregation strategy comparison. EXP-11b fixes the candidate paths and changes only aggregation. PRODUCT, MIN, and WAVG inspect the same 184.5 average paths but yield mean confidences of 0.0248, 0.1347, and 0.3786, respectively. Thus aggregation is a semantic choice: PRODUCT is conservative, WAVG preserves exploratory recall, and MIN reflects the weakest edge.

The large gap between PRODUCT and WAVG also explains why TCMaster exposes the policy instead of hard-coding one score. Long biomedical paths often combine curated prescription edges with molecular evidence; PRODUCT penalizes such chains strongly, which is appropriate for safety inspection but may suppress useful candidates in discovery tasks. WAVG and MIN let applications choose a less conservative interpretation while keeping provenance visible.

Incremental feedback cost. EXP-11c measures expert confidence updates. A 10-edge batch costs 28.78 ms per edge, while 100-, 500-, and 1,000-edge batches amortize to 5.37, 4.67, and 4.83 ms per edge, supporting small-batch review and larger background curation without full-graph recomputation.

## J. Case Study: Prescription Evidence Inspection

We use Ephedra Decoction (Ma Huang Tang) to illustrate evidence inspection, not clinical deployment. Given the query “What are the ingredients, natures/flavors, and toxicity warnings?”, TCMaster traverses Prescription→Herb, Herb→Nature/Flavor, and Herb→Toxicity/Warning paths. With PRODUCT confidence ≥ 0.7, it returns facts such as Ephedra having nature Hot and flavor Pungent (0.90, AUTHORITATIVE PHARMA), an Ephedra toxicity warning (0.85, CURATED), and full path confidence 0.72 for Ephedra Decoction→Ephedra→Hot. The query completes in 6.7 ms and supports expert evidence triage without replacing clinical judgment.

This example shows the interaction among the paper’s three mechanisms. Confidence filtering controls which paths are shown, provenance labels explain why they are trusted, and physical design keeps the multi-hop inspection interactive. The same workflow can be applied to other prescriptions by changing the start node, while the returned paths remain auditable because they include edge-level source labels.

## K. Discussion and Threats to Validity

TCMaster’s gains concentrate where its assumptions hold: confidence heterogeneity, ontology skew, and repeated cross-layer traversal. Shortcuts help Q3/EXP12 because Prescription→Herb→Ingredient→Target paths create large frontiers; direction reversal helps when ontology nodes are selective; PRODUCT filtering affects heterogeneous paths most strongly. Negative results are also informative: native top-k lookup, bitmap predicates, and shortcut-plus-confidence composition do not always beat Neo4j, so TCMaster reports workload-dependent rewrites rather than a universal graph optimizer.

This is the intended system boundary. The contribution is not that every TCM query becomes faster, but that reliability and stable ontology structure can be surfaced to query execution in a controlled way. When a workload lacks those properties, the native graph engine remains competitive and TCMaster should avoid unnecessary rewrites.

Threats to validity are fourfold. Latency is measured on Neo4j 4.4 and one hardware setting; other engines may differ. Query templates reflect our clinical workloads, not all TCM applications. Confidence values are source-category scores rather than learned probabilities. Safety Audit accuracy is insufficient for autonomous clinical use. We mitigate these threats by reporting workload structure, separating techniques, fixing templates and seeds, and releasing scripts and results; the external-validity claim is limited to domains with small controlled ontologies linked to high-cardinality entities.

A further limitation is that the current system uses rulebased rewrite selection. This choice is deliberate for the first artifact because each rule can be inspected and connected to a measured workload. A learned or fully cost-based selector would be a natural next step, but would require a larger training set of query templates and cardinality observations. The current evaluation therefore emphasizes explainable rewrite applicability: each optimization has a stated structural precondition, a measured positive case, and at least one boundary case where it should not be applied.

## VIII. RELATED WORK

TCM resources such as TCMSP [1], TCMID [2]/TCMID 2.0 [6], SymMap [3], HERB 2.0 [4], and ETCM [5] provide molecular, syndrome/symptom, and prescription data, while surveys note gaps in normalization and reliability annotation [19]. Biomedical KGs such as Hetionet and PrimeKG support drug-repurposing integration but do not model TCM attributes, prescriptions, or provenance-aware clinical query processing [40], [41]. TCMaster instead targets confidence-annotated path querying and ontology-aware execution over a multi-source TCM KG.

Data quality, provenance, and uncertainty. ActiveClean, HoloClean, and KATARA study cleaning, repair, and validation using model feedback, probabilistic inference, or external knowledge [12]–[14]. Probabilistic databases, ULDBs, and provenance semirings provide uncertainty and lineage semantics [17], [18], [37]. TCMaster takes an operational propertygraph point in this space: deterministic cleaning before ingestion, source-level confidence on edges, confidence predicates pushed into Cypher, and path ranking by aggregate confidence for low-latency traversal.

Graph query processing and optimization. Prior work studies graph languages, Cypher, RDF indexing, join ordering, and worst-case optimal joins [9]–[11], [25], [26], [38], [39]. Beyond labels, predicates, indexes, and join patterns, TC-Master exploits stable domain-structural skew between small ontology attributes and high-cardinality herbs, prescriptions, ingredients, and targets. Direction reversal, compact attribute bitmaps, and materialized cross-layer shortcuts exploit this skew when it matches the workload; native top-k lookup and shortcut-plus-confidence predicates show why rewrites must remain selective.

Knowledge graph embeddings and structural validation. Models such as TransE, RotatE, ComplEx, DistMult, TuckER, MuRE, and R-GCN are widely used for graph representation learning [7], [16], [20]–[24], [31]. TCMaster uses them only as diagnostic probes: the S1–S4 ablation measures how prescriptions, ontology attributes, and micro-semantics change structural predictability. Since MRR depends on candidate space, relation heterogeneity, and negative sampling, these results complement extraction audits and query benchmarks rather than proving data quality alone.

TCM retrieval and KG-RAG. Retrieval-augmented generation and GraphRAG motivate structured retrieval for knowledge-intensive QA [15], [32]; recent TCM work studies LLM-assisted diagnosis, prescription generation, KGenhanced retrieval, and benchmark construction [27]–[30], [33]. OpenTCM [29] is closest in application motivation, but evaluates an LLM-facing pipeline. TCMaster focuses on the data-management substrate: confidence/provenance edge properties, confidence-bounded traversal, and a separate evaluation of graph-query workloads and downstream KG-RAG accuracy.

## IX. CONCLUSION AND LIMITATIONS

We presented TCMaster, a confidence-aware graph query substrate for multi-source TCM knowledge graphs. By annotating 723K base edges with source-level confidence scores and exploiting ontology skew through direction selection, compact attribute bitmaps, and materialized shortcuts, TCMaster supports millisecond-level multi-hop evidence retrieval. Direction selection improves attribute lookup by a factor of 1.47, materialized shortcuts accelerate high-fanout target counting by a factor of 4.42, and downstream KG-RAG improves clinical QA by 20.0 pp.

Limitations. TCMaster currently uses fixed source-category confidence scores; learning dynamic confidences from user feedback remains future work. The scalability benchmark covers L1–L3, which contain the latency-critical paths used by our workloads, but extending controlled scale-up to L4–L5 would give broader evidence. KG-RAG uses fixed top-3 fullrecall retrieval without confidence thresholding, and the case study covers one prescription. Safety-oriented outputs are evidence inspection aids for expert review, not deployable clinical safety decisions. Finally, bitmap filtering and shortcut-plusconfidence predicates do not consistently outperform Neo4j on top-k lookup, motivating future cost-based rewrite selection. These limitations do not change the main claim: confidence and ontology structure can be first-class execution signals for TCM KGs, but rewrite policy depends on workload shape and evidence requirements.

Future work includes dynamic confidence learning, adaptive KG-RAG retrieval, and generalizing ontology-skew optimization to domains with similar hierarchical structure.

## ARTIFACT AND DATA AVAILABILITY

For review, the supplemental artifact is provided as a separate open repository/archive URL with source code, experiment scripts, query templates, configurations, raw/processed results, and a README mapping reported tables to reproduction commands. To avoid requiring reviewers to rerun the 36- hour ETL pipeline, it includes a Neo4j 4.4 dump of the materialized graph, including 28.75M REACHES\_TARGET edges, plus load instructions, validation queries, schema documentation, checksums, and lightweight verification scripts. After publication, the code will be released on GitHub and archived with a persistent snapshot. TCMaster-KG will be available through an open-access browsing website and downloadable Neo4j/CSV/JSONL snapshots with schema documentation and a data card; bulk reuse should use snapshots rather than crawling.

## AI-GENERATED CONTENT ACKNOWLEDGEMENT

This work uses DeepSeek-V3 for LLM-based microsemantic extraction and audit scoring as described in Section III-A. The authors also used Claude (Anthropic) for data processing, experiment scripting, and manuscript editing. All AI-generated content, extracted triples, scripts, and manuscript edits were reviewed and verified by the authors, who remain responsible for the correctness and originality of all content.

## REFERENCES

[1] J. Ru et al., “TCMSP: a database of systems pharmacology for drug discovery from herbal medicines,” J. Cheminform., vol. 6, p. 13, 2014.

[2] R. Xue et al., “TCMID: Traditional Chinese Medicine integrative database for herb molecular mechanism analysis,” Nucleic Acids Res., vol. 41, pp. D1089–D1095, 2013.

[3] Y. Wu et al., “SymMap: an integrative database of traditional Chinese medicine enhanced by symptom mapping,” Nucleic Acids Res., vol. 47, pp. D1110–D1117, 2019.

[4] K. Gao et al., “HERB 2.0: an updated database integrating clinical and experimental evidence for traditional Chinese medicine,” Nucleic Acids Res., vol. 53, no. D1, pp. D1404–D1414, 2025.

[5] H.-Y. Xu et al., “ETCM: an encyclopaedia of traditional Chinese medicine,” Nucleic Acids Res., vol. 47, pp. D976–D982, 2019.

[6] L. Huang et al., “TCMID 2.0: a comprehensive resource for TCM,” Nucleic Acids Res., vol. 46, no. D1, pp. D1117–D1120, 2018.

[7] A. Bordes et al., “Translating embeddings for modeling multi-relational data,” in Proc. NeurIPS, 2013, pp. 2787–2795.

[8] Z. Ji et al., “Survey of hallucination in natural language generation,” ACM Comput. Surv., vol. 55, no. 12, pp. 1–38, 2023.

[9] R. Angles et al., “Foundations of modern query languages for graph databases,” ACM Comput. Surv., vol. 50, no. 5, pp. 1–40, 2018.

[10] A. Mhedhbi and S. Salihoglu, “Optimizing subgraph queries by combining binary and worst-case optimal joins,” Proc. VLDB Endow., vol. 12, no. 11, pp. 1692–1704, 2019.

[11] N. Francis et al., “Cypher: An evolving query language for property graphs,” in Proc. SIGMOD, 2018, pp. 1433–1445.

[12] S. Krishnan et al., “ActiveClean: Interactive data cleaning for statistical modeling,” Proc. VLDB Endow., vol. 9, no. 12, pp. 948–959, 2016.

[13] T. Rekatsinas et al., “HoloClean: Holistic data repairs with probabilistic inference,” Proc. VLDB Endow., vol. 10, no. 11, pp. 1190–1201, 2017.

[14] X. Chu, J. Morcos, I. F. Ilyas, M. Ouzzani, P. Papotti, N. Tang, and Y. Ye, “KATARA: A data cleaning system powered by knowledge bases and crowdsourcing,” in Proc. SIGMOD, 2015, pp. 1247–1261.

[15] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Proc. NeurIPS, 2020, pp. 9459–9474.

[16] Q. Wang et al., “Knowledge graph embedding: A survey of approaches and applications,” IEEE Trans. Knowl. Data Eng., vol. 29, no. 12, pp. 2724–2743, 2017.

[17] D. Suciu, D. Olteanu, C. Re, and C. Koch, Probabilistic Databases. San Rafael, CA, USA: Morgan & Claypool, 2011.

[18] O. Benjelloun, A. D. Sarma, A. Y. Halevy, and J. Widom, “ULDBs: Databases with uncertainty and lineage,” in Proc. VLDB, 2006, pp. 953– 964.

[19] X.-L. Li et al., “Overview and limitations of database in global traditional medicines: A narrative review,” Acta Pharmacol. Sin., vol. 46, no. 2, pp. 235–263, 2025.

[20] Z. Sun, Z.-H. Deng, J.-Y. Nie, and J. Tang, “RotatE: Knowledge graph embedding by relational rotation in complex space,” in Proc. ICLR, 2019.

[21] T. Trouillon, J. Welbl, S. Riedel, E. Gaussier, and G. Bouchard, “Complex embeddings for simple link prediction,” in Proc. ICML, 2016, pp. 2071–2080.

[22] B. Yang, W.-T. Yih, X. He, J. Gao, and L. Deng, “Embedding entities and relations for learning and inference in knowledge bases,” in Proc. ICLR, 2015.

[23] I. Balazevic, C. Allen, and T. M. Hospedales, “Multi-relational Poincare graph embeddings,” in Proc. NeurIPS, 2019, pp. 4463–4473.

[24] I. Balazevic, C. Allen, and T. M. Hospedales, “TuckER: Tensor factorization for knowledge graph completion,” in Proc. EMNLP-IJCNLP, 2019, pp. 5185–5194.

[25] H. Q. Ngo, E. Porat, C. Re, and A. Rudra, “Worst-case optimal join algorithms,” J. ACM, vol. 65, no. 3, pp. 1–40, 2018.

[26] H. Q. Ngo, “Worst-case optimal join algorithms: Techniques, results, and open problems,” in Proc. PODS, 2018, pp. 111–124.

[27] Y. Jia et al., “Qibo: A large language model for traditional Chinese medicine,” Expert Syst. Appl., vol. 284, p. 127672, 2025.

[28] Y. Zhuang, L. Yu, N. Jiang, and Y. Ge, “TCM-KLLaMA: Intelligent generation model for traditional Chinese medicine prescriptions based on knowledge graph and large language model,” Comput. Biol. Med., vol. 189, p. 109887, 2025.

[29] J. He et al., “OpenTCM: A GraphRAG-empowered LLM-based system for traditional Chinese medicine knowledge retrieval and diagnosis,” 2025, arXiv:2504.20118. [Online]. Available: https://arxiv.org/abs/2504.20118

[30] W. Yue et al., “TCMBench: A comprehensive benchmark for evaluating large language models in traditional Chinese medicine,” 2024, arXiv:2406.01126. [Online]. Available: https://arxiv.org/abs/2406.01126

[31] M. Schlichtkrull et al., “Modeling relational data with graph convolutional networks,” in Proc. ESWC, 2018, pp. 593–607.

[32] D. Edge et al., “From local to global: A graph RAG approach to queryfocused summarization,” 2024, arXiv:2404.16130. [Online]. Available: https://arxiv.org/abs/2404.16130

[33] T. Huang, L. Lu, J. Chen et al., “A triaxial benchmark for assessing responses from large language models in traditional Chinese medicine,” Commun. Med., 2026, doi: 10.1038/s43856-026-01631-5.

[34] Chinese Pharmacopoeia Commission, Pharmacopoeia of the People’s Republic of China, 2020 ed. Beijing, China: China Medical Science Press, 2020.

[35] C. Knox, M. Wilson, C. M. Klinger et al., “DrugBank 6.0: the Drug-Bank Knowledgebase for 2024,” Nucleic Acids Res., vol. 52, no. D1, pp. D1265–D1275, 2024, doi: 10.1093/nar/gkad976.

[36] Y. Zhou et al., “TTD: Therapeutic Target Database describing target druggability information,” Nucleic Acids Res., vol. 52, no. D1, pp. D1465–D1477, 2024.

[37] T. J. Green, G. Karvounarakis, and V. Tannen, “Provenance semirings,” in Proc. PODS, 2007, pp. 31–40.

[38] T. Neumann and G. Weikum, “RDF-3X: A RISC-style engine for RDF,” Proc. VLDB Endow., vol. 1, no. 1, pp. 647–659, 2008.

[39] C. Weiss, P. Karras, and A. Bernstein, “Hexastore: Sextuple indexing for semantic web data management,” Proc. VLDB Endow., vol. 1, no. 1, pp. 1008–1019, 2008.

[40] D. S. Himmelstein et al., “Systematic integration of biomedical knowledge prioritizes drugs for repurposing,” eLife, vol. 6, p. e26726, 2017.

[41] P. Chandak, K. Huang, and M. Zitnik, “Building a knowledge graph to enable precision medicine,” Sci. Data, vol. 10, p. 67, 2023.