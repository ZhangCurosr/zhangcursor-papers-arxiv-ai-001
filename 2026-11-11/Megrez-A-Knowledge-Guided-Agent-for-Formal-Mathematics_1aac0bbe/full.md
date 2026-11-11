# Megrez: A Knowledge-Guided Agent for Formal Mathematics

over Morse–Kelley Set Theory

Tiancheng Xu<sup>1</sup> Wensheng Yu<sup>1</sup>

## Abstract

General-purpose coding agents equipped with proof-assistant tools have shown strong performance on standalone competition problems, yet cumulative textbook formalization inside a low-resource mathematical library still faces three obstacles: aligning textbook concepts with library declarations, keeping long-horizon search synchronized with live proof states, and integrating locally verified proofs reliably into a growing codebase. We present Megrez, a knowledge-guided formalization agent for Rocq libraries built on Morse– Kelley (MK) set theory. Megrez coordinates a general coding agent through three mechanisms: (1) a source-rebuildable knowledge graph merging formal declarations, compiler references (.glob), and curated metadata to support retrieval via Chinese aliases, concept clusters, proof dependencies, and multi-lane full-text ranking; (2) a feedback-driven 1–5 tactic short-burst proving loop paired with a goal-local branch ledger and exponential-interval structured reflection; and (3) a three-check acceptance pipeline (statement fidelity, axiom auditing, and full builds) coupled with a multi-corpus Provider stack and a candidate-bound recoverable batch state machine. Deployment and evaluation on a point-set topology corpus demonstrate efective cross-chapter formalization and scalable premise retrieval.

Keywords: formal mathematics; Morse–Kelley set theory; Rocq; general coding agents; knowledge graph; interactive theorem proving

## 1 Introduction

Formal mathematics represents mathematical definitions, theorems, and proofs as objects that a proof assistant can check mechanically. In Rocq, users construct proofs interactively with tactics and the kernel type-checks the resulting proof term [1]. While kernel verification guarantees logical soundness, formalizing a textbook section by section inside a maintainable library remains labor-intensive: one must align informal terminology with existing declarations, discharge set-theoretic side conditions left tacit on paper, and ensure that new developments integrate cleanly with the surrounding codebase.

Large language models (LLMs) are advancing proof automation along two main lines: training specialized provers via synthetic data and reinforcement learning for whole-proof generation [12, 13, 14], and conditioning step-by-step tactic generation on proof states [8, 10, 11]. Recently, Numina-Lean-Agent demonstrated a more general paradigm: coupling a general-purpose coding agent with Lean tools exposed through the Model Context Protocol (MCP) [3] achieves competitive results on both competition benchmarks and research-level formalization without training a task-specific model [2]. Megrez adopts this general-agent-plus-tools paradigm for Rocq libraries built over Morse–Kelley (MK) axiomatic set theory [4, 5, 6], targeting cumulative textbook formalization across three core challenges:

1. Concept Alignment: MK set-theory libraries are much smaller than Mathlib and rarely appear in pretraining data. A single textbook concept may correspond to a definition, abbreviation, or numbered theorem, and set-theoretic encoding requires explicit side conditions (e.g., proving that a class is a set Ensemble rather than a proper class, or that an argument lies in a domain). Identifier string search alone rarely bridges textbook intent to formal infrastructure.

2. State Synchronization: Mathematically sound plans often fail locally due to hypothesis shapes, notation bindings, or higher-order unification direction. Generating long scripts without intermediate feedback builds on broken states; conversely, repeated local failures usually signal a flaw in the mathematical route or representation rather than a single tactic.

3. Library Integration: Textbook formalization is cumulative. A locally accepted script must still be checked to confirm that it proves the exact reviewed statement, introduces no unapproved axioms or transitive dependencies on other Admitted placeholders, compiles across the full project, and re-enters the retrieval index; multi-theorem batches further require dependency scheduling, concurrency control, and crash recovery.

To address these challenges, we design and implement Megrez (Figure 1). Our main contributions are:

• A source-rebuildable formal knowledge graph and multi-lane retrieval (Section 3): We combine Rocq sources, compiler-resolved .glob references, and human-reviewed metadata into a SQLite graph indexing Chinese aliases, concept clusters, proof dependencies, and textbook mappings, supporting scope-before-cap filtering and seven-column FTS5 + Reciprocal Rank Fusion (RRF) ranking.

• Short-burst interactive proving with goal-local reflection (Section 4): We pair 1–5 tactic bursts with a per-goal branch ledger, structured reflection at exponential branch checkpoints {2, 4, 8, 16, 32}, and a three-tier knowledge triage discipline (capping top-level skills at ≤ 150 lines while discarding noisy cross-theorem execution trajectories).

• Three-check acceptance, Provider stacks, and recoverable batches (Sections 2 and 5): We organize formalization around two human review gates, a single-writable-Workspace Provider stack, and a concurrent batch scheduler featuring proof-stripped context fingerprints, canonical proof-hash binding, and four-way deterministic crash recovery.

• Deployment, scaling benchmarks, and collaborative case studies (Sections 6 and 7): We deploy Megrez on MKTopos, a Rocq formalization of Xiong’s Lectures on Point-Set Topology [7] comprising 262 files, 3,894 declarations, 2,529 completed Qed proofs (102,122 lines of proof script), and 60,982 reference edges. We evaluate retrieval latency up to 100,000 nodes (58.8× exact-lookup speedup, bounded 3–7 SQL queries per call), verify system invariants across 319 automated contract tests, and present three collaborative case studies spanning Tukey’s lemma, connectedness of finite and arbitrary product spaces, and the 2D Brouwer fixed-point and Jordan separation theorems.

## 2 System Overview and Formalization Lifecycle

## 2.1 Architecture and Division of Responsibilities

Megrez follows three principles: traceable knowledge, feedback-grounded decisions, and verified sedimentation. As shown in Figure 1, the system is organized into four layers: (1) the Human layer, where a mathematician supplies textbook statements and reviews formal statements (Gate 1) and semantic annotations (Gate 2); (2) the Control layer, where a general coding agent (currently Codex) loaded with three workflow skills (mk-formalize, mk-prove/rocq-prove, mk-kg-update) plans and coordinates tools; (3) the Tool layer, comprising the built-in mk-kg MCP server (graph queries, rebuilds, and batch state machine) and an external rocq-mcp server (interactive tactic execution, statement verification, and axiom auditing); and (4) the Data and Execution layer, consisting of a Provider dependency stack with a single writable Workspace, an idempotently rebuildable SQLite database, and the Rocq compiler and kernel. Workflow skills define behavioral contracts for the agent, whereas hard safety invariants—capability authentication, compare-and-swap revisions, and candidate proof-hash binding—are enforced deterministically in mk-kg tool code.

![](images/9bd9aef11f13529f25f3ea6af7cf948e8587662695b3a414a8667d3632985217.jpg)  
Figure 1: Architecture of Megrez. Solid arrows denote queries or execution; the dashed arrow indicates serial source writes performed by the main session according to batch state.

## 2.2 Closed-Loop Lifecycle and Three-Check Acceptance Criterion

Figure 2 illustrates the formalization lifecycle. mk-formalize inventories textbook concepts and drafts Rocq statements ending with Admitted; after Gate 1 human review confirms statement fidelity, mk-prove autonomously coordinates retrieval, interactive proving, and verification; once the project builds, mk-kg-update

rebuilds the graph and drafts concept and Chinese-alias annotations for Gate 2 approval. Between the two gates, stage handofs and crash recovery proceed autonomously within the authorized scope.

![](images/6c1e4075f4b119c11f948cabab1b0de94aa478a6aef66a61946913b39b8b5c90.jpg)  
Figure 2: Formalization lifecycle bracketed by two human review gates.

Let $\varphi$ be the reviewed statement, Γ the current Rocq environment, p a candidate proof term, A the axiom whitelist declared by the Workspace Provider, and $\Gamma [ p ]$ the project state after inserting p. A candidate proof is accepted if and only if all three checks pass:

$$
\Gamma \vdash p : \varphi ( \mathtt { r o c q _ { - } v e r i f y } ) , \qquad \mathrm { A s s u m p t i o n s } ( p ) \subseteq \mathcal { A } \left( \mathtt { r o c q _ { - } a s s u m p t i o n s } \right) , \qquad \mathrm { B u i l d } ( \Gamma [ p ] ) = \mathtt { p a s s } \left( \mathtt { f u l l \ m a k e } \right) .\tag{1}
$$

Each check prevents a distinct failure mode: rocq\_verify prevents the model from weakening the conclusion or strengthening hypotheses; rocq\_assumptions audits transitive dependencies so p cannot silently route through another unproven Admitted item in the same file (which a single-file compile would not catch); and full make ensures project-wide compatibility and fresh .glob references.

## 3 Knowledge Graph: Bridging Textbook Language and Formal Libraries

## 3.1 Graph Representation and Idempotent Construction

Megrez represents library knowledge as a five-tuple K = (V, E, C, A, T) stored in a single SQLite database (schema v2):

• Declaration nodes (V): canonical ledger IDs (‘corpus:file:kind:name‘), formal statements, source spans, comments, and proof status (qed/admitted/none).

• Reference edges (E): typed relations including proof\_uses, statement\_mentions, uses\_notation, uses\_tactic, file\_requires, and dual\_of. The extractor resolves references primarily from compilergenerated .glob files, falling back to lexical matching (tagged with source token) only when .glob data is missing.

• Curated semantic layers (C, A, T): concept clusters C group declarations on the same topic by role (def/intro/key/helper) and attach reviewed usage\_hints; aliases A map Chinese textbook terms (e.g., “nest”, “finite character”, “connected component”) to formal identifiers or concepts; textbook mappings T track the formal counterpart and status of each numbered textbook item.

As shown in Figure 3, the builder merges .v sources, .glob references, and version-controlled YAML metadata into deterministic search projections (search\_documents, search\_terms) and a seven-column FTS5 index (node\_fts), computes a path- and timestamp-independent content hash (kg\_content\_hash), and retains a .previous snapshot for atomic rollback.

![](images/a45f87c0f693433d3c6024df74b73da476abb61b6d25b0c2eeb4ca1bf1540a66.jpg)  
Figure 3: Knowledge graph construction and KG-first retrieval discipline.

## 3.2 KG-First Retrieval and Multi-Lane Fusion

Megrez enforces a KG-first retrieval sequence: (1) kg\_search by identifier, Chinese alias, or statement text; (2) kg\_concepts to inspect cluster members and usage\_hints; (3) kg\_deps to inspect the proof\_uses neighborhood of proved theorems on the same topic as a structural template; and (4) kg\_node to check exact types and dual theorems. Exploratory Rocq Search is permitted only when the graph misses, and any fallback hit must be reported as a missing annotation.

Retrieval is dispatched through versioned profiles. While legacy-search-v1 and legacy-premise-v1 preserve fixed-tier compatibility, the schema $\mathrm { v 2 }$ profiles (fts-interactive-v1 and fts-premise-v1) pin exact identifier and node-alias matches at the top and fuse prefix, concept-cluster, graph-neighborhood, and seven-column FTS5 lanes (including a symbol lane that maps operators such as /\ , ->, and := into safe tokens, plus a Chinese character bigram lane) using multi-column BM25 and Reciprocal Rank Fusion $\begin{array} { r } { \left( \mathrm { R R F } ( v ) = \sum _ { \ell } ( 6 0 + \mathrm { r a n k } _ { \ell } ( v ) ) ^ { - 1 } \right) } \end{array}$ This eliminates both full-table SQL LIKE substring scans and per-hit N + 1 concept queries.

## 3.3 Budgeted Premise Packets and Scope-Before-Cap Filtering

To assemble a bounded context for proof search or read-only workers, kg\_context\_packet packs ranked premises greedily within total budget B and premise-channel budget $B _ { P }$ , enforcing τ(Packet $( P ) ) \leq B$ and $\begin{array} { r } { \sum _ { v \in P } \widehat { \tau } ( v ) \le B _ { P } } \end{array}$ under a deterministic token estimator τ. When a caller supplies a candidate scope snapshot $S$ (CandidateScopeSnapshot), the engine materializes a request-private TEMP eligible\_nodes table and joins it before per-lane caps so that $P \subseteq S ,$ , while physically masking outgoing proof\_uses edges of the target theorem itself to prevent benchmark leakage.

## 4 Feedback-Driven Interactive Proving

## 4.1 Short-Burst Loop and Goal-Local Branch Ledger

As shown in Figure 4, for a nontrivial theorem the agent first retrieves KG context $K _ { t }$ and writes a mathematical proof sketch, then interacts with Rocq in short bursts of 1–5 tactics $a _ { t } \gets \mathrm { A g e n t } ( s _ { t } , K _ { t } , H _ { t } )$ , immediately inspecting the returned state or error $o _ { t } \gets \operatorname { R o c q } ( s _ { t } , a _ { t } )$ and using rocq\_step\_multi to probe alternative tactics at choice points without advancing the main state.

![](images/8fe40360ecaee4de53262c73863bde20b96e62d9271f62c970f8b56da5514b67.jpg)  
Figure 4: Feedback-driven short-burst proving loop with exponential-interval goal-local reflection.

We distinguish a failed tactical probe (burst) from an abandoned strategy (branch). Each abandoned route is logged in the per-goal ledger $H _ { t }$ with its key tactics, last valid state, and blocking subgoal. When failed branches on the current goal reach an exponential checkpoint in {2, 4, 8, 16, 32}, the session triggers structured reflection: it inspects the exact goal, frozen premise packet, and ledger, diagnoses the root cause (mathematical route, formal representation, or tactic usage), writes a revised proof sketch specifying how the next falsifiable burst difers from prior failures, and resumes search automatically.

## 4.2 No Cross-Theorem Trajectory Memory and Three-Tier Knowledge Triage

In early prototypes, we experimented with persisting successful script templates and failure-recovery trajectories across theorems for retrieval into subsequent proofs. Controlled ablations showed that cross-theorem execution trajectories yielded negligible transfer gains on new theorems while adding context overhead and benchmark leakage risks. Megrez therefore keeps the branch ledger strictly local to the active goal and discards it upon completion.

Instead, reusable knowledge is routed through a three-tier knowledge triage discipline governed by attention cost: (1) top-level workflow skills (SKILL.md, hard $\mathrm { c a p } \leq 1 5 0$ lines), loaded whenever a skill triggers and restricted to workflow invariants; (2) reference idiom manuals (reference-\*.md, soft $\mathrm { c a p } \leq 1 5 0 ~ \mathrm { l i n e s } )$ loaded on demand for corpus-wide MK and Rocq idioms; and (3) concept-cluster usage\_hints in the knowledge graph (unbounded capacity), recalled only when a relevant topic cluster is queried. In-proof subgoals (assert/enough) are used freely, whereas any new standalone helper lemma (including Local Lemma) must be stated with its exact type and dependencies and approved at Gate 1.

## 5 Corpus Providers and Recoverable Multi-Theorem Batches

## 5.1 Provider Stack and Proof-Stripped Dual Fingerprints

Megrez decouples workflow tooling from mathematical corpora via local Providers, each carrying a provider.json manifest declaring its version, Rocq version, corpus paths, metadata directories, axiom whitelist, and Provider dependencies. Providers form a directed acyclic stack in which exactly one corpus is mounted as the writable Workspace and all others are read-only dependencies.

For multi-theorem workloads, FORMALIZATION\_BATCH.yaml is initialized as the single source of truth only after all target statements pass Gate 1 (generating CHECKLIST.md automatically). Over an explicit dependency DAG D = (U, F), the ready set is R = {v ∈ U | status(v) = todo, $\forall ( u , v ) \in F ,$ , status(u) = qed}. To allow parallel worker search across independent theorems in the same .v file, the parser computes two SHA-256 fingerprints per target: a normalized statement fingerprint (statement\_sha256) and a proof-stripped context fingerprint (context\_sha256) over the preceding file prefix with earlier proof bodies removed. Replacing an earlier theorem’s Admitted with a completed Qed proof in the same file therefore preserves later theorems’ context fingerprints without false stale invalidation, while any edit to preceding definitions, notations, or target statements is caught immediately.

## 5.2 Parallel Read-Only Search, Hash-Bound Integration, and Four-Way Recovery

As shown in Figure 5, read-only workers launched with isolated context (fork\_turns="none") claim ready goals in parallel (todo→leased), returning locally verified scripts (candidate) or entering blocked when attempts cross {2, 4, 8, 16, 32}. The main session holds a private integration capability token (integration\_capability, stored only as a SHA-256 hash in the manifest; losing it fails closed) and performs serial integration under compare-and-swap revision numbers: it verifies both fingerprints (integrating), runs the four acceptance checks (rocq\_verify, assumptions\_ok, file\_compile, full\_build) plus intra-batch dependency verification, and hash-binds the canonicalized source proof (normalizing comments and whitespace while preserving token boundaries such as :=) to the persisted candidate before recording qed. If any check fails, the main session reverts the file to Admitted, returns to candidate, and invokes batch\_reject\_candidate to atomically archive the rejected script and failure evidence into the goal’s ledger before transitioning to blocked.

![](images/74cc7a1671da346601a0c80986fce5b2b7f6b0c97242f97cb450e8d4f1179e6a.jpg)  
Figure 5: Per-theorem state machine in a recoverable batch.

After an interruption, sequential execution of validate, recover, and status classifies each interrupted theorem into one of four deterministic recovery actions: (1) redispatch (expired leased entry with Admitted source resets to todo); (2) rewrite\_candidate (integrating interrupted before file write returns to candidate); (3) reverify (integrating whose persisted candidate is already written into the file with Qed resumes directly at the four acceptance checks); and (4) manual (unexpected source drift freezes the entry for human inspection).

## 6 System Evaluation

## 6.1 MKTopos Corpus and Knowledge Graph Scale

We deploy Megrez on MKTopos (Rocq 9.1), a formalization of Xiong’s Lectures on Point-Set Topology [7] built over MK set theory. Tables 1 and 2 summarize the corpus and its runtime knowledge graph. Across 12 modules/chapters and 262 .v files totaling 102,122 lines of proof script, the graph indexes 3,894 formal declarations (including 2,529 completed Qed proofs) and 60,982 reference edges, of which 96.8% (60,795) are resolved precisely from compiler .glob files. Curated metadata covers 56 concept clusters (1,678 memberships), 982 aliases, and 415 textbook items across 54 sections (331 formalized).

## 6.2 Retrieval Latency and Scalability Benchmarks

To evaluate scalability as libraries grow by 1–2 orders of magnitude, we use a deterministic synthetic generator (seed 20260901, matching the corpus distribution with 5% Chinese comments, 4% formal operator tokens, 10% node aliases, and 0.1% naming collisions) to construct 10,000-node (2.5× MKTopos, 16.9 MB, built in 1.05 s)

Table 1: Overall scale and structure of the MKTopos knowledge graph.
<table><tr><td>Category</td><td>Breakdown</td><td>Count</td></tr><tr><td>Source scale</td><td>.v source files / Total lines of completed proof script</td><td>262 / 102,122</td></tr><tr><td rowspan="2">Declarations (V)</td><td>Theorem (652) / Lemma (1,718) / Corollary (97) / Prop. (59) / Fact (116) Definition (903) / Abbreviation (189) /</td><td>2,642</td></tr><tr><td>Notation (55) / Other (Instance, Ltac, etc. 105)</td><td>1,252</td></tr><tr><td rowspan="2">Reference edges (E)</td><td>Total nodes (with completed Qed proofs) proof_uses/statement_mentions uses_notation /uses_tactic /</td><td>3,894 (2,529) 32,227 /  12,979</td></tr><tr><td>file_requires /dual_of &amp; notation_of Total reference edges (resolved from compiler</td><td>12,455 / 1,745 / 1,493 / 83 60,982 (96.8%)</td></tr><tr><td>Semantics &amp; FTS</td><td>·glob) Concept clusters (members) / Aliases / Textbook sections (items, formalized)/</td><td>56 (1,678) / 982 / 54 (415, 331) / 23,384</td></tr></table>

Table 2: Module and chapter breakdown of the MKTopos corpus.
<table><tr><td>Module / Chapter</td><td>Mathematical topic</td><td>Files</td><td>Decls</td><td>Proved</td><td>Proof lines</td></tr><tr><td>MK</td><td>MK set theory (axioms, ordinals, cardinals)</td><td>3</td><td>437</td><td>284</td><td>2,517</td></tr><tr><td>Real_Axioms + Real_Analysis</td><td>Real number axioms and basic analysis</td><td>6</td><td>340</td><td>227</td><td>3,543</td></tr><tr><td>Chapter1_NaiveSetTheory</td><td>Set operations, countability, AC &amp; Tukey</td><td>9</td><td>149</td><td>112</td><td>1,720</td></tr><tr><td>Chapter2_MetricSpaces</td><td>Metric and topological spaces, continuity</td><td>7</td><td>172</td><td>46</td><td>2,453</td></tr><tr><td>Chapter3_SubspacesProducts</td><td>Subspaces, finite/arbitrary products, quotients</td><td>4</td><td>117</td><td>47</td><td>1,628</td></tr><tr><td>Chapter4_Connectedness</td><td>Connectedness, components, path/local conn.</td><td>5</td><td>148</td><td>111</td><td>5,555</td></tr><tr><td>Chapter5_CountabilityAxioms</td><td>First/second countability, Lindelöf, separable</td><td>4</td><td>91</td><td>64</td><td>2,643</td></tr><tr><td>Chapter6_SeparationAxioms</td><td>Separation axioms, Urysohn, Tietze, metrization</td><td>7</td><td>356</td><td>270</td><td>10,048</td></tr><tr><td>Chapter7_Compactness</td><td>Compactness, Tychonoff, local compactness</td><td>8</td><td>93</td><td>62</td><td>8,468</td></tr><tr><td>Chapter8_CompleteMetricSpaces</td><td>Complete metric spaces, Baire category, completion</td><td>2</td><td>161</td><td>132</td><td>6,324</td></tr><tr><td>Chapter9_MappingSpaces</td><td>Mapping spaces, compact-open, Arzelà-Ascoli</td><td>3</td><td>35</td><td>14</td><td>3,025</td></tr><tr><td>Chapter10_FundamentalGroup</td><td>π1(S1), covering maps, Brouwer &amp; Jordan thms</td><td>204</td><td>1,795</td><td>1,160</td><td>54,198</td></tr><tr><td colspan="2">Total</td><td>262</td><td>3,894</td><td>2,529</td><td>102,122</td></tr></table>

and 100,000-node (25× MKTopos, 170.4 MB, built in 10.84 s) databases. Table 3 reports 95th-percentile latency (p95 over 7 repetitions on Apple Silicon arm64, 10 cores, Python 3.12.11, SQLite 3.50.3) and executed SQL statements per call.

Table 3: Interactive and premise retrieval performance on 10k- and 100k-node graphs (p95 latency in ms; SQL statement count in parentheses).
<table><tr><td rowspan="2">Task &amp; workload</td><td rowspan="2">Query example</td><td colspan="2">10,000 nodes (16.9 MB)</td><td colspan="2">100,000 nodes (170.4 MB)</td></tr><tr><td>legacy-v1</td><td>fts-v1</td><td>legacy-v1</td><td>fts-v1</td></tr><tr><td>Interactive: Exact identifier</td><td>Synthetic_000001</td><td>8.11 (7–24)</td><td>0.71 (3)</td><td>72.29 (7–24)</td><td>1.23 (3)</td></tr><tr><td>Interactive: Chinese concept</td><td>conn. space (ZH)</td><td>8.30 (16)</td><td>6.21 (7)</td><td>76.87 (16)</td><td>19.73 (7)</td></tr><tr><td>Interactive: Formal operators</td><td>/ -&gt; : =</td><td>6.33 (6)</td><td>6.12 (7)</td><td>58.51 (6)</td><td>18.10 (7)</td></tr><tr><td>Interactive: Common token</td><td>topology common</td><td>40.11 (36)</td><td>24.33 (7)</td><td>410.80 (36)</td><td>199.24 (7)</td></tr><tr><td>Interactive: Single CJK char</td><td>conn. char (ZH)</td><td>8.99 (36)</td><td>0.91 (5)</td><td>82.55 (36)</td><td>0.99 (5)</td></tr><tr><td>Premise: Exact identifier</td><td>Synthetic_000001</td><td>3.02 (3)</td><td>25.42 (6)</td><td>20.54 (3)</td><td>213.98 (6)</td></tr><tr><td>Premise: Chinese semantic</td><td>conn. space (ZH)</td><td>2.92 (2)</td><td>9.06 (5)</td><td>19.83 (2)</td><td>41.92 (5)</td></tr><tr><td>Premise: Formal operators</td><td>八 -&gt; : =</td><td>2.89 (2)</td><td>8.88 (5)</td><td>20.04 (2)</td><td>40.38 (5)</td></tr><tr><td>Premise: Common token</td><td>topology common</td><td>3.09 (3)</td><td>26.26 (6)</td><td>20.34 (3)</td><td>217.18 (6)</td></tr></table>

Two findings stand out. First, in interactive search (kg\_search), fts-interactive-v1 outperforms legacy-search-v1 on every workload: at 100k nodes, exact lookup accelerates by 58.8× (72.29 ms → 1.23 ms), Chinese concept search by 3.9×, and common-token search by 2.1×, while reducing SQL statements from up to 36 to a bounded 3–7 per call and recalling 10 hits on pure operator queries (/\ -> :=) where legacy search returns 0. Second, in premise retrieval (kg\_context\_packet), legacy-premise-v1 performs only identifier and passed-concept lookups without full-text matching, whereas fts-premise-v1 materializes a request-private TEMP eligible\_nodes table for scope-before-cap filtering and fuses seven-column BM25 with graph neighborhoods via RRF in under 27 ms at 10k nodes and 40–217 ms at 100k nodes.

## 6.3 Batch Scheduling and Engineering Contract Verification

Megrez includes 39 test modules with 319 automated contract and fault-injection tests (Table 4), all passing in continuous integration. In end-to-end multi-theorem scenario tests (test\_batch\_scenario.py) over dependent theorems A, B → C in a single .v file, two read-only workers claim A and B in parallel; injecting crashes before and after source writing on B deterministically triggers rewrite\_candidate and reverify; completing B’s proof preserves A’s proof-stripped context fingerprint without false stale invalidation; and C unlocks automatically once both A and B reach qed, completing after simulated lease expiration and redispatch.

Table 4: Product-level engineering contract and fault-injection test suite (39 files, 319 tests).
<table><tr><td>Test category</td><td>Files</td><td>Tests</td><td>Verified invariants and fault-injection scenarios</td></tr><tr><td>Source &amp; ref parsing</td><td>5</td><td>43</td><td>.v declaration/proof boundary parsing, terminator detection after focused blocks, .glob extraction, proof-stripped context fingerprints</td></tr><tr><td>KG build &amp; Providers</td><td>10</td><td>80</td><td>Idempotent rebuilds, fault injection with atomic .previous rollback, dual-schema support, Provider topological sort, single writable Workspace</td></tr><tr><td>Retrieval &amp; scope safety</td><td>11</td><td>69</td><td>FTS5 injection-safe compilation, pinned navigation + RRF ranking, scope-before-cap filtering, self-reference masking, token budgeting</td></tr><tr><td>Batch &amp; crash recovery</td><td>7</td><td>120</td><td>DAG readiness, CAS revisions, capability-token auth, same-file parallel claims, 4-way recovery (redispatch/rewrite_candidate/reverify/manual), candidate</td></tr><tr><td>Workflow &amp; packaging</td><td>6</td><td>7</td><td>rejection Attention-budget line caps (≤ 150 lines), cooperative read-only worker contracts, Gate 1 review before batch init, zero trajectory memory</td></tr></table>

## 7 Collaborative Formalization Case Studies

## 7.1 Case Study 1: Axiom of Choice and Tukey’s Lemma (Textbook §1.8)

Section 1.8 of Xiong’s textbook [7] covers 4 definitions and 5 theorems/lemmas/corollaries (Table 5), illustrating concept alignment and reflection-driven lemma decomposition:

• Concept alignment (mk-formalize): Running kg\_textbook("1.8") and kg\_search uncovered two structural alignments: Definition 1.8.2 (“nest of a family”) directly matched Nest in the foundational MK library, avoiding duplicate definitions; and Axiom 1.8.1 (“every set admits a choice function”) became a provable theorem Xiong\_1\_8\_1 (12 lines) because MK’s global choice axiom AxiomIX already supplies a choice function on the universe $\mu \setminus \{ \Phi \}$ , which need only be restricted to ${ \mathcal { P } } ( X ) \setminus \{ \Phi \}$ • Reflection-driven helper lemmas (mk-prove): Proving Lemma 1.8.3 (Xiong\_1\_8\_3, closure of finitecharacter families under nest unions) requires showing that every finite subset $B \subset \cup \mathcal { C }$ lies in a single member of the nest C. Initial bursts attempting direct induction on B failed because finiteness in MK is defined cardinally as $P [ B ] \in \omega \ \left( \mathtt { F i n i t e } \right)$ and applying Mathematical\_Induction directly with a quantified set variable B fails higher-order unification. At the reflection checkpoint, the main session factored the proof into two standalone lemmas: remove\_one\_card (16 lines, using Lemma163b, finue, and successor injectivity MKT136 to reduce cardinality from n<sup>+</sup> to n) and fin\_chain\_union (40 lines, passing an explicit predicate apply (Mathematical\_Induction (fun $\textbf { n } = > \ \ldots ) )$ and merging nest members by total ordering). This closed Xiong\_1\_8\_3 in 14 lines. For Tukey’s Lemma (Xiong\_1\_8\_4), KG search on “maximal” surfaced Hausdorf’s maximal principle MKT143, yielding a 32-line proof via a maximal nest ${ \mathcal { N } } \subset { \mathcal { F } }$ instead of the textbook’s five-step “t-subfamily” construction.

• Knowledge sedimentation (mk-kg-update): Following full compilation, the update stage registered the concept cluster xiong\_choice\_tukey and recorded the global-choice restriction and explicit-predicate cardinal induction idioms into usage\_hints and the idiom manual.

Table 5: Formalization summary for Textbook §1.8 in Sec1\_8\_AxiomOfChoiceTukey.v.
<table><tr><td>Textbook item</td><td>Formal identifier</td><td>Kind</td><td>Lines</td><td>Key design or proof route</td></tr><tr><td>Defs. 1.8.1–1.8.4</td><td>SelectionFun, FiniteChar, MaxMember/</td><td>definition</td><td></td><td>Local choice function, finite-character family, maximal/minimal members; Def.</td></tr><tr><td>Axiom 1.8.1 / Thm. 1.8.2</td><td> $\tt X i o n g \_ 1 \_ 8 \_ 1 $  Xiong_1_8_2</td><td>theorem</td><td>12 / 19</td><td>Restricts global choice AxiomIX to P(X) \ {Φ} and indexed family A via MKT126a/b/c and MKT30</td></tr><tr><td>(Helper lemmas)</td><td>remove_one_card/ fin_chain_union</td><td>lemma</td><td>16 / 40</td><td>Removing one element reduces  $P [ B ] = n ^ { + }$  to n; explicit-predicate induction on  $n = P [ B ]$  over nests</td></tr><tr><td>Lemma 1.8.3</td><td>Xiong_1_8_3</td><td>theorem</td><td>14</td><td>Nonempty finite-character families are closed under subsets and nest unions (via fin_chain_union)</td></tr><tr><td>Thm. 1.8.4 / Cor. 1.8.5</td><td>xiong_1_8_4/ xiong_1_8_5</td><td>thm / cor</td><td>32  / 63</td><td>Tukey&#x27;s Lemma via Hausdorff&#x27;s principle MKT143 on a maximal nest; Corollary on  ${ \mathcal { F } } _ { A } = \{ B \in { \mathcal { F } } \mid A \subset B \}$ </td></tr></table>

## 7.2 Case Study 2: Connectedness from Finite to Arbitrary Products (Chapter 4)

Chapter 4 (Connectedness; 148 declarations, 111 proofs, 5,555 lines) demonstrates cross-section reuse via the knowledge graph:

1. Coordinate-update chains on indexed products (§4.1): In MK set theory, binary Cartesian products $X \times Y$ (ordered pairs) and indexed products $\Pi _ { \gamma \in G } X _ { \gamma }$ (IProduct, choice functions on G) have distinct encodings. Rather than inducting on binary products and transporting across homeomorphisms, the agent worked directly on IProduct: it defined single-coordinate updates (CoordinateUpdateMap), proved their images are connected slices (41 lines), and inducted over finite index sets G to link slices into FiniteCoordinateUpdateChain (87 lines), proving finite-product connectedness (Xiong\_4\_1\_9) in 36 lines.

2. Cross-section reuse I: punctured R<sup>n</sup> connectedness and $\mathbb { R } ^ { n } \not \cong \mathbb { R } \ ( \ S 4 . 2 )$ : Whereas the textbook proves only for $n = 2 \operatorname { t h a t } \mathbb { R } ^ { 2 } \backslash \{ 0 \}$ is connected and $\mathbb { R } ^ { 2 } \not \cong \mathbb { R }$ , querying kg\_concepts for xiong\_connectedness retrieved the §4.1 coordinate slices. Adding a 13-line lemma (CoordinateUpdateSlice\_avoids\_point, showing that updating coordinate α avoids the origin whenever some other coordinate $\beta \neq \alpha$ is nonzero) generalized punctured connectedness and $\mathbb { R } ^ { n } \neq \mathbb { R }$ to all finite dimensions $n \geq 2$ (Xiong\_4\_2\_6, 129 lines; Xiong\_4\_2\_7, 26 lines).

3. Cross-section reuse II: arbitrary product connectedness (§4.3): In §4.3, querying kg\_deps on FiniteCoordinateUpdateChain provided the template for Tychonof’s connectedness theorem: generalizing coordinate updates to finite subsets $F \subset G$ (FiniteGeneralCoordinateUpdateChain, 72 lines) showed that the connected component C of a basepoint meets every basic open box and is dense; combined with component closure (Xiong\_4\_3\_1), this proved arbitrary product connectedness (Xiong\_4\_3\_2) in 59 lines.

## 7.3 Case Study 3: 2D Brouwer Fixed-Point and Jordan Separation Theorems (Chapter 10)

In Chapter 10 (204 files, 1,795 declarations, 54,198 lines), two mechanisms proved decisive: (1) in §10.5 (2D noretraction Xiong\_10\_5\_1 and Brouwer fixed-point theorem Xiong\_10\_5\_2), reflection replaced pointwise algebraic composition on R with subspace domain restriction (restricting square roots to nonnegative reals and inversion to positive reals, composed via subspace corestriction), eliminating nonnegativity and nonzero-denominator obligations from continuity proofs; and (2) in §10.6 (Jordan separation theorem Xiong\_10\_6\_6), decomposing non-separation by arcs (Xiong\_10\_6\_1) and disks (Xiong\_10\_6\_2), boundary properties (Xiong\_10\_6\_4), and the square-crossing lemma (Xiong\_10\_6\_5) into modular batches enabled parallel worker search followed by serial main-session integration into xiong\_jordan\_curve.

## 8 Related Work, Limitations, and Conclusion

Related work. Neural theorem provers have evolved from state-conditioned step generation [8, 10], informalto-formal sketching [9], and in-context proving agents [11] to reinforcement-learned whole-proof models [12, 13, 14] and general coding agents coupled with MCP tools such as Numina-Lean-Agent [2]. In Coq/Rocq, classical automation includes CoqHammer [15], Tactician [16], and Proverbot9001 [17], while recent LLM systems include PALM [18], CoqPilot [19], Rango [20], and Baldur [21] in Isabelle. Premise selection and agent reflection have been studied via dense retrievers [10, 22] and verbal reflection loops [23, 24]. Megrez complements these eforts by targeting cumulative textbook formalization over low-resource MK set theory, replacing black-box embeddings with an explicit, source-rebuildable knowledge graph and integrating short-burst proving, goal-local reflection, three-check acceptance, and recoverable batches.

Limitations and discussion. (1) Metadata and edge semantics: Retrieval quality depends on curated concepts and Chinese aliases, though reporting every exploratory Search fallback hit exposes missing annotations organically; compiler .glob edges record symbol occurrences rather than minimal logical dependencies, so soundness always rests on the Rocq kernel and rocq\_assumptions. (2) Candidate scope boundaries: CandidateScopeSnapshot enforces node-set membership prior to SQL lane caps rather than full Rocq mod ule/notation visibility, and full-text IDF statistics span all configured corpora; audit envelopes therefore explicitly record scope\_authority and all\_configured. (3) Trust boundaries: CAS revisions, capability hashes, and canonical proof-hash bindings prevent concurrent drift and proof swapping, whereas cooperative read-only workers share a workspace and do not replace OS-level sandboxing for untrusted processes.

Conclusion. We presented Megrez, a knowledge-guided formalization agent for Rocq libraries over Morse– Kelley set theory. By combining a source-rebuildable knowledge graph, a 1–5 tactic short-burst proving loop with exponential-interval goal-local reflection, and a three-check acceptance pipeline with recoverable multi-theorem batches, Megrez supports scalable retrieval and cross-chapter formalization across the 3,894 declarations and 102,122 proof lines of the MKTopos corpus, demonstrating a practical engineering path toward textbook-scale formal mathematics.

Code and corpus availability. The Megrez formalization agent and knowledge-graph toolchain are publicly available at https://github.com/dgjszy/Megrez-Rocq-Agent, and the accompanying point-set topology formalization corpus over Morse–Kelley set theory is available at https://github.com/dgjszy/RocqAtlas.

Acknowledgments. GPT 5.6 was used during this research to assist with system development, formalization exploration, and manuscript preparation.

## References

[1] The Rocq Development Team. The Rocq Prover Reference Manual, version 9.1. https://rocq-prover.org/doc/ v9.1/refman/index.html.

[2] Junqi Liu, Zihao Zhou, Zekai Zhu, Marco Dos Santos, Weikun He, Jiawei Liu, Ran Wang, Yunzhou Xie, Junqiao Zhao, Qiufeng Wang, Lihong Zhi, Jia Li, and Wenda Li. Numina-Lean-Agent: An open and general agentic reasoning system for formal mathematics. arXiv:2601.14027, 2026.

[3] Model Context Protocol. Specification, version 2025-11-25. https://modelcontextprotocol.io/specification/ 2025-11-25.

[4] John L. Kelley. General Topology, Appendix: Elementary Set Theory. Van Nostrand, 1955.

[5] Anthony P. Morse. A Theory of Sets. Academic Press, 1965.

[6] Tianyu Sun and Wensheng Yu. A formal system of axiomatic set theory in Coq. IEEE Access, 8:21510–21523, 2020.

[7] Jincheng Xiong. Lectures on Point-Set Topology (4th ed., in Chinese). Higher Education Press, 2011.

[8] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. arXiv:2009.03393, 2020.

[9] Albert Q. Jiang, Sean Welleck, Jin Peng Zhou, Wenda Li, Jiacheng Liu, Mateja Jamnik, Timothée Lacroix, Yuhuai Wu, and Guillaume Lample. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In ICLR, 2023.

[10] Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. LeanDojo: Theorem proving with retrieval-augmented language models. In NeurIPS Datasets and Benchmarks Track, 2023.

[11] Amitayush Thakur, George Tsoukalas, Yeming Wen, Jimmy Xin, and Swarat Chaudhuri. An in-context learning agent for formal theorem-proving. In COLM, 2024.

[12] Z. Z. Ren, Zhihong Shao, et al. DeepSeek-Prover-V2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv:2504.21801, 2025.

[13] Haiming Wang et al. Kimina-Prover Preview: Towards large formal reasoning models with reinforcement learning. arXiv:2504.11354, 2025.

[14] Yong Lin et al. Goedel-Prover-V2: Scaling formal theorem proving with scafolded data synthesis and self-correction. arXiv:2508.03613, 2025.

[15] Łukasz Czajka and Cezary Kaliszyk. Hammer for Coq: Automation for dependent type theory. Journal of Automated Reasoning, 61:423–453, 2018.

[16] Lasse Blaauwbroek, Josef Urban, and Herman Geuvers. The Tactician: A seamless, interactive tactic learner and prover for Coq. In CICM, 2020.

[17] Alex Sanchez-Stern, Yousef Alhessi, Lawrence Saul, and Sorin Lerner. Generating correctness proofs with neural networks. In MAPL, 2020.

[18] Minghai Lu, Benjamin Delaware, and Tianyi Zhang. Proof automation with large language models. In ASE, 2024.

[19] Andrei Kozyrev, Gleb Solovev, Nikita Khramov, and Anton Podkopaev. CoqPilot, a plugin for LLM-based generation of proofs. In ASE (Tool Demonstrations), 2024.

[20] Kyle Thompson, Nuno Saavedra, Pedro Carrott, Kevin Fisher, Alex Sanchez-Stern, Yuriy Brun, João F. Ferreira, Sorin Lerner, and Emily First. Rango: Adaptive retrieval-augmented proving for automated software verification. In ICSE, 2025.

[21] Emily First, Markus N. Rabe, Talia Ringer, and Yuriy Brun. Baldur: Whole-proof generation and repair with large language models. In ESEC/FSE, 2023.

[22] Maciej Mikuła, Szymon Tworkowski, Szymon Antoniak, Bartosz Piotrowski, Albert Q. Jiang, Jin Peng Zhou, Christian Szegedy, Łukasz Kuciński, Piotr Miłoś, and Yuhuai Wu. Magnushammer: A transformer-based approach to premise selection. In ICLR, 2024.

[23] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In ICLR, 2023.

[24] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In NeurIPS, 2023.