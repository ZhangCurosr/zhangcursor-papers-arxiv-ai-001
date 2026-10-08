# Does Document Structure Help Dense Retrieval? A Placebo-Controlled Ablation of Four Mechanisms Across Two Corpora

Andrey Kuehlkamp akuehlka@nd.edu

Priscila Correa Saboia Moreira Samuel Rund

pmoreira@nd.edu

srund@nd.edu

Center for Research Computing, University of Notre Dame Notre Dame, Indiana, USA

## Abstract

Retrieval-augmented generation systems increasingly rely on document-structure treatments: structure-aligned chunking, LLMgenerated chunk contexts, heading-path metadata, and hierarchical two-stage retrieval. Separate studies support each on different corpora, embedders, and metrics, and none control for a shared confound: any text prepended to a chunk perturbs its embedding.

We present a mechanism-isolating ablation testing all four treatments under one protocol, matching chunk sizes across conditions and adding a semantically null placebo—heading paths that are structurally valid but shuffled across documents. We score retrieval with a coverage-aware nDCG and test four pre-registered contrasts via document-clustered bootstrap with Holm correction, on two distant corpora: 200 Wikipedia Featured Articles (951 queries) and 1,585 QASPER papers (4,303 questions).

Organization helps, and the cause is content, not tokens: structure-aligned chunks with real heading paths beat contextualized fixed windows (+0.022 / +0.012 covnDCG@10) and the placebo (+0.010 / +0.016). Naive two-stage hierarchical retrieval hurts (-0.033 / -0.015), traceable to first-stage section recall. Gold structure beats LLM-induced structure on Wikipedia but not on QASPER. Effects are small (dz 0.06-0.11) but Holm-significant and consistent across corpora.

## 1 Introduction

Emerging systems that put large language models to work on the scientific literature begin by turning a flat PDF into a structured object — a hierarchy of sections and passages — before any downstream reading, extraction, or search. Alexandria (Moreira et al., 2026), for instance, decomposes each paper into pages, sections, and machine-readable markdown as a first step toward a provenancelinked, queryable record of its findings. Inducing that structure is not free, and it encodes an implicit bet: that document structure pays off downstream. For retrieval — the component that determines which passages such a system ever surfaces — that bet is largely untested. Does an induced hierarchy actually help retrieval, and if so, which mechanism carries the benefit?

Dense retrieval over chunked documents is the core component of most retrieval-augmented generation (RAG) systems. A growing literature argues that document structure — the boundaries, headings, and hierarchy an author placed — should inform how systems chunk, embed, and search documents. Four families of structurederived treatments each claim retrieval gains. (i) Structure-aligned chunking cuts at discourse boundaries instead of fixed token windows (Qu et al., 2025; Shaukat et al., 2026). (ii) Chunk contextualization prepends an LLM-written blurb that situates each chunk in its document (Anthropic, 2024). Late chunking and contextual embeddings are architectural variants (Günther et al., 2024; Morris and Rush, 2025). (iii) Structuremetadata injection prepends the chunk’s native heading path. (iv) Hierarchical retrieval routes queries to sections or tree nodes before it ranks chunks (Sarthi et al., 2024; Liu et al., 2021). Two problems keep these claims from being actionable. This paper attacks both at once.

The mechanism problem. The four treatments appear in separate papers, on different corpora, with different embedders and metrics. As a result, the field cannot say which mechanism carries the benefit. Worse, published results disagree in sign. Semantic-chunking gains fail to replicate across tasks (Qu et al., 2025). Contextualization helps in-corpus retrieval but hurts in-document retrieval (Zhou et al., 2026). Naive late chunking loses 21 nDCG points on an extractive benchmark (Conti et al., 2025). Flat retrieval matches or beats multistage retrieval in some studies (Laitenberger et al., 2025; Shah et al., 2026), while two-stage retrieval reports +12% top-1 elsewhere (Liu et al., 2021). A practitioner cannot tell whether to cut differently, prepend text, or restructure retrieval. The practitioner also cannot tell whether the reported gains are artifacts of corpus and embedder choice (Bhat et al., 2025).

The measurement problem. The treatments themselves confound the measurement. Any prefix — a situating context, a heading path — changes the chunk’s embedding. A gain over a bare baseline therefore never separates organizational content from generic token injection. No prior study we could verify runs a semantically null control. Structure-aligned chunks also differ from fixed windows in size distribution, which confounds boundary placement with chunk length (Bhat et al., 2025). Evidence for one query frequently spans multiple chunks, and binary relevance metrics score that incoherently. Also, the headline method in the contextualization family rests on a vendor self-benchmark, not on peerreviewed measurement (Anthropic, 2024).

We contribute a single controlled instrument that addresses both problems: an 11-condition ablation. The construction isolates every mechanism, and the design closes every confound we identified. We size-match the fixed-window baselines per document to the structure-aligned conditions. A placebo arm prepends heading paths that are structurally valid but shuffled across documents. The placebo has the same token count and format but null content, so we measure the token-injection confound instead of assuming it away. The primary metric, coverage-nDCG@10, extends nDCG with novelty-aware partial credit over multi-span evidence, in the lineage of α- nDCG (Clarke et al., 2008). We pre-registered four contrasts on one fixed cell (primary metric, strong embedder, no reranker). We test them with a document-clustered bootstrap under Holm correction. The full grid runs on two deliberately distant corpora: 200 Wikipedia Featured Articles with 951 leakage-filtered synthetic queries, and 1,585 QASPER papers with 4,303 native reader questions. Every mechanism claim therefore carries an explicit cross-corpus replication status instead of resting on a single measurement. All LLM stages (structure induction, contextualization, query generation) ran on a local Gemma-4- 26B model. The measured half of the pipeline is fully deterministic.

The findings, with their cross-corpus status:

1. Organization helps dense retrieval — small but real, on both corpora. Structure-aligned chunks with heading-path prefixes beat contextualized fixed windows by +0.022 covnDCG@10 on Wikipedia-FA and +0.012 on QASPER. Both gains are Holm-significant. The dominant share of the gain comes from where the text is cut, not from what is prepended. On both corpora, the move from fixed windows to bare structure-aligned chunks (A0→A2) is worth two to four times the marginal heading-path step (A2→A3).

2. The benefit is organizational content, not token injection. Real heading paths beat the placebo on both corpora (+0.010 Wikipedia-FA, +0.016 QASPER — the largest QASPER contrast). Shuffled paths buy nothing over no prefix at all, or actively hurt. To our knowledge, this is the first placebo-controlled measurement of structure-metadata injection.

3. LLM chunk contextualization underperforms a simple cut at structure. Contextualized fixed windows (the Anthropic-style arm) rank below bare structure-aligned chunks on both corpora. Under realistic token budgets, where prefix tokens count against the context window, the prepended contexts are strictly harmful.

4. Naive two-stage hierarchical retrieval hurts. Routing to sections before chunk ranking loses to flat search over the identical index on both corpora (-0.033, -0.015). Both losses are Holm-significant, and this is the only contrast the conservative secondary model also flags on Wikipedia-FA. The failure is diagnosable. First-stage section recall is 0.60-0.68 on the 200-document corpus and collapses to 0.17–0.20 on the 1,508-document corpus. On QASPER, a cross-encoder reranker recovers nearly all of the loss (-0.015 → -0.004).

5. The value of gold structure is corpusdependent. Gold heading trees beat LLMinduced trees on Wikipedia-FA (+0.013, Holm-significant) but not on QASPER (+0.001, null). On QASPER, induced structure is statistically as good as gold. We report both readings: the gold advantage does not generalize, and a 26B model’s induced structure loses nothing measurable on scientific papers.

The methodological contribution is inseparable from these results. Without the size control, finding (1) would be confounded. Without the placebo, finding (2) would be unmeasurable. Without cross-corpus replication, we would have reported the gold-structure advantage in finding (5) as general, and it is not. We offer the design as a template for mechanism claims in retrieval research. The study itself is evidence that the template changes conclusions.

## 2 Related Work

## 2.1 Chunking strategy

Fixed-size chunking fragments discourse (Merola and Singh, 2026). Structure-aligned alternatives can outperform it by large margins. A 36-method, 6-domain, 5-embedder comparison reports paragraph-group chunking at mean nDCG@5 ≈ 0.46, versus < 0.25 for fixed character windows (Shaukat et al., 2026). But the gains are unstable. Qu et al. (Qu et al., 2025) — the keystone non-replication result — find semantic chunking’s advantages inconsistent across document retrieval, evidence retrieval, and generation, and not worth the compute. A follow-up analysis attributes the measured gains chiefly to synthetic corpora stitched from unrelated documents (Zhou et al., 2026). Chunk-size effects also interact with task and embedder, and they flip sign between models (Bhat et al., 2025). This instability motivates our design rather than obstructing it. We hold corpus, embedder, and chunk size fixed, and we vary boundary placement alone (Section 3.2).

## 2.2 Contextualization and context-aware embeddings

Anthropic’s Contextual Retrieval (Anthropic, 2024) prepends a 50–100-token LLM-generated situating context to each chunk. It reports 35–49% reductions in top-20 retrieval failure. Those figures come from an internal, non-peer-reviewed benchmark, which is itself part of the gap we address. Architectural variants reach the same goal without generation. Late chunking pools token embeddings after a full-document encoder pass (Günther et al., 2024). Contextual document embeddings train the encoder to represent neighbors (Morris and Rush, 2025). The Con-TEB benchmark and InSeNT training target the setting directly (Conti et al., 2025). Yet added context is not reliably helpful. Late chunking drops 21 nDCG@10 on CovidQA (Conti et al., 2025). Anthropic-style contexts produce only marginal gains in independent evaluation (Merola and Singh, 2026). Contextualization helps incorpus retrieval while it hurts in-document retrieval (Zhou et al., 2026). Our A1 arm implements the Anthropic recipe verbatim: per-chunk LLM contexts on fixed windows. This fragile mechanism therefore competes directly against boundary placement and metadata injection under one metric.

## 2.3 Structure-metadata injection

Production RAG systems commonly prepend a chunk’s native heading path — the document title and section trail, essentially a breadcrumb. But we could not verify any study that isolates this practice as its own treatment, separate from boundary placement and from LLM-generated context. It is the least-studied of the four mechanisms. It is also the site of our strongest novelty claims: the isolated measurement (A2 vs. A3) and the placebo control (A3 vs. P) that separates path content from token injection.

## 2.4 Hierarchical retrieval

RAPTOR builds a synthetic summary tree by recursive clustering and retrieves across its levels (Sarthi et al., 2024). Note that RAPTOR constructs the tree — it does not use the document’s native hierarchy. Our G-conditions close that scope difference with real heading trees. Twostage retrieval helps in some settings. Dense Hierarchical Retrieval reports up to +12% absolute top-1 over flat DPR (Liu et al., 2021). But strong recent results go the other way. A flat retrievethen-read baseline that preserves document order matches or beats multi-stage systems (Laitenberger et al., 2025). Topology-aware analyses find that RAPTOR-style semantics-first retrieval loses to flat baselines on structural tasks (Shah et al., 2026). The hierarchy that does help is parent–child reranking, not route-to-section (Singh and Mohapatra, 2025). Our A4/G4 arms contribute a controlled data point to this debate. A diagnostic (first-stage section recall) localizes why routing loses (Section 6.4).

## 2.5 Evaluation methodology

Coverage- and diversity-aware effectiveness measures descend from α-nDCG (Clarke et al., 2008), which scores novel information rather than repeated relevance. Our coverage-nDCG applies the same principle at the character-span level, where one question’s evidence spans several chunks (Section 5.1). Placebo-style null controls are standard in medicine, and researchers increasingly urge them for ML ablations. None of the structure/context retrieval studies above includes one. Finally, non-replication is documented within this subfield (Qu et al., 2025; Zhou et al., 2026),. We respond in kind: replication across corpora is part of the result, not future work.

## 3 Study Design

## 3.1 Conditions and pre-registered contrasts

All conditions share one corpus, one embedder per run, and one retrieval budget. They differ only in (a) where chunk boundaries fall, (b) what prefix precedes the content at embedding time, and (c) whether retrieval is flat or two-stage. Chunks are contiguous, non-overlapping spans that exactly tile each document. The prefix, when present, renders as [Title › Section › ... ] \n\n cont Prefix tokens are tracked separately from content tokens throughout.

Figure 1 shows what each mechanism does to a document at embedding and retrieval time.

## 3.2 Controls and design invariants

We fixed four contrasts in the configuration before the runs. Each contrast isolates one mechanism question:

• C1 (organization vs. contextualization), A3- A1: do structure-aligned boundaries plus native metadata beat the strongest generative treatment on fixed windows?

• C2 (content vs. tokens), A3-P: does the content of the heading path matter, or is the gain token injection?

• C3 (cost of induction), G3-A3: how much is lost when an LLM induces the structure instead of the author?

• C4 (routing vs. flat), A4-A3: does two-stage hierarchical retrieval beat flat search over the identical index?

Table 4 in the appendix lists the experimental conditions employed in this study.

Size matching (guards C1). The fixed-window baseline is not cut at an arbitrary size. Per document, its target window equals the mean token length of that document’s induced chunks. Realized means differ by ≤ 4 tokens per chunkset (Wikipedia-FA: fixed 255.4 vs. induced 259.6 content tokens; QASPER: 244.9 vs. 249.8). The comparison of A0/A1 vs. A2/A3 therefore isolates boundary placement, not chunk size. Window edges snap to sentence boundaries. Structural chunks split (at sentence boundaries) or merge when they exceed 350 or fall below 60 content tokens.

Placebo construction (guards C2). Placebo paths are real heading paths from other documents. We bucket unique (document, path) pairs by depth and permute them within-bucket toward a cross-document derangement (fixed seed). We count and report residual same-document assignments. Placebo prefixes therefore match real breadcrumbs in format, depth, and length distribution, while they carry no information about the chunk. This is a semantically null treatment in the <sup>nt.</sup>strict sense.

Representation constancy (guards C4). The two-stage conditions build section centroids as the renormalized mean of the same chunk vectors that flat retrieval searches. C4 therefore compares retrieval algorithms, never representations. Stage 1 ranks section centroids and keeps the top 5 (a section = document + top-level heading). Stage 2 ranks chunks within the routed sections. If fewer than k candidates remain, the system backfills from the global ranking. A per-query sectionrecall diagnostic records whether stage 1 retained any section that overlaps the gold evidence. This separates routing failures from ranking failures.

Leakage and provenance controls. The pipeline selects the gold evidence spans. The query-writing LLM never self-reports them. Synthetic queries pass a two-round lexical-leakage filter (Section 5.1). Relevance is computed purely from character-span overlap against content-only offsets, so prefixes can never create spurious relevance.

![](images/048f5af4b37ee6b7bd4bce3a332a935c7a764c076abbc4d6f5efa417d33c4b8e.jpg)  
Figure 1: The four document-structure mechanisms under test. (a) Structure-aligned chunking cuts at section boundaries (A2) instead of ignoring them (A0). (b) Chunk contextualization prepends an LLM-generated blurb to a fixed-size window (A1). (c) Structure-metadata injection prepends the chunk’s real heading path (A3); the placebo (P) prepends an equally-shaped but shuffled, semantically null path. (d) Hierarchical retrieval routes queries to sections before ranking chunks (A4); flat retrieval (A3) searches the full chunk index directly. A4 underperforms because stage 1 frequently drops the gold section before stage 2 ever sees it (Section 6.4).

## 3.3 Structure induction

The induced-structure conditions (A2–A4, P) use a hierarchy from a local Gemma-4-26B-A4B model (the Mixture-of-Experts variant) (Gemma Team, Google DeepMind, 2026). The model sees the document’s paragraphs with all headings stripped. It returns sections as (title, level, paragraph-range) tuples under hard constraints: sections tile the paragraphs exactly, and levels increase by at most one. We verify the constraints mechanically. A failure gets one self-correction round with the specific violations, then a deterministic repair that snaps near-valid output to a legal tiling. On Wikipedia-FA, 198/200 documents produced usable structure (116 required repair; mean 21.9 sections/doc). On QASPER, 1,579/1,585 did (590 repaired; mean 14.5 sections/doc vs. 14.8 gold). One fact is critical for C3: the inducer does not simply recover native sectioning. Only 78.9% (Wikipedia-FA) and 75.8% (QASPER) of induced chunk spans are byte-identical to a gold span. Gold and induced conditions therefore measure genuinely different segmentations.

## 4 Corpora and Query Sets

We chose the two corpora to be maximally distant while both satisfy the protocol’s requirements: long documents with native hierarchical structure (so gold structure exists) and multi-span evidence (so coverage matters).

## 4.1 Wikiperia Featured Articles (200 documents, synthetic queries)

We sample documents from Wikipedia’s Featured Articles under quality filters (≥ 3, 000 words, heading depth ≥ 3, ≥ 12 sections) with topicbucket balancing <sup>1</sup>. A single canonical normalizer flattens the wikitext. Gold structure is the article’s own heading tree, offset-verified.

Queries are LLM-generated but provenanceand leakage-controlled. For each document, the pipeline itself samples gold spans and requests four query types. A factoid query targets one span. An underspecified query targets one span but names the referent only vaguely — "the commander," never the proper name. A multihop query targets two spans in different top-level sections. A structural query targets a whole toplevel section. Each generated query must pass a lexical-leakage filter: reject if ROUGE-L F1 vs. its gold span $> ~ 0 . 3 5$ or if any verbatim 4-gram is shared. A second LLM call then paraphrases the query. The paraphrase — which is what enters the benchmark — must pass the same filter again. The funnel: 2,306 generated → 1,945 survivors → 951 balanced finals (250 factoid, 250 underspecified, 250 structural, 201 multihop) over 184 documents. Shipped queries are lexically far from their evidence (typical ROUGE-L 0.02–0.13). BM25 places the gold chunk first for only 36.3% of queries, so the benchmark is not trivially lexical.

## 4.2 QASPER (1,585 papers, native questions)

QASPER (Dasigi et al., 2021) provides NLP papers with questions from real readers and gold evidence paragraphs — human-authored relevance on a genre far from encyclopedic prose. We merge all splits (train/val/test), because the study is an Information Retrieval (IR) ablation, not supervised learning. Gold structure is the paper’s own section tree. We harvest questions instead of generating them. We match each question’s evidence paragraphs to the flattened document (exact → fuzzy → section fallback; 95.7% exact). This yields $5 , 0 4 9  4 , 3 0 3$ usable questions over 1,508 papers, after we drop unanswerable, no-evidence, and numeric-only items. The set is hard and nonlexical (BM25 gold-top-1 22.7%). It is also genuinely multi-span: 43% of questions cite multiple evidence paragraphs (mean 2.10 spans/question). That is precisely the regime coverage-nDCG exists for.

One structural property matters when we interpret C3. QASPER’s native hierarchy is shallow: 61% of papers have depth-1 section lists, versus depth ≥ 3 for every Wikipedia-FA document by construction. Gold and induced trees therefore have less room to diverge (Section 6.5).

## 5 Evaluation Protocol

## 5.1 Metrics

Coverage-nDCG@10 (primary). Walking down the ranking, a chunk’s gain is the fraction of goldevidence characters it covers that earlier chunks did not already cover. The DCG of these marginal gains is normalized by the DCG of the ideal ordering achievable from that condition’s own chunk set. The novelty term prevents small-chunk conditions from banking the same span repeatedly. Per-condition normalization scores each condition against the best it could have done with its own segmentation, so a condition is not penalized merely for a different chunking. Both properties are required for fair comparison across chunkings. This is why binary nDCG is insufficient here.

Secondary metrics form a ladder from rankinsensitive to application-realistic: recall@50 over gold spans (≥ 50% covered by the union of top-50), Mean Reciprocal Rank (MRR), binary nDCG@10, and token-budget coverage at 512/1024/2048 tokens. In the token-budget metrics, prefix tokens count against the budget. Each treatment is charged its real context-window cost.

Relevance rule. A chunk is relevant to a span if and only if it covers $\geq 5 0 \%$ of the span, $\mathrm { o r } \geq 8 0 \%$ of the chunk lies inside the span. All overlap is computed on content-only character offsets.

## 5.2 Statistical protocol

We test all four contrasts on one cell fixed in advance — cov-nDCG@10, strong embedder, no reranker — to prevent metric shopping. Each contrast is the unweighted mean of paired perquery deltas. Multiple queries share a document, so we estimate uncertainty with a cluster bootstrap that resamples documents with replacement (10,000 iterations) (Field and Welsh, 2007). Pvalues are two-sided and Holm-corrected across the four contrasts (Holm, 1979). Effect sizes are Cohen’s dz (Lakens, 2013). Everything outside the primary cell is explicitly exploratory, and Section 6 labels it as such. This includes per-condition means across metrics and cells, the breadcrumbdepth dose-response, and per-query-type breakdowns.

We report a mixed-effects regression (score condition, random intercept per document) as a secondary check. We flag its specification honestly. It omits a per-query random effect although treatment varies within query. Query-level variance therefore inflates its standard errors, and the model is conservative for small paired effects. We report it because pre-registration said we would. Discordances are discussed where they occur (Section 6.1).

## 5.3 Implementation

Embedders are bge-small-en-v1.5 ("weak") and bge-large-en-v1.5 ("strong") (Xiao et al., 2024). Every run uses exactly one embedder. The pair exists to test robustness across encoder capacity, not as a two-stage system. The reranked cells apply bge-reranker-v2-m3 (Chen et al., 2024) over the top 50 candidates of the strong embedder. All generative stages (structure induction, A1 contexts, query generation and paraphrase) ran on local Gemma-4-26B-A4B via vLLM (Kwon et al., 2023) on a single H200 GPU. Generation is confined to one-time corpus and query construction under fixed seeds with on-disk caching. The measured half of the pipeline — embedding, retrieval, scoring, statistics — is deterministic. Code, configurations, and per-query results will be released.

## 6 Results

## 6.1 The four pre-registered contrasts

Table 1 reports the primary cell. Three contrasts replicate across both corpora with the same sign and Holm significance. One is corpus-dependent.

On Wikipedia-FA the secondary MixedLM flags only C4. This is the expected signature of its specification, not evidence that the other effects are fragile. The model lacks a query random effect, so it absorbs large query-to-query variance into its residual. It can therefore detect only the largest effect $( | d z | = 0 . 1 7$ , roughly twice the others). At QASPER’s n = 4, 303 queries, even this conservative model confirms all three non-null contrasts. The bootstrap is correctly specified for a paired, document-clustered design, and it is primary throughout.

Effect sizes are small: +0.01-0.02 covnDCG@10, dz 0.06-0.11. We ask the reader to hold both halves of that sentence. The effects are small, but they are also real: given tight CIs, Holm correction, and same-sign replication across two genres and two query provenances. Steady gains of this size are within the range IR practice treats as meaningful. In a subfield documented for non-replication, the contrasts’ cross-corpus consistency is itself a finding.

## 6.2 Where you cut matters more than what you prepend

Table 2 shows per-condition means decompose the organization effect (strong embedder, no reranker):

Two regularities hold on both corpora. First, the boundary step dominates the metadata step. A0→A2 — a re-cut of the same text at induced structural boundaries, with no prefix — is worth +0.041 on Wikipedia-FA and +0.019 on QASPER. A2→A3, the addition of the real heading path, adds only +0.011 and +0.009. Second, $_ \mathrm { A 1 ~ < }$ A2. LLM-written per-chunk contexts on fixed windows lose to bare structural cutting with no prefix at all. The generative treatment does not recover what bad boundaries destroy. Both regularities also hold on the weak embedder (Wikipedia-FA A0 0.306 → A3 0.343; QASPER 0.087 → 0.107). The ordering is therefore not an artifact of encoder capacity.

Contextualization’s cost becomes explicit under token budgets, where prefixes are charged to the context window. At a 512-token budget, A1 falls below plain A0 on both corpora (Wikipedia-FA 0.167 vs. 0.187; QASPER 0.046 vs. 0.052). A1 never clearly recovers by 2,048 tokens (0.396 vs. 0.397). An 80-token generated context that mostly restates document-level information displaces retrievable evidence.

## 6.3 Content, not tokens: the placebo verdict

The placebo arm exists to answer one question: is the A2→A3 gain merely "more tokens move the embedding somewhere useful"? On both corpora the answer is no. Real paths beat shuffled paths of identical format, depth, and length (+0.0099 Wikipedia-FA; +0.0161 QASPER, where C2 is the largest and most significant contrast). The placebo’s position against the bare structural arm tells the same story. Shuffled paths buy nothing over no prefix at all (Wikipedia-FA: P 0.403 vs. A2 0.403), or they actively hurt (QASPER: P 0.109 vs. A2 0.117). The full QASPER ordering is instructive: $\mathrm { A 0 ~ 0 . 0 9 8 < P ~ 0 . 1 0 9 < A 1 ~ 0 . 1 1 4 < }$ $\mathrm { A } 2 0 . 1 1 7 < \mathrm { A } 3 0 . 1 2 5$ . Null tokens buy a little over raw windows. Contextualization buys more. Bare structural cutting buys more still. Real metadata buys the most.

Without arm P, none of this would be attributable. A gain from any added prefix could be an embedding-geometry artifact, and the A2→A3 step would remain ambiguous between content and token injection. With arm P, the attribution is measured. The prefix’s value depends on its being true of the chunk. Semantically null tokens are at best neutral and at worst noise.

Table 1: Pre-registered contrasts, cov-nDCG@10, strong embedder, no reranker. Bootstrap CIs and Holmcorrected p-values. MixedLM (Seabold and Perktold, 2010; Seabold et al., 2025) is the conservative secondary model (See sec. 5.2).
<table><tr><td>Contrast</td><td>Corpus</td><td>Δ</td><td>95% CI</td><td>p(Holm)</td><td> $\pmb { d z }$ </td><td>p(MixedLM)</td></tr><tr><td rowspan="2">C1: A3-A1</td><td>Wikipedia-FA</td><td>+0.0218</td><td>[+0.0052, +0.0380]</td><td>0.017</td><td>+0.09</td><td>0.120</td></tr><tr><td>QASPER</td><td>+0.0115</td><td>[+0.0055, +0.0175]</td><td>&lt;0.001</td><td>+0.06</td><td>0.008</td></tr><tr><td rowspan="2">C2: A3-P</td><td>Wikipedia-FA</td><td>+0.0099</td><td>[+0.0024, +0.0173]</td><td>0.017</td><td>+0.08</td><td>0.487</td></tr><tr><td>QASPER</td><td>+0.0161</td><td>[+0.0114, +0.0207]</td><td>&lt;0.001</td><td>+0.11</td><td>&lt;0.001</td></tr><tr><td rowspan="2">C3: G3-A3</td><td>Wikipedia-FA</td><td>+0.0129</td><td>[+0.0038, +0.0228]</td><td>0.010</td><td>+0.10</td><td>0.373</td></tr><tr><td>QASPER</td><td>+0.0011</td><td>[-0.0020, +0.0043]</td><td>0.485 (n.s.)</td><td>+0.01</td><td>0.801</td></tr><tr><td rowspan="2">C4: A4-A3</td><td>Wikipedia-FA</td><td>-0.0326</td><td>[-0.0442, -0.0217]</td><td>&lt;0.001</td><td>-0.17</td><td>0.027</td></tr><tr><td>QASPER</td><td>-0.0147</td><td>[-0.0192, -0.0104]</td><td>&lt;0.001</td><td>-0.11</td><td>&lt;0.001</td></tr></table>

Table 2: Per-condition cov-nDCG@10 (strong embedder, no reranker).
<table><tr><td>Corpus</td><td>A0</td><td>A1</td><td>A2</td><td>A3</td><td>P</td><td>A4</td><td>G2</td><td>G3</td><td>G4</td></tr><tr><td>Wikipedia-FA</td><td>0.362</td><td>0.391</td><td>0.403</td><td>0.413</td><td>0.403</td><td>0.381</td><td>0.417</td><td>0.426</td><td>0.400</td></tr><tr><td>QASPER</td><td>0.098</td><td>0.114</td><td>0.117</td><td>0.125</td><td>0.109</td><td>0.111</td><td>0.120</td><td>0.126</td><td>0.111</td></tr></table>

## 6.4 Naive hierarchical retrieval hurts, and we can say why

C4 is negative on both corpora. It is the most robust single effect in the study (Wikipedia-FA dz = −0.17). The two-stage conditions retrieve over the identical vectors as A3, so the penalty is attributable entirely to routing. The section-recall diagnostic localizes it. The probability that stage 1 even retains a gold-bearing section is only 0.60–0.68 on the 200-document Wikipedia-FA corpus. It collapses to 0.17–0.20 on QASPER, where the router faces ≈ 22, 000 candidate sections. Whatever stage 1 drops, stage 2 cannot recover. Recall@50 shows the same shadow (QASPER: A4 0.293 vs. A3 0.302; Wikipedia-FA: 0.675 vs. 0.730). The penalty scales with the routing failure. Proportionally, the loss is ≈ 8% of A3’s score on Wikipedia-FA and ≈ 12% on QASPER — larger where section recall is worse — even though QASPER’s absolute delta is smaller (its whole scoreboard is ≈ 3× lower).

One qualification keeps this negative honest: it is largely a no-reranker phenomenon. Under bgereranker-v2-m3, the QASPER gap shrinks from -0.0147 to -0.0038 (Wikipedia-FA: -0.0326 → - 0.0273). The reranker re-scores a wide flat candidate list that routing would otherwise have pruned. The correct reading is therefore narrow. Route-tosection-then-rank under-recalls as the section inventory grows. It loses to flat search over the same index unless a reranker restores the candidates. This is a result about one widely-used naive architecture, not about hierarchy in general. It is consistent with the literature’s split verdict (Laitenberger et al., 2025; Liu et al., 2021; Singh and Mohapatra, 2025).

## 6.5 Gold vs. induced structure is corpus-dependent

On Wikipedia-FA, native heading trees beat Gemma’s induced trees by +0.0129 (Holmsignificant). Imperfect induction has a small, measurable cost. On QASPER the advantage vanishes (+0.0011, n.s.). This is not because the inducer copied the papers’ sectioning: span identity is 75.8%, comparable to Wikipedia’s 78.9%. Both readings are findings. For the gold-structure story: the advantage of native structure does not generalize across genres, and no one should claim it corpus-generally. For the induction story: a 26B open-weights model re-segments a quarter of all spans and loses nothing measurable against author-written structure on scientific papers. A plausible mechanism — untested, offered as hypothesis — is depth headroom. 61% of QASPER papers are depth-1, so gold and induced trees have little room to differ in the information the breadcrumb carries. Wikipedia-FA’s ≥ 3-deep trees give gold structure more leverage.

## 6.6 Exploratory analyses

Breadcrumb depth dose-response. We regress cov-nDCG@10 on injected path depth (0→3, i.e. $\mathrm { A } 2 \  \ \mathrm { A } 3 \mathrm { - } \mathrm { d } 1 \  \ \mathrm { A } 3 \mathrm { - } \mathrm { d } 2 \  \ \mathrm { A } 3 )$ The strong embedder shows a small positive slope on both corpora (Wikipedia-FA +0.0032/level, bootstrap p = 0.02; QASPER +0.0027/level, p < 0.001). The weak embedder is flat on Wikipedia-FA (+0.0006/level, p = 0.65) and marginal on QASPER (+0.0012/level, p = 0.052). In every cell, most of the benefit arrives at the first path level and then plateaus (e.g., Wikipedia-FA strong: $0 . 4 0 3  0 . 4 1 2  0 . 4 1 3  0 . 4 1 3 )$ . We do not promote this to a finding. It sits outside the preregistered family. The weak-embedder slopes disagree across corpora. QASPER’s shallow native depth also partly collapses the depth arms by construction.

Per-query-type cuts. On Wikipedia-FA, the C1 gain concentrates on factoid (+0.046) and multihop (+0.031) queries. It is absent on structural queries (-0.002). C2 is largest on structural queries (+0.023). On QASPER, C1, C2, and C4 hold their signs across all three native question types (extractive/abstractive/boolean). The two corpora use incommensurable query taxonomies, so no specific type interaction can be called replicated. We report these cuts as hypothesis-generating only.

Reranking lifts all boats. A cross-encoder adds +0.04-0.06 cov-nDCG to essentially every condition on both corpora. It leaves the organization contrasts intact (Wikipedia-FA A3: 0.413 → 0.474; QASPER: 0.125 → 0.165). It disproportionately rescues only A4 (Section 6.4).

Recall, not just ranking. The contrasts partly live in recall. On QASPER, strongembedder recall@50 spans 0.248 (A0) to 0.302 (A3). Structure-aligned chunking retrieves more of the evidence, not merely in better order. Absolute recall (≈ 0.25˘0.30) shows this benchmark is far from saturated.

## 7 Discussion

A mechanism ranking, for one well-specified regime. For dense retrieval over long structured documents with strong open-weights embedders, the mechanisms rank as follows. Boundary placement comes first, with the largest and most reliable gain. Native heading-path metadata comes second: small, real, and content-driven. LLM per-chunk contextualization comes third: positive against raw windows, negative against structural cutting, and budget-expensive. Naive two-stage routing comes last: harmful without a reranker. For a practitioner, the actionable form is:

1. Cut where the author cut.

2. Prepend one level of real heading path.

3. Spend the LLM budget you would spend on per-chunk contexts on structure induction instead. Induction needs only 26B-class capability and matched gold structure on one of two corpora.

4. Do not route-to-section without a reranker behind it.

What the controls changed. Each methodological element did visible work. The size control is why the largest effect in the study (A0→A2) is attributable to boundary placement rather than chunk length. The placebo is why the metadata effect is attributable to content rather than token injection. The placebo also caught the null case: on QASPER, semantically null paths actively hurt relative to no prefix, an outcome no bare-baseline comparison could surface. The fixed primary cell is why the depth dose-response and query-type patterns remain labeled exploratory instead of headline claims. Cross-corpus replication demoted one of the three Wikipedia-FA "organization" pillars (C3) to a boundary condition. With one corpus we would have gotten that conclusion wrong, in a literature where single-corpus structure claims are the norm. We take this as direct evidence for the paper’s methodological thesis. In a subfield with documented non-replication, mechanism claims need controls of roughly this strength before they are worth acting on.

Reconciling the literature. Our results locate several published disagreements. The chunking literature’s instability (Qu et al., 2025) is consistent with our small effect sizes. Gains of +0.01–0.02 under tight controls will oscillate through zero in uncontrolled comparisons. Contextualization’s mixed record (Merola and Singh, 2026; Conti et al., 2025) matches A1’s position: real against raw windows, dominated by structural cutting. The hierarchical-retrieval split (Liu et al., 2021) vs. (Laitenberger et al., 2025) resolves, in our data, into a statement about first-stage recall as a function of section inventory. That statement predicts both the successes (small inventories, strong routers) and the failures (large inventories, centroid routers).

## 8 Limitations

Small effects. The replicated organization effects are +0.01-0.02 cov-nDCG@10 $( d z \leq 0 . 1 1 )$ . Applications where retrieval quality is not the bottleneck will not notice them. We claim reliability, not magnitude.

One inducer. Both corpora share one structureinduction model (Gemma-4-26B-A4B). Induction quality plausibly varies with the inducing model. That would move C3 in particular, whose corpusdependence already shows the gold/induced gap is sensitive to conditions. A second-inducer replication is the clear next experiment.

Two corpora, one language. Wikipedia-FA and QASPER differ usefully in genre, query provenance, and structure depth. But both are English and expository. The C3 reversal between them is itself a warning against generalization of any structure result from ≤ 2 corpora.

Synthetic queries on one corpus. Wikipedia-FA queries are LLM-generated, though leakagefiltered, paraphrased, and provenance-controlled. QASPER’s native questions mitigate but do not eliminate the concern, because the corpora differ in more than query provenance.

Secondary model specification. The MixedLM omits a query random effect in a within-query paired design. We report it for pre-registration fidelity and treat the documentclustered bootstrap as primary. A qid-randomeffect respecification is planned.

One routing architecture. C4 indicts centroidbased route-to-section-then-rank with top-5 sections — the naive but common design. It does not indict hierarchical retrieval broadly. Parent–child reranking schemes (Singh and Mohapatra, 2025) are untested here.

Retrieval only. No answer-generation stage is evaluated. Whether +0.01–0.02 cov-nDCG@10 moves downstream answer quality is out of scope.

## 9 Conclusion

We asked which of four document-structure mechanisms actually helps dense retrieval. We built the controls the answer needs to be trustworthy: size-matched baselines, a semantically null placebo, a coverage-aware primary metric, preregistered contrasts under clustered inference, and replication across two distant corpora with different query provenances. The answer is consistent across both corpora. Organization helps, modestly. Boundary placement helps most. The content of native heading paths helps secondarily. LLM chunk contextualization underperforms structural cutting. Naive hierarchical routing hurts in proportion to its first-stage recall failure. The value of gold over induced structure, by contrast, did not survive the corpus change. The analyses we left uncontrolled — the depth dose-response and the query-type interactions — are precisely the ones that remain unsettled. Both facts argue the same point. In retrieval-mechanism research, the framing choice between methodology and mechanism is false. The mechanisms can only be seen through the methodology.

## Ethics and Privacy Statement

This study evaluates retrieval only. It runs no answer-generation stage and makes no deployment claim. Both corpora are public research datasets: Wikipedia Featured Articles and QASPER (Dasigi et al., 2021). Neither contains personal data collected for this work, and the study involved no human subjects. All LLM stages reported in this paper ran on a local open-weights model.

The findings are small in magnitude, and we report them as such. The practical risk we see is overreading: a +0.01 to +0.02 cov-nDCG@10 gain is not a license to spend LLM inference on structure induction without measuring the downstream effect on a target corpus. The negative result carries the same caution in reverse. It indicts one common routing design, not hierarchical retrieval as a family.

## References

Anthropic. 2024. Introducing contextual retrieval. https://www.anthropic.com/ news/contextual-retrieval. Accessed: 2026-07-31.

Sinchana Ramakanth Bhat, Max Rudat, Jannis Spiekermann, and Nicolas Flores-Herr. 2025. Rethinking chunk size for long-document retrieval: A multi-dataset analysis.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Charles L.A. Clarke, Maheedhar Kolla, Gordon V. Cormack, Olga Vechtomova, Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and diversity in information retrieval evaluation. In Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SI-GIR ’08, page 659–666, New York, NY, USA. Association for Computing Machinery.

Max Conti, Manuel Faysse, Gautier Viaud, Antoine Bosselut, Céline Hudelot, and Pierre Colombo. 2025. Context is gold to find the gold passage: Evaluating and training contextual document embeddings. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 22594–22608, Suzhou, China. Association for Computational Linguistics.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 4599–4610. Association for Computational Linguistics.

Christopher A. Field and Alan H. Welsh. 2007. Bootstrapping clustered data. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(3):369–390.

Gemma Team, Google DeepMind. 2026. Gemma 4 technical report.

Michael Günther, Isabelle Mohr, Daniel J Williams, Bo Wang, and Han Xiao. 2024. Late chunking: Contextual chunk embeddings using long-context embedding models.

Sture Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal ofStatistics, 6(2):65–70.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP ’23), pages 611–626, New York, NY, USA. Association for Computing Machinery.

Alex Laitenberger, Christopher D. Manning, and Nelson F. Liu. 2025. Stronger baselines for retrieval-augmented generation with longcontext language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 32559–32569, Suzhou, China. Association for Computational Linguistics.

Daniël Lakens. 2013. Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs. Frontiers in Psychology, 4:863.

Ye Liu, Kazuma Hashimoto, Yingbo Zhou, Semih Yavuz, Caiming Xiong, and Philip Yu. 2021. Dense hierarchical retrieval for open-domain question answering. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 188–200, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Carlo Merola and Jaspinder Singh. 2026. Reconstructing context: Evaluating advanced chunking strategies for retrieval-augmented generation. In Knowledge-Enhanced Information Retrieval (KEIR 2025), volume 16086 of Lecture Notes in Computer Science, pages 3–18, Cham. Springer.

Priscila Saboia Moreira, Faith Wangui, Don Brower, Andrey Kuehlkamp, and Samuel S.C. Rund. 2026. Alexandria: Verification-first science gateway for curating data from the scientific literature. In Proceedings of the 2026 Improving Scientific Software Conference / Gateways 2026, Washington, D.C., USA. Association for Computing Machinery. To appear.

John X. Morris and Alexander M. Rush. 2025. Contextual document embeddings. In The Thirteenth International Conference on Learning Representations (ICLR).

Renyi Qu, Ruixuan Tu, and Forrest Sheng Bao. 2025. Is semantic chunking worth the computational cost? In Findings of the Association for Computational Linguistics: NAACL 2025, pages 2155–2177, Albuquerque, New Mexico. Association for Computational Linguistics.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations (ICLR).

Skipper Seabold and Josef Perktold. 2010. statsmodels: Econometric and statistical modeling with python. In Proceedings of the 9th Python in Science Conference (SciPy 2010), pages 92–96.

Skipper Seabold, Josef Perktold, and statsmodels contributors. 2025. statsmodels: statistical modeling and econometrics in python.

Deep Shah, Sanket Badhe, and Nehal Kathrotia. 2026. Taxonomy of the retrieval system framework: Pitfalls and paradigms.

Muhammad Arslan Shaukat, Muntasir Adnan, and Carlos C. N. Kuhn. 2026. A systematic investigation of document chunking strategies and embedding sensitivity.

Ashish Singh and Priti Mohapatra. 2025. Hierarchical re-ranker retriever (hrr).

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), pages 641–649, New York, NY, USA. Association for Computing Machinery.

Yongjie Zhou, Shuai Wang, Bevan Koopman, and Guido Zuccon. 2026. Beyond chunk-thenembed: A comprehensive taxonomy and evaluation of document chunking strategies for information retrieval.

This appendix documents the experimental instrument in full. Section A names the eleven conditions and the grid that holds them. Sections B and C define the primary metric and its relatives. Section D states the statistical protocol as the code runs it. Sections E to G build both corpora and state the ground-truth checks. Section H covers structure induction. Section I covers the non-lexical query sets. Section J covers the reproducibility envelope. When this appendix reports an implementation detail, it names the module and the behavior. The released code is the final authority.

## A Conditions, naming conventions, and the experimental grid

## A.1 The two condition families

Every condition is one point in a design that varies exactly three things. First, where the chunk boundaries fall. Second, what prefix precedes the chunk content at embedding time. Third, whether retrieval is flat or two-stage. Chunks are contiguous, non-overlapping spans that tile each document exactly. One run holds all other factors fixed: corpus, embedder, and retrieval budget.

Condition names encode two axes. The letter encodes the source of structure:

• A means induced structure. The local Gemma-4-26B-A4B model produces the section tree after the pipeline strips the native headings (See appendix H).

• G means gold structure. The section tree is the document’s own author-written heading tree, offset-verified (See appendix E.2).

The number encodes the treatment level. The numbering stays parallel between the two families. A G-condition is therefore the exact counterpart of the A-condition with the same number. The two differ only in gold structure versus induced structure:

There is no G0 or G1, by design. Levels 0 and 1 use fixed windows and therefore have no structure. "Gold versus induced structure" is undefined for them. The G family exists only where a structural tree is in use (levels 2 to 4).

## A.2 The placebo and the depth ladder

Two conditions extend the induced family for control purposes:

• P (placebo) matches A3 (induced structure, depth-3 heading-path prefix, flat). It differs in one way: each chunk’s prefix is a real heading path from a different document. The prefix matches the real ones in format, depth, and length distribution. It carries no information about the chunk. P isolates the tokeninjection confound (main text 3.2; construction in E.3 here).

• A3-d1 / A3-d2 / A3 form the breadcrumb depth ladder. The -dN suffix means "heading path cut to depth N." The bare name A3 is the full-depth (depth-3) rung. The "3" in "A3" is the treatment-level index. It coincides with the depth, which is why the name is not A3-d3. Read as a dose-response series, the ladder is A2 (depth 0, no prefix), then A3-d1, then A3-d2, then A3 (depth 3). The prefix renders as [Title › Section › ... ] \n\n content. The separator is ›.

## A.3 The eleven conditions

## A.4 Contrasts, and the cell that tests them

Each pre-registered contrast is a single subtraction that isolates one mechanism:

A cell is one fixed combination of the noncondition factors: one metric, one embedder, one reranker setting, one corpus. The full experiment is a grid over those factors. The conditions above re-run in each cell. One pre-registered primary cell gives the four contrasts their confirmatory test, which forecloses metric-shopping: cov-nDCG@10 × strong embedder (bge-large) × no reranker. Every other cell is exploratory, and main-text 6 labels it as such. This covers other metrics, the weak embedder, and the reranked variants. The grid axes are these. Metric is one of cov-nDCG@10, recall@50, MRR, binary-nDCG@10, or token-budget coverage at 512/1024/2048. Embedder is the weak bge-smallen-v1.5 or the strong bge-large-en-v1.5. Reranker is none or bge-reranker-v2-m3 over the top 50. Corpus is Wikipedia-FA or QASPER.

## B Coverage-nDCG@10: formal definition

## B.1 Motivation

The evidence for a single query often spreads across several disjoint character spans. On

Table 3: Condition levels.
<table><tr><td>Level</td><td>Meaning</td><td>Induced</td><td>Gold</td></tr><tr><td>0</td><td>fixed-window chunks (size-matched), no prefix</td><td>A0</td><td></td></tr><tr><td>1</td><td>fixed-window chunks + LLM situating context (≤80 tok)</td><td>A1</td><td></td></tr><tr><td>2</td><td>structural chunks, no prefix</td><td>A2</td><td>G2</td></tr><tr><td>3</td><td>structural chunks + heading-path prefix (depth 3)</td><td>A3</td><td>G3</td></tr><tr><td>4</td><td>structural chunks + heading-path prefix + two-stage routing</td><td>A4</td><td>G4</td></tr></table>

<table><tr><td>Condition</td><td>Boundaries</td><td>Prefix</td><td>Retrieval</td><td>Structure</td></tr><tr><td>A0</td><td>fixed windows (size-matched)</td><td>none</td><td>flat</td><td></td></tr><tr><td>A1</td><td>fixed windows (size-matched)</td><td>LLM situating context (≤80 tok)</td><td>flat</td><td></td></tr><tr><td>A2</td><td>induced structure</td><td>none</td><td>flat</td><td>induced</td></tr><tr><td>A3-d1</td><td>induced structure</td><td>heading path, depth 1</td><td>flat</td><td>induced</td></tr><tr><td>A3-d2</td><td>induced structure</td><td>heading path, depth 2</td><td>flat</td><td>induced</td></tr><tr><td>A3</td><td>induced structure</td><td>heading path, depth 3</td><td>flat</td><td>induced</td></tr><tr><td>P</td><td>induced structure</td><td>shuffled heading path, depth 3</td><td>flat</td><td>induced</td></tr><tr><td>A4</td><td>induced structure</td><td>heading path, depth 3</td><td>two-stage</td><td>induced</td></tr><tr><td>G2</td><td>gold structure</td><td>none</td><td>flat</td><td>gold</td></tr><tr><td>G3</td><td>gold structure</td><td>heading path, depth 3</td><td>flat</td><td>gold</td></tr><tr><td>G4</td><td>gold structure</td><td>heading path, depth 3</td><td>two-stage</td><td>gold</td></tr></table>

Table 4: Experimental conditions.
<table><tr><td>Contrast</td><td>Subtraction</td><td>Question</td></tr><tr><td>C1</td><td>A3 - A1</td><td>Do structure-aligned boundaries plus native metadata beat the strongest generative treatment on fixed windows?</td></tr><tr><td>C2</td><td>A3-P</td><td>Does the content of the heading path matter, or is the gain token injection?</td></tr><tr><td>C3</td><td>G3- A3</td><td>How much is lost when an LLM induces the structure instead of the author?</td></tr><tr><td>C4</td><td>A4 - A3</td><td>Does two-stage hierarchical retrieval beat flat search over the identical index?</td></tr></table>

Table 5: Pre-registered contrasts.

QASPER, 43.9% of questions cite multiple evidence paragraphs (mean 2.10 spans/question). The primary metric therefore needs two properties. Binary nDCG provides neither:

1. Novelty (coverage) awareness. A chunk that repeats evidence already retrieved higher in the ranking should earn little. Credit accrues for new gold coverage only. Without this property, a condition that cuts text into many small chunks banks the same span again and again.

2. Per-condition ideal normalization. A condition is scored against the best ranking that its own chunk set allows. A condition is therefore not penalized for segmenting the document differently from another condition.

Coverage-nDCG applies the novelty principle of α-nDCG (Clarke et al., 2008) one level below subtopic diversity. It applies it at the level of goldevidence characters within a single query.

## B.2 Definition

Let a query belong to document d with goldevidence spans $G ~ = ~ \{ g _ { 1 } , . . . , g _ { S } \}$ . Each $g _ { j } =$ $\left[ \mathrm { s t a r t } _ { j } , \mathrm { e n d } _ { j } \right)$ is a half-open character interval on d’s flat text, and $| g _ { j } |$ is its length. A retrieval run returns a ranked list of chunks $c _ { 1 } , c _ { 2 } , \ldots .$ . A chunk from any document other than d contributes zero, because offsets are per-document coordinate spaces.

For each gold span $g _ { j }$ , keep the set of its characters already covered by chunks ranked above the current one. For the chunk $c _ { i }$ at rank i, the marginal gain is the mean over gold spans of the fraction of newly covered characters:

$$
\mathrm { g a i n } ( c _ { i } ) = \frac { 1 } { S } \sum _ { j = 1 } ^ { S } \frac { \mid ( c _ { i } \cap g _ { j } ) \setminus \mathrm { c o v e r e d } _ { < i } ( g _ { j } ) \mid } { | g _ { j } | }
$$

Here covered ${ < i ( g _ { j } ) }$ is the union of the portions of $g _ { j }$ covered by $c _ { 1 } , \ldots , c _ { i - 1 }$ . After the metric scores $c _ { i } .$ , it adds the overlap of $c _ { i }$ with each $g _ { j }$ to covered $( g _ { j } )$ .

The discounted cumulative gain at cutoff k uses the standard log discount:

$$
\operatorname { D C G @ } k \ = \ \sum _ { i = 1 } ^ { k } { \frac { \operatorname { g a i n } ( c _ { i } ) } { \log _ { 2 } ( i + 1 ) } }
$$

The ideal DCG runs over the same condition’s own chunk set by a greedy maximum-marginalcoverage ordering. It repeatedly appends the not-yet-selected chunk with the largest current marginal gain, up to k chunks. This greedy order is the standard tractable stand-in for the ideal ranking under a coverage objective. Finally,

$$
\operatorname { c o v - n D C G @ } k \ = \ { \frac { \operatorname { D C G @ } k } { \operatorname { I D C G @ } k } } , \qquad k = 1 0 .
$$

The value is 0 when IDCG@k = 0. That case means no chunk of the condition covers any gold character.

## B.3 Worked micro-example

Suppose one query has a single gold span $g _ { 1 }$ of length 100 characters. Condition X returns, in rank order, chunks that cover characters [0, 60), then [40, 100), then an irrelevant chunk:

• Rank 1 covers 60 new chars. The gain is $6 0 / 1 0 0 = 0 . 6 0$ . The discount is $\log _ { 2 } 2 = 1$

• Rank 2 covers [40, 100), but [40, 60) is already covered, so 40 new chars. The gain is 0.40. The discount is $\log _ { 2 } 3 \approx 1 . 5 8 5$

• Rank 3 covers 0 new chars. The gain is 0.

The result is $\mathrm { D C G } = 0 . 6 0 / 1 + 0 . 4 0 / 1 . 5 8 5 =$ 0.852. The greedy ideal from this chunk set places the same two chunks first, because their gains are identical. So $\mathrm { { I D C G } ~ = ~ 0 . 8 5 2 }$ and $\mathrm { c o v - n D C G = 1 . 0 }$ . Suppose instead a third chunk covered [0, 100) in one shot. The ideal would rank it first (gain 1.0 at rank 1), and the observed ranking would score below 1.0. The example shows the two active ingredients. The rank-2 chunk is discounted for redundancy (40 new chars, not 60). The score is relative to the best that the condition’s own chunks allow.

## C Relevance rule and secondary metrics

Coverage-nDCG uses graded character coverage and never thresholds. The binary relevance rule below serves only the secondary metrics that need a yes/no judgment.

Binary relevance. A chunk c is relevant to a gold span g if

$$
{ \frac { \left| c \cap g \right| } { \left| g \right| } } \geq 0 . 5 \quad { \mathrm { ( t h e ~ c h u n k ~ s u b s t a n t i a l l y ~ c o n t a i n s ~ t h e ~ e v i d e n c e ) } }
$$

$\frac { | c \cap g | } { | c | } \geq 0 . 8$ (the chunk lies substantially inside the evidence).

All overlap runs on content-only character offsets. A prefix therefore can never create spurious relevance.

Secondary metrics form a ladder from rankinsensitive to application-realistic:

• recall@50 is the fraction of gold spans that the union of the top-50 chunks covers by $\geq$ 50%.

• MRR is the reciprocal rank of the first binary-relevant chunk.

• binary-nDCG@10 is nDCG with 0/1 gains under the binary rule. It normalizes by the ideal binary ranking of the condition’s own chunk set.

• token-budget coverage at 512/1024/2048 tokens is the gold coverage that top-ranked chunks reach until the budget runs out. Prefix tokens count against the budget. Each treatment therefore pays its real contextwindow cost. This is where contextualization’s cost becomes explicit. Main-text 6.2 shows that at 512 tokens A1 falls below plain A0 on both corpora.

## D Statistical protocol

## D.1 Paired per-query deltas

Each contrast runs on paired per-query deltas. For every query answered under both conditions, $\delta _ { q } = \mathsf { s c o r e } _ { A } ( q ) - \mathsf { s c o r e } _ { B } ( q )$ . The reported effect is the unweighted mean $\bar { \delta }$ over queries. Pairing is the correct design, because every query is scored under both arms. Pairing removes the large queryto-query difficulty variance from the comparison.

## D.2 Cluster bootstrap (primary)

Queries are not independent. Several queries share a document and move together. The protocol therefore estimates uncertainty with a documentclustered bootstrap, not an ordinary per-query bootstrap. The difference is the resampling unit:

• Group the per-query deltas by their document.

• For each of $B = 1 0 , 0 0 0$ iterations, resample documents with replacement. Every query of a drawn document enters the replicate as a bundle.

• Recompute the mean delta on each replicate: $\begin{array} { r } { \bar { \delta } ^ { ( b ) } = \big ( \sum _ { d \in \mathrm { d r a w } } \mathrm { s u m } _ { d } \big ) / \big ( \sum _ { d \in \mathrm { d r a w } } \mathrm { c o u n t } _ { d } \big ) } \end{array}$

The bootstrap distribution gives three quantities:

• The 95% CI is the 2.5th and 97.5th percentiles of $\{ \bar { \delta } ^ { ( b ) } \}$

• The two-sided p-value is 2 · min $( \mathrm { P r } [ \bar { \delta } ^ { ( b ) } \leq$ 0], $\operatorname* { P r } [ \bar { \delta } ^ { ( b ) } \geq 0 ] )$ , clipped to 1.

• The \*\*effect size\*\* is Cohen’s $\begin{array} { r l } { d z } & { { } = } \end{array}$ $\textstyle { \bar { \delta } } / \operatorname { s d } ( \delta _ { q } )$ . The standard deviation runs over the per-query deltas (ddof = 1).

Resampling documents rather than queries widens the intervals. The wider intervals reflect the true number of independent units, which is documents, not queries. Take QASPER’s design: 4,303 queries over 1,508 documents. An ordinary bootstrap would treat the effective sample size as about 3× too large and understate uncertainty. The cluster bootstrap is the correctly specified test, and it is primary throughout.

## D.3 Holm correction

The protocol tests four contrasts as a family. It therefore adjusts the per-test p-values by the Holm step-down procedure. Sort the four p-values in ascending order. Inflate the i-th smallest (0-indexed) by the factor $( m - i )$ with $m = 4$ , under a running maximum so the adjusted values stay monotone.

Clip to 1. Holm controls the family-wise error rate at 0.05. It rejects more true effects than the blunt Bonferroni factor of 4.

## D.4 MixedLM (secondary) and its known misspecification

The protocol reports a mixed-effects regression as a pre-registered secondary check. This model is misspecified for the design, by design, and the paper discloses this. The study is paired within query, because the same query appears under both conditions. But the model has no per-query random effect. The large query-to-query baseline variance therefore lands in the residual and inflates the standard error of the treatment coefficient. The model is therefore conservative for small paired effects. It resolves only the largest effect. This matches the observed pattern. On Wikipedia-FA $( n ~ \approx ~ 9 5 1$ per condition) it flags only C4 $( | d z | = 0 . 1 7$ , roughly twice the others) and misses C1/C2/C3 $( d z \approx 0 . 0 8 ^ { \smile } 0 . 1 0 )$ . At $\mathrm { Q A S P E R " s ~ n = }$ 4,303 the larger sample overwhelms the inflated errors, and it confirms all three non-null contrasts.

The disagreements between the bootstrap and the MixedLM on the small Wikipedia-FA effects are therefore a specification artifact of the secondary model, not evidence that the effects are fragile. The bootstrap is correctly specified and primary. The paper reports the MixedLM for preregistration fidelity only.

## E Wikipedia Featured Articles: corpus construction

The pipeline samples documents from Wikipedia’s Featured Articles under hard quality filters, with a fixed RNG seed $( \mathsf { s e e d } \ = \ 2 0 2 6 0 7 1 0 )$ :

The pipeline fetches the FA title list, shuffles it under the seed, and walks it. It fetches each candidate, flattens it with the single canonical normalizer, and accepts it if it passes the filters and the per-topic cap (E.1). The loop stops at 200 accepted documents.

## E.1 Topic distribution: a per-topic cap

The classifier assigns each accepted document to one of six keyword-defined topic buckets. It matches the document’s non-hidden Wikipedia categories against a fixed keyword table. The first matching bucket in table order wins. An other bucket is the fallback:

<table><tr><td>Filter</td><td>Threshold</td><td>Rationale</td></tr><tr><td>min_words</td><td>≥ 3,000</td><td>long documents where structure can matter</td></tr><tr><td>min_depth</td><td>≥3</td><td>heading depth ≥ 3 , so a non-trivial tree exists</td></tr><tr><td>min_sections</td><td>≥ 12</td><td>enough sections to route among / cut at</td></tr><tr><td>n_docs</td><td>200</td><td>scale-up target</td></tr></table>

Table 6: Document filtering criteria.
<table><tr><td>Bucket</td><td>Example keywords</td></tr><tr><td>history_war</td><td>battle, war, military, campaign, siege, history, empire, dynasty</td></tr><tr><td>biography</td><td>births, deaths, people, biograph, politicians, writers, actors, monarchs</td></tr><tr><td></td><td>science_nature species, birds, mammals, physics, chemistry, biology, astronom, geolog</td></tr><tr><td>arts_media</td><td>albums, songs, films, novels, paintings, opera, television, video games</td></tr><tr><td>geo_places</td><td>geography, cities, rivers, mountains, roads, buildings, hurri- canes, storms</td></tr><tr><td>other</td><td>(fallback when nothing matches)</td></tr></table>

Table 7: Topic buckets and example keywords.

Table 8: Realized distribution, verified 2026-08-04.
<table><tr><td>Bucket</td><td>Documents</td><td>Share</td></tr><tr><td>history_war</td><td>60 (at cap)</td><td>30.0%</td></tr><tr><td>other</td><td>60 (at cap)</td><td>30.0%</td></tr><tr><td>biography</td><td>30</td><td>15.0%</td></tr><tr><td>arts_media</td><td>22</td><td>11.0%</td></tr><tr><td>science_nature</td><td>15</td><td>7.5%</td></tr><tr><td>geo_places</td><td>13</td><td>6.5%</td></tr><tr><td>Total</td><td>200</td><td>100%</td></tr></table>

The "balancing" is a single hard ceiling: max\_per\_bucket = 60. The seeded walk enforces it. It rejects a candidate whose bucket already holds 60 accepted documents. There is no minimum-per-bucket guarantee and no roundrobin. The cap only limits over-represented topics. It does not equalize them.

The cap binds on the two largest buckets. Both land exactly at 60. Without it, history\_war and other would each be larger. 60% of the corpus still sits in two buckets. This reflects the genuine topic skew of Featured Articles. The paper discloses this skew rather than hides it, and the skew strengthens the generalization claim. The replicated organization effects hold over a topicimbalanced corpus, not a curated one.

## E.2 Offset-verified gold structure

Gold structure is the article’s own heading tree.   
Two layers make its character offsets trustworthy.

Exact by construction. The pipeline builds the flat document string paragraph by paragraph. It records each paragraph’s as it emits the paragraph. It reads gold section offsets directly from this paragraph table. It never recovers them by searching for a heading’s text. A text search is the step that would introduce off-by-N drift.

Mechanically verified. Segment verification returns a list of violations. An empty list means verified. It enforces that the sections:

• tile the document with no gaps or overlaps. Each segment starts exactly where the previous one ended. The final segment ends at the last paragraph.

• have offsets that agree with the paragraph table exactly. Otherwise it reports "char offsets disagree with paragraph table".

• have in-bounds paragraph ranges and legal nesting levels $\begin{array} { r l r l r l } { ( 1 } & { { } } & { \le } & { { } \ l e v e l } & { { } \le } & { } & { { } } \end{array}$ MAX\_LEVEL = 4).

Relevance runs purely as character-span overlap (C). These offsets are therefore load-bearing. An off-by-N error would silently corrupt every relevance judgment. The same offset discipline applies to induced structure (H). Induced structure emits paragraph index ranges rather than raw offsets. This makes "structure without content modification" a verified property and holds the gold and induced arms to one integrity standard.

## E.3 Placebo construction

Placebo prefixes are real heading paths from other documents. The pipeline buckets unique (document, path) pairs by depth. It permutes them within depth-bucket toward a crossdocument derangement under a fixed seed. It counts and reports residual same-document assignments. The result matches real breadcrumbs in format, depth, and length distribution. It carries no information about the chunk it attaches to. This is a semantically null treatment in the strict sense.

## F Wikipedia-FA query generation and leakage control

Queries are LLM-generated but provenance- and leakage-controlled. The pipeline itself samples the gold spans. The query-writing model never selfreports them.

Four query types per document, over sampled gold spans:

• factoid targets one span.

• underspecified targets one span but names the referent only vaguely ("the commander", never the proper name).

• multihop targets two spans in different toplevel sections.

• structural targets a whole top-level section.

Two-stage leakage filter. Each generated query must pass a lexical-leakage filter. The filter rejects it if ROUGE-L F1 against its gold span exceeds 0.35, or if it shares any verbatim 4-gram with the span. A second LLM call then paraphrases the query. The paraphrase enters the benchmark, and it must pass the same filter again.

Funnel: 2,306 generated → 1,945 survivors → 951 balanced finals (250 factoid, 250 underspecified, 250 structural, 201 multihop) over 184 documents. Shipped queries stay lexically far from their evidence (typical ROUGE-L 0.02–0.13).

Non-lexicality check. BM25 places a gold chunk first for only 36.3% of the shipped queries. This is direct evidence that the leakage controls worked and that keyword overlap does not solve the benchmark (see I).

## G QASPER: corpus construction and evidence mapping

QASPER (Dasigi et al., 2021) supplies NLP papers with questions from real readers and gold evidence paragraphs. This is human-authored relevance on a genre far from encyclopedic prose. The study is an IR ablation and not supervised learning, so the pipeline merges all splits. Gold structure is each paper’s own section tree.

Evidence mapping. A cascade matches each question’s gold evidence paragraphs into the flattened document: exact match, then fuzzy match, then section fallback. 95.7% of evidence paragraphs match exactly.

Filtering funnel: 5,049 → 4,303 usable questions over 1,508 papers, after the pipeline drops unanswerable, no-evidence, and numeric-only items.

Multi-span profile. 43.9% of questions cite multiple evidence paragraphs (mean 2.10 spans/question). This is the regime that coveragenDCG exists for.

Difficulty and non-lexicality. BM25 gold-top-1 is 22.7%. This set is harder and more non-lexical than the Wikipedia-FA set (see I).

Shallow native depth. 61% of QASPER papers have depth-1 section lists. Every Wikipedia-FA document has depth $\geq 3$ by construction. Gold and induced trees therefore have less room to diverge in the information a breadcrumb can carry. This is the offered explanation, untested, for the null C3 on QASPER against the positive C3 on Wikipedia-FA (main-text 6.5).

## H Structure induction protocol

The induced-structure conditions (A2–A4, P) get their hierarchy from the local Gemma-4-26B-A4B model.

Input and output. The model sees the document’s paragraphs with all native headings stripped. It returns sections as (title, level, paragraph-range) tuples under hard constraints. The sections must tile the paragraphs exactly. The levels may increase by at most one between consecutive sections. The model emits paragraph index ranges rather than character offsets. This makes "structure induction without content modification" a mechanically verifiable property.

Validation and repair. The pipeline checks the constraints mechanically. A failing output gets one self-correction round that quotes the specific violations back. Then a deterministic repair snaps a near-valid tiling to a legal one. The pipeline clamps nesting deeper than MAX\_LEVEL = 4, so the gold and induced arms share one level budget. The clamp bounds the stored level integer only. The true depth still drives the breadcrumb path.

The inducer does not merely recover native sectioning. Only 78.9% (Wikipedia-FA) and 75.8% (QASPER) of induced chunk spans are byte-identical to a gold span. Gold and induced conditions therefore measure genuinely different segmentations. This is what makes C3 (G3 − A3) a substantive comparison rather than a tautology.

## I Non-lexicality and the BM25 baseline

What it measures. BM25 is the classic lexical (keyword-overlap) retriever. "BM25 gold-top-1" is the fraction of queries for which BM25 ranks a gold-relevant chunk in position 1. This is a direct probe of how solvable the benchmark is by word matching alone.

Both are low, and QASPER is the harder of the two. A keyword search puts a correct chunk first for under a quarter of its questions. Three consequences matter for this study:

• Headroom. The benchmarks are far from lexically saturated. A meaning-based (dense) retriever therefore has room to matter, and structure has room to help it.

• Regime. The paper’s claims are about dense retrieval and how document structure helps it. A non-lexical benchmark forces the retriever into exactly that regime. The gains therefore cannot be dismissed as trivial keyword effects.

• Leakage defense (Wikipedia-FA especially). For LLM-generated queries, the sharpest objection is that the generator copied words from the evidence. The low BM25 hit rate is the empirical proof that the ROUGE-L/4-gram filters and the paraphrase step (F) worked. The shipped queries really are lexically distant from their evidence. QASPER’s questions are human-written, so its 22.7% is naturally occurring hardness. Two genres, two query provenances, both genuinely nonlexical.

## J Reproducibility and configuration

Determinism. The measured half of the pipeline is deterministic: embedding, retrieval, scoring, statistics. All LLM generation stays in one-time corpus and query construction under fixed seeds with on-disk caching. This covers structure induction, A1 contexts, query generation, and paraphrase. Generation therefore does not enter the measured loop. All generative stages ran on a local Gemma-4-26B-A4B model via vLLM on a single H200 GPU. Table 11 lists reproducibility parameters used in the study.

Size matching (guards C1). The fixed-window baseline’s per-document target window equals the mean token length of that document’s induced chunks. Realized means differ by ≤ 4 content tokens per chunkset (Wikipedia-FA: 255.4 fixed vs. 259.6 induced; QASPER: 244.9 vs. 249.8). The A0/A1-vs-A2/A3 comparison therefore isolates boundary placement, not chunk size. Window edges snap to sentence boundaries. Structural chunks split at sentence boundaries or merge when they exceed 350 or fall below 60 content tokens.

Representation constancy (guards C4). Twostage conditions build section centroids as the renormalized mean of the same chunk vectors that flat retrieval searches. C4 therefore compares retrieval algorithms, never representations. Stage 1 ranks section centroids and keeps the top 5 (a section is a document plus a top-level heading). Stage 2 ranks chunks within the routed sections. If fewer than k candidates remain, the system backfills from the global ranking. A per-query sectionrecall diagnostic records whether stage 1 retained any gold-overlapping section. This separates routing failures from ranking failures.

Release. The authors will release code, configurations, and per-query results.

<table><tr><td>Corpus</td><td>Usable structure</td><td>Required repair</td><td>Mean sections/doc</td></tr><tr><td>Wikipedia-FA</td><td>198 / 200</td><td>116</td><td>21.9</td></tr><tr><td>QASPER</td><td>1,579 / 1,585</td><td>590</td><td>14.5 (gold 14.8)</td></tr></table>

Table 9: Corpus structure statistics.

<table><tr><td>Corpus</td><td>BM25 gold-top-1</td></tr><tr><td>Wikipedia-FA</td><td>36.3%</td></tr><tr><td>QASPER</td><td>22.7%</td></tr></table>

Table 10: BM25 gold-top-1 retrieval rate.

Table 11: Key configuration (configs/scale\_crc.yaml).
<table><tr><td>Group</td><td>Parameter</td><td>Value</td></tr><tr><td>global</td><td>seed</td><td>20260710</td></tr><tr><td>selection</td><td>n_docs/min_words/min_depth/min_sections/max_per_bucket</td><td>200 / 3000 / 3 / 12 / 60</td></tr><tr><td>chunking</td><td>t_max/t_min (content tokens)/ fixed_snap_tokens</td><td>350 / 60 / 15</td></tr><tr><td>breadcrumb</td><td>sep / depths</td><td>&quot; &gt; &quot; / [1, 2, 3]</td></tr><tr><td>querygen</td><td>per_type/rouge_max/ngram_block/context_max_tokens</td><td>250 / 0.35 / 4 / 80</td></tr><tr><td>embedders</td><td>weak / strong</td><td>bge-small-en-v1.5 / bge-large-en-v1.5</td></tr><tr><td>reranker</td><td>model / top / arm</td><td>bge-reranker-v2-m3 / 50 / strong</td></tr><tr><td>retrieval</td><td>top_k/hier_top_sections</td><td>100 / 5</td></tr><tr><td>metrics</td><td>ndcg_k/recall_k/budgets</td><td>10 / 50 / [512, 1024, 2048]</td></tr><tr><td>stats</td><td>bootstrap_iters</td><td>10,000</td></tr></table>