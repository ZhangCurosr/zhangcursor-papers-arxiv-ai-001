# From Intent to Improvement: Agentic Dataset Search for Beter Model Training

Qiaosheng Chen<sup>∗</sup> Nanjing University State Key Laboratory for Novel Software Technology Nanjing, Jiangsu, China qschen@smail.nju.edu.cn

Fanqi Zhou Nanjing University State Key Laboratory for Novel Software Technology Nanjing, Jiangsu, China fqzhou@smail.nju.edu.cn

Kaijia Huang<sup>∗</sup> Nanjing University State Key Laboratory for Novel Software Technology Nanjing, Jiangsu, China kaijiahuang@smail.nju.edu.cn

Gong Cheng<sup>†</sup> Nanjing University State Key Laboratory for Novel Software Technology Nanjing, Jiangsu, China gcheng@nju.edu.cn

## Abstract

Finding training data in million-scale machine learning repositories is a Web search problem where text relevance is not enough. A query couples a semantic training goal with constraints such as task, license, and training stage, and the metadata that would verify them are often missing. Text retrieval ranks dataset cards by similarity but cannot enforce hard constraints or follow the relations that reach a sparsely described dataset. Free-form tool-calling agents are more flexible, but the same model that picks a dataset also declares it a match, and nothing checks that the query’s constraints are actually met. Neither approach establishes that the data it returns will improve model training. To address it, we propose AgenticDS, a contract-governed agentic framework in which every language model decision instantiates a typed object before deterministic code consumes it. The framework compiles a query into a typed search plan, executes it through admissible deterministic operators over hybrid retrieval and bounded graph expansion, and verifies the provenance-grounded evidence before accepting a rank ing or emitting a typed diagnosis that permits only local repairs. On HuDS-Bench, a new benchmark of 200 queries with 24,908 graded judgments pooled over 15 systems, AgenticDS improves NDCG@10 by 25.7% over the strongest dense–sparse hybrid and by 18.9% over a commercial LLM agent. A controlled search-to-train study further shows that the datasets it retrieves give the best downstream accuracy in mathematics, code, and medicine. Code and data are publicly available at https://github.com/nju-websoft/AgenticDS.

## CCS Concepts

• Information systems → Specialized information retrieval; • Computing methodologies → Multi-agent systems; Machine learning.

## Keywords

dataset search, LLM agent, data acquisition, training utility, machine learning

## 1 Introduction

Open machine learning (ML) platforms, exemplified by Hugging Face,<sup>1</sup> now host millions of interconnected datasets, models, papers, and curated collections. For a researcher who wants to improve a model, dataset acquisition is already a Web search problem: an underspecified training need must be turned into search operations over inconsistent metadata, relations across heterogeneous resources resolved, and candidates judged by whether they will change the behavior of the model being trained [2–4, 27]. Keyword and embedding-based retrieval help when dataset cards are complete, but they fail on hard structured constraints and ignore evidence implicit in training provenance, task tags, paper citations, or community collections [7, 8, 12, 29, 47].

Natural language dataset acquisition poses three coupled technical challenges. (1) A query mixes a semantic training goal with typed constraints such as task, modality, license, and training stage. Existing dataset search systems either flatten the query into one embedding [9, 46] or treat metadata fields independently [3, 11], so constraint violations stay invisible. Handing the tools to an LLM restores flexibility but adds no verification, so the same model that selects a dataset also declares it a match [22, 45]. (2) Evidence is heterogeneous. Text retrieval is precise when dataset cards are well written [7, 47], while knowledge graph (KG) relations recover resources whose descriptions are sparse [8, 29, 31]. Structural connectivity does not imply semantic fit, however, so a naively added graph channel can hurt relevance when nothing suppresses the noise it introduces [31]. (3) Retrieval relevance does not entail training utility. Standard benchmarks score ranking against graded relevance judgments [10, 30] but never measure whether the selected data improve a downstream model, so linking search quality to data-centric recursive self-improvement (RSI), the loop in which evaluation results drive changes to the training data [49], requires a protocol that measures that change directly.

To address these challenges, we present AgenticDS, a contractgoverned agentic framework for Web dataset acquisition. A Planner compiles the query into a typed search plan of constraints, allowed KG relations, and acceptance criteria. An Engineer executes only admissible deterministic operators over hybrid retrieval, bounded

KG expansion, and provenance-preserving pre-ranking. A Critic scores a fixed-format evidence package and either accepts the result or emits a typed diagnosis permitting only local plan repairs. The contracts stop an LLM from silently redefining retrieval semantics during tool use, while the evidence trace keeps text, metadata, content, and KG support separately inspectable.

To evaluate this design, we build HuDS-Bench, a KG-grounded benchmark of 200 natural language dataset search queries with 24,908 graded judgments pooled over 15 systems. AgenticDS out ranks every system in the pool, and removing the Critic or the Planner costs 0.149 and 0.086 NDCG@10. We then test whether better retrieval translates into better training, feeding each system’s top-5 datasets into corresponding supervised fine-tuning (SFT) of three base models across mathematics, code, and medicine. AgenticDS gives the best downstream accuracy in all three domains, with the largest gain on the weakest base model, supporting the view that external acquisition complements within-corpus data-centric RSI. Ablation then isolates where relational evidence pays: graph expansion whose candidates pass straight into the ranking lowers accuracy on every text backbone, whereas the same expansion behind a provenance-aware Critic improves it, so expansion coupled to verification rather than the graph itself is the mechanism.

In summary, our contributions are threefold:

• We formulate natural language dataset acquisition on the Web as contract-governed agentic retrieval, and present AgenticDS, whose typed plans, evidence records, and diagnoses separate semantic reasoning from deterministic execution.

• We show that relational evidence is conditional rather than additive, and contribute the per-candidate provenance representation that lets a verifier turn graph expansion from a liability into a gain.

• We release HuDS-Bench and a search-to-train protocol that measures whether retrieved datasets improve a model, and use them to compare 15 retrieval systems and to fine-tune three base models across mathematics, code, and medicine.

## 2 Related Work

## 2.1 Dataset Search on the Web

Dataset search difers from document retrieval because relevance depends on structured constraints, data content, and intended use beyond text [4]. Beyond Google Dataset Search’s Web metadata index [3], later work targets dense representations [9, 31], queryfocused snippets [7, 12, 47], and integrated public portals [11]. Closest to our setting, DataFinder recommends scientific datasets from a natural language description of a research idea [39]. It maps one free-text need to a flat bi-encoder ranking over a fixed corpus, without typed constraints, relational evidence, or any check that the recommended data help training. These are the three axes this paper adds. Benchmarks such as ACORDAR evaluate public-data retrieval [10, 30] but do not model natural language data needs over modern ML hubs. Relational evidence has been explored through general knowledge bases [31] and taxonomies of dataset relationships [29]. HuggingKG links datasets with models, tasks, papers, spaces, and collections [8]. These lines supply the primitives and relational infrastructure AgenticDS composes under a typed contract. We adopt the HuggingKG snapshot as auxiliary infrastructure rather than as a contribution of this paper, using it as one candidate source alongside dense and sparse retrieval, and design HuDS-Bench so that natural language data needs become directly measurable.

## 2.2 Agentic Information Retrieval

LLM agents combine planning, tool use, and reflection [43], building on retrieval-augmented generation [25] and LLM rankers [37]. Zhang et al. [45] formalize “agentic information retrieval” as agents that plan, execute, and refine multi-step search strategies. Search-R1 [22] learns when to call a search engine during reasoning, and Search-o1 [28] embeds agentic Web search into extended reasoning chains. A study of 14 million real queries shows that agentic intents and trajectories difer markedly from traditional Web search [32]. Most such systems target open-domain question answering and give the LLM direct control of tools, which entangles query interpretation with index operations and makes errors and reruns hard to compare. Commercial coding agents prompted to search Hugging Face behave the same way. AgenticDS instead keeps semantic reasoning inside typed contracts and leaves indexing, traversal, filtering, and scoring deterministic, trading open-ended tool flexibility for reproducibility, provenance-transparent ranking, and bounded token cost.

## 2.3 Data-Centric Recursive Self-Improvement

Data-centric RSI systems filter, select, rewrite, or synthesize samples from a known corpus. DataEvolver learns preparation pipelines from raw data and seeds through operator- and pipeline-level feedback [17]. DataFoundry evolves a data-construction runtime and synthesizes instruction–response pairs from supplied documents [42]. The broader literature studies evaluation-driven updates to data, retrieval state, memory, or policies [13, 40, 44, 49]. All presuppose a corpus already in hand and optimize data within it, while our work targets the preceding step. We acquire external datasets from an open Web ecosystem given a user query, and verify utility through controlled fine-tuning. That search-to-train primitive is the ingredient these systems assume, positioning AgenticDS as a complementary upstream component of future data-centric loops rather than a competing in-corpus optimizer.

## 3 AgenticDS: Framework and Mechanism

## 3.1 Task and Design Principles

Let � be a query containing a natural language description of a user’s training data need, and D a dataset collection in which each � ∈ D carries heterogeneous and often incomplete metadata. We seek a ranked list L(�) of up to � datasets that meet the goal and hard constraints of � and rank highly on its soft preferences. Our implementation uses a heterogeneous graph G from HuggingKG (Appendix A), whose nodes include datasets, models, papers, and collections, with relations such as model–training-dataset and collection membership. Missing metadata or graph links indicate unknown support, not irrelevance.

AgenticDS follows three principles. Contracts before actions makes typed plans delimit the admissible operators. Provenance before fusion keeps evidence sources separately inspectable. Bounded autonomy lets budgets and diagnoses constrain search and local plan repairs. These principles bound what the LLM may redefine without discarding the semantic flexibility needed to interpret complex queries. Figure 1 gives an overview of the pipeline, and Algorithm 1 formalizes the search loop.

![](images/6c8d7498c74aad63383b38c31cbdde594ca7d3111807752e34bf509ff495fa36.jpg)  
Figure 1: Overview of AgenticDS.

Algorithm 1: AgenticDS search loop.   
Input: query �, corpus D, graph G, result count �, round limit �   
1 B, Q, $R _ { \mathrm { b e s t } } , \mathbf { P } _ { 0 } , \mathcal { F } _ { 0 } \gets \emptyset ;$   
for � = 1 to � do   
3 P<sub>�</sub> ← Planner.Plan(�, �, G, P<sub>�−1</sub>, F<sub>�−1</sub>, Q);   
4 if ¬Planner.HasNewQuery $( \mathbf { P } _ { t } , \boldsymbol { Q } )$ then break;   
5 S<sub>�</sub> ← Engineer.Compile(P<sub>�</sub> );   
6 $( \{ L _ { j } ^ { \mathrm { d i r e c t } } \} , A _ { t } )$ ← Engineer.HybridRetrieve $( \mathbf { S } _ { t } , \mathcal { D } , \mathcal { G } ) ;$   
7 $\{ { L } _ { j } ^ { \mathrm { g r a p h } } \}$ ← Engineer.TypedKGExpand(�<sub>�</sub>, G, S<sub>�</sub> );   
8 B ← Engineer.UpdateBranchLists( B, $\{ L _ { j } ^ { \mathrm { d i r e c t } } \} , \{ L _ { j } ^ { \mathrm { g r a p h } } \} ) ;$   
9 Q ← Q ∪ Signatures(H );   
10 �<sub>RRF</sub> (·) ← Engineer.GroupRRF( B, $\mathbf { P } _ { t } )$ ;   
11 $C _ { t } \gets$ Engineer.ConstraintPrerank( B, �<sub>RRF</sub>, $\textstyle ( \mathbf { P } _ { t } , \mathbf { S } _ { t } ) ;$   
12 $( R _ { t } , \mathcal { F } _ { t } )$ ← Critic.Review $( q , \mathbf { P } _ { t } , C _ { t } ) ;$   
13 $R _ { \mathrm { b e s t } }  R _ { \mathrm { a } }$ <sub>rg</sub> <sub>max1≤�≤� �</sub> <sub>(F� )</sub> ;   
14 if Stop $( \mathcal { F } _ { 1 : t } )$ then break;   
15 return $R _ { \mathrm { b e s t } } [ 1 : K ] ;$

## 3.2 Planner: Compiling Intent into a Typed Contract

The Planner produces a typed search plan (line 3) from �, �, and the allowed path types of G. On later rounds it also reads the previous plan $\mathbf { P } _ { t - 1 } ,$ feedback $\mathcal { F } _ { t - 1 } ,$ and executed-query set $\scriptstyle Q ,$ all initialized empty in line 1. In local repair the Planner preserves the original requirements while adding new queries that address diagnosed gaps. The Planner ends the search once it has no new query, that is, no query-branch signature outside Q (line 4). The contract is

$$
\mathbf P _ { t } = ( \mathcal H _ { t } , \mathcal R _ { t } , C _ { \mathrm { h a r d } } , C _ { \mathrm { s o f t } } , I _ { q } , N _ { t } , \mathcal T _ { \mathrm { q u a l } } ) .\tag{1}
$$

In Eq. (1), $\mathcal { H } _ { t }$ contains several query branches, each grounded in � and associated with a specific evidence purpose. The Planner marks a branch direct-retrieval when its candidates come from the dataset index itself, and keeps one such branch at the original user query, preserving joint semantics that decomposition may separate. It marks a branch graph-expansion when its candidates must instead be reached from entity anchors over declared graph paths. An anchor starts a graph path and may be a dataset, model, paper, or collection. In $\mathcal { R } _ { t }$ the Planner declares the one- or two-hop paths to datasets in $\mathcal { G } .$ It records requirements in $C _ { \mathrm { h a r d } } ,$ preferences in $C _ { \mathrm { s o f t } }$ , and the intents a candidate should jointly support in $I _ { q } .$ In N<sub>�</sub> it reserves bounded pool slots for selected query branches, because broad text matches could otherwise crowd out candidates with specific provenance or evidence before verification. In $\mathcal { T } _ { \mathrm { q u a l } }$ it sets quality, intent-coverage, and minimum-count thresholds.

As illustrated in Figure 1, the Planner emits a Task query branch targeting Python coding, a Format query branch targeting instruction– code pairs, and a graph-expansion query branch anchored on models for Python instruction tuning, alongside the retained original query.

## 3.3 Engineer: Hybrid Retrieval and Typed Relation Expansion

The Engineer compiles $\mathbf { P } _ { t }$ into an action protocol S<sub>�</sub> (line 5): predefined operators with bounded anchor, neighbor, and pool budgets. Within these budgets, the Engineer adaptively allocates retrieval efort according to $\mathbf { P } _ { t } ,$ , which indicates where useful evidence is likely to lie. It then applies hybrid retrieval to collect candidate datasets for direct-retrieval query branches and anchors for graphexpansion query branches (line 6). For a retrieved entity (i.e., a candidate dataset or an anchor) �, its dense score is

$$
s _ { \mathrm { d e n s e } } ( e ) = \eta s _ { \mathrm { d e s c } } ( e ) + ( 1 - \eta ) s _ { \mathrm { m e t a } } ( e ) ,\tag{2}
$$

where $s _ { \mathrm { d e s c } }$ and $s _ { \mathrm { m e t a } }$ are cosine similarities between the branch query and the entity’s description and metadata embeddings. Equation (2) balances semantic descriptions and structured attributes through configurable $\eta \in \left[ 0 , 1 \right]$ . Dense and BM25 ranks are combined by reciprocal rank fusion (RRF) [16]:

$$
s _ { \mathrm { h y b r i d } } ( e ) = \alpha ( k + r _ { \mathrm { d e n s e } } ( e ) ) ^ { - 1 } + ( 1 - \alpha ) ( k + r _ { \mathrm { B M 2 5 } } ( e ) ) ^ { - 1 } .\tag{3}
$$

In Eq. $( 3 ) , r _ { \mathrm { d e n s e } }$ and $r _ { \mathrm { B M } 2 5 }$ denote the ranks of � in the dense and BM25 retrieval lists, respectively; $\alpha \in [ 0 , 1 ]$ controls their relative contributions, while $k > 0$ smooths rank diferences. The outputs are direct-retrieval lists $L _ { j } ^ { \mathrm { d i r e c t } }$ , indexed by query branch $j ,$ and anchors $A _ { t }$ for graph-expansion query branches.

The Engineer then performs typed KG expansion from $A _ { t }$ , following only the paths declared in $\mathcal { R } _ { t }$ , to produce graph-expansion lists $L _ { j } ^ { \mathrm { g r a p h } }$ (line 7). Let $\mathcal { P } _ { d }$ be the permitted paths from anchors in $A _ { t }$ to dataset $d ,$ each written $p = ( a = v _ { 0 } , \ldots , v _ { l } = d )$ with edge types $r _ { h }$ for $1 \leq h \leq l .$ . The relation score of dataset � is

$$
\begin{array} { r l } & { s _ { \mathrm { r e l } } ( d ) = \displaystyle \sum _ { \mathscr { p } \in \mathcal { P } _ { d } } s _ { \mathrm { h y b r i d } } ( a ) ( \rho + ( 1 - \rho ) c ( a ) ) } \\ & { ~ \cdot \gamma ^ { l - 1 } \displaystyle \prod _ { h = 1 } ^ { l } [ \log _ { 2 } ( 2 + \deg _ { r _ { h } } ( v _ { h - 1 } ) ) ] ^ { - 1 } . } \end{array}\tag{4}
$$

Equation (4) aggregates support from all permitted anchor-to-dataset paths. Here �<sub>hybrid</sub> (�) is given by $\mathrm { E q . } ( 3 ) , c ( a ) \in [ 0 , 1 ]$ measures its name/identifier match to the query, and deg (�) counts �’s neighbors under relation $r .$ Configurable $\rho \in \left[ 0 , 1 \right]$ and $\gamma \in \mathsf { \Gamma } ( 0 , 1 ]$ set the confidence floor and path decay. This lets AgenticDS reach a sparsely described dataset through a supported entity anchor. The degree-based penalties reduce support from widely connected en tities, while path decay discounts indirect paths. In Figure 1, the Engineer retrieves a relevant model, follows its training-data links to candidate datasets B and D, and orders them by relation score.

## 3.4 Constraint Handling and Provenance-Preserving Pre-Ranking

The Engineer updates the branch lists held in the cross-round archive B with $\{ L _ { j } ^ { \mathrm { d i r e c t } } \}$ and $\{ L _ { j } ^ { \mathrm { g r a p h } } \}$ , replacing previous lists with the same query-branch identifier (line 8). It also adds the signatures of the current query branches to the query history Q (line 9). The Engineer then fuses the archive into a single ranking by groupnormalized RRF (line 10). Each list $L _ { j }$ belongs to a group �(�) defined by evidence purpose, such as Task, Format, or Graph expansion. For nonempty lists, weights are $\begin{array} { r } { \widetilde { \boldsymbol { w } } _ { j } = \boldsymbol { w } _ { j } \big ( \sum _ { \ell : g ( \ell ) = g ( j ) } \boldsymbol { w } _ { \ell } \big ) ^ { - 1 } } \end{array}$ , where $w _ { j } \geq ($ 0 weights query branch � and group totals are positive. With $\begin{array} { r } { Z = \sum _ { j } \widetilde { w } _ { j } } \end{array}$ and list rank $r _ { j } ( d )$

$$
s _ { \mathrm { R F F } } ( d ) = ( k + 1 ) Z ^ { - 1 } \sum _ { j } \widetilde { w } _ { j } \mathbf { 1 } [ d \in L _ { j } ] ( k + r _ { j } ( d ) ) ^ { - 1 } .\tag{5}
$$

Equation (5) bounds the fusion score of dataset � within [0, 1] and prevents additional queries from disproportionately increasing the influence of any single group.

The Engineer then pre-ranks the fused candidates, discarding those with explicit hard-constraint violations (line 11). Missing fields remain unknown, so potentially relevant datasets are not rejected solely for incomplete metadata. Satisfactory candidates precede unknown ones, and within each of those two classes Eq. (6) adjusts retrieval support for soft preferences:

$$
s _ { \mathrm { p r e } } ( d ) = \beta s _ { \mathrm { R R F } } ( d ) + ( 1 - \beta ) U _ { \mathrm { p r e f } } ( d ) ,\tag{6}
$$

where configurable $\beta \in \left[ 0 , 1 \right]$ balances relevance and preferences, and $U _ { \mathrm { p r e f } } \in [ 0 , 1 ]$ is a weighted mean of preference utilities. For download or like count $x \geq 0 ,$ , the utility is min $\{ 1 , \log ( 1 + x ) / \log ( 1 +$ $\tau ) \}$ , with configurable saturation threshold $\tau > 0 ;$ minimization uses its complement. Without preferences, we set $s _ { \mathrm { p r e } } = s _ { \mathrm { R R F } }$ . From this ordering and the bounded reservations $N _ { t }$ the Engineer selects the Critic pool $C _ { t }$ . In Figure 1, for the train-split requirement above, dataset E explicitly marked evaluation-only is discarded, while an unreported split remains unknown. Candidates retain their querybranch ranks, fusion contributions, constraint status, anchors, and paths. Retaining this record rather than collapsing it into a single score makes the ranking auditable, because a user can tell whether a candidate is supported by its dataset card or by a relation path.

## 3.5 Critic: Evidence Compression, Reranking, and Diagnosis

The Critic reviews the pool $C _ { t }$ against � and $\mathbf { P } _ { t }$ (line 12). Each candidate has an evidence package with four parts: concise card excerpts, normalized metadata, sample fields and instruction–response snippets when available, and retrieval and graph provenance. These sources support judgments of semantic fit, constraint satisfaction, and relational support from the graph. Conditioned on � and $\mathbf { P } _ { t }$ , the Critic judges how strongly each evidence package supports joint satisfaction of the query requirements and returns a semantic score $s _ { \mathrm { l l m } } ( d ) \in [ 0 , 1 ]$ , combined with its deterministic pre-score $s _ { \mathrm { p r e } } ( d )$

$$
s _ { \mathrm { f i n a l } } ( d ) = w _ { \mathrm { l l m } } s _ { \mathrm { l l m } } ( d ) + ( 1 - w _ { \mathrm { l l m } } ) s _ { \mathrm { p r e } } ( d ) .\tag{7}
$$

The configurable $w _ { \mathrm { l l m } } \ \in \ \left[ 0 , 1 \right]$ balances semantic judgment and retrieval support in Eq. (7), allowing content evidence to refine the retrieval ranking while retaining its support. Sorting candidates by $s _ { \mathrm { f i n a l } } ( d )$ yields the current ranking $R _ { t } .$ In Figure 1 this lifts the graph-reached dataset B to the top of the pool.

In $\mathcal { F } _ { t }$ the Critic records acceptance, ranking quality $u _ { t } ,$ , intent coverage $v _ { t } ,$ , and typed diagnoses. Quality is the mean of final scores over the top $\lceil K / 4 \rceil$ datasets in $R _ { t }$ (all if fewer), emphasizing the first results users inspect. Intent coverage is the maximum fraction of intents in $I _ { q }$ supported by at least one dataset among the top $\lceil K / 2 \rceil$ results. Thus, a Python corpus and an unrelated instruction corpus cannot jointly establish instruction–response pairs for Python coding. Acceptance requires that ranking quality and intent coverage meet their respective thresholds in $\mathcal { T } _ { \mathrm { q u a l } }$ , that the minimum result count is satisfied, and that no critical diagnosis remains unresolved. Otherwise the Critic emits typed diagnoses that guide next-round local plan repair.

AgenticDS retains the round of maximal utility $U ( \mathcal { F } _ { s } ) = \lambda u _ { s } +$ $( 1 - \lambda ) v _ { s } \ \mathrm { o v e r } s \leq t \ ( \mathrm { l i n e } \ 1 3 )$ , where configurable $\lambda \in \left[ 0 , 1 \right]$ balances quality against joint intent coverage. Given the feedback history $\mathcal { F } _ { { 1 : t } } , \mathrm { A g e n t i c D S }$ stops on deterministic criteria, namely acceptance, two consecutive rounds without utility improvement, or a critical diagnosis recurring in consecutive rounds (line 14). Finally, AgenticDS returns the first � datasets of $R _ { \mathrm { b e s t } }$ (line 15), which is the ranked list $\mathcal { L } ( q )$ in Section 3.1.

Appendix B lists all contracts, score weights, and prompts. Because the Planner fixes the query branches and the Engineer may run only the declared operators, a module can be disabled while everything downstream of it behaves as before, which is what makes component-level ablation well defined.

Table 1: Composition of HuDS-Bench.
<table><tr><td>Query type</td><td>Count</td><td>Primary grounding</td><td>Typical requirement</td></tr><tr><td>General</td><td>90</td><td>Metadata + task tags</td><td>Discover data satisfying task, language, modality, or scale constraints</td></tr><tr><td>Fine-tuning</td><td>40</td><td>Metadata + training stage</td><td>Find instruction, dialogue, preference, or domain-adaptation data</td></tr><tr><td>Paper reference</td><td>40</td><td>Paper/model relations</td><td>Locate data associated with a specific paper or training lineage</td></tr><tr><td>Entity neighbor</td><td>30</td><td>Dataset/collection relations</td><td>Find datasets reachable through multi-hop collection, model, or task paths</td></tr></table>

## 4 Evaluation Methodology

We evaluate two research questions. RQ1: Can AgenticDS retrieve datasets that satisfy complex natural language training needs on a modern ML hub? RQ2: Can the retrieved datasets improve models when used as training interventions? RQ1 uses a dataset search benchmark and ranking metrics. RQ2 holds training budgets and optimization settings constant so that diferences arise from dataset acquisition rather than training recipe.

## 4.1 HuDS-Bench

HuDS-Bench targets realistic Web dataset search over HuggingFace and consists of 200 English queries of four types, two semantic (general and fine-tuning) and two relational (paper-reference and entity-neighbor). Its construction is KG-grounded. An extractor derives structured conditions from a gold dataset’s metadata and, for relational types, from HuggingKG [8] anchors. An LLM, GLM 5.2, writes a concise user need from a controlled subset of those conditions, seeing the anchor and target relation but not the gold dataset’s identity or card text for the two relational types. Rulebased checks and manual verification then reject queries that expose the gold dataset’s HuggingFace identifier, copy long card spans, omit intent, or collapse to an exact-name lookup.

The four types stress complementary evidence channels (Table 1). The semantic types emphasize semantic goals plus structured con straints that a dataset card can usually answer. The relational types instead anchor on a paper, model, or collection, and 29 of the 30 entity-neighbor queries are strictly multi-hop. KG relations therefore enter benchmark construction, so the benchmark can produce needs whose correct answer is unreachable by lexical matching alone.

## 4.2 Pooling and Relevance Annotation

Candidate pools merge the top-20 results of all 15 evaluated systems, with the gold dataset injected only when no system retrieves it. The pool holds 24,908 query–candidate pairs. Rather than pass raw dataset cards to an annotator, we compile each query into a structured rubric and assemble a per-candidate evidence package, from which the LLM annotator, DeepSeek-V4-Pro-0813, resolves each constraint to a three-valued status that we deterministically project to a graded 0/1/2 label for the candidate. A second, independent pass by the same model re-checks every disputed item, and residual disagreements go to human adjudication. Against a 2,490-pair human-annotated sample the pipeline reaches linearly weighted Cohen’s $\kappa = 0 . 7 5 4 ,$ substantial agreement on the conventional scale [24]. Appendix C gives the rubric, the label distribution by query type, and the annotation procedure.

## 4.3 Baselines, Backbone, and Metrics

We compare AgenticDS against four strong baselines spanning the Web-agent and traditional retrieval families. Two LLM agents are implemented with Claude Code and Codex harnesses, both driven by GLM-5.2 [18] and given the same HuggingFace Hub tools. Qwen3-Embedding-0.6B + BM25 is a dense–sparse hybrid fused by RRF [16, 46], and monoT5 (OOD) + BM25 is a classic retrievalthen-rerank pipeline [33]. We mark monoT5 (OOD) because it is applied zero-shot, no in-domain training signal existing for this task. Nine further systems span platform search, sparse, dense, and hybrid retrieval. Appendix G lists every system with its configuration and gives the agent tool interface and execution protocol.

AgenticDS runs with GLM-5.2 as the agent backbone, Qwen3- Embedding-0.6B as its embedding model, and $T \ = \ 3$ iterations unless stated. We also report a second variant that swaps only the embedding model for bge-m3 [5]. Because the same model family generated the HuDS-Bench queries, a shared prior over phrasing could flatter the system. The backbone sweep bounds that concern, since substituting Qwen3.5-35B-A3B from a diferent family still places AgenticDS above every baseline (Appendix D). All hyperparameters are fixed across reported runs, tuned beforehand on a pilot set disjoint from HuDS-Bench.

For RQ1, retrieval quality is measured by NDCG and MAP at cutofs 5/10/20. Statistical significance is assessed by two-sided paired permutation tests over the 200 per-query scores with Holm correction within each family of comparisons, and confidence intervals by paired bootstrap resampling stratified by query type. We fix AgenticDS (Qwen3-Emb) as the reference and correct over the 14 comparisons against the other systems. Appendix E gives the resampling configuration and every test outcome.

## 4.4 Utility-Grounded Search-to-Train Protocol

Ranking does not establish training utility. For RQ2 we treat each search output as a controlled training intervention. For retrieval method � and query �, let $E ( \mathcal { L } _ { R } ( q ) )$ be the eligible datasets among its top-� results, and let Sample(�, �) draw at most � samples from dataset �. The acquired corpus is

$$
{ \mathcal { A } } _ { R } ( q ) = \bigcup _ { d \in E ( { \mathcal { L } } _ { R } ( q ) ) } { \mathrm { S a m p l e } } ( d , b ) ,\tag{8}
$$

with identical � = 2000, eligibility rules, and $K = 5$ for every method. For base model $f _ { 0 } { } ;$ , training algorithm Train, and downstream-task

Table 2: Retrieval quality on HuDS-Bench. Bold marks the best system and underline the second best. <sup>†</sup> marks a cell where AgenticDS (Qwen3-Emb) is significantly higher than that system (� < 0.05).
<table><tr><td></td><td colspan="3">NDCG</td><td colspan="3">MAP</td></tr><tr><td>Method</td><td>@5</td><td>@10</td><td>@20</td><td>@5</td><td>@10</td><td>@20</td></tr><tr><td>Platform search</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GoogleDS</td><td></td><td>.0301† .0217†</td><td>.0165†</td><td>.0262†</td><td></td><td>.0146† .0091†</td></tr><tr><td>HFFullText</td><td>.2786†</td><td>.2438†</td><td>.2100†</td><td>.2989†</td><td>.2301†</td><td>.1661†</td></tr><tr><td>Sparse</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td></td><td></td><td>.4719† .4717† .4797†</td><td>† .5912† .5575† .5252†</td><td></td><td></td></tr><tr><td>BM25+RM3</td><td>.4772†</td><td></td><td>.4703† .4732†</td><td>.5994†</td><td>.5482†</td><td>.5047†</td></tr><tr><td>Dense</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>bge-m3</td><td>.3102†</td><td></td><td>.3204† .3309†</td><td>.4751† .4475† .4084†</td><td></td><td></td></tr><tr><td>Qwen3-Emb</td><td>.4728†</td><td>.4699†</td><td>.4808†</td><td>.6129†</td><td></td><td>.5830† .5570†</td></tr><tr><td>Hybrid / rerank</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>bge-m3+BM25</td><td></td><td></td><td>.4039† .4036† .3972† .5507† .5144† .4659†</td><td></td><td></td><td></td></tr><tr><td>MDRBM25</td><td></td><td></td><td>.4602† .4578† .4698† .5804† .5420† .5107†</td><td></td><td></td><td></td></tr><tr><td>ANCE (OOD)+BM25</td><td></td><td></td><td>.4617† .4570† .4644† .5913† .5469† .5069†</td><td></td><td></td><td></td></tr><tr><td>Qwen3-Emb+BM25</td><td></td><td></td><td>.4911† .4893† .4964†</td><td></td><td>.6202† .5919† .5663†</td><td></td></tr><tr><td>monoT5 (OOD)+BM25</td><td></td><td></td><td>.4976† .4923† .5006†</td><td>.6140†</td><td>.5795†</td><td>.5525†</td></tr><tr><td>LLM agents</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Claude Code</td><td>.5328†</td><td>† .5170†</td><td>† .5169†</td><td>.7009</td><td>.6700</td><td>.6381</td></tr><tr><td>Codex</td><td>.5370†</td><td>.5140†</td><td>.5082†</td><td>.7094</td><td>.6764</td><td>.6313</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AgenticDS (bge-m3)</td><td>.5692†</td><td>.5518†</td><td>.5424†</td><td>.7374</td><td>.6871</td><td>.6311</td></tr><tr><td>AgenticDS (Qwen3-Emb)</td><td>.6297</td><td>.6148</td><td>.6114</td><td>.7316</td><td>.6973</td><td>.6630</td></tr></table>

benchmark set $\mathcal { B } _ { q }$ scored by metric $m _ { B } ,$ , acquisition utility is

$$
\begin{array} { r } { U ( R , q ) = \frac { 1 } { | \mathcal { B } _ { q } | } \displaystyle \sum _ { B \in \mathcal { B } _ { q } } \left[ m _ { B } ( \mathrm { T r a i n } ( f _ { 0 } , \mathcal { A } _ { R } ( q ) ) ) - m _ { B } ( f _ { 0 } ) \right] . } \end{array}\tag{9}
$$

Only the retrieval method varies. Base model, data budget, trainer, and evaluation suite are fixed.

We instantiate the protocol with three base models (Qwen2.5- 7B [34], Llama-3.1-8B [19], and Olmo-3-7B-Instruct [38]), across three domains (mathematics, code, and medicine) with one realistic query per domain. Learning rate and epoch are selected on a held-out development split, never on the evaluation benchmarks, and we report the test score of the checkpoint that development selection picked. Scores are calculated for a grid of three base models and eight benchmarks: {GSM8K [15], MATH [21], AMC23, OCW-Courses [26]} for math, {HumanEval [6], BigCodeBench [48]} for code, and {MedQA [23], MMLU-Medical [20]} for medicine, then averaged across base models. The same split, selection procedure, and grid apply to every method, so none is given more selection freedom than another. For statistical significance we block by (domain, base model) and enumerate all $2 ^ { 9 }$ sign flips of the nine block diferences, Holm-correcting the comparison of AgenticDS with four strong baselines (Section 4.3). Because those blocks share three training corpora and the study has no independent training seeds, we read these tests as a consistency check across blocks rather than as a measure of how much retraining would move the scores.

![](images/29a50bc3c12b7ac66e1d4218e355d810a7b04256247127ea4ae012f93869e581.jpg)  
Figure 2: Search-pipeline ablation. Each stem is a configuration’s distance from the full system (vertical line).

Appendix F gives the benchmark splits, the training grid and search queries, the per-model scores, and the full ablation grid.

## 5 Results

## 5.1 Dataset Retrieval (RQ1)

Table 2 reports all 15 systems. AgenticDS (Qwen3-Emb) leads on every NDCG cutof, beating the best agent baseline, Claude Code, by 18.9% and the strongest hybrid baseline, Qwen3-Emb+BM25, by 25.7% on NDCG@10. These NDCG margins hold against all fourteen other systems after Holm correction, the weakest of them being the 0.063 gap to our own bge-m3 variant $( p = 0 . 0 0 3 )$ . The picture is narrower on MAP: AgenticDS posts the best MAP@10 and MAP@20, but its margins there over the two agent baselines do not survive correction, so we claim an advantage in graded ranking rather than in binary relevance. The same caution applies between our two variants, whose MAP diferences are all within noise. Platform search is worst by a wide margin, confirming that literal string matching over inconsistent cards is inadequate for ML training data needs. AgenticDS outranks all thirteen other systems at every NDCG cutof, by 18.9% over the strongest LLM agent and 25.7% over the strongest hybrid, and the advantage lies in ordering quality, which is what matters when only the top few datasets reach training.

Where does agentic search win most? The gain is mechanismaligned rather than uniform: over the hybrid baseline AgenticDS gains +0.075 and +0.105 NDCG@10 on general and fine-tuning queries, but +0.206 and +0.197 on paper-reference and entity-neighbor queries, so the advantage roughly doubles once an answer stops being recoverable from card text alone. All four gains survive Holm correction over the 56 per-type comparisons (Appendix D). Against the two agent baselines, however, gains separate only on general and fine-tuning queries. On the 40 paper-reference and 30 entity-neighbor queries the diferences do not reach the threshold, which we read as insuficient power on those subsets rather than as equivalence. One query makes the mechanism concrete. Asked for “datasets used to train or finetune the PicoLM-0.5M model”, text retrieval returns benchmarks that merely share the model’s name, whereas the Planner makes the training-source relation a hard constraint and the Engineer reaches three supported corpora whose cards never mention the model at all. The gain therefore tracks the mechanism that produces it, widening where an answer has to be reached through a relation, which is the case the typed anchor constraint and the whitelisted traversal were designed for.

Table 3: Training utility on eight benchmarks from three domains. Each block averages the three base models. Avg is the within-domain mean and All the mean of the three domain averages. Bold marks the best system.
<table><tr><td></td><td colspan="5">Math (EM)</td><td colspan="3">Code (pass@1)</td><td colspan="3">Medicine (acc.)</td><td></td></tr><tr><td>Acquisition method</td><td>GSM8K</td><td>MATH</td><td>AMC23</td><td>OCWCourses</td><td> $A \nu g$ </td><td>HumanEval</td><td>BigCodeBench</td><td> $A \nu g$ </td><td>MedQA</td><td>MMLU-Med</td><td> $A \nu g$ </td><td>All</td></tr><tr><td>No fine-tuning (zero-shot)</td><td>54.54</td><td>42.91</td><td>27.50</td><td>16.42</td><td>35.34</td><td>55.08</td><td>29.04</td><td>42.06</td><td>51.40</td><td>54.84</td><td>53.12</td><td>43.51</td></tr><tr><td>Qwen3-Emb+BM25</td><td>79.78</td><td>52.73</td><td>34.17</td><td>17.16</td><td>45.96</td><td>66.06</td><td>32.37</td><td>49.21</td><td>49.36</td><td>58.82</td><td>54.09</td><td>49.75</td></tr><tr><td>monoT5 (OOD)+BM25</td><td>69.34</td><td>48.87</td><td>29.17</td><td>17.15</td><td>41.13</td><td>57.93</td><td>31.78</td><td>44.85</td><td>52.58</td><td>63.66</td><td>58.12</td><td>48.03</td></tr><tr><td>Claude Code</td><td>77.05</td><td>45.25</td><td>30.00</td><td>17.16</td><td>42.37</td><td>64.02</td><td>29.97</td><td>47.00</td><td>55.77</td><td>61.25</td><td>58.51</td><td>49.29</td></tr><tr><td>Codex</td><td>65.56</td><td>50.82</td><td>30.83</td><td>17.16</td><td>41.09</td><td>62.60</td><td>29.73</td><td>46.17</td><td>53.44</td><td>61.86</td><td>57.65</td><td>48.30</td></tr><tr><td>AgenticDS (ours)</td><td>82.66</td><td>57.01</td><td>40.00</td><td>19.61</td><td>49.82</td><td>67.07</td><td>32.16</td><td>49.62</td><td>55.72</td><td>62.28</td><td>59.00</td><td>52.81</td></tr></table>

## 5.2 Component Ablation

Figure 2 ablates the AgenticDS search pipeline on HuDS-Bench over ten configurations. This is our primary ablation: it is measured on all 200 queries, every configuration loses accuracy relative to the full system, and the three largest losses all involve the Critic. The Critic and the Planner dominate (−0.149 and −0.086 NDCG@10), KG expansion follows at −0.084, and the remaining three components fall between −0.017 and −0.036. The feedback-loop entry removes replanning entirely, stopping after the first round, so its −0.036 measures what the whole revision mechanism buys. The adaptive budget is the Engineer’s per-branch allocation of the anchor, neighbor, and pool budgets, replaced here by an equal split, and sample evidence is the sample fields in the Critic’s evidence package, withheld here. Verification is therefore the load-bearing component, since every configuration that removes the Critic, alone or in combination, falls below both agent baselines.

One combination is not additive. Removing the Planner and the Critic (−0.121) is less damaging than removing the Critic alone (−0.149), because without a Planner the Engineer falls back to a wide default retrieval that leaves the unverified ranking more usable candidates. Without the Critic, KG candidates introduce noise and degrade sparse, dense, and hybrid retrieval alike (Appendix D), confirming that relational expansion requires semantic verification. The components therefore interact rather than contribute independently, so the loop has to be read as a whole even though the Critic carries by far the largest single efect.

## 5.3 Training Utility (RQ2)

Table 3 answers whether retrieval gains transfer to model gains. AgenticDS leads the other four acquisition methods in all three domains, with a grand mean of 52.81 against 49.75 for the secondbest, a +3.06 margin. Margins in this subsection are means of the three domain averages, each of which first averages the three base models. The equal-weight 24-block mean gives the same ordering. The advantage over each of the four methods is statistically signif icant after Holm correction (� = 0.016 to 0.020), with AgenticDS ahead in 7 of the nine (domain, base model) blocks against Claude Code, 8 against the hybrid of Qwen3-Embedding-0.6B and BM25, and all nine against Codex and monoT5-based reranking. Under the broader correction that also tests the baselines against each other, the margins over Codex and monoT5 remain significant while those over Claude Code and the hybrid do not, and no two baselines separate from one another. Ranking quality therefore does convert into model quality, which is the link the search-to-train protocol was built to test and the one that gradedjudgments alone cannot establish.

Table 4: Search-end and training-end ablation compared. Training entries are means over the nine (domain, base model) blocks, and s.e. is the standard error of that mean.
<table><tr><td rowspan="2">Component removed</td><td>Search</td><td colspan="2">Training</td><td>Sign</td></tr><tr><td>∆NDCG@10</td><td>mean</td><td>s.e.</td><td>agrees</td></tr><tr><td>Critic</td><td>-0.149</td><td>-1.85</td><td>1.36</td><td>yes</td></tr><tr><td>Planner</td><td>-0.086</td><td>-0.60</td><td>0.92</td><td>yes</td></tr><tr><td>KG expansion</td><td>-0.084</td><td>+0.10</td><td>0.56</td><td>no</td></tr><tr><td>Feedback loop</td><td>-0.036</td><td></td><td></td><td>一</td></tr><tr><td>Adaptive budget</td><td>-0.031</td><td>+0.43</td><td>0.85</td><td>no</td></tr><tr><td>Sample evidence</td><td>-0.017</td><td>-1.43</td><td>1.15</td><td>yes</td></tr></table>

Domain-level analysis. Math shows the clearest gain (+3.86 over the next best), driven by the competition benchmarks, where AgenticDS retrieves specialized SFT corpora rather than generic math data. Across base models the pattern is monotone in model weakness: Llama-3.1-8B gains +25.12 with AgenticDS against +21.81 for the best alternative, Qwen2.5-7B +14.66 against +7.39, and the strongest base, Olmo-3-7B-Instruct, only +3.67 against +2.65.

Code and medicine show narrower three-model margins (+0.41 and +0.49), and in both the aggregate hides the structure. In code, dataset cards are self-documenting enough that dense retrieval is already near-optimal on the strong bases, and AgenticDS separates clearly only on Llama-3.1-8B. In medicine the margin again concentrates on Llama-3.1-8B, where AgenticDS reaches 61.16 against 48.41 for the hybrid baseline, a gap of nearly 13 points on a model whose medical cards are too terse for dense retrieval to exploit. The reranking baseline is not subject to the same failure and reaches 60.31, so what separates methods in medicine is whether the acquisition stage can judge a terse card at all, not whether it uses a graph. Across the nine (domain, base model) blocks AgenticDS wins seven and loses two to Claude Code, both in medicine and both by under one point. The practically relevant regularity is therefore that the weaker the base model, the more acquisition quality decides the outcome, so the systems that most need good data are exactly the ones an acquisition system helps most.

![](images/1722e0fdf0f78666a7d578f05f3c06b26706950d8d09248202a4e7fc8e3b0c89.jpg)

![](images/283e97a6fd44c5923d68749ed7eca4e037d5b8dbb7f5d8373ae73c30eefaecca.jpg)

![](images/2b4cc433f485ab6aaf28f685e2281a5d182d213ca8e85a3ccd873289fbadd693.jpg)  
Figure 3: Cost over the 200 HuDS-Bench queries. (a) billed tokens by type, (b) cost in USD against mean latency, and (c) AgenticDS token and latency share by stage.

## 5.4 Downstream Component Ablation

Does the search-end ablation survive fine-tuning? Table 4 repeats each single-module ablation through the search-to-train protocol, aggregated over all nine (domain, base model) blocks. It does not resolve the question for any individual component. The per-block changes are dispersed enough that a nine-block mean carries a standard error of 0.6–1.4 accuracy points, so a component would have to move that mean by roughly two points to be separated, and the largest efect, removing the Critic at −1.85, falls just under the bar. We therefore read no per-component downstream attribution out of this experiment in either direction. In particular, the small positive means for KG expansion and the adaptive budget are not evidence against those modules.

What the experiment does support is a sign check. Three of the five components move both ends in the same direction, and the largest search-end efect is also the most negative downstream. The exception is informative. Sample evidence is the smallest searchend efect (−0.017) and the second largest downstream one (−1.43), consistent with sample snippets mattering for whether a corpus trains well rather than for whether it ranks well. Appendix F.4 gives the per-block grid and discusses why a module can be clearly efective at the search stage and invisible after fine-tuning. The protocol is therefore the right instrument for ranking acquisition methods and the wrong one for assigning credit inside a pipeline, which is why we report its standard error beside its means and keep component attribution at the search end.

## 5.5 Cost Analysis

Figure 3 compares AgenticDS with the two agent baselines over the 200 queries, with AgenticDS running under its full � = 3 cap. AgenticDS matches Claude Code’s wall-clock time (150.5 vs. 139.3 seconds per query) while billing 12.3M tokens against 111.4M and 280.9M. Most of that token gap is cached prompt reuse billed at a lower rate, so AgenticDS’s relative cost savings are smaller than its relative token savings. Because all three systems run the same GLM-5.2 backbone, one price list settles the comparison: the 200- query evaluation costs \$19.86 for AgenticDS against \$48.78 and \$112.40, savings of 59% and 82%, and AgenticDS is 5.8× faster than Codex end to end. The saving is a lower bound, since AgenticDS has no cache-hit breakdown and is billed entirely at the uncached input rate. The architectural reason is that the Engineer reads each dataset card at most once and the Critic receives a fixed-format evidence package, whereas free-form tool-calling agents re-fetch and re-read cards during open-ended exploration.

Within AgenticDS the two costs sit in diferent places. The Critic accounts for 83.0% of the tokens but only 28% of the wall clock, the two Planner calls take half the wall clock on 13% of the tokens, and the retrieval and fusion stage is deterministic and spends no tokens at all. Appendix G gives the token totals, the per-system costs, and the price list behind these figures. Bounded autonomy therefore has a measurable payof. Even with three rounds available, fixing the operator vocabulary and capping the budget holds cost near-constant per query, whereas an open-ended agent’s cost grows with however long it chooses to explore.

## 6 Discussion

From retrieval to acquisition. AgenticDS makes dataset search behave more like product search or paper discovery. It interprets a natural language intent, orchestrates structured and unstructured evidence, and returns candidates that serve the operational purpose of training rather than merely matching a description. Applying the separation between interpretation, execution, and verification that is familiar across Web information systems to ML data acquisition buys a guarantee unconstrained tool-calling agents cannot ofer, namely that the same plan reaches the same corpus and yields the same candidate set.

The role of KG evidence. HuggingKG is infrastructure rather than a contribution, and it does two jobs. During search, bounded paths supplement lexical retrieval and record why a resource was reached. During benchmark construction, relation-grounded queries create needs that copying card text cannot meet. The boundary condition on the first job is sharp. Graph candidates added without semantic verification lower NDCG@10 by 0.025–0.029 across three backbones, so expansion and verification are not separable features. Where the graph does pay, it pays as designed: entity-neighbor queries needing one- or two-hop traversal gain +0.197 NDCG@10, against +0.075 for general queries already answerable from card text. Appendix H traces both behaviours on individual queries, including one the system correctly reports as unsatisfiable.

## 7 Conclusion

In this paper we presented AgenticDS, a contract-governed agentic framework for natural language instructed dataset acquisition on the Web, in which a Planner, an Engineer, and a Critic exchange typed plans, evidence records, and diagnoses so that semantic reasoning never silently rewrites what the retrieval stack does. Experiments on HuDS-Bench, a new benchmark of 200 queries and 24,908 graded judgments pooled over 15 systems, show consistent improvements over both commercial LLM agents and strong dense– sparse retrieval pipelines, and a controlled search-to-train study confirms that the datasets AgenticDS returns yield the best downstream accuracy in all three domains we study. Further analysis shows that relational evidence is conditional rather than additive, since graph expansion pays of only once a provenance-aware verifier checks the path behind each candidate. Dataset acquisition can therefore be treated as an upstream stage of a data-centric improvement loop whose quality is judged by the model it trains rather than by relevance alone.

Two limitations bound these claims, since benchmark construction shares a graph with the retrieval engine and the utility study is narrow in coverage. Future work addresses both with a KG-free gold-label pipeline and a distilled Critic. Appendix I gives the details.

## Acknowledgments

This work was supported by the Noncommunicable Chronic Diseases-National Science and Technology Major Project (2025ZD0544900) and the Fundamental Research Funds for the Central Universities (No. 2026300429).

## References

[1] Nasreen Abdul-Jaleel, James Allan, W. Bruce Croft, Fernando Diaz, Leah S. Larkey, Xiaoyan Li, Mark D. Smucker, and Courtney Wade. 2004. UMass at TREC 2004: Novelty and HARD. In TREC

[2] Omar Benjelloun, Shiyu Chen, and Natasha F. Noy. 2020. Google Dataset Search by the Numbers. In ISWC. 667–682.

[3] Dan Brickley, Matthew Burgess, and Natasha F. Noy. 2019. Google Dataset Search: Building a search engine for datasets in an open Web ecosystem. In WWW. 1365–1375.

[4] Adriane Chapman, Elena Simperl, Laura Koesten, George Konstantinidis, Luis-Daniel Ibáñez, Emilia Kacprzak, and Paul Groth. 2020. Dataset search: a survey. VLDB J. 29, 1 (2020), 251–272.

[5] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. In Findings ofACL. 2318– 2335.

[6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, et al. 2021. Evaluating Large Language Models Trained on Code. CoRR abs/2107.03374 (2021). arXiv:2107.03374

[7] Qiaosheng Chen, Jiageng Chen, Xiao Zhou, and Gong Cheng. 2024. Enhancing Dataset Search with Compact Data Snippets. In SIGIR. 1093–1103.

[8] Qiaosheng Chen, Kaijia Huang, Xiao Zhou, Weiqing Luo, Yuanning Cui, and Gong Cheng. 2025. Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph. In SIGIR. 3433–3443.

[9] Qiaosheng Chen, Zixian Huang, Zhiyang Zhang, Weiqing Luo, Tengteng Lin, Qing Shi, and Gong Cheng. 2023. Dense Re-Ranking with Weak Supervision for RDF Dataset Search. In ISWC. 23–40.

[10] Qiaosheng Chen, Weiqing Luo, Zixian Huang, Tengteng Lin, Xiaxia Wang, Ahmet Soylu, Basil Ell, Baifan Zhou, Evgeny Kharlamov, and Gong Cheng. 2024. ACORDAR 2.0: A Test Collection for Ad Hoc Dataset Retrieval with Densely Pooled Datasets and Question-Style Queries. In SIGIR. 303–312.

[11] Qiaosheng Chen, Qing Shi, and Gong Cheng. 2023. Dataset Search over Integrated Metadata from China’s Public Data Open Platforms. In CCF BigData. 32–43.

[12] Qiaosheng Chen, Xiao Zhou, Zhiyang Zhang, and Gong Cheng. 2024. DUNKS: Chunking and Summarizing Large and Heterogeneous Data for Dataset Search.

In ISWC. 78–97.

[13] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. In ICML. 6621–6642.

[14] Zhiyu Chen, Haiyan Jia, Jef Heflin, and Brian D. Davison. 2020. Leveraging Schema Labels to Enhance Dataset Search. In ECIR. 267–280.

[15] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. CoRR abs/2110.14168 (2021). arXiv:2110.14168

[16] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In SIGIR. 758–759.

[17] Chao Deng, Shaolei Zhang, Ju Fan, and Xiaoyong Du. 2026. DataEvolver: Automatic Data Preparation for Large Language Models Through Multi-Level Self-Evolving. CoRR abs/2606.07001 (2026). arXiv:2606.07001

[18] GLM-5 Team. 2026. GLM-5: From Vibe Coding to Agentic Engineering. CoRR abs/2602.15763 (2026). arXiv:2602.15763

[19] Aaron Grattafiori, Abhimanyu Dubey, AbhinavJauhri, Abhinav Pandey, Abhishek Kadian, et al. 2024. The Llama 3 Herd of Models. CoRR abs/2407.21783 (2024). arXiv:2407.21783

[20] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Un derstanding. In ICLR.

[21] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring Mathematical Problem Solving With the MATH Dataset. In NeurIPS Datasets and Benchmarks.

[22] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. In COLM.

[23] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences 11, 14 (2021), 6421.

[24] J. Richard Landis and Gary G. Koch. 1977. The Measurement of Observer Agreement for Categorical Data. Biometrics 33, 1 (1977), 159–174.

[25] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Genera tion for Knowledge-Intensive NLP Tasks. In NeurIPS. 9459–9474.

[26] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving Quantitative Reasoning Problems with Language Models. In NeurIPS.

[27] Pengyue Li, Sheng Wang, Hua Dai, Zhiyu Chen, Zhifeng Bao, and Brian D. Davison. 2025. A Survey on Open Dataset Search in the LLM Era: Retrospectives and Perspectives. CoRR abs/2509.00728 (2025). arXiv:2509.00728

[28] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large Reasoning Models. In EMNLP. 5420–5438.

[29] Kate Lin, Tarfah Alrashed, and Natasha F. Noy. 2024. Relationships Are Complicated! An Analysis of Relationships Between Datasets on the Web. In ISWC. 47–66.

[30] Tengteng Lin, Qiaosheng Chen, Gong Cheng, Ahmet Soylu, Basil Ell, Ruoqi Zhao, Qing Shi, Xiaxia Wang, Yu Gu, and Evgeny Kharlamov. 2022. ACORDAR: A Test Collection for Ad Hoc Content-Based (RDF) Dataset Retrieval. In SIGIR. 2981–2991.

[31] Weiqing Luo, Qiaosheng Chen, Zhiyang Zhang, Zixian Huang, and Gong Cheng. 2023. An Empirical Investigation of Implicit and Explicit Knowledge-Enhanced Methods for Ad Hoc Dataset Retrieval. In Findings ofEMNLP. 14349–14360.

[32] Jingjie Ning, João Coelho, Yibo Kong, Yunfan Long, Bruno Martins, João Magalhães, Jamie Callan, and Chenyan Xiong. 2026. Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests. In SIGIR. 1404–1415.

[33] Rodrigo Frassetto Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document Ranking with a Pretrained Sequence-to-Sequence Model. In Findings ofEMNLP. 708–718

[34] Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. 2024. Qwen2.5 Technical Report. CoRR abs/2412.15115 (2024). arXiv:2412.15115

[35] Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (2009), 333–389.

[36] Mark D. Smucker, James Allan, and Ben Carterette. 2007. A comparison of statistical significance tests for information retrieval evaluation. In CIKM. 623– 632.

[37] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. In EMNLP. 14918–14937.

[38] Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, et al. 2025. Olmo 3. CoRR abs/2512.13961 (2025). arXiv:2512.13961

[39] Vijay Viswanathan, Luyu Gao, Tongshuang Wu, Pengfei Liu, and Graham Neubig. 2023. DataFinder: Scientific Dataset Recommendation from Natural Language Descriptions. In ACL. 10288–10303.

[40] Peidong Wang, Ming Wang, Zhiming Ma, Xiaocui Yang, Shi Feng, Daling Wang, Yifei Zhang, and Kaisong Song. 2025. Language Models as Continuous Self-Evolving Data Engineers. In EMNLP. 18097–18116.

[41] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In ICLR.

[42] Cehao Yang, Xiaojun Wu, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Hui Xiong, and Jian Guo. 2026. DataFoundry: Evolving Data Preparators via Recursive Self-Improvement. CoRR abs/2608.29966 (2026). arXiv:2608.29966

[43] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In ICLR.

[44] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-Rewarding Language Models. In ICML. 57905–57923.

[45] Weinan Zhang, Junwei Liao, Ning Li, Kounianhua Du, and Jianghao Lin. 2024. Agentic Information Retrieval. CoRR abs/2410.09713 (2024). arXiv:2410.09713

[46] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. CoRR abs/2506.05176 (2025). arXiv:2506.05176

[47] Xiao Zhou, Qiaosheng Chen, Jiageng Chen, and Gong Cheng. 2025. �DS: Multi-Objective Data Snippet Extraction for Dataset Search. In SIGIR. 116–126.

[48] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, et al. 2025. BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions. In ICLR.

[49] Yanyan Zou, Chao Chen, Tung Sum Thomas Kwok, Jiayu Yang, Yunyun Hou, Jingmin Zhu, Zehang Luo, Shengjie Li, Yinhong Liu, Yingjia Wan, Chengzu Li, Tao Yu, Zhijiang Guo, Jiachen Liu, Sirui Wang, and Nan Duan. 2026. Data-Centric Recursive Improvement for Foundation Models: A Survey. Preprints.org.

## A Updated HuggingKG Snapshot

The experiments use a March 12, 2026 snapshot of HuggingKG [8]. Crawling and normalization yield 4,928,681 entities (910,119 datasets, 2,704,939 models, 56 tasks, 33,524 papers, 1,099,292 spaces, and 180,751 collections) and 3,951,507 relations. The update changes coverage, not the schema or method. KG expansion resolves typed paper, model, collection, and dataset anchors and traverses one- or two-hop paths under per-branch budgets.

## B Contracts, Parameters, and Prompts

Contracts specify retrieval hypotheses and query-linked constraints and restrict Engineer execution to admissible operators, declared paths, and configurable query, anchor, neighbor, and pool budgets. Search uses a configurable iteration limit, set to � = 3 unless stated otherwise, with scoring coeficients and acceptance thresholds fixed across rounds. With preferences, pre-ranking mixes grouped retrieval support and utility through $\beta$ and $1 - \beta .$ The final score combines semantic judgment and pre-ranking with weights � and $1 - w _ { \mathrm { { l l m } } }$ . Missing constraint fields remain unknown, and candidates with explicit hard-constraint violations are discarded. Compressed metadata, content, and paths support independent Critic scoring. Acceptance tests configured quality and single-candidate joint-intent targets, at least $\bar { K }$ candidates, and absence of criti cal diagnoses. Feedback permits bounded local plan repair tied to the original query. Search retains the best historical round using quality–coverage utility, with configurable weight �. The Planner contract, Engineer action protocol, evidence record, and diagnosis fields are summarized in Table 5. The full prompts are included in the released code.

Table 5: Typed objects exchanged between the three agents.
<table><tr><td>Object</td><td>Fields and validation</td></tr><tr><td>Plan Pt</td><td>Complementary queries; declared typed paths; query-linked hard/soft requirements; intent and acceptance targets.</td></tr><tr><td>Protocol  $\mathsf { S } _ { t }$ </td><td>Ordered predefined operators; index bindings; query, anchor, neighbor, and pool budget limits; all arguments logged.</td></tr><tr><td>Evidence</td><td>Metadata; content snippet; branch ranks; three-valued constraints; typed anchors and paths; field-aware truncation.</td></tr><tr><td>Diagnosis  $\mathcal { F } _ { t }$ </td><td>Failure code/severity; affected intents; evidence and confidence; per- mitted local plan repair within protocol limits.</td></tr></table>

Table 6: Graded relevance judgments by query type.
<table><tr><td>Query type</td><td>Pairs</td><td>Label 0</td><td>Label 1</td><td>Label 2</td></tr><tr><td>General</td><td>10,730</td><td>2,990</td><td>6,388</td><td>1,352</td></tr><tr><td>Fine-tuning</td><td>4,968</td><td>1,134</td><td>3,210</td><td>624</td></tr><tr><td>Paper reference</td><td>5,049</td><td>4,659</td><td>1</td><td>389</td></tr><tr><td>Entity neighbor</td><td>4,161</td><td>3,842</td><td>164</td><td>155</td></tr><tr><td>Total</td><td>24,908</td><td>12,625</td><td>9,763</td><td>2,520</td></tr></table>

## C HuDS-Bench Construction Details

The construction pipeline derives structured conditions (task, modal ity, language, scale, format) from each gold dataset’s textual metadata and KG neighborhood, generates a need-centric query, and verifies that the result does not leak the repository identifier or collapse to a single lookup. Candidate pools merge the top-20 results of all 15 evaluated systems. Each candidate’s evidence package contains concise card excerpts, normalized metadata, sample fields and instruction–response snippets when available, and retrieval and graph provenance. Using a structured rubric (released with the code), the LLM annotator marks every constraint satisfied, violated, or unknown and cites an evidence code for each satisfied or violated decision, which we deterministically project to a graded label. Label 2 requires an exact or strong intent match with satisfied hard constraints, a matched dataset role, and direct evidence, label 1 covers partial matches, and label 0 marks hard conflicts or role mismatches. A verifier model re-checks any item with an evidence conflict, a role mismatch, an unknown on a hard constraint, or an unresolved relation, and disputes are escalated to human review. Table 6 gives the distribution.

Human agreement study. To check that the automated pipeline reproduces human judgment rather than being self-consistent, we drew 2,490 distinct query–dataset pairs (10.0% of the 24,908 judged pairs) by random sampling without replacement, spanning 1,051 general, 534 fine-tuning, 490 paper-reference, and 415 entityneighbor pairs, including 861 flagged for review, which were not excluded. One researcher familiar with machine-learning datasets independently applied the same 0/1/2 label criteria, seeing the query, metadata, full dataset card, and the necessary entity-page and graph evidence. Because partial credit is the hardest grade to assign, label 1 is also the grade on which the two relational query types almost never fire (Table 6), so the overall agreement score mainly reflects the two semantic types. Comparing the annotator’s first labels with the previously frozen automated labels, before any disagreement analysis or correction, yields 84.1% exact agreement (2,095/2,490) and linearly weighted Cohen’s � = 0.754, substantial agreement [24].

Table 7: NDCG@10 gain over the Qwen3-Emb+BM25 hybrid by query type. All four survive Holm correction.
<table><tr><td>Answer reachable from</td><td>Query type</td><td>Queries</td><td>∆NDCG@10</td></tr><tr><td>Card text</td><td>General</td><td>90</td><td>+0.075</td></tr><tr><td></td><td>Fine-tuning</td><td>40</td><td>+0.105</td></tr><tr><td>Relations only</td><td>Paper reference</td><td>40</td><td>+0.206</td></tr><tr><td></td><td>Entity neighbor</td><td>30</td><td>+0.197</td></tr></table>

Table 8: Sensitivity to the agent backbone.
<table><tr><td>Backbone</td><td>NDCG@5</td><td>NDCG@10</td><td>MAP@10</td></tr><tr><td>GLM-5.2</td><td>0.6297</td><td>0.6148</td><td>0.6973</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.5417</td><td>0.5244</td><td>0.5702</td></tr><tr><td>GLM-4-9B</td><td>0.4809</td><td>0.4706</td><td>0.5750</td></tr></table>

## D Extended Retrieval Results

The complete 15-system comparison is reported in the main text (Table 2). This appendix reports per-query-type retrieval gains, the agent-backbone sweep, and the Critic-free graph experiment.

Table 7 lists the per-type gains summarized in Section 5. The two semantic types, which together account for 130 of the 200 queries, gain least. The two relational types, whose gold answers are reachable only through the graph, gain twice as much. The four gains weight-average to the +0.126 overall margin of Table 2. Because the relational types carry a near-binary label distribution (Table 6) and rest on only 40 and 30 queries, their margins are more volatile than the semantic types’, so we read the split between the two families rather than the gap between the relational rows.

Table 8 also bounds the architecture’s claim. It varies only the agent backbone, holding the retrieval stack and all contracts fixed. The contracts constrain what the Planner and Critic may emit, not how well they reason: with Qwen3.5-35B-A3B the system still clears every retrieval baseline (0.5244 NDCG@10), but with GLM-4-9B it falls to 0.4706, level with BM25 (0.4717) and below the hybrid baseline (0.4893). The contract-governed loop therefore converts backbone capability into ranking quality without substituting for it, and a frontier-scale Critic is a prerequisite, not an optimization.

## E Significance Testing

We follow standard practice for ranking comparisons [36]. Retrieval comparisons take the per-query diference against AgenticDS (Qwen3-Emb) and run a two-sided paired permutation test with Holm correction, with paired bootstrap intervals stratified by query type. Training comparisons block by the nine (domain, base model) pairs, average benchmarks within a domain, and enumerate all 2<sup>9</sup> sign assignments, making that test exact rather than sampled.

Outcomes. All fourteen retrieval comparisons are significant on NDCG@5, NDCG@10, and NDCG@20. On MAP the comparisons against Claude Code and Codex are not, with Holm � between 0.13 and 0.45, and neither are the diferences between our two variants. Correcting jointly across metrics leaves every NDCG@10 comparison significant except the one against our own bge-m3 variant (� = 0.063), and the two LLM agents do not separate from each other $( \Delta = 0 . 0 0 3 , p = 1 . 0 0 )$ . On the training side the four comparisons against AgenticDS give raw � of 0.0195 (Claude Code),

Table 9: Evaluation split of every benchmark.
<table><tr><td>Domain</td><td>Benchmark</td><td>Items</td><td>Metric</td></tr><tr><td rowspan="4">Math</td><td>GSM8K</td><td>1,319</td><td>exact match</td></tr><tr><td>MATH</td><td>5,000</td><td>exact match</td></tr><tr><td>AMC23</td><td>40</td><td>exact match</td></tr><tr><td>OCWCourses</td><td>272</td><td>exact match</td></tr><tr><td rowspan="2">Code</td><td>HumanEval</td><td>164</td><td>pass@1</td></tr><tr><td>BigCodeBench</td><td>1,140</td><td>pass@1</td></tr><tr><td rowspan="2">Medicine</td><td>MedQA</td><td>1,273</td><td>accuracy</td></tr><tr><td>MMLU-Medical</td><td>1,784</td><td>accuracy</td></tr></table>

0.0039 (Codex), 0.0078 (hybrid), and 0.0039 (monoT5), all significant after Holm correction, and a Wilcoxon signed-rank test agrees.

Scope. The training result is sensitive to the pairing unit, since collapsing to three domain-level units leaves only 2<sup>3</sup> sign assignments and a smallest two-sided � of 0.25, so we report every grouping rather than the best. More broadly, every system contributes a single run, the 200 queries come from graph anchors and are not fully independent, the graded labels are silver, and the training side has no independent seeds, none of which Holm correction repairs.

## F Training-Study Details

## F.1 Benchmark configuration

Table 9 fixes the evaluation split and metric of every benchmark so that absolute numbers are comparable with other work. All scores are normalized to the 0–100 range.

We evaluate the 272-problem OCWCourses set of [26], not the 5,000-problem MATH split that some harnesses report under the same name. MMLU-Medical is the union of nine subjects from the cais/mmlu test splits, used as published: anatomy, clinical knowledge, college biology, college medicine, human aging, medical genetics, nutrition, professional medicine, and virology (1,784 items).

## F.2 Training grid and search queries

For training interventions, every method uses the same eligibility rules, � = 2000 per dataset, an identical optimizer, batch size = 4, and a grid of three epochs and learning rates in $\{ 1 0 ^ { - 5 } , 5 { \times } 1 0 ^ { - 6 } , 1 0 ^ { - 6 } \}$ chosen on a held-out development split.

The math query reads “advanced competition mathematics instructiontuning data with dificult problems, complete derivations, and verified boxed answers.” The code query reads “competitive programming instruction-tuning data with problem statements, test cases, accepted code solutions, and code generation supervision.” The medicine query reads “medical question-answering SFT data with professional multiple-choice questions, clinical explanations, and final answers across broad specialties.”

## F.3 Per-model scores

In the training study, AgenticDS denotes the Qwen3-Emb variant.   
Table 10 expands the means of Table 3 into their nine blocks.

## F.4 Ablation grid

Figure 4 gives the full three-base × three-domain × five-ablation grid, computed under the eight-benchmark grid of Section 4 with the per-benchmark best learning rate, then averaged within a domain. Table 4 aggregates these 45 blocks into the five row means and standard errors discussed in Section 5.

<table><tr><td rowspan="2"></td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="3">Medicine</td><td rowspan="2">Mean 9-cell</td><td rowspan="2">helps</td></tr><tr><td>Qwen2.5</td><td>Llama-3.1</td><td>Olmo-3</td><td>Qwen2.5</td><td>Llama-3.1</td><td>Olmo-3</td><td>Qwen2.5</td><td>Llama-3.1</td><td>Olmo-3</td></tr><tr><td>w/o Critic</td><td>-10.0</td><td>-5.4</td><td>-0.8</td><td>-1.3</td><td>-3.8</td><td>+0.9</td><td>+3.7</td><td>+1.4</td><td>-1.5</td><td>-1.85</td><td>+10</td></tr><tr><td>w/o Planner</td><td>-1.8</td><td>-2.9</td><td>-0.7</td><td>+2.2</td><td>+1.7</td><td>+0.0</td><td>+2.8</td><td>-6.0</td><td>-0.8</td><td>-0.60</td><td></td></tr><tr><td>w/o KG Expansion</td><td>+0.2</td><td>+0.7</td><td>-2.7</td><td>+2.4</td><td>-0.5</td><td>-0.5</td><td>+2.2</td><td>-1.7</td><td>+0.7</td><td>+0.10</td><td>0</td></tr><tr><td>w/o Adaptive Budget</td><td>-2.0</td><td>+0.4</td><td>+2.0</td><td>+3.3</td><td>+1.0</td><td>+0.5</td><td>+3.5</td><td>-4.6</td><td>-0.3</td><td>+0.43</td><td></td></tr><tr><td>w/o Sample Evidence</td><td>-3.8</td><td>-2.6</td><td>-0.4</td><td>+2.2</td><td>+2.9</td><td>+1.5</td><td>-1.0</td><td>-7.9</td><td>-3.8</td><td>-1.43</td><td>-10</td></tr><tr><td>Full AgenticDS:</td><td>56.9</td><td>32.6</td><td>59.9</td><td>56.8</td><td>37.6</td><td>54.5</td><td>62.6</td><td>61.2</td><td>53.2</td><td></td><td>hurts</td></tr></table>

Figure 4: Component ablation under the search-to-train protocol, all base models and domains. Each block is the change in domain accuracy when a component is removed. The full system’s score is printed beneath each column.

Table 10: Per-model domain scores. Bold: best per row.
<table><tr><td>Domain</td><td>Model</td><td>Qwen3-Emb+BM25</td><td>monoT5 (OOD)</td><td>Claude Code</td><td>Codex</td><td>AgenticDS</td></tr><tr><td rowspan="3">Math</td><td>Qwen</td><td>49.63</td><td>46.40</td><td>39.60</td><td>45.84</td><td>56.90</td></tr><tr><td>Llama</td><td>29.34</td><td>21.79</td><td>29.20</td><td>19.45</td><td>32.64</td></tr><tr><td>Olmo</td><td>58.90</td><td>55.21</td><td>58.30</td><td>57.98</td><td>59.93</td></tr><tr><td rowspan="3">Code</td><td>Qwen</td><td>56.58</td><td>52.79</td><td>55.48</td><td>53.56</td><td>56.80</td></tr><tr><td>Llama</td><td>36.86</td><td>36.69</td><td>34.32</td><td>32.23</td><td>37.55</td></tr><tr><td>Olmo</td><td>54.20</td><td>45.09</td><td>51.19</td><td>52.71</td><td>54.50</td></tr><tr><td rowspan="3">Medicine</td><td>Qwen</td><td>62.59</td><td>61.54</td><td>63.47</td><td>62.17</td><td>62.58</td></tr><tr><td>Llama</td><td>48.41</td><td>60.31</td><td>57.99</td><td>59.28</td><td>61.16</td></tr><tr><td>Olmo</td><td>51.28</td><td>52.51</td><td>54.08</td><td>51.52</td><td>53.25</td></tr></table>

Blue blocks mean that removing the component costs accuracy and red blocks that removing it gains accuracy. Removal helps in 21 of the 45 blocks, and the per-block changes scatter widely enough (standard deviation 1.7–4.1 points) that the standard error of any nine-block mean is 0.6–1.4 points. No component clears two standard errors, so the means for KG expansion (+0.10) and the adaptive budget (+0.43) should be read as unresolved rather than as evidence against those modules. The useful content of this figure is the dispersion itself, which is why Section 5 reports the aggregate with its standard error rather than per-block wins.

## G Implementation and Cost Comparison

Full baseline inventory. Beyond the four strong baselines detailed in Section 4, nine more span four families.

• Platform search: Google Dataset Search [3] and HuggingFace full-text search.<sup>2</sup>

• Sparse: BM25 [35] and BM25+RM3 [1].

• Dense: bge-m3 [5] and Qwen3-Embedding-0.6B [46].

• Hybrid: bge-m3+BM25, MDR-BM25 [14], and ANCE+BM25 [41]. All hybrids fuse their two branches with RRF [16] under the same smoothing constant, and monoT5 and ANCE are marked (OOD) because both are applied zero-shot.

Table 11 prices every system against the same GLM-5.2 list, \$1.19 per million uncached input tokens, \$0.30 per million cached input tokens, and \$4.17 per million output tokens.

Dense search uses separate FAISS indexes for descriptions and concatenated metadata, while sparse search uses a Lucene BM25 index. All hyperparameters (RRF smoothing, dense/sparse fusion weight, �<sub>llm</sub>, acceptance thresholds) were tuned on an internal pilot set drawn before HuDS-Bench was released, and no query from the 200-query release is used for tuning.

Table 11: End-to-end cost of the 200-query evaluation. A dash marks no cached-prompt reuse, so AgenticDS is billed entirely at the uncached rate and its cost is an upper bound.
<table><tr><td></td><td></td><td colspan="3">Billed tokens (M)</td><td colspan="2">Cost (USD)</td></tr><tr><td>System</td><td>s/query</td><td>in, unc.</td><td>in, cached</td><td>output</td><td>total</td><td>per query</td></tr><tr><td>AgenticDS (Qwen3-Emb)</td><td>150.5</td><td>10.6</td><td>一</td><td>1.7</td><td>19.86</td><td>0.099</td></tr><tr><td>AgenticDS (bge-m3)</td><td>212.7</td><td>13.1</td><td>一</td><td>2.1</td><td>24.35</td><td>0.122</td></tr><tr><td>Claude Code</td><td>139.3</td><td>13.1</td><td>97.2</td><td>1.0</td><td>48.78</td><td>0.244</td></tr><tr><td>Codex</td><td>868.5</td><td>20.0</td><td>258.1</td><td>2.8</td><td>112.40</td><td>0.562</td></tr></table>

LLM agent baseline configuration. We implement the two LLM agents with the Claude Code 2.1.228 and Codex 0.147.0 harnesses, both using GLM-5.2 and the same HuggingFace hub tools.

Each agent was invoked with (1) the raw HuDS-Bench query as the user turn, (2) a system prompt requesting 30 unique, ranked HuggingFace dataset identifiers satisfying the query, (3) shared readonly MCP tools for dataset search and metadata/card inspection, and (4) a JSON output schema for the ranked identifiers. Outputs are normalized, deduplicated, and filtered once against the evaluation snapshot to retain up to 20 datasets, and short lists remain unfilled. Each query was run once, with retries only upon execution errors, so the comparison is single-shot. System prompts, tool specifications, and raw response logs are released with HuDS-Bench.

## H Case Studies

Two cases show how contracts organize evidence. (1) For the finetuning query “instruction tuning data with code commit messages as natural language instructions across programming languages”, the Planner separates code generation, supervision format, and commit provenance into complementary hypotheses. The Engineer’s dense branch returns four commit-oriented corpora in the top-10, two of which the Critic rejects because their sample fields show synthetic rather than real-commit prompts, and those rejections guide a local repair hypothesis. (2) For the general query “large Hungarian-only medical consultation data”, none of the top five results is relevant. The Critic diagnoses unmet language and domain requirements, and local repair must preserve the Hungarian-only constraint, which broader Finno-Ugric or English candidates cannot satisfy. Reporting an unresolvable constraint is the correct behavior here, and separates AgenticDS from both commercial agent baselines, which silently returned near-miss English or Spanish corpora.

## I Limitations and Future Work

Limitations. (1) The two relational query types are built from HuggingKG anchors that the Engineer can also traverse. The design bounds what that overlap can buy: 130 of the 200 queries carry no KG-derived gold label, every non-KG baseline is given Hug gingFace’s full-text search, and the variant without KG expansion still reaches 0.5307 NDCG@10, above every non-agent baseline. Extending construction to KG-free gold labels would remove the overlap entirely. (2) With one query per domain across three base models, the utility study is powered to separate acquisition methods in aggregate rather than individual components. Its scope is English queries on a single ML hub, at a latency suited to ofline acquisition rather than interactive search. (3) The agreement study reports a single � over all four query types. Because the relational types rarely use the middle grade, that figure speaks mainly for the semantic types, and per-type coeficients would be more informative.

Future work. Four directions follow. Sharpening the searchto-train protocol calls for more queries per domain and repeated seeds rather than more benchmarks. A gold-label pipeline that never consults the graph would settle the overlap above. Distilling the Critic into a smaller pairwise scorer attacks the dominant cost. Finally, only the Engineer is tied to a hub, so any ecosystem with typed metadata and a navigable relation graph could be served by swapping it. A federation layer routing typed constraints to several hubs would test the contracts outside HuggingFace.