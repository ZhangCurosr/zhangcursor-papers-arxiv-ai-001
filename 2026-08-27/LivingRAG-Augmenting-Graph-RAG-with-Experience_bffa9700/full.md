# LivingRAG: Augmenting Graph RAG with Experience

Yuzhuo Cui<sup>1,2</sup>, Zongye Zhang<sup>1,2</sup>, Qingjie Liu<sup>1,2,\*</sup>

<sup>1</sup>State Key Laboratory of Virtual Reality Technology and Systems,

Beihang University, Beijing, China

<sup>2</sup>Hangzhou Innovation Institute, Beihang University,

Hangzhou, Zhejiang, China

{cyz020403,zhangzongye,qingjie.liu}@buaa.edu.cn

<sup>\*</sup>Corresponding author

## Abstract

Graph-based RAG improves multi-hop question answering by organizing evidence as a knowledge graph. However, most existing RAG systems process each query in isolation and discard useful reasoning from the LLM’s response after inference. As a result, later related queries need to retrieve evidence and reason from scratch. We propose LivingRAG, a Graph RAG framework with writable and reusable reasoning experience. LivingRAG adds a writable experience store to a graphbased retrieval backbone, enabling verified experiences to be reused during inference in two ways. Stored graph signals help retrieval find entities and passages that were useful in earlier related queries. Stored summaries provide a reference reasoning pattern for answer generation. We analyze online QA streams and find reusable signals from shared entities, graph neighborhoods, and question templates. Experiments on multi-hop QA benchmarks show that LivingRAG improves accuracy over strong RAG baselines and reduces completion-token use when relevant prior experience is reused.

## 1 Introduction

Large language models perform well on many language understanding and reasoning tasks. However, they still face limits in knowledge-intensive tasks. They may hallucinate facts. They may also lack access to domain-specific or up-to-date knowledge. Retrieval-augmented generation addresses these limits by grounding model outputs in external documents (Lewis et al., 2020; Karpukhin et al., 2020; Gao et al., 2024).

Graph-based RAG further improves retrieval for complex questions. It represents passages, entities, and their relations in a graph. The graph structure helps the retriever find multi-hop evidence that dense retrieval may miss. Recent systems show that graph structure is useful for global context and entity-level reasoning (Edge et al., 2024; Guo et al.,

2024; Gutiérrez et al., 2025; Zhuang et al., 2025; Luo et al., 2026b). However, online QA streams raise a further question. When related queries arrive over time, should the system reuse useful reasoning from earlier queries?

Despite this progress, most RAG and Graph RAG systems still answer each query independently at inference time (Edge et al., 2024; Guo et al., 2024; Gutiérrez et al., 2025; Zhuang et al., 2025; Luo et al., 2026b). These methods retrieve evidence for the current query and then generate an answer, but they discard most query-level reasoning signals after inference. As a result, later related queries often need to retrieve and reason again from scratch. This read-only inference process limits reuse in online QA streams.

This limitation matters because useful reuse is not limited to repeated entities. Two questions may visit nearby regions of the retrieval graph. They may also follow the same reasoning pattern with different entities. For example, two comparison questions may both ask which person was born earlier, even when the people and evidence passages differ. Therefore, a useful online Graph RAG system should preserve more than final answers. It should preserve graph-neighborhood priors and reusable reasoning patterns.

To address this problem, we introduce LivingRAG, a Graph RAG framework with writable and reusable reasoning experience. LivingRAG adds a writable experience store to a graph-based retrieval backbone. After each query, it can write a compact experience if the experience is grounded and novel. Each stored experience keeps the source query, activated graph entities, a short reasoning summary, and validation metadata. During later inference, LivingRAG reuses stored experiences in two ways. Activation maps guide graph retrieval toward useful evidence neighborhoods. Reasoning summaries provide scaffolds for answer generation.

A writable experience store also introduces a risk. If unsupported reasoning is written into the store, later queries may reuse it and amplify the error. LivingRAG reduces this risk with quality gates. A candidate is stored only when its claims are supported by retrieved evidence and when it is not redundant with stored experiences. Thus, LivingRAG can return an answer without storing low-quality experience.

We evaluate LivingRAG on multi-hop QA benchmarks. The experience store is initialized at the beginning of each run and updated online during that run. Across these benchmarks, LivingRAG improves accuracy over strong RAG baselines. It also reduces completion-token use and estimated API cost when later queries retrieve useful prior experience.

Our contributions are: (1) We introduce LivingRAG, a Graph RAG framework with writable and reusable reasoning experience. It stores grounded and novel experiences, and reuses them through activation maps for retrieval and reasoning summaries for generation. (2) We analyze online QA streams and find reusable signals from shared entities, graph neighborhoods, and question templates. We show that LivingRAG can reuse these signals in both multi-hop QA and support-style QA settings. (3) We show that LivingRAG improves accuracy over strong RAG baselines. It also reduces completion-token use and estimated API cost when relevant prior experience is reused.

## 2 Related Work

Retrieval-augmented generation grounds language models in external documents and reduces reliance on parametric knowledge (Lewis et al., 2020; Karpukhin et al., 2020; Gao et al., 2024). Early systems usually retrieve independent text chunks. Recent Graph RAG methods add graph structure to support complex retrieval. GraphRAG builds entity graphs and community summaries (Edge et al., 2024). LightRAG combines graph indexing with vector retrieval (Guo et al., 2024). HippoRAG and HippoRAG2 use graph-based associative retrieval (Gutiérrez et al., 2024; Gutiérrez et al., 2025). LinearRAG uses a lightweight passagesentence-entity graph for efficient retrieval (Zhuang et al., 2025). These systems show that graph structure helps multi-hop evidence search. LivingRAG builds on this direction, but studies a different problem. It asks how a Graph RAG system can write and reuse verified reasoning experience across an

online query stream.

Several methods also improve RAG with reasoning or memory. Reasoning-enhanced RAG methods use decomposition, intermediate notes, selfreflection, or tool use to improve single-query reasoning (Trivedi et al., 2023; Jiang et al., 2023; Asai et al., 2024; Yu et al., 2024; Kabir et al., 2025). Memory-based agents store past observations or feedback for later decisions (Shinn et al., 2023; Madaan et al., 2023). Recent memory-augmented RAG systems further show that reusable memory can support retrieval. REMem builds a hybrid episodic memory graph for language-agent histories and uses an agentic retriever for episodic recollection and reasoning (Shu et al., 2026). MemoRAG builds a global memory for long-context processing and uses it to produce clues for retrieval (Qian et al., 2025). LivingRAG is complementary to these works. It focuses on online Graph RAG for QA streams. Its written unit is a verified reasoning experience, not a raw interaction history or a global long-context memory. Each experience stores graph activation information and a compact reasoning scaffold. It is written only after grounding and novelty checks. The experience is not only retrieved as text. It also changes the initial graph activation used by the retriever.

## 3 Method

## 3.1 Motivating Analysis

LivingRAG is designed for online QA streams. In this setting, the system answers queries one by one. For a query q<sub>t</sub>, only earlier queries $q _ { 1 } , \ldots , q _ { t - 1 }$ can be used as reuse sources. This causal setup matches an online experience store and avoids using future information.

We measure three reuse signals. Entity reuse occurs when the current query shares at least one seed entity with an earlier query. Graph-neighborhood reuse occurs when the current query and an earlier query visit nearby regions of the retrieval graph. We measure this signal with a fixed Linear-RAG retriever. Two queries are counted as graphneighborhood reuse when their retrieved passages or filtered context entities have Jaccard similarity of at least 0.05. Template reuse occurs when two queries have no seed-entity overlap but have similar masked questions. We mask entity mentions, quoted spans, and numeric as Section 3.4. We then compute TF-IDF cosine similarity between the masked questions. Two queries are counted as template reuse when the score is at least 0.35.

Table 1: Reuse signals in online QA streams. Only earlier queries are used as reuse sources. All values are percentages.
<table><tr><td>Dataset</td><td>Entity reuse</td><td>Graph-neigh. reuse</td><td>Template reuse</td></tr><tr><td>2Wiki</td><td>4.30</td><td>99.70</td><td>96.30</td></tr><tr><td>HotpotQA</td><td>26.53</td><td>99.10</td><td>21.82</td></tr><tr><td>MuSiQue</td><td>40.34</td><td>95.60</td><td>39.14</td></tr><tr><td>MuSiQue-full</td><td>54.10</td><td>96.32</td><td>54.30</td></tr><tr><td>WixQA</td><td>76.69</td><td>80.45</td><td>16.04</td></tr></table>

Table 1 shows that reusable signals exist across online QA streams, but they appear in different forms. Direct entity reuse is limited in most datasets, so an entity-only cache would miss many opportunities. In contrast, graph-neighborhood reuse is common across datasets. This suggests that later queries often need evidence from regions that are close to earlier retrieval paths.

Template reuse is also useful, but it is more dataset-dependent. It is stronger in formulaic multihop streams and weaker in more diverse supportstyle questions. These results motivate LivingRAG to reuse experience through two paths.

## 3.2 Overview

LivingRAG extends a graph-based RAG system with a writable experience store. The base retriever follows LinearRAG (Zhuang et al., 2025). Given a document corpus D, it builds a retrieval graph with passage nodes, sentence nodes, and entity nodes. The graph is constructed with lightweight entity extraction and does not require LLM calls during indexing. For each query, LinearRAG first activates query-related entities, propagates relevance over the sentence-entity graph, and then ranks passages with Personalized PageRank over the passage-entity graph. LivingRAG keeps this retrieval backbone unchanged. It adds a writable experience store during an online QA stream. LivingRAG does not reuse previous answers as a response cache. It reuses verified activation maps and reasoning scaffolds during retrieval and generation. Figure 1 gives an overview of the full workflow.

At time step t, the system receives a query q<sub>t</sub> and has access to a static retrieval graph $\mathcal { G }$ and an experience store $\mathcal { E } _ { t }$ . A standard graph-based RAG system answers each query independently:

$$
a _ { t } = \operatorname { L L M } \big ( q _ { t } , \mathcal { R } ( q _ { t } , \mathcal { G } ) \big ) ,\tag{1}
$$

where $\mathcal { R }$ retrieves passages from the graph. Livin-

gRAG instead uses stored experiences during both retrieval and generation:

$$
\begin{array} { r l } & { a _ { t } = \mathrm { L L M } \big ( q _ { t } , \mathcal { R } _ { \exp } ( q _ { t } , \mathcal { G } , \mathcal { E } _ { t } ) , } \\ & { \qquad \quad S _ { \exp } ( q _ { t } , \mathcal { E } _ { t } ) \big ) . } \end{array}\tag{2}
$$

Here $\mathcal { R } _ { \exp }$ is experience-augmented graph retrieval, and $\mathcal { S } _ { \mathrm { e x p } }$ is an optional reasoning scaffold selected from the experience store. If no useful stored experience is found, LivingRAG reduces to the base graph RAG system.

After answering query $q _ { t } .$ , LivingRAG forms a candidate experience $C _ { t }$ . The candidate contains the query, retrieved passages, generated explanation, and final answer. LivingRAG then decides whether the candidate should be written to the store. If the candidate passes the grounding and novelty checks, it becomes a verified experience $e _ { t }$ . We use $C _ { t }$ for the candidate before validation and $e _ { t }$ for the verified experience after validation. The experience-store update is:

$$
\begin{array} { r } { \mathcal { E } _ { t + 1 } = \left\{ \begin{array} { l l } { \mathcal { E } _ { t } \cup \{ e _ { t } \} , } & { \mathrm { i f } \mathrm { S t o r e } ( C _ { t } , \mathcal { E } _ { t } ) = 1 , } \\ { \mathcal { E } _ { t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{3}
$$

The function Store checks whether the candidate is grounded in retrieved evidence and sufficiently different from stored experiences. This write-back mechanism lets LivingRAG accumulate reusable experiences without changing model parameters.

## 3.3 Experience Store

The basic unit of the store is a verified experience. After this definition, we use experience as a short form when the meaning is clear. Each verified experience is a compact form of one candidate experience that passed validation:

$$
e _ { t } = \left( q _ { t } , \mathbf { v } _ { t } , \tilde { \mathbf { a } } _ { t } , \mathrm { s u m } _ { t } , a _ { t } , t s _ { t } , \mathrm { c o n f } _ { t } \right) .\tag{4}
$$

Here $q _ { t }$ is the source query, $\mathbf { v } _ { t } = \operatorname { E m b } ( q _ { t } )$ is its query embedding, $\tilde { \mathbf { a } } _ { t }$ is a sparse activation map over graph entities, sum<sub>t</sub> is a compact reasoning summary, $a _ { t }$ is the final answer, $t s _ { t }$ is the timestamp, and conf<sub>t</sub> is the grounding confidence.

The activation map is the main graph-space representation of a verified experience. After the base retriever finishes entity propagation for query $q _ { t } .$ it produces a final entity activation vector $\mathbf { a } _ { q _ { t } } ^ { \mathrm { f i n a l } }$ LivingRAG sparsifies this vector:

$$
\tilde { \mathbf { a } } _ { t } = \mathrm { s p a r s i f y } ( \mathbf { a } _ { q _ { t } } ^ { \mathrm { f i n a l } } , \epsilon ) ,\tag{5}
$$

![](images/b116eeab36d2921e6adb01660a1daa8d95d25c4ac6bdec326ee0b322f5ee1eca.jpg)  
Figure 1: Overview of LivingRAG. The framework first constructs a static retrieval graph offline. It then performs experience-augmented retrieval, scaffolded generation, and write-back through quality gates.

where entries below threshold ϵ are set to zero. The sparse activation map stores which graph entities were useful for the candidate and how strongly they were activated. Unlike a plain entity list, the map preserves graded graph-neighborhood information and can be reused directly in later graph retrieval.

The compact reasoning summary becomes a reasoning scaffold during later generation. It is not the full raw reasoning trace. To keep the prompt short, LivingRAG selects a small number of key sentences from the generated explanation. The selection is anchored by the most activated entities in $\mathbf { a } _ { q _ { t } } ^ { \mathrm { f i n a l } }$ . Sentences that mention more activated entities are ranked higher, and the selected sentences are kept in their original order. The stored summary is:

$$
\mathrm { s u m } _ { t } = \bigl ( q _ { t } , \mathrm { K e y S e n t } ( r _ { t } , \mathbf { a } _ { q _ { t } } ^ { \mathrm { f n a l } } ) , a _ { t } \bigr ) ,\tag{6}
$$

where $r _ { t }$ is the generated explanation. The compact summary preserves the main reasoning pattern while adding few tokens to future prompts.

## 3.4 Experience-Augmented Retrieval and Scaffolded Generation

LivingRAG uses the experience store in two places. First, it uses activation maps to guide graph retrieval. Second, it uses compact summaries as reasoning scaffolds for answer generation. The two paths are related but separate. The retrieval path transfers graph-neighborhood information. The generation path transfers a reusable reasoning pattern.

For graph retrieval, LivingRAG first matches the current query with stored experiences. Each stored experience is scored by combining semantic similarity and activation-map similarity:

$$
\begin{array} { r l } & { s _ { k } ^ { \mathrm { r e t } } = \alpha \cos \bigl ( \mathrm { E m b } ( q _ { t } ) , \mathbf { v } _ { k } \bigr ) } \\ & { ~ + \left( 1 - \alpha \right) \cos \bigl ( \mathbf { a } _ { q _ { t } } ^ { \mathrm { b a s e } } , \tilde { \mathbf { a } } _ { k } \bigr ) . } \end{array}\tag{7}
$$

Here $\mathbf { a } _ { q _ { t } } ^ { \mathrm { b a s e } }$ is the initial entity activation vector produced from the current query before stored experiences are used. The first term captures surface-level query similarity. The second term captures whether the current query starts near an evidence neighborhood that was useful before. LivingRAG retrieves the top-K experiences according to $s _ { k } ^ { \mathrm { r e t } }$

The retrieved activation maps are then fused into the initial entity activation vector:

$$
\mathbf { a } _ { q _ { t } } ^ { 0 } = \mathrm { n o r m a l i z e } \left( \mathbf { a } _ { q _ { t } } ^ { \mathrm { b a s e } } + \beta \sum _ { e _ { k } \in \mathcal { N } _ { t } ^ { \mathrm { r e t } } } w _ { k } \tilde { \mathbf { a } } _ { k } \right) .\tag{8}
$$

We first compute a softmax weight over the selected experiences:

$$
\bar { w } _ { k } = \frac { \exp ( s _ { k } ^ { \mathrm { r e t } } / \tau _ { s } ) } { \sum _ { j \in \mathcal { N } _ { t } ^ { \mathrm { r e t } } } \exp ( s _ { j } ^ { \mathrm { r e t } } / \tau _ { s } ) } .\tag{9}
$$

We then scale this weight by the confidence of the stored experience:

$$
w _ { k } = \bar { w } _ { k } \cdot \mathrm { c o n f } _ { k } .\tag{10}
$$

The scaled weights are not renormalized. Thus, low-confidence experiences have less influence on the fused activation vector. The resulting vector $\mathbf { a } _ { q _ { t } } ^ { 0 }$ replaces the original initial activation vector of the base retriever. All later retrieval steps are unchanged. When the experience store is empty, or when the current query has no seed entities for graph fusion, LivingRAG falls back to the base retriever.

For generation, LivingRAG may also select a reasoning scaffold. Scaffold selection can use a template signal because the selected scaffold is added only to the LLM prompt and does not inject historical entities into graph retrieval. For a query $q ,$ we define a masked template:

$$
T ( q ) = \mathrm { M a s k } ( q ) ,\tag{11}
$$

where entity mentions, quoted spans, and numeric expressions are replaced with placeholders. The scaffold score is:

$$
\begin{array} { r l } & { s _ { k } ^ { \mathrm { s c a f } } = ( 1 - \lambda ) s _ { k } ^ { \mathrm { r e t } } } \\ & { \qquad + \lambda \cos \bigl ( \mathrm { E m b } ( T ( q _ { t } ) ) , \mathrm { E m b } ( T ( q _ { k } ) ) \bigr ) . } \end{array}\tag{12}
$$

When stored experiences are available, LivingRAG selects the highest-scoring experience as the scaffold:

$$
e _ { t } ^ { \mathrm { s c a f } } = \underset { e _ { k } \in \mathcal { E } _ { t } } { \arg \operatorname* { m a x } } s _ { k } ^ { \mathrm { s c a f } } .\tag{13}
$$

If the experience store is empty, or if no useful scaffold is selected, no scaffold is inserted. In the special case $\lambda = 0$ , scaffold selection reduces to the highest-scoring experience from experienceaugmented retrieval.

The LLM receives the retrieved passages and, when available, the selected scaffold:

$$
{ \mathrm { C o n t e x t } } _ { t } = \{ p _ { 1 } , \ldots , p _ { k } \} \cup \{ { \mathrm { s u m } } ( e _ { t } ^ { \mathrm { s c a f } } ) \} .\tag{14}
$$

The scaffold shows how a related problem was solved. The model can still use the current passages to produce a different answer when the new query requires different evidence.

## 3.5 Experience Write-Back

After generation, LivingRAG checks whether the new candidate experience should be stored. LivingRAG separates write-back validation from answer generation. The system can return an answer even when the candidate is not stored. This separation prevents unsupported or redundant reasoning from entering the experience store.

The first criterion is factual grounding. LivingRAG extracts atomic claims from the available reasoning trace and the final answer. Each claim is checked against the retrieved passages with an NLI model:

$$
g _ { i } = \underset { p _ { j } \in \mathcal { D } _ { t } ^ { \mathrm { u s e d } } } { \operatorname* { m a x } } P _ { \mathrm { N L I } } ( \mathrm { e n t a i l } \mid p _ { j } , c l _ { i } ) .\tag{15}
$$

The grounding score is:

$$
\mathrm { G S } ( C _ { t } ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } { \mathcal { H } } [ g _ { i } \geq \tau _ { \mathrm { n l i } } ] .\tag{16}
$$

If $\mathrm { G S } ( C _ { t } ) < \tau _ { \mathrm { g s } }$ , the candidate is not stored. The grounding score measures evidential support. It does not explicitly measure whether a candidate will be useful for future questions.

The second criterion is novelty. LivingRAG avoids storing candidates that are near duplicates of stored experiences. Novelty is measured before experience fusion, using the current query embedding and its base activation vector:

$$
\begin{array} { r } { \eta _ { k } = \alpha \cos \bigl ( \mathrm { E m b } ( q _ { t } ) , \mathbf v _ { k } \bigr ) } \\ { + \left( 1 - \alpha \right) \cos \bigl ( \mathbf { a } _ { q _ { t } } ^ { \mathrm { b a s e } } , \tilde { \mathbf { a } } _ { k } \bigr ) , } \end{array}\tag{17}
$$

$$
\mathrm { N S } ( C _ { t } , { \mathcal { E } } _ { t } ) = 1 - \operatorname* { m a x } _ { e _ { k } \in { \mathcal { E } } _ { t } } \eta _ { k } .\tag{18}
$$

Using the base activation vector avoids circularity. A new candidate should not look redundant only because a previous stored experience influenced retrieval. If $\mathrm { N S } ( C _ { t } , \mathcal { E } _ { t } ) < \tau _ { \mathrm { n o v e l } }$ , LivingRAG returns the answer but does not store the candidate.

The write-back rule is:

$$
\begin{array} { r } { \mathrm { S t o r e } ( C _ { t } , \mathcal { E } _ { t } ) = 1 \quad \mathrm { i f f } \quad \mathrm { G S } ( C _ { t } ) \geq \tau _ { \mathrm { g s } } } \\ { \wedge \mathrm { N S } ( C _ { t } , \mathcal { E } _ { t } ) \geq \tau _ { \mathrm { n o v e l } } . } \end{array}\tag{19}
$$

The rule defines the storage condition. In implementation, LivingRAG checks novelty before grounding to avoid unnecessary NLI calls. $\mathsf { A p - }$ pendix D reports gate dynamics and examples of accepted and rejected candidates. For accepted candidates, the confidence score is set to the grounding score co $\mathrm { _ { l f } } _ { t } = \mathrm { G S } ( C _ { t } )$ . The new verified experience is then appended to the store and can be used by later queries.

## 3.6 Complexity

LivingRAG preserves the static graph construction and graph retrieval backbone of LinearRAG. Therefore, the corpus-side complexity remains unchanged. For each query, the added retrieval cost comes from matching the query against stored experiences. If the experience store contains $| \mathcal { E } _ { t } |$ experiences and the average number of nonzero entries in an activation map is z¯, the added retrieval cost is $O ( | \mathcal { E } _ { t } | d + | \mathcal { E } _ { t } | \bar { z } )$ , where d is the embedding dimension. Since only the top-K experiences are fused, and K is fixed, the experience-augmented retrieval overhead is linear in the store size and independent of the corpus graph size.

Table 2: Overall accuracy (%) on QA benchmarks. Cont.-Acc. denotes Contain-Match Accuracy. WixQA is reported only for LinearRAG and LivingRAG; unavailable entries are marked with –. The best result is marked in bold, and the second-best result is underlined.
<table><tr><td>Method</td><td colspan="2">2Wiki</td><td colspan="2">HotpotQA</td><td colspan="2">MuSiQue</td><td colspan="2">MuSiQue-full</td><td>WixQA</td></tr><tr><td></td><td>Cont.-Acc. LLM-Acc.</td><td></td><td></td><td>Cont.-Acc. LLM-Acc.</td><td>Cont.-Acc.</td><td>LLM-Acc.</td><td>Cont.-Acc. LLM-Acc. LLM-Acc</td><td></td><td></td></tr><tr><td>Vanilla RAG (Top-5) (Karpukhin et al., 2020)</td><td>58.80</td><td>59.40</td><td>65.20</td><td>77.10</td><td>34.00</td><td>43.70</td><td>27.35</td><td>41.33</td><td>一</td></tr><tr><td>LightRAG (Guo et al., 2024)</td><td>60.60</td><td>48.20</td><td>62.20</td><td>71.70</td><td>39.20</td><td>49.10</td><td>29.83</td><td>44.89</td><td></td></tr><tr><td>GFM-RAG (Luo et al., 2026b)</td><td>75.50</td><td>77.80</td><td>72.50</td><td>86.10</td><td>42.70</td><td>52.50</td><td>37.28</td><td>47.75</td><td>一</td></tr><tr><td>HippoRAG2 (Gutiérrez et al., 2025)</td><td>76.50</td><td>68.00</td><td>71.20</td><td>84.60</td><td>45.40</td><td>52.90</td><td>40.92</td><td>44.06</td><td></td></tr><tr><td>LinearRAG (Zhuang et al., 2025)</td><td>79.60</td><td>84.40</td><td>72.90</td><td>86.80</td><td>48.50</td><td>60.70</td><td>41.00</td><td>52.71</td><td>66.00</td></tr><tr><td>LivingRAG (Ours)</td><td>82.70</td><td>87.00</td><td>73.20</td><td>88.00</td><td>50.80</td><td>64.40</td><td>44.39</td><td>58.42</td><td>70.25</td></tr></table>

Write-back validation is also selective. Novelty scoring uses the same lightweight store scan. Grounding verification is executed only for candidates that pass the novelty gate. If the candidate contains M claims and the validator keeps $K _ { g }$ candidate evidence sentences per claim, the write-back overhead is:

$$
\begin{array} { r l } { T _ { \mathrm { w r i t e } } ( q _ { t } ) = O ( | \mathcal { E } _ { t } | ( d + \bar { z } ) ) } & { } \\ { + \mathcal { H } _ { \mathrm { n o v e l } } O ( M K _ { g } ) . } \end{array}\tag{20}
$$

Thus, the NLI grounding cost is paid only for candidates that pass novelty filtering. The NLI cost is used for write-back validation. It does not change the base graph index and does not require additional LLM calls.

The additional storage is compact. Each stored experience contains a dense embedding, a sparse activation map, a short reasoning scaffold, and metadata. The experience-store overhead is $O ( | \mathcal { E } _ { t } | ( d + \bar { z } + \bar { \ell } ) )$ ), where <sup>¯</sup>ℓ is the average scaffold length. Appendix C provides the full time and space analysis.

When useful stored experiences are retrieved, LivingRAG can reduce generation cost in two ways. Activation maps can move retrieval toward relevant evidence earlier, and scaffolds can reduce the reasoning the LLM must reconstruct from scratch. We write the expected token cost as:

$$
\begin{array} { r } { \mathbb { E } [ \mathrm { C o s t } ( t ) ] = ( 1 - f _ { \mathrm { e x p } } ( t ) ) C _ { \mathrm { b a s e } } } \\ { + f _ { \mathrm { e x p } } ( t ) C _ { \mathrm { e x p } } , \quad } \end{array}\tag{21}
$$

where $f _ { \mathrm { e x p } } ( t )$ is the fraction of queries with useful support from stored experiences. Since $C _ { \mathrm { e x p } }$ depends on the quality of retrieved experiences, LivingRAG does not assume that cost always decreases. It predicts lower cost when later queries can reuse relevant stored experiences, which we test in the experiments.

## 4 Experiments

## 4.1 Overall Accuracy on QA Benchmarks

We first evaluate whether LivingRAG improves end-to-end QA accuracy over representative RAG and GraphRAG baselines. Later sections isolate experience reuse, token use, and online efficiency.

Setup. We evaluate all methods on four multihop QA streams: 2WikiMultiHopQA, HotpotQA, MuSiQue, and MuSiQue-full. MuSiQue-full uses the 2,417-question answerable stream described in Appendix E. We additionally report WixQA only for LinearRAG and LivingRAG, since its LLM-based entity extraction is not suitable for the other baselines. Appendix F describes these choices. We compare LivingRAG with Vanilla RAG (Karpukhin et al., 2020), LightRAG (Guo et al., 2024), GFM-RAG (Luo et al., 2026b), HippoRAG2 (Gutiérrez et al., 2025), and Linear-RAG (Zhuang et al., 2025). All methods use Qwen3.6 Plus for answer generation. We report Contain-Match Accuracy and LLM-Evaluation Accuracy; WixQA uses only LLM-Evaluation Accuracy.

For LivingRAG, the experience store is initialized as empty at the beginning of each dataset and updated online during evaluation. Only earlier queries can contribute experience to later queries, following the online setting defined in Section 3.1.

Table 2 shows that LivingRAG achieves the best overall accuracy on the four multi-hop benchmarks and in the WixQA comparison against LinearRAG. Under our evaluation setting, the writable experience store does not weaken the base QA ability of the graph-RAG backbone. Since LivingRAG builds retrieval framework based on LinearRAG, later experiments use LinearRAG as the main baseline to better isolate the effect of experience reuse. Appendix K.3 reports a retrieval-level transfer test with HippoRAG2’s entity-passage PPR.

Table 3: Token consumption and cost of LinearRAG and LivingRAG. Prompt and completion tokens are per-query averages. Rel. ∆ columns report (LivingRAG − LinearRAG)/LinearRAG. Negative values indicate reductions. Run cost is computed with Qwen3.6 Plus pricing: \$0.50 per million prompt tokens and \$3.00 per million completion tokens.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2"># Queries</td><td colspan="3">Prompt tokens</td><td colspan="3">Completion tokens</td><td colspan="3">Run cost (USD)</td></tr><tr><td>Linear</td><td>Living</td><td>Rel. ∆</td><td>Linear</td><td>Living</td><td>Rel. ∆</td><td>Linear</td><td>Living</td><td>Rel. ∆</td></tr><tr><td>2WikiMultiHopQA</td><td>1,000</td><td>6,038.2</td><td>6,241.6</td><td>+3.4%</td><td>1,067.6</td><td>906.4</td><td>-15.1%</td><td>$6.22</td><td>$5.84</td><td>-6.1%</td></tr><tr><td>HotpotQA</td><td>1,000</td><td>5,779.8</td><td>5,985.8</td><td>+3.6%</td><td>941.9</td><td>767.0</td><td>-18.6%</td><td>$5.72</td><td>$5.29</td><td>-7.4%</td></tr><tr><td>MuŠiQue</td><td>1,000</td><td>5,888.4</td><td>6,092.4</td><td>+3.5%</td><td>2,098.8</td><td>1,641.7</td><td>-21.8%</td><td>$9.24</td><td>$7.97</td><td>-13.7%</td></tr><tr><td>MuSiQue-full</td><td>2,417</td><td>8,639.3</td><td>8,861.0</td><td>+2.6%</td><td>2,223.8</td><td>1,643.7</td><td>-26.1%</td><td>$26.57</td><td>$22.63</td><td>-14.8%</td></tr><tr><td>WixQA</td><td>400</td><td>6,633.0</td><td>7,412.8</td><td>+11.8%</td><td>1,653.8</td><td>1,378.1</td><td>-16.7%</td><td>$3.31</td><td>$3.14</td><td>-5.3%</td></tr><tr><td>Overall (weighted)</td><td>5,817</td><td>7,089.7</td><td>7,340.9</td><td>+3.5%</td><td>1,744.0</td><td>1,347.6</td><td>-22.7%</td><td>$51.05</td><td>$44.87</td><td>-12.1%</td></tr></table>

![](images/06b80ba0b76a941f10d93061d93b7841fff8495e71dd70c0093b1f8a3d4ffff2.jpg)  
Figure 2: Completion-token trends over online QA streams. Each panel corresponds to one dataset. The blue and orange curves show the average completion tokens of LinearRAG and LivingRAG in each stream segment. Green shading between the two curves marks segments where LivingRAG uses fewer completion tokens, while red shading marks segments where it uses more. The first 10% of each stream is expanded to show the startup stage.

## 4.2 Online Token Efficiency

We next study whether online experience reuse reduces token use and estimated cost. We use the same datasets and online setting as in Section 4. Since LivingRAG uses LinearRAG as its retrieval backbone, we compare only these two systems. This comparison isolates the effect of the writable experience store.

Setup. We record prompt tokens, completion tokens, and total run cost for each run. Prompt tokens measure the input context. Completion tokens measure the generated output. Cost is estimated with the Qwen3.6 Plus prices used in Table 3.

We report prompt tokens, completion tokens, and total run cost. Prompt tokens correspond to the input context, while completion tokens correspond to generated output. We report them separately because they reflect different parts of inference and are priced differently by many LLM APIs. In our estimate, prompt tokens are charged at \$0.5 per million tokens, and completion tokens are charged at \$3.0 per million tokens.

Table 3 shows a consistent trade-off. LivingRAG increases prompt tokens on all datasets because it may add compact reasoning scaffolds or experience-related context. The weighted prompttoken increase is 3.5%. However, LivingRAG reduces completion tokens on all five datasets, with a weighted reduction of 22.7%. The reduction is largest on MuSiQue and MuSiQue-full, where queries often require longer multi-hop reasoning.

The total cost reduction is smaller than the completion-token reduction because the added prompt tokens partly offset the savings. Even so, LivingRAG reduces total run cost on every dataset. Overall, the estimated cost decreases from \$51.05 to \$44.87, a 12.1% reduction. Section 4.3 further analyzes which reuse paths are used during inference.

Figure 2 shows the online trend. In the earliest segments, LivingRAG has few stored experiences and may temporarily use more completion tokens. After experience accumulates, the area between the two curves is green in most segments, indicating that LivingRAG usually uses fewer completion tokens. The segment-level trend suggests that the savings are not caused by a single late-stage average and become more stable after the startup stage.

Table 4: Reuse opportunities and realized reuse in LivingRAG. “Opp.” denotes the template-reuse opportunity from Section 3.1. “Reuse” denotes realized scaffold reuse during inference. Graph transfer is measured from activation-map fusion. The opportunity and realized values use different measurements, so the table should be read as a trend comparison rather than a direct numerical comparison. All values are percentages.
<table><tr><td rowspan="2">Dataset</td><td>Graph signal</td><td colspan="2">Template signal</td></tr><tr><td>Graph trans.</td><td>Opp.</td><td>Reuse</td></tr><tr><td>2Wiki</td><td>0.00</td><td>96.30</td><td>84.15</td></tr><tr><td>HotpotQA</td><td>0.00</td><td>21.82</td><td>9.04</td></tr><tr><td>MuSiQue</td><td>91.20</td><td>39.14</td><td>40.69</td></tr><tr><td>MuSiQue-full</td><td>82.42</td><td>54.30</td><td>50.44</td></tr><tr><td>WixQA</td><td>91.50</td><td>16.04</td><td>7.03</td></tr></table>

## 4.3 Realized Reuse Signals

We next examine whether LivingRAG uses the reuse signals that motivate our design. This analysis is not an accuracy evaluation. It is also not a direct conversion from reuse opportunity to realized reuse. The opportunity values in Section 3.1 describe whether a stream contains possible reuse signals. The realized values here are measured from LivingRAG inference traces.

We measure two realized signals. Graphneighborhood transfer occurs when activation-map fusion introduces new above-threshold graph entities. Template reuse occurs when the selected scaffold has masked-template similarity of at least 0.35 with the current query.

Table 4 shows that LivingRAG uses different reuse paths on different datasets. On 2Wiki, template reuse is high. This matches the strong template signal observed in Section 3.1. Graph transfer is 0.00% under our strict definition. This does not mean that graph signals are absent. It only means that activation-map fusion does not introduce new above-threshold entities in this setting.

On MuSiQue, MuSiQue-full, and WixQA, graph transfer is much stronger. This shows that stored activation maps often change the graph retrieval state. Template reuse is also visible on MuSiQue and MuSiQue-full. These results support the two reuse paths in LivingRAG. Activation maps help retrieval reuse graph neighborhoods. Scaffolds help generation reuse reasoning patterns.

Table 5: Ablation results. 2W, MQ-F, and Wix report LLM accuracy (%) on 2WikiMultiHopQA, MuSiQuefull, and WixQA. Red. is average completion-token reduction against LinearRAG.
<table><tr><td>Variant</td><td>2W</td><td>MQ-F</td><td>Wix</td><td>Red.</td></tr><tr><td>LivingRAG</td><td>87.00</td><td>58.42</td><td>70.25</td><td>23.6%</td></tr><tr><td>w/o Act. fusion</td><td>86.70</td><td>56.32</td><td>67.45</td><td>15.1%</td></tr><tr><td>w/o Scaffold</td><td>83.80</td><td>56.92</td><td>69.55</td><td>8.3%</td></tr><tr><td>w/o Quality Gate</td><td>84.60</td><td>55.52</td><td>68.25</td><td>18.0%</td></tr></table>

HotpotQA has low values under both strict trace metrics. This suggests that these two measurements may not capture all experience effects on this dataset. Overall, the table shows that the realized behavior of LivingRAG is consistent with the reuse signals that motivated the method. Appendix K further analyzes performance by realized reuse signal and robustness across query orders.

## 4.4 Ablation Study

We ablate three key components on 2WikiMulti-HopQA, MuSiQue-full, and WixQA: activationmap fusion, reasoning scaffolds, and the write-back quality gate. Each run starts with an empty experience store and updates it online.

Table 5 shows that the two experience-use paths play different roles. Removing activation-map fusion hurts MuSiQue-full and WixQA more, where activation-map transfer is frequently realized during inference. Removing scaffolds hurts 2WikiMultiHopQA most and also reduces completion-token savings, suggesting that scaffolds help avoid repeated reasoning. Across the five streams, the quality gates store only 27.4% of candidate experiences (Appendix D). Removing the quality gate lowers accuracy on all datasets, confirming that writable experience must pass checks before reuse. Overall, the ablation supports the design of LivingRAG. Activation maps help retrieval, scaffolds help generation, and quality gates keep online experience reliable.

## 5 Conclusion

We presented LivingRAG, a Graph RAG framework with writable and reusable reasoning experience. LivingRAG stores verified experiences. It reuses them through activation-map fusion and reasoning scaffolds. Experiments on multi-hop QA benchmarks show that LivingRAG improves accuracy and reduces completion tokens over strong RAG baselines. The results suggest that verified experience reuse is a simple way to augment Graph RAG.

## 6 Limitations

One limitation of LivingRAG is long-term experience growth. As shown in our complexity analysis, LivingRAG keeps the LinearRAG graph unchanged and adds experience-store costs that are linear in the number of accepted experiences. The runtime breakdown also shows that this overhead is acceptable in our experiments, and that post-answer validation does not need to increase user-visible latency. However, in much larger deployments, experiences may accumulate over a long period. Too many repeated or weakly useful experiences may increase matching cost and make reuse less stable. Future systems may need stronger store maintenance, such as pruning, compression, or indexed experience search.

LivingRAG also assumes that stored experiences remain valid for later queries. The current store does not update or delete accepted experiences. The assumption can fail in dynamic domains. For example, product policies, event information, or other time-sensitive facts may change after an experience is written. Although each experience stores a timestamp, the current system does not explicitly reduce the influence of old experience. The fixed-corpus benchmarks contain no chronological updates or fact-validity intervals, so they cannot directly evaluate expiration under changing knowledge. Future work should explore time-aware refresh and adaptive pruning, so that older experiences can be downweighted, updated, or removed when new evidence appears.

Our evaluation also treats controlled QA benchmarks as proxies for online query streams. MuSiQue is constructed by composing singlehop questions and may therefore contain denser reusable structure than naturally occurring traffic. WixQA is more deployment-oriented, but it is not timestamped and its questions share an enterprise knowledge base. In natural streams, reusable experiences may be more scattered, while topic drift and noise may reduce their value. Evaluation on naturally timestamped streams with topic drift and controlled noise remains future work.

## Acknowledgments

This work was supported by the Zhejiang Provincial Natural Science Foundation of China (Youth Program) under Grant No. LQN26F020018 and by the “Pioneer” and “Leading Goose” R&D Program of Zhejiang under Grant No. 2024C01020.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations.

Yuzheng Cai, Zhenyue Guo, YiWen Pei, WanRui Bian, and Weiguo Zheng. 2025. SimGRAG: Leveraging similar subgraphs for knowledge graphs driven retrieval-augmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 3139–3158, Vienna, Austria. Association for Computational Linguistics.

Junnan Dong, Siyu An, Yifei Yu, Qian-Wen Zhang, Linhao Luo, Xiao Huang, Yunsheng Wu, Di Yin, and Xing Sun. 2025. Youtu-graphrag: Vertically unified agents for graph retrieval-augmented complex reasoning. Preprint, arXiv:2508.19855.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald. 2025. KiRAG: Knowledge-driven iterative retriever for enhancing retrieval-augmented generation. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 18969–18985, Vienna, Austria. Association for Computational Linguistics.

Guangze Gao, Zixuan Li, Chunfeng Yuan, Jiawei Li, Wu Jianzhuo, Yuehao Zhang, Xiaolong Jin, Bing Li, and Weiming Hu. 2025a. D-RAG: Differentiable retrieval-augmented generation for knowledge graph question answering. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 35398–35417, Suzhou, China. Association for Computational Linguistics.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. Preprint, arXiv:2312.10997.

Zengyi Gao, Yukun Cao, Hairu Wang, Ao Ke, Yuan Feng, S Kevin Zhou, and Xike Xie. 2025b. FRAG: A flexible modular framework for retrieval-augmented

generation based on knowledge graphs. In Findings of the Association for Computational Linguistics: ACL 2025, pages 6178–6192, Vienna, Austria. Association for Computational Linguistics.

Ziyi Guan, Jason Chun Lok Li, Zhijian Hou, Pingping Zhang, Donglai Xu, Yuzhi Zhao, Mengyang Wu, Jinpeng Chen, Thanh-Toan Nguyen, Pengfei Xian, Wenao Ma, Shengchao Qin, Graziano Chesi, and Ngai Wong. 2025. KG-RAG: Enhancing GUI agent decision-making via knowledge graph-driven retrieval-augmented generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5396–5405, Suzhou, China. Association for Computational Linguistics.

Kai Guo, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, and Jiliang Tang. 2025. Empowering GraphRAG with knowledge filtering and integration. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25439–25453, Suzhou, China. Association for Computational Linguistics.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao Huang. 2024. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2(3).

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025. From RAG to memory: Non-parametric continual learning for large language models. In Forty-second International Conference on Machine Learning.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. Hipporag: Neurobiologically inspired long-term memory for large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Mossad Helali, Yutai Luo, Tae Jun Ham, Jim Plotts, Ashwin Chaugule, Jichuan Chang, Parthasarathy Ranganathan, and Essam Mansour. 2025. Reliable and cost-effective exploratory data analysis via graphguided RAG. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 16536–16553, Suzhou, China. Association for Computational Linguistics.

Xinke Jiang, Ruizhe Zhang, Yongxin Xu, Rihong Qiu, Yue Fang, Zhiyuan Wang, Jinyi Tang, Hongxin Ding, Xu Chu, Junfeng Zhao, and Yasha Wang. 2025. HyKGE: A hypothesis knowledge graph enhanced RAG framework for accurate and reliable medical LLMs responses. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational

Linguistics (Volume 1: Long Papers), pages 11836– 11856, Vienna, Austria. Association for Computational Linguistics.

Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, Singapore. Association for Computational Linguistics.

Imran Kabir, Md Alimoor Reza, and Syed Billah. 2025. Logic-rag: Augmenting large multimodal models with visual-spatial knowledge for road scene understanding. In 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory OS of AI agent. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 25961–25970, Suzhou, China. Association for Computational Linguistics.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.

Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa Rangwala, and Christos Faloutsos. 2025. HybGRAG: Hybrid retrieval-augmented generation on textual and relational knowledge bases. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 879–893, Vienna, Austria. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation. In ICLR 2025 Workshop on Foundation Models in the Wild.

Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and Wentao Zhang. 2025. HopRAG: Multi-hop reasoning for logic-aware retrievalaugmented generation. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 1897–1913, Vienna, Austria. Association for Computational Linguistics.

Yayu Long, Kewei Chen, Long Jin, and Mingsheng Shang. 2025. DRAE: Dynamic retrieval-augmented expert networks for lifelong learning and task adaptation in robotics. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23098– 23141, Vienna, Austria. Association for Computational Linguistics.

Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, and Anh Tuan Luu. 2026a. HypergraphRAG: Retrieval-augmented generation via hypergraph-structured knowledge representation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

LINHAO LUO, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on graphs: Faithful and interpretable large language model reasoning. In The Twelfth International Conference on Learning Representations.

Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and Shirui Pan. 2026b. GFM-RAG: Graph foundation model for retrieval augmented generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. 2025. Think-on-graph 2.0: Deep and faithful large language model reasoning with knowledge-guided retrieval augmented generation. In The Thirteenth International Conference on Learning Representations.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Preprint, arXiv:2303.17651.

Costas Mavromatis, Soji Adeshina, Vassilis N. Ioannidis, Zhen Han, Qi Zhu, Ian Robinson, Bryan Thompson, Huzefa Rangwala, and George Karypis. 2025. BYOKG-RAG: Multi-strategy graph retrieval for knowledge graph question answering. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 27881–27898, Suzhou, China. Association for Computational Linguistics.

Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. 2025. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, WWW ’25, page 2366–2377, New York, NY, USA. Association for Computing Machinery.

Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, and Yassine Benajiba.

2025. MemInsight: Autonomous memory augmentation for LLM agents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 33136–33152, Suzhou, China. Association for Computational Linguistics.

Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Enting Chen, Damien Graux, Andre Melo, Ruofei Lai, Zeren Jiang, Zhongyang Li, Ye Qi, Yang Ren, Dandan Tu, and Jeff Z. Pan. 2025. GeAR: Graph-enhanced agent for retrieval-augmented generation. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 12049–12072, Vienna, Austria. Association for Computational Linguistics.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. 2024. Continual learning of large language models: A comprehensive survey. arXiv preprint arXiv:2404.16789.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.

Yiheng Shu, Saisri Padmaja Jonnalagedda, Xiang Gao, Bernal Jiménez Gutiérrez, Weijian Qi, Kamalika Das, Huan Sun, and Yu Su. 2026. REMem: Reasoning with episodic memory in language agent. In The Fourteenth International Conference on Learning Representations.

Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, and Wenjie Zhang. 2025. HydraRAG: Structured cross-source enhanced large language model reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 14431–14459, Suzhou, China. Association for Computational Linguistics.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving retrieval with chain-of-thought reasoning for knowledgeintensive multi-step questions. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10014–10037, Toronto, Canada. Association for Computational Linguistics.

Jingjin Wang and Jiawei Han. 2025. PropRAG: Guiding retrieval with beam search over proposition paths. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6212–6227, Suzhou, China. Association for Computational Linguistics.

Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma, and Vipin Chaudhary. 2025a. CausalRAG: Integrating causal graphs into retrieval-augmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 22680–22693, Vienna, Austria. Association for Computational Linguistics.

Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, and Han Li. 2025b. InfoGain-RAG: Boosting retrieval-augmented generation through document information gain-based reranking and filtering. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7190–7204, Suzhou, China. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and Jinsong Su. 2026. When to use graphs in RAG: A comprehensive analysis for graph retrieval-augmented generation. In The Fourteenth International Conference on Learning Representations.

Yilong Xu, Jinhua Gao, Xiaoming Yu, Yuanhai Xue, Baolong Bi, Huawei Shen, and Xueqi Cheng. 2025. Training a utility-based retriever through shared context attribution for retrieval-augmented language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 629–648, Suzhou, China. Association for Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Junchi Yu, Yujie Liu, Jindong Gu, Philip Torr, and Dongzhan Zhou. 2026. Can knowledge-graph-based retrieval augmented generation really retrieve what you need? In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong Yu. 2024. Chain-of-note: Enhancing robustness in retrieval-augmented language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 14672–14685, Miami, Florida, USA. Association for Computational Linguistics.

Haozhen Zhang, Tao Feng, and Jiaxuan You. 2025a. Graph of records: Boosting retrieval augmented generation for long-context summarization with graphs. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 23780–23799, Vienna, Austria. Association for Computational Linguistics.

Hengrui Zhang, Pin-Siang Huang, Zhen Zhang, Peican Lin, Yao-Ching Yu, Bo Hu, and Yulu Du. 2025b. PathwiseRAG: Multi-dimensional exploration and integration framework. In Proceedings

of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22904–22925, Suzhou, China. Association for Computational Linguistics.

Zhuoping Zhou, Davoud Ataee Tarzanagh, Sima Didari, Wenjun Hu, Baruch Gutow, Oxana Verkholyak, Masoud Faraki, Heng Hao, Hankyu Moon, and Seungjai Min. 2026. Query-aware flow diffusion for graphbased RAG with retrieval guarantees. In The Fourteenth International Conference on Learning Representations.

Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao Huang. 2025. Linearrag: Linear graph retrieval augmented generation on large-scale corpora. arXiv preprint arXiv:2510.10114.

## A Additional Related Work

Graph-based RAG. Graph-based RAG organizes external knowledge with explicit graph structures. GraphRAG builds an entity graph and community summaries to support global sensemaking over a corpus (Edge et al., 2024). LightRAG combines graph indexing with vector retrieval to retrieve entities, relations, and chunks efficiently (Guo et al., 2024). HippoRAG and HippoRAG2 use Personalized PageRank over an automatically constructed graph to support associative and multi-hop retrieval (Gutiérrez et al., 2024; Gutiérrez et al., 2025). LinearRAG builds a lightweight passage-sentence-entity graph and performs sparse graph propagation for efficient retrieval (Zhuang et al., 2025). GFM-RAG trains a graph foundation model to capture queryknowledge relations over graph structure (Luo et al., 2026b). GraphFlow learns a retrieval policy over text-rich knowledge graphs (Yu et al., 2026). These methods show that graph structure is useful for complex retrieval. LivingRAG builds on this line of work. It adds verified experience write-back to reuse prior reasoning in later graph retrieval and generation.

Reasoning-Enhanced RAG. Another line of work improves RAG with stronger reasoning. Iterative retrieval methods decompose a query and retrieve evidence step by step (Trivedi et al., 2023; Jiang et al., 2023). Self-RAG uses self-reflection to decide when and how to retrieve (Asai et al., 2024). Chain-of-Note generates intermediate notes over retrieved evidence (Yu et al., 2024). LogicRAG decomposes complex questions into subqueries according to logical dependencies (Kabir et al., 2025). These methods mainly improve reasoning for the current query. LivingRAG instead studies how useful reasoning from earlier queries can be stored and reused in later queries.

Memory-augmented LLMs and RAG. Memory-based language agents store past observations, feedback, or plans to improve later decisions (Shinn et al., 2023; Madaan et al., 2023). Recent systems also study memory for retrieval and reasoning. REMem builds a hybrid episodic memory graph from interaction histories and uses an agentic retriever for episodic recollection and reasoning (Shu et al., 2026). MemoRAG builds a global memory for long-context processing and uses it to generate retrieval clues (Qian et al., 2025). LivingRAG is complementary to these methods. It targets online Graph RAG over QA streams. Its stored unit is a verified reasoning experience. Each experience contains a sparse activation map, a compact reasoning summary, and validation metadata. The activation map guides later graph retrieval. The summary provides a reasoning scaffold for generation. A candidate is written only when it passes grounding and novelty checks.

## B Experimental Details

Entity extraction model. For experiments that use spaCy-based entity extraction, we use the English transformer pipeline en\_core\_web\_trf.

NLI grounding model. For the grounding check in experience write-back, we use the cross-encoder/nli-deberta-v3-large. The model is used as the NLI cross-encoder to score entailment between retrieved evidence and extracted claims.

Parameter setting. For fair comparison, Linear-RAG and LivingRAG use the same LinearRAG retrieval settings. LivingRAG keeps the original LinearRAG parameters unchanged and only adds the experience-store components described in Section 3.

## C Detailed Complexity Analysis

This appendix summarizes the extra cost introduced by LivingRAG over LinearRAG. Let $N _ { t } =$ $| \mathcal { E } _ { t } |$ be the number of stored experiences before query $q _ { t } .$ , d the embedding dimension, $b _ { t }$ the number of nonzero entities in the current base activation vector, and z¯ the average number of nonzero entries in a stored activation map. LivingRAG does not rebuild or modify the LinearRAG corpus graph. It only changes the initial entity activation vector and then calls the same graph retrieval procedure. Thus, the corpus-side construction and retrieval complexity remains the LinearRAG complexity.

The main added online cost is the exact scan over the experience store. For each stored experience, LivingRAG computes one dense query-embedding similarity and one sparse activation-map similarity:

$$
O \big ( N _ { t } ( d + b _ { t } + \bar { z } ) \big ) \approx O \big ( N _ { t } ( d + \bar { z } ) \big ) .\tag{22}
$$

After matching, only the top-K experiences are fused into the seed activation vector. If each selected activation map is optionally truncated to its top H entities, the fusion cost is

$$
{ \cal O } \big ( K \operatorname* { m i n } ( \bar { z } , H ) \big ) ,\tag{23}
$$

where K and H are fixed hyperparameters. Template-aware scaffold selection, when enabled, adds at most one additional store scan. Template embeddings may be computed lazily from stored scaffold or source question text and cached transiently, but they are not part of persistent experience storage.

Write-back validation is also store-side and selective. Novelty Score (NS) uses the same dualchannel scan, $O ( N _ { t } ( d + \bar { z } ) )$ . Grounding Score (GS) is computed only after NS passes. If $c _ { t }$ claims are extracted and the evidence pre-filter keeps $K _ { g }$ candidate evidence sentences per claim, the NLI stage costs

$$
O ( c _ { t } K _ { g } ) ,\tag{24}
$$

where $K _ { g }$ is fixed and the evidence comes only from the small set of retrieved passages. The combined per-query cost can therefore be written as

$$
\begin{array} { r l } & { T _ { \mathrm { L i v i n g } } ( q _ { t } ) = T _ { \mathrm { L i n e a r } } ( q _ { t } ) } \\ & { ~ + O \bigl ( N _ { t } ( d + \bar { z } ) \bigr ) } \\ & { ~ + O \bigl ( K \operatorname* { m i n } ( \bar { z } , H ) \bigr ) } \\ & { ~ + { \mathcal { H } _ { \mathrm { G S } } O } ( c _ { t } K _ { g } ) . } \end{array}\tag{25}
$$

The expression separates the unchanged corpusside LinearRAG cost from the added experiencestore cost.

Over a stream of Q queries, exact store scanning contributes $\textstyle \sum _ { t = 1 } ^ { Q } N _ { t }$ . The worst case is quadratic in Q if every query is stored, but LivingRAG does not store every candidate. Across our five streams, only 1806 of 5817 candidates pass NS and reach GS. The other 4011 candidates skip the NLI check.

Finally, 1592 candidates are written to the store, giving a final write-back ratio of 27.4%. Thus, store growth is tied to verified and novel experiences rather than to the raw number of queries.

The additional storage is similarly compact. Each stored experience contains one query embedding, a sparse activation map, a short scaffold, and metadata. Let <sup>¯</sup>ℓ denote the average scaffold length. The experience-store overhead is

$$
S _ { \mathrm { L i v i n g } } ( t ) = S _ { \mathrm { L i n e a r } } + O \bigl ( N _ { t } ( d + \bar { z } + \bar { \ell } ) \bigr ) .\tag{26}
$$

Template embeddings, when used for scaffold selection, are transient cached values and are not counted in this persistent storage term. Crucially, LivingRAG stores sparse activation maps rather than dense graph-entity vectors, and stores compact scaffolds rather than full reasoning traces. The extra space is therefore linear in the number of accepted experiences and remains separate from the static corpus graph. Overall, LivingRAG preserves LinearRAG’s corpus-side scalability while adding a controlled experience-store overhead that is bounded by sparsification, top-K fusion, and selective write-back. Appendix G reports the measured runtime of these components. Appendix D explains why grounding checks are run only for a subset of candidates.

## D Quality Gates for Experience Write-Back

Figure 3 and Table 6 analyze the two gates used before a candidate is written as a verified experience. For implementation efficiency, LivingRAG evaluates NS before GS. The change affects only the order of computation. A new experience is stored only when both gates pass. When NS fails, the system skips GS and avoids the NLI check.

The NS curve is high at the beginning because there are few stored experiences. As more experiences are written, the NS pass rate decreases. The write-back curve follows the same trend. Across the five streams, only 1806 of 5817 candidates pass NS. The number of GS checks is also 1806, which confirms that GS is skipped when NS fails. The final number of stored experiences is 1592. The count shows that LivingRAG does not store every query. It mainly stores candidates that remain novel relative to earlier experiences. GS is then applied to the smaller candidate set. In total, 4011 candidates fail NS and skip GS. Among the 1806 candidates checked by GS, 1592 pass and 214 are rejected. The GS pass rate among checked candidates is 88.15%. The results show that NS controls store growth. GS removes ungrounded candidates before write-back.

WixQA is a special case in Table 6. Its GS pass rate is 100% among the candidates that reach the GS check. The value does not mean that all GS scores are saturated. The average GS on the checked WixQA candidates is 0.886, and the lowest score is 0.653. Thus, the result means that all checked candidates are above the grounding threshold of 0.6. The likely reason is that the WixQA corpus consists of official help documents. Many answers contain procedural information that is directly supported by the retrieved help documents. For this dataset, we therefore use GS as a light grounding constraint rather than as the main selective gate. In parameter tuning, this corresponds to keeping the grounding threshold permissive and relying on the novelty gate and WixQA-specific retrieval settings to control write-back. The effective selectivity is shifted toward NS, which filters repeated support patterns before GS is run. The setting prevents WixQA from being dominated by grounding rejection, while still requiring each stored experience to have explicit passage support.

Table 7 gives concrete examples of these decisions. The two stored cases show when a candidate is written to the store. It must be new enough and supported by retrieved passages. The WixQA grounding case also explains why this dataset can have high GS pass rates. Its answer is stated directly in the help articles. The NS rejection case shows the opposite behavior. A later domainrefund query retrieves an almost identical stored experience. NS assigns a very low score and prevents another copy from being written. Finally, the GS rejection case shows why grounding is still needed after NS. The model predicts the correct answer, but the retrieved passages do not contain the key entities needed to support it. LivingRAG therefore returns the answer but refuses to store the candidate. The grounding gate prevents unsupported reasoning from becoming reusable experience. The selectivity also explains the validation term in the complexity analysis in Appendix C.

## E Dataset Details

We evaluate LivingRAG on five online QA streams. Three of them are standard multi-hop QA benchmarks: 2WikiMultiHopQA, HotpotQA, and MuSiQue. They mainly test whether a system can retrieve and combine evidence across multiple facts. Most answers in these datasets are short entities, dates, locations, or yes/no labels.

![](images/22facf304bc416fd3abdb327e85913dcc4dbd40d482c4036141026b75f7d5242.jpg)  
Figure 3: Dynamics of the NS and GS gates during online inference. NS pass and experience write-back are computed over each window. GS pass is computed only over candidates that reach the GS check. The black curve shows the cumulative number of stored experiences. The first 10% of each stream is expanded to show the startup stage.

Table 6: Effect of the NS and GS gates on experience write-back. All runs use $\tau _ { \mathrm { n o v e l } } = 0 . 7$ and $\tau _ { \mathrm { g s } } = 0 . 6$ . NS pass is measured over all questions. GS pass is measured only on candidates that reached the GS check. GS reject counts the checked candidates that failed the grounding gate.
<table><tr><td>Dataset</td><td>Questions</td><td>NS pass (%)</td><td>GS checked</td><td>GS pass (%)</td><td>GS reject</td><td>t Stored experiences</td></tr><tr><td>2Wiki</td><td>1000</td><td>13.80</td><td>138</td><td>72.46</td><td>38</td><td>100</td></tr><tr><td>HotpotQA</td><td>1000</td><td>63.30</td><td>633</td><td>86.73</td><td>84</td><td>549</td></tr><tr><td>MuŠiQue</td><td>1000</td><td>36.30</td><td>363</td><td>86.23</td><td>50</td><td>313</td></tr><tr><td>MuSiQue-full</td><td>2417</td><td>21.31</td><td>515</td><td>91.84</td><td>42</td><td>473</td></tr><tr><td>WixQA</td><td>400</td><td>39.25</td><td>157</td><td>100.00</td><td>0</td><td>157</td></tr><tr><td>All</td><td>5817</td><td>31.05</td><td>1806</td><td>88.15</td><td>214</td><td>1592</td></tr></table>

We further include MuSiQue-full and WixQA. These two datasets test different aspects of online experience reuse. MuSiQue-full provides a larger multi-hop stream with denser reuse structure. WixQA provides a customer-support stream over a shared enterprise knowledge base. Together, they make the evaluation less dependent on a single dataset style.

MuSiQue-full. MuSiQue is built by composing single-hop questions into multi-hop questions. It provides decomposition annotations that expose which single-hop questions are reused across different multi-hop questions. This makes it useful for studying whether an experience-reusing RAG system can reuse prior reasoning steps.

Prior RAG evaluations often use a 1,000- question MuSiQue subset. We additionally evaluate on a larger answerable split, which we call MuSiQue-full. It contains 2,417 answerable questions. This name is used only to distinguish it from the smaller subset in our experiments. It does not refer to the official unanswerable MuSiQue-Full setting.

We define an experience cluster using the official MuSiQue decomposition. Two questions are linked if they share at least one decomposed single-hop question. We then compute connected components over this question graph. A component with at least two questions is treated as an experience cluster. This definition is based on the dataset structure. It does not use text similarity, embedding similarity, or model predictions.

Table 8 shows that MuSiQue-full contains a stronger reuse structure than the smaller MuSiQue subset. It has more clusters, higher cluster coverage, and larger average cluster size. This makes it a useful stress test for online experience reuse. A system that stores verified experiences should benefit more when later questions reuse earlier reasoning components.

WixQA. WixQA is different from the Wikipediastyle benchmarks. It is built from product-support questions over the Wix Help Center. The questions are about product features, account settings, payments, domains, site editing, email, bookings, and related user tasks. The answers are usually support-style solutions. They often describe what the user should do, which product should be used, or whether a feature is supported.

Table 7: Representative NS and GS case studies. The cases are selected from the final LivingRAG runs without re-running retrieval or validation.
<table><tr><td>Case</td><td>Dataset</td><td>Idx. Query</td><td></td><td>Answer</td><td>NS</td><td>GS Decision</td><td>Interpretation</td></tr><tr><td>Stored multi-hop candidate</td><td>HotpotQA</td><td></td><td>33 What subsidiary of the largest airline of the Republic of China (Taiwan) has a main</td><td>Mandarin Airlines</td><td>0.793</td><td>1.000 Stored</td><td>Both gates pass. The retrieved airline passage directly supports the answer, so the candidate becomes a</td></tr><tr><td>Direct WixQA WixQA grounding</td><td></td><td></td><td>hub at Taichung Airport? 57 I would like to know how many sites I can have.</td><td>There is no limit to the number of sites you can create under a single Wix account;</td><td></td><td>0.829 1.000 Stored</td><td>verified experience. The answer contains directly supported procedural information from the retrieved help articles. This illustrates why WixQA GS scores are often high under the</td></tr><tr><td>Novelty rejection</td><td>WixQA</td><td></td><td>391 can I get a refund on the domain</td><td>as you want. No. Wix domain purchases are non-refundable and do not include a 14-day money-back guarantee or trial</td><td>0.213</td><td>– Rejected by NS The query repeats a stored</td><td>permissive grounding threshold. domain-refund experience. NS assigns a low score, skips GS, and avoids storing another copy of the</td></tr><tr><td>Grounding rejection</td><td></td><td></td><td>HotpotQA 858 Spawn, a fictional character who Mustafa Shakir has expressed interest in playing, was created by?</td><td>longer... Todd McFarlane</td><td>0.701 0.368 Rejected by GS NS passes, but the retrieved</td><td></td><td>passages do not mention the key entities. The answer matches the gold label, yet GS blocks write-back because the candidate is not</td></tr></table>

Table 8: Reuse structure in the MuSiQue streams. Experience clusters are connected components of questions that share at least one decomposed single-hop question.
<table><tr><td>Dataset</td><td></td><td></td><td></td><td></td><td>#Q #Chunks #Clusters Coverage Avg. Size Max Size</td><td></td></tr><tr><td>MuSiQue</td><td>1,000</td><td>1,354</td><td>178</td><td>79.2</td><td>4.4</td><td>21</td></tr><tr><td>MuSiQue-full</td><td>2,417</td><td>2,379</td><td>300</td><td>89.9</td><td>7.2</td><td>58</td></tr></table>

This setting is important for LivingRAG. All questions are grounded in the same enterprise knowledge base. Many user issues refer to related products, settings, and help articles. This creates repeated evidence neighborhoods and recurring support patterns. At the same time, the language is less formulaic than standard multi-hop QA. This makes WixQA a useful test for experience reuse in a more deployment-like setting.

We use 400 WixQA questions in the main evaluation. The sample is balanced between ExpertWritten and Simulated questions. We exclude the Synthetic subset because it is generated and more tightly tied to individual articles. The retrieval corpus is the Wix Help Center corpus. Following the LinearRAG setting, articles are segmented into chunks of roughly 600 tokens.

## F WixQA-specific Processing and Evaluation

WixQA requires several dataset-specific choices. These choices are not changes to the LivingRAG framework. They are needed because WixQA differs from the other datasets in entity type, passage length distribution, and answer format. In the experiments that compare LinearRAG and LivingRAG on WixQA, both systems use the same WixQAspecific processing.

## F.1 LLM-based Entity Extraction

The graph retriever relies on query entities to initialize graph activation. For Wikipedia-style QA, a general-purpose NER model is usually sufficient. Most questions contain people, locations, organizations, works, dates, or named events. These entities are also the main anchors of the reasoning chain. For example, in 2WikiMultiHopQA and HotpotQA, spaCy often extracts useful entities such as person names, film titles, places, and organizations.

WixQA has a different language style. Its important query cues are often product names, UI labels, plan names, feature names, settings, and user actions. These cues are not ordinary named entities.

They are domain terms from the Wix Help Center. If they are missed, graph retrieval starts from weak or empty query seeds.

Table 9 compares representative cases. For standard multi-hop QA, spaCy usually extracts useful named entities. For WixQA, spaCy often returns no entity or only a broad brand-level entity. LLM NER instead extracts product-domain terms that are more useful for support retrieval.

These examples show that the issue is not only low recall. It is also a mismatch between general NER labels and product-support terminology. For example, the last WixQA question cannot be answered by recognizing only the word Wix. The retrieval system must distinguish whether the user should use Wix Stores or Wix Bookings. This distinction is central to the answer.

We therefore use LLM-based entity extraction for WixQA. The extractor is implemented with Qwen3.5-9B. This is a relatively small LLM, but we find it sufficient for this task in practice. The task only requires extracting short productdomain seed terms, not solving the question. Thus, a smaller LLM can provide useful domain entities without changing the retrieval framework. For fairness, LinearRAG and LivingRAG use the same passage-level and query-level extractor in all WixQA runs.

The goal is not to add reasoning to the retriever. The goal is to produce better product-domain seed terms. The downstream retrieval pipeline remains unchanged. Extracted query terms are mapped to graph entities and then used by the same graph propagation and passage ranking steps.

Figure 4 shows the passage-level prompt used during indexing. Figure 5 shows the question-level prompt used for query-side entity extraction.

Table 10 shows that the examples in Table 9 are not isolated cases. On 2WikiMultiHopQA, spaCy almost always extracts at least one query entity. On WixQA, it extracts no entity for 52.5% of the questions. This weakens graph retrieval because many queries have no useful starting nodes. LLM NER removes this failure case.

LLM NER is not noise-free. Some extracted terms can be mapped to imperfect graph entities. We therefore only keep the top mapped candidates for each query term and remove high-degree hub entities. This keeps query activation focused while still giving the retriever product-specific seeds.

We also apply the LLM extractor to WixQA passages during indexing. This keeps the entity space consistent between queries and passages. The resulting graph remains sparse enough for propagation. Almost every chunk has extracted entities, and the average number of entities per entity-bearing sentence is below two.

## F.2 Passage Length Normalization

WixQA also differs in passage length distribution. The Wikipedia-style datasets are mostly segmented into near-fixed long chunks. In contrast, WixQA keeps the natural granularity of help-center articles. Some chunks are short feature notes or knownissue pages. Others are long tutorials with detailed steps.

This creates a different retrieval problem. Long chunks naturally contain more words and more entity mentions. They can therefore receive higher retrieval scores because they cover more topics. Short chunks may be more precise, but they provide fewer matching signals. Without correction, a long generic article may dominate a short article that directly answers the user issue.

Table 12 shows this difference. In WixQA, 44.4% of chunks contain fewer than 1,000 characters, and 59.6% contain fewer than 2,000 characters. The corresponding ratios are almost zero in the other datasets. This means that WixQA retrieval must handle a mix of very short and long passages.

We apply a weak BM25-style length normalization to WixQA passage scores:

$$
\mathrm { s c o r e } ^ { \prime } ( p ) = { \frac { \mathrm { s c o r e } ( p ) } { ( 1 - b ) + b \cdot | p | / \overline { { | p | } } } } .\tag{27}
$$

Here |p| is the passage length, and $\overline { { | p | } }$ is the average passage length in the corpus. We set $b ~ = ~ 0 . 1$ This value is intentionally small. It mildly corrects length bias, but it does not strongly favor short passages. This is important because some long Wix articles contain necessary step-by-step evidence.

## F.3 Evaluation Metric for WixQA

We do not use contain-match accuracy as the main metric for WixQA. Contain-match accuracy is suitable for short-answer QA. In 2WikiMultiHopQA, HotpotQA, and MuSiQue, the gold answer is usually a short entity, date, place, or yes/no label. A correct generated answer often contains the exact gold answer span.

WixQA is different. Its gold answers are supportstyle solutions. They often include conditions, product-specific guidance, operation steps, caveats, and official wording. A generated answer may solve the user’s problem without copying the full gold answer. Thus, requiring the full gold answer to appear as a substring is too strict.

Table 9: Examples of query-side entity extraction. spaCy works well for standard multi-hop QA because the key retrieval cues are ordinary named entities. On WixQA, the key cues are product-domain terms, so spaCy often fails or extracts only broad entities.
<table><tr><td>Dataset</td><td>Question</td><td>spaCy NER</td><td>LLM NER or useful seeds</td></tr><tr><td>2Wiki</td><td>When did Lothair II&#x27;s mother die?</td><td>Lothair II</td><td>Not needed</td></tr><tr><td>2Wiki</td><td>Which film was released first, Aas Ka Aas Ka Panchhi or Phoolwari?</td><td>Phoolwari</td><td>Panchhi; Not needed</td></tr><tr><td>HotpotQA</td><td>In which county is the town in which Ray- Raymond Robertsen mond Robertsen was born?</td><td></td><td>Not needed</td></tr><tr><td>MuSiQue</td><td>When was the person who Messi&#x27;s goals in Messi; Copa del Rey; Not needed Copa del Rey compared to get signed by Barcelona Barcelona?</td><td></td><td></td></tr><tr><td>WixQA</td><td>I&#x27;m want to know how much it would cost to [] upgrade my email plan.</td><td></td><td>Email Marketing Plan; upgrade Email Marketing Plan; Email Marketing Plan pricing</td></tr><tr><td>WixQA</td><td>How do I sync the hotel app with my calen- [] dars ical link to allow visitors to book avail- able dates on my site?</td><td></td><td>Wix Bookings; iCal URL; sync- ing calendars; booking avail- able dates</td></tr><tr><td>WixQA</td><td>I have not received my payment in my bank [] account yet.</td><td></td><td>Wix Payments; payout delay; failed payout; Payouts on Hold</td></tr><tr><td>WixQA</td><td>I want to know if the Wix store function work Wix for selling services instead of just physical goods.</td><td></td><td>Wix Stores; selling services; ser- vice bookings; Wix Bookings</td></tr></table>

Table 10: Query-side entity extraction statistics. LLM NER addresses the low-recall problem of standard NER on WixQA while keeping the number of query seeds small.
<table><tr><td>Method</td><td>Query Set</td><td>0 Ent.</td><td>Avg.</td><td>Med.</td><td>Range</td></tr><tr><td>spaCy NER</td><td>2Wiki, first 400</td><td>1.2</td><td>1.68</td><td>2</td><td>0-4</td></tr><tr><td>spaCy NER</td><td>WixQA, all 400</td><td>52.5</td><td>0.61</td><td>0</td><td>0-4</td></tr><tr><td>LLM NER + mapping</td><td>WixQA, all 400</td><td>0.0</td><td>3.77</td><td>4</td><td>1-5</td></tr></table>

Table 11: Index-side entity extraction statistics for WixQA with LLM NER.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Chunks</td><td>7,132</td></tr><tr><td>Chunks with no entity</td><td>1</td></tr><tr><td>Avg. entities per chunk</td><td>11.23</td></tr><tr><td>Median entities per chunk</td><td>10</td></tr><tr><td>Sentences with entities</td><td>68,798</td></tr><tr><td>Avg. entities per entity-bearing sentence Median entities per entity-bearing sentence</td><td>1.79 1</td></tr></table>

Table 13 gives representative examples. The standard multi-hop QA answers are short spans. In contrast, WixQA answers describe how to solve a user-support problem. The answer may be 60, 120, or more than 300 words.

Table 12: Passage length distribution by character count. WixQA contains many short chunks, while the Wikipedia-style datasets are close to fixed-length chunks.
<table><tr><td>Dataset</td><td>#Chk.</td><td>Median</td><td>&lt; 1000</td><td>&lt; 2000</td></tr><tr><td>WixQA</td><td>7,132</td><td>1,321</td><td>44.4</td><td>59.6</td></tr><tr><td>2Wiki</td><td>658</td><td>4,449</td><td>0.0</td><td>0.0</td></tr><tr><td>HotpotQA</td><td>1,311</td><td>4,651</td><td>0.0</td><td>0.0</td></tr><tr><td>MuSiQue</td><td>1,354</td><td>4,701</td><td>0.0</td><td>0.0</td></tr></table>

Table 14 shows the scale of this mismatch. The median answer length in WixQA is 59 words. For the other datasets, the median answer length is only 2 words. For WixQA, 85.0% of answers are longer than 30 words. For the other datasets, this ratio is zero.

The following example illustrates why containmatch is misleading. A generated answer can be useful and correct even if it does not contain the full official support answer.

![](images/6c99f3604d73984e9ef3a1e21dff04262e0879a7a8777b0a4d3a8d64de4f498e.jpg)  
Figure 4: Passage-level prompt used for WixQA LLM-based entity extraction during indexing. The extractor identifies Wix product-documentation entities and article-level action-topic terms from each passage.

![](images/7d56e38228e720634423e830a9500e0dcef36ec596a38cbc55efbfb60b97ab88.jpg)

This is why contain-match accuracy severely underestimates answer quality on WixQA. In our 400-question WixQA evaluation, LinearRAG has 0 contain-match hits. LivingRAG has only 1 containmatch hit. However, LLM-based evaluation gives 66.00% accuracy for LinearRAG and 70.25% for LivingRAG.

We use LLM accuracy for all datasets, but the evaluation prompt differs by dataset type. For standard multi-hop QA, the evaluator checks whether the generated answer contains the key gold information and does not contradict it. For WixQA, the evaluator checks whether the answer correctly resolves the support issue. This requires a taskoriented prompt. The answer must recommend the right Wix product or feature. It must describe the core operation correctly. It must not claim that a supported feature is unsupported, or the reverse. It does not need to reproduce the full official wording.

This difference is important in product-support QA. For example, if a user asks whether Wix Stores can be used to sell services, the key issue is not whether the answer mentions the words Wix Stores. The key issue is whether it recommends the right product. An answer that tells the user to use Wix Stores for service booking would be misleading. A correct answer should identify Wix Bookings as the suitable product for booking and paying for services.

## G Runtime Breakdown

We further measure the runtime of the main online components. We use the same concurrent setting as the main experiments. The system answers up to 16 questions in parallel. We exclude onetime indexing, model loading, and evaluation-only LLM calls. The reported values are per-question averages. They are computed by summing the measured time of each component and dividing it by the number of questions. Because the system runs concurrently, these values are module-level averages. They should not be read as serialized wall-clock latency. The corresponding asymptotic cost is analyzed in Appendix C. Gate selectivity is analyzed in Appendix D.

The implementation keeps the LinearRAG retrieval path unchanged. For each query, entity matching and graph retrieval are run on CPU in the same way as LinearRAG. LLM calls are issued concurrently. For LivingRAG, answer generation and experience write-back are separated. After an answer is generated, the system forms a candidate experience. The candidate is then checked by the novelty and grounding gates described in Section 3.5. Grounding uses batched NLI crossencoder calls. Accepted candidates are written to the store in order. The asynchronous design keeps online answering separate from experience-store maintenance.

![](images/39a53ca497df351d4e37e9ea8e3c3f430c9ea1ea423a79632974db6e54ecd955.jpg)  
Figure 5: Question-level prompt used for WixQA LLM-based entity extraction. The extractor maps user questions to canonical Wix product names and action-topic phrases for query-side graph activation.

![](images/22716cd4ec7be9f7455e34be0cca70aa3b80d0411bfbf6857c7f012e1cc7b0f8.jpg)  
Figure 6: LLM-based accuracy evaluation prompt used for standard multi-hop QA datasets.

Table 15 shows that retrieval is much cheaper than LLM inference. Across the measured datasets, retrieval takes about 1.7% to 5.1% of the LLM time. The retrieval time of LivingRAG is close to that of LinearRAG. The result supports the complexity analysis. The added experience-augmented retrieval step is not the main runtime bottleneck.

Write-back validation is more expensive than retrieval. The cost comes from local NLI validation and experience write-back. All NLI checks are run on a single NVIDIA GeForce RTX 3060 GPU. The RTX 3060 is a modest GPU for cross-encoder validation. More importantly, write-back validation is a post-answer maintenance step. The answer has already been generated before this step starts. In a serving system, the answer can be returned before this experience update finishes. Thus, the validation column describes the cost of maintaining a reliable experience store. It does not need to increase user-visible answer latency.

## H Prompt Token Trends

Figure 8 complements the completion token analysis in Figure 2. It shows the input cost of Linear-

Table 13: Gold answer format differs sharply between standard multi-hop QA and WixQA. Standard QA usually expects a short span. WixQA expects a support solution.
<table><tr><td>Dataset</td><td>Question</td><td>Gold answer</td><td>Words</td></tr><tr><td>2Wiki</td><td>When did Lothair II&#x27;s mother die?</td><td>20 March 851</td><td>3</td></tr><tr><td>2Wiki</td><td>Which film was released first, Aas Ka Panchhi or Phoolwari?</td><td>Phoolwari</td><td>1</td></tr><tr><td>HotpotQA</td><td>In which county is the town in which Raymond Robertsen was born?</td><td>Finnmark county</td><td>2</td></tr><tr><td>MuSiQue</td><td>When was the person who Messi&#x27;s goals in Copa del Rey compared to get signed by Barcelona?</td><td>June 1982</td><td>2</td></tr><tr><td>WixQA</td><td>Can I start accepting payments on my site while my Wix You can start accepting payments Payments account is still under verification?</td><td>almost immediately, but Wix must verify your identity before the account is fully activated.</td><td>27</td></tr><tr><td>WixQA</td><td>I want to know if the Wix store function work for selling The answer distinguishes Wix services instead of just physical goods.</td><td>Stores from Wix Bookings. For selling services, it recommends Wix Bookings because it supports service booking and online payment.</td><td>66</td></tr><tr><td>WixQA</td><td>I am inquiring about purchasing the yearly premium plan with a free domain for 1 year. The voucher does not voucher is not shown at checkout. show up at checkout. Do I need to purchase the plan first?</td><td>The answer explains that the It becomes available after purchasing a yearly Premium plan and can be claimed from the</td><td>120</td></tr><tr><td>WixQA</td><td>How can I add discounts to my service prices when customers pay for a plan?</td><td>Premium Vouchers page. The answer explains how to create a Pricing Plans coupon, including discount type, applicable plans, billing-cycle behavior, dates. usage limits, and coupon creation.</td><td>328</td></tr></table>

Table 14: Answer length statistics. WixQA answers are long support solutions rather than short answer spans.
<table><tr><td>Dataset</td><td>Median words</td><td> ${ \le } 5$  words</td><td> ${ \bf \ } > 3 0$  words</td></tr><tr><td>2WikiMultiHopQA</td><td>2</td><td>95.5</td><td>0.0</td></tr><tr><td>HotpotQA</td><td>2</td><td>96.1</td><td>0.0</td></tr><tr><td>MuSiQue</td><td>2</td><td>91.2</td><td>0.0</td></tr><tr><td>WixQA</td><td>59</td><td>0.2</td><td>85.0</td></tr></table>

RAG and LivingRAG over the online stream. At the very beginning, LivingRAG has not yet stored useful experiences. It also has no reasoning scaffold to add to the prompt. Thus, its prompt is the same as the LinearRAG prompt for the earliest queries. After experiences are written, LivingRAG can select a reasoning scaffold and place it in the input context. The selected scaffold makes the LivingRAG prompt longer than the LinearRAG prompt in later segments. The trend is therefore consistent with Table 3. LivingRAG pays a small input token increase, but this increase is much smaller than the reduction in completion tokens. Appendix I further shows that scaffold use increases after the startup stage.

## I Experience Use During Online Inference

Figure 9 and Table 16 describe how LivingRAG uses experiences after they are written. The figure shows that the use of stored experiences rises quickly after the startup stage. Activation-map retrieval and reasoning scaffolds are used for most later queries in all datasets. At the same time, experience write-back is more selective. The pattern matches the design of LivingRAG. The system can reuse earlier experiences often, but it only writes a new experience when the candidate is grounded and sufficiently novel. The table further shows that reuse is spread across many stored experiences. For activation-map retrieval, the coverage is above

![](images/dd125130c8b11a91b76e3537cf7bd540744fd0237b03180123756da565e668b9.jpg)  
Figure 7: LLM-based accuracy evaluation prompt used for WixQA. The prompt evaluates whether a generated support answer resolves the user issue without recommending the wrong Wix product, operation, or unsupported behavior.

Table 15: LLM, retrieval, and validation times are module-level averages. Indexing, model loading, and evaluationonly LLM calls are excluded. The Retrieval / LLM column reports the ratio between retrieval time and LLM time. Validation refers to the post-answer novelty and grounding checks used for experience write-back.
<table><tr><td>Dataset</td><td>Method</td><td>LLM s/q</td><td>Retrieval s/q</td><td>Retrieval / LLM</td><td>Validation s/q</td></tr><tr><td>2Wiki</td><td>LinearRAG</td><td>30.07</td><td>0.50</td><td>1.7%</td><td>N/A</td></tr><tr><td>2Wiki</td><td>LivingRAG</td><td>24.04</td><td>0.54</td><td>2.2%</td><td>2.13</td></tr><tr><td>HotpotQA</td><td>LinearRAG</td><td>15.42</td><td>0.72</td><td>4.7%</td><td>N/A</td></tr><tr><td>HotpotQA</td><td>LivingRAG</td><td>14.45</td><td>0.74</td><td>5.1%</td><td>2.35</td></tr><tr><td>MuŠiQue</td><td>LinearRAG</td><td>37.23</td><td>1.49</td><td>4.0%</td><td>N/A</td></tr><tr><td>MuSiQue</td><td>LivingRAG</td><td>33.88</td><td>1.50</td><td>4.4%</td><td>6.64</td></tr></table>

79% on every dataset. For reasoning-scaffold reuse, the coverage is also broad, ranging from 58.11% on HotpotQA to 85.00% on 2Wiki. These results indicate that the stored experiences are not only accumulated. They are repeatedly selected by the retrieval and generation paths during later inference. Appendix J gives concrete examples of these reuse patterns.

## J Direct Case Study: LinearRAG vs. LivingRAG

We next give aligned case studies from the final runs. Each case compares the same stream index under LinearRAG and LivingRAG. The first two cases show how activation-map fusion changes retrieval toward the answer-bearing evidence. The last two cases show how a selected scaffold gives the LLM a reusable reasoning form and sharply shortens generation.

![](images/9ad62c24d1570814d7377e48903873df423ac7b33806b4b1ce60773d982bef2f.jpg)

![](images/da177c63f3ac80caa7593f0b203a53827acfb00f1c61d76f6e30511f3fdf020c.jpg)

Figure 8: Prompt token trends over online QA streams. Each panel corresponds to one dataset. The blue curve shows LinearRAG and the orange curve shows LivingRAG. The first 10% of each stream is expanded to show the startup stage.  
![](images/eb0b6947c3b9fceb413907b4d074e1a3af9f55cd0bf13ff809a65cd5408eabd2.jpg)  
Figure 9: Online use of stored experiences in LivingRAG. The curves show activation-map retrieval, reasoning scaffold use, experience write-back, and the cumulative number of stored experiences. The first 10% of each stream is expanded to show the startup stage.

VII explored the coast of North America. Sebastian Cabot (explorer): Sebastian Cabot was the son of John Cabot.

Summary of the reasoning chain: The current question asks for the child of a navigator. Ignacio Esparza gives Guadalajara, and the Guadalajara evidence links the location to North America. The retrieved Cabot evidence identifies the relevant navigator as John Cabot, not Francisco de Orellana. Since the Sebastian Cabot article states that Sebastian Cabot was John Cabot’s son, the answer is Sebastian Cabot.

In E1, the useful stored experience is not only a previous answer. It injects a coherent Cabotcentered entity neighborhood, which moves retrieval from a generic explorer branch to the evidence chain that contains the gold answer.

## E2: Experience transfers the Central Europe–406 event

Question: At the end of which year did tribes from the area where Linear Pottery culture was found invade the Roman Empire?

LinearRAG output: 113 BC. Result: incorrect. Completion tokens: 4,360.

LivingRAG output: 406. Result: correct. Completion tokens: 847. Experience retrieved by LivingRAG: Source question: At the end of which year did tribes from the place where the Celts are said to have migrated from invade the Roman empire? Stored answer: 406. Activation signal: 11 experience entities are added, and 10 new entities cross the retrieval threshold. Important activated entities include Suebi, Vandals,

Alani, Central Europe, and 31 December 406.

Retrieved articles and answer evidence: Linear Pottery (LBK) context: the relevant Neolithic evidence places the cultural area in Central Europe. Galicia (Spain): in the early fifth century, tribes of Central Europe, including the Suebi, Vandals, and Alani, crossed the Rhine on 31 December 406.

Summary of the reasoning chain: The question first resolves the Linear Pottery area to Central Europe. The reused experience points to the same historical event pattern: Central European tribes entering Roman territory. The Galicia passage gives the date explicitly as 31 December 406. Therefore the requested year is 406.

E2 shows the same mechanism in a different domain. The earlier experience supplies eventspecific entities, so LivingRAG retrieves the passage that directly states the answer year instead of following a distracting ancient-date association.

## S1: Scaffold reuse for the Moon River–Ondine template

Dataset / index: MuSiQue-full / 627

Question: For what did the singer of Moon River in Breakfast at Tiffany’s win a Tony?

LinearRAG output: Best Actress in a Play for Ondine. Result: correct.   
Completion tokens: 3,628.

LivingRAG output: Ondine. Result: correct. Completion tokens: 325. Selected scaffold: Source question: For what did the singer of the original version of Moon River win a Tony award? Stored answer: Ondine.

Table 16: Coverage of stored experiences during LivingRAG inference. Activation-map retrieval counts how often stored experiences are selected in the top-K experience match set used for retrieval guidance. Reasoning scaffold reuse counts how often stored experiences are selected as generation scaffolds. Coverage is the percentage of stored experiences selected at least once.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Stored experiences</td><td colspan="2">Activation-map retrieval</td><td colspan="3">Reasoning scaffold reuse</td></tr><tr><td>Top-K selections</td><td>Coverage (%)</td><td>Uses</td><td>Coverage (%)</td><td>Max uses</td></tr><tr><td>2Wiki</td><td>100</td><td>4820</td><td>97.00</td><td>984</td><td>85.00</td><td>103</td></tr><tr><td>HotpotQA</td><td>549</td><td>2814</td><td>79.60</td><td>984</td><td>58.11</td><td>20</td></tr><tr><td>MuSiQue</td><td>313</td><td>2799</td><td>86.90</td><td>983</td><td>70.29</td><td>30</td></tr><tr><td>MuSiQue-full</td><td>473</td><td>6828</td><td>93.02</td><td>2401</td><td>81.61</td><td>68</td></tr><tr><td>WixQA</td><td>157</td><td>1910</td><td>88.54</td><td>384</td><td>67.52</td><td>20</td></tr></table>

Scaffold score: 0.598. Template score: 0.877. Semantic score: 0.798. Retrieved articles and answer evidence: Moon River: the song was originally performed by Audrey Hepburn in Breakfast at Tiffany’s. List of awards and honours received by Audrey Hepburn: Hepburn won the 1954 Tony Award for Best Performance by a Leading Actress in a Play in Ondine.

Scaffold shown to the LLM: Resolve the Moon River performer as Audrey Hepburn. Then use the Audrey Hepburn awards passage. The Tony Award evidence points to Ondine. Answer with the award object, not the whole intermediate chain.

LLM response after scaffold reuse: The current question uses the same template as the stored case. Moon River in Breakfast at Tiffany’s identifies Audrey Hepburn. Her awards passage states that the Tony Award was for Ondine. The answer is Ondine.

S1 illustrates scaffold reuse without an accuracy change. LivingRAG still uses the same evidence chain, but the scaffold makes the decomposition explicit before generation, reducing completion tokens by an order of magnitude.

S2: Scaffold reuse for a repeated Clarksville–   
Nashville distance hop   
Dataset / index: MuSiQue / 948   
Question: How many miles are there between Nashville and the place in   
Tennessee where Mark Day was born?   
LinearRAG output: 45. Result: correct. Completion tokens: 7,054.   
LivingRAG output: 45 miles. Result: correct. Completion tokens:   
1,042.   
Selected scaffold: Source question: How far is Hod Lisenbee’s place of   
death in TN from Nashville? Stored answer: 45 miles. Scaffold score:   
0.475. Template score: 0.692. Semantic score: 0.732.   
Retrieved articles and answer evidence: Mark Day (racing driver):   
Mark Day was born in Clarksville, Tennessee. Tennessee: Clarksville is   
about 45 miles northwest of Nashville.   
Scaffold shown to the LLM: Resolve the person-specific Tennessee   
location first. Then reuse the Clarksville-to-Nashville distance passage. If   
the current person also maps to Clarksville, answer 45 miles.   
LLM response after scaffold reuse: Mark Day’s birthplace is Clarksville,   
Tennessee. The Tennessee passage gives Clarksville as about 45 miles   
northwest of Nashville. The answer is 45 miles.

S2 shows that the scaffold can be reused even when the surface entity changes. The old case binds Hod Lisenbee to Clarksville, while the new case binds Mark Day to Clarksville; the second hop and the answer form remain the same.

## K Additional Robustness and Transfer Analyses

## K.1 Benefits by Realized Reuse Signal

We further conduct a query-level analysis of realized graph and template reuse. Table 17 reports

Table 17: Performance changes conditioned on realized graph and template reuse.
<table><tr><td>Dataset</td><td>Graph</td><td>Template</td><td>∆LLM- Acc.</td><td>△ Completion tokens</td></tr><tr><td>2Wiki</td><td>一</td><td>1</td><td>+0.59</td><td>-12.8</td></tr><tr><td></td><td>一</td><td>√</td><td>+2.97</td><td>-189.1</td></tr><tr><td>MuSiQue-full</td><td>一</td><td>一</td><td>+0.56</td><td>+23.6</td></tr><tr><td></td><td>√</td><td>一</td><td>+2.12</td><td>-297.2</td></tr><tr><td></td><td>一</td><td>√</td><td>+4.73</td><td>-1352.9</td></tr><tr><td></td><td>√</td><td>√</td><td>+9.81</td><td>-964.7</td></tr><tr><td>WixQA</td><td>一</td><td>一</td><td>+1.03</td><td>-155.6</td></tr><tr><td></td><td>√</td><td>一</td><td>+4.23</td><td>-272.9</td></tr><tr><td></td><td>√</td><td>√</td><td>+8.39</td><td>-454.8</td></tr></table>

paired changes; ∆ = LivingRAG − LinearRAG, and negative completion values indicate savings.

Either signal yields greater benefits than neither, while their combination can further strengthen the gains. Missing rows contain no samples.

## K.2 Query-Order Robustness

We evaluate the same complete 2Wiki, MuSiQuefull, and WixQA streams under two additional random query permutations that change only arrival order. In Table 18, “3 orders” reports the mean and sample standard deviation over the paper order and two permutations.

Accuracy and token cells report LinearRAG / LivingRAG. Accuracies and reuse rates are percentages; tokens are per-query averages. LivingRAG achieves higher accuracy than LinearRAG on every reported metric and uses fewer completion tokens in every tested order. The realized reuse patterns also remain stable after reordering. These consistent results provide evidence that LivingRAG’s benefits are not only driven by one favorable stream arrangement.

Table 18: Robustness across different online query orders.
<table><tr><td>Dataset</td><td>Setting</td><td>Cont.-Acc.</td><td>LLM-Acc.</td><td>Completion tokens</td><td>Graph trans.</td><td>Template reuse</td></tr><tr><td rowspan="2">2Wiki</td><td>Paper</td><td>79.60 / 82.70</td><td>84.40 / 87.00</td><td>1067.6 / 906.4</td><td>0.00</td><td>84.15</td></tr><tr><td>3 orders</td><td>79.13±0.42 / 83.33±2.03</td><td>83.80±1.22 / 89.40±5.14</td><td>1038.7±25.4 / 806.4±92.7</td><td>0.00±0.00</td><td>81.54±2.69</td></tr><tr><td rowspan="2">MuSiQue-full</td><td>Paper</td><td>41.00 / 44.39</td><td>52.71 / 58.42</td><td>2223.8 / 1643.7</td><td>82.42</td><td>50.44</td></tr><tr><td>3 orders</td><td>41.36±1.12/44.21±0.90</td><td>53.10±0.44 /58.09±0.88</td><td>2231.8±26.5 / 1744.8±94.4</td><td>81.84±0.55</td><td>50.15±0.64</td></tr><tr><td rowspan="2">WixQA</td><td>Paper</td><td>一</td><td>66.00 / 70.25</td><td>1653.8 / 1378.1</td><td>91.50</td><td>7.03</td></tr><tr><td>3 orders</td><td></td><td>66.25±1.39 /70.25±1.00</td><td>1689.4±38.6 /1403.6±22.2</td><td>91.92±0.38</td><td>7.29±0.30</td></tr></table>

Table 19: Retrieval-level transfer on WixQA.
<table><tr><td>Setting</td><td>Article R@5</td><td>Hit@5</td><td>MRR@5</td></tr><tr><td>Native PPR</td><td>62.42</td><td>68.00</td><td>42.74</td></tr><tr><td>+ Experience fusion</td><td>63.42</td><td>69.00</td><td>44.82</td></tr></table>

## K.3 Transfer to an Entity-Passage PPR Retriever

We chose LinearRAG because it is a strong, lightweight retriever with an explicit entity state before PPR. This allows us to isolate experience reuse without changing the corpus graph or ranking procedure. To test compatibility beyond LinearRAG, we run a preliminary retrieval-level test on a separate entity-passage PPR retriever. As the test isolates interface compatibility rather than endto-end integration, we report retrieval rather than generation metrics.

The adapter treats verified states as sparse priors over target graph nodes, an interface also exposed by HippoRAG2 (Gutiérrez et al., 2025). For native reset $\mathbf { r } _ { q } = [ \mathbf { r } _ { q } ^ { E } ; \mathbf { r } _ { q } ^ { P } ]$ , our adapter uses

$$
\mathbf { r } _ { q } ^ { \prime } = \left[ ( 1 - \lambda _ { q } ) \mathbf { r } _ { q } ^ { E } + \lambda _ { q } \tilde { \mathbf { h } } _ { q } ; \mathbf { r } _ { q } ^ { P } \right] ,\tag{28}
$$

where $\tilde { \mathbf { h } } _ { q }$ mixes earlier verified states. PPR and ranking remain unchanged.

We use all 400 WixQA queries and 7,132 corpus chunks. The graph uses cached entity annotations and maps only earlier quality-gated states.

R@5 measures top-five gold-article coverage, Hit@5 any gold hit, and MRR@5 the first-hit rank. The gains provide preliminary evidence of transfer through this interface.