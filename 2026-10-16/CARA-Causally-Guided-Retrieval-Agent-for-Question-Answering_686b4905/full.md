# CARA: Causally Guided Retrieval Agent for Question Answering

Weicheng Liu, Xiangcheng Wu, Depeng Xu, Haopeng Zhang, Xi Niu

University of North Carolina at Charlotte

{wliu23, xwu20, dxu7, hzhang55, xniu2}@charlotte.edu

## Abstract

Retrieval-augmented generation over text-rich knowledge graphs requires identifying evidence that contributes to answering a question, rather than merely matching its textual content. Existing agentic retrievers adaptively balance global search and multi-hop exploration but primarily rely on textual similarity to guide retrieval. We propose CARA, a training-free, CAusally guided Retrieval Agent that explicitly evaluates the causal relevance of retrieved evidence through two complementary criteria: consequentiality, measuring whether adding evidence improves answerability, and essentiality, assessing whether removing evidence weakens answerability. CARA integrates these assessments with path consistency to guide graph exploration and employs reciprocal rank fusion to aggregate evidence from multiple agents. We evaluate CARA on three STaRK datasets and augment STaRK with ground-truth answers to enable downstream question answering evaluation. Experimental results demonstrate that CARA consistently outperforms existing retrievers. CARA also improves answer quality and faithfulness while examining fewer graph nodes and requiring fewer retrieval operations. These findings demonstrate that causal relevance assessment enables more focused and effective agentic graph retrieval, improving both retrieval performance and downstream question answering.

## 1 Introduction

As a promising direction in retrieval-augmented generation (RAG), retrieval over text-rich knowledge graphs (KGs) rather than flat documents is increasingly used to help large language models (LLMs) answer questions and ground answers in evidence. Unlike flat document retrieval, KG retrieval must jointly exploit nodes (breadth-oriented) and traverse along relations (depth-oriented). Balancing between the search breadth and depth is a long standing challenge for KG retrieval (Polonuer et al., 2026). With the advances of agents, several recent studies (Sun et al., 2024; Yu et al., 2025) delegate such tasks to retrieval agents which have autonomy to balance search breadth and depth without a pre-specified number of node searches or hops. However, these agents’ retrieval process is driven primarily by textual similarity and graph structure, which can still favor evidence that is only associated with the query but not necessary for answering it. For example, consider the question, “Which medicine should this patient avoid?” A similarity-based retriever may return a clinical note stating, “The patient currently takes several medicines, including aspirin and metformin,” because it contains many words similar to the query. However, this information does not answer the question. Another record may simply state, “Allergy: amoxicillin.” Although this short record has no textual overlap with the question, it provides the key evidence needed to answer the question. This illustrates why textual similarity alone is not enough to make a substantive contribution to the answer.

In this study, we advocate for a causal retrieval agent, which has autonomy, but more importantly, is able to retrieve evidence that is “causally relevant” to answering the query, rather than merely textually similar. In this study, we operationalize the causal relevance using two complementary criteria:

• Consequential: does adding the retrieved evidence materially increase answerability?

• Essential: does removing the retrieved evidence materially reduce answerability?

This operationalization is inspired by recent work on faithful rationale evaluation (DeYoung et al., 2020; Yu et al., 2019). It worthwhile to note that this causal relevance does not focus on the literal “cause-effect” relations, which is only one special case of this broader notion.

In this paper, we design and implement a CAusally guided Retrieval Agent for question answering, named CARA, by retrieving causally relevant evidence, defined as consequential and essential evidence that is directly helpful to answer a query. After global anchoring and during neighborhood hop, the retrieval agent invokes a causal reasoning module, asking whether the current hop is consequential and essential for resolving the information need. This reasoning guides the agent’s next decision on whether to continue hopping, to reanchor globally, or to finish when sufficient support has been found.

Our contributions are threefold: (1) we formulate causal relevance as being consequential and essential, distinguishing answer-bearing evidence from merely textually similar evidence; (2) we introduce a training-free prompt-based causal reasoning module that evaluates the causal relevance from the perspectives of both the current hop and the accumulated path developed so far; and (3) we augment the well-known STaRK benchmark dataset with ground truth answers to the queries (questions), expanding its utility from only evaluating retrieval performance to evaluating both retrieval and question answering performance.

## 2 Related Work

Recent work on retrieval-augmented generation has increasingly moved beyond flat passage retrieval toward structured knowledge, multi-hop reasoning, and autonomous search. Three lines of research are particularly relevant to our work: knowledge-graph retrieval for question answering, causal retrieval and reasoning, and agentic retrieval.

## 2.1 Knowledge Graph Retrieval for LLM Question Answering

Knowledge-graph retrieval represents entities and their relations explicitly, enabling LLMs to access evidence that is distributed across multiple connected pieces of information. Representative works include GraphRAG (Edge et al., 2024), KG2RAG (Zhu et al., 2025), GRAG (Hu et al., 2025), G-Retriever (He et al., 2024), SubgraphRAG (Li et al., 2025a), KAR (Xia et al., 2025), GraphSearch (Yang et al., 2025), HybGRAG (Lee et al., 2025), CoRAG (Zheng et al., 2025), and MoR (Lei et al., 2025).

These approaches demonstrate the value of combining search with graph structure, but they primarily determine relevance through textual, relational, or structural compatibility. They do not explicitly ask whether a retrieved piece of evidence is consequential or essential to answering the query, which motivates our causal relevance formulation.

## 2.2 Causal Reasoning for LLM Question Answering

A growing body of work incorporates causal relations into retrieval and reasoning. Representative works include CDF-RAG (Khatibi et al., 2025), HCG-RAG (Saouda et al., 2026), CausalAgent (Wang et al., 2026), and CausalRAG (Wang et al., 2025).

Although these approaches differ in implementation, they generally associate causality with causeeffect relations, dependency structures, or causal mechanisms represented in the underlying knowledge. Our notion is different. CARA does not require a retrieved edge or node to represent a literal cause-effect relation. Instead, we define causal relevance at retrieval time in terms of whether evidence is consequential and essential for answering the query. Thus, our focus is not causal structure in the world being described, but the causal contribution of retrieved evidence to the answering process.

## 2.3 Agentic Retrieval for LLM Question Answering

Agentic retrieval turns retrieval from a fixed pipeline into an adaptive decision process. Represented works include ReAct (Yao et al., 2023), AvaTaR (Wu et al., 2024a), MA-RAG (Kumar and Jain, 2025), SEMA-RAG (Huang et al., 2026), and OmniSearch (Li et al., 2025b). Several recent methods extend agentic retrieval to knowledge graphs, including Tree-of-Traversals (Markowitz et al., 2024), Think-on-Graph (Sun et al., 2024), Think-on-Graph 2.0 (Ma et al., 2025), RoG (Luo et al., 2024), Search-on-Graph (Sun et al., 2026), and Explore-on-Graph (Yan et al., 2026). Together, these methods establish graph retrieval as an adaptive search process in which an agent can decide where and how far to traverse.

However, existing agentic retrieval approaches are primarily based on textual similarity instead of causal retrieval we advocate for. Our work addresses this gap by introducing causal retrieval to agents in the broader sense of identifying evidence whose presence and absence materially contributes to resolving the information need.

![](images/fc6e0171f8136c961459eae96b8fb2343b13942c8f955a41bd846a3dd2aa8c7d.jpg)  
Figure 1: Overview of CARA. Each retrieval agent explores the knowledge graph with GlobalSearch and Neighbors. The causal reasoning module C assesses the neighborhood candidates. The scores guide the agent’s next action. The agents’ result lists are aggregated into one ranking, which the answer generator uses to answer $Q$

## 3 Method: CARA over Knowledge Graph

The overview of CARA (CAusally guided Retrieval Agent) is in Figure 1. We study agentic retrieval over a knowledge graph $G \ =$ $\langle V , E , \phi _ { V } , \phi _ { E } , d _ { V } \rangle$ , where V and E denote entities and edges, $\phi _ { V }$ and $\phi _ { E }$ assign a type to each node and relation, and $d _ { V } ( v )$ denotes the text attributes associated with node v, such as titles, descriptions, or other metadata. Given a natural language query (or question) $Q ,$ retrieval is formulated as a process in which an agent $A = \langle \mathrm { L L M } , \mathcal { T } , \mathcal { C } \rangle$ interacts with the graph through two retrieval tools T and a causal reasoning module C. The retrieval tools include: 1) GlobalSearch that conducts global node search and 2) Neighbors that conducts one-hop neighborhood exploration. The causal module C is neither an independent agent nor a tool. It is an LLMbased reasoning and assessment module embedded within an agent. It analyzes the candidate nodes returned by the Neighbors tool and provides causal assessments that inform the agent’s subsequent decisions and actions.

The agent produces an interaction trajectory $\tau ~ = ~ \left( ( s _ { 1 } , A _ { 1 } , o _ { 1 } ) , \dots , ( s _ { T } , A _ { T } , o _ { T } ) \right)$ where at step $t , s _ { t }$ contains 1) the original query Q, 2) the agent interaction history $I _ { t }$ , and 3) the current paths $P _ { t }$ that records one or multiple traversed paths so far. $A _ { t }$ denotes the agent’s actions such as calling a retrieval tool, terminate, etc. $O t$ is the output (a set of retrieved nodes $R _ { t }$ and the corresponding paths $P _ { t + 1 }$ leading to those nodes). At each step, the agent may Append the newly retrieved nodes to $R _ { t }$ , perform additional global or neighborhood retrieval, or terminate by issuing Finish. Execution ends when the agent calls Finish or when the maximum trajectory length $T _ { \mathrm { m a x } }$ is reached.

## 3.1 Retrieval Tools

For an agent, there are two retrieval tools: 1) GlobalSearch for seed nodes search (breadth-oriented search) and 2) Neighbors for one-step hopping along the relations (depth-oriented search).

GlobalSearch retrieves the k highest-scoring nodes in the entire graph for an agent-generated subquery q based on the original Q:

$$
\mathcal { G } ( q , k ) = \mathrm { T o p - k } _ { v \in V } r e l ( q , d _ { V } ( v ) ) .\tag{1}
$$

We implement rel() using BM25 (Robertson and Zaragoza, 2009) over an inverted index of node text attributes $d _ { V } ( v )$ . The agent-generated subquery q can evolve throughout exploration, allowing the agent to issue focused retrieval requests based on the trajectory state $s _ { t }$ . Global search is a breadth oriented search over KG, providing broad entry points into the graph or re-anchoring points when a local path becomes unproductive.

Neighbors retrieves adjacent (one-hop neighbor) nodes subject to optional node- and edge-type constraints $F _ { t } = ( F _ { V , t } , F _ { E , t } )$ selected by the agent. The result is a filtered set $N _ { F _ { t } } ( v _ { t } )$ of one-hop neighborhood of the current node $v _ { t }$ . To further trim this filtered set, this Neighbors tool conducts another BM25 search to keep the top k from this filtered set:

$$
\begin{array} { r l } & { \mathcal { N } _ { t } = \mathrm { N e i g h b o r s } ( v _ { t } , q _ { t } , F _ { t } ) } \\ & { \quad = \mathrm { T o p } \mathrm { - k } _ { u \in N _ { F _ { t } } ( v _ { t } ) } r e l ( q _ { t } , d _ { V } ( u ) ) . } \end{array}\tag{2}
$$

Here $\mathcal { N } _ { t } = \{ u _ { 1 } , \ldots , u _ { k } \}$ denotes the candidate neighbors retrieved at hop t. The repeated applications of the Neighbors tool compose into multi-hop traversal. GlobalSearch remains available along the way for re-anchoring. Unlike traditional nonagentic retrieval, the number of hops is not specified in advance. Instead, the agent decides whether to continue hopping, re-anchor globally, or finish based on the accumulated evidence. It is noteworthy that each candidate neighborhood node $u _ { i } .$ , will further be passed through the causal reasoning module C before the agent chooses its next action.

## 3.2 Causal Reasoning Module

Neighbors retrieval identifies nodes that are structurally adjacent and textually similar to the current exploration state. However, neither graph adjacency nor textual similarity guarantees that a candidate node provides evidence that materially contributes to answering the query. Therefore we introduce a causal reasoning module C that evaluates each candidate according to its consequentiality and essentiality for resolving the information need.

Given the candidate set $\mathcal { N } _ { t } = \{ u _ { 1 } , . . . , u _ { k } \}$ returned by the Neighbors tool, the module receives the query $Q ,$ the current node $v _ { t } .$ , the candidate nodes $\mathcal { N } _ { t }$ , their connecting relations, and the accumulated paths $P _ { t + 1 }$ . Unlike literal causal reasoning that seeks cause-effect relations in the underlying domain, our notion of causal relevance is retrievaloriented: evidence is considered causally relevant when its presence or absence materially affects the ability to support an answer to $Q .$ . This operationalization is motivated by two complementary properties: consequentiality and essentiality.

For each candidate $u _ { i } .$ , the module C estimates a consequentiality score $c _ { i } \in [ 0 , 1 ]$ and an essentiality score $e _ { i } \in [ 0 , 1 ]$ . Consequentiality asks whether incorporating the candidate node and its associated hop materially changes the evidence available for answering the query. It captures the candidate’s incremental answer contribution, rather than simply its textual similarity.

The LLM is therefore asked whether adding $u _ { i }$ together with its relation to $v _ { t }$ , introduces information that substantially strengthens or advances the answer.

Essentiality instead asks whether removing the candidate evidence from the currently available evidence would materially reduce the ability to justify an answer.

More directly, the LLM is prompted to reason counterfactually: if this candidate node and its connecting relation were unavailable, would an important part of the evidence needed to answer the query be lost? A high $e _ { i }$ therefore indicates that the candidate carries information that is difficult to replace with evidence already contained in the current trajectory.

We combine these two judgments into a nodelevel causal relevance score:

$$
c a u _ { i } = \lambda c _ { i } + ( 1 - \lambda ) e _ { i } ,\tag{3}
$$

where $\lambda \in [ 0 , 1 ]$ controls the relative emphasis on consequentiality and essentiality. Importantly, cau<sub>i</sub> should not be interpreted as a probability that a literal causal relationship exists between $v _ { t }$ and $u _ { i }$ Rather, it measures whether the candidate evidence has a materially incremental and counterfactual contribution to answering $Q$

Causal relevance alone, however, evaluates a candidate primarily with respect to the query and the evidence currently available. In multi-hop retrieval, a locally useful candidate may still lead the agent away from the reasoning trajectory developed so far. We therefore introduce a separate path consistency score $p _ { i } \in [ 0 , 1 ]$ . Unlike $c a u _ { i }$ , which measures the node-level contribution to answering the query, $p _ { i }$ evaluates whether the transition $v _ { t } \to u _ { i }$ forms a coherent continuation of the accumulated path developed so far. In particular, the module examines whether the relation connecting $v _ { t }$ and $u _ { i }$ is compatible with the information need represented by the current trajectory and whether following this candidate preserves a plausible chain of evidence toward the answer.

The final score for candidate node $u _ { i }$ is:

$$
s _ { t } ( u _ { i } ) = \alpha c a u _ { i } + ( 1 - \alpha ) p _ { i } ,\tag{4}
$$

where $\alpha \in [ 0 , 1 ]$ controls the relative importance of node-level causal relevance and the trajectory-level consistency. The highest-scoring candidates are returned to the retrieval agent.

Importantly, the causal reasoning module C is an assessment module rather than an autonomous agent. It does not independently invoke GlobalSearch or Neighbors. Instead, its assessments enhance the information presented to the retrieval agent and therefore influence subsequent search decisions. Based on the causal and path-consistency assessments, the agent may continue along a promising neighborhood path with Neighbors, reformulate the subquery, modify node or relation filters, re-anchor through GlobalSearch, Append nodes to the retrieved set, or terminate with Finish when additional exploration is unlikely to provide consequential evidence. In this way, causal reasoning affects not merely the reranking of already retrieved nodes, but the direction and termination of the retrieval process itself.

## 3.3 Aggregation of Each Agent Result

To improve robustness and reduce dependence on a single retrieval trajectory, we run multiple agents in parallel. Each agent j independently interacts with the knowledge graph, producing a retrieval trajectory $\tau ^ { ( j ) }$ and an ordered list of retrieved nodes $R ^ { ( j ) }$ Independent trajectories may differ in their initial anchors, generated subqueries, neighborhood expansions, and traversal decisions. Such diversity allows different agents to explore alternative paths through the graph and reduces sensitivity to any single search trajectory.

Rather than directly concatenating the node lists from different agents, inspired by the classic idea of reciprocal rank fusion (RRF) (Cormack et al., 2009), we aggregate them in the following way. For a candidate node $u ,$ its cross-agent fusion score is:

$$
\mathrm { R R F } ( u ) = \sum _ { j : u \in R ^ { ( j ) } } \frac { 1 } { \operatorname { r a n k } _ { j } ( u ) } ,\tag{5}
$$

where rank $\dot { \mathbf { \rho } } _ { j } ( u )$ denotes the position of node u in the ranked evidence list $R ^ { ( j ) }$ returned by agent $j .$ In this new context, RRF captures two complementary properties of the retrieved evidence. First, it rewards high within-agent rank. Second, it rewards cross-agent consensus. Thus, evidence that is repeatedly identified across different trajectories receives stronger aggregate support than evidence found by only one trajectory. The final candidate set is formed from the union of the nodes retrieved by all agents,

ranked in descending order of their RRF scores. In the rare case that two nodes receive identical RRF scores, the node with the better best rank across agents is placed first.

## 4 Experiment Setup

We evaluate CARA from two complementary perspectives: retrieval performance and question answering performance.

## 4.1 STaRK and Ground-Truth Answer Augmentation

We evaluate on the three text-rich knowledge graphs in the well-known benchmark STaRK (Wu et al., 2024b). It contains three datasets: AMA-ZON, MAG, and PRIME. AMAZON represents an e-commerce domain, MAG a scholarly domain, and PRIME a biomedical domain. STaRK provides natural-language queries $Q$ together with ground-truth entities relevant to answer $Q ,$ allowing retrieval systems to be evaluated against known nodes. However, it does not provide a ground truth for evaluating generated answers. We therefore augment STaRK with ground-truth answers. For each query $Q ,$ , we use the ground truth entities’ text attributes $d _ { V } ( v )$ and related relations as the source information and employ Qwen3.8-27B to generate answers. We instruct the model in the prompt to satisfy four requirements: Relevance, ensuring that the answer directly addresses $Q ;$ Faithfulness, ensuring that all factual claims are supported by the provided entity information; Coverage, ensuring that the names of all ground-truth entities are explicitly mentioned; and Naturalness, ensuring that the answer is fluent and natural rather than enumerating a list of retrieved entities. After answer generation, we took two extra steps to validate the answers. First, we validate the answer using an automatically calculated indicator Completeness defined as the proportion of ground-truth entities whose complete names are explicitly mentioned in the generated answer. If Completeness is below a threshold, we prompt the model to regenerate a new version of answer until it reaches the threshold. We repeat this generation-and-validation process, resulting in more complete ground truth answers. Second, we manually inspect stratified samples (approximately 10% of the generated answers) across AMAZON, MAG, and PRIME to validate that the resulting ground truth answers are appropriate across different domains and query types.

## 4.2 Baselines

We compare CARA with conventional retrievers and agentic retrievers used in prior work. The conventional retrievers include the classic sparse retrieval, BM25 (Robertson and Zaragoza, 2009); dense embedding retrieval that rank nodes by cosine similarity, ada-002 (OpenAI, 2022) and GritLM-7B (Muennighoff et al., 2025); knowledgeaware retrieval KAR (Xia et al., 2025) that conducts query expansion and applies relation type constraints during retrieval. Agentic retrievers include Think-on-Graph (Sun et al., 2024), which is a training-free LLM agent that iteratively expands paths in the graph using beam search; GraphFlow (Yu et al., 2025), which learns to generate multihop retrieval trajectories using GFlowNets (Bengio et al., 2021); and ARK (Polonuer et al., 2026), which has adaptive use of global search and neighborhood exploration but without causal reasoning during graph traversal.

All agentic systems (CARA and agentic baselines), the causal module, the common answer generator use the same open-weight backbone, Qwen3.8-27B, served with vLLM with thinking disabled. The agents’ LLM calls sample with temperature 1.0, and all other calls decode greedily. CARA runs m = 3 agents per query (question) with $T _ { \mathrm { m a x } } = 1 5 0$ steps each; GlobalSearch returns $k = 3 0$ candidates and Neighbors returns k = 20.

## 4.3 Retrieval Evaluation

We first evaluate the performance of the entityretrieval task. We report Hit@1, Recall@20, and Mean Reciprocal Rank (MRR). These metrics capture complementary aspects of retrieval performance: the proportion of a correct entity ranked first, the proportion of ground truth entities successfully retrieved among the top 20 ranked results, and how highly the first correct entity is ranked overall.

## 4.4 Question Answering Evaluation

We next evaluate whether the retrieved evidence improves downstream question answering. To isolate the effect of retrieval, all retrieval systems use the same answer generator (LLM, Qwen3.8-27B), prompt, decoding configuration, and context budget. The only difference between systems is the retrieved evidence supplied to the answer generator. In this end-to-end setting, each retriever produces its own ranked evidence set, and the common generator answers the question using that evidence. To test whether better retrieval leads to better answers, we evaluate answer correctness using BERTScore (Zhang et al., 2020) between the generated answer and the ground truth answer. We also evaluate answer faithfulness using the Ragas framework (Es et al., 2024), measuring whether factual claims in the generated answer are supported by the retrieved context.

## 5 Results

## 5.1 Retrieval Performance

As the first step of experiment, we tried different λ values from 0 to 1 (in Equation 3, balancing consequentiality and essentiality) and α values from 0 to 1 (in Equation 4, balancing node-level causal relevance and path consistency). It is worthnoting that the setting of λ = 0 or 1 or α = 0 or 1 is equivalent to a component-level ablation study. Averaged across datasets and metrics, λ = 0.5 and α = 0.5 achieve the highest performance, supporting the selection of these values.

Table 1 compares CARA $( \lambda = 0 . 5 , \alpha = 0 . 5 )$ with conventional and agentic retrievers on the three STaRK datasets. CARA achieves the highest Hit@1 and MRR across AMAZON, MAG, and PRIME, demonstrating its effectiveness in ranking relevant entities. Compared with ARK, its closest agentic baseline, CARA’s improvements are particularly pronounced on MAG, whose questions tend to be relation-heavy. These findings suggest that incorporating causal relevance assessment and the rank-based aggregation into adaptive graph exploration improves retrieval effectiveness across diverse knowledge domains, especially in relationheavy domains.

## 5.2 Question Answering Performance

Table 2 evaluates the downstream question answering performance of CARA against KAR and ARK using BERTScore and the RAGAS faithfulness. CARA achieves the highest BERTScore across all three datasets, demonstrating that improved retrieval effectiveness translates into better answer quality. CARA also consistently improves faithfulness over ARK, with particularly notable gains on MAG and PRIME. Interestingly, KAR achieves the highest faithfulness across all datasets despite substantially lower BERTScore values. This suggests that generating answers closely grounded in retrieved evidence does not necessarily guarantee their correctness or completeness relative to ground-truth answers. Overall, these findings highlight the importance of evaluating both answer quality and faithfulness, and demonstrate that causalguided retrieval improves downstream question answering while maintaining strong evidence grounding.

## 5.3 Impact of Exploration Budget on Performance

Figure 2 investigates how CARA’s retrieval and question answering performance varies with the exploration budget, controlled by the number of agents (1–3) and maximum exploration steps $( T _ { \mathrm { m a x } } )$ , ranging from 5 to 150. Overall, both Recall@20 and BERTScore improve substantially as the exploration budget increases, but exhibit diminishing returns beyond a certain point. Increasing the number of agents consistently improves Recall@20 for both AMAZON and PRIME datasets, highlighting the benefits of exploring diverse retrieval trajectories. However, further increasing $T _ { \mathrm { m a x } }$ to 150 produces only marginal improvements, suggesting that most relevant evidence is discovered relatively early during exploration.

<table><tr><td></td><td colspan="3">AMAZON</td><td colspan="3">MAG</td><td colspan="3">PRIME</td></tr><tr><td>Method</td><td>Hit@1</td><td>R@20</td><td>MRR</td><td>Hit@1</td><td>R@20</td><td>MRR</td><td>Hit@1</td><td>R@20</td><td>MRR</td></tr><tr><td>Conventional retrievers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>45.3</td><td>53.7</td><td>55.4</td><td>25.9</td><td>45.7</td><td>34.9</td><td>12.7</td><td>31.3</td><td>19.8</td></tr><tr><td>ada-002</td><td>39.7</td><td>53.8</td><td>50.9</td><td>29.1</td><td>48.4</td><td>38.6</td><td>12.6</td><td>36.0</td><td>21.3</td></tr><tr><td>GritLM-7B</td><td>49.8</td><td>61.8</td><td>60.5</td><td>39.1</td><td>61.4</td><td>50.5</td><td>15.0</td><td>38.4</td><td>23.4</td></tr><tr><td>KAR</td><td>49.6</td><td>59.7</td><td>59.0</td><td>50.1</td><td>63.9</td><td>58.6</td><td>23.3</td><td>46.1</td><td>32.2</td></tr><tr><td>Agentic retrievers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Think-on-Graph</td><td>22.2</td><td>27.8</td><td>29.2</td><td>36.8</td><td>52.7</td><td>43.6</td><td>15.8</td><td>37.8</td><td>23.5</td></tr><tr><td>Think-on-Graph 2.0</td><td>48.8</td><td>37.3</td><td>54.6</td><td>43.9</td><td>50.8</td><td>48.6</td><td>23.8</td><td>43.8</td><td>30.9</td></tr><tr><td>GraphFlow†</td><td>47.9</td><td>36.2</td><td>55.5</td><td>39.1</td><td>57.2</td><td>47.8</td><td>51.4</td><td>79.7</td><td>61.4</td></tr><tr><td>ARK</td><td>59.0</td><td>60.2</td><td>67.7</td><td>80.1</td><td>77.9</td><td>83.1</td><td>58.6</td><td>71.9</td><td>65.3</td></tr><tr><td>CARA (ours)*</td><td>62.4</td><td>60.5</td><td>71.3</td><td>84.9</td><td>80.8</td><td>88.1</td><td>59.5</td><td>73.4</td><td>67.7</td></tr></table>

Table 1: Retrieval performance (%) on the full STaRK test sets (AMAZON n=1,642, MAG n=2,665, PRIME $\scriptstyle n = 2 , 8 0 1 )$ . <sup>†</sup>Reported by Yu et al. (2025) (LLaMA3-8B-Instruct, test-0.1 subset). <sup>∗</sup>CARA version: $\lambda = 0 . 5 , \alpha = 0 . 5$ Bold: best; underline: second best.

<table><tr><td colspan="3">AMAZON</td><td colspan="2">MAG</td><td colspan="2">PRIME</td></tr><tr><td>Method</td><td>BS</td><td>Faith.</td><td>BS</td><td>Faith.</td><td>BS</td><td>Faith.</td></tr><tr><td>No retrieval</td><td>7.5</td><td></td><td>16.7</td><td></td><td>27.9</td><td></td></tr><tr><td>KAR</td><td>28.3</td><td>88.3</td><td>33.0</td><td>86.6</td><td>36.0</td><td>88.6</td></tr><tr><td>ARK</td><td>30.1</td><td>87.4</td><td>39.2</td><td>78.6</td><td>40.5</td><td>80.5</td></tr><tr><td>CARA (ours)*</td><td>30.7</td><td>88.0</td><td>40.2</td><td>82.6</td><td>41.3</td><td>86.1</td></tr></table>

Table 2: Question answering on the full STaRK test sets. All methods share the same answer generator. BERTScore (BS, ×100) compares each answer with the ground-truth answer. Faithfulness (Faith., %) is the fraction of answer claims supported by the retrieved context. Faithfulness is not defined in the “No retrieval” approach. <sup>∗</sup>CARA version: $\lambda = 0 . 5 , \alpha = 0 . 5$ . Bold: best; underline: second best.

Interestingly, question answering performance exhibits a similar but less consistent trend. With three agents, BERTScore improves rapidly at smaller exploration budgets and largely stabilizes beyond $T _ { \mathrm { m a x } } = 3 0$ . On PRIME, BERTScore even decreases slightly from 42.1 at $T _ { \mathrm { m a x } } = 3 0$ to 41.6 at $T _ { \mathrm { m a x } } ~ = ~ 1 5 0$ , suggesting that retrieving additional evidence does not necessarily improve answer quality.

## 5.4 Ablation Study

Table 3 investigates the contributions of the causal reasoning module (C) and reciprocal rank fusion (RRF), compared with the count-based aggregation used by ARK. Removing C consistently degrades retrieval and question answering performance. Replacing RRF with ARK’s count-based aggregation yields smaller performance differences. However, count-based aggregation achieves marginally higher Hit@1 on PRIME and faithfulness on MAG and PRIME. Overall, these results demonstrate that causal reasoning provides the primary performance gains, while RRF offers additional improvements in evidence ranking and answer quality.

![](images/99807537674c1db4748b84a19684efce079252d28d793c87238e03586e335134.jpg)  
Figure 2: Retrieval (Recall@20, %) and answer quality (BERTScore, ×100) of CARA on AMAZON and PRIME under different exploration budgets: maximum steps per agent $T _ { \mathrm { m a x } }$ (rows) and number of agents (columns). The outlined cell marks each panel’s peak (earliest budget when tied). All values are on the STaRK test-0.1 subset (10% of each test set).

<table><tr><td></td><td colspan="5">AMAZON (n=1,642)</td><td colspan="5">MAG (n=2,665)</td><td colspan="5">PRIME (n=2,801)</td></tr><tr><td></td><td colspan="2">Retrieval</td><td colspan="2"></td><td colspan="2">QA</td><td colspan="2">Retrieval</td><td colspan="2">QA</td><td colspan="2">Retrieval</td><td colspan="2"></td><td colspan="2">QA</td></tr><tr><td>Variant</td><td>Hit@1 R@20 MRR</td><td></td><td></td><td>BS</td><td>Faith.</td><td>Hit@1 R@20 MRR</td><td></td><td></td><td>BS</td><td>Faith.</td><td></td><td>Hit@1 R@20 MRR</td><td></td><td></td><td>BS Faith.</td><td></td></tr><tr><td>CARA (ours)*</td><td>62.4</td><td>60.5</td><td>71.3</td><td>30.7</td><td>88.0</td><td>84.9</td><td></td><td>80.8</td><td>88.1</td><td>40.2</td><td>82.6</td><td>59.5</td><td>73.4</td><td>67.7</td><td>41.3</td><td>86.1</td></tr><tr><td>w/o C</td><td>60.5</td><td>60.2</td><td>69.1</td><td>30.4</td><td>87.7</td><td>80.3</td><td></td><td>77.9</td><td>83.5</td><td>39.2</td><td>78.5</td><td>58.1</td><td>72.1</td><td>65.4</td><td>40.8</td><td>80.7</td></tr><tr><td>count-based</td><td>62.2</td><td>60.5</td><td>70.3</td><td>30.6</td><td>88.0</td><td>84.4</td><td>80.8</td><td></td><td>87.4</td><td>40.1</td><td>82.8</td><td>60.0</td><td>73.1</td><td>67.2</td><td>40.9</td><td>86.3</td></tr><tr><td>w/o C, count-based</td><td>59.0</td><td>60.2</td><td>67.7</td><td>30.1</td><td>87.4</td><td>80.1</td><td></td><td>77.9</td><td>83.1</td><td>39.2</td><td>78.6</td><td>58.6</td><td>71.9</td><td>65.3</td><td>40.5</td><td>80.5</td></tr></table>

Table 3: Ablation of the causal reasoning module C and the rank-based aggregation (RRF) on the full STaRK test sets. w/o C: agents without C. count-based: nodes ranked by the frequency of agents’ votes instead of RRF; w/o ${ \mathcal { C } } ,$ count-based is equivalent to ARK (Tables 1 and 2). <sup>∗</sup>CARA version: $\lambda = 0 . 5 , \alpha = 0 . 5$ . Bold: best; underline: second best.
<table><tr><td colspan="3"></td><td colspan="3">LLM calls</td><td colspan="2">Tool calls</td><td rowspan="2">Nodes examined</td><td rowspan="2">Latency (s)</td></tr><tr><td>Dataset</td><td>Method</td><td>Agent</td><td>C module</td><td>Total</td><td>(k)</td><td>GlobalSearch</td><td>Neighbors</td></tr><tr><td rowspan="2">AMAZON</td><td>ARK</td><td>26.9</td><td></td><td>26.9</td><td>310</td><td>18.2</td><td>15.2</td><td>152</td><td>105</td></tr><tr><td>CARA (ours)*</td><td>22.1</td><td>8.4</td><td>30.6</td><td>244</td><td>13.2</td><td>10.9</td><td>122</td><td>145</td></tr><tr><td rowspan="2">MAG</td><td>ARK</td><td>31.5</td><td>一</td><td>31.5</td><td>338</td><td>16.1</td><td>30.4</td><td>163</td><td>136</td></tr><tr><td>CARA (ours)*</td><td>25.8</td><td>20.1</td><td>45.9</td><td>340</td><td>11.9</td><td>22.3</td><td>123</td><td>223</td></tr><tr><td rowspan="2">PRIME</td><td>ARK</td><td>41.2</td><td>一</td><td>41.2</td><td>469</td><td>20.2</td><td>47.8</td><td>249</td><td>141</td></tr><tr><td>CARA (ours)*</td><td>32.4</td><td>24.1</td><td>56.5</td><td>511</td><td>14.9</td><td>32.1</td><td>198</td><td>276</td></tr></table>

Table 4: Per-query cost on the full STaRK test sets, summed over the three agents. Lower is better. Total counts agent calls and C module calls. Tokens cover these calls. Latency is the mean wall-clock time per query on Nvidia H200 GPUs (Qwen3.8-27B served with vLLM).<sup>∗</sup>CARA version: $\lambda = 0 . 5 , \alpha = 0 . 5$

## 5.5 Efficiency and Exploration Behavior

Table 4 compares the computational efficiency and exploration behavior of CARA and ARK across three datasets. Interestingly, CARA achieves more focused graph exploration despite introducing an additional causal reasoning module. Compared with ARK (the closest agentic baseline), CARA reduces GlobalSearch calls as well as Neighbors calls across the datasets. The number of examined nodes also decreases for the three datasets. These results suggest that causal relevance assessment helps agents identify promising evidence paths while avoiding unnecessary exploration.

Compared with AMAZON whose questions are more entity-oriented, MAG and PRIME involve more relation-intensive biomedical questions that require deeper multi-hop exploration. This difference is reflected in CARA’s substantially higher number of Neighbors calls and therefore causal reasoning calls per query, suggesting that causal guidance is particularly important for navigating complex relational structures. However, the additional causal reasoning introduces latency overhead. The particularly pronounced overhead on MAG and PRIME highlights a trade-off between retrieval effectiveness and computational efficiency, suggesting opportunities for reducing latency through parallel or batched causal assessments.

## 6 Conclusion

In this paper, we proposed CARA, a training-free causally guided retrieval agent for question answering over text-rich knowledge graphs. Unlike existing agentic retrievers that primarily rely on textual similarity, CARA evaluates retrieved evidence through consequentiality and essentiality, together with path consistency, to guide graph exploration. Reciprocal rank fusion further aggregates evidence from multiple independent agents. We also augmented the STaRK benchmark with ground-truth answers to enable downstream question answering evaluation. Experiments on three datasets demonstrate that CARA consistently improves retrieval effectiveness, while also improving answer quality and faithfulness compared with ARK. Moreover, CARA achieves more focused exploration by examining fewer nodes and requiring fewer retrieval operations. However, the additional causal reasoning introduces inference latency overhead, particularly for relation-intensive questions.

## Limitations

Despite its promising results, CARA has several limitations. First, causal relevance is operationalized through LLM-based assessments of consequentiality and essentiality rather than formal inference. These assessments may be sensitive to model biases, prompting strategies, and errors in evaluating evidence contributions. Second, although CARA reduces retrieval operations and examined nodes, the additional causal reasoning module increases computational overhead and inference latency, particularly for relation-intensive questions. Third, downstream question answering evaluation relies on automatically constructed ground-truth answers and LLM-based faithfulness assessments, which may introduce evaluation noise. Future work will explore more efficient causal assessments and alternative evaluations.

## References

Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. 2021. Flow network based generative models for non-iterative diverse candidate generation. Advances in neural information processing systems, 34:27381–27394.

Gordon V Cormack, Charles LA Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval, pages 758–759.

Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C. Wallace. 2020. ERASER: A benchmark to evaluate rationalized NLP models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4443–4458. Association for Computational Linguistics.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th conference of the european chapter of the associationfor computational linguistics: system demonstrations, pages 150–158.

Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented

generation for textual graph understanding and question answering. Advances in Neural Information Processing Systems, 37:132876–132907.

Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025. Grag: Graph retrievalaugmented generation. In Findings ofthe association for computational linguistics: NAACL 2025, pages 4145–4157.

Yongfeng Huang, Ruiying Chen, and James Cheng. 2026. Sema-rag: A self-evolving multi-agent retrieval-augmented generation framework for medical reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 18423– 18442.

Elahe Khatibi, Ziyu Wang, and Amir M Rahmani. 2025. Cdf-rag: Causal dynamic feedback for adaptive retrieval-augmented generation. arXiv preprint arXiv:2504.12560.

Deepak Kumar and Bhavesh Jain. 2025. Multi-agent retrieval-augmented generation for enhancing answer generation and knowledge retrieval. In EPIA Conference on Artificial Intelligence, pages 289–302. Springer.

Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N Ioannidis, Huzefa Rangwala, and Christos Faloutsos. 2025. Hybgrag: Hybrid retrieval-augmented generation on textual and relational knowledge bases. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 879–893.

Yongjia Lei, Haoyu Han, Ryan A Rossi, Franck Dernoncourt, Nedim Lipka, Mahantesh M Halappanavar, Jiliang Tang, and Yu Wang. 2025. Mixture of structuraland-textual retrieval over text-rich graph knowledge bases. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 18306–18321.

Mufei Li, Siqi Miao, and Pan Li. 2025a. Simple is effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation. In International Conference on Learning Representations, volume 2025, pages 6061–6089.

Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao Zheng, Fei Huang, Jingren Zhou, et al. 2025b. Benchmarking multimodal retrieval augmented generation with dynamic vqa dataset and self-adaptive planning agent. In International Conference on Learning Representations, volume 2025, pages 95582–95604.

Linhao Luo, Yuan-Fang Li, Reza Haffari, and Shirui Pan. 2024. Reasoning on graphs: Faithful and interpretable large language model reasoning. In International Conference on Learning Representations, volume 2024, pages 14400–14423.

Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo.

2025. Think-on-graph 2.0: Deep and faithful large language model reasoning with knowledge-guided retrieval augmented generation. In International Conference on Learning Representations, volume 2025, pages 52782–52806.

Elan Markowitz, Anil Ramakrishna, Jwala Dhamala, Ninareh Mehrabi, Charith Peris, Rahul Gupta, Kai-Wei Chang, and Aram Galstyan. 2024. Tree-oftraversals: A zero-shot reasoning algorithm for augmenting black-box language models with knowledge graphs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12302–12319.

Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. 2025. Generative representational instruction tuning. In International Conference on Learning Representations (ICLR).

OpenAI. 2022. New and improved embedding model.

Joaquín Polonuer, Lucas Vittor, Iñaki Arango, Ayush Noori, David A Clifton, Luciano Del Corro, and Marinka Žitnik. 2026. Autonomous knowledge graph exploration with adaptive breadth-depth retrieval. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15694–15709.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and trends® in information retrieval, 4(1-2):1–174.

Marc Saouda, Rajprakash Bale, Eren Aldis, and Cloves Almeida. 2026. Structure over scale: Schemaconstrained causal graphs for rag. arXiv preprint arXiv:2607.22592.

Jia Ao Sun, Hao Yu, Fabrizio Gotti, Fengran Mo, Yihong Wu, Yuchen Hui, Zhan Su, Lingfeng Xiao, and Jian-Yun Nie. 2026. Search-on-graph: Iterative informed navigation for large language model reasoning on knowledge graphs. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 4475–4485.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In International Conference on Learning Representations, volume 2024, pages 3868–3898.

Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma, and Vipin Chaudhary. 2025. Causalrag: Integrating causal graphs into retrieval-augmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 22680–22693.

Tianyi Wang, Tao Shen, Zhiyuan Zhang, Shuangping Huang, Huiguo He, Qingguang Chen, and Houqiang

Yang. 2026. Causalagent: A hierarchical graphenhanced multi-agent framework for causal question answering in production safety accident reports. Algorithms, 19(5):355.

Shirley Wu, Shiyu Zhao, Qian Huang, Kexin Huang, Michihiro Yasunaga, Kaidi Cao, Vassilis N Ioannidis, Karthik Subbian, Jure Leskovec, and James Zou. 2024a. Avatar: Optimizing llm agents for tool usage via contrastive reasoning. Advances in Neural Information Processing Systems, 37:25981–26010.

Shirley Wu, Shiyu Zhao, Michihiro Yasunaga, Kexin Huang, Kaidi Cao, Qian Huang, Vassilis N Ioannidis, Karthik Subbian, James Zou, and Jure Leskovec. 2024b. Stark: Benchmarking llm retrieval on textual and relational knowledge bases. Advances in Neural Information Processing Systems, 37:127129–127153.

Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A Rossi, Haoliang Wang, and Julian McAuley. 2025. Knowledge-aware query expansion with large language models for textual and relational retrieval. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4275–4286.

Shiqi Yan, Yubo Chen, Ruiqi Zhou, Zhengxi Yao, Shuai Chen, Tianyi Zhang, Shijie Zhang, Wei-Qiang Zhang, Yongfeng Huang, Haixin Duan, et al. 2026. Exploreon-graph: Incentivizing autonomous exploration of large language models on knowledge graphs with path-refined reward modeling. In International Conference on Learning Representations, volume 2026, pages 56761–56789.

Cehao Yang, Xiaojun Wu, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Jia Li, Hui Xiong, and Jian Guo. 2025. Graphsearch: An agentic deep searching workflow for graph retrieval-augmented generation. arXiv preprint arXiv:2509.22009.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Junchi Yu, Yujie Liu, Jindong Gu, Philip Torr, and Dongzhan Zhou. 2025. Can knowledge-graphbased retrieval augmented generation really retrieve what you need? In Advances in Neural Information Processing Systems (NeurIPS). ArXiv:2510.16582; code: https://github.com/ Samyu0304/GraphFlow.

Mo Yu, Shiyu Chang, Yang Zhang, and Tommi Jaakkola. 2019. Rethinking cooperative rationalization: Introspective extraction and complement control. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4094–4103.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations (ICLR).

Zaiyi Zheng, Song Wang, Zihan Chen, Yaochen Zhu, Yinhan He, Liangjie Hong, Qi Guo, and Jundong Li. 2025. Corag: Enhancing hybrid retrieval-augmented generation through a cooperative retriever architecture. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 16088– 16101. Association for Computational Linguistics.

Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge graph-guided retrieval augmented generation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8912–8924.

<table><tr><td colspan="2">Backbone</td></tr><tr><td>LLM</td><td>Qwen3.8-27B for every call, thinking mode off</td></tr><tr><td>Serving</td><td>vLLM 0.30.0; 6 replicas, 1 H200 each</td></tr><tr><td colspan="2">Agents</td></tr><tr><td>Agents</td><td>m = 3 per question, run in parallel</td></tr><tr><td>Step budget</td><td>Tmax = 150 LLM calls per agent temperature 1.0, top-p 0.95, top-k 20</td></tr><tr><td>Decoding Actions</td><td>GlobalSearch, Neighbors, Append, Finish</td></tr><tr><td colspan="2">Tools</td></tr><tr><td>GlobalSearch</td><td>BM25 over node names and descriptions; up to k = 30 results</td></tr><tr><td>Neighbors</td><td>one hop; type filters; BM25 subquery; k = 20 shown</td></tr><tr><td colspan="2">Causal module C</td></tr><tr><td>Calls</td><td>one per Neighbors call with candidates, all at once</td></tr><tr><td>Output</td><td> $c _ { i } , e _ { i } , p _ { i } \in [ 0 , 1 ]$  per candidate</td></tr><tr><td>Decoding Weights</td><td>temperature 0, at most 1,024 tokens  $\lambda = 0 . 5 , \alpha = 0 . 5$ </td></tr><tr><td colspan="2">Aggregation</td></tr><tr><td>Rule</td><td>RRF  $\textstyle \sum _ { j }$  1/rankj(u)</td></tr><tr><td>Ties</td><td>best rank, then first position</td></tr><tr><td colspan="2">QA evaluation</td></tr><tr><td>Gen. input</td><td>top-5 nodes, evidence body ≤1,500 characters each</td></tr><tr><td>Gen. decoding BERTScore</td><td>temperature 0, at most 2,000 tokens F1, roberta-large, rescaled, × 100</td></tr><tr><td>Faithfulness</td><td>RAGAS 0.4.3; judge Qwen3.8-27B; mean</td></tr><tr><td>Test questions QA questions</td><td>of 2 passes; empty context = 0 Amazon 1,642; MAG 2,665; Prime 2,801 shared by all methods</td></tr></table>

Table 5: Settings of the implementation

## A.2 Prompts of the Causal Reasoning Module

Table 5 lists the settings of the implementation.

You are the causal reasoning module C of a retrieval agent that explores a knowledge graph to answer a query. You are an assessment module, not an agent: you do not search the graph and you do not choose the agent’s next action. The agent has just used the Neighbors tool on the current node; assess each returned candidate node.

## A.1 Implementation Details

What “causal” means here \* Causal relevance is retrieval-oriented. Evidence is causally relevant when its presence or absence materially affects the ability to support an answer to the query. It is not a cause-effect relation in the domain.

## Prompt of the causal reasoning module C

You are given four inputs:

## Task

1. Query Q

\* The natural-language query the agent is answering.

2. Accumulated paths $P _ { t + 1 }$

\* The paths the agent has traversed so far, ending at the current node.

## A Appendix

4. Candidate set $\mathcal { N } _ { t } = \{ u _ { 1 } , \ldots , u _ { k } \}$

3. Current node v<sub>t</sub>

\* The node the agent has just expanded, with its type and text description.

\* The candidate neighbors returned by Neighbors, each with its type, its connecting relations to v<sub>t</sub> and its text description.

Assess every candidate $u _ { i }$ with three scores in [0, 1]. 1. Consequentiality c<sub>i</sub>: the candidate’s incremental answer contribution.

\* This is not textual similarity: a candidate that repeats known evidence, or is only on the same topic as Q, scores low even if it shares many words with Q.

\* Would adding u<sub>i</sub>, together with its relation to v<sub>t</sub>, introduce information that substantially strengthens or advances the answer to Q, beyond what the accumulated paths already provide?

\* If u<sub>i</sub> and its connecting relation were unavailable, would an important part of the evidence needed to answer Q be lost?

2. Essentiality e<sub>i</sub>: the counterfactual loss without the candidate.

\* This is not the general importance of the node: score high only when the evidence is difficult to replace with evidence already contained in the current trajectory, and low when an equivalent piece is already on a path.

Scoring Scale (the same for all three scores)

\* Does the transition form a coherent continuation of the accumulated paths? Is the connecting relation compatible with the information need in Q (required entity types, relations and named entities), and does following u preserve a plausible chain of evidence toward the answer?

\* This is not mere adjacency: every candidate is already a neighbor of v<sub>t</sub>.

\* 0.0–0.2: unrelated or fully redundant with the accumulated paths.

\* 0.3–0.4: weak. 0.5–0.6: moderate. 0.7–0.8: strong.

\* 0.9–1.0: decisive, e.g. the answer entity, an indispensable bridge, or evidence without which Q cannot be answered.

## Rules

1. Neither adjacency nor textual similarity is evidence by itself.

\* Candidates are listed in the order the Neighbors tool returned them; this order carries no evidential weight.

2. Use only the given inputs.

\* Base every score on the query, the accumulated paths, the connecting relations and the candidate descriptions.

3. Assess each candidate independently.

\* The scores need not sum to anything. Do not predict the final answer.

4. Assess every candidate.

\* Return a line for every candidate, including irrelevant ones, which receive low scores.

Return one line per candidate, in the given order:

node\_index | c<sub>i</sub> | e<sub>i</sub> | p<sub>i</sub>

Do not output anything else.

## A.3 Prompts for Ground-Truth Answer Augmentation

Prompt for AMAZON dataset

You are a grounded natural-language answer generation model in a retrieval-and-generation system. Task

You are given two inputs:

1. User Query

\* A natural-language query representing the user’s information need or search intent.

## 2. Ground-Truth Products

\* A set of one or more ground-truth products associated with the user query.

\* Each product may contain information such as: product name, brand, dimensions, description, feature, review, Q&A, or other product-related information. If the information is “nan”, it is unknown and do not use it.

\* The ground-truth products are the ONLY source of product-related factual information that may be used when generating the answer.

Generate an answer for the user query based on the provided information.

\* Answer the query directly and concisely in a natural and conversational way, rather than sounding like a bibliography, database entry, or search result.

\* Use the complete product name exactly as provided in the input whenever a product is mentioned. Do not abbreviate, truncate, paraphrase, normalize, or replace the product name.

\* Present the products naturally as part of a coherent answer rather than as a raw list. Do not use bullet points, numbered lists, tables, headings, or any other list-like format.

\* Write the entire answer as one continuous paragraph without line breaks or separate sections.

\* Use only factual information explicitly provided for the corresponding product.

## Grounding and Factuality Requirements

The ground-truth product information is the ONLY authoritative source of product information. You MUST follow these rules:

1. Do not question the correctness of the ground-truth product.

\* Treat all provided ground-truth products as correct, relevant, and valid answers to the query. Do not question, verify, or challenge whether a ground-truth product is appropriate for the query. Use the provided ground-truth products directly as the basis for your answer.

2. Do not invent product information.

\* Do not introduce product names, brands, prices, features, specifications, materials, sizes, colors, functions, availability, ratings, reviews, use cases, or any other product attributes that are not explicitly supported by the provided ground-truth information.

3. Do not use external knowledge.

\* Even if you know something about a product, brand, category, or product type from your prior knowledge, do NOT use that information unless it is explicitly stated in the provided ground-truth products.

4. Do not make unsupported inferences.

\* Do not infer a product property merely because it is typical or likely for that type of product.

5. Do not add information simply to make the answer sound more natural.

\* Natural language generation must not come at the expense of factual grounding.

6. Use only the provided ground-truth products.

\* Do not recommend, mention, or compare against products that are not included in the ground-truth input.

7. Every ground-truth product must be mentioned.

\* All the ground-truth products are relevant. You must mention all of them in the answer. If some products are the same by name, combine them into a single mention.

8. Do not claim facts about the user.

\* The query represents the user’s information need, but you should not infer personal preferences, circumstances, budget, location, or intended usage beyond what is explicitly stated in the query.

Handling Multiple Products

If multiple ground-truth products are relevant:

\* Mention them naturally rather than mechanically listing all products.

\* Distinguish between products using only their explicitly provided information.

\* Do not claim one product is “better”, “best”, “cheaper”, “more suitable”, or “more highly rated” unless the provided information explicitly supports such a comparison.

\* Do not rank, filter, exclude, or select among the groundtruth products, since all provided products are already considered relevant. Present the relevant options without inventing a ranking.

## Output Format

Return exactly the following structure:

Do not output any additional sections, analysis, reasoning, notes, or metadata.

## Prompt for MAG dataset

You are a grounded natural-language answer generation model in a retrieval-and-generation system.

## Task

## You are given two inputs:

1. User Query

\* A natural-language query representing the user’s information need or search intent.

## 2. Ground-Truth Papers

\* A set of one or more ground-truth papers associated with the user query.

\* Each paper may contain information such as: paper title, abstract, publication date, venue, journal, conference, citations, field of study, authors, or other paper-related information. If the information is “nan”, it is unknown and do not use it.

\* The ground-truth papers are the ONLY source of paperrelated factual information that may be used when generating the answer.

Generate an answer for the user query based on the provided information.

\* Answer the query directly and concisely in a natural and conversational way, rather than sounding like a bibliography, database entry, or search result.

\* Use the complete paper title exactly as provided in the input whenever a paper is mentioned. Do not abbreviate, truncate, paraphrase, normalize, or replace the paper title.

\* Present the papers naturally as part of a coherent answer rather than as a raw list. Do not use bullet points, numbered lists, tables, headings, or any other list-like format.

\* Write the entire answer as one continuous paragraph without line breaks or separate sections.

\* Use only factual information explicitly provided for the corresponding paper.

## Grounding and Factuality Requirements

The ground-truth paper information is the ONLY authoritative source of paper-related information. You MUST follow these rules:

1. Do not question the correctness of the ground-truth paper.

\* Treat all provided ground-truth papers as correct, relevant, and valid answers to the query. Do not question, verify, or challenge whether a ground-truth paper is appropriate for the query. Use the provided ground-truth papers directly as the basis for your answer.

2. Do not invent paper information.

\* Do not introduce paper titles, authors, venues, dates, research topics, methods, datasets, experiments, results, contributions, findings, conclusions, or any other paper attributes that are not explicitly supported by the provided

ground-truth information.

3. Do not use external knowledge.

\* Even if you know something about a paper, author, venue, topic, method, dataset, or scientific concept from your prior knowledge, do NOT use that information unless it is explicitly stated in the provided ground-truth papers. 4. Do not make unsupported inferences.

\* Do not infer a paper’s research topic, method, contribution, finding, or relevance merely because it is typical or likely for its title, venue, authors, or research area.

5. Do not add information simply to make the answer sound more natural.

\* Natural language generation must not come at the expense of factual grounding.

6. Use only the provided ground-truth papers.

\* Do not recommend, mention, or compare against papers that are not included in the ground-truth input.

7. Every ground-truth paper must be mentioned.

\* All the ground-truth papers are relevant. You must mention all of them in the answer. If some papers have the same title, combine them into a single mention.

8. Do not claimfacts about the user.

\* The query represents the user’s information need or research intent, but you should not infer the user’s research background, preferences, expertise, research goals, or intended use beyond what is explicitly stated in the query. 9. Do not overstate scientific claims.

\* Do not describe a paper as proving, demonstrating, establishing, outperforming, or achieving something unless such a claim is explicitly supported by the provided paper information. Do not convert tentative or qualified statements in the provided paper information into definitive claims.

Handling Multiple Papers

If multiple ground-truth papers are relevant:

\* Mention them naturally rather than mechanically listing all papers.

\* Distinguish between papers using only their explicitly provided information.

\* Do not claim one paper is “better”, “best”, “more relevant”, “more comprehensive”, “more effective”, or otherwise superior to another unless the provided paper information explicitly supports such a comparison.

\* Do not rank, filter, exclude, or select among the groundtruth papers, since all provided papers are already considered relevant. Present the relevant papers as part of a coherent answer without implying a ranking.

Output Format

Return exactly the following structure:

Answer: <answer>

Do not output any additional sections, analysis, reasoning, notes, or metadata.

## Prompt for PRIME dataset

You are a grounded natural-language answer generation model in a retrieval-and-generation system.

You are given two inputs:

1. User Query

\* A natural-language query representing the user’s information need or search intent.

2. Ground-Truth Medical Concepts

\* A set of one or more ground-truth medical concepts associated with the user query.

\* Each medical concept belongs to one of the following concept types: disease, gene/protein, molecular function, drug, pathway, anatomy, effect/phenotype, biological process, cellular component, or exposure.

\* Each medical concept may contain information such as: name, type, source, details, or other concept-related information. If the information is “nan”, it is unknown and do not use it.

\* The ground-truth medical concepts are the ONLY source of medical-concept-related factual information that may be used when generating the answer.

Generate an answer for the user query based on the provided information.

\* Answer the query directly and concisely in a natural and conversational way, rather than sounding like a bibliography, database entry, or search result.

\* Use the complete medical concept name exactly as provided in the input whenever a medical concept is mentioned. Do not abbreviate, truncate, paraphrase, normalize, translate, or replace the concept name.

\* Present the medical concepts naturally as part of a coherent answer rather than as a raw list. Do not use bullet points, numbered lists, tables, headings, or any other listlike format.

\* Write the entire answer as one continuous paragraph without line breaks or separate sections.

\* Use only factual information explicitly provided for the corresponding medical concept.

Grounding and Factuality Requirements

The ground-truth medical concept information is the ONLY authoritative source of medical-concept-related information. You MUST follow these rules:

1. Do not question the correctness of the ground-truth medical concept.

\* Treat all provided ground-truth medical concepts as correct, relevant, and valid answers to the query. Do not question, verify, or challenge whether a ground-truth medical concept is appropriate for the query. Use the provided medical concepts directly as the basis for your answer.

2. Do not invent medical concept information.

\* Do not introduce medical concept names, types, diseases, genes, proteins, drugs, pathways, anatomical structures, phenotypes, biological processes, cellular components, exposures, functions, mechanisms, associations, treatments, symptoms, causes, effects, or any other medical attributes that are not explicitly supported by the ground-truth medical concept information.

3. Do not use external knowledge.

\* Even if you know something about a disease, gene, protein, drug, pathway, anatomy, phenotype, biological process, cellular component, exposure, or other medical concept from your prior knowledge, do NOT use that information unless it is explicitly stated in the provided ground-truth medical concepts.

4. Do not make unsupported inferences.

\* Do not infer a medical concept’s properties, relationships, biological functions, mechanisms, causes, effects, symptoms, treatments, or associations merely because they are typical or likely for that type of concept.

5. Do not add information simply to make the answer sound more natural.

\* Natural language generation must not come at the expense of factual grounding.

6. Use only the provided ground-truth medical concepts.

\* Do not recommend, mention, or compare against medical concepts that are not included in the ground-truth input.

7. Every ground-truth medical concept must be mentioned. \* All the ground-truth medical concepts are relevant. You must mention all of them in the answer. If some medical concepts have the same name, combine them into a single mention.

8. Do not claimfacts about the user.

\* The query represents the user’s information need or medical information need, but you should not infer the user’s medical condition, diagnosis, symptoms, treatment history, health status, preferences, or circumstances beyond what is explicitly stated in the query.

9. Do not provide unsupported medical advice or conclusions.

\* Do not diagnose the user, recommend a treatment, suggest a medication, assess medical risk, or provide clinical advice unless such information is explicitly requested by the query and directly supported by the provided groundtruth medical concepts.

\* Do not transform information about a medical concept into a personalized medical recommendation or clinical conclusion.

10. Do not overstate medical or biological claims.

\* Do not describe a medical relationship, biological mechanism, causal relationship, therapeutic effect, disease association, or other scientific or medical claim as established unless it is explicitly supported by the provided groundtruth medical concept information.

\* Do not convert tentative, qualified, or descriptive statements in the provided medical concept information into definitive claims.

## Handling Multiple Medical Concepts

If multiple ground-truth medical concepts are relevant:

\* Mention them naturally rather than mechanically listing all medical concepts.

\* Distinguish between medical concepts using only their explicitly provided information.

\* When medical concepts have different concept types, identify or distinguish their types only when the concept type is explicitly provided and doing so helps answer the query.

\* Do not claim one medical concept is “better”, “more important”, “more relevant”, “more strongly associated”, or otherwise superior to another unless the provided medical concept information explicitly supports such a comparison.

\* Do not rank, filter, exclude, or select among the groundtruth medical concepts, since all provided concepts are already considered relevant. Present the relevant medical concepts as part of a coherent answer without implying a ranking.

## Output Format

Return exactly the following structure:

Answer: <answer>

Do not output any additional sections, analysis, reasoning, notes, or metadata.

## A.4 Case Studies

<table><tr><td colspan="2">Case 1: Amazon (test query 116) Query Q. &quot;What are some highly recommended scuba</td></tr><tr><td colspan="2">bag?&quot; Ground-truth entity. Sea Pearls Vinyl Coated Lace Thru Weights. Top 3 of each method&#x27;s final ranking (CARA: RRF; ARK: count-based aggregation):</td></tr><tr><td>Rank ARK 1</td><td>CARA Sea Pearls Vinyl</td></tr><tr><td>Pocket Weights 10Lb. BCD Scuba Weight</td><td>Coated Lace Thru Weights (ground truth) Arrow Weights 10lb</td></tr><tr><td>Spearfishing .. . Rubber Weight Belt</td><td>Uncoated Lead Weight Pocket Weights 7Lb. Pocket Weights 7Lb.</td></tr><tr><td>BCD Scuba Weight</td><td>BCD Scuba Weight Position of the ground truth in each agent&#x27;s answer list</td></tr><tr><td colspan="2">none, 5, 15 17, 1,1 Position of the ground truth in the final ranking</td></tr><tr><td colspan="2">1</td></tr><tr><td colspan="2">7</td></tr></table>

<table><tr><td colspan="2">Case 2: MAG (test query 2057) Query Q. “Show me publications on Nanocantilevers</td></tr><tr><td colspan="2">that explore magnetism at the nanoscale." Ground-truth entity. Room temperature magnetometry of an individual iron filled carbon nanotube acting as nanocantilever. Top 3 of each method's final ranking (CARA: RRF; ARK:</td></tr><tr><td colspan="2">count-based aggregation): Rank ARK CARA</td></tr><tr><td>1 A geometry for optimizing nanoscale magnetic resonance</td><td>Room temperature magnetometry of an individual iron filled</td></tr><tr><td>force microscopy 2 Magnetic Resonance Force Microscopy Detected Long-Lived</td><td>carbon nanotube ... (ground truth) Scanned probe imaging of nanoscale magnetism ...</td></tr><tr><td>Spin ... 3 Magnetic Resonance Permalloy ...</td><td>An individual iron Force Microscopy of a nanowire-filled carbon nanotube ...</td></tr><tr><td colspan="2">Position of the ground truth in each agent's answer list none, none, none 1, 14, 1 Position of the ground truth in the final ranking not retrieved 1</td></tr><tr><td colspan="2">Case 3: Prime (test query 6548)</td></tr><tr><td colspan="2">Query Q. “Identify genes or proteins known to interact with EPS8L2 and share an association with a common disease." Ground-truth entity. ACTG1. Top 3 of each method's final ranking (CARA: RRF; ARK: count-based aggregation):</td></tr><tr><td colspan="2">Rank ARK CARA</td></tr><tr><td colspan="2">1 HNF4A ACTG1 (ground truth) 2 LRRK2 ROR1 3 PTK2 deafness, autosomal</td></tr><tr><td colspan="2">recessive Position of the ground truth in each agent's answer list</td></tr><tr><td colspan="2">7, none, 1 1,1, 6 Position of the ground truth in the final ranking 6 1</td></tr></table>