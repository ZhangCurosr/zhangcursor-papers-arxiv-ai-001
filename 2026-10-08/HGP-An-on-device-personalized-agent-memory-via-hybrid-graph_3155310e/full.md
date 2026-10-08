# HGP: An on-device personalized agent memory via hybrid graph storage

Ran Zhou<sup>1</sup>, Xueming Han<sup>2</sup>, Jiaheng Liu<sup>1</sup>, Yuyao Zhang<sup>2</sup>, Fanyu Meng<sup>2</sup>, Junlan Feng<sup>2</sup>, Yuxiang Ren<sup>1†</sup>

<sup>1</sup>School of Intelligence Science and Technology, Nanjing University <sup>2</sup>Jiutian Research Correspondence: renyuxiang@nju.edu.cn

## Abstract

LLM-based agents face challenges in personalized interactive tasks due to heterogeneous, multi-typed, and implicitly constrained longterm traces. Existing memory mechanisms struggle with accurate routing and retrieval, especially on-device where personalization is critical. Most methods use single-vector representations, blurring type distinctions and relational structure. We propose HGP, a hybrid graph memory framework. HGP employs a lightweight self-enhancement classifier for personalized memory routing and constructs episodic, semantic, and procedural memories as graphs. It also extracts working memory as a state trajectory to capture current state and implicit constraints, ensuring reliable decisionmaking. The classifier reduces large-model calls, enabling on-device deployment, while graph storage enables accurate retrieval and incremental user profile refinement. Experiments on two benchmarks show that on PAL-Set solution selection, HGP achieves an S-score of 35.58, nearly 7 points above the strongest baseline. Code and data are at https://github. com/Ouan6/HGP-.git.

## 1 Introduction

LLM(Large Language Models)-based agents are increasingly deployed in personalized interactive tasks, such as long-term conversational companionship and constraint-aware task planning (Yao et al., 2023; Park et al., 2023). These applications require the agent to access user-related information across turns and generate responses that align with a specific user’s preferences and constraints (Zhong et al., 2023; Packer et al., 2024). To compensate for limited context windows and unreliable long-context recall, external memory modules have been widely adopted (Lewis et al., 2021; Zhao et al., 2024; Fan et al., 2024). Most agent memory systems adopt a unified memory pool, compress each memory item into a single-vector representation, retrieve by semantic similarity, and then concatenate retrieved snippets to support generation (Lewis et al., 2021; Zhong et al., 2023). However, such flat memory designs often blur distinctions among memory types and relational structure, which are critical for personalized retrieval and decision making. Robust personalization demands continuously summarizing interaction experience, distilling user preferences and behavioral habits, and reliably activating such signals in subsequent decisions. These capabilities are essential for maintaining consistent user-specific behavior over long interactions (Park et al., 2023; Liu et al., 2023). Moreover, a critical but underexplored bottleneck is personalized memory routing: dynamically assigning incoming interaction traces to the appropriate memory type. In practice, histories contain heterogeneous signals (events, profiles, strategies, and actionable states) (Atkinson and Shiffrin, 1968; Tulving, 1972; Baddeley and Hitch, 1974; Squire, 1992), and inaccurate routing stores information in the wrong memory type, causing cascading retrieval errors (Xu et al., 2025).

Personalized memory framework is needed. We argue that achieving personalization requires a memory system with two key properties: (a) enabling personalized memory routing with low supervision cost; and (b) explicitly modeling heterogeneous memory types and their relations. First, personalized routing is important because memory distributions vary across users, making it difficult to rely solely on a fixed general-purpose classifier. Given the cost of repeatedly invoking LLMs during online interaction, especially in on-device contexts, a lightweight routing mechanism that amortizes supervisory cost over time becomes essential. In our setting, on-device operation refers to latencycritical online routing and retrieval on a local device, while graph construction and classifier updates are performed asynchronously. Second, an explicit modeling of memory heterogeneity is required: heterogeneous memory types must not be collapsed into a uniform vector space, but should instead retain their distinct attributes and interrelations through structured representations to facilitate retrieval.

Presented work. To address these challenges, we propose HGP, a Hybrid Graph Personalized memory framework. Our main contributions are summarized as follows:

• Framework design. We develop a framework that moves beyond single-vector compression by organizing episodic, semantic, and procedural memories with type-specific graph structures, including event graphs, knowledge graphs, and rule trees. This is augmented by dynamic working memory-state trajectory graphs that capture and preserve implicit contextual constraints.

• Self-Enhancement Classifier. We propose a confidence-based LLM-assisted annotation policy for personalized memory routing. A lightweight classifier progressively takes over routing decisions, adapts to user-specific memory distributions, and substantially reduces supervision cost by recycling low-confidence LLM annotations as training data.

• Empirical validation. We evaluate HGP on LoCoMo (Maharana et al., 2024) and PAL-Set (Huang et al., 2025), demonstrating consistent improvements in personalization quality and decision reliability over strong baselines. We also provide a thorough analysis of how working memory improves robustness under implicit constraints.

## 2 Related Work

## 2.1 Long-context Modeling and Retrieval-Augmented Memory.

One research direction improves model capacity by extending context windows and developing memory-optimized attention mechanisms, thereby enabling training and inference over significantly longer sequences. Representative approaches include sparse-attention models such as Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2021), as well as I/O-efficient exact attention methods like FlashAttention (Dao et al.,

2022). Key innovations in positional encoding and length extrapolation, such as RoPE, ALiBi, and their derivatives (Su et al., 2023; Press et al., 2022; Ding et al., 2024), have driven increases in the effective context window of LLM. Representing a complementary strategy to long-context modeling, retrieval augmentation enhances LLMs by coupling them with an external knowledge store, as exemplified by RAG (Lewis et al., 2021), REALM (Guu et al., 2020), and RETRO (Borgeaud et al., 2022). However, increased context capacity does not guarantee effective information utilization. This limitation is evidenced by findings such as “Lost in the Middle” (Liu et al., 2023), which demonstrate that models often access salient information unevenly within long contexts. More critically, even with unbounded context length, both long-context modeling and retrieval augmentation lack the requisite mechanisms for continuously capturing and structuring user-specific interaction patterns, which are essential for genuine personalization.

## 2.2 Graph-structured memory.

Recent memory frameworks are increasingly adopting graph-based representations, driven by the promise of explicit relational reasoning (Sun et al., 2025; Li et al., 2025a). Moving beyond flat similarity search over text chunks, graph-structured memory enables more structured storage and retrieval through explicit modeling of relations among memory units. GraphRAG-style systems (Han et al., 2025; Zhang et al., 2024b,a) index knowledge as graphs and retrieve query-relevant subgraphs to support multi-hop evidence aggregation. Representative approaches include graph-traversal-based retrieval, such as HippoRAG (Gutiérrez et al., 2025), and hierarchical indexing with structure-aware retrieval, exemplified by LeanRAG (Zhang et al., 2025b) and HiRAG (Jiao et al., 2025). Graphstructured memory has also been explored in agentic settings, including G-Memory (Zhang et al., 2025a), G-Refer (Li et al., 2025b), and the concurrent PlugMem (Yang et al., 2026), which organizes agent experiences into structured memory graphs for efficient retrieval. These works are closely related to HGP in moving beyond raw trajectory retrieval and flat memory storage. However, most of them mainly focus on structure-aware retrieval or task-agnostic agent memory, while HGP focuses on personalized agent memory where heterogeneous user traces must be routed into distinct memory types and stored with type-specific graph structures for later personalized retrieval and decision making.

## 2.3 Agent memory system.

Memory is a core capability of LLM-based agents. Early systems mainly rely on passive retrieve-and inject mechanisms such as retrieval-augmented generation (RAG) (Lewis et al., 2021), while agent frameworks like ReAct (Yao et al., 2023) and Toolformer (Schick et al., 2023) treat memory as an implicit or short-term component. To support personalization and long-term consistency, recent work increasingly incorporates cross-session memory. MemoryBank (Zhong et al., 2023) continuously updates persistent user profiles; MemGPT (Packer et al., 2024) introduces a memory hierarchy between the context window and external storage; Generative Agents (Park et al., 2023) and MemoryOS (Kang et al., 2025) further explore structured memory storage and lifecycle management. Typed memory has also been discussed in cognitive agent architectures. For example, CoALA (Sumers et al., 2024) formalizes language agents with multiple memory components, including working, episodic, semantic, and procedural memories. Productionoriented systems like Mem0 (Chhikara et al., 2025) and A-MEM (Xu et al., 2025) target scalable memory pipelines, while recent work ScrapMem (Chang and Ren, 2026) further considers storage-efficient personalized memory on edge devices. These works motivate persistent and cognitively inspired agent memory, whereas HGP studies how user traces can be routed into heterogeneous memory types and stored with type-specific graph structures for personalization. Our self-enhanced router is related to cost-aware model routing and deferral methods, such as FrugalGPT (Chen et al., 2023) and RouteLLM (Ong et al., 2025). Unlike these methods, which route inputs across models, HGP routes memory writes into different memory stores; only low-confidence traces invoke LLM annotation, which is then cached for incremental classifier updates.

## 3 Methodology

In this work, we propose HGP, a hybrid graph memory framework for user personalization. As illustrated in Figure 1, HGP comprises two core components: (1) a self-enhancement classifier; (2) a hybrid graph-based storage.

## 3.1 Personalized Memory Routing

In cross-user scenarios, individual differences in memory-type signals cause substantial distribution shifts, making a general classifier prone to misrouting. To address this, HGP employs a confidencebased, LLM-assisted annotation mechanism that continuously refines the router and amortizes supervision cost. For an input segment $x ,$ the router first predicts a label ${ \hat { c } } _ { \theta } ( x )$ with confidence score $s ( x )$ . The final routing decision is governed by an iteration-adaptive threshold $\tau _ { t }$ :

$$
c ( x ) = \left\{ \begin{array} { l l } { \hat { c } _ { \mathrm { L L M } } ( x ) , } & { s ( x ) < \tau _ { t } , } \\ { \hat { c } _ { \theta } ( x ) , } & { s ( x ) \geq \tau _ { t } , } \end{array} \right.\tag{1}
$$

where ${ \hat { c } } _ { \mathrm { L L M } } ( x )$ denotes the label from the LLMbased router, and $\tau _ { t }$ is the confidence threshold at iteration t. If $s ( x ) \geq \tau _ { t }$ , the predicted label ${ \hat { c } } _ { \theta } ( x )$ is retained for downstream memory construction. Otherwise, the segment is deferred to the LLM for annotation, and the resulting labeled pair $( x , \hat { c } _ { \mathrm { L L M } } ( x ) )$ is cached into an incremental training pool. Periodically, the classifier is updated on the accumulated pool. We initialize the router conservatively (LLM-dominant) and gradually shift the workload to the classifier via exponential annealing:

$$
\tau _ { t } = \operatorname* { m a x } \left( \tau _ { \operatorname* { m i n } } , \tau _ { 0 } \cdot e ^ { - \beta t } \right) .\tag{2}
$$

where a high initial threshold $\tau _ { 0 }$ encourages deferral, a decay rate $\beta$ enables a smooth transition, and a lower bound $\tau _ { \mathrm { m i n } }$ prevents overly aggressive selflabeling. As the data pool grows with user-specific samples, the classifier’s decision boundary progressively adapts to the current domain and user patterns, effectively mitigating cross-user distribution shifts. This continuous adaptation sustains longterm personalization, which is especially critical as user behaviors evolve and memory distributions drift over time.

## 3.2 Hybrid Graph Storage

To preserve heterogeneous memory types and their relations, HGP adopts a hybrid graph storage design. We detail each memory structure below.

## 3.2.1 Episodic Memory – Event Graph

Episodic memory stores situated interaction events experienced, preserving traceable trajectories, temporal order, and contextual dependencies. In HGP, episodic memory is represented as an Event Graph. Formally, the event graph is modeled as

$$
\mathcal G _ { \mathrm { e p i } } = ( V _ { e } \cup V _ { \mathrm { e n t } } , E _ { \mathrm { e p i } } ) ,\tag{3}
$$

![](images/eebadd2e0351da45fa2c41994d5e17558a6c3b93754dee827ad7f6f4764de799.jpg)  
Figure 1: Overview of our HGP framework with two components: (1) a self-enhancement classifier that routes inputs via a classifier/LLM and incrementally updates the classifier using high-confidence samples, and (2) hybrid graph storage that preserves structural properties of heterogeneous memory types.

where $V _ { e }$ is the set of atomic episode nodes, $V _ { \mathrm { e n t } }$ denotes contextual entity nodes, and $E _ { \mathrm { e p i } }$ represents typed edges encoding event–event and event–entity relations.

In the event graph, episode nodes serve as structural anchors, connected to other episodes or contextual entities via semantically typed edges. These edges capture structured dependencies between events, such as temporal order, contextual association, and other semantic relations. Each edge is defined as

$$
e _ { i j } = ( v _ { i } , v _ { j } , r ) , \quad r \in \{ t e m p o r a l , c o n t e x t u a l , \ldots \} .\tag{4}
$$

where r specifies the semantic role of the relation between two nodes. Unlike isolated record storage, the event graph links events through shared entities, preserving structured relationships and contextual continuity.

As a core component of the hybrid graph storage, the event graph further supports structure-aware retrieval and long-term evolution. As event graphs accumulate, recurring relational patterns and substructures can be abstracted as:

$$
\mathcal { K } = \mathcal { A } _ { \mathrm { a b s } } ( \mathcal { G } _ { \mathrm { e p i } } ^ { ( 1 ) } , \dots , \mathcal { G } _ { \mathrm { e p i } } ^ { ( n ) } ) ,\tag{5}
$$

where Aabs(·) denotes a structure abstraction operator instantiated via LLM. Through this process, episodic memory functions not only as a repository of past experiences but also as a structural foundation for deriving higher-level semantic and procedural knowledge.

## 3.2.2 Semantic Memory – Knowledge Graph

Semantic memory captures long-term, decontextualized knowledge such as facts, relations, and user preferences, supporting consistent personalized behavior. In HGP, it is explicitly represented as a knowledge graph:

$$
\mathcal { G } _ { \mathrm { s e m } } = ( V _ { \mathrm { s e m } } , E _ { \mathrm { s e m } } ) ,\tag{6}
$$

where $V _ { \mathrm { s e m } }$ denotes semantic nodes encoding usercentric long-term knowledge, and $E _ { \mathrm { s e m } }$ represents typed semantic relations among them.

Each semantic relation is represented as a triple:

$$
( u , r , v ) , \quad u \in V _ { \mathrm { u s e r } } , \ v \in V _ { \mathrm { e n t } } , \ r \in \mathcal { R } _ { \mathrm { s e m } } ,\tag{7}
$$

where r is a predicate drawn from a predefined semantic relations (e.g., HAS\_SKILL, PREFERS, WORKS\_AS).

Unlike episodic memory, which captures temporally grounded experience fragments, semantic memory focuses on regularities that generalize across interaction contexts. It can be obtained either directly via the self-enhancement classifier or through semantic abstraction over accumulated episodic event graphs:

$$
\mathcal { G } _ { \mathrm { s e m } } = \mathcal { A } _ { \mathrm { s e m } } \big ( \mathcal { G } _ { \mathrm { e p i } } ^ { ( 1 ) } , \dots , \mathcal { G } _ { \mathrm { e p i } } ^ { ( n ) } \big ) .\tag{8}
$$

The detailed design and prompt templates of $\mathcal { A } _ { \mathrm { s e m } }$ are provided in Appendix D.1.

3.2.3 Procedural Memory – Rule Tree Graph Procedural memory stores reusable, experiencedriven behavioral strategies and decision flows, enhancing long-term planning capability. In HGP, procedural memory is structured as a Rule Tree, where nodes represent conditions, actions, or intermediate goals, and edges encode their dependencies. Formally, it is defined as

$$
\mathcal G _ { \mathrm { p r o c } } = ( V _ { c } \cup V _ { a } \cup V _ { g } , E _ { \mathrm { p r o c } } ) ,\tag{9}
$$

where $V _ { c } , V _ { a } , V _ { g }$ denote condition, action, and subgoal nodes, respectively, and $E _ { \mathrm { p r o c } }$ represents the set of edges encoding control or dependency relations among these nodes.

Procedural memory is sourced directly from the self-enhancement classifier or extracted from episodic event graphs. In the latter case, it is constructed via

$$
\mathcal { G } _ { \mathrm { p r o c } } = \mathcal { A } _ { \mathrm { p r o c } } ( \mathcal { G } _ { \mathrm { e p i } } ^ { ( 1 ) } , \dots , \mathcal { G } _ { \mathrm { e p i } } ^ { ( n ) } ) ,\tag{10}
$$

where $A _ { \mathrm { p r o c } } ( \cdot )$ is a procedural-abstraction operator instantiated with an LLM. The detailed design and prompt template of $\scriptstyle A _ { \mathrm { p r o c } }$ are provided in $\mathsf { A p - }$ pendix D.2.

## 3.2.4 Working Memory – State Trajectory Graph

Working memory maintains the states of ongoing interactions or tasks, supplying state information for reasoning and decision-making. In HGP, it is implemented as a State Trajectory Graph, where nodes represent current system or user states and edges encode state transitions. Formally, working memory is defined as

$$
\mathcal { G } _ { \mathrm { w m } } = ( V _ { \mathrm { s t a t e } } , E _ { \mathrm { s t a t e } } ) ,\tag{11}
$$

where $V _ { \mathrm { s t a t e } }$ represents the current states and $E _ { \mathrm { s t a t e } }$ denotes transitions between states.

Unlike episodic and semantic memory, which retain long-term information, working memory maintains short-lived, dynamic context extracted from recent episodic memory and continuously updated during interaction. Its update process can be expressed as

$$
\mathcal { G } _ { \mathrm { w m } } ^ { ( t + 1 ) } = \mathcal { U } \big ( \mathcal { G } _ { \mathrm { w m } } ^ { ( t ) } , \mathcal { G } _ { \mathrm { e p i } } ^ { ( t ) } \big ) ,\tag{12}
$$

where $\mathcal { U } ( \cdot )$ denotes a state update operator. Graphbased modeling of working memory provides an explicit representation of the current state and its evolution, enabling timely reasoning, strategy selection, and action triggering.

## 3.2.5 Query Retrieval Workflow

Upon receiving a user query $q ,$ HGP executes a structured retrieval protocol over its hybrid graph memory $\mathcal { G } = \{ \mathcal { G } _ { \mathrm { e p i } } , \mathcal { G } _ { \mathrm { s e m } } , \mathcal { G } _ { \mathrm { p r o c } } , \mathcal { G } _ { \mathrm { w m } } \}$ . A detailed example of this retrieval workflow is provided in Appendix B. For the query $q ,$ HGP issues parallel queries to each subgraph, each designed to retrieve a particular type of structured evidence:

• Episodic graph $\mathcal { G } _ { \mathrm { e p i } } \colon$ : Retrieves a set of temporally or contextually related past interaction nodes. For instance, in a health management scenario, a query about “evening exercise planning” may retrieve nodes representing “night run with elevated heart rate” and “previous over-exertion alert”, connected via shared activity and time entities.

• Semantic graph $\mathcal { G } _ { \mathrm { s e m } } .$ : Retrieves a subgraph of factual triples about the user that are semantically relevant to q. Examples include (User, allergy, Ibuprofen) and (User, condition, Hypertension), providing stable profile constraints.

• Procedural graph $\mathcal { G } _ { \mathrm { p r o c } } \mathrm { : }$ Retrieves applicable rule nodes that match the current context. For example, the rule “IfNSAID combines with aspirin → check bleeding risk” is retrieved when the query involves taking painkillers after aspirin intake.

• Working memory graph $\mathcal { G } _ { \mathrm { w m } }$ : Retrieves the current state trajectory, capturing immediate session constraints such as “aspirin already taken this morning” and “pending dinner recommendation task”.

The retrieved subgraphs are merged into a unified context and passed, together with the original query q, to the LLM. The model then generates a response grounded in both long-term memory and the user’s current state and implicit constraints, ensuring coherence, personalization, and safety.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate on two long-term memory benchmarks: LoCoMo (Maharana et al., 2024) provides 10 multi-session conversations with ∼200 QA pairs each, covering single-hop, multi-hop, temporal, and open-domain questions. Following Mem0 (Chhikara et al., 2025), we exclude the adversarial category due to missing ground-truth answers. PAL-Set (Huang et al., 2025) evaluates memory-aware personalization via user logs and interaction histories. We focus on the Solution Proposal tasks: (1)QA: memory-grounded answer generation; (2)Selection: selecting the best response from candidates based on retrieved memories.

Table 1: Unified evaluation results on LoCoMo (%). Evaluation metrics are reported for each category and overall, including F1 score (F1), BLEU-1 (B1), Exact Match score (EM) and LLM-as-a-Judge score (J), with higher values indicating better performance. Best results are in bold, and second-best results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="4">Single-hop</td><td colspan="4">Multi-hop</td><td colspan="4">Temporal</td><td colspan="4">Open-domain</td><td colspan="4">Overall</td></tr><tr><td>J</td><td>F1</td><td>B1</td><td>EM |</td><td>J</td><td>F1</td><td>B1</td><td>EM</td><td>J</td><td>F1</td><td>B1</td><td>EM</td><td>J</td><td>F1</td><td>B1</td><td>EM</td><td>J</td><td>F1</td><td>B1</td><td>EM</td></tr><tr><td>TurnRAG</td><td>||47.52</td><td>34.31</td><td>28.43</td><td>6.03</td><td>38.54</td><td>20.33</td><td>16.60</td><td>6.25</td><td>60.44</td><td>51.71</td><td>47.86</td><td>19.94</td><td>65.28</td><td>48.34</td><td>43.76</td><td>24.97</td><td>59.35</td><td>44.73</td><td>40.11</td><td>19.29</td></tr><tr><td>LangMem</td><td>53.55</td><td>32.78</td><td>26.35</td><td>3.90</td><td>41.67</td><td>22.97</td><td>19.99</td><td>10.42</td><td>17.76</td><td>10.23</td><td>8.67</td><td>3.74</td><td>54.22</td><td>34.30</td><td>30.64</td><td>14.10</td><td>45.65</td><td>24.74</td><td>21.49</td><td>8.46</td></tr><tr><td>MemoryBank</td><td>47.87</td><td>19.26</td><td>24.29</td><td>3.55</td><td>37.50</td><td>10.89</td><td>15.00</td><td>7.29</td><td>26.17</td><td>6.63</td><td>7.51</td><td>4.05</td><td>72.65</td><td>40.97</td><td>44.37</td><td>26.27</td><td>56.12</td><td>39.83</td><td>35.03</td><td>16.22</td></tr><tr><td>A-Mem</td><td>51.42</td><td>20.43</td><td>25.29</td><td>4.96</td><td>44.79</td><td>13.60</td><td>17.00</td><td>7.29</td><td>59.19</td><td>35.21</td><td>38.63</td><td>19.31</td><td>67.59</td><td>38.42</td><td>41.09</td><td>24.82</td><td>61.41</td><td>32.87</td><td>36.15</td><td>18.90</td></tr><tr><td>HippoRAG2</td><td>52.84</td><td>21.16</td><td>27.69</td><td>4.61</td><td>40.62</td><td>13.25</td><td>17.08</td><td>8.33</td><td>60.75</td><td>38.00</td><td>41.75</td><td>19.94</td><td>68.31</td><td>39.75</td><td>42.84</td><td>26.39</td><td>62.13</td><td>34.29</td><td>38.20</td><td>19.88</td></tr><tr><td>Mem0</td><td>52.13</td><td>35.18</td><td>26.15</td><td>5.67</td><td>38.54</td><td>28.95</td><td>22.85</td><td>13.54</td><td>59.81</td><td>52.71</td><td>39.70</td><td>19.31</td><td>66.39</td><td>47.98</td><td>40.81</td><td>26.14</td><td>60.63</td><td>45.42</td><td>36.75</td><td>20.14</td></tr><tr><td>Ours</td><td>56.03</td><td>34.44</td><td>30.39</td><td>5.67</td><td>46.88</td><td>27.69</td><td>23.70</td><td>12.50</td><td>60.75</td><td>50.31</td><td>46.24</td><td>19.63</td><td>71.45</td><td>51.33</td><td>46.59</td><td>26.87</td><td>65.08</td><td>45.07</td><td>42.09</td><td>20.54</td></tr></table>

Metrics. On LoCoMo, we report F1, Exact Match (EM), BLEU-1 (B1), and supplement with Mem0- style LLM-as-a-judge(J) to overcome the limits of lexical overlap in long-context reasoning. On PAL-Set, for the QA subtask, we use BLEU-1/2/3/4 as automatic metrics and report an LLM-as-a-judge score as a complementary assessment of answer correctness and usefulness. For comparability, the same judging protocol is applied to both memorybased and non-memory systems. For Selection, we use the benchmark-defined S-score, which measures how well the selected solution matches the user’s needs. The LLM-as-a-judge score is treated as supplementary evidence rather than the sole evaluation criterion; our conclusions are based jointly on automatic QA metrics, the PAL-Set S-score, ablation results, and long-horizon memory evaluation. Full QA judge prompts are provided in Appendix A.1.

Baselines. We compare against two paradigms:

(1) Non-memory methods. On LoCoMo, we use a standard RAG pipeline. On PAL-Set, following the official benchmark, we include four retrieval-based variants with different context granularities: Vanilla (w/o log) (no history), Vanilla (with log) (logs concatenated into the prompt), Turn-level RAG, and Session-level RAG.

(2) Memory-based frameworks. We compare with representative memory-augmented systems: Mem0 (Chhikara et al., 2025), Memory-Bank (Zhong et al., 2023), A-Mem (Xu et al., 2025), HippoRAG2 (Gutiérrez et al., 2025), and additionally LangMem on LoCoMo.

Implementation Details. In HGP, memory routing is performed by a lightweight BERT-based classifier (roberta\_L-4\_H-512). The confidence threshold is initialized at $\tau _ { 0 } = 0 . 9 0$ , annealed with rate $\beta \ : = \ : 0 . 0 5$ , and lower-bounded by $\tau _ { \mathrm { m i n } } = 0 . 5 0$ Since working-memory annotations are not available in LoCoMo or PAL-Set, our experiments focus on episodic, semantic, and procedural memories, stored as Neo4j graph databases. Memory extraction uses deepseek-chat; response generation and LLM-as-a-judge use gpt-4o-mini (temperature 0.0). On LoCoMo, max output length is 100 tokens; dialogue history for TurnRAG is truncated to 4K tokens; retrieval-based methods use K = 30 items. All methods share the same answer/judge prompts adapted from Mem0 for fairness. On PAL-Set, we follow the official setup with K = 3 and introduce an additional QA-specific judge prompt for semantic correctness beyond lexical overlap.

## 4.2 Main Results

LoCoMo performance. Experiments on the Lo-CoMo dataset demonstrate the strong memory recall capability of HGP. As shown in Table 1, HGP achieves the best overall performance in Judge (65.08), BLEU-1 (42.09), and EM (20.54), while maintaining competitive F1 (45.07)—outperforming Mem0 by +4.45 (Judge) and +5.34 (BLEU-1). Gains are most pronounced on Open-domain tasks, where HGP attains the highest F1 (51.33), BLEU-1 (46.59), and EM (26.87). This improvement supports the effectiveness of graph-based memory storage: structuring episodic events, semantic profiles, and procedural rules as interconnected graphs provides more reliable evidence for diverse, unconstrained questions. Temporal questions remain comparatively challenging, suggesting opportunities for enhancing temporal ordering.

PAL-Set Solution Proposal tasks. Table 2 reports results on Solution QA and Selection. On QA, HGP achieves the highest LLM-judge score (7.46) and the best BLEU-2/3/4 (8.67/4.70/2.78), indicating stronger multi-step coherence. On Selection, non-memory baselines improve only marginally over Vanilla (w/o log) (+4.66 to +4.90), confirming that raw log retrieval is insufficient. HGP attains an S-score of 35.58, surpassing Vanilla (w/o log) (20.34) by +15.24 and the top memory baseline HippoRAG2 (29.22) by +6.36. These results show that graph-structured memory enables more precise personalized retrieval, better aligning responses with user-specific needs.

Table 2: Performance comparison across different methods on PAL-Set Solution QA tasks. B-1 to B-4 represent BLEU scores. ∆ (vs. Vanilla w/o log) reports the absolute improvement in S-score compared to the Vanilla (w/o log) baseline. LLM-judge denotes the score evaluated by GPT-4o-mini.
<table><tr><td>Group</td><td>Method</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>QA Judge</td><td>S-score</td><td>∆ (vs. Vanilla w/o log)</td></tr><tr><td rowspan="4">Non-memory</td><td>Vanilla (w/o log)</td><td>19.68</td><td>7.72</td><td>3.80</td><td>2.11</td><td>7.31</td><td>20.34</td><td>+0.00</td></tr><tr><td>Vanilla (with log)</td><td>18.96</td><td>7.51</td><td>3.82</td><td>2.22</td><td>7.32</td><td>25.00</td><td>+4.66</td></tr><tr><td>Turn-level RAG</td><td>19.48</td><td>7.72</td><td>3.93</td><td>2.23</td><td>7.41</td><td>25.06</td><td>+4.72</td></tr><tr><td>Session-level RAG</td><td>19.17</td><td>7.55</td><td>3.82</td><td>2.22</td><td>7.30</td><td>25.24</td><td>+4.90</td></tr><tr><td rowspan="5">Memory-based</td><td>MemoryBank</td><td>20.83</td><td>7.36</td><td>4.23</td><td>2.56</td><td>7.13</td><td>28.85</td><td>+8.51</td></tr><tr><td>Mem0</td><td>19.20</td><td>7.66</td><td>3.82</td><td>2.12</td><td>7.34</td><td>26.50</td><td>+6.16</td></tr><tr><td>A-Mem</td><td>19.33</td><td>8.12</td><td>3.96</td><td>2.17</td><td>7.04</td><td>27.27</td><td>+6.93</td></tr><tr><td>HippoRAG2</td><td>18.84</td><td>8.06</td><td>4.30</td><td>2.34</td><td>7.31</td><td>29.22</td><td>+8.88</td></tr><tr><td>HGP (Ours)</td><td>19.62</td><td>8.67</td><td>4.70</td><td>2.78</td><td>7.46</td><td>35.58</td><td>+15.24</td></tr></table>

![](images/87f9833256f3cc0c8a2f6aa5f31167a3553e861bf523ae34ed23dce5c5e7a8f7.jpg)  
Figure 2: Working Memory Analysis: working-memory state tracking in a personal healthcare scenario. Yellow highlights denote pending to-do reminders, and blue highlights indicate preference-aligned personalized suggestions. Compared with baselines, our method surfaces the working-memory state (pre-meal medication: pending) and reminds the user while providing personalized dinner recommendations.

Table 3: Ablation study on memory components and representation designs. The table reports generation quality (BLEU and LLM-judge) and selection performance (S-score). Higher is better.
<table><tr><td>Method</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>LLM-judge</td><td>S-score</td></tr><tr><td>Unified-vector</td><td>18.15</td><td>7.24</td><td>3.60</td><td>2.05</td><td>7.18</td><td>25.15</td></tr><tr><td>Typed-vector</td><td>18.61</td><td>7.51</td><td>3.74</td><td>2.09</td><td>7.29</td><td>33.66</td></tr><tr><td>w/o E-Mem</td><td>19.53</td><td>7.96</td><td>3.96</td><td>2.23</td><td>7.09</td><td>28.83</td></tr><tr><td>w/o S-Mem</td><td>19.69</td><td>8.30</td><td>4.38</td><td>2.57</td><td>7.23</td><td>25.71</td></tr><tr><td>w/o P-Mem</td><td>19.59</td><td>8.63</td><td>4.66</td><td>2.76</td><td>7.31</td><td>33.10</td></tr><tr><td>Full (Ours)</td><td>19.62</td><td>8.67</td><td>4.70</td><td>2.78</td><td>7.46</td><td>35.58</td></tr></table>

## 4.3 Ablation Study

To quantify the contribution of each memory component, We ablate each memory type while keeping the rest of the pipeline unchanged (Table 3). Graph and type structure. Unified-vector collapses all memories into a single vector space, while Typed-vector preserves memory types but removes graph structure. Typed-vector substantially improves over Unified-vector on Selection (33.66 vs. 25.15), showing that memory typing is important for personalized decision making. Full HGP further improves the S-score to 35.58, indicating that type-specific graph storage provides additional gains beyond typed vector retrieval.

Episodic Memory(E-Mem). Removing episodic memory causes the largest drop on Solution QA: BLEU-1/2/3/4 fall from 19.66/8.67/4.70/2.78 to 19.53/7.96/3.96/2.23, and the LLM-judge score declines from 7.46 to 7.09. On Selection, the S-score drops from 35.58 to 28.83 (∆ = −6.75). These results confirm that episodic memory, as the foundational trace of user interactions, is essential for both precise answer generation and context-aware solution matching.

Semantic Memory(S-Mem). Removing semantic memory causes the largest drop on Selection (∆ = −9.87,S-score; Table 3), confirming that stable user profiles (long-term preferences, attributes, health facts) are essential for personalized decisionmaking. On Solution QA, the LLM-judge score also declines (7.46 to 7.23), indicating its contribution to coherent memory recall beyond pure retrieval.

Procedural Memory(P-Mem). Removing procedural memory causes a modest but non-negligible drop in Selection $( \Delta = - 2 . 4 8 )$ , while the impact on Solution QA is minor (judge score 7.46 → 7.31, BLEU-4 2.78 → 2.76). This aligns with its role as high-level strategic guidance rather than direct factual evidence.

Overall, Table 3 highlights complementary roles: semantic graphs drive personalized selection, episodic graphs supply fine-grained event evidence for QA, and procedural graphs enhance strategy reuse. Integrating all components yields the best performance, validating our hybrid graph memory design.

## 4.4 Working Memory Analysis

Figure 2 illustrates the role of working memory in HGP using a healthcare scenario. Working memory explicitly extracts and maintains the current actionable state, enabling the agent to prioritize immediate actions during response generation.

![](images/88b0d3bddc6fbea3d035e0b775d7239113b476333d1d8d868a34a2ecd7d8f161.jpg)  
Figure 3: Classifier routing statistics across iterations. (a) Routing mix and adaptive threshold. (b) Cumulative workload handled by the LLM and the classifier.

Setup. We compare three systems: (i) a vanilla LLM without memory, (ii) an LLM augmented with Mem0, and (iii) HGP with working memory enabled. The dialog history contains a pre-meal medication reminder. The user query asks for dinner recommendations without explicitly mentioning medication.

Observation. The vanilla LLM directly suggests dinner options. Mem0 generates preferencealigned recommendations but fails to remind the user about the medication. In contrast, HGP retrieves the state “pre-meal medication: unfinished” from the working memory graph and injects it into the prompt, enabling the LLM to proactively issue a medication reminder before dinner and demonstrating state awareness under implicit constraints. (Further details in Appendix C.)

Table 4 compares our self-enhancement classifier with an offline classifier, a fixed LLM router, and a zero-shot LLM router. Our classifier achieves the highest macro-F1 (0.9339) and micro-F1 (0.9105) with an average latency of 4.98 ms, roughly 360× faster than the fixed LLM router. Fig. 3 shows the training dynamics: the confidencebased gating progressively shifts routing decisions from the LLM to the classifier, and incremental training every 200 samples drives the system from LLM-dominant to classifier-dominant (Fig. 3a). Overall, LLM calls drop from 2400 to 807 (a 66.4% reduction), confirming the workload transfer visible in the cumulative curves (Fig. 3b). These results demonstrate that the classifier effectively captures LLM routing knowledge while substantially lowering inference costs for on-device deployment. Hyperparameter sensitivity and per-stage performance are detailed in Appendix A.2.

Table 4: Performance comparison of the selfenhancement classifier against other baselines. Best results are in bold, and second-best results are underlined.
<table><tr><td>Router</td><td>Micro-F1</td><td>Macro-F1</td><td>Avg Latency (ms)</td></tr><tr><td>Offline Classifier</td><td>0.8638</td><td>0.8790</td><td>4.06</td></tr><tr><td>Fixed LLM Router</td><td>0.8953</td><td>0.9224</td><td>1803.69</td></tr><tr><td>Zero-shot LLM Router</td><td>0.8286</td><td>0.8436</td><td>1572.64</td></tr><tr><td>Ours</td><td>0.9105</td><td>0.9339</td><td>4.98</td></tr></table>

## 5 Conclusion

In this paper, we address the key challenge of achieving robust and adaptive personalization in LLM-based agents under cross-user heterogeneity and distribution shift. To this end, we propose HGP, a memory-augmented framework that integrates a self-enhancement routing mechanism with a hybrid graph-based memory storage, enabling costefficient and structured personalization. Extensive experiments on LoCoMo and PAL-Set validate the effectiveness of HGP in improving memory-recall accuracy and personalization capability, while routing statistics confirm its ability to amortize supervision cost over time. Future work will explore finer-grained modeling of memory graphs, incorporate cross-memory consistency constraints, and extend the framework to a broader range of interactive tasks and tool-augmented agents.

## 6 Limitations

While HGP demonstrates improvements in personalized memory management, several limitations remain. First, although working memory shows promising qualitative results, existing benchmarks lack explicit working-memory annotations, preventing systematic quantitative evaluation. Constructing datasets with ground-truth state trajectories is an important direction for rigorous assessment. Second, graph construction and maintenance introduce additional offline computation and storage costs, while our on-device claim primarily concerns latency-critical online routing and retrieval. A more complete end-to-end cost analysis remains for future work. In addition, this work focuses mainly on graph storage, while finer-grained retrieval and pruning mechanisms require further study. Finally, we use LLM-as-a-judge evaluation to complement lexical-overlap metrics. Although standardized prompts and temperature 0.0 improve evaluation consistency, LLM judges may still exhibit biases, and their reliability warrants continued attention.

## Acknowledgments

This work was supported by the Nanjing University-China Mobile Joint Research Institute.

## References

Richard C Atkinson and Richard M Shiffrin. 1968. Human memory: A proposed system and its control processes. In Kenneth W Spence and Janet Taylor Spence, editors, The Psychology of Learning and Motivation: Advances in Research and Theory, volume 2, pages 89–195. Academic Press, New York.

Alan D Baddeley and Graham J Hitch. 1974. Working memory. In Gordon H Bower, editor, The Psychology of Learning and Motivation: Advances in Research and Theory, volume 8, pages 47–89. Academic Press, New York.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. Preprint, arXiv:2004.05150.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, and 9 others. 2022. Improving language models by retrieving from trillions of tokens. Preprint, arXiv:2112.04426.

Jiale Chang and Yuxiang Ren. 2026. Scrapmem: A bio-inspired framework for on-device personalized agent memory via optical forgetting. Preprint, arXiv:2605.03804.

Lingjiao Chen, Matei Zaharia, and James Zou. 2023. Frugalgpt: How to use large language models while reducing cost and improving performance. Preprint, arXiv:2305.05176.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. Preprint, arXiv:2504.19413.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Preprint, arXiv:2205.14135.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. 2024. Longrope: Extending llm context window beyond 2 million tokens. Preprint, arXiv:2402.13753.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards retrieval-augmented large language models. Preprint, arXiv:2405.06211.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2025. Hipporag: Neurobiologically inspired long-term memory for large language models. Preprint, arXiv:2405.14831.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Realm: Retrievalaugmented language model pre-training. Preprint, arXiv:2002.08909.

Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and Jiliang Tang. 2025. Retrieval-augmented generation with graphs (graphrag). Preprint, arXiv:2501.00309.

Zhaopei Huang, Qifeng Dai, Guozheng Wu, Xiaopeng Wu, Kehan Chen, Chuan Yu, Xubin Li, Tiezheng Ge, Wenxuan Wang, and Qin Jin. 2025. Mem-pal: Towards memory-based personalized dialogue assistants for long-term user-agent interaction. Preprint, arXiv:2511.13410.

YiHan Jiao, ZheHao Tan, Dan Yang, DuoLin Sun, Jie Feng, Yue Shen, Jian Wang, and Peng Wei. 2025. Hirag: Hierarchical-thought instruction-tuning retrievalaugmented generation. Preprint, arXiv:2507.05714.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory os of ai agent. Preprint, arXiv:2506.06326.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks. Preprint, arXiv:2005.11401.

Yuhan Li, Yuyao Wang, Jianheng Tang, Heng Chang, Yuxiang Ren, and Jia Li. 2025a. Advancing graph foundation models: A data-centric perspective. KDD ’25, page 1635–1646, New York, NY, USA. Association for Computing Machinery.

Yuhan Li, Xinni Zhang, Linhao Luo, Heng Chang, Yuxiang Ren, Irwin King, and Jia Li. 2025b. Grefer: Graph retrieval-augmented large language model for explainable recommendation. Preprint, arXiv:2502.12586.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. Preprint, arXiv:2307.03172.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. Preprint, arXiv:2402.17753.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed

Kadous, and Ion Stoica. 2025. Routellm: Learning to route llms with preference data. Preprint, arXiv:2406.18665.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Memgpt: Towards llms as operating systems. Preprint, arXiv:2310.08560.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. Preprint, arXiv:2304.03442.

Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. Preprint, arXiv:2108.12409.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Preprint, arXiv:2302.04761.

Larry R Squire. 1992. Declarative and nondeclarative memory: Multiple brain systems supporting learning and memory. Journal of Cognitive Neuroscience, 4(3):232–243.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2023. Roformer: Enhanced transformer with rotary position embedding. Preprint, arXiv:2104.09864.

Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. 2024. Cognitive architectures for language agents. Preprint, arXiv:2309.02427.

Shengyin Sun, Yuxiang Ren, Jiehao Chen, and Chen Ma. 2025. Large language models as topological structure enhancers for text-attributed graphs. Preprint, arXiv:2311.14324.

Endel Tulving. 1972. Episodic and semantic memory. In Endel Tulving and Wayne Donaldson, editors, Organization of Memory, pages 381–403. Academic Press, New York.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. Preprint, arXiv:2502.12110.

Ke Yang, Zixi Chen, Xuan He, Jize Jiang, Michel Galley, Chenglong Wang, Jianfeng Gao, Jiawei Han, and ChengXiang Zhai. 2026. Plugmem: A task-agnostic plugin memory module for llm agents. Preprint, arXiv:2603.03296.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. Preprint, arXiv:2210.03629.

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. 2021. Big bird: Transformers for longer sequences. Preprint, arXiv:2007.14062.

Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, and Shuicheng Yan. 2025a. G-memory: Tracing hierarchical memory for multi-agent systems. Preprint, arXiv:2506.07398.

Wanhao Zhang, Qianli Zhang, Enyu Yu, Yuxiang Ren, Yeqing Meng, Mingxi Qiu, and Jilong Wang. 2024a. Leveraging rag-enhanced large language model for semi-supervised log anomaly detection. In 2024 IEEE 35th International Symposium on Software Reliability Engineering (ISSRE), pages 168–179.

Wanhao Zhang, Qianli Zhang, Enyu Yu, Yuxiang Ren, Yeqing Meng, Mingxi Qiu, and Jilong Wang. 2024b. Lograg: Semi-supervised log-based anomaly detection with retrieval-augmented generation. In 2024 IEEE International Conference on Web Services (ICWS), pages 1100–1102.

Yaoze Zhang, Rong Wu, Pinlong Cai, Xiaoman Wang, Guohang Yan, Song Mao, Ding Wang, and Botian Shi. 2025b. Leanrag: Knowledge-graph-based generation with semantic aggregation and hierarchical retrieval. Preprint, arXiv:2508.10391.

Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024. Retrieval-augmented generation for ai-generated content: A survey. Preprint, arXiv:2402.19473.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2023. Memorybank: Enhancing large language models with long-term memory. Preprint, arXiv:2305.10250.

## A More Experiment Details

## A.1 Prompts for PAL-Set Solution QA

We provide the full LLM-as-a-Judge prompt template for the PAL-Set Solution QA task in Figure 4. This template is used to compute the LLM-judge scores reported in Table 2. During evaluation, we supply the judge LLM with the model-generated solution, the task question, and relevant conversation context (when available), along with the corresponding positive and negative reference solutions from the dataset. The judge is instructed to score outputs across multiple dimensions: requirement satisfaction, semantic correctness, step completeness, and user-level personalization. Positive and negative references are provided to calibrate judgments and penalize common error patterns. For ease of aggregation, the judge is constrained to produce strictly formatted JSON outputs, with temperature fixed at 0.0.

## A.2 Supplementary Experiments on the Self-Enhancement Classifier

## A.2.1 Hyperparameter Sensitivity Analysis

To evaluate the robustness of our self-enhancement classifier under different hyperparameter settings, we conducted a full-factor grid search over the three key parameters: initial threshold $\tau _ { 0 } ~ \in$ $\{ 0 . 8 , 0 . 9 , 0 . 9 5 \}$ , decay rate $\beta \in \{ 0 . 0 2 , 0 . 0 5 , 0 . 1 \}$ and lower bound $\tau _ { \mathrm { m i n } } \in \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ . This resulted in 27 configurations. For each configuration, we measured the macro F1 score (with standard deviation) and the average inference latency on the same validation set. To comprehensively evaluate each hyperparameter configuration, we define a combined score that balances accuracy, stability, and efficiency. Let $\mu _ { \mathrm { l a t } } ( c )$ denote the average inference latency of configuration c, and let $\mu _ { \mathrm { l a t } } ^ { \mathrm { m i n } }$ be the minimum average latency among all candidate configurations:

$$
\mu _ { \mathrm { l a t } } ^ { \mathrm { m i n } } = \operatorname* { m i n } _ { c } \mu _ { \mathrm { l a t } } ( c ) .\tag{13}
$$

The normalized latency is then defined as:

$$
\mathrm { L a t e n c y N o r m } ( c ) = \frac { \mu _ { \mathrm { l a t } } ( c ) } { \mu _ { \mathrm { l a t } } ^ { \mathrm { m i n } } } .\tag{14}
$$

With penalty coefficients $\lambda _ { s } = 0 . 5$ and $\lambda _ { l } = 0 . 5 .$ the overall score for configuration c is:

$$
\begin{array} { c } { { \mathrm { S c o r e } ( c ) = \mu _ { \mathrm { F 1 } } ( c ) - \lambda _ { s } \cdot \sigma _ { \mathrm { F 1 } } ( c ) - } } \\ { { \lambda _ { l } \cdot \mathrm { m a x } \bigl ( 0 , \mathrm { L a t e n c y N o r m } ( c ) \bigr ) } } \end{array}\tag{15}
$$

where $\mu _ { \mathrm { F } 1 } ( c )$ and $\sigma _ { \mathrm { F 1 } } ( c )$ are the mean and standard deviation of the macro F1 score, respectively. This score penalizes both performance variability (σ<sub>F1</sub>) and any latency exceeding the optimal value $( \mathrm { L a t e n c y N o r m } ( c ) - 1 )$ , thereby encouraging configurations that are accurate, stable, and efficient.

As shown in the table 5, the macro F1 values across all 27 configurations vary within a very small range, demonstrating that the classifier is remarkably insensitive to the choice of these hyperparameters. The best macro F1 (0.9528) is achieved by configuration #21 $( \tau _ { 0 } = 0 . 9 5 , \beta =$ $0 . 0 2 , \tau _ { \mathrm { m i n } } = 0 . 3 )$ , but its latency (5.93 ms) is significantly higher than the optimum. The configuration with the lowest latency (4.23 ms) is #1 $( \tau _ { 0 } ~ = ~ 0 . 9 , \beta ~ = ~ 0 . 0 2 , \tau _ { \mathrm { m i n } } ~ = ~ 0 . 5 )$ , which also attains a macro F1 of 0.9490.

Our default configuration #5 $( \tau _ { 0 } ~ = ~ 0 . 9 , \beta$ $0 . 0 5 , \tau _ { \mathrm { m i n } } = 0 . 5 )$ yields a macro F1 of 0.9474 and a latency of 4.44 ms, which is only marginally worse than the best trade-off. The differences from the top-scoring configuration (#1) are minimal: macro-F1 differs by only 0.0016 (0.9474 vs. 0.9490) and latency by 0.21 ms (4.44 vs.4.23). Both configurations lie on the same near-Pareto frontier. We choose #5 over #1 for the following reasons:

PROMPT TEMPLATE FOR SOLUTION QA RESULTS GENERATIOM (PAL-SET)   
You are a strict, impartial, and reproducible Solution QA evaluator (LLM Judge).   
Your task is to evaluate only the relevance and fit of the generated solution to the complete user requirement   
Assign an integer score from 1 to 10.   
Evaluate only the final solution text. Do not evaluate the method that produced it, the retrieval process, the memory system, or the storage   
structure. Logs and Memories are provided only as user context for checking whether the solution satisfies the user's needs, preferences, and   
constraints.   
#Evaluation objectives:   
1) Primary: Does the solution accurately satisfy the requirement, including explicit needs, implicit constraints, and current context?   
2) Secondary: Is the solution consistent with the provided user context, including relevant preferences, background information, historical   
constraints, and prohibitions?   
3) Tertiary: Does the solution avoid error patterns in the negative references, such as being off-topic, misleading, violating preferences, or ignoring   
key constraints?   
#Important rules:   
Do not give extra credit merely because the solution explicitly mentions memory, history, logs, or retrieved memories.   
Do not evaluate whether the solution uses any specific retrieval or memory process.   
If Logs/Memories are empty or insufficient, do not penalize solely for the lack of memory information. Judge based on the requirement, available   
context, and final solution.   
If Logs/Memories contain relevant user preferences or constraints and the solution clearly ignores, conflicts with, or fabricates them, deduct points   
If the solution does not explicitly cite Logs/Memories but correctly satisfies the relevant preferences and constraints, do not penalize it.   
#Scoring anchors:   
9-10: Highly satisfies the requirement; covers key points completely; is consistent with relevant user context; no major issues.   
7-8: Generally satisfies the requirement; mostly consistent with relevant context and preferences; minor omissions only.   
5-6: Partially satisfies the requirement; has noticeable omissions, generic content, or insufficient use of relevant context; still somewhat usable.   
3-4: Largely fails or deviates from the topic; ignores key context, preferences, or constraints; may be misleading.   
1-2: Severely off-topic, obviously wrong, strongly matches negative error patterns, or may cause clear negative impact.   
Please evaluate the relevance/fit of the following AI-generated solution (1-10 points)   
[User Requirement]   
{requirement}   
[Current Conversation Context Logs]   
{logs}   
[Provided User Context / Memories]   
{memories}   
[Generated Solution]   
{solution}   
[Positive Reference: high-scoring direction/key points]   
{references\_pos}   
[Negative Reference: error patterns to avoid]   
{references\_neg}   
Return strict JSON only:   
{{"relevance": <integer from 1 to 10>, "explanation": "<one or two concise sentences>"}}   
The first character must be "{{" and the last character must be "}}".  
Figure 4: LLM-as-a-Judge prompt used for PAL-Set Solution QA evaluation, enforcing requirement-fit scoring with implicit constraints, retrieved memories/logs, and pos/neg reference calibration.

• Adaptive threshold updates: The moderate decay rate (β = 0.05) triggers more effective threshold updates during online selfenhancement, whereas a smaller decay (β = 0.02) would keep the threshold nearly unchanged, leading to under-adaptation.

• Consistency with our experimental protocol: This setting avoids post-hoc overfitting to a single best point and maintains consistency with the rest of our experiments.

• Robustness: The performance gap between #5 and the best configuration is negligible, and our main conclusions remain unchanged regardless of which near-optimal configuration is chosen.

Therefore, we adopt $\tau _ { 0 } = 0 . 9 , \beta = 0 . 0 5 , \tau _ { \mathrm { m i n } } =$ 0.5 as the default setting for all main experiments. The full ranking of 27 configurations is provided to demonstrate that our results are insensitive to the specific choice among near-optimal settings.

Table 5: Hyperparameter sensitivity analysis of the router with different values of $\tau _ { 0 } , \beta ,$ and $\tau _ { \mathrm { m i n } }$ . The configuration selected for our experiments is highlighted in bold.
<table><tr><td>Rank</td><td> $\tau _ { 0 }$ </td><td> $\beta$ </td><td> $\tau _ { \mathrm { m i n } }$ </td><td>F1 (mean±std)</td><td>Latency (ms)</td><td>Norm Latency</td><td>Score</td></tr><tr><td>1</td><td>0.9</td><td>0.02</td><td>0.5</td><td> $0 . 9 4 9 0 \pm 0 . 0 0 3 4$ </td><td>4.23</td><td>1.000</td><td>0.947</td></tr><tr><td>2</td><td>0.9</td><td>0.02</td><td>0.3</td><td>0.9490 ± 0.0034</td><td>4.36</td><td>1.031</td><td>0.932</td></tr><tr><td>3</td><td>0.9</td><td>0.02</td><td>0.7</td><td> $0 . 9 4 9 0 \pm 0 . 0 0 3 4$ </td><td>4.40</td><td>1.040</td><td>0.927</td></tr><tr><td>4</td><td>0.8</td><td>0.1</td><td>0.7</td><td> $0 . 9 4 8 2 \pm 0 . 0 0 2 3$ </td><td>4.46</td><td>1.055</td><td>0.919</td></tr><tr><td>5</td><td>0.9</td><td>0.05</td><td>0.5</td><td>0.9474 ± 0.0060</td><td>4.44</td><td>1.050</td><td>0.919</td></tr><tr><td>6</td><td>0.8</td><td>0.02</td><td>0.5</td><td>0.9482 ± 0.0023</td><td>4.72</td><td>1.118</td><td>0.888</td></tr><tr><td>7</td><td>0.9</td><td>0.05</td><td>0.3</td><td>0.9474 ± 0.0060</td><td>4.83</td><td>1.143</td><td>0.873</td></tr><tr><td>8</td><td>0.95</td><td>0.1</td><td>0.5</td><td>0.9526 ± 0.0001</td><td>4.96</td><td>1.174</td><td>0.866</td></tr><tr><td>9</td><td>0.9</td><td>0.1</td><td>0.5</td><td>0.9474 ± 0.0060</td><td>5.06</td><td>1.198</td><td>0.846</td></tr><tr><td>10</td><td>0.8</td><td>0.05</td><td>0.3</td><td>0.9482 ± 0.0023</td><td>5.10</td><td>1.207</td><td>0.843</td></tr><tr><td>11</td><td>0.95</td><td>0.05</td><td>0.5</td><td>0.9522 ± 0.0049</td><td>5.13</td><td>1.213</td><td>0.843</td></tr><tr><td>12</td><td>0.8</td><td>0.05</td><td>0.5</td><td>0.9482 ± 0.0023</td><td>5.22</td><td>1.235</td><td>0.830</td></tr><tr><td>13</td><td>0.9</td><td>0.1</td><td>0.3</td><td>0.9474 ± 0.0060</td><td>5.26</td><td>1.244</td><td>0.822</td></tr><tr><td>14</td><td>0.95</td><td>0.1</td><td>0.7</td><td>0.9526 ± 0.0001</td><td>5.42</td><td>1.281</td><td>0.812</td></tr><tr><td>15</td><td>0.9</td><td>0.05</td><td>0.7</td><td>0.9474 ± 0.0060</td><td>5.40</td><td>1.277</td><td>0.806</td></tr><tr><td>16</td><td>0.8</td><td>0.05</td><td>0.7</td><td>0.9482 ± 0.0023</td><td>5.47</td><td>1.295</td><td>0.800</td></tr><tr><td>17</td><td>0.95</td><td>0.02</td><td>0.7</td><td>0.9509 ± 0.0030</td><td>5.52</td><td>1.307</td><td>0.796</td></tr><tr><td>18</td><td>0.95</td><td>0.05</td><td>0.7</td><td>0.9522 ± 0.0049</td><td>5.64</td><td>1.335</td><td>0.782</td></tr><tr><td>19</td><td>0.8</td><td>0.02</td><td>0.7</td><td>0.9482 ± 0.0023</td><td>5.63</td><td>1.332</td><td>0.781</td></tr><tr><td>20</td><td>0.8</td><td>0.1</td><td>0.5</td><td>0.9482 ± 0.0023</td><td>5.80</td><td>1.371</td><td>0.761</td></tr><tr><td>21</td><td>0.95</td><td>0.02</td><td>0.3</td><td>0.9528 ± 0.0009</td><td>5.93</td><td>1.402</td><td>0.751</td></tr><tr><td>22</td><td>0.95</td><td>0.02</td><td>0.5</td><td>0.9509 ± 0.0030</td><td>5.94</td><td>1.405</td><td>0.747</td></tr><tr><td>23</td><td>0.95</td><td>0.1</td><td>0.3</td><td>0.9526 ± 0.0001</td><td>6.05</td><td>1.431</td><td>0.737</td></tr><tr><td>24</td><td>0.8</td><td>0.02</td><td>0.3</td><td> $0 . 9 4 8 2 \pm 0 . 0 0 2 3$ </td><td>6.01</td><td>1.421</td><td>0.737</td></tr><tr><td>25</td><td>0.9</td><td>0.1</td><td>0.7</td><td> $0 . 9 4 7 4 \pm 0 . 0 0 6 0$ </td><td>6.05</td><td>1.431</td><td>0.729</td></tr><tr><td>26</td><td>0.8</td><td>0.1</td><td>0.3</td><td> $0 . 9 4 8 2 \pm 0 . 0 0 2 3$ </td><td>6.24</td><td>1.477</td><td>0.709</td></tr><tr><td>27</td><td>0.95</td><td>0.05</td><td>0.3</td><td> $0 . 9 5 2 2 \pm 0 . 0 0 4 9$ </td><td>6.44</td><td>1.525</td><td>0.687</td></tr></table>

## A.2.2 Stage-wise Performance of the Self-Enhancement Classifier

Table 6: Performance evolution of the self-enhancement classifier across incremental training stages.
<table><tr><td>Stage</td><td>Macro-F1</td><td>Micro-F1</td><td>∆Macro (vs. S0)</td></tr><tr><td>SO (offline init)</td><td>0.8980</td><td>0.8859</td><td>0.0000</td></tr><tr><td>S1</td><td>0.9162</td><td>0.9068</td><td>+0.0182</td></tr><tr><td>S2</td><td>0.9282</td><td>0.9192</td><td>+0.0302</td></tr><tr><td>S3</td><td>0.9445</td><td>0.9275</td><td>+0.0465</td></tr><tr><td>S4</td><td>0.9527</td><td>0.9379</td><td>+0.0547</td></tr><tr><td>S5</td><td>0.9554</td><td>0.9416</td><td>+0.0574</td></tr></table>

To investigate how the self-enhancement mechanism improves classifier performance over time, we record its performance after each incremental training round. The initial stage (S0) uses an offlinetrained classifier without any LLM supervision. Subsequently, every 200 newly LLM-labeled samples trigger one round of incremental fine-tuning, resulting in five rounds (S1 to S5). As shown in Table 6, the macro F1 steadily increases from 0.8980 (S0) to 0.9554 (S5), an absolute gain of 5.74 percentage points. The micro F1 follows a similar upward trend. The most significant improvements occur in the first two rounds (S0→S1: +1.82 points; S1→ S2: +1.20 points), after which the gains gradually diminish, indicating convergence. These results demonstrate that the self-enhancement mechanism effectively leverages LLM-labeled hard samples to continuously improve routing accuracy without causing performance oscillation or catastrophic forgetting.

![](images/68983ef7d26998722dead20da75aa5f18b102c6fb39d31a1152bf095ca1cc2cf.jpg)  
Figure 5: An illustrative example of HGP in a personalized health management scenario. Given a user’s current query and long-term interaction history, HGP organizes heterogeneous memories, including episodic, semantic, procedural, and working memory, within a hybrid graph storage, and retrieves task-relevant evidence to generate personalized health recommendations.

It is important to emphasize the training protocol for this stage-wise evaluation. The classifier first makes predictions on the training set. Samples with low confidence are sent to an LLM for re-labeling, and the (sample, label) pairs are cached. Once a sufficient number of such samples is accumulated, incremental training is performed, and the cycle repeats. The training and test sets are strictly separated. The test set is never used for training, nor does it ever involve LLM inference. For rigorous evaluation, the test set is further partitioned into multiple disjoint subsets, each used exactly once for evaluating the classifier after one incremental training round. This prevents any data leakage across evaluation steps.

## B Retrieval Workflow Example

As shown in Figure 5, upon receiving a user query q , HGP executes a structured retrieval protocol over its hybrid graph memory $\mathcal { G } ^ { \mathrm { ~ ~ } } =$ $\{ \mathcal { G } _ { \mathrm { e p i } } , \mathcal { G } _ { \mathrm { s e m } } , \mathcal { G } _ { \mathrm { p r o c } } , \mathcal { G } _ { \mathrm { w m } } \}$ Each sub-graph is queried for a specific type of evidence. The retrieved sub-graphs are merged into a unified context and passed, together with q, to the LLM. The model then generates a response that is aware of both long-term user history and the current implicit constraints, thereby ensuring personalization, safety, and coherence.

## C Details of special case

We construct a personalized health scenario where the dialogue history includes a to-do item: take pre-meal medication before dinner. During evaluation, the user only requests dinner recommendations without explicitly mentioning medication. This case tests whether the agent can adhere to implicit constraints inferred from prior interactions and prioritize actions that are required but not explicitly stated in the query.

Figure 6 illustrates the full pipeline for the case study in Section 4.3. During retrieval, HGP queries heterogeneous external memories separately: (i) the event graph for diet- and symptom-related logs (episodic memory); (ii) the knowledge graph for user preferences and profile facts (semantic memory); and (iii) the rule tree for reusable strategies that previously received positive feedback (procedural memory). Together, these long-term memories supply user-relevant evidence for personalized dinner suggestions (highlighted in blue).

Crucially, HGP also queries the workingmemory states from the state trajectory graph to retrieve the current actionable status. In this case, working memory returns “pre-meal medication: incomplete” and injects it as an actionable constraint into the prompt. Then the LLM reminds the user to take the medication before offeringrecommendations (highlighted in yellow).This behavior is difficult to trigger reliably using only episodic, semantic, or procedural retrieval, as it hinges on shorthorizon execution state rather than long-term facts, preferences, or strategies.

![](images/1531b260bcca15cc78412d612e5e7048271e6cc2b760ebd3c5a9d26b068e4ea0.jpg)  
Figure 6: Special case study: illustrating the detailed process by which an LLM retrieves user-relevant memories via HGP in the special case. Working memory addresses implicit constraints in a personalized health assistant scenario; its effect is highlighted in yellow (pre-meal medication: pending), while personalized recommendation cues are highlighted in blue.

Overall, this case demonstrates HGP ’s ability to deliver personalized, context-aware responses. It also shows that long-term memory retrieval alone cannot guarantee compliance with implicit, actionable constraints. By explicitly tracking short-term states via working memory, HGP enhances behavioral consistency and safety in long-horizon interactions.

## D Abstraction Operators for Semantic and Procedural Memory from Episodic Memory (LLM Prompts)

To construct semantic and procedural memory from raw episodic traces, we design two abstraction operators, $\mathcal { A } _ { \mathrm { s e m } }$ and $\scriptstyle A _ { \mathrm { p r o c } } .$ , each implemented via a dedicated LLM prompt template. These operators extract structured knowledge from episodic event graphs and convert them into reusable memory formats.

## D.1 Semantic Abstraction Operator $\mathcal { A } _ { \mathrm { s e m } }$

The semantic abstraction operator $\mathcal { A } _ { \mathrm { s e m } }$ extracts static factual knowledge from episodic interactions to construct user profiles, including user attributes, preferences, skills, and long-term constraints. As shown in Figure $7 , A _ { \mathrm { s e m } }$ employs a structured LLM prompt to identify and summarize semantic triples in the form of a knowledge graph, given episodic memories.

## D.2 Procedural Abstraction Operator $\mathcal { A } _ { \mathrm { p r o c } }$

The procedural abstraction operator $\mathcal { A } _ { \mathrm { p r o c } }$ extracts reusable rules, workflows, and decision strategies from episodic traces. As illustrated in Figure 8, $\mathcal { A } _ { \mathrm { p r o c } }$ uses a structured LLM prompt that requires the model to output each rule in a unified JSON format, containing fields such as title, trigger conditions, execution steps, tags, and a tree-structured workflow that supports sequential and branching logic. If no clear workflow is present, the operator returns an empty list. The extracted rules are then stored in the procedural memory graph $\mathcal { G } _ { \mathrm { p r o c } }$

![](images/558c226933d179b0bb7cad9603e268df624b78d2c1660936e4291596b215f895.jpg)  
Figure 7: Illustration of the procedural abstraction operator $\mathcal { A } _ { \mathrm { s e m } }$

![](images/861271e21d0996868d2d68a7df8a220d2d7e5e75ab496a7f3dfe4dad56627dae.jpg)  
Figure 8: Illustration of the procedural abstraction operator $\mathcal { A } _ { \mathrm { p r o c } }$