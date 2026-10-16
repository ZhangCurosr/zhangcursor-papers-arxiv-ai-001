# ALL-IN-ONE MEMORY: PROCESS-REWARDED REIN-FORCEMENT LEARNING FOR MEMORY AGENTS

Bingbing Wang<sup>1,2∗</sup>, Jingjie Lin<sup>1∗</sup>,Weiming Qiao<sup>3</sup>, Jing Li<sup>2</sup>, Ruifeng Xu<sup>1,4†</sup>

<sup>1</sup> Harbin Institute of Technology, Shenzhen, <sup>2</sup> The Hong Kong Polytechnic University,

<sup>3</sup> Fudan University , <sup>4</sup> Peng Cheng Laboratory, bingbing.wang@stu.hit.edu.cn

## ABSTRACT

Large language models increasingly rely on external memory to answer queries grounded in long-horizon histories such as multi-session dialogues and lengthy documents. However, existing memory methods often optimize retrieval and memory construction separately, without a shared reward space for evaluating search, writing, and stopping decisions. To address this, we propose All-in-One Memory (AiO-Mem), which organizes these operations as transitions over a shared evidence state and directly evaluates the semantic quality of each action within a unified reward space. AiO-Mem compiles heterogeneous histories into source-traceable multi-view events, providing a shared substrate for retrieval and memory writing. A unified memory policy actor then rolls out complete interaction trajectories over this shared evidence state. A separate trajectoryconditioned process reward model is trained through evidence-grounded supervised fine-tuning and trajectory-level DPO to assign process scores to the collected trajectories. These scores are aggregated into an action-balanced trajectory reward and used by Process-Aligned DAPO (PA-DAPO) to optimize the memory policy. Across long-context memory benchmarks, AiO-Mem consistently improves answer accuracy over both static memory systems and RL-trained memory agents.

## 1 INTRODUCTION

Large language models (LLMs) have achieved remarkable progress across diverse natural language processing benchmarks (OpenAI, 2023; Touvron et al., 2023). Yet in long-context, multi-step reasoning scenarios, their static parametric knowledge and limited context windows remain critical bottlenecks (Liu et al., 2024). External memory-augmented architectures have therefore become important extensions of LLMs, ranging from dense retrieval systems (Lewis et al., 2020; Guu et al., 2020) and knowledge-graph memories (Edge et al., 2024) to persistent memory agents (Packer et al., 2023; Chhikara et al., 2025). Such systems rely on tightly coupled memory retrieval and memory construction: retrieval supplies relevant evidence, while construction converts it into compact working memory. Coordinating these stages is challenging because the evolving memory state continually changes what evidence should be searched for, written, or discarded.

Recent studies have introduced reinforcement learning (RL) into memory systems to enable selfimproving memory behavior. As illustrated in Figure 1, existing RL-driven memory systems are typically optimized along two separate routes: retrieval-oriented methods, such as Search-R1 (Jin et al., 2025), learn how to search for evidence, while construction-oriented methods, such as Mem-α (Wang et al., 2025), learn when to insert, update, or merge memory entries. However, these methods still optimize retrieval and construction in isolation, leaving the two stages governed by separate objectives and evaluated through partially disconnected feedback signals. This separation exposes two key limitations. First, there is no shared evidence-state space in which retrieved evidence, written memories, and stopping decisions can be jointly evaluated, making it difficult to coordinate whether the system should continue searching, write a memory, skip redundant information, or stop. Second, existing supervision is largely outcome-centered: it assigns rewards to the final answer but provides little process-level credit for the intermediate memory operations that produced it. As a result, trajectories with the same final-answer score may still differ substantially in evidence relevance, memory faithfulness, redundancy control, and stopping sufficiency; answer correctness alone cannot reliably evaluate the semantic quality of memory operations.

![](images/6c8fa636e8a4fbe5b0d39d5a714bb0e59066da58b987ce1a773908cb87ff6c74.jpg)  
Figure 1: Comparison of existing RL-driven memory methods and AiO-Mem.

To overcome these limitations, we propose All-in-One Memory (AiO-Mem), which unifies memory retrieval and construction as evidence-state transitions of a single memory-interaction agent, and uses a separate trajectory-conditioned reward model to evaluate the resulting trajectories. AiO-Mem first compiles heterogeneous histories into a traceable multi-view event bank, providing a shared source-grounded substrate for retrieval and memory writing. Built on this event bank, the mem ory policy actor operates over a shared evidence state and rolls out complete interaction trajectories through two sequential decision steps: decide, which determines whether to continue searching or stop, and extract, which determines whether retrieved candidates should be written into working memory or discarded. A unified process reward model is then trained through evidence-grounded SFT and trajectory-level DPO to assign process scores to each action in the collected trajectories, using rule-derived supervision and reasoning-model scores as complementary training signals. Finally, these process scores are aggregated into an action-balanced trajectory reward. Building on DAPO (Yu et al., 2026b), we propose Process-Aligned DAPO (PA-DAPO) to optimize the memory policy actor. We summarize our primary contributions as follows:

• We propose AiO-Mem, an all-in-one memory agent that unifies retrieval, writing, filtering, and termination within a single evidence-state transition loop. This formulation turns memory retrieval and construction from separately optimized modules into one jointly trainable policy over source-traceable evidence.

• We introduce a trajectory-conditioned unified reward model that directly assesses the semantic quality of each memory action within a shared evidence state, scoring search, add, skip, and stop in a common reward space. Action-balanced aggregation then converts these scores into a trajectory reward for memory-policy optimization with our proposed PA-DAPO.

• We conduct extensive experiments on long-context memory benchmarks, where AiO-Mem consistently improves both answer accuracy and reasoning faithfulness over static memory systems and RL-trained memory agents.

## 2 RELATED WORK

Memory-Augmented Large Language Models. External memory equips LLMs with nonparametric knowledge beyond their context windows. Early retrieval-augmented models, such as RAG (Lewis et al., 2020) and REALM (Guu et al., 2020) established dense retrieval as a standard way to ground generation in external evidence. Later systems enrich retrieval with structured or taskspecific memory organization: GraphRAG (Edge et al., 2024), HippoRAG (Gutierrez et al., 2024),´ KG-RAG (Sanmartin, 2024), and Emotional RAG (Huang et al., 2024) incorporate entity graphs, graph traversal, knowledge-graph grounding, or affective similarity. A complementary line studies persistent memory across interactions. MemGPT (Packer et al., 2023) introduces virtual memory tiers; MemoryBank (Zhong et al., 2024) and Mem0 (Chhikara et al., 2025) maintain user-centric summaries; MemoryOS (Kang et al., 2025) and LightMem (Fang et al., 2026) organize memories into tiered or compressed stores; and A-Mem (Xu et al., 2026), Zep (Rasmussen et al., 2025), EverMemOS (Hu et al., 2026a), LangMem (LangChain, 2025), and GAM (Yan et al., 2025) explore atomic notes, MemCell structures, online consolidation, and iterative evidence gathering. These systems provide increasingly rich retrieval and construction mechanisms, but most coordinate memory operations through predefined workflows, hand-designed heuristics, or modular pipelines rather than a learned policy that jointly optimizes retrieval, writing, filtering, and termination.

![](images/198708819b6145a01cc9444954a17732a092180a2d7427aa18c03f03c9c9ab8a.jpg)  
Figure 2: Overview of our AiO-Mem method.

Reinforcement Learning for Memory Policies. Recent work has applied RL to memory systems at different stages of the pipeline. Retrieval-oriented methods, including Search-R1 (Jin et al., 2025), R1-Searcher (Song et al., 2025), and DeepRetrieval (Jiang et al., 2025b), learn search or retrieva strategies that improve downstream answers. Persistent-memory methods, such as Memory-R1 (Yan et al., 2026), Mem-α (Wang et al., 2025), and AtomMem (Huo et al., 2026), optimize memory insertion, update, merging, or atomic memory management. More recent studies, including Fine-Mem (Ma et al., 2026), TRUSTMEM (Yang et al., 2026), and Mem-T (Yue et al., 2026)), move toward process-level supervision for memory use. Despite this progress, most RL-based memory methods either rely primarily on final-answer correctness or provide process feedback within a restricted stage of the memory pipeline. As a result, they do not directly evaluate heterogeneous memory actions within a shared evidence state and a common reward space. AiO-Mem addresses this gap by learning a trajectory-conditioned reward model that assigns process-level semantic scores to different memory actions and uses them to optimize a unified memory policy.

## 3 METHOD

## 3.1 PROBLEM FORMULATION AND METHOD OVERVIEW

Problem Formulation. We study memory-augmented question answering (QA) over long-horizon histories. Let q be a query and $\mathcal { H } = ( h _ { 1 } , \ldots , h _ { N } )$ be its associated history of N heterogeneous source units, such as dialogue sessions, document passages, or facts. Given (q, H), a memory policy constructs a compact working memory W from which a frozen downstream generator $\hat { y } = \mathcal G ( \boldsymbol q , \mathcal W )$ The learning objective is to acquire an optimal sequence of memory operations that transforms a dispersed history into a reasoning-sufficient evidence state.

Method Overview. As shown in Figure 2, we propose AiO-Mem, a four-component framework that trains a memory policy actor π<sub>θ</sub> to construct working memory through sequential evidence-state transitions. AiO-Mem integrates four components for memory construction, interaction modeling, process supervision, and policy optimization: (i) Traceable Multi-View Event Bank (Section 3.2), which compiles H into source-traceable event units and provides the retrieval substrate for memory interaction; (ii) Evidence-State Transition Modeling (Section 3.3), which formulates the interaction of π as a finite-horizon MDP over a shared evidence-state space, with four action types organized into a decide step (search or stop) and, after each search, an extract step (add or skip); (iii) Unified Reward Modeling (Section 3.4), which trains a trajectory-conditioned reward model via evidencegrounded SFT and trajectory-level DPO to provide dense per-step process supervision; and (iv) Process-Aligned Policy Optimization (Section 3.5), which aggregates action-typed rewards into a balanced trajectory signal and optimizes $\pi _ { \theta }$ through PA-DAPO.

## 3.2 TRACEABLE MULTI-VIEW EVENT BANK

The source units in H often have incompatible granularities: multi-session dialogues are organized as turns, documents may spread a coherent events across multiple paragraphs, and database records are already atomic. Direct fixed-size chunking can therefore break event boundaries and detach memory entries from their source contexts. To preserve event integrity and source traceability, we apply a structure-aware compiler $\mathcal { C }$ to transform the history into a traceable event bank $\begin{array} { r } { B = \mathcal { C } ( \mathcal { \dot { H } } ) = } \end{array}$ $\{ e _ { m } \} _ { m = 1 } ^ { M }$ Each event is defined as $e _ { m } ~ = ~ ( \{ x _ { m } ^ { v } \} _ { v \in \mathcal { V } } , \ell _ { m } )$ , where $\gamma ~ = ~ \{ \mathrm { r a w }$ , sum, anc, phr} denotes four complementary retrieval views and $\ell _ { m } = ( n , i _ { \mathrm { s t a r t } } , i _ { \mathrm { e n d } } )$ is a provenance locator that anchors the event to its source unit $h _ { n }$ . Specifically, The compiler first partitions $\mathcal { H }$ along topic-shift boundaries, then constructs four views for each event. The raw view $x _ { m } ^ { \mathrm { r a w } }$ preserves lexical details, summary $x _ { m } ^ { \mathrm { s u m } }$ captures macroscopic semantic coherence, query-anchor $x _ { m } ^ { \mathrm { a n c } }$ encodes hypothetical queries the event can answer, and event-phrase $x _ { m } ^ { \mathrm { p h r } }$ provides a dense, low-noise semantic label.

Since no single representation captures lexical, semantic, and query-intent cues simultaneously, for each search query $u _ { t }$ issued by the $\pi _ { \theta } .$ , we fuse view-specific rankings via equal-weight Reciprocal Rank Fusion (Cormack et al., 2009):

$$
\operatorname { s c o r e } ( e _ { m } \mid u _ { t } ) = \sum _ { v \in \mathcal { V } } { \frac { 1 } { \kappa + \operatorname { r a n k } _ { v } ( e _ { m } \mid u _ { t } ) } } ,
$$

where $\kappa > 0$ is a smoothing constant. Dense retrieval is used for $x ^ { \mathrm { s u m } }$ and $x ^ { \mathrm { p h r } }$ , BM25 for $x ^ { \mathrm { a n c } }$ , and a hybrid of both for $x ^ { \mathrm { { r a w } } }$ . The retrieval operator returns the top- $K _ { \mathrm { r e t } }$ events as candidate list $\mathcal { R } ( u _ { t } )$ with each retaining its locator $\ell _ { m }$ for source-verifiable memory writing and reward evaluation.

## 3.3 EVIDENCE-STATE TRANSITION MODELING

A single retrieval pass is often insufficient when the answer depends on evidence scattered across the history. We therefore formulate memory interaction as a finite-horizon Markov Decision Process (MDP) over evidence states. The memory policy actor $\pi _ { \theta }$ sequentially selects memory operations and the process is specified by a state space ${ \mathcal { S } } _ { : }$ , an action space ${ \mathcal { A } } .$ , a deterministic transition function ${ \mathcal F } ,$ and a maximum horizon $\dot { T } _ { \mathrm { m a x } }$

At step $t \in \{ 1 , \ldots , T \}$ , the evidence state is ${ z } _ { t } = ( q , \mathcal { W } _ { t } , \mathcal { R } _ { t } ) \in \mathcal { S } _ { : }$ , where $q$ is the query, ${ \mathcal { W } } _ { t }$ is the current working memory, $\mathcal { R } _ { t }$ is the latest candidate list, and $m _ { t } \in$ {decide, extract} indicates the current decision mode. The memory policy actor samples an action $a _ { t } \sim \pi _ { \theta } ( \cdot \mid z _ { t } ) $ under a mode-specific action mask, and $c _ { t } = \mathrm { t y p e } ( a _ { t } )$ denotes its action type. In the decide mode, the policy chooses either search $\left( { { u } _ { t } } \right)$ , which issues a search query $u _ { t }$ to the event bank and retrieves $\mathcal { R } ( u _ { t } )$ , or stop, which terminates the episode. After a search, the process enters the extract mode, where the policy chooses either $\mathbf { a d d } ( n _ { t } )$ , which appends a source-bound note $n _ { t }$ to the working memory, or skip, which discards the current candidates. This constrained structure prevents invalid operations, such as adding a note before retrieval or issuing a new search before deciding whether to use the current candidates. The four action types {search, add, skip, stop} induce the following deterministic transitions:

$$
z _ { t + 1 } = \mathcal { F } ( z _ { t } , a _ { t } ; \mathcal { B } ) = \left\{ \begin{array} { l l } { ( q , \mathcal { W } _ { t } , \mathcal { R } ( u _ { t } ) ) , } & { c _ { t } = \mathrm { s e a r c h } } \\ { ( q , \mathcal { W } _ { t } \oplus [ n _ { t } ] , \mathcal { R } _ { t } ) , } & { c _ { t } = \mathrm { a d d } } \\ { ( q , \mathcal { W } _ { t } , ( ) ) , } & { c _ { t } = \mathrm { s k i p } } \\ { ( q , \mathcal { W } _ { t } , \mathcal { R } _ { t } ) , } & { c _ { t } = \mathrm { s t o p } } \end{array} \right.\tag{1}
$$

The episode ends when the memory policy actor selects stop, or when the maximum horizon $T _ { \mathrm { m a x } }$ is reached. The final working memory is passed to the frozen generator to produce $\hat { y } = \mathcal G ( q , \mathcal W _ { T } )$ Each interaction trajectory is written as $\tau = ( z _ { 1 } , a _ { 1 } , \dots , z _ { T } , a _ { T } , z _ { T + 1 } )$ , and the collected trajecto ries form $\mathcal { D } _ { \mathrm { t r a j } }$ , which is used for subsequent reward modeling and policy optimization.

## 3.4 UNIFIED REWARD MODELING

Final-answer correctness provides only terminal feedback and can not identify which memory operations make a trajectory useful. We therefore train a generative reward model $\mathcal { I } _ { \phi }$ to assign dense per-step process rewards in complete memory trajectories. Unlike the memory policy actor, $\mathcal { I } _ { \phi }$ evaluates the resulting actions and provides process supervision without affecting state transitions.

Stage 1: Evidence-grounded SFT. Let ${ \mathcal { E } } ^ { \star }$ denote the gold evidence spans and Src(·) the provenance projection that maps an event or note to its referenced raw source spans. At step t, the uncovered gold evidence is $\mathcal { E } _ { t } ^ { \mathrm { m i s s } } \dot { = } \mathcal { E } ^ { \star } \backslash \operatorname { S r c } ( \mathcal { W } _ { t } )$ . For a search action, the newly exposed gold evidence is $\Delta _ { t } ^ { \mathrm { r } \bar { \mathrm { e t } } } =$ $\mathcal { E } _ { t } ^ { \mathrm { m i s s } } \cap \mathrm { S r c } ( \mathbf { \dot { \mathcal { R } } } ( u _ { t } ) )$ , and for add action, the newly commits gold evidence is $\Delta _ { t } ^ { \mathrm { w r i t e } } = \mathcal { E } _ { t } ^ { \mathrm { m i s s } } \cap$ $\mathrm { S r c } ( n _ { t } )$ . We use action-specific deterministic rules over these evidence-grounded information to compute a step-level rule score $r _ { t } ^ { \mathrm { r u l e } }$ for each action type $c \in { \mathcal { A } }$ . For each trajectory τ produced by the memory policy actor, we serialize the trajectory as $\dot { x } _ { \tau } = \mathrm { S e r i a l i z e } ( y ^ { \star } , \mathcal { E } ^ { \star } , \tau )$ , and fine-tune $\dot { \mathcal { T } _ { \phi } }$ on the rule reward sequence $\mathbf { v } _ { b } ^ { \mathrm { r u l e } } = \{ ( c _ { t } , r _ { t } ^ { \mathrm { r u l e } } ) \}$ :

$$
{ \mathcal { L } } _ { \mathrm { S F T } } ( \phi ) = - { \frac { 1 } { \sum _ { b } L _ { b } ^ { \mathrm { R M } } } } \sum _ { b } \sum _ { t } \log p _ { \phi } { \bigl ( } v _ { b , t } ^ { \mathrm { r u l e } } \mid x _ { b } , \mathbf { v } _ { b , < t } ^ { \mathrm { r u l e } } { \bigr ) }\tag{2}
$$

where $L _ { b } ^ { \mathrm { R M } }$ the target token length, and $p _ { \phi }$ the autoregressive output distribution of $\mathcal { I } _ { \phi }$

Stage 2: Trajectory-level DPO alignment. Rule-derived scores provide action-level supervision based on evidence coverage and provenance, but do not directly assess the semantic quality of an action in its current evidence state. We therefore provide a reasoning-capable model with the full trajectory context $x _ { \tau }$ and have it independently score each step via chain-of-thought reasoning, producing $\mathbf { v } _ { b } ^ { \mathrm { C o T } }$ . We treat $\mathbf { v } _ { b } ^ { + } = \mathbf { v } _ { b } ^ { \mathrm { C o T } }$ as chosen and $\mathbf { v } _ { b } ^ { - } = \mathbf { v } _ { b } ^ { \mathrm { r u l e } }$ as rejected, forming preference dataset $\mathcal { D } _ { \mathrm { p r e f } } = \{ ( x _ { b } , \mathbf { v } _ { b } ^ { + } , \mathbf { v } _ { b } ^ { - } ) \}$ , and minimize the DPO loss with frozen reference $\phi _ { \mathrm { r e f } }$ and strength $\beta > 0 !$

$$
\mathcal { L } _ { \mathrm { D P O } } ( \phi ) = - \mathbb { E } \biggl [ \log \mathrm { s i g m o i d } \biggl ( \beta \log \frac { p _ { \phi } ( \mathbf { v } ^ { + } \mid x ) } { p _ { \phi _ { \mathrm { r e f } } } ( \mathbf { v } ^ { + } \mid x ) } - \beta \log \frac { p _ { \phi } ( \mathbf { v } ^ { - } \mid x ) } { p _ { \phi _ { \mathrm { r e f } } } ( \mathbf { v } ^ { - } \mid x ) } \biggr ) \biggr ]\tag{3}
$$

After training, $\mathcal { I } _ { \phi }$ directly outputs $\{ ( c _ { t } , \hat { r } _ { t } ) \} _ { t = 1 } ^ { T }$ , where $\hat { r } _ { t } ~ \in \mathbb { R }$ assesses the semantic quality of action $c _ { t }$ in its current evidence state.

## 3.5 PROCESS-ALIGNED MEMORY POLICY OPTIMIZATION

AiO-Mem unifies heterogeneous memory action types within a shared evidence-state transition process. We aggregate their step-level rewards into an action-balanced trajectory signal, averaging within action types before integrating across types. For each category $c \in A .$ , let $\begin{array} { r } { \mathcal { T } _ { c } ( \tau ) = \mathbf { \bar { \{ } } } \mathbf { \bar { { t } } } \mathbf { \bar { { : } } }  \end{array}$ $c _ { t } = c \}$ be the steps where c occurs, $\begin{array} { r } { \bar { \boldsymbol { r } } _ { c } ( \tau ) = | \bar { \mathcal { T } } _ { c } ( \tau ) | ^ { - 1 } { \sum } _ { t \in { \mathcal { T } } _ { c } ( \tau ) } \hat { \boldsymbol { r } } _ { t } } \end{array}$ the within-category mean, and $g _ { c } ( r ) = ( r - r _ { c } ^ { \mathrm { l o } } ) / ( r _ { c } ^ { \mathrm { h i } } - r _ { c } ^ { \mathrm { l o } } )$ a normalization to $[ 0 , 1 ]$ using the rubric bounds $r _ { c } ^ { \mathrm { l o } }$ and $r _ { c } ^ { \mathrm { h i } }$ . Denoting by $\mathcal { A } ( \tau ) = \{ c \in \mathcal { A } : \mathbb { Z } _ { c } ( \tau ) \neq \emptyset \}$ the categories present in $\tau _ { \ast }$ , the unified trajectory reward is:

$$
R _ { \mathrm { { A i O } } } ( \tau ) = \frac { 1 } { | \mathcal { A } ( \tau ) | } \sum _ { c \in \mathcal { A } ( \tau ) } g _ { c } ( \overline { { r } } _ { c } ( \tau ) )\tag{4}
$$

PA-DAPO. We optimize the memory policy actor $\pi _ { \theta }$ with our proposed PA-DAPO. Building on DAPO (Yu et al., 2026b), AiO-Mem extends the loss computation from the conventional (group, token) structure to a new (group, action, token) hierarchy over trajectories. Formally, let $\mathcal { D } _ { \mathrm { R L } }$ denote the distribution of query–history pairs used for policy optimization. For $( q , \mathcal { H } ) \sim \mathcal { D } _ { \mathrm { R L } }$ , the old memory policy actor samples G trajectories $\{ \tau _ { j } \}$ with rewards $R _ { j } ~ =$ $R _ { \mathrm { A i O } } ( \tau _ { j } )$ . The group-relative advantage $\hat { A } _ { j } = ( R _ { j } - \mu ) / \sigma$ is computed from the within-group mean µ and standard deviation $\sigma .$ Let $o _ { j , k , t }$ be the t-th policy-generated token of the k-th action step in $\tau _ { j } , \mathbf { h } _ { j , k , t }$ the preceding context, and $L _ { j , k }$ the step’s token count. PA-DAPO maximizes:

$$
\mathcal { I } _ { \mathrm { P A - D A P O } } ( \theta ) = \mathbb { E } \left[ \frac { \sum _ { j } \sum _ { k } \sum _ { t } \Gamma _ { j , k , t } ( \theta ) } { \sum _ { j } \sum _ { k } L _ { j , k } } \right] , \quad \sigma > 0\tag{5}
$$

where $\Gamma _ { j , k , t } ( \theta ) \ = \ \operatorname * { m i n } ( \rho _ { j , k , t } \hat { A } _ { j } , \mathrm { c l i p } ( \rho _ { j , k , t } , 1 \ - \ \epsilon _ { \mathrm { l o w } } , 1 \ + \ \epsilon _ { \mathrm { h i g h } } ) \hat { A } _ { j } )$ is the clipped surrogate, $\rho _ { j , k , t } ( \theta ) = \pi _ { \theta } ( o _ { j , k , t } \mid \mathbf { h } _ { j , k , t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { j , k , t } \mid \mathbf { h } _ { j , k , t } )$ the token-level importance ratio, and $\epsilon _ { \mathrm { l o w } } , \epsilon _ { \mathrm { h i g h } }$

Table 1: Main results on LoCoMo.
<table><tr><td rowspan="2">Method</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Open-domain</td><td colspan="2">Single-hop</td><td colspan="2">Overall</td></tr><tr><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td></tr><tr><td colspan="9">Reference</td><td></td></tr><tr><td>Long-Context</td><td>23.2</td><td>16.8</td><td>19.0</td><td>13.4</td><td>13.9</td><td>10.7</td><td>40.7</td><td>31.5</td><td>31.5</td><td>23.9</td></tr><tr><td>RAG</td><td>23.5</td><td>17.1</td><td>43.1</td><td>37.4</td><td>20.2</td><td>14.9</td><td>49.5</td><td>44.9</td><td>41.6</td><td>36.5</td></tr><tr><td colspan="9">Agent Memory Harness</td><td></td></tr><tr><td>MemGPT</td><td>16.7</td><td>14.0</td><td>12.6</td><td>10.9</td><td>11.6</td><td>9.2</td><td>14.0</td><td>11.8</td><td>14.1</td><td>11.8</td></tr><tr><td>MemoryBank</td><td>25.5</td><td>19.4</td><td>9.2</td><td>7.4</td><td>16.4</td><td>12.4</td><td>26.7</td><td>17.7</td><td>22.3</td><td>15.7</td></tr><tr><td>Mem0</td><td>35.4</td><td>27.4</td><td>46.8</td><td>39.5</td><td>26.6</td><td>21.0</td><td>47.3</td><td>40.7</td><td>43.7</td><td>36.8</td></tr><tr><td>MemoryOS</td><td>35.2</td><td>27.3</td><td>41.0</td><td>32.7</td><td>22.1</td><td>17.9</td><td>48.4</td><td>42.6</td><td>42.8</td><td>36.3</td></tr><tr><td>LightMem</td><td>30.8</td><td>25.8</td><td>44.7</td><td>40.7</td><td>18.9</td><td>14.4</td><td>43.8</td><td>38.8</td><td>40.0</td><td>35.3</td></tr><tr><td>A-Mem</td><td>27.2</td><td>21.1</td><td>43.9</td><td>36.0</td><td>15.4</td><td>12.7</td><td>44.6</td><td>38.3</td><td>39.4</td><td>33.0</td></tr><tr><td>GAM</td><td>32.2</td><td>28.7</td><td>26.3</td><td>22.5</td><td>18.5</td><td>14.5</td><td>32.2</td><td>25.5</td><td>30.2</td><td>24.8</td></tr><tr><td colspan="9">Trained Method</td><td></td></tr><tr><td>MEM1</td><td>19.0</td><td>15.6</td><td>30.5</td><td>23.5</td><td>14.2</td><td>11.4</td><td>27.5</td><td>22.1</td><td>25.7</td><td>20.5</td></tr><tr><td>MemAgent</td><td>27.9</td><td>22.7</td><td>37.9</td><td>31.9</td><td>20.3</td><td>16.5</td><td>35.9</td><td>29.6</td><td>33.8</td><td>28.0</td></tr><tr><td>Mem-T</td><td>45.1</td><td>36.6</td><td>65.1</td><td>60.1</td><td>33.0</td><td>28.9</td><td>63.8</td><td>58.0</td><td>58.7</td><td>52.6</td></tr><tr><td colspan="9">Ours</td><td></td><td></td></tr><tr><td>AiO-Mem</td><td>47.1</td><td>38.8</td><td>66.0</td><td>59.4</td><td>33.7</td><td>30.5</td><td>67.0</td><td>61.5</td><td>61.0</td><td>55.0</td></tr></table>

the clipping bounds. The constraint $\sigma > 0$ retains only queries with reward variation, filtering fully converged samples. Operating on an action-balanced trajectory reward rather than binary answer correctness, PA-DAPO provides a graded training signal for end-to-end policy optimization.

## 4 EXPERIMENT

## 4.1 EXPERIMENTAL SETTING

Datasets and Evaluation We evaluate AiO-Mem on three complementary memory benchmarks, including LoCoMo (Maharana et al., 2024), MemoryAgentBench (Hu et al., 2026b), and PersonaMem (Jiang et al., 2025a). LoCoMo evaluates long-range conversational memory across multisession interactions. Following MEM-T (Yue et al., 2026), we adopt the same data configuration by splitting the LoCoMo dataset into training, validation, and test sets with a ratio of 1:1:8 to ensure a fair comparison. MemoryAgentBench valuates memory agents under incremental multi-turn interactions across diverse tasks. Following Fine-Mem (Ma et al., 2026), we adopt the same evaluation scope and baselines to assess out-of-distribution generalization. PersonaMem further extends evaluation from factual memory to dynamic user profiling. More details are shown in Appendix C.1.

Baselines. Twenty systems in three groups. ① Reference: Long-Context and RAG. ② Agent Memory Harness: MemGPT (Packer et al., 2023), MemoryBank (Zhong et al., 2024), Mem0 (Chhikara et al., 2025), MemoryOS (Kang et al., 2025), LightMem (Fang et al., 2026), A-Mem (Xu et al., 2026), GAM (Yan et al., 2025), Zep (Rasmussen et al., 2025), LangMem (LangChain, 2025), EverMemOS (Hu et al., 2026a), and Emotional RAG (Huang et al., 2024). ③ RL-Trained Method: MEM1 (Zhou et al., 2026), MemAgent (Yu et al., 2026a), Mem-T (Yue et al., 2026), Atom-Mem (Huo et al., 2026), Mem-α (Wang et al., 2025), FineMem (Ma et al., 2026), and Trust-Mem (Yang et al., 2026). Details are provided in Appendix C.1.

Experimental Details. We instantiate AiO-Mem with Qwen3-8B as the reward model and Qwen3-4B as the memory policy actor. BGE-M3<sup>1</sup> and BM25 are used for multi-view event retrieval. For policy training, we set $K _ { \mathrm { r e t } } = 1 0 , T _ { \mathrm { m a x } } = 8$ . For reward-model SFT, we train for 2 epochs with a batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 4 }$ . Trajectory-level DPO is then performed for 1 epoch with a batch size of 8, a learning rate of $2 \times 1 0 ^ { - 6 }$ , and $\beta = 0 . 1$ . For PA-DAPO, we sample 8 trajectories per query and optimize the memory policy with a learning rate of $2 \times 1 0 ^ { - 5 }$ More training setup and parameter configurations are listed in Appendix C.2.

![](images/f4224cd9be8fd09c7f47b125305923642f46bedb085310c79b8f5109e4aa6a38.jpg)  
Figure 3: Main results on MemoryAgentBench.

## 4.2 MAIN RESULTS

LoCoMo. As shown in Table 1, , AiO-Mem achieves the strongest overall performance and consistently outperforms competing methods across diverse question types. The gains span both evidenceintensive reasoning and direct memory queries, reflecting a coherent improvement pattern rather than isolated advantages on specific categories. In contrast, existing memory systems exhibit more uneven performance across different reasoning demands. These results suggest that jointly modeling heterogeneous memory actions under shared evidence states and unified process supervision enables more reliable memory interaction throughout the reasoning process.

MemoryAgentBench. As shown in Figure 3, AiO-Mem achieves the strongest overall performance across MemoryAgentBench, with gains that form a consistent pattern across retrieval, testtime learning, and long-range understanding. It achieves an average score of 74.7 across the nine tasks, outperforming Fine-Mem, the strongest baseline, by 8.3 points. Compared with the best competing result on each task, AiO-Mem improves LME(S), TREC-F, and InfBench by 12.6, 19.0, and 4.5 points, respectively. Such gains across heterogeneous task formats demonstrate strong outof-distribution generalization beyond the LoCoMo training setting. This further supports the design of AiO-Mem for generalizable memory interaction.

PersonaMem. AiO-Mem achieves the best overall performance on PersonaMem, surpassing the strongest baseline by 3.9 points and demonstrating stronger long-term user profiling and personalization. More details are shown in Appendix C.1.

## 4.3 ABLATION STUDY

To examine the contribution of each component in AiO-Mem, we conduct ablation experiments by removing individual modules, as shown in Table 2. Removing Event Compilation (w/o Event Comp.) or Multi-view Retrieval (w/o Multi-view Ret.) consistently degrades performance, showing that structured event organization and complementary retrieval cues provide a reliable evidence basis for memory interaction. Removing Working-Memory Construction (w/o WM Constr.) causes the most substantial degradation, highlighting its role in progressively transforming retrieved evidence into reasoning-sufficient working memory. For Unified Reward Modeling, removing Evidence-Grounded SFT (w/o Evid. SFT) weakens performance, demonstrating the importance of evidencebased supervision for establishing reliable process rewards. Further removing Trajectory-Level DPO (w/o Traj. DPO) also leads to consistent degradation, showing that trajectory-level alignment improves reward estimation beyond rule-based supervision. Overall, AiO-Mem achieves the best performance by jointly integrating these components within a unified memory-interaction framework.

Table 2: Ablation on LoCoMo. Row abbreviations: Event Comp. = Event Compilation, Multi-view Ret. = Multi-view Retrieval, WM Constr. = Working-Memory Construction, Evid. SFT = Evidence-Grounded SFT, and Traj. DPO = Trajectory-Level DPO. Red values denote drops over AiO-Mem.
<table><tr><td rowspan="2">Method</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Open-domain</td><td colspan="2">Single-hop</td><td colspan="4">Overall</td></tr><tr><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td></td><td>B1</td><td></td></tr><tr><td colspan="9">Base Model 39.3</td><td colspan="5">55.8 (-5.2)</td></tr><tr><td colspan="9">Memory Representation and Interaction Ablations</td><td colspan="4"></td></tr><tr><td>w/o Event Čomp.</td><td>43.2</td><td>34.6</td><td>66.9</td><td>61.0</td><td>27.2</td><td>22.6</td><td>64.3</td><td>58.7</td><td>58.6</td><td>(-2.4)</td><td>52.5</td><td>(-2.5)</td></tr><tr><td>w/o Multi-view Ret.</td><td>41.3</td><td>33.4</td><td>65.8</td><td>60.4</td><td>29.5</td><td>24.4</td><td>64.2</td><td>58.6</td><td>58.1</td><td>(-2.9)</td><td>52.1</td><td>(-2.9)</td></tr><tr><td>w/o WM Constr.</td><td>33.2</td><td>26.4</td><td>57.0</td><td>50.9</td><td>24.3</td><td>20.0</td><td>57.6</td><td>52.3</td><td>50.9</td><td>(-10.1)</td><td>45.2</td><td>(-9.8)</td></tr><tr><td colspan="9">Unified Reward Modeling Ablations</td><td colspan="4"></td></tr><tr><td>w/o Evid. SFT</td><td>41.4</td><td>33.6</td><td>68.4</td><td>60.6</td><td>28.6</td><td>24.5</td><td>62.9</td><td>57.4</td><td>57.9</td><td>(-3.1)</td><td>51.6</td><td>(-3.4) (-2.4)</td></tr><tr><td colspan="9">w/o Traj. DPO</td><td colspan="4">52.6</td></tr><tr><td>AiO-Mem</td><td>43.4 47.1</td><td>35.2 38.8</td><td>66.5 66.0</td><td>59.5 59.4</td><td>34.2 33.7</td><td>29.9 30.5</td><td>64.0 67.0</td><td>58.4 61.5</td><td colspan="4">58.8 (-2.2) 61.0</td></tr></table>

![](images/c00aa8c52f60eeb9b794c2d839830b75c78050ef95146b1b27b1e61f8bdd85f0.jpg)  
Figure 4: Parameter analysis on LoCoMo over top-K $( K \in \{ 5 , 1 0 , 1 5 \} )$ ) and maximum interaction steps $T _ { \mathrm { m a x } } \in \{ 4 , 6 , 8 , 1 0 \}$ , reporting F1 across four question types (MH = Multi-hop, Tmp = Temporal, OD = Open-domain, SH = Single-hop) and overall. To facilitate comparison, all values are normalized for visualization on the chart.

## 4.4 ANALYSIS OF COMPLEXITY AND SENSITIVITY

Complexity Analysis. Existing memory systems repeatedly summarize and rewrite accumulated memories, incurring a history-dependent cost of $O ( \dot { N } \ell + \dot { N } M _ { 1 } R _ { 1 } ) \mathcal { C } _ { \mathrm { L L M } }$ . In contrast, AiO-Mem decouples persistent storage from query-time working-memory construction. The event bank B is built in a single pass, while query-time reasoning is bounded by $K _ { \mathrm { r e t } }$ and $T _ { \mathrm { m a x } } ,$ yielding $O ( N \ell + T _ { \mathrm { m a x } } K _ { \mathrm { r e t } } \ell _ { e } ) \mathcal { C } _ { \mathrm { L L M } }$ . By bounding the working context to a constant size, AiO-Mem transforms a history-dependent scaling cost into a local, horizon-bounded overhead that remains invariant to the stream length. Empirically, this decoupling mechanism reduces retrieval and construction token consumption by 79.9% and 73.8% respectively, bypassing the active memory maintenance bottleneck of traditional architectures. Detailed derivations are provided in Appendix F.

Sensitivity Analysis. We further examine the robustness of AiO-Mem to two key hyperparameters, the retrieval width $K _ { \mathrm { r e t } }$ and maximum interaction horizon $T _ { \mathrm { m a x } }$ . As shown in Figure 4, performance remains relatively stable across configurations, while increasing either parameter yields no consistent gains. A moderate retrieval width provides sufficient evidence coverage without introducing excessive irrelevant candidates, while an appropriate interaction horizon enables effective multi-step memory interaction without redundant actions. The configuration $( K _ { \mathrm { r e t } } , T _ { \mathrm { m a x } } ) = ( 1 0 , 8 )$ achieves the strongest overall performance, suggesting that AiO-Mem benefits from a balanced retrieval and interaction budget. Detailed F1 and B1 results are provided in Appendix D.1.

![](images/4608087b027d93792baccbd722c5d2e35858ddf117179e486f25714b4ea2141a.jpg)  
Figure 5: Two cases between the MEM-T and our AiO-Mem method.

## 4.5 CASE STUDY

To further illustrate the advantage of AiO-Mem over Mem-T, Figure 5 presents two memoryinteraction cases. In Example (a), Mem-T incorrectly binds “she is 5 years old” to Deborah rather than Luna during memory construction. Although relevant evidence is retrieved, the incorrect binding is retained in memory and leads to an incorrect assessment. In contrast, AiO-Mem retrieves source-grounded evidence and constructs a working-memory note that preserves the correct entity relation. In Example (b), Mem-T conflates the temporally distinct robotics and water-purifier projects, causing the assessment to overlook the query-time cutoff and select evidence from a later event. AiO-Mem instead retrieves dated evidence and preserves its temporal relation during working-memory construction, leading to the correct answer. These cases show that AiO-Mem improves memory interaction by jointly coordinating retrieval and construction under shared evidence states and process-level supervision, preventing errors from propagating into later reasoning.

## 5 ADDITIONAL ABLATIONS IN APPENDIX

We further examine AiO-Mem across the following key dimensions. (i) Multi-View Retrieval Complementarity and Sensitivity. The four retrieval views provide complementary evidence coverage, jointly covering 97.03% of question–evidence pairs while remaining robust to retrieval configurations and context cost (Appendix D.3, Appendix D.4).(ii) Unified Reward Alignment. The trained reward model shows stronger correlations with evidence Recall and Precision, better reflecting evidence-grounded trajectory quality (Appendix E.1). (iii) Trajectory-level DPO Alignment Study. We visualize how trajectory-level DPO distinguishes trajectories with identical Evidence-ID metrics through finer-grained semantic judgments(Appendix E.2). (iv) Training Dynamics. Overall, consistent gains across Search, Extract, and Answer-Decision rewards indicate stable PA-DAPO training and coordinated step-wise optimization(Appendix E.3). (v) Process-Aligned Evidence Flow. PA-DAPO improves evidence retrieval, working-memory quality, and Answer F1 across the full pipeline (Appendix E.4).

## 6 CONCLUSION

Existing memory systems often optimize retrieval and memory construction separately, making it difficult to coordinate heterogeneous memory operations under a shared objective. To address this, we introduce AiO-Mem, which unifies retrieval, writing, filtering, and termination as transitions over a shared evidence state, and learns a trajectory-conditioned reward model to provide process-level supervision for memory-policy optimization. Experiments across long-context memory benchmarks show that AiO-Mem consistently outperforms both static memory systems and RL-trained memory agents, with strong out-of-distribution generalization beyond conversational QA. These results highlight the promise of an All-in-One formulation for learning reliable and generalizable memory interaction in LLM agents.

## AI USE STATEMENT

LLMs were used to generate data and to evaluate memory-interaction trajectories as judges, as described in the paper. The core method development in this paper does not involve LLM usage. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

AiO-Mem organizes long interaction histories into retrievable events and constructs query-specific working memory. Although this can improve an agent’s ability to use historical evidence, such histories may contain sensitive information. Retaining source-linked events may increase the risk of inappropriate access to or disclosure of that information. Inaccurate or biased information in a history may also be retrieved and propagated into later answers. Our experiments use the existing LoCoMo, MemoryAgentBench, and PersonaMem benchmarks. Deployment with real user histories would require appropriate safeguards for user consent, data access, retention and deletion, as well as mechanisms for identifying and correcting inaccurate memories.

## REPRODUCIBILITY STATEMENT

The method is described in Section 3. Appendix C provides the experimental details. Prompts and memory-agent tool calls are provided in Appendix G. To support reproducibility, we will release our code on GitHub .

## REFERENCES

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Gordon V Cormack, Charles LA Clarke, and Stefan Buettcher. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval, pp. 758–759, 2009.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, et al. Lightmem: Lightweight and efficient memoryaugmented generation. In International Conference on Learning Representations, volume 2026, pp. 98706–98729, 2026.

Bernal J Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiolog-´ ically inspired long-term memory for large language models. Advances in neural information processing systems, 37:59532–59569, 2024.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pp. 3929–3938. PMLR, 2020.

Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, et al. Evermemos: A self-organizing memory operating system for structured long-horizon reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 45836–45853, 2026a.

Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental multi-turn interactions. In International Conference on Learning Representations, volume 2026, pp. 156259–156291, 2026b.

Le Huang, Hengzhi Lan, Zijun Sun, Chuan Shi, and Ting Bai. Emotional rag: Enhancing roleplaying agents through emotional retrieval. In 2024 IEEE International Conference on Knowledge Graph (ICKG), pp. 120–127. IEEE, 2024.

Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, and Yankai Lin. Atommem : Learnable dynamic agentic memory with atomic memory operation, 2026. URL https://arxiv.org/ abs/2601.08323.

Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J Taylor, and Dan Roth. Know me, respond to me: Benchmarking llms for dynamic user profiling and personalized responses at scale. arXiv preprint arXiv:2504.14225, 2025a.

Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, and Jiawei Han. Deepretrieval: Hacking real search engines and retrievers with large language models via reinforcement learning. arXiv preprint arXiv:2503.00223, 2025b.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Stephen C Johnson. Hierarchical clustering schemes. Psychometrika, 32(3):241–254, 1967.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 25972–25981, 2025.

LangChain. LangMem: Long-term memory for language agents. https://github.com/ langchain-ai/langmem, 2025. Official software repository and documentation, accessed 2026-09-19.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, et al. Retrieval-augmented gener-¨ ation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics (TACL), 12:157–173, 2024.

Weitao Ma, Xiaocheng Feng, Lei Huang, Xiachong Feng, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, and Bing Qin. Fine-mem: Fine-grained feedback alignment for longhorizon memory management. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19666–19684, 2026.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, 2024.

OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.

Diego Sanmartin. Kg-rag: Bridging the gap between knowledge and creativity. arXiv preprint arXiv:2405.12035, 2024.

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. arXiv preprint arXiv:2503.05592, 2025.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Ar-\` mand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-{\alpha}: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911, 2025.

Joe H. Ward. Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301):236–244, 1963.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2026.

BY Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, and Zheng Liu. General agentic memory via deep research. arXiv preprint arXiv:2511.18423, 2025.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z Pan, et al. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12805– 12825, 2026.

Tianyu Yang, Sudipta Paul, Vijay Srinivasan, Vivek Kulkarni, and Srinivas Chappidi. Trustmem: Learning trustworthy memory consolidation for llm agents with long-term memory. arXiv preprint arXiv:2606.25161, 2026.

Xiaoju Ye. calflops: a flops and params calculate tool for neural networks in pytorch framework. https://github.com/MrYxJ/calculate-flops.pytorch, 2023.

Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. In International Conference on Learning Representations, volume 2026, pp. 39458–39486, 2026a.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026b.

Yanwei Yue, Boci Peng, Xuanbo Fan, Jiaxin Guo, Qiankun Li, and Yan Zhang. Mem-t: Densifying rewards for long-horizon memory agents. arXiv preprint arXiv:2601.23014, 2026.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 19724–19731, 2024.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low, and Paul Liang. Mem1: Learning to synergize memory and reasoning for efficient longhorizon agents. In International Conference on Learning Representations, volume 2026, pp. 58413–58438, 2026.

## A APPENDIX OVERVIEW AND ORGANIZATION

This appendix provides supplementary details to support and extend the main paper. The organization of the appendix is as follows:

1. Additional Related Work (Section B): This section provides further discussion of memory-augmented large language models and reinforcement learning methods for memory policies, with particular emphasis on persistent memory management and process-level memory optimization.

2. Experimental Setup and Supplementary Experiments (Sections C and D): These sections describe the datasets, baselines, and implementation details, and provide additional experiments on parameter sensitivity, PersonaMem, multi-view retrieval complementarity, and inference-time retrieval configurations.

3. Reward, Optimization, and Efficiency Analysis (Sections E and F): These sections further analyze reward alignment, qualitative reward cases, PA-DAPO training dynamics, evidence flow through the memory pipeline, as well as the computational complexity and inference efficiency of AiO-Mem.

4. Prompts and Tool Calls (Section G): This section provides the memory-agent tool definitions and prompt templates used for memory interaction and stepwise reward modeling, supporting reproducibility of the proposed framework.

## B RELATED WORK

## B.1 ADDITIONAL DISCUSSION ON MEMORY-AUGMENTED LARGE LANGUAGE MODELS

Memory-augmented language models have evolved from external retrieval toward persistent and structured memory management. Early retrieval-augmented methods, including RAG (Lewis et al., 2020) and REALM (Guu et al., 2020), introduce non -parametric knowledge through dense retrieval, while subsequent systems such as GraphRAG (Edge et al., 2024), HippoRAG (Gutierrez´ et al., 2024), KG-RAG (Sanmartin, 2024), and Emotional RAG (Huang et al., 2024) enrich retrieval with graph structures, multi-step exploration, or task-specific similarity signals. These methods primarily focus on improving how external evidence is accessed for a given query. Another line of work develops persistent memory across interactions. MemGPT (Packer et al., 2023), Memory-Bank (Zhong et al., 2024), and Mem0 (Chhikara et al., 2025) introduce explicit mechanisms for storing, updating, and maintaining long-term information, while MemoryOS (Kang et al., 2025) and LightMem (Fang et al., 2026) improve scalability through hierarchical storage and lightweight consolidation. More recent systems, including A-Mem (Xu et al., 2026), Zep (Rasmussen et al., 2025), EverMemOS (Hu et al., 2026a), LangMem (LangChain, 2025), and GAM (Yan et al., 2025), further explore atomic memory units, temporal structures, background consolidation, and iterative evidence gathering. These methods substantially expand the capabilities of long-term memory systems in construction, organization, updating, and retrieval. However, the individual operations are still largely coordinated through predefined workflows, prompts, or heuristic rules, rather than being jointly optimized according to their contribution to downstream evidence acquisition.

## B.2 ADDITIONAL DISCUSSION ON REINFORCEMENT LEARNING FOR MEMORY POLICIES

Reinforcement learning has increasingly been introduced to replace manually specified memory behaviors with learned policies. Early efforts mainly focus on retrieval, where Search-R1 (Jin et al., 2025), R1-Searcher (Song et al., 2025), and DeepRetrieval (Jiang et al., 2025b) optimize multi-step search, tool invocation, or query generation through task- or retrieval-oriented rewards. Another line applies RL to recurrent memory states: MEM1 (Zhou et al., 2026) and MemAgent (Yu et al., 2026a) learn to continually compress or overwrite bounded memory representations over long contexts and propagate downstream rewards to preceding memory decisions. Persistent-memory methods such as Memory-R1 (Yan et al., 2026), Mem-α (Wang et al., 2025), and AtomMem (Huo et al., 2026) further extend the learned policy from memory construction and updating toward broader combinations of reading, editing, and answering. More recent work introduces finer-grained supervision to improve credit assignment across long memory-operation sequences. Fine-Mem (Ma et al., 2026)

and TRUSTMEM (Yang et al., 2026) provide localized feedback for memory construction or updating through intermediate QA signals, downstream usage, or verifier-based evaluation, while keeping retrieval or reasoning largely fixed. Mem-T (Yue et al., 2026) further connects memory formation, evolution, and multi-step retrieval by evaluating retrieval trajectories and propagating later retrieval outcomes back to earlier construction decisions. These methods move toward process-aware memory optimization, but their process feedback typically targets specific memory stages or estimates the contribution of earlier actions from later outcomes. In contrast, AiO-Mem unifies memory interactions as transitions over a shared evidence state and uses a trajectory-conditioned reward model to directly assess the semantic quality of each memory action in a unified reward space, enabling coordinated optimization of retrieval and construction.

## C EXPERIMENT DETAILS

## C.1 DATASETS AND BASELINES

Datasets The LoCoMo dataset (Maharana et al., 2024) targets the evaluation of long-range conversational memory. It features extremely long dialogues, with each conversation spanning roughly 300 turns and around 9K tokens on average. The accompanying questions cover four categories—Single-hop, Multi-hop, Temporal, and Open-domain—capturing different aspects of memory and reasoning in LLMs. Following the evaluation setup of previous work, we report F1 and BLEU-1 (B1) scores for each category as well as the overall performance.

MemoryAgentBench (Hu et al., 2026b) targets the evaluation of memory agents under incremental, multi-turn interactions. In our experiments, we evaluate three memory capabilities— Accurate Retrieval, Test-Time Learning, and Long-Range Understanding—using nine tasks from the benchmark. The selected tasks include SH-Doc QA (SH-Doc), MH-Doc QA (MH-Doc), Long-MemEval (LME(S)), TREC Coarse (TREC-C), NLU, TREC Fine (TREC-F), CLINC150 (CLINC), BANKING77 (Banking), and ∞Bench-Sum (InfBench).

The PersonaMem benchmark (Jiang et al., 2025a) targets the evaluation of dynamic user profiling and personalized response generation over long-term user–LLM interactions. Unlike factual memory benchmarks that primarily focus on retrieving explicitly stated information from past interactions, PersonaMem emphasizes user-profile memory, requiring models to infer, maintain, and update users’ preferences and underlying traits over time. In our experiments, we evaluate five query types corresponding to five dimensions: Recall User-Shared Facts (Rec.), Generalize to New Scenarios (Gen.), Track Full Preference Evolution (Evol.), Revisit Reasons Behind Preference Updates (Reason), and Provide Preference-Aligned Recommendations (Reco.). Together, these query types assess whether a memory system can recover user-specific facts, generalize learned preferences to new situations, track evolving preferences, reason about the causes of preference changes, and provide recommendations aligned with the user’s current profile.

Baselines. We compare AiO-Mem against several representative baselines across three groups. ① Reference: Long-Context and RAG. ② Agent Memory Harness: MemGPT (Packer et al., 2023), MemoryBank (Zhong et al., 2024), Mem0 (Chhikara et al., 2025), MemoryOS (Kang et al., 2025), LightMem (Fang et al., 2026), A-Mem (Xu et al., 2026), GAM (Yan et al., 2025), Zep (Rasmussen et al., 2025), LangMem (LangChain, 2025), EverMemOS (Hu et al., 2026a), and Emotional RAG (Huang et al., 2024). These approaches equip a frozen backbone model with explicit memory storage, retrieval, summarization, organization,or update mechanisms. ③ Trained Method: MEM1 (Zhou et al., 2026), MemAgent (Yu et al., 2026a), Mem-T (Yue et al., 2026), AtomMem (Huo et al., 2026), Mem-α (Wang et al., 2025), FineMem (Ma et al., 2026), and Trust-Mem (Yang et al., 2026). Compared with agent memory harnesses, these baselines either learn memory-oriented agent behaviors or benefit from stronger post-trained reasoning capabilities. To ensure a fair comparison, we follow the same baseline configurations as prior work (Yue et al., 2026; Ma et al., 2026).

## C.2 IMPLEMENTATION DETAILS

For traceable multi-view event bank and evidence-state transition modeling, we use Qwen3-8B as the structure-aware compiler ${ \mathcal { C } } ,$ , dense retrieval for the summary and event-phrase views, BM25 for the query-anchor view, and a combination of BM25 and dense retrieval for the raw-turn view. We use BGE-M3<sup>2</sup> as the embedding model. We set the maximum interaction horizon to $T _ { \mathrm { m a x } } = 8$ and the observation text budget to 10,240 tokens for evidence extraction.

For unified reward modeling, we construct the training data from policy rollouts. We use Qwen3-8B as the reward model. We filter low-quality and near-duplicate samples using BGE-M3 similarity, then apply agglomerative clustering (Johnson, 1967; Ward, 1963) to improve diversity and retain at most 10 representative trajectories per question. This yields 1,248 evidence-grounded SFT examples covering 150 questions and 4,162 scoreable steps, with the selected trajectories spanning diverse behavioral regions as shown in Figure 6. We derive action-specific rule scores for retrieval, note construction, citation, and stopping, prioritizing recall of missing evidence over evidence precision. We then fine-tune the reward model with LoRA SFT for 2 epochs using a global batch size of 8, a learning rate of $1 \times 1 0 ^ { - 4 }$ and LoRA rank, alpha, and dropout of 32, 64, and 0.05. For trajectory-level DPO alignment,

![](images/ec5f11d4236ef289056eb5364984adb453a871d386865bc50c30ed93ef348687.jpg)  
Figure 6: Reward-model trajectories by behavioral region in t-SNE space.

we use DeepSeek- $. { \mathrm { V } } 4 { \mathrm { - F l a s h } } ^ { 3 }$ to review scoreable steps and retain cases differing from the rulebased score by one quality level, yielding 358 preference pairs from 296 trajectories and 102 questions. Human evaluation on 100 sampled steps shows strong agreement with DeepSeek-V4-Flash $( \mathrm { I C C } = 0 . 8 7 )$ . Finally, the reward model is further optimized with DPO for one epoch using a batch size of 8, a learning rate of $2 \times 1 0 ^ { - 6 }$ , and $\beta = 0 . 1$

For process-aligned policy optimization, we use Qwen3-4B as the memory policy actor. We first aggregate step-level score with the action-balanced trajectory reward so that repeated search and add operations do not dominate less frequent but decisive actions such as stop. For reward normalization, we set $r _ { c } ^ { \mathrm { l o } } = 1$ and $r _ { c } ^ { \mathrm { h i } } = 1 0 \dot { 0 }$ for every $c \in A .$ . Following MEM-T (Yue et al., 2026), we adopt the same data configuration by splitting the LoCoMo dataset into training, validation, and test sets with a ratio of 1:1:8 to ensure a fair comparison. During training, we set the retrieval width to $K _ { \mathrm { r e t } } = 1 0 .$ , the maximum interaction horizon to $T _ { \mathrm { m a x } } = 8 .$ , and the number of rollouts to 8. The policy is optimized with LoRA using a learning rate of $2 \times 1 0 ^ { - 5 }$ , with the LoRA rank and alpha set to 32 and 64, respectively. We use Qwen3-4B as the answer model (fine-tuned on our LoCoMo training split for LoCoMo) and enable thinking mode during inference. All experiments are conducted on a single node with 8× NVIDIA A6000 GPUs (40 GB each), with a training batch size of 8. Detailed prompts are provided in Appendix G.

## D SUPPLEMENTARY EXPERIMENTS

## D.1 SENSITIVITY ANALYSIS

Table 3 reports the detailed F1 and B1 results for all evaluated top-K values and maximum interaction steps $T _ { \mathrm { m a x } }$ on LoCoMo. Models trained with different K and $T _ { \mathrm { m a x } }$ settings consistently achieve strong performance, demonstrating the generalizability of our method across training budgets. Overall F1 is highest at $K = 1 0 , T _ { \mathrm { m a x } } = 8 ,$ , which we adopt as our configuration.

Table 3: Parameter analysis on LoCoMo with different top-K values and maximum interaction steps $T _ { \mathrm { m a x } } .$
<table><tr><td rowspan="2">top-K</td><td rowspan="2">Max Step</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Open-domain</td><td colspan="2">Single-hop</td><td colspan="2">Overall</td></tr><tr><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td><td>F1</td><td>B1</td></tr><tr><td rowspan="4">K = 5</td><td> $T _ { \mathrm { m a x } } = 4$ </td><td>44.6</td><td>35.7</td><td>66.2</td><td>59.6</td><td>30.3</td><td>25.6</td><td>62.7</td><td>57.3</td><td>58.0</td><td>51.8</td></tr><tr><td> $T _ { \mathrm { m a x } } = 6$ </td><td>43.5</td><td>35.4</td><td>62.4</td><td>53.3</td><td>34.9</td><td>30.9</td><td>65.9</td><td>61.5</td><td>59.2</td><td>53.2</td></tr><tr><td> $T _ { \mathrm { m a x } } = 8$ </td><td>44.0</td><td>36.6</td><td>63.9</td><td>55.7</td><td>34.4</td><td>30.3</td><td>65.4</td><td>60.9</td><td>59.2</td><td>53.5</td></tr><tr><td> $T _ { \mathrm { m a x } } = 1 0$ </td><td>43.0</td><td>35.1</td><td>61.9</td><td>53.1</td><td>33.5</td><td>29.3</td><td>65.8</td><td>61.1</td><td>58.8</td><td>52.8</td></tr><tr><td rowspan="4">K = 10</td><td> $T _ { \mathrm { m a x } } = 4$ </td><td>43.6</td><td>35.6</td><td>63.4</td><td>54.6</td><td>33.7</td><td>30.2</td><td>65.9</td><td>60.9</td><td>59.3</td><td>53.3</td></tr><tr><td> $T _ { \mathrm { m a x } } = 6$ </td><td>46.9</td><td>38.1</td><td>66.5</td><td>60.6</td><td>32.4</td><td>29.6</td><td>66.1</td><td>61.2</td><td>60.7</td><td>54.9</td></tr><tr><td> $\mathbf { T } _ { \mathrm { m a x } } = \mathbf { 8 }$ </td><td>47.1</td><td>38.8</td><td>66.0</td><td>59.4</td><td>33.7</td><td>30.5</td><td>67.0</td><td>61.5</td><td>61.0</td><td>55.0</td></tr><tr><td> $T _ { \mathrm { m a x } } = 1 0$ </td><td>43.5</td><td>35.3</td><td>66.1</td><td>59.2</td><td>33.0</td><td>28.9</td><td>65.9</td><td>60.1</td><td>59.8</td><td>53.4</td></tr><tr><td rowspan="4">K = 15</td><td> $T _ { \mathrm { m a x } } = 4$ </td><td>43.5</td><td>35.4</td><td>61.2</td><td>52.0</td><td>32.6</td><td>29.2</td><td>66.5</td><td>61.9</td><td>59.1</td><td>53.0</td></tr><tr><td> $T _ { \mathrm { m a x } } = 6$ </td><td>43.5</td><td>35.7</td><td>63.3</td><td>53.6</td><td>34.8</td><td>30.9</td><td>65.8</td><td>61.0</td><td>59.3</td><td>53.0</td></tr><tr><td> $T _ { \mathrm { m a x } } = 8$ </td><td>45.5</td><td>36.7</td><td>65.7</td><td>60.6</td><td>31.2</td><td>28.2</td><td>66.5</td><td>60.9</td><td>60.3</td><td>54.4</td></tr><tr><td> $T _ { \mathrm { m a x } } = 1 0$ </td><td>45.1</td><td>36.1</td><td>65.0</td><td>58.8</td><td>30.6</td><td>26.9</td><td>65.8</td><td>60.5</td><td>59.6</td><td>53.6</td></tr></table>

## D.2 ADDITIONAL RESULTS ON PERSONAMEM

Table 4: Main results on PersonaMem.
<table><tr><td>Method</td><td>Rec.</td><td>Gen.</td><td>Evol.</td><td>Reason</td><td>Reco.</td><td>Overall</td></tr><tr><td>Claude 3.7-Sonnet</td><td>25.0</td><td>29.0</td><td>45.0</td><td>57.0</td><td>43.0</td><td>40.0</td></tr><tr><td>Mem0</td><td>32.1</td><td>57.9</td><td>54.7</td><td>80.8</td><td>52.7</td><td>53.4</td></tr><tr><td>Zep</td><td>35.3</td><td>42.1</td><td>34.5</td><td>50.5</td><td>36.4</td><td>39.1</td></tr><tr><td>LangMem</td><td>31.3</td><td>8.8</td><td>53.2</td><td>81.8</td><td>40.0</td><td>46.4</td></tr><tr><td>A-MEM</td><td>63.0</td><td>57.9</td><td>54.7</td><td>85.9</td><td>69.1</td><td>65.3</td></tr><tr><td>MemoryOS</td><td>47.1</td><td>56.1</td><td>32.4</td><td>48.5</td><td>47.3</td><td>44.3</td></tr><tr><td>MemoryBank</td><td>35.3</td><td>49.1</td><td>51.1</td><td>56.6</td><td>40.0</td><td>46.5</td></tr><tr><td>EverMemOS</td><td>47.1</td><td>47.4</td><td>41.0</td><td>43.4</td><td>56.4</td><td>45.7</td></tr><tr><td>Emotional RAG</td><td>52.9</td><td>61.4</td><td>44.6</td><td>58.6</td><td>43.6</td><td>51.6</td></tr><tr><td>Mem-T</td><td>61.6</td><td>61.4</td><td>56.1</td><td>72.7</td><td>45.5</td><td>60.5</td></tr><tr><td>AiO-Mem</td><td>71.9</td><td>68.4</td><td>62.6</td><td>82.8</td><td>54.6</td><td>69.2</td></tr></table>

We further conduct experiments on PersonaMem (Jiang et al., 2025a). It evaluates the ability of memory systems to maintain and use dynamic user profiles over long-term interactions. As shown in Table 4, AiO-Mem achieves the best overall performance with an overall score of 0.692. In contrast, existing memory systems show less balanced performance across different capabilities. A-MEM performs well on reasoning and recommendation but is weaker on generalization and preference evolution. Mem-T achieves competitive performance on recall and generalization but performs less effectively on preference-aligned recommendation. These results suggest that the unified memory interaction in AiO-Mem enables effective retrieval, organization, and use of user-profile information from long-term interaction histories.

## D.3 MULTI-VIEW RETRIEVAL COMPLEMENTARITY STUDY

We further analyze the complementarity of the four retrieval views using queries generated during actual inference trajectories. After merging candidates across all views, 1,990 of 2,051 question– evidence pairs are detected by at least one view, yielding 97.03% candidate-pool coverage. As shown in Figures 7a and 7b, although evidence detected by all four views forms the largest group, substantial evidence is distributed across other view combinations, demonstrating their complementary coverage. Figure 7c further shows that single-view hits vary across question categories; for example, raw turns and query anchors contribute relatively more to Single-hop questions. These results indicate that different views capture distinct evidence patterns and jointly improve coverage across diverse question types.

![](images/0d6c3c3ab83e802bee880425548fd96d85abb34915a3ee8d44e3370282edcffe.jpg)

(a) Major view combinations  
![](images/92ffb6b1d7641c64874155282fcdddf7e9b61a1eb8e15563668ede6856a2a6f2.jpg)  
(b) Remaining view combinations

![](images/f266a0a72e0193271939ff8a9fbb37cfc2786b39ced4295108cadeb3ecd51f56.jpg)  
(c) Single-view hits by category  
Figure 7: Complementary evidence coverage across retrieval views. Panel a summarizes the major view combinations, panel b expands the remaining combinations, and panel c shows the sources of evidence detected by exactly one view across question categories.

![](images/502aadf213bf6e2889bdc52a3eb00e8dcc45de705597d86ba88c9592d3459f07.jpg)  
Figure 8: Sensitivity to retrieval modes and RRF parameters. Rows and columns denote the summary and raw-turn retrieval modes, respectively, while each panel uses a different aggregation RRF parameter. Red and blue indicate higher and lower evidence Recall.

Table 5: Correlations between Reward scores and evidence metrics.
<table><tr><td></td><td colspan="2">Recall</td><td colspan="2">Precision</td></tr><tr><td>Reward model</td><td>Pearson</td><td>Spearman</td><td>Pearson</td><td>Spearman</td></tr><tr><td>Qwen3-8B</td><td>0.877</td><td>0.783</td><td>0.640</td><td>0.675</td></tr><tr><td>AiO Reward Model (Ours)</td><td>0.958</td><td>0.800</td><td>0.686</td><td>0.861</td></tr></table>

![](images/83926691a88a0cdd790aaa27fd7e442ba2c78203dac7283f1c3aebbd5d0e01dc.jpg)  
(a) Reward vs. Recall

![](images/2a4f466a3a0c3ddbb87a4ff0506f7ead9b7c9d6f8fd0621e8635b984ee3c4ada.jpg)  
(b) Reward vs. Precision  
Figure 10: Sample-level Reward scores versus evidence metrics.

## D.4 MULTI-VIEW RETRIEVAL SENSITIVITY STUDY

The retrieval-budget analysis in Figure 9 characterizes the trade-off between evidence coverage and context cost, while Figure 8 examines sensitivity to retrieval configurations. As $K _ { \mathrm { r e t } }$ increases from 5 to 30, evidence Recall improves steadily from 73.62% to 90.82%, while the average retrieved context grows substantially from 2,197 to 11,137 tokens. At $K _ { \mathrm { r e t } } ^ { \mathrm { ^ { - } } } ~ = ~ 2 0$ , the system already achieves 88.16% Recall with 7,743 tokens; increasing the budget to 30 brings only a further 2.66-point Recall gain at approximately 44% higher context cost, indicating diminishing returns from larger retrieval budgets. We therefore set $K _ { \mathrm { r e t } } ~ = ~ 2 0$ as a practical balance between evidence coverage and context efficiency. Figure 8 further shows that moderate variations around the default retrieval modes and RRF parameters cause only limited changes in Recall, whereas larger deviations can lead to noticeable degradation. This suggests that the adopted retrieval configuration lies in a relatively stable and high-performing region rather than depending on a narrowly tuned parameter setting.

![](images/5e70600f91feb851a88223243bc00b063e4b7d63f8030daf635c422b5d8b1452.jpg)  
Figure 9: Trade-off between retrieval Recall and context cost.

## E REWARD AND OPTIMIZATION ANALYSIS

## E.1 UNIFIED REWARD MODEL ALIGNMENT STUDY

We further evaluate whether the trained reward model aligns with evidence usage by measuring its correlation with trajectory-level Recall and Precision. As shown in Table 5, training increases the Pearson correlation with Recall from 0.877 to 0.958 and the Spearman correlation with Precision from 0.675 to 0.861, indicating stronger alignment with evidence-based trajectory quality. Figures 9a and 9b further show that the base reward model assigns relatively concentrated scores across different evidence qualities, whereas the trained model exhibits clearer positive trends with both Recall and Precision. Notably, some trajectories still receive high rewards at different Precision levels, suggesting that the trained reward model captures broader trajectory quality beyond evidence Precision alone.

## E.2 TRAJECTORY-LEVEL DPO ALIGNMENT STUDY

Evidence-grounded SFT provides action-level supervision based on evidence coverage and provenance, but such rule-based signals cannot fully assess the semantic quality of an action within its trajectory context. As introduced in Section 3.4, trajectory-level DPO addresses this limitation by using full trajectory context to align the reward model with finer-grained semantic judgments. To visualize how trajectory-level DPO distinguishes trajectories beyond rule-based evidence matching, we compare four matched trajectory pairs for the same questions. Within each pair, the trajectories have identical Evidence-ID Recall and Precision but receive different scores after trajectory-level DPO alignment.

Specifically, in q114, both trajectories cite the correct evidence ID, but the lower-Reward trajec tory mixes visual details from another event into the adoption meeting, whereas the higher-Reward trajectory remains focused on the target event. In q76, both obtain zero Evidence-ID Recall and Precision, yet the higher-Reward trajectory retains evidence supporting “three children,” while the lower-Reward one infers “two children” from insufficient observations. In q65, despite complete evidence coverage, the lower-Reward trajectory repeats the same fact and conflates Melanie’s stated preference with the inference about Vivaldi, whereas the higher-Reward trajectory separates the source fact from the inference. In q7, the lower-Reward trajectory mixes the target future camping plan with a past camping event, while the higher-Reward trajectory preserves the correct temporal focus. These cases show that trajectory-level DPO complements rule-based supervision by aligning the reward model with evidence attribution, relevance, factual organization, temporal consistency, and redundancy beyond Evidence-ID matching.

## Matched Trajectories for q114

Question: What did Caroline see at the council meeting for adoption?

Gold answer: Many people wanting to create loving homes for children in need.

Gold evidence: D8:9

Both trajectories: Evidence-ID Recall = 1.00; Precision = 1.00.

Lower Trajectory-level Reward Model Score

Score: 0.67

Case annotation. The notes retain the relevant observation about prospective adoptive families, but also attach visual details from another event to the same meeting. Both statements cite the gold evidence ID, concealing the mismatch between the cited source and the additional claim.

• Caroline described the adoption council meeting as inspiring and emotional, noting that many people wanted to create loving homes for children in need. (Evidence: D8:9)

• Caroline saw a sign for a new arrival and an information and domestic building during the adoption council meeting. (Evidence: D8:9)

## Higher Trajectory-level Reward Model Score

Score: 0.95

Case annotation. The notes consistently describe the adoption meeting and the people seeking to provide loving homes. Although the two entries overlap, both remain focused on what Caroline observed at the target event, without introducing the unrelated visual details.

• Caroline attended an adoption council meeting and saw people seeking to create loving homes for children in need. (Evidence: D8:9)

• Caroline described the adoption council meeting as inspiring and emotional, with attendees focused on creating loving homes for children in need. (Evidence: D8:9)

## Matched Trajectories for q76

Question: How many children does Melanie have?

Gold answer: Three.

Gold evidence: D18:1, D18:7

Both trajectories: Evidence-ID Recall = 0.00; Precision = 0.00.

Lower Trajectory-level Reward Model Score

Score: 0.27

Case annotation. The notes use observations of two children in particular scenes to support a claim about Melanie’s total number of children. Seeing two children at a campfire or museum does not establish that she has only two, leaving the numerical conclusion insufficiently supported.

• Melanie has two children based on the image of a man and two children in the campfire scene. (Evidence: D8:32)

• Melanie took two children to the museum. (Evidence: D6:4)

Higher Trajectory-level Reward Model Score

Score: 0.53

Case annotation. The notes preserve an observation involving three children and additional family context. Neither entry matches the annotated gold evidence IDs, but the beach observation is more relevant to the reference answer. The displayed notes alone still do not conclusively establish the total number of Melanie’s children.

• Melanie went to the beach with three children. (Evidence: D10:8)

• Melanie’s children were at the Grand Canyon with her. (Evidence: D18:5)

## Matched Trajectories for q65

Question: Would Melanie likely enjoy the song “The Four Seasons” by Vivaldi?

Gold answer: Yes; it is classical music.

Gold evidence: D15:28

Both trajectories: Evidence-ID Recall = 1.00; Precision = 1.00.

Lower Trajectory-level Reward Model Score

Score: 0.55

Case annotation. The notes repeat Melanie’s classical music preference and conflate it with an inferred preference for Vivaldi.

• Melanie enjoys classical music including composers like Bach and Mozart. (Evidence: D15:28)

• Melanie enjoys classical music including composers like Bach and Mozart. (Evidence: D15:28)

• Melanie appreciates classical music genres. (Evidence: D15:28)

• Melanie appreciates classical music genres including composers like Bach, Mozart, and implicitly Vivaldi. (Evidence: D15:28)

## Higher Trajectory-level Reward Model Score

Score: 0.95

Case annotation. The notes distinguish Melanie’s stated preference, the genre connection, and the inference that she would likely enjoy Vivaldi.

• Melanie enjoys classical music by Bach and Mozart, as well as modern music like Ed Sheeran’s “Perfect”. (Evidence: D15:28)

• Vivaldi’s “The Four Seasons” is a classical composition, and Melanie appreciates classical music by Bach and Mozart. (Evidence: D15:28)

• Melanie’s appreciation for classical music suggests she would likely enjoy Vivaldi’s “The Four Seasons”. (Evidence: D15:28)

## Matched Trajectories for q7

Question: When is Melanie planning on going camping?

Gold answer: June 2023.

Gold evidence: D2:7

Both trajectories: Evidence-ID Recall = 1.00; Precision = 0.50.

Lower Trajectory-level Reward Model Score

Score: 0.54

Case annotation. The notes repeatedly alternate between a future camping plan and a separate camping trip that had already occurred. Although the relevant expression “next month” eventually appears, repetition and the interleaving of different events make the target plan less prominent.

• Melanie plans a camping trip for summer break. (Evidence: D2:7)

• Melanie went camping with kids a few weeks ago. (Evidence: D16:2)

• Melanie plans a camping trip for summer break. (Evidence: D2:7)

• Melanie went camping with kids a few weeks ago. (Evidence: D16:2)

• Melanie plans a camping trip for next month during summer break. (Evidence: D2:7)

## Higher Trajectory-level Reward Model Score

Score: 0.92

Case annotation. The notes foreground the upcoming trip and its time expression, “next month,” followed by broader summer-outing context. They avoid repeatedly mixing the plan with the completed camping event, making the information relevant to the temporal question more accessible.

• Melanie mentioned planning a camping trip for next month during summer break. (Evidence: D2:7)

• Melanie agreed to plan a family outing this summer, which includes camping activities. (Evidence: D12:18)

![](images/1736338fb5a3df46a158fd91e9d5d7da564c50d7f3126025a8af894b27ffd5be.jpg)  
(a) Overall Reward

![](images/caa1e4d7c2293fe40dcf7ccb0bdddb5a03a313cc20db6c6508730fbcb5b74f73.jpg)  
(b) Search Reward

![](images/1fd27fce8c1db35b18b01f527198833cdbb5ac9e4b83ec4c8cb4365377527018.jpg)  
(c) Extract Reward

![](images/a1b7d0f52c5d953eb7ec87a104b4fd9e5fd46e0848e170c02e0f3c79b8ba7264.jpg)  
(d) Answer-Decision Reward  
Figure 11: Training dynamics.

## E.3 TRAINING DYNAMICS

Figure 11 presents the Reward dynamics during PA-DAPO training. As shown in Figure 11a, the Overall Reward exhibits a clear upward trend as training progresses. Although the Reward varies across individual training steps, the smoothed curve increases overall, indicating a gradual improvement in the quality of the sampled trajectories.

Figures 11b–11d further decompose the Overall Reward by operation: Search corresponds to search, Extract includes add and skip, and Answer-Decision corresponds to stop.The Search Reward shows an overall increase, indicating improved evidence retrieval behavior. The Extract Reward also increases over training, reflecting improvements in evidence extraction and organization. The Answer-Decision Reward exhibits a similar upward trend, indicating that the model progressively learns when to stop retrieving and proceed to answer generation.

The overall improvement across all three stages indicates that the increase in Overall Reward is not driven by a single interaction stage. Instead, it reflects joint improvements in evidence retrieval, evidence extraction and organization, and the decision to stop retrieving and generate an answer.

## E.4 PROCESS-ALIGNED EVIDENCE FLOW STUDY

We first analyze how training affects evidence flow through the memory pipeline. Figures 12a, 12b, and 12c compare the trained and untrained systems at the Search, Final working memory, and Answer stages, respectively, with further breakdowns by question category.

After training, the system covers more gold evidence during Search and retains more evidence in the Final working memory, with improvements across all question categories. The final Answer F1 also increases accordingly. These results show that policy training improves the entire information processing pipeline, from evidence retrieval and organization to answer generation.

![](images/47b820cd6b7232a69f225a7bc1d90ce85bef649732df1dfde3fbd913ed9d3541.jpg)  
(a) Search

![](images/eb9d712da09567ac3d6152b4c41d7dbf52e3e0073b3dbc1352fc8f07787f1d17.jpg)  
(b) Final working memory

![](images/678da5e1f651220207301fae771311aee8c7c0e3d03d583722d6f06597902cba.jpg)  
(c) Answer  
Figure 12: Effects of policy training on evidence flow through the memory pipeline. The three panels report Search Recall, Final working memory Recall, and Answer F1, respectively.

## F COMPUTATIONAL COMPLEXITY

Table 6: Complexity comparison between AiO-Mem and OMS.
<table><tr><td>System</td><td>(I) Segment &amp; Retrieval</td><td>(II) Construction</td><td>(III) Inference</td></tr><tr><td>OMS</td><td> $O ( N ) \cdot \mathcal { C } _ { \mathrm { s e g } }$   $+ O ( N \ell ) \cdot \mathcal { C } _ { \mathrm { L L M } }$   $+ O ( N M _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { e m b } }$ </td><td> $O ( N M _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { s i m } }$   $+ O ( N M _ { 1 } R _ { 1 } ) \cdot \left( \mathcal { C } _ { \mathrm { L L M } } + \mathcal { C } _ { \mathrm { e m b } } \right)$ </td><td> $O ( N \ell + N M _ { 1 } R _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { L L M } }$   $+ O ( N M _ { 1 } ( 1 + R _ { 1 } ) ) \cdot \mathcal { C } _ { \mathrm { e m b } }$   $+ O ( N M _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { s i m } }$ </td></tr><tr><td>AiO-Mem</td><td> $O ( T _ { \mathrm { m a x } } M ) \cdot { \mathcal { C } } _ { \mathrm { s i m } }$   $+ O ( N \ell ) \cdot \mathcal { C } _ { \mathrm { L L M } }$   $+ O ( M + T _ { \mathrm { m a x } } ) \cdot \mathcal { C } _ { \mathrm { e m b } }$ </td><td> $O ( T _ { \mathrm { m a x } } K _ { \mathrm { r e t } } \ell _ { e } ) \cdot \mathcal { C } _ { \mathrm { L L M } }$ </td><td> $O ( N \ell + T _ { \mathrm { m a x } } K _ { \mathrm { r e t } } \ell _ { e } ) \cdot \mathcal { C } _ { \mathrm { L L M } }$   $+ O ( M + T _ { \mathrm { m a x } } ) \cdot { \mathcal { C } } _ { \mathrm { e m b } }$   $+ O ( T _ { \operatorname* { m a x } } M ) \cdot \mathcal { C } _ { \mathrm { s i m } }$ </td></tr></table>

As shown in Table $^ { 6 , }$ we analyze a long-horizon history $\mathcal { H } = ( h _ { 1 } , \ldots , h _ { N } )$ consisting of $N$ source units, each containing an average of ℓ tokens, and evaluate the cost of answering a memory query q. In other memory systems (OMS), the history is first segmented into source units, where each unit requires an LLM-based summarization and is further encoded into $M _ { 1 }$ persistent memory entries:

$$
\mathcal { C } _ { \mathrm { O M S } } ^ { \mathrm { I } } = O ( N ) \cdot \mathcal { C } _ { \mathrm { s e g } } + O ( N \ell ) \cdot \mathcal { C } _ { \mathrm { L L M } } + O ( N M _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { e m b } } .\tag{6}
$$

During memory construction, the resulting $N M _ { 1 }$ entries are compared with the existing memory. A fraction $R _ { 1 }$ of these entries triggers an LLM-based rewrite, followed by re-embedding for index update:

$$
\mathcal { C } _ { \mathrm { O M S } } ^ { \mathrm { I I } } = O ( N M _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { s i m } } + O ( N M _ { 1 } R _ { 1 } ) \cdot \left( \mathcal { C } _ { \mathrm { L L M } } + \mathcal { C } _ { \mathrm { e m b } } \right) .\tag{7}
$$

This results in a total inference cost of:

$$
\begin{array} { r l } & { \mathcal { C } _ { \mathrm { O M S } } ^ { \mathrm { i n f e r } } = O ( N \ell + N M _ { 1 } R _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { L L M } } } \\ & { \qquad + O ( N M _ { 1 } ( 1 + R _ { 1 } ) ) \cdot \mathcal { C } _ { \mathrm { e m b } } + O ( N M _ { 1 } ) \cdot \mathcal { C } _ { \mathrm { s i m } } . } \end{array}\tag{8}
$$

Thus, both the LLM-based summarization and memory rewriting costs increase with the history size and the number of accumulated memory entries.

In AiO-Mem, the history is first compiled into the traceable event bank $\begin{array} { r } { B = \mathcal { C } ( \mathcal { H } ) } \end{array}$ . The resulting M events and their indices are reused across queries. Since each trajectory contains at most $T _ { \mathrm { m a x } }$ interaction steps, the segmentation and retrieval cost is:

$$
\mathcal { C } _ { \mathrm { A i O } } ^ { \mathrm { I } } = O ( N \ell ) \cdot \mathcal { C } _ { \mathrm { L L M } } + O ( M + T _ { \mathrm { m a x } } ) \cdot \mathcal { C } _ { \mathrm { e m b } } + O ( T _ { \mathrm { m a x } } M ) \cdot \mathcal { C } _ { \mathrm { s i m } } .\tag{9}
$$

On top of this, AiO-Mem keeps B unchanged and constructs only the query-specific working memory $\mathcal { W } _ { t }$ . Each interaction step operates on at most $K _ { \mathrm { r e t } }$ retrieved events with an average exposed length of $\ell _ { e }$ tokens. Therefore, the construction cost is:

$$
\mathcal { C } _ { \mathrm { A i O } } ^ { \mathrm { I I } } = O ( T _ { \mathrm { m a x } } K _ { \mathrm { r e t } } \ell _ { e } ) \cdot \mathcal { C } _ { \mathrm { L L M } } .\tag{10}
$$

![](images/6343c2df72640a34e61124ddc964c9f70bb04dcd9d8abb31d51aaddc35cc3d43.jpg)  
(a) FLOPs comparison on LoCoMo.

![](images/1674245273289ff340d3ccb836682f590e78f1084796a17be4914e2940284a83.jpg)  
(b) Token comparison on LoCoMo.  
Figure 13: Efficiency comparison between AiO-Mem and Baseline on LoCoMo.

Table 7: Notation used in the complexity analysis.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $\mathcal { H } = ( h _ { 1 } , \ldots , h _ { N } )$ </td><td>Long-horizon history consisting of N heterogeneous source units.</td></tr><tr><td> $N$ </td><td>Number of source units in H.</td></tr><tr><td> $\ell$ </td><td>Average number of tokens in each source unit  $h _ { n } .$ </td></tr><tr><td> $\begin{array} { r } { B = \mathcal { C } ( \mathcal { H } ) } \end{array}$ </td><td>Traceable event bank constructed from H by the structure-aware compiler C.</td></tr><tr><td> $M$ </td><td>Number of traceable events in the event bank B.</td></tr><tr><td> $M _ { 1 }$ </td><td>Number of persistent memory entries produced per source unit by OMS.</td></tr><tr><td> $R _ { 1 }$ </td><td>Fraction of OMS memory entries that trigger memory rewriting and re-indexing.</td></tr><tr><td> $K _ { \mathrm { r e t } }$ </td><td>Number of candidate events returned by each SEARCH action.</td></tr><tr><td> $\ell _ { e }$ </td><td>Average number of tokens exposed by one retrieved event.</td></tr><tr><td> $T _ { \mathrm { m a x } }$ </td><td>Maximum interaction horizon, with  $\dot { T } \le T _ { \mathrm { m a x } }$ </td></tr><tr><td> $\mathcal { C } _ { \mathrm { s e g } }$ </td><td>Cost of one segmentation operation.</td></tr><tr><td> $\mathcal { C } _ { \mathrm { L L M } }$ </td><td>Cost of LLM computation.</td></tr><tr><td> $\mathcal { C } _ { \mathrm { e m b } }$ </td><td>Cost of an embedding operation.</td></tr><tr><td> ${ \mathcal { C } } _ { \mathrm { s i m } }$ </td><td>Cost of a sparse or dense similarity operation.</td></tr></table>

Combining event-bank construction, multi-view retrieval, bounded working-memory construction, and final response generation, AiO-Mem requires

$$
\begin{array} { r l } & { \mathcal { C } _ { \mathrm { A i O } } ^ { \mathrm { i n f e r } } = O ( N \ell + T _ { \mathrm { m a x } } K _ { \mathrm { r e t } } \ell _ { e } ) \cdot \mathcal { C } _ { \mathrm { L L M } } } \\ & { \qquad + O ( M + T _ { \mathrm { m a x } } ) \cdot \mathcal { C } _ { \mathrm { e m b } } + O ( T _ { \mathrm { m a x } } M ) \cdot \mathcal { C } _ { \mathrm { s i m } } . } \end{array}\tag{11}
$$

Following prior efficiency-oriented works, we report FLOPs measured by (Ye, 2023) and token consumption, using Mem-T as the baseline. As shown in Figure 13, AiO-Mem exhibits substantially slower FLOPs growth with increasing context length. By decoupling large-scale persistent memory management from query-time working-memory construction, AiO-Mem avoids repeated memory rewriting and re-indexing and confines costly LLM computation to selected evidence. It further reduces retrieval and construction tokens by 79.9% and 73.8%, respectively.

## G PROMPTS AND TOOL CALLS

## G.1 MEMORY AGENT TOOL CALLS

The LoCoMo memory agent alternates between DECIDE and EXTRACT. DECIDE chooses between search and stop; after each search, EXTRACT chooses between add, which writes visible evidence into working memory, and skip.Table 8 lists the tools available at each stage.

Table 8: Tool calls available to the memory agent.
<table><tr><td>Stage</td><td>Tool call and function</td></tr><tr><td></td><td>DECIDE search_memory_units (query) : Retrieves event-level memory units concerning information missing from the current notes.</td></tr><tr><td></td><td>DECIDE answer_from_notes () (no arguments): Ends memory retrieval and proceeds to the answering stage using the existing working memory.</td></tr><tr><td></td><td>EXTRACTadd_note(event_id, datetime, content, evidence_turns): Saves aclue, its event time, and the original dialogue supporting it. An EXTRACT step may invoke add_note multiple times, with each invocation writing one note.</td></tr><tr><td></td><td>EXTRACTskip_ext raction (reason) : Skips extraction when the current search results contain no relevant evidence to write.</td></tr></table>

## G.2 LOCOMO MEMORY AGENT PROMPTS

Table 9 presents condensed prompt templates for the two model-call stages. Each call includes only the tool definitions permitted at that stage; the model returns tool calls as JSON objects enclosed in <tool call> tags.

Table 9: Prompt templates for the LoCoMo memory agent.
<table><tr><td>Stage</td><td>Prompt template</td></tr><tr><td>DECIDE</td><td>You are a memory assistant. Given the current question and saved working memory, choose the next action: search memory, or stop retrieval and proceed to the answering stage. Rules: (1) Judge whether the question can be answered completely using only the working memory, including the relationships and all necessary parts requested by the question. If the evidence is sufficient, call answer_from_notes. (2) Otherwise, generate a query targeting one specific fact or clue that is still missing, and call search_memory-units. The query should identify the relevant person or object, relationship, and information needed. Avoid repeating previous searches that produced no useful notes. (3) At the final decision step, call answer_from_notes. Input: Question {question}, working memory {working_notes}, previous queries {previous_searches}, and step {step}/{max_steps}.</td></tr><tr><td>EXTRACT</td><td>Output: A brief explanation of the decision and one &lt;t ool_ca11&gt;. You are a memory system analyst. Given the question, the latest search results, and the existing working memory, identify new evidence useful for answering the question and write it into the working memory. Search results are temporary; useful information not written into the notes will not be available to subsequent decisions. Rules: (1) Read the visible original dialogue in retrieved events. Highlighted turns are only hints; relevant evidence may also appear in other turns. (2) For each new useful clue, call add_note to write a concise, faithful note, preserving the exact event ID, recorded time, and original dialogue supporting its content. Each note should capture one natural clue and should be compared with existing notes to avoid duplication. (3) If the search results contain no relevant evidence, call skip_extraction. Input: Question {question}, search observation {search_observation }, and working memory {working_notes}. The search observation contains event identifiers, recorded times, retrieved fields, summaries, and visible original dialogue. Output: A brief assessment of the evidence and the required &lt;t ool_cal1&gt;; distinct clues</td></tr></table>

## G.3 STEPWISE REWARD MODEL PROMPT

For SFT, the prompt below is the reward model input, with rule-derived scores as targets. For DPO, the same prompt is given to both the reward model and DeepSeek-V4-Flash; DeepSeek’s score sequence serves as chosen and the rule-derived sequence as rejected. Reference answers and annotated evidence are used only for reward scoring, not by the memory policy or answer model. Table 10 summarizes the prompt’s objectives, error categories, score ranges, and output format.

Table 10: Prompt template and scoring rules for the stepwise reward model.
<table><tr><td>Prompt template</td></tr><tr><td>Task and input You are a reward model evaluator of memory trajectories. Given a complete scoreable memory trajectory, evaluate each step according to whether the required evidence was found, whether useful evidence was written into working memory, and whether retrieval stopped when the evidence was sufficient. You may use trajectory context to understand the current action, but each score must evaluate only the current step; do not evaluate the</td></tr><tr><td>quality of the final answer model&#x27;s response. Input: {question}, {reference_answer}, {gold_evidence}, {trajectory_information}, and {scoreable_steps}. Objectives by action type</td></tr><tr><td>deci de_search: Does the query target information missing from the current notes and help locate evidence needed to answer the question? A reasonable query should not receive the lowest score solely because it failed to retrieve a match.</td></tr><tr><td>ext ract _add_note: Does the action faithfully write useful, previously unsaved evidence from the latest search into the notes, and do the cited turns support the note&#x27;s content? ext ract _skip: Does the latest search truly contain nothing worth writing? Skipping clearly critical</td></tr><tr><td>evidence is a severe problem. decide_answer_from_notes: Are the existing notes sufficient to support an answer when retrieval stops? Evidence assessment</td></tr><tr><td>When judging whether the notes are sufficient, consider both their content and the original dialogue they cite. Annotated evidence indicates information needed for the question, but it is not a mechanical hit-rate scoring formula: unannotated dialogue turns may also support the answer semantically, and matching annotated turns does not automatically warrant a high score.</td></tr><tr><td>Error severity</td></tr><tr><td>Category A, high severity: Failure of the current stage&#x27;s core objective. Category B, high severity: Factual or attribution errors in a note, or citations that do not support key content. Category C, medium severity: Problems in assessing the current state or respecting stage boundaries.</td></tr><tr><td>Category D, medium severity: An incomplete contribution or substantial noise in the action. Category E, low severity: Wording or formatting issues. Score ranges 90–100: The action fulfills the stage&#x27;s core objective very well, with at most minor Category E issues. 75–89: The action is clearly useful and largely fulfills the core objective; it has no Category A or B issues,</td></tr><tr><td>though minor Category C, D, or E issues are allowed. 60–74: The action makes a partial contribution but falls short of the core objective; it has no Category A or B issues, though Category C or D issues are more pronounced.</td></tr></table>