# CoMem: Collective-Individual Memory Synergy for Evolutionary Multi-Agent Systems

Chengxin Yu<sup>1</sup>, Zhaoxin Fan<sup>1,</sup> Faguo Wu<sup>1</sup>, Hongwei Zheng<sup>2</sup>,

Yun Zhou<sup>3</sup>, Zhiyu Li<sup>4</sup>

<sup>1</sup> Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, School of Artificial Intelligence, Beihang University

<sup>2</sup> Beijing academy of blockchain and edge computing

<sup>3</sup> National University of Defense Technology

<sup>4</sup> MemTensor (Shanghai) Technology Co., Ltd.

## Abstract

Designing efective memory mechanisms is crucial for advancing LLM-driven Multi-Agent Systems (MAS), helping agents learn together and perform better over time. While recent work has led to strong cooperation skills, most methods still use flat, unstructured memories, which easily get filled with noise and erase diferences between agents. To address this, we introduce the concept of collective-individual memory synergy and propose CoMem, an architecture that unifies both private experience and shared knowledge for multi-agent learning. CoMem features: (i) Private Experience Sedimentation, which lets each agent keep and update its own useful memories over time; (ii) Collective Wisdom Curation, which carefully selects only widely proven ideas to be shared among agents; (iii) Parallel Dual-Stream Retrieval, which allows agents to draw both from their own memory and the group’s wisdom, using clustering to ensure diversity. Experiments on ALFWorld and PDDL benchmarks show that CoMem achieves strong overall performance and robustly avoids memory pollution.

## 1 Introduction

Multi-Agent Systems (MAS) have emerged as a key paradigm for tackling complex, real-world problems that require coordination, cooperation, and collective problemsolving(Wooldridge 2009). Recent advances in Large Language Models (LLMs)(Yao et al. 2023; Madaan et al. 2023; Xi et al. 2025) have greatly enhanced these systems, enabling autonomous agents to reason, plan, and interact more efectively in diverse and dynamic environments(Wu et al. 2024; Liu et al. 2024; Qian et al. 2025; Guo et al. 2024; Wang et al. 2024b; Han et al. 2024; Du et al. 2024; Wang et al. 2024a). As MAS are deployed in increasingly challenging scenarios, it becomes essential for agents not only to act collaboratively, but also to learn from past experiences—highlighting the critical role of efective memory mechanisms in supporting continual adaptation and long-term group intelligence, which we call group memory(Xu et al. 2025; Chhikara et al. 2025; Wang et al. 2025).

This importance of group memory in MAS is evident in everyday life. For instance, people who come from the same cultural background—having heard the same childhood stories or traditions—often find it easier to understand each other and work together. Such collective memories create a sense of belonging and common ground, fostering greater trust and cohesion when tackling problems as a group(Khushiyant 2025; Woolley et al. 2010).

![](images/5f17c1453ccb862e93ed8e42d793b6d20caf25adfb4fc52cff49ed121151dd7b.jpg)  
Figure 1: Illustration of memory pollution in MAS. We conceptualize the problem as comprising two intertwined degradation processes: (a) noise accumulation, where unverified trial-and-error behaviors and low-quality decisions overwhelm the shared repository, and (b) behavioral homogenization, where heterogeneous agents lose their role-specific characteristics through blind cross-writing. These two processes mutually reinforce each other, ultimately driving the collective toward mediocrity.

Recent work has begun to explore memory mechanisms for MAS, such as introducing shared memory spaces that enable agents to store and retrieve strategic experiences across tasks (Zhang et al. 2025a). However, these approaches typically employ simple, flat memory structures without explicitly modeling the concept of group memory or considering the synergy between collective and individual memories outlined above(Zhang et al. 2025b; Liu et al. 2025; Du 2026). As a result, such designs often overlook the need for coordination and quality control in shared memory, making them susceptible to memory pollution—a phenomenon illustrated in Figure 1. This “memory pollution” mainly appears in two forms: (i) noise accumulation, where unverified or low-quality agent behaviors flood the shared memory and reduce its reliability; and (ii) behavioral homogenization, where agents lose their individual specializations as their experiences are repeatedly averaged out(Wang et al. 2026; Torra and Bras-Amorós 2026). These issues reinforce each other and often lead to worse performance in flat memory designs, rather than the intended improvement.

To tackle the issue, we argue that sustainable collective intelligence in multi-agent systems requires more than shared memory: it demands a careful balance between robust individual learning and meaningful group knowledge to construct group memory(Du 2026; Rezazadeh et al. 2025; Xu et al. 2025). Mirroring the dynamics of human societies, we hypothesize that efective population-level memory emerges only when individual agents are able to learn, test, and refine experiences in isolation, while genuine consensus is distilled through selective sharing and rigorous screening. In this way, collective progress is grounded in the diversity and integrity of private exploration, with only the most reliable insights crystallizing at the group level to inform future decisionmaking.

Inspired by the knowledge evolution process observed in human communities—“individual proposal, collective screening, consensus crystallization” (Bruggeman 2023)—we model group memory as a tightly coupled ecosystem with two distinct components: a private layer for interference-free accumulation of agent-specific experience, and a curated collective pool for distilled group wisdom. These layers interact through a structured promotion pathway: private memory retains only frequently validated knowledge through usage-aware rolling pruning, while collective memory admits new insights only after repeated, cross-agent verification. This bidirectional flow forms a virtuous cycle, where trustworthy private discoveries are promoted upward and refined collective knowledge guides subsequent individual behavior.

To operationalize this principle, we introduce CoMem, a novel memory architecture designed to support continual agent learning while suppressing memory pollution. CoMem centers on three synergistic mechanisms: (i) Private Experience Sedimentation with rolling pruning, allowing each agent to preserve and refine a pure history of its own validated experiences; (ii) Collective Wisdom Curation, employing a stringent promotion process so that only knowledge proven robust across tasks enters the shared pool; and (iii) Parallel Dual-Stream Retrieval, enabling each agent to blend private and collective memories for decision-making, while maintaining diversity through clustering-based filtering(Lu et al. 2026). Together, these components establish a structured and dynamic memory ecosystem that enables both individual exploration and lasting group intelligence. Experiments on ALFWorld and PDDL benchmarks demonstrate the efectiveness of CoMem, achieving consistent gains over strong memory-augmented MAS baselines.

Our main contributions are as follows:

• We propose CoMem, a novel memory architecture designed to mitigate memory pollution in multi-agent systems with flat group memory.

• CoMem introduces a collective-individual memory synergy with key components: private experience sedimentation, collective wisdom curation, and parallel dualstream retrieval, structurally balancing individual and group learning.

• Extensive experiments on ALFWorld and PDDL benchmarks demonstrate that CoMem consistently improves over strong baselines while efectively mitigating memory pollution.

## 2 Related Work

## 2.1 LLM-based Multi-Agent Systems

Recent advances have expanded LLMs from single-turn language generators into autonomous agents capable of independent reasoning, planning, and tool utilization (Brooks 1991; Yao et al. 2023; Xi et al. 2025). Building on this individual empowerment, MAS orchestrate multiple LLM agents to tackle intricate tasks via collaboration(Du et al. 2024; Wang et al. 2024a). Frameworks like AutoGen (Wu et al. 2024) and CAMEL (Li et al. 2023) introduce customizable agent communication topologies, while ChatDev (Qian et al. 2024) and MetaGPT (Hong et al. 2024) simulate software development workflows using role-specific agents. To optimize collaborative dynamics, DyLAN (Liu et al. 2024) dynamically selects agents based on structural importance, AgentVerse (Chen et al. 2024) explores emergent behavioral patterns in groups, and MacNet (Qian et al. 2025) and CARD (Wu et al. 2026) pattern agent interactions after specialized multi-agent structures and conditional topologies. Despite their remarkable designs, existing methods largely overlook group memory mechanisms: they discard interaction histories after each task and fail to preserve or leverage shared strategic knowledge. In this work, we focus on developing group memory for MAS to enable continual learning and collaboration across tasks.

## 2.2 Memory Mechanisms for Autonomous Agents

Memory serves as a critical pillar for lifelong learning and continuous adaptation in autonomous agents (Sumers et al. 2024; Xu et al. 2025; Chhikara et al. 2025; Wang et al. 2025; Zhao et al. 2024). In single-agent setups, frameworks like MemGPT (Packer et al. 2023) emulate OS-level virtual memory hierarchies to bypass context limitations. Reflexion (Shinn et al. 2023) utilizes linguistic reinforcement to remember structural failures, HippoRAG (Jiménez Gutiérrez et al. 2024) performs neurobiologically inspired associative retrieval, and Voyager (Wang et al. 2023) develops a persistent skill library to navigate open-world environments. However, translating single-agent memory into MAS poses immense friction due to cross-agent interference. Unrestricted shared configurations, as seen in MemoryBank (Zhong et al. 2024) or simple generative agent shared logs (Park et al. 2023), lead to massive information dilution. While G-Memory (Zhang et al. 2025a) attempts to structure this data using multi-layered graphs, it maintains a flat read/write structure across agents. Recent works have highlighted the critical drawbacks of such shared architectures, such as information contamination (Wang et al. 2026; Torra and Bras-Amorós 2026) and context homogenization (Zhang et al. 2025b). This paper overcomes this limitation by upgrading the memory mechanism to group memory (Rezazadeh et al. 2025; Du 2026).

![](images/2470238d7312b8daaf1cad42b4b023b1fa6a393e7e78417adca3fc556063b5ae.jpg)  
Figure 2: The overall structural overview of the CoMem architecture. Agents maintain isolated Private Memory slots and share a Collective Memory Pool via a stringent empirical promotion gate.

## 3 Methodology

## 3.1 Overview

Let $\mathcal { T } ~ = ~ \{ T _ { 1 } , T _ { 2 } , \ldots , T _ { N } \}$ be a sequence of collaborative tasks drawn from a dynamic execution environment. A MAS consisting of a set of autonomous agents ${ \mathcal { A } } =$ $\{ a _ { 1 } , a _ { 2 } , \dotsc , a _ { | \mathcal { A } | } \}$ is deployed to solve these tasks sequentially. For each task $T _ { k } ,$ , the agents collaborate over an episodic horizon of execution steps.

During the execution of task $T _ { k }$ , an agent $a _ { i } \in { \mathcal { A } }$ generates a localized execution trace

$$
\tau _ { i } ^ { k } = \{ ( s _ { 1 } , x _ { 1 } , y _ { 1 } ) , \ldots , ( s _ { t } , x _ { t } , y _ { t } ) \} ,
$$

which encapsulates environmental states $s ,$ individual chainof-thought rationales $x ,$ and executed actions y.

In conventional shared-memory MAS frameworks, al agents indiscriminately write their episodic summaries into a monolithic, flat collective repository $\mathcal { M } _ { \mathrm { s h a r e d } }$ . We formalize the Memory Pollution Problem as follows (Wang et al. 2026; Torra and Bras-Amorós 2026): Let $\mathcal { M } _ { t }$ be the state of the shared memory at time step t. The quality of $\mathcal { M } _ { t }$ degrades over time due to two primary noise sources:

(1) Execution Noise Dilution: unverified trial-and-error behaviors and circumstantial successes contaminate the repository (Wang et al. 2026).

(2) Specialization Homogenization: cross-agent overwriting forces role-specific cognitive trajectories to conform to a noisy average (Zhang et al. 2025b), such that

$$
\operatorname* { l i m } _ { t \to \infty } H ( \pi _ { a _ { i } } \parallel \pi _ { a _ { j } } ) = 0 , \quad \forall a _ { i } , a _ { j } \in \mathcal { A } .
$$

The objective of CoMem is to maintain a decoupled memory space

$$
\mathcal { M } = \left. \{ \mathcal { M } _ { \mathrm { p r i v } } ^ { ( i ) } \} _ { i = 1 } ^ { | \mathcal { A } | } , \mathcal { M } _ { \mathrm { c o l l } } \right.
$$

such that individual learning purity is preserved in the isolated private stores $\mathcal { M } _ { \mathrm { p r i v } } ^ { ( i ) }$ (Rezazadeh et al. 2025), while organization-level general wisdom is filtered and sustained in the pristine global store $\mathcal { M } _ { \mathrm { c o l l } }$ (Du 2026).

Figure 2 illustrates the overall structure of CoMem. At its core, CoMem maintains two logically and physically isolated memory tiers: a Private Layer and a Collective Pool. The Private Layer consists of exclusive collections, each dedicated to a single agent, ensuring that no cross-agent contamination can occur at the storage level. The Collective Pool is a shared repository that stores only those insights that have passed a rigorous empirical validation gateway.

The lifecycle of each memory unit is governed by a unified metadata schema. Every memory entry m is stored as a structured tuple:

$$
m = \langle { \mathrm { d o c \_ c o n t e n t , m e t a d a t a } } \rangle
$$

where metadata = {agent\_name, $S , n _ { \mathrm { u s e } } , n _ { \mathrm { s u c c } } , \iota _ { \mathrm { p r o m } } \}$ , representing the owner agent, utility score, invocation counter, successful invocation counter, and promotion status, respectively.

Memory flows through the system along a bidirectional pathway. New experiences are first generated by individual agents and stored in their private collections. These entries undergo rolling pruning based on usage recency: frequently retrieved entries are retained, while entries that remain unused beyond a threshold are systematically evicted. When an entry demonstrates suficient empirical success, it qualifies for promotion and is replicated into the Collective Pool. Promoted entries are then accessible to all agents, while collective entries that fail to maintain their utility over time are purged from the pool. This design ensures that the Private Layer remains compact and role-specific, while the Collective Pool evolves into a curated repository of generalizable, high-value organizational wisdom.

Next, we introduce the core components Private Experience Sedimentation, Collective Wisdom Curation and Parallel Dual-Stream Retrieval in detail.

## 3.2 Private Experience Sedimentation

Building on the layered memory architecture outlined above, we first focus on how each agent preserves and refines its personal learning trajectory. At the conclusion of every episode, agent $a _ { i }$ extracts its own execution trace $\boldsymbol { \tau } _ { i } ^ { k }$ from the sequence of runtime states and decisions. Rather than storing these raw, noisy trajectories directly, $a _ { i }$ engages in a structured reflection process(Shinn et al. 2023; Zhao et al. 2024): a dedicated LLM-based distiller, operating from the agent’s unique perspective, transforms the episodic trace into a small set of succinct, actionable textual insights. Each insight distills a strategic lesson or behavioral pattern that captures individual learning value.

These distilled insights are then stored in the agent’s private memory collection, $\mathcal { M } _ { \mathrm { p r i v } } ^ { ( i ) }$ , each assigned an initial utility score $S _ { 0 }$ . Associated metadata counters—including usage count $n _ { \mathrm { u s e } } .$ , success count $n _ { \mathrm { s u c c } } .$ , and a promotion flag ι —are initialized to their default states, indicating that the entry is newly created and yet to be validated or considered for promotion. Importantly, $\mathcal { M } _ { \mathrm { p r i v } } ^ { ( i ) }$ enforces strict access control by filtering entries based on agent identity; this structural separation guarantees that no cross-agent contamination can occur, and preserves the integrity of each agent’s own reflective history.

To maintain the eficiency and relevance of private memory, each agent employs a rolling pruning strategy for nonpromoted entries(Zhong et al. 2024; Chhikara et al. 2025). Specifically, a consecutive miss counter $K _ { \mathrm { m i s s } } ( m )$ is assigned to every private memory item $m .$ After each episode, entries are evaluated for activity: if an insight m was retrieved and injected into a prompt during decision-making, $K _ { \mathrm { m i s s } } ( m )$ is reset to zero; otherwise, it is incremented by one. Once an entry’s miss counter surpasses a predefined threshold $K _ { \mathrm { m a x } } ,$ it is permanently removed from the private collection. This lightweight mechanism allows private memories to remain concise and focused by naturally filtering out stale or unused knowledge, even in the sparse and uncertain feedback setting typical of multi-agent environments.

Through this private sedimentation process, CoMem ensures that each agent can robustly preserve and refine its unique experiential base, laying the groundwork for both long-term individual expertise and reliable contributions to

collective memory.

## 3.3 Collective Wisdom Curation

Building upon the foundation of private experience sedimentation, CoMem ensures that only truly robust and generalizable insights are elevated to the organizational memory. To this end, we introduce a carefully designed empirical promotion gateway that rigorously filters candidate knowledge before admitting it into the collective pool.

For each private memory entry m generated by agent $a _ { i } .$ promotion is governed by an empirical, baseline-conditioned rule:

$$
\iota _ { \mathrm { p r o m } } = \mathbb { I } \Bigl [ \bigl ( N _ { \mathrm { u s e } } ( m ) \geq N _ { \mathrm { m i n } } \bigr ) \wedge \bigl ( \bar { R } _ { m } \geq \bar { R } \bigr ) \Bigr ]
$$

where $N _ { \mathrm { u s e } } ( m )$ counts the total number of episodes in which m has been successfully retrieved—enforcing a statistical floor $N _ { \mathrm { m i n } }$ to guard against incidental or anomalous activation. The empirical value $\bar { R } _ { m }$ captures the average task score achieved in episodes where m was actively used; $\bar { R }$ serves as an environment-wide moving baseline of historical average performance. This conjunctive criterion ensures that only entries demonstrating both suficient exposure and consistent, above-average impact are promoted. Upon satisfaction, m is replicated to the collective memory $\mathcal { M } _ { \mathrm { c o l l } }$ and removed from the originating private store, avoiding redundancy while underpinning group knowledge with individually vetted experience(Rezazadeh et al. 2025; Zhang et al. 2025a).

Once admitted, collective entries are not static: each is subject to continuous quality control via a hybrid utility assessment. At the end of every episode, for each retrieved collective entry, we compute a comprehensive utility score $U _ { t } ( m )$

$$
U _ { t } ( m ) = w \cdot R _ { \mathrm { n o r m } } + ( 1 - w ) \cdot \left( \frac { 1 } { | \boldsymbol { A } _ { m } | } \sum _ { a \in \boldsymbol { A } _ { m } } \frac { \mathrm { L L M } _ { a } ( m ) } { L _ { \mathrm { m a x } } } \right) ,
$$

where $R _ { \mathrm { n o r m } }$ is the normalized episodic reward, $A _ { m }$ is the subset of agents evaluating m, $\bar { \mathrm { L L M } } _ { a } ( m )$ is a subjective score from ofline LLM aggregation, and w sets the weighting between objective performance and subjective insight quality.

The long-term utility $S _ { t } ( m )$ is updated as an exponential moving average:

$$
S _ { t + 1 } ( m ) = ( 1 - \alpha ) S _ { t } ( m ) + \alpha \cdot \left( S _ { \operatorname* { m a x } } \cdot U _ { t } ( m ) \right) ,
$$

with smooth score adjustment via EMA update rate α and capped at $S _ { \mathrm { m a x } }$ . Any entry whose score persistently drops below an adaptive threshold is systematically removed, ensuring that the collective memory remains lean, relevant, and filled only with organizational wisdom proven valuable over time.

Through this stringent curation process, CoMem’s collective pool matures into a trusted knowledge base, continuously distilled and refined by empirical feedback—ensuring that only generalizable and enduring insights guide future multi-agent collaboration.

<table><tr><td rowspan="2">Memory Mechanism</td><td colspan="2">AutoGen</td><td colspan="2">DyLAN</td><td colspan="2">MacNet</td><td colspan="2">CARD</td></tr><tr><td>ALFWorld</td><td>PDDL</td><td>ALFWorld</td><td>PDDL</td><td>ALFWorld</td><td>PDDL</td><td>ALFWorld</td><td>PDDL</td></tr><tr><td>No-memory</td><td> $7 9 . 1 0 ^ { + 0 . 0 0 }$ </td><td> $6 9 . 5 2 ^ { + 0 . 0 0 }$ </td><td> $8 0 . 6 0 ^ { + 0 . 0 0 }$ </td><td> $6 7 . 7 8 ^ { + 0 . 0 0 }$ </td><td> $7 9 . 8 5 ^ { + 0 . 0 0 }$ </td><td> $6 0 . 7 8 ^ { + 0 . 0 0 }$ </td><td> $8 4 . 3 3 ^ { + 0 . 0 0 }$ </td><td> $7 0 . 0 1 ^ { + 0 . 0 0 }$ </td></tr><tr><td>Voyager</td><td> $7 6 . 8 7 \AA ^ { - 2 . 2 3 }$ </td><td> $6 7 . 6 0 ^ { - 1 . 9 2 }$ </td><td> $7 2 . 3 9 ^ { - 8 . 2 1 }$ </td><td> $5 8 . 1 7 ^ { - 9 . 6 1 }$ </td><td> $7 7 . 6 1 ^ { - 2 . 2 4 }$ </td><td> $5 2 . 0 7 ^ { - 8 . 7 1 }$ </td><td> $8 7 . 3 1 ^ { + 2 . 9 8 }$ </td><td> $5 1 . 9 2 ^ { - 1 8 . 0 9 }$ </td></tr><tr><td>MemoryBank</td><td> $7 0 . 9 0 ^ { - 8 . 2 0 }$ </td><td> $5 3 . 4 5 ^ { - 1 6 . 0 7 }$ </td><td> $7 1 . 6 4 ^ { - 8 . 9 6 }$ </td><td> $5 0 . 9 2 ^ { - 1 6 . 8 6 }$ </td><td> $7 3 . 1 3 ^ { - 6 . 7 2 }$ </td><td> $5 0 . 9 4 ^ { - 9 . 8 4 }$ </td><td> $7 7 . 6 1 ^ { - 6 . 7 2 }$ </td><td> $4 9 . 8 9 ^ { - 2 0 . 1 2 }$ </td></tr><tr><td>Generative</td><td> $7 2 . 3 9 ^ { - 6 . 7 1 }$ </td><td> $6 1 . 2 9 ^ { - 8 . 2 3 }$ </td><td> $6 5 . 6 7 ^ { - 1 4 . 9 3 }$ </td><td> $7 0 . 3 4 ^ { + 2 . 5 6 }$ </td><td> $7 3 . 1 3 ^ { - 6 . 7 2 }$ </td><td> $5 9 . 9 6 ^ { - 0 . 8 2 }$ </td><td> $9 0 . 3 0 ^ { + 5 . 9 7 }$ </td><td> $5 8 . 0 0 ^ { - 1 2 . 0 1 }$ </td></tr><tr><td>MetaGPT</td><td> $6 5 . 6 7 ^ { - 1 3 . 4 3 }$ </td><td> $6 9 . 8 5 ^ { + 0 . 3 3 }$ </td><td> $6 7 . 1 6 ^ { - 1 3 . 4 4 }$ </td><td> $5 2 . 8 0 ^ { - 1 4 . 9 8 }$ </td><td> $7 1 . 6 4 ^ { - 8 . 2 1 }$ </td><td> $5 0 . 8 3 ^ { - 9 . 9 5 }$ </td><td> $9 0 . 3 0 ^ { + 5 . 9 7 }$ </td><td> $5 3 . 5 3 ^ { - 1 6 . 4 8 }$ </td></tr><tr><td>ChatDev</td><td> $7 5 . 3 7 \AA ^ { - 3 . 7 3 }$ </td><td> $6 1 . 5 8 ^ { - 7 . 9 4 }$ </td><td> $7 9 . 8 5 ^ { - 0 . 7 5 }$ </td><td> $6 8 . 6 6 ^ { + 0 . 8 8 }$ </td><td> $7 6 . 1 2 ^ { - 3 . 7 3 }$ </td><td> $6 1 . 9 5 ^ { + 1 . 1 7 }$ </td><td> $7 6 . 1 2 ^ { - 8 . 2 1 }$ </td><td> $6 6 . 5 2 ^ { - 3 . 4 9 }$ </td></tr><tr><td>G-Memory</td><td> $7 2 . 3 9 ^ { - 6 . 7 1 }$ </td><td> $7 2 . 0 5 ^ { + 2 . 5 3 }$ </td><td> $7 1 . 6 4 ^ { - 8 . 9 6 }$ </td><td> $5 9 . 5 3 ^ { - 8 . 2 5 }$ </td><td> $8 4 . 3 3 ^ { + 4 . 4 8 }$ </td><td> $5 7 . 0 8 ^ { - 3 . 7 0 }$ </td><td> $9 1 . 0 4 ^ { + 6 . 7 1 }$ </td><td> $5 6 . 6 2 ^ { - 1 3 . 3 9 }$ </td></tr><tr><td>CoMem (Ours)</td><td> $\mathbf { 8 8 . 8 1 ^ { + 9 . 7 1 } }$ </td><td> $7 4 . 4 4 ^ { + 4 . 9 2 }$ </td><td> $\mathbf { 8 7 . 3 1 ^ { + 6 . 7 1 } }$ </td><td> $\mathbf { 6 9 . 5 0 ^ { + 1 . 7 2 } }$ </td><td> ${ \bf 8 9 . 5 5 ^ { + 9 . 7 0 } }$ </td><td> $\mathbf { 7 0 . 1 9 ^ { + 9 . 4 1 } }$ </td><td> $\mathbf { 9 0 . 3 0 ^ { + 5 . 9 7 } }$ </td><td> $7 2 . 7 4 ^ { + 2 . 7 3 }$ </td></tr></table>

Table 1: Success rates on ALFWorld and PDDL across diferent MAS frameworks. Superscripts indicate absolute percentagepoint diferences from the No-memory baseline. The underlying LLM backbone is DeepSeek-V4-Flash.

## 3.4 Parallel Dual-Stream Retrieval

Building on the stratified structure of both private and collective memories, CoMem empowers agents to leverage the strengths of both perspectives during task execution. When a new task initialization query arrives, each agent simultaneously retrieves information from the two isolated repositories through a parallel dual-stream mechanism. This design not only eliminates the bottleneck of sequential lookups, but also ensures that the agent’s decision context is enriched by both individualized tactical know-how and distilled organizational wisdom—allowing for synergy without cross-tier contamination.

Private Stream Retrieval. The private retrieval stream zeroes in on the agent’s own archive of experience. Using semantic embedding similarity(Lewis et al. 2020), the top-ranked private memory entries—those most relevant to the current task context—are rapidly retrieved from the agent’s individual tier. Injecting these insights directly into the agent’s prompt equips the agent with tailored, personal behavioral guidance honed across its own learning history, supporting both specialization and agility in complex environments.

Collective Stream Retrieval and Diversity Filtering. Concurrently, the collective stream taps into the shared pool of group-sourced wisdom, retrieving entries that have been empirically validated and promoted from across the entire team. To address the risk of semantic overlap—inevitable when multiple agents converge on similar insights—CoMem integrates an online hierarchical clustering filter(Murtagh and Contreras 2012). By grouping candidates based on semantic distance and selecting the most representative entry from each cluster, the mechanism constructs a consensus context that is both succinct and diverse. This not only minimizes redundant knowledge within the prompt, conserving valuable token space, but also ensures that agents benefit from a structurally rich set of perspectives distilled from the broader collective experience.

Through parallel dual-stream retrieval, CoMem instantiates a memory interface where individual expertise and shared knowledge dynamically inform each agent’s reasoning process. This architecture tightly couples personal exploration with the evolving consensus of the group, fostering both innovation and reliability in multi-agent coordination.

## 4 Experiments

In this section, we conduct experiments to answer: (RQ1) How does CoMem perform compared to single-/multi-agent memory architectures? (RQ2) Does CoMem incur excessive resource overhead? (RQ3) How sensitive is CoMem to its key components and hyper-parameters?

## 4.1 Experiment Setup

Datasets and Baselines. We evaluate CoMem on two standard decision-making benchmarks: ALFWorld (Shridhar et al. 2021) and PDDL (Ghallab et al. 1998). ALFWorld tests embodied multi-step control with dense environmental feedback, while PDDL requires long-horizon symbolic planning and strict action syntax. We compare against No-memory, four single-agent memory methods adapted to MAS (Voyager (Wang et al. 2023), MemoryBank (Zhong et al. 2024), Generative Agents (Park et al. 2023)), and three cooperative shared-memory schemes (MetaGPT (Hong et al. 2024), ChatDev (Qian et al. 2024), G-Memory (Zhang et al. 2025a)). All memory modules are plugged into the host MAS without modifying its native coordination protocol.

MAS Frameworks and LLM Backbones. Experiments are conducted on four representative MAS frameworks: AutoGen (Wu et al. 2024), DyLAN (Liu et al. 2024), Mac-Net (Qian et al. 2025), and CARD (Wu et al. 2026). We adopt DeepSeek-V4-Flash (DeepSeek-AI 2026) as the primary backbone, with Qwen3-32B (Yang et al. 2025) for supplementary validation. DeepSeek-V4-Flash represents a large-scale LLM backbone with a sparse architecture, while Qwen3-32B serves as a dense transformer backbone for supplementary validation. In our framework, the backbone LLM is only responsible for general reasoning and generation, whereas memory organization, evolution, and retrieval are explicitly managed by CoMem. Therefore, the efectiveness of CoMem does not depend on any specific internal architecture of the underlying LLM and can be integrated with diverse backbone models. To verify this property, we additionally evaluate CoMem with the dense Qwen3-32B backbone and observe consistent improvements, demonstrating that CoMem is backbone-agnostic and generalizes across diferent model families.

(d) PDDL Dylan  
![](images/3679f73cb3338ce343dc5ef413469f75c5150a284621c3197bff61751eda099e.jpg)

![](images/d2300f1c563f082cf5914235d883a837eddea18d329598fc306afd716e08110e.jpg)

![](images/6629ae0565749f49a52599250496cebbe6af427cc4e67ed9f1a77ddddcc3fffb.jpg)

![](images/92fa13f2323a6ecb9598899bdf095ca373b705e2b397800cc6d2e0b3529402d5.jpg)  
Figure 3: Performance versus token cost across memory mechanisms.

Parameter Configurations. Unless otherwise stated, CoMem uses retrieval depth k=3 for both personal and collective pools, private rolling window size $\scriptstyle \dot { K } _ { \mathrm { m a x } } = 5 \ ( \mathrm { P D D L } )$ / 6 (ALFWorld), collective forgetting threshold 30, collective score EMA $\alpha { = } 0 . 5 ,$ , and reward–utility mixing weight $w { = } 0 . 5$ . Promotion requires a minimum usage count of 3 and above-baseline mean reward $( \bar { R } _ { m } \geq \bar { R } )$ . Detailed ablations are reported in Section 4.4.

## 4.2 Main Results (RQ1)

Table 1 reports comprehensive success rates across all configurations. We summarize the key observations as follows.

Takeaway 1: CoMem consistently achieves strong performance across frameworks. On MacNet, CoMem elevates success rates from 79.85 to 89.55 on ALFWorld and from 60.78 to 70.19 on PDDL. Consistent improvements are observed across AutoGen, DyLAN, and CARD, with average gains of 8.02 on ALFWorld and 4.70 on PDDL, computed as the mean of absolute percentage-point increases over the No-memory baseline. Notably, even on the strong CARD backbone, where the No-memory baseline already achieves 84.33 on ALFWorld and 70.01 on PDDL, CoMem still delivers non-trivial improvements of 90.30 and 72.74, further validating its efectiveness as a plug-and-play memory layer that consistently enhances performance regardless of the underlying MAS strength.

Takeaway 2: Conventional cooperative shared memory frequently underperforms the No-memory baseline. For instance, under DyLAN on PDDL, G-Memory yields 59.53, significantly below No-memory’s 67.78; MetaGPT drops from 67.78 to 52.80. Single-agent memory methods adapted to MAS, such as Voyager and MemoryBank, also exhibit inconsistent behavior, improving marginally in some configurations while degrading in others. This counterintuitive pattern arises from indiscriminate cross-agent read/write, causing noise accumulation and behavioral homogenization. In a shared repository without structural isolation, a large volume of unverified trial-and-error records is continuously written, progressively degrading storage quality—a phenomenon we refer to as noise accumulation. Simultaneously, strategic trajectories of diferent roles are repeatedly overwritten and averaged out during cross-agent interactions, gradually diminishing the value of role diferentiation—a phenomenon we refer to as behavioral homogenization. These two degradation patterns are consistent with the memory pollution problem formalized in Section 3. CoMem prevents individual experiences from being contaminated through structured isolation of the private layer, and ensures that only suficiently validated insights can enter the collective pool via the empirical promotion gateway, thereby efectively suppressing both degradation pathways.

![](images/893c47532c3abe8c3884d943574afe6f4af523e491a27e9fd54037c69e0c8dd0.jpg)

![](images/30637698549f6841d6f74db0b05d241c73a00ae950800961a8cb9d9ad6989284.jpg)  
Figure 4: Impact of retrieval depth k on system performance.

## 4.3 Cost Analysis (RQ2)

Figure 3 visualizes the performance–token trade-of.

Takeaway 3: CoMem achieves the highest success rates with a modest increase in token consumption, primarily attributable to parallel dual-stream retrieval and empirical promotion validation. This overhead translates into substantial and consistent performance gains, with average absolute improvements of 8.02 on ALFWorld and 4.70 on PDDL. In contrast, cooperative baselines often incur comparable or even higher token costs without yielding consistent performance benefits, further underscoring CoMem’s favorable cost-efectiveness profile.

## 4.4 Framework Analysis (RQ3)

We systematically evaluate CoMem’s retrieval depth, component necessity, and lifecycle hyper-parameters. All ablation trials are baseline-controlled and executed with the AutoGen and DyLAN frameworks on both benchmarks.

Retrieval Depth Sensitivity. We ablate the operational retrieval depth $k \in \{ 1 , 3 , 5 , 7 \}$ for both personal and collective pools. Figure 4 shows domain-dependent optima: on Auto-Gen, ALFWorld peaks at k=3 (88.80) while PDDL peaks at $k { = } 5 ~ ( 8 0 . 4 0 ) ;$ ; on DyLAN, ALFWorld peaks at $k { = } 5 \ ( 8 8 . 8 0 )$ and PDDL at k=3 (69.50). Moderate $k \in \{ 3 , 5 \}$ yields robust performance; larger k injects noise, smaller k misses context. Notably, the gap between $k { = } 1 \ ( 7 4 . 8 0 )$ and $k { = } 5 \ ( 8 0 . 4 0 )$ is substantial (5.60 points), demonstrating that retrieval scaling is contingent upon task dificulty. These results suggest that optimal retrieval depth varies across both backbone frameworks and task domains, and that a moderate retrieval budget $( k \in \{ 3 , 5 \} )$ consistently yields robust performance.

<table><tr><td>MAS</td><td>Private</td><td>Collective</td><td>ALFWorld</td><td>PDDL</td></tr><tr><td rowspan="3">AutoGen</td><td>√</td><td></td><td>83.58</td><td>62.41</td></tr><tr><td></td><td>√</td><td>85.07</td><td>65.10</td></tr><tr><td>√</td><td>√</td><td>88.31</td><td>74.44</td></tr><tr><td rowspan="3">DyLAN</td><td>√</td><td></td><td>80.25</td><td>59.17</td></tr><tr><td></td><td>√</td><td>77.61</td><td>67.22</td></tr><tr><td>√</td><td>√</td><td>87.31</td><td>69.50</td></tr></table>

Table 2: Component ablation of CoMem on PDDL and ALF-World.

Memory Granularity and Component Necessity. To confirm whether the two-tiered paradigm is structurally necessary, we isolate the Personal Layer and Collective Pool on both benchmarks. In the w/o Collective configuration, cross-task empirical promotion is disabled (personal-only); in the w/o Personal configuration, trajectories are distilled directly into the shared pool at episode termination without individual bufering (collective-only).

Table 2 reveals that the full CoMem setup consistently outperforms either isolated variant. On AutoGen with PDDL, the full configuration achieves 74.44, surpassing collectiveonly (65.10) and personal-only (62.41). The performance gap between personal-only and collective-only suggests that maintaining individual episodic bufers is more fundamental to preserving operational integrity than immediately pooling unvalidated collaborative knowledge. Critically, removing the collective layer leads to a 12.03-point drop on AutoGen with PDDL $( 7 4 . 4 4  6 2 . 4 1 )$ , while removing the private layer causes a 9.34-point drop (74.44 → 65.10). This asymmetry indicates that both layers contribute essential yet distinct functions, with the private layer providing a slightly stronger safeguard against memory pollution.

Threshold Sensitivity: Private Rolling Window and Collective EMA. We conduct a fine-grained sweep on Auto-Gen across two key dimensions: (i) the private rolling window size $K _ { \operatorname* { m a x } } .$ , which governs the eviction of stale private entries upon consecutive non-activation, and (ii) the collective EMA update rate $\alpha ,$ which controls how rapidly collective memory scores adapt to newly observed performance. Results are in Table 3.

Private Rolling Window Size $( K _ { \mathbf { m a x } } ) .$ We ablate $K _ { \mathrm { m a x } } \in$ {3, 4, 5, 6, 7, 8} on AutoGen and observe an inverted-U relationship on both benchmarks. For PDDL, the optimal window size is $K _ { \operatorname* { m a x } } = 5 ,$ , yielding the best success rate of 74.44. When the window is too small $( K _ { \operatorname* { m a x } } = 3 )$ , performance drops to 66.21 due to over-pruning of infrequently retrieved but potentially useful experiences. When the window is excessively large $( K _ { \operatorname* { m a x } } = 8 )$ , performance declines to 67.33 as task-specific noise accumulates and degrades the signal-tonoise ratio. For ALFWorld, the optimal window size shifts to $K _ { \operatorname* { m a x } } = 6$ , achieving 88.81. ALFWorld exhibits a flatter performance curve than PDDL, suggesting less sensitivity to private memory pruning than the structurally complex planning domain. The suboptimal performance at $K _ { \operatorname* { m a x } } = 4$ (82.09) and $K _ { \operatorname* { m a x } } = 7 ( 8 7 . { \dot { 3 } } 1 )$ further confirms that moderate window size strikes the best trade-of between retaining useful experiences and eliminating obsolete knowledge.

<table><tr><td colspan="3">Pruning  $\left( K _ { \operatorname* { m a x } } \right)$ </td><td colspan="3">Fusion (α)</td></tr><tr><td> $K _ { \mathrm { m a x } }$ </td><td>ALFWorld</td><td>PDDL</td><td>α</td><td>ALFWorld</td><td>PDDL</td></tr><tr><td>3</td><td>84.33</td><td>66.21</td><td>0.3</td><td>81.34</td><td>67.80</td></tr><tr><td>4</td><td>82.09</td><td>70.80</td><td>0.4</td><td>85.07</td><td>71.06</td></tr><tr><td>5</td><td>84.33</td><td>74.44</td><td>0.5</td><td>88.81</td><td>74.44</td></tr><tr><td>6</td><td>88.81</td><td>71.14</td><td>0.6</td><td>85.82</td><td>72.41</td></tr><tr><td>7</td><td>87.31</td><td>71.57</td><td>0.7</td><td>80.60</td><td>69.53</td></tr><tr><td>8</td><td>85.07</td><td>67.33</td><td>0.8</td><td>84.33</td><td>65.89</td></tr></table>

Table 3: Sensitivity analysis of private rolling window size $K _ { \mathrm { m a x } }$ and collective EMA update rate α on PDDL and ALF-World success rates.

Collective EMA Update Rate (α). We ablate $\alpha \in$ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8} on AutoGen and observe a consistent inverted-U trend on both benchmarks. The momentum coeficient α governs the collective score update rule $S _ { t + 1 } = ( 1 - \alpha ) \bar { S _ { t } + \alpha } \cdot \left( 1 0 0 \cdot U _ { t } \right)$ , where a smaller α emphasizes historical scores while a larger α favors newly observed performance. When α is too small $( \alpha = 0 . 3 ) \ :$ , collective scores lag behind environmental changes, yielding 67.80 on PDDL and 81.34 on ALFWorld, failing to promptly deprecate entries whose utility has diminished. When α is too large $( \alpha = 0 . 8 )$ , the system overreacts to single-trial anomalies, yielding 65.89 on PDDL and 84.33 on ALFWorld, promoting transient noise into the collective pool. The optimal equilibrium is achieved at $\alpha = 0 . 5$ on both benchmarks (PDDL: 74.44; ALFWorld: 88.81), where the system maintains sufficient responsiveness while efectively suppressing noise. Across all tested α values, ALFWorld consistently outperforms PDDL, and both benchmarks exhibit the same optimal EMA setting, confirming that $\alpha = 0 . 5$ is a robust default across task domains.

## 5 Conclusion

This paper introduces CoMem, a group memory architecture designed to overcome the memory pollution problem in LLM-based multi-agent systems. By structurally separating agents’ private experiences from the collective memory pool and implementing a rigorous empirical promotion mechanism, CoMem preserves individual learning while ensuring that only validated, high-value insights are shared at the organizational level. A parallel dual-stream retrieval strategy allows each agent to leverage both personal knowledge and generalized collective wisdom for improved collaboration. Comprehensive experiments demonstrate that CoMem provides a robust and efective solution for mitigating memory pollution in LLM-based multi-agent systems.

## References

Brooks, R. A. 1991. Intelligence without Representation. Artificial Intelligence, 47(1–3): 139–159.

Bruggeman, J. 2023. Collective Memory, Consensus, and Learning Explained by Social Cohesion. arXiv preprint arXiv:2311.14386.

Chen, W.; Su, Y.; Zuo, J.; Yang, C.; Yuan, C.; Chan, C.-M.; Yu, H.; Lu, Y.; Hung, Y.-H.; Qian, C.; Qin, Y.; Cong, X.; Xie, R.; Liu, Z.; Sun, M.; and Zhou, J. 2024. AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors. In International Conference on Learning Representations (ICLR).

Chhikara, P.; Khant, D.; Aryan, S.; Singh, T.; and Yadav, D. 2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv preprint arXiv:2504.19413.

DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelligence. arXiv preprint arXiv:2606.19348.

Du, P. 2026. Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers. arXiv preprint arXiv:2603.07670.

Du, Y.; Li, S.; Torralba, A.; Tenenbaum, J. B.; and Mordatch, I. 2024. Improving Factuality and Reasoning in Language Models through Multiagent Debate. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, 11733–11763. PMLR.

Ghallab, M.; Howe, A.; Knoblock, C.; McDermott, D.; Ram, A.; Veloso, M.; Weld, D.; and Wilkins, D. 1998. PDDL – The Planning Domain Definition Language. Technical report, Yale Center for Computational Vision and Control.

Guo, T.; Chen, X.; Wang, Y.; Chang, R.; Pei, S.; Chawla, N. V.; Wiest, O.; and Zhang, X. 2024. Large Language Model based Multi-Agents: A Survey of Progress and Challenges. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI), 8048–8057.

Han, S.; Zhang, Q.; Yao, Y.; Jin, W.; Xu, Z.; and He, C. 2024. LLM Multi-Agent Systems: Challenges and Open Problems. arXiv preprint arXiv:2402.03578.

Hong, S.; Zhuge, M.; Chen, J.; Zheng, X.; Cheng, Y.; Zhang, C.; Wang, J.; Wang, Z.; Yau, S. K. S.; Lin, Z.; Zhou, L.; Ran, C.; Xiao, L.; Wu, C.; and Schmidhuber, J. 2024. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. In International Conference on Learning Representations (ICLR).

Jiménez Gutiérrez, B.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS).

Khushiyant. 2025. Emergent Collective Memory in Decentralized Multi-Agent AI Systems. arXiv preprint arXiv:2512.10166.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS).

Li, G.; Hammoud, H. A. A. K.; Itani, H.; Khizbullin, D.; and Ghanem, B. 2023. CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society. In Advances in Neural Information Processing Systems (NeurIPS).

Liu, J.; Kong, Z.; Yang, C.; Yang, F.; Li, T.; Dong, P.; Nanjekye, J.; Tang, H.; Yuan, G.; Niu, W.; Zhang, W.; Zhao, P.; Lin, X.; Huang, D.; and Wang, Y. 2025. RCR-Router: Eficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory. arXiv preprint arXiv:2508.04903.

Liu, Z.; Zhang, Y.; Li, P.; Liu, Y.; and Yang, D. 2024. A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration. In Proceedings of the Conference on Language Modeling (COLM).

Lu, M.; Wu, M.; Liu, F.; Xu, J.; Li, W.; Wang, H.; Hu, Z.; Ding, Y.; Sun, Y.; Lu, J.; and Zhang, Y. 2026. Choosing How to Remember: Adaptive Memory Structures for LLM Agents. arXiv preprint arXiv:2602.14038.

Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegrefe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; Gupta, S.; Majumder, B. P.; Hermann, K.; Welleck, S.; Yazdanbakhsh, A.; and Clark, P. 2023. Self-Refine: Iterative Refinement with Self-Feedback. In Advances in Neural Information Processing Systems (NeurIPS).

Murtagh, F.; and Contreras, P. 2012. Algorithms for Hierarchical Clustering: An Overview. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(1): 86–97.

Packer, C.; Wooders, S.; Lin, K.; Fang, V.; Patil, S. G.; Stoica, I.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as Operating Systems. arXiv preprint arXiv:2310.08560.

Park, J. S.; O’Brien, J. C.; Cai, C. J.; Morris, M. R.; Liang, P.; and Bernstein, M. S. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST).

Qian, C.; Liu, W.; Liu, H.; Chen, N.; Dang, Y.; Li, J.; Yang, C.; Chen, W.; Su, Y.; Cong, X.; Xu, J.; Li, D.; Liu, Z.; and Sun, M. 2024. ChatDev: Communicative Agents for Software Development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15174–15186.

Qian, C.; Xie, Z.; Wang, Y.; Liu, W.; Zhu, K.; Xia, H.; Dang, Y.; Du, Z.; Chen, W.; Yang, C.; Liu, Z.; and Sun, M. 2025. Scaling Large Language Model-based Multi-Agent Collaboration. In International Conference on Learning Representations (ICLR).

Rezazadeh, A.; Li, Z.; Lou, A.; Zhao, Y.; Wei, W.; and Bao, Y. 2025. Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control. arXiv preprint arXiv:2505.18279.

Shinn, N.; Cassano, F.; Berman, E.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS).

Shridhar, M.; Yuan, X.; Côté, M.-A.; Bisk, Y.; Trischler, A.; and Hausknecht, M. 2021. ALFWorld: Aligning Text and Embodied Environments for Actionable Commonsense Reasoning. In International Conference on Learning Representations (ICLR).

Sumers, T. R.; Yao, S.; Narasimhan, K.; and Grifiths, T. L. 2024. Cognitive Architectures for Language Agents. Transactions on Machine Learning Research.

Torra, V.; and Bras-Amorós, M. 2026. Memory Poisoning and Secure Multi-Agent Systems. arXiv preprint arXiv:2603.20357.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2023. Voyager: An Open-Ended

Embodied Agent with Large Language Models. arXiv preprint arXiv:2305.16291.

Wang, J.; Wang, J.; Athiwaratkun, B.; Zhang, C.; and Zou, J. 2024a. Mixture-of-Agents Enhances Large Language Model Capabilities. arXiv preprint arXiv:2406.04692.

Wang, L.; Ma, C.; Feng, X.; Zhang, Z.; Yang, H.; Zhang, J.; Chen, Z.; Tang, J.; Chen, X.; Lin, Y.; Zhao, W. X.; Wei, Z.; and Wen, J.-R. 2024b. A Survey on Large Language Model based Autonomous Agents. Frontiers of Computer Science, 18(6): 186345.

Wang, Y.; Goyal, A.; Chen, Y.; and Sundaram, H. 2026. State Contamination in Memory-Augmented LLM Agents. arXiv preprint arXiv:2605.16746.

Wang, Z. Z.; Mao, J.; Fried, D.; and Neubig, G. 2025. Agent Workflow Memory. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 63897–63911. PMLR.

Wooldridge, M. 2009. An Introduction to MultiAgent Systems. John Wiley & Sons, 2 edition.

Woolley, A. W.; Chabris, C. F.; Pentland, A.; Hashmi, N.; and Malone, T. W. 2010. Evidence for a Collective Intelligence Factor in the Performance of Human Groups. Science, 330(6004): 686–688.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Li, B.; Zhu, E.; Jiang, L.; Zhang, X.; Zhang, S.; Liu, J.; Awadallah, A. H.; White, R. W.; Burger, D.; and Wang, C. 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. In Proceedings ofthe Conference on Language Modeling (COLM).

Wu, T.; Li, Y.; Tang, Z.; Jiang, C.; Luo, L.; Qi, G.; Pan, S.; and Hafari, G. 2026. CARD: Towards Conditional Design of Multiagent Topological Structures. In International Conference on Learning Representations (ICLR).

Xi, Z.; Chen, W.; Guo, X.; He, W.; Ding, Y.; Hong, B.; Zhang, M.; Wang, J.; Jin, S.; Zhou, E.; Zheng, R.; Fan, X.; Wang, X.; Xiong, L.; Zhou, Y.; Wang, W.; Jiang, C.; Zou, Y.; Liu, X.; Yin, Z.; Dou, S.; Weng, R.; Zhang, Q.; Qin, W.; Zheng, Y.; Qiu, X.; Huang, X.; and Gui, T. 2025. The Rise and Potential of Large Language Model Based Agents: A Survey. Science China Information Sciences, 68(121101).

Xu, W.; Liang, Z.; Mei, K.; Gao, H.; Tan, J.; and Zhang, Y. 2025. A-Mem: Agentic Memory for LLM Agents. In Advances in Neural Information Processing Systems (NeurIPS).

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR).

Zhang, G.; Fu, M.; Wang, K.; Wan, G.; Yu, M.; and Yan, S. 2025a. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems. In Advances in Neural Information Processing Systems (NeurIPS).

Zhang, Z.; Bo, X.; Ma, C.; Li, R.; Chen, X.; Dai, Q.; Zhu, J.; Dong, Z.; and Wen, J.-R. 2025b. A Survey on the Memory Mechanism of Large Language Model-based Agents. ACM Transactions on Information Systems, 43(6): 1–47.

Zhao, A.; Huang, D.; Xu, Q.; Lin, M.; Liu, Y.-J.; and Huang, G. 2024. ExpeL: LLM Agents Are Experiential Learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 19632–19642.

Zhong, W.; Guo, L.; Gao, Q.; Ye, H.; and Wang, Y. 2024. MemoryBank: Enhancing Large Language Models with Long-Term Memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 19724–19731.