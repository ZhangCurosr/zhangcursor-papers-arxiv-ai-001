# HeMeR: Heterogeneous Memory Reconciliation for Embodied Agents via Structured KV Reuse

Saehun Chun<sup>1</sup>, Wonje Choi<sup>1</sup>, Jinwoo Jang<sup>1</sup>, Taeyoon Kwack<sup>2</sup>, Honguk Woo<sup>1,2∗</sup> <sup>1</sup>Department of Computer Science and Engineering, Sungkyunkwan University <sup>2</sup>Department of Artificial Intelligence, Sungkyunkwan University {thousae,wjchoi1995,jinustar,njj05043,hwoo}@skku.edu

## Abstract

Embodied agents make decisions by combining information from heterogeneous memory modules, including declarative memory for explicit knowledge, procedural memory for learned routines, and working memory for current observations. During interaction, the memory contents required for decision making change with the agent’s state, goal, and task progress, while many contents remain shared across consecutive decisions. However, existing memory modules are typically accessed through separate interfaces, so each updated memory context is often processed as a new input to the foundation model. This prevents selective key-value (KV) cache reuse for unchanged memory contents and forces full KV cache recomputation even when much of the memory context remains valid. We introduce HEMER, an inference framework that performs KV reconciliation between dynamically updated memory contexts and reusable KV states. HEMER tracks declarative, procedural, and working memory contents across action decisions, preserves KV states for unchanged contents, and reconciles only the KV states affected by memory-content updates. This enables efficient and stable inference as memory contents evolve during interaction. We evaluate HEMER on ALFWorld, Habitat, and real-world robotic deployments requiring coordinated use of heterogeneous memory contents. In the real-world setting, HEMER achieves 83.20% task success and reduces perstep inference time from 2.94 to 1.87 seconds compared with MemRL, while preserving compatible KV states as task progress updates the memory context.

## 1 Introduction

Embodied agents increasingly build on foundation models, using language or vision-language models to interpret observations, reason about goals, and select actions [1, 2, 3, 4, 5, 6]. However, embodied tasks often require information beyond immediate observations, including explicit environmental knowledge, learned behavioral routines, and ongoing task progress [4, 5, 6, 7]. Accordingly, recent studies augment foundation models with memory modules that store, retrieve, and update taskrelevant memory contents across interactions [6, 7, 8, 9, 10]. These modules help agents preserve continuity over long-horizon tasks, but also make each action decision depend on memory contents that change with accumulated experience, current observations, and task progress.

Recent memory-augmented agents use heterogeneous memory modules that maintain distinct forms of memory content for embodied decision making [2, 6, 8, 11]. A declarative memory module maintains explicit content about environmental objects, relations, and task conditions, often as structured representations such as symbolic facts [8, 9, 12, 13]. A procedural memory module maintains learned behavioral routines that guide future actions [5, 6, 14, 15], including parametric representations such as neural adapters [16]. A working memory module maintains content about the current task state, including recent observations, intermediate decisions, and goal progress [2, 4, 17]. Although these modules are often studied through separate storage, retrieval, and update mechanisms, action selection requires their contents to be composed into a memory context. The resulting memory context is processed by the foundation model to form transformer KV states during inference [18, 19]. As the agent receives new observations and progresses through the task, the memory context must be updated while many memory contents remain unchanged across consecutive decisions.

![](images/adcb77216aee9f393ed35bb232f31687e72edcff0fc2672c3c899c111fc5ef23.jpg)  
Figure 1: Motivation of HEMER. Environment changes can trigger memory-content updates, including procedural changes that switch the active adapter. A naive implementation recomputes the full KV cache, whereas HEMER updates affected KV ranges and reuses compatible KV states.

This modular separation is well motivated, since each memory module serves a distinct role in embodied decision making [11, 14, 17, 20]. However, after memory contents from multiple memory modules are composed into a single memory context, the foundation model must process this memory context to form the hidden representations and KV states used for action selection [18, 19]. As illustrated in Figure 1, changes in the environment can trigger updates to declarative and working memory contents and procedural adapter switches, thereby changing the memory context across decision steps. Such changes can make some previously cached KV states incompatible with the updated memory context, while others may remain reusable [21, 22, 23, 24, 25, 26]. Existing memory-augmented agents mainly focus on what to store, retrieve, and update, but do not explicitly address how to reconcile updated memory contexts with reusable KV states during foundation model inference [9, 10, 12, 27, 28]. Without such reconciliation, a naive implementation processes each updated memory context as a new input and recomputes the full KV cache, causing redundant computation during long-horizon embodied interaction [22, 29, 30, 31].

To address this problem, we introduce HEMER, a memory-serving framework for embodied agents that reconciles dynamically updated memory contexts with reusable KV states during foundation model inference. Although declarative, procedural, and working memory serve different roles, their effects on foundation model inference are ultimately reflected in the KV states used for action selection. Rather than treating each memory context as a flat token sequence, HEMER represents it as a sequence of structured memory chunks associated with corresponding KV ranges, providing a common basis for reconciling heterogeneous memory updates with cached KV states. HEMER realizes this reconciliation through two complementary mechanisms, Chunk-Level KV Reconciliation for content-level changes and rearrangements within the memory context, and Adapter-Stable KV Reconciliation for procedural changes that alter the active adapter.

Specifically, Chunk-Level KV Reconciliation handles updates to declarative and working memory contents by mapping content-level changes and rearrangements to their corresponding KV ranges. HEMER identifies reusable KV ranges by tracking changes in memory contents and their dependencies, selectively recomputing only the affected ranges. When an unchanged memory chunk shifts position, RoPE-based remapping preserves positional consistency, allowing its KV states to be reused [24]. Adapter-Stable KV Reconciliation handles procedural memory updates that alter the active adapter. Because KV states computed under a previous adapter may become incompatible even when other memory contents remain unchanged, HEMER treats the adapter configuration as part of the KV compatibility condition, selectively recomputing affected KV ranges while preserving compatible KV states across adapter switches. By jointly reconciling memory content updates and procedural adapter changes without rebuilding the full KV cache, HEMER supports efficient and stable inference over long-horizon embodied tasks.

We evaluate HEMER on unseen simulated tasks from ALFWorld [32] and EmbodiedBench Habitat [33, 34], where evaluation episodes require the agent to combine memory contents across different situations rather than rely on a directly applicable memory content. We further evaluate HEMER on real-world embodied tasks in a UR7e manipulation setup, requiring the agent to coordinate heterogeneous memory contents as procedures and task states change. We highlight task success rate and inference time per decision step, with each step corresponding to one action selection. On ALFWorld, HEMER achieves a 71.43% task success rate with 0.18 seconds per decision step, compared with a 52.86% task success rate and 0.90 seconds per decision step for MemRL [35], the strongest baseline in task success (Table 1). On EmbodiedBench Habitat, HEMER achieves a 76.27% task success rate with 1.82 seconds per decision step, compared with a 61.80% task success rate and 2.68 seconds per decision step for MemRL (Table 1). In the real-world setting, HEMER achieves an 83.20% goal-conditioned success rate with 1.87 seconds per decision step, compared with a 64.28% goal-conditioned success rate and 2.94 seconds per decision step for MemRL (Table 2). These results show that KV reconciliation improves task robustness and inference efficiency as memory contexts change across embodied decision steps.

We make the following contributions.

• We formulate heterogeneous memory use as a KV reconciliation problem for foundation modelbased embodied agents, where memory contents from declarative, procedural, and working memory modules change across action decisions while many contents remain reusable.

• We introduce HEMER, a heterogeneous memory reconciliation framework that anchors memory contents as typed chunks with KV ranges and selectively reuses compatible KV states.

• We develop chunk-level and adapter-stable KV reconciliation mechanisms that update only affected KV ranges, preserve positional consistency under memory-content updates, and maintain compatible KV states across procedural adapter switches.

• We evaluate HEMER on ALFWorld, Habitat, and real-world robotic deployments, showing that it improves task success and reduces per-step inference time over strong baselines.

## 2 Related work

Foundation Model-based Embodied Control. Recent studies use language and vision-language foundation models as general-purpose decision modules for embodied agents, enabling them to interpret observations, reason about goals, and select actions in interactive environments [1, 3, 4, 5, 6, 36]. These agents are further augmented with memory modules to store and retrieve task-relevant memory contents across decision steps and adapt action selection over long-horizon interactions [7, 10, 37, 35, 38]. HEMER follows this line of work, but focuses on the inference problem that arises when memory contents from multiple memory modules are composed into a dynamically updated memory context that must be reconciled with reusable KV states across action decisions.

Memory-Augmented Embodied Agents. Recent memory-augmented embodied agents introduce specialized memory modules for different roles in decision making. Declarative memory modules maintain explicit content about environmental objects, relations, and task conditions [8, 10, 12, 13]. Procedural memory modules maintain learned behavioral routines, including reusable procedures and skills [5, 6, 14, 15, 39, 40, 41, 25]. Working memory modules maintain memory content about recent observations, intermediate decisions, and goal progress during the current interaction [2, 4, 11, 17]. These memory modules help embodied agents preserve task-relevant memory content beyond immediate observations and adapt action selection over time. However, existing studies mainly focus on how each memory module stores, retrieves, or updates its own memory content for a role-specific objective. In contrast, HEMER jointly handles memory contents from declarative, procedural, and working memory modules once they are composed into a single memory context, reconciling their updates with reusable KV states across action decisions.

Efficient KV Reuse. KV caching is a standard mechanism for efficient autoregressive inference, in which previously computed KV states are stored and reused across decoding steps [18, 19, 42]. Prior work on KV reuse reduces inference cost by reusing cached KV states for repeated prefixes, shared prompts, or recurring context segments [29, 43, 44]. Recent systems further exploit structured generation programs, retrieved context chunks, position-independent context segments, or embodied reasoning traces to reuse computation across related inference requests or decision steps [21, 22,

23, 45, 46, 47]. Complementary work reduces KV cache storage requirements, transfer costs, or loading overhead through eviction, compression, streaming, or cache-layer management [30, 31, 48, 49]. In parallel, multi-adapter serving systems improve the efficiency of models augmented with LoRA adapters by batching requests, switching adapters, or reusing KV states across adapters during inference [16, 25, 26, 50, 51, 52, 53]. Building on these advances, HEMER integrates KV reuse and multi-adapter serving techniques within a memory-serving framework to efficiently reuse KV states as memory contexts change across action decisions.

## 3 Problem formulation

We consider a foundation model-based embodied agent that acts over a sequence of decision steps. At step t, the agent receives an observation $o _ { t }$ , maintains a task goal $^ { g , }$ and selects an action $a _ { t }$ using a foundation model $F _ { \theta }$ . The agent is augmented with heterogeneous memory modules

$$
\mathcal { M } = \{ \mathcal { M } ^ { \mathrm { d e c } } , \mathcal { M } ^ { \mathrm { p r o c } } , \mathcal { M } ^ { \mathrm { w o r k } } \} ,\tag{1}
$$

corresponding to declarative, procedural, and working memory modules. A memory module stores and retrieves information, and each item it returns is a memory content. For each module ${ \mathcal { M } } ^ { m }$ , let $R ^ { m }$ denote its retrieval function and $\mathcal { R } _ { t } ^ { m }$ the memory contents obtained at step t:

$$
\begin{array} { r } { \mathcal { R } _ { t } ^ { m } = R ^ { m } ( o _ { t } , g , h _ { t } ) , \quad m \in \{ \mathrm { d e c } , \mathrm { p r o c } , \mathrm { w o r k } \} . } \end{array}\tag{2}
$$

Here, $h _ { t }$ denotes the interaction history available at step t.

For readability, we denote the declarative, procedural, and working-memory contents as

$$
D _ { t } = \mathcal { R } _ { t } ^ { \mathrm { d e c } } , \quad P _ { t } = \mathcal { R } _ { t } ^ { \mathrm { p r o c } } , \quad W _ { t } = \mathcal { R } _ { t } ^ { \mathrm { w o r k } } .\tag{3}
$$

A memory context is formed by composing multiple memory contents to support action selection. The function Γ defines the step-specific memory context $X _ { t } ,$ , which the foundation model uses together with the observation and task goal to generate an action $a _ { t } \colon$

$$
X _ { t } = \Gamma ( D _ { t } , P _ { t } , W _ { t } ) , \qquad a _ { t } \sim F _ { \theta } ( \cdot \mid X _ { t } , o _ { t } , g ) .\tag{4}
$$

Let D denote the distribution of tasks encountered by the agent. Across decision steps, $X _ { t }$ may change as memory contents are inserted, evicted, reordered, or associated with different procedural adapters. Let Π denote a serving policy that reconciles the previous KV cache with the current memory context $X _ { t }$ . Π determines which KV states are reused or recomputed during action generation. Our objective is to maximize task success while minimizing inference cost:

$$
\Pi ^ { \star } = \operatorname * { a r g m a x } _ { \Pi } \operatorname * { \mathbb { E } } _ { \tau \sim \mathcal { D } } \left[ \operatorname { S u c c } ( \tau ; \Pi ) - \lambda \sum _ { t = 1 } ^ { T } \operatorname { C o s t } ( \Pi , X _ { t - 1 } , X _ { t } ) \right] .\tag{5}
$$

Here, τ denotes a task instance sampled from $\mathcal { D } , \operatorname { S u c c } ( \tau ; \Pi )$ measures whether τ is successfully completed under Π, Cost $( \Pi , X _ { t - 1 } , \dot { X } _ { t } )$ measures the inference computation incurred under Π when processing the memory context update from $X _ { t - 1 }$ to $X _ { t }$ , and λ controls the trade-off between task success and inference cost.

## 4 Method

## 4.1 Overall Architecture

We present HEMER, a heterogeneous memory reconciliation framework that enables foundation model-based embodied agents to process changing memory contexts without rebuilding the full KV cache, as illustrated in Figure 2. At each decision step, the agent obtains memory contents from multiple memory modules and composes them into a single memory context, which the foundation model uses together with the current observation and task goal to generate the next action. During interaction, new observations may introduce memory contents, outdated contents may be removed, existing contents may be reordered, and the active procedural adapter may switch. To handle these changes, HEMER uses a memory schema to anchor each memory content as a typed memory chunk with an associated KV range. It then reconciles memory-content updates through localized cache operations while preserving compatible KV states across procedural adapter switches.

![](images/e03488fbdf923e8241fc1347741d3f06d96a2546d7b11f1243da0ad985cf9e53.jpg)  
Figure 2: HEMER overview. During long-horizon inference, new observations can modify the memory context and switch the active procedural adapter. HEMER indexes memory chunks, updates affected KV ranges, and preserves compatible cache states across decision steps.

We distinguish three levels of memory representation: memory modules, memory contents, and memory contexts. A memory module stores and retrieves a specific type of information, and each item retrieved at a decision step is a memory content. A memory context is formed by composing multiple memory contents to support action selection. In our setting, the agent uses declarative, procedural, and working memory modules. Declarative memory stores explicit task and environment knowledge, such as object locations, receptacle relations, and task conditions. Procedural memory stores reusable behavioral procedures, instantiated as LoRA adapters in our LRAgent-based implementation [25]. Working memory stores the current interaction state, including recent observations, intermediate decisions, subgoals, and action history.

Rather than treating the composed memory context $X _ { t }$ defined in Sec. 3 solely as a flat token sequence, HEMER organizes it as a sequence of memory chunks. Each memory chunk represents a single memory content, with schema-defined metadata linking it to a contiguous KV range after tokenization and encoding. The memory schema records each chunk’s persistent ID, memory module type, semantic role, normalized content, dependency annotation, position, token span, and KV range, as well as its adapter identifier when applicable. This index maps memory-content updates to their corresponding KV ranges, enabling HEMER to reconcile the KV cache through localized operations on affected ranges. The initial memory context is prefilled as a single causal sequence. At subsequent decision steps, HEMER prefills newly inserted memory chunks and selectively recomputes affected KV ranges at their positions in the current memory context, recording the resulting KV ranges in the memory schema.

## 4.2 Chunk-Level KV Reconciliation

Chunk-level KV reconciliation aligns the previous KV cache with the memory contents in $X _ { t } .$ . As the agent progresses, memory contents may be inserted, evicted, or reordered, potentially altering token positions and dependencies in $X _ { t }$ . Naive KV reuse is therefore unreliable, while rebuilding the full KV cache can introduce redundant computation when many memory contents remain unchanged.

Schema-Guided Reconciliation. To address this trade-off, HEMER uses the chunk-level index to compare memory chunks across consecutive decision steps. Chunks in $X _ { t - 1 }$ and $X _ { t }$ are first matched by persistent chunk IDs, regardless of their positions or token spans. Matched chunks are eligible for KV reuse only when their memory module type, semantic role, dependency annotation, normalized content, and adapter identifier (when applicable) agree. Otherwise, their KV ranges are locally recomputed. Given two consecutive memory contexts, HEMER computes a schema-guided reconciliation plan

$$
\Delta _ { t } = \mathrm { D i f f } _ { \mathcal { S } } ( X _ { t - 1 } , X _ { t } ) ,\tag{6}
$$

where $\operatorname { D i f f } _ { \mathbf { \phi } _ { \mathbf { \boldsymbol { S } } } }$ matches chunks by persistent IDs and checks schema compatibility. Matched chunks with changed positions are reordered, chunks only in $X _ { t }$ are inserted, and chunks only in $X _ { t - 1 }$ are evicted. The index maps these operations to KV ranges, so HEMER updates only affected ranges.

KV Operations and RoPE Remapping. Based on the reconciliation plan $\Delta _ { t } ,$ HEMER performs the corresponding KV operations. New memory chunks are prefilled, while the KV ranges of evicted chunks are removed. When chunks are reordered, HEMER relocates their KV ranges according to the current $X _ { t } ,$ reusing compatible KV states when possible. These operations allow the KV cache to be updated without full recomputation.

RoPE remapping is used only for matched chunks whose update is a position shift. If a reusable token shifts from position $p _ { j }$ to $p _ { j } ^ { \prime }$ , its cached key is updated as

$$
\tilde { k } _ { j } = { \cal R } _ { \theta ( p _ { j } ^ { \prime } ) } { \cal R } _ { \theta ( p _ { j } ) } ^ { - 1 } k _ { j } ,\tag{7}
$$

where $k _ { j }$ denotes the cached key at position $p _ { j } , R _ { \theta ( p ) }$ denotes the RoPE rotation at position $p ,$ and $\tilde { k } _ { j }$ denotes the remapped key at position $p _ { j } ^ { \prime }$ . This corrects the cached key’s positional phase under a pure position shift. RoPE remapping updates only the cached keys. For multimodal backbones, we restrict RoPE-based remapping to text-side memory chunks whose schema fields remain unchanged [24]. Visual tokens from current observations are not position-remapped and are encoded normally at each decision step. When content, dependencies, or adapters change, HEMER recomputes the affected KV range to avoid stale reuse.

Chunk-level KV reconciliation differs from prompt-level or segment-level cache reuse. Prefix caching reuses only stable prefixes, Prompt Cache reuses segments computed without their preceding context, and CacheBlend-style reuse still relies on reusable segment boundaries and selective recomputation around them [21, 22, 43]. HEMER makes the same class of non-prefix reuse approximation, but grounds it in schema-indexed memory chunks and reuses a KV range only when the corresponding memory content remains schema-compatible. Thus, HEMER avoids both naive KV reuse and full KV cache recomputation by selectively reconciling KV ranges affected by memory-content updates across decision steps.

## 4.3 Adapter-Stable KV Reconciliation

Procedural memory introduces a second source of cache incompatibility. In our LRAgent-based implementation, procedural memory contents correspond to reusable behavioral procedures implemented as LoRA adapters. When the procedural memory content changes across decision steps, the active adapter may also change. Even if declarative and working-memory contents remain unchanged, KV states computed under the previous adapter may no longer be compatible with the new adapter configuration. This issue is particularly important in embodied tasks, where agents switch among procedures such as searching, manipulating, cleaning, heating, cooling, and placing objects.

Adapter-Aware Reconciliation. HEMER maintains adapter identifiers as part of the chunk-level index. For each procedural memory chunk, the index records the adapter under which its KV states were computed, and the reconciliation plan marks adapter switches as procedural updates. When the active adapter changes, HEMER recomputes the procedural memory chunk’s KV range and those of its direct and transitive dependents under the declared dependency annotations. Other unchanged ranges are approximately reused under the shared-attention LoRA design. This allows adapter switches to be handled within the same indexed reconciliation process without rebuilding the full KV cache.

Cross-Adapter KV Compatibility. To stabilize cross-adapter reuse, HEMER adopts the sharedattention LoRA design used by LRAgent. The rank-8 adapters are applied to the query and value projections, with a low-rank down-projection shared across adapters and adapter-specific up-projections. The key, output, and MLP paths remain shared, and the active adapter is applied to the entire forward sequence rather than only to procedural-memory tokens. This design limits, but does not eliminate, cross-adapter drift. Therefore, HEMER includes the adapter identifier in its KV compatibility condition and recomputes the procedural memory chunk’s KV range while preserving ranges that remain compatible across the switch.

While LRAgent primarily addresses KV cache reuse across adapter switches, HEMER incorporates adapter compatibility into the same reconciliation process used for updates to declarative and working memory contents. Such methods reduce overhead for procedural changes, but they do not explicitly address updates to declarative and working memory contents that arise during embodied interaction. HEMER jointly handles both sources of change: adapter switches from procedural memory and chunk-level reconciliation of updates to declarative and working-memory contents. Consequently, HEMER preserves compatible KV states across heterogeneous memory-content updates while avoiding the need to rebuild the full KV cache at every decision step.

## 5 Evaluation

## 5.1 Experimental setting

Environment. We evaluate on ALFWorld [32], EmbodiedBench Habitat [33, 34], and Realworld, a UR7e manipulation setup. Each evaluation episode is unseen in the sense that no single memory fragment covers the required procedure-room combination. Solving these tasks therefore requires composing fragmented memories across different situations rather than retrieving a directly applicable past experience. Real-world evaluation comprises two long-horizon tasks in two tabletop environments, with five independent physical repetitions per task, yielding 10 trials per method. Initial object positions and instruction phrasings vary across repetitions. Benchmark configurations, task details, and evaluation procedures are provided in Appendix A.

Model. We use two backbone models matched to the observation modality of each environment. For ALFWorld, a text-based environment, we use Llama-3.2-3B-Instruct [54]. For EmbodiedBench Habitat, a vision-based 3D environment, we use Qwen3-VL-4B-Instruct [55], which natively processes RGB observations. Both are served with greedy decoding (temperature = 0). For procedural memory, we use rank-8 LoRA adapters [16] fine-tuned on task-type-specific trajectories for each backbone. Each procedural memory content is linked to an adapter identifier used for adapter selection and KV compatibility checking, as described in Sec. 4.3.

Baselines. We compare HEMER with six baselines covering agent-level reasoning, memory use, KV reuse, and procedural adapter switching. ReAct [2] tests prompting without explicit external memory, and Reflexion [7] tests self-reflective feedback from prior failures. MemRL [35] retrieves episodic memory from previous trajectories to guide action selection, but treats the retrieved context as a conventional input across decision steps. CacheBlend [21] and KEEP [47] evaluate segment-level and invalidation-based KV reuse, respectively. LRAgent [25] tests LoRA-based procedural skill switching, but does not explicitly track updates in declarative or working-memory contents. Together, these baselines separate the effects of prompting, episodic memory retrieval, cache reuse, procedural adapter switching, and the proposed heterogeneous-memory KV reconciliation.

Metrics. We evaluate task performance, action behavior, and inference efficiency with six metrics. SR measures binary end-to-end task success in ALFWorld and Habitat [32, 34], and goal-conditioned success rate in real-world tasks. Latency measures average wall-clock time per decision step, while TTFT isolates prefill and cache-processing latency before the first generated token, a common serving-level efficiency metric [19, 21]. Reuse ratio measures the fraction of KV states preserved after memory-content updates, following prior cache reuse studies [21, 29]. Active action ratio measures the fraction of generated actions belonging to predefined task-progressing action categories [56]. Action diversity measures the average per-episode ratio of unique actions among all generated actions [57]. In the tables, active ratio and diversity ratio denote active action ratio and action diversity ratio, respectively.

## 5.2 Main result

Table 1 reports the main results on ALFWorld and Habitat. Across both benchmarks, HEMER achieves the highest task success while also reducing per-step inference time and improving memory utilization. On ALFWorld, HEMER reaches 71.43% SR with 0.18 seconds per step, outperforming the strongest baseline MemRL at 52.86% SR with 0.90 seconds per step. On Habitat, HEMER achieves 76.27% SR with 1.82 seconds per step, improving over MemRL at 61.80% SR with 2.68 seconds per step. These gains indicate that reconciling dynamically updated memory contents with reusable KV states provides a more effective and efficient decision context than using episodic traces, reactive prompting, or adapter switching in isolation.

Table 1: Performance on embodied tasks with dynamic memory contexts. HEMER improves task success while reducing inference time through higher KV reuse and effective memory utilization.
<table><tr><td>ALFWorld</td><td>SR (%)</td><td>Latency (s/step)</td><td>Reuse ratio (%)</td><td>Active ratio (%)</td><td>Diversity ratio (%)</td></tr><tr><td>HEMER</td><td> ${ \bf 7 1 . 4 3 \pm 2 . 0 5 }$ </td><td> ${ \bf 0 . 1 8 \pm 0 . 0 2 }$ </td><td> ${ \bf 7 5 . 3 0 \pm 4 . 3 3 }$ </td><td> ${ \bf 6 7 . 6 9 \pm 2 . 2 7 }$ </td><td> ${ \bf 7 1 . 4 3 \pm 3 . 3 2 }$ </td></tr><tr><td>ReAct</td><td> $4 7 . 1 4 \pm 6 . 3 9$ </td><td> $0 . 8 8 \pm 0 . 0 4$ </td><td></td><td> $5 2 . 0 4 \pm 2 . 4 5$ </td><td> $6 4 . 9 5 \pm 8 . 9 4$ </td></tr><tr><td>Reflexion</td><td> $4 4 . 2 9 \pm 5 . 7 4$ </td><td> $0 . 9 8 \pm 0 . 0 5$ </td><td></td><td> $4 5 . 5 7 \pm 4 . 3 0$ </td><td> $6 4 . 3 8 \pm 6 . 1 2$ </td></tr><tr><td>MemRL</td><td> $5 2 . 8 6 \pm 4 . 8 3$ </td><td> $0 . 9 0 \pm 0 . 0 4$ </td><td></td><td> $4 6 . 2 6 \pm 5 . 1 6$ </td><td> $6 5 . 0 5 \pm 4 . 4 8$ </td></tr><tr><td>CacheBlend</td><td> $4 8 . 3 9 \pm 4 . 2 9$ </td><td> $0 . 4 8 \pm 0 . 0 4$ </td><td> $5 0 . 3 0 \pm 2 . 3 3$ </td><td> $4 6 . 0 1 \pm 3 . 6 7$ </td><td> $5 8 . 4 9 \pm 2 . 9 2$ </td></tr><tr><td>KEEP</td><td> $5 1 . 8 1 \pm 3 . 9 2$ </td><td> $0 . 4 6 \pm 0 . 0 6$ </td><td> $4 8 . 3 0 \pm 2 . 3 3 $ </td><td> $5 0 . 4 0 \pm 4 . 9 2$ </td><td> $6 5 . 0 1 \pm 4 . 9 5$ </td></tr><tr><td>LRAgent</td><td> $4 6 . 4 3 \pm 3 . 1 9$ </td><td> $0 . 5 2 \pm 0 . 0 3$ </td><td> $4 8 . 5 0 \pm 3 . 3 3$ </td><td> $5 1 . 1 7 \pm 2 . 3 1$ </td><td> $4 9 . 9 8 \pm 3 . 0 6$ </td></tr><tr><td>Habitat</td><td>SR (%)</td><td>Latency (s/step)</td><td>Reuse ratio (%)</td><td>Active ratio (%)</td><td>Diversity ratio (%)</td></tr><tr><td>HEMER</td><td> ${ \bf 7 6 . 2 7 \pm 3 . 3 2 }$ </td><td> ${ \bf 1 . 8 2 \pm 0 . 0 3 }$ </td><td> ${ \bf 7 2 . 8 1 \pm 4 . 2 4 }$ </td><td> ${ \bf 7 1 . 3 8 \pm 3 . 2 9 }$ </td><td> ${ \bf 5 6 . 2 9 \pm 2 . 8 2 }$ </td></tr><tr><td>ReAct</td><td> $5 2 . 0 0 \pm 3 . 3 2$ </td><td> $2 . 4 1 \pm 0 . 0 7$ </td><td></td><td> $5 1 . 8 1 \pm 6 . 3 3$ </td><td> $4 2 . 0 3 \pm 2 . 1 2$ </td></tr><tr><td>Reflexion</td><td> $5 0 . 4 7 \pm 5 . 4 0$ </td><td> $2 . 3 8 \pm 0 . 1 8$ </td><td></td><td> $4 9 . 6 6 \pm 8 . 7 1$ </td><td> $3 9 . 2 7 \pm 3 . 0 4$ </td></tr><tr><td>MemRL</td><td> $6 1 . 8 0 \pm 2 . 9 2$ </td><td> $2 . 6 8 \pm 0 . 1 2$ </td><td></td><td> $5 7 . 5 5 \pm 4 . 7 8$ </td><td> $5 4 . 3 6 \pm 1 . 3 1$ </td></tr><tr><td>CacheBlend</td><td> $5 6 . 3 3 \pm 2 . 2 4$ </td><td> $1 . 9 7 \pm 0 . 0 3$ </td><td> $4 6 . 2 5 \pm 3 . 2 9$ </td><td> $5 1 . 3 9 \pm 3 . 2 9$ </td><td> $5 3 . 2 0 \pm 2 . 3 0$ </td></tr><tr><td>KEEP</td><td> $5 9 . 4 0 \pm 2 . 2 4$ </td><td> $2 . 0 9 \pm 0 . 0 2$ </td><td> $4 3 . 0 5 \pm 4 . 1 2$ </td><td> $6 4 . 5 8 \pm 1 . 7 5$ </td><td> $5 2 . 4 0 \pm 1 . 2 0$ </td></tr><tr><td>LRAgent</td><td> $5 7 . 4 7 \pm 3 . 5 4$ </td><td> $2 . 0 3 \pm 0 . 3 1$ </td><td> $4 2 . 9 6 \pm 3 . 9 8$ </td><td> $5 8 . 3 4 \pm 4 . 5 3$ </td><td> $4 4 . 2 8 \pm 3 . 3 9$ </td></tr></table>

The efficiency and memory behavior metrics explain this trend. Although CacheBlend, KEEP, and LRAgent reuse KV states, their reuse ratios remain lower than HEMER because they do not jointly handle memory-content updates and procedural adapter changes. CacheBlend relies on non-prefix segments, KEEP invalidates affected memory regions, and LRAgent supports procedural LoRA switching without explicitly processing updates to declarative and working-memory contents. In contrast, HEMER preserves more KV states by indexing memory chunks, updating only affected KV ranges, and maintaining adapter-compatible KV states. Higher active action ratio and action diversity indicate a higher fraction of actions in task-progressing categories and a higher fraction of unique actions. Together, these results show that HEMER improves long-horizon embodied inference through memory-aware KV reconciliation without rebuilding the full KV cache.

Table 2 reports real-world embodied task results. We focus on MemRL, CacheBlend, KEEP, and LRAgent because they cover the most relevant memory- and cache-specific mechanisms while limiting costly physical robot trials. HEMER achieves the highest SR, outperforming MemRL by 18.92 percentage points, and reduces inference time from 2.94 to 1.87 seconds per step. This shows that KV reconciliation remains effective when real-world task progress changes action-relevant memory contents. CacheBlend and LRAgent preserve part of the cache but achieve lower SR, while KEEP improves SR at the cost of lower reuse and higher latency. By contrast, HEMER achieves the highest KV reuse ratio, active action ratio, and action diversity, indicating stronger cache preservation and action-level behavior.

Table 2: Performance on real-world embodied tasks. HEMER improves task success and reduces per-step inference time by preserving reusable KV states while using task-relevant memory contents.
<table><tr><td>Real-world</td><td>SR (%)</td><td>Latency (s/step)</td><td></td><td></td><td>Reuse ratio (%) Active ratio (%) Diversity ratio (%)</td></tr><tr><td>HEMER</td><td> ${ \bf 8 3 . 2 0 \pm 4 . 3 9 }$ </td><td> ${ \bf 1 . 8 7 \pm 0 . 1 2 }$ </td><td> ${ \bf 7 2 . 1 8 \pm 3 . 2 8 }$ </td><td> ${ \bf 7 5 . 4 5 \pm 3 . 9 2 }$ </td><td> ${ \bf 6 4 . 8 5 \pm 2 . 2 4 }$ </td></tr><tr><td>MemRL</td><td> $6 4 . 2 8 \pm 5 . 2 5$ </td><td> $2 . 9 4 \pm 0 . 3 2$ </td><td></td><td> $6 1 . 0 4 \pm 4 . 7 4$ </td><td> $5 8 . 1 9 \pm 2 . 7 3$ </td></tr><tr><td>CacheBlend</td><td> $5 1 . 2 1 \pm 4 . 0 2$ </td><td> $2 . 1 4 \pm 0 . 1 2$ </td><td> $5 4 . 0 2 \pm 4 . 3 0$ </td><td> $5 4 . 2 7 \pm 3 . 8 2$ </td><td> $5 4 . 3 7 \pm 4 . 1 9$ </td></tr><tr><td>KEEP</td><td> $6 3 . 4 0 \pm 3 . 2 7$ </td><td> $2 . 4 3 \pm 0 . 2 4$ </td><td> $4 8 . 6 4 \pm 4 . 2 4$ </td><td> $6 2 . 0 3 \pm 3 . 6 8$ </td><td> $5 6 . 2 0 \pm 3 . 2 8$ </td></tr><tr><td>LRAgent</td><td> $5 3 . 4 3 \pm 4 . 1 8$ </td><td> $2 . 2 1 \pm 0 . 1 8$ </td><td> $5 0 . 2 2 \pm 3 . 8 2$ </td><td> $5 6 . 3 0 \pm 2 . 3 9$ </td><td> $4 9 . 9 7 \pm 1 . 2 9$ </td></tr></table>

(B) KV reuse under memory perturbation  
![](images/248b7f6a7f3bf7d6b8dfccffaa5245cb43536117659e185b61f84385bc6d1143.jpg)

![](images/754e7210ca1c26e4473d9f9d43e4a7eacb229300383fd3cdd283fe8760047317.jpg)  
Figure 3: Analysis on Cost efficiency and robustness under memory-content updates. (A) HEMER exhibits slower TTFT growth than the baselines as memory length increases. (B) HEMER preserves higher KV reuse under increasing invalidated memory ratio $\rho .$

## 5.3 Analysis

Cost efficiency under memory-content updates. Figure 3(a, b) evaluates cost efficiency as the memory context becomes longer and more frequently invalidated. Figure 3(a) varies the number of memory chunks in the memory context, while Figure 3(b) varies the invalidated memory ratio $\rho ,$ defined as the fraction of memory chunks affected between consecutive action decisions. Full Rebuild shows the steepest TTFT growth because it recomputes the full memory context at every step. CacheBlend, KEEP, and LRAgent reduce this cost partially, but their gains are limited by narrower reuse assumptions. CacheBlend depends on reusable non-prefix segments, KEEP relies on invalidating affected memory regions, and LRAgent mainly handles procedural LoRA switching without explicitly reconciling updates to declarative and working-memory contents.

In contrast, HEMER achieves the lowest TTFT by localizing memory-content updates to affected KV ranges and preserving matched chunks through RoPE-based remapping. The reuse-ratio results explain this trend. Baselines rapidly lose reusable cache regions as the invalidated memory ratio increases, whereas HEMER maintains roughly 1.8× higher average reuse and retains more than half of its initial reuse at the largest invalidation ratio. These results reflect the interdependent roles of our components. The memory schema identifies affected regions induced by memory-content updates, chunk-level KV reconciliation preserves positional consistency under insertion, eviction, and reordering, and adapter-stable KV reconciliation supports procedural switches. Together, they enable efficient KV reconciliation without full cache reconstruction.

Operation-level cost decomposition. Table 3 decomposes cache-processing cost by reconciliation operation, replaying the sequences of memory-content updates collected from ALFWorld episodes, where the reconciliation plan $\Delta _ { t }$ maps each update to insertion, eviction, reordering, recomputation, or reuse. Most of the memory context remains reusable across decision steps: 78.3% of the processed tokens are reused and 7.9% are relocated with RoPE-based remapping, both without recomputation, while only 4,219 tokens, 5.6% of the 74,784 tokens that Full Rebuild recomputes over the same updates, are recomputed. Because insertion and eviction costs are identical to Full Rebuild by construction, the entire reduction in cache-processing time, from 199.4 to 25.8 seconds, comes from recomputation avoided on schema-compatible KV ranges. Assembling the reconciled KV cache $K _ { t }$ adds only 0.5 seconds in total, indicating low cache-assembly overhead relative to recomputation. Note that the reuse ratio in Table 1 is a per-step average over episodes, whereas the token counts here aggregate all replayed memory-content updates.

Table 3: Analysis on cache-processing cost across reconciliation operations. HEMER recomputes affected KV ranges, whereas Full Rebuild recomputes the full context.
<table><tr><td></td><td></td><td>Reuse</td><td>Reorder</td><td>Recompute</td><td>Insert</td><td>Evict</td><td>Assemble  $\displaystyle \kappa _ { t }$ </td><td>Total</td></tr><tr><td rowspan="3">HEMER</td><td>Events</td><td>5,167</td><td>528</td><td>418</td><td>408</td><td>113</td><td></td><td> $^ { 6 , 6 3 4 }$ </td></tr><tr><td>Tokens</td><td>64,128</td><td>6,437</td><td>4,219</td><td>5,358</td><td>1,742</td><td></td><td>81,884</td></tr><tr><td>Time (s)</td><td>0.0</td><td>0.0</td><td>12.9</td><td>12.4</td><td>0.0</td><td>0.5</td><td>25.8</td></tr><tr><td rowspan="3">Full Rebuild</td><td>Events</td><td>0</td><td>0</td><td>6,113</td><td>408</td><td>113</td><td></td><td>6,634</td></tr><tr><td>Tokens</td><td>0</td><td>0</td><td>74,784</td><td>5,358</td><td>1,742</td><td></td><td>81,884</td></tr><tr><td>Time (s)</td><td>一</td><td>一</td><td>186.5</td><td>12.4</td><td>0.0</td><td>0.5</td><td>199.4</td></tr></table>

Table 4: Ablation on HEMER. $\cdot _ { \mathrm { { W } } } / \mathrm { { o } } . \mathrm { { ? } }$ denotes the removal of a component.
<table><tr><td>Variant</td><td>SR (%)</td><td>Latency (s/step)</td><td></td><td></td><td>Reuse ratio(%) Active ratio (%) Diversity ratio(%)</td></tr><tr><td>Ours</td><td> ${ \bf 7 1 . 4 3 \pm 2 . 0 5 }$ </td><td> ${ \bf 0 . 1 8 \pm 0 . 0 2 }$ </td><td> ${ \bf 7 5 . 3 0 \pm 4 . 3 3 }$ </td><td> ${ \bf 6 7 . 6 9 \pm 2 . 2 7 }$ </td><td> ${ \bf 7 1 . 4 3 \pm 3 . 3 2 }$ </td></tr><tr><td>w/o. Memory Schema</td><td> $6 2 . 6 7 \pm 3 . 4 9$ </td><td> $0 . 3 8 \pm 0 . 1 2$ </td><td> $5 1 . 3 9 \pm 5 . 8 4$ </td><td> $5 7 . 2 0 \pm 4 . 1 2$ </td><td> $6 1 . 0 0 \pm 5 . 0 3$ </td></tr><tr><td>w/o. Chunk-Level KV Reconciliation</td><td> $5 9 . 4 0 \pm 4 . 2 7$ </td><td> $0 . 4 7 \pm 0 . 0 5$ </td><td> $4 2 . 7 5 \pm 6 . 2 0$ </td><td> $5 5 . 1 0 \pm 4 . 7 5$ </td><td> $5 9 . 2 5 \pm 5 . 6 1$ </td></tr><tr><td>w/o. Adapter-Stable KV Reconciliation</td><td> $6 4 . 8 5 \pm 3 . 5 8$ </td><td> $0 . 2 8 \pm 0 . 0 3$ </td><td> $6 6 . 4 0 \pm 4 . 9 2$ </td><td> $6 2 . 7 1 \pm 3 . 6 6$ </td><td> $6 5 . 1 5 \pm 4 . 4 4$ </td></tr></table>

Further analysis in Appendix C.2 evaluates scalability under increasing memory chunk counts and dependency density, as well as the memory footprint and switching overhead of procedural adapters. Appendix C.2 further decomposes per-step inference latency, showing that the reduction in local memory-context recomputation accounts for most of the observed TTFT improvement.

## 5.4 Ablation

Table 4 ablates the main components of HEMER. Removing the memory schema decreases SR from 71.43% to 62.67% and reuse from 75.30% to 51.39%. This shows that anchoring memory contents as typed chunks with KV ranges is necessary for identifying which contents remain reusable after memory-content updates. Without this schema, the model cannot reliably map memory contents to affected KV ranges, leading to lower reuse, higher latency, and a lower active action ratio.

Removing chunk-level KV reconciliation causes the largest degradation, reducing SR to 59.40% and reuse to 42.75%, while increasing latency to 0.47 seconds per step. This indicates that the reconciliation plan is the main mechanism for localizing memory-content updates. When chunk-level reconciliation is removed, insertions, evictions, and reorderings can no longer be handled by updating only affected KV ranges, forcing broader recomputation and making cache reuse less stable.

Removing adapter-stable KV reconciliation also degrades performance, but differently. The reuse ratio remains high at 66.40%, yet SR drops to 64.85% and latency increases to 0.28 seconds per step. This suggests that reusing many KV states is insufficient when procedural memory changes the active adapter. Without adapter-stable reconciliation, KV states computed under a previous adapter can become incompatible with the current procedural configuration, reducing decision quality even when cache reuse is preserved. Overall, the ablation results show that the memory schema identifies reusable contents, chunk-level reconciliation localizes memory-content updates, and adapter-stable reconciliation ensures reused states remain compatible under procedural changes.

## 6 Conclusion

We presented HEMER, a heterogeneous memory reconciliation framework for foundation modelbased embodied agents. HEMER anchors memory contents as typed memory chunks with associated KV ranges and reconciles memory-content updates through localized KV operations. By combining chunk-level and adapter-stable KV reconciliation, it preserves compatible KV states across changes to declarative, procedural, and working memory contents without rebuilding the full KV cache. Experiments on ALFWorld, Habitat, and real-world tabletop manipulation tasks show that HEMER improves task success and reduces per-step inference time over strong baselines. These results demonstrate that reconciling dynamically updated memory contexts with reusable KV states supports efficient and stable inference in long-horizon embodied tasks.

Limitation and future work. Our evaluation focuses on single-agent embodied tasks, where memory contexts are updated across successive decision steps. In multi-agent embodied systems, concurrent updates to shared memory contents may introduce consistency issues between memory contexts and cached KV states, which are not addressed by the current framework. As future work, we plan to extend HEMER to multi-agent embodied systems by incorporating concurrency control for shared memory updates and KV reuse. This extension may support efficient memory reconciliation across agents while preserving valid KV states under concurrent updates.

## Acknowledgment

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (RS-2025-02218768, Accelerated Insight Reasoning via Continual Learning (70%); RS-2025-25442569, AI Star Fellowship Support Program (Sungkyunkwan Univ.) (10%)), by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25474409, 10%), and by Samsung Electronics Co., Ltd. (10%).

## References

[1] Michael Ahn et al. “Do as i can, not as i say: Grounding language in robotic affordances”. In: arXiv preprint arXiv:2204.01691 (2022).

[2] Shunyu Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models”. In: The Eleventh International Conference on Learning Representations (ICLR). 2023.

[3] Chan Hee Song et al. “Llm-planner: Few-shot grounded planning for embodied agents with large language models”. In: Proceedings ofthe IEEE/CVF international conference on computer vision (ICCV). 2023.

[4] Wenlong Huang et al. “Inner Monologue: Embodied Reasoning through Planning with Language Models”. In: 6th Annual Conference on Robot Learning (CoRL). 2022.

[5] Ishika Singh et al. “ProgPrompt: Program generation for situated robot task planning using large language models”. In: Autonomous Robots (2023).

[6] Guanzhi Wang et al. “Voyager: An Open-Ended Embodied Agent with Large Language Models”. In: Transactions on Machine Learning Research (TMLR) (2024).

[7] Noah Shinn et al. “Reflexion: language agents with verbal reinforcement learning”. In: Thirtyseventh Conference on Neural Information Processing Systems (NeurIPS). 2023.

[8] Joon Sung Park et al. “Generative agents: Interactive simulacra of human behavior”. In: Proceedings ofthe 36th annual acm symposium on user interface software and technology. 2023.

[9] Charles Packer et al. “MemGPT: Towards LLMs as Operating Systems”. In: ArXiv (2023).

[10] Wanjun Zhong et al. “Memorybank: Enhancing large language models with long-term memory”. In: Proceedings ofthe AAAI conference on artificial intelligence (AAAI). 2024.

[11] Theodore Sumers et al. “Cognitive Architectures for Language Agents”. In: Transactions on Machine Learning Research (TMLR) (2024).

[12] Wujiang Xu et al. “A-Mem: Agentic Memory for LLM Agents”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS). 2025.

[13] Wonje Choi et al. “NeSyC: A Neuro-symbolic Continual Learner for Complex Embodied Tasks in Open Domains”. In: Proceedings ofthe 13th International Conference on Learning Representations (ICLR 2025). 2025.

[14] Neal J Cohen and Larry R Squire. “Preserved learning and retention of pattern-analyzing skill in amnesia: Dissociation of knowing how and knowing that”. In: Science (1980).

[15] Wonje Choi, Jooyoung Kim, and Honguk Woo. “NeSyPr: Neurosymbolic Proceduralization for Efficient Embodied Reasoning”. In: Advances in Neural Information Processing Systems (NeurIPS 2025). 2025.

[16] Edward J Hu et al. “LoRA: Low-Rank Adaptation of Large Language Models”. In: International Conference on Learning Representations (ICLR). 2022.

[17] Alan Baddeley. “Working memory”. In: Science (1992).

[18] Reiner Pope et al. “Efficiently scaling transformer inference”. In: Proceedings of machine learning and systems (MLSys) (2023).

[19] Woosuk Kwon et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention”. In: Proceedings of the 29th Symposium on Operating Systems Principles (SOSP). 2023.

[20] Endel Tulving et al. “Episodic and semantic memory”. In: Organization of memory (1972).

[21] Jiayi Yao et al. “Cacheblend: Fast large language model serving for rag with cached knowledge fusion”. In: Proceedings of the twentieth European conference on computer systems (EuroSys). 2025.

[22] Jingbo Yang et al. “KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS). 2025.

[23] Yang Liu et al. “CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving”. In: 24th USENIX Conference on File and Storage Technologies (FAST 26). USENIX Association, Feb. 2026.

[24] Jianlin Su et al. “RoFormer: Enhanced transformer with Rotary Position Embedding”. In: Neurocomputing (2024).

[25] Hyesung Jeon, Hyeongju Ha, and Jae-Joon Kim. “LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents”. In: arXiv preprint arXiv:2602.01053 (2026).

[26] Allison Li et al. “Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA”. In: arXiv preprint arXiv:2512.17910 (2025).

[27] Zhiyu Li et al. “Memos: A memory os for ai system”. In: arXiv preprint arXiv:2507.03724 (2025).

[28] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. “Titans: Learning to Memorize at Test Time”. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS). 2025.

[29] Junhao Hu et al. “EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models”. In: Forty-second International Conference on Machine Learning (ICML). 2025.

[30] Yuhan Liu et al. “CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving”. In: ACM SIGCOMM ’24 (2024).

[31] Yihua Cheng et al. “LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference”. In: arXiv preprint arXiv:2510.09665 (2025).

[32] Mohit Shridhar et al. “Alfworld: Aligning text and embodied environments for interactive learning”. In: arXiv preprint arXiv:2010.03768 (2020).

[33] Rui Yang et al. “EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents”. In: Forty-second International Conference on Machine Learning (ICML). 2025.

[34] Andrew Szot et al. “Habitat 2.0: training home assistants to rearrange their habitat”. In: Proceedings ofthe 35th International Conference on Neural Information Processing Systems (NIPS). 2021.

[35] Shengtao Zhang et al. “Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory”. In: arXiv preprint arXiv:2601.03192 (2026).

[36] Michał Zawalski et al. “Robotic Control via Embodied Chain-of-Thought Reasoning”. In: 8th Annual Conference on Robot Learning (CoRL). 2024.

[37] Andrew Zhao et al. “Expel: Llm agents are experiential learners”. In: Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI). 2024.

[38] Zihao Wang et al. “ JARVIS-1: Open-World Multi-Task Agents With Memory-Augmented Multimodal Language Models ”. In: IEEE Transactions on Pattern Analysis & Machine Intelligence (TPAMI) (2025).

[39] Runnan Fang et al. “Memp: Exploring Agent Procedural Memory”. In: arXiv preprint arXiv:2508.06433 (2025).

[40] Minghao Chen et al. “AutoManual: Generating Instruction Manuals by LLM Agents via Interactive Environmental Learning”. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS). 2024.

[41] Zhenhang Yuan, Shenghai Yuan, and Lihua Xie. “RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy”. In: arXiv preprint arXiv:2603.17831 (2026).

[42] Gyeong-In Yu et al. “Orca: A distributed serving system for Transformer-Based generative models”. In: 16th USENIX symposium on operating systems design and implementation (OSDI 22). 2022.

[43] In Gim et al. “Prompt cache: Modular attention reuse for low-latency inference”. In: Proceedings ofMachine Learning and Systems (MLSys) (2024).

[44] Saehun Chun et al. “Functional Cache Grafting: Robust and Rapid Code-Policy Synthesis for Embodied Agents”. In: Proceedings of the 43rd International Conference on Machine Learning (ICML 2026). 2026.

[45] Lianmin Zheng et al. “SGLang: efficient execution of structured language model programs”. In: Proceedings ofthe 38th International Conference on Neural Information Processing Systems (NIPS). 2024. ISBN: 9798331314385.

[46] Zhekai Duan et al. “Fast ecot: Efficient embodied chain-of-thought via thoughts reuse”. In: arXiv preprint arXiv:2506.07639 (2025).

[47] Zebin Yang et al. “KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning”. In: arXiv preprint arXiv:2602.23592 (2026).

[48] Zhenyu Zhang et al. “H O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models”. In: Advances in Neural Information Processing Systems (NIPS). 2023.

[49] Guangxuan Xiao et al. “Efficient Streaming Language Models with Attention Sinks”. In: The Twelfth International Conference on Learning Representations (ICLR). 2024.

[50] Ying Sheng et al. “SLoRA: Scalable Serving of Thousands of LoRA Adapters”. In: Conference on Machine Learning and Systems (MLSys). 2024.

[51] Lequn Chen et al. “Punica: Multi-tenant lora serving”. In: Proceedings of Machine Learning and Systems (MLSys). 2024.

[52] Bingyang Wu et al. “dLoRA: Dynamically orchestrating requests and adapters for LoRA LLM serving”. In: 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 2024.

[53] Hang Zhang et al. “Improving the serving performance of multi-lora large language models via efficient lora and kv cache management”. In: arXiv preprint arXiv:2505.03756 (2025).

[54] AI@Meta. Llama 3.2 Model Card. 2024.

[55] Shuai Bai et al. “Qwen3-vl technical report”. In: arXiv preprint arXiv:2511.21631 (2025).

[56] Mohit Shridhar et al. “ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). June 2020.

[57] Homanga Bharadhwaj et al. “RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking”. In: 2024 IEEE International Conference on Robotics and Automation (ICRA). 2024.

## A Experiment setting

## A.1 ALFWorld

![](images/41a47cd2edb766ddbd6de2c599ca8d87f8159e5a6e3bad9fe9780c85dd72cd5c.jpg)  
(a) Bathroom scene example

![](images/fa8dd3ebf7e2f1076e7bd0c88f2ea3ed1d466f76ad19b11e0d3cf8a49de0cf94.jpg)  
(b) Bedroom scene example

![](images/6a502d5fbadf28df88645abd5d25aba60a1b3ca82b533f3ee7e43117f6df70fb.jpg)  
(c) Kitchen scene example

![](images/0f710d697f3e571ea7b6113d2d3fad01532aeb2478667566e7b60f09e59f63a1.jpg)  
(d) Kitchen scene example  
Figure 4: Example ALFWorld scenes.

Environment. ALFWorld [32] is a text-based household environment in which an agent solves natural-language instructions through a discrete language action interface, as shown in Figure 4. It inherits the task structure of ALFRED, exposing textual observations and an admissible-command list at every decision step. Each episode starts from the initial observation produced by the TextWorld backend. At every step, the agent receives a textual observation, an admissible-command list with about 30 commands on average, and selects a single command. We cap each episode at 30 steps, after which the episode is marked as failed if the goal predicate is not satisfied. Across all baselines and our method, observations are normalized through the same name-canonicalization layer. For example, desk\_bar\_\_+01\_dot\_49\_\_... is rewritten to desk\_1, so prompt content depends on the agent’s behavior rather than positional artifacts in receptacle names. ALFWorld is released under the MIT License, and we used it in accordance with the terms specified.

Task. Each episode is one of six task types: put, put-two, heat, cool, clean, and examine. The instruction templates and text action grammar used in the ALFWorld experiments are summarized in Table 5. Procedural memory uses room-specific LoRA adapters for bathrooms, bedrooms, and living rooms, together with a shared adapter. The agent selects the adapter according to the room context and uses the shared adapter when no room-specific adapter applies. ALFWorld therefore provides a text-only setting in which memory-content updates arise through changing observations, admissible actions, object locations, and subgoal progress.

Table 5: Instructions and actions in ALFWorld experiment.
<table><tr><td></td><td>Template</td><td>Example</td></tr><tr><td>Instruction</td><td>Put Object Put Two Objects Heat Object Cool Object Clean Object Examine Object</td><td>“&quot;Put the apple on the dining table.&quot; “Put two pens in the drawer.&quot; &quot;Heat the potato with the microwave.&#x27; &quot;Cool the lettuce with the fridge.&quot; &quot;Clean the mug with the sink.&quot; “Examine the lamp with the desk lamp.&quot;</td></tr><tr><td>Text Action Grammar Cool Object</td><td>Go to Receptacle Take Object Move Object Open Receptacle Close Receptacle Use Object Clean Object Heat Object</td><td>go to recep take object from recep move object to recep open recep close recep use object clean object with recep heat object with recep</td></tr></table>

## A.2 EmbodiedBench Habitat

Environment. EmbodiedBench Habitat [33, 34] shown in Figure 5 exposes ReplicaCAD apartment scenes through a high-level skill action space. Unlike ALFWorld, Habitat provides RGB observations, so the agent must condition on visual input in addition to language goals. In our evaluation, episodes are constructed from six ReplicaCAD apartments, apt\_0–apt\_5, with randomized object placements. Visual observations are rendered as 512×512 head-mounted RGB images and downsampled to the resolution accepted by the backbone vision encoder. The agent acts through high level skills: navigate\_to(receptacle\_or\_object), pick(object), place(receptacle), open(receptacle), and close(receptacle), as summarized in Table 6. Each skill is internally executed by a low-level controller, while the high-level decision step is the unit at which HEMER reconciles memory chunks. Each episode is capped at 50 high-level steps. Habitat-Sim and Habitat-Lab are released under the MIT License, and we used them in accordance with the terms specified in their official repositories. ReplicaCAD scene assets are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License, and we used them in accordance with its terms.

Task. EmbodiedBench Habitat includes six evaluation sets that emphasize different aspects of compositional reasoning: base, common sense, complex instruction, spatial relationship, visual appearance, and long horizon. Each set contains 50 episodes, and we report aggregate scores across all six sets in the main results. Habitat provides a complementary multimodal setting where memory-content updates arise from both visual observations and high-level skill transitions. The apartment layout supplies persistent declarative structure, while object visibility, object state, and manipulation outcomes change across decision steps. This produces a mixture of stable memory contents, newly grounded visual facts, and updated working-memory states. The high-level skill interface also makes procedural boundaries explicit: switching from navigation to manipulation, or from opening to placing, changes the procedural context that determines the active adapter. For multimodal backbones, HEMER restricts RoPE-based remapping to text-side memory chunks and encodes current visual observations normally at each decision step.

![](images/b58dbe91b51627eaf1d0030b32157d3ea01267556978736b2aee92b9cf3d5ad3.jpg)

(a) Top-view image of an EmbodiedBench Habitat environment  
![](images/63966f44a51ac6e425f578742f8352139b33d2a7ef24ef1a6e8463b1c59f2a7b.jpg)  
(b) Task example: “deposit apples on tv stand”  
Figure 5: Example EmbodiedBench Habitat scene and task observations.

Table 6: Instructions and skill APIs in EmbodiedBench Habitat experiment.
<table><tr><td></td><td>Template</td><td>Example</td></tr><tr><td rowspan="6">Instruction</td><td>Base Task</td><td>&quot;Put the apple on the table.&quot;</td></tr><tr><td>Common-sense Task</td><td>&quot;Place the mug near the coffee machine.&quot;</td></tr><tr><td>Complex Instruction</td><td>&quot;Pick up the book, open the cabinet, and place it inside.&quot;</td></tr><tr><td>Spatial Relationship</td><td>&quot;Put the bowl to the left of the plate.&quot;</td></tr><tr><td>Visual Appearance</td><td>&quot;Pick up the red object on the counter.&quot;</td></tr><tr><td>Long-horizon Task</td><td>&quot;Move the apple from the kitchen counter to the TV stand.&quot;</td></tr><tr><td rowspan="5">High-level Skill APIs</td><td>Navigate to Object or Receptacle</td><td>navigate_to(receptacle_or_object)</td></tr><tr><td>Pick Object</td><td>pick(object)</td></tr><tr><td>Place Object</td><td>place(receptacle)</td></tr><tr><td>Open Receptacle</td><td>open(receptacle)</td></tr><tr><td>Close Receptacle</td><td>close(receptacle)</td></tr></table>

## A.3 Real-world Test

Environment. Our real-world setup employed a UR7e robotic arm mounted on a tabletop workspace. For perception, we used two Intel RealSense D435 RGB-D cameras: one attached to the gripper for close-range, viewpoint-aligned observation during manipulation, and another mounted overhead in a bird’s-eye configuration to capture the global scene layout. The two depth streams were fused into a unified point cloud, yielding a high-resolution 3D reconstruction of the workspace that served as the basis for object localization and visual grounding. Perception and control were integrated in a single real-time loop, ensuring consistent behavior throughout manipulation.

![](images/600d77203bc98257a7508f3414a430b93d868a8ea9334972396822e7b1fc3f97.jpg)

<table><tr><td>ID</td><td>Object</td></tr><tr><td>1</td><td>Bread slices</td></tr><tr><td>2</td><td>Spatula</td></tr><tr><td>3</td><td>Frying pan</td></tr><tr><td>4</td><td>Cookie</td></tr><tr><td>5</td><td>Cup</td></tr><tr><td>6</td><td>Vanilla syrup bottle</td></tr><tr><td>7</td><td>Classic syrup bottle</td></tr><tr><td>8</td><td>Coffee machine</td></tr><tr><td>9</td><td>Water dispenser</td></tr></table>

(a) Example of Dual-Section Kitchen-Café environment  
![](images/67255c51edf252c716e34f126d6c13dd051ee8fa8e5e2b4191e2b3c13ef4bf42.jpg)

<table><tr><td>ID</td><td>Object</td></tr><tr><td>1</td><td>Ice container</td></tr><tr><td>2</td><td>Ice scoop</td></tr><tr><td>3</td><td>Vanilla syrup bottle</td></tr><tr><td>4</td><td>Classic syrup bottle</td></tr><tr><td>5</td><td>Coffee machine</td></tr><tr><td>6</td><td>Water dispenser</td></tr><tr><td>7</td><td>cup</td></tr><tr><td>8</td><td>coffee cup</td></tr><tr><td>9</td><td>water cup</td></tr></table>

(b) Example of Single-Section Beverage Station set environment  
Figure 6: Example real-world environments.

Task. Real-world tasks are executed using a compact set of predefined task-level skill APIs. As summarized in Table 7, these APIs cover common tabletop manipulation operations, including pose-conditioned motion, object picking, object placement, pick-and-place transfer, directional pushing, tool grasping, and object flipping. Each skill exposes only the task-relevant arguments needed for planning, such as the target object, source and target locations, end-effector pose, or push direction. This abstraction hides low-level control details while preserving sufficient flexibility for the LLM-based agent to compose actions across cooking and beverage-preparation tasks.

Real-world experiments were conducted in two tabletop environments: a Dual-Section Kitchen-Café environment and a Single-Section Beverage Station environment. Each of the two evaluated longhorizon tasks is repeated with varying object configurations. At the beginning of each trial, objects were sampled from the corresponding object pool and placed at randomized positions within the workspace. We repeat each task across multiple randomized physical trials (N = 5 per task) to evaluate robustness under varied instructions and spatial layouts. Simulator results are averaged over five seeds, while real-world results are averaged over these physical trials.

The first environment is a Dual-Section Kitchen-Café setup, which combines a kitchen section and a café section within a single tabletop workspace. The kitchen section includes objects such as bread slices, a spatula, and a frying pan, while the café section includes a cup, syrup bottles, a coffee machine, and a water dispenser. Tasks in this environment require the agent to compose skills across the two sections, such as placing bread on the pan, flipping toast, moving a cup to the appropriate station, and operating beverage-related devices. This setup is designed to evaluate whether HEMER can compose procedural knowledge across heterogeneous but co-located task domains, while reusing shared manipulation skills such as picking, placing, pushing, and tool grasping.

The second environment is a Single-Section Beverage Station setup, which focuses only on the café-side task domain. It contains beverage-related objects such as an ice container, ice scoop, syrup bottles, a coffee machine, a water dispenser, and a cup. Tasks in this environment focus on beverage preparation, including placing the cup, checking the state of the ice, adding ingredients, operating the dispenser or coffee machine through directional pushes, and manipulating tools when needed. Compared with the dual-section setup, this environment isolates a single procedural domain and is used to evaluate whether HEMER can efficiently execute and reuse a focused set of beveragepreparation skills while adapting to local state changes, such as whether ice is available or needs to be added. Example images of both environments are shown in Figure 6.

## Table 7: Instructions and APIs in Real-world Experiment

<table><tr><td></td><td>Template</td><td>Example</td></tr><tr><td>Instruction</td><td>Pick Object Pick &amp; Place Place Object Push Grasp Tool</td><td>&quot;Pick up the cup from the table.&quot; &quot;Pick up a bread slice and place it on the frying pan.&quot; &quot;Place the cup under the water dispenser.&quot; &quot;Push the dispenser lever downward.&quot; &quot;Grasp the spatula.&quot;</td></tr><tr><td></td><td>Serve Beverage Move to Pose [x,y,z,r,p,y] Pick [object, location] Place [object, location] Pick &amp; Place [object, start, target] Push [location, direction]</td><td>&quot;Flip the toast on the frying pan.&quot; &quot;Prepare a drink using water, syrup, and espresso.&quot; pose(x, y, z, r, p, y) pick(object, location) place(object, location) pick_and_place(object, start_location, target_location) push(location, direction)</td></tr></table>

## A.4 Real-world experimental details

## A.4.1 Dual-Section Kitchen-Café

The first real-world environment includes a dual-section kitchen-café environment designed for a longhorizon task of "Make a vanilla Americano and a slice of toast". As shown in Figure 7, the workspace is divided into a Kitchen section and a Café section: the kitchen side contains bread slices, a spatula, a frying pan, and cookies, while the café side contains a cup, vanilla syrup bottle, classic syrup bottle, coffee machine, and water dispenser. Each section is associated with a distinct LoRA adapter, reflecting the different procedural knowledge required for kitchen-side food preparation and café-side beverage preparation. Based on this setting, the agent jointly manages heterogeneous memory over a shared task context. This environment therefore evaluates whether the agent can coordinate sectionspecific LoRA procedures and heterogeneous memory contents across two functionally different regions while executing a consistent long-horizon policy for preparing both the toast and the vanilla Americano.

## A.4.2 Single-Section Beverage Station

The second real-world environment is a single-section beverage station designed to evaluate whether the agent can consistently update dynamic object states while performing repeated beverage-making tasks. As shown in Figure 8, the station contains an ice container, ice scoop, syrup bottles, a coffee machine, water dispenser, and multiple cups. The task requires the agent to prepare six iced coffees sequentially within the same workspace, reusing the same beverage-making procedure across subtasks. The agent must repeatedly execute the same beverage-making procedure while monitoring changes in the shared resources, especially the ice container. Since the ice is reused across multiple subtasks and can change in quantity or condition over time, the agent needs to recognize whether sufficient ice remains, detect melting or depletion, and adapt subsequent actions accordingly.

## A.5 Baseline

To provide a comprehensive comparison, we group the baselines by their assumptions about memorycontent updates and reusable KV states during embodied inference.

• General embodied reasoning baselines generate actions from observations, goals, and interaction history without explicit memory reconciliation.

![](images/146ea7a645c9a9cc097eb845cd9afca5f8c691ae0b6e2ac1d7b48d9c7021c60b.jpg)

Make me an Americano with vanilla syrup and a piece of toast.

Subtask1:Americano with vanilla syrup

Subtask2: A piece of toast.

![](images/fc04daec3b5d03c401a65dc23122d52de09800ed9d03540df957eccab7dda2d4.jpg)

![](images/66fb806c1948b8af9e43c8100c152e37f1ff6f3e72f3a59bb124a598e1979e22.jpg)

Figure 7: Real-world Dual-Section Kitchen-Café. The sequence includes two main subtasks: (1) making coffee with vanilla syrup and (2) making toast.

– ReAct [2] interleaves reasoning traces and actions for step-by-step embodied decision making.

– Reflexion [7] uses verbal self-reflection from previous failures to guide future action decisions.

• Memory-augmented embodied baselines retrieve past experience or task-relevant memory to support long-horizon action selection.

– MemRL [35] retrieves episodic memory from trajectories to improve embodied decision making.

• KV cache reuse baselines reduce inference cost by reusing cached states under repeated or partially shared contexts.

– CacheBlend [21] reuses non-prefix segments and selectively recomputes boundary regions.

– KEEP [47] reuses KV states across embodied reasoning by invalidating affected cache portions.

• Adapter-based baselines improve efficiency under procedural skill switching.

– LRAgent [25] represents procedural skills with LoRA adapters and supports adapter switching during embodied inference.

We compare HEMER against six baselines that span both agent-level memory mechanisms and cachelevel reuse strategies. All baselines share the same backbone model and decoding configuration as HEMER (Llama-3.2-3B-Instruct on ALFWorld; Qwen3-VL-4B-Instruct on Habitat; greedy decoding) so that any difference reflects memory or cache handling rather than model capacity.

![](images/c9d68cfce31ee39c5f2e63df385e84e320440521844e7a04598ed1a40162f353.jpg)  
Figure 8: Real-world Single-Section Beverage Station. The sequence is divided into six repeated coffee-making subtasks, requiring the robot to monitor the state changes of ice while preparing each beverage within a single beverage station.

ReAct [2]. Reasoning-and-acting prompting baseline. The agent emits a Thought followed by an Action at every step, with the conversation history accumulated as a chat trace. ReAct has no explicit memory module beyond the natural in-context history, so the cache contains only the running conversation and is rebuilt as the prompt grows.

Reflexion [7]. ReAct extended with verbal self-reflection on episode failure. After a failed episode, the agent generates a free-form reflection that is stored and retrieved in subsequent episodes. Reflexion tests whether self-reflective episodic feedback suffices in the absence of structured memory composition.

MemRL [35]. Memory-augmented decision making with episodic memory and Q-learned utility, but no structured KV reuse. We use the published MemRL retrieval policy (top-k word-overlap retrieval with ϵ-greedy exploitation) and inject retrieved trajectories as in-context successful and failed memory snippets. MemRL serves as the strongest agent-level memory baseline.

CacheBlend [21]. Segment-level KV reuse that fuses precomputed chunks at chunk-boundary positions with selective recompute of the boundary tokens. We adapt CacheBlend to embodied trajectories by treating each per-step memory composition as a chunked input; off-boundary updates are ignored, and at chunk-boundary steps the trailing chunks are re-prefilled with selective calibration. We use the boundary period K = 5 throughout, matching the recompute budget of the original system.

KEEP [47]. Embodied-memory cache management with internal-trigger invalidation. KEEP evicts memory regions that conflict with the agent’s most recent action and keeps unaffected regions. We implement KEEP-style invalidation as the eviction-only subset of HEMER’s policy: the dispatcher honors only evict and adapter\_switch edits, which mirrors KEEP’s self-induced invalidation scope without re-injecting the inserted or reordered content.

LRAgent [25]. Procedural memory through LoRA adapter switching with shared-attention crossadapter cache. LRAgent reuses the prefill cache across adapter switches but does not edit the cache for declarative or working-memory updates. We use the same adapter pool as HEMER while retaining LRAgent’s adapter-level cache-reuse policy.

Faithfulness statement. CacheBlend, KEEP, and LRAgent are implemented according to their public descriptions and adapted to the embodied-agent evaluation setting. Because the original systems target different serving stacks or task interfaces, we keep their core reuse assumptions unchanged while integrating them into our shared observation, action, and memory-context pipeline. Within this harness, all baselines and HEMER use the same observation pipeline, action filter, prompt template, backbone model, decoding setting, and admissible-command set when applicable. Thus, performance differences primarily reflect how each method handles memory use and KV reuse under dynamically changing embodied contexts.

## A.6 Metrics

We report six metrics that characterize task performance, action behavior, KV reuse, and inference efficiency.

Success Rate (SR). For ALFWorld and Habitat, SR measures binary end-to-end task completion:

$$
\mathrm { S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ \mathrm { g o a l \_ s a t i s f i e d } _ { i } ] ,
$$

where N is the number of evaluation episodes and goal\_satisfied is the binary task-completion indicator returned by the environment. For real-world tasks, we use goal-conditioned success rate, which measures the proportion of satisfied goal conditions within each trial, averaged across physical trials.

Latency. Latency is the average wall-clock inference time per decision step. It is measured after the step input is finalized and includes KV reconciliation, prefill, local recomputation, and decoding. Latency is reported in seconds per step.

Time-to-first-token (TTFT). We report TTFT as the sum of measured KV-cache update execution and fresh-tail prefill times through the first generated token. It includes required prefill, local recomputation, and cache assembly. For cache-reuse methods, KV ranges cached at previous steps are available before timing begins. It excludes environment execution, memory retrieval, prompt construction, and schema-diff planning.

Reuse ratio. Reuse ratio measures the fraction of previous-step cached tokens whose KV states are preserved at the current step. Let $S _ { t }$ be the number of tokens in the previous cache and $C _ { t }$ the number of those tokens whose KV states are not preserved, including evicted and locally recomputed tokens. We average over T steps with an available previous cache:

$$
{ \mathrm { R e u s e } } = 1 - { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \frac { C _ { t } } { S _ { t } } } .
$$

Inserted tokens are excluded, while RoPE-remapped KV states count as preserved. A value of 1 indicates that all previous-step cached KV states are preserved, while 0 indicates that none are preserved.

Active Action Ratio. Active action ratio measures the fraction of generated actions that are expected to advance the task or change the environment state. We map benchmark-specific action strings or skill API calls to a shared set of task-progressing action categories. For episode i with action sequence $\mathcal { A } _ { i } = \left( a _ { i , 1 } , \ldots , a _ { i , T _ { i } } \right)$ , we compute

$$
\mathrm { A c t i v e A c t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } { \bf 1 } \big [ a _ { i , t } \in { \mathcal A } ^ { \mathrm { a c t } } \big ] .
$$

A higher value indicates that the agent more frequently performs task-progressing actions rather than passive, invalid, or redundant steps.

Action Diversity Ratio. Action diversity measures the average per-episode ratio of unique actions among all generated actions. For episode i with action sequence $\mathcal { A } _ { i } = \left( a _ { i , 1 } , \ldots , a _ { i , T _ { i } } \right)$ , we compute

$$
{ \mathrm { D i v e r s i t y } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \frac { | { \mathrm { U n i q u e } } ( A _ { i } ) | } { T _ { i } } } .
$$

A higher value indicates that the agent uses a broader range of actions during the trajectory.

## B HEMER: Heterogeneous Memory Reconciliation

## B.1 Algorithm Details

Overview. Algorithm 1 summarizes the inference procedure of HEMER. At each decision step, the agent first obtains memory contents from declarative, procedural, and working memory modules and forms the memory context $X _ { t } .$ . Instead of rebuilding the full KV cache for ${ \bar { X } } _ { t } ,$ HEMER compares the current memory context with the previous one using schema fields attached to each memory chunk. The resulting reconciliation plan identifies inserted, evicted, and reordered chunks, maps these operations to KV ranges, and determines which cached states can be reused. A cached KV range is reused only when the corresponding memory content remains compatible under the schema. If the chunk is unchanged but only its position changes, HEMER applies RoPE-based remapping to correct the positional phase of cached key states. If the memory content, dependency annotation, or adapter identifier changes, the affected KV range is locally recomputed to avoid stale reuse.

## B.2 Implementation

System Configuration. Table 8 summarizes the software and hardware configuration used in our experiments.

Table 8: System configuration.
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Python</td><td>3.10.12</td></tr><tr><td>CUDA / cuDNN</td><td>CUDA 12.4 / cuDNN 9.1</td></tr><tr><td>PyTorch</td><td>2.5.1</td></tr><tr><td>Transformers</td><td>4.46.3</td></tr><tr><td>PEFT</td><td>0.13.2</td></tr><tr><td>GPU</td><td>2× NVIDIA RTX A6000 (48GB)</td></tr><tr><td>Random Seeds</td><td>0, 7, 42, 99, 123</td></tr></table>

Backbone Models. Table 9 summarizes the backbone and decoding configuration. For multimodal backbones, RoPE-based remapping is restricted to text-side memory chunks in the memory context. Visual tokens from the current observation are encoded normally at each decision step and are not position-remapped.

Algorithm 1 Heterogeneous Memory Reconciliation in HEMER   
Require: $o _ { t } , g , h _ { t } , X _ { t - 1 } , \mathcal { K } _ { t - 1 } ;$ memory modules $\overline { { \mathcal { M } ^ { \mathrm { d e c } } , \mathcal { M } ^ { \mathrm { p r o c } } , \mathcal { M } } }$ work   
Ensure: $a _ { t } , X _ { t } , \mathcal { K } _ { t }$   
1: Retrieve $D _ { t } , P _ { t } , W _ { t }$ using $\left( o _ { t } , g , h _ { t } \right)$ and construct $X _ { t } = \Gamma ( D _ { t } , P _ { t } , W _ { t } )$   
2: Index chunks $\{ c _ { i } ^ { t } \} _ { i = 1 } ^ { n _ { t } }$ with persistent IDs, positions, token spans, KV ranges, and the remaining   
schema metadata.   
3: if $\cdot t = 1$ or no previous cache is available then   
4: Prefill $X _ { t }$ as a single causal sequence to obtain $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$   
5: else   
6: Compute $\Delta _ { t } = \mathrm { D i f f } _ { S } ( X _ { t - 1 } , X _ { t } )$ by ID matching and schema checks, assigning INSERT,   
EVICT, RECOMPUTE, REORDER, or REUSE.   
7: Propagate RECOMPUTE to matched direct and transitive dependents under declared depen  
dencies, overriding REUSE or REORDER.   
8: Exclude EVICT ranges from the current cache layout and determine positions from $X _ { t }$   
9: for each chunk $c _ { i } ^ { t }$ in the order of $X _ { t }$ do   
10: if marked as INSERT or RECOMPUTE then   
11: Prefill or recompute its KV range once at its current position using preceding recon  
ciled KV ranges; allocate a new range for INSERT.   
12: else if marked as REORDER then   
13: Relocate its KV range; for eligible text-side keys shifted from $p _ { j }$ to $p _ { j } ^ { \prime }$ , apply $\tilde { k } _ { j } =$   
$R _ { \theta ( p _ { j } ^ { \prime } ) } R _ { \theta ( p _ { j } ) } ^ { - 1 } k _ { j }$   
14: else   
15: Preserve the corresponding KV range.   
16: end if   
17: end for   
18: Assemble $\textstyle { \mathcal { K } } _ { t }$ in the order of $X _ { t } .$   
19: end if   
20: Update the chunk-to-KV index.   
21: Generate $a _ { t } \sim F _ { \theta } ( \cdot \mid X _ { t } , o _ { t } , g )$ using $\textstyle { \mathcal { K } } _ { t }$   
22: return $a _ { t } , X _ { t } , \mathcal { K } _ { t } .$

System architecture. HEMER is implemented as an inference framework placed between the memory-augmented embodied agent and the backbone foundation model. At each decision step, the agent retrieves declarative, procedural, and working-memory contents, forms the memory context X , and passes it to HEMER for KV reconciliation. The framework maintains a chunk-level memory schema, maps each memory chunk to its KV range, and reconciles the previous KV cache with the current memory context before action generation. Environment execution and memory retrieval are handled by the agent loop, while HEMER handles cache reuse, affected-range detection, RoPE-based remapping, local recomputation, and adapter compatibility checks.

Memory schema and reconciliation rules. Each memory content is represented as a memory chunk and stored with schema fields used for reconciliation. The schema records each chunk’s persistent ID, memory module type, semantic role, dependency annotation, temporal origin, position, token span, KV range, supporting evidence, normalized content, and adapter identifier when applicable. HEMER computes $\Delta _ { t } = \mathrm { D i f f } _ { \mathcal { S } } ( X _ { t - 1 } , X _ { t } )$ by first matching chunks by persistent IDs and then comparing their schema fields rather than raw token overlap. Inserted chunks are prefilling targets, evicted chunks are removed, and reordered chunks are reused only when schema fields, normalized content, and adapter identifiers remain compatible. For schema-compatible position shifts, RoPE remapping corrects the cached key’s positional phase; unsafe cases are handled by local recomputation.

Table 9: Backbone and decoding configuration.
<table><tr><td>Setting</td><td>ALFWorld</td><td>Habitat &amp; Real-world</td></tr><tr><td>Model</td><td>Llama-3.2-3B-Instruct</td><td>Qwen3-VL-4B-Instruct</td></tr><tr><td>Parameters</td><td>3.2B</td><td>4B</td></tr><tr><td>Precision</td><td>bfloat16</td><td>bfloat16</td></tr><tr><td>Temperature</td><td>0.0 (greedy)</td><td>0.0 (greedy)</td></tr><tr><td>Max new tokens</td><td>256</td><td>256</td></tr><tr><td>Observation</td><td>Text</td><td>RGB image</td></tr></table>

Adapter compatibility. Procedural memory contents are associated with LoRA adapter configurations. HEMER records adapter identifiers in the memory schema and uses them as a compatibility condition for KV reuse. If a procedural change affects an adapter-dependent range, that range is locally recomputed to avoid reusing KV states from an incompatible adapter configuration.

Fair comparison protocol. All methods use the same backbone model, task split, decoding configuration, action interface, and maximum step budget within each environment. Memoryaugmented baselines are given access to the same memory-content pool when applicable. Cache-level baselines are evaluated on the same sequence of memory contexts and memory-content updates as HEMER. Thus, the comparison isolates how each method processes the memory context and reuses or recomputes KV states.

## B.3 Prompt Templates

We document the prompt templates used in our experiments. The system prompt, task brief, action grammar, and observation-action format are shared across methods within each benchmark. Methods differ only in whether and how they use the provided memory contents, and in how their serving policy reuses or recomputes KV states. This ensures that performance differences are not caused by benchmark-specific prompt engineering. Although the prompt template is shown in full, cache-reuse methods do not prefill the entire rendered prompt at every decision step. TTFT includes KV cache updates, required prefill, and local recomputation up to the first generated token, excluding memory retrieval, prompt construction, and schema-diff planning.

Shared System Prompt. The same system prompt is used by ReAct, Reflexion, MemRL, CacheBlend, KEEP, LRAgent, and HEMER.

Interact with a household to solve a task. Imagine you are an   
intelligent agent in a household environment and your target is   
to perform actions to complete the task goal. At the beginning of   
your interactions, you will be given the detailed description of the   
current environment and your goal to accomplish.   
For each turn, you will be given the observation from the last turn.   
You should first think about the current condition and plan your   
future actions, and then output your action for this turn. Your   
output must strictly follow this format:   
“Thought: your thoughts.\nAction: your next action”.   
The available actions are:   
1. go to {recep}   
2. take {obj} from {recep}   
3. move {obj} to {recep}   
4. open {recep}   
5. close {recep}   
6. use {obj}   
7. clean {obj} with {recep}   
8. heat {obj} with {recep}   
9. cool {obj} with {recep}   
where {obj} and {recep} correspond to objects and receptacles.   
After each turn, the environment will provide feedback. If the   
environment outputs “Nothing happened”, the previous action is   
invalid and you should try another valid action.   
Your response should use the following format:   
Thought: <your thoughts>   
Action: <your next action>

The same Thought-Action output format is used for all methods, including cache-reuse baselines, so that differences are not caused by response-format prompting.

Memory Context Block. For memory-augmented methods, retrieved memory contents are rendered into a typed memory context block. For HEMER, this block corresponds to the memory context X used by the reconciliation policy. The same text-side memory context format is used across benchmarks, while visual observations are provided through the multimodal input interface when applicable.

[Memory context X<sub>t</sub>]

[Declarative memory]

\- <declarative memory content 1>

\- <declarative memory content 2>

[Procedural memory]

\- <procedural memory content>

[Working memory]

\- <working-memory summary 1>

\- <working-memory summary 2>

At each decision step, the block is rendered from the current memory contents and their schema fields. Memory-content updates, such as insertion, eviction, and reordering, appear to the foundation model through the updated memory context, while procedural adapter changes are tracked as schema metadata for KV compatibility. For cache-level baselines, we use the same rendered memory context when applicable, but each method applies its own cache reuse policy. For HEMER, the rendered chunks are additionally aligned with KV ranges through the memory schema, enabling schema-guided KV reconciliation. Adapter identifiers are stored as schema metadata for compatibility checking and are not used as additional natural-language hints unless required by a baseline interface.

Task Brief. The first user turn specifies the task type, goal, and initial admissible commands. The format is shared across methods:

You are solving a {task\_type} task.   
Goal: {task}   
Initial admissible commands: {cmd\_1}, {cmd\_2}, ..., {cmd\_K}

We use the same admissible-command formatting across methods. When the admissible-command list is long, it is truncated using the same limit for all methods. Subsequent observation turns include the dynamically updated admissible-command list.

Observation-Action History. After the task brief, each step appends a user turn of the form:

Observation: {obs}

The assistant response follows the shared ReAct-style format:

Thought: <reasoning about the current state> Action: <next action>

The conversation history persists within an episode so that each method has access to previous observations and actions. Long observations are truncated using the same character budget across all methods. The memory context block may change across steps as memory contents are updated, but past assistant turns are not rewritten.

Retrieved Trajectory Memory. For methods that use episodic or trajectory memory, we optionally include a compact retrieved trajectory from the same task type. This block is used only for methods whose baseline design includes retrieved experience, such as MemRL and memory-augmented variants of HEMER. The retrieved trajectory format is:

–- RETRIEVED TRAJECTORY MEMORY –-   
Task: {retrieved task description}   
Action sequence: {action\_1} > {action\_2} > ...

When this block is used, the same retrieval source and formatting are applied to all methods that support trajectory memory. This isolates the effect of memory reconciliation and KV reuse from differences in retrieved-memory quality.

Fairness note. Prompt templates are shared at the level of system instruction, action grammar, task brief, and observation-action history. When a method supports memory input, it receives the same rendered memory-content format; methods that do not use memory ignore this block by design. For cache-reuse comparisons, the same memory context sequence is used whenever the method can consume it, and differences arise from how KV states are reused or recomputed.

## B.4 Hyperparameters

Table 10: Hyperparameters used by HEMER across experiments.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Decoding temperature</td><td>0</td></tr><tr><td>Max new tokens</td><td>256</td></tr><tr><td>Maximum ALFWorld steps</td><td>30</td></tr><tr><td>Maximum Habitat steps</td><td>50</td></tr><tr><td>LoRA rank</td><td>8</td></tr></table>

Reproducibility notes. Local recomputation is included in the reported latency and TTFT. Active action ratio and action diversity are computed from generated action sequences, as described in Appendix A.6.

## C Additional experimental result

## C.1 Main table

Tables 11, 12, and 13 provide the full per-environment results used in the main evaluation. We provide additional analysis for each environment below.

ALFWorld results. Table 11 reports the full ALFWorld results. HEMER achieves the highest SR at 71.43%, outperforming the strongest baseline MemRL by 18.57 percentage points. It also reduces latency from 0.90 seconds per step for MemRL to 0.18 seconds per step. This indicates that the improvement is not only due to better memory use, but also to lower inference cost under changing memory contexts. ReAct and Reflexion perform worse because they rely mainly on prompting or self-reflection without explicit structured reuse of memory contents. MemRL improves over these prompting baselines by retrieving episodic information from previous trajectories, but it still processes the retrieved memory context as a conventional input. As a result, it does not reduce the cost of repeatedly processing updated memory contents.

The cache-reuse baselines reduce latency compared with memory-only baselines, but their task success remains lower than HEMER. CacheBlend and KEEP achieve lower latency than MemRL, but their reuse ratios remain around 50%, substantially below the 75.30% reuse achieved by HEMER. LRAgent also reuses part of the computation through procedural adapter switching, but its lower SR and diversity suggest that procedural reuse alone is insufficient when declarative and working-memory contents change across steps. By contrast, HEMER achieves both the highest reuse ratio and the highest action behavior metrics. This shows that schema-indexed memory chunks help preserve compatible KV states while still allowing the agent to use a broader and more task-relevant set of actions. Overall, the ALFWorld results demonstrate that HEMER is especially effective in text-only environments where admissible actions, object locations, and working-memory summaries change frequently across decision steps.

Table 11: Main results on ALFWorld.
<table><tr><td>Method</td><td>SR (%)</td><td>Latency (s/step)</td><td></td><td></td><td>Reuse ratio(%) Active ratio (%) Diversity ratio(%)</td><td>TTFT (ms)</td></tr><tr><td>HEMER</td><td> $7 1 . 4 3 \pm 2 . 0 5$ </td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td> $7 5 . 3 0 \pm 4 . 3 3$ </td><td> $6 7 . 6 9 \pm 2 . 2 7$ </td><td> $7 1 . 4 3 \pm 3 . 3 2$ </td><td> $2 1 . 5 9 \pm 2 . 5 3$ </td></tr><tr><td>ReAct</td><td> $4 7 . 1 4 \pm 6 . 3 9$ </td><td> $0 . 8 8 \pm 0 . 0 4$ </td><td></td><td> $5 2 . 0 4 \pm 2 . 4 5$ </td><td> $6 4 . 9 5 \pm 8 . 9 4$ </td><td> $2 3 . 3 2 \pm 3 . 2 4$ </td></tr><tr><td>Reflexion</td><td> $4 4 . 2 9 \pm 5 . 7 4$ </td><td> $0 . 9 8 \pm 0 . 0 5$ </td><td></td><td> $4 5 . 5 7 \pm 4 . 3 0$ </td><td> $6 4 . 3 8 \pm 6 . 1 2$ </td><td> $2 3 . 1 2 \pm 2 . 8 9$ </td></tr><tr><td>MemRL</td><td> $5 2 . 8 6 \pm 4 . 8 3$ </td><td> $0 . 9 0 \pm 0 . 0 4$ </td><td></td><td> $4 6 . 2 6 \pm 5 . 1 6$ </td><td> $6 5 . 0 5 \pm 4 . 4 8$ </td><td> $2 2 . 8 8 \pm 2 . 7 1$ </td></tr><tr><td>CacheBlend</td><td> $4 8 . 3 9 \pm 4 . 2 9$ </td><td> $0 . 4 8 \pm 0 . 0 4$ </td><td> $5 0 . 3 0 \pm 2 . 3 3$ </td><td> $4 6 . 0 1 \pm 3 . 6 7$ </td><td> $5 8 . 4 9 \pm 2 . 9 2$ </td><td> $2 2 . 3 5 \pm 3 . 2 2$ </td></tr><tr><td>KEEP</td><td> $5 1 . 8 1 \pm 3 . 9 2$ </td><td> $0 . 4 6 \pm 0 . 0 6$ </td><td> $4 8 . 3 0 \pm 2 . 3 3 $ </td><td> $5 0 . 4 0 \pm 4 . 9 2$ </td><td> $6 5 . 0 1 \pm 4 . 9 5$ </td><td> $2 2 . 3 9 \pm 3 . 2 4$ </td></tr><tr><td>LRAgent</td><td> $4 6 . 4 3 \pm 3 . 1 9$ </td><td> $0 . 5 2 \pm 0 . 0 3$ </td><td> $4 8 . 5 0 \pm 3 . 3 3$ </td><td> $5 1 . 1 7 \pm 2 . 3 1$ </td><td> $4 9 . 9 8 \pm 3 . 0 6$ </td><td> $2 2 . 3 8 \pm 1 . 9 9$ </td></tr></table>

Habitat results. Table 12 reports the full results on EmbodiedBench Habitat. HEMER achieves 76.27% SR, improving over MemRL by 14.47 percentage points and over the best cache baseline KEEP by 16.87 percentage points. It also obtains the lowest latency at 1.82 seconds per step, compared with 2.68 seconds for MemRL and 1.97–2.09 seconds for the cache-reuse baselines. This is important because Habitat introduces RGB observations and high-level skill transitions, making the memory context depend not only on text history but also on visual grounding and changing object states. The result therefore suggests that HEMER is not limited to text-only memory updates, but can also support multimodal embodied inference where visual observations update the relevant memory contents.

The behavior metrics further show that HEMER maintains stronger action-level performance. It achieves the highest active action ratio at 71.38% and the highest action diversity at 56.29%. MemRL obtains relatively high diversity, but its latency remains high because it does not reuse KV states. CacheBlend, KEEP, and LRAgent reduce inference time relative to MemRL, but their lower SR and reuse ratios show that their reuse mechanisms are less robust to multimodal memory-content changes. CacheBlend relies on reusable non-prefix segments, KEEP invalidates affected regions, and LRAgent focuses on procedural adapter switching. These mechanisms do not fully address the case where visual facts, working-memory summaries, and procedural skill choices change together. By contrast, HEMER maintains a higher reuse ratio by matching schema-indexed chunks and locally updating affected KV ranges. This explains why it improves both task success and inference efficiency on Habitat.

Table 12: Main results on Habitat.
<table><tr><td>Method</td><td>SR (%)</td><td>Latency (s/step)</td><td></td><td>Reuse ratio(%) Active ratio (%)</td><td>Diversity ratio(%)</td><td>TTFT (ms)</td></tr><tr><td>HEMER</td><td> $7 6 . 2 7 \pm 3 . 3 2$ </td><td> $1 . 8 2 \pm 0 . 0 3$ </td><td> $7 2 . 8 1 \pm 4 . 2 4$ </td><td> $7 1 . 3 8 \pm 3 . 2 9$ </td><td> $5 6 . 2 9 \pm 2 . 8 2$ </td><td> $3 8 . 2 9 \pm 3 . 2 9$ </td></tr><tr><td>ReAct</td><td> $5 2 . 0 0 \pm 3 . 3 2$ </td><td> $2 . 4 1 \pm 0 . 0 7$ </td><td></td><td> $5 1 . 8 1 \pm 6 . 3 3$ </td><td> $4 2 . 0 3 \pm 2 . 1 2$ </td><td> $4 4 . 3 0 \pm 5 . 7 8$ </td></tr><tr><td>Reflexion</td><td> $5 0 . 4 7 \pm 5 . 4 0$ </td><td> $2 . 3 8 \pm 0 . 1 8$ </td><td></td><td> $4 9 . 6 6 \pm 8 . 7 1$ </td><td> $3 9 . 2 7 \pm 3 . 0 4$ </td><td> $4 4 . 3 4 \pm 4 . 9 0$ </td></tr><tr><td>MemRL</td><td> $6 1 . 8 0 \pm 2 . 9 2$ </td><td> $2 . 6 8 \pm 0 . 1 2$ </td><td></td><td> $5 7 . 5 5 \pm 4 . 7 8$ </td><td> $5 4 . 3 6 \pm 1 . 3 1$ </td><td> $4 3 . 8 9 \pm 3 . 2 9$ </td></tr><tr><td>CacheBlend</td><td> $5 6 . 3 3 \pm 2 . 2 4$ </td><td> $1 . 9 7 \pm 0 . 0 3$ </td><td> $4 6 . 2 5 \pm 3 . 2 9$ </td><td> $5 1 . 3 9 \pm 3 . 2 9$ </td><td> $5 3 . 2 0 \pm 2 . 3 0$ </td><td> $3 9 . 8 2 \pm 4 . 8 3$ </td></tr><tr><td>KEEP</td><td> $5 9 . 4 0 \pm 2 . 2 4$ </td><td> $2 . 0 9 \pm 0 . 0 2$ </td><td> $4 3 . 0 5 \pm 4 . 1 2$ </td><td> $6 4 . 5 8 \pm 1 . 7 5$ </td><td> $5 2 . 4 0 \pm 1 . 2 0$ </td><td> $4 0 . 7 1 \pm 6 . 0 8$ </td></tr><tr><td>LRAgent</td><td> $5 7 . 4 7 \pm 3 . 5 4$ </td><td> $2 . 0 3 \pm 0 . 3 1$ </td><td> $4 2 . 9 6 \pm 3 . 9 8$ </td><td> $5 8 . 3 4 \pm 4 . 5 3$ </td><td> $4 4 . 2 8 \pm 3 . 3 9$ </td><td> $4 0 . 5 7 \pm 5 . 0 6$ </td></tr></table>

RealWorld-Tabletop results. Table 13 reports results on the real-world tabletop manipulation setting. HEMER achieves the highest SR at 83.20%, outperforming MemRL by 18.92 percentage points. It also reduces latency from 2.94 to 1.87 seconds per step, showing that the efficiency benefit remains under real sensor observations and physical manipulation. Unlike simulation, the realworld setting introduces perception noise, object pose changes, execution uncertainty, and task-state changes caused by robot actions. These factors can change which memory contents are relevant at each decision step. The strong result of HEMER indicates that memory-aware KV reconciliation remains useful when memory-content updates arise from physical interaction rather than only from simulated state transitions.

The baseline trends show a similar pattern to the simulation results. MemRL improves task success over some cache-only baselines by using retrieved memory, but it has the highest latency because it lacks structured KV reuse. CacheBlend and LRAgent reduce latency relative to MemRL, but their lower SR suggests that preserving some cached computation is not enough when the action context changes after perception and manipulation updates. KEEP achieves a stronger SR than CacheBlend and LRAgent, but it has lower reuse and higher latency than HEMER, reflecting the cost of broader invalidation. By contrast, HEMER achieves the highest reuse ratio at 72.18%, the highest active action ratio at 75.45%, and the highest action diversity at 64.85%. This indicates that HEMER preserves compatible KV states while continuing to generate varied and task-progressing actions. Overall, the real-world results support the claim that heterogeneous-memory KV reconciliation improves both robustness and efficiency in embodied tasks with dynamic observations and physical state changes.

Table 13: Main results on Real-world.
<table><tr><td>Method</td><td>SR (%)</td><td>Latency (s/step)</td><td></td><td></td><td>Reuse ratio(%) Active ratio (%) Diversity ratio(%)</td><td>TTFT (ms)</td></tr><tr><td>HEMER</td><td> $8 3 . 2 0 \pm 4 . 3 9$ </td><td> $1 . 8 7 \pm 0 . 1 2$ </td><td> $7 2 . 1 8 \pm 3 . 2 8$ </td><td> $7 5 . 4 5 \pm 3 . 9 2$ </td><td> $6 4 . 8 5 \pm 2 . 2 4$ </td><td> $5 7 . 2 9 \pm 5 . 3 9$ </td></tr><tr><td>MemRL</td><td> $6 4 . 2 8 \pm 5 . 2 5$ </td><td> $2 . 9 4 \pm 0 . 3 2$ </td><td></td><td> $6 1 . 0 4 \pm 4 . 7 4$ </td><td> $5 8 . 1 9 \pm 2 . 7 3$ </td><td> $8 3 . 2 0 \pm 7 . 1 8$ </td></tr><tr><td>CacheBlend</td><td> $5 1 . 2 1 \pm 4 . 0 2$ </td><td> $2 . 1 4 \pm 0 . 1 2$ </td><td> $5 4 . 0 2 \pm 4 . 3 0$ </td><td> $5 4 . 2 7 \pm 3 . 8 2$ </td><td> $5 4 . 3 7 \pm 4 . 1 9$ </td><td> $5 9 . 2 9 \pm 4 . 8 7$ </td></tr><tr><td>KEEP</td><td> $6 3 . 4 0 \pm 3 . 2 7$ </td><td> $2 . 4 3 \pm 0 . 2 4$ </td><td> $4 8 . 6 4 \pm 4 . 2 4$ </td><td> $6 2 . 0 3 \pm 3 . 6 8$ </td><td> $5 6 . 2 0 \pm 3 . 2 8$ </td><td> $6 3 . 9 3 \pm 5 . 9 2$ </td></tr><tr><td>LRAgent</td><td> $5 3 . 4 3 \pm 4 . 1 8$ </td><td> $2 . 2 1 \pm 0 . 1 8$ </td><td> $5 0 . 2 2 \pm 3 . 8 2$ </td><td> $5 6 . 3 0 \pm 2 . 3 9$ </td><td> $4 9 . 9 7 \pm 1 . 2 9$ </td><td> $6 3 . 2 1 \pm 7 . 3 8$ </td></tr></table>

## C.2 Analysis

Cost-analysis raw data. The TTFT values are measured after the step input is finalized and reusable KV ranges are available, so they reflect the incremental cache-processing and first-token cost rather than full prompt construction or environment execution time. Tables 14 and 15 provide the raw numerical values used for the cost-efficiency analysis in Figure 3. Table 14 reports TTFT as the memory length increases from 5 to 50 chunks. Full Rebuild shows a nearly monotonic increase in TTFT, reaching 148.59 ms at 50 chunks, because the entire memory context must be prefilled at each decision step. CacheBlend, KEEP, and LRAgent reduce this cost by reusing part of the cached states, but their TTFT still grows substantially as the memory context becomes longer. In contrast, HEMER maintains the lowest TTFT across all memory lengths, reaching 73.49 ms at 50 chunks. This indicates that chunk-level reconciliation reduces the amount of memory context that must be reprocessed as the memory length grows.

Table 15 reports KV reuse ratio as the invalidated memory ratio $\rho$ increases. Full Rebuild has zero reuse by construction. CacheBlend, KEEP, and LRAgent preserve some cached states when $\rho$ is small, but their reuse ratios decrease rapidly as more memory regions are invalidated. For example, at $\rho = 0 . 5$ , their reuse ratios drop to 0.184, 0.152, and 0.134, respectively. By comparison, HEMER maintains a reuse ratio of 0.453 at the same invalidation level. This shows that HEMER can preserve compatible KV states even when a substantial fraction of the memory context is perturbed. The gap comes from grounding cache reuse in schema-indexed memory chunks rather than relying only on prefix stability, segment boundaries, or adapter-level reuse.

Together, these results show that HEMER improves inference efficiency from two complementary directions. First, it lowers TTFT under long memory contexts by avoiding full cache reconstruction. Second, it maintains higher KV reuse under memory perturbation by locally updating only affected KV ranges. These trends support the main claim that memory-aware KV reconciliation can reduce redundant inference computation while preserving reusable memory contents across decision steps.

Table 14: Analysis of TTFT (ms) under increasing memory length. Column headings indicate the number of memory chunks. Values are mean ± std.
<table><tr><td>Method</td><td>5</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td></tr><tr><td>Full Rebuild</td><td> $2 2 . 8 8 \pm 1 . 6 8$ </td><td> $3 9 . 9 9 \pm 1 . 8 7$ </td><td> $5 5 . 5 4 \pm 4 . 3 0$ </td><td> $9 0 . 2 0 \pm 4 . 3 9$ </td><td> $1 1 2 . 4 6 \pm 7 . 2 7$ </td><td> $1 4 8 . 5 9 \pm 8 . 2 0$ </td></tr><tr><td>CacheBlend</td><td> $2 2 . 8 0 \pm 3 . 2 3$ </td><td> $2 3 . 9 2 \pm 1 . 5 8$ </td><td> $4 3 . 3 9 \pm 4 . 0 1$ </td><td> $6 5 . 9 3 \pm 4 . 2 9$ </td><td> $7 8 . 2 3 \pm 8 . 5 4$ </td><td> $9 2 . 1 5 \pm 8 . 3 0$ </td></tr><tr><td>KEEP</td><td> $2 2 . 3 5 \pm 1 . 9 9$ </td><td> $2 3 . 8 9 \pm 1 . 9 8$ </td><td> $4 3 . 3 8 \pm 8 . 0 7$ </td><td> $6 6 . 8 4 \pm 6 . 1 9$ </td><td> $7 8 . 3 5 \pm 7 . 4 2$ </td><td> $9 0 . 8 2 \pm 8 . 1 1$ </td></tr><tr><td>LRAgent</td><td> $2 2 . 3 9 \pm 3 . 2 4$ </td><td> $2 6 . 3 9 \pm 2 . 9 5$ </td><td> $4 6 . 3 0 \pm 3 . 2 7$ </td><td> $6 8 . 3 0 \pm 4 . 4 7$ </td><td> $8 2 . 2 4 \pm 5 . 2 8$ </td><td> $1 0 2 . 3 9 \pm 8 . 4 2$ </td></tr><tr><td>HEMER</td><td> ${ \bf 2 1 . 6 0 \pm 2 . 5 3 }$ </td><td> ${ \bf 2 1 . 9 1 \pm 1 . 8 3 }$ </td><td> ${ \bf 2 7 . 3 7 \pm 5 . 4 4 }$ </td><td> ${ \bf 4 3 . 4 9 \pm 3 . 2 9 }$ </td><td> ${ \bf 5 8 . 5 3 \pm 4 . 1 1 }$ </td><td> ${ \bf 7 3 . 4 9 \pm 5 . 3 4 }$ </td></tr></table>

Hidden-State Drift Analysis. We further analyze representation-level robustness under memorycontent updates. While task success measures whether the final action trajectory satisfies the goal, it does not directly show whether KV reuse destabilizes internal model representations. To inspect this effect, we compare each cache-reuse method against Full Rebuild, which recomputes the full KV cache after every memory-content update. For each method, we measure hidden-state drift at each layer and token position:

$$
\begin{array} { r } { \mathrm { D r i f t } _ { \ell , i } = \left. h _ { \ell , i } ^ { \mathrm { m e t h o d } } - h _ { \ell , i } ^ { \mathrm { f u l l } } \right. _ { 2 } , } \end{array}\tag{8}
$$

Table 15: Analysis on KV reuse ratio under increasing invalidated memory ratio $\rho .$ Values are mean ± std.
<table><tr><td>Method</td><td> $\mathbf { \rho } _ { \rho } = \mathbf { 0 . 0 }$ </td><td> $\mathbf { \rho } _ { \rho } = \mathbf { 0 . 1 }$ </td><td> $\mathbf { \rho } \mathbf { \rho } \rho = \mathbf { 0 . 2 }$ </td><td> ${ \boldsymbol \rho } = { \bf 0 . 3 }$ </td><td> $\rho = \mathbf { 0 . 4 }$ </td><td> ${ \boldsymbol \rho } = { \bf 0 . 5 }$ </td></tr><tr><td>Full Rebuild</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>CacheBlend</td><td> $0 . 5 0 3 \pm 0 . 0 2 3$ </td><td> $0 . 3 8 7 \pm 0 . 0 3 2$ </td><td> $0 . 3 8 5 \pm 0 . 0 4 0$ </td><td> $0 . 3 2 4 \pm 0 . 0 3 4$ </td><td> $0 . 3 1 3 \pm 0 . 0 8 3$ </td><td> $0 . 1 8 4 \pm 0 . 0 5 3$ </td></tr><tr><td>KEEP</td><td> $0 . 4 8 3 \pm 0 . 0 2 3$ </td><td> $0 . 4 2 7 \pm 0 . 0 3 2$ </td><td> $0 . 3 8 5 \pm 0 . 0 1 0$ </td><td> $0 . 3 2 4 \pm 0 . 0 3 4$ </td><td> $0 . 2 0 1 \pm 0 . 0 3 3$ </td><td> $0 . 1 5 2 \pm 0 . 0 9 4$ </td></tr><tr><td>LRAgent</td><td> $0 . 4 8 5 \pm 0 . 0 3 3$ </td><td> $0 . 4 6 3 \pm 0 . 0 2 3$ </td><td> $0 . 3 2 3 \pm 0 . 0 8 1$ </td><td> $0 . 2 9 2 \pm 0 . 0 8 1$ </td><td> $0 . 2 8 2 \pm 0 . 0 8 3$ </td><td> $0 . 1 3 4 \pm 0 . 0 4 2$ </td></tr><tr><td>HEMER</td><td> $\mathbf { 0 . 7 5 3 \pm 0 . 0 4 3 }$ </td><td> $\mathbf { 0 . 7 0 9 \pm 0 . 0 4 6 }$ </td><td> $\mathbf { 0 . 6 5 2 \pm 0 . 0 5 7 }$ </td><td> $\mathbf { 0 . 6 0 5 \pm 0 . 0 5 7 }$ </td><td> $\mathbf { 0 . 5 0 4 \pm 0 . 0 4 9 }$ </td><td> $\mathbf { 0 . 4 5 3 \pm 0 . 0 2 1 }$ </td></tr></table>

![](images/5def985bd2dadfd7cf27ffb15e6660d5c96817e16acaa16ae5eb4b9caf707db0.jpg)  
Figure 9: Analysis on Hidden-state drift from Full Rebuild shows that HEMER localizes drift around affected chunks, while baselines show broader boundary-level or downstream drift.

where $h _ { \ell , i } ^ { \mathrm { m e t h o d } }$ is the hidden state at layer ℓ and token position i after applying the method’s cache update rule, and $h _ { \ell , i } ^ { \mathrm { f u l l } }$ is the corresponding hidden state from Full Rebuild. The heatmap in Figure 9 visualizes this drift across layers and token positions. Brighter regions indicate larger representation mismatch from Full Rebuild.

The drift patterns reveal how each method propagates memory-content updates through the cached representation. CacheBlend produces drift around reusable segment boundaries. This suggests that segment-level reuse is sensitive to boundary mismatch: even when a non-prefix segment is reused, local changes near the segment boundary can perturb nearby reused states. KEEP shows broader downstream propagation. Because KEEP invalidates affected regions and preserves later cache regions under a coarser update rule, representation mismatch can propagate to later token positions after a memory-content update. LRAgent shows gradual drift accumulation over later positions. This pattern indicates that adapter-level reuse alone does not fully account for updates to declarative or working-memory contents, so representation mismatch accumulates when the memory context changes across decision steps.

In contrast, HEMER confines high-drift regions to a small number of affected memory chunks. This pattern follows from the schema-indexed reconciliation rule. A KV range is reused only when the corresponding memory content remains compatible under the memory schema, including its normalized content, dependency annotation, and adapter identifier. When the update is limited to a position shift, RoPE remapping corrects the positional phase of cached key states. When the memory content, dependency annotation, or adapter identifier changes, HEMER locally recomputes the affected KV range rather than reusing stale states. In the evaluated settings, representation deviations remain comparatively small outside the affected memory chunks.

This analysis supports the robustness claim of HEMER. The benefit is not only that more KV states are reused, but that reused states remain localized and compatible with the current memory context. Thus, HEMER reduces redundant recomputation while limiting broad hidden-state drift under the evaluated memory-content updates.

Deviation Accumulation over Successive Memory-Content Updates. While the hidden-state drift analysis above measures deviation after individual memory-content updates, we further examine whether the deviation accumulates over successive updates. Table 16 reports the hidden-state cosine similarity to Full Rebuild over 30 successive memory-content updates whose operation distribution follows the ALFWorld episodes. Naive reuse preserves all cached states without reconciliation, so its deviation accumulates with no mechanism to reset it, ending at 0.949; its transient rise at step 20 comes from evictions that remove previously changed content together with its accumulated deviation, not from any correction of cached states. CacheBlend recomputes trailing chunks with a boundary period of 5 updates, so its values fluctuate with the distance to the most recent recomputation. HEMER reaches its lowest value in the middle of the horizon and recovers to 0.974, because every content change and adapter switch triggers recomputation that refreshes the affected KV ranges and limits accumulated deviation.

Table 16: Analysis on Deviation accumulation over successive memory-content updates. Values are hidden-state cosine similarity to Full Rebuild, mean ± std over five update sequences.
<table><tr><td>Method</td><td>Step 1</td><td>Step 5</td><td>Step 10</td><td>Step 20</td><td>Step 30</td></tr><tr><td>Naive reuse</td><td> $0 . 9 9 3 \pm 0 . 0 0 7$ </td><td> $0 . 9 7 4 \pm 0 . 0 1 2$ </td><td> $0 . 9 4 8 \pm 0 . 0 1 2$ </td><td> $0 . 9 7 1 \pm 0 . 0 1 5$ </td><td> $0 . 9 4 9 \pm 0 . 0 1 6$ </td></tr><tr><td>CacheBlend</td><td> $0 . 9 9 3 \pm 0 . 0 0 7$ </td><td> $0 . 9 7 4 \pm 0 . 0 1 2$ </td><td> $0 . 9 4 7 \pm 0 . 0 1 2$ </td><td> $0 . 9 8 0 \pm 0 . 0 0 4$ </td><td> $0 . 9 7 2 \pm 0 . 0 0 9$ </td></tr><tr><td>HEMER</td><td> $0 . 9 9 3 \pm 0 . 0 0 7$ </td><td> $0 . 9 8 9 \pm 0 . 0 0 6$ </td><td> $0 . 9 6 9 \pm 0 . 0 0 8$ </td><td> $0 . 9 7 9 \pm 0 . 0 0 3$ </td><td> $0 . 9 7 4 \pm 0 . 0 0 7$ </td></tr></table>

Robustness to distractor action candidates. The results show that HEMER remains stable under moderate distractor levels. From 0% to 100% distractor noise, SR decreases only slightly, from 71.43% to 68.70%, while latency increases from 0.18 to 0.42 seconds per step. Although the reuse ratio decreases as more irrelevant candidates are injected, HEMER still preserves more than half of the cached states at 100% noise and keeps TTFT within a moderate range. This suggests that the framework can maintain task-relevant decision behavior under noisy action contexts while limiting unnecessary cache reconstruction. At higher distractor levels, SR drops sharply and TTFT increases, indicating that excessive irrelevant candidates eventually dominate action selection and reduce cache reuse.

Table 17: Analysis on Robustness under distractor action candidates. We inject irrelevant admissibleaction candidates into the decision context and measure task success, action behavior, inference time, KV reuse, and TTFT. HEMER remains stable under moderate distractor ratios, while extreme distractor levels degrade action selection and reduce reusable cache states.
<table><tr><td>Noise</td><td>SR (%)</td><td>Latency (s/step)</td><td>Reuse ratio(%)</td><td></td><td>Active ratio(%) Diversity ratio(%)</td><td>TTFT (ms)</td></tr><tr><td>0%</td><td> ${ \bf 7 1 . 4 3 \pm 2 . 0 5 }$ </td><td> ${ \bf 0 . 1 8 \pm 0 . 0 2 }$ </td><td> ${ \bf 7 5 . 3 0 \pm 4 . 3 3 }$ </td><td> ${ \bf 6 7 . 6 9 \pm 2 . 2 7 }$ </td><td> ${ \bf 7 1 . 4 3 \pm 3 . 3 2 }$ </td><td> ${ \bf 2 1 . 5 9 \pm 2 . 5 3 }$ </td></tr><tr><td>30%</td><td> $7 1 . 0 6 \pm 2 . 4 3$ </td><td> $0 . 2 1 \pm 0 . 1 2$ </td><td> $7 0 . 2 3 \pm 1 . 3 1$ </td><td> $7 6 . 0 9 \pm 1 . 9 1$ </td><td> $8 2 . 2 9 \pm 2 . 0 1$ </td><td> $2 5 . 9 4 \pm 3 . 2 4$ </td></tr><tr><td>60%</td><td> $7 0 . 7 3 \pm 3 . 8 9$ </td><td> $0 . 2 3 \pm 0 . 0 8$ </td><td> $6 3 . 7 2 \pm 2 . 0 4$ </td><td> $6 4 . 6 0 \pm 0 . 7 1$ </td><td> $7 7 . 4 2 \pm 1 . 2 1$ </td><td> $3 0 . 8 9 \pm 4 . 4 4$ </td></tr><tr><td>100%</td><td> $6 8 . 7 0 \pm 2 . 8 6$ </td><td> $0 . 4 2 \pm 0 . 0 2$ </td><td> $5 5 . 4 4 \pm 4 . 1 2$ </td><td> $4 5 . 8 3 \pm 2 . 3 1$ </td><td> $6 9 . 5 8 \pm 1 . 4 3$ </td><td> $4 8 . 0 2 \pm 5 . 1 0$ </td></tr><tr><td>150%</td><td> $5 3 . 2 9 \pm 5 . 5 5$ </td><td> $0 . 4 8 \pm 0 . 1 0$ </td><td> $4 2 . 4 2 \pm 2 . 6 0$ </td><td> $3 5 . 9 8 \pm 2 . 3 9$ </td><td> $5 3 . 4 2 \pm 2 . 2 4$ </td><td> $5 2 . 0 8 \pm 5 . 4 0$ </td></tr><tr><td>200%</td><td> $3 1 . 0 1 \pm 2 . 0 6$ </td><td> $0 . 6 2 \pm 0 . 2 4$ </td><td> $3 1 . 2 9 \pm 8 . 2 0$ </td><td> $3 4 . 6 9 \pm 2 . 0 7$ </td><td> $4 3 . 9 2 \pm 1 . 2 0$ </td><td> $6 5 . 1 9 \pm 3 . 2 9$ </td></tr><tr><td>300%</td><td> $1 5 . 3 0 \pm 1 . 2 8$ </td><td> $0 . 9 3 \pm 0 . 1 1$ </td><td> $2 5 . 6 4 \pm 6 . 6 4$ </td><td> $3 3 . 1 3 \pm 1 . 9 2$ </td><td> $3 3 . 2 3 \pm 1 . 4 2$ </td><td> $8 3 . 2 4 \pm 5 . 4 1$ </td></tr></table>

Approximation quality of non-prefix KV reuse. Since reused KV states retain the influence of their previous preceding context, chunk-level KV reconciliation approximates full recomputation (Sec. 4.2). Table 18 compares it with Full Rebuild across memory-content update patterns using five memory contexts composed as in the ALFWorld experiments.

K cos measures reused-key cosine similarity, and hidden cos measures last-layer hidden-state cosine similarity over a fixed action-decision prompt appended after the memory context. KL measures the divergence between next-token distributions at the prompt’s final position. We report two complementary agreement metrics against Full Rebuild. Top-1 agreement measures the fraction of evaluation cases in which the highest-probability next token is identical between the two methods. Top-5 overlap measures the average fraction of shared tokens between their top-5 next-token sets, computed as the intersection size divided by five. Unlike hit@5, Top-5 overlap measures agreement between candidate sets rather than the inclusion of a reference token, and can therefore be lower than Top-1 agreement. Content changes and adapter switches trigger recomputation of affected KV ranges, yielding KL below 0.01. Insertion, eviction, and reordering reuse downstream ranges despite changes to their preceding context, yet hidden-state cosine remains at least 0.984. Eviction produces the largest output deviation, while reordering produces the largest key-state deviation. Across 30 successive updates following the ALFWorld operation distribution, recomputation on content changes and adapter switches limits accumulated deviation, yielding a final hidden-state cosine of 0.974, compared with 0.949 for naive reuse (Appendix C.2). These results complement the hidden-state drift analysis above by quantifying deviation from Full Rebuild and its output-level effects across memory-content update patterns.

Table 18: Analysis on Deviation from Full Rebuild under memory-content update patterns. HEMER keeps reconciled KV states close to full recomputation by recomputing affected KV ranges.
<table><tr><td>Update pattern</td><td>K cos</td><td>Hidden cos</td><td>KL (nats)</td><td>Top-1 agreement</td><td>Top-5 overlap</td></tr><tr><td>Reordering</td><td> $0 . 9 7 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 3 6 \pm 0 . 0 0 8$ </td><td>0.80</td><td>0.96</td></tr><tr><td>Insertion before reused ranges</td><td> $0 . 9 8 7 \pm 0 . 0 0 1$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 1 \pm 0 . 0 1 3$ </td><td>1.00</td><td>0.92</td></tr><tr><td>Eviction before reused ranges</td><td> $0 . 9 8 6 \pm 0 . 0 0 1$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 1$ </td><td> $0 . 0 7 8 \pm 0 . 0 1 7$ </td><td>0.80</td><td>0.76</td></tr><tr><td>Content change</td><td> $0 . 9 9 7 \pm 0 . 0 0 1$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 3$ </td><td>0.80</td><td>1.00</td></tr><tr><td>Adapter switch</td><td> $0 . 9 9 9 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 8$ </td><td> $0 . 0 0 9 \pm 0 . 0 1 0$ </td><td>1.00</td><td>0.88</td></tr></table>

Memory scaling under increasing chunk counts. We further evaluate the scalability of HEMER as the number of memory chunks increases under the ALFWorld memory composition. Table 19 reports the relative step cost and GPU memory usage with 10% of memory chunks updated at each decision step. As the number of chunks increases from 8 to 128, the step cost of HEMER relative to Full Rebuild decreases from 0.330 to 0.206. At 128 chunks, HEMER requires 22.0 GB of peak GPU memory, compared with 46.8 GB for Full Rebuild.

We additionally measure the overhead of reconciliation planning and cache assembly as the memory context grows. Reconciliation planning increases from 37 to 310 microseconds, while cache assembly increases from 0.7 to 5.7 ms across the evaluated memory sizes. These results indicate that HEMER reduces redundant recomputation as the memory context grows, while maintaining limited overhead for planning and assembling reusable KV ranges.

Table 19: Analysis on memory scaling under increasing chunk counts. HEMER reduces step cost relative to Full Rebuild and maintains lower peak GPU memory usage by recomputing only affected KV ranges under a fixed memory-content update ratio.
<table><tr><td rowspan="2">Chunks (N)</td><td rowspan="2">Relative cost</td><td colspan="2">GPU peak (GB)</td></tr><tr><td>HEMER</td><td>Full Rebuild</td></tr><tr><td>8</td><td>0.330</td><td>7.1</td><td>7.6</td></tr><tr><td>32</td><td>0.255</td><td>8.7</td><td>11.3</td></tr><tr><td>128</td><td>0.206</td><td>22.0</td><td>46.8</td></tr></table>

Sensitivity to dependency density. We further analyze how dependencies among memory chunks affect KV reuse and recomputation cost. Table 20 reports the step cost as dependency density increases from 0% to 100%. Under schema-guided reconciliation, invalidation propagates from updated memory chunks along their declared dependencies. Therefore, denser dependencies expand the affected KV ranges requiring local recomputation.

As dependency density increases from 0% to 100%, the KV reuse ratio gradually decreases from 85.4% to 61.3%, while the step cost relative to Full Rebuild increases from 0.247 to 0.368. The intermediate densities of 25% and 50% exhibit reuse ratios of 81.8% and 73.1%, respectively, indicating a gradual reduction in KV reuse as dependencies become denser.

Adapter pool scaling and switching overhead. We further evaluate the memory footprint and switching overhead of procedural adapters. The adapter memory footprint increases approximately linearly with pool size, requiring about 9.18 MB per adapter and 587.2 MB for a pool of 64 adapters. The larger adapter pools are constructed by replicating the shared, bathrooms, bedrooms, and livingrooms adapters under distinct adapter identifiers. Therefore, this experiment measures systemlevel memory and switching overhead rather than semantic interference among distinct procedural skills.

Table 20: Analysis on KV reuse and step cost under increasing dependency density. Denser dependencies expand the affected KV ranges requiring local recomputation, while HEMER maintains lower step cost than Full Rebuild across the evaluated densities.
<table><tr><td></td><td>Density (%) Reuse ratio (%)</td><td>Relative cost</td><td>Step cost (ms)</td></tr><tr><td>0</td><td> $8 5 . 4 \pm 0 . 3$ </td><td> $0 . 2 4 7 \pm 0 . 0 1 6$ </td><td>18</td></tr><tr><td>25</td><td> $8 1 . 8 \pm 1 . 5$ </td><td> $0 . 2 7 4 \pm 0 . 0 1 7$ </td><td>21</td></tr><tr><td>50</td><td> $7 3 . 1 \pm 2 . 2$ </td><td> $0 . 3 0 4 \pm 0 . 0 1 4$ </td><td>23</td></tr><tr><td>100</td><td> $6 1 . 3 \pm 7 . 5$ </td><td> $0 . 3 6 8 \pm 0 . 0 6 2$ </td><td>27</td></tr></table>

We additionally evaluate adapter-stable KV reconciliation under forced adapter switching at every decision step. HEMER maintains a KV reuse ratio of 77.9%, whereas disabling adapter-stable KV reconciliation reduces reuse to zero and yields latency comparable to Full Rebuild. This result shows that adapter-stable KV reconciliation preserves compatible KV states across frequent procedural adapter switches, avoiding unnecessary recomputation of the memory context.

Per-step latency breakdown. We profile end-to-end step latency on ALFWorld, including memory retrieval and prompt construction, whereas benchmark inference latency is measured after the step input is finalized. Table 21 decomposes the measured step latency into memory retrieval and context construction, prompt construction, KV reconciliation, local recomputation, fresh-tail prefill, and decoding. Both HEMER and Full Rebuild use the same evaluation protocol, prompts, and decoding configuration.

The largest difference occurs in local recomputation, which decreases from 267.4 ms for Full Rebuild to 39.5 ms for HEMER. Memory retrieval, prompt construction, KV reconciliation, and freshtail prefill show closely matched costs between the two methods. The amortized decode cost is approximately 24.0 ms per generated token for both methods. Overall, the measured step latency decreases from 1,287.5 to 1,043.3 ms. In particular, the 227.9 ms reduction in local recomputation accounts for nearly all of the 228.2 ms reduction in TTFT. This shows that the efficiency gain primarily comes from avoiding redundant memory-context recomputation rather than changes to other inference components.

This instrumented profiling experiment is intended to attribute latency to individual inference components, rather than reproduce the benchmark-level TTFT averages. Therefore, its absolute latency values are not directly comparable with those reported in the main evaluation.

Table 21: Analysis on per-step latency across inference components. HEMER reduces local memorycontext recomputation while maintaining comparable retrieval, prompt construction, fresh-tail prefill, and decoding costs to Full Rebuild. Amortized decode cost is the decoding time after the first token divided by the total number of generated tokens.
<table><tr><td>Component</td><td>HEMER</td><td>Full Rebuild</td></tr><tr><td>Memory retrieval and context build (ms)</td><td>0.6</td><td>0.6</td></tr><tr><td>Prompt construction (ms)</td><td>2.3</td><td>2.4</td></tr><tr><td>KV reconciliation (ms)</td><td>0.6</td><td>0.6</td></tr><tr><td>Local recomputation (ms)</td><td>39.5</td><td>267.4</td></tr><tr><td>Fresh-tail prefill (ms)</td><td>131.4</td><td>131.7</td></tr><tr><td>Decode after the first token (ms)</td><td>868.9</td><td>884.8</td></tr><tr><td>Total latency (ms)</td><td>1,043.3</td><td>1,287.5</td></tr><tr><td>Generated tokens per step</td><td>36.2</td><td>36.9</td></tr><tr><td>Amortized decode cost (ms/token)</td><td>24.0</td><td>24.0</td></tr></table>

## C.3 Ablation

Table 22 provides a more detailed ablation of the main components of HEMER. The full framework achieves the best result across all metrics, with 71.43% SR, 0.18 seconds per step, 75.30% reuse, and 21.59 ms TTFT. This shows that the full framework improves both task-level robustness and serving-level efficiency.

Removing the memory schema mainly weakens the link between memory contents and KV ranges. SR drops from 71.43% to 62.67%, and reuse decreases from 75.30% to 51.39%. TTFT also increases from 21.59 ms to 30.39 ms. This indicates that typed chunk anchoring is necessary for deciding which memory contents remain reusable after memory-content updates. Without the schema, the system cannot reliably identify reusable chunks or map updated contents to affected KV ranges, leading to lower reuse and less effective action behavior. The decrease in active action ratio and action diversity ratio further suggests that the agent uses a narrower and less productive set of actions when memory-to-cache alignment becomes unreliable.

Removing chunk-level KV reconciliation produces the largest degradation. SR decreases to 59.40%, reuse drops to 42.75%, latency increases to 0.47 seconds per step, and TTFT rises to 45.10 ms. This variant still has memory chunks, but it no longer localizes insertions, evictions, and reorderings to affected KV ranges. As a result, the system must rely on broader recomputation or less stable reuse when the memory context changes. The large TTFT increase shows that chunk-level reconciliation is the main source of prefill-cost reduction, while the SR drop shows that localized cache updates also matter for decision quality.

Removing adapter-stable KV reconciliation yields a different failure mode. Reuse remains relatively high at 66.40%, but SR still drops to 64.85% and TTFT increases to 39.75 ms. This shows that a high reuse ratio alone is not sufficient. When procedural memory changes the active adapter, KV states computed under the previous adapter may no longer be compatible with the current procedural configuration. Thus, adapter-stable reconciliation improves the compatibility of reused KV states with the active procedural memory. The gap between this variant and the full framework supports the role of adapter-aware compatibility checks in maintaining stable action decisions.

Overall, the ablation results separate the roles of the three components. The memory schema identifies which contents can be reused, chunk-level KV reconciliation localizes memory-content updates to affected ranges, and adapter-stable KV reconciliation prevents incompatible reuse under procedural changes. Together, these components allow HEMER to reduce inference cost while preserving task success and action-level robustness.

Table 22: Detailed ablation on HEMER. ‘w/o.’ denotes the removal of a component.
<table><tr><td>Method</td><td>SR (%)</td><td>Latency (s/step)</td><td>Reuse ratio(%)</td><td>Active ratio(%)</td><td>Diversity ratio(%)</td><td>TTFT (ms)</td></tr><tr><td>HEMER</td><td>71.43 ± 2.05</td><td>0.18 ± 0.02</td><td>75.30 ± 4.33</td><td>67.69 ± 2.27</td><td>71.43 ± 3.32</td><td>21.59 ± 2.53</td></tr><tr><td>w/o Memory Schema</td><td>62.67 ± 3.49</td><td>0.38 ± 0.12</td><td>51.39 ± 5.84</td><td>57.20 ± 4.12</td><td>61.00 ± 5.03</td><td>30.39 ± 3.24</td></tr><tr><td>w/o Chunk-Level KV Reconciliation</td><td>59.40 ± 4.27</td><td>0.47 ± 0.05</td><td>42.75 ± 6.20</td><td>55.10 ± 4.75</td><td>59.25 ± 5.61</td><td>45.10 ± 1.21</td></tr><tr><td>w/o Adapter-Stable KV Reconciliation</td><td>64.85 ± 3.58</td><td>0.28 ± 0.03</td><td>66.40 ± 4.92</td><td>62.71 ± 3.66</td><td>65.15 ± 4.44</td><td>39.75 ± 2.22</td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and introduction are aligned with the proposed method and the reported experimental results.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses limitations and future directions of the proposed approach. Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not present formal theoretical results or proofs.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper provides implementation details, experimental settings, and supplementary materials to support reproducibility.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The source code and reproduction instructions are provided in the supplementary materials.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies the main experimental settings, datasets, models, hyperparameters, and evaluation protocols.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports mean results with standard deviations over multiple runs where applicable.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper reports the main software and hardware configuration used in the experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conforms to the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: This work is primarily methodological and is evaluated in simulated benchmarks and controlled robot environments. We do not anticipate direct societal deployment or direct negative societal impacts from the work as presented. Any real-world deployment of embodied agents based on this approach would require appropriate safety validation and human oversight.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release high-risk models, datasets, or assets requiring special safeguards.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the existing datasets, models, benchmarks, software packages and their licenses used in the experiments.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not release new datasets or models, but the supplementary materials document the code and experimental scripts.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve human subjects, so IRB approval is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: The paper uses foundation models as embodied-agent backbones and describes the corresponding model, decoding, prompt, and adapter settings in the appendix.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.