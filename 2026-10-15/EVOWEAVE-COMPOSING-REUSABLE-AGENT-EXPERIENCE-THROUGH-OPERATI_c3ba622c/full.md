# EVOWEAVE: COMPOSING REUSABLE AGENT EXPERIENCE THROUGH OPERATION LEVEL GRAPHS

Guanhua Zhu<sup>1</sup>, Bo Shen<sup>2</sup>, Lei Zhang<sup>1</sup>, Fadi Deek<sup>1</sup>, Zhi Wei<sup>3</sup>, Mengnan Du<sup>4,∗</sup>, Chenxi Yuan<sup>1,∗</sup>

<sup>1</sup>Department of Informatics, New Jersey Institute of Technology

<sup>2</sup>Department of Mechanical and Industrial Engineering, New Jersey Institute of Technology

<sup>3</sup>Department of Computer Science, New Jersey Institute of Technology

<sup>4</sup>School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen {gz72,bo.shen,lei.zhang,fadi.deek,zhi.wei,chenxi.yuan}@njit.edu mengnandu@cuhk.edu.cn

<sup>∗</sup>Corresponding authors

## ABSTRACT

Different tasks often require interactive agents to perform similar operations at intermediate steps. Retrieving an entire past workflow can introduce irrelevant steps, whereas retrieving isolated operations can discard the dependencies needed to reuse them. We introduce EvoWeave, a framework for experience memory that represents abstract states as nodes and reusable operations as directed edges with input and output contracts, provenance, and capability references. A planner decomposes the current task into a dependency graph; operation retrieval and bounded directed search then assemble relevant experience fragments. An execution agent can inspect these fragments and their referenced capabilities, while reflection separates concrete execution evidence from reusable abstractions. We study the framework across four benchmarks, five evaluation settings, and three language model backbones. Across backbones, AppWorld Normal and Challenge task goal completion average 51.91 and 27.99, respectively, exceeding the baseline without memory by 6.99 and 4.99 percentage points. An AppWorld ablation associates directed connections with an additional improvement of 2.08 points over independent operation cards. The results motivate composition of reusable operations while exposing tradeoffs in retrieval overhead and conditions for valid reuse.

## 1 INTRODUCTION

Language model agents solve interactive tasks by interleaving reasoning with actions and environ mental feedback (Yao et al., 2022). Although task goals vary, many constituent operations may recur across tasks: locating a contact, filtering records, reading an attribute, modifying an object, or submitting a result. An agent that has already performed these operations should be able to reuse that experience when pursuing a different goal. Approaches based on experience therefore reuse insights or workflows that an agent has learned from prior interactions (Zhao et al., 2024a; Wang et al., 2024; Ouyang et al., 2026), making the organization of experience a central design choice.

One part of that choice is granularity: the unit at which experience is stored and retrieved, which decides how much of a past task is carried into a new one. Figure 1 illustrates this choice. A new request to contact a collaborator may require finding contact information, drafting an email, and sending a message. A historical workflow containing these operations, however, may also include unrelated research, profile inspection, or record keeping, so supplying that entire workflow couples useful experience to unnecessary context. We therefore propose retrieving the relevant operations separately, which allows partial reuse across historical tasks. We use three terms consistently: an operation is one abstract step of a task, such as sending a message; an action is the concrete environment call that realizes an operation during execution; and a capability is the external resource that such an action needs, such as an API definition or a library function. The same granularity principle then applies to capabilities: an operation can reference only the API definition or function required by its action, rather than an entire manual or code library. Longer context does not guarantee that a model uses the relevant parts of it (Liu et al., 2024), so we treat narrower retrieval as a hypothesis to be measured rather than a benefit to be assumed.

![](images/7e6001fc2525c30094d9d0bdaa8818553d8acbed652589533a7a69c8b7d1f1c1.jpg)  
Figure 1: Motivation for reusing individual operations and capability fragments. The illustrative comparison contrasts retrieving an entire historical workflow or capability file with selecting relevant operations and functions for a new task. Selective reuse aims to reduce irrelevant context while supporting composition across partially overlapping experiences. The figure is conceptual; its step counts and anticipated benefits are not experimental measurements or claims about every prior method.

Fine grained reuse alone does not solve composition. By composition we mean assembling several separately retrieved operations into one executable sequence for the current task, such that every operation finds the inputs it requires either in the environment or in the outputs of the operations placed before it. Two individually relevant operations may disagree about the intermediate object they consume, or require an unobserved prerequisite. A path to a downstream operation does not establish that every input is available. Prior work has already explored reusable programs (Wang et al., 2023; Yuan et al., 2024), linked memories (Xu et al., 2026), and modular procedural memory (Han et al., 2025). What these representations do not retain is, for each reusable unit, the input conditions and the directed transitions that determine whether the unit can be executed after the units retrieved alongside it. The question we address is therefore more specific: how can operation level experience preserve its conditions and historical connections while remaining selectable for a new task?

We propose EvoWeave, which represents an operation as a directed transition between abstract states, annotated with its contract, evidence, and capability references. A current task plan specifies which steps directly depend on one another. These dependencies guide retrieval of historical operations and search for connecting fragments. The execution agent receives the assembled operation fragments as context, together with an explicit list of the plan dependencies that retrieval could not connect, and it can query the experience graph again during execution. Reflection retains concrete observations in task history while proposing abstract, versioned updates to shared experience.

Our contributions are threefold: (i) an operation level representation that separates reusable transitions from concrete execution evidence; (ii) a plan guided retrieval and connection procedure that retains input requirements, candidate consistency, and search failures; and (iii) an empirical analysi spanning three backbones and five evaluation settings, with ablations that distinguish operation decomposition, directed connections, and active retrieval. We report heterogeneous gains and resource tradeoffs rather than assuming that a graph representation improves every task.

## 2 RELATED WORK

Experiential and procedural memory. ExpeL distills experiential knowledge (Zhao et al., 2024a), Reflexion uses verbal feedback (Shinn et al., 2023), and ReasoningBank organizes reasoning memories from previous interactions (Ouyang et al., 2026). CLIN maintains textual causal abstractions across trials, while Buffer of Thoughts retrieves reusable reasoning templates (Majumder et al., 2023; Yang et al., 2024). Synapse retrieves trajectories with abstracted states, Agent Workflow Memory induces workflows, and LEGOMem studies modular procedural memory (Zheng et al., 2024; Wang et al., 2024; Han et al., 2025). These approaches show that neither experience reuse nor decomposition originates with our work. EvoWeave focuses on operations as state transition edges and uses direct plan dependencies to retrieve and connect historical fragments. Ou ablations examine these connections within the reported configurations.

Structured memory and retrieval. MemoryBank and Generative Agents connect persistent memories to later interactions and behavior (Zhong et al., 2024; Park et al., 2023). Mem0 and A-MEM investigate scalable storage and memory organization, while MemGPT manages context levels (Chhikara et al., 2025; Xu et al., 2026; Packer et al., 2023). Retrieval augmented generation provides external evidence; Self-RAG further learns when to retrieve and how to critique generations (Lewis et al., 2020; Asai et al., 2024). GraphRAG and HippoRAG organize retrieval through graphs (Edge et al., 2024; Gutiérrez et al., 2024). Closer to interactive agents, AriGraph combines semantic and episodic memory in a world model, and G-Memory traces collaborative experience through a graph hierarchy (Anokhin et al., 2024; Zhang et al., 2026). EvoWeave instead centers retrieval on operation contracts and historical transitions. Reachability supplies a candidate fragment, not a certificate that a live action is feasible.

Reusable capabilities and planning. Voyager, LATM, CRAFT, and Code as Policies explore reusable skills, tool creation, or executable programs (Wang et al., 2023; Cai et al., 2024; Yuan et al., 2024; Liang et al., 2023). LearnAct improves action functions using execution feedback, while Toolformer studies learning tool use (Zhao et al., 2024b; Schick et al., 2023). Skill Retrieval Augmentation distinguishes retrieving a skill, incorporating it, and completing the task (Su et al., 2026), supporting our separation of memory references from execution evidence. LLMCompiler and SwiftSage address planning and execution, and LATS searches action trajectories using environmental feedback (Kim et al., 2023; Lin et al., 2023; Zhou et al., 2023). Our bounded search connects stored operations rather than simulating future actions, and it requires no update to backbone parameters. AgentBoard’s analysis of progress beyond final success further motivates examining category behavior and reuse alongside aggregate scores (Ma et al., 2024).

## 3 METHOD: EVOWEAVE

Figure 2 shows the flow from task planning to experience retrieval, capability access, execution, and reflection.

## 3.1 EXPERIENCE AS OPERATIONS WITH EXPLICIT CONTRACTS

Given a task $q ,$ an environment, and a concrete interaction history $H _ { t }$ , the agent must produce actions within an interaction budget. We distinguish its current plan $\dot { P _ { t } }$ , which is a directed acyclic graph (DAG), from the historical experience graph $G = ( V , { \dot { E } } )$ , which can contain cycles and multiple edges between states. A reusable operation is

$$
e = ( u , v , d , \mathcal { I } , \mathcal { O } , \mathcal { C } , \mathcal { R } , \pi ) ,\tag{1}
$$

where u, v are abstract source and target states, d is an operation description, I and O specify inputs and outputs, C records conditions and effects, R contains capability references, and π records provenance and versions. Task specific entities and values become placeholders in shared experience; their actual bindings remain in $H _ { t }$ . An operation with several required inputs retains their conjunction, including when a retrieved path supplies only one of them.

![](images/c4225be7f23853fed33c038a106ce74ab818e30a557e30d3503dcbe931c823ca.jpg)  
Figure 2: Overview of EvoWeave. A planner generates a task DAG that guides retrieval from an experience graph, whose nodes represent states and edges represent operations. The main agent uses retrieved experience and relevant capabilities, while the reflector separates concrete step summaries from abstract operation summaries. Shared memory updates follow the declared construction or evaluation protocol. Capability and role refinement in the diagram describe the broader architecture; evaluated implementation boundaries are detailed in Appendix B.

## 3.2 RETRIEVAL AND CONNECTIONS GUIDED BY THE PLAN

The planner uses q and $H _ { t }$ to specify remaining steps $S _ { t }$ and their direct dependencies $D _ { t }$ , without first conditioning the plan on available memory. For each step $s _ { i }$ , operation retrieval uses description embeddings $f \colon$

$$
C _ { i } = \mathrm { T o p K } _ { k = 3 } \{ e \in E : \cos \left( f ( s _ { i } ) , f ( d _ { e } ) \right) \geq \tau _ { \mathrm { r e t r i e v e } } \} \ .\tag{2}
$$

Filtering precedes ranking, and an empty candidate set is valid. Candidates retain edge identities, endpoints, and contracts; operations sharing a source state remain distinct.

For each direct dependency $( i , j ) \in D _ { t } .$ , consider a candidate pair $a \in C _ { i } , b \in C _ { j }$ . A connection searches from the target state $v _ { a }$ to the source state $u _ { b }$ and retains both matched operations:

$$
\Gamma _ { i j } = a \circ \mathrm { B F S } _ { G } ( v _ { a } , u _ { b } ) \circ b .\tag{3}
$$

Breadth first search minimizes the number of connecting operation edges under fixed search limits. At most nine candidate pairs are examined per plan dependency. Ties use candidate similarity and stable identifiers. We do not add connection requests for transitive plan dependencies. An unsuccessful search reports either a gap or a budget truncation, instead of inventing an edge.

Connections sharing a plan step must use a consistent candidate to form one chain. If local choices cannot be reconciled, we retain separate fragments and expose a candidate conflict. This conservative procedure is not a global assignment of minimum total cost. Similarly, a graph path does not establish that all inputs are bound, entities are consistent, or concurrent writes are safe. The main agent checks these conditions against actual observations before acting.

## 3.3 EXECUTION AND CAPABILITY ACCESS

The main agent receives the plan, retrieved operation cards, connecting fragments, and missing conditions. Before opening capability bodies, it can inspect an operation, retrieve neighbors, search descriptions, or request a path. Cards expose identifiers, contracts, provenance, and versions; subsequent requests load the relevant API or function body. Official API definitions remain immutable, with learned usage information stored separately. New code capabilities require validation and tests. The agent generates current task actions using these references and can reject a retrieved suggestion or request replanning. Events recording memory references are distinct from evidence that a stored program was directly executed.

## 3.4 REFLECTION AND CONTROLLED MEMORY UPDATES

Following execution, reflection produces two distinct artifacts. Concrete history records actual in puts, outputs, errors, and evidence references. Abstract experience records reusable descriptions, placeholders, conditions, and capability references. Every update branch performs this abstraction, including additions to existing operations. A separate similarity threshold $\tau _ { \mathrm { m e r g e } }$ selects candidates for comparison; similarity alone does not authorize a merge. Contract checks distinguish compatible evidence additions, variants, and new operations. For example, requesting a payment and transferring money must remain distinct despite lexical similarity.

Failures retain the observed error and any evidenced recovery, without promoting expected outputs into successful outcomes. Embedding failures are distinguished from legitimate retrieval without a match. Memory proposals undergo structural, capability, and contract validation before publication. Having three persistent roles (planner, main, and reflector) does not imply exactly three model calls per round: active retrieval and merge decisions may introduce additional calls. Appendix B provides pseudocode and default parameters for the reference implementation.

## 4 EXPERIMENTS

## 4.1 BENCHMARKS, BACKBONES, AND REPORTING PROTOCOL

We analyze five settings from four benchmarks: AppWorld Test-Normal and Test-Challenge (Trivedi et al., 2024), WebArena (Zhou et al., 2024), ALFWorld (Shridhar et al., 2020), and ScienceWorld (Wang et al., 2022). AppWorld reports task goal completion (TGC) and scenario goal completion (SGC); WebArena reports success rate (SR). ALFWorld and ScienceWorld are reported with their own benchmark specific scores, whose scoring semantics need not be identical. Higher values are better for all seven columns. We average matching metrics equally over backbones and never combine heterogeneous metrics into one score. Differences are computed before rounding; display values are rounded afterward.

The backbones are DeepSeek-V3.2 (Liu et al., 2025), GPT-4.1-mini, and Qwen3.8-27B. Qwen is served locally on an institutional HPC cluster; DeepSeek-V3.2 and GPT-4.1-mini are accessed through API calls. The analysis uses aggregate results for these configurations. Token budgets are not equalized by construction across methods; Appendix F reports the tokens each one actually consumes.

We compare No-Memory and Flat-Memory with AWM (Wang et al., 2024), ReasoningBank (Ouyang et al., 2026), Mem0 (Chhikara et al., 2025), ExpeL (Zhao et al., 2024a), and A-MEM (Xu et al., 2026). These comparisons concern the evaluated implementations and do not establish the maximum performance attainable by each method under other adaptations or budgets.

Memory construction protocol. No memory is built from training or development data. Every run starts from an empty memory and accumulates experience online during evaluation: test tasks are executed one at a time in the order in which the benchmark releases them, and experience written after a task becomes available to all later tasks in the same run. Each of the 15 combinations of backbone and evaluation setting is an independent run that starts empty, so no experience crosses backbones or benchmarks. Reflection records what the agent observes within an episode, namely API return values, environment errors, and observations, together with an episode level success judgement that a language model derives from the trajectory itself. Methods that need a success signal, such as ExpeL and AWM, take it from this same module, so no method receives a more reliable signal than another. The official task level score is computed after an episode terminates, is used only for reporting, and never reaches the memory of any method. Every memory based baseline runs under exactly this protocol, from the same empty start and the same task order, so the comparison isolates how experience is organized rather than how much data each method was allowed to see.

Table 1: Main results reported as mean ± standard deviation (SD), with the $\pm \mathrm { \bf ~ S D }$ terms shown as subscripts. All entries retain the original score scale from 0 to 100. Within each backbone block, the highest mean in a column is shaded and the second highest is underlined; tied values receive the same mark.
<table><tr><td rowspan="2">Method</td><td colspan="2">AppWorld Normal</td><td colspan="2">AppWorld Challenge</td><td rowspan="2"></td><td colspan="2">WebArena ALFWorld ScienceWorld</td></tr><tr><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td><td>SR Score</td><td>Score</td></tr><tr><td>DeepSeek-V3.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Memory</td><td> $4 7 . 0 2 { \scriptstyle \pm 2 . 7 5 }$ </td><td> $3 0 . 3 6 { \scriptstyle \pm 3 . 3 0 }$ </td><td> $2 3 . 0 2 { \scriptstyle \pm 1 . 9 7 }$ </td><td> $9 . 3 5 { \scriptstyle \pm 1 . 8 3 }$ </td><td> $4 2 . 0 0 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 9 . 4 0 { \scriptstyle \pm 2 . 4 1 }$ </td><td> $6 4 . 7 0 { \scriptstyle \pm 2 . 5 2 }$ </td></tr><tr><td>Flat-Memory</td><td> $4 9 . 0 5 _ { \pm 3 . 0 3 }$ </td><td> $3 2 . 1 4 { \scriptstyle \pm 3 . 6 8 }$ </td><td> $2 3 . 0 2 _ { \pm 2 . 1 6 }$ </td><td> $9 . 3 5 { \scriptstyle \pm 2 . 0 1 }$ </td><td> $4 3 . 0 0 { \scriptstyle \pm 1 . 4 9 }$ </td><td> $7 1 . 9 4 { \scriptstyle \pm 2 . 5 9 }$ </td><td> $6 7 . 2 0 { \scriptstyle \pm 2 . 7 7 }$ </td></tr><tr><td>AWM</td><td> $5 2 . 0 2 { \scriptstyle \pm 2 . 9 2 }$ </td><td> $3 5 . 0 0 { \scriptstyle \pm 3 . 6 2 }$ </td><td> $2 4 . 9 9 { \scriptstyle \pm 2 . 1 2 }$ </td><td> $1 0 . 5 0 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $4 6 . 0 1 _ { \pm 1 . 4 4 }$ </td><td> $7 3 . 8 8 { \scriptstyle \pm 2 . 4 7 }$ </td><td> $6 8 . 7 0 { \scriptstyle \pm 2 . 6 7 }$ </td></tr><tr><td>ReasoningBank</td><td> $\overline { { 5 2 . 0 1 { \pm } 3 . 9 9 } }$ </td><td> $\overline { { 3 5 . 0 0 { \pm } 4 . 9 5 } }$ </td><td> $2 6 . 0 0 { \scriptstyle \pm 2 . 9 2 }$ </td><td> $1 1 . 8 8 { \scriptstyle \pm 2 . 7 6 }$ </td><td> $\overline { { 4 3 . 0 0 { \scriptstyle \pm 1 . 9 7 } } }$ </td><td> $7 1 . 3 4 { \scriptstyle \pm 3 . 4 6 }$ </td><td> $6 7 . 7 0 { \scriptstyle \pm 3 . 6 6 }$ </td></tr><tr><td>Mem0</td><td> $4 7 . 5 0 { \scriptstyle \pm 3 . 0 8 }$ </td><td> $3 0 . 7 1 { \scriptstyle \pm 3 . 7 1 }$ </td><td> $2 2 . 4 9 _ { \pm 2 . 1 9 }$ </td><td> $9 . 0 6 _ { \pm 2 . 0 4 }$ </td><td> $4 2 . 5 1 { \scriptstyle \pm 1 . 5 2 }$ </td><td> $6 9 . 4 0 { \scriptstyle \pm 2 . 6 9 }$ </td><td> $6 5 . 2 0 { \scriptstyle \pm 2 . 8 2 }$ </td></tr><tr><td>ExpeL</td><td> $5 0 . 9 5 { \scriptstyle \pm 2 . 9 5 }$ </td><td> $3 3 . 9 3 { \scriptstyle \pm 3 . 6 3 }$ </td><td> $2 4 . 9 9 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $1 0 . 5 0 { \scriptstyle \pm 2 . 0 1 }$ </td><td> $4 5 . 0 0 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $7 6 . 1 2 _ { \pm 2 . 4 5 }$ </td><td> $7 4 . 1 0 { \scriptstyle \pm 2 . 7 0 }$ </td></tr><tr><td>A-MEM</td><td> $4 7 . 9 8 _ { \pm 3 . 2 5 }$ </td><td> $3 1 . 0 7 { \scriptstyle \pm 3 . 9 2 }$ </td><td> $2 3 . 0 2 _ { \pm 2 . 3 2 }$ </td><td> $9 . 3 5 { \scriptstyle \pm 2 . 1 6 }$ </td><td> $4 3 . 0 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $7 0 . 9 0 { \scriptstyle \pm 2 . 8 1 }$ </td><td> $6 6 . 7 0 { \scriptstyle \pm 2 . 9 8 }$ </td></tr><tr><td>EvoWeave</td><td> $5 4 . 0 5 { \scriptstyle \pm 2 . 8 8 }$ </td><td> $3 7 . 1 4 { \scriptstyle \pm 3 . 6 3 }$ </td><td> $2 7 . 0 0 { \scriptstyle \pm 2 . 1 3 }$ </td><td> $1 1 . 6 5 { \scriptstyle \pm 2 . 0 2 }$ </td><td> $4 9 . 0 0 { \scriptstyle \pm 1 . 4 3 }$ </td><td> $7 5 . 8 8 { \scriptstyle \pm 2 . 4 5 }$ </td><td> $7 6 . 2 0 { \scriptstyle \pm 2 . 6 5 }$ </td></tr><tr><td>GPT-4.1-mini</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Memory</td><td> $3 5 . 7 1 { \scriptstyle \pm 3 . 1 8 }$ </td><td> $2 0 . 7 1 { \scriptstyle \pm 3 . 5 3 }$ </td><td> $1 6 . 9 8 { \scriptstyle \pm 2 . 2 2 }$ </td><td> $6 . 3 3 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $3 0 . 6 9 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $4 5 . 9 7 _ { \pm 3 . 0 7 }$ </td><td> $4 0 . 0 0 { \scriptstyle \pm 3 . 0 3 }$ </td></tr><tr><td>Flat-Memory</td><td> $3 6 . 6 7 _ { \pm 3 . 5 2 }$ </td><td> $2 1 . 4 3 { \scriptstyle \pm 3 . 9 3 }$ </td><td> $1 6 . 0 2 _ { \pm 2 . 4 2 }$ </td><td> $5 . 9 0 { \scriptstyle \pm 2 . 2 1 }$ </td><td> $3 0 . 6 9 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $4 8 . 9 6 _ { \pm 3 . 3 7 }$ </td><td> $4 1 . 0 0 { \scriptstyle \pm 3 . 3 3 }$ </td></tr><tr><td>AWM</td><td> $3 8 . 6 9 { \scriptstyle \pm 3 . 4 2 }$ </td><td> $2 2 . 8 6 { \scriptstyle \pm 3 . 8 6 }$ </td><td> $1 7 . 9 9 { \scriptstyle \pm 2 . 3 8 }$ </td><td> $6 . 7 6 { \scriptstyle \pm 2 . 1 8 }$ </td><td> $3 3 . 6 9 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $5 1 . 0 4 { \scriptstyle \pm 3 . 2 7 }$ </td><td> $4 4 . 0 0 { \scriptstyle \pm 3 . 2 1 }$ </td></tr><tr><td>ReasoningBank</td><td> $3 9 . 7 1 _ { \pm 3 . 4 2 }$ </td><td> $2 4 . 6 4 _ { \pm 3 . 9 2 }$ </td><td> $2 0 . 0 0 { \scriptstyle \pm 2 . 4 1 }$  </td><td> $7 . 7 7 { \scriptstyle \pm 2 . 2 2 }$ </td><td> $3 5 . 6 9 { \scriptstyle \pm 1 . 6 9 }$ </td><td> $5 1 . 9 4 { \scriptstyle \pm 3 . 2 3 }$ </td><td> $4 6 . 0 0 { \scriptstyle \pm 3 . 1 8 }$ </td></tr><tr><td>Mem0</td><td> $3 5 . 7 1 { \scriptstyle \pm 3 . 5 7 }$ </td><td> $\overline { { 2 0 . 7 1 _ { \pm 3 . 9 5 } } }$ </td><td> $\overline { { 1 6 . 0 2 _ { \pm 2 . 4 6 } } }$ </td><td> $5 . 9 0 { \scriptstyle \pm 2 . 2 5 }$ </td><td> $\overline { { 3 0 . 6 9 _ { \pm 1 . 7 7 } } }$ </td><td> $4 5 . 0 7 _ { \pm 3 . 4 2 }$ </td><td> $4 0 . 0 0 { \scriptstyle \pm 3 . 3 9 }$ </td></tr><tr><td>ExpeL</td><td> $3 8 . 6 9 { \scriptstyle \pm 3 . 4 6 }$ </td><td> $2 2 . 8 6 { \scriptstyle \pm 3 . 9 0 }$ </td><td> $1 7 . 9 9 2 . 4 1$ </td><td> $6 . 7 6 { \scriptstyle \pm 2 . 2 0 }$ </td><td> $3 2 . 7 1 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $5 4 . 0 3 { \scriptstyle \pm 3 . 2 9 }$ </td><td></td></tr><tr><td>A-MEM</td><td> $3 6 . 6 7 _ { \pm 3 . 7 8 }$ </td><td> $2 1 . 4 3 { \scriptstyle \pm 4 . 2 1 }$ </td><td> $1 6 . 9 8 { \scriptstyle \pm 2 . 6 2 }$ </td><td> $6 . 3 3 { \scriptstyle \pm 2 . 4 0 }$ </td><td> $3 1 . 7 0 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $\overline { { 4 8 . 0 6 _ { \pm 3 . 6 2 } } }$ </td><td> $4 8 . 0 0 { \scriptstyle \pm 3 . 2 4 }$ </td></tr><tr><td>EvoWeave</td><td> $4 0 . 6 0 { \scriptstyle \pm 4 . 5 9 }$ </td><td> $2 6 . 7 9 { \scriptstyle \pm 5 . 0 3 }$ </td><td> $2 1 . 0 1 _ { \pm 3 . 1 5 }$ </td><td></td><td></td><td></td><td> $\overline { { 4 2 . 0 0 { \scriptstyle \pm 3 . 5 7 } } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $7 . 4 7 _ { \pm 2 . 8 7 }$ </td><td> $3 6 . 7 0 { \scriptstyle \pm 2 . 2 9 }$ </td><td> $5 7 . 9 7 _ { \pm 4 . 4 7 }$ </td><td> $5 1 . 0 0 { \scriptstyle \pm 4 . 3 9 }$ </td></tr><tr><td>Qwen3.8-27B No-Memory</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Flat-Memory</td><td> $5 2 . 0 2 _ { \pm 3 . 0 3 }$ </td><td> $3 5 . 0 0 { \scriptstyle \pm 3 . 7 5 }$ </td><td> $2 9 . 0 2 _ { \pm 2 . 2 6 }$ </td><td> $1 2 . 9 5 { \scriptstyle \pm 2 . 1 7 }$ </td><td> $4 3 . 9 9 { \scriptstyle \pm 1 . 5 0 }$ </td><td> $7 1 . 9 4 { \scriptstyle \pm 2 . 6 0 }$ </td><td> $6 6 . 1 0 { \scriptstyle \pm 2 . 7 7 }$ </td></tr><tr><td>AWM</td><td> $5 4 . 0 5 { \scriptstyle \pm 3 . 3 2 }$ </td><td> $3 7 . 1 4 { \scriptstyle \pm 4 . 1 8 }$ </td><td> $3 0 . 0 2 { \scriptstyle \pm 2 . 5 0 }$ </td><td> $1 3 . 5 3 { \scriptstyle \pm 2 . 4 1 }$ </td><td> $4 5 . 0 0 { \scriptstyle \pm 1 . 6 4 }$ </td><td> $7 4 . 0 3 { \scriptstyle \pm 2 . 8 0 }$ </td><td> $6 8 . 0 0 { \scriptstyle \pm 3 . 0 5 }$ </td></tr><tr><td>ReasoningBank</td><td> $5 7 . 0 2 _ { \pm 3 . 1 8 }$ </td><td> $4 0 . 3 6 { \scriptstyle \pm 4 . 0 9 }$ </td><td> $3 0 . 9 8 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $1 4 . 2 4 { \scriptstyle \pm 2 . 3 5 }$ </td><td> $4 8 . 0 0 { \scriptstyle \pm 1 . 5 9 }$ </td><td> $7 5 . 9 7 { \scriptstyle \pm 2 . 6 7 }$ </td><td> $7 0 . 5 0 { \scriptstyle \pm 2 . 9 4 }$ </td></tr><tr><td></td><td> $5 7 . 9 8 _ { \pm 3 . 1 5 }$ </td><td> $4 1 . 4 3 { \scriptstyle \pm 4 . 0 6 }$ </td><td> $3 3 . 0 0 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $1 5 . 5 4 { \scriptstyle \pm 2 . 3 8 }$ </td><td> $5 0 . 0 0 { \scriptstyle \pm 1 . 5 8 }$ </td><td> $7 7 . 0 1 { \scriptstyle \pm 2 . 6 1 }$ </td><td> $7 2 . 2 0 { \scriptstyle \pm 2 . 9 1 }$ </td></tr><tr><td>Mem0</td><td> $5 2 . 9 8 { \scriptstyle \pm 3 . 3 9 }$ </td><td> $3 6 . 0 7 { \scriptstyle \pm 4 . 2 3 }$ </td><td> $2 9 . 0 2 { \scriptstyle \pm 2 . 5 3 }$ </td><td> $1 2 . 9 5 { \scriptstyle \pm 2 . 4 3 }$ </td><td> $4 5 . 0 0 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $7 1 . 9 4 { \scriptstyle \pm 2 . 9 0 }$ </td><td> $6 7 . 0 0 { \scriptstyle \pm 3 . 1 1 }$ </td></tr><tr><td>ExpeL</td><td> $5 5 . 9 5 { \scriptstyle \pm 3 . 2 2 }$ </td><td> $3 9 . 2 9 _ { \pm 4 . 1 1 }$ </td><td> $3 0 . 9 8 { \scriptstyle \pm 2 . 4 4 }$ </td><td> $1 4 . 2 4 { \scriptstyle \pm 2 . 3 7 }$ </td><td> $4 7 . 0 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $7 8 . 9 6 _ { \pm 2 . 6 1 }$ </td><td> $7 4 . 1 0 { \scriptstyle \pm 2 . 9 7 }$ </td></tr><tr><td>A-MEM</td><td> $5 4 . 0 5 { \scriptstyle \pm 3 . 5 7 }$ </td><td> $3 7 . 1 4 { \scriptstyle \pm 4 . 4 8 }$ </td><td> $3 0 . 0 2 { \scriptstyle \pm 2 . 6 8 }$ </td><td> $1 3 . 5 3 { \scriptstyle \pm 2 . 5 8 }$ </td><td> $4 6 . 0 1 _ { \pm 1 . 7 7 }$ </td><td> $\overline { { 7 4 . 0 3 _ { \pm 3 . 0 1 } } }$ </td><td> $6 9 . 0 0 { \scriptstyle \pm 3 . 2 7 }$ </td></tr><tr><td>EvoWeave</td><td> $6 1 . 0 7 { \scriptstyle \pm 4 . 3 6 }$ </td><td> $4 3 . 9 2 { \scriptstyle \pm 5 . 5 9 }$ </td><td> $3 5 . 9 7 { \scriptstyle \pm 3 . 3 3 }$ </td><td> $1 7 . 8 2 { \scriptstyle \pm 3 . 2 5 }$ </td><td> $4 9 . 0 0 { \scriptstyle \pm 2 . 1 7 }$ </td><td> $7 9 . 0 3 { \scriptstyle \pm 3 . 7 2 }$ </td><td> $7 3 . 7 0 { \scriptstyle \pm 4 . 0 2 }$ </td></tr></table>

The setting is therefore continual learning over an evaluation stream, applied identically to every method; Appendix A.3 records the protocol in full.

Main results are reported as mean ± standard deviation (SD). We report descriptive differences without significance claims. Appendix A details the evaluation scope, aggregation rules, and comparability of benchmark and category summaries.

## 4.2 MAIN RESULTS

Table 1 shows that EvoWeave has the highest reported mean in 16 of the 21 combinations of backbone and metric. Across backbones, Normal TGC increases from 44.92 to 51.91 and Challenge TGC from 23.01 to 27.99 relative to No-Memory. In both cases the strongest baseline macro mean comes from ReasoningBank, over which EvoWeave gains 2.01 and 1.66 points, respectively. Averages across backbones for WebArena SR, ALFWorld Score, and ScienceWorld Score reach 44.90, 70.96, and 66.97, respectively, compared with 38.89, 62.44, and 56.93 without memory.

The gains are not uniform. ReasoningBank exceeds EvoWeave on DeepSeek and GPT Challenge SGC and on Qwen WebArena SR. ExpeL has the higher mean on DeepSeek ALFWorld and Qwen ScienceWorld. Some differences are small relative to the reported standard deviations. These exceptions matter: matching and connecting historical operations does not guarantee that the selected experience fits the current task or improves every backbone.

![](images/1b12549affe629c607aa74225d3aebdad7c5c2d626b762d072cc475d75b25e13.jpg)  
Figure 3: Category performance averaged equally over three backbones. Each spoke uses a local linear range determined by all eight methods; brackets give its inner and outer bounds in original score units. Compare methods on the same spoke: distances across spokes, panels, and polygon areas are not directly comparable. Labels and the scaling rule are defined in Appendix C. These category aggregates are distinct from Table 1; see Appendix A.

## 4.3 CATEGORY PERFORMANCE

Figure 3 separates category behavior from benchmark averages. For EvoWeave, AppWorld Challenge ranges from 17.01 on Gmail to 36.83 on Spotify, while ALFWorld ranges from 61.57 for Use

Table 2: AppWorld ablation, averaged equally over backbones. Overall TGC uses weights based on task counts of 168 Normal and 417 Challenge tasks within each backbone. These configurations are specific to AppWorld; only point estimates are shown. Configurations are ordered so that the full system appears last, and the highest value in each column is shown in bold.
<table><tr><td>Configuration</td><td>Normal TGC</td><td>Challenge TGC</td><td>Overall TGC</td></tr><tr><td>A: No memory</td><td>44.92</td><td>23.01</td><td>29.30</td></tr><tr><td>B: Whole workflow RAG</td><td>46.59</td><td>23.02</td><td>29.79</td></tr><tr><td>C: Operation cards</td><td>47.93</td><td>24.72</td><td>31.39</td></tr><tr><td>D: C + directed connections</td><td>50.74</td><td>26.51</td><td>33.47</td></tr><tr><td>E: C + active retrieval</td><td>49.90</td><td>25.94</td><td>32.82</td></tr><tr><td>F: EvoWeave (full)</td><td>51.91</td><td>27.99</td><td>34.86</td></tr></table>

Table 3: Mean memory size over the five settings. Counts describe the reported graph snapshots, not an inferred average topology. Capability links connect the corresponding node type to operations. Counts of role nodes do not establish that learning of role prompts was evaluated.
<table><tr><td>Experience layer Mean count</td><td>States 27.0</td><td>Operations 157.6</td><td></td></tr><tr><td>Capability layer</td><td>Code</td><td>Skills</td><td>Roles</td></tr><tr><td>Mean nodes</td><td>185.2</td><td>82.2</td><td>6.8</td></tr><tr><td>Mean links to operations</td><td>157.6</td><td>362.0</td><td>99.4</td></tr></table>

Two Objects to 79.26 for Examine in Light. ScienceWorld varies from 45.63 for Mendelian genetics to 88.41 for Living and Non-living Things. Thus, a strong aggregate does not imply uniform performance. The local axis ranges expose differences among methods without altering the original scores. All 15 panels for individual backbones appear in Appendix C, preserving reversals and the original Supervisor category. WebArena category means weighted by task counts sit slightly above the main table because the five site categories cover 764 of the 812 tasks and exclude the 48 cross site tasks; Appendix A quantifies that difference for each method.

## 4.4 ABLATION: DECOMPOSITION, CONNECTIONS, AND ACTIVE RETRIEVAL

Table 2 reports six configurations. A removes memory and B retrieves whole workflows, so they coincide with the No-Memory and Flat-Memory rows of Table 1; C retrieves independent operation cards; D adds directed connections to C; E adds active retrieval to C without historical connections; and F is the full system, which adds active graph retrieval to D. Appendix D reports every backbone separately.

Operation cards improve Overall TGC from 29.79 for whole workflow retrieval to 31.39. Adding directed connections further raises it to 33.47, a difference of 2.08 points. Active retrieval without connections reaches 32.82, improving on C by 1.44 points. The full system reaches 34.86, another 1.39 points above D and 2.04 points above E. These comparisons support the hypothesis that relational context contributes beyond decomposition and additional lookup opportunities. The ordering of D and E is not reproduced by every backbone: E slightly exceeds D on DeepSeek, while GPT and Qwen keep D above E (Appendix D). All six configurations share the memory construction protocol of Appendix A.3, the same candidate pool, and the same capability library, so they differ only in how retrieved experience is connected and queried.

## 4.5 MEMORY STRUCTURE AND OPERATION REUSE

Table 3 summarizes graph size. An average snapshot contains 27.0 state nodes and 157.6 operation edges; capability references extend across code, skills, and role nodes. These values describe the separation of transitions from the capabilities they reference, but do not measure successful composition or graph quality. The counts for individual settings vary substantially: AppWorld Challenge has 431 operations, compared with 14 in ALFWorld (Appendix E). These arithmetic summaries characterize size and composition without defining a shared adjacency structure.

![](images/eeb4e84a132c0ab5241e9f6939ec9b85e892f8be022becfd7e76df3b8c34eadb.jpg)  
Figure 4: Mean operation overlap and reported memory referencing across task types. Each type receives equal weight; overlap comparisons are restricted to tasks of the same type. Labels show type counts, and bars show percentages on a scale from 0 to 100. Benchmark specific overlap proxie and reference rates are distinct measures (Appendix G).

Figure 4 summarizes recurring operations within task types. The equal type means of overlap/reference (%) are 48.5/49.2 for AppWorld Normal, 57.3/58.8 for Challenge, 35.3/36.7 for WebArena, 96.5/96.9 for ALFWorld, and 90.2/92.1 for ScienceWorld. Cross type comparisons are excluded from the overlap averages. These substantial within type overlaps motivate operation level abstraction: related tasks can reuse shared steps even when complete workflows differ. Appendix G defines the averages and provides all type specific values.

## 4.6 CONTEXT AND RUNTIME TRADEOFFS

The efficiency results show 6,155.46 injected memory tokens per question for EvoWeave, versus 26,300.80 for Flat-Memory, 19,890.00 for AWM, and 11,872.00 for ReasoningBank: reductions of 76.60%, 69.05%, and 48.15%, respectively. These figures support selective context delivery, but the total input token measure falls by only 0.99% relative to No-Memory. EvoWeave has the fewest rounds (15.94) and the highest seconds per round (10.44), alongside lower KV reuse (45%). Less injected memory therefore does not by itself demonstrate lower latency or cost. Appendix F gives all efficiency metrics and explains the boundaries of token and runtime comparisons.

## 5 CONCLUSION

EvoWeave organizes reusable experience as operation edges with explicit contracts and connects retrieved fragments according to a current task dependency plan. This design preserves local reuse opportunities while exposing missing conditions and inconsistent connections. Results across three backbones and five settings are broadly favorable, and AppWorld ablations associate both operation decomposition and directed connections with improved task completion. The framework also highlights that smaller memory context can incur additional retrieval and reasoning overhead. Further work should examine generalization to unseen operation compositions and the effect of retrieval overhead on end to end performance.

## ETHICS STATEMENT

This work follows the ICLR Code of Ethics. No human participants or animal experiments were involved in this study. All benchmarks used in this work, namely AppWorld, WebArena, ALFWorld, and ScienceWorld, are publicly released research benchmarks that run in simulated environments, and we used them in accordance with their respective licenses and usage policies. Throughout the research process, we made efforts to minimize potential biases and prevent discriminatory outcomes. No personally identifiable information was collected or utilized, and no experiments were performed that would pose potential privacy or security risks. We remain committed to ensuring transparency, ethical conduct, and research integrity throughout the study.

## REPRODUCIBILITY STATEMENT

We have taken the following steps to make this work reproducible. The operation representation, the plan guided retrieval and connection procedure, and the reflection and memory update rules are defined in Section 3; Appendix B gives the corresponding pseudocode (Algorithm 1), the complexity bound on the number of bounded searches, and the complete default parameters of the reference implementation (Table 5), including retrieval and merge thresholds, candidate counts, search depth and expansion limits, and the budget for visible operation cards. All four benchmarks are publicly available, and we use their released task splits without modification; the benchmarks, backbones, baseline methods, metrics, and aggregation rules are enumerated in Section 4 and summarized in Table 4. Appendix A states the aggregation formulas, the task counts used for weighted means, and the exact sense in which category, graph, overlap, and efficiency summaries are computed over populations that differ from the main benchmark means. Appendices D to G report the per backbone tables, per setting graph statistics, token and runtime metrics, and category level overlap definitions underlying every figure in the main text. Two limitations bound exact reproduction and we state them explicitly. First, decoding is stochastic and two of the three backbones are served through provider APIs whose deployed checkpoints we do not control, so individual runs will not match digit for digit; we therefore report means with standard deviations and make only descriptive comparisons, as explained in Appendix A. Second, the baselines are re-implementations evaluated under our harnes rather than numbers copied from their original papers. They all run under the common memory construction protocol of Appendix A.3, so the comparison holds the memory source, the task order, the success judgement module, and the per run isolation fixed. Token budgets are not equalized by construction; Appendix F reports the injected and total input tokens of every method, and EvoWeave has the smallest counts of both, so its results are not obtained from a larger context.

## AI USE STATEMENT

Large Language Models (LLMs) were utilized to support the writing and language refinement of this manuscript. In particular, the LLMs were employed to improve the clarity, readability, and overall linguistic quality of the text across different sections of the paper. Their assistance included rephrasing sentences, correcting grammatical issues, and improving the coherence and flow of the manuscript.

It should be emphasized that the LLM was not involved in developing the research ideas, designing the methodology, or conducting the experiments. All research questions, concepts, methodological decisions, and analyses were independently developed and carried out by the authors. The role of the LLM was strictly limited to language-related assistance and did not extend to the scientific content, experimental analysis, or interpretation of the data.

The authors remain fully responsible for all content presented in the manuscript, including any text that was generated or subsequently refined with the assistance of an LLM. All LLM-assisted text was reviewed by the authors to ensure compliance with relevant ethical standards and to avoid plagiarism or any form of scientific misconduct.

## REFERENCES

Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry Evseev, Andrey Kravchenko, Mikhail Burtsev, and Evgeny Burnaev. Arigraph: Learning knowledge graph world models with episodic memory for llm agents. arXiv preprint arXiv:2407.04363, 2024.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pp. 9112–9141, 2024.

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. Large language models as tool makers. In International Conference on Learning Representations, volume 2024, pp. 54067– 54089, 2024.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. Advances in neural information processing systems, 37:59532–59569, 2024.

Dongge Han, Camille Couturier, Daniel Madrigal Diaz, Xuchao Zhang, Victor Rühle, and Saravan Rajmohan. Legomem: Modular procedural memory for multi-agent llm systems for workflow automation. arXiv preprint arXiv:2510.04851, 2025.

Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. An llm compiler for parallel function calling. arXiv preprint arXiv:2312.04511, 2023.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International conference on robotics and automation (ICRA), pp. 9493–9500. IEEE, 2023.

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. Advances in Neural Information Processing Systems, 36:23813–23825, 2023.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the associationfor computational linguistics, 12:157–173, 2024.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents. Advances in neural information processing systems, 37:74325–74362, 2024.

Bodhisattwa Prasad Majumder, Bhavana Dalvi Mishra, Peter Jansen, Oyvind Tafjord, Niket Tandon, Li Zhang, Chris Callison-Burch, and Peter Clark. Clin: A continually learning language agent for rapid task adaptation and generalization. arXiv preprint arXiv:2310.10134, 2023.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. In International Conference on Learning Representations, volume 2026, pp. 94327–94354, 2026.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology, pp. 1–22, 2023.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539– 68551, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.

Weihang Su, Jianming Long, Qingyao Ai, Qiaozhi He, Yichen Tang, Changyue Wang, Yiteng Tu, Yingbo Wang, and Yiqun Liu. Skill retrieval augmentation for agentic ai. arXiv preprint arXiv:2604.24594, 2026.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. Appworld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16022– 16076, 2024.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, 2022.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2026.

Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. Buffer of thoughts: Thought-augmented reasoning with large language models. Advances in Neural Information Processing Systems, 37:113519–113544, 2024.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi Fung, Hao Peng, and Heng Ji. Craft: Customizing llms by creating and retrieving from specialized toolsets. In International Conference on Learning Representations, volume 2024, pp. 40097–40125, 2024.

Guibin Zhang, Muxin Fu, Kun Wang, Frank Wan, Miao Yu, and Shuicheng Yan. G-memory: Tracing hierarchical memory for multi-agent systems. Advances in Neural Information Processing Systems, 38:12988–13018, 2026.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024a.

Haiteng Zhao, Chang Ma, Guoyin Wang, Jing Su, Lingpeng Kong, Jingjing Xu, Zhi-Hong Deng, and Hongxia Yang. Empowering large language model agents through action learning. arXiv preprint arXiv:2402.15809, 2024b.

Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. Synapse: Trajectory-as-exemplar prompting with memory for computer control. In International Conference on Learning Representations, volume 2024, pp. 19036–19066, 2024.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 19724–19731, 2024.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406, 2023.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pp. 15585–15606, 2024.

## A EVALUATION SCOPE AND STATISTICAL REPORTING

This appendix describes the scope and aggregation of the experimental results. The analysis combines benchmark summaries, category scores, ablations, graph statistics, and efficiency measurements. These summaries answer different questions and need not have identical task populations or aggregation rules. We therefore interpret each metric within its evaluation setting and distinguish descriptive comparisons from statistical significance.

## A.1 EVALUATION SCOPE

<table><tr><td>Item</td><td>Evaluation scope and reporting convention</td></tr><tr><td>Benchmarks</td><td>AppWorld Test-Normal and Test-Challenge, WebArena, ALFWorld, and Sci- enceWorld. These give five settings from four benchmarks.</td></tr><tr><td>Backbones</td><td>DeepSeek-V3.2, GPT-4.1-mini, and Qwen3.8-27B.</td></tr><tr><td>Deployment</td><td>Qwen served locally on an institutional HPC cluster; DeepSeek-V3.2 and GPT-4.1-mini accessed through API calls.</td></tr><tr><td>Methods</td><td>No-Memory, Flat-Memory, AWM, ReasoningBank, Mem0, ExpeL, A-MEM, and EvoWeave.</td></tr><tr><td>Main metrics</td><td>AppWorld TGC and SGC, WebArena SR, and separate scores for ALFWorld and ScienceWorld, all on scales from 0 to 100.</td></tr><tr><td>Memory construction</td><td>Online over the evaluation stream. Every run starts from an empty mem- ory, uses the released benchmark task order, and is independent across the 15 backbone and setting combinations. No training or development data is used, and episode level success is judged by a language model module shared by all</td></tr><tr><td>Aggregation</td><td>methods rather than by the benchmark scorer. Matching metrics are averaged equally over backbones. AppWorld Overall TGC weights Normal and Challenge by 168 and 417 tasks, respectively.</td></tr><tr><td>Uncertainty</td><td>Main results show mean ± standard deviation. Ablations show point esti- mates. Cross-method differences are descriptive.</td></tr><tr><td>Auxiliary analyses</td><td>Category performance, graph composition, token and runtime statistics, and operation overlap are analyzed separately from benchmark means.</td></tr></table>

Table 4: Evaluation scope and reporting conventions.

The AppWorld ablation combines Normal and Challenge using the two test counts in Table 4. This arithmetic relation does not by itself establish the identity of the evaluated tasks. Experience is acquired online across test tasks under the protocol of Section 4, which is applied identically to every memory based method; Appendix A.3 records it in full.

## A.2 BENCHMARK AND CATEGORY AGGREGATION

Benchmark means and category summaries are treated as distinct measurements. For ScienceWorld, weighting category scores by their task counts reproduces the main means to numerical precision. This relation does not hold uniformly across all benchmarks.

For AppWorld, all 48 comparisons between main TGC and the corresponding Supervisor score agree to within 0.014 points, so the two views of the same runs are mutually consistent. The category analysis nevertheless keeps the Supervisor score as a distinct measure rather than treating it as interchangeable with main TGC, because the two are aggregated over different populations.

For WebArena, the main table reports success rate over the full benchmark of 812 tasks (Zhou et al., 2024). The category analysis covers the 764 tasks that belong to a single site, namely Shopping, Shopping Admin, GitLab, Map, and Reddit, whose counts appear in Figure 11. The remaining 48 tasks span more than one site and therefore belong to no single site category. Weighting the five category scores by their task counts consequently yields values 1.23 to 1.69 points above the main means: the category weighted mean omits the cross site tasks, on which every configuration scores lower than on single site tasks. For a fixed task split this difference is proportional to the gap between a method’s single site and cross site success rates, and it is smallest for EvoWeave in all three backbones, at most 1.26 points against at least 1.41 points for every baseline. Among the compared methods, EvoWeave therefore has the narrowest gap between tasks confined to one site and tasks that must be composed across sites. ALFWorld means weighted by category counts differ from the main table by approximately −0.060 to +0.074 points. We therefore use category plots to describe variation within categories, without deriving additional benchmark means from them.

AppWorld category memberships in the overlap analysis sum to 183 for Normal and 795 for Challenge, exceeding their respective task totals. Memberships can overlap, so these category counts do not form a partition. Moreover, a global pairwise overlap measure includes tasks from different categories, whereas within category overlap excludes those pairs. The global Jaccard mean is therefore not generally equal to an average of within category Jaccard means, even when weighted by task counts. Figure 4 specifically averages within type overlap with equal weight per type.

## A.3 MEMORY CONSTRUCTION AND LEAKAGE CONTROLS

Section 4 states the protocol; this subsection records it in full. Experience is acquired online: a run visits the test tasks of one evaluation setting in sequence, and whatever the reflector writes after a task is available to the tasks that follow it in the same run. The setting is therefore continual learning over an evaluation stream, and it is the setting in which reusable experience is intended to be useful.

Three properties fix what the protocol can and cannot draw on. First, no task is attempted twice, and the agent receives no ground truth for the task it is solving. Several of the compared methods distinguish successful from unsuccessful episodes: ExpeL contrasts the two to distil insights and AWM induces workflows from episodes it treats as successful. That distinction is produced by a single language model judgement module that reads the trajectory, and every method, including EvoWeave, uses the same module with the same prompt. The official evaluator runs only after an episode terminates and its output reaches no method’s memory. Stored experience therefore consists of observations the agent obtained by acting and ofjudgements formed from those observations, not of labels supplied by the benchmark, and a difference between two methods cannot arise from one of them having been given a more reliable success signal. Second, the task order is the order in which the benchmark releases its tasks; it is not chosen by us and it is identical for every method. Third, runs do not share memory, so a result for one backbone or benchmark is not carried by experience gathered in another.

Every memory based method in Table 1 runs under these same conditions: the same empty start, the same task order, the same absence of benchmark labels in memory, and the same per run isolation. The methods therefore differ only in how experience is represented, selected, and injected, which is the comparison the main results are intended to support.

## A.4 UNCERTAINTY AND CLAIMS

The report of 16 highest means counts metric comparisons; it is not a sample of 21 independent trials or a test of significance. The ± values denote standard deviations, not standard errors or confidence intervals. Paired significance tests require outcomes on the same tasks and an explicit treatment of repeated seeds and shared task templates. The displayed marginal means and standard deviations alone do not determine confidence intervals for differences, so no statistical significance is claimed.

## B RETRIEVAL PROCEDURE AND CAPABILITY MANAGEMENT

## B.1 BOUNDED RETRIEVAL AND CONNECTION PROCEDURE

Algorithm 1 specifies the retrieval flow. The search optimizes the number of historical connector edges locally. Candidate endpoint operations remain part of the returned fragment even when the connector is empty because $v _ { a } = u _ { b }$ . A path that exceeds the limit on visible operation cards is returned with an explicit truncation status or omitted as a complete connection.

With at most $k = 3$ candidates per step, each direct plan dependency induces at most $k ^ { 2 } = 9$ pair searches. If each search expands at most B states, the expansion budget is at most $9 | D _ { t } | B ;$ ; actual runtime also depends on examined adjacency lists and embedding retrieval. This is a budget bound, not a claim that all retrieved fragments have globally optimal joint cost. For a failed bounded search we record whether the reachable subgraph was exhausted or the search stopped at an unexhausted frontier.

Algorithm 1 Retrieval of operation fragments guided by the plan   
Input: Task q, concrete history H , graph G, embedding function ${ \overline { { f } } } ,$ search limits L   
1: $P _ { t } = ( S _ { t } , D _ { t } ) $ Planner $\left( q , H _ { t } \right)$   
2: for each step ${ \bf \Phi } _ { s _ { i } } \in S _ { t }$ do   
3: $C _ { i } \gets { \mathsf { u p } }$ to three highest scoring operations matching $s _ { i }$ above the threshold   
4: Retain edge IDs, endpoints, contracts, references, and versions   
5: end for   
6: $F \gets \emptyset ;$ initialize candidate assignments for shared steps   
7: for each direct dependency $\bar { ( i , j ) } \in D _ { t }$ in a stable order do   
8: Search $v _ { a }  u _ { b }$ for compatible $a \in C _ { i } , b \in C _ { j }$ with bounded BFS   
9: Rank admissible paths by length, candidate similarity, then stable IDs   
10: if a consistent connection is available within the budget then   
11: Add a, its connector, and b to $F ;$ record candidate assignments   
12: else   
13: Retain separate candidates; record gap, conflict, or search truncation   
14: end if   
15: end for   
16: Expose $P _ { t } , F ,$ unmatched candidates, and all unbound prerequisites to the main agent   
17: return operation cards and statuses, subject to the budget for visible context   
Reference parameter Default value   
Operation retrieval cosine threshold / maximum candidates 0.72 / 3   
Cosine threshold for merge candidates / maximum candidates 0.82 / 3   
BFS maximum depth / expansions 8 / 1,000   
Maximum visible operation cards 18   
Skill retrieval cosine threshold / maximum cards 0.35 / 8   
Code retrieval cosine threshold / maximum cards 0.35 / 4  
Table 5: Default parameters of the reference implementation. These specify the retrieval procedure and are distinct from the settings of individual experimental runs.

## B.2 EXECUTION EVIDENCE, CAPABILITIES, AND UPDATES

The main agent can use graph operations corresponding to open, neighbors, search, and path. The response format specifies direction, input requirements, observed or abstract outputs, versions, and truncation status. Retrieving a card and opening a referenced implementation are separate events. If generated code invokes a registered API or stored function whose definition has not been opened, the execution check first opens that definition and requests regenerated code. An unknown API is not created from a guessed name.

Capability counts describe stored code, skill, and role records. They do not by themselves establish automatic decomposition of arbitrary programs or learning of role prompts. The retrieval formulation assumes stable identifiers for plan steps across rounds so that dependencies and candidate assignments remain consistent.

The reflector generates concrete and abstract summaries in one response, with a further merge decision when needed. Concrete event summaries preserve actual observed values and their sources. Abstract summaries parameterize those values and retain distinctions between contracts. Validation checks placeholders against concrete evidence, verifies capability references and versions, and keeps incompatible effects separate. A memory reference indicates that an operation informed the new action; it is not evidence that a stored code template was executed unchanged. Additional retrieval and merge calls can increase the number of model calls beyond the three persistent roles, and bounded connection search does not guarantee global path optimality.

## B.3 ILLUSTRATIVE DEPENDENCY EXAMPLE

Consider a request to split a bill among roommates. Reading the bill produces an amount and currency; retrieving the roommate list produces recipients. Computing a split requires both outputs. A historical path from the operation that reads the bill to the computation does not satisfy the recipient prerequisite. If another fragment supplies the roommate list, the two fragments must still refer to compatible entities and the same current task instance. Finally, requesting payments and sending money have different effects and remain distinct operations. This example illustrates contract checks and AND dependencies; it is not presented as an observed benchmark trajectory.

## C PER BACKBONE CATEGORY PERFORMANCE

Figures 5 to 7 present 15 radar plots, with five settings for each backbone. Every plot includes all eight methods and all categories. A local linear scale maps each category’s scores to radial positions, using the same bounds for all methods on that spoke.

Local axis ranges. For a spoke with minimum a and maximum b over the eight methods, let $p = \operatorname* { m a x } ( 0 . 1 5 ( \breve { b } - a ) , 0 . 5 )$ . We round $a - p$ downward and $b + p$ upward, using increments of 1 when $b - a \geq 3$ and 0.5 otherwise, and clip the resulting bounds $L , U$ to the valid score range from 0 to 100. The radius is $r = ( x - L ) / ( U - \bar { L } )$ . All methods on that spoke share these bounds, printed as $[ L , U ]$ beside its label. The center denotes $L ,$ the outside circle denotes $U ,$ and intermediate rings divide this interval into quarters. Ranges are computed separately for each plotted population. Consequently, compare methods within a spoke; equal radii across spokes or backbones need not represent equal scores, and polygon area is not an aggregate performance measure.

Label key. Files denotes File\_System, Notes denotes SimpleNote, Superv. denotes Supervisor, Shop denotes Shopping, and Admin in WebArena denotes Shopping Admin. ALFWorld labels are Pick and Place, Examine in Light, Clean and Place, Heat and Place, Cool and Place, and Use Two Objects. The ScienceWorld mapping is S1: Changes in the state of matter; S2: Temperature and Melting Point; S3: Electricity and Electrical Conductivity; S4: Living and Non-living Things; S5: Plant growth; S6: Chemical mixing; S7: Animal Lifespan; S8: life cycle; S9: Physics Mechanics; S10: Mendelian genetics. Supervisor is displayed as a distinct score, with its relationship to main TGC discussed in Appendix A.

## D ABLATION RESULTS FOR EACH BACKBONE

Tables 6 to 8 preserve the complete ablation values. For each configuration and backbone b, the reported Overall TGC is consistent with

$$
\mathrm { O v e r a l l } _ { b } = { \frac { 1 6 8 { \mathrm { N o r m a l } } _ { b } + 4 1 7 { \mathrm { C h a l l e n g e } } _ { b } } { 5 8 5 } } .\tag{4}
$$

The main text table then takes an equal mean over the three backbones. These ablations report point estimates. F is the full system, that is D with active graph retrieval; E is C with active retrieval but without directed historical connections. The highest value in each column is shown in bold. The configurations share the memory construction protocol of Appendix A.3, the same candidate pool, and the same capability library, as described in Section 4.

Table 6: DeepSeek-V3.2 AppWorld ablation.
<table><tr><td>Configuration</td><td>Normal TGC</td><td>Challenge TGC</td><td>Overall TGC</td></tr><tr><td>A: No memory</td><td>47.02</td><td>23.02</td><td>29.91</td></tr><tr><td>B: Whole workflow RAG</td><td>49.05</td><td>23.02</td><td>30.50</td></tr><tr><td>C: Operation cards</td><td>49.09</td><td>24.36</td><td>31.46</td></tr><tr><td>D: C + directed connections</td><td>52.75</td><td>25.21</td><td>33.12</td></tr><tr><td>E: C + active retrieval</td><td>52.20</td><td>25.67</td><td>33.29</td></tr><tr><td>F: EvoWeave (full)</td><td>54.05</td><td>27.00</td><td>34.77</td></tr></table>

## E GRAPH STATISTICS BY EVALUATION SETTING

The macro means in Table 3 weight each setting equally, including Normal and Challenge as separate settings. The mean state and operation counts are 27.0 and 157.6. Mean counts of code, skill, and role nodes are 185.2, 82.2, and 6.8; the corresponding mean counts of capability links are 157.6, 362.0, and 99.4. These are arithmetic summaries of five graph snapshots. They do not identify average paths, motifs, reachability, degree distributions, or provenance quality. Such analyses require exported adjacency lists and the exact snapshot associated with each run.

![](images/0a3908cfe1f6471eec184426ecccfd856195cea7cf2b787bf01639590c3a8f46.jpg)

![](images/988c12957ea1d3f7605aec75d2c1bf9f4775b1987360442b50c676d89c8d5137.jpg)

![](images/e7466b47c7755e4340925235d605ad0d825bf14d348fa0a84e57d3e771bdb70b.jpg)  
Figure 5: DeepSeek-V3.2 category performance across five settings. Each spoke uses the local range printed beside its category. All eight methods are retained.

![](images/e25fdd80f550a9ba9445f33f1dd72a7bcbe8d81d388e117d537d633df0553a35.jpg)

![](images/f20cba3d5142dfce6daa12545281091917daa50cc275a062624fcfbf617565c3.jpg)  
Figure 6: GPT-4.1-mini category performance across the five settings. These curves are inputs from individual backbones to the equal averages over backbones in Figure 3.

![](images/fcdb94ef6d67f4e08ecde74f2a7d2e9e35fafa6953438d2a4b721eaba07c417f.jpg)

![](images/69953a1bb703891090ef68dc2c85ff50c3c860f578067bcbe667606ebbda128d.jpg)

![](images/a99b3b160caac59ea1864b13dd456d74b5621826b5f23dc0d1e5207bfc9c2dbd.jpg)

![](images/9ab0a821379c1f273eedfdeb7084d378025a28dd3a4624c4d7b156a5ef17cf1b.jpg)  
Figure 7: Qwen3.8-27B category performance across the five settings. Each spoke uses the local score range printed beside its category.

Table 7: GPT-4.1-mini AppWorld ablation.
<table><tr><td>Configuration</td><td>Normal TGC</td><td>Challenge TGC</td><td>Overall TGC</td></tr><tr><td>A: No memory</td><td>35.71</td><td>16.98</td><td>22.36</td></tr><tr><td>B: Whole workflow RAG</td><td>36.67</td><td>16.02</td><td>21.95</td></tr><tr><td>C: Operation cards</td><td>38.50</td><td>17.81</td><td>23.75</td></tr><tr><td>D: C + directed connections</td><td>39.84</td><td>19.80</td><td>25.56</td></tr><tr><td>E: C + active retrieval</td><td>39.11</td><td>18.86</td><td>24.68</td></tr><tr><td>F: EvoWeave (full)</td><td>40.60</td><td>21.01</td><td>26.64</td></tr></table>

Table 8: Qwen3.8-27B AppWorld ablation.
<table><tr><td>Configuration</td><td>Normal TGC</td><td>Challenge TGC</td><td>Overall TGC</td></tr><tr><td>A: No memory</td><td>52.02</td><td>29.02</td><td>35.63</td></tr><tr><td>B: Whole workflow RAG</td><td>54.05</td><td>30.02</td><td>36.92</td></tr><tr><td>C: Operation cards</td><td>56.20</td><td>31.99</td><td>38.94</td></tr><tr><td>D: C + directed connections</td><td>59.63</td><td>34.51</td><td>41.72</td></tr><tr><td>E: C + active retrieval</td><td>58.40</td><td>33.30</td><td>40.51</td></tr><tr><td>F: EvoWeave (full)</td><td>61.07</td><td>35.97</td><td>43.18</td></tr></table>

Table 9: Reported graph statistics for each setting. Ops. denotes directed operation edges. Links denotes references from capabilities to operations, not state transition edges.
<table><tr><td>Setting</td><td rowspan="2">States Ops.</td><td rowspan="2"></td><td colspan="2">Code</td><td colspan="2">Skills</td><td colspan="2">Roles</td></tr><tr><td></td><td></td><td>Nodes Links</td><td></td><td>Nodes Links</td><td>Nodes Links</td><td></td></tr><tr><td>AppWorld Normal</td><td>37</td><td>173</td><td>405</td><td>173</td><td>56</td><td>142</td><td>9</td><td>117</td></tr><tr><td>AppWorld Challenge</td><td>51</td><td>431</td><td>467</td><td>431</td><td>139</td><td>825</td><td>9</td><td>216</td></tr><tr><td>WebArena</td><td>18</td><td>67</td><td>17</td><td>67</td><td>180</td><td>562</td><td>5</td><td>61</td></tr><tr><td>ALFWorld</td><td>9</td><td>14</td><td>11</td><td>14</td><td>6</td><td>27</td><td>1</td><td>8</td></tr><tr><td>ScienceWorld</td><td>20</td><td>103</td><td>26</td><td>103</td><td>30</td><td>254</td><td>10</td><td>95</td></tr></table>

Figure 8 visualizes the composition of each snapshot with two horizontal bars. The experience layer combines state nodes and operation edges. The tool layer, termed the capability layer in the method, contains code, skill, and role nodes together with their links to operations. Each bar is normalized separately: a component with count $n _ { k }$ has share $1 0 0 n _ { k } / \sum _ { j } n _ { j }$ , where the sum includes the displayed components of that layer. The labels retain absolute counts so that equal bar lengths do not imply equal graph sizes. These shares describe the composition of recorded graph elements, rather than memory usage or execution frequency.

## F TOKEN USAGE AND RUNTIME METRICS

For a memory baseline injecting $I _ { b }$ tokens per question, the injection saving is 100(1 − $I _ { \mathrm { E v o W e a v e } } / I _ { b } )$ . This yields 76.60% for Flat-Memory, 69.05% for AWM, and 48.15% for Reasoning-Bank. The reduction in total input relative to No-Memory is $1 0 0 ( 1 - 1 0 2 6 6 5 . 4 6 / 1 0 3 6 9 0 ) = 0 . 9 9 \%$ Injected memory tokens are zero for No-Memory, so a relative injection saving against that baseline is undefined. Lower KV reuse and more seconds per round qualify any interpretation of token savings as a runtime improvement.

Mean total latency cannot be recovered by multiplying average round count by average seconds per round: the two quantities can covary across tasks and exclude different sources of overhead. The summaries also do not establish a common accounting of reflection, validation, active search, retries, and offline construction. API monetary cost and local GPU usage require different accounting.

AppWorld Challenge  
![](images/61aab948381b3f9ceee19a2e4b5acf8c554add6a3e8a3e39bdd9b114de42defa.jpg)

![](images/edc4fabb84e9233c7c75d657d11fffaccb174335c2d8038f3bc5fda91fc879a3.jpg)

![](images/7222d1b7c41000b239bbf9e038b5bd083ba90024d3fe373b4199ff72f1706add.jpg)

![](images/eff17f1c778bfcf630c8f431f62f69833846e8e5affef90d41465fa7530b2fc3.jpg)

![](images/b386527be73d939f136df44783e460046ccc5268a67e55066a61456af1e14398.jpg)  
Figure 8: Graph composition for the five evaluation settings. Each panel shows the experience layer and tool layer, each normalized to 100%. Labels report counts and shares within the corresponding layer; totals appear beside the bars. Solid fills indicate nodes and hatched fills indicate operation edges or capability links. The keys below each panel give all six tool components, including those with very small shares. Source counts are retained in Table 9.

Table 10: Efficiency metrics. Injected, Input, and Output denote token counts per question. Rounds and seconds per round are averages. The summaries do not fully specify the model calls included in each token metric.
<table><tr><td>Method</td><td>Injected</td><td>Input KV reuse (%)</td><td>Output</td><td>Rounds</td><td>s/round</td></tr><tr><td>No-Memory</td><td>0.00</td><td>103,690.00</td><td>70 4,925.00</td><td>18.52</td><td>9.52</td></tr><tr><td>Flat-Memory</td><td>26,300.80</td><td>127,710.80</td><td>784,829.20</td><td>17.96</td><td>9.79</td></tr><tr><td>AWM</td><td>19,890.00</td><td>114,140.00</td><td>784,479.80</td><td>16.78</td><td>9.66</td></tr><tr><td>ReasoningBank</td><td>11,872.00</td><td>103,192.00</td><td>774,341.00</td><td>16.32</td><td>9.54</td></tr><tr><td>Mem0</td><td>8,500.00</td><td>113,140.00</td><td>764,974.00</td><td>18.66</td><td>9.51</td></tr><tr><td>ExpeL</td><td>28,011.60</td><td>123,961.60</td><td>784,564.20</td><td>17.02</td><td>9.82</td></tr><tr><td>A-MEM</td><td>16,490.00</td><td>118,160.00</td><td>704,839.20</td><td>18.04</td><td>9.88</td></tr><tr><td>EvoWeave</td><td> $^ { 6 , }$ </td><td>155.46 102,665.46</td><td>454,574.00</td><td>15.94</td><td>10.44</td></tr></table>

These metrics therefore characterize context and execution tradeoffs without establishing a ranking by total latency or monetary cost.

## G CATEGORY OPERATION OVERLAP AND REFERENCING

Let $O _ { i }$ denote the operation set associated with task i and $\mathcal { T } _ { d , c }$ the tasks of type c in setting d. Within a type, overlap is summarized by the mean pairwise Jaccard similarity

$$
J _ { d , c } = \frac { 1 0 0 } { \left( { } ^ { | T _ { d , c } | } \right) } \sum _ { i < j , ~ i , j \in { \mathcal { T } } _ { d , c } } \frac { | O _ { i } \cap O _ { j } | } { | O _ { i } \cup O _ { j } | } ,\tag{5}
$$

using benchmark specific operation proxies: gold database model sets for AppWorld, evaluation operators for WebArena, high level actions for ALFWorld, and action verbs for ScienceWorld. These sets omit ordering, arguments, and repeat counts. Overlap therefore describes shared vocabulary, with executable reuse additionally requiring compatible inputs and preconditions.

For Figure 4, we average the type specific percentages with equal weight. With $K _ { d }$ types and reference percentage $R _ { d , c } ,$

$$
\overline { { J } } _ { d } = \frac { 1 } { K _ { d } } \sum _ { c = 1 } ^ { K _ { d } } J _ { d , c } , \qquad \overline { { R } } _ { d } = \frac { 1 } { K _ { d } } \sum _ { c = 1 } ^ { K _ { d } } R _ { d , c } .\tag{6}
$$

The type counts are 9, 9, 5, 6, and 10 for AppWorld Normal, AppWorld Challenge, WebArena, ALFWorld, and ScienceWorld. Overlap is computed within each type, excluding cross type pairs. The averages give equal weight to types, without weighting by task counts or averaging across backbones. Memory reference percentages are analyzed separately from operation overlap. Their event definition and denominator are not sufficiently specified to interpret them as execution success rates.

Figures 9 to 13 show all 39 types and their task counts. AppWorld memberships can overlap. ScienceWorld’s auxiliary sample covers 90 tasks from 30 templates with three instances each, grouped into ten topics; complete benchmark coverage is not established. Perfect overlap in small types should be interpreted in light of the size of the underlying vocabulary and of the sample.

![](images/d84cf71e9d3de8fd5b63a187a8e5f20e6f6ef68fa12bf61b30946d74fa79154f.jpg)  
Figure 9: AppWorld Normal: category overlap and reported memory referencing. Categories reflect proxies based on database models; their task counts need not partition the overall task set.

![](images/88ea5ed1c26264af909ef867f1a7e525dc60ac07775c4665ad3380c919280e58.jpg)  
Figure 10: AppWorld Challenge: category percentages, with task counts shown alongside labels. The nine types receive equal weight in Figure 4. Task membership can overlap across categories.

![](images/3b8f9dc480b27e233b1e4df15fd71d42cf24a378e3fa9046346a81c48927022a.jpg)  
Figure 11: WebArena: overlap of evaluation operators and reported memory referencing. The five single site categories cover 764 of the 812 tasks; the remaining 48 span more than one site and are included in the main table but not in any category, as explained in Appendix A.

![](images/e46b9f7e9020fa934cf4c663b227b553d75405deb2477e71ec73900280c2d56a.jpg)  
Figure 12: ALFWorld: overlap of high level action sets and reported memory referencing. Repeated action vocabulary does not itself establish matching objects, preconditions, or successful execution.

![](images/9526546310528bb89497fcee1094b35db0a40ff806d81609fa99a02867b09fed.jpg)  
Figure 13: ScienceWorld: overlap of gold verb sets and reported memory referencing across 90 tasks. S1 to S10 follow the topic key in Appendix C.

## H FURTHER EVALUATION DIRECTIONS

The existing results motivate, but do not replace, stronger mechanism tests. A token matched comparison of C and D should preserve retrieved operation text and capability references while controlling the number of visible tokens; otherwise, the graph variant can benefit simply from more context. F should receive the same budget for searching descriptions and opening bodies as E, with adjacency and path information as the intended difference. A comparison with fixed capabilities should freeze API definitions and code functions across configurations, isolating changes to experience organization.

A composition test should distinguish no overlap, overlap in one step, partial overlap in several steps, novel combinations of known operations, and requirements with AND joins. The task split must be determined without selecting on final success. Retrieval sensitivity should report both behavior when there are no matches and the effect of thresholds, candidate counts, path depth, and expansion limits. Candidate conflicts, truncated searches, and missing prerequisites should be logged rather than absorbed into a generic failure rate.

A frozen memory condition would complement the online protocol of Appendix A.3 by building memory on a separate split and then evaluating with it fixed, which isolates transfer from continual accumulation. Reporting accuracy as a function of position in the evaluation stream would characterize how quickly useful experience is acquired. Human review of a sample of reused operations could measure whether a reference was relevant, correctly parameterized, and actually adopted. These directions extend the present analyses; they are not additional experimental results.