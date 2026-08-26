# Adaptive Influence Graphs for Failure Attribution in Multi-Agent Systems

Yarden Bakish<sup>1</sup>\*<sup>†</sup> Amir Dudai<sup>2</sup>\*<sup>‡</sup> Roy Ganz<sup>2</sup> Oren Nuriel<sup>2</sup> Elad Ben Avraham<sup>2</sup> Mor Shpigel Nacson<sup>2</sup> Ron Litman<sup>2</sup> <sup>1</sup>Tel Aviv University <sup>2</sup>AWS Agentic AI

## Abstract

Multi-agent LLM systems are increasingly deployed in real-world applications, where failures can be costly and difficult to localize. Despite growing efforts to automate failure attribution, diagnosing failed runs still largely relies on human engineers. Yet engineers rarely debug complex systems by reading raw logs end to end. Instead, observability tools organize traces around components, actions, and dependencies to support targeted navigation. We hypothesize that modern LLMs can benefit from the same paradigm. To test this hypothesis, we introduce Adaptive Influence Graphs (AIGs), a two-stage agentic framework that first transforms a failed trace into a structured graph and then navigates it to identify the critical error. Across multiple models, we show that richer trace representations consistently improve failure attribution, with adaptive graph construction and agent-directed traversal yielding the strongest results. AIGs establish a new state of the art on Who&When, the standard benchmark for multi-agent failure attribution. This affirms our hypothesis that attribution depends not only on the diagnosing model, but also on how the trace is represented and explored.

## 1 Introduction

Multi-agent systems are increasingly deployed to solve complex real-world tasks, with executions spanning specialized agents, tools, and intermediate artifacts. When these systems fail, diagnosis still largely falls to human engineers, who must inspect long, interleaved logs to determine where the execution first went wrong. This process becomes increasingly costly as agentic systems grow more complex and operate over longer horizons, making automated diagnosis essential for scalable development and deployment. Who&When formalized this problem as failure attribution: identifying the failure-responsible agent and the decisive error step in a failed execution (Zhang et al., 2025b; Wang et al., 2026a). We illustrate the task in the upper panel of Figure 1.

![](images/71a4a27953f135bf1c74628a1dd8e894503105c984973431b129a96d83967c2b.jpg)  
Figure 1: The same trace yields different attributions depending on its representation. The raw log highlights the visible symptom (step 3); the Adaptive Influence Graph traces the error back to its origin (step 2).

Prior work on multi-agent failure attribution has primarily focused on improving the reasoning procedure applied to an execution trace, for example through multi-stage diagnosis pipelines (e.g., Zhang et al., 2025b; Zhu et al., 2025; West et al., 2025). A complementary line of work restructures the trace before attribution using structured records or dependency representations (Ge et al., 2025; Ma et al., 2025; Li et al., 2026b; Rafi et al., 2026). CHIEF (Wang et al., 2026b) further reconstructs flat logs as hierarchical causal graphs, organizing attribution through a predefined hierarchy and edge schema. Thus, existing approaches either reason over a largely fixed trace interface or restructure it using a predefined representation.

Human engineers rarely debug complex systems by reading raw logs end to end. Instead, they rely on agent observability systems<sup>1</sup> that organize executions around components, actions, and dependencies, enabling targeted traversal of the trace. As modern agentic LLMs become increasingly capable of planning, tool use, and iterative reasoning, we ask whether they can benefit from the same paradigm. We hypothesize that failure attribution is partly an interface-design problem: it depends not only on the diagnosing model, but also on how the trace is represented and traversed. Figure 1 illustrates this hypothesis: on the same failed execution, the same reader misattributes the failure from the raw log but identifies the correct critical step from the structured graph.

To test this hypothesis, we separate failure attribution into two components: how the trace is represented and how that representation is read. We first evaluate a four-level representation ladder while holding the reader fixed as a single call over the complete serialized interface (Figure 2): (i) the raw log; (ii) a structured log that groups consecutive steps from the same agent into segments; (iii) an Influence Graph that summarizes what each segment received and produced and adds edges tracing how earlier outputs were inherited downstream; and (iv) an Adaptive Influence Graph, whose agentic builder chooses the action boundaries and influence structure separately for each failed trace. A critic–refiner audits the resulting adaptive graph for structural and semantic consistency. We then replace the single-call reader with an agentic reader. The graph provides a high-level map of the execution, while the underlying log is exposed only on demand. The reader iteratively chooses which steps to inspect, follows influence links, and verifies the graph’s claims before predicting the critical agent and step. This design isolates the benefit of richer trace representations from the additional benefit of agent-directed traversal.

We evaluate this design on both partitions of Who&When, using the Algorithm-Generated partition for controlled ablations. There, with Opus-

![](images/45a4e4cfc193ea3d2484a9064d251fcff6d5657b849523548548c0f93b25ecc1.jpg)  
Figure 2: Step accuracy improves with richer trace representations, and improves further when the adaptive graph is traversed by an agentic reader (Algorithm-Generated Who&When, Opus-5).

5, exact step accuracy rises from 46.40% on raw logs to 48.80% with structured logs and 52.00% with influence graphs; adaptive graph construction paired with agent-directed traversal raises it further to 55.20% (Figure 2). This establishes a new state of the art on the Algorithm-Generated partition, surpassing the previous best reported result of 51.60%, while also reaching 71.20% agent accuracy and localizing 84.00% of failures within a ±3-step window. We further demonstrate the robustness of AIGs across models by evaluating them with DS-V3.2, Sonnet-4, and GPT-5.6-sol in addition to Opus-5. Our analysis shows that the gains concentrate on failures later in the trace, where AIGs substantially improve exact step localization over raw-log reading. Together, these findings provide empirical support for our hypothesis that failure attribution depends not only on the diagnosing model, but also on how the trace is represented and traversed.

Contributions. Failure attribution as interface design. We systematically demonstrate that multiagent failure attribution depends not only on the diagnosing model, but also on how the execution trace is represented and traversed. A controlled representation ladder isolates the benefit of progressively richer structure, while evaluations across multiple models show that the effect is not specific to a single reasoner. Adaptive Influence Graphs. We introduce AIGs, a two-stage agentic framework that adaptively constructs an influence graph and traverses it while verifying evidence against the original log. The complete system establishes a new state of the art on the Algorithm-Generated Who&When partition.

## 2 Related Work

Failure attribution in multi-agent systems. A growing line of work studies how to diagnose failed executions by identifying the agent and step that first caused the error, where step-level accu racy remains limited (Wang et al., 2026a). The Who&When benchmark (Zhang et al., 2025b) formalizes the setting, and its prompted baselines differ only in how the log is sliced. Subsequent methods improve the inference procedure applied to that transcript—scaffolding it with abductive prompts (West et al., 2025), distributing judgment across a judge and criterion-specific evalu ators (Zhu et al., 2025), compressing context by positional distance from a candidate step and voting across analyst personas (Banerjee et al., 2025), retrieving condensed error knowledge from previ ously annotated traces (Yu et al., 2025), verifying predefined error hypotheses against the full tra jectory before localizing responsible agents (Qiao et al., 2026), or using model-based attribution through prefill-stage signals (Liu et al., 2026) and synthetic-failure fine-tuning (Zhang et al., 2025a). In each case the diagnosing model still receives the trace as a flat transcript, modified at most by step indexing, a fixed slicing policy, or distancebased truncation. Recent work broadens the task itself, releasing the inputs and tool logs each step saw (Chen et al., 2026b), allowing several defensible attributions per failure (In et al., 2026), introducing a cross-domain benchmark of first unrecoverable failures (Barke et al., 2026), or moving failure analysis online to detect decisive errors during execution (Zhang et al., 2026). We work in Who&When’s partially observable setting but focus on a different axis: not only how a model reasons over a trace, but how that trace is represented before reasoning begins. Our method is also training-free: both the builder and the reader are prompted frontier models.

Restructuring the trace. A second line of work does restructure the trace before attribution. FAMAS (Ge et al., 2025) abstracts each log record into a canonical ⟨agent, action, state⟩ triple and attributes via spectrum analysis; CDC-MAS (Ma et al., 2025) derives a data-dependency graph by a deterministic rule and recovers a causal DAG by constraint-based discovery; and StepFinder (Zhu et al., 2026), arguing that noisy logs interfere with LLM reasoning, encodes each step into a feature vector and trains a small sequential model, removing the LLM from attribution altogether. Closest to our work, prior methods construct structured dependency representations and search them for failure sources (Wang et al., 2026b; Li et al., 2026b; Rafi et al., 2026). These systems and related agentdebugging work establish that restructuring the trace helps (Chen et al., 2026a; Li et al., 2026a; Kang et al., 2026). Across these methods, however, the construction policy is fixed a priori through predefined record types, node boundaries, hierarchies, or relation vocabularies, whereas our builder jointly chooses node granularity and edge topology per trace. More broadly, LLM performance is sensitive to the position and spacing of relevant evidence, prompt formatting, and graph encoding (Liu et al., 2023; Tian et al., 2025; Sclar et al., 2024; Fatemi et al., 2023).

Trace observability. Our approach is inspired by how humans debug agent systems in practice. Rather than reading a raw log top to bottom, they use observability platforms such as Arize Phoenix (Arize AI, 2023), built on OpenTelemetry (Cloud Native Computing Foundation, 2019), which ingest an execution trace and present it as a hierarchical, filterable object: spans nested by call structure, attributes to group and search on, and a tree to be expanded on demand. Adaptive Influence Graphs carry that design from a human consumer to an automated one: the builder acts as the instrumentation layer, turning the trace into a navigable graph, and the reader drills into its nodes and edges much as an engineer drills into spans.

## 3 Method

A trajectory is a sequence of T steps

$$
( a _ { 1 } , s _ { 1 } ) , \ldots , ( a _ { T } , s _ { T } ) ,
$$

where $a _ { t }$ is the agent acting at step t and $s _ { t }$ is the emitted log content; the same agent may act at multiple steps. Given the original task query q and a failed final answer, the system must identify the agent and step $( a ^ { \star } , t ^ { \star } )$ that first derailed the trajectory from solving the task correctly. We evaluate exact agent accuracy and exact step accuracy.

We decompose attribution into two stages: a builder (section 3.1) constructs a structured representation G from the query and trajectory, and a reader (section 3.2) uses a set of tools to produce an attribution (ˆa, t<sup>ˆ</sup>), as shown in Figure 3. The stages are coupled: the reader navigates G and accesses the underlying log entries referenced by its nodes on demand. Accordingly, G is designed as an interface rather than a summary, with inheritance edges for backtracking and nodes grounded in the original log.

![](images/9e3be8d5dbc3eec1d521d339fccff312857c3f49b464c9a27ffd3f38f308fd73.jpg)  
Figure 3: Build-read pipeline for Adaptive Influence Graphs (AIGs). The builder uses log-inspection and graphconstruction tools to produce G; a critic–refiner audits its structural and semantic validity; and a reader uses graph and log-inspection tools to produce the final attribution.

## 3.1 Build: Constructing the Graph

We evaluate builders from deterministic raw-log mapping to fully agentic graph construction.

Builder 1: Structured logs. The first structured builder is deterministic and LLM-free. It groups consecutive steps from the same agent into one node, yielding a chain of nodes $v _ { 1 } , \ldots , v _ { K }$ in execution order, each node holding the verbatim steps of one agent’s uninterrupted turn. This changes the surface form while preserving the evidence, so gains isolate the value of explicit structure rather than content.

Builder 2: Influence Graphs. The second builder uses an LLM to enrich this chain into an Influence Graph (IG), adding node fields and inheritance edges in a single pass while leaving the node partition untouched. Each node receives a four-field abstraction $\alpha ( v _ { k } ) =$ ⟨summary, input, output, authorship⟩—what the node did end-to-end, the input it received, the output it generated, and the node’s role in producing that output—in addition to its raw content. The LLM also adds a directed edge from an earlier node to a later one when the later node reused part of the earlier node’s work in a way tied to what went wrong. Each edge is annotated with two fields: inherited, naming the element carried over, and $e f f e c t ,$ describing how it went wrong downstream. An edge is drawn only under inheritance, so a node that starts fresh induces none; the edge set is sparse and may be empty because unsupported causal links can mislead the reader. See section E for an example.

Builder 3: Adaptive Influence Graph. The Adaptive Influence Graph (AIG) generalizes the IG by replacing its fixed, single-pass construction procedure with an agentic builder. Both representations use step-grounded nodes and annotated inheritance edges. However, whereas the IG enriches a node partition fixed in advance, the AIG builder jointly determines the node boundaries and roles, fills their role-specific abstractions, and constructs the inheritance topology for each trace. The topology is therefore not fixed a priori and may connect non-adjacent actions, subject to the graph validity constraints described below.

The builder starts from an empty graph and interacts with the trace through two tool families. Inspection tools $\mathcal { T } _ { \mathrm { l o g } }$ expose selected parts of the trajectory without modifying the graph, while construction tools $\mathcal { T } _ { \mathrm { b u i l d } }$ create, modify, and delete nodes and inheritance edges. The builder interleaves inspection and construction until it submits the completed graph. App. A lists the full tool set.

The builder outputs a directed graph ${ \cal G } =$ $( V , E )$ , where V is the set of nodes and E is the set of directed edges. Every node is grounded in one or more raw-log steps and assigned a type from a fixed vocabulary. The node type determines which fields the builder must provide.

In the single-type variant, every node uses the same four fields as the IG—summary, input, output, and authorship—and inheritance edges use the same annotations. This preserves the IG schema while allowing the agentic builder to choose the node boundaries and graph topology.

The agentic framework benefits further from enlarging this output space to typed, grounded graphs. We extend the fixed node vocabulary with additional roles, each governing a distinct node type; per-role descriptions appear in App. B.

Critic–Refiner Loop. Admissible graphs must satisfy a validity predicate Φ(G), including connectivity, acyclicity, one root, and one sink. After submission, a critic audits the graph for at most R rounds, checking structural violations programmatically against Φ and semantic violations against the log, such as a node that is unfaithful to its referenced steps. A refiner applies minimal edits to resolve any violations, and the loop stops once none remain or the round budget is exhausted. See section F for an example.

## 3.2 Read: Attributing Failure from the Representation

We evaluate two readers that differ in how they consume the representation.

Single-call reading. The single-call reader receives the serialized representation along with the raw log and emits one attribution.

Agentic graph reading. The agentic reader retrieves the finalized graph G, then selectively inspects the underlying trajectory τ using its inspection tools $\mathcal { T } _ { \mathrm { l o g } } .$ . Starting from a candidate node from the graph, it works backward over incoming inheritance edges. We instruct it to move blame from the current node to an edge’s source only when the raw log confirms that the current node’s output exhibits the downstream error described by the edge’s effect annotation. This process continues until no incoming inheritance edge explains the observed failure. The reader then resolves the selected node to an exact agent and step using its step references and the raw log, and submits (ˆa, t<sup>ˆ</sup>).

## 3.3 Implementation Details

Each stage is a separate agent built with the Strands Agents SDK v1.37 (Strands Agents, 2025), where tools are Python functions whose schemas the SDK derives and drives, and the graph is shared mutable state that construction tools edit in place and read\_graph() serializes on demand.

We run the full AIG pipeline with the following backbones—Opus-5, Sonnet-4, and DeepSeek-V3.2 (DS-V3.2)—and repeat the representation ablations under each, holding the model fixed within a ladder. Reported results are Opus-5 unless stated otherwise. Models are accessed via the Amazon Bedrock Converse API (Sonnet-4 and DS-V3.2 at temperature 0.6; Opus-5 and GPT-5.6-sol at their defaults, which cannot be set), with no task-specific fine-tuning. The critic–refiner loop runs at most R = 3 rounds, exiting early on an empty violation set. A stage ends at submit.

## 4 Results

We evaluate Adaptive Influence Graphs on both partitions of Who&When (Zhang et al., 2025b). We first report benchmark performance and consistency across models, then use a controlled interface ladder on the Algorithm-Generated partition to isolate the contributions of trace representation and agent-directed traversal. Section 5 examines where the gains occur and how the builder and reader use the resulting interface.

## 4.1 Who&When Benchmark Results

Algorithm-Generated partition. AIGs with Opus-5 achieve 55.20% step accuracy and 71.20% agent accuracy without the ground-truth answer at inference time. This establishes a new state of the art on the Algorithm-Generated partition, improving over the previous best step accuracy of 51.60% by 3.6 percentage points (Table 1).

Consistency across models. The result is not tied to Opus-5. AIGs reach 54.76% step accuracy with GPT-5.6-sol, 53.97% with Sonnet-4, and 53.17% with DS-V3.2; all three surpass the previous best result. The advantage holds in matched-model comparisons: AIGs outperform CHIEF by 9.96 points with Opus-5 and 7.57 points with DS-V3.2, and RAFFLES by 2.37 points with Sonnet-4. It therefore persists across model families and matchedmodel comparisons.

Hand-Crafted partition. With Opus-5, AIGs reach 29.31% step accuracy on the Hand-Crafted partition, matching A2P and CHIEF for the highest result without ground-truth access. Their 63.79% agent accuracy remains below CHIEF’s 72.41%, showing that the step-level advantage does not extend to agent accuracy on this partition.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Model</td><td colspan="2">Algorithm-Generated Dataset</td><td colspan="2">Hand-Crafted Dataset</td></tr><tr><td>Step Acc. ↑</td><td>Agent Acc. ↑</td><td>Step Acc. ↑</td><td>Agent Acc. ↑</td></tr><tr><td>All-at-Once</td><td>Opus-5</td><td>46.40</td><td>67.20</td><td>22.41</td><td>50.00</td></tr><tr><td>Step-by-Step</td><td>Opus-5</td><td>42.40</td><td>54.40</td><td>25.86</td><td>58.62</td></tr><tr><td>CDC-MAS (Ma et al., 2025)</td><td>GPT-40</td><td>36.00</td><td>48.50</td><td>18.20</td><td>56.80</td></tr><tr><td>AgenTracer-8B (Zhang et al., 2025a)</td><td>Qwen3-8B</td><td>37.30</td><td>63.73</td><td>20.68</td><td>63.82</td></tr><tr><td>CORRECT (Yu et al., 2025)</td><td>GPT-5</td><td>38.10</td><td></td><td>17.20</td><td></td></tr><tr><td>CHIEF† (Wang et al., 2026b)</td><td>Opus-5</td><td>45.24</td><td>64.29</td><td>18.97</td><td>51.72</td></tr><tr><td>CHIEF (Wang et al., 2026b)</td><td>DS-V3.2</td><td>45.60</td><td>68.80</td><td>29.31</td><td>72.41</td></tr><tr><td>A2P (West et al., 2025)</td><td>GPT-40</td><td>47.50</td><td>65.40</td><td>29.31</td><td>58.62</td></tr><tr><td>RAFFLES (Zhu et al., 2025)</td><td>Llama-3.3-70B</td><td>43.65</td><td>一</td><td>20.69</td><td>一</td></tr><tr><td>RAFFLES (Zhu et al., 2025)</td><td>Sonnet-4</td><td>51.60</td><td>一</td><td>22.41</td><td>一</td></tr><tr><td>Adaptive Influence Graph</td><td>DS-V3.2</td><td>53.17</td><td>65.08</td><td>24.14</td><td>60.34</td></tr><tr><td>Adaptive Influence Graph</td><td>Sonnet-4</td><td>53.97</td><td>67.46</td><td>29.31</td><td>53.45</td></tr><tr><td>Adaptive Influence Graph</td><td>GPT-5.6-sol</td><td>54.76</td><td>67.46</td><td>24.14</td><td>55.17</td></tr><tr><td>Adaptive Influence Graph</td><td>Opus-5</td><td>55.20</td><td>71.20</td><td>29.31</td><td>63.79</td></tr></table>

Table 1: Who&When benchmark. Rows use no ground-truth access at inference time. One Algorithm-Generated log is content-filtered by Opus-5; therefore, for this model we use n=125. <sup>†</sup>Run using the authors’ released code. Our methods are shaded.

<table><tr><td>Representation</td><td></td><td>Step ↑ Agent ↑</td></tr><tr><td>Raw log</td><td>46.40</td><td>67.20</td></tr><tr><td>Structured log</td><td>48.80</td><td>63.20</td></tr><tr><td>Influence Graph</td><td>52.00</td><td>70.40</td></tr><tr><td>Adaptive Influence GraphLLM reader</td><td>50.40</td><td>64.80</td></tr><tr><td>Adaptive Influence Graph</td><td>55.20</td><td>71.20</td></tr></table>

Table 2: Opus-5 representation ablations on the Algorithm-Generated Who&When partition (no groundtruth access). Our AIGs are in bold.

## 4.2 The Interface Ladder

We next isolate how much of the improvement comes from the trace representation and how much requires agent-directed traversal. On the Algorithm-Generated partition, we hold Opus-5 fixed and progressively compare the raw log, a structured log, an Influence Graph, an adaptively constructed graph read as a single prompt, and the complete AIG system (Table 2).

Explicit structure improves attribution. Grouping consecutive steps from the same agent into segments raises exact step accuracy from 46.40% to 48.80%, despite preserving the original evidence and adding no semantic abstraction. This isolates the benefit of making the execution structure explicit. The effect is even larger with Sonnet-4 and DS-V3.2, whose accuracy rises from 20.63% to 48.41% and from 19.84% to 45.24%, respectively (App. D). Once the trace is structured, the three models perform within four points of one another, suggesting that Opus-5 already recovers much of this structure from the raw log.

Influence structure adds further gains. Enriching the structured log into an Influence Graph raises exact step accuracy from 48.80% to 52.00%. The representation augments each segment with a compact account of what it received and produced, then connects segments with inheritance edges that record how earlier outputs propagate downstream. The reader still consumes the entire representation in a single call, so this gain comes from making the execution’s influence structure explicit rather than from agent-directed traversal.

Agent-directed traversal unlocks the adaptive graph. When the adaptively constructed graph is flattened into a single prompt, step accuracy reaches 50.40%. Giving the reader control over how to traverse the same graph raises accuracy by 4.8 points to 55.20%, the strongest result in the ladder. The graph therefore serves not merely as a richer prompt, but as a high-level map that directs the reader toward relevant evidence and supports selective verification against the original log. Overall, adaptive construction paired with agentdirected traversal improves exact step accuracy by 8.8 points over raw-log reading. This interaction supports our central claim that trace representation and reading should be designed jointly.

<table><tr><td>Representation</td><td>Step Acc.</td><td>±1</td><td>±2</td><td>±3</td></tr><tr><td>RAFFLES (Llama-3.3-70B)</td><td>43.65</td><td>58.73</td><td>73.81</td><td>82.54</td></tr><tr><td>Structured log</td><td>48.80</td><td>60.00</td><td>72.00</td><td>80.00</td></tr><tr><td>Influence Graph</td><td>52.00</td><td>61.60</td><td>73.60</td><td>80.00</td></tr><tr><td>Adaptive Influence Graph</td><td>55.20</td><td>64.00</td><td>78.40</td><td>84.00</td></tr></table>

Table 3: Step accuracy under relaxed tolerance windows (±n steps) on the Algorithm-Generated Who&When partition (Opus-5). RAFFLES results as published by the authors, who report tolerance windows only for Llama-3.3-70B.

## 5 Analysis

We next examine where AIGs improve attribution and how the two interface stages produce that gain. We analyze which failures benefit most, then characterize the builder’s graphs and the reader’s evidence gathering.

## 5.1 Where AIGs Help

AIGs improve localization deeper in the trace. A paired Sonnet-4 analysis shows that the gains concentrate on later failures, where downstream consequences can obscure their origin. Exact accuracy on failures at step 5 or later rises from 10.8% with raw-log reading to 40.5% with AIGs. Among incorrect raw-log predictions, 76% still identify the correct agent, suggesting that the main difficulty is precise step localization rather than agent identification. This is consistent with AIGs helping the reader trace downstream consequences to an earlier source. Opus-5 shows the same qualitative pattern, with a smaller margin because its raw-log baseline is already stronger.

AIGs remain strongest under relaxed localization. Among methods reporting tolerance windows, AIGs lead at every window (Table 3): 55.20% exactly, 64.00% within one step, 78.40% within two, and 84.00% within three. Even when missing the exact annotation, they often localize the failure nearby.

## 5.2 What the Builder Constructs

Because AIGs are model-authored, we can examine how adaptive construction changes the resulting representation. Table 4 compares AIGs with the fixed Influence Graph builder on the same failed traces, while Figure 4 illustrates both representations for a single execution. Edge counts include the sequential links between adjacent nodes and the inheritance links added on top of them.

<table><tr><td>Property</td><td>AIGs</td><td>IGs</td></tr><tr><td>Nodes per graph</td><td> $9 . 5 \pm 2 . 0$ </td><td> $7 . 5 \pm 2 . 2$ </td></tr><tr><td>Edges per graph</td><td> $1 3 . 5 \pm 3 . 6$ </td><td> $9 . 0 \pm 3 . 2$ </td></tr><tr><td>Edge density (edges / node)</td><td> $1 . 4 \pm 0 . 2$ </td><td> $1 . 2 \pm 0 . 2$ </td></tr><tr><td>Max hub out-degree</td><td> $3 . 6 \pm 1 . 1$ </td><td> $1 . 8 \pm 0 . 6$ </td></tr><tr><td>Max steps merged into one node</td><td> $2 . 8 \pm 1 . 2$ </td><td> $1 . 6 \pm 0 . 5$ </td></tr><tr><td>Mean cross-range distance</td><td> $3 . 6 \pm 1 . 0$ </td><td> $3 . 1 \pm 1 . 0$ </td></tr><tr><td>Longest single cross-edge</td><td> $5 . 6 \pm 2 . 1$ </td><td> $4 . 2 \pm 2 . 2$ </td></tr></table>

Table 4: Properties of the constructed graphs averaged across the Algorithm-Generated partition (± standard deviation across graphs).

![](images/0a5b0757b6b4f875574db037fc17db2498b3997d6a20cca58e90e344de92a699.jpg)  
Figure 4: The same trace encoded as an Influence Graph and as an Adaptive Influence Graph.

Adaptive construction produces richer, variablegranularity graphs. AIGs contain more nodes and edges than fixed Influence Graphs on average (9.5 vs. 7.5 nodes and 13.5 vs. 9.0 edges), yielding a denser representation of the execution. Their most connected node also has twice the out-degree (3.6 vs. 1.8), indicating that the builder often identifies one upstream action whose output influences several later steps. At the same time, AIGs merge longer runs of consecutive same-agent steps into single action-level nodes: the most compacted node covers 2.8 steps on average, compared with 1.6 for the fixed builder. Thus, the adaptive builder does not apply uniformly finer or coarser segmentation; it varies granularity across the trace while exposing broader patterns of downstream influence.

Adaptive graphs expose longer-range influence. Inheritance links in both span multiple steps, but AIGs reach farther: their mean cross-range distance is 3.6 steps versus 3.1 for fixed Influence Graphs, and their longest edge spans 5.6 steps on average versus 4.2. Together with the higher hub out-degree, this suggests the adaptive builder connects an upstream action to multiple downstream consequences rather than merely restating the local execution order. The resulting graph explicitly encodes long-range propagation for the reader.

![](images/6c3934808c4758365bb1d4f0cf256f2e6987f89d34130eafac215bd605944820.jpg)  
Figure 5: Each panel highlights one move in orange; chords are inheritance edges and numbers are the raw-log steps a node merges. Rather than follow transcript order, the reader walks inheritance edges from the node the graph implicates, then commits to step 6, matching the annotation.

<table><tr><td>Agentic reading profile</td><td></td></tr><tr><td>First raw read is already t</td><td>61.3%</td></tr><tr><td>Opened steps within ±1 of t</td><td>48.4%</td></tr><tr><td>Multi-step investigation</td><td>86.5%</td></tr><tr><td>Single-step verification</td><td>11.9%</td></tr><tr><td>Commits from the graph alone</td><td>1.6%</td></tr></table>

Table 5: Reader behavior on the Algorithm-Generated partition (Sonnet-4, no ground-truth access). The last three rows partition executions by reading depth. t<sup>ˆ</sup> is the step the execution submits.

## 5.3 How the Reader Uses the Graph

The reader’s interaction traces let us examine how it uses the graph rather than only whether the final prediction is correct. Table 5 summarizes the reader executions behind the Sonnet-4 results, and Figure 5 visualizes one complete trajectory.

The graph focuses, but does not replace, evidence gathering. The graph quickly directs the reader toward a candidate region: in 61.3% of executions, the first raw step inspected is also the submitted step, and 48.4% of inspected steps lie within one step of the final prediction. Nevertheless, the reader rarely accepts the graph without verification. It investigates multiple raw steps in 86.5% of executions and commits from the graph alone in only 1.6%. Thus, the graph acts as a highlevel map that focuses the search, while the reader chooses which evidence to inspect before attributing the failure. Figure 5 illustrates this behavior: the reader moves across influence-linked nodes, checks their underlying steps, and commits to the annotated critical step.

Taken together, these analyses show that the two stages play complementary roles: the builder exposes action-level and long-range influence structure, while the reader uses that structure to focus a selective verification process over the original log.

## 6 Conclusion

Failure attribution in multi-agent systems is not only about reasoning over execution traces, but also representing them. We adapt the observability practice of rendering a trace as a structured, navigable object (Arize AI, 2023) to automated attribution: an agentic builder turns a failed trace into an Adaptive Influence Graph, and an independent reader navigates it while checking its claims against the underlying log.

Holding the model backbone fixed, enriching how the trace is represented and read moves step accuracy from 46.40% to 55.20% on the Algorithm-Generated Who&When partition without groundtruth access. This is a new state of the art with +3.6 points over the best reported result (Zhu et al., 2025). AIGs retain their advantage across the backbones we test.

The interface ladder shows that explicit structure carries much of the gain. Structured logs outperform raw transcripts, and influence graphs improve further. The strongest result comes from pairing adaptive graph construction with dynamic navigation. Analysis of the constructed graphs shows how this adaptivity materializes. The builder merges consecutive steps into coarser actions and draws inheritance edges between nodes several steps apart. The reader follows these edges selectively and verifies graph claims against the original log.

As agent traces grow longer and more interleaved, the interface to the trace becomes a central part of the attribution method. These results show that failure attribution depends not only on the diagnosing model, but also on the interface through which it understands and explores the trace.

## Limitations

We evaluate on Who&When, the standard benchmark for this task, across both of its partitions; whether the representation ladder transfers to other multi-agent frameworks and task distributions remains open. Who&When exposes agent outputs but not the inputs each agent saw; on benchmarks that also release inputs and tool logs (Chen et al., 2026b), a builder could ground nodes in what each agent actually observed, which we expect to help rather than hinder. Following the benchmark protocol, we score the final attribution against a single annotated step, so we do not measure the intermediate graph directly: absent gold annotations for node granularity or inheritance edges, the builder’s structure is validated by the critic and by downstream accuracy rather than against a reference graph. Finally, AIGs add a build stage and critic–refiner loop, costing more tokens than raw-log reading; the method is therefore best suited to settings where a reliable diagnosis is worth that overhead.

## Ethical considerations

This work studies failure attribution over execution traces from publicly available data; it involves no human subjects, no personal data, and no usergenerated content beyond those benchmarks. The intended use is diagnostic—helping developers localize the agent and step responsible for a failed multi-agent run so they can debug it. As with any attribution method, outputs are imperfect and should not be treated as definitive assignments of blame, particularly in settings where an incorrect attribution could have major downstream consequences.

## References

Amazon Web Services. 2026. Observe your agent applications with Amazon Bedrock AgentCore Observability. https://docs.aws.amazon. com/bedrock-agentcore/latest/devguide/ observability.html. Accessed: 2026-08-04.

Arize AI. 2023. Phoenix: AI observability and evalua tion. https://github.com/Arize-ai/phoenix.

Adi Banerjee, Anirudh Nair, and Tarik Borogovac. 2025. Where did it all go wrong? a hierarchical look into multi-agent error attribution. arXiv preprint arXiv:2510.04886.

Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, and Chetan Bansal. 2026. Agentrx: Diagnosing ai agent failures from execution trajectories. Preprint, arXiv:2602.02475.

Mengzhuo Chen, Junjie Wang, Zhe Liu, Yawen Wang, Haiming Zheng, and Qing Wang. 2026a. From failed trajectories to reliable llm agents: Diagnosing and repairing harness flaws. Preprint, arXiv:2606.06324.

Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, and Qing Wang. 2026b. Seeing the whole elephant: A benchmark for failure attribution in LLM-based multi-agent systems. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 19888–19905.

Cloud Native Computing Foundation. 2019. Open-Telemetry. https://opentelemetry.io.

Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. 2023. Talk like a graph: Encoding graphs for large language models. Preprint, arXiv:2310.04560.

Yu Ge, Linna Xie, Zhong Li, Yu Pei, and Tian Zhang. 2025. Who is introducing the failure? automatically attributing failures of multi-agent systems via spectrum analysis. arXiv preprint arXiv:2509.13782.

Yeonjun In, Mehrab Tanjim, Jayakumar Subramanian, Sungchul Kim, Uttaran Bhattacharya, Wonjoong Kim, Sangwu Park, Somdeb Sarkhel, and Chanyoung Park. 2026. Rethinking failure attribution in multi-agent systems: A multi-perspective benchmark and evaluation. arXiv preprint arXiv:2603.25001.

Dong Ho Kang, Hyeonjeong Cha, and Daein Weon. 2026. Knowledge-based zero-replay debugging of multi-agent llm traces. Preprint, arXiv:2606.14805.

LangChain. 2026. LangSmith Observability. https://docs.langchain.com/langsmith/ observability. Accessed: 2026-08-04.

Han Li, Yifan Yao, Letian Zhu, Rili Feng, Hongyi Ye, Jiaming Wang, Yancheng He, Pengyu Zou, Lehan Zhang, Xinping Lei, Haoyang Huang, Ken Deng, Ming Sun, Zhaoxiang Zhang, He Ye, and Jiaheng Liu. 2026a. Codetracer: Towards traceable agent states. Preprint, arXiv:2604.11641.

Jiazheng Li, Emine Yilmaz, Bei Chen, and Dieu-Thu Le. 2026b. Towards self-improving error diagnosis in multi-agent systems. Preprint, arXiv:2604.17658.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. Preprint, arXiv:2307.03172.

Yang Liu, Hongjiang Feng, Junsong Pu, and Zhuangbin Chen. 2026. Masprism: Lightweight failure attribution for multi-agent systems using prefill-stage signals. Preprint, arXiv:2605.07509.

Guoqing Ma, Jia Zhu, Hanghui Guo, Weijie Shi, Jiawei Shen, Jingjiang Liu, and Yidan Liang. 2025. Automatic failure attribution and critical step prediction method for multi-agent systems based on causal inference. arXiv preprint arXiv:2509.08682.

Hezhe Qiao, Hanghang Tong, Ee-Peng Lim, Bing Liu, and Guansong Pang. 2026. Verifymas: Hypothesis verification for failure attribution in llm multi-agent systems. Preprint, arXiv:2605.17467.

Md Nakhla Rafi, Md Ahasanuzzaman, Dong Jae Kim, Zhijie Wang, and Tse-Hsun Chen. 2026. Falat: Tracing failures in llm agent trajectories via dependencyguided search. Preprint, arXiv:2606.00765.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. Preprint, arXiv:2310.11324.

Strands Agents. 2025. Strands Agents SDK. https: //github.com/strands-agents/sdk-python.

Runchu Tian, Yanghao Li, Yuepeng Fu, Siyang Deng, Qinyu Luo, Cheng Qian, Shuo Wang, Xin Cong, Zhong Zhang, Yesai Wu, Yankai Lin, Huadong Wang, and Xiaojiang Liu. 2025. Distance between relevant information pieces causes bias in long-context llms. Preprint, arXiv:2410.14641.

Junjie Wang, Yawen Wang, Mengzhuo Chen, Xiaofei Xie, Chunyang Chen, Fangwen Mu, Zhe Liu, and Qing Wang. 2026a. A survey for LLM agent trajectory analysis: From failure attribution to enhancement. IEEE Transactions on Software Engineering, pages 1–23.

Yawen Wang, Wenjie Wu, Junjie Wang, and Qing Wang. 2026b. From flat logs to causal graphs: Hierarchical failure attribution for LLM-based multi-agent systems. arXiv preprint arXiv:2602.23701.

Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Zhiyuan Ning, and Yue Zhang. 2025. Abduct, act, predict: Scaffolding causal inference for automated failure attribution in multi-agent systems. arXiv preprint arXiv:2509.10401.

Yifan Yu, Moyan Li, Shaoyuan Xu, Jinmiao Fu, Xinhai Hou, Fan Lai, and Bryan Wang. 2025. COR-RECT: COndensed eRror RECognition via knowledge Transfer in multi-agent systems. arXiv preprint arXiv:2509.24088.

Boxuan Zhang, Jianing Zhu, Zeru Shi, Dongfang Liu, and Ruixiang Tang. 2026. Agentforesight: Online auditing for early failure prediction in multi-agent systems. Preprint, arXiv:2605.08715.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. 2025a. AgenTracer: Who is inducing failure in the LLM agentic systems? arXiv preprint arXiv:2509.03312.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. 2025b. Which agent causes task failures and when? on automated failure attribution of LLM multiagent systems. arXiv preprint arXiv:2505.00212.

Chenyang Zhu, Spencer Hong, Jingyu Wu, Kushal Chawla, Charlotte Tang, Youbing Yin, Nathan Wolfe, Erin Babinsky, and Daben Liu. 2025. RAFFLES: Reasoning-based attribution of faults for LLM systems. arXiv preprint arXiv:2509.06822.

Taiyu Zhu, Yifan Wu, Weilin Jin, Ying Li, and Gang Huang. 2026. StepFinder: A temporal semantic framework for failure attribution in multi-agent systems. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD).

<table><tr><td>Tool</td><td>Description</td></tr><tr><td colspan="2">Inspection  $( \mathcal { T } _ { \mathrm { l o g } } )$ </td></tr><tr><td>get_overview</td><td>Compact metadata about the log: step count, per-step agent, and sizes.</td></tr><tr><td>read_full_log</td><td>Return every step concatenated.</td></tr><tr><td>read_step</td><td>Return one step&#x27;s content.</td></tr><tr><td>read_step_range</td><td>Return an inclusive range of steps.</td></tr><tr><td>list_agents</td><td>List every agent and its system prompt.</td></tr><tr><td>get_question</td><td>Return the original user question verbatim.</td></tr><tr><td>search</td><td>Regex-search all step contents, returning up to 20 matches.</td></tr><tr><td>list_block_kinds</td><td>Return the closed block-kind vocabulary and required fields.</td></tr><tr><td>get_story</td><td>Render the current graph as Markdown.</td></tr><tr><td>get_graph</td><td>Return the current graph as JSON.</td></tr><tr><td colspan="2">Construction  $( \overline { { \jmath _ { \mathrm { b u i l d } } } } )$ </td></tr><tr><td>add_block</td><td>Create a block (task, orchestrator, agent, or conclusion) from its fields and from/to lists; inheritance notes may accompany it.</td></tr><tr><td>add_inheritance_note</td><td>Attach a (bias, anomaly) note to a direct source→target edge.</td></tr><tr><td>add_skip_attribution</td><td>Attach a (bias, anomaly〉 note to a non-adjacent source→target propagation.</td></tr><tr><td>modify_block</td><td>Overwrite an existing block&#x27;s fields; kind and ID are immutable.</td></tr><tr><td>delete_block</td><td>Delete a block and all edges, including notes, touching it.</td></tr><tr><td>delete_inheritance_note</td><td>Clear the inheritance notes on a direct edge.</td></tr><tr><td>delete_skip_attribution</td><td>Remove a non-adjacent skip attribution.</td></tr><tr><td>mark_defect_recorded</td><td>Flag the node whose output fails its optional system-produced instruction_or_question yet is committed downstream; record what fell short as the</td></tr><tr><td>submit</td><td>critical-step anchor. Emit the final prediction: agent, step, reason, and critical block. Called once.</td></tr></table>

Table 6: Builder tools. Inspection reads the log τ ; construction edits the graph.

## A Tools

Table 6 lists the full tool surface available to the agentic builder, $\mathcal { T } _ { \mathrm { l o g } } \cup \mathcal { T } _ { \mathrm { b u i l d } }$ (section 3.1). The split reflects the two things a builder does. Inspection tools query the trajectory τ at a chosen granularity—an overview, one step, a range, a regex match—and leave the graph untouched; the builder never receives τ in full unless it asks for it. Construction tools are the only way the graph changes: they create and edit typed nodes, ground them in log steps, and attach the ⟨inherited, effect⟩ annotations that make an edge an inheritance claim rather than an adjacency. The reader is given a read-only subset: the graph accessors plus $\tau _ { \mathrm { l o g } }$ , so it can verify any node against the steps that node cites, but cannot revise what it is reading.

## B Typed Nodes

Table 7 gives the node vocabulary $\Theta _ { V }$ (section 3.1) and the fields each type carries. The type is a workflow role: task and conclusion are the graph’s two endpoints, fixed by the validity predicate Φ to one root and one sink, while orchestrator and agent distinguish a turn that routes work from a turn that performs it. Fields follow from the role.

Only agent nodes carry input and output, because only a working turn has an instruction it received and a result it committed—and that pair is what an inheritance edge connects, since what a later node inherits is some earlier node’s output. Every node carries $\gamma ( v )$ as step\_refs, so any field can be checked against the steps it claims to abstract; this is what makes the reader’s verification against τ (section 3.2) possible at all.

## C With-Ground-Truth Results

Table 8 reports the with-ground-truth setting, where the attributor has access to the correct final answer during diagnosis. Ground-truth access does not consistently help: AIG drops are at most four traces, within sampling noise at these sample sizes. Similar regressions appear wherever both settings are reported—in the benchmark’s own baselines (Zhang et al., 2025b), in CHIEF’s re-evaluation of them (Wang et al., 2026b), and across the baselines of Zhang et al. (2025a), who note that ground truth “may sometimes mislead the attribution process.”

<table><tr><td>Node type</td><td>Role &amp; fields</td></tr><tr><td>task</td><td>The user&#x27;s question that seeds the trajectory—the root of the graph (no incoming edges). Fields: content (the question), operative_brief (the instructions the trajectory starts under), step_refs</td></tr><tr><td>orchestrator</td><td>A coordinating turn—an agent that plans, delegates, or judges progress and routes the work onward. Fields: agent_name, summary (what the agent did in this turn, start to end), authorship (as in agent), step_refs</td></tr><tr><td>agent</td><td>A working turn—an invoked agent that executes a delegated instruction and produces a result. Fields: agent_name, summary, input (the instruction, context, or prior result it worked from), output (the concrete result, decision, tool output, or handoff it produced), authorship (whether it authored that output) defect_recorded (the node whose output fails the optional system-emitted instruction_or_question it was given, yet is committed downstream), step_refs</td></tr><tr><td>conclusion</td><td>The final answer the system produced—the terminal sink (no outgoing edges). Fields: content (the final answer), step_refs</td></tr></table>

Table 7: Node types in the agentic graph. Every node carries step\_refs, the raw-log steps it covers.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Model</td><td colspan="2">Algorithm-Generated</td><td colspan="2">Hand-Crafted</td></tr><tr><td>Step Acc. ↑</td><td>Agent Acc. ↑</td><td>Step Acc. ↑</td><td>Agent Acc. ↑</td></tr><tr><td>All-at-Once</td><td>Opus-5</td><td>44.00</td><td>63.20</td><td>18.97</td><td>51.72</td></tr><tr><td>Step-by-Step</td><td>Opus-5</td><td>40.80</td><td>54.40</td><td>22.40</td><td>50.00</td></tr><tr><td>FAMAS (Ge et al., 2025)</td><td>Qwen2.5-72B</td><td>23.81</td><td>55.56</td><td>41.38</td><td>62.07</td></tr><tr><td>AgenTracer- 8B (Zhang et al., 2025a)</td><td>Qwen3-8B</td><td>42.86</td><td>69.62</td><td>20.68</td><td>69.10</td></tr><tr><td>CHIEF (Wang et al., DS-V3.2 2026b)</td><td></td><td>52.00</td><td>76.80</td><td>29.31</td><td>77.59</td></tr><tr><td>CHIEF† (Wang et al., 2026b)</td><td>Opus-5</td><td>44.44</td><td>61.90</td><td>27.59</td><td>63.79</td></tr><tr><td>Adaptive Influence Graph</td><td>DS-V3.2</td><td>50.00</td><td>66.67</td><td>22.41</td><td>60.34</td></tr><tr><td>Adaptive Influence</td><td>Sonnet-4</td><td>51.59</td><td>69.05</td><td>25.86</td><td>50.00</td></tr><tr><td>Graph Adaptive Influence</td><td>GPT-5.6-sol</td><td>51.59</td><td>67.46</td><td>24.14</td><td>55.17</td></tr><tr><td>Graph Adaptive Influence Graph</td><td>Opus-5</td><td>52.00</td><td>67.20</td><td>27.59</td><td>62.07</td></tr></table>

Table 8: Who&When benchmark with ground-truth access at inference time. Baseline numbers are as reported by their authors. One Algorithm-Generated log is content-filtered by Opus-5; therefore, we use n=125 for this model. <sup>†</sup>Run using the authors’ released code. Our methods are shaded.

## D Representation Ablations Across Backbones

Table 2 reports the representation ladder under Opus-5. Tables 9 and 10 repeat it under Sonnet-4 and DS-V3.2. The ordering is the same in all three: the raw transcript is the weakest interface, structuring it helps, the Influence Graph adds a further gain, and the AIG is strongest. The size of each step differs—the two weaker backbones, DS-V3.2 and Sonnet-4, gain far more from structure alone than Opus-5 does, which already reads the flat transcript comparatively well—but no backbone reverses the ladder, so the effect of the interface is not an artifact of one model.

## E IG Example

An IG is what the enrichment builder produces: the deterministic node partition of Section 3.1 left untouched, with an abstraction written into each node and inheritance edges added over the chain. Every node carries a short account of what it did, the input it operated under, and the artifact it committed, alongside the verbatim log content it abstracts. Edges are added only where a later node reused part of an earlier node’s work in a way tied to what went wrong. Figure 6 shows an IG for one trace, where two of the five edges are non-adjacent and one carries the opening node’s reframing of the task directly into the final verification.

<table><tr><td>Model</td><td>Representation</td><td></td><td>Step Acc. ↑ Agent Acc. ↑</td></tr><tr><td rowspan="5">Sonnet-4</td><td>Raw log</td><td>20.63</td><td>61.90</td></tr><tr><td>Structured log</td><td>48.41</td><td>60.32</td></tr><tr><td>IGs</td><td>49.60</td><td>66.40</td></tr><tr><td>AIGs</td><td>53.97</td><td>67.46</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 9: Sonnet-4 representation ablations on the Algorithm-Generated Who&When partition (no groundtruth access). Our AIGs are in bold.

## F AIG Example

An AIG is what the build stage produces: a failed trajectory rewritten as a compact, query-specific object rather than a transcript to be read end to end. Consecutive log steps are merged into nodes at the granularity of an action and given a short abstraction of what that action received and produced, and edges are drawn between nodes—adjacent or several steps apart—where a later action reused part of an earlier one’s work in a way tied to the failure. The result is a small graph the reader can navigate and backtrack along, verifying each claim against the steps the node cites. Figure 7 shows a finalized AIG for one trace.

<table><tr><td>Model</td><td>Representation</td><td></td><td>Step Acc. ↑ Agent Acc. ↑</td></tr><tr><td rowspan="4">DS-V3.2</td><td>Raw log</td><td>19.84</td><td>62.70</td></tr><tr><td>Structured log</td><td>45.24</td><td>62.70</td></tr><tr><td>IGs</td><td>51.20</td><td>64.00</td></tr><tr><td>AIGs</td><td>53.17</td><td>65.08</td></tr></table>

Table 10: DeepSeek-V3.2 (DS-V3.2) representation ablations on the Algorithm-Generated Who&When partition (no ground-truth access). Our AIGs are in bold.

```jsonl
{
2 "nodes": [
3 {
"id": "A",
5 "step_refs": [0],
"agent": "JSON_Expert",
"summary": "Received a reframed task: instead of the original DVD-to-Colombia shipping
8 "input": "A general task showing 'exitcode: 1 (execution failed), Code output: unknown
9 "output": "Restated task framing: debug the 'unknown language json' error following the
10 "authorship": "relayed",
11 "raw_content": "You are given: (1) a task and advises from your manager with a specific
12 13 },<sub>{</sub>
14 "id": "B",
15 "step_refs": [1],
16 "agent": "Debugging_Expert",
17 "summary": "Interpreted 'unknown language json' as a language-code parsing issue. ....
18 "input": "Task from Step 0: analyze and resolve the 'unknown language json' error."
19 "output": "A mock Python script plus a modified version defaulting unsupported languages .
20 "authorship": "authored",
21 "raw_content": "# Analyzing the Error Message and Identifying Potential Causes The error
22 },
23
24 ...
25
26 {
27 "id": "F"
28 "step_refs": [5, 6],
29 "agent": "Verification_Expert",
30 "summary": "Declared the debugging task resolved based on the successful script execution ....
31 "input": "Successful execution output from Step 4 confirming the script runs without error."
32 "output": "Verification write-up concluding the 'unknown language json' issue is resolved ....",
33 "authorship": "authored"
34 }
35 ],
36 "edges": [
37 {
38 "from": "A",
39 "to": "B",
40 "inherited": "Step 0 reframed the whole task as a 'debug the unknown language json error' problem, discarding the
,→ DVD-shipping ...
41 "effect": "Step 1 inherited that framing and pursued a fabricated language-parsing bug, ensuring the shipping question
,→ was
42 },
43 {
44 "from": "B",
45 "to": "C",
46 "inherited": "Step 1 delivered code blocks whose fence/language tagging was itself malformed.",
47 "effect": "Step 2 refused them with 'unknown language json' -- the same symptom under investigation, now reproduced by
,→ the ...
48 },
49 {
50 "from": "C",
51 "to": "D",
52 "inherited": "Step 2 reported the error literally, as a language-code failure rather than a code-fence tagging issue.",
53 "effect": "Step 3 kept treating it as a bug in the script and resubmitted the same fabricated fix."
54 55 },<sub>{</sub>
56 "from": "A",
57 "to": "F",
58 "inherited": "Step 0's reframed objective was to fix the 'unknown language json' error.",
59 "effect": "Step 5 declared success on that objective, never returning the DVD-to-Colombia sender/price JSON the
,→ original ....
60 },
61 {
62 "from": "B",
63 "to": "D",
64 "inherited": "Step 1 fabricated a `parse_language_setting` mock and an interpretation of the error to go with it.",
65 "effect": "Step 3 reused both, propagating the misdiagnosis instead of revisiting the framing."
66 }
67 ]
68 }
```  
Figure 6: Example of an IG. Nodes follow the deterministic partition and carry an abstraction of the log steps they ground in; edges annotate what a later node reused from an earlier one and how it went wrong downstream.

```jsonl
1 {
2 "nodes": [
3 {
4 "id": "A",
5 "step_refs": [0],
6 "agent": "Data_Extraction_Expert",
7 "summary": "Received the manager's task framing: count High Energy Physics - Lattice ....",
8 "input": "General question about counting hep-lat Arxiv articles from January 2020 with ps ...
9 "output": "Restated plan and constraints; carried forward prior result of 0 with the ....
10 "authorship": "relayed",
11 "raw_content": "You are given: (1) a task and advises from your manager with a specific ...."
12 },
13 {
14 "id": "B",
15 "step_refs": [1],
16 "agent": "Verification_Expert",
17 "summary": "Proposed a Python script using arxiv_search with query 'cat:hep-lat AND ...."
18 "input": "Plan from Step 0 to extract hep-lat Jan 2020 articles and count those with ps
19 "output": "Python code invoking arxiv_search and computing ps_count via substring check
20 "authorship": "authored",
21 "raw_content": "To begin solving this task, we should follow the given plan step-by-step
22 },
23 {
24 "id": "C",
25 "step_refs": [2],
26 "agent": "Computer_terminal",
27 "summary": "Executed the provided Python script. Execution succeeded (exitcode 0) and printed 0.",
28 "input": "Python script from Step 1 querying arxiv_search and counting ps in entry_id.",
29 "output": "exitcode 0; stdout: 0.",
30 "authorship": "executed",
31 "raw_content": "exitcode: 0 (execution succeeded) Code output: 0"
32 },
33 {
34 "id": "D",
35 "step_refs": [3, 4],
36 "agent": "Verification_Expert",
37 "summary": "Interpreted the execution output of 0 as confirmation that no hep-lat January ....
38 "input": "Computer_terminal result of 0 from Step 2.",
39 "output": "Final answer 0; TERMINATE signal issued. raw_content (step 3): Based on the ....",
40 "authorship": "relayed"
41 }
42 ],
43 "edges": [
44 {
45 "from": "B",
46 "to": "C",
47 "inherited": "Step 1 defined ps-detection as substring 'ps' in entry_id, a field that holds an arxiv URL, not format
,→ info.",
48 "effect": "Step 2 executed this flawed heuristic and produced 0."
49 },
50
51 "from": "C",
52 "to": "D",
53 "inherited": "Step 2 returned 0, a figure produced entirely by the entry_id substring methodology.",
54 "effect": "Step 3 accepted that 0 at face value without questioning whether arxiv_search returns any indicator of ps
,→
55 },
56 {
57 "from": "A",
58 "to": "D",
59 "inherited": "Step 0 propagated the prior '0 / no articles found' framing.",
60 "effect": "Step 3 leaned on that same conclusion to justify terminating with 0 rather than probing the ps-detection
,→ method."
61 }
62 ]
63 }
```  
Figure 7: Example of an AIG. The representation the builder produces and the reader consumes: typed nodes carrying a short abstraction of the log steps they ground in, and inheritance edges annotating what a later node reused from an earlier one.