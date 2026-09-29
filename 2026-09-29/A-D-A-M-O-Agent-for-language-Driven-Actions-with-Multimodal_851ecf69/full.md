# A.D.A.M.O. (Agent for language-Driven Actions with Multimodal Observations): A Visual-Symbolic Framework for Virtual Humans

Alessandro Emmanuel Pecora<sup>[ORCID:0009-0007-6826-5514]\*</sup>

Stefano Calzolari<sup>[ORCID:0009-0009-7038-2212]\*</sup>

Francesco Strada<sup>[ORCID:0000-0001-9197-9100]</sup>

Andrea Bottino<sup>[ORCID:0000-0002-8894-5089]</sup>

DAUIN, Politecnico di Torino, Torino, Italy {name}.{surname}@polito.it

![](images/256ecf373688e86179401fe517f84192f3913a597905efde34af14f9b90b72ce.jpg)  
Figure 1: A.D.A.M.O. architecture follows a Perceive–Reason–Act loop. Observations combine symbolic state, egocentric visual input, and action feedback. A VLM with tool calling and multimodal short-term memory selects actions, which are executed as primitive operations (Walk, Pick/Drop, Look) and fed into the next cycle.

## Abstract

Creating believable Virtual Humans (VHs) requires the coherent integration of perception, reasoning, and action mediated by language. A central challenge is to combine these components into a control loop grounded in interactive 3D environments. To this end, we present A.D.A.M.O. (Agent for

language-Driven Actions with Multimodal Observations), a visual-symbolic framework for language-driven VHs that leverages a pretrained Vision–Language Model (VLM) with tool calling to unify perception, reasoning, and action within a single control loop. A.D.A.M.O. maintains a dual visual-symbolic world model that combines egocentric visual input and synchronized symbolic state to support grounded

task-oriented behavior from natural language prompts. To support diagnostic evaluation, we introduce a controlled task suite organized by a Capability-Difficulty (C-D) taxonomy that breaks down spatial tasks into procedural and linguistic complexity. Experiments in controlled scenes show that semantic labeling strongly influences task completion and failure modes, reducing perceptual ambiguity while shifting failures toward downstream execution, whereas reasoning errors remain comparatively rare.

Keywords: Virtual Humans, Agentic AI, Vision Language Model, Embodied AI, Spatial Reasoning, Cognitive Architecture

## 1 Introduction

Creating believable Virtual Humans (VHs) – artificial characters that look and act like humans in simulated environments [1] – remains a central challenge at the intersection of computer graphics and embodied AI. Beyond visual realism, believability requires agents to connect multimodal perception of 3D environments with goal-directed action through coherent reasoning. When such grounding is achieved, VHs can exhibit human-like behavior in virtual worlds, enabling applications in training, interactive entertainment, and collaborative environments [2, 3]. However, current VHs systems still rely on extensive manual scripting and domain-specific design, which limits flexibility and makes grounded behavior difficult to scale across tasks and environments.

Natural language provides a natural interface for specifying goals and interacting with VHs, yet existing approaches often remain only partially integrated. Embodied Conversational Agents (ECAs) support rich language-driven interaction but typically lack perceptual grounding in 3D environments, limiting situated interaction, such as spatial reference and object manipulation [4, 5]. Conversely, classical General Capable Agents (GCAs) achieve robust control through reinforcement learning but often operate over symbolic states with limited linguistic and visual grounding [6]. Recent LLM-driven embodied agents integrate language and action via tool use and memory [7, 8], but tightly coupling visual observations, symbolic reasoning, and embodied control in dynamic 3D settings remains challenging.

To address this gap, we introduce A.D.A.M.O. (Agent for Language-Driven Actions with Multimodal Observations), a visual-symbolic framework for language-driven VH control that unifies perception, reasoning, and embodied action within a single decision loop. A.D.A.M.O. builds on tool-calling and agentic paradigms [9, 10] and integrates two complementary representations: a visual stream, where egocentric renderings processed by a pretrained Vision–Language Model (VLM) encode spatial and semantic context, and a synchronized symbolic stream that captures object properties and relations as structured text. Cross-referencing these streams supports grounded multimodal reasoning and tool-based interaction, providing a system-level alternative to fully handcrafted control pipelines in controlled 3D environments.

This paper contributes a VHs-specific systemlevel integration of language-based decision making, multimodal perception, short-term memory, and tool-based action, together with an interpretable diagnostic evaluation framework for studying language-driven embodied behavior in VH. To support this evaluation, we introduce a controlled task suite organized by a Capability-Difficulty (C-D) taxonomy that decomposes spatial tasks by procedural and linguistic complexity. Using this setup, we analyze semantic scaffolding, task difficulty, and failure modes under controlled observability. Our experiments show that semantic labeling is a major factor in task completion and failure distribution, reducing perceptual ambiguity while shifting failures toward downstream execution effects, whereas reasoning-related errors remain comparatively rare. Code and data are available at our project page <sup>1</sup>.

## 2 Related Work

Virtual Agents. The landscape of virtual agents reveals a persistent challenge in integrating linguistic interaction with embodied behavior, evident across different generations of agent architectures. Traditional Game AI for Non-Player Characters (NPCs) relies on rule-based behaviors [11, 12], achieving reliable yet manually scripted interactions that require extensive authoring to generalize. Research then diverged into two specialized directions. ECAs prioritized linguistic sophistication, achieving human-like interaction through coordinated verbal and nonverbal behavior [4, 13, 14], yet decoupling communicative intent from physical realization. Recent Large Language Model (LLM)-enhanced systems [4, 15] improve dialogue but remain disconnected from spatial reasoning and object manipulation. Conversely, GCAs prioritized embodied behavior, leveraging LLMs to decompose goals and acquire skills in sandbox environments [16, 17]. Most rely on symbolic world states offering limited perceptual grounding, while recent vision language variants [7, 18, 19] still lack robust contextual grounding. Even attempts to bridge these dimensions fall short: language-driven NPCs [20] remain heavily user-dependent, lacking autonomous perception-action loops for independent spatial reasoning.

Regarding spatial awareness, early systems such as Max [21] introduced symbolic control for environment-aware interaction but relied on handcrafted rules, which limited adaptability. Li et al. [22] later proposed virtual humans operating from text-encoded scene descriptions, enabling rich linguistic reasoning over spatial relations but requiring exhaustive manual annotation. These omniscient setups lack scalability to complex environments and cannot exploit visual cues for disambiguation [23].

A.D.A.M.O. addresses these limitations by unifying egocentric vision and structured symbolic memory within a single language-based control layer. The symbolic stream encodes spatial structure for explicit reasoning, while the visual stream supports perceptual verification and disambiguation. Leveraging pretrained VLMs, the framework interprets highlevel commands and autonomously composes primitive actions, reducing reliance on fully handcrafted pipelines and supporting grounded perception–reasoning–action in controlled virtual environments.

Embodied AI Evaluation. Beyond virtual agents, robotics and embodied Artificial Intelligence (AI) pursue similar goals – grounded perception, linguistic understanding, and goaldirected action – in physical or simulated settings [24, 8, 25]. Although operating under different constraints (partial observability, noisy sensing, physical dynamics), the embodied AI community has developed unified benchmarks to evaluate language-conditioned task execution. Frameworks such as AL-FRED [26] emphasize procedural execution, while more recent efforts like LoTa-Bench [27] and OpenEQA [28] incorporate richer language understanding but rely on binary success metrics that obscure task difficulty and error sources. As a result, it remains difficult to identify where agents fail, which reasoning skills are most demanding, or how to systematically scale task complexity. Comparable evaluation frameworks are still missing in VH research, despite the need to jointly assess linguistic, spatial, and embodied competence. A.D.A.M.O. addresses this gap through a C-D taxonomy that decomposes tasks along two orthogonal axes – procedural capability (what to do) and linguistic complexity (how it is described) – offering interpretable, scalable measures of embodied reasoning difficulty that explain not only whether agents succeed, but also why they struggle on specific task dimensions.

## 3 Methods

Our goal is to equip VHs with language-driven control that integrates perception, reasoning, and action within a coherent decision loop in controlled 3D environments. In this setup, the agent receives natural language instructions, reasons over spatial and semantic context, and selects goal-directed actions aligned with the task.

This section describes the design of A.D.A.M.O. through two complementary layers. The logical architecture (Sec. 3.1) defines how the agent perceives, reasons, and selects actions from multimodal inputs under partial observability. The infrastructure architecture (Sec. 3.2) details the system-level implementation that supports this loop, separating high-level reasoning from real-time simulation and action execution.

## 3.1 Logical Architecture

In embodied AI, agents must integrate perception, reasoning, and action to operate continuously within their environment. Following robotics literature [29, 5], we adopt the perceive–reason–act paradigm as our architectural foundation, instantiated through the Re-Act framework [9] and adapted here to a multimodal, language-driven VH setting. In A.D.A.M.O., reasoning (Thought), action (Act), and perception (Observation) form a closed feedback loop. The simulation engine maintains full access to the world state, while the agent operates under partial and sequential observability through egocentric perception exposing only task-relevant local context at each step (Fig. 1).

Thought. Reasoning is implemented through a VLM with tool-calling capabilities that, at each iteration, generates a structured reasoning note followed by a specific tool selection. The reasoning note serves as a short-term sketchpad that summarizes perceptual evidence and selects the next action, functioning as an explicit intermediate reasoning trace in the spirit of chain-ofthought prompting [30] and making the decision process interpretable within the Perceive– Reason–Act loop.

Act. Actions are executed as tool invocations that translate linguistic intent into embodied control. Tools interact directly with the simulation engine, which enforces physical consistency (e.g., automatic repositioning when targets are out of reach), thereby decoupling highlevel reasoning from low-level execution details. Tool execution produces new observations that update the world state and close the control loop. In our setup, the agent uses a minimal set of primitive tools – Look, Walk, Pick, and Drop – covering perception, navigation, and object manipulation. This core interface supports the experiments presented in this paper, and its modular design allows additional primitives to be incorporated in future extensions.

Observation. Observations integrate perceptual observation (what does the agent sense?) and feedback (what happened during this action?). Perceptual observation provides a multimodal view of the environment by combining egocentric visual input with a synchronized symbolic state. Visual perception consists of an RGB frame augmented with object tags, bounding boxes for interactable entities, and sparse tagged reference points projected into the camera view, which serve as stable 3D spatial anchors. Symbolic perception mirrors this view as structured text, encoding the currently available interactable object list (using the same tags as the visual stream), along with 3D coordinates, object volumes, reference-point positions, and the agent state, which stores the current agent pose to preserve spatial consistency across reasoning iterations.

Execution feedback is returned as a concise textual summary of the last tool outcome (e.g., success, failure, or automatic correction), providing contextual signals that support coherent reasoning in subsequent iterations.

Multimodal Short-Term Memory (MSTM). To preserve temporal and multimodal coherence across the Thought–Act–Observation loop, A.D.A.M.O. maintains a Multimodal Short-Term Memory (MSTM) that unifies recent visual inputs, symbolic state, and interaction history within a single prompt. The MSTM follows a message-based ChatML structure [31], which organizes dialogue into explicit roles (system, AI, tool, human), and comprises (i) fixed instructions (system) defining behavioral rules and tool semantics, (ii) a dynamically updated symbolic state storing objects, spatial references, and agent pose (system), and (iii) a rolling buffer that retains the last k human (initial prompt and egocentric images), AI (reasoning notes and tool calls), and tool messages (returning environmental observations, i.e., perceptual updates and feedback). All prompts are provided in the project repository.

## 3.2 Infrastructure Architecture

The infrastructure architecture (Fig. 2) implements the Thought–Act–Observation loop through a modular system that decouples highlevel reasoning from simulation and action execution. This separation supports flexibility and interoperability without entangling cognitive decisions with real-time environment dynamics.

The Runtime Engine, implemented in Unity, manages the 3D environment and agent embodiment, handling rendering, physics, and I/O. It captures perceptual data and executes embodied actions through an embedded Action Server that translates high-level tool calls into executable Unity operations and returns both feedback and updated observations.

The Cognitive Server, implemented in Python, orchestrates the Thought–Act– Observation loop, managing prompt construction, memory updates, and tool invocations. Reasoning and action selection are delegated to a VLM Inference Server, which provides access to pretrained VLMs or LLMs through a unified interface.

At runtime, a task prompt and initial observations are sent to the Cognitive Server, which iteratively queries the VLM for the next action, executes it through the Action Server, and updates the Multimodal Short Term Memory (MSTM)

with new observations until task completion (Fig. 2).

Execution Flow. As shown in Fig. 2, the Runtime Engine receives a task prompt and initial observations (1) and forwards them to the Cognitive Server (2), which assembles a multimodal prompt and queries the VLM Inference Server (3). The VLM returns either a tool invocation or a termination decision (4). Tool calls are executed via the Action Server in the Unity Runtime (5), which updates the environment and returns perceptual observations and execution feedback (6). These updates are integrated into the MSTM, enabling iterative Thought– Act–Observation cycles (3–6) until completion, after which the Cognitive Server signals termination (7) and the Runtime Engine delivers the final agent response (8).

## 4 Experiments

We evaluate A.D.A.M.O. on spatial reasoning and manipulation tasks in a controlled simulated 3D environment, aiming to analyze performance across task types and sensitivity to key design choices in perception and reasoning.

Our experimental setup includes 15 selfcontained episodes, each defined by a triplet (S, T, C): the scene S specifies the 3D environment, the interactable objects layout, and initial agent pose; the task prompt T is a natural-language instruction describing an object-centric manipulation task, and the checker C deterministically evaluates task completion. We consider two scenes (Fig. 3): a simple tabletop scene where all interactable objects are placed on a single table (S1) and a livingroom scene with the same objects distributed across multiple pieces of furniture, yielding a more complex spatial layout (S2), both containing the same 19 interactable objects but differing in spatial arrangement and contextual complexity. Task prompts combine basic manipulation primitives (pick, place) with varying spatial constraints (near, on).

![](images/e686806b56bc0353c49df7920d4a88819d19d4a7d4db9de6d9aa67ff754faa39.jpg)  
Figure 2: Infrastructure architecture. Numbered labels indicate dataflow and execution sequence across modules; dotted steps (3–6) implement the Thought–Act–Observation cycle.

For each episode, the checker returns the completion rate, a normalized score in [0, 1] based on the fraction of satisfied atomic conditions (e.g., placing one of four required objects yields a score of 0.25). Task difficulty is characterized using the agent-agnostic C-D taxonomy described in Sec. 4.1. Full details on scenes, and checker logic are provided in the online repository. Tasks are listed in Table 2.

To study sensitivity to architectural design choices, we compare variants of the perceptual and reasoning stack that differ in (i) the VLM backbone and (ii) the object-labeling scheme used for visual and symbolic references. Specifically, we test GPT-4o-vision (G4O) and Claude-Sonnet-3.5 (S35) under identical inputs, and contrast a semantic labeling scheme (SEM), where tags encode class and instance information, with an opaque one (OPAQ) using numeric identifiers only. This design allows us to examine how semantic scaffolding affects grounding performance and action efficiency.

![](images/b1d0ca26014115d35a500cf07c8e49b9d2e5ade2dac8ea7e0f3996eb66f71cd2.jpg)  
Figure 3: Scene in our benchmark (top left: S1; bottom left: S2) with closeups of the respective objects arrangement (right).

## 4.1 Capability-Difficulty (C-D) Taxonomy

We use a Capability–Difficulty (C-D) taxonomy as a simple, interpretable heuristic to organize task difficulty within our experimental setup. Each episode, defined by a natural-language instruction, is decomposed into three unit types: Actions ACT, which count explicit manipulation primitives; References REF, which capture the effort required to uniquely identify target objects through linguistic or perceptual descriptors; and Relations REL, which denote spatial constraints that must be satisfied in the final configuration. Each unit is assigned an integer cost reflecting its relative complexity, and the C-D level of an episode is given by the sum of its unit costs.

This taxonomy provides an agent-agnostic ordering of task difficulty that supports diagnostic analysis of performance trends and failure modes across task families. Locomotion and perceptual actions are treated as implicit and are not explicitly scored. Table 1 summarizes the units and costs used in our experiments; full task specifications and cost assignments are provided in the online repository.

Example. In a scene with a yellow ball, a black cup, and a whitefork, the task Pick the black object and place it near the spherical object yields ACT=2 (pick+drop), REF=3 (color+shape), and REL=2 (proximity), for a total C-D level of 7.

Table 1: Capability–Difficulty (C–D) taxonomy: units, specification, and levels.
<table><tr><td>Dim</td><td>Unit</td><td>Specification</td><td>Level</td></tr><tr><td rowspan="2">Act</td><td>pick</td><td>Pick an object.</td><td>1</td></tr><tr><td>drop</td><td>Release or place an object.</td><td>1</td></tr><tr><td rowspan="9">Ref</td><td>id</td><td>Numerical identifier, unique</td><td>0</td></tr><tr><td>tag</td><td>in scene. Semantic tag or name.</td><td>1</td></tr><tr><td>attrColor</td><td>Color attribute.</td><td>1</td></tr><tr><td>attrShape</td><td>Shape attribute.</td><td>2</td></tr><tr><td>groupSimple</td><td>Single-criterion group filter.</td><td>2</td></tr><tr><td>attrPattern</td><td>Pattern attribute.</td><td>3</td></tr><tr><td>superExtremum</td><td>Superlative over an attribute:</td><td>3</td></tr><tr><td>floor</td><td>min or max. Place on the floor.</td><td>1</td></tr><tr><td>prox</td><td>Place near another object.</td><td>2</td></tr><tr><td>Rel area</td><td>Place within a designated</td><td>4</td></tr></table>

## 4.2 Instrumentation

For each VLM and labeling scheme, every episode is executed 15 times (with a 10-minute cap) to account for model stochasticity and runtime failures. Results are reported as episodelevel averages and aggregated by scene, VLM, and labeling scheme. We measure completion rate (CR), model usage (number of tool calls and per-call generation time), and control activity (counts of Walk, Look, Pick, and Drop actions). All agentic models are evaluated under identical settings. The MSTM rolling buffer stores the last k=50 interaction steps. Decoding uses greedy sampling (temperature = 0.0, top-p=0.0) with a maximum output length of 4000 tokens per message. Visual observations use a 1920×1080 RGB frame and a 6×6 grid of raycasted reference points uniformly distributed across the agent’s field of view. Tasks are ordered for increasing C-D levels, which range between 1 for T1 until 20 for T15 (Table 2).

## 5 Results

We analyze A.D.A.M.O.’s behavior on spatial reasoning and manipulation tasks to examine performance under controlled variation, identify dominant failure modes, and assess sensitivity to key design choices in perception and reasoning.

Table 2: Full list of task prompts and corresponding C-D levels of our taxonomy.
<table><tr><td>Task ID</td><td>Task Prompt</td><td>ACT</td><td>REF</td><td>REL</td><td>TOT</td></tr><tr><td>T1</td><td>Pick the object tagged as 34</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>T2</td><td>Pick the plate</td><td>1</td><td>1</td><td>0</td><td>2</td></tr><tr><td>T3</td><td>Pick the blue circular object</td><td>1</td><td>1</td><td>0</td><td>2</td></tr><tr><td>T4</td><td>Pick the plate and place it on the floor</td><td>2</td><td>1</td><td>1</td><td>4</td></tr><tr><td>T5</td><td>Pick the object tagged as 4 and place it near the indoor plant</td><td>2</td><td>1</td><td>2</td><td>5</td></tr><tr><td>T6</td><td>Pick the smallest object then place it on the floor</td><td>2</td><td>3</td><td>1</td><td>6</td></tr><tr><td>T7</td><td>Pick the flashlight and place it on the floor in a corner of the room</td><td>2</td><td>1</td><td>4</td><td>7</td></tr><tr><td>T8</td><td>Pick the glass and the plate then put them on the floor</td><td>4</td><td>2</td><td>2</td><td>8</td></tr><tr><td>T9</td><td>Pick object tagged as 37 then place it next to the plate, then go pick object tagged as 30 and place it next to object tagged as 3</td><td>4</td><td>1</td><td>4</td><td>9</td></tr><tr><td>T10</td><td>Pick the glass then place it next to the plate, moreover pick the binoculars and place it next to the screwdriver</td><td>4</td><td>4</td><td>4</td><td>11</td></tr><tr><td>T11</td><td>Pick object tagged as 4 and place it near the indoor plant, then pick the glass and place it next to the plate</td><td>4</td><td>3</td><td>4</td><td>12</td></tr><tr><td>T12</td><td>Pick the cup, peanut butter, and pills and place them on the floor</td><td>6</td><td>3</td><td>3</td><td>12</td></tr><tr><td>T13</td><td>Pick the cylindrical can and place it next to the indoor plant, and pick the binoculars and place it next to the screwdriver</td><td>4</td><td>4</td><td>4</td><td>12</td></tr><tr><td>T14</td><td>Pick all objects with white parts and place them on the floor</td><td>10</td><td>3</td><td>5</td><td>15</td></tr><tr><td>T15</td><td>Pick all objects that have a printed product label and place them on the floor</td><td>10</td><td>5</td><td>5</td><td>20</td></tr></table>

## 5.1 Model and Labeling Scheme

Table 3 reports average performance (completion rate, walk, look and total tool calls, and call execution time) across all tasks and scenes, aggregating 15 repetitions per episode for each backbone (G4O, S35) and labeling scheme (SEM, OPAQ). Across both models, the labeling scheme is the dominant factor: semantic labeling (SEM) markedly increases completion relative to opaque labeling (OPAQ), while backbone choice primarily affects efficiency and the way failures manifest.

Backbone effects. Averaging results across labeling schema, G4O achieves higher average completion than S35 (0.68 vs. 0.61) and lower per-call latency (6.85s vs. 8.01s), but executes more actions per episode (4.70 vs. 4.19). This difference reflects distinct failure behaviors: S35 more frequently terminates early, inferring task completion before all goals are satisfied, whereas G4O tends to maintain the reasoning–action loop until environment feedback confirms completion. Thus, the backbone appears to modulate decisiveness versus completion persistence more than overall task resolvability.

Effect of semantic labeling. SEM labeling yields the largest gains, increasing completion from 0.48→0.88 (+83%) for G4O and from 0.38→0.83 (+118%) for S35, while reducing exploratory perception. The number of Look actions drops by 12% for G4O and 41% for S35, and total action counts decrease across both models, suggesting more direct, plan-driven trajectories under semantic grounding.

Failure modes and exploration. A quantitative error breakdown clarifies these trends. Specifically, we parsed execution logs from failed episodes to categorize failure causes at scale. Under OPAQ, failures are overwhelmingly perceptual (83.5% against 14.2% of execution-related), reflecting ambiguity in object identification without semantic cues. Under SEM, perceptual errors drop to 65.8%, while execution-related failures increase to 31.1%, suggesting a shift from misidentification toward downstream action and termination effects. Reasoning-related errors remain rare in both settings (2–3%).

## 5.2 Tasks and Scenarios

Given the strong performance difference between the two labeling conditions, we next analyze how scene complexity and task structure affect performance under semantic labeling. Figure 4 reports per-task completion across both models and environments, while Table 3 summarizes scene-level averages.

![](images/9b46336469e855a9a42e33e1261432263de4dfd091ce7bdac9eb4880fb5e9dc8.jpg)  
Figure 4: Completion rate across all tasks for different scenes and models using SEM labeling scheme.

Table 3: Average performances across all episodes for labeling scheme (top blocks) and, for SEM scheme only, per-scene (bottom blocks).
<table><tr><td rowspan="2">Model</td><td rowspan="2">CR↑</td><td rowspan="2">Time per Call (s) ↓</td><td rowspan="2">#Walk</td><td rowspan="2">#Look</td><td rowspan="2">#Total</td></tr><tr><td></td></tr><tr><td></td><td></td><td>Labeling Scheme: OPAQ</td><td></td><td></td><td></td></tr><tr><td>G4O</td><td>0.48</td><td>7.63</td><td>0.53</td><td>1.61</td><td>4.80</td></tr><tr><td>S35</td><td>0.38</td><td>8.31</td><td>0.38</td><td>2.10</td><td>4.57</td></tr><tr><td></td><td></td><td>Labeling Scheme: SEM</td><td></td><td></td><td></td></tr><tr><td>G40</td><td>0.88</td><td>6.06</td><td>0.62</td><td>1.41</td><td>4.60</td></tr><tr><td>S35</td><td>0.83</td><td>7.72</td><td>0.23</td><td>1.25</td><td>3.80</td></tr><tr><td></td><td>Scene: S1 (SEM Labeling Scheme)</td><td></td><td></td><td></td><td></td></tr><tr><td>G40</td><td>0.93</td><td>4.71</td><td>0.00</td><td>1.08</td><td>3.67</td></tr><tr><td>S35</td><td>0.90</td><td>7.94</td><td>0.19</td><td>1.19</td><td>3.75</td></tr><tr><td></td><td>Scene: S2 (SEM Labeling Scheme)</td><td></td><td></td><td></td><td></td></tr><tr><td>G4O</td><td>0.83</td><td>7.12</td><td>1.24</td><td>1.74</td><td>5.53</td></tr><tr><td>S35</td><td>0.76</td><td>7.49</td><td>0.28</td><td>1.30</td><td>5.89</td></tr></table>

Scene effects. Although S1 (tabletop) and S2 (living room) contain the same 19 objects, S2 introduces greater spatial dispersion and visual clutter. Completion rates are consistently higher in S1 (G4O: 0.93, S35: 0.90) than in S2 (G4O: 0.83, S35: 0.76). The more complex layout in S2 increases exploratory behavior: Walk actions rise from near zero in S1 to 1.24 (G4O) and 0.28 (S35) in S2, with corresponding increases in Look actions. G4O’s broader exploration partially compensates for reduced visibility, yielding higher completion at the cost of additional actions.

Task difficulty and task-family trends. Tasks naturally cluster into families with increasing procedural and perceptual demands. Single-object manipulation tasks (T1–T5, C-D $\leq 5 )$ achieve near-perfect completion, indicating reliable performance on basic grounding tasks. A notable exception is T3 (“pick the blue circular object”), where color–shape discrimination stresses perceptual grounding and reveals differences in how models balance visual evidence and pretrained semantic priors.

Comparative and relational tasks (e.g., T6: “smallest object”, T10: “next to the screw-$d r i \nu e r ^ { \prime \prime } )$ and moderate multi-object tasks (T8– T13, C-D ∈ [6, 12]) remain highly reliable, despite requiring attributes and global spatial relations that are not explicitly encoded in the semantic labels (e.g., T7: “flashlight in a corner”, where the target location is not associated with any semantically labeled object). These results suggest that, even without explicit semantic labels supporting comparative and relational task structure, such tasks can still be handled through the complementarity of the dual stream, with visual and symbolic representations jointly providing the information needed to resolve object comparisons and relations.

Performance degrades for attribute-heavy and long-horizon task families (T14–T15, C-D ≥ 14). T14 (“all objects with white parts”) exposes conflicts between pretrained category–attribute priors and visual evidence, while T15 (“printed product labels”) combines finegrained attribute recognition with extended action sequences, resulting in the lowest completion rates. These effects are amplified in more cluttered scenes.

Across all tasks under SEM labeling, and considering all trials and all models, we observe a significant negative rank correlation between C-D levels and completion rate (Spearman $\rho ~ =$ −0.53, $p \ < \ 0 . 0 5 )$ , indicating that increasing task complexity tends to place greater demands on multimodal grounding.

## 5.3 Ablation Studies

This section examines the sensitivity of key architectural and inference hyperparameters on A.D.A.M.O.’s performance by varying one parameter at a time while keeping all others fixed. All ablations are performed on Scene S1 using the G4O backbone with the SEM labeling scheme, and report completion rate across all 15 tasks along with selected efficiency metrics.

MSTM Rolling Buffer Size (k). Varying the MSTM buffer size controls the amount of recent multimodal context available to the agent. As shown in Table 4, completion improves from 0.82 at $k { = } 4$ to 0.86 at $k { = } 1 0 .$ , and peaks at 0.93 for the baseline $k { = } 5 0$ . Smaller buffers increase exploratory behavior, with more Look actions and model calls (e.g., 6.34 calls at k=4 vs. 4.67 at $k { = } 5 0 )$ , indicating compensation for truncated context through repeated perception. Larger buffers preserve temporal coherence at the cost of higher latency.

Table 4: MSTM rolling-buffer size.
<table><tr><td>k</td><td>CR↑</td><td>#Look</td><td>#Model Call↓</td><td>Time per Call (s) ↓</td></tr><tr><td>4</td><td>0.82</td><td>1.27</td><td>6.34</td><td>3.94</td></tr><tr><td>6</td><td>0.84</td><td>1.20</td><td>9.20</td><td>3.85</td></tr><tr><td>10</td><td>0.86</td><td>1.47</td><td>6.99</td><td>4.05</td></tr><tr><td>50†</td><td>0.93</td><td>1.08</td><td>4.67</td><td>4.71</td></tr></table>

VLM Sampling Hyperparameters. Table 5 shows that deterministic decoding is most reliable for structured tool use: the quasideterministic setting achieves the highest completion rate (0.93). Moderate stochasticity leads to a sharp drop (0.82), while high stochasticity partially recovers performance (0.85), suggesting that increased output diversity occasionally helps, but overall degrades tool-execution reliability.

Table 5: VLM sampling configurations.
<table><tr><td>Configuration</td><td>(Temp ; Top-p)</td><td>CR↑</td></tr><tr><td>Quasi-deterministic</td><td> $( 0 . 0 ; 0 . 0 )$ </td><td>0.93</td></tr><tr><td>Moderately stochastic</td><td> $( 0 . 3 ; 0 . 5 )$ </td><td>0.82</td></tr><tr><td>Highly stochastic</td><td> $( 0 . 8 ; 1 . 0 )$ </td><td>0.85</td></tr></table>

3D Reference-Point Density. Spatial anchoring improves performance up to a point. Increasing reference-point density from 3×3 to $6 \times 6$ raises completion to 0.93 while reducing model calls $_ { ( 4 . 6 7 ) }$ and tokens (32,977) (Table 6). Further densification to $7 \times 7$ lowers completion to 0.87, consistent with increased visual and symbolic clutter, which may dilute salient geometric cues and make action selection less reliable.

Table 6: Number of 3D reference points.
<table><tr><td># Ref. Points</td><td>CR↑</td><td># Model Call ↓</td><td>Tokens per Ep. ↓</td></tr><tr><td>3x3</td><td>0.92</td><td>5.71</td><td>42963.89</td></tr><tr><td>4x4</td><td>0.90</td><td>5.48</td><td>41023.82</td></tr><tr><td>5x5</td><td>0.89</td><td>5.77</td><td>44816.31</td></tr><tr><td>6x6†</td><td>0.93</td><td>4.67</td><td>32977.61</td></tr><tr><td>7x7</td><td>0.87</td><td>5.34</td><td>43372.84</td></tr></table>

Image Resolution. Higher input resolution improves perceptual grounding but increases latency (Table 7). Completion rises from 0.76 at 640×360 to 0.93 at $1 9 2 0 \times 1 0 8 0 .$ , while per-call time increases from $2 . 6 5 \mathrm { ~ s ~ t o ~ } 4 . 7 1$ s, highlighting a clear performance–efficiency trade-off in the perception stage.

Table 7: Input image resolution.
<table><tr><td>Resolution</td><td>CR↑</td><td>Time per Call (s) ↓</td></tr><tr><td> $6 4 0 \times 3 6 0$ </td><td>0.76</td><td>2.65</td></tr><tr><td> $1 2 8 0 \times 7 2 0$ </td><td>0.85</td><td>3.52</td></tr><tr><td> $1 9 2 0 \times 1 0 8 0 ^ { \dag }$ </td><td>0.93</td><td>4.71</td></tr></table>

## 5.4 Limitations and Future Work

A.D.A.M.O. is currently evaluated in controlled simulated environments, where the agent operates under partial and sequential observability through egocentric perception. While this setup supports systematic analysis and still includes occlusions and action-induced scene changes, it does not explicitly model perceptual noise, exogenous environment dynamics, or uncertaintyaware action selection. The framework also relies only on multimodal short-term memory, without persistent spatial or episodic memory. This limits coherence in extended interactions and makes long-horizon tasks more challenging. From a modeling perspective, the current implementation depends on VLMs with native tool-calling support, which restricts compatibility with part of the open-source ecosystem. In addition, fine-grained visual attribute grounding remains difficult, as pretrained semantic priors can conflict with visual evidence in attribute-sensitive tasks. Most importantly, semantic labeling is a major factor in task completion and failure distribution. Although it substantially improves performance, it still requires lightweight scene-side authoring and highlights the current dependence of the framework on semantic scaffolding. The present results should therefore not be interpreted as evidence that raw visual grounding alone is sufficient for reliable language-driven virtual human control. Rather, they point to a trade-off between performance and manual semantic scaffolding, showing that strong performance can already be achieved with lightweight scene-side authoring. Future work will extend the framework with persistent memory, broaden support to open-source VLMs, further reduce manual semantic scaffolding through automatic semantic extraction, and evaluate the system in more complex settings, including dynamic environments, multiagent scenarios, and human–VH collaboration.

## 6 Conclusion

We presented A.D.A.M.O., a visual-symbolic framework for language-driven virtual humans that integrates multimodal perception, reasoning, and embodied control within a unified decision loop. By combining egocentric visual input with structured symbolic state and tool-based actions, A.D.A.M.O. supports interpretable, language-mediated behavior in controlled 3D environments.

Using a Capability–Difficulty-based evaluation protocol, we showed that semantic labeling is a major factor in reliable performance, substantially reducing perceptual ambiguity and shifting failures toward downstream execution effects, while reasoning-related errors remain comparatively rare. Task difficulty and finegrained visual attributes emerge as the main bottlenecks under increased complexity. Experimental results show that A.D.A.M.O. offers a system-level framework and diagnostic setup for studying grounded language-driven behavior in virtual humans, while helping identify the current bottlenecks across perception, memory, and action execution.

## References

[1] Birgit Lugrin. Introduction to socially interactive agents. In The Handbook on Socially Interactive Agents: 20 Years of Research on Embodied Conversational Agents, Intelligent Virtual Agents, and Social Robotics Volume 1: Methods, Behavior, Cognition, pages 1–20. 2021.

[2] Fu-Chia Yang, Pedro Acevedo, Siqi Guo, Minsoo Choi, and Christos Mousas. Embodied conversational agents in extended reality: A systematic review. IEEE Access, 2025.

[3] Siqi Guo, Nicoletta Adamo-Villani, and Christos Mousas. Developing a Scale for Measuring the Believability of Virtual Agents. ICAT-EGVE 2023 - International Conference on Artificial Reality and Telexistence and Eurographics Symposium on Virtual Environments, pages 45–52, 2023. Artwork Size: 8 pages ISBN: 9783038682189 Publisher: The Eurographics Association.

[4] Takeshi Saga, Lucie Galland, Nezih Younsi, and Catherine Pelachaud. Greta 2.0: Social interactive agent system, optimized for neural network integration. In

Proceedings of the 25th ACM International Conference on Intelligent Virtual Agents, pages 1–10, 2025.

[5] Stefan Kopp and Teena Hassan. The fabric of socially interactive agents: Multimodal interaction architectures. In The handbook on socially interactive agents: 20 years of research on embodied conversational agents, intelligent virtual agents, and social robotics volume 2: Interactivity, platforms, application, pages 77–112. 2022.

[6] Open Ended Learning Team, Adam Stooke, Anuj Mahajan, Catarina Barros, Charlie Deck, Jakob Bauer, Jakub Sygnowski, Maja Trebacz, Max Jaderberg, Michael Mathieu, et al. Open-ended learning leads to generally capable agents. arXiv preprint arXiv:2107.12808, 2021.

[7] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Hou, et al. Jarvis-1: Openworld multi-task agents with memoryaugmented multimodal language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[8] Zixuan Wang, Bo Yu, Junzhe Zhao, Wenhao Sun, Sai Hou, Shuai Liang, Xing Hu, Yinhe Han, and Yiming Gan. Karma: Augmenting embodied ai agents with longand-short term memory systems. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1–8. IEEE, 2025.

[9] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

[10] Timo Schick, Jane Dwivedi-Yu, Roberto Dess\`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539– 68551, 2023.

[11] Muhtar C¸ agkan Uluda˘ glı and Kaya O˘ guz.˘ Non-player character decision-making in computer games. Artificial Intelligence Review, 56(12):14159–14191, 2023.

[12] Ian Millington. AI for Games. CRC Press, 2019.

[13] Paulo Ricardo Knob, Natalia Dal Pizzol, Soraia Raupp Musse, and Catherine Pelachaud. Arthur and Bella: multipurpose empathetic AI assistants for daily conversations. The Visual Computer, 40(4):2933–2948, April 2024.

[14] Stefano Calzolari, Francesco Strada, and Andrea Bottino. Toward believable emotions: Evaluating facs coding for virtual human expressions. IEEE Consumer Electronics Magazine, 2025.

[15] Jose Llanes-Jurado, Luc´ıa Gomez-´ Zaragoza, Maria Eleonora Minissi, Mar-´ iano Alcaniz, and Javier Mar˜ ´ın-Morales. Developing conversational virtual humans for social emotion elicitation based on large language models. Expert Systems with Applications, 246:123261, 2024.

[16] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35:18343–18362, 2022.

[17] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Mandlekar, et al. Voyager: An open-ended embodied agent with large language models. arXiv:2305.16291, 2023.

[18] Zhonghan Zhao, Wenhao Chai, Xuan Wang, Ke Ma, Chen, et al. Steve series: Step-by-step construction of agent systems in minecraft. arXiv preprint arXiv:2406.11247, 2024.

[19] Zhonghan Zhao, Wenhao Chai, Wang, et al. See and think: Embodied agent in virtual environment. In European Conference on Computer Vision. Springer, 2024.

[20] Sudha Rao, Weijia Xu, Michael Xu, Jorge Leandro, Ken Lobb, Gabriel DesGarennes, Chris Brockett, and Bill Dolan. Collaborative quest completion with llm-driven non-player characters in minecraft. arXiv preprint arXiv:2407.03460, 2024.

[21] Stefan Kopp, Bernhard Jung, Nadine Lessmann, and Ipke Wachsmuth. Max-a multimodal assistant in virtual reality construction. KI, 17(4):11, 2003.

[22] Ziming Li, Huadong Zhang, Chao Peng, and Roshan Peiris. Exploring large language model-driven agents for environment-aware spatial interactions and conversations in virtual reality roleplay scenarios. In 2025 IEEE Conference Virtual Reality and 3D User Interfaces (VR), pages 1–11. IEEE, 2025.

[23] Alessandro Emmanuel Pecora, Francesco Strada, and Andrea Bottino. A survey of memory models for virtual agents and humans: From psychological foundations to computational architectures. In International Conference on Extended Reality, pages 113–133, 2025.

[24] Yang Liu, Weixing Chen, Yongjie Bai, Xiaodan Liang, Guanbin Li, Wen Gao, and Liang Lin. Aligning cyber space with physical world: A comprehensive survey on embodied ai. IEEE/ASME Transactions on Mechatronics, 2025.

[25] Manoj Ramanathan, Nidhi Mishra, and Nadia Magnenat Thalmann. Nadine humanoid social robotics platform. In Computer Graphics International Conference, pages 490–496. Springer, 2019.

[26] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020.

[27] Jae-Woo Choi, Youngwoo Yoon, Hyobin Ong, Jaehong Kim, and Minsu Jang. Lotabench: Benchmarking language-oriented

task planners for embodied agents. arXiv preprint arXiv:2402.08178, 2024.

[28] Arjun Majumdar and et al. Openeqa: Embodied question answering in the era of foundation models. In IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024.

[29] Robin R Murphy. Introduction to AI robotics. MIT press, 2019.

[30] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[31] OpenAI. Chat completions api and chatml message format. https://platform. openai.com/docs/guides/chat, 2023. Accessed: 2025-11-12.