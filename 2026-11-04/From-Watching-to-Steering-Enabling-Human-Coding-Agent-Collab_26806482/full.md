# From Watching to Steering: Enabling Human–Coding Agent Collaboration through Execution Traces

Xiaoke Yan, Jiaying Lei, Danqing Shi, Zichen Wang, and Nan Cao

![](images/388d6ab07822ee9bbf00f7eb4f1578751dc3cf503da95d7d7a84c83424337eeb.jpg)  
Fig. 1. VibeTrace enables a continuous human–coding agent collaboration loop. The user first assigns a coding task, after which the agent executes it while VibeTrace records and visualizes the execution trace. By inspecting the trace and step details, the user can observe and understand the agent’s behavior, identify errors, and intervene by forking the execution and providing feedback. VibeTrace then distills successful human interventions into reusable skills and updates the skill pool, allowing the evolved agent to apply learned knowledge to subsequent tasks. This trace–feedback–skill loop transforms transient human interventions into persistent collaboration knowledge and supports continuous agent improvement. The source code is available at https://github.com/idvxlab/VibeTrace.

Abstract—Recent advances in large language models (LLMs) have enabled coding agents to autonomously perform complex software development tasks, yet their long-running execution remains difficult for users to inspect, understand, and steer. We present VibeTrace, an interactive trace visualization and intervention system for agentic programming built upon OpenCode. VibeTrace introduces a human– agent collaboration paradigm of Observe, Understand, Intervene, and Evolve. It (1) visualizes agent workflows as interactive execution traces for inspecting actions, intermediate states, and failures; (2) enables users to fork execution from prior steps and provide targeted corrections without restarting validated progress; and (3) distills traces and human feedback into reusable skills for continual agent evolution. On SWEbench Verified, distilled skills reduce input tokens by 23.0% and cost by 15.5%, while improving pass rate from 65.0% to 68.0%. User studies further demonstrate improved accuracy and efficiency over conventional log-based interaction.

## 1 INTRODUCTION

ECENT advances in large language models (LLMs) have such as Claude Code [1], OpenAI Codex [2], and Open-Code [3]. By integrating LLMs with external tools, file systems, terminals, and execution environments, these systems can autonomously plan tasks, generate code, execute commands, debug failures, and iteratively refine solutions [4], [5]. As a result, coding agents have expanded developers’ ability to accomplish complex software tasks through natural language interaction.

Despite these advances, coding agents largely follow an autonomous execution paradigm: users specify highlevel goals while agents independently perform multi-step workflows. This improves efficiency, but shifts users from active problem solvers to passive observers. On complex tasks, developers must still verify the agent’s direction, diagnose failures, and correct deviations [6]–[8]. Existing systems, however, expose mainly final outputs or linear interaction logs, offering limited support for inspection, diagnosis, and steering. Interaction is also append-only: users cannot re-enter execution at an earlier step, nor see the branch structure. Successful corrections and workflow adjustments are rarely captured as reusable knowledge, limiting improvement across tasks.

We conducted a formative study with 12 developers who regularly use Claude Code, Codex, Cursor, and OpenCode, identifying the following key for human—agent collaboration in coding tasks: Execution transparency: providing reliable traces that allow users to observe agent actions, decisions, and intermediate states; Failure understanding: revealing causal relationships between execution steps and outcomes to support effective diagnosis; Interactive intervention: enabling users to correct specific execution steps without restarting entire workflows; Experience reuse: transforming human interventions and execution traces into reusable knowledge artifacts, enabling coding agents to continuously improve through accumulated collaboration experiences.

To address these challenges, we present VibeTrace, an interactive trace-based system built upon OpenCode [3] that enables users to observe, understand, and steer coding agents while supporting their continual evolution. As illustrated in Fig. 1, VibeTrace establishes an iterative human– agent collaboration loop of Observe, Understand, Intervene, and Evolve through three key mechanisms: (1) structured trace visualization that makes agent execution processes and dependencies inspectable; (2) trace-level intervention that allows users to redirect execution from prior steps while preserving validated progress; and (3) online trace-to-skill distillation that transforms execution traces and human feedback into reusable skills for future tasks. We evaluate VibeTrace through benchmark experiments, a controlled user study, and expert interviews. Results on SWE-bench Verified [9] demonstrate improved task performance and reduced execution costs, while the controlled study shows that trace visualization improves execution understanding and error diagnosis compared to conventional logs. Expert interviews further highlight the system’s usability in real-world coding tasks. The contributions are summarized as follows:

• We present VibeTrace, an interactive trace intervention system that instantiates a new human–agent collaboration paradigm from Observe to Understand, Intervene, and Evolve. Through execution visualization, trace-level intervention, and experience reuse, VibeTrace transforms coding-agent interaction from passive monitoring of autonomous execution into active steering and continual collaboration.

• We introduce a visualization-oriented trace interaction paradigm that makes agent execution processes observable and actionable. By representing agent workflows as structured execution traces, VibeTrace enables users to inspect agent trajectories, understand failures, and perform fine-grained interventions during execution rather than relying on post-hoc artifact modification.

• We propose a human-in-the-loop trace-to-skill distillation mechanism that incrementally transforms execution traces and human interventions into reusable collaboration knowledge. By capturing users’ corrections, preferences, and problem-solving strategies, the mechanism enables coding agents to continuously adapt to individual workflows, improve task performance, and better align with user intentions over time.

## 2 RELATED WORK

For background, we begin by investigating how people program with AI coding agents. Then, we review related work on agent behavior visualization, intervention, and selfevolution. We highlight the research gap.

## 2.1 Programming with AI Coding Agents

“Vibe coding”, originally introduced by Andrej Karpathy [10], describes an emerging programming style in which developers collaborate with AI coding agents through natural-language interaction, code generation and iterative refinement to the agent. Such AI coding agents now support a wide range of programming tasks, including explaining code, generating tests, fixing bugs, and offering in-context code suggestions [11]. Rather than producing snippets, modern coding agents participate in multi-step developer workflows, translating high-level intent into code, synthesizing tests, and iteratively refining solutions [4], [5]. This shift has been shown to reshape software development practices and to improve developer efficiency and productivity, particularly for novice programmers [12]. Prior work further shows that developers adopt different interaction strategies depending on their familiarity with the task and the complexity of the code [13]. With an appropriate strategy, developers can more effectively leverage AI coding agents to improve their programming experience and outcomes.

However, this new paradigm also introduces challenges related to understanding, trust, and control in AI-generated outputs [7]. Developers often find it difficult to follow the reasoning behind AI-generated code, which complicates debugging and long-term maintenance [6]. This situation highlights the need for more transparent and interpretable agents that can effectively communicate their decisionmaking processes to developers. Additionally, the existing abstraction gap between users and the underlying models hampers effective usage, while limited transparency and buggy outputs can lead to frustration and decreased trust in AI coding agents [14], [15]. These challenges highlight the importance of designing systems that provide developers with a better understanding and control over agent behavior [16]. Our work builds on this area of research by exploring agent execution traces as an interaction medium, allowing users to transition from passively monitoring agent behavior to actively steering it.

## 2.2 Agent Behavior Visualization and Intervention

Visualization has long been used to make AI behavior observable, from reinforcement learning agents [17] to large language models [18]. Building on visualizations, interactive systems let users conduct inspection and shape agent behavior, for example by directly tuning reward functions [19], or elicit human feedback for behavioral alignment [20], [21]. As LLM agents take on increasingly long-horizon tasks, the execution traces the agents produce become an object of study. One line of work makes execution traces more understandable. Explainable-AI techniques transform raw traces into structured, human-interpretable accounts of coding-agent failures [22], and multi-agent pipelines such as TraceSIR distill executions into reports that summarize what an agent did and why [23]. AgenTracer goes further by attributing an observed failure to the specific step responsible in a multiagent system [24]. A second line of work targets trust in what a trace reports. VET shifts the trust anchor for agent outputs to the verifiable definition of the agent itself [25]. EnvTrace assesses generated code by aligning its execution against a reference run in simulation [26].

The existing efforts in execution trace analysis make agent behavior more observable and easier to trust, but they treat the trace as a record to be read rather than as an ongoing process to steer. There is still a gap remains between observing an execution and steering it, even though where and how an agent is steered strongly shapes the outcome of a run [27]. When an agent begins to deviate from the user’s goal, the available way is typically to abort that execution and restart the run. In contrast, VibeTrace treats the execution trace as an interactive artifact: users can inspect agent behavior, follow dependencies among execution steps, and intervene at the level where a problem arises rather than only at the end of the run.

## 2.3 Agent Self-Evolution

As LLM agents operate in complex environments, they increasingly require mechanisms for accumulating experience and improving future behavior across tasks. This process is often referred to as agent self-evolution [28], [29], where past interactions are distilled into reusable knowledge that can be stored, retrieved, and refined over time. Voyager [5] provides an early example by continuously acquiring executable code skills from successful trajectories in Minecraft, while ReasoningBank [30] extends this idea by distilling both successful and failed experiences into structured reasoning memories for test-time self-evolution. A central form of such reusable knowledge is the agent skill. Anthropic’s Agent Skills instantiate this abstraction in practice by packaging task-specific instructions, scripts, and resources into structured, retrievable modules [31]. However, a static skill set is rarely sufficient: as agents encounter new tasks and failures, their skill repertoire must be continuously updated. This gives rise to skill evolution, where execution experience is transformed into long-lived and reusable skill artifacts.

Recent work explores different mechanisms for skill evolution. MUSE-Autoskill [32] organizes skill management into a lifecycle of creation, memory, management, evaluation, and refinement. CODESKILL [33] learns a skillmanagement policy for coding agents, deciding when to extract, evolve, or prune skills from trajectories. SkillOpt [34] optimizes natural-language skills through trajectory-driven edits and validation-gated updates, while Skill-Pro [35] formalizes reusable skills with activation, execution, and termination conditions and maintains them through nonparametric policy optimization. Trace2Skill [36] takes a batch distillation view, consolidating lessons from multiple execution traces into transferable skill directories. Memento-Skills [37] and SkillEvolver [38] further show that externalized skills can be iteratively written, deployed, and refined without updating model parameters.

Despite this progress, most existing methods remain primarily offline or self-supervised: they rely on accumulated trajectories, model-judged outcomes, or validation feedback to update skills after execution. Recent human-in-the-loop systems begin to incorporate user feedback. AgentClick [39] provides structured interfaces for reviewing and correcting terminal agents, and TRACE [40] compiles user corrections into runtime enforcement rules. However, these approaches either treat feedback as memory or enforceable constraints, rather than using interactive execution traces as direct evidence for evolving a shared skill representation. Our work addresses this gap by performing online trace-to-skill distillation during human-agent collaboration. By converting trace-level interventions, corrections, and preferences into structured skill updates, VibeTrace enables coding agents to evolve reusable skills that are temporally grounded in execution and better aligned with human intentions.

## 3 DESIGN OVERVIEW

This section presents the design process of VibeTrace, which aims to transform coding-agent interaction from passive observation into active collaboration. We first conduct a formative study to understand how developers monitor, interpret, and intervene in agent execution during realworld coding tasks. Based on the identified findings, we derive a set of design implications that guide the design of VibeTrace. Finally, we introduce the system design, which operationalizes these principles into a trace-based human– agent collaboration framework.

## 3.1 Formative Study

To understand how users collaborate with AI coding agents beyond task completion, we conducted a formative study focusing on how users observe, diagnose, and steer agent execution in real-world software development. Our analysis revealed a fundamental gap between watching and controlling agent behavior. While users increasingly rely on execution traces to understand agent actions, they often struggle to trust process representations, diagnose failures, intervene at appropriate levels of granularity, and preserve successful intervention experiences.

Participants and Method. We conducted 40-minute semi-structured interviews with 12 experienced users of coding agents to understand how developers monitor, diagnose, and intervene in agent execution. Participants averaged 4.75 years of professional experience: eight software engineers, two product managers, and two UI/UX designers. All had used coding agents for over a year, regularly working with Claude Code, Codex, Cursor, and OpenCode.

The interview protocol focused on three stages of human-coding agent collaboration: (i) process monitoring, (ii) execution review, and (iii) intervention. Participants were asked to describe when and why they monitored agent execution, what information they relied on to evaluate agent behavior, and how they responded when execution diverged from their expectations. Follow-up questions probed recent examples, decision rationales, and breakdown situations encountered during real tasks. All interviews were transcribed and analyzed using thematic analysis [41]. Through iterative coding and constant comparison, we identified recurring themes concerning execution transparency, causal understanding, intervention practices, and knowledge reuse. These themes revealed a progression from process awareness to active steering and informed the design requirements for our trace intervention framework.

Findings. Our analysis identified five interconnected findings that reveal challenges in human–coding agent collaboration and motivate a shift from passive observation to active steering and continuous adaptation.

F1 Users need visibility into agent execution beyond final outcomes. Participants sought to understand how agents plan, act, and recover, rather than merely inspecting final artifacts. Monitoring practices varied with task complexity and confidence: most participants reviewed execution after completion, while others monitored intermediate steps for ambiguous or high-risk tasks. This highlights the need for flexible representations supporting different levels of oversight.

F2 Existing agent traces provide limited evidence for trustworthy understanding. Participants valued information about task decomposition, tool usage, and intermediate states but questioned the reliability of agent-generated summaries. Existing traces often describe claimed actions without verifiable evidence, making them difficult to trust. Participants therefore preferred representations grounded in observable actions and execution states.

F3 Users lack support for understanding failure causality. Participants found that agents could report errors but rarely explained their underlying causes. Existing logs expose isolated symptoms, such as failed commands or compilation errors, without revealing dependencies among plans, actions, intermediate states, and outcomes. Users must therefore reconstruct failure causes through additional debugging, highlighting the need for tracebased causal diagnosis.

F4 Users require fine-grained intervention to steer agent trajectories. Participants found existing correction mechanisms too coarse-grained when agents deviated from expectations. Continuing execution risked propagating errors, while restarting discarded validated progress. They preferred intervening at specific execution steps to redirect agent behavior while preserving successful work, highlighting the need for trajectory-level steering.

F5 Human interventions should be transformed into reusable agent skills. Participants frequently encountered recurring workflows, preferences, and debugging strategies but rarely retained them across tasks. Manual maintenance left valuable knowledge embedded in transient interactions, forcing users to repeat similar corrections. This motivates transforming trace-based interventions into persistent, reusable agent skills.

## 3.2 Design Implications: From Watching to Steering

Our findings reveal a progression from observing agent behavior to actively steering and improving future collaborations. Based on this progression, we derive four design goals that establish an iterative human–agent collaboration loop: Observe execution through trustworthy traces, Understand behavior through causal reasoning, Intervene by redirecting execution trajectories, and Evolve future collaborations through reusable intervention knowledge.

DG1 (Observe): Provide trustworthy and navigable representations of agent execution. To support monitoring and build confidence in agent behavior (F1–F2), execution should be represented through observable actions rather than agent-generated narratives. Execution traces should capture key artifacts, including task decomposition, planning, tool use, code changes, and outcomes. Given the complexity of coding workflows, traces should be organized hierarchically, allowing users to seamlessly navigate between high-level workflows and low-level execution details while maintaining situational awareness without excessive cognitive load.

(Understand): Support causal diagnosis through tracebased reasoning. To help users explain execution outcomes and diagnose failures (F3), the system should expose causal relationships among actions, intermediate states, tool calls, and outcomes rather than isolated events. Making these dependencies explicit enables users to understand why failures occur, verify agent reasoning, and identify intervention opportunities.

(Intervene): Enable trajectory-level steering through localized trace intervention. To support corrective actions without discarding successful progress (F4), execution traces should serve as interactive control surfaces rather than immutable records. Users should be able to inspect, modify, replace, or reject individual actions and immediately understand their downstream effects, enabling precise trajectory-level steering while preserving validated execution.

DG4 (Evolve): Enable coding agents to improve through execution traces and human interventions. Coding tasks often involve recurring workflows and failure patterns, yet current agents rarely learn from prior interactions (F5). The system should transform human interventions into reusable skills that capture user preferences and effective strategies, enabling agents to continuously improve future executions, avoid repeated mistakes, and better align with user expectations.

## 3.3 System Design

To achieve the above goals, we designed the VibeTrace system by following the architecture shown in Fig. 2. Vibe-Trace introduces a trace-based human–agent collaboration framework that transforms coding-agent execution from an opaque process into an observable, steerable, and continuously evolving workflow. The system consists of three core components: (1) Trace Extraction module, which captures agent execution as structured traces, enabling users to observe agent workflows and understand execution states (DG1, DG2); (2) Interactive Trace Visualization and Intervention module, which transforms traces into interactive representations that support inspection, diagnosis, and finegrained trajectory steering during execution (DG2, DG3); and (3) Online Trace-to-Skill Distillation module, which converts execution traces and human interventions into reusable skills, allowing agents to continuously improve through accumulated collaboration experiences (DG4). In the following section, we present the technical details of each module.

![](images/32fa4b10b2bff9d4945b181e63bc6e5c6f14f234f7da909f30a1bcedea9cc8f3.jpg)  
Fig. 2. System framework and running pipeline. VibeTrace transforms opaque agent execution into interactive traces that support workflow inspection, fine-grained intervention, and continuous skill evolution.

## 4 TECHNIQUE DETAILS

To enable the transition from observing agent execution to actively steering and improving agent behaviors, we further detail the underlying techniques of VibeTrace. This section introduces the trace construction mechanism, interactive trace-based intervention approach, and online trace-to-skill distillation algorithm that together support human-guided coding agent evolution.

## 4.1 Execution Trace Extraction

The Trace Extraction module transforms coding-agent execution into structured traces that capture the complete workflow beyond final outputs. During task execution, VibeTrace records agent activities, including planning, information acquisition, code modification, user interaction, skill utilization, and execution management, as a unified trace representation. Rather than treating execution logs as isolated events, VibeTrace organizes agent behaviors into structured trace units that preserve temporal order, intermediate states, and execution dependencies. Specifically, we define an execution trace as a sequence of trace events:

$$
\tau = \{ e _ { 1 } , e _ { 2 } , . . . , e _ { n } \} , \quad e _ { i } = ( a _ { i } , s _ { i } , o _ { i } , t _ { i } , r _ { i } )\tag{1}
$$

where $a _ { i }$ represents the executed action, $s _ { i }$ denotes the corresponding execution state, $o _ { i }$ captures the generated output or intermediate artifact, $t _ { i }$ records the execution time, and $r _ { i }$ represents the resulting state after the action.

Based on the execution schema of OpenCode, VibeTrace extracts 12 primitive actions grouped into seven categories. The first category includes planning actions, which capture agent planning and decision-making processes through LLM invocation. The second category contains information acquisition actions, including reading and searching operations that collect contextual information from files, code repositories, or external resources. The third category consists of code modification actions, which record writing operations that update source code or other development artifacts. The fourth category includes communication actions, including responding, clarifying, and permission requests, where the agent communicates intermediate results or involves users to resolve ambiguity and obtain authorization. The fifth category contains skill management actions, including skill routing and skill loading, which record the retrieval and application of reusable procedural knowledge during execution. The sixth category includes execution actions, such as shell execution and subagent invocation, representing external process execution and task delegation. The final category is context management actions, such as compaction operations that compress execution history when context limitations are reached.

## 4.2 Interactive User Interface

Fig. 3 illustrates the interactive interface of VibeTrace. Unlike conventional chat-based coding agents that expose execution as linear conversation logs, VibeTrace treats execution traces as the primary interaction medium through which users observe, understand, and steer agent behaviors. The interface comprises five coordinated views: a Project Panel for project management (Fig. 3(a)), a Session Panel for interaction history (Fig. 3(b)), a Dialog Panel for naturallanguage communication (Fig. 3(c)), a Trace Panel for trace representation and interaction (Fig. 3(d)), and a Skill Panel for displaying reusable skills distilled from previous executions (Fig. 3(e)). Among them, the Trace Panel serves as the primary workspace where users inspect and steer agent execution based on an interactive trace visualization which is detailed in the next.

![](images/d696f2dbdf98d4e92f83e1340def519e3b9b2a36073607cded9ba9ba9a32b78e.jpg)  
Fig. 3. Interactive user interface.

## 4.2.1 Visual Trace Representation

To improve execution transparency (DG1) and facilitate users’ understanding of agent behaviors (DG2), VibeTrace organizes execution traces hierarchically, enabling users to progressively inspect agent execution from task-level workflows to individual execution actions.

Task-level Organization. Each coding task is presented as an independent tab (Fig. 3(d-1)), corresponding to a single user request. Instead of visualizing the entire execution as one continuous timeline, VibeTrace further decomposes each task into a sequence of trace cards, each representing a subtask or sub-agent execution. This organization preserves workflow context while preventing long-running executions from becoming difficult to inspect.

Trace Cards. Trace cards are the fundamental representation unit in VibeTrace (Fig. 4). Each card summarizes one execution stage through two complementary components: an Execution Flow (Fig. 4(a-1)) that explains how the agent performs the task and an Execution Summary (Fig. 4(a-2)) that characterizes the corresponding execution cost. Together, they enable users to understand both the execution process and its efficiency:

Execution Flow. The execution flow visualizes the temporal trajectory of agent behaviors as a directed workflow beginning with the user request and ending with the execution outcome. Intermediate nodes represent primitive agent actions extracted in Section 4.1, while directed edges encode temporal and causal dependencies between actions. Each node visually encodes key execution attributes, with semantic icons representing action types and visual variables (i.e., node size and opacity) indicating execution duration (Fig. 4(b)) and token consumption (Fig. 4(c)), respectively. Failed actions are highlighted in red (Fig. 4(a-3)) to facilitate rapid identification of execution failures. To show execution revisions, VibeTrace represents redirected executions as alternative branches that diverge from the selected action while preserving the shared execution history (Fig. 4(a-4)). By presenting the original and redirected trajectories within the same execution graph, users can directly compare their behavioral differences, understand how interventions propagate through subsequent execution steps, and evaluate alternative execution outcomes without losing previously validated execution context.

Execution Summary. Complementing the execution flow, each trace card presents aggregated execution statistics, including runtime, token consumption, monetary cost, and the number of LLM invocations. These summaries provide an at-a-glance overview of execution efficiency, enabling users to rapidly identify computationally expensive or abnormal executions before inspecting detailed traces.

## 4.2.2 Trace Interaction

Beyond representing execution, VibeTrace transforms execution traces into interactive control surfaces that support both execution inspection and trajectory-level intervention, enabling users to progressively steer agent behaviors (DG3).

Inspection. Inspection supports a progressive exploration of agent execution through four complementary interaction techniques, enabling users to move from observing overall execution to inspecting individual actions: (i) Streaming. Execution traces are streamed incrementally as the agent executes, allowing users to monitor execution progress in real time and promptly detect unexpected behaviors.

![](images/9cf709c43be87a6c3d687b0047bfdd83dcbc5bff390ef234f1644791a4fc43d0.jpg)  
Fig. 4. Visualization of an execution shown in a trace card.

(ii) Cross-view Linking. After identifying an execution stage of interest, users can select a trace card or an action node to automatically link to the corresponding conversation and execution records in the Dialog Panel (Fig. 3(d-3)), enabling seamless navigation between high-level workflows and lowlevel execution details. (iii) Cost Filtering. To further narrow the inspection scope, users can filter execution nodes by runtime or token consumption, allowing computationally expensive or abnormal execution steps to be quickly located in long-running tasks. (iv) Contextual Inspection. Finally, hovering over a trace element reveals contextual information, including action parameters, execution outputs, and resulting states, enabling detailed inspection without disrupting the overall workflow visualization. Together, these interaction techniques support a coarse-to-fine exploration of execution traces, helping users efficiently develop a causal understanding of agent behaviors.

Intervention. Building upon inspection, VibeTrace enables users to progressively steer agent execution through three complementary intervention techniques spanning different levels of granularity. (i) Comment. Users can annotate individual execution traces with preferences, observations, or issues, providing localized feedback that is subsequently incorporated into trace-to-skill distillation. (ii) Redirect. When localized feedback is insufficient, users can fork execution from any previous action node and continue along a revised trajectory. By reusing all validated execution before the fork point, Redirect enables efficient correction without restarting the entire workflow. (iii) Outcome Correction. After task completion, users can provide feedback on the final execution outcome, complementing step- and trajectory-level interventions with task-level supervision. Together, these intervention techniques transform execution traces from passive records into editable execution trajectories, enabling users to iteratively refine agent behaviors while preserving previously validated progress.

The resulting interaction traces, together with users’ explicit interventions, serve as the foundation for the traceto-skill distillation mechanism described in the next section.

## 4.3 Online Trace-to-Skill Distillation

Trace-based visualization and intervention enable users to actively steer coding agents during individual tasks. However, such collaboration knowledge is often transient: successful corrections, debugging strategies, and workflow preferences disappear once a session ends. Consequently, the agent repeatedly makes similar mistakes even when users have already demonstrated better solutions. VibeTrace addresses this limitation by transforming human interventions into reusable skills that continuously improve future executions. The key insight is that user feedback should not be memorized as execution trajectories. Instead, it provides evidence about how the agent should behave under similar situations. For example, when a user redirects an execution to inspect the repository before modifying code, the reusable knowledge is not the corrected trajectory itself, but the behavioral strategy of inspecting existing implementations before editing.

Based on this insight, VibeTrace models skill evolution as a human-guided knowledge refinement process. As illustrated in Fig. 5, each interaction experience goes through three sequential stages: Experience Estimation determines whether an interaction contains reusable knowledge. Skill Patch Distillation extracts transferable behavioral changes from accepted interactions. Skill Patch Validation evaluates candidate patches and incorporates only high-quality ones into the evolving skill pool. Together, these stages continuously transform transient human interventions into persistent collaboration knowledge without modifying the underlying foundation model.

Evolution Objective. We model the coding agent as

$$
H _ { \Omega } = ( H _ { 0 } , \Omega ) ,\tag{2}
$$

where $H _ { 0 }$ denotes the fixed foundation model together with its tools and execution environment, while Ω represents the evolving skill pool. Given a task stream $\{ x _ { 1 } , x _ { 2 } , \ldots \}$ , after completing task $x _ { t } ,$ the agent updates the skill pool using the execution trace $\tau _ { t }$ and the associated user feedback $f _ { t } { \mathrm { : } }$

$$
\Omega _ { t + 1 } = \mathrm { U p d a t e } ( \Omega _ { t } , \tau _ { t } , f _ { t } ) .\tag{3}
$$

Unlike model fine-tuning, VibeTrace never modifies the parameters of $H _ { 0 }$ . Instead, it achieves continual adaptation by incrementally refining the external skill pool. The objective of evolution is to improve future task quality while reducing execution costs. Let M = {token, time, call} denote the three execution-cost dimensions. The overall utility of a skill pool is defined as

$$
J ( \Omega ) = \mathbb E _ { x \sim \mathcal X } \left[ Q ( \tau _ { x } ^ { \Omega } ) \prod _ { m \in \mathcal { M } } ( 1 - \bar { c } _ { m } ( \tau _ { x } ^ { \Omega } ) ) \right] ,\tag{4}
$$

where $\tau _ { x } ^ { \Omega }$ is the execution trace produced by the agent based on skill pool Ω on task x; $\dot { Q } ( \tau _ { x } ^ { \Omega } ) \in [ 0 , \dot { 1 } ]$ measures task quality using user feedback and available environment verification, and $\bar { c } _ { m } ( \tau _ { x } ^ { \Omega } ) \ \in \ [ 0 , 1 ]$ denotes the normalized execution cost along dimension m. Intuitively, Equation (4) rewards skill pools that simultaneously improve task quality while reducing token consumption, execution time, and tool usage. Rather than optimizing this objective directly, VibeTrace approximates it through the three-stage evolution pipeline described below.

![](images/996414d82d927b007bd42dc1e10bb321c8414efc54f8125d942ad7d10b2a8294.jpg)  
Fig. 5. Overview of the online trace-to-skill distillation process in VibeTrace. Execution traces and human feedback are analyzed to extract reusable skill patches, which are evaluated and integrated into the skill pool to continuously improve future agent behaviors.

Experience Estimation. Not every user intervention should become a reusable skill. Some interactions are highly task-specific, ambiguous, or even misleading. Learning from such experiences may negatively affect future executions. Therefore, before generating any skill, VibeTrace first estimates whether an interaction contains reusable evolution signals. Specifically, each trace-feedback pair is evaluated from three complementary perspectives: (i) Reusability: can the observed behavior generalize beyond the current task? (ii) Informativeness: does the feedback imply a concrete behavioral improvement? (iii) Safety: can the behavior be reused without violating user constraints or introducing undesirable side effects? To this end, VibeTrace introduces an LLM-based estimator Estimate(·) to predict the reusability, informativeness, and safety scores from the trace $\tau _ { t }$ and user feedback $f _ { t } { \mathrm { : } }$

$$
\mathbf { z } _ { t } = \mathrm { E s t i m a t e } ( \tau _ { t } , f _ { t } ) \in [ 0 , 1 ] ^ { 3 } ,\tag{5}
$$

Skill Patch Distillation. After an intervention passes the estimation stage, VibeTrace distills the observed behavior into reusable knowledge. Instead of memorizing the corrected execution trace, the distillation process extracts the behavioral difference between the original and corrected executions. A reusable skill should answer three questions: (i) When should the behavior be activated (Context)? (ii) What behavior should change (Decision)? (iii) How should future execution proceed (Strategy)? For example, when a user redirects the agent to inspect the repository before modifying code, the distilled skill records the reusable strategy rather than the corrected trajectory itself. The distillation process generates a set of candidate skill patches:

$$
\mathcal { Q } _ { t } = \mathrm { D i s t i l l } ( \tau _ { t } , f _ { t } , \Omega _ { t } ) ,\tag{6}
$$

where Distill(·) is implemented using a large language model. Rather than producing standalone skills, each candidate $q = ( \mathrm { o p } , \psi , \pi ) \bar { \bf \Phi } \in \mathcal { Q } _ { t }$ represents a skill patch—a structured modification to the current skill pool. Here, op is one of four update operations: Create (introduce a new skill), Revise (update an existing skill), Merge (consolidate overlapping skills), or Remove (discard obsolete or conflicting knowledge); ψ encodes the distilled behavioral knowledge (context, decision, and strategy); and π records the originating execution trace and user feedback for provenance.

Skill Patch Validation and Skill Pool Evolution. Although estimator-approved interactions provide reliable evidence, the resulting skill patches may still be overly specialized, redundant, or offer limited practical benefit. Before updating the skill pool, VibeTrace uses an LLM-based scorer to evaluate each patch through quality assessment followed by efficiency estimation.

For a patch q, the scorer first predicts $\mathbf { h } ( q ) \in \mathbf { \phi } [ 0 , 1 ] ^ { 2 }$ representing its expected reusability and correctness. Only patches satisfying the quality thresholds α are retained. For each qualified patch, the scorer further predicts its expected efficiency improvement ${ \pmb \rho } ( q ) \in [ - 1 , 1 ] ^ { 3 }$ over token consumption, execution time, and tool calls, where positive values indicate expected cost reduction and negative values indicate additional overhead. Quality and efficiency are jointly combined into the patch score:

$$
\mathrm { S c o r e } ( q ) = \mathbb { I } ( { \mathbf { h } } ( q ) \succeq \alpha ) \left( \prod _ { m \in { \cal { M } } } \operatorname* { m a x } ( 0 , \rho _ { m } ( q ) ) \right) ^ { 1 / 3 } ,\tag{7}
$$

where the quality gate ensures that efficiency gains cannot compensate for unreliable knowledge, while the geometric mean prioritizes patches that consistently reduce execution costs across all three dimensions. Finally, patches are ranked by their scores, and the highest-scoring updates are selected to modify the skill pool:

$$
\begin{array} { r } { S _ { t } = \mathrm { T o p K } _ { K } \left\{ q \in \mathcal { Q } _ { t } \vert \mathrm { S c o r e } ( q ) \geq \eta \right\} . } \end{array}\tag{8}
$$

$$
\Omega _ { t + 1 } = \Omega _ { t } \oplus S _ { t } ,\tag{9}
$$

where ⊕ denotes applying the selected create, revise, merge, and remove operations. $\mathrm { I f } \ : S _ { t } = \emptyset .$ , the skill pool remains unchanged. Rather than continuously accumulating new skills, VibeTrace maintains the skill pool through iterative refinement. By repeatedly creating, revising, merging, and removing skills, transient human interventions gradually become persistent collaboration knowledge that guides future executions. Consequently, the coding agent evolves through an increasingly capable human-guided skill pool instead of modifying the parameters of its foundation model.

## 5 EVALUATION

We first provide a quantitative evaluation of the online trace-to-skill algorithm outlined in Section 4.3 in an opensource coding benchmark, and then a user study for evaluating the effectiveness of the visualization design. Last, we present a user study with eight experts to evaluate VibeTrace in real-world coding tasks and demonstrate usage cases.

## 5.1 Skill-Based Agent Evolving Mechanism

We conduct a quantitative evaluation of the proposed online trace-to-skill distillation mechanism to highlight the VibeTrace’s ability of evolving coding agents through tracebased interaction (DG4). Specifically, we ask whether skills distilled online from prior tasks improve accuracy and efficiency on subsequent related tasks, and whether online distillation from user feedback provides additional benefit over learning from execution traces alone.

Dataset and Metrics. We use SWE-bench Verified [9], a human-validated benchmark of real-world GitHub issues with executable test suites. To study skill transfer across related tasks, we cluster tasks by repository and subsystems, inferred from the files modified by the reference patch, and split each cluster into distillation and held-out test tasks. We then sample tasks from each retained difficulty, prioritizing distillation tasks from clusters represented in the test set. The result data contains 30 distillation tasks and 20 held-out tasks. Skill evolution starts from an empty pool, updates it sequentially over the distillation stream, and then freezes the learned pool for evaluation on the held-out set.

We consider the metrics regarding both task completion and resource consumption of AI, including:

• Pass rate : The proportion of tasks accepted by the official SWE-bench verifier.

• Input tokens: The total number of cached input tokens, read tokens, and write tokens.

• Output tokens: This includes both generated response tokens and reasoning tokens.

• Cost: This metric translates the different token categories into actual financial cost.

• Time: The duration taken to execute a task.

• The overall utility J(Ω) (Equation 4): Its quality term assigns unit quality to a verifier-accepted trace and zero quality otherwise, while its token-consumption term uses Cost to account for the different prices of uncached input, cached input, and output tokens.

• Tool calls: The count of tool invocations during the task.

Baselines. We compare our method with two baselines using the same coding agent and evaluation harness. Base has no distilled skills, so the agent has no persistent experience for new tasks. Trace-only evolves skills from execution traces without user feedback, updating the skill pool after each task with the same skill-distillation framework as our method. Our method, Trace-fb, simulates two forms of feedback using the benchmark verifier. For Redirect, when the verifier marks the original execution as unresolved, we fork the trace with the user hint Locate the root cause more carefully and make a minimal, correct fix; avoid unrelated, broad changes.”. If the verifier marks the fork as resolved, the abandoned and corrected branches form a pairwise feedback. For Outcome Correction, we append I’ve reviewed the result; the requested behavior is resolved.” to a trace accepted by the verifier and the correction above to a rejected trace. We do not additionally involve Comment as it follows the same processing as Outcome Correction.

Implementation. We use DeepSeek-V4-Flash [42] as the shared LLM backbone for task execution and all three skilldistillation stages. Each task starts from a clean repository checkout with isolated OpenCode state. External skills are disabled, and verification uses the official Docker-based SWE-bench harness. The estimator, distiller, and scorer are each executed in a separate one-shot OpenCode session, isolated from the task-execution session and from one another, to prevent conversational state from leaking across modules or tasks. The estimator thresholds for reusability, informativeness, and safety are 0.34, 0.50, and 0.60. Candidate patches must reach 0.50 in both reusability and correctness, predict positive savings in all three resource dimensions, and obtain an aggregate score of at least 0.10. At most the top three patches are applied per task. The cost/time/tool-call budgets are (0.02, 600, 60), (0.04, 900, 90), and (0.05, 1200, 100) for the three retained difficulty levels. Trace-only and Trace-fb evolve separate skill pools under the same settings.

Results. Table 1 shows that our method performs best in all seven metrics. It improves both quality and efficiency. Compared to Base, our approach increases the pass rate from 65.0% to 68.0%, reduces input tokens by 23.0%, and lowers cost and time by 15.5% and 14.5%.These improvements result in a 14.0% enhancement in J(Ω). Furthermore, our method consistently surpasses Trace-only, which only focuses on improving efficiency. This outcome indicates that feedback not only enhances quality by utilizing human corrections as signals but also supports more targeted skillpool updates for better efficiency.

## 5.2 The Trace Visualization Design

To evaluate whether VibeTrace helps users better understand agent execution processes and locate errors, we conducted a controlled study comparing trace visualizations from VibeTrace with OpenCode’s native logs.

Participants. We recruited 20 participants with codingagent experience (age: M=24.1, SD=3.7; 13 females). Participants were randomly assigned to two groups, each completing one of the two questionnaires.

TABLE 1  
Held-out SWE-bench Verified results over 20 tasks and five repeated runs. The split and stream order use seed 42. Values are mean ± population standard deviation. Tokens are in thousands and cost is in $1 0 ^ { - 3 } \$ 8 .$ Parentheses in the input-token column show mean cache-read tokens.
<table><tr><td>Method</td><td> $\mathrm { P a s s \ ( \% ) \uparrow }$ </td><td> $\mathrm { I n - T o k . } \ ( \mathrm { C a c h e d } ) \downarrow$ </td><td> $\mathrm { O u t - T o k . \downarrow }$ </td><td> $\mathrm { C o s t } \left( 1 0 ^ { - 3 } \$ 5 \right) \downarrow$ </td><td> ${ \mathrm { T i m e } } \left( s \right) \downarrow$ </td><td> $\mathrm { C a l l s \downarrow }$ </td><td>J(Ω) ↑</td></tr><tr><td>Base</td><td> $6 5 . 0 0 { \pm } 3 . 1 6$ </td><td> $1 4 5 9 . 6 0 \pm 4 6 7 . 9 4 ( 1 4 3 1 . 4 5 )$ </td><td> $1 7 . 6 2 { \pm } 0 . 9 7 $ </td><td> $1 2 . 8 8 \pm 1 . 8 9$ </td><td> $2 6 8 . 2 5 { \pm } 1 5 . 7 6$ </td><td> $3 8 . 2 6 \pm 3 . 9 1$ </td><td> $0 . 2 5 { \pm } 0 . 0 1$ </td></tr><tr><td>Trace-only</td><td> $6 5 . 0 0 \pm 4 . 4 7$ </td><td> $1 3 0 3 . 2 6 \pm 2 8 4 . 7 4 \ : ( 1 2 7 5 . 6 0 )$ </td><td> $1 6 . 4 3 { \pm } 2 . 4 0 $ </td><td> $1 2 . 0 4 \pm 1 . 6 6$ </td><td> $2 5 7 . 0 5 { \pm 2 5 . 0 5 }$ </td><td> $3 6 . 3 5 { \pm } 3 . 8 3$ </td><td> $0 . 2 6 \pm 0 . 0 2$ </td></tr><tr><td>Trace-fb (ours)</td><td> ${ \bf 6 8 . 0 0 \pm 2 . 4 5 }$ </td><td> $\mathbf { 1 1 2 3 . 3 3 \pm 1 3 7 . 3 2 ( 1 0 9 8 . 6 8 ) }$ </td><td> $\mathbf { 1 5 . 5 6 \pm 1 . 3 4 }$ </td><td> $\mathbf { 1 0 . 8 8 \pm 0 . 9 4 }$ </td><td> $\mathbf { 2 2 9 . 3 2 \pm 1 6 . 0 0 }$ </td><td> $\mathbf { 3 4 . 4 7 \pm 2 . 6 4 }$ </td><td> $\mathbf { 0 . 2 8 \pm 0 . 0 4 }$ </td></tr></table>

(a)  
![](images/160c14131054a808a90d48089988b1b5c9086259c5a0fb279343ba137d4533bb.jpg)

![](images/c43ee9d01fb44cf338958932eeed0deace39fbadb975c471a79931a0da9ab1d1.jpg)

![](images/a4f194105e941ff5f03807e34cf4deadaab3551b851e509408903efd48e6acfe.jpg)

(d)  
![](images/b64a85c43dc7a2bc3f2b15cf59c1e2fd892e543a463b88e04322fb3cd9ea858a.jpg)  
Fig. 6. Accuracy and response time comparison between VibeTrace and the baseline; \*\* denotes $p < 0 . 0 1$ and \*\*\* denotes $p < 0 . 0 0 1$ . (a) Overall accuracy: VibeTrace (95%) and the baseline (76%). (b) Average response time: VibeTrace (44.0 s) and the baseline (99.7 s). (c) Per-question accuracy across Q1–Q10. (d) Per-question response time across Q1–Q10.

Materials. We collected 10 agent execution traces from real coding sessions. Each trace captures a portion of the agent’s dialogue history and execution steps. Based on each trace, we designed one multiple-choice question targeting execution process comprehension, error localization, or root cause analysis. Each trace was presented in two formats: (i) Baseline: Direct output from OpenCode’s native terminal user interface, and (ii) VibeTrace’s trace visualization. This led to 10 questions with agent execution traces paired with 2 presentation formats each. We created two counterbalanced questionnaire versions, each containing all 10 questions with five per format.

Procedure. The experiment was conducted through an online questionnaire platform. Participants first read a brief introduction to the VibeTrace and OpenCode user interfaces. They then completed 10 multiple-choice questions in sequence. The system automatically recorded answers and response time for each question.

Results. We analyze the study results based on the accuracy and the response time per question (Fig. 6). VibeTrace (M=95.0%, SD=11.4) achieves significantly higher accuracy than the baseline (M=76.0%, SD=17.9), as confirmed by the Wilcoxon signed-rank test [43] $( | Z | = 3 . 0 9 ,$ $p \ < \ 0 . 0 1 )$ ). In terms of response time (Fig. 6b), VibeTrace (M=44.0 s, SD=20.6) is also significantly faster than the baseline (M=99.7 s, SD=45.2) (|Z| = 3.92, p < 0.001). These results suggest that the trace visualization proposed by VibeTrace can more effectively and efficiently support users in understanding agent execution traces, localizing errors, and analyzing root causes.

## 5.3 System Usability

To evaluate how VibeTrace supports human–coding agent collaboration, we conducted a third experiment comprising case studies and interviews with expert users. We examined four aspects of collaboration: whether users could (i) observe the agent’s ongoing and completed activities efficiently (DG1); (ii) understand the agent’s decisions and identify potential errors (DG2); (iii) steer the agent effectively based on this understanding (DG3); and (iv) evolve the agent by distilling reusable knowledge from their interactions (DG4).

Participants. We recruited eight expert users from diverse backgrounds, including both researchers and industry practitioners (age: M=25.5, SD=4.1; four females). All participants had expertise and solid experience in using coding agents such as Cursor, Claude Code, or Copilot, 3 had built agents of their own, providing a deeper understanding of agent execution. This composition allowed us to observe VibeTrace across users with varying levels of coding and agent expertise.

Procedure. The study consists of three phases: onboarding, tasks, and an interview. Participants were asked to share their screen during the study. Each session was recorded and transcribed for subsequent analysis, and informed consent was obtained from all participants.

Onboarding. We began with a brief overview of the study procedure and assisted participants in setting up VibeTrace on their own machines. Then, we played an 8- minute demonstration video that highlighted VibeTrace’s core features to help them get familiar with the system. We also collected the pre-study demographic questionnaire.

Tasks. Participants were requested to use VibeTrace to complete two vibe-coding tasks. The two tasks were designed to be complementary. The first is a lightweight data analysis task that aims to uncover the return-on-investment patterns of the film industry over the past two decades, finding which genres achieve the highest ROI and whether these profitable genres also hold an advantage in audience ratings, and finally presenting the results as an HTML data report. The second is a debugging-heavy task that depends on multiple external services and aims to build a web application that converts an uploaded image into a bead-art pattern annotated with color codes. In both tasks, participants were asked to interpret what the agent was doing and how it proceeded, to judge whether errors arose and, if so, analyze their causes and decide whether to intervene, and ultimately to complete the task through several rounds of iteration. Participants were asked to think aloud throughout and were free to ask questions at any time. In parallel, VibeTrace logged fine-grained behavioral traces, including the timing of panel openings and the sequence and frequency of feature invocations, providing a quantitative complement to the think-aloud data for characterizing participants’ usage patterns. After the two assigned tasks, we invited participants to continue using VibeTrace in their own day-to-day work and to share their experiences with us. We collected the resulting real-world usage traces.

![](images/a15534da6cee47bf8f1e8e7dc264be80bd114c96c5b894ce9d6133ed0f682e7a.jpg)  
Fig. 7. Distribution of interaction time across the four panels for each participant (N = 8).

Interview. At the end of the tasks, participants completed the System Usability Scale (SUS) [44] and took part in a 15-minute open-ended discussion about VibeTrace’s observability, explainability, controllability in the human— coding agent collaboration. They were also asked with their suggestions for improvement.

Results. We report three sets of results: the System Usability Scale scores, the logged usage patterns across the four panels, and the interview findings.

System Usability Scale (SUS). VibeTrace achieved a mean SUS score of $M \ = \ 8 5 . 9 \ ( S D \ = \ 9 . 4 ) .$ , indicating strong functional integration, interaction consistency, and ease of operation. Decomposed under the two-factor structure of Lewis and Sauro [45], VibeTrace scores M = 89.4 on the eight-item Usability subscale and M = 71.9 on the twoitem Learnability subscale. Several participants reported that setup, not the interface, was the main barrier to first use. A subsequent release addresses this: VibeTrace now installs as a single-command OpenCode plugin that launches its services automatically.

Usage Patterns. We analyzed the usage patterns of participants by measuring the proportion of interaction time spent on each panel of VibeTrace (Fig. 7). The numbers were measured by cursor movement. Most participants were highly engaged with the Trace panel. Six participants dedicated the majority of their time to it. This indicates that the visualization was effectively utilized during their interaction. Two participants (P1 and P6) spent more than 50% of their time on the Dialog panel, with the Trace panel being their second most-used feature. Overall, participants agreed that the trajectory visualization was the most helpful component of VibeTrace.

Interviews. Participants provided positive feedback regarding VibeTrace’s observability, explainability, and controllability in human–agent coding collaboration. When observing a long and complex workflow, participants appreciated the visualization provided in the Trace panel. For instance, P6 said, “The tree structure clearly lays out the task flow and the workflow, and the linkage between panels lets me quickly locate the step I want to view.” When an execution encountered an error, participants found it convenient to understand with VibeTrace. P5 stated, “When something goes wrong, I can easily see the content before and after the error point, which makes locating and analyzing it convenient.” P3 said, “The output of the middle nodes helps me understand.” Participants valued the ability to intervene in the agent’s execution. P7 stated, “Ask about the current state more clearly, without having to stop it.” P8 noted, “Quickly see which steps are error-prone, and find a relevant skill to remedy those errors.” Additionally, participants appreciated the online skill distillation mechanism. P4 mentioned, “A skill summary is really necessary; this knowledge used to come from human instructions, but I couldn’t follow the trial and error inside the command line, so having it distilled automatically is very helpful.” P3 said, “There’s no need to explain some processes over and over.”

## 5.4 Example Cases

We present three example cases from participants’ own projects in the expert user study (Section 5.3) to illustrate how VibeTrace supports observing, understanding, intervening in, and improving human–agent collaboration.

## 5.4.1 Observing and Understanding Agent Execution.

This case demonstrated how P7 uses VibeTrace to do technical investigation. The task is to compare two candidate user interface techniques for an agent chat application: one embeds UI components directly in the chat, and one renders the selection on a separate web page and returns the result. The user wants to synthesize the comparison into an HTML webpage. This task is long-running and requires multiple steps of reasoning and tool calls (see Fig. 8). After the user prompt (Fig. 8-a), the agent automatically explores the two techniques in parallel, one for in-chat UI components (Fig. 8-b) and one for the separate web page (Fig. 8- c), where the overall structure is clearly visualized in the trace visualization panel (DG1). When the user checks the execution of the two branches, he identifies three errors highlighted in red in the second branch, which might affect the final result. Therefore, the user clicks each error node to view the details one by one. He reads that the first error is a rate limit from a web search tool (Fig. 8-d), but the following web search actions complete normally, and no downstream effect propagates to the overall execution. Then, the user reads the remaining two errors: the second error is an oversized argument list from a shell command (Fig. 8-e), and the third error is an invalid tool call (Fig. 8-f). Rather than traversing each corresponding node, the user clicks the final node to check the summary of the execution (Fig. 8-g). He understands that the agent has completed the technical investigation but failed to synthesize the results into a webpage due to the previous errors. The root cause is that the agent was not provisioned with the write tool, so it first attempted to persist its markdown notes by inlining the entire body into a single shell command, whose overlong argument list was rejected. Then, a subsequent call to the write tool failed for the same reason (DG2). The way to fix the errors is to grant the agent write access, so that the tasks can save their outputs directly. Through VibeTrace, the user can easily observe the agent execution trace (DG1), and understand its execution and failures during a long-running task (DG2).

![](images/b1c5e9468b49e5c88656fc8397837eaa2b934d9ca46f32b495bd8ca89489d8c2.jpg)

Fig. 8. VibeTrace supports observing and understanding long-running agent execution through interactive visualization of agent execution traces. The figure illustrates a representative workflow: (a) the user asks the agent to investigate two UI techniques and synthesize the findings into a webpage; (b–c) the interaction trace branches into two subagent workflows, each exploring one technique; (d–f) the visualization reveals three failures in the second branch, including a rate-limit error, an oversized argument list, and an invalid tool invocation; (g) the terminal state summarizes the execution, identifies the root causes, and recommends corresponding fixes.  
![](images/80518e63ff21daf3aabc955062445defc176200250e56e5e999b3b7f46971360.jpg)  
Fig. 9. VibeTrace lets the user intervene in the agent execution by forking the execution trace. Left: The user forks a new branch before development begins, prompting a request for design clarification. The greyed-out branch is the original trace that implements the application without consulting design styles, while the lower branch is the newly forked branch that includes user feedback. Middle: the user feedback interface of the forked branch for reminding the agent to settle the frontend design details before building. Right: On the forked branch, the agent uses the question too to ask design details.

## 5.4.2 Intervening Agent Execution.

This case demonstrated a key intervening process when P2 builds an application via VibeTrace. It’s a simple web application generating meal suggestions. The user prompts the agent to build a frontend demo, and the agent scaffolds the entire project step by step and delivers the final implementation. However, the user finds that the agent has chosen the wrong visual style and color palette without consulting the user. Rather than terminating the session and re-prompting from scratch, the user forks a new branch immediately before the node where the development begins (Fig. 9, left). The user deliberately selects that node: the earlier discussion that fixed the feature set is the valuable groundwork he wants to carry forward rather than reconstruct, while the design decisions the agent has since committed to are exactly what he wants to change. On the forked branch, the user prompts the agent to align on the design styles before implementing anything (Fig. 9, middle). Given that instruction, the agent then invokes the question tool several times to request design clarification from the user (Fig. 9, right). The resulting implementation in this forked branch better satisfies the user’s preferences. Through VibeTrace, the user is able to precisely intervene in the agent execution trace to align the agent’s behavior with user expectations, thus improving the efficiency of the collaboration (DG3).

![](images/2670ea103642acc0f2d4bcf291580f89d2bc273e1baefb2df0b415f05812b1dd.jpg)  
Fig. 10. VibeTrace distills a skill from human feedback and reuses it in a later task. Left: the agent automatically matches the distilled skill. Right: VibeTrace automatically distilled front-end-design-alignment-before-build.skill.

## 5.4.3 Agent Self-Evolving from Human Feedback.

After the user forks the trace and accept the changes in Case 2, VibeTrace automatically distills a new front-end-design-alignment-before-build skill by comparing the traces before and after the fork. According to the skill description (Fig. 10, right top), whenever a task involves frontend development and the visual direction is not yet fixed, the agent should use the question tool to confirm design styles with the user before any implementations. The Distill History (Fig. 10, right bottom) describes how that skill was distilled from human feedback. When the user works with VibeTrace on another similar frontend project, the agent automatically matches the skill to confirm design styles with the user before building anything (Fig. 10, left). This demonstrate how VibeTrace learn from human intervention and distill a reusable skill that can be applied to future tasks (DG4).

## 6 DISCUSSION

Our evaluation demonstrates that VibeTrace can improve both the efficiency of agent execution and users’ ability to observe, understand, and steer coding agents. Beyond these results, the studies reveal several design tensions in human–agent collaboration and suggest broader principles for designing agentic coding systems.

## 6.1 Design Tensions Revealed by User Feedback.

Rather than treating all user behaviors as design failures, our interviews reveal several natural tensions that arise when users collaborate with increasingly autonomous coding agents.

Selective oversight vs. comprehensive understanding Participants did not continuously monitor the entire execution trace. Instead, they typically started from the Dialog Panel, used the task overview to locate a relevant trajectory segment, and then inspected details in the Trace Panel. This pattern suggests that users often seek answers to specific questions rather than reconstructing the complete execution history. As P6 noted, “the tree structure clearly lays out the task flow and the workflow, and the linkage between panels lets me quickly locate the step I want to view.” Thus, trace visualization is most useful as an informationseeking layer that supports selective oversight, rather than as a replacement for conversational interaction.

Automation vs. user control Participants also differed in how much they wanted to inspect and intervene in agent behavior. Some users primarily cared about the final outcome and trusted the agent to handle intermediate steps, whereas others frequently inspected the Trace Panel to maintain a stronger sense of control. This difference suggests that effective human–agent collaboration should accommodate a spectrum of oversight rather than assume a single monitoring style. As P7 observed, being able to “ask about the current state more clearly, without having to stop it” provides a lightweight alternative to interrupting the agent entirely.

Adaptability vs. predictability The skill-evolution mechanism introduces another tension: agents should adapt to users’ preferences without making their behavior opaque. Participants appreciated the automatic extraction of reusable knowledge from interaction histories. P4 described the distilled skill as “really necessary” because such knowledge previously required repeated human instructions. However, effective skill distillation requires transparency about the origin and applicability of learned behaviors. VibeTrace therefore exposes skill distillation histories and application conditions, allowing users to inspect learned behaviors rather than treating them as hidden adaptations.

## 6.2 Implications for Human–Agent Collaboration

Taken together, our findings suggest three design principles. First, agent observability should be organized around information seeking rather than exhaustive monitoring: hierarchical traces, summaries, and cross-panel linking allow users to quickly move from a high-level question to the relevant execution details. Second, control should be granular and non-disruptive: mechanisms such as trace forking allow users to revise a problematic decision while preserving useful preceding context, reducing the cost of restarting an interaction. Third, agent learning should be grounded in observable interaction history: distilling reusable skills from execution traces and user corrections turns individual interventions into persistent knowledge while keeping the provenance of that knowledge visible.

These principles point toward a broader shift in human– agent collaboration. Conventional coding-agent interaction largely follows a prompt–response paradigm, in which users specify goals and react to completed outputs. Our findings suggest a complementary observe–intervene–learn paradigm, where users can selectively inspect an agent’s ongoing process, intervene at the level of execution context, and allow successful interventions to become reusable skills. In this model, the agent is not merely an autonomous executor that produces outputs; it becomes a continuously evolving collaborator whose behavior can be observed, corrected, and improved through interaction. VibeTrace provides one concrete example of this paradigm by connecting three previously separated layers—execution traces, user intervention, and persistent agent knowledge—into a single interaction loop.

## 7 CONCLUSION

We presented VibeTrace, an interactive trace visualization and intervention system for human—coding agent collaboration. We contribute both a structured trace visualization that helps users better observe and understand the behavior of agent execution and a trace-level interaction control that allows for more targeted agent interventions. Additionally, the system features a trace-to-skill distillation mechanism that uses human interactions as feedback to automatically build or improve the reusable agent skills. The design of VibeTrace is validated through a quantitative evaluation of the online trace-to-skill algorithm, a user study assessing the trace visualization, and an expert interview to evaluate the entire system in real-world coding tasks.

## REFERENCES

[1] Anthropic. (2025) Claude code. Anthropic. [Online]. Available: https://www.anthropic.com/claude-code

[2] OpenAI. (2025, May) Introducing codex. OpenAI. [Online]. Available: https://openai.com/index/introducing-codex/

[3] OpenCode Team. (2025) OpenCode. OpenCode. [Online]. Available: https://opencode.ai/

[4] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in The Eleventh International Conference on Learning Representations. Red Hook, NY, USA: Curran Associates, Inc., 2023, pp. 1–18. [Online]. Available: https: //openreview.net/forum?id=WE vluYUL-X

[5] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar, “Voyager: An open-ended embodied agent with large language models,” 2024. [Online]. Available: https://openreview.net/forum?id=ehfRiF0R3a

[6] N. Tang, M. Chen, Z. Ning, A. Bansal, Y. Huang, C. McMillan, and T. J.-J. Li, “An empirical study of developer behaviors for validating and repairing ai-generated code,” in Proceedings of the 13th Annual Workshop on the Intersection of Programming Languages and Human-Computer Interaction (PLATEAU 2023). Pittsburgh, PA, USA: Carnegie Mellon University, 2023. [Online]. Available: https://kilthub.cmu.edu/articles/conference contribution/an empirical study of developer behaviors for validating and repairing ai-generated code/22223533

[7] J. T. Liang, C. Yang, and B. A. Myers, “A large-scale survey on the usability of ai programming assistants: Successes and challenges,” in Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, ser. ICSE ’24. New York, NY, USA: Association for Computing Machinery, 2024.

[8] V. Chen, A. Talwalkar, R. Brennan, and G. Neubig, “Code with me or for me? how increasing ai automation transforms developer workflows,” in Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, ser. CHI ’26. New York, NY, USA: Association for Computing Machinery, 2026.

[9] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan, “Swe-bench: Can language models resolve real-world github issues?” in International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, Eds., vol. 2024. Red Hook, NY, USA: Curran Associates, Inc., 2024, pp. 54 107–54 157. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2024/ file/edac78c3e300629acfe6cbe9ca88fb84-Paper-Conference.pdf

[10] A. Karpathy. (2025) There’s a new kind of coding i call vibe coding. X. [Online]. Available: https://x.com/karpathy/status/ 1886192184808149383

[11] M. Wermelinger, “Using github copilot to solve simple programming problems,” in Proceedings of the 54th ACM Technical Symposium on Computer Science Education V. 1, ser. SIGCSE 2023. New York, NY, USA: Association for Computing Machinery, 2023, pp. 172–178.

[12] A. Ziegler, E. Kalliamvakou, X. A. Li, A. Rice, D. Rifkin, S. Simister, G. Sittampalam, and E. Aftandilian, “Measuring github copilot’s impact on productivity,” Communications ofthe ACM, vol. 67, no. 3, pp. 54–63, Mar. 2024.

[13] P. Vaithilingam, T. Zhang, and E. L. Glassman, “Expectation vs. experience: Evaluating the usability of code generation tools powered by large language models,” in Extended Abstracts of the 2022 CHI Conference on Human Factors in Computing Systems, ser. CHI EA ’22. New York, NY, USA: Association for Computing Machinery, 2022.

[14] M. X. Liu, A. Sarkar, C. Negreanu, B. Zorn, J. Williams, N. Toronto, and A. D. Gordon, ““what it wants me to say”: Bridging the abstraction gap between end-user programmers and codegenerating large language models,” in Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, ser. CHI ’23. New York, NY, USA: Association for Computing Machinery, 2023.

[15] R. Wang, R. Cheng, D. Ford, and T. Zimmermann, “Investigating and designing for trust in ai-powered code generation tools,” in Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, ser. FAccT ’24. New York, NY, USA: Association for Computing Machinery, 2024, pp. 1475–1493.

[16] S. Amershi, D. Weld, M. Vorvoreanu, A. Fourney, B. Nushi, P. Collisson, J. Suh, S. Iqbal, P. N. Bennett, K. Inkpen, J. Teevan, R. Kikin-Gil, and E. Horvitz, “Guidelines for human-ai interaction,” in Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, ser. CHI ’19. New York, NY, USA: Association for Computing Machinery, 2019, pp. 1–13.

[17] J. Wang, L. Gou, H.-W. Shen, and H. Yang, “Dqnviz: A visual analytics approach to understand deep q-networks,” IEEE Transactions on Visualization and Computer Graphics, vol. 25, no. 1, pp. 288–298, 2019.

[18] F. Cheng, V. Zouhar, R. S. M. Chan, D. Furst, H. Strobelt, and¨ M. El-Assady, “Understanding large language model behaviors through interactive counterfactual generation and analysis,” IEEE Transactions on Visualization and Computer Graphics, vol. 32, no. 1, pp. 846–856, 2026.

[19] D. Shi, S. Zhu, T. Weinkauf, and A. Oulasvirta, “Interactive reward tuning: Interactive visualization for preference elicitation,” in 2024

IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Piscataway, NJ, USA: IEEE, 2024, pp. 9254–9261.

[20] J. Kompatscher, D. Shi, G. Varni, T. Weinkauf, and A. Oulasvirta, “Interactive groupwise comparison for reinforcement learning from human feedback,” 2026.

[21] D. Shi, F. Cheng, T. Weinkauf, A. Oulasvirta, and M. El-Assady, “Dxhf: Providing high-quality human feedback for llm alignment with interactive decomposition,” in Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology, ser. UIST ’25. New York, NY, USA: Association for Computing Machinery, 2025.

[22] A. Joshi, “Xai for coding agent failures: Transforming raw execution traces into actionable insights,” 2026.

[23] S.-X. Yang, C. Wang, H. Zhang, W. Yu, L. Wu, J. Gui, D. Yang, Y. Cen, Z. Feng, B. Wen, Y. Wang, L. Zhong, J. Ren, L. Zhang, and J. Tang, “Tracesir: A multi-agent framework for structured analysis and reporting of agentic execution traces,” 2026.

[24] G. Zhang, J. Wang, J. Chen, W. Zhou, K. Wang, and S. Yan, “Agentracer: Who is inducing failure in the llm agentic systems?” in International Conference on Learning Representations, C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, Eds., vol. 2026. Red Hook, NY, USA: Curran Associates, Inc., 2026, pp. 11 377–11 399. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ file/134ed7a477770f227f12450ef0cbb8f4-Paper-Conference.pdf

[25] A. Grigor, C. S. de Witt, S. Birnbach, and I. Martinovic, “Vet your agent: Towards host-independent autonomy via verifiable execution traces,” in Proceedings of the ACM Asia Conference on Computer and Communications Security, ser. ASIA CCS ’26. New York, NY, USA: Association for Computing Machinery, 2026, pp. 820–834.

[26] N. van der Vleuten, A. Flores, S. Mathur, M. Rakitin, T. Hopkins, K. G. Yager, and E. H. R. Tsai, “Envtrace: Simulation-based semantic evaluation of llm code via execution trace alignment – demonstrated at synchrotron beamlines,” 2025.

[27] Y. Chen, H. Jhamtani, S. Sharma, C. Fan, and C. Wang, “Steering large language models between code execution and textual reasoning,” in International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., vol. 2025. Red Hook, NY, USA: Curran Associates, Inc., 2025, pp. 99 676–99 707. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ file/f763f7c9a6599e14b07add5937d8189c-Paper-Conference.pdf

[28] Z. Tao, T.-E. Lin, X. Chen, H. Li, Y. Wu, Y. Li, Z. Jin, F. Huang, D. Tao, and J. Zhou, “A survey on self-evolution of large language models,” 2024.

[29] J. Fang, Y. Peng, X. Zhang, Y. Wang, X. Yi, G. Zhang, Y. Xu, B. Wu, S. Liu, Z. Li, Z. Ren, N. Aletras, X. Wang, H. Zhou, and Z. Meng, “A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems,” 2025.

[30] S. Ouyang, J. Yan, I.-H. Hsu, Y. Chen, K. Jiang, Z. Wang, R. Han, L. Le, S. Daruki, X. Tang, V. Tirumalashetty, G. Lee, M. Rofouei, H. Lin, J. Han, C.-Y. Lee, and T. Pfister, “Reasoningbank: Scaling agent self-evolving with reasoning memory,” in International Conference on Learning Representations, C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, Eds., vol. 2026. Red Hook, NY, USA: Curran Associates, Inc., 2026, pp. 94 327–94 354. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ file/980ea04d23d1f6908964eba2a74afe45-Paper-Conference.pdf

[31] Anthropic. (2025, Oct.) Equipping agents for the real world with Agent Skills. Anthropic Engineering. [Online]. Available: https://www.anthropic.com/engineering/ equipping-agents-for-the-real-world-with-agent-skills

[32] H. Lin, P. Li, J. Song, F. Jiang, and T. Zhang, “Muse-autoskill: Self-evolving agents via skill creation, memory, management, and evaluation,” 2026.

[33] Y. Li, Y. Zhang, X. Zhang, X. Liu, and Y. Liu, “Codeskill: Learning self-evolving skills for coding agents,” 2026.

[34] Y. Yang, Z. Gong, W. Huang, Q. Yang, Z. Zhou, Z. Huang, Y. Li, X. Gao, Q. Dai, B. Liu, K. Qiu, Y. Yang, D. Chen, X. Yang, and C. Luo, “Skillopt: Executive strategy for self-evolving agent skills,” 2026.

[35] Q. Mi, Z. Ma, M. Yang, H. Li, Y. Wang, H. Zhang, and J. Wang, “Skill-pro: Learning reusable skills from experience via nonparametric ppo for llm agents,” 2026.

[36] J. Ni, Y. Liu, X. Liu, Y. Sun, M. Zhou, P. Cheng, D. Wang, E. Zhao, X. Jiang, and G. Jiang, “Trace2skill: Distill trajectory-local lessons into transferable agent skills,” 2026.

[37] H. Zhou, S. Guo, A. Liu, Z. Yu, Z. Gong, B. Zhao, Z. Chen, M. Zhang, Y. Chen, J. Li, R. Yang, Q. Liu, X. Yu, J. Zhou, N. Wang, C. Sun, and J. Wang, “Memento-skills: Let agents design agents,” 2026.

[38] G. Zhang, E. Zhu, J. Zhou, C. Jia, and H. Wang, “Skillevolver: Skill learning as a meta-skill,” 2026.

[39] H. Zhuang, H. Xing, and X. Zhang, “Agentclick: A skill-based human-in-the-loop review layer for terminal ai agents,” in Proceedings of the ACM Conference on AI and Agentic Systems, ser. CAIS ’26. New York, NY, USA: Association for Computing Machinery, 2026, pp. 1372–1378.

[40] Y. Zhou, K. Guo, H. Zhuang, X. Wang, Y. Huang, Z. Liang, P.-Y. Chen, T. Gao, N. Moniz, N. V. Chawla, and X. Zhang, “Getting better at working with you: Compiling user corrections into runtime enforcement for coding agents,” 2026.

[41] V. Braun and V. Clarke, “Using thematic analysis in psychology,” Qualitative Research in Psychology, vol. 3, no. 2, pp. 77–101, 2006.

[42] DeepSeek-AI et al., “DeepSeek-V4: Towards highly efficient million-token context intelligence,” 2026.

[43] R. F. Woolson, “Wilcoxon signed-rank test,” in Wiley Encyclopedia of Clinical Trials, R. B. D’Agostino, L. M. Sullivan, and J. Massaro, Eds. Hoboken, NJ, USA: John Wiley & Sons, Ltd, 2008, pp. 1–3.

[44] J. Brooke, “Sus: A “quick and dirty” usability scale,” in Usability Evaluation in Industry, P. W. Jordan, B. Thomas, I. L. McClelland, and B. A. Weerdmeester, Eds. London, UK: Taylor & Francis, 1996, pp. 189–194.

[45] J. R. Lewis and J. Sauro, “The factor structure of the system usability scale,” in Human Centered Design, ser. Lecture Notes in Computer Science, M. Kurosu, Ed., vol. 5619. Berlin, Heidelberg: Springer, 2009, pp. 94–103.

![](images/ebd12c9572e86b55924df61eecbab2af7a48af8a542e7702c64fc540f3441669.jpg)  
Xiaoke Yan received her bachelor’s degree from the Department of Software Engineering, Tongji University in 2024. She is currently pursuing her master’s degree at the College of Design and Innovation, Tongji University, where she is a member of the Intelligent Design & Innovation Lab (iDVx Lab). Her research interests include information visualization, AI-supported design, and human-computer interaction.

![](images/218208e2d9b039b57aa437af3e17685091ee8a8e002a04e1978adcf262e56898.jpg)

Jiaying Lei received her bachelor’s degree from the School of Mathematical Sciences, Tongji University, in 2024. She is currently a Ph.D. candidate jointly affiliated with the Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University, and the Shanghai Innovation Institute. Her research interests include generative AI and human-computer interaction.

![](images/2166384713a0b515a30ef03d99be8628b2ca826cf3ab7dc976b60c737cf70ca4.jpg)

Danqing Shi received his PhD degree from the College of Design and Innovation at Tongji University, China, in 2022. He is currently an assistant professor in the Department of Computer Science at Lund University, Sweden, where he leads the Lund HCI group. Before that, he was a postdoctoral research associate with the Department of Engineering at University of Cambridge, and an academic guest at ETH Zurich. His research interests include human-AI interaction, human-computer interaction, and visualization.

![](images/f0ee98712f7c604bde7302afb6dce510e4511d1dda01187e94cb7750a81be6eb.jpg)

Zichen Wang Wang Zichen, admitted to the School of Mathematical Sciences, Tongji University in 2023, is currently pursuing a Bachelor of Science in Mathematics and Applied Mathematics.

![](images/23e39393903c7a04f65e4e8806362bae4ad324782a98e2efdb3ebabf5d915b11.jpg)

Nan Cao received his Ph.D. degree in Computer Science and Engineering from the Hong Kong University of Science and Technology (HKUST), Hong Kong, China in 2012. He is currently a professor at Tongji University and the Vice Dean of the Tongji College of Design and Innovation. He also directs the Tongji Intelligent Big Data Visualization Lab (iDV<sup>x</sup> Lab) and conducts interdisciplinary research across multiple fields, including data visualization, human-computer interaction, and artificial intelligence. He was a research staff member at the IBM T.J. Watson Research Center, New York, NY, USA before joining the Tongji faculty in 2016.