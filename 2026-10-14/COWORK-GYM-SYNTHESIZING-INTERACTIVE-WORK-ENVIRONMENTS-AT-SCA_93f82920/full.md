# COWORK-GYM: SYNTHESIZING INTERACTIVE WORK ENVIRONMENTS AT SCALE

Anni Zou<sup>†</sup>, Cheng Chen<sup>†</sup>, Jiaji Deng<sup>†</sup>, Weikang Zhou<sup>†</sup>, Xinji Mai<sup>†</sup>, Yue Wu<sup>†</sup>, Jianing Lin, Lipeng Xie, Lixiong Qin, Qingyue Zhao, Que Shen, Shuchang Tao, Xuchen Pan, Yaliang Li, Yunpeng Zhai, Yushuo Chen, Zhihai Wang, Zhuo Zhang, Zhaoyang Liu<sup>\*</sup>, Bolin Ding<sup>\*</sup>, Jingren Zhou

Alibaba Token Foundry

{jingmu.lzy, bolin.ding}@alibaba-inc.com

## Abstract

Cowork agents navigate dependencies across tools and files while adapting to changing user requirements and edits to shared workspaces. Scaling such environments requires maintaining consistency among task objectives, executable resources, user behavior, and success criteria. We introduce Cowork-Gym, a framework that synthesizes Cowork environments through a build–execute–revise loop that uses execution evidence and review feedback to validate and repair generated environments. To capture realistic Cowork dynamics, a stateful user simulator responds to the agent’s actions through dialogue and direct edits to the shared workspace. We synthesize 64.9K environments across 14 application scenarios and use them to post-train 9B, 27B, and 35B-A3B Qwen models with supervised fine-tuning, reinforcement learning, and on-policy distillation. Averaged across six agentic benchmarks, these models outperform their corresponding base checkpoints by 8.7, 6.4, and 4.1 points, respectively. Ours-27B ranks second overall among the evaluated models and achieves the highest average scores on PawBench and ClawEval, outperforming models with over 100B total parameters on both benchmarks.

Model: https://huggingface.co/collections/agentscope-ai/qwenpaw-flash

Dataset: https://huggingface.co/datasets/agentscope-ai/Cowork-Gym

Environment Overview  
![](images/5dcb312c80d0680f224867b7a0dc64db9526f18db3e5e4b013841ceac0095acd.jpg)

Model Leaderboard  
![](images/93060b4e465d88d8d24aa48bfe985a7fe015ef14ccda9f5533b9967409cf992d.jpg)  
Figure 1: Scale, diversity, and downstream impact of synthesized Cowork environments. Left: Our environment-synthesis pipeline produces 64.9K tasks across 14 top-level scenarios, with representative workspaces illustrating diverse personal and professional workflows. Right: The leaderboard demonstrates the effectiveness of training on these environments: Ours-27B ranks second across six agentic benchmarks and achieves the highest average scores on PawBench and ClawEval among the evaluated models.

## 1 INTRODUCTION

Language-model agents increasingly undertake multi-step work that combines search, software development, computer use, and other capabilities (Zhou et al., 2023; Xie et al., 2024; Bai et al., 2026; GLM-5 Team, 2026), as illustrated by systems such as Claude Cowork, Codex, QwenPaw, and OpenClaw (Anthropic, 2026; OpenAI, 2026; AgentScope Team, 2026; OpenClaw, 2026). In these settings, an agent works through multiple tools in a persistent workspace, creating and revising artifacts as the task unfolds. At the same time, the user may supply new information, revise constraints, or replace task inputs, so the relevant state is shaped by both parties rather than fixed at the outset.

We use cowork as a generic term for this interaction pattern, in which a user and an agent work toward an objective through a shared workspace. Progress is reflected in the artifacts they produce and the changes they make to that workspace. To train and evaluate this behavior, we define a Cowork environment that specifies the task objective, initial workspace, available interfaces, user interaction policy, and verification criteria. It describes both the work to be completed and the conditions under which the interaction unfolds.

Constructing these environments at scale is challenging, which requires covering combinations of work domains, deliverable requirements, user profiles, and interaction patterns. Each combination should be realized in an executable workspace with compatible user actions and reliable outcome checks. These components must remain consistent as users reveal information, revise requirements, or replace inputs during execution. A final response may appear correct even when a required artifact is missing or the workspace is left in an incorrect state. Preparing and checking these conditions manually is costly; errors can make a task unsolvable or incorrectly scored. Recent advances in automatic environment synthesis (Zhang et al., 2026; Dong et al., 2026; Li et al., 2026; Shi et al., 2026b; Chae et al., 2026; Mai et al., 2025) motivate us to explore automated synthesis for cowork environments, where workspace state and user requirements evolve during interaction.

We introduce Cowork-Gym, a scalable framework for synthesizing Cowork environments (Figure 3). It generates tasks through two complementary routes: taxonomy-guided generation targets selected capabilities and scenario conditions, while resource-grounded generation derives tasks from existing resources, such as code repositories and document collections. Each task is instantiated as an executable environment and refined through a build–execute–revise loop. Sandbox execution and verifier checks identify inconsistencies and guide repairs before trajectory collection.

Within these environments, a stateful user simulator responds to the agent’s progress through dialogue and direct workspace edits. For example, after reviewing an initial analysis, it may replace the source table, requiring the agent to revisit its inputs and revise its outputs. This supports collaboration in which users provide feedback and modify shared resources as work progresses. We collect trajectories across agent harnesses and filter them using artifact checks and execution records.

Using this framework, we synthesize 64.9K Cowork environments across 14 top-level scenarios, spanning office productivity, software engineering, data science, and other personal and professional workflows (Figure 1). We use the collected trajectories and verification signals to post-train Qwen3.5- 9B, Qwen3.6-27B, and Qwen3.6-35B-A3B through supervised fine-tuning, reinforcement learning, and optional on-policy distillation (Lu et al., 2026b; Shenfeld et al., 2026). Across six agentic benchmarks, the resulting models improve upon their base checkpoints in 17 of 18 matched model– benchmark comparisons, demonstrating consistent gains across model scales and task settings.

Our main contributions are threefold:

• We develop Cowork-Gym for large-scale synthesis of cowork environments spanning diverse work domains and deliverables. It combines taxonomy-guided and resource-grounded task generation with a build–execute–revise loop to expand task coverage and validate environment consistency.

• We introduce a stateful user simulator that conditions conversational feedback and sharedworkspace changes on the agent’s progress, creating multi-turn collaboration in which the agent must adapt its ongoing work to revised requirements and inputs.

• We demonstrate the training value of synthesized environments across three model scales and six agentic benchmarks. Ours-27B ranks second overall among the evaluated models and achieves the best scores on PawBench and ClawEval, surpassing evaluated models with over 100B total parameters on both benchmarks.

## 2 PROBLEM FORMULATION

We consider Cowork tasks in which an agent uses multiple tools in a persistent workspace, while a stateful user may update the task through dialogue or workspace changes. To describe how such tasks become training data, we distinguish three levels: the environment specifies the task, a particular model–harness configuration realizes it as a trajectory, and filtering, selection, and transformation turn that trajectory into a training sample.

![](images/169987dc7edfef5e3e0d8a98c3cb5c9c52c66fcbfbe21d6e49359ce34a096297.jpg)  
Figure 2: Cowork environment packaging and rollout. (a) Each environment is represented as a Harborcompatible task package augmented with a user simulator, bundling the task instruction, runtime, reference solution, verifier, and user profiles. (b) During rollout, the user simulator communicates with the acting agent. It also observes and edits the shared workspace state in response to the agent’s progress. (c) The resulting trajectory records user messages, agent actions, and tool observations; the hybrid verifier then evaluates it.

Cowork environment. We represent a Cowork environment through five coupled components:

$$
\mathcal { E } = ( g , s _ { 0 } , \mathcal { T } , \pi _ { U } , \mathcal { V } ) ,\tag{1}
$$

Here, g contains the task instruction and completion requirements, while s contains the initial $g$ $s _ { 0 }$ workspace and other persistent state. The interface set I covers the tools, skills, memory, permissions, and other ways to observe or modify that state, whereas π<sub>U</sub> specifies the stateful user’s private context and intervention policy. Finally, the hybrid verifier $\mathcal { V } = ( \mathcal { V } _ { \mathrm { c o d e } } , \mathcal { V } _ { \mathrm { L L M } } )$ combines deterministic checks of the final workspace and artifacts with LLM-based evaluation of semantic correctness and task-relevant process evidence.

Trajectory. Once an environment is defined, pairing it with an agent policy $\pi _ { A }$ and harness H yields a trajectory

$$
\tau \sim \operatorname { R o l l o u t } ( \pi _ { A } , H , \mathcal { E } ) ,\tag{2}
$$

where τ records the evolving workspace together with agent actions, user interventions, tool calls, and tool results. Because the harness mediates observations, tool schemas, execution, context management, and stopping behavior, the same environment may produce different trajectories under different policies or harnesses.

Training sample. Before a trajectory is used for training, a filtering operator $F ( \tau ; \mathcal { E } )$ evaluates its task outcome and structural validity using evidence from V and checks of the event log. For each retained trajectory, Forma $\rangle _ { H }$ then normalizes harness-specific messages, tool events, and metadata into the representation required by the downstream training stage, including supervision masks when applicable. The resulting dataset is

$$
{ \mathcal { D } } _ { \mathrm { t r a i n } } = \{ { \mathrm { F o r m a t } } _ { H } ( \tau ) : F ( \tau ; \mathcal { E } ) = 1 \} .\tag{3}
$$

![](images/f28cbc7325e33d194d8abb5c463ac75514ba8f3394f1cb7b476b8fb0e3d56da9.jpg)  
Figure 3: Overview of Cowork environment synthesis and agent training. Left: Environment synthesis starts with (a) Task Specification (Section 3.1) and proceeds through (b) the Build–Execute– Revise loop (Section 3.2). Right: (c) Trajectory Generation with stateful user simulation is followed by (d) Filtering (Section 3.4). Bottom: The resulting environments and trajectories support agent training with SFT, RL, and OPD.

## 3 SCALABLE COWORK DATA SYNTHESIS

This section describes how Cowork-Gym constructs Cowork environments E and converts their rollouts into training samples (Figure 3). We first derive the task specification g from taxonomy conditions or grounded resources (Section 3.1). We then instantiate and validate the components of E through a build–execute–revise loop: the Builder constructs $s _ { 0 } , \mathcal { T } ,$ and $\pi _ { U } ;$ the Verifier Constructor constructs V; and the Issue Detector executes the assembled environment and routes defects for repair (Section 3.2). During rollout, the stateful user simulator implements $\pi _ { U }$ while an acting policy $\pi _ { A }$ operates through harness H (Section 3.3). Trajectory filtering and model-conditioned selection then convert the resulting trajectories τ into training samples (Section 3.4).

## 3.1 TASK SPECIFICATION

We construct the task specification g through two complementary routes. The top-down, taxonomyguided route uses a task taxonomy to control coverage and composition. The bottom-up, resourcegrounded route derives task requirements from existing work artifacts and resource collections based on their content and available operations. Regardless of the route, the resulting specification constrains the remaining components of E without prescribing a privileged solution trace.

Taxonomy-guided route. This route samples task conditions from a versioned, six-dimensional taxonomy covering capabilities, application scenarios, task structure and difficulty, user and deliverable requirements, input and output modalities, and safety. It includes 17 capability labels and 14 top-level application scenarios, with a three-level scenario hierarchy. Capabilities remain separate from application scenarios: a finance task, for example, may require data analysis without introducing a distinctfinancial data analysis label. We discard incompatible combinations through schema and cross-field checks, then use accepted labels to condition generation, track coverage, and rebalance the task distribution. Capability definitions and the scenario hierarchy are summarized in Section A of Appendix; the resulting distributions are reported in Figure 5.

Resource-grounded route. This route starts from an existing resource and constructs a task around its content and available operations. It complements taxonomy-guided sampling with artifact specific dependencies and interface constraints that are difficult to anticipate from labels alone. Illustrative sources include financial reports for cross-period analysis, legal documents for clause extraction and comparison, and diagnostic reports for summarizing recorded findings; repositories and paper collections similarly support software and literature tasks. We retain only tasks in which the resource contributes materially to completion. When applicable, complexity arises from cross-file dependencies, incomplete evidence, or required artifact transformations.

## 3.2 BUILD–EXECUTE–REVISE LOOP

Given the task specification g, the Build–Execute–Revise loop instantiates and validates the remaining components of $\mathbf { \bar { \mathcal { E } } } = ( g , s _ { 0 } , \bar { \mathcal { T } } , \pi _ { U } , \mathcal { V } )$ . During Build, the Builder constructs $s _ { 0 } , \mathcal { T } ,$ , and $\pi _ { U }$ , while the Verifier Constructor independently constructs $\nu$ in an isolated sandbox. During Execute, the Issue Detector attempts to run the assembled environment and identifies defects across its components. During Revise, each defect is returned to the responsible constructor, and the revised environment enters the next iteration. The loop continues until the environment satisfies the release criteria or reaches the maximum iteration budget. We instantiate the Builder, Verifier Constructor, and Issue Detector as role-specialized agents on a common coding-agent runtime with file, shell, web search and structured-output capabilities. Their prompts, visible state, tool permissions, and sandboxes are configured separately.

## 3.2.1 BUILDER

The Builder compiles a task specification $g$ into an executable task package with three components: the initial workspace state $s _ { 0 } ,$ the interface set $\mathcal { T } ,$ , and the stateful user policy $\pi _ { U }$ . Their construction follows the state visibility and isolation principle. The acting agent receives only the public portion of the specification, the shared workspace state, and metadata for permitted interfaces. The user policy’s private state, scheduled workspace updates, reference solutions, the implementation of $\nu ,$ and construction metadata remain outside the agent’s sandbox. Private information becomes observable only through a user message or an authorized workspace change produced by $\pi _ { U } .$

Initial workspace state $\mathbf { \delta _ { s _ { 0 } } } .$ . The initial workspace state contains the inputs, configurations, and editable files specified by $^ { g , }$ together with the locations and formats of required deliverables. The Builder records dependencies among these resources so that later changes remain consistent: replacing an input table may invalidate an analysis, while revising a source document may require regenerating its report. These dependencies also identify which artifacts the user policy may update and which outputs the verifier should inspect.

Interface set I. The interface set exposes only the operations required to complete the task, such as file operations, shell execution, application tools, skills, memory, or MCP services. Each interface specifies its version, working-directory rules, execution limits, and permission scope. Credentials, privileged operations, and implementation details remain outside the sandbox. This restricted boundary prevents unintended shortcuts and supports reproducible execution.

Stateful user policy $\pi _ { U }$ . The Builder creates the simulator’s private profile, including its latent goal, knowledge boundary, signal plan, and permitted workspace operations. These assets determine what the simulated user knows, when it may provide additional information, and which workspace changes it may perform. They are not directly visible to the acting agent; their effects become observable only through simulator-generated messages or authorized updates to the shared workspace. Section 3.3 describes how the stateful user simulator executes $\pi _ { U }$ during rollout.

## 3.2.2 VERIFIER CONSTRUCTOR

The Verifier Constructor builds a hybrid verifier $\mathcal { V } = ( \mathcal { V } _ { \mathrm { c o d e } } , \mathcal { V } _ { \mathrm { L L M } } )$ in a sandbox isolated from the Builder. Builder-owned artifacts remain read-only, and persistent writes are restricted $\mathrm { \sf t o } \mathrm { \sf t e s t { s } } / .$ Each task requirement is mapped to a success condition, observable evidence, a checker or rubric item, and a weight. Expected values are derived from source data and the realized interaction, with reference materials used only for independent cross-checking.<sup>2</sup>

$\nu _ { \mathrm { c o d e } }$ performs deterministic checks on the final workspace, artifacts, and trajectory, whereas V<sub>LLM</sub> evaluates semantic and procedural evidence. Together, they cover output validity, structural correctness, semantic cross-checking, and red-line constraints. Their results are combine using normalized weights, with red-line failures limiting the maximum score. The complete verifier is self-validated on a reference completion and controlled failure cases before being passed to the Issue Detector.

## 3.2.3 ISSUE DETECTOR

The Issue Detector reviews each assembled environment E before release. It combines executionbased review with a rule audit to assess consistency across all five components. Only environments that pass both stages proceed to trajectory generation.<sup>3</sup>

Execution-based review. The Issue Detector runs the environment in a sandbox to inspect the task specification, initial workspace state, and interfaces, and to exercise the user policy’s planned interventions. It tests the code-based and LLM-based verifier components on a valid reference completion and controlled failure states, checking whether they distinguish valid completion from incomplete, malformed, and untouched states.

Rule audit. The Rule Audit applies fixed release criteria to the task package and execution evidence. It checks package structure, required fields, interface and permission allowlists, user assets, scheduled workspace patches, and leaks of answers, verifier logic, private information, or construction artifacts. Release also requires at least three meaningful score components and correct verifier behavior across the tested states.

Targeted repair feedback. Findings from both stages are routed by ownership: issues with the task specification, workspace state, interfaces, or user policy return to the Builder, while verifier issues return to the Verifier Constructor. Each revised environment is reviewed again in a reset sandbox until it passes or the iteration budget is exhausted, after which unresolved candidates are rejected.

## 3.3 STATEFUL USER SIMULATOR

As shown on the right of Figure 4, the stateful user simulator is the executable implementation of the user policy π<sub>U</sub>. The Builder initializes its private state and permissions during environment construction (Section 3.2.1); during rollout, the simulator runs outside the acting agent’s sandbox and reacts to intermediate results, corrects misunderstandings, reveals new information, and modifies the shared workspace. The acting agent observes only public messages and resulting workspace changes.

Private user state. Each environment defines a private user profile containing a persona, latent goal, knowledge boundary, interaction history, and signal plan. The persona governs how the user communicates, while the knowledge boundary limits the simulator to information that the represented user could reasonably know. The signal plan identifies possible stages of interaction without fixing messages in advance. The initial query reveals enough information to begin the task, while further constraints or preferences are disclosed only when relevant.

State-dependent interaction. The user simulator communicates with the acting agent to finish the task. After each agent work segment, the simulator reviews the response, public interaction history, and permitted workspace state. It may intervene when the task remains ambiguous, a draft needs revision, an error occurs, or an input changes. Because each response and workspace update depends on the agent’s actual progress, the same initial query can produce different follow-ups across rollouts.

Workspace intervention. In Cowork settings, users and agents collaboratively create and revise artifacts in a shared workspace. In addition to conversational feedback, the simulator may revise an agent-produced draft or apply controlled file patches or tool-based edits to update source data before the next user message. Such changes can invalidate earlier observations and intermediate results, requiring the agent to re-read relevant files, revise assumptions, and update dependent artifacts.

![](images/b69b337b4540c7a079b7d6790de5755821c563887ab5efb37e4e464a9cd34c80.jpg)  
Figure 4: Cowork environment synthesis and stateful user simulation. Left: The interaction loop among user, agent, and the environment in the real world. Right: The user simulator runs outside the acting-agent sandbox and conditions each intervention on private user state and current task progress. The .user/ directory stores persistent context; .patch/ provides turn-specific behavioral instructions and authorized workspace updates. The acting agent observes only public messages and the resulting changes(Section 3.3).

## 3.4 TRAJECTORY GENERATION AND FILTERING

For each released environment ${ \mathcal E } = ( g , s _ { 0 } , { \mathcal T } , \pi _ { U } , \mathcal V )$ , the acting agent starts from $s _ { 0 }$ and operates through I, while the stateful user simulator executes $\pi _ { U }$ outside the agent’s sandbox. The resulting trajectory τ includes an event log and final workspace state. The filtering operator $F ( \tau ; \mathcal { E } )$ determines whether τ provides reliable supervision before model-conditioned selection is applied.

Cross-harness rollout with user simulator. To enable comparable rollouts across harnesses, we represent each E using the Harbor task format (Harbor Framework Team, 2026), as illustrated in Figure 2. This representation keeps E fixed while allowing the acting policy $\pi _ { A }$ or harness H to vary. The acting agent and stateful user simulator interact through public messages and workspace changes. We collect training trajectories with QwenPaw and OpenClaw (AgentScope Team, 2026; OpenClaw, 2026), and evaluate transfer to Hermes Agent (Nous Research, 2026) (Section 5.4.3).

Trajectory filtering. The filtering operator $F ( \tau ; \mathcal { E } )$ determines whether an individual trajectory provides reliable supervision along three dimensions:

(1) Record integrity. The event log must be complete, uncorrupted, and free of leaked private information.

(2) Task outcome. The hybrid verifier evaluates the final workspace state, produced artifacts, and task-relevant interaction evidence.

(3) Claims and recovery. We compare success claims against observed outcomes and reject unsupported claims or unresolved failures. Intermediate tool errors are permitted when the agent recovers and reaches a verified outcome.

Model-conditioned selection. The preceding filtering stage identifies valid trajectories that can be used for training. Beyond this validity-based filtering, we investigate whether model-conditioned metrics can further improve training effectiveness by prioritizing trajectories according to their potential learning value for a target student $\pi _ { \theta }$ . We use average negative log-likelihood (NLL) and Rank–Surprisal Ratio (RSR) (Yang et al., 2026b) as alternative criteria under a fixed retention budget. Let $\mathcal { M } _ { \tau }$ denote the scored token positions in trajectory τ, with $y _ { t }$ the recorded token and $h _ { t }$ its preceding interaction context. The two scores are

$$
\begin{array} { r l } & { \mathrm { N L L } _ { \theta } ( \tau ) = - \displaystyle \frac { 1 } { | \mathcal { M } _ { \tau } | } \sum _ { t \in \mathcal { M } _ { \tau } } \log \pi _ { \theta } ( y _ { t } \mid h _ { t } ) , } \\ & { \mathrm { R S R } _ { \theta } ( \tau ) = \displaystyle \frac { \sum _ { t \in \mathcal { M } _ { \tau } } \operatorname* { m i n } \bigl ( r _ { \theta , t } , r _ { \mathrm { m a x } } \bigr ) } { | \mathcal { M } _ { \tau } | \mathrm { N L L } _ { \theta } ( \tau ) } . } \end{array}\tag{4}
$$

Here, ${ r } _ { \theta , t }$ is the rank of $y _ { t }$ under $\pi _ { \boldsymbol { \theta } } ( \cdot \mid h _ { t } )$ , with rank one assigned to the most probable token, and $r _ { \mathrm { m a x } }$ caps the rank. NLL measures how unexpected the scored tokens are to the student. RSR divides the mean clipped rank by NLL, combining relative token preference and surprisal as a proxy for data–student suitability.

To limit shifts in the trajectory-length distribution, we rank scores within equal-frequency length bins. For a retention fraction $0 < \rho \le 1$ , NLL retains the central fraction $\rho$ of trajectories in each bin, whereas RSR retains the fraction $\rho$ with the lowest scores (see Section 5.4.4 for the experimental settings). Finally, Format $H$ normalizes the selected harness-specific trajectories into the representation required by the downstream training stage, producing samples for $\mathcal { D } _ { \mathrm { t r a i n } }$ 1

## 4 AGENTIC POST-TRAINING

We consider two post-training routes. The first directly adapts the model through SFT followed by outcome-based RL. The second uses on-policy distillation: a student initialized from the base checkpoint learns the agentic capabilities of a same-size SFT or SFT+RL teacher from interaction histories generated by the student itself. Direct adaptation can improve agentic performance while weakening capabilities outside the training distribution (Shenfeld et al., 2026; Phan et al., 2025). Motivated by work on on-policy distillation (Shenfeld et al., 2026; Lu et al., 2026b), we investigate whether the second route improves agent performance while better preserving the model’s general capabilities.

Supervised fine-tuning. We perform SFT on filtered trajectories converted from harness-specific formats into the target model’s training format. This conversion standardizes the representation while preserving the recorded actions and observations. The loss is computed only on tokens generated by the agent, including assistant messages and tool calls. Public user messages, tool observations, and workspace events are included as context but excluded from the loss; private simulator state and host metadata are excluded from the training input. The mixture includes successful executions, revision-driven interactions, and grounded recoveries from tool errors.

Outcome-based reinforcement learning. Starting from the SFT checkpoint, we apply Group Relative Policy Optimization (GRPO) using the environment’s verification suite $\mathcal { V } = ( \bar { \mathcal { V } _ { \mathrm { c o d e } } } , \mathcal { V } _ { \mathrm { L L M } } )$ as outcome supervision. For each environment, we sample a group of trajectories and denote the scores assigned to $\tau _ { i }$ by the two verifier components as $s _ { i } : = \mathcal { V } _ { \mathrm { c o d e } } ( \tau _ { i } ) \in [ 0 , 1 ]$ and $c _ { i } : = \mathcal { V } _ { \mathrm { L L M } } ( \tau _ { i } ) \in [ 0 , 1 ]$ The code-based score checks required artifacts and deterministic constraints, while the LLM-based score applies a reference-conditioned rubric to semantic correctness, coverage, contradictions, and other task-specific criteria. We combine them as

$$
R \varepsilon ( \tau _ { i } ) = \left\{ \begin{array} { l l } { s _ { i } ^ { \frac { w _ { s } } { w _ { s } + w _ { c } } } c _ { i } ^ { \frac { w _ { c } } { w _ { s } + w _ { c } } } , } & { s _ { i } > 0 \mathrm { ~ a n d ~ } c _ { i } > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{5}
$$

where $w _ { s } , w _ { c } > 0$ determine the relative importance of executable verification and semantic correctness. Their normalized exponents lie in (0, 1) and sum to one, so only the ratio $w _ { s } : w _ { c }$ affects the reward. The weighted geometric mean favors trajectories that score well on both criteria and assigns zero reward whenever either score is zero. Compared with an arithmetic mean using the same normalized weights, it more strongly penalizes trajectories with an imbalanced score pair. We use standard GRPO updates and exclude groups with zero reward variance, since equal rewards provide no within-group reward differences for estimating relative advantages. For MoE models, we freeze the router weights during training (Blakeman et al., 2025).

On-policy distillation. We initialize the student from the base checkpoint used before agentic post-training and use a same-size SFT or SFT+RL checkpoint as the teacher. The student interacts with the environment to collect trajectories. At each agent-generated token position, the teacher provides a next-token probability distribution conditioned on the same interaction prefix. We train the student to match this distribution by minimizing $D _ { \mathrm { K L } } ( \pi _ { T } \left. \pi _ { \theta } \right)$ at these positions. User messages, tool observations, and workspace events supply context but are not prediction targets. This provides teacher supervision on interaction histories encountered by the student itself.

## 5 EXPERIMENTS

Our experiments examine the distribution and complexity of the synthesized data, its downstream training value, and the trade-off between agentic gains and general-capability retention through four questions:

Q1. How are the synthesized tasks distributed across work scenarios and capabilities, and what interaction and execution demands do they place on agents? (Section 5.2)

Q2. Does training on the synthesized data yield agentic gains across model scales and evaluation benchmarks? (Section 5.3)

Q3. How do environment exposure, multi-turn supervision, harness diversity, and trajectory selection affect the training value and transferability of the synthesized data? (Section 5.4)

Q4. Can on-policy distillation transfer agentic gains while reducing the loss of general capabilities relative to direct SFT? (Section 5.5)

## 5.1 EXPERIMENTAL SETUP

We evaluate models on six benchmarks: Workspace-Bench (Tang et al., 2026), ALE (Agents’ Last Exam) (Sun et al., 2026), WildClawBench (Ding et al., 2026), PawBench (The OpenJudge Team, 2026), ClawEval (Ye et al., 2026), and ClawMark (Meng et al., 2026). These benchmarks span file-centric workflows, professional tasks, long-horizon tool use, multimodal inputs, and dynamic multi-turn interaction. We use each benchmark’s official tasks, metrics, and evaluators. OpenClaw serves as the default harness unless another runtime is required; the main results report PawBench under OpenClaw, with QwenPaw (AgentScope Team, 2026) and Hermes Agent (Nous Research, 2026) used only for cross-harness analysis.

Our base checkpoints are Qwen3.5-9B, Qwen3.6-27B, and Qwen3.6-35B-A3B (Qwen Team, 2026a;b;c). The comparison models are MiniMax-M2.5, GLM-5.2, Kimi-K2.5, DeepSeek-V4- Pro, Agents-A1, Nex-N2-mini, and Macaron-V1-Tall (MiniMax, 2026; Z.ai, 2026; Kimi Team, 2026; DeepSeek-AI, 2026; Bai et al., 2026; Nex AGI, 2026; Mind Lab, 2026).

## 5.2 DATA SCALE AND DIVERSITY

We characterize the synthesized data from the environment and trajectory aspects. Figure 5 reports the scale and composition of the environment pool, while Table 1 summarizes the interaction and execution structure of the collected trajectories.

Environment coverage. The pipeline synthesizes 64,888 environments spanning 14 application scenarios and 17 capabilities, with variation across scenarios, capabilities, and deliverables (Figure 5). Scenario coverage is broad but non-uniform: office productivity and software engineering form the largest groups, while less frequent domains remain represented. These statistics capture the scale and compositional breadth of the environment pool, supporting the pipeline’s capacity to synthesize heterogeneous task configurations without implying uniform coverage across labels.

Trajectory structure. Table 1 summarizes the corpus scale and execution distributions. Validitybased filtering first retains 24,732 of 77,729 raw trajectories (31.82%), after which model-conditioned selection retains 17,312 trajectories for training. The evidence-valid set contains a median of 10 tool-using assistant steps per trajectory, with the middle 50% spanning 6–16 steps, compared with 6–19 steps in the raw corpus. Single- and multi-turn trajectories account for 67.01% and 32.99% of the raw corpus, respectively. Single-turn interactions can involve substantial tool execution: their median step count is 10, with an interquartile interval of 6–17 steps, while multi-turn trajectories have a median of 15 and an interval of 9–24 steps. The difference also extends into the upper tail: 32.69% of multi-turn trajectories exceed 20 tool-using steps, compared with 15.80% of single-turn trajectories. These statistics capture complementary aspects of interaction structure: the number of user turns and the depth of tool execution within a trajectory.

(a) Application scenarios  
![](images/0c0ee80efd5371bc001c25203d348685b80aa06a08ba4beec46b807c895e6700.jpg)

(b) Capability distribution  
![](images/4d1e2bba05bce087969d34f16d3439a5289e10477a43b4e2eab29caa44b022e9.jpg)

(c) Output-format distribution  
![](images/daefe30c948e6dd70de8267c0f133f1b8e05633473adbd722bdca5451550ee3c.jpg)

![](images/d09ce588873ba1a872e5b899b92d9d73d61c1d756b2674e3005df14d8a494845.jpg)  
Figure 5: Distribution across (a) application scenarios, (b) capability labels, (c) output formats, and (d) persona categories. Percentages are computed over 64,888 environments.  
Table 1: Trajectory scale and interaction structure. Panel (a) summarizes SFT trajectory filtering, RL environment scale, and execution depth after judge filtering; panel (b) describes interaction composition in the raw corpus.

(a) Collection and filtering  
(b) Interaction composition
<table><tr><td>Measure</td><td>Statistic</td><td>Measure</td><td>Percentage</td></tr><tr><td>Training data</td><td></td><td>Interaction types</td><td></td></tr><tr><td>Raw trajectories</td><td>77,729</td><td>Single-turn trajectories</td><td>67.01%</td></tr><tr><td>After validity-based filtering</td><td>24,732</td><td>Multi-turn trajectories</td><td>32.99%</td></tr><tr><td>After model-conditioned selection (SFT trained)</td><td>17,312</td><td></td><td></td></tr><tr><td>RL training environments</td><td>2,873</td><td></td><td></td></tr><tr><td>Trajectories after validity-based filtering</td><td></td><td>User turns (multi-turn)</td><td></td></tr><tr><td>Tool-using steps, median</td><td>10</td><td>2–4 turns</td><td>63.16%</td></tr><tr><td>&gt; 20 tool-using steps</td><td>3,437 (13.90%) 5–8 turns</td><td></td><td>22.53%</td></tr><tr><td>&gt; 40 tool-using steps</td><td>303 (1.23%)</td><td>≥ 9 turns</td><td>14.31%</td></tr></table>

Note. Model-conditioned selection uses NLL/RSR. A tool-using step is an assistant message containing at least one tool call. Single- and multi-turn denote trajectories with one and more than one user-role message, respectively.

Table 2: Main results on six agentic benchmarks under OPENCLAW. Entries report average score/pass rate where both are available and average score otherwise; Avg. denotes the mean score across the six benchmarks, and MoE size is total / activated parameters. The best and second-best scores in each metric column are bolded and underlined, respectively.
<table><tr><td>Model</td><td>Params.</td><td> $\operatorname { A v g } .$ </td><td>Workspace Bench</td><td>ALE</td><td>WildClaw Bench</td><td>PawBench</td><td>ClawEval</td><td>ClawMark</td></tr><tr><td colspan="9">&gt;100B models</td></tr><tr><td>MiniMax-M2.5</td><td>230B / 10B</td><td>0.4084</td><td>0.2517</td><td>0.2622 / 9.90%</td><td>0.3936</td><td>0.5431 / 10.75%</td><td>0.6526 / 59.60%</td><td>0.3472</td></tr><tr><td>GLM-5.2</td><td>744B / 40B</td><td>0.6157</td><td>0.5178</td><td>0.3040 / 12.80%</td><td>0.5866</td><td>0.7878 / 20.70%</td><td>0.7601 / 80.70%</td><td>0.7377</td></tr><tr><td>Kimi-K2.5</td><td>1T/32B</td><td>0.4930</td><td>0.3949</td><td>0.2880 / 11.30%</td><td>0.3486</td><td>0.7331 / 13.44%</td><td>0.6967 / 69.6%</td><td>0.4968</td></tr><tr><td>DeepSeek-V4-Pro</td><td>1.6T / 49B</td><td>0.5744</td><td>0.5525</td><td>0.2760 / 12.40%</td><td>0.4614</td><td>0.7729 / 15.32%</td><td>0.7727 / 75.5%</td><td>0.6109</td></tr><tr><td colspan="9">9B models</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>0.3627</td><td>0.2372</td><td>0.0205 / 0.70%</td><td>0.2302</td><td>0.7071 / 24.55%</td><td>0.6535 / 62.10%</td><td>0.3276</td></tr><tr><td>Ours-9B</td><td>9B</td><td>0.4494</td><td>0.3836</td><td>0.1544 / 4.30%</td><td>0.2438</td><td>0.7699 / 31.82%</td><td>0.7259 / 75.80%</td><td>0.4185</td></tr><tr><td colspan="9">35B-A3B models</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>35B / 3B</td><td>0.5251</td><td>0.5077</td><td>0.1794 / 5.00%</td><td>0.4300</td><td>0.7739 / 30.91%</td><td>0.7254 / 67.70%</td><td>0.5344</td></tr><tr><td>Agents-A1</td><td>35B/ 3B</td><td>0.4515</td><td>0.3599</td><td>0.2263 / 9.20%</td><td>0.2794</td><td>0.7045 / 25.3%</td><td>0.7331 / 78.30%</td><td>0.4057</td></tr><tr><td>Nex-N2-mini</td><td>35B/ 3B</td><td>0.5014</td><td>0.4426</td><td>0.2112 / 7.10%</td><td>0.3201</td><td>0.6878 / 21.77%</td><td>0.7389 / 78.90%</td><td>0.6078</td></tr><tr><td>Macaron-V1</td><td>35B / 3B</td><td>0.4269</td><td>0.2562</td><td>0.1617 / 7.10%</td><td>0.2721</td><td>0.7120 / 18.55%</td><td>0.6796 / 70.20%</td><td>0.4796</td></tr><tr><td>Ours-35B-A3B</td><td>35B / 3B</td><td>0.5665</td><td>0.5227</td><td>0.2584 / 8.50%</td><td>0.4248</td><td>0.8187 / 37.27%</td><td>0.7734 / 82.00%</td><td>0.6009</td></tr><tr><td colspan="9">27B models</td></tr><tr><td>Qwen3.6-27B</td><td>27B</td><td>0.5478</td><td>0.5230</td><td>0.2311 / 8.50%</td><td>0.3725</td><td>0.8057 / 39.09%</td><td>0.7311 / 68.90%</td><td>0.6233</td></tr><tr><td>Ours-27B</td><td>27B</td><td>0.6119</td><td>0.5409</td><td>0.2903 / 10.6%</td><td>0.4241</td><td>0.8400 / 41.82%</td><td>0.8508 / 80.6%</td><td>0.7250</td></tr></table>

## 5.3 MAIN RESULTS

Our models in Table 2 are first trained with SFT on 17,312 trajectories selected from the 24,732 evidence-valid trajectories retained from an initial collection of 77,729. Starting from the SFT checkpoints, we then perform outcome-based RL on 2,873 environments. Averaged over the six benchmarks, the 9B, 27B, and 35B-A3B models improve over their matched baselines by +8.7, +6.4, and +4.1 points, respectively, with gains in 17 of the 18 model–benchmark comparisons. Beyond this broad improvement, the gains exhibit distinct patterns across backbones: 9B benefits most on Workspace-Bench (+14.64 points) and ALE (+13.39), whereas 27B improves most on ClawEval (+11.97) and ClawMark (+10.17). This suggests that the same training collection helps address different weaknesses across models, rather than yielding gains that simply increase with parameter count.

Ours-27B ranks second by the six-benchmark mean (0.6119 versus 0.6157 for GLM-5.2), while achieving the best scores on PawBench and ClawEval among all evaluated models, including those with over 100B total parameters. Within the 35B-A3B group, Ours-35B-A3B also achieves the highest mean score. The gains nevertheless have a clear boundary: on WildClawBench, 9B improves by only 1.36 points and 35B-A3B declines by 0.52 points, while all three of our models remain substantially behind GLM-5.2.

## 5.4 ABLATION STUDIES

To better understand the factors underlying the observed performance gains, we next examine four complementary dimensions of the data pipeline: how supervised environment exposure interacts with model capacity; the value of interactive supervision from multi-turn trajectories; cross-harness transfer across execution runtimes; and trajectory selection via model-conditioned filtering.

## 5.4.1 SUPERVISED ENVIRONMENT EXPOSURE

![](images/671003b4dc258cf26463365e6f70a4b254726fd3439480be9669421a8f61334c.jpg)

![](images/0092a35169bd9eaa9fa2cc20113d24cf76485b9cdbcbb0104292918a2df60255.jpg)

![](images/a22c876185efae2200cef6d870723da015519c0128b3939c5a61544f446105f0.jpg)  
Figure 6: ClawEval performance of Ours-9B, Ours-27B, and Ours-35B-A3B over supervised training, with progress normalized to the final checkpoint.

Figure 6 tracks ClawEval performance from the initial to the final supervised-training checkpoint. Comparing these endpoints, all three model variants improve across all three metrics. The 9B model achieves the largest gains in Score (+4.66) and pass@3 (+5.00 percentage points), showing that the smaller dense model continues to acquire agentic capabilities throughout training. The 27B model also improves at intermediate checkpoints and achieves the highest final performance across all three metrics. The 35B-A3B model exhibits a more metric-dependent trajectory: Score and pass@3 plateau relatively early, whereas pass<sup>3</sup> continues to improve and yields the largest endpoint gain in this metric (+5.80 percentage points). These patterns indicate that the benefits of continued supervised training accrue at different rates across model scales and metrics; a plateau in one metric does not necessarily imply that further training provides no benefit on others.

## 5.4.2 INTERACTIVE SUPERVISION

Table 3: Effect of interactive supervision on 27B models. Metrics are reported on a 0–100 scale; green badges indicate absolute gains over Qwen3.6-27B.
<table><tr><td rowspan="2">Model</td><td rowspan="2">turn</td><td rowspan="2">Single Multi turn</td><td colspan="2">ClawEval</td><td colspan="2">ClawEval-MT</td><td rowspan="2">ClawMark Score</td></tr><tr><td>Score</td><td>Pass@3 (%)</td><td>Score</td><td>Pass (%)</td></tr><tr><td>Qwen3.6-27B</td><td>一</td><td>一</td><td>73.11</td><td>68.9</td><td>80.57</td><td>71.1</td><td>62.33</td></tr><tr><td>Ours-27B</td><td>√</td><td>一</td><td>84.13 +11.02</td><td>79.7 +10.8</td><td>84.01+3.44</td><td>83.3+12.2</td><td>69.27+6.94</td></tr><tr><td>Ours-27B</td><td>√</td><td>V</td><td>85.08 +11.97</td><td>80.6 +11.7</td><td>84.44 +3.87</td><td>81.6 +10.5</td><td>72.50+10.17</td></tr></table>

Table 3 shows that adding multi-turn trajectories to the single-turn training set improves average scores across all three benchmarks, with the largest gain on ClawMark (+3.23 points). This suggests that interactive supervision provides complementary benefits for cowork task performance.

## 5.4.3 CROSS-HARNESS TRANSFER

Table 4: Cross-harness generalization of Qwen3.5-9B on PawBench. Results are pass rates; green badges show absolute gains over the base model, with darker shades indicating larger gains.
<table><tr><td>Training source</td><td>QwenPaw SEEN</td><td>OpenClaw SEEN</td><td>Hermes UNSEEN</td><td></td></tr><tr><td>Base model</td><td>20.0</td><td>24.5</td><td>15.5</td><td></td></tr><tr><td>QwenPaw only</td><td>29.1 +9.1</td><td>27.3 +2.8</td><td>22.7 +7.2</td><td></td></tr><tr><td>OpenClaw only</td><td>30.0 +10.0</td><td>29.1 +4.6</td><td>24.5 +9.0</td><td></td></tr><tr><td>QwenPaw + OpenClaw</td><td>31.8 +11.8</td><td>31.8 +7.3</td><td>24.5 +9.0</td><td></td></tr></table>

Table 4 shows that the benefits of our training data extend beyond the harness used for collection. Training on either source improves pass rates across all three evaluation harnesses, including Hermes, which is never used for data collection: QwenPaw-only and OpenClaw-only training yield absolute gains of +7.2 and +9.0 points on Hermes, respectively. Transfer is also evident between the two collection harnesses. In particular, OpenClaw-only training improves performance on QwenPaw by +10.0 points, exceeding the gain from QwenPaw-only training (+9.1 points) despite the mismatch between collection and evaluation harnesses. These findings suggest that the data teaches reusable capabilities for solving cowork tasks, with benefits that are not confined to harness-specific interaction patterns. Combining both sources achieves the highest pass rates on QwenPaw and OpenClaw (31.8% on both), while retaining the full improvement on the unseen Hermes harness. Taken together, the results support the cross-harness generalizability of our data and its effectiveness in improving cowork task performance across different runtimes.

## 5.4.4 TRAJECTORY SELECTION

Table 5: Effect of model-conditioned trajectory filtering across model sizes. NLL and RSR each retain 70% of the candidate trajectories.
<table><tr><td>Selection</td><td>ClawEval Avg.</td><td>PawBench Pass (%)</td><td>ClawMark Avg.</td></tr><tr><td colspan="4">Qwen3.5-9B</td></tr><tr><td>Base model</td><td>0.6535</td><td>24.55</td><td>0.3276</td></tr><tr><td>No filter</td><td>0.7238</td><td>27.88</td><td>0.3267</td></tr><tr><td>NLL</td><td>0.7295</td><td>29.39</td><td>0.4010</td></tr><tr><td>RSR</td><td>0.7174</td><td>28.48</td><td>0.3991</td></tr><tr><td colspan="4">Qwen3.6-27B</td></tr><tr><td>Base model</td><td>0.7311</td><td>39.09</td><td>0.6233</td></tr><tr><td>No filter</td><td>0.8065</td><td>39.09</td><td>0.6557</td></tr><tr><td>NLL</td><td>0.8078</td><td>43.03</td><td>0.6806</td></tr><tr><td>RSR</td><td>0.7971</td><td>43.94</td><td>0.6730</td></tr><tr><td colspan="4">Qwen3.6-35B-A3B</td></tr><tr><td>Base model</td><td>0.7254</td><td>30.91</td><td>0.5344</td></tr><tr><td>No filter</td><td>0.7667</td><td>39.39</td><td>0.5394</td></tr><tr><td>NLL</td><td>0.7682</td><td>40.91</td><td>0.5678</td></tr><tr><td>RSR</td><td>0.7676</td><td>43.03</td><td>0.5739</td></tr></table>

Base model is the original checkpoint before post-training. No filter uses the same post-training recipe without NLL/RSR selection. NLL and RSR filter independently within ten equal-frequency log-length bins. Bold and underlined values denote the best and second-best result within each model size.

Retaining only 70% of the evidence-valid pool, NLL selection improves every reported metric across all three model sizes. This result indicates that validity is necessary but does not make all trajectories equally valuable for learning. The preferred criterion also shifts with scale: NLL is strongest at 9B and 27B, whereas RSR performs best on most metrics at 35B-A3B. This reversal suggests that trajectory value is not intrinsic to the data alone, but depends on the capacity and predictive distribution of the target model. Since selection is performed within length bins, the effect cannot be readily attributed to length bias. A practical implication is that filtering criteria should be calibrated to the student model rather than applied uniformly across scales.

## 5.5 ON-POLICY DISTILLATION

![](images/c19c7c5a3c4509ce28480e4ef7c2d6bc21bfadb441ee8de4fdb29febf33823fd.jpg)  
Base  SFT teacherOPD student  
Figure 7: Agentic transfer and general-capability retention under OPD. Dashed lines mark SFT teacher performance; exposure is normalized to the final checkpoint.

We finally examine whether agentic gains can be transferred while preserving broader model capabilities. On ClawEval, OPD improves pass<sup>3</sup> by +4.5 points over the base model, recovering approximately 80% of the teacher’s gain and leaving a gap of only 1.1 points. On GPQA-Diamond, the OPD student remains close to the base model while outperforming the SFT teacher by +4.54 points. This cross-benchmark pattern suggests that OPD transfers most of the specialized agentic capability while substantially limiting degradation in general performance.

## 6 RELATED WORK

Executable agent environments. WebArena and OSWorld established reproducible, executable testbeds for web and computer-use agents (Zhou et al., 2023; Xie et al., 2024). Recent work frames generalization in terms of scaling executable rules and state transitions, and automates environment discovery, task synthesis, validation, or solver-adaptive generation (Zhang et al., 2026; Dong et al., 2026; Yang et al., 2026a; Zhai et al., 2025; Li et al., 2026; Shi et al., 2026b; Chae et al., 2026; Kwan et al., 2026; Wolf et al., 2026). Complementary systems convert existing web, mobile, and desktop software into trainable environments (Aggarwal et al., 2026; Yuan et al., 2026; Kar et al., 2026; Savadikar et al., 2026; Wu et al., 2026). Our pipeline applies these principles to personal work, jointly constructing heterogeneous workspaces, interaction interfaces, user policies, and outcome verifiers, with package-level validation before rollout.

Workspace and cowork evaluation. Workspace-Bench evaluates reasoning over large, heterogeneous file collections, while Agents’ Last Exam and WildClawBench target economically grounded and long-horizon professional workflows (Tang et al., 2026; Sun et al., 2026; Ding et al., 2026). ClawMark further introduces multi-turn, evolving service state (Meng et al., 2026). Claw-Eval and AJ-Bench emphasize trajectory, environment-state, and process evidence for reliable grading (Ye et al., 2026; Shi et al., 2026a). These benchmarks characterize end-to-end agent systems; our work synthesizes the training environments and verified interaction data required to improve such systems.

Stateful user–agent interaction. ToolSandbox and τ-bench combine tool execution with simulated multi-turn users and state-based evaluation (Lu et al., 2024; Yao et al., 2024); τ<sup>2</sup>-Bench gives both participants tools over a shared environment (Barres et al., 2025). VISTA extends user actions across UI and API interfaces (Lu et al., 2026a). Human-grounded studies find that general-purpose language models often simulate users with excessive cooperation and limited behavioral diversity, motivating data-grounded or explicitly optimized simulators (Zhou et al., 2026b; Zhu et al., 2026; Wang et al.,

2026; Zhou et al., 2026a). We model task-conditioned interventions that can revise requirements, reject intermediate results, or mutate the shared workspace.

Agent post-training and harness effects. Interactive post-training combines online curricula, environment-grounded reinforcement learning, long-horizon trajectories, and on-policy distillation (Qi et al., 2024; Cheng et al., 2025; GLM-5 Team, 2026; Bai et al., 2026; Lu et al., 2026b; Shenfeld et al., 2026). Runtime research exposes context, memory, tools, scheduling, and access control as system-level services (Mei et al., 2024); Harness-Bench and PawBench measure substantial variation across model–harness configurations (Yao et al., 2026; The OpenJudge Team, 2026), while EvoTrainer co-evolves policies and training-side diagnostics (Chen et al., 2026). We treat the harness as an explicit rollout variable and derive SFT, RL, and distillation signals from validated environments and filtered trajectories.

## 7 LIMITATIONS AND FUTURE DIRECTIONS

Task coverage. Our environments are synthetic approximations of personal work. Taxonomyguided generation may omit tasks, cultures, languages, accessibility needs, and failure modes, while resource-grounded generation inherits the limits of available workspaces and interfaces. The current pool also emphasizes browser, file, data, and office workflows, leaving complex visual interfaces, specialized desktop software, and richer audio or video tasks underrepresented. Its scale should therefore not be interpreted as representative coverage. Future work should compare the synthesized distribution with real-world task distributions and extend multimodal state inspection and verification.

Verifier reliability. Verifier quality limits both data selection and reinforcement learning. Programmatic checks can be brittle, while language-model graders can be biased or overly permissive. Positive, negative, and mutation tests reduce but do not eliminate these risks, particularly for open-ended artifacts. Larger-scale human audits and calibration studies are needed to quantify false-acceptance and false-rejection rates.

## 8 CONCLUSION

We introduced Cowork-Gym, a scalable framework for synthesizing executable Cowork environments. It combines taxonomy-guided and resource-grounded task generation with a build–execute–revise loop that constructs and validates the task, workspace, interfaces, user policy, and verifier as a complete environment. A stateful user simulator supports multi-turn collaboration through dialogue and shared-workspace edits. Using this framework, we synthesized 64.9K environments across 14 application scenarios and collected filtered trajectories for post-training.

Using these data, we post-trained 9B, 27B, and 35B-A3B Qwen models, all of which improve over their corresponding base checkpoints on average across six agentic benchmarks. Ours-27B ranks second overall and achieves the highest average scores on PawBench and ClawEval, outperforming evaluated models with more than 100B total parameters on both.

Scalable construction of executable environments can provide a data-driven path toward self-evolving agents. As tools, workspaces, and user requirements evolve, the environment pool can expand accordingly, supplying new tasks, interactions, and verification signals for subsequent training. This offers a path toward agents that improve continuously while adapting to changing work settings.

## REFERENCES

AgentScope Team. QwenPaw: Your personal AI assistant, 2026. URL https://github.com /agentscope-ai/QwenPaw. Software repository, accessed July 28, 2026.

Pranjal Aggarwal, Graham Neubig, and Sean Welleck. Gym-Anything: Turn any software into an agent environment, 2026. URL https://arxiv.org/abs/2604.06126.

Anthropic. Claude cowork, 2026. URL https://www.anthropic.com/product/claude -cowork. Product page, accessed July 26, 2026.

Lei Bai et al. Scaling the horizon, not the parameters: Reaching trillion-parameter performance with a 35B agent, 2026. URL https://arxiv.org/abs/2606.30616.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment, 2025. URL https://arxiv.org/abs/ 2506.07982.

Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, et al. Nemotron 3 Nano: Open, efficient mixture-of-experts hybrid Mamba–Transformer model for agentic reasoning, 2025. URL https://arxiv.org/abs/2512.20848.

Hyungjoo Chae, Jungsoo Park, and Alan Ritter. Safe and scalable web agent learning via recreated websites, 2026. URL https://arxiv.org/abs/2603.10505.

Guhong Chen, Yingcheng Shi, Yongbin Li, Binhua Li, Xander Xu, Hu Wei, Shiwen Ni, Min Yang, and Jieping Ye. EvoTrainer: Co-evolving LLM policies and training harnesses for autonomous agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2606.03108.

Mingyue Cheng, Shuo Yu, Daoyu Wang, Qingchuan Li, Xiaoyu Tao, Jie Ouyang, Yucong Luo, Yitong Zhou, Qi Liu, and Enhong Chen. Agent-R1: A unified and modular framework for agentic reinforcement learning, 2025. URL https://arxiv.org/abs/2511.14460.

DeepSeek-AI. DeepSeek-V4-Pro, 2026. URL https://huggingface.co/deepseek-ai/ DeepSeek-V4-Pro. Model card, accessed September 28, 2026.

Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Jingyi Yang, Penghui Yang, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. WildClawBench: A benchmark for real-world, long-horizon agent evaluation, 2026. URL https://arxiv.org/abs/2605.10912.

Guanting Dong et al. Agent-World: Scaling real-world environment synthesis for evolving general agent intelligence, 2026. URL https://arxiv.org/abs/2604.18292.

GLM-5 Team. GLM-5: From vibe coding to agentic engineering, 2026. URL https://arxiv. org/abs/2602.15763.

Harbor Framework Team. Harbor: A framework for evaluating and optimizing agents and models in container environments, 2026. URL https://doi.org/10.5281/zenodo.20953922.

Oguzhan Fatih Kar, Roman Bachmann, Yuanzheng Gong, Anders Boesen Lindbo Larsen, and Afshin˘ Dehghan. Weblica: Scalable and reproducible training environments for visual web agents, 2026. URL https://arxiv.org/abs/2605.06761.

Kimi Team. Kimi K2.5: Visual agentic intelligence, 2026. URL https://arxiv.org/abs/ 2602.02276.

Wai-Chung Kwan, Aryo Pradipta Gema, Joshua Ong Jun Leang, and Pasquale Minervini. SCOPE: Self-play via co-evolving policies for open-ended tasks, 2026. URL https://arxiv.org/ abs/2605.31433.

Xirui Li, Ming Li, Ion Stoica, Cho-Jui Hsieh, and Tianyi Zhou. ClawEnvKit: Automatic environment generation for claw-like agents, 2026. URL https://arxiv.org/abs/2604.18543.

Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Felix Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. ToolSandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities, 2024. URL https://arxiv.org/abs/2408.04682.

Yunan Lu, Ryan Shea, Yusen Zhang, and Zhou Yu. VISTA: A versatile interactive user simulation toolkit for agent evaluation, 2026a. URL https://arxiv.org/abs/2606.11079.

Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Self-distilled agentic reinforcement learning, 2026b. URL https://arxiv.org/abs/2605.15155.

Shinji Mai, Yunpeng Zhai, Ziqian Chen, Cheng Chen, Anni Zou, Shuchang Tao, Zhaoyang Liu, and Bolin Ding. CuES: A curiosity-driven and environment-grounded synthesis framework for agentic RL, 2025. URL https://arxiv.org/abs/2512.01311.

Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. AIOS: LLM agent operating system, 2024. URL https://arxiv.org/abs/2403.16971.

Fanqing Meng, Lingxiao Du, Zijian Wu, Guanzheng Chen, Xiangyan Liu, Jiaqi Liao, Chonghe Jiang, Zhenglin Wan, Jiawei Gu, Pengfei Zhou, Rui Huang, Ziqi Zhao, Shengyuan Ding, et al. ClawMark: A living-world benchmark for multi-turn, multi-day, multimodal coworker agents, 2026. URL https://arxiv.org/abs/2604.23781.

Mind Lab. Macaron-V1: Towards open continual learning with self-improvement and mixture-of-LoRA, 2026. URL https://arxiv.org/abs/2608.09819.

MiniMax. MiniMax-M2.5, 2026. URL https://huggingface.co/MiniMaxAI/MiniMax -M2.5. Model card, accessed September 28, 2026.

Nex AGI. Nex-N2-mini, 2026. URL https://huggingface.co/nex-agi/Nex-N2-mini. Model card, accessed September 28, 2026.

Nous Research. Hermes Agent: The agent that grows with you, 2026. URL https://github.c om/NousResearch/hermes-agent. Software repository, accessed July 28, 2026.

OpenAI. The ChatGPT and Codex experience is coming together, July 2026. URL https:// openai.com/index/introducing-upgrades-to-codex/. Product announcement, accessed July 26, 2026.

OpenClaw. OpenClaw: The personal AI assistant for everyone, 2026. URL https://openclaw .ai/. Project website, accessed July 26, 2026.

Hoang Phan, Xianjun Yang, Kevin Yao, Jingyu Zhang, Shengjie Bi, Xiaocheng Tang, Madian Khabsa, Lijuan Liu, and Deren Lei. Beyond reasoning gains: Mitigating general-capability forgetting in large reasoning models, 2025. URL https://arxiv.org/abs/2510.21978.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, Tianjie Zhang, Wei Xu, Jie Tang, and Yuxiao Dong. WebRL: Training LLM web agents via self-evolving online curriculum reinforcement learning, 2024. URL https://arxiv.org/abs/2411.02337.

Qwen Team. Qwen3.5-9B, 2026a. URL https://huggingface.co/Qwen/Qwen3.5-9B. Model card, accessed September 28, 2026.

Qwen Team. Qwen3.6-27B, 2026b. URL https://huggingface.co/Qwen/Qwen3.6-27B. Model card, accessed September 28, 2026.

Qwen Team. Qwen3.6-35B-A3B, 2026c. URL https://huggingface.co/Qwen/Qwen3. 6-35B-A3B. Model card, accessed September 28, 2026.

Chinmay Savadikar, Mingyu Zhao, Yuanzheng Zhu, Han Li, Shuang Xie, Alberto Castelo, Tianfu Wu, and Lingyun Wang. ShopGym: An integrated framework for realistic simulation and scalable benchmarking of e-commerce web agents, 2026. URL https://arxiv.org/abs/2605.1 6116.

Idan Shenfeld, Mehul Damani, Jonas Hubotter, and Pulkit Agrawal. Self-distillation enables continual¨ learning, 2026. URL https://arxiv.org/abs/2601.19897.

Wentao Shi, Yu Wang, Yuyang Zhao, Yuxin Chen, Fuli Feng, Xueyuan Hao, Xi Su, Qi Gu, Hui Su, Xunliang Cai, and Xiangnan He. AJ-Bench: Benchmarking agent-as-a-judge for environmentaware evaluation, 2026a. URL https://arxiv.org/abs/2604.18240.

Yucheng Shi, Zhenwen Liang, Kishan Panaganti, Dian Yu, Wenhao Yu, and Haitao Mi. Learning to build the environment: Self-evolving reasoning RL via verifiable environment synthesis, 2026b. URL https://arxiv.org/abs/2605.14392.

Yiyou Sun et al. Agents’ last exam, 2026. URL https://arxiv.org/abs/2606.05405.

Zirui Tang, Xuanhe Zhou, Yumou Liu, Linchun Li, Yukai Wu, Weizheng Wang, Hongzhang Huang, Wei Zhou, Jun Zhou, Jiachen Song, Shaoli Yu, Jinqi Wang, Zihang Zhou, Hongyi Zhou, Yuting Lv, Jinyang Li, Jiashuo Liu, Ruoyu Chen, Chunwei Liu, GuoLiang Li, Jihua Kang, and Fan Wu. Workspace-Bench 1.0: Benchmarking AI agents on workspace tasks with large-scale file dependencies, 2026. URL https://arxiv.org/abs/2605.03596.

The OpenJudge Team. PawBench: A benchmark for evaluating LLM × harness performance, June 2026. URL https://github.com/agentscope-ai/PawBench.

Yingshan Susan Wang, Cedegao E. Zhang, Linlu Qiu, Zexue He, Pengyuan Li, Alex Pentland, Roger P. Levy, and Yoon Kim. Learning user simulators with turing rewards, 2026. URL https://arxiv.org/abs/2606.19336.

Lorenz Wolf, Connor Watts, Roger Creus Castanyer, Geoffrey Bradway, Maxwill Lin, Augustine N. Mavor-Parker, and Matthew Daborn-Sargent. Breaking the solver bottleneck: Training task generators at the learnable frontier, 2026. URL https://arxiv.org/abs/2606.18284.

Dingbang Wu, Rui Hao, Haiyang Wang, Shuzhe Wu, Han Xiao, Zhenghong Li, Bojiang Zhou, Zheng Ju, Zichen Liu, Lue Fan, and Zhaoxiang Zhang. MobileGym: A verifiable and highly parallel simulation platform for mobile GUI agent research, 2026. URL https://arxiv.org/abs/ 2605.26114.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024. URL https://arxiv.org/abs 2404.07972.

Shidong Yang, Ziyu Ma, Tongwen Huang, Yiming Hu, Yong Wang, and Xiangxiang Chu. CoEvolve: Training LLM agents via agent-data mutual evolution, 2026a. URL https://arxiv.org/ab s/2604.15840.

Yuming Yang, Mingyoung Lai, Wanxu Zhao, Xiaoran Fan, Zhiheng Xi, Mingqi Wu, Chiyue Huang, Jun Zhao, Haijun Lv, Jian Tong, Yunhua Zhou, Yicheng Zou, Qipeng Guo, Tao Gui, Qi Zhang, and Xuanjing Huang. Which reasoning trajectories teach students to reason better? a simple metric of informative alignment. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 42123–42150, San Diego, California, United States, July 2026b. Association for Computational Linguistics. doi: 10.18653/v1/2026.acl -long.1950. URL https://aclanthology.org/2026.acl-long.1950/.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains, 2024. URL https://arxiv.org/abs/ 2406.12045.

Yilun Yao, Xinyu Tan, Chao-Hsuan Liu, Yaoming Li, Zhengyang Wang, Wenhan Yu, Zhewen Tan, Yuxuan Tian, Guangxiang Zhao, Lin Sun, Xiangzheng Zhang, and Tong Yang. Harness-Bench: Measuring harness effects across models in realistic agent workflows, 2026. URL https: //arxiv.org/abs/2605.27922.

Bowen Ye, Rang Li, Qibin Yang, Yuanxin Liu, Linli Yao, Hanglong Lv, Zhihui Xie, Chenxin An, Lei Li, Lingpeng Kong, Qi Liu, Zhifang Sui, and Tong Yang. Claw-Eval: Towards trustworthy evaluation of autonomous agents, 2026. URL https://arxiv.org/abs/2604.06132.

Peng Yuan, Yuyang Yin, Yuxuan Cai, and Zheng Wei. WebForge: Breaking the realismreproducibility-scalability trilemma in browser agent benchmark, 2026. URL https://arxiv. org/abs/2604.10988.

Z.ai. GLM-5.2, 2026. URL https://huggingface.co/zai-org/GLM-5.2. Model card, accessed September 28, 2026.

Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, and Jingren Zhou. AgentEvolver: Towards efficient self-evolving agent system, 2025. URL https://arxiv.org/abs/2511.10395.

Jiayi Zhang, Fanqi Kong, Guibin Zhang, Maojia Song, Zhaoyang Yu, Jianhao Ruan, Jinyu Xiang, Bang Liu, Chenglin Wu, and Yuyu Luo. Scalable environments drive generalizable agents, 2026. URL https://arxiv.org/abs/2605.18181.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic web environment for building autonomous agents, 2023. URL https://arxiv.org/abs/ 2307.13854.

Xuhui Zhou, Weiwei Sun, Weihua Du, Jiarui Liu, Haojia Sun, Qianou Ma, Tongshuang Wu, Yiming Yang, and Maarten Sap. OdysSim: Building foundation models for human behavior simulation, 2026a. URL https://arxiv.org/abs/2606.14199.

Xuhui Zhou, Weiwei Sun, Qianou Ma, Yiqing Xie, Jiarui Liu, Weihua Du, Sean Welleck, Yiming Yang, Graham Neubig, Sherry Tongshuang Wu, and Maarten Sap. Mind the Sim2Real gap in user simulation for agentic tasks, 2026b. URL https://arxiv.org/abs/2603.11245.

Ming Zhu, Juntao Tan, Rithesh Murthy, Jielin Qiu, Liangwei Yang, Wenting Zhao, Silvio Savarese, Shelby Heinecke, and Huan Wang. RealUserSim: Bridging the reality gap in agent benchmarking via grounded user simulation, 2026. URL https://arxiv.org/abs/2605.20204.

## A TASK TAXONOMY

The taxonomy-guided route uses a versioned taxonomy to control task generation and dataset coverage. The taxonomy separates the operations required by a task from its application domain and other task-level constraints. Labels describe the task to be constructed; they do not specify a solution trace or events observed during rollout.

## A.1 TAXONOMY OVERVIEW

The taxonomy contains six dimensions, summarized in Table 6. Capabilities and most task attributes may take multiple values, whereas each task is assigned one primary application path.

Table 6: The six dimensions of the task taxonomy.
<table><tr><td>Dimension</td><td>Assignment</td><td>What it controls</td></tr><tr><td>Capabilities</td><td>Multi-label</td><td>Operations expected to be required by the task.</td></tr><tr><td>Application scenario</td><td>Single path</td><td>Primary domain and use case.</td></tr><tr><td>Structure and difficulty</td><td>Structured attributes</td><td>Anticipated workflow, feedback, and task depth.</td></tr><tr><td>User and deliverable</td><td>Structured attributes</td><td>Target user, output format, style, and length.</td></tr><tr><td>Modality</td><td>Multi-label</td><td>Input and output channels.</td></tr><tr><td>Safety and alignment</td><td>Multi-label</td><td>Relevant risks and expected response behavior.</td></tr></table>

## A.2 CAPABILITIES

Capability labels are independent of application scenarios. A task may therefore combine several operations within one domain without introducing a new domain-specific capability label.

Table 7: Capability labels and their scope.
<table><tr><td>Capability</td><td>Scope</td></tr><tr><td>Logic_Reasoning</td><td>Causal, deductive, abductive, and constraint-based reasoning.</td></tr><tr><td>Math_Scientific</td><td>Arithmetic, statistical, and scientific calculation.</td></tr><tr><td>Code_Manipulation</td><td>Code generation, debugging, review, refactoring, and testing. Tool selection, argument construction, and chained tool use.</td></tr><tr><td>Tool_Calling Knowledge_Retrieval</td><td>Evidence extraction, cross-document comparison, and retrieval</td></tr><tr><td></td><td>verification.</td></tr><tr><td>Web_Browsing</td><td>Web search, page retrieval, extraction, and interaction.</td></tr><tr><td>Data_Analysis</td><td>Data cleaning, analysis, visualization, and reporting.</td></tr><tr><td>Content_Generation</td><td>Composition and synthesis of written content.</td></tr><tr><td>Multimodal_Perception</td><td>Interpretation of image, video, and audio inputs.</td></tr><tr><td>Multimedia_Processing</td><td>Editing or transformation of image, video, and audio assets.</td></tr><tr><td>Calendar_Scheduling</td><td>Calendar operations and temporal coordination.</td></tr><tr><td>Email_Communication</td><td>Email retrieval, classification, drafting, and sending.</td></tr><tr><td>File_Management</td><td>File operations, conversion, archiving, and directory management.</td></tr><tr><td>Security_Awareness</td><td>Recognition and mitigation of injection, leakage, and social-engineering risks.</td></tr><tr><td>Agent_Self_Config</td><td>Configuration of agents, skills, personas, and scheduled tasks.</td></tr><tr><td>Translation</td><td>Translation and cross-lingual information alignment.</td></tr><tr><td>Spatial_Visual_Reasoning</td><td>Spatial inference, visual puzzles, and geometric reconstruction.</td></tr></table>

## A.3 APPLICATION SCENARIOS

Each task is assigned one primary scenario path. The hierarchy has three levels: a broad domain, a subdomain, and a specific use case. Table 8 summarizes the top two levels and gives one representative leaf for each domain.

Table 8: Top-level application scenarios, subdomains, and representative leaves.
<table><tr><td>Scenario</td><td>Subdomains</td><td>Example leaf</td></tr><tr><td>Office_Productivity</td><td>Email, Calendar, Document, Meeting, Task Management, Expense</td><td>Inbox_Triage</td></tr><tr><td>Information_Retrieval</td><td>Academic, Market, News, Social Platform, Knowledge Base</td><td>Literature_Review</td></tr><tr><td>Finance_Investment</td><td>Stock, Financial Report, Forex, Investment Research</td><td>Earnings_Analysis</td></tr><tr><td>Software_Engineering</td><td>Code, Frontend, DevOps, GitHub, Database</td><td>Debugging</td></tr><tr><td>Data_Science</td><td>Business Intelligence, Financial Analysis, Text Analytics</td><td>KPI_Dashboard</td></tr><tr><td>Content_Media</td><td>Writing, Social Media, Design, Video and Audio</td><td>Academic_Writing</td></tr><tr><td>Safety_Compliance</td><td>Prompt Injection, Data Protection, Harmful Content, User</td><td>Content_Injection</td></tr><tr><td>Automation_Platform</td><td>Wellbeing Agent, Skill, Scheduling</td><td>Skill_Installation</td></tr><tr><td>Travel_Lifestyle</td><td>Travel, Daily Life</td><td>Itinerary_Planning</td></tr><tr><td>Education_Knowledge</td><td>Learning, Knowledge QA, Exam Preparation</td><td>Concept_Explanation</td></tr><tr><td>Legal</td><td>Contract, Litigation</td><td>Clause_Extraction</td></tr><tr><td>Healthcare</td><td>Medical Policy, Pet Care</td><td>Policy_QA</td></tr><tr><td>Manufacturing_Engineering</td><td>Quality Control, Aerospace, Smart Agriculture</td><td>Defect_Detection</td></tr><tr><td>Multimodal_Generation</td><td>Webpage, score, map, video, and document transformations</td><td>Video_To_Webpage</td></tr></table>

## B DATA SOURCES

Our environment file pool draws on diverse public and versioned sources; representative examples are listed below.

Table 9: Representative sources used to construct the environment file pool.
<table><tr><td>Source type</td><td>Representative sources</td></tr><tr><td>Hugging Face datasets</td><td>superdoc-dev/docx-corpus; WikiArt; EuroSAT; MathVista; Food-101; common-pile/arxiv_papers; Zenodo10K; Enron Corpus; LibriSpeech; ESC-50; UrbanSound8K; and MINDS-14</td></tr><tr><td>Regulatory and disclosure portals</td><td>SEC EDGAR, CNINFO, and CSE filings</td></tr><tr><td>Government open data</td><td>dat a . gov . hk; national portals for the United Kingdom, Australia, Ireland, and Italy; and the Toronto CKAN catalog</td></tr><tr><td>Fixed GitHub revisions</td><td>Zeek, Lucide, Simple Icons, MutopiaProject, music21, and OpenSheetMusicDisplay</td></tr><tr><td>Vertical-domain websites</td><td>ypppt . com and domain-specific public-sector sites such as b jf sh . gov. cn</td></tr></table>

## C PROMPTS FOR ENVIRONMENT CONSTRUCTION AND QUALITY CONTROL

We provide abridged versions of the prompts used by the three major agents in our environmentconstruction pipeline. The prompts preserve the main role definitions, directory structure, runtime interfaces, workspace construction rules, user-simulator design, verifier construction, and qualitycontrol procedures, while omitting repeated engineering instructions and deployment-specific special cases.

## C.1 BUILDER

The Builder instantiates an executable sample from the task specification. It constructs the visible workspace, configures the runtime tools and skills, prepares the private user-simulator context, and

designs multi-turn or collaborative workspace updates. It also executes the task once to obtain a grounded reference answer, but does not construct the verifier.

Builder Prompt   
You are a sample Builder. Construct an executable agentic sample in the current task root.   
Construct the visible workspace and task inputs; configure runtime tools, skills, metadata,   
and user messages; prepare private user-simulator and collaborative-update state; execute   
the task to derive a grounded reference answer; then restore the intended initial state. Do   
not implement verifier logic or modify tests/.   
Use only configured runtime interfaces. The runtime allowlist includes   
execute\_shell\_command; file read, write, edit, listing, glob, and grep operations;   
browser\_use and download\_url; image/video inspection; time, token, collaboration, memory,   
and Tavily retrieval tools. Builder-only generate\_image, generate\_video, generate\_music,   
and generate\_speech tools must not appear in the execution agent's requirements.   
Construct real, usable visible inputs under:   
environment/data/local\_files/   
workspace/   
downloads/   
notes/   
references/   
Expose only runtime paths such as workspace/..., downloads/..., notes/..., and   
references/...; never expose construction paths. Maintain state\_manifest.json for   
organization and provenance without answers, expected classifications, exact values,   
verifier details, or hidden evidence. Do not pre-create the deliverable. Multimodal and GUI   
tasks must provide accessible assets or operable interfaces and require meaningful   
observation or interaction rather than direct source-file shortcuts.   
For multi-turn or collaborative tasks, construct private context under:   
.user/   
persona.md   
profile.json   
domain\_knowledge.md   
world\_model.md   
private\_rules.md   
memory/..., history/..., state/...   
Follow the supplied schemas. Private context must model a coherent user's knowledge,   
terminology, tools, constraints, and plausible blind spots without encoding the task or   
solution. Later turns must introduce new information, changed inputs or requirements, or   
decisions based on prior progress rather than splitting deterministic cleanup steps.   
For collaborative tasks, create .patch/instructions.md and turn\_NN\_hint.md files using   
supported create, overwrite, append, delete, or edit actions. Patches must use   
runtime-visible paths, satisfy action preconditions, and materially change subsequent work.   
Update task.yaml with the required metadata, labels, tools, skills, environment paths,   
setup scripts, messages, and evaluation fields while preserving pipeline identifiers. When   
applicable, configure successful/not\_successful, edit, undo, sentiment, behavior, or   
artifact-download signals only when they affect active interaction state.   
Execute the task on the real inputs and derive values, records, summaries, artifacts, and   
state changes rather than guessing. Store a non-placeholder reference answer and, when   
applicable, answer\_facts for commands, files, values, and GUI state. For multi-turn tasks,   
organize the answer as Round 1, Round 2, ....   
Difficulty should arise from meaningful dependencies, cross-format reasoning, tool use,   
revisions, or conflicting evidence. Before finishing, inspect all visible files; remove   
logs, caches, debug artifacts, temporary screenshots, and reference-run outputs; restore   
the initial workspace; and confirm that no hidden answer or verifier information is   
exposed.

## C.2 VERIFIER CONSTRUCTOR

The Verifier Constructor receives the constructed task and independently builds the evaluation logic. It derives expected values from the task inputs, combines deterministic checks with trajectory-level rubric criteria, and self-validates the verifier on both correct and controlled failure states.

Verifier Constructor Prompt   
You are the Verifier Constructor for an agentic task.   
Your responsibility is verification only. Do not redesign the task, workspace, user   
context, or inputs; report inconsistencies to the Builder.   
Construct:   
tests/   
checklist.jsonl   
verification.py   
verify.py   
test.sh   
test\_outputs.py   
Read task.yaml, the real files under environment/data/local\_files/,   
evaluation.inputs.answer, answer\_facts, and relevant multi-turn or workspace-update state.   
Treat reference claims as hypotheses: first reproduce the key expected results from the   
inputs.

Verifier Constructor Prompt continued   
For each user requirement, build:   
requirement   
-> expected outcome   
-> observable evidence   
-> checker or rubric item   
-> weight   
Prefer deterministic, format-aware checks for file contents, JSON/CSV/XLSX fields, numeric   
results, schemas, sets, generated artifacts, cross-file consistency, and observable state   
changes. Use numeric tolerances; normalize dates, currencies, and percentages; and use   
case-insensitive or regex matching where wording is flexible. Avoid checks tied to   
incidental phrasing, CSS classes, pixel positions, or one implementation when equivalent   
outputs are valid.   
Cover output validity, structural correctness, semantic correctness, and red-line   
constraints, with semantic and safety requirements weighted above file existence. Inspect   
trajectory evidence only when completion requires a particular process, such as viewing   
required media, interacting through browser\_use, or re-reading an updated source before   
recomputation.   
Create tests/checklist.jsonl with entries:   
{   
"id": "...",   
"item": "...",   
"weight": ...,   
"status": "pending",   
"evidence": ""   
}   
Checklist items must be task-relevant trajectory assertions using real runtime tool names   
and observable evidence, not static package checks. Typical categories cover required   
tools, observed inputs, required procedure, final-answer facts, and shortcut avoidance.   
Emit structured results containing passed, score, max\_score, score\_points, summary, and   
errors. Missing outputs, malformed artifacts, or absent trajectory evidence must yield a   
normal low score rather than an exception.   
Reject untouched states, empty or copied outputs, unsupported claims, pre-existing   
artifacts presented as completed work, and task-specific shortcuts. For in-place editing,   
verify that the requested change occurred.   
Self-validate using a correct completion, the untouched state, missing or malformed   
outputs, wrong key values, cross-file inconsistencies, missing trajectory evidence, and   
relevant shortcuts. Do not weaken core checks to make the reference pass; locate defects in   
the task, data, reference, or verifier and report them to the responsible constructor.

## C.3 ISSUE DETECTOR

The Issue Detector performs the final package-level quality review. It actively inspects the environment, reproduces the task, checks multi-turn and workspace-update behavior, and audits whether the verifier distinguishes valid completions from incomplete or shortcut states. It does not repair the package directly; instead, it returns evidence-based defects to the responsible constructor.

Issue Detector Prompt   
You are an active quality Issue Detector.   
Evaluate the sample before release. Use tools to inspect real files, reproduce the task,   
and test the verifier; do not judge from prompt text alone and do not repair the package   
directly.   
1. Inspect the package   
Read the complete task.yaml; environment/data/local\_files/; optional .user/, .patch/, and   
solution/ artifacts; tests/checklist.jsonl; and the verifier entry points and   
implementation. Inspect actual contents rather than filenames.   
Evaluate the execution agent under its real visibility boundary. Hidden user state, future   
patches, solutions, tests, and construction metadata are shortcuts only if leaked into   
runtime-visible state. Check visible inputs for real usable resources, placeholders, answer   
or verifier leakage, private-state exposure, and construction residue.   
2. Inspect user context and patches   
For multi-turn tasks, verify that persona, profile, knowledge, memory, history, and state   
describe a coherent user; schemas are valid; domain knowledge is task-relevant rather than a   
solution; and latent goals do not expose the task.   
For collaborative tasks, validate patch YAML, allowed actions, runtime-visible paths,   
create/edit/overwrite preconditions, and cross-turn consistency. Workspace changes must   
materially affect later work; empty patches or explanatory notes alone are insufficient.   
3. Reproduce the task   
Actually perform the task using the supplied inputs and allowed tools. Parse structured   
data, inspect documents and required media, and interact with GUI state when applicable.   
Record tools used, files observed, important values, intermediate calculations, generated   
artifacts, and final conclusions.   
Compare the reproduction with evaluation.inputs.answer and answer\_facts, and report exact   
mismatches. For each later turn, verify that it depends on new information, changed inputs   
or requirements, user preferences, workspace updates, or actual prior results; reject turns   
that merely split deterministic work or request confirmation.

Issue Detector Prompt continued   
4. Review the checklist   
Each applicable item must use real runtime tool names, refer to observable evidence, match   
the task, be satisfiable by a valid execution, and avoid imposing one arbitrary   
implementation. Identify missing anti-shortcut coverage, but do not modify the checklist.   
5. Audit the verifier   
Inspect the mapping:   
requirement   
-> success condition   
-> evidence   
-> checker   
-> score weight   
Audit layered scoring, deterministic priority, task specificity, rigor, and anti-cheating.   
Semantic correctness and red-line constraints must outweigh superficial file-existence   
checks.   
Run the verifier, when possible, on the untouched state, a correct reference completion,   
and controlled failures such as missing or malformed outputs, wrong key values, cross-file   
inconsistencies, absent trajectory evidence, and task-specific shortcuts. It must reject   
incomplete, incorrect, untouched, and shortcut states without crashing; passing the initial   
state is a serious defect.   
6. Score and route   
Score realism, separation, solvability, interaction\_potential, consistency, and   
patch\_quality when applicable, then compute the applicable average. Critical executability,   
leakage, and verifier failures override a high average.   
Use repair targets task\_spec, environment, solution, verifier, trajectory, or packaging.   
Task, environment, user-context, and reference-answer defects normally return to the   
Builder; verifier defects return to the Verifier Constructor. Do not recommend weakening   
core checks merely to force a pass.   
7. Final output   
Return exactly one JSON object:   
{"realism": 0, "separation": 0, "solvability": 0,   
"interaction\_potential": 0, "consistency": 0, "patch\_quality": 0,   
"average": 0.0, "passed": false,   
"fail\_reasons": ["..."], "repair\_suggestions": ["..."],   
"repair\_targets": ["..."]}   
Failure reasons must cite concrete evidence. Set passed=true only when the quality   
threshold and all hard executability and verification checks pass.  
These prompts summarize method-level instructions. Runtime-specific interface, sandbox, and execution fragments are omitted for space without altering the roles or pipeline.