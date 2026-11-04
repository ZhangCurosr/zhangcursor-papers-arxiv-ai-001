# LLMs as MDP Designers: A Dependency-Aware Agentic Framework for Automated Robot Policy Generation

Hong Je-Gal   
Department of AI & Robotics   
Sejong University   
Seoul 05006   
jagrhong@sju.ac.kr   
Jinwoo Kim   
Department of AI & Robotics   
Sejong University   
Seoul 05006   
JinwooKim@sju.ac.kr   
Chan-Bin Yi   
Department of AI   
Sejong University   
Seoul 05006   
22012165@sju.ac.kr   
Hyun-Suk Lee<sup>∗</sup>   
Department of AI & Robotics   
AI Robotics Institute (AIRI)   
Sejong University   
Seoul 05006   
hyunsuk@sejong.ac.kr

## Abstract

We investigate the automated generation of executable robot reinforcement learning (RL) policies from natural-language task descriptions. Rather than using large language models (LLMs) as execution-time decision makers, we use them as designers that construct RL pipelines before deployment, including Markov decision process (MDP) formulation and simulator implementation. We identify a phasewise structure in this problem: MDP formulation is dependency-structured and sequential, whereas simulator implementation becomes modular once the formulation is fixed. Based on this structure, we propose ARPG, a dependency-aware agentic framework that performs sequential MDP modeling, localized verification and correction, and simulator-structured implementation while externalizing intermediate artifacts. ARPG targets a key failure mode of automated policy generation: executable code may still yield non-learnable policies when state, action, reward, and termination artifacts are not semantically aligned. Experiments in Isaac Sim show that ARPG improves modeling correctness, environment executability, and policy-generation success compared with monolithic LLM baselines and off-theshelf agentic systems. Additional demonstrations on custom robotics, standard RL, and engineering-domain MDPs further suggest that the same modeling pipeline can be reused beyond the main benchmark suite.

## 1 Introduction

The recent emergence of physical artificial intelligence (AI) has led to an increase in the complexity and diversity of robotic tasks [1, 2]. In modern industrial and service environments, robots must operate under changing task requirements, unstructured surroundings, and strict safety constraints [3–5]. Consequently, designing robot control policies has become more costly and difficult, particularly for tasks that require long-horizon reasoning, contact-rich interactions, and adaptation to uncertainty [6, 7].

Reinforcement learning (RL) is one of the most promising solutions for acquiring robot policies that enable agents to optimize long-horizon objectives, such as stability, efficiency, and safety [8]. However, despite significant progress in RL algorithms, generating robot RL policies remains challenging. A primary bottleneck lies in formulation of the learning problem, rather than in policy optimization itself [9]. Formulating an RL problem for robotics requires careful specification of observations, actions, reward functions, and termination logic. Even minor formulation errors can render learning infeasible, leading to unstable training or degenerate behaviors [10]. As a result, robot RL formulation is still largely performed manually by domain experts [11]. Whenever a task, robot, or environment changes, the entire design procedure must be repeated by human experts, resulting in long development cycles. This reliance on manual formulation fundamentally limits the scalability and reproducibility of robot RL.

Recent advances in large language models (LLMs) have led to rapid progress across a wide range of domains, including mathematical reasoning, long-horizon planning, code synthesis, and problem solving in both scientific and human-centered tasks [12, 13]. These developments have motivated growing interest in integrating LLMs into complex decision-making systems beyond pure language tasks [14, 15]. However, these approaches primarily emphasize flexible adaptation in open-ended and dynamically changing environments rather than the specialized, long-horizon, and safety-critical behaviors required in robot RL, especially in industrial domains. As a result, directly employing LLMs as acting agents introduces substantial uncertainty and execution risk, making such approaches difficult to deploy in practice. Instead, the advanced reasoning capabilities of LLMs are more effectively utilized in the RL formulation stage, where design decisions can be externalized as explicit artifacts. RL formulation can be structured as a sequence of dependency-aware stages, such as observation specification, reward design, and termination logic. These stages produce artifacts that admit direct verification and localized correction, prior to policy training and deployment.

Motivated by this perspective, we present a dependency-aware agentic framework for automated, end-to-end generation of executable robot RL policies from free-form natural language. Drawing on an analysis of how human experts formulate and debug robot RL problems in practice, the roles of individual agents in the framework are guided by an explicit design rationale distilled from this human-driven process. The framework treats LLM agents as designers and engineers who construct, ground, and iteratively refine complete RL pipelines, including sequential Markov decision process (MDP) formulation, domain-specific implementation, and debugging. Rather than targeting code executability alone, ARPG explicitly aligns state, action, reward, and termination artifacts before implementation, since executable environments may still yield non-learnable policies when these artifacts are inconsistent. This exposes intermediate artifacts for expert inspection, targeted modification, and localized re-execution without requiring full regeneration. Consequently, the final output is a standalone RL policy that operates independently of language models at deployment time.

The contributions of this work are summarized as follows:

• We propose a dependency-aware agentic framework that enables automated, end-to-end generation of robot RL policies without manual formulation.

• We conceptualize the human-driven robot RL formulation as a structured, dependency-aware decision process, and distill explicit rationale to guide framework design with stage-wise verification and correction. This analysis clarifies the conditions under which localized correction improves the success of modeling and policy generation.

• We design the framework to be explicitly expert-inspectable, i.e., exposing all intermediate formulation artifacts and implementation records in human-readable form, ensuring interpretability and enabling practical usability.

• We validate the proposed framework through extensive experiments on manipulation and locomotion tasks in Isaac Sim, demonstrating substantial improvements in end-to-end policy generation success compared to unstructured baselines.

## 2 Related Work

Recent advances in robotics have increasingly leveraged AI to enable a wide range of robot behaviors, including perception-driven control [16, 17], language-conditioned action generation [15], and generalist RL policies [18]. One major line of research focuses on vision-language-action models [19, 20] and large-scale generalist policies that directly map sensory observations and language instructions to control actions. These approaches demonstrate strong generalization across diverse tasks, particularly in open-ended and dynamically changing environments. However, in most existing systems, AI models are primarily used as black-box executors that make decisions at execution time. These models implicitly encode task objectives, constraints, and control logic as internal representations.

In contrast, the formulation of robot RL problems, including the specification of states, actions, objectives, constraints, and termination conditions, has traditionally relied heavily on human experts [8, 9]. Even in learning-based robotic systems, designing MDP structures and training pipelines typically requires manual engineering and domain knowledge [21, 22]. Recent studies in RL have explored the use of learning-based or language-based tools to assist with specific components, such as reward function design, observation selection, or policy initialization [23–25]. However, these efforts generally provide partial guidance rather than fully automated formulation. Consequently, although execution and control have increasingly benefited from large models, formulation remains predominantly human-driven.

Outside of robotics, recent progress in LLMs has enabled a growing class of agentic frameworks that treat language models as structured problem solvers rather than execution-time agents [14, 26, 27]. Across various domains, such as mathematical reasoning and optimization, these frameworks decompose complex tasks into explicit intermediate steps, iteratively refine partial solutions, and correct errors through structured feedback [12, 27–31]. These developments suggest that LLMs can serve as designers or engineers that construct and refine problem formulations, rather than merely producing final outputs.

Several recent studies have explored automated formulation from natural language descriptions, providing important foundations for this line of research. Autoformalism [32] and Optimus [33] both aim to translate free-form specifications into formal optimization models using LLMs. However, these approaches are inherently limited to deterministic problems, such as linear programming and mixed-integer nonlinear programming, rather than stochastic problems. Consequently, they cannot address the stronger dependencies induced by temporal dynamics in MDP-based formulations.

## 3 Phase-wise Analysis of RL Policy Generation Problem

Robot RL policy generation has traditionally relied on human experts who translate task intent and robot knowledge into a learnable control problem. We view this process as decomposable into two coupled phases: a modeling phase, which interprets the task and specifies the corresponding MDP, and an implementation phase, which realizes the MDP as an executable simulator environment and training pipeline. Figure 1 illustrates this conceptual decomposition and the interaction between the two phases, as well as the role of human expertise in bridging them. Although human experts often iterate across these phases through simulation feedback and debugging, the two phases have different dependency structures.

![](images/266789bcf0023f552d82666deea9c62b7789ac7a722106d7480c2824d90fc5d0.jpg)  
Figure 1: Human-driven RL policy generation process. The arrows illustrate iterative dependencies and back-and-forth adjustments across stages.

The modeling phase transforms a real-world task into a formal RL problem. The key property of this phase is sequential dependency. To define a valid objective or reward, the expert must first identify task parameters, decision variables, state variables, and feasibility constraints. Similarly, constraints cannot be specified independently of the variables and robot capabilities they restrict. Thus, formulation artifacts are not independent modules; each artifact constrains the feasible choices of later artifacts. We write this dependency-structured formulation process as

$$
x \ \to \ y _ { 1 } \ \to \ y _ { 2 } \ \to \ \cdots \to \ y _ { L } \ \to \ z ,\tag{1}
$$

where x is the task description, $\{ y _ { i } \} _ { i = 1 } ^ { L }$ are intermediate formulation artifacts such as parameters, variables, objectives, and constraints, and z is the complete MDP formulation. This structure makes early formulation errors especially costly, because they can propagate to later artifacts and eventually yield an invalid or non-learnable MDP.

Unlike the modeling phase, the implementation phase has a modular structure rather than a semantic sequential dependency structure. Once z is fixed, this phase translates the $\mathbf { M D P }$ into an executable learning pipeline by instantiating it in a target simulator. In manager-based environments such as Isaac Lab, this instantiation is modular: scene, action, observation, event, termination, reward, and policy configurations are composed into an executable environment. Although some modules have explicit simulator-level dependencies, many can be generated independently after the MDP formulation is fixed. We therefore view implementation as a modular transformation

$$
z \ \to \ g ( z ) = \{ \ g _ { 1 } ( z ) , \ g _ { 2 } ( z ) , \ \dots , \ g _ { M } ( z ) \} ,\tag{2}
$$

where each $g _ { m } ( \cdot )$ produces a simulator-specific configuration module. In practice, these modules can be generated in simulator-dependent stages when explicit inter-module dependencies exist.

Based on this phase-wise structure, the objective of automated RL policy generation is to produce a formulation and implementation whose trained policy satisfies a task-specific success criterion:

$$
\begin{array} { r } { \mathbb { E } [ \mathbb { I } ( J ( g ( z ) ) \ge \tau ) ~ | ~ z = f ( x ) ] , } \end{array}\tag{3}
$$

where $J ( g ( z ) )$ is the performance of the trained policy produced from the implementation $g ( z )$ , τ is a task-specific requirement, f denotes the sequential formulation process in Equation (1), and g denotes the simulator implementation process in Equation (2). This objective captures the design goal of generating an MDP formulation and implementation that lead to a task-valid learned policy, while reflecting the different dependency structures of modeling and implementation.

## 4 ARPG: Agentic LLM-Based Framework for Automated Robotics Policy Generation

## 4.1 Framework Overview and Design Rationale

We propose ARPG, an agentic framework for automated robot policy generation using multi-agent LLMs. ARPG is designed to reflect the phase-wise structure identified in Section 3: dependencyaware sequential modeling followed by simulator-structured modular implementation. For MDP modeling, parallel generation can break dependency chains among parameters, variables, objectives, constraints, and rewards [34], while monolithic generation makes intermediate decisions opaque and difficult to correct [35]. A further challenge in iterative refinement is that fixing one component may inadvertently invalidate others. We refer to this phenomenon as error reintroduction. Unbounded error reintroduction leads to unstable refinement, motivating localized correction with explicit artifact boundaries.

![](images/e2b4b18409fbc8c12381d9ccce01eb954ff32554076ac75ee3c056005d5c3f73.jpg)  
Figure 2: Overview of the ARPG framework for structured formulation and policy generation.

To address the formulation-side challenges, we introduce the following design principles: 1) dependency-respecting sequential formulation, 2) explicit externalization of formulation artifacts, and 3) stage-wise verification with localized correction to limit error reintroduction. In addition, for implementation, we consider simulator-structured modular generation, exploiting parallelism where configuration modules have no unresolved dependencies. Together, these choices align the agentic structure with the sequential nature of MDP formulation and the modular structure of simulator implementation.

Figure 2 illustrates the overall structure of the proposed framework. Given a natural-language task description and a robot specification, the framework first constructs a formal MDP through a modeling phase. Then, it generates an executable Isaac Lab training pipeline through an implementation phase. All intermediate artifacts produced during this process, such as texts, JSON records, and LaTeX scripts, are explicitly materialized and stored in an ARPG logbook. This artifact log supports expert inspection, partial re-execution, and targeted modification without regenerating the entire pipeline, although such human intervention is not used in our automated evaluation.

## 4.2 Modeling Phase

The modeling phase requires two types of input. The first is a free-form task description provided by a human user, which specifies the desired behavior, success criteria, and operating conditions. The second is a Python or USD file describing a robot already integrated into the target simulator. This file encodes the robot’s kinematic structure, actuators, joints, and physical constraints. Together, these inputs define the task intent and the feasible control space of the agent.

These two inputs are processed by different specialized agents. The task description is processed by a parameter agent that extracts task-specific parameters, such as object properties and workspace settings. This agent provides the semantic basis for identifying decision variables, objectives, and constraints. The robot specification is processed by a dedicated robotics agent. This agent parses the code file and translates simulator-level representations into a concise natural-language summary. This summary reveals the controllable components, admissible motions, and relevant physical limitations of the robot. This allows subsequent modeling agents to reason about the robot’s capabilities without interacting directly with simulator-specific formats. It also enables the modeling phase in the framework to operate across different robot types, with only a change in the robot description.

After this input processing, the core formulation of the MDP proceeds through a fixed and strictly ordered sequence of specialized agents, in accordance with the rationale of dependency-respecting sequential formulation. Based on the extracted parameters and the robot capability description, first, an objective agent specifies the optimization objective. Then, a variable agent identifies the decision variables and state variables, which are required to characterize the objective. A constraint agent defines the feasibility conditions that restrict valid actions and state transitions based on the identified variables. The objective, variables, and constraint specifications are then formalized into mathematical expressions by a modeling agent. Finally, a state-action-reward (SAR) agent integrates these elements to produce a complete MDP formulation z.

We illustrate this sequence with a simplified Humanoid locomotion example below. More detailed task descriptions and artifact traces for all tasks are provided in Appendix H and the supplementary material, respectively. Notably, all the intermediate artifacts during the modeling phase are provided in a textual form (e.g., objective, constraints), JSON records (e.g., variables, SAR structure), or LaTeX scripts (e.g., MDP formulation), making intermediate decisions inspectable and locally revisable.

## Running example: Humanoid locomotion

For “make the humanoid walk forward without falling,” ARPG constructs the MDP through an artifact chain:

• Parameters: control interval, 16s horizon, +X forward direction, base-height fall threshold, initialization ranges, and actuator limits.

• Robot summary: actuated joints, actuator semantics, feasible commands, base link, contact links, and physical limits.

• Objective: forward locomotion with upright posture, heading alignment, lateral stability, and fall avoidance.

• Variables: joint states, joint command targets, base pose and velocities, base height, heading error, uprightness, forward/lateral motion, and contact signals.

• Constraints: timing, action decimation, actuator bounds, orientation consistency, contact consistency, timeout termination, and fall termination.

• SAR: proprioceptive state, continuous joint-level action space, and reward combining forward progress, stability, lateral-drift, and fall-penalty terms.

The sequential execution of the modeling agents can be formalized according to the process in Equation (1) as

$$
z = f ( x ) = f _ { L } \circ f _ { L - 1 } \circ \cdot \cdot \cdot \circ f _ { 1 } ( x ) ,\tag{4}
$$

where $f _ { i }$ denotes the i-th stage-specific formulation transformation that produces the intermediate formulation artifact $y _ { i } = f _ { i } ( x , \{ y _ { j } \} _ { j = 1 } ^ { i - 1 } )$ This explicit externalization of formulation artifacts ensures that intermediate decisions are materialized, inspectable, and independently revisable. This separation explicitly distinguishes formulation artifacts from the transformations that generate them, eliminating ambiguity in the sequential modeling process.

To maximize the effectiveness of this externalization, the framework introduces an error correction (EC) agent to limit error reintroduction in the sequential process. Each agent in the modeling phase produces both a formulation artifact and an associated self-assessed confidence score in [0, 1]. When this score falls below a predefined threshold, the EC agent evaluates the artifact using agent-specific checklists and returns a localized diagnosis. For example, the parameter agent’s checklist examines task-critical values and bounds, consistency of units and scales, and potential contradictions. The responsible modeling agent is then re-invoked with its previous artifact, the diagnosis, and the same upstream inputs. This correction cycle is repeated within a configured retry limit while accepted upstream artifacts remain fixed, enabling localized revision of the artifacts without re-executing the entire modeling process. Further analysis of EC invocation patterns and correction outcomes is provided in Appendix D.2.

## 4.3 Implementation Phase

After the modeling phase produces an MDP formulation z, ARPG instantiates it as an executable training pipeline in the target simulator. The implementation structure depends on the simulator backend. For Isaac Lab, ARPG exploits its manager-based design, where scene, policy, action, observation, event, termination, and reward configurations are composed into a single environment. This modular structure allows ARPG to generate independent components in parallel while respecting explicit inter-module dependencies, such as shared scene definitions or observation specifications.

ARPG represents this process as simulator-structured staged generation:

$$
g ( z ; \mathcal { D } ) = G _ { N } \circ G _ { N - 1 } \circ \cdots \circ G _ { 1 } ( z ; \mathcal { D } ) , \qquad G _ { k } = \{ g _ { i } ^ { ( k ) } \} _ { i \in \mathcal { T } _ { k } } .\tag{5}
$$

Here, $G _ { k }$ denotes the set of configuration-module generators executed at stage k. Generators within the same $G _ { k }$ can run in parallel, while the composition across stages captures explicit simulator-level dependencies. For Isaac Lab, we use three stages $( N = 3 )$ : global scene and policy specifications; scene-dependent action, observation, event, and termination configurations; and reward configurations that depend on scene and observation specifications.

After module generation, a coding agent aggregates the resulting artifacts into executable environment and training code. If compilation or runtime errors occur, the error trace is analyzed to localize the failure to a specific configuration stage or module, and ARPG selectively re-invokes the responsible configuration agent for localized correction. When errors arise from cross-module interactions or inconsistent assumptions, the coding agent performs a coordinated revision of the assembled code. Upon successful execution, the pipeline yields both an executable simulation environment and a trained RL policy.

## 4.4 Common Artifact Management and Logbook

Beyond the phase-specific agents, the ARPG framework provides a shared artifact management mechanism that governs how intermediate results are externalized, modified, and reused across stages. All intermediate artifacts, including formulation artifacts, configuration specifications, generated code fragments, and policy validation traces, are systematically collected in the ARPG logbook. This representation explicitly records stage-wise dependencies and configuration boundaries. This enables both localized modification of individual components and global revision of the assembled code. Human users can directly inspect and modify specific artifacts when the framework is used as an expert-facing design tool, while the same artifact boundaries are also used by agents to support targeted re-invocation in the automated pipeline. This supports efficient iterative refinement with minimal recomputation, independent of the underlying simulator-specific stage structure.

## 4.5 Theoretical Justification

We provide a conditional explanatory model for the proposed ARPG framework, which consists of the modeling phase and the implementation phase. This model focuses on the tailored sequential

agentic structure of the modeling phase, which determines the correctness of the resulting formulation. In the framework, end-to-end success occurs if and only if both phases succeed

$$
\mathbb { P } ( S _ { \mathrm { A R P G } } ) = \mathbb { P } ( S _ { M } ) \mathbb { P } ( S _ { I } \mid S _ { M } ) ,\tag{6}
$$

where $S _ { M }$ and $S _ { I }$ denote the success events of the modeling and implementation phases, respectively, since the implementation phase operates conditional on a completed formulation.

We consider the modeling phase that produces a set of L formulation artifacts. Each artifact is either valid or invalid at termination. Let $A _ { i } ^ { ( t ) }$ denote the event that formulation artifact i is valid after the t-th verification-correction iteration. To reflect the sequential dependency among artifacts, we interpret $p _ { i } ^ { ( t ) }$ as the conditional residual error probability that artifact i remains invalid after iteration $t ,$ conditioned on the preceding artifacts being valid:

$$
p _ { i } ^ { ( t ) } : = \mathbb { P } \Big ( ( A _ { i } ^ { ( t ) } ) ^ { c } \mid A _ { 1 } ^ { ( t ) } , \ldots , A _ { i - 1 } ^ { ( t ) } \Big ) .
$$

The quantity $p _ { i } ^ { ( \infty ) }$ denotes the corresponding conditional residual error probability at termination. We define the modeling success event as $S _ { M } : =$ {all formulation artifacts are valid at termination}, with corresponding success probability

$$
\mathbb { P } ( S _ { M } ) = \prod _ { i = 1 } ^ { L } ( 1 - p _ { i } ^ { ( \infty ) } ) .\tag{7}
$$

The ARPG framework enforces explicit externalization of formulation artifacts, stage-wise verification, and localized correction. These structural constraints influence the effectiveness of error correction and the degree of unintended interference across formulation artifacts. This intuition can be formalized as the following comparison result.

Proposition 4.1. Consider a baseline formulation strategy and the ARPG framework, both of which admit artifact-wise formulation errors. Under the assumption that ARPG induces strictly better effective correction behavior (e.g., higher correctability or lower interference) for at least oneformulation artifact, while all other artifacts are unchanged, the modeling success probability satisfies

$$
\mathbb { P } ( S _ { M } ^ { \mathrm { A R P G } } ) > \mathbb { P } ( S _ { M } ^ { \mathrm { b a s e } } ) .
$$

This proposition does not assume that alternative strategies lack error correction capabilities. Single agent reasoning may allow for implicit self-correction, and parallel formulation strategies may use post hoc reconciliation. However, in practice, single-pass monolithic reasoning may exhibit weaker effective error correction due to the lack of explicit, verifiable intermediate artifacts. Meanwhile, the parallel modification of multiple formulation artifacts may increase cross-artifact interference. An LLM may miss an error, fail to repair it, or introduce a new inconsistency, so the assumed correction behavior need not hold in every invocation. Under the proposition’s assumptions, ARPG induces a higher modeling success probability by promoting effective correction and limiting unintended interactions. More detailed analyses are provided in Appendix B.

## 5 Experimental Results

## 5.1 Experimental Setup

For evaluation, we choose five main benchmark tasks that cover a variety of state dimensionality, control complexity, and interaction structure, ranging from low-dimensional control to high-degreeof-freedom locomotion and contact-rich manipulation. All experiments are implemented in Isaac Lab (v2.1.0) on top of NVIDIA Isaac Sim (v4.5.0).

• CartPole is a low-dimensional classic control task for evaluating basic MDP formulation with simple state and action spaces.

• Ant and Humanoid are high-dimensional locomotion tasks requiring coordinated long-horizon control across many degrees of freedom.

• Reach and Lift are manipulation tasks in which the robot must move its end-effector to a target pose and grasp and lift an object.

We evaluate ARPG to empirically assess whether dependency-aware sequential MDP formulation improves formulation correctness and translates into higher end-to-end policy generation success. To this end, we introduce three complementary metrics that reflect different stages of the policy generation pipeline. Their detailed criteria are provided in the Appendix F.

• Modeling success rate (MSR): The proportion of trials whose MDP components are logically correct and task-aligned for implementation, or whose trained policies satisfy the PGSR criteria.

• Environment implementation success rate (EISR): The proportion of trials in which the generated code executes without syntax or runtime errors in a standard Isaac Sim environment.

• Policy generation success rate (PGSR): The proportion of trials in which RL training converges and the trained policy satisfies the task-specific success criteria.

We use the following language model configurations for ARPG and the baselines. In ARPG, all agents use GPT-5.2. The monolithic baselines use GPT-5.2 and Gemini-3. We also evaluate off-theshelf agentic systems, Claude Code with Claude Sonnet 4.6 and Codex with GPT-5.2. All systems receive the same task description, robot/USD information, and per-configuration documentation. This documentation is embedded directly in the prompt for monolithic baselines and provided as a text file to the off-the-shelf agentic systems. For all configurations that allow code revision using execution feedback, we fix the maximum number of revision rounds at five. Unless otherwise noted, results are evaluated over 10 independent runs.

Table 1 summarizes the architectural ingredients present in each evaluated system. Monolithic LLM baselines may reason over the full task in a single context, but they do not externalize MDP formulation as a dependencyaware artifact chain. Off-the-shelf agentic systems provide stronger implementation decomposition and execution-based debugging, but they do not explicitly structure MDP formulation as a dependency-ordered artifact chain. Consequently, such systems may improve environment executability without ensuring that MDP-related

Table 1: Comparison of structural components.
<table><tr><td>Component</td><td>Monolithic Off-the-shelf ARPG LLM</td><td>agents</td><td>(Ours)</td></tr><tr><td>Component-wise implementation</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Dependency-aware MDP artifactization</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Artifact-localized correction</td><td>x</td><td>x</td><td>√</td></tr></table>

artifacts are semantically aligned for policy learning. ARPG differs by coupling component-wise implementation with dependency-aware MDP artifactization and artifact-localized correction. This structural distinction motivates our stage-wise evaluation using MSR, EISR, and PGSR.

## 5.2 Overall Performance

We first evaluate ARPG against monolithic LLM baselines and off-the-shelf agentic systems across the benchmark tasks, instantiating three practical design regimes: monolithic generation, generalpurpose agentic code generation, and dependency-aware MDP-centered generation. The monolithic baselines test whether a single model can generate the full pipeline, with one-shot generation limited to a single pass and iterative generation allowing execution-based debugging. The agentic baselines test whether component-wise code generation and execution-based debugging are sufficient without dependency-aware MDP artifactization. We report MSR, EISR, and PGSR to decompose the pipeline into formulation correctness, implementation executability, and final policy success. This decomposition captures the central failure-mode separation targeted by ARPG: executable environments do not necessarily encode learnable, task-aligned MDPs, while valid MDPs can still fail due to simulator-specific implementation errors.

As shown in Table 2, ARPG achieves consistently high MSR across all benchmark tasks, indicating reliable construction of valid and task-aligned MDP formulations. It also obtains the highest PGSR across all tasks, showing that the generated formulations can be grounded into executable environments and trained into task-satisfying policies. Rollout videos are provided in the supplementary material, and Appendix E further analyzes where implementation corrections arise across tasks.

Gemini-3 and GPT-5.2, the two one-shot monolithic LLM baselines, exhibit a similar failure pattern. Although they sometimes satisfy parts of the modeling criteria, they fail to generate executable Isaac Lab environments even for the simplest CartPole task, resulting in zero EISR and therefore zero PGSR across all tasks. Iterative GPT-5.2 extends the one-shot baseline by allowing the same model to revise its generated code using execution error logs. It achieves full success across all three metrics on CartPole and improves EISR across all tasks. On the other four tasks, however, reported MSR remains unchanged from one-shot GPT-5.2, while PGSR remains low. Thus, the one-shot monolithic baselines face bottlenecks in both MDP modeling and simulator-specific implementation. Iterative GPT-5.2 partially alleviates the implementation bottleneck through execution feedback, while task-aligned MDP formulation remains a challenge on the more complex tasks.

Table 2: Success rates across benchmark tasks. All results are reported as triplets in the order of MSR, EISR, and PGSR. Bold indicates the best value for each metric within a task. Corresponding 95% Wilson confidence intervals are provided in Appendix F.3.
<table><tr><td rowspan="2">Task</td><td colspan="4">GPT-5.2</td><td>Gemini-3</td><td>Sonnet 4.6</td></tr><tr><td>ARPG</td><td>One-shot</td><td>Iterative</td><td>Codex</td><td>One-shot</td><td>Claude</td></tr><tr><td>CartPole</td><td></td><td>1.0 / 1.0 / 1.00.4 / 0.0 / 0.0 1.0 / 1.0 / 1.0 1.0 / 0.6 / 0.6 0.6 / 0.0 / 0.01.0 / 1.0 / 1.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Ant</td><td></td><td>1.0 / 0.9 / 0.90.2 / 0.0 / 0.00.2 / 0.3 / 0.1 0.4 / 0.8 / 0.40.1 / 0.0 / 0.0 0.0 / 0.9 / 0.0</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Humanoid0.9 / 0.7 / 0.40.1 / 0.0 / 0.0 0.1 / 0.3 / 0.00.1 / 0.5 / 0.1 0.3 / 0.0 / 0.0 0.0 / 0.9 / 0.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Reach</td><td></td><td>0.9 / 0.9 / 0.90.0 / 0.0 / 0.0 0.0 / 0.6 / 0.0 0.4 / 0.3 / 0.2 0.0 / 0.0 / 0.00.5 / 1.0 / 0.3</td><td></td><td></td><td></td><td></td></tr><tr><td>Lift</td><td></td><td>1.0 / 0.9 / 0.40.0 / 0.0 / 0.0 0.0 / 0.9 / 0.0 0.0 / 0.6 / 0.0 0.0 / 0.0 / 0.0 0.0 / 0.9 / 0.0</td><td></td><td></td><td></td><td></td></tr></table>

A similar gap between environment executability and policy generation success is observed for Claude and Codex, the two off-the-shelf agentic systems. Despite this limitation, both systems obtain higher average PGSR than iterative GPT-5.2. Their native planning and context management, together with component-wise implementation, may improve cross-module consistency and task alignment. For example, on Reach, Codex achieves higher PGSR despite lower EISR than iterative GPT-5.2. However, the generated MDPs still frequently fail the validity and task-alignment criteria captured by MSR on the more complex tasks. Appendix C presents representative failures involving misaligned reward and termination logic, over-specified reward functions, and poorly scaled auxiliary terms that suppress the main task signal during training. ARPG mitigates this failure mode by externalizing dependency-aware MDP artifacts, verifying them before implementation, and applying artifact-localized correction when inconsistencies are detected.

## 5.3 Ablation Study

To assess the contributions of modeling-phase EC and sequential formulation to ARPG’s performance gains over the baselines, we evaluate two ablated variants. ARPG w/o EC disables modeling-phase EC, whereas ARPG w/o sequential generates artifacts in parallel with EC retained. Both variants retain the same implementation phase, including debugging through execution feedback. All other experimental settings remain the same as in Section 5.2.

Table 3: Ablation results.
<table><tr><td>Task</td><td>ARPG</td><td>w/o EC</td><td>w/o sequential</td></tr><tr><td>CartPole</td><td></td><td>1.0 / 1.0 / 1.0 1.0 / 1.0 / 1.0 0.9 / 0.9 / 0.9</td><td></td></tr><tr><td>Ant</td><td>1.0 / 0.9 / 0.9 0.9 / 0.9 / 0.6 0.3 / 0.9 / 0.1</td><td></td><td></td></tr><tr><td></td><td>Humanoid 0.9 / 0.7 / 0.4 0.3 / 1.0 / 0.3 0.1 / 1.0 / 0.1</td><td></td><td></td></tr><tr><td>Reach</td><td></td><td></td><td>0.9 / 0.9 / 0.9 0.5 / 0.8 / 0.1 0.0 / 1.0 / 0.0</td></tr><tr><td>Lift</td><td></td><td></td><td>1.0 / 0.9 / 0.40.2 / 1.0 / 0.20.0 / 0.9 / 0.0</td></tr></table>

In Table 3, both ablated variants retain high success rates on the simpler CartPole task, indicating that removing either component has little effect in this task. However, both variants yield lower MSR and PGSR than ARPG on the more complex tasks while maintaining high EISR. In some cases, EISR even exceeds that of ARPG despite the declines in formulation correctness and policy success. This shared pattern shows that high environment executability does not necessarily translate into a task-aligned MDP formulation or a successful learned policy.

The declines in MSR and PGSR are larger for ARPG w/o sequential than for ARPG w/o EC across these tasks. In particular, Reach and Lift obtain zero MSR and PGSR when artifacts are generated in parallel, despite retaining modeling-phase EC. This pattern is consistent with the correctioninterference account in Section 4.5, suggesting that the effectiveness of localized correction depends on how dependencies among formulation artifacts are organized. Together, these results support the contributions of both components to policy generation, with the larger decline under parallel formulation underscoring the central role of dependency-ordered sequential formulation.

## 5.4 Additional Generality Demonstrations Beyond Benchmark Tasks

Beyond the five main Isaac Lab tasks, we further examine whether ARPG can be applied to tasks with different objectives, embodiments, and implementation backends. These additional studies include custom Isaac Sim robotics tasks, standard Gymnasium examples, grid-based decision problems, and domain-specific engineering environments as summarized in Table 4. These cases are not treated as success-rate benchmarks; instead, they test whether the same dependency-aware modeling pipeline can be reused across various task families while only adapting the environmentspecific implementation interface. Across these demonstrations, ARPG automatically generates coherent MDP artifacts and executable policies. Details of these demonstrations, including objectives and supporting evidence, are provided in Appendix G and the supplementary material.

Table 4: Examples of demonstrations.
<table><tr><td>Setting</td><td>Examples</td><td>Backend</td></tr><tr><td>Robotics variants</td><td>Jumping Ant, Jumping Humanoid, Isaac Sim Pick-and-place</td><td></td></tr><tr><td>Classic control</td><td>MountainCar, Drone delivery</td><td>Gymnasium, Grid env.</td></tr><tr><td>Engineering MDPs</td><td>Wireless, Energy</td><td>Custom Env.</td></tr></table>

## 5.5 Computational Cost and System Overhead

We next compare the computational overhead of ARPG with off-the-shelf agentic systems. This comparison is important because ARPG invokes multiple specialized agents and may perform localized re-invocation during verification or debugging, which could increase cost despite improving reliability. We therefore report three quantities: direct token usage, cached token usage, and wallclock runtime. Direct tokens measure the explicit input/output tokens consumed during generation, cached tokens capture repeated context reuse in agentic systems, and runtime measures the end-to-end time required to produce the final pipeline. Table 5 summarizes these costs across benchmark tasks.

Table 5: Computational cost comparison. Token usage is reported as total tokens, with direct and cached tokens shown in parentheses. Bold indicates the lowest value within each task.
<table><tr><td rowspan="2">Task</td><td colspan="3">Tokens (M)</td><td colspan="3">Runtime (min)</td></tr><tr><td>ARPG</td><td>Claude</td><td>Codex</td><td>ARPG</td><td>Claude</td><td>Codex</td></tr><tr><td>CartPole</td><td>0.160 (0.140, 0.020)</td><td>2.586 (0.036, 2.550)</td><td>2.665 (0.199, 2.466)</td><td>2.9</td><td>8.2</td><td>7.4</td></tr><tr><td>Ant</td><td>0.149 (0.129, 0.020)</td><td>5.711 (0.061, 5.650)</td><td>2.671 (0.177, 2.494)</td><td>3.9</td><td>17.2</td><td>7.7</td></tr><tr><td>Humanoid</td><td>0.248 (0.208, 0.040)</td><td>7.882 (0.092, 7.790)</td><td>4.201 (0.244, 3.957)</td><td>4.5</td><td>24.1</td><td>11.8</td></tr><tr><td>Reach</td><td>0.195 (0.165, 0.030)</td><td>1.818 (0.178, 1.640)</td><td>2.433 (0.173, 2.260)</td><td>4.6</td><td>11.8</td><td>9.3</td></tr><tr><td>Lift</td><td>0.232 (0.192, 0.040)</td><td>1.769 (0.119, 1.650)</td><td>2.315 (0.165, 2.150)</td><td>6.2</td><td>15.0</td><td>8.2</td></tr></table>

Although ARPG invokes multiple specialized agents, it incurs lower measured computational overhead than the evaluated off-the-shelf systems. Each ARPG agent receives a bounded role-specific context and operates on compact intermediate artifacts, whereas off-the-shelf systems may repeatedly reintroduce full documents and accumulated interaction histories during code generation and debugging. In our evaluation, ARPG uses fewer reported total tokens and has a shorter wall-clock runtime across all tasks. The gap is largest on Humanoid: ARPG takes 4.5 minutes with 0.248 M tokens, while Claude and Codex require 24.1 and 11.8 minutes, respectively, with 7.882 and 4.201 M tokens. These results show a practical efficiency advantage for ARPG under the evaluated configurations, with each system retaining its native scaffolding and token accounting.

## 6 Conclusion

In this paper, we studied automated generation of executable robot RL policies from natural-language task descriptions. We proposed ARPG, an agentic framework that treats LLMs as MDP designers and enforces staged formulation, artifact externalization, verification, and localized correction. Experiments on Isaac Lab control, locomotion, and manipulation tasks show that ARPG improves modeling correctness and policy-generation success over monolithic LLM baselines and off-theshelf agentic systems. The results also indicate that executable implementation alone is insufficient: reliable policy generation requires dependency-aligned state, action, reward, and termination artifacts. Additional demonstrations on custom robotics, classic controls, and engineering-domain MDPs suggest that ARPG can be instantiated beyond the main benchmark suite.

## Acknowledgments and Disclosure of Funding

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-24523498 & RS-2026-25482257), in part by the IITP (Institute of Information & Communications Technology Planning & Evaluation)–ITRC (Information Technology Research Center) grant funded by the Korea government (Ministry of Science and ICT) (IITP-2026-RS-2021-II211816), and in part by the Technology Innovation Program (RS-2022- 00154678, Development of Intelligent Sensor Platform Technology for Connected Sensor) funded by the Ministry of Trade, Industry & Energy (MOTIE, Korea). We thank all reviewers for their valuable comments and suggestions. The authors declare no financial conflicts of interest related to this work.

## References

[1] Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, and Philip S Yu. Large language models for robotics: A survey. arXiv preprint arXiv:2311.07226, 2023.

[2] Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, and Yuke Zhu. Vision-languageaction models for robotics: A review towards real-world applications. IEEE Access, 2025.

[3] Jiaqi Wang, Enze Shi, Huawen Hu, Chong Ma, Yiheng Liu, Xuhui Wang, Yincheng Yao, Xuan Liu, Bao Ge, and Shu Zhang. Large language models for robotics: Opportunities, challenges, and perspectives. Journal ofAutomation and Intelligence, 4(1):52–64, 2025.

[4] Yutong Liu, Qingquan Sun, and Dhruvi Rajeshkumar Kapadia. Integrating large language models into robotic autonomy: A review of motion, voice, and training pipelines. AI, 6(7):158, 2025.

[5] Lukas Brunke, Melissa Greeff, Adam W Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, and Angela P Schoellig. Safe learning in robotics: From learning-based control to safe reinforcement learning. Annual Review ofControl, Robotics, and Autonomous Systems, 5(1):411–444, 2022.

[6] Sergey Levine, Peter Pastor, Alex Krizhevsky, Julian Ibarz, and Deirdre Quillen. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International Journal ofRobotics Research, 37(4-5):421–436, 2018.

[7] Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, and Sergey Levine. Learning agile robotic locomotion skills by imitating animals. arXiv preprint arXiv:2004.00784, 2020.

[8] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal ofRobotics Research, 32(11):1238–1274, 2013.

[9] Gabriel Dulac-Arnold, Nir Levine, Daniel J Mankowitz, Jerry Li, Cosmin Paduraru, Sven Gowal, and Todd Hester. Challenges of real-world reinforcement learning: definitions, benchmarks and analysis. Machine Learning, 110(9):2419–2468, 2021.

[10] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

[11] OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob Mc-Grew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. The International Journal ofRobotics Research, 39(1):3–20, 2020.

[12] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

[13] Mark Chen. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2022.

[15] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. PaLM-E: an embodied multimodal language model. In Proceedings of the International Conference on Machine Learning, pages 8469–8488, 2023.

[16] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[17] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open xembodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In Proceedings of the IEEE International Conference on Robotics and Automation, pages 6892–6903. IEEE, 2024.

[18] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.

[19] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of the Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[20] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[21] Kumater Ter, Ore-Ofe Ajayi, and Daniel Udekwe. Taxonomy and trends in reinforcement learning for robotics and control systems: A structured review. arXiv preprint arXiv:2510.21758, 2025.

[22] Timo Kaufmann, Paul Weng, Viktor Bengs, and Eyke Hüllermeier. A survey of reinforcement learning from human feedback. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

[23] Souradip Chakraborty, Anukriti Singh, Amisha Bhaskar, Pratap Tokekar, Dinesh Manocha, and Amrit Singh Bedi. REBEL: A regularization-based solution for reward overoptimization in robotic reinforcement learning from human feedback. arXiv preprint arXiv:2312.14436, 2023.

[24] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

[25] Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, and Chelsea Finn. Roboreward: General-purpose vision-language reward models for robotics. arXiv preprint arXiv:2601.00675, 2026.

[26] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36: 68539–68551, 2023.

[27] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36:11809–11822, 2023.

[28] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

[29] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

[30] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.

[31] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In International Conference on Learning Representations, 2023.

[32] Nicolás Astorga, Tennison Liu, Yuanzhang Xiao, and Mihaela van der Schaar. Autoformulation of mathematical optimization models using LLMs. In Proceedings of the International Conference on Machine Learning, 2025.

[33] Ali AhmadiTeshnizi, Wenzhi Gao, Herman Brunborg, Shayan Talaei, Connor Lawless, and Madeleine Udell. OptiMUS-0.3: Using large language models to model and solve optimization problems at scale. arXiv preprint arXiv:2407.19633, 2024.

[34] Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Mark Malhotra, et al. Towards a science of scaling agent systems. arXiv preprint arXiv:2512.08296, 2025.

[35] Young-Jun Lee, Seungone Kim, Byung-Kwan Lee, Minkyeong Moon, Yechan Hwang, Jong Myoung Kim, Graham Neubig, Sean Welleck, and Ho-Jin Choi. RefineBench: Evaluating refinement capability in language models. In First Workshop on Multi-Turn Interactions in Large Language Models, 2025.

## A Discussion and Future Directions

From our experimental analysis, we identified several significant limitations of the ARPG framework, as well as opportunities to extend it.

Parameter Sensitivity in Policy Training. Even after a semantically valid MDP formulation and executable environment are obtained, some failures still occur during policy training. These failures often arise from numerical calibration issues, such as improperly balanced reward weights or coordinate-frame inconsistencies, rather than from missing state, action, reward, or termination semantics. These issues do not reflect fundamental limitations of the proposed framework, but rather stem from its sensitivity to implementation-level parameters that are only weakly constrained by natural language descriptions. As experience accumulates across tasks and environments, such parameterization errors could be progressively mitigated through data-driven calibration, metaoptimization of reward coefficients, or improved priors over common coordinate conventions.

Trade-offs in Agent Granularity. We observed that the ARPG framework achieves high success rates, partly due to its relatively fine-grained decomposition of the formulation process. However, increased agent granularity can incur higher computational and budget costs when agent contexts and interactions are not explicitly bounded. In ARPG, role-specific contexts and compact artifact interfaces mitigate this overhead, which is consistent with the lower token usage and shorter runtime observed against general-purpose agentic systems. These results suggest that a more fine-grained decomposition is not universally optimal, but instead reflects a trade-off between formulation success, inference cost, and the level of reasoning transparency. Depending on the complexity of the task and the desired degree of interpretability, different levels of agent granularity may be preferable.

Importantly, the optimal degree of decomposition is also expected to evolve with advances in single agent reasoning capabilities. As individual language models become more reliable at handling longer dependency chains and structured verification internally, fewer formulation agents may suffice to achieve comparable performance. This motivates the need for a manager-level agent that can adaptively control the degree of agent decomposition based on the task description, target transparency, resource constraints, and the underlying capabilities of the language model, rather than relying on a fixed agent structure.

Simulator-Specific Structure in Environment Implementation. During the implementation phase, variations in configuration structures that depend on tasks and simulators pose a recurring challenge. Different simulators expose distinct component abstractions, dependency structures, and configuration units. Even within the same simulator, different tasks may require different combinations of managers (e.g., scene, observation, command, reward, and termination). Incorrect assumptions about these structures often lead to subtle runtime errors. While grounding code generation in simulator documentation via retrieval-augmented generation (RAG) can improve robustness within a fixed simulator, a more general approach is to design an agent that explicitly reasons about the target simulator architecture itself. Such an agent could dynamically determine how the implementation phase should be decomposed, selecting the appropriate configuration modules, agent roles, and generation order, based on the simulator’s structural properties. This would enable the framework to adapt its implementation-agent structure across different simulators, moving toward a simulator-agnostic policy generation framework rather than one tailored to a single execution backend.

Prompt Engineering and Reuse. Although the prompts and per-configuration documents used by ARPG may appear lengthy, most of their content specifies reusable simulator-interface structure rather than task-specific tuning. Within Isaac Lab, the same agent prompts are reused across tasks without modification; task variation is introduced through the natural language task description, robot specification, and the corresponding configuration document. Thus, the prompt design is primarily a reusable interface specification for the target environment, rather than per-task prompt engineering.

Generality Beyond Robotics. Although this paper focuses on robotic policy generation, ARPG is not conceptually tied to robot morphology or to Isaac Lab. Its core assumption is that a task can be formulated as an MDP through dependent choices over states, actions, objectives, constraints, rewards, and termination conditions. This structure also appears in engineering domains such as wireless resource allocation and energy management, where task objectives and constraints must be reformulated under changing operating conditions. The additional demonstrations in Appendix G suggest that the dependency-aware modeling stage can transfer beyond the main robotics benchmark suite. However, each new execution backend still requires an implementation interface that exposes the relevant environment dynamics, constraints, and executable code structure. The evaluated scope is state-based MDPs with supported simulator interfaces; visual perception and long-horizon compositional task planning are not demonstrated.

Toward Integrated Autonomous Policy Generation. More broadly, these findings highlight that end-to-end automated RL policy generation is a multi-faceted problem, spanning formulation, environment implementation, and policy training. While ARPG primarily addresses dependency-aware formulation and localized implementation correction, future systems will benefit from integrating adaptive decomposition strategies, documentation- and architecture-aware implementation agents, and training-time optimization mechanisms. We believe that these extensions are a promising step toward fully autonomous and scalable robot policy generation frameworks.

Broader Impacts. ARPG is intended to reduce the manual effort required to formulate and implement robot RL policies. Because the framework operates before deployment, the generated policies are standalone and do not rely on language models during execution, avoiding runtime LLM decision-making in the control loop. However, this does not provide formal safety guarantees for the generated policies. As with other robotics automation tools, the framework may improve development efficiency and accessibility while raising broader concerns related to labor displacement or misuse. We do not identify immediate negative societal impacts unique to this work beyond those commonly associated with robotic automation and RL-based control.

## B Theoretical Analysis on Correction Dynamics and Modeling Success

This appendix provides a conditional explanatory analysis of the correction dynamics underlying Proposition 4.1. We model formulation as an iterative verification-correction process over explicit formulation artifacts. Let

$$
p _ { i } ^ { ( t ) } : = \mathbb { P } ( \arctan { i } { \mathrm { ~ i s ~ i n v a l i d ~ a f t e r ~ i t e r a t i o n ~ } } t ) .
$$

As in Section 4.5, this artifact-wise probability is interpreted conditionally on the validity of preceding artifacts in the sequential formulation chain. The product form in Equation (7) therefore follows the chain rule for successful sequential construction, rather than requiring independent artifact errors.

We characterize the correction dynamics using two effective parameters:

(i) Effective correctability $\alpha _ { i } \ > \ 0$ , the probability that an invalid artifact is successfully corrected in a single iteration.

(ii) Error reintroduction rate $\beta _ { i } \in [ 0 , 1 )$ , the probability that a valid artifact becomes invalid due to the correction of other artifacts.

Under these assumptions, the artifact-wise error probability satisfies

$$
p _ { i } ^ { ( t + 1 ) } \leq \left( 1 - \alpha _ { i } - \beta _ { i } \right) p _ { i } ^ { ( t ) } + \beta _ { i } .\tag{8}
$$

Equation (8) defines a stable linear recurrence. For each formulation artifact $i ,$ the residual error probability satisfies

$$
p _ { i } ^ { ( \infty ) } : = \operatorname* { l i m } _ { t \to \infty } \operatorname* { s u p } _ { i } p _ { i } ^ { ( t ) } \leq \frac { \beta _ { i } } { \alpha _ { i } + \beta _ { i } } .\tag{9}
$$

We note that the residual error bound in Equation (9) exhibits monotonicity with respect to the effective correction dynamics parameters. Specifically, for each formulation artifact $i ,$

$$
\frac { \partial } { \partial \alpha _ { i } } \left( \frac { \beta _ { i } } { \alpha _ { i } + \beta _ { i } } \right) = - \frac { \beta _ { i } } { ( \alpha _ { i } + \beta _ { i } ) ^ { 2 } } \leq 0 ,
$$

and

$$
\frac { \partial } { \partial \beta _ { i } } \left( \frac { \beta _ { i } } { \alpha _ { i } + \beta _ { i } } \right) = \frac { \alpha _ { i } } { ( \alpha _ { i } + \beta _ { i } ) ^ { 2 } } > 0 .
$$

Thus, increasing effective correctability $\alpha _ { i }$ does not increase the residual error bound, while increasing error reintroduction $\beta _ { i }$ strictly increases it.

Proof of Proposition 4.1. Consider two formulation strategies, ARPG and a baseline, with residual error probabilities $p _ { i } ^ { ( \infty , \mathrm { A R P G } ) }$ and $p _ { i } ^ { ( \infty , \mathrm { b a s e } ) }$ , respectively. Suppose that for all $i ,$

$$
p _ { i } ^ { ( \infty , \mathrm { A R P G } ) } \leq p _ { i } ^ { ( \infty , \mathrm { b a s e } ) } ,
$$

with strict inequality for at least one artifact k. Then, by Equation (7) under the conditional residual error interpretation,

$$
\mathbb { P } ( S _ { M } ^ { \mathrm { A R P G } } ) = \prod _ { i } \bigl ( 1 - p _ { i } ^ { ( \infty , \mathrm { A R P G } ) } \bigr ) > \prod _ { i } \bigl ( 1 - p _ { i } ^ { ( \infty , \mathrm { b a s e } ) } \bigr ) = \mathbb { P } ( S _ { M } ^ { \mathrm { b a s e } } ) .
$$

Equation (9) implies that any formulation strategy that induces higher effective correctability $\alpha _ { i }$ or lower error reintroduction $\bar { \beta } _ { i }$ yields a weakly smaller residual error bound. Therefore, under the condition that ARPG reduces at least one residual artifact error probability without increasing the others relative to a baseline, Proposition 4.1 follows.

Relation to Single-Agent and Parallel Strategies. A single-agent formulation strategy that generates a formulation in a single pass without iterative correction yields $p _ { i } ^ { ( \infty ) } = p _ { i } ^ { ( 0 ) }$ . Parallel formulation strategies that modify multiple artifacts simultaneously may increase cross-artifact interference, cor responding to higher effective $\beta _ { i }$ . Under the assumed correction dynamics, these architectural differences may lead to larger residual errors and lower modeling success. These cases illustrate how architectural choices may influence the effective correction dynamics captured by the above analysis.

## C Failure Cases of Executable but Non-learnable Environments

This appendix provides representative failure cases observed in the generated MDP and reward artifacts of the off-the-shelf agentic baselines. The cases come from manually inspected artifacts that are associated with executable generated environments that nevertheless failed the stricter MSR/PGSR criteria. They illustrate the distinction emphasized in Section 5.2: simulator-level execution does not guarantee a learnable or semantically aligned MDP. We group them by the artifact dependency that is broken, using concrete patterns from the inspected Ant, Humanoid, and Franka Lift artifacts. The snippets below are shown with only minimal editorial modifications applied for presentation. Highlighted terms indicate the artifact components most directly involved in the failure pattern.

## C.1 Reward Terms Depend on Unavailable or Inconsistent State Variables

Ant frame convention mismatch. Some Ant MDPs define the observation using body-frame base velocity while the reward is written in terms of world-frame forward velocity along global +X, yaw, heading projection, and target direction. This is valid only if the implementation explicitly reconstructs world-frame quantities. In the failed artifacts, that conversion is left ambiguous, so body-frame velocity can be used as if it were world-frame progress, making the forward reward inconsistent after yaw changes.

Observed failure snippet: Ant frame mismatch (Codex)

$$
\begin{array} { r l } { \mathrm { O b s e r v a t i o n : } \ } & { v _ { \mathrm { b a s e } } \mathrm { i n ~ b o d y / r o o t } \mathrm { f r a m e } } \\ { \mathrm { R e w a r d : } \ } & { r _ { \mathrm { f w d } } = \operatorname* { m a x } ( v _ { x } , w , 0 ) , \quad r _ { \mathrm { h e a d i n g } } = \mathrm { R e L U } ( \cos ( \mathrm { y a w } ) ) } \end{array}
$$

The artifact mixes body-frame observations with world-frame reward quantities without specifying the required frame conversion.

## C.2 Termination and Reward Are Semantically Inconsistent

Humanoid contact and height criteria. In Codex Humanoid artifacts, broad non-foot contacts and low body height are used both as dense penalties and as hard failure events. One representative artifact penalizes low height below 1.10m and any non-foot contact above 1N, while termination fires when root height falls below 0.85m or when non-foot contact exceeds 10N. Isaac Lab locomotion examples do use contact penalties and illegal-contact termination, but they typically separate the intended body set and threshold semantics. Here, crouching, incidental non-foot contact, or recovery attempts can be treated as both heavily penalized behavior and terminal failure. This can make recovery dynamics difficult to learn.

Observed failure snippet: Humanoid termination (Codex)   
Reward: −3.0 max(1.10 − z<sub>root</sub>, 0) − 2.0 N<sub>nonfoot</sub>(F > 1N)   
Termination: z<sub>root</sub> < 0.85 ∨ N<sub>nonfoot</sub>(F > 10N) > 0.   
The artifact does not clearly separate recoverable contact/height deviations from terminal fall events, making the failure   
boundary too broad.

Lift drop threshold versus lift shaping. In several Lift artifacts, reward shaping is expressed relative to the object’s initial height, while termination is specified as an absolute object-height threshold such as $z _ { \mathrm { o b j e c t } } < - 0 . 0 5 , z _ { \mathrm { o b j e c t } } < 0 . 2 5 , \mathrm { o r } z _ { \mathrm { o b j e c t } } < 0 . 3 2 ,$ , depending on the run. An absolute threshold is valid when anchored to the same scene frame, as in the successful ARPG Lift MDP. This also matches Isaac Lab’s Lift example, which uses an absolute object-height drop threshold. The baseline issue is narrower: some artifacts do not state how the threshold is tied to tabletop height, object size, reset distribution, or height-frame convention. The termination rule can therefore be ambiguous relative to the shaping objective.

Observed failure snippet: Lift drop threshold (Claude)

Reward shaping: h<sub>lift</sub> = z<sub>object</sub> − z<sub>init</sub>   
Termination examples: z < {−0.05, 0.25, 0.32} without a shared scene/frame anchor.   
The failure is not the use of an absolute threshold itself, but the lack of an explicit scene/frame anchor for how that   
threshold matches the shaping objective.

## C.3 Sparse or Delayed Task Signals

Lift reward gated by a hard grasp predicate. Many Lift artifacts reward object height only after a binary is\_grasped predicate becomes true. Before that predicate is satisfied, approach and aperture shaping may exist, but the lift and success signals are withheld. The problem is therefore not sparse shaping in general, but delaying the main object-control signal behind a brittle grasp proxy.

Observed failure snippet: delayed Lift signal (Claude)

r<sub>grasp</sub> = I[d<sub>grasp</sub> < 0.10] exp(−|aperture − 0.04|/0.025),   
r<sub>lift</sub> = I[is\_grasped] tanh(h<sub>lift</sub>/0.06),   
r<sub>success</sub> = I[is\_grasped] I[h<sub>lift</sub> > 0.04] 2.0.   
The lift and success signals are unavailable until a hard grasp predicate is already satisfied.

Ant forward progress gated by stability. In some locomotion artifacts, forward reward is multiplied by uprightness and heading alignment rather than given as a standalone dense velocity signal. Early policies that are not already upright and aligned may receive an attenuated movement reward. The agent must therefore improve stability before forward motion becomes strongly profitable.

Observed failure snippet: gated locomotion signal (Claude)

r<sub>fwd</sub> = v<sup>+</sup><sub>x</sub> r<sub>heading</sub> r<sub>upright</sub>   
Forward progress remains present but is strongly attenuated until heading and uprightness are sufficiently good.

## C.4 Auxiliary Reward Terms Obscure the Primary Task Objective

The following cases involve reward decompositions in which auxiliary regularization terms dominate or gate the primary objective signal. Although such failures partly involve numerical calibration, they also reflect a formulation-level dependency error: the primary objective is not preserved as a direct reward signal before auxiliary terms are added.

Ant locomotion. In several Claude Ant artifacts, forward speed is multiplied by heading and uprightness gates, then surrounded by many stability, height, action, joint, and termination penalties. Although each term is individually plausible, the main progress signal is weakened before a stable gait has emerged. Early policies may therefore receive little positive progress reward for exploratory movement unless posture and heading are already good. This is not a criticism of the auxiliary stability terms commonly used in Isaac Lab locomotion tasks. This failure reflects a dependency error between the objective artifact and the reward artifact: auxiliary stability terms are used to gate the primary progress signal, rather than serving as regularizers around it.

Observed failure snippet: Ant reward (Claude)   
r<sub>t</sub> = 2.0 v<sup>+</sup><sub>x</sub> r<sub>heading</sub> r<sub>upright</sub> + 0.8 r<sub>upright</sub> + 0.5 r<sub>heading</sub> + 0.4 r<sub>height</sub>   
−0.30 |v<sub>y</sub>|−0.50 |v<sub>z</sub>|−0.40 r<sub>jump</sub>−4.0 p<sub>low\_height</sub>−5.0 I[early termination].   
The task reward is attenuated by stability gates and surrounded by penalties, so exploratory motion can be weakly   
rewarded before a stable gait is found.

Humanoid locomotion. The same pattern is sharper in Claude Humanoid artifacts. One reward makes forward speed conditional on both uprightness and heading, while adding large penalties for hopping, abnormal height, joint motion, and early termination. Unlike the successful ARPG Humanoid MDP, where forward velocity remains an additive dense signal, this design may weaken the exploratory signal needed to discover walking.

Observed failure snippet: Humanoid reward (Claude)

```html
r<sub>t</sub> = 3.5 tanh(v<sup>+</sup><sub>x</sub> ) q<sub>upright</sub> q<sub>heading</sub> + 1.0 q<sub>upright</sub> + 0.8 q<sub>heading</sub> + 0.5 q<sub>height</sub>
−0.80 |v<sub>z</sub>|−0.60 p<sub>hop</sub>−8.0 p<sub>low_height</sub>−8.0 I[early termination].
Forward velocity is not rewarded as a standalone dense signal; it is multiplied by posture and heading gates and then
dominated by large instability penalties.
```

## C.5 Summary

Overall, these cases show that the failed artifacts are not merely isolated reward-design mistakes. They reflect broken dependencies among state variables, reward terms, termination events, and simulatorgrounded quantities. This is the failure mode targeted by dependency-aware MDP artifactization: reward and termination logic must remain consistent with the state and action artifacts that make the environment both executable and learnable.

## D Robustness to Underspecified Task Descriptions

In practical usage, users rarely provide task descriptions in a fully specified or formally structured manner. Instead, natural-language inputs are often ambiguous, incomplete, or expressed at mixed levels of abstraction. To evaluate the sensitivity of MDP modeling to such input variation, we conduct a case study using an underspecified Humanoid locomotion description, derived from the same setting used in Section 5.2 but stripped of explicit formulation cues. We use the ARPG configuration available for these runs, instantiated with GPT-5.0, and compare it with monolithic LLM baselines and off-the-shelf agentic systems.

## D.1 Modeling Under Underspecified Task Descriptions

Table 6 reports MSR over 10 trials with the underspecified Humanoid description. ARPG maintains a high MSR of 0.8, suggesting that its dependency-aware modeling process remains stable even when explicit formulation cues are removed. This result is consistent with the overall Humanoid results in Table 2, where ARPG also achieves high modeling success under a more detailed task description.

Table 6: MSR under underspecified Humanoid descriptions.
<table><tr><td colspan="4">ARPG Gemini-3 GPT-5.2 Claude Codex</td></tr><tr><td>0.8</td><td>0.3</td><td>0.0</td><td>0.0 0.7</td></tr></table>

Thus, the main observation is not that underspecification uniformly degrades performance, but that ARPG remains relatively robust across different levels of input specificity.

The baseline results show a less consistent pattern. Gemini-3 and Claude remain at the same low MSR levels observed in the detailed-description setting, while GPT-5.2 decreases slightly. In contrast, Codex improves substantially under the underspecified description. This suggests that underspecification is not necessarily harder for all LLM-based systems. In some cases, a shorter and less constrained description may allow a strong general-purpose agentic system to rely on generic locomotion priors, whereas a long, detailed description may introduce additional constraints, distractors, or implementation-specific details that make coherent MDP formulation more difficult.

These results therefore indicate that robustness to natural-language task descriptions should be understood as stability under changes in input specificity, rather than performance under ambiguity alone. Off-the-shelf systems can sometimes infer plausible MDP artifacts from underspecified inputs, as shown by Codex, but their behavior is not consistent across systems. ARPG achieves the highest MSR in this setting while also maintaining strong modeling performance in the detailed-description setting, supporting the claim that dependency-aware artifactization provides a more stable formulation process. Overall, this analysis reinforces the conclusion of Section 5.2: reliable policy generation requires not only executable simulator code, but also an MDP formulation process that remains stable as task descriptions vary in detail and specificity.

## D.2 Modeling Correction Dynamics

This section analyzes the internal behavior of the error-correction mechanism in the modeling phase. Rather than treating the underspecified Humanoid setting only as a performance benchmark, we use it as a stress case that exposes where uncertainty arises during dependency-aware MDP formulation. Because the input intentionally removes explicit formulation cues, such as precise objectives, constraints, and reward-relevant variables, the resulting correction traces provide insight into which modeling artifacts require the most refinement.

![](images/200989862dac0d729615c17050b2a14f120e2807194ad9d0aaec9f83b05bf749.jpg)  
Figure 3: Error-correction agent invocation counts by modeling component with underspecified Humanoid task descriptions.

Figure 3 reports the number of error-correction (EC) agent invocations triggered by each modeling component for the underspecified Humanoid task. Each bar corresponds to a formulation stage whose artifact was flagged for revision during the sequential modeling process. A larger count indicates that the corresponding artifact more frequently required additional verification or refinement before being passed to later stages.

The correction traces show that EC invocations are concentrated in the objective and variable agents. This suggests that, under ambiguous task descriptions, the main uncertainty lies in high-level semantic interpretation: determining what constitutes successful behavior and which state variables are needed to represent progress toward that behavior. In contrast, constraint-related artifacts trigger fewer corrections, suggesting that feasibility conditions and physical limits can often be inferred more directly from the robot specification and simulator priors.

Later-stage modeling components require fewer corrections than the earlier semantic interpretation stages. This pattern is consistent with the dependency-aware design of ARPG. Once objectives and variables are externalized and refined, downstream formalization stages operate on more explicit artifacts, reducing ambiguity in the construction of mathematical and SAR-level representations.

Overall, these traces show that modeling difficulty under underspecified input is not uniformly distributed across the MDP formulation pipeline. Uncertainty is concentrated in objective definition and variable selection, while later formalization and constraint encoding are comparatively more stable. This supports the role of stage-wise verification and localized correction: the framework can refine fragile upstream artifacts without regenerating the entire formulation. The observation is also consistent with the theoretical analysis in Section 4.5, which links localized correction and reduced error propagation to higher modeling success.

Pre/post audit of triggered corrections. To examine whether EC repairs errors or introduces new ones, we retrospectively audited 50 Humanoid generation trials. EC was triggered in 60 of 350 possible stage events. Each of the 60 triggered EC events used a seven-item checklist, yielding 420 paired pre/post item decisions.

Table 7: Pre/post checklist outcomes for triggered Humanoid EC events.
<table><tr><td>Checklist outcome</td><td>Count</td><td>Percentage</td></tr><tr><td>Violation corrected  $( \mathrm { X }  \mathrm { O } )$ </td><td>130</td><td>30.95%</td></tr><tr><td>Violation unresolved  $( \mathrm { X } \to \mathrm { X } )$ </td><td>74</td><td>17.62%</td></tr><tr><td>Satisfied item preserved  $( \mathrm { O } \to \mathrm { O } )$ </td><td>186</td><td>44.29%</td></tr><tr><td>Violation introduced  $( \mathrm { O } \to \mathrm { X } )$ </td><td>30</td><td>7.14%</td></tr></table>

In Table 7, checklist satisfaction increased from 216/420 (51.4%) to 316/420 (75.2%). Of the 60 triggered events, 10 resolved all violations and 34 reduced the violation count without resolving all violations, giving a net improvement in $4 4 / 6 0$ events (73.3%). The other 16 events did not reduce the violation count. EC can therefore leave errors unresolved or introduce new violations. This audit is limited to triggered Humanoid events; untriggered artifacts were not independently labeled, so it does not estimate the false-negative rate of confidence-based routing. Item-level percentages are descriptive because checklist items within an artifact are not independent.

## E Implementation Difficulty Analysis via Debug Statistics

This section analyzes how implementation difficulty in ARPG varies across tasks with different structural properties. After the MDP formulation is fixed, the implementation phase translates the formulation into simulator-specific environment code and invokes debugging when execution or integration errors arise. We therefore use coding-agent debug invocations as a proxy for the corrective effort required to ground a modeled MDP in an executable Isaac Lab pipeline.

Rather than treating implementation difficulty as a simple function of control dimensionality, we examine where debugging effort is spent across simulator components. This distinction is important because different tasks stress different parts of the implementation interface: low-dimensional tasks may expose strict API assumptions, locomotion tasks may require careful reward and termination grounding, and manipulation tasks may depend heavily on observation construction.

![](images/c99f8673907db27aa78c8eabcb30c588b3a06a761b5954d548bb877ab5f1b402.jpg)  
(a) Debug effort vs. DoF.

![](images/b5aaec9327966b970c9385c52c26e8891536ff4a2afe608e41d54434fb2bba16.jpg)  
(b) Debug locations by task.  
Figure 4: Implementation difficulty statistics from coding-agent debug invocations.

Figure 4a compares normalized degrees of freedom (DoF) with normalized average debug iterations across the benchmark tasks. The results show that DoF alone does not explain implementation difficulty. Humanoid has the largest DoF and also requires substantial debugging effort, but Reach exhibits the highest normalized debug effort despite having a much lower DoF. CartPole also requires non-negligible debugging effort even though its control dimensionality is minimal, while Lift shows moderate overall effort despite involving object manipulation. Thus, corrective effort is better explained by simulator-grounding requirements than by action-space dimensionality alone.

Figure 4b decomposes debug invocations according to the revised implementation component. Here, obs, reward\_custom, termination, and code denote corrections to observation construction, custom reward implementation, termination logic, and integrated environment code, respectively. Reach and Lift are dominated by obs corrections, with 13 and 9 invocations, indicating that manipulation tasks are especially sensitive to how object poses, end-effector states, command targets, and relative geometric quantities are exposed to the policy. Ant is dominated by reward\_custom revisions, with 11 invocations, suggesting that task-specific locomotion rewards require careful grounding in simulator-available quantities beyond the base template. Humanoid shows a more distributed pattern, with corrections concentrated in both reward\_custom and termination, reflecting the coupling among balance, forward progress, failure detection, and high-dimensional body-state representation. CartPole, despite its low DoF, concentrates most corrections in code, indicating that even simple control problems can require repeated simulator-interface fixes when the MDP itself is relatively straightforward.

These results suggest that implementation difficulty in ARPG is component-specific rather than uniformly task-specific. The source of difficulty depends on which part of the generated environment must be grounded most precisely: observation construction for object-centric manipulation, reward implementation for task-specific locomotion, termination logic for unstable high-dimensional systems, and code integration for lightweight environments with strict API assumptions. Average debug effort and component-level debug location therefore provide complementary signals. The former estimates total corrective effort, while the latter identifies which implementation artifacts would benefit most from additional validation or more specialized implementation agents.

Overall, this analysis provides a more fine-grained view of ARPG’s implementation phase. When transferring to new simulators or new robot tasks, tasks with rich object-state structure may benefit from stronger observation validation, tasks with custom behavioral objectives may benefit from reward-specific checks, and systems with fragile failure modes may require more explicit termination verification. This component-level interpretation also clarifies why ARPG separates MDP formulation from simulator implementation: once a valid MDP is fixed, remaining failures are often localized to particular simulator components rather than to the entire policy generation pipeline.

## F Details of Success Criteria

## F.1 Modeling Success Criteria

This appendix defines the evaluation criteria used to assess whether an MDP formulated by each model constitutes a valid and meaningful RL problem. We evaluate MDPs along four components: State, Action, Reward, and Overall. An MDP is counted as an MSR success if its trained policy passes PGSR criteria or all applicable component-level criteria are satisfied.

Operational MSR protocol. A single researcher with robotics and RL expertise evaluates the realized MDP with method and model identities removed. Trials whose trained policies satisfy the PGSR criteria are counted as MSR successes. For the remaining trials, all applicable criteria in Tables 8–12 must pass.

CartPole Evaluation Criteria. An extracted MDP is evaluated using the following implementationoriented criteria:

• State: Includes all four core continuous state variables–cart position, cart velocity, pole angle, and pole angular velocity (notation variants allowed)–such that the Markov property is naturally satisfied.

• Action: A single one-dimensional continuous control applied exclusively to the cart (e.g., force or effort on slider\_to\_cart or equivalent cart actuation).

• Reward: Aligned with the objective of keeping the pole upright and explicitly specifies at least one task-relevant failure or termination condition (e.g., exceeding an angular limit and/or a cart position limit).

• Overall: State, Action, and Reward jointly define a coherent SAR-formatted MDP, and the final formulation or implementation has clearly identifiable state, action, reward, and termination definitions.

• Rubric: The exact criteria used to assess the MDP are provided in Table 8.

Table 8: Operational MSR criteria for CartPole.
<table><tr><td>Component</td><td>Pass criterion</td><td>Failure criterion</td></tr><tr><td>Structure</td><td>Observation/state, action, reward, and failure/termina- A field is missing or contradictory. tion are identifiable and mutually consistent.</td><td></td></tr><tr><td>Cart position</td><td>Continuous cart displacement is observed or explicitly Position is missing or only a Boolean is derivable from observations.</td><td>given.</td></tr><tr><td>Cart velocity</td><td>Continuous cart velocity is observed or explicitly deriv- Velocity is unavailable. able from observations.</td><td></td></tr><tr><td>Pole angle</td><td>Pole angle relative to upright is observed or equivalently Pole angle is unavailable or only an up- represented.</td><td>right Boolean is given.</td></tr><tr><td>Pole angular velocity</td><td>y Continuous pole angular velocity is observed or explic- Angular velocity is unavailable. itly derivable.</td><td></td></tr><tr><td>Action type</td><td>Continuous force/effort control has explicit numerical Control is discrete or its type/bounds are or symbolic bounds.</td><td>unresolved.</td></tr><tr><td>Action dimension</td><td>One declared schema: 1D cart force.</td><td>Cart force is absent or dimensions con- flict.</td></tr><tr><td>Upright objective</td><td>The task reward favors smaller pole-angle error, with The upright objective is absent or re- other inputs fixed.</td><td>versed.</td></tr><tr><td>Failure boundary</td><td>At least one finite pole-angle or cart-position failure Only timeout is specified, or the boundary boundary is defined.</td><td>is unresolved.</td></tr><tr><td>Joint coherence</td><td>State, action, and failure logic refer to consistent Joint mappings conflict. cart/pole joints.</td><td></td></tr></table>

Ant Evaluation Criteria. An extracted MDP is evaluated using the following implementationoriented criteria:

• State: Includes a sufficient set of proprioceptive variables for quadruped locomotion, such as base orientation or uprightness cues, base linear and angular velocities, and joint positions and velocities for all eight actuated joints. In addition, the state (or reward logic) must include information relevant to detecting collapse, such as the base height. Missing essential proprioceptive components or collapse-related signals is considered invalid.

• Action: An eight-dimensional continuous control corresponding to joint-level effort or torque commands applied to the ant’s actuated leg joints. Actions must be conceptually consistent with joint-effort actuation in Isaac Lab. Discrete controls, incorrect dimensionality, or ambiguous actuation semantics are considered invalid.

• Reward: Aligned with the objective of stable forward locomotion along the positive X-axis and includes explicit incentives for forward progress as well as penalties related to body stability (e.g., tilt or angular velocity) and control efficiency (e.g., effort magnitude or action smoothness). The reward formulation must also clearly specify termination conditions, including collapse due to insufficient base height and a finite episode horizon. Rewards that omit or ambiguously define these termination conditions are considered invalid.

• Rubric: The exact criteria used to assess the MDP are provided in Table 9.

Table 9: Operational MSR criteria for Ant.
<table><tr><td>Component</td><td>Pass criterion</td><td>Failure criterion</td></tr><tr><td>Structure</td><td>Observation/state, action, reward, collapse, and timeout A required field is missing. are identifiable.</td><td></td></tr><tr><td>Base state</td><td>velocity have explicit raw or derived sources.</td><td>Orientation/uprightness, height, and 3D linear/angular Essential base-state information is un- available.</td></tr><tr><td>Joint state Action dimension</td><td>Positions and velocities cover all eight actuated joints. A required joint or its velocity is missing. One continuous 8D action vector is fully specified.</td><td>The action is noncontinuous, or its dimen-</td></tr><tr><td></td><td></td><td>sion is unresolved or differs from eight. Joint effort/torque or normalized commands map explic- Commands conflict with the declared ac-</td></tr><tr><td>Action semantics</td><td>itly to the effort actuators.</td><td>tuator interface. Global +X progress or speed is rewarded as a locomo- Forward progress is absent, direction is</td></tr><tr><td>Forward signal</td><td>tion objective.</td><td>unresolved, or progress is only indirectly gated.</td></tr><tr><td>Stability</td><td>Uprightness, tilt, angular stability, or height is shaped A stability incentive is absent or reversed. in the correct direction.</td><td></td></tr><tr><td>Efficiency</td><td>Effort, action magnitude/change, or energy is penalized. A control-efficiency term is absent or re-</td><td>versed.</td></tr><tr><td>Collapse</td><td>A resolved base-height collapse threshold is at least The threshold is unresolved, below the 0.31 m and distinguishes collapse from the declared benchmark lower bound, or inconsistent upright configuration.</td><td>with the upright configuration.</td></tr><tr><td>Horizon</td><td>A resolved finite horizon lies in [16, 20] s.</td><td>The horizon is absent, unresolved, or out- side the benchmark range.</td></tr><tr><td>Forward frame</td><td>Forward motion uses world +X or an explicit body-to- Body-frame velocity is used as world- world transform.</td><td>frame forward velocity without a trans-</td></tr><tr><td>Temporal source</td><td>Any previous action used has an observation or declared A temporal dependency has no available manager-memory source.</td><td>form. source.</td></tr></table>

Humanoid Evaluation Criteria. An extracted MDP for the humanoid locomotion task is evaluated using the following implementation-oriented criteria:

• State: Includes base position or height and orientation together with linear and angular velocities, as well as joint positions and joint velocities. The state also contains at least one stability-relevant signal, such as uprightness, forward heading (+X), base height, or foot contact.

• Action: A continuous per-joint control vector compatible with Isaac Lab’s implicit actuator model (e.g., joint position targets or normalized joint commands).

• Reward: Encourages forward progression and stable upright locomotion, and explicitly specifies both termination conditions: a finite episode horizon and a base-height collapse threshold.

• Overall: State, Action, and Reward jointly define a coherent SAR-formatted MDP that can be used consistently across episodes.

• Rubric: The exact criteria used to assess the MDP are provided in Table 10.

Table 10: Operational MSR criteria for Humanoid.
<table><tr><td>Component</td><td>Pass criterion</td><td>Failure criterion</td></tr><tr><td>Structure</td><td>Observation/state, action, reward, collapse, and timeout A field is missing or the schema is contra- are identifiable and consistent.</td><td>dictory.</td></tr><tr><td>Base state</td><td>Base position/height, orientation, and linear/angular ve- Required pose or velocity information is locities have explicit raw or derived sources.</td><td>unavailable.</td></tr><tr><td>Joint state</td><td>Positions and velocities cover every controlled joint.</td><td>A controlled joint or its velocity is miss- ing, or the state and action joint sets dis-</td></tr><tr><td>Stability cue</td><td>At least one uprightness, heading, height, or foot- No stability cue is available. contact cue has a declared raw or derived source.</td><td>agree.</td></tr><tr><td>Action schema</td><td>One continuous per-joint vector has a uniquely deter- The action is noncontinuous, its dimen- mined dimension.</td><td>sion is unresolved, or conflicting schemas</td></tr><tr><td>Action semantics</td><td>Position targets, deltas, or normalized targets map to the Raw effort-only commands or other se- implicit actuators.</td><td>remain. mantics conflict with the actuators.</td></tr><tr><td>Forward signal</td><td>Global +X progress or velocity is rewarded as a loco- The reward encourages standing only or motion objective.</td><td>forward progress is only indirectly gated.</td></tr><tr><td>Stability</td><td>Uprightness, height, angular stability, or contact pro- Balance shaping is absent or reversed. vides balance shaping.</td><td></td></tr><tr><td>Collapse</td><td>An explicit base-height collapse threshold is at least The threshold is unresolved, below the 0.60 m and distinguishes collapse from the declared benchmark lower bound, or inconsistent upright configuration.</td><td>with the upright configuration.</td></tr><tr><td>Horizon</td><td>A resolved finite horizon lies in [16, 20] s.</td><td>The horizon is absent, unresolved, or out- side the benchmark range.</td></tr><tr><td>Recoverability</td><td>Recoverable posture/contact deviations are distin- Minor recoverable non-foot contact is un- guished from catastrophic termination.</td><td>conditionally terminal.</td></tr><tr><td>Temporal source</td><td>Any previous action/action change has an observation A temporal dependency has no available or declared manager-memory source.</td><td>source.</td></tr></table>

Reach Evaluation Criteria. An extracted MDP for the end-effector reaching task is evaluated using the following implementation-oriented criteria:

• State: Includes the robot joint positions and joint velocities, as well as the commanded end-effector position, with orientation and temporal information included as required by the reward formulation.

• Action: Joint-space continuous actions compatible with the implicit actuator formulation, controlling the robot arm consistently across all degrees of freedom.

• Reward: Encourages accurate tracking of the commanded end-effector pose, including position and (optionally) orientation tracking, and explicitly specifies a time-based termination condition.

• Overall: State, Action, and Reward jointly define a coherent SAR-formatted MDP that can be used consistently across episodes.

• Rubric: The exact criteria used to assess the MDP are provided in Table 11.

Table 11: Operational MSR criteria for Franka Reach.
<table><tr><td>Component</td><td>Pass criterion</td><td>Failure criterion</td></tr><tr><td>Structure</td><td>Observation/state, action, reward, and timeout are iden- A required field is missing. tifiable.</td><td></td></tr><tr><td>Joint state</td><td>Positions and velocities cover all seven arm joints.</td><td>A required joint or its velocity is missing. The commanded 3D end-effector position is in, or ex- The commanded position is unavailable</td></tr><tr><td>Target command</td><td>plicitly derivable from, realized observations.</td><td>from the realized observations.</td></tr><tr><td>Orientation</td><td>If orientation is rewarded, actual and commanded orien- One orientation source is unavailable. tations have declared observable sources or equivalents.</td><td></td></tr><tr><td>Action history</td><td>If action rate is rewarded, the previous action has an The previous action is only optional or observation or declared manager-memory source.</td><td>has no available temporal source.</td></tr><tr><td>Action dimension</td><td>A continuous 7D arm joint-space command is fully Cartesian/effort-only control conflicts specified.</td><td>with the interface.</td></tr><tr><td>Action semantics</td><td>Position targets, deltas, or normalized targets map to the Action semantics conflict with the actua- implicit actuators.</td><td>tor interface.</td></tr><tr><td>Position tracking</td><td>The position-tracking task reward increases as position The task incentive is reversed or flat near error decreases, including near the target.</td><td>the target.</td></tr><tr><td>Orientation metric</td><td>If quaternion orientation is rewarded, equivalent q and The orientation metric depends on quater- —q have equal reward.</td><td>nion sign.</td></tr><tr><td>Timeout</td><td>A finite timeout is resolved in seconds or steps.</td><td>Timeout is absent or unresolved.</td></tr><tr><td>Pose frame</td><td>Actual and commanded poses share a frame or have an World and base coordinates are mixed explicit transform.</td><td>without a transform.</td></tr></table>

Franka Lift Evaluation Criteria. An extracted MDP for the Franka lift manipulation task is evaluated using the following implementation-oriented criteria:

• State: Includes the robot joint positions and joint velocities, including the gripper joints, the object pose and velocities relative to the robot, and the commanded target pose. Temporal information is available as required by the reward and termination logic.

• Action: Joint-space actions consistent with the implicit actuator formulation, where actions represent joint position targets, joint deltas, or normalized joint commands. Arm and gripper dimensions are defined consistently.

• Reward: Reflects the full manipulation process, including approach, grasp, lift, and maintenance, and explicitly includes a drop-related termination condition (e.g., object height falling below −0.05 m).

• Overall: State, Action, and Reward jointly define a coherent SAR-formatted MDP that can be used consistently across episodes.

• Rubric: The exact criteria used to assess the MDP are provided in Table 12.

Table 12: Operational MSR criteria for Franka Lift.
<table><tr><td>Component</td><td>Pass criterion</td><td>Failure criterion</td></tr><tr><td>Structure</td><td>Observation/state, action, staged reward, drop, and time- A required field is missing. out are identifiable.</td><td></td></tr><tr><td>Robot state</td><td>Positions/velocities cover seven arm and two gripper A required arm or gripper position or ve- joints.</td><td>locity is missing.</td></tr><tr><td>Object position</td><td>The object 3D position is observed or explicitly deriv- It is absent or available only as latent sim- able from observations.</td><td>ulator state.</td></tr><tr><td>Object orientation</td><td>Object orientation is observed or equivalently repre- It is optional only, absent, or latent only. sented.</td><td></td></tr><tr><td></td><td>Object linear velocity The object 3D linear velocity is observed or explicitly It is absent or latent only. derivable.</td><td></td></tr><tr><td>ity</td><td>Object angular veloc- The object 3D angular velocity is observed or explicitly It is absent or latent only. derivable.</td><td></td></tr><tr><td>Relative geometry</td><td>Hand/fingertip-to-object geometry is available or com- Relative geometry cannot be computed putable in a common frame.</td><td>consistently.</td></tr><tr><td>Lift target</td><td>The 0.04 m lift target relative to initial/table height, or Lift direction, reference height, or target the equivalent commanded target, is explicit.</td><td>magnitude is unresolved.</td></tr><tr><td>Arm action</td><td>All seven arm joints receive continuous position targets, Arm dimensions are incomplete or com- deltas, or normalized targets.</td><td>mands conflict with the interface.</td></tr><tr><td>Gripper action</td><td>A two-finger mapping completes one consistent 9D Gripper control is absent or 8D/9D action vector.</td><td>schemas conflict.</td></tr><tr><td>Approach</td><td>Reducing approach distance provides a task incentive All task incentives vanish in the pre-grasp before grasping.</td><td>region.</td></tr><tr><td>Grasp acquisition</td><td>Proximity, centering, aperture, or contact supplies a Only binary grasp success is rewarded. continuous grasp-acquisition signal.</td><td></td></tr><tr><td>Lift</td><td>Lifting is rewarded through object height or progress. Only end-effector height is rewarded.</td><td></td></tr><tr><td>Stage bridge</td><td>grasp gate on the lift reward.</td><td>A continuous approach/grasp signal bridges any hard All task gradients vanish before a hard grasp gate.</td></tr><tr><td>Maintenance</td><td>Grasp/height/hold reward continues at the target height. Reward occurs only once at threshold</td><td>crossing.</td></tr><tr><td>Drop termination</td><td>Both timeout and object-drop termination are defined. Drop has only a penalty without termina-</td><td>tion, or timeout is absent.</td></tr><tr><td>Drop frame</td><td>The drop threshold has an explicit table/reset/object- The threshold frame or scene anchor is frame reference.</td><td>unresolved.</td></tr><tr><td>Temporal source</td><td>Initial height, previous action, or a phase latch has an A temporal dependency has no available observation or declared manager-memory source when source. used.</td><td></td></tr></table>

## F.2 Task-Specific Policy Generation Success Criteria

For policy generation evaluation, we define task-specific success criteria in behavioral terms. Each criterion specifies the observable conditions under which a learned policy is considered to have successfully completed the intended task. These criteria are used solely for evaluation and do not affect reward design, termination conditions, or training dynamics. Each trained policy is evaluated automatically on 20 fixed-seed episodes and passes if at least 16 satisfy all task-specific conditions. PGSR is computed over all 10 independent generation-and-training trials, including failures at earlier stages.

CartPole. An evaluation episode is successful if it completes the 5-s horizon without failure, keeps the cart within 3 m of the track center throughout the episode, and maintains an absolute pole-angle error of at most 0.20 rad throughout the interval from 1 s to 5 s.

Ant. An evaluation episode is successful if it completes the full 16-s horizon without early termination, never lets the base height fall below 0.31 m, and achieves a mean forward velocity of at least 0.50 m/s. Throughout the interval from 1 s to 16 s, the up-axis projection must remain above 0.93 and the heading projection above 0.80.

Humanoid. An evaluation episode is successful if it completes the full 16-s horizon without early termination, never lets the base height fall below 0.60 m, and achieves a mean forward velocity of at least 0.50 m/s. Throughout the interval from 1 s to 16 s, the up-axis projection must remain above 0.93 and the heading projection above 0.80.

Reach. An evaluation episode is successful if, for each of three 4-s targets, the end-effector position error remains at most 0.10 m throughout the final 1-s hold window.

Lift. An evaluation episode is successful if it completes the full 5-s horizon without failure, never lets the object height fall below -0.05 m, and maintains an object height above 0.04 m and a goalposition error of at most 0.05 m throughout the final second.

## F.3 Confidence Intervals for Success Rates

All task-wise MSR, EISR, and PGSR values in Table 2 and the matched ablations are proportions of $n = 1 0$ independent generation/training trials. For a fixed confidence level and sample size, the binomial Wilson interval depends only on the number of successes k. Thus, the same reported rate has the same interval across methods, tasks, and metrics. Table 13 provides all possible intervals without adding columns to the main results tables.

For $\hat { p } = k / n$ and $z = 1$ .959964, the two endpoints are

$$
\frac { \hat { p } + z ^ { 2 } / ( 2 n ) \pm z \sqrt { \hat { p } ( 1 - \hat { p } ) / n + z ^ { 2 } / ( 4 n ^ { 2 } ) } } { 1 + z ^ { 2 } / n } .
$$

Table 13: 95% Wilson CIs.
<table><tr><td>Successes</td><td>Rate</td><td>95% CI</td></tr><tr><td>0/10</td><td>0.0</td><td>[0.00, 0.28]</td></tr><tr><td>1/10</td><td>0.1</td><td>[0.02, 0.40]</td></tr><tr><td>2/10</td><td>0.2</td><td>[0.06, 0.51]</td></tr><tr><td>3/10</td><td>0.3</td><td>[0.11, 0.60]</td></tr><tr><td>4/10</td><td>0.4</td><td>[0.17, 0.69]</td></tr><tr><td>5/10</td><td>0.5</td><td>[0.24, 0.76]</td></tr><tr><td>6/10</td><td>0.6</td><td>[0.31, 0.83]</td></tr><tr><td>7/10</td><td>0.7</td><td>[0.40, 0.89]</td></tr><tr><td>8/10</td><td>0.8</td><td>[0.49, 0.94]</td></tr><tr><td>9/10</td><td>0.9</td><td>[0.60, 0.98]</td></tr><tr><td></td><td></td><td></td></tr><tr><td>10/10</td><td>1.0</td><td>[0.72, 1.00]</td></tr></table>

The intervals are computed using 10 independent trials for each method and task, treating each trial outcome as a Bernoulli observation under the stated evaluation protocol.

## G Additional Generality Demonstrations

## G.1 Overview

To further examine whether ARPG can be instantiated beyond the Isaac Lab tasks used in the main experiments, we provide additional demonstrations on a broader set of sequential decision-making problems. These demonstrations cover custom robotics tasks, standard RL benchmark tasks, and domain-specific engineering problems. Across these settings, the dependency-aware modeling pipeline is retained, while only the simulator- or environment-specific implementation interface is adapted to the target domain. Because the tasks use heterogeneous evidence types, including rollout videos, evaluation curves, quality-of-service (QoS) metrics, and expert comparisons, we report them as generality demonstrations rather than as a unified success-rate benchmark.

Table 14 summarizes the task diversity in these additional demonstrations. The tasks span additional Isaac Sim robot behaviors, Gymnasium benchmarks, lightweight decision-making environments, and engineering-domain MDPs, thereby testing whether generated MDP artifacts remain useful across different state, action, reward, and constraint structures. The Isaac Sim robotics tasks use the same implementation-phase structure as the main robotics experiments, including per-configuration implementation agents. The non-robotic tasks use a simplified implementation phase with a single coding agent. Thus, the robotics demonstrations provide qualitative evidence of transfer to additional robot behaviors, while the quantitative plots below focus on settings evaluated with the simplified implementation pipeline.

Table 14: Summary of additional demonstrations beyond the main Isaac Lab benchmark tasks.
<table><tr><td>Task</td><td>Task type</td><td>Backend</td><td>Objective</td><td>Reported evidence</td></tr><tr><td>Diagonal CartPole</td><td>Robotics variant</td><td>Isaac Sim</td><td>Balance a cart-pole system with a diagonal initial pole.</td><td>Rollout video</td></tr><tr><td>Jumping Ant</td><td>Robotics variant</td><td>Isaac Sim</td><td>Perform repeated vertical jumps in place.</td><td>Rollout video</td></tr><tr><td>In-place rotation Ant</td><td>Robotics variant</td><td>Isaac Sim</td><td>Rotate the Ant in place without translational drift.</td><td>Rollout video</td></tr><tr><td></td><td>Jumping Humanoid Robotics variant</td><td>Isaac Sim</td><td>Generate vertical jumping behavior for a humanoid robot.</td><td>Rollout video</td></tr><tr><td>Pick-and-place</td><td>Robotics variant</td><td>Isaac Sim</td><td>Transfer a cube between two tables.</td><td>Rollout video</td></tr><tr><td>CartPole</td><td>Classic control</td><td>Gymnasium</td><td>Balance the pole with discrete cart forces.</td><td>Evaluation curve</td></tr><tr><td>MountainCar</td><td>Classic control</td><td>Gymnasium</td><td>Reach the hilltop goal under sparse rewards.</td><td>Evaluation curve</td></tr><tr><td>Drone delivery</td><td>Grid-world MDP</td><td>Grid env.</td><td>Complete pickups and deliveries before Evaluation curve energy depletion.</td><td></td></tr><tr><td>Inventory management</td><td>Stochastic control</td><td>Custom env.</td><td>Choose item-wise orders under demand Evaluation curve uncertainty and costs.</td><td></td></tr><tr><td>Wireless scheduling Engineering MDP</td><td></td><td>Engineering env.</td><td>Satisfy QoS while minimizing transmit power.</td><td>Expert comparison (reward &amp; QoS)</td></tr><tr><td>Energy management</td><td>Engineering MDP</td><td>Engineering env.</td><td>Satisfy demand using grid energy and battery storage.</td><td>Expert comparison (pattern &amp; cost)</td></tr></table>

## G.2 Additional Isaac Sim Robotics Tasks

For the additional Isaac Sim robotics tasks, we provide rollout videos in the supplementary material rather than static figures. These demonstrations verify that ARPG can instantiate new robot behaviors by changing the task description and robot/task configuration without changing the dependency-aware modeling pipeline.

## G.3 Standard RL and Decision-Making Tasks

We next consider standard RL and lightweight decision-making tasks outside Isaac Sim. These tasks include classic control benchmarks, grid-based planning, and stochastic inventory control, each with different state, action, and reward structures from the robotics environments. Unlike the Isaac Sim tasks, they use lightweight Gymnasium or Python-based backends and are implemented with a simplified implementation phase using a single coding agent. This setting tests whether the dependency-aware modeling pipeline can generate executable MDP formulations when the simulator interface and task dynamics differ substantially from robot control.

Figure 5 reports evaluation results on four additional standard RL and stochastic decision-making tasks. These non-Isaac tasks use different simulator interfaces and reward templates from the main robotics experiments. Nevertheless, ARPG produces executable formulations whose learning curves indicate successful policy optimization, suggesting that the explicit MDP artifact chain is not tied to a single simulator family.

## G.4 Engineering-Domain MDPs

We consider two engineering-domain MDPs with objectives and constraints that differ substantially from robot locomotion and manipulation. For both tasks, ARPG uses the same modeling phase agents, while the implementation phase is adapted to lightweight Python environments.

## G.4.1 Wireless Scheduling

We consider a wireless scheduling problem in which a base station allocates radio resources to multiple users over a sequence of time slots. At each slot, the controller observes the users’ channel conditions and traffic-related state, selects which user to serve, and chooses a transmit-power level. The objective is to reduce power consumption while maintaining communication quality, expressed through a minimum average-rate requirement. This task differs from the robotics benchmarks because the dynamics arise from stochastic channel variation and queue/resource-allocation decisions rather than physical robot motion. It therefore tests whether ARPG can formulate a constrained MDP whose reward and constraints capture a domain-specific trade-off between efficiency and service reliability.

![](images/bf8172441efbf3439277e0e8da56d460fae03cacf3d2867935fe8bc14658bea6.jpg)  
(a) CartPole

![](images/3e0abac9aed6821b82cb6fc3296cce4a53d81721d0b6ce65829a25f2ff845334.jpg)  
(b) MountainCar

![](images/5947687e655850bf7622b40716551b4be25c5a357b3f873451888cc5328a6e53.jpg)  
(c) Drone delivery

![](images/352101240ca235f27933aea27e4063b5470ed1471f944527e01a4f267df2b597.jpg)  
(d) Inventory management  
Figure 5: Evaluation results for additional standard RL and stochastic decision-making tasks.

![](images/1c9b0cdd79e8e50450df4762705d5836ffdb06cec11bd7e4715133b2ea7cb681.jpg)  
(a) Wireless scheduling reward

![](images/376a87e2fa4453cfd4ced26c0cbe4265a5d35c33a8a0ab1765973e88c212d502.jpg)  
(b) Wireless QoS satisfaction  
Figure 6: Comparison with the expert-designed DQN policy for the wireless scheduling task.

Figure 6 shows the wireless scheduling experiment. ARPG formulates the task as a constrained MDP in which the agent observes time-varying channel conditions and jointly selects the scheduled user and transmit-power level at each slot. The ARPG-generated policy reaches reward levels close to those of the expert-designed DQN baseline while satisfying the required QoS constraint throughout evaluation. The policy also avoids unnecessarily high transmit-power allocation, indicating that the generated reward and constraint artifacts capture the intended trade-off between service reliability and power efficiency. These results provide evidence that ARPG can generate meaningful MDP formulations for constrained resource-allocation problems beyond physical robot control.

## G.4.2 Energy Management

We consider an energy management problem in which a controller schedules grid purchase, renewable usage, and battery charging or discharging over a sequence of time steps. At each step, the controller observes demand, renewable generation, battery state of charge, and time-of-use (TOU) electricity prices. The objective is to satisfy demand while minimizing total electricity cost, using the battery to shift energy consumption away from high-price periods when possible. This task differs from the robotics benchmarks because the dynamics are governed by storage constraints, exogenous demand, renewable availability, and time-varying prices rather than physical robot motion. It therefore tests whether ARPG can formulate a cost-sensitive scheduling MDP with resource-balance constraints and temporally coupled decisions.

![](images/5f8fa7a0ea5e11bdfafe056d9d73f4cecd7effe0fc28c197444301274ce78d34.jpg)  
(a) Expert energy-management policy

![](images/de55e0e37adcbf5a15794863d534f228344431296fd1f87fef977ade979a17ba.jpg)  
(b) ARPG-generated energy-management policy  
Figure 7: Comparison of daily patterns with the expert-designed DQN policy for the energy management task.

Figure 7 shows the energy-management experiment. ARPG formulates the task as a cost-minimization MDP over demand, renewable generation, battery state, and time-of-use prices. The ARPG-generated policy follows a scheduling pattern similar to the expert-designed policy: it uses renewable energy when available, charges or discharges the battery according to price and demand conditions, and reduces grid purchases during expensive periods while satisfying demand. This behavior indicates that the generated state, reward, and constraint artifacts capture the intended trade-off among demand satisfaction, storage usage, and electricity cost.

Table 15: Monthly and annual electricity costs for the expert and ARPG-generated energy management policies.
<table><tr><td>Policy</td><td>Jan</td><td>Feb</td><td>Mar</td><td>Apr</td><td>May</td><td>Jun</td><td>Jul</td><td>Aug</td><td>Sep</td><td>Oct</td><td>Nov</td><td>Dec</td><td>Total</td></tr><tr><td>Expert</td><td>13.65</td><td>11.91</td><td>9.50</td><td>9.27</td><td>7.99</td><td>10.27</td><td>19.27</td><td>20.69</td><td>11.35</td><td>9.27</td><td>10.59</td><td>13.62</td><td>147.40</td></tr><tr><td>ARPG</td><td>13.63</td><td>11.86</td><td>9.52</td><td>9.15</td><td>8.00</td><td>10.31</td><td>20.05</td><td>21.11</td><td>11.39</td><td>9.27</td><td>10.64</td><td>13.75</td><td>148.72</td></tr></table>

Table 15 reports the monthly and annual electricity costs of the expert-designed and ARPG-generated policies. The ARPG-generated policy obtains an annual cost of 148.72, close to the expert-designed policy cost of 147.40, corresponding to a 0.90% relative gap. This result suggests that ARPG can generate a meaningful MDP formulation for cost-sensitive energy scheduling, a setting whose objectives and constraints differ substantially from standard robot locomotion and manipulation rewards.

## G.5 Summary

Overall, these demonstrations provide supporting evidence that ARPG’s dependency-aware decomposition is not limited to the specific robot embodiments used in the main benchmark. The main quantitative evaluation remains the Isaac Lab experiments in the main paper, while these additional cases illustrate that explicit objective, state, action, reward, and constraint artifacts can be instantiated across substantially different sequential decision-making domains.

## H Representative ARPG Logbook Trace of Humanoid Task

This section provides a compact representative artifact trace for the Humanoid locomotion task to illustrate the intermediate outputs generated by ARPG. The trace is shortened for readability and omits repeated low-level simulator fields. Additional task descriptions, full artifact traces, representative prompt excerpts, per-configuration documents, and implementation details are provided as separate supplementary files.

## Task summary: Humanoid locomotion

The humanoid must move forward along the global +X direction while maintaining upright balance. Each episode lasts 16 seconds and terminates early if the base height falls below 0.6m. The policy outputs joint-level commands under action decimation, and observations include proprioceptive and base-state information such as joint states, base velocity, orientation, and height. The task requires forward progress, heading alignment, upright posture, and stable locomotion under randomized initial conditions.

## Parameters (summary)

• Timing: physics timestep $\Delta t = 1 / 1 2 0 \mathrm { s }$ , action decimation $d = 2 ,$ , control frequency $\approx 6 0 \mathrm { { H z } }$

• Episode: horizon $T = 1 6 \mathrm { s } ;$ early termination if base height $h _ { t } < 0 . 6 \mathrm { m }$

• Task frame: forward direction is global $+ X ;$ upright direction is global $+ Z$

• Robot: actuated humanoid joints with joint-specific actuator gains and limits.

• Environment: flat ground with randomized initial root pose, root velocity, joint positions, and joint velocities.

## Variables (summary)

• Joint state: joint positions $q _ { t }$ and joint velocities $\dot { q } _ { t }$ for all actuated degrees of freedom.

• Action: joint-level command targets $a _ { t }$ applied through the actuator model.

• Base state: base position, orientation, linear velocity, angular velocity, and height $h _ { t }$

• Task variables: forward velocity $\boldsymbol { v } _ { t } ^ { x }$ , heading error $\theta _ { t } ,$ , uprightness $u _ { t } ,$ lateral position $y _ { t }$ , and lateral velocity $v _ { t } ^ { y }$

• Contact variables: optional left/right foot contact indicators and contact forces.

## Objective

Learn a policy π that maximizes forward locomotion while maintaining upright posture, heading alignment, and lateral stability:

$$
\operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T _ { c } - 1 } \left( r _ { t } ^ { \mathrm { f w d } } - r _ { t } ^ { \mathrm { h e a d } } - r _ { t } ^ { \mathrm { u p r } } - r _ { t } ^ { \mathrm { l a t } } - r _ { t } ^ { \mathrm { f a l l } } \right) \right] .
$$

Here, $r _ { t } ^ { \mathrm { f w d } }$ rewards forward velocity, while the penalty terms discourage heading error, nonupright posture, lateral drift, and falling.

## Constraints (summary)

• Timing: actions are held for $d = 2$ physics steps; control horizon is derived from the 16s episode duration.

• Termination: terminate if $t \geq T$ or $h _ { t } < 0 . 6 \mathrm { m }$

• Orientation: base orientation is represented as a unit quaternion; heading and uprightness are derived from the base frame.

• Actuation: normalized joint commands are clipped and mapped to joint-specific actuator targets.

• Contacts: foot contact indicators and forces describe ground interaction when available.

## State

The SAR agent constructs a proprioceptive and base-centric state:

$$
\boldsymbol { s } _ { t } = \left[ \boldsymbol { q } _ { t } , \dot { \boldsymbol { q } } _ { t } , \boldsymbol { v } _ { t } , \omega _ { t } , \theta _ { t } , \boldsymbol { u } _ { t } , \boldsymbol { v } _ { t } ^ { x } , \boldsymbol { y } _ { t } , \boldsymbol { v } _ { t } ^ { y } , \boldsymbol { h } _ { t } , c _ { t } ^ { L } , c _ { t } ^ { R } \right] .
$$

This state contains joint motion, base motion, heading, uprightness, forward progress, lateral deviation, height, and optional foot-contact information.

## Action

The action is a continuous joint-level command vector:

$$
a _ { t } \in [ - 1 , 1 ] ^ { n _ { \mathrm { d o f } } } .
$$

Each action dimension is clipped to $[ - 1 , 1 ]$ , scaled to joint-specific actuator ranges, and held constant over the action-decimation interval.

## Reward

The reward combines forward progress with stability penalties:

$$
r _ { t } = r _ { t } ^ { \mathrm { f w d } } - \lambda _ { \mathrm { h e a d } } r _ { t } ^ { \mathrm { h e a d } } - \lambda _ { \mathrm { u p r } } r _ { t } ^ { \mathrm { u p r } } - \lambda _ { \mathrm { l a t } } r _ { t } ^ { \mathrm { l a t } } - \lambda _ { \mathrm { f a l l } } r _ { t } ^ { \mathrm { f a l l } } .
$$

A representative instantiation uses

$$
r _ { t } ^ { \mathrm { f w d } } = v _ { t } ^ { x } , \quad r _ { t } ^ { \mathrm { h e a d } } = | \theta _ { t } | , \quad r _ { t } ^ { \mathrm { u p r } } = 1 - u _ { t } , \quad r _ { t } ^ { \mathrm { l a t } } = | y _ { t } | + | v _ { t } ^ { y } | , \quad r _ { t } ^ { \mathrm { f a l l } } = \mathbb { I } \{ h _ { t } < 0 . 6 \} .
$$