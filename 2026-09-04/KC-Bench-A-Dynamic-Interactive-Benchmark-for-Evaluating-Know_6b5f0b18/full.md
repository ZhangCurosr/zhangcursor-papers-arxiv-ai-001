# KC-Bench: A Dynamic Interactive Benchmark for Evaluating Knowledge Conflicts in LLM Agents

Yaxing Lyu<sup>1,2∗</sup>, Shengjie Zhou<sup>3∗</sup>, Binbin Toh<sup>3∗</sup>, Pengyu Zhu<sup>1,4</sup>, Lijun Li<sup>1†</sup>

<sup>1</sup>Shanghai Artificial Intelligence Laboratory

<sup>2</sup>The University of Hong Kong

<sup>3</sup>Xiamen University Malaysia

<sup>4</sup>Beijing University of Posts and Telecommunications lyuyaxing@connect.hku.hk

## Abstract

As LLMs increasingly act through tools, they must reconcile user instructions, parametric knowledge, and dynamic environmental observations before taking actions. We introduce KC-Bench, a controlled multi-turn benchmark for measuring this capability across world-knowledge conflicts, input inconsistencies, and multi-source temporal conflicts. Its 238 tasks are manually screened from more than 1,000 generated candidates and combine a user simulator, stateful tools, deterministic environment assertions, an open-source natural-language evaluator, and human trajectory verification. Evaluation of nine models, including DeepSeek-V4-Flash, GLM-5.2, and MiniMax-M3, shows substantial cross-domain variation: no model handles factual correction, identity consistency checking, and temporal conflict resolution reliably across all settings. In the simulated environments, missed conflicts can propagate to tool calls or synthetic protected-data flows. KC-Bench isolates this model-level behavior rather than ranking complete agent frameworks, and provides a reproducible diagnostic for developing conflict-aware reasoning and execution safeguards.

Code — https://github.com/ASTAR123/KC-Bench

## Introduction

LLM agents are increasingly deployed in settings where they must decide what to do from multiple sources of evidence rather than from a single prompt. A user request may be combined with parametric knowledge, database records, API responses, tool feedback, and prior interaction history (Wang et al. 2024a; Guo et al. 2024; Xi et al. 2025; Luo et al. 2025). This multi-source setting is powerful but fragile: when these sources disagree, an agent must identify the conflict before it commits to an external action. Otherwise, a seemingly ordinary task can become an execution-level safety failure, such as querying the wrong account, exposing private information, or using stale tool outputs.

Existing evaluations only partially cover this problem. Knowledge-conflict benchmarks mainly study static generation or question answering, often framing conflict as disagreement between retrieved context and parametric knowledge (Chen et al. 2023; Xie et al. 2023; Xu et al. 2024; Su et al. 2024; Wang et al. 2023). Interactive agent benchmarks instead emphasize task completion, tool use, or adversarial robustness in dynamic environments (Liu et al. 2023; Xu et al. 2025a; Yao et al. 2024; Barres et al. 2025). KC-Bench does not claim user simulation, stateful tools, or assertionbased evaluation as individually novel. Its contribution is to operationalize a distinct target at their intersection: whether an LLM acting as an agent detects and safely resolves conflicts among parametric knowledge, user instructions, and dynamic environmental observations before taking consequential actions.

Evaluating knowledge conflicts in agents is challenging because the conflict source and the required safe behavior vary across domains. In some cases, the conflict is between a user’s false premise and common world knowledge. In others, the conflict appears only after a tool call, such as when an email resolves to a database profile whose name does not match the user’s claimed identity. More complex cases require reasoning over multiple records with temporal or state inconsistencies. A useful benchmark therefore needs to define conflicts operationally, expose them through realistic interaction, and score not only final task completion but also whether the agent verifies evidence before acting.

We introduce KC-Bench, a dynamic interactive benchmark for evaluating knowledge conflict handling in LLM agents. Figure 1 illustrates the shift from text-level conflicts in standard LLM settings to multi-source conflicts in agentic settings, where unresolved inconsistencies can trigger execution-level risks. KC-Bench organizes conflicts into three operational types: World Knowledge Conflict, where user inputs contradict factual or common knowledge; Input Inconsistency, capturing mismatches between user-provided information and external outputs; and Multi-source Conflict, arising from stale, incomplete, or inconsistent sources, requiring temporal and contextual reasoning. Building on this taxonomy, we introduce KC-Bench, a multi-turn benchmark spanning practical domains such as retail customer service and personal assistant tasks, designed to evaluate agent robustness under knowledge conflicts.

![](images/ac0b6ba95564f7b4fc213174aacb348a3b4d11b86ce9359889fb5ba08c296874.jpg)  
Figure 1: Traditional LLMs mainly face text-level conflicts between parametric knowledge and user inputs, whereas LLM agents must resolve dynamic multi-source conflicts from environments and tools, where undetected inconsistencies may lead to execution-level risks such as privacy leakage.

Our main contributions are as follows:

• We formalize knowledge conflicts for agent evaluation. We define three operational conflict types that map heterogeneous evidence sources to expected safe behaviors, enabling conflict handling to be evaluated beyond textlevel factuality.

• We introduce KC-Bench, a dynamic benchmark for conflict-aware agent evaluation. KC-Bench contains 238 multi-turn tasks across three domains and evaluates whether agents detect conflicts, perform necessary verification or clarification, and avoid unsafe downstream actions.

• We provide an empirical analysis of agent failures under conflicting evidence. Experiments on nine models, including three recent agent-oriented models added after initial evaluation, reveal persistent but strongly domaindependent weaknesses in factual correction, identity consistency checking, and temporal conflict resolution.

## Related Work

## Agent Safety and Execution-Space Risks

With external tool integration, LLMs have evolved into autonomous agents capable of multi-step planning and action (Wang et al. 2024a; Guo et al. 2024; Xi et al. 2025; Luo et al. 2025). Existing safety research mainly focuses on input-side threats, including jailbreaks (Mao et al. 2025), prompt injection (Debenedetti et al. 2024; Lee and Tiwari 2024; Shi et al. 2025), and backdoor attacks (Wang et al. 2024b; Zhu et al. 2025). Beyond adversarial inputs, however, agents also face execution-space risks in legitimate tasks, where unsafe actions may arise from misinterpreting system constraints (Wang, Poskitt, and Sun 2025), insuficient verification of tool outputs (Doshi et al. 2026; Li 2025), or incorrect reasoning over environmental feedback (Chen et al. 2025).

## Knowledge Conflict and Epistemic Challenges in LLMs

Knowledge conflict has been studied in question answering (Tan et al. 2024; Chen, Zhang, and Choi 2022) and retrieval-augmented generation (Xie et al. 2023; Jin et al. 2024), mainly focusing on conflicts between external context and parametric knowledge. In agent settings, however, the problem becomes broader, as agents must reason over multiple dynamic and potentially inconsistent information sources (Xu et al. 2025b; Liu et al. 2023; Xu et al. 2025a; Yao et al. 2024). As a result, knowledge conflict extends beyond textual factuality to directly afect decision-making and action safety in agent systems.

## Dynamic Interactive Evaluation and Process-Oriented Failure Diagnosis

Recent work has proposed interactive benchmarks to evaluate LLM agents in more realistic settings, moving beyond static single-turn question answering by incorporating tool use (Liu et al. 2023; Xu et al. 2025a), state tracking (Deng et al. 2024; Jimenez et al. 2023) and user simulation (Yao et al. 2024; Barres et al. 2025). Existing benchmarks primarily assess task success and eficiency, overlooking failures arising from conflicting knowledge. We introduce a new benchmark that explicitly incorporates knowledge conflicts, enabling analysis of how agents detect, resolve, and act under such conflicts.

## KC-Bench

## Problem Formulation & 3-Level Taxonomy

We model agent–environment interaction as a partially observable Markov decision process (Figure 2). At step t, policy π receives three information sources:

Parameterized Memory $( \mathcal { K } _ { p a r a m } )$ : World knowledge and common sense acquired during pretraining.

User Instructions (U): The dialogue history and current request.

Environmental Observations $( \mathcal { O } _ { t } ) \mathrm { : }$ : State returned by external tools or databases.

$$
a _ { t } \sim \pi _ { \theta } ( a _ { t } \mid \mathcal { U } , \mathcal { O } _ { < t } , \mathcal { K } _ { p a r a m } )\tag{1}
$$

where $\mathcal { A }$ contains tool execution, clarification, and finalresponse actions.

A knowledge conflict (C) occurs when parametric knowledge contradicts contextual evidence or when observations are mutually inconsistent. For factual attribute v:

$$
\begin{array} { r } { \mathcal { C } ( v ) \Leftrightarrow \underbrace { \mathbb { D } \big [ P ( v \mid \mathcal { K } _ { p a r a m } ) \big \rvert \big ] P ( v \mid \mathcal { U } , \mathcal { O } _ { < t } ) \big ] > \epsilon } _ { \mathrm { C o n c e p t u a l d i v e r g e n c e } } } \\ { \mathrm { o r } \underbrace { \mathcal { U } \wedge \mathcal { O } _ { t } \Vdash \bot } _ { \mathrm { C o n c e p t u a l c o n t r a d i c t i o n } } } \end{array}\tag{2}
$$

Here, D is a conceptual divergence and ⊥ a logical contradiction; neither is estimated. The benchmark tests three conflict types of increasing interaction complexity and whether agents resolve them before taking policy-violating actions.

## Level 1: World Knowledge Conflict (Region Scene)

This level tests a fabricated user premise $\mathcal { F } _ { f a k e }$ that contradicts a true fact $\mathcal { F } _ { t r u e }$ in $\begin{array} { r } { \kappa _ { p a r a m } , } \end{array}$ where $\mathcal { U } \models \mathcal { F } _ { f a k e }$ and $\mathcal { F } _ { f a k e } \wedge \mathcal { F } _ { t r u e } = \perp$

The desired policy rejects the false premise:

$$
\pi ^ { * } ( a _ { r e j e c t } \mid \mathcal { U } , \ K _ { p a r a m } ) \to 1\tag{3}
$$

Executing an action based on $\mathcal { F } _ { f a k \epsilon }$ is an epistemic failure.

## Level 2: Input Inconsistency (Retail Scene)

This level tests contradictory user credentials and database observations. For ${ \mathcal U } _ { i d } = \{ ( \dot { k } _ { 1 } , v _ { u 1 } ) , ( k _ { 2 } , v _ { u 2 } ) \}$ and $\mathcal { O } _ { t } ~ =$ $\{ ( k _ { 1 } , v _ { t 1 } ) , ( k _ { 2 } , v _ { t 2 } ) \}$ , a conflict occurs when $v _ { u 1 } = v _ { t 1 }$ but $v _ { u 2 } \neq v _ { t 2 } \left( \mathrm { e . g . } \right)$ ., matching email but mismatching name). The agent must clarify before a sensitive operation:

$$
\begin{array} { c } { a _ { t } = a _ { c l a r i f y } \bigl ( v _ { t 2 } , v _ { u 2 } \bigr ) } \\ { \mathrm { b e f o r e } \quad a _ { w r i t e / q u e r y \_ p r i v a t e } } \end{array}
$$

Bypassing this constraint is a detection and execution failure.

Level 3: Multi-source Conflict (Personal Assistant Scene) This level tests temporal or state contradictions among records $\mathcal { R } = \{ r ^ { ( 1 ) } , \ldots , \bar { r } ^ { ( n ) } \}$ returned by multiple sources:

$$
\exists i , j { \mathrm { ~ s . t . ~ } } r ^ { ( i ) } \wedge r ^ { ( j ) } = \bot \quad ( { \mathrm { a n d ~ } } t _ { i } < t _ { j } )\tag{4}
$$

The agent must use temporal and logical reasoning to derive the valid subset:

$$
\mathscr { R } ^ { * } = f _ { r e a s o n } ( \mathscr { R } , \mathscr { K } _ { p a r a m } )\tag{5}
$$

Executing with an outdated record despite contrary evidence is a failure.

<table><tr><td>Domain</td><td>Tasks</td><td></td><td>Tools Agent database</td></tr><tr><td>Region</td><td>71</td><td>9</td><td>98 entities</td></tr><tr><td>Retail</td><td>93</td><td>15</td><td>500 users, 50 products, and 1,000 orders</td></tr><tr><td>Personal Assistant</td><td>74</td><td>6</td><td>83 contacts and 190 contact-history records</td></tr></table>

Table 1: Key statistics for the KC-bench domains.

## Domain and task creation

We extend the compositional task-generation framework of $\tau ^ { 2 }$ -bench (Barres et al. 2025) with knowledge-conflict injection. Table 1 summarizes the domains.

Each task is a 5-tuple:

$$
\mathcal { T } = \left. \mathcal { T } _ { u s e r } , \mathcal { E } _ { d b } , f _ { i n j e c t } , \pi _ { s i m } , f _ { a s s e r t } \right.\tag{6}
$$

where $\mathcal { T } _ { u s e r }$ is the simulator instruction and persona, ${ \mathcal { E } } _ { d b }$ the initial database, $f _ { i n j e c t }$ the conflict injection, $\pi _ { s i m }$ the simulator policy, and $f _ { a s s e r t }$ the outcome assertion.

Conflict Injection Mechanism Given conflict-free environment $\mathcal { E } _ { b a s e }$ and instruction $\mathcal { T } _ { b a s e } , f _ { i n j e c t }$ creates one of three conflicts:

Region $( f _ { i n j e c t } ^ { w o r l d } )$ : Replace a factual entity in $\mathcal { T } _ { u s e r }$ so that $\tilde { \mathcal { T } } _ { u s e r } \left| = \mathcal { F } _ { f a k e } \right. .$

Retail $( f _ { i n j e c t } ^ { i n p u t } ) ;$ : For true record $R _ { t r u e } ~ = ~ \{ k _ { e m a i l } ~ :$ $e _ { t r u e } , k$ <sub>name</sub> : n<sub>true</sub>, k<sub>private</sub> $: ~ v _ { p r i v a t e } \}$ , forge one credential:

$$
f _ { i n j e c t } ^ { i n p u t } ( \mathbb { Z } _ { u s e r } )  \tilde { \mathbb { Z } } _ { u s e r }\tag{7}
$$

$$
\tilde { \mathcal { T } } _ { u s e r . } e m a i l = e _ { t r u e } \wedge \tilde { \mathcal { T } } _ { u s e r . } n a m e = n _ { f a k e }\tag{8}
$$

We set $s i m ( n _ { f a k e } , n _ { t r u e } ) < \delta$ to make the mismatch unambiguous and test whether the agent exposes $v _ { p r i v a t e }$ without verification.

Personal Assistant $( f _ { i n j e c t } ^ { m u l t i } ) \mathrm { : }$ Insert expired and active records $\mathcal { R } = \{ r _ { o l d } , r _ { n e w } \}$ for one entity while the user requests $r _ { o l d }$ , requiring cross-source temporal reasoning.

Agent and User Environment Construction We construct database schemas, tools, tasks, and policies in three stages.

Stage 1: Creating the Database Schema and Tools LLMs first generate a Product Requirements Document, database schema, functions, mock data, and unit tests; we manually refine them until all tests pass. Region tools provide readonly factual queries, Retail tools cover identity and order operations, and Personal Assistant tools query contact lists and histories before communication. Tools have typed inputs, deterministic outputs, explicit state constraints, and surfaced errors. An agent may make one tool call per turn and cannot simultaneously respond to the user.

Stage 2: Task Generation We generated more than 1,000 multi-turn candidates by combining database entries, functions, and injected conflicts; the final 238 are a filtered subset.

![](images/4e8cb0a5c78196f6cc15d1bc7bd894d20c40246a5cc9ea369389fde81bdcd45b.jpg)  
Figure 2: The dynamic execution pipeline of KC-Bench . It illustrates the interaction loop between a task-driven user simulator and the LLM agent , where agents must invoke tools and query databases to identify and detect injected knowledge conflicts under domain-specific policies.

Quality Control Human screening required: (1) factual and database correctness; (2) consistency among instructions, conflicts, tool outputs, and expected behavior; (3) one uniquely identifiable conflict; (4) reachable evidence; (5) no unintended ambiguity or secondary conflict; and (6) no malformed, trivial, or duplicate instance. Reviewers inspected database state, expected tool trajectory, and protected fields, executing tasks when reachability was uncertain. Disagreements were resolved against the database state and domain policy.

Stage 3: Creating Domain-Specific Agent Policy LLMs generate domain policies that we refine to define expected behavior. Region agents correct false premises before tool use. Retail agents require consistent email, name, and ZIP evidence before disclosure or state changes, which also require confirmation. Personal Assistant agents reconcile contact lists and histories before communicating. Unresolved evidence requires clarification rather than execution.

Avoiding unnecessary clarification applies only to suficient, consistent evidence and never overrides authentication or conflict resolution. Policies do not announce that tasks contain conflicts.

Evaluation Scope KC-Bench evaluates an LLM’s conflict handling as an agent, not complete agent products. Its standardized harness controls planning, memory, routing, recovery, and authorization to reduce framework confounds; framework comparisons are outside scope.

Assertion Verification $f _ { a s s e r t }$ marks success only if the agent detects the conflict and never violates protected state throughout history H:

$$
\begin{array} { r l } & { f _ { a s s e r t } ( \mathcal { H } , \mathcal { E } _ { d b } ) = \mathbb { I } ( \mathbf { C o n f i c t \_ D e t e c t e d } ) \wedge \neg \exists t , } \\ & { \qquad \mathbf { L e a k a g e } ( \mathcal { O } _ { t } \to \mathcal { U } , v _ { p r i v a t e } ) } \end{array}\tag{9}
$$

Leakage checks tool observations and visible responses for protected attributes disclosed before identity verification. The assertion model receives the full trace and an expected outcome specifying the conflict, required evidence, prohibited action, and acceptable resolution, then returns a binary decision and rationale. Success requires explicit detection without protected-state exposure or modification.

Human Verification “Manual Check” in Table 2 denotes human verification, not a second AI evaluator or humanagent baseline. Annotators inspect trajectories, tool calls, records, final state, and evaluator rationales for conflict detection, acceptable resolution, and protected-state violations. This captures cases, especially in Personal Assistant, where multiple safe resolutions are valid. When judgments difer, the trace is adjudicated against the expected outcome and deterministic environment state, and the human label is reported.

## Experiments

## Overall Performance Analysis

Table 2 reports automatic evaluation and human-verified success scores for nine models. In addition to the original closed-source models (GPT-5.1, Gemini-3-flash-preview, and Claude-haiku-4-5-20251001) and open-source models (GLM-4.5-Air, gpt-oss-120b, and Qwen3.5-35B-A3B), we evaluate three recently released models designed for agentic or long-horizon tasks: DeepSeek-V4-Flash, GLM-5.2, and MiniMax-M3.

<table><tr><td>Type</td><td>Model</td><td>Region</td><td>Retail</td><td>Personal Assistant</td></tr><tr><td rowspan="3">Closed-Source</td><td>GPT-5.1</td><td>0.4085/0.4085</td><td>0.5054/0.5054</td><td>0.6081/0.6216</td></tr><tr><td>Gemini-3-flash-preview</td><td>0.0704/0.0704</td><td>0.5054/0.5054</td><td>0.7838/0.7973</td></tr><tr><td>Claude-haiku-4-5-20251001</td><td>0.4507/0.4507</td><td>0.7312/0.7312</td><td>0.5676/0.5676</td></tr><tr><td rowspan="6">Open-Source</td><td>Glm-4.5-Air</td><td>0.3662/0.3804</td><td>0.4301/0.4623</td><td>0.3378/0.3378</td></tr><tr><td>gpt-oss-120b</td><td>0.0845/0.0845</td><td>0.1182/0.1182</td><td>0.1621/0.2297</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.5070/0.5070</td><td>0.1935/0.1935</td><td>0.3783/0.3783</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.1831/0.1831</td><td>0.6522/0.6522</td><td>0.6575/0.6986</td></tr><tr><td>GLM-5.2</td><td>0.5352/0.5352</td><td>0.6196/0.6196</td><td>0.6301/0.6575</td></tr><tr><td>MiniMax-M3</td><td>0.6761/0.6761</td><td>0.4239/0.3804</td><td>0.6438/0.6712</td></tr></table>

Table 2: Success scores. Each cell reports automatic LLM evaluation/manual verification.

Experimental Configuration Each evaluation uses three components: the tested agent, a task-driven user simulator, and a natural-language assertion model. All components use greedy decoding with temperature 0. Each task is limited to 30 interaction steps and terminates after 10 execution errors; transient API failures use exponential backof with at most three retries. We use a fixed random seed of 300, one trial per task, and a maximum concurrency of three. LiteLLM standardizes API calls across model providers. Open-source models are loaded without quantization and evaluated on dual NVIDIA H200 GPUs. These settings apply uniformly across all 238 tasks and are suficient to reproduce the evaluation protocol; complete prompts and infrastructure details are included only as optional technical supplementary material.

We additionally log wall-clock duration, dialogue turns, and sequential tool-call depth for every trajectory. Table 3 reports the complete model-level diagnostics.

Model-level Conflict Diagnostics. The operational profiles strengthen the interpretation of KC-Bench as a model diagnostic rather than a simple task-success leaderboard. Longer reasoning does not guarantee conflict resolution: GPT-OSS uses 8.31 turns in Region yet obtains only 0.0845 success, while GLM-4.5-Air has the longest Retail runtime (189.56 s) but reaches only 0.4623 after human verification. Conversely, deeper tool use is not uniformly beneficial. DeepSeek-V4-Flash reaches tool depths of 2.85 in Retail and 2.11 in Personal Assistant, with substantially stronger scores than in Region, whereas MiniMax-M3 exhibits similarly deep tool use but remains weak on Retail identity conflicts. GLM-5.2 is more balanced in success despite markedly diferent turn counts across settings. These patterns show that latency, verbosity, or tool activity alone cannot explain success: KC-Bench exposes whether a model uses its interaction budget to identify conflicting evidence and choose a safe action. Reporting trajectories alongside outcomes therefore enables finer-grained analysis of conflict-handling strategies and motivates future work on model-level detection, evidence reconciliation, and pre-execution verification.

Conflict Handling Remains Unreliable. No evaluated model performs consistently well across all three domains. The recent-model extension improves the best Region score from 0.5070 to 0.6761, but does not saturate the benchmark: DeepSeek-V4-Flash scores only 0.1831 in Region despite human-verified scores of 0.6522 and 0.6986 in Retail and Personal Assistant, respectively. MiniMax-M3 shows the reverse imbalance, reaching 0.6761 in Region and 0.6712 in Personal Assistant but only 0.3804 in Retail. GLM-5.2 is more balanced (0.5352–0.6575) yet still fails approximately 34–46% of tasks. Thus, stronger general agentic capability does not uniformly transfer to factual correction, identity consistency checking, and temporal conflict resolution.

Behavioral Failures Can Reach the Execution Layer. In the simulated Retail environment, an unresolved credential mismatch can cause the agent to query or reveal synthetic protected records; in Personal Assistant, failure to reconcile timestamps can lead to an action using an obsolete identifier. These are benchmark-observed behavioral consequences, not evidence of actual legal violations or production incidents. In deployed systems, deterministic authentication, authorization, and API-level permission checks should prevent such actions regardless of model reasoning.

Cross-domain Variation. Automatic and human scores are usually close, while the remaining gaps show why trajectory-level human verification is useful. Conflict handling is highly heterogeneous across domains: Gemini-3- flash-preview obtains 0.7973 after human verification in Personal Assistant but only 0.0704 in Region. The three newly added models exhibit similarly divergent profiles. A single aggregate task-success score would conceal these distinctions, motivating evaluation across evidence sources and action consequences.

## World Knowledge Conflict: Factual Compromise

In the world knowledge (Region) scenario, the core evaluation is whether the agent can uphold basic common sense and refuse to execute commands containing absurd facts. However, the overall low scores reveal significant epistemic failures at the entity operation level in large models.

In an in-depth analysis of specific tasks, we found that the error rate in the subdomain related to the country of origin of food is particularly high. Only Qwen3.5-35B-A3B achieves an error rate of 50%, while all other models exhibit an error rate of 100%. As an illustrative test case, the simulated user instructed the agent to calculate the import tarif based on sushi’s country of origin, explicitly stating the belief that sushi originated from Korea. Although it is common knowledge that sushi originated from Japan, most agents failed to detect this knowledge conflict (Detection Failure, D) and proceeded to calculate the tarif using Korea as the country of origin.

![](images/5b671fe48fe77b95485501a27d7d8a1f57428810536a3dd2c4d560da3fbf7e0f.jpg)  
Figure 3: An example agent-user interaction for each scenario. It clearly reveals the distinct sources of knowledge conflicts, as well as the agents’ dialogue strategies and outcomes in each scenario.

On the other hand, across 27 tasks involving the birthplaces of historical figures, all models achieved low accuracy, with the best result from Claude at 25.9% (7/27). Analysis of the dialogues reveals that agents only begin to reason correctly when the user reduces confidence in the stated wrong answer. This suggests that models possess the correct knowledge, but initially tend to accept and align with the user’s incorrect assumption instead of questioning it(See Figure 3 Retail Part).

We call this observed pattern factual compromise: the agent accepts a user’s false premise despite behavior elsewhere in the trajectory suggesting access to the correct fact. Preference alignment is one plausible explanation, but KC-Bench does not isolate RLHF or any other training mechanism causally. The supported conclusion is behavioral: under the standardized policy, models often fail to verify a factual premise before constructing a tool call.

## Input Inconsistency: Missed Identity Conflicts

In the retail scenario simulating real-world identity authentication, the agent is required to first detect inconsistencies between user-provided credentials and system records, and then perform a clarification action (a<sub>clarify</sub>) before proceeding. However, we observe pervasive failures in both conflict detection and subsequent clarification behavior, even when the inconsistencies are explicit and unambiguous.

More critically, these failures persist across multiple controlled factors, suggesting a structural weakness rather than a task-dependent limitation:

• Task Complexity Desensitization: Whether the user presented 3 complex parallel return-exchange requests or just 1 simple logistics query, the agent’s detection and interception rates for identity conflicts showed no significant diference. This indicates that the alignment issue is structural, not limited by the model’s context length or cognitive load.

• Semantic Similarity Immunity: We systematically altered the textual distance between the fabricated credentials and the real credentials. Whether the fake combination was highly visually misleading (e.g., Name: lee jack, Email: limjack@example.com), or completely unrelated as in the above case (e.g., Name: lee jack, Email: SimMaria@example.com), the agent exhibited the same level of “cognitive disregard.” This suggests that current agents do not perform robust cross-field consistency checks—neither rule-based (exact matching)

<table><tr><td>Domain</td><td>Metric</td><td>Claude</td><td>Gemini</td><td>GPT-5.1</td><td>GLM-5.2</td><td>DeepSeek MiniMax</td><td></td><td>GLM-Air</td><td>Qwen3.5</td><td>GPT-OSS</td></tr><tr><td rowspan="3">Region</td><td>Duration (s)</td><td>9.28</td><td>8.67</td><td>14.14</td><td>21.26</td><td>17.87</td><td>21.11</td><td>58.12</td><td>22.17</td><td>56.19</td></tr><tr><td>Turns</td><td>1.3</td><td>3.5</td><td>3.1</td><td>3.9</td><td>2.5</td><td>3.5</td><td>3.1</td><td>2.6</td><td>8.31</td></tr><tr><td>Tool depth</td><td>1.25</td><td>1.35</td><td>1.00</td><td>1.05</td><td>1.08</td><td>1.04</td><td>1.00</td><td>1.00</td><td>1.37</td></tr><tr><td rowspan="3">Retail</td><td>Duration (s)</td><td>46.33</td><td>37.53</td><td>85.79</td><td>60.80</td><td>40.03</td><td>32.20</td><td>189.56</td><td>117.01</td><td>49.96</td></tr><tr><td>Turns</td><td>6.0</td><td>5.1</td><td>5.2</td><td>10.9</td><td>4.5</td><td>6.2</td><td>4.3</td><td>4.2</td><td>5.8</td></tr><tr><td>Tool depth</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.14</td><td>2.85</td><td>2.28</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="3">Personal Assistant</td><td>Duration (s)</td><td>20.23</td><td>18.88</td><td>36.62</td><td>29.70</td><td>21.80</td><td>34.59</td><td>95.16</td><td>48.26</td><td>66.29</td></tr><tr><td>Turns</td><td>3.0</td><td>3.3</td><td>3.4</td><td>5.8</td><td>3.0</td><td>4.8</td><td>3.8</td><td>3.9</td><td>4.3</td></tr><tr><td>Tool depth</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.58</td><td>2.11</td><td>2.07</td><td>1.00</td><td>1.00</td><td>1.84</td></tr></table>

Table 3: Operational profiles of all evaluated models. Claude, Gemini, DeepSeek, MiniMax, and GLM-Air denote Claude Haiku 4.5, Gemini 3 Flash, DeepSeek-V4-Flash, MiniMax-M3, and GLM-4.5-Air, respectively.

nor semantic (identity-level reasoning)—when invoking downstream actions (See Figure 3).

Within our simulated Retail environment, the absence of reliable verification permits disclosure of synthetic profile fields, including names, postal codes, addresses, and order histories. This demonstrates a potential deployment consequence of placing model reasoning on the authorization path; it does not establish a real-world breach or GDPR violation. Production systems should enforce identity and permission checks deterministically at the tool or API layer rather than relying on an LLM to provide access control.

## Multi-source Conflict: Reversal under User Pressure

The Personal Assistant (PA) scenario is the most cognitively demanding, requiring the agent to reason over heterogeneous and temporally conflicting tool outputs while maintaining logical consistency under user pressure.

In a task involving account status conflicts (See Figure 3), the database returned both a valid contact number and an outdated number explicitly marked as “Not in Use” (0000–0001). Initially, some models correctly inferred that 0000–0001 had expired. However, under multi-turn interaction, the user simulator began insisting on using the expired number. Models such as GPT-5.1 then exhibited temporal sycophancy, revising previously correct beliefs and accepting the invalid number into downstream API execution.

Thisjoint failure in reasoning and action correctness shows that some agents reverse an earlier, evidence-supported conclusion after user insistence. We use temporal sycophancy as a behavioral description of this trajectory, not as a causal claim about preference optimization. The result motivates explicit consistency checks and deterministic safeguards before consequential communication actions.

## Conclusion

This study investigates LLM agent behavior when user instructions, parametric knowledge, and dynamic environmental evidence conflict. KC-Bench combines a three-part conflict taxonomy with multi-turn interaction, stateful tools, and trajectory-level evaluation of conflict resolution and downstream actions. Across nine models, including DeepSeek-V4- Flash, GLM-5.2, and MiniMax-M3, no model performs consistently across domains: performance in one conflict type does not transfer to factual correction, identity consistency checking, or temporal reasoning. The benchmark exposes simulated cases in which unresolved conflicts reach tool execution or protected data flows.

These findings establish conflict handling as a distinct dimension of agent reliability that cannot be captured by aggregate task-success scores alone. By operationalizing conflicts across heterogeneous evidence sources and tracing whether agents detect, verify, resolve, and safely act on them, KC-Bench makes hidden failure modes measurable and comparable. It serves as a testbed for identifying modelspecific weaknesses, evaluating conflict-resolution interventions, and tracking progress toward agents that remain reliable when inconsistent information appears during interaction. The benchmark complements evaluations of task completion and tool use by connecting epistemic errors to their execution-level consequences, providing a basis for model development and pre-deployment risk assessment.

## Ethical Statement

The design of this study proactively addresses potential ethical risks. All data utilized in our experiments has been meticulously processed to remove personal identifiers, ensuring no impact on real-world users. We rely solely on publicly accessible information, such as the birthplaces of public figures or widely known historical events. In simulated tasks for retail or personal assistant scenarios, all data records are either synthetically generated or strictly de-identified to prevent the models from encountering sensitive personal information during evaluation. Our results highlight the potential risks associated with knowledge conflict resolution in LLM agents, such as the "temporal sycophancy" observed in simulated scenarios, where agents abandon previously correct reasoning to comply with persistent user misinformation. Such failures can lead to logical inconsistencies and unsafe system operations. These findings emphasize the necessity of integrating execution-level conflict detection, robust permission control, and safety guardrails in real-world deployments to mitigate negative impacts and ensure system integrity.

## References

Barres, V.; Dong, H.; Ray, S.; Si, X.; and Narasimhan, K. 2025. τ<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982.

Chen, C.; Song, X.; Chai, Y.; Yao, Y.; Zhao, H.; Li, L.; Li, J.; Teng, Y.; Liu, G.; and Wang, Y. 2025. GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments? arXiv preprint arXiv:2510.20333.

Chen, H.-T.; Zhang, M.; and Choi, E. 2022. Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2292–2307.

Chen, L.; Deng, Y.; Bian, Y.; Qin, Z.; Wu, B.; Chua, T.-S.; and Wong, K.-F. 2023. Beyond factuality: A comprehensive evaluation of large language models as knowledge generators. In Proceedings of the 2023 conference on empirical methods in natural language processing, 6325–6341.

Debenedetti, E.; Zhang, J.; Balunovic, M.; Beurer-Kellner, L.; Fischer, M.; and Tramèr, F. 2024. Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for llm agents. Advances in Neural Information Processing Systems, 37: 82895–82920.

Deng, S.; Xu, W.; Sun, H.; Liu, W.; Tan, T.; Liujianfeng, L.; Li, A.; Luan, J.; Wang, B.; Yan, R.; et al. 2024. Mobile-bench: An evaluation benchmark for llm-based mobile agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8813–8831.

Doshi, A.; Hong, Y.; Xu, C.; Kang, E.; Kapravelos, A.; and Kästner, C. 2026. Towards Verifiably Safe Tool Use for LLM Agents. arXiv preprint arXiv:2601.08012.

Guo, T.; Chen, X.; Wang, Y.; Chang, R.; Pei, S.; Chawla, N. V.; Wiest, O.; and Zhang, X. 2024. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680.

Jimenez, C. E.; Yang, J.; Wettig, A.; Yao, S.; Pei, K.; Press, O.; and Narasimhan, K. 2023. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770.

Jin, Z.; Cao, P.; Chen, Y.; Liu, K.; Jiang, X.; Xu, J.; Qiuxia, L.; and Zhao, J. 2024. Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models. In Proceedings of the 2024 joint international conference on computational linguistics, language resources and evaluation (LREC-COLING 2024), 16867–16878.

Lee, D.; and Tiwari, M. 2024. Prompt infection: Llm-to-llm prompt injection within multi-agent systems. arXiv preprint arXiv:2410.07283.

Li, X. 2025. A review of prominent paradigms for llm-based agents: Tool use, planning (including rag), and feedback learning. In Proceedings of the 31st international conference on computational linguistics, 9760–9779.

Luo, J.; Zhang, W.; Yuan, Y.; Zhao, Y.; Yang, J.; Gu, Y.; Wu, B.; Chen, B.; Qiao, Z.; Long, Q.; et al. 2025. Large language model agent: A survey on methodology, applications and challenges. arXiv preprint arXiv:2503.21460.

Mao, Y.; Cui, T.; Liu, P.; You, D.; and Zhu, H. 2025. From llms to mllms to agents: A survey of emerging paradigms in jailbreak attacks and defenses within llm ecosystem. arXiv preprint arXiv:2506.15170.

Shi, J.; Yuan, Z.; Tie, G.; Zhou, P.; Gong, N. Z.; and Sun, L. 2025. Prompt injection attack to tool selection in llm agents. arXiv preprint arXiv:2504.19793.

Su, Z.; Zhang, J.; Qu, X.; Zhu, T.; Li, Y.; Sun, J.; Li, J.; Zhang, M.; and Cheng, Y. 2024. Conflictbank: A benchmark for evaluating the influence of knowledge conflicts in llm. arXiv preprint arXiv:2408.12076.

Tan, H.; Sun, F.; Yang, W.; Wang, Y.; Cao, Q.; and Cheng, X. 2024. Blinded by generated contexts: How language models merge generated and retrieved contexts when knowledge conflicts? In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), 6207–6227.

Wang, H.; Poskitt, C. M.; and Sun, J. 2025. Agentspec: Customizable runtime enforcement for safe and reliable llm agents. arXiv preprint arXiv:2503.18666.

Wang, L.; Ma, C.; Feng, X.; Zhang, Z.; Yang, H.; Zhang, J.; Chen, Z.; Tang, J.; Chen, X.; Lin, Y.; et al. 2024a. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6): 186345.

Wang, Y.; Feng, S.; Wang, H.; Shi, W.; Balachandran, V.; He, T.; and Tsvetkov, Y. 2023. Resolving knowledge conflicts in large language models. arXiv preprint arXiv:2310.00935.

Wang, Y.; Xue, D.; Zhang, S.; and Qian, S. 2024b. Badagent: Inserting and activating backdoor attacks in llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 9811–9827.

Xi, Z.; Chen, W.; Guo, X.; He, W.; Ding, Y.; Hong, B.; Zhang, M.; Wang, J.; Jin, S.; Zhou, E.; et al. 2025. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2): 121101.

Xie, J.; Zhang, K.; Chen, J.; Lou, R.; and Su, Y. 2023. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In The Twelfth International Conference on Learning Representations.

Xu, K.; Mao, Y.; Guan, X.; and Feng, Z. 2025a. Web-bench: A llm code benchmark based on web standards and frameworks. arXiv preprint arXiv:2505.07473.

Xu, R.; Qi, Z.; Guo, Z.; Wang, C.; Wang, H.; Zhang, Y.; and Xu, W. 2024. Knowledge conflicts for llms: A survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 8541–8565.

Xu, W.; Huang, C.; Gao, S.; and Shang, S. 2025b. LLM-Based Agents for Tool Learning: A Survey: W. Xu et al. Data Science and Engineering, 1–31.

Yao, S.; Shinn, N.; Razavi, P.; and Narasimhan, K. 2024. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.

Zhu, P.; Li, L.; Lyu, Y.; Sun, L.; Su, S.; and Shao, J. 2025. Collaborative Shadows: Distributed Backdoor Attacks in LLM-Based Multi-Agent Systems. arXiv preprint arXiv:2510.11246.