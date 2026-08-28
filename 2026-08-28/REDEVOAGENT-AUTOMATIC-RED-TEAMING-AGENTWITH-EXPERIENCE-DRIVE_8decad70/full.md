# REDEVOAGENT: AUTOMATIC RED-TEAMING AGENTWITH EXPERIENCE-DRIVEN SKILL EVOLUTION

Junjie Zhang<sup>1</sup> Hui Liu<sup>1</sup> Kecheng Chen<sup>1</sup> Xianbo Mo<sup>2</sup> Changsheng Chen<sup>2</sup> Haoliang Li<sup>1</sup>

<sup>1</sup> City University of Hong Kong <sup>2</sup> Shenzhen MSU-BIT University

## ABSTRACT

LLM-based agents are increasingly deployed in product-level execution harnesses, where jailbreaks can trigger harmful tool use and persistent state changes, creating greater risks than unsafe text generation alone. Existing automatic redteaming methods often rely on fixed attacks, while recent agentic attackers coordinate multiple jailbreak tools and show stronger potential through trajectory-based retrieval. However, such retrieval can reuse misleading experiences due to retrieval bias and unclear tool credit, and full trajectories add context overhead while reducing interpretability. We propose RedEvoAgent, a black-box red-teaming agent that distills cross-case attack trajectories into a concise, human-readable attack skill. The attack skill adaptively evolves through tool-effectiveness profiling and Deciding-Tool Attribution for skill updates, and a validation ratchet that retains only updates improving validation performance. Experiments on multiple benchmarks, target models, and target execution harnesses show that RedEvoAgent outperforms fixed and agentic baselines, improves tool efficiency, and transfers across attacker models and target execution harnesses.

## 1 INTRODUCTION

LLM-based agents are increasingly integrated into product-level harnesses such as Claude Code (Anthropic, 2026a) and Codex (OpenAI, 2025), which allow them to modify local files, transfer data, and call external APIs (Yao et al., 2023; Yang et al., 2024b). These capabilities broaden the security implications of jailbreaks, as successful attacks may cause destructive tool use and persistent changes to system state rather than unsafe text generation (Greshake et al., 2023; Ruan et al., 2024; Andriushchenko et al., 2025). Therefore, product-level black-box agents require adversarial evaluation to characterize their security robustness. Automatic red-teaming provides a scalable approach for continuous evaluation by simulating real-world malicious attacks, adversarial inputs, and exploit attempts (Perez et al., 2022; Ganguli et al., 2022; Mazeika et al., 2024).

Most existing jailbreak methods rely on a fixed attack mechanism. Some transform or rewrite the original attack prompt to circumvent an agent’s safety-alignment defenses, such as FlipAttack (Liu et al., 2025b), whereas others optimize adversarial prompts within a predefined search space, as in GCG (Zou et al., 2023) and AutoDAN (Liu et al., 2024). Because such methods explore only a limited subset of the attack space, they may miss vulnerabilities exposed by alternative mechanisms and are more likely to fail. Recent work, therefore, investigates agentic automatic red-teaming, in which an LLM-driven attacker agent leverages prior knowledge about existing attack methods to adaptively coordinate them within a unified workflow. For example, RedCodeAgent (Guo et al., 2026) models attack methods as sequentially invocable tools and utilizes semantic similarity to retrieve successful trajectories as priors to guide the attacker agent’s subsequent decisions.

Despite its effectiveness, trajectory-based retrieval has several limitations. First, this paradigm assumes that all retrieved trajectories are useful, which may not hold because semantic retrieval can introduce bias, and the contribution of any individual tool-choice decision to final attack success is often unclear. Consequently, irrelevant or misleading experiences may be reused, causing unstable optimization and degraded attack performance. Second, full-trajectory contexts introduce substantial overhead by consuming context-window budget and increasing inference cost. Their low-level action records are also difficult for humans to interpret and monitor, hindering auditing over the evolution of strategy.

![](images/d5d15eb33b6fc8083efbff2128f1f342318eef0e8d335fa5513367aad6c05645.jpg)  
Figure 1: RedEvoAgent overview. Given a jailbreak prompt, the attacker agent uses the current attack skill to select and call tools from an extensible attack toolbox, such as GCG and FlipAttack, and iteratively attacks a product-level target agent. On the training split, RedEvoAgent collects successful and failed attack trajectories and independently evaluates each attack tool to build a tooleffectiveness profile. Skill evolution combines the profile with these trajectories to extract tool priorities and attack strategies. A candidate attack skill update is accepted only if it improves attack performance on the validation split, while the test split remains locked for final evaluation.

To address these gaps, as illustrated in Figure 1, we propose RedEvoAgent, an automatic blackbox red-teaming agent with experience-driven attack-skill evolution. RedEvoAgent integrates di verse jailbreak methods into an attack toolbox and distills attack trajectories into a concise, humanreadable attack skill that summarizes high-level orchestration strategies. Compared with trajectorybased historical priors, such a skill-level abstraction reduces context overhead and improves interpretability, consistent with evidence from general agent tasks that high-level skill summaries can outperform similarity-based trajectory retrieval (Ni et al., 2026; Yang et al., 2026). Concretely, we construct training and validation splits for experience evolution. RedEvoAgent first measures the effectiveness ofjailbreak tools on the training split to build a tool-effectiveness profile. To prevent skil evolution from conflating frequently co-occurring tools in successful trajectories with genuinely effective tools, we introduce Deciding-Tool Attribution, which provides clearer tool-selection signals by labeling the tool that directly leads to attack success as the deciding tool. We further design a validation-ratchet mechanism that accepts a candidate skill update only if it yields a improvement on an independent validation set. Otherwise, the previous skill is retained, and failed candidates are recorded to guide later iterations, mitigating irrelevant or misleading update signals. In this way, tool-effectiveness measurements and attack trajectories determine what the skill learns, while validation feedback determines which revisions persist.

To validate the effectiveness of our framework, we conduct experiments on Agent Security Bench (ASB) (Zhang et al., 2025a) and AgentHarm (Andriushchenko et al., 2025) across multiple target models, as well as the Claude Code and Codex target execution harnesses. The results show that RedEvoAgent outperforms existing fixed and agentic baselines, and that the evolved attack skill transfers across different attacker models and target execution harnesses.

## We summarize our contributions as follows:

• We introduce RedEvoAgent, an automatic red-teaming agent that distills cross-case attack experience into an evolving and human-readable attack skill. The framework learns skill updates from training-set attack experience and employs a validation ratchet to reduce noisy tool credit assignment and restrict skill evolution to revisions that improve held-out attack performance.

• We conduct extensive experiments across multiple target models and target execution harnesses, demonstrating improvements in attack effectiveness, attack-tool efficiency, and cross-attacker and cross-harness transferability. Moreover, our reproduced target execution harnesses can provide reusable infrastructure for future agentic red-teaming research.

## 2 RELATED WORK

Jailbreak Attacks and Automatic Red-Teaming. Earlier jailbreak attacks usually rely on one predefined optimization method, such as gradient-based search in GCG (Zou et al., 2023), evolutionary search in AutoDAN (Liu et al., 2024), LLM-based iterative refinement in PAIR (Chao et al., 2025), or tree-structured black-box search in TAP (Mehrotra et al., 2024). A recent line of work instead selects and combines multiple attack strategies or tools. JailbreakOPT (Shi et al., 2026) packages di verse attacks as tools and models tool selection across cases as a contextual bandit, using past attack outcomes to balance exploration and exploitation. MAJIC (Qi et al., 2026) updates the Markov transition probabilities between disguise strategies based on attack feedback. RedCodeAgent (Guo et al., 2026) introduces agent-based automatic red-teaming and retrieves successful attack trajectories similar to the current task to guide tool selection and prompt optimization. Although all three systems reuse past experience, RedCodeAgent stores it in case-specific trajectories, whereas JailbreakOPT and MAJIC encode it in a contextual-bandit policy and a Markov transition matrix, respectively. Retrieving trajectories adds context overhead at inference time, while these numerical states do not explicitly present the learned preferences for choosing attacks as interpretable guidance. Related experience-centric systems use retrievable natural-language strategies, structured experience pools, hybrid text/code libraries, or skill-structured memories (Liu et al., 2025a; Wang et al., 2025; Zhang et al., 2025b; 2026). These methods do not apply incumbent-relative held-out validation to a complete orchestration artifact; AutoRedTeamer instead validates newly implemented attack operators against an absolute ASR threshold before library insertion (Zhou et al., 2025). In contrast, we distill feedback across cases into a compact and interpretable attack skill. A validation ratchet retains a candidate only when it outperforms the current skill on an independent validation set.

Skill Learning and Evolution from Experience. Skill learning from experience uses records of past runs to produce reusable natural-language skill documents: concise instructions that an agent can follow in later runs. Skill evolution then revises these documents using feedback from later runs. Trace2Skill (Ni et al., 2026) extracts lessons from individual trajectories in parallel, then combines recurring lessons in stages to form a unified skill. In its experiments, the unified skill outperforms a memory baseline that retrieves individual past episodes, while requiring no retrieval at test time. SkillOpt (Yang et al., 2026) uses scored rollouts to propose a limited set of additions, deletions, or replacements in a single skill document, and accepts a candidate only when it improves performance on a held-out validation set. SkillOpt-Lite (Shen et al., 2026) simplifies this process by exploring stored trajectories, extracting patterns shared across cases, and independently validating each candidate update. To our knowledge, this work is the first automatic red-teaming method to evolve a single natural-language tool-orchestration skill through incumbent-relative held-out validation.

## 3 METHOD

In this section, we present RedEvoAgent, an experience-driven framework for evolving an attack skill against black-box target agents, as illustrated in Figure 1. §3.1 formalizes the problem setting, optimization objective, and data splits; the following subsection describes the attacker agent architecture, including the toolbox, skill document, and attack workflow; §3.3 explains how attack experience is collected and structured; and §3.4 describes validation-guided skill evolution, retaining only candidates that improve validation performance.

## 3.1 PROBLEM FORMULATION

We consider jailbreak attacks against a black-box target agent $M _ { \mathrm { t a r } } = ( m _ { \mathrm { t a r } } , h _ { \mathrm { t a r } } )$ , where $m _ { \mathrm { t a r } }$ is the target model and $h _ { \mathrm { t a r } }$ is its execution harness, which provides the interaction interface and tool environment. Symmetrically, the attacker agent is $M _ { \mathrm { { a t t } } } ( s ) = ( m _ { \mathrm { { a t t } } } , h _ { \mathrm { { a t t } } } ( s , T ) )$ , an adaptive automatic red-teaming agent that optimizes a jailbreak prompt and launches it against $M _ { \mathrm { t a r } } ; m _ { \mathrm { a t t } }$ is the attacker model, and $h _ { \mathrm { a t t } } ( s , \bar { T } )$ implements this optimize-then-attack loop in a ReAct framework (Yao et al., 2023). It integrates an attack toolbox $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { K } \}$ containing single-turn jailbreak tools (e.g., GCG, FlipAttack), alongside an attack skill s that provides attack strategy guidance.

For each jailbreak case, the attack budget B is the maximum number of execution turns available to $M _ { \mathrm { a t t } } ( s )$ . In each ReAct turn, $M _ { \mathrm { a t t } }$ performs reasoning based on skill s and execution history and decides its action: either calling a jailbreak tool $t _ { k } \in \mathcal T$ to optimize the current jailbreak prompt, or executing QUERYTARGET—submitting the current prompt to $M _ { \mathrm { t a r } }$ and receiving the target agent’s response as an observation for the next turn. Our goal is to learn the attack skill s that best guides $M _ { \mathrm { a t t } }$ to maximize attack effectiveness against $M _ { \mathrm { t a r } }$ . Throughout evolution, $M _ { \mathrm { t a r } } , m _ { \mathrm { a t t } } , h _ { \mathrm { a t t } }$ , and T remain frozen; only s is updated based on attack experience.

For a jailbreak case $x ,$ one complete attack rollout under skill s produces a trajectory $\tau _ { x } ( s )$ and a benchmark-native case score $r _ { x } ( s )$ evaluated by an external judge based on the target agent’s response:

$$
\left( \tau _ { x } ( s ) , r _ { x } ( s ) \right) = \mathrm { R o l l o u t } \bigl ( M _ { \mathrm { a t t } } ( s ) , M _ { \mathrm { t a r } } , x ; B \bigr ) .\tag{1}
$$

For a set of cases $D .$ , we measure the skill’s mean effectiveness as

$$
J _ { D } ( s ) = { \frac { 1 } { | D | } } \sum _ { x \in D } r _ { x } ( s ) .\tag{2}
$$

RedEvoAgent uses the training split $D _ { \mathrm { t r } }$ to collect attack experience and generate candidate skills, the validation split $D _ { \mathrm { v a } }$ to select updates, and the test split $D _ { \mathrm { t e } }$ only to evaluate the final accepted skill.

## 3.2 ATTACKER AGENT ARCHITECTURE

The action space A of $M _ { \mathrm { a t t } } ( s )$ comprises QUERYTARGET and tool-call actions from the toolbox T:

$$
\begin{array} { r } { \mathcal { A } = \{ \mathrm { Q u g e R Y T A R G E T } \} \cup \mathcal { T } . } \end{array}\tag{3}
$$

QUERYTARGET submits the attacker’s current jailbreak prompt to $M _ { \mathrm { t a r } }$ and returns the target agent’s response as an observation. Calling a tool $t _ { k } \in \bar { \mathcal { T } }$ applies that jailbreak tool to the current prompt and returns an optimized variant for subsequent QUERYTARGET actions; each action from A consumes one turn of the budget B.

The attack toolbox T integrates seven complementary jailbreak tools: gradient-based search (GCG), evolutionary search (AutoDAN), generator-based suffix generation (AmpleGCG), predefined template wrapping (Template), character flipping disguise (FlipAttack), role-playing reformulation (RolePlay), and our proposed Prompt Substitution, which uses an auxiliary LLM to rephrase a failed prompt while keeping the same intent.

The attack skill s is a Markdown document inserted into $M _ { \mathrm { a t t } } \mathbf { \ ' } _ { \mathbf { s } }$ system prompt. It summarizes measured tool efficacy and attack orchestration heuristics as compact, reusable guidance.

For a given case $x ,$ the execution flow proceeds as follows: $M _ { \mathrm { a t t } } ( s )$ iteratively reasoning based on skill s and execution history to select actions from A—either optimizing the jailbreak prompt via $t _ { k } \in \mathcal { T }$ or probing $M _ { \mathrm { t a r } }$ via QUERYTARGET. This process iterates until a jailbreak succeeds or the budget B is exhausted, yielding the rollout trajectory $\tau _ { x } ( s )$

## 3.3 EXPERIENCE COLLECTION

Experience determines what the skill learns. RedEvoAgent constructs two complementary inputs for skill distillation: an empirical attack tool efficacy profile and annotated trajectories from adaptive attack rollouts.

Tool-Efficacy Profile. Different target agents exhibit varying resistance to different jailbreak tools. A tool ordering that works well against one target agent may be suboptimal against another. RedEvoAgent therefore measures each tool $t \in \tau$ in isolation on $D _ { \mathrm { t r } }$ before skill evolution, with efficacy defined as:

$$
e _ { t } = \frac { 1 } { | D _ { \mathrm { t r } } | } \sum _ { x \in D _ { \mathrm { t r } } } r _ { x } ( t ) , \qquad t \in \mathcal { T } .\tag{4}
$$

![](images/e0d4dadae891d9c8bcdd5ce2b6d5bab218f084c00ff4ceadf96e5296e4b5673a.jpg)  
Figure 2: Skill evolution from attack experience. (a) On the training split, RedEvoAgent collects successful and failed attack trajectories. Deciding-Tool Attribution assigns each successful trajectory to the attack tool immediately preceding its first successful target query, while isolated tool evaluations form a tool-efficacy profile. (b) In each round, the distiller agent uses both sources, the current skill, and the rejection context, which records previously rejected candidates and their val idation scores, to produce a candidate skill. The validation ratchet accepts the candidate only if it improves attack performance on the validation split; otherwise, the current skill is retained and the candidate is added to the rejection context. (c) Once accepted, the candidate becomes the current skill, the rejection context is cleared, and new training trajectories are collected for the next round. After the final round, the accepted skill records tool priorities and attack strategies.

Together, these measurements form the tool-efficacy profile:

$$
\mathbf { e } = ( e _ { t } ) _ { t \in \mathcal { T } } ,\tag{5}
$$

The profile supplies the distiller agent with an empirical tool ranking derived from prior training cases.

Trajectory Collection. Before the first iteration of skill evolution and after accepting any updated skill, RedEvoAgent runs $M _ { \mathrm { a t t } } ( s )$ over the training split $D _ { \mathrm { t r } }$ <sub>r</sub> using the current accepted-best skill s. The system logs the complete execution trace for each case—including tool calls, candidate prompts, QUERYTARGET actions, target agent responses, and case scores—yielding the set of training trajectories:

$$
\Gamma ( s ) = \{ \tau _ { x } ( s ) \mid x \in D _ { \mathrm { t r } } \} .\tag{6}
$$

These raw trajectories form the foundational experience used by the distiller agent to update the skill.

Deciding-Tool Attribution. While the tool-efficacy profile reveals which tools are most effective overall, the skill must also extract dynamic orchestration strategies from adaptive attack trajectories. However, adaptive attack trajectories can create a self-reinforcing tool-selection bias. Because $M _ { \mathrm { a t t } }$ has its own tool preferences, certain tools may frequently appear in successful trajectories even when a subsequent tool produces the successful QUERYTARGET action. If the distiller agent misinterprets mere co-occurrence frequency as genuine contribution, it erroneously promotes these co-occurring tools in the skill. The updated skill then compels $M _ { \mathrm { a t t } }$ to select these tools even more frequently, amplifying the bias and ultimately leading to a strategy collapse.

To break this self-reinforcing misattribution, RedEvoAgent introduces Deciding-Tool Attribution, which assigns each successful trajectory to the jailbreak tool immediately preceding its successful QUERYTARGET action. This mechanism serves as a crucial stabilizer for skill evolution rather than a source of new attack capabilities: when tool co-occurrence in raw trajectories is low, its impact may be modest; however, under severe selection bias, it significantly mitigates the risk of strategy collapse.

As illustrated in Figure 2, RedEvoAgent feeds both the tool-efficacy profile e and the attack trajectories $\Gamma ( s )$ with Deciding-Tool Attribution into the distiller agent to produce candidate skills, whose updates are subsequently gated by validation feedback.

Algorithm 1 Experience-driven attack-skill evolution   
1: Input: frozen target agent $M _ { \mathrm { t a r } } .$ , attacker agent $M _ { \mathrm { a t t } } ( \cdot )$ , toolbox $\tau ,$ , train split $D _ { \mathrm { t r } }$ , val split   
$D _ { \mathrm { v a } }$ , attack budget B, max rounds R   
2: Output: final accepted attack skill $s ^ { * }$   
3: $s ^ { * } \in \mathring {  } \varnothing , \quad v ^ { * }  \mathring { J _ { D _ { \mathrm { v a } } } } ( s ^ { * } ) , \quad \mathcal { C }  \varnothing$   
4: $\mathbf { e } \gets$ EVALUATETOOL $\mathfrak { s } ( \mathcal { T } , D _ { \mathrm { t r } } )$ Γ<sup>∗</sup> ← COLLECTEXPERIENCE $( s ^ { * } , D _ { \mathrm { t r } } ; B )$   
5: for $i = 1$ to R do   
6: $\hat { s } _ { i } \gets S$ YNTHESIZE $( s ^ { * } , \Gamma ^ { * } , \mathbf { e } , \mathcal { C } ) , \quad \hat { v } _ { i } \gets J _ { D _ { \mathrm { v a } } } ( \hat { s } _ { i } )$   
7: if $\hat { v } _ { i } > v ^ { * }$ then   
8: $\boldsymbol { s } ^ { * } \gets \hat { \boldsymbol { s } } _ { i } , \quad \boldsymbol { v } ^ { * } \gets \hat { \boldsymbol { v } } _ { i }$ Γ<sup>∗</sup> ← COLLECTEXPERIENCE $( s ^ { * } , D _ { \mathrm { t r } } ; B )$ , C ← ∅   
9: else   
10: $\mathcal { C } \gets \mathrm { A P P E N D } ( \mathcal { C } , ( \hat { s } _ { i } , \hat { v } _ { i } ) )$   
11: end if   
12: end for   
13: return $s ^ { * }$

## 3.4 VALIDATION RATCHET

Validation feedback determines whether a skill update is accepted. Section 3.3 yields the toolefficacy profile e and the attack trajectories $\Gamma ( s )$ with Deciding-Tool Attribution. However, synthesizing a new skill from historical trajectories does not guarantee improved subsequent attack performance. Attack trajectories contain both genuinely effective attack strategies and failed attempts, redundant steps, and spurious patterns induced by the attacker agent’s own preferences; text rewrites by the distiller agent may also introduce ineffective or overfit rules. For example, Red-CodeAgent (Guo et al., 2026) retrieves complete historical trajectories by case similarity; this reuses similar attack experience, but similarity alone does not verify that retrieved trajectories help the current case, and redundant steps can increase context overhead and interfere with the attacker agent’s decisions. Therefore, RedEvoAgent does not adopt distiller agent output directly; each rewrite is treated as a candidate that must demonstrate effectiveness on the validation split $D _ { \mathrm { v a } }$ . Throughout, the system distinguishes the current accepted-best skill $s ^ { * }$ from the candidate skill $\hat { s } _ { i }$ proposed in round i; a candidate replaces $s ^ { * }$ only if it strictly outperforms $s ^ { * }$ on $D _ { \mathrm { v a } }$

Candidate Skill Synthesis. We define $\mathcal { C }$ as the rejection context, which sequentially records rejected candidate skills and their validation scores since the last accepted update. In round $i ,$ the distiller agent receives the current skill $s ^ { * }$ , tool-efficacy profile $\mathbf { e } ,$ current training trajectories $\Gamma ( s ^ { * } )$ and rejection context $\mathcal { C } .$ The system first stratifies trajectories by attack success and failure and partitions them into minibatches. The distiller agent analyzes each minibatch in parallel, compares effective versus ineffective attack processes, and proposes edits; it then merges these suggestions and rewrites $s ^ { * }$ under a bounded-change constraint. Here, skill rewrite means the distiller agent integrates the merged edit suggestions into the current skill and outputs a complete candidate skill $\hat { s } _ { i }$ The tool-efficacy profile e anchors tool priorities in isolated measurements, while $\mathcal { C }$ tells the distiller agent which edit directions have already failed validation on the same evidence, reducing repeated generation of the same ineffective candidates.

Candidate Validation and Update. After candidate $\hat { s } _ { i }$ is synthesized, RedEvoAgent runs $M _ { \mathrm { a t t } } ( \hat { s } _ { i } )$ independently on $D _ { \mathrm { v a } }$ and computes $J _ { D _ { \mathrm { v a } } } ( \hat { s } _ { i } )$ . If $J _ { D _ { \mathrm { v a } } } ( \hat { s } _ { i } )$ strictly exceeds the validation score of the current accepted skill, the system accepts $\hat { s } _ { i }$ as the new $s ^ { * }$ , re-collects and annotates trajectories on $D _ { \mathrm { t r } }$ with the new skill, and refreshes experience for the next evolution round. Upon acceptance, rejection context $\mathcal { C }$ is cleared, because subsequent candidates start from the new skill and new training trajectories. If the candidate yields no strict improvement, the system retains the previous $s ^ { * }$ and trajectories, and appends $\left( \hat { s } _ { i } , J _ { D _ { \mathrm { v a } } } \left( \hat { s } _ { i } \right) \right)$ to $\mathcal { C }$ , steering the next round to explore different edits on the same evidence. Thus, a rejected candidate is merely an update attempt that failed validation, not the round’s algorithmic output.

The update rule above can be written as

$$
s _ { i } ^ { * } = \left\{ \begin{array} { l l } { \hat { s } _ { i } , } & { J _ { D _ { \mathrm { v a } } } ( \hat { s } _ { i } ) > J _ { D _ { \mathrm { v a } } } ( s _ { i - 1 } ^ { * } ) , } \\ { s _ { i - 1 } ^ { * } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

The distiller agent reads only the tool-efficacy profile e and attack trajectories built from $D _ { \mathrm { t r } } ; D _ { \mathrm { v a } }$ supplies the scalar used for candidate selection, and $D _ { \mathrm { t e } }$ evaluates the final accepted $s ^ { * }$ once.

## 4 EXPERIMENTS

## 4.1 SETUP

Datasets and Target Agents. We evaluate RedEvoAgent on Agent Security Bench (ASB) (Zhang et al., 2025a), a benchmark with 400 cases (10 agents × 40 attacker tools) and no prompt-variant augmentation, and AgentHarm (Andriushchenko et al., 2025), which includes 52 base behaviors and four prompt variants (208 cases). We split them into train/validation/test as 80/40/280 and 40/20/148, respectively. Target models are MiniMax-M2.5 (MiniMax, 2026), DeepSeek-V4- Flash (DeepSeek-AI, 2026), and Qwen3.5-35B (Qwen Team, 2026), each paired with Claude Code (Anthropic, 2026a) or Codex (OpenAI, 2025) as the target execution harness.

Baselines and Metrics. We define No Jailbreak as a single target query using the original benchmark prompt, without attack tool transformation or the attacker loop. The six isolated attack tool baselines are GCG (Zou et al., 2023), AmpleGCG (Liao & Sun, 2024), AutoDAN (Liu et al., 2024), Template from GPTFuzzer (Yu et al., 2023), FlipAttack (Liu et al., 2025b), and RolePlay, an LLMbased role-play rewriter adopted from RedCodeAgent (Guo et al., 2026). We also compare with RedCodeAgent (Guo et al., 2026), an automatic red-teaming agent, and MAJIC (Qi et al., 2026), an adaptive jailbreaking framework. We report ASR and RR on ASB (Zhang et al., 2025a), HarmScore and RR on AgentHarm (Andriushchenko et al., 2025), and average attack-tool calls per case in Table 2 as an efficiency measure. ASR is the fraction of cases in which the target executes the attacker goal, using ASB’s native substring check. HarmScore follows the AgentHarm grading protocol: the mean of each case’s grader checks, in [0, 1] and reported as a percentage. RR is the fraction of cases in which the target explicitly refuses the request.

Implementation Details. We use GPT-4o mini (OpenAI, 2024) as the attacker model and an interaction budget of B=20 attacker-agent loop steps. The MiniMax and DeepSeek campaigns use Claude Opus 4.8 (Anthropic, 2026b) for skill evolution; the Qwen campaign uses Codex GPT-5.6 Terra (OpenAI, 2026) at high reasoning effort. Skill evolution runs for R=4 rounds using SkillOpt’s default minibatch size 8 and edit cap L=6 (Yang et al., 2026), with stratified sampling and seed=42. Candidate attack skills are selected by ASR on ASB and HarmScore on AgentHarm. GCG and AutoDAN optimize on the white-box surrogate Qwen2.5-7B-Instruct (Yang et al., 2024a); AmpleGCG uses the public Llama-2-7B-chat generator (Touvron et al., 2023; Liao & Sun, 2024). The optimized prompts are then submitted to the black-box target agent, which is the standard transfer protocol for these tools when target weights are unavailable (Zou et al., 2023; Liu et al., 2024; Guo et al., 2026).

## 4.2 COMPARISON WITH ATTACK BASELINES

Table 1 compares RedEvoAgent with baselines at two levels: six isolated attack tools and two automatic red-teaming methods, RedCodeAgent and MAJIC. No Jailbreak provides a reference without attack transformation. The results show that different target agents, each formed by pairing a target model with an execution harness, resist different jailbreak tools, and that no single tool succeeds against all of these combinations. This uneven resistance indicates that composing jailbreak tools is necessary rather than relying on any one attack method. On many rows, GCG, AmpleGCG, and AutoDAN fall below No Jailbreak: the transferred suffix or templated prefix can look more anomalous in a product-level agent harness than the original request, which raises refusal, whereas semantic disguises such as FlipAttack and RolePlay remain stronger. Guided by an attack skill, RedEvoAgent matches or exceeds the strongest isolated tool in every setting. On AgentHarm with DeepSeek-V4-Flash under Claude Code, it reaches 74.3 HarmScore versus 67.9 for FlipAttack; on ASB with MiniMax-M2.5 under Codex, 92.8 ASR versus 81.1 for FlipAttack. Compared with automatic red-teaming methods, RedEvoAgent also improves over RedCodeAgent and MAJIC in most settings, and is slightly behind MAJIC only on MiniMax-M2.5 / AgentHarm under Claude Code (20.8 vs. 22.6). On AgentHarm with DeepSeek-V4-Flash under Claude Code, the retrieved trajectories used by RedCodeAgent do not help beyond FlipAttack (37.5 vs. 67.9 HarmScore), indicating that memory retrieved by vector similarity has no positive effect on the attack in this setting. In contrast, RedEvoAgent, guided by the attack skill, reaches 74.3 HarmScore.

Table 1: RedEvoAgent vs. attack baselines. Cells report ASR/RR (%) on ASB and mean Harm-Score/RR (%) on AgentHarm. Bold and underline mark the best and second-best ASR or HarmScore within each displayed row, respectively.
<table><tr><td colspan="10">(a) Claude Code harness</td></tr><tr><td>Benchmark Target Model</td><td></td><td>No Jailbreak</td><td>GCG</td><td>AmpleGCG AutoDAN Template FlipAttack RolePlay RedCodeAgent MAJIC RedEvoAgent</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">ASB</td><td>MiniMax-M2.5</td><td>72.5/26.1</td><td>63.6/28.9</td><td>61.8/26.4</td><td>38.6/53.2</td><td>40.4/47.1</td><td>91.8/3.9 68.2/23.2</td><td>76.4/21.4</td><td>74.2/15.1</td><td>93.2/6.1</td></tr><tr><td>DeepSeek-V4-Flash</td><td>83.9/7.5</td><td>81.1/8.2</td><td>77.9/13.6</td><td>40.7/21.1</td><td>66.4/15.7</td><td>94.3/2.1 91.8/2.5</td><td>90.7/8.6</td><td>94.6/2.5 80.6/12.9</td><td>99.3/0.0</td></tr><tr><td>Qwen3.5-35B</td><td>82.8/15.1</td><td>81.7/16.1</td><td>77.4/16.1</td><td>77.4/16.1</td><td>47.3/20.4</td><td>83.9/15.1 84.9/15.1</td><td></td><td>79.6/11.8</td><td>87.1/12.9</td></tr><tr><td></td><td>MiniMax-M2.5</td><td>9.2/85.1</td><td>7.9/81.1</td><td>6.1/84.2 1.0/89.9</td><td>0.0/88.5</td><td>18.6/28.1</td><td>8.7/80.8</td><td>20.9/75.5</td><td>22.6/34.7</td><td>20.8/58.9</td></tr><tr><td>AgentHarm</td><td>DeepSeek-V4-Flash</td><td>47.7/44.3</td><td>37.2/48.3</td><td>34.8/51.7</td><td>10.5/60.7 28.5/48.3</td><td>67.9/11.7</td><td>43.8/43.3</td><td>37.5/67.6</td><td>45.5/32.4</td><td>74.3/15.9</td></tr><tr><td></td><td>Qwen3.5-35B</td><td>5.1/81.6</td><td>4.1/77.6</td><td>4.5/75.5</td><td>0.0/87.5</td><td>0.0/97.5 5.7/75.0</td><td>6.7/80.0</td><td>9.6/66.7</td><td>11.1/49.0</td><td>15.1/63.3</td></tr><tr><td colspan="9">(b) Codex harness</td></tr><tr><td>Benchmark Target Model</td><td></td><td>No Jailbreak</td><td>GCG</td><td></td><td></td><td></td><td></td><td></td><td></td><td>AmpleGCG AutoDAN Template FlipAttack RolePlay RedCodeAgent MAJIC RedEvoAgent</td></tr><tr><td>ASB</td><td>MiniMax-M2.5 DeepSeek-V4-Flash</td><td>63.2/33.2 97.1/1.8</td><td>62.9/28.5 92.1/2.6</td><td>61.8/29.6</td><td>49.3/42.8 91.1/4.7</td><td>38.2/40.9 90.4/6.6</td><td>81.1/7.0 94.6/0.4</td><td>70.4/26.6 93.9/2.6</td><td>75.3/23.3 98.6/1.4</td><td>71.0/22.6 98.9/0.7</td><td>92.8/6.9 100.0/0.0</td></tr><tr><td></td><td>Qwen3.5-35B</td><td>89.2/10.8</td><td>86.0/12.9</td><td>91.4/4.0 87.1/11.8</td><td>77.4/15.1</td><td>68.8/21.5</td><td>92.5/7.5 87.1/10.8</td><td>88.2/9.7</td><td>80.6/6.5</td><td>94.6/5.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.7/45.0</td></tr><tr><td>AgentHarm</td><td>MiniMax-M2.5</td><td>9.4/79.6</td><td>4.8/87.2</td><td>7.8/85.0</td><td>0.0/95.9</td><td>0.1/87.8</td><td>13.3/39.9 67.3/9.9</td><td>10.8/81.8</td><td>14.1/81.1</td><td>17.5/40.8</td></tr><tr><td></td><td>DeepSeek-V4-Flash</td><td>61.9/25.2</td><td>59.5/28.4</td><td>62.3/25.0</td><td>53.6/22.6</td><td>40.1/44.2</td><td>61.8/26.8</td><td>69.4/32.4</td><td>71.5/25.0</td><td>74.4/18.0</td></tr><tr><td>Qwen3.5-35B</td><td></td><td></td><td></td><td>0.0/95.0</td><td>0.0/87.5</td><td>10.1/72.5</td><td>5.4/77.5</td><td>16.7/61.2</td><td>20.1/61.2</td><td>20.9/64.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>14.0/61.2</td><td>3.7/89.8</td><td>6.7/75.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Skill contribution. Cells report ASR / average attack tool calls per case on ASB and mean HarmScore / average attack tool calls per case on AgentHarm. Bold marks the best ASR or HarmScore.
<table><tr><td>Benchmark Harness</td><td></td><td>Target Model</td><td>No Skill</td><td>Human Skill</td><td>RedEvo Agent</td></tr><tr><td rowspan="3">ASB</td><td rowspan="2">Claude Code</td><td>MiniMax-M2.5</td><td>77.5/3.0</td><td>78.3/3.2</td><td>93.2/1.8</td></tr><tr><td>DeepSeek-V4-Flash</td><td>96.4/2.6</td><td>96.4/2.7</td><td>99.3/0.9</td></tr><tr><td rowspan="2">Codex</td><td>MiniMax-M2.5</td><td>80.8/2.7</td><td>81.3/3.0 98.9/2.8</td><td>92.8/1.8</td></tr><tr><td></td><td>DeepSeek-V4-Flash</td><td>97.9/2.7</td><td></td><td>100.0/0.8</td></tr><tr><td rowspan="3">AgentHarm</td><td>Claude Code</td><td>MiniMax-M2.5</td><td>15.4/3.9</td><td>13.1/4.2</td><td>20.8/2.4</td></tr><tr><td rowspan="2">Codex</td><td>DeepSeek-V4-Flash</td><td>34.3/3.0</td><td>40.5/3.0</td><td>74.3/2.2</td></tr><tr><td>MiniMax-M2.5 DeepSeek-V4-Flash</td><td>15.5/3.9 59.4/3.2</td><td>13.5/4.1 59.7/2.9</td><td>21.7/2.3 74.4/2.6</td></tr></table>

## 4.3 ABLATION STUDIES

We conducted ablation studies to assess the contributions of RedEvoAgent’s core components. The results show that all proposed components contribute to RedEvoAgent’s performance.

Skill Contribution. This study asks whether encoding the attack strategy as an attack skill document is useful. Table 2 compares No Skill, Human Skill, and the evolved attack skill. No Skill uses the attacker agent’s default system prompt with no attack skill document, Human Skill inserts generic human-written guidance adapted from RedCodeAgent’s attacker prompt (Guo et al., 2026), and RedEvoAgent inserts the attack skill produced by skill evolution. RedEvoAgent achieves the highest attack performance in every evaluated setting. It also uses the fewest attack-tool calls per case, indicating lower attack cost and higher efficiency through more directed tool selection.

Experience Components. This study tests the necessity of each experience component used to construct the attack skill. Table 3 removes the Tool-Effectiveness Profile, trajectory collection, or Deciding-Tool Attribution, lowering ASR from 93.2% to 76.9%, 85.0%, and 87.5%, respectively. Removing the profile is the largest degradation (−16.3): without it, skill evolution lacks this prior knowledge, and the attacker agent selects tools blindly. Removing trajectory collection is milder (−8.2): skill evolution then cannot observe how jailbreak tools are composed and sequenced. Removing Deciding-Tool Attribution is smaller still (−5.7): the attacker model already favors some tools even when they cannot jailbreak the target, and without this mechanism that bias is written back into the skill, which can lower attack success and increase the number of attack-tool calls.

Ratchet Depth. This study varies the maximum number of validation-ratchet rounds R, both as a test of iterative validation and as a hyperparameter study of ratchet depth. Figure 3 shows validation and test performance as the maximum number of ratchet rounds R increases, using at each R the skill

Accepted skill Accepted candidate

Table 3: Experience-component ablations on the ASB test set with MiniMax-M2.5 and the Claude Code harness. Cells are ASR (%). ✓ = included, × = removed. $^ { 6 6 } - ^ { 5 9 } =$ not applicable: Deciding-Tool Attribution labels collected trajectories, so it is undefined when trajectory collection is off. Bold is the full system.  
Table 4: Zero-shot transfer of the MiniMax-M2.5 attack skill evolved with GPT-4o mini and Claude Code. Cells are ASR (%) on the ASB test set. No Skill is the destination baseline with the default attacker prompt; Transferred reuses the source skill without re-evolution. Parentheses are gains over No Skill.
<table><tr><td>Tool-Effectiveness Trajectory Deciding-Tool Profile</td><td>Collection Attribution</td><td>ASR</td></tr><tr><td>×</td><td>√</td><td>√ 76.9</td></tr><tr><td>√</td><td>×</td><td>85.0 一</td></tr><tr><td>√</td><td>√</td><td>× 87.5</td></tr><tr><td>√</td><td>√</td><td>√ 93.2</td></tr></table>

<table><tr><td colspan="3">(a) Attacker transfer (from GPT-4o mini)</td></tr><tr><td>Destination</td><td>No Skill</td><td>Transferred</td></tr><tr><td>Qwen3-8B</td><td>89.7</td><td>95.3 (+5.6)</td></tr><tr><td>Qwen3-4B</td><td>83.3</td><td>90.6 (+7.3)</td></tr><tr><td>(b) Harness transfer (from Claude Code)</td><td></td><td></td></tr><tr><td>Destination</td><td>No Skill</td><td>Transferred</td></tr><tr><td>Codex</td><td>80.8</td><td>90.5 (+9.7)</td></tr></table>

![](images/b77c8ee5e7db7944f4e86c6d0ba909e38572dfbeb7ee6e4784962e8d6d4befd8.jpg)

![](images/9522b52ab670c9c6431ae5b8514ae46027b638bfc136b55b31f0e344a6722dd1.jpg)  
Rejected candidate Accepted-skill test

![](images/4bf647f997ce34b59e9ffb1ca698f6a545653956d551cba0d7d057d718b93a63.jpg)

![](images/aa21051c2adbfe990b131da9b6e4bcfa784700ee01794eb8eafbe5e8d65a8b06.jpg)  
Figure 3: Validation-ratchet evolution. The x-axis is the maximum number of ratchet rounds R; R=0 is the No Skill condition. The y-axis is ASR on ASB and HarmScore on AgentHarm, with an independent scale per panel. Orange step lines show the validation score of the skill accepted after each round; filled orange circles mark accepted candidate skills, gray crosses show rejected candidate skills at their validation scores, and solid green lines show test scores for accepted skill versions. Rejected candidates are not tested, and flat orange segments carry forward the previous accepted skill version. Candidate skills are accepted based only on validation scores; test scores are not used by the gate.

accepted under that budget. Accepted rounds raise test scores when there is room above the isolatedtool ceiling: ASB/MiniMax accepts R1 and R3 (test ASR 93.2) and AgentHarm/DeepSeek accepts through R3 (test HarmScore 74.3), rejecting the remaining candidates. ASB/DeepSeek is already near saturation after R1, where validation ASR reaches 100 and later rounds stop (test ASR 99.3). The exception is AgentHarm/MiniMax: R4 improves validation HarmScore from 18.5 to 28.1 but reduces test HarmScore from 28.8 to 20.8 relative to R2. Inspection of the accepted skills suggests a plausible explanation: R4 shifts its guidance from broad one-pass coverage of distinct tools toward repeated sampling of FlipAttack and RolePlay. This more concentrated policy is rewarded on the validation split but generalizes less well to the broader test behaviors; a more diverse validation split is one possible remedy.

## 4.4 ZERO-SHOT TRANSFER

Table 4 tests zero-shot transfer of the MiniMax-M2.5 attack skill evolved with GPT-4o mini and Claude Code, reusing it without re-running skill evolution while fixing the target model and ASB test set and varying either the attacker model (Qwen3-8B and Qwen3-4B (Yang et al., 2025)) or the target execution harness. The transferred attack skill improves over the corresponding No Skill baseline in every setting. When the target execution harness is switched from Claude Code to Codex, Transferred reaches 90.5 ASR versus 80.8 for No Skill, retaining most of the 92.8 ASR of a skill evolved directly for Codex. Thus, the transferred attack skill remains useful beyond the attacker model and target execution harness used to evolve it.

## 4.5 WHEN DO SKILLS HELP?

An attack skill is most useful when a target agent exhibits uneven resistance across the attack toolbox and different attack tools expose complementary weaknesses across cases. In this regime, no isolated attack tool is uniformly effective, so an evolved attack skill can prioritize promising tools and guide the attacker agent to switch or sequence them after failures. Conversely, when No Jailbreak or one isolated attack tool already approaches saturation, little room remains. Our cross-target results are qualitatively consistent with this pattern: RedEvoAgent gains more where fixed baselines leave room and little where they are already near ceiling.

## 5 CONCLUSION

We present RedEvoAgent, an automatic red-teaming agent that learns and evolves an attack skill document from past attack experience. Across all experimental settings, RedEvoAgent outperforms the strongest isolated attack tool baseline and existing automatic red-teaming methods. In the skill ablation, the evolved skill achieves the highest attack effectiveness with the fewest attack tool calls. The skill document explicitly records tool priorities and attack strategies and can be transferred across attacker models and target execution harnesses. New attack tools can also be added to the same evaluation and evolution process, although performance depends on the strength and diversity of the toolbox.

## REFERENCES

Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, Yarin Gal, and Xander Davies. AgentHarm: A benchmark for measuring harmfulness of LLM agents. In International Conference on Learning Representations, pp. 79185–79220, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ c493d23af93118975cdbc32cbe7323f5-Abstract-Conference.html.

Anthropic. Claude Code Overview. https://code.claude.com/docs/en/overview, 2026a.

Anthropic. Introducing Claude Opus 4.8. https://www.anthropic.com/news/ claude-opus-4-8, 2026b.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 23–42. IEEE, 2025. doi: 10.1109/ SATML64287.2025.00010.

DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026. URL https://arxiv.org/abs/2606.19348.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, and Jack Clark. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022. URL https://arxiv.org/abs/2209.07858.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. arXiv preprint arXiv:2302.12173, 2023.

Chengquan Guo, Chulin Xie, Yu Yang, Zhaorun Chen, Zinan Lin, Xander Davies, Yarin Gal, Dawn Song, and Bo Li. RedCodeAgent: Automatic red-teaming agent against diverse code agents. In International Conference on Learning Representations, 2026. URL https://openreview. net/pdf/e7bfe29791552fb28e63fdbb4a355263d3610fc2.pdf.

Zeyi Liao and Huan Sun. AmpleGCG: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed LLMs. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=UfqzXg95I5.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations, pp. 56174–56194, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ f83cb637e159e789f5576ff6848874de-Abstract-Conference.html.

Xiaogeng Liu, Peiran Li, G. Edward Suh, Yevgeniy Vorobeychik, Zhuoqing Mao, Somesh Jha, Patrick McDaniel, Huan Sun, Bo Li, and Chaowei Xiao. AutoDAN-Turbo: A lifelong agent for strategy self-exploration to jailbreak LLMs. In International Conference on Learning Representations, pp. 10313–10360, 2025a. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 1bff3663270ba47f801e917f782d7935-Abstract-Conference.html.

Yue Liu, Xiaoxin He, Miao Xiong, Jinlan Fu, Shumin Deng, Yingwei Ma, Jiaheng Zhang, and Bryan Hooi. FlipAttack: Jailbreak LLMs via flipping. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 38623–38663. PMLR, 2025b. URL https://proceedings.mlr.press/ v267/liu25z.html.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 35181–35224. PMLR, 2024. URL https://proceedings.mlr. press/v235/mazeika24a.html.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box LLMs automatically. In Advances in Neural Information Processing Systems, volume 37, pp. 61065–61105. Curran Associates, Inc., 2024. doi: 10.52202/079017-1952. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 70702e8cbb4890b4a467b984ae59828a-Abstract-Conference.html.

MiniMax. The MiniMax-M2 series: Mini activations unleashing max real-world intelligence. arXiv preprint arXiv:2605.26494, 2026. URL https://arxiv.org/abs/2605.26494.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. Trace2Skill: Distill trajectory-local lessons into transferable agent skills. arXiv preprint arXiv:2603.25158, 2026. URL https://arxiv. org/abs/2603.25158.

OpenAI. GPT-4o mini: Advancing cost-efficient intelligence. https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024.

OpenAI. Codex. https://developers.openai.com/codex, 2025.

OpenAI. Codex Models. https://developers.openai.com/codex/models, 2026.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. Red teaming language models with language models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pp. 3419–3448. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022. emnlp-main.225.

Weiwei Qi, Shuo Shao, Wei Gu, Tianhang Zheng, Puning Zhao, Zhan Qin, and Kui Ren. MA-JIC: Markovian adaptive jailbreaking via iterative composition of diverse innovative strategies. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 32755–32763, 2026. doi: 10.1609/aaai.v40i39.40554. URL https://ojs.aaai.org/index.php/ AAAI/article/view/40554.

Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id= qwen3.5, 2026.

Yangjun Ruan, Haonan Dong, Andrew Wang, et al. Identifying the risks of lm agents with an lm-emulated sandbox. In International Conference on Learning Representations, 2024.

Yifei Shen, Bo Li, and Xinjie Zhang. SkillOpt-Lite: Better and faster agent self-evolution via one line of vibe. arXiv preprint arXiv:2607.03451, 2026. URL https://arxiv.org/abs/ 2607.03451.

Ge Shi, Jun Yin, Donglin Xie, Fangyi Liu, Yucan Li, and Menglin Liu. JailbreakOPT: Tool-assisted iterative jailbreak prompt optimization. arXiv preprint arXiv:2606.11425, 2026. URL https: //arxiv.org/abs/2606.11425.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. URL https: //arxiv.org/abs/2307.09288.

Xi Wang, Songlei Jian, Shasha Li, Xiaopeng Li, Bin Ji, Jun Ma, Jing Wang, Xiaodong Liu, Feilong Bao, Jianfeng Zhang, Baosheng Wang, and Jie Yu. Stand on the shoulders of giants: Building JailExpert from previous attack experience. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 3826–3843, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.190. URL https://aclanthology.org/2025.emnlp-main.190/.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a. URL https://arxiv.org/abs/2412.15115.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv. org/abs/2505.09388.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37. Curran Associates, Inc., 2024b. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. SkillOpt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904, 2026. URL https://arxiv.org/abs/2605.23904.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE\_ vluYUL-X.

Jiahao Yu, Xingwei Lin, Zheng Yu, and Xinyu Xing. GPTFUZZER: Red teaming large language models with auto-generated jailbreak prompts. arXiv preprint arXiv:2309.10253, 2023. URL https://arxiv.org/abs/2309.10253.

Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. Agent security bench (ASB): Formalizing and benchmarking attacks and defenses in LLM-based agents. In International Conference on Learning Representations, pp. 35331–35366, 2025a. URL https://proceedings.iclr.cc/paper\_files/paper/ 2025/hash/5750f91d8fb9d5c02bd8ad2c3b44456b-Abstract-Conference. html.

Junke Zhang, Jianwei Wang, Sishuo Chen, Yizhang He, Qingshuai Feng, and Zhengyi Yang. Evolving skill-structured attack memory enhances LLM jailbreaking. arXiv preprint arXiv:2605.29237, 2026. doi: 10.48550/arXiv.2605.29237. URL https://arxiv.org/abs/2605.29237. Under review.

Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, and Hao Wang. Genesis: Evolving attack strategies for LLM web agent red-teaming. arXiv preprint arXiv:2510.18314, 2025b. doi: 10.48550/arXiv.2510.18314. URL https://arxiv.org/abs/2510.18314. Accepted by IEEE ICME 2026.

Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, and Bo Li. AutoRedTeamer: Autonomous red teaming with lifelong attack integration. In Advances in Neural Information Processing Systems, volume 38, pp. 169852–169895. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ f810a445357d94070669d970a95fc5d8-Abstract-Conference.html.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023. URL https://arxiv.org/abs/2307.15043.