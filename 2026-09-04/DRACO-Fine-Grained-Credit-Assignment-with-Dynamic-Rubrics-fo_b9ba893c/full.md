# DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training

Shubham Gandhi<sup>1,∗</sup> Saurabh Goyal<sup>2,∗</sup> Kiran Kate<sup>2</sup> Yara Rizk<sup>2</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>IBM Research Correspondence: srgandhi@andrew.cmu.edu

## Abstract

Reinforcement learning from verifiable rewards works well when a task has a programmatic checker, but most long-horizon agent domains have none. We work in the outcome-blind setting, where ground-truth success signals are not available. Multi-criteria rubrics are a popular way to supply such a reward; they are scored once per trajectory, but a single scalar is a poor signal across tens of steps. We propose DRACO: Distributing Rubric-based Advantage for Credit Optimization. It generates rubrics dynamically during training to track the policy’s evolving capability, scores those rubrics once per completed trajectory, and redistributes that judgment over the steps responsible for annotated rubrics to produce differentiated per-step advantages in GRPO. The redistribution is closed-form and does not introduce any trained attribution module. On App-World, DRACO gains 15.9 points over the base model and 5.3 points over GRPO trained with a sparse ground-truth reward, despite not using any verifiers itself. On out-of-domain τ-bench, it gains 5.3 points over the base model even without a frontier judge, beating both groundtruth-reward training and other rubric-based training settings.<sup>1</sup>

## 1 Introduction

Reinforcement learning from verifiable rewards (RLVR) has driven much of the recent progress in large language models (LLMs), from mathematical reasoning (DeepSeek-AI, 2025; Shao et al., 2024) to interactive tool-using agents (Trivedi et al., 2024; Zhou et al., 2024; Yao et al., 2024). The recipe sidesteps a learned reward model: a programmatic verifier (unit tests for code, exact-match for math, task-pass checks for agents) supplies a reliable terminal reward that is difficult to game. Yet it rests on an assumption that fails in many real-world settings: that a ground-truth verifier exists. Many agent domains, such as customer-support and open-ended research agents, have no programmatic oracle for success, and constructing one is often as hard as the task itself.

We study this problem under an outcome-blind regime: the training signal is derived entirely from process criteria, with no access to a ground-truth success or gold-answer signal at any point during training. This is a strictly harder setting than most of the reward literature assumes, and it interacts sharply with a second difficulty specific to longhorizon agents. A trajectory on a benchmark such as AppWorld (Trivedi et al., 2024) spans tens of interdependent tool-calling steps. Even when a reward is available, attributing a single trajectorylevel scalar uniformly to every step is statistically wasteful and can be actively harmful: successful trajectories contain redundant or lucky steps, and failed trajectories contain mostly correct ones (Arjona-Medina et al., 2019; Harutyunyan et al., 2019). This is the classic credit-assignment problem, and it is well studied for agents when a terminal reward exists (Li et al., 2026b; Feng et al., 2025; Wang et al., 2025). What is missing is credit assignment in the outcome-blind regime: routing a rubric-derived signal, rather than a verifier outcome, to individual steps.

These two difficulties define our target, and DRACO addresses them with two complementary components. First, a fixed rubric authored once for a task distribution cannot anticipate the diverse ways a long trajectory can succeed or fail; we therefore generate dynamic, per-trajectory rubrics that adapt the evaluation criteria to what a given rollout actually does. Second, to convert rubric verdicts into a learning signal that respects the structure of a long trajectory, we perform distribution of rubric-conditioned advantage to steps, mapping criterion-level verdicts to per-step contributions that are then used within GRPO (Shao et al., 2024). Figure 1 shows an overview of the two steps. Together, these instantiate the two axes (taskadaptive coverage and faithful per-step attribution) that the “verification horizon” framing (Wang et al., 2026a) argues are jointly necessary when verification is harder than generation. Crucially, neither component assumes access to a ground-truth outcome at training time, unlike prior methods which tie step rewards to terminal verifiers or gold answers (Tian et al., 2026; Jiang and Ferraro, 2026; Xu et al., 2026b).

![](images/72702a124a6e2af01f2fdc6f886cf3e18b56eeaa08cfccd843b2747c83d72e13.jpg)  
Figure 1: Overview of DRACO. Top: for each task, the judge proposes criteria from the instruction and each sampled trajectory; proposals are merged, deduplicated, and scored to produce an outcome-blind reward $R _ { i }$ (Section 3.2). Bottom: within a rollout group, GRPO normalizes rewards into trajectory advantages $A _ { i } ;$ DRACO then reallocates each $A _ { i }$ across steps according to the judge’s per-criterion verdicts, yielding step advantages $a _ { j }$ that concentrate credit on the steps the rubric implicates (Section 3.3).

We instantiate DRACO on AppWorld, a benchmark whose every previously reported RL result (PPO, GRPO, RLOO, DPO variants, and LOOP (Chen et al., 2025)) trains against the environment’s ground-truth unit tests. To our knowledge, ours is the first RL training on AppWorld that never accesses that signal. On AppWorld<sub>TN</sub>, DRACO achieves 85.3 TGC $( p ^ { 1 } )$ , a +15.9-point gain over the untrained policy, with zero-shot transfer to τ-bench and performance at or above a verifiertrained run of the same budget.

Our contributions include: (1) formulating outcome-blind rubric-based RL for long-horizon tool-using agents, positioning it against the rubric, step-credit, and process-reward literature, nearly all of which anchor their signal to a ground-truth outcome (Section 2); (2) proposing two complementary mechanisms: dynamic per-trajectory rubric generation, and distribution of rubric-conditioned advantage to steps within GRPO, neither of which requires a verifier (Section 3); the credit rule satisfies seven formal properties including total-push conservation, sign preservation, and length independence (Appendix E.9); and, (3) reporting results on AppWorld and τ-bench, and analyze the two components separately and jointly, an outcomereward baseline, a self-judge in place of the frontier judge, and how training affects efficiency and the reward the policy model optimizes (Section 4).

## 2 Related Work

Training long-horizon agents without a verifier raises two coupled problems: what to reward when task success cannot be checked, and how to attribute that reward to individual steps. Table 1 positions prior work along these axes.

Rubric-based rewards. Several methods evolve a rubric during training (adding, pruning, or rewriting criteria as the policy improves) and score it once on the finished rollout (Shao et al., 2026a; Guan et al., 2026; Rezaei et al., 2025; Xu et al., 2026a). A second family scores a fresh rubric at every position, binding criteria to a task-specific decomposition: per-step generation (Tian et al., 2026), deep-research stages (Li et al., 2026a), or subgoal prototypes (Jiang and Ferraro, 2026). This is expensive (a judge call per position) and fixes the notion of a step to the task at hand. Xu et al. (2026b) is the only method that propagates a trajectorylevel rubric score to individual tokens, by training a learned discriminator based on fixed criteria. DRACO retains the evolving, trajectory-scored rubric of the first family but redistributes the judgment over steps in closed form, requiring no perposition judge call and no learned module.

Step-level credit assignment. Most fine-grained credit for LLM agents decomposes the episode outcome: via trained progress estimators (Wang et al., 2025; Zhou et al., 2025), recurrence across rollouts (Li et al., 2026b; Feng et al., 2025), or a reference model’s likelihood of the gold answer (Tao et al., 2026). All require a reliable outcome signal. A second group attributes credit post hoc with an LLM: hindsight Q-value refinement (Tan et al., 2026), failure-step localization (Yeo et al., 2026), or per-action good/bad labels (Zhai et al., 2025). A third specializes to an action space: tool-call formats (Yu et al., 2025), CLI commands (Su and Wen, 2026), or next-state signals that bypass the episode outcome (Wang et al., 2026b). DRACO’s attribution departs on three counts: it consumes no ground-truth outcome, it assumes no particular action structure, and the redistribution is closed-form with no learned component.

## 3 Method

A policy model $\pi _ { \theta }$ interacts with an environment over a long-horizon episode. Given a task instruction x, it produces a trajectory τ = $( s _ { 1 } , a _ { 1 } , \dotsc , s _ { T } , a _ { T } )$ of interleaved reasoning and tool calls, with $T$ on the order of tens of steps. We assume no verifier of task success at training time; the reward must come from process criteria alone. Our method has two parts: a dynamic per-trajectory rubric scored by a frozen judge (Section 3.2), and a step-level credit rule that turns the judge’s verdicts into per-step advantages for GRPO (Section 3.3).

## 3.1 Background: GRPO

We train with Group Relative Policy Optimization (Shao et al., 2024; DeepSeek-AI, 2025). For each task x, we sample a group of $G$ trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ and give each a scalar reward $R _ { i }$ . GRPO standardizes rewards within the group to form the advantage

$$
A _ { i } = \frac { R _ { i } - \operatorname* { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } .\tag{1}
$$

Let trajectory $\tau _ { i }$ emit tokens $y _ { i , 1 } , \ldots , y _ { i , N _ { i } }$ . GRPO assigns the same scalar $A _ { i }$ to every one of these tokens and takes a policy-gradient step in which each token’s log-probability is pushed by its advantage,

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { I } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { N _ { i } } A _ { i } \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( y _ { i , t } \mid y _ { i , < t } , x ) \right] .\tag{2}
$$

The advantage is thus a per-token multiplier on the log-probability gradient: $A _ { i } > 0$ makes the tokens of $\tau _ { i }$ more likely in their context, $A _ { i } < 0$ less likely. Because a single $A _ { i }$ multiplies all $N _ { i }$ tokens, every decision in the episode receives identical credit. In RLVR, $R _ { i }$ is a terminal verifier outcome; we change both the source of $R _ { i }$ (a rubric judge, not a verifier; Section 3.2) and its uniform use, replacing $A _ { i }$ with a per-step advantage $a _ { j }$ that concentrates on the steps the judge implicates (Section 3.3).

## 3.2 Dynamic Per-Trajectory Rubrics

Instead of building one generic rubric for the entire task distribution, we build one rubric per task and score it per trajectory. As shown in figure 1, a judge first proposes criteria from the task instruction alone, then extends them once per sampled rollout, adding the sub-goals that rollout reveals and the ways the policy model actually fails; the proposals from all G rollouts are merged into a single set $\mathcal { R } = \{ c _ { 1 } , \ldots , c _ { K } \}$ with duplicates removed. All generation calls are instructed to keep criteria mutually exclusive and collectively exhaustive (Zhang et al., 2025), since the reward in Eq. (3) is a rate and two overlapping criteria would count one mistake twice. We then apply discriminative dropout, keeping a criterion only if some group member failed it. A frozen external judge scores each trajectory against every surviving criterion, returning pass, fail, or not applicable, coarse by design (Viswanathan et al., 2026), together with a justification and the steps responsible, which the credit rule consumes (Section 3.3). Letting $p _ { i }$ and $f _ { i }$ count the applicable passes and fails,

$$
R _ { i } = { \frac { p _ { i } - f _ { i } } { p _ { i } + f _ { i } } } ,\tag{3}
$$

with $R _ { i } = 0$ when no criterion applies. The criterion set is shared across the group, but applicability is not: a criterion about pagination is moot for a trajectory that never listed a collection. Because $R _ { i }$ normalizes by the verdicts actually cast rather than by $K$ , trajectories with different numbers of applicable criteria remain comparable within the group.

<table><tr><td>Method</td><td>Outcome- Blind</td><td>Rubric Reward</td><td>Dynamic Rubrics</td><td>Traj.-Level Scoring</td><td>Step Attribution</td><td>No Learned Attribution</td></tr><tr><td>DR Tülu (Shao et al., 2026a)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td></td></tr><tr><td>EvoRubric (Guan et al., 2026)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td></td></tr><tr><td>Online Rubrics Elicit. (Rezaei et al., 2025)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td></td></tr><tr><td>RubricARM (Xu et al., 2026a)</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td></td></tr><tr><td>ARCO (Tian et al., 2026)</td><td>0</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>RubricEM (Li et al., 2026a)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td></tr><tr><td>SCRIBE (Jiang and Ferraro, 2026)</td><td>0</td><td>0</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Rubrics to Tokens (Xu et al., 2026b)</td><td>0</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>AgentEvolver (Zhai et al., 2025)</td><td>x</td><td>0</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>HCAPO (Tan et al., 2026)</td><td>X</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>SALT (Li et al., 2026b)</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>DRACO (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison with the closest prior work. ✓ yes, ✗ no, partial, – N/A. Outcome-Blind: the training reward consults no ground-truth outcome ( : judge plus verifier). Rubric Reward: reward derived from written criteria ( : a criteria-guided judge emitting one verdict rather than per-criterion scores). Dynamic Rubrics: criteria that change during training, driven by the training algorithm rather than generated per instance by a fixed procedure. Traj.-Level Scoring: the rubric is scored once on the finished rollout, not at each position. Step Attribution: differentiated credi reaches the policy update at step granularity. No Learned Attribution: attribution requires no trained module.

Note that the reward is completely outcome-blind as no task-success or gold-answer term appears.

## 3.3 Rubric-Conditioned Step Credit

Placing $R _ { i }$ into Eq. (1) still gives every token the same advantage $A _ { i }$ . But each criterion is usually decided by a few steps, not the whole trajectory. Our second change reallocates $A _ { i }$ across the steps of $\tau _ { i }$ according to which steps the judge’s verdicts implicate, without changing the trajectory’s total push. We index the steps of $\tau _ { i }$ by $j = 1 , \dots , M$ (one step is one agent turn, i.e. one emitted code block). Each response token belongs to exactly one step, or to a gap (turn glue, tool-result echoes) that belongs to no step; gap tokens receive no credit and are excluded from the reallocation.

Step quality. When evaluating against the rubric, the judge cites the steps that justify each criterion’s rating. Let $p _ { j }$ and $f _ { j }$ be the number of passed and failed criteria that cite step j. The quality of step $j$ is the pass fraction of the criteria citing it,

$$
Q _ { j } = \frac { p _ { j } } { p _ { j } + f _ { j } } , \qquad Q _ { j } \in [ 0 , 1 ] .\tag{4}
$$

A step cited by no criterion inherits ${ \bar { Q } } ,$ the mean $Q$ over cited steps.

Sign-preserving step weights. The sign of $A _ { i }$ fixes whether the whole trajectory is reinforced $( A _ { i } \ \geq \ 0$ , a “winner”) or suppressed $( A _ { i } ~ < ~ 0$ , a “loser”); credit only decides where within the trajectory that push lands. The step weight makes the reallocation sign-correct:

$$
w _ { j } = \left\{ { Q _ { j } , \atop 1 - Q _ { j } , } \right. \ { A _ { i } \atop A _ { i } < 0 \mathrm { ( r e i n f o r e s s ~ b a d ~ s t e p s ) } . }\tag{5}
$$

A step all of whose citing criteria agree takes weight 0 and is left out of the update, which is the intended reading: on a winner, a step every criterion failed should not be reinforced.

Step advantage. Credit is assigned at the step level: every token in step j receives the same advantage $a _ { j } ,$ , since the rubric grades steps, not tokens. Let $n _ { j }$ be the number of tokens in step j and $N = \textstyle \sum _ { k } n _ { k }$ the tokens lying inside some step. The step advantage is

$$
a _ { j } = A _ { i } \cdot \frac { N w _ { j } } { n _ { j } \sum _ { k } w _ { k } } ,\tag{6}
$$

and it replaces the uniform $A _ { i }$ on every token of step j in the GRPO update. What the rule equalizes is each step’s total contribution: step j contributes $\begin{array} { r } { n _ { j } a _ { j } \ = \ A _ { i } N w _ { j } / \sum _ { k } w _ { k } } \end{array}$ , which depends on its quality weight and not on its length. A step therefore earns influence by being judged good, not by being verbose, and the $1 / n _ { j }$ factor is what spreads that fixed total over however many tokens the step happens to contain. Summing over steps conserves the trajectory’s total push,

$$
\sum _ { j } n _ { j } a _ { j } = A _ { i } N \frac { \sum _ { j } w _ { j } } { \sum _ { k } w _ { k } } = A _ { i } N ,\tag{7}
$$

which is exactly the total baseline GRPO applies to those same N tokens. Credit reallocation therefore never inflates or deflates a trajectory’s overall influence; it moves the existing influence onto the steps that earned it. Because every $w _ { j } \geq 0$ , no $a _ { j }$ ever takes the opposite sign to $A _ { i } ,$ so credit is never inverted relative to the judge’s trajectory-level verdict. When the rubric cites no step at all there is nothing to reallocate on, and the update falls back to baseline GRPO. A full derivation, worked example, a traced real rollout, and the configuration are given in Appendix E.

Training loop. In each step: sample a group of G trajectories for each task; generate the rubric in three stages, merging into one set shared by the group; score every trajectory against that set with the frozen judge, obtaining a verdict and its step citations per criterion; drop the criteria no member failed and compute Eq. (3) over the survivors; standardize within the group into $A _ { i }$ (Eq. (1)); compute step qualities Eq. (4), weights Eq. (5), and step advantages Eq. (6) from the retained criteria’s citations; update the policy model. The four settings in Section 4.1.1 isolate these parts by removing per-trajectory rubrics, step credit, or both from DRACO.

## 4 Experimentation

## 4.1 Experimental Setup

## 4.1.1 Settings

We evaluate four outcome-blind settings initialized from the same base policy model (Qwen3.6-27B), each named for what it removes from DRACO: (i) DRACO w/o Dyn. & Cred. (static rubric and no step credit), (ii) DRACO w/o Dyn. (static rubric with rubric-conditioned step credit; Section 3.3), (iii) DRACO w/o Cred. (dynamic rubrics without step credit), and (iv) DRACO. An outcome reward model trained on AppWorld unit tests (binary) serves as an outcome-aware reference.

## 4.1.2 Training and Implementation

Data and models. We train on the AppWorld training split (90 tasks). We report results on Qwen3.6-27B (used for all ablations) and Qwen2.5-32B-Instruct (Team, 2024).

RL and hyperparameters. All runs train LoRA adapters (Hu et al., 2022) using GRPO (Section 3.1). Each training step uses batch size B = 16 with GRPO group size G = 6 per task. All settings share a single pre-committed hyperparameter configuration, differing strictly in their reward signals (Section 4.1.1), and every run trains on 8 H100 GPUs. All training-time reward operations such as rubric generation, union, scoring, and credit reallocation use a single judge model (GPT-5.4, temperature 0.1) for consistency. Appendix C details static rubric construction and lists the resulting criteria (Table 9), Appendix D gives full hyperparameters (Table 10) and model-specific serving configurations, and Appendix F includes every judge prompt and the agent’s system prompt.

## 4.1.3 Evaluation

Benchmarks. We evaluate on two long-horizon tool-use benchmarks: (i) AppWorld (Trivedi et al., 2024), a stateful environment spanning 9 apps and 457 APIs evaluated via programmatic unit tests. We report on its two held-out splits, AppWorld<sub>TN</sub> (test\_normal, 168 tasks) and AppWorld<sub>TC</sub> (test\_challenge, 417 tasks featuring unseen apps and composition patterns). (ii) τ-bench Banking (Yao et al., 2024), evaluating multi-turn customer service in the banking domain (all-tools setting, using GPT-5.4 as the user simulator and judge). Crucially, we train exclusively on AppWorld; τ -bench is a zero-shot transfer benchmark. Ground-truth success signals are used only for evaluation, never during training.

Metrics. On AppWorld, success is measured by Task Goal Completion (TGC) (unit test pass rate) and Scenario Goal Completion (SGC) (fullscenario success). On τ-bench, success is measured by the state-matching Success Rate (SR). To evaluate both discovery and consistency across n=3 runs, we report pass@k (discovery: ≥ 1 success in k trials) and pass<sup>k</sup> (consistency: success in all k trials) as defined by Yao et al. (2024); at k=1 both equal mean success. We write $p ^ { k }$ for $\mathrm { p a s s } ^ { k }$ and report $p ^ { 1 } , p ^ { 2 }$ , and $p ^ { 3 }$ over the full task split (unsolved tasks count as failures). We also report two efficiency metrics: average agent turns per episode and evaluation cost in USD.<sup>2</sup>

Evaluation protocol. We train Qwen3.6-27B for 100 steps (20 epochs) and Qwen2.5-32B-Instruct for 75 (15 epochs). For each setting, we report the mean over the final three epoch checkpoints, running three independent evaluation runs per checkpoint using AppWorld’s official harness and Simplified ReAct Code Agent scaffold. We avoid selecting checkpoints by held-out task performance to prevent leaking implicit supervision (Li et al., 2026b). Episodes are capped at 50 turns on both AppWorld splits and 100 on τ-bench.

<table><tr><td rowspan="3">Setting</td><td colspan="6">AppWorldTN</td><td colspan="6">AppWorldTC</td><td colspan="3"> $\tau _ { \mathrm { B } }$ </td><td rowspan="2" colspan="3">Average</td></tr><tr><td colspan="3">TGC</td><td colspan="3">SGC</td><td colspan="3">TGC</td><td colspan="3">SGC</td><td colspan="3">SR</td></tr><tr><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td></tr><tr><td>Qwen2.5-32B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $p ^ { 3 }$ </td><td></td></tr><tr><td>Base</td><td>35.7</td><td>27.4</td><td>23.2</td><td>17.3</td><td>11.3</td><td>8.9</td><td>17.3</td><td>10.6</td><td>7.4</td><td>5.8</td><td>2.9</td><td>2.2</td><td>3.4</td><td>1.7</td><td>1.0</td><td>15.9</td><td>10.8</td><td>8.5</td></tr><tr><td>SALT (Outcome-aware)</td><td>66.2</td><td></td><td></td><td>47.9</td><td></td><td></td><td>36.8</td><td></td><td></td><td>20.9</td><td></td><td></td><td>–</td><td></td><td>-</td><td></td><td></td><td></td></tr><tr><td>DRACO (Ours)</td><td>62.9</td><td>52.6</td><td>46.0</td><td>42.3</td><td>32.7</td><td>26.8</td><td>34.0</td><td>24.8</td><td>20.2</td><td>19.2</td><td>13.2</td><td>10.3</td><td>4.0</td><td>1.9</td><td>1.5</td><td>32.5</td><td>25.0</td><td>21.0</td></tr><tr><td>Qwen3.6-27B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>69.4</td><td>55.2</td><td>47.6</td><td>41.1</td><td>28.0</td><td>21.4</td><td>49.7</td><td>38.3</td><td>31.7</td><td>30.2</td><td>18.7</td><td>12.9</td><td>15.8</td><td>9.6</td><td>7.2</td><td>41.2</td><td>30.0</td><td>24.2</td></tr><tr><td>Outcome reward</td><td>80.0</td><td>70.0</td><td>63.3</td><td>59.3</td><td>45.0</td><td>38.1</td><td>59.9</td><td>47.2</td><td>40.4</td><td>38.3</td><td>27.7</td><td>21.3</td><td>17.6</td><td>11.0</td><td>9.3</td><td>51.0</td><td>40.2</td><td>34.5</td></tr><tr><td>DRACO w/o Dyn. &amp; Cred.</td><td>81.1</td><td>71.4</td><td>64.7</td><td>59.9</td><td>45.4</td><td>37.5</td><td>60.5</td><td>50.4</td><td>45.6</td><td>42.4</td><td>33.3</td><td>27.3</td><td>19.4</td><td>11.1</td><td>8.9</td><td>52.7</td><td>42.3</td><td>36.8</td></tr><tr><td>DRACO w/o Dyn.</td><td>81.9</td><td>72.0</td><td>65.5</td><td>60.7</td><td>45.6</td><td>39.3</td><td>59.4</td><td>48.1</td><td>41.9</td><td>39.2</td><td>28.4</td><td>21.8</td><td>18.7</td><td>10.9</td><td>7.9</td><td>52.0</td><td>41.0</td><td>35.3</td></tr><tr><td>DRACO w/o Cred.</td><td>82.1</td><td>73.1</td><td>66.3</td><td>64.9</td><td>52.2</td><td>44.6</td><td>59.3</td><td>48.4</td><td>42.5</td><td>40.6</td><td>31.3</td><td>26.4</td><td>19.7</td><td>11.9</td><td>9.3</td><td>53.3</td><td>43.4</td><td>37.8</td></tr><tr><td>DRACO (Ours)</td><td>85.3</td><td>78.1</td><td>72.8</td><td>70.6</td><td>59.5</td><td>51.8</td><td>61.5</td><td>49.7</td><td>43.9</td><td>40.7</td><td>32.3</td><td>27.3</td><td>20.4</td><td>12.0</td><td>8.6</td><td>55.7</td><td>46.3</td><td>40.9</td></tr><tr><td>DRACO (Ours - self-judge)</td><td>81.1</td><td>72.1</td><td>65.5</td><td>62.7</td><td>48.2</td><td>38.1</td><td>61.0</td><td>45.6</td><td>36.9</td><td>34.7</td><td>21.7</td><td>16.1</td><td>21.1</td><td>14.7</td><td>11.7</td><td>52.1</td><td>40.5</td><td>33.7</td></tr></table>

Table 2: Task success and consistency (%) on $\mathrm { A p p W o r l d _ { T N } }$ (168 tasks), $\mathrm { \ A p p W o r l d _ { T C } }$ (417 tasks), $\tau _ { \mathrm { B } } \ ( 9 7$ tasks). $p ^ { k }$ denotes $\mathrm { p a s s } ^ { k }$ over n=3 runs. Average is a row-level summary only. Trained variants are compared to their block’s Base. Outcome-aware comparisons and our methods are highlighted. DRACO ablations remove dynamic rubrics or step credit. – = not evaluated on that split, or not reported by the cited work.

## 4.2 Overall Results

We report mean over a run’s last three checkpoints.   
Additional results are in Appendix A.

Gains over baselines. DRACO consistently outperforms the untrained policy models Qwen3.6-27B and Qwen2.5-32B-Instruct. On Qwen3.6-27B, it improves AppWorld<sub>TN</sub> TGC/SGC from 69.4/41.1 to 8 $3 5 . 3 / 7 0 . 6 \left( + 1 5 . 9 / + 2 9 . 5 \right)$ and AppWorld<sub>TC</sub> TGC from 49.7 to 61.5 (as shown in Table 2). It achieves the best results on all reported $\mathrm { \ A p p W o r l d _ { T N } }$ metrics and the highest average across benchmarks at every consistency level, with larger gains at higher consistency levels (e.g., AppWorld<sub>TN</sub> $p ^ { 3 }$ TGC: 72.8 vs. 47.6). With Qwen2.5-32B-Instruct, DRACO raises AppWorld<sub>TN</sub> TGC/SGC from 35.7/17.3 to $6 2 . 9 / 4 2 . 3 \ : ( + 2 7 . 2 / + 2 5 . 0 )$ , the larger gain of the two base policy models, and closes most of the distance to the outcome-aware baseline SALT (Li et al., 2026b) for step-wise credit assignment (66.2/47.9). Note that SALT uses the ground-truth reward signal and their method needed special processing to be applied to AppWorld due to its continuous textual state and action space. DRACO also transfers to τ -bench without training, raising

SR from 15.8 to 20.4, despite using no verifier, gold answers, or reference trajectories.

Consistency and discovery. Table 2 reports $p ^ { 1 } -$ $p ^ { 3 }$ , while Figure 5 compares consistency with pass@k. On AppWorld<sub>TN</sub>, DRACO improves both discovery and consistency, with substantially larger gains in consistency: TGC $p ^ { 3 }$ increases by 25.2 points, whereas pass@3 rises by only 3.9. The untrained policy model could often solve tasks at least once; DRACO makes those successes reliable across repeated attempts. Retention $( p ^ { 3 } / p ^ { 1 } )$ shows the same pattern, with DRACO achieving the highest retention on both TGC and SGC.

## 4.3 Ablations

We ablate along three axes: whether the reward sees task outcomes at all, which of our two components is responsible for the outcome-blind regime’s gains, and whether the judge has to be a frontier model.

![](images/17f6d3a6f620fb8d37b42d78fb6f61b60d19d473d581b04649c6edeb81dca84e.jpg)

![](images/1cbf56913bfd3a0045e686e978f2b20c5b0973d126f7683fef7cca874700844b.jpg)

![](images/e7e65c1bf280cfae535d7a52f1faa1ad594b6d2c894863a0a572ea9b2cdd0b41.jpg)  
Figure 2: Performance vs. evaluation cost, one panel per benchmark; up and to the left is better. AppWorld panels plot TGC, τ -bench banking plots success rate; SGC follows the same ordering (Table 2). Cost is the US-dollar total for one pass over the full split at the rates of Section 4.1.3; the τ-bench axis also includes the GPT-5.4 user simulator and judge. Points are Qwen3.6-27B $p ^ { 1 }$ values from Table 2.

Outcome reward. The outcome-reward setting is trained on AppWorld’s unit tests instead of a judge using vanilla GRPO keeping the same base model and hyperparameters. Per Table 2, DRACO outperforms it by +5.3 TGC and +11.3 SGC on $\mathrm { \ A p p W o r l d _ { T N } }$ and +1.6 TGC and +2.4 SGC on $\mathrm { A p p W o r l d _ { T C } }$ . The margin widens with consistency: on $\mathrm { \ A p p W o r l d _ { T N } }$ it reaches +9.5 TGC and +13.7 SGC at $p ^ { 3 }$ , and AppWorld widens the same way (+3.5 TGC, +6.0 SGC). Training on random rewards has been reported to improve performance in some settings (Shao et al., 2026b). Doing so here reaches 74.0 TGC and 50.0 SGC on AppWorld , well below DRACO’s 85.3 and 70.6.

Rubric type and step credit. Per Table 2, DRACO leads every ablated setting on the $p ^ { 1 }$ average (55.7 against 52.0–53.3) and by a wider margin at $p ^ { 3 }$ (40.9 against 35.3–37.8). The four-way comparison locates that gain in the combination of the two components. On $\mathrm { \Delta A p p W o r l d _ { T N } . }$ , adding step credit to per-trajectory rubrics is worth +3.2 TGC and +5.7 SGC, and the two components together are worth +4.2 TGC and +10.7 SGC over the fixed rubric alone, growing to +8.1 TGC and $+ 1 4 . 3 \operatorname { S G C } \mathrm { a t } p ^ { 3 }$ . Either component on its own contributes far less (+0.8 and +1.0 TGC), which is what the mechanism predicts: step credit needs criteria specific enough to implicate particular steps, and a rubric written for the whole task distribution gives the attributor little to attribute. AppWorld<sub>TC</sub> sharpens the interaction into a sign change. There step credit on a fixed rubric costs 3.7 TGC at $p ^ { 3 }$ while on per-trajectory rubrics it adds +1.4 TGC, so the same intervention pays off precisely once the rubric is specific to the episode. The gains hold across task difficulty (Table 6): on $\mathrm { A p p W o r l d _ { T N } } .$ step credit on per-trajectory rubrics improves TGC at every level and is largest on medium tasks (+7.2 TGC, +14.6 SGC), and it lifts easy scenario completion by +8.2 SGC. All four settings also transfer to τ-bench, gaining between +2.9 and +4.6 SR over the untrained policy model.

Self-judge. Every setting above scores with a frozen frontier judge, which is the single largest cost in the pipeline. We replace it with the policy model judging its own rollouts, with thinking enabled in every judge phase; scoring is repeated k=3 times per trajectory, and a criterion counts as passed only if all three calls pass it, so a single lenient call cannot result in a pass. Appendix A.1 compares this judge with a variant that disables thinking and scores with a single call. This cuts the judge cost for 100 training steps from \$1607 to \$316, a 5.1× saving, and the saving holds across all five phases that issue judge calls (Figure 8 in the Appendix). On AppWorld<sub>TN</sub> it exceeds the outcome-aware reference, 81.1/62.7 TGC/SGC against its 80.0/59.3 and 65.5/38.1 against 63.3/38.1 at $p ^ { 3 }$ , and on τ- bench it achieves the best SR of any setting (21.1): a policy model grading its own rollouts reaches verifier-trained performance without a verifier. Replaying every scoring call through the policy checkpoint, with a single call and no thinking, shows the verdicts largely agree with the frontier judge’s (89.4% of 60,689 criterion-level verdicts, against 72.0% for a judge that passes everything), and that where they disagree the self-judge is lenient rather than noisy: it passes 30.4% of the criteria the frontier judge failed while failing only 1.3% of the ones it passed (Appendix B.1).

## 4.4 Analysis

We turn from scores to what the gains cost and how training produced them: evaluation cost and episode length (Figure 2, Table 3), rollout termination (Figure 3), and reward progression during training (Figure 4, and Figure 7 in Appendix A). Appendix B has further analysis.

Evaluation cost and episode length. All four rubric settings on both AppWorld splits in Figure 2, have higher accuracy gains with lower evaluation cost rather than Base. On AppWorld<sub>TN</sub>, DRACO reaches 85.3 TGC for \$8.27 against 69.4 at \$10.77 untrained, and on AppWorld<sub>TC</sub> 61.5 for \$38.03 against 49.7 at \$43.56. The untrained policy model is thus the most expensive split to evaluate despite scoring lowest. Appendix A.2 compares DRACO’s performance and evaluation cost with those of frontier models on AppWorld<sub>TN</sub>. Table 3 shows where that saving comes from. On Qwen3.6-27B, every rubric setting shortens the episode on both AppWorld splits, DRACO by the most (18.7 to 14.7 turns on AppWorld<sub>TN</sub>, 22.9 to 20.7 on AppWorld<sub>TC</sub>), so it is solving more tasks in fewer agent turns rather than buying accuracy with a longer rollout. On τ-bench the picture reverses: turns rise slightly for every setting and the gains cost more, which is what we would expect on a benchmark held out of training, where the untrained policy model’s shorter episodes reflect giving up earlier rather than finishing sooner.

<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>|AppWorldTN</td><td rowspan=1 colspan=1>|AppWorldTC</td><td rowspan=1 colspan=2>TB|Avg.</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-32B-Instruct</td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>16.4</td><td rowspan=1 colspan=1>23.8</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=1>20.6</td></tr><tr><td rowspan=1 colspan=1>DRACO (Ours)</td><td rowspan=1 colspan=1>19.6</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>27.1</td><td rowspan=1 colspan=1>22.6</td></tr><tr><td rowspan=1 colspan=5>Qwen3.6-27B</td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>18.7</td><td rowspan=1 colspan=1>22.9</td><td rowspan=1 colspan=1>27.7</td><td rowspan=1 colspan=1>23.1</td></tr><tr><td rowspan=1 colspan=1>Outcome reward</td><td rowspan=1 colspan=1>17.4</td><td rowspan=1 colspan=1>23.7</td><td rowspan=1 colspan=1>29.6</td><td rowspan=1 colspan=1>23.6</td></tr><tr><td rowspan=1 colspan=1>DRACO w/o Dyn. &amp; Cred.</td><td rowspan=1 colspan=1>15.9</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>30.1</td><td rowspan=1 colspan=1>22.4</td></tr><tr><td rowspan=2 colspan=1>DRACO w/o Dyn.DRACO w/o Cred.</td><td rowspan=1 colspan=1>15.6</td><td rowspan=1 colspan=1>21.6</td><td rowspan=1 colspan=1>29.3</td><td rowspan=1 colspan=1>22.2</td></tr><tr><td rowspan=1 colspan=1>16.4</td><td rowspan=1 colspan=1>22.2</td><td rowspan=1 colspan=1>30.1</td><td rowspan=1 colspan=1>22.9</td></tr><tr><td rowspan=2 colspan=1>DRACO (Ours)DRACO (Ours - self-judge)</td><td rowspan=1 colspan=1>14.7</td><td rowspan=1 colspan=1>20.7</td><td rowspan=1 colspan=1>27.9</td><td rowspan=1 colspan=1>21.1</td></tr><tr><td rowspan=1 colspan=1>17.4</td><td rowspan=1 colspan=1>24.6</td><td rowspan=1 colspan=1>28.9</td><td rowspan=1 colspan=1>23.6</td></tr></table>

Table 3: Average agent turns, lower is better. – = not evaluated on that split.

![](images/7c905d9637c7f44118b1c79ea883af4f2912abafe5803b8cb76f19ff544b7e58.jpg)

![](images/f118495afef1edb7424005ad2436b693f9f9e86366dc470923ef3bfeba61b5bf.jpg)  
Figure 3: Rollout termination modes across training steps for DRACO with and without step credit. Bars show the percentage of rollouts per termination mode (normal completion/give-up vs. length, turn, or server errors), independent of task correctness. We interpret broad trends rather than exact error distributions.

![](images/c8f2094daaec2dd0ec74d378ee626866bb924568e75f7ed7f54d794e24c896ab.jpg)  
Figure 4: Overall rubric pass rate across training steps. Each setting is evaluated on its training rubrics (dynamic rubrics vary per step). The dotted line shows DRACO re-scored on the static held-out rubric set. Figure 10 in Appendix A reports per-criterion pass rates.

How rollouts end. Figure 3 shows how each rollout terminated over the course of training. Both dynamic settings converge on the same behaviour: the agent almost always submits an answer by the end, and the modes where the episode simply ran out (response length exceeded, turn budget exhausted, an environment server failure, or no action taken at all) fade to negligible. These errors dominate early training and adding credit clears them sooner: dynamic alone spends first 30 steps oscillating, at times dropping below 30% submitted, whereas DRACO settles within roughly the first 40.

Static rubrics saturate; dynamic ones do not. Figure 4 shows the reward the policy model actually optimizes. Against static rubrics it saturates early, reaching the mid-90s within about 25 steps and staying there, so the two static settings are indistinguishable for most of training (94.8% and 95.8% mean) and the reward stops discriminating long before training ends. The dynamic settings sit lower and keep moving (66.9% and 74.2%). This is by design rather than a sign of worse rollouts: the generator is prompted for criteria at least one member of the group is likely to fail, and discriminative dropout then discards the rest (Section 3.2). Step credit helps here too, with DRACO better than DRACO w/o Cred. for nearly all of training. Furthermore, re-scoring DRACO’s rollouts against the fixed static rubrics they never trained on recovers 91.3%, close to the static runs’ own curve, so per-trajectory rubrics subsume the static criteria. Figure 4 shows on the weaker base policy model the pass rate climbs steeply from below 40% to around 75%, the largest change of any run in the figure; thus, the signal is not confined to a strong base policy model.

## 5 Conclusion

We study reinforcement learning for long-horizon tool-using agents in the outcome-blind setting, where ground-truth success signal is not available and rewards must come entirely from process criteria. We introduce two components: task-specific rubrics generated from policy model rollouts and merged at the group level, and a step-credit assignment rule that redistributes trajectory advantage using the judge’s per-criterion attributions while preserving its total magnitude and sign. Our method improves AppWorld’s TGC by 15.9 points on AppWorld<sub>TN</sub>, transfers zero-shot to τ−bench, and matches or exceeds a verifier-trained baseline with the same budget. Ablations show that steplevel credit assignment on top of per-trajectory rubrics is key; the main challenge is making evaluation criteria specific enough to support meaningful attribution. Since the judge defines the objective rather than estimating it, understanding the required judge quality is a natural direction for future work.

## 6 Limitations

A limitation inherent to using rubrics as rewards on unverifiable tasks is that there is no independent signal against which to check whether the criteria describe it faithfully. Validating the criteria would require human raters, which we leave to future work. We use a single frozen judge for most of our experiments, and do not report a chance-corrected measure of its per-criterion agreement with human annotators (Norman et al., 2026). A judge that is internally consistent can still be systematically wrong, and our experiments cannot separate the two.

The same gap appears one level down, in the attribution. We validate DRACO by end-task performance, which does not establish that credit lands on the right steps: a redistribution that credits the wrong steps can still improve a policy model, and one that credits the right steps can fail to.

Another characteristic of DRACO is that discriminative dropout makes the surviving criterion set a function of the sampled group: a criterion is kept only if some member failed it, so the same prompt can be scored against different criteria at different points in training. We do not measure the resulting variance; our repeated runs are inferencetime repeats of fixed checkpoints, so training-time variability is not characterized.

## 7 Ethics Statement

Both benchmarks, AppWorld (Trivedi et al., 2024) and τ-bench (Yao et al., 2024), are used under their released licenses and are fully simulated: every app, account and record is a synthetic fixture, so no real user data or live service is touched and no human subjects were involved. Because the reward is a set of natural-language criteria, the target of optimization is explicit and auditable, which we regard as a safety property; the corresponding risk is that any bias the judge carries is inherited by the policy with no verifier to catch it. Agents that use tools more competently are dual-use: the same capability that completes a booking or a refund can execute an unwanted transaction more effectively but our contribution is to the training signal rather than to autonomy or tool access, and all evaluation is sandboxed.

## References

Jose A Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brandstetter, and Sepp Hochreiter. 2019. RUDDER: Return decomposition for delayed rewards. In NeurIPS.

Kevin Chen, Marco Cusumano-Towner, Brody Huval, Aleksei Petrenko, Jackson Hamburger, Vladlen Koltun, and Philipp Krähenbühl. 2025. Reinforcement learning for long-horizon interactive llm agents. Preprint, arXiv:2502.01600.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM agent training. arXiv preprint arXiv:2505.10978.

Xin Guan, Xiaomeng Hu, Shen Huang, Zhenyi Wang, Bo Zhang, Zijian Li, Pengjun Xie, Bo Liu, and Jiuxin Cao. 2026. Evorubric: Self-evolving rubricdriven rl for open-ended generation. Preprint, arXiv:2605.29847.

Anna Harutyunyan, Will Dabney, Thomas Mesnard, Nicolas Heess, Mohammad Gheshlaghi Azar, Bilal Piot, Hado van Hasselt, Greg Wayne, Satinder Singh, Doina Precup, and Rémi Munos. 2019. Hindsight credit assignment. In NeurIPS.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In ICLR.

Yuxuan Jiang and Francis Ferraro. 2026. SCRIBE: Structured mid-level supervision for tool-using language models. arXiv preprint arXiv:2601.03555.

Gaotang Li, Bhavana Dalvi Mishra, Zifeng Wang, Jun Yan, Yanfei Chen, Chun-Liang Li, Long T. Le, Rujun Han, George Lee, Hanghang Tong, Chen-Yu Lee, and Tomas Pfister. 2026a. Rubricem: Meta-rl with rubric-guided policy decomposition beyond verifiable rewards. Preprint, arXiv:2605.10899.

Jiazheng Li, Yawei Wang, Qiaojing Yan, Yijun Tian, Zhichao Xu, Huan Song, Panpan Xu, and Lin Lee Cheong. 2026b. SALT: Step-level advantage assignment for long-horizon agents via trajectory graph. In Findings of the Association for Computational Linguistics: EACL 2026, pages 4709–4725, Rabat, Morocco. Association for Computational Linguistics.

Justin D. Norman, Michael U. Rivera, and D. Alex Hughes. 2026. Reliability without validity: A systematic, large-scale evaluation of LLM-as-a-judge models across agreement, consistency, and bias. arXiv preprint arXiv:2606.19544.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. YaRN: Efficient context window extension of large language models. Preprint, arXiv:2309.00071.

MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Bing Liu, Yunzhong He, and Afra Feyza Akyürek. 2025. Online rubrics elicitation from pairwise comparisons. arXiv preprint arXiv:2510.07284.

John Schulman. 2020. Approximating KL divergence. http://joschu.net/blog/kl-approx.html.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, and 2 others. 2026a. DR tulu: Reinforcement learning with evolving rubrics for deep research. In Forty-third International Conference on Machine Learning.

Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. 2026b. Spurious rewards: Rethinking training signals in RLVR. Preprint, arXiv:2506.10947.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y K Li, Y Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Haoyang Su and Ying Wen. 2026. Learning CLI agents with structured action credit under selective observation. Preprint, arXiv:2605.08013.

Hui-Ze Tan, Xiao-Wen Yang, Hao Chen, Jie-Jing Shao, Yi Wen, Yuteng Shen, Weihong Luo, Xiku Du, Lan-Zhe Guo, and Yu-Feng Li. 2026. Hindsight credit assignment for long-horizon LLM agents. arXiv preprint arXiv:2603.08754.

Leitian Tao, Baolin Peng, Wenlin Yao, Tao Ge, Hao Cheng, Mike Hang Wang, Jianfeng Gao, and Sharon Li. 2026. TRACE: Turn-level reward assignment via credit estimation. arXiv preprint arXiv:2607.13988.

Qwen Team. 2024. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Zihang Tian, Jingsen Zhang, Rui Li, Xiaohe Bo, Yuanzi Li, and Xu Chen. 2026. ARCO: Adaptive rubric with co-evolution for multi-step LLM-based agents. arXiv preprint arXiv:2606.21262.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In ACL.

Vijay Viswanathan, Shiqi Wang, Devamanyu Hazarika, Chirag Nagpal, Tongshuang Wu, Graham Neubig, and Yuning Mao. 2026. Discretizing reward models. arXiv preprint arXiv:2606.21795.

Binghai Wang, Chenlong Zhang, Dayiheng Liu, Jiajun Zhang, Jiawei Chen, Mingze Li, Mouxiang Chen, Rongyao Fang, Siyuan Zhang, Xuwu Wang, Yuheng Jing, Zeyao Ma, and Zeyu Cui. 2026a. The verification horizon: No silver bullet for coding agent rewards. arXiv preprint arXiv:2606.26300.

Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, and Wenjie Li. 2025. SPA-RL: Reinforcing LLM agents via stepwise progress attribution. arXiv preprint arXiv:2505.20732.

Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. 2026b. OpenClaw-RL: Train any agent simply by talking. arXiv preprint arXiv:2603.10165.

Ran Xu, Tianci Liu, Zihan Dong, Tony Yu, Ilgee Hong, Carl Yang, Linjun Zhang, Tuo Zhao, and Haoyu Wang. 2026a. Alternating reinforcement learning for rubric-based reward modeling in non-verifiable LLM post-training. arXiv preprint arXiv:2602.01511.

Tianze Xu, Yanzhao Zheng, Pengrui Lu, Lyumanshan Ye, Yong Wu, Zhentao Zhang, Yuanqiang Yu, Chao Ma, Jihuai Zhu, Pengfei Liu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, and Gang Yu. 2026b. Rubrics to tokens: Bridging response-level rubrics and token-level rewards in instruction following tasks. Preprint, arXiv:2604.02795.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for toolagent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045.

Woongyeng Yeo, Yumin Choi, Taekyung Ki, and Sung Ju Hwang. 2026. Hint-sd: Targeted hindsight self-distillation for long-horizon agents. Preprint, arXiv:2605.17873.

Yuanqing Yu, Zhefan Wang, Weizhi Ma, Shuai Wang, Chuhan Wu, Zhiqiang Guo, and Min Zhang. 2025. Steptool: Enhancing multi-step tool usage in llms via step-grained reinforcement learning. In Proceedings

of the 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, page 3952–3962. ACM.

Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, Zhaoyang Liu, Bolin Ding, and Jingren Zhou. 2025. AgentEvolver: Towards efficient self-evolving agent system. arXiv preprint arXiv:2511.10395.

Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, and Lifeng Jin. 2025. Chasing the tail: Effective rubric-based reward modeling for large language model post-training. arXiv preprint arXiv:2509.21500.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. 2024. WebArena: A realistic web environment for building autonomous agents. In ICLR. ArXiv:2307.13854.

Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, and Xian Li. 2025. SWEET-RL: Training multi-turn LLM agents on collaborative reasoning tasks. arXiv preprint arXiv:2503.15478.

## A Additional Results

This section holds breakdowns that the main text summarizes but does not have room to show. Tables 4 and 5 restate Tables 2 and 3 with the standard deviation across the three checkpoints each reported mean averages. Table 6 splits task success by AppWorld’s own difficulty label on the same basis, and is where the by-difficulty reading in Section 4.3 comes from. Figure 5 plots pass<sup>k</sup> against pass@k for every setting, separating consistency from discovery (Section 4.2). Figure 7 tracks TGC on the training tasks over the course of training, an optimization signal rather than a held-out metric. Figure 8 breaks the training-time judge cost down by pipeline phase for a frontier judge and for the self-judge variant. Figure 10 opens up the aggregate static pass rate of Figure 4 one criterion at a time: most of the 21 static criteria are already near ceiling within 20 steps, and the aggregate curve’s early rise comes from a few initially-hard criteria, chief among them Protects secret values, which starts below 20% and is where step credit shows its clearest per-criterion advantage.

## A.1 Self-judge variants

The self-judge of Section 4.3 enables thinking in every judge phase and repeats scoring k=3 times per trajectory, counting a criterion as passed only if all three calls pass it. Table 7 compares it against an earlier variant with neither change: no thinking, and a single scoring call. The unanimity rule targets the self-judge’s dominant error, lenient false passes (Appendix B.1).

## A.2 Comparison with frontier models

Figure 6 and Table 8 compare DRACO with six frontier and open-weight models on AppWorld<sub>TN</sub>, plotting TGC and SGC against evaluation cost as in Figure 2. The frontier models are run once each, with no training on AppWorld.<sup>3</sup> DRACO is the Qwen3.6-27B frontier-judge setting of Table 2, shown next to its untrained policy.

Our training brings a 27B model close to models one to two orders of magnitude larger, after 100 steps on outcome-blind rewards alone. It lifts the policy from 69.4/41.1 TGC/SGC to 85.3/70.6, and lowers its evaluation cost from \$10.77 to \$8.27. That places DRACO within 4.6 TGC of claude-sonnet-4-6 and 8.2 of claude-opus-4-7, at 44% and 25% of their cost, and ahead of DeepSeek-V4-Flash (284B) and gpt-oss-120b by 14.5 and 41.3 TGC. Only Kimi-K2.7-Code, a 1T model, is both cheaper and stronger. The largest gain is on SGC, where every task in a scenario must pass: training raises it by +29.5 points, from below DeepSeek-V4-Flash to within 11.5 of Kimi-K2.7-Code.

## B Additional Analysis

## B.1 Self-judge vs. frontier judge

The self-judge variant replaces GPT-5.4 with the policy model. To measure what that costs we replay the run’s judge calls rather than compare two trained runs, which would confound the judge with the policy it shaped. At each step N we re-send that step’s calls to the step-(N−1) checkpoint, using the archived prompt verbatim, so the only thing that changes is which model answers. GPT-5.4’s output is the reference: every number below measures agreement with it, not correctness. This replay predates the k=3-with-thinking judge used for the self-judge row of Table 2; every self-judge number in this section is a single call with no thinking, so it is a lower bound on what that stronger judge agrees on.

<table><tr><td rowspan="3">Setting</td><td colspan="4"> $\mathbf { A p p W o r l d } _ { \mathrm { T N } }$ </td><td colspan="4"> $\mathbf { A p p W o r l d } _ { \mathrm { T C } }$ </td><td colspan="2"> $\tau _ { \mathrm { B } }$ </td><td rowspan="2" colspan="2">Average</td></tr><tr><td colspan="2">TGC</td><td colspan="2">SGC</td><td colspan="2">TGC</td><td colspan="2">SGC</td><td colspan="2">SR</td></tr><tr><td></td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 3 }$ </td><td></td><td> $p ^ { 1 }$   $p ^ { 3 }$ </td></tr><tr><td>Qwen2.5-32B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td> $3 5 . 7 _ { \pm 4 . 6 }$ </td><td>23.2</td><td> $1 7 . 3 { \scriptstyle \pm 3 . 7 }$ </td><td>8.9</td><td> $1 7 . 3 { \scriptstyle \pm 0 . 6 }$ </td><td> $7 . 4$ </td><td> $5 . 8 { \scriptstyle \pm 0 . 7 }$ </td><td>2.2</td><td> $3 . 4 { \scriptstyle \pm 0 . 6 }$ </td><td>1.0</td><td></td><td>15.9 8.5</td></tr><tr><td>SALT (Outcome-aware)</td><td> $6 6 . 2 { \scriptstyle \pm 2 . 5 }$ </td><td></td><td> $4 7 . 9 _ { \pm 4 . 1 }$ </td><td></td><td> $3 6 . 8 { \scriptstyle \pm 1 . 5 }$ </td><td></td><td> $2 0 . 9 { \scriptstyle \pm 1 . 8 }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DRACO (Ours)</td><td> $6 2 . 9 { \scriptstyle \pm 3 . 8 }$ </td><td> $4 6 . 0 { \scriptstyle \pm 7 . 9 }$ </td><td> $4 2 . 3 { \scriptstyle \pm 7 . 2 }$ </td><td> $2 6 . 8 { \scriptstyle \pm 6 . 4 }$ </td><td> $3 4 . 0 { \scriptstyle \pm 1 . 2 }$ </td><td> $2 0 . 2 { \scriptstyle \pm 2 . 5 }$ </td><td> $1 9 . 2 { \scriptstyle \pm 1 . 7 }$ </td><td> $1 0 . 3 { \scriptstyle \pm 3 . 7 }$ </td><td> $4 . 0 { \scriptstyle \pm 1 . 0 }$ </td><td>1.5</td><td></td><td>32.5 21.0</td></tr></table>

<table><tr><td colspan="10">Qwen3.6-27B</td></tr><tr><td>Base</td><td> $6 9 . 4 _ { \pm 6 . 4 }$ </td><td> $4 7 . 6$ </td><td> $4 1 . 1 _ { \pm 1 0 . 9 }$ </td><td> $2 1 . 4$ </td><td> $4 9 . 7 _ { \pm 1 . 2 }$ </td><td> $3 1 . 7$ </td><td> $3 0 . 2 { \scriptstyle \pm 1 . 2 }$ </td><td>12.9</td><td> $1 5 . 8 { \scriptstyle \pm 2 . 1 }$ </td><td> $7 . 2$ </td><td>41.2 24.2</td></tr><tr><td>Outcome reward</td><td> $8 0 . 0 { \scriptstyle \pm 1 . 9 }$  </td><td> $6 3 . 3 { \scriptstyle \pm 3 . 6 }$  </td><td> $5 9 . 3 { \scriptstyle \pm 2 . 5 }$  </td><td> $3 8 . 1 { \scriptstyle \pm 3 . 7 }$ </td><td> $5 9 . 9 { \scriptstyle \pm 1 . 2 }$  </td><td> $4 0 . 4 _ { \pm 2 . 4 }$  </td><td> $3 8 . 3 { \scriptstyle \pm 2 . 4 }$  </td><td> $2 1 . 3 { \scriptstyle \pm 2 . 3 }$ </td><td> $1 7 . 6 { \scriptstyle \pm 0 . 9 }$  </td><td> $9 . 3 { \scriptstyle \pm 0 . 0 }$ </td><td>51.034.5</td></tr><tr><td>DRACO w/o Dyn. &amp; Cred.</td><td> $8 1 . 1 { \scriptstyle \pm 2 . 3 }$ </td><td> $6 4 . 7 _ { \pm 4 . 5 }$ </td><td> $5 9 . 9 { \scriptstyle \pm 5 . 5 }$ </td><td> $3 7 . 5 { \scriptstyle \pm 7 . 8 }$ </td><td> $6 0 . 5 { \scriptstyle \pm 1 . 0 }$ </td><td> ${ \bf 4 5 . 6 { \scriptstyle \pm 0 . 5 } }$ </td><td> $4 2 . 4 _ { \pm 0 . 5 }$ </td><td> $2 7 . 3 _ { \pm 3 . 3 }$ </td><td> $1 9 . 4 { \scriptstyle \pm 2 . 2 }$   $8 . 9 { \scriptstyle \pm 1 . 6 }$ </td><td></td><td>52.7 36.8</td></tr><tr><td>DRACO w/o Dyn.</td><td> $8 1 . 9 { \scriptstyle \pm 2 . 2 }$ </td><td> $6 5 . 5 { \scriptstyle \pm 3 . 7 }$ </td><td> $6 0 . 7 _ { \pm 3 . 7 }$ </td><td> $3 9 . 3 _ { \pm 4 . 7 }$ </td><td> $5 9 . 4 { \scriptstyle \pm 0 . 7 }$ </td><td> $4 1 . 9 { \scriptstyle \pm 1 . 0 }$ </td><td> $3 9 . 2 { \scriptstyle \pm 1 . 6 }$ </td><td> $2 1 . 8 { \scriptstyle \pm 2 . 7 }$ </td><td> $1 8 . 7 { \scriptstyle \pm 1 . 5 }$ </td><td> $7 . 9 { \scriptstyle \pm 1 . 2 }$ </td><td>52.0 35.3</td></tr><tr><td>DRACO w/o Cred.</td><td> $8 2 . 1 _ { \pm 1 . 4 }$ </td><td> $6 6 . 3 { \scriptstyle \pm 3 . 4 }$ </td><td> $6 4 . 9 { \scriptstyle \pm 3 . 0 }$ </td><td>44.6±3.1</td><td> $5 9 . 3 _ { \pm 1 . 0 }$ </td><td> $4 2 . 5 { \scriptstyle \pm 1 . 4 }$ </td><td> $4 0 . 6 { \scriptstyle \pm 1 . 8 }$ </td><td> $2 6 . 4 _ { \pm 3 . 0 }$ </td><td> $1 9 . 7 _ { \pm 0 . 5 }$ </td><td> $9 . 3 { \scriptstyle \pm 0 . 0 }$ </td><td>53.3 37.8</td></tr><tr><td>DRACO (Ours)</td><td> ${ \bf 8 5 . 3 _ { \pm 1 . 5 } }$ </td><td> $7 2 . 8 { \scriptstyle \pm 3 . 8 }$ </td><td> $7 0 . 6 { \scriptstyle \pm 2 . 4 }$ </td><td>51.8±4.7</td><td> ${ \bf 6 1 . 5 _ { \pm 1 . 0 } }$ </td><td> $4 3 . 9 2 1 . 6$  </td><td> $4 0 . 7 _ { \pm 1 . 4 }$ </td><td> $2 7 . 3 { \scriptstyle \pm 1 . 2 }$ </td><td> $2 0 . 4 _ { \pm 1 . 0 }$   $8 . 6 { \scriptstyle \pm 1 . 2 }$ </td><td></td><td>55.7 40.9</td></tr><tr><td>DRACO (Ours - self-judge)</td><td> $8 1 . 1 { \scriptstyle \pm 2 . 2 }$ </td><td> $6 5 . 5 { \scriptstyle \pm 6 . 4 }$ </td><td> $6 2 . 7 _ { \pm 3 . 1 }$ </td><td> $3 8 . 1 { \scriptstyle \pm 5 . 5 }$ </td><td> $6 1 . 0 { \scriptstyle \pm 0 . 6 }$ </td><td> $3 6 . 9 { \scriptstyle \pm 1 . 5 }$ </td><td> $3 4 . 7 _ { \pm 2 . 7 }$ </td><td> $1 6 . 1 { \scriptstyle \pm 2 . 2 }$ </td><td> $2 1 . 1 { \pm } 1 . 4$ </td><td> $1 1 . 7 _ { \pm 0 . 6 }$ </td><td>52.1 33.7</td></tr></table>

Table 4: Task success and consistency with variability, the counterpart of Table 2. Means are identical to Table 2, and best values per column are in bold as there. For trained rows, subscripts give the standard deviation over the three checkpoints each mean averages $( 9 0 / 9 5 / 1 0 0$ steps for Qwen3.6-27B, 65/70/75 for Qwen2.5-32B-Instruct), with three evaluation runs per checkpoint. Base rows are single-checkpoint, so their subscripts instead give the standard deviation over three independent evaluation runs; SALT reports that same quantity, and its published values are quoted. Only $p ^ { 1 }$ admits an across-run spread, since $\mathrm { p a s s } ^ { k }$ for $k { \geq } 2$ is a joint statistic over all three runs. We report $p ^ { 1 }$ and $p ^ { 3 }$ here; $p ^ { 2 }$ is in Table 2. Average columns are row-level summaries and are reproduced without subscripts. – = not evaluated on that split, or not reported by the cited work.

![](images/84c8a443029add582796652339ab4d7ed5182acdee14e7acecd95d78f403f886.jpg)  
Figure 5: $\mathbf { p a s s } ^ { k }$ vs. pass@k over $n { = } 3$ evaluation runs, in %. Solid: $\mathrm { p a s s } ^ { k }$ , all k runs succeed. Dashed: pass@k, at least one succeeds. The two coincide at k=1 by construction. Values and protocol follow Table 2. All panels are Qwen3. $6 - 2 7 \mathsf { B } ;$ axes do not start at zero.

## B.1.1 Annotation: applying a given rubric

The judge does three jobs, and the self-model is good at one of them (Figure 9). It applies a rubric nearly as well as GPT-5.4. It writes different criteria. And when it merges criteria into the set a group is scored on, it keeps the right ones but cannot discard the rest.

This is the easy job and the self-model does it well. Over 100 steps, 9,590 rollouts and 60,689 verdicts, it matches GPT-5.4 on 89.4% of them (Figure 9a). A judge that answers “pass” every time would match 72.0%, so the agreement is not just an artifact of pass-skewed labels. Verdicts the selfmodel failed to produce count as disagreements: it left 1.3% of them unanswered, and GPT-5.4 left none.

Qwen3.6-27B (untrained) Claude Sonnet 4.6 gpt-oss-120b (117B, 5.1B act.)   
DRACO (Qwen3.6-27B) Kimi-K2.7-Code (1T, 32B act.) Gemini 2.5 Flash-Lite   
Claude Opus 4.7 DeepSeek-V4-Flash (284B, 13B act.)

![](images/6c7368258d03cf88669406a51f5d49a8a5df507684e4b7d6ca4269e31028f40a.jpg)

![](images/3922cd219255f9b6178d6bdbe8fff9ef6c1ee1862d927f0771d98121c217b459.jpg)  
Figure 6: Frontier models vs. DRACO on AppWorld . TGC (left) and SGC (right) against the cost of one pass over the 168 tasks; up and to the left is better. Circles: frontier models, one run each. Star: DRACO on Qwen3.6-27B with the frontier judge. Hollow circle: the same policy untrained. Brackets give parameter counts (total, active) for open-weight models; closed models publish none. Values are in Table 8.

<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>|AppWorldTN |</td><td rowspan=1 colspan=1>AppWorldTC |</td><td rowspan=1 colspan=3>TB  |Avg.</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-32B-Instruct</td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1> $1 6 . 4 _ { \pm 0 . 4 }$ </td><td rowspan=1 colspan=1> $2 3 . 8 { \scriptstyle \pm 0 . 5 }$ </td><td rowspan=1 colspan=2> $2 1 . 5 { \scriptstyle \pm 0 . 7 }$ </td><td rowspan=1 colspan=1>20.6</td></tr><tr><td rowspan=1 colspan=1>DRACO (Ours)</td><td rowspan=1 colspan=1> $1 9 . 6 { \scriptstyle \pm 0 . 7 }$ </td><td rowspan=1 colspan=1> $2 1 . 0 { \scriptstyle \pm 0 . 6 }$ </td><td rowspan=1 colspan=2> $2 7 . 1 _ { \pm 1 . 1 }$ </td><td rowspan=1 colspan=1>22.6</td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-27B</td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1> $1 8 . 7 _ { \pm 1 . 3 }$ </td><td rowspan=1 colspan=1> $2 2 . 9 { \scriptstyle \pm 0 . 4 }$ </td><td rowspan=1 colspan=2> $2 7 . 7 _ { \pm 0 . 5 }$ </td><td rowspan=1 colspan=1>23.1</td></tr><tr><td rowspan=1 colspan=1>Outcome reward</td><td rowspan=1 colspan=1> $1 7 . 4 _ { \pm 1 . 0 }$ </td><td rowspan=1 colspan=1> $2 3 . 7 _ { \pm 1 . 2 }$ </td><td rowspan=1 colspan=2> $2 9 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td rowspan=1 colspan=1>23.6</td></tr><tr><td rowspan=1 colspan=1>DRACO w/o Dyn. &amp; Cred.</td><td rowspan=1 colspan=1> $1 5 . 9 _ { \pm 0 . 9 }$ </td><td rowspan=2 colspan=1> $2 1 . 2 _ { \pm 0 . 4 }$  $2 1 . 6 { \scriptstyle \pm 0 . 5 }$ </td><td rowspan=2 colspan=2> $3 0 . 1 _ { \pm 0 . 2 }$  $2 9 . 3 { \scriptstyle \pm 0 . 1 }$ </td><td rowspan=2 colspan=1>22.422.2</td></tr><tr><td rowspan=2 colspan=1>DRACO w/o Dyn.DRACO w/o Cred.</td><td rowspan=1 colspan=1> $1 5 . 6 { \scriptstyle \pm 0 . 7 }$ </td><td rowspan=1 colspan=1>29.3±</td></tr><tr><td rowspan=1 colspan=1> $1 6 . 4 _ { \pm 0 . 2 }$ </td><td rowspan=1 colspan=1> $2 2 . 2 { \scriptstyle \pm 0 . 2 }$ </td><td rowspan=1 colspan=2> $3 0 . 1 { \pm } 0 . 4$ </td><td rowspan=1 colspan=1>22.9</td></tr><tr><td rowspan=2 colspan=1>DRACO (Ours)DRACO (Ours - self-judge)</td><td rowspan=1 colspan=1> $\mathbf { 1 4 . 7 _ { \pm 0 . 5 } }$ </td><td rowspan=1 colspan=1> ${ \bf 2 0 . 7 _ { \pm 0 . 2 } }$ </td><td rowspan=1 colspan=2> $2 7 . 9 _ { \pm 0 . 4 }$ </td><td rowspan=2 colspan=1>21.123.6</td></tr><tr><td rowspan=1 colspan=1> $1 7 . 4 { \scriptstyle \pm 0 . 2 }$ </td><td rowspan=1 colspan=1> $2 4 . 6 { \scriptstyle \pm 0 . 2 }$ </td><td rowspan=1 colspan=2> $2 8 . 9 { \scriptstyle \pm 0 . 7 }$ </td></tr></table>

Table 5: Average agent turns with variability, the counterpart of Table 3; lower is better. Means are identical to Table 3, and best values per column are in bold as there. For trained rows, subscripts give the standard deviation over the three checkpoints each mean averages, with three evaluation runs per checkpoint. Base rows are single-checkpoint, so their subscripts instead give the standard deviation over three independent evaluation runs. Avg. is a row-level summary across benchmarks and is reproduced without a subscript. – = not evaluated on that split.

The errors run one way. The self-model passes 30.4% of the criteria GPT-5.4 failed (4,817 of 15,840), but fails only 1.3% of those it passed (560 of 43,706) — a lenient error is about 24 times more likely than a strict one. That is the better failure mode to have, because passing a criterion wrongly narrows the reward spread within a rollout group rather than reordering the group, and GRPO learns from that spread. But leniency grows over training: the false-pass rate rises from 22.1% over the first quarter to 34.9% over the last $\scriptstyle ( p = 2 \mathrm { e } - 6 )$ , while the false-fail rate stays flat and below 5% at every step (p=0.34).

![](images/a5601408eb298bc4ec1a0eab60c9f202f875b20eff8ba60f232e96649845db65.jpg)  
Figure 7: Train TGC vs. training step This is TGC on the training tasks, for post-hoc analysis, and it is not part of any reward.

Agreement itself barely moves (90.8% to 89.0% between the first and last quarter). We attribute that small drop to the criteria rather than to the judge: over the same span the criteria per rollout fall from 7.1 to 5.8, and the share GPT-5.4 itself fails falls from 31.5% to 23.8%, so later steps ask for finer calls over fewer failures. Both are properties of the generator and of the frontier judge’s own labels. (The figure’s legend averages over steps; the totals here pool every verdict, so the two differ by up to two points.)

<table><tr><td rowspan="3">Setting</td><td colspan="6">AppWorldTN</td><td colspan="6">AppWorldTC</td></tr><tr><td colspan="2">Easy (57)</td><td colspan="2">Medium (48)</td><td colspan="2">Hard (63)</td><td colspan="2">Easy (72)</td><td colspan="2">Medium (150)</td><td colspan="2">Hard (195)</td></tr><tr><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td><td>TGC</td><td>SGC</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td> $6 4 . 3 { \scriptstyle \pm 8 . 8 }$ </td><td> $3 8 . 6 _ { \pm 1 3 . 2 }$ </td><td> $3 2 . 6 { \scriptstyle \pm 4 . 8 }$ </td><td> $8 . 3 { \scriptstyle \pm 9 . 5 }$ </td><td> $1 2 . 2 { \scriptstyle \pm 3 . 3 }$ </td><td> $4 . 8 { \scriptstyle \pm 0 . 0 }$ </td><td> $5 0 . 5 { \scriptstyle \pm 2 . 9 }$ </td><td> $2 3 . 6 { \scriptstyle \pm 2 . 4 }$ </td><td>15.1±0.8</td><td> $4 . 0 { \scriptstyle \pm 2 . 0 }$ </td><td> $6 . 7 \pm 0 . 0$ </td><td> $0 . 5 { \scriptstyle \pm 0 . 9 }$ </td></tr><tr><td>DRACO (Ours)</td><td> $8 3 . 2 { \scriptstyle \pm 2 . 9 }$ </td><td> $6 6 . 7 _ { \pm 5 . 3 }$ </td><td> $6 6 . 9 2 7 . 6 $ </td><td> $4 4 . 4 _ { \pm 1 1 . 5 }$ </td><td> $4 1 . 4 _ { \pm 2 . 1 }$ </td><td> $1 8 . 5 { \scriptstyle \pm 8 . 0 }$ </td><td> $7 7 . 5 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 7 . 4 _ { \pm 5 . 3 }$ </td><td> $2 2 . 6 { \scriptstyle \pm 2 . 8 }$ </td><td> $9 . 3 { \scriptstyle \pm 1 . 3 }$ </td><td> $2 6 . 7 _ { \pm 0 . 9 }$ </td><td> $1 2 . 7 _ { \pm 2 . 6 }$ </td></tr><tr><td>Qwen3.6-27B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td> $\mid 8 4 . 2 _ { \pm 5 . 3 }$ </td><td> $6 6 . 7 _ { \pm 1 1 . 0 }$ </td><td> $6 3 . 2 { \scriptstyle \pm 6 . 7 }$ </td><td> $2 9 . 2 { \scriptstyle \pm 9 . 5 }$ </td><td> $6 0 . 8 { \scriptstyle \pm 7 . 8 }$ </td><td> $2 7 . 0 _ { \pm 1 4 . 5 }$ </td><td></td><td> $7 3 . 1 _ { \pm 2 . 1 } 5 5 . 6 _ { \pm 6 . 4 }$ </td><td> $4 2 . 0 _ { \pm 4 . 1 }$ </td><td> $2 7 . 3 { \scriptstyle \pm 3 . 1 }$ </td><td> $4 7 . 0 { \scriptstyle \pm 1 . 3 }$ </td><td> $2 3 . 1 _ { \pm 4 . 1 }$ </td></tr><tr><td>Outcome reward</td><td> $9 1 . 6 { \scriptstyle \pm 0 . 3 }$  </td><td> $8 2 . 5 { \scriptstyle \pm 1 . 8 }$  1</td><td> $8 0 . 5 { \scriptstyle \pm 3 . 7 }$ </td><td>54.8±4.8</td><td> $6 9 . 1 _ { \pm 3 . 6 }$  </td><td> $4 1 . 8 { \scriptstyle \pm 5 . 6 }$ </td><td> $8 3 . 8 { \scriptstyle \pm 1 . 2 }$  </td><td> $6 8 . 1 { \scriptstyle \pm 1 . 4 }$ </td><td> $5 1 . 4 _ { \pm 0 . 1 }$ </td><td> $3 3 . 1 _ { \pm 0 . 4 }$ </td><td>_  $5 7 . 6 { \scriptstyle \pm 2 . 6 }$ </td><td> $3 1 . 3 { \scriptstyle \pm 5 . 1 }$ </td></tr><tr><td>DRACO w/o Dyn. &amp; Cred.</td><td> $9 1 . 4 { \scriptstyle \pm 1 . 2 }$ </td><td> $8 1 . 3 { \scriptstyle \pm 1 . 0 }$ </td><td> $8 1 . 0 { \scriptstyle \pm 2 . 4 }$ </td><td> $5 3 . 5 { \scriptstyle \pm 7 . 3 }$ </td><td> $7 1 . 8 { \scriptstyle \pm 3 . 2 }$ </td><td> $4 5 . 5 { \scriptstyle \pm 8 . 7 }$ </td><td> $7 8 . 2 { \scriptstyle \pm 2 . 1 }$ </td><td> $6 4 . 8 { \scriptstyle \pm 4 . 5 }$ </td><td> $5 1 . 4 _ { \pm 1 . 4 }$ </td><td> ${ \bf 3 6 . 7 \pm 1 . 3 }$ </td><td> $6 0 . 9 { \scriptstyle \pm 0 . 3 }$ </td><td> $3 8 . 5 { \scriptstyle \pm 1 . 5 }$ </td></tr><tr><td>DRACO w/o Dyn.</td><td> $9 1 . 8 _ { \pm 3 . 8 }$ </td><td> $8 0 . 1 _ { \pm 1 0 . 0 }$ </td><td> $8 4 . 7 _ { \pm 1 . 2 }$ </td><td> $6 0 . 4 _ { \pm 5 . 5 }$ </td><td> $7 0 . 9 _ { \pm 3 . 2 }$ </td><td> $4 3 . 4 _ { \pm 1 . 8 }$ </td><td></td><td> $7 7 . 3 _ { \pm 0 . 5 } 6 1 . 1 _ { \pm 3 . 7 }$ </td><td> $5 1 . 3 _ { \pm 2 . 1 }$ </td><td> $3 4 . 4 _ { \pm 3 . 3 }$ </td><td> $5 9 . 1 _ { \pm 1 . 0 }$ </td><td> $3 4 . 9 _ { \pm 1 . 8 }$ </td></tr><tr><td>DRACO w/o Cred.</td><td> $9 2 . 0 { \scriptstyle \pm 1 . 8 }$ </td><td> $8 1 . 3 { \scriptstyle \pm 3 . 7 }$ </td><td> $8 1 . 2 { \scriptstyle \pm 2 . 1 }$ </td><td> $5 8 . 3 { \scriptstyle \pm 4 . 2 }$ </td><td> $7 3 . 7 _ { \pm 1 . 2 }$ </td><td> ${ \bar { 5 } } 5 . 0 _ { \pm 3 . 3 }$ </td><td></td><td> $7 7 . 9 _ { \pm 3 . 7 } 6 1 . 6 { \scriptstyle \pm 2 . 1 }$ </td><td> $4 8 . 2 _ { \pm 1 . 0 }$ </td><td> $3 2 . 9 { \scriptstyle \pm 2 . 3 }$ </td><td> $6 1 . 0 { \scriptstyle \pm 0 . 5 }$ </td><td> $3 8 . 8 { \scriptstyle \pm 1 . 6 }$ </td></tr><tr><td>DRACO (Ours)</td><td> ${ \bf 9 3 . 4 } _ { \pm 1 . 2 }$  </td><td> $\mathbf { 8 9 . 5 } _ { \pm 3 . 0 }$ </td><td> ${ \bf 8 8 . 4 _ { \pm 1 . 4 } }$  </td><td> $\mathbf { 7 2 . 9 _ { \pm 4 . 2 } }$ </td><td> $7 5 . 7 _ { \pm 3 . 2 }$ </td><td> $5 1 . 9 { \scriptstyle \pm 4 . 6 }$ </td><td> ${ \bf 8 1 . 6 { \scriptstyle \pm 2 . 5 } }$ </td><td> ${ \bf 6 5 . 3 2 5 . 6 }$ </td><td> $5 0 . 6 { \scriptstyle \pm 2 . 8 }$ </td><td> $2 9 . 6 { \scriptstyle \pm 3 . 2 }$ </td><td> ${ \bf 6 2 . 4 _ { \pm 0 . 6 } }$ </td><td>40.2±0.6</td></tr><tr><td>DRACO (Ours - self-judge)</td><td> $9 0 . 3 { \scriptstyle \pm 3 . 8 }$ </td><td> $7 9 . 5 { \scriptstyle \pm 3 . 7 }$ </td><td> $8 4 . 5 { \scriptstyle \pm 4 . 9 }$ </td><td> $6 2 . 5 { \scriptstyle \pm 6 . 3 }$ </td><td> $7 0 . 2 { \scriptstyle \pm 1 . 3 }$ </td><td> $4 7 . 6 { \scriptstyle \pm 2 . 8 }$ </td><td> $7 7 . 9 { \scriptstyle \pm 3 . 3 }$ </td><td> $5 3 . 2 { \scriptstyle \pm 3 . 5 }$ </td><td> ${ \pm 2 . 8 \mathrm { _ { \pm 1 . 6 } } }$ </td><td> $2 7 . 6 { \scriptstyle \pm 2 . 8 }$ </td><td> $6 1 . 1 { \scriptstyle \pm 1 . 4 }$ </td><td> $3 3 . 3 { \scriptstyle \pm 3 . 6 }$ </td></tr></table>

Table 6: Task-success by difficulty on both AppWorld test splits, in %, by the AppWorld metadata label and scored against each stratum’s own denominator (counts in the header). Entries are the $p ^ { 1 }$ level of Table 2 and follow its conventions throughout. For trained rows, subscripts give the standard deviation over the three checkpoints each mean averages, with three evaluation runs per checkpoint. Base rows are single-checkpoint, so their subscripts instead give the standard deviation over three independent evaluation runs. – = not evaluated on that split.
<table><tr><td rowspan="3">Setting</td><td colspan="6">AppWorldTN</td><td colspan="8">AppWorldTC</td><td colspan="4">TB</td><td rowspan="2"></td></tr><tr><td colspan="3">TGC</td><td colspan="3">SGC</td><td colspan="3">TGC</td><td colspan="3"></td><td colspan="3">SR</td><td colspan="3">Average</td></tr><tr><td> $\overline { { { p } ^ { 1 } } }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td>2</td><td></td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { \overline { { 1 } } }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $p ^ { 1 }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td><td> $\overline { { { p } ^ { 1 } } }$ </td><td> $p ^ { 2 }$ </td><td> $p ^ { 3 }$ </td></tr><tr><td>DRACO (self-judge)</td><td>81.1</td><td>72.1</td><td>65.5</td><td>62.7</td><td>48.2</td><td>38.1</td><td>61.0</td><td>45.6</td><td>36.9</td><td>34.7</td><td>21.7</td><td>16.1</td><td>21.1</td><td></td><td>14.7</td><td>11.7</td><td>52.1</td><td>40.5</td><td>33.7</td></tr><tr><td>DRACO (self-judge w/o Think.  $\& k = 3 )$ </td><td>79.3</td><td>68.2</td><td>60.9</td><td>59.3</td><td>45.6</td><td>38.1</td><td></td><td>54.3</td><td>43.0</td><td>36.7</td><td>36.2</td><td>25.0</td><td>18.0</td><td>18.6</td><td>11.7</td><td>9.6</td><td>49.5</td><td>38.7</td><td>32.7</td></tr></table>

Table 7: Self-judge variants on Qwen3.6-27B, following the protocol and conventions of Table 2. The first row is the self-judge of the main text: thinking enabled in every judge phase, and scoring repeated k=3 times per trajectory with a criterion counted as passed only if all three calls pass it. The second row disables thinking and scores with a single call.

<table><tr><td>Model</td><td>TGC SGC</td><td>Cost ($)</td><td>$/task</td></tr><tr><td>claude-opus-4-7 claude-sonnet-4-6 Kimi-K2.7-Code DeepSeek-V4-Flash gpt-oss-120b</td><td>93.5 89.9 89.9 70.8 44.0</td><td>91.1 32.71 85.7 18.81 82.1 6.65 50.0 2.42 26.8 3.50</td><td>0.195 0.112 0.040 0.014 0.021</td></tr><tr><td>gemini-2.5-flash-lite</td><td>33.9</td><td>12.5</td><td>0.92 0.005</td></tr><tr><td>Qwen3.6-27B (untrained) DRACO (Qwen3.6-27B)</td><td>69.4 41.1 85.3 70.6</td><td>10.77 8.27</td><td>0.064 0.049</td></tr></table>

wrote — and precision — how many of its own GPT-5.4 also wrote. This covers 1,024 rollouts over 11 steps.

An overlap of 1.0 is not the target, because two authors given one task pick different checks. We therefore score GPT-5.4 against itself on the same metric, comparing its criteria for two different rollouts of the same task, and get recall 0.43 and precision 0.48. Those are the dotted lines in Figure 9b, and they are what “as similar as GPT-5.4 is to itself” looks like.

Table 8: Frontier models vs. DRACO on $\mathrm { { A p p W o r l d } _ { T N } }$ (168 tasks), in %, with the US-dollar cost of one pass over the split and per task. Frontier models: one run each, priced at the provider’s API rates. Qwen rows: protocol and reference price of Table 2. Best per column in bold.

## B.1.2 Generation: writing the criteria

Applying a rubric is not the same as writing one, so we replayed the generation calls too. Here there is no reference label to match, so we measure overlap instead: a third model (Claude Sonnet) matches the two arms’ criteria one to one, which gives recall — how many of GPT-5.4’s criteria the self-model also

The self-model stays below them. Recall averages 0.21, half the ceiling, and sits between 0.16 and 0.28 at every step. Precision averages 0.39 and declines from 0.44 to 0.33 (p=0.006). Precision is above recall at all 11 steps, and the self-model writes fewer criteria than GPT-5.4 (3.8 against 5.1 per rollout): what it writes usually resembles something GPT-5.4 wrote, but it misses most of what GPT-5.4 wrote.

![](images/dc86f5328766ea3932efdfa994ddc24deb5fc18e798c3fe7d46cab393f339223.jpg)  
(a) Cost by pipeline phase.

![](images/d41fcc8ab9260baeab1af0826576657eaa056fa7eb136bad63540dbda298f5a6.jpg)  
(b) Cost against AppWorld<sub>TN</sub> TGC.  
Figure 8: Training-time judge cost, in USD, for the four outcome-blind settings and the self-judge variant, which scores with the policy model itself, with thinking enabled and k=3 scoring calls per trajectory (Section 4.3). All settings use Qwen3.6-27B as the policy model and GPT-5.4 as the judge, except the self-judge. (a) Cost over 100 training steps, split across the five phases that issue judge calls: task-specific and rollout-specific rubric generation, union, scoring, and credit reallocation. A setting that does not run a phase draws no bar. (b) The same cost per training step against $\mathrm { \ A p p W o r l d _ { T N } }$ TGC $( p ^ { 1 }$ , Table 2); up and to the left is better. Costs use the token rates of Section 4.1.3.

## B.1.3 Union: merging candidates into the scored set

A group’s final rubric set is merged down from the candidates the previous two stages proposed, and this is where the substitution breaks. We score the merges the same way, over 175 groups; the ceiling here is recall 0.55 and precision 0.57 (Figure 9c).

Recall averages 0.80 and is above that ceiling at all 11 steps, so the self-model keeps four in five of the checks GPT-5.4 kept — it is not throwing away the right criteria. Precision averages 0.44 and falls from 0.57 to 0.36 $( p { = } 0 . 0 0 2 )$ , below recall at every step. The counts say why: the self-model keeps 15.3 criteria per group where GPT-5.4 keeps 6.3. It finds the right criteria and then keeps far too much else alongside them.

That surplus costs reward signal. Scoring both arms’ merged sets on the same six rollouts with one fixed scorer, the share of criteria that at least one rollout fails — the criteria that survive our discriminativeness filter and so produce a withingroup gradient — is lower for the self-model at all 11 steps (31.3% against 46.5%). The extra criteria are ones every rollout passes, so they lengthen the rubric without adding signal.

Two caveats bound these readings. The replay is pinned to the archived prompts, so the self-model’s generation extends GPT-5.4’s proposals and its merge consumes GPT-5.4’s candidate pool; a fully self-judged run would feed weaker candidates into a weaker merge, and that compounding is not measured here. And the two ceilings rest on 24 and

![](images/0fc99b1a27ff32518baeea0c8941dafbc2075da8ce8dcb11acba57b0acd70f96.jpg)

![](images/0567f5a757a8ccc9fa4ea2ae441cd07693929fd77634985088ec9067d72aedb0.jpg)

![](images/40c254a8638816bdc28f7e2f7351d1d38a3c9fd7eb450bc3f4ab492ae1a9aeb2.jpg)  
Figure 9: The three jobs a judge does, self-model vs. GPT-5.4, measured by replaying the run’s judge calls through the policy checkpoint on the archived prompts, with a single call and no thinking (the ablated self-judge of Table 7). (a) Scoring a rubric it is given. How often the self-model returns the same verdict as GPT-5.4, over all 60,689 verdicts, and the two ways it can differ. Each series has its own denominator, named in the legend. Faint lines are per-step values, bold lines smoothed. (b) Writing the criteria and (c) merging them into the set a group is scored on. Here the two models write their own criteria, so we compare the sets themselves: a third model pairs up criteria that mean the same thing, and we then ask how much the two sets overlap. Recall is how much of GPT-5.4’s set the self-model also wrote; precision is how much of the self-model’s set GPT-5.4 also wrote. Legends give means, and the shading is simply the gap between the two curves. Perfect overlap is not the target, because two judges writing criteria for the same task will not pick the same checks: the dotted lines show GPT-5.4 compared against itself on two rollouts of one task, and are the reference to read the curves against. The self-model scores a given rubric well but too leniently, reproduces only about a fifth of the criteria GPT-5.4 writes, and at the merge keeps most of GPT-5.4’s criteria while adding roughly twice as many of its own.

14 scored pairs respectively, with the union ceiling built from the same task at different steps because union makes only one call per task per step, so both are scale references rather than exact bounds.

## B.2 What the dynamic rubrics contain

Section 4.4 shows that DRACO’s rollouts recover 91.3% on the fixed static set they never trained on, so the per-trajectory criteria subsume the static ones. This section asks the complementary question: what do they add? We tally every criterion the judge scored over all 96 rollouts of DRACO’s 99 completed steps (6,245 distinct criteria, 60,119 judgments), and label each one by how many of the 90 training tasks it was ever written for.

Almost every criterion is a sub-goal of one task. 84.4% of the distinct criteria are written for a single task, and they carry 57.2% of all judgments; only 18 criteria are shared across ten or more tasks. A static set has no such register by construction: with one set covering all 90 tasks, every criterion must be phrased generally enough to apply everywhere. So the two regimes differ less in how many questions they ask than in what kind: for the Spotify task “Follow all artists ofall indie-genre songs in any of my playlists”, DRACO writes Indie songs were identified correctly, Deduplicates artists beforefollow, and Only own playlists used; for “Accept all pending Venmo payment requests from my coworkers andfriends” it writes Used contactsfor eligibility, Checks funding before approval, and Approved all and only eligible requests. None is expressible in a rubric shared with the other task, and each names a specific way that instruction can be misread — following artists from someone else’s playlist, approving a request from a non-coworker.

The recurring criteria are the static ones, rediscovered. The 18 task-general criteria fall into four families: not exposing secrets (seven phrasings), handling pagination (five), stopping after completion (three), and recovering from an API or execution error (three). All four are already in the static set (Protects secret values, Fetches complete data, Stops after completion, Recovers from failures; Table 9), which is the subsumption result seen from the generator’s side: given only the instruction and a rollout, it re-derives the same cross-cutting concerns a human-curated set was built around, and does so on the tasks where they bite rather than on all of them.

The task-specific criteria are the ones that discriminate. 26.7% of DRACO’s applicable verdicts are failures, against 5.5% for the static set, so the per-trajectory criteria are far harder to satisfy — which is what the reward needs, since a criterion the whole group passes contributes nothing to the advantage after group normalization (Eq. (1)). The same asymmetry appears in what the judge can score at all: a static criterion draws not applicable on 10.7% of judgments and a per-trajectory one on 1.9%, because a set shared across 90 tasks must include criteria that fit only some of them (Handles time correctly is inapplicable on 67.6% of the rollouts it was scored against, and six criteria account for 95.6% of all such verdicts).

![](images/0e430176d75836ab933e0dae89924174387cabe998a747ea5f9b34bfb8e1b845.jpg)  
Figure 10: Per-rubric pass rate vs. training step, static rubrics with and without step credit. One panel per static criterion; faint lines are raw per-step values, bold lines smoothed. The dotted series re-scores DRACO against the same fixed set, which it never trained on.

## B.3 Why step credit needs dynamic rubrics

Section 4.3 finds that step credit is worth +3.2 TGC on per-trajectory rubrics but only +0.8 on the static set. The credit rule itself explains why. It reallocates a trajectory’s advantage only when the rubric’s verdicts are mixed — some criteria pass and others fail on the same trajectory. If they are unanimous, every step takes the same weight and the update is exactly baseline GRPO (Appendix E.7); the same is true when discriminative dropout leaves no criterion at all. Credit is then inert: present in the pipeline, has no effect on the gradient.

Because the condition is exact, we can count it. Over each arm’s 100 steps we classify all 96 rollouts per step from the logged verdicts, and average the inert share within five windows of training. On the static set it climbs from 23.6% of rollouts over steps 1–10 to 55.4%, 82.0%, 88.9% and 91.9% by steps 76–100. DRACO starts at a similar 29.7% and stays near it: 44.7%, 44.5%, 52.9% and 56.5%. Averaged over training, credit is inert on 76.4% of static rollouts against 48.1% of DRACO’s, and the spread of step qualities within a trajectory — how unevenly credit divides the advantage when it does fire — is correspondingly larger for DRACO (0.215 against 0.144).

So the static arm does not attribute badly; for most of training it does not attribute at all. The rise mirrors the saturation of Figure 4: once the policy passes almost every fixed criterion, unanimous rollouts are the common case, and by the last quarter of training the credit component is switched off on nine rollouts in ten. Per-trajectory rubrics keep producing criteria some group member fails, which is what leaves credit something to divide. The pattern is not specific to one policy model: the Qwen2.5-32B-Instruct run under DRACO moves only from 15.7% to 30.9% inert over the same windows.

This measures when credit is inert, which follows from the rule, not whether an active reallocation lands on the steps truly responsible — the attribution-quality question we flag in the Limitations section.

## C Rubric Construction

The static rubric set was authored once before training and then frozen. We sampled 8 rollouts for each of the 90 training tasks from the untrained base policy model (720 trajectories at temperature 1.0), had GPT-5.4 propose criteria per task, reduced them to a single set, and pruned criteria that were rarely applicable or rarely discriminative. The dynamic settings instead generate criteria per task at every step during training (Section 3.2).

Every generation call, static and dynamic alike, instructs the judge to keep the criteria MECE (Section 3.2), and the pruning above enforces the same property on the static set after the fact: a criterion that is rarely applicable is not covering a dimension the tasks actually exercise, and one that is rarely discriminative is usually restating a criterion already in the set. Table 9 lists the resulting 21 criteria, ordered by their failure rate over those 720 rollouts. Nothing was pruned at this stage: the pruning happened while reducing the per-task proposals to the single set. Applicability survived unevenly — 13 of the 21 apply to at least 99% of those rollouts, while 8 apply to fewer, down to 29.9% for Handles time correctly, and those 8 are what Appendix B.2 measures the cost of at training time. The failure rates span two orders of magnitude, from 81.8% on Protects secret values down to 2.8% on Uses allowed tools only, which is what discriminative dropout (Section 3.2) exists to handle at training time: the criteria near the bottom of the table pass for almost every group and are dropped from the reward.

## D Training Hyperparameters

Table 10 gives the full training configuration (Section 4.1.2). It is shared by all four settings, which differ only in the reward (Section 4.1.1); the credit rule’s own knobs are in Appendix E.8. Advantages are the within-group standardized rewards of Eq. 1, so the group mean is the only baseline and no value network is trained; the 16 prompts and G=6 rollouts each give 96 rollouts per step, and every run trains for at most 20 epochs (100 optimizer steps) with a checkpoint written at every step. Every run trains on 8 NVIDIA H100 GPUs.

Three choices are worth flagging because they are easy to assume otherwise. First, we train LoRA adapters rather than the full model, so the reported gains come from a rank-16 update on top of frozen base weights. Second, we run a single PPO epoch per batch, so every update is on-policy and the importance ratio is 1 at the time of the step, which makes the clipping range inactive in practice. Third, the learning rate is constant at $5 \times 1 0 ^ { - 5 }$ with no warmup, so no schedule interacts with the lastthree-checkpoint reporting protocol (Section 4.1.3).

Serving configuration for Qwen2.5-32B-Instruct. We serve Qwen2.5-32B-Instruct at its native 32,768- token context, using its own chat template and a tool-call parser matching the JSON format it emits. On τ-bench we extend the window to 131,072 tokens with YaRN rope scaling (Peng et al., 2023). Thinking mode is off, since its chat template defines no reasoning channel.

τ -bench retrieval. Both base policies retrieve with Qwen embeddings in the all-tools setting.

## E Credit Reallocation: Details

This appendix restates the preliminaries and gives the full derivation of the credit-reallocation rule (Section 3.3), the significance of each term, a worked example, a traced real rollout, and the configuration used in our runs.

## E.1 Preliminaries: GRPO and the token-level advantage

We train with Group Relative Policy Optimization (GRPO). For a task x the policy model samples a group of G trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ , and each trajectory τ<sub>i</sub> receives a scalar reward $R _ { i }$ . GRPO standardizes rewards within the group to form the trajectory

advantage

$$
A _ { i } = \frac { R _ { i } - \mu _ { g } } { \sigma _ { g } } ,\tag{8}
$$

where $\mu _ { g }$ and $\sigma _ { g }$ are the mean and standard deviation of the group’s rewards. Writing the tokens of $\tau _ { i }$ as $y _ { i , 1 } , \ldots , y _ { i , N _ { i } }$ , GRPO takes a policy-gradient step in which every token is pushed by the same advantage,

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { I } = \mathbb { E } \left[ \sum _ { t = 1 } ^ { N _ { i } } A _ { i } \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( y _ { i , t } \mid y _ { i , < t } , x ) \right] .\tag{9}
$$

So the advantage acts as a per-token scalar multiplier on that token’s log-probability gradient: a positive advantage makes the token more likely in its context, a negative advantage less likely, and a zero advantage leaves it untouched. The defining limitation is that a single $A _ { i }$ multiplies all $N _ { i }$ tokens, so every decision in a long trajectory receives identical credit. Credit reallocation keeps the GRPO reward and group standardization intact and replaces only the uniform multiplier: every token in step j receives a per-step advantage $a _ { j }$ (Eq. (14)) in place of $A _ { i }$ in Eq. (9).

## E.2 Setup and notation

A trajectory i is a multi-turn agent rollout, decomposed into steps $j = 1 , \dots , M$ , where one step is approximately one agent turn (one emitted code block). Each response token belongs to exactly one step, or to a gap (turn glue, tool-result echoes) that belongs to no step. Two quantities enter the assignment:

$A _ { i } ,$ the trajectory-level GRPO advantage of Eq. (8), whose sign says whether the rollout is reinforced or suppressed and whose magnitude says how strongly.

$Q _ { j } \in [ 0 , 1 ]$ , the quality of step j, derived from the rubric grader (Eq. (12)).

Credit is assigned at the step level: the rubric grades steps, not tokens, so there is no token-resolution signal to reallocate on. The scheme replaces the flat $A _ { i }$ with a per-step advantage $a _ { j }$ that depends on the step’s quality, and every token inside step j receives the same $a _ { j }$

## E.3 Formulation

Let $p _ { i }$ and $f _ { i }$ count the passed and failed criteria over the whole trajectory $\tau _ { i }$ (surviving dropout and applicable), let $p _ { j }$ and $f _ { j }$ count those citing step $j ,$ and let $n _ { j }$ be the number of tokens in step $j ,$ with $\begin{array} { r } { N = \sum _ { k } n _ { k } } \end{array}$ . The assignment, from rubric reward through to the step advantage that replaces $A _ { i }$ (restating Sections 3.2 and 3.3), is

$$
R _ { i } = { \frac { p _ { i } - f _ { i } } { p _ { i } + f _ { i } } } ,\tag{10}
$$

$$
A _ { i } = \frac { R _ { i } - \mu _ { g } } { \sigma _ { g } } ,\tag{11}
$$

$$
Q _ { j } = \frac { p _ { j } } { p _ { j } + f _ { j } } ,\tag{12}
$$

$$
w _ { j } = \left\{ { \begin{array} { l l } { Q _ { j } , } & { A _ { i } \geq 0 , } \\ { 1 - Q _ { j } , } & { A _ { i } < 0 , } \end{array} } \right.\tag{13}
$$

$$
a _ { j } = A _ { i } \cdot \frac { N w _ { j } } { n _ { j } \sum _ { k } w _ { k } } ,\tag{14}
$$

where $\mu _ { g }$ and $\sigma _ { g }$ are the group’s reward mean and standard deviation (Eq. (11) repeats Eq. (8) so the chain reads without a jump), and $R _ { i } = 0$ when no criterion applies. A step cited by no criterion takes $Q _ { j } = \bar { Q }$ , the mean $Q$ over cited steps. Gap tokens receive advantage 0 and are excluded from $N$ , so conservation holds over exactly the tokens credit is distributed across. Every token in step j takes the advantage $a _ { j }$ in place of $A _ { i }$ in Eq. (9).

## E.4 Significance of each term

$A _ { i }$ (trajectory advantage): sets the total push. The method is a redistribution of $A _ { i }$ , not a replacement of it. The sign of $A _ { i }$ says whether this rollout should be reinforced (winner, $A _ { i } \geq 0 )$ or suppressed (loser, $A _ { i } ~ < ~ 0 )$ relative to its group; its magnitude says how strongly. Credit only decides where within the trajectory that push lands.

$Q _ { j }$ (step quality): the per-step signal. $Q _ { j }$ (Eq. (12)) is the pass fraction of the rubric checks that cite step $j ,$ where $p _ { j }$ and $f _ { j }$ are the number of passed and failed citing checks. A step cited by no rubric check inherits $\bar { Q }$ , the mean $Q$ over cited steps. Evidence per step is thin: measured on the credit run, the number of rubrics citing a single step has median 1 and mean 1.95, and only 0.6% of steps are cited by ten or more rubrics. So the majority of cited steps take $Q _ { j } \in \{ 0 , 1 \}$ from a single verdict, and the rule is deliberately literal about the judge’s attribution rather than hedging it toward a neutral value.

$w _ { j }$ (step weight): converts quality into directioncorrect push. The winner/loser split in Eq. (13)

is what makes credit sign-correct. On a winning trajectory we reinforce the steps that were actually good, so weight rises with $Q _ { j } ;$ on a losing trajectory we suppress the steps that were actually bad, so weight rises with $1 - Q _ { j }$ . A step whose citing checks unanimously disagree with the trajectory’s direction takes $w _ { j } = 0 \colon$ on a winner, a step every citing criterion failed contributes nothing to the update, and on a loser, a step every citing criterion passed is not suppressed. Both are the intended reading of the attribution.

$n _ { j }$ and the $1 / n _ { j }$ factor: length handling. The quantity the rule holds length-independent is each step’s total contribution, not its per-token push. Step j contributes

$$
n _ { j } a _ { j } = A _ { i } N \frac { w _ { j } } { \sum _ { k } w _ { k } } ,\tag{15}
$$

which depends on $w _ { j }$ and not on $n _ { j } \colon$ : a step earns influence by being judged good, not by being long. The $1 / n _ { j }$ factor spreads that fixed total over however many tokens the step contains, so a verbose step pushes each of its tokens proportionally less hard. The alternative, dropping $1 / n _ { j }$ , would equalize the per-token push instead and let a long step accumulate a proportionally larger total, which would reward verbosity.

N and conservation: same total push as baseline GRPO. The factor N is chosen so the total advantage over the trajectory is preserved. Summing Eq. (15) over steps,

$$
\sum _ { j } n _ { j } a _ { j } = A _ { i } N \frac { \sum _ { j } w _ { j } } { \sum _ { k } w _ { k } } = A _ { i } N .\tag{16}
$$

Baseline GRPO assigns $A _ { i }$ to each of those same N tokens, so its total push is also $A _ { i } N \colon$ only the distribution across steps changes. Gap tokens sit outside this accounting entirely, receiving advantage 0 and being excluded from N, so the conserved quantity is the baseline total over the credited tokens rather than over the full response. Note that this conserves the scalar sum $\textstyle \sum _ { j } n _ { j } a _ { j }$ ; it does not conserve the total gradient vector, since moving advantage between steps whose tokens have different gradients is precisely the intended effect.

From advantage to update. At ppo\_epochs = 1, the policy-gradient update contributed by a token in step j is $\Delta \theta = \eta a _ { j } \nabla _ { \theta }$ log π<sub>θ</sub>(token), so $a _ { j }$ is a scalar multiplier on that token’s log-probability gradient: $a _ { j } > 0$ makes the token more likely in its context, $a _ { j } < 0$ less likely, and $a _ { j } = 0$ leaves it untouched.

## E.5 Worked example

Consider a trajectory with $A _ { i } = 1$ and two steps of equal quality $( w _ { 1 } = w _ { 2 } = 1 )$ , with $n _ { 1 } = 2$ and $n _ { 2 } = 4$ tokens, so $N = 6$ and $\textstyle \sum _ { k } w _ { k } = 2$ . Then $a _ { 1 } = 6 ( 1 ) / ( 2 \cdot 2 ) = 1 . 5 0$ and $a _ { 2 } = 6 ( 1 ) / ( 4 \cdot 2 ) =$ 0.75, so $n _ { 1 } a _ { 1 } = n _ { 2 } a _ { 2 } = 3$ and $\begin{array} { r } { \sum _ { j } n _ { j } a _ { j } = 6 = } \end{array}$ $A _ { i } N$ . Equal quality buys equal total influence: the two steps split the trajectory’s push evenly despite one being twice as long, which is why the shorter step’s tokens each carry more.

Now let step 2 be higher quality, $w _ { 1 } ~ = ~ 0 . 5$ and $\begin{array} { r } { w _ { 2 } \ = \ 1 . 0 , \ \mathrm { s o } \ \sum _ { k } w _ { k } \ = \ 1 . 5 } \end{array}$ . Step 1 gets $a _ { 1 } ~ = ~ 6 ( 0 . 5 ) / ( 2 \cdot 1 . 5 ) ~ = ~ 1 . 0 0$ and step 2 gets $a _ { 2 } = 6 ( 1 . 0 ) / ( 4 \cdot 1 . 5 ) = 1 . 0 0$ , giving $n _ { 1 } a _ { 1 } = 2 .$ n<sub>2</sub>a<sub>2</sub> = 4, and again $\textstyle \sum _ { i } n _ { j } a _ { j } = 6 = A _ { i } N$ . The good step now claims twice the total influence of the weak one (4 against 2), in the ratio of their weights; here that happens to land both per-token advantages at 1.00, because step 2 is both twice as good and twice as long.

## E.6 Running example (real logged rollout)

The following is a real rollout from the credit run (step 11), traced from the raw rubric verdicts to the per-step advantage, so every $Q _ { j }$ and the uncited fallback are the actual logged values. Two quantities are illustrative and labelled where used: the per-step token counts $n _ { j }$ (the log records qualities and citations, not token spans), and $A _ { i } .$ , which we set to 1 for readability. Every $a _ { j }$ below scales linearly in $A _ { i } ,$ so the choice fixes the units of the table and nothing else.

The rollout has $M = 7$ steps and takes the winner branch, meaning $A _ { i } \geq 0 !$ its reward exceeded its group’s mean. It is worth noting what that does not mean. Three of its five checks failed, so by Eq. (10) its own reward is $R _ { i } = ( 2 - 3 ) / 5 = - 0 . 2 ;$ it is a winner because the rest of its group scored below −0.2, not because it did well in absolute terms and not because it passed the task, which the reward never observes (Appendix E.8). Five rubric checks fired, each citing one or more steps:

<table><tr><td>Check</td><td>Verdict</td><td>Cites steps</td></tr><tr><td>R1</td><td>FAIL</td><td>4,6</td></tr><tr><td>R2</td><td>FAIL</td><td>5,6</td></tr><tr><td>R3</td><td>FAIL</td><td>3,4,5</td></tr><tr><td>R4</td><td>PASS</td><td>5</td></tr><tr><td>R5</td><td>PASS</td><td>1,2,3,4, 5, 6</td></tr></table>

Stage 1: tally passes/fails per step. For each step j, count the citing checks that passed $( p _ { j } )$ and failed $( f _ { j } )$ . Step 7 is cited by nothing.
<table><tr><td>Step</td><td> $p _ { j }$ </td><td> $f _ { j }$ </td><td>Cited by</td></tr><tr><td>1</td><td>1</td><td>0</td><td>R5 (PASS)</td></tr><tr><td>2</td><td>1</td><td>0</td><td>R5 (PASS)</td></tr><tr><td>3</td><td>1</td><td>1</td><td>R5 (PASS); R3 (FAIL)</td></tr><tr><td>4</td><td>1</td><td>2</td><td>R5 (PASS); R1, R3 (FAIL)</td></tr><tr><td>5</td><td>2</td><td>2</td><td>R4, R5 (PASS); R2, R3 (FAIL)</td></tr><tr><td>6</td><td>1</td><td>2</td><td>R5 (PASS); R1, R2 (FAIL)</td></tr><tr><td>7</td><td>0</td><td>0</td><td>(uncited)</td></tr></table>

Stage 2: quality $Q _ { j } ~ = ~ p _ { j } / ( p _ { j } + f _ { j } ) . ~ Q _ { 1 } ~ =$ $Q _ { 2 } = 1 / 1 = 1 . 0 0 0 ; Q _ { 3 } = 1 / 2 = 0 . 5 0 0 ; Q _ { 4 } =$ $Q _ { 6 } = 1 / 3 = 0 . 3 3 3 ; Q _ { 5 } = 2 / 4 = 0 . 5 0 0$ . Steps 1 and 2 are cited only by the one check that passed, so they take the maximum; steps 4 and 6 are cited by one passing and two failing checks and land lowest. Note how thin the evidence behind $Q _ { 1 } =$ $Q _ { 2 } = 1$ is: their only citation is R5, a broad check spanning six steps, so a single verdict decides both. That is representative rather than cherry-picked, since the median step draws exactly one citation (Appendix E.4), but it is the reason the uncitedstep fallback averages over cited steps rather than trusting any one of them.

Stage 3: uncited step 7 inherits Q<sup>¯</sup>. $Q _ { 7 } = \bar { Q } =$ $\begin{array} { r } { \frac { 1 } { 6 } ( 1 . 0 + 1 . 0 + 0 . 5 + 0 . 3 3 3 + 0 . 5 + 0 . 3 3 3 ) = 0 . 6 1 1 . } \end{array}$

Stage 4: winner weights. This is a winner, so the weight is the quality itself, $w _ { j } = Q _ { j }$ , for every step (including the uncited step $\bar { 7 } , w _ { 7 } = \bar { Q } = 0 . 6 1 1 ; \mathrm { a }$ n uncited step is still a step, unlike a gap, which is outside N entirely).

Stage 5: step advantage. The share of the trajectory’s push each step claims is the ratio of weights and needs no token counts: steps 1 and 2 (the strongest, $w = 1 . 0 0 0 )$ claim $1 . 0 0 0 / 0 . 3 3 3 = 3 \times$ the total influence of the weakest steps (4 and $6 , w = 0 . 3 3 3 )$ . This is the entire behavioral effect: the good early steps are reinforced harder than the steps the rubric flagged as faulty. Token counts enter only in spreading each step’s share over its own tokens. Taking an illustrative $n = [ 4 0 , 4 0 , 3 0 , 5 0 , 3 0 , 6 0 , 6 0 ]$ , so $N = 3 1 0$ , with $\Sigma _ { k } w _ { k } = 4 . 2 7 8$ and $A _ { i } = 1$

<table><tr><td>Step</td><td> $n _ { j }$ </td><td> $w _ { j }$ </td><td> $a _ { j }$ </td><td>step total  $n _ { j } a _ { j }$ </td></tr><tr><td>1</td><td>40</td><td>1.000</td><td>1.812</td><td>72.47</td></tr><tr><td>2</td><td>40</td><td>1.000</td><td>1.812</td><td>72.47</td></tr><tr><td>3</td><td>30</td><td>0.500</td><td>1.208</td><td>36.23</td></tr><tr><td>4</td><td>50</td><td>0.333</td><td>0.483</td><td>24.16</td></tr><tr><td>5</td><td>30</td><td>0.500</td><td>1.208</td><td>36.23</td></tr><tr><td>6</td><td>60</td><td>0.333</td><td>0.403</td><td>24.16</td></tr><tr><td>7</td><td>60</td><td>0.611</td><td>0.738</td><td>44.29</td></tr><tr><td>∑</td><td>310</td><td></td><td></td><td>310.00</td></tr></table>

Baseline GRPO would put $a _ { j } = 1 . 0$ on all seven steps $\textstyle ( \sum _ { j } n _ { j } a _ { j } = 3 1 0 )$ . Credit keeps the same total, $\begin{array} { r } { \sum _ { j } n _ { j } ^ { - } a _ { j } = A _ { i } N = 3 1 0 } \end{array}$ , but rebalances it: the strong steps take totals of 72.47 against 24.16 for the flagged ones, a $3 \times$ gap matching their weights, while the uncited step sits between them. Changing the token counts changes the per-token advantages but never the step totals (they depend on $w _ { j }$ only) and never the conserved total. The $a _ { j }$ column is therefore not a ranking of the steps: a short step spreads its share over fewer tokens and so shows a larger per-token value, which is why the step totals rather than $a _ { j }$ are the quantity to read across rows.

## E.7 Unanimous rollouts stop discriminating between steps

Credit discriminates between steps only when the rubric’s verdicts are mixed within a trajectory. If every citing verdict is a PASS, then $f _ { j } = 0$ and $Q _ { j } ~ = ~ 1$ for every cited step, the uncited steps inherit $\bar { Q } = 1$ , and all weights are equal. Eq. (14) then reduces to $a _ { j } ~ = ~ A _ { i } N / ( n _ { j } M )$ : every step claims an equal share $A _ { i } N / M$ of the trajectory’s total push, and the only remaining variation is the $1 / n _ { j }$ spreading of that share over each step’s own tokens. The citation count carries no signal here, only the pass/fail composition does. The mirror case behaves the same way: on a loser whose every citing verdict is a FAIL, all $Q _ { j } = 0$ so all $w _ { j } =$ $1 - Q _ { j } = 1$ , again uniform.

One degenerate case needs an explicit fallback. If a winner’s citing verdicts are all FAIL, every weight is $w _ { j } = Q _ { j } = 0$ , the normalizer $\sum _ { k } w _ { k }$ vanishes, and Eq. (14) is undefined. The implementation detects the collapsed normalizer and falls back to the uniform $A _ { i } ,$ , so such a trajectory trains as baseline GRPO rather than not at all. The same guard covers a loser whose verdicts all pass.

## E.8 Configuration

<table><tr><td>Knob</td><td>Symbol</td><td>Value</td></tr><tr><td>Length handling</td><td> $1 / n _ { j }$ </td><td>step total is length-free</td></tr><tr><td>Gap tokens</td><td> $\mathrm { n / a }$ </td><td>advantage 0, outside N</td></tr><tr><td>Uncited-step quality</td><td> $\mathrm { n / a }$ </td><td>Q over cited</td></tr></table>

Two degenerate cases: if no step is cited there is no attribution to reallocate on and the implementation falls back to the uniform $A _ { i }$ , i.e. baseline GRPO. And because the reward is outcome-blind, the winner/loser branch is keyed on the sign of the GRPO advantage $A _ { i }$ (its group-relative reward), not on task success, so a rollout can be a “winner” for the weighting branch while still failing the task, and vice versa.

## E.9 Properties

We state the invariants that justify the formulation, each with a short proof. Throughout, step $j$ has $n _ { j } > 0$ tokens, $N = \textstyle \sum _ { j } n _ { j }$ over the steps, and weight $w _ { j } \geq 0$ with $\textstyle \sum _ { k } w _ { k } > 0$ (the collapsed case falls back to baseline, Appendix E.7). It is convenient to write step j’s total contribution as a share of the trajectory $\mathrm { ^ { , } s }$ total push,

$$
n _ { j } a _ { j } = A _ { i } N \rho _ { j } , \qquad \rho _ { j } : = \frac { w _ { j } } { \sum _ { k } w _ { k } } \ : ( \ge 0 ) ,\tag{17}
$$

so $\rho _ { j }$ is the fraction of the trajectory’s push that step j claims, the $\rho _ { j }$ sum to one by construction, and every property below follows in one line from this form.

P1: total-advantage conservation (exact).

$$
\sum _ { j } n _ { j } a _ { j } = A _ { i } N \sum _ { j } \rho _ { j } = A _ { i } N .
$$

The total push equals baseline GRPO’s over the same N credited tokens, for any non-negative weights, winner or loser. Nothing in the reward, the judge, or the weights can break this. Gap tokens sit outside the accounting, taking advantage 0 and being excluded from N, so the conserved quantity is the baseline total over the credited tokens.

P2: no signal equalizes step totals (exact). If $w _ { j } \equiv w$ is constant across steps, then $\rho _ { j } = 1 / M$ and every step claims the same total $A _ { i } N / M ,$ giving $a _ { j } = A _ { i } N / ( n _ { j } M )$ . Weights become constant whenever the judge’s verdicts are unanimous, in three ways: no citations at all $( p _ { j } = f _ { j } = 0$ for every j, so every step takes $\bar { Q } )$ , every citing verdict a PASS $( Q _ { j } = 1$ throughout), and every citing verdict a FAIL $( Q _ { j } = 0$ throughout). Credit then draws no distinction between steps, because the rubric drew none within the trajectory; what remains is the $1 / n _ { j }$ spreading of each equal share over each step’s own tokens. This is a structural guarantee, not a performance-dominance claim: it does not prove credit never underperforms uniform advantage on a given task; an adversarially wrong judge can still hurt.

P3: sign preservation (exact). $N > 0 , n _ { j } > 0$ and $w _ { j } \geq 0 \Rightarrow \rho _ { j } \geq 0 \Rightarrow a _ { j }$ is either 0 or carries $\mathrm { s i g n } ( A _ { i } )$ , for every step. Credit never flips a reinforce into a suppress (or vice versa); it only rescales magnitudes, possibly to zero. This rules out the worst failure mode, a mistaken judge inverting the push on a good trajectory.

P4: monotone correctness (exact). On a winner $( A _ { i } \ \ge \ 0 ) , w _ { j } \ = \ Q _ { j }$ increases in $Q _ { j } .$ so $\rho _ { j }$ and hence $a _ { j }$ increase in $Q _ { j } { \mathrm { : } }$ higher-quality steps get more positive push. On a loser $( A _ { i } ~ < ~ 0 )$ $w _ { j } = 1 - Q _ { j } , \mathrm { s o } | a _ { j } |$ increases as $Q _ { j }$ decreases: lower-quality steps get more suppression. In both cases credit flows toward the good steps of a good trajectory and the bad steps of a bad one.

P5: step totals are length-independent (exact). $n _ { j } a _ { j } = A _ { i } N \rho _ { j }$ depends on $w _ { j }$ only, never on $n _ { j }$ : a step’s total influence is set by how the judge graded it, not by how many tokens it spent, so padding a step cannot buy it more total reinforcement. The per-token advantage does depend on length, falling as $1 / n _ { j }$ , which is the intended trade: a verbose step spreads its fixed share thinner. This property is about length, not granularity; the rule is not invariant to how finely turns are cut into steps, since splitting a step of weight w in two adds w to $\sum _ { k } w _ { k }$ and so changes every step’s share. Our step boundaries are fixed by the environment (one step is one emitted code block), not chosen, so granularity is not a free parameter here.

P6: reward-scale invariance (exact). $\begin{array} { r l } { a _ { j } } & { { } = } \end{array}$ $A _ { i } \cdot N w _ { j } / ( n _ { j } \sum _ { k } w _ { k } )$ , with the factor multiplying $A _ { i }$ independent of $A _ { i }$ . The reallocation pattern therefore does not depend on the reward magnitude or on $\mathbf { G R P O ' s } \sigma _ { g }$ normalization; credit composes linearly with the advantage. Rescaling rewards rescales all $a _ { j }$ uniformly and leaves the relative per-step allocation fixed. The group baseline $\mu _ { g }$ is subtracted upstream, inside $A _ { i } ,$ so reallocation never touches the baseline computation.

P7: winner/loser symmetry (exact). The loser branch is the winner branch with quality flipped. A loser step of quality Q gets weight 1 − Q, which is exactly the weight a winner step of quality $1 - Q$ would get. So suppressing a bad step (Q low) on a loser mirrors reinforcing a good step (1 − Q high) on a winner: a step at $Q = 0 . 2$ on a loser carries the same weight (0.8) as a step at $Q = 0 . 8$ on a winner. Neither branch is favored; the mechanism treats “reinforce the good” and “suppress the bad” as the same rule seen from opposite signs of A<sub>i</sub>.

<table><tr><td>Property</td><td>Statement</td></tr><tr><td>P1 conservation</td><td> $\begin{array} { r } { \sum _ { j } n _ { j } a _ { j } = A _ { i } N \colon } \end{array}$  same total push as baseline</td></tr><tr><td>P2 no signal</td><td>unanimous verdicts give constant weights, hence equal step totals</td></tr><tr><td>P3 sign preserva- tion</td><td>no step&#x27;s push is inverted relative to  $A _ { i }$ </td></tr><tr><td>P4 monotone cor- push rises with rectness</td><td> $Q _ { j }$  on winners, with  $\hat { 1 } - Q _ { j }$  on losers</td></tr><tr><td>P5 length indepen- step total dence</td><td> $n _ { j } a _ { j }$  depends on wj only, not on length</td></tr><tr><td></td><td>P6 scale invariance pattern independent of reward scale /</td></tr><tr><td></td><td> $\sigma _ { g }$  P7 winner/loser loser branch is winner branch re-</td></tr><tr><td>symmetry</td><td>flected at  $\begin{array} { r } { Q = \frac { 1 } { 2 } } \end{array}$ </td></tr></table>

## F Prompts

Six prompts constitute the full pipeline: five judge prompts and the agent’s own system prompt. They are reproduced verbatim below, with two mechanical changes so they typeset: a handful of non-ASCII characters (em dashes, arrows, the tick and cross in the atomicity example) are transliterated, and in the system prompt the doubled braces Python needs for escaping are shown singly, as the model actually receives them. The five judge prompts are byte-identical across every training setting.

Placeholders in the judge prompts are written {{ name }} and filled by string substitution. agent\_trajectory is the whole conversation flattened to [ROLE]-tagged blocks, so the judges see the same rollout text the agent produced, with no tool-call structure preserved. Log labels do not line up with the file names: stage1 is rubric generation from the task, stage2 generation per trajectory, union the merge, stage3 scoring, and credit the step attribution. A group of G=6 rollouts therefore costs about 20 judge calls, one cached stage1 call plus one merge and six each of the rest.

Rubric generation from the task (Listing 1) is called once per task and cached across the group. It takes the agent’s system prompt and first user message, and returns at most 15 criteria. Note the explicit ban on table-stakes conformance criteria by name, which is the same concern discriminative dropout addresses arithmetically.

Rubric generation from a trajectory (Listing 2) is called once per rollout, with the stage-1 set supplied so it only proposes what that set does not already cover. It is instructed to anchor each criterion on something the rollout actually fell short on, since a criterion the rollout plainly passes cannot carry group signal, and to generalize the wording without loosening that anchor.

Merging into one canonical set (Listing 3) reduces the candidates from the previous two stages to one set with a ceiling of 24, stated in the prompt as a ceiling rather than a target. All G rollouts are rendered in full so the merge can check discriminativeness against the real group instead of inferring it from wording; its most consequential rule is to drop any criterion that would pass for every rollout shown.

Scoring (Listing 4) produces the reward. Verdicts are +1, −1 and 0 for not applicable. Two rules shape the reward distribution: a conditional criterion whose trigger never occurred must score 0 rather than +1, so an untested criterion cannot earn a free pass; and when the choice is between −1 and 0 the judge is told to prefer −1, so the not-applicable escape is correspondingly narrow.

Step attribution (Listing 5) turns verdicts into the step citations the credit rule consumes (Section 3.3). The judge may confirm or override the scoring verdict. The load-bearing constraint is that every pass or fail must cite at least one step: only a not-applicable verdict may cite none, and absencetype criteria are attributed to the step where the omission mattered most. Without it, part of the trajectory’s advantage would have nowhere to land.

Agent system prompt (Listing 6) is applied once per sample at dataset construction, not at rollout time, and is adapted from verl-agent’s AppWorld template. Instruction 16 is why every judge prompt opens with the same caveat about complete\_task: the episode does not end when the agent calls it, so a brief closing turn afterwards is correct and must not be penalized. The version shown is the one used for Qwen3.6-27B; the Qwen2.5-32B-Instruct runs use an identical prompt with the final two lines replaced by You present the solution code body within <code> </code> tags., dropping the <think> instruction because that model has no reasoning mode. This is the only prompt that differs between the two base policies.

<table><tr><td># Title</td><td></td><td>Description</td></tr><tr><td></td><td>1 Protects secret values</td><td>The agent should avoid printing or restating passwords, tokens, or other secrets. Using a secret inside an API call is fine, but exposing it is not.</td></tr><tr><td></td><td>2 Stops after completion</td><td>After the task is completed, the agent should stop sending more code. Extra code after completion is unnecessary and risky.</td></tr><tr><td></td><td>3 Fetches complete data</td><td>The agent should collect all needed records before deciding. This includes checking later pages, reading needed details, and not trusting visibly truncated output as complete.</td></tr><tr><td></td><td>4 Completes task properly</td><td>The agent should explicitly finish with one completion call after the work is done. If an answer is required, the completion call should contain only the requested value or type.</td></tr><tr><td></td><td>5 Computes from live data</td><td>The agent should work from returned objects and compute results in code when calculation is needed. Manual retyping, eyeballing, or unsupported assumptions</td></tr><tr><td></td><td>6 Handles time correctly</td><td>should fail. For time-based tasks, the agent should get the current time from the environment and use the correct calendar boundaries. Hardcoded dates or wrong windows</td></tr><tr><td></td><td>7 Processes items com- pletely</td><td>should fail. The agent should cover the full required set of items and handle duplicates or repeated occurrences correctly. Missing items, double-counting, or acting only</td></tr><tr><td></td><td>8 Targets exact items</td><td>once when each occurrence matters should fail here. When making changes, the agent should act on exactly the intended items and preserve any required exceptions. Wrong targets, overbroad edits, or changing</td></tr><tr><td></td><td>9 Applies correct selection logic</td><td>protected items should fail here. The agent should identify targets using the exact task conditions. Wrong filters, wrong fields, wrong matching keys, or wrong comparison rules should fail here.</td></tr><tr><td></td><td>10 Uses complete working set</td><td>After fetching data, the agent should use all relevant fetched items in later reasoning. Fetching extra pages or sources but then ignoring part of them is still</td></tr><tr><td></td><td>11 Performs required actions</td><td>incomplete. The agent should actually carry out the requested create, update, delete, send, move, or similar actions. Listing targets without acting, or using the wrong</td></tr><tr><td></td><td>12 Verifies action outcome</td><td>action type, should fail. After changing state, the agent should confirm that the action worked. A follow- up read or a clear success response can both count when they directly confirm</td></tr><tr><td></td><td>13 Uses exact action content</td><td>the intended result. When an action needs exact text, names, amounts, paths, or preserved identity, the agent should use the exact required values. Unrequested renaming or altered</td></tr><tr><td></td><td>14 Uses correct data source</td><td>content should fail. The agent should gather evidence from the app, endpoint, or object set the task actually requires. Using a different source that could change the result should</td></tr><tr><td></td><td>15 Authenticates correctly</td><td>fail. For protected APIs, the agent should log in with the documented identity fields and then use the returned token. Missing login, wrong login fields, or missing</td></tr><tr><td></td><td>16 Retrieves needed creden-</td><td>token are distinct auth failures and should be merged here When login is required, the agent should get credentials from the supervisor</td></tr><tr><td></td><td>tials 17 Uses runnable Python</td><td>data. Guessing, inventing, or hardcoding unseen secrets is not acceptable. The agent should act through executable Python in the REPL. Non-code pay-</td></tr><tr><td>18 Checks</td><td>docswhen</td><td>loads or syntax-breaking text prevent normal execution. The agent should consult app or API docs when it does not already know how</td></tr><tr><td></td><td>needed 19 Recovers from failures</td><td>to call something. Repeated blind guessing should fail this rubric. When code or API calls fail, the agent should change approach and continue</td></tr><tr><td></td><td>20 Uses valid API calls</td><td>productively. Repeating the same mistake, or endlessly retrying after a persistent failure pattern, should fail here. The agent should use real documented endpoints with supported parameters.</td></tr><tr><td></td><td></td><td>Made-up APIs, wrong endpoints, or unsupported arguments are the same ob- servable mistake.</td></tr><tr><td></td><td>21 Uses allowed tools only</td><td>The agent should stay within provided app APIs and allowed Python tools. Forbidden imports, OS access, or invented helpers are not allowed.</td></tr></table>

Table 9: The static rubric set (21 criteria), ordered by failure rate on the 720 base policy model rollouts used to author it. The evaluator\_instruction field is omitted for space.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td colspan="2">Objective</td></tr><tr><td>Advantage estimator</td><td>GRPO</td></tr><tr><td>Group-std normalization</td><td>yes (Eq. 1)</td></tr><tr><td>Value network</td><td>none; group mean</td></tr><tr><td>KL penalty</td><td>loss term, k3 (Schulman, 2020)</td></tr><tr><td>KL coefficient</td><td>0.01</td></tr><tr><td colspan="2">Batching</td></tr><tr><td>Prompts per step</td><td>16</td></tr><tr><td>Group size G</td><td>6</td></tr><tr><td>Rollouts per step</td><td>96</td></tr><tr><td>Minibatch size</td><td>8</td></tr><tr><td>PPO epochs per batch</td><td>1 (on-policy)</td></tr><tr><td>Training length</td><td>20 epochs (100 steps)</td></tr><tr><td>Checkpoint interval</td><td>every step</td></tr><tr><td colspan="2">Optimizer</td></tr><tr><td>Optimizer</td><td></td></tr><tr><td>Learning rate</td><td>AdamW 5×10−5</td></tr><tr><td>Schedule</td><td>constant, no warmup</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td colspan="2">Policy loss</td></tr><tr><td>Clip range (low / high)</td><td>0.2 / 0.2</td></tr><tr><td>Dual-clip constant</td><td>3.0</td></tr><tr><td>Entropy bonus</td><td>0</td></tr><tr><td>Loss aggregation</td><td>token-mean</td></tr><tr><td colspan="2">Rollout sampling</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Top-p / top-k</td><td>1.0 / unrestricted</td></tr><tr><td>Sampling</td><td>stochastic</td></tr><tr><td>Max assistant turns</td><td>50</td></tr><tr><td colspan="2">Sequences</td></tr><tr><td>Max prompt length</td><td>4096</td></tr><tr><td>Max response length</td><td>24576</td></tr><tr><td>Truncation side</td><td>left</td></tr><tr><td>Overlong prompts</td><td>filtered</td></tr><tr><td colspan="2">Trainable parameters</td></tr><tr><td>Adaptation</td><td>LoRA (Hu et al., 2022)</td></tr><tr><td>Rank / α</td><td>16 / 32</td></tr><tr><td>Target modules</td><td>all linear</td></tr><tr><td>Base models</td><td>Qwen3.6-27B,</td></tr><tr><td></td><td>Qwen2.5-32B-Instruct</td></tr><tr><td></td><td>(both frozen)</td></tr><tr><td colspan="2">Hardware</td></tr><tr><td colspan="2">GPUs per run</td></tr></table>

Table 10: Training hyperparameters, shared by all four settings and by both base policies.

# Listing 1: Rubric generation from the task (generate\_rubrics.txt).

You are an expert evaluation scientist specializing in LLM agent assessment. Your task is to generate a comprehensive set of ,→ evaluation rubrics for an AI agent given its system prompt.

In this environment, \`complete\_task\` does NOT end the task. The episode ends when the agent stops writing \`<code>...</code>\`. A ,→ brief closing turn after \`complete\_task\` is normal and correct - do not penalize it.

The agent is required to wrap code in \`<code>...</code>\` tags; do not create rubrics that penalize this required format.

## ## Context

Below is the full prompt given to the agent. It defines the agent's role, environment, available tools, expected behavior, and ,→ any explicit rules or constraints. Study it carefully -- every detail matters for rubric design.

## ## Your Task

Generate a set of evaluation rubrics that an evaluator LLM can use to score an agent's trajectory (the full multi-turn ,→ conversation of the agent attempting a task). Each rubric must be:

\- \*\*Discriminative\*\*: It reliably separates good trajectories from bad ones. Avoid rubrics where nearly all trajectories would ,→ score the same.

\- \*\*Observable\*\*: The evaluator can assess it purely from the trajectory text, without needing to run code or access external ,→ systems.

\*\*Avoid pure conformance / table-stakes rubrics.\*\* Do NOT generate rubrics that simply restate a basic requirement nearly every ,→ competent rollout already satisfies -- e.g. "uses only allowed/documented APIs", "stays within allowed tools", "wraps code ,→ correctly", "finalizes with the supervisor / complete\_task", "answer is in the requested format", "no unsupported guessing". ,→ These pass for almost every trajectory and provide no training signal. Only include a conformance dimension if violating it ,→ is a realistic, observed failure mode for THIS task -- and phrase it around that failure, not the generic rule. Spend your ,→ rubric budget on dimensions where trajectories genuinely differ in success.

## ## Rubric Design Principles

Follow these principles from rubric design literature:

1. \*\*MECE\*\* (Mutually Exclusive, Collectively Exhaustive): rubrics should cover different aspects without overlap, and together ,→ capture all dimensions of quality.

2. \*\*No overlapping\*\*: the same error from the agent shouldn't be punished multiple times. If two criteria would both fail ,→ because of one observable mistake, they measure the same thing and must be merged.

\- [GOOD] Agent authenticates correctly

\- [GOOD] Agent paginates to exhaustion when fetching lists

,→ thorough" -- instead specify the observable behavior: "the agent checks all pages until receiving an empty result."

5. \*\*Self-contained\*\*: each criterion should contain all information needed to evaluate it, so a judge seeing only that rubric' ,→ s instruction can score consistently.

6. \*\*Verifiable from trajectory alone\*\*: a judge should be able to decide the verdict from the trajectory text, without ,→ external search or domain knowledge beyond what the agent itself could access.

7. \*\*Ground rubrics in the prompt, not in abstract ideals.\*\* Every rubric should trace back to specific behaviors encouraged,

,→ demonstrated, or required by the agent prompt. Reference the relevant sections, rules, or demonstrated patterns where ,→ applicable.

8. \*\*Prioritize outcome rubrics.\*\* These rubrics will be used as reward signals for RL-based optimization of the agent. Outcome ,→ rubrics -- those that assess \*what the agent achieved\* (correctness of results, completeness of task execution, proper ,→ formatting of outputs, successful state changes) -- provide the strongest training signal. Prioritize these.

9. \*\*Supplement with behavioral rubrics where they predict outcomes.\*\* Behavioral rubrics (how the agent reasons, plans, or

,→ explores) are valuable when they capture patterns that \*causally contribute to or detract from task success\*. Include

,→ behavioral rubrics when: (a) they diagnose failure modes that outcome rubrics alone would miss, or (b) they provide

,→ intermediate signal for tasks where the final outcome is binary (pass/fail) but the agent was partially on the right track.

,→ Do not include behavioral rubrics that measure "good process" without a clear link to outcomes.

10. \*\*Target failure modes, not just best practices.\*\* For each rubric, consider: what does a bad trajectory look like on this ,→ dimension? If you can't articulate a clear failure mode, the rubric probably isn't discriminative enough.

11. \*\*Calibrate granularity.\*\* A rubric should not be so broad that it becomes a subjective overall quality judgment, nor so ,→ narrow that it only applies to a rare edge case. Aim for rubrics that are relevant to a meaningful fraction of tasks.

## ## Language Guidelines

Write rubrics in SIMPLE, CLEAR language:

\- Use plain English, not jargon or technical terms

\- Short, direct sentences (10-15 words per sentence)

```markdown
- Avoid complex vocabulary -- write for a general audience
- Be concrete and specific -- say exactly what to look for
- No abstract concepts -- describe visible actions
## Output Format
For each rubric, provide:
`json
[
{
"title": "<short descriptive title, 3-8 words>",
"description": "<1-3 sentences explaining what this rubric measures, why it matters, and what the spectrum from poor to
,→ excellent looks like.>",
"evaluator_instruction": "<Specific, actionable instruction for the evaluator LLM. Tell it exactly what to look for in the
,→ trajectory, what counts as good vs bad, and any edge cases to watch for. Be precise enough that two independent evaluators
,→ would give similar scores. Reference concrete patterns from the agent prompt where relevant.>",
"grounding_references": ["<list of specific sections, rules, examples, or patterns in the agent prompt that ground this
,→ rubric>"],
},
]
## Guidelines for the Rubric Set
Generate a maximum of **15 rubrics**. Enough to cover important dimensions, few enough that evaluation stays tractable.
Order rubrics by importance: **outcome rubrics first**, then behavioral rubrics that provide complementary signal.
Ensure the rubrics collectively cover distinct aspects of agent performance. Consider dimensions such as:
- Correctness and completeness of the final result
- Proper task completion and output formatting per the prompt's requirements
- Adherence to explicit rules and constraints in the prompt
- Correct use of tools, APIs, or environment features
- Handling of edge cases, errors, and unexpected situations
- Quality of reasoning and planning (where it predicts outcomes)
- Efficiency and avoidance of unnecessary or harmful actions
- Any domain-specific outcomes emphasized by the prompt
Not all dimensions will be equally relevant -- weight them based on what the prompt emphasizes.
- The `evaluator_instruction` is the most critical field. Write it as a clear briefing for an evaluator who has access to the
,→ trajectory but no other context beyond the agent prompt. Be concrete: reference specific tools, APIs, patterns, or rules
,→ from the prompt. Describe what to look for and what red flags indicate poor performance.
## Important Considerations
- The evaluator will see the full trajectory but NOT the ground-truth solution. Rubrics must be assessable without knowing the
,→ correct answer (except for format-level checks derivable from the prompt's instructions).
- Tasks may vary in complexity. Rubrics should be applicable across this range -- use language like "appropriate to the task
,→ complexity" rather than demanding fixed behaviors.
- The agent sometimes fails tasks. Rubrics should still provide useful signal on failed trajectories -- a well-reasoned near-
,→ miss should score differently from a completely off-track attempt.
- **RL optimization context**: These rubrics will serve as reward components. Design them so that higher scores on each rubric
,→ genuinely correlate with better agent performance. Avoid rubrics that could reward degenerate strategies (e.g., a "
,→ conciseness" rubric that rewards skipping necessary steps).
Now generate the rubrics. Think carefully about what actually distinguishes successful agent trajectories from unsuccessful
,→ ones given this specific prompt, and craft rubrics that will surface those differences.
```

Listing 2: Rubric generation from a trajectory (generate\_rubrics\_from\_trajectory.txt).  
You are an expert evaluation scientist specializing in LLM agent assessment. Your task is to generate evaluation rubrics by   
,→ analyzing an agent's actual execution trajectory.   
In this environment, \`complete\_task\` does NOT end the task. The episode ends when the agent stops writing \`<code>...</code>\`. A   
,→ brief closing turn after \`complete\_task\` is normal and correct - do not penalize it.   
The agent is required to wrap code in \`<code>...</code>\` tags; do not create rubrics that penalize this required format.   
## Context   
You are given two inputs:   
1. \*\*The agent's system prompt\*\* -- the instructions the agent was given.   
2. \*\*The agent's trajectory\*\* -- the actual multi-turn execution trace showing the agent's actions and the environment's   
,→ responses.   
<agent\_prompt>   
{{ agent\_prompt }}   
</agent\_prompt>   
<agent\_trajectory>   
{{ agent\_trajectory }}   
</agent\_trajectory>   
Additionally, rubrics have already been generated from the system prompt alone. Your job is to generate \*\*new rubrics that can

,→ only be discovered by observing actual execution\*\*, not by reading instructions.

<existing\_rubrics>

{{ existing\_rubrics }}

</existing\_rubrics>

## ## Your Task

Study the trajectory carefully. Look for patterns, mistakes, strengths, and weaknesses that reveal dimensions of quality \*\*not ,→ already covered\*\* by the existing rubrics. The trajectory shows you what actually happens when an agent runs -- this exposes ,→ quality dimensions that no amount of instruction-reading can predict.

\*\*Prioritize rubrics that THIS trajectory FAILS or only partially satisfies.\*\* These rubrics are used as RL reward signals, and ,→ a rubric only provides signal when a trajectory can fail it. A rubric this rollout clearly and fully passes contributes ,→ nothing to learning. So anchor each rubric on a real gap, mistake, shortfall, or skipped step you observe in THIS trajectory ,→ -- what the agent got wrong, omitted, did inefficiently, or only half-completed. If the trajectory is largely successful, ,→ look harder for the subtle shortfalls (a missed edge case, an unverified result, an unnecessary detour); do NOT manufacture ,→ rubrics describing things the agent plainly did well, since those will pass for every rollout and provide no signal.

## Each rubric you generate must be:

\- \*\*Informative\*\*: It captures a meaningful dimension of agent quality visible in execution.

\- \*\*Discriminative\*\*: It reliably separates good trajectories from bad ones.

\- \*\*Distinct\*\*: It does NOT overlap with the existing rubrics already extracted from the system prompt. Check each candidate ,→ rubric against the existing set and drop it if it's redundant.

\- \*\*Observable\*\*: The evaluator can assess it from the trajectory text alone.

## ## What to Look For in the Trajectory

Focus on execution-level patterns that only become visible when you watch the agent work:

\- \*\*How the agent recovers from mistakes.\*\* Does it get stuck in loops? Does it recognize errors and adjust? Does it try the ,→ same failing approach repeatedly?

\- \*\*How the agent sequences its actions.\*\* Does it gather information before acting, or act blindly and backtrack? Does it ,→ build on previous results or ignore them?

\- \*\*How the agent handles unexpected outputs.\*\* When an API returns something surprising, does the agent adapt or plow ahead ,→ with wrong assumptions?

\- \*\*Whether the agent's reasoning matches its actions.\*\* Does it say one thing in comments but do another in code? Are its ,→ stated plans coherent with what it actually executes?

\- \*\*Progress and momentum.\*\* Does the trajectory show steady progress toward the goal, or does it wander, stall, or go in ,→ circles?

\- \*\*Wasted effort.\*\* Does the agent repeat calls it already made? Does it fetch information it never uses? Does it take

\- \*\*How close the agent gets before failing\*\* (if it fails). Did it get 90% of the way there and trip at the last step, or was ,→ it lost from the start?

These are examples -- the trajectory may reveal other dimensions. Let the actual execution guide you.

## ## Rubric Design Principles

Follow these principles from rubric design literature:

1. \*\*MECE\*\* (Mutually Exclusive, Collectively Exhaustive): your new rubrics, together with existing rubrics, should cover ,→ different aspects without overlap, and together capture all dimensions of quality visible in execution.

2. \*\*No overlapping with existing rubrics\*\*: the same error from the agent shouldn't be punished multiple times. Before

,→ including a rubric, rigorously check: does an existing rubric already cover this? If two rubrics would both fail because of ,→ one observable mistake, they measure the same thing. When in doubt, drop it.

3. \*\*Atomicity\*\*: each rubric criterion should evaluate exactly one distinct aspect. Avoid bundling multiple criteria into a ,→ single rubric.

4. \*\*Specificity\*\*: criteria should be binary (pass/fail) and objective. Specify the observable behavior rather than vague ,→ descriptions.

5. \*\*Self-contained\*\*: each criterion should contain all information needed to evaluate it, so a judge seeing only that rubric' ,→ s instruction can score consistently.

6. \*\*Verifiable from trajectory alone\*\*: a judge should be able to decide the verdict from the trajectory text, without ,→ external search or domain knowledge.

7. \*\*Ground rubrics in observed patterns, not hypotheticals.\*\* Every rubric should be motivated by something you actually see ( ,→ or notably don't see) in the trajectory. Cite specific moments or patterns.

8. \*\*Prioritize outcome rubrics, supplement with behavioral ones.\*\* These rubrics serve as reward signals for RL training.

,→ Rubrics measuring what the agent achieved matter more than rubrics measuring how it thought. Include behavioral rubrics only ,→ when they causally predict success or failure.

9. \*\*Target failure modes visible in execution.\*\* The strongest rubrics name a specific mistake, gap, or shortfall the agent

,→ actually exhibits in THIS trajectory. Keep the failure as the anchor -- do not soften it into a generic "best practice" the ,→ agent already satisfies.

10. \*\*Generalize the WORDING, not the anchor.\*\* Phrase each rubric so a judge could apply it to any rollout of this task (avoid ,→ one-off specifics like exact variable names or values). But keep it anchored on the real shortfall you observed -- a rubric

,→ that is so generalized it becomes a broad quality dimension the agent already met provides no training signal. Generalize \*

,→ how\* it reads, not \*whether\* it can fail.

```markdown
## Language Guidelines
Write rubrics in SIMPLE, CLEAR language:
- Use plain English, not jargon or technical terms
- Short, direct sentences (10-15 words per sentence)
- Avoid complex vocabulary -- write for a general audience
- Be concrete and specific -- say exactly what to look for
- No abstract concepts -- describe visible actions
## Output Format
For each rubric, provide:
```json
{
"title": "<short descriptive title, 3-8 words>",
"description": "<1-3 sentences explaining what this rubric measures and why it matters.>",
"evaluator_instruction": "<Specific, actionable instruction for the evaluator LLM. Tell it exactly what to look for in the
,→ trajectory, what counts as good vs bad, and edge cases to watch for. Be concrete.>",
"trajectory_evidence": "<Quote or describe the specific moment(s) in the provided trajectory that motivated this rubric. This
,→ grounds the rubric in real observed behavior.>",
"grounding_references": ["<list of specific sections, rules, examples, or patterns in the agent prompt that ground this
,→ rubric>"]
}
## Guidelines
- Generate a maximum of **10 rubrics**. Quality over quantity -- only include rubrics that are clearly distinct from the
,→ existing set and clearly useful.
- Order by importance (strongest signal for task success first).
- If the trajectory doesn't reveal enough distinct dimensions to fill 10 rubrics, generate fewer. Do not pad with weak or
,→ redundant rubrics.
- The evaluator will see the full trajectory but NOT the ground-truth answer.
- These rubrics will be used alongside the existing ones, so together they should give a comprehensive picture of agent quality.
,→
Now analyze the trajectory, identify execution-level quality dimensions not covered by the existing rubrics, and generate new
,→ rubrics that capture them.
```

Listing 3: Merging the candidates into one canonical set (union\_rubrics.txt).  
![](images/d2ed9e8217f5fd9b7d6178ac49c16d16d51962d7569dff1ee4aaeffe81839cd2.jpg)

```csv
,→ capture all dimensions of quality.
2. **No overlapping**: the same error from the agent shouldn't be punished multiple times. If two criteria would both fail
,→ because of one observable mistake, they measure the same thing and must be merged.
3. **Atomicity**: each rubric criterion should evaluate exactly one distinct aspect. Avoid bundling multiple criteria into a
,→ single rubric. Criteria stacked with "and" can often be broken into separate rubrics.
4. **Specificity**: criteria should be binary (pass/fail) and objective. Avoid vague descriptions -- instead specify the
,→ observable behavior.
5. **Self-contained**: each criterion should contain all information needed to evaluate it, so a judge seeing only that rubric'
,→ s instruction can score consistently.
6. **Verifiable from trajectory alone**: a judge should be able to decide the verdict from the trajectory text, without
,→ external search or domain knowledge beyond what the agent itself could access.
## Guidelines for Merging
1. **Merge TRUE paraphrases / near-duplicates** into one rubric with the clearest wording.
2. **Merge overlapping rubrics** -- if two rubrics would BOTH fail because of the same observable agent mistake (even if titled
,→ differently), they are overlapping and must merge. The same error shouldn't be punished multiple times.
Example: "Agent gathers evidence from correct source" and "Agent applies correct selection logic" both fail when the agent
,→ uses liked songs instead of library -> they overlap, merge them.
3. **Keep DISTINCT failure modes SEPARATE** -- do NOT collapse criteria that fail for genuinely different reasons or at
,→ different moments in execution (e.g., authentication failure vs API parameter error vs missing pagination are distinct).
4. **Drop overly task-specific criteria** that cannot generalize to other rollouts of this task, and any rubric that penalizes
,→ the required `<code>...</code>` wrapping. In this environment `complete_task` does NOT end the task -- the episode ends when
,→ the agent stops writing `<code>...</code>`; a brief closing turn after `complete_task` is normal and must not be penalized.
5. **Drop vague criteria** that cannot be scored objectively and consistently.
5b. **Drop pure conformance / table-stakes criteria.** Cut rubrics that merely restate a basic requirement nearly every
,→ competent rollout already meets -- "uses only allowed/documented APIs", "stays within allowed tools", "wraps code correctly
,→ ", "finalizes with the supervisor / complete_task", "answer is in the requested format", "no unsupported guessing". They
,→ pass for almost every rollout and give no training signal. Keep such a dimension ONLY if violating it is a realistic failure
,→ mode seen in the candidate pool -- and phrase it around that failure.
6. **Prioritize outcome rubrics.** These rubrics serve as RL reward signals: rubrics measuring what the agent achieved matter
,→ more than how it reasoned. Keep behavioral rubrics only when they causally predict success or failure.
7. **Drop non-discriminative rubrics -- verified against the actual rollouts above.** For every rubric you are about to keep,
,→ mentally score it against each rollout in `<group_trajectories>`. If it would PASS for all of them (or is not-applicable to
,→ all of them), it carries no group-relative signal -- DROP it. Keep a rubric ONLY if at least one rollout in the group would
,→ clearly FAIL it. A rubric that all rollouts fail is fine (it is discriminative against the ideal). This is the single most
,→ important filter: do not keep a rubric whose verdict is identical across every rollout shown above.
## Language Guidelines
Write rubrics in SIMPLE, CLEAR language:
- Use plain English, not jargon or technical terms
- Short, direct sentences (10-15 words per sentence)
- Avoid complex vocabulary -- write for a general audience
- Be concrete and specific -- say exactly what to look for
- No abstract concepts -- describe visible actions
## Output Format
Return a JSON array of merged rubrics. There is a hard upper limit of **24** rubrics -- never return more than 24. This is a
,→ ceiling, NOT a target: return as FEW rubrics as the task genuinely requires. Do NOT pad toward 24 -- in practice a well-
,→ merged set is usually far smaller. Every rubric you keep must independently earn its place by satisfying ALL the merging and
,→ quality principles above (a distinct, non-overlapping failure mode; outcome-focused; objectively scorable; discriminative
,→ -- fails for at least one rollout). If adding a rubric would violate any of those principles, drop it rather than include it
,→ to reach a count. Adherence to the principles always wins over quantity. Order by importance (strongest signal for task
,→ success first). Each rubric must use exactly this schema and nothing else:
`json
{
"title": "<short descriptive title, 3-8 words>",
"description": "<1-3 sentences explaining what this rubric measures and why it matters, phrased generically for any rollout
,→ of this task.>",
"evaluator_instruction": "<Specific, actionable instruction for the evaluator LLM. Tell it exactly what to look for in the
,→ trajectory, what counts as good vs bad, and edge cases to watch for. Be concrete.>"
Output ONLY the JSON array -- no surrounding prose. Now merge the candidate rubrics into the canonical set.
```  
Listing 4: Scoring a trajectory against the rubric set (score\_rubrics.txt).

You are an evaluator assessing the quality of an AI agent's execution on a task. You will score the agent's trajectory against ,→ a set of rubrics.

```markdown
In this environment, `complete_task` does NOT end the task. The episode ends when the agent stops writing `<code>...</code>`. A
,→ brief closing turn after `complete_task` is normal and correct - do not penalize it.
## Task Input
This is the task the agent was given:
<task_input>
{{ task_input }}
</task_input>
## Agent Trajectory
This is the full execution trace -- the agent's actions and the environment's responses, in order:
<agent_trajectory>
{{ agent_trajectory }}
</agent_trajectory>
## Rubrics
Score the trajectory on each of the following rubrics. For each rubric, you are given a title, description, and specific
,→ evaluation instructions.
<rubrics>
{{ rubrics }}
</rubrics>
## Scoring Instructions
**Make sure your evaluation is as objective and consistent as it could be. By consistent we mean that a different evaluator's
,→ assessment of the task should agree with yours.**
For each rubric, do the following in order:
1. **Identify evidence.** Find specific moments in the trajectory that are relevant to this rubric. Quote or reference them
,→ briefly.
2. **Assess.** Based on the evidence (or lack of it), decide whether the agent met the rubric's criteria. Follow the evaluator
,→ instructions for each rubric closely -- they tell you what to look for and what counts as good vs bad.
3. **Score.** Assign a score of -1 (fail), 0 (not applicable), or +1 (pass).
- **+1 (pass)**: The trajectory clearly satisfies the rubric's criteria based on the evidence.
- **-1 (fail)**: The trajectory clearly violates the rubric's criteria, or there is no evidence of meeting it.
- **0 (not applicable)**: The rubric cannot meaningfully be assessed for this trajectory because the task never touches the
,→ dimension it measures (e.g. a date-handling rubric on a task with no dates, an error-recovery rubric when no errors occurred
,→ and the rubric only measures recovery quality rather than absence of errors). Use sparingly -- only when scoring would be
,→ genuinely arbitrary.
- **Conditional rubrics whose trigger never occurred MUST be 0, not +1.** If a rubric only applies when some situation
,→ arises (the agent hits an error, results are paginated, an edge case appears) and that situation never happened in this
,→ trajectory, score it 0 (not applicable) -- do NOT award +1 for trivially "satisfying" it by never being tested. Awarding +1
,→ here is a false pass that provides no signal.
- When in doubt between -1 and 0, lean toward -1 (a genuinely violated or unmet rubric is a fail, not N/A). The 0 case above
,→ is specifically for conditional rubrics whose precondition was absent.
4. **Justify.** Write a brief explanation (1-3 sentences) of why you gave that score. Be specific -- mention what the agent did
,→ or failed to do. This justification will be used as feedback to improve the agent, so make it actionable: say what should
,→ have been done differently if the score is -1.
## Output Format
Return a JSON array with one entry per rubric, in the same order as the rubrics above:
`json
[
{
"rubric_title": "<title of the rubric>",
"score": -1, 0, or 1,
"evidence": "<brief quotes or references to specific parts of the trajectory>",
"justification": "<1-3 sentences explaining the score and what should change if score is -1>"
},
Score every rubric. Do not skip any. Do not add rubrics that were not listed.
```

Listing 5: Step attribution (credit\_assignment.txt).  
```jinja
You are an expert at credit assignment for multi-step agent trajectories.
You are given:
1. A trajectory of {{ num_steps }} steps (each step is one code execution by the agent)
2. Per-rubric evaluation scores (PASS/FAIL) with evidence from a prior scoring stage
Your task: For each rubric, identify which steps in the trajectory were **causally responsible** for that rubric's PASS or FAIL
,→ verdict. A step is relevant if removing or changing it would have altered the rubric outcome.
## Trajectory Steps
```

```jinja
{{ steps_summary }}
## Rubric Scores ({{ num_rubrics }} rubrics)
{{ rubric_scores }}
## Instructions
For each rubric above, determine:
1. Whether its verdict is PASS, FAIL, or NOT_APPLICABLE (confirm or override the given verdict based on your analysis)
2. Which specific step numbers (1 to {{ num_steps }}) directly contributed to that outcome
3. A brief explanation of the causal link between those steps and the verdict
Guidelines for step attribution:
- A step is relevant if it DIRECTLY caused or enabled the rubric outcome
- Steps that merely set up context (e.g., reading docs) are relevant only if the rubric specifically evaluates that behavior
- For FAIL verdicts: identify steps where the agent made the critical mistake(s)
- For PASS verdicts: identify steps where the agent took the correct action(s)
- A step can be relevant to multiple rubrics
- **Every PASS or FAIL rubric MUST cite at least one relevant step** -- `relevant_steps`
must be non-empty. This is required so each rubric's verdict contributes to the
per-step reward signal. Even for a rubric that evaluates an ABSENCE (e.g. "agent
never did X" or "agent failed to do Y"), attribute it to the step(s) where the
omission was most consequential -- the moment the agent should have acted but did
not, or the final step by which the required action should have occurred. Do NOT
return an empty `relevant_steps` list for a PASS or FAIL rubric.
- Only a NOT_APPLICABLE rubric may have an empty `relevant_steps` list. Mark a rubric
NOT_APPLICABLE only if it does not meaningfully apply to this trajectory at all.
## Output Format
Return a JSON array with one entry per rubric (in the same order as the rubrics above):
```json
[
{
"rubric_title": "exact title from above",
"rubric_index": 0,
"verdict": "PASS",
"relevant_steps": [3, 5],
"explanation": "1-2 sentences explaining why these steps caused the PASS"
},<sub>{</sub>
"rubric_title": "exact title from above",
"rubric_index": 1,
"verdict": "FAIL",
"relevant_steps": [7, 8],
"explanation": "1-2 sentences explaining why these steps caused the FAIL"
}
]
Return ONLY the JSON array, no other text.
```

## Listing 6: The AppWorld agent system prompt, Qwen3.6-27B variant (prompts.py).

```prolog
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely
,→ autonomously.
To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this
,→ you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and
,→ the environment will execute it and show you the result, based on which, you will write python code for the next step and
,→ so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my
→behalf.
Here are three key APIs that you need to know to get more information
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())
# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code,
,→ that the environment will execute, to solve the task.
Here is an example:
My name is: supervisor_first_name supervisor_last_name. My personal email is supervisor_email and phone number is
,→ supervisor_phone_number.
```

Your task is: What is the password for my Spotify account?   
Code 1:   
print(apis.api\_docs.show\_app\_descriptions())   
Result 1:   
[   
{   
"name": "api\_docs",   
"description": "An app to search and explore API documentation."   
},<sub>{</sub>   
"name": "supervisor",   
"description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and   
,→ h d k "   
},   
...   
{   
"name": "spotify",   
"description": "A music streaming app to stream songs and manage song, album and playlist libraries."   
},   
{   
"name": "venmo",   
"description": "A social payment app to send, receive and request money to and from others."   
},   
]   
Code 2:   
print(apis.api\_docs.show\_api\_descriptions(app\_name='supervisor'))   
Result 2:   
  
"show\_account\_passwords : Show your supervisor's account passwords."   
]   
Code 3:   
print(apis.api\_docs.show\_api\_doc(app\_name='supervisor', api\_name='show\_account\_passwords'))   
Result 3:   
{   
'app\_name': 'supervisor',   
'api\_name': 'show\_account\_passwords',   
'path': '/account\_passwords',   
'method': 'GET'   
'description': "Show your supervisor's app account passwords.",   
'parameters': [],   
'response\_schemas': {   
'success': [{'account\_name': 'string', 'password': 'string'}],   
'failure': {'message': 'string'}   
}   
}   
Code 4:   
print(apis.supervisor.show\_account\_passwords())   
Result 4:   
[   
{   
"account\_name": "spotify",   
"password": "dummy\_spotify\_pass"   
},<sub>{</sub>   
"account\_name": "file\_system",   
"password": "dummy\_fs\_pass"   
},   
]   
Code 5:   
# So the Spotify password is an entry in the \`passwords\` list with the account\_name=spotify.   
spotify\_password = [account\_password["account\_name"] == "spotify" for account\_password in passwords][0]["password"]   
print(spotify\_password)   
Result 5:   
dummy\_spotify\_pass   
Code 6:   
# When the task is completed, I need to call apis.supervisor.complete\_task(). If there is an answer, I need to pass it as an   
,→ argument \`answer\` with status="success". I will pass the spotify\_password as an answer.   
apis.supervisor.complete\_task(answer=spotify\_password, status="success")   
Result 6:   
Marked the active task complete.

1. The email addresses, access tokens and variables (e.g. spotify\_password) in the example above were only for demonstration. ,→ Obtain the correct information by calling relevant APIs yourself.

,→ consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.

2. Only generate valid code blocks, i.e., do not put them in \`\`\`...\`\`\` or add any extra formatting. Any thoughts should be put ,→ as code comments.

3. You can use the variables from the previous code blocks in the subsequent code blocks.

4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making ,→ any irreversible change.

5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting ,→ the underlying OS, file system or process are disabled. You will get an error if do call them.

7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use \`spotipy\` for ,→ Spotify. Remember, the environment only has the standard library.

8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its ,→ outputs must be as per this documentation.

9. For APIs that return results in "pages", make sure to consider all pages.

10. To obtain current date or time, use Python functions like \`datetime.now()\` or obtain it from the phone app. Do not rely on ,→ your existing knowledge of what the current date or time is.

11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to

12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.

13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards

15. You can also pass \`status="fail"\` in the complete\_task API if you are sure you cannot solve it and want to exit, e.g., \` ,→ apis.supervisor.complete\_task(answer=None, status="fail")\`.

16. Once you believe the task is complete, you MUST call \`apis.supervisor.complete\_task()\` to finalize it. If the task requires

My name is: {supervisor\_first\_name} {supervisor\_last\_name}. My personal email is {supervisor\_email} and phone number is { ,→ supervisor\_phone\_number}.