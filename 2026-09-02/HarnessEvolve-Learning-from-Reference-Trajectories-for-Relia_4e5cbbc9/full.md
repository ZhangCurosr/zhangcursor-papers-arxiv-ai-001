# HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution

Wen Jiang<sup>∗</sup>, Mingmin Chu<sup>∗</sup>, Yimeng Tian<sup>∗</sup>, Qianxin Zhang<sup>∗</sup>, Haofei Yang Rui Yang, Yang Liu, Tao Lv, Fangming Li

ICT AI Competence Center, Huawei Technologies, Shanghai, China

{yangrui235, lifangming1}@huawei.com

<sup>∗</sup>Equal contribution. Corresponding authors: Rui Yang, Fangming Li

## Abstract

Self-evolving agents advance toward autonomy by optimizing their harness— prompts, skills, tools, and execution logic—based on environmental feedback. This paradigm, however, is hampered by three challenges: credit assignment failure, where terminal success/failure feedback makes it ambiguous which step caused the error; shortcut learning, where agents memorize task-specific patterns rather than acquire generalizable capabilities; and catastrophic forgetting, where unguarded updates degrade previously acquired competence. In this paper, we introduce HarnessEvolve, a self-evolving framework that learns from reference trajectories to achieve reliable agent self-evolution. HarnessEvolve decouples the execution agent from the evolutionary pipeline, assigning execution, evaluation, optimization, and gating to independent agent modules, enabling generalizable and stable harness improvements. Specifically, HarnessEvolve overcomes credit assignment failure by generating reference trajectories (execution paths produced when given the ground-truth answers) and aligning failed executions against them to extract error signals, which are clustered to reveal systematic failure patterns. To prevent shortcut learning and catastrophic forgetting, candidate harness updates must pass two gates: a quality gate that filters data leakage and prompt bloat, and a performance gate that accepts each update if it improves on the current batch without degrading recent batches, with epoch-end validation on a held-out set selecting the best-performing accepted agent snapshot. We conduct extensive experiments on several benchmarks spanning open-domain and enterprise scenarios, using different models and agent frameworks. Results demonstrate that HarnessEvolve consistently outperforms state-of-the-art baselines across all benchmarks and settings, confirming reliability across task domains.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in reasoning, planning, and tool use, enabling the development of autonomous agents that tackle complex tasks across domains ranging from software engineering to enterprise data analysis. At the core of such agents is their harness: the prompts, skills, tools, and execution logic that collectively shape their capabilities and execution behavior.

The design of such harnesses has evolved through several waves of decreasing reliance on human priors. Early systems relied on handcrafted workflows [Wei et al., 2022, Yao et al., 2023b], where human experts explicitly design execution logic, search topologies, and collaboration structures, while LLMs serve as execution nodes within predefined boundaries. While this paradigm offers high controllability, it lacks flexibility: deviations from anticipated workflows can cause the entire pipeline to fail, and agents remain bounded within human-designed frameworks, unable to expand without human intervention [Sumers et al., 2024]. Skill-augmented agents relax this rigidity: humans provide skills (described in natural language) and models operate within autonomous perception-planningaction-observation loops, exemplified by systems such as Claude Code and OpenClaw. This paradigm enables dynamic trial-and-error and self-healing through iterative reasoning, but it remains bounded by the quality of human-provided skills and can become stuck in iterative retry loops when initial planning decisions are incorrect. These limitations motivate the pursuit of self-evolving agents that can autonomously synthesize, refine, and iteratively improve their own harness from environmental feedback, progressively reducing reliance on human priors.

Despite recent progress, self-evolving agents remain difficult to deploy in real-world settings. Existing frameworks typically provide only sparse, terminal rewards (i.e., a binary success/failure signal at the end of execution), without mechanisms for fine-grained trajectory analysis to indicate which intermediate actions were correct or incorrect. Under such sparse signals, models tend to take shortcuts and overfit to specific evaluation instances, while unverified updates compound over iterations, gradually degrading prior capabilities. We identify three fundamental challenges that hinder reliable self-evolution:

• Credit Assignment Failure: In long-horizon tasks, agents receive only binary success/failure signals at the end. When a failure occurs, the agent cannot identify which specific action caused it, because later errors are downstream consequences of earlier mistakes. Solely reflecting on the whole trajectory may misidentify the root cause, leaving optimization directionless.

• Shortcut Learning: Without safeguards, agents tend to hardcode answers or inject excessive task-specific in-context examples into the harness, causing data leakage and prompt bloat that inflate training accuracy without genuine capability gain.

• Catastrophic Forgetting: Since each harness update is optimized on a subset of trajectory data, it may conflict with previously acquired competence, gradually degrading existing capabilities over multiple iterations.

These challenges cannot be resolved by designing yet another iterative prompt optimization algorithm; they necessitate a system-level architectural shift. To this end, we introduce HarnessEvolve, a self-evolving framework that learns from reference trajectories to achieve reliable agent self-evolution. Rather than entangling execution and optimization in a single reasoning loop, HarnessEvolve decouples the execution agent from the evolutionary pipeline, assigning execution, evaluation, optimization, and gating to independent modules, ensuring that optimization decisions are auditable and cannot be gamed by the execution agent.

Concretely, the execution agent runs tasks using its harness, which serves as the target of optimization. An evaluation agent assesses both the accuracy of the execution agent and the reliability of reference trajectories, the latter produced by the execution agent given the question and its ground-truth answer. Based on these assessments, an optimization agent analyzes failed executions by comparing them against reference trajectories to identify the first point of divergence, isolating the root-cause action from downstream cascading errors; these error signals are aggregated via error clustering to identify systematic failure patterns. A gate agent then inspects candidate updates for data leakage (by comparing content against failed training instances) and prompt bloat (by counting injected in-context examples), rejecting updates that exhibit either. Finally, the performance gate accepts each candidate only if it improves on the current batch without degrading recent batches, with epoch-end validation on a held-out set selecting the best-performing accepted agent snapshot.

We evaluate HarnessEvolve against several self-evolving baselines (GEPA [Agrawal et al., 2025], ACE [Zhang et al., 2026b], and SkillOpt [Yang et al., 2026]), each optimizing only a single component of the harness (e.g., SkillOpt optimizes solely skill.md), on five benchmarks spanning open-domain tasks (SearchQA [Dunn et al., 2017], OfficeQA [Opsahl-Ong et al., 2026], SpreadsheetBench [Ma et al., 2024b]) and specialized enterprise QA and text-to-SQL tasks (CloudCoreNetwork-QA and Wireless-QA, two in-house datasets from cloud core network and wireless network domains). HarnessEvolve consistently outperforms all baselines on all five benchmarks in terms of accuracy. Notably, on CloudCoreNetwork-QA, HarnessEvolve outperforms the strongest baseline by 21.6 percentage points, demonstrating substantial gains on complex multi-skill enterprise tasks. In summary, the main contributions of HarnessEvolve are fourfold:

• Reference-Guided Error Diagnosis: A diagnostic method that overcomes credit assignment failure by comparing failed executions against reference trajectories. The resulting error signals are aggregated via error clustering to identify systematic failure patterns.

• Quality Gate against Shortcut Learning: A filtering mechanism that inspects candidate updates for data leakage and prompt bloat. Only harness updates that pass both checks are retained for downstream validation.

• Performance Gate for Stable Evolution: A validation-gated update protocol that mitigates catastrophic forgetting. Each harness candidate is accepted into an agent snapshot pool only if it improves on the current batch without degrading recent batches, with epoch-end validation on a held-out set selecting the best-performing snapshot.

• Comprehensive and Reliable Harness Self-Evolution: Built on a decoupled multi-agent architecture, HarnessEvolve constitutes a comprehensive and reliable harness self-evolution framework that optimizes the entire harness rather than a single component, achieving state-of-the-art accuracy on both in-house enterprise and open-source benchmarks.

## 2 Related Work

Handcrafted agents rely on human-designed static workflows, where experts define prompt templates, reasoning structures, search topologies, and collaborative pipelines [Wei et al., 2022, Yao et al., 2023b, Press et al., 2023, Yao et al., 2023a, Besta et al., 2024, Hong et al., 2024, Qian et al., 2024, Wu et al., 2023, Schick et al., 2023]. Representative systems include LangChain/LangGraph and visual builders like Flowise and Dify. While offering high controllability, this paradigm confines agents within predefined boundaries and scales poorly as task complexity grows [Khattab et al., 2024, Zhou et al., 2024, Sumers et al., 2024, Liu et al., 2024, Ma et al., 2024a].

Skill-augmented agents relax this rigidity: humans provide skills in natural language while models operate within autonomous reasoning loops [Yao et al., 2023b], exemplified by Claude Code and OpenClaw. While enabling dynamic trial-and-error, this paradigm faces a critical limitation: agent performance remains heavily dependent on the quality of human-provided skills [Yang et al., 2026, Liang et al., 2026], creating a bottleneck for scalability. These limitations motivate self-evolving systems that can autonomously synthesize, refine, and iteratively improve their own harness from environmental feedback, progressively reducing reliance on human priors.

Self-evolving agents aim to autonomously optimize their inner logic. Current approaches optimize along multiple dimensions: workflow-level architecture design [Zhang et al., 2025, Liu et al., 2025], prompt-level optimization (e.g., OPRO [Yang et al., 2023], APO [Pryzant et al., 2023], and GEPA [Agrawal et al., 2025]), context-level evolution such as Reflexion [Shinn et al., 2023], ExpeL [Zhao et al., 2024], and ACE [Zhang et al., 2026b], and tool/skill-level refinement including Voyager [Wang et al., 2024] and SkillOpt [Yang et al., 2026]. However, each optimizes only a single component of the harness independently, leaving the rest untouched and limiting the space of achievable improvements.

More recently, methods that directly evolve a more comprehensive harness system have emerged: Self-Harness [Zhang et al., 2026a] introduces an iterative loop of weakness mining, harness proposal, and regression-testing-based validation, while Agentic Harness Engineering (AHE) [Lin et al., 2026] evolves coding-agent harnesses through observability-driven falsifiable contracts. Despite their advances, these methods keep credit assignment unresolved (i.e., no reference-based rootcause localization), leave shortcut learning unguarded (i.e., no inspection of data leakage or prompt bloat), and only partially mitigate catastrophic forgetting (i.e., no agent snapshot pool)—gaps that HarnessEvolve addresses.

## 3 Methodology

## 3.1 Overview

We consider the problem of agent self-evolution. Given a task corpus $\mathcal { T } = \{ ( q _ { i } , a _ { i } ^ { * } ) \} _ { i = 1 } ^ { M }$ of M queryanswer pairs (where $a _ { i } ^ { * }$ is the ground-truth answer to query $q _ { i } )$ split into training $\mathcal { T } _ { \mathrm { t r a i n } } .$ validation $\mathcal { T } _ { \mathrm { v a l } } .$ , and test $\mathcal { T } _ { \mathrm { t e s t } }$ sets, and a base execution agent $A _ { \mathrm { e x e c } }$ powered by a frozen LLM that runs tasks from $\tau ,$ the goal is to find the optimal execution agent $\mathcal { A } ^ { \ast }$ by iteratively optimizing its harness— prompts, skills, tools, and execution logic—on $\tau _ { \mathrm { t r a i n } }$ without human intervention, such that accuracy on the validation set $\mathcal { T } _ { \mathrm { v a l } }$ is maximized. The test set $\mathcal { T } _ { \mathrm { t e s t } }$ is used to evaluate the final accuracy of $\ b { A } ^ { * }$

![](images/ddf7225c2c62eca0112fbd844efbd940b024d381578be5748822bda3198b8148.jpg)  
Figure 1: Overview of the HarnessEvolve self-evolution framework. Four agents— $- \mathcal { A } _ { \mathrm { e x e c } } , \mathcal { A } _ { \mathrm { e v a l } } , \mathcal { A } _ { \mathrm { o p t } } .$ $\mathcal { A } _ { \mathrm { g a t e } ^ { - } }$ —collaborate in a batch-driven loop: the execution agent produces trajectories, the evaluation agent identifies failures and verifies references, the optimization agent diagnoses and clusters errors to generate candidate harness updates, and the gate agent enforces quality and performance check before accepting updates into the snapshot pool V.

HarnessEvolve addresses this through an architecture (Figure 1) that separates the execution agent from the evolutionary pipeline, assigning execution, evaluation, optimization, and gating to independent agent modules:

• Execution: The execution agent $A _ { \mathrm { e x e c } } ,$ the target of optimization, runs tasks from $\tau _ { \mathrm { t r a i n } }$ using its harness. For each task in $\mathcal { T } _ { \mathrm { t r a i n } } .$ , the execution agent is also run with the question and its ground-truth answer to produce reference trajectories that reach the correct answer.

• Evaluation: The evaluation agent $\mathcal { A } _ { \mathrm { e v a l } }$ performs two assessments: (1) it evaluates the task accuracy of the execution agent and identifies failed trajectories; (2) it verifies that reference trajectories are genuine execution paths rather than shortcuts where the agent directly returns the known answer, ensuring reference trajectory reliability.

• Optimization: The optimization agent $\mathcal { A } _ { \mathrm { o p t } }$ analyzes failed trajectories and, when reference trajectories are available, compares them to identify the root cause of failures. It clusters batch-level errors to identify systematic error patterns, and based on these diagnoses, proposes targeted modifications for the harness of $A _ { \mathrm { e x e c } }$

• Gating: The gate agent $\mathcal { A } _ { \mathrm { g a t e } }$ enforces two checks for the harness modifications proposed by $A _ { \mathrm { o p t } } \colon ( 1 )$ a data leakage and prompt bloat inspection, where modifications violating these constraints are returned to $\mathcal { A } _ { \mathrm { o p t } }$ for revision; (2) a performance check, where a modification is accepted only if it improves accuracy on the current batch without degrading performance on $R$ previously executed batches.

To optimize the harness of the base execution agent $A _ { \mathrm { e x e c } } .$ , HarnessEvolve runs in a batch-driven loop. At the start of each epoch, $\tau _ { \mathrm { t r a i n } }$ is randomly permuted and partitioned into mini-batches B = PermuteAndPartition $( \mathcal { T } _ { \mathrm { t r a i n } } , b ) = \{ B _ { 1 } , B _ { 2 } , \dotsc \}$ , where b is the batch size and $B _ { j }$ denotes the j-th mini-batch. For each mini-batch $B _ { j }$ , the four agents collaborate in sequence: the execution agent produces trajectories, the evaluation agent identifies failed trajectories, the optimization agent generates candidate modifications, and the gate agent validates and accepts beneficial modifications, storing each accepted harness as a snapshot in a pool V. After all mini-batches in B complete, the best-performing snapshot on $\mathcal { T } _ { \mathrm { v a l } }$ is selected as the final harness of the epoch, and the optimization loop continues for several epochs.

## 3.2 Execution

The execution agent $A _ { \mathrm { e x e c } }$ runs tasks from $\tau _ { \mathrm { t r a i n } }$ using its harness, the collection of prompts, skills, tools, and execution logic that defines its behavioral space. For each query $q _ { i }$ in $\mathcal { T } _ { \mathrm { t r a i n } } , \mathcal { A } _ { \mathrm { e x e c } }$ produces a trajectory $\tau _ { i } = [ ( o _ { 1 } , a _ { 1 } ) , ( o _ { 2 } , a _ { 2 } ) , \dots , ( o _ { T _ { i } } , a _ { T _ { i } } ) ]$ ], where $o _ { t }$ and $a _ { t }$ denote the observation and action at step t, and $T _ { i }$ is the trajectory length.

To address credit assignment failure, prior to the optimization loop, $A _ { \mathrm { e x e c } }$ is also run on all tasks in $\tau _ { \mathrm { t r a i n } }$ with both the questions and their ground-truth answers. For each task, $A _ { \mathrm { e x e c } }$ attempts up to $T _ { \mathrm { a t t } }$ times; after each attempt, the evaluation agent $A _ { \mathrm { e v a l } }$ verifies whether the produced trajectory is genuine (Section 3.3). Once a trajectory passes verification, it is stored as the verified reference trajectory $\tau _ { i } ^ { + }$ in a global cache $\mathcal { C } .$ If all $T _ { \mathrm { a t t } }$ attempts fail to produce the correct answer or to pass $\mathbf { \mathcal { A } } _ { \mathrm { e v a l } } \mathbf { \dot { s } }$ verification, the task is left without a reference trajectory and falls back to single-trajectory analysis during optimization. These reference trajectories provide successful execution paths that can be compared against failed trajectories to pinpoint the first point of divergence (the specific action where $A _ { \mathrm { e x e c } }$ went wrong) rather than attributing failure to the entire trajectory. This reference-guided comparison enables fine-grained error diagnosis that cannot be achieved by reflecting on failed trajectories alone.

## 3.3 Evaluation

The evaluation agent $A _ { \mathrm { e v a l } }$ serves a dual role: assessing execution accuracy during optimization and verifying reference trajectory genuineness during reference trajectory generation. Both assessments are essential for reliable error diagnosis.

Accuracy Assessment. During each optimization batch, $A _ { \mathrm { e v a l } }$ collects the trajectories produced by $A _ { \mathrm { e x e c } }$ on the current mini-batch $B _ { j }$ , and compares the final output of each trajectory against the corresponding ground-truth answer $a _ { i } ^ { * }$ . Trajectories that fail to produce the correct answer are collected as the failed set $F _ { j } = \{ ( q _ { i } , \tau _ { i } ^ { - } ) \}$ , which serves as input to the error diagnosis pipeline (Section 3.4). Moreover, the accuracy assessment on $B _ { j }$ also determines whether the candidate harness updates are accepted by the performance gate (Section 3.5).

Reference Verification. Since $A _ { \mathrm { e x e c } }$ is given the ground-truth answer during reference trajectory generation, it might produce a trajectory that shortcuts through the reasoning: for example, directly outputting the answer without invoking the necessary tools or following the expected workflow. To prevent this, $\mathcal { A } _ { \mathrm { e v a l } }$ verifies each produced trajectory before it enters the cache $\mathcal { C } .$ . The verification checks that the trajectory follows a legitimate reasoning chain: $A _ { \mathrm { e x e c } }$ invokes appropriate tools, processes intermediate observations, and arrives at the answer through valid steps rather than trivially restating the provided answer. Only verified trajectories are stored as $\tau _ { i } ^ { + }$ ; unverified ones are rejected, and $A _ { \mathrm { e x e c } }$ continues to the next attempt (Section 3.2).

## 3.4 Optimization

The optimization agent $\mathcal { A } _ { \mathrm { o p t } }$ analyzes failed trajectories, compares them against reference trajectories to identify root causes, clusters errors into systematic patterns, and generates candidate modifications to the harness.

Trajectory Comparison. Given a failed trajectory $\tau _ { i } ^ { - }$ , the question $q _ { i } ,$ , and a verified reference trajectory $\tau _ { i } ^ { + }$ (if available), $\mathcal { A } _ { \mathrm { o p t } }$ produces a structured error signal $\mathcal { F } _ { i } = ( s _ { i } , m _ { i } , h _ { i } )$ , where $s _ { i }$ is the severity, $m _ { i }$ is the error cause (e.g., tool hallucination, argument omission, premature termination), and $h _ { i }$ is a natural-language fix hint. When $\tau _ { i } ^ { + }$ is available, $\mathcal { A } _ { \mathrm { o p t } }$ can identify more fine-grained differences by comparing $\tau _ { i } ^ { + }$ and $\tau _ { i } ^ { - }$ , such as the first action divergence point $t _ { i } ^ { * }$ (the earliest step at which the action in $\tau _ { i } ^ { - }$ deviates from that in $\tau _ { i } ^ { + } )$ , enabling more accurate error cause identification. When no reference trajectory is available, $\mathcal { A } _ { \mathrm { o p t } }$ analyzes only $\tau _ { i } ^ { - }$ and $q _ { i }$ , outputting its best assessment of $s _ { i } , m _ { i }$ , and $h _ { i }$ without divergence-point localization. While less precise than reference-guided diagnosis, this fallback ensures that all failed instances contribute to the optimization signal.

Error Clustering. Individual error signals ${ \mathcal { F } } _ { i }$ are instance-level diagnoses that, if processed independently, would lead to fragmented and potentially conflicting modifications. To address this, $\mathcal { A } _ { \mathrm { o p t } }$ clusters error signals $\{ \mathcal { F } _ { i } \} _ { i = 1 } ^ { \lvert F _ { j } \rvert }$ by error cause $m _ { i }$ into groups $\mathcal { P } = \{ C _ { 1 } , \ldots , C _ { K } \}$ , where each $C _ { k } = ( \bar { s } _ { k } , \bar { m } _ { k } , \bar { e } _ { k } , \bar { h } _ { k } ) \colon \bar { s } _ { k }$ is the aggregated severity level, $\bar { m } _ { k }$ is the shared error cause, $\bar { e } _ { k }$ contains representative failed trajectories for $\bar { m } _ { k }$ , and $\bar { h } _ { k }$ is the suggested fix direction derived from the fix hints of its members. This aggregation enables the optimization agent to identify systematic failure patterns and generate coherent modifications. The clustering enforces three principles:

• Cause-Based Grouping: Groups are formed by error cause $m _ { i }$ rather than coarse severity labels, preventing semantically distinct errors from being merged. Thus, each cluster corresponds to a coherent fix strategy, enabling $\mathcal { A } _ { \mathrm { o p t } }$ to generate targeted modifications that uniformly address all instances within the cluster.

• Root-Cause Priority: When a reference trajectory is available, $\mathcal { A } _ { \mathrm { o p t } }$ is instructed to prioritize the first action divergence point $t _ { i } ^ { * }$ between $\tau _ { i } ^ { + }$ and $\tau _ { i } ^ { - }$ as the root error cause, since subsequent divergence points are likely cascading effects of the initial error rather than independent root causes. Clustering by the root cause ensures that $\mathcal { A } _ { \mathrm { o p t } }$ addresses the original failure rather than its downstream symptoms.

• Long-Tail Protection: Unlike conventional distance-based clustering $( { \mathrm { e . g . , ~ K . } }$ Means) [Xiong et al., 2006], which forces every sample into a fixed number of clusters, $\mathcal { A } _ { \mathrm { o p t } }$ preserves single-member clusters, ensuring rare but critical failure patterns are not absorbed into dominant clusters [Liu et al., 2019, Feldman, 2020].

The complete diagnosis and clustering procedure is formalized in Algorithm 1 (Appendix).

Candidate Generation. The clustering result $\mathcal { P }$ is provided to $\mathcal { A } _ { \mathrm { o p t } }$ , which generates targeted modifications to the current harness. $\mathcal { A } _ { \mathrm { o p t } }$ is instructed to prioritize root causes with more instances $( \mathrm { i . e . } ,$ , clusters containing more error signals $\mathcal { F } _ { i } )$ and higher severity. Based on the root cause, severity, and fix direction of each cluster, the agent synthesizes specific edits to the harness of $A _ { \mathrm { e x e c } }$ to address the root cause. The output is a candidate harness update set $\Delta _ { \mathrm { h a r n e s s } } ,$ , which is passed to the gate agent for validation.

## 3.5 Gating

Quality Gate. The gate agent $\mathcal { A } _ { \mathrm { g a t e } }$ ensures generalization by verifying that the optimization agent has not introduced shortcuts into the harness. The gate agent inspects candidate updates for data leakage and prompt bloat. For data leakage, the gate agent checks whether the edits proposed by the optimization agent directly embed failed queries and their ground-truth answers into the harness files, using LLM-as-judge with a score ranging from 0 to 1; if the score exceeds the threshold $\eta _ { \mathrm { l e a k } }$ , the update is rejected. For prompt bloat, the gate agent counts newly injected in-context examples; if the count exceeds the threshold $\eta _ { \mathrm { b l o } }$ , the update is rejected. A candidate update passes the quality gate if and only if, for every file $f$ in $\Delta _ { \mathrm { h a r n e s s } }$ (the set of harness files edited by the optimization agent), the data leakage score of $f$ does not exceed $\eta _ { \mathrm { l e a k } } ,$ and the number of injected in-context examples in $f$ does not exceed $\eta _ { \mathrm { b l o } } . ~ \mathrm { A n }$ update that passes yields a verified update $\Delta _ { \mathrm { v e r i f i e d } } ;$ a rejected update is returned to the optimization agent along with a rejection reason $\rho$ (a natural-language explanation of the violated constraint) for revision. If the gate remains unsatisfied after a fixed number of revisions, the update is abandoned $( \mathrm { i . e . , } \Delta _ { \mathrm { v e r i f i e d } } = \emptyset )$ . The gate agent operates as an isolated judge, structurally decoupled from the optimization agent, preventing self-serving evaluation.

Performance Gate. The verified update $\Delta _ { \mathrm { v e r i f i e d } }$ is used to construct a candidate agent $\mathcal { A } _ { \mathrm { c a n d i d a t e } } =$ $\mathrm { B u i l d } ( \mathcal { A } _ { \mathrm { c u r r e n t } } , \Delta _ { \mathrm { v e r i f i e d } } )$ , where $A _ { \mathrm { c u r r e n t } }$ denotes the current version of $A _ { \mathrm { e x e c } }$ being optimized. The candidate is evaluated on the current mini-batch $B _ { j }$ and a buffer of R recent mini-batches $\{ B _ { \operatorname* { m a x } ( 1 , j - R ) } , \dotsc , B _ { j - 1 } \}$ (empty when $j = 1 )$ . A candidate is accepted if and only if it improves accuracy on ${ \dot { B } } _ { j }$ by at least a margin δ while not degrading on any recent batch beyond tolerance ϵ:

$$
\mathcal { A } _ { \mathrm { c a n d i d a t e } } \mathrm { ~ i s ~ a c c e p t e d } \iff \left\{ \begin{array} { l l } { \mathrm { A c c } ( \mathcal { A } _ { \mathrm { c a n d i d a t e } } , B _ { j } ) - \mathrm { A c c } ( \mathcal { A } _ { \mathrm { c u r r e n t } } , B _ { j } ) \geq \delta , } \\ { \underset { l \in [ \operatorname* { m a x } ( 1 , j - R ) , j - 1 ] } { \operatorname* { m a x } } \left[ \mathrm { A c c } ( \mathcal { A } _ { \mathrm { c u r r e n t } } , B _ { l } ) - \mathrm { A c c } ( \mathcal { A } _ { \mathrm { c a n d i d a t e } } , B _ { l } ) \right] \leq \epsilon . } \end{array} \right.\tag{1}
$$

Accepted candidates are stored as snapshots in the pool $\mathbb { V }$ and immediately become the active agent $A _ { \mathrm { c u r r e n t } }$ for subsequent batches; performance-gate rejections increment a consecutive-rejection counter c. If c reaches the patience threshold $\bar { P _ { \mathrm { b a t c h } } }$ , the current epoch terminates early (bypassing remaining batches) and proceeds directly to epoch-end selection, thereby mitigating redundant computation upon convergence.

Table 1: Main results on in-house datasets. Accuracy (%) is reported. Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">CloudCoreNetwork-QA</td><td colspan="2">Wireless-QA</td></tr><tr><td>Qwen</td><td>DeepSeek</td><td>Qwen</td><td>DeepSeek</td></tr><tr><td>Base</td><td>43.4</td><td>47.5</td><td>79.0</td><td>85.9</td></tr><tr><td>GEPA</td><td>65.3</td><td>57.6</td><td>82.4</td><td>86.5</td></tr><tr><td>ACE</td><td>59.3</td><td>64.6</td><td>84.3</td><td>90.1</td></tr><tr><td>SkillOpt</td><td>61.9</td><td>65.3</td><td>89.0</td><td>89.3</td></tr><tr><td>HarnessEvolve</td><td>86.9</td><td>85.9</td><td>89.7</td><td>92.8</td></tr></table>

Epoch-End Selection. At epoch completion, all snapshots accumulated in the pool V are evaluated on the validation set $\mathcal { T } _ { \mathrm { v a l } }$ , and the best-performing one is selected:

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { c u r r e n t } }  \underset { \mathcal { A } \in \mathbb { V } } { \mathrm { a r g m a x } } \mathrm { A c c } ( \ r A , \mathcal { T } _ { \mathrm { v a l } } ) , } \\ { \mathcal { A } ^ { * }  \mathcal { A } _ { \mathrm { c u r r e n t } } . \quad \quad } \end{array}\tag{2}
$$

This two-tiered architecture (batch-level updates within each epoch and epoch-level selection on $\mathcal { T } _ { \mathrm { v a l } } )$ ensures that the final agent is no worse than the best-performing snapshot in the pool. The optimization loop terminates when the epoch count reaches E, or when the validation accuracy of $\bar { \mathcal { A } } ^ { * }$ fails to improve for $P _ { \mathrm { e p } }$ consecutive epochs, indicating convergence.

The complete HarnessEvolve self-evolution procedure (encompassing execution, evaluation, optimization, and gating) is formalized in Algorithm 2 (Appendix). After all epochs complete, the final agent $\ b { A } ^ { * }$ is evaluated on the test set $\mathcal { T } _ { \mathrm { t e s t } }$ to report accuracy.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate all methods on three open-source datasets (SearchQA [Dunn et al., 2017], OfficeQA [Opsahl-Ong et al., 2026], SpreadsheetBench [Ma et al., 2024b]) and two in-house datasets (CloudCoreNetwork-QA and Wireless-QA) to assess generalization across open-domain and enterprise scenarios. SearchQA is an open-domain QA dataset where agents retrieve and synthesize information from a search engine. OfficeQA is an enterprise-level document reasoning benchmark where agents parse, retrieve, and reason across unstructured text and tabular data to answer complex questions. SpreadsheetBench is a spreadsheet manipulation benchmark where agents perform data and formula operations. CloudCoreNetwork-QA and Wireless-QA are constructed from cloud core network and wireless network domains, where agents perform network-domain QA and text-to-SQL tasks such as KPI metric querying and network element configuration querying. Queries in both datasets require selecting and combining multiple skills.

Baselines. We compare HarnessEvolve against three state-of-the-art agent self-optimization frameworks, each operating at a different granularity: GEPA [Agrawal et al., 2025] performs prompt-level optimization via reflective prompt evolution, ACE [Zhang et al., 2026b] optimizes at the context level, and SkillOpt [Yang et al., 2026] refines skill descriptions at the text level. All baselines are deployed under identical environmental constraints and an initial harness to ensure fair comparison. We also report the accuracy of the agent with the unoptimized harness (Base) as a reference.

Implementation Details. We use two models of different scales as reasoning engines: Qwen3.6-27B, post-trained with domain-specific fine-tuning, and DeepSeek-V4-Flash, used without fine-tuning. On the in-house datasets (CloudCoreNetwork-QA and Wireless-QA), both models are used under our in house LAMAgent framework, where the harness being optimized includes the full project code, skills, prompts, and tools. On the open-source datasets (SearchQA, OfficeQA, and SpreadsheetBench), DeepSeek-V4-Flash is used under OpenClaw, where the harness being optimized comprises skills, AGENTS.md, SOUL.md, and tools. Unlike SkillOpt, which optimizes only skill.md, HarnessEvolve optimizes the entire skill directory including reference files and scripts. All methods use the same training, validation, and test split on each dataset. We set max reference attempts $T _ { \mathrm { a t t } } = 5$ in reference trajectory generation (Section 3.2), and batch patience $P _ { \mathrm { b a t c h } } = 1 0$ , epoch patience $P _ { \mathrm { e p } } = 5$ , epoch limit $E = 2 0 .$ , batch size $b = 4 0 ,$ replay buffer size $R = 2 .$ , step margin $\delta = 0 . 0 $ , degradation tolerance $\epsilon = 0 . 0 2 5$ , data leakage threshold $\eta _ { \mathrm { l e a k } } = 0 . 8$ , prompt bloat threshold $\eta _ { \mathrm { b l o } } = 5 ,$ and revision limit $T _ { \mathrm { r e v } } = 3$ in Algorithm 2 (Appendix).

Table 2: Main results on open-source datasets. Accuracy (%) is reported. Best results are in bold.
<table><tr><td>Method</td><td>SearchQA</td><td>OfficeQA</td><td>SpreadsheetBench</td></tr><tr><td>Base</td><td>86.5</td><td>62.8</td><td>44.3</td></tr><tr><td>GEPA</td><td>88.6</td><td>64.0</td><td>69.6</td></tr><tr><td>ACE</td><td>90.0</td><td>68.9</td><td>52.2</td></tr><tr><td>SkillOpt</td><td>89.4</td><td>66.9</td><td>74.6</td></tr><tr><td>HarnessEvolve</td><td>92.9</td><td>70.9</td><td>76.4</td></tr></table>

Table 3: Cross-framework transfer: skills optimized on OpenClaw, evaluated on four frameworks. Accuracy (%) is reported. Best results are in bold.
<table><tr><td rowspan="2">Framework</td><td colspan="2">SearchQA</td><td colspan="2">OfficeQA</td><td colspan="2">SpreadsheetBench</td></tr><tr><td>Base</td><td>HarnessEvolve</td><td>Base</td><td>HarnessEvolve</td><td>Base</td><td>HarnessEvolve</td></tr><tr><td>Hermes</td><td>95.0</td><td>95.0</td><td>77.9</td><td>80.8</td><td>72.1</td><td>73.2</td></tr><tr><td>OpenCode</td><td>90.0</td><td>92.7</td><td>75.6</td><td>80.3</td><td>57.5</td><td>87.9</td></tr><tr><td>LAMAgent</td><td>92.9</td><td>94.3</td><td>74.4</td><td>75.0</td><td>45.0</td><td>80.0</td></tr><tr><td>DeepSeek Harness</td><td>93.6</td><td>95.0</td><td>79.1</td><td>79.7</td><td>56.4</td><td>86.8</td></tr></table>

## 4.2 Results and Analysis

Main Results. Tables 1 and 2 compare HarnessEvolve against three baselines on the in-house and open-source datasets, respectively. HarnessEvolve consistently achieves the highest accuracy across all evaluated settings. Notably, on the in-house datasets, HarnessEvolve improves accuracy on CloudCoreNetwork-QA with Qwen3.6-27B from 43.4% (Base) to 86.9%, outperforming the strongest baseline GEPA at 65.3% by 21.6 percentage points, and reaches 92.8% on Wireless-QA with DeepSeek-V4-Flash, surpassing the best baseline ACE at 90.1% by 2.7 percentage points. On the open-source datasets, HarnessEvolve consistently achieves significant accuracy improvements over all baselines across all three benchmarks. These improvements on both enterprise and open-domain benchmarks confirm that the gains from reference-guided diagnosis and comprehensive harnes optimization generalize across task domains and model settings.

Cross-Framework Generalization. To evaluate whether the harness optimized by HarnessEvolve can transfer across frameworks, we take the skills optimized on SearchQA, OfficeQA, and SpreadsheetBench and deploy them directly on four frameworks to test on $\mathcal { T } _ { \mathrm { t e s t } }$ , namely Hermes, OpenCode, LAMAgent, and DeepSeek Harness, without re-optimization. As shown in Table 3, HarnessEvolveoptimized skills consistently improve or maintain accuracy over Base across all four frameworks. These results indicate that HarnessEvolve produces transferable improvements rooted in general error patterns rather than framework-specific shortcuts.

Self-Evolution Curve. Figure 2 traces the self-evolution procedures of HarnessEvolve, with each point corresponding to one optimization step evaluated on $\mathcal { T } _ { \mathrm { v a l } }$ . A substantial fraction of candidate updates are rejected by the two-tier gating mechanism, indicating effective filtering of low-quality modifications. The peak accuracy in the snapshot pool V increases gradually throughout evolution, reaching its highest point at the final harness $\lambda ^ { * }$ . This also demonstrates that the gating mechanism ensures strict quality control while retaining only beneficial updates.

Ablation Study. To assess the contribution of each component in HarnessEvolve, we conduct ablation studies on CloudCoreNetwork-QA using Qwen3.6-27B. We consider three variants:

• M1 (w/o reference trajectory): bypassing reference trajectory comparison and performing error diagnosis solely on failed trajectories. This variant isolates the contribution of referenceguided error diagnosis, reverting to a coarser single-trajectory diagnosis.

![](images/5677a3a40f9a6d1f0253b0aab56accac222f0a7d069b03fbc6a58b4a214da0b6.jpg)

![](images/71c407cf82f460e8a24f80c16d2c69c1578182ab6df57b7d7751875ca53eefb6.jpg)  
Figure 2: HarnessEvolve self-evolution curves on CloudCoreNetwork-QA and SpreadsheetBench. Green points denote accepted snapshots that pass both the quality gate and performance gate and are saved to the snapshot pool V; gray points denote rejected candidates. The blue curve tracks the peak accuracy in V. The red star marks the final harness $\ b { A } ^ { * }$

Table 4: Ablation study results on CloudCoreNetwork-QA (Qwen3.6-27B). Accuracy (%) is reported. Best results are in bold.
<table><tr><td>Variant</td><td>CloudCoreNetwork-QA</td></tr><tr><td>HarnessEvolve (full)</td><td>86.9</td></tr><tr><td>M1: w/o reference trajectory</td><td>57.8</td></tr><tr><td>M2: w/o error clustering</td><td>68.6</td></tr><tr><td>M3: w/o quality gate</td><td>80.1</td></tr></table>

• M2 (w/o error clustering): optimizing directly over instance-level error signals ${ \mathcal { F } } _ { i }$ without grouping. This variant tests whether aggregating errors into systematic patterns is necessary, as processing each error signal independently may lead to fragmented and conflicting modifications.

• M3 (w/o quality gate): removing the quality gate, accepting all candidate updates without filtering for data leakage and prompt bloat. This variant evaluates whether the quality gate meaningfully prevents shortcut learning, or whether the optimization agent alone can produce generalizable updates.

Table 4 presents the ablation results. M1, which removes reference trajectory comparison, yields the largest degradation from 86.9% to 57.8%, confirming that reference-guided diagnosis is the most critical component. Without this comparison, the agent misidentifies root causes and generates ineffective modifications. M2, which removes error clustering, shows significant degradation to 68.6%, confirming that aggregating instance-level errors into systematic patterns is essential for coherent modifications. M3, which removes the quality gate, shows moderate degradation to 80.1%, indicating that the quality gate helps prevent shortcut learning and overfitting to training data.

Qualitative Analysis. To inspect what HarnessEvolve modifies during self-evolution, Figure 3 (Appendix) compares the harness of the execution agent before and after optimization. The optimized harness exhibits targeted edits that span the entire execution agent, covering skill.md files, Python scripts within skills, prompt instructions, tool-argument specifications, and code related to agent execution logic, instead of being confined to a single component. This contrasts with SkillOpt [Yang et al., 2026], which optimizes only skill.md, and corroborates the comprehensiveness of the harness optimization of HarnessEvolve.

## 5 Conclusion

We have presented HarnessEvolve, a self-evolving framework that learns from reference trajectories for reliable agent self-evolution. By decoupling execution, evaluation, optimization, and gating into independent modules, HarnessEvolve addresses three fundamental challenges: credit assignment failure via reference-guided error diagnosis, shortcut learning via the quality gate, and catastrophic forgetting via the performance gate with epoch-end validation. Extensive evaluations on both open-domain and enterprise benchmarks demonstrate that HarnessEvolve consistently outperforms state-of-the-art baselines across all five benchmarks.

## References

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Daniel Klein, Matei Zaharia, and Omar Khattab. GEPA: reflective prompt evolution can outperform reinforcement learning. CoRR, abs/2507.19457, 2025.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Hoang, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of thoughts: Solving elaborate problems with large language models. In AAAI, 2024.

Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Güney, Volkan Cirik, and Kyunghyun Cho. Searchqa: A new q&a dataset augmented with context from a search engine. CoRR, abs/1704.05179, 2017.

Vitaly Feldman. Does learning require memorization? a short tale about a long tail. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing. Association for Computing Machinery, 2020.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for A multi-agent collaborative framework. In ICLR. OpenReview.net, 2024.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into state-of-the-art pipelines. In ICLR. OpenReview.net, 2024.

Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Xin Xu, Tongtong Wu, Kun Wang, Yang Liu, Zhen Bi, Jungang Lou, Yuchen Eleanor Jiang, Hangcheng Zhu, Gang Yu, Haiwen Hong, Longtao Huang, Hui Xue, Chenxi Wang, Yijun Wang, et al. Skillnet: Create, evaluate, and connect ai skills. arXiv preprint arXiv:2603.04448, 2026.

Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui, and Yu-Gang Jiang. Agentic harness engineering: Observabilitydriven automatic evolution of coding-agent harnesses. arXiv preprint arXiv:2604.25850, 2026.

Siwei Liu, Jinyuan Fang, Han Zhou, Yingxu Wang, and Zaiqiao Meng. SEW: self-evolving agentic workflows for automated code generation. CoRR, abs/2505.18646, 2025.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating llms as agents. In ICLR. OpenReview.net, 2024.

Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X. Yu. Large-scale long-tailed recognition in an open world. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2532–2541, 2019.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn LLM agents. In NeurIPS, 2024a.

Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang. Spreadsheetbench: Towards challenging real world spreadsheet manipulation. In NeurIPS, 2024b.

Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, et al. Officeqa pro: An enterprise benchmark for end-to-end grounded reasoning. arXiv preprint arXiv:2603.08655, 2026.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In EMNLP (Findings), volume EMNLP 2023 of Findings of ACL, pages 5687–5711. Association for Computational Linguistics, 2023.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with "gradient descent" and beam search. In EMNLP, pages 7957–7968. Association for Computational Linguistics, 2023.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. Chatdev: Communicative agents for software development. In ACL (1), pages 15174–15186. Association for Computational Linguistics, 2024.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In NeurIPS, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In NeurIPS, 2023.

Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive architectures for language agents. Trans. Mach. Learn. Res., 2024, 2024.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Trans. Mach. Learn. Res., 2024, 2024.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen LLM applications via multi-agent conversation framework. CoRR, abs/2308.08155, 2023.

Hui Xiong, Junjie Wu, and Jian Chen. K-means clustering versus validation measures: a data distribution perspective. In KDD, pages 779–784. ACM, 2006.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. CoRR, abs/2309.03409, 2023.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. Skillopt: Executive strategy for self-evolving agent skills. CoRR, abs/2605.23904, 2026.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS, 2023a.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR. OpenReview.net, 2023b.

Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, and Shuyue Hu. Self-harness: Harnesses that improve themselves. arXiv preprint arXiv:2606.09498, 2026a.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation. In ICLR. OpenReview.net, 2025.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations, volume 2026, pages 86069–86100, 2026b.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: LLM agents are experiential learners. In AAAI, pages 19632–19642. AAAI Press, 2024.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In ICLR. OpenReview.net, 2024.

## A Before-and-After Comparison of Harness Optimization

Figure 3 compares the harness before and after the optimization of HarnessEvolve across six cases: (a) skill.md modification on OfficeQA, (b) skill.md modification on CloudCoreNetwork QA, (c) script creation within the skill on SpreadsheetBench, (d) system prompt modification on CloudCoreNetwork-QA, (e) tool description modification on CloudCoreNetwork-QA, and (f) execution logic modification on CloudCoreNetwork-QA.

## B Reference-Guided Error Diagnosis and Clustering

Algorithm 1 details the error diagnosis and clustering procedure. For each failed instance, the optimization agent retrieves the corresponding reference trajectory from the global cache C and performs trajectory comparison to identify the first action divergence point as the root cause. When no reference trajectory is available, the agent falls back to isolated trajectory analysis. The collected error signals are then grouped via error clustering to preserve rare failure patterns.

Algorithm 1 Reference-Guided Error Diagnosis and Clustering   
Require: Failed instances $F _ { j } = \{ ( q _ { i } , \tau _ { i } ^ { - } ) \}$ , global cache C (containing verified reference trajectories   
$\tau _ { i } ^ { + }$ for queries that yielded a genuine execution path).   
Ensure: Error groups $\mathcal { P } = \{ C _ { 1 } , \ldots , C _ { K } \}$ , where each $C _ { k } = ( \bar { s } _ { k } , \bar { m } _ { k } , \bar { e } _ { k } , \bar { h } _ { k } )$   
1: Initialize error batch ${ \mathcal { M } }  \emptyset .$   
2: for $i = 1$ to $| F _ { j } |$ do   
3: $\tau _ { i } ^ { + } \gets { \mathcal { C } } [ q _ { i } ]$ ▷ look up pre-computed reference   
4: $\mathbf { i f } \tau _ { i } ^ { + } = \varnothing$ then   
5: $\mathcal { F } _ { i } \gets \mathrm { A }$ nalyzeTrajectory $\cdot ( \tau _ { i } ^ { - } , q _ { i } )$ ▷ without reference   
6: else   
7: ${ \mathcal { F } } _ { i }$ ← AnalyzeTrajectory $( \tau _ { i } ^ { - } , \tau _ { i } ^ { + } , q _ { i } )$ ▷ compare for fine-grained root cause   
8: end if   
9: ${ \mathcal { M } } \gets { \mathcal { M } } \cup \{ { \mathcal { F } } _ { i } \}$   
10: end for   
11: P ← ErrorCluster(M).   
12: return P

## C HarnessEvolve Self-Evolution Procedure

Algorithm 2 presents the complete self-evolution loop. Each epoch partitions the training set into minibatches and iterates through them. For each batch, the optimization agent generates candidate updates, the gate agent filters them, and the performance gate decides acceptance based on improvement margin δ and replay degradation ϵ. Accepted snapshots are added to the pool V. At epoch end, the best-performing snapshot on the validation set is selected as $\ b { A } ^ { * }$ . The loop terminates early when either batch-level patience $P _ { \mathrm { b a t c h } }$ or epoch-level patience $P _ { \mathrm { e p } }$ is exhausted.

![](images/5d7189f15c37d7c7a2dc3e50fb6cb1be1715a2bd52065411147fb5c20baaf4bb.jpg)  
(a) Skill.md (OfficeQA)

![](images/562ddd63652eaa28441ccdb14b51bab7d236dd3bc1d6660fabf17f6ccf9f1a81.jpg)  
(f) Execution logic (CloudCoreNetwork-QA)  
Figure 3: Before-and-after comparison of the harness optimization of HarnessEvolve across six cases: (a) skill.md modification on OfficeQA, (b) skill.md modification on CloudCoreNetwork-QA, (c) script creation within the skill on SpreadsheetBench, (d) system prompt modification on CloudCoreNetwork-QA, (e) tool description modification on CloudCoreNetwork-QA, and (f) execution logic modification on CloudCoreNetwork-QA. HarnessEvolve modifies the entire execution agent—including skill.md files, scripts, prompts, tool descriptions, and execution-logic code— rather than optimizing a single component in isolation.

Algorithm 2 HarnessEvolve Self-Evolution Procedure   
Require: Task corpora $\mathcal { T } _ { \mathrm { t r a i n } } , \mathcal { T } _ { \mathrm { v a l } } , \mathcal { T } _ { \mathrm { t e s t } } .$ , agents $( \mathcal { A } _ { \mathrm { e x e c } } , \mathcal { A } _ { \mathrm { e v a l } } , \mathcal { A } _ { \mathrm { o p t } } , \mathcal { A } _ { \mathrm { g a t e } } )$ , global cache $\mathcal { C }$ (con  
taining verified reference trajectories $\tau _ { i } ^ { + }$ for queries that yielded a genuine execution path), batch   
patience $P _ { \mathrm { b a t c h } }$ , epoch patience $P _ { \mathrm { e p } } .$ , step margin $\delta \geq { \mathrm { { 0 } } } ,$ degradation tolerance $\epsilon \geq 0 ,$ replay   
buffer size $R ,$ , filter bounds $( \eta _ { \mathrm { l e a k } } , \bar { \eta _ { \mathrm { b l o } } } )$ , epoch limit $E ,$ batch size $b ,$ revision limit $T _ { \mathrm { r e v } }$   
Ensure: Optimized agent $\ b { A } ^ { * }$ , snapshot pool $\begin{array} { r } { \bar { \mathbb { V } } . } \end{array}$   
1: Initialize $\mathcal { A } _ { \mathrm { c u r r e n t } }  \mathcal { A } _ { \mathrm { e x e c } } , \hat { \mathcal { A } } ^ { * }  \hat { \mathcal { A } } _ { \mathrm { e x e c } } ,$ snapshot pool $\mathbb { V }  \{ \mathcal A _ { \mathrm { e x e c } } \}$ , index $n \gets 0 ,$ rejection   
count $c  0 ,$ best validation accuracy $a _ { \mathrm { b e s t } }  \mathrm { A c c } ( A _ { \mathrm { e x e c } } , \mathcal { T } _ { \mathrm { v a l } } )$ , epoch non-improvement count   
$c _ { \mathrm { e p } } \gets 0 .$   
2: for $e = 1$ to $E$ do   
3: B ← PermuteAndPartition $( \tau _ { \mathrm { t r a i n } } , b )$   
4: for each batch $B _ { j } \in B$ do   
5: Run $\mathcal { A } _ { \mathrm { c u r r e n t } }$ on $B _ { j }$ to produce trajectories $\{ \tau _ { i } \}$   
6: $F _ { j } \gets \mathcal { A } _ { \mathrm { e v a l } } ( \{ \tau _ { i } \} , \mathbf { \bar { \boldsymbol { B } } } _ { j } )$ ▷ identify failed instances   
7: P ← DiagnoseAndCluster $( F _ { j } , \mathcal { C } )$ ▷ Algorithm 1   
8: $\Delta _ { \mathrm { h a r n e s s } }  A _ { \mathrm { o p t } } ( \mathcal { P } , \mathcal { A } _ { \mathrm { c u r r e n t } } ) \dot { }$   
9: $\Delta _ { \mathrm { v e r i f i e d } } , \rho \gets \dot { \mathcal { A } } _ { \mathrm { g a t e } } ( \Delta _ { \mathrm { h a r n e s s } } ; \eta _ { \mathrm { l e a k } } , \eta _ { \mathrm { b l o } } )$   
10: for $r = 1$ to $T _ { \mathrm { r e v } }$ do   
11: if $\Delta _ { \mathrm { v e r i f i e d } } \neq \emptyset$ then   
12: break   
13: end if   
14: $\Delta _ { \mathrm { h a r n e s s } }  \mathcal { A } _ { \mathrm { o p t } } ( \mathcal { P } , \mathcal { A } _ { \mathrm { c u r r e n t } } , \rho )$ ▷ revise with rejection feedback   
15: $\Delta _ { \mathrm { v e r i f i e d } } , \rho \gets \mathcal { \dot { A } } _ { \mathrm { g a t e } } ( \Delta _ { \mathrm { h a r n e s s } } ; \eta _ { \mathrm { l e a k } } , \eta _ { \mathrm { b l o } } )$   
16: end for   
17: if $\Delta _ { \mathrm { v e r i f i e d } } = \emptyset$ then   
18: continue   
19: end if   
20: A<sub>candidate</sub> $ \mathrm { B u i l d } ( \mathcal { A } _ { \mathrm { c u r r e n t } } , \Delta _ { \mathrm { v e r i f i e d } } )$   
21: $\Delta _ { \mathrm { A c c } }  \mathrm { A c c } ( A _ { \mathrm { c a n d i d a t e } } , B _ { j } ) - \mathrm { A c c } ( \dot { A } _ { \mathrm { c u r r e n t } } , B _ { j } )$   
22: $\begin{array} { r } { \Delta _ { \mathrm { d e g } }  \mathrm { m a x } _ { l \in [ \mathrm { m a x } ( 1 , j - R ) , j - 1 ] } [ \mathrm { A c c } ( \dot { A } _ { \mathrm { c u r r e n t } } , \dot { B _ { l } } ) - \mathrm { A c c } ( \dot { A } _ { \mathrm { c a n d i d a t e } } , B _ { l } ) ] } \end{array}$   
23: if $\Delta _ { \mathrm { A c c } } \geq \delta$ and $\Delta _ { \mathrm { d e g } } \leq \epsilon$ then ▷ accept   
24: $n  n + 1 ;$   
25: ${ \mathcal { A } } ^ { ( n ) } \gets { \mathcal { A } } _ { \mathrm { c a n d i d a t e } } ; \mathbb { V } \gets \mathbb { V } \cup \{ { \mathcal { A } } ^ { ( n ) } \} ; { \mathcal { A } } _ { \mathrm { c u r r e n t } } \gets { \mathcal { A } } ^ { ( n ) } ; c \gets 0$   
26: else ▷ reject   
27: $c \gets c + 1$   
28: end if   
29: if $c \geq P _ { \mathrm { b a t c h } }$ then   
30: break   
31: end if   
32: end for   
33: $\begin{array} { r } { \mathcal { A } _ { \mathrm { c u r r e n t } }  \mathrm { a r g m a x } _ { A \in \mathbb { V } } \mathrm { A c c } ( A , \mathcal { T } _ { \mathrm { v a l } } ) ; \mathcal { A } ^ { * }  \mathcal { A } _ { \mathrm { c u r r e n t } } } \end{array}$   
34: if $\mathrm { A c c } ( \mathcal { A } ^ { * } , \mathcal { T } _ { \mathrm { v a l } } ) > a _ { \mathrm { b e s t } }$ then   
35: $a _ { \mathrm { b e s t } }  \mathrm { A c c } ( \mathcal { A } ^ { \ast } , \mathcal { T } _ { \mathrm { v a l } } ) ; c _ { \mathrm { e p } }  0$   
36: else   
37: $c _ { \mathrm { e p } } \gets c _ { \mathrm { e p } } + 1$   
38: end if   
39: if $c _ { \mathrm { e p } } \geq P _ { \mathrm { e p } }$ then   
40: break   
41: end if   
42: end for   
43: return $\ b { A } ^ { * }$ ▷ Report $\mathrm { A c c } ( \mathcal { A } ^ { \ast } , \mathcal { T } _ { \mathrm { t e s t } } )$