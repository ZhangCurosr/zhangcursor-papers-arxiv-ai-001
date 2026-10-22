# Parallel Thinking for Large Language Model with Path Diversity

Jiaru Qian<sup>1,4</sup>, Yihong Dong<sup>2</sup>, Xue Jiang<sup>1,4</sup>, Chun-Wah Hsu<sup>1,4</sup>, Zhi Jin<sup>1,3</sup>, Ge Li<sup>1</sup>

<sup>1</sup>School of Computer Science, Peking University

<sup>2</sup>Shanghai Jiao Tong University

<sup>3</sup>School of Computer Science, Wuhan University

<sup>4</sup>Key Laboratory of High Confidence Software Technologies, Peking University, Ministry of Education qianjiaru26@stu.pku.edu.cn lige@pku.edu.cn

## Abstract

Parallel thinking has emerged as a promising paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs) by exploring multiple reasoning paths simultaneously. However, we identify that parallel thinking via Reinforcement Learning (RL) typically suffers from the mode collapse of reasoning paths, where parallel thinking trajectories degenerate into similar ones. This homogeneity undermines the fundamental motivation of parallel thinking, that is, maintaining multiple hypotheses concurrently to prevent premature loss into a suboptimal solution. In this paper, we propose Diverse Parallel Thinking (DPT), a simple yet effective RL approach for parallel thinking designed to encourage path diversity. DPT rewards the model to develop distinct reasoning strategies by penalizing the prefix similarity of parallel thinking paths, thereby enhancing a more comprehensive exploration of the reasoning space. We conduct various experiments and analyses across multiple reasoning benchmarks to demonstrate the effectiveness of DPT. Results show that DPT successfully outperforms previous parallel thinking methods, achieving state-of-the-art performance over baselines. We validate path diversity as a key element to improve parallel thinking. Our code is available at https: //github.com/JiaruQian/DPT.

## 1 Introduction

Parallel thinking has emerged as a transformative paradigm for augmenting the reasoning capabilities of Large Language Models (LLMs), particularly in challenging tasks like mathematical reasoning (Wen et al., 2025a; Zheng et al., 2025b; Wu et al., 2025; Jia et al., 2025; Tu et al., 2025). The practical efficacy of this approach has been recently epitomized by Google’s Gemini at the IMO (IMO), where the integration of parallel and sequential thinking proved instrumental to its success (Huang and Yang, 2025). Beyond empirical gains, parallel thinking aligns with principles in cognitive science as human experts often engage in divergent thought, maintaining multiple hypotheses concurrently to prevent premature loss into a suboptimal solution (Clark, 1993; Jackendoff, 2003). By synthesizing these concurrent paths into a coherent conclusion, parallel thinking enables a more structured and deliberate reasoning process.

![](images/64853f48420be01f255faac04262ca6322c49112439ae42ee9a43252417e06b8.jpg)  
Figure 1: Comparison of Traditional Sequential Thinking, Naive Parallel Thinking (RL with outcome reward), and Ideal Parallel Thinking.

Despite this promise, we observe mode collapse of reasoning paths in a reproduced parallelthinking model trained with outcome rewards. Standard RL reward formulations (Yu et al., 2025b; Gao et al., 2024; Eschmann, 2021) optimize for correct final answers without directly rewarding diversity among internal paths. Exploitation of reliable patterns can therefore leave multiple paths highly redundant. As illustrated in Figure 1, paths that repeatedly cover the same approach provide limited additional exploration. Our goal is to discourage this redundancy within a trajectory while preserving task correctness.

The pursuit of path diversity in parallel thinking presents two key challenges. First, the evaluation of reasoning path diversity is difficult. Second, naively maximizing diversity risks sacrificing correctness. A straightforward way is to employ lexical-level similarity like Jaccard (Niwattanakul et al., 2013) between entire paths as a metric in the training reward to encourage diversity across paths. However, such method is strongly influenced by factors such as overall path length and lexical choice, computing the reward at the whole-path level did not yield satisfactory results. Another way is to use a reward model to score the reasoning path diversity. However, it is costly and susceptible to reward hacking (Eisenstein et al., 2023; Khalaf et al., 2025). Therefore, how to develop a simple yet effective metric that captures path diversity of parallel thinking remains an open problem.

In this paper, we propose Diverse Parallel Thinking (DPT), a simple yet effective RL approach for parallel thinking that enforces path diversity as an intrinsic optimization objective, thereby exploring multiple hypotheses concurrently to prevent premature convergence on suboptimal solutions. DPT integrates diversity into the reward signal through a token-level diversity metric that shifts the focus to the early part of each reasoning path. By explicitly encouraging greater diversity in the tokens generated at the beginning, we aim to steer the reasoning trajectories in different directions, allowing them to branch out and explore a broader solution space. DPT can be integrated into existing RL pipelines without requiring architectural modifications, making it applicable to parallel thinking frameworks. We evaluated DPT across a diverse range of reasoning benchmarks and various LLMs. Our results show that DPT consistently outperforms the stateof-the-art (SOTA) baseline, with ablation studies further revealing that path diversity is the primary driver behind these performance gains. These results establish path diversity as a design principle for parallel thinking in LLMs. Our main contribution can be summarized as follows:

• We identify and formally analyze the path mode collapse problem in parallel thinking, showing theoretically that outcome-supervised RL inherently drives parallel paths toward identical trajectories due to exploitation dynamics.

• We propose DPT, a general diversity-aware RL framework for parallel thinking. DPT introduces prefix-based diversity metrics grounded in autoregressive strategy anchoring, requiring zero additional computational overhead.

• Extensive experiments validate that DPT outperforms existing SOTA parallel thinking approaches, establishing path diversity as a key design principle for parallel thinking.

## 2 Motivation

In this section, we analyze the parallel thinking models trained with the existing RL approach (Zheng et al., 2025b). Despite the intuitive appeal of parallel exploration, our findings suggest that existing RL training for parallel thinking leads to the mode collapse of reasoning paths.

Empirical Evidence. We sampled 1,156 parallel reasoning blocks from validation rollouts of the reproduced Parallel-R1 model at training step 200 on AIME24, AIME25, AMC23, and MATH. Each included <Parallel> span contains $K = 2$ parsed <Path> spans; malformed or incomplete spans are excluded. GPT-5-mini (GPT-5-Mini) judges solution approach, reasoning structure, mathematical technique, expression style, and overall strategy diversity. Figure 2 (left) shows that 61.81% of path pairs receive similarity scores $\geq 0 . 9$ This is a judge-based indication of overlap whose reliability depends on the evaluation protocol; Appendix H distinguishes this assessment from lexical diagnostics and states its validation limits.

The result reveals a pervasive mode collapse of parallel thinking paths in the reasoning space, which converge onto identical semantic manifolds. Figure 2 (right) provides an example of this phenomenon in a math reasoning task. Both paths anchor on the same insight, i.e., concluding that vertex alignment constrains the probability to 1/8. Although the phrasing fluctuates, the underlying logical heuristic remains monolithic. The model essentially learns to produce high-reward paraphrases rather than exploring the disjoint solution spaces that parallel thinking is intended to cover.

Theoretical Analysis. Why does standard RL drive parallel reasoning paths toward mode collapse, as observed in Figure 2? The failure arises from the exploitation dynamics inherent in maximizing the expected reward under outcomesupervision. Consider a generator $\pi \theta$ that produces an ordered tuple of K latent reasoning paths $\mathcal { T } = \left( \tau _ { 1 } , \dots , \tau _ { K } \right)$ given an input $x .$ A deterministic aggregation function $\hat { y } ( \mathcal { T } )$ extracts the final answer from the summary of all paths. The standard RL objective maximizes:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } } \left[ \mathbb { E } _ { \mathcal { T } \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ r ( \hat { y } ( \mathcal { T } ) , y ^ { * } ) \right] \right] ,\tag{1}
$$

![](images/1edfea4d02d22486387f6e549c576f3f1ee9f765509deaa2adbb9f297bedadae.jpg)  
Figure 2: Path Mode Collapse in Naive Parallel Thinking.

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { J } \approx \mathbb { E } _ { { x } \sim \mathcal { D } } \mathbb { E } _ { \mathcal { T } } [ \nabla _ { \theta } \log \pi _ { \theta } ( \mathcal { T } | x ) } \\ & { ~ \cdot \underbrace { r ( \hat { y } ( \mathcal { T } ) , y ^ { * } ) } _ { \mathrm { R e w a r d S i g n a l } } ] . } \end{array}\tag{2}
$$

where $y ^ { * }$ is the ground truth and $r ( \cdot , \cdot )$ is a binary correctness metric. Crucially, the reward depends on $\pi _ { \theta }$ only through $\hat { y } ( \mathcal { T } )$ and is agnostic to the diversity of $\tau$ .

The core problem stems from the exploitation dynamics inherent in maximizing Eq. (1). Let $\tau ^ { * }$ denote the “safe mode”, i.e., the single reasoning pattern with the highest marginal probability of yielding the correct answer under the current policy. Because all $K$ paths are sampled from the same shared policy $\pi _ { \theta }$ , the gradient reinforces whichever pattern most reliably produces the correct answer. This drives the marginal distribution $\pi _ { \boldsymbol { \theta } } ( \tau | \boldsymbol { x } )$ to concentrate on $\tau ^ { * }$ , causing all path slots to simultaneously collapse:

$$
\pi _ { \theta } ( \tau _ { k } | x ) \to \mathbb { I } [ \tau _ { k } = \tau ^ { * } ] , \quad \forall k \in \{ 1 , \dots , K \} .\tag{3}
$$

Since all generated paths share the same policy, the model is not incentivized to diversify and instead converges toward $\tau ^ { * }$

Analysis Result. Our analysis reveals a critical design principle: Path diversity must be an explicit optimization constraint rather than an emergent byproduct. To unlock the true potential of parallel thinking to align with its motivation, we must penalize redundancy in the latent reasoning space, specifically at the prefix level, where early strategic decisions are made.

## 3 Related Work

Enhancing LLM Reasoning Capability via RL. RL has become a central technique for enhancing LLM reasoning, with advances in policy optimization (Shao et al., 2024; Zheng et al., 2025a; Yu et al., 2025a), on/off-policy learning (Dong et al., 2025), self-improvement without explicit labels (Zuo et al., 2025), and search-augmented reasoning (Jin et al., 2025). LTE (Tang et al., 2026) further explores trial-and-error learning to overcome exploration stagnation. More recently, RL with rollout diversity has gained attention. DivPO (Lanchantin et al., 2025), unlikeliness reward (He et al., 2025), and EVOL-RL (Zhou et al., 2025) encourage rare, highquality generations through reward shaping or preference optimization, while PODS (Xu et al., 2025), DARLING (Li et al., 2025), and DQO (Chen et al., 2025) improve rollout selection and aggregation. However, these methods still treat each rollout as a sequential generation process, motivating RL for parallel inference.

Parallel Thinking. Parallel thinking has recently emerged as a promising area in LLM reasoning. A set of works (Brown et al., 2024; Huang et al., 2025) explores test-time compute scaling by generating multiple reasoning samples per input and aggregating them to improve answer quality. However, they typically rely on post-hoc sampling as well as simple aggregation, and tend to generate redundant or low-value traces. Recent works (Jia et al., 2025; Tu et al., 2025; Wen et al., 2025a; Wu et al., 2025; Zheng et al., 2025b) aim to endow models with native parallel reasoning capabilities, producing and integrating multiple thought trajectories within a single inference process. Among them, SFT-based methods such as SSFT (Jia et al., 2025) and ParaThinker (Wen et al., 2025a) inevitably lead to superficial parallel thinking, where models merely imitate parallel patterns without genuine parallel reasoning. RL-based methods such as NPR (Wu et al., 2025) and Parallel-R1 (Zheng et al., 2025b) enable models to autonomously explore parallel reasoning, but often suffer from path mode collapse. To this end, we propose DPT to enforce path diversity while maintaining high reasoning accuracy in parallel thinking.

![](images/5d2509332c3d67abb5a5a2858f71d85775ec2754ab66d97b735240b88281a3bb.jpg)  
Figure 3: Overview of DPT. (I) Format learning teaches the parallel thinking format via SFT on easy problems. (II) Diverse parallel thinking develops autonomous parallel thinking on hard problems with combined rewards.

## 4 Diversity Parallel Thinking

We present DPT, an RL approach that enforces path diversity in parallel thinking while maintaining reasoning accuracy. We first formalize parallel thinking as a structured generation process that produces multiple independent reasoning paths within a single trajectory. We then introduce DPT’s training algorithm, which integrates cold-start initialization with SFT and employs a diversity-aware RL reward function to explicitly penalize mode collapse among reasoning paths. An overview of the DPT framework is shown in Figure 3.

## 4.1 Parallel Thinking Formulation

Given a reasoning problem x, parallel thinking generates K distinct reasoning paths $\{ \tau _ { 1 } , \tau _ { 2 } , \dots , \tau _ { K } \}$ within a single coherent generation trajectory. Unlike traditional beam search or sampling that produces independent sequences, parallel thinking enables the model to adaptively decide when to branch into multiple exploration threads during the reasoning process.

We implement this behavior through three special control tags that structure the generation:

• <Parallel>...</Parallel>: Marks the parallel exploration phase, indicating where the model suspends the main reasoning chain to explore multiple solution strategies concurrently.

• <Path>...</Path>: Delimits an internal reasoning thread intended to explore a solution approach. Information isolation depends on the attention mode.

• <Summary>...</Summary>: Synthesizes insights across parallel paths and resolves conflicts before resuming the main reasoning chain.

Formally, let z denote the complete generation trajectory for problem x. The model autoregressively generates tokens according to policy π<sub>θ</sub>(·|x). When the model emits a <Parallel> token, it spawns K parallel paths $\{ \tau _ { 1 } , \dots , \tau _ { K } \}$ within separate <Path> blocks. Each path τ<sub>i</sub> is intended to explore a distinct reasoning strategy. In the “Unseen” mode, structural attention masking enforces true information isolation between paths; in the “Seen” mode, paths may attend to prior paths. A <Summary> block then integrates the parallel explorations before resuming the main trajectory.

The objective of parallel thinking is twofold: exploring multiple strategies increases the probability of finding correct answers, and diverse paths enable multi-perspective verification. However, standard RL optimizes only for correctness without penalizing redundancy, causing paths to collapse into identical strategies. DPT addresses this by explicitly rewarding diversity among parallel paths.

## 4.2 Training Algorithm

We train DPT through a two-stage curriculum that addresses the cold-start problem in parallel thinking. The model first learns the basic format and structure of parallel reasoning on simpler problems with supervised finetuning, then transitions to reinforcement learning on challenging tasks to enhance performance with path diversity.

## 4.2.1 Cold Start Initialization

Since pretrained language models have not encountered the special control tags (<Parallel>, <Path>, <Summary>) during pretraining, they cannot generate valid parallel thinking trajectories for RL exploration. We address this cold-start problem through supervised fine-tuning on a curated dataset of parallel thinking examples constructed from the GSM8K training set. We prompt a capable language model to generate solutions in the parallel thinking format and filter outputs to retain only those with valid tag structure. This phase instills familiarity with the parallel thinking format—when to trigger <Parallel> blocks, how to structure independent <Path> segments, and how to synthesize conclusions in <Summary> blocks—enabling subsequent RL exploration on harder tasks.

## 4.2.2 Reinforcement Learning

After cold-start initialization, we train the model using Group Relative Policy Optimization (GRPO) (Shao et al., 2024), a well-known policy gradient method designed for grouped samples.

Training Objective. For each problem x, GRPO samples N trajectories $\{ z _ { 1 } , \dots , z _ { N } \}$ from the policy $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { x } )$ and computes rewards for each trajectory. The policy gradient estimates the advantage function using group-relative normalization:

$$
\hat { A } ( z _ { i } ) = \frac { R ( z _ { i } ) - \bar { R } } { \sigma _ { R } + \epsilon } ,\tag{4}
$$

where $\begin{array} { r } { \bar { R } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } R ( z _ { j } ) } \end{array}$ and $\sigma _ { R }$ are the mean and standard deviation of rewards within the group, and ϵ is a small constant for numerical stability. This normalization reduces variance compared to absolute rewards. The policy is updated via:

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { D } , z _ { i } \sim \pi _ { \theta } ( \cdot | x ) } \left[ \hat { A } ( z _ { i } ) \log \pi _ { \theta } ( z _ { i } | x ) \right] , } \end{array}\tag{5}
$$

subject to a KL divergence constraint KL $_ { \prime } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) = \delta$ to prevent the policy from deviating too far from the reference model $\pi _ { \mathrm { r e f } }$ We use the pretrained base model as the reference.

Reward Modeling. The reward function $R ( z _ { i } )$ determines what behaviors the model learns. Here $z _ { i }$ denotes the complete generation trajectory, which may contain multiple parallel paths $\{ \tau _ { 1 } , \dots , \tau _ { K } \}$ Standard RL for reasoning uses only a correctness reward $r _ { c } ( z _ { i } ) = \mathbb { I } [ \hat { y } ( z _ { i } ) = y ^ { * } ]$ , where $\hat { y } ( z _ { i } )$ is the final answer extracted from trajectory $z _ { i }$ and $y ^ { * }$ is the ground truth. Existing parallel thinking approaches also introduce format reward $r _ { f } ( z _ { i } )$

$$
\begin{array} { r } { r _ { f } ( z _ { i } ) = \mathbb { I } [ \mathrm { A t } \mathrm { l e a s t o n e } \mathrm { c o r r e c t } } \\ { < \mathrm { P a r a l } 1 \in \mathrm { 1 } > \mathrm { b l o c k } \mathrm { i n } z _ { i } ] . } \end{array}
$$

However, optimizing correctness alone causes mode collapse. The model learns to exploit a single high-reward strategy, generating semantically identical paths despite the parallel thinking structure.

DPT addresses this by incorporating a diversity reward that explicitly penalizes mode collapse. For each trajectory $z _ { i } ,$ the total reward combines correctness and diversity of all parallel paths $\{ \tau _ { 1 } , \tau _ { 2 } , \dots , \tau _ { K } \}$

$$
R ( z _ { i } ) = r _ { c } ( z _ { i } ) + \lambda _ { f } { \cdot } r _ { f } ( z _ { i } ) + \lambda _ { d } { \cdot } r _ { d } ( \{ \tau _ { 1 } , . . . , \tau _ { K } \} ) ,\tag{6}
$$

where $\lambda _ { d }$ controls the trade-off between correctness and diversity, and $r _ { d }$ measures the diversity across all K parallel paths generated for problem x.

Diversity Metric Design. Measuring trajectory diversity presents a key challenge. When parallel paths share the same preceding context, the model tends to fall into similar reasoning patterns despite independent generation processes. A natural approach is to compute similarity metrics over entire paths, such as Jaccard similarity between complete token sequences. However, whole-path metrics suffer from two fundamental issues. First, path length creates spurious diversity signals. Two paths employing identical reasoning strategies but differing in verbosity yield low similarity scores, falsely indicating strategic divergence. Second, lexical choice confounds semantic similarity. Paths using the same underlying approach but varying word choices appear diverse under token-level metrics, rewarding superficial paraphrasing rather than strategic exploration.

We therefore shift focus to the early part of each reasoning path. In autoregressive generation, each prefix conditions subsequent tokens through $P ( x _ { t } | x _ { < t } )$ . Encouraging different prefixes supplies a local symmetry-breaking signal that can create opportunities for different continuations. However, paths with different openings can still use the same strategy, and paths with shared openings can diverge later. Prefix divergence is therefore an inexpensive training proxy rather than a sufficient condition for semantic strategy diversity. We compare the first n content tokens of each path pair:

$$
d ( \tau _ { i } , \tau _ { j } ) = \mathbb { I } [ \mathrm { p r e f i x } ( \tau _ { i } , n ) \neq \mathrm { p r e f i x } ( \tau _ { j } , n ) ] ,\tag{7}
$$

where prefix(τ, n) extracts the first n content tokens from path τ (starting after the <Path> tag). This binary metric returns 1 if the prefixes differ and 0 if they are identical. We also explore a similarity variant that measures token-level overlap: $\begin{array} { r } { d _ { \mathrm { s i m } } ( \tau _ { i } , \tau _ { j } ) = 1 - \frac { | \{ t : \mathrm { p r e f i x } ( \tau _ { i } , n ) _ { t } = \mathrm { p r e f i x } ( \tau _ { j } , n ) _ { t } \} | } { n } } \end{array}$ which provides a finer-grained signal. We ablate both variants in Section D.

The diversity reward aggregates pairwise comparisons across all K paths $\left( K \geq 2 \right)$

$$
r _ { d } ( \{ \tau _ { 1 } , \dots , \tau _ { K } \} ) = \frac { 1 } { { \binom { K } { 2 } } } \sum _ { i = 1 } ^ { K - 1 } \sum _ { j = i + 1 } ^ { K } d ( \tau _ { i } , \tau _ { j } ) .\tag{8}
$$

This normalized score ranges from 0 (all paths identical) to 1 (all paths distinct). The metric penalizes mode collapse by rewarding models that generate diverse prefixes. Token-level comparison offers several advantages. Computational efficiency improves because we do not need to encode full trajectories into dense representations. Interpretability improves because differences are human-verifiable at the token level. The early divergence focus captures strategic differences effectively, as high-level approach selection manifests in the initial tokens.

Training Curriculum. If diversity pressure is applied too aggressively, the model may prioritize superficially different paths over correct answers. During RL training, the full reward $R ( z _ { i } ) =$ $r _ { c } ( z _ { i } ) + \lambda _ { f } \cdot r _ { f } ( z _ { i } ) + \lambda _ { d } \cdot r _ { d } ( \{ \tau _ { 1 } , \dots , \tau _ { K } \} )$ is applied only for the last 2 steps of every 10 training steps, while the remaining 8 steps use the correctness reward $r _ { c } ( z _ { i } )$ alone. This periodic schedule limits the influence of artificial divergence but does not eliminate proxy failures.

## 5 Experiments

In this section, we evaluate DPT on multiple mathematical reasoning benchmarks.We conduct experiments on coding tasks and different LLM backbones to demonstrate the effectiveness of DPT and the necessity of path diversity in parallel thinking.

Datasets. For reinforcement learning, we train on the DAPO dataset (Yu et al., 2025a) using GRPO. We evaluate mainly on four challenging mathematical reasoning benchmarks: AIME24, AIME25, AMC23 (AMC23) and MATH (Hendrycks et al., 2021). We report pass@1 and pass@16 accuracy, where pass@k measures the fraction of problems solved correctly in at least one of k sampled solutions. We also conduct experiments on HumanEval (Liu et al., 2023), ARC-c (Clark et al., 2018), MMLU-Pro (Wang et al., 2024) to evluate the generalization of DPT on coding and knowledge reasoning tasks.

Baselines. We compare against nine baselines: Direct, Cold-start SFT, GRPO, LTE (Tang et al., 2026), TTRL (Zuo et al., 2025), EVOL-RL (Zhou et al., 2025), Parallel-R1 (Seen/Unseen) and Parallel-R1 (w/ Jaccard). Unseen mode uses structured attention and multiverse position encodings to prevent cross-path information leakage. LTE, TTRL and EVOL-RL results are reported from their original papers.

Implementation Details. The main experiments are conducted using 8 A100 GPUs with Qwen-3-4B-Base (Yang et al., 2025). Our codebase is adapted from VERL (Sheng et al., 2025). For coldstart initialization, we construct parallel-thinking examples from GSM8K by prompting DeepSeek-R1-0528-Qwen-3-8B and filtering outputs for valid tag structure. For RL, we train on DAPO (Yu et al., 2025a) using GRPO. The main configuration uses $K = 2 , \lambda _ { f } = 0 . 2 , \lambda _ { d } = 0 . 1$ , and n = 1. Details can be found in Appendix A.

## 5.1 Experimental Results

Overall Performance. Table 1 shows that DPT (Unseen) achieves the best overall performance among the evaluated methods, outperforming Parallel-R1 (Unseen) across every reported benchmark and metric. The gains span both singlesample accuracy and multi-sample coverage, with a particularly strong improvement on challenging AIME problems. For instance, DPT improves AIME24 pass@16 by 6.2 percentage points over Parallel-R1 (Unseen), highlighting its ability to uncover correct solutions through broader exploration. DPT (Unseen) also outperforms DPT (Seen) on average, suggesting that maintaining independent paths complements diversity-driven exploration. We use DPT to denote the Unseen configuration in the following experiments.

<table><tr><td rowspan="2">Method</td><td colspan="2">AMC23</td><td colspan="2">AIME25</td><td colspan="2">AIME24</td><td rowspan="2">MATH pass@1</td><td rowspan="2">Average</td></tr><tr><td>pass@1</td><td>pass@16</td><td>pass@1</td><td>pass@16</td><td>pass@1</td><td>pass@16</td></tr><tr><td>Base Model</td><td>8.1</td><td>51.2</td><td>1.3</td><td>10.2</td><td>2.9</td><td>16.5</td><td>13.9</td><td>14.9</td></tr><tr><td>SFT</td><td>39.1</td><td>77.3</td><td>4.4</td><td>17.2</td><td>6.5</td><td>24.6</td><td>65.8</td><td>33.6</td></tr><tr><td>GRPO</td><td>61.4</td><td>88.8</td><td>16.0</td><td>33.3</td><td>17.9</td><td>39.9</td><td>82.9</td><td>48.6</td></tr><tr><td>LTE (Tang et al., 2026)</td><td>56.7</td><td>85.0</td><td>14.2</td><td>46.7</td><td>18.3</td><td>30.0</td><td>74.6</td><td>46.5</td></tr><tr><td>TTRL (Zuo et al., 2025)</td><td>42.5</td><td>75.2</td><td>6.8</td><td>28.6</td><td>12.1</td><td>23.2</td><td>75.4</td><td>37.7</td></tr><tr><td>EVOL-RL (Zhou et al., 2025)</td><td>51.4</td><td>80.3</td><td>17.5</td><td>39.9</td><td>20.5</td><td>47.6</td><td>80.0</td><td>48.2</td></tr><tr><td>Parallel-R1 (Seen)</td><td>62.0</td><td>88.8</td><td>18.1</td><td>36.3</td><td>17.1</td><td>35.1</td><td>82.0</td><td>48.5</td></tr><tr><td>Parallel-R1 (Unseen)</td><td>63.6</td><td>88.2</td><td>18.3</td><td>38.5</td><td>16.5</td><td>37.1</td><td>83.2</td><td>49.3</td></tr><tr><td>Parallel-R1 (w/ Jaccard)</td><td>62.0</td><td>85.2</td><td>18.1</td><td>38.1</td><td>17.1</td><td>31.8</td><td>82.6</td><td>47.8</td></tr><tr><td>DPT (Seen)</td><td>62.0</td><td>88.7</td><td>17.7</td><td>38.4</td><td>18.3</td><td>36.6</td><td>83.5</td><td>49.3</td></tr><tr><td>DPT (Unseen)</td><td>64.4</td><td>89.9</td><td>18.5</td><td>40.2</td><td>18.1</td><td>43.3</td><td>84.5</td><td>51.3</td></tr></table>

Table 1: Performance comparison on mathematical reasoning benchmarks, where the best results shown in bold and second-best results underlined.

<table><tr><td>Method</td><td>HumanEval</td><td>ARC-c</td><td>MMLU-Pro</td></tr><tr><td>GRPO</td><td>60.98</td><td>64.68</td><td>43.40</td></tr><tr><td>Parallel-R1</td><td>67.12</td><td>64.88</td><td>47.76</td></tr><tr><td>DPT</td><td>69.51</td><td>65.27</td><td>49.40</td></tr></table>

Table 2: Reported scores (%) on three additional on coding and knowledge reasoning tasks.

<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH</td></tr><tr><td>Parallel-R1</td><td>0.581</td><td>0.542</td><td>0.529</td><td>0.567</td></tr><tr><td>DPT</td><td>0.923</td><td>0.927</td><td>0.911</td><td>0.886</td></tr><tr><td>∆</td><td>↑0.342</td><td>↑0.385</td><td>↑0.382</td><td>↑0.319</td></tr></table>

Table 3: Lexical path diversity measured by Jaccard distance. Higher values indicate lower lexical overlap; ∆ denotes the increase over Parallel-R1.

Reasoning Beyond Mathematics. We extend the evaluation to code generation on HumanEval, science and commonsense reasoning on ARC-c, and knowledge-intensive reasoning on MMLU-Pro. As shown in Table 2, DPT outperforms both GRPO and Parallel-R1 on all three benchmarks. The advantages are most pronounced on code generation and knowledge-intensive reasoning, while DPT also maintains the lead on ARC-c. These consistent gains broaden the applicability of our approach. Encouraging diverse internal paths benefits reasoning across tasks with different solution structures and correctness signals. In particular, the improvement over Parallel-R1 highlights the additional value of DPT within the parallel-thinking framework beyond mathematics.

Path Diversity. Beyond accuracy, DPT produces substantially less overlapping reasoning paths across all four mathematical benchmarks (Table 3). Although our reward acts only on path prefixes, the increase in whole-path Jaccard distance shows that early divergence is accompanied by broader differences in the generated continuations. This local intervention thus encourages broader textual exploration while leaving the remainder of each path free to develop. The complementary LLM-as-judge assessment in Figure 4 also favors DPT across solution approach, reasoning structure, mathematical techniques, explanation style, and strategy diversity. Together, these observations support our goal of making parallel branches more distinct and useful for exploration. Appendix H provides further details on the lexical and judge-based evaluations.

![](images/56298b965aa9358a50855974f33f5bf8dc32710db5cf35f70bdbd55ad55e3e75.jpg)  
Figure 4: Multi-dimensional diversity evaluation. LLM-as-judge comparing DPT and Parallel-R1 across five dimensions. Lower score denotes more diversity.

## 5.2 Further Analysis

Scaling the Number of Internal Paths. We further explore scaling within a single trajectory by increasing the number of internal paths from K = 2 to K = 4 under the same 4B backbone and training setup. Our normalized pairwise diversity reward extends directly to more paths without changing its definition. Table 4 reveals a clear division of strengths: $K \ : = \ : 2$ delivers the best pass@1 accuracy on all three benchmarks, while $K \ = \ 4$ achieves higher pass@16 on AIME25 and AMC23 and remains competitive on AIME24. This pattern shows that additional paths strengthen exploration and solution coverage, even when a smaller path set is preferable for the most likely individual answer.

<table><tr><td></td><td colspan="2">AIME25</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td></tr><tr><td>Setting</td><td>pass@1</td><td>pass@16</td><td>pass@1</td><td>pass@16</td><td>pass@1</td><td>pass@16</td></tr><tr><td>DPT (K = 2)</td><td>18.5</td><td>40.2</td><td>18.1</td><td>40.2</td><td>64.4</td><td>89.9</td></tr><tr><td>DPT (K = 4)</td><td>17.5</td><td>41.2</td><td>17.5</td><td>38.2</td><td>61.9</td><td>90.4</td></tr></table>

Table 4: Results (%) with two or four internal reasoning paths. K counts paths within a trajectory. We report pass@1 and pass@16 accuracy on three mathematical reasoning benchmarks.

<table><tr><td rowspan="2">Method</td><td colspan="2">pass@k (%)</td></tr><tr><td>1 16</td><td>64 128 256</td></tr><tr><td>GRPO</td><td>16.033.3 48.3</td></tr><tr><td>Parallel-R1 18.3 38.5</td><td>52.4 56.0 47.9 49.3 50.9</td></tr><tr><td>DPT</td><td>18.5 40.2 49.2 53.5 57.0</td></tr></table>

Table 5: AIME25 pass@k accuracy (%) with increasing numbers of sampled complete solutions.

Test-time Scaling. DPT maintains the strongest performance at every sampling budget in Table 5, combining higher single-sample accuracy with sustained gains as more complete solutions are sampled. Parallel-R1’s initial advantage over GRPO diminishes at larger budgets, whereas DPT continues to outperform both methods. At pass@256, DPT exceeds Parallel-R1 by 6.1 percentage points. This pattern is consistent with diversity helping parallel thinking retain broader solution coverage as sampling increases. We further discuss this interpretation in Appendix E and report inference-cost measurements in Appendix F. Together with the paths scaling results, it illustrates how DPT supports broader exploration both within a trajectory and across sampled solutions.

Adaptive Parallel Thinking Usage. Figure 5 shows that the model shifts from near-universal parallel thinking after cold-start initialization to more selective use during RL training, while accuracy continues to improve. The trained model increasingly combines selective branching with effective solution generation. This behavior complements our diversity objective: DPT encourages distinct paths when the model branches, while the training process allows it to adapt how frequently it invokes parallel thinking. We provide the ablation of prefix length and reward design in Appendix D.

![](images/bb385c486a3cbe0d0e0095c89d6c30b84535148f34783b0cb2c1a0eb14ae37ea.jpg)  
Figure 5: Training dynamics of accuracy and parallel ratio, where the parallel ratio (purple) decreases from near 100% to ∼65% as the model learns selective parallel thinking, while accuracy (blue) continues improving.

## 6 Conclusion

In this paper, we first discover the problem of path mode collapse in parallel thinking, where multiple parallel reasoning paths degenerate into similar ones, contradicting the core motivation of parallel thinking. Therefore, we propose a simple but effective RL approach, DPT, to address path mode collapse in parallel thinking for LLMs. Specifically, DPT enforces path diversity through a tokenlevel diversity metric that detects early reasoning divergence, combined with a two-stage training curriculum that balances correctness and exploration. Extensive experiments demonstrate that DPT consistently outperforms the best-performing baseline on mathematical reasoning, coding and knowledge reasoning benchmarks. The results establish that path diversity is a key component for effective parallel thinking. By addressing path mode collapse, we take a step toward realizing the full potential of parallel thinking, unlocking the test-time scaling.

## Limitations

Our study focuses on text-based reasoning tasks with verifiable answers, including mathematics, coding, and knowledge reasoning. Extending DPT to multimodal inputs and interactive reasoning would further broaden its application scope. We evaluate a small set of internal path counts and keep the path count fixed within each configuration. An adaptive allocation policy that selects the number of paths according to problem difficulty and inference budget could provide greater flexibility. We leave these directions for future work.

## Ethical Considerations

Within the evaluated research setting, we do not identify substantial additional ethical, societal, or environmental risks beyond those generally associated with LLM training and inference.

## References

AIME. 2025. https://maa.org/ maa-invitational-competitions/.

AMC23. 2025. https://huggingface.co/ datasets/math-ai/amc23.

Evan Becker and Stefano Soatto. 2024. Cycles of thought: Measuring llm confidence through stable explanations. Preprint, arXiv:2406.03441.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference compute with repeated sampling. Preprint, arXiv:2407.21787.

Yilei Chen, Souradip Chakraborty, Lorenz Wolf, Yannis Paschalidis, and Aldo Pacchiano. 2025. Post-training large language models for diverse high-quality responses. Preprint, arXiv:2509.04784.

Andy Clark. 1993. Associative engines: Connectionism, concepts, and representational change. MIT Press.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, and Ge Li. 2025. Rl-plus: Countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization. Preprint, arXiv:2508.00222.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. 2023. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv preprint arXiv:2312.09244.

Jonas Eschmann. 2021. Reward function design in reinforcement learning. Reinforcement learning algorithms: Analysis and Applications, pages 25–33.

Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. 2024. On designing effective rl reward at training time for llm reasoning. Preprint, arXiv:2410.15115.

GPT-5-Mini. 2025. https://platform.openai. com/docs/models/gpt-5-mini.

Blessed Guda, Lawrence Francis, Gabrial Zencha Ashungafac, Carlee Joe-Wong, and Moise Busogi. Tiny: Rethinking selection bias in llms: Quantification and mitigation using efficient majority voting. In ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI.

Andre He, Daniel Fried, and Sean Welleck. 2025. Rewarding the unlikely: Lifting grpo beyond distribution sharpening. Preprint, arXiv:2506.02355.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. 2025. Efficient test-time scaling via self-calibration. Preprint, arXiv:2503.00031.

Yichen Huang and Lin F. Yang. 2025. Winning gold at imo 2025 with a model-agnostic verification-andrefinement pipeline. Preprint, arXiv:2507.15855.

IMO. https://www.imo-official.org/.

Ray Jackendoff. 2003. Précis of foundations of language: Brain, meaning, grammar, evolution. Behavioral and brain sciences, 26(6):651–665.

Sheng Jia, Xiao Wang, and Shiva Prasad Kasiviswanathan. 2025. Training large language models to reason in parallel with global forking tokens. Preprint, arXiv:2510.05132.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. Preprint, arXiv:2503.09516.

Ariel Kamen and Yakov Kamen. 2025. Majority rules: Llm ensemble is a winning approach for content categorization. arXiv preprint arXiv:2511.15714.

Hadi Khalaf, Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, and Flavio du Pin Calmon. 2025. Inference-time reward hacking in large language models. arXiv preprint arXiv:2506.19248.

Jack Lanchantin, Angelica Chen, Shehzaad Dhuliawala, Ping Yu, Jason Weston, Sainbayar Sukhbaatar, and Ilia Kulikov. 2025. Diverse preference optimization. Preprint, arXiv:2501.18101.

Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, Jason Weston, Jack Lanchantin, and Tianlu Wang. 2025. Jointly reinforcing diversity and quality in language model generations. Preprint, arXiv:2509.02534.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Ling ming Zhang. 2023. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in neural information processing systems, 36:21558–21572.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20286–20332.

Suphakit Niwattanakul, Jatsada Singthongchai, Ekkachai Naenudorn, and Supachanun Wanapu. 2013. Using of jaccard coefficient for keywords similarity. In Proceedings of the international multiconference of engineers and computer scientists, volume 1, pages 380–384.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297. ACM.

Chenming Tang, Hsiu-Yuan Huang, Weijie Liu, Clive Bai, Saiyong Yang, and Yunfang Wu. 2026. Do not step into the same river twice: Learning to reason from trial and error. Preprint, arXiv:2510.26109.

Shangqing Tu, Yaxuan Li, Yushi Bai, Lei Hou, and Juanzi Li. 2025. Deepprune: Parallel scaling without inter-trace redundancy. Preprint, arXiv:2510.08483.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. 2025. Reinforcement learning for reasoning in large language models with one training example. Preprint, arXiv:2504.20571.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266– 95290.

Hao Wen, Yifan Su, Feifei Zhang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, and Yuanchun Li. 2025a. Parathinker: Native parallel thinking as a new paradigm to scale llm test-time compute. Preprint, arXiv:2509.04475.

Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, Jiang Bian, and Mao Yang. 2025b. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. Preprint, arXiv:2506.14245.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. 2023. Large language models are better reasoners with self-verification. Preprint, arXiv:2212.09561.

Tong Wu, Yang Liu, Jun Bai, Zixia Jia, Shuyi Zhang, Ziyong Lin, Yanting Wang, Song-Chun Zhu, and Zilong Zheng. 2025. Native parallel reasoner: Reasoning in parallelism via self-distilled reinforcement learning. Preprint, arXiv:2512.07461.

Yixuan Even Xu, Yash Savani, Fei Fang, and J. Zico Kolter. 2025. Not all rollouts are useful: Downsampling rollouts in llm reinforcement learning. Preprint, arXiv:2504.13818.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025a. Dapo: An open-source llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Rui Yu, Shenghua Wan, Yucen Wang, Chen-Xiao Gao, Le Gan, Zongzhang Zhang, and De-Chuan Zhan. 2025b. Reward models in deep reinforcement learning: A survey. Preprint, arXiv:2506.15421.

Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, and Bo An. 2025a. Incentivizing llms to self-verify their answers. Preprint, arXiv:2506.01369.

Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo,

Yufei Wang, Niklas Muennighoff, et al. 2025b. A survey on test-time scaling in large language models: What, how, where, and how well? arXiv preprint arXiv:2503.24235.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. 2025a. Group sequence policy optimization. Preprint, arXiv:2507.18071.

Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, and Dong Yu. 2025b. Parallel-r1: Towards parallel thinking via reinforcement learning. Preprint, arXiv:2509.07980.

Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, and Dong Yu. 2025. Evolving language models without labels: Majority drives selection, novelty promotes variation. Preprint, arXiv:2509.15194.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. 2025. Ttrl: Test-time reinforcement learning. Preprint, arXiv:2504.16084.

## A Implementation Details

The main experiments are conducted using 8 A100 GPUs with Qwen-3-4B-Base (Yang et al., 2025), following concurrent parallel-thinking research (Zheng et al., 2025b; Wu et al., 2025). The objective-level discussion in Section 2 motivates the reward design without establishing architectureinvariant behavior or inevitable collapse. Our codebase is adapted from VERL (Sheng et al., 2025); we follow its training recipe and separately ablate prefix length and comparison method. For coldstart initialization, we construct parallel-thinking examples from GSM8K by prompting a capable model and filtering outputs for valid tag structure. We perform supervised fine-tuning with batch size 128, learning rate 1e-5, weight decay of 0.01, and warm-up ratio 0.1 with a cosine learning-rate schedule. For RL, we train on DAPO (Yu et al., 2025a) using GRPO with batch size 256, learning rate 1e-6, and 300 gradient update steps. The main configuration uses K = 2, λ<sub>f</sub> = 0.2, λ<sub>d</sub> = 0.1, and Prefix-1 exact matching (n = 1). Diversity and format rewards are applied only for the last two steps of each ten-step cycle; the remaining eight use correctness alone. This periodic RL schedule follows cold-start SFT.

![](images/ee1bd4fe8a400a633e4089e1537e0dd0f657afdc2abb557fb3c814c2f4f064bf.jpg)  
Figure 6: Prompt Construction.

## B Evaluation Benchmarks

We evaluate on four challenging mathematical reasoning benchmarks. AIME (AIME) (American Invitational Mathematics Examination) tests advanced problem-solving with AIME24 and AIME25 versions. AMC23 (AMC23) (American Mathematics Competition) evaluates mathematical competition skills. MATH (Hendrycks et al., 2021) contains high-difficulty problems across multiple mathematical domains. Following prior work, we report pass@1 and pass@16 accuracy, where pass@k measures the fraction of problems solved correctly in at least one of k sampled solutions. For AIME and AMC benchmarks, we sample 16 responses per question at temperature 1.0 and report mean accuracy. For MATH, we evaluate on a 300-problem subset and report pass@1 with single sampling. For DPT and other parallel thinking baselines, we construct specific instruction prompt for all problems in the training and testing datasets to guide the model to conduct parallel thinking, as shown in Figure 6.

## C Baselines

In this section, we introduce three RL baselines adopted in our experiment: LTE (Tang et al., 2026), TTRL (Zuo et al., 2025) and EVOL-RL (Zhou et al., 2025).

LTE: LTE (Learning to Reason from Trial and Error) addresses the exploration stagnation issue in reinforcement learning with verifiable rewards (RLVR) (Wang et al., 2025; Wen et al., 2025b), where models are often limited by their initial on-policy generation capability. Unlike standard RLVR methods that rely solely on on-policy responses or off-policy approaches requiring external expert solutions, LTE leverages models’ own previously generated mistakes as learning signals, without any external supervision. By incorporat ing trial-and-error feedback into training, LTE improves both exploration and exploitation. LTE con sistently outperforms GRPO on multiple mathematical and even surpasses methods using external gold solutions under aligned settings, demonstrating its effectiveness in enhancing reasoning performance. TTRL: TTRL (Test-Time Reinforcement Learning) studies reinforcement learning for LLM reasoning tasks in the absence of explicit supervision, where ground-truth labels are unavailable and re ward estimation becomes the central challenge. In stead of relying on oracle rewards, TTRL exploits implicit rewards derived from common test-time scaling strategies (Muennighoff et al., 2025; Zhang et al., 2025b), such as majority voting (Guda et al.; Kamen and Kamen, 2025), to guide RL training on unlabeled data. By leveraging the inductive priors of pre-trained models, TTRL enables selfimprovement during inference without additional annotations. Extensive experiments show that TTRL consistently improves performance across tasks and model scales, and achieves substantial gains on challenging benchmarks such as AIME 2024. Notably, despite being supervised only by majority-based rewards, TTRL surpasses the ini tial model’s maj@n upper bound and approaches the performance of models trained with groundtruth supervision, demonstrating the effectiveness of test-time RL with implicit rewards.

EVOL-RL: EVOL-RL targets label-free selfimprovement for LLMs trained with reinforcement learning from verifiable rewards, where no groundtruth labels or external judges are available at deployment time. Unlike prior self-improvement methods that rely on self-confirmation signals such as confidence or majority consistency (Weng et al., 2023; Zhang et al., 2025a; Becker and Soatto, 2024), EVOL-RL adopts an evolution-inspired training paradigm. It stabilizes learning by anchoring rewards to the majority-voted solution, while simultaneously encouraging exploration through a novelty-aware reward that promotes reasoning diversity among sampled responses. This balance between majority-based selection and noveltydriven variation prevents over-confident collapse and sustains complex reasoning behaviors. Experimental results demonstrate that EVOL-RL consistently outperforms majority-only baselines on labelfree reasoning benchmarks and improves both indomain performance and out-of-domain generalization across diverse tasks.

## D Ablation on Diversity Metric Design

Exact and Similarity Settings. Both settings compare the first n content tokens after the $\mathtt { < P a t h > }$ tag. Let $u _ { i } = \mathrm { p r e f i x } ( \tau _ { i } , n )$ and $u _ { j } =$ prefix $( \tau _ { j } , n )$ denote the two prefixes. The exact setting uses the binary score:

$$
d _ { \mathrm { e x a c t } } ( \tau _ { i } , \tau _ { j } ) = \mathbb { I } [ u _ { i } \neq u _ { j } ] .\tag{9}
$$

It assigns a diversity score of 1 whenever at least one corresponding token differs, and 0 only when the entire prefixes are identical. The similarity setting instead assigns the proportion of corresponding token positions that differ:

$$
d _ { \mathrm { s i m } } ( \tau _ { i } , \tau _ { j } ) = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { I } [ ( u _ { i } ) _ { t } \neq ( u _ { j } ) _ { t } ] .\tag{10}
$$

For example, if two three-token prefixes differ at exactly one position, exact gives a score of 1, whereas similarity gives $1 / 3 .$ . Thus, exact rewards any prefix divergence, while similarity provides graded credit according to its extent. The two variants coincide when $n = 1$ . For either variant, the trajectory-level diversity reward averages these scores over all pairs of internal paths, as in the main text.

Table 6 compares prefix lengths and the exact and similarity settings across all four mathematical reasoning benchmarks. Every tested prefixdiversity setting improves average accuracy over Parallel-R1 without a diversity reward, supporting the effectiveness of early path divergence across different reward designs. Prefix-1 exact achieves the highest average accuracy and improves every reported accuracy metric over the baseline. We therefore select $n = 1$ with exact matching for our main configuration based on its strongest aggregate performance across the full evaluation suite.

Longer prefixes offer advantages on individual benchmarks: Prefix-4 similarity achieves the strongest AIME25 pass@16 result, Prefix-4 exact leads on AMC23, and Prefix-2 exact performs best on MATH. Prefix-1 exact combines the strongest

<table><tr><td rowspan="2">Metric</td><td colspan="2">AIME25</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td><td rowspan="2">MATH</td><td rowspan="2">Average</td></tr><tr><td>pass@1</td><td>pass@16</td><td>pass@1</td><td>pass@16</td><td>pass@1</td><td>pass@16 pass@1</td></tr><tr><td>No diversity reward</td><td>18.33</td><td>38.51</td><td>16.45</td><td>37.11</td><td>63.59</td><td>88.20</td><td>83.22</td><td>49.34</td></tr><tr><td>Prefix-1 exact</td><td>18.54</td><td>40.18</td><td>18.12</td><td>43.34</td><td>64.37</td><td>89.88</td><td>84.49</td><td>51.27</td></tr><tr><td>Prefix-2 exact</td><td>17.91</td><td>42.56</td><td>16.66</td><td>35.88</td><td>62.81</td><td>88.50</td><td>85.12</td><td>49.92</td></tr><tr><td>Prefix-3 exact</td><td>17.08</td><td>40.92</td><td>17.70</td><td>33.72</td><td>63.75</td><td>89.10</td><td>84.49</td><td>49.54</td></tr><tr><td>Prefix-4 exact</td><td>20.83</td><td>42.88</td><td>17.91</td><td>36.77</td><td>64.53</td><td>91.26</td><td>82.59</td><td>50.97</td></tr><tr><td>Prefix-2 similarity</td><td>19.37</td><td>42.50</td><td>17.70</td><td>36.86</td><td>63.12</td><td>89.42</td><td>83.22</td><td>50.31</td></tr><tr><td>Prefix-3 similarity</td><td>17.70</td><td>39.82</td><td>17.50</td><td>39.51</td><td>63.12</td><td>90.25</td><td>83.22</td><td>50.16</td></tr><tr><td>Prefix-4 similarity</td><td>20.83</td><td>46.27</td><td>17.29</td><td>40.16</td><td>63.58</td><td>87.39</td><td>82.75</td><td>51.18</td></tr></table>

Table 6: Ablation study on diversity metric design across four mathematical reasoning benchmarks. Scores are percentages; Average is the arithmetic mean of the seven reported accuracy metrics. The best results in each column are bold.

<table><tr><td>Metric</td><td>AIME25</td><td>AIME24</td><td>AMC23</td><td>MATH</td></tr><tr><td>No diversity reward</td><td>54.58</td><td>52.08</td><td>77.81</td><td>94.62</td></tr><tr><td>Prefix-1 exact</td><td>60.42</td><td>61.67</td><td>82.50</td><td>95.89</td></tr><tr><td>Prefix-2 exact</td><td>75.00</td><td>77.71</td><td>81.80</td><td>95.89</td></tr><tr><td>Prefix-3 exact</td><td>48.75</td><td>49.17</td><td>73.75</td><td>92.72</td></tr><tr><td>Prefix-4 exact</td><td>51.88</td><td>51.04</td><td>77.34</td><td>89.87</td></tr><tr><td>Prefix-2 similarity</td><td>42.08</td><td>44.79</td><td>71.88</td><td>92.72</td></tr><tr><td>Prefix-3 similarity</td><td>38.12</td><td>41.67</td><td>70.94</td><td>87.66</td></tr><tr><td>Prefix-4 similarity</td><td>52.92</td><td>55.21</td><td>77.34</td><td>93.35</td></tr></table>

Table 7: Parallel ratio (%) for each diversity metric setting, measuring how often the model triggers parallel thinking during the reasoning process.

AIME24 results with improvements across the remaining benchmarks, yielding the best overall average of 51.27%. Its selection thus reflects performance across the full suite rather than optimizing for a single benchmark. Table 7 additionally shows that the configuration with the highest parallelthinking usage on AIME25 and AIME24 is Prefix-2 exact. This distinction reinforces that effective path exploration, rather than branching frequency alone, is important for overall reasoning performance.

## E Interpreting Test-Time Scaling

The pass@k curves in Table 5 highlight DPT’s ability to convert path diversity into stronger testtime scaling. DPT is competitive at pass@1 and becomes increasingly advantageous as more complete solutions are sampled, remaining ahead of both baselines at every larger budget. From k = 64 to k = 256, DPT rises from 49.2 to 57.0, compared with 47.9 to 50.9 for Parallel-R1, demonstrating broader coverage of correct solution modes.

The scaling pattern is naturally explained by DPT’s reduced redundancy: its internal paths cover more distinct solution directions, so additional complete samples continue to contribute useful coverage rather than revisiting the same modes. The agreement between the path-diversity measurements and the pass@k curves supports diversity as a central mechanism behind DPT’s gains in both within-response exploration and test-time scaling.

## F Inference Cost and Response Length

The diversity reward is used only during training. DPT adds no inference-time judge, reward module, or additional model call beyond the native parallelthinking generation process. We analyze 1,916 validation trajectories at checkpoint 175 after removing trailing padding. Table 8 reports the measured token counts and latency together with the reported FLOP estimates.

Relative to Parallel-R1, DPT reduces tokens per problem by 4.9%, latency by 4.5%, and FLOPs by 4.2%, while also reducing tokens per solved problem by 5.5%. Thus, DPT combines stronger reasoning performance with a leaner inference profile within the parallel-thinking setting. The comparison with sequential GRPO further shows that DPT achieves its gains with substantially fewer raw tokens, highlighting the practical value of selective, diverse internal exploration.

Response length provides an intuitive explanation for this efficiency profile. DPT produces the shortest median responses and reaches the 3,000- token cap least often (Table 9), indicating that its diversity reward helps the model explore alternatives without unnecessarily prolonging each branch. The combination of concurrent generation and concise paths gives DPT a favorable latency and token profile while preserving the benefits of parallel exploration.

Overall, the measurements show that DPT’s training-time diversity objective carries through to an efficient inference profile: the model generates shorter responses, uses fewer tokens, and maintains the strongest reasoning results in the evaluated parallel setting. These properties make DPT attractive for deployments where both solution coverage and response efficiency matter.

<table><tr><td>Method</td><td>Tokens/problem</td><td>Latency (s)</td><td>TFLOPs/problem</td><td>Tokens/solved problem</td></tr><tr><td>GRPO (sequential)</td><td>1,783</td><td>0.778</td><td>17.25</td><td>3,544</td></tr><tr><td>Parallel-R1</td><td>1,666</td><td>0.710</td><td>15.64</td><td>4,004</td></tr><tr><td>DPT</td><td>1,584</td><td>0.678</td><td>14.99</td><td>3,785</td></tr></table>

Table 8: Inference cost in the checkpoint-175 validation analysis. FLOPs are reported in trillions per problem. The last column normalizes token consumption by solved problems.

<table><tr><td>Method</td><td>Median tokens</td><td>At token cap (%)</td></tr><tr><td>GRPO</td><td>1,715</td><td>24.0</td></tr><tr><td>Parallel-R1</td><td>1,491</td><td>13.0</td></tr><tr><td>DPT</td><td>1,401</td><td>10.2</td></tr></table>

Table 9: Response lengths and frequency of reaching the 3,000-token cap in the validation cost analysis.

## G Prefix Divergence as a Training Proxy

Prefix diversity provides a low-cost and effective control point for shaping exploration. Because early tokens establish the conditional context for the remainder of an autoregressive path, encouraging distinct prefixes gives parallel branches different starting points from which to develop. The resulting signal is simple to compute, easy to interpret, and directly aligned with the goal of avoiding duplicated internal reasoning.

The practical rationale is local symmetry breaking. For paths sampled from the same preceding context, changing an initial token changes the conditional distribution of subsequent tokens, creating room for branches to develop different continuations instead of immediately reproducing the same trajectory. This gives DPT a targeted exploration signal while leaving most of each path unconstrained to follow the task reward.

The prefix length n controls the extent of the intervention. The default n = 1 encourages branches to make different initial decisions while leaving almost all subsequent tokens governed by the task objective. Larger prefixes extend the exploration window, and the similarity variant provides graded credit according to how many positions differ. The ablation shows that these choices offer complementary operating points for balancing immediate accuracy and broader solution coverage.

The ablation in Appendix D illustrates these design choices empirically. Prefix-1 exact achieves the highest average accuracy across the full evaluation suite and is therefore selected as the default configuration. The strengths of other settings on individual benchmarks further motivate treating prefix length and comparison granularity as tunable components. Correctness remains the primary task signal throughout training. Diversity and format rewards are applied for the last two steps of each ten-step RL cycle, while the remaining eight steps use correctness alone. This curriculum lets DPT introduce exploration periodically while continuously anchoring learning to verifiable task success. The same design also leaves room for future extensions such as adaptive prefix selection and strategy-aware objectives.

## H Lexical Overlap and Semantic Evaluation

Whole-path Jaccard distance is useful as a diagnostic of the diversity induced by training. As a training reward, however, it couples the signal to path length and lexical choices. The controlled comparison in Table 1 shows that the prefix formulation yields a higher average score than both the whole-path Jaccard variant and Parallel-R1 without diversity regularization, supporting the value of applying the signal at the point where branches first separate. As a post-hoc diagnostic, Jaccard distance reveals how the local prefix reward propagates through complete responses. Table 3 shows substantially larger whole-path distances for DPT across all four benchmarks, indicating that early branching decisions lead to broader textual exploration throughout the generated solutions.

The 1,156 blocks used in the motivation analysis originate from validation rollouts of the reproduced Parallel-R1 model at training step 200. They cover AIME24, AIME25, AMC23, and MATH. Each included block is a generated <Parallel> span containing two parsed <Path> spans; malformed or incomplete parallel spans are excluded.

GPT-5-mini compares each pair on solution approach, reasoning structure, mathematical technique, expression style, and overall strategy diversity. It also provides an explanation for each judgment, all of which are manually reviewed to ensure reasonableness and fairness. DPT is favored across these dimensions, complementing the lexical results with evidence that its paths differ in their reasoning organization as well as their surface form. Together, the two views show that the prefix reward produces diverse responses at both the token and solution-structure levels.

## I Case Studies

In this section, we conduct a detailed analysis of our DPT and Parallel-R1 on a specific test case. For this complex-number computation problem, our DPT performs parallel reasoning by approaching the solution for r from the perspective of $r ^ { 5 } = r$ as shown in Figure 7. In particular, the two reasoning paths solve the problem from algebraic and geometric viewpoints, respectively. Although the unit-circle remark in the second path is imprecise, it still helps explore potential possibilities in the solution space. In contrast, in Parallel-R1, the model first makes an incorrect calculation (highlighted in red in Figure 8). During parallel reasoning, the two paths rely on the same core reasoning strategy: separating the modulus and the argument, identifying r = 0 or r = 1, deriving $\theta = 2 \pi k / 7 ( k = 0 , \dots , 6 )$ for $r = 1$ , and including the trivial solution $z = 0 ,$ which yields eight solutions. The differences between the two paths are limited to minor presentation details rather than substantive semantic divergence. As a result, parallel reasoning fails to explore a more diverse solution space and does not correct the earlier error.

Case: DPT   
Problem: How many complex numbers satisfy the equation ${ \pmb z } ^ { 5 } = { \overline { { \pmb z } } } ,$   
where Z is the conjugate of the complex number z?   
To solve the equation $z ^ { 5 } { = } { \overline { { z } } } ,$ , where z is a complex number and Z is its conjugate, I   
need to find all complex numbers that satisfy this relationship. I'll approach this step by   
step, inserting a parallel block at an intermediate step to present diverse perspectives or   
alternative strategies.\n\nFirst, let's express z in polar form. A complex number z can be   
written as $z = r e ^ { i \theta } ,$ , where r = |z| is the magnitude and θ is the argument. The   
conjugate of z is $\overline { { z } } = r e ^ { i \theta }$ .Now, substitute this into the equation:   
z⁵ = z   
$( r e ^ { i \theta } ) ^ { 5 } = r e ^ { - i \theta }$   
$r ^ { 5 } e ^ { i 5 \theta } = r e ^ { - i \theta }$   
For this equation to hold, the magnitudes and the arguments must be equal. First   
consider the magnitudes: $r ^ { 5 } = { \dot { r } }$ This equation must be satisfied for the equality to hold   
Now, let's solve for r.   
<Parallel>   
<Path> Think about the equation $r ^ { 5 } = r$ separately. It factors as $r ( r ^ { 4 } - 1 ) = 0 ,$ leading   
to r = 0 or r4 = 1. But r4 = 1 implies r = 1 since r ≥ 0. So the only possibilities are   
r = 0 or r = 1.</Path>   
</Parallel>   
<Summary>The parallel block presented two distinct perspectives on the equationr⁵ =   
r: one focused on solving it algebraically to find possible values for r. and the other   
considered geometric interpretation. The key takeaway was that r can only be 0 or 1.   
</Summary>   
Now, continue with the equation considering the two cases for r:   
Case 1: r = 0   
If r = 0, then z = 0. Substitute into the original equation $0 ^ { 5 } = \overline { { { 0 } } } , 0 = 0 . 7$ This holds true,   
so z = 0 is a solution   
Case 2: r = 1   
If r = 1, then from $r ^ { 5 } e ^ { i 5 \theta } = r e ^ { - i \theta } ,$ , we have $e ^ { i 5 \theta } = e ^ { - i \theta } ,$ This implies: $i 5 \theta = -$   
iθ + 2π kiθ for some integer k. Simplify: $i 5 \theta + i \theta = 2 \pi k$ i, i6θ = 2πki, 6θ =   
2πk since i cancels, θ = kπ/3. Now, θ must be in the interval [0.2π) for distinct   
solutions. Le $; \mathbf { k } = 0 , 1 , 2 , 3 , 4 , 5$ : k=0: θ = 0: k=1 $\theta = \pi / 3 ;$ k=2 $\theta = 2 \pi / 3 ;$ k=3:   
θ = π; k=4: θ = 4 π/3; k=5: θ = 5π /3. These values of θ give 6 distinct complex   
numbers on the unit circle   
In total, there is 1 solution from r = 0 and 6 from r = 1, so there are 7 solutions.  
Figure 7: Case Study of DPT

![](images/cf3c2b99ef5838333be8203eca80cedfed14d8f35079a8fabe62deaaca268f22.jpg)  
Figure 8: Case Study of Parallel-R1