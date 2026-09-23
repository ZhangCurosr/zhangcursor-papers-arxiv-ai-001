Progressive Simplification (a)

# Ladders of Thought: A Self-Evolving Curriculum of Progressively Simplified Reasoning Traces

Minghui Liu<sup>1</sup>, Thomas Magelinski<sup>2</sup>, Dehao Yuan<sup>2</sup>, Qi Yu<sup>2</sup>, Furong Huang<sup>1,2</sup>

<sup>1</sup>University of Maryland, College Park, <sup>2</sup>Capital One

Large language models (LLMs) excel at reasoning when scaled to hundreds of billions of parameters, but small- and mid-scale models remain brittle reasoners even with knowledge distillation (KD). We present Ladders-of-Thought (LoT), a framework that improves reasoning by combining progressive question rewrites with a self-evolving curriculum. LoT automatically generates semantically faithful but easier variants of reasoning problems, organizes them into dificulty buckets using step-based measures, and employs a self-evolving bandit scheduler to allocate training adaptively. Evaluated on two reasoning domains, math and multi-hop reasoning, across 1-8B models from diferent families, LoT consistently improves over KD. It delivers large gains on arithmetic tasks $( \mathrm { e . g . , + 3 2 }$ percentage points on AddSub, +25pp on SVAMP), +2–8pp improvements on in-domain test splits, and strong though dataset-dependent benefits on multi-hop reasoning (e.g., +16pp on $\mathrm { Q A S C , + 2 5 p p }$ on StrategyQA). LoT also converges faster than staged curricula, highlighting the value of adaptive progression. These results show that progressive rewrites coupled with adaptive curricula provide a simple yet efective recipe for strengthening reasoning in smaller LLMs.

Correspondence: Furong Huang; https://furong-huang.com; furongh@umd.edu Contact: minghui@umd.edu

![](images/759e7faccbbc8683ed1da7ba8ed4fc4d06dca4d68a7460c05e3761db400a27aa.jpg)  
Figure 1 Overview of our framework. (a) Progressive simplification: original reasoning questions are rewritten into semantically faithful but progressively easier variants, forming a dificulty ladder. (b) Step-based difficulty measure bucketing: each question is assigned a dificulty score based on the number of required reasoning steps. This score is then used to place each example in a bucket. (c) Self-evolving curriculum: a multi-armed bandit scheduler adaptively selects examples from diferent buckets to maximize student learning progress.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable progress on complex reasoning benchmarks, especially when augmented with test-time prompting strategies such as chain-of-thought (CoT) reasoning (Wei et al., 2022; Zhang et al., 2022b), self-consistency (Wang et al., 2022), and structured search methods including tree-of-thoughts (Yao et al., 2023), cumulative reasoning (Zhang et al., 2023), and DUP (Zhong et al., 2024). These approaches highlight the power of explicit reasoning traces in guiding LLMs toward more accurate and robust answers.

Our focus is on small- and mid-scale LLMs, where limited capacity, brittle chain evaluation, and

large student–teacher gaps make reasoning training especially challenging.

Despite recent advances, most improvements are concentrated in very large models. Smaller models, while cheaper and more eficient, often fail to benefit from CoT-style prompting and remain brittle reasoners. A key limitation is their poor ability to generalize learned reasoning beyond a specific dataset. This challenge has motivated extensive work on reasoning distillation from large to small models, spanning standard distillation (Hinton et al., 2015; Ho et al., 2022; Magister et al., 2022; Mitra et al., 2023; Fu et al., 2023), symbolic distillation (West et al., 2021), verifier-assisted methods (Li et al., 2023; Zhang et al., 2024; Liu et al., 2023), knowledge-augmented reasoning (Kang et al., 2023), and self-consistent objectives (Wang et al., 2023a). While encouraging, these approaches struggle when the gap between student and teacher is large: small models often overfit to surface heuristics instead of acquiring transferable reasoning skills (Wang et al., 2023b; Li et al., 2025).

Curriculum learning (CL) ofers a natural remedy. CL suggests that ordering training examples from easy to hard improves both sample eficiency and generalization (Bengio et al., 2009; Matiisen et al., 2019; Soviany et al., 2022; Narvekar et al., 2020). Adaptive curricula, which dynamically select training examples, often work even better (Jiang et al., 2015; Kong et al., 2021). While CL has been explored in in-context learning (Liu et al., 2024) and reinforcement learning (Chen et al., 2025; Parashar et al., 2025), its potential for supervised fine-tuning of reasoning remains underexplored. A major obstacle is defining dificulty for reasoning problems: length, number of inferential steps, and information structure all interact in complex ways (Jin et al., 2024; Wang et al., 2025; Shi et al., 2025).

Our Approach. We introduce Ladders-of-Thought (LoT), a framework for training stronger reasoning in smalland mid-scale LLMs through a combination of progressive question rewrites and self-evolving curricula. Our method builds on three insights: (1) Reasoning questions can be automatically rewritten into progressively easier versions while preserving semantics, forming a natural “ladder” of dificulty. (2) The minimal number of reasoning steps provides a principled dificulty measure for organizing training buckets. (3) A self-evolving curriculum scheduler, framed as a multi-armed bandit, can adaptively allocate training to dificulty levels where the student learns fastest, avoiding rigid or suboptimal schedules.

Contributions. This paper makes three contributions:

• We introduce a progressive rewrite framework that generates semantically faithful but easier variants of reasoning problems, creating a principled dificulty ladder.

• We propose an adaptive, self-evolving curriculum scheduler that dynamically allocates training across dificulty buckets using bandit-based updates.

• We demonstrate through extensive experiments that LoT improves pass@5 accuracy, accelerates convergence, and strengthens out-of-distribution generalization for small- and mid-scale LLMs.

## 2 Background

We briefly review the foundations of our approach: chain-of-thought (CoT) distillation, curriculum learning, and multi-armed bandit scheduling.

Chain-of-Thought Distillation. CoT distillation transfers reasoning ability from a large teacher to a smaller student by supervising on teacher-generated rationales (Ho et al., 2022; Chae et al., 2023). Given $\mathcal { D } =$ $\{ ( \boldsymbol { x } ^ { ( i ) } , \boldsymbol { y } ^ { ( i ) } ) \}$ , we prompt the teacher with zero-shot CoT instructions (Wei et al., 2022; Zhang et al., 2022b) to obtain rationales $\mathbf { \chi } _ { r } ( i )$ . Training instances are formatted as

$$
\mathrm { Q u e s t i o n : } \ x ^ { ( i ) } \quad \mathrm { A n s w e r : } \ r ^ { ( i ) } , y ^ { ( i ) } .
$$

The student autoregressively generates $r ^ { ( i ) }$ and $\boldsymbol y ^ { ( i ) }$ , optimized via negative log-likelihood:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C o T } } ( \theta ) = - \displaystyle \sum _ { i } \Big ( \sum _ { j } \log P _ { \theta } ( r _ { j } ^ { ( i ) } \mid r _ { < j } ^ { ( i ) } , x ^ { ( i ) } ) } \\ { + \sum _ { j } \log P _ { \theta } ( y _ { j } ^ { ( i ) } \mid y _ { < j } ^ { ( i ) } , r ^ { ( i ) } , x ^ { ( i ) } ) \Big ) . } \end{array}
$$

This encourages the student to reproduce step-by-step reasoning and final answers.

Curriculum Learning. Curriculum learning (Bengio et al., 2009) presents data in a structured order. A dificulty function $d ( x )$ partitions D into buckets $\{ B _ { 1 } , \ldots , B _ { K } \}$ , ordered by dificulty. A curriculum defines a sequence of sampling distributions $\{ p _ { t } \} _ { t = 1 } ^ { T } .$ , where $p _ { t } ( b )$ is the probability of drawing from bucket $B _ { b }$ at step t. Fixed curricula move gradually from easy to hard, while self-evolving ones adjust $p _ { t }$ based on model progress.

Multi-Armed Bandits. Self-evolving curricula can be framed as a multi-armed bandit (MAB) problem, where each bucket $\boldsymbol { B } _ { k }$ corresponds to an arm. At step t, the scheduler selects arm $a _ { t } \in \{ 1 , \ldots , K \}$ according to $p _ { t } .$ samples from $B _ { a _ { t } }$ , and receives reward $r _ { t }$ (e.g., validation improvement). The objective is to minimize regret

$$
R _ { T } = \operatorname* { m a x } _ { k } \sum _ { t = 1 } ^ { T } r _ { t } ^ { ( k ) } - \sum _ { t = 1 } ^ { T } r _ { t } ,
$$

where $r _ { t } ^ { ( k ) }$ is the reward had arm k been played. Strategies such as ϵ-greedy and Boltzmann exploration balance exploration with exploitation. We employ such a scheduler to adapt $p _ { t }$ online.

## 3 Method: Ladders-of-Thought (LoT)

LoT constructs curricula for reasoning tasks through two key components: (i) progressive rewrites, which generates graded versions of each question by injecting intermediate reasoning steps (Figure 2), and (ii) step-based dificulty labeling, which assigns a consistent measure of problem dificulty. These components together yield dificulty-labeled question sets that can be organized into either staged or adaptive self-evolving curricula (Figure 1).

## 3.1 Progressive Rewrites

We start with a question–solution pair $( q , s )$ where the question q contains an explicit set of premises $\mathcal { P } = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { m } \}$ and the solution is expressed as a chain-of-thought (CoT) sequence $s = ( c _ { 1 } , c _ { 2 } , \ldots , c _ { n } )$ Each reasoning step derives a new conclusion $c _ { i }$ from a small set of antecedents $A _ { i } \subseteq { \mathcal { P } } \cup \left\{ c _ { 1 } , \ldots , c _ { i - 1 } \right\}$ ; for example, $p _ { 1 } + p _ { 2 } \Rightarrow c _ { 1 }$ and then $c _ { 1 } + p _ { 3 } \Rightarrow c _ { 2 }$

Rewrite operation. Rather than merely appending conclusions to the context, we replace the antecedents of each step by the derived conclusion. Concretely, let $q ^ { ( 0 ) } = q .$ For $i = 1 , \ldots , n ,$ , form

$$
q ^ { ( i ) } \ = \ { \big ( } q ^ { ( i - 1 ) } \setminus A _ { i } { \big ) } \ \cup \ \{ c _ { i } \} .
$$

Intuitively, if $p _ { 1 }$ and $p _ { 2 }$ entail $c _ { 1 } .$ , we remove $p _ { 1 } , p _ { 2 }$ from the question and insert $c _ { 1 }$ instead, yielding an easier instance. Applying this transformation step-by-step produces a sequence $\boldsymbol q ^ { ( 0 ) } , \boldsymbol q ^ { ( 1 ) } , \dots , \boldsymbol q ^ { ( n ) }$ of strictly decreasing dificulty, terminating when the answer is trivial (or explicitly recoverable) in the context (Figure 2).

![](images/d8b2adda9e2d0a26fa2ad63c4e63f0d071ee1e6319c669e28741b0ac8c41a9fc.jpg)  
Figure 2 Progressive rewrites. Each task can be converted into a series of standalone premises $( P 1 , P 2 , \ldots )$ . Each reasoning step combines two pieces of information to make a conclusion (C), or new piece of information. After each step of reasoning, the total amount of information is smaller, giving an easier sub-question to solve. Thus, progressively easier questions arise naturally from step-by-step problem solving, while preserving semantics and solvability (no answer leakage), since each rewrite replaces a subset of premises with their logically entailed conclusion.

Practical generation. We prompt a capable instruction-tuned LLM to (i) identify $A _ { i }$ for each CoT step and (ii) produce the simplified question $\boldsymbol { q } ^ { ( i ) }$ while preserving semantics and well-posedness. The rewriting model need not coincide with the teacher used for CoT supervision; in practice, we may use a strong CoT generator as the teacher and a separate model for controlled rewriting. This procedure pairs every complex question with progressively easier counterparts, forming the backbone of our curriculum.

Comparison to decomposition. This simplification difers from problem decomposition methods such as Simonds and Yoshiyama (2025), which generate related but distinct subproblems. Our rewrites retain the original problem identity while replacing subsets of premises with intermediate conclusions, i.e., they are the same task presented with precomputed inferences in the premise.

## 3.2 Difficulty Labeling via Step Definition

We define the dificulty of a reasoning example by the model-estimated minimal number of steps required to reach a solution, denoted $\phi ( x )$ . While $\phi ( x )$ is model-estimated rather than ground-truth minimal, it provides a consistent ordering aligned with the supervision signal used during training. For instance, a math problem that requires three arithmetic operations has $\phi ( x ) = 3$ . Rather than relying on raw chain-of-thought (CoT) length, which can be inflated by verbosity or stylistic padding, our progressive rewriting procedure enforces a one-step decrement at each stage $( \mathrm { e . g . , ~ 3 \to 2 \to 1 \to 0 } )$ . Thus $\phi ( x )$ aligns directly with the number of rewrites available for each example, providing a consistent and interpretable dificulty measure. The rewriting model is given explicit instructions on what constitutes a reasoning step to maintain consistent granularity, and we manually spot-check examples to verify the monotonic decrease. This process yields well-calibrated step counts that serve as interpretable dificulty labels. Training data are then bucketed by these labels, ${ \mathcal { B } } _ { k } = \{ x : \phi ( x ) \in I _ { k } \}$ , providing a structured progression from easier to harder questions. Complete prompts for step counting and rewriting are provided in Appendix A.8.

## 3.3 Curriculum Construction

The dificulty-labeled questions naturally form a curriculum. Because the distribution of step counts is often imbalanced, we group adjacent levels into buckets $( \mathrm { e . g . } , 1 \mathrm { - 3 } , 4 \mathrm { - } 5 .$ , and 6+ steps as “easy,” “medium,” and “hard”). These buckets support both staged and self-evolving curriculum strategies.

A simple baseline is the staged curriculum, where buckets are ordered by dificulty and the model trains on one bucket at a time for a fixed number of steps. This provides a straightforward schedule against which self-evolving methods can be compared.

For adaptivity, we follow the multi-armed bandit (MAB) framework of Matiisen et al. (2019), which treats each bucket $B _ { k } \in \{ B _ { 1 } , \ldots , B _ { K } \}$ as an arm. At step t, the learner selects an arm $a _ { t } ,$ , trains on samples from bucket $B _ { a _ { t } }$ , and receives a reward derived from validation performance.

The Q-value update is

$$
Q _ { t + 1 } ( a ) = \alpha r _ { t } ( a ) + ( 1 - \alpha ) Q _ { t } ( a ) ,
$$

with learning rate α and $Q _ { 0 } ( a ) = 0$

Every m steps, we compute rewards as

$$
r _ { t } ( a ) = \mathrm { A c c } _ { t } ( a ) - \overline { { \mathrm { A c c } } } _ { t } ( a ) ,
$$

where ${ \overline { { \operatorname { A c c } } } } _ { t } ( a )$ is an exponential moving average with smoothing coeficient $\beta .$ This measures the accuracy gain relative to baseline.

Buckets are then sampled either from a Boltzmann distribution

$$
\pi _ { t } ( a ) \propto \exp ( Q _ { t } ( a ) / \tau ) ,
$$

with temperature $\tau ,$ or via an ϵ-greedy policy that chooses the best bucket with probability 1 − ϵ and explores otherwise.

This bandit-based scheduler dynamically focuses training on the levels that yield the greatest marginal improvement, producing a self-evolving curriculum. The full training procedure, including progressive rewrites, bucketization, and adaptive scheduling, is summarized in Algorithm 1 (see Appendix A.1 for details).

## 4 Experiments

We evaluate whether progressive rewrites combined with a self-evolving curriculum improve reasoning generalization. Our experiments focus on two questions: (i) Does LoT provide consistent gains over strong baselines across models and domains? (ii) How do rewrite depth and curriculum scheduling afect performance?

## 4.1 Setup

We study two domains: math and multi-hop reasoning. For math, models are trained on GSM8K Cobbe et al. (2021) and evaluated on its test split plus AddSub, ASDiv, MultiArith, and SVAMP Hosseini et al. (2014); Miao et al. (2020); Roy and Roth (2015); Patel et al. (2021). For multi-hop, models are trained on EntailmentBank Dalvi et al. (2021) and tested on its split plus StrategyQA, OpenBookQA, QASC, and MuSiQue Geva et al. (2021); Mihaylov et al. (2018); Yang et al. (2018); Khot et al. (2020); Trivedi et al. (2022).

We evaluate OPT-1.3B/2.7B Zhang et al. (2022a) and Pythia-1.4B/2.8B Biderman et al. (2023), using knowledge distillation (KD) from a strong CoT teacher. Baselines include: (i) the base model, (ii) CoT KD on original data, and (iii) LoT (ours): KD with rewrites under a self-evolving curriculum.

We report pass@5 accuracy<sup>1</sup> to reduce decoding variance; mean ± standard error and greedy decoding pass@1 results are reported in Appendix A.4 and show consistent trends. Further experimental details including hyperparameters, hardware and evaluation harness are provided in Appendix A.3.

## 4.2 Main Results

Tables 1 and 2 summarize pass@5 accuracy across both domains and model families. LoT consistently outperforms KD on original data, with especially large gains on math reasoning. For example, on OPT-2.7B, AddSub accuracy jumps from 8.26 to 40.37 (+32.11 percentage points), and SVAMP from 19.06 to 44.15 (+25.09 percentage points).

Improvements are also evident on in-domain test splits: GSM8K rises from 31.01 to 33.97 (+2.96), while EntailmentBank improves by +3–8 percentage points across all model families. Across architectures, Pythia-1.4B improves on ASDiv from 21.36 to 40.78 (+19.42), while Pythia-2.8B gains +20.18 on AddSub and +20.74 on SVAMP.

Two trends stand out in math reasoning. First, LoT yields the largest gains on smaller, compositional arithmetic datasets such as AddSub, ASDiv, and SVAMP. These datasets difer substantially from the GSM8K training distribution, highlighting LoT’s strength in improving out-of-distribution generalization. Second, while improvements on GSM8K itself are more modest (+2–3 points), LoT consistently prevents degradation and provides robustness, suggesting that introducing easier rewrites does not harm in-domain accuracy while improving transferability.

For multi-hop reasoning, LoT provides both in-domain and out-of-domain benefits when trained on EntailmentBank. In-domain accuracy rises on the EntailmentBank test split (+3–8), showing that rewrites help the model capture inference patterns more reliably. Out-of-domain, LoT delivers strong improvements on QASC (+4–16) and StrategyQA (+17–25), and also boosts MuSiQue substantially for OPT-1.3B (+25) and Pythia-2.8B (+2.8).

Although LoT improves substantially on QASC and StrategyQA, we observe some regressions on MuSiQue and OpenBookQA. Both tasks lie far outside the supervision domain: models are trained only on EntailmentBank, while MuSiQue and OpenBookQA rely more heavily on factual retrieval, entity grounding, and multi-evidence aggregation than on compositional inference. In these settings, stronger sensitivity to step-structured reasoning patterns and reduced exposure to factual variability in training may limit transfer. These results suggest that LoT provides the largest benefits when the target task shares the same underlying inferential structure as the curriculum, and that complementary mechanisms (e.g., retrieval augmentation) may be required to support transfer to knowledge-centric QA.

Overall, LoT delivers improvements across all four model checkpoints and both reasoning domains. Its benefits are architecture-agnostic and extend beyond in-domain test sets to multiple out-of-distribution benchmarks, though the magnitude of gains is more uniform in arithmetic reasoning than in multi-hop tasks.

## 4.3 Larger Students: Qwen2.5–7B and Llama3.1–8B

To evaluate whether Ladders-of-Thought scales beyond small and mid-sized students, we additionally trained Qwen2.5–7B and Llama3.1–8B models on the same GSM8K-based LoT curriculum. Results are shown in Table 3.

The results show that LoT scales efectively to larger student models. For both Qwen2.5-7B and Llama3.1- 8B, LoT matches and sometimes slightly improves over KD on GSM8K and yields substantially larger gains on out-of-distribution tasks. Qwen2.5-7B LoT achieves strong improvements on AddSub (+20.2) and SVAMP (+11.0), while Llama3.1–8B shows similar boosts (+31.2 AddSub, +10.4 SVAMP). These gains mirror the trends observed at the 1-3B scale: LoT mainly enhances compositional generalization rather than in-distribution accuracy alone. Overall, the results indicate that LoT is not limited to small models and continues to strengthen transfer beyond the training distribution as model size increases.

## 4.4 Ablation: Rewrite Depth

Table 4 shows that rewrite depth has a pronounced efect on performance. Introducing shallow rewrites (≤1) yields the largest single jump in accuracy (+28.48 percentage points on average), and performance continues to increase up to ≤3, especially on benchmarks requiring multi-step arithmetic composition (e.g., +6.11 on MultiArith at ≤3 and +15.05 on SVAMP at ≤2). However, using all rewritten variants leads to diminishing or negative returns, suggesting that excessive exposure to very easy variants can dilute the core reasoning signal and reduce generalization.

<table><tr><td>Methods</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td></tr><tr><td colspan="6">OPT-1.3B</td></tr><tr><td>Base</td><td>3.79</td><td>1.83</td><td>4.05</td><td>2.22</td><td>5.69</td></tr><tr><td>CoT KD</td><td>27.75</td><td>9.17</td><td>20.23</td><td>63.33</td><td>20.74</td></tr><tr><td>LoT (Ours)</td><td>31.16 (+3.41)</td><td> $3 3 . 0 3 \ ( + 2 3 . 8 6 )$ </td><td> $4 1 . 7 5 \ \scriptstyle ( + 2 1 . 5 2 )$ </td><td> $6 8 . 3 3 \ \left( + 5 . 0 0 \right)$ </td><td>38.46 (+17.72)</td></tr><tr><td colspan="6">OPT-2.7B</td></tr><tr><td>Base</td><td>3.34</td><td>1.83</td><td>4.21</td><td>3.33</td><td>6.69</td></tr><tr><td>CoT KD</td><td>31.01</td><td>8.26</td><td>26.38</td><td>71.11</td><td>19.06</td></tr><tr><td>LoT (Ours)</td><td>33.97 (+2.96)</td><td> $4 0 . 3 7 \ _ { ( + 3 2 . 1 1 ) }$ </td><td>45.95 (+19.57)</td><td>80.56 (+9.45)</td><td>44.15 (+25.09)</td></tr><tr><td colspan="6">Pythia-1.4B</td></tr><tr><td>Base</td><td>2.96</td><td>0.00</td><td>4.85</td><td>1.67</td><td>8.03</td></tr><tr><td>CoT KD</td><td>26.00</td><td>3.67</td><td>21.36</td><td>59.44</td><td>19.73</td></tr><tr><td>LoT (Ours)</td><td> $2 8 . 3 5 \ _ { ( + 2 . 3 5 ) }$ </td><td> $2 4 . 7 7 \ ( + 2 1 . 1 0 )$ </td><td> $4 0 . 7 8 \ ( + 1 9 . 4 2 )$ </td><td> $6 3 . 3 3 \ ( + 3 . 8 9 )$ </td><td> $3 8 . 4 6 ~ ( + 1 8 . 7 3 )$ </td></tr><tr><td colspan="6">Pythia-2.8B</td></tr><tr><td>Base</td><td>3.71</td><td>1.83</td><td>6.63</td><td>4.44</td><td>9.70</td></tr><tr><td>CoT KD</td><td>33.43</td><td>11.93</td><td>31.88</td><td>74.44</td><td>24.08</td></tr><tr><td>LoT (Ours)</td><td> $3 2 . 9 8 \ _ { ( - 0 . 4 5 ) }$ </td><td> $3 2 . 1 1 \ ( + 2 0 . 1 8 )$ </td><td> $4 6 . 7 6 \ \scriptstyle ( + 1 4 . 8 8 )$ </td><td> $7 2 . 7 8 \ \scriptstyle ( - 1 . 6 6 )$ </td><td> $4 4 . 8 2 \ ( + 2 0 . 7 4 )$ </td></tr></table>

Table 1 Pass@5 accuracy (%) on GSM8K and out-of-distribution math benchmarks. Each entry shows absolute accuracy with ∆ relative to CoT KD. LoT consistently improves generalization, with the largest gains on AddSub, ASDiv, and SVAMP (+15–30 percentage points). (∆s are rendered in green/red for increases/decreases.)
<table><tr><td>Methods</td><td>EntailmentBank</td><td>QASC</td><td>OpenBookQA</td><td>StrategyQA</td><td>MuSiQue</td></tr><tr><td colspan="6">OPT-1.3B</td></tr><tr><td>Base</td><td>22.0</td><td>17.2</td><td>14.8</td><td>32.4</td><td>2.2</td></tr><tr><td>CoT KD</td><td>36.0</td><td>46.0</td><td>47.4</td><td>22.6</td><td>14.2</td></tr><tr><td> $\operatorname { L o T } \left( \operatorname { O u r s } \right)$ </td><td> $4 1 . 0 \ _ { ( + 5 . 0 ) }$ </td><td> $5 2 . 8 \ _ { \textrm { ( + 6 . 8 ) } }$ </td><td> $4 3 . 6 \textrm {  { \Omega } } _ { ( - 3 . 8 ) }$ </td><td> $4 7 . 8 \ ( + 2 5 . 2 )$ </td><td>39.2 (+25.0)</td></tr><tr><td colspan="6">OPT-2.7B</td></tr><tr><td>Base</td><td>21.0</td><td>28.8</td><td>12.6</td><td>33.6</td><td>2.2</td></tr><tr><td>CoT KD</td><td>40.0</td><td>56.2</td><td>46.0</td><td>54.0</td><td>43.6</td></tr><tr><td>LoT (Ours)</td><td> $4 1 . 0 \ _ { ( + 1 . 0 ) }$ </td><td> $6 0 . 4 \textrm { } ( + 4 . 2 )$ </td><td> $4 7 . 8 \ \scriptstyle ( + 1 . 8 )$ </td><td> $5 1 . 2 \ _ { ( - 2 . 8 ) }$ </td><td>24.0 (-19.6)</td></tr><tr><td colspan="6">Pythia-1.4B</td></tr><tr><td>Base</td><td>13.0</td><td>45.0</td><td>34.2</td><td>24.4</td><td>12.0</td></tr><tr><td>CoT KD</td><td>36.0</td><td>55.2</td><td>44.6</td><td>36.4</td><td>42.0</td></tr><tr><td>LoT (Ours)</td><td> $3 9 . 0 \ \scriptstyle ( + 3 . 0 )$ </td><td> $5 6 . 6 \ \AA \ ( + 1 . 4 )$ </td><td> $3 6 . 8 \ _ { \textrm { ( - 7 . 8 ) } }$ </td><td> $5 3 . 2 \ _ { ( + 1 6 . 8 ) }$ </td><td>39.2 (−2.8)</td></tr><tr><td colspan="6">Pythia-2.8B</td></tr><tr><td>Base</td><td>10.0</td><td>39.0</td><td>32.4</td><td>29.2</td><td>18.8</td></tr><tr><td>CoT KD</td><td>32.0</td><td>32.2</td><td>28.2</td><td>55.6</td><td>42.2</td></tr><tr><td>LoT (Ours)</td><td> $4 0 . 0 \ _ { ( + 8 . 0 ) }$ </td><td> $4 8 . 8 \ \left( + 1 6 . 6 \right)$ </td><td> $3 7 . 8 \ \scriptstyle ( + 9 . 6 )$ </td><td> $6 0 . 0 \ _ { ( + 4 . 4 ) }$ </td><td> $4 5 . 0 \ _ { \textrm { ( + 2 . 8 ) } }$ </td></tr></table>

Table 2 Pass@5 accuracy (%) on EntailmentBank (in-domain) and four out-of-domain multi-hop benchmarks. LoT improves EntailmentBank by +3–8 percentage points across model families and yields strong gains on QASC (+4–16) and StrategyQA (+17–25). Performance is more mixed on OpenBookQA and MuSiQue (some regressions for smaller models; Pythia-2.8B still improves). ∆ values are relative to CoT KD (green/red = increase/decrease).

To better understand this trend, Appendix A.5 presents a distributional analysis of minimal reasoning steps under diferent rewrite depths (Table 13 and Figure 5). As rewrite depth increases, the training distribution becomes increasingly skewed toward low-step (i.e., easier) instances. Taken together, these results indicate that LoT benefits from a moderate curriculum ladder: shallow-to-intermediate rewrites broaden exposure to simpler reasoning structures, while preserving a suficient range of dificulty to avoid over-regularizing the model toward trivial problems.

<table><tr><td>Methods</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td></tr><tr><td colspan="6">Qwen2.5-7B</td></tr><tr><td>Base</td><td>21.61</td><td>5.50</td><td>11.17</td><td>15.00</td><td>10.70</td></tr><tr><td>KD</td><td>73.84</td><td>50.46</td><td>77.67</td><td>98.33</td><td>60.54</td></tr><tr><td>LoT (Ours)</td><td>74.60 (+0.76)</td><td>70.64 (+20.18)</td><td> $7 4 . 9 2 \ _ { ( - 2 . 7 5 ) }$ </td><td> $9 8 . 8 9 \ \scriptstyle ( + 0 . 5 6 )$ </td><td>71.57 (+11.03)</td></tr><tr><td colspan="6">Llama3.1-8B</td></tr><tr><td>Base</td><td>16.38</td><td>35.78</td><td>29.45</td><td>17.78</td><td>30.77</td></tr><tr><td>KD</td><td>53.22</td><td>25.69</td><td>50.32</td><td>92.22</td><td>40.13</td></tr><tr><td>LoT (Ours)</td><td> $5 6 . 1 0 \ \scriptstyle ( + 2 . 8 8 )$ </td><td>56.88 (+31.19)</td><td> $4 7 . 9 0 \ \scriptstyle ( - 2 . 4 2 )$ </td><td> $9 3 . 3 3 \ ( + 1 . 1 1 )$ </td><td>50.50 (+10.37)</td></tr></table>

Table 3 Pass@5 accuracy (%) on GSM8K and out-of-distribution math benchmarks. ∆ indicates absolute change vs KD.
<table><tr><td>Depth</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td><td>Average</td></tr><tr><td>0</td><td>10.46</td><td>6.42</td><td>9.22</td><td>17.78</td><td>7.02</td><td>10.18</td></tr><tr><td> ${ \leq } 1$ </td><td>30.55 (+20.09)</td><td>26.61 (+20.19)</td><td> $4 0 . 6 1 \ \scriptstyle ( + 3 1 . 3 9 )$ </td><td> $6 7 . 7 8 \ ( + 5 0 . 0 0 )$ </td><td> $2 7 . 7 6 \ \scriptstyle ( + 2 0 . 7 4 )$ </td><td> $3 8 . 6 6 \ _ { \mathrm { ~ ( + 2 8 . 4 8 ) ~ } }$ </td></tr><tr><td> $\leq 2$ </td><td>28.81  $\left( - 1 . 7 4 \right)$ </td><td>32.11 (+5.50)</td><td> $4 8 . 2 2 \ _ { ( + 7 . 6 1 ) }$ </td><td> $7 1 . 1 1 \ \mathrm { \Omega } ( + 3 . 3 3 )$ </td><td> $4 2 . 8 1 \ _ { ( + 1 5 . 0 5 ) }$ </td><td> $4 4 . 6 1 \ _ { ( + 5 . 9 5 ) }$ </td></tr><tr><td> ${ \le } 3$ </td><td>32.07  $( + 3 . 2 6 )$ </td><td> $3 0 . 2 8 \ _ { \ ( - 1 . 8 3 ) }$ </td><td>47.73 (-0.49)</td><td> $7 7 . 2 2 \ ( + 6 . 1 1 )$ </td><td> $4 3 . 1 4 \ \scriptstyle ( + 0 . 3 3 )$ </td><td> $4 6 . 0 9 \ \scriptstyle ( + 1 . 4 8 )$ </td></tr><tr><td>All</td><td>31.16 (−0.91)</td><td> $3 3 . 0 3 \ \scriptstyle ( + 2 . 7 5 )$ </td><td> $4 1 . 7 5 \ \scriptstyle ( - 5 . 9 8 )$ </td><td> $6 8 . 3 3 \ ( - 8 . 8 9 )$ </td><td> $3 8 . 4 6 \ \scriptstyle \left( - 4 . 6 8 \right)$ </td><td> $4 2 . 5 5 \ \scriptstyle ( - 3 . 5 4 )$ </td></tr></table>

Table 4 Pass@5 accuracy (%) when varying maximum rewrite depth. Performance improves sharply when adding shallow rewrites (≤1), continues to grow up to depth 3, and declines when all rewrites are included. Deltas are relative to the row above (green = improvement, red = decrease).

## 4.5 Ablation: Curriculum Scheduling

We compare four curriculum strategies: (i) Flat Sampling (random training without curriculum), (ii) Staged Curriculum (Easy→Hard), (iii) Staged Curriculum (Hard→Easy), and (iv) Self-evolving Curriculum (ours).

Figure 3 and Figure 4 highlight the importance of curriculum design. LoT’s Self-evolving Curriculum achieves both the fastest convergence and the highest final accuracy, outperforming all fixed schedules. Easy→Hard also improves over Flat sampling, confirming that sequencing problems from simple to complex is more efective than random order. By contrast, Hard→Easy performs worst across the board, lagging in both early and late training. This supports the intuition that exposing models to dificult problems before they have acquired simpler reasoning patterns hinders progress.

Interestingly, Flat sampling often shows reasonable early learning speed, but plateaus at lower accuracy. LoT combines the best of both worlds: it retains early learning eficiency while ultimately achieving stronger final performance. This indicates that adaptivity, rather than a fixed progression, is key for balancing eficiency and generalization.

## 4.6 Overhead of LoT

LoT adds two sources of overhead: ofline rewrite generation and the online MAB scheduler. Rewrite generation is performed once before training, and its token counts and cost estimates are reported in Appendix A.2. During training, we profiled the wall-clock time across all models and found that the MAB scheduler accounts for only 3-8% of the total runtime. The remaining compute is identical to standard supervised fine-tuning. Thus, LoT introduces minimal computational overhead in practice.

## 4.7 Discussion

Taken together, these analyses show that: (1) LoT consistently boosts reasoning performance, with especially large gains on OOD arithmetic benchmarks; (2) Rewrite depth should be moderate—shallow to intermediate levels provide strong generalization benefits, while excessive depth can hurt; and (3) Curriculum scheduling strongly afects outcomes, with self-evolving strategies clearly outperforming static or reversed schedules, underscoring the importance of curriculum direction and adaptivity.

![](images/b9983634b4a48b4bb583d5917db62f8bb409dbace52bb5e50d75c6e83a49d8e8.jpg)  
Figure 3 GSM8K validation accuracy over training steps under diferent curriculum strategies. Self-evolving (Ours) (green) and Flat (orange) achieve both faster learning and high final accuracy. Easy→Hard (blue) is moderately efective, while Hard→Easy (red) consistently underperforms.

![](images/6d9aee83a6aed7455cde43f6413ec05174d7b71b02b8c4612dbc24acc4ee207b.jpg)  
Figure 4 Average test accuracy across math reasoning benchmarks under diferent curriculum strategies. Selfevolving (Ours) (green) achieves the best performance. Both Easy→Hard (blue) and Self-evolving outperform Flat (orange), showing that introducing easier problems first leads to stronger learning, while Hard→Easy (red) harms performance.

Overall, these findings suggest that LoT provides a principled recipe for enhancing reasoning models: use faithful but easier rewrites, structure them into a moderate-depth ladder, and adaptively adjust exposure to maximize sample eficiency and generalization. Importantly, LoT achieves these gains with minimal computational overhead, since rewrite generation is performed entirely ofline and the MAB scheduler adds only a small fraction of total training time.

## 5 Related Works

LLM Reasoning and Distillation. To transfer reasoning ability to compact LLMs, many works explore distillation (Xu et al., 2024; Yang et al., 2024). Supervised fine-tuning on teacher-generated CoT traces improves small models (Mitra et al., 2023; Magister et al., 2022; Ho et al., 2022; Gu et al., 2023), with variants such as symbolic distillation (West et al., 2021), verifier-assisted training (Liu et al., 2023; Zhang et al., 2024), knowledge-augmented objectives (Kang et al., 2023), and white-box supervision using hidden states (Deng et al., 2023). Despite progress, distillation often breaks down when the student–teacher gap is large, leading to overfitting to shallow heuristics and poor generalization (Li et al., 2025).

Question Decomposition. A line of recent work improves reasoning by decomposing questions into smaller sub-tasks or auxiliary queries. Least-to-Most Prompting (Zhou et al., 2022) and Self-Ask (Press et al., 2023) generates a sequence of sub-questions at inference time, without modifying the underlying training distribution. Divide-or-Conquer (Wu et al., 2024) similarly constructs sub-questions but focuses on disentangling decomposition from solving to study which component is easier to distill. LADDER (Simonds and Yoshiyama, 2025) produces hierarchical supervision but still introduces additional sub-problems rather than modifying the original instance. In contrast, LoT does not decompose problems into multiple auxiliary tasks. Instead, it performs progressive simplification of the same question, replacing antecedent premises with their entailed intermediate conclusions. This preserves semantic equivalence while inducing a structured, monotonic dificulty ladder tied directly to minimal reasoning depth.

Rationale Refinement and Process-Supervision. LoT is also related to methods that refine rationales or generate structured intermediate representations. Self-Refine (Madaan et al., 2023) and Reflexion (Shinn et al., 2023) iteratively improve model outputs via self-feedback loops, while Program-of-Thoughts (Chen et al., 2022) and PAL (Gao et al., 2023) disentangle computation from reasoning using executable programs.

These approaches operate on generated rationales or outputs, rather than rewriting the input problem itself, and do not yield dificulty-aligned variants of training examples. Moreover, unlike fixed curricula or staged supervision used in some earlier reasoning pipelines, LoT pairs its automatically simplified variants with a non-stationary multi-armed-bandit scheduler, yielding an adaptive training curriculum that evolves with student performance.

Curriculum Learning. Curriculum learning (CL) suggests ordering examples from easy to hard to accelerate training and improve generalization (Bengio et al., 2009; Narvekar et al., 2020; Soviany et al., 2022). Extensions include self-paced (Jiang et al., 2015) and adaptive methods (Matiisen et al., 2019; Kong et al., 2021). For LLMs, curricula have been studied in in-context learning (Liu et al., 2024) and reinforcement learning (Shi et al., 2025; Chen et al., 2025; Parashar et al., 2025). Closest to our setting, Chen et al. (2025) also propose self-evolving curricula, but in RL optimization rather than supervised fine-tuning.

Dificulty Estimation. Dificulty measures are critical to CL. Prior work has used proxy signals such as MCTS heuristics (Wang et al., 2025), dataset-provided dificulty labels (Chen et al., 2025), or model hit rates (Shi et al., 2025). Other analyses show that longer chains help only when they add true inferential depth (Jin et al., 2024). We instead introduce a step-based measure grounded in the minimal number of reasoning steps, which directly aligns with our progressive rewrites and avoids noisy proxies such as raw CoT length.

Positioning. Ladders-of-Thought (LoT) integrates these threads by combining: (i) progressive rewrites inspired by distillation, (ii) step-based dificulty estimation, and (iii) adaptive scheduling from CL. Unlike prior eforts focused on large models, heuristic dificulty proxies, or alternate settings such as RL and in-context learning—LoT provides a scalable curriculum for improving reasoning in small to mid scale LLMs through supervised fine-tuning.

## 6 Conclusion

We introduced Ladders-of-Thought (LoT), a framework that combines progressive rewrites with an adaptive self-evolving curriculum to improve reasoning in small- to mid-scale LLMs. Our experiments on math and multi-hop reasoning demonstrate that LoT consistently outperforms strong knowledge distillation and curriculum baselines, delivering substantial gains in out-of-distribution arithmetic tasks (e.g., +32 percentage points on AddSub, +25pp on SVAMP), modest but robust improvements on in-domain test sets (GSM8K, EntailmentBank), and dataset-dependent benefits on multi-hop reasoning (notably +25pp on StrategyQA). LoT also accelerates convergence compared to flat or staged curricula, highlighting the value of adaptivity in balancing eficiency with final performance. These findings show that carefully structured training signals—semantically faithful rewrites organized into adaptive curricula—provide a principled recipe for strengthening reasoning in smaller LLMs without requiring more scale or data. We believe LoT ofers a practical foundation for future reasoning-focused training pipelines and can complement other emerging curriculum-based strategies.

## Limitations

Our study focuses on small- to mid-scale LLMs (1–8B parameters); scalability to larger foundation models remains untested. LoT also depends on a capable generator for progressive rewrites—low-quality or unfaithful rewrites may add noise, and the balance between fidelity and diversity is not fully explored. Evaluation is limited to English math and text-only multi-hop benchmarks; extending to multilingual, multimodal, and interactive domains (e.g., vision–language or embodied agents) is a natural next step. Finally, LoT’s mixed results on certain multi-hop tasks indicate that benefits are dataset-dependent, raising open questions about which reasoning settings gain most from progressive curricula.

## Reproducibility Statement

We have made every efort to ensure the reproducibility of our results. All datasets used in this work are publicly available. We provide details of data preprocessing, rewrite generation, and filtering rules in Appendix A.3. Model architectures (OPT and Pythia) are open-source, and all training hyperparameters, curriculum schedules, and evaluation settings are fully specified in Section 4 and Appendix A.3. We will release our training scripts, curriculum scheduler implementation, and rewrite datasets to facilitate replication and extension by the community.

## Impact Statement

This work aims to improve the reasoning capabilities of small and mid-scale language models, which can reduce energy consumption and computational cost compared to reliance on very large models, supporting more sustainable and accessible deployment. By strengthening smaller open or locally deployable models, our approach may also reduce dependence on massive proprietary systems, broadening access to advanced reasoning capabilities. At the same time, improved reasoning ability could be misused in harmful domains (e.g., facilitating more efective planning or deception), underscoring the importance of responsible deployment and complementary safety measures.

## References

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raf, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.

Hyungjoo Chae, Yongho Song, Kai Tzu-iunn Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo. Dialogue chain-of-thought distillation for commonsense-aware conversational agents. arXiv preprint arXiv:2310.09343, 2023.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. Explaining answers with entailment trees. arXiv preprint arXiv:2104.08661, 2021.

Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. arXiv preprint arXiv:2311.01460, 2023.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning, pages 10421–10430. PMLR, 2023.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. Transactions of the Association for Computational Linguistics (TACL), 2021.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543, 2023.

Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. arXiv preprint arXiv:2212.10071, 2022.

Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 523–533, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1058. https://aclanthology.org/D14-1058/.

Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, and Alexander Hauptmann. Self-paced curriculum learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. The impact of reasoning step length on large language models. arXiv preprint arXiv:2401.04925, 2024.

Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, and Sung Ju Hwang. Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks. Advances in Neural Information Processing Systems, 36:48573–48602, 2023.

Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. Qasc: A dataset for question answering via sentence composition. arXiv:1910.11473v2, 2020.

Yajing Kong, Liu Liu, Jun Wang, and Dacheng Tao. Adaptive curriculum learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5067–5076, 2021.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5315–5333, 2023.

Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, and Radha Poovendran. Small models struggle to learn from strong reasoners. arXiv preprint arXiv:2502.12143, 2025.

Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. Tinygsm: achieving> 80% on gsm8k with small language models. arXiv preprint arXiv:2312.09241, 2023.

Yinpeng Liu, Jiawei Liu, Xiang Shi, Qikai Cheng, Yong Huang, and Wei Lu. Let’s learn step by step: Enhancing in-context learning ability with curriculum learning. arXiv preprint arXiv:2402.10738, 2024.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Teaching small language models to reason. arXiv preprint arXiv:2212.08410, 2022.

Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher–student curriculum learning. IEEE transactions on neural networks and learning systems, 31(9):3732–3740, 2019.

Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing English math word problem solvers. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 975–984, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.92. https://aclanthology.org/2020.acl-main.92/.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al. Orca 2: Teaching small language models how to reason. arXiv preprint arXiv:2311.11045, 2023.

Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E Taylor, and Peter Stone. Curriculum learning for reinforcement learning domains: A framework and survey. Journal of Machine Learning Research, 21(181):1–50, 2020.

Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, et al. Curriculum reinforcement learning from easy to hard tasks improves llm reasoning. arXiv preprint arXiv:2506.06632, 2025.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191, 2021.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5687–5711, 2023.

Subhro Roy and Dan Roth. Solving general arithmetic word problems. In Lluís Màrquez, Chris Callison-Burch, and Jian Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1743– 1752, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1202. https://aclanthology.org/D15-1202/.

Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao. Eficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520, 2025.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Toby Simonds and Akira Yoshiyama. Ladder: Self-improving llms through recursive problem decomposition. arXiv preprint arXiv:2503.00735, 2025.

Petru Soviany, Radu Tudor Ionescu, Paolo Rota, and Nicu Sebe. Curriculum learning: A survey. International Journal of Computer Vision, 130(6):1526–1565, 2022.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on machine learning research, 2023.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.

Peifeng Wang, Zhengyang Wang, Zheng Li, Yifan Gao, Bing Yin, and Xiang Ren. Scott: Self-consistent chain-of-thought distillation. arXiv preprint arXiv:2305.01879, 2023a.

Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, and Zhifang Sui. Making large language models better reasoners with alignment. arXiv preprint arXiv:2309.02144, 2023b.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-eficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Peter West, Chandra Bhagavatula, Jack Hessel, Jena D Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. Symbolic knowledge distillation: from general language models to commonsense models. arXiv preprint arXiv:2110.07178, 2021.

Zhuofeng Wu, Richard He Bai, Aonan Zhang, Jiatao Gu, VG Vinod Vydiswaran, Navdeep Jaitly, and Yizhe Zhang. Divide-or-conquer? which part should you distill your llm? In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 2572–2585, 2024.

Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. A survey on knowledge distillation of large language models. arXiv preprint arXiv:2402.13116, 2024.

Chuanpeng Yang, Yao Zhu, Wang Lu, Yidong Wang, Qian Chen, Chenlong Gao, Bingjie Yan, and Yiqiang Chen. Survey on knowledge distillation for large language models: methods, evaluation, and application. ACM Transactions on Intelligent Systems and Technology, 2024.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Tom Grifiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022a.

Yifan Zhang, Jingqin Yang, Yang Yuan, and Andrew Chi-Chih Yao. Cumulative reasoning with large language models. arXiv preprint arXiv:2308.04371, 2023.

Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, and Lu Wang. Small language models need strong verifiers to self-correct reasoning. arXiv preprint arXiv:2404.17140, 2024.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022b.

Qihuang Zhong, Kang Wang, Ziyang Xu, Juhua Liu, Liang Ding, and Bo Du. Achieving> 97% on gsm8k: Deeply understanding the problems makes llms better solvers for math word problems. arXiv preprint arXiv:2404.14963, 2024.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022.

## Contents

1 Introduction 2   
2 Background 2   
3 Method: Ladders-of-Thought (LoT) 3   
3.1 Progressive Rewrites 3   
3.2 Dificulty Labeling via Step Definition 4   
3.3 Curriculum Construction 4   
4 Experiments 5   
4.1 Setup 5   
4.2 Main Results 6   
4.3 Larger Students: Qwen2.5–7B and Llama3.1–8B 6   
4.4 Ablation: Rewrite Depth . 6   
4.5 Ablation: Curriculum Scheduling 8   
4.6 Overhead of LoT 8   
4.7 Discussion . 8   
5 Related Works 9   
6 Conclusion 10   
A Appendix 17   
A.1 Algorithms 17   
A.2 Dataset Statistics . 17   
A.3 Experimental Setup Details 19   
A.3.1 Environment Details 19   
A.3.2 Hyperparameters . 19   
A.3.3 Bucketing by Step Count 20   
A.3.4 Answer Verification Procedure 20   
A.3.5 Evaluation Setup . 20   
A.4 Additional Experiment Results 21   
A.5 Rewrite Depth Shifts the Dificulty Distribution. 22   
A.6 Ablation: Empirical Dificulty vs. Step-Based Dificulty. 23   
A.7 Ablation: Efect of Rewriter Model Scale . 25   
A.8 Progressive Rewrite Prompts 26   
A.8.1 EntailmentBank Prompt . 26   
A.8.2 GSM8K Prompt 27

## A Appendix

## A.1 Algorithms

Algorithm 1 describes the overall Ladders-of-Thought (LoT) training procedure, which combines progressive problem rewriting with an adaptive curriculum to fine-tune a student model. Each training example is rewritten into a sequence of simpler variants that preserve the original solution and are grouped into dificulty-based buckets, from which mini-batches are sampled during training. The student is optimized with a chain-ofthought loss, while periodic balanced evaluation provides feedback to a self-evolving scheduler (Algorithm 2) that dynamically adjusts the curriculum over time.

Algorithm 2 presents the Self-Evolving Curriculum Scheduler, a non-stationary multi-armed bandit (MAB) strategy that dynamically allocates training batches across curriculum buckets of increasing dificulty. The scheduler maintains Q-values for each bucket, reflecting recent improvements in model accuracy, and uses these values to guide bucket selection according to either a Boltzmann exploration policy or an ϵ-greedy policy. Periodically, the algorithm evaluates the model on a balanced validation set, computes the reward as the gain over a running accuracy baseline, and updates both the Q-values (via temporal diference learning) and the baselines (via exponential moving average). This design allows the scheduler to adaptively focus training on buckets that yield the greatest learning progress while still preserving exploration.

Algorithm 3 defines the auxiliary procedure ValidateBalanced, which ensures fair assessment of performance across curriculum buckets. The method constructs a validation set that samples an equal number of items from each bucket, evaluates the model independently on each subset, and returns per-bucket accuracies. These balanced evaluations are used by the scheduler (Algorithm 2) to compute bucket-wise rewards and update the learning signals that drive curriculum adaptation.

Algorithm 1 Ladders-of-Thought (LoT): Training with Progressive Rewrites and Self-evolving Curriculum   
1: Input: Original data $\mathcal { D } _ { \mathrm { o r i g } } ,$ teacher T, rewriting model R, student $S _ { \theta }$ , budget S steps, buckets $\{ B _ { k } \}$   
2: Output: Fine-tuned student $S _ { \theta }$   
3: Progressive Rewriting.   
4: for each $( x , y ) \in \mathcal { D } _ { \mathrm { o r i g } }$ do   
5: Generate rationale r and answer y from $T$   
6: Iteratively rewrite x with R into $\bar { \boldsymbol { x } } ^ { ( d ) }$ such that $\phi ( x ^ { ( d + 1 ) } ) = \phi ( x ^ { ( d ) } ) - 1$   
7: Collect $( \tilde { x } ^ { ( d ) } , r ^ { ( d ) } , y ^ { ( d ) } , \phi ( x ^ { ( d ) } ) )$ until trivial   
8: end for   
9: Bucketization.   
10: Group examples by step count ϕ(x) into buckets $\{ \boldsymbol { B } _ { k } \}$   
11: Training with Bandit Curriculum.   
12: Initialize bandit over buckets   
13: for t = 1 to S do   
14: Sample batch $\mathcal { M } _ { t }$ according to bandit distribution   
15: Update $S _ { \theta }$ with CoT loss on $\mathcal { M } _ { t }$ (rationale $^ +$ answer tokens)   
16: if t mod $E = 0$ then   
17: Evaluate on held-out validation splits $B _ { k } ^ { \mathrm { v a l } }$   
18: Compute rewards and update bandit sampling probabilities   
19: end if   
20: end for   
21: return $S _ { \theta }$

## A.2 Dataset Statistics

For all experiments, we use OpenAI GPT-5-mini as both the rewriter and teacher model to generate the progressive rewrite curricula. Rewrite generation is performed entirely ofline. Table 5 reports the corresponding input/output token counts and cost estimates.

```csv
Algorithm 3 ValidateBalanced (helper)
1: Require: Buckets $\mathcal { C } ;$ validation sampler that draws an equal number of items per bucket
2: Build validation set $\textstyle { \mathcal { V } } = \bigcup _ { c \in { \mathcal { C } } } { \mathcal { V } } ( c )$ with $| \nu ( c ) |$ equal across buckets
3: for each $c \in { \mathcal { C } }$ do
4: Evaluate current model on $\mathcal { V } ( c )$ to obtain accuracy $\operatorname { A c c } _ { t } ( c )$
5: end for
6: return $\{ \operatorname { A c c } _ { t } ( c ) \} _ { c \in { \mathcal { C } } }$
Dataset # Train Qs Input Tokens Output Tokens GPT-5-Mini Batch API Cost<sup>2</sup>
GSM8K 7,473 5,200,915 14,550,446 $15.20
EntailmentBank 1,836 2,127,678 6,518,870 $6.78
```

Algorithm 2 Self-Evolving Curriculum Scheduler (Non-stationary MAB)   
1: Require: Buckets $\mathcal { C } = \{ c _ { 1 } , \ldots , c _ { N } \}$ (ordered by dificulty); learning rate $\alpha \in ( 0 , 1 ] ;$ EMA coeficient   
$\beta \in ( 0 , 1 ] ;$ validation period m (steps); policy policy $\in$ {boltzmann, epsilon\_greedy}; temperature   
$\tau > 0$ (Boltzmann); exploration rate $\epsilon \in [ 0 , 1 ]$ (ϵ-greedy)   
2: Initialize: $Q _ { 0 } ( c ) \gets 0$ and $\overline { { \mathrm { A c c } } } _ { 0 } ( c ) \gets 0$ for all $c \in { \mathcal { C } }$   
3: Initialize: step counter $t \gets 0$   
4: while training not converged do   
5: $t \gets t + 1$   
6: $\nearrow$ Select a bucket (action) $^ * /$   
7: if policy = boltzmann then   
8: π<sub>t</sub>(c) ∝ exp $\left( Q _ { t - 1 } ( c ) / \tau \right)$ (normalize over $c \in { \mathcal { C } } )$   
9: Sample $c _ { t } \sim \pi _ { t } ( \cdot )$   
10: else if policy = epsilon\_greedy then   
11: With prob. $1 - \epsilon \colon c _ { t } \gets \arg \operatorname* { m a x } _ { c \in \mathcal { C } } Q _ { t - 1 } ( c )$ ; else sample $c _ { t }$ uniformly from $\mathcal { C }$   
12: end if   
13: TrainStep on a mini-batch from bucket $c _ { t }$ (one or more gradient updates)   
14: $/ ^ { * } P e$ riodic validation and updates $^ * /$   
15: if t mod m = 0 then   
16: $\{ \operatorname { A c c } _ { t } ( c ) \} _ { c \in { \mathcal { C } } }  \operatorname { V a L I D A T E B A L A N C E D } ( { \mathcal { C } } )$   
17: for each $c \in { \mathcal { C } }$ do   
18: $r _ { t } ( c ) \gets \mathrm { A c c } _ { t } ( c ) - \overline { { \mathrm { A c c } } } _ { t - 1 } ( c )$ (improvement over running baseline)   
19: $Q _ { t } ( c )  \alpha \cdot r _ { t } ( c ) + \underline { { { ( 1 - \alpha ) } } } \cdot Q _ { t - 1 } ( c ) \ ( T D ( O )$ on non-stationary reward)   
20: ${ \overline { { \operatorname { A c c } } } } _ { t } ( c ) \gets ( 1 - \beta ) \cdot { \overline { { \operatorname { A c c } } } } _ { t - 1 } ( c ) + \beta \cdot \operatorname { A c c } _ { t } ( c )$ (EMA baseline)   
21: end for   
22: else   
23: for each $c \in { \mathcal { C } }$ do   
24: $Q _ { t } ( c )  Q _ { t - 1 } ( c ) ; \quad \overline { { \mathrm { A c c } } } _ { t } ( c )  \overline { { \mathrm { A c c } } } _ { t - 1 } ( c )$   
25: end for   
26: end if   
27: end while  
Table 5 Rewrite-generation overhead. All rewrites are generated once ofline. Costs estimated using the GPT-5-mini batch API.

To ensure that rewritten questions preserve semantic fidelity and form a coherent dificulty ladder, we conducted a 500-sample rewrite quality audit evaluated by GPT-5. Each rewrite was assessed along three criteria:

• Question validity: whether the rewritten question is clear, solvable, and self-contained.

• Difficulty decrease: whether the rewrite is strictly easier than the original.

• Answer preservation: whether solving the rewritten question yields the same answer.

Across the 500 sampled rewrites, we find that:

• 99.2% were valid and solvable,

• 98.0% exhibited a clear decrease in dificulty, and

• 98.6% preserved the original answer.

These results confirm that progressive rewrites reliably maintain semantic identity while producing wellcontrolled dificulty reductions.

Table 6 provides step-count distributions for all dataset splits used in training, rewriting, validation, and testing.
<table><tr><td>Dataset</td><td>Split</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6-7</td><td>8-15</td></tr><tr><td rowspan="3">GSM8K</td><td>Train (all)</td><td>7320</td><td>7352</td><td>6920</td><td>5063</td><td>2989</td><td>1601</td><td>1100</td><td>577</td></tr><tr><td>Validation</td><td>100</td><td>96</td><td>87</td><td>90</td><td>92</td><td>90</td><td>74</td><td>10</td></tr><tr><td>Test</td><td>-</td><td>-</td><td>一</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">EntailmentBank</td><td>Train (all)</td><td>1660</td><td>1642</td><td>1226</td><td>745</td><td>421</td><td>258</td><td>268</td><td>134</td></tr><tr><td>Validation</td><td>100</td><td>99</td><td>94</td><td>96</td><td>95</td><td>59</td><td>46</td><td>16</td></tr><tr><td>Test</td><td>1</td><td>30</td><td>29</td><td>14</td><td>12</td><td>8</td><td>3</td><td>3</td></tr></table>

Table 6 Step-count distributions for all splits of GSM8K and EntailmentBank. GSM8K test data lacks step annotations (shown as “–”).

## A.3 Experimental Setup Details

## A.3.1 Environment Details

All experiments were conducted on cluster nodes equipped with 4 NVIDIA RTX A6000 GPUs (48GB VRAM each), 4 CPU cores, and 64GB of host memory.

Training. Models were fine-tuned using PyTorch, with the trl and accelerate libraries handling supervised fine-tuning and multi-GPU execution.

Evaluation. Performance was assessed using the lm-eval-harness, with our multi-stage answer verification pipeline integrated into the evaluation loop (see Section A.3.4.

## A.3.2 Hyperparameters

We list below the key hyperparameters fed to the HuggingFace TRL trainer. Unless otherwise noted, all other hyperparameters follow library defaults.

"weight\_decay": 0.05,

"warmup\_ratio": 0.1,

"lr\_scheduler\_type": "constant",

Validation is performed 100 times during training. The interval is set to 50 steps for math reasoning and 10 steps for multi-hop reasoning. In validation the generation arguments are set to:

## A.3.3 Bucketing by Step Count

To reduce variance and maintain balanced sampling, we group questions by their reasoning step count into buckets. Specifically, questions with shorter derivations (0, 1, 2, or 3 steps) are each assigned their own bucket, while questions requiring four or more steps are merged into a single "4+" bucket. This grouping scheme has two advantages: (i) it preserves granularity for very short reasoning chains, which difer substantially in dificulty, and (ii) it avoids fragmentation of the long-tail distribution of high-step examples, which are sparse and uneven across datasets. All curriculum schedules and sampling strategies described in the main text are applied over these step-count buckets.

## A.3.4 Answer Verification Procedure

Evaluating free-form reasoning outputs requires robust answer verification, as model predictions may vary in surface form while being semantically correct. Our validation loop employs a four-stage verification process:

1. Exact Match. We first check whether the predicted answer string exactly matches the ground-truth string after normalization (e.g., case-folding and whitespace trimming).

2. Containment. If exact match fails, we check whether the normalized gold answer appears as a substring within the model output. This captures predictions where the answer is embedded in additional text.

3. Token-level F1. We compute token-level precision, recall, and F1 between the predicted output and the gold answer. Predictions are accepted if the F1 score ≥ 0.90, ensuring high lexical overlap even under paraphrasing.

4. Semantic Similarity. Finally, we compute cosine similarity between SBERT embeddings of the predicted answer and the gold answer. Predictions are marked correct if the similarity score ≥ 0.8.

A prediction is considered correct if it satisfies any of the four criteria. This layered procedure provides robustness to surface-level variation while enforcing semantic fidelity to the ground-truth answer.

For arithmetic datasets, we additionally use the math\_verify library to parse, simplify, and compare numeric expressions. This ensures that mathematically equivalent forms (e.g ., “ <sup>3</sup> ” vs. “1.5”) are treated as correct, even if their textual forms difer.

Finally, in our evaluation experiments (Section 4), when testing trained models on external benchmarks via the LM Evaluation Harness (Srivastava et al., 2023), we adapt the same four-stage verification method (including thresholds and math\_verify) to ensure consistency across training validation and benchmark evaluation.

## A.3.5 Evaluation Setup

All evaluations are conducted using the LM Evaluation Harness (Srivastava et al., 2023). To ensure consistency with our training validation, we adapt the same four-stage answer verification procedure (Section A.3.4), including thresholds for token-level F1 and semantic similarity, as well as the use of math\_verify for numeric equivalence checking.

Multiple-choice tasks. For benchmarks originally framed as multiple-choice question answering (e.g., StrategyQA, QASC), we convert them into free-form generation tasks. Specifically, we discard option letters and use the text of the correct option as the gold answer. Model outputs are then evaluated against these free-form answers using the verification pipeline.

Contextual tasks. For tasks that provide long passages as context (e.g., MuSiQue, OpenBookQA), we extract only the sentences marked as relevant by dataset annotations and provide these as the model’s context. This reduces context length while preserving all information necessary to answer the question.

Metrics. We report pass@5 accuracy, where a prediction is considered correct if any of the top-5 generated candidates passes verification. All reported results include the standard error (stderr) across evaluation runs.

## A.4 Additional Experiment Results

Table 7 and table 8 show the accuracies and standard error of the math reasoning and multi-hop reasoning benchmarks.
<table><tr><td>Methods</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>Multi-Arith</td><td>SVAMP</td></tr><tr><td colspan="6">OPT-1.3B</td></tr><tr><td>Base</td><td>3.79±0.53</td><td>1.83±1.29</td><td> $4 . 0 5 { \pm } 0 . 7 9$ </td><td> $2 . 2 2 { \pm } 1 . 1 0 $ </td><td> $5 . 6 9 { \pm } 1 . 3 4 $ </td></tr><tr><td>CoT KD</td><td>27.75±1.23</td><td> $9 . 1 7 { \pm } 2 . 7 8$ </td><td> $2 0 . 2 3 { \pm } 1 . 6 2$ </td><td> $6 3 . 3 3 { \pm } 3 . 6 0$ </td><td> $2 0 . 7 4 { \pm } 2 . 3 5$ </td></tr><tr><td>LoT (Ours)</td><td>31.16±1.28</td><td> $3 3 . 0 3 { \pm } 4 . 5 3 $ </td><td> $4 1 . 7 5 { \pm } 1 . 9 9 $ </td><td> $6 8 . 3 3 { \pm } 3 . 4 8 $ </td><td> $3 8 . 4 6 { \pm } 2 . 8 2 $ </td></tr><tr><td colspan="6">OPT-2.7B</td></tr><tr><td>Base</td><td>3.34±0.49</td><td> $1 . 8 3 { \pm } 1 . 2 9 $ </td><td> $4 . 2 1 { \pm } 0 . 8 1 $ </td><td> $3 . 3 3 { \pm } 1 . 3 4 $ </td><td> $6 . 6 9 { \pm } 1 . 4 5$ </td></tr><tr><td>CoT KD</td><td> $3 1 . 0 1 { \pm } 1 . 2 7 $ </td><td> $8 . 2 6 { \pm } 2 . 6 5$ </td><td> $2 6 . 3 8 { \pm } 1 . 7 7$ </td><td> $7 1 . 1 1 { \pm } 3 . 3 9$ </td><td> $1 9 . 0 6 { \pm 2 . 2 8 }$ </td></tr><tr><td>LoT (Ours)</td><td> $3 3 . 9 7 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $4 0 . 3 7 { \pm } 4 . 7 2$ </td><td> $4 5 . 9 5 { \pm } 2 . 0 1 $ </td><td> $8 0 . 5 6 { \pm } 2 . 9 6 $ </td><td> $4 4 . 1 5 { \pm } 2 . 8 8 $ </td></tr><tr><td colspan="6">Pythia-1.4B</td></tr><tr><td>Base</td><td> $2 . 9 6 { \pm } 0 . 4 7$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $4 . 8 5 { \pm } 0 . 8 7$ </td><td> $1 . 6 7 { \pm } 0 . 9 6 $ </td><td> $8 . 0 3 { \pm } 1 . 5 7$ </td></tr><tr><td>CoT KD</td><td> $2 6 . 0 0 { \pm } 1 . 2 1 $ </td><td> $3 . 6 7 { \pm } 1 . 8 1 $ </td><td> $2 1 . 3 6 { \pm } 1 . 6 5$ </td><td> $5 9 . 4 4 \pm 3 . 6 7$ </td><td> $1 9 . 7 3 { \pm } 2 . 3 1 $ </td></tr><tr><td>LoT (Ours)</td><td> $2 8 . 3 5 { \pm } 1 . 2 4 $ </td><td> $2 4 . 7 7 { \scriptstyle \pm 4 . 1 5 }$ </td><td> $4 0 . 7 8 { \pm } 1 . 9 8$ </td><td> $6 3 . 3 3 { \pm } 3 . 6 0$ </td><td> $3 8 . 4 6 { \pm } 2 . 8 2 $ </td></tr><tr><td colspan="6">Pythia-2.8B</td></tr><tr><td>Base</td><td> $3 . 7 1 { \pm } 0 . 5 2 $ </td><td> $1 . 8 3 { \pm } 1 . 2 9 $ </td><td> $6 . 6 3 { \pm } 1 . 0 0 \ $ </td><td> $4 . 4 4 { \pm } 1 . 5 4 $ </td><td> $9 . 7 0 { \pm } 1 . 7 1 $ </td></tr><tr><td>CoT KD</td><td> $3 3 . 4 3 { \pm } 1 . 3 0 $ </td><td> $1 1 . 9 3 { \pm } 3 . 1 2 $ </td><td> $3 1 . 8 8 { \pm } 1 . 8 8 $ </td><td> $7 4 . 4 4 { \pm } 3 . 2 6 $ </td><td> $2 4 . 0 8 { \pm } 2 . 4 8 $ </td></tr><tr><td>LoT (Ours)</td><td> $3 2 . 9 8 { \pm } 1 . 2 9 $ </td><td> $3 2 . 1 1 { \pm } 4 . 4 9$ </td><td> $4 6 . 7 6 { \pm } 2 . 0 1$ </td><td> $7 2 . 7 8 { \pm } 3 . 3 3$ </td><td> $4 4 . 8 2 { \pm } 2 . 8 8 $ </td></tr></table>

Table 7 Pass@5 mean accuracy (%) and standard error on GSM8K test split and four math reasoning benchmarks.

<table><tr><td>Methods</td><td>EntailmentBank</td><td>QASC</td><td>OpenBookQA StrategyQA</td><td></td><td>MuSiQue</td></tr><tr><td colspan="6">OPT-1.3B</td></tr><tr><td>Base</td><td> $2 2 . 0 { \pm } 4 . 1 6$ </td><td> $1 7 . 2 { \pm } 1 . 6 9 $ </td><td> $1 4 . 8 { \pm } 1 . 5 9 \ $ </td><td> $3 2 . 4 { \pm } 2 . 1 0 $ </td><td> $2 . 2 0 { \pm } 0 . 6 6$ </td></tr><tr><td>CoT KD</td><td> $3 6 . 0 { \pm } 4 . 8 2 \ $ </td><td> $4 6 . 0 { \pm } 2 . 2 3 $ </td><td> $4 7 . 4 { \pm } 2 . 2 4 $ </td><td> $2 2 . 6 { \pm } 1 . 8 7$ </td><td> $1 4 . 2 { \pm } 1 . 5 6 $ </td></tr><tr><td>LoT (Ours)</td><td> $4 1 . 0 { \pm } 4 . 9 4 \ $ </td><td> $5 2 . 8 { \pm } 2 . 2 3 $ </td><td> $4 3 . 6 { \pm } 2 . 2 2 $ </td><td> $4 7 . 8 { \pm } 2 . 2 4 $ </td><td> $3 9 . 2 { \pm } 2 . 1 9 $ </td></tr><tr><td colspan="6">OPT-2.7B</td></tr><tr><td>Base</td><td> $2 1 . 0 { \pm } 4 . 0 9 \ $ </td><td> $2 8 . 8 { \pm } 2 . 0 3 $ </td><td> $1 2 . 6 { \pm } 1 . 4 9$ </td><td> $3 3 . 6 { \pm } 2 . 1 1 $ </td><td> $2 . 2 0 { \pm } 0 . 6 6$ </td></tr><tr><td>CoT KD</td><td> $4 0 . 0 { \pm } 4 . 9 2 \ $ </td><td> $5 6 . 2 { \pm } 2 . 2 2 $ </td><td> $4 6 . 0 { \pm } 2 . 2 3 $ </td><td> $5 4 . 0 { \pm } 2 . 2 3 $ </td><td> $4 3 . 6 { \pm } 2 . 2 2 $ </td></tr><tr><td>LoT (Ours)</td><td> $4 1 . 0 { \pm } 4 . 9 4 \ $ </td><td> $6 0 . 4 { \pm } 2 . 1 9 $ </td><td> $4 7 . 8 { \pm } 2 . 2 4 $ </td><td> $5 1 . 2 { \pm } 2 . 2 4 $ </td><td> $2 4 . 0 { \pm } 2 . 1 2 \ $ </td></tr><tr><td colspan="6">Pythia-1.4B</td></tr><tr><td>Base</td><td> $1 3 . 0 { \pm } 3 . 3 8 $ </td><td> $4 5 . 0 { \pm } 2 . 2 3 $ </td><td> $3 4 . 2 { \pm } 2 . 1 2 $ </td><td> $2 4 . 4 { \pm } 1 . 9 2 $ </td><td> $1 2 . 0 { \pm } 1 . 4 5 $ </td></tr><tr><td>CoT KD</td><td> $3 6 . 0 { \pm } 4 . 8 2 \ $ </td><td> $5 5 . 2 { \pm } 2 . 2 3 $ </td><td> $4 4 . 6 { \pm } 2 . 2 3 $ </td><td> $3 6 . 4 { \pm } 2 . 1 5 $ </td><td> $4 2 . 0 { \pm } 2 . 2 1 $ </td></tr><tr><td>LoT (Ours)</td><td> $3 9 . 0 { \pm } 4 . 9 0 \ $ </td><td> $5 6 . 6 { \pm } 2 . 2 2 $ </td><td> $3 6 . 8 { \pm } 2 . 1 6 $ </td><td> $5 3 . 2 { \pm } 2 . 2 3 $ </td><td> $3 9 . 2 { \pm } 2 . 1 9 $ </td></tr><tr><td colspan="6">Pythia-2.8B</td></tr><tr><td>Base</td><td> $1 0 . 0 { \pm } 3 . 0 2 $ </td><td> $3 9 . 0 { \pm } 2 . 1 8 $ </td><td> $3 2 . 4 { \pm } 2 . 1 0 $ </td><td> $2 9 . 2 { \pm } 2 . 0 4 $ </td><td> $1 8 . 8 { \pm } 1 . 7 5 $ </td></tr><tr><td>CoT KD</td><td> $3 2 . 0 { \pm } 4 . 6 9$ </td><td> $3 2 . 2 { \pm } 2 . 0 9 $ </td><td> $2 8 . 2 { \pm } 2 . 0 1 $ </td><td> $5 5 . 6 { \pm } 2 . 2 2 $ </td><td> $4 2 . 2 { \pm } 2 . 2 1 $ </td></tr><tr><td>LoT (Ours)</td><td> $4 0 . 0 { \pm } 4 . 9 2 \ $ </td><td> $4 8 . 8 { \pm } 2 . 2 4 $ </td><td> $3 7 . 8 { \pm } 2 . 1 7 $ </td><td> $6 0 . 0 { \pm } 2 . 1 9 \ $ </td><td> $4 5 . 0 { \pm } 2 . 2 3 $ </td></tr></table>

Table 8 Pass@5 mean accuracy (%) and standard error on EntailmentBank test split and four multi-hop reasoning benchmarks.  
Table 9 and table 10 show the pass@1 accuracies and standard errors of the math reasoning and multi-hop

reasoning benchmarks with model using greedy decoding.
<table><tr><td>Methods</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td></tr><tr><td colspan="6">OPT-1.3B</td></tr><tr><td>Base</td><td> $1 . 1 4 { \pm } 0 . 2 9$ </td><td> $0 . 9 2 { \pm } 0 . 9 2$ </td><td> $0 . 8 1 { \pm } 0 . 3 6 $ </td><td> $0 . 5 6 { \pm } 0 . 5 6 $ </td><td> $2 . 3 4 { \pm } 0 . 8 8$ </td></tr><tr><td>CoT KD</td><td> $1 6 . 7 6 { \pm } 1 . 0 3 $ </td><td> $6 . 4 2 { \pm } 2 . 3 6$ </td><td> $1 2 . 6 2 { \pm } 1 . 3 4 $ </td><td> $4 6 . 6 7 { \scriptstyle \pm 3 . 7 3 }$ </td><td> $1 0 . 7 0 { \pm } 1 . 7 9 $ </td></tr><tr><td>LoT (Ours)</td><td> $1 7 . 3 6 { \pm } 1 . 0 4 $ </td><td> $2 4 . 7 7 { \pm } 4 . 1 5$ </td><td>26.86±1.78</td><td> $4 7 . 2 2 { \pm } 3 . 7 3 $ </td><td> $2 1 . 4 0 { \pm } 2 . 3 8 $ </td></tr><tr><td colspan="6">OPT-2.7B</td></tr><tr><td>Base</td><td> $1 . 1 4 { \pm } 0 . 2 9$ </td><td>0.92±0.92</td><td>1.94±0.56</td><td> $0 . 5 6 { \pm } 0 . 5 6 $ </td><td> $3 . 0 1 { \pm } 0 . 9 9$ </td></tr><tr><td>CoT KD</td><td> $1 9 . 1 8 { \pm } 1 . 0 8 $ </td><td>4.59±2.01</td><td>15.70±1.46</td><td> $5 2 . 2 2 { \pm } 3 . 7 3$ </td><td> $9 . 0 3 { \pm } 1 . 6 6 $ </td></tr><tr><td>LoT (Ours)</td><td> $2 1 . 8 3 { \pm } 1 . 1 4 $ </td><td> $2 9 . 3 6 { \pm } 4 . 3 8 $ </td><td>34.79±1.92</td><td> $5 3 . 8 9 { \pm } 3 . 7 3 $ </td><td> $3 2 . 7 8 { \scriptstyle \pm 2 . 7 2 }$ </td></tr><tr><td colspan="6">Pythia-1.4B</td></tr><tr><td>Base</td><td> $1 . 5 9 { \pm } 0 . 3 4 $ </td><td> $0 . 9 2 { \pm } 0 . 9 2$ </td><td> $1 . 2 9 { \pm } 0 . 4 6 $ </td><td> $1 . 6 7 { \pm } 0 . 9 6 $ </td><td> $1 . 3 4 { \pm } 0 . 6 7$ </td></tr><tr><td>CoT KD</td><td> $1 3 . 2 7 { \pm } 0 . 9 3 $ </td><td> $3 . 6 7 { \pm } 1 . 8 1 $ </td><td> $1 1 . 8 1 { \pm } 1 . 3 0 $ </td><td> $3 7 . 2 2 { \pm } 3 . 6 1$ </td><td> $1 0 . 0 3 { \pm } 1 . 7 4 $ </td></tr><tr><td>LoT (Ours)</td><td> $1 7 . 0 6 { \pm } 1 . 0 4 $ </td><td> $1 8 . 3 5 { \pm } 3 . 7 2 $ </td><td> $2 8 . 1 6 { \pm } 1 . 8 1 $ </td><td> $4 1 . 1 1 \pm 3 . 6 8$ </td><td> $2 7 . 7 6 { \pm } 2 . 5 9$ </td></tr><tr><td colspan="6">Pythia-2.8B</td></tr><tr><td>Base</td><td> $1 . 5 2 { \pm } 0 . 3 4$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $1 . 7 8 { \pm } 0 . 5 3 $ </td><td> $1 . 6 7 { \pm } 0 . 9 6 $ </td><td> $1 . 6 7 { \pm } 0 . 7 4$ </td></tr><tr><td>CoT KD</td><td> $1 9 . 7 9 { \pm } 1 . 1 0 $ </td><td> $2 . 7 5 { \pm } 1 . 5 7$ </td><td> $1 9 . 5 8 { \pm } 1 . 6 0 $ </td><td> $5 2 . 2 2 { \pm } 3 . 7 3$ </td><td> $1 1 . 3 7 { \pm } 1 . 8 4$ </td></tr><tr><td>LoT (Ours)</td><td> $2 0 . 7 0 { \pm } 1 . 1 2 $ </td><td> $2 3 . 8 5 { \pm } 4 . 1 0 $ </td><td> $3 1 . 5 5 { \pm } 1 . 8 7$ </td><td> $5 1 . 6 7 { \pm } 3 . 7 4 $ </td><td> $3 0 . 1 0 { \pm } 2 . 6 6 $ </td></tr></table>

Table 9 Pass@1 mean accuracy (%) and standard error on GSM8K test split and four math reasoning benchmarks using greedy decoding.

<table><tr><td>Methods</td><td>EntailmentBank</td><td>QASC</td><td>OpenBookQA</td><td>StrategyQA</td><td>MuSiQue</td></tr><tr><td colspan="6">OPT-1.3B</td></tr><tr><td>Base</td><td> $8 . 0 0 { \pm } 2 . 7 3 $ </td><td> $1 . 4 0 { \pm } 0 . 5 3 $ </td><td> $7 . 6 0 { \pm } 1 . 1 9$ </td><td> $1 3 . 6 0 { \pm } 1 . 5 3 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>CoT KD</td><td> $1 6 . 0 0 { \pm } 3 . 6 8 $ </td><td> $3 5 . 0 0 { \pm } 2 . 1 4 $ </td><td> $3 7 . 6 0 { \pm } 2 . 1 7$ </td><td> $9 . 0 0 { \pm } 1 . 2 8 \ $ </td><td> $4 . 2 0 { \pm } 0 . 9 0 \ $ </td></tr><tr><td>LoT (Ours)</td><td> $2 1 . 0 0 { \pm } 4 . 0 9$ </td><td> $3 7 . 6 0 { \pm } 2 . 1 7$ </td><td> $3 2 . 2 0 { \pm } 2 . 0 9 $ </td><td> $2 2 . 0 0 { \pm } 1 . 8 5 $ </td><td> $2 8 . 0 0 { \pm } 2 . 0 1 \ $ </td></tr><tr><td colspan="6">OPT-2.7B</td></tr><tr><td>Base</td><td> $6 . 0 0 { \pm } 2 . 3 9$ </td><td> $1 3 . 6 0 { \pm } 1 . 5 3 $ </td><td> $0 . 8 0 { \pm } 0 . 4 0 $ </td><td> $1 4 . 6 0 { \pm } 1 . 5 8 \ $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td></tr><tr><td>CoT KD</td><td> $2 2 . 0 0 { \pm } 4 . 1 6$ </td><td>41.00±2.20</td><td> $3 1 . 8 0 { \pm } 2 . 0 8 $ </td><td> $1 9 . 4 0 { \pm } 1 . 7 7 $ </td><td> $2 2 . 4 0 { \pm } 1 . 8 7$ </td></tr><tr><td>LoT (Ours)</td><td> $1 7 . 0 0 { \scriptstyle \pm 3 . 7 8 }$ </td><td>45.80±2.23</td><td>34.80±2.13</td><td> $2 0 . 6 0 { \pm } 1 . 8 1 $ </td><td> $2 1 . 2 0 { \pm } 1 . 8 3 $ </td></tr><tr><td colspan="6">Pythia-1.4B</td></tr><tr><td>Base</td><td>4.00±1.97</td><td>18.60±1.74</td><td> $1 4 . 8 0 { \pm } 1 . 5 9 \ $ </td><td> $5 . 8 0 { \pm } 1 . 0 5 $ </td><td> $1 . 6 0 { \pm } 0 . 5 6 \ ^ { }$ </td></tr><tr><td>CoT KD</td><td> $1 7 . 0 0 { \scriptstyle \pm 3 . 7 8 }$ </td><td>39.00±2.18</td><td> $3 1 . 4 0 { \pm } 2 . 0 8 $ </td><td> $1 6 . 2 0 { \pm } 1 . 6 5$ </td><td> $1 9 . 8 0 { \pm } 1 . 7 8 $ </td></tr><tr><td>LoT (Ours)</td><td> $2 3 . 0 0 { \pm } 4 . 2 3 $ </td><td>37.40±2.17</td><td> $2 4 . 0 0 { \pm } 1 . 9 1 $ </td><td> $2 4 . 4 0 { \pm } 1 . 9 2 \ $ </td><td> $2 4 . 4 0 { \pm } 1 . 9 2 \ $ </td></tr><tr><td colspan="6">Pythia-2.8B</td></tr><tr><td>Base</td><td> $2 . 0 0 { \pm } 1 . 4 1 $ </td><td> $1 7 . 0 0 { \pm } 1 . 6 8 $ </td><td> $1 1 . 2 0 { \pm } 1 . 4 1 $ </td><td> $3 . 4 0 { \pm } 0 . 8 1$ </td><td> $3 . 4 0 { \pm } 0 . 8 1$ </td></tr><tr><td>CoT KD</td><td> $1 8 . 0 0 { \scriptstyle \pm 3 . 8 6 }$ </td><td> $1 9 . 2 0 { \pm } 1 . 7 6 $ </td><td> $1 8 . 6 0 { \pm } 1 . 7 4 \ $ </td><td> $3 0 . 2 0 { \pm } 2 . 0 6$ </td><td> $1 5 . 6 0 { \pm } 1 . 6 2 $ </td></tr><tr><td>LoT (Ours)</td><td> $2 8 . 0 0 { \pm } 4 . 5 1 $ </td><td> $3 7 . 8 0 { \pm } 2 . 1 7 $ </td><td> $2 7 . 4 0 { \scriptstyle \pm 2 . 0 0 }$ </td><td> $2 8 . 2 0 { \pm } 2 . 0 1 \ $ </td><td> $2 7 . 8 0 { \pm } 2 . 0 1 $ </td></tr></table>

Table 10 Pass@1 mean accuracy (%) and standard error on EntailmentBank test split and four multi-hop reasoning benchmarks using greedy decoding.

## A.5 Rewrite Depth Shifts the Difficulty Distribution.

To better understand how rewrite depth afects the structure of the training signal, we analyze the distribution of examples by their minimal number of reasoning steps under diferent maximum rewrite depths. Table 13 and Figure 5 show that progressively adding deeper rewrites systematically increases the share of low-step (i.e., easier) questions, while reducing the long tail of high-step instances. For example, questions solvable in 0-1 steps rise from 0.1% at depth 0 to 45.1% at depth 4+. Conversely, high-step examples (6+) become increasingly rare as depth increases.

This confirms that rewrite depth does not simply add more training data, but rebalances the efective

<table><tr><td>Depth</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td><td>Average</td></tr><tr><td>0</td><td> $1 0 . 4 6 { \pm } 0 . 8 4$ </td><td> $6 . 4 2 { \pm } 2 . 3 6 $ </td><td> $9 . 2 2 { \pm } 1 . 1 6$ </td><td> $1 7 . 7 8 { \pm } 2 . 8 6 $ </td><td> $7 . 0 2 { \pm } 1 . 4 8$ </td><td>10.18</td></tr><tr><td> ${ \leq } 1$ </td><td> $3 0 . 5 5 { \pm } 1 . 2 7 $ </td><td> $2 6 . 6 1 { \pm } 4 . 2 5 $ </td><td> $4 0 . 6 1 { \pm } 1 . 9 8 $ </td><td> $6 7 . 7 8 { \pm } 3 . 4 9$ </td><td> $2 7 . 7 6 { \pm } 2 . 5 9$ </td><td>38.66</td></tr><tr><td> $\leq 2$ </td><td> $2 8 . 8 1 { \pm } 1 . 2 5 $ </td><td> $3 2 . 1 1 { \pm } 4 . 4 9$ </td><td> $4 8 . 2 2 \pm 2 . 0 1$ </td><td> $7 1 . 1 1 { \pm } 3 . 3 9$ </td><td> $4 2 . 8 1 { \pm } 2 . 8 7$ </td><td>44.61</td></tr><tr><td> ${ \le } 3$ </td><td> $3 2 . 0 7 { \pm } 1 . 2 9 $ </td><td> $3 0 . 2 8 { \pm } 4 . 4 2$ </td><td> $4 7 . 7 3 { \pm } 2 . 0 1 $ </td><td> $7 7 . 2 2 { \pm } 3 . 1 3$ </td><td> $4 3 . 1 4 { \pm } 2 . 8 7$ </td><td>46.09</td></tr><tr><td>All</td><td> $3 1 . 1 6 { \pm } 1 . 2 8 $ </td><td> $3 3 . 0 3 { \pm } 4 . 5 3 $ </td><td> $4 1 . 7 5 { \pm } 1 . 9 9 $ </td><td> $6 8 . 3 3 { \pm } 3 . 4 8 $ </td><td> $3 8 . 4 6 { \pm } 2 . 8 2 $ </td><td>42.55</td></tr></table>

Table 11 Math reasoning benchmark performance across reasoning depths. Numbers are pass@5 mean accuracy (%) with standard error.
<table><tr><td>Method</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td><td>Average</td></tr><tr><td>Random</td><td> $2 9 . 3 4 { \pm } 1 . 2 5 $ </td><td> $2 7 . 5 2 { \pm } 4 . 3 0 $ </td><td> $3 2 . 6 9 { \pm } 1 . 8 9 $ </td><td> $6 6 . 6 7 { \scriptstyle \pm 3 . 5 2 }$ </td><td> $3 1 . 7 7 { \scriptstyle \pm 2 . 7 0 }$ </td><td>37.60</td></tr><tr><td>Easy→Hard</td><td> $2 8 . 9 6 { \pm } 1 . 2 5 $ </td><td> $3 1 . 1 9 { \pm } 4 . 4 6$ </td><td> $4 2 . 3 9 { \pm } 1 . 9 9$ </td><td> $6 1 . 6 7 { \pm } 3 . 6 3$ </td><td> $4 0 . 4 7 { \pm } 2 . 8 4$ </td><td>40.94</td></tr><tr><td>Hard→Easy</td><td> $2 1 . 3 8 { \pm } 1 . 1 3$ </td><td> $5 . 5 0 { \pm } 2 . 1 9 $ </td><td>11.33±1.28</td><td> $5 6 . 6 7 { \pm } 3 . 7 0 $ </td><td> $6 . 6 9 { \pm } 1 . 4 5$ </td><td>20.31</td></tr><tr><td>Self-evolving</td><td> $3 1 . 1 6 { \pm } 1 . 2 8$ </td><td> $3 3 . 0 3 { \pm } 4 . 5 3 $ </td><td> $4 1 . 7 5 { \pm } 1 . 9 9$ </td><td> $6 8 . 3 3 { \pm } 3 . 4 8 $ </td><td> $3 8 . 4 6 { \pm } 2 . 8 2 $ </td><td>42.55</td></tr></table>

Table 12 Comparison of curriculum strategies on math reasoning benchmarks. Numbers are pass@5 mean accuracy (%) with standard error.

curriculum: shallow depths preserve a wide dificulty spectrum, while deeper depths concentrate probability mass on easier instances. Combined with Table 4, these findings suggest that moderate rewrite depth is beneficial because it increases exposure to simpler reasoning patterns without collapsing the full dificulty range.
<table><tr><td>Reasoning Steps</td><td>Depth 0</td><td>Depth 1</td><td>Depth 2</td><td>Depth 3</td><td>Depth 4+ (AII)</td></tr><tr><td>0</td><td>0 (0.0%)</td><td>2 (0.0%)</td><td>1804 (8.2%)</td><td>4235 (15.3%)</td><td>7320 (22.5%)</td></tr><tr><td>1</td><td>9 (0.1%)</td><td>1880 (12.7%)</td><td>4310 (19.5%)</td><td>6087 (22.0%)</td><td>7352 (22.6%)</td></tr><tr><td>2</td><td>1868 (25.3%)</td><td>4110 (27.9%)</td><td>5734 (26.0%)</td><td>6529 (23.6%)</td><td>6920 (21.2%)</td></tr><tr><td>3</td><td>2124 (28.8%)</td><td>3738 (25.4%)</td><td>4598 (20.8%)</td><td>4940 (17.9%)</td><td>5063 (15.6%)</td></tr><tr><td>4</td><td>1554 (21.1%)</td><td>2443 (16.6%)</td><td>2818 (12.7%)</td><td>2950 (10.7%)</td><td>2989 (9.2%)</td></tr><tr><td>5</td><td>951 (12.9%)</td><td>1371 (9.3%)</td><td>1542 (7.0%)</td><td>1592 (5.8%)</td><td>1601 (4.9%)</td></tr><tr><td>6</td><td>451 (6.1%)</td><td>654 (4.4%)</td><td>729 (3.3%)</td><td>745 (2.7%)</td><td>745 (2.3%)</td></tr><tr><td>7</td><td>241 (3.3%)</td><td>333 (2.3%)</td><td>354 (1.6%)</td><td>355 (1.3%)</td><td>355 (1.1%)</td></tr><tr><td>8</td><td>105 (1.4%)</td><td>135 (0.9%)</td><td>137 (0.6%)</td><td>137 (0.5%)</td><td>137 (0.4%)</td></tr><tr><td>9</td><td>45 (0.6%)</td><td>50 (0.3%)</td><td>52 (0.2%)</td><td>52 (0.2%)</td><td>52 (0.2%)</td></tr><tr><td>10</td><td>16 (0.2%)</td><td>17 (0.1%)</td><td>19 (0.1%)</td><td>19 (0.1%)</td><td>19 (0.1%)</td></tr><tr><td>11</td><td>3 (0.0%)</td><td>6 (0.0%)</td><td>6 (0.0%)</td><td>6 (0.0%)</td><td>6 (0.0%)</td></tr><tr><td>12</td><td>3 (0.0%)</td><td>3 (0.0%)</td><td>3 (0.0%)</td><td>3 (0.0%)</td><td>3 (0.0%)</td></tr><tr><td>13</td><td>1 (0.0%)</td><td>1 (0.0%)</td><td>1 (0.0%)</td><td>1 (0.0%)</td><td>1 (0.0%)</td></tr><tr><td>14</td><td>1 (0.0%)</td><td>2 (0.0%)</td><td>2 (0.0%)</td><td>2 (0.0%)</td><td>2 (0.0%)</td></tr><tr><td>15</td><td>1 (0.0%)</td><td>1 (0.0%)</td><td>1 (0.0%)</td><td>1 (0.0%)</td><td>1 (0.0%)</td></tr><tr><td>Total</td><td>7373 (100%)</td><td>14746 (100%)</td><td>22110 (100%)</td><td>27654 (100%)</td><td>32566 (100%)</td></tr></table>

Table 13 Distribution of minimal reasoning steps by maximum rewrite depth. Percentages are computed column-wise relative to each depth total. Deeper rewrite depths increasingly skew the distribution toward low-step (easier) questions.

## A.6 Ablation: Empirical Difficulty vs. Step-Based Difficulty.

We first examine whether the step-based dificulty measure aligns with empirical dificulty. Figure 6 plots the average model success rate against minimal step count for three models of diferent sizes. We observe a clear negative correlation between the required number of reasoning steps and empirical success rate: lower-step problems are consistently easier across models, while higher-step problems are more frequently failed. Motivated by this correlation, we construct an alternative curriculum that replaces step-based dificulty buckets with empirical dificulty buckets derived directly from the Qwen2.5-7B model.

![](images/91ab24cc8198ab998b11aef30dc273584fb29be89253767d3d277dd5105719df.jpg)  
Figure 5 Distribution of minimal reasoning steps under diferent rewrite depths.

Each training example is assigned to one of five buckets based on its observed success rate: from bucket 0 $( \mathrm { s u c c e s s } \geq 0 . 8 )$ to bucket 4 (success < 0.2). The resulting data distribution is shown in Table 14. We then train a curriculum model using the same training setup as LoT, but sampling examples from empirical buckets instead. Final evaluation across in-domain and OOD benchmarks is presented in Table 15.

<table><tr><td>Bucket</td><td>Success Range</td><td>Count (%)</td></tr><tr><td>0</td><td>≥ 0.8</td><td>13,558 (40.83%)</td></tr><tr><td>1</td><td>[0.6, 0.8)</td><td>6,668 (20.08%)</td></tr><tr><td>2</td><td>[0.4, 0.6)</td><td>5,213 (15.70%)</td></tr><tr><td>3</td><td>[0.2, 0.4)</td><td>4,179 (12.59%)</td></tr><tr><td>4</td><td>&lt; 0.2</td><td>3,587 (10.80%)</td></tr><tr><td>Total</td><td></td><td>33,205 (100%)</td></tr></table>

Table 14 Data distribution under empirical dificulty bucketing based on Qwen2.5–7B success rate estimates.
<table><tr><td>Methods</td><td>GSM8K</td><td>ASDiv</td><td>AddSub</td><td>MultiArith</td><td>SVAMP</td></tr><tr><td>Base</td><td> $2 1 . 6 1 { \pm } 1 . 1 3$ </td><td> $1 1 . 1 7 { \pm } 1 . 2 7$ </td><td> $5 . 5 0 { \pm } 2 . 1 9 $ </td><td> $1 5 . 0 0 { \scriptstyle \pm 2 . 6 7 }$ </td><td> $1 0 . 7 0 { \pm } 1 . 7 9 $ </td></tr><tr><td>CoT KD</td><td> $7 3 . 8 4 \pm 1 . 2 1 $ </td><td> $7 7 . 6 7 { \pm } 1 . 6 8 $ </td><td> $5 0 . 4 6 { \pm } 4 . 8 1$ </td><td> $9 8 . 3 3 { \pm } 0 . 9 6 \ $ </td><td> $6 0 . 5 4 { \pm } 2 . 8 3 $ </td></tr><tr><td>LoT</td><td> $\pm 1 . 6 0 \pm 1 . 2 0$ </td><td> $7 4 . 9 2 \pm 1 . 7 5$ </td><td> ${ \pm } 0 . 6 4 \pm 4 . 3 8$ </td><td> ${ \pm \mathbf { 8 . 8 9 } } { \pm \mathbf { 0 . 7 8 } }$ </td><td> ${ \bf 7 1 . 5 7 \pm 2 . 6 1 }$ </td></tr><tr><td>Success-rate LoT</td><td> $6 3 . 6 8 { \pm } 1 . 3 2 $ </td><td> $5 7 . 7 7 { \pm } 1 . 9 9$ </td><td> $6 4 . 2 2 { \pm } 4 . 6 1 $ </td><td> $5 9 . 4 4 \pm 3 . 6 7$ </td><td> $4 6 . 4 9 { \pm } 2 . 8 9$ </td></tr></table>

Table 15 Pass@5 accuracy (%) for curricula constructed using empirical dificulty bucketing.

While empirical dificulty correlates with minimal step count, the success-rate curriculum performs markedly worse than LoT. One explanation is that empirical dificulty mixes structural dificulty with surface-level distribution artifacts, leading to uneven exposure across reasoning patterns. In contrast, step-based dificulty explicitly targets inferential depth, producing a more balanced ladder of abstractions. These results suggest that while empirical hardness is partially aligned with step complexity, it is a weaker organizing principle for reasoning curricula than step-based progressive rewrites.

Min Steps vs. Average Success Rate  
![](images/01f8492d07f3950bda3fc4b2ef62691836f0a8cd6175f1db9995e5325a688bdb.jpg)  
Figure 6 Average model success rate as a function of minimal reasoning steps. Lower-step examples are generally easie across three model families.

## A.7 Ablation: Effect of Rewriter Model Scale

In this section we examine whether the scale of the rewrite model matters. We generate LoT rewrites using three Qwen2.5 Instruct models of increasing size (7B, 14B, 72B) and vary whether the rewriter is also used as the teacher model or whether the teacher is held fixed. In the rewriter-as-teacher setting, the same model provides both the rewritten training instances and the reference reasoning traces. In the fixed-teacher setting, each rewriter only produces rewritten questions, while a single teacher model (GPT-5-mini) provides the answers and reasoning traces for all variants. Table 16 summarizes the number of rewrites produced under each configuration.

<table><tr><td>Rewriter Model</td><td>Original Questions</td><td>Rewrite Questions</td><td>Train Size</td><td>Val Size</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>7373</td><td>17611</td><td>24984</td><td>635</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>7373</td><td>23028</td><td>30401</td><td>623</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>7372</td><td>22335</td><td>29707</td><td>616</td></tr><tr><td>GPT-5-mini</td><td>7373</td><td>25193</td><td>32566</td><td>639</td></tr></table>

Table 16 Dataset statistics when using Qwen2.5 models of diferent sizes to generate LoT rewrites.

We then train Qwen2.5-7B using these rewritten datasets and evaluate across in-distribution and OOD benchmarks. Results are shown in Table 17.

Our ablation results indicate that LoT is sensitive to the capability of the rewriting model. Stronger and more instruction-following rewriters (e.g., GPT-5-mini) generate more accurate and semantically consistent progressive rewrites, which leads to better downstream performance. We observe that when Qwen models are used as both rewriter and teacher, performance generally lags behind GPT-5-mini, reflecting diferences in rewrite quality.

When keeping GPT-5-mini as the teacher but using Qwen models as rewriters, only the Qwen-7B rewriter shows improvement over its rewriter-as-teacher configuration. Qwen-14B remains comparable, while Qwen-72B degrades significantly. This suggests that the rewriter has a stronger influence than the teacher on downstream performance: if the rewrites introduce semantic inconsistencies, the curriculum becomes less helpful regardless of teacher strength.

<table><tr><td>Rewriter</td><td>Teacher</td><td>GSM8K</td><td>AddSub</td><td>ASDiv</td><td>MultiArith</td><td>SVAMP</td></tr><tr><td>Qwen2.5-7B-Instr.</td><td>Qwen2.5-7B-Instr.</td><td>17.66±1.05</td><td>70.64±4.38</td><td>69.74±1.85</td><td>30.56±3.44</td><td>58.86±2.85</td></tr><tr><td>Qwen2.5-14B-Instr.</td><td>Qwen2.5-14B-Instr.</td><td>22.06±1.14</td><td>84.40±3.49</td><td>73.95±1.77</td><td>37.78±3.62</td><td>63.88±2.78</td></tr><tr><td>Qwen2.5-72B-Instr.</td><td>Qwen2.5-72B-Instr.</td><td>16.68±1.03</td><td>65.14±4.59</td><td>67.80±1.88</td><td>28.33±3.37</td><td>54.18±2.89</td></tr><tr><td>Qwen2.5-7B-Instr.</td><td>GPT-5-mini</td><td>25.70±1.20</td><td>78.90±3.93</td><td>71.04±1.83</td><td>30.56±3.44</td><td>54.85±2.88</td></tr><tr><td>Qwen2.5-14B-Instr.</td><td>GPT-5-mini</td><td>19.94±1.10</td><td>77.06±4.05</td><td>72.82±1.79</td><td>30.00±3.43</td><td>62.88±2.80</td></tr><tr><td>Qwen2.5-72B-Instr.</td><td>GPT-5-mini</td><td>11.75±0.89</td><td>31.19±4.46</td><td>34.95±1.92</td><td>16.67±2.79</td><td>23.08±2.44</td></tr><tr><td>GPT-5-mini</td><td>GPT-5-mini</td><td>74.60±1.20</td><td>70.64±4.38</td><td>74.92±1.75</td><td>98.89±0.78</td><td>71.57±2.61</td></tr></table>

Table 17 Performance of Qwen2.5-7B trained using rewrites generated by Qwen2.5 models of diferent sizes. A fixed-teacher setting (GPT-5-mini) and a rewriter-as-teacher setting are both shown.

Overall, these results show that LoT benefits from higher-quality rewriting models, but continues to outperform standard CoT distillation even when the rewriter is significantly weaker. This dependency is consistent with broader observations in model-generated data pipelines, where data quality strongly conditions downstream performance.

## A.8 Progressive Rewrite Prompts

To construct progressive dificulty ladders, we prompted a rewrite model with carefully designed instructions.   
Below we include the exact prompts used for each dataset to ensure reproducibility.

## A.8.1 EntailmentBank Prompt

You are an expert at reasoning question simplification.

I will provide you with a reasoning problem in JSON format that contains:

\- "instruction": the solving instruction

\- "input": the context and question

\- "output": the reasoning chain and final answer

Your task is to automatically generate a progressive dificulty ladder of simplified versions of this problem.

Each new version should make the reasoning easier by moving more intermediate conclusions (from the reasoning steps in the output) directly into the input context.

Stop when the problem has become trivial (e.g., the final hypothesis is already in the input).

For each version, also output the minimum number of reasoning steps required to reach the final answer from that version’s input.

Treat a reasoning step as a necessary inferential move that derives a new statement from previous facts/conclusions (e.g., one arithmetic operation, one logical implication, one factual lookup from the provided context).

Count merged paraphrases/restatements as 0 additional steps; do not double-count trivially equivalent rewrites.

When multiple independent sub-derivations are needed before a final combination, count each indispensable sub-derivation as one step.

The count must be a non-negative integer; use 0 for a trivial version where the answer is directly stated in the input.

Ensure monotonic non-increase across versions (later versions should never require more steps than earlier ones).

## Guidelines:

1. Identify all intermediate conclusions (int1, int2, . . . ) in the

original reasoning chain.

2. Create Version 1 as the original (no added intermediates).

3. Then generate subsequent versions, each time inserting one or more intermediates into the input.

4. You may decide the number of versions automatically — fewer if the chain is short, more if it is long.

5. For each version, output in a fenced JSON code block with the

\- "instruction"

\- "input"

\- "answer" (string, the final answer to the problem)

\- "reasoning" (string, the reasoning chain leading to the answer)

\- "min\_steps" (integer, the minimum number of steps to reach the answer)

\- "min\_steps\_note" (a short explanation explaining the count)

6. Precede each block with a Markdown label like:

\## Version N — [dificulty descriptor]

Then immediately follow with:

‘‘‘json

{ ... }

Goal: produce a set of progressively easier problems, where the solver needs fewer reasoning steps at each level, and report the minimum required steps for each version.

## A.8.2 GSM8K Prompt

You are an expert at math word problem simplification.

I will provide you with a math problem in JSON format that contains:

\- "question": the text of the problem

\- "answer": the worked-out reasoning and final numeric answer

Your task is to automatically generate a \*progressive dificulty ladder\* of simplified versions of this problem.

Each new version should make the reasoning easier by moving more intermediate results (from the solution steps in the answer) directly into the problem statement.

Stop when the problem has become trivial (e.g., the final numeric answer is already stated in the problem).

For each version, also output the \*\*minimum number of reasoning steps\*\* required to reach the final answer from that version’s problem statement.

simplification, one comparison).

\- Do not double-count trivial rewrites or restatements.

\- When multiple sub-calculations are required before combining, count each indispensable sub-calculation as one step.

\- The count must be a non-negative integer; use \*\*0\*\* when the answer is already stated in the problem.

\- Ensure the counts are \*\*monotonic non-increasing\*\* across versions (later versions should never require more steps than earlier ones).

## Guidelines:

1. Identify all intermediate results (e.g., partial sums,

multiplications, divisions) in the original worked-out solution.

2. Create \*\*Version 1\*\* as the original (no added intermediates).

3. Then generate subsequent versions, each time inserting one or more intermediate results directly into the problem statement.

4. You may decide the number of versions automatically — fewer if the chain is short, more if it is long.

5. For each version, output in a fenced JSON code block with the following keys:

\- "question" (string, the modified problem statement)

\- "answer" (string, the final numeric answer only)

\- "reasoning" (string, the reasoning steps leading to the answer)

\- "min\_steps" (integer, the minimum number of steps required) - "min\_steps\_note" (short explanation for the count)

6. Precede each block with a Markdown label like:

\## Version N — [dificulty descriptor]

Then immediately follow with:

‘‘‘json

Goal: produce a set of progressively easier GSM8K problems, where the solver needs fewer reasoning steps at each level, and report the minimum required steps for each version.