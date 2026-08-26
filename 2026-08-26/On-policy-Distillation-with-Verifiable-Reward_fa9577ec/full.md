# On-policy Distillation with Verifiable Reward

Wenze Lin <sup>1</sup> <sup>∗</sup> <sup>†</sup>, Jiale Zhao <sup>1,2</sup> <sup>∗</sup>, Xitai Jiang <sup>1</sup> <sup>∗</sup>, Songde Rao <sup>3</sup>, Yining Li <sup>1</sup>, Shenzhi Wang <sup>1</sup>, Bingxiang He <sup>4</sup>, and Gao Huang<sup>1</sup> <sup>B</sup>

<sup>1</sup> LeapLab, Tsinghua University <sup>2</sup> Beihang University <sup>3</sup> SMS, Peking University <sup>4</sup> NLPLab, Tsinghua University

∗ Equal Contribution <sup>†</sup> Project Lead <sup>B</sup> Corresponding Author

Reinforcement Learning with Verifiable Rewards (RLVR) and on-policy distillation (OPD) have become two widely adopted paradigms for post-training large language models. However, RLVR sufers from sparse task-level feedback, while OPD provides dense token-level guidance but ignores trajectory correctness, limiting its performance to that of the teacher. Combining them is a promising direction: OPD supplies dense supervisory signals, while RLVR provides task-level correctness. Nevertheless, existing integrations often rely on weighted combination or heuristic switching, introducing extra hyperparameters and trade-ofs. We propose Onpolicy Distillation with Verifiable Reward (OPDVR), a simple yet efective method that seamlessly combines OPD and RLVR without adding any hyperparameters. We first reformulate the implicit reward of sampled-token OPD based on trajectory correctness, then apply a ReLU gating mechanism to ensure that correct trajectories receive non-negative rewards and incorrect ones receive non-positive rewards—thereby aligning the distillation signal with task success while preserving the teacher’s distributional guidance. Furthermore, our modification transforms sampled-token OPD into a proper RLVR method, making it readily combinable with any policy gradient algorithm, such as GRPO. Experiments on six reasoning benchmarks show that OPDVR consistently outperforms standard OPD. Our code is available at https: //github.com/LeapLabTHU/OPDVR.

## 1. Introduction

Reinforcement Learning with Verifiable Rewards (RLVR) has become a prevalent and efective post training paradigm for reasoning tasks (Guo et al., 2025; Shao et al., 2024; Jaech et al., 2024; Trinh et al., 2024; Yang et al., 2024). RLVR delivers explicit reward signals based on task outcomes, such as correct mathematical final answers or compilable code, which directly align model optimization with core task objectives. However, these rewards are typically sparse, providing little supervision for intermediate steps and making credit assignment challenging. Recently, On-policy Distillation (OPD) has emerged as a promising post-training paradigm (Xu et al., 2026; Xiao et al., 2026; Yang et al., 2025a; Zeng et al., 2026). Unlike RLVR, which relies on sparse outcome-level rewards, OPD leverages a teacher model to provide dense token-level guidance. However, OPD’s objective is purely distributional—it drives the student to mimic the teacher’s output distribution without considering whether the generated response is correct or incorrect, limiting the student’s performance to that of the teacher.

The complementary limitations of RLVR and OPD make combining them a natural and promising direction, where OPD provides fine-grained dense guidance and RLVR ofers reliable task-level correctness supervision. Existing approaches typically treat OPD and RLVR as two separate methods, either combining them explicitly or selectively applying one based on heuristic criteria. Despite their empirical efectiveness, such designs often introduce extra hyperparameters and heuristic trade-ofs (Cai et al., 2026; Yang et al., 2026a; Li et al., 2026a; Wang et al., 2026).

In this work, we propose On-policy Distillation with Verifiable Reward (OPDVR). We apply an extremely simple ReLU gating mechanism to sampled-token OPD, seamlessly combining OPD and RLVR without introducing any hyperparameters. We first revisit sampled-token OPD from an RLVR perspective and provide a mathematical reformulation of sample-token $\mathrm { \Delta \mathrm { O P D ^ { \prime } s } }$ reward. From the RLVR view, if we separate by trajectory correctness, sampled-token OPD can be viewed as applying token-level supervision by weighting binary task correctness rewards with teacher-student distribution discrepancies. Specifically, for sampled tokens that constitute correct reasoning trajectories, the model is assigned a token-level reward +1 weighted by $\log ( \pi _ { T } / \pi _ { \theta } ) ;$ ; for tokens leading to incorrect trajectories, the model receives a reward 1 weighted by $\log ( \pi _ { \theta } / \pi _ { T } )$ . However, we identify a critical limitation of this inherent reward design: both $\log ( \pi _ { T } / \pi _ { \theta } )$ and $\log ( \pi _ { \theta } / \pi _ { T } )$ are unbounded and can be either positive or negative. However, a key empirical principle shared by mainstream RLVR algorithms (Schulman et al., 2017; Guo et al., 2025; Shao et al., 2024) is that all tokens leading to a correct final outcome should receive non-negative advantages, while tokens leading to an incorrect outcome should receive non-positive advantages. This principle ensures that every token in a correct trajectory is treated as a valid prediction and reinforced accordingly, while every token in an incorrect trajectory is treated as an erroneous prediction and suppressed accordingly. Sampled-token OPD violates this principle: on correct trajectories, a negative value of $\log ( \pi _ { T } / \pi _ { \theta } )$ penalizes valid token behaviors, while on incorrect trajectories, a negative value of $\log ( \pi _ { \theta } / \pi _ { T } )$ encourages erroneous token predictions.

Motivated by this gap, we use a ReLU gating mechanism to standardize the reward signs according to trajectory correctness: it enforces non-negative $\log ( \pi _ { T } / \pi _ { \theta } )$ values for all verified correct trajectories and ensures non-negative log $\left( \pi _ { \boldsymbol { \theta } } / \pi _ { T } \right)$ values for all incorrect trajectories. This minimal correction directly converts conventional sampled-token OPD into a valid RLVR method while preserving the guidance from the teacher model. Specifically, for correct trajectories, larger $\log ( \pi _ { T } / \pi _ { \theta } )$ values—indicating accurate student predictions where the teacher is more confident than the student—yield stronger rewards. For incorrect trajectories, larger $\log ( \pi _ { \theta } / \pi _ { T } )$ values—corresponding to wrong student predictions where the student is more confident than the teacher—incur heavier penalties. This learning paradigm aligns well with human learning intuition, which reinforces reliable correct behaviors and suppresses overconfident erroneous predictions. Furthermore, by reformulating sampled-token OPD into an RLVR formulation, our OPDVR supports seamless integration with existing prominent RL algorithms, such as GRPO, DAPO and PPO. To instantiate this, we combine OPDVR with GRPO to obtain a new variant, which we call Group Relative Policy Distillation (GRPD). Overall, our method endows the model with dual supervisory signals: the rigorous task-level verifiable correctness from RLVR and the dense token-level distributional guidance from teacher distillation, leading to more robust and efective post-training optimization for reasoning tasks. Our experiments show that both OPDVR and GRPD consistently outperform OPD across all benchmarks.

## 2. Related Work

Reinforcement Learning with Verifiable Rewards (RLVR) Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a highly efective paradigm for post-training large language models (Schulman et al., 2017; Jaech et al., 2024; Trinh et al., 2024; Yang et al., 2024; Guo et al., 2025; Shao et al., 2024). RLVR leverages rule-based verifiers to provide objective, deterministic signals such as answer correctness or code compilation. Various extensions have been proposed to improve RLVR, such as controlling length (Liu et al., 2025; Sui et al., 2025; Xiang et al., 2025; Hammoud et al., 2025; Hou et al., 2025; Huang et al., 2026) and stabilizing entropy (Petrenko et al., 2026; Yang et al., 2025b; Jiang et al., 2025; Su et al., 2025). However, a fundamental limitation persists: the reward is sparse, leading to a severe credit assignment problem.

On-policy Distillation On-policy Distillation (OPD) has recently emerged as an eficient post-training paradigm that provides dense, token-level supervisory signals by distilling a teacher model’s output distribution into the student (Xu et al., 2026; Xiao et al., 2026; Yang et al., 2025a; Zeng et al., 2026). Unlike RLVR’s sparse outcome-based rewards, OPD ofers fine-grained guidance at every generation step. A growing line of work has explored various aspects of OPD, including its failure modes (Fu et al., 2026; Li et al., 2026b), how to determine which tokens are worth training (Xing et al., 2026; Jin et al., 2026), and its training stability issues in practice (Oh et al., 2026). However, OPD’s objective is purely distributional: it drives the student to mimic the teacher’s output distribution without considering whether the generated response is correct or incorrect. This observation naturally raises the question of whether OPD can be combined with RLVR, where trajectory correctness provides the ultimate optimization signal. Combining the dense supervision of OPD with the task-level verifiability of RLVR thus represents a promising and increasingly relevant direction.

![](images/d158cdeca81794dabf517c54ecd0e02259e7463bbfe0f086037a9d74e002d82c.jpg)

![](images/4661b5833f14e0b6e50ed2b814dfc1c418035dca559f436d8618ad192267cff7.jpg)  
Figure 1: Left: Overview of OPDVR. The ReLU gating mechanism ensures correct trajectories receive non-negative rewards and incorrect ones receive non-positive rewards, while preserving the teacher’s distributional guidance. Right: Results on AIME24, AIME25, and AMC under the same-architecture setting (Qwen3-4B  Qwen3-4B-RL), reported as avg@16 accuracy.

Combining OPD with RLVR Given the complementary strengths of OPD (dense token-level guidance) and RLVR (task-level correctness), several recent works have attempted to combine them. These include directly weighting the two objectives (Hubotter et al., 2026), applying OPD and RLVR separately depending on the sign of the advantage (Wang et al., 2026; Cai et al., 2026), aligning outcome rewards with process-level distillation signals (Hou et al., 2026), and leveraging teacher-student probability ratios for to weight the advantage in GRPO (Yang et al., 2026a). Despite their empirical gains, these approaches often treat OPD and RLVR as distinct objectives or losses that must be either explicitly combined or selectively applied based on heuristic criteria, often introducing additional hyperparameters for balancing the two terms, or relies on heuristic trade-ofs. Instead, our method implicitly transforms sampled-token OPD into a proper RLVR method using a simple gated mechanism, while preserving the teacher’s distributional guidance—achieving the benefits of both paradigms without explicit multi-objective balancing.

## 3. Preliminaries

## 3.1. Reinforcement Learning with Verifiable Rewards (RLVR)

Reinforcement Learning with Verifiable Rewards (RLVR) optimizes a policy using reward signals that can be automatically verified against ground truth. In mathematical reasoning tasks, the reward indicates whether the final answer is correct. A common choice for REINFORCE is $R = + 1$ for a correct trajectory and R = 1 for an incorrect one, while many other RLVR methods (including GRPO) use $R = + 1$ for correct and R = 0 for incorrect.

Consider a policy π<sub>θ</sub> that generates a response $o = \{ o _ { 1 } , \ldots , o _ { | o | } \}$ given a prompt q. The RLVR objective is to maximize the expected reward:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ R \right] ,
$$

where R is the trajectory-level reward.

To optimize this objective via gradient ascent, we compute the gradient:

$$
\begin{array} { r } { \nabla _ { \theta } \mathcal { I } ( \theta ) = \nabla _ { \theta } \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ R \right] . } \end{array}
$$

Using the REINFORCE (log-derivative) trick, we rewrite this as:

$$
\begin{array} { r } { \nabla _ { \theta } \mathcal { I } ( \theta ) = \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot \vert q ) } \left[ R \cdot \nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) \right] , } \end{array}
$$

where log $\begin{array} { r } { \pi _ { \theta } ( o \mid q ) = \sum _ { t = 1 } ^ { | o | } \log \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } \end{array}$

We sample a single response $o \sim \pi _ { \boldsymbol { \theta } } ( { \cdot } \mid q )$ to obtain a Monte Carlo estimate:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { I } ( \boldsymbol { \theta } ) \approx \boldsymbol { R } \cdot \sum _ { t = 1 } ^ { | o | } \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( o _ { t } \mid \boldsymbol { q } , o _ { < t } ) .
$$

This gradient can be implemented by minimizing the following loss over the sequence:

$$
{ \mathcal { L } } _ { \mathrm { R L V R } } ( \theta ) = - R \cdot \sum _ { t = 1 } ^ { | o | } \log \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) .
$$

When $R = + 1$ , minimizing this loss increases the probability of the entire response (reward). When $R = - 1$ , it decreases the probability (penalty). This formulation embodies the core RL principle: correct actions are rewarded, incorrect ones are penalized.

The REINFORCE formulation above sufers from high variance, as it relies on a single trajectory-level reward. A more stable alternative is Group Relative Policy Optimization (GRPO) (Guo et al., 2025; Shao et al., 2024). Specifically, for a group of G responses $\left\{ o 1 , \ldots , o _ { G } \right\}$ sampled from the policy, GRPO normalizes each response’s reward against the group mean and standard deviation:

$$
\hat { A } _ { i , t } = \frac { R _ { i } - \operatorname* { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } .
$$

Note that GRPO typically uses $R \in \{ 0 , 1 \}$ (1 for correct, 0 for incorrect) rather than $\{ - 1 , + 1 \}$ , since the group-relative normalization naturally centers the advantages around zero.

This group-relative advantage reduces variance and serves as a stable, computationally eficient alternative to direct binary rewards. For simplicity, we present the on-policy version of GRPO without clipping or KL penalty; the loss is:

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = - \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { \left| o _ { i } \right| } \sum _ { t = 1 } ^ { \left| o _ { i } \right| } \hat { A } _ { i , t } \log \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) .
$$

## 3.2. On-policy Distillation

On-policy Distillation (OPD) optimizes a student policy π<sub>θ</sub> toward a teacher policy π<sub>T</sub> by minimizing their reverse KL divergence:

$$
\operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ \sum _ { t = 1 } ^ { | o | } D _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot  { | } q , o _ { < t } ) \parallel \pi _ { T } ( \cdot  { | } q , o _ { < t } ) ) \right] .
$$

The full-vocabulary OPD computes this exact KL at every token position:

$$
\mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { f u l l } } ( \theta ) = \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ \sum _ { { t = 1 } } ^ { | o | } \sum _ { v \in \mathcal { V } } \pi _ { \theta } ( v \mid q , o _ { < t } ) \log \frac { \pi _ { \theta } ( v \mid q , o _ { < t } ) } { \pi _ { T } ( v \mid q , o _ { < t } ) } \right] .
$$

This loss is directly minimized. In practice, the log-ratio log $( \pi _ { \boldsymbol { \theta } } / \pi _ { T } )$ is treated as a constant reward with stop-gradient when computing gradients, which is equivalent to the exact gradient of the reverse KL via the log-derivative trick. However, computing the reverse KL over the full vocabulary at every token position is computationally expensive, especially for large language models with very large vocabularies.

A practical compromise is Top-k OPD, which restricts the KL computation to the k tokens with the highest student probabilities:

$$
\mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { t o p - } k } ( \theta ) = \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ \sum _ { { t = 1 } } ^ { | o | } D _ { \mathrm { K L } } \left( \bar { \pi } _ { \theta } ^ { ( S _ { t } ) } \parallel \bar { \pi } _ { T } ^ { ( S _ { t } ) } \right) \right] ,
$$

where $S _ { t }$ is the set of top-k tokens.

The most widely adopted variant is sampled-token OPD. It uses a single Monte Carlo sample $o _ { t } \sim \pi _ { \theta } ( \cdot \mid q , o _ { < t } )$ to approximate the reverse KL gradient:

$$
\mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { s a m p l e } } ( \theta ) = \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ \sum _ { t = 1 } ^ { | o | } \log \frac { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } \right] .
$$

The log-ratio $\log ( \pi _ { \theta } / \pi _ { T } )$ is treated as a constant reward with stop-gradient.

## 4. Method

## 4.1. Revisiting Sampled-Token OPD from an RLVR Perspective

We begin with the sampled-token OPD loss. Recall that sampled-token OPD approximates the reverse KL gradient using a single Monte Carlo sample. The loss over the entire response $o = \{ o _ { 1 } , \ldots , o _ { | o | } \}$ is defined as:

$$
\mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { s a m p l e } } ( \theta ) = \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } \left[ \sum _ { t = 1 } ^ { | o | } \log \frac { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } \right] .
$$

For a sampled response $o \sim \pi _ { \theta } ( \cdot \mid q )$ , the per-token loss is:

$$
\ell _ { t } = \log { \frac { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } } .
$$

The gradient of the sequence-level loss with respect to $\theta$ is:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { O P D } } ^ { \mathrm { s a m p l e } } ( \boldsymbol { \theta } ) = \sum _ { t = 1 } ^ { | o | } \log \frac { \pi _ { \boldsymbol { \theta } } \left( o _ { t } \mid \boldsymbol { q } , o _ { < t } \right) } { \pi _ { T } \left( o _ { t } \mid \boldsymbol { q } , o _ { < t } \right) } \cdot \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } \left( o _ { t } \mid \boldsymbol { q } , o _ { < t } \right) .
$$

We now compare this with the standard RLVR gradient. Recall from Section 3 that the RLVR loss for a trajectory with reward R is:

$$
{ \mathcal { L } } _ { \mathrm { R L V R } } ( \theta ) = - R \cdot \sum _ { t = 1 } ^ { | o | } \log \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) .
$$

Its gradient is:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { R L V R } } ( \theta ) = - R \cdot \sum _ { t = 1 } ^ { | o | } \nabla _ { \theta } \log \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) .
$$

![](images/e64119be3ce390d7edf3600e5a8650e7302286bb1e3d68e312917bb63b89fa10.jpg)  
Figure 2: OPDVR method

We observe that the sampled-token OPD gradient shares the same form as the RLVR gradient, where the log-ratio term corresponds to the reward coeficient. By matching the coeficients of $\nabla _ { \theta } \log \pi _ { \theta } \big ( o _ { t } | q , o _ { < t } \big )$ we obtain:

$$
- R = \log { \frac { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } } .
$$

this observation allows us to reinterpret the log-ratio term as an implicit token-level reward $R _ { \mathrm { O P D } } ( o _ { t } )$

$$
R _ { \mathrm { O P D } } ( o _ { t } ) = - \log \frac { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } = \log \frac { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } .
$$

However, unlike RLVR, where the reward sign is determined by the verifier outcome, the sign of Ropd is solely determined by the teacher-student probability ratio. To analyze their alignment, we consider two cases based on trajectory correctness:

$$
R _ { { \mathrm { O P D } } } ( o _ { t } ) = \left\{ \begin{array} { l l } { \log \displaystyle \frac { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } \cdot ( + 1 ) , \quad \mathrm { t r a j e c t o r y ~ c o r r e c t } , \smallskip } \\ { \log \displaystyle \frac { \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) } { \pi _ { T } ( o _ { t } \mid q , o _ { < t } ) } \cdot ( - 1 ) , \quad \mathrm { t r a j e c t o r y ~ i n c o r r e c t } . } \end{array} \right.
$$

In other words, on correct trajectories, the gradient is weighted by $\log ( \pi _ { T } / \pi _ { \theta } )$ ; on incorrect trajectories, it is weighted by $\log ( \pi _ { \theta } / \pi _ { T } )$ with a negative sign.

The empirical RLVR principle requires that correct trajectories receive positive rewards and incorrect trajectories receive negative rewards. However, the sign of log $\left( \pi _ { T } / \pi _ { \theta } \right)$ is determined by whether $\pi _ { T } > \pi _ { \theta }$ or $\pi _ { T } < \pi _ { \theta }$ , which is independent of trajectory correctness. This creates two failure cases: correct trajectories can receive negative rewards when $\pi _ { T } < \pi _ { \theta }$ , and incorrect trajectories can receive positive rewards when $\pi _ { T } > \pi _ { \theta }$ . Both violate the RL principle.

## 4.2. OPDVR: A Simple Gated Mechanism

We propose On-policy Distillation with Verifiable Reward (OPDVR) to resolve this issue. The solution is extremely simple: apply a ReLU gate to enforce RLVR compliance on the sampled token while preserving the teacher’s distributional guidance.

The reward can be written as:

$$
R _ { \mathrm { O P D V R } } ( o _ { t } ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \left( 0 , \log \frac { \pi _ { T } \left( o _ { t } \mid q , o _ { < t } \right) } { \pi _ { \theta } \left( o _ { t } \mid q , o _ { < t } \right) } \right) \cdot ( + 1 ) , } & { \mathrm { t r a j e c t o r y ~ c o r r e c t } , } \\ { \operatorname* { m a x } \left( 0 , \log \frac { \pi _ { \theta } \left( o _ { t } \mid q , o _ { < t } \right) } { \pi _ { T } \left( o _ { t } \mid q , o _ { < t } \right) } \right) \cdot ( - 1 ) , } & { \mathrm { t r a j e c t o r y ~ i n c o r r e c t } . } \end{array} \right.
$$

The corresponding loss is:

$$
{ \mathcal { L } } _ { \mathrm { O P D V R } } ( \theta ) = - \sum _ { t = 1 } ^ { | o | } R _ { \mathrm { O P D V R } } ( o _ { t } ) \cdot \log \pi _ { \theta } ( o _ { t } \mid q , o _ { < t } ) .
$$

This gated mechanism ensures that:

• Correct trajectories receive non-negative rewards, and incorrect trajectories receive non-positive rewards, aligning the reward sign with trajectory correctness.

• For tokens on correct trajectories, the reward magnitude is larger when the teacher is more confident than the student, $\mathrm { i . e . , } \log ( \pi _ { T } / \pi _ { \theta } )$ is larger. This encourages the model to reinforce choices that are both correct and reliable according to the teacher.

• For tokens on incorrect trajectories, the penalty magnitude is larger when the student is more confident than the teacher, i.e., $\log ( \pi _ { \theta } / \pi _ { T } )$ is larger. This forces the model to suppress overconfident mistakes.

## 4.3. Interpreting the ReLU Gating Mechanism as a Conditional Mask

In this subsection, we interpret the ReLU gating mechanism in OPDVR as a conditional token mask: it selectively zeroes out gradient updates on tokens whose learning direction conflicts with the verifier signal. Standard sampled-token OPD lacks this mask, so it inevitably updates all tokens based solely on the teacher-student confidence ratio, regardless of trajectory correctness.

To see this, recall the standard OPD gradient for a sampled token a:

$$
g _ { \mathrm { O P D } } = \underbrace { \left[ \log \frac { \pi } { \pi _ { \theta } } \right] _ { + } \nabla _ { \theta } \log \pi _ { \theta } ( a ) } _ { \mathrm { T e r m ~ A : ~ p u s h ~ u p ~ ( t e a c h e r ~ m o r e ~ c o n f i d e n t ) } } - \underbrace { \left[ \log \frac { \pi _ { \theta } } { \pi _ { T } } \right] _ { + } \nabla _ { \theta } \log \pi _ { \theta } ( a ) } _ { \mathrm { T e r m ~ B : ~ p u l l ~ d o w n ~ ( s t u d e n t ~ m o r e ~ c o n f i d e n t ) } } .\tag{1}
$$

The OPDVR gradient instead applies the ReLU gate conditioned on the verifier reward $R \in \{ + 1 , - 1 \}$

$$
g _ { \mathrm { O P D V R } } = \left\{ { \begin{array} { l l } { { \mathrm { T e r m ~ A } } , } & { R = + 1 , } \\ { - { \mathrm { T e r m ~ B } } , } & { R = - 1 . } \end{array} } \right.\tag{2}
$$

Thus, compared to standard OPD, OPDVR removes exactly two types of tokens whose updates conflict with the verifier:

• Type I Conflicting Token (Correct Trajectory, $\pi _ { \boldsymbol { \theta } } > \pi _ { T } )$ : When the trajectory is correct $( R = + 1 )$ , the student has already produced the right answer. If the student is more confident than the teacher on a specific token $( \pi _ { \theta } > \pi _ { T } )$ , then Term B is activated. Standard OPD applies a negative gradient, pulling down this correct token’s probability to match the teacher’s lower confidence. This update works against the verifier signal in two ways:

1. The student’s higher confidence is consistent with the correct outcome.

2. Reducing the student’s confidence on a correct answer introduces a distributional distortion that is not driven by task performance.

• Type II Conflicting Token (Incorrect Trajectory, $\pi _ { T } > \pi _ { \theta } )$ : When the trajectory is incorrect $( R = - 1 )$ , the student has produced a wrong answer. If the teacher is more confident than the student on a specific token $\left( \pi _ { T } > \pi _ { \theta } \right)$ , then Term A is activated. Standard OPD applies a positive gradient, pushing up this incorrect token’s probability to align with the teacher. This update works against the verifier signal in two ways:

1. The teacher’s high confidence does not accompany a correct trajectory here.

2. Increasing the probability of tokens on an incorrect trajectory reinforces a reasoning pattern that the verifier has marked as wrong.

By masking out these two token types, OPDVR preserves the student’s correct high-confidence predictions and withholds the teacher’s guidance on wrong answers. The teacher still controls the magnitude of the update via the log-ratio, but the verifier now determines the direction—reinforce or suppress—so that the distillation process never works against the task reward. We provide a further analysis in Appendix A.

## 4.4. Group Relative Policy Distillation (GRPD)

Having established that OPDVR transforms sampled-token OPD into a proper RLVR method with a binary verifier reward, a natural extension is to replace this coarse binary signal with a more nuanced, group-relative advantage estimate like GRPO (Guo et al., 2025; Shao et al., 2024).

Recall from Section 3 that GRPO computes a group-relative advantage $\hat { A } _ { i , t }$ for each token position t in response $o _ { i } \colon$

$$
\hat { A } _ { i , t } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } ,
$$

where $R _ { i } \in \{ 0 , 1 \}$ is the verifier reward for response $o _ { i } ,$ , and G is the group size. The key property of $\hat { A } _ { i , t }$ is that it is positive for responses that are better than the group average, and negative for those that are worse—capturing relative performance within the sampled batch.

We now apply the same ReLU gating logic, but with the binary correctness sign R replaced by the group-relative advantage $\hat { A } _ { i , t }$ . The GRPD reward becomes:

$$
R _ { { \mathrm { G R P D } } } ( o _ { i , t } ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \bigg ( 0 , \log \frac { \pi _ { T } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } \bigg ) \cdot ( + 1 ) , } & { \hat { A } _ { i , t } > 0 , } \\ { \operatorname* { m a x } \bigg ( 0 , \log \frac { \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { T } ( o _ { i , t } \mid q , o _ { i , < t } ) } \bigg ) \cdot ( - 1 ) , } & { \hat { A } _ { i , t } < 0 . } \end{array} \right.
$$

Equivalently, this can be written compactly as:

$$
R _ { \mathrm { G R P D } } ( o _ { i , t } ) = \mathrm { s i g n } ( \hat { A } _ { i , t } ) \cdot \mathrm { R e L U } \biggl ( \mathrm { s i g n } ( \hat { A } _ { i , t } ) \cdot \log \frac { \pi _ { T } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } \biggr ) ,
$$

where $\mathrm { s i g n } ( \hat { A } _ { i , t } )$ ensures the direction of the update is governed by the advantage sign.

The corresponding loss is:

$$
\mathcal { L } _ { \mathrm { G R P D } } ( \theta ) = - \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } R _ { \mathrm { G R P D } } ( o _ { i , t } ) \cdot \log \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) .
$$

## 5. Experiments

## 5.1. Main Experimental Settings

We conduct experiments on both same-architecture and cross-architecture distillation settings to evaluate the efectiveness of our method.

Same-architecture setting. We use Qwen3-4B-nonthinking as the student model and distill it from a teacher model of the same architecture, which is obtained by training Qwen3-4B with GRPO on the DeepMath dataset (He et al., 2026). Following Yang et al. (2026b), we use the filtered subset of the DeepMath dataset consisting of 57k samples with dificulty level $\geq 6$

Cross-architecture setting. We use the DAPO-Math-17k dataset (Yu et al., 2026). The teacher model is Qwen3-4B-base, fine-tuned with GRPO for 3 epochs on the same dataset. The student model is Qwen3-1.7B-base. The distillation training runs for 3 epochs.

Table 1: Results on same-architecture distillation (Qwen3-4B  Qwen3-4B-RL). All models are evaluated with avg@16 accuracy.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>Student (Qwen3-4B)</td><td>24.0</td><td>15.8</td><td>60.8</td><td>80.9</td><td>27.6</td><td>42.9</td><td>42.0</td></tr><tr><td rowspan="2">Teacher (Qwen3-4B-RL) Sampled-Token OPD</td><td>36.0</td><td>29.0</td><td>65.9</td><td>87.0</td><td>35.4</td><td>49.3</td><td>50.4</td></tr><tr><td>34.2</td><td>26.0</td><td>63.1</td><td>85.5</td><td>31.6</td><td>46.5</td><td>47.8</td></tr><tr><td>Top-64 OPD</td><td>34.6</td><td>23.5</td><td>62.0</td><td>85.0</td><td>32.2</td><td>46.8</td><td>47.4</td></tr><tr><td>OPDVR (Ours)</td><td>36.9</td><td>28.1</td><td>64.8</td><td>84.7</td><td>33.2</td><td>47.0</td><td>49.1</td></tr></table>

Table 2: Results on cross-architecture distillation (Qwen3-1.7B-Base  Qwen3-4B-Base-RL). All models are evaluated with avg@16 accuracy.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>Student (Qwen3-1.7B-Base)</td><td>4.1</td><td>1.7</td><td>23.2</td><td>48.9</td><td>8.9</td><td>17.1</td><td>17.3</td></tr><tr><td>Teacher (Qwen3-4B-Base-RL)</td><td>10.6</td><td>13.1</td><td>40.3</td><td>74.2</td><td>17.2</td><td>30.0</td><td>30.9</td></tr><tr><td>Sampled-Token OPD</td><td>6.5</td><td>2.1</td><td>24.8</td><td>59.1</td><td>11.5</td><td>21.6</td><td>20.9</td></tr><tr><td>Top-64 OPD</td><td>8.5</td><td>3.3</td><td>26.4</td><td>60.1</td><td>10.7</td><td>21.4</td><td>21.7</td></tr><tr><td>OPDVR (Ours)</td><td>8.5</td><td>3.3</td><td>30.3</td><td>60.8</td><td>11.6</td><td>22.0</td><td>22.8</td></tr></table>

Benchmarks. We evaluate all models on six reasoning benchmarks: AIME24, AIME25, AMC, MATH500, Minerva, and OlympiadBench.

## 5.2. Main Results

We compare our method OPDVR against standard sampled-token OPD and top-64 OPD on both same-architecture and cross-architecture distillation settings. Tables 1 and 2 report the performance on six reasoning benchmarks.

As shown in both tables, OPDVR consistently outperforms standard sampled-token OPD and top-64 OPD across all six benchmarks in both distillation settings. In the same-architecture setting, OPDVR achieves gains of 2.7 points on AIME24 and 2.1 points on AIME25 over sampled-token OPD, and even surpasses the teacher model on AIME24. In the cross-architecture setting, OPDVR delivers substantial improvements over the baselines, with gains of 5.5 points on AMC and 1.7 points on MATH500 over sampled-token OPD, highlighting its robustness to architectural diferences.

## 5.3. Group Relative Policy Distillation

To further validate the efectiveness of replacing the binary verifier reward with group-relative advantages, we instantiate OPDVR with GRPO-style advantage estimation (describe in Section 4.4) and evaluate it under the same-architecture setting. The teacher model is the same Qwen3-4B-Nonthinking model trained with GRPO on DeepMath. The student model is Qwen3-4B-nonthinking. To more cleanly evaluate the efectiveness of replacing binary rewards with group-relative advantages, we conduct this experiment on DAPO-Math-17K, distinct from the teacher’s DeepMath training data. The group size is set to G = 8. We compare GRPD against GRPO and standard sampled-token OPD.

Table 3 reports the results. GRPD consistently outperforms both GRPO and OPD across all six benchmarks, with notable gains of 6.5 points on AIME24 and 10.9 points on AIME25 over GRPO. It also surpasses OPD on five out of six benchmarks, achieving a 2.8-point improvement on AIME24. These results demonstrate that combining group-relative advantage estimation with the ReLU-gated distillation signal yields stronger and more stable improvements than either method alone.

Table 3: Results on Group Relative Policy Distillation $( \mathrm { Q w e n 3 - 4 B  Q w e n 3 - 4 B - R L } )$ . All models are evaluated with avg@16 accuracy.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>Student (Qwen3-4B)</td><td>24.0</td><td>15.8</td><td>60.8</td><td>80.9</td><td>27.6</td><td>42.9</td><td>42.0</td></tr><tr><td>Teacher (Qwen3-4B-RL)</td><td>36.0</td><td>29.0</td><td>65.9</td><td>87.0</td><td>35.4</td><td>49.3</td><td>50.4</td></tr><tr><td>GRPO</td><td>28.3</td><td>20.8</td><td>62.3</td><td>83.9</td><td>28.9</td><td>44.6</td><td>44.8</td></tr><tr><td>OPD</td><td>32.0</td><td>31.7</td><td>65.6</td><td>85.4</td><td>28.9</td><td>46.6</td><td>48.4</td></tr><tr><td>GRPD (Ours)</td><td>34.8</td><td>31.7</td><td>67.0</td><td>85.6</td><td>30.5</td><td>47.0</td><td>49.4</td></tr></table>

![](images/50cda42ef9adf8119dedf02d9976804d483feb65fbda6999aa1c9362069e024d.jpg)

![](images/d48a5b096cc618cc9edf9a4629be90dfc1e6468449a8c81a8631631f701c98a5.jpg)  
Figure 3: Ablation study on the gating mechanism. Left: training-time accuracy reward of OPD, OPDVR, and the inverse-gated variant. Right: average accuracy over six benchmarks.

## 5.4. Ablation Study: Inverse-Gated Experiment

To examine the efectiveness of the gating mechanism in OPDVR, we conduct an inverse-gated ablation under the same-architecture setting (Qwen3-4B Qwen3-4B-RL) on DeepMath, following the same experimental setup as the main experiments.. Recall that OPDVR keeps a token only when its update direction agrees with the verifier signal: on a correct trajectory $( R { = } { + } 1 )$ , tokens where the teacher is more confident than the student $\left( \pi _ { T } > \pi _ { \theta } \right)$ are kept and tokens where the student is already more confident $( \pi _ { \theta } > \pi _ { T } )$ are gated out; on an incorrect trajectory $( R { = } - 1 )$ , the roles are reversed – tokens where the student is more confident than the teacher are kept, and tokens where the teacher is more confident are gated out. The inverse-gated variant swaps these two sets exactly: it keeps the tokens that OPDVR gates out (the student-more-confident tokens on correct trajectories and the teacher-more-confident tokens on incorrect trajectories), and gates out the tokens that OPDVR keeps, while keeping the masking ratio and everything else unchanged.

Table 4 shows the final benchmark results, and Figure 3 tracks the accuracy reward on the training set. The three variants start from the same point and separate monotonically throughout training, yielding a consistent ordering OPDVR > OPD > Inverse-Gated in both the training curves and the final benchmarks. Reversing the gate falls below vanilla OPD on all six benchmarks, confirming the efectiveness of the gating mechanism. Notably, even the inverse-gated variant still improves over the initial model, indicating that the teacher’s distributional guidance remains useful—though its benefits are substantially hindered when the sign is misaligned with trajectory correctness

Table 4: Ablation study on the gating mechanism. We compare OPD, OPDVR, and an inverse-gated variant where the ReLU gate is applied in the opposite direction.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>Student (Qwen3-4B)</td><td>24.0</td><td>15.8</td><td>60.8</td><td>80.9</td><td>27.6</td><td>42.9</td><td>42.0</td></tr><tr><td>Teacher (Qwen3-4B-RL)</td><td>36.0</td><td>29.0</td><td>65.9</td><td>87.0</td><td>35.4</td><td>49.3</td><td>50.4</td></tr><tr><td>OPD</td><td>34.2</td><td>26.0</td><td>63.1</td><td>85.5</td><td>31.6</td><td>46.5</td><td>47.8</td></tr><tr><td>OPDVR (Ours)</td><td>36.9</td><td>28.1</td><td>64.8</td><td>84.7</td><td>33.2</td><td>47.0</td><td>49.1</td></tr><tr><td>Inverse-Gated</td><td>30.3</td><td>21.2</td><td>62.3</td><td>83.6</td><td>27.7</td><td>42.8</td><td>44.6</td></tr></table>

![](images/9ff75410a03e94242212e6a010480825e528a9dd7460c07071343b87bfd2251c.jpg)

![](images/30dff90fc16924057937c4ee7ab4d857f0aae9a8eccafc46f3e896bdb781a793.jpg)

![](images/9b3ee36e0c6e3c15cd9fca2e9b6f502b115c41f0a38b5e0c5f592bbbb5e62320.jpg)  
(a) Training dynamic of same-architecture distillation (Qwen3-4B ← Qwen3-4B-RL).

![](images/f9a2d078f37cca726108f53c433ee803d2de1f5fc79f1e5e2f8dbd7c649a5705.jpg)

![](images/9cf4686c57073eb9d4cc5addecce2982041c0cb4fbf9e7a1714fee6ec82938f4.jpg)

![](images/58bf3493a5deb2d964bd23677d9e4d37464e20e3c79cce1c8cbdb03fda6b4d60.jpg)  
(b) Training dynamic of cross-architecture distillation (Qwen3-1.7B-Base ← Qwen3-4B-Base-RL).  
Figure 4: Training dynamics

## 5.5. Training Dynamics

To understand how token-level gating shapes the distillation process, we track three quantities throughout training: the student policy entropy, the average response length, and the zero-gated token ratio, i.e., the fraction of tokens whose distillation loss is masked by the gate. Figure 4 shows the dynamics of the same-architecture setting (Qwen3-4B  Qwen3-4B-RL) and the cross-architecture setting (Qwen3-1.7B-Base Qwen3-4B-Base-RL), respectively.

The entropy and response-length trajectories of the sampled-token OPD baseline show no consistent pattern across settings; instead, they depend strongly on the teacher–student pair. In the samearchitecture setting, the entropy of both methods drifts mildly upward (from 0.33 to 0.40) while the response length inflates by more than four times, from roughly 1.6k to over 6.7k tokens. In the cross-architecture setting, the picture inverts: the student entropy collapses rapidly from 2.0 at the start of training and the response length stays nearly flat throughout. These quantities therefore do not follow a setting-independent trend – their evolution is dictated by the specific teacher and student models rather than by the distillation objective itself.

In contrast, the zero-gated token ratio is consistent across both settings. It remains in a band around fifty percent during the entire course of training ( 0.48–0.50 for the 4B student and 0.40–0.44 for the 1.7B student), never degenerating toward the trivial extremes of gating all or no tokens. This means that roughly half of the sampled tokens are zeroed by the ReLU gate at any point of training. Recalling the failure cases identified in Section 4, these are exactly the tokens whose implicit OPD reward pushes against the RLVR direction: on correct trajectories where the student already assigns higher probability than the teacher $( \pi _ { \theta } > \pi _ { T } )$ , and on incorrect trajectories where the teacher still outranks the student $\left( \pi _ { T } > \pi _ { \theta } \right)$ . The stable ratio indicates that such RLVR-violating tokens constitute a persistent fraction of the data throughout training – the gate consistently filters them out while distilling the remaining tokens with their full teacher–student log-ratio magnitude.

## 6. Conclusion

We revisited sampled-token OPD from an RLVR perspective and observed that its implicit reward is governed by the teacher-student probability ratio rather than trajectory correctness, which can lead to updates that can penalize tokens in correct trajectories or reward tokens in incorrect ones. To address this, we proposed OPDVR, which adds a simple ReLU gate on the sampled token to enforce RLVR compliance. This minimal modification combines OPD and RLVR without any hyperparameter or heuristic trade-of. Experiments across mathematical reasoning benchmarks demonstrate that OPDVR consistently outperforms standard OPD.

More importantly, by reformulating OPD as an RLVR method, OPDVR becomes a general framework that can be readily integrated with any policy gradient algorithm, including REINFORCE, GRPO, and DAPO. We instantiate this with GRPO and present Group Relative Policy Distillation (GRPD), which further demonstrates the versatility and strength of our approach. The empirical success of both OPDVR and GRPD across multiple benchmarks confirms that aligning distillation signals with trajectory correctness is not only efective but also broadly applicable, ofering a simple yet powerful recipe for future post-training methods.

## References

Cai, Q., Ma, Y., Li, L., Li, P., Chen, Y., Guo, Q., Zou, Y., Gui, T., Feng, X., and Qin, B. Hˆ2 sd: Hybrid hindsight self-distillation. arXiv preprint arXiv:2607.18955, 2026.

Fu, Y., Huang, H., Jiang, K., Liu, J., Jiang, Z., Zhu, Y., and Zhao, D. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562, 2026.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Hammoud, H. A. A. K., Alhamoud, K., Hammoud, A., Bou-Zeid, E., Ghassemi, M., and Ghanem, B. Train long, think short: Curriculum learning for eficient reasoning. arXiv preprint arXiv:2508.08940, 2025.

He, Z., Liang, T., Xu, J., Liu, Q., Chen, X., Wang, Y., Song, L., Yu, D., Liang, Z., Wang, W., et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. In International Conference on Learning Representations, volume 2026, pp. 138306–138322, 2026.

Hou, B., Zhang, Y., Ji, J., Liu, Y., Qian, K., Andreas, J., and Chang, S. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

Hou, W., Peng, S., Wang, W., Ruan, Z., Zhang, Y., Zhou, Z., Gao, M., Chen, Y., Wang, K., Yang, H., et al. Uni-opd: Unifying on-policy distillation with a dual-perspective recipe. arXiv preprint arXiv:2605.03677, 2026.

Huang, C., Zhang, Z., and Cardie, C. Hapo: Training language models to reason concisely via history aware policy optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 31122–31130, 2026.

Hubotter, J., Lubeck, F., Behric, L. D., Baumann, A., Bagatella, M., Marta, D., Hakimi, I., Shenfeld, I., Buening, T. K., Guestrin, C., and Krause, A. Reinforcement learning via self-distillation. ArXiv, abs/2601.20802, 2026. URL https://api.semanticscholar.org/CorpusID:285102353.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jiang, Y., Li, Y., Chen, G., Liu, D., Cheng, Y., and Shao, J. Rethinking entropy regularization in large reasoning models. arXiv preprint arXiv:2509.25133, 2025.

Jin, W., Min, T., Yang, Y., Wei, D., Zhou, Y., Kadhe, S. R., Baracaldo, N., and Lee, K. Entropy-aware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026.

Li, G., Yang, T., Fang, J., Song, M., Zheng, M., Guo, H., Zhang, D., Wang, J., and Chua, T.-S. Unifying group-relative and self-distillation policy optimization via sample routing. arXiv preprint arXiv:2604.02288, 2026a.

Li, Y., Zuo, Y., He, B., Zhang, J., Xiao, C., Qian, C., Yu, T., Gao, H.-a., Yang, W., Liu, Z., et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026b.

Liu, W., Zhou, R., Deng, Y., Huang, Y., Liu, J., Deng, Y., Zhang, Y., and He, J. Learn to reason eficiently with adaptive length-based reward shaping. arXiv preprint arXiv:2505.15612, 2025.

Oh, M., Song, S., Choi, G., Choi, Y., and Jo, Y. Kl for a kl: On-policy distillation with control variate baseline. arXiv preprint arXiv:2605.07865, 2026.

Petrenko, A., Lipkin, B., Chen, K., Wijmans, E., Cusumano-Towner, M., Giryes, R., and Krähenbühl, P. Entropy-preserving reinforcement learning. arXiv preprint arXiv:2603.11682, 2026.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Su, Z., Pan, L., Lv, M., Li, Y., Hu, W., Zhang, F., Gai, K., and Zhou, G. Ce-gppo: Coordinating entropy via gradient-preserving clipping policy optimization in reinforcement learning. ArXiv, abs/2509.20712, 2025. URL https://api.semanticscholar.org/CorpusID:281526201.

Sui, Y., Chuang, Y.-N., Wang, G., Zhang, J., Zhang, T., Yuan, J., Liu, H., Wen, A., Zhong, S., Zou, N., et al. Stop overthinking: A survey on eficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

Trinh, T. H., Wu, Y., Le, Q. V., He, H., and Luong, T. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.

Wang, C., Li, Z., Bai, J., Zhang, Y., Deng, H., Lan, G., and Wang, Y. Distilled reinforcement learning for llm post-training. arXiv preprint arXiv:2607.17247, 2026.

Xiang, V., Blagden, C., Rafailov, R., Lile, N., Truong, S., Finn, C., and Haber, N. Just enough thinking: Eficient reasoning with adaptive length penalties reinforcement learning. arXiv preprint arXiv:2506.05256, 2025.

Xiao, B., Xia, B., Yang, B., Gao, B., Shen, B., Zhang, C., He, C., Lou, C., Luo, F., Wang, G., et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.

Xing, X., Wang, H., Gao, B., Li, Z., and Tang, Y. Trust region on-policy distillation. arXiv preprint arXiv:2606.01249, 2026.

Xu, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., Ling, C., et al. Deepseekv4: Towards highly eficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, C., Qin, C., Si, Q., Chen, M., Gu, N., Yao, D., Lin, Z., Wang, W., Wang, J., and Duan, N. Self-distilled rlvr. arXiv preprint arXiv:2604.03128, 2026a.

Yang, K., Xu, X., Chen, Y., Liu, W., Lyu, J., Lin, Z., Ye, D., and Yang, S. Entropic: Towards stable long-term training of llms via entropy stabilization with proportional-integral control. arXiv preprint arXiv:2511.15248, 2025b.

Yang, W., Liu, W., Xie, R., Yang, K., Yang, S., and Lin, Y. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026b.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

Zeng, A., Lv, X., Hou, Z., Du, Z., Zheng, Q., Chen, B., Yin, D., Ge, C., Huang, C., Xie, C., et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

## A. Theoretical Analysis: Why OPDVR Improves over Sampled-Token OPD

We now provide a formal characterization of the advantage of OPDVR over the sampled-token OPD objective. Throughout this section, let $s _ { t } = \left( q , o _ { < t } \right)$ denote the state at step $t , a _ { t } = o _ { t }$ the sampled action, and define the token-level teacher-student log-ratio as

$$
r _ { t } = \log { \frac { \pi _ { T } ( a _ { t } \mid s _ { t } ) } { \pi _ { \theta } ( a _ { t } \mid s _ { t } ) } } .\tag{3}
$$

As before, $R \in \{ + 1 , - 1 \}$ denotes the trajectory-level verifier reward.

## A.1. Directional Alignment with the Verifier Gradient

Let $u _ { t } = \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } )$ denote the policy gradient direction of the sampled token. The token-level update direction of sampled-token OPD is

$$
\Delta _ { \mathrm { O P D } , t } = r _ { t } u _ { t } ,\tag{4}
$$

whereas the OPDVR update direction is

$$
\Delta _ { \mathrm { O P D V R } , t } = R \mathrm { R e L U } ( R r _ { t } ) u _ { t } ,\tag{5}
$$

with ${ \mathrm { R e L U } } ( x ) = \operatorname* { m a x } ( 0 , x )$

Recall that the pure verifier-driven RLVR update direction is $\Delta _ { \mathrm { R L V R } , t } = R u _ { t }$ . The following proposition shows that OPDVR is always aligned with this verifier direction, while OPD may be anti-aligned.

Proposition A.1 (Verifier alignment). For any token with $\boldsymbol u _ { t } \ne 0$ , the OPDVR update is never in the opposite direction of the verifier gradient. Specifically,

$$
\langle \Delta _ { \mathrm { O P D V R } , t } , \Delta _ { \mathrm { R L V R } , t } \rangle \geq 0 .\tag{6}
$$

In contrast, sampled-token OPD satisfies

$$
\langle \Delta _ { \mathrm { O P D } , t } , \Delta _ { \mathrm { R L V R } , t } \rangle = r _ { t } R \Vert u _ { t } \Vert ^ { 2 } ,\tag{7}
$$

which becomes negative whenever $r _ { t }$ and R have opposite signs.

Proof. Using (5) and $\Delta _ { \mathrm { R L V R } , t } = R u _ { t }$ , we have

$$
\langle \Delta _ { \mathrm { O P D V R } , t } , \Delta _ { \mathrm { R L V R } , t } \rangle = \langle R \operatorname { R e L U } ( R r _ { t } ) u _ { t } , R u _ { t } \rangle
$$

$$
= R ^ { 2 } \operatorname { R e L U } ( R r _ { t } ) \| u _ { t } \| ^ { 2 }
$$

$$
= \mathrm { R e L U } ( R r _ { t } ) \| u _ { t } \| ^ { 2 } \geq 0 .\tag{8}
$$

For sampled-token OPD, using (4) gives

$$
\langle \Delta _ { \mathrm { O P D } , t } , \Delta _ { \mathrm { R L V R } , t } \rangle = \langle r _ { t } u _ { t } , R u _ { t } \rangle = r _ { t } R \Vert u _ { t } \Vert ^ { 2 } ,\tag{9}
$$

which is negative when $r _ { t } R < 0$

□

The condition $r _ { t } R < 0$ corresponds exactly to the two conflict cases identified earlier: (i) a correct trajectory with $\pi _ { \boldsymbol { \theta } } > \pi _ { T }$ , and (ii) an incorrect trajectory with $\pi _ { T } > \pi _ { \theta }$ . In both cases, OPD pushes the policy in a direction that reduces the verifier reward.

## A.2. Decomposition of the OPD Gradient

The gated mechanism can be interpreted as explicitly removing the verifier-conflicting component from the OPD gradient.

Proposition A.2 (Conflict removal). The sampled-token OPD update can be decomposed as

$$
\Delta _ { \mathrm { O P D } , t } = \Delta _ { \mathrm { O P D V R } , t } + \Delta _ { \mathrm { c o n f i c t } , t } ,\tag{10}
$$

where

$$
\Delta _ { \mathrm { c o n f i c t } , t } = r _ { t } { \bf 1 } ( r _ { t } R < 0 ) u _ { t } .\tag{11}
$$

Moreover, the conflict term always has a non-positive projection onto the verifier gradient:

$$
\langle \Delta _ { \mathrm { c o n f i c t } , t } , \Delta _ { \mathrm { R L V R } , t } \rangle = - | r _ { t } | \mathbf { 1 } ( r _ { t } R < 0 ) \| u _ { t } \| ^ { 2 } \leq 0 .\tag{12}
$$

Proof. When $r _ { t } R \geq 0$ , we have $R \mathrm { R e L U } ( R r _ { t } ) = r _ { t }$ , so $\Delta _ { \mathrm { O P D V R } , t } = r _ { t } u _ { t }$ and the conflict term vanishes. When $r _ { t } R < 0$ , we have $\mathrm { R e L U } ( R r _ { t } ) = 0$ , so $\Delta _ { \mathrm { O P D V R } , t } = 0$ and the conflict term equals $\Delta _ { \mathrm { O P D } , t }$ . The projection onto $\Delta _ { \mathrm { R L V R } , t }$ follows directly. □

Thus OPDVR is not an arbitrary modification of OPD; it is precisely OPD with the harmful verifieropposing component removed.

## A.3. A Simplified Token-Level Analysis: OPDVR Can Strictly Outperform the Teacher

To illustrate the mechanism while retaining the token-level structure of LLM generation, we consider a simplified single-token decision problem. Let s be a fixed prefix. At this prefix, the model must choose between two candidate next tokens: $a _ { 1 }$ denotes a correct key token that leads to a correct final answer, and $a _ { 2 }$ denotes an incorrect key token that leads to a wrong final answer. The trajectory-level verifier reward is therefore $R ( a _ { 1 } ) = + 1$ and $R ( a _ { 2 } ) = - 1$

Let $\pi _ { \theta } ( a _ { 1 } \mid s ) = q$ and $\pi _ { T } ( a _ { 1 } \mid s ) = p$ . Assume the teacher is suboptimal, i.e., $\begin{array} { r } { p < { \frac { 1 } { 2 } } } \end{array}$ , and that the initial student policy is already better than the teacher, i.e., $q _ { 0 } > p$

Under sampled-token OPD, the expected loss reduces to the reverse KL divergence

$$
\mathcal { L } _ { \mathrm { O P D } } = D _ { \mathrm { K L } } ( \pi _ { \boldsymbol { \theta } } ( \cdot  { \left| \ s \right) \left\| \pi _ { T } ( \cdot  { \left| \ s \right) } \right) } ,\tag{13}
$$

whose global minimizer is $\pi _ { \theta } ( \cdot \mid s ) = \pi _ { T } ( \cdot \mid s )$ . Hence the OPD optimum satisfies $q = p ,$ with expected verifier reward

$$
J _ { \mathrm { O P D } } = 2 p - 1 .\tag{14}
$$

Under OPDVR, the expected update for token $a _ { 1 }$ is

$$
R ( a _ { 1 } ) \mathrm { R e L U } \Bigg ( R ( a _ { 1 } ) \log \frac { p } { q _ { 0 } } \Bigg ) = \mathrm { R e L U } \Bigg ( \log \frac { p } { q _ { 0 } } \Bigg ) = 0 ,\tag{15}
$$

because log $\cdot ( p / q _ { 0 } ) < 0$ since $q _ { 0 } > p$ . Similarly, for token $a _ { 2 }$ we have $R ( a _ { 2 } ) = - 1$ and

$$
\log { \frac { 1 - p } { 1 - q _ { 0 } } } > 0\tag{16}
$$

since $q _ { 0 } > p ,$ , so the OPDVR update for $a _ { 2 }$ is

$$
R ( a _ { 2 } ) \mathrm { R e L U } \bigg ( R ( a _ { 2 } ) \log \frac { 1 - p } { 1 - q _ { 0 } } \bigg ) = - \mathrm { R e L U } \bigg ( - \log \frac { 1 - p } { 1 - q _ { 0 } } \bigg ) = 0 .\tag{17}
$$

Thus the expected OPDVR gradient vanishes, and the policy remains at $q _ { 0 }$ . Consequently,

$$
J _ { \mathrm { O P D V R } } = 2 q _ { 0 } - 1 > 2 p - 1 = J _ { \mathrm { O P D } } = J _ { \mathrm { t e a c h e r } } .\tag{18}
$$

Therefore, OPDVR strictly outperforms both standard sampled-token OPD and the teacher policy.

## B. Hyperparameters

We provide the detailed hyperparameter configurations used in our experiments in Table 5. All models are trained using the Verl (Sheng et al., 2025) framework with the settings specified below.

Table 5: Hyperparameter settings.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td>1e-6</td></tr><tr><td>Train Batch Size</td><td>256</td></tr><tr><td>PPO Mini-Batch Size Max Response Length</td><td>256</td></tr><tr><td>Max Prompt Length</td><td>8192 1024</td></tr><tr><td>Rollout Temperature</td><td>1.0</td></tr><tr><td>Evaluation Temperature</td><td></td></tr><tr><td></td><td>0.7</td></tr><tr><td>Evaluation Top-p</td><td>0.95</td></tr></table>

## C. Hardware Setup

All experiments in this paper are conducted on NVIDIA GeForce RTX 5090 GPUs.