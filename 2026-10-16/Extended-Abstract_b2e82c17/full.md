# Extended Abstract

Motivation Training Large Language Models (LLMs) for multi-step arithmetic reasoning remains a significant challenge due to sparse terminal rewards and the lack of intermediate guidance. In tasks like Countdown, where models must generate sequential operations to reach a target number, rewards are only allocated at the sequence’s end. Standard reinforcement learning approaches assign credit solely to the final step, leaving intermediate reasoning steps without meaningful learning signals. This project addresses this critical bottleneck: the fundamental mismatch between sparse terminal rewards and the dense, step-wise supervision required to learn effective sequential strategies and intermediate representations.

Method To overcome this, we propose a Bellman-consistent trajectory RL framework with three core objectives: (1) transforming sparse terminal rewards into dense token-level supervision via Trajectory Bellman Residual Minimization (TBRM); (2) reducing gradient variance during optimization using a Reinforce Leave-One-Out (RLOO) or Group Relative Policy Optimization (GRPO) strategy; and (3) enabling effective intermediate step learning via hindsight goal relabeling. We evaluate our framework on the Countdown task, conducting ablation studies to quantify each component’s contribution to reasoning accuracy and training stability.

Implementation Our method builds on TBRM, utilizing soft reward and Q-functions to provide analytical solutions that allow a single LLM to serve as the Q, V, and policy functions, significantly reducing optimization complexity. The soft performance difference lemma guarantees that TBRM satisfies the optimal BRM solution, preserving token-level consistent optimization for the final reward. Additionally, we prove that the RLOO grouping strategy considerably reduces the variance of soft Bellman residuals, enhancing training stability. Finally, hindsight goal relabeling is applied to failure samples to strengthen intermediate reasoning steps. We evaluate our approach on the Countdown task, benchmarking against standard policy gradient methods and ablated baselines.

Results The TBRM-RLOO framework significantly reduces gradient variance compared to vanilla TBRM or TBRM-GRPO, leading to superior training stability and performance, with hindsight goal relabeling further accelerating learning efficiency. TBRM-RLOO improves rollout accuracy from 0.61 to 0.73 (+20%), Pass@1 from 0.55 to 0.56, and Pass@16 from 0.74 to 0.84 (+14%). Incorporating goal relabeling further boosts Pass@1 to 0.64. RLOO grouping consistently outperforms the GRPO strategy. Compared to SFT, IPO/DPO, RLOO, and vanilla TBRM, our method achieves the top overall Pass@16 performance at 84.0%, outperforming the RLOO baseline by 10.0% (absolute) and SFT by 8.0%. It also exhibits superior single-sample efficiency, dominating the low-compute regime (56.1-64.0% Pass@1 within 100 steps), and successfully circumvents the 72.0% plateau encountered by existing methods via test-time scaling.

Discussion Our results demonstrate that a Bellman-consistent trajectory RL method combined with RLOO variance reduction and hindsight goal relabeling significantly improves LLM performance on multi-step arithmetic tasks. The TBRM framework effectively densifies sparse rewards into token-level supervision, while the RLOO strategy stabilizes training by mitigating optimization variance stemming from prompt or problem difficulty. Hindsight goal relabeling enriches data and accelerates intermediate step learning. While constrained by model complexity and broader task generalization, our findings suggest that this framework provides a robust blueprint for training LLMs on complex, sparsely rewarded reasoning tasks.

Conclusion We introduced a Bellman-consistent trajectory RL method that successfully transforms sparse rewards into dense token-level supervision via TBRM, stabilizes optimization via RLOO variance reduction, and enhances sample efficiency through hindsight goal relabeling. Our framework achieves top-tier results on the Countdown task, balancing exploration and exploitation while successfully bypassing traditional performance plateaus. Future work will investigate the scalability of this approach across diverse multi-step reasoning domains.

# Bellman-Consistent Trajectory RL for LLM Arithmetic Reasoning

Lianfa Li<sup>1,2</sup>

<sup>1</sup>Department of Computer Science, Stanford University <sup>2</sup>LREIS, Institute of Geographical Sciences and Natural Resources Research, Chinese Academy of Sciences lianfali@stanford.edu, lilf@lreis.ac.cn

## Abstract

To address sparse terminal rewards in LLM arithmetic reasoning, we propose a Bellman-consistent trajectory RL method. Our framework introduces Trajectory Bellman Residual Minimization (TBRM) to transform sparse rewards into dense token-level signals. We integrate a Reinforce Leave-One-Out (RLOO) group strategy to reduce gradient variance of Bellman residuals and use hindsight goal relabeling to accelerate intermediate learning. Evaluated on the Countdown task, our method alleviates optimization bottlenecks and stabilizes gradient variance significantly better than GRPO. The TBRM-RLOO framework boosts Pass@1 to 0.55–0.64 and achieves a state-of-the-art Pass@16 accuracy of 76.0–84.0%, outperforming the SFT, IPO/DPO, and RLOO baselines. Our approach demonstrates exceptional single-sample efficiency and successfully circumvents standard performance plateaus via test-time scaling.

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a broad spectrum of natural language processing tasks. However, multi-step arithmetic reasoning remains a significant frontier and a core challenge (Li et al., 2025). Tasks such as the Countdown game require the model to plan, execute, and verify a precise sequence of arithmetic operations to arrive at a target number (Herr et al., 2025; Katz et al., 2025; Lian, 2025). In these complex, multi-step settings, Reinforcement Learning (RL) has emerged as a core paradigm for enhancing reasoning capabilities via reward-driven optimization (Ouyang et al., 2022a; Shao et al., 2024; Guo et al., 2025). Existing approaches typically fall into two categories: online policy-gradient methods (e.g., REINFORCE, PPO, RLOO) that optimize token-level policies directly from trajectory rollouts, and preference-based optimization methods (e.g., DPO, IPO) that align models offline using pairwise critique datasets.

Despite these algorithmic advances, training LLMs for sequential arithmetic reasoning is bottlenecked by three major challenges:

1. Sparse Terminal Rewards vs. Dense Supervision: Standard RL approaches assign credit only at the final step based on the correctness of the final output. This leaves intermediate reasoning tokens without local, step-wise learning signals, making it difficult for the model to learn effective intermediate representations (Lightman et al., 2024).

2. High Gradient Variance: Large action spaces and the structural inconsistency between policy and parameter spaces—both inherent to token-level generation—lead to severe training instability and high gradient variance during online trajectory rollouts (Schulman et al., 2017).

3. Poor Sample Efficiency: Standard online RL requires a prohibitive volume of environmental interactions, heavily taxing computational resources when high-quality trajectories are rare (Shao et al., 2024).

To address these fundamental limitations, this project introduces a Bellman-consistent trajectory RL method tailored for LLM arithmetic reasoning. Our framework leverages Trajectory Bellman Residual Minimization (TBRM) (Yuan et al., 2026) to transform sparse terminal rewards into dense, token-level supervision signals. By shifting the optimization objective from individual transitions to the entire rollout, TBRM provides a value-based method for reasoning that operates effectively with just a single rollout, bypassing the need for separate critic networks or complex importance ratios.

To stabilize training dynamics and maximize data utility, we integrate a grouping statistical strategy drawing on Reinforce Leave-One-Out (RLOO) (Ahmadian et al., 2024) or Group Relative Policy Optimization (GRPO) (Schulman et al., 2017), which drastically reduces variance during gradient optimization via multi-sample group baselines. Finally, we employ hindsight goal relabeling (Andrychowicz et al., 2017) to replay high-reward historical trajectories, accelerating the learning of intermediate reasoning steps from sparse feedback.

We evaluate our proposed Bellman-consistent trajectory RL framework on the rigorous Countdown task. Empirical results demonstrate that our method significantly alleviates optimization bottlenecks and stabilizes gradient variance compared to SFT, IPO/DPO, and RLOO baselines. Specifically, our approach boosts Pass@1 to 55.0–64.0% and achieves a state-of-the-art Pass@16 accuracy of 76.0– 84.0%. This demonstrates exceptional single-sample efficiency—driven by TBRM and hindsight goal relabeling—while successfully circumventing standard performance plateaus via test-time scaling. Ultimately, this framework provides a good blueprint for training LLMs on complex, sparsely rewarded reasoning tasks, with potential applications extending well beyond arithmetic to broader multi-step reasoning domains.

## 2 Related Work

## 2.1 RLHF and Preference Alignment for Mathematical Reasoning

Reinforcement Learning from Human (or AI) Feedback has become the standard for aligning LLMs with complex downstream tasks (Ouyang et al., 2022b). Early frameworks relied primarily on tokenlevel policy gradient methods like REINFORCE (Williams, 1992) and Proximal Policy Optimization (PPO) (Schulman et al., 2017). While PPO optimizes token policies directly from online trajectory rollouts, it fundamentally suffers from a lack of intermediate feedback and high gradient variance during long rollouts, especially when reward signals are sparse.

To circumvent the instability and computational overhead of training separate online reward and critic models, preference-based optimization methods such as Direct Preference Optimization (DPO) (Rafailov et al., 2023) and Identity Policy Optimization (IPO)(Ahmadian et al., 2024) have been introduced. These methods bypass online sampling entirely by directly optimizing a preference loss over an offline dataset of pairwise comparisons. However, because offline preference methods lack exploratory online rollouts, they can struggle to discover novel, valid mathematical paths in challenging reasoning tasks.

More recently, group-based multi-sample baseline methods such as Reinforce Leave-One-Out (RLOO) (Ahmadian et al., 2024) and Group Relative Policy Optimization (GRPO) (Shao et al., 2024) have gained traction. These algorithms reduce policy gradient variance by computing advantages relative to a group of outputs sampled from the same prompt, stabilizing training without requiring a separate critic network. However, they still inherently rely on sparse terminal rewards and lack native token-level credit assignment. In a different direction, Zhu et al. (2025) propose FlowRL, which replaces reward maximization with flow-based distribution matching, learning to balance the full reward distribution across trajectories rather than optimizing a scalar signal directly.

## 2.2 Bellman Residual Minimization

Value-based reinforcement learning relies heavily on the Bellman equation to propagate rewards backward through time (Baird, 1995; Sutton and Barto, 1998). Traditional Bellman Residual Minimization (BRM) aims to minimize the squared error of the Bellman equation across individual transitions (Antos et al., 2008). However, classical BRM is severely limited by the fundamental double sampling problem under non-linear transitions (Dann et al., 2014). Specifically, obtaining an unbiased gradient of the Bellman residual requires two independent samples of the next state from the same current state-action pair. In environments with non-linear transitions or vast token action spaces, single-sample estimates become heavily biased, which severely limits the stability of stochastic residual optimization (Baird, 1995; Dann et al., 2014).

Our work departs from traditional BRM by utilizing Trajectory Bellman Residual Minimization (TBRM) that has been proposed in Yuan et al. (2026). Instead of minimizing errors over isolated transitions, TBRM shifts the objective to the scale of the entire rollout trajectory. This formulation allows us to derive a simple, value-based method explicitly tuned for LLM reasoning text. By evaluating consistency across the complete trajectory, TBRM effectively sidesteps the double sampling problem, allowing for stable optimization using only a single rollout sample per prompt without a dedicated critic network.

## 2.3 Hindsight Goal Relabeling

Originally introduced in goal-conditioned reinforcement learning, Hindsight Goal Relabeling (HGR) addresses the challenge of sparse rewards by treating failed trajectories as successful rollouts for alternative goals (Andrychowicz et al., 2017). By dynamically relabeling unsuccessful attempts with the actually achieved states, HGR transforms counterfactual experiences into valuable training data, thereby maximizing sample efficiency.

In this work, we adapt HGR to the domain of multi-step arithmetic reasoning in LLMs. Specifically, when a model generates an incorrect final answer, we relabel the intermediate multi-step derivation with its actual mathematical outcome, effectively converting a failed rollout into a successful demon stration for a modified problem context. This mechanism enables the model to extract meaningful gradient signals from its own errors, accelerating the optimization of intermediate reasoning steps when the ground-truth reward is highly sparse. While HGR is heavily utilized in continuous control and robotic manipulation, its application to structured language generation and complex reasoning tasks is a novel extension that yields significant gains in convergence speed and final task accuracy.

## 3 Method

The proposed method introduces a unified value-based framework for reinforcing multi-step language model reasoning without relying on separate critic architectures or complex importance sampling ratios. As illustrated in Figure 1, the framework consists of three tightly integrated components:

1) Trajectory Bellman Residual Minimization (TBRM): A value-based objective that maps token logits directly to action-values, converting sparse terminal signals into dense tokenlevel supervision while preserving token-level consistency.

2) Reinforce Leave-One-Out (RLOO) Grouping Strategy: A variance-reduction technique tailored to group-wise Bellman residuals to stabilize policy updates.

3) Hindsight Goal Relabeling (HGR): An experience augmentation strategy that transforms failed rollouts into high-value successful trajectories to accelerate intermediate learning.

## 3.1 Trajectory Bellman Residual Minimization (TBRM)

Under the Kullback-Leibler (KL) regularized reinforcement learning paradigm with a reference policy $\pi _ { \mathrm { r e f } } .$ , the TBRM objective maximizes the regularized trajectory-level return (Yuan et al., 2026):

$$
\begin{array} { r l } & { J _ { \beta } ( \pi ) = \beta \mathbb { E } _ { \tau \sim \pi } \left[ R _ { \beta } ( \tau ) - \log \pi ( \tau ) \right] } \\ & { \quad \quad \quad = \beta \mathbb { E } _ { \tau \sim \pi } \left[ \displaystyle \sum _ { h = 1 } ^ { H } \left( R _ { \beta } ( s _ { h } , a _ { h } ) - \log \pi ( a _ { h } | s _ { h } ) \right) \right] } \end{array}\tag{1}
$$

where $\beta > 0$ is the temperature hyperparameter, and $R _ { \beta } ( s _ { h } , a _ { h } )$ is the token-level soft reward function:

$$
R _ { \beta } ( s _ { h } , a _ { h } ) = \frac { r ( s _ { h } , a _ { h } ) } { \beta } + \log \pi _ { \mathrm { r e f } } ( a _ { h } | s _ { h } )\tag{2}
$$

![](images/740f118f720f8d809d54e3f77496814cd63a0bf111b0236393ffa7af7d91a93e.jpg)  
Figure 1: Overview of the Bellman-consistent trajectory RL framework, detailing the coupling of Trajectory Bellman Residual Minimization (TBRM) with group-wise RLOO baseline adjustment and hindsight goal relabeling.

The optimal policy maximizing the regularized objective, $\pi _ { \beta } ^ { * } = \arg \operatorname* { m a x } _ { \pi } J _ { \beta } ( \pi )$ , can be derived analytically in closed form as a reweighted softmax distribution. Depending on whether the reference policy is explicitly decoupled or implicitly absorbed, the solution can be equivalently expressed in two standard forms (Yuan et al., 2026):

$$
\pi _ { \beta } ^ { * } ( a _ { h } | s _ { h } ) \propto \pi _ { \mathrm { r e f } } ( a _ { h } | s _ { h } ) \exp \left( Q _ { r } ^ { * } ( s _ { h } , a _ { h } ) / \beta \right) \mathrm { o r } \pi _ { \beta } ^ { * } ( a _ { h } | s _ { h } ) \propto \exp \left( Q _ { R _ { \beta } } ^ { * } ( s _ { h } , a _ { h } ) \right) .
$$

where $Q _ { R _ { \beta } } ^ { * } ( s _ { h } , a _ { h } ) = Q _ { r } ^ { * } ( s _ { h } , a _ { h } ) + \beta \log \pi _ { \mathrm { r e f } } ( a _ { h } | s _ { h } )$ represents the optimal action-value function. Consequently, the soft Bellman operator $\mathcal { T } _ { \beta }$ for the $Q$ function can be expressed in closed form as:

$$
\begin{array} { c l } { ( \mathcal { T } _ { \beta } Q ) ( s , a ) : = R _ { \beta } ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } \sim \mathcal { P } ( \cdot \vert s , a ) } \left[ V _ { Q } ( s ^ { \prime } ) \right] } \\ { = R _ { \beta } ( s , a ) + \gamma \log \displaystyle \sum _ { a ^ { \prime } \in \mathcal { A } } \exp \left( Q ( s ^ { \prime } , a ^ { \prime } ) \right) } \end{array}\tag{3}
$$

where $s ^ { \prime }$ is the deterministic next state resulting from taking action a in state $s ,$ and $\gamma$ is the discount factor (often set to 1 in finite-horizon token generation).

Under the autoregressive language modeling setting, state transitions are deterministic: the state $s _ { h }$ consists of the prompt concatenated with all generated tokens up to step $h - 1$ , and the action $a _ { h }$ is the next token in the vocabulary ${ \mathcal { A } } .$ . We parameterize the action-value function $Q _ { \theta }$ , the value function $V _ { \theta }$ and the resulting policy $\pi _ { \theta }$ directly via the model’s unnormalized log-probabilities (logits) without an independent critic network:

$$
Q _ { \theta } ( s , a ) : = \log \operatorname * { i t } _ { \theta } ( s , a ) ; V _ { \theta } ( s ) : = \log \sum _ { a ^ { \prime } \in \mathcal { A } } \exp ( \log \operatorname * { i t } _ { \theta } ( s , a ^ { \prime } ) ) ; \log \pi _ { \theta } ( a | s ) = Q _ { \theta } ( s , a ) - V _ { \theta } ( s )
$$

where $V _ { \theta } ( s )$ is the log-sum-exp operator over the token vocabulary.

In deterministic environments, classical Bellman Residual Minimization (BRM) minimizes the squared per-step temporal difference error over a trajectory dataset D:

$$
\begin{array} { l } { { \displaystyle { \mathcal E } _ { \mathcal D } ^ { \mathtt { B R M } } ( \theta ) = \frac { 1 } { | \mathcal D | } \sum _ { \tau \in \mathcal D } \sum _ { h = 1 } ^ { H } \left( \log \mathrm { i } _ { \boldsymbol \ell } ( s _ { h } , a _ { h } ) - ( \mathcal T _ { \boldsymbol \beta } \log \mathrm { i } _ { \boldsymbol \ell } ) ( s _ { h } , a _ { h } ) \right) ^ { 2 } } } \\ { { \displaystyle = \frac { 1 } { | \mathcal D | } \sum _ { \tau \in \mathcal D } \sum _ { h = 1 } ^ { H } \left( \log \mathrm { i } _ { \boldsymbol \ell } ( s _ { h } , a _ { h } ) - \frac { r \left( s _ { h } , a _ { h } \right) } { \beta } - \log \pi _ { \mathrm { r e f } } ( a _ { h } | s _ { h } ) - \log \sum _ { a ^ { \prime } \in \mathcal A } \exp ( \log \mathrm { i } _ { \boldsymbol \ell } ( s _ { h + 1 } , a ^ { \prime } ) ) \right) ^ { 2 } } } \end{array}\tag{4}
$$

However, directly optimizing $\mathcal { L } _ { \mathcal { D } } ^ { \mathrm { B R M } } ( \theta )$ fails in language model post-training because the process reward $r ( s _ { h } , a _ { h } )$ is typically unavailable or extremely sparse, only appearing as a terminal signal $r ( \tau )$ at the final token $\mathbf { \bar { \boldsymbol { h } } } = \mathbf { \bar { \boldsymbol { H } } }$

To bypass this limitation, Yuan et al. (2026) lift the step-wise residual to a trajectory-level objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { D } } ^ { \mathrm { T B R M } } ( \theta ) = \displaystyle \frac { 1 } { | \mathcal { D } | } \sum _ { \tau \in \mathcal { D } } \bigg ( \sum _ { h = 1 } ^ { H } [ \log \mathrm { i t } _ { \theta } ( s _ { h } , a _ { h } ) - ( \mathcal { T } _ { \beta } \mathrm { l o g i t } _ { \theta } ) ( s _ { h } , a _ { h } ) ] \bigg ) ^ { 2 } } \\ { = \displaystyle \frac { 1 } { | \mathcal { D } | } \sum _ { \tau \in \mathcal { D } } \bigg ( V _ { \theta } ( s _ { 1 } ) - \frac { r ( \tau ) } { \beta } + \sum _ { h = 1 } ^ { H } \log \frac { \pi _ { \theta } ( a _ { h } | s _ { h } ) } { \pi _ { \mathrm { r e f } } ( a _ { h } | s _ { h } ) } \bigg ) ^ { 2 } } \end{array}\tag{5}
$$

TBRM eliminates the need for intermediate token-level reward labeling or value scoring, using only the final trajectory outcome $r ( \tau )$ . Crucially, the formulation satisfies the solution of the original token-level BRM framework, maintaining token-level consistent optimization for the final reward while propagating the sparse terminal reward back into dense token-level supervision signals.

The theoretical validity of optimizing the policy via trajectory-level residuals is justified by the soft performance difference lemma. Let $\begin{array} { r } { \hat { \pi } \big ( a | s \big ) = \frac { \exp \left( Q \left( s , a \right) \right) } { \sum _ { a ^ { \prime } \in \mathcal { A } } \exp \left( Q \left( s , a ^ { \prime } \right) \right) } } \end{array}$ be the soft-greedy policy induced by the learned value function. The suboptimality gap between the globally optimal policy $\pi ^ { * }$ and πˆ is strictly upper-bounded by the expected trajectory Bellman residual:

$$
J _ { \beta } ( \pi ^ { * } ) - J _ { \beta } ( \hat { \pi } ) \leq 2 \beta \operatorname* { m a x } _ { \pi \in \{ \pi ^ { * } , \hat { \pi } \} } \left| \mathbb { E } _ { \pi } \left[ \sum _ { h = 1 } ^ { H } ( Q ( s _ { h } , a _ { h } ) - ( { \mathcal T } _ { \beta } Q ) ( s _ { h } , a _ { h } ) ) \right] \right|\tag{6}
$$

The soft performance difference lemma establishes that minimizing the TBRM objective directly compresses the suboptimality gap, mathematically guaranteeing policy convergence toward the global optimum from arbitrary off-policy data. In this work, we build directly upon the TBRM framework introduced by Yuan et al. (2026); we refer the reader to their work for expanded theoretical derivations.

## 3.2 Reinforce Leave-One-Out (RLOO) Grouping Strategy

While policy-gradient methods like GRPO utilize group-relative reward normalization to mitigate gradient variance, value-based trajectory objectives require variance control centered around their residual errors. We introduce a modified Reinforce Leave-One-Out (RLOO) grouping strategy that leverages collective prompt statistics. Crucially, the adjustment is performed using Bellman residuals rather than raw rewards.

For each unique prompt group $g \in \{ 1 , \ldots , G \}$ , we sample $b \in \{ 1 , \ldots , n _ { g } \}$ parallel trajectory rollouts. We define the raw trajectory Bellman error $E _ { b , g }$ for response b within group g as:

$$
E _ { b , g } = V _ { \theta } ( s _ { 1 } ^ { ( b , g ) } ) - \frac { r ( \tau ^ { ( b , g ) } ) } { \beta } + \sum _ { h = 1 } ^ { H } \log \frac { \pi _ { \theta } ( a _ { h } ^ { ( b , g ) } | s _ { h } ^ { ( b , g ) } ) } { \pi _ { \mathrm { r e f } } ( a _ { h } ^ { ( b , g ) } | s _ { h } ^ { ( b , g ) } ) }\tag{7}
$$

To minimize variance without introducing out-of-sample bias, each rollout’s residual is adjusted against a counterfactual baseline computed from the remaining samples within its prompt group:

$$
\tilde { E } _ { b , g } = E _ { b , g } - \frac { 1 } { n _ { g } - 1 } \sum _ { i \neq b } E _ { i , g }\tag{8}
$$

This baseline measures the residual variance relative to the group mean, thereby facilitating more stable optimization of the neural network. To prevent optimization drifting while maintaining low variance, we optimize a combinational loss function that penalizes both group-relative and absolute residuals:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \alpha \cdot { \mathcal { L } } _ { \mathrm { R L O O } } + ( 1 - \alpha ) \cdot { \mathcal { L } } _ { \mathrm { a b s o l u t e } }\tag{9}
$$

where $\mathcal { L } _ { \mathrm { { R L O O } } }$ and $\mathcal { L } _ { \mathrm { a b s o l u t e } }$ are defined over a prompt batch size $B$ and group size $G$ as:

$$
\mathcal { L } _ { \mathrm { R L O O } } = \frac { 1 } { B \cdot G } \sum _ { b = 1 } ^ { B } \sum _ { g = 1 } ^ { G } ( \tilde { E } _ { b , g } ) ^ { 2 }\tag{10}
$$

$$
{ \mathcal { L } } _ { \mathrm { a b s o l u t e } } = { \frac { 1 } { B \cdot G } } \sum _ { b = 1 } ^ { B } \sum _ { g = 1 } ^ { G } ( E _ { b , g } ) ^ { 2 }\tag{11}
$$

The weighting hyperparameter α is tuned within the stable regime of [0.8, 0.99]. By standardizing optimization over group-relative residuals rather than rewards, this TBRM-RLOO configuration structurally out-performs GRPO-enhanced implementations of TBRM.

## 3.3 Hindsight Goal Relabeling

In complex multi-step arithmetic tasks, random generation during early post-training phases leads to a high frequency of failed rollouts. These failed trajectories yield persistent zero rewards, leading to uninformative gradient updates and slow optimization. To alleviate this sample inefficiency, we integrate Hindsight Goal Relabeling (HGR).

HGR systematically converts failed trajectories into valuable learning opportunities via the following sequence:

1) Extraction: Trajectories that fail to satisfy the original terminal condition or target goal are filtered from the active generation batch.

2) Relabeling: The actual outcome produced by the model’s multi-step derivation is evaluated. The target goal within the initial prompt context is then ex-post relabeled to match this actual achieved outcome, effectively converting a failed rollout into a successful demonstration for a modified task instance.

3) Augmentation: The newly relabeled valid samples are combined with the original successful trajectories, inflating the coverage of the active replay buffer.

The augmented dataset is passed directly into the group TBRM-RLOO optimization pipeline, ensuring that the model extracts meaningful token-level supervision signals even from sub-optimal rollouts, drastically increasing sample efficiency.

## 3.4 β Hyperparameter and Algorithmic Procedure

The temperature hyperparameter $\beta$ controls the trade-off between maximizing the raw reward signal and remaining regularized against the reference policy $\pi _ { \mathrm { r e f } } .$

• High β Regularization: Excessively high values over-dampen the reward scale and minimize the impact of optimization updates, leading to slow training velocity, objective instability, and failure to reach convergence.

• Low β Regularization: Overly aggressive low values amplify the reward scale and optimization updates, causing the policy to overfit rapidly to sparse signals, collapsing the distribution variance and degrading generation quality.

For intensive token-level arithmetic reasoning tasks such as the Countdown math puzzle, configuring β within the range of [0.01, 0.02] provides the optimal boundary to maximize learning velocity while preventing model collapse.

The complete training sequence is formalized in Algorithm 1.

Algorithm 1 Bellman-Consistent Group TBRM-RLOO with Hindsight Relabeling   
Require: Task dataset D, reference model $\pi _ { \mathrm { r e f } }$ with weights $\theta _ { \mathrm { r e f } } .$ temperature ${ \overline { { \beta } } } ,$ combination coeffi   
cient $\alpha ,$ group rollout size $G ,$ iterations $T ,$ , group rollout size $G .$   
1: Initialize model weights $\theta  \theta _ { \mathrm { r e f } }$   
2: Initialize global step: global\_step $ 0$   
3: Initialize storage: $\bar { \mathcal { D } } _ { \mathrm { b a t c h } }  \emptyset$ and ${ \mathcal { D } } _ { \mathrm { r e l a b e l } }  \emptyset$   
4: while global\_step $< T$ do   
▷ Phase 1: Rollout Generation & Hindsight Relabeling   
5: $\mathbf { i f } \left| \mathcal { D } _ { \mathrm { b a t c h } } \right| = 0$ and $| \mathcal { D } _ { \mathrm { r e l a b e l } } | = 0$ then   
6: if D iterator is exhausted then   
7: Reset dataset iterator $\mathcal { D }$   
8: end if   
9: $\mathcal { D } _ { \mathrm { b a t c h } } $ next batch from $\mathcal { D }$   
10: for each prompt $q \in \mathcal { D } _ { \mathrm { b a t c h } }$ do   
11: Generate $G$ parallel rollouts $\tau ^ { ( 1 ) } , \dots , \tau ^ { ( G ) } \sim \pi _ { \theta } ( \cdot | q )$   
12: for each rollout $\tau ^ { ( i ) }$ do   
13: $\mathbf { i f } \tau ^ { ( i ) }$ fails target goal $g _ { \mathrm { o r i g } }$ then   
14: Extract actual outcome $g _ { \mathrm { a c t u a l } }$ from terminal state   
15: Relabel prompt $q  q _ { \mathrm { n e w } }$ using g<sub>actual</sub>   
16: Collect to relabel pool: ${ \mathcal { D } } _ { \mathrm { r e l a b e l } }  { \mathcal { D } } _ { \mathrm { r e l a b e l } } \cup \{ \tau _ { \mathrm { r e l a b e l e d } } ^ { ( i ) } \}$   
17: end if   
18: end for   
19: end for   
▷ Phase 2: Train on Standard Batch Data   
20: else if $| \mathcal { D } _ { \mathrm { b a t c h } } | > 0$ then   
21: Rollout $G$ generation for each prompt in $\mathcal { D } _ { \mathrm { b a t c h } }$   
22: Compute Bellman residuals $E _ { b , g }$ and RLOO-adjusted residuals $\tilde { E } _ { b , g }$ for $\mathcal { D } _ { \mathrm { b a t c h } }$   
23: Evaluate combinational loss $\mathcal { L } _ { \mathrm { t o t a l } } = \alpha \mathcal { L } _ { \mathrm { R L O O } } + ( 1 - \alpha ) \mathcal { L } _ { \mathrm { a b s o l u t e } }$   
24: Update parameters: $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { t } }$ otal   
25: Clear batch data: $\mathcal { D } _ { \mathrm { b a t c h } }  \emptyset$   
26: Increment global step: global\_step ← global\_step + 1   
▷ Phase 3: Retrain using Hindsight Relabeled Data (Multi-step Pop)   
27: else if $| \mathcal { D } _ { \mathrm { b a t c h } } | = 0$ and $\mathcal { D } _ { \mathrm { r e l a b e l } } \neq \emptyset$ then   
28: Pop a mini-batch ${ \cal B } _ { \mathrm { r e l a b e l } } \subset { \cal D } _ { \mathrm { r e l a b e l } }$   
29: Rollout $G$ generation for each prompt in $\scriptstyle B _ { \mathrm { r e l a b e l } }$   
30: Compute Bellman residuals $E _ { b , g }$ and RLOO-adjusted residuals $\tilde { E } _ { b , g }$ for ${ \cal B } _ { \mathrm { r e l a b e l } }$   
31: Evaluate combinational loss $\mathcal { L } _ { \mathrm { r e l a b e l } } = \alpha \mathcal { L } _ { \mathrm { R L O O } } \overset { \cdot } { + } ( 1 - \alpha ) \mathcal { L }$ absolute   
32: Retrain/Update parameters: $\theta  \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { r e l a b e l } }$   
33: Remove trained samples from pool: ${ \mathcal { D } } _ { \mathrm { r e l a b e l } }  { \mathcal { D } } _ { \mathrm { r e l a b e l } } \backslash { \mathcal { B } } _ { \mathrm { r e l a b e l } }$   
34: end if   
35: end while

## 4 Experimental Setup

All fine-tuning experiments were executed on the Modal cloud platform (https://modal.com) utilizing NVIDIA H100 GPU instances. The software framework was implemented within a Conda environment using native PyTorch code, leveraging the Hugging Face transformers and datasets libraries. Depending on the optimization paradigm, two distinct model initializations were utilized. For SFT, IPO and DPO training configurations, the foundational base model was Qwen/Qwen2.5-0.5B. Conversely, for the online reinforcement learning methods—specifically RLOO, FlowRL, and our proposed framework—training commenced from the recommended finetuned checkpoint asingh15/qwen-sft-countdown-defaultproj available on Hugging Face.

The default Countdown pipeline integrates three distinct public datasets from Hugging Face, each serving a tailored role within our multi-phase training workflow. To initialize the model before alignment, a warm-start SFT phase is performed using the strategy dataset (https://huggingface. co/datasets/Asap7772/cog\_behav\_all\_strategies). For offline, preference-based optimization paradigms that contrast paired trajectories—such as DPO—we employ the pairwise preference dataset (https://huggingface.co/datasets/asingh15/countdown\_tasks\_3to4-dpo). Finally, online verifier-based reinforcement learning methods (such as RLOO, flowRL and TBRM) and overall tracking utilize the prompt and evaluation dataset (https://huggingface.co/datasets/ asingh15/countdown\_tasks\_3to4), maintaining its default training and testing partitions in the default project. Within this online RL regime, prompts are extracted from the dataset to generate par allel rollouts, which are subsequently evaluated and scored by a deterministic, rule-based countdown verifier.

During the optimization phase, models are generally trained for a maximum horizon of 200 steps. Model validation is performed continuously, and the optimal parameters are checkpointed and saved based on the highest achieved rollout mean accuracy. To mitigate overfitting and minimize computational waste, an early stopping criterion is enforced, which automatically terminates the training process if the rollout mean accuracy fails to improve for 20 consecutive evaluation iterations.

## 5 Results

## 5.1 Quantitative Evaluation

## 5.1.1 β Hyperparameter Exploration

The temperature hyperparameter, β, serves as a critical regularizer governing the trade-off between reward maximization and policy divergence. As illustrated in Figure 2, the magnitude of β dictates optimization stability. An excessively high β over-penalizes deviations from the reference policy, effectively dampening the relative reward scale and stalling optimization progress; under extreme settings, this over-regularization induces training instability and prevents convergence.

Conversely, a small β value amplifies the policy gradient updates by scaling up the effective reward variance, yet values that are too aggressive precipitating catastrophic policy collapse. For our specific countdown task, empirical tuning demonstrates that $\beta = 0 . 0 0 1$ consistently yields the optimal balance between exploration and stable policy updates.

![](images/34e565c1bef8258a55bd2c91f05ed8ffc050b7494be9af9ab74f166b4d8adcbc.jpg)  
Figure 2: Empirical evaluation of the impact of the temperature hyperparameter $\beta$ on optimization stability and reward scaling.

## 5.1.2 Learning Curve Analysis

We evaluated the training dynamics of four distinct variants: vanilla TBRM, TBRM-GRPO, TBRM-RLOO, and TBRM-RLOO enhanced with hindsight relabeling. The Bellman residual loss trajectories presented in Figure 3 reveal that vanilla TBRM exhibits severe high-variance oscillations and fails to establish a clear downward trend, underscoring the instability of unconstrained Bellman residual minimization.

In contrast, incorporating REINFORCE Leave-One-Out (RLOO) significantly stabilizes training, mitigating variance and producing a steady, monotonic decline in the residual loss. Furthermore, integrating hindsight relabeling (TBRM-RLOO-Hindsight) yields the most favorable optimization profile, characterized by minimized baseline loss and accelerated variance reduction compared to standard RLOO execution.

Notably, GRPO-TBRM induces an artificially suppressed loss magnitude. The group-relative advantage normalization inherent to GRPO over-flattens the gradient variance across samples, which ultimately stymies robust representation learning and leads to premature stagnation.

![](images/2a97888ebfa9fc75de3f7d214169f63b59b680cf9146f929326aba7c86fb1150.jpg)  
Figure 3: Comparison of Bellman residual loss learning curves across the four evaluated TBRM variants.

These optimization behaviors are closely mirrored by the roll-out reward scores during training (Figure 4). TBRM-RLOO with hindsight relabeling exhibits the most rapid and superior reward trajectory among all configurations. Additionally, the baseline TBRM-RLOO maintains a consistently higher reward profile than both the vanilla and GRPO-regularized variants, validating the efficacy of the variance-reduced policy gradient formulation.

![](images/ed36ea9615d58df1431e2bb03b90e4a79ab1932946714804dcc0159e5e117205.jpg)  
Figure 4: Roll-out reward score trajectories during training across different TBRM configurations.

## 5.1.3 Comprehensive Performance Comparison

Table 1 and Figure 5 summarize the global performance metrics across baselines and proposed methods evaluated under identical compute budgets.

![](images/dd7ed21606dde0b05b45016f052d7105eeb86762d5fb55cb037ccf9d00d069ac.jpg)  
Figure 5: Comprehensive Pass@k performance comparison across baseline architectures and extension models.

Table 1: Performance Comparison Across Baseline and Proposed Methods
<table><tr><td>Method</td><td>Rollout Accuracy*</td><td>Pass@1</td><td>Pass@3</td><td>Pass@6</td><td>Pass@10</td><td>Pass@16</td></tr><tr><td>SFT Baseline</td><td>一</td><td>0.33</td><td>0.55</td><td>0.65</td><td>0.71</td><td>0.76</td></tr><tr><td>IPO</td><td></td><td>0.40</td><td>0.62</td><td>0.69</td><td>0.71</td><td>0.72</td></tr><tr><td>DPO</td><td></td><td>0.45</td><td>0.61</td><td>0.67</td><td>0.70</td><td>0.72</td></tr><tr><td>RLOO</td><td>0.66</td><td>0.51</td><td>0.69</td><td>0.72</td><td>0.73</td><td>0.74</td></tr><tr><td>Extension Models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FlowRL</td><td>0.70</td><td>0.52</td><td>0.66</td><td>0.68</td><td>0.70</td><td>0.72</td></tr><tr><td>TBRM (reward)</td><td>0.61</td><td>0.55</td><td>0.66</td><td>0.70</td><td>0.72</td><td>0.74</td></tr><tr><td>TBRM-RLOO (100 step)</td><td>0.70</td><td>0.56</td><td>0.71</td><td>0.74</td><td>0.76</td><td>0.78</td></tr><tr><td>TBRM-RLOO (150 step)</td><td>0.73</td><td>0.46</td><td>0.69</td><td>0.77</td><td>0.81</td><td>0.84</td></tr><tr><td>TBRM-RLOO-Hindsight (130 step)</td><td>0.71</td><td>0.58</td><td>0.72</td><td>0.75</td><td>0.76</td><td>0.76</td></tr><tr><td>TBRM-RLOO-Hindsight (200 step)</td><td>0.71</td><td>0.64</td><td>0.71</td><td>0.72</td><td>0.73</td><td>0.74</td></tr></table>

∗ Reported highest accuracy reached during training.

Significant Gains over Vanilla TBRM: TBRM-RLOO (150 steps) markedly improves rollout accuracy from 0.61 to 0.73 (about 20% relative improvement) and Pass@16 from 0.74 to 0.84 (about 14% relative improvement) compared to vanilla TBRM, proving that structured variance reduction translates directly into better test-time generalization.

Superior Peak Generation Accuracy: TBRM-RLOO at 150 steps achieves the top overall multisample performance with a Pass@16 score of 84.0%. This outpaces the standalone RLOO baseline by 10.0% (absolute) and the supervised Fine-Tuning (SFT) baseline by 8.0% (absolute).

Single-Sample Compute Efficiency: Within low-compute regimes (Pass@1), TBRM-RLOO at 100 steps yields a strong performance of 56.1%, outperforming standard baseline models. This single-generation efficiency is only surpassed by configurations integrating hindsight relabeling; notably, TBRM-RLOO-Hindsight at 200 steps achieves a peak Pass@1 score of 64.0%, highlighting that hindsight trajectories significantly accelerate early-stage sample efficiency.

Surpassing the Alignment Ceiling: Traditional alignment frameworks (such as DPO and IPO) plateau early, capping performance near 72.0% on Pass@16. TBRM breaks through this apparent bottleneck, leveraging test-time trajectory modeling to continuously scale accuracy with extended training steps.

Crucially, the data indicates that hindsight relabeling acts as an effective catalyst during the initial phases of training, accelerating early reward acquisition and driving substantial gains in intermediate representation learning capacity.

## 5.2 Qualitative Analysis

To investigate the structural shifts in model behavior and the underlying reasoning patterns induced by our framework, we conducted a qualitative analysis on randomly sampled generation test cases from the evaluation set.

As illustrated by the representative countdown task instance in Figure 6, TBRM demonstrates a clear capacity to construct more concise, streamlined mathematical reasoning paths to arrive at the correct target value compared to standard baseline alternatives. Incorporating the hybrid TBRM-RLOO framework with hindsight relabeling also optimizes this trajectory, significantly mitigating logical redundancy and enabling the agent to satisfy numeric constraints using fewer intermediate operational steps.

In contrast, offline preference-based alignment methods like IPO and DPO generate highly redundant inference sequences. While online RLOO reduces the total number of predicted tokens, its outputs still retain substantial structural redundancy, though they remain markedly superior to the unaligned SFT baseline. Meanwhile, FlowRL produces shorter individual responses but frequently requires multiple circuitous internal attempts within its generation token sequence to ultimately converge on the correct answer.

These qualitative observations demonstrate that TBRM successfully propagates terminal reward signals backward to intermediate decision nodes. Furthermore, the integration of hindsight relabeling strengthens this credit assignment mechanism, optimizing step-by-step intermediate logic and effectively discouraging mathematically circuitous generation paths.

## 6 Discussion

This project introduces a Bellman-consistent trajectory reinforcement learning framework designed to enhance intermediate learning signals for large language model mathematical inference, specifically evaluated on the countdown task. By leveraging TBRM, our approach successfully backpropagates terminal reward signals to individual intermediate inference steps, thereby systematically optimizing the reasoning chain. Distinct from traditional token-level policy optimization methods, the theoretical foundation of TBRM relies on the soft performance difference lemma. This ensures that the framework converges toward a near-optimal solution by minimizing Bellman-consistent residuals, effectively transforming sparse, terminal rewards into dense, token-level credit assignments. Empirically, we observed that the temperature hyperparameter β plays a critical role in stabilizing policy optimization. Our behavioral analysis confirms that TBRM successfully prunes redundant reasoning steps, encouraging the model to discover concise and direct paths to the target value.

To address the inherent instability of policy gradient methods in language generation, we introduced two key enhancements:

• Robust Variance Reduction via RLOO: Incorporating the REINFORCE Leave-One-Out (RLOO) estimator to adjust Bellman residuals drastically reduces gradient variance. In mathematical reasoning tasks, context and prompt difficulty vary widely, typically causing severe reward fluctuations. RLOO effectively normalizes these baseline discrepancies across trajectories, establishing a stable, steadily declining loss curve.

• Sample Efficiency via Hindsight Experience Replay: Hindsight goal relabeling acts as a powerful data augmentation mechanism to reinforce suboptimal or failed trajectories. Our experimental results demonstrate a clear performance hierarchy: while vanilla TBRM suffers from high variance, and TBRM-RLOO stabilizes training, the integration of hindsight relabeling yields the fastest convergence, the lowest loss variance, and the higher overall reward curves.

![](images/44410f5869b90e3ffaa098bc0b9d93f46cdd65345cc17b7a49322f1fc0426e6e.jpg)  
Figure 6: Case study comparison of reasoning path generations for a representative countdown task.

## 6.1 Limitations and Future Work

Despite these promising results, several limitations remain to be addressed in future research. First, we observe a distinct trade-off in policy behavior over extended optimization horizons. As shown in Table 1, increasing the training steps of the RLOO-TBRM method from 100 to 150 steps results in a degradation in greedy decoding performance (Pass@1 dropping from 0.56 to 0.46) alongside a substantial improvement in exploratory coverage (Pass@16 rising from 0.78 to 0.84). This pattern indicates that while extended optimization enhances the diversity and maximum capacity of the sampled reasoning paths, it simultaneously degrades top-1 deterministic precision—potentially due to high policy variance or distribution drift during late-stage reinforcement learning. Investigating the underlying mechanics of this trade-off to preserve greedy precision while retaining wide coverage remains an immediate priority.

Furthermore, we observed a susceptibility to late-stage mode collapse when training the trajectorybased TBRM-RLOO-Hindsight configuration over extended horizons beyond 200 steps. This phenomenon is likely driven by the policy overfitting to synthetic, relabeled target goals. To mitigate this structural collapse, future iterations will explore dynamic curriculum scheduling mechanisms. Such schedules will gradually decay or balance the ratio between original task distributions and hindsight-relabeled trajectories as training progresses, preventing the policy from destabilizing on out-of-distribution synthetic objectives.

Hyperparameter sensitivity and architectural constraints also present opportunities for expansion. Currently, the local policy temperature parameter β is held static; however, given its heavy influence on optimization stability, transitioning β into an adaptive, dynamically learnable parameter warrants systematic exploration. Mechanistically, embedding this unified TBRM framework within a Mixture of Experts (MoE) architecture could allow the model to dynamically route and fuse heterogeneous loss objectives (e.g., balancing RLOO and trajectory Bellman residuals) conditioned on the complexity of the prompt. Finally, while the current empirical evaluation demonstrates strong validation on the Countdown math puzzle, expanding the benchmark scope to more complex, open-ended multistep reasoning environments such as GSM8K and MATH is essential to rigorously evaluate the cross-domain generalization and scalability of our framework.

## 7 Conclusion

This paper presented a Bellman-consistent trajectory reinforcement learning framework designed to address the pervasive challenge of sparse rewards in LLM-based mathematical reasoning. By leveraging the existing methodology of TBRM, we successfully map sparse terminal rewards into dense credit assignments along the inference trajectory. Grounded in the soft performance difference lemma, our framework builds upon the theoretical guarantees of TBRM—specifically, that minimizing the trajectory-level Bellman residual drives the policy toward a near-optimal solution. To overcome the optimization instabilities and sample inefficiencies that arise when applying this trajectory method to sequential language generation, we integrated the REINFORCE Leave-One-Out (RLOO) estimator for robust variance reduction alongside hindsight goal relabeling for data augmentation.

Our empirical evaluations demonstrate that the unified TBRM-RLOO-Hindsight framework significantly outperforms baseline token-level policy optimization methods, delivering accelerated convergence, gradient variance reduction, and a marked reduction in reasoning redundancy. While future work is required to alleviate late-stage mode collapse and extend validation to broader mathematical benchmarks such as GSM8K and MATH, this work validates the practical efficacy of trajectory-level credit assignment for language models. Future extensions exploring dynamically learnable hyperparameters and Mixture-of-Experts (MoE) architectures will further unlock the generalizability of Bellman-consistent policy optimization for complex, multi-step cognitive tasks.

## 8 Team Contributions

As the sole investigator, Lianfa Li is responsible for the execution and management of all aspects of this project, structured into four core operational phases:

• Environment Setup & Data Preparation: Configure the Countdown mathematical inference environment, curate training and evaluation datasets, and establish baseline evaluation setups for token-level policy optimization methods and standalone TBRM variants.

• Algorithmic Framework Design: Develop the unified Bellman-consistent trajectory reinforcement learning framework, combining the TBRM loss formulation with the REIN-

FORCE Leave-One-Out (RLOO) variance reduction estimator and a hindsight experience replay mechanism via goal relabeling.

• Implementation & Empirical Evaluation: Implement the complete algorithmic pipeline utilizing Qwen as the base large language model (LLM), execute systematic ablation studies to isolate the contributions of TBRM, RLOO, and hindsight data augmentation, and track training metrics including Pass@k, loss variance, and cumulative reward curves.

• Analysis & Comprehensive Reporting: Profile policy training convergence and late-stage mode collapse dynamics, perform behavioral analyses regarding reasoning step redundancy pruning, evaluate the sensitivity of the temperature hyperparameter β, and compose the final research manuscript.

Changes from Proposal While the core objective of transforming sparse terminal rewards into dense token-level feedback via TBRM remains fully aligned with the original proposal, the specific variance reduction and sample efficiency mechanisms were adapted during empirical iteration.

First, rather than utilizing Group Relative Policy Optimization (GRPO) for trajectory-level advantage estimation as initially planned, we integrated the REINFORCE Leave-One-Out (RLOO) estimator applied directly to the Bellman residuals. This variation provides a more structurally sound and targeted variance reduction mechanism by effectively normalizing baseline discrepancies caused by varying prompt difficulties.

Second, we replaced the proposed prioritized historical trajectory replay mechanism and its associated off-policy importance-weight corrections with hindsight goal relabeling (Hindsight Experience Replay). For the sequential structures inherent to the Countdown arithmetic task, hindsight relabeling proved to be a far more appropriate data augmentation strategy, allowing the framework to directly reinforce failed or suboptimal trajectories by matching them to synthetic goals. These architectural modifications evolved naturally from the iterative nature of the research and development process, ultimately yielding better training stability and faster convergence.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers). 12248–12267.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. 2017. Hindsight experience replay. Advances in neural information processing systems 30 (2017).

Andras Antos, Csaba Szepesvari, and Remi Munos. 2008. Learning near-optimal policies with Bellman-residual minimization based fitted policy iteration. Machine Learning (2008).

Leemon Baird. 1995. Residual Algorithms: Reinforcement Learning with Function Approximation. ICML (1995).

Christoph Dann, Gerhard Neumann, and Jan Peters. 2014. Policy Evaluation with Temporal Differences: A Survey and Comparison. In JMLR.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).

Nathan Herr, Tim Rocktäschel, and Roberta Raileanu. 2025. LLM-first search: Self-guided exploration of the solution space. arXiv preprint arXiv:2506.05213 (2025).

Michael Katz, Harsha Kokel, and Sarath Sreedharan. 2025. Seemingly Simple Planning Problems are Computationally Challenging: The Countdown Game. arXiv preprint arXiv:2508.02900 (2025).

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. 2025. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419 (2025).

Yongsheng Lian. 2025. Reinforcement Learning is all You Need. arXiv preprint arXiv:2503.09512 (2025).

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, Vol. 2024. 39578–39601.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022a. Training language models to follow instructions with human feedback. Advances in neural information processing systems 35 (2022), 27730–27744.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022b. Training language models to follow instructions with human feedback. arXiv:2203.02155 [cs.CL]

Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS (2023).

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).

Richard S. Sutton and Andrew G. Barto. 1998. Reinforcement Learning: An Introduction. MIT Press (1998).

Ronald J Williams. 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning 8, 3 (1992), 229–256.

Yurun Yuan, Fan Chen, Zeyu Jia, Alexander Rakhlin, and Tengyang Xie. 2026. Trajectory bellman residual minimization: A simple value-based method for llm reasoning. Advances in Neural Information Processing Systems 38 (2026), 116994–117032.

Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, et al. 2025. Flowrl: Matching reward distributions for llm reasoning. arXiv preprint arXiv:2509.15207 (2025).