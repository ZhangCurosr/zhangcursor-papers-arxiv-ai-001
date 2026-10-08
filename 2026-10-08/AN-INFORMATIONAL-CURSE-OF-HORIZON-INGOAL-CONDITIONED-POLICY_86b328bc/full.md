# AN INFORMATIONAL CURSE OF HORIZON INGOAL-CONDITIONED POLICY LEARNING

John L. Zhou<sup>∗</sup> Yuxuan Dong<sup>∗</sup> Jonathan C. Kao

University of California, Los Angeles

john.ly.zhou@gmail.com, dyx723@g.ucla.edu

## ABSTRACT

The difficulty of learning goal-reaching policies is often attributed to a “curse of horizon” that manifests as bias accumulation in temporal-difference backups and noisy advantage estimates. In this work, we identify an additional informational curse of horizon in goal-conditioned policy learning, where increasing the goal relabeling horizon can significantly reduce policy generalization and performance. Through a series of controlled experiments with oracle planners, we decouple the goal horizons sampled during training from those that the policy is asked to reach at test time. Even when evaluated only on a sequence of nearby subgoals, goal-conditioned behavioral cloning (BC) policies suffer from severe, training horizon-dependent performance degradation that is mitigated by reinforcement learning (RL) objectives. We explain this phenomenon as a horizon-dependent decrease in the conditional mutual information between actions and hindsightrelabeled goals, and find empirically that both BC and RL policies trained on longer-horizon goals exhibit a shift in sensitivity from goal to state information, as measured by the policy’s input Jacobians. Motivated by this observation, we find that distilling the input Jacobians of short-horizon policies into long-horizon policies yields significant performance gains, especially in combinatorial manipulation tasks. Taken together, our results highlight goal relabeling horizon as an important consideration when learning generalist policies from offline data.

## 1 INTRODUCTION

Learning to reach any state from any other state is an elegantly simple problem that allows an agent to acquire diverse behaviors from data, without supervision (Kaelbling, 1993). A core component of standard goal-conditioned policy learning involves hindsight goal relabeling (Andrychowicz et al., 2017), which allows an agent to learn from arbitrary experiences regardless of the actor’s original intentions. However, simply providing a policy with auxiliary input in the form of goal observations or natural language instructions does not necessarily produce a policy that is capable of following such instructions. A goal-reaching policy must learn to meaningfully utilize information about the desired goal to generate actions, and we argue that this dependence can only emerge when knowing the goal provides meaningful additional information about the policy learning objective.

We formalize this additional goal-related information as the conditional mutual information (MI) I(A; G | S) between the goal G and action A conditioned on the current state S, which we refer to as the action-goal mutual information. Under standard hindsight relabeling, action-goal MI is non-increasing with the relabeling horizon [Theorem 3.2]. This points to a distinct informational curse of horizon in goal-conditioned policy learning: increasing the horizon of relabeled training goals not only introduces well-characterized learning challenges (Park et al., 2025), but may also alter how the policy is learned. Specifically, as distant goals provide progressively less information about which action should be taken, the policy may converge to a solution that is globally less sensitive to goal information—even when conditioned on nearby goals.

However, disentangling these two effects is surprisingly difficult in standard goal-reaching benchmarks (Park et al., 2024). Long-horizon evaluation goals are typically more challenging and require longer sequences of correct actions, while policies must also be trained on distant goals to simply keep them in-distribution at evaluation time. This makes it virtually impossible to distinguish failures due to increased task difficulty from failures due to weaker goal conditioning.

To isolate the effect of training horizon $H _ { \mathrm { t r a i n } }$ on performance, we construct a series of controlled experiments using an oracle hierarchical planner that “prompts” a learned, goal-conditioned policy with optimal, fixed-distance (sub)goals at test time. We find that behavioral cloning (BC) policies degrade severely as $H _ { \mathrm { t r a i n } }$ increases, even when the distance of goals presented at test time $H _ { \mathrm { p l a n } }$ remains fixed. On the other hand, reinforcement learning (RL) policies are substantially more robust to increased training goal horizons, although this robustness is closely tied to the RL objective (weighted BC or reparameterized policy gradient) and policy class (Gaussian or conditional flow matching) [Section 3.1]. These results suggest that poor policy performance is not only due to higher task difficulty, but also a degradation in the learned policy’s ability to achieve simpler goals.

Our theoretical analysis predicts that longer-horizon goals provide less action-relevant information, but how might this be reflected in a policy parameterized by a neural network? In our empirical studies, we use the Jacobian norm of the action with respect to the state or goal as a local measure of policy sensitivity. For certain parameterizations, e.g., fixed-variance Gaussians, the Jacobian norm can be directly related to Fisher information [Box 3.3]. Across BC and RL objectives, as well as Gaussian and flow-matching policy classes, we find that increasing the training goal horizon systematically shifts sensitivity away from goal conditioning towards the current state. Furthermore, explicitly reducing the action-goal MI by training on some proportion of arbitrary, unrelated goa states produces a similar reduction in goal sensitivity.

To determine if there is a causal relationship between goal sensitivity and performance, we then ask if preserving the sensitivity structure of policies trained exclusively on short-horizon goals can improve the performance of policies trained on longer-horizon goals. We propose Sobolev bootstrapping, a simple modification of existing policy bootstrapping methods that additionally distills the input Jacobians of a short-horizon teacher policy into a long-horizon student policy. This simple intervention significantly boosts performance in a suite of 16 diverse locomotion and manipulation environments, with especially large gains in combinatorial manipulation tasks.

## 2 PRELIMINARIES

We consider the problem of offline goal-conditioned RL, described by a Markov decision process (MDP) $\mathcal { M } = ( \bar { { S } } , \mathcal { A } , \mathcal { R } , \mathcal { P } , \mathcal { G } )$ where S is the state space, A the action space, $\mathcal { R } : \mathcal { S } \times \mathcal { S } $ R the goal-conditioned reward function, $\mathcal { P } : \mathcal { S } \times \mathcal { A }  \mathcal { S }$ the transition function, and G the goal space. In this work, we assume the goal space is equivalent to the state space.

In the offline setting, we are given a reward-free dataset of environment interactions D and asked to learn a policy $\pi ( \cdot \mid s , g )$ to reach a specified goal g. At test time, the learned policy is deployed into the environment for a set of evaluation goals drawn from some goal distribution $p ( g )$ . Then, we can write the objective as

$$
\mathcal { I } ( \pi ) = \mathbb { E } _ { g \sim p ( g ) , \tau \sim p ^ { \pi } ( \tau ) } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r \left( s _ { t } , g \right) \right] ,\tag{1}
$$

where $p ^ { \pi } ( \tau )$ is the distribution of trajectories generated by the policy π and the transition function P during (online) evaluation. Importantly, we do not assume access to $p ( g )$ and must learn a general policy that is capable of reaching any goal encountered in the dataset.

## 2.1 INPUT JACOBIANS AND POLICY SENSITIVITY

Our analyses draw inspiration from prior work that studies input gradients, defined simply as the gradient of the model output with respect to its inputs, from the perspective of classifier robustness in computer vision applications. In our work, we extend these ideas to the parameters of a policy action distribution π. For exposition, we first study policies modeled as fixed-variance Gaussian distributions, where $a \sim \mathcal { N } ( \mu _ { \theta } ( s , g ) , \mathbf { I } )$ . Differentiating this output with respect to the state or goal yields the policy’s state and goal Jacobians,

$$
J _ { \pi } ( s ) = { \frac { \partial \mu _ { \theta } ( a \mid s , g ) } { \partial s } } \quad { \mathrm { a n d } } \quad J _ { \pi } ( g ) = { \frac { \partial \mu _ { \theta } ( a \mid s , g ) } { \partial g } } .\tag{2}
$$

![](images/c3d3bd9784b07fe5a9f886c95bf4fa69456be10614aa519b5ff26b6a4e368252.jpg)  
Figure 1: Training on more distant relabeled goals harms nearby goal-reaching capabilities. We record the success rates across policies trained on a geometric distribution of relabeling goal offsets with discount γ, where $\gamma = 1$ corresponds to a uniform distribution over future states. At evaluation, policies are given optimal $H _ { \mathrm { p l a n } } { \mathrm { - s t e p } }$ subgoals from oracle planners. Success rates are averaged over the last three checkpoints, three seeds, and four environments: cube-double, cube-triple, cube-quadruple, and scene.

Prior work has established that properties of input gradients such as the magnitude of the $\ell _ { 1 }$ norm and degree of alignment with human perception strongly correlate with model robustness (Tsipras et al., 2018; Ganz et al., 2023), and either regularizing the input gradient norm (Drucker & LeCun, 1992; Ross & Doshi-Velez, 2018; Rodríguez-Muñoz et al., 2024) or distilling input gradients from a robust teacher through Sobolev training (Czarnecki et al., 2017; Chan et al., 2020) can significantly improve generalization under perturbations.

## 3 AN INFORMATIONAL CURSE OF HORIZON

Horizon has many opportunities to influence the performance of a goal-reaching policy in subtly different ways. Longer evaluation horizons allow compounding error to accumulate over time, reducing the success rate of a policy with a fixed per-step error rate ϵ (Ross & Bagnell, 2010). Furthermore, directly learning to reach distant goals is simply harder in the first place, due to noisy advantage estimates and compounding bias in value learning [Appendix A.2].

We identify a third challenge in which increasing the horizon of relabeled training goals, $H _ { \mathrm { t r a i n } } ,$ can degrade the policy’s ability to reach nearby goals as well. Said differently, the difficulties induced by long-horizon training are not confined to reaching distant goals.

To study this effect, we introduce a third horizon: the planning horizon, $H _ { \mathrm { p l a n } }$ , which separates the total evaluation horizon $H _ { \mathrm { e v a l } }$ from the horizon of the goals presented to the learned policy. Using an oracle planner, we decouple $H _ { \mathrm { p l a n } }$ from $H _ { \mathrm { t r a i n } }$ and $H _ { \mathrm { e v a l } }$ . This reveals severe, training horizondependent degradation for goal-conditioned BC policies and a milder, albeit objective-dependent form of the same phenomenon in RL policies. We provide an information-theoretic explanation for these findings and study how it manifests in policies parameterized by neural networks.

## 3.1 HOW DOES HORIZON AFFECT PERFORMANCE?

To isolate policy quality from task difficulty, we use an oracle planner that presents the policy with optimal subgoals at a fixed $H _ { \mathrm { p l a n } }$ while leaving the underlying evaluation task unchanged. $H _ { \mathrm { p l a n } }$ is similar in spirit to the subgoal horizon used in hierarchical approaches, but here we use it only at evaluation rather than learning a separate high-level policy.

We vary the distribution over training goal horizons in three complementary ways. First, we sample future in-trajectory goals from a geometric distribution with increasing discount factors γ, which assign more probability mass to longer horizons while maintaining the same goal support set [Figures 1 & 6]. Then, we directly control the range of goal distances the actor sees by truncating uniform distributions to different maximum offsets [Figure 5]. Finally, we mix in arbitrary, unrelated dataset states as goals, directly weakening the predictive relationship between actions and goals [Figure 7].

Across both the fixed- and geometric-horizon experiments, we observe a clear degradation in nearby goal-reaching competence as BC policies are trained on increasingly distant relabeled goals. Because the evaluation task and $H _ { \mathrm { p l a n } }$ are fixed, this effect cannot simply be attributed to the policy being asked to reach more distant goals. Instead, it is clear that increasing $H _ { \mathrm { t r a i n } }$ changes the quality of the learned actor itself. RL policies exhibit a milder version of the same phenomenon, where the degree of robustness to horizon depends heavily on the objective.

Our results demonstrate more nuance than a monotonic relationship between training goal horizon and performance. Geometric relabeling can retain strong performance even with exposure to long-horizon goals, and providing unrelated dataset states as goals results in consistent performance decline across all algorithms. Taken together, our experiments suggest that temporal or spatial training goal distance alone cannot fully account for policy degradation. To resolve this discrepancy, we hypothesize that the informational relationship between actions and goals is the critical factor. Next, we formalize this intuition and investigate how training on extended horizons alters the learned policy for nearby goals.

## 3.2 WHY SHOULD TRAINING HORIZON MATTER?

To understand why training on extended horizons can degrade nearby goal-reaching performance [Section 3.1], we theoretically analyze the divergence between a goal-conditioned policy and its unconditional behavioral prior, focusing on how sources of goal-dependent action information change with relabeling horizon. To make the analysis tractable, we focus on the KL-regularized objective corresponding to Peng et al. (2019). We use uppercase letters $\left( \mathrm { e . g . , } S _ { t } \right)$ to denote random variables, and lower-case letters $( \mathrm { e } . \mathrm { g } . , s _ { t } )$ to denote their instantiations. The full proof is provided in Section B.

Theorem 3.1 (Divergence Bound from the Goal-Agnostic Prior). Let $\pi ^ { * } ( a \mid s , g )$ ∝ $p _ { \mathrm { H R } } ( a$ $s , g ) \exp ( Q ( s , a , g ) / \alpha )$ be the target policy induced by a KL-regularized policy update, where α is a temperature parameter and $p _ { \mathrm { H R } }$ is the distribution of tuples $( S , A , \bar { G } ) \stackrel { . } { = } ( \bar { S } _ { t } , A _ { t } , S _ { t + K } )$ induced by the dataset trajectory distribution and a hindsight relabeling rule $K \sim q _ { \mathrm { { H R } } }$ Let $\beta ( a \mid s )$ be the goal-agnostic behavior policy induced by the dataset, and let the critic value span be $\begin{array} { r } { \dot { \Delta } _ { Q } ( s , g ) = \operatorname* { s u p } _ { a } Q ( s , a , g ) - \operatorname* { i n f } _ { a } Q ( s , a , g ) } \end{array}$ over the support of p<sub>HR</sub>. Then, the expected squared total variation distance between the two policies is bounded by:

$$
\mathbb { E } _ { S , G } \left[ D _ { \mathrm { T V } } \left( \pi ^ { * } ( \cdot \mid S , G ) , \beta ( \cdot \mid S ) \right) ^ { 2 } \right] \leq \frac { \mathbb { E } _ { S , G } \left[ \Delta _ { Q } ( S , G ) ^ { 2 } \right] } { 8 \alpha ^ { 2 } } + I _ { \mathrm { H R } } ( A ; G \mid S )
$$

This bound reveals that $\pi ^ { * }$ degenerates toward the goal-agnostic prior when the value function weakly distinguishes among supported actions relative to $\alpha ,$ and the relabeled goal contains little additional information about the action when conditioned on the current state. Critically, both terms decay over the horizon. Notably, the first value-based term applies only to RL, whereas the second is applicable to both BC and RL.

The diminishing action gap. The action gap quantifies the difference in value between the optimal action and sub-optimal actions at a given state (Farahmand, 2011). In goal-conditioned RL with sparse rewards and a discount factor $\gamma < 1$ , the relative advantage of any single primitive action decays exponentially as the goal distance $d ( s , g )$ increases. Consequently, the expected value span $\Delta _ { Q } ( \bar { S } , G )$ shrinks, and the Q-values provide progressively less information about the optimal action distribution. As $d ( s , g ) \to \infty ,$ the first term approaches zero.

Proposition 3.2 (Monotonic Decay of Mutual Information). Let dataset trajectories be generated by a Markovian behavior policy and transition kernel. For any time horizon (relabeling offset) $\dot { k ^ { \prime } } > k > 0 ,$ given the current state $S _ { t } ,$ the sequence $A _ { t } \to S _ { t + k } \to S _ { t + k ^ { \prime } }$ forms a conditional Markov chain. The mutual information between the action and future states monotonically decreases over time:

$$
\begin{array} { r l } & { I \left( A _ { t } ; S _ { t + k } \ | \ S _ { t } \right) - I \left( A _ { t } ; S _ { t + k ^ { \prime } } \ | \ S _ { t } \right) } \\ & { \qquad = \mathbb { E } _ { S _ { t } , S _ { t + k } , S _ { t + k ^ { \prime } } } \left[ D _ { \mathrm { K L } } \left( p ( A _ { t } \ | \ S _ { t } , S _ { t + k } ) \| p ( A _ { t } \ | \ S _ { t } , S _ { t + k ^ { \prime } } ) \right) \right] \ge 0 } \end{array}
$$

This inequality is strict when there exist at least two intermediate states, $s _ { t + k } ^ { ( 1 ) }$ and $s _ { t + k } ^ { ( 2 ) }$ , that share a common future successor state $s _ { t + k ^ { \prime } }$ with non-zero probability, but yield distinct intermediate action posteriors $p ( A _ { t } \mid s _ { t } , s _ { t + k } ^ { ( 1 ) } ) \neq p ( A _ { t } \mid s _ { t } , s _ { t + k } ^ { ( 2 ) } )$

![](images/c8ffdae3c17844b803b22052f2e80b3d78bb3eee724a0cee24d00eabbc9b39b8.jpg)  
Figure 2: Increasing training goal horizon globally decreases policy goal sensitivity. We measure the proportion of the input Jacobian norm corresponding to the goal at fixed state-goal offsets, plotted with log scaling in ${ \mathsf { c u b e - t r i p 1 e } }$ . Results are averaged over the last three training checkpoints, three seeds, and 1024 randomly sampled state-goal pairs per horizon, with additional environments in Figure 8.

Non-increasing mutual information. BC suffers a catastrophic performance degradation when trained on more distant relabeled goals (see GCBC and GCFBC in Figure 1), which is unsurprising as it does not benefit from the additional goal-related information imparted by a goal-conditioned value function. We formalize this as a value-independent reduction in mutual information $I _ { \mathrm { H R } } ( A ; G \mid S )$ as the relabeling horizon grows [Theorem 3.2]. An extended discussion is included in Section B.

This bound only guarantees non-increasing information. Strictly decreasing information occurs in environments and datasets where distinct trajectories eventually merge at a shared future state, which can occur in two common scenarios. First, when learning from a sub-optimal policy, subsequent behavior can “overwrite” the effects of earlier actions. An agent might make a mistake and later correct it, reaching the same future state as a trajectory that took an optimal path. Second, merging occurs when there are multiple valid ways to accomplish the same goal. For example, in a sequential pick-and-place task over three objects, there are up to six valid orderings that all lead to the same final configuration. In both scenarios, longer task and relabeling horizons provide more opportunities for these action-distinguishable trajectories to merge. Thus, temporally distant relabeled goals provide progressively less information about the immediate action.

## 3.3 FROM ACTION-GOAL MUTUAL INFORMATION TO POLICY SENSITIVITY

While our theoretical results characterize goal dependence as a property of the full distribution, it is difficult to measure this quantity in a policy parameterized by a neural network. One local measure of sensitivity is the input gradient (or Jacobian, for vector-valued outputs). As introduced in Section 2.1, we use the Jacobian of the policy action with respect to the state and goal as a first-order measure of policy sensitivity. A large goal Jacobian indicates that small changes in the goal induce large changes in the action, whereas a small goal Jacobian indicates local invariance to the goal.

It is important to acknowledge that this quantity is not an estimator of $I ( A ; G \mid S )$ . Rather, it allows us to locally probe the action-goal dependence whose global behavior is captured by mutual information. For fixed-variance Gaussian policies, the squared Frobenius norm of the input Jacobian corresponds to the trace of the Fisher information matrix (FIM), and directly measures how rapidly the policy distribution changes with respect to changes in goal information [Box 3.3].

To investigate whether the goal Jacobian norm reflects the predicted decrease in action-goal MI, we measure the $\ell _ { 1 }$ goal Jacobian norm proportion for policies trained on the goal distributions described in Section 3.1. The geometric [Figures 2 & 8] and truncated uniform distributions [Figure 9] vary the proportion of temporally distant goals, which we expect to be less informative about the immediate action. The random goal mixture more directly reduces action-goal MI by introducing unrelated target states [Figure 10]. Across BC and RL objectives, as well as Gaussian and flow-matching policy classes, we observe a consistent, global shift in sensitivity away from the goal towards the current state across matched state-goal distances as the training horizon increases. As predicted, we see that goal sensitivity collapses for the BC policies, which do not utilize additional value-related information [Section 3.2].

## Intuition: Fisher Information Regularization

When $\pi$ is a Gaussian distribution with fixed variance $\sigma ^ { 2 }$ , the state Jacobian $J _ { \pi } ( s )$ is the square root of the Fisher information matrix (FIM)

$$
\begin{array} { r } { \mathcal { T } ( s ) = \mathbb { E } _ { \pi _ { \theta } } \left[ \nabla _ { s } \log \pi _ { \theta } ( a \mid s , g ) \cdot \nabla _ { s } \log \pi _ { \theta } ( a \mid s , g ) ^ { T } \right] } \end{array}
$$

with respect to the state. Consequently, the squared Frobenius norm of the state Jacobian, $\| J _ { s } ( s , \acute { g } ) \| _ { F } ^ { 2 }$ , when scaled by $\sigma ^ { 2 }$ , corresponds exactly to the trace of the FIM and serves as a scalar measure of the local information the policy extracts from the state input s.

Viewed through this lens, our Sobolev bootstrapping objective [Equation 3] acts as a dynamic form of Fisher information regularization. By penalizing deviations between the student’s state Jacobian and that of the short-horizon teacher, we enforce the local information geometry of the long-horizon policy to be similar to that of the short-horizon policy.

This also provides a mechanical explanation for the findings in Section 5.3 that regularizing the state Jacobian also increases the norm of the goal Jacobian. By regularizing local state dependence, Sobolev bootstrapping prevents the network from converging to a solution which exploits state-only optimization shortcuts and ignores goal conditioning inputs [Section 3].

## 4 SOBOLEV BOOTSTRAPPING

The above analysis establishes a strong relationship between action-goal information and local policy sensitivity, where policies trained on longer-horizon goals exhibit reduced sensitivity to goal conditioning and a corresponding increase in sensitivity to the current state. This motivates us to investigate whether intervening on the policy input Jacobian can mitigate performance degradation. Direct $\ell _ { 1 }$ and $\ell _ { 2 }$ penalties on the state Jacobian yield modest gains on simpler tasks [Table 2], providing evidence for a causal link between the policy sensitivity and downstream performance. However, these naive penalties fail to scale to more complex, multi-object environments.

We argue that a more principled Jacobian intervention should be context-dependent, and propose Sobolev bootstrapping. Our approach uses a policy trained only on short-horizon goals as a firstorder teacher for the full goal-conditioned “student” policy. Rather than globally suppressing state sensitivity via a uniform norm penalty, we regularize the long-horizon policy’s state Jacobian towards that of the short-horizon policy, explicitly distilling the desirable sensitivity structure learned in a regime of high action-goal information. We additionally note that, for fixed-variance Gaussian policies, this objective can be interpreted as a form of Fisher information regularization [Box 3.3].

## 4.1 USING SHORT-HORIZON POLICIES AS TEACHERS

As the name suggests, Sobolev bootstrapping builds on a class of policy extraction algorithms which perform policy bootstrapping (Chane-Sane et al., 2021; Zhou & Kao, 2025). These algorithms provide a natural foundation for our proposed method, since they already maintain a separate one-step policy trained only on short-horizon goals.

Due to space constraints, we provide only a brief intuition here and defer detailed descriptions of the value learning and policy extraction algorithms to Section D. Policy bootstrapping methods leverage the insight that, if an action a is optimal for reaching a subgoal $s _ { g }$ along the optimal path to goal $^ { g , }$ then a is also optimal for reaching g. Additionally, goal-conditioned policies are more likely to be accurate for short-horizon subgoals $s _ { g } ,$ due to the decreasing “signal-to-noise” ratio of advantage estimates over the horizon (Park et al., 2023). Policy bootstrapping therefore uses $\pi ( \boldsymbol { a } \mid \boldsymbol { s } , \boldsymbol { s } _ { g } )$ as a stronger, short-horizon supervisory signal than pure value-based policy extraction, by minimizing the Kullback-Leibler (KL) divergence between their action distributions

$$
\underset { \theta } { \arg \operatorname* { m i n } } D _ { \mathrm { K L } } ( \pi _ { \theta } ( a  { | \ s , g  }  { \| \pi ^ { \mathrm { s u b } } ( a  { | \ s , { s } _ { g } } )  } .
$$

But how do we find good subgoals for a given goal? Existing approaches such as Reinforcement Learning with Imagined Subgoals (Chane-Sane et al., 2021, RIS) and Subgoal Advantage-Weighted

<table><tr><td>Environment</td><td>Dataset</td><td>GCBC</td><td>GCIVL</td><td>GCIQL</td><td>QRL</td><td>CRL</td><td>HIQL</td><td>SAW</td><td>S2AW</td></tr><tr><td rowspan="3">pointmaze</td><td>pointmaze-medium-navigate-v0</td><td> $9 \pm 6$ </td><td> $6 3 \pm 6$ </td><td> $5 3 \pm 8$ </td><td> $8 2 \pm 5$ </td><td> $2 9 \pm 7$ </td><td>79 ±5</td><td>97 ±2</td><td>96 ±3</td></tr><tr><td>pointmaze-large-navigate-v0</td><td> $2 9 \pm 6$ </td><td>45 ±5</td><td> $3 4 \pm 3$ </td><td> ${ \bf 8 6 } _ { \pm 9 }$ </td><td>39 ±7</td><td>58 ±5</td><td>85 ±10</td><td>83 ±8</td></tr><tr><td>pointmaze-giant-navigate-v0</td><td> $1 \pm 2$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $6 8 \pm 7$ </td><td> $2 7 \pm 1 0$ </td><td>46 ±9</td><td>68 ±8</td><td>73 ±9</td></tr><tr><td rowspan="3">antmaze</td><td> $\mathtt { a n t m a z e - m e d i u m - n a v i g a t e - v 0 }$ </td><td> $2 9 \pm 4$ </td><td> $7 2 \pm 8$ </td><td> $7 1 \pm 4$ </td><td> $8 8 \pm 3$ </td><td> ${ \bf 9 5 \pm 1 }$ </td><td>96 ±1</td><td> $\mathbf { 9 7 \pm 1 }$ </td><td>97 ±1</td></tr><tr><td>antmaze-large-navigate-v0</td><td> $2 4 \pm 2$ </td><td> $1 6 \pm 5$ </td><td> $3 4 \pm 4$ </td><td>75 ±6</td><td> $8 3 \pm 4$ </td><td>91 ±2</td><td>90 ±3</td><td>93 ±2</td></tr><tr><td> $\mathtt { a n t m a z e - g i a n t - n a v i g a t e - v 0 }$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td>14 ±3</td><td> $1 6 \pm 3$ </td><td>65 ±5</td><td>73 ±4</td><td>87 ±3</td></tr><tr><td rowspan="3">humanoidmaze</td><td>humanoidmaze-medium-navigate-v0</td><td>8 ±2</td><td> $2 4 \pm 2$ </td><td> $2 7 \pm 2$ </td><td> $2 1 \pm 8$ </td><td> $6 0 \pm 4$ </td><td>89 ±2</td><td>88 ±3</td><td>90 ±2</td></tr><tr><td>humanoidmaze-large-navigate-v0</td><td>1 ±0</td><td> $2 \pm 1$ </td><td> $2 \pm 1$ </td><td>5 ±1</td><td> $2 4 \pm 4$ </td><td>49 ±4</td><td>46 ±4</td><td> ${ \bf 5 7 \pm 5 }$ </td></tr><tr><td>humanoidmaze-giant-navigate-v0</td><td>0 ±0</td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td>1 ±0</td><td> $3 \pm 2$ </td><td>12 ±4</td><td>35 ±4</td><td> ${ \bf 3 9 \pm 5 }$ </td></tr><tr><td rowspan="4">cube</td><td>cube-single-play-v0</td><td>6 ±2</td><td> $5 3 \pm 4$ </td><td>68 ±6</td><td>5 ±1</td><td> $1 9 \pm 2$ </td><td>44 ±9</td><td> $7 2 \pm 5$ </td><td> ${ \bf 8 6 \pm 5 }$ </td></tr><tr><td> $\mathtt { c u b e - d o u b l e - p l a y - v 0 }$ </td><td>1 ±1</td><td> $3 6 \pm 3$ </td><td> $4 0 \pm 5$ </td><td>1 ±0</td><td> $1 0 \pm 2$ </td><td>6 ±2</td><td> $4 0 \pm 7$ </td><td> ${ \bf 5 7 \pm 6 }$ </td></tr><tr><td> $\mathtt { c u b e - t r i p l e - p l a y - v 0 }$ </td><td>1 ±1</td><td> $1 \pm 0$ </td><td> $3 \pm 1$ </td><td>0 ±0</td><td> $4 \pm 1$ </td><td>3 ±1</td><td>4 ±2</td><td> $\mathbf { 3 0 \pm 5 }$ </td></tr><tr><td>cube-quadruple-play-v0</td><td>0 ±0</td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td>0 ±0</td><td>0 ±0</td><td>0 ±0</td><td>0 ±0</td><td> $\mathbf { 3 \pm 2 }$ </td></tr><tr><td>scene</td><td>scene-play-v0</td><td>5 ±1</td><td> $4 2 \pm 4$ </td><td> $5 1 \pm 4$ </td><td>5 ±1</td><td>19 ±2</td><td>38 ±3</td><td> $6 3 \pm 6$ </td><td> ${ \bf 7 4 } \pm { \bf 6 }$ </td></tr><tr><td rowspan="2">puzzle</td><td> $\mathtt { p u z z 1 e - 3 x 3 - p 1 a y - v 0 }$ </td><td>2 ±0</td><td> $6 \pm 1$ </td><td> ${ \bf 9 5 \pm 1 }$ </td><td>1 ±0</td><td>3 ±1</td><td>12 ±2</td><td> $9 \pm 3$ </td><td> $1 5 \pm 4$ </td></tr><tr><td> $\mathtt { p u z z 1 e - 4 x 4 - p 1 a y - v 0 }$ </td><td>0 ±0</td><td> $1 3 \pm 2$ </td><td> $2 6 \pm 3$ </td><td>0 ±0</td><td>0 ±0</td><td>7 ±2</td><td> $2 6 \pm 3$ </td><td> ${ \bf 4 1 \pm 3 }$ </td></tr></table>

Table 1: Evaluating Sobolev bootstrapping on offline goal-conditioned RL tasks. We compare our method’s average (binary) success rate (%) against the numbers reported in Park et al. (2024) and Zhou & Kao (2025) across the five test-time goals for each environment, averaged over the last three checkpoints and eight seeds ± standard deviations. Numbers within 5% of the best value in the row are in bold.

Policy Bootstrapping (Zhou & Kao, 2025, SAW) differ primarily in how they answer this question. Similar to a bilevel hierarchical policy, RIS learns a generative model $\pi ^ { h } ( s _ { g } \mid s , g )$ to produce “imagined” subgoals. However, explicitly learning a hierarchical policy is known to suffer from compounding error and bottlenecks performance (Ahn et al., 2025). On the other hand, SAW avoids these optimization difficulties by sampling only from in-trajectory waypoints $s _ { g } \sim p ^ { D } ( s _ { t + k } \mid s _ { t } , g ) $ It then re-weights the bootstrapping loss by exp $( A ( s , s _ { g } , g ) )$ , using a subgoal advantage estimate to prioritize high-quality waypoints [Equation 12]. The advantage $A ( s , s _ { g } , g )$ is approximated by the learned value function as $\bar { V } ( s _ { g } , g ) - \mathrm { \bar { \it V } } ( s , g )$ , following Park et al. (2023).

## 4.2 INPUT JACOBIAN DISTILLATION

While standard policy bootstrapping only distills the action distribution of the short-horizon teacher, our experiments in Section 3 suggest that the input Jacobian of a short-horizon teacher may also provide useful supervisory signals. Drawing inspiration from Sobolev training (Czarnecki et al., 2017), we augment the standard zeroth-order distillation with a first-order objective that matches the student’s state Jacobian to that of the teacher policy $\pi ^ { \mathrm { s u b } } ( a \mid s , s _ { g } ) ;$

$$
\mathcal { T } _ { \mathrm { S B } } = \mathbb { E } _ { s , a , s _ { g } , g } \left[ \left. \frac { \partial \pi _ { \theta } ( a \mid s , g ) } { \partial s } - \frac { \partial \pi ^ { \mathrm { s u b } } ( a \mid s , s _ { g } ) } { \partial s } \right. _ { 2 } ^ { 2 } \right] .\tag{3}
$$

Intuitively, the short-horizon teacher provides a prior not only for what the long-horizon policy should $d o ,$ but also what it should pay attention to when deciding what to do. We match only the state Jacobian because the two policies receive different goal inputs: $\pi ^ { \mathrm { s u b } }$ is conditioned on a nearby subgoal $s _ { g }$ and π on the final goal g. Interestingly, we find that regularizing the state Jacobian induces a corresponding change in the goal Jacobian, which we discuss in more detail in Section 5.3.

## 4.3 A PRACTICAL POLICY EXTRACTION ALGORITHM

Sobolev bootstrapping requires access to differentiable subgoal-conditioned “teacher” policies, making existing policy bootstrapping algorithms a natural foundation for our approach. For our primary results in Table 1, we instantiate Sobolev bootstrapping on top of SAW, which we denote as S<sup>2</sup>AW [Algorithm 1], due to its simplicity and strong performance. We modify the objective in Equation 3 to add the same subgoal advantage-weighted coefficient as in SAW [Section D], resulting in the objective

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { S B } } = \mathbb { E } _ { s , a , s _ { g } , g } \left[ e ^ { A ( s , s _ { g } , g ) } \left. J _ { \pi _ { \theta } } ( s ) - J _ { \pi ^ { \mathrm { s u b } } } ( s ) \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{4}
$$

While we focus our main analysis on $\mathrm { S ^ { 2 } A W , }$ we also demonstrate the effectiveness of Sobolev bootstrapping on RIS, achieving substantial gains over the baseline algorithm [Section H.2].

![](images/744257639a99847dbf504b19c772dd499ba97dfa64aee0fe81944346ca7d5404.jpg)  
Figure 3: Sobolev bootstrapping exhibits more interpretable sensitivity structure. We plot the proportion of the total input Jacobian norm corresponding to each cube during a triple pick-and-place on-policy rollout for SAW without (top) and with (bottom) Sobolev bootstrapping. For fairness, we also analyze a scripted optimal trajectory in Figure 11.

## 5 EXPERIMENTS

We evaluate Sobolev bootstrapping on the OGBench suite (Park et al., 2024), which spans longhorizon locomotion and combinatorial manipulation tasks. Our experiments address two central questions: (1) does Sobolev bootstrapping improve goal-reaching performance, and (2) does it reverse the horizon-induced shift in policy sensitivity identified in our previous experiments? Interestingly, although Sobolev bootstrapping only distills the state Jacobian, we find that it also restores goalconditioning sensitivity [Section 5.3].

## 5.1 SOBOLEV BOOTSTRAPPING BOOSTS LONG-HORIZON PERFORMANCE

Adding Sobolev bootstrapping to SAW $( \mathsf { S } ^ { 2 } \mathsf { A W } )$ improves performance across most environments [Table 1], with the largest gains in combinatorial manipulation tasks such as the cube and puzzle environments. In particular, S<sup>2</sup>AW improves from 40% to 57% on cube-double, 4% to 30% on cube-triple, and 26% to 41% on puzzle-4x4. We also observe substantial improvements in long-horizon locomotion, although these results may be partially attributed to architectural differences [Appendix H.1]. Additional horizon reduction baselines are provided in Section H.3.

The improvements in the discrete puzzle environments demonstrate that first-order distillation can be practically useful even when the true decision boundary is non-smooth, since the short-horizon teacher policy provides a (piecewise) smooth, differentiable approximation of these hard boundaries.

## 5.2 EFFECTS ON STATE SENSITIVITY

To examine how Sobolev bootstrapping qualitatively changes policy sensitivity to state information, we decompose the state Jacobian norm by object over the course of a cube-triple rollout [Figure 3], allowing us to visualize where the policy’s “attention” is allocated. Vanilla SAW exhibits diffuse sensitivity across all cubes and immediately fails at the first task stage, whereas S<sup>2</sup>AW dynamically allocates sensitivity to the focal cube at each stage and successfully completes the full triple pick-and place task. To control for differences in policy competence, we repeat the analysis on an independent, scripted optimal trajectory in Figure 11 and observe the same pattern.

We hypothesize that the observed shift in sensitivity away from goal information and towards state inputs may also result in policies that overfit to task-irrelevant “distractors.” To test this, we evaluate policies trained in multi-cube environments on single pick-and-place tasks. The first evaluation task in cube-double and triple already tests this capability, and we add a similar task to cube-quadruple, which we refer to as Task 6. While naive $\ell _ { 1 }$ and $\ell _ { 2 }$ Jacobian norm penalties produce small improvements, context-dependent Sobolev bootstrapping significantly improves performance even on cube-triple and quadruple [Table 2].

![](images/a6e427eb74ca91475e5f03ceae65c4dfae049ae9b813a2ca3f2d85352e3a8018.jpg)

Figure 4: Sobolev bootstrapping recovers policy goal sensitivity. We compare the absolute goal Jacobian norm between $\mathrm { S } ^ { 2 } \mathrm { A W }$ and SAW, averaging over batches of goals sampled at fixed intervals from the current state up to the maximum horizon. Results are averaged across four seeds and 1024 state-goal pairs for each horizon, with shaded standard deviations
<table><tr><td>Environment</td><td>GCIQL</td><td> $\mathbf { G C I O L } _ { \ell _ { 1 } }$ </td><td>SAW</td><td> $\mathbf { S A W } _ { \ell _ { 1 } }$ </td><td> $\mathbf { S A W } _ { \ell _ { 2 } }$ </td><td>S2AW</td></tr><tr><td>cube-  $\mathtt { - s i n g l e }$ </td><td> $6 8 \pm 6$ </td><td> $7 4 \pm 4$ </td><td> $7 6 \pm 8$ </td><td> ${ \bf 8 5 \pm 6 }$ </td><td> ${ \bf 8 5 \pm 4 }$ </td><td> ${ \bf 8 6 \pm 5 }$ </td></tr><tr><td> $\mathtt { c u b e - d o u b l e - t a s k 1 }$ </td><td> $7 4 \pm 8$ </td><td> $^ { 7 6 } \pm 8$ </td><td> $6 4 \pm 9$ </td><td> ${ \bf 8 1 \pm 9 }$ </td><td> $6 5 \pm 1 4$ </td><td> ${ \bf 7 8 \pm 1 2 }$ </td></tr><tr><td> $\mathtt { c u b e - t r i p l e - t a s k 1 }$ </td><td> $1 3 \pm 3$ </td><td> $1 6 \pm 1 0$ </td><td> $1 8 \pm 1 1$ </td><td> $2 4 \pm 9$ </td><td> $1 7 \pm 1 2$ </td><td> ${ \bf 7 4 \pm 9 }$ </td></tr><tr><td> $\mathtt { c u b e - q u a d r u p l e - t a s k 6 }$ </td><td> $0 \pm 1$ </td><td> $0 \pm 1$ </td><td> $2 \pm 2$ </td><td> $2 \pm 2$ </td><td> $1 \pm 1$ </td><td> $\mathbf { 3 5 \pm 1 4 }$ </td></tr></table>

Table 2: Sobolev bootstrapping improves single pick-and-place performance in the presence of distractor objects. We evaluate policies trained in multi-cube environments on single pick-and-place tasks. While naive $\ell _ { 1 }$ or $\ell _ { 2 }$ Jacobian regularization [Appendix H.4] yields modest improvements on simpler tasks, Sobolev bootstrapping matches or outperforms all baselines. Other settings are identical to Table 1.

## 5.3 EFFECTS ON GOAL CONDITIONING

Although Sobolev bootstrapping explicitly matches only the state Jacobian, we find that it induces an increase in the policy’s absolute goal Jacobian norm [Figure 4]. This suggests that the policy has converged to a different, more goal-sensitive solution instead of simply smoothing out state sensitivity. We note that a decrease in sensitivity to distant goals is unsurprising—for instance, perturbing the target placement in a pick-and-place task should not affect the policy before grasping the cube. However, standard long-horizon policies exhibit global insensitivity across all goal distances, including nearby goals [Figure 4], and it is this pathology that Sobolev bootstrapping addresses.

## 6 DISCUSSION

In this work, we identify an informational curse of horizon in goal-conditioned policy learning. By decoupling the training goal horizon from the subgoal distance presented at test time, we show that policies trained on distant relabeled goals can become worse at reaching nearby goals. We explain this effect through the decay of action-goal mutual information under hindsight relabeling: as training goals become more distant, they provide progressively less information about the immediate action. This is accompanied by a global shift in sensitivity away from the goal and toward the current state in neural network policies, across all goal distances. Although input Jacobians are only a local probe of this dependence, explicitly weakening action-goal information produces a corresponding collapse in goal sensitivity. We then propose Sobolev bootstrapping, which exploits this observation by distilling the sensitivity structure of short-horizon policies into long-horizon policies and achieves large gains in combinatorial manipulation tasks. Overall, our results suggest that the goal-relabeling horizon should be treated as an important design choice, rather than a mere consequence of task horizon.

Our work does have several limitations. The optimal goal-horizon distribution depends strongly on the policy objective, prompting further study on how different policy extraction methods utilize goal information. Additionally, input Jacobians capture only local sensitivity and are not direct estimators of mutual information, and the Sobolev bootstrapping algorithm derived from those insights introduces significant additional computation. These limitations motivate future work on information-aware relabeling and policy-learning objectives that preserve goal dependence without explicit Jacobian distillation.

## ACKNOWLEDGMENTS

This work was supported by NIH R01NS121097 awarded to JCK.

## REFERENCES

Ahn, H., Choi, H., Han, J., and Moon, T. Option-aware temporally abstracted value for offline goal-conditioned reinforcement learning. In Neural Information Processing Systems, 2025.

Amit, R., Meir, R., and Ciosek, K. Discount factor as a regularizer in reinforcement learning. In International Conference on Machine Learning, 2020.

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., and Zaremba, W. Hindsight experience replay. In Neural Information Processing Systems, 2017.

Boucheron, S., Lugosi, G., and Massart, P. Concentration inequalities : a non asymptotic theory of independence. Oxford University Press, 2013.

Chan, A., Tay, Y., and Ong, Y.-S. What it thinks is important is important: Robustness transfers through input gradients. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Chane-Sane, E., Schmid, C., and Laptev, I. Goal-conditioned reinforcement learning with imagined subgoals. In International Conference on Machine Learning, 2021.

Czarnecki, W. M., Osindero, S., Jaderberg, M., Swirszcz, G., and Pascanu, R. Sobolev training for<sup>´</sup> neural networks. In Neural Information Processing Systems (NeurIPS), 2017.

Drucker, H. and LeCun, Y. Improving generalization performance using double backpropagation. IEEE Transactions on Neural Networks, 3(6):991–997, 1992.

Du, Y., Yang, M., Dai, B., Dai, H., Nachum, O., Tenenbaum, J. B., Schuurmans, D., and Abbeel, P. Learning universal policies via text-guided video generation. arXiv preprint arXiv:2302.00111, 2023.

Eysenbach, B., Zhang, T., Salakhutdinov, R., and Levine, S. Contrastive learning as goal-conditioned reinforcement learning. In Neural Information Processing Systems, 2022.

Farahmand, A.-m. Action-gap phenomenon in reinforcement learning. In Neural Information Processing Systems, 2011.

Fujimoto, S. and Gu, S. A minimalist approach to offline reinforcement learning. In Neural Information Processing Systems, 2021.

Ganz, R., Kawar, B., and Elad, M. Do perceptually aligned gradients imply robustness? In International Conference on Machine Learning, 2023.

Hu, H., Yang, Y., Zhao, Q., and Zhang, C. On the role of discount factor in offline reinforcement learning. In International Conference on Machine Learning, 2022.

Jiang, N., Kulesza, A., Singh, S., and Lewis, R. The dependence of effective planning horizon on model accuracy. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2015.

Jiang, N., Singh, S., and Tewari, A. On structural properties of MDPs that bound loss due to shallow planning. In International Joint Conferences on Artificial Intelligence, 2016.

Kaelbling, L. P. Learning to achieve goals. In IJCAI, 1993.

Korniak, M., Dybek, K., Eysenbach, B., Bagatella, M., and Bortkiewicz, M. Three steps at a time: Learning representations from action sequences in contrastive rl. arXiv preprint arXiv:2608.30640, 2026.

Kostrikov, I., Nair, A., and Levine, S. Offline reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2021.

Levine, S. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018.

Li, L., Zhang, Q., Luo, Y., Yang, S., Wang, R., Han, F., Yu, M., Gao, Z., Xue, N., Zhu, X., Shen, Y., and Xu, Y. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026a.

Li, Q., Zhou, Z., and Levine, S. Reinforcement learning with action chunking. In Neural Information Processing Systems, 2025.

Li, Q., Park, S., and Levine, S. Decoupled q-chunking. In International Conference on Learning Representations, 2026b.

Mishra, U. A., Chen, Y., Xu, D., Liu, Y., Chen, X., and Mao, J. Understanding and mitigating the video-action generalization gap via temporal ratio. arXiv preprint arXiv:2607.08127, 2026.

Park, S., Ghosh, D., Eysenbach, B., and Levine, S. HIQL: Offline goal-conditioned RL with latent states as actions. In Neural Information Processing Systems, 2023.

Park, S., Frans, K., Eysenbach, B., and Levine, S. OGBench: Benchmarking offline goal-conditioned RL. In International Conference on Learning Representations (ICLR), 2024.

Park, S., Frans, K., Mann, D., Eysenbach, B., Kumar, A., and Levine, S. Horizon reduction makes RL scalable. In Neural Information Processing Systems, 2025.

Park, S., Oberai, A., Atreya, P., and Levine, S. Transitive RL: Value learning via divide and conquer. In International Conference on Learning Representations, 2026.

Peng, X. B., Kumar, A., Zhang, G., and Levine, S. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

Physical Intelligence, Ai, B., Amin, A., Aniceto, R., Balakrishna, A., Balke, G., Black, K., Bokinsky, G., Cao, S., Charbonnier, T., Choudhary, V., Collins, F., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., Dhaka, M., DiCarlo, J., Driess, D., Equi, M., Esmail, A., Fang, Y., Finn, C., Glossop, C., Godden, T., Goryachev, I., Groom, L., Habeeb, H., Hancock, H., Hausman, K., Hussein, G., Hwang, V., Ichter, B., Jacobsen, C., Jakubczak, S., Jen, R., Jones, T., Kammerer, G., Katz, B., Ke, L., Khadikov, M., Kuchi, C., Lamb, M., LeBlanc, D., LeCount, B., Levine, S., Li, X., Li-Bell, A., Lialin, V., Liang, Z., Lim, W., Lu, Y., Luo, E., Mano, V., Marwaha, N., Mongush, A., Murphy, L., Nair, S., Patterson, T., Pertsch, K., Ren, A. Z., Schelske, G., Sharma, C., Shi, B., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Stoeckle, W., Tang, J., Tanner, J., Tekeste, S., Torne, M., Vedder, K., Vuong, Q., Walling, A., Wang, H., Wang, J., Wang, X., Whalen, C., Whitmore, S., Williams, B., Xu, C., Yoo, S., Yu, L., Zhang, W., Zhang, Z., and Zhilinsky, U. π : a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

Rodríguez-Muñoz, A., Wang, T., and Torralba, A. Characterizing model robustness via natural input gradients. In European Conference on Computer Vision, 2024.

Ross, A. and Doshi-Velez, F. Improving the adversarial robustness and interpretability of deep neural networks by regularizing their input gradients. In AAAI Conference on Artificial Intelligence, 2018.

Ross, S. and Bagnell, D. Efficient reductions for imitation learning. In International Conference on Artificial Intelligence and Statistics, 2010.

Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., and Madry, A. Robustness may be at odds with accuracy. In International Conference on Learning Representations, 2018.

Wang, T., Torralba, A., Isola, P., and Zhang, A. Optimal goal-reaching reinforcement learning via quasimetric learning. In International Conference on Machine Learning, 2023.

Zheng, B., Myers, V., Eysenbach, B., and Levine, S. Scaling goal-conditioned reinforcement learning with multistep quasimetric distances. In International Conference on Learning Representations, 2026.

Zhou, J. L. and Kao, J. Flattening hierarchies with policy bootstrapping. In Neural Information Processing Systems, 2025.

Zhou, P., Chen, L., Chen, S., Chen, D., Zhao, W., Jin, R., Ren, G., and Luo, J. Act2goal: From world model to general goal-conditioned policy. arXiv preprint arXiv:2512.23541, 2025.

## A RELATED WORK

## A.1 OFFLINE GOAL-CONDITIONED POLICY LEARNING

The objective of offline goal-conditioned policy learning is to learn a policy which can reach any state from any other state from a fixed dataset of offline trajectories. This can be accomplished with goal-conditioned behavioral cloning (GCBC) that relabels state-action pairs with future trajectory states as goals; or goal-conditioned reinforcement learning (GCRL), which makes use of learned goal-conditioned value functions. It combines the challenges of conservative offline RL with the sparse rewards and recursive structure of goal-reaching tasks (Kaelbling, 1993). Standard algorithms typically use behavior-regularized actor-critic methods or learn an implicit value function with insample value iteration (Kostrikov et al., 2021), then extract a policy using weighted behavioral cloning (Peng et al., 2019) or reparameterized policy gradient with a behavioral cloning penalty (Fujimoto & Gu, 2021). The recursive structure of goal-conditioned RL also admits a variety of non-traditional value learning algorithms based on successor measures (Eysenbach et al., 2022) and quasimetric structure (Wang et al., 2023; Park et al., 2026).

## A.2 THE CURSE OF HORIZON

The challenges introduced by horizon in sequential decision-making tasks are well known. In addition to the error bounds of Ross & Bagnell (2010) in imitation learning, reinforcement learning introduces a whole host of additional horizon-related difficulties. The effective task horizon, set by the discount factor γ, is known to control the complexity of the policy class (Jiang et al., 2015; 2016), and lower discount factors can provide pessimistic (Hu et al., 2022) and regularizing effects (Amit et al., 2020). However, artificially decreasing the discount factor is not feasible when the true task horizon is long and reward signals are sparse, as in goal-reaching tasks.

Increasing the discount factor in off-policy RL leads to a host of additional problems, typically attributed to bias accumulation in value learning (Park et al., 2025) and difficulty in advantage estimation for primitive actions in policy extraction. To mitigate these difficulties, recent approaches have employed various techniques to reduce the effective horizon relative to decision-making frequency, both for value learning (Ahn et al., 2025; Zheng et al., 2026) and policy extraction (Li et al., 2025; 2026b). Our work specifically focuses on the effects of horizon on goal-conditioned policy learning, where we identify a particular informational curse of horizon: as the horizon increases, the informative signal between the policy and the goal diminishes, causing the policy to overfit to spurious state features at the expense of goal conditioning.

## A.3 STEERABLE ROBOT POLICIES

Effective goal conditioning is increasingly critical for modern robotic systems, including subgoalconditioned policies (Zhou et al., 2025; Physical Intelligence et al., 2026) and inverse-kinematic-style World Action Models (Du et al., 2023; Li et al., 2026a). Recent works demonstrate the significance of effectively utilizing goal information: Mishra et al. (2026) links video-action model (VAM) performance to a model’s reliance on the generated future frames, as measured through crossattention ratios. Korniak et al. (2026) shows that action chunking increases the mutual information between the state-action pair and the goal, which facilitates critic learning in goal-conditioned RL.

## B EXTENDED PROOFS

Below, we use uppercase letters $\left( \mathrm { e . g . , } S _ { t } \right)$ to denote random variables, and lower-case letters $( \mathrm { e } . \mathrm { g } . , s _ { t } )$ to denote their instantiations.

Lemma B.1. Let P be a base probability distribution and let $f ( x )$ be a bounded scalarfunction such that $f ( x ) \in [ a , b ]$ for all x in the support of P. Define the exponentially reweighted distribution $P _ { f }$ as $\begin{array} { r } { P _ { f } ( x ) = \frac { P ( x ) e ^ { f ( x ) } } { Z } } \end{array}$ , where $Z = \mathbb { E } _ { X \sim P } [ e ^ { f ( X ) } ]$ is the partitionfunction. Then, the Kullback-Leibler divergence is bounded by:

$$
D _ { \mathrm { K L } } ( P _ { f } \| P ) \leq \frac { ( b - a ) ^ { 2 } } { 8 }
$$

Proof. The KL divergence between a base distribution $P$ and its exponentially reweighted version $P _ { f } \propto P e ^ { f }$ can be expressed in terms of the log-partition function $\psi ( \bar { \lambda } ) = \mathrm { l o g } \bar { \mathbb { E } } _ { X \sim P } \big [ e ^ { \bar { \lambda } f ( X ) } \big ]$ , where $\lambda \doteq [ 0 , 1 ]$ is a scalar parameter that defines the intermediate distributions between the base distribution and the exponentially reweighted target. By taking a second-order Taylor expansion of $\psi ( \lambda )$ , we can rewrite this KL divergence as the remainder $\scriptstyle { \frac { 1 } { 2 } } \psi ^ { \prime \prime } ( c )$ for some $c \in ( 0 , 1 )$ ). Because $f ( X )$ is bounded in $[ a , b ]$ , we can apply Hoeffding’s Lemma (see Lemma 2.2 in Boucheron et al. (2013)) to bound this second derivative, yielding a maximum KL divergence of $\frac { ( b - a ) ^ { 2 } } { 8 }$ □

Theorem B.2 (Divergence Bound from the Goal-Agnostic Prior). Let $\pi ^ { * } ( a \mid s , g ) \propto p _ { \mathrm { H R } } ( a$ $s , g ) e ^ { Q ( s , a , g ) / \alpha }$ be the target distribution induced by the KL-regularized policy update, where α is a temperature parameter and $p _ { \mathrm { H R } }$ is the distribution of tuples $( S , A , \bar { G } ) \stackrel { . } { = } ( \bar { S } _ { t } , A _ { t } , S _ { t + K } )$ induced by the dataset trajectory distribution and a hindsight relabeling rule K ∼ q<sub>HR</sub>. Let $\beta ( a \mid s )$ be the goal-agnostic behavior policy induced by the dataset, and let the critic value span be $\begin{array} { r } { \dot { \Delta } _ { Q } ( s , g ) = \operatorname* { s u p } _ { a } Q ( s , a , g ) - \operatorname* { i n f } _ { a } Q ( s , a , g ) } \end{array}$ over the support ofp . Then, the expected squared total variation distance between the two policies is bounded by:

$$
\mathbb { E } _ { S , G } \left[ D _ { \mathrm { T V } } \left( \pi ^ { * } ( \cdot \mid S , G ) , \beta ( \cdot \mid S ) \right) ^ { 2 } \right] \leq \frac { \mathbb { E } _ { S , G } \left[ \Delta _ { Q } ( S , G ) ^ { 2 } \right] } { 8 \alpha ^ { 2 } } + I _ { \mathrm { H R } } ( A ; G \mid S )
$$

Proof. We bound the total variation distance (TVD) between the target policy $\pi ^ { * } ( \cdot \mid s , g )$ and the behavioral policy $\beta ( \cdot \mid s )$ by introducing the hindsight-relabeled distribution $p _ { \mathrm { H R } } ( \cdot \mid s , g )$ as an intermediate distribution. We denote $\pi ^ { * } ( \bar { \cdot } \mid s , g )$ as $\pi ^ { * }$ below for brevity. The same notation applies to $\beta ( \cdot \mid s )$ and $p _ { \mathrm { H R } } ( \cdot \mid s , g )$

Because TVD is a metric, it satisfies the triangle inequality. For any given state s and goal $g \colon$

$$
D _ { \mathrm { T V } } ( \pi ^ { * } , \beta ) \leq D _ { \mathrm { T V } } ( \pi ^ { * } , p _ { \mathrm { H R } } ) + D _ { \mathrm { T V } } ( p _ { \mathrm { H R } } , \beta ) .
$$

After squaring both sides and applying the inequality $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ , we have

$$
D _ { \mathrm { T V } } ( \pi ^ { * } , \beta ) ^ { 2 } \leq 2 D _ { \mathrm { T V } } ( \pi ^ { * } , p _ { \mathrm { H R } } ) ^ { 2 } + 2 D _ { \mathrm { T V } } ( p _ { \mathrm { H R } } , \beta ) ^ { 2 } .
$$

Pinsker’s inequality, $\begin{array} { r } { D _ { \mathrm { T V } } ( P , Q ) ^ { 2 } \leq \frac { 1 } { 2 } D _ { \mathrm { K L } } ( P \| Q ) } \end{array}$ , applied to $2 D _ { \mathrm { T V } } ( \pi ^ { * } , p _ { \mathrm { H R } } ) ^ { 2 }$ , gives us

$$
2 D _ { \mathrm { T V } } ( \pi ^ { * } , p _ { \mathrm { H R } } ) ^ { 2 } \leq D _ { \mathrm { K L } } ( \pi ^ { * } \Vert p _ { \mathrm { H R } } ) .
$$

Because $\pi ^ { * }$ is an exponentially reweighted version of p<sub>HR</sub>, we can apply Lemma 1 to $D _ { \mathrm { K L } } ( \pi ^ { * } \| p _ { \mathrm { H R } } )$ with $f ( a ) = Q ( s , \bar { a , } g ) / \alpha$ , yielding

$$
D _ { \mathrm { K L } } ( \pi ^ { * } \| p _ { \mathrm { H R } } ) \leq \frac { \Delta _ { Q } ( s , g ) ^ { 2 } } { 8 \alpha ^ { 2 } } .
$$

Now we have shown

$$
D _ { \mathrm { T V } } ( \pi ^ { * } , \beta ) ^ { 2 } \leq \frac { \Delta _ { Q } ( s , g ) ^ { 2 } } { 8 \alpha ^ { 2 } } + 2 D _ { \mathrm { T V } } ( p _ { \mathrm { H R } } , \beta ) ^ { 2 } .\tag{5}
$$

Next, we apply Pinsker’s inequality again to $2 D _ { \mathrm { T V } } ( p _ { \mathrm { H R } } , \beta ) ^ { 2 }$ , leading to

$$
2 D _ { \mathrm { T V } } ( p _ { \mathrm { H R } } , \beta ) ^ { 2 } \leq D _ { \mathrm { K L } } ( p _ { \mathrm { H R } } ( \cdot \mid s , g ) \| \beta ( \cdot \mid s ) ) .
$$

Note that hindsight relabeling does not change the marginal state distribution, implying $\beta ( a \mid s ) =$ $p _ { \mathrm { H R } } ( a \mid s )$ . Then taking the expectation of this KL divergence over S and G yields the conditional mutual information between actions and goals given the state:

$$
\mathbb { E } _ { S , G } \left[ D _ { \mathrm { K L } } ( p _ { \mathrm { H R } } ( \cdot \mid S , G ) \| p _ { \mathrm { H R } } ( \cdot \mid S ) ) \right] = I _ { \mathrm { H R } } ( A ; G \mid S )\tag{6}
$$

Taking the expectation of equation 5 over S and G and combining it with equation 6 concludes the proof. □

Proposition B.3 (Monotonic Decay of Mutual Information). Let dataset trajectories be generated by a Markovian behavior policy and transition kernel. For any time horizon (relabeling offset) $\dot { k ^ { \prime } } > k > 0 ,$ given the current state $S _ { t } ,$ the sequence $A _ { t } \to S _ { t + k } \to S _ { t + k ^ { \prime } }$ forms a conditional Markov chain. The mutual information between the action and future states monotonically decreases over time:

$$
\begin{array} { r l } & { I \left( A _ { t } ; S _ { t + k } \ | \ S _ { t } \right) - I \left( A _ { t } ; S _ { t + k ^ { \prime } } \ | \ S _ { t } \right) } \\ & { \qquad = \mathbb { E } _ { S _ { t } , S _ { t + k } , S _ { t + k ^ { \prime } } } \left[ D _ { \mathrm { K L } } \left( p ( A _ { t } \ | \ S _ { t } , S _ { t + k } ) \| p ( A _ { t } \ | \ S _ { t } , S _ { t + k ^ { \prime } } ) \right) \right] \ge 0 } \end{array}
$$

This inequality is strict when there exist at least two intermediate states, $s _ { t + k } ^ { ( 1 ) }$ and $s _ { t + k } ^ { ( 2 ) }$ , that share a common future successor state $s _ { t + k ^ { \prime } }$ with non-zero probability, but yield distinct intermediate action posteriors $p ( A _ { t } \mid s _ { t } , s _ { t + k } ^ { ( 1 ) } ) \neq p ( A _ { t } \mid s _ { t } , s _ { t + k } ^ { ( 2 ) } )$

Proof. Expand $I ( A _ { t } ; S _ { t + k } , S _ { t + k ^ { \prime } } \mid S _ { t } )$ with mutual information chain rule in both ways and leverage $I \left( A _ { t } ; S _ { t + k ^ { \prime } } \mid S _ { t } , S _ { t + k } \right) = 0$ implied by the Markov chain, we have

$$
I \left( A _ { t } ; S _ { t + k } ~ \vert ~ S _ { t } \right) - I \left( A _ { t } ; S _ { t + k ^ { \prime } } ~ \vert ~ S _ { t } \right) = I \left( A _ { t } ; S _ { t + k } ~ \vert ~ S _ { t } , S _ { t + k ^ { \prime } } \right) \geq 0 .
$$

Next, we rewrite $I \left( A _ { t } ; S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } \right)$ as an expected KL divergence

$$
\begin{array} { r l } & { I ( A _ { t } ; S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) } \\ & { \quad = \mathbb { E } _ { S _ { t } , S _ { t + k ^ { \prime } } } \left[ D _ { \mathrm { K L } } \left( p ( A _ { t } , S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) \Vert p ( A _ { t } \mid S _ { t } , S _ { t + k ^ { \prime } } ) p ( S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) \right) \right] . } \end{array}
$$

Since $p ( A _ { t } , S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) = p ( S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) p ( A _ { t } \mid S _ { t } , S _ { t + k } )$ , we can simplify the KL term

$$
\begin{array} { r l } & { I ( A _ { t } ; S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) } \\ & { \quad = \mathbb { E } _ { S _ { t } , S _ { t + k ^ { \prime } } , ( S _ { t + k } \mid S _ { t } , S _ { t + k ^ { \prime } } ) } \left[ D _ { \mathrm { K L } } \left( p \left( A _ { t } \mid S _ { t } , S _ { t + k } \right) \parallel p \left( A _ { t } \mid S _ { t } , S _ { t + k ^ { \prime } } \right) \right) \right] . } \end{array}
$$

To see when this inequality is strict $( \mathrm { i . e . }$ , when mutual information strictly decreases with an increase in relabeling horizon), we compare the intermediate action posterior $p ( \boldsymbol { \dot { A } } _ { t } \mid S _ { t } , S _ { t + k } )$ and the future action posterior p( $\boldsymbol { A } _ { t } \mid S _ { t } , S _ { t + k ^ { \prime } } )$ within the KL term. We instantiate some start state $s _ { t }$ and future state $s _ { t + k ^ { \prime } }$ to focus on the KL term inside the expectation. By the law of total probability, the future posterior is the marginalization of the intermediate posteriors over all possible intermediate states:

$$
p ( A _ { t } \mid s _ { t } , s _ { t + k ^ { \prime } } ) = \mathbb { E } _ { S _ { t + k } \mid s _ { t } , s _ { t + k ^ { \prime } } } [ p ( A _ { t } \mid s _ { t } , S _ { t + k } ) ] .
$$

Now, to make the KL term non-zero, we need $\mathbb { E } _ { S _ { t + k } \mid s _ { t } , s _ { t + k ^ { \prime } } } [ p ( A _ { t } \mid s _ { t } , S _ { t + k } ) ] \neq p ( A _ { t } \mid s _ { t } , s _ { t + k } )$ for some instantiation $s _ { t + k }$ . In other words, the intermediate posterior $p ( A _ { t } \mid s _ { t } , S _ { t + k } )$ cannot be almost-surely constant with respect to $S _ { t + k }$

For the intermediate posterior $p ( A _ { t } \mid s _ { t } , S _ { t + k } )$ to vary depending on $S _ { t + k }$ , we need at least two instantiations, $s _ { t + k } ^ { ( 1 ) }$ and $s _ { t + k } ^ { ( 2 ) }$ , that simultaneously satisfy the two conditions below:

Merging paths: both intermediate states must be viable nodes on the path from the start state $s _ { t }$ to the future state $s _ { t + k ^ { \prime } }$ . In other words, they must both occur with non-zero probability given the endpoints:

$$
p ( s _ { t + k } ^ { ( 1 ) } \mid s _ { t } , s _ { t + k ^ { \prime } } ) > 0 \quad \mathrm { a n d } \quad p ( s _ { t + k } ^ { ( 2 ) } \mid s _ { t } , s _ { t + k ^ { \prime } } ) > 0
$$

Different actions: knowing which of these intermediate states the agent passed through must change our belief about the action taken at time t:

$$
p ( A _ { t } \mid s _ { t } , s _ { t + k } ^ { ( 1 ) } ) \neq p ( A _ { t } \mid s _ { t } , s _ { t + k } ^ { ( 2 ) } )
$$

The two conditions are satisfied and strict loss of information occurs in environments and datasets where distinct trajectories eventually “merge” at a shared future state. Intuitively, this can occur in two common scenarios. First, when learning from a sub-optimal policy, subsequent behavior can overwrite the effects of earlier actions. An agent might make a mistake and later correct it, reaching the same future state as a trajectory that took an optimal path. Second, merging occurs when there are multiple valid ways to accomplish the same goal. For example, in a task requiring an agent to pick and place three cubes, there are six valid orderings that all lead to the same final state. In both scenarios, longer task and relabeling horizons provide more opportunities for these action-distinguishable trajectories to merge. Thus, temporally distant relabeled goals become progressively less informative about the original action.

Our analysis extends to distributions over relabeling offsets when $K \sim q _ { \mathrm { H R } }$ is sampled independently of the dataset trajectory. The chain rule for mutual information gives $I _ { \mathrm { H R } } ( { \bar { A } } ; G \mid S ) =$ $\begin{array} { r } { \mathbb { E } _ { K \sim q _ { \mathrm { H R } } } [ I ( A _ { t } ; S _ { t + K } \mid \bar { S _ { t } } ) ] - \dot { I } _ { \mathrm { H R } } ( A ; K \mid S , G ) \le \mathbb { E } _ { K \sim q _ { \mathrm { H R } } } [ I ( A _ { t } ; S _ { t + K } \mid \bar { S _ { t } } ) ] } \end{array}$ ]. Therefore, by Theorem $\mathrm { B } . 3 ,$ , shifting q<sub>HR</sub> towards longer offsets decreases this upper bound on the action-goal mutual information.

## C ALGORITHM PSEUDOCODE

Algorithm 1 Sobolev Bootstrapping   
1: Input: offline dataset ${ \mathcal { D } } ,$ goal distribution $p ( g )$   
2: Initialize value function $V _ { \beta } { \mathrm { , } }$ , target subpolicy $\cdot \pi _ { \phi } ^ { \mathrm { s u b } }$ , and policy $\pi _ { \theta }$   
3: while not converged do   
4: Train value function: $\beta \gets \beta - \lambda \nabla _ { \beta } \mathcal { L } _ { \mathrm { G C I V L } } ( \beta )$ with $( s _ { t } , s _ { t + 1 } ) \sim p ^ { \mathcal { D } } , g \sim p ( g )$ [Equation 9]   
5: end while   
6: while not converged do   
7: Train target subpolicy: $\phi  \phi + \lambda \nabla _ { \phi } \mathcal { T } _ { \mathrm { A W R } } ( \phi )$ with $( s _ { t } , a , s _ { g } ) \sim p ^ { \mathcal { D } }$ [Equation 13]   
8: end while   
9: while not converged do   
10: Train policy: $\breve { \theta }  \theta - \lambda \nabla _ { \theta } ( \mathcal { I } _ { \mathrm { S B } } ( \theta ) - \mathcal { I } _ { \mathrm { S A W } } ( \theta ) )$ with $( s _ { t } , a , s _ { g } ) \sim p ^ { D } , g \sim p ( g )$ [Equation   
4 & 12]   
11: end while

## D ALGORITHMIC DETAILS

In this section, we provide full details of the value and policy learning objectives for our Sobolev bootstrapping algorithm, which is instantiated on top of Subgoal Advantage-Weighted Policy Bootstrapping (Zhou & Kao, 2025, SAW).

## D.1 GOAL-CONDITIONED IMPLICIT VALUE LEARNING (GCIVL)

To learn a value function $V ( s , g )$ that can operate both on single- and multi-step transitions, we use a goal-conditioned variant of implicit Q-learning (IQL) (Kostrikov et al., 2021), which approximates the max operator with an expectile regression on in-sample state-action pairs:

$$
\mathcal { L } _ { V } ( \beta _ { V } ) = \mathbb { E } _ { ( s , a , g ) \sim D } \left[ L _ { 2 } ^ { \tau } ( Q _ { \hat { \beta } _ { Q } } ( s , a , g ) - V _ { \beta _ { V } } ( s , g ) ) \right]\tag{7}
$$

$$
\begin{array} { r } { \mathcal { L } _ { Q } ( \beta _ { Q } ) = \mathbb { E } _ { ( s , a , s ^ { \prime } , g ) \sim D } \left[ ( r ( s , g ) + \gamma V _ { \beta _ { V } } ( s ^ { \prime } , g ) - Q _ { \beta _ { Q } } ( s , a , g ) ) ^ { 2 } \right] , } \end{array}\tag{8}
$$

where $L _ { 2 } ^ { \tau } ( x ) = | \tau - \mathbb { 1 } ( x < 0 ) | x ^ { 2 }$ is the expectile loss and $\tau \in [ 0 . 5 , 1 )$ is the expectile parameter. Following previous works (Park et al., 2023), we adopt an action-free variant of this objective, which estimates the state-action value function $Q ( s , a , g )$ as $r ( s , g ) + \gamma \bar { V } ( s ^ { \prime } , g )$ , leading to

$$
\mathcal { L } _ { \mathrm { G C I V L } } ( \beta ) = \mathbb { E } _ { ( s , s ^ { \prime } , g ) \sim D } \left[ L _ { 2 } ^ { \tau } ( r ( s , g ) + \gamma V _ { \bar { \beta } } ( s ^ { \prime } , g ) - V _ { \beta } ( s , g ) ) \right] .\tag{9}
$$

Note that using the action-free estimator with expectile regression is optimistically biased in stochastic environments, since it directly regresses towards high-value transitions without using Q-values to marginalize over environment stochasticity. However, it performs well in practice and can be used to measure the relative advantages of intermediate subgoals without modification.

## D.2 OFFLINE POLICY EXTRACTION

The above value iteration algorithm already performs implicit policy improvement. Then, we simply need to extract the optimal policy from the learned value function. To remain inside the convex hull of dataset actions and avoid querying the value function for out-of-distribution actions, we use a form of weighted behavioral cloning called Advantage-Weighted Regression (Peng et al., 2019, AWR):

$$
\mathcal { I } _ { \pi _ { \theta } } ( \theta ) = \mathbb { E } _ { ( s , a , g ) \sim D } \left[ \exp ( \alpha ( \bar { Q } ( s , a , g ) - V ( s , g ) ) ) \log \pi _ { \theta } ( a \mid s , g ) \right] ,\tag{10}
$$

where $\alpha \in [ 0 , \infty )$ is an inverse temperature parameter. Because we do not directly parameterize a critic $Q ( s , a , g )$ in GCIVL [Equation 9], we instead estimate the $\mathrm { \bf \ddot { \tau } o n e - s t e p ^ { \prime } }$ advantage $\bar { Q } ( s , a , g ) \_$ $V ( s , g )$ as $V ( s ^ { \prime } , g ) - V ( s , g )$ to reweight the corresponding action in the transition tuple $( s , a , s ^ { \prime } )$ This estimate is biased in stochastic environments, since it optimistically assumes that the transition from s to $s ^ { \prime }$ was caused by action a.

## D.3 POLICY BOOTSTRAPPING ALGORITHMS

Policy bootstrapping algorithms are based on the insight that, if an action a is optimal for reaching a subgoal $s _ { g }$ along the optimal trajectory toward goal g, then a should also be optimal for reaching g. To evaluate the optimality of subgoals $s _ { g }$ , we can use a multistep variant of the action-free advantage estimator for GCIVL, where we approximate $A ( s , s _ { g } , g )$ as $V ( s _ { g } , g ) - V ( s , g )$

Reinforcement Learning with Imagined Subgoals [RIS] (Chane-Sane et al., 2021): To produce subgoals, RIS first learns a generative model $\pi _ { \psi } ^ { h } ( s _ { g } \mid s , g )$ to produce “imagined” subgoals $s _ { g }$ , and then bootstraps a goal-conditioned flat policy $\pi _ { \theta } ( a \mid s , g )$ on a short-horizon policy $\pi ^ { \mathrm { s u b } } ( a \mid s , s _ { g } )$ conditioned on the imaginary way-points.

The generative model $\pi _ { \psi } ^ { h } ( s _ { g } \mid s , g )$ is optimized to produce subgoals with high advantages while remaining close to the dataset state distribution $\mu _ { D }$ , in a manner similar to AWR:

$$
\mathcal { T } _ { \pi _ { \psi } ^ { h } } ( \psi ) = \mathbb { E } _ { ( s , s _ { g } , g ) \sim D } \left[ \exp ( \alpha A ( s _ { g } \mid s , g ) ) \log \pi _ { \psi } ^ { h } ( s _ { g } \mid s , g ) \right] .\tag{11}
$$

Subgoal Advantage-Weighted Policy Bootstrapping [SAW] (Zhou & Kao, 2025): SAW introduces a unifying view of RIS and a related hierarchical method (Park et al., 2023) under the “RL as inference” framework (Levine, 2018). Rather than modeling the optimal subgoal distribution $p ^ { * } ( s _ { g } \mid s , g )$ . SAW samples in-trajectory waypoints in dataset and uses a similar multistep advantage weight to prioritize bootstrapping on high-advantage subgoals.

$$
p ^ { * } ( s _ { g } \mid s , g ) \propto p ^ { D } ( s _ { g } \mid s ) \exp ( A ( s _ { g } \mid s , g ) ) .
$$

This eliminates the need for a parametric subgoal generator and replaces Equation 11 and 14 with

$$
\begin{array} { r } { \mathcal { I } _ { \pi _ { \theta } } ^ { \mathrm { S A W } } ( \theta ) = \mathbb { E } _ { ( s , a , s _ { g } , g ) \sim D } \left[ e ^ { \alpha _ { 1 } A ( a | s , g ) } \log \pi _ { \theta } ( a \mid s , g ) - e ^ { \alpha _ { 2 } A ( s _ { g } | s , g ) } D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi ^ { \mathrm { s u b } } ) \right] , } \end{array}\tag{12}
$$

where the target subpolicy $\pi ^ { \mathrm { s u b } }$ is extracted with Equation 13, π is a shorthand for $\pi _ { \theta } ^ { f } ( \cdot \mid s , g )$ , and $\pi ^ { \mathrm { s u b } }$ denotes $\pi ^ { \mathrm { { s u b } } } ( \cdot \mid s , s _ { g } )$

Following Zhou & Kao (2025), we implement the offline variant of RIS and SAW with an independently parameterized short-horizon policy with AWR, instead of using an exponential moving average of the long-horizon policy $\pi ^ { \theta }$ as in the original implementation by Chane-Sane et al. (2021):

$$
\begin{array} { r } { \mathcal { I } _ { \pi ^ { \mathrm { s u b } } } ( \phi ) = \mathbb { E } _ { ( s , a , s _ { g } ) \sim D } \left[ \exp ( \alpha A ( a \mid s , s _ { g } ) ) \log \pi ^ { \mathrm { s u b } } ( a \mid s , s _ { g } ) \right] . } \end{array}\tag{13}
$$

Finally, we train the long-horizon policy $\pi _ { \theta }$ with “one-step” transitions while regressing it toward the short-horizon policy $\pi ^ { \mathrm { s u b } }$ conditioned on generated subgoals:

$$
\begin{array} { r } { \mathcal { I } _ { \pi _ { \theta } } ^ { \mathrm { R I S } } ( \theta ) = \mathbb { E } _ { ( s , a , g ) \sim D , s _ { g } \sim \pi _ { \psi } ^ { h } ( \cdot \vert s , g ) } \left[ \exp ( \alpha A ( a \vert s , g ) ) \log \pi _ { \theta } ( a \vert s , g ) - D _ { \mathrm { K L } } ( \pi _ { \theta } \Vert \pi ^ { \mathrm { s u b } } ) \right] , } \end{array}\tag{14}
$$

where the “one-step” advantage $A ( a \mid s , g )$ is estimated by $V _ { \beta _ { V } } ( s ^ { \prime } , g ) - V _ { \beta _ { V } } ( s , g )$ , π<sub>θ</sub> is a shorthand for $\pi _ { \theta } ^ { f } ( \cdot \mid s , g )$ , and $\pi ^ { \mathrm { s u b } }$ denotes $\pi ^ { \mathrm { { s u b } } } ( \cdot \mid s , s _ { g } )$

## E ADDITIONAL FIGURES

![](images/d66dbcd8f7dd127740e950295c7ba350589e2d4e962933ef7b392d8710dd9cea.jpg)  
Figure 5: Training on distant relabeled goals harms nearby goal-reaching capabilities. We record the success rates across learned policies trained on a uniform distribution of relabeling goal offsets $[ 1 , H _ { \mathrm { t r a i n } } ]$ , where $H _ { \mathrm { t r a i n } } = \infty$ corresponds to a uniform distribution over all future in-trajectory states. At evaluation, policies are given optimal $\bar { H } _ { \mathrm { p l a n } }$ -step subgoals from oracle planners. Success rates are averaged over the last three checkpoints and three seeds.

![](images/78fcfccf4acac1df56b9a95b4aaa29cb10d8a0a6dc611ead266e7c2a82f9f213.jpg)  
Figure 6: Training on a larger proportion of distant relabeled goals harms nearby goal-reaching capabilities. We record the success rates across learned policies trained on a geometric distribution of relabeling goal offsets with discount $\gamma ,$ where $\gamma = 1$ corresponds to a uniform distribution over future states. At evaluation, policies are given optimal $H _ { \mathrm { p l a n } }$ -step subgoals from oracle planners. Success rates are averaged over the last three checkpoints and three seeds.

![](images/2ae0d3d9f881bcc7cb7cc29d7ded90b2ac336191185b4b32cad51db23d33e123.jpg)  
Figure 7: Training on more random relabeled goals harms nearby goal-reaching capabilities. We record the success rates across learned policies trained on a mixture of goals either sampled from a uniform distribution over future states or from an arbitrary dataset state with probability $p _ { \mathrm { r a n d } } \in \{ 0 . 2 5 , 0 . 5 0 , 0 . 7 5 \}$ . At evaluation, policies are given optimal $H _ { \mathrm { p l a n } }$ -step subgoals from oracle planners. Success rates are averaged over the last three checkpoints and three seeds.

![](images/0ce4c4218f5e15077105f6d478b51df1edffd198a03e5b0a3d505dd83227147a.jpg)  
Figure 8: Increasing training goal horizon globally decreases policy goal sensitivity. We measure the proportion of the input Jacobian norm corresponding to the goal at fixed state-goal offsets, plotted with log scaling. Results are averaged over the last three training checkpoints, three seeds, and 1024 randomly sampled state-goal pairs per horizon.

![](images/5ca8e587ffeb34d7081bbbfa942e58e82edad95f25f6f9850c9591bb114c9671.jpg)  
Figure 9: Training on distant relabeled goals globally decreases policy goal sensitivity. We measure the proportion of the input Jacobian norm corresponding to the goal at fixed state-goal offsets, plotted with log scaling. Results are averaged over the last three training checkpoints, three seeds, and 1024 randomly sampled state-goal pairs per horizon.

![](images/945fa1200e59c4db50dfc98586cfa43782cc921643671f52b6265c5520f03b2c.jpg)  
Figure 10: Random goal sampling globally decreases policy goal sensitivity. We directly reduce the actiongoal MI by replacing in-trajectory goals with randomly sampled dataset states with probability $p _ { \mathrm { r a n d o m } } \in$ {0.25, 0.5, 0.75}, and measure the goal Jacobian norm proportion at fixed state-goal offsets.

![](images/34f7c5c83dff49eeb3f4156eae3d176a27055fe01c5052f7ae1088e6f5a1fb5b.jpg)  
Figure 11: Sobolev bootstrapping pays attention to the correct object during optimal trajectories. To enable fair comparisons across later task stages, we plot the input Jacobian norms with respect to each object over an independent scripted optimal trajectory Ahn et al. (2025) for the cube-triple triple pick-and-place task. $\mathrm { S ^ { 2 } A W }$ (top) allocates a higher proportion of the Jacobian norm to the focal cube both during approach and grasp, whereas SAW’s (bottom) Jacobian norm only changes after the cube is already grasped. The shaded area denotes when the cube is actively grasped.

## F HYPERPARAMETERS

We provide hyperparameters related to our Sobolev bootstrapping objective in Table 3. We use value learning hyperparameters from Zhou & Kao (2025) for HIQL, SAW, and S<sup>2</sup>AW in cube-single, which significantly improves performance. If not listed, all other hyperparameters follow the default values from Park et al. (2024) and Zhou & Kao (2025).

<table><tr><td>Dataset</td><td>Expectile τ</td><td>AWRα</td><td>KLDβ</td><td>Subgoal steps k</td><td>Sobolev weight λ</td></tr><tr><td>pointmaze-medium-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>25</td><td>0.03</td></tr><tr><td>pointmaze-large-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>25</td><td>1.00</td></tr><tr><td> $\mathtt { p o i n t m a z e - g i a n t - n a v i g a t e - v 0 }$ </td><td>0.7</td><td>3.0</td><td>3.0</td><td>25</td><td>1.00</td></tr><tr><td>antmaze-medium-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>25</td><td>0.03</td></tr><tr><td>antmaze-large-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>25</td><td>0.03</td></tr><tr><td>antmaze-giant-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>25</td><td>0.30</td></tr><tr><td>humanoidmaze-medium-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>100</td><td>0.03</td></tr><tr><td>humanoidmaze-large-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>100</td><td>0.03</td></tr><tr><td>humanoidmaze-giant-navigate-v0</td><td>0.7</td><td>3.0</td><td>3.0</td><td>100</td><td>0.03</td></tr><tr><td>cube-single-play-v0</td><td>0.9</td><td>3.0</td><td>0.3</td><td>10</td><td>0.02</td></tr><tr><td>cube-double-play-v0</td><td>0.7</td><td>3.0</td><td>1.0</td><td>10</td><td>0.02</td></tr><tr><td>cube-triple-play-v0</td><td>0.7</td><td>3.0</td><td>1.0</td><td>10</td><td>0.03</td></tr><tr><td>cube-quadruple-play-v0</td><td>0.7</td><td>3.0</td><td>1.0</td><td>10</td><td>0.03</td></tr><tr><td>scene-play-v0</td><td>0.7</td><td>3.0</td><td>1.0</td><td>10</td><td>0.02</td></tr></table>

Table 3: Hyperparameters for SAW with Sobolev bootstrapping.

## F.1 SENSITIVITY ANALYSIS

We examine sensitivity to the choice of Sobolev weight λ in Table 4. We do not perform additional tuning of the expectile τ, AWR α, subgoal steps k, and KLD β, which are common hyperparameters shared by HIQL and SAW. Our results show that smaller weights can be insufficient to realize the benefits of Sobolev training, while larger weights may “oversmooth” in some manipulation environments. Nevertheless, we observe consistently substantial gains in cube-triple and antmaze-giant across all values of λ.

<table><tr><td>Dataset</td><td>λ = 0 0.001 0.01</td><td></td><td></td><td>0.1</td></tr><tr><td> $\mathtt { c u b e - d o u b l e - p l a y - v 0 }$ </td><td>40 ±7 40 ±11 55 ±7 35 ±9</td><td></td><td></td><td></td></tr><tr><td> $\mathtt { c u b e - t r i p l e - p l a y - v 0 }$ </td><td>4 ±2 9 ±2 30 ±7 23 ±10</td><td></td><td></td><td></td></tr><tr><td> $\mathtt { s c e n e - p l a y - v 0 }$ </td><td>63 ±6 54 ±10 71 ±5 62 ±5</td><td></td><td></td><td></td></tr><tr><td> $\mathtt { a n t m a z e - g i a n t - n a v i g a t e - v 0 }$ </td><td></td><td>73 ±4 76 ±4 82 ±4 85 ±4</td><td></td><td></td></tr></table>

Table 4: Sobolev loss weight sensitivity analysis. Sobolev bootstrapping is robust to different weighting factors within a reasonable scale (around 0.01 for cube-double and scene). Notably, improvements on antmaze-giant and cube-triple are robust across various weighting scales. Experimental results are averaged across four seeds.

## G COMPUTATIONAL RESOURCES

All experiments were run on an internal cluster of NVIDIA 5090 and NVIDIA PRO 6000 Blackwell GPUs. In Table 5, We select the highest-dimensional environments and report the hours required to complete a standard 1M-step training run on an NVIDIA 5090 GPU.

<table><tr><td>hours / 1M iters</td><td>GCIQL</td><td>SAW</td><td>S2AW</td></tr><tr><td>cube-quadruple-play</td><td>0.26</td><td>0.46</td><td>0.75</td></tr><tr><td>scene-play</td><td>0.25</td><td>0.45</td><td>0.74</td></tr><tr><td>humanoidmaze-giant-navigate</td><td>0.26</td><td>0.46</td><td>1.75</td></tr></table>

Table 5: Training time comparison. We select the highest-dimensional environments and report the hours required to complete a standard 1M-step training run on an NVIDIA 5090 GPU.

## H ADDITIONAL EXPERIMENTS

## H.1 ARCHITECTURAL ABLATIONS

While we have made the case that differences in state-goal distributions are responsible for the improved properties of the target policy $\pi ^ { \mathrm { s u b } }$ , we note that the experiments in Table 1 use different architectures for each policy. The full goal-conditioned policy π consists of a three-layer MLP, followed by a length-normalized bottleneck of dimension $^ { 1 0 , }$ and another three-layer MLP; the target policy consists only of a three-layer MLP. Then, our results may be explained by a natural simplicity bias introduced by smaller network sizes. To control for this potential confound, we train Sobolev SAW using identical architectures for $\pi ^ { \mathrm { s u b } }$ and π, which we denote as $\mathbf { S } ^ { 2 } \mathbf { A } \mathbf { W } _ { \ell }$

<table><tr><td>Environment</td><td>Dataset</td><td>GCBC</td><td>GCIVL</td><td>GCIQL</td><td>QRL</td><td>CRL</td><td>HIQL</td><td>SAW</td><td>S2AW</td><td>S2AWe</td></tr><tr><td rowspan="2">maze-giant</td><td>antmaze-giant-navigate-v0</td><td>0 ±0</td><td>0 ±0</td><td>0 ±0</td><td>14 ±3</td><td>16 ±3</td><td>65 ±5</td><td>73 ±4</td><td>87 ±3</td><td>80 ±5</td></tr><tr><td>humanoidmaze-giant-navigate-v0</td><td>0 ±0</td><td>0 ±0</td><td>0 ±0</td><td>1 ±0</td><td>3 ±2</td><td>12 ±4</td><td>35 ±4</td><td>38 ±6</td><td>35 ±5</td></tr><tr><td rowspan="4">cube</td><td> $\mathtt { c u b e - s i n g l e - p l a y - v 0 }$ </td><td>6 ±2</td><td>53 ±4</td><td>68 ±6</td><td>5 ±1</td><td>19 ±2</td><td>44 ±9</td><td>72 ±5</td><td>86 ±5</td><td>79 ±8</td></tr><tr><td> $\mathtt { c u b e - d o u b l e - p l a y - v 0 }$ </td><td>1 ±1</td><td>36 ±3</td><td>40 ±5</td><td>1 ±0</td><td> $1 0 \pm 2$ </td><td>6 ±2</td><td>40 ±7</td><td>50 ±6</td><td>55 ±8</td></tr><tr><td> $\mathtt { c u b e - t r i p l e - p l a y - v 0 }$ </td><td>1±1</td><td>1 ±0</td><td>3±1</td><td>0 ±0</td><td> $^ { 4 \pm 1 }$ </td><td>3±1</td><td>4 ±2</td><td>30 ±5</td><td>30 ±4</td></tr><tr><td> $\mathtt { c u b e - q u a d r u p l e - p l a y - v 0 }$ </td><td>0 ±0</td><td>0 ±0</td><td>0 ±0</td><td>0 ±0</td><td> $0 \pm 0$ </td><td>0 ±0</td><td>0 ±0</td><td>3 ±2</td><td>2 ±1</td></tr><tr><td>scene</td><td>scene-play-v0</td><td> $5 \pm 1$ </td><td>42 ±4</td><td>51 ±4</td><td>5 ±1</td><td> $1 9 \pm 2$ </td><td>38 ±3</td><td>63 ±6</td><td>73 ±6</td><td>72 ±5</td></tr></table>

Table 6: Architectural ablations to the Sobolev SAW target actor. We report the performance of Sobolev SAW where the target actor has the same architecture as the full policy, averaged over 4 seeds with standard deviations after the ± sign. Numbers within 5% of the best value in the row are in bold.

We find that results remain consistent across all manipulation environments where combinatorial generalization is most important, supporting the primary conclusion that the difference in training curricula, and not architectural differences, produces desirable properties of input gradients. Performance does drop slightly in locomotion environments compared to Sobolev SAW with the smaller target actor, suggesting that a smaller actor network does provide some beneficial regularization. Indeed, we observe that Sobolev bootstrapping with a smaller target policy achieves lower SAW validation loss than its counterpart.

## H.2 EXTENDING SOBOLEV BOOTSTRAPPING TO RIS

Adding Sobolev bootstrapping to RIS (Chane-Sane et al., 2021) also yields significant gains, albeit less consistently. In our implementation of RIS, subgoals generated by $\pi _ { \psi } ^ { h }$ lie in a lower-dimensional latent space rather than the raw observation space, making it impractical to compute advantages directly from the learned value function $V _ { \beta _ { V } }$ . Thus, we adopt a constant weighting scheme for the Sobolev bootstrapping loss, under the assumption that all generated subgoals are near-optimal. If the subgoal generator proposes suboptimal waypoints, then the uniformly weighted Sobolev term in RIS can force the flat policy to match noisy input gradients, which can degrade its performance.

Furthermore, it is well-known that planning with a latent subgoal representation extracted from an intermediate layer of the value function, as done in Park et al. (2023) and our implementation of RIS, is suboptimal and can significantly hamper performance in larger environments such as antmaze-giant and humanoidmaze-giant (Zhou & Kao, 2025; Ahn et al., 2025). Although not a focus of our work, we expect that improvements to the high-level policy, such as modern generative models and new representation learning objectives, may significantly improve RIS’s performance.

## H.3 ADDITIONAL HORIZON REDUCTION BASELINES

Existing horizon-reduction approaches address one of two issues: compounding bias in off-policy value learning, e.g., n-step TD (Park et al., 2025), QC (Li et al., 2025; 2026b), OTA (Ahn et al., 2025); or noisy advantage estimates during policy extraction, e.g., HIQL (Park et al., 2023), SAW (Zhou & Kao, 2025). In contrast, our work identifies an additional bottleneck within the actor itself, related to decreased policy sensitivity to goal information scaling with horizon.

For completeness, we compare our method against OTA (Ahn et al., 2025), DQC (Li et al., 2026b), and SHARSA (Park et al., 2025). Because the original OTA paper did not report results on statebased manipulation tasks, we use the official implementation and sweep the abstraction factor

<table><tr><td>Dataset</td><td>RIS</td><td>Sobolev RIS</td></tr><tr><td> $\mathtt { p o i n t m a z e - m e d i u m - n a v i g a t e - v 0 }$ </td><td> $8 8 \pm 6$ </td><td> ${ \bf 9 6 } _ { \pm 1 }$ </td></tr><tr><td> $\mathtt { p o i n t m a z e - l a r g e - n a v i g a t e - v 0 }$ </td><td> $6 3 \pm 1 3$ </td><td> ${ \bf 7 5 \pm 3 }$ </td></tr><tr><td> $\mathtt { p o i n t m a z e - g i a n t - n a v i g a t e - v 0 }$ </td><td> $5 7 \pm 1 2$ </td><td> ${ \bf 6 7 \pm 5 }$ </td></tr><tr><td> $\mathtt { a n t m a z e - m e d i u m - n a v i g a t e - v 0 }$ </td><td> ${ \bf 9 6 } _ { \pm 1 }$ </td><td> ${ \bf 9 7 \pm 1 }$ </td></tr><tr><td> $\mathtt { a n t m a z e - l a r g e - n a v i g a t e - v 0 }$ </td><td> ${ \bf 8 9 \pm 3 }$ </td><td> ${ \bf 9 0 } _ { \pm 1 }$ </td></tr><tr><td> $\mathtt { a n t m a z e - g i a n t - n a v i g a t e - v 0 }$ </td><td> $\mathbf { 6 5 \pm } \mathbf { 4 }$ </td><td> ${ \bf 6 9 \pm 0 }$ </td></tr><tr><td>humanoidma  $\mathtt { z e - m e d i u m - n a v i g a t e - v 0 }$ </td><td> $7 3 \pm 5$ </td><td> ${ \bf 8 4 } \pm \mathrm { 7 }$ </td></tr><tr><td>humanoidma  $\mathtt { z e - l a r g e - n a v i g a t e - v 0 }$ </td><td> $2 1 \pm 7$ </td><td> ${ \bf 4 7 \pm 2 }$ </td></tr><tr><td>humanoidma  $\mathtt { . z e - g i a n t - n a v i g a t e - v 0 }$ </td><td> $3 \pm 2$ </td><td> ${ \bf 1 3 } \pm 3$ </td></tr><tr><td> $\mathtt { c u b e - s i n g l e - p l a y - v 0 }$ </td><td> ${ \bf 8 1 \pm 6 }$ </td><td> $5 9 \pm 3$ </td></tr><tr><td> $\mathtt { c u b e - d o u b l e - p l a y - v 0 }$ </td><td> $3 6 \pm 4$ </td><td> ${ \bf 4 1 \pm 2 }$ </td></tr><tr><td> $\mathtt { c u b e - t r i p l e - p l a y - v 0 }$ </td><td> ${ \bf 3 } _ { \pm 2 }$ </td><td> ${ \bf 3 } _ { \pm 0 }$ </td></tr><tr><td> $\mathtt { c u b e - q u a d r u p l e - p l a y - v 0 }$ </td><td> $\mathbf { 0 } \pm \mathbf { 0 }$ </td><td> $\mathbf { 0 } \pm \mathbf { 0 }$ </td></tr><tr><td>scene-play-v0</td><td> ${ \bf 6 4 } \pm \mathrm { 7 }$ </td><td> ${ \bf f 6 6 } \pm \bf { 1 }$ </td></tr></table>

Table 7: Evaluating RIS with Sobolev bootstrapping. We evaluate the effectiveness of Sobolev bootstrapping on RIS by comparing it against the numbers reported in Zhou & Kao (2025). Results are averaged over 4 seeds, and other settings are identical to Table 1.

$n \in \{ 5 , 1 0 , 2 0 \}$ . Furthermore, adapting DQC and SHARSA to our standard dataset is non-trivial, as they were evaluated using larger datasets with oracle representations. To ensure a fair comparison, we adopt the documented hyperparameters and use the standard batch and model sizes consistent with the OGBench suite. Our results demonstrate that OTA, DQC, and SHARSA underperform with the standard datasets.
<table><tr><td>Environment</td><td>OTA n=5</td><td> $\mathbf { O T A \ n { = } 1 0 }$ </td><td> $\mathbf { O T A _ { \lambda } n } \mathbf { = } 2 \mathbf { 0 }$ </td><td>SHARSA DQC S2AW</td><td></td><td></td></tr><tr><td> $\mathtt { a n t m a z e - g i a n t - n a v i g a t e }$ </td><td> $7 7 \pm 4$ </td><td>1</td><td></td><td> $7 2 \pm 1$ </td><td> $2 5 { \pm } 6$ </td><td> $^ { 8 7 \pm 3 }$ </td></tr><tr><td> $\mathtt { c u b e - d o u b l e - p l a y }$ </td><td> $2 { \pm } 0$ </td><td> $3 { \pm } 1$ </td><td> $2 { \pm } 1$ </td><td> $5 { \pm } 1$ </td><td> $2 4 { \pm } 4$ </td><td> $5 7 \pm 6$ </td></tr><tr><td> $\mathtt { c u b e - t r i p l e - p l a y }$ </td><td> $1 { \pm } 0$ </td><td> $_ { 0 \pm 0 }$ </td><td> $1 { \pm } 0$ </td><td> $2 { \pm } 1$ </td><td> $1 1 { \pm } 4$ </td><td> $3 0 { \pm } 5$ </td></tr></table>

Table 8: Additional horizon reduction baselines. The additional baselines do not readily generalize to the standard datasets and perform poorly. All results are averaged across 4 seeds.

## H.4 UNIFORM JACOBIAN REGULARIZATION

<table><tr><td></td><td>SAW</td><td> $\mathbf { S A W } _ { \ell _ { 1 } }$ </td><td> $\mathbf { S A W } _ { \ell _ { 2 } }$ </td><td> $\mathbf { S } ^ { 2 } \mathbf { A } \mathbf { W }$ </td></tr><tr><td> $\mathtt { a n t m a z e - g i a n t - n a v i g a t e }$ </td><td> $7 3 { \pm } 4$ </td><td> $7 2 \pm 3$ </td><td> $7 4 \pm 2$ </td><td> $8 7 \pm 3$ </td></tr><tr><td> $\mathtt { c u b e - d o u b l e - p l a y }$ </td><td> $4 0 { \pm } 7$ </td><td> $4 7 \pm 1 0$ </td><td> $3 7 \pm 7$ </td><td> $5 7 \pm 6$ </td></tr><tr><td> $\mathtt { c u b e - t r i p l e - p l a y }$ </td><td> $4 { \pm } 2$ </td><td> $5 { \pm } 4$ </td><td> $4 { \pm } 1$ </td><td> $3 0 { \pm } 5$ </td></tr><tr><td> $\mathtt { s c e n e - p l a y }$ </td><td> ${ 6 3 \pm 6 }$ </td><td> $_ { 6 9 \pm 8 }$ </td><td> $6 2 \pm 2$ </td><td> $7 4 \pm 6$ </td></tr></table>

Table 9: Uniform Jacobian regularization. We apply $\ell _ { 1 }$ and $\ell _ { 2 }$ norm penalization with coefficient 0.3 to the SAW baseline. We observe that $\ell _ { 1 }$ penalization is more effective, leading to notable performance gains in cube-double and scene environments. Results are averaged across four seeds and five evaluation tasks.