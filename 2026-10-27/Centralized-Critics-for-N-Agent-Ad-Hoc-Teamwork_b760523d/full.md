# Centralized Critics for N-Agent Ad Hoc Teamwork

Andrew McAlinden

Caroline Wang

Steven Shi

Arrasy Rahman

Elad Liebman

Peter Stone

## Abstract

Centralized training with decentralized execution (CTDE) has become the dominant paradigm in multi-agent reinforcement learning (MARL), motivating extensive study of what privileged information should be made available to the critic during training. While prior work has examined the role of global state in fully cooperative settings, the design of centralized critics for N-agent ad hoc teamwork—in which dynamically varying set of learning agents must coordinate with an arbitrary set of uncontrolled teammates drawn from an unknown distribution—remains poorly understood. We study this question by comparing two natural classes of advantage estimators: controlled history estimators, which condition only on the observation and action histories of the learner’s own agents, and joint history estimators, which additionally condition on the histories of the uncontrolled teammates. We prove that both estimators are unbiased and characterize the conditions under which the joint history estimate attains strictly lower variance than its controlled counterpart. Empirically, joint history critics match or outperform controlled history critics across a range of NAHT tasks, yielding improved sample efficiency and final returns. Together, these results provide both a theoretical justification and an empirical recommendation for incorporating uncontrolled agent histories into centralized critics for N-agent ad hoc teamwork.

## 1 Introduction

Cooperative multi-agent reinforcement learning (CMARL) is a paradigm for learning coordinated team policies on fully cooperative tasks, assuming full control over all agents [Gronauer and Diepold, 2022]. In contrast, ad hoc teamwork (AHT) considers the challenge of training a single agent to collaborate with previously unknown teammates on fully cooperative tasks [Stone et al., 2010]. The recently introduced N-agent ad hoc teamwork (NAHT) problem generalizes both paradigms by considering the scenario where a set of controlled agents must collaborate with dynamically varying numbers and types of uncontrolled teammates without the ability to pre-coordinate. The primary challenges of the NAHT problem are (1) dealing with a changing number of (un)controlled agents, and (2) generalizing to unseen uncontrolled teammate behavior [Wang et al., 2024].

For the NAHT problem, we assume that controlled agents only have access to controlled agent information, as the number of uncontrolled agents changes per episode, and the controlled agents must generalize to differing types of uncontrolled agents without being given direct information about their partners at inference time. Previous successful algorithms for NAHT have chosen to also include this restriction for critics, with algorithms [Wang et al., 2025b, 2024, 2025a] only including controlled agent information in critics. The CTDE paradigm then poses the question of if uncontrolled agent information can be helpful in training while still restricting this information during inference.

Since the purpose of critics is to estimate the advantage function, it follows that better advantage estimation should lead to better sample efficiency when training policies. We study this question by comparing two natural classes of advantage estimators: controlled history estimators, which condition only on the observation and action histories of the learner’s own agents, and joint history estimators, which additionally condition on the histories of the uncontrolled teammates. We prove in Section 4 that both estimators are unbiased and characterize the conditions under which the joint history estimate attains strictly lower variance than its controlled counterpart. Empirically (Section 6), joint history critics match or outperform controlled history critics across a range of NAHT tasks, yielding improved sample efficiency and final returns. Together, these results provide both a theoretical justification and an empirical recommendation for incorporating uncontrolled agent histories into centralized critics for N-agent ad hoc teamwork.

## 2 Related Work

Previous algorithms for NAHT rely only on controlled agent information in critics. There is no existing study on including privileged information in centralized critics for NAHT, although this kind of analysis exists for MARL. The most relevant previous work focuses on the variance of the Q function and policy gradients for different types of MARL critics.

Previous Algorithms for NAHT. POAM [Wang et al., 2024] was the first proposed algorithm for NAHT. It uses decentralized, RNN-based critics that attempt to model the value function using only information from an individual controlled agent. Similarly, the Shapley Machine algorithm [Wang et al., 2025b] uses the POAM critic, but models a different value function per-agent. The MAT-NAHT algorithm [Wang et al., 2025a] is the first to use a single attention-based centralized critic with information from all controlled agents. The same paper introduced Cent Obs POAM, which uses a single centralized RNN-based critic with information from all controlled agents. Since our work is focused on centralized critics, we use MAT-NAHT and Cent Obs Poam for experiments, and abbreviate both methods to “MAT” and “POAM.”

Centralized Critics in MARL. A separate line of work has analyzed the role of privileged information in centralized critics for cooperative MARL. Lyu et al. [2023] provide an extensive theoretical and empirical study of history-based, state-based, and history-state-based critics for Dec-POMDPs. Among their results, they show that centralized history-based critics yield the same expected policy gradient as decentralized history-based critics but with strictly higher variance, that state-based critics may introduce unbounded bias when used to train history-based policies, and that history-state critics are unbiased but again carry higher variance than purely history-based critics. While their analysis covers what kind of global information should be exposed to the critic in MARL, the NAHT setting introduces a different design choice: whether the critic should condition on the trajectories of agents that are not under the learner’s control. Our work extends the bias-variance style of analysis pioneered by Lyu et al. [2023] to this setting, deriving an unbiasedness result and a variance comparison between controlled-history and joint-history advantage estimators in the NAHT problem. Since we use PPO for optimization instead of the vanilla policy gradient, we focus on analyzing the bias and variance of advantage function.

## 3 Background

The N-agent ad hoc teamwork (NAHT) problem is formalized in the framework of Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) [Bernstein et al., 2002]. A Dec-POMDP is described by the tuple $\langle M , S , \{ O ^ { i } \} _ { i = 1 } ^ { M } , A , T , r , \gamma \rangle$ . Here, M denotes the number of agents, S the global state space, $O ^ { i }$ the local observation space for agent i, and $\textstyle A = \prod _ { i = 1 } ^ { M } A ^ { i }$ the joint action space. Let the notation $\Delta ( \bullet )$ denote a probability distribution over the space •. The transition function $T : S \times A \to \Delta ( S )$ governs the dynamics, and the reward function $r : S \times A $ R provides a shared reward signal. In this work, we examine a control scheme where each agent selects actions according to the stochastic policy $\pi ^ { i } : H ^ { i } \times A ^ { 1 } \times \cdot \cdot \cdot A ^ { i - 1 } \to \Delta ( A ^ { i } )$ , where $H ^ { i }$ is the fixed length history of observations and actions available to agent i. The decision-making problem occurs under the horizon T.

The NAHT problem considers a controlled set of agents C, and an uncontrolled set U. The learning algorithm generates policies for agents in C, while uncontrolled agents in U follow fixed, potentially unknown policies. Each episode begins with a team sampling process $X ( U , C )$ , consisting of two steps: first, the number of controlled agents $N < M$ is sampled, and second, a set of

$N$ agents is drawn from the controlled set $C ,$ while $M - N$ agents are drawn from the uncontrolled set $U .$ The NAHT learning problem considers a parameterized set of controlled agents, $C ( \theta )$ , and aims to maximize the expected return under varying teammates and team compositions: max<sub>θ</sub> $\begin{array} { r } { \mathbb { E } _ { \{ \pi ^ { i } \} _ { i = 1 } ^ { M } \sim X ( U , C ( \theta ) ) } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } \right] } \end{array}$

A trajectory $\tau = ( s _ { 0 } , a _ { 0 } , r _ { 0 } , s _ { 1 } , a _ { 1 } , r _ { 1 } , \dots , s _ { T } )$ is sampled under the joint policy: at each timestep, the controlled agents select $a _ { t } ^ { c } \sim \pi ( \cdot \mid h _ { t } ^ { c } )$ and the uncontrolled agents select $a _ { t } ^ { u } \sim \pi ^ { u } ( \cdot \mid h _ { t } ^ { u } )$ , with joint action $a _ { t } = \left( a _ { t } ^ { c } , a _ { t } ^ { u } \right)$ , rewards $r _ { t } = R ( s _ { t } , a _ { t } )$ , and transitions $s _ { t + 1 } \sim T ( \cdot \mid s _ { t } , a _ { t } )$ . We denote by $h _ { t } ^ { c }$ and $h _ { t } ^ { u }$ the observation–action histories of the controlled and uncontrolled agents, respectively, up to timestep t. Note that throughout all future sections, we abbreviate $a _ { t } ^ { c }$ to $a _ { t }$ . The symbol $a _ { t }$ denotes the controlled joint action, and $a _ { t } ^ { u }$ denotes the uncontrolled joint action. Uncontrolled actions are sampled from $\pi ^ { u }$ and marginalized as part of the environment randomness.

We define action-value functions, value functions, and advantage functions at two levels of information access, differing in whether the centralized critic has access to the uncontrolled agents’ histories.

The joint history action-value function (or Q-function) is a centralized action-value function conditioned on the full joint history of all agents. For a fixed policy π, it is defined as:

$$
Q ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) = \mathbb { E } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } } \left[ \sum _ { k = 0 } ^ { T - t } \gamma ^ { k } r _ { t + k } \right]\tag{1}
$$

where the expectation is with respect to the randomness of the trajectory $\tau$ (state transitions, observations, and any stochasticity in π) from timestep t onward, conditioned on the joint history $( h _ { t } ^ { c } , h _ { t } ^ { u } )$ and action $a _ { t } . \ Q ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } )$ represents the expected discounted return of the team when the joint history $( h _ { t } ^ { c } , h _ { t } ^ { u } )$ has been observed and the controlled agents take action $a _ { t }$

The controlled history action-value function is defined as:

$$
Q ^ { \pi } ( h _ { t } ^ { c } , a _ { t } ) = \mathbb { E } _ { \tau \sim \pi , h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } \left[ \sum _ { k = 0 } ^ { T - t } \gamma ^ { k } r _ { t + k } \right]\tag{2}
$$

where the expectation additionally averages over the uncontrolled history $h _ { t } ^ { u }$ given $h _ { t } ^ { c } .$ The two levels are related by:

$$
Q ^ { \pi } ( h _ { t } ^ { c } , a _ { t } ) = \mathbb { E } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } } [ Q ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) ]\tag{3}
$$

The joint history and controlled history value functions are defined as:

$$
V ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) = \mathbb { E } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } } \left[ \sum _ { k = 0 } ^ { T - t } \gamma ^ { k } r _ { t + k } \right] , \qquad V ^ { \pi } ( h _ { t } ^ { c } ) = \mathbb { E } _ { \tau \sim \pi , h _ { t } ^ { u } | h _ { t } ^ { c } } \left[ \sum _ { k = 0 } ^ { T - t } \gamma ^ { k } r _ { t + k } \right]\tag{4}
$$

Equivalently, these are recovered from the action-value functions by averaging over actions:

$$
V ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) = \mathbb { E } _ { a _ { t } \sim \pi ( \cdot | h _ { t } ^ { c } ) } \bigl [ Q ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) \bigr ] , \qquad V ^ { \pi } ( h _ { t } ^ { c } ) = \mathbb { E } _ { a _ { t } \sim \pi ( \cdot | h _ { t } ^ { c } ) } \bigl [ Q ^ { \pi } ( h _ { t } ^ { c } , a _ { t } ) \bigr ]\tag{5}
$$

And they are related by:

$$
V ^ { \pi } ( h _ { t } ^ { c } ) = \mathbb { E } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } } [ V ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) ]\tag{6}
$$

Finally, we define the joint and controlled history advantage functions:

$$
A ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) = Q ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) - V ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) , \qquad A ^ { \pi } ( h _ { t } ^ { c } , a _ { t } ) = Q ^ { \pi } ( h _ { t } ^ { c } , a _ { t } ) - V ^ { \pi } ( h _ { t } ^ { c } )\tag{7}
$$

Note again that:

$$
A ^ { \pi } ( h _ { t } ^ { c } , a _ { t } ) = \mathbb { E } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } } [ A ^ { \pi } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) ]\tag{8}
$$

## 3.1 Proximal Policy Optimization

Controlled policies are trained with the Proximal Policy Optimization (PPO) algorithm [Schulman et al., 2017]. Recall that since controlled policies are conditioned only controlled histories, the PPO actor loss is defined as:

$$
\mathcal { L } ^ { P P O } ( \theta ) = - \mathbb { E } _ { h ^ { c } , a \sim \pi _ { \theta _ { \mathrm { o l d } } } } \Big [ \operatorname* { m i n } \Big ( \rho ( \theta ) A ( h ^ { c } , a ) , \ \mathrm { c l i p } \left( \rho ( \theta ) , 1 - \epsilon , 1 + \epsilon \right) A ( h ^ { c } , a ) \Big ) \Big ]\tag{9}
$$

where $\begin{array} { r } { \rho ( \theta ) = \frac { \pi _ { \theta } ( a \vert h ^ { c } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a \vert h ^ { c } ) } } \end{array}$ is the importance sampling ratio between the current policy $\pi _ { \theta }$ and the behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ used to collect the trajectory. Note that the true loss we wish to estimate has no dependence on uncontrolled histories: they have been marginalized out. We care only about choosing the best action given a controlled history. This paper will compare various estimators for $A ( h ^ { c } , a )$ and demonstrate that estimators utilizing joint history critics (taking into account uncontrolled histories) can be more accurate than those using only controlled histories.

## 3.2 Generalized Advantage Estimation

The advantage $A ( h ^ { c } , a )$ in the PPO objective is not directly observed and must be estimated from rollouts. A commonly used technique (and the technique used for all experiments in this paper) is Generalized Advantage Estimation (GAE) [Schulman et al., 2016], which trades off bias and variance through an exponentially-weighted average of k-step TD residuals. Given a learned value function $V _ { \phi }$ and a trajectory τ, the one-step TD residual at timestep t is

$$
\delta _ { t } = r _ { t } + \gamma V _ { \phi } ( h _ { t + 1 } ^ { c } ) - V _ { \phi } ( h _ { t } ^ { c } ) ,\tag{10}
$$

and the GAE advantage is defined as

$$
\hat { A } _ { t } ^ { \mathrm { G A E } ( \gamma , \lambda ) } = \sum _ { l = 0 } ^ { \infty } ( \gamma \lambda ) ^ { l } \delta _ { t + l } ,\tag{11}
$$

where $\gamma \in [ 0 , 1 ]$ is the discount factor and $\lambda \in [ 0 , 1 ]$ controls the bias-variance tradeoff. Setting $\lambda = 0$ recovers the one-step TD estimator $\hat { A } _ { t } = \delta _ { t }$ , which has low variance but is biased by errors in $V _ { \phi }$ . Setting $\lambda = 1$ recovers the Monte Carlo advantage $\begin{array} { r } { \hat { A } _ { t } = \sum _ { l = 0 } ^ { \infty } \gamma ^ { l } r _ { t + l } - V _ { \phi } ( h _ { t } ^ { c } ) } \end{array}$ , which is unbiased (with respect to the true value function) but suffers from high variance. Intermediate values of λ interpolate between these regimes.

Note also that we can recover an estimate of the action-value function from the GAE advantage by adding back the value baseline:

$$
\begin{array} { r } { \hat { Q } _ { t } ^ { \mathrm { G A E } ( \gamma , \lambda ) } = \hat { A } _ { t } ^ { \mathrm { G A E } ( \gamma , \lambda ) } + V _ { \phi } ( h _ { t } ^ { c } ) . } \end{array}\tag{12}
$$

In the standard single-agent setting, the value function $V _ { \phi }$ is conditioned on the same information as the policy. In our multi-agent setting, however, we have a choice: $V _ { \phi }$ may condition on the controlled history $\bar { h } ^ { c }$ alone, or on the joint history $( h ^ { c } , h ^ { u } )$ that additionally includes uncontrolled agents’ observations and actions. Section 4 compares these two choices and analyzes how the conditioning of the critic affects the variance of the resulting advantage estimator.

Training and Evaluation in Practice Similar to the training and evaluation procedure specified by Wang et al. [2024], we assume access to a training and test set of teammates, $\bar { X } _ { \mathrm { t r a i n } }$ and $\dot { X } _ { \mathrm { t e s t } }$ , which are drawn from the same distribution over teammate types. NAHT algorithms learn from interaction with teammates within the training set $X _ { \mathrm { t r a i n } }$ , while evaluations are conducted with teammates in the test set, $X _ { \mathrm { t e s t } }$

## 4 Theoretical Analysis

Existing algorithms for NAHT are policy-gradient based approaches, implemented via $\mathrm { P P O } ,$ so our theoretical analysis examines the bias and variance of advantage estimators. We will show that both estimators are unbiased for the controlled advantage function, and then show that the joint estimator is often lower variance than the controlled estimator.

The following assumptions and definitions are used throughout this section:

Assumption 4.1 (Unbiased value function estimators). Let $\hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ and $\hat { V } ( h _ { t } ^ { c } )$ be any unbiased estimators of the joint and controlled value functions, respectively, i.e., any random variables satisfying:

$$
\mathbb { E } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] = V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) , \qquad \mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } } [ \hat { V } ( h _ { t } ^ { c } ) ] = V ( h _ { t } ^ { c } )\tag{13}
$$

This is an idealization standard in the theoretical MARL literature [Lyu et al., 2023]; in practice it holds approximately for sufficiently expressive critics trained to convergence.

Assumption 4.2 (Unbiased action-value function estimator). Let $\hat { Q } ( h _ { t } ^ { c } , a _ { t } )$ be any unbiased estimator of the controlled action-value function $Q ( h _ { t } ^ { c } , a _ { t } )$ i.e., any random variable satisfying:

$$
\mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } ( h _ { t } ^ { c } , a _ { t } ) \big ] = Q ( h _ { t } ^ { c } , a _ { t } )\tag{14}
$$

This is satisfied with a monte-carlo returns estimator, or a GAE estimator bootstrapping with a value function satisfying Assumption 4.1.

Definition 4.3 (Controlled and joint advantage estimators). Let $\hat { Q } _ { t } ^ { ( 1 ) }$ and $\hat { Q } _ { t } ^ { ( 2 ) }$ be any two (possibly distinct) estimators satisfying Assumption 4.2. Note that in practice, $\hat { Q } _ { t } ^ { ( 1 ) }$ will likely be constructed with access to a controlled critic, and $\hat { Q } _ { t } ^ { ( 2 ) }$ will likely be constructed with access to a joint critic.

$$
\hat { A } _ { c o n t r o l l e d } ( a _ { t } , h _ { t } ^ { c } ) = \hat { Q } _ { t } ^ { ( 1 ) } - \hat { V } ( h _ { t } ^ { c } ) , \qquad \hat { A } _ { j o i n t } ( a _ { t } , h _ { t } ^ { c } , h _ { t } ^ { u } ) = \hat { Q } _ { t } ^ { ( 2 ) } - \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )\tag{15}
$$

## 4.1 Bias of Controlled and Joint Estimators

We first establish conditions under which both controlled and joint advantage estimators are unbiased for the controlled advantage that we wish to estimate as a term of the PPO loss.

Theorem 4.4 (Unbiasedness of Advantage Estimators).

$$
\mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { c o n t r o l l e d } \big ] = \mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { j o i n t } \big ] = A ( h _ { t } ^ { c } , a _ { t } )\tag{16}
$$

The proof is provided in Appendix B.1. Since both joint and controlled estimators are unbiased, they appear to each be reasonable choices to use in practice.

## 4.2 Variance of Controlled and Joint Estimators

When we examine the variances of joint and controlled estimators, we find that the joint estimator is often preferable.

Definition 4.5. Let $q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) : = \mathbb { E } [ \hat { Q } _ { t } ^ { ( 2 ) } \mid h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ]$ denote the conditional expectation of the joint-history action-value estimator given the full joint history and action.

Assumption 4.6 (Deterministic value function estimators). The value function estimators $\hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ and $\hat { V } ( h _ { t } ^ { c } )$ are deterministic functions of their inputs. That is, conditioned on $\left( h _ { t } ^ { c } , h _ { t } ^ { u } \right)$ (resp. h<sup>c</sup><sub>t</sub> ), they take a single fixed value with no remaining randomness:

$$
\begin{array} { r } { \operatorname { V a r } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] = 0 , \qquad \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } } \big [ \hat { V } ( h _ { t } ^ { c } ) \big ] = 0 } \end{array}\tag{17}
$$

This holds for any critics which are deterministic neural networks, so their outputs vary only across their inputs $( h _ { t } ^ { c } , \dot { h } _ { t } ^ { u }$ or $h _ { t } ^ { c } )$ , not across the trajectory rollout randomness conditional on those inputs.

We decompose the variance difference between a joint and controlled estimator for $A ( h _ { t } ^ { c } , a _ { t } )$ . Note that the joint estimator has lower variance (and we achieve a “variance reduction”) when the difference is positive.

Theorem 4.7 (Advantage Variance Decomposition). For anyfixed $\left( h _ { t } ^ { c } , a _ { t } \right)$

$$
\begin{array} { r l } & { \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { c o n t r o l l e d } \big ] - \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { j o i n t } \big ] } \\ & { \quad = \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } \big ] - \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] } \\ & { \qquad + \operatorname { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) \big ] - \operatorname { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) - \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{18}
$$

The variance difference decomposes into two contributions:

$\mathrm { V a r } [ \hat { Q } _ { t } ^ { ( 1 ) } ] - \mathrm { V a r } [ \hat { Q } _ { t } ^ { ( 2 ) } ]$ captures the difference in raw variance between the two action-value estimators. For example, if both estimators are the monte-carlo estimator, then this term is 0. If both estimators are deep-Q networks, or GAE estimators, then it is more difficult to make rigorous claims about the sign of this term. Note that this term is “out of control" for the critic, which just exists to subtract a baseline from the action-value estimator. As a result, and because we have estimators for which the term is surely 0, we focus on the second term for our analysis.

• The last two terms, Var $\cdot [ q ^ { ( 2 ) } ] - \mathrm { V a r } [ q ^ { ( 2 ) } - \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) ]$ , measure how much of the $h _ { t } ^ { u } .$ variation in $q ^ { ( 2 ) }$ is actually cancelled by subtracting the joint baseline. These terms capture the variance reduction benefit of having a baseline that co-varies with the return across uncontrolled histories. We define this to be the baseline-cancellation term.

The proof is provided in Appendix B.2.

Assumption 4.8. The special case where $\hat { Q } _ { t } ^ { ( 2 ) }$ additionally is an unbiased estimator for the joint action-value function:

$$
\mathbb { E } [ \hat { Q } _ { t } ^ { ( 2 ) } \mid h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ] = Q ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } )\tag{19}
$$

This occurs when $\hat { Q } _ { t } ^ { ( 2 ) }$ is GAE bootstrapped through the joint history critic $\hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )$

Corollary 4.9 (Variance Decomposition Simplification). Under Assumption 4.8 and the exact-critic idealization $\hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) = V ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ , Theorem 4.7 reduces to:

$$
\begin{array} { r l } & { \mathrm { V a r } [ \hat { A } _ { c o n t r o l l e d } ] - \mathrm { V a r } [ \hat { A } _ { j o i n t } ] } \\ & { \quad = \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } [ \hat { Q } _ { t } ^ { ( 1 ) } ] - \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } [ \hat { Q } _ { t } ^ { ( 2 ) } ] } \\ & { \quad \quad + \mathrm { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } [ Q ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) ] - \mathrm { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } [ A ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) ] } \end{array}\tag{20}
$$

## 4.3 When Is the Baseline-Cancellation Term Large?

We now consider conditions under which the baseline-cancellation term, $\mathrm { V a r } [ q ^ { ( 2 ) } ] - \mathrm { V a r } [ q ^ { ( 2 ) } - $ $V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) ]$ ], is positive, creating a variance reduction when the same action-value estimator is used for each type of critic. For clarity, we restrict attention to the case where $\hat { Q } _ { t } ^ { ( 2 ) }$ satisfies Assumption 4.8. In this case, the baseline-cancellation term reduces to:

$$
\begin{array} { r } { \operatorname { V a r } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } \left[ Q ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) \right] - \operatorname { V a r } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } \left[ A ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) \right] } \end{array}\tag{21}
$$

Using the identity $\operatorname { V a r } [ Q ] - \operatorname { V a r } [ Q - V ] = 2 \operatorname { C o v } [ Q , V ] - \operatorname { V a r } [ V ]$ , the baseline-cancellation term is positive if and only if:

$$
\mathrm { C o v } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } \left[ Q ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) , V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \right] > \frac { 1 } { 2 } \mathrm { V a r } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } \left[ V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \right]\tag{22}
$$

Writing $\begin{array} { r l r } { \sigma _ { Q } } & { = } & { \sqrt { \mathrm { V a r } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } [ Q ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) ] } , \sigma _ { V } = \sqrt { \mathrm { V a r } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } [ V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) ] } } \end{array}$ , and $\rho \quad =$ $\operatorname { C o r r } _ { h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } } [ Q ( h _ { t } ^ { c } , \dot { h } _ { t } ^ { u } , a _ { t } ) , V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) ]$ , this becomes:

$$
\rho > { \frac { \sigma _ { V } } { 2 \sigma _ { Q } } }\tag{23}
$$

Intuitively, the joint baseline $V ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ reduces variance by canceling variation in $Q ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } )$ that is driven by uncontrolled histories rather than by the action itself. For this cancellation to be effective, two conditions must hold:

1. V and $Q$ must move together across uncontrolled histories. If uncontrolled histories that are generally good for the current policy (high V ) are also good for the specific action taken (high $Q )$ then subtracting $V$ strips out shared variation. High correlation $\rho$ captures this: if $Q$ and $V$ move in lockstep across $h _ { t } ^ { u }$ , then $Q - V$ is nearly constant in $h _ { t } ^ { u }$ and the cancellation is near-perfect. Note that if $a _ { t }$ is chosen with probability close to 1 (the policy is nearly deterministic), then $V \approx Q ( a _ { t } )$ and the correlation is certain. In contrast, if $a _ { t }$ is chosen with low probability by the current policy, then a correlation (and variance reduction) is unlikely.

2. V must not swing more than $Q .$ . Even with perfect correlation, if the baseline varies more than the quantity it is trying to cancel $( \sigma _ { V } > 2 \sigma _ { Q } )$ , subtracting it introduces more variance than it removes. Putting these together, the baseline-cancellation term is most positive when $Q$ and $V$ are highly correlated and have comparable magnitudes of variation across uncontrolled histories.

## 4.4 From Per-Pair Advantage Variance to Visitation Weighted Advantage Variance

Our variance analysis thus far fixes a specific state-action pair $\left( h _ { t } ^ { c } , a _ { t } \right)$ and compares the conditional variance of the two advantage estimators at that pair. For some $\left( h _ { t } ^ { c } , a _ { t } \right)$ , the joint baseline will reduce variance; for others, the condition $\rho > \sigma _ { V } / 2 \sigma _ { Q }$ will fail and the joint baseline will increase variance. PPO, however, is intuitively influenced by aggregate gradient variance, which is a visitation-weighted average:

$$
\mathbb { E } _ { ( h _ { t } ^ { c } , a _ { t } ) \sim \pi _ { \theta _ { \mathrm { o l d } } } } \bigg [ \operatorname { V a r } \big [ \hat { A } ( h _ { t } ^ { c } , a _ { t } ) \big ] \bigg ]\tag{24}
$$

The relevant question is therefore whether this expected variance is lower under $\hat { A } _ { j o i n t }$ than under $\hat { A } _ { c o n t r o l l e d } .$

Intuitive argument. We conjecture that the aggregate gradient variance is typically lower under the joint critic, i.e.,

$$
\mathbb { E } _ { ( h _ { t } ^ { c } , a _ { t } ) \sim \pi _ { \theta _ { \mathrm { o l d } } } } \Big [ \mathrm { V a r } \big [ \hat { A } _ { c o n t r o l l e d } \big ] - \mathrm { V a r } \big [ \hat { A } _ { j o i n t } \big ] \Big ] > 0\tag{25}
$$

The intuition is that the state-action pairs where the joint critic most effectively reduces variance are precisely the pairs the current policy visits most often (Section 4.3). We refer to this intuition to explain the empirical greater performance of joint history critics for multiple tasks showcased in Section 6.

## 5 Illustrative Example: A Coordination Game with Variance Reduction

To build intuition for when the joint baseline meaningfully reduces variance, we construct a simple coordination game where the joint history critic produces a substantial — though not complete — variance reduction over the controlled critic.

Setup. We consider a game with:

• Agents: 2 controlled agents, 4 uncontrolled agents.

• Action space: $\{ 0 , 1 \} ^ { 6 }$

• Uncontrolled policy: At the start of each episode, the uncontrolled team samples a shared policy uniformly from {Mostly 0, Mostly 1}. Under Mostly 0, each uncontrolled agent independently outputs 0 with probability 0.9 and 1 with probability 0.1. Under Mostly 1, each uncontrolled agent independently outputs 0 with probability 0.25 and 1 with probability 0.75.

• Observation space: Each agent observes only its own past actions. No additional observations are available, and the optimal policy does not condition on any observation.

• Reward: $r ( \mathbf { a } ) = 1$ if all six agents output the same value (either all zeros or all ones), and $r ( \mathbf { a } ) = 0$ otherwise.

• Horizon and discount: Episode length T, discount $\gamma = 1$

We analyze a controlled policy that plays {0, 0} with probability 0.8 and {1, 1} with probability 0.2 at every history, and we focus on the advantage of the action $a _ { t } = \{ 0 , 0 \}$ . We take $\hat { Q } _ { t }$ to be a single-step TD estimator that bootstraps through the appropriate critic:

$$
\hat { Q } _ { t } ^ { ( 1 ) } = r _ { t } + V ( h _ { t + 1 } ^ { c } ) , \qquad \hat { Q } _ { t } ^ { ( 2 ) } = r _ { t } + V ( h _ { t + 1 } ^ { c } , h _ { t + 1 } ^ { u } )\tag{26}
$$

Value functions. Let $p _ { 0 } = 0 . 8$ and $p _ { 1 } = 0 . 2$ denote the controlled policy’s probabilities of playing $\{ 0 , 0 \}$ and {1, 1}, respectively. Under each uncontrolled policy, we compute the expected per-step reward:

$$
\mathbb { E } [ r \mid \mathbf { M o s t l y ~ } 0 ] = p _ { 0 } \cdot 0 . 9 ^ { 4 } + p _ { 1 } \cdot 0 . 1 ^ { 4 } = 0 . 8 ( 0 . 6 5 6 1 ) + 0 . 2 ( 0 . 0 0 0 1 ) = 0 . 5 2 4 9\tag{27}
$$

$$
\mathbb { E } [ r \mid \mathrm { M o s t l y ~ 1 } ] = p _ { 0 } \cdot 0 . 2 5 ^ { 4 } + p _ { 1 } \cdot 0 . 7 5 ^ { 4 } = 0 . 8 ( 0 . 0 0 3 9 1 ) + 0 . 2 ( 0 . 3 1 6 4 ) = 0 . 0 6 6 5\tag{28}
$$

Marginalizing over the uncontrolled policy, $\mathbb { E } [ r ] = 0 . 5 ( 0 . 5 2 4 9 ) + 0 . 5 ( 0 . 0 6 6 5 ) = 0 . 2 9 5 7$ . The value functions at timestep t (with $K = \bar { T } - t$ steps remaining) are these per-step expectations scaled by K:

$$
V ( h _ { t } ^ { c } ) = K \cdot 0 . 2 9 5 7 , \qquad V ( h _ { t } ^ { c } , h _ { t } ^ { M _ { 0 } } ) = K \cdot 0 . 5 2 4 9 , \qquad V ( h _ { t } ^ { c } , h _ { t } ^ { M _ { 1 } } ) = K \cdot 0 . 0 6 6 5\tag{29}
$$

where $h _ { t } ^ { M _ { 0 } }$ and $h _ { t } ^ { M _ { 1 } }$ denote uncontrolled histories consistent with Mostly 0 and Mostly 1, respectively. We also compute the controlled action-value for $a _ { t } = \{ 0 , 0 \}$ . At the current step, the expected reward given $a _ { t } = \{ 0 , 0 \}$ depends on the uncontrolled policy:

$$
\mathbb { E } [ r \mid a _ { t } = \{ 0 , 0 \} , \mathbf { M o s t l y } \ 0 ] = 0 . 9 ^ { 4 } = 0 . 6 5 6 1 ,
$$

$$
\operatorname { \mathbb { E } } [ r \mid a _ { t } = \{ 0 , 0 \} , \operatorname { M o s t l y } 1 ] = 0 . 2 5 ^ { 4 } \approx 0 . 0 0 3 9 1\tag{30}
$$

Marginalizing: $\mathbb { E } [ r \mid a _ { t } = \{ 0 , 0 \} ] = 0 . 5 ( 0 . 6 5 6 1 ) + 0 . 5 ( 0 . 0 0 3 9 1 ) \approx 0 . 3 3 0$ . Adding the expected value of the remaining $K - 1$ steps:

$$
Q ( h _ { t } ^ { c } , \{ 0 , 0 \} ) = 0 . 3 3 0 + ( K - 1 ) \cdot 0 . 2 9 5 7\tag{31}
$$

The true controlled advantage is therefore:

$$
A ( h _ { t } ^ { c } , \{ 0 , 0 \} ) = Q ( h _ { t } ^ { c } , \{ 0 , 0 \} ) - V ( h _ { t } ^ { c } ) = 0 . 3 3 0 - 0 . 2 9 5 7 = 0 . 0 3 4 3\tag{32}
$$

which is positive, reflecting that {0, 0} is a better-than-average action under the current policy.

Advantage estimators. Substituting the value functions:

$$
\hat { A } _ { c o n t r o l l e d } = r _ { t } + V ( h _ { t + 1 } ^ { c } ) - V ( h _ { t } ^ { c } ) = r _ { t } + ( K - 1 ) ( 0 . 2 9 5 7 ) - K ( 0 . 2 9 5 7 ) = r _ { t } - 0 . 2 9 5 7\tag{33}
$$

For the joint history estimator, the form depends on the uncontrolled policy in effect:

$$
\hat { A } _ { j o i n t } \ | \ \mathrm { M o s t l y } \ 0 = r _ { t } + V \big ( h _ { t + 1 } ^ { c } , h _ { t + 1 } ^ { M _ { 0 } } \big ) - V \big ( h _ { t } ^ { c } , h _ { t } ^ { M _ { 0 } } \big ) = r _ { t } - 0 . 5 2 4 9\tag{34}
$$

$$
\hat { A } _ { j o i n t } \ | \ \mathrm { M o s t l y } \ 1 = r _ { t } + V \big ( h _ { t + 1 } ^ { c } , h _ { t + 1 } ^ { M _ { 1 } } \big ) - V \big ( h _ { t } ^ { c } , h _ { t } ^ { M _ { 1 } } \big ) = r _ { t } - 0 . 0 6 6 5\tag{35}
$$

Both estimators are unbiased for $A ( h _ { t } ^ { c } , \{ 0 , 0 \} ) = 0 . 0 3 4 3 \colon$ for instance, the joint estimator gives $\mathbb { E } [ \hat { A } _ { j o i n t } ] = 0 . 5 ( 0 . 6 5 6 1 - 0 . 5 2 4 9 ) + 0 . 5 ( 0 . 0 0 3 9 1 - 0 . 0 6 6 5 ) = 0 . 5 ( 0 . 1 3 1 2 ) + 0 . 5 ( - 0 . 0 6 2 6 ) = 0 . 2 5 ( 0 . 0 3 9 1 - 0 . 0 2 6 ) .$ 0.0343, as required.

Variance computation. All variances are taken with respect to the joint distribution over the uncontrolled policy $\theta ^ { u } \in \{ \mathrm { M o s t l y ~ 0 , M o s t l y ~ 1 } \}$ and the realized reward $r _ { t } \in \{ 0 , 1 \}$ } given $a _ { t } = \{ 0 , 0 \}$ There are four possible outcomes. We first enumerate their probabilities:

$$
\operatorname* { P r } ( \mathrm { M o s t l y } 0 , r _ { t } = 1 ) = 0 . 5 \cdot 0 . 6 5 6 1 = 0 . 3 2 8 1\tag{36}
$$

$$
\operatorname* { P r } ( \mathrm { M o s t l y } 0 , r _ { t } = 0 ) = 0 . 5 \cdot 0 . 3 4 3 9 = 0 . 1 7 2 0\tag{37}
$$

$$
\operatorname* { P r } ( \mathrm { M o s t l y ~ } 1 , \ : r _ { t } = 1 ) = 0 . 5 \cdot 0 . 0 0 3 9 1 = 0 . 0 0 1 9 5\tag{38}
$$

$$
\operatorname* { P r } ( \mathrm { M o s t l y } 1 , r _ { t } = 0 ) = 0 . 5 \cdot 0 . 9 9 6 0 9 = 0 . 4 9 8 0\tag{39}
$$

Controlled advantage estimator. Recall $\hat { A } _ { c o n t r o l l e d } = r _ { t } - 0 . 2 9 5 7$ , which takes value 0.7043 when $r _ { t } = 1$ and −0.2957 when $r _ { t } = 0$ . Its mean is $A ( h _ { t } ^ { c } , \{ 0 , 0 \} ) = 0 . 0 3 4 3$ . Computing the variance as expected squared deviation from the mean:

$$
\begin{array} { r l } & { \mathrm { V a r } \big [ \hat { A } _ { c o n t r o l l e d } \big ] = \mathrm { P r } ( r _ { t } = 1 ) \cdot ( 0 . 7 0 4 3 - 0 . 0 3 4 3 ) ^ { 2 } + \mathrm { P r } ( r _ { t } = 0 ) \cdot ( - 0 . 2 9 5 7 - 0 . 0 3 4 3 ) ^ { 2 } } \\ & { \quad \quad = 0 . 3 3 0 \cdot ( 0 . 6 7 0 0 ) ^ { 2 } + 0 . 6 7 0 \cdot ( - 0 . 3 3 0 ) ^ { 2 } } \\ & { \quad \quad = 0 . 3 3 0 \cdot 0 . 4 4 8 9 + 0 . 6 7 0 \cdot 0 . 1 0 8 9 } \\ & { \quad \quad \approx 0 . 2 2 1 } \end{array}\tag{40}
$$

Joint history advantage estimator. Recall that $\hat { A } _ { j o i n t } = r _ { t } - 0 . 5 2 4 9$ under Mostly 0 and $\hat { A } _ { j o i n t } =$ $r _ { t } - 0 . 0 6 6 5$ under Mostly 1. The four possible values of $\hat { A } _ { j o i n t }$ are therefore:

$$
\hat { A } _ { j o i n t } ( \mathbf { M o s t l y } \ 0 , r _ { t } = 1 ) = 1 - 0 . 5 2 4 9 = 0 . 4 7 5 1\tag{41}
$$

$$
\hat { A } _ { j o i n t } ( \mathbf { M o s t l y } \ 0 , r _ { t } = 0 ) = 0 - 0 . 5 2 4 9 = - 0 . 5 2 4 9\tag{42}
$$

$$
\hat { A } _ { j o i n t } ( \mathbf { M o s t l y } \ 1 , r _ { t } = 1 ) = 1 - 0 . 0 6 6 5 = 0 . 9 3 3 5\tag{43}
$$

$$
\hat { A } _ { j o i n t } ( \mathbf { M o s t l y } \ 1 , r _ { t } = 0 ) = 0 - 0 . 0 6 6 5 = - 0 . 0 6 6 5\tag{44}
$$

Its mean is 0.0343. Computing the variance as the expected squared deviation from the mean:

$$
\begin{array} { r l } & { \mathrm { V a r } \big [ \hat { A } _ { j o i n t } \big ] = 0 . 3 2 8 1 \cdot ( 0 . 4 7 5 1 - 0 . 0 3 4 3 ) ^ { 2 } + 0 . 1 7 2 0 \cdot \big ( - 0 . 5 2 4 9 - 0 . 0 3 4 3 \big ) ^ { 2 } } \\ & { \qquad + 0 . 0 0 1 9 5 \cdot ( 0 . 9 3 3 5 - 0 . 0 3 4 3 ) ^ { 2 } + 0 . 4 9 8 0 \cdot \big ( - 0 . 0 6 6 5 - 0 . 0 3 4 3 \big ) ^ { 2 } } \\ & { \qquad = 0 . 3 2 8 1 \cdot \big ( 0 . 4 4 0 8 \big ) ^ { 2 } + 0 . 1 7 2 0 \cdot \big ( - 0 . 5 5 9 2 \big ) ^ { 2 } } \\ & { \qquad + 0 . 0 0 1 9 5 \cdot ( 0 . 8 9 9 2 ) ^ { 2 } + 0 . 4 9 8 0 \cdot \big ( - 0 . 1 0 0 8 \big ) ^ { 2 } } \\ & { \qquad = 0 . 3 2 8 1 \cdot 0 . 1 9 4 3 + 0 . 1 7 2 0 \cdot 0 . 3 1 2 7 + 0 . 0 0 1 9 5 \cdot 0 . 8 0 8 6 + 0 . 4 9 8 0 \cdot 0 . 0 1 0 1 6 } \\ & { \qquad \approx 0 . 1 2 4 } \end{array}\tag{45}
$$

Result. The joint history baseline nearly halves the variance of the advantage estimator:

$$
\mathrm { V a r } \left[ \hat { A } _ { c o n t r o l l e d } \right] \approx 0 . 2 2 1 , \qquad \mathrm { V a r } \left[ \hat { A } _ { j o i n t } \right] \approx 0 . 1 2 4\tag{46}
$$

This matches our first insight about variance reduction. For this scenario, the action $\{ 0 , 0 \}$ receives a high reward when paired with the same types of uncontrolled agents that the policy performs well for: Mostly 0. Action {0, 0} is also currently chosen with high probability, as we note can be beneficial for variance reduction in insight 1.

Negative Example Note that if we change the conditions of this example, we can end up with a scenario where the joint estimator has greater variance than the controlled estimator. Such an example is provided in Appendix A to help readers gain intuition about the differences between estimators.

## 6 Empirical Results

This section addresses the primary research question of if and when performance differs between joint and controlled critics in the NAHT problem. We compare performance across a variety of games and previously proposed algorithms for NAHT. Across every (game, algorithm) pair, a joint critic is as effective as a controlled critic, and for many scenarios, a joint critic exhibits significantly better performance.

## 6.1 Algorithms

As a first solution method for NAHT, Wang et al. [2024] proposed the Policy Optimization with Agent Modeling (POAM) algorithm. POAM deals with a potentially varying number of controlled agents by employing an independent policy gradient approach, where each agent learns an independent policy while performing teammate modeling of all other agents in the team. Each agent’s policy conditions on the learned teammate representations, thus enabling the policy to approximate the optimal behavior for collaborating with each type of uncontrolled team. Since we study centralized critics in this paper, we adapt POAM to include centralized observations in the same fashion as Wang et al. [2025a], creating the method Cent Obs POAM. We simply refer to the method as "POAM" throughout the paper.

Wang et al. [2025a] proposed MAT-NAHT, a transformer-based method adapted from the Multi-Agent Transformer [Wen et al., 2022], in which an encoder-decoder transformer jointly serves as the centralized policy and value network for the controlled subteam. The architecture handles the variable number of controlled agents through attention masking, and the centralized critic is conditioned on the history of observations and actions of all controlled agents. We abbreviate the method to MAT for this paper.

## 6.2 Connection to Theory

All experiments were conducted in an environment that satisfies all assumptions from Section 4. Specifically, controlled critic training runs use GAE estimation and boostrap through the controlled critic while joint critic training runs use GAE estimation bootstrapped through a joint critic. This satisfies Assumption 4.2, Assumption 4.6, and also the stronger condition Assumption 4.8. As noted previously Assumption 4.1 is satisfied when the value networks have converged, and is a standard assumption in previous literature.

![](images/67453e5a34dbad30d3033ce81fe8b76aaa5dc5a02ceb19110cac139120a177ad.jpg)  
(a) Entropy of MAT policies.

![](images/91ad11f3da5b1a2c9f33df687f4a42e6b0af8060b5ba62546827c2fe298a3468.jpg)  
(b) Entropy of POAM policies.

![](images/8b24a617db9854bc5d57746f11c8c794d941d8d466eb467f79ac6b4f43187ed1.jpg)  
(c) Return curves of MAT policies.

![](images/b7197786073c8e3e4917b0e5f1c2a9711bd2cc9e73ed1cf7630ba81a4ff5fcdb.jpg)  
(d) Return curves of POAM policies.  
Figure 1: Joint history critics have significantly higher sampling efficiency than controlled history critics for coordination game.

## 6.3 Coordination Game

We first present results for the coordination game described previously. Recall that the optimal policy is to always output {0, 0} for every history, so the optimal policy has 0 entropy. We can see that for the entropy curves of both MAT and POAM, the joint history critic achieves a near-optimal, 0 entropy policy significantly faster than the controlled history critic. The same story is reflected in the return curves, which demonstrate the greater sampling efficiency of a joint history critic for this task.

## 6.4 Stag Hunt

In Stag Hunt [Böhmer et al., 2020], N predator agents must capture stags and hares in a grid world. Agents and prey can move along the four cardinal directions or stay still, and agents additionally have a catch action. Stags yield a high reward but require coordinated capture: at least two adjacent agents must simultaneously execute the catch action to capture a stag, while a single adjacent agent suffices to capture a hare. If an agent attempts to capture a stag without sufficient teammates adjacent to the same prey, the team is penalized by the miscapture punishment. Upon a successful capture, the participating agents are frozen for the remainder of the episode and removed from the grid, and the captured prey is removed. Agents observe a $( 2 k + 1 ) \times ( 2 k + 1 )$ patch centered on their position, while prey movement is governed by a fixed stochastic policy: prey move to a random unoccupied adjacent cell. An episode terminates when all prey are captured, all agents are frozen, or after T timesteps.

From the return curves we see that, like coordination game, joint critics offer improved sample efficiency for both MAT and POAM. From the bar charts, we can compare joint and controlled critic policies with individual uncontrolled agents . We see that joint critics perform better with every type of uncontrolled agent.

We use hypothesis tests across all (controlled, uncontrolled) seed pairs for each comparison to see if the difference in performance between joint and controlled critics is statistically significant. At $\alpha = 0 . 0 5$ , we find statistically significant differences for 3 of the 4 uncontrolled agent types. Note that for the 4th uncontrolled agent type, the mean returns are still greater for a joint critic, and we do not have data to suggest that controlled critics are better for this type of agent. In order to make a statistically significant claim for the superiority of one critic over another, more data would need to be collected.

![](images/3955cc554ba071134061a250f05c839954b5edfbaac046d1b78f9db25fce0dc3.jpg)  
(a) Return curves of MAT policies. OOD: MAT variants

![](images/d72ae53270baa3c0a9d57f983843ccc3545db7701da7f185d6b3f17aaba2c227.jpg)  
(b) Return curves of POAM policies. OOD: POAM variants

![](images/b097628f97fe73362df50e6cbeb95a91fd29cc7f4dd5939fc22bcf98f42cf559.jpg)  
(c) Returns with fixed uncontrolled agent policies (MAT).

![](images/64e5529ceeeb43796240229a9ebfdfdcedddc558889e44b511f1be164e36fa25.jpg)  
(d) Returns with fixed uncontrolled agent policies (POAM).

Figure 2: Comparison of stag hunt return curves and out of distribution performance across MAT and POAM variants.  
Table 1: Stag Hunt MAT Hypothesis Tests.
<table><tr><td>Target algorithm</td><td>Mean difference (C-J)</td><td>Paired p-value</td></tr><tr><td>catch-req1_ippo*</td><td>-0.501652</td><td>0.010758</td></tr><tr><td>catch-req2-risky_ippo***</td><td>-0.376259</td><td>0.000388</td></tr><tr><td>catch-req2_ippo*</td><td>-0.272097</td><td>0.017871</td></tr><tr><td>catch-req3_ippo</td><td>-0.195542</td><td>0.426157</td></tr></table>

## 6.5 SMAC 8v9

The SMAC 8v9 challenge [Samvelyan et al., 2019] is a hard-difficulty StarCraft II micromanagement scenario in which N = 8 allied Marines must defeat a slightly larger enemy team of 9 Marines controlled by the built-in game AI. At each timestep, every allied agent selects from a discrete action set consisting of move (in the four cardinal directions), stop, no-op, or attack-target actions, where the attack actions are restricted to enemy units within the agent’s fixed shooting range. Agents observe a partial view of the battlefield limited to a circular sight range, receiving features such as relative positions, health, shield, and unit type of allies and enemies within range; units outside this range are masked from observation. The team receives a shaped reward based on damage dealt to enemies, kills, and a large terminal bonus for winning the engagement, with success requiring coordinated focus-fire and positioning to overcome the numerical disadvantage. An episode terminates when all units on one side are eliminated or after T timesteps, and the scenario is considered solved when the allied team achieves a high win rate against the scripted opponent.

![](images/4e260224856b6a276c278e4e71c168f0a98858af9832d8127580e13f794505cc.jpg)  
(a) Return curves of MAT policies.

![](images/d5c93ee42ab78372e44ff485f36141dc2378744a68638dccd98757b5ed386d15.jpg)  
(b) Return curves of POAM policies.

![](images/21c0e6c861280f6ab57256945fb138e30e3135bad9c9d5fba2b4e3b888ac5a22.jpg)  
(c) Entropy curves of MAT policies

![](images/912a440a4ca3549bc5dffd29f727dc2c76bc3ab85b9fe2b7c8621b0279bb72de.jpg)  
(d) Entropy curves of POAM policies.  
Figure 3: Comparison of SMAC return curves and out of entropy curves across MAT and POAM variants.

Table 2: 8v9 POAM Hypothesis Tests.
<table><tr><td>Target algorithm</td><td>Mean difference (C-J)</td><td>Paired p-value</td></tr><tr><td> $\dot { \operatorname { i p p o } } ^ { * * * * }$ </td><td>-0.765862</td><td>0.000775</td></tr><tr><td> $\mathrm { i q l } ^ { * * }$ </td><td>-0.589716</td><td>0.005665</td></tr><tr><td> $\operatorname* { m a p p o } ^ { * * }$ </td><td>-0.539812</td><td>0.008801</td></tr><tr><td> $\operatorname { q m i x } ^ { * * }$ </td><td>-1.003580</td><td>0.001541</td></tr><tr><td> $\mathrm { v d n } ^ { * * * }$ </td><td>-0.74291</td><td>0.000558</td></tr></table>

Results for the 8v9 SMAC challenge are similar to stag hunt. We see that joint critics perform better than or equal to controlled critics, and joint critics show improved sampling efficiency for the POAM policies. Hypothesis tests for out of distribution performance were not statistically significant for MAT policies, but showed a certain improvement for joint critic POAM policies.

## 7 Conclusion

We studied the design of centralized critics for N-agent ad hoc teamwork, focusing on whether conditioning the critic on the histories of uncontrolled teammates is beneficial. We compared two natural advantage estimators: a controlled history estimator that conditions only on the learner’s own observation-action histories, and a joint history estimator that additionally conditions on the uncontrolled teammates’ histories.

On the theoretical side, we proved that both estimators are unbiased for the controlled advantage, and we derived a variance decomposition that isolates the primary force governing whether the joint history estimator reduces variance: the benefit of subtracting a critic correlated to the action-value function as a baseline. We characterized the conditions under which the baseline-cancellation effect dominates—namely, when the joint Q and V functions are highly correlated across uncontrolled histories and have comparable magnitudes of variation—and illustrated both regimes through a simple coordination game in which the joint baseline can either reduce or increase variance depending on the policy under analysis. We made an argument for the joint history estimator having lower variance on average, providing some intuition for our experimental results.

Empirically, joint history critics match or outperform controlled history critics across both the coordination game and the Stag Hunt environment, for both the POAM and MAT base algorithms. In the coordination game, joint critics reach near-optimal policies markedly faster. In Stag Hunt, joint critics yield improved sample efficiency during training and higher returns against uncontrolled agents at evaluation, with statistically significant gains across most uncontrolled agent types tested.

Taken together, our theoretical analysis and experimental results provide both a justification and a practical recommendation: when training a centralized critic in the NAHT setting, conditioning on uncontrolled teammate histories is generally beneficial, and the variance decomposition we derive offers a principled lens for understanding when and why this benefit arises.

Limitations and future work. Our variance analysis assumes access to unbiased value functions; extending these results to learned critics, where approximation error interacts with the variance tradeoff, is a natural next step. Furthermore, our Visitation Weighted Advantage Variance argument is also currently a conjecture, supported by experimental evidence but not formally established; proving conditions under which the visitation-weighted variance is provably lower under the joint critic would strengthen the theoretical picture. We also restrict our analysis to the variance of the advantage function, but do not connect the ideas that a lower variance results in greater sampling efficiency. One way to conduct this analysis might be by looking at the variance of the policy gradient.

## References

Daniel S. Bernstein, Robert Givan, Neil Immerman, and Shlomo Zilberstein. The complexity of decentralized control of Markov decision processes. Mathematics of Operations Research, 27(4):819–840, November 2002. ISSN 0364-765X. doi: 10.1287/moor.27.4.819.297. URL https://pubsonline.informs.org/doi/10.1287/moor.27.4.819.297.

Wendelin Böhmer, Vitaly Kurin, and Shimon Whiteson. Deep coordination graphs. In Proceedings ofthe 37th International Conference on Machine Learning (ICML), 2020.

Sven Gronauer and Klaus Diepold. Multi-agent deep reinforcement learning: a survey. Artificial Intelligence Review, 55(2):895–943, 2022.

Xueguang Lyu, Andrea Baisero, Yuchen Xiao, Brett Daley, and Christopher Amato. On centralized critics in multi-agent reinforcement learning. Journal of Artificial Intelligence Research, 77: 295–354, 2023. doi: 10.48550/arXiv.2408.14597.

Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim G. J. Rudner, Chia-Man Hung, Philip H. S. Torr, Jakob Foerster, and Shimon Whiteson. The starcraft multi-agent challenge. arXiv preprint arXiv:1902.04043, 2019.

John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. In Proceedings of the International Conference on Learning Representations (ICLR), 2016. URL https://arxiv. org/abs/1506.02438.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxiv.org/ abs/1707.06347.

Peter Stone, Gal A. Kaminka, Sarit Kraus, and Jeffrey S. Rosenschein. Ad hoc autonomous agent teams: Collaboration without pre-coordination. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, pages 1504–1509, 2010.

Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone. N-agent ad hoc teamwork. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Caroline Wang, Di Yang Shi, Elad Liebman, Ishan Durugkar, Arrasy Rahman, and Peter Stone. Sequence modeling for n-agent ad hoc teamwork. In Multi-disciplinary Conference on Reinforcement Learning and Decision Making (RLDM), 2025a.

Jianhong Wang, Yang Li, Samuel Kaski, and Jonathan Lawry. Shapley machine: A game-theoretic framework for n-agent ad hoc teamwork, 2025b. URL https://arxiv.org/abs/2506. 11285.

Muning Wen, Jakub Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, and Yaodong Yang. Multi-agent reinforcement learning is a sequence modeling problem. In Advances in Neural Information Processing Systems, volume 35, 2022.

## A Counterexample: When the Joint Baseline Increases Variance

We now modify the example in Section 5 to illustrate a case where the joint history baseline increases variance over the controlled critic. We retain the setup from before, but flip the controlled policy: $\{ 0 , 0 \}$ is now played with probability $p _ { 0 } = 0 . 2$ and $\{ 1 , 1 \}$ with probability $p _ { 1 } = 0 . 8$ . We again analyze the advantage of the action $a _ { t } = \{ 0 , 0 \}$

Value functions. Recomputing the per-step expected rewards under each uncontrolled policy:

$$
\operatorname { \mathbb { E } } [ r \mid \operatorname { M o s t l y } 0 ] = 0 . 2 ( 0 . 6 5 6 1 ) + 0 . 8 ( 0 . 0 0 0 1 ) = 0 . 1 3 1 3\tag{47}
$$

$$
\operatorname { \mathbb { E } } [ r \mid \operatorname { M o s t l y } 1 ] = 0 . 2 ( 0 . 0 0 3 9 1 ) + 0 . 8 ( 0 . 3 1 6 4 ) = 0 . 2 5 4 0\tag{48}
$$

Marginalizing, $\mathbb { E } [ r ] = 0 . 5 ( 0 . 1 3 1 3 ) + 0 . 5 ( 0 . 2 5 4 0 ) = 0 . 1 9 2 6$ . The controlled advantage of $a _ { t } =$ $\{ 0 , 0 \}$ is unchanged from before in expected one-step reward, $\mathbb { E } [ r \mid a _ { t } = \{ 0 , 0 \} ] = 0 . 3 3 \bar { 0 } , \mathrm { s o }$

$$
A ( h _ { t } ^ { c } , \{ 0 , 0 \} ) = 0 . 3 3 0 - 0 . 1 9 2 6 = 0 . 1 3 7 4\tag{49}
$$

Advantage estimators. As before, $\hat { A } _ { c o n t r o l l e d } = r _ { t } - 0 . 1 9 2 6$ , and

$$
\hat { A } _ { j o i n t } \ | \ \mathrm { M o s t l y } \ 0 = r _ { t } - 0 . 1 3 1 3 , \qquad \hat { A } _ { j o i n t } \ | \ \mathrm { M o s t l y } \ 1 = r _ { t } - 0 . 2 5 4 0\tag{50}
$$

Both are unbiased for $A ( h _ { t } ^ { c } , \{ 0 , 0 \} ) = 0 . 1 3 7 4$

Variance computation. The outcome probabilities are unchanged from the previous example, since they depend only on the uncontrolled policy and the conditional reward distribution given $a _ { t } = \{ 0 , 0 \}$ Computing variances as expected squared deviations from the mean:

$$
{ \begin{array} { r l } & { { \mathrm { V a r } } { \left[ { \hat { A } } _ { c o n t r o l l e d } \right] } = 0 . 3 3 0 \cdot ( 0 . 8 0 7 4 - 0 . 1 3 7 4 ) ^ { 2 } + 0 . 6 7 0 \cdot ( - 0 . 1 9 2 6 - 0 . 1 3 7 4 ) ^ { 2 } } \\ & { \qquad = 0 . 3 3 0 \cdot 0 . 4 4 8 9 + 0 . 6 7 0 \cdot 0 . 1 0 8 9 \approx 0 . 2 2 1 } \end{array} }\tag{51}
$$

$$
\begin{array} { r l } & { \mathrm { V a r } \big [ \hat { A } _ { j o i n t } \big ] = 0 . 3 2 8 1 \cdot ( 0 . 8 6 8 7 - 0 . 1 3 7 4 ) ^ { 2 } + 0 . 1 7 2 0 \cdot ( - 0 . 1 3 1 3 - 0 . 1 3 7 4 ) ^ { 2 } } \\ & { \phantom { \hat { = } } + 0 . 0 0 1 9 5 \cdot ( 0 . 7 4 6 0 - 0 . 1 3 7 4 ) ^ { 2 } + 0 . 4 9 8 0 \cdot ( - 0 . 2 5 4 0 - 0 . 1 3 7 4 ) ^ { 2 } } \\ & { \approx 0 . 2 8 1 } \end{array}\tag{52}
$$

Result. The joint history baseline now increases the variance of the advantage estimator:

$$
\mathrm { V a r } \left[ \hat { A } _ { c o n t r o l l e d } \right] \approx 0 . 2 2 1 , \qquad \mathrm { V a r } \left[ \hat { A } _ { j o i n t } \right] \approx 0 . 2 8 1\tag{53}
$$

The intuition mirrors the previous example. The action {0, 0} is now played with low probability and yields high reward precisely when paired with the uncontrolled policy (Mostly 0) under which the controlled policy performs poorly. Conditioning on the uncontrolled history therefore amplifies, rather than absorbs, the reward fluctuations driven by $r _ { t } ,$ , and the joint baseline ends up tracking variation that the controlled baseline averaged out.

## B Proofs of the Main Results

## B.1 Unbiasedness of Advantage Estimators

ProofofTheorem 4.4. For the controlled estimator, since $V ( h _ { t } ^ { c } )$ is constant under the conditioning on $h _ { t } ^ { c . }$

$$
\begin{array} { r l } & { \mathbb { E } _ { h _ { t } ^ { u } } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } \big [ \hat { A } _ { c o n t r o l l e d } \big ] = \mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } [ \hat { Q } _ { t } ^ { ( 1 ) } ] - \mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } } [ \hat { V } ( h _ { t } ^ { c } ) ] } \\ & { \qquad = Q ( h _ { t } ^ { c } , a _ { t } ) - V ( h _ { t } ^ { c } ) } \\ & { \qquad = A ( h _ { t } ^ { c } , a _ { t } ) } \end{array}\tag{54}
$$

For the joint history estimator:

$$
\mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { j o i n t } \big ] = \mathbb { E } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] - \mathbb { E } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ V ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ]\tag{55}
$$

Since the policy $\pi ( a _ { t } \ | \ h _ { t } ^ { c } )$ does not depend on $h _ { t } ^ { u }$ , we have $p ( h _ { t } ^ { u } \mid h _ { t } ^ { c } , a _ { t } ) = p ( h _ { t } ^ { u } \mid h _ { t } ^ { c } )$ therefore:

$$
\mathbb { E } _ { \boldsymbol { h } _ { t } ^ { u } | { \boldsymbol { h } } _ { t } ^ { c } , { \boldsymbol { a } } _ { t } } \big [ V ( { \boldsymbol { h } } _ { t } ^ { c } , { \boldsymbol { h } } _ { t } ^ { u } ) \big ] = \mathbb { E } _ { \boldsymbol { h } _ { t } ^ { u } | { \boldsymbol { h } } _ { t } ^ { c } } \big [ V ( { \boldsymbol { h } } _ { t } ^ { c } , { \boldsymbol { h } } _ { t } ^ { u } ) \big ] = V ( { \boldsymbol { h } } _ { t } ^ { c } )\tag{56}
$$

Substituting, and again using the unbiasedness of $\hat { Q } _ { t } ^ { ( 2 ) }$ :

$$
\mathbb { E } _ { \boldsymbol { h } _ { t } ^ { u } , \tau \sim \pi | \boldsymbol { h } _ { t } ^ { c } , a _ { t } } \big [ \hat { \boldsymbol { A } } _ { j o i n t } \big ] = \boldsymbol { Q } ( \boldsymbol { h } _ { t } ^ { c } , a _ { t } ) - \boldsymbol { V } ( \boldsymbol { h } _ { t } ^ { c } ) = \boldsymbol { A } ( \boldsymbol { h } _ { t } ^ { c } , a _ { t } )\tag{57}
$$

## B.2 Advantage Variance Decomposition

Proof of Theorem 4.7. For the controlled estimator, expanding the variance gives:

$$
\mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { \mathrm { c o n t r o l l e d } } \big ] =\tag{58}
$$

$$
\begin{array} { r l } & { \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } \big ] + \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } ) \big ] - 2 \operatorname { C o v } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } , \hat { V } ( h _ { t } ^ { c } ) \big ] } \end{array}\tag{59}
$$

By Assumption 4.6, $\mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } } \big [ \hat { V } ( h _ { t } ^ { c } ) \big ] = 0 , \mathrm { s o } \mathrm { C o v } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } , \hat { V } ( h _ { t } ^ { c } ) \big ] = 0 ,$ , and we are left with

$$
\mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { \mathrm { c o n t r o l l e d } } \big ] = \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } \big ]\tag{60}
$$

For the joint history estimator:

$$
\begin{array} { r l } & { \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi \mid h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { j o i n t } \big ] = \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi \mid h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] + \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi \mid h _ { t } ^ { c } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \\ & { \phantom { \operatorname { = } \operatorname { 0 } } - 2 \operatorname { C o v } _ { h _ { t } ^ { u } , \tau \sim \pi \mid h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{61}
$$

Taking the difference:

$$
\begin{array} { r l } & { { \mathrm { V a r } } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \left[ \hat { A } _ { c o n t r o l l e d } \right] - { \mathrm { V a r } } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \left[ \hat { A } _ { j o i n t } \right] } \\ & { \quad = { \mathrm { V a r } } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } \big ] - { \mathrm { V a r } } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] } \\ & { \qquad + 2 { \mathrm { C o v } } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] - { \mathrm { V a r } } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{62}
$$

We now simplify the last two terms. By Assumption $4 . 6 , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ is a deterministic function of $\left( h _ { t } ^ { c } , h _ { t } ^ { u } \right)$ , so all variability in $\hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ comes from $h _ { t } ^ { u }$

$$
\begin{array} { r } { \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] = \operatorname { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{63}
$$

For the covariance, we apply the law of total covariance, conditioning on $h _ { t } ^ { u }$ :

$$
\begin{array} { r l } & { \mathrm { C o v } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \\ & { \quad = \underbrace { \mathbb { E } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \Big [ \mathrm { C o v } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] \Big ] } _ { \qquad = 0 \mathrm { b y ~ A s s u m p t i o n ~ 4 . 6 ~ } } } \\ & { \qquad + \mathrm { C o v } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \Big [ \mathbb { E } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] , \mathbb { E } _ { \tau \sim \pi | h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] \Big ] } \\ & { \qquad = \mathrm { C o v } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{64}
$$

where in the last step follows from Assumption 4.6. Substituting both simplifications:

$$
\begin{array} { r l } & { \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { c o n t r o l l e d } \big ] - \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { j o i n t } \big ] } \\ & { \quad = \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } \big ] - \mathrm { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] } \\ & { \qquad + 2 \mathrm { C o v } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) , \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] - \mathrm { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{65}
$$

To obtain the final form, we apply the identity $2 \operatorname { C o v } [ X , Y ] - \operatorname { V a r } [ Y ] = \operatorname { V a r } [ X ] - \operatorname { V a r } [ X - Y ]$ , with $X = q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } )$ and $Y = \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } )$ :

$$
\begin{array} { r l } & { \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { c o n t r o l l e d } \big ] - \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { A } _ { j o i n t } \big ] } \\ & { \quad = \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 1 ) } \big ] - \operatorname { V a r } _ { h _ { t } ^ { u } , \tau \sim \pi | h _ { t } ^ { c } , a _ { t } } \big [ \hat { Q } _ { t } ^ { ( 2 ) } \big ] } \\ & { \qquad + \operatorname { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) \big ] - \operatorname { V a r } _ { h _ { t } ^ { u } | h _ { t } ^ { c } , a _ { t } } \big [ q ^ { ( 2 ) } ( h _ { t } ^ { c } , h _ { t } ^ { u } , a _ { t } ) - \hat { V } ( h _ { t } ^ { c } , h _ { t } ^ { u } ) \big ] } \end{array}\tag{66}
$$