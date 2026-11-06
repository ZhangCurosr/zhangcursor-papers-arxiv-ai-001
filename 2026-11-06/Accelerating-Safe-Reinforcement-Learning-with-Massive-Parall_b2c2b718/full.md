# Accelerating Safe Reinforcement Learning with Massive Parallelism

Joonyoung Lim Department of Information Convergence Engineering Pusan National University ingyuh1015@pusan.ac.kr

Younghwan Yoo<sup>∗</sup> School of Computer Science and Engineering Pusan National University ymomo@pusan.ac.kr

## Abstract

We investigate the problem of scaling safe reinforcement learning (safe RL) to massively parallel training regimes, where a large number of environments are executed with short rollout horizons. Although this setting improves data throughput and wall-clock efficiency, it creates a mismatch with constrained Markov decision processes (CMDPs), whose safety constraints are defined over full-episode cost returns. We address this mismatch by analyzing staggered environment resets as a phase-mixture distribution over short trajectory segments. This analysis characterizes the bias induced by phase coverage and motivates a phase-aggregated cost estimator that reconstructs episode-level cost estimates from short rollout batches. We further incorporate finite-sample uncertainty and distribution mismatch into a constraint-tightening framework, yielding a scalable Safe RL method compatible with on-policy optimization. Using an MJX-based implementation of Safety Gym, our experiments show that the proposed method achieves performance comparable to non-massively parallel implementations while substantially reducing wall-clock training time. Code is available at: GitHub Repository.

## 1 Introduction

Reinforcement learning (RL) provides a general framework for learning decision-making policies through sequential interaction with an environment and is commonly formulated as a Markov decision process (MDP). In many real-world applications, however, policies must satisfy safety requirements such as collision avoidance, energy limits, and operational constraints [7]. Safe RL addresses this challenge through constrained MDPs (CMDPs), where policy optimization is performed under explicit constraints [3].

Recent advances in accelerator hardware have enabled massively parallel simulation, where simulation and learning can be executed with high throughput on devices such as GPUs and TPUs [11]. This regime is particularly beneficial for on-policy RL, which requires data collected from the current policy for each update and has traditionally suffered from limited sample efficiency. By collecting short rollouts concurrently from many environments, massively parallel training enables frequent policy updates with large effective batches [16]. It also substantially reduces wall-clock training time, making repeated runs for statistical evaluation and rapid parameter tuning feasible within a limited experimental budget [2, 5]. Despite these advantages, Safe RL has not fully incorporated this training regime, and existing methods remain misaligned with its data-processing characteristics.

A key challenge in applying massively parallel training to Safe RL is estimating reliable cost returns for policy update. In CMDPs, constraints are typically formulated in terms of cumulative costs, and existing Safe RL implementations commonly rely on episodic cost returns from completed or terminated episodes to balance reward maximization and constraint satisfaction [7, 18]. However, updates in massively parallel training are highly likely to occur without any episode terminating in any of the parallel environments, particularly in tasks without early termination [4]. Even when an episode terminates, the cost return of the full trajectory is biased by earlier policies in two ways: P1 the cost return includes costs incurred by actions taken under earlier policies, and P2 these actions influence the initial-state distributions of subsequent rollouts.

In this paper, we first revisit staggered reset [4], originally introduced to address the cyclical nonstationarity problem. By initializing parallel environments at different episode time steps, each update batch covers the entire episode horizon through short rollouts from different environments. This enables cost estimation by aggregating short rollout segments from different phases, each generated entirely by the current policy, which can address P1. We further analyze how policy history biases the phase-initial state distribution and apply tightening to handle P2. The resulting framework enables existing Safe RL algorithms to train in the massively parallel regime, improving wall-clock efficiency while maintaining reliable constraint satisfaction. The main contributions are summarized as follows:

• We analyze staggered reset as a phase-mixture batch collection scheme, characterize how policy history biases cost returns and derive computable bounds for the return discrepancy.

• We propose a massively parallel safe-RL framework with staggered reset that estimates cost directly from the phase-mixture batch and instantiate it with phase-initial state distribution correction estimation (PIS-DICE), a phase-initial density-ratio estimator for rollout-start state distributions.

• We provide an implementation of Safety Gym exploration tasks using MuJoCo XLA (MJX) for end-to-end accelerator side training and evaluation, showing improved wall-clock efficiency with reliable cost control.

## 2 Preliminaries

## 2.1 Constrained Markov Decision Processes

A CMDP is defined by $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , d _ { 0 } , r , c , \gamma , b _ { c } )$ , where $s$ is the state space, A is the action space, $P ( s ^ { \prime } \mid s , a )$ is the transition probability, d is the initial state distribution, $r : \mathcal { S } \times \mathcal { A } $ R is the reward function, $c : \mathcal { S } \times \mathcal { A } \stackrel { \textstyle - } { \to } \left[ 0 , c _ { \mathrm { m a x } } \right]$ is the cost function, $\gamma \in [ 0 , 1 )$ is the discount factor, and $b _ { c }$ is the cost limit. Under a stationary policy $\pi ,$ the discounted CMDP objective is defined as follows [7, 18]:

$$
\operatorname* { m a x } _ { \pi } J _ { r } ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { t } \right] \quad \mathrm { s . t . } \quad J _ { c } ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } c _ { t } \right] \leq b _ { c } .\tag{1}
$$

Assumption 2.1 (Bounded rewards, costs, and returns). For $f \in \{ r , c \}$ , there exists $f _ { \mathrm { m a x } } < \infty$ such that $| f ( s , a ) | \leq f _ { \operatorname* { m a x } }$ for all $( s , a ) \in S \times A$

For discounted infinite-horizon returns, the bound is $B _ { f } = f _ { \mathrm { m a x } } / ( 1 - \gamma )$ . For finite-horizon returns, the bound is $\begin{array} { r } { B _ { f } = f _ { \operatorname* { m a x } } \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } } \end{array}$ , with $\gamma = 1$ allowed for undiscounted episodic returns. The cost limit $b _ { c }$ must be expressed on the same return scale as $J _ { c }$

Although we treat discounted returns, many safe RL tasks set the cost threshold as an undiscounted episodic sum. To support both, the return bound is $B _ { f } = f _ { \mathrm { m a x } } / ( 1 - \gamma )$ for the discounted infinitehorizon case and $\begin{array} { r } { B _ { f } = f _ { \operatorname* { m a x } } \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } } \end{array}$ for a finite horizon $H$

## 2.2 Massively parallel data collection under staggered reset

In massively parallel on-policy training, each policy update uses length-K rollouts from $E$ parallel environments. We consider episodic tasks with a fixed horizon $H ,$ , so the discounted returns in Eq. (1)

truncate at H. When $K \ll H$ , synchronous reset concentrates each batch on a narrow part of the task horizon, inducing a repeating distribution shift known as cyclical nonstationarity [16, 11, 4]. Staggered reset mitigates this by starting groups of environments at different episode steps, so that a single batch covers the whole horizon rather than a narrow slice.

On-policy RL has traditionally updated frequently from relatively small batches, whereas massively parallel simulation collects short rollouts across a large number of environments and updates from a much larger batch at far lower wall-clock cost. Safe RL fits this regime naturally, since its on-policy algorithms already rely on large batches to estimate the cost constraint with low variance. Massively parallel safe RL has been demonstrated on Isaac Gym [12], but it has not become widespread and has not been studied from an algorithm training perspective, in part because the available tasks have short horizons which avoid the short rollout regime $( K \ll H )$

## 2.3 Distribution correction from flow identities

Distribution correction is commonly used in off-policy and offline RL to correct the mismatch between the distribution induced by a target policy and the distribution represented by sampled transitions [14, 22, 23]. For the discounted infinite-horizon formulation, let $d ^ { \pi } ( s , a )$ denote the normalized discounted state–action occupancy distribution induced by a stationary policy π, and let $d ^ { D } ( s , a )$ denote the underlying state–action sampling distribution from which the transitions in $D$ are collected.

Assumption 2.2 (Absolute continuity). $d ^ { \pi }$ is absolutely continuous with respect to $d ^ { D }$

With Assumption 2.2, the density ratio is defined as $\begin{array} { r } { \rho ^ { \pi / D } ( s , a ) = \frac { d ^ { \pi } ( s , a ) } { d ^ { D } ( s , a ) } } \end{array}$ . Multiplying by $\rho ^ { \pi / D }$ turns an expectation under $d ^ { D }$ into an expectation under $d ^ { \pi }$ , so for a bounded function $f ,$

$$
\begin{array} { r } { {  { \mathbb E } } _ { ( s , a ) \sim d ^ { \pi } } [ f ( s , a ) ] = {  { \mathbb E } } _ { ( s , a ) \sim d ^ { D } } \left[ \rho ^ { \pi / D } ( s , a ) f ( s , a ) \right] . } \end{array}\tag{2}
$$

However, neither density is available in closed form and $d ^ { D }$ is accessed only through samples, so $\rho ^ { \pi / D }$ cannot be computed directly. Instead, under a stationary policy $\pi ,$ , the discounted state-action distribution $d ^ { \pi }$ satisfies the flow identity

$$
d ^ { \pi } ( s ^ { \prime } , a ^ { \prime } ) = ( 1 - \gamma ) d _ { 0 } ( s ^ { \prime } ) \pi ( a ^ { \prime } \mid s ^ { \prime } ) + \gamma \pi ( a ^ { \prime } \mid s ^ { \prime } ) \sum _ { s , a } d ^ { \pi } ( s , a ) P ( s ^ { \prime } \mid s , a ) .\tag{3}
$$

The flow into $( s ^ { \prime } , a ^ { \prime } )$ splits into the initial-state contribution and the transition flow from previous state-action pairs. Substituting $d ^ { \pi } = \rho ^ { \pi / D } d ^ { D }$ into Eq. (3) leaves $\rho ^ { \pi / D }$ as the only unknown, so distribution correction estimation (DICE) methods [14, 22, 23] estimate $\rho ^ { \pi / D }$ by enforcing Eq. (3) on samples of D, without estimating the two densities separately.

## 3 Methodology

We formalize constraint estimation from phase-mixture batches collected from environments with different phases. We first analyze the return computed from the collected batch and compare it with the corresponding current policy return. We then construct phase-wise cost surrogates and tighten the update by considering finite-sample error and distribution mismatch. Finally, we specialize the construction to uniform staggered reset and introduce PIS-DICE as the distribution correction estimator for phase-initial states.

## 3.1 Phase-mixture return analysis

When training with parallel environments and short rollouts, early termination of individual environments or staggered reset distributes environments over phases $\phi \in \{ 0 , \ldots , H - 1 \}$ [2, 4]. At epoch $n ,$ define the phase distribution by $\begin{array} { r } { q _ { n } ( \phi ) = \frac { 1 } { \cal E } | \{ e : \phi _ { e } \dot { = } \phi \} | } \end{array}$ , and let $E _ { \phi } = E q _ { n } ( \phi )$ be the number of environments in phase ϕ. Let $d _ { n , \phi }$ be the collected phase-initial state distribution [21, 8, 20]. Under policy $\pi _ { n } .$ , define $\zeta _ { n , \phi , k } ( s , a ) = ( P _ { \pi _ { n } } ^ { k } d _ { n , \phi } ) ( s ) \pi _ { n } ( a | s )$ . For any return function $f \in \{ r , c \}$ , set $\ell _ { f , n } ( \phi , k ) = \mathbb { E } _ { ( s , a ) \sim \zeta _ { n , \phi , k } } [ f ( s , a ) ]$ . The expected K-step return from rollouts initialized at phase $\phi$

is

$$
j _ { f , n } ^ { K } ( \phi ) = \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \ell _ { f , n } ( \phi , k ) .\tag{4}
$$

For the collected batch, the empirical K-step phase-mixture return is

$$
\widehat { j } _ { f , n } ^ { K } = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } f \big ( s _ { e , \phi _ { e } + k } , a _ { e , \phi _ { e } + k } \big ) .\tag{5}
$$

Lemma 3.1. For a batch with phase distribution $q _ { n } ,$ , the empirical K-step return is an unbiased estimator ofthe phase-mixture return, given by

$$
\mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) j _ { f , n } ^ { K } ( \phi ) .\tag{6}
$$

Proof. See Appendix A.1.1.

The collected phase-initial distribution $d _ { n , \phi }$ may be shaped by previous policies, which changes the state–action distribution $\zeta _ { n , \phi , k }$ and the expected return $\ell _ { f , n } ( \phi , k )$ . Let $\bar { d } _ { n , \phi } = P _ { \pi _ { n } } ^ { \phi } d _ { 0 }$ denote the phase-initial distribution induced by the current policy $\pi _ { n } .$ . The corresponding current policy state–action distribution is $\bar { \zeta } _ { n , \phi , k } ( s , a ) \overset { \cdot } { = } ( P _ { \pi _ { n } } ^ { k } \bar { d } _ { n , \phi } ) \hat { ( } s ) \pi _ { n } \cdot \bar { ( } a | s )$ ). The current policy one-step return is $\bar { \ell } _ { f , n } ( \phi , k ) = \mathbb { E } _ { ( s , a ) \sim \bar { \zeta } _ { n , \phi , k } } [ f ( s , a ) ]$ , and the corresponding K-step phase-mixture return is

$$
\bar { j } _ { f , n } ^ { K } = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \bar { \ell } _ { f , n } ( \phi , k ) .\tag{7}
$$

Proposition 3.2 (Initial phase state bias bound). Under Assumption 2.1, for $f \in \{ r , c \}$ , the initial phase state mismatch bias can be bounded as

$$
\left| \mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] - \bar { j } _ { f , n } ^ { K } \right| \leq 2 f _ { \operatorname* { m a x } } \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \left. \zeta _ { n , \phi , k } - \bar { \zeta } _ { n , \phi , k } \right. _ { \mathrm { T V } } .\tag{8}
$$

Proof. See Appendix A.1.2.

Proposition 3.2 identifies the distributional term separating the phase-mixture return estimated from the batch and the corresponding current policy phase-mixture return. Even in on-policy algorithms, this correction is required when the collected phase-initial distribution $d _ { n , \phi }$ differs from the current policy phase-initial distribution $\bar { d } _ { n , \phi }$

## 3.2 Phase-wise cost surrogates and tightening

Since safe RL algorithms need cost returns for dual updates or feasibility recovery, we consider two cost surrogates from the collected phase-mixture batch. The sampled cost surrogate aggregates the sampled cost $c ( s , a )$ with phase-coverage correction. The critic value surrogate evaluates the cost critic $V _ { c , n } ( s )$ on states sampled from the same phase-mixture batch. Define the binary indicator $I _ { \phi , k } ( t ) = \dot { \mathbb { I } } \{ t = \phi + k \}$ . The coverage mass of timestep t and the coverage-correction weight are

$$
m _ { q _ { n } } ( t ) = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } I _ { \phi , k } ( t ) , \qquad w _ { q _ { n } } ( \phi , k ) = \frac { \gamma ^ { \phi + k } } { m _ { q _ { n } } ( \phi + k ) } .\tag{9}
$$

Assume $m _ { q _ { n } } ( t ) > 0$ for $t = 0 , \ldots , H - 1$ , and let $\Phi _ { n } : = \{ \phi : q _ { n } ( \phi ) > 0 \}$ denote the set of observed phases. The two phase-wise surrogates are

$$
\widehat { J } _ { c , n } ^ { K , \mathrm { s a m p l e } } = \sum _ { \phi \in \Phi _ { n } } q _ { n } ( \phi ) \frac { 1 } { E _ { \phi } } \sum _ { e : \phi _ { e } = \phi } \sum _ { k = 0 } ^ { K - 1 } w _ { q _ { n } } ( \phi , k ) c ( s _ { e , \phi _ { e } + k } , a _ { e , \phi _ { e } + k } ) ,
$$

$$
\widehat { J } _ { c , n } ^ { K , \mathrm { v a l u e } } = \sum _ { \phi \in \Phi _ { n } } q _ { n } ( \phi ) \frac { 1 } { E _ { \phi } K } \sum _ { e : \phi _ { e } = \phi } \sum _ { k = 0 } ^ { K - 1 } V _ { c , n } \big ( s _ { e , \phi _ { e } + k } \big ) .\tag{10}
$$

Here $V _ { c , n } ( s _ { t } )$ predicts $\begin{array} { r } { \mathbb { E } _ { \pi _ { n } } [ \sum _ { \tau = t } ^ { \infty } \gamma ^ { \tau - t } c ( s _ { \tau } , a _ { \tau } ) \mid s _ { t } ] , } \end{array}$

Proposition 3.3 (Finite-sample error bound). Under Assumption 2.1, choose either surrogate in $E q .$ (10) and write it as $\begin{array} { r } { \widehat { J } _ { c , n } ^ { K } = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } X _ { e } } \end{array}$ , where the $X _ { e }$ are independent per-environment random variables. Let $R _ { n }$ be a deterministic range satisfying $0 \leq X _ { e } \leq R _ { n } ^ { - }$ for every environment e, and let $\begin{array} { r } { S ^ { 2 } = \frac { 1 } { E - 1 } \sum _ { e = 1 } ^ { E } ( X _ { e } - \widehat { J } _ { c , n } ^ { K } ) ^ { 2 } . } \end{array}$ . By the one-sided empirical Bernstein inequality, the finite-sample error of the cost surrogate is bounded with probability at least $1 - \delta ,$

$$
\mathbb { E } \left[ \widehat { J } _ { c , n } ^ { K } \right] - \widehat { J } _ { c , n } ^ { K } \leq \sqrt { \frac { 2 S ^ { 2 } \log ( 2 / \delta ) } { E } } + \frac { 7 R _ { n } \log ( 2 / \delta ) } { 3 ( E - 1 ) } .\tag{11}
$$

Proof. See Appendix A.1.3.

Theorem 3.4 (Constraint satisfaction with tightened cost surrogate). Assume that the selected surrogate scale satisfies the distribution mismatch upper bound $J _ { c } ( \pi _ { n } ) \le \mathbb { E } [ \widehat { J } _ { c , n } ^ { K } ] + \varepsilon _ { n }$ . Together with Propositions 3.2 and 3.3, ifthe cost surrogatefrom the phase-mixture batch satisfies

$$
\begin{array} { r } { \widehat { J } _ { c , n } ^ { K } + \beta _ { n } + \varepsilon _ { n } \leq b _ { c } , } \end{array}\tag{12}
$$

then $J _ { c } ( \pi _ { n } ) \le b _ { c }$ with probability at least $1 - \delta .$

Proof. See Appendix A.1.4.

## 3.3 Uniform staggered reset

Massively parallel environment training is typically executed under JIT/XLA compilation [11, 16], where keeping rollout tensors shape stable after compilation is important. Therefore, staggered reset partitions the $\bar { \boldsymbol { { \mathbf { \mathit { E } } } } }$ environments into M phase groups and assigns each group a fixed phase offset m $K$ for $m = 0 , \ldots , M - 1$ along the task horizon $\dot { H } = M K$ . Environments that terminate early are kept in a masked terminal state until the group reset point or the episode end point, so no individual environment reset is performed. This group-wise distribution yields a uniform phase allocation over non-overlapping horizon segments, with mass $1 / M$ on each observed phase and no mass on other phases, and provides a practical instance of the phase-mixture analysis in Section 3.1.

Theorem 3.5 (Uniform staggered reset induces a Cesàro phase mixture). Under uniform staggered reset, the phase-mixture return in Eq. (6) is the finite Cesàro mean ofthe phase-wise returns $\breve { j } _ { f , n } ^ { K } ( \phi )$ from Eq. (4), namely $\begin{array} { r } { \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } j _ { f , n } ^ { K } ( m K ) } \end{array}$

Proof. See Appendix A.1.5.

Corollary 3.6 (Cesàro phase-mixture return). Under uniform staggered reset, the phase-mixture return in Lemma 3.1 becomes

$$
\mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } j _ { f , n } ^ { K } ( m K ) .\tag{13}
$$

Corollary 3.7 (Cesàro initial phase state bias bound). Under uniform staggered reset and Assumption 2.1, the mismatch bound in Proposition 3.2 becomes

$$
\left| \mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] - \bar { j } _ { f , n } ^ { K } \right| \leq \frac { 2 f _ { \operatorname* { m a x } } } { M } \sum _ { m = 0 } ^ { M - 1 } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \left\| \zeta _ { n , m K , k } - \bar { \zeta } _ { n , m K , k } \right\| _ { \mathrm { T V } } .\tag{14}
$$

Corollary 3.8 (Uniform phase-wise cost surrogate under staggered reset). Under uniform staggered reset, the phase-wise surrogate in $E q .$ (10) becomes

$$
\widehat { J } _ { c , n } ^ { K , \mathrm { s a m p l e } } = \sum _ { m = 0 } ^ { M - 1 } \frac { 1 } { E _ { m K } } \sum _ { e : \phi _ { e } = m K } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { m K + k } c ( s _ { e , m K + k } , a _ { e , m K + k } ) ,
$$

$$
\widehat { J } _ { c , n } ^ { K , \mathrm { v a l u e } } = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } \frac { 1 } { E _ { m K } K } \sum _ { e : \phi _ { e } = m K } \sum _ { k = 0 } ^ { K - 1 } V _ { c , n } ( s _ { e , m K + k } ) .\tag{15}
$$

The corollary proofs are given in Appendices A.1.6, A.1.7, and A.1.8, respectively.

## 3.4 PIS-DICE: phase-initial state distribution correction

The TV term in Eq. (14) compares collected state–action distributions with current policy state–action distributions at each phase and rollout offset, which cannot be estimated directly in the short-rollout training regime. Therefore, we extend the concept of distribution correction from state–action distributions to phase-initial state distributions.

Under uniform staggered reset, a phase-initial pair is $x = ( s , \phi )$ , where s is the state at the start of phase $\phi .$ . Let ${ \mathcal { K } } _ { n } ( { \bar { x ^ { + } } } \mid x )$ be the transition kernel obtained after executing $\pi _ { n }$ for $K$ environment steps. For consecutive phase-initial pairs used by the estimator, $x ^ { + } = ( s ^ { + } , \bar { \phi } + K )$ with $\phi + K < H$ For every observed phase, assume $d _ { n , \phi }$ is absolutely continuous with respect to $d _ { n , \phi }$ . Writing their densities with the same symbols, define the phase-initial correction weight $\begin{array} { r } { \bar { \rho } _ { n } ( s , \phi ) = \frac { \bar { d } _ { n , \phi } ( s ) } { d _ { n , \phi } ( s ) } } \end{array}$ , so $\mathbb { E } _ { s \sim d _ { n , \phi } } [ \bar { \rho } _ { n } ( s , \phi ) ] { = } 1$ . For each interior phase $m = 0 , \ldots , M - 2$ , the phase-initial one-block relation is

$$
\mathbb { E } _ { s \sim d _ { n , m K } , x ^ { + } \sim K _ { n } ( \cdot \vert ( s , m K ) ) } \left[ \bar { \rho } _ { n } ( s , m K ) h ( x ^ { + } ) \right] = \mathbb { E } _ { s \sim \bar { d } _ { n , ( m + 1 ) K } } \left[ h ( s , ( m + 1 ) K ) \right] , \qquad \forall h \in \mathcal { H } .\tag{16}
$$

Here $\mathcal { H }$ is a class of bounded measurable functions. The sampled one-block residual below is used as an empirical fitting objective for the learned weights, while the formal bounds use the change-of-measure identity directly. For a fixed phase $\phi ,$ let $j _ { c , n , \phi } ^ { K } ( s )$ be the $K \cdot$ step expected cost return under the current policy from phase state $s .$ Averaging this conditional expected return over the collected and current policy phase-initial state distributions gives $C _ { n , \phi } ^ { K } { = } \mathbb { E } _ { s \sim d _ { n , \phi } } ^ { \bullet } [ j _ { c , n , \phi } ^ { K } ( s ) ]$ and $\bar { C } _ { n , \phi } ^ { K } { = } \mathbb { E } _ { s \sim \bar { d } _ { n , \phi } } [ j _ { c , n , \phi } ^ { K } ( s ) ]$ ]. The phase-initial change of measure gives $\bar { C } _ { n , \phi } ^ { K } - C _ { n , \phi } ^ { K } = \mathbb { E } _ { s \sim d _ { n , \phi } } [ ( \bar { \rho } _ { n } ( s , \phi ) - 1 ) j _ { c , n , \phi } ^ { K } ( s ) ]$ . Thus, we use this phase-initial correction to handle the TV term in Eq. (14).

PIS-DICE uses two learned models. The ratio model outputs a nonnegative phase-initial correction weight. The residual model maps a phase-initial pair to a scalar whose difference across one rollout block defines an auxiliary fitting objective for the ratio model. In the implementation, the ratio-model output is $\rho _ { n } ( s , \phi ) = \epsilon _ { \rho } + \mathrm { s o f t p l u s } ( u _ { \eta } ( s , \phi / ( H - 1 ) ) )$ , with ${ \epsilon _ { \rho } } > 0$ . Since the exact ratio $\bar { \rho } _ { n } ( s , \phi )$ is not observed, the implementation fits $\rho _ { n }$ using collected phase-initial samples and the states reached after the rollout block. The residual model is used only for this empirical fitting objective. The theoretical statements below are stated for the exact ratio $\bar { \rho } _ { n } .$ , and Proposition 3.10 quantifies the additional error introduced by replacing $\bar { \rho } _ { n }$ with learned weights.

For each environment $e , x _ { \epsilon }$ denotes the phase-initial pair and $ { \boldsymbol { { x } } } _ { e } ^ { + }$ denotes the pair obtained after the collected rollout block. Let $f _ { \omega }$ be the residual-model output. The sampled residual is $\Delta _ { n } ( \eta , \omega ) =$ $\begin{array} { r } { E ^ { - 1 } \sum _ { e = 1 } ^ { E } \rho _ { n } ( x _ { e } ) ( f _ { \omega } ( x _ { e } ) - f _ { \omega } ( x _ { e } ^ { + } ) ) } \end{array}$ . The residual model is trained to expose this discrepancy, and the ratio model is trained to reduce the exposed discrepancy together with phase-wise normalization and ratio regularization. Let $\operatorname { s g } ( \cdot )$ denote stop-gradient. The losses are

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \rho , n } ( \eta ) = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \rho _ { n } ( x _ { e } ) \mathrm { s g } \big [ f _ { \omega } ( x _ { e } ) - f _ { \omega } ( x _ { e } ^ { + } ) \big ] } \\ { \displaystyle + \lambda _ { \mathrm { n o r m } } \mathcal { R } _ { \mathrm { n o r m } , n } ( \rho _ { n } ) + \lambda _ { \rho 2 } \frac { 1 } { E } \sum _ { e = 1 } ^ { E } ( \rho _ { n } ( x _ { e } ) - 1 ) ^ { 2 } , } \\ { \displaystyle \mathcal { L } _ { f , n } ( \omega ) = - \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \mathrm { s g } [ \rho _ { n } ( x _ { e } ) ] \big ( f _ { \omega } ( x _ { e } ) - f _ { \omega } ( x _ { e } ^ { + } ) \big ) + \lambda _ { f } \frac { 1 } { E } \sum _ { e = 1 } ^ { E } f _ { \omega } ( x _ { e } ) ^ { 2 } , } \end{array}\tag{17}
$$

where $\mathcal { R } _ { \mathrm { n o r m } , n }$ is the phase-wise normalization term, $\begin{array} { r } { \mathcal { R } _ { \mathrm { n o r m } , n } ( { \rho } ) = \sum _ { \phi \in \Phi _ { n } } ( | I _ { \phi } | ^ { - 1 } \sum _ { e \in I _ { \phi } } \rho ( x _ { e } ) - } \end{array}$ $1 ) ^ { 2 }$ . After training, the raw weights $\rho _ { n } ( x _ { e } )$ are clipped to $[ 1 / \rho _ { \mathrm { m a x } } , \rho _ { \mathrm { m a x } } ]$ and normalized within each phase. We write the final normalized weight as $\begin{array} { r l r } { \tilde { \rho } _ { n , e } } & { = } & { \frac { | I _ { \phi _ { e } } | \rho _ { n , e } ^ { \mathrm { c l i p } } } { \sum _ { j \in I _ { \phi _ { e } } } \rho _ { n , j } ^ { \mathrm { c l i p } } } } \end{array}$ , where $\rho _ { n , e } ^ { \mathrm { c l i p } } =$ $\begin{array} { r } { \operatorname* { l i p } ( \rho _ { n } ( x _ { e } ) , 1 / \rho _ { \operatorname* { m a x } } , \rho _ { \operatorname* { m a x } } ) , \mathrm { s o } | I _ { \phi } | ^ { - 1 } \sum _ { e \in I _ { \phi } } \tilde { \rho } _ { n , e } = 1 } \end{array}$ for every observed phase. The losses above are used only to fit $\widetilde { \rho } _ { n }$ . The formal bounds below use the exact change-of-measure identity for $\bar { \rho } _ { n }$ and then account for replacing $\bar { \rho } _ { n }$ with $\tilde { \rho } _ { n }$ through an explicit learned-ratio error term.

Proposition 3.9 (PIS-DICE phase-initial discrepancy upper bound). With the exact phase-initial ratio $\bar { \rho } _ { n }$ defined above, under Assumption $2 . I , 0 \overset { \cdot } { \leq } j _ { c , n , \phi } ^ { \check { K } } ( s ) \overset { \cdot } { \leq } B _ { c } ,$ , the phase-wise cost discrepancy

satisfies

$$
\left| \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } \left( \bar { C } _ { n , m K } ^ { K } - C _ { n , m K } ^ { K } \right) \right| \leq \frac { B _ { c } } { M } \sum _ { m = 0 } ^ { M - 1 } \mathbb { E } _ { s \sim d _ { n , m K } } \left[ | \bar { \rho } _ { n } ( s , m K ) - 1 | \right] .\tag{18}
$$

Proof. See Appendix A.1.9.

Proposition 3.9 gives a phase-initial change-of-measure bound for the cost discrepancy induced by initial phase-state mismatch. The bound replaces the state–action TV terms over all phase offsets with a phase-initial correction term. The practical correction therefore depends on the accuracy of the learned ratio.

Proposition 3.10 (Learned PIS-DICE correction error). For the exact phase-initial ratio $\bar { \rho } _ { n } ,$ condition on a learned nonnegative estimate ${ \tilde { \rho } } _ { n } .$ . Assume the selected surrogate cost scalar $Y _ { n } ( s , \phi )$ satisfies $| Y _ { n } ( s , \phi ) | \le B _ { Y }$ . Then

$$
\begin{array} { l } { \displaystyle \left| \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } \mathbb { E } _ { s \sim \bar { d } _ { n , m K } } [ Y _ { n } ( s , m K ) ] - \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } \mathbb { E } _ { s \sim d _ { n , m K } } [ \tilde { \rho } _ { n } ( s , m K ) Y _ { n } ( s , m K ) ] \right| } \\ { \displaystyle \quad \leq \frac { B _ { Y } } { M } \sum _ { m = 0 } ^ { M - 1 } \mathbb { E } _ { s \sim d _ { n , m K } } [ | \bar { \rho } _ { n } ( s , m K ) - \tilde { \rho } _ { n } ( s , m K ) | ] . } \end{array}\tag{19}
$$

Proof. See Appendix A.1.10.

## 3.5 Practical update rule

The implementation uses the uniform staggered reset form in Eq. (15). The sampled cost form accumulates binary costs from the phase-mixture batch and compares the result with $b _ { c }$ on the task cost-return scale. The critic value form evaluates the same batch with $V _ { c , n }$ and compares it with the cost limit on the critic-return scale.

For the selected cost surrogate, write the per-environment scalar as $Y _ { n , e }$ For the sampled cost form, $\begin{array} { r } { Y _ { n , e } { = } M \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { \phi _ { e } + k } c ( s _ { e , \phi _ { e } + k } , a _ { e , \phi _ { e } + k } ) } \end{array}$ ; for the critic value form, $\begin{array} { r } { Y _ { n , e } { = } K ^ { - 1 } \sum _ { k = 0 } ^ { K - 1 } V _ { c , n } ( s _ { e , \phi _ { e } + k } ) } \end{array}$ . The finite-sample term $\beta _ { n }$ is computed from Proposition 3.3 for the selected scalar $Y _ { n , e } .$

PIS-DICE provides the normalized learned weight $\tilde { \rho } _ { n , e }$ . The update keeps the unweighted cost surrogate and uses $D _ { n , e } { = } ( \tilde { \rho } _ { n , e } - 1 ) Y _ { n , e }$ to compute the mismatch correction

$$
\varepsilon _ { n } ^ { \mathrm { P I S } } = \left[ \frac { 1 } { E } \sum _ { e = 1 } ^ { E } D _ { n , e } + \sqrt { \frac { 2 S _ { D } ^ { 2 } \log ( 1 / \delta ) } { E } } \right] _ { + } , S _ { D } ^ { 2 } = \frac { 1 } { E - 1 } \sum _ { e = 1 } ^ { E } \left( D _ { n , e } - \frac { 1 } { E } \sum _ { j = 1 } ^ { E } D _ { n , j } \right) ^ { 2 } .
$$

The policy update uses $\varepsilon _ { n } = \varepsilon _ { n } ^ { \mathrm { P I S } }$ in Eq. (12), combining the finite-sample term $\beta _ { n }$ with the empirical correction computed from the learned weights. In the main experiments, PIS-DICE changes only the cost-side tightening, while the actor objective remains unchanged.

## 4 Evaluation

We focus on evaluating whether the proposed massively parallel safe RL training framework can maintain reliable cost control with short, accelerator-friendly updates. The experiments first evaluate the effect of uniform staggered reset and the additional effect of tightening. We then compare the massively parallel training approach with the small-vectorized environment setting used in existing implementations in terms of both sample efficiency measured by total environment steps and wall-clock efficiency.

Table 1: Evaluation of CPO across 2k, 4k, and 8k parallel environments with a fixed rollout length of 10. Values are mean ± standard deviation of the last 10 epochs over eight seeds. Shading indicates violation of the cost threshold (cost > 25). Bold highlights, for each environment count, the highest reward that satisfies the cost threshold.
<table><tr><td rowspan="2">Init</td><td rowspan="2">Cost est. Env</td><td rowspan="2"></td><td colspan="4">Point/Goal</td><td colspan="4">Car/Goal</td></tr><tr><td colspan="2">w/o tightening</td><td colspan="2">w/ tightening</td><td colspan="2">w/o tightening</td><td colspan="2">w/ tightening</td></tr><tr><td></td><td></td><td></td><td>Reward</td><td>Cost</td><td>Reward</td><td>Cost</td><td>Reward</td><td>Cost</td><td>Reward</td><td>Cost</td></tr><tr><td>Standard</td><td>LR</td><td>2K</td><td>15.13±9.94</td><td>55.48±24.54</td><td>一</td><td>一</td><td>29.14±4.76</td><td>65.87±12.06</td><td>一</td><td>一</td></tr><tr><td rowspan="6">Staggered</td><td></td><td>2K</td><td>14.57±2.78</td><td>27.08±0.84</td><td>7.56±3.06</td><td>13.88±1.59</td><td>25.12±2.11</td><td>28.63±1.46</td><td>18.52±2.83</td><td>14.38±1.18</td></tr><tr><td>SC</td><td>4K</td><td>17.28±5.37</td><td>26.27±1.23</td><td> $1 3 . 4 2 \pm 3 . 5 2$ </td><td>20.00±3.49</td><td>27.57±1.76</td><td> $2 6 . 8 7 { \scriptstyle \pm 1 . 1 8 }$ </td><td>24.38±2.04</td><td>18.43±0.75</td></tr><tr><td></td><td>8K</td><td>21.23±1.05</td><td> $2 7 . 0 2 { \scriptstyle \pm 1 . 5 0 }$ </td><td> ${ \bf 1 8 . 8 3 } \pm 0 . 7 3$ </td><td>21.19±2.05</td><td>32.17±0.95</td><td> $2 6 . 5 2 { \scriptstyle \pm 1 . 3 8 }$ </td><td>29.57±0.70</td><td>21.06±0.70</td></tr><tr><td></td><td>2K</td><td>15.29±3.76</td><td>29.55±3.43</td><td>13.42±2.06</td><td>24.53±3.93</td><td>26.13±2.57</td><td>28.80±3.16</td><td>22.69±3.00</td><td>22.19±3.23</td></tr><tr><td>CV</td><td>4K</td><td>19.30±2.27</td><td>29.61±3.04</td><td>17.40±2.23</td><td>24.57±2.98</td><td>30.16±2.04</td><td>29.87±3.98</td><td>28.46±1.37</td><td>25.78±3.53</td></tr><tr><td></td><td>8K</td><td> $2 1 . 4 6 { \pm } 1 . 1 7$ </td><td> $2 7 . 8 2 \pm 2 . 2 5$ </td><td> $2 0 . 5 5 { \scriptstyle \pm 2 . 2 9 }$ </td><td>26.91±3.21</td><td> $3 2 . 6 7 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $2 9 . 7 5 { \scriptstyle \pm 2 . 0 6 }$ </td><td>31.21±1.40</td><td>26.37±3.17</td></tr></table>

Setup We implement Goal tasks with Point and Car agents in MJX, following Safety Gym-style navigation task definitions [15, 9] and the accelerator-side simulation design of JAX-based physics engines [6]. We further implement a locomotion task with a Laikago robot [10] in both MJX and MuJoCo Warp. To compare different constraint-handling paradigms, CPO, TRPO-Lag, and CRPO were chosen as representative primal-dual, Lagrangian, and primal methods, respectively [1, 17, 19]. By default, the massively parallel setting uses 2,000 environments collecting a batch size of 20,000 steps per update over 500 epochs. All experiments on the Goal tasks were conducted on Google TPU v4, while locomotion tasks were conducted on machine with NVIDIA GeForce RTX 4090 GPU.

Effect of staggered reset and tightening. We first compare massively parallel training with shortrollout updates using the two initialization schemes, standard synchronous reset and staggered reset, and then evaluate the additional effect of the tightening method. Since standard initialization does not provide a finished-episode cost return at every update, the latest completed-episode cost return is reused for updates without completed episodes; we denote this estimate by LR. Staggered reset uses two cost surrogates, denoted as SC for sampled cost and CV for critic value estimates, evaluated with and without tightening. We further scale parallelism from 2k to 4k and 8k environments, so the batch size increases to 40,000 and 80,000 transitions, respectively.

Table 1 shows the results of CPO on the Goal task across different environment scales. At 2k environments, standard initialization exceeds the cost limit, with completed episode cost returns available only sparsely. Staggered reset reduces cost returns by providing estimates from every phasemixture batch. Without tightening, both surrogates still exceed the cost limit, even as increasing parallelism leads to higher rewards. Applying tightening further reduces costs at the expense of reward: SC keeps the reported mean episodic cost below the limit across the tested scales, whereas CV exceeds it in some configurations. The reward gap between tightened and untightened training narrows at larger scales, consistent with the smaller finite-sample bound afforded by increased sampling.

Comparison with non-massive environment training. We compare the proposed massively parallel setting against the small-vectorized environment setting used in existing implementations, denoted as the standard setting. Figure 1 shows the learning curves of the Goal tasks in terms of wall-clock time and total environment steps. With the same number of environment steps, the massively parallel setting generally converges to lower rewards than the standard setting using full task-horizon rollouts, which achieves better sample efficiency by providing complete task trajectories. However, the primary advantage of massively parallel training lies in wall-clock efficiency rather than sample efficiency. By processing many short rollouts from different phases concurrently on accelerators, the massively parallel setting increases transition throughput and enables more frequent policy updates within the same elapsed time. In terms of wall-clock time, training in the massively parallel setting converges to higher reward levels across algorithms as parallelism increases. In particular, 8k configuration achieves rewards comparable to the standard setting with better constraint satisfaction in substantially less training time.

![](images/83d5ff9691e00e78cd1202d5ba6fb66f84131b02fdf4f967dab23dcd8c7513bc.jpg)

(a) Point/Goal. The left side shows the wall-clock wise comparison and the right side shows the total sample wise comparison.  
![](images/6496613e987f752f885073d79aab8e62ee39a2ae7eb623e298def0544c2abece.jpg)  
(b) Car/Goal. The left side shows the wall-clock wise comparison and the right side shows the total sample wise comparison.  
Figure 1: Wall-clock wise and total sample wise learning curves of all algorithms on the Goal task. Columns correspond to CPO, CRPO, and TRPO-Lag. Markers are shown every 100 epochs and report trailing 10-epoch averages, and the horizontal line marks the cost limit.

Figure 2 shows the learning curves of the Laikago locomotion task implemented using MJX and MuJoCo Warp. Consistent with the trends observed in the Goal tasks, reward convergence improves as parallelism increases, reaching a level comparable to the standard setting at 4K environments and surpassing it at 8K. In terms of wall-clock time, both MJX and MuJoCo Warp achieve substantially shorter training times at 8K environments than the standard setting. The difference between the two implementations further suggests that additional efficiency gains can be obtained through the choice of GPU simulation backend. Overall, these results demonstrate that massively parallel training could provide substantial wall-clock efficiency gains for safe RL research while maintaining or improving performance.

Ablation studies We conduct ablation studies to compare the effects of increasing rollout length against increasing the number of parallel environments and to examine how each tightening term affects training. The results are provided in Appendix B.

![](images/2f58e544ef7ddf91c6aec1618d2e4f665b00104c29e61e2b7d4039e76e9cf3e1.jpg)  
-O- Standard - MASSIVE 2k  MASSIVE 4k → MASSIVE 8k --- Cost limit

(a) Comparison across environment scales in terms of total environment steps, averaged over MJX and MuJoCo Warp.  
![](images/3969a153b5cec96d33866b1dfc2a57470c1154ee0beff17bfabf671e2f6e4c99.jpg)  
-O- Standard MJX -— Standard Warp → MASSIVE 8k MJX → MASSIVE 8k Warp --- Cost limit  
(b) Comparison of the standard and 8k environment settings in terms of wall-clock time using MJX and MuJoCo Warp.

Figure 2: Learning curves of CRPO on the Laikago locomotion task. Markers are shown every 100 epochs and report trailing 10-epoch averages. Costs are normalized by their respective limits, with horizontal lines indicating the normalized cost limit of 1.

## 5 Limitations

The theoretical results assume bounded costs, independent samples across environments, sufficient phase coverage, and uniform staggered resets for the simplified corollaries. Freezing terminated environments wastes some computation but preserves fixed computation shapes for efficient accelera tor execution, allowing the throughput gains from massive parallelism to potentially outweigh this overhead. Nevertheless, extending the implementation to arbitrary asynchronous or highly nonuniform reset schedules remains for future work. The practical tightening rule combines finite-sample tightening with density ratio correction, but the ablations show that most of the improvement in constraint satisfaction comes from the finite-sample term. Further evaluation on complex tasks with greater phase-initial distribution mismatch would help clarify the contribution of distribution correction. Applying tightening to multiple constraints may also become overly conservative and restrict learning, motivating further study of alternatives to tightening. The CV surrogate relies on critics trained with bootstrapped k-step targets from short rollouts and shows less consistent cost control than SC in the reported experiments. The current tightening rule does not account for critic estimation errors, and extending it to cover these errors could further increase conservatism. An alternative is to investigate distributional critics that exploit the large training batches to improve cost estimation.

## 6 Conclusion

We propose a massively parallel Safe RL framework that aligns short-rollout training with episodelevel CMDP constraints through phase-covered cost estimation. By analyzing staggered reset as a phase-mixture data collection scheme, we derive cost surrogates with finite-sample and phaseinitial distribution corrections, and instantiate the correction with PIS-DICE. Experiments on Safety Gym-style Goal tasks implemented in MJX and Laikago locomotion tasks implemented in both MJX and MuJoCo Warp show that the proposed framework improves wall-clock efficiency while keeping costs near their limits. These results demonstrate that safe RL can be effectively extended to massively parallel training regimes, achieving substantial gains in wall-clock efficiency while maintaining constraint satisfaction.

## Acknowledgments and Disclosure of Funding

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) through the Information Technology Research Center program (IITP-2026-RS-2023-00260098) and the Artificial Intelligence Convergence Innovation Human Resources Development program (IITP-2026-RS-2023-00254177).

This work was also supported with Cloud TPUs from Google’s TPU Research Cloud (TRC) program, and we sincerely thank the TRC team for their generous and continuous support.

## References

[1] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pages 22–31. Pmlr, 2017.

[2] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C Courville, and Marc Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Advances in Neural Information Processing Systems, 34, 2021.

[3] Eitan Altman. Constrained Markov Decision Processes. Technical Report RR-2574, INRIA, May 1995. URL https://inria.hal.science/inria-00074109.

[4] Sid Bharthulwar, Stone Tao, and Hao Su. Staggered environment resets improve massively parallel onpolicy reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=hesM5BWtOJ.

[5] Theresa Eimer, Marius Lindauer, and Roberta Raileanu. Hyperparameters in reinforcement learning and how to tune them. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

[6] C Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax–a differentiable physics engine for large scale rigid body simulation. arXiv preprint arXiv:2106.13281, 2021.

[7] Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, and Alois Knoll. A review of safe reinforcement learning: Methods, theories, and applications. IEEE Trans. Pattern Anal. Mach. Intell., 46(12):11216–11235, December 2024. ISSN 0162-8828. doi: 10.1109/TPAMI.2024.3457538. URL https://doi.org/10.1109/TPAMI.2024.3457538.

[8] Ashish Kumar Jayant and Shalabh Bhatnagar. Model-based safe deep reinforcement learning via a constrained proximal policy optimization algorithm. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

[9] Jiaming Ji, Jiayi Zhou, Borong Zhang, Qingyang Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, and Yaodong Liu. Safety-gymnasium: A unified safe reinforcement learning benchmark. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023.

[10] Dohyeong Kim, Taehyun Cho, Seungyub Han, Hojun Chung, Kyungjae Lee, and Songhwai Oh. Spectral risk safe reinforcement learning with convergence guarantees. Advances in Neural Information Processing Systems, 37:92465–92498, 2024.

[11] Sotetsu Koyamada, Shinri Okano, Soichiro Nishimori, Yu Murata, Keigo Habara, Haruka Kita, and Shin Ishii. Pgx: Hardware-accelerated parallel game simulators for reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=UvX8QfhfUx.

[12] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021.

[13] Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization. In Conference on Learning Theory, 2009.

[14] Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in neural information processing systems, 32, 2019.

[15] Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking safe exploration in deep reinforcement learning. arXiv preprint arXiv:1910.01708, 7(1):2, 2019.

[16] Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes using massively parallel deep reinforcement learning. In 5th Annual Conference on Robot Learning, 2021. URL https://openreview.net/forum?id=wK2fDDJ5VcF.

[17] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning, 2015.

[18] Akifumi Wachi, Xun Shen, and Yanan Sui. A survey of constraint formulations in safe reinforcement learning. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI ’24, 2024. ISBN 978-1-956792-04-1. doi: 10.24963/ijcai.2024/913. URL https://doi.org/10. 24963/ijcai.2024/913.

[19] Tengyu Xu, Yingbin Liang, and Guanghui Lan. CRPO: A new approach for safe reinforcement learning with convergence guarantee. In International Conference on Machine Learning, 2021.

[20] Moritz A. Zanger, Karam Daaboul, and J. Marius Zöllner. Safe continuous control with constrained model-based policy optimization. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3512–3519, 2021. doi: 10.1109/IROS51168.2021.9635984.

[21] Sihan Zeng, Thinh T. Doan, and Justin Romberg. Finite-time complexity of online primal-dual natural actor-critic algorithm for constrained markov decision processes. In 2022 IEEE 61st Conference on Decision and Control (CDC), pages 4028–4033, 2022. doi: 10.1109/CDC51059.2022.9992837.

[22] Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. Gendice: Generalized offline estimation of stationary values. In International Conference on Learning Representations, 2020.

[23] Shangtong Zhang, Bo Liu, and Shimon Whiteson. Gradientdice: Rethinking generalized offline estimation of stationary values. In International Conference on Machine Learning, pages 11194–11203, 2020.

## A Supplementary proofs and implementation details

## A.1 Proofs omitted in the main text

## A.1.1 Proof of Lemma 3.1

Proof. Condition on the phase allocation $\{ \phi _ { e } \} _ { e = 1 } ^ { E }$ . For each environment e with $\phi _ { e } \ = \ \phi ,$ , the distribution of $\left( s _ { e , \phi + k } , a _ { e , \phi + k } \right)$ under the rollout policy is $\zeta _ { n , \phi , k }$ . Hence the definition of $\ell _ { f , n }$ gives

$$
\begin{array} { r } { \mathbb { E } \left[ f ( s _ { e , \phi + k } , a _ { e , \phi + k } ) \mid \phi _ { e } = \phi \right] = \ell _ { f , n } ( \phi , k ) . } \end{array}\tag{20}
$$

Starting from the empirical phase-mixture return in Eq. (5) and using linearity of expectation,

$$
\mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] = \mathbb { E } \left[ \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } f ( s _ { e , \phi _ { e } + k } , a _ { e , \phi _ { e } + k } ) \right]\tag{21}
$$

$$
= { \frac { 1 } { E } } \sum _ { \phi = 0 } ^ { H - 1 } \sum _ { e : \phi _ { e } = \phi } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \mathbb { E } \left[ f ( s _ { e , \phi + k } , a _ { e , \phi + k } ) \mid \phi _ { e } = \phi \right]\tag{22}
$$

$$
= \frac { 1 } { E } \sum _ { \phi = 0 } ^ { H - 1 } \sum _ { e : \phi _ { e } = \phi } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \ell _ { f , n } ( \phi , k )\tag{23}
$$

$$
= \sum _ { \phi = 0 } ^ { H - 1 } \frac { E _ { \phi } } { E } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \ell _ { f , n } ( \phi , k ) .\tag{24}
$$

By the definition $q _ { n } ( \phi ) = E _ { \phi } / E$ and the phase return in Eq. (4), the final line equals

$$
\sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) j _ { f , n } ^ { K } ( \phi ) ,\tag{25}
$$

which is Eq. (6).

## A.1.2 Proof of Proposition 3.2

Proof. Lemma 3.1 gives the collected phase-mixture expectation in Eq. (6). Expanding $j _ { f , n } ^ { K } ( \phi )$ by Eq. (4),

$$
\mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \mathbb { E } _ { ( s , a ) \sim \zeta _ { n , \phi , k } } [ f ( s , a ) ] .\tag{26}
$$

The current policy phase-mixture return is Eq. (7), namely

$$
\bar { j } _ { f , n } ^ { K } = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \mathbb { E } _ { ( s , a ) \sim \bar { \zeta } _ { n , \phi , k } } [ f ( s , a ) ] .\tag{27}
$$

Subtracting the two displays and applying the triangle inequality yields

$$
\left| \mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] - \bar { j } _ { f , n } ^ { K } \right| = \left| \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \left( \mathbb { E } _ { ( s , a ) \sim \zeta _ { n , \phi , k } } [ f ( s , a ) ] - \mathbb { E } _ { ( s , a ) \sim \bar { \zeta } _ { n , \phi , k } } [ f ( s , a ) ] \right) \right|\tag{28}
$$

$$
\leq \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \left| \mathbb { E } _ { ( s , a ) \sim \zeta _ { n , \phi , k } } [ f ( s , a ) ] - \mathbb { E } _ { ( s , a ) \sim \bar { \zeta } _ { n , \phi , k } } [ f ( s , a ) ] \right| .\tag{29}
$$

For any measurable f with $| f | \leq f _ { \operatorname* { m a x } }$ , the TV variational inequality gives

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | = \left| \int f d ( P - Q ) \right| \le f _ { \operatorname* { m a x } } \int | d P - d Q | = 2 f _ { \operatorname* { m a x } } \| P - Q \| _ { \mathrm { T V } } .\tag{30}
$$

Applying this inequality with $P = \zeta _ { n , \phi , k }$ and $Q = \bar { \zeta } _ { n , \phi , k }$ for every (ϕ, k) gives Eq. (8).

## A.1.3 Proof of Proposition 3.3

Proof. After selecting one of the two cost surrogates in Eq. (10), write it in the form assumed by the proposition,

$$
\widehat { J } _ { c , n } ^ { K } = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } X _ { e } , \qquad 0 \leq X _ { e } \leq R _ { n } .\tag{31}
$$

The variables $X _ { e }$ are independent across environments under the batch construction. The one-sided empirical Bernstein inequality for independent bounded variables [13] states that, with probability at least $1 - \delta$

$$
\mathbb { E } \left[ \frac { 1 } { E } \sum _ { e = 1 } ^ { E } X _ { e } \right] - \frac { 1 } { E } \sum _ { e = 1 } ^ { E } X _ { e } \leq \sqrt { \frac { 2 S ^ { 2 } \log ( 2 / \delta ) } { E } } + \frac { 7 R _ { n } \log ( 2 / \delta ) } { 3 ( E - 1 ) } ,\tag{32}
$$

where $\begin{array} { r } { S ^ { 2 } = ( E - 1 ) ^ { - 1 } \sum _ { e = 1 } ^ { E } ( X _ { e } - \widehat { J } _ { c , n } ^ { K } ) ^ { 2 } } \end{array}$ . Replacing the sample mean by $\widehat { J } _ { c , n } ^ { K }$ gives Eq. (11). Thus the right-hand side of Eq. (11) is a valid $\beta _ { n }$ term for the selected surrogate. □

## A.1.4 Proof of Theorem 3.4

Proof. Let $\mathcal { E } _ { \beta }$ be the event in Proposition 3.3. On this event, Eq. (11) implies

$$
\mathbb { E } \left[ \widehat { J } _ { c , n } ^ { K } \right] \leq \widehat { J } _ { c , n } ^ { K } + \beta _ { n } .\tag{33}
$$

The theorem assumes the distribution mismatch upper bound

$$
J _ { c } ( \pi _ { n } ) \leq \mathbb { E } \left[ \widehat { J } _ { c , n } ^ { K } \right] + \varepsilon _ { n } .\tag{34}
$$

Combining the two inequalities on $\mathcal { E } _ { \beta }$ ,

$$
J _ { c } ( \pi _ { n } ) \le \mathbb { E } \left[ \widehat { J } _ { c , n } ^ { K } \right] + \varepsilon _ { n }\tag{35}
$$

$$
\leq { \widehat { J } } _ { c , n } ^ { K } + \beta _ { n } + \varepsilon _ { n } .\tag{36}
$$

If the tightened surrogate constraint in Eq. (12) holds, then

$$
J _ { c } ( \pi _ { n } ) \le \widehat { J } _ { c , n } ^ { K } + \beta _ { n } + \varepsilon _ { n } \le b _ { c } .\tag{37}
$$

The event $\mathcal { E } _ { \beta }$ has probability at least $1 - \delta ,$ , so the same probability applies to the final constraint statement.

## A.1.5 Proof of Theorem 3.5

Proof. Uniform staggered reset assigns equal phase mass $q _ { n } ( m K ) = 1 / M$ for $m = 0 , \ldots , M - 1$ and zero mass to other phases. Substituting this phase distribution into the phase-mixture return from Eq. (6) gives

$$
\sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) j _ { f , n } ^ { K } ( \phi ) = \sum _ { m = 0 } ^ { M - 1 } q _ { n } ( m K ) j _ { f , n } ^ { K } ( m K )\tag{38}
$$

$$
= \sum _ { m = 0 } ^ { M - 1 } \frac { 1 } { M } j _ { f , n } ^ { K } ( m K )\tag{39}
$$

$$
= \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } j _ { f , n } ^ { K } ( m K ) .\tag{40}
$$

The final expression is the finite Cesàro mean stated in Theorem 3.5.

## A.1.6 Proof of Corollary 3.6

Proof. Lemma 3.1 gives

$$
\mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) j _ { f , n } ^ { K } ( \phi ) .\tag{41}
$$

Theorem 3.5 evaluates the right-hand side under uniform staggered reset as the Cesàro mean. Therefore,

$$
\mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } j _ { f , n } ^ { K } ( m K ) ,\tag{42}
$$

which is Eq. (13).

## A.1.7 Proof of Corollary 3.7

Proof. Start from the distribution mismatch bound in Eq. (8). Under uniform staggered reset, the only nonzero phase masses are $q _ { n } ( m K ) = 1 / M$ for $m = 0 , \ldots , M - 1$ . Substituting these identities gives

$$
\left| \mathbb { E } \left[ \widehat { j } _ { f , n } ^ { K } \right] - \bar { j } _ { f , n } ^ { K } \right| \leq 2 f _ { \operatorname* { m a x } } \sum _ { m = 0 } ^ { M - 1 } \frac { 1 } { M } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \left. \zeta _ { n , m K , k } - \bar { \zeta } _ { n , m K , k } \right. _ { \mathrm { T V } }\tag{43}
$$

$$
= \frac { 2 f _ { \mathrm { m a x } } } { M } \sum _ { m = 0 } ^ { M - 1 } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { k } \left. \zeta _ { n , m K , k } - \bar { \zeta } _ { n , m K , k } \right. _ { \mathrm { T V } } .\tag{44}
$$

The final expression is Eq. (14).

## A.1.8 Proof of Corollary 3.8

Proof. Begin with the general phase-wise surrogates in Eq. (10). Uniform staggered reset gives $\phi = m K$ and $q _ { n } ( m K ) \bar { = } 1 / M$ . For every time step $t \in \{ \bar { 0 } , \ldots , M K - 1 \}$ , there is a unique pair $( m , k )$ with $t = m K + k$ and $0 \leq k < K$ . Using the coverage mass definition in Eq. (9),

$$
m _ { q _ { n } } ( m K + k ) = \sum _ { \phi = 0 } ^ { H - 1 } q _ { n } ( \phi ) \sum _ { k ^ { \prime } = 0 } ^ { K - 1 } \mathbb { I } \{ m K + k = \phi + k ^ { \prime } \}\tag{45}
$$

$$
= q _ { n } ( m K )\tag{46}
$$

$$
= { \frac { 1 } { M } } .\tag{47}
$$

The coverage-correction weight in Eq. (9) is therefore

$$
{ { w } _ { { { q } _ { n } } } } ( m K , k ) = \frac { \gamma ^ { m K + k } } { { { m } _ { { { q } _ { n } } } } ( m K + k ) } = M \gamma ^ { m K + k } .\tag{48}
$$

For the sampled cost surrogate in Eq. (10), substituting $q _ { n } ( m K ) = 1 / M$ and the preceding weight gives

$$
\widehat { J } _ { c , n } ^ { K , \mathrm { s a m p l e } } = \sum _ { m = 0 } ^ { M - 1 } \frac { 1 } { M } \frac { 1 } { E _ { m K } } \sum _ { e : \phi _ { e } = m K } \sum _ { k = 0 } ^ { K - 1 } M \gamma ^ { m K + k } c ( s _ { e , m K + k } , a _ { e , m K + k } )\tag{49}
$$

$$
= \sum _ { m = 0 } ^ { M - 1 } { \frac { 1 } { E _ { m K } } } \sum _ { e : \phi _ { e } = m K } \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { m K + k } c ( s _ { e , m K + k } , a _ { e , m K + k } ) .\tag{50}
$$

For the critic value surrogate, Eq. (10) has no coverage-correction weight. Substituting only $q _ { n } ( m K ) = 1 / M$ gives

$$
\widehat { J } _ { c , n } ^ { K , \mathrm { v a l u e } } = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } \frac { 1 } { E _ { m K } K } \sum _ { e : \phi _ { e } = m K } \sum _ { k = 0 } ^ { K - 1 } V _ { c , n } ( s _ { e , m K + k } ) .\tag{51}
$$

The two derived expressions are exactly Eq. (15).

Table 2: Goal-task configurations. The cost limit specifies the maximum cumulative episodic cost.
<table><tr><td>Task</td><td>Agent</td><td>Horizon</td><td>Objects</td><td>Cost event</td><td>Limit</td></tr><tr><td>Goal</td><td>Point</td><td>1000</td><td>goal, 8 hazards, 1 vase</td><td>hazard proximity</td><td>25</td></tr><tr><td>Goal</td><td>Car</td><td>1000</td><td>goal, 8 hazards, 1 vase</td><td>hazard proximity</td><td>25</td></tr></table>

## A.1.9 Proof of Proposition 3.9

Proof. For each phase m $K .$ , the change-of-measure identity gives

$$
\begin{array} { r } { \bar { C } _ { n , m K } ^ { K } - C _ { n , m K } ^ { K } = \mathbb { E } _ { s \sim d _ { n , m K } } \left[ ( \bar { \rho } _ { n } ( s , m K ) - 1 ) j _ { c , n , m K } ^ { K } ( s ) \right] . } \end{array}\tag{52}
$$

Taking absolute values, applying the triangle inequality, and using $0 \le j _ { c , n , m K } ^ { K } ( s ) \le B _ { c }$ gives

$$
\begin{array} { r } { \left| \bar { C } _ { n , m K } ^ { K } - C _ { n , m K } ^ { K } \right| \le B _ { c } \mathbb { E } _ { s \sim d _ { n , m K } } \bigl [ | \bar { \rho } _ { n } ( s , m K ) - 1 | \bigr ] . } \end{array}\tag{53}
$$

Applying the triangle inequality to the uniform phase average and then averaging the preceding bound over $m = 0 , \ldots , M - 1$ proves Eq. (18). □

## A.1.10 Proof of Proposition 3.10

Proof. For each phase mK, change of measure gives

$$
\mathbb { E } _ { s \sim \bar { d } _ { n , m K } } [ Y _ { n } ( s , m K ) ] = \mathbb { E } _ { s \sim d _ { n , m K } } [ \bar { \rho } _ { n } ( s , m K ) Y _ { n } ( s , m K ) ] .\tag{54}
$$

Therefore, the difference between the exact-ratio correction and the learned-ratio correction is

$$
\mathbb { E } _ { s \sim d _ { n , m K } } \bigl [ ( \bar { \rho } _ { n } ( s , m K ) - \tilde { \rho } _ { n } ( s , m K ) ) Y _ { n } ( s , m K ) \bigr ] .\tag{55}
$$

Taking absolute values and using $| Y _ { n } ( s , m K ) | \leq B _ { Y }$ gives

$$
\begin{array} { r l } & { \Bigl | \mathbb { E } _ { s \sim \bar { d } _ { n , m K } } [ Y _ { n } ( s , m K ) ] - \mathbb { E } _ { s \sim d _ { n , m K } } [ \tilde { \rho } _ { n } ( s , m K ) Y _ { n } ( s , m K ) ] \Bigr | } \\ & { \qquad \le B _ { Y } \mathbb { E } _ { s \sim d _ { n , m K } } [ | \bar { \rho } _ { n } ( s , m K ) - \tilde { \rho } _ { n } ( s , m K ) | ] . } \end{array}\tag{56}
$$

Applying the triangle inequality to the uniform phase average and then averaging the preceding bound over $m = 0 , \ldots , M - 1$ gives the result. □

## A.2 Implementation details

We implement the Safety Gym-style Goal tasks in MJX and the Laikago locomotion task in both MJX and MuJoCo Warp to support massively parallel training on accelerators.

Goal Tasks Following Safety Gym and Safety-Gymnasium [9, 15], we implement the Point/Goal and Car/Goal tasks with MuJoCo agent models, goals, hazards, and vases. The environments are vectorized in MJX, with rewards, costs, and observations computed at each simulation step.

Both agents use two-dimensional continuous action vectors, with 12-dimensional and 24-dimensional sensor vectors for Point and Car, respectively. Observations combine these sensor vectors with three 16-bin pseudo-lidar groups for the goal, hazards, and vase, resulting in observation dimensions of 60 and 72, respectively.

Let $p _ { t }$ and $g _ { t }$ denote the agent and goal positions, respectively, and let $d _ { t } = \| p _ { t } - g _ { t } \| _ { 2 }$ . The reward is defined as

$$
r _ { t } = d _ { t - 1 } - d _ { t } + \mathbb { I } \{ d _ { t } < r _ { \mathrm { g o a l } } \} ,\tag{57}
$$

where $r _ { \mathrm { g o a l } }$ is the goal radius. The reward consists of progress toward the goal and a unit reward upon reaching it, after which a new goal is sampled. The binary safety cost is defined as

$$
c _ { t } = \mathbb { I } \{ \operatorname* { m i n } _ { i } \| p _ { t } - h _ { i } \| _ { 2 } < r _ { \mathrm { h a z a r d } } \} ,\tag{58}
$$

where $h _ { i }$ denotes the position of hazard $i$ and $r _ { \mathrm { h a z a r d } }$ is the hazard radius. Both tasks use a single safety constraint with an episodic cost limit of 25.

Table 3: Algorithm configuration. Rows with shared values span all three algorithms.
<table><tr><td>Config CPO</td><td>CRPO TRPO-Lag</td></tr><tr><td>Critic learning rate</td><td>10-3</td></tr><tr><td>Discount factor</td><td>0.99</td></tr><tr><td>Reward trace parameter</td><td>1.0</td></tr><tr><td>Cost trace parameter</td><td>1.0</td></tr><tr><td>Target KL</td><td>0.01</td></tr><tr><td>CG iterations</td><td>15</td></tr><tr><td>CG damping</td><td>0.1</td></tr><tr><td>Line search coefficient</td><td>0.8</td></tr><tr><td>Rollout mode</td><td>staggered</td></tr><tr><td>Rollout length</td><td>K = 10</td></tr><tr><td>Policy updates</td><td>500</td></tr><tr><td>Cost limit</td><td>25</td></tr><tr><td>Tightening confidence level</td><td>0.05</td></tr><tr><td>Tightening ratio clip</td><td>3.0</td></tr><tr><td>Hidden layers</td><td>(64, 64)</td></tr><tr><td>Activation</td><td>tanh</td></tr><tr><td>Observation normalization</td><td>true</td></tr><tr><td>Lagrange optimizer</td><td>Adam</td></tr><tr><td>Lagrange learning rate</td><td>0.035</td></tr><tr><td>Initial Lagrange multiplier</td><td>0.001</td></tr></table>

Locomotion Task We adopt the Laikago locomotion task from [10] and implement it in both MJX and MuJoCo Warp. The robot tracks velocity commands on flat terrain. At the beginning of each episode, the forward and angular velocity commands are sampled as

$$
v _ { x } ^ { \mathrm { c m d } } \sim \mathcal { U } [ - 1 , 2 ] , \quad \omega _ { z } ^ { \mathrm { c m d } } \sim \mathcal { U } [ - 0 . 5 , 0 . 5 ] , \quad v _ { y } ^ { \mathrm { c m d } } = 0 .\tag{59}
$$

The reward penalizes squared tracking errors between commanded and actual base velocities, with an additional power regularization term.

The task uses three costs for balance, center-of-mass (CoM) height, and foot contact timing:

$$
\begin{array} { r l r } & { c _ { \mathrm { b a l } , t } = \mathbb { I } \{ \theta _ { t } > 1 5 ^ { \circ } \} , } & \\ & { c _ { \mathrm { h g t } , t } = \mathbb { I } \{ h _ { t } < 0 . 3 5 \} , } & \\ & { c _ { \mathrm { f o o t } , t } = \displaystyle \frac { 1 } { n _ { \mathrm { l e g s } } } \sum _ { i = 1 } ^ { n _ { \mathrm { l e g s } } } \mathbb { I } \{ \hat { \xi } _ { t , i } \neq \xi _ { t , i } \} . } & \end{array}\tag{60}
$$

Here, $\theta _ { t }$ is the angle between the robot base and world vertical axes, $h _ { t }$ is the CoM height, and $\hat { \xi } _ { t , i }$ and $\xi _ { t , i }$ denote the measured and desired contact states of leg i, respectively. The desired contact pattern follows per-leg phase variables with a fixed frequency. The foot timing constraint uses a looser limit than the balance and height constraints, as it primarily serves as a motion regularizer.

## B Ablation studies

This appendix reports the Goal-task ablations that isolate the effects of staggered reset, rollout length, and constraint-tightening terms. All entries report the mean and standard deviation of the last 10 epochs over eight seeds unless otherwise specified. SC denotes the sampled cost estimate, CV denotes the critic value estimate.

## B.1 Rollout length

To examine the trade-off between rollout length and environment parallelism in massively parallel safe RL, Table 4 compares increasing batch size through longer rollouts with the environment scaling results reported in Table 1.

Both approaches tend to achieve higher rewards as batch size increases, although the cost limit remains violated without tightening. Once tightening is applied, all configurations satisfy the constraint, with increasing environment parallelism achieving higher rewards in most cases than extending rollout length. This difference is consistent with the larger number of samples collected per phase, which reduces the finite-sample bound and allows less conservative tightening. Moreover, the higher transition throughput obtained through greater parallelism reduces wall-clock time, making short-rollout training more effective in terms of both reward performance and training efficiency.

Table 4: Comparison of increasing rollout length and environment scale at the same batch size using CPO with the SC surrogate. Values are mean ± standard deviation of the last 10 updates over eight seeds. Shading indicates cost > 25. Bold highlights the highest feasible reward and its corresponding cost and time for each task and batch size.
<table><tr><td rowspan="3"></td><td rowspan="3">Task Batch size</td><td rowspan="3"></td><td colspan="6">Fixed envs (2k)</td><td rowspan="2"></td><td colspan="7">Fixed rollout (10)</td></tr><tr><td colspan="3">w/o tightening</td><td colspan="3">w/ tightening</td><td rowspan="2">Envs</td><td colspan="3">w/o tightening</td><td colspan="3">w/ tightening</td></tr><tr><td>Reward</td><td>Cost</td><td>Time</td><td>Reward</td><td>Cost</td><td>Time</td><td>Reward</td><td>Cost</td><td></td><td>Time</td><td>Reward</td><td>Cost Time</td></tr><tr><td rowspan="4">Poinoal</td><td rowspan="2">20k 40k</td><td>10</td><td>14.57 ±2.78</td><td>27.08 ±0.84</td><td>4.10</td><td>7.56 ±3.06</td><td>13.88 ±1.59</td><td>4.09</td><td>2k</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>20</td><td>17.96 ±1.12</td><td>27.05 ±1.10</td><td>7.77</td><td>13.96 ±1.41</td><td>15.05 ±0.63</td><td>7.88</td><td>4k</td><td>17.28 ±5.37</td><td>26.27 ±1.23</td><td>6.84</td><td>13.42 ±3.52</td><td>20.00 ±3.49</td><td>7.09</td></tr><tr><td>80k</td><td>40</td><td>20.46 ±0.48</td><td>27.18 ±0.91</td><td>15.91</td><td>16.43 ±0.81</td><td>16.36 ±0.39</td><td>16.20</td><td>21.23 8k ±1.05</td><td></td><td>27.02 ±1.50</td><td>12.87</td><td>18.83 ±0.73</td><td>21.19 ±2.05</td><td>13.03</td></tr><tr><td>20k</td><td></td><td>25.12</td><td>28.63</td><td></td><td>18.52</td><td>14.38</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Caro0al</td><td rowspan="4">40k</td><td rowspan="2">10</td><td>±2.11</td><td>±1.46</td><td>12.92</td><td>±2.83</td><td>±1.18</td><td>13.03</td><td>2k</td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>29.11</td><td>26.45</td><td></td><td>22.29</td><td>15.49</td><td></td><td></td><td>27.57</td><td>26.87</td><td></td><td>24.38</td><td>18.43</td><td></td></tr><tr><td>20</td><td>±1.69</td><td>±0.77</td><td>25.64</td><td>±2.90</td><td>±0.66</td><td>25.75</td><td>4k</td><td>±1.76</td><td>±1.18</td><td>24.82</td><td>±2.04</td><td>±0.75</td><td>24.96</td></tr><tr><td>40</td><td>32.55 ±0.52</td><td>26.54 ±0.74</td><td>51.22</td><td>26.76 ±1.16</td><td>16.69 ±0.70</td><td>51.32</td><td>8k</td><td>32.17 ±0.95</td><td>26.52 ±1.38</td><td>50.24</td><td>29.57 ±0.70</td><td>21.06</td><td>50.53</td></tr></table>

## B.2 Tightening term ablation

To examine the contribution of each tightening term to constraint satisfaction, Table 5 compares CPO with the finite-sample term β and the phase-initial distribution correction term ε, applied individually and jointly to the SC and CV surrogates across environment scales from 2k to 32k.

The results show that applying the finite-sample term β accounts for most of the improvement in constraint satisfaction, whereas ε alone has little effect on cost returns. For the SC surrogate, β satisfies the constraint across all environment scales, and combining both terms provides further cost reductions in some configurations. However, the CV surrogate continues to violate the constraint in several configurations even with both terms, since critic approximation errors are not covered by the finite-sample bound. The limited effect of ε in these experiments also indicates that further evaluation on tasks with greater phase-initial distribution mismatch is needed to assess the contribution of distribution correction.

Table 5: Ablation of finite-sample tightening (β) and phase-initial distribution correction (ε) across environment scales using SC and CV surrogates. Values are mean ± standard deviation of the last 10 epochs over eight seeds. Shading indicates cost > 25.
<table><tr><td rowspan="2"></td><td rowspan="2">Cost est. Envs</td><td colspan="2">No tightening</td><td colspan="2">ε only</td><td colspan="2">β only</td><td colspan="2">Both (ε + β)</td></tr><tr><td>Reward</td><td>Cost</td><td>Reward</td><td>Cost</td><td>Reward</td><td>Cost</td><td>Reward</td><td>Cost</td></tr><tr><td rowspan="5">SC</td><td>2k</td><td>14.57 ±2.78</td><td>27.08 ±0.84</td><td>12.87 ±3.83</td><td>25.77 ±1.87</td><td>9.55 ±4.08</td><td>17.43 ±2.39</td><td>7.56 ±3.06</td><td> $1 3 . 8 8 \pm 1 . 5 9$ </td></tr><tr><td>4k</td><td>17.28 ±5.37</td><td>26.27 ±1.23</td><td>16.44 ±3.21</td><td>27.10 ±2.27</td><td>12.71 ±1.99</td><td>18.19 ±1.91</td><td>13.42 ±3.52</td><td> $2 0 . 0 0 \pm 3 . 4 9$ </td></tr><tr><td>8k</td><td>21.23 ±1.05</td><td>27.02 ±1.50</td><td>21.27 ±0.69</td><td>27.02 ±1.62</td><td>18.68 ±1.43</td><td>22.69 ±1.70</td><td>18.83 ±0.73</td><td>21.19 ±2.05</td></tr><tr><td>16k</td><td>23.07 ±0.46</td><td>25.73 ±0.84</td><td>23.09 ±0.48</td><td>25.72 ±1.20</td><td>22.09 ±0.55</td><td>21.86 ±1.02</td><td>22.30 ±0.68</td><td>22.66 ±1.41</td></tr><tr><td>32k</td><td>23.81 ±0.64</td><td>25.07 ±0.33</td><td>24.37 ±0.54</td><td>25.88 ±0.74</td><td>23.48 ±0.70</td><td>23.19 ±0.42</td><td>23.43 ±0.70</td><td>23.06 ±0.59</td></tr><tr><td rowspan="5">CV</td><td>2k</td><td>15.29 ±3.76</td><td>29.55 ±3.43</td><td>16.41 ±2.95</td><td>26.62 ±1.28</td><td>13.49 ±2.16</td><td>24.60 ±5.18</td><td>13.42 ±2.06</td><td>24.53 ±3.93</td></tr><tr><td>4k</td><td>19.30 ±2.27</td><td>29.61 ±3.04</td><td>19.91 ±1.09</td><td>28.82 ±2.96</td><td>17.10 ±1.24</td><td>23.31 ±1.66</td><td>17.40 ±2.23</td><td>24.57 ±2.98</td></tr><tr><td>8k</td><td>21.46 ±1.17</td><td>27.82 ±2.25</td><td>21.71 ±1.58</td><td>27.16 ±2.66</td><td>20.43 ±2.75</td><td>26.16 ±2.34</td><td>20.55 ±2.29</td><td>26.91 ±3.21</td></tr><tr><td>16k</td><td>23.10 ±0.87</td><td>26.78 ±2.76</td><td>23.38 ±0.59</td><td>26.07 ±1.56</td><td>23.05 ±0.81</td><td>25.55 ±1.69</td><td>22.79 ±1.18</td><td>23.94 ±2.26</td></tr><tr><td>32k</td><td>24.82 ±0.36</td><td>26.82 ±1.42</td><td>24.39 ±0.73</td><td>25.37 ±2.50</td><td>24.42 ±0.47</td><td>25.28 ±1.84</td><td>24.41 ±0.86</td><td>24.92 ±1.61</td></tr></table>