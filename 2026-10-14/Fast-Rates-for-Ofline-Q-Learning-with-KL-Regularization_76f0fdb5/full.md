# Fast Rates for Ofline Q-Learning with KL Regularization

Yize Hao<sup>1,2</sup> Guangxin Jiang<sup>3</sup> Shuang Qiu<sup>†4</sup>

October 14, 2026

## Abstract

This paper investigates ofline Q-learning for finite-horizon Markov decision processes with Kullback-Leibler (KL) regularization. Recent studies have shown that KL regularization can yield fast rates in ofline decision-making problems. However, it remains unclear whether Q-learning, which adopts incremental stochastic approximation updates, can achieve such fast rates in the ofline setting. We first propose KL-regularized Q-Learning (KLQ), a simple variant of classical Q-learning. By exploiting KL regularization, we prove that KLQ achieves a sample complexity of Oe(1/ϵ) without explicit pessimism. This guarantee, however, has an exponential dependence on the KL regularization parameter and the horizon. To address this limitation, we further propose KL-regularized Pessimistic Q-Learning (KLPQ), which incorporates pessimism into KLQ. We prove that KLPQ retains the same Oe(1/ϵ) rate while removing the explicit exponential dependence, thereby further improving the sample complexity. To the best of our knowledge, these results establish the first fast-rate guarantees for ofline Q-learning in tabular finite-horizon KL-regularized MDPs.

## 1 Introduction

Ofline reinforcement learning (RL) aims to learn a near-optimal policy from a fixed dataset without further interaction with the environment (Levine et al., 2020). A central challenge in this setting is limited data coverage: since the learner cannot collect additional samples, the learned policy may select actions that are poorly represented in the ofline data, leading to inaccurate value estimates and extrapolation error (Fujimoto et al., 2019; Kumar et al., 2020). Two prominent approaches used to stabilize ofline learning are pessimism and policy regularization. Pessimistic methods penalize value estimates in poorly covered regions to mitigate estimation errors caused by insuficient data (Rashidinejad et al., 2021; Jin et al., 2021; Shi et al., 2022). Policy regularization instead constrains the learned policy relative to a reference policy, with the Kullback–Leibler (KL) divergence being a widely used choice. When the reference policy coincides with the behavior policy that generates the ofline data, KL regularization serves as behavior regularization, keeping the learned policy close to the behavior policy and thereby reducing distribution shift and extrapolation error (Wu et al., 2019). The reference policy, however, may difer from the behavior policy. In this case, KL regularization no longer directly constrains the learned policy toward the behavior policy. Nevertheless, Chen et al. (2026) show that KL regularization with respect to a fixed reference policy can stabilize ofline learning in Markov games under unilateral concentrability. The practical importance of regularization is also reflected in reinforcement learning from human feedback and preference-based learning, where KL regularization is widely used to keep the learned policy close to a reference model (Ouyang et al., 2022; Xiong et al., 2023; Rafailov et al., 2023). These developments motivate a deeper understanding of the statistical role of KL regularization in ofline learning.

Recent theoretical work has revealed that KL regularization can substantially change the statistical behavior of ofline learning. In single-step decision problems, fast rates have been established for KL-regularized contextual and multi-armed bandits (Zhao et al., 2025a; 2026b; Ji et al., 2026a). Related results have also been established under forward-KL regularization (Zhao et al., 2026a). With KL regularization, these works achieve a sample complexity of $\widetilde { \cal O } ( 1 / \epsilon )$ , in contrast to the $\widetilde { \cal O } ( 1 / \epsilon ^ { 2 } )$ sample complexity typically obtained in standard unregularized settings. Fast rates have also been established for single-step zero-sum games (Zhang et al., 2026) and generalsum games (Chen & Zhang, 2026b). More recently, these results have been extended to multi-step sequential decision problems. In particular, Chen et al. (2026) establish a sample complexity of $\widetilde { \cal O } ( 1 / \epsilon )$ for ofline two-player zero-sum Markov games with KL regularization. From an algorithmic perspective, the methods of Chen et al. (2026) can be regarded as fitted Q-iteration. Their approach proceeds via backward induction, where Q-functions are estimated stagewise from the end of the horizon to the beginning. In contrast, another important class of algorithms is Q-learning, which adopts incremental stochastic approximation updates to solve the Bellman optimality equation (Watkins & Dayan, 1992). Understanding the statistical eficiency and convergence behavior of Q-learning remains an important topic in the theory of reinforcement learning (Li et al., 2024; Liu et al., 2025). The resulting update scheme difers fundamentally from backward induction, and existing fast-rate analyses based on backward induction do not establish analogous guarantees for Q-learning. In the ofline RL setting, Shi et al. (2022) established finite-sample guarantees for pessimistic Q-learning under the standard unregularized objective and single-policy concentrability. Nevertheless, whether ofline Q-learning can achieve fast rates for the KL-regularized objective remains unclear. This leads to the following question:

## Can ofline Q-learning achieve fast rates with KL regularization?

The above question poses three main challenges. First, although fast rates have been established for KL-regularized ofline learning in several settings, the incremental updates of Q-learning pose additional challenges for finite-sample analysis, making it unclear how to exploit KL regularization to achieve similar fast rates. Second, existing finite-sample guarantees for ofline Q-learning are established for the unregularized objective and rely on the principle of pessimism. It remains unclear whether pessimism is still necessary with KL regularization, or whether KL regularization alone can enable Q-learning to achieve fast rates. Third, the impact of pessimism on sample complexity under KL regularization remains underexplored for ofline Q-learning. In this paper, we address these challenges and therefore answer this question afirmatively by developing two ofline Q-learning algorithms for tabular finite-horizon KL-regularized MDPs. For an MDP with S states and A actions, let $\eta > 0$ denote the KL regularization parameter and let $C _ { \eta } ^ { \star }$ denote the single-policy concentrability coeficient for the corresponding KL-regularized optimal policy. Given K ofline trajectories of horizon H, our main contributions are summarized as follows.

• We first introduce KL-regularized Q-Learning (KLQ), a simple variant of classical Q-learning without the principle of pessimism. We prove that KLQ achieves a sample complexity of $\widetilde { \cal O } \Bigl ( \eta e ^ { \eta H ( H + 1 ) } H ^ { 8 } S A C _ { \eta } ^ { \star } / \epsilon \Bigr )$ for finding an ϵ-optimal policy. To the best of our knowledge, this is the first $\widetilde { O } ( 1 / \epsilon )$ sample complexity guarantee for Q-learning in an ofline finite-horizon tabular setting. This result also demonstrates that pessimism is not necessary for achieving fast rates.

• Based on KLQ, we propose KL-regularized Pessimistic Q-Learning (KLPQ), which further incorporates pessimism into KL-regularized Q-Learning. We prove that KLPQ achieves an improved sample complexity of $\widetilde { \cal O } \left( \eta H ^ { 7 } S A C _ { \eta } ^ { \star } / \epsilon \right)$ . This removes KLQ’s explicit exponential dependence on the regularization parameter and horizon while retaining the fast rate.

• On the technical side, we exploit the structure induced by KL regularization to establish a fast rate analysis of incremental Q-learning updates through a recursion for the squared Q-estimation error. We further show that pessimism sharpens both this error recursion and the conversion from Q-estimation error to policy suboptimality.

## 1.1 Related Work

Ofline RL. A large body of theoretical work studies ofline RL under coverage conditions, with pessimism playing an important role in controlling estimation errors caused by insuficient data coverage (Rashidinejad et al., 2021; Xie et al., 2021b; Jin et al., 2021; Shi et al., 2022; Xie et al., 2021a; Yan et al., 2023; Qiao et al., 2026). This principle has been developed for both model-based and model-free methods (Rashidinejad et al., 2021; Xie et al., 2021b; Uehara & Sun, 2021; Shi et al., 2022; Zhang et al., 2024). Beyond the tabular setting, finite-sample guarantees have also been established with function approximation under partial coverage assumptions (Zhan et al., 2022; Xiong et al., 2022; Ye et al., 2023; Uehara et al., 2023).

Fast Rates in KL-Regularized Learning. KL regularization can fundamentally change the statistical behavior of learning problems. In the online setting, for single-step problems, fast rates have been established for KL-regularized contextual bandits and RLHF (Zhao et al., 2025a), while logarithmic regret has also been established for KL-regularized multi-armed bandits (Ji et al., 2026b). For multi-step tabular problems, fast rates have been established for maximum entropy exploration (Tiapkin et al., 2023) and sequential RL with demonstration regularization (Tiapkin et al., 2024). In particular, the entropy regularization considered by Tiapkin et al. (2023) is equivalent to KL regularization with a uniform reference policy, up to an additive constant. Zhao et al. (2025b) establish logarithmic regret for KL-regularized episodic RL, while logarithmic regret has also been established for KL-regularized zero-sum Markov games (Nayak et al., 2025). Beyond the tabular setting, Hong et al. (2026) study KL-regularized RL with function approximation under misspecification. In ofline learning, for single-step problems, fast rates are known for KL-regularized contextual and multi-armed bandits (Zhao et al., 2026b; Ji et al., 2026a; Zhao et al., 2026a), as wel as preference learning and games (Wu et al., 2025; Zhang et al., 2026; Chen & Zhang, 2026b). The role of pessimism has also been studied in this setting. Ji et al. (2026a) establish matching upper and lower bounds across diferent regularization regimes for ofline KL-regularized multi-armed bandits. Their algorithm constructs a pessimistic reward estimate and achieves the optimal sample complexity in both regimes. In contrast, Wu et al. (2025), Zhang et al. (2026), and Chen & Zhang (2026b) obtain fast rates without explicit pessimistic estimates in their respective settings. For multi-step decision problems, Chen et al. (2026) establish a sample complexity of $\widetilde { \cal O } ( 1 / \epsilon )$ for ofline Markov games with KL regularization. Their theoretical construction uses backward induction and is therefore diferent from the incremental Q-learning updates considered in this work.

Q-Learning. Q-learning is a classical model-free method for solving the Bellman optimality equation (Watkins & Dayan, 1992; Even-Dar & Mansour, 2003). Under the standard unregularized objective, its non-asymptotic theory has been developed in several settings. For finite-horizon episodic MDPs, regret guarantees have been established for online Q-learning (Jin et al., 2018; Li et al., 2021b), while finite-sample guarantees have also been established for ofline Q-learning under single-policy concentrability (Shi et al., 2022). For discounted MDPs, Q-learning has been studied under a generative model (Li et al., 2021a; 2024), with Markovian samples (Qu & Wierman, 2020; Li et al., 2020), and in the ofline setting under partial coverage (Yan et al., 2023). These results typically exhibit an $\widetilde O ( \epsilon ^ { - 2 } )$ dependence on accuracy for Q-function estimation or policy suboptimality. With linear function approximation, Liu et al. (2025) establish $L ^ { 2 }$ convergence rates of Q-learning to a bounded set. Finite-sample analyses have also been developed for average-reward Q-learning, under span contraction conditions (Zhang et al., 2021) and, more recently, for weakly communicating MDPs (Jiao et al., 2026). For regularized objectives, Uehara et al. (2023) study ofline learning under partial coverage using soft Q-functions, but estimate the Q-functions through minimax optimization. Jeong & Lee (2024; 2025) provide finite-time analyses of soft Q-learning in discounted tabular MDPs under full state-action coverage, while Xi et al. (2025) study regularized Q-learning with linear function approximation using an algorithm with two time scales. These works do not address ofline Q-learning in finite-horizon MDPs.

## 1.2 Notation

For a positive integer n, we write $[ n ] : = \{ 1 , \dots , n \}$ . For a finite set $x ,$ let $\Delta ( \mathcal { X } )$ denote the probability simplex over X. For functions $f , g : \mathcal { X }  \mathbb { R }$ , we write $\langle f , g \rangle _ { \mathcal { X } } : = \sum _ { x \in \mathcal { X } } f ( x ) g ( x )$ , and omit the subscript when the underlying set is clear from context. For $p \in \Delta ( \mathcal X )$ , we denote its support by supp $( p ) : = \{ x \in \mathcal { X } : p ( x ) > 0 \}$ . For a function $f : \mathcal { X }  \mathbb { R }$ and $p \in \Delta ( \mathcal X )$ , we use $\begin{array} { r } { \mathbb { E } _ { x \sim p } [ f ( x ) ] : = \sum _ { x \in \mathcal { X } } p ( x ) f ( x ) } \end{array}$ when X is finite. We use $\mathbf { 1 } \{ \mathcal { E } \}$ for the indicator of an event $\mathcal { E }$ , and write $x \vee y : = \operatorname* { m a x } \{ x , y \}$ for scalars. Inequalities between vectors or functions are understood pointwise unless otherwise stated. We use $\widetilde O ( f )$ to suppress factors polylogarithmic in $f ,$ other relevant problem parameters, and the inverse failure probability.

## 2 Problem Setup

## 2.1 Finite-Horizon KL-Regularized MDPs

Markov Decision Process. We consider a finite-horizon episodic Markov decision process (MDP) denoted by the tuple $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , H , \rho , \mathbb { P } , r )$ , where S and A are finite state and action sets, H is the episode length, and $\rho \in \Delta ( \mathcal { S } )$ is the initial state distribution. We write $S : = | S | , A : = | A |$ , and $L _ { h } : = H - h + 1$ with $L _ { H + 1 } : = 0$ . The transition kernel is $\mathbb { P } : = \{ \mathbb { P } _ { h } \} _ { h = 1 } ^ { H }$ , where $\mathbb { P } _ { h } : { \mathcal { S } } \times { \mathcal { A } }  \Delta ( { \mathcal { S } } )$ , and the deterministic reward function is $r : = \{ r _ { h } \} _ { h = 1 } ^ { H } ,$ where $r _ { h } : S \times A  [ 0 , 1 ]$ . We write $\mathbb { P } _ { h , s , a } : = \mathbb { P } _ { h } ( . ~ |$ $s , a )$ . For $f : S  \mathbb { R }$ , the transition operator is $\begin{array} { r } { ( \mathbb { P } _ { h } f ) ( s , a ) : = \langle \mathbb { P } _ { h , s , a } , f \rangle _ { \mathcal { S } } = \sum _ { s ^ { \prime } \in \mathcal { S } } \mathbb { P } _ { h } ( s ^ { \prime } \mid s , a ) f ( s ^ { \prime } ) } \end{array}$ A Markov policy is $\pi : = \{ \pi _ { h } \} _ { h = 1 } ^ { H }$ with $\pi _ { h } ( \cdot \mid s ) \in \Delta ( { \mathcal { A } } )$ . Starting from $s _ { 1 } \sim \rho _ { ; }$ , a trajectory is generated according to $a _ { h } \sim \pi _ { h } ( \cdot \mid s _ { h } )$ and $s _ { h + 1 } \sim \mathbb { P } _ { h } ( \cdot \mid s _ { h } , a _ { h } )$ . We denote the state occupancy by $d _ { h } ^ { \pi } ( s ) : = \operatorname* { P r } _ { \rho , \pi , \mathbb { P } } ( s _ { h } = s )$ for $h \in [ H + 1 ]$ , and the state-action occupancy by $d _ { h } ^ { \pi } ( s , a ) : = d _ { h } ^ { \pi } ( s ) \pi _ { h } ( a \mid s )$ for $h \in [ H ]$ . Accordingly, $d _ { 1 } ^ { \pi } ( s ) = \rho ( s )$ and $\begin{array} { r } { d _ { h + 1 } ^ { \pi } ( s ^ { \prime } ) = \sum _ { s , a } d _ { h } ^ { \pi } ( s , a ) \mathbb { P } _ { h } ( s ^ { \prime } \mid s , a ) } \end{array}$ for $h \in [ H ]$

KL-regularized Value Functions. We now incorporate KL regularization into the MDP defined above. Following the standard framework of regularized MDPs (Geist et al., 2019), fix a reference policy $\pi ^ { \mathrm { r e f } } : = \{ \pi _ { h } ^ { \mathrm { r e f } } \} _ { h = 1 } ^ { H }$ and a regularization parameter $\eta > 0$ . The coeficient $1 / \eta$ determines the strength of the KL regularization with respect to the reference policy. For any policy π, we define its KL-regularized value function at step h by

$$
V _ { h } ^ { \pi } ( s ) : = \mathbb { E } _ { \pi , \mathbb { P } } \left[ \sum _ { t = h } ^ { H } \left( r _ { t } ( s _ { t } , a _ { t } ) - \frac { 1 } { \eta } \log \frac { \pi _ { t } ( a _ { t } \mid s _ { t } ) } { \pi _ { t } ^ { \mathrm { r e f } } ( a _ { t } \mid s _ { t } ) } \right) \Bigg | s _ { h } = s \right] .
$$

Set $V _ { H + 1 } ^ { \pi } \equiv 0$ and define the corresponding Q-function by

$$
\begin{array} { r } { Q _ { h } ^ { \pi } ( s , a ) : = r _ { h } ( s , a ) + ( \mathbb { P } _ { h } V _ { h + 1 } ^ { \pi } ) ( s , a ) . } \end{array}
$$

We use the standard convention that $\mathrm { K L } ( p | | q ) = + \infty$ whenever $p$ is not absolutely continuous with respect to $q ,$ and otherwise define $\begin{array} { r } { \mathrm { K L } ( p | | q ) : = \sum _ { a } p ( a ) \log \frac { p ( a ) } { q ( a ) } } \end{array}$ , with $0 \log 0 : = 0$ and $0 \log ( 0 / 0 ) : = 0$ Under this convention, the corresponding value functions $\dot { V } _ { h } ^ { \pi } ( s )$ and $Q _ { h } ^ { \pi } ( s , a )$ may take the value $- \infty$ . Throughout the paper, we adopt the convention $0 \cdot ( \pm \infty ) = 0$ . These functions satisfy the regularized Bellman consistency identity

$$
V _ { h } ^ { \pi } ( s ) = \langle \pi _ { h } ( \cdot \mid s ) , Q _ { h } ^ { \pi } ( s , \cdot ) \rangle _ { \cal A } - \frac { 1 } { \eta } \mathrm { K L } \left( \pi _ { h } ( \cdot \mid s ) \Big \Vert \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \right) .
$$

Regularized Bellman Optimality. For $Q _ { h } ( s , \cdot ) \in \mathbb { R } ^ { A }$ , define the regularized Bellman operator and Gibbs map by

$$
F _ { h , s } ( Q _ { h } ( s , \cdot ) ) : = \frac { 1 } { \eta } \log \left. \pi _ { h } ^ { \mathrm { r e f } } ( \cdot  { | } s ) , e ^ { \eta Q _ { h } ( s , \cdot ) } \right. _ { \cal A } ,\tag{1}
$$

$$
G _ { h , s } ( Q _ { h } ( s , \cdot ) ) ( a ) : = \frac { \pi _ { h } ^ { \mathrm { r e f } } ( a  { | } s ) e ^ { \eta Q _ { h } ( s , a ) } } { \sum _ { a ^ { \prime } \in \mathcal { A } } \pi _ { h } ^ { \mathrm { r e f } } ( a ^ { \prime }  { | } s ) e ^ { \eta Q _ { h } ( s , a ^ { \prime } ) } } .\tag{2}
$$

The Gibbs variational identity is

$$
F _ { h , s } ( Q _ { h } ( s , \cdot ) ) = \operatorname* { m a x } _ { \pi _ { h } ( \cdot \vert s ) \in \Delta ( { \mathcal A } ) } \Bigg \{ \langle \pi _ { h } ( \cdot \vert s ) , Q _ { h } ( s , \cdot ) \rangle _ { \mathcal { A } } - \frac { 1 } { \eta } \mathrm { K L } \left( \pi _ { h } ( \cdot \vert s ) \Big \Vert \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \vert s ) \right) \Bigg \} ,
$$

with unique maximizer $G _ { h , s } ( Q _ { h } ( s , \cdot ) )$ . For every $h , s .$ , define the optimal value function by $V _ { h } ^ { \star } ( s ) : =$ $\operatorname { s u p } _ { \pi \in \Pi } V _ { h } ^ { \pi } ( s )$ , where Π denotes the class of Markov policies, and let $V _ { H + 1 } ^ { \star } \equiv 0$ . These results follow from the standard characterization of KL-regularized Bellman optimality (Geist et al., 2019; Zhao et al., 2025b).

Proposition 2.1. There exists a KL-regularized optimal policy $\pi ^ { \star }$ , whose value and Q-functions satisfy, for every $( h , s , a )$

$$
Q _ { h } ^ { \star } ( s , a ) = r _ { h } ( s , a ) + ( \mathbb { P } _ { h } V _ { h + 1 } ^ { \star } ) ( s , a ) , \qquad V _ { h } ^ { \star } ( s ) = F _ { h , s } ( Q _ { h } ^ { \star } ( s , \cdot ) ) ,
$$

with the optimal policy given by $\pi _ { h } ^ { \star } ( \cdot \mid s ) = G _ { h , s } ( Q _ { h } ^ { \star } ( s , \cdot ) )$

Moreover, $V _ { h } ^ { \pi } ( s ) \leq V _ { h } ^ { \star } ( s )$ and $Q _ { h } ^ { \pi } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a )$ for every policy $\pi ,$ and the value ranges satisfy

$$
0 \leq V _ { h } ^ { \star } ( s ) \leq L _ { h } , \qquad 0 \leq Q _ { h } ^ { \star } ( s , a ) \leq L _ { h } .\tag{3}
$$

## 2.2 Ofline Learning and Data Coverage

Ofline Data. Let $\mu : = \{ \mu _ { h } \} _ { h = 1 } ^ { H }$ be a behavior policy. The ofline dataset $D _ { \mu } : = \{ \tau ^ { k } \} _ { k = 1 } ^ { K }$ consists of K independent trajectories $\tau ^ { k } = ( s _ { 1 } ^ { k } , a _ { 1 } ^ { k } , r _ { 1 } ^ { k } , s _ { 2 } ^ { k } , \dot { \ldots } , s _ { H } ^ { k } , a _ { H } ^ { k } , r _ { H } ^ { k } , s _ { H + 1 } ^ { k } )$ , where $s _ { 1 } ^ { k } \sim \rho , a _ { h } ^ { k } \sim \mu _ { h } ( \cdot \mid s _ { h } ^ { k } )$ $s _ { h + 1 } ^ { k } \sim \mathbb { P } _ { h } ( \cdot \mid s _ { h } ^ { k } , a _ { h } ^ { k } )$ , and $r _ { h } ^ { k } = r _ { h } ( s _ { h } ^ { k } , a _ { h } ^ { k } )$ . The total number of transitions is $T = K H$ . Since no further interaction with the environment is allowed, a finite-sample guarantee requires the ofline data to provide suficient coverage of the state-action pairs relevant to the target policy. We formalize this requirement through single-policy concentrability.

Single-policy Concentrability. Following prior analyses of ofline RL under single-policy concentrability (Rashidinejad et al., 2021; Xie et al., 2021b; Shi et al., 2022), we impose the following coverage assumption. It compares the occupancy measure of the KL-regularized optimal policy with that induced by the behavior policy. The coeficient $C _ { \eta } ^ { \star }$ therefore measures how well the behavior policy covers the KL-regularized optimal policy $\pi ^ { \star }$ , where the subscript η reflects the dependence of $\pi ^ { \star }$ and its occupancy measure on the regularization strength.

Assumption 2.2 (Single-policy concentrability). The behavior policy µ satisfies

$$
\operatorname* { m a x } _ { ( h , s , a ) \in [ H ] \times S \times \mathcal { A } } \frac { d _ { h } ^ { \pi ^ { \star } } ( s , a ) } { d _ { h } ^ { \mu } ( s , a ) } \leq C _ { \eta } ^ { \star }
$$

for some $C _ { \eta } ^ { \star } \in [ 1 , \infty )$ , where $0 / 0 : = 0$ and $x / 0 : = + \infty f o r x > 0$

Remark 2.1. Several recent fast rate analyses consider the setting where $\pi ^ { \mathrm { r e f } }$ is also the behavior policy generating the ofline data (Zhao et al., 2026b; Ji et al., 2026a; Zhao et al., 2026a). By Proposition 2.1 and the Gibbs map in equation (2), supp $( \pi _ { h } ^ { \star } ( \cdot \mid s ) ) = \operatorname { s u p p } ( \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) )$ for every $( h , s )$ . Consequently, when $\mu = \pi ^ { \mathrm { r e f } }$ , the occupancy measure of $\pi ^ { \star }$ is automatically covered by that of $\mu ,$ and hence $C _ { \eta } ^ { \star }$ is finite. In contrast, we do not require the behavior policy $\mu$ to coincide with the reference policy $\pi ^ { \mathrm { r e f } }$ . In this more general setting, Assumption 2.2 requires the occupancy measure of $\pi ^ { \star }$ to be covered by that of $\mu .$ In particular, the behavior policy must cover the action support of the reference policy at states visited by $\pi ^ { \star }$

Learning Objective. Our goal is to learn a policy π that maximizes the KL-regularized return $J _ { \eta } ( \pi ) : = \mathbb { E } _ { s _ { 1 } \sim \rho } [ V _ { 1 } ^ { \pi } ( s _ { 1 } ) ]$ , and we measure the policy’s performance by

$$
\operatorname { S u b O p t } ( \pi ) : = J _ { \eta } ( \pi ^ { \star } ) - J _ { \eta } ( \pi ) ,
$$

that is, the suboptimality of policy π under the initial distribution $\rho .$ In addition, we say that a policy π is ϵ-optimal if Su $\mathrm { { b O p t } } ( \pi ) \leq \epsilon .$

## 3 Pessimism-Free KL-Regularized Q-Learning

In this section, we propose KL-regularized Q-Learning (KLQ), a pessimism-free algorithm for the KL-regularized objective. We first recall the classical Q-learning update (Watkins & Dayan, 1992). Classical Q-learning can be viewed as a stochastic approximation scheme for solving the Bellman optimality equation. Upon observing a transition $\left( { { s _ { h } } , { a _ { h } } , { r _ { h } } , { s _ { h + 1 } } } \right)$ at step $h ,$ , the corresponding Q-estimate is updated according to

$$
Q _ { h } ( s _ { h } , a _ { h } ) \gets ( 1 - \alpha _ { n } ) Q _ { h } ( s _ { h } , a _ { h } ) + \alpha _ { n } \big ( r _ { h } + V _ { h + 1 } ( s _ { h + 1 } ) \big ) ,
$$

where $\alpha _ { n }$ denotes the learning rate associated with the n-th visit to $\left( s _ { h } , a _ { h } \right)$ . In the standard unregularized setting, the value estimate is obtained by the hard maximum $V _ { h } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { h } ( s , a )$ For our KL-regularized objective, we retain the same stochastic approximation form of the Qlearning update and replace the hard maximum with the regularized Bellman operator $F _ { h , s }$ defined in equation (1). Specifically, we set $V _ { h } ( s ) = F _ { h , s } ( Q _ { h } ( s , \cdot ) )$ , and the policy associated with the Q estimate is given by $G _ { h , s } ( Q _ { h } ( s , \cdot ) )$ , where $G _ { h , s }$ is defined in equation (2). Similar to standard Q-learning, KLQ only updates the visited state-action pair when processing the ofline trajectories. For the n-th visit to a state-action pair, we use the standard finite-horizon Q-learning step size $\textstyle \alpha _ { n } = { \frac { H + 1 } { H + n } }$ from Jin et al. (2018). Let k denote the trajectory index, while n counts how many times the current state-action pair has been visited, including the current visit. Thus, $n = N _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } )$ .

Algorithm 1 KL-Regularized Q-Learning (KLQ)   
Require: Ofline dataset $D _ { \mu }$ , reference policy $\pi ^ { \mathrm { r e f } }$ , regularization parameter $\eta > 0$   
1: Initialize $Q _ { h } ^ { 1 } ( s , a ) = 0$ and $N _ { h } ^ { 1 } ( s , a ) = 0$ for every $( h , s , a ) \in [ H ] \times \mathcal { S } \times \mathcal { A }$   
2: Set $V _ { h } ^ { 1 } ( s ) \stackrel { \cdot \cdot } { = } F _ { h , s } ( Q _ { h } ^ { 1 } ( s , \cdot ) ) = \stackrel { \cdot \cdot } { 0 }$ for every $( h , s ) \in [ H ] \times S$ , and set $V _ { H + 1 } ^ { k } \equiv 0$   
3: for $k = 1 , \ldots , K$ do   
4: Read the k-th trajectory $\tau ^ { k }$ from $D _ { \mu }$   
5: Copy $Q ^ { k } , V ^ { k }$ , and $N ^ { k }$ to $Q ^ { k + 1 } , V ^ { k + 1 }$ , and $N ^ { k + 1 }$ , respectively   
6: for $h = 1 , \ldots , H$ do   
7: $N _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = N _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + 1$   
8: $n = N _ { \underline { { h } } } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } )$   
9: $\textstyle \alpha _ { n } = { \frac { H + 1 } { H + n } }$   
10: $Q _ { h } ^ { k + 1 } ( \bar { s } _ { h } ^ { k } , a _ { h } ^ { k } ) = ( 1 - \alpha _ { n } ) Q _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + \alpha _ { n } ( r _ { h } ^ { k } + V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) )$   
11: $V _ { h } ^ { k + 1 } ( s _ { h } ^ { k } ) = F _ { h , s _ { h } ^ { k } } ( Q _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , \cdot ) )$   
12: end for   
13: end for   
Ensure: $\widehat { Q } _ { h } = Q _ { h } ^ { K + 1 }$ and $\widehat { \pi } _ { h } ( \cdot \mid s ) = G _ { h , s } ( \widehat { Q } _ { h } ( s , \cdot ) )$

After processing all K trajectories, the output policy is given by $\widehat { \pi } _ { h } ( \cdot \mid s ) = G _ { h , s } \big ( \widehat { Q } _ { h } ( s , \cdot ) \big )$ with $\widehat { Q } _ { h } = Q _ { h } ^ { K + 1 }$ . Algorithm 1 summarizes the resulting procedure.

Compared with the pessimistic Q-learning algorithm of Shi et al. (2022), KLQ does not use an explicit pessimistic bonus or a monotone max update. While the algorithm of Shi et al. (2022) adjusts the Q-learning target by a pessimistic penalty and updates the value estimate monotonically, the Q update of KLQ is the same as in classical Q-learning, with the value function and policy determined by the KL-regularized Bellman equations. We provide the finite-sample guarantee for KLQ in the following theorem.

Theorem 3.1. Suppose Assumption 2.2 holds. For any $\delta \in ( 0 , 1 )$ , define $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ . Then, with probability at least 1 − δ, the policy πb returned by Algorithm 1 satisfies

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq c \frac { \eta e ^ { \eta H ( H + 1 ) } H ^ { 7 } S A C _ { \eta } ^ { \star } } { K } \iota ^ { 4 } ( 1 + \log K ) ,
$$

where $c > 0$ is a suficiently large universal constant. As a consequence, KLQ is guaranteed to find an ϵ-optimal policy with sample complexity $\widetilde { \cal O } \left( \eta e ^ { \eta H \left( H + 1 \right) } H ^ { 8 } S A C _ { \eta } ^ { \star } / \epsilon \right)$

Remark 3.1. Notably, the bound in Theorem 3.1 contains the exponential factor $e ^ { \eta H ( H + 1 ) }$ , which can become large as the horizon increases for fixed $\eta > 0$ . Similar exponential dependence appears in recent analyses of KL-regularized ofline learning without pessimism. In single-step problems, the bound can depend exponentially on the KL-regularization parameter (Wu et al., 2025; Chen & Zhang, 2026a), while in multi-step Markov games the corresponding factors may accumulate over the horizon (Chen et al., 2026). In our analysis, the likelihood-ratio argument compares $G _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) )$ with $\pi _ { h } ^ { \star } ( \cdot \mid s )$ using the Gibbs map defined in equation (2). These comparisons contribute to both stagewise error propagation and the conversion from Q-estimation error to policy suboptimality, as explained in Section 5.1.

Algorithm 2 KL-Regularized Pessimistic Q-Learning (KLPQ)   
Require: Ofline dataset $D _ { \mu } .$ , reference policy $\pi ^ { \mathrm { r e f } }$ , regularization parameter $\eta > 0$ , failure proba  
bility $\delta \in ( 0 , 1 )$   
1: Set $\begin{array} { r } { \iota = \log \frac { S A T } { \delta } } \end{array}$ and $\begin{array} { r } { b _ { h , n } = c _ { b } L _ { h + 1 } \iota \sqrt { \frac { H } { n } } } \end{array}$ for a fixed suficiently large universal constant $c _ { b }$   
2: Initialize $\widetilde { Q } _ { h } ^ { 1 } ( s , a ) = 0 , Q _ { h } ^ { 1 } ( s , a ) = 0 ,$ and $N _ { h } ^ { 1 } ( s , a ) = 0$ for every $( h , s , a ) \in [ H ] \times \mathcal { S } \times \mathcal { A }$   
3: Set $V _ { h } ^ { 1 } ( s ) = F _ { h , s } ( Q _ { h } ^ { 1 } ( s , \cdot ) ) = 0$ for every $( h , s ) \in [ H ] \times S .$ and set $V _ { H + 1 } ^ { k } \equiv 0$   
4: for $k = 1 , \ldots , K$ do   
5: Read the k-th trajectory $\tau ^ { k }$ from $D _ { \mu }$   
6: Copy $\tilde { Q } ^ { k } , Q ^ { k } , V ^ { k }$ , and $N ^ { k }$ to $\tilde { Q } ^ { k + 1 } , Q ^ { k + 1 } , V ^ { k + 1 }$ , and $N ^ { k + 1 }$ , respectively   
7: for $h = 1 , \ldots , H$ do   
8: $N _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = N _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + 1$   
9: $n \stackrel { \cdot \cdot } { = } N _ { \underline { { h } } . } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } )$   
10: $\begin{array} { r } { \alpha _ { n } = \frac { H + 1 } { H + n } } \end{array}$   
11: $\tilde { Q } _ { h } ^ { k + 1 } \tilde { ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } = ( 1 - \alpha _ { n } ) \tilde { Q } _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + \alpha _ { n } ( r _ { h } ^ { k } + V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) - b _ { h , n } )$   
12: $Q _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = \operatorname* { m a x } \{ Q _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , \widetilde Q _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \}$   
13: $V _ { h } ^ { \bar { k } + 1 } ( s _ { h } ^ { \bar { k } } ) \stackrel {  } { = } F _ { h , s _ { h } ^ { k } } ( Q _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , \cdot ) ) ^ { \cdot }$   
14: end for   
15: end for   
Ensure: $\widehat { Q } _ { h } = Q _ { h } ^ { K + 1 }$ and $\widehat { \pi } _ { h } ( \cdot \mid s ) = G _ { h , s } ( \widehat { Q } _ { h } ( s , \cdot ) )$

## 4 KL-Regularized Pessimistic Q-Learning

Although KLQ achieves a fast rate, Theorem 3.1 contains an exponential factor in η and H. This motivates us to further propose KL-regularized Pessimistic Q-Learning (KLPQ) to improve this dependence. KLPQ retains the regularized value and policy updates of KLQ in Algorithm 1, while modifying the Q-estimation procedure through a pessimistic bonus and a monotone update. More specifically, KLPQ introduces the pessimistic bonus

$$
b _ { h , n } : = c _ { b } L _ { h + 1 } \iota \sqrt { \frac { H } { n } } \quad \mathrm { w i t h ~ } \iota : = \log \frac { S A T } { \delta } \mathrm { ~ a n d ~ } n \geq 1 ,\tag{4}
$$

where $c _ { b } > 0$ is a fixed universal constant. In KLPQ, we maintain an auxiliary Q estimate $\widetilde Q ^ { k }$ in addition to $Q ^ { k }$ . At the n-th visit to $( s _ { h } ^ { k } , a _ { h } ^ { k } )$ , we first update $\widetilde { Q } _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } )$ using the pessimistic Q-learning target:

$$
\begin{array} { r } { \widetilde { Q } _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( 1 - \alpha _ { n } ) \widetilde Q _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + \alpha _ { n } ( r _ { h } ^ { k } + V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) - b _ { h , n } ) . } \end{array}
$$

Inspired by Shi et al. (2022), we use the monotone max update

$$
Q _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = \operatorname* { m a x } \left\{ Q _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) , \widetilde Q _ { h } ^ { k + 1 } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) \right\} ,
$$

which ensures that the Q estimate does not decrease after each update. The two $\mathrm { Q }$ estimates serve diferent roles. The auxiliary iterate $\widetilde Q ^ { k }$ follows the Q-learning update with the pessimistic bonus, while $Q ^ { k }$ records the monotone estimate used to compute the regularized value function. In the basic LCB-Q algorithm of Shi et al. (2022), monotonicity is enforced directly at the value level by taking the maximum of the previous value estimate and ma $\mathrm { x } _ { a } Q _ { h } ( s , a )$ . In our regularized setting, the value estimate is instead given by $F _ { h , s } ( Q _ { h } ( s , \cdot ) )$ , where $F _ { h , s }$ is defined in equation (1). Accordingly, we apply the max update to the $\mathrm { Q }$ estimate and then compute the value through $F _ { h , s } .$ . Since $F _ { h , s }$ is monotone in each coordinate, the value estimates are also nondecreasing. This monotone construction supports the pessimism guarantee and enables direct control of the final iterate error. For each $k \in [ K + 1 ]$ , let $\pi ^ { k }$ denote the policy induced by $Q ^ { k }$ , i.e., $\pi _ { h } ^ { k } ( \cdot \mid s ) : = G _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) )$ . In particular, $\widehat { \pi } = \pi ^ { K + 1 }$ . Algorithm 2 summarizes the resulting procedure. We provide the finite-sample guarantee for KLPQ in the following theorem.

Theorem 4.1. Suppose Assumption 2.2 holds. For any $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , the policy $\widehat { \pi }$ returned $b y$ Algorithm $\mathcal { Q }$ satisfies, with probability at least $1 - \delta$

$$
\operatorname { S u b O p t } ( { \widehat { \pi } } ) \leq c { \frac { \eta H ^ { 6 } S A C _ { \eta } ^ { \star } } { K } } \iota ^ { 3 } ( 1 + \log K ) ,
$$

where $c > 0$ is a universal constant. As a consequence, KLPQ is guaranteed to find an ϵ-optimal policy with sample complexity $\widetilde { \cal O } \left( \eta H ^ { 7 } S A C _ { \eta } ^ { \star } / \epsilon \right)$

Compared with Theorem 3.1, Theorem 4.1 removes the explicit factor $e ^ { \eta H ( H + 1 ) }$ and improves the polynomial dependence on H by one factor. This improvement holds for any fixed $\eta > 0$ , without changing the KL-regularized objective or the concentrability coeficient $C _ { \eta } ^ { \star }$ . The efect of varying η on the learning objective and data coverage is discussed later in this section.

Remark 4.1. A similar role of pessimism has also been observed in KL-regularized ofline bandits, where Ji et al. (2026a) construct a pessimistic reward estimate and obtain polynomial rather than exponential dependence on $\eta .$ . Their setting is single-step, so pessimism is imposed directly on the reward estimate before forming the Gibbs policy. KLPQ, by contrast, maintains pessimism throughout the multi-step Q-learning updates.

Remark 4.2. Explicit exponential dependence can also be avoided without pessimism when additiona structure is available. For example, in single-step zero-sum games, Zhang et al. (2026) exploit equilibrium stability and the skew-symmetry of the game operator to localize the analysis to unilateral estimation errors. Their argument relies on a problem-specific structure. In the multi-step Q-learning problem considered here, we instead use pessimism to remove the exponential factor.

Comparison with Unregularized Ofline Q-Learning. Our work is closely related to the ofline pessimistic Q-learning analysis of Shi et al. (2022), which also considers finite-horizon tabular MDPs under single-policy concentrability. However, their work studies the standard unregularized objective. Their LCB-Q-Advantage algorithm achieves a sample complexity of $\widetilde { O } ( H ^ { 4 } S C ^ { \star } / \epsilon ^ { 2 } + H ^ { 5 } S C ^ { \star } / \epsilon )$ Its leading $\epsilon ^ { - 2 }$ term matches the minimax lower bound $\widetilde \Omega ( H ^ { 4 } S C ^ { \star } / \epsilon ^ { 2 } )$ up to logarithmic factors (Xie et al., 2021b). We note that our guarantee concerns the KL-regularized objective and its corresponding optimal policy. Thus, its dependence on the problem parameters is not directly comparable with the bound of Shi et al. (2022). In particular, the KL-regularized optimal policy is generally stochastic, so the corresponding coverage requirement can involve multiple actions at each state, leading to the additional factor A in our current bound. The main distinction is the dependence on the target accuracy. For the KL-regularized objective, we bound policy suboptimality by the squared Q-estimation error. Thus, an ${ \cal \tilde { O } } ( 1 / K )$ bound on the squared estimation error yields an ${ \cal \tilde { O } } ( 1 / K )$ policy-suboptimality rate. Accordingly, KLPQ achieves $\widetilde { \cal O } ( \eta H ^ { 7 } S A C _ { \eta } ^ { \star } / \epsilon )$ for fixed $\eta ,$ rather than the $\epsilon ^ { - 2 }$ dependence in the unregularized setting. This diference does not contradict the minimax lower bound of Xie et al. (2021b), which applies to the standard unregularized objective.

Efect of Regularization Strength. An important point in interpreting our results is that the learning objective itself depends on the regularization strength. Although Theorems 3.1 and 4.1 do not impose additional restrictions on $\eta > 0$ beyond Assumption 2.2, choosing η afects both the learning objective and the sample complexity guarantees. For example, the exponential factor in Theorem 3.1 can make the fast-rate guarantee less informative as the horizon increases. A natural way to control this factor is to choose $\eta = \Theta ( H ^ { - 2 } )$ . However, this also strengthens the KL regularization and changes the learning objective itself. In particular, as η becomes suficiently small, the regularized optimal policy approaches the reference policy. In this regime, the learning problem can become trivial, since the reference policy itself may already achieve the desired accuracy without using any ofline data. Thus, the choice of η should reflect the tradeof between staying close to the reference policy and improving reward performance through ofline learning.

## 5 Theoretical Analysis

In this section, we provide proof sketches for Theorems 3.1 and 4.1. We first analyze KLQ and then show how pessimism changes the analysis for KLPQ. For simplicity, throughout this section, we use ${ \widetilde { O } } ( \cdot )$ to suppress polynomial dependencies on H, S, A, and $C _ { \eta } ^ { \star }$ , as well as logarithmic factors in the problem parameters, while keeping the dependence on K and η explicit. The full parameter dependencies are provided in the main theorems and their detailed proofs in Appendices C and D.

## 5.1 Analysis for KLQ

For each k and h, define $e _ { h } ^ { k } ( s ) : = \mathbb { E } _ { a \sim \pi _ { h } ^ { \star } ( \cdot | s ) } [ ( Q _ { h } ^ { k } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } ]$ , with $e _ { H + 1 } ^ { k } \equiv 0$ . Let $\gamma _ { h } : = e ^ { 2 \eta L _ { h } }$ $\gamma _ { H + 1 } : = 1$ , and $\Gamma _ { a : b } : = \Pi _ { j = a } ^ { b } \gamma _ { j }$ , with $\Gamma _ { a : b } : = 1$ when $a > b$

Proof Sketch of Theorem 3.1. By Lemma C.2, the policy suboptimality satisfies

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \frac { \eta } { 2 } \sum _ { h = 1 } ^ { H } \Gamma _ { 1 : h } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] .\tag{5}
$$

Thus, it sufices to control the squared error of the final Q iterate. To this end, we first relate the error at stage h to that at stage $h + 1$ and then apply a backward recursion. Define $\mathcal { E } _ { h } : =$ $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } [ e _ { h } ^ { k } ( s ) ] } \end{array}$ , with $\mathcal { E } _ { H + 1 } : = 0$ . By Lemma C.4, with high probability,

$$
\mathcal { E } _ { h } \leq \left( 1 + \frac { 5 } { H } \right) \gamma _ { h + 1 } \mathcal { E } _ { h + 1 } + \widetilde O ( \gamma _ { h + 1 } ( 1 + \log K ) ) .\tag{6}
$$

In the recursion (6), the first term corresponds to the error propagated from stage $h + 1$ , while the second term is the accumulated statistical error. As the statistical error grows only polylogarithmically with K, backward iteration yields

$$
\begin{array} { r } { \mathcal { E } _ { h } \leq \Gamma _ { h + 1 : H } \widetilde { O } ( 1 ) . } \end{array}\tag{7}
$$

The bound in (7) controls the cumulative squared Q-estimation error $\mathcal { E } _ { h }$ , whereas bound (5) requires controlling the final-iterate error. Since the Q-estimates of KLQ are not monotone, the cumulative error bound does not directly yield a bound for the final iterate. To address this issue, we apply Lemma C.5 to relate the final-iterate error to the cumulative error at the next stage. Specifically, we obtain $\begin{array} { r } { \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } [ e _ { h } ^ { K + 1 } ( s ) ] \le \frac { \gamma _ { h + 1 } } { K } \widetilde O ( 1 + \mathcal { E } _ { h + 1 } ) \le \frac { \Gamma _ { h + 1 : H } } { K } \widetilde O ( 1 ) } \end{array}$ . Substituting this into (5) and using $\Gamma _ { 1 : h } \Gamma _ { h + 1 : H } = e ^ { \tilde { \eta } H ( H + 1 ) }$ , we obtain

$$
\operatorname { S u b O p t } ( { \widehat { \pi } } ) \leq \eta e ^ { \eta H ( H + 1 ) } { \widetilde { O } } ( K ^ { - 1 } ) ,
$$

which proves Theorem 3.1. The exponential factor arises from both stagewise error propagation and the conversion to policy suboptimality.

## 5.2 Analysis for KLPQ

Proof Sketch of Theorem 4.1. For KLPQ, let $Q _ { h } ^ { k }$ and $V _ { h } ^ { k }$ be its iterates, and define $e _ { h } ^ { k }$ and $\mathcal { E } _ { h }$ as in Section 5.1. By the construction of pessimistic Q-estimates and the monotone update in Algorithm 2, Lemma D.2 ensures, with high probability, $Q _ { h } ^ { k } ( s , a ) \leq Q _ { h } ^ { \pi ^ { k } } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a )$ . The pointwise pessimism property plays a crucial role in removing the exponential factors in the KLQ analysis. For KLQ, the bound $( V _ { h } ^ { k } ( s ) - V _ { h } ^ { \star } ( s ) ) ^ { 2 } \leq \gamma _ { h } e _ { h } ^ { k } ( s )$ in inequality (14) introduces the factor $\gamma _ { h + 1 }$ into the squared-error recursion (6). In contrast, for KLPQ, the pointwise pessimism property and the convexity of the regularized Bellman operator yield $( V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) ) ^ { 2 } \leq e _ { h } ^ { k } ( s )$ as shown in inequality (40), avoiding this additional factor in the error recursion. For KLQ, as detailed in our proof, the likelihood-ratio argument is used both to bound the KL divergence between $\widehat { \pi }$ and $\pi ^ { \star }$ and to compare their occupancy measures, resulting in the factor $\Gamma _ { 1 : h }$ in bound (5). In contrast, fo ${ \mathrm { K L P Q } } .$ , this property and the Gibbs formula allow us to control policy suboptimality directly under $d _ { h } ^ { \pi ^ { \star } }$ , without requiring such an occupancy comparison. Specifically, by Lemma D.3, we obtain

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \eta \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } [ e _ { h } ^ { K + 1 } ( s ) ] .
$$

Applying Lemma D.4 and following the analysis for KLQ in Lemma C.4, we obtain, with high probability,

$$
\mathcal { E } _ { h } \leq \left( 1 + \frac { 5 } { H } \right) \mathcal { E } _ { h + 1 } + \widetilde O ( 1 + \log K ) .
$$

Iterating this recursion backward and using the polylogarithmic dependence of the statistical error on $K ,$ we obtain $\mathcal { E } _ { h } \leq \widetilde { O } ( 1 )$ . Finally, by the pointwise pessimism property and the monotone update in Algorithm 2, we have $0 \leq Q _ { h } ^ { \star } - Q _ { h } ^ { K + 1 } \leq Q _ { h } ^ { \star } - Q _ { h } ^ { k }$ for every $k \in [ K ]$ , which further yields $\begin{array} { r } { \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } [ e _ { h } ^ { K + 1 } ( s ) ] \le \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } [ e _ { h } ^ { k } ( s ) ] = \frac { \bar { \mathcal { E } } _ { h } } { K } \le \widetilde { O } ( K ^ { - 1 } ) } \end{array}$ . Combining the above bounds, we obtain

$$
\operatorname { S u b O p t } ( { \widehat { \pi } } ) \leq \eta { \widetilde O } ( K ^ { - 1 } ) .
$$

Compared with the final-iterate argument in Lemma C.5, this direct averaging argument also saves one factor of H, yielding the sharper horizon dependence in Theorem 4.1.

## 6 Conclusion

In this paper, we study ofline Q-learning for finite-horizon KL-regularized MDPs. We first propose KL-regularized Q-Learning (KLQ) and establish an $\widetilde { \cal O } ( 1 / \epsilon )$ sample complexity without explicit pessimism. We then develop KL-regularized Pessimistic Q-Learning (KLPQ), which achieves the same rate while removing the explicit exponential dependence on the regularization parameter and horizon. These results establish fast-rate guarantees for ofline Q-learning with KL regularization and demonstrate the benefit of pessimism in improving sample complexity.

## Appendix

A Proof of Proposition 2.1 13   
B Supporting Lemmas 13   
C Proofs for Algorithm 1 16   
C.1 Lemmas for Theorem 3.1 17   
C.2 Proof of Theorem 3.1 . 24   
D Proofs for Algorithm 2 24   
D.1 Lemmas for Theorem 4.1 24   
D.2 Proof of Theorem 4.1 . 29

## A Proof of Proposition 2.1

Proof. Define $\bar { V } _ { H + 1 } \equiv 0$ and recursively set ${ \bar { Q } } _ { h } : = r _ { h } + \mathbb { P } _ { h } { \bar { V } } _ { h + 1 } , { \ \bar { V } } _ { h } ( s ) : = F _ { h , s } ( { \bar { Q } } _ { h } ( s , \cdot ) )$ , and $\bar { \pi } _ { h } ( \cdot \mid s ) : = G _ { h , s } ( \bar { Q } _ { h } ( s , \cdot ) )$ . We prove by backward induction that $V _ { h } ^ { \pi } \leq \bar { V } _ { h }$ for every policy π and $V _ { h } ^ { \bar { \pi } } = \bar { V } _ { h }$ . Both claims are immediate at $H + 1$ . If $V _ { h + 1 } ^ { \pi } \leq \bar { V } _ { h + 1 }$ , then

$$
\begin{array} { r l } & { V _ { h } ^ { \pi } ( s ) = \big \langle \pi _ { h } ( \cdot  { \lvert { s } \rvert }  s ) , r _ { h } ( s , \cdot ) + ( { \mathbb P } _ { h } V _ { h + 1 } ^ { \pi } ) ( s , \cdot ) \big \rangle _ { \mathscr { A } } - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ( \cdot  { \lvert { s } \rvert } s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot  { \lvert { s } \rvert } { s } ) \Big ) } \\ & { \quad \quad \le \Big \langle \pi _ { h } ( \cdot  { \lvert { s } \rvert }  s ) , \bar { Q } _ { h } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ( \cdot  { \lvert { s } \rvert } { s } ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot  { \lvert { s } \rvert } { s } ) \Big ) } \\ & { \quad \quad \le F _ { h , s } ( \bar { Q } _ { h } ( s , \cdot ) ) = \bar { V } _ { h } ( s ) , } \end{array}
$$

where the last inequality is the Gibbs variational identity. If $V _ { h + 1 } ^ { \bar { \pi } } = \bar { V } _ { h + 1 }$ , then $Q _ { h } ^ { \bar { \pi } } = \bar { Q } _ { h }$ , and because $\bar { \pi } _ { h }$ is the maximizer in the Gibbs variational identity,

$$
V _ { h } ^ { \bar { \pi } } ( s ) = \Bigl \langle \bar { \pi } _ { h } ( \cdot \mid s ) , \bar { Q } _ { h } ( s , \cdot ) \Bigr \rangle _ { A } - \frac { 1 } { \eta } \mathrm { K L } \Bigl ( \bar { \pi } _ { h } ( \cdot \mid s ) \Bigr \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Bigr ) = \bar { V } _ { h } ( s ) .
$$

Thus $\bar { V } _ { h } = V _ { h } ^ { \star } , \bar { Q } _ { h } = Q _ { h } ^ { \star }$ , and $\bar { \pi } = \pi ^ { \star }$ . The same induction also gives $Q _ { h } ^ { \pi } \leq Q _ { h } ^ { \star }$ for every policy π. For the range, suppose $0 \leq V _ { h + 1 } ^ { \star } \leq L _ { h + 1 }$ . Then

$$
0 \leq Q _ { h } ^ { \star } ( s , a ) = r _ { h } ( s , a ) + ( \mathbb { P } _ { h } V _ { h + 1 } ^ { \star } ) ( s , a ) \leq 1 + L _ { h + 1 } = L _ { h } .
$$

Since $F _ { h , s } ( q )$ lies between the minimum and maximum coordinates of $q .$

$$
0 \leq V _ { h } ^ { \star } ( s ) = F _ { h , s } ( Q _ { h } ^ { \star } ( s , \cdot ) ) \leq L _ { h } .
$$

Backward induction from ${ \cal V } _ { H + 1 } ^ { \star } = 0$ completes the proof.

## B Supporting Lemmas

We first introduce the notation used throughout the statistical analysis. Let $N _ { h } ^ { k } ( s , a )$ denote the number of visits to $( h , s , a )$ before episode k, and let $k _ { h , n } ( s , a )$ denote the episode index of its n-th visit. For the learning rate $\textstyle \alpha _ { n } = { \frac { H + 1 } { H + n } }$ used in Algorithm 1, for integers $N \geq 0$ and $n \geq 1$ , define

$$
\alpha _ { 0 } ^ { N } : = \left\{ \begin{array} { l l } { \displaystyle \prod _ { j = 1 } ^ { N } ( 1 - \alpha _ { j } ) = 0 , } & { N > 0 , } \\ { 1 , } & { N = 0 , } \end{array} \right. \quad \alpha _ { n } ^ { N } : = \left\{ \begin{array} { l l } { \displaystyle \alpha _ { n } \prod _ { j = n + 1 } ^ { N } ( 1 - \alpha _ { j } ) , } & { N > n , } \\ { \displaystyle \alpha _ { n } , } & { N = n , } \\ { 0 , } & { N < n . } \end{array} \right.
$$

These coeficients and their basic identities are standard in finite-sample analyses of Q-learning; see Jin et al. (2018, Lemma 4.1) and Li et al. (2021b, Lemma 1).

Let $\mathcal { D } _ { < k } : = \sigma ( \tau ^ { 1 } , \dots , \tau ^ { k - 1 } )$ . Let $\widehat { \mathbb { P } } _ { h } ^ { k }$ be the one-hot row vector satisfying ${ \widehat { \mathbb { P } } } _ { h } ^ { k } f = f ( s _ { h + 1 } ^ { k } )$ , and regard $\widehat { \mathbb { P } } _ { h } ^ { k }$ and $\mathbb { P } _ { h , s , a }$ as row vectors over S. Under Assumption 2.2,

$$
0 \leq \frac { d _ { h } ^ { \pi ^ { \star } } ( s , a ) } { d _ { h } ^ { \mu } ( s , a ) } \leq C _ { \eta } ^ { \star } ,
$$

with the convention $0 / 0 = 0$ , as in Assumption 2.2. Throughout the appendix, we use the convention that empty sums are zero. We record the standard results below.

Lemma B.1. For every $N \geq 1$ and $a \in [ 1 / 2 , 1 ]$

$$
\frac { 1 } { N ^ { a } } \leq \sum _ { n = 1 } ^ { N } \frac { \alpha _ { n } ^ { N } } { n ^ { a } } \leq \frac { 2 } { N ^ { a } } , \qquad \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } = 1 , \qquad \operatorname* { m a x } _ { 1 \leq n \leq N } \alpha _ { n } ^ { N } \leq \frac { 2 H } { N } ,
$$

$$
\sum _ { n = 1 } ^ { N } ( \alpha _ { n } ^ { N } ) ^ { 2 } \leq \frac { 2 H } { N } , \qquad \sum _ { M = n } ^ { \infty } \alpha _ { n } ^ { M } = 1 + \frac { 1 } { H } .
$$

We also need the following standard consequence of Freedman’s martingale inequality (Freedman, 1975, Theorem 1.6).

Lemma B.2. Let $X _ { 1 } , \ldots , X _ { m }$ be martingale diferences with respect to a filtration $\{ \mathcal { F } _ { i } \} _ { i = 0 } ^ { m } ,$ let $| X _ { i } | \leq R ,$ , and set $\begin{array} { r } { W _ { m } : = \sum _ { i = 1 } ^ { m } \mathbb { E } [ X _ { i } ^ { 2 } \mid \mathcal { F } _ { i - 1 } ] } \end{array}$ . If $W _ { m } \leq \sigma ^ { 2 }$ almost surely for a deterministic $\sigma ^ { 2 } ,$ then, for every $u > 0$ , with probability at least $1 - 2 e ^ { - u }$

$$
\left| \sum _ { i = 1 } ^ { m } X _ { i } \right| \leq c { \sqrt { \sigma ^ { 2 } u } } + c R u .
$$

Lemma B.3. Suppose that, for every $k \in [ K ]$ and $h \in [ H ] , V _ { h + 1 } ^ { k }$ is $\mathcal { D } _ { < k }$ -measurable and satisfies $0 \leq V _ { h + 1 } ^ { k } ( s ) \leq L _ { h + 1 }$ for every $s \in { \mathcal { S } }$ . For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ . With probability at least $1 - \delta$ , simultaneously for every $h \in [ H ] , ( s , a ) \in \mathcal { S } \times \mathcal { A }$ , and $1 \leq N \leq \bar { N } _ { h } ^ { K + 1 } ( s , a )$

$$
\left| \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } ( s , a ) } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k _ { h , n } ( s , a ) } \right| \leq c L _ { h + 1 } \iota \sqrt { \frac { H } { N } } ,
$$

where $c > 0$ is a universal constant.

Proof. If $L _ { h + 1 } = 0$ , the claim is immediate. Hence suppose $L _ { h + 1 } > 0$ . Then $H \geq 2 .$ , so $S A T \ge 2$ and $\iota \geq \log 2$ . Fix $( h , s , a , N )$ with $1 \leq N \leq K$ and define

$$
\begin{array} { r } { \overline { N } _ { h } ^ { k } ( s , a ) : = N _ { h } ^ { k } ( s , a ) + \mathbf { 1 } \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( s , a ) \} , \quad \mathcal { F } _ { k , h } ^ { - } : = \sigma ( \mathcal { D } _ { < k } , s _ { 1 } ^ { k } , a _ { 1 } ^ { k } , \ldots , s _ { h } ^ { k } , a _ { h } ^ { k } ) , } \\ { \mathcal F _ { k , h } ^ { + } : = \mathcal { F } _ { k , h } ^ { - } \vee \sigma ( s _ { h + 1 } ^ { k } ) , \quad X _ { k } : = \alpha _ { \overline { N } _ { h } ^ { k } ( s , a ) } ^ { N } \mathbf { 1 } \{ ( s _ { h } ^ { k } , a _ { h } ^ { k } ) = ( s , a ) \} ( \widehat { \mathbb { P } } _ { h } ^ { k } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k } . } \end{array}
$$

Conditional on $\mathcal { F } _ { k , h } ^ { - }$ , every factor in $X _ { k }$ except $s _ { h + 1 } ^ { k }$ is fixed, and $s _ { h + 1 } ^ { k } \sim \mathbb { P } _ { h } ( \cdot \mid s _ { h } ^ { k } , a _ { h } ^ { k } )$ . Hence $\mathbb { E } [ X _ { k } \ | \ \mathcal { F } _ { k , h } ^ { - } ] = 0$ . Moreover, $\mathcal { F } _ { k , h } ^ { - } \subseteq \mathcal { F } _ { k , h } ^ { + } \subseteq \mathcal { F } _ { k + 1 , h } ^ { - }$ for $k < K$ . Define $\mathcal { G } _ { 0 } : = \mathcal { F } _ { 1 , h } ^ { - } , \mathcal { G } _ { k } : = \mathcal { F } _ { k + 1 , h } ^ { - }$ for $1 \leq k < K$ , and $\mathcal G _ { K } : = \mathcal F _ { K , h } ^ { + }$ . Then $\{ \mathcal { G } _ { k } \} _ { k = 0 } ^ { K }$ is a filtration, $X _ { k }$ is $\mathcal { G } _ { k } .$ -measurable, and $\operatorname { \mathbb { E } } [ X _ { k } \mid { \mathcal { G } } _ { k - 1 } ] = \operatorname { \mathbb { E } } [ X _ { k } \mid { \mathcal { F } } _ { k , h } ^ { - } ] = 0$ for every $k \in [ K ]$ . At the n-th visit to $( s , a ) , \overline { { N } } _ { h } ^ { k } ( s , a ) = n$ . Since $\alpha _ { n } ^ { N } = 0$ for $n > N$ , using $0 \leq V _ { h + 1 } ^ { k } \leq L _ { h + 1 }$ and Lemma B.1, we have $\begin{array} { r } { | X _ { k } | \le \frac { 2 H L _ { h + 1 } } { N } } \end{array}$ . Furthermore,

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } [ X _ { k } ^ { 2 } \mid \mathcal { G } _ { k - 1 } ] = \sum _ { k = 1 } ^ { K } \mathbb { E } [ X _ { k } ^ { 2 } \mid \mathcal { F } _ { k , h } ^ { - } ] \leq L _ { h + 1 } ^ { 2 } \displaystyle \sum _ { k = 1 } ^ { K } \big ( \alpha _ { \overline { { N } } _ { h } ^ { k } ( s , a ) } ^ { N } \big ) ^ { 2 } \mathbf { 1 } \{ \big ( s _ { h } ^ { k } , a _ { h } ^ { k } \big ) = \big ( s , a \big ) \} } \\ & { \qquad \leq L _ { h + 1 } ^ { 2 } \displaystyle \sum _ { n = 1 } ^ { N } ( \alpha _ { n } ^ { N } ) ^ { 2 } \leq \frac { 2 H L _ { h + 1 } ^ { 2 } } { N } . } \end{array}
$$

Hence, for $N \geq H$ , applying Lemma B.2 with $u = 2 \iota \mathrm { ~ y ~ }$ ields, with probability at least $1 - 2 e ^ { - 2 \iota }$

$$
\left. \sum _ { k = 1 } ^ { K } X _ { k } \right. \leq c L _ { h + 1 } \sqrt { \frac { H \iota } { N } } + c L _ { h + 1 } \frac { H \iota } { N } \leq c L _ { h + 1 } \iota \sqrt { \frac { H } { N } } ,
$$

where the last inequality uses $\iota \geq$ log 2 and $H / N \le \sqrt { H / N }$ , after absorbing numerical factors into the universal constant. Whenever $\bar { N } \leq N _ { h } ^ { K + 1 } \dot { ( s , a ) }$ , by the definition of the visit indices,

$$
\sum _ { k = 1 } ^ { K } X _ { k } = \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \big ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } ( s , a ) } - \mathbb { P } _ { h , s , a } \big ) V _ { h + 1 } ^ { k _ { h , n } ( s , a ) } .
$$

Thus the desired bound holds for $N \geq H$ . When $N < H$ and $N \leq N _ { h } ^ { K + 1 } ( s , a )$ , the same quantity is deterministically bounded by

$$
\left| \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } ( s , a ) } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k _ { h , n } ( s , a ) } \right| \leq L _ { h + 1 } \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } = L _ { h + 1 } \leq c L _ { h + 1 } \iota \sqrt { \frac { H } { N } } ,
$$

where we used the weight normalization in Lemma B.1, $\iota \geq \log 2$ , and $N < H$ , after enlarging c if necessary. A union bound over at most $H S A K = S A T$ choices of $( h , s , a , N )$ has failure probability at most $2 S A T e ^ { - 2 \iota } \leq \delta$ , proving the lemma. □

Lemma B.4. Suppose $S A T > 1$ . For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ . With probability at least $1 - \delta$ , simultaneously for every $h \in [ H ] , ( s , a ) \in \mathcal { S } \times \mathcal { A }$ , and $m \in [ K ]$ 2

$$
\frac { d _ { h } ^ { \mu } ( s , a ) } { N _ { h } ^ { m + 1 } ( s , a ) \vee 1 } \leq \frac { 8 \iota } { m } ,\tag{8}
$$

and whenever m $\mathcal { I } _ { h } ^ { \mu } ( s , a ) \geq 8 \iota$

$$
N _ { h } ^ { m + 1 } ( s , a ) \geq \frac { m d _ { h } ^ { \mu } ( s , a ) } { 2 } \geq 1 .\tag{9}
$$

Under Assumption 2.2, on the same event, simultaneously for every $h \in [ H ]$ ，

$$
\sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) = 0 \} \le 1 + 8 S A C _ { \eta } ^ { \star } \iota ,\tag{10}
$$

$$
\sum _ { k = 1 } ^ { K } \sum _ { s , a } { d _ { h } ^ { \pi ^ { \star } } ( s , a ) \frac { \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) \geq 1 \} } { N _ { h } ^ { k } ( s , a ) \vee 1 } \leq 8 S A C _ { \eta } ^ { \star } \iota ( 1 + \log K ) } ,\tag{11}
$$

$$
\sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { K + 1 } ( s , a ) = 0 \} \le \frac { 8 S A C _ { \eta } ^ { \star } \iota } { K } ,\tag{12}
$$

and

$$
\sum _ { s , a } { d _ { h } ^ { \pi ^ { \star } } ( s , a ) \frac { \mathbf { 1 } \{ N _ { h } ^ { K + 1 } ( s , a ) \geq 1 \} } { N _ { h } ^ { K + 1 } ( s , a ) \vee 1 } } \leq \frac { 8 S A C _ { \eta } ^ { \star } \iota } { K } .\tag{13}
$$

Proof. Fix $( h , s , a , m )$ . The count $N _ { h } ^ { m + 1 } ( s , a )$ is binomial with parameters m and $d _ { h } ^ { \mu } ( s , a )$ . If $m d _ { h } ^ { \mu } ( s , a ) < 8 \iota$ , then

$$
\frac { d _ { h } ^ { \mu } ( s , a ) } { N _ { h } ^ { m + 1 } ( s , a ) \vee 1 } \leq d _ { h } ^ { \mu } ( s , a ) < \frac { 8 \iota } { m } .
$$

Otherwise, the multiplicative Chernof bound gives

$$
\operatorname* { P r } \biggr ( N _ { h } ^ { m + 1 } ( s , a ) < \frac { m d _ { h } ^ { \mu } ( s , a ) } { 2 } \biggr ) \leq e ^ { - m d _ { h } ^ { \mu } ( s , a ) / 8 } \leq e ^ { - \iota } .
$$

On the complementary event, $N _ { h } ^ { m + 1 } ( s , a ) \geq m d _ { h } ^ { \mu } ( s , a ) / 2 \geq 1$ , and hence

$$
\frac { d _ { h } ^ { \mu } ( s , a ) } { N _ { h } ^ { m + 1 } ( s , a ) \vee 1 } \leq \frac { 2 } { m } \leq \frac { 8 \iota } { m } .
$$

A union bound over at most $H S A K = S A T$ choices proves the first two claims of Lemma B.4 simultaneously.

To prove (10), observe that (9) with $m = k - 1$ implies, for $k \ge 2 , N _ { h } ^ { k } ( s , a ) = 0$ implies $( k - 1 ) d _ { h } ^ { \mu } ( s , a ) < 8 \iota$ . Thus a state-action pair with positive behavior occupancy can have zero count for at most $1 + 8 \iota / d _ { h } ^ { \mu } ( s , a )$ episodes. Using Assumption 2.2,

$$
\sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) = 0 \} \leq \sum _ { ( s , a ) : d _ { h } ^ { \mu } ( s , a ) > 0 } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \left( 1 + \frac { 8 \iota } { d _ { h } ^ { \mu } ( s , a ) } \right) \leq 1 + 8 S A C _ { \eta } ^ { \star } \iota ,
$$

which proves the third claim of Lemma B.4.

In the sum on the left-hand side of (11), the $k = 1$ contribution is zero. For $k \geq 2 , ( 8 )$ with $m = k - 1$ gives

$$
\sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \frac { \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) \geq 1 \} } { N _ { h } ^ { k } ( s , a ) \vee 1 } = \sum _ { s , a } \frac { d _ { h } ^ { \pi ^ { \star } } ( s , a ) } { d _ { h } ^ { \mu } ( s , a ) } \frac { d _ { h } ^ { \mu } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) \geq 1 \} } { N _ { h } ^ { k } ( s , a ) \vee 1 } \leq \frac { 8 S A C _ { \eta } ^ { \star } L } { k - 1 } .
$$

Summing over k and using $\textstyle \sum _ { j = 1 } ^ { K - 1 } 1 / j \leq 1 +$ log K proves the fourth claim of Lemma B.4.

Finally, (8) with $m = K$ gives $d _ { h } ^ { \mu } ( s , a ) / ( N _ { h } ^ { K + 1 } ( s , a ) \vee 1 ) \le 8 \iota / K$ . Hence

$$
\sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { K + 1 } ( s , a ) = 0 \} \leq \frac { 8 \iota } { K } \sum _ { s , a } \frac { d _ { h } ^ { \pi ^ { \star } } ( s , a ) } { d _ { h } ^ { \mu } ( s , a ) } \leq \frac { 8 S A C _ { \eta } ^ { \star } L } { K } ,
$$

$$
\sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \frac { \mathbf { 1 } \{ N _ { h } ^ { K + 1 } ( s , a ) \geq 1 \} } { N _ { h } ^ { K + 1 } ( s , a ) \vee 1 } \leq \frac { 8 \iota } { K } \sum _ { s , a } \frac { d _ { h } ^ { \pi ^ { \star } } ( s , a ) } { d _ { h } ^ { \mu } ( s , a ) } \leq \frac { 8 S A C _ { \eta } ^ { \star } \iota } { K } .
$$

This proves the final two claims of Lemma B.4.

The following concentration bound is an immediate consequence of Lemma B.6 of Shi et al. (2022), applied with $C _ { d } = 1 , C _ { w } = B _ { h }$ , and $W _ { h + 1 } ^ { k } ( s , a ) = U _ { h } ^ { k } ( s , a ) \mathbf { 1 }$ , where 1 denotes the all-ones state vector, together with $( U _ { h } ^ { k } ( s , a ) ) ^ { 2 } \leq B _ { h } U _ { h } ^ { k } ( \dot { s } , a )$ . The logarithmic factors are absorbed into ι.

Lemma B.5. Suppose Assumption 2.2 holds. Fix nonnegative $\mathcal { D } _ { < k }$ -measurable functions $U _ { h } ^ { k }$ $s \times \mathcal { A } \to [ 0 , B _ { h } ]$ . For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ . With probability at least $1 - \delta$ , simultaneously for every $h \in [ H ]$ 2

$$
\left| \sum _ { k = 1 } ^ { K } \frac { d _ { h } ^ { \pi ^ { \star } } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } { d _ { h } ^ { \mu } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } U _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) - \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) U _ { h } ^ { k } ( s , a ) \right|
$$

$$
\leq c \sqrt { C _ { \eta } ^ { \star } B _ { h } \iota \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) U _ { h } ^ { k } ( s , a ) } + c C _ { \eta } ^ { \star } B _ { h } \iota .
$$

## C Proofs for Algorithm 1

Throughout, constant factors in logarithmic confidence terms are absorbed into universal constants. For notational convenience, we define $\Delta _ { h } ^ { k } ( s , a ) : = Q _ { h } ^ { k } ( s , a ) - Q _ { h } ^ { \star } ( s , a )$

## C.1 Lemmas for Theorem 3.1

Lemma C.1. For every $k \in [ K + 1 ] , h \in [ H ] , a n d ( s , a ) \in \mathcal { S } \times \mathcal { A }$

$$
0 \leq Q _ { h } ^ { k } ( s , a ) \leq L _ { h } , \qquad 0 \leq V _ { h } ^ { k } ( s ) \leq L _ { h } .
$$

Proof. The claim holds at initialization. Suppose it holds before episode k. At a stage-h update,

$$
0 \leq r _ { h } ^ { k } + V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) \leq 1 + L _ { h + 1 } = L _ { h } .
$$

Thus the updated Q-value is a convex combination of two values in $[ 0 , L _ { h } ]$ , while all other Q-values are unchanged. Since $F _ { h , s } ( q )$ lies between $\operatorname* { m i n } _ { a } q ( a )$ and $\operatorname* { m a x } _ { a } q ( a )$ , the updated value also lies in $[ 0 , L _ { h } ]$ . Induction over episodes proves the claim. □

Lemma C.2. For every $k \in [ K + 1 ] , h \in [ H ]$ , and $s \in { \mathcal { S } }$

$$
( V _ { h } ^ { k } ( s ) - V _ { h } ^ { \star } ( s ) ) ^ { 2 } \leq \gamma _ { h } e _ { h } ^ { k } ( s ) .\tag{14}
$$

Moreover, the output policy in Algorithm 1 satisfies

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \frac { \eta } { 2 } \sum _ { h = 1 } ^ { H } \Gamma _ { 1 : h } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] .\tag{15}
$$

Proof. Fix $k \in [ K + 1 ] , h \in [ H ]$ , and $s \in { \mathcal { S } }$ . By Lemma C.1 and Proposition 2.1, for every $a \in { \mathcal { A } } .$

$$
Q _ { h } ^ { k } ( s , a ) , Q _ { h } ^ { \star } ( s , a ) \in [ 0 , L _ { h } ] , \qquad - L _ { h } \leq \Delta _ { h } ^ { k } ( s , a ) \leq L _ { h } .
$$

For $t \in [ 0 , 1 ]$ , set $Q _ { h , t } ^ { k } ( s , \cdot ) : = Q _ { h } ^ { \star } ( s , \cdot ) + t \Delta _ { h } ^ { k } ( s , \cdot )$ . Using the Gibbs formula and $\pi _ { h } ^ { \star } ( \cdot \mid s ) =$ $G _ { h , s } ( Q _ { h } ^ { \star } ( s , \cdot ) )$ , for every action in the support of $\pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s )$ 2

$$
\begin{array} { r l } & { G _ { h , s } ( Q _ { h , t } ^ { k } ( s , \cdot ) ) ( a ) = \frac { \pi _ { h } ^ { \star } ( a \mid s ) e ^ { \eta t \Delta _ { h } ^ { k } ( s , a ) } } { \sum _ { b \in \mathcal { A } } \pi _ { h } ^ { \star } ( b \mid s ) e ^ { \eta t \Delta _ { h } ^ { k } ( s , b ) } } } \\ & { \qquad \leq \pi _ { h } ^ { \star } ( a \mid s ) e ^ { 2 \eta t L _ { h } } \leq \gamma _ { h } \pi _ { h } ^ { \star } ( a \mid s ) . } \end{array}\tag{16}
$$

The same inequality is trivial outside the support, where both probabilities are zero. Since $\nabla F _ { h , s } = G _ { h , s }$ , the fundamental theorem of calculus gives

$$
\begin{array} { r l r } & { } & { ( V _ { h } ^ { k } ( s ) - V _ { h } ^ { \star } ( s ) ) ^ { 2 } = \left( \displaystyle \int _ { 0 } ^ { 1 } \left. { G } _ { h , s } ( Q _ { h , t } ^ { k } ( s , \cdot ) ) , \Delta _ { h } ^ { k } ( s , \cdot ) \right. _ { \mathscr { A } } d t \right) ^ { 2 } } \\ & { } & { \quad \leq \displaystyle \int _ { 0 } ^ { 1 } \displaystyle \sum _ { a \in \mathcal { A } } G _ { h , s } ( Q _ { h , t } ^ { k } ( s , \cdot ) ) ( a ) ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } d t } \\ & { } & { \quad \leq \gamma _ { h } \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { h } ^ { \star } ( a \mid s ) ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } = \gamma _ { h } e _ { h } ^ { k } ( s ) , } \end{array}
$$

where the first inequality follows from Jensen’s inequality, applied first to the integral and then to the action distribution, and the second follows from (16). This proves the first claim of Lemma C.2.

We next bound the KL divergence between the corresponding Gibbs policies. Define

$$
\phi ( t ) : = \log \sum _ { b \in \mathcal { A } } \pi _ { h } ^ { \mathrm { r e f } } ( b \mid s ) e ^ { \eta Q _ { h , 1 - t } ^ { k } ( s , b ) } , \qquad t \in [ 0 , 1 ] .
$$

The Gibbs formulas imply

$$
\mathrm { K L } \Big ( G _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) ) \Big \| \pi _ { h } ^ { \star } ( \cdot \mid s ) \Big ) = \phi ( 1 ) - \phi ( 0 ) - \phi ^ { \prime } ( 0 ) ,
$$

and diferentiating ϕ(t) twice yields

$$
\phi ^ { \prime \prime } ( t ) = \eta ^ { 2 } \operatorname { V a r } _ { a \sim G _ { h , s } ( Q _ { h , 1 - t } ^ { k } ( s , \cdot ) ) } ( \Delta _ { h } ^ { k } ( s , a ) ) .
$$

Therefore, Taylor’s theorem with integral remainder, together with $\operatorname { V a r } ( X ) \leq \mathbb { E } [ X ^ { 2 } ]$ and (16), yields

$$
\begin{array} { r l } {  { \mathrm { K L } \Big ( G _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) ) \big \| \pi _ { h } ^ { \star } ( \cdot \mid s ) \Big ) = \eta ^ { 2 } \int _ { 0 } ^ { 1 } ( 1 - t ) \operatorname { V a r } _ { a \sim G _ { h , s } ( Q _ { h , 1 - t } ^ { k } ( s , \cdot ) ) } ( \Delta _ { h } ^ { k } ( s , a ) ) d t } } \\ & { \leq \eta ^ { 2 } \gamma _ { h } \int _ { 0 } ^ { 1 } ( 1 - t ) d t \sum _ { a \in \mathcal A } \pi _ { h } ^ { \star } ( a \mid s ) ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } = \frac { \eta ^ { 2 } \gamma _ { h } } { 2 } \epsilon _ { h } ^ { k } ( s ) . } \end{array}\tag{17}
$$

We now turn to the output policy. Taking $k = K + 1$ and $t = 1$ in (16) gives ${ \widehat { \pi } } _ { h } ( a \mid s ) \leq \gamma _ { h } \pi _ { h } ^ { \star } ( a \mid s )$ Since $\hat { d _ { 1 } ^ { \pi } } = \boldsymbol { d _ { 1 } ^ { \pi ^ { \star } } } = \rho .$ , induction over h gives

$$
\begin{array} { r l } & { d _ { h + 1 } ^ { \widehat { \pi } } ( s ^ { \prime } ) = \displaystyle \sum _ { s , a } d _ { h } ^ { \widehat { \pi } } ( s ) \widehat { \pi } _ { h } ( a \mid s ) \mathbb { P } _ { h } ( s ^ { \prime } \mid s , a ) } \\ & { \qquad \leq \Gamma _ { 1 : h - 1 } \gamma _ { h } \displaystyle \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s ) \pi _ { h } ^ { \star } ( a \mid s ) \mathbb { P } _ { h } ( s ^ { \prime } \mid s , a ) = \Gamma _ { 1 : h } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ^ { \prime } ) . } \end{array}\tag{18}
$$

Thus $d _ { h } ^ { \widehat { \pi } } ( s ) \leq \Gamma _ { 1 : h - 1 } d _ { h } ^ { \pi ^ { \star } } ( s )$ for every h and s.

It remains to convert the pointwise KL bound into policy suboptimality. For any policy π with finite regularized value, the Gibbs formula for $\pi ^ { \star }$ gives

$$
\log { \frac { \pi _ { h } ^ { \star } ( a \mid s ) } { \pi _ { h } ^ { \mathrm { r e f } } ( a \mid s ) } } = \eta ( Q _ { h } ^ { \star } ( s , a ) - V _ { h } ^ { \star } ( s ) )
$$

on the common support. Combining this identity with Bellman consistency yields

$$
V _ { h } ^ { \star } ( s ) - V _ { h } ^ { \pi } ( s ) = \frac { 1 } { \eta } \mathrm { K L } ( \pi _ { h } ( \cdot \mid s ) \| \pi _ { h } ^ { \star } ( \cdot \mid s ) ) + \sum _ { a } \pi _ { h } ( a \mid s ) ( \mathbb { P } _ { h } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { \pi } ) ) ( s , a ) .
$$

Taking expectation with respect to $s \sim d _ { h } ^ { \pi }$ and summing over $h ,$ the next-stage value terms telescope, yielding

$$
J _ { \eta } ( \pi ^ { \star } ) - J _ { \eta } ( \pi ) = \frac { 1 } { \eta } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi } } \left[ \mathrm { K L } ( \pi _ { h } ( \cdot  { | } s ) \| \pi _ { h } ^ { \star } ( \cdot  { | } s ) ) \right] .
$$

Taking $\pi = { \widehat { \pi } }$ , using (17) and (18), and then expanding $e _ { h } ^ { K + 1 }$ gives

$$
\begin{array} { l } { { \mathrm { S u b O p t } ( \widehat \pi ) \leq \displaystyle \frac { \eta } { 2 } \sum _ { h = 1 } ^ { H } \gamma _ { h } \sum _ { s } d _ { h } ^ { \widehat \pi } ( s ) e _ { h } ^ { K + 1 } ( s ) } } \\ { { \qquad \leq \displaystyle \frac { \eta } { 2 } \sum _ { h = 1 } ^ { H } \Gamma _ { 1 : h } \sum _ { s } d _ { h } ^ { \pi ^ { \star } } ( s ) e _ { h } ^ { K + 1 } ( s ) } } \\ { { \qquad = \displaystyle \frac { \eta } { 2 } \sum _ { h = 1 } ^ { H } \Gamma _ { 1 : h } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ \big ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) \big ) ^ { 2 } \big ] , } } \end{array}
$$

which proves the second claim of Lemma C.2.

We next combine Lemma B.3 with the KL value-error bound in Lemma C.2 to derive a pointwise recursion for the squared Q-estimation error. This recursion will be used to control both the cumulative squared error and the error of the final iterate.

Lemma C.3. For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ . With probability at least $1 - \delta$ , simultaneously for every $k \in [ K + 1 ] , h \in [ H ]$ , and $( s , a ) \in \mathcal S \times \mathcal A$ 2

$$
\begin{array} { r } { ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } \le L _ { h } ^ { 2 } \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) = 0 \} + c _ { 0 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } \frac { \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) \geq 1 \} } { N _ { h } ^ { k } ( s , a ) \vee 1 } } \\ { + \left( 1 + \frac { 1 } { H } \right) \gamma _ { h + 1 } \displaystyle \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s , a ) } ( { \mathbb { P } } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) , } \end{array}\tag{19}
$$

where $c _ { 0 } > 0$ is a universal constant.

Proof. By Lemma $\mathrm { C . 1 } , V _ { h + 1 } ^ { k }$ satisfies $0 \leq V _ { h + 1 } ^ { k } \leq L _ { h + 1 }$ , and by the update order of Algorithm 1, $V _ { h + 1 } ^ { k }$ is $\mathcal { D } _ { < k ^ { - } }$ measurable. Hence Lemma B.3 applies. Work on its event of probability at least $1 - \delta$ Fix $\mathbf { \xi } ( k , h , s , a )$ , write $N : = N _ { h } ^ { k } ( s , a )$ , and suppress the dependence of $k _ { h , n }$ on $( s , a )$ . If $N = 0$ , then $Q _ { h } ^ { k } ( s , a ) = 0$ and $( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } \leq L _ { h } ^ { 2 }$ by (3).

Suppose $N \geq 1$ . Unrolling the Q-learning update and subtracting the optimal Bellman equation with the same coeficients $\alpha _ { n } ^ { N }$ gives

$$
\Delta _ { h } ^ { k } ( s , a ) = \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h } , n } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k _ { h , n } } + \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { k _ { h , n } } - V _ { h + 1 } ^ { \star } ) ,
$$

since $\alpha _ { 0 } ^ { N } = 0$ . Applying $( x + y ) ^ { 2 } \leq ( H + 1 ) x ^ { 2 } + ( 1 + 1 / H ) y ^ { 2 }$ and Lemma B.3 yields

$$
\begin{array} { r l r } {  { ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } \leq \frac { c _ { 0 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } } { N } + ( 1 + \frac { 1 } { H } ) ( \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { k _ { h , n } } - V _ { h + 1 } ^ { \star } ) ) ^ { 2 } } } \\ & { } & { \leq \frac { c _ { 0 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } } { N } + ( 1 + \frac { 1 } { H } ) \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } \big [ ( V _ { h + 1 } ^ { k _ { h , n } } - V _ { h + 1 } ^ { \star } ) ^ { 2 } \big ] } \\ & { } & { \leq \frac { c _ { 0 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } } { N } + ( 1 + \frac { 1 } { H } ) \gamma _ { h + 1 } \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } } ) ( s , a ) . ~ } \end{array}
$$

The second inequality follows from Jensen’s inequality, first with respect to the weights $\{ \alpha _ { n } ^ { N } \}$ and then with respect to $\mathbb { P } _ { h } ( \cdot \mid s , a )$ , the last follows from (14) at stage $h + 1$ . Together with the zero-count case, this proves Lemma C.3. □

Lemma C.4. Suppose Assumption 2.2 holds and $S A T > 1$ . For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ Then, with probability at least $1 - \delta$ , simultaneously for every $h \in [ H ]$

$$
\begin{array} { c } { { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } { d _ { h } ^ { \pi } } ^ { \star } ( s ) e _ { h } ^ { k } ( s ) \le \left( 1 + \frac { 5 } { H } \right) \gamma _ { h + 1 } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } { d _ { h + 1 } ^ { \pi } } ( s ) e _ { h + 1 } ^ { k } ( s ) } } \\ { { + c _ { 0 } S A C _ { \eta } ^ { \star } \gamma _ { h + 1 } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } ( 1 + \log K ) . } } \end{array}\tag{20}
$$

Consequently,

$$
\sum _ { k = 1 } ^ { K } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { { n } ^ { \star } } } \big [ ( Q _ { h } ^ { k } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] \le c _ { 1 } S A C _ { \eta } ^ { \star } \Gamma _ { h + 1 : H } H ^ { 2 } L _ { h } ^ { 3 } \iota ^ { 3 } ( 1 + \log K ) ,\tag{21}
$$

where $c _ { 0 } , c _ { 1 } > 0$ are universal constants.

Proof. Invoke Lemmas C.3 and B.4 with failure probability $\delta / 4$ each, and invoke Lemma B.5 twice with failure probability $\delta / 4 ,$ , first with

$$
U _ { h } ^ { k } ( s , a ) : = \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s , a ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) ,\tag{22}
$$

and then with

$$
U _ { h } ^ { k } ( s , a ) : = ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) .\tag{23}
$$

In both applications take $B _ { h } = L _ { h + 1 } ^ { 2 }$ . By Lemma C.1 and Proposition 2.1, $0 \leq e _ { h + 1 } ^ { k } ( s ) \leq L _ { h + 1 } ^ { 2 }$ Moreover, by the update order in Algorithm 1, $Q ^ { k }$ , and hence $e ^ { k }$ , are $\mathcal { D } _ { < k } .$ -measurable. Since $k _ { h , n } ( s , a ) < k$ for every $1 \leq n \leq N _ { h } ^ { k } ( s , a )$ , the historical test function in (22) is also $\mathcal { D } _ { < k }$ -measurable. Thus both choices satisfy the hypotheses of Lemma B.5. By a union bound, all four events hold simultaneously with probability at least $1 - \delta$

Summing (19) over $k \in [ K ]$ and averaging with respect to $d _ { h } ^ { \pi ^ { \star } }$ gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h } ^ { \pi ^ { k } } ( s ) e _ { h } ^ { k } ( s ) \leq L _ { h } ^ { 2 } \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) = 0 \} } \\ & { \qquad + c _ { 0 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \frac { \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) \geq 1 \} } { N _ { h } ^ { k } ( s , a ) \vee 1 } } \\ & { \qquad + \left( 1 + \frac { 1 } { H } \right) \gamma _ { h + 1 } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \displaystyle \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s , a ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) . } \end{array}\tag{24}
$$

By (10) and (11), we have

$$
\begin{array} { r l } & { \displaystyle L _ { h } ^ { 2 } \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) = 0 \} + c _ { 0 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \frac { \mathbf { 1 } \{ N _ { h } ^ { k } ( s , a ) \geq 1 \} } { N _ { h } ^ { k } ( s , a ) \vee 1 } } \\ & { \qquad \leq c S A C _ { \eta } ^ { \star } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } ( 1 + \log K ) . } \end{array}\tag{25}
$$

It remains to bound (I). Applying Lemma B.5 with the test function in (22) gives

$$
( \mathrm { I } ) \leq \sum _ { k = 1 } ^ { K } \frac { d _ { h } ^ { \pi ^ { \star } } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } { d _ { h } ^ { \mu } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h } , n ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } ) ( s _ { h } ^ { k } , a _ { h } ^ { k } ) + c \sqrt { C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota \left( \mathrm { I } \right) } + c C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota
$$

$$
\overset { ( i ) } { \leq } \left( 1 + \frac { 1 } { 8 H } \right) ( \operatorname { I I } ) + c C _ { \eta } ^ { \star } H L _ { h + 1 } ^ { 2 } \iota .\tag{26}
$$

Here (i) uses $\begin{array} { r } { c \sqrt { x y } \leq \frac { x } { 1 6 H } + 4 c ^ { 2 } H y } \end{array}$ with $x = \left( \mathrm { I } \right)$ and $y = C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota$ . After moving $x / ( 1 6 H )$ to the left, we use $( 1 \stackrel { \cdot } { - } 1 / ( 1 6 \stackrel { \cdot } { H } ) ) ^ { - 1 } \leq 1 + 1 / ( 8 H )$ and absorb numerical factors into the universal constant. We next reindex (II). For each state-action pair $( s , a )$ , the m-th visit occurs in episode $k _ { h , m } ( s , a )$

and satisfies $N _ { h } ^ { k _ { h , m } ( s , a ) } ( s , a ) = m - 1$ . Hence

$$
\begin{array} { r l } & { ( \Pi ) = \displaystyle \sum _ { s , a } \frac { d _ { h } ^ { n + } ( s , a ) } { d _ { h } ^ { n } ( s , a ) } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) } \sum _ { n = 1 } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) } \sum _ { n = 1 } ^ { \infty n - 1 } ( \mathbb { P } _ { h } \epsilon _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) } \\ & { \quad = \displaystyle \sum _ { s , a } \frac { d _ { h } ^ { n + } ( s , a ) } { d _ { h } ^ { n } ( s , a ) } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) - 1 } \sum _ { n = 1 } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) } \left( \sum _ { m = n + 1 } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) } \alpha _ { n } ^ { m - 1 } \right) ( \mathbb { P } _ { h } \epsilon _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) } \\ & { \quad \le \left( 1 + \displaystyle \frac { 1 } { H } \right) \displaystyle \sum _ { s , a } \frac { d _ { h } ^ { n + } ( s , a ) } { d _ { h } ^ { n } ( s , a ) } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) } \sum _ { n = 1 } ^ { N _ { h } ^ { \kappa + 1 } ( s , a ) } ( \mathbb { P } _ { h } \epsilon _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) } \\ & { \quad = \displaystyle \left( 1 + \displaystyle \frac { 1 } { H } \right) \displaystyle \sum _ { b = 1 } ^ { K } \frac { d _ { h } ^ { n + } ( s , b _ { h } ^ { \kappa } ) } { d _ { h } ^ { n } ( s _ { h } ^ { \kappa } , a ) } ( \mathbb { P } _ { h } \epsilon _ { h + 1 } ^ { k } ) ( s _ { h } ^ { k } , a ) . } \end{array}\tag{27}
$$

The inequality in (27) uses

$$
\sum _ { m = n + 1 } ^ { N ^ { K + 1 } ( s , a ) } \alpha _ { n } ^ { m - 1 } \le \sum _ { M = n } ^ { \infty } \alpha _ { n } ^ { M } = 1 + \frac { 1 } { H } ,
$$

and adds the nonnegative term corresponding to $n = N _ { h } ^ { K + 1 } ( s , a )$ . It remains to control (III). Applying Lemma B.5 with (23) and the occupancy recursion under $\pi ^ { \star }$ gives

$$
\begin{array} { r l } & { \mathrm { ( I I I ) } \le \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) + c \sqrt { C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota _ { k = 1 } ^ { K } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) } + c C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota } \\ & { \quad = \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c \sqrt { C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota _ { k = 1 } ^ { K } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) } + c C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota } \\ & { \quad \stackrel { ( i i ) } { \le } \displaystyle \left( 1 + \frac { 1 } { 1 6 H } \right) \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c C _ { \eta } ^ { \star } H L _ { h + 1 } ^ { 2 } \iota . } \end{array}\tag{28}
$$

Here (ii) uses

$$
c \sqrt { x y } \leq \frac { x } { 1 6 H } + 4 c ^ { 2 } H y ,
$$

with

$$
x = \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) , \qquad y = C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota .
$$

Combining (26), (27), and (28) gives

$$
\begin{array} { l } { { \displaystyle \mathrm { ( I ) } \le \left( 1 + \frac 1 { 8 H } \right) \left( 1 + \frac 1 { H } \right) \left( 1 + \frac 1 { 1 6 H } \right) \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c C _ { \eta } ^ { \star } H L _ { h + 1 } ^ { 2 } \upsilon } } \\ { { \displaystyle \quad \le \left( 1 + \frac 2 { H } \right) \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c C _ { \eta } ^ { \star } H L _ { h + 1 } ^ { 2 } \iota . } } \end{array}\tag{29}
$$

Combining (24), (25), and (29) yields

$$
\begin{array} { r l } { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } { d _ { h } ^ { \pi } ( s ) e _ { h } ^ { k } ( s ) } \le c S A C _ { \eta } ^ { \star } { H ^ { 2 } } { L _ { h } ^ { 2 } { \iota _ { l } ^ { 3 } } ( 1 + \log K ) } + } & { } \\ & { \quad \quad \quad \quad \quad \left( 1 + \displaystyle \frac { 1 } { H } \right) \gamma _ { h + 1 } \Bigg [ \left( 1 + \displaystyle \frac { 2 } { H } \right) \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } { d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) } + c C _ { \eta } ^ { \star } { H } { L _ { h + 1 } ^ { 2 } } \iota \Bigg ] } \\ & { \quad \quad \quad \quad \le \left( 1 + \displaystyle \frac { 5 } { H } \right) \gamma _ { h + 1 } \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } { d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) } } \\ & { \quad \quad \quad \quad + c _ { 0 } S A C _ { \eta } ^ { \star } \gamma _ { h + 1 } H ^ { 2 } { L _ { h } ^ { 2 } } ^ { 3 } ( 1 + \log K ) . } \end{array}
$$

Here we used

$$
\left( 1 + \frac { 1 } { H } \right) \left( 1 + \frac { 2 } { H } \right) = 1 + \frac { 3 } { H } + \frac { 2 } { H ^ { 2 } } \leq 1 + \frac { 5 } { H } ,
$$

together with $\gamma _ { h + 1 } \geq 1 , L _ { h + 1 } \leq L _ { h } , S A \geq 1 , H \geq 1 , \iota \geq \log 2$ , and $1 + \log K \geq 1$ , absorbing lower-order terms into the universal constant. This proves the first claim of Lemma C.4.

Iterating (20) backward, with the stage- $H + 1$ term equal to zero, gives

$$
\begin{array} { r l } { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h } ^ { \pi ^ { \star } } ( s ) e _ { h } ^ { k } ( s ) \leq c _ { 0 } S A C _ { \eta } ^ { \star } H ^ { 2 } \iota ^ { 3 } ( 1 + \log K ) \displaystyle \sum _ { t = h } ^ { H } \left( 1 + \frac { 5 } { H } \right) ^ { t - h } \left( \displaystyle \prod _ { j = h + 1 } ^ { t + 1 } \gamma _ { j } \right) L _ { t } ^ { 2 } } & { } \\ { \leq c S A C _ { \eta } ^ { \star } \Gamma _ { h + 1 : H } H ^ { 2 } \iota ^ { 3 } ( 1 + \log K ) \displaystyle \sum _ { t = h } ^ { H } L _ { t } ^ { 2 } } & { } \\ { \leq c _ { 1 } S A C _ { \eta } ^ { \star } \Gamma _ { h + 1 : H } H ^ { 2 } L _ { h } ^ { 3 } \iota ^ { 3 } ( 1 + \log K ) , } \end{array}
$$

where we used

$$
\left( 1 + \frac 5 H \right) ^ { t - h } \leq \left( 1 + \frac 5 H \right) ^ { H } \leq e ^ { 5 } , \qquad \prod _ { j = h + 1 } ^ { t + 1 } \gamma _ { j } \leq \prod _ { j = h + 1 } ^ { H } \gamma _ { j } = \Gamma _ { h + 1 : H } ,
$$

and

$$
\sum _ { t = h } ^ { H } L _ { t } ^ { 2 } = \sum _ { \ell = 1 } ^ { L _ { h } } \ell ^ { 2 } \leq L _ { h } ^ { 3 } .
$$

By the definition of $e _ { h } ^ { k }$ , this proves the second claim of Lemma C.4.

Lemma C.5. Suppose Assumption ${ \it 2 . 2 }$ holds. For any $\delta \in ( 0 , 1 )$ , let $\begin{array} { r } { \iota : = \log \frac { S A T } { \delta } } \end{array}$ . Then, with probability at least $1 - \delta$ , simultaneously for every $h \in [ H ]$

$$
\begin{array} { r l r } & { } & { { \mathbb E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] \leq \frac { c _ { 0 } S A C _ { \eta } ^ { \star } \gamma _ { h + 1 } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } } { K } } \\ & { } & { \quad \quad \quad + \frac { c _ { 0 } H \iota \gamma _ { h + 1 } } { K } \displaystyle \sum _ { k = 1 } ^ { K } { \mathbb E } _ { s \sim d _ { h + 1 } ^ { \pi ^ { \star } } } \big [ e _ { h + 1 } ^ { k } ( s ) \big ] , } \end{array}\tag{30}
$$

where $c _ { 0 } > 0$ is a universal constant, and the sum on the right is zero when $h = H$

Proof. If $S A T = 1$ , the single Q-update is exact, so the left-hand side is zero. Hence suppose $S A T > 1$ . Invoke Lemmas C.3, B.4, and B.5 with failure probability $\delta / 3$ each, for Lemma B.5, take

$U _ { h } ^ { k } ( s , a ) = ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k } ) ( s , a )$ and $B _ { h } = L _ { h + 1 } ^ { 2 }$ . The range bounds ensure $0 \leq U _ { h } ^ { k } ( s , a ) \leq L _ { h + 1 } ^ { 2 }$ , and $U _ { h } ^ { k }$ is $\mathcal { D } _ { < k }$ -measurable. By a union bound, all three events hold simultaneously with probability at least $1 - \delta$

Fix $h \in [ H ]$ . For every state-action pair with $d _ { h } ^ { \pi ^ { \star } } ( s , a ) > 0$ and $N _ { h } ^ { K + 1 } ( s , a ) \geq 1$ , Assumption 2.2 gives $d _ { h } ^ { \mu } ( s , a ) > 0$ , and Lemma B.1 together with (8) at $m = K$ and $\log ( 3 S A T / \delta ) \leq 3 \iota$ gives

$$
\operatorname* { m a x } _ { n } \alpha _ { n } ^ { N _ { h } ^ { K + 1 } ( s , a ) } \leq \frac { 2 H } { N _ { h } ^ { K + 1 } ( s , a ) } \leq \frac { 4 8 H \iota } { K d _ { h } ^ { \mu } ( s , a ) } .\tag{31}
$$

Averaging (19) at $k = K + 1$ with respect to $d _ { h } ^ { \pi ^ { \star } }$ and using (12)–(13), we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { m ^ { \star } } } [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } ] } \\ & { \quad \le \frac { c S A C _ { \eta } ^ { \star } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } } { K } + \left( 1 + \frac { 1 } { H } \right) \gamma _ { h + 1 } \displaystyle \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) \displaystyle \sum _ { n = 1 } ^ { N _ { h } ^ { K + 1 } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { K + 1 } ( s , a ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) } \\ & { \quad \overset { ( i ) } { \le } \frac { c S A C _ { \eta } ^ { \star } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } } { K } + \frac { 4 8 H \iota } { K } \left( 1 + \frac { 1 } { H } \right) \gamma _ { h + 1 } \displaystyle \sum _ { k = 1 } ^ { K } \frac { d _ { h } ^ { \pi ^ { \star } } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } { d _ { h } ^ { \mu } ( s _ { h } ^ { k } , a _ { h } ^ { k } ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k } ) ( s _ { h } ^ { k } , a _ { h } ^ { k } ) . } \end{array}\tag{32}
$$

Here (i) follows from (31) after reindexing the pairs $( ( s , a ) , n )$ by their visit episodes $k _ { h , n } ( s , a )$ , the factor $d _ { h } ^ { \pi ^ { \star } } ( s , a ) / d _ { h } ^ { \mu } ( s , a )$ then yields (I).

We next control (I). Applying Lemma B.5 with $U _ { h } ^ { k } ( s , a ) = ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k } ) ( s , a )$ and $B _ { h } = L _ { h + 1 } ^ { 2 }$ , and then using the occupancy recursion under $\pi ^ { \star }$

(33)

$$
\begin{array} { r l } & { ( \mathrm { I } ) \le \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) ( { \mathbb { P } } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) + c \sqrt { C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota _ { k = 1 } ^ { K } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s , a } d _ { h } ^ { \pi ^ { \star } } ( s , a ) ( { \mathbb { P } } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) + c C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota _ { k = 1 } ^ { K } \zeta _ { \eta } ^ { \star } } , } \\ & { \quad = \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c \sqrt { C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota _ { k = 1 } ^ { K } \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) } + c C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota } \\ & { \quad \overset { ( i i ) } { \le } c \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } . } \end{array}\tag{34}
$$

Here (ii) uses $\sqrt { x y } \leq ( x + y ) / 2$ with

$$
x = \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) , \qquad y = C _ { \eta } ^ { \star } L _ { h + 1 } ^ { 2 } \iota ,
$$

after adjusting the universal constant c. Combining (32) and (34) gives

$$
\begin{array} { r l } {  { \mathbb { E } _ { ( s , a ) \sim d _ { \eta } ^ { \kappa } } \big [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] } } \\ & { \le \frac { c S A C _ { \eta } ^ { \star } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } } { K } + \frac { c H \tau \gamma _ { h + 1 } } { K } \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum _ { s } \int _ { h + 1 } ^ { \pi } ( s ) e _ { h + 1 } ^ { k } ( s ) + \frac { c C _ { \eta } ^ { \star } \gamma _ { h + 1 } H L _ { h + 1 } ^ { 2 } \iota ^ { 2 } } { K } } \\ & { \le \frac { c _ { 0 } S A C _ { \eta } ^ { \star } \gamma _ { h + 1 } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } } { K } + \frac { c _ { 0 } H \tau \gamma _ { h + 1 } } { K } \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } _ { s \sim d _ { h + 1 } ^ { \pi ^ { \star } } } \big [ e _ { h + 1 } ^ { k } ( s ) \big ] . } \end{array}
$$

Here we used $L _ { h + 1 } \leq L _ { h }$ 2 $S A \ge 1$ , ι ≥ log 2, $\gamma _ { h + 1 } \geq 1$ , and the definition of $e _ { h + 1 } ^ { k }$ . This proves Lemma C.5. When $h = H$ , the $\mathrm { s t a g e } { - } H + 1$ terms are zero. □

## C.2 Proof of Theorem 3.1

Proof. If $S A T = 1$ , the output is the unique policy, so $\mathrm { S u b o p t } ( { \widehat { \pi } } ) = 0$ . Hence suppose $S A T > 1$ Apply Lemmas C.5 and C.4 with failure probability $\delta / 2$ each. With probability at least $1 - \delta .$ for every $h \in [ H ]$ 2

$$
\begin{array} { r l } & { \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { { - k } } } \bigl [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \bigr ] \le \frac { c S A C _ { \eta } ^ { \star } \gamma _ { h + 1 } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } } { K } + \frac { c H \nu \gamma _ { h + 1 } } { K } \underset { k = 1 } { \sum } \underset { s } { \sum } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) } \\ & { \qquad \le c \frac { S A C _ { \eta } ^ { \star } } { K } \Gamma _ { h + 1 : H } H ^ { 3 } L _ { h } ^ { 3 } \iota ^ { 4 } ( 1 + \log K ) , } \end{array}
$$

where the second inequality uses (21) at stage $h + 1$ , when $h = H , e _ { H + 1 } ^ { k } \equiv 0$ . Substituting this estimate into (15) gives

$$
\begin{array} { r l } { \mathrm { S u b O p t } ( \widehat \pi ) \leq \displaystyle c \frac { \eta S A C _ { \eta } ^ { \star } H ^ { 3 } \iota ^ { 4 } ( 1 + \log K ) } { K } \sum _ { h = 1 } ^ { H } \Gamma _ { 1 : h } \Gamma _ { h + 1 : H } L _ { h } ^ { 3 } } & \\ { = \displaystyle c \frac { \eta e ^ { \eta H ( H + 1 ) } S A C _ { \eta } ^ { \star } H ^ { 3 } \iota ^ { 4 } ( 1 + \log K ) } { K } \sum _ { h = 1 } ^ { H } L _ { h } ^ { 3 } } & \\ { \leq \displaystyle c \frac { \eta e ^ { \eta H ( H + 1 ) } H ^ { 7 } S A C _ { \eta } ^ { \star } \iota ^ { 4 } ( 1 + \log K ) . } { K } } & { } \end{array}
$$

Here $\Gamma _ { 1 : h } \Gamma _ { h + 1 : H } = \Gamma _ { 1 : H } = e ^ { \eta H ( H + 1 ) } \mathrm { a n d } \sum _ { h = 1 } ^ { H } L _ { h } ^ { 3 } = ( H ( H + 1 ) / 2 ) ^ { 2 } \leq H ^ { 4 }$ . This proves Theorem 3.1.

## D Proofs for Algorithm 2

Throughout this appendix, fix the confidence level $\delta \in ( 0 , 1 )$ used in Algorithm 2 and let $\iota : =$ log $\frac { S A { \overset { \smile } { T } } } { \delta }$ . All iterate-dependent quantities refer to this fixed run of KLPQ. We use the shorthand $\Delta _ { h } ^ { k } ( s , a ) : = Q _ { h } ^ { k } ( s , a ) - Q _ { h } ^ { \star } ( s , a )$ , where $Q _ { h } ^ { k }$ now denotes the KLPQ iterates, and retain the definition of $e _ { h } ^ { k }$ from Section 5.1. We also retain $\mathcal { D } _ { < k }$ and $\widehat { \mathbb { P } } _ { h } ^ { k }$ from Appendix B. Constant-factor changes in concentration thresholds are absorbed into universal constants, throughout, the algorithm is run with the original confidence level.

## D.1 Lemmas for Theorem 4.1

Lemma D.1. For every $k \in [ K + 1 ] , h \in [ H ]$ , and $( s , a ) \in \mathcal S \times \mathcal A$

$$
\widetilde { Q } _ { h } ^ { k } ( s , a ) \le L _ { h } , \qquad 0 \le Q _ { h } ^ { k } ( s , a ) \le L _ { h } , \qquad 0 \le V _ { h } ^ { k } ( s ) \le L _ { h } .\tag{35}
$$

Moreover, $Q _ { h } ^ { k + 1 } \geq Q _ { h } ^ { k }$ and $V _ { h } ^ { k + 1 } \geq V _ { h } ^ { k }$ coordinatewise for every $k \in [ K ]$ and $h \in [ H ]$ , and

$$
Q _ { h } ^ { k } ( s , a ) = \operatorname * { m a x } _ { 1 \leq j \leq k } \widetilde { Q } _ { h } ^ { j } ( s , a ) , \quad \quad V _ { h } ^ { k } ( s ) = F _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) ) .
$$

Proof. We prove the stated properties by induction over episodes.

For the base case $k \ = \ 1$ , all claims follow directly from the initialization, and $V _ { h } ^ { 1 } ( s ) ~ =$ $F _ { h , s } ( Q _ { h } ^ { 1 } ( s , \cdot ) )$ for every $( h , s ) \in [ H ] \times \mathcal { S }$ . For the induction hypothesis, suppose that the range bounds in (35), the invariant $V _ { h } ^ { k } ( s ) = F _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) )$ , and the identity $\begin{array} { r } { Q _ { h } ^ { k } ( s , a ) = \operatorname* { m a x } _ { 1 \leq j \leq k } \widetilde { Q } _ { h } ^ { j } ( s , a ) } \end{array}$ hold at the beginning of episode k.

For the induction step from k to $k + 1$ , the algorithm carries $Q _ { h } ^ { k } , V _ { h } ^ { k }$ , and $\widetilde Q _ { h } ^ { k }$ forward except at the visited coordinates. At a stage-h update, $r _ { h } ^ { k } + V _ { h + 1 } ^ { k } ( s _ { h + 1 } ^ { k } ) - \bar { b _ { h , n } } \leq 1 + L _ { h + 1 } = L _ { h }$ . Hence the update of ${ \widetilde { Q } } _ { h }$ remains at most $L _ { h }$ . The max update in Line 12 therefore ensures that $Q _ { h } ^ { k + 1 }$ remains in $[ 0 , L _ { h } ]$ , preserves coordinatewise monotonicity, and yields $Q _ { h } ^ { k + 1 } ( s , a ) = \mathrm { m a x } _ { 1 \leq j \leq k + 1 } \widetilde Q _ { h } ^ { j } ( s , a )$ Since $F _ { h , s }$ is coordinatewise nondecreasing and lies between the minimum and maximum coordinates of its argument, the recomputed value also lies in $[ 0 , L _ { h } ]$ and satisfies $V _ { h } ^ { k + 1 } \geq V _ { h } ^ { k }$ coordinatewise. The invariant $V _ { h } ^ { k + 1 } ( s ) = \bar { F } _ { h , s } ( Q _ { h } ^ { k + 1 } ( s , \cdot ) )$ is preserved because the algorithm carries $Q _ { h } ^ { k }$ and $V _ { h } ^ { k }$ forward together and recomputes the value at every updated state. This completes the induction over k. □

We now introduce a concentration event for later use. Notably, $Q ^ { k }$ and $V ^ { k }$ before episode k depend only on $\tau ^ { 1 } , \dots , \tau ^ { k - 1 }$ , so $Q _ { h } ^ { k } , V _ { h } ^ { k }$ , and $e _ { h } ^ { k }$ are $\mathcal { D } _ { < k } .$ -measurable. The deterministic bounds (35) and (3) give

$$
0 \leq e _ { h } ^ { k } ( s ) \leq L _ { h } ^ { 2 } , \qquad 0 \leq ( { \mathbb { P } } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) \leq L _ { h + 1 } ^ { 2 } ,
$$

$$
0 \leq \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s , a ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) \leq L _ { h + 1 } ^ { 2 } .
$$

Moreover, for every $\begin{array} { r l } { k , h , s , a , } & { { } \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s , a ) } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) } \end{array}$ is $\mathcal { D } _ { < k } .$ -measurable because $k _ { h , n } ( s , a ) < k$ for every $1 \leq n \leq N _ { h } ^ { k } ( s , a )$

By Lemma D.1, $0 \leq V _ { h + 1 } ^ { k } \leq L _ { h + 1 }$ . Moreover, $V _ { h + 1 } ^ { k }$ is $\mathcal { D } _ { < k }$ -measurable, since the iterate before episode k depends only on $\begin{array} { r } { { \dot { \tau } } ^ { 1 } , \dots , \tau ^ { k - 1 } } \end{array}$ . Therefore, Lemma B.3 also holds for $\mathrm { K L P Q }$ . Apply it with failure probability $\delta / 4$ and denote the resulting event by $\mathcal { E } _ { 1 }$ . Choose the fixed universal constant $c _ { b }$ so that $c _ { b } \geq 4 \operatorname* { m a x } \{ c , 1 \}$ , where c is the universal constant in Lemma B.3. For $N \geq 1$ , Lemma B.1 with exponent $1 / 2$ gives

$$
c _ { b } L _ { h + 1 } \iota \sqrt { \frac { H } { N } } \leq \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } b _ { h , n } \leq 2 c _ { b } L _ { h + 1 } \iota \sqrt { \frac { H } { N } } .\tag{36}
$$

Moreover, on $\mathcal { E } _ { 1 }$ , for $1 \leq N \leq N _ { h } ^ { K + 1 } ( s , a )$

$$
\left| \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } ( s , a ) } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k _ { h , n } ( s , a ) } \right| \leq 4 c L _ { h + 1 } \iota \sqrt { \frac { H } { N } } \leq c _ { b } L _ { h + 1 } \iota \sqrt { \frac { H } { N } } \leq \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } b _ { h , n } .\tag{37}
$$

The first inequality is immediate when $L _ { h + 1 } ~ = ~ 0$ . Otherwise, $H \ \geq \ 2$ , so $\iota \geq \log 2$ and $\log ( 4 S A T / \delta ) = \iota + \log 4 \leq 4 \iota$ . The second follows from $c _ { b } \geq 4 c .$ , and the last is the lower bound in (36). For $N = 0$ , both weighted sums are zero.

Lemma D.2. On the event $\mathcal { E } _ { 1 } , ~ i f ~ c _ { b }$ is fixed as above,

$$
\widetilde { Q } _ { h } ^ { j } ( s , a ) \leq Q _ { h } ^ { k } ( s , a ) \leq Q _ { h } ^ { \pi ^ { k } } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a ) , \qquad V _ { h } ^ { k } ( s ) \leq V _ { h } ^ { \pi ^ { k } } ( s ) \leq V _ { h } ^ { \star } ( s )\tag{38}
$$

simultaneously for every $k \in [ K + 1 ] , ~ j \in [ k ] , ~ h \in [ H ] , ~ a n d ~ ( s , a ) \in \mathcal { S } \times \mathcal { A }$

Proof. We work on the event $\mathcal { E } _ { 1 }$ defined above. Fix $k \in [ K + 1 ]$ and proceed by backward induction on h. For the base case $h = H + 1 , V _ { H + 1 } ^ { \pi ^ { k } } = V _ { H + 1 } ^ { k } = V _ { H + 1 } ^ { \star } = 0$

For the induction hypothesis, suppose that the desired value ordering holds at $h + 1$ , namely, $V _ { h + 1 } ^ { k } \le V _ { h + 1 } ^ { \pi ^ { k } } \le V _ { h + 1 } ^ { \star }$ . For the induction step from $h + 1$ to h, the induction hypothesis, monotonicity, and nonnegativity give ${ V _ { h + 1 } ^ { \pi ^ { k } } \geq V _ { h + 1 } ^ { k } \geq V _ { h + 1 } ^ { i } \geq 0 }$ for every $i \leq k$ , and therefore $Q _ { h } ^ { \pi ^ { k } } ( s , a ) \geq 0$

Fix $j \ \leq \ k$ and $( s , a )$ , write $N : = N _ { h } ^ { j } ( s , a )$ , and suppress the dependence of $k _ { h , n }$ on $( s , a )$ Unrolling Line 11 gives

$$
\widetilde { Q } _ { h } ^ { j } ( s , a ) = \alpha _ { 0 } ^ { N } \widetilde { Q } _ { h } ^ { 1 } ( s , a ) + \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \left( r _ { h } ( s , a ) + \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } } V _ { h + 1 } ^ { k _ { h , n } } - b _ { h , n } \right) .\tag{39}
$$

Writing the Bellman equation for $Q _ { h } ^ { \pi ^ { k } } ( s , a )$ with the same weights and subtracting (39) yields

$$
\begin{array} { r l } & { \displaystyle Q _ { h } ^ { \pi ^ { k } } ( s , a ) - \widetilde Q _ { h } ^ { j } ( s , a ) = \alpha _ { 0 } ^ { N } Q _ { h } ^ { \pi ^ { k } } ( s , a ) + \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { \pi ^ { k } } - V _ { h + 1 } ^ { k _ { h , n } } ) } \\ & { \qquad + \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } b _ { h , n } - \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k _ { h , n } } } \\ & { \qquad \geq \alpha _ { 0 } ^ { N } Q _ { h } ^ { \pi ^ { k } } ( s , a ) + \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { \pi ^ { k } } - V _ { h + 1 } ^ { k _ { h , n } } ) \geq 0 . } \end{array}
$$

The first inequality follows from (37), the second uses $Q _ { h } ^ { \pi ^ { k } } \geq 0$ and $V _ { h + 1 } ^ { \pi ^ { k } } \geq V _ { h + 1 } ^ { k _ { h , n } }$ for every historical visit $k _ { h , n } < j \leq k$ . This also covers $N = 0$ , since every sum $\scriptstyle \sum _ { n = 1 } ^ { N } ( \cdots )$ above is then zero. Thus $\widetilde { Q } _ { h } ^ { j } ( s , a ) \leq Q _ { h } ^ { \pi ^ { k } } ( s , a )$ for every $j \leq k$ , and hence

$$
Q _ { h } ^ { k } ( s , a ) = \operatorname* { m a x } _ { 1 \leq j \leq k } \widetilde { Q } _ { h } ^ { j } ( s , a ) \leq Q _ { h } ^ { \pi ^ { k } } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a ) ,
$$

where the last inequality follows from Proposition 2.1. Since $\pi _ { h } ^ { k } ( \cdot \mid s ) = G _ { h , s } ( Q _ { h } ^ { k } ( s , \cdot ) )$ ，

$$
\begin{array} { r l } & { V _ { h } ^ { k } ( s ) = \Big \langle \pi _ { h } ^ { k } ( \cdot \mid s ) , Q _ { h } ^ { k } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { k } ( \cdot \mid s ) \Big | \Big | \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) } \\ & { \qquad \le \Big \langle \pi _ { h } ^ { k } ( \cdot \mid s ) , Q _ { h } ^ { \pi ^ { k } } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { k } ( \cdot \mid s ) \Big | \Big | \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) } \\ & { \qquad = V _ { h } ^ { \pi ^ { k } } ( s ) \le V _ { h } ^ { \star } ( s ) . } \end{array}
$$

This proves the induction step at stage h and hence establishes the inequalities in (38) by backward induction. Since all arguments beyond the deterministic range and monotonicity properties use only $\mathcal { E } _ { 1 }$ , this completes the proof. □

Lemma D.3. On $\mathcal { E } _ { 1 } , f o r$ every $k \in [ K + 1 ] , h \in [ H ]$ , and $s \in { \mathcal { S } }$

$$
( V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) ) ^ { 2 } \leq e _ { h } ^ { k } ( s ) .\tag{40}
$$

Moreover, the output policy in Algorithm $\mathcal { Q }$ satisfies

$$
\mathrm { S u b O p t } ( \widehat { \pi } ) \leq \eta \sum _ { h = 1 } ^ { H } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \star } } \big [ ( Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ) ^ { 2 } \big ] .\tag{41}
$$

Proof. Work on $\mathcal { E } _ { 1 }$ . For the value error, convexity of $F _ { h , s }$ at $Q _ { h } ^ { \star } ( s , \cdot )$ and the ordering $Q _ { h } ^ { k } \leq Q _ { h } ^ { \star }$ give

$$
0 \leq V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) \leq \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) , Q _ { h } ^ { \star } ( s , \cdot ) - Q _ { h } ^ { k } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } .
$$

Squaring and applying Jensen’s inequality yields

$$
( V _ { h } ^ { \star } ( s ) - V _ { h } ^ { k } ( s ) ) ^ { 2 } \leq \sum _ { a } \pi _ { h } ^ { \star } ( a \mid s ) [ Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { k } ( s , a ) ] ^ { 2 } = e _ { h } ^ { k } ( s ) ,
$$

which proves the first claim.

It remains to prove the output-policy bound. Let $\widehat { \pi } = \pi ^ { K + 1 }$ . By (38),

$$
0 \leq Q _ { h } ^ { \widehat \pi } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) .
$$

By Bellman consistency,

$$
V _ { h } ^ { \star } ( s ) = \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) , Q _ { h } ^ { \star } ( s , \cdot ) \rangle _ { \cal A } - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { \star } ( \cdot \mid s ) \Big | \Big | \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) ,
$$

$$
V _ { h } ^ { \widehat { \pi } } ( s ) = \left. \widehat { \pi } _ { h } ( \cdot  { | } s ) , Q _ { h } ^ { \widehat { \pi } } ( s , \cdot ) \right. _ { A } - \frac { 1 } { \eta } \mathrm { K L } \Bigl ( \widehat { \pi } _ { h } ( \cdot  { | } s ) \Bigl | \Big | \pi _ { h } ^ { \mathrm { r e f } } ( \cdot  { | } s ) \Bigr ) .
$$

Subtracting the two identities and adding and subtracting $\langle \pi _ { h } ^ { \star } ( \cdot \mid s ) , Q _ { h } ^ { \widehat { \pi } } ( s , \cdot ) \rangle _ { \cal A }$ yields

$$
\begin{array} { r l } & { V _ { h } ^ { \star } ( s ) - V _ { h } ^ { \widehat { \pi } } ( s ) = \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) - \widehat { \pi } _ { h } ( \cdot \mid s ) , Q _ { h } ^ { \widehat { \pi } } ( s , \cdot ) \Big \rangle _ { \cal A } + \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) , Q _ { h } ^ { \star } ( s , \cdot ) - Q _ { h } ^ { \widehat { \pi } } ( s , \cdot ) \Big \rangle _ { \cal A } } \\ & { \quad \quad \quad \quad \quad \quad - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { \star } ( \cdot \mid s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) + \frac { 1 } { \eta } \mathrm { K L } \Big ( \widehat { \pi } _ { h } ( \cdot \mid s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) } \\ & { \quad \quad \quad \quad = \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) - \widehat { \pi } _ { h } ( \cdot \mid s ) , Q _ { h } ^ { \widehat { \pi } } ( s , \cdot ) \Big \rangle _ { \cal A } + \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) , \mathbb { P } _ { h } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { \widehat { \pi } } ) ( s , \cdot ) \Big \rangle _ { \cal A } } \\ & { \quad \quad \quad \quad \quad - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { \star } ( \cdot \mid s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) + \frac { 1 } { \eta } \mathrm { K L } \Big ( \widehat { \pi } _ { h } ( \cdot \mid s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) . } \end{array}
$$

Taking expectation of this identity with respect to $s \sim d _ { h } ^ { \pi ^ { \star } }$ and summing over $h ,$ , using ${ \cal V } _ { H + 1 } ^ { \star } =$ $V _ { H + 1 } ^ { \widehat { \pi } } = 0$ , gives

$$
\begin{array} { l } { \mathrm { S u b O p t } ( \widehat \pi ) = \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } \bigg [ \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) - \widehat \pi _ { h } ( \cdot \mid s ) , Q _ { h } ^ { \widehat \pi } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } } \\ { \displaystyle \qquad - \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { \star } ( \cdot \mid s ) \Big | \Big | \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) + \frac { 1 } { \eta } \mathrm { K L } \Big ( \widehat \pi _ { h } ( \cdot \mid s ) \Big | \Big | \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \mid s ) \Big ) \bigg ] . } \end{array}
$$

The Gibbs formula for $\widehat { \pi } _ { h }$ gives

$$
\log \frac { \widehat { \pi } _ { h } ( a \mid s ) } { \pi _ { h } ^ { \mathrm { r e f } } ( a \mid s ) } = \eta Q _ { h } ^ { K + 1 } ( s , a ) - \log \sum _ { b } \pi _ { h } ^ { \mathrm { r e f } } ( b \mid s ) e ^ { \eta Q _ { h } ^ { K + 1 } ( s , b ) } .
$$

Hence, we have

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \eta } \mathrm { K L } \big ( \pi _ { h } ^ { \star } ( \cdot \ \vert \ s ) \| \widehat { \pi } _ { h } ( \cdot \vert \ s ) \big ) = \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { \star } ( \cdot \vert \ s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \vert \ s ) \Big ) - \Big \langle \pi _ { h } ^ { \star } ( \cdot \vert \ s ) , Q _ { h } ^ { K + 1 } ( s , \cdot ) \Big \rangle _ { \mathcal { A } } } \\ { \displaystyle \quad \quad \quad + \frac { 1 } { \eta } \log \sum _ { b } \pi _ { h } ^ { \mathrm { r e f } } ( b \vert s ) e ^ { \eta Q _ { h } ^ { K + 1 } ( s , b ) } } \\ { \displaystyle \quad \quad \quad \stackrel { ( i ) } { = } \frac { 1 } { \eta } \mathrm { K L } \Big ( \pi _ { h } ^ { \star } ( \cdot \vert \ s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \vert \ s ) \Big ) - \frac { 1 } { \eta } \mathrm { K L } \Big ( \widehat { \pi } _ { h } ( \cdot \ \vert \ s ) \Big \| \pi _ { h } ^ { \mathrm { r e f } } ( \cdot \vert \ s ) \Big ) } \\ { \displaystyle \quad \quad \quad - \Big \langle \pi _ { h } ^ { \star } ( \cdot \vert \ s ) - \widehat { \pi } _ { h } ( \cdot \vert \ s ) , Q _ { h } ^ { K + 1 } ( s , \cdot ) \Big \rangle _ { \mathcal { A } } , } \end{array}
$$

where (i) uses the Gibbs variational identity. Substituting this identity into the preceding expression for $\operatorname { S u b O p t } ( { \widehat { \pi } } )$ gives

$$
\begin{array} { r l } { \mathrm { S u b O p t } ( \widehat \pi ) = \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } \bigg [ \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) - \widehat \pi _ { h } ( \cdot \mid s ) , Q _ { h } ^ { \widehat \pi } ( s , \cdot ) - Q _ { h } ^ { K + 1 } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } } & { } \\ { \displaystyle } & { \quad - \frac { 1 } { \eta } \mathrm { K L } ( \pi _ { h } ^ { \star } ( \cdot \mid s ) \| \widehat \pi _ { h } ( \cdot \mid s ) ) \bigg ] } \\ { \displaystyle } & { \quad \leq \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } \left[ \Big \langle \pi _ { h } ^ { \star } ( \cdot \mid s ) - \widehat \pi _ { h } ( \cdot \mid s ) , Q _ { h } ^ { \widehat \pi } ( s , \cdot ) - Q _ { h } ^ { K + 1 } ( s , \cdot ) \Big \rangle _ { \mathscr { A } } \right] , } \end{array}
$$

where the inequality uses the nonnegativity of KL divergence.

For every action with $\pi _ { h } ^ { \star } ( a \mid s ) > 0$ , the two Gibbs formulas give

$$
\begin{array} { r l } & { \pi _ { h } ^ { \star } ( a \mid s ) - \widehat { \pi } _ { h } ( a \mid s ) = \pi _ { h } ^ { \star } ( a \mid s ) \left( 1 - \frac { e ^ { - \eta [ Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ] } } { \sum _ { b } \pi _ { h } ^ { \star } ( b \mid s ) e ^ { - \eta [ Q _ { h } ^ { \star } ( s , b ) - Q _ { h } ^ { K + 1 } ( s , b ) ] } } \right) } \\ & { \quad \quad \stackrel { ( i ) } { \le } \pi _ { h } ^ { \star } ( a \mid s ) \left( 1 - e ^ { - \eta [ Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ] } \right) } \\ & { \quad \quad \stackrel { ( i i ) } { \le } \eta \pi _ { h } ^ { \star } ( a \mid s ) [ Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ] . } \end{array}\tag{42}
$$

Here (i) uses $Q _ { h } ^ { K + 1 } \ \leq \ Q _ { h } ^ { \star }$ , which implies $\begin{array} { r } { \sum _ { b } \pi _ { h } ^ { \star } ( b \ | \ s ) e ^ { - \eta [ Q _ { h } ^ { \star } ( s , b ) - Q _ { h } ^ { K + 1 } ( s , b ) ] } \ \leq \ 1 } \end{array}$ , and (ii) uses $1 - e ^ { - v } \leq v$ for $v \geq 0$ . For actions outside the common support, both probabilities are zero, so (42) holds for every action. Since $Q _ { h } ^ { \widehat { \pi } } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) \geq 0$ , applying (42) yields

$$
\begin{array} { r l } & { \mathrm { S u b O p t } ( \widehat \pi ) \leq \eta \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } \sum _ { a } \pi _ { h } ^ { \star } ( a \mid s ) [ Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ] [ Q _ { h } ^ { \widehat \pi } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ] } \\ & { \quad \quad \quad \leq \eta \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { s \sim d _ { h } ^ { \pi ^ { \star } } } \sum _ { a } \pi _ { h } ^ { \star } ( a \mid s ) [ Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ] ^ { 2 } } \\ & { \quad \quad \quad = \eta \displaystyle \sum _ { h = 1 } ^ { H } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) ) ^ { 2 } \big ] , } \end{array}
$$

where the second inequality uses $Q _ { h } ^ { \widehat { \pi } } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a )$ from (38), and the last equality follows from $d _ { h } ^ { \pi ^ { \star } } ( s , a ) \stackrel { \cdot \cdot } { = } d _ { h } ^ { \pi ^ { \star } } ( s ) \pi _ { h } ^ { \star } ( a \mid s )$ . This proves the second claim. □

Lemma D.4. On $\mathcal { E } _ { 1 }$ , simultaneously for every $k \in [ K + 1 ] , h \in [ H ]$ , and $( s , a ) \in \mathcal S \times \mathcal A$ , writing $N : = N _ { h } ^ { k } ( s , a )$

$$
\begin{array} { r } { ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } \le L _ { h } ^ { 2 } \mathbf { 1 } \{ N = 0 \} + c H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } \frac { \mathbf { 1 } \{ N \geq 1 \} } { N \vee 1 } } \\ { + \left( 1 + \frac { 1 } { H } \right) \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) , } \end{array}
$$

where $c > 0$ is a universal constant.

Proof. Fix $k , h , s , a$ , let $N : = N _ { h } ^ { k } ( s , a )$ , and suppress the dependence of $k _ { h , n }$ on $( s , a )$ . If $N = 0$ then $Q _ { h } ^ { k } ( s , a ) = 0$ and $( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } \le L _ { h } ^ { 2 }$ by (3).

Suppose $N \geq 1$ . Subtracting (39) with $j = k$ from the optimal Bellman equation gives

$$
\begin{array} { r l } & { \displaystyle Q _ { h } ^ { \star } ( s , a ) - \widetilde Q _ { h } ^ { k } ( s , a ) = \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k _ { h , n } } ) + \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } b _ { h , n } } \\ & { \quad \quad \quad \quad - \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \widehat { \mathbb { P } } _ { h } ^ { k _ { h , n } } - \mathbb { P } _ { h , s , a } ) V _ { h + 1 } ^ { k _ { h , n } } , } \end{array}
$$

since $\alpha _ { 0 } ^ { N } = 0$ . By (38) and (37),

$$
\begin{array} { r l r } {  { 0 \leq Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { k } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a ) - \widetilde Q _ { h } ^ { k } ( s , a ) } } \\ & { } & { \leq \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k _ { h , n } } ) + 2 \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } b _ { h , n } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l } & { ( \Delta _ { h } ^ { k } ( s , a ) ) ^ { 2 } \le \left( 1 + \displaystyle \frac { 1 } { H } \right) \left( \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k _ { h , n } } ) \right) ^ { 2 } + 4 ( H + 1 ) \left( \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } b _ { h , n } \right) ^ { 2 } } \\ & { \qquad \le \left( 1 + \displaystyle \frac { 1 } { H } \right) \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } \mathbb { P } _ { h , s , a } \Big [ ( V _ { h + 1 } ^ { \star } - V _ { h + 1 } ^ { k _ { h , n } } ) ^ { 2 } \Big ] + \displaystyle \frac { 3 2 c _ { h } ^ { 2 } H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } } { N } } \\ & { \qquad \le \left( 1 + \displaystyle \frac { 1 } { H } \right) \displaystyle \sum _ { n = 1 } ^ { N } \alpha _ { n } ^ { N } ( \mathbb { P } _ { h } e _ { h + 1 } ^ { k _ { h , n } } ) ( s , a ) + \displaystyle \frac { c H ^ { 2 } L _ { h + 1 } ^ { 2 } \iota ^ { 2 } } { N } . } \end{array}
$$

The first inequality uses $( x + y ) ^ { 2 } \leq ( 1 + 1 / H ) x ^ { 2 } + ( H + 1 ) y ^ { 2 }$ . The second follows from Jensen’s inequality, first with respect to the weights $\{ \alpha _ { n } ^ { N } \}$ and then with respect to $\mathbb { P } _ { h } ( \cdot \mid s , a )$ , together with the upper bound in (36). The last inequality follows from (40), after enlarging the universal constant c to dominate $3 2 c _ { b } ^ { 2 }$ . Together with the zero-count case, this proves the lemma. □

We now introduce a common event for the proof of Theorem 4.1: If $S A T > 1$ , apply Lemma B.4 with failure probability $\delta / 4$ and denote its event by $\mathcal { E } _ { 2 } , \mathrm { i f } \ S A T = 1$ , let ${ \mathcal { E } } _ { 2 }$ be the sure event. Apply Lemma B.5 twice with failure probability $\delta / 4$ , using

$$
U _ { h } ^ { k } ( s , a ) = \sum _ { n = 1 } ^ { N _ { h } ^ { k } ( s , a ) } \alpha _ { n } ^ { N _ { h } ^ { k } ( s , a ) } ( { \mathbb { P } } _ { h } e _ { h + 1 } ^ { k _ { h , n } ( s , a ) } ) ( s , a ) \quad \mathrm { a n d } \quad U _ { h } ^ { k } ( s , a ) = ( { \mathbb { P } } _ { h } e _ { h + 1 } ^ { k } ) ( s , a ) ,
$$

respectively, with $e _ { h } ^ { k }$ defined as in Section 5.1 for Algorithm 2. The deterministic bounds and predictability above ensure that both choices of $U _ { h } ^ { k } ( s , a )$ with $B _ { h } = L _ { h + 1 } ^ { 2 }$ satisfy the conditions of Lemma B.5. Denote the two events by $\mathcal { E } _ { 3 }$ and ${ \mathcal { E } } _ { 4 }$ , and set

$$
\mathcal { E } : = \mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } \cap \mathcal { E } _ { 3 } \cap \mathcal { E } _ { 4 } , \qquad \operatorname* { P r } ( \mathcal { E } ) \geq 1 - \delta .
$$

All four events are defined for the same iterates generated using the original bonus in (4).

## D.2 Proof of Theorem 4.1

Proof. If $S A T = 1$ , the output is the unique policy, so $\mathrm { S u b o p t } ( { \widehat { \pi } } ) = 0$ . Hence suppose $S A T > 1$ Work on the common event $\mathcal { E }$ defined above, which has probability at least $1 - \delta$ . Starting from Lemma D.4, we repeat the argument used in the proof of Lemma C.4. The count-control step, the two applications of Lemma B.5, and the reindexing argument are unchanged, up to universal constants, because the required boundedness and predictability conditions are verified in Appendix D.1. The only diference is that the propagated error term in Lemma D.4 has no factor $\gamma _ { h + 1 }$ . Therefore, the same calculation yields, for every $h \in [ H ]$ ，

$$
\sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h } ^ { \pi ^ { \star } } ( s ) e _ { h } ^ { k } ( s ) \leq \left( 1 + \frac { 5 } { H } \right) \sum _ { k = 1 } ^ { K } \sum _ { s } d _ { h + 1 } ^ { \pi ^ { \star } } ( s ) e _ { h + 1 } ^ { k } ( s ) + c _ { 0 } S A C _ { \eta } ^ { \star } H ^ { 2 } L _ { h } ^ { 2 } \iota ^ { 3 } ( 1 + \log K ) .
$$

Applying the same backward iteration argument as in the proof of Lemma C.4, now without the products of $\gamma _ { j }$ , and using $e _ { H + 1 } ^ { k } \equiv 0$ together with

$$
\left( 1 + \frac { 5 } { H } \right) ^ { H } \leq e ^ { 5 } , \qquad \sum _ { t = h } ^ { H } L _ { t } ^ { 2 } \leq L _ { h } ^ { 3 } ,
$$

gives

$$
\sum _ { k = 1 } ^ { K } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { k } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] \le c _ { 1 } S A C _ { \eta } ^ { \star } H ^ { 2 } L _ { h } ^ { 3 } { \iota } ^ { 3 } ( 1 + \log K ) ,
$$

which gives the desired accumulated squared Q-estimation error bound.

By Line 12, $Q _ { h } ^ { k + 1 } \geq Q _ { h } ^ { k }$ coordinatewise, while the ordering (38) gives $Q _ { h } ^ { k } \leq Q _ { h } ^ { \star }$ . Therefore, for every $k \in [ K ]$ and $( s , a ) \in \mathcal S \times \mathcal A$

$$
0 \leq Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { K + 1 } ( s , a ) \leq Q _ { h } ^ { \star } ( s , a ) - Q _ { h } ^ { k } ( s , a ) .
$$

Squaring, averaging first with respect to $d _ { h } ^ { \pi ^ { \star } }$ , and then averaging over $k \in [ K ]$ gives

$$
\begin{array} { r l } & { \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { K + 1 } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] \le \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } _ { ( s , a ) \sim d _ { h } ^ { \pi ^ { \star } } } \big [ ( Q _ { h } ^ { k } ( s , a ) - Q _ { h } ^ { \star } ( s , a ) ) ^ { 2 } \big ] } \\ & { \qquad \le c _ { 1 } \displaystyle \frac { S A C _ { \eta } ^ { \star } } { K } H ^ { 2 } L _ { h } ^ { 3 } \iota ^ { 3 } ( 1 + \log K ) . } \end{array}
$$

Substituting this into (41) gives

$$
\begin{array} { l } { { \mathrm { S u b O p t } ( \widehat \pi ) \leq c \displaystyle \frac { \eta S A C _ { \eta } ^ { \star } H ^ { 2 } \iota ^ { 3 } \left( 1 + \log K \right) } { K } \sum _ { h = 1 } ^ { H } L _ { h } ^ { 3 } } } \\ { { \leq c \displaystyle \frac { \eta H ^ { 6 } S A C _ { \eta } ^ { \star } } { K } \iota ^ { 3 } ( 1 + \log K ) . } } \end{array}
$$

The second inequality uses $\begin{array} { r } { \sum _ { h = 1 } ^ { H } L _ { h } ^ { 3 } \leq H ^ { 4 } } \end{array}$ . This proves Theorem 4.1.

## References

Claire Chen and Yuheng Zhang. Fast rates in α-potential games via regularized mirror descent. arXiv preprint arXiv:2605.00268, 2026a.

Claire Chen and Yuheng Zhang. Pessimism-free ofline learning in general-sum games via kl regularization. arXiv preprint arXiv:2605.00264, 2026b.

Claire Chen, Yuheng Zhang, Xinyu Liu, Zixuan Xie, Shuze Daniel Liu, and Nan Jiang. Ofline twoplayer zero-sum markov games with KL regularization. In Forty-third International Conference on Machine Learning, 2026.

Eyal Even-Dar and Yishay Mansour. Learning rates for q-learning. Journal of machine learning Research, 5(Dec):1–25, 2003.

David A Freedman. On tail probabilities for martingales. the Annals of Probability, pp. 100–118, 1975.

Scott Fujimoto, David Meger, and Doina Precup. Of-policy deep reinforcement learning without exploration. In International conference on machine learning, pp. 2052–2062. PMLR, 2019.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International conference on machine learning, pp. 2160–2169. PMLR, 2019.

Haoyang Hong, Zichen Wang, Quanquan Gu, and Huazheng Wang. Online kl-regularized reinforcement learning with function approximation under misspecification. arXiv preprint arXiv:2606.06053, 2026.

Narim Jeong and Donghwan Lee. Finite-time error analysis of soft q-learning: Switching system approach. In 2024 IEEE 63rd Conference on Decision and Control (CDC), pp. 3897–3904. IEEE, 2024.

Narim Jeong and Donghwan Lee. Unified finite-time error analysis of soft q-learning. Neurocomputing, 626:129582, 2025.

Kaixuan Ji, Qiwei Di, Heyang Zhao, Qingyue Zhao, and Quanquan Gu. On the optimal sample complexity of ofline multi-armed bandits with kl regularization. arXiv preprint arXiv:2605.02141, 2026a.

Kaixuan Ji, Qingyue Zhao, Heyang Zhao, Qiwei Di, and Quanquan Gu. Near-optimal regret for kl-regularized multi-armed bandits. arXiv preprint arXiv:2603.02155, 2026b.

Yuchen Jiao, Jiin Woo, Gen Li, Gauri Joshi, and Yuejie Chi. Sample complexity of average-reward qlearning: From single-agent to federated reinforcement learning. arXiv preprint arXiv:2601.13642, 2026.

Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael Jordan. Is q-learning provably eficient? Advances in neural information processing systems, 31, 2018.

Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably eficient for ofline rl? In International conference on machine learning, pp. 5084–5096. PMLR, 2021.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for ofline reinforcement learning. Advances in neural information processing systems, 33:1179–1191, 2020.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Ofline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Sample complexity of asynchronous q-learning: Sharper analysis and variance reduction. Advances in neural information processing systems, 33:7031–7043, 2020.

Gen Li, Changxiao Cai, Yuxin Chen, Yuantao Gu, Yuting Wei, and Yuejie Chi. Tightening the dependence on horizon in the sample complexity of q-learning. In International Conference on Machine Learning, pp. 6296–6306. PMLR, 2021a.

Gen Li, Laixi Shi, Yuxin Chen, Yuantao Gu, and Yuejie Chi. Breaking the sample complexity barrier to regret-optimal model-free reinforcement learning. Advances in Neural Information Processing Systems, 34:17762–17776, 2021b.

Gen Li, Changxiao Cai, Yuxin Chen, Yuting Wei, and Yuejie Chi. Is q-learning minimax optimal? a tight sample complexity analysis. Operations Research, 72(1):222–236, 2024.

Xinyu Liu, Zixuan Xie, and Shangtong Zhang. Linear Q-learning does not diverge in L<sup>2</sup>: Convergence rates to a bounded set. In Forty-second International Conference on Machine Learning, 2025.

Anupam Nayak, Tong Yang, Osman Yagan, Gauri Joshi, and Yuejie Chi. Achieving logarithmic regret in kl-regularized zero-sum markov games. arXiv preprint arXiv:2510.13060, 2025.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Zhongjian Qiao, Jiafei Lyu, Boxiang Lyu, Yao Shu, Siyang Gao, and Shuang Qiu. Model-based ofline rl via robust value-aware model learning with implicitly diferentiable adaptive weighting. arXiv preprint arXiv:2603.08118, 2026.

Guannan Qu and Adam Wierman. Finite-time analysis of asynchronous stochastic approximation and q-learning. In Conference on learning theory, pp. 3185–3205. PMLR, 2020.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging ofline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems, 34:11702–11716, 2021.

Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. Pessimistic q-learning for ofline reinforcement learning: Towards optimal sample complexity. In International conference on machine learning, pp. 19967–20025. PMLR, 2022.

Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey Naumov, Pierre Perrault, Yunhao Tang, Michal Valko, and Pierre Menard. Fast rates for maximum entropy exploration. In International Conference on Machine Learning, pp. 34161–34221. PMLR, 2023.

Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Alexey Naumov, Pierre Perrault, Michal Valko, and Pierre Menard. Demonstration-regularized RL. In The Twelfth International Conference on Learning Representations, 2024.

Masatoshi Uehara and Wen Sun. Pessimistic model-based ofline reinforcement learning under partial coverage. arXiv preprint arXiv:2107.06226, 2021.

Masatoshi Uehara, Nathan Kallus, Jason D Lee, and Wen Sun. Ofline minimax soft-q-learning under realizability and partial coverage. Advances in Neural Information Processing Systems, 36: 12797–12809, 2023.

Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279–292, 1992.

Di Wu, Chengshuai Shi, Jing Yang, and Cong Shen. Greedy sampling is provably eficient for RLHF. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized ofline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019.

Jiachen Xi, Alfredo Garcia, and Petar Momčilović. Regularized q-learning with linear function approximation. IEEE Transactions on Automatic Control, 71(1):504–511, 2025.

Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for ofline reinforcement learning. Advances in neural information processing systems, 34:6683–6694, 2021a.

Tengyang Xie, Nan Jiang, Huan Wang, Caiming Xiong, and Yu Bai. Policy finetuning: Bridging sample-eficient ofline and online reinforcement learning. Advances in neural information processing systems, 34:27395–27407, 2021b.

Wei Xiong, Han Zhong, Chengshuai Shi, Cong Shen, Liwei Wang, and Tong Zhang. Nearly minimax optimal ofline reinforcement learning with linear function approximation: Single-agent mdp and markov game. arXiv preprint arXiv:2205.15512, 2022.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. arXiv preprint arXiv:2312.11456, 2023.

Yuling Yan, Gen Li, Yuxin Chen, and Jianqing Fan. The eficacy of pessimism in asynchronous q-learning. IEEE Transactions on Information Theory, 69(11):7185–7219, 2023.

Chenlu Ye, Rui Yang, Quanquan Gu, and Tong Zhang. Corruption-robust ofline reinforcement learning with general function approximation. Advances in Neural Information Processing Systems, 36:36208–36221, 2023.

Wenhao Zhan, Baihe Huang, Audrey Huang, Nan Jiang, and Jason Lee. Ofline reinforcement learning with realizability and single-policy concentrability. In Conference on Learning Theory, pp. 2730–2775. PMLR, 2022.

Dake Zhang, Boxiang Lyu, Shuang Qiu, Mladen Kolar, and Tong Zhang. Pessimism meets risk: Risk-sensitive ofline reinforcement learning. In Proceedings of the 41st International Conference on Machine Learning, pp. 59459–59489, 2024.

Sheng Zhang, Zhe Zhang, and Siva Theja Maguluri. Finite sample analysis of average-reward td learning and q-learning. Advances in Neural Information Processing Systems, 34:1230–1242, 2021.

Yuheng Zhang, Claire Chen, and Nan Jiang. Beyond pessimism: Ofline learning in kl-regularized games. arXiv preprint arXiv:2604.06738, 2026.

Heyang Zhao, Chenlu Ye, Quanquan Gu, and Tong Zhang. Sharp analysis for KL-regularized contextual bandits and RLHF. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Heyang Zhao, Chenlu Ye, Wei Xiong, Quanquan Gu, and Tong Zhang. Logarithmic regret for online kl-regularized reinforcement learning. arXiv preprint arXiv:2502.07460, 2025b.

Qingyue Zhao, Kaixuan Ji, Heyang Zhao, and Quanquan Gu. Fast rates for ofline contextual bandits with forward-kl regularization under single-policy concentrability. arXiv preprint arXiv:2605.09214, 2026a.

Qingyue Zhao, Kaixuan Ji, Heyang Zhao, Tong Zhang, and Quanquan Gu. Towards a sharp analysis of ofline policy learning for f-divergence-regularized contextual bandits. In The Fourteenth International Conference on Learning Representations, 2026b.