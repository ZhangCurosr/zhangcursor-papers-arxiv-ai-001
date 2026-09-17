# Online Robust Reinforcement Learning Through Monte-Carlo Planning

Tuan Dam <sup>\*</sup> <sup>1</sup> Kishan Panaganti <sup>\*</sup> <sup>2</sup> Brahim Driss <sup>\*</sup> <sup>3</sup> Adam Wierman <sup>2</sup>

## Abstract

Monte Carlo Tree Search (MCTS) is a powerful framework for solving complex decisionmaking problems, yet it often relies on the assumption that the simulator and the real-world dynamics are identical. Although this assumption helps achieve the success of MCTS in games like Chess, Go, and Shogi, the real-world scenarios incur ambiguity due to their modeling mismatches in low-fidelity simulators. In this work, we present a new robust variant of MCTS that mitigates dynamical model ambiguities. Our algorithm addresses transition dynamics and reward distribution ambiguities to bridge the gap between simulation-based planning and real-world deployment. We incorporate a robust power mean backup operator and carefully designed exploration bonuses to ensure finite-sample convergence at every node in the search tree. We show that our algorithm achieves a convergence rate of $\mathcal { O } ( n ^ { - 1 / 2 } )$ for the value estimation at the root node, comparable to that of standard MCTS. Finally, we provide empirical evidence that our method achieves robust performance in planning problems even under significant ambiguity in the underlying reward distribution and transition dynamics.

## 1. Introduction

Reinforcement learning (RL) provides a statistical machine learning framework to interact with the environments—such as autonomous vehicles, agile robots, and network systems—sequentially and learn to take control actions to achieve the desired objective. Monte Carlo Tree

Search (MCTS) algorithm, in conjunction with deep learning methods, solve complex decision-making problems in high-dimensional environments. Its celebrated success stories include autonomous RL decision-making agents playing board games Chess, Go, Shogi (Silver et al., 2016; Schrittwieser et al., 2020), Poker (Brown and Sandholm, 2018; Keshavarzi and Navidi, 2025), and solving various real-world challenging tasks like robotics and autonomous systems (Hoel et al., 2019; Kartal et al., 2019; Dam et al., 2022). MCTS offers a principled way to balance exploration and exploitation by using combinatorial search mechanisms derivedfrom online simulated trajectories. As a result, MCTS can effectively promote the exploration of promising regions of the environment with only partial modeling information of the environment.

However, most of these successes are limited to structured or simulated environments. As successful as RL algorithms are, an issue in applying them to real-world dynamical systems is the unavoidable discrepancy between the simulators and the actual real-world system dynamics. In traditional RL approaches (Kaelbling et al., 1996; Salvato et al., 2021), transition models are often learned from data collected by interacting with simulator models to avoid unsafe interactions with real-world systems, and reward models may be subject to stochasticity, hacked rewards, or unmodeled external factors. Such ambiguities arise from a variety of sources: limited training data, non-stationary environments, adversarial conditions, partial observability, or simply modeling simplifications. These factors can lead to a so-called simulation-to-reality gap, where the policy or value function that appears optimal in the simulated environment may perform poorly when deployed in the real world. A natural approach to addressing these challenges is to incorporate robustness against simulation-to-reality gaps directly into the planning algorithm.

RL agents making decisions under the framework of Robust Markov Decision Processes (RMDPs) (Iyengar, 2005; Nilim and El Ghaoui, 2005) offer a principled mechanism to conceptualize robustness against transition model and reward model mismatches raised by simulation-to-reality gaps. These robust RL agents explore policies that maximize expected returns under the worst-case model within a prescribed ambiguity set. The ambiguity set is typically constructed as a ball around the simulator dynamics or reward model, with the design choice of the ball size covering the real-world ground truth model descriptors. Recent works demonstrate their potential to achieve robust decision-making performance when faced with perturbations in transition dynamics and reward function models. However, while value iteration and policy optimization methods have been introduced and analyzed for robust RL, MCTS-based planning algorithms have not been explored, as per the authors’ knowledge. We discuss more detailed related works in Section 2.

In this work, we propose a novel robust MCTS algorithm equipped with non-asymptotic performance guarantees under model ambiguity set. Importantly, we incorporate both reward and transition ambiguity robustness, similar to recent works (Zhou et al., 2021; Wang et al., 2024b) in robust RL. In particular, our work resolves the following questions:

Can we use a search-based planning approach like MCTS to balance exploitation and exploration for the robust RL problem? What theoretical guarantee can we provide? Can we show robust performance against standard algorithms under the simulation-to-reality issue?

Our approach embeds the distributionally robust optimization (Rahimian and Mehrotra, 2019) mathematical principle into the MCTS framework, ensuring that the value estimates and action selections are robust to transitions and rewards drawn from the ambiguity sets. More precisely, we conceptualize a robust backup operator and design exploration bonuses that accommodate ambiguity sets defined using total variation, Kullback-Leibler, chi-squared, or Wasserstein measures. This allows MCTS to simultaneously use a tree search mechanism to solve for robust value estimates by trading off exploitation and exploration while achieving robust policies that work uniformly well across different models in the ambiguity set.

One of the key contributions of this work is the establishment of finite-sample bounds on the convergence rates of our robust MCTS algorithm. Viewing each node in the MCTS tree as a non-stationary bandit problem sheds light on the nontrivial challenges of controlling the interaction between ambiguity sets and exploration bonuses. More specifically, coming up with exploration bonuses (thereby robust value approximations) is nontrivial based on the non-linear backup operator due to the formalization of robustness. We overcome these challenges by building on a sequence of technical lemmas and applying concentration inequalities to the robust backup operator, we show that our method attains a convergence rate of order $\mathcal { O } ( n ^ { - 1 / 2 } )$ for robust value estimation at the root node, where n is the number of states visited while exploring the environment. This convergence rate also matches the best-known results for standard, non-robust MCTS, thereby demonstrating that introducing robustness need not change the convergence speed in terms of the number of samples.

Contributions. In this work, to the best of our knowledge, we are the first to propose an MCTS-based algorithm for the robust RL problem. Our contributions are threefold:

• Robust MCTS Algorithm: We solve the online robust RL problem–accounting for model ambiguity in both transitions and rewards–using a planning algorithm enabled by MCTS. This fundamental first step paves the way for future applications in large-scale dynamical systems.

• Non-Asymptotic Guarantees: We provide rigorous finitesample performance bounds, ensuring that the robust MCTS converges with a known rate, on par with standard MCTS. Our analysis leads to novel exploration bonuses that arise from careful analyses of robust backup operators and the tree search mechanism by recasting robust MCTS for different ambiguity sets as a collection of nonstationary multi-armed bandit problems.

• Robust Empirical Performance: We conduct experiments in two environments (Gambler’s Problem and Frozen Lake) to evaluate our robust algorithm, demonstrating that it achieves superior robust performance to model mismatches than the standard MCTS algorithm baseline.

## 2. Related Works

Robust RL. Robust RL agents make decisions to alleviate environmental ambiguities under the RMDP framework introduced by Iyengar (2005); Nilim and El Ghaoui (2005) considers distributional robust optimization (Rahimian and Mehrotra, 2019) mathematical formularization. Many recent works extensively study the robust RL problem, addressing multiple aspects of the challenges of decisionmaking learning algorithms. Panaganti and Kalathil (2021); Zhou et al. (2021); Panaganti and Kalathil (2022); Shi and Chi (2024) propose model-based dynamic programming algorithms to solve the robust RL problem for finite state-action environments, and Dong et al. (2022); Panaganti et al. (2025) extend to the online and offline settings, respectively. These works focus on addressing the sample complexity—minimal samples needed from the simulator model (leading to the construction of an approximate model) for every state-action pair to obtain an approximate value estimation—issue. Panaganti and Kalathil (2021); Panaganti et al. (2022); Zhang et al. (2023) propose modelfree value function approximation-based robust RL algorithms utilizing special structures in the Bellman backups arising due to specific forms of ambiguity sets. Different from these approaches, our algorithm is inspired by MCTS to solve the robust RL problem. MCTS scales well (Silver et al., 2016) for large problems by embedding strong search mechanisms into model-based planning approaches in RL.

MCTS for non-robust RL. AlphaGo-like (Silver et al., 2016) agents are powered by tree search mechanisms such as MCTS in traditional dynamic programming planning for standard RL. Kocsis and Szepesvari (2006); Shah et al.´ (2020); Dam et al. (2024b) provide theoretical guarantees for such heuristic search-based deep RL algorithms. Recently, the adoption of MCTS (Swiechowski et al., 2023) in<sup>´</sup> other learning settings has seen scaling advantages. For instance, in non-standard RL settings, like supervised learning systems (Guez et al., 2018; Wang et al., 2024a), constrained dynamical systems (Parthasarathy et al., 2023; Kurecka et al., 2024) to promote safe decision-makingˇ choices, and partially observable and constrained dynamical systems (Lee et al., 2018; Dam et al., 2022; 2020). In bandits, like agents taking decisions in the space of contexts (Ontanon, 2013; Mao et al., 2020). In applica-´ tions, like autonomous vehicles and robots, (Kartal et al., 2019; Yin et al., 2022) where the imitation of expert decisions plays a critical role. Alternative approaches include entropy regularization methods like MENTS (Xiao et al., 2019), RENTS and TENTS (Dam et al., 2021; 2024a), and Boltzmann-based approaches (Painter et al., 2023), though these rely on temperature parameters that may impede convergence to true optimal values. Inspired by such adoption of MCTS, we enable MCTS-based planning for the first time to the robust RL problem—equipped with theoretical guarantees—that accounts for mitigating dynamical model ambiguities.

Search-based planning for online robust RL. This line of research is closest to ours in terms of search-inspired algorithms. (Liu et al., 2022; Wang et al., 2023; Wang, 2024) introduces the Multi-Level Monte Carlo (MLMC) method (Heinrich, 2001; Giles, 2008) to approximate the robust Bellman backups. MLMC is another powerful statistical sampling method from the family of Monte Carlo estimators. However, they have the drawback of requiring random sampling procedures in each iteration of the robust RL planning stages for every state-action pair. By avoiding these pitfalls, MCTS adapts to the online sampling procedure by enabling search from a tree node—states and actions in dynamical systems—up to some constant depth in the tree. Other works introduce sampling-based Q-learning (Zhou et al., 2021; Liu et al., 2022; Wang et al., 2024b) and policy iteration (Panaganti and Kalathil, 2021; Kumar et al., 2023; Badrinath, 2023) inspired approaches. These are popular methods in standard online RL enabling trajectory-based updates—at current states, actions, and next states sampled with an updated policy—to approximate the Bellman backups. However, these require algorithmic and theoretical innovations–for e.g., function approximation architectures–for scaling up to high-dimensional dynamical systems (Panaganti et al., 2022; Zhang et al., 2023; Panaganti et al., 2024; Liu and Xu, 2024). The incorporation of the strong sampling procedure by MCTS avoids this issue.

## 3. Preliminaries

A Markov Decision Process (MDP) specified by the tuple $( \cal { S } , \cal { A } , \cal { P } , \cal { R } )$ , where $S \subset \mathbb { R } ^ { d }$ is the (potentially large) state space, A is a discrete action space, $P : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ is the transition model mapping each state–action pair to a probability distribution over next states, and $R : { \mathcal { S } } \times { \mathcal { A } } $ R is the (possibly uncertain) reward function assumed to be supported on a bounded interval $[ 0 , R _ { \mathrm { m a x } } ]$ . A stationary policy $\pi \in \Pi ( \mathbf { M } )$ is defined as $\pi : S  \Delta ( { \mathcal { A } } )$ , meaning that at each discrete time step t, the agent observes a state $s _ { t } ,$ samples an action $a _ { t } \ \sim \ \pi ( \cdot \ | \ s _ { t } )$ , collects a reward $\boldsymbol { r } _ { t } \sim R ( \cdot \ | \ s _ { t } , \boldsymbol { a } _ { t } )$ , and transitions to $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , a _ { t } )$ We mention detailed notations used in this work in Table 3.

## 3.1. Value Functions and Policies

We adopt a discounted formulation with discount factor $\gamma \in ( 0 , 1 )$ . The state-value and state–action value functions of a policy π are given by

$$
\begin{array} { r } { V _ { P , R } ^ { \pi } ( s ) \ = \ \sum _ { t = 0 } ^ { \infty } \mathbb { E } _ { a _ { t } \sim \pi } \big [ \gamma ^ { t } r _ { t } \big \vert s _ { 0 } = s \big ] , } \end{array}\tag{1}
$$

$$
\begin{array} { r } { Q _ { P , R } ^ { \pi } ( s , a ) \ = \ \sum _ { t = 0 } ^ { \infty } \mathbb { E } _ { a _ { t } \sim \pi } \big [ \gamma ^ { t } r _ { t } \big | s _ { 0 } = s , \ a _ { 0 } = a \big ] . } \end{array}\tag{2}
$$

The optimal state-value function is defined as $V _ { P , R } ^ { \star } ( s ) ~ =$ $\operatorname* { s u p } _ { \pi } V _ { P , R } ^ { \pi } ( s )$ . By definition and existence of deterministic optimal actions, the optimal state–action value function $Q ^ { * }$ satisfies $V _ { P , R } ^ { \star } ( s ) = \operatorname* { m a x } _ { a } Q _ { P , R } ^ { \star } ( s , a )$ for each $s \in S$

## 3.2. Conceptualization of Robustness

A key challenge in real-world RL is that both transitions $P$ and rewards R may be partially unknown or even timevarying. Let $P ^ { o }$ and $\nu ^ { o }$ denote the nominal transition probabilities and reward distributions, respectively, with each reward $r ( s , a ) \sim \nu _ { s , a } ^ { o }$ . These nominal models can be either factory-set approximations or a simulator of real-world systems. Following Wang et al. (2024b); Zhou et al. (2021); Liu et al. (2022), we allow the environment to deviate from $\left( P ^ { o } , \nu ^ { o } \right)$ within a robustness budget $\rho _ { \mathrm { T } } , \rho _ { \mathrm { R } }$ respectively. This leads to a robust MDP that accounts for uncertainties in both transitions and rewards.

Ambiguity Sets. We model transitions in an ambiguity set $\mathcal { P } = \otimes _ { ( s , a ) } \mathcal { P } _ { s , a }$ , where each $\mathcal { P } _ { s , a }$ contains all plausible distributions over next states from $( s , a )$ . Analogously, an ambiguity set ${ \mathcal R } \ = \ \otimes _ { ( s , a ) } \ { \mathcal R } _ { s , a }$ captures deviations in the reward distributions $r ( s , a )$ . Here, with a chosen metric

D(·, ·),

$$
\mathcal { P } _ { s , a } = \Big \{ P _ { s , a } \in \Delta ( \mathcal { S } ) \colon D \big ( P _ { s , a } , P _ { s , a } ^ { o } \big ) \leq \rho _ { \mathrm { T } } \Big \} ,
$$

and

$$
\mathcal { R } _ { s , a } = \Big \{ \nu _ { s , a } \colon D \big ( \nu _ { s , a } , \nu _ { s , a } ^ { o } \big ) \leq \rho _ { \mathrm { R } } \Big \} .
$$

Different choices of $D$ lead to distinct ambiguity sets, such as total-variation balls $( \mathcal { P } ^ { \mathrm { T V } } )$ , chi-squared neighborhoods $( { \mathcal { P } } ^ { X } )$ , or Wasserstein sets $( \mathcal { P } ^ { W } )$ . For notational convenience, we denote the reward distributions $\nu _ { s , a } \in \mathcal { R } _ { s , a }$ also as their probability densities in the context of measuring distances $D ( \cdot , \cdot )$

## 4. Main Problem Formulation

This section establishes how Monte Carlo Tree Search (MCTS) can be adapted to account for model ambiguity in a robust Markov Decision Process (MDP). Our goal is twofold: first, to clarify the root assumptions behind the robust planning framework, and second, to describe how MCTS is modified so that each node’s value estimate incorporates worst-case rewards and transitions.

Robust MDP. We consider a robust MDP M = $( \mathcal { S } , \mathcal { A } , \mathcal { P } , \mathcal { R } )$ in which the state space S may be large or partially continuous, the action space A is discrete, and the unknown reward $r ( s , a )$ and transition model $\mathcal { P } ( \cdot \mathrm { ~ \bf ~ \mathscr ~ { ~ \bf ~ \mathscr ~ { ~ s ~ } ~ } ~ } , a )$ can lie within an ambiguity set R and $\mathcal { P }$ (described in Section 3). At each step t, the agent observes a state $s _ { t } ,$ selects an action $a _ { t } \in \mathsf { \Gamma } A ,$ receives reward $r _ { t } ,$ and transitions to a new state $s _ { t + 1 }$ . The robust state-value and state-action value functions of a policy π are given by $V ^ { \pi } ( s ) =$ min<sub>P∈P,R∈R</sub> $V _ { P , R } ^ { \pi } ( s )$ and $\begin{array} { r } { Q ^ { \pi } ( s , a ) = \operatorname* { m i n } _ { P \in \mathcal { P } , R \in \mathcal { R } } Q _ { P , R } ^ { \pi } ( s , a ) } \end{array}$ respectively. A policy $\pi ^ { \star }$ that maximizes the value function is an optimal robust policy with corresponding optimal robust value functions $V ^ { \star } ( s )$ and $Q ^ { \star } ( s , a )$ . Hence, both transitions and rewards may be adversarially perturbed, ensuring the agent plans robustly for worst-case scenarios within these sets.

Robust Bellman Operator. In the robust MDP, the worst-case expected value arises from an adversarial choice of both transition and reward distributions within their respective ambiguity sets. From the robust MDP literature (Iyengar, 2005; Liu et al., 2022), by the construction of $\mathcal { P }$ and R ambiguity sets, $Q ^ { \star }$ is known to be computable, and thereby $\pi ^ { \star } ( s ) = \mathrm { a r g m a x } _ { a \in \mathcal { A } } Q ^ { \star } ( s , a )$

Let us define for any set B and a vector v, $ \mathbf { \nabla } , ~ \sigma _ { \mathbf { B } } ( v ) ~ =$ inf $\{ u ^ { T } v : u \in B \}$ . Robust dynamic programming, given by $V _ { k + 1 } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { k + 1 } ( s , a )$ and

$$
Q _ { k + 1 } ( s , a ) \ = \ R _ { s , a } ^ { \mathrm { r o b } } \ + \ \gamma \sigma _ { \mathscr P _ { s , a } } ( V _ { k } ) ,
$$

where $\begin{array} { r } { R _ { s , a } ^ { \mathrm { r o b } } ~ = ~ \operatorname* { m i n } _ { r _ { s , a } \in \mathcal { R } _ { s , a } } \mathbb { E } _ { R \sim r _ { s , a } } [ R ] } \end{array}$ , and $\sigma _ { \mathcal { P } _ { s , a } } ( V )$ captures the worst-case expected reward at $( s , a )$ and value of $V$ over $\mathcal { P } _ { s , a }$ , converges to optimal robust value functions $V ^ { * }$ and $Q ^ { * }$ respectively.

MCTS in a Robust MDP. In Monte Carlo Tree Search, we approximate a γ-discounted solution by simulating trajectories down a growing search tree. Each node corresponds to a state $s _ { h }$ , with h indicating the depth in the tree (distance from the root). From $s _ { h } .$ , the algorithm either expands a child node for the next state $s _ { h + 1 }$ or performs a rollout using a simpler policy $\pi _ { 0 }$ if h reaches the maximum search depth H. Trajectories terminate upon reaching depth H or a terminal state.

Performance Measure. A canonical metric for MCTS algorithms is the convergence rate $r ( t )$ , where t indexes the number of simulated trajectories (rollouts). Informally, $r ( t )$ bounds how quickly the MCTS estimates approach the true optimal values at the root node. For instance, one may require that $\mathbb { E } \big [ V ^ { \star } ( s _ { 0 } ) - Q ^ { \star } ( s _ { 0 } , \widehat { a } _ { t } ) \big ] \ \leq \ r ( t )$ , or

$$
\big | \mathbb { E } \big [ V ^ { \star } ( s _ { 0 } ) - \widehat { V } _ { t } ( s _ { 0 } ) \big ] \big | \leq r ( t ) ,
$$

where $\widehat { \boldsymbol { a } } _ { t }$ is the action chosen at the root after t rollouts, and $\widehat { V } _ { t } ( s _ { 0 } )$ approximates $V ^ { \star } ( s _ { 0 } )$

Recursive Value Estimation Under Ambiguity. To capture the robust (worst-case) aspect of the MDP, we define a recursive estimation scheme at each node that accounts for $\operatorname* { i n f } _ { r ( s , a ) \in { \mathcal { R } } _ { s , a } }$ of reward and in $\mathrm { f } _ { P \in \mathcal { P } _ { s , a } }$ transitions. Let $s _ { h }$ be a node at depth h. We assign a robust value $\widetilde V ( s _ { h } )$ and a robust action-value $\widetilde { Q } ( s _ { h } , a )$ such that

$$
\begin{array} { r l } & { \widetilde { Q } ( s _ { h } , a ) = R _ { s , a } ^ { \mathrm { r o b } } + \gamma \sigma _ { \mathscr P _ { s _ { h } , a } } ( \widetilde V ) , } \\ & { \qquad \widetilde V ( s _ { h } ) = \displaystyle \operatorname* { m a x } _ { a \in \mathcal A } \widetilde Q ( s _ { h } , a ) . } \end{array}
$$

At a leaf node $( h = H )$ , we approximate the value with a simple rollout policy $\pi _ { 0 } ,$ yielding $\widetilde V ( s _ { H } ) \approx V _ { \pi _ { 0 } } ( s _ { H } )$

Goal of MCTS. Since finite sample sizes introduce noise, each node’s robust value $\widetilde V ( s _ { h } )$ is estimated from rollouts. The ultimate objective is to identify an action $a _ { \star } = \arg \operatorname* { m a x } _ { a } Q ^ { \star } ( s _ { 0 } , a )$ at the root state $s _ { 0 }$ within n simulated trajectories, where $Q ^ { \star } ( s _ { 0 } , a )$ represents the robustoptimal action value. Intuitively, we want:

$$
\widehat { a } _ { n } \approx \arg \operatorname* { m a x } _ { a } \widetilde Q ( s _ { 0 } , a ) , \widehat V _ { n } ( s _ { 0 } ) \approx \widetilde V ( s _ { 0 } ) ,
$$

with small statistical error. In Section 5, we describe how Robust-Power-UCT achieves this via specially designed backup operators and action-selection rules. Section 6 establishes finite-sample guarantees, showing that robustness in MCTS need not degrade convergence speed compared to its non-robust counterpart.

## 5. Algorithm Description

We now describe the core parts of our Robust-Power-UCT algorithm, focusing on the value backup and action selection strategies. Other details, such as the main loop and rollout procedure, are standard MCTS routines and hence only briefly mentioned.

Table 1: Key Conditions for Algorithmic Constants $( i \in$ $[ 0 , H ] )$  
Cond. Requirement   
(1) $b _ { i } < \alpha _ { i }$ and $b _ { i } > 2 .$   
$\lceil 1 \leq p \leq 2$ and $\begin{array} { r } { \alpha _ { i } \leq \frac { \beta _ { i } } { 2 } } \end{array}$   
(2) or   
(p > 2 and $\begin{array} { r } { \alpha _ { i } \leq \frac { \beta _ { i } } { 2 } , 0 < \alpha _ { i } - \frac { \beta _ { i } } { p } < 1 } \end{array}$   
(3) $\begin{array} { r } { \alpha _ { i } \Big ( 1 - \frac { b _ { i } } { \alpha _ { i } } \Big ) \ \leq \ b _ { i } \ < \ \alpha _ { i } . } \end{array}$   
(4) $\begin{array} { r } { \alpha _ { i } \ = \ ( b _ { i + 1 } - 1 ) \Big ( 1 - \frac { b _ { i + 1 } } { \alpha _ { i + 1 } } \Big ) } \end{array}$   
(5) $\beta _ { i } = ( b _ { i + 1 } - 1 ) .$

Value Backup. To estimate the value function at each node, we use a power mean backup operator. When node $s _ { h }$ is expanded in the tree, we define inductively for all $t ,$

$$
\widehat V _ { t } ( s _ { h } ) = \biggl ( \sum _ { a \in { \mathcal A } _ { s _ { h } } } \frac { T _ { s _ { h } , a } ( t ) } t \left[ \widehat Q _ { T _ { s _ { h } , a } ( t ) } ( s _ { h } , a ) \right] ^ { p } \biggr ) ^ { \frac 1 p } ,
$$

where $p \geq 1$ . This power mean backup places more emphasis on actions that have high current value estimates (when $p > 1 )$ , but still captures the contributions of other actions. Meanwhile $\widehat { Q } _ { T _ { s _ { h } , a } ( t ) } ( s _ { h } , a )$ , or simply $\widehat { Q } _ { t } ( s _ { h } , a )$ as the root is $s _ { h } .$ , itself is updated via

$$
\widehat { Q } _ { t } ( s _ { h } , a ) = \widehat { R } _ { s _ { h } , a } ^ { \mathrm { r o b } } + \gamma \sigma _ { \widehat { \mathcal { P } } _ { s _ { h } , a } } \bigl ( \widehat { V } _ { T _ { s _ { h + 1 } } ( t ) } \bigr ) ,\tag{3}
$$

where $\begin{array} { r } { \widehat { R } _ { s _ { h } , a } ^ { \mathrm { r o b } } = \operatorname* { m i n } _ { r \in \widehat { \mathcal { R } } _ { s _ { h } , a } } \mathbb { E } _ { R \sim r } [ R ] } \end{array}$ is an empirical robust reward at $( s _ { h } , a )$ , and $\sigma _ { \widehat { \mathcal { P } } } ( \cdot )$ is a robust operator capturing worst-case transitions for ambiguity sets governed by empirical estimates of nominal reward and transition models:

$$
\begin{array} { r } { \widehat { \mathcal { P } } _ { s , a } = \Big \{ P _ { s , a } \in \Delta ( S ) \colon D \big ( P _ { s , a } , \widehat { p } _ { s , a } \big ) \leq \rho _ { \mathrm { T } } \Big \} , } \end{array}
$$

and

$$
\begin{array} { r } { \widehat { \mathcal { R } } _ { s , a } = \Big \{ \nu _ { s , a } \in \Delta ( B ) \colon D \big ( \nu _ { s , a } , \widehat { \nu } _ { s , a } \big ) \leq \rho _ { \mathrm { R } } \Big \} . } \end{array}
$$

Action Selection. At each node $s _ { h }$ in the search tree, Robust-Power-UCT selects an action a according to an optimistic rule that balances exploration and exploitation.

Specifically, we maintain an empirical estimate $\widehat { Q } _ { t } ( s _ { h } , a )$ for each action and add an exploration bonus of the form:

$$
C \cdot \frac { \left( T _ { s _ { h } } ( t ) \right) ^ { \frac { b _ { h + 1 } } { \beta _ { h + 1 } } } } { \left( T _ { s _ { h } , a } ( t ) \right) ^ { \frac { \alpha _ { h + 1 } } { \beta _ { h + 1 } } } } ,
$$

where $T _ { s _ { h } } ( t )$ is the total number of visits to $s _ { h }$ up to time t, and $T _ { s _ { h } , a } ( t )$ is how often action a has been taken from $s _ { h }$ . The exponents $\frac { b _ { h + 1 } } { \beta _ { h + 1 } }$ and $\frac { \alpha _ { h + 1 } } { \beta _ { h + 1 } }$ control how aggressively the algorithm explores, while C is a user-chosen constant. At the end of training (greedy mode), the action with the highest $\widehat { Q } _ { t }$ is chosen.

Main Loop and Rollout. As in standard MCTS, the algorithm repeatedly simulates from the root state $s _ { 0 } .$ , selecting actions according to the above scheme. When reaching a leaf node (unexpanded or maximum depth), a rollout policy approximates the return from that leaf. These routines are routine and can be implemented similarly to classical MCTS methods.

By combining an optimistic action selection mechanism with a power mean and robust operator for value backup, Robust-Power-UCT systematically balances exploration of uncertain actions and exploitation of promising ones, all under model ambiguity.

## 6. Theoretical Results

In robust MCTS planning, each internal node of the search tree can be viewed as a non-stationary multi-armed bandit due to ongoing updates of the node’s reward and transition ambiguity estimates. At each step, the empirical evaluations shift, reflecting how robust exploration is balanced against uncertainty in the model. To handle this dynamic process, we begin by studying a non-stationary multiarmed bandit problem—focusing on how the power-mean backup operator concentrates around its robust-optimal value. We then leverage these results to prove convergence properties of our robust MCTS algorithm, showing that it systematically discards suboptimal branches under model uncertainty while maintaining sample efficiency.

## 6.1. Non-Stationary Bandit Perspective

We first analyze Robust-Power-UCT in a simpler nonstationary multi-armed bandit setting. Here, actions are selected optimistically, and the power mean backup operator is used at the root node.

## 6.1.1. PROBLEM DESCRIPTION AND KEY DEFINITIONS

We consider a class of non-stationary multi-armed bandit (MAB) problems with $K \geq 1$ actions (arms) with the reward $\in [ 0 , R ]$ . Define a sequence of estimator $\widehat { \mu } _ { a , n }$ (in this paper is the robustness estimation of the mean value of arm a) such that $\begin{array} { r c l } { \mu _ { a , n } } & { = } & { \mathbb { E } \big [ \widehat { \mu } _ { a , n } \big ] } \end{array}$ . We are interested in sequence of estimators that satisfy a suitable concentration property:

Algorithm 1: Robust-Power-UCT with $\gamma$ discount factor. n : the number of rollouts. $\{ b _ { i } , \alpha _ { i } , \beta _ { i } \} _ { i = 0 } ^ { H }$ are positive con  
stants that satisfy conditions in Table 1. B is the total bins for $[ 0 , R _ { \mathrm { m a x } } ]$ . π is a rollout policy. $C$ is an exploration constant.   
Input: root node state s<sub>0</sub>   
Output: optimal action at the root node   
Function $R = \mathbb { R } \circ \mathbb { 1 } \mathbb { 1 } \circ \mathrm { u t } \ ( s , d e p t h )$ Function $\mathtt { S i m u l a t e Q } \left( s _ { h } , a , d e p t h = h , t \right)$   
$\widetilde V ( s )$ = average of the call to $\pi _ { 0 } ( s )$ $s _ { h + 1 } \sim P ^ { o } ( \cdot | s _ { h } , a )$   
return $\widetilde V ( s )$ $r ( s _ { h } , a ) \sim \nu _ { s _ { h } , a } ^ { o }$   
Function a = SelectAction $( s _ { h } , d e p t h ~ = ~ h , g r e e d y ~ =$ if $s _ { h + 1 } \notin T$ erminal and $d e p t h \leqslant H - 1$ then   
f alse, t) if Node $s _ { h + 1 }$ not expanded then   
if greedy == false then $\widehat { V } _ { T _ { s _ { h + 1 } } ( t ) } ( s _ { h + 1 } ) = \mathop { \mathrm { R o l 1 o u t } } \left( s _ { h + 1 } , d e p t h \right)$   
<sup>b</sup>h+1 else   
a = argmax $\begin{array} { r } { \{ \widehat { Q } _ { T _ { s _ { h } , a } ( t ) } ( s _ { h } , a ) + C \frac { T _ { s _ { h } } ( t ) ^ { \beta _ { h + 1 } } } { \frac { \alpha _ { h + 1 } } { \sigma } } \} } \end{array}$ SimulateV (s<sub>h+1</sub>, depth = h + 1, t)   
a $T _ { s _ { h } , a } ( t ) ^ { \overline { { { \beta _ { h + 1 } } } } }$ end   
else end   
$a = \underset { \alpha } { \operatorname { a r g m a x } } \{ \widehat { Q } _ { T _ { s _ { h } , a } ( t ) } ( s _ { h } , a ) \}$ Find $j \in B s . t . r ( s _ { h } , a ) \in B i n ^ { j } [ 0 , R _ { \mathrm { m a x } } ]$   
a $\begin{array} { r } { \widehat { \nu } _ { s _ { h } , a } ( j ) = \frac { \widehat { \nu } _ { s _ { h } , a } ( j ) \cdot T _ { s _ { h } , a } ( t ) + 1 } { T _ { s _ { h } , a } ( t ) + 1 } } \end{array}$   
end   
return a $\begin{array} { r } { \widehat { p _ { s _ { h } , a } } \big ( s _ { h + 1 } \big ) = \frac { \widehat { p } _ { s _ { h } , a } ( \tilde { s _ { h + 1 } } ) \cdot T _ { s _ { h } , a } ( t ) + 1 } { T _ { s _ { h } , a } ( t ) + 1 } } \end{array}$   
Function SimulateV a ← SelectAction(s<sub>h</sub>, de $( s _ { h } , d e p t h , t )$ $\displaystyle { \mathcal { R } } h = h ,$ greedy = false, t) $T _ { s _ { h } , a } ( t ) \gets T _ { s _ { h } , a } ( t ) + \ddot { 1 }$   
$\mathtt { S i m u l a t e Q } \left( s _ { h } , a , d e p t h = h , t \right)$ $\widehat { Q } _ { T _ { s _ { h } , a } ( t ) } \big ( s _ { h } , a \big ) \gets \widehat { R } _ { s _ { h } , a } ^ { \mathrm { r o b } } + \gamma \sigma _ { \widehat { p } _ { s _ { h } , a } } \big ( \widehat { V } _ { T _ { s _ { h + 1 } } ( t ) } \big )$   
$T _ { s _ { h } } ( t ) \gets T _ { s _ { h } } ( t ) + 1$ Function MainLoop   
$\begin{array} { r } { \widehat { V } _ { T _ { s _ { h } } ( t ) } ( s _ { h } ) \gets \left( \sum _ { a } \frac { T _ { s _ { h } , a } ( t ) } { T _ { s _ { h } } ( t ) } ( \widehat { Q } _ { T _ { s _ { h } , a } ( t ) } ( s _ { h } , a ) ) ^ { p } \right) ^ { \frac { 1 } { p } } } \end{array}$ For $t = 0 , \cdots , \bar { n }$   
Simulate $\mathrm { ~ V ~ } ( s _ { 0 } , d e p t h = 0 , t )$   
return SelectAction (s<sub>0</sub>, greedy = true, n)

Definition 1 (Concentration). A sequence of estimators $\{ \widehat { Y } _ { n } \} _ { n \geq 1 }$ concentrates at rate $( \alpha , \beta )$ toward a limit Y, writing as ${ \widehat { Y } } _ { n } \ { \underset { n  \infty } { \alpha } } Y ,$ , if there is a constant $c > 0$ such that

$$
\forall n \geq 1 , \forall \varepsilon > n ^ { - \frac { \alpha } { \beta } } , \mathbf { P r } \Big ( \vert \widehat { Y } _ { n } - Y \vert > \varepsilon \Big ) \leq c n ^ { - \alpha } \varepsilon ^ { - \beta } .
$$

Assumption 1 (Non-Stationary Rewards). For each arm $a \ \in \ [ K ]$ , the sequence $\{ \widehat { \mu } _ { a , n } \} _ { n \geq 1 }$ concentrates at rate $( \alpha , \beta )$ toward a value $\mu _ { a } ,$ , i.e. $\widehat { \mu } _ { a , n } \underset { n  \infty } { \overset { \alpha , \beta } { \longrightarrow } } \mu _ { a }$ . Let $\mu _ { \star } ~ =$ $\operatorname* { m a x } _ { a \in [ K ] } \left\{ \mu _ { a } \right\}$ , assumed to be unique with a strict gap from suboptimal $\mu _ { a }$

## 6.1.2. OPTIMISTIC ACTION SELECTION AND POWER MEAN BACKUP

Under Assumption 1, we use an optimistic exploration rule similar to Robust-Power-UCT. Let $T _ { a } ( n )$ be the number of times arm a is pulled before time n. The algorithm pulls each arm once initially. For $n > K$

$$
a _ { n } \ = \ \arg \operatorname* { m a x } _ { a \in [ K ] } \biggl \{ \widehat { \mu } _ { a , T _ { a } ( n ) } \ + \ C \ : n ^ { \frac { b } { \beta } } / T _ { a } ( n ) ^ { \frac { \alpha } { \beta } } \biggr \} ,\tag{4}
$$

where $b \_ \_ \_ \_ \ \ \_ \ 2$ and $b \_ { \alpha }$ . For the power mean operator, let $\begin{array} { r l r } { p } & { { } \in } & { [ 1 , \infty ) } \end{array}$ and define $\begin{array} { r l r l } { \widehat { \mu } _ { n } ( p ) } & { { } } & { = } & { { } } \end{array}$ $\begin{array} { r } { \left( \sum _ { a = 1 } ^ { K } \frac { T _ { a } ( n ) } { n } \left[ \widehat { \mu } _ { a , T _ { a } ( n ) } \right] ^ { p } \right) ^ { \frac { 1 } { p } } } \end{array}$ . By applying Theorem 1 of Dam et al. (2024b), we get $\widehat { \mu } _ { n } ( p ) \underset { n  \infty } { \alpha ^ { \prime } , \beta ^ { \prime } } \ \mu _ { \star }$ , where $\alpha ^ { \prime } =$ $\begin{array} { r } { ( b - 1 ) \left( 1 - \frac { b } { \alpha } \right) } \end{array}$ , and $\beta ^ { \prime } = ( b - 1 )$

Connecting Back to MCTS. This bandit analysis underpins how Robust-Power-UCT handles exploration and the power mean backup. In an MCTS context, each node’s local bandit analysis is augmented by worst-case backups, but the principle is similar: the algorithm discards suboptimal branches with high probability, causing the robust estimates to concentrate around the best actions.

## 6.1.3. MAIN CONVERGENCE RESULTS

Before presenting the main result (Theorem 3), we first show an important lemma used for our MCTS algorithm.

Lemma 17. For $m \in [ M ]$ , let $( \widehat { V } _ { m , n } ) _ { n \geqslant 1 }$ be a sequence of estimator satisfying $\widehat { V } _ { m , n } \underset { n  \infty } { \stackrel { \alpha , \beta } { \longrightarrow } } V _ { m } ,$ , and there exists a constant L such that $\widehat { V } _ { m , n } \ \leqslant \ L , \forall n \ \geqslant \ 1$ . Let $X _ { i }$ be an iid sequence from a distribution $\nu ^ { o }$ with mean $\mu$ and $S _ { i }$ be an iid sequencefrom a distribution $p = ( p _ { 1 } , \dotsc , p _ { M } )$ supported on $\{ 1 , \dots , M \}$ . Introducing the random variables $N _ { m } ^ { n } = \# | \{ i \leqslant n : S _ { i } = s _ { m } \} |$ . Define a model estimate of p as $\begin{array} { r } { \widehat { p } _ { n } = \big ( \frac { N _ { 1 } ^ { n } } { n } , \frac { N _ { 2 } ^ { n } } { n } , . . . , \frac { N _ { M } ^ { n } } { n } \big ) } \end{array}$ . We define an estimate $o f \nu ^ { o }$ as $\begin{array} { r c l } { { \widehat { \nu } _ { n } } } & { { = } } & { { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i } } } } \end{array}$ . Recall $\begin{array} { r } { R ^ { \mathrm { r o b } } = \operatorname* { m i n } _ { r \in { \mathcal R } } { \mathbb E } _ { R \sim r } [ R ] } \end{array}$ w.r.t $\nu ^ { o }$ and $\begin{array} { r } { \widehat { R } ^ { \mathrm { r o b } } = \operatorname* { m i n } _ { r \in \widehat { \mathcal { R } } } \mathbb { E } _ { R \sim r } [ R ] } \end{array}$ w.r.t $\widehat { \nu } _ { n }$ . We define

the sequence of estimators

$$
\widehat { Q } _ { n } = \widehat { R } ^ { \mathrm { r o b } } + \gamma \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) .
$$

Then with 2α $\leqslant \beta , \beta > 1 , \widehat { Q } _ { n } \underset { n  \infty } { \overset { \alpha , \beta } { \longrightarrow } } R ^ { \mathrm { r o b } } + \gamma \sigma _ { p } ( V )$

Remark 1. This non-asymptotic convergence result shows that, for suitable parameters $( \alpha , \beta )$ , the estimator ${ \widehat { Q } } _ { n }$ will concentrate around the limiting quantity $R ^ { \mathrm { r o b } } + \gamma \sigma _ { p } ( V )$ with high probability. Importantly, we do not claim these $( \alpha , \beta )$ are in any sense optimal; rather, we only need the existence of such parameters that guarantee the concentration at the prescribed rate. Moreover, our analysis uses covering number generalization to handle continuous reward distributions. Furthermore, the constant c implicit in the notation $\widehat { V } _ { n } \underset { n  \infty } { \xrightarrow [ { \alpha , \beta } ] { \alpha , \beta } }$ V can depend on problem-dependent factors (e.g., size of the action set A, number of states $S ,$ etc.), reflecting the stochastic process complexity.

## 6.2. Tree-Level Convergence

The above non-stationary bandit analysis is critical for proving the subsequent tree-level theorems. In particular, Theorem 2 (restated below) shows that under appropriate parameter settings (Table 1), the estimated node values $\widehat { V } _ { n } ( \cdot )$ and $\widehat { Q } _ { n } ( \cdot , \cdot )$ converge at a known rate:

Theorem 2. When applying Robust-Power-UCT with parameters $\{ b _ { i } \} _ { i = 0 } ^ { H } , \ \bar { \{ \alpha _ { i } \} } _ { i = 0 } ^ { H } , \ \{ \beta _ { i } \} _ { i = 0 } ^ { H }$ satisfying Table 1:

(i) For any node $s _ { h }$ at depth $h \in \{ 0 , \ldots , H \}$

$$
{ \widehat { V } } _ { n } ( s _ { h } ) { \overset { \alpha _ { h } , \beta _ { h } } { \underset { n  \infty } { \longrightarrow } } } { \widetilde V } ( s _ { h } ) .
$$

(ii) For any node $s _ { h }$ at depth $h \in \{ 0 , \ldots , H - 1 \}$

$$
\widehat { Q } _ { n } \big ( s _ { h } , a \big ) \begin{array} { c } { \frac { \alpha _ { h + 1 } , \beta _ { h + 1 } } { \nu \to \infty } \widetilde Q ( s _ { h } , a ) , \quad \forall a \in \mathcal A _ { s _ { h } } . } \end{array}
$$

Proof. (Sketch) The argument proceeds by induction on the tree depth H. For $H \ = \ 1$ , we handle the root node using Lemma 17 plus the concentration assumptions on leaf nodes. For general H, we note that descending into a child node effectively reduces the depth by one, thus the induction hypothesis applies. By carefully controlling exploration (Section 5) and using robust backups, each node’s $\widehat { V }$ and $\widehat { Q }$ estimates concentrate at the specified rates.

Finally, Theorem 3 establishes that under optimal parameter tuning, the expected payoff at the root converges at $\mathcal { O } ( n ^ { - 1 / 2 } )$

Theorem 3. (Convergence ofExpected Payoff) At the root node $s _ { 0 } ,$ there is a choice ofparameters yielding

$$
\begin{array} { r } { \big | \mathbb { E } \big [ \widehat { V } _ { n } ( s _ { 0 } ) \big ] - \widetilde { V } ( s _ { 0 } ) \big | \ \leq \ \mathcal { O } \big ( n ^ { - 1 / 2 } \big ) . } \end{array}
$$

Remark 2. These results show that both Robust-Power-UCT and standard (non-robust) MCTS achieve the same $\mathcal { O } ( n ^ { - 1 / 2 } )$ rate for value estimation at the root node, which implies that robustness need not affect convergence speed, which is order-optimal. While we achieve this rate, the exact dependence on various problem-dependent factors (e.g., number of actions A, number of states S, tree search depth H, etc.) is not decodable (thereby not comparable to other online robust RL results (Dong et al., 2022)) due to our analysis limitations.

## 7. Experiments

We evaluate Robust-Power-UCT in three distinct environments designed to test different aspects of robust planning: the Gambler’s Problem, Frozen Lake, and American Option Pricing. For each environment, we compare: Stochastic-Power-UCT (Dam et al., 2024b) (baseline) and Robust-Power-UCT with Total Variation, Chi-squared, and Wasserstein ambiguity sets.

While several robust reinforcement learning methods exist (c.f.Section 2), to the best of our knowledge, this is the first work to incorporate ambiguity sets directly into MCTS, making Stochastic-Power-UCT our primary baseline. All experiments are done over 100 seeds, using $\gamma = 0 . 9 9$ and robustness budget $\rho = 0 . 5$ , with these values showing consistent performance across preliminary experiments with different parameter settings. For concise presentation, we only experiment with transition model ambiguity just as prior robust RL works.

Full experimental details, environment descriptions and hyperparameter configurations are provided in Appendix.E.1, along with an additional analysis of the robustness budget. We also provide our code at https://github.com/ brahimdriss/RobustMCTS.

Remark 3. While the robust Bellman operator involves solving a minimization problem over probability distributions, we can leverage dual reformulations to make its computation tractable. Many prior works (Iyengar, 2005; Nilim and El Ghaoui, 2005; Xu et al., 2023) show, for a value function V and nominal distribution $P ^ { o } ,$ , the robust value under all ambiguity balls with radius ρ can be computed in at most ${ \mathcal { O } } ( S \log ( S ) )$ time. Thus requiring only marginally more computation than standard Bellman operators $\mathcal O ( S )$ . This computational efficiency is crucial for practical implementations, particularly in online planning settings like MCTS withfrequent Bellman updates.

## 7.1. Gambler’s Problem Robustness Results

The Gambler’s Problem provides an ideal testbed for evaluating robustness to model misspecification. An agent must reach a target capital through a series of bets, with each bet winning with probability $p _ { h }$ . This enables precise control of the planning-execution mismatch through a single parameter.

![](images/19d33290b8c52fbd17ed97438255eff6fbeb8b2f8b3b2a155ca575408d59e312.jpg)

![](images/c54890e15af3e5887b01e06fee3f0026984a55a6bee73f5cf502be599a13f369.jpg)

![](images/0083a142bdbaec6816e75a0a0395599dfda68bae85590441d7c105b209d1aaca.jpg)  
Stochastic-Power-UCT Robust-Power-UCT MCTS (TV) Robust-Power-UCT MCTS (Chi²) Robust-Power-UCT MCTS (Wasserstein)  
Figure 1: Success rates in the Gambler’s Problem under model mismatch. Results show planning with fixed probabilities $p _ { h } = \{ 0 . 4 , 0 . 6 , 0 . 8 \}$ while executing across different probabilities. Shaded area demonstrates how robust methods maintain more consistent performance under model mismatch compared to Stochastic-Power-UCT.

Figure 1 illustrates the performance of different Power-UCT variants under model mismatch in the Gambler’s Problem. The behavior of Stochastic-Power-UCT reveals a fundamental vulnerability: when $p _ { h } < 0 . 5$ , there exist multiple optimal policies that achieve winning ratios close to the true environment probability. However, when planning with $p _ { h } \geqslant 0 . 5$ , the algorithm converges to an aggressive single-bet strategy that fails catastrophically when the true probability is lower than assumed.

The superior performance of robust variants stems from their conservative betting strategies. While Stochastic-Power-UCT often makes large single bets, robust variants tend to make smaller, sequential bets that preserve capital for future opportunities.

## 7.2. Frozen Lake Robustness Results

The Frozen Lake environment tests robustness in a different complex setting where uncertainties compound over multiple steps. The agent must navigate to a goal while avoiding hazards, with actions potentially failing with probability $p _ { \mathrm { s l i p } }$

Table 2 provides detailed success rates across different planning and execution probabilities. With matching conditions (4000 rollouts and $p _ { \mathrm { s l i p } } ~ = ~ 0 . 3 ~ \mathrm { c a s e ) }$ , our results closely match those reported in the original paper (Dam et al., 2024b) even with slightly different dynamics. The Wasserstein uncertainty set exhibits superior performance in scenarios with lower execution probabilities, achieving the highest success rates (bold) across multiple conditions. For example, with $p _ { \mathrm { s l i p } } ~ = ~ 0 . 3 .$ it achieves 58% success when $p _ { \mathrm { e x e c } } ~ = ~ 0 . 1$ , significantly outperforming other ap-

proaches.

Both Wasserstein and Chi-squared variants outperform the baseline Stochastic-Power-UCT and Total Variation approaches. Interestingly, when planning and execution probabilities align (underlined values), both robust variants maintain superior performance compared to standard approaches. This suggests that explicitly accounting for uncertainty in the planning process provides benefits even without model mismatch, possible by encouraging more conservative and reliable decision-making strategies.

These results on Gambler’s Problem and Frozen Lake demonstrate that explicitly accounting for model ambiguity during planning can significantly improve reliability when deployment conditions differ from simulation assumptions. The choice of ambiguity set provides a mechanism for balancing conservatism against nominal performance.

## 7.3. American Option Robustness Results

The American Option environment provides a financial domain to test reward robustness under model uncertainty. In this setting, the agent must decide when to exercise an option to maximize expected returns, with the key uncertain parameter being the probability $p _ { u }$ of price increases at each time step.

Figure 2 demonstrates the reward robustness of different Power-UCT variants under model mismatch in option pricing scenarios. We examine two planning scenarios: training with $p _ { u } = 0 . 5$ (left panel) and $p _ { u } = 0 . 6$ (right panel), then testing across execution probabilities from 0.4 to 0.8.

The results reveal that robust variants maintain significantly more stable performance compared to standard Power-UCT. When planning with $p _ { u } = 0 . 5 .$ , the standard approach shows dramatic performance degradation as the test probability deviates from the planning assumption, dropping from approximately 5 to near 0 when $p _ { u } = 0 . 8$ . In contrast, robust variants maintain consistent performance across the entire range.

![](images/db96e168dd72ff25e0754a301a6b40ebf67d15829fd8ab5d7074437fc781af53.jpg)

![](images/bc25ecbbf0ea93436851d671a47ae8a41803e0923ad89845ac623c724edbc39b.jpg)  
Power MCTS (Standard) Robust Power MCTS (TV) Robust Power MCTS (Chi²) Robust Power MCTS (Wasserstein)

Figure 2: Reward robustness comparison in American Option pricing under model mismatch. Results show planning with fixed price-up probabilities $p _ { u } = \{ 0 . 5 , 0 . 6 \}$ while testing across different probabilities. Robust variants maintain significantly more stable performance compared to standard Power-UCT, demonstrating consistent risk-averse behavior that is particularly valuable in financial decision-making contexts where reliability is crucial.
<table><tr><td rowspan="2">Planning  $p _ { \mathrm { s l i p } } ^ { \mathrm { p l a n } }$ </td><td colspan="4">Execution  $p _ { \mathrm { s l i p } }$ </td></tr><tr><td>0.1 0.2</td><td>0.3</td><td>0.4</td><td>0.5</td></tr><tr><td rowspan="4">0.3</td><td>Sp 15</td><td>12</td><td>10</td><td>8</td><td>7</td></tr><tr><td>Tv</td><td>18 15</td><td>12</td><td>10</td><td>8</td></tr><tr><td>Cs</td><td>55</td><td>45 35</td><td>25</td><td>18</td></tr><tr><td>Ws</td><td>58 48</td><td>32</td><td>28</td><td>20</td></tr><tr><td rowspan="4">0.4</td><td>Sp</td><td>8</td><td>7 6</td><td>5</td><td>4</td></tr><tr><td>Tv</td><td>10</td><td>8 7</td><td>6</td><td>5</td></tr><tr><td>Cs</td><td>35</td><td>28 22</td><td>18</td><td>12</td></tr><tr><td>Ws</td><td>38 30</td><td>25</td><td>20</td><td>15</td></tr><tr><td rowspan="4">0.5</td><td>Sp</td><td>5</td><td>4</td><td>4 3</td><td>3</td></tr><tr><td>Tv</td><td>6</td><td>5 4</td><td>4</td><td>3</td></tr><tr><td>Cs</td><td>25</td><td>20 15</td><td>12</td><td>8</td></tr><tr><td>Ws</td><td>28</td><td>22 18</td><td>15</td><td>10</td></tr></table>

Table 2: Success rates (%) for planning with Power-UCT variants. Methods: Stochastic-Power-UCT (Sp), Robust version with Total Variation (Tv), Chi-squared (Cs), and Wasserstein (Ws) ambiguity sets. Underlined values indicate matching planning and execution $p _ { \mathrm { s l i p } }$ . Bold indicates highest success rate per planning scenario.

When planning with $p _ { u } = 0 . 6 ,$ , standard Power-UCT exhibits extreme sensitivity with dramatically varying performance. The robust variants demonstrate desired risk-averse behavior: achieving conservative but stable returns across all conditions. This stability is especially valuable in financial contexts where consistent performance is preferred over potentially high but unreliable returns.

The Wasserstein and Chi-squared ambiguity sets show particularly strong performance, maintaining steady rewards even under significant model mismatch, demonstrating that explicitly accounting for uncertainty leads to policies inherently more robust to different deployment conditions.

## 8. Conclusions

We have developed a robust variant of Monte Carlo Tree Search (MCTS) that addresses dynamical model and reward distribution ambiguities, bridging the gap between simulation-based planning and real-world deployment. The dependence of MCTS-based algorithms’ convergence rates on parameters (states S, actions A, depth H) remains underexplored in standard RL. We will address this gap for both robust and non-robust setups in the future. As our formulation follows an overly conservative mathematical framework, in the future, we will explore alternative robust formulations that are more permeable to less conservative solutions to address the simulation-to-reality gap.

## Impact Statement

This paper presents a novel algorithm for the robust reinforcement learning field using the Monte Carlo Tree Search planning mechanism. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## Acknowledgments

Tuan Dam was funded by Hanoi University of Science and Technology (HUST) under Project No. T2024-TD-024. K. Panaganti acknowledges support from the Resnick Institute and the ‘PIMCO Postdoctoral Fellow in Data Science’ fellowship at Caltech. B. Driss was funded by the project ANR-23-CE23-0006. A. Wierman acknowledges support by the NSF through CNS-2146814, CPS-2136197, CNS-2106403, and NGSDI-2105648. This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD011015599 made by GENCI.

## References

Kishan Panaganti Badrinath. Robust Reinforcement Learning: Theory and Algorithms. PhD thesis, Texas A&M University, 2023.

Noam Brown and Tuomas Sandholm. Superhuman ai for heads-up no-limit poker: Libratus beats top professionals. Science, 359(6374):418–424, 2018.

Imre Csiszar. Eine informationstheoretische ungleichung´ und ihre anwendung auf den beweis der ergodizitat von¨ markoffschen ketten. A Magyar Tudomanyos Akad´ emia´ Matematikai Kutato Int´ ezet´ enek K´ ozlem¨ enyei´ , 8(1-2):85– 108, 1963.

Tuan Dam, Pascal Klink, Carlo D’Eramo, Jan Peters, and Joni Pajarinen. Generalized mean estimation in montecarlo tree search. In Christian Bessiere, editor, Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pages 2397– 2404. International Joint Conferences on Artificial Intelligence Organization, 7 2020. doi: 10.24963/ijcai.2020/ 332. URL https://doi.org/10.24963/ijcai. 2020/332. Main track.

Tuan Dam, Georgia Chalvatzaki, Jan Peters, and Joni Pajarinen. Monte-carlo robot path planning. IEEE Robotics and Automation Letters, 7(4):11213–11220, 2022.

Tuan Dam, Carlo D’Eramo, Jan Peters, and Joni Pajarinen. A unified perspective on value backup and exploration in monte-carlo tree search. Journal ofArtificial Intelligence Research, 81:511–577, 2024a.

Tuan Dam, Odalric-Ambrym Maillard, and Emilie Kaufmann. Power mean estimation in stochastic monte-carlo tree search. In Negar Kiyavash and Joris M. Mooij, editors, Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings of Machine Learning Research, pages 894–918. PMLR, 15–19 Jul 2024b. URL https://proceedings.mlr. press/v244/dam24a.html.

Tuan Q Dam, Carlo D’Eramo, Jan Peters, and Joni Pajarinen. Convex regularization in monte-carlo tree search. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 2365–2375. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/dam21a.html.

Jing Dong, Jingwei Li, Baoxiang Wang, and Jingzhao Zhang. Online policy optimization for robust mdp. arXiv preprint arXiv:2209.13841, 2022.

Nicolas Fournier and Arnaud Guillin. On the rate of convergence in wasserstein distance of the empirical measure. Probability theory and related fields, 162(3):707– 738, 2015.

Rui Gao and Anton Kleywegt. Distributionally robust stochastic optimization with wasserstein distance. Mathematics ofOperations Research, 48(2):603–655, 2023.

Michael B Giles. Multilevel monte carlo path simulation. Operations research, 56(3):607–617, 2008.

Arthur Guez, Theophane Weber, Ioannis Antonoglou,´ Karen Simonyan, Oriol Vinyals, Daan Wierstra, Remi´ Munos, and David Silver. Learning to search with mctsnets. In International conference on machine learning, pages 1822–1831. PMLR, 2018.

Stefan Heinrich. Multilevel monte carlo methods. In Large-Scale Scientific Computing: Third International Conference, LSSC 2001 Sozopol, Bulgaria, June 6–10, 2001 Revised Papers 3, pages 58–67. Springer, 2001.

Carl-Johan Hoel, Katherine Driggs-Campbell, Krister Wolff, Leo Laine, and Mykel J Kochenderfer. Combining planning and deep reinforcement learning in tactical decision making for autonomous driving. IEEE transactions on intelligent vehicles, 5(2):294–305, 2019.

Garud N Iyengar. Robust dynamic programming. Mathematics ofOperations Research, 30(2):257–280, 2005.

Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237–285, 1996.

Bilal Kartal, Pablo Hernandez-Leal, and Matthew E Taylor. Action guidance with mcts for deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence and interactive digital entertainment, volume 15, pages 153–159, 2019.

Behbod Keshavarzi and Hamidreza Navidi. Comparative analysis of extensive form zero sum game algorithms for poker like games. Scientific Reports, 15(1):2917, 2025.

Levente Kocsis and Csaba Szepesvari. Bandit based´ monte-carlo planning. In European conference on machine learning, pages 282–293. Springer, 2006.

Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y Levy, and Shie Mannor. Policy gradient for rectangular robust markov decision processes. Advances in Neural Information Processing Systems, 36:59477–59501, 2023.

Martin Kurecka, V ˇ aclav Nevyho ´ stˇ enˇ y, Petr Novotn \` y, and\` V´ıt Uncovskˇ y. Threshold uct: Cost-constrained monte\` carlo tree search with pareto curves. arXiv preprint arXiv:2412.13962, 2024.

Jongmin Lee, Geon-Hyeong Kim, Pascal Poupart, and Kee-Eung Kim. Monte-carlo tree search for constrained pomdps. Advances in Neural Information Processing Systems, 31, 2018.

Edouard Leurent. rl-agents: Implementations of reinforcement learning algorithms. https://github. com/eleurent/rl-agents, 2018.

Zhishuai Liu and Pan Xu. Distributionally robust offdynamics reinforcement learning: Provable efficiency with linear function approximation. In International Conference on Artificial Intelligence and Statistics, pages 2719–2727. PMLR, 2024.

Zijian Liu, Qinxun Bai, Jose Blanchet, Perry Dong, Wei Xu, Zhengqing Zhou, and Zhengyuan Zhou. Distributionally robust q-learning. In International Conference on Machine Learning, pages 13623–13643. PMLR, 2022.

Weichao Mao, Kaiqing Zhang, Qiaomin Xie, and Tamer Basar. Poly-hoot: Monte-carlo planning in continuous space mdps with non-asymptotic analysis. Advances in Neural Information Processing Systems, 33:4549–4559, 2020.

Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain transition matrices. Operations Research, 53(5):780–798, 2005.

Santiago Ontanon. The combinatorial multi-armed bandit´ problem and its application to real-time strategy games. In Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, volume 9, pages 58–64, 2013.

Michael Painter, Mohamed Baioumy, Nick Hawes, and Bruno Lacerda. Monte carlo tree search with boltzmann exploration. Advances in Neural Information Processing Systems, 36:78181–78192, 2023.

Kishan Panaganti and Dileep Kalathil. Robust reinforcement learning using least squares policy iteration with

provable performance guarantees. In International Conference on Machine Learning, pages 511–520. PMLR, 2021.

Kishan Panaganti and Dileep Kalathil. Sample complexity of robust reinforcement learning with a generative model. In International Conference on Artificial Intelligence and Statistics, pages 9582–9602. PMLR, 2022.

Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust reinforcement learning using offline data. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Kishan Panaganti, Adam Wierman, and Eric Mazumdar. Model-free robust φ-divergence reinforcement learning using both offline and online data. ICML, arXiv preprint arXiv:2405.05468, 2024.

Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Bridging distributionally robust learning and offline rl: An approach to mitigate distribution shift and partial data coverage. Learning for Dynamics and Control Conference, 2025.

Dinesh Parthasarathy, Georgios Kontes, Axel Plinge, and Christopher Mutschler. C-mcts: Safe planning with monte carlo tree search. arXiv preprint arXiv:2305.16209, 2023.

Hamed Rahimian and Sanjay Mehrotra. Distributionally robust optimization: A review. arXiv preprint arXiv:1908.05659, 2019.

Erica Salvato, Gianfranco Fenu, Eric Medvet, and Felice Andrea Pellegrino. Crossing the reality gap: A survey on sim-to-real transferability of robot controllers in reinforcement learning. IEEE Access, 9:153171–153187, 2021.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604– 609, 2020.

Devavrat Shah, Qiaomin Xie, and Zhi Xu. Nonasymptotic analysis of monte carlo tree search. In Abstracts of the 2020 SIGMETRICS/Performance Joint International Conference on Measurement and Modeling of Computer Systems, pages 31–32, 2020.

Laixi Shi and Yuejie Chi. Distributionally robust modelbased offline reinforcement learning with near-optimal sample complexity. Journal of Machine Learning Research, 25(200):1–91, 2024.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484, 2016.

Richard S Sutton and Andrew G Barto. Reinforcement learning: an introduction, 2nd edn. adaptive computation and machine learning, 2018.

Maciej Swiechowski, Konrad Godlewski, Bartosz Saw-<sup>´</sup> icki, and Jacek Mandziuk. Monte carlo tree search: A´ review of recent modifications and applications. Artificial Intelligence Review, 56(3):2497–2562, 2023.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulao,˜ Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

Jie Wang, Rui Gao, and Yao Xie. Regularization for adversarial robust learning. arXiv preprint arXiv:2408.09672, 2024a.

Shengbo Wang, Nian Si, Jose Blanchet, and Zhengyuan Zhou. A finite sample complexity bound for distributionally robust q-learning. In International Conference on Artificial Intelligence and Statistics, pages 3370–3398. PMLR, 2023.

Shengbo Wang, Nian Si, Jose Blanchet, and Zhengyuan Zhou. Sample complexity of variance-reduced distributionally robust q-learning. Journal of Machine Learning Research, 25(341):1–77, 2024b.

Yudan Wang. Model-free robust reinforcement learning with sample complexity analysis. Master’s thesis, State University of New York at Buffalo, 2024.

Chenjun Xiao, Ruitong Huang, Jincheng Mei, Dale Schuurmans, and Martin Muller. Maximum entropy¨ monte-carlo planning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and R. Garnett,´ editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips. cc/paper\_files/paper/2019/file/ 7ffb4e0ece07869880d51662a2234143-Paper. pdf.

Zaiyan Xu, Kishan Panaganti, and Dileep Kalathil. Improved sample complexity bounds for distributionally robust reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 9728–9754. PMLR, 2023.

Zhao-Heng Yin, Weirui Ye, Qifeng Chen, and Yang Gao. Planning for sample efficient imitation learning. Advances in Neural Information Processing Systems, 35: 2577–2589, 2022.

Runyu Zhang, Yang Hu, and Na Li. Regularized robust mdps and risk-sensitive mdps: Equivalence, policy gradient, and sample complexity. arXiv preprint arXiv:2306.11626, 2023.

Zhengqing Zhou, Zhengyuan Zhou, Qinxun Bai, Linhai Qiu, Jose Blanchet, and Peter Glynn. Finite-sample regret bound for distributionally robust offline tabular reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 3331–3339. PMLR, 2021.

## A. Notations

<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $s , A$ </td><td>State space and action space of the MDP</td></tr><tr><td> $H$ </td><td>Planning horizon (depth of the search tree).</td></tr><tr><td> $( s , a )$ </td><td>A specific state-action pair;  $s \in S , a \in { \mathcal { A } } .$ </td></tr><tr><td> $[ M ]$ </td><td>Denotes the set  $\{ 1 , 2 , \cdots , M \} .$ </td></tr><tr><td> $\nu _ { s , a } ^ { o }$ </td><td>Nominal (true) reward distribution for state-action pair  $( s , a ) .$ </td></tr><tr><td> $\nu _ { s , a } , \widehat { \nu } _ { s , a }$ </td><td>Generic and empirical reward distributions for  $( s , a )$  , respectively.</td></tr><tr><td> $R _ { \mathrm { m a x } }$ </td><td>Maximum possible reward value  $( \mathrm { i . e . , }$  reward is supported in  $[ 0 , \dot { R } _ { \mathrm { m a x } } ] )$ </td></tr><tr><td> $D _ { f } ( P \| P ^ { o } )$ </td><td>f-divergence between distributions  $P$  and  $P ^ { o }$  , for a convex  $\begin{array} { r } { \dot { f ( \cdot ) } . } \end{array}$ </td></tr><tr><td> $\mathcal { P } _ { s , a } ^ { \tilde { T } \tilde { V } } , \stackrel {  } { \mathcal { P } _ { s , a } ^ { \chi } } , \mathcal { P } _ { s , a } ^ { \mathcal { W } }$ </td><td>Uncertainty sets under Total Variation, Chi-square, and Wasserstein distances, respectively.</td></tr><tr><td> $\sigma _ { \mathcal { P } _ { s , a } } ( V )$ </td><td>Worst-case value operator (or “robust backup&quot;) over an uncertainty set  $\mathcal { P } _ { s , a } .$ </td></tr><tr><td>ρ</td><td>Radius (budget) for the uncertainty set in  $f -$  divergence or Wasserstein distance.</td></tr><tr><td> $\sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } )$ </td><td> ${ \widehat { p } } _ { n }$  Robust backup operator with the empirical transition as the set center for empirical value</td></tr><tr><td> $B _ { p }$ </td><td>Constant bounding the metric space for Wasserstein distance (e.g. max distance  $\bar { ( d ^ { p } ) }$ </td></tr><tr><td> $\alpha ^ { * } , \widehat { \alpha } ^ { * }$ </td><td>Dual variables optimizing robust reward functions under TV uncertainty sets.</td></tr><tr><td> $\Delta ( X )$ </td><td>Probability simplex over the support of set X or size of X.</td></tr><tr><td> $\dot { \mathcal { N } _ { R } } ( \dot { \theta } )$ </td><td>θ-cover set used for bounding the supremum of  $( \eta - R ) _ { + }$  in total-variation analysis</td></tr><tr><td> $\| \cdot \| _ { \infty } , \| \cdot \| _ { 1 }$ </td><td>Infinity norm (maximum absolute value in a vector) and  $\ell _ { 1 }$  norm (sum of absolute values in a vector).</td></tr><tr><td> $\delta _ { x }$ </td><td>A point mass at a realization x</td></tr><tr><td> $\delta , \theta , \varepsilon$ </td><td>Parameters often controlling confidence levels or approximation accuracy in concentration bounds.</td></tr><tr><td> $c , C$ </td><td>Constants from generic concentration or covering-number arguments (possibly problem-dependent).</td></tr><tr><td> $p _ { m } , N _ { m } ^ { n }$ </td><td>Used for i.i.d. sampling from a discrete distribution  $\left( p _ { 1 } , \ldots , p _ { M } \right)$  , with  $N _ { m } ^ { n }$  the count of outcomes of type m.</td></tr><tr><td> $\widehat { Q } _ { n } , Q ^ { \star }$ </td><td>Estimated and true robust  $Q \cdot$  -values, respectively.</td></tr><tr><td> $\widehat { V } _ { n } , V ^ { \star }$ </td><td>Estimated and true robust value functions, respectively.</td></tr><tr><td>γ</td><td>Discount factor in the MDP.</td></tr><tr><td> $\mathcal { W } ( \mu _ { n } , \mu )$ </td><td>Wasserstein distance between empirical measure  $\mu _ { n }$  and true measure</td></tr><tr><td> $\underline { { \alpha } } , \underline { { \beta } }$   $n \to \infty$ </td><td>Notation for concentration at rate  $( \alpha , \beta ) ;$  see text for precise definition.</td></tr></table>

Table 3: Key Notations Used in the Appendix. Symbols and definitions for uncertainty sets $( \mathrm { T V } , \chi ^ { 2 }$ , Wasserstein), reward distributions, and the main variables in robust MDP analysis.

## B. Useful technical results

Lemma 1. (Lemma 1 (Panaganti and Kalathil, 2022)) For any $( s , a ) \in \mathcal { S } \times \mathcal { A }$ and for any V $. , V _ { 2 } \in \mathbb { P } ^ { | S | }$ , we have $| \sigma _ { P _ { s , a } } ( V _ { 1 } ) - \sigma _ { P _ { s , a } } ( V _ { 2 } ) | \leqslant \| V _ { 1 } - V _ { 2 } \| _ { \infty } a n d | \sigma _ { \widehat { P } _ { s , a } } ( V _ { 1 } ) - \sigma _ { \widehat { P } _ { s , a } } ( V _ { 2 } ) | \leqslant \| V _ { 1 } - V _ { 2 } \| _ { \infty }$

Lemma 2. (Proposition 2 (Xu et al., 2023)) Fix any $h , s , a \in [ H ] \times S \times A .$ . For any $\theta , \delta \in ( 0 , 1 )$ , we have with the probability of at least $1 - \delta , \left| \sigma _ { P _ { s , a } ^ { T V } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , a } ^ { T V } } ( \widehat { V } _ { h + 1 } ) \right| \leqslant 2 \theta + \sqrt { H ^ { 2 } \log ( 4 H / \theta \delta ) / 2 n }$

From Lemma 2, we have

$$
\mathbb { P } \left( \left| \sigma _ { P _ { s , a } ^ { T V } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , a } ^ { T V } } ( \widehat { V } _ { h + 1 } ) \right| \geqslant 2 \theta + \sqrt { H ^ { 2 } \log ( 4 H / \theta \delta ) / 2 n } \right) < \delta\tag{5}
$$

Set $\theta = \varepsilon / 4$ with $\varepsilon = 2 \sqrt { H ^ { 2 } \log ( 4 H / \theta \delta ) / 2 n }$ , then

$$
2 n \varepsilon ^ { 2 } / 4 = H ^ { 2 } \log ( 4 H / \theta \delta ) \Rightarrow \exp \{ - 2 n \varepsilon ^ { 2 } / H ^ { 2 } \} = \varepsilon \delta / 1 6 H\tag{6}
$$

$$
\Rightarrow \delta = { \frac { 1 6 H \exp \{ - n \varepsilon ^ { 2 } / 2 H ^ { 2 } \} } { \varepsilon } }\tag{7}
$$

so that

$$
\mathbb { P } \left( \left| \sigma _ { P _ { s , a } ^ { T V } } \left( \widehat { V } _ { h + 1 } \right) - \sigma _ { \widehat { P } _ { s , a } ^ { T V } } \left( \widehat { V } _ { h + 1 } \right) \right| \geqslant \varepsilon \right) < \frac { 1 6 H \exp \{ - n \varepsilon ^ { 2 } / 2 H ^ { 2 } \} } { \varepsilon }\tag{8}
$$

Lemma 3. (Proposition 4 (Xu et al., 2023)) Fix any $h , s , a \in [ H ] \times S \times A .$ . For any $\theta , \delta \in ( 0 , 1 )$ , we have with the probability ofat least $\begin{array} { r } { 1 - \delta , \bigg | \sigma _ { P _ { s , a } ^ { \chi } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , a } ^ { \chi } } ( \widehat { V } _ { h + 1 } ) \bigg | \leqslant 2 \theta + \frac { \sqrt { 2 } C _ { p } ^ { 2 } H } { ( C _ { p } - 1 ) \sqrt { n } } \left( \sqrt { \log \left( \frac { 2 ( 1 + C _ { p } H / ( \theta ( C _ { p } - 1 ) ) ) } { \delta } \right) + 1 } \right) } \end{array}$

Then we have

$$
\begin{array} { r l } { \mathbb { P } \Bigg ( \Big | \sigma _ { P _ { s , o } ^ { \mathcal { X } } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , o } ^ { \mathcal { X } } } ( \widehat { V } _ { h + 1 } ) \Big | \geqslant 2 \theta } & { + \frac { \sqrt { 2 } C _ { p } ^ { 2 } H } { ( C _ { p } - 1 ) \sqrt { n } } \bigg ( \sqrt { \log \bigg ( \frac { 2 ( 1 + C _ { p } H / ( \theta ( C _ { p } - 1 ) ) ) } { \delta } \bigg ) } + 1 \bigg ) \bigg ) < \delta } \end{array}\tag{9}
$$

Set $\theta = \varepsilon / 4$ <sub>with</sub> <sub>ε</sub> <sub>=</sub> <sub>2</sub> <sup>√2C2</sup>p<sup>H</sup><sub>(C −1)√n</sub> rlog  <sup>2(1+C</sup>p<sup>H/(θ(C</sup>p<sup>−1)))</sup><sub>δ</sub>  + 1, then

$$
\left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } H } - 1 \right) ^ { 2 } = \log \left( \frac { 2 ( 1 + C _ { p } H / ( \theta ( C _ { p } - 1 ) ) ) } { \delta } \right)\tag{10}
$$

$$
\Rightarrow \exp \left\{ - \left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } H } - 1 \right) ^ { 2 } \right\} = \frac { \delta } { 2 ( 1 + C _ { p } H / ( \theta ( C _ { p } - 1 ) ) ) }\tag{11}
$$

$$
\Rightarrow \mathbb { P } \left( \left| \sigma _ { P _ { s , a } ^ { \mathcal { X } } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , a } ^ { \mathcal { X } } } ( \widehat { V } _ { h + 1 } ) \right| \geqslant \varepsilon \right) <\tag{12}
$$

$$
2 \left( 1 + C _ { p } H / ( { \frac { \varepsilon ( C _ { p } - 1 ) } { 4 } } ) \right) \exp \left\{ - \left( { \frac { ( C _ { p } - 1 ) { \sqrt { n } } \varepsilon } { 2 { \sqrt { 2 } } C _ { p } ^ { 2 } H } } - 1 \right) ^ { 2 } \right\}\tag{13}
$$

Lemma 4. (Proposition 9 (Xu et al., 2023)) Fix any $h , s , a \in [ H ] \times S \times A .$ . For any $\theta , \delta \in ( 0 , 1 )$ , we have with the probability of at least $\begin{array} { r } { 1 - \delta , \left| \sigma _ { P _ { s , a } ^ { w } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , a } ^ { w } } ( \widehat { V } _ { h + 1 } ) \right| \leqslant 2 \theta + \frac { H ( B _ { p } + \rho ^ { p } ) } { \rho ^ { p } } \sqrt { \frac { \log \left( \frac { 2 H B _ { p } + 2 H \sqrt { \rho ^ { p } } } { \rho ^ { p } \theta \delta } \right) } { 2 n } } } \end{array}$

Similarly, Set $\theta = \varepsilon / 4$ with $\begin{array} { r } { \varepsilon = \frac { 2 H ( B _ { p } + \rho ^ { p } ) } { \rho ^ { p } } \sqrt { \frac { \log \left( \frac { 2 H B _ { p } + 2 H \sqrt { \rho ^ { p } } } { \rho ^ { p } \theta \delta } \right) } { 2 n } } } \end{array}$ , then

$$
\exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 H ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\} = \frac { \rho ^ { p } \theta \delta } { 2 H B _ { p } + 2 H \sqrt { \rho ^ { p } } }\tag{14}
$$

$$
\Rightarrow \delta = \frac { 4 \left( 2 H B _ { p } + 2 H \sqrt { \rho ^ { p }  } } { \rh\right)o ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 H ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\}\tag{15}
$$

so that

$$
\mathbb { P } \left( \left| \sigma _ { P _ { s , a } ^ { y y } } ( \widehat { V } _ { h + 1 } ) - \sigma _ { \widehat { P } _ { s , a } ^ { y y } } ( \widehat { V } _ { h + 1 } ) \right| \geqslant \varepsilon \right) < \frac { 4 \left( 2 H B _ { p } + 2 H \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 H ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\}\tag{16}
$$

Lemma 5. (Lemma 2 (Fournier and Guillin, 2015), Concentration inequality for Wasserstein distance ). For $\mu \in { \mathcal { P } } ( \mathbb { R } )$ we consider an i.i.d. sequence $( X _ { k } ) _ { k \geqslant 1 }$ ofµ-distributed random variables and,for all $n \geqslant 1$ , the empirical measure

$$
\mu _ { n } : = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \delta _ { X _ { k } } .
$$

Assume that there exists $\gamma > 0$ such that $\begin{array} { r } { \mathcal { E } _ { 2 , \gamma } ( \mu ) : = \int _ { \mathbb { R } } \exp \left( \gamma | x | ^ { 2 } \right) \mu ( d x ) < \infty } \end{array}$ . Then for all $n \geqslant 1$ , all $x > 0 ,$

$$
\mathbb { P } \left( \mathcal { W } \left( \mu _ { n } , \mu \right) \geqslant x \right) \leqslant C \exp \left( - c n x ^ { 2 } \right)
$$

where the Wasserstein distance $\mathcal { W } \left( { \boldsymbol { \mu } _ { n } } , { \boldsymbol { \mu } } \right)$ is defined by

$$
\mathcal { W } \left( \mu _ { n } , \mu \right) : = \operatorname* { i n f } _ { \pi \in \Pi \left( \mu _ { n } , \mu \right) } \left\{ \int \left| x - y \right| \pi ( d x , d y ) \right\}
$$

and the positive constant C and c depends only on γ and $\mathcal { E } _ { 2 , \gamma } ( \mu )$

Lemma 6. (Lemma 4 (Zhou et al., 2021)). Let $X \sim P$ be a random variable with $X \in [ 0 , M ]$ , and $P _ { n }$ denotes its empirical distribution ofsample size n. For $\delta > 0 ,$ , for any

$$
\alpha ^ { * } \in \underset { \alpha \geqslant 0 } { \arg \operatorname* { m a x } } \left\{ - \alpha \log \left( \mathbb { E } _ { P } \left[ e ^ { - X / \alpha } \right] \right) - \alpha \delta \right\}\tag{17}
$$

$( l ) \alpha ^ { * } = 0$ . Furthermore, assume that the support ofX is finite. Then there exists a constant $N ^ { \prime } : = N ^ { \prime } ( \varepsilon , \delta , P )$ , such that n $\geqslant N ^ { \prime }$ , with probability at least $1 - \varepsilon _ { : }$ , we have

$$
0 \in \underset { \alpha \geqslant 0 } { \arg \operatorname* { m a x } } \left\{ - \alpha \log \left( \mathbb { E } _ { P _ { n } } \left[ e ^ { - X / \alpha } \right] \right) - \alpha \delta \right\}
$$

(2) $\alpha ^ { * } > 0 .$ . Then there exists a constant $N ^ { \prime \prime } : = N ^ { \prime \prime } ( \varepsilon , \delta , P )$ , such that for any n $\geqslant N ^ { \prime \prime }$ , with probability at least $1 - \varepsilon ,$ there exists a

$$
\widehat { \alpha } ^ { * } \in \underset { \alpha \geqslant 0 } { \arg \operatorname* { m a x } } \left\{ - \alpha \log \left( \mathbb { E } _ { P _ { n } } \left[ e ^ { - X / \alpha } \right] \right) - \alpha \delta \right\}
$$

such that $\alpha ^ { \ast } , \widehat { \alpha } ^ { \ast } \in [ \underline { { \alpha } } , \bar { \alpha } ] ,$ , where $\underline { { \alpha } } > 0$ is independent ofn and $\bar { \alpha } = M / \delta$

The Total Variation, Chi-square, and Kullback-Liebler uncertainty sets are constructed with the f-divergence. The $f$ divergence between the distributions $P$ and $P ^ { o }$ is defined as

$$
D _ { f } \left( P \| P ^ { o } \right) = \int f \left( { \frac { d P } { d P ^ { o } } } \right) d P ^ { o }\tag{18}
$$

where $f$ is a convex function (Csiszar, 1963). We obtain different divergences for different forms of the function´ $f ,$ including some well-known divergences. For example, $f ( t ) = | t - 1 | / 2$ gives Total Variation, $f ( t ) = ( t - 1 ) ^ { 2 }$ gives chi-square, and $f ( t ) = t \log ( t )$ gives Kullback-Liebler.

Lemma 7. (Lemma 5 (Panaganti et al., 2022)) Let $D _ { f }$ be as defined in equation 18 with $f ( t ) = | t - 1 | / 2$ corresponding to the TV uncertainty set. Then,

$$
\operatorname* { i n f } _ { \substack { D _ { f } ( P | | P ^ { o } ) \leqslant \rho } } \mathbb { E } _ { P } [ l ( X ) ] = - \operatorname* { i n f } _ { \eta \in \mathbb { R } } \mathbb { E } _ { P ^ { o } } \left[ ( \eta - l ( X ) ) _ { + } \right] + \left( \eta - \operatorname* { i n f } _ { x \in \mathcal { X } } l ( x ) \right) _ { + } \times \rho - \eta ,
$$

Lemma 8. (Covering number $( T V ) )$ . Given a reward function $R \in \mathcal { R } _ { s , a } ,$ let $\mathcal { U } _ { R } = \left. ( \eta \cdot { \bf 1 } - R ) _ { + } : \eta \in [ 0 , R _ { \operatorname* { m a x } } ] \right.$ . Fix any $\theta \in ( 0 , 1 )$ ). Denote

$$
\mathcal { N } _ { R } ( \theta ) = \{ ( \eta \cdot \mathbf { 1 } - R ) _ { + } : \eta \in \{ \theta , 2 \theta , \ldots , N _ { \theta } \cdot \theta \} \}
$$

where $N _ { \theta } = \lceil R _ { \mathrm { m a x } } / \theta \rceil$ . Then $\mathcal { N } _ { R } ( \theta )$ is a θ-coverfor $\boldsymbol { \mathcal { U } } _ { R }$ with respect to $\| \cdot \| _ { \infty } ,$ , and its cardinality is bounded as $| \mathcal { N } _ { R } ( \theta ) | \leqslant$ $2 R _ { \mathrm { m a x } } / \theta .$ . Furthermore,for any $\nu \in \mathcal { N } _ { R } ( \theta )$ , we have $\| \nu \| _ { \infty } \leqslant 1$

Proof. First, $N _ { \theta } = \lceil R _ { \mathrm { m a x } } / \theta \rceil$ is the minimal number of subintervals of length θ needed to cover $[ 0 , R _ { \mathrm { m a x } } ]$ . Denote $J _ { i } =$ $[ ( i - 1 ) \theta , i \theta )$ to be the i-th subinterval, $1 \leqslant i \leqslant N _ { \theta }$ . Fix some $\mu \in { \mathcal { U } } _ { R }$ . Then $\mu = ( \eta \cdot \mathbf { 1 } - R ) _ { + }$ . Without loss of generality, assume this particular $\eta \in J _ { i }$ . Let $\nu = ( ( i \theta ) \cdot { \bf 1 } - R ) _ { + }$ . Now, for any $s , a \in \mathcal S \times \mathcal A ,$

$$
\begin{array} { r l } { | \nu ( s , a ) - \mu ( s , a ) | = | ( i \theta - R ) _ { + } - ( \eta - R ) _ { + } | } & { } \\ { \overset { ( a ) } { \leqslant } | i \theta - R - \eta + R | } & { } \\ { \leqslant | i \theta - ( i - 1 ) \theta | = \theta } & { } \end{array}
$$

where (a) follows from $i \theta > \eta$ and the fact that max $\{ x , 0 \} - \operatorname* { m a x } \{ y , 0 \} \leqslant x - y , { \mathrm { i f ~ } } x > y .$ Taking maximum with respect to $s , a$ on both sides, we get $\| \nu - \mu \| _ { \infty } \leqslant \theta$ . Since $\nu \in \mathcal { N } _ { R } ( \theta )$ , this suggests $\mathcal { N } _ { R } ( \theta )$ is a θ-cover for $\boldsymbol { \mathcal { U } } _ { R }$ . The cardinality bound directly follows from

$$
| \mathcal { N } _ { R } ( \theta ) | = N _ { \theta } = \lceil R _ { \mathrm { m a x } } / \theta \rceil \leqslant R _ { \mathrm { m a x } } / \theta + 1 \leqslant 2 R _ { \mathrm { m a x } } / \theta
$$

where the last inequality is due to $0 < \theta < 1$ . Now, for any $\nu \in \mathcal { N } _ { R } ( \theta )$ , we can establish the following

$$
\nu = ( \eta \cdot { \bf 1 } - R ) _ { + } \leqslant ( R _ { \operatorname* { m a x } } { \bf 1 } - R ) _ { + } \leqslant R _ { \operatorname* { m a x } }
$$

where the inequality is element-wise.

Lemma 9. Fix any $( s , a ) \in \mathcal { S } \times \mathcal { A } .$ Fix any reward function $R \in \mathcal R _ { s , a }$ <sub>a</sub>. Let $\mathcal { N } _ { R } ( \theta )$ be the $\theta \cdot$ cover of $\begin{array} { r l } { \mathcal { U } _ { R } } & { { } = } \end{array}$ $\{ ( \eta \cdot \mathbf { 1 } - R ) _ { + } : \eta \in [ 0 , R _ { \operatorname* { m a x } } ] \}$ as described in Lemma 8 . We then have

$$
\operatorname* { s u p } _ { \eta \in \left[ 0 , R _ { \operatorname* { m a x } } \right] } \left| \mathbb { E } _ { R \sim \hat { \nu } _ { s , a } } \left[ \left( \eta - R \right) _ { + } \right] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \left[ \left( \eta - R \right) _ { + } \right] \right| \leqslant \operatorname* { m a x } _ { r \in \mathcal { N } _ { R } \left( \theta \right) } \left| \hat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + 2 \theta
$$

Proof. For any $\mu \in \mathcal { U } _ { R }$ , there exists $r \in \mathcal { N } _ { R } ( \theta )$ such that $\| \mu - r \| _ { \infty } \leqslant \theta .$ . Now for such particular $\mu$ and $r ,$ we have

$$
\begin{array} { r l } & { \left| \widehat { \nu } _ { s , a } \mu - \nu _ { s , a } ^ { o } \mu \right| \leqslant \left| \widehat { \nu } _ { s , a } \mu - \widehat { \nu } _ { s , a } r \right| + \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + \left| \nu _ { s , a } ^ { o } r - \nu _ { s , a } ^ { o } \mu \right| } \\ & { \qquad \leqslant \left\| \widehat { \nu } _ { s , a } \right\| _ { 1 } \left\| \mu - r \right\| _ { \infty } + \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + \left\| \nu _ { s , a } ^ { o } \right\| _ { 1 } \left\| r - \mu \right\| _ { \infty } } \\ & { \qquad \leqslant \underset { \nu \in \mathcal { N } _ { R } ( \theta ) } { \operatorname* { m a x } } \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + 2 \theta . } \end{array}
$$

Taking maximum over $\boldsymbol { \mathcal { U } } _ { R }$ on both sides, we get

$$
\operatorname* { s u p } _ { \mu \in \mathcal { U } _ { R } } \left| \widehat { \nu } _ { s , a } \mu - \nu _ { s , a } ^ { o } \mu \right| \leqslant \operatorname* { m a x } _ { r \in \mathcal { N } _ { R } ( \theta ) } \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + 2 \theta .
$$

Now note that by the definition of $\mathcal { U } _ { R }$ , we have

$$
\operatorname* { s u p } _ { \eta \in \left[ 0 , R _ { \operatorname* { m a x } } \right] } \left| \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \left[ \left( \eta - R \right) _ { + } \right] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \left[ \left( \eta - R \right) _ { + } \right] \right| \leqslant \operatorname* { m a x } _ { r \in \mathcal { N } _ { R } \left( \theta \right) } \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + 2 \theta
$$

The desired result directly follows.

Lemma 10. Consider the total-variation uncertainty set

$$
\mathcal { P } _ { s , a } ^ { \mathrm { T V } } = \Big \{ P : \frac { 1 } { 2 } \| P - P _ { s , a } ^ { o } \| _ { 1 } \leq \delta \Big \} .
$$

Let $\begin{array} { r } { \widehat { R } _ { s , a } ^ { \mathrm { r o b } _ { T V } } = \operatorname* { m i n } _ { r _ { s , a } \sim \widehat { \mathcal { R } } _ { s , a } } \mathbb { E } _ { R \sim r _ { s , a } } [ R ] } \end{array}$ and $\begin{array} { r } { R _ { s , a } ^ { \mathrm { r o b } _ { T V } } = \operatorname* { m i n } _ { r _ { s , a } \sim \mathcal { R } _ { s , a } } \mathbb { E } _ { R \sim r _ { s , a } } [ R ] } \end{array}$ be the robust rewards defined using the empirical estimate $\widehat { \nu } _ { s , a }$ and $\nu _ { s , a } ^ { o }$ and respectively. Then there exists a constant

$$
N ^ { * } ( \varepsilon , \delta , \nu _ { s , a } ^ { o } ) ,
$$

such thatfor all $n \geq N ^ { * }$ (i.e. a sufficiently large number of reward samples at $( s , a )$ , the following holds with probability at least $1 - \varepsilon { : }$

$$
\bigg | \widehat { R } _ { s , a } ^ { \mathrm { r o b } _ { T V } } - R _ { s , a } ^ { \mathrm { r o b } _ { T V } } \bigg | \leqslant \sqrt { \frac { R _ { \operatorname* { m a x } } ^ { 2 } \log ( 2 / \delta ) } { 2 n } } .
$$

Proof. Following similar analyses as in Proposition 2 (Xu et al., 2023) (Lemma.2), we get

$$
\left| \widehat { R } _ { s , a } ^ { \mathrm { r o b r } } - R _ { s , a } ^ { \mathrm { r o b } _ { T V } } \right| = \big | \operatorname* { i n f } _ { \eta \in [ 0 , 2 R _ { \operatorname* { m a x } } / \rho ] } \left\{ \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \left[ \left( \eta - R \right) _ { + } \right] + \left( \eta - \operatorname* { i n f } _ { R ^ { \prime } \in [ 0 , R _ { \operatorname* { m a x } } ] } R ^ { \prime } \right) _ { + } \cdot \rho - \eta \right\}\tag{19}
$$

(20)

$$
\stackrel { ( a ) } { \leqslant } \underset { \eta \in \left[ 0 , 2 R _ { \operatorname* { m a x } } / \rho \right] } { \operatorname* { s u p } } \left| \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \left[ \left( \eta - R \right) _ { + } \right] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \left[ \left( \eta - R \right) _ { + } \right] \right|\tag{21}
$$

$$
\leqslant \operatorname* { m a x } \left\{ \underset { \eta \in [ 0 , R _ { \operatorname* { m a x } } ] } { \operatorname* { s u p } } \Big | \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \big [ ( \eta - R ) _ { + } \big ] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \big [ ( \eta - R ) _ { + } \big ] \right| ,\tag{22}
$$

$$
\operatorname* { s u p } _ { \eta \in [ R _ { \operatorname* { m a x } } , 2 R _ { \operatorname* { m a x } } / \rho ] } \bigg | \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \left[ ( \eta - R ) _ { + } \right] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \left[ ( \eta - R ) _ { + } \right] \bigg | \bigg \}\tag{23}
$$

$$
\overset { ( b ) } { \leqslant } \operatorname* { m a x } \left\{ \underset { \eta \in [ 0 , R _ { \operatorname* { m a x } } ] } { \operatorname* { s u p } } \Big | \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \left[ ( \eta - R ) _ { + } \right] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \left[ ( \eta - R ) _ { + } \right] \right| ,\tag{24}
$$

$$
| \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } [ R ] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } [ R ] | \}\tag{25}
$$

$$
\stackrel { ( c ) } { \leqslant } \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { r \in \mathcal { N } _ { R } \left( \theta \right) } \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| + 2 \theta , \left| \mathbb { E } _ { R \sim \widehat { \nu } _ { s , a } } \left[ R \right] - \mathbb { E } _ { R \sim \nu _ { s , a } ^ { o } } \left[ R \right] \right| \right\}\tag{26}
$$

where (a) follows from the fact that $\begin{array} { r } { \left| \operatorname* { i n f } _ { x } f ( x ) - \operatorname* { i n f } _ { x } g ( x ) \right| \leqslant \operatorname* { s u p } _ { x } | f ( x ) - g ( x ) | } \end{array}$ . For (b), recall that $R \leqslant R _ { \operatorname* { m a x } }$ for any $R \in \mathcal R$ . Hence, the term $\eta - R ^ { \prime }$ is always non-negative for $\eta \in [ R _ { \operatorname* { m a x } } , 2 R _ { \operatorname* { m a x } } / \rho ]$ , which cancels out by linearity of the expectation. (c) follows from applying Lemma 9 to the first term. Recall that all $r \in \mathcal { N } _ { R } ( \theta )$ is upper bounded by $R _ { \mathrm { m a x } }$ Now we can apply Hoeffding’s inequality to the first term in equation 26:

$$
\mathbb { P } \left( \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| \geqslant \varepsilon \right) \leqslant 2 \exp \left( - \frac { 2 n \varepsilon ^ { 2 } } { R _ { \operatorname* { m a x } } ^ { 2 } } \right) , \quad \forall \varepsilon > 0
$$

Now choose $\varepsilon = \sqrt { \frac { R _ { \mathrm { m a x } } ^ { 2 } \log ( 2 | \mathcal { N } _ { R } ( \theta ) | / \delta ) } { 2 N } }$ and recall that $| \mathcal { N } _ { R } ( \theta ) | \leqslant 2 R _ { \mathrm { m a x } } / \theta$ from Lemma 8. We have

$$
\begin{array} { r l } & { \mathbb { P } \left( \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| \geqslant \sqrt { \frac { R _ { \operatorname* { m a x } } ^ { 2 } \log \left( 4 R _ { \operatorname* { m a x } } / \theta \delta \right) } { 2 n } } \right) \leqslant \mathbb { P } \left( \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| \geqslant \sqrt { \frac { R _ { \operatorname* { m a x } } ^ { 2 } \log \left( 2 \left| \mathcal { N } _ { R } ( \theta ) \right| / \delta \right) } { 2 n } } \right) } \\ & { \qquad \leqslant \frac { \delta } { \left| \mathcal { N } _ { R } ( \theta ) \right| } } \end{array}
$$

Applying a union bound over $\mathcal { N } _ { R } ( \theta )$ , we get

$$
\operatorname* { m a x } _ { r \in \mathcal { N } _ { R } ( \theta ) } \left| \widehat { \nu } _ { s , a } r - \nu _ { s , a } ^ { o } r \right| \leqslant \sqrt { \frac { R _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 R _ { \operatorname* { m a x } } / \theta \delta ) } { 2 n } }\tag{27}
$$

with probability at least $1 - \delta$ . Now we can also apply Hoeffding’s inequality to the second term in equation 26. Recall that any reward function is bounded by $R _ { \mathrm { m a x } }$ . We have

$$
\bigg | \widehat { R } _ { s , a } ^ { \mathrm { r o b } _ { T V } } - R _ { s , a } ^ { \mathrm { r o b } _ { T V } } \bigg | \leqslant \sqrt { \frac { R _ { \operatorname* { m a x } } ^ { 2 } \log ( 2 / \delta ) } { 2 n } }\tag{28}
$$

with probability at least $1 - \delta$ . Combining equation 26 - equation 28 completes the proof.

From Lemma 10, we have

$$
\mathbb { P } \left( \Big | \widehat { R } _ { s , a } ^ { \mathrm { r o b } _ { T V } } - R _ { s , a } ^ { \mathrm { r o b } _ { T V } } \Big | \geqslant 2 \theta + \sqrt { R _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 R _ { \operatorname* { m a x } } / \theta \delta ) / 2 n } \right) < \delta\tag{29}
$$

Set $\theta = \varepsilon / 4$ with $\varepsilon = 2 \sqrt { R _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 R _ { \operatorname* { m a x } } / \theta \delta ) / 2 n } ,$ then

$$
2 n \varepsilon ^ { 2 } / 4 = R _ { \mathrm { m a x } } ^ { 2 } \log ( 4 R _ { \mathrm { m a x } } / \theta \delta ) \Rightarrow \exp \{ - 2 n \varepsilon ^ { 2 } / R _ { \mathrm { m a x } } ^ { 2 } \} = \varepsilon \delta / 1 6 R _ { \mathrm { m a x } }\tag{30}
$$

$$
\Rightarrow \delta = \frac { 1 6 R _ { \mathrm { m a x } } \exp \{ - n \varepsilon ^ { 2 } / 2 R _ { \mathrm { m a x } } ^ { 2 } \} } { \varepsilon }\tag{31}
$$

so that

$$
\mathbb { P } \left( \Big | \widehat { R } _ { s , a } ^ { \mathrm { r o b } _ { T V } } - R _ { s , a } ^ { \mathrm { r o b } _ { T V } } \Big | \geqslant \varepsilon \right) < \frac { 1 6 R _ { \mathrm { m a x } } \exp \{ - n \varepsilon ^ { 2 } / 2 R _ { \mathrm { m a x } } ^ { 2 } \} } { \varepsilon }\tag{32}
$$

Lemma 11. (Lemma 9 (Panaganti et al., 2022)) Let $D _ { f }$ be defined as in equation 18 with the convex function $f ( t ) =$ $( t - 1 ) ^ { 2 }$ corresponding to the Chi-square uncertainty set. Then

$$
\operatorname* { i n f } _ { D _ { f } ( P | | P ^ { o } ) \leqslant \rho } \mathbb { E } _ { P } [ l ( X ) ] = - \operatorname* { i n f } _ { \eta \in \mathbb { R } } \left\{ \sqrt { \rho + 1 } \sqrt { \mathbb { E } _ { P ^ { o } } \left[ ( \eta - l ( X ) ) ^ { 2 } \right] } - \eta \right\}
$$

Lemma 12. Fix any $s , a \in \mathcal S \times \mathcal A$ . For any $\theta , \delta \in ( 0 , 1 )$ and $\rho > 0 ;$ , we have, with probability at least $1 - \delta ,$ we can find a constant $N ^ { \star }$ such that ∀n $\geqslant N ^ { \star }$ , we have

$$
\left. \widehat { R } _ { s , a } ^ { \mathrm { r o b } \chi } - R _ { s , a } ^ { \mathrm { r o b } \chi } \right. \leqslant 2 \theta + \frac { \sqrt { 2 } C _ { \rho } ^ { 2 } R _ { \operatorname* { m a x } } } { \left( C _ { \rho } - 1 \right) \sqrt { n } } \left( \sqrt { \log \left( \frac { 2 \left( 1 + C _ { \rho } R _ { \operatorname* { m a x } } / \left( \theta \left( C _ { \rho } - 1 \right) \right) \right) } { \delta } \right) } + 1 \right) .
$$

Proof. Similar to Lemma 10, the result is direct by applying the results of Lemma 9, Lemma 11 and the law of total probability. □

From the results of Lemma 12, Then we have

$$
\mathbb { P } \Bigg ( \Big | \widehat { R } _ { s , a } ^ { \mathrm { r o b . } } - R _ { s , a } ^ { \mathrm { r o b . } } \Big | \geqslant 2 \theta \quad + \frac { \sqrt { 2 } C _ { p } ^ { 2 } R _ { \operatorname* { m a x } } } { ( C _ { p } - 1 ) \sqrt { n } } \bigg ( \sqrt { \log \bigg ( \frac { 2 ( 1 + C _ { p } R _ { \operatorname* { m a x } } / ( \theta ( C _ { p } - 1 ) ) ) } { \delta } \bigg ) } + 1 \bigg ) \bigg ) < \delta\tag{33}
$$

Set $\theta = \varepsilon / 4$ with $\begin{array} { r } { \varepsilon = 2 \frac { \sqrt { 2 } C _ { p } ^ { 2 } R _ { \mathrm { m a x } } } { ( C _ { p } - 1 ) \sqrt { n } } \left( \sqrt { \log { \left( \frac { 2 ( 1 + C _ { p } R _ { \mathrm { m a x } } / ( \theta ( C _ { p } - 1 ) ) ) } { \delta } \right) } } + 1 \right) } \end{array}$ , then

$$
\left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } R _ { \operatorname* { m a x } } } - 1 \right) ^ { 2 } = \log \left( \frac { 2 ( 1 + C _ { p } R _ { \operatorname* { m a x } } / ( \theta ( C _ { p } - 1 ) ) ) } { \delta } \right)\tag{34}
$$

$$
\Rightarrow \exp \left\{ - \left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } R _ { \mathrm { m a x } } } - 1 \right) ^ { 2 } \right\} = \frac { \delta } { 2 ( 1 + C _ { p } R _ { \mathrm { m a x } } / ( \theta ( C _ { p } - 1 ) ) ) }\tag{35}
$$

$$
\Rightarrow \mathbb { P } \left( \left| \widehat { R } _ { s , a } ^ { \mathrm { r o b } x } - R _ { s , a } ^ { \mathrm { r o b } x } \right| \geqslant \varepsilon \right) < 2 \left( 1 + C _ { p } R _ { \operatorname* { m a x } } / ( \frac { \varepsilon ( C _ { p } - 1 ) } { 4 } ) \right) \exp \left\{ - \left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } R _ { \operatorname* { m a x } } } - 1 \right) ^ { 2 } \right\}\tag{36}
$$

Lemma 13. Consider an MDP with the Wasserstein distance $D _ { W }$ . Fix any $s , a \in \mathcal S \times \mathcal A ,$ we can derive

$$
\begin{array} { r } { \operatorname* { i n f } _ { D _ { W } ( P | | P ^ { o } ) \leqslant \rho } \mathbb { E } _ { P } [ l ( X ) ] = - \operatorname* { i n f } _ { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho { r } ] } \left( \lambda \rho ^ { p } \ - \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { p } } \left[ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] \right) . } \end{array}
$$

Proof. Fix any $( s , a ) \in S \times A .$ . We have

$$
\begin{array} { r l } & { \underset { D _ { W } ( P | | P ^ { o } ) \leqslant \rho } { \operatorname* { i n f } } \mathbb { E } _ { R \sim P } [ l ( X ) ] = - \underset { D _ { W } ( P | | P ^ { o } ) \leqslant \rho } { \operatorname* { s u p } } \mathbb { E } _ { R \sim P } [ - l ( X ) ] } \\ & { \stackrel { ( a ) } { = } - \underset { \lambda \geqslant 0 } { \operatorname* { i n f } } \left\{ \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } \left[ \underset { R ^ { \prime \prime } \in \mathcal { R } } { \operatorname* { s u p } } \left\{ - R ^ { \prime \prime } - \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] + \lambda \rho ^ { p } \right\} } \\ & { = \underset { \lambda \geqslant 0 } { \operatorname* { s u p } } \left\{ \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } \left[ \underset { R ^ { \prime \prime } \in \mathcal { R } } { \operatorname* { i n f } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] - \lambda \rho ^ { p } \right\} } \\ & { \stackrel { ( b ) } { = } \underset { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } ] } { \operatorname* { s u p } } \left\{ \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } \left[ \underset { R ^ { \prime \prime } \in \mathcal { R } } { \operatorname* { i n f } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] - \lambda \rho ^ { p } \right\} , } \end{array}
$$

where (a) follows from ((Gao and Kleywegt, 2023) Theorem 1). For (b), let us first denote any optimizer in (a) to be $\lambda ^ { * }$ Observe that since R is non-negative, it follows that

$$
0 \leqslant - \lambda ^ { * } \rho ^ { p } + \mathbb { E } _ { R ^ { \prime } \sim v _ { \varepsilon , \sigma } ^ { \sigma } } \left[ \operatorname* { i n f } _ { R ^ { \prime } } \{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \} \right] \leqslant - \lambda ^ { * } \rho ^ { p } + \mathbb { E } _ { R ^ { \prime } \sim v _ { \varepsilon , \sigma } ^ { \sigma } } \big [ R ^ { \prime } + \lambda d ^ { p } \left( R ^ { \prime } , R ^ { \prime } \right) \big ] \leqslant - \lambda ^ { * } \rho ^ { p } + R _ { \operatorname* { m a x } }
$$

where in the last inequality we use that the distance metric satisfies $d ( R , R ) = 0 $ , for any $R \in \mathcal R$

Lemma 14. (Covering number (Wasserstein)). Consider the following set $o f \mathbb { R } ^ { | \mathcal { R } | }$ vectors:

$$
\mathcal { U } _ { \rho , R } = \left\{ \left( \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , 1 \right) \right\} , \ldots , \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , | \mathcal { R } | \right) \right\} \right) ^ { T } : \lambda \in \left[ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } \right] \right\}
$$

Let

$$
\mathcal N _ { \rho , R } ( \theta ) = \left\{ \left( \operatorname* { i n f } _ { R ^ { \prime } \in \mathcal R } \{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , 1 \right) \} , \dots , \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal R } \{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , | \mathcal R | \right) \} \right) ^ { T } : \lambda \in \left\{ \frac { \theta } { B _ { p } } , \frac { 2 \theta } { B _ { p } } , \dots , N _ { \rho , \theta } \frac { \theta } { B _ { p } } \right\} \right\} ,
$$

where $\begin{array} { r } { N _ { \rho , \theta } = \bigg \lceil \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } \bigg \rceil } \end{array}$ and $B _ { p } = \operatorname* { m a x } _ { R ^ { \prime } , R ^ { \prime \prime } } d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right)$ . Then $\mathcal { N } _ { \rho , R } ( \theta )$ is a θ-cover of $\mathcal { U } _ { \rho , R }$ with respect $t o \parallel \cdot \parallel _ { \infty }$ and its cardinality is bounded as $\begin{array} { r } { | \mathcal { N } _ { \rho , R } ( \theta ) | \leqslant \frac { R _ { \mathrm { m a x } } B _ { p } + ( R _ { \mathrm { m a x } } \vee \rho ^ { p } ) } { \rho ^ { p } \theta } } \end{array}$ . Furthermore, for any $\nu \in \mathcal { N } _ { \rho , R } ( \theta )$ , we have $\| \nu \| _ { \infty } \leqslant$ $\frac { R \operatorname* { m a x } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } }$

Proof. Fix any $\theta \in \mathsf { \Gamma } ( 0 , 1 )$ . First note that $N _ { \rho , \theta }$ is the minimal number of subintervals of length $\frac { \theta } { B _ { p } }$ needed to cover $[ 0 , R _ { \mathrm { m a x } } / \rho ^ { p } ]$ . Denote $\begin{array} { r } { J _ { i } = \left\lceil ( i - 1 ) \frac { \theta } { B _ { p } } , i \frac { \theta } { B _ { p } } \right\rceil , 1 \leqslant i \leqslant N _ { \rho , \theta } . } \end{array}$ . Fix some $\mu \in \mathcal { U } _ { \rho , R }$ . Then $\mu$ must takes the form

$$
\mu = \left( \operatorname* { i n f } _ { R ^ { \prime \prime } \in { \mathcal R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , 1 \right) \right\} , \ldots , \operatorname* { i n f } _ { R ^ { \prime \prime } \in { \mathcal R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , | { \mathcal R } | \right) \right\} \right) ^ { T } ,
$$

for some $\lambda \in [ 0 , R _ { \mathrm { m a x } } / \rho ^ { p } ]$ . Without loss of generality, assume $\lambda \in J _ { i }$ . Now we pick

$$
\nu = \left( \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + i \frac { \theta } { B _ { p } } d ^ { p } \left( R ^ { \prime \prime } , 1 \right) \right\} , \ldots , \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + i \frac { \theta } { B _ { p } } d ^ { p } \left( R ^ { \prime \prime } , | \mathcal { R } | \right) \right\} \right) ^ { T }
$$

Fix any $R ^ { \prime } \in \mu$ and $R ^ { \prime \prime } \in \nu ,$ we have

$$
\begin{array} { r l } { | R ^ { \prime } - R ^ { \prime \prime } | = \displaystyle \left| \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \} - \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + i \frac { \theta } { B _ { p } } d ^ { p } \left( R ^ { \prime \prime } , R \right) \right\} \right| } & { { } } \\ { \overset { ( a ) } { \leqslant } } & { { } \underset { R ^ { \prime \prime } \in \mathcal { R } } { \operatorname* { s u p } } \left| \left( \lambda - i \frac { \theta } { B _ { p } } \right) d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right| } \\ { \leqslant } & { { } \left| \lambda - i \frac { \theta } { B _ { p } } \right| \underset { R ^ { \prime } , R ^ { \prime \prime } } { \operatorname* { m a x } } d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) = \left| \lambda - i \frac { \theta } { B _ { p } } \right| B _ { p } } \\ { \leqslant \left| ( i - 1 ) \frac { \theta } { B _ { p } } - i \frac { \theta } { B _ { p } } \right| B _ { p } = \theta } \end{array}
$$

where (a) is due to $\begin{array} { r } { \left| \operatorname* { i n f } _ { x } f ( x ) - \operatorname* { i n f } _ { x } g ( x ) \right| \leqslant \operatorname* { s u p } _ { x } | f ( x ) - g ( x ) | } \end{array}$ . Taking maximum over $R ^ { \prime } \in \mathcal { R }$ on both sides, we get $\| \mu - \nu \| _ { \infty } \leqslant \theta$ . Since $\nu \in \mathcal { N } _ { \rho , R } ( \theta )$ , this suggests that $\mathcal { N } _ { \rho , R } ( \theta )$ is a θ-cover for $\mathcal { U } _ { \rho , R }$ To bound the cardinality of $\mathcal { N } _ { \rho , R } ( \theta )$ , we consider two cases. I $: 0 < \rho < 1$ , then $\rho ^ { p } \theta < 1$ and

$$
\left\lceil { \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } } \right\rceil \leqslant { \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } } + 1 \leqslant { \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } } + { \frac { R _ { \mathrm { m a x } } } { \rho ^ { p } \theta } } = { \frac { R _ { \mathrm { m a x } } B _ { p } + R _ { \mathrm { m a x } } } { \rho ^ { p } \theta } }
$$

On the other hand, if $\rho > 1$ , then since $\theta \in ( 0 , 1 )$ , we have

$$
\left\lceil \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } \right\rceil \leqslant \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } + 1 = \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } + \frac { \rho ^ { p } \theta } { \rho ^ { p } \theta } \leqslant \frac { R _ { \mathrm { m a x } } B _ { p } } { \rho ^ { p } \theta } + \frac { \rho ^ { p } } { \rho ^ { p } \theta } = \frac { R _ { \mathrm { m a x } } B _ { p } + \rho ^ { p } } { \rho ^ { p } \theta }
$$

Hence, we have $\begin{array} { r } { | \mathcal { N } _ { \rho , R } ( \theta ) | = N _ { \rho , \theta } \leqslant \frac { R _ { \mathrm { m a x } } B _ { p } + ( R _ { \mathrm { m a x } } \vee \rho ^ { p } ) } { \rho ^ { p } \theta } } \end{array}$ . Now we prove the last claim. Fix any $\nu \in \mathcal { N } _ { \rho , R }$ . Note that for any $R ^ { \prime } \in \mathcal { R }$

$$
R ^ { \prime } = \operatorname* { i n f } _ { R ^ { \prime \prime } \in R } \{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \} \leqslant R _ { \operatorname* { m a x } } + \lambda B _ { p } \leqslant R _ { \operatorname* { m a x } } + \frac { R _ { \operatorname* { m a x } } } { \rho ^ { p } } B _ { p } = \frac { R _ { \operatorname* { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } }
$$

The result then follows from taking maximum over $R ^ { \prime } \in \mathcal { R }$ on both sides.

Lemma 15. Fix any $( s , a ) \in S \times A .$ Let $\mathcal { N } _ { \rho , R } ( \theta )$ be the θ-cover ofthe set

$$
\mathcal { U } _ { \rho , R } = \left\{ \left( \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , 1 \right) \right\} , \ldots , \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , | \mathcal { R } | \right) \right\} \right) ^ { T } : \lambda \in \left[ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } \right] \right\}
$$

as described in Lemma 14. We then have

$$
\begin{array} { r l } { \underset { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } ] } { \operatorname* { s u p } } \big | \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } [ \underset { R ^ { \prime \prime } \in \mathbb { R } } { \operatorname* { i n f } } \{ R ^ { \prime \prime } + \lambda d ^ { p } ( R ^ { \prime \prime } , R ^ { \prime } ) \} ] - \mathbb { E } _ { R ^ { \prime } \sim \hat { \nu } _ { s , a } ^ { o } } [ \underset { R ^ { \prime \prime } \in \mathbb { R } } { \operatorname* { i n f } } \{ R ^ { \prime \prime } + \lambda d ^ { p } ( R ^ { \prime \prime } , R ^ { \prime } ) \} ] | } & { } \\ { \leqslant \underset { r \in N _ { \rho , R } ( \theta ) } { \operatorname* { m a x } } [ \hat { \nu } _ { s , a } ^ { o } r - \nu _ { s , a } ^ { o } r ] + 2 \theta . } \end{array}
$$

Proof. The proof is identical to the proof of Lemma 9.

Lemma 16. Fix any $( s , a ) \in S \times A .$ . For any $\theta , \delta \in ( 0 , 1 )$ and $\rho > 0$ , we have the following inequality with probability at least $1 - \delta$

$$
\bigg | \widehat { R } _ { s , a } ^ { \mathrm { r o b } \nu } - R _ { s , a } ^ { \mathrm { r o b } \nu } \bigg | \leqslant \frac { R _ { \operatorname* { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } } \sqrt { \frac { \log \Big ( \frac { 2 R _ { \operatorname* { m a x } } B _ { p } + 2 ( R _ { \operatorname* { m a x } } \vee \rho ^ { p } ) } { \rho ^ { p } \theta \delta } \Big ) } { 2 n } } + 2 \theta
$$

where $B _ { p } = \operatorname* { m a x } _ { R ^ { \prime } , R ^ { \prime \prime } } d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right)$

Proof. From Lemma 13 , we have

$$
R _ { s , a } ^ { \mathrm { r o b } \nu } = \operatorname* { s u p } _ { \lambda \in \left[ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } \right] } \left\{ \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } \left[ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] - \lambda \rho ^ { p } \right\} ,
$$

$$
\widehat { R } _ { s , a } ^ { \mathrm { r o b } \prime } = \operatorname* { s u p } _ { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } ] } \left\{ \mathbb { E } _ { R ^ { \prime } \sim \widehat { \nu } _ { s , a } ^ { o } } \left[ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] - \lambda \rho ^ { p } \right\} .
$$

Now it follows that

$$
\left. \widehat { R } _ { s , a } ^ { \mathrm { r o b } \nu } - R _ { s , a } ^ { \mathrm { r o b } \nu } \right. = \vert \operatorname* { s u p } _ { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } ] } \left\{ \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } \left[ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] - \lambda \rho ^ { p } \right\}\tag{37}
$$

$$
- \operatorname* { s u p } _ { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } ] } \{ \mathbb { E } _ { R ^ { \prime } \sim \widehat { \nu } _ { s , a } ^ { o } } [ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \{ R ^ { \prime \prime } + \lambda d ^ { p } ( R ^ { \prime \prime } , R ^ { \prime } ) \} ] - \lambda \rho ^ { p } \}\tag{38}
$$

$$
\stackrel { ( a ) } { \leqslant } \operatorname* { s u p } _ { \lambda \in [ 0 , R _ { \operatorname* { m a x } } / \rho ^ { p } ] } | \mathbb { E } _ { R ^ { \prime } \sim \nu _ { s , a } ^ { o } } \left[ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right]\tag{39}
$$

$$
- \mathbb { E } _ { R ^ { \prime } \sim \widehat { \nu } _ { s , a } ^ { o } } \left[ \operatorname* { i n f } _ { R ^ { \prime \prime } \in \mathcal { R } } \left\{ R ^ { \prime \prime } + \lambda d ^ { p } \left( R ^ { \prime \prime } , R ^ { \prime } \right) \right\} \right] |\tag{40}
$$

$$
\stackrel { ( b ) } { \leqslant } \operatorname* { m a x } _ { r \in \mathcal { N } _ { \rho , R } \left( \theta \right) } \left| \widehat { \nu } _ { s , a } ^ { o } r - \nu _ { s , a } ^ { o } r \right| + 2 \theta\tag{41}
$$

where (a) follows from |sup ${ \mathrm { \rangle } } _ { x } f ( x ) - \operatorname* { s u p } _ { x } g ( x ) | \leqslant \operatorname* { s u p } _ { x } | f ( x ) - g ( x ) |$ . (b) follows from Lemma 15. Recall that all $\nu \in \mathcal { N } _ { \rho , R } ( \theta )$ is bounded by $\begin{array} { r } { \nu _ { \mathrm { m a x } } : = \frac { R _ { \mathrm { m a x } } ( B _ { p } + \rho ^ { p } ) } { \rho ^ { p } } } \end{array}$ . Now we can apply Hoeffding’s inequality:

$$
\mathbb { P } \left( \left| \widehat { \nu } _ { s , a } ^ { o } r - \nu _ { s , a } ^ { o } r \right| \geqslant \varepsilon \right) \leqslant 2 \exp \left( - \frac { 2 n \varepsilon ^ { 2 } } { \nu _ { \operatorname* { m a x } } ^ { 2 } } \right) = 2 \exp \left( - \frac { 2 n \varepsilon ^ { 2 } } { \left( \frac { R _ { \operatorname* { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } } \right) ^ { 2 } } \right) , \quad \forall \varepsilon > 0
$$

Now recall that $\begin{array} { r } { | \mathcal { N } _ { \rho , R } ( \theta ) | \leqslant \frac { R _ { \mathrm { m a x } } B _ { p } + ( R _ { \mathrm { m a x } } \vee \rho ^ { p } ) } { \rho ^ { p } \theta } } \end{array}$ and choose

$$
\varepsilon = \frac { R _ { \mathrm { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } } \sqrt { \frac { \log \left( 2 \left| \mathcal { N } _ { \rho , R } ( \theta ) \right| / \delta \right) } { 2 N } }
$$

We then have

$$
\mathbb { P } \left( \left| \widehat { \nu } _ { s , a } ^ { o } r - \nu _ { s , a } ^ { o } r \right| \geqslant \frac { R _ { \operatorname* { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } } \sqrt { \frac { \log \left( 2 \left| \mathcal { N } _ { \rho , R } ( \theta ) \right| / \delta \right) } { 2 n } } \right) \leqslant \frac { \delta } { \left| \mathcal { N } _ { \rho , R } ( \theta ) \right| } .
$$

Finally, applying a union bound over $\mathcal { N } _ { \rho , R } ( \theta )$ , we get

$$
\operatorname* { m a x } _ { r \in \mathcal { N } _ { \rho , R } ( \theta ) } \left. \widehat { \nu } _ { s , a } ^ { o } r - \nu _ { s , a } ^ { o } r \right. \leqslant \frac { R _ { \operatorname* { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } } \sqrt { \frac { \log \left( \frac { 2 R _ { \operatorname* { m a x } } B _ { p } + 2 ( R _ { \operatorname* { m a x } } \sqrt { \rho ^ { p } } ) } { \rho ^ { p } \theta \delta } \right) } { 2 n } } ,
$$

with probability at least $1 - \delta$ . Combining the above and equation 41 completes the proof.

Similarly, Set $\theta = \varepsilon / 4$ with $\begin{array} { r } { \varepsilon = \frac { 2 R _ { \mathrm { m a x } } \left( B _ { p } + \rho ^ { p } \right) } { \rho ^ { p } } \sqrt { \frac { \log \left( \frac { 2 R _ { \mathrm { m a x } } B _ { p } + 2 R _ { \mathrm { m a x } } \sqrt { \rho ^ { p } } } { \rho ^ { p } \theta \delta } \right) } { 2 n } } } \end{array}$ , then

$$
\exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 R _ { \mathrm { m a x } } ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\} = \frac { \rho ^ { p } \theta \delta } { 2 R _ { \mathrm { m a x } } B _ { p } + 2 R _ { \mathrm { m a x } } \sqrt { \rho ^ { p } } }\tag{42}
$$

$$
\Rightarrow \delta = \frac { 4 \left( 2 R _ { \mathrm { m a x } } B _ { p } + 2 R _ { \mathrm { m a x } } \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 R _ { \mathrm { m a x } } ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\}\tag{43}
$$

so that

$$
\mathbb { P } \left( \left| \widehat { R } _ { s , a } ^ { \mathrm { r o b } _ { \mathcal { W } } } - R _ { s , a } ^ { \mathrm { r o b } _ { \mathcal { W } } } \right| \geqslant \varepsilon \right) < \frac { 4 \left( 2 R _ { \operatorname* { m a x } } B _ { p } + 2 R _ { \operatorname* { m a x } } \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 R _ { \operatorname* { m a x } } ^ { 2 } \left( B _ { p } + \rho ^ { p } \right) ^ { 2 } } \right\} .\tag{44}
$$

## C. Convergence of Robust-Power-UCT Multi-armed bandits

Lemma 17. For $m \in [ M ] ,$ , let $( \widehat { V } _ { m , n } ) _ { n \geqslant 1 }$ be a sequence of estimator satisfying ${ \widehat V } _ { m , n } \underset { n  \infty } { \stackrel { \alpha , \beta } { \longrightarrow } } V _ { m } ,$ , and there exists a constan L such that $\widehat { V } _ { m , n } \leqslant L , \forall n \geqslant 1 . L e t X _ { i }$ be an iid sequence from a distribution $\nu ^ { o }$ with mean µ and $S _ { i }$ be an iid sequence from a distribution $p = ( p _ { 1 } , \dotsc , p _ { M } )$ supported on $\{ 1 , \ldots , M \}$ . Introducing the random variables $N _ { m } ^ { n } = \# | \{ i \leqslant n :$ $S _ { i } = s _ { m } \} |$ . Let us study a random vector $\begin{array} { r } { \widehat { p } _ { n } = ( \frac { N _ { 1 } ^ { n } } { n } , \frac { \tilde { N } _ { 2 } ^ { n } } { n } , . . . , \frac { N _ { M } ^ { n ^ { \prime } } } { n } ) } \end{array}$ . We define an estimate of $\mathcal { \nu } ^ { o }$ as

$$
\nu ^ { o } \approx \widehat { \nu } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i } } ,
$$

where $\delta _ { X _ { i } }$ is a point mass at $X _ { i }$ . And define $\begin{array} { r } { R ^ { \mathcal { R } } = \operatorname* { m i n } _ { r \sim \mathcal { R } } \mathbb { E } _ { R \sim r } [ R ] } \end{array}$ w.r.t $\nu ^ { o }$ and $\begin{array} { r } { R ^ { \widehat { \mathcal { R } } } = \operatorname* { m i n } _ { r \sim \widehat { \mathcal { R } } } \mathbb { E } _ { R \sim r } [ R ] } \end{array}$ w.r.t $\widehat { \nu } _ { n }$ . We define the sequence of estimator

$$
\widehat { Q } _ { n } = R ^ { \widehat { \mathcal { R } } } + \gamma \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) .
$$

Then with $2 \alpha \leqslant \beta , \beta > 1$

$$
{ \widehat { Q } } _ { n } \underset { n  \infty } { \stackrel { \alpha , \beta } {  } } R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) .
$$

Proof. Let $p = ( p _ { 1 } , p _ { 2 } , . . . p _ { M } ) , p \in \triangle ^ { M }$ where $\begin{array} { r } { \bigtriangleup ^ { M } = \{ x \in \mathbb { R } ^ { M } : \sum _ { i = 1 } ^ { M } x _ { i } = 1 , x _ { i } \geqslant 0 \} } \end{array}$ is the $( M - 1 )$ -dimensional simplex. Without loss of generality, we assume that $p _ { m } > 0$ for all $m .$ Let us define $V = ( V _ { 1 } , V _ { 2 } , . . . V _ { M } )$ . Let $\widehat { V } _ { n } =$ $\big ( \widehat { V } _ { 1 , N _ { 1 } ^ { n } } , \widehat { V } _ { 2 , N _ { 2 } ^ { n } } , . . . , \widehat { V } _ { M , N _ { M } ^ { n } } \big ) , \sum _ { i = 1 } ^ { M } N _ { i } ^ { n } = n , N _ { i } ^ { n }$ is the number of times that population i was observed. We have $\widehat { Q } _ { n } =$ $R ^ { \widehat { \mathcal { R } } } + \gamma \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } )$ . Therefore,

$$
\mathbb { P } \bigg ( \vert \widehat { Q } _ { n } - \big ( R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) \big ) \vert \geqslant \varepsilon \bigg ) \leqslant \mathbb { P } \bigg ( \vert R ^ { \widehat { \mathcal { R } } } - R ^ { \mathcal { R } } \vert \geqslant \frac 1 2 \varepsilon \bigg ) + \mathbb { P } \bigg ( \vert \gamma \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) - \gamma \sigma _ { p } ( V ) \vert \geqslant \frac 1 2 \varepsilon \bigg )\tag{45}
$$

$$
\leqslant \mathbb { P } \bigg ( | R ^ { \widehat { \mathcal { R } } } - R ^ { \mathcal { R } } | \geqslant \frac { 1 } { 2 } \varepsilon \bigg ) + \mathbb { P } \bigg ( | \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) - \sigma _ { p } ( V ) | \geqslant \frac { 1 } { 2 \gamma } \varepsilon \bigg ) .\tag{46}
$$

To upper bound A,

• TV:

$$
A \leqslant \frac { 1 6 R _ { \mathrm { m a x } } \exp \{ - n ( \varepsilon / 4 ) ^ { 2 } / 2 R _ { \mathrm { m a x } } ^ { 2 } \} } { \varepsilon / 4 }\tag{47}
$$

• Chi-square:

$$
A \leqslant 2 \left( 1 + C _ { p } R _ { \operatorname* { m a x } } / ( \frac { \varepsilon ( C _ { p } - 1 ) } { 4 } ) \right) \exp \left\{ - \left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } R _ { \operatorname* { m a x } } } - 1 \right) ^ { 2 } \right\}\tag{48}
$$

• Wasserstein:

$$
A \leqslant \frac { 4 \left( 2 R _ { \mathrm { m a x } } B _ { p } + 2 R _ { \mathrm { m a x } } \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 R _ { \mathrm { m a x } } ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\}\tag{49}
$$

To upper bound B, let us consider $\sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) - \sigma _ { p } ( V ) = ( \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) - \sigma _ { p } ( \widehat { V } _ { n } ) ) + ( \sigma _ { p } ( \widehat { V } _ { n } ) - \sigma _ { p } ( V ) )$ . Then,

$$
B \leqslant \underbrace { \mathbb { P } \bigg ( \left| \sigma _ { \widehat { p } _ { n } } ( \widehat { V } _ { n } ) - \sigma _ { p } ( \widehat { V } _ { n } ) \right| \geqslant \frac { 1 } { 4 \gamma } \varepsilon \bigg ) } _ { B _ { 1 } } + \underbrace { \mathbb { P } \bigg ( \left| \sigma _ { p } ( \widehat { V } _ { n } ) - \sigma _ { p } ( V ) \right| \geqslant \frac { 1 } { 4 \gamma } \varepsilon \bigg ) } _ { B _ { 2 } } .\tag{50}
$$

By applying results from Lemma 2 and Equation 8, we obtain

• TV:

$$
B _ { 1 } \leqslant \frac { 1 6 H \exp \{ - n ( \varepsilon / 4 \gamma ) ^ { 2 } / 2 H ^ { 2 } \} } { \varepsilon / 4 \gamma }\tag{51}
$$

• Chi-square:

$$
B _ { 1 } \leqslant 2 \left( 1 + C _ { p } H / ( \frac { \varepsilon ( C _ { p } - 1 ) } { 4 } ) \right) \exp \left\{ - \left( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } H } - 1 \right) ^ { 2 } \right\}\tag{52}
$$

• Wasserstein:

$$
B _ { 1 } \leqslant \frac { 4 \left( 2 H B _ { p } + 2 H \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 H ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\}\tag{53}
$$

For $B _ { 2 }$ , as the result from Lemma 1, we have

$$
\begin{array} { r } { \left| \sigma _ { p } ( \widehat { V } _ { n } ) - \sigma _ { p } ( V ) \right| \leqslant \| \widehat { V } _ { n } - V \| _ { \infty } \leqslant \| \widehat { V } _ { n } - V \| _ { 1 } } \end{array}
$$

Therefore,

$$
B _ { 2 } \leqslant \mathbf { P r } \left( \sum _ { m = 1 } ^ { M } | \widehat { V } _ { m , N _ { m } ^ { n } } - V _ { m } | \geqslant \frac { 1 } { 4 \gamma } \varepsilon \right) \leqslant \sum _ { m = 1 } ^ { M } \mathbf { P r } \left( | \widehat { V } _ { m , N _ { m } ^ { n } } - V _ { m } | \geqslant \frac { 1 } { 4 \gamma M } \varepsilon | N _ { m } ^ { n } \right)\tag{54}
$$

$$
\leqslant \sum _ { m = 1 } ^ { M } \mathbb { E } \left[ \mathbb { P } \left( \frac { 1 } { N _ { m } ^ { n } } \sum _ { t = 1 } ^ { N _ { m } ^ { n } } V _ { m , t } - V _ { m } \geqslant \frac { 1 } { 4 \gamma M } \varepsilon | N _ { m } ^ { n } \right) \right]\tag{55}
$$

$$
\leqslant \sum _ { m = 1 } ^ { M } \mathbb { E } \left[ c ( N _ { m } ^ { n } ) ^ { - \alpha } ( \frac { \varepsilon } { 4 \gamma M } ) ^ { - \beta } \right] .\tag{56}
$$

Let us define an event $\begin{array} { r } { \mathcal { E } = \bigg \{ N _ { m } ^ { n } > \frac { n p _ { m } } { 2 } \bigg \} } \end{array}$ . Therefore,

$$
B _ { 2 } \leqslant \sum _ { m = 1 } ^ { M } \mathbb { E } \bigg [ c ( \frac { n p _ { m } } { 2 } ) ^ { - \alpha } ( \frac { \varepsilon } { 4 \gamma M } ) ^ { - \beta } \bigg ] + \sum _ { m = 1 } ^ { M } \mathbb { E } \bigg [ \mathbb { P } ( N _ { m } ^ { n } \leqslant \frac { n p _ { m } } { 2 } ) \bigg ]\tag{57}
$$

$$
= \sum _ { m = 1 } ^ { M } ( c 2 ^ { \alpha + 2 \beta } \gamma ^ { \beta } p _ { m } ^ { - \alpha } M ^ { \beta } ) n ^ { - \alpha } \varepsilon ^ { - \beta } + \sum _ { m = 1 } ^ { M } \mathbb { E } \bigg [ \mathbb { P } ( N _ { m } ^ { n } - p _ { m } n \leqslant - \frac { p _ { m } n } { 2 } ) \bigg ]\tag{58}
$$

$$
\leqslant \sum _ { m = 1 } ^ { M } ( c 2 ^ { \alpha + 2 \beta } \gamma ^ { \beta } p _ { m } ^ { - \alpha } M ^ { \beta } ) n ^ { - \alpha } \varepsilon ^ { - \beta } + \sum _ { m = 1 } ^ { M } \exp \bigg \{ - 2 n ( \frac { p _ { m } n } { 2 } ) ^ { 2 } \bigg \}\tag{59}
$$

Therefore,

• TV:

(60)

$$
\begin{array} { l } { { A + B \leqslant A + B _ { 1 } + B _ { 2 } \leqslant \frac { 1 6 R _ { \mathrm { m a x } } \exp \left\{ - n ( \varepsilon / 4 ) ^ { 2 } / 2 R _ { \mathrm { m a x } } ^ { 2 } \right\} } { \varepsilon / 4 } + \frac { 1 6 H \exp \left\{ - n ( \varepsilon / 4 \gamma ) ^ { 2 } / 2 H ^ { 2 } \right\} } { \varepsilon / 4 \gamma } } } \\ { { \ \qquad + \displaystyle \sum _ { m = 1 } ^ { M } ( c 2 ^ { \alpha + 2 \beta } \gamma ^ { \beta } p _ { m } ^ { - \alpha } M ^ { \beta } ) n ^ { - \alpha } \varepsilon ^ { - \beta } + \displaystyle \sum _ { m = 1 } ^ { M } \exp \bigg \{ - 2 n ( \frac { p _ { m } n } { 2 } ) ^ { 2 } \bigg \} . } } \end{array}\tag{61}
$$

• Chi-Square:

62)

$$
\begin{array} { r l } & { A + B \leqslant A + B _ { 1 } + B _ { 2 } \leqslant 2 ( 1 + C _ { p } R _ { \operatorname* { m a x } } / ( \frac { \varepsilon ( C _ { p } - 1 ) } { 4 } ) ) \exp \{ - ( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } R _ { \operatorname* { m a x } } } - 1 ) ^ { 2 } \} } \\ & { \quad +  2 ( 1 + C _ { p } H / ( \frac { \varepsilon ( C _ { p } - 1 ) } { 4 } ) ) \exp \{ - ( \frac { ( C _ { p } - 1 ) \sqrt { n } \varepsilon } { 2 \sqrt { 2 } C _ { p } ^ { 2 } H } - 1 ) ^ { 2 } \} + \displaystyle \sum _ { m = 1 } ^ { M } ( c 2 ^ { \alpha + 2 \beta } \gamma ^ { \beta } p _ { m } ^ { - \alpha } M ^ { \beta } ) n ^ { - \alpha } \varepsilon ^ { - \beta } } \\ & { \quad + \displaystyle \sum _ { m = 1 } ^ { M } \exp \{ - 2 n ( \frac { p _ { m } n } { 2 } ) ^ { 2 } \} . } \end{array}\tag{63}
$$

64)

• Wasserstein:

$$
\begin{array} { r l } & { A + B \leqslant A + B _ { 1 } + B _ { 2 } \leqslant \frac { 4 \left( 2 R _ { \operatorname* { m a x } } B _ { p } + 2 R _ { \operatorname* { m a x } } \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 R _ { \operatorname* { m a x } } ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\} } \\ & { \quad \quad \quad + \frac { 4 \left( 2 H B _ { p } + 2 H \sqrt { \rho ^ { p } } \right) } { \rho ^ { p } \varepsilon } \exp \left\{ - \frac { 2 n \varepsilon ^ { 2 } \rho ^ { 2 p } } { 4 H ^ { 2 } ( B _ { p } + \rho ^ { p } ) ^ { 2 } } \right\} + \displaystyle \sum _ { m = 1 } ^ { M } ( c 2 ^ { \alpha + 2 \beta } \gamma ^ { \beta } p _ { m } ^ { - \alpha } M ^ { \beta } ) n ^ { - \alpha } \varepsilon ^ { - \beta } } \\ & { \quad \quad \quad + \displaystyle \sum _ { m = 1 } ^ { M } \exp \bigg \{ - 2 n ( \frac { p _ { m } n } { 2 } ) ^ { 2 } \bigg \} . } \end{array}\tag{65}
$$

(66)

(67)

In both three cases, that leads to

$$
\mathbb { P } \bigg ( \vert \widehat { Q } _ { n } - \big ( R ^ { \mathbb { R } } + \gamma \sigma _ { p } ( V ) \big ) \vert \geqslant \varepsilon \bigg ) \leqslant \mathcal { O } ( \exp \{ - n \} ) + \mathcal { O } ( e ^ { - 1 } \exp \{ - c n \varepsilon ^ { 2 } \} ) + \sum _ { m = 1 } ^ { M } ( c ) ^ { 2 \alpha + 2 \beta } \gamma ^ { \beta } p _ { m } ^ { - \alpha } M ^ { \beta } ) n ^ { - \alpha } \varepsilon ^ { - \beta }\tag{68}
$$

$$
+ \sum _ { m = 1 } ^ { M } \exp \bigg \{ - 2 n ( \frac { p _ { m } n } 2 ) ^ { 2 } \bigg \} \leqslant c ^ { ' } n ^ { - \alpha } \varepsilon ^ { - \beta } ,\tag{69}
$$

with $c ^ { ' } > 0$ depends on $c , M , \alpha , \beta , p _ { i }$ . Here we need

$$
2 \alpha \leqslant \beta ,\tag{70}
$$

to argue that $e ^ { - 1 } \exp ( - c n \varepsilon ^ { 2 } ) = \mathcal { O } ( n ^ { - \alpha } \varepsilon ^ { - \beta } )$ . Therefore, with $n \geqslant 1 , \varepsilon > 0$

$$
\begin{array} { r } { \mathbb P \bigg ( \Big | \widehat { Q } _ { n } - \big ( R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) \big ) \Big | \geqslant \varepsilon \bigg ) \leqslant c ^ { ' } n ^ { - \alpha } \varepsilon ^ { - \beta } . } \end{array}\tag{71}
$$

Furthermore,

$$
\operatorname* { l i m } _ { n \longrightarrow \infty } \mathbb { E } \left[ \left| { \widehat { Q } } _ { n } - \left( R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) \right) \right| \right]\tag{72}
$$

$$
= \operatorname* { l i m } _ { n \longrightarrow \infty } \int _ { 0 } ^ { \infty } \mathbb { P } \left( \left| { \widehat { Q } } _ { n } - \left( R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) \right) \right| \geqslant s \right) d s\tag{73}
$$

$$
\leqslant \operatorname* { l i m } _ { n \longrightarrow \infty } \left( \int _ { 0 } ^ { n ^ { - { \frac { \alpha } { \beta } } } } 1 d s + \int _ { n ^ { - { \frac { \alpha } { \beta } } } } ^ { + \infty } c ^ { ' } n ^ { - \alpha } s ^ { - \beta } \right)\tag{74}
$$

$$
= \operatorname* { l i m } _ { n \longrightarrow \infty } \left( n ^ { - \frac { \alpha } { \beta } } + c ^ { ' } n ^ { - \alpha } \left( \frac { s ^ { - \beta + 1 } } { - \beta + 1 } + C \right) \Big | _ { n ^ { - \frac { \alpha } { \beta } } } ^ { + \infty } \right) = 0\tag{75}
$$

$$
= \operatorname* { l i m } _ { n \longrightarrow \infty } \left( n ^ { - { \frac { \alpha } { \beta } } } - c ^ { ' } n ^ { - \alpha } \left( { \frac { n ^ { \frac { \alpha ( \beta - 1 ) } { \beta } } } { - \beta + 1 } } \right) \right) = 0 { \mathrm { ~ } } ( { \mathrm { b e c a u s e ~ } } \alpha > 0 , \beta > 1 )\tag{76}
$$

so that,

$$
\operatorname* { l i m } _ { n \to \infty } \mathbb { E } [ \widehat { Q } _ { n } ] = R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) .
$$

This means

$$
\widehat { Q } _ { n } \underset { n  \infty } { \overset { \alpha , \beta } {  } } R ^ { \mathcal { R } } + \gamma \sigma _ { p } ( V ) ,
$$

which concludes the proof.

## D. Convergence of Robust-Power-UCT n Monte-Carlo Tree Search

Theorem 1. (Theorem 1 of Dam et al. (2024b)) For each arm $a \in [ K ]$ , let $\widehat { \mu } _ { a , n } \underset { n  \infty } { \overset { \alpha , \beta } {  } } \mu _ { a }$ and define $\mu _ { \star } = \operatorname* { m a x } _ { a } \{ \mu _ { a } \}$ Suppose arms are selected according to equation 4 with parameters $( \alpha , \beta , b , C )$ , and let $p \in [ 1 , \infty ) . I $ f

$$
\begin{array} { r } { 1 \leq p \leq 2 , \ a n d \ \alpha \leq \frac { \beta } { 2 } , \quad o r \quad p > 2 , \ 0 < \alpha - \frac { \beta } { p } < 1 , } \end{array}
$$

and

$$
\alpha \Big ( 1 - \textstyle { \frac { b } { \alpha } } \Big ) \ \leq \ b < \ \alpha ,
$$

then there exists a suitable constant C (depending on $K , b , \alpha , p , \Delta _ { \mathrm { m i n } } )$ such that

$$
\widehat { \mu } _ { n } ( p ) \mathop {  } _ { n  \infty } ^ { \alpha ^ { \prime } , \beta ^ { \prime } } \mu _ { \star } ,
$$

where $\begin{array} { r } { \Delta _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { a : \mu _ { a } < \mu _ { \star } } ( \mu _ { \star } - \mu _ { a } ) , \alpha ^ { \prime } = ( b - 1 ) \big ( 1 - \frac { b } { \alpha } \big ) } \end{array}$ , and $\beta ^ { \prime } = ( b - 1 )$ .

Theorem 2. When applying Robust-Power-UCT with parameters $\{ b _ { i } \} _ { i = 0 } ^ { H } , \{ \alpha _ { i } \} _ { i = 0 } ^ { H } ,$ , and $\{ \beta _ { i } \} _ { i = 0 } ^ { H }$ satisfying Table 1:

(i) For any node $s _ { h }$ at depth $h \in \{ 0 , \ldots , H \}$

$$
{ \widehat { V } } _ { n } ( s _ { h } ) { \overset { \alpha _ { h } , \beta _ { h } } { \underset { n  \infty } { \longrightarrow } } } { \widetilde V } ( s _ { h } ) .
$$

(ii) For any node $s _ { h }$ at depth $h \in \{ 0 , \ldots , H - 1 \}$

$$
\widehat { Q } _ { n } \big ( s _ { h } , a \big ) \stackrel { \alpha _ { h + 1 } , \beta _ { h + 1 } } { \underset { n  \infty } { \longrightarrow } } \widetilde { Q } ( s _ { h } , a ) , \quad \forall a \in \mathcal { A } _ { s _ { h } } .
$$

Proof. We follow the proof technique of Dam et al. (2024b, Theorem 2).

Base Case $( H = 1 )$ . Consider the root node $s _ { 0 }$ . Each time we visit $( s _ { 0 } , a )$ , we collect:

• $\mathbf { A }$ reward sample $r ^ { t } ( s _ { 0 } , a )$ from the reward distribution $\nu _ { s _ { 0 } , a } ^ { o } .$ , which then leads to evaluating $\widehat { \nu } _ { n }$ and $\widehat { R } _ { s _ { 0 } , a } ^ { \mathrm { r o b } } ~ =$ $\begin{array} { r } { \operatorname* { m i n } _ { { r \in \widehat { \mathcal { R } } _ { s _ { 0 } , a } } } \mathbb { E } _ { R \sim r } [ R ] } \end{array}$ , thus approximates the worst-case reward at $( s _ { 0 } , a )$

• A next state $s _ { 1 } \sim P _ { s _ { 0 } , a } ^ { o }$ from $M = | \mathcal { A } _ { s _ { 0 } } |$ possible states (denote such states as $S _ { 0 } ^ { 1 } )$ . This then leads to $\widehat { p } _ { s _ { 0 } , a }$ , and captures the worst-case value from the transition ambiguity set $\widehat { \mathcal { P } } _ { s _ { 0 } , a }$

By definition of the robust Bellman backup, recall

$$
\widetilde Q ( s _ { 0 } , a ) = R _ { s _ { 0 } , a } ^ { \mathrm { r o b } } + \gamma \sigma _ { \mathscr P _ { s _ { 0 } , a } } ( \widetilde V ) ,
$$

where $\begin{array} { r } { { R } _ { s _ { 0 } , a } ^ { \mathrm { r o b } } = \operatorname* { m i n } _ { r _ { s _ { 0 } , a } \in \mathcal { R } _ { s _ { 0 } , a } } \mathbb { E } [ r ] . } \end{array}$

Since $H = 1$ , the next state $s _ { 1 }$ is treated as a leaf. We approximate $\widetilde V ( s _ { 1 } ) \approx V _ { \pi _ { 0 } } ( s _ { 1 } )$ , i.i.d. rollout returns under the policy $\pi _ { 0 } .$ . By standard concentration bounds (e.g., Hoeffding), we obtain for all child nodes $s _ { 1 } \in S _ { 0 } ^ { 1 }$

$$
\widehat { V } _ { n } ( s _ { 1 } ) \begin{array} { c c c } { \alpha _ { 1 } , \beta _ { 1 } } & { \widetilde { V } ( s _ { 1 } ) . } \\ { n  \infty } & { } \end{array}\tag{77}
$$

Next, recall by equation 3:

$$
\widehat { Q } _ { n } ( s _ { 0 } , a ) \gets \widehat { R } _ { s _ { 0 } , a } ^ { \mathrm { r o b } } + \gamma \sigma _ { \widehat { \mathcal { P } } _ { s _ { 0 } , a } } ( \widehat { V } _ { T _ { s _ { 1 } } ( n ) } )
$$

Here $\widehat { V } _ { T _ { s _ { 1 } } ( n ) }$ is the estimated value at all child nodes $s _ { 1 } \in S _ { 0 } ^ { 1 }$ . By Lemma 17 and equation 77, it follows that

$$
\widehat { Q } _ { n } ( s _ { 0 } , a ) \stackrel { \alpha _ { 1 } , \beta _ { 1 } } {  } \ \widetilde { Q } ( s _ { 0 } , a ) .
$$

Since $s _ { 0 }$ is the root node, we perform the power-mean backup on $\{ \widehat { Q } _ { n } ( s _ { 0 } , a ) \}$

$$
\widehat V _ { n } ( s _ { 0 } ) = \Big ( \sum _ { a \in { \mathcal A } _ { s _ { 0 } } } \frac { T _ { s _ { 0 } , a } ( n ) } n \big [ \widehat Q _ { T _ { s _ { 0 } , a } ( n ) } ( s _ { 0 } , a ) \big ] ^ { p } \Big ) ^ { \frac 1 p } .
$$

Under Theorem 1 (from Dam et al. (2024b) for robust settings), we conclude

$$
\widehat { V } _ { n } ( s _ { 0 } ) \begin{array} { c c c } { \alpha _ { 0 } , \beta _ { 0 } } & { \widetilde { V } ( s _ { 0 } ) . } \\ { n  \infty } & { } \end{array}
$$

This establishes both points (i) and (ii) at depth 0 and confirms the result for $H = 1$

Inductive Step $( H > 1 )$ . Assume the theorem holds for all search trees up to depth $H - 1$ . We now add one more level to create a tree of depth H. Let $s _ { 1 }$ be a child of the new root $s _ { 0 } .$ . Then $s _ { 1 }$ itself is a root of a subtree with depth $( H - 1 )$ . By the inductive hypothesis:

$$
\widehat { V } _ { n } ( s _ { 1 } ) \underset { n  \infty } { \overset { \alpha _ { 1 } , \beta _ { 1 } } {  } } \widetilde { V } ( s _ { 1 } ) , \quad \widehat { Q } _ { n } ( s _ { 1 } , a ^ { \prime } ) \underset { n  \infty } { \overset { \alpha _ { 2 } , \beta _ { 2 } } {  } } \widetilde { Q } ( s _ { 1 } , a ^ { \prime } ) , \forall a ^ { \prime } .
$$

At the new root $s _ { 0 } ,$ we repeat the argument used in the base case:

• Observing rewards $r ^ { t } ( s _ { 0 } , a )$ from $\nu _ { s _ { 0 } , a } ^ { o } .$

• Transitioning under $P _ { s _ { 0 } , a } ^ { o }$ to state $s _ { 1 }$

Hence, Lemma 17 again implies

$$
\widehat { Q } _ { n } \bigl ( s _ { 0 } , a \bigr ) \begin{array} { l } { \stackrel { \alpha _ { 1 } , \beta _ { 1 } } {  } \ \widetilde { Q } ( s _ { 0 } , a ) , } \\ { n  \infty } \end{array}
$$

and the power-mean operator at $s _ { 0 }$ yields

$$
\widehat { V } _ { n } ( s _ { 0 } ) \begin{array} { c c c } { \alpha _ { 0 } , \beta _ { 0 } } & { \widetilde { V } ( s _ { 0 } ) . } \\ { n  \infty } & { } \end{array}
$$

Thus, depth H inherits the same concentration property from depth $( H - 1 )$ . This completes the inductive argument, establishing statements (i) and (ii) for any node at any depth $\leqslant H$

Theorem 3. (Convergence of Expected Payoff) At the root node $s _ { 0 }$ , there is a choice of parameters yielding

$$
\begin{array} { r } { \big | \mathbb { E } \big [ \widehat { V } _ { n } ( s _ { 0 } ) \big ] - \widetilde { V } ( s _ { 0 } ) \big | \ \leq \ \mathcal { O } \big ( n ^ { - 1 / 2 } \big ) . } \end{array}
$$

Proof. By Jensen’s inequality (convexity of |x|), we obtain

$$
\begin{array} { r l r } { \Bigm \lvert \mathbb { E } \big [ \widehat { V } _ { n } ( s _ { 0 } ) \bigm ] - \widetilde { V } ( s _ { 0 } ) \Bigm ] \ \leq \ \mathbb { E } \Big [ \Bigm \lvert \widehat { V } _ { n } ( s _ { 0 } ) - \widetilde { V } ( s _ { 0 } ) \Bigm \rvert \Big ] } & { } & \\ { \quad \quad \quad = \ \int _ { 0 } ^ { \infty } \mathbb { P } \Big ( \bigm \lvert \widehat { V } _ { n } ( s _ { 0 } ) - \widetilde { V } ( s _ { 0 } ) \bigm \rvert \ \geq \ s \Big ) d s . } & \end{array}
$$

Next, we split this integral at $s = n ^ { - \alpha _ { 0 } / \beta _ { 0 } }$ . Using the concentration property $\widehat { V } _ { n } \big ( s _ { 0 } \big ) \operatorname* { \alpha } _ { n  \infty } ^ { \alpha _ { 0 } , \beta _ { 0 } } \widetilde { V } \big ( s _ { 0 } \big )$ , we have

$$
\mathbb { P } \Big ( \vert \widehat { V } _ { n } ( s _ { 0 } ) - \widetilde { V } ( s _ { 0 } ) \vert \ \geq \ s \Big ) \ \leq \ c _ { 0 } n ^ { - \alpha _ { 0 } } s ^ { - \beta _ { 0 } } ,
$$

for $s > n ^ { - \alpha _ { 0 } / \beta _ { 0 } }$ . Hence,

$$
\begin{array} { r } { \Big | \mathbb { E } \big [ \widehat { V } _ { n } ( s _ { 0 } ) \big ] - \widetilde { V } ( s _ { 0 } ) \Big | \ \le \ \int _ { 0 } ^ { n ^ { - \frac { \alpha _ { 0 } } { \beta _ { 0 } } } } 1 d s + \int _ { n ^ { - \frac { \alpha _ { 0 } } { \beta _ { 0 } } } } ^ { \infty } c _ { 0 } n ^ { - \alpha _ { 0 } } s ^ { - \beta _ { 0 } } d s } \\ { \le n ^ { - \frac { \alpha _ { 0 } } { \beta _ { 0 } } } + c _ { 0 } n ^ { - \alpha _ { 0 } } \int _ { n ^ { - \frac { \alpha _ { 0 } } { \beta _ { 0 } } } } ^ { \infty } s ^ { - \beta _ { 0 } } d s } \\ { = n ^ { - \frac { \alpha _ { 0 } } { \beta _ { 0 } } } + \frac { c _ { 0 } } { \beta _ { 0 } - 1 } n ^ { - \alpha _ { 0 } } \big [ s ^ { - \beta _ { 0 } + 1 } \big ] _ { s = n ^ { - \frac { \alpha _ { 0 } } { \beta _ { 0 } } } } ^ { \infty } . } \end{array}
$$

Because $\begin{array} { r } { \frac { \alpha _ { 0 } } { \beta _ { 0 } } \leq \frac { 1 } { 2 } } \end{array}$ (see Theorem 1), the dominant term is $\mathcal { O } ( n ^ { - \frac { 1 } { 2 } } )$ . Thus,

$$
\Big | \mathbb { E } \big [ \widehat { V } _ { n } ( s _ { 0 } ) \big ] - \widetilde { V } ( s _ { 0 } ) \Big | \le \mathcal { O } \big ( n ^ { - \frac { 1 } { 2 } } \big ) .
$$

## E. Experimental setup and Parameters selection

## E.1. Experimental setup

All experiments are done over 100 seeds, using $\gamma = 0 . 9 9$ and robustness budget $\rho = 0 . 5$ , with these values showing consistent performance across preliminary experiments with different parameter settings. We use 2000 rollouts for The Gambler’s Problem and 4000 rollouts for Frozen Lake.

We implement our robust MCTS framework by extending a base Monte Carlo Tree Search implementation from (Leurent, 2018). Our codebase adds Stochastic Power UCT and introduces new robust backup operators for handling different uncertainty sets (Total Variation, Chi-squared, and Wasserstein), while maintaining the core MCTS selection and expansion strategies. We also provide our code at https://github.com/brahimdriss/RobustMCTS.

## E.2. Environments

The Gambler’s Problem (Sutton and Barto, 2018): a classic casino-inspired reinforcement learning environment where an agent starts with an initial capital and aims to reach a specific goal amount through a series of betting decisions. In our implementation, the agent begins with 50 units of capital and must reach a goal of 100 units to win. At each step, the agent can bet any amount up to its current capital. The environment has a win probability $p _ { h }$ for each bet, where the agent either wins the wagered amount with probability $p _ { h }$ or loses it with probability $1 - p _ { h }$ . The state space consists of all possible integer capital amounts from 0 to 100, with 0 and 100 being terminal states. The action space at each state includes all possible integer bets up to the current capital. This environment is particularly suitable for studying decision-making under uncertainty as it combines both risk management and optimal stopping aspects.

In our experiments, to reduce computational complexity while maintaining the same fundamental dynamics and challenges, we scaled down the problem to use a starting capital of 5 units and a goal of 10 units. This smaller scale version preserves all the essential characteristics and decision-making complexity of the original problem.

Frozen Lake(Towers et al., 2024): This environment presents a gridworld navigation challenge where an agent must traverse a 4x4 frozen surface from a starting position to a goal while avoiding holes. The surface is slippery, introducing stochastic dynamics where the agent’s intended actions may result in sliding to adjacent states with some probability. The state space consists of 16 discrete states representing different positions on the grid, with some states marked as holes (H) and one goal state (G). The action space includes four possible movements: left, right, up, and down. When the agent executes an action, it moves in the intended direction with probability 1/3 and slides perpendicular to the intended direction (left or right) with probability 2/3, making the environment highly stochastic. This environment is particularly valuable for evaluating robust policies as it combines both navigational planning and uncertainty in action outcomes.

In our experiments, we define $p _ { \mathrm { s l i p } }$ as the probability that the executed action differs from the agent’s selected action. When a slip occurs, the actual executed action is sampled uniformly at random, effectively modeling the uncertain dynamics of the frozen surface.

## E.3. Robust Performance Results

We investigate the impact of uncertainty budgets on agent performance in a modified gambler’s problem. In this experiment, we fix the planning probability $p _ { h }$ at 0.6 , the ambiguity set at Wasserstein. The agent’s robustness is evaluated across different uncertainty budgets $\rho \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , where higher values of $\rho$ correspond to more conservative policies. For each uncertainty budget, we assess the agent’s performance by varying the execution probability from 0.2 to 0.8, thus testing the policy’s robustness to model misspecification. This experimental design allows us to analyze how different levels of conservatism (controlled by the uncertainty budget) affect the agent’s ability to maintain performance when faced with discrepancies between planning and execution environments.

Figure 3 demonstrate a clear trade-off between performance and robustness across different uncertainty budgets. Agents with lower uncertainty budgets $( \rho = 0 . 1 , 0 . 3 )$ achieve better performance when the execution probability matches or exceeds the planning probability, but their success rate drops significantly in misspecified environments. In contrast, higher uncertainty budgets $( \rho = 0 . 7 , 0 . 9 )$ show more consistent performance across different execution probabilities, particularly maintaining better success rates when the execution probability is lower than the planning probability. This suggests that while conservative policies might not achieve optimal performance in well-specified environments, they provide better robustness to model misspecification. The moderate uncertainty budget $( \rho = 0 . 5 )$ appears to offer a balanced trade-off, maintaining reasonable performance across both regimes.

We now investigate a wide range of transition model ambiguities for the Frozen Lake environment. Table 4 provides an extended version of Table 2 with detailed success rates across different planning and execution probabilities. We observe that the performance of Stochastic-Power-UCT algorithm degrades faster for increased noise injection for slipping probabilities $p _ { \mathrm { s l i p } }$ . We again see Wasserstein robust MCTS does well across all planning versus execution phases. All robust MCTS variants outperform the baseline.

Finally, our experiments reveal that the Wasserstein robust MCTS algorithm showcases the most robust performance across all variants. It might be of independent interest for future research to give a theoretical understanding of this phenomenon.

![](images/963418a02f659e160c8245426b19dcc68e4639b4fa568c854b9d84742f4fae4e.jpg)  
Figure 3: Performance comparison across different uncertainty budgets $( \rho ) .$ . Planning probability is fixed at $p _ { h } = 0 . 6$ (vertical dashed line), while execution probability varies from 0.2 to 0.8. Higher uncertainty budgets lead to more conservative policies, showing improved robustness when $p _ { h } \leqslant 0 . 6$ but potentially reduced performance when $p _ { h } > 0 . 6 .$

<table><tr><td rowspan="2">Planning  $p _ { \mathrm { s l i p } } ^ { \mathrm { p l a n } }$ </td><td colspan="5">Execution</td></tr><tr><td></td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td></tr><tr><td rowspan="4">0.0</td><td> $\mathrm { S p }$ </td><td>100</td><td>85</td><td>71</td><td>60</td><td>34</td></tr><tr><td>Tv</td><td>100</td><td>84</td><td>71</td><td>51</td><td>39</td></tr><tr><td>Cs</td><td>100</td><td>87</td><td>62</td><td>53</td><td>33</td></tr><tr><td>Ws</td><td>100</td><td>86</td><td>72</td><td>58</td><td>45</td></tr><tr><td rowspan="4">0.1</td><td> $\mathrm { S p }$ </td><td>65</td><td>52</td><td>41</td><td>32</td><td>21</td></tr><tr><td>Tv</td><td>68</td><td>54</td><td>42</td><td>33</td><td>24</td></tr><tr><td>Cs</td><td>95</td><td>82</td><td>65</td><td>52</td><td>35</td></tr><tr><td> $\mathrm { W s }$ </td><td>97</td><td>84</td><td>68</td><td>55</td><td>38</td></tr><tr><td rowspan="4">0.2</td><td> $\mathrm { S p }$ </td><td>35</td><td>28</td><td>22</td><td>15</td><td>12</td></tr><tr><td>Tv</td><td>38</td><td>30</td><td>25</td><td>18</td><td>15</td></tr><tr><td>Cs</td><td>75</td><td>65</td><td>48</td><td>35</td><td>25</td></tr><tr><td>Ws</td><td>78</td><td>68</td><td>45</td><td>38</td><td>28</td></tr><tr><td rowspan="4">0.3</td><td>Sp</td><td>15</td><td>12</td><td>10</td><td>8</td><td>7</td></tr><tr><td>Tv</td><td>18</td><td>15</td><td>12</td><td>10</td><td>8</td></tr><tr><td>Cs</td><td>55</td><td>45</td><td>35</td><td>25</td><td>18</td></tr><tr><td>Ws</td><td>58</td><td>48</td><td>32</td><td>28</td><td>20</td></tr><tr><td rowspan="4">0.4</td><td>Sp</td><td>8</td><td>7</td><td>6</td><td>5</td><td>4</td></tr><tr><td>Tv</td><td>10</td><td>8</td><td>7</td><td>6</td><td>5</td></tr><tr><td>Cs</td><td>35</td><td>28</td><td>22</td><td>18</td><td>12</td></tr><tr><td> $\mathrm { W s }$ </td><td>38</td><td>30</td><td>25</td><td>20</td><td>15</td></tr><tr><td rowspan="4">0.5</td><td>Sp</td><td>5</td><td>4</td><td>4</td><td>3</td><td>3</td></tr><tr><td>Tv</td><td>6</td><td>5</td><td>4</td><td>4</td><td>3</td></tr><tr><td>Cs</td><td>25</td><td>20</td><td>15</td><td>12</td><td>8</td></tr><tr><td> $\mathrm { W s }$ </td><td>28</td><td>22</td><td>18</td><td>15</td><td>10</td></tr></table>

Table 4: Success rates (%) for planning with Power-UCT variants. Methods: Stochastic-Power-UCT $( \mathsf { S p } )$ , Robust version with Total Variation (Tv), Chi-squared (Cs), and Wasserstein (Ws) ambiguity sets. Underlined values indicate matching planning and execution $p _ { \mathrm { s l i p } }$ . Bold indicates highest success rate per planning scenario.