# BELIEF-BASED MAXIMUM OCCUPANCY PRINCIPLE AND ACTIVE INFERENCE

Manolis Mylonas Center for Brain and Cognition & Department of Engineering Universitat Pompeu Fabra Barcelona, Spain manolis.mylonas@upf.edu

Rubén Moreno-Bote Center for Brain and Cognition & Department of Engineering Universitat Pompeu Fabra Barcelona, Spain ruben.moreno@upf.edu

## ABSTRACT

Intrinsic motivation plays a central role in adaptive and goal-directed behavior by conferring agents reward-independent objectives and biases useful to act in noisy and uncertain environments. Active Inference addresses the problem of acting in a partially observable environment through a principled framework for belief updating and action selection. A key component of Active Inference is the specification of prior preferences, which shapes behavior by encoding desirable future outcomes. An intrinsic motivation approach called the Maximum Occupancy Principle (MOP) proposes that agents act so as to maximize occupancy over future paths of states and actions, with no preferences or epistemic targets. Despite its simple formulation, MOP gives rise to rich and adaptive behaviors that combine exploratory variability with goal-directed dynamics. In this work, we extend MOP to partially observable environments and introduce a Bellman reformulation of the Expected Free Energy for Active Inference, both incorporating belief-based inference over hidden states as part of the agent state. The Bellman formulation enables tractable offline computation via value iteration over the full belief-state space. We compare the resulting behaviors in a set of minimal experimental settings with uncertain food sources. We find that MOP agents switch between goal-directed (food seeking) behavior and exploration between different food sources, depending on their energy available and their belief state. In contrast, Active Inference agents mostly inhabit regions around a single food source, a strategy having both high pragmatic and epistemic value. We finally compare with Empowerment, which is shown to be qualitatively similar to Active Inference.

Keywords intrinsic motivation · maximum occupancy principle · active inference · empowerment

## 1 Introduction

While extrinsic reward maximization –the idea that there is a fixed reward function that must be maximized by the agent in the long run– has dominated reinforcement learning [1], many behaviors observed in biological agents cannot be fully explained through reward optimization alone. Curiosity-driven exploration, spontaneous action selection, and behavioral variability are often attributed to intrinsic motivational mechanisms that operate independently of extrinsic rewards [2]. Active Inference [3, 4, 5] is a prominent theory of decision making not purely based on extrinsic rewards. In this framework, the agent aims to reach states that provide information about the environment, but also agree with some prior preferences that ensure homeostasis, by minimizing its free energy. Another popular theory is Empowerment [6, 7], which offers an alternative information-theoretic perspective, quantifying the influence an agent has on its environment by maximizing the mutual information between actions and future states. This yields behavior that favors controllability and observability of the environment by the agent. A more recent intrinsic motivation theory, the Maximum Occupancy Principle (MOP) [8, 9], departs from reward-based formulations by proposing that agents should act to maximize the entropy of future action-state paths: actions and states are preferred when they support a diverse set of possible future trajectories, thereby maintaining behavioral flexibility and adaptability. The resulting control problem can be expressed through a value function that quantifies the expected diversity of future paths under a given policy. While simple in formulation, this approach can generate complex behaviors, combining both variability and goal-directedness without relying on explicit rewards; it also produces life-preserving behaviors to avoid falling in terminal states from where no more path entropy can be generated [9].

Partially observable Markov decision processes (POMDPs) provide the standard formalism for decision-making under uncertainty [10, 11, 12, 13, 14]. In this setting, agents do not directly observe the full system state but instead receive noisy observations, requiring the construction of a belief over hidden variables. This belief evolves in time as a stochastic process and acts as a sufficient statistic for optimal control [15]. MOP was originally formulated for fully observable environments. However, most realistic settings involve partial observability. In this work, we extend MOP to POMDPs by introducing belief-based inference over hidden variables and deriving the corresponding value function (functional) and optimal policy. We demonstrate the resulting behavior in a two-food-source partially observable grid-world environment and compare it with Active Inference and Empowerment. Our results show that MOP induces adaptive exploration strategies between and around the food sources that are strongly modulated by internal energetic constraints. We also introduce two extensions of the Sophisticated Inference framework: we reformulate the Expected Free Energy as a Bellman equation with a temporal discount factor, replacing the original tree-search procedure with tractable offline value iteration over the full belief-state space, and we derive time-stationary policies that do not require replanning at each decision step. Both Active Inference and Empowerment agents tend to inhabit a single food source, where higher epistemic value can be gained and also larger controllability can be exerted. While the three frameworks combine exploratory and homeostatic pressures in their induced behavior, they do so in fundamentally different ways. This comparison is part of a broader effort to build a theory of behavior: we compare the three frameworks in terms of the richness and spatial extent of the behavior they induce, and in terms of whether goal-directed action emerges without being explicitly encoded.

## 2 Partial Observability Formulation

Although our framework is general and applies to any partially observable Markov decision process (POMDP), here we specialize the notation and show results for the particular case of a two-food-source partially observable grid-world (Fig. 1). Extending our equations and notation to any general POMDP is straightforward, and the exact general update equations for MOP, EFE and Empowerment are given in Eqs. 6, 10 and 15, respectively. Our specific environment includes the following key ingredients, enabling a systematic comparison between different intrinsic motivation approaches: (1) multiple sources of evidence, allowing us to assess whether different methods preferentially focus on a single source or distribute attention across both; (2) an internal energy variable, which permits evaluation of how conservatively each approach behaves with respect to avoiding low-energy (terminal) states; and (3) a spatial separation between food sources, which makes it possible to measure the extent of exploratory excursions and the ability of each method to sustain long-range exploration.

## 2.1 A two-food-source partially observable grid-world environment

We consider a POMDP in discrete time defined by the tuple $M = \{ S , H , E _ { \mathrm { g a i n } } , E _ { \mathrm { m a x } } , A , \Omega , p , q \}$ . The fully observable state space is $S = \mathcal { X } _ { 1 } \times \mathcal { X } _ { 2 } \times \mathcal { E }$ , where $x = ( x _ { 1 } , x _ { 2 } ) \in \mathcal { X } _ { 1 } \times \mathcal { X } _ { 2 }$ denotes the agent’s position in a $5 \times 5$ grid world and $E \in { \mathcal { E } }$ its energy level. The environment contains two food sources $i \in \{ \bar { 0 } , 1 \}$ , located at positions $x _ { f , i = 1 } = ( 1 , 1 )$ and $x _ { f , i = 2 } = ( 5 , 5 )$ . Their availability is represented by the hidden state $\bar { H } = \{ h _ { 1 } , h _ { 2 } \}$ , where $\bar { h _ { i } } \in \{ 0 , 1 \}$ } is a binary variable indicating whether food source $x _ { f , i }$ contains food $( h _ { i } = 1 )$ or not $( h _ { i } = 0 )$ . We denote the observable state by

$$
s = ( x , E ) = ( x _ { 1 } , x _ { 2 } , E )
$$

and the hidden state by

$$
h = \left( h _ { 1 } , h _ { 2 } \right) .
$$

![](images/da348ad11116b32b75d9aab61804bfe27a5776218481d1975029822814087c58.jpg)  
Figure 1: Two-Food-Source Partially Observable Grid-World Environment

The action space is $A = \{ \mathrm { u p } $ , down, left, right, stay}. A table with all the definitions of the terms used throughout and the implementation values can be found in Appendix A.1 and A.2 respectively.

## 2.2 Environment Dynamics

The probability transition matrix of the world (observable and hidden) state $( s , h )$ to its successor $( s ^ { \prime } , h ^ { \prime } )$ , given that action a is performed, is

$$
p ( s ^ { \prime } , h ^ { \prime } | s , h , a ) = p ( E ^ { \prime } | E , x ^ { \prime } , h ^ { \prime } ) p ( x ^ { \prime } | s , a ) p ( h ^ { \prime } | h , x ) .\tag{1}
$$

We assume that the transition matrix factorizes into

$$
p ( s ^ { \prime } , h ^ { \prime } | s , h , a ) = \underbrace { \delta \bigl ( E ^ { \prime } - E ^ { \prime } ( E , x ^ { \prime } , h ^ { \prime } ) \bigr ) } _ { \mathrm { e n e r g y t r a n s i t i o n } } \underbrace { \delta \bigl ( x ^ { \prime } - x ^ { \prime } ( s , a ) \bigr ) } _ { \mathrm { s p a t i a l ~ t r a n s i t i o n } } \underbrace { p ( h _ { 1 } ^ { \prime } | h _ { 1 } , x ) p ( h _ { 2 } ^ { \prime } | h _ { 2 } , x ) } _ { \mathrm { f o o d t r a n s i t i o n } } ,
$$

which models a scenario where food sources evolve independently of each other, and whose state might depend on the visitation of the agent to the food source location, but not on the agent’s energy. Here, δ is the Kronecker delta, and we write $x ^ { \prime } ( s , a )$ and $E ^ { \prime } ( E , x ^ { \prime } , h ^ { \prime } )$ for the resulting successor position and energy, $s ^ { \prime } = ( x ^ { \prime } , E ^ { \prime } )$ . The spatial transition is deterministic given the current state and action. The agent moves one cell in the chosen direction, subject to the boundary constraints of the grid-world and the availability of energy,

$$
x ^ { \prime } ( s , a ) \equiv x ^ { \prime } ( x , E , a ) = { \left\{ \begin{array} { l l } { x + \Delta ( a ) } & { { \mathrm { i f ~ } } x + \Delta ( a ) { \mathrm { ~ l i e s ~ w i t h i n ~ t h e ~ g r i d ~ a n d ~ } } E > 0 , } \\ { x } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }
$$

where $\Delta ( a )$ is the displacement associated with action a. In particular, if the agent reaches zero energy, $E = 0 , { \mathrm { i t } } " { \mathrm { d i e s } } "$ and cannot move, i.e., the only available action is to $" s t a y "$ with probability one. Thus, the resulting state has zero action entropy. The energy transition depends not only on the current energy but also on the new position $x ^ { \prime }$ and the new food state $h ^ { \prime }$ deterministically as

$$
E ^ { \prime } ( E , x ^ { \prime } , h ^ { \prime } ) = \left\{ \begin{array} { l l } { \operatorname* { m i n } \big ( E - 1 + E _ { \mathrm { g a i n } } , ~ E _ { \mathrm { m a x } } \big ) } & { \mathrm { i f ~ } x ^ { \prime } = x _ { f , i } \mathrm { ~ a n d ~ } h _ { i } ^ { \prime } = 1 , } \\ { E - 1 } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

since the agent can only gain energy if it lands on a food source where food is present. This makes explicit that the spatial dynamics are autonomous (a function of s and a alone), whereas the energy dynamics are coupled to the hidden food state through $h ^ { \prime }$

For each food source, we consider two possible dynamics:

Location-independent dynamics. Here, $p ( h _ { i } ^ { \prime } | h _ { i } , x ) = p ( h _ { i } ^ { \prime } | h _ { i } )$ , independent of the agent’s location, with

$$
p ( h _ { i } ^ { \prime } = 0 | h _ { i } = 1 ) = 1 - \lambda ,
$$

$$
p ( h _ { i } ^ { \prime } = 0 | h _ { i } = 0 ) = 1 - \mu .
$$

Depletion at visit. Here, $p ( h _ { i } ^ { \prime } | h _ { i } , x )$ explicitly depends on the agent’s location,

$$
\begin{array} { c } { p ( h _ { i } ^ { \prime } = 0 | h _ { i } = 1 , x \neq x _ { f , i } ) = 1 - \lambda , \qquad p ( h _ { i } ^ { \prime } = 0 | h _ { i } = 1 , x = x _ { f , i } ) = 1 - \rho , } \\ { p ( h _ { i } ^ { \prime } = 0 | h _ { i } = 0 , x ) = 1 - \mu , \forall x , } \end{array}
$$

where $\mu , \lambda , \rho \in [ 0 , 1 ]$ are parameters that determine the stochasticity of the food-state transitions.

## 2.3 Observation Model

The agent receives noisy binary observations $\omega = ( \omega _ { 1 } , \omega _ { 2 } )$ related to the presence or absence of food at each food source location. The observation model is

$$
q ( \omega | s , h ) = q ( \omega _ { 1 } | x , h _ { 1 } ) q ( \omega _ { 2 } | x , h _ { 2 } ) \ ,\tag{2}
$$

assuming conditional independence of the observations given the agent’s location x and hidden state $h ,$ and no dependence on the agent’s current energy. We assume that the observation is fully informative at the food location, and totally uninformative away from it, that is

$$
\begin{array} { r } { q ( \omega _ { i } = h _ { i } | x = x _ { f , i } , h _ { i } ) = 1 , \qquad q ( \omega _ { i } = \{ 1 , 0 \} | x \neq x _ { f , i } , h _ { i } ) = \frac { 1 } { 2 } . } \end{array}
$$

This means $\mathrm { e . g . }$ that observation $\omega _ { i } = 1$ at the food source implies with certainty that the food source i is full, $h _ { i } = 1$ In contrast, outside the food source, observations are noisy and non-informative. This choice is made to keep a simple representation of the environment while preserving the structure of partial observability.

## 2.4 Belief-State Representation

We define the belief of the agent about the hidden variable $h ,$ and denote it as $b ( h )$ , as the probability distribution over h inferred from all past observations, $b ( h ) \equiv P ( h$ |past observations) (we often denote this probability distribution simply by b). Starting from belief $b ,$ an agent at observable state $\boldsymbol { s } = \left( \boldsymbol { x } , E \right)$ performs an action a, experiences a transition to new location ${ \bf { \bar { \mathbf { \Phi } } } } _ { x ^ { \prime } }$ and observes $\omega ^ { \prime }$ . After this observation, the agent can update the belief about the hidden variables using Bayes rule as

$$
b ^ { \prime } ( h ^ { \prime } ) \equiv P ( h ^ { \prime } | \omega ^ { \prime } , s , b , a ) \propto \sum _ { x ^ { \prime } , h } q ( \omega ^ { \prime } | x ^ { \prime } , h ^ { \prime } ) p ( h ^ { \prime } | h , x ) \delta ( x ^ { \prime } - x ^ { \prime } ( s , a ) ) b ( h ) .\tag{3}
$$

We will use further below the notation $b ^ { \prime } ( \omega ^ { \prime } , s , b , a )$ to indicate this updated belief (probability distribution) and make explicit all its dependencies. Because both the observation and the hidden state dynamics are conditionally independent given the world state $( s , h )$ (see Eqs. 1,2), if the initial belief is factorized, then the updated belief is also factorized (note that initially the agent can start with a belief that is factorized to represent the real world state of food source independence, which is what we assume here). Thus, the belief factors in $\dot { b ^ { \prime } } ( h ^ { \prime } ) = b ^ { \prime } ( h _ { 1 } ^ { \prime } ) b ^ { \prime } ( h _ { 2 } ^ { \prime } )$ can be computed as

$$
b ^ { \prime } ( h _ { i } ^ { \prime } ) \equiv P ( h _ { i } ^ { \prime } | \omega _ { i } ^ { \prime } , s , b , a ) \propto \sum _ { h _ { i } } q ( \omega _ { i } ^ { \prime } | x ^ { \prime } ( s , a ) , h _ { i } ^ { \prime } ) p ( h _ { i } ^ { \prime } | h _ { i } , x ) b ( h _ { i } ) .
$$

Note that belief factorization holds in both the location-independent and the depletion-at-visit environments.

In general, the belief b is a continuous quantity, as it represents a probability distribution over the hidden state $h \in \{ 0 , 1 \}$ and can take any value in $[ 0 , 1 ] \times [ 0 , 1 ]$ . For computational tractability, however, we discretize the belief state, so that the full agent state $( s , b )$ lies on a finite grid and standard dynamic programming methods can be applied (see Appendix A.3 and A.4 for details).

## 2.5 Predictive Model

The agent uses a predictive model during planning by first replacing the unknown hidden states by beliefs. The sufficient statistics for the problem is the agent state $( s , b )$ , which summarizes all information available for the agent to compute an optimal policy and select actions. Optimal policies depend on the optimization objective, which will be described in detail in Secs. 3.1-3.2-3.3 for MOP, EFE and Empowerment, respectively.

The probability distribution of the next agent’s state $( s ^ { \prime } , b ^ { \prime } )$ given $( s , b )$ and the performed action a is computed as

$$
P ( s ^ { \prime } , b ^ { \prime } | s , b , a ) = \sum _ { \omega ^ { \prime } } \delta ( b ^ { \prime } - b ^ { \prime } ( \omega ^ { \prime } , s , a ) ) P ( s ^ { \prime } , \omega ^ { \prime } | s , b , a ) ,\tag{4}
$$

with $\begin{array} { r } { P ( s ^ { \prime } , \omega ^ { \prime } | s , b , a ) = \sum _ { h . h ^ { \prime } } q ( \omega ^ { \prime } | x ^ { \prime } ( s , a ) , h ^ { \prime } ) p ( s ^ { \prime } , h ^ { \prime } | s , h , a ) b ( h ) } \end{array}$ , where we have used the updated belief in Eq. 3, the observation model in Eq. 2, and the world (observable and hidden) state transition dynamics in Eq. 1. Note that the belief evolves deterministically, a fact that is represented by the Kronecker delta function (it is one if the probability distribution $b ^ { \prime }$ matches $b ^ { \prime } ( \omega ^ { \prime } , \dot { x } , a )$ , and zero otherwise, noting again that probabilities are discretized). Eq. 4 is used in the definitions of MOP and Active Inference in Eqs. 6 and 11.

The full predictive model is the joint probability distribution over observations, future observable states and hidden states, given the current agent’s state $( s , b )$ and performed action a. This joint, used in Eq. 11, is expressed as

$$
P ( \omega ^ { \prime } , s ^ { \prime } , h ^ { \prime } | s , b , a ) = \sum _ { h } q ( \omega ^ { \prime } | x ^ { \prime } , h ^ { \prime } ) p ( s ^ { \prime } , h ^ { \prime } | s , h , a ) b ( h ) .\tag{5}
$$

## 3 Intrinsic motivation models

## 3.1 Maximum Occupancy Principle

In POMDPs, the policy can only depend on the observable state s and the current belief of the agent b about the hidden states, and not on the hidden states h themselves. Indeed, and as we said before, the agent state $( s , b )$ corresponds to the sufficient statistics in our problem. We define the policy $\pi ( a | s , b )$ as the probability of selecting action a given the observable state s and belief b. Starting at $t = 0$ in state $( s _ { 0 } , b _ { 0 } )$ , an agent performing a sequence of actions and experiencing state transitions $\tau = ( s _ { 0 } , b _ { 0 } , a _ { 0 } , s _ { 1 } , b _ { 1 } , . . . , a _ { t } , s _ { t + 1 } , b _ { t + 1 } , . . . )$ gets the return (intrinsic reward)

$$
R ( \tau ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R ( s _ { t } , b _ { t } , a _ { t } ) = - \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \ln \left( \pi ( a _ { t } | s _ { t } , b _ { t } ) P ^ { \beta } ( s _ { t + 1 } , b _ { t + 1 } | s _ { t } , b _ { t } , a _ { t } ) \right) ,
$$

with $\beta$ being a fixed real number (chosen for simplicity to be $\beta = 0$ in our simulations), and discount factor $0 < \gamma < 1$ The corresponding value function (functional) in the POMDP setting is given by

$$
V _ { \pi } ( s , b ) = \mathbb { E } _ { \pi , P } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \left( H ( a | s _ { t } , b _ { t } ) + \beta H ( s ^ { \prime } , b ^ { \prime } | s _ { t } , b _ { t } , a _ { t } ) \right) \Bigg | s _ { 0 } = s , b _ { 0 } = b \right] \mathrm { ~ , ~ }
$$

where $H ( a | s _ { t } , b _ { t } )$ is the policy entropy, and $H ( \cdot , \cdot | s _ { t } , b _ { t } , a _ { t } )$ is the entropy of the agent state transition kernel $P ( s ^ { \prime } , b ^ { \prime } | s , \dot { b } , a )$ defined in Eq. 4. The optimal policy is such that the value function is optimized in every agent’s state. This optimal value function $\dot { V } ^ { * } ( s , b )$ satisfies the self-consistency equation

$$
V ^ { * } ( s , b ) = \ln \sum _ { a } \exp \left[ \beta H ( \cdot , \cdot | s , b , a ) + \gamma \sum _ { s ^ { \prime } , b ^ { \prime } } P ( s ^ { \prime } , b ^ { \prime } | s , b , a ) V ^ { * } ( s ^ { \prime } , b ^ { \prime } ) \right] .\tag{6}
$$

The sum over $b ^ { \prime }$ is taken to be a finite sum rather than an integral, because we discretize belief state onto a finite grid $( \mathrm { S e c } . 2 . 4 )$ . This equation can be solved by iterative recursion [8], where we have to impose the boundary conditions $V ( s ^ { + } , b ) = 0$ for every observable state ${ \bar { s } } ^ { + }$ with $E = 0$ regardless of b so that terminal states have zero value due to the termination of the episode and the impossibility of generating any further action-state path entropy. Implementation details can be found in Appendix $\mathrm { A } . 7$

The optimal action-value function is defined as

$$
Q ^ { * } ( s , b , a ) = \gamma \sum _ { s ^ { \prime } , b ^ { \prime } } P ( s ^ { \prime } , b ^ { \prime } | s , b , a ) V ^ { * } ( s ^ { \prime } , b ^ { \prime } ) \ : ,\tag{7}
$$

from where the optimal policy is computed as

$$
\pi ^ { * } ( a | s , b ) = \sigma \bigl ( Q ^ { * } ( s , b , a ) \bigr ) = \frac { \exp \left[ Q ^ { * } ( s , b , a ) \right] } { \sum _ { a ^ { \prime } } \exp \left[ Q ^ { * } ( s , b , a ^ { \prime } ) \right] } ,\tag{8}
$$

where $\sigma$ denotes the softmax operation.

## 3.2 Sophisticated Inference

The implementation of Active Inference builds on Sophisticated Inference [5], with the novelty of including exact belief propagation and temporal discounting, which enables time-stationary solutions. In the Active Inference framework, the prior preferences over the location-energy state s are embodied in a target distribution $P ( s )$ , which is taken here to be uniform across states except for $E = 0$ , which is assigned a low probability (low preference). With this choice, we aim at introducing the least structure into the agent’s preferences.

## 3.2.1 Variational Free Energy

The agent forms beliefs by minimizing its Variational Free Energy (VFE). The VFE over hidden food states $h$ is defined as

$$
F [ Q ] = D _ { \mathrm { K L } } \big ( Q ( h ^ { \prime } ) \| p ( h ^ { \prime } ) \big ) - \mathbb { E } _ { Q ( h ^ { \prime } ) } \big [ \ln q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } ) \big ] ,\tag{9}
$$

where $Q ( h ^ { \prime } )$ is the variational distribution over the updated hidden food state $h ^ { \prime }$ obtained after taking action a and transitioning from state s to $\begin{array} { r } { s ^ { \prime } , p ( h ^ { \prime } ) = \sum _ { h } p ( h ^ { \prime } | h , \bar { x } ) \ Q ( h ) } \end{array}$ is the predictive prior over $h ^ { \prime }$ obtained by propagating the current variational belief $Q ( h )$ through the transition dynamics, and $q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } )$ is the observation model defined in Eq. 2. The first term penalizes deviations from the prior (complexity), while the second term encodes the fit to observations (accuracy). Minimizing the VFE in the grid-world scenario is mathematically equivalent to the belief propagation equations described in Sec. 2.4 (see Appendix A.5 for details) and the belief $b ( h )$ used throughout this work corresponds to the minimizer b(h) = arg min<sub>Q</sub> $F [ Q ]$ . In the following section, the beliefs used in the computation of the Expected Free Energy are obtained through the same belief update mechanism employed in MOP (Eq. 3).

## 3.2.2 Expected Free Energy

In the Sophisticated Inference framework, the agent selects actions so as to minimize its Expected Free Energy (EFE), which balances pragmatic drives (risk) with epistemic drives (ambiguity resolution). The EFE is written as

$$
{ \cal G } ( s , b ) = \sum _ { a } \pi ( a | s , b ) { \cal G } ( s , b , a ) ,\tag{10}
$$

where $G ( s , b , a )$ is the per-action EFE, defined as

$$
\begin{array} { r l } & { G ( s , b , a ) = \underbrace { \mathbb { E } _ { P ( \omega ^ { \prime } , s ^ { \prime } , h ^ { \prime } \mid s , b , a ) } \left[ \widehat { \ln { P ( s ^ { \prime } \mid s , b , a ) } - \ln { P ( s ^ { \prime } ) } } \widehat { - \ln { q ( \omega ^ { \prime } \mid s ^ { \prime } , h ^ { \prime } ) } } \right] } _ { \mathrm { E x p e c t e d f r e e ~ e n c r g y ~ o f ~ n e x t a c t i o n } } } \\ & { \quad \quad \quad + \gamma \underbrace { \mathbb { E } _ { P ( s ^ { \prime } , b ^ { \prime } \mid s , b , a ) } \left[ G ( s ^ { \prime } , b ^ { \prime } ) \right] } _ { \mathrm { E x p e c t e d f r e e ~ e n e r g y ~ o f ~ s u b s e q u e n t ~ a c t i o n s } } . } \end{array}\tag{11}
$$

The policy $\pi ( a | s , b )$ is computed as

$$
\pi ( a | s , b ) = \sigma { \big ( } - d G ( s , b , a ) { \big ) } ~ ,\tag{12}
$$

where d is the inverse temperature parameter. The negative sign ensures that actions with lower EFE are assigned higher probability, consistent with the agent’s objective of minimizing $G ( s , b , a )$ . Since beliefs b are discretized onto a finite grid (Sec. 2.4), expectations become finite sums, making $\bar { G } ( s , \bar { b } )$ tractable to compute via value iteration over the full state-belief space (s, b). The choice of terminal boundary condition $G ^ { + }$ is important, since it governs the trade-off between exploration and conservatism in an active inference agent; this is discussed in detail in Appendix A.10.

We can also recover the optimal policy by taking the limit $d \to \infty$ in Eq. 12, which reduces the softmax to a hard minimization over actions (and thus a deterministic policy, except for ties), replacing Eq. 10 with

$$
G ( s , b ) = \operatorname* { m i n } _ { a } G ( s , b , a ) .\tag{13}
$$

In the deterministic (optimal) EFE (Eq. 13), the probability mass concentrates entirely on the action that minimizes $G ( s , b , a )$ . In the stochastic EFE (Eqs. 10-12), d is a tunable parameter controlling the degree of randomness in the policy. The effect of d on the agent’s behavior is analyzed in Results and Appendix A.9.

Eqs. 10-12 have the form of a Bellman equation and can be solved via value iteration [1]. In contrast to [5], where the EFE is computed by a deep tree search whose cost scales exponentially with depth, this allows the EFE to be computed offline over the full belief-state space with cost linear in depth, and the discount factor γ guarantees a unique fixed point and a time-stationary policy. An interpretation of the risk and ambiguity terms can be found in Appendix A.8.

## 3.3 Empowerment

Empowerment is an intrinsic objective that encourages the agent to select actions that maximize its influence over future observations. Specifically, it measures the mutual information between an n-step action sequence and the resulting future observation, quantifying how much control the agent has over what it will perceive. Implementation details can be found in Appendix A.6.

## 4 Results

We first evaluate and compare $\mathrm { M O P \left( E q . \Delta \ 6 \right) } .$ , Active Inference (Eqs. 10-12) and Empowerment (Eq. 15). For each method, we compute the corresponding objective, we derive the induced policy and simulate trajectories from identical initial conditions $( \mathrm { F i g } . 2 )$ . The resulting behaviors exhibit distinct qualitative structures:

· MOP. The induced policy explores the environment extensively, visiting both food sources and frequently switching between them. The resulting paths show sustained exploration without convergence to a single attractor.

Food depletion transition dynamics  
Independent food transition dynamics  
a  
![](images/c84ae33caea8a7b01fab40105ea2b805946edaaa0139b6d1237764128a8ac34a.jpg)

b  
![](images/c36460140d353fbe6722e4c4a8d732cb721388e8cbd3f5b992547ba6e9a5b6c6.jpg)  
c

![](images/ace12e46ea566f7bbb362ff483071426b9d4b065ec237017b87670989200bad6.jpg)

d  
![](images/2563181dd3498285624a2d86c56e90e11480de53d5e5043886329efd15bad3c8.jpg)  
e

![](images/9e7771e7c5508e32666b158eaca9b7a10ea29d1acadecc61155aa07c8c6ac515.jpg)

f  
![](images/91c952316e27d40b52ba833b0f5e53ba840cad6553664fcd4b699a15c3f9f082.jpg)  
Figure 2: Space habitation heatmaps (average over 100 episodes) and a sample trajectory for MOP, EFE and Empowerment (MPOW), under independent (top) and depletion (bottom) food dynamics. $\check { T }$ denotes average survival duration in steps.

· Active Inference. The policy converges to a single food source and remains in its vicinity. This behavior is consistent with the minimization of expected free energy, which jointly favors energy preservation and reliable observations, leading to a stable attractor state. The agent records high average survival steps, at the expense of exploration. Here the deterministic policy of EFE is used (using a high inverse temperature d in Eq. 12). A comparison between stochastic and deterministic EFE is provided in Appendix A.9

· Empowerment. The policy reaches a food source and performs limited local exploration. Due to the finite planning horizon $( n = 5 )$ , the agent does not systematically evaluate distant alternatives, resulting in localized rather than global exploration behavior. A higher planning horizon $( \mathrm { e } . \mathrm { g } . , n = 1 0 )$ that would allow the agent to see the other food state is significantly more computationally expensive than the other two methods, and thus is not considered. The policy of empowerment is by definition deterministic, as the one maximizing the mutual information.

We next analyze how MOP modulates behavioral variability as a function of the agent’s energy E. To this end, we compute the (normalized) entropy of the induced policy for different energy levels $\breve { E }$ and examine the corresponding state visitation patterns (Fig. 3a). The value $E = 1 5$ roughly separates two energy-dependent regimes in the policy structure. For high energy levels $( E > 1 5 )$ , the agent exhibits increased policy entropy, corresponding to a broader distribution over actions. In this regime, the agent can afford exploratory behavior, leading to more uniform state visitation and frequent switching between food sources. In contrast, for low energy levels $( E \leq 1 5 )$ , the policy becomes significantly more concentrated, similar to the deterministic EFE and Empowerment agents. MOP prioritizes reaching food sources to avoid starvation, resulting in reduced entropy and more directed trajectories toward high-reward states. Overall, these results indicate that MOP induces an adaptive control strategy in which internal energetic constraint modulate the level of stochasticity in the policy.

Active Inference exhibits the same qualitative pattern (Fig. 3b). The policy entropy is highest when the agent is well fed and decreases as its energy depletes, so that behavioral variability is again modulated by the internal energetic state. What differs is the level at which this modulation operates, and that level is set by d rather than by the agent: the larger the inverse temperature, the lower the entropy at every energy level. Because d is fixed externally rather than modulated by the agent’s own energetic state, it acts as a single control knob that trades exploration against survival $( { \mathrm { F i g s . ~ } } 3 \mathbf { c } { \mathrm { - e } } )$ For small d the agent is adventurous: it spreads its visits over most of the grid, reaching a state entropy close to $\mathbf { M O P } \mathbf { \vec { s } }$ (Fig. 3d), but it attains the shortest lifespans of the range (Fig. 3c). As d increases the agent becomes conservative: it settles on a single food source, the location that simultaneously minimizes risk and ambiguity in Eq. 11, and remains in its vicinity. State entropy falls from 2.7 to 0.7 nats between $d = 0 . 5$ and $d = 1 0$ , while survival rises monotonically with d. Survival is therefore bought by abandoning exploration, and the exchange rate is set by d alone. A high policy entropy does not by itself imply exploratory behavior, and it is here that the two frameworks separate. MOP has no parameter to tune and is adventurous by default; the comparison is sharpest at $d = 0 . 8$ , the precision at which the two agents have similar policy entropy (Figs. 3a-c) and comparable lifespans. They nevertheless do not occupy the environment in the same way. MOP travels between the two food sources 13.3 times per thousand steps against the Active Inference agent’s 3.8, and more often than Active Inference at any precision we tested; the same separation appears in where the time is spent, with the Active Inference agent standing most often on a food source or within one step of it, while MOP spreads its occupancy more evenly over the grid (Fig. 6 in Appendix A.9). Fig. 3e summarizes this as a single trade-off curve: MOP lies off the trace that Active Inference follows as d is swept, since no precision matches its occupancy of the grid, and the one that comes closest in lifespan does so with a visibly narrower one. MOP therefore achieves exploration and survival jointly, whereas in Active Inference the two are exchanged against one another.

a  
![](images/a5135811c71966545bd58b2f83e63613453825a2af4bd05e6f2de5729cbb8070.jpg)  
d

b  
![](images/6021fd7e93642ca4d3222a316c0fcb0ce70523a46ee552dc3576faeba80f35b2.jpg)

c  
![](images/6ef4b57bfffa56d913823b62bdd3981a07289730a5dd192133d1dd46fdb0ede1.jpg)

![](images/df5015cd4c80881114f288c0010a2ec8768c44e05fd4841fd4de2f75d82a88df.jpg)  
e

![](images/7ba7068d74ebc54e6c542b81d4560af722a972389363b89c04bd55b71c371b58.jpg)  
Figure 3: Normalized policy entropy, survival and exploration for MOP and Active Inference, in the location-independent dynamics; depletion-at-visit shows similar behavior. (a) MOP exhibits a transition when the agent’s internal energy is around $E = 1 5 .$ , switching between broad exploration at high energy and goal-directed behavior at low energy (insets are corresponding state visitation heatmaps). (b) Active Inference policy entropy for six values of the inverse temperature $d .$ Every curve increases with energy, but d modulates the behavior: for small d the policy entropy remains high and the policy stays stochastic, whereas for large d it falls toward zero at every energy level and the policy becomes nearly deterministic (see heatmaps in Appendix A.9). (c) Average survival duration of the Active Inference agent as a function of $d ,$ compared with MOP. (d) Entropy of the empirical state-visitation distribution over the grid, over the same range of $d ,$ compared with MOP. Panels (c) and (d) are approximately mirror images of each other: the values of d at which its occupancy of the environment collapses (e) The same trade-off as a single curve, survival duration against the entropy of the empirical state-visitation distribution over the grid, one point per value of $d ,$ with MOP as a single point lying off the trace, indicated best survival for matched fixed state entropy, or best state entropy for matched fixed survival.

## 5 Discussion

We introduced a belief-based extension of the Maximum Occupancy Principle (MOP) and Active Inference to partially observable environments. The resulting framework generalizes naturally to POMDP settings via belief updating over hidden environmental variables. Our work introduces a novel reformulation of the Expected Free Energy (EFE) within the Sophisticated Inference framework. We express the EFE as a Bellman equation with a discount factor, enabling offline computation via value iteration over the full belief-state space, replacing the tree-search procedure of the original framework with a tractable dynamic programming solution that yields time-stationary policies directly applicable to POMDP settings. In our experiments, the induced policy of MOP exhibits structured adaptive behavior driven by internal energetic constraints. In particular, the agent balances exploratory behavior under high-energy conditions with increasingly goal-directed behavior under energy depletion, while maintaining flexible switching between available resources. The Active Inference agent, by contrast, exhibits robust survival behavior under uncertainty, driven by the joint minimization of risk and ambiguity that naturally balances food-seeking with information gathering about hidden food states. From a broader perspective, we identify a conceptual connection between MOP and Active Inference. Belief propagation under MOP is closely related to Variational Free Energy minimization for hidden-state inference.

Although our toy example is intentionally simple, we expect the qualitative similarities between the two frameworks to persist in more complex and stochastic environments. This connection has been discussed by Kiefer [16], who casts both frameworks as forms of constrained entropy maximization and argues that, in their full formulation, they differ mainly in whether the constraint that keeps the agent alive is explicit, as in Active Inference, or implicit, as in MOP. The ambiguity term in the EFE formulation (Eq. 11) drives the agent to seek observations that reduce uncertainty about hidden states. In the current grid-world, uncertainty is concentrated around food sources, leading the agent to revisit them to refine its beliefs. In richer environments with uncertainty distributed across many latent variables, an active inference agent would be naturally encouraged to explore broadly to reduce uncertainty. Therefore, even though active inference does not explicitly maximize state-action entropy in the same manner as MOP, it nevertheless induces exploratory behavior through epistemic value. The risk term in EFE plays a complementary role by enforcing homeostatic behavior: policies leading to preferred outcomes are favored, while trajectories associated with undesirable outcomes are avoided. At first glance, one may argue that such a survival-oriented mechanism is absent from MOP, since homeostatic constraints are not explicitly encoded as prior preferences. However, an analogous mechanism emerges implicitly through the treatment of terminal states. In practice, terminal states can be implemented as absorbing states that restrict the agent to a single self-transition action, eliminating all future behavioral possibilities. Since these states collapse the future trajectory distribution and reduce future path entropy to zero, they become strongly disfavored under MOP. Consequently, the agent is incentivized to avoid entering terminal states, producing behavior that resembles survival-driven action selection. From this perspective, both MOP and Active Inference can be interpreted as combining two complementary pressures: one favoring exploration and the maintenance of future possibilities, and another favoring continued viability or survival. Exploring this relationship further may help develop more unified and expressive models of adaptive behavior.

The two frameworks, however, arrive at these pressures by different means. Compared to fixed-horizon or preferencedriven frameworks, MOP does not rely on engineered reward shaping or strongly specified priors, and no parameter tuning was performed to elicit the observed behaviors, hinting at the generalizability of these results. Moreover, Active Inference biases the agent toward confident (low-uncertainty) belief states through its prior preferences, whereas MOP does not encode such a bias, leading to a more neutrality-preserving belief evolution driven by path occupancy maximization. These results suggest that entropy-based control objectives provide a unifying perspective on intrinsic motivation in partially observable environments, bridging information-theoretic and free-energy based formulations.

## References

[1] Richard S Sutton, Andrew G Barto, and Andrew Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

[2] Rubén Moreno-Bote, Ralf Haefner, Jordi Galiano-Landeira, Tianming Yang, and Pedro Maldonado. How intrinsic motivation underlies embodied open-ended behavior. arXiv preprint arXiv:2601.10276, 2026.

[3] Ryan Smith, Karl J Friston, and Christopher J Whyte. A step-by-step tutorial on active inference and its application to empirical data. Journal ofmathematical psychology, 107:102632, 2022.

[4] Lancelot Da Costa, Noor Sajid, Thomas Parr, Karl Friston, and Ryan Smith. Reward maximization through discrete active inference. Neural Computation, 35(5):807–852, 2023.

[5] Karl Friston, Lancelot Da Costa, Danijar Hafner, Casper Hesp, and Thomas Parr. Sophisticated inference. Neural Computation, 33(3):713–763, 2021.

[6] Tobias Jung, Daniel Polani, and Peter Stone. Empowerment for continuous agent—environment systems. Adaptive Behavior, 19(1):16–39, 2011.

[7] Christoph Salge, Cornelius Glackin, and Daniel Polani. Empowerment–an introduction. In Guided selforganization: Inception, pages 67–114. Springer, 2014.

[8] Jorge Ramirez-Ruiz, Dmytro Grytskyy, Chiara Mastrogiuseppe, Yamen Habib, and Ruben Moreno-Bote. Complex behavior from intrinsic motivation to occupy future action-state path space. Nature Communications, 15(1):6368, 2024.

[9] Rubén Moreno-Bote and Jorge Ramirez-Ruiz. Empowerment, free energy principle and maximum occupancy principle compared. In NeurIPS 2023 workshop: Information-Theoretic Principles in Cognitive Systems, 2023.

[10] Richard D Smallwood and Edward J Sondik. The optimal control of partially observable markov processes over a finite horizon. Operations research, 21(5):1071–1088, 1973.

[11] Edward Jay Sondik. The optimal control of partially observable Markov processes. Stanford University, 1971.

[12] Hao Zhang. Partially observable markov decision processes: A geometric technique and analysis. Operations Research, 58(1):214–228, 2010.

[13] William S Lovejoy. Computationally feasible bounds for partially observed markov decision processes. Operations research, 39(1):162–175, 1991.

[14] William S Lovejoy. A survey of algorithmic methods for partially observed markov decision processes. Annals of Operations Research, 28(1):47–65, 1991.

[15] Edward J Sondik. The optimal control of partially observable markov processes over the infinite horizon: Discounted costs. Operations research, 26(2):282–304, 1978.

[17] Richard Blahut. Computation of channel capacity and rate-distortion functions. IEEE transactions on Information Theory, 18(4):460–473, 1972.

[16] Alex B Kiefer. Intrinsic motivation as constrained entropy maximization. Entropy, 27(4):372, 2025.

## A Appendix

## A.1 Definition of various state spaces

We list the definitions of the terms used throughout:

$\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } )$ Spatial Position   
${ x _ { f , i } } = ( 1 , 1 ) \mathrm { o r } ( 5 , 5 )$ Food Source Position   
$\boldsymbol { s } = ( \boldsymbol { x } , E )$ Location and Energy (Observable) State   
$h = ( h _ { 1 } , h _ { 2 } )$ Hidden Food State   
$( s , h ) = ( x , E , h _ { 1 } , h _ { 2 } )$ World State   
$\omega = ( \omega _ { 1 } , \omega _ { 2 } )$ Food Observation   
$b ( h ) = ( b ( h _ { 1 } ) , b ( h _ { 2 } ) )$ Belief(about hiddenfood state)   
$( s , b ) = ( x , E , b _ { 1 } , b _ { 2 } )$ Agent State (sufficient statistics)   
$\boldsymbol { s } ^ { \prime } = ( \boldsymbol { x } ^ { \prime } , E ^ { \prime } )$ Next Observable State   
$b ^ { \prime } ( h ^ { \prime } ) = ( b ^ { \prime } ( h _ { 1 } ^ { \prime } ) , b ^ { \prime } ( h _ { 2 } ^ { \prime } ) )$ Propagated Belief(one stepforward)

## A.2 Environment Implementation

Table 1: Parameters used in the experiments.
<table><tr><td>Variable</td><td>Value</td></tr><tr><td> $\overline { { E _ { \mathrm { m a x } } } }$ </td><td>30</td></tr><tr><td> $E _ { \mathrm { g a i n } }$ </td><td>15</td></tr><tr><td>Grid size Food source locations</td><td> $5 \times 5$ </td></tr><tr><td>Initial  $h _ { 1 }$ </td><td>(1, 1) and (5, 5) 1</td></tr><tr><td>Initial  $h _ { 2 }$ </td><td>1</td></tr><tr><td>Initial  $b ( h _ { 1 } = 1 )$ </td><td>1</td></tr><tr><td>Initial  $b ( h _ { 2 } = 1 )$ </td><td>1</td></tr><tr><td> $\lambda$ </td><td>0.8</td></tr><tr><td> $\mu$ </td><td>0.2</td></tr><tr><td> $\rho$ </td><td>0.8</td></tr><tr><td></td><td></td></tr><tr><td>Belief discretization step</td><td>0.1</td></tr></table>

Results remain qualitatively unchanged for other values of $E _ { \mathrm { m a x } } , E _ { \mathrm { g a i n } }$ and the food-regeneration probabilities: these parameters modulate how conservatively the agents behave, but not the relative comparison between the three methods.

## A.3 Discretization of beliefs

The belief state $b _ { i } \in [ 0 , 1 ]$ is discretized onto a uniform grid with step size $\Delta$ . The choice of $\Delta$ involves a trade-off: a smaller step yields a finer approximation but increases the number of belief states quadratically, since the agent maintains two independent beliefs $( b _ { 1 } , b _ { 2 } )$ . At the same time, the total number of belief combinations scales as $( \breve { 1 } / \Delta + 1 ) ^ { 2 }$ , so reducing ∆ from 0.1 to 0.01 increases the number of belief grid points by a factor of 100. We therefore adopt $\Delta = 0 . 1$ as a practical choice that keeps the state space tractable while preserving the structure of the belief dynamics. Since the belief propagated by Eq. 3 does not in general land on a grid point, we perform value iteration (Eq. 6) on the grid and evaluate the successor value $V ^ { * } ( s ^ { \prime } , b ^ { \prime } )$ by linear interpolation between the neighboring grid points at each iteration; for instance, for a propagated belief $b _ { 1 } ^ { \prime } = 0 . 8 4$ the value $\bar { V } ^ { * } ( s ^ { \prime } , b ^ { \prime } )$ is obtained by interpolating linearly between the stored values $V ^ { * } ( s ^ { \prime } , \bar { 0 } . 8 )$ and $V ^ { * } ( s ^ { \prime } , 0 . 9 )$

Figure 4 shows the value function $V ^ { * } ( s , b )$ , computed via Eq. 6, evaluated at three representative states as a function of $b _ { 1 }$ , with $b _ { 2 } = 0 . 5$ fixed, for both $\Delta = 0 . 1$ (coarse grid, black dots) and $\Delta = 0 . 0 1$ (fine grid, blue curve). In all three cases, the coarse grid points lie close to the fine curve, confirming that the discretization does not distort the value function — it merely quantizes it. The overall shape and magnitude of $V ^ { * }$ are preserved, and the differences between the two grids are negligible relative to the range of values. We therefore conclude that $\Delta = 0 . 1$ is a sufficient discretization for the purposes of this work.

a

![](images/1fe7e61de58f268fa1fbe4333f80d7defa8c780855b4dc87680ad18c52d6e2eb.jpg)  
Figure 4: Value function $V ^ { * } ( s , b )$ as a function of belief $b _ { 1 }$ (with $b _ { 2 } = 0 . 5$ fixed) for three representative states, computed with a fine grid $( \Delta \stackrel { \cdot } { = } 0 . 0 1$ , black curve) and a coarse grid $( \Delta = 0 . 1$ , blue curve). The coarse grid points closely follow the fine curve in all cases, confirming that the $\Delta = 0 . 1$ discretization introduces negligible approximation error.

## A.4 Beliefs in the two-food-source partially observable grid-world environment

Using Eq. 3, it is easy to check that when the agent is at the food source $i ,$ beliefs saturate

$$
\begin{array} { r l } & { b ^ { \prime } ( h _ { i } ^ { \prime } = 1 | x ^ { \prime } = x _ { f , i } , \ \omega _ { i } = 1 ) = 1 } \\ & { b ^ { \prime } ( h _ { i } ^ { \prime } = 1 | x ^ { \prime } = x _ { f , i } , \ \omega _ { i } = 0 ) = 0 , } \end{array}
$$

because the observation $\omega _ { i }$ is then fully informative about the hidden state $h _ { i }$ of the food source $x _ { f , i } .$ Here, we have replaced the conditioning on $( s , b , a )$ with $x ^ { \prime } = x _ { f , i }$ explicitly, since the saturation of beliefs happens only when the next position $x ^ { \prime } ( s , a )$ coincides with food source $x _ { f , i }$ . In contrast, outside food source $x _ { f , i }$ , the belief diffuses to the prior as

$$
b ^ { \prime } ( h _ { i } ^ { \prime } ) = \sum _ { h _ { i } } p ( h _ { i } ^ { \prime } | x , h _ { i } ) \ b ( h _ { i } ) \ .\tag{14}
$$

## A.5 Variational free energy and belief propagation

The VFE is expressed as

$$
F  { [ Q ] } = D _ { \mathrm { K L } } \big ( Q ( h ^ { \prime } ) \ \lVert \ p ( h ^ { \prime } ) \big ) - \mathbb { E } _ { Q ( h ^ { \prime } ) } \big [ \ln q ( \omega ^ { \prime } \ \lvert \ s ^ { \prime } , h ^ { \prime } ) \big ] ,
$$

where $Q ( h ^ { \prime } )$ is the variational distribution over the updated hidden food state $\begin{array} { r } { h ^ { \prime } , p ( h ^ { \prime } ) = \sum _ { h } p ( h ^ { \prime } | h , x ) Q ( h ) } \end{array}$ is the predictive prior induced by the transition dynamics, and $q ( \omega ^ { \prime } \mid s ^ { \prime } , h ^ { \prime } )$ is the observation model defined in Eq. 2. To prove that minimizing this equation is equivalent to belief propagation described in Sec. 2.4, we consider the two regimes induced by the structure of the observation model.

Case 1: Non-informative observations (agent away from food). When $\boldsymbol { x } ^ { \prime } \neq \boldsymbol { x } _ { f , i }$ <sub>i</sub> for the corresponding food source, the observation model is uniform:

$$
q ( \omega ^ { \prime } \mid s ^ { \prime } , h ^ { \prime } ) = \frac { 1 } { 2 } .
$$

Hence the accuracy term becomes independent of $h ^ { \prime } \colon$

$$
\begin{array} { r } { \mathbb { E } _ { Q ( h ^ { \prime } ) } [ \ln q ( \omega ^ { \prime } \mid s ^ { \prime } , h ^ { \prime } ) ] = \ln \frac { 1 } { 2 } , } \end{array}
$$

which is constant with respect to $Q ( h ^ { \prime } )$ . The VFE therefore reduces to

$$
F [ Q ] = D _ { \mathrm { K L } } \big ( Q ( h ^ { \prime } ) \| p ( h ^ { \prime } ) \big ) + \mathrm { c o n s t . }
$$

Minimization yields

$$
Q ( h ^ { \prime } ) = p ( h ^ { \prime } ) = \sum _ { h } p ( h ^ { \prime } | h , x ) Q ( h ) ,
$$

so that the posterior belief is entirely driven by the transition model. In this regime, belief updating reduces to pure prediction under the dynamics, recovering the propagation rule in Eq. (14).

Case 2: Informative observations (agent at food location). When the agent is located at food source $x _ { f , i }$ (i.e. $x ^ { \prime } = x _ { f , i } )$ , the observation model becomes deterministic as specified in Eq. 2, so observations perfectly identify the hidden state. In this case, the accuracy term becomes

$$
- \mathbb { E } _ { Q ( h ^ { \prime } ) } [ \ln q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } ) ] = - \sum _ { h ^ { \prime } \in \{ 0 , 1 \} } Q ( h ^ { \prime } ) \ln q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } ) .
$$

Since $q ( \omega ^ { \prime } \mid s ^ { \prime } , h ^ { \prime } )$ is degenerate, there exists a unique state $h ^ { \star }$ such that

$$
q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \star } ) = 1 , \qquad q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } \neq h ^ { \star } ) = 0 .
$$

If $Q ( h ^ { \prime } \neq h ^ { \star } ) > 0$ , the corresponding term in the expectation contains ln $0 = - \infty$ , implying

$$
F [ Q ] = + \infty .
$$

Therefore, any finite minimizer must satisfy

$$
Q ( h ^ { \star } ) = 1 , \qquad Q ( h ^ { \prime } \neq h ^ { \star } ) = 0 .
$$

Thus, the optimal posterior collapses to a point mass:

$$
Q ( h ^ { \prime } ) = \delta ( h ^ { \prime } - h ^ { \star } ) ,
$$

which assigns full belief to the observed-consistent hidden state. In both cases, the resulting updates coincide with the belief propagation dynamics derived in Eq. (14) and the deterministic update at food locations, respectively. The minimizer $Q ^ { * } ( \bar { h ^ { \prime } } )$ corresponds to the updated belief $b ^ { \prime } ( h ^ { \prime } )$ used throughout this work, $\mathrm { i . e . } \ b ^ { \prime } ( h ^ { \prime } ) = \bar { \mathrm { a r g } } \operatorname* { m i n } _ { Q } \cal { F } [ Q ]$ Hence, variational free energy minimization provides a unified principle underlying the belief update equations in this grid-world POMDP.

## A.6 Empowerment

Consider a state $( s , b )$ at time $t ,$ which serves as the initial condition. The agent then executes an n-step action sequence $a ^ { n } = ( a _ { t } , \ldots , a _ { t + n - 1 } )$ , leading to a future observation state $o _ { t + n } = ( s _ { t + n } , \omega _ { t + n } )$ at time $t + n$ . The empowerment of state (s, b) is defined as

$$
C ( s , b ) = \operatorname* { m a x } _ { p ( a ^ { n } ) } \mathbb { E } _ { P ( a ^ { n } , o _ { t + n } | s , b ) } \left[ \log \frac { P ( o _ { t + n } | a ^ { n } , s , b ) } { P ( o _ { t + n } | s , b ) } \right] ,\tag{15}
$$

where the expectation is taken over the joint distribution of action sequences and future observations, conditioned on the initial state $( s , b )$ at time t. The marginal predictive distribution over future observations, obtained by averaging over all possible action sequences, is

$$
P ( o _ { t + n } | s , b ) = \sum _ { \tilde { a } ^ { n } } { \cal P } ( \tilde { a } ^ { n } ) { \cal P } ( o _ { t + n } | \tilde { a } ^ { n } , s , b ) .\tag{16}
$$

The predictive distribution over future observations factorizes as

$$
P ( o _ { t + n } | a ^ { n } , s , b ) = P ( s _ { t + n } | a ^ { n } , s , b ) \ P ( \omega _ { t + n } | s _ { t + n } , b _ { t + n } ) ,\tag{17}
$$

where $s _ { t + n } = ( x _ { t + n } , y _ { t + n } , E _ { t + n } )$ is the future observable state reached after executing $a ^ { n }$ from $( s , b ) , \omega _ { t + n }$ denotes the observation generated at that state through the observation model (Eq. 2), and $b _ { t + n } ( a ^ { n } , s , b )$ is the belief propagated forward n steps from the initial belief b under action sequence $a ^ { n }$ , updated at each intermediate step $k = t , \ldots , t + n - 1$ via the prediction step

$$
b _ { k + 1 } ( h ^ { \prime } ) = \sum _ { h } p ( h ^ { \prime } | h , x _ { k } ) b _ { k } ( h ) ,\tag{18}
$$

noting that intermediate observations are not conditioned upon, as empowerment considers all possible future trajectories rather than a single realized path. This factorization follows from the conditional independence of $\omega _ { t + n }$ and $a ^ { n }$ given $s _ { t + n }$ and $b _ { t + n } \colon$ once the future state $s _ { t + n }$ and the propagated belief $b _ { t + n }$ are known, the observation $\omega _ { t + n }$ depends only on the hidden food state and the agent’s location, and not on the specific action sequence that led there.

Computing the empowerment $C ( s , b )$ requires maximizing the mutual information over all input distributions $p ( a ^ { n } )$ , which is a concave optimization problem solved here using the Blahut–Arimoto algorithm [17] (see Appendix A.7).

Unlike the EFE and the value function $V ^ { * }$ of MOP, which both follow a Bellman structure and are computed via value iteration until convergence over the full state-belief space, empowerment does not admit a recursive decomposition of this form. Instead, for each state $( s , b )$ independently, all possible n-step action sequences are enumerated explicitly and their associated observation distributions are precomputed via forward belief propagation. The mutual information is then maximized over the input distribution $p ( a ^ { n } )$ using the Blahut–Arimoto algorithm, which iterates until convergence for that specific state. The key distinction from value iteration is therefore that empowerment requires no information from neighboring states in the belief-state space — each state $( s , b )$ is solved as an entirely independent optimization problem. This makes the calculation exact for a fixed horizon n, but comes at a significant computational cost: the number of candidate action sequences grows as $| A | ^ { n }$ , where |A| is the number of available actions, so increasing n by even one step multiplies the number of sequences by a factor of |A|. This exponential scaling imposes a practical upper limit on the horizon n, beyond which the computation of $C ( s , \dot { b } )$ becomes intractable.

## A.7 Implementation of MOP, EFE and Empowerment

MOP The value function for MOP is computed by solving the Eq. 6 via value iteration, starting from an initial guess $V _ { 0 } ( s , b ) = 0$ and iterating until the value function converges. The discount factor $\gamma$ controls the effective planning horizon: a value close to one weights distant future states almost as heavily as immediate ones, while a smaller value emphasizes short-term outcomes. We set $\gamma = 0 . 9 9$ , which corresponds to an effective horizon of approximately $1 / ( \bar { 1 } - \gamma )$ ≈ 100 steps into the future. We impose the boundary conditions $V ( s ^ { + } , b ) = 0$ for every terminal observable state $s ^ { + }$ (observable states s with $E = 0 )$ regardless of $b ,$ so that terminal states have zero value due to the termination of the episode and the impossibility of generating any further action-state path entropy. The value iteration is terminated once the maximum change in the value function across all states falls below a threshold $\theta _ { \mathrm { M O P } } = 1 0 ^ { - 2 }$ , which is usually reached after $N _ { \mathrm { M O P } } = \bar { 2 } 5 0$ iterations.

EFE The EFE is computed via the same value-iteration procedure applied to the recursive expression in Eq. 11, starting from $G _ { 0 } ( s , b ) = 1 0 0$ and iterating until convergence. As in MOP, the discount factor γ governs how far into the future the agent plans, and we use the same value $\gamma = 0 . 9 9$ for both schemes to ensure a fair comparison; this again corresponds to an effective horizon of roughly 100 steps. The EFE additionally involves the inverse temperature d, which controls the stochasticity of the policy used in the recursive backup: as $d \to \infty$ the policy becomes deterministic (Eq. 13), while finite d yields a soft policy. The effect of d is explained in detail in Appendix A.9. The value iteration is terminated once the maximum change in $G ( s , b )$ across all states falls below $\theta _ { \mathrm { E F E } } = 1 0 ^ { - 2 }$ , reached after $N _ { \mathrm { E F E } } = 3 0 0 - 4 0 0$ iterations, depending on d.

Empowerment Computing $C ( s , b )$ requires maximizing the mutual information over all input distributions $p ( a ^ { n } )$ a concave optimization problem solved here using the Blahut–Arimoto algorithm [17]. For a fixed state $( s , b )$ , all $| A | ^ { n }$ action sequences are enumerated and the resulting observation distributions $P ( o _ { t + n } | a ^ { n } , s , b )$ are precomputed via forward belief propagation. The algorithm iteratively reweights the input distribution $p ( a ^ { n } )$ to maximize the mutual information between action sequences and future observations, until convergence within a threshold $\theta _ { \mathrm { B A } } = 1 0 ^ { - 5 }$ typically reached after $N _ { \mathrm { B A } } = 5 0$ iterations, yielding the optimal input distribution $p ^ { * } ( a ^ { n } )$ and the empowerment value $C ( s , b )$ . This procedure is repeated independently for every state (s, b) in the discretized belief-state space.

## A.8 Explaining the EFE equation

The per-action EFE is defined in Eq. 11, as

$$
\begin{array} { r } { G ( s , b , a ) = \underbrace { \mathbb { E } _ { P ( \omega ^ { \prime } , s ^ { \prime } , h ^ { \prime } \mid s , b , a ) } \left[ \widehat { \ln { P ( s ^ { \prime } \mid s , b , a ) } - \ln { P ( s ^ { \prime } ) } } \widehat { - \ln { q ( \omega ^ { \prime } \mid s ^ { \prime } , h ^ { \prime } ) } } \right] } _ { \mathrm { E x p e c t e d f r e e ~ e n c r g y ~ o f ~ n e x t a c t i o n } } } \\ { + \gamma \underbrace { \mathbb { E } _ { P ( s ^ { \prime } , b ^ { \prime } \mid s , b , a ) } \left[ G ( s ^ { \prime } , b ^ { \prime } ) \right] } _ { \mathrm { E x p e c t e d f r e e ~ e n c e q u s e q u e n ~ a t i o n s } } } \end{array}
$$

Since the risk depends only on the marginal distribution over $s ^ { \prime } ,$ , it can be simplified by integrating $\omega ^ { \prime }$ and $h ^ { \prime }$ in the predictive model $( \operatorname { E q . 5 } )$

$$
\begin{array} { r l } & { \mathbb { E } _ { P ( \omega ^ { \prime } , s ^ { \prime } , h ^ { \prime } \mid s , b , a ) } \Big [ \ln P ( s ^ { \prime } \mid s , b , a ) - \ln P ( s ^ { \prime } ) \Big ] = \mathbb { E } _ { P ( s ^ { \prime } \mid s , b , a ) } \Big [ \ln P ( s ^ { \prime } \mid s , b , a ) - \ln P ( s ^ { \prime } ) \Big ] } \\ & { \qquad = D _ { K L } \Big [ P ( s ^ { \prime } \mid s , b , a ) \ \parallel \ P ( s ^ { \prime } ) \Big ] , } \end{array}
$$

where $D _ { K L }$ denotes the KL divergence between the predicted distribution over future states and the agent’s prior preferences. This term measures how far the agent expects to deviate from its preferred states, minimizing risk i equivalent to selecting actions that drive the agent towards states that conform to its prior preferences.

![](images/7758ea36f6dc14f8b04a1d67b8d77125ae675d74b2dcd552523bb904a58bf3ff.jpg)  
Figure 5: Behavior of an active inference agent under stochastic policy, for various values of the inverse temperature, in the case of location-independent dynamics $( G ^ { + } = 1 0 0$ , 100 episodes per panel). One complete episode is drawn on each panel (start •, end ×); the line retraces itself, so a committed agent appears as a few repeated edges rather than as a short path. In panels (b)–(f) the episodes are conditioned on the food source reached first, so that each heatmap shows a single commitment rather than a mixture of two.

Similarly, the ambiguity term, for this environment, can be simplified

$$
\begin{array} { r } { \mathbb { E } _ { P ( s ^ { \prime } , \omega ^ { \prime } , h ^ { \prime } | s , b , a ) } \Big [ - \ln q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } ) \Big ] \propto H \Big [ q ( \omega ^ { \prime } | s ^ { \prime } , h ^ { \prime } ) \Big ] , } \end{array}
$$

where H denotes the Shannon entropy of the observation model. In the grid-world environment examined here, this term takes one of two values depending on the agent’s location, following directly from the observation model in Eq. 2. When the agent is at a food source, q collapses to a delta function and its entropy is zero — the observation is fully informative about the hidden food state. Away from a food source, the observation is completely uninformative and the entropy attains its maximum value, $\begin{array} { r } { H \big [ q ( \omega ^ { \prime } | \dot { s } ^ { \prime } , h ^ { \prime } ) \big ] = \ln \frac { 1 } { 2 } } \end{array}$ . Minimizing ambiguity therefore drives the agent towards food locations, where it can resolve its uncertainty about the hidden state of the environment most effectively.

## A.9 Stochastic and deterministic policies of Active Inference

The inverse temperature parameter d, introduced in Eq. 12, controls the degree of stochasticity in the EFE policy and has a significant effect on the agent’s behavior and survival. Fig. 5 shows the spatial visitation heatmaps of the agent for six values of d, and Fig. 6 quantifies the same behavior per episode, revealing three distinct regimes.

For $d \leq 1$ the policy is stochastic and the agent is explorative: it reaches a food source in every episode, visits both sources in 80–85% of episodes, covers a median of 22–23 of the 25 grid cells, and travels between the two sources several times per thousand steps. Survival is shortest in this regime $( 5 3 9 \pm 6 0$ steps at $d = 0 . 5 )$ . For $1 \leq d \leq 2$ the agent still uses both sources, but typically only once in a lifetime: at $d = 1 . 6$ half of the episodes reach the second source, and they do so at a twentieth of the rate observed at $d = 0 . 5$ . For d $\gtrsim 2$ the agent commits to a single source and stays there: by $d = 2 . 5$ only 5% of episodes ever reach the other one, from $d = 3$ almost none do, and occupancy of a single source rises from 68% of the time at $d = 2 \ : \mathrm { t o } 9 9 . 8 \%$ at $d = 1 0$ , over a median of 6 cells, with the longest survival of the range $( 4 8 8 3 \pm 3 8 1$ steps at $d = 1 0 )$ . While the normalized policy entropy is high (at $d = 2 . 5$ the normalized policy entropy is still 0.60 at $E = 2 2 ( \mathrm { F i g . 3 b } ) $ ), the direction in which the policy points becomes concentrated, and looks towards the food source.

MOP is drawn as a reference in each panel of Fig. 6. Of the three regimes, it most resembles the EFE agent at small d: it reaches both food sources in 94% of episodes and spends only 26% of its time standing on one (Fig. 6b), figures that only the lowest precisions approach. It nevertheless behaves differently. MOP moves between the two sources continually, 13.3 trips per thousand steps, more than any EFE agent at any precision $( \mathrm { F i g . 6 a } )$ , and it spends far more of its time away from food. Fig. 6c makes this comparison at $d = 0 . 8 .$ , the precision with comparable policy entropy and lifespan: the EFE agent stands on a source $4 1 \%$ of the time against $\mathbf { M O P } \mathbf { \bar { s } }$ 26%, while MOP spends 42% of its time two or more steps from the nearest source against the EFE agent’s 22%, nearly twice as much. Matched for how variable its policy is and for how long it lives, the EFE agent still hovers around food where MOP ranges away from it; food-source occupancy dominates its behavior across the whole range of d tested.

a  
![](images/5240ed050d2cd56e3f1909ed1c3a4b0a1d921ad0a565d7bb74fe48492b3df680.jpg)  
b

![](images/65039caab37eeabaacd50d44dc967181a1f1eb72fad52569899a099928b04093.jpg)

c  
![](images/e92dd006f392013bd2cfaad1bb32c27abac37bdf2ce62f89571a6f2b1e7fd4c7.jpg)  
Figure 6: $( \mathbf { a } , \mathbf { b } )$ Per-episode behavior of the active inference agent across precision, 300 episodes per point; MOP (black) is drawn in both panels as a reference. (a) Number of trips between the two sources per $1 0 ^ { \bar { 3 } }$ steps, over the range of precision d. (b) Share of time spent standing on a food source. (c) Occupancy resolved by Manhattan distance to the nearest food source, for MOP and for the agent at $d = 0 . 8$ , the precision whose lifespan comes closest to MOP’s: with lifespans matched, the active inference agent still spends 41% of its time standing on a source against MOP’s 26%, while MOP retains the occupancy at distances 3 and 4 that the active inference agent has largely given up.

## A.10 Boundary condition on the calculation of the EFE

The terminal boundary condition $G ^ { + } = G ( s ^ { + } , b )$ , imposed on every terminal observable state $s ^ { + }$ (observable states with $E = 0 )$ regardless of b, is the price the agent pays for dying, and it is not free to be chosen independently: the model already assigns a prior preference to $E = 0$ through the target distribution $P ( s )$ (Sec. 3.2), and consistency with that preference fixes its value. Terminal states are absorbing, so the only path available from $s ^ { + }$ is the indefinite repetition of $s ^ { + }$ itself, and evaluating Eq. 11 along that path requires $G ^ { + }$ to satisfy its own Bellman equation,

$$
G ^ { + } = g ^ { + } + \gamma G ^ { + } \qquad \Longrightarrow \qquad G ^ { + } = \frac { g ^ { + } } { 1 - \gamma } = \frac { - \ln P ( s \mid E = 0 ) } { 1 - \gamma } ~ ,\tag{19}
$$

where $g ^ { + } = -$ ln $P ( s | E { = } 0 )$ is the risk incurred per step of occupying a terminal state, the self-transition being deterministic. With $\stackrel { \cdot } { P } ( \stackrel { \cdot } { s } | E { = } 0 ) = 1 0 ^ { - 6 } \mathrm { a n d } \gamma = \stackrel { \cdot } { 0 . 9 9 }$ this gives $G ^ { + }$ ≈ 1382. Read in the opposite direction, Eq. 19 assigns to every choice of $G ^ { + }$ an implied preference for the terminal state,

$$
P ( s \mid E { = } 0 ) = \exp \big [ - G ^ { + } ( 1 - \gamma ) \big ] ,\tag{20}
$$

so that $G ^ { + } = 0$ implies $P = 1$ , death exactly as preferred as any other state; $G ^ { + } = 1 0 0$ implies $P = 0 . 3 7 ;$ ; and only $G ^ { + } \approx 1 3 8 2$ recovers the $1 0 ^ { - 6 }$ that the risk term itself already uses.

a  
b  
![](images/7c9dc920e21d9e5aac467d71109915e5743da790a821f07ae5b339b604b23d34.jpg)  
Figure 7: Normalized policy entropy of the EFE agent against energy level, for three terminal boundary conditions $G ^ { + }$ with the MOP agent (black) as a fixed reference in each panel. MOP value function has no EFE terminal, so the same curve is the appropriate comparison throughout. Curves are labeled by the inverse temperature d.

The choice is behavioral, not merely numerical (Fig. 7). With $G ^ { + } = 0$ death carries no cost, and for every $d \lesssim 2 . 5$ the converged G ranks a nearly dead state as better than a full one, so the agent is rewarded for starving; its policy entropy is flat in energy and close to uniform at low d (panel a), giving behavior that is at once random and self-destructive. At the consistent value $G ^ { + } = 1 3 8 2$ the opposite holds: the agent becomes strongly risk-averse, its policy entropy collapses to near zero below E ≈ 10–15 at every precision (panel c), and it survives far longer while occupying a much smaller part of the grid — long-lived, but nearly deterministic and behaviorally impoverished. The intermediate value $G ^ { + } = \mathrm { \hat { 1 } } 0 0$ (panel b) is the most informative of the three: the ordering of G in energy is correct, so the agent no longer seeks death, yet its policy remains genuinely stochastic and its entropy increases with energy as $\mathbf { \vec { M O P } \vec { s } }$ does, so that the agent both survives and explores. We therefore use $G ^ { + } = 1 0 0$ throughout this paper. For large d the three panels are nearly indistinguishable, since as $d \to \infty$ only the arg-min of $G ( s , b , a )$ matters and the terminal enters as a nearly uniform offset across actions, leaving the deterministic agent insensitive to $G ^ { + }$ . A larger $G ^ { + }$ also slows the convergence of value iteration, as the terminal value must propagate outward across the belief-state space before G stabilizes.