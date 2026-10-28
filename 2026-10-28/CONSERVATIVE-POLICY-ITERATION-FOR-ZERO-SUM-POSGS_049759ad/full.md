# CONSERVATIVE POLICY ITERATION FOR ZERO-SUM POSGS

Erwan C. Escudie e.c.escudie@rug.nl University of Groningen

Matthia Sabatelli m.sabatelli@rug.nl University of Groningen

Jilles Steeve Dibangoye j.s.dibangoye@rug.nl University of Groningen

## ABSTRACT

Occupancy-state reductions make dynamic programming possible for zero-sum partially observable stochastic games (zs-POSGs). Existing point-based solvers usually optimise a central value function. This restores Bellman recursion, but it requires point-based updates over high-dimensional occupancy spaces. We introduce an approximate policy-iteration algorithm for finite-horizon zs-POSGs. The method maintains one policy per player and updates both jointly along occupancy trajectories induced by conservative policy updates. Instead of approximating a full central value function, each update uses a single supporting hyperplane induced by the current joint policy. The algorithm alternates local policy improvements, conservative updates, and persistent exploration. Computation is therefore concentrated on occupancy regions where the current joint policy is exploitable. On finite-horizon zero-sum POSG benchmarks, the method is competitive at small horizons and often improves the runtime–exploitability trade-off at larger horizons compared with PBVI, HSVI, and CFR+.

## 1 INTRODUCTION

Planning under imperfect information requires agents to act from histories rather than from the true state. In partially observable Markov decision processes (POMDPs), belief states restore a Markovian dynamic program and support exact and point-based solvers (Aström, 1965; Smallwood & Sondik, 1973; Sondik, 1978; Pineau et al., 2003; Smith & Simmons, 2004). In partially observable stochastic games (POSGs), partial observability is coupled with strategic uncertainty. Each player must reason about the hidden state, its own information, and the possible information and actions of the other players (Hansen et al., 2004; Kovaˇrík et al., 2022). The two-player zero-sum case, called a zero-sum partially observable stochastic game (zs-POSG), is the adversarial form of this problem. Dynamic programming is classical for fully observable stochastic games (Shapley, 1953; Filar & Vrieze, 1997), but partial observability requires a sufficient statistic for planning. Common-information, plan-time, and occupancy-state reductions provide such statistics for decentralised control and Dec-POMDPs (Nayyar et al., 2013; Dibangoye et al., 2013; 2016a; Oliehoek, 2013). Recent work has extended this programme to adversarial settings, including value-structure results, one-sided partially observable stochastic games, public-observation games, and general twoplayer zs-POSGs (Wiggers et al., 2016; Horák et al., 2017; Horák & Bošanský, 2019; Kartik & Nayyar, 2019; Buffet et al., 2020; Delage et al., 2023; Sokota et al., 2023; Escudie et al., 2025). In these reductions, a zs-POSG is lifted to an occupancy stochastic game. The lifted state is a distribution over hidden states and histories. The lifted actions are decision rules. The reduced game is fully observed by the planner. It preserves the value of the original game and restores Bellman recursion.

This progress leaves an algorithmic gap. Existing dynamic-programming solvers for zs-POSGs are mostly point-based value-iteration methods (Pineau et al., 2003; Smith & Simmons, 2004; Delage et al., 2023; Escudie et al., 2025). They approximate the optimal central value on sampled occupancy states and apply Bellman backups, for all samples in sync. This is principled, but it creates a coverage problem. The approximation must support backups over a large occupancy space. Playerspecific policy improvement needs accurate value information only where the current policy can fail to be secure. In zero-sum games, these regions are not determined by the current policy alone. They are exposed by unilateral deviations and worst-case responses. The difficulty is therefore not only to extract executable policies from a central planner. It is the mismatch between global value approximation and local policy improvement.

Policy-update methods address another side of the same problem. Fictitious play, counterfactual regret minimisation, and CFR+ update policies directly in imperfect-information games (Robinson, 1951; Zinkevich et al., 2007; Neller & Lanctot, 2013; Tammelin, 2014; Moravcík et al.ˇ , 2017; Brown et al., 2020). Classical fictitious play, regret minimisation, and CFR-style methods show that repeated local improvement can solve large zero-sum imperfect-information games without explicitly representing a global dynamic program (Robinson, 1951; Zinkevich et al., 2007; Neller & Lanctot, 2013; Tammelin, 2014). Sequence-form methods and information-set decompositions provide compact representations of extensive-form games (Koller et al., 1996; Moravcík et al. ˇ , 2017; Brown et al., 2020; Kovaˇrík et al., 2022). Policy-improvement and trust-region methods provide another route to executable updates in reinforcement learning and multi-agent reinforcement learning (Bertsekas & Tsitsiklis, 1996; Munos, 2003; Kakade & Langford, 2002; Schulman et al., 2015; 2017; Yu et al., 2022; Zhong et al., 2024). Search and policy improvement have also been combined in cooperative or partially observable multi-agent settings (Lerer et al., 2020; Van der Pol & Oliehoek, 2016; Wei et al., 2019; Pieroth et al., 2024). These methods are attractive because they update policies at histories or information states that are executable by construction.

Existing methods provide either a principled value equation or executable policy updates. They do not yet provide both in a form suited to focused approximation. The central occupancy reduction provides a Bellman equation, but its state is broad and not directly executable. Focal planning makes policy extraction player-specific, but its state is set-valued and lacks a tractable value representation. Policy-space methods update executable policies, but they do not provide a Bellman representation for focused point-based backups. The missing object is therefore a player-specific value representation that keeps executable decision rules while supporting local Bellman backups.

To address this gap, we introduce Conservative Policy Iteration (CPI) for finite-horizon zs-POSGs. CPI maintains one policy per player and improves both policies along occupancy trajectories induced by conservative policy updates. For each player, the current policy induces a value function in the corresponding player-specific planning process. CPI uses this policy-induced value function for local Bellman backups, instead of constructing a global central value function. Each backup computes an improved decision rule for one player from its current value function. The two policies are updated together, so the opposing policy used during rollouts is supplied by the other playerspecific planning process. The improved decision rules are averaged into the current policies, and GLIE pertubations maintain exploration. Thus, updates are performed where the current joint policy is tested by adversarial interaction, rather than by uniformly covering the central occupancy space. The convergence guarantees are conditional on exact local backups, sufficient coverage, and the stated regularity assumptions. We evaluate CPI on standard finite-horizon zero-sum POSG benchmarks. The method is competitive at small horizons. At larger horizons, it often improves the runtime-exploitability trade-off compared with PBVI (Escudie et al., 2025), HSVI (Delage et al., 2023), and CFR+ (Tammelin, 2014).

## 2 PRELIMINARIES

Finite-horizon zero-sum POSGs are stochastic games with hidden states and imperfect information.

Definition 1. A finite-horizon zero-sum partially observable stochastic game (zs-POSG) is a tuple $\mathcal { M } = ( \mathcal { T } , \mathcal { S } , \mathcal { A } , \mathcal { Z } , \mathcal { Z } ^ { 0 } , p , ( r ^ { i } ) _ { i \in \mathcal { T } } , T , b _ { 0 } )$ , where $\dot { \mathcal { T } } = \left\{ 1 , 2 \right\}$ , S is a finite state space, $\mathcal { A } = \times _ { i \in \mathcal { T } } \mathcal { A } ^ { i }$ is the joint action space, ${ \mathcal Z } = \times _ { i \in { \mathcal T } } { \mathcal Z } ^ { i }$ is the joint private observation space, $\mathcal { Z } ^ { 0 }$ is the public observation space, $p \dot { : } \mathcal { S } \times \mathcal { A }  \Delta ( \bar { S } \times \mathcal { Z } ^ { 0 } \times \mathcal { Z } )$ is the transition kernel, $r ^ { i } : \mathcal { S } \times \mathcal { A } $ R is player $i \ ' s$ reward, $r ^ { 2 } = - \bar { r } ^ { 1 } , b _ { 0 } \in \Delta ( \mathcal { S } )$ is the initial state distribution, and $T$ is the horizon.

The game unfolds over stages $t = 0 , \ldots , T - 1$ . At stage t, the players choose $a _ { t } ,$ each player i receives $r ^ { i } ( s _ { t } , a _ { t } )$ , and the game samples $( s _ { t + 1 } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } )$ according to $p ( \cdot \mid s _ { t } , a _ { t } )$ . Because the state is hidden, i acts from its private history $h _ { t } ^ { i }$ and the public history $h _ { t } ^ { 0 }$ . These histories are empty at $t = 0$ . For $t \geq 1 , h _ { t } ^ { i } \doteq \dot { a } _ { 0 } ^ { i } z _ { 1 } ^ { i } \cdot \cdot \cdot a _ { t - 1 } ^ { i } \dot { z } _ { t } ^ { i }$ records $i \mathrm { \ ' } _ { \mathrm { s } }$ past actions and private observations, and $h _ { t } ^ { 0 } \doteq z _ { 1 } ^ { 0 } \cdots z _ { t } ^ { 0 }$ records the public observations. We write $h _ { t } \doteq \left( h _ { t } ^ { 1 } , h _ { t } ^ { 2 } \right)$ for a joint private history, with reachable sets $\mathcal { H } _ { t } ^ { i } , \mathcal { H } _ { t }$ , and $\mathcal { H } _ { t } ^ { 0 }$

A decision rule for player i is a map $\pi _ { t } ^ { i } : \mathcal { H } _ { t } ^ { i } \times \mathcal { H } _ { t } ^ { 0 } \to \Delta ( A _ { i } )$ . A joint decision rule is $\begin{array} { r l } { \pi _ { t } } & { { } = } \end{array}$ $( \pi _ { t } ^ { 1 } , \pi _ { t } ^ { 2 } ) \in \Pi _ { t } \doteq \Pi _ { t } ^ { \bar { 1 } } \times \bar { \Pi } _ { t } ^ { 2 }$ . A behavioural policy profile is $\pi = ( \pi _ { 0 } , \ldots , \pi _ { T - 1 } ) \in \Pi$ . We write $\pi ^ { i } \in$ Π<sup>i</sup> for a full behavioural policy of player i. We write $\pi _ { t : T } ^ { i } \in \Pi _ { t : T }$ for player i’s policy suffix and $\pi _ { t : T } \in \Pi _ { t : T }$ for a policy-profile suffix from stage t onward. We write $\pi _ { 0 : t } ^ { i } \in \Pi _ { 0 : t } ^ { i }$ for player i’s policy prefix and $\textstyle \pi _ { 0 : t } \in \Pi _ { 0 : \iota }$ <sub>t</sub> for a policy-profile prefix.

Policies are evaluated by expected cumulative payoff. Given $b _ { 0 } , \ i ^ { \prime } { \bf s }$ payoff under π is $J _ { i } ^ { \pi } \stackrel { . } { = }$ $\begin{array} { r } { \mathbb { E } _ { b _ { 0 } , p , \pi } [ \sum _ { k = 0 } ^ { T - 1 } r ^ { i } ( S _ { k } , A _ { k } ) ] } \end{array}$ . Since $r ^ { 2 } = - r ^ { 1 }$ , we write the zero-sum game from player 1’s perspective. The finite-horizon zs-POSG induces a finite zero-sum extensive-form game with perfect recall, so the classical minimax and behavioural-policy foundations apply (Neumann, 1928; Kuhn, 1953). Thus,

$$
v _ { 0 } ^ { 1 , * } ( b _ { 0 } ) = \operatorname* { m a x } _ { \pi ^ { 1 } \in \Pi ^ { 1 } } \operatorname* { m i n } _ { \pi ^ { 2 } \in \Pi ^ { 2 } } J _ { 1 } ^ { ( \pi ^ { 1 } , \pi ^ { 2 } ) } = \operatorname* { m i n } _ { \pi ^ { 2 } \in \Pi ^ { 2 } } \operatorname* { m a x } _ { \pi ^ { 1 } \in \Pi ^ { 1 } } J _ { 1 } ^ { ( \pi ^ { 1 } , \pi ^ { 2 } ) } .
$$

A policy profile attaining this value is a Nash equilibrium of the induced zero-sum game. Equivalently, player i’s component is maximin optimal for $i \mathrm { \ ' } _ { \mathrm { s } }$ payoff for any $i \in \mathcal { T }$ . This player-specific view is central below. A value-based method must ultimately select decision rules that can be implemented from each player’s private and public information.

## 2.1 OCCUPANCY-STATE DYNAMIC PROGRAMMING

Bellman recursion relies on a state that is sufficient for the continuation problem. For a policyprofile prefix $\pi _ { 0 : t }$ , the central occupancy state is $o _ { t } ( \pi _ { 0 : t } ) = \mathbb { P } _ { b _ { 0 } , p , \pi _ { 0 : t } } ( S _ { t } ^ { \phantom { : } } , H _ { t } , H _ { t } ^ { 0 } )$ . We write $o _ { t }$ when the prefix is clear. It belongs to $\mathcal { O } _ { t } \subseteq \Delta ( \boldsymbol { S } \times \mathcal { H } _ { t } \times \mathcal { H } _ { t } ^ { 0 } )$ . Although $o _ { t }$ is not observed by either player, it is Markovian for a central planner. Given $o _ { t }$ and $\pi _ { t }$ , the next occupancy is $\tau _ { t } ( o _ { t } , \pi _ { t } ) =$ $\mathbf { \tilde { \mathbb { P } } } _ { o _ { t } , \pi _ { t } } ( S _ { t + 1 } , H _ { t + 1 } , H _ { t + 1 } ^ { 0 } )$ . Player $i \gamma _ { \mathrm { s } }$ one-step expected reward is $\rho _ { i , t } \big ( o _ { t } , \pi _ { t } \big ) \bar { = } \mathbb { E } _ { o _ { t } , \pi _ { t } } \big [ r ^ { i } ( S _ { t } , \varDelta _ { t } ) \big ]$ The central occupancy state restores Bellman’s principle for the zero-sum value. With $v _ { T } ^ { 1 , * } ( \cdot ) \stackrel { } { = } 0$ the central dynamic program is

$$
v _ { t } ^ { 1 , * } ( o _ { t } ) = N E _ { ( \pi _ { t } ^ { 1 } , \pi _ { t } ^ { 2 } ) } \{ \rho _ { 1 , t } ( o _ { t } , \pi _ { t } ) + v _ { t + 1 } ^ { 1 , * } ( \tau _ { t } ( o _ { t } , \pi _ { t } ) ) \}
$$

where NE denotes the operator that returns the value associated to the Nash equilibrium of the corresponding subgame.

This recursion is the basis of exact and approximate occupancy-based planning methods for Dec-POMDPs and zs-POSGs (Dibangoye et al., 2012; 2013; 2014; 2016b;a; Dibangoye & Buffet, 2018; Delage et al., 2023; Peralez et al., 2025; 2024; Escudie et al., 2025). The difficulty is not the recursion itself, but the space on which it acts. The number of coordinates needed to represent $o _ { t }$ grows exponentially with the horizon. Point-based value iteration is a natural approximation paradigm for such continuous-state dynamic programs, but its zs-POSG instantiations provide limited guidance on where to place samples (Pineau et al., 2003; Smith & Simmons, 2004; Shani et al., 2013; Delage et al., 2023; Escudie et al., 2025).

The central planner game provides the correct value equation and, in principle, produces the optimal decision rules. The difficulty is instead computational: the central occupancy state induces a very large decision space. In particular, solving the corresponding subgame requires optimizing over decision rules defined on the possible central states, which can be viewed as a large matrix whose rows correspond to decision rules. This representation quickly becomes intractable as the horizon and information spaces grow.

The challenge lies in exploiting the decision rules produced by the central formulation without having to explicitly enumerate or sample the entire central occupancy space. In particular, we seek a representation that retains the correct Bellman recursion while allowing the relevant parts of the decision-rule space to be explored efficiently.

## 3 FOCAL BELIEF GAMES

We rewrite the central state by conditioning on the information relevant to the focal player i and on the policies that determine the other players’ behaviour. More precisely, we condition on the public history, the opponent’s private history, and the focal player’s policy prefix. This representation allows us to separate two steps of the solution procedure. First, during policy evaluation, we fix the focal player’s current policy and compute the value it can guarantee against an opponent that best responds to it. This gives a pessimistic continuation value for the focal player’s policy. Second, during policy improvement, this continuation value is used to determine a better decision rule for the focal player at the current stage, in the same spirit as a Bellman backup.

![](images/fa8605cb1f8fba027afe2c5799cfa748d7af6af4a10b44d4f4be3a90fb0bcc9b.jpg)  
Figure 1: The optimal value function of the game $\mathcal { M } ^ { i }$ is convex in the belief space. While valueiteration methods such as PBVI sample this space and operate on a set of vectors to represent $\underline { { v } } _ { i , t } ,$ our method works with a single vector $\underline { { v } } _ { i , t } ^ { \pi _ { t : T } ^ { i } }$ at a time, reducing the complexity of the associated algorithms.

## 3.1 CONDITIONAL OCCUPANCY STATES

Fix focal player i. At stage t, fix the public history $h _ { t } ^ { 0 }$ , the opponent private history $h _ { t } ^ { - i }$ , and the focal policy prefix $\pi _ { 0 : t } ^ { i }$ . Conditioning the central occupancy state on these variables gives the conditional occupancy state

$$
c _ { t } ^ { i } ( h _ { t } ^ { 0 } , h _ { t } ^ { - i } , \pi _ { 0 : t } ^ { i } ) \doteq \mathbb { P } _ { b _ { 0 } , p } \bigl ( S _ { t } , H _ { t } ^ { i } \mid H _ { t } ^ { 0 } = h _ { t } ^ { 0 } , H _ { t } ^ { - i } = h _ { t } ^ { - i } , \pi _ { 0 : t } ^ { i } \bigr ) .
$$

We write $c _ { t } ^ { i }$ when the histories and prefix are clear. It is a distribution over the hidden state and the focal player’s private history. If the conditioning event has probability zero, the conditional update is fixed arbitrarily.

Let $ { \mathcal { C } } _ { t } ^ { i }$ be the set of reachable conditional occupancy states at stage t. In this conditional representation, the opponent’s plan-time decision rule is written as $\pi _ { t } ^ { - i } \colon { \mathcal { C } } _ { t } ^ { i } \to A ^ { - i }$ , with space $\Pi _ { t } ^ { - i }$ . It does not denote an execution-time policy of the original zs-POSG. We use this representation for fixed-policy evaluation and improvement. For fixed-policy evaluation, pure opponent actions are sufficient: after fixing the focal player’s policy suffix, the objective is affine in the opponent’s mixed action, i.e., in its distribution over $\ r ^ { A ^ { - i } }$ . Hence, the minimum over mixed actions is attained by a pure action.

Given $c _ { t } ^ { i } , \ \pi _ { t } ^ { i } .$ , and $a _ { t } ^ { - i }$ , the model induces a probability kernel $p _ { i , t } ^ { \mathsf { c } } ( \cdot \mid c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } )$ on $\mathcal { C } _ { t + 1 } ^ { i }$ . This kernel is obtained by conditioning on the next public observation and the next opponent private observation. The conditional Bayes update and its observation likelihood are denoted by $\tau _ { i , t } ^ { \mathrm { c } }$ and $w _ { i , t } ^ { \mathsf { c } }$ , and are given in Appendix A.1. The one-step conditional payoff is $g _ { i , t } ^ { \mathrm { c } } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } )$

For a fixed focal policy suffix $\pi _ { t : T } ^ { i } .$ , these objects define the ingredients for a fixed-policy evaluation. The policy suffix induces a decision rule $\pi _ { k } ^ { i }$ at every stage $k \geq t$ . With terminal value $v _ { i , T } \equiv 0$ , the fixed-policy value function satisfies

$$
\begin{array} { r } { v _ { i , k } ^ { \pi _ { k : T } ^ { i } } ( c _ { k } ^ { i } ) = \underset { a _ { k } ^ { - i } } { \operatorname* { m i n } } \left[ g _ { i , k } ^ { \mathtt { c } } ( c _ { k } ^ { i } , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) + \sum _ { c _ { k + 1 } ^ { i } \in { \mathcal { C } } _ { k + 1 } ^ { i } } p _ { i , k } ^ { \mathtt { c } } ( c _ { k + 1 } ^ { i } \mid c _ { k } ^ { i } , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) v _ { i , k + 1 } ^ { \pi _ { k + 1 : T } ^ { i } } ( c _ { k + 1 } ^ { i } ) \right] . } \end{array}
$$

This recursion evaluates a fixed focal policy suffix by accumulating one-step conditional payoffs. It is the value function used later as the continuation term in local policy improvement.

## 3.2 THE FOCAL BELIEF GAME

We now lift the conditional occupancy states to distributions over such states. This gives the playerspecific planning state used below.

Fix a focal player i. The distribution over conditional occupancy state induced at stage t the focal belief. A focal belief $b _ { t } ^ { i } \in B _ { t } ^ { i }$ is a probability distribution over $\mathcal { C } _ { t } ^ { i }$ . Given $b _ { t } ^ { i } .$ , the focal player chooses $\pi _ { t } ^ { i }$ . The opponent’s rule is represented in the conditional coordinates by $\pi _ { t } ^ { - i }$ . The transition $\tau _ { t } ^ { i }$ is the law of the next focal belief under $( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } )$ . The reward $g _ { t } ^ { i }$ is the expected conditional payoff under the same belief and decision rules. Both are given in Appendix A.2

Definition 2. $\mathcal { M } ^ { i } \dot { = } ( \mathcal { Z } , ( \mathcal { B } _ { t } ^ { i } ) _ { t = 0 } ^ { T } , ( \Pi _ { t } ^ { i } ) _ { t = 0 } ^ { T - 1 } , ( \Pi _ { t } ^ { - i } ) _ { t = 0 } ^ { T - 1 } , ( \tau _ { t } ^ { i } ) _ { t = 0 } ^ { T - 1 } , ( g _ { t } ^ { i } ) _ { t = 0 } ^ { T - 1 } , b _ { 0 } ^ { i } , T )$ is the focal belief game associated with the original game $\mathcal { M }$

The focal belief game is fully observed at the planning level. It is not a strict lossless reduction of the original zs-POSG M. The opponent’s decision problem is represented in conditional coordinates and can therefore be viewed as a virtual player. Fixed-policy value function have the value in both representations, and the focal decision-rule space is unchanged. The problem is inherently onesided: we fix the policy of a focal player and evaluate how well this player performs against an opponent that minimizes the remaining payoff. For a fixed focal policy suffix $\pi _ { t : T } ^ { i }$ , let $\underline { { v } } _ { i , t } ^ { \pi _ { t : T } ^ { i } } ( b _ { t } ^ { i } )$ denote the value obtained from $b _ { t } ^ { i }$ when player i follows $\pi _ { t : T } ^ { i }$ and the opponent chooses actions to minimize the remaining payoff. This one-sided fixed-policy value function is the value function used by CPI.

## 3.3 STRUCTURE OF THE FOCAL BELIEF VALUE

The focal belief game reveals structural properties of the value function that are not explicit in the set-valued focal state representation, as the theorem below states. In particular, the fixed-policy value function $\underline { { v } } _ { i , t } ^ { \pi _ { t : T } ^ { i } }$ defines a linear function over the belief space $B _ { t } ^ { i }$

Theorem 1. For any fixed focal policy suffix $\pi _ { t : T } ^ { i } ,$ , the function $\underline { v } _ { i , t } ^ { \pi _ { t : T } ^ { i } }$ is linear on $B _ { t } ^ { i }$ and satisfies

$$
\begin{array} { r } { \underline { { v } } _ { i , t } ^ { \pi _ { t : T } ^ { i } } ( b _ { t } ^ { i } ) = \sum _ { c _ { t } ^ { i } \in \mathsf { s u p p } ( b _ { t } ^ { i } ) } b _ { t } ^ { i } ( c _ { t } ^ { i } ) v _ { i , t } ^ { \pi _ { t : T } ^ { i } } ( c _ { t } ^ { i } ) . } \end{array}
$$

Although beliefs $b _ { t } ^ { i }$ typically have sparse support, their representation remains intractable due to the continuous nature of the conditional state space $ { \mathcal { C } } _ { t } ^ { i }$ . Nevertheless, this complexity can be mitigated: the value function induced by a policy $\pi _ { t : T } ^ { i }$ can be exactly characterized using a reduced representation, relying only on a subset of relevant components.

Proposition 1. For any fixed focal policy $s u f f i x \pi _ { t : T } ^ { i } , v _ { i , t } ^ { \pi _ { t : T } ^ { 2 } }$ is piecewise linear and concave on $B _ { t } ^ { i }$ That is, there exists a finite collection $\Lambda _ { i , t } ^ { \pi _ { t : T } ^ { i } }$ of α-vectors $\alpha _ { i , t } ^ { \pi _ { t : T } ^ { i } } \in \mathbb { R } ^ { S _ { t } \times H _ { t } ^ { i } }$ , each defining a linear function over the ConditionalSpace space $ { \mathcal { C } } _ { t } ^ { i }$ , such that:

$$
\begin{array} { r } { v _ { i , t } ^ { \pi _ { t : T } ^ { i } } \big ( c _ { t } ^ { i } \big ) = \operatorname* { m i n } _ { \alpha _ { i , t } ^ { \pi _ { t : T } ^ { i } } \in \Lambda _ { i , t } ^ { \pi _ { t : T } ^ { i } } } \alpha _ { i , t } ^ { \pi _ { t : T } ^ { i } } \cdot c _ { t } ^ { i } . } \end{array}
$$

## 4 CONSERVATIVE POLICY ITERATION

We describe Conservative Policy Iteration (CPI) for finite-horizon zs-POSGs. CPI maintains one behavioural policy per player. At each iteration, these policies generate an occupancy trajectory after conservative averaging and GLIE exploration. For each player, it evaluates the current policy through the fixed-policy value function of Section 3. The resulting value function is used as the continuation term in a local Bellman backup. The backup returns an improved decision rule for the focal player. This rule is averaged into the current policy instead of being installed directly. CPI therefore performs local policy improvement along sampled occupancy trajectories. It does not construct the optimal value function of the focal belief game. It also does not attempt to cover the full central occupancy space.

## 4.1 LOCAL POLICY IMPROVEMENT

Fix a stage t, a focal belief $b _ { t } ^ { i } .$ , and a current focal policy suffix $\pi _ { t + 1 : T } ^ { i }$ . The fixed-policy value function $\underline { v } _ { i , t + 1 } ^ { \pi _ { t + 1 : T } ^ { i } }$ is used as the continuation value function after stage t. We write $\hat { q } _ { i , t }$ for the corresponding local Q-value, defined by $\begin{array} { r } { \hat { q } _ { i , t } ( b _ { t } ^ { i } , \pi _ { t } ) \doteq g _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ) + \underline { { v } } _ { i , t + 1 } ^ { \pi _ { t + 1 : T } ^ { \iota } } ( \tau _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ) ) } \end{array}$

Algorithm 1 Conservative policy iteration.   
1: Initialise $( \Lambda _ { i , t } ) _ { t \in \{ 0 , \dots , T \} , i \in \mathcal { Z } }$ 1: Function $\mathtt { R e c u r s e ( } o _ { t } , k \mathtt { ) }$   
2: Initialise $( \alpha _ { k } ) _ { k \geq 0 } , ( \epsilon _ { k } ) _ { k \geq 0 } , k  0$ 2: if $t < T$ then   
3: Initialise $( \bar { \pi } _ { i , t } , u _ { i , t } ) _ { t \in \{ 0 , \ldots , T - 1 \} , i \in \mathcal { T } }$ 3: $\mathtt { R e c u r s e } ( \tau _ { t } ( o _ { t } , ( \pi _ { t } ^ { 1 } , \pi _ { t } ^ { 2 } ) ) , k )$   
4: while not converged do 4: $\pi _ { t } ^ { i }  \mathtt { L P } ( \mathtt { B E L I E F } _ { i } ( o _ { t } ) , \Lambda _ { i , t + 1 } ) , \forall i \in \mathcal { I } .$   
5: Recurse $( o _ { 0 } , k )$ 5: $\tilde { \pi } _ { i , t }  \mathtt { C G L I E } ( \pi _ { t } ^ { i } , k ) , \forall i \in \mathcal { I }$   
6: $\pi _ { t } ^ { i } \gets \tilde { \pi } _ { i , t } , \forall t \in \{ 0 , . . . , T - 1 \}$ 6: $\Lambda _ { i , t }  \mathtt { V E C T O R } ( \tilde { \pi } _ { i , t } ) , \forall i \in \mathcal { T }$   
7: $k \gets k + 1$ 7: end if   
8: end while 8:   
9: 9: Function $\mathtt { B E L I E F } _ { i } ( o _ { t } ) \mathtt { \Gamma } _ { i } ( \tau _ { \mathrm { ~ \normalfont ~ ( ~ - ~ } } ) )$   
10: Function $\mathsf { C G L I E } ( \pi _ { t } ^ { i } , k ) :$ 10: return $\mathbb { P } _ { o _ { t } } ( C _ { t } ^ { i } )$   
11: $\bar { \pi } _ { i , t }  ( 1 - \alpha _ { k } ) \bar { \pi } _ { i , t } + \alpha _ { k } \pi _ { t } ^ { i }$   
12: return $( 1 - \epsilon _ { k } ) \bar { \pi } _ { i , t } + \epsilon _ { k } u _ { i , t }$

The Bellman backup at $( t , b _ { t } ^ { i } )$ induces a local zero-sum continuation game, $i . e .$ , the finite zero-sum game rooted at $( t , b _ { t } ^ { i } )$ with a fixed-policy value function. The focal player chooses a decision rule $\pi _ { t } ^ { i }$ The opponent chooses a decision rule $\pi _ { t } ^ { - i }$ on conditional states. The payoff is $\hat { q } _ { i , t } \big ( b _ { t } ^ { i } , ( \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) \big )$ We denote this local continuation game by $\mathcal { M } _ { t } ^ { i } ( \hat { q } _ { i , t } ( b _ { t } ^ { i } , \cdot ) )$ . We write $\mathcal { M } _ { t } ^ { i }$ when the continuation value is clear from context.

Definition 3. Tuple $\mathcal { M } _ { t } ^ { i } \doteq ( \mathcal { T } , \Pi _ { t } ^ { i } , \Pi _ { t } ^ { - i } , \hat { q } _ { i , t } ( b _ { t } ^ { i } , \cdot ) )$ is the finite zero-sum game with players $\mathcal { T } ,$ action spaces $\Pi _ { t } ^ { i }$ and $\Pi _ { t } ^ { - i }$ , and payoff $\hat { q } _ { i , t } ( b _ { t } ^ { i } , \cdot )$

Revisiting stage t at another belief therefore defines another local continuation game. It does not perturb the previous one. For fixed $\hat { q } _ { i , t } .$ , the local backup at $b _ { t } ^ { i }$ is $\operatorname* { m a x } _ { \pi _ { t } ^ { i } }$ min $\pi _ { \star } ^ { - i } \hat { q } _ { i , t } ( b _ { t } ^ { i } , ( \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) )$ . This is the local Bellman optimisation problem represented by $\mathcal { M } _ { t } ^ { i } ( \hat { q } _ { i , t } ( b _ { t } ^ { i } , \cdot ) )$ . A local solution is a max–min optimal focal decision rule together with an opponent minimising decision rule (see Theorem 6). Since histories, observations, and actions are finite, deterministic local decision rules are finite. The focal planner may commit to mixed decision rules. The opponent admits deterministic minimising decision rules because the local payoff is affine in its mixed rule.

## 4.2 CONSERVATIVE POLICY ITERATION

We now introduce conservative policy iteration (CPI). The algorithm alternes central occupancy expansion with local policy improvement in the two player-specific conditional representations. $\mathbf { A l - }$ gorithm 1 maintains a fixed-policy value representation, step sizes, and averaged decision rules for both players; see Lines 1–3. Each iteration starts from the initial central occupancy state $o _ { 0 }$ and calls Recurse; see Line 5. The recursion propagates the central occupancy state because the joint policy acts on the original zs-POSG. Line 3 expands then central occupancy state using the current joint decision rule. At each reached $o _ { t } ,$ , Line 10 constructs, for each player $i ,$ the focal belief induced by conditioning $o _ { t }$ in player $i \ ' s$ basis Line 4 applies the LP backup (see Theorem 6) at this focal belief and returns an improved decision rule for each player. Line 11–12 average the improved decision rule into the current decision rule and add GLIE exploration. GLIE is used to ensure that all components of the belief remain represented. Without exploration, the policy may assign zero probability to some actions, causing the induced belief to become degenerate and eliminating support from parts of the distribution. Maintaining exploration therefore prevents components of the belief from vanishing entirely, similarly to the distribution restart mechanism used in Kakade & Langford (2002). Line 6 computes the vector associated with the decision rule just computed, as established by Proposition 1.

Linear program. For a reached belief $b _ { t } ^ { i }$ , the local LP is written at the conditional level. The input $\Lambda _ { i , t + 1 }$ denotes the stored lower envelop of α-vectors representing the fixed-policy value function used for the continuation term. The LP implements the local backup in $\mathcal { M } _ { t } ^ { i } , c f .$ Appendix A.6.1. In Algorithm $1 , \mathsf { L P } ( b _ { t } ^ { i } , \lambda _ { i , t + 1 } )$ denotes this local policy-improvement routine.

Theorem 2. Let the policy updates of player i be given by

$$
\pi ^ { i , k + 1 } = ( 1 - \alpha _ { k } ) \pi ^ { i , k } + \alpha _ { k } \tilde { F } ^ { i } \left( \pi ^ { i , k } , \pi ^ { - i , k } \right) + U _ { i , k + 1 } ,
$$

where $\tilde { F } ^ { i }$ represents all the stage-wise policy optimization operator $\tilde { F } _ { t } ^ { i }$ obtained from the corresponding LP (Line 4 in Algorithm 1), and $U _ { i , k + 1 }$ is a noise term. Assume that the step sizes satisfy the Robbins-Monro conditions

$$
\sum _ { k = 0 } ^ { \infty } \alpha _ { k } = \infty , \qquad \sum _ { k = 0 } ^ { \infty } \alpha _ { k } ^ { 2 } < \infty .
$$

Under the standard stochastic approximation conditions on $U _ { i , k }$ and the approximation error between the discrete operator $\tilde { F } ^ { i }$ and the continuous dynamics $F ^ { i }$ (see Definition 4), the algorithm converges almost surely to the set ofNash equilibria.

Theorem 2 establishes the convergence of Algorithm 1, since its discrete update operator coincides with the operator defined in the algorithm. The GLIE noise introduced during exploration does not affect this convergence result, as it satisfies the conditions required by the convergence theorem.

## 5 EMPIRICAL EVALUATION

We evaluate CPI on standard finite-horizon zero-sum POSG benchmarks: Adversarial Tiger, Recycling, Mabc, and Matching Pennies. Benchmark descriptions are available at $\mathtt { h t t p } \colon / / \mathrm { m a s p l a n }$ org/. Some instances originate from common-payoff problems and are converted to the zero-sum setting by negating player 1’s payoff for player 2. We compare CPI with PBVI (Escudie et al., 2025), HSVI (Delage et al., 2023) and CFR+(Zinkevich et al., 2007; Tammelin, 2014). The CFR+ baseline adapts the extensive-form game CFR+ directly to the occupancy-state reduction of zs-POSGs; implementation details are given in Appendix D.1. In contrast, our CFR+ formulation naturally leverages the objects introduced in this paper, further strengthening the theoretical contribution of the paper. In particular, the POSG setting provides a compact representation of histories and occupancy states that can be exploited to speed up CFR. This allows CFR+ to reason directly over clusters of probabilistically equivalent private histories, rather than operating on the significantly larger extensive-form representation. All methods are evaluated on the same finite-horizon POSG instances. Table 1 reports runtime, value, and exploitability. Results for extensive-form games are discussed solely in Appendix C.

CPI vs PBVI. CPI is competitive with PBVI at small horizons and improves the runtime– exploitability trade-off at larger horizons. Indeed, PBVI grows exponentially with the horizon, as it adds an exponentially increasing number of occupancy states by construction. This difference is particularly visible in games such as Matching Pennies (T = 25, 50 in Table 1 and Figure 2), where PBVI is sensitive to the sampled belief set. For instance, insufficient or poorly distributed samples may slow convergence or degrade the quality of its central value function approximation. This sensi tivity is particularly relevant because exploitability is approximated in our experiments: PBVI must converge to sufficiently low exploitability for this approximation to provide a reliable indication of safety, so a temporary increase in measured exploitability can indicate that the policy remains exploitable. In contrast, CPI updates policies along trajectories induced by the current conservative joint policy, which can guide updates toward regions where the policies are still exploitable. From a computational perspective, PBVI applies a relatively expensive backup operator over a growing collection of sampled belief points. As the horizon increases, maintaining and updating this point set becomes prohibitive. CPI instead performs local policy updates along sampled trajectories, avoid ing the explicit construction of large point sets. In our experiments, this generally resulted in more stable scaling behaviour (Section A.7).

Finally, although CPI and CFR+ outperform PBVI for higher horizon, PBVI achieves stronger results on Adversarial Tiger and Recycling (see Figure 2). Adversarial Tiger is particularly well suited to methods that cover a broader occupancy space such as point-based value iteration algorithms. Notably, computation time increases proportionally with the horizon for CPI, whereas PBVI’s computation time may vary depending on the sampling, as observed for Recycling in Table 1.

CPI vs CFR. Both CPI and CFR-based methods improve policies iteratively, but through different update mechanisms. In our experiments, CFR iterations are computationally cheaper than CPI up dates. However, exploitability often evolves non-monotonically across iterations (see Figure 2 for Recycling and Adversarial Tiger). This finding is consistent with observations reported in (Delage et al., 2023). CPI instead performs more expensive local updates based on explicit continuation evaluations. This generally produces smoother exploitability profiles in the reported experiments (see Figure 2). A practical difference concerns exploitability evaluation. For CFR, exploitability must be estimated through an additional evaluation procedure whose cost increases with horizon. For fairness, these evaluation costs are included in the reported runtimes (see Table 1). Runtimes excluding the measurement cost are reported in Table 4. In CPI, the quantities required to estimate deviations are maintained during policy improvement steps via the fixed-policy value function. So exploitability estimates are obtained directly from the algorithm. For larger horizons, exploitability estimates in CFR also become increasingly noisy, which can make convergence trends more difficult to interpret empirically. CFR+ remains attractive because of its inexpensive iterations, whereas CPI performs fewer but more costly updates. The two approaches therefore reflect different computational trade-offs between update cost and exploitability estimation.

![](images/60277eff1a70161db96363a63005f77f2de3d01c17a49ad50f1dadda05d07df7.jpg)  
Figure 2: Evolution of value and exploitability across different games under a fixed horizon ℓ = 5. The results compare three methods: CFR+, PBVI, and CPI, highlighting their respective performance profiles in terms of convergence behaviour and solution quality across the considered benchmarks.

Table 1: Snapshot of the empirical results, including the computational cost of exploitability evaluation. For each game, horizon and algorithm, we report time, value and exploitability ϵ associated with the computed ϵ-Nash equilibrium. OOT occurs when the target exploitability threshold cannot be reached within the prescribed time limit, while OOM means that the algorithms ran out of memory. magenta = lowest exploitability/value per benchmark
<table><tr><td rowspan="2">Problem</td><td rowspan="2">T</td><td colspan="2">CPI (Ours)</td><td colspan="2">PBVI</td><td colspan="3">HSVI</td><td colspan="3">CFR+</td></tr><tr><td>t</td><td>val. €</td><td>t</td><td>val. €</td><td></td><td>t</td><td>val. ∈</td><td></td><td>t</td><td>val. €</td></tr><tr><td></td><td>2</td><td>0.3</td><td>0.078 0.</td><td>0.2</td><td>0.078 0.</td><td></td><td>0.4</td><td>0.078 0.</td><td></td><td>0.3</td><td>0.078 0.</td></tr><tr><td rowspan="7">MABC</td><td>3</td><td>11</td><td>0.097 0.</td><td>24</td><td>0.096 0.</td><td></td><td>70</td><td>0.097 0.</td><td></td><td>12</td><td>0.095 0.</td></tr><tr><td>4</td><td>23</td><td>0.11 0.</td><td>65</td><td>0.11 0.01</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>49</td><td>0.11 0.01</td></tr><tr><td>5</td><td>29</td><td>0.12 0.</td><td>31</td><td>0.12 0.</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>82</td><td>0.12 0.01</td></tr><tr><td>7</td><td>97</td><td>0.14 0.01</td><td>190</td><td>0.14 0.005</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>292</td><td>0.14 0.02</td></tr><tr><td>10</td><td>100</td><td>0.18 0.01</td><td>637</td><td>0.17 0.01</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>380</td><td>0.18 0.02</td></tr><tr><td>100</td><td>1920</td><td>1.06 0.03</td><td>OOT</td><td>OOT OOT</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>10042</td><td>1.06 0.2</td></tr><tr><td>2</td><td>0.01</td><td>0.4 0.</td><td>2</td><td>0.39 0.</td><td></td><td>0.1</td><td></td><td></td><td>0.06</td><td></td></tr><tr><td rowspan="8">MATCHING PENNIES</td><td>3</td><td>0.02</td><td>0.6 0.</td><td>18</td><td>0.6 0.</td><td></td><td>1</td><td>0.4 0. 0.6 0.</td><td></td><td>0.1</td><td>0.4 0.005 0.6 0.001</td></tr><tr><td>4</td><td>0.04</td><td>0.8 0.</td><td>34</td><td>0.77 0.</td><td></td><td>4</td><td>0.8 0.</td><td></td><td>0.2</td><td>0.8 0.</td></tr><tr><td>5</td><td>0.04</td><td>1.0.</td><td>52</td><td>0.95 0.</td><td></td><td>60</td><td>1.0.</td><td></td><td>0.25</td><td>1.0.</td></tr><tr><td>7</td><td>13</td><td>1.4 0.</td><td>180</td><td>1.31 0.03</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>15</td><td>1.39 0.01</td></tr><tr><td>10</td><td>92</td><td>2.0.</td><td>OOT</td><td>OOT OOT</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>103</td><td>1.99 0.02</td></tr><tr><td>25</td><td>15</td><td>5.0.</td><td>OOT</td><td>OOT OOT</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>132</td><td>5.0.02</td></tr><tr><td>50</td><td>217</td><td>10.0.</td><td>OOT</td><td>OOT OOT</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>718</td><td>10.0.03</td></tr><tr><td>2</td><td>0.4</td><td>0.26 0.</td><td>0.4</td><td></td><td></td><td>1.2</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">RECYCLING</td><td>3</td><td>17</td><td>0.32 0.01</td><td>78</td><td>0.26 0. 0.32 0.</td><td></td><td></td><td>0.26 0.</td><td></td><td>2.8 75</td><td>0.26 0.004 0.32 0.01</td></tr><tr><td>4</td><td>42</td><td>0.36 0.03</td><td>56</td><td></td><td></td><td>120</td><td>0.32 0.</td><td></td><td></td><td></td></tr><tr><td>5</td><td>213</td><td>0.4 0.05</td><td>98</td><td>0.35 0.01</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>139</td><td>0.35 0.04</td></tr><tr><td>7</td><td>271</td><td>0.5 0.06</td><td>1481</td><td>0.38 0.05</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>239</td><td>0.38 0.06</td></tr><tr><td>10</td><td>505</td><td>0.63 0.07</td><td>3232</td><td>0.48 0.04</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>624</td><td>0.46 0.12</td></tr><tr><td>100</td><td>4180</td><td>4.9 0.68</td><td>OOT</td><td></td><td>0.6 0.06</td><td>OOT OOT</td><td>OOT OOT</td><td></td><td>1249</td><td>0.63 0.15</td></tr><tr><td></td><td></td><td></td><td></td><td>OOT OOT</td><td></td><td></td><td>OOT OOT</td><td></td><td>13087</td><td>4.96 2</td></tr><tr><td rowspan="6">ADVERSARIAL TIGER</td><td>2</td><td>0.1</td><td>-0.4 0.</td><td>0.1</td><td>-0.4 0.</td><td></td><td>0.2</td><td>-0.4 0.</td><td></td><td>0.1</td><td>-0.4 0.0</td></tr><tr><td>3</td><td>0.2</td><td>-0.56 0.</td><td>0.3</td><td>-0.56 0.</td><td></td><td>500</td><td>-0.56 0.</td><td></td><td>5</td><td>-0.56 0.</td></tr><tr><td>4</td><td>16</td><td>-0.75 0.01</td><td>46</td><td>-0.76 0.02</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>76</td><td>-0.75 0.03</td></tr><tr><td>5</td><td>82</td><td>-0.96 0.05</td><td>141</td><td>-0.96 0.05</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>91</td><td>-0.95 0.05</td></tr><tr><td>7</td><td>248</td><td>-1.36 0.13</td><td>866</td><td>-1.36 0.1</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>459</td><td>-1.35 0.13</td></tr><tr><td>10</td><td>824</td><td>-1.970.24</td><td>956</td><td>-1.98 0.18</td><td></td><td>OOT</td><td>OOT OOT</td><td></td><td>3901</td><td>-1.95 0.3</td></tr></table>

Takehome message. CFR+ benefits from inexpensive iterations, but typically requires more iterations and repeated exploitability evaluations to reach solution quality comparable to CPI. In our experiments, this often resulted in higher overall runtime at larger horizons. CPI instead performs fewer but more expensive updates, leading to a different trade-off between update cost, exploitability estimation, and empirical stability.

## 6 CONCLUSION

We introduced an approximate policy-iteration algorithm for finite-horizon zs-POSGs. The method combines elements from point-based value iteration and policy-search approaches. It maintains one policy per player and updates both policies along occupancy trajectories induced by conservative policy updates. Instead of approximating a global central value function, each update relies on a fixed-policy continuation value function associated with the current joint policy. The algorithm combines local policy improvements, conservative updates, and persistent exploration. The empirical results show that this approach is competitive on standard finite-horizon zs-POSG benchmarks. In several instances and horizons, it improves the runtime–exploitability trade-off compared with PBVI, HSVI, and CFR+.

Several limitations remain. First, the algorithm requires solving local optimisation problems for both players throughout training. Second, convergence can remain slow and the sample complexity high, which may limit scalability at larger horizons. Third, the current implementation still relies on linear programming subroutines, which may become computationally demanding in larger games. Future work will investigate iterative approximations of these optimisation steps, improved exploration and scheduling mechanisms, and neural approximations of the continuation value functions.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with translation, provide critical ingredients for proving mathematical claims, assist in the development and refinement of mathematical proofs, edit the manuscript to improve readability, create and modify scientific figures, create and edit software code, summarize and analyze existing literature, and assist with sourcing and searching for relevant information.

We did not use generative AI tools to develop theoretical models or conceptual frameworks, formulate mathematical claims, propose or refine hypotheses, design research methodology or experiments, provide feedback on research methodology or experiments, implement the methods underlying the research, or interpret the results. The remaining categories of required disclosure are not applicable to this work.

All AI-assisted content and outputs were reviewed by the authors. The authors take full responsibility for the final content of this work, including all text, mathematical claims, proofs, figures, code, and other artifacts produced with the assistance of generative AI.

## REFERENCES

Karl J. Aström. Optimal control of Markov decision processes with incomplete state estimation. Journal ofMathematical Analysis and Applications, 10:174–205, 1965.

Michel Benaïm, Josef Hofbauer, and Sylvain Sorin. Stochastic approximations and differential inclusions. SIAM Journal on Control and Optimization, 44(1):328–348, 2005. doi: 10.1137/ S0363012904439301. URL https://doi.org/10.1137/S0363012904439301.

Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-dynamic programming., volume 3 of Optimization and neural computation series. Athena Scientific, 1996. ISBN 1886529108.

Noam Brown, Anton Bakhtin, Adam Lerer, and Qucheng Gong. Combining deep reinforcement learning and search for imperfect-information games. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Olivier Buffet, Jilles Steeve Dibangoye, Abdallah Safidine, and Vincent Thomas. ϵ-optimally solving zs-POSGs using Bellman’s optimality principle. Technical report, Inria Nancy Grand-Est, 2020.

Rafael F. Cunha, Jacopo Castellini, Johan Peralez, and Jilles Steeve Dibangoye. On convex optimal value functions for POSGs. arXiv preprint arXiv:2311.09459, 2023.

Aurélien Delage, Olivier Buffet, Jilles Steeve Dibangoye, and Abdallah Saffidine. HSVI can solve zero-sum partially observable stochastic games. Dynamic Games and Applications, pp. 1–55, 2023.

Jilles Steeve Dibangoye and Olivier Buffet. Learning to act in decentralized partially observable MDPs. In Proceedings of the 35th International Conference on Machine Learning (ICML), pp. 1233–1242, 2018.

Jilles Steeve Dibangoye, Christopher Amato, and Arnaud Doniec. Scaling up decentralized MDPs through heuristic search. In Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence (UAI), pp. 217–226, 2012.

Jilles Steeve Dibangoye, Christopher Amato, Olivier Buffet, and François Charpillet. Optimally solving Dec-POMDPs as continuous-state MDPs. In Proceedings ofthe 23rd International Joint Conference on Artificial Intelligence (IJCAI), pp. 90–96, 2013.

Jilles Steeve Dibangoye, Christopher Amato, Olivier Buffet, and François Charpillet. Exploiting separability in multiagent planning with continuous-state MDPs. In Proceedings of the 13th International Conference on Autonomous Agents and Multiagent Systems (AAMAS). ACM, 2014.

Jilles Steeve Dibangoye, Christopher Amato, Olivier Buffet, and François Charpillet. Optimally solving Dec-POMDPs as continuous-state MDPs. Journal of Artificial Intelligence Research, 55: 443–497, 2016a.

Jilles Steeve Dibangoye, Christopher Amato, Olivier Buffet, and François Charpillet. Optimally solving Dec-POMDPs as continuous-state MDPs. Journal of Artificial Intelligence Research, 2016b.

Erwan C. Escudie, Matthia Sabatelli, Olivier Buffet, and Jilles Steeve Dibangoye. ϵ-optimally solving two-player zero-sum posgs. The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

J. Filar and K. Vrieze. Competitive Markov Decision Processes. Springer, 1997.

Eric A. Hansen, Daniel S. Bernstein, and Shlomo Zilberstein. Dynamic programming for partially observable stochastic games. In Proceedings of the 19th National Conference on Artificial Intelligence (AAAI), 2004.

Karel Horák and Branislav Bošanský. Solving partially observable stochastic games with public observations. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2019.

Karel Horák, Branislav Bošanský, and Michal Pechouˇ cek. Heuristic search value iteration for one-ˇ sided partially observable stochastic games. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 31, 2017.

Sham M. Kakade and John Langford. Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning, 2002.

Dhruva Kartik and Ashutosh Nayyar. Zero-sum stochastic games with asymmetric information. In 2019 IEEE 58th Conference on Decision and Control (CDC), pp. 4061–4066, 2019. doi: 10.1109/CDC40024.2019.9030177.

Daphne Koller, Nimrod Megiddo, and Bernhard von Stengel. Efficient computation of equilibria for extensive two-person games. Games and Economic Behavior, 14(2):247–259, June 1996.

Vojtech Kovaˇ ˇrík, Martin Schmid, Neil Burch, Michael Bowling, and Viliam Lisý. Rethinking formal models of partially observable multiagent decision making. Artificial Intelligence, 303:103645, 2022. ISSN 0004-3702. doi: https://doi.org/10.1016/j.artint.2021.103645.

H. W. Kuhn. Extensive Games and the Problem ofInformation. Princeton University Press, Princeton, NJ, 1953.

Adam Lerer, Hengyuan Hu, Jakob Foerster, and Noam Brown. Improving policies via search in cooperative partially observable games. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 7187–7194, 2020.

Matej Moravcík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevorˇ Davis, Kevin Waugh, Michael B. Johanson, and Michael H. Bowling. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356:508–513, 2017.

Rémi Munos. Error bounds for approximate policy iteration. In Tom Fawcett and Nina Mishra (eds.), Machine Learning, Proceedings of the Twentieth International Conference (ICML 2003), August 21-24, 2003, Washington, DC, USA, pp. 560–567. AAAI Press, 2003.

Ashutosh Nayyar, Aditya Mahajan, and Demosthenis Teneketzis. Decentralized stochastic control with partial history sharing: A common information approach. IEEE Transactions on Automatic Control, 58(7):1644–1658, 2013.

T. Neller and Marc Lanctot. An introduction to counterfactual regret minimization. 2013.

J. von Neumann. Zur theorie der gesellschaftsspiele. Mathematische Annalen, 100:295–320, 1928.

Frans Adriaan Oliehoek. Sufficient plan-time statistics for decentralized POMDPs. In Proceedings ofthe 23rd International Joint Conference on Artificial Intelligence (IJCAI), 2013.

Johan Peralez, Aurélien Delage, Olivier Buffet, and Jilles Steeve Dibangoye. Solving hierarchical information-sharing dec-POMDPs: An extensive-form game approach. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

Johan Peralez, Aurélien Delage, Jacopo Castellini, Rafael F. Cunha, and Jilles Steeve Dibangoye. Optimally solving simultaneous-move dec-POMDPs: The sequential central planning approach. Proceedings of the AAAI Conference on Artificial Intelligence, 38(8):9411–9419, 2025.

Fabian Raoul Pieroth, Katherine Fitch, and Lenz Belzner. Detecting influence structures in multiagent reinforcement learning. In Proceedings of the 41st International Conference on Machine Learning (ICML), pp. 40740–40761. PMLR, 2024.

Joelle Pineau, Geoff Gordon, Sebastian Thrun, et al. Point-based value iteration: An anytime algorithm for POMDPs. In Proceedings of the 18th International Joint Conference on Artificial Intelligence (IJCAI), volume 3, pp. 1025–1032, 2003.

Julia Robinson. An iterative method of solving a game. Annals of Mathematics, 54(2):296–301, 1951. ISSN 0003486X, 19398980.

John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, and Pieter Abbeel. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning - Volume 37, ICML’15, pp. 1889–1897. JMLR.org, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Guy Shani, Joelle Pineau, and Robert Kaplow. A survey of point-based pomdp solvers. Autonomous Agents and Multi-Agent Systems, 27(1):1–51, 2013.

L. S. Shapley. Stochastic games\*. Proceedings ofthe National Academy ofSciences, 39(10):1095– 1100, 1953.

Richard D. Smallwood and Edward J. Sondik. The optimal control of partially observable Markov decision processes over a finite horizon. Operations Research, 21(5):1071–1088, 1973.

Trey Smith and Reid G. Simmons. Heuristic search value iteration for POMDPs. In Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI-2004), pp. 520–527. AUAI Press, 2004.

Samuel Sokota, Ryan D’Orazio, Chun Kai Ling, David J. Wu, J. Zico Kolter, and Noam Brown. Abstracting imperfect information away from two-player zero-sum games. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pp. 32169–32193, 2023.

Edward J. Sondik. The optimal control of partially observable Markov decision processes over the infinite horizon: Discounted cost. Operations Research, 12:282–304, 1978.

Oskari Tammelin. Solving large imperfect information games using CFR+. CoRR, 2014.

Elise Van der Pol and Frans A. Oliehoek. Coordinated deep reinforcement learners for traffic light control. In NeurIPS Workshop on Learning, Inference and Control ofMulti-Agent Systems, 2016.

Hua Wei, Chacha Chen, Guanjie Zheng, Kan Wu, Vikash Gayah, Kai Xu, and Zhenhui Li. Presslight: Learning max pressure control to coordinate traffic signals in arterial network. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1290–1298, 2019.

Auke J. Wiggers, Frans A. Oliehoek, and Diederik M. Roijers. Structure in the value function of two-player zero-sum games of incomplete information. In Proceedings of the 22nd European Conference on Artificial Intelligence (ECAI), 2016.

Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. Advances in neural information processing systems, 35:24611–24624, 2022.

Yifan Zhong, Jakub Grudzien Kuba, Xidong Feng, Siyi Hu, Jiaming Ji, and Yaodong Yang. Heterogeneous-agent reinforcement learning. J. Mach. Learn. Res., 25(1), January 2024. ISSN 1532-4435.

Martin A. Zinkevich, Michael Bradley Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. In Neural Information Processing Systems, 2007.

## A PROOFS

Lemma 1. In afinite-horizonfinite model, all behavioural-policy spaces are compact, and allpayoff maps are continuous.

Proof. Each Π<sup>i</sup> is a finite product of finite-dimensional simplexes. Each $\Pi ^ { i }$ and each $\Pi _ { t : T }$ is a finite product of decision-rule spaces. Finite products of compact sets are compact. Every finitehorizon payoff is a finite sum of multilinear functions of behavioural-policy coordinates. Hence every finite-horizon payoff map is continuous. The extreme-value theorem gives attainment for every optimisation over these spaces. □

## A.1 CONDITIONAL KERNELS

If a conditioning event has zero probability, $\tau _ { i , t } ^ { \mathrm { c } } \colon \mathcal { C } _ { t } ^ { i } \times \Pi _ { t } ^ { i } \times \mathcal { A } ^ { - i } \times \mathcal { Z } ^ { 0 } \times \mathcal { Z } ^ { - i } \to \mathcal { C } _ { t + 1 } ^ { i }$ is assigned an arbitrary fixed element of $\mathcal { C } _ { t + 1 } ^ { i } .$ . All such terms are multiplied by their zero likelihood.

Lemma 2. $\tau _ { i , t } ^ { \mathrm { c } }$ is the Bayes update of $( S _ { t + 1 } , H _ { t + 1 } ^ { i } )$ given $( Z _ { t + 1 } ^ { 0 } , Z _ { t + 1 } ^ { - i } )$

Proof. Fix $( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } )$

For $( s _ { t + 1 } , h _ { t + 1 } ^ { i } )$ , define

$$
\begin{array} { r l } & { m ( s _ { t + 1 } , h _ { t + 1 } ^ { i } ) \doteq \sum _ { s _ { t } , h _ { t } ^ { i } , a _ { t } ^ { i } , z _ { t + 1 } ^ { i } } c _ { t } ^ { i } ( s _ { t } , h _ { t } ^ { i } ) \pi _ { t } ^ { i } ( a _ { t } ^ { i } \mid h _ { t } ^ { i } ) } \\ & { \qquad \cdot p ( s _ { t + 1 } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { i } , z _ { t + 1 } ^ { - i } \mid s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } ) \cdot \mathbf { 1 } \{ h _ { t + 1 } ^ { i } = ( h _ { t } ^ { i } , a _ { t } ^ { i } , z _ { t + 1 } ^ { i } ) \} . } \end{array}
$$

Let $\begin{array} { r } { W \doteq \sum _ { s _ { t + 1 } , h _ { t + 1 } ^ { i } } m ( s _ { t + 1 } , h _ { t + 1 } ^ { i } ) } \end{array}$

If $W > 0$ , then

$$
\tau _ { i , t } ^ { \mathrm { c } } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } ) ( s _ { t + 1 } , h _ { t + 1 } ^ { i } ) = \frac { m ( s _ { t + 1 } , h _ { t + 1 } ^ { i } ) } { W } .
$$

If $W = 0 ;$ , every weighted term involving this update is zero.

Lemma 3. $w _ { i , t } ^ { \mathrm { c } } \colon \mathcal { C } _ { t } ^ { i } \times \Pi _ { t } ^ { i } \times \mathcal { A } ^ { - i }  \Delta ( \mathcal { Z } ^ { 0 } \times \mathcal { Z } ^ { - i } )$ is a probability law over $\mathcal Z ^ { 0 } \times \mathcal Z ^ { - i }$

Proof. Fix $( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } )$

With m as in Lemma 2, define

$$
w _ { i , t } ^ { \mathrm { c } } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } ) \doteq \sum _ { s _ { t + 1 } , h _ { t + 1 } ^ { i } } m ( s _ { t + 1 } , h _ { t + 1 } ^ { i } ) .
$$

Then $w _ { i , t } ^ { \mathrm { c } } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } ) \geq 0 ,$ , and

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } } w _ { i , t } ^ { c } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } ) } \\ & { = \displaystyle \sum _ { s _ { t } , h _ { t } ^ { i } } c _ { t } ^ { i } ( s _ { t } , h _ { t } ^ { i } ) \sum _ { a _ { t } ^ { i } } \pi _ { t } ^ { i } ( a _ { t } ^ { i } \mid h _ { t } ^ { i } ) \cdot \sum _ { \scriptstyle s _ { t + 1 } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { i } , z _ { t + 1 } ^ { - i } } p ( s _ { t + 1 } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { i } , z _ { t + 1 } ^ { - i } \mid s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } ) } \\ & { = \displaystyle \sum _ { s _ { t } , h _ { t } ^ { i } } c _ { t } ^ { i } ( s _ { t } , h _ { t } ^ { i } ) \sum _ { a _ { t } ^ { i } } \pi _ { t } ^ { i } ( a _ { t } ^ { i } \mid h _ { t } ^ { i } ) = 1 . } \end{array}
$$

Lemma 4. $g _ { i , t } ^ { \mathrm { c } } \colon \mathcal { C } _ { t } ^ { i } \times \Pi _ { t } ^ { i } \times \mathcal { A } ^ { - i }  \mathbb { R }$ is the conditional expectation of $\cdot r ^ { i } ( S _ { t } , A _ { t } ^ { i } , a _ { t } ^ { - i } )$ .

Proof. Fix $( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } )$

$$
\begin{array} { l } { \mathbb { E } [ r ^ { i } ( S _ { t } , A _ { t } ^ { i } , a _ { t } ^ { - i } ) \mid c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } ] = \displaystyle \sum _ { s _ { t } , h _ { t } ^ { i } } c _ { t } ^ { i } ( s _ { t } , h _ { t } ^ { i } ) \sum _ { a _ { t } ^ { i } } \pi _ { t } ^ { i } ( a _ { t } ^ { i } \mid h _ { t } ^ { i } ) r ^ { i } ( s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } ) } \\ { = g _ { i , t } ^ { \mathrm { c } } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } ) . } \end{array}
$$

Lemma 5. $p _ { i , t } ^ { \mathrm { c } } \colon \mathcal { C } _ { t } ^ { i } \times \Pi _ { t } ^ { i } \times A ^ { - i } \to \Delta ( \mathcal { C } _ { t + 1 } ^ { i } )$ is the pushforward of $w _ { i , t } ^ { \mathrm { c } }$ <sub>t</sub> by $\tau _ { i , t } ^ { \mathrm { c } }$

Proof. Fix $( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } )$ . For every $c _ { t + 1 } ^ { i } \in \mathcal { C } _ { t + 1 } ^ { i }$

$$
\begin{array} { r l } & { \mathbb { P } ( C _ { t + 1 } ^ { i } = c _ { t + 1 } ^ { i } \mid c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } ) } \\ & { = \displaystyle \sum _ { z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } } w _ { i , t } ^ { \circ } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } ) \cdot \mathbf { 1 } \{ c _ { t + 1 } ^ { i } = \tau _ { i , t } ^ { \circ } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t + 1 } ^ { 0 } , z _ { t + 1 } ^ { - i } ) \} } \\ & { \quad = p _ { i , t } ^ { \circ } ( c _ { t + 1 } ^ { i } \mid c _ { t } ^ { i } , \pi _ { t } ^ { i } , a _ { t } ^ { - i } ) . } \end{array}
$$

For fixed $\pi _ { t : T } ^ { i }$ , the recursion below defines $v _ { i , k } ^ { \pi _ { t : T } ^ { i } } : \mathcal { C } _ { k } ^ { i } \to \mathbb { R }$ on $\mathcal { C } _ { k } ^ { i }$ for every $k \in \{ t , \ldots , T \}$ . Define

$$
\begin{array} { r } { v _ { i , k } ^ { \pi _ { t : T } ^ { \iota } } \colon c _ { k } ^ { i } \mapsto \operatorname* { m i n } _ { \pi _ { t : T } ^ { - i } } \mathbb { E } _ { c _ { k } ^ { i } , \pi _ { t : T } ^ { i } } [ \sum _ { \kappa = k } ^ { T - 1 } r ^ { i } ( S _ { \kappa } , A _ { \kappa } ^ { i } , A _ { \kappa } ^ { - i } ) ] . } \end{array}
$$

Theorem 3. Forfixed $\pi _ { t : T } ^ { i } , v _ { i , k } ^ { \pi _ { t : T } ^ { 2 } }$ satisfies the conditional Bellman recursionfor every $k \geq t .$

Proof. $\mathrm { A t } k = T$ , set $v _ { i , T } ^ { \pi _ { t : T } ^ { i } } \equiv 0$

Fix $k < T , c _ { k } ^ { i } \in \mathcal { C } _ { k } ^ { i }$ , and the rule $\pi _ { k } ^ { i }$ induced by $\pi _ { t : T } ^ { i }$

For every $a _ { k } ^ { - i } \in \mathcal { A } ^ { - i }$ , Lemmas 4 and 5 give

$$
\begin{array} { r l } & { \mathbb { E } [ r ^ { i } ( S _ { k } , A _ { k } ^ { i } , a _ { k } ^ { - i } ) + v _ { i , k + 1 } ^ { \pi _ { t : T } ^ { i } } ( C _ { k + 1 } ^ { i } ) \mid c _ { k } ^ { i } , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ] } \\ & { = g _ { i , k } ^ { \mathtt { c } } ( c _ { k } ^ { i } , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) + \displaystyle \sum _ { \substack { c _ { k + 1 } ^ { i } \in \mathrm { s u p p } ( p _ { i , k } ^ { \mathtt { c } } ( \cdot \vert c _ { k } ^ { i } , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) ) } } p _ { i , k } ^ { \mathtt { c } } ( c _ { k + 1 } ^ { i } \mid c _ { k } ^ { i } , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) v _ { i , k + 1 } ^ { \pi _ { t : T } ^ { i } } ( c _ { k + 1 } ^ { i } ) . } \end{array}
$$

Taking $\mathrm { m i n } _ { a _ { k } ^ { - i } \in \mathcal { A } ^ { - i } } \textbf { g }$ ives the recursion.

## A.2 BELIEF DYNAMICS

Define $g _ { t } ^ { i } \colon ( b _ { t } ^ { i } , \pi _ { t } ) \mapsto \mathbb { E } _ { b _ { t } ^ { i } , \pi _ { t } } [ r ^ { i } ( S _ { t } , A _ { t } ) ]$

Lemma 6. $g _ { t } ^ { i }$ is the expectation of $\cdot _ { g _ { i , t } ^ { \mathrm { c } } }$ under b.

Proof. Fix $b \in B _ { t } ^ { i }$ and $\pi _ { t } = ( \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } )$

By definition,

$$
\begin{array} { r l } & { g _ { t } ^ { i } ( b , \pi _ { t } ) \doteq \mathbb { E } _ { b _ { t } ^ { i } , \pi _ { t } } [ r ^ { i } ( S _ { t } , A _ { t } ) ] } \\ & { \quad \quad \quad = \mathbb { E } _ { b _ { t } ^ { i } , \pi _ { t } } [ \mathbb { E } _ { C _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ( C _ { t } ^ { i } ) } [ r ^ { i } ( S _ { t } , A _ { t } ) ] ] } \\ & { \quad \quad \quad = \displaystyle \sum _ { c _ { t } ^ { i } \in \mathcal { C } _ { t } ^ { i } } b ( c _ { t } ^ { i } ) g _ { i , t } ^ { \tt c } ( c _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ( c _ { t } ^ { i } ) ) . } \end{array}
$$

Lemma 7. $\tau _ { t } ^ { i }$ is the law of $\cdot C _ { t + 1 } ^ { i }$ under $( b , \pi _ { t } )$

Proof. Fix $b \in B _ { t } ^ { i }$ and $\pi _ { t } = ( \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } )$

For every $c _ { t + 1 } ^ { i } \in \mathcal { C } _ { t + 1 } ^ { i }$

$$
\begin{array} { r l } & { \mathbb { P } _ { b , \pi _ { t } } ( C _ { t + 1 } ^ { i } = c _ { t + 1 } ^ { i } ) } \\ & { \ = \displaystyle \sum _ { c _ { t } ^ { i } \in \mathcal { C } _ { t } ^ { i } } b ( c _ { t } ^ { i } ) p _ { i , t } ^ { \mathrm { c } } ( c _ { t + 1 } ^ { i } \mid c _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ( c _ { t } ^ { i } ) ) } \\ & { \ = \tau _ { t } ^ { i } ( b , \pi _ { t } ) ( c _ { t + 1 } ^ { i } ) . } \end{array}
$$

Lemma 8. For separable payoffs, the minimisation over $\pi _ { t } ^ { - i }$ decomposes pointwise.

Proof. Let $F \colon \mathcal { C } _ { t } ^ { i } \times \mathcal { A } ^ { - i }  \mathbb { R }$

For every π<sup>−</sup><sub>t</sub> $\pi _ { t } ^ { - i }$

$$
\sum _ { c _ { t } ^ { i } } b ( c _ { t } ^ { i } ) F ( c _ { t } ^ { i } , \pi _ { t } ^ { - i } ( c _ { t } ^ { i } ) ) \geq \sum _ { c _ { t } ^ { i } } b ( c _ { t } ^ { i } ) \operatorname* { m i n } _ { a _ { t } ^ { - i } } F ( c _ { t } ^ { i } , a _ { t } ^ { - i } ) .
$$

Choose

$$
\pi _ { t , * } ^ { - i } ( c _ { t } ^ { i } ) \in \arg \operatorname* { m i n } _ { a _ { t } ^ { - i } \in A ^ { - i } } F ( c _ { t } ^ { i } , a _ { t } ^ { - i } ) .
$$

Then

$$
\sum _ { c _ { t } ^ { i } } b ( c _ { t } ^ { i } ) F ( c _ { t } ^ { i } , \pi _ { t , * } ^ { - i } ( c _ { t } ^ { i } ) ) = \sum _ { c _ { t } ^ { i } } b ( c _ { t } ^ { i } ) \operatorname* { m i n } _ { a _ { t } ^ { - i } } F ( c _ { t } ^ { i } , a _ { t } ^ { - i } ) .
$$

## A.3 CONTINUATION VALUES

Define

$$
v _ { i , t } ^ { \mathrm { c o n t } } ( b ) \doteq \operatorname* { m a x } _ { \pi _ { t : T } ^ { i } } \operatorname* { m i n } _ { \pi _ { t : T } ^ { - i } } \mathbb { E } _ { b , \pi _ { t : T } ^ { i } , \pi _ { t : T } ^ { - i } } \left[ \sum _ { k = t } ^ { T - 1 } r ^ { i } ( S _ { k } , A _ { k } ) \right] .
$$

Lemma 9. $v _ { i , t } ^ { \mathrm { c o n t } }$ satisfies thefocal beliefBellman recursion.

Proof. $\mathrm { A t } \ i = T , v _ { i , T } ^ { \mathrm { c o n t } } = 0 .$

Fix $t < T$ . Rearranging max and min operators require replacing opponent deterministic policy suffix $\pi _ { t } ^ { - i }$ by a mixed one. For any $( \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } )$ , Lemmas 6 and 7 give

$$
\begin{array} { r l } & { \mathbb { E } _ { b , \pi _ { t } , \pi _ { t + 1 : T } ^ { i } , \pi _ { t + 1 : T } ^ { - i } } \left[ \displaystyle \sum _ { k = t } ^ { T - 1 } r ^ { i } ( S _ { k } , A _ { k } ) \right] } \\ & { = g _ { t } ^ { i } ( b , \pi _ { t } ) + \mathbb { E } _ { \tau _ { t } ^ { i } ( b , \pi _ { t } ) , \pi _ { t + 1 : T } ^ { i } , \pi _ { t + 1 : T } ^ { - i } } \left[ \displaystyle \sum _ { k = t + 1 } ^ { T - 1 } r ^ { i } ( S _ { k } , A _ { k } ) \right] . } \end{array}
$$

Decomposing the policy space and optimising gives

$$
v _ { i , t } ^ { \mathrm { c o n t } } ( b ) = \operatorname* { m a x } _ { \pi _ { t } ^ { i } } \operatorname* { m i n } _ { \pi _ { t } ^ { - i } } \left[ g _ { t } ^ { i } ( b , \pi _ { t } ) + v _ { i , t + 1 } ^ { \mathrm { c o n t } } ( \tau _ { t } ^ { i } ( b , \pi _ { t } ) ) \right] .
$$

where the interchange of max and min is established using arguments similar to those in Delage et al. (2023) and Escudie et al. (2025). □

Theorem 4. The optimal focal belief value function $\underline { { v } } _ { t } ^ { i , * }$ satisfies the focal belief Bellman equation and equals $v _ { i , t } ^ { \mathrm { c o n t } }$

Proof. Both functions have terminal value zero.

Assume $\underline { { v } } _ { t + 1 } ^ { i , * } = v _ { i , t + 1 } ^ { \mathrm { c o n t } }$ . By Lemma $^ { 9 , }$

$$
v _ { i , t } ^ { \mathrm { c o n t } } ( b ) = \operatorname* { m a x } _ { \pi _ { t } ^ { i } } \operatorname* { m i n } _ { \pi _ { t } ^ { - i } } \left[ g _ { t } ^ { i } ( b , \pi _ { t } ) + { \underline { { v } } } _ { t + 1 } ^ { i , * } ( \tau _ { t } ^ { i } ( b , \pi _ { t } ) ) \right] .
$$

The right-hand side is the Bellman definition of $\underline { { v } } _ { t } ^ { i , * } ( b )$ . Backward induction gives the claim.

## A.4 FOCAL DECOMPOSITION

Lemma 10. A prefix pair $( \pi _ { 0 : t } ^ { i } , \pi _ { 0 : t } ^ { - i } )$ induces $b _ { k + 1 } ^ { i } = \tau _ { k } ^ { i } ( b _ { k } ^ { i } , \pi _ { k } )$

Proof. $\mathrm { A t } \ k = 0 , b _ { 0 } ^ { i }$ is induced by $b _ { 0 }$ . If $b _ { k } ^ { i }$ is the law of $C _ { k } ^ { i }$ , then Lemma 7 gives

$$
b _ { k + 1 } ^ { i } = \tau _ { k } ^ { i } ( b _ { k } ^ { i } , \pi _ { k } ) .
$$

## A.5 PROOF OF THEOREM 1

For fixed $\pi _ { t : T } ^ { i } .$ , define

$$
v _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( b ) \doteq \operatorname* { m i n } _ { \pi _ { k : T } ^ { - i } } \mathbb { E } _ { b , \pi _ { k : T } ^ { i } , \pi _ { k : T } ^ { - i } } \left[ \sum _ { \ell = k } ^ { T - 1 } r ^ { i } ( S _ { \ell } , A _ { \ell } ) \right] ,
$$

where $\pi _ { k : T } ^ { i }$ is the tail of $\pi _ { t : T } ^ { i }$

Lemma 11. For fixed $\pi _ { t : T } ^ { i }$ and $k \geq t , \underline { { v } } _ { i , k } ^ { \pi _ { t : T } ^ { i } }$ is linear on $B _ { k } ^ { i }$

Proof. For $c \in \mathcal { C } _ { k } ^ { i }$ , set

$$
w _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( c ) \doteq v _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( c ) .
$$

We prove by backward induction that

$$
\underline { { v } } _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( b ) = \sum _ { c \in \mathcal { C } _ { k } ^ { i } } b ( c ) w _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( c ) .
$$

The claim is zero at $k = T$

Assume it holds at $k + 1$ . The stage-k focal rule $\pi _ { k } ^ { i }$ is fixed. By Lemmas 6, 7, and the induction hypothesis,

$$
\begin{array} { r l } & { \underline { { v } } _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( b ) = \underset { \pi _ { k } ^ { - i } } { \operatorname* { m i n } } \underset { c \in \mathcal { C } _ { k } ^ { i } } { \sum } b ( c ) \Bigg [ g _ { i , k } ^ { \mathfrak { c } } ( c , \pi _ { k } ^ { i } , \pi _ { k } ^ { - i } ( c ) ) } \\ & { \qquad + \underset { c ^ { \prime } \in \mathcal { C } _ { k + 1 } ^ { i } } { \sum } p _ { i , k } ^ { \mathfrak { c } } ( c ^ { \prime } \mid c , \pi _ { k } ^ { i } , \pi _ { k } ^ { - i } ( c ) ) w _ { i , k + 1 } ^ { \pi _ { t : T } ^ { i } } ( c ^ { \prime } ) \Bigg ] . } \end{array}
$$

By Lemma $^ { 8 , }$

$$
\begin{array} { r l } { \displaystyle \boldsymbol { v } _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( \boldsymbol { b } ) = \sum _ { c \in \mathcal { C } _ { k } ^ { i } } b ( c ) \operatorname* { m i n } _ { a _ { k } ^ { - i } } \Biggl [ g _ { i , k } ^ { \mathsf { c } } ( c , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) } & { } \\ { + \sum _ { c ^ { \prime } \in \mathcal { C } _ { k + 1 } ^ { i } } p _ { i , k } ^ { \mathsf { c } } ( c ^ { \prime } \mid c , \pi _ { k } ^ { i } , a _ { k } ^ { - i } ) w _ { i , k + 1 } ^ { \pi _ { t : T } ^ { i } } ( c ^ { \prime } ) \Biggr ] . } \end{array}
$$

By Theorem 3, the inner minimum equals $w _ { i , k } ^ { \pi _ { t : T } ^ { 2 } } ( c )$ . Thus

$$
\underline { { v } } _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( b ) = \sum _ { c \in \mathcal { C } _ { k } ^ { i } } b ( c ) w _ { i , k } ^ { \pi _ { t : T } ^ { i } } ( c ) .
$$

Linearity follows.

Theorem 5. $\underline { v } _ { t } ^ { i , * }$ is convex, and $\underline { { v } } _ { t } ^ { i , * } ( b ) = \operatorname* { m a x } _ { \underline { { v } } \in \Gamma _ { i , t } } \underline { { v } } ( b )$ for all $b \in B _ { t } ^ { i }$

Proof. Let

$$
\Gamma _ { i , t } \doteq \{ \underline { { v } } _ { i , t } ^ { \pi _ { t : T } ^ { i } } : \pi _ { t : T } ^ { i } \in \Pi _ { t : T } \} .
$$

By Lemma $1 , \Gamma _ { i , t }$ is compact in the finite-dimensional space of linear functionals.

By Theorem $^ { 4 , }$

$$
\underline { v } _ { t } ^ { i , * } ( b ) = v _ { i , t } ^ { \mathrm { c o n t } } ( b ) .
$$

By definition of $v _ { i , t } ^ { \mathrm { c o n t } }$

$$
v _ { i , t } ^ { \mathrm { c o n t } } ( b ) = \operatorname* { m a x } _ { \pi _ { t : T } ^ { i } \in \Pi _ { t : T } } \underline { { v } } _ { i , t } ^ { \pi _ { t : T } ^ { i } } ( b ) .
$$

Hence

$$
{ \underline { { v } } } _ { t } ^ { i , * } ( b ) = \operatorname* { m a x } _ { { \underline { { v } } } \in \Gamma _ { i , t } } { \underline { { v } } } ( b ) .
$$

For $b , b ^ { \prime } \in B _ { t } ^ { i }$ and $\lambda \in [ 0 , 1 ]$

$$
\begin{array} { r l } & { \underline { { v } } _ { t } ^ { i , * } ( \lambda b + ( 1 - \lambda ) b ^ { \prime } ) = \displaystyle \operatorname* { m a x } _ { \underline { { v } } \in \Gamma _ { i , t } } \underline { { v } } ( \lambda b + ( 1 - \lambda ) b ^ { \prime } ) } \\ & { \qquad = \displaystyle \operatorname* { m a x } _ { \underline { { v } } \in \Gamma _ { i , t } } \big [ \lambda \underline { { v } } ( b ) + ( 1 - \lambda ) \underline { { v } } ( b ^ { \prime } ) \big ] } \\ & { \qquad \leq \lambda \underline { { v } } _ { t } ^ { i , * } ( b ) + ( 1 - \lambda ) \underline { { v } } _ { t } ^ { i , * } ( b ^ { \prime } ) . } \end{array}
$$

Thus $\underline { v } _ { t } ^ { i , * }$ is convex.

## A.6 LOCAL BELLMAN BACKUPS

The theorem is conditional on finite coverage, Bellman closure, globally valid inserted vectors, infinite exact updates, and covered-backup consistency.

## A.6.1 LOCAL POINT-BASED BACKUP

We introduce the following linear program to compute a greedy decision rule with respect to a fixed continuation policy $\pi _ { t + 1 : T } ^ { i }$ and focal belief $b _ { t } ^ { i }$

The program uses the following variables:

$\mathbf { v } ( \mathbf { c _ { t } ^ { i } } )$ , representing the value associated with conditional state $c _ { t } ^ { i } .$

$\eta ( \mathbf { h } _ { \mathbf { t } + \mathbf { 1 } } ^ { - \mathbf { i } } , \mathbf { h } _ { \mathbf { t } + \mathbf { 1 } } ^ { \mathbf { 0 } } )$ , representing auxiliary continuation values,

$\pi _ { \mathbf { t } } ^ { \mathbf { i } } \in \Pi _ { \mathbf { t } } ^ { \mathbf { i } }$ , the focal decision rule to optimize.

Theorem 6. For anyfixed suffix policy $\pi _ { t + 1 : T } ^ { i }$ (associated with its valuefunction $\Lambda _ { i , t + 1 } )$ , andfocal belief $b _ { t } ^ { i } \in B _ { t } ^ { i }$ , solving the linear program

Max $\begin{array} { r } { \sum _ { c _ { t } ^ { i } } b _ { t } ^ { i } ( c _ { t } ^ { i } ) \mathbf { v } ( \mathbf { c _ { t } ^ { i } } ) } \end{array}$

$$
\begin{array} { r l } & { S u b j e c t \ t o \ v ( \mathbf { c _ { t } ^ { i } } ) \leq \displaystyle \sum _ { z _ { t } ^ { - i } , z _ { t } ^ { 0 } } \eta ( \mathbf { h _ { t + 1 } ^ { - i } } , \mathbf { h _ { t + 1 } ^ { 0 } } ) , \quad \forall c _ { t } ^ { i } , a _ { t } ^ { - i } } \\ & { \qquad \eta ( \mathbf { h _ { t + 1 } ^ { - i } } , \mathbf { h _ { t + 1 } ^ { 0 } } ) \leq \displaystyle \sum _ { h _ { t } ^ { i } , a _ { t } ^ { i } } \pi _ { \mathbf { t } } ^ { \mathbf { i } } ( \mathbf { a _ { t } ^ { i } } \mid \mathbf { h _ { t } ^ { i } } , \mathbf { h _ { t } ^ { 0 } } ) Q _ { ( h _ { t + 1 } ^ { - i } , h _ { t + 1 } ^ { i } , \alpha ) } , } \\ & { \qquad \quad \forall h _ { t + 1 } ^ { - i } , h _ { t + 1 } ^ { 0 } , \alpha \in \Lambda _ { i , t + 1 } , } \end{array}
$$

$$
\pi _ { \mathbf { t } } ^ { \mathbf { i } } \in \Pi _ { t } ^ { i }
$$

where

$$
( h _ { t + 1 } ^ { - i } , h _ { t + 1 } ^ { 0 } ) = ( c _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t } ^ { - i } , z _ { t } ^ { 0 } ) ,
$$

yields a decision rule $\pi _ { t } ^ { i }$ that greedily maximizes

$$
\underline { { q } } _ { t } ^ { \pi _ { t + 1 : T } ^ { i } } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } ) = \operatorname* { m i n } _ { \pi _ { t } ^ { - i } } g _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) + \underline { { v } } _ { i , t + 1 } ^ { \pi _ { t + 1 : T } ^ { i } } \big ( \tau _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) \big ) ,
$$

corresponding to the Q-value induced by thefixed-policy continuation $( \pi _ { t } ^ { i } , \pi _ { t + 1 : T } ^ { i } )$

Proof. By construction, $q _ { t } ^ { \pi _ { t + 1 : T } ^ { 2 } }$ is consistent with $\underline { v } _ { i , t } ^ { \pi _ { t + 1 : T } ^ { 2 } }$ , and an optimal $\pi _ { t } ^ { - i }$ can be taken deterministic (since the suffix policy $\pi _ { t + 1 : T } ^ { i }$ is fixed).

A greedy decision rule maximizes $q _ { t } ^ { \pi _ { t + 1 : T } ^ { i } }$ over reachable beliefs:

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { \pi _ { t } ^ { i } } \underline { { q } } _ { t } ^ { \pi _ { t + 1 : T } ^ { i } } \big ( b _ { t } ^ { i } , \pi _ { t } ^ { i } \big ) = \operatorname* { m a x } _ { \pi _ { t } ^ { i } } \operatorname* { m i n } _ { \pi _ { t } ^ { - i } } \Big [ g _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) + \underline { { v } } _ { i , t } ^ { \pi _ { t + 1 : T } ^ { i } } \big ( \tau _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) \big ) \Big ] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \operatorname* { m a x } _ { \pi _ { t } ^ { i } } \operatorname* { m i n } _ { \pi _ { t } ^ { - i } } \Big [ g _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } ) + \sum _ { c _ { t + 1 } ^ { i } } b _ { t + 1 } ^ { i } \big ( c _ { t + 1 } ^ { i } \big ) \displaystyle \operatorname* { m i n } _ { \alpha ^ { \pi _ { t + 1 : T } ^ { i } } } \alpha ^ { \pi _ { t + 1 : T } ^ { i } } \cdot c _ { t + 1 } ^ { i } \Big ] , } \end{array}
$$

where $b _ { t + 1 } ^ { i } = \tau _ { t } ^ { i } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t } ^ { - i } )$

Expanding the belief update and rearranging terms, we obtain:

$$
\begin{array} { r l } { \underset { \pi _ { t } ^ { i } } { \operatorname* { m a x } } \underset { i } { \underline { { q } } } _ { t } ^ { \pi _ { t + 1 : T } ^ { i } } ( b _ { t } ^ { i } , \pi _ { t } ^ { i } ) = \underset { \pi _ { t } ^ { i } } { \operatorname* { m a x } } \underset { \pi _ { t } ^ { - i } } { \operatorname* { m i n } } \underset { h _ { t } ^ { - i } , a _ { t } ^ { - i } , z _ { t } ^ { - i } , z _ { t } ^ { 0 } } { \sum } \pi _ { t } ^ { - i } ( a _ { t } ^ { - i } \mid h _ { t } ^ { - i } , h _ { t } ^ { 0 } ) } & { } \\ { \cdot } & { \underset { \alpha ^ { \pi _ { t + 1 : T } ^ { i } } } { \operatorname* { m i n } } \underset { h _ { t } ^ { i } , a _ { t } ^ { i } } { \sum } \pi _ { t } ^ { i } ( a _ { t } ^ { i } \mid h _ { t } ^ { i } , h _ { t } ^ { 0 } ) \underset { s _ { t } } { \sum } b _ { t } ^ { i } ( s _ { t } , h _ { t } ) } \\ & { \cdot \left[ g _ { t } ^ { i } ( s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } ) + \underset { s _ { t + 1 , z _ { t } } } { \sum } p ( s _ { t + 1 } , z _ { t } , z _ { t } ^ { 0 } \mid s _ { t } , a _ { t } ) \alpha ^ { \pi _ { t + 1 : T } ^ { i } } ( c _ { t + 1 } ^ { i } ) \right] . } \end{array}
$$

This expression makes explicit the max–min structure together with the inner minimization over $\alpha ,$ where α is taken after the expectation over next-stage variables. This placement is crucial: it preserves the piecewise-linear concave structure of the value function and yields directly the desired linear programming formulation.

If we denote $( h _ { t + 1 } ^ { - i } , h _ { t + 1 } ^ { 0 } ) = ( c _ { t } ^ { i } , a _ { t } ^ { - i } , z _ { t } ^ { - i } , z _ { t } ^ { 0 } )$ , using the bijection between $c _ { t } ^ { i }$ and the pair of histories $( h _ { t } ^ { - i } , h _ { t } ^ { 0 } )$ , we can define

$$
\begin{array} { r l } & { Q _ { ( h _ { t + 1 } ^ { - i } , h _ { t + 1 } ^ { i } , \alpha ) } } \\ & { = \displaystyle \sum _ { s _ { t } } b _ { t } ^ { i } ( s _ { t } , h _ { t } ) \Biggl [ g _ { t } ^ { i } ( s _ { t } , a _ { t } ^ { i } , a _ { t } ^ { - i } ) + \sum _ { s _ { t + 1 } , z _ { t } } p ( s _ { t + 1 } , z _ { t } , z _ { t } ^ { 0 } \mid s _ { t } , a _ { t } ) \alpha ^ { \pi _ { t + 1 : T } ^ { i } } ( c _ { t + 1 } ^ { i } ) \Biggr ] , } \end{array}
$$

then the above quantity corresponds to the q-value of the best response where player −i plays $a _ { t } ^ { - i }$ in $c _ { t } ^ { i }$ and follows $\alpha ^ { \pi _ { t + 1 : T } ^ { i } }$ at $( h _ { t + 1 } ^ { - i } , h _ { t } ^ { 0 } )$ . This yields the coefficients of the linear program described in the theorem.

We also impose an additional constraint on $\pi _ { t } ^ { i }$ to ensure that it is a valid decision rule, i.e., that each local strategy belongs to the simplex:

$$
\pi _ { t } ^ { i } ( \cdot  { | } h _ { t } ^ { i } , h _ { t } ^ { 0 } ) \in \Delta ( \mathcal { A } ^ { i } ) , \quad \forall h _ { t } ^ { i } , h _ { t } ^ { 0 }
$$

ensuring that $\pi _ { t } ^ { i }$ defines a proper probability distribution over actions at every local history.

## A.7 COMPARATIVE ANALYSIS OF BACKUP COMPLEXITY

Comparative analysis: Delage et al. (2023) The OMG greedy-update formulation Delage et al. (2023) enforces pessimism against all deterministic opponent decision rules $\pi _ { t } ^ { 2 } : H _ { t } ^ { 2 } \to \breve { A } _ { t } ^ { 2 }$ . Because there are $| A _ { t } ^ { 2 } | ^ { | H _ { t } ^ { 2 } | }$ such mappings, the LP description contains $O ( | A _ { t } ^ { 2 } | ^ { | H _ { t } ^ { 2 } | } )$ constraints (Table 2), which is exponential in the size of the opponent’s private-history space. Our backup removes this bottleneck by never indexing constraints by opponent decision rules: our LP constrains only against finitely many vectors in a cached envelope Λ. Hence our LP admits polynomial-size descriptions in the explicit parameters $| H _ { t } ^ { 1 } | , | H _ { t } ^ { 2 } | , | \dot { A } _ { t } ^ { 1 } | , | A _ { t } ^ { 2 } | , | Z _ { t } ^ { 2 } | , | \Lambda |$

<table><tr><td>Update step</td><td># variables</td><td># constraints</td></tr><tr><td>Ours: LP</td><td> $\lvert H _ { t } ^ { 1 } \rvert \lvert A _ { t } ^ { 1 } \rvert + \lvert H _ { t } ^ { 2 } \rvert \lvert A _ { t } ^ { 2 } \rvert \lvert Z _ { t } ^ { 2 } \rvert + \lvert H _ { t } ^ { 2 } \rvert$ </td><td> $O ( | H _ { t } ^ { 2 } | | A _ { t } ^ { 2 } | | Z _ { t } ^ { 2 } | | \Lambda | )$ </td></tr><tr><td>(Delage et  ${ \mathrm { a l . } } $  2023)</td><td>(rule variables + epigraph)</td><td>Exponential:  $O \big ( | A _ { t } ^ { 2 } | ^ { | H _ { t } ^ { 2 } | } \big )$ </td></tr><tr><td>(Escudie et  ${ \mathrm { a l . } } ,$  2025)</td><td> $\begin{array} { r } { | H _ { t } ^ { 1 } | | A _ { t } ^ { 1 } | | \Gamma _ { 1 } | + | H _ { t } ^ { 2 } | | A _ { t } ^ { 2 } | | Z _ { t } ^ { 2 } | + | H _ { t } ^ { 2 } | } \end{array}$ </td><td> $O \big ( | \Gamma _ { 1 } | \cdot | \Gamma _ { 2 } ^ { * } | \cdot | H _ { t } ^ { 2 } | \cdot | A _ { t } ^ { 2 } | \cdot | Z _ { t } ^ { 2 } | \big )$ </td></tr></table>

Table 2: LP complexity comparison (input-size view). The OMG greedy-update LP has constraints indexed by opponent decision rules, yielding an exponential dependence on $| H _ { t } ^ { 2 } |$ (Delage et al., 2023). Both the cached-envelope greedy $\mathrm { L P }$ (Escudie et al., 2025) and our backup scale polynomially in the same explicit representation objects, but expose different optimization structures. In our formulation, both the number of variables and constraints are reduced by a factor of $\left| \Gamma _ { 1 } \right|$ . Here, $\left| \Gamma _ { 1 } \right|$ denotes the number of piecewise-linear concave functions maintained, while $| \Lambda |$ is the number of piecewise-linear vectors associated with each element of $\Gamma _ { 1 }$

Comparative analysis: Escudie et al. (2025) In (Escudie et al., 2025), the LP explicitly incorporates all candidate next policies of i through the sets $\Gamma _ { 1 }$ of piecewise linear concave function $\Lambda ,$ which grow over iterations. Specifically, a single monolithic greedy LP jointly couples the outer cache index $\Gamma _ { 1 }$ , the inner value index $\alpha \in \Lambda$ , and the opponent action-observation indices $( a _ { t } ^ { 2 } , z _ { t } ^ { 2 } )$ This full coupling within a single program leads to a constraint count of $O \big ( | \Gamma _ { 1 } | \cdot | \Lambda | \cdot | H _ { t } ^ { 2 } | \cdot | A _ { t } ^ { 2 } | \cdot | Z _ { t } ^ { 2 } | \big )$ as reported in Table 2.

Although the reduction in | $\Gamma _ { 1 } ~ |$ has little impact for short horizons, it becomes critical for longer horizons. Indeed, in PBVI (Escudie et al., 2025), the size of $\Gamma _ { 1 }$ grows with the number of sampled points, which itself increases with the number of stages required to cover an ever-expanding continuous state space.

## B ALGORITHM

Value iteration in (Escudie et al., 2025) amounts to solving a focal belief game using sampled opponent behaviors. However, this unguided sampling process induces an exponential growth in the number of sampled points, resulting in a rapidly increasing computational complexity. To address this limitation, we introduce a policy iteration approach that exploits the structural properties of the game together with the reduced formulations defined previously. Leveraging this structure yields a more efficient iterative procedure while preserving optimality guarantees.

We prove that the dynamics induced by the game can be viewed as a discretization of a continuoustime dynamical system. Along the resulting continuous trajectory, the dual gap is non-increasing and converges to zero. The proof relies on three key properties: the convexity of the dual gap in the policy space, the differentiability of the value function almost everywhere, and the uniform continuity of the value function. Together, these properties allow us to characterize the evolution of the dual gap along the continuous dynamics and establish its convergence.

This section introduces and formally defines the main objects used throughout the analysis, including the dual gap and the dynamics guiding the update $F _ { t } ^ { i }$ , and related quantities.

The continuous dynamics describe the infinitesimal policy updates underlying the algorithm. At each stage t, the update acts on the policy $\pi _ { t } ^ { i }$ while the policies at other stages are kept fixed. The resulting direction therefore belongs to the local policy space $\Pi _ { t } ^ { i } ,$ , while depending on the complete policy of player i and on the opponent’s policies up to stage $t - 1$ through the induced belief state. This defines a vector field over the policy space, whose integral curves describe the continuous version of the algorithm.

Definition 4 (Continuous Dynamics). For player i and stage t, the continuous dynamics are defined as the vector field

$$
F _ { t } ^ { i } : \Pi _ { 0 : T - 1 } ^ { i } \times \Pi _ { 0 : t - 1 } ^ { - i }  \Pi _ { t } ^ { i } ,
$$

given by

$$
F _ { t } ^ { i } \left( \pi _ { 0 : T - 1 } ^ { i } , \pi _ { 0 : t - 1 } ^ { - i } \right) = \nabla _ { \pi _ { t } ^ { i } \underline { { q } } } \pi _ { t + 1 : T } ^ { i } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } \right) ,
$$

where $b _ { t } ^ { i }$ is the focal belief induced by $\pi _ { 0 : t - 1 } ^ { i }$ and $\pi _ { 0 : t - 1 } ^ { - i }$ and $\underline { { q } } ^ { \pi _ { t + 1 : T - 1 } ^ { 2 } }$ is defined in Theorem 6. Thus, $F _ { t } ^ { i }$ gives the safest most improving partial direction of the pessimistic $Q \cdot$ -value with respect to $\pi _ { t } ^ { i } .$ , while keeping the belief fixed.

We may ask whether having the agents follow their respective induced flows improves their value ${ \underline { { v } } } _ { t } ^ { i }$ . The direction induced by $F _ { t } ^ { i }$ is safely improving for timestep t to T, as all future best-responses are considered in $q ^ { \pi _ { t + 1 : T - 1 } ^ { i } }$ . However, it remains unclear whether this improvement is preserved globally across all stages. Indeed, $F _ { t } ^ { i }$ is defined under the assumption that the opponent’s past policy remains fixed, as reflected by its dependence on $\pi _ { 0 : t - 1 } ^ { - i }$ When the policies are updated simultaneously across iterations, this assumption may no longer hold: changes to the opponent’s policy at earlier stages can modify the induced belief $\hat { b } _ { t } ^ { i }$ and, consequently, the continuation value at later stages. This creates a situation similar to fictitious play, where both players update their policies simultaneously and each player’s update can therefore affect the optimization problem faced by the other.

Definition 5 (Dual gap). Given a joint policy $\pi _ { t : T - 1 }$ and the belief state $b _ { t } ^ { i }$ induced by the joint policy up to time $t - 1$ , the dual gap at time t is defined as

$$
d _ { t } ^ { \pi _ { t : T - 1 } } \left( b _ { t } ^ { i } \right) = \underline { { q } } _ { t } ^ { \pi _ { t + 1 : T - 1 } ^ { - i } } \left( b _ { t } ^ { i } , \pi _ { t } ^ { - i } \right) - \underline { { q } } _ { t } ^ { \pi _ { t + 1 : T - 1 } ^ { i } } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } \right) .
$$

Lemma 12 (Local stability of the best response). At any regular policy profile, the best response of the opponent in the constrained subgame rooted at the current policy is locally constant. Consequently, the pessimistic $Q \cdot$ -value

$$
\underline { { q } } _ { t } ^ { i } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } \right) \dot { = } \underline { { q } } _ { t } ^ { i , \pi _ { t + 1 : T - 1 } ^ { i } } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } \right)
$$

defined in Theorem 6 is differentiable with respect to $\pi _ { t } ^ { i }$ in a sufficiently small neighborhood of the current policy.

Proof. At a regular policy profile, the opponent’s best response is unique, since the game is concaveconvex and admits a finite set of deterministic policies. Consequently, there exists a neighborhood of the current policy in which this best response remains unchanged. Within this neighborhood, the pessimistic continuation value is therefore evaluated against a fixed best response. The remaining terms in the Bellman recursion are differentiable with respect to $\pi _ { t } ^ { i }$ . Hence,

$$
\underline { { q } } _ { t } ^ { i } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } \right)
$$

is differentiable with respect to $\pi _ { t } ^ { i }$

The Q-value function is not differentiable everywhere, but this issue is not specific to our setting: the same non-differentiability arises for the dual gap in normal-form games. Since our method uses the same type of local improvement mechanism in policy space, it can handle these points in the same way. Moreover, our discretization does not encounter pathological cases in practice, and the value function explicitly accounts for possible deviations from the current policy. This provides additional protection against numerical instabilities caused by nondifferentiable points and ensures that the improvement step remains robust to such cases.

Lemma 13 (Local improving direction). At a regular policy profile, the direction $F _ { t } ^ { i }$ is an improving direction for player i’s pessimistic value at stage t. In particular,

$$
\left. \frac { d } { d \alpha } \underline { { { q } } } _ { t } ^ { i } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } + \alpha F _ { t } ^ { i } \right) \right| _ { \alpha = 0 } > 0
$$

whenever $\pi _ { t } ^ { i }$ is not locally optimal.

Proof. By Lemma 12, the local pessimistic $Q \cdot$ -value is differentiable around the current policy. By construction of $F _ { t } ^ { i }$

$$
F _ { t } ^ { i } = \nabla _ { \pi _ { t } ^ { i } } \underline { { q } } _ { t } ^ { i } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } \right) .
$$

Therefore,

$$
\left. { \frac { d } { d \alpha } } { \underline { { q } } } _ { t } ^ { i } \left( b _ { t } ^ { i } , \pi _ { t } ^ { i } + \alpha F _ { t } ^ { i } \right) \right| _ { \alpha = 0 } = \left\| F _ { t } ^ { i } \right\| ^ { 2 } > 0
$$

whenever $F _ { t } ^ { i } \neq 0$

Lemma 14 (Safe improvement for $T = 2 )$ . Suppose that $T = 2 .$ . For sufficiently small step sizes, the update of player i following the dynamics $\hat { F } _ { t } ^ { i }$ strictly improves her pessimistic value whenever the current policy is not locally optimal.

Proof. Consider the joint policy $( \pi _ { 0 : 1 } ^ { 1 , k } , \pi _ { 0 : 1 } ^ { 2 , k } )$ at iteration k. The decision rule $\pi _ { 0 } ^ { i , k }$ is optimal for the pessimistic continuation problem induced by $\pi _ { 1 } ^ { i , k }$

We first update the last-stage decision rule according to the above dynamics. Since there is no subsequent decision rule, the associated one-stage pessimistic problem is solved directly. Thus, considering any step size $\alpha _ { 1 } ^ { k } > 0$ , the update

$$
\pi _ { 1 } ^ { i , k + 1 } = \pi _ { 1 } ^ { i , k } + \alpha _ { 1 } ^ { k } F _ { 1 } ^ { i }
$$

improves the corresponding local pessimistic value, and hence decreases the corresponding local dual gap.

We then update the first-stage decision rule while keeping the newly obtained $\pi _ { 1 } ^ { i , k + 1 }$ fixed. Define

$$
v ^ { k } = \underline { { v } } _ { 0 } ^ { i , [ \pi _ { 0 } ^ { i , k } , \pi _ { 1 } ^ { i , k } ] } ,
$$

$$
\widetilde { v } ^ { k } = \underline { { v } } _ { 0 } ^ { i , [ \pi _ { 0 } ^ { i , k } , \pi _ { 1 } ^ { i , k + 1 } ] } ,
$$

and

$$
v ^ { k + 1 } = \underline { { v } } _ { 0 } ^ { i , [ \pi _ { 0 } ^ { i , k + 1 } , \pi _ { 1 } ^ { i , k + 1 } ] } .
$$

By construction of the first-stage update, $F _ { 0 } ^ { i }$ is an improving direction for the local function. Therefore, the newly obtained policy $\pi _ { 0 } ^ { i , \dot { k } + 1 }$ strictly improve the overall pessimistic value

$$
\pi _ { 0 } ^ { i } \mapsto \underline { { { v } } } _ { 0 } ^ { i , [ \pi _ { 0 } ^ { i } , \pi _ { 1 } ^ { i , k + 1 } ] } ,
$$

and thus,

$$
v ^ { k + 1 } > \widetilde { v } ^ { k } .
$$

It remains to ensure that the improvement at stage $t = 0$ is not cancelled by the preceding update at stage $t = 1$ . To this end, we regard the stage $t = 1$ update as a homotopy

$$
\pi _ { 1 } ^ { i , k + 1 } ( \alpha ) = \pi _ { 1 } ^ { i , k } + \alpha F _ { 1 } ^ { i } , \qquad \alpha \in [ 0 , \bar { \alpha } ] .
$$

Let

$$
\Delta ( \alpha ) = \underline { { v } } _ { 0 } ^ { i , [ \pi _ { 0 } ^ { i , k + 1 } ( \alpha ) , \pi _ { 1 } ^ { i , k + 1 } ( \alpha ) ] } - \underline { { v } } _ { 0 } ^ { i , [ \pi _ { 0 } ^ { i , k } , \pi _ { 1 } ^ { i , k } ] } ,
$$

where $\pi _ { 0 } ^ { i , k + 1 } ( \alpha )$ denotes the stage $t = 0$ update computed from the continuation policy $\pi _ { 1 } ^ { i , k + 1 } ( \alpha )$

At $\alpha = 0 ;$ the continuation policy coincides with the policy at iteration k. Since $\pi _ { 0 } ^ { i , k + 1 }$ is an improving direction for the corresponding local problem, we have $\Delta ( 0 ) > 0$ at every non-degenerate point where the stage 0 direction is strictly improving. By continuity of the pessimistic value in the policy and, locally, of the stage 0 update, there exists α¯ $> 0$ such that

$$
\Delta ( \alpha ) > 0 \qquad \mathrm { f o r ~ a l l ~ } 0 < \alpha < \bar { \alpha } .
$$

Hence, the stage 1 step can be chosen sufficiently small so that the improvement obtained at stage 0 is not cancelled by the preceding continuation update. Consequently,

$$
v ^ { k + 1 } > v ^ { k } .
$$

Lemma 15. For a horizon T, ifthe stage wise updatesfrom stage 1 to stage $T - 1$ can be chosen to strictly improve the pessimistic value, then the update at stage $t = 0$ can also be chosen so that the complete iteration strictly improves the pessimistic value.

Proof. Assume that the result holds for a horizon T. Consider a horizon $T + 1$ . Starting from the last stage, we apply the construction from Lemma 14 to stages

$$
1 , \ldots , T - 1 .
$$

By the induction hypothesis, the continuation problem from stage 1 onwards can be updated with sufficiently small step sizes so that its pessimistic value is improved.

We then consider stage $t = 0$ while keeping the resulting continuation policies fixed. The same local argument applies: the continuation pessimistic value induces a differentiable local Q-value around the current stage $t = 0$ policy, and the update follows an improving direction of this function.

Finally, the step sizes at stages $1 , \ldots , T - 1$ can be chosen sufficiently small so that the improvement produced at stage 0 dominates the change induced by the continuation updates. By continuity, there therefore exist positive step sizes

$$
\alpha _ { 0 } ^ { k } , \ldots , \alpha _ { T } ^ { k }
$$

for which the complete iteration yields a strict improvement of the pessimistic value.

Lemma 16. For any starting joint policy π, there exists a discretized trajectory

$$
\phi = ( \pi ^ { 1 , k } , \pi ^ { 2 , k } ) _ { k \geq 0 }
$$

starting from any $( \pi ^ { 1 , 0 } , \pi ^ { 2 , 0 } )$ and following the flow induced by $( F _ { 0 : T - 1 } ^ { i } , F _ { 0 : T - 1 } ^ { - i } )$ such that the dual gap strictly decreases at every non-degenerate iteration.

Proof. We construct the trajectory iteratively. At iteration k, the policy at each stage is updated according to

$$
\pi _ { t } ^ { i , k + 1 } = \pi _ { t } ^ { i , k } + \alpha _ { t } ^ { k } F _ { t } ^ { i } \left( \pi _ { 0 : T - 1 } ^ { i , k } , \pi _ { 0 : T - 1 } ^ { - i , k } \right) ,
$$

with $\alpha _ { t } ^ { k } > 0 .$

By Lemma 15, whenever an improvement at an earlier stage could be affected by the updates of the continuation stages, the corresponding continuation step sizes can be chosen sufficiently small so that this improvement is preserved.

Consequently, at every non-degenerate iteration, the step sizes

$$
\alpha _ { 0 } ^ { k } , \dots , \alpha _ { T - 1 } ^ { k }
$$

![](images/7b4e2b2ac164febc26190483ef9107fdc2375cee8696587cc09297b33db41029.jpg)  
Figure 3: Example of a situation at $T = 2 .$ . The background represents the gradient of the dual gap at $t = 0$ . The black arrows represent updates of the policy where only $\pi _ { 0 } ^ { i }$ is changing. $\mathbf { A } \mathfrak { t } : = 0$ since i has an optimal view of the pessimistic value function, the move strictly improves the global dual gap. The dashed arrows represent updates at $t = 1 ;$ since the update is also influenced by −i past, nothing guarantees that the update will follow the global dual gap $d _ { 0 }$ gradient direction. Here, we present an extreme case where the update at $t = 1$ almost cancels the one at $t = 0$ . On the left, we do not adapt the step sizes, on the right, we apply the proof such that destructive updates are avoided and we are sure the updates go to the right direction.

can be reduced arbitrarily while remaining strictly positive and can be chosen so that the complete iteration strictly improves the pessimistic value. Equivalently, the associated dual gap satisfies

$$
d _ { 0 } ( b _ { 0 } , ( \pi ^ { 1 , k + 1 } , \pi ^ { 2 , k + 1 } ) ) < d _ { 0 } ( b _ { 0 } , ( \pi ^ { 1 , k } , \pi ^ { 2 , k } ) ) .
$$

Starting from any $\pi ,$ choosing such step sizes at each iteration defines a discretized trajectory $\phi$ following the induced flow. Along this trajectory, the dual gap therefore strictly decreases at every non-degenerate iteration. □

We have established the existence of a trajectory following the flow induced by $F _ { t } ^ { i }$ . Since the step size can be chosen arbitrarily small, we can construct a sequence of discrete trajectories with vanishing step sizes that increasingly closely approximates the continuous flow. Along this trajectory, the dual gap $d _ { 0 }$ is non-increasing. It remains to show that the trajectory converges to the set of Nash equilibria.

Lemma 17. Suppose that, at a joint policy π, a safe local improving direction existsfor the overall objective. Then there exists a player i and a stage t such that the dynamics $F _ { t } ^ { i }$ identify a strictly improving direction at the corresponding induced belief state $b _ { t } ^ { i }$

Proof. We prove the result by contradiction. Consider first the case $T = 2$ . Suppose that a safe improving direction exists (considering all stages at once), but that no improving direction can be identified inductively at either stage.

In particular, suppose that no improving direction can be identified at $t = 0$ . Then the improvement cannot be obtained from the current decision alone and must arise from the continuation decision at $t = 1$ . Hence, there exists an induced belief state $b _ { 1 } ^ { i }$ , arbitrarily close to the current one, at which a local change of the policy at $t = 1$ strictly improves the continuation objective.

By the local improvement property at $t = 1$ , this strict improvement implies the existence of an improving direction at that belief state. Since the continuation value is continuous with respect to the induced belief state, the same improving direction remains improving in a neighborhood of the current belief state. It can therefore be detected at the current belief state, contradicting the assumption that no improving direction can be identified inductively.

Thus, for $T = 2$ , every safe local improvement is detected at one of the two stages.

The argument extends inductively to an arbitrary finite horizon. Suppose that no improving direction can be identified at any stage. Then, after ruling out an improvement at the current stage, any remaining improvement must arise from the continuation problem. Applying the same argument to this continuation problem yields a later stage at which a strict local improvement exists. By continuity, this improvement must already be detectable at the corresponding current induced belief state, contradicting the assumption.

Therefore, whenever a safe global improving direction exists, the dynamics identify a strictly improving direction at some stage t. □

The next theorem characterizes the points toward which the continuous dynamics converges.

Theorem 7. The continuous dynamics $\phi$ introduced in Lemma 16 converges to the set of Nash equilibria ofthe game.

Proof. Consider a joint policy

$$
\pi \not \in { \mathcal { N } } ,
$$

outside of the set of the Nash Equilibria. Since the dual gap vanishes if and only if the joint policy is a Nash equilibrium,

$$
d _ { 0 } ( \pi ) > 0 .
$$

$$
D _ { F _ { t } ^ { i } } d _ { 0 } ( \pi ) < 0 .
$$

Therefore, at every joint policy outside ${ \mathcal { N } } ,$ the dynamics admit a direction that strictly decreases the dual gap. By Lemma 16, the updates can be chosen sufficiently small so that the simultaneous evolution of the two players does not cancel this strict improvement. Consequently, along the continuous dynamics,

$$
\frac { \mathrm { d } } { \mathrm { d } \tau } d _ { 0 } ( \phi ( \tau ) ) < 0 \qquad \mathrm { w h e n e v e r } \phi ( \tau ) \notin \mathcal { N } .
$$

Thus, the dual gap is a strict Lyapunov function outside the set ${ \mathcal { N } } .$

Because the joint policy space is compact and the dual gap is non-negative, the function

$$
g ( \tau ) = d _ { 0 } ( \phi ( \tau ) )
$$

is monotonically decreasing and bounded from below. Hence, it converges to some $g _ { \infty } \geq 0$

Suppose, for contradiction, that

$$
g _ { \infty } > 0 .
$$

Then $\phi ( \tau )$ remains outside $\mathcal { N }$ for all sufficiently large τ. At every such point, the argument above provides a strictly improving direction for the dynamics, and therefore the dual gap must continue to decrease. This contradicts the convergence of $g ( \tau )$ to the strictly positive limit $g _ { \infty }$

$$
g _ { \infty } = 0 ,
$$

and therefore

$$
\operatorname * { l i m } _ { \tau  \infty } d _ { 0 } ( \phi ( \tau ) ) = 0 .
$$

Since the dual gap is zero if and only if the joint policy belongs to ${ \mathcal { N } } ,$ the trajectory $\phi ( \tau )$ converges to the set of Nash equilibria. □

The previous result establishes that the continuous dynamics converge to the set of Nash equilibria. We now relate these continuous trajectories to the discrete-time policy updates considered in practice. In particular, we show that, under the appropriate conditions, the discrete dynamics track the continuous dynamics sufficiently closely to inherit the same convergence property. This connection is formalized in the following theorem.

Proof of Theorem 2. The discrete algorithm can be written as a stochastic approximation of the continuous dynamics introduced in Definition 4. In particular, for each player i,

$$
\pi ^ { i , k + 1 } = ( 1 - \alpha _ { k } ) \pi ^ { i , k } + \alpha _ { k } \tilde { F } _ { t } ^ { i } \left( \pi ^ { i , k } , \pi ^ { - i , k } \right) + U _ { i , k + 1 } .
$$

The operator $\tilde { F } _ { t } ^ { i }$ is the discrete counterpart of the continuous improving direction $F _ { t } ^ { i }$ . Both operators perform the same local pessimistic optimization, but they differ in how this optimization is carried out. While $F _ { t } ^ { i }$ describes the continuous direction of improvement, $\ddot { F } _ { t } ^ { i }$ is obtained by solving the stage-wise LPs used by the algorithm. These LPs exploit the uniform continuity of the value function to construct a local approximation of the pessimistic Q-value. In particular, $\tilde { F } _ { t } ^ { i }$ maximizes this local approximation while matching the exact pessimistic value at the current point (see Theorem 6). Thus, the LP provides a discrete approximation of the same local optimization characterized by $F _ { t } ^ { i }$

Consequently, the interpolated sequence of policies can be viewed as a stochastic approximation of the continuous system

$$
\dot { \pi ^ { i } } = F ^ { i } \left( \pi ^ { i } , \pi ^ { - i } \right) .
$$

Under the Robbins-Monro conditions and the standard assumptions on the noise, the cumulative effect of the stochastic perturbations remains controlled, while the updates continue to explore the policy space. Together with the assumed consistency of the LP operator with the local continuous dynamics, this ensures that the interpolated trajectory is an asymptotic pseudo-trajectory of the flow induced by $\left( { F } ^ { 1 } , { F } ^ { 2 } \right)$

Proposition 1.3 of Benaïm et al. (2005) provides precisely the required connection. It states, under the corresponding stochastic approximation conditions, that the asymptotic behavior of the algorithm is preserved when its iterates are viewed as perturbed solutions of the limiting differential inclusion. Since we have established that the trajectories of this continuous dynamics converge to a Nash equilibrium, the proposition allows us to transfer this convergence result to the discrete algorithm.

By Lemma 7, the limiting continuous dynamics converges to the set $\mathcal { N }$ of Nash equilibria. Hence, the only internally chain-transitive invariant set of the limiting dynamics is contained in $\mathcal { N }$ . Benaïm et al. (2005)’s stochastic approximation result then implies that the limit set of the discrete iterates is almost surely contained in ${ \mathcal { N } } .$ . Therefore,

$$
\begin{array} { r } { \mathrm { d i s t } \left( \pi ^ { k } , \mathcal { N } \right) \xrightarrow [ k \to \infty ] { \mathrm { a . s . } } 0 . } \end{array}
$$

Thus, the algorithm converges almost surely to the set of Nash equilibria.

## C EXTENSIVE FORM GAME

## C.1 A HIERARCHY OF PLANNERS

Escudie et al. (2025) introduce a hierarchy of planners for zero-sum partially observable stochastic games (zs-POSGs), based on the observation that different planning procedures arise from progressively stronger assumptions about how future decisions are treated during planning. In particular, the hierarchy distinguishes planners according to the information they have access to and the extent to which future decisions—both their own and those of the opponent—are treated as fixed or optimized during planning.

At the two levels relevant here, the hierarchy can be summarized as follows:

• Focal planner: a player-specific planning process in which only the focal player’s policy is known. The planner therefore reasons about its own decisions without access to the opponent’s policy.

• Uninformed planner: a planner that additionally knows the opponent’s policy, together with the information available to the focal planner. This setting can be formulated as a transition-independent zero-sum stochastic game, and is closely related to the stochasticgame formulation introduced at the beginning of this paper.

Beyond this operational characterization, we argue that the hierarchy admits a deeper structural interpretation. To make this interpretation explicit, we first consider the example of Sokota et al. (2023), which illustrates why a reduction based solely on public policies can fail. In particular, their discussion of the Backward Dependence Problem highlights how future decisions may depend on information and decisions made earlier in the game, even when these dependencies are not captured by the public policy representation alone.

## C.2 EQUILIBRIUM AND DYNAMIC PROGRAMMING

The distinction between simultaneous- and sequential-move games provides a deeper motivation for the hierarchy of planners introduced by Escudie et al. (2025). We illustrate this point by revisiting the example of Sokota et al. (2023) and comparing it with the approach developed in Escudie et al. (2025).

In a simultaneous-move POSG, the equilibrium sought by the dynamic-programming formulation is a Subgame Perfect Nash Equilibrium (SPNE). This is reflected directly in the corresponding Bellman recursion. In the two-player zero-sum case, the backward recursion can be written schematically as

$$
\begin{array} { r } { N E _ { ( \pi _ { 0 } ^ { 1 } , \pi _ { 0 } ^ { 2 } ) } N E _ { ( \pi _ { 1 } ^ { 1 } , \pi _ { 1 } ^ { 2 } ) } \cdot \cdot \cdot N E _ { ( \pi _ { T - 1 } ^ { 1 } , \pi _ { T - 1 } ^ { 2 } ) } v ^ { \pi ^ { 1 } , \pi ^ { 2 } } , } \end{array}
$$

where NE denotes a Nash equilibrium of the corresponding subgame. Thus, the equilibrium is constructed recursively, with the decision rules at each stage being optimized within the corresponding subgame. Because the recursion solves each subgame, the resulting strategy profile satisfies the subgame-perfectness requirement.

This correspondence between dynamic programming and SPNE does not hold in the same way for sequential POSGs, and consequently for extensive-form games (EFGs). In an EFG, the Nash equilibrium sought by the game formulation need not be subgame perfect, even after a reduction to public policies.

This distinction is particularly apparent in Kuhn Poker, where the optimal decision rule at the final stage is not deterministic. If the Bellman recursion above is nevertheless applied directly to the se quential game after reducing it to a public-policy representation, the final stage is optimized through a max or min over the available decision rules. This optimization selects a deterministic decision rule at the final stage and therefore fails to capture the dependence of the final decision on deviations at earlier stages. Sokota et al. (2023) identify the same structural issue in their example of the “Backward Dependence Problem.”

This observation motivates changing the equilibrium formulation rather than simply applying the simultaneous-game Bellman recursion to a sequential game. Nevertheless, an extensive-form game can be transformed into a lossless game for which a Subgame Perfect Nash Equilibrium exists. The resulting formulation is closely related to the reduction introduced at the beginning of this paper.

## C.3 ONE-SIDED EXTENSIVE-FORM GAMES

How should we handle terminal values in a finite-horizon formulation? In an extensive-form game (EFG), terminal nodes may be reached before the last possible time step, whereas the dynamicprogramming formulation evaluates the value at the final time step. Without loss of generality, we can postpone the realization of any terminal reward until the final time step by introducing absorbing states. More precisely, whenever a terminal node is reached at time $t < T - 1$ , we transition to an absorbing state and keep the process there until time T−1, at which point the corresponding terminal reward is received. This transformation preserves the value of every policy profile while ensuring that all rewards are realized at the final time step. This is done for readability and to maintain consistency across the different game formulations, thereby simplifying the notation and equations.

Definition 6. Let i be a player.We define the associated one-sided EFG (oEFG) as

$$
\mathcal { E } ^ { i } \doteq ( \mathcal { T } , \mathcal { D } , \mathbf { H } , \mathbf { H } ^ { k } , \mathcal { D } _ { t } ^ { i } , u ) ,
$$

where the components are defined as follows:

• I is the set of n agents in the original game.

• D is the decision tree. Each node at time t represents a joint policy prefix

$$
\pi _ { 0 : t } = \{ \pi _ { 0 } ^ { 1 } , \pi _ { 0 } ^ { 2 } , \ldots , \pi _ { t } ^ { k } \} ,
$$

where k denotes the player k. A child node $\pi ^ { \prime }$ is accessible from π if

$$
\exists \pi _ { t } ^ { k } \quad { \mathrm { s u c h ~ t h a t } } \quad \pi ^ { \prime } = [ \pi , \pi _ { t } ^ { k } ] .
$$

$\mathbf { H } ^ { k }$ is the information partition of player k:

$$
\mathbf { H } ^ { k } ( \pi _ { 0 : t } ) \doteq { \left\{ \begin{array} { l l } { \pi _ { 0 : t } ^ { k } } & { { \mathrm { i f ~ } } k = i , } \\ { \pi _ { 0 : t } } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Thus, the focal player i observes only her own policy prefix, whereas the opponent −i observes the complete joint policy prefix.

$\mathcal { D } _ { t } ^ { k } = \Pi _ { t } ^ { k }$ is the space of decision rules available to player k at time t.

• u is the reward function defined on terminal nodes:

$$
u ( \pi _ { 0 : T } ) \stackrel { . } { = } v ^ { \pi _ { 0 : T } ^ { 1 } , \pi _ { 0 : T } ^ { 2 } } .
$$

The players make their decisions sequentially, with i choosing first and −i choosing after observing the complete policy prefix of i. Consequently, the value of the game is characterized by a max-min optimization: i chooses a decision rule to maximize the value, while −i observes this choice and subsequently chooses a decision rule to minimize it.

Under this construction, i has access only to her own policy, whereas −i has access to both players’ policies. We then show that any zero-sum EFG admits a lossless reduction to the corresponding zero-sum oEFG.

Theorem 8. Let E be a zero-sum extensive-form game and let i be the focal player. The reduction

$$
\mathcal { T } _ { \mathcal { E }  \mathcal { E } ^ { i } }
$$

that projects $\mathcal { E }$ onto the one-sided EFG $\mathcal { E } ^ { i }$ is lossless for player i. In particular, it preserves the value ofevery joint policy and thefocal player’s maximin value and optimal policies.

Proof. We establish the result by showing that the reduction preserves (i) the correspondence between joint policies and paths in the decision tree, (ii) the associated values, and (iii) the information available to the focal player.

Policy and path correspondence. Consider a joint policy

$$
\pi _ { 0 : T } = ( \pi _ { 0 : T } ^ { i } , \pi _ { 0 : T } ^ { - i } )
$$

in the original EFG E. The reduction associates with this joint policy a policy π˜ in the one-sided EFG E<sup>i</sup>.

At each node corresponding to a policy prefix $\pi _ { 0 : t } .$ , the next decision of the focal player is determined by her own policy prefix $\pi _ { 0 : t } ^ { i } ,$ whereas the next decision of the opponent is determined by the complete policy prefix $\pi _ { 0 : t }$ . Thus, the complete joint policy uniquely determines a path in the decision tree ${ \bar { \mathcal { D } } } ,$ and conversely, a path in $\mathcal { D }$ specifies the sequence of decision rules constituting the corresponding joint policy.

Value preservation. By construction, the terminal node reached by the path associated with $\pi _ { 0 : T }$ has reward

$$
u ( \pi _ { 0 : T } ) \dot { = } v ^ { \pi _ { 0 : T } ^ { i } , \pi _ { 0 : T } ^ { - i } } .
$$

Consequently, the expected value induced by any joint policy is unchanged by the reduction:

$$
u ( \tilde { \pi } ) = v ^ { \pi _ { 0 : T } ^ { i } , \pi _ { 0 : T } ^ { - i } } .
$$

Hence, optimizing over corresponding policies in the two representations yields the same payoff. In particular, the reduction does not introduce or remove any payoff-relevant trajectories.

Information preservation for the focal player. The reduction is one-sided precisely because it preserves the information available to the focal player while explicitly exposing additional information to the opponent. In ${ \mathcal { E } } ^ { i }$ , player i observes only her own policy prefix through the information partition $\mathbf { H } ^ { i }$

$$
\mathbf { H } ^ { i } ( \pi _ { 0 : t } ) = \pi _ { 0 : t } ^ { i } .
$$

This is exactly the information on which her decision rule is allowed to depend. By contrast, the opponent observes the complete policy prefix,

$$
\mathbf { H } ^ { - i } ( \pi _ { 0 : t } ) = \pi _ { 0 : t } ,
$$

and can therefore condition its decision on the focal player’s previous decisions. The opponent thus acts as an informed best responder. Since the theorem concerns losslessness for player i, granting this additional information to the opponent does not alter the focal player’s policy space or information structure; rather, it makes the opponent’s minimization explicit.

Preservation of the focal player’s maximin problem. Because the game is zero-sum, the value associated with a joint policy can be written as

$$
v ^ { \pi ^ { i } , \pi ^ { - i } } .
$$

The focal player’s security problem in the original game is therefore

$$
\operatorname* { m a x } _ { \pi ^ { i } } \operatorname* { m i n } _ { \pi ^ { - i } } v ^ { \pi ^ { i } , \pi ^ { - i } } .
$$

The policy and value correspondence established above implies that the same optimization problem is represented in ${ \mathcal { E } } ^ { i }$ :

$$
\operatorname* { m a x } _ { \tilde { \pi } ^ { i } } \operatorname* { m i n } _ { \pi ^ { - i } } u ( \tilde { \pi } ) .
$$

Thus, the two formulations have the same maximin value, and a policy is optimal for player i in the original EFG if and only if its corresponding policy is optimal in the one-sided EFG.

Finally, by the minimax theorem for finite zero-sum games, the maximin value coincides with the value attained at a Nash equilibrium. Hence, the focal player’s equilibrium value and the set of focal-player maximin strategies are preserved by the reduction, even though the one-sided representation does not claim to preserve the opponent’s information structure or the full set of equilibrium strategies. □

Theorem 9. Under the convention introduced in Definition 6, the one-sided EFG ${ \mathcal { E } } ^ { i }$ admits a unilateral Subgame Perfect Nash Equilibrium. In particular, its value satisfies the following Bellman recursion:

![](images/473fff30dbb14cb93bf1e0a791c6377d363785b36343ae99aa3eb1b4374b8090.jpg)  
Figure 4: $\mathcal { T } _ { \mathcal { G }  \mathcal { G } ^ { \prime } }$ denotes the reduction from game $\mathcal { G }$ to game $\mathcal { G } ^ { \prime } .$ . The reduction $\mathcal { T } _ { P O S G  E F G }$ is almost transparent: it simply reformulates the POSG as an extensive-form game (EFG). However, unlike $\mathcal { T } _ { P O S G  S G }$ , this reduction does not induce an equilibrium concept that can be characterized inductively in the reduced game. The composition $\mathcal { T } _ { P O S G  E F G } \circ \mathcal { T } _ { E F G  O E F G }$ corresponds to the focal-planner reduction introduced in Escudie et al. (2025).

$$
v _ { t } ^ { * , i } ( \pi _ { 0 : t - 1 } ^ { i } ) = \operatorname* { m a x } _ { \pi _ { t } ^ { i } } \operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { - i } } v _ { t + 1 } ^ { * , i } ( \pi _ { 0 : t } ^ { i } ) ,
$$

where $v _ { t } ^ { * } \big ( \pi _ { 0 : t - 1 } ^ { i } \big )$ denotes the optimal continuation value for player i after fixing her policy prefix $\pi _ { 0 : t - 1 } ^ { i }$

At thefinal decision stage, the recursion reduces to

$$
v _ { T - 1 } ^ { * , i } ( \pi _ { 0 : T - 2 } ^ { i } ) = \operatorname* { m a x } _ { \pi _ { T - 1 } ^ { i } } \operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { - i } } u _ { T - 1 } \left( \pi _ { 0 : T - 1 } ^ { i } , \pi _ { 0 : T - 1 } ^ { - i } \right) .
$$

Proof. We prove by backward induction the stronger statement (∗),

$$
v _ { t } ^ { * , i } ( \pi _ { 0 : t - 1 } ^ { i } ) = \operatorname* { m a x } _ { \pi _ { t : T - 1 } ^ { i } } \operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { i } } u \left( [ \pi _ { 0 : t - 1 } ^ { i } , \pi _ { t : T - 1 } ^ { i } ] , \pi _ { 0 : T - 1 } ^ { - i } \right)
$$

for every t.

Base case. $\mathrm { A t } t = T - 1$ , there is no future decision of player i after $\pi _ { T - 1 } ^ { i }$ . Therefore, by definition,

$$
v _ { T - 1 } ^ { * , i } ( \pi _ { 0 : T - 2 } ^ { i } ) = \operatorname* { m a x } _ { \pi _ { T - 1 } ^ { i } } \operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { - i } } u \left( [ \pi _ { 0 : T - 2 } ^ { i } , \pi _ { T - 1 } ^ { i } ] , \pi _ { 0 : T - 1 } ^ { - i } \right) ,
$$

so property (∗) holds.

Induction step. Assume that (∗) holds at time $t + 1$ . Then

$$
\begin{array} { r l } & { v _ { t } ^ { * , i } ( \pi _ { 0 : t - 1 } ^ { i } ) = \underset { \pi _ { t } ^ { i } } { \operatorname* { m a x } } \underset { \pi _ { 0 : T - 1 } ^ { - i } } { \operatorname* { m i n } } v _ { t + 1 } ^ { * , i } ( \pi _ { 0 : t } ^ { i } ) } \\ & { \qquad = \underset { \pi _ { t } ^ { i } } { \operatorname* { m a x } } \underset { \pi _ { 0 : T - 1 } ^ { - i } } { \operatorname* { m i n } } \underset { \pi _ { t + 1 : T - 1 } ^ { i } } { \operatorname* { m a x } } \underset { \pi _ { 0 : T - 1 } ^ { - i } } { \operatorname* { m i n } } u \left( [ \pi _ { 0 : t - 1 } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t + 1 : T - 1 } ^ { i } ] , \pi _ { 0 : T - 1 } ^ { - i } \right) . } \end{array}
$$

We now justify the exchange of the middle min and max. Fix a prefix $\pi _ { 0 : t } ^ { i }$ . Player i chooses a complete continuation policy

$$
\pi _ { t + 1 : T - 1 } ^ { i } ,
$$

while player −i chooses a complete policy

$$
\pi _ { 0 : T - 1 } ^ { - i } .
$$

For fixed $\pi _ { 0 : t } ^ { i }$ , let the payoff of the corresponding pure-strategy pair be

$$
\begin{array} { r } { U \left( \pi _ { t + 1 : T - 1 } ^ { i } , \pi _ { 0 : T - 1 } ^ { - i } \right) = u \left( [ \pi _ { 0 : t } ^ { i } , \pi _ { t + 1 : T - 1 } ^ { i } ] , \pi _ { 0 : T - 1 } ^ { - i } \right) . } \end{array}
$$

When the two players randomize over these complete policies, the expected payoff is bilinear in their mixed strategies. Hence, by the minimax theorem,

$$
\operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { - i } } \operatorname* { m a x } _ { \pi _ { t + 1 : T - 1 } ^ { i } } U = \operatorname* { m a x } _ { \pi _ { t + 1 : T - 1 } ^ { i } } \operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { - i } } U .
$$

This is the same minimax argument used in Delage et al. (2023). Importantly, the argument is applied only after fixing the focal player’s prefix $\pi _ { 0 : t } ^ { i } ;$ the continuation policies of the two players then constitute the strategy spaces of the induced zero-sum normal-form game.

Applying this equality inside the backward recursion gives

$$
\begin{array} { r l } & { v _ { t } ^ { * , i } ( \pi _ { 0 : t - 1 } ^ { i } ) = \underset { \pi _ { t } ^ { i } } { \operatorname* { m a x } } \underset { \pi _ { t + 1 : T - 1 } ^ { i } } { \operatorname* { m a x } } \underset { \pi _ { 0 : T - 1 } ^ { - i } } { \operatorname* { m i n } } u \left( [ \pi _ { 0 : t - 1 } ^ { i } , \pi _ { t } ^ { i } , \pi _ { t + 1 : T - 1 } ^ { i } ] , \pi _ { 0 : T - 1 } ^ { - i } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \pi _ { t : T - 1 } ^ { i } \underset { \pi _ { 0 : T - 1 } ^ { - i } } { \operatorname* { m a x } } \underset { \pi _ { 0 : T - 1 } ^ { - i } } { \operatorname* { m i n } } u \left( [ \pi _ { 0 : t - 1 } ^ { i } , \pi _ { t : T - 1 } ^ { i } ] , \pi _ { 0 : T - 1 } ^ { - i } \right) . } \end{array}
$$

This is precisely property (∗) at time t, completing the induction.

Finally, applying (∗) at t = 0 yields

$$
v _ { 0 } ^ { * , i } = \operatorname* { m a x } _ { \pi _ { 0 : T - 1 } ^ { i } } \operatorname* { m i n } _ { \pi _ { 0 : T - 1 } ^ { - i } } u \left( \pi _ { 0 : T - 1 } ^ { i } , \pi _ { 0 : T - 1 } ^ { - i } \right) .
$$

Thus, the policy generated by the Bellman recursion is a maximin-optimal policy for player i. Since the opponent observes the complete policy prefix of player i, this sequential maximin construction corresponds to the unilateral Subgame Perfect Nash Equilibrium of the one-sided EFG. Hence, the Bellman recursion computes the equilibrium value and an equilibrium strategy for the focal player. □

A few observations follow from this construction. First, by reducing an EFG to a one-sided EFG, we obtain a well-defined notion of a proper subgame for the sequential game. This allows us to formulate and compute a Subgame Perfect Nash Equilibrium, thereby addressing the backwarddependence issue identified by Sokota et al. (2023). In particular, the one-sided construction makes the information available to the two players explicit and ensures that the continuation problem at each stage is formulated as the appropriate sequential zero-sum game.

Second, this result is closely related to Theorem 3 of Escudie et al. (2025). While our proof is formulated directly in terms of the decision tree and does not explicitly use the occupancy-state representation, the correspondence between the two formulations is immediate: the nodes of the one-sided EFG can equivalently be represented by the corresponding occupancy states. Thus, the construction provides an alternative, more explicit interpretation of the same underlying reduction.

The main distinction is the scope in which this construction is required. The one-sided reduction is necessary when dealing with sequential games, such as EFGs or sequential POSGs, because future decisions may depend on previously revealed decisions and therefore cannot in general be optimized independently through the standard simultaneous-move Bellman recursion. In contrast, most of the games considered in this paper are simultaneous-move POSGs. In that setting, the standard dynamic-programming formulation already corresponds to the appropriate subgame-perfect solution concept, and the additional one-sided reduction is therefore unnecessary. The relationship between these two cases is illustrated in Figure 4.

We work with the uninformed planner introduced by Escudie et al. (2025) because deviations in the past cannot depend on the public history. If we were to use an informed planner instead, the resulting optimization would allow a different deviation for each public history. This would effectively give the best response access to the public history before the corresponding decision is made, which is not available to the player and would therefore be inconsistent with the information structure of the game.

## C.4 EXPERIMENTS

We extend Algorithm 1 to extensive-form games by adapting the local policy-improvement step to account for deviations from decisions made at earlier stages of the game. In particular, the linear program is modified to retain the relevant information about past decisions when computing the current deviation, allowing the local improvement procedure to operate directly on the extensiveform representation.

To obtain a deviation model consistent with the intended information structure, we include public histories in the support of the belief representation. The deviation is therefore defined with respect to the information available to the player rather than separately for each public history. Without this restriction, the optimization could effectively select a different deviation for every public history, leading to an overly pessimistic evaluation that does not correspond to the intended player model.

We implemented this extension and evaluated it on Kuhn Poker. The resulting algorithm exhibits convergence to the expected equilibrium behavior in our experiments. However, the current implementation is primarily intended as a proof-of-concept: the code requires further cleanup and validation before the experimental results can be presented as part of the main empirical evaluation. We therefore leave a more systematic experimental study of the extensive-form extension to future work.

## D EXPERIMENTS

## D.1 BASELINES

PBVI in zs-SG In a POMDP, the value function is defined over the belief simplex and is piecewiselinear and convex, making exact value iteration intractable. PBVI addresses this issue by restricting computation to a finite set of representative beliefs. At each iteration, Bellman backups are performed only on this set, yielding an approximate value function that is accurate in the vicinity of the sampled beliefs while remaining tractable.

PBVI is inherently a single-agent method. In adversarial settings, it is therefore applied to a onesided (focal-planner) surrogate, where one player is optimized while the opponent is treated as part of the environment. Two levels of sampling arise. First, beliefs are expanded via forward simulation to cover reachable regions of the belief space. Second, in the one-sided reduction, the opponent is simulated (e.g., via randomized policies), so that the focal player optimizes against sampled behaviors. This relies on the assumption that relevant opponent responses lie near those induced by sampling.

Each PBVI backup requires a maximization over actions and observations, which in adversarial settings leads to linear programs for worst-case evaluation. The number of α-vectors grows rapidly with iterations, increasing the size of these programs. Consequently, the Bellman operator quickly becomes intractable despite the point-based approximation.

CFR in POSG. Counterfactual Regret Minimization (CFR) is an iterative algorithm for solving extensive-form games (EFGs) that minimizes counterfactual regret at each information set, yielding convergence to a Nash equilibrium in two-player zero-sum settings. At each iteration, CFR evaluates the current strategy profile, computes counterfactual values, and updates strategies via regret matching.

<table><tr><td>Name in EFG</td><td>Symbol</td><td>Name in POSG</td><td>Symbol</td></tr><tr><td>History</td><td> $h$ </td><td>State and Joint history</td><td> $\left( { { s _ { t } } , { h _ { t } } } \right)$ </td></tr><tr><td>i&#x27;s information set</td><td>I</td><td>i&#x27;s private history</td><td> $h _ { i }$ </td></tr><tr><td>Strategy profile</td><td> $\sigma$ </td><td>Joint policy</td><td> $\pi$ </td></tr><tr><td>i’s private strategy</td><td> $\sigma _ { i }$ </td><td>i&#x27;s private policy</td><td> $\pi _ { i }$ </td></tr><tr><td>Players’ contribution except i</td><td> $\pi _ { i } ^ { \sigma } ( I )$ </td><td>Players&#x27; contribution except i</td><td> $\operatorname* { P r } ( h _ { i } \mid \pi , b _ { 0 } )$ </td></tr><tr><td>Players’ contribution</td><td> $\pi _ { i } ^ { \sigma } ( h )$ </td><td>Counterfactual state</td><td> $c _ { i } ( s , h _ { - i } )$ </td></tr><tr><td>except i Counterfactual value of σ at h</td><td> $v _ { i } ( \sigma , h )$ </td><td>Value of π at  $( s , h )$ </td><td> $v _ { \mathcal { M } } ^ { \pi } ( s , h )$ </td></tr><tr><td>Deviation at I</td><td> $\sigma _ { I  a }$ </td><td>Deviation at  $h _ { i }$ </td><td> $\pi _ { h _ { i } \to a _ { i } }$ </td></tr></table>

Table 3: Translation from EFG to POSG. For more information, we refer the reader to (Kovaˇrík et al., 2022). We connect the history h in EFG to the pair $( s , h )$ in POSG since the reward is not associated to the history but to the state in POSG. The quantity $v _ { \mathcal { M } } ^ { \pi } ( s , h )$ is formally defined in (Cunha et al., 2023), we refer the reader for more information. It is simply the expected cumulated value a player can get at this pair.

The connection between POSGs and EFGs has been established in (Kovaˇrík et al., 2022), laying the groundwork for the application of extensive-form techniques such as CFR. Notably, the theoretical objects introduced in our paper naturally embed CFR and yield a refined variant tailored to POSGs. To make this connection explicit, Table 3 presents a correspondence between standard EFG notions and their POSG counterparts, thereby providing a bridge for transferring the algorithm. The relevant notation is summarized in Table 3.

From (Zinkevich et al., 2007; Neller & Lanctot, 2013), we adapt the key ingredients of CFR to the POSG setting. We first define the (instantaneous) regret at $\left( { { s _ { t } } , { h _ { t } } } \right)$ for player −i when deviating only at $h _ { - i , t }$ to action $\boldsymbol { a } _ { - i , t } \boldsymbol { : }$

$$
\begin{array} { r } { r _ { \mathcal { M } } ( s _ { t } , h _ { t } , a _ { i , t } ) = v _ { \mathcal { M } } ^ { \pi _ { h _ { i , t } }  a _ { i , t } } ( s _ { t } , h _ { t } ) - v _ { \mathcal { M } } ^ { \pi } ( s _ { t } , h _ { t } ) . } \end{array}
$$

In contrast to EFGs, where regret is indexed by histories $h ,$ in POSGs it is defined over $\left( { { s _ { t } } , { h _ { t } } } \right)$ since rewards depend on the (hidden) state $s _ { t } .$ . The history remains necessary to condition the belief over states.

Aggregating over histories that share the same private history $h _ { i , t } :$ yields a vector-valued regret:

$$
\mathbf { r } _ { \mathcal { M } } ^ { h _ { i } , t  a _ { i } , t } ( s _ { t } , h _ { i , t } ) = v _ { \mathcal { M } } ^ { \pi _ { h _ { - } i , t }  a _ { - } { i } , t } ( s _ { t } , h _ { i , t } , h _ { - i , t } ) - v _ { \mathcal { M } } ^ { \pi } ( s _ { t } , h _ { i , t } , h _ { - i , t } ) ,
$$

which is a vector over $\mathcal { C } _ { i }$ (one coordinate per state). Using the conditional state $c _ { i }$ notation, we recover a scalar regret:

$$
R _ { \mathcal { M } } ^ { h _ { - i , t }  a _ { - i , t } } = \mathbf { r } ^ { h _ { - i , t }  a _ { - i , t } } \cdot c _ { i } .
$$

This corresponds to the EFG quantity

$$
r _ { E F G } ( I , a ) = \sum _ { h \in I } \bigl ( v _ { i } ( \sigma _ { I  a } , h ) - v _ { i } ( \sigma , h ) \bigr ) ,
$$

where the summation over histories is replaced by expectation under the counterfactual occupancy. The policy is then updated via regret matching:

$$
\pi _ { - i , \mathrm { n e w } } ( a _ { - i , t } \mid h _ { - i , t } ) = \frac { [ R _ { \mathcal { M } } ^ { h _ { - i , t }  a _ { - i , t } } ] _ { + } } { \sum _ { a _ { - i , t } ^ { \prime } } [ R _ { \mathcal { M } } ^ { h _ { - i , t }  a _ { - i , t } ^ { \prime } } ] _ { + } } .
$$

The regret of player $- i$ is inherently vector-valued, as it is defined conditionally on the hidden state. The conditional state $c _ { i }$ acts as a belief, projecting this vector onto a scalar quantity, exactly as counterfactual reach probabilities do in EFGs. This establishes a direct connection with the focalplanner view: regret minimization operates on belief-weighted values, consistent with optimizing a surrogate game for a fixed opponent.

Table 4: Snapshot of empirical results. For each game, horizon and algorithm, we report time, value and exploitability ϵ associated with the computed ϵ-Nash equilibrium. OOT occurs when the target exploitability threshold cannot be reached within the prescribed time limit, while OOM means that the algorithms ran out of memory. magenta = lowest exploitability value per benchmark, while ‘− means the time budget for computing the exploitability has been exceeded.
<table><tr><td>Problem</td><td>T</td><td colspan="2">CPI (Ours)</td><td colspan="2">PBVI</td><td colspan="3">HSVI</td><td colspan="2">CFR+</td></tr><tr><td></td><td></td><td>t</td><td>val. €</td><td>t</td><td>val. €</td><td>t</td><td>val. €</td><td></td><td>t</td><td>val. €</td></tr><tr><td></td><td>2 3</td><td>0.3 11</td><td>0.078 0. 0.097 0.</td><td>0.2 24</td><td>0.078 0.</td><td>0.4</td><td>0.078 0.</td><td></td><td>0.3</td><td>0.078 0.</td></tr><tr><td></td><td>4</td><td>23</td><td>0.11 0.</td><td></td><td>0.096 0.</td><td>70</td><td>0.097 0.</td><td></td><td>7</td><td>0.095 0.</td></tr><tr><td></td><td>5</td><td></td><td></td><td>65 31</td><td>0.11 0.01</td><td>OOT</td><td>OOT OOT</td><td></td><td>25</td><td>0.11 0.01</td></tr><tr><td>MABC</td><td></td><td>29</td><td>0.12 0.</td><td></td><td>0.12 0.</td><td>OOT</td><td>OOT OOT</td><td></td><td>69</td><td>0.12 0.01</td></tr><tr><td></td><td>7</td><td>97</td><td>0.14 0.01</td><td>190</td><td>0.14 0.0</td><td>OOT</td><td>OOT OOT</td><td></td><td>92</td><td>0.14 0.02</td></tr><tr><td></td><td>10</td><td>100</td><td>0.18 0.01</td><td>637</td><td>0.17 0.01</td><td>OOT</td><td>OOT OOT</td><td></td><td>122</td><td>0.18 0.02</td></tr><tr><td></td><td>15 30</td><td>143</td><td>0.22 0.01</td><td>668</td><td>0.22 0.02</td><td>OOT</td><td>OOT OOT</td><td></td><td>222</td><td>0.22 0.03</td></tr><tr><td></td><td></td><td>360</td><td>0.36 0.01</td><td>OOT</td><td>OOT OOT</td><td>OOT</td><td>OOT OOT</td><td></td><td>433</td><td>0.36 0.06</td></tr><tr><td></td><td>2</td><td>0.01</td><td>0.4 0.</td><td>2</td><td>0.39 0.</td><td>0.1</td><td>0.4 0.</td><td></td><td>0.06</td><td>0.4 0.005</td></tr><tr><td></td><td>3 4</td><td>0.02</td><td>0.6 0.</td><td>18</td><td>0.6 0.</td><td>1</td><td>0.6 0.</td><td></td><td>0.1</td><td>0.6 0.001</td></tr><tr><td></td><td></td><td>0.04</td><td>0.8 0.</td><td>34</td><td>0.77 0.</td><td>4</td><td>0.8 0.</td><td></td><td>0.2</td><td>0.8 0.</td></tr><tr><td>MATCHING</td><td>5 7</td><td>0.04</td><td>1. 0.</td><td>52</td><td>0.95 0.</td><td>60</td><td>1.0.</td><td></td><td>0.25</td><td>1.0.</td></tr><tr><td>PENNIES</td><td>10</td><td>13</td><td>1.4 0.</td><td>180</td><td>1.31 0.03</td><td>OOT</td><td>OOT OOT</td><td></td><td>15</td><td>1.39 0.01</td></tr><tr><td></td><td></td><td>92</td><td>2.0.</td><td>OOT</td><td>OOT OOT</td><td>OOT</td><td>OOT OOT</td><td></td><td>103</td><td>1.99 0.02</td></tr><tr><td></td><td>25 50</td><td>15</td><td>5.0.</td><td>OOT</td><td>OOT OOT</td><td>OOT</td><td>OOT OOT</td><td></td><td>132</td><td>5.0.02</td></tr><tr><td></td><td></td><td>217</td><td>10.0.</td><td>OOT</td><td>OOT OOT</td><td>OOT</td><td>OOT OOT</td><td></td><td>718</td><td>10.0.03</td></tr><tr><td></td><td></td><td>0.4</td><td>0.26 0.</td><td>0.4</td><td>0.26 0.</td><td>1.2</td><td>0.26 0.</td><td></td><td>2.8</td><td>0.26 0.004</td></tr><tr><td></td><td>23</td><td>17</td><td>0.32 0.01</td><td>78</td><td>0.32 0.</td><td>120</td><td>0.32 0.</td><td></td><td>75</td><td>0.32 0.01</td></tr><tr><td></td><td>4</td><td>42</td><td>0.36 0.03</td><td>56</td><td>0.350.01</td><td>OOT</td><td>OOT OOT</td><td></td><td>104</td><td>0.35 0.04</td></tr><tr><td>RECYCLING</td><td>5</td><td>213</td><td>0.4 0.05</td><td>98</td><td>0.38 0.05</td><td>OOT</td><td>OOT OOT</td><td></td><td>177</td><td>0.38 0.05</td></tr><tr><td></td><td>7</td><td>271</td><td>0.5 0.06</td><td>1481</td><td>0.480.04</td><td>OOT</td><td>OOT OOT</td><td></td><td>503</td><td>0.49 0.12</td></tr><tr><td></td><td>10</td><td>505</td><td>0.63 0.07</td><td>3232</td><td>0.6 0.06</td><td>OOT</td><td>OOT OOT</td><td></td><td>1151</td><td>0.64 0.15</td></tr><tr><td></td><td>15</td><td>703</td><td>0.87 0.08</td><td>OOT</td><td>OOT OOT</td><td>OOT</td><td>OOT OOT</td><td></td><td>2415</td><td>0.86 0.23</td></tr><tr><td></td><td>2</td><td>0.1</td><td>-0.4 0.</td><td>0.1</td><td>-0.4 0.</td><td>0.2</td><td>-0.4 0.</td><td></td><td>0.1</td><td>-0.4 0.0</td></tr><tr><td></td><td>3</td><td>0.2</td><td>-0.56 0.</td><td>0.3</td><td>-0.56 0.</td><td>500</td><td>-0.56 0.</td><td></td><td>5</td><td>-0.56 0.</td></tr><tr><td>ADVERSARIAL</td><td>4</td><td>16</td><td>-0.75 0.01</td><td>46</td><td>-0.76 0.02</td><td>OOT</td><td></td><td>OOT OOT</td><td>76</td><td>-0.75 0.03</td></tr><tr><td>TIGER</td><td>5</td><td>82</td><td>-0.96 0.05</td><td>141</td><td>-0.960.05</td><td>OOT</td><td>OOT OOT</td><td></td><td>87</td><td>-0.96 0.07</td></tr><tr><td></td><td>7</td><td>248</td><td>-1.36 0.13</td><td>866</td><td>-1.36 0.1</td><td>OOT</td><td>OOT OOT</td><td></td><td>172</td><td>-1.35 0.13</td></tr><tr><td></td><td>10</td><td>824</td><td>-1.97 0.24</td><td>956</td><td>-1.980.18</td><td>OOT</td><td>OOT OOT</td><td></td><td>556</td><td>-1.95 0.3</td></tr></table>

Our framework can be interpreted as performing regret minimization directly in the occupancy space. Instead of explicitly maintaining regrets per history, we operate on regret functions over counterfactual occupancy states, and policy updates correspond to regret matching under these occupancy-induced weights. This yields a compact implementation that avoids explicit enumeration of histories.

## D.2 IMPLEMENTATION DETAILS

The results presented in this paper were run on CPU (AMD Ryzen 7 PRO).

Table 5: A snapshot of empirical results with no pruning mechanism applied (see Escudie et al. (2025), Dibangoye et al. (2016a)). For each game, horizon and algorithm, we report time, value and exploitability ϵ associated with the computed ϵ-Nash equilibrium. OOT occurs when the target exploitability threshold cannot be reached within the prescribed time limit, while OOM means that the algorithms ran out of memory. magenta = lowest exploitability value per benchmark, while ‘− means the time budget for computing the exploitability has been exceeded.
<table><tr><td>Problem</td><td>T</td><td colspan="2">CPI (Ours)</td><td colspan="2">PBVI</td><td colspan="2">HSVI</td><td colspan="2">CFR+</td></tr><tr><td></td><td></td><td>t</td><td>val. €</td><td>t</td><td>V €</td><td>t</td><td>val. €</td><td>t</td><td>val. €</td></tr><tr><td rowspan="4">MABC</td><td>23</td><td>0.3</td><td>0.078 0.</td><td>0.2</td><td>0.077 0.</td><td>0.3</td><td>0.078 0.</td><td>0.3</td><td>0.078 0.003</td></tr><tr><td></td><td>5</td><td>0.097 0.</td><td>19</td><td>0.097 0.</td><td>70</td><td>0.097 0.</td><td>12</td><td>0.096 0.002</td></tr><tr><td>4 5</td><td>38</td><td>0.11 0.001</td><td>50</td><td>0.1 0.001</td><td>OOT</td><td>OOT OOT</td><td>73</td><td>0.11 0.003</td></tr><tr><td></td><td>2176</td><td>0.12 0.01</td><td>1365</td><td>0.12 0.013</td><td>OOT</td><td>OOT OOT</td><td>1614</td><td>0.12 0.005</td></tr><tr><td rowspan="4">MATCHING PENNIES</td><td>2</td><td>0.01</td><td>0.4 0.</td><td>0.13</td><td>0.38 0.</td><td>0.1</td><td>0.4 0.</td><td>0.01</td><td>0.4 0.</td></tr><tr><td>3</td><td>0.05</td><td>0.6 0.</td><td>9</td><td>0.6 0.</td><td>1</td><td>0.6 0.</td><td>0.2</td><td>0.6 0.</td></tr><tr><td>4 5</td><td>0.5</td><td>0.8 0.</td><td>250</td><td>0.77 0.</td><td>4</td><td>0.8 0.</td><td>2</td><td>0.8 0.005</td></tr><tr><td></td><td>11</td><td>1.0 0.003</td><td>3207</td><td>1.14 0.12</td><td>60</td><td>1.0.</td><td>5</td><td>1.0.006</td></tr><tr><td rowspan="4">RECYCLING</td><td></td><td>0.4</td><td>0.26 0.</td><td>0.1</td><td>0.26 0.</td><td>1.2</td><td>0.26 0.</td><td>1.5</td><td>0.26 0.01</td></tr><tr><td>23</td><td>14</td><td>0.32 0.</td><td>286</td><td>0.32 0.</td><td>120</td><td>0.32 0.</td><td>131</td><td>0.32 0.01</td></tr><tr><td>4 5</td><td>681</td><td>0.35 0.</td><td>418</td><td>0.34 0.01</td><td>OOT</td><td>OOT OOT</td><td>848</td><td>0.35 0.03</td></tr><tr><td></td><td>5146</td><td>0.36 0.05</td><td>OOT</td><td>OOT OOT</td><td>OOT</td><td>OOT OOT</td><td>7611</td><td>0.36 0.1</td></tr><tr><td rowspan="4">ADVERSARIAL TIGER</td><td></td><td>0.</td><td>-0.4 0.</td><td>0.</td><td>-0.4 0.</td><td>0.</td><td>-0.4 0.</td><td>0.</td><td>-0.4 0.002</td></tr><tr><td>23</td><td>2</td><td>-0.56 0.</td><td>0.17</td><td>-0.56 0.</td><td>500</td><td>-0.56 0.</td><td>6</td><td>-0.57 0.007</td></tr><tr><td>4 5</td><td>146</td><td>-0.76 0.01</td><td>213</td><td>-0.78 0.04</td><td>OOT</td><td>OOT OOT</td><td>171</td><td>-0.73 0.03</td></tr><tr><td></td><td>2082</td><td>-0.96 0.02</td><td>3237</td><td>-0.96 0.06</td><td>OOT</td><td>OOT OOT</td><td>2583</td><td>-0.96 0.02</td></tr></table>