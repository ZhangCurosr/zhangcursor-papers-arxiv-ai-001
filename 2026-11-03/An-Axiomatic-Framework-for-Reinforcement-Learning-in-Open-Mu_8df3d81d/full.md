# An Axiomatic Framework for Reinforcement Learning in Open Multi-Agent Systems

Jianhong Wang\* INFORMED-AI Hub University of Bristol

Samuel Kaski† ELLIS Institute Finland

Yang Li\* School of Computer Science Shanghai Jiao Tong University

Jonathan Lawry School of Engineering Mathematics and Technology University of Bristol

## Abstract

Open multi-agent systems, where only a subset of agents are controllable, are central to real-world domains such as smart grids, yet lack a principled foundation for reinforcement learning algorithm design. We study this setting through n-agent ad hoc teamwork (NAHT), a recent framework for learning in open multi-agent environments. We propose an axiomatic framework that derives value function learning from the Shapley axioms (Efficiency, Symmetry and Linearity), treating them as structural constraints rather than heuristic design choices. This formulation builds on a novel NAHT decomposition, which characterises the fundamental difference between open and fully controllable multi-agent systems, and establishes a cooperative game-theoretic foundation for learning. We further show that enforcing these axioms on learning individual value functions recovers the Shapley value for Dec-POMDPs, providing a principled bridge between reinforcement learning and classical cooperative game theory. Building on this framework, we introduce Shapley Machines and Banzhaf Machines, two generic algorithmic templates that can be instantiated with standard reinforcement learning methods. When applied to strong base algorithms such as IPPO and POAM, our methods consistently improve learning efficiency and generalisation to unseen agent types and conventions across NAHT benchmark tasks. The code of experiments is published on https://github.com/hsvgbkhgbv/axiomatic-rl-naht.

## 1 Introduction

Multi-agent systems (MAS) have become a central paradigm for modelling real-world domains such as smart grids [Wang et al., 2021], railway networks [Zhang et al., 2024], and swarm robotics [Nayak et al., 2023, Li et al., 2025], where multiple agents must coordinate under uncertainty. Reinforcement learning (RL) [Sutton and Barto, 2018], and in particular cooperative multi-agent reinforcement learning (CMARL), has shown promise in solving such problems. However, existing CMARL methods typically assume that all agents are controllable and that team composition is fixed, which we call the closed multi-agent systems; these assumptions rarely hold in practice.

In many real-world scenarios, agents must operate in open multi-agent systems, where only a subset of agents is controllable and the remaining agents may be unknown or dynamically changing. This setting introduces two fundamental challenges: (i) generalisation to previously unseen teammate behaviours, and (ii) openness to varying team compositions. To address these challenges, recent work has introduced the n-agent ad hoc teamwork (NAHT) [Wang et al., 2025], which extends CMARL to settings with partially controllable and dynamically formed teams.

Although the NAHT objective is to maximise the team value over varying team compositions, this objective does not specify how agents can form cooperation in an open system with both controlled and uncontrolled agents. From the perspective of CMARL, an agent is represented by its individual value. This raises a fundamental question:

How should individual value functions be structured and learned to support cooperation in open multi-agent systems?

In this paper, we address this question from the perspective of cooperative game theory. Specifically, we introduce an NAHT decomposition that represents a team value by two parts: internal cooperation among controlled agents and external teamwork involving uncontrolled agents. This decomposition reveals a structural distinction between open and closed multi-agent systems, which is not innately captured by the NAHT objective.

Building on this structure, we propose an axiomatic framework to derive individual value functions, grounded in the axioms of the Shapley value [Shapley, 1953, Dubey, 1975]. Rather than constructing individual value functions directly by the explicit form of the Shapley value, we derive the individual value function learning by enforcing its axioms (Efficiency, Symmetry and Linearity) as structural constraints. This yields a principled approach to designing learning algorithms, where individual value functions are shaped to satisfy some properties, such as cooperative game-theoretic properties applied in this paper. Notably, we prove in theory that enforcing these axioms on learning individual value functions recovers the Shapley value for Dec-POMDPs [Oliehoek et al., 2016], while relaxing specific axioms leads to alternative payoff allocations, such as the Banzhaf index.

More importantly, our axiomatic framework also reveals a novel connection between cooperative game theory and reinforcement learning. In particular, we show that the Linearity axiom induces a structure equivalent to the truncated TD(λ) [Cichosz, 1994], where the number of n-step return components can be estimated by the number of non-empty coalitions of cooperative games. This provides a theoretical explanation for the role of n-step returns and suggests a new perspective on the design of temporal-difference learning methods in the NAHT (see Section 5.2).

Since NAHT includes single-agent RL with an environment of other uncontrolled agents and CMARL with fully controllable agents as its two extreme cases, the above result also extends to both settings.

Based on this axiomatic framework, we introduce two generic algorithmic templates: Shapley Machines, which enforce all axioms, and Banzhaf Machines, which relax the Efficiency axiom. These templates can be instantiated with existing RL algorithms, such as IPPO [De Witt et al., 2020] and POAM, to yield practical methods for solving the NAHT.

We evaluate the proposed methods on modified MPE and SMAC benchmarks adapted to the NAHT setting [Wang et al., 2025]. Our results show that the proposed framework consistently improves learning efficiency and generalisation to unseen agent types and conventions. Furthermore, we demonstrate that relaxing specific axioms can lead to improved performance in certain regimes, highlighting the flexibility of the proposed axiomatic perspective. Due to the limit of space, we leave the related work section to Appendix A.

## 2 Background

## 2.1 N-Agent Ad Hoc Teamwork

In this section, we describe a problem setting for open multi-agent systems, referred to as n-agent ad hoc teamwork (NAHT) [Wang et al., 2025]. The main challenges of NAHT are as follows: (1) cooperating with potentially unknown types of teammates (generalization), and (2) coping with a varying number of uncontrolled teammates (openness). A decentralized partially observable Markov decision process (Dec-POMDP) [Oliehoek et al., 2016] is considered to formalize the problem. There is a team of agents M whose size is denoted as M, a state space S, a joint action space $\mathcal { A } = \times _ { i \in \mathcal { M } } \mathcal { A } _ { i }$ a per-agent observation space $\mathcal { O } _ { i } ,$ a transition function $\bar { P } _ { T } : \mathcal { S } \times \bar { \mathcal { A } }  \Delta ( \mathcal { S } ) ,$ a common reward function $R _ { t } : S \times \mathcal { A }  \bar { \mathbb { R } }$ , a discount factor $\gamma \in [ 0 , 1 ]$ and an episode length $T \in \mathbb { Z } ^ { + }$ . Each agent receives observations via an observation function $O _ { i } : S \times A  \Delta ( O _ { i } )$ $\mathcal { H } _ { i }$ is defined as an agent ¿'s space of observation and action histories. Each agent is equipped with a policy $\pi _ { i } : \mathcal { H } _ { i }  \Delta ( \bar { \mathcal { A } } _ { i } )$

In this problem, there are two groups of agents: a set of controlled agents denoted as $\mathcal { C }$ whose size is denoted as $N _ { \ast }$ , and a set of uncontrolled agents denoted as $\mathcal { U }$ whose size is denoted as $M - N . \operatorname { A s }$ NAHT is an open multi-agent system, at the beginning of a multi-agent interaction process, a team of agents $\mathcal { M }$ are randomly drawn from the controlled agent pool and the uncontrolled agent pool. For conciseness, each controlled agent is characterised by its policy, and the set of controlled agents can be expressed as $\mathcal { C } ( \theta ) = \{ \pi _ { i } ^ { \theta } \} _ { i = 1 } ^ { N }$ . The above sampling procedure is denoted as $X ( \mathcal { U } , \mathcal { C } ( \theta ) )$ . The aim of NAHT is learning parameters θ to solve the following optimization problem:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \pi ^ { ( \mathcal { M } ) } \sim X ( \mathcal { U } , \mathcal { C } ( \theta ) ) } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } R _ { t } \right] ,\tag{1}
$$

where $\pi ^ { ( \mathcal { M } ) }$ denotes the joint policy of a team of agents $\mathcal { M } .$ Note that different number of controlled agents $N$ would lead to various teams of agents $\mathcal { M } ;$ the expectation in Eq. 1 is usually approximated by a team value which we will specify in the following section.

## 2.2 Cooperative Games and Shapley Value

Representation of Cooperative Games. A cooperative game is usually described as a characteristic function $v : 2 ^ { \mathcal { M } } \to \mathbb { R } _ { > 0 } ^ { \overline { { \mathbf { \alpha } } } }$ , where M is a team of agents. An arbitrary cooperative game v can be uniquely represented by a set of basis games $\{ v _ { C } ^ { 1 } | { \overset { \vartriangle } { \varnothing } } \neq C \subseteq { \mathcal { M } } \}$ [Dubey, 1975], as delineated in Definition 2.1. In this paper, we focus on investigating how cooperative games are approximated over a team of agents $\mathcal { M } .$ Therefore, we only consider the function values $\mathsf { \Pi } _ { v ( \mathcal { M } ) }$ and $\overset { \bullet } { v } _ { C } ^ { 1 } ( \mathcal { M } )$ for a team of agents $\mathcal { M }$ in the presentation of this work $\cdot ^ { 2 }$ As per previous work in the MARL [Wang et al., 2020], we consider the superadditive game as the game class of ${ \mathcal { G } } ,$ which is suitable for a cooperative multi-agent task, such as NAHT (see Appendix B.2 for more details).

Definition 2.1 (Cooperative Games [Dubey, 1975]). We define a function $v _ { C } ^ { z } ( D )$ for an arbitrary $z \in \mathbb { R }$ such that: If a coalition $C \subseteq D , v _ { C } ^ { z } ( D ) = z ,$ otherwise, $v _ { C } ^ { z } ( D ) = { \bf \check { 0 } } .$ For a cooperative game described by a characteristic function $v : 2 ^ { \mathcal { M } } \to \mathbb { R } _ { > 0 }$ , we can describe a set of such games over a team of agents M by a set of basis games $\{ v _ { C } ^ { 1 } \mid \bar { \varnothing } \neq C \subseteq { \mathcal { M } } \}$ , such that $\mathcal { G } = \{ w \mid w =$ $\begin{array} { r } { \sum k _ { C } v _ { C } ^ { 1 } , k _ { C } \in \mathbb { R } , \emptyset \ne C \subseteq M \} . ^ { \frac { 2 } { 2 } } } \end{array}$

Shapley Value. Shapley value is a multidimensional linear transformation $\phi : \mathcal { G }  \mathbb { R } ^ { M }$ defined on the set of cooperative game ${ \mathcal { G } } ,$ given that there are M agents in total. Each dimension of φ indicates the payoff allocation to an agent. It has been proved that Shapley value is a unique payoff allocation function on the set of cooperative games G which satisfies the following axioms: Efficiency, Symmetry and Additivity [Dubey, 1975],4 highlighted in the following Theorem 2.2:

Theorem 2.2 (Axioms of Shapley Value [Dubey, 1975]). Shapley value is a unique payoff allocation function on the cooperative game space ${ \mathcal { G } } ,$ satisfying Efficiency, Symmetry and Additivity.

## 3 N-Agent Ad Hoc Teamwork Decomposition

To understand how learning differs in open multi-agent systems, we first characterise the cooperativegame structure of team values of the NAHT. Recall that in the NAHT a team consists of two groups of agents: a set of controlled agents ${ \mathcal { C } } ,$ whose policies are optimized, and a set of uncontrolled agents U, whose behaviours are unknown and may vary across episodes. Unlike the closed multi-agent systems, the performance of a team in the NAHT depends not only on coordination among controlled agents, but also on their ability to adapt to diverse and potentially unseen teammates. To make this distinction explicit, we represent a team value at a given state in the NAHT as a state-specific cooperative game that is the weighted sum of non-empty coalition basis games, from the perspective of cooperative game theory, as follows:

Definition 3.1 (State-Specific Cooperative Games). We define a function $v _ { C , s } ^ { z } ( D )$ , for any real value z ∈ R and state $s \in S$ , such that if a coalition $C \subseteq D , v _ { C , s } ^ { z } ( D ) = z ,$ otherwise, $v _ { C , s } ^ { z } ( D ) = 0$ It is not difficult to observe that we have $v _ { C , s } ^ { z } = z v _ { C , s } ^ { 1 } ,$ where $v _ { C , s } ^ { 1 }$ is a state-specific basis game for state s. Similarly, a set of generalised cooperative games to state s, referred to as state-specific cooperative games, is represented as: $\begin{array} { r } { \mathcal { G } ( s ) \dot { = } \{ w _ { s } \ | \ \tilde { w } _ { s } = \sum k _ { C } v _ { C , s } ^ { 1 } , \dot { k _ { C } } \in \mathbb { R } , \emptyset \neq C \subseteq \dot { \mathcal { M } } \} } \end{array}$ . In other words, an arbitrary state-specific game $v _ { s } \in \mathcal { G } ( s )$ can be represented as: $\begin{array} { r } { v _ { s } = \sum k _ { C } v _ { C , s } ^ { 1 } } \end{array}$

Definition 3.2 (Game Structure for NAHT). $\mathcal { G } _ { N A H T } = \times _ { s \in S } \mathcal { G } ( s )$ denotes the set of all NAHT processes, where a team value of an NAHT process is represented as a state-specific cooperative game $v _ { s } \in \mathcal G ( s )$ . Each element belonging to Gaûr is a tuple of state-specific cooperative games, defined by a team of agents M, a state space S, and weightings $k _ { C }$ associated with the state-specific basis games of all non-empty coalitions $\varnothing \neq C \subseteq { \mathcal { M } }$

We decompose a team value into two parts. Based on generalisation of the cooperative game representation from the classic cooperative game as shown in Definition 3.1 and the game structure for the NAHT in Definition 3.2, we define a state-specific cooperative game $v _ { s } ,$ as follows:

$$
v _ { s } = \underbrace { \sum _ { C _ { I } \in \mathcal { P } _ { + } ( \mathcal { C } ) } k _ { C _ { I } } v _ { C _ { I } , s } ^ { 1 } } _ { \mathrm { I n t e r n a l ~ C o o p e r a t i o n } } + \underbrace { \sum _ { C _ { E } \in \mathcal { P } _ { + } ( \mathcal { M } ) \backslash \mathcal { P } _ { + } ( \mathcal { C } ) } k _ { C _ { E } } v _ { C _ { E } , s } ^ { 1 } } _ { \mathrm { E x t e r n a l ~ T e a m w o r k } } ,\tag{2}
$$

where M = C∪U denotes the full team, $\mathcal { P } _ { + } ( \cdot )$ indicates the powerset of a set excluding the empty set, and each term corresponds to a contribution from a subset (coalition) of agents. The first term captures internal cooperation, i.e., interactions solely among controlled agents C. This corresponds to the standard task in the CMARL, where all agents are optimised jointly. The second term captures external teamwork, i.e., interactions between controlled agents C and uncontrolled agents U, along with interactions among uncontrolled agents U. The second term is unique to open multi-agent systems, and reflects the need to coordinate with agents whose behaviours are not controlled and may vary across episodes. When all agents are controlled, $\mathfrak { i . e . , } \mathcal { C } = \mathcal { M }$ , the external teamwork part vanishes, and the decomposition reduces to internal cooperation, recovering the fully controllable setting of the CMARL.

Example 3.3. As shown in Figure 1, suppose we have a team of agents $\begin{array} { r } { \mathcal { M } \ = \ \{ 1 , 2 \} } \end{array}$ for a state s, where $\begin{array} { r l } { \mathcal { C } } & { { } = \ \left\{ 1 \right\} } \end{array}$ and $\mathcal { U } = \{ 2 \}$ . θ is the angle between two vectors representing two situations, such that $w _ { s } \in W = s p a n \{ v _ { \{ 1 \} , s } ^ { 1 } \}$ and $u _ { s } \in \mathcal G ( s ) =$ span $\{ v _ { \{ 1 \} , s } ^ { 1 } , v _ { \{ 2 \} , s } ^ { 1 } , v _ { \{ 1 , 2 \} , s } ^ { 1 } \}$ . It is obvious that $w _ { s } ( \mathcal { M } )$ , as a projection of $u _ { s } ( \mathcal { M } )$ , loses information about the external teamwork.

![](images/83a6634eaf68289de63b7ea18da6dd96c9bed48aef8b536d5c6279e8353b261a.jpg)

Implication. This decomposition highlights a fundamental difference between NAHT and CMARL: In the CMARL, all agents are controllable, and learning focuses exclusively on internal cooperation. In the NAHT, learning must additionally account for external teamwork with uncontrolled agents. Ignoring the external component effectively reduces the problem to a closed multi-agent system, which introduces bias and limits generalisation to unseen team compositions. The NAHT decomposition therefore provides a structural characterisation of openness, and it will serve as the foundation for the axiomatic framework introduced as follows.

## 4 From Shapley Axioms to Learning Algorithms

Building on the NAHT decomposition, we now derive an axiomatic framework for learning in open multi-agent systems. Our approach

Figure 1: Example of statespecific cooperative games. The red and blue vectors are two games represented by different basis game sets, denoted by $w _ { s } ( \mathcal { M } )$ and $u _ { s } ( \mathcal { M } )$ . Their difference is measured by the θ. The shading area in red indicates the range of generated cooperative games with that $k _ { C } \dot { \geq } 0 \mathrm { a n d } \breve { \sum } ( k _ { C } ) ^ { 2 } = 1$

is based on the axioms of the Shapley value, which we interpret as structural constraints on individual value functions. In detail, we aim to show that how the individual values, as the output of a multidimensional linear transformation φ applied on a team value (represented as a cooperative game), satisfy the Shapley value's axioms in learning. When we enforce all these axioms together, the Shapley-form individual values are uniquely shaped (see Theorem 4.7). Moreover, the axioms can be directly translated into algorithmic design principles: (1) The Efficiency axiom corresponds to reward shaping; (2) The Linearity axiom corresponds to truncated TD(λ)-style updates; (3) The Symmetry axiom corresponds to individual value function structures. These principles will later lead to two generic algorithmic templates, referred to as Shapley Machine and Banzhaf Machine.

## 4.1 From Game Structure to Dec-POMDPs

The game structure for the NAHT in Definition 3.2 can be used to represent any sequential decisionmaking process evaluated in states. However, it does not necessarily imply that the process embedded in the game structure strictly follows the problem setting of the NAHT — Dec-POMDPs [Oliehoek et al., 2016]. To bridge this gap, we augment the game structure with Markov dynamics and per-step common rewards to link the values of different states. When the instantiation of each state value is represented as: $\begin{array} { r } { { v _ { s } } _ { t } ( \mathcal { M } ) = \mathbb { E } _ { \pi } \bigl [ \sum _ { \tau = 0 } ^ { T - t - 1 } \gamma ^ { \tau } R _ { t + \tau } \ | \ s _ { t } \bigr ] } \end{array}$ , we can derive the famous n-step return in the RL [Sutton and Barto, 2018][Chap. 7], with a horizon ñ to link $v _ { s _ { t } } ( \mathcal { M } )$ and $v _ { s _ { t + \bar { n } } } ( \mathcal { M } )$ , such that:

$$
v _ { s _ { t } } ( \mathcal { M } ) = \mathbb { E } _ { \boldsymbol \pi } \left[ \sum _ { \tau = 0 } ^ { \bar { n } } \gamma ^ { \tau } R _ { t + \tau } + \gamma ^ { \bar { n } } v _ { s _ { t + \bar { n } } } ( \mathcal { M } ) ~ \Bigg | ~ s _ { t } \right] : = \mathbb { E } _ { \boldsymbol \pi } \left[ G _ { t : t + \bar { n } } ~ \middle | ~ s _ { t } \right] .\tag{3}
$$

For ease of the following presentation, we simplify $v _ { C , s } ^ { v _ { s } ( \mathcal { M } ) } ( \mathcal { M } )$ as $V ( C , s ) , ^ { 5 }$ for all $C \subseteq { \mathcal { M } }$

Proposition 4.1. By Deinition D.2, for every state $s \in S ,$

$$
\bigl ( V ( C _ { 1 } , s ) , \ldots , V ( C _ { m } , s ) \bigr ) \ = \ \bigl ( \mathbb { E } _ { \pi } [ G _ { t : t + n _ { 1 } } \mid s ] , \ldots , \mathbb { E } _ { \pi } [ G _ { t : t + n _ { m } } \mid s ] \bigr ) ,
$$

and, each $V ( C _ { i } , s )$ is equal to $V ( \mathcal { M } , s )$ in value.

By Proposition 4.1 (see the proof and full explanation of Definition D.2 in Appendix D), we obtain the following formula that aligns state-specific basis games of various coalitions to their n-step returns with the corresponding horizons $n \in \{ 1 , . . . , m \}$ , for all $\emptyset \neq C _ { n } \subseteq C _ { n + 1 } , C _ { m } = \mathcal { M }$ , such that:

$$
V ( C _ { n } , s _ { t } ) = \mathbb { E } _ { \boldsymbol { \pi } } \left[ \sum _ { \tau = 0 } ^ { n - 1 } \gamma ^ { \tau } R _ { t + \tau } + \gamma ^ { n } V ( \mathcal { M } , s _ { t + n } ) ~ \Bigg | ~ s _ { t } \right] .\tag{4}
$$

## 4.2 Linearity Axiom and Temporal-Difference Learning

The Linearity axiom plays a central role in connecting cooperative game theory to temporal-difference learning. We show this in the following Proposition 4.2 (see the proof in Appendix F):

Proposition 4.2. Given the expression of state-specific basis games in Eq. (4), applying the Linearity axiom to state-specific cooperative games expressed as Eq. (2) can lead to the following equation:

$$
\phi ( V ( \mathcal { M } , s _ { t } ) ) = \mathbb { E } _ { \boldsymbol { \pi } } \left[ \sum _ { n = 1 } ^ { m } k _ { C _ { n } } ^ { \prime } \cdot \sum _ { \tau = 0 } ^ { n - 1 } \gamma ^ { \tau } \phi ( R _ { t + \tau } ) + \gamma ^ { n } \phi ( V ( \mathcal { M } , s _ { t + n } ) ) \middle | s _ { t } \right] .\tag{5}
$$

In Proposition 4.2, the term within the $\mathbb { E } _ { \pi } [ \cdot ]$ takes the form of truncated λ-return [Sutton and Barto, 2018][Chap. 12], if all $k _ { C _ { r } } ^ { \prime }$ are seen as weightings for the n-step return components. This indicates that the horizon can be estmated by the number of non-empty coalition basis games m.

In practice, we only need to learn $\phi _ { i } ( V ( \mathcal { M } , s _ { t } ) )$ directly for each controlled agent $i \in { \mathcal { C } }$ For brevity, we represent each $\mathsf { \bar { \phi } } _ { i } ( V ( \mathcal { M } , s _ { t } ) )$ as $V _ { i } ( s _ { t } )$ in the following paragraphs. Based on Eq. (5), we derive the λ-return error $\delta _ { t , i }$ [Sutton and Barto, 2018][Chap. 12] of truncated TD(λ) (TTD(λ)) for each agent i purely from the game structure for the NAHT, such that:

$$
\delta _ { t , i } : = \sum _ { n = 1 } ^ { m } k _ { C _ { n } } ^ { \prime } \cdot \sum _ { \tau = 0 } ^ { n - 1 } \left( \gamma ^ { \tau } \phi _ { i } ( R _ { t + \tau } ) + \gamma ^ { n } V _ { i } ( s _ { t + n } ) \right) - V _ { i } ( s _ { t } ) .\tag{6}
$$

Remark 4.3. The above result provides a theoretical explanation for the role of the n-step return in the NAHT; it gives an insight that the truncated TD(λ) learning arises naturally from the Linearity axiom and the NAHT decomposition to capture both internal cooperation and external teamwork, given an appropriate horizon.

## 4.3 Efficiency Axiom and Reward Shaping

Note that φ( $R _ { t + \tau } )$ in Eq. (6) has not yet been defined so far, because $R _ { t + \tau }$ is not belonging to the state-specific cooperative game space $\mathcal { G } ( s )$ . As a result, it is necessary to represent $\phi ( R _ { t + \tau } ^ { - } )$ in the form of $\phi ( V ( \mathcal { M } , s ) )$ , where $V ( { \mathcal { M } } , s ) = v _ { { \mathcal { M } } , s } ^ { v _ { s } ( { \mathcal { M } } ) } ( { \mathcal { M } } )$ and $v _ { \mathcal { M } , s } ^ { v _ { s } ( \mathcal { M } ) } \in \mathcal { G } ( s )$ . The Efficiency axiom can address this problem. To satisfy the Efficiency axiom such that $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \phi _ { i } ( V ( \mathcal { M } , s _ { t } ) ) = V ( \mathcal { M } , s _ { t } ) } \end{array}$ we reasonably presume that its sufficient condition holds: $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \phi _ { i } ( R _ { t } ) = R _ { t } } \end{array}$ . Then, we demonstrate that how $\phi _ { i } ( \bar { R _ { t } } )$ is represented by the Efficiency axiom in the following Proposition 4.4 (see the proof in Appendix G):

Proposition 4.4. Given the condition $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \phi _ { i } ( R _ { t } ) = R _ { t } } \end{array}$ , each agent's payoff allocation defined on rewards $R _ { t } , i . e . , \phi _ { i } ( R _ { t } )$ , can be expressed as:

$$
\phi _ { i } ( R _ { t } ) : = R _ { t } - \sum _ { j \neq i } \left( \phi _ { j } ( V ( \mathcal { M } , s _ { t } ) ) - \gamma \phi _ { j } ( V ( \mathcal { M } , s _ { t + 1 } ) ) \right) .\tag{7}
$$

By replacing $\phi _ { j } ( V ( \mathcal { M } , s _ { t } ) )$ with $V _ { j } ( s _ { t } )$ as there is in Eq. (6), we get that:

$$
\phi _ { i } ( R _ { t } ) = R _ { t } - \sum _ { j \neq i } \left( V _ { j } ( s _ { t } ) - \gamma V _ { j } ( s _ { t + 1 } ) \right) .\tag{8}
$$

The condition $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \phi _ { i } ( R _ { t } ) = R _ { t } } \end{array}$ for deriving the Efficiency axiom is also referred to as additively factored immediate reward function. The following Theorem 4.5 shows that: The Efficiency axiom always exists in the Dec-POMDP, given a factorization scheme of the transition function.

Theorem 4.5 (Principle of Decomposing Value Functions [Oliehoek et al., 2016]). Given an additively factored reward function, for any timestep t there is a factorization of the transition function, such that the value of a finite-horizon factored Dec-POMDP is decomposable across agents.

Since NAHT is considered in Dec-POMDPs (see Section 2.1), to guarantee the feasibility of the Efficiency axiom as shown in Theorem 4.5, another constraint is added to implicitly facilitate searching for a factorization of the transition function, such that:

$$
\sum _ { i = 1 } ^ { M } V _ { i } ( s _ { t } ) = V ( \mathcal { M } , s _ { t } ) .\tag{9}
$$

## 4.4 Symmetry Axiom and Value Function Structure

The Symmetry axiom here means that two agents who contribute equally to every possible coalition (excluding the ones involving themselves) should receive equal payoff allocations [Chalkiadakis et al., 2011][Chap. 2]. In the literature from Dubey [1975], it was described as the permutationequivariance of an multidimensional operator φ defined on a cooperative game, as shown in the following Theorem 4.6 (see the proof in Appendix G):

Theorem 4.6. The payoff allocation operator satisfying permutation-equivariance is a sufficient condition for the Symmetry axiom.

Next, we aim to show how to construct such an operator to fulfil the permutation-equivariance, such that $\phi _ { \sigma ( i ) } ( v _ { s } ) = \phi _ { i } ( \sigma ( v _ { s } ) )$ can be constructed in practice, where σ means relabelling agent IDs to another ordering, $\mathbf { e . g . , ( 1 , 2 ) } \mapsto ( 2 , 1 )$ . For consistency of notations, we also denote $\phi _ { i } ( v _ { s } ( \mathcal { M } ) )$ as $V _ { i } ( s )$ . This can be achieved by the following two conditions: (1) The operator φ's output is changeable consistently with reordering agent IDs, $\mathrm { e . g . } , ( 1 , 2 ) \mapsto ( 2 , 1 ) \Rightarrow ( V _ { 1 } ( s ) , V _ { 2 } ( s ) ) \mapsto ( \hat { V } _ { 2 } ( s ) , V _ { 1 } ( \bar { s ) } ) ; ( 2 )$ The operator's output with respect to each agent should be invariant to reordering agent IDs on its input inp (e.g., the world information or other agents' information), $\mathrm { e . g . , } V _ { i } ( \sigma ( i n p ) ) \stackrel { \textstyle - } { = } V _ { i } ( i n p )$ . Note that the second condition usually holds when other agents' information is not directly included in the input to an agent's individual value function, such as POAM [Wang et al., 2025] and IPPO [De Witt et al., 2020].

## 4.5 From Shapley Axioms to Learning Algorithms

The algorithm that implements $V _ { i }$ as agents' individual values to fulfil the Efficiency, Linearity and Symmetry axioms through learning is named as Shapley Machine: (1) Shape individual rewards φi(R) following Eq. (8) and set Eq. (9) as a regularization term, to fulfil the Efficiency axiom. (2) Implement the TTD(λ) following Eq. (6), to fulfil the Linearity axiom. The $k _ { C } ^ { \prime } > 0 ^ { 6 }$ is implemented as geometric distribution following the convention of TTD(λ) [Sutton and Barto, 2018][Chap. 12]. (3) Structure the policy and individual value networks to fulfil the Symmetry axiom.

Since the Linearity axiom is a sufficient condition for the Additivity axiom (see Appendix B.3), the individual values satisfying all those axioms in implementation realizes Shapley values for the Dec-POMDPs, as highlighted in the following Theorem 4.7 (see the proof in Appendix G):

Theorem 4.7. Shapley Machine is an algorithm enforcing Vi to fulfil Efficiency, Symmetry and Linearity, so the Vi is the Shapley value for Dec-POMDPs.

To realize the Shapley Machine, we use two base algorithms structurally satisfying the Symmetry axiom: POAM [Wang et al., 2025] and IPPO [De Witt et al., 2020]. We then shape the objective functions following conditions imposed by other two axioms, to obtain the algorithms SM-POAM and SM-IPPO, respectively. The details of implementation are left to Appendix H.1.

Following the same principle of designing algorithms via axiomatic characterization, we also propose a variant that enforces the Linearity and Symmetry axioms while omitting Efficiency. Since the resulting Vi aligns with the Banzhaf index, we call this algorithm Banzhaf Machine. Analogous to the implementation of Shapley Machine, its instantiation on POAM and IPPO are named as BM-POAM and BM-IPPO, respectively.

Remark 4.8. Banzhaf Machine is an algorithm generating Vi, fulfilling Symmetry and Linearity.

## 5 Experiments

In experiments, we evaluate the proposed SM-POAM, SM-IPPO, BM-POAM and BM-IPPO on the modified MPE-PP and SMAC benchmark tasks tailored to the NAHT settings [Wang et al., 2025], against the counterpart baseline algorithms POAM and IPPO. The uncontrolled agents are taken from five pre-trained policies, which were trained in fully controllable CMARL settings using the algorithms IPPO, MAPPO, VDN, QMIX, and IQL. To avoid confusion between the standard IPPO algorithm (trained under the CMARL settings) and the version trained within the NAHT process (where only a subset of agents is controllable), we refer to the latter as IPPO-NAHT.

More importantly, we validate the game structure for the NAHT, a core concept in our theory, by setting the number of n-step return components considered in TTD(λ) using the number of basis games m. In practice, the horizon in our theoretical framework is the episode length. Since the number of basis games is far more than the preset episode length (511 vs. 150) in 10v11, we approximate the number of basis games to the episode length as a trade-off. About the 8v9 scenario, the number of basis games needed in theory is close to the episode length (127 vs. 120), so the number of basis games can be naturally approximated by the episode length. As a result, for 8v9 and 10v11 Banzhaf Machine is equivalent to the baseline counterparts in implementation. The details of baselines, experimental settings and evaluation metrics are provided in Appendix H. All results are obtained by first computing each metric with 128 episodes and then averaging these per-seed metrics across 5 random seeds, with 95% confidence intervals reported.

Research Questions. We focus on the following research questions in experimentation: (1) Is the number of coalitions m used to decide the horizon of TTD(λ) (see Section 4.2) a reasonable choice? (2) Are all Shapley value's axioms equally appropriate across scenarios? (3) Which axiom is key to generalization to unseen conventions or unseen agent types?

## 5.1 Main Results and Analysis

Answer to Question 1. We conduct experiments with varying numbers of basis games m using SM-POAM as the candidate algorithm. As shown in Figure 2, learning performance in MPE-PP and

![](images/f7d44e2076b88cbad127aeb1bb51cc2d142c5154e950c6e79e5e5d3f6cfc296a.jpg)

![](images/9167dcefb5d4d676a7678e9e10424e1ca4cf714ac8f5937a49d014de51415c09.jpg)

![](images/b345b29ade619e35f4aff7ca03c86bf87fa2770afca3d9300502fc9176261843.jpg)  
Figure 2: Experiments verifying that the number of n-step return components in TTD(λ) can be determined by the number of non-empty coalition basis games. In MPE-PP the number of basis games is $m = 7 ,$ in 5v6 it is $m = 3 1$ , and in 8v9 it is $m = 1 2 0$

8v9 is sensitive to the choice of m, while in 5v6 it is comparatively robust. Overall, these results indicate that setting the number of n-step return components as the number of non-empty coalition basis games can nearly maximize learning performance, thereby providing an empirical evidence to support the mathematical rationale behind the Linearity axiom that derives this setting for the NAHT. Although we do not deny that there may exist other preferred choices of m (as observed in the results of MPÉ-PP and 5v6), this does not undermine the effectiveness of using the number of non-empty coalition basis games, as a criterion, to decide the horizon for TTD(λ) in practice.

![](images/aa8274977a4f6e0c982e653c4a95e369f7f9420725419495f56dac7061a927e3.jpg)

![](images/f522d9af221bd9d1b4ec8828abd3c0760a53bd07b24bf0c521f62e4f30fa7ad0.jpg)

![](images/35f28a53dc84aad2bc2aeb3bcfd5afdd836b2f3c50cd719a8611d3477e9fbff9.jpg)

![](images/5256ae2bcb9424a595e6089669f95adc10e355fc676fcefabbb1e09976e0b7d9.jpg)

![](images/105185bb9e7fd716ad1b1082a2342b388c4e0065cd1c235723d155e5e458b668.jpg)  
Figure 3: Evaluation curves during training in NAHT.

Answer to Question 2. We evaluate all baselines alongside our proposed algorithms. For the largescale tasks 8v9 and 10v11, the base algorithms IPPO-NAHT and POAM are effectively equivalent to BM-IPPO and BM-POAM under the approximation of m due to limited episode lengths, and are therefore omitted in implementation. Figure 3 shows that, for either IPPO-NAHT or POAM, at least one of their Shapley Machine or Banzhaf Machine variants achieves the best performance. Recall that Banzhaf Machine corresponds to dropping the Efficiency axiom. In MPE-PP and 3sv5z with IPPO-NAHT as the base algorithm, Banzhaf Machine performs on par with Shapley Machine. In most other scenarios (except 10v11 with IPPO-NAHT), Shapley Machine outperforms Banzhaf Machine, although sometimes by a narrow margin. Moreover, comparisons between the base algorithms and their Banzhaf counterparts in MPE-PP, 5v6, and 3sv5z underscore the importance of the Linearity axiom. Overall, the results suggest that all three Shapley axioms generally enhance learning performance (in possibly unseen team compositions), but the Efficiency axiom can, in some cases, reduce it. This justifies the value of our proposed axiomatic framework, which provides a novel perspective on analysing performance through the falsifiable axioms. We believe that this axiomatic framework, as a paradigm, can help design new algorithms in the future.

Answer to Question 3. We evaluate all baselines alongside our proposed algorithms after training, to test their capability of generalization to unseen conventions and unseen agent types, respectively. The evaluation is conducted by the best model saved during training for each random seed of running experiments. Note that the wide range of confidence intervals is due to that the result averages performance across various number of controlled agents, and we further demonstrate the result per the number of controlled agents in Appendix I.5. It can be seen from Figure 4a that SM-POAM outperforms both BM-POAM and POAM on all scenarios, except for the 10v11. This implies that in general all Shapley axioms may be instrumental in tackling unseen conventions. As seen from Figure 4b, the performance of BM-POAM is comparable to and even better than SM-POAM in most cases. This implies that the Efficiency axiom could weaken the capability of generalization to unseen agent types. This phenomenon recurs, and becomes even more pronounced, in the evaluation of IPPO-NAHT related algorithms, as shown in Figure 11b in Appendix I.3.

m = 70 for λ = 0.85.  
![](images/bf6ce2b6b7f4754ec8c99ccaac463e62ea871c3adf94e8b4870329feac4a8e96.jpg)

![](images/f1210600c882f20a4e8ebec58103090cb4c1a1fc2315cf43cb45fadacca19057.jpg)

![](images/d766267d28b66e0bee926b277304734d0c166f55353fc04a278cc78bff7e94be.jpg)

![](images/51ccfc91ce96bf7d0b8e7ffd426f75950c893805dbc3355f4e8d81ac8581da2e.jpg)

![](images/a59c4645c0169cc5183e45e8ac84c9125c3ff18eb8e5ce73c35490ad0a541074.jpg)

(a) Evaluation on unseen conventions (identical algorithms with different random seeds). The star denotes the in-convention baseline, and the histograms show performance on unseen conventions.  
![](images/d830a8e9d07dba20fbda5e903b6d97e97f87983240a37a2c6758e91b8e747928.jpg)

![](images/83e39fe564cf699c9540d8ce56b6f3abd8037f71a1f3b27c20eba58ea832a22e.jpg)

![](images/fe9b02883e6f67a9b64a7b9fb34cbad53d780b80d1fbeaecebaf664dab238c58.jpg)

![](images/62d55222e5b459d42b7b810ccb455c31242c3fc8b4c5a95a5d0ffba6a69ffa0e.jpg)

![](images/10ac1c44e909a9ea77981a90981248951e9cbab7b9bcc48fc31695b3fdc3c016.jpg)  
(b) Evaluation on unseen agent types. Train set: MAPPO, QMIX, IQL; test set: IPPO, VDN.

Figure 4: Out-of-distribution (OOD) evaluation after training, assessing unseen conventions and unseen agent types during training, between learned agents (POAM, SM-POAM and BM-POAM) and uncontrolled agent types (IPPO, VDN, IQL, QMIX and MAPPO). The performance is evaluated by averaging all pairings of N controlled and $M - N$ uncontrolled agents and their corresponding random seeds, called M–N score [Wang et al., 2025].

## 5.2 Number and Weightings of Basis Games

The weightings of basis games $( k _ { C } ^ { \prime }$ in Eq. (6)) are set by the probability values of a geometric distribution governed by λ, respecting the convention of RL. λ will influence the shape of the geometric distribution. As λ increases, the geometric distribution becomes a longer tail with the probability decaying slowly as the number of basis games m increases. In other words, controlling m can be achieved by simply changing the weightings of basis games, or equivalently, the λ, under the assumption of the geometric distribution. To verify this result in the NAHT, we conduct a case study, as shown in Figure 5. It can be observed that the number of effective basis games (the probability values over which are non-zero) for $m = 3 1$ and $m = 7 0$ is the same under $\lambda = 0 . 8 \bar { 5 }$ , resulting in nearly identical learning performance. This aligns with the conventional view of λ's impact on TTD(λ) in RL, primarily as a mechanism for controlling the tail length of the return [Sutton and Barto, 2018][Chap. 12.3].

![](images/de9d6ea8ed8086225c85b66ebca8dd951f53b90c83eead1ca89c009e2ec19432.jpg)  
(a) SM-POAM: m = 31 vs.

![](images/cace7fce67dc00290bd7593d4ed1c74ac1fa524484a3337d9c5ccd7047946540.jpg)  
(b) Demonstration of the shape of weightings of basis games with $\lambda = 0 . 8 5$  
Figure 5: Controlling the number of effective basis games can be implemented by changing the parameters to control the shape of weightings of basis games. Note that the probability values for $m > 3 1$ are nearly zero.

In contrast, we demonstrate that even with the same number of basis games, the values of their corresponding weightings significantly influence the learning performance. As shown in Figure ??, given the same number of active basis games as $m = 7 .$ the learning processes with $\lambda = 0 . 5$ and $\lambda = 0 . 8 5$ result in very different performance. This provides an alternative insight into how weightings influence the TTD(λ), different from the traditional RL perspective.

![](images/b3126842aca6e2dd03000e350f563764635d420bc9c930db2e56b69e3078353c.jpg)

![](images/b4da6199d1b1b0171f4bd4204587c66fd505bcafad1faa3d68288af227144a65.jpg)  
(a) SM-POAM: $\lambda = 0 . 5$ vs. (b) Demonstration of weighting shapes of the basis games with $m = 7 ,$ λ = 0.85 for $m = 7 .$ for $\lambda = 0 . 8 5$ and $\lambda = 0 . 5$ , respectively.  
Figure 6: Influence of different values of basis games given the same number of effective basis games m. Note that the shapes of geometric distributions vary significantly with different values of λ, under the same effective number of basis games.

## 6 Discussion, Limitations and Future Work

We studied reinforcement learning (RL) in open multi-agent systems through the lens of the n-agent ad hoc teamwork (NAHT), and introduced a cooperative game-theoretic framework that makes explicit how openness is distinguished from the cooperative multi-agent reinforcement learning (CMARL). Central to our approach is the NAHT decomposition, which separates internal cooperation among controlled agents from external teamwork with uncontrolled agents. Building on this structure we proposed an axiomatic framework that derives value function learning from the Shapley axioms (Efficiency, Symmetry, and Linearity) by enforcing them as structural constraints. This perspective shifts the design of RL algorithms from heuristic constructions to principled designs in the NAHT. Since CMARL is an extreme case of the NAHT, our framework also works on it.

Our framework also reveals a novel connection between cooperative game theory and reinforcement learning: The Linearity axiom provides a theoretical explanation for the n-step return methods and the TTD(λ) in the NAHT. This result is coincidentally related to the latest finding in single-agent offline RL that the n-step return methods with selected horizons can improve scalability to the increasing size of datasets [Park et al., 2026]. As the NAHT encompasses the case of single-agent RL with other unknown agents, our framework suggests that, through the TTD(λ), the increasing size of datasets could be a promising way to address the NAHT, which is left for the future work.

More broadly, the axiomatic perspective suggests new opportunities for exploring richer classes of weighting schemes beyond the standard geometric forms in temporal-difference learning for the NAHT (as demonstrated in Section 5.2). For example, weightings $k _ { C }$ satisfying $\sum ( k _ { C } ) ^ { \sum } = 1$ as shown in Figure 1 could define alternative weighting schedules. Because the NAHT problem can be reduced to the special case of single-agent RL with other uncontrolled agents, with an appropriate connection between the world model (transition function) and the concept of other uncontrolled agents, this result can be relaxed to the broader single-agent RL.

A key limitation of our work is that it is still unclear about how to deal with the scenario where the predefined horizon is far less than the number of non-empty coalition basis games. Our tentative strategy to mitigate this shortcoming is using the horizon under the minimal horizon schedule (e.g., $n _ { i } = n _ { i - 1 } + 1 )$ as the number of random basis games with no control. In the future work, this can be further investigated by relaxing the minimal horizon schedule $( { \bf e . g . } , n _ { i } = f ( i , s )$ , where $f ( i , s ) \in \mathbb { N }$ is a function of states s and coalition indices ¿), which can be used to control the effective number of basis games to be applied

## Acknowledgements

Jianhong Wang and Jonathan Lawry are supported by the Engineering and Physical Sciences Research Council (EPSRC) [Grant Ref: EP/Y028732/1]. Samuel Kaski is supported by UKRI Turing AI World-Leading Researcher Fellowship, EP/W002973/1 and EU, ERC ODD-ML 101201120.

## References

Noa Agmon and Peter Stone. Leading ad hoc agents in joint action settings with multiple teammates. In AAMAS, pages 341–348, 2012.

Noa Agmon, Samuel Barrett, and Peter Stone. Modeling uncertainty in leading ad hoc teams. In Proceedings of the 2014 international conference on Autonomous agents and multi-agent systems, pages 397–404, 2014.

Stefano V Albrecht and Subramanian Ramamoorthy. A game-theoretic model and best-response learning method for ad hoc coordination in multiagent systems. In Proceedings of the 2013 international conference on Autonomous agents and multi-agent systems, pages 1155–1156, 2013.

Ronen I Brafman and Moshe Tennenholtz. On partially controlled multi-agent systems. Journal of Artificial Intelligence Research, 4:477–507, 1996.

Jiajun Chai, Yuqian Fu, Dongbin Zhao, and Yuanheng Zhu. Aligning credit for multi-agent cooperation via model-based counterfactual imagination. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, pages 281–289, 2024.

Georgios Chalkiadakis, Edith Elkind, and Michael Wooldridge. Computational aspects of cooperative game theory. Morgan & Claypool Publishers, 2011.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

Pawel Cichosz. Truncating temporal differences: On the efficient implementation of td (lambda) for reinforcement learning. Journal of Artificial Intelligence Research, 2:287–318, 1994.

Peter Dayan. The convergence of td (λ) for general λ. Machine learning, 8:341–362, 1992.

Peter Dayan and Terrence J Sejnowski. Td (λ) converges with probability 1. Machine Learning, 14: 295–301, 1994.

Christian Schroeder De Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip HS Torr, Mingfei Sun, and Shimon Whiteson. Is independent learning all you need in the starcraft multi-agent challenge?arXiv preprint arXiv:2011.09533, 2020.

Pradeep Dubey. On the uniqueness of the shapley value. International Journal of Game Theory, 4(3): 131–139, 1975.

Gianluigi Greco, Enrico Malizia, Luigi Palopoli, Francesco Scarcello, et al. On the complexity of the core over coalition structures. In IJCAI, volume 11, pages 216–221, 2011.

Dongge Han, Chris Xiaoxuan Lu, Tomasz P. Michalak, and Michael J. Wooldridge. Multiagent modelbased credit assignment for continuous control. In 21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2022, Auckland, New Zealand, May 9-13, 2022, pages 571–579. International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS), 2022.

Jiahui Li, Kun Kuang, Baoxiang Wang, Furui Liu, Long Chen, Fei Wu, and Jun Xiao. Shapley counterfactual credits for multi-agent reinforcement learning. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 934–942, 2021.

Yang Li, Junfan Chen, Feng Xue, Jiabin Qiu, Wenbin Li, Qingrui Zhang, Ying Wen, and Wei Pan. At-drone: Benchmarking adaptive teaming in multi-drone pursuit. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 1144–1161. PMLR, 2025.

Siddharth Nayak, Kenneth Choi, Wenqi Ding, Sydney Dolan, Karthik Gopalakrishnan, and Hamsa Balakrishnan. Scalable multi-agent reinforcement learning through intelligent information aggregation. In International Conference on Machine Learning, pages 25817–25833. PMLR, 2023.

Frans A Oliehoek, Christopher Amato, et al. A concise introduction to decentralized POMDPs, volume 1. Springer, 2016.

Seohong Park, Kevin Frans, Deepinder Mann, Benjamin Eysenbach, Aviral Kumar, and Sergey Levine. Horizon reduction makes rl scalable. Advances in Neural Information Processing Systems, 38:8350–8389, 2026.

Jing Peng and Ronald J Williams. Incremental multi-step q-learning. In Machine Learning Proceedings 1994, pages 226–232. Elsevier, 1994.

Muhammad A Rahman, Niklas Hopner, Filippos Christianos, and Stefano V Albrecht. Towards open ad hoc teamwork using graph-based policy learning. In International Conference on Machine Learning, pages 8776–8786. PMLR, 2021.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

Harm Seijen and Rich Sutton. True online td (lambda). In International Conference on Machine Learning, pages 692–700. PMLR, 2014.

LS Shapley. A value for n-person games. Annals of Mathematics Studies, 28:307–318, 1953.

Peter Stone and Sarit Kraus. To teach or not to teach? decision making under uncertainty in ad hoc teams. In Proceedings of the 9th International Conference on Autonomous Agents and Multiagent Systems: volume 1-Volume 1, pages 117–124, 2010.

Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3: 9–44, 1988.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction second edition. Adaptive computation and machine learning: The MIT Press, Cambridge, MA and London, 2018.

John Tsitsiklis and Benjamin Van Roy. Analysis of temporal-diffference learning with function approximation. Advances in neural information processing systems, 9, 1996.

Caroline Wang, Muhammad Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone. Nagent ad hoc teamwork. Advances in Neural Information Processing Systems, 37:111832–111862, 2025.

Jianhong Wang, Yuan Zhang, Tae-Kyun Kim, and Yunjie Gu. Shapley q-value: A local reward approach to solve global reward games. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7285–7292, 2020.

Jianhong Wang, Wangkun Xu, Yunjie Gu, Wenbin Song, and Tim C Green. Multi-agent reinforcement learning for active voltage control on power distribution networks. Advances in Neural Information Processing Systems, 34:3271–3284, 2021.

Jianhong Wang, Yuan Zhang, Yunjie Gu, and Tae-Kyun Kim. Shaq: Incorporating shapley value theory into multi-agent q-learning. Advances in Neural Information Processing Systems, 35: 5941–5954, 2022.

Jianhong Wang, Yang Li, Yuan Zhang, Wei Pan, and Samuel Kaski. Open ad hoc teamwork with cooperative game theory. In International Conference on Machine Learning, pages 50902–50930. PMLR, 2024.

Christopher John Cornish Hellaby Watkins et al. Learning from delayed rewards. PhD Thesis, 1989. David Williams. Probability with martingales. Cambridge university press, 1991.

Annie Xie, Dylan Losey, Ryan Tolsma, Chelsea Finn, and Dorsa Sadigh. Learning latent representations to influence multi-agent interaction. In Conference on robot learning, pages 575–588. PMLR, 2021.

Di Xue, Lei Yuan, Zongzhang Zhang, and Yang Yu. Efficient multi-agent communication via shapley message value. In IJCAI, pages 578–584, 2022.

H Peyton Young. Monotonic solutions of cooperative games. International Journal of Game Theory, 14(2):65–72, 1985.

Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. Advances in neural information processing systems, 35:24611–24624, 2022.

Yuan Zhang, Umashankar Deekshith, Jianhong Wang, and Joschka Boedecker. Improving the efficiency and efficacy of multi-agent reinforcement learning on complex railway networks with a local-critic approach. In Proceedings of the International Conference on Automated Planning and Scheduling, volume 34, pages 698–706, 2024.

Luisa M. Zintgraf, Sam Devlin, Kamil Ciosek, Shimon Whiteson, and Katja Hofmann. Deep interactive bayesian reinforcement learning via meta-learning. In Frank Dignum, Alessio Lomuscio, Ulle Endriss, and Ann Nowé, editors, AAMAS '21: 20th International Conference on Autonomous Agents and Multiagent Systems, Virtual Event, United Kingdom, May 3-7, 2021, pages 1712–1714. ACM, 2021.

## A Related Work

## A.1 TD(λ) in Reinforcement Learning and Multi-Agent Reinforcement Learning

Related work on temporal-difference learning with eligibility traces centers on the seminal TD(λ) algorithm proposed by Sutton [1988], which unified one-step TD and Monte Carlo methods via a tracedecay parameter λ and introduced the forward- and backward-view equivalence for efficient multistep credit assignment. Subsequent theoretical analyses [Dayan, 1992, Dayan and Sejnowski, 1994, Tsitsiklis and Van Roy, 1996] established convergence guarantees and characterized the bias-variance trade-off inherent in λ-returns, while extensions by Watkins et al. [1989], Peng and Williams [1994] adapted eligibility traces for off-policy control, Q(λ). Advances in function approximation, including true online TD(λ) [Seijen and Sutton, 2014] and gradient-TD methods, further broadened applicability to large-scale and nonlinear settings, inspiring n-step return techniques such as generalized advantage estimation (GAE) [Schulman et al., 2015] for policy optimization. In the multi-agent reinforcement learning (MARL), algorithms of PPO family such as MAPPO [Yu et al., 2022] and IPPO [De Witt et al., 2020], applied GAE to optimize policies and thus used the TD(λ) as the target values to train critics. Our work highlights the theoretical link between the truncated TD(λ), denoted as TTD(λ) [Cichosz, 1994], in the NAHT and the Linearity axiom of the Shapley value. It further shows that the number of n-step return components in the TTD(λ) can be estimated by the number of non-empty coalition basis games. This insight provides a theoretical foundation that can inform the design of a broader class of MARL algorithms.

## A.2 Theoretical Models for Ad Hoc Teamwork

We now discuss theoretical models describing the ad hoc teamwork (AHT). Brafman and Tennenholtz [1996] pioneered research into the AHT by investigating repeated matrix games involving a single teammate. Subsequent studies expanded this framework to scenarios with multiple teammates, notably by Agmon and Stone [2012]. Later, Agmon et al. [2014] further relaxed earlier assumptions by allowing teammates' policies to be selected from a known set. Stone and Kraus [2010] initially formalized the AHT through collaborative multi-armed bandits, albeit under notable assumptions such as prior knowledge of teammates' policies and environmental conditions. Albrecht and Ramamoorthy [2013] advanced this field significantly by introducing the stochastic Bayesian game (SBG), the first comprehensive theoretical framework accommodating dynamic environments and unknown teammate behaviours in the AHT. Building on the SBG, Rahman et al. [2021] proposed the open stochastic Bayesian game (OSBG), addressing the open ad hoc teamwork (OAHT). Zintgraf et al. [2021] modelled the AHT through interactive Bayesian reinforcement learning (IBRL) within Markov games, specifically targeting non-stationary teammate policies within single episodes. Xie et al. [2021] introduced the hidden parameter Markov decision process (HiP-MDP) to handle situations where teammates'policies vary across episodes but remain stationary during individual episodes. Most recently, Wang et al. [2024] extended the OSBG by incorporating principles from cooperative game theory, introducing the open stochastic Bayesian coalitional affinity game (OSBG-CAG), which theoretically justifies a graph-based representation for joint Q-value functions and includes rigorous convergence proofs for Q-learning algorithms in open team settings. Our work adopts the perspective of cooperative game theory, introducing an NAHT decomposition based on state-specific cooperative games to address the NAHT. This well captures the structural difference between the CMARL and the NAHT. Based on the NAHT decomposition, we derive an axiomatic framework to address the NAHT, which introduces a novel perspective for solving this problem. The key difference in assumptions between the work by Wang et al. [2024] and our work is that: the work by Wang et al. [2024] was built on the non-transferable utility games, while our work is built on the transferable utility games. For more details about the difference between these two cooperative game models, please refer to Chalkiadakis et al. [2011].

## A.3 Shapley Value in Multi-Agent Reinforcement Learning

Early studies [Wang et al., 2020] incorporated the Shapley value into credit assignment scheme in a principled way and proposed an algorithm named as SQDDPG, underpinned by the equivalence between cooperative-game theoretical models in dynamic scenarios and the team reward Markov game. Wang et al. [2022] further improved the theory by proving the existence of the Shapley value under the team reward Markov game through construction, and proposed an algorithm, relying on the construction, named as SHAQ — an algorithm with promising convergence to find an optimal joint policy. It also shed light on the relationship between the Shapley value and the relevant value decomposition and credit assignment approaches for the CMARL. Li et al. [2021] improved the stability of SQDDPG and proposed an algorithm named Shapley Counterfactual Credits. Han et al. [2022] incorporated the Shapley value into the model-based PPO as each agent's advantage value, through the estimated coalition values by trained transition and reward models. Chai et al. [2024] continued the idea from Han et al. [2022], by replacing the transition and reward models with more powerful world models. On the other hand, Xue et al. [2022] incorporated the Shapley value into multi-agent communication, as a criterion for forming communication between agents. One of our work's aims is learning the Shapley value to address the NAHT, a generalized problem to the CMARL More importantly, the Shapley value is learned by satisfying its axioms, rather than by constructing their explicit formulas as implemented in the previous work. This may help overcome the notorious learning instability issues caused by learning substantial “coalitional values" in the existing algorithms based on the Shapley value. Furthermore, the insight of axiomatic characterization can facilitate improving the flexibility of the Shapley value approaches in a wide variety of tasks. Specifically, one can safely drop any part of a learning algorithm, if it is justified that the targeting task does not align with the corresponding axiom.

## B Extra Background

## B.1 λ-Return and TD(λ)

We now introduce an extension of return named λ-return. Mathematically, a λ-return $G _ { t } ^ { \lambda }$ for infinitehorizon cases can be expressed as follows:

$$
\begin{array} { r } { G _ { t : t + n } = R _ { t } + \gamma R _ { t + 1 } + \cdot \cdot \cdot + \gamma ^ { n - 1 } R _ { t + n - 1 } + \gamma ^ { n } V ( s _ { t + n } ) , } \\ { G _ { t } ^ { \lambda } = ( 1 - \lambda ) \displaystyle \sum _ { n = 1 } ^ { \infty } \lambda ^ { n } G _ { t : t + n } . } \end{array}\tag{10}
$$

Similarly, the λ-return for finite-horizon cases can be expressed as follows:

$$
G _ { t } ^ { \lambda } = ( 1 - \lambda ) \sum _ { n = 1 } ^ { T - t } \lambda ^ { n } G _ { t : t + n } + \lambda ^ { T - t } G _ { t : T } .\tag{11}
$$

Note that if $\lambda = 1$ , updating according to the λ-return is a Monte Carlo algorithm. In contrast, if $\lambda = 0$ , then the λ-return reduces to $G _ { t : t + 1 }$ , the one-step return. The Monte Carlo algorithm is known as its high variance but low bias, while one-step return is known as its low variance but high bias. For this reason, the change of λ can be seen as a tradeoff between bias and variance. The TD prediction using λ-return as the target value is referred to as TD(λ) [Sutton, 1988].

A variant of TD(λ) by shaping $G _ { t } ^ { \lambda }$ with a designated horizon $H \in \{ t + 1 , . . . , T \}$ is referred to as truncated TD(λ), shortened as TTD(λ) [Cichosz, 1994]. Mathematically, $G _ { t } ^ { \lambda }$ with a designated horizon H can be formulated as follows:

$$
G _ { t } ^ { \lambda , H } = ( 1 - \lambda ) \sum _ { n = 1 } ^ { H - t } \lambda ^ { n } G _ { t : t + n } + \lambda ^ { H - t } G _ { t : H } .\tag{12}
$$

Note that in the convention of RL, the horizon H is defined as the time difference from the starting point of an episode, denoted by $t = 0$

Remark B.1. In this paper, to simplify the presentation of our work, we redefine the horizon as the time difference from each timestep t, denoted as m, such that $m = H - t .$ With the new defnition of horizon, we can rewrite the above formula $o f G _ { t } ^ { \lambda , H }$ as follows:

$$
G _ { t } ^ { \lambda , m } = ( 1 - \lambda ) \sum _ { n = 1 } ^ { m } \lambda ^ { n } G _ { t : t + n } + \lambda ^ { m } G _ { t : t + m } .\tag{13}
$$

## B.2 Superadditive Game and Cooperative Multi-Agent Reinforcement Learning

The superadditive game is a subclass of characteristic games that satisfies an additional condition: $v ( C \cup \mathbf { \bar { \phi } } D ) \geq v ( C ) \bar { + } v ( D )$ , for any two distinct coalitions $C , D \subseteq { \mathcal { M } }$ with $C \cap D = \emptyset$ . Previous work in the CMARL [Wang et al., 2020] has shown the equivalence on the task objective function between the state-based superadditive games with action spaces and the team reward Markov games with coalition structures.7 Intuitively, the Shapley value is a solution to the stability of a team formation, resembling cooperation. Since the NAHT is a generalization of the CMARL, this conclusion is still valid in principle. It has been already proved that the Shapley value exists in superadditive games [Shapley, 1953] and each game instance in superadditive games can be uniquely represented as $\sum k _ { C } v _ { C } ^ { 1 }$ , where $k _ { C } \geq 0$ [Dubey, 1975]. In cooperative game theory, any game can be transformed to a superadditive game if its value v is transformed by a superadditive cover $v ^ { * }$ such that $v ^ { * } ( C ) = \mathrm { m a x } _ { C S ( C ) } \sum _ { D \in C S ( C ) } v ( D )$ [Greco et al., 2011]. As a result, it will not lose generality if we consider the cooperative game space as superadditive games in our work.

## B.3 Axioms of Shapley Values

Efficiency and Symmetry have been well investigated in the literature of multi-agent reinforcement learning [Wang et al., 2020, Li et al., 2021, Han et al., 2022, Wang et al., 2022, Chai et al., 2024], but Additivity still lacks attention. In detail, Additivity means that $\phi ( w _ { 1 } + w _ { 2 } ) = \phi ( w _ { 1 } ) + \phi ( w _ { 2 } )$ , for any cooperative games $w _ { 1 } , w _ { 2 } \in \mathcal { G }$ . If we consider m possible games, then we have the following expression: $\begin{array} { r } { \phi ( \sum _ { i = 1 } ^ { m ^ { - } } w _ { i } ) = \mathbf { \bar { \sum } } _ { i = 1 } ^ { \bar { m } } \phi ( w _ { i } ) } \end{array}$ , where $w _ { 1 } , w _ { 2 } , . . . , w _ { m } \in \mathcal { G }$ and the sum $\sum _ { i = 1 } ^ { m } w _ { i }$ can be seen as a game reproduced by the above games. Linearity is a stronger condition than Additivity [Dubey, 1975, Young, 1985]. Specifically, Linearity requires $\begin{array} { r } { \dot { \phi } ( \sum _ { i = 1 } ^ { m } \alpha _ { i } w _ { i } ) = \sum _ { i = 1 } ^ { m } \alpha _ { i } \phi ( w _ { i } ) } \end{array}$ , for all $\alpha _ { i } \in \mathbb { R }$ In contrast, Additivity only requires to satisfy the above equation when $\alpha _ { i } = 1$ . In our work, we use the Linearity axiom to establish the Additivity axiom, to adapt to our derivation

## B.4 Baseline Algorithms: POAM and IPPO

POAM. In general, POAM is an algorithm built upon IPPO [De Witt et al., 2020]. The primary difference between POAM and IPPO is as follows: For each agent's policy and value, in addition to its observation used in IPPO, POAM takes as input an embedding vector to predict other agents' potential behaviours. More specifically, this embedding vector is trained by an encoder-decoder structure, wherein the encoder is modeled as RNNs and decoder is modeled as MLPs. For brevity, we use —i to denote the set of all agents excluding agent i. Let $h _ { t , i } = \{ o _ { k , i } , a _ { k - 1 , i } \} _ { k = 1 } ^ { t }$ denote agent i's history of observations and actions up to timestep t and $\boldsymbol { e } _ { t , i } \in \mathbb { R } ^ { n }$ denote the resulting embedding vector of n dimensions. The encoder parameterized by $\theta ^ { e }$ is defined as $f _ { \theta ^ { e } } ^ { e n c } : \mathcal { H } _ { i }  \mathbb { R } ^ { n }$ . The embedding vector is decoded by two decoder networks: the observation decoder $f _ { \theta ^ { o } } ^ { d e c } : \mathbb { R } ^ { n }  \mathcal { O } _ { - i } ,$ and the action decoder $f _ { \theta ^ { a } } ^ { d e c } : \mathbb { R } ^ { n }  \Delta ( { \mathcal { A } } _ { - i } )$ . The decoder networks are respectively trained to predict the observations and actions of all other agents in the team at timestep $t , o _ { t , - i }$ and $a _ { t , - i } ,$ to encourage $e _ { t , \cdot }$ to contain information about collective behaviors corresponding to $h _ { t , i }$ . While the observation decoder directly predicts the observed —¿'s observations, the action decoder predicts the parameters of a probability distribution over the $- i \ ' s$ actions $\pi _ { - i } ( a _ { t , - i } ; f _ { \theta ^ { a } } ^ { d e c } ( f _ { \theta ^ { e } } ^ { e n c } ( h _ { t , i } ) ) )$ . As we consider continuous observations and discrete actions, the loss function with using categorical distribution to model $\pi _ { - i }$ is as follows:

$$
\begin{array} { r } { L _ { \theta ^ { e } , \theta ^ { o } , \theta ^ { a } } ( h _ { t , i } , o _ { t , - i } , a _ { t , - i } ) = | | f _ { \theta ^ { o } } ^ { d e c } ( f _ { \theta ^ { e } } ^ { e n c } ( h _ { t , i } ) ) - o _ { t , - i } | | ^ { 2 } - \log \pi _ { - i } ( a _ { t , - i } ; f _ { \theta ^ { o } } ^ { d e c } ( f _ { \theta ^ { e } } ^ { e n c } ( h _ { t , i } ) ) ) . } \end{array}\tag{14}
$$

For brevity, we define $g ( h ) : = ( h , f _ { \theta ^ { e } } ^ { e n c } ( h ) )$ in the following description.

POAM employed generalized advantage estimation (GAE) [Schulman et al., 2015] to form the policy gradient, and the value function used to form GAE is trained by the following loss function:

$$
L _ { \theta ^ { c } } ( h _ { t , i } ) = \frac { 1 } { 2 } \left( V _ { i } ^ { \theta ^ { c } } ( g ( h _ { t , i } ) ) - \hat { V } _ { t , i } \right) ^ { 2 } ,\tag{15}
$$

where $\hat { V } _ { t , i }$ is the finite-horizon TD(λ) return, for which the horizon is up to the episode length T. Each agent i's GAE estimation is as follows:

$$
A _ { t , i } = \sum _ { l = 0 } ^ { T } ( \gamma \lambda ) ^ { l } \delta _ { t + l , i } ,\tag{16}
$$

where $\delta _ { t , i } = R _ { t } + \gamma V _ { i } ( g ( h _ { t , i } ) ) - V _ { i } ^ { \theta ^ { c } } ( g ( h _ { t , i } ) )$ and $V _ { i } ( g ( h _ { t , i } ) )$ denotes the non-parameterized target individual value. Based on the GAE defined above, the policy optimization loss is defined as:

$$
L _ { \theta } ( h _ { t , i } , a _ { t , i } ) = \operatorname* { m i n } \left\{ \frac { \pi _ { \theta } ( a _ { t , i } | g ( h _ { t , i } ) ) } { \pi _ { \theta _ { o l d } } ( a _ { t , i } | g ( h _ { t , i } ) ) } A _ { t , i } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( a _ { t , i } | g ( h _ { t , i } ) ) } { \pi _ { \theta _ { o l d } } ( a _ { t , i } | g ( h _ { t , i } ) ) } , 1 - \epsilon , 1 + \epsilon \right) A _ { t , i } \right\} .\tag{17}
$$

IPPO. By removing the loss for the encoder-decoder structure as shown in Eq. (14) and replacing the input $g ( \bar { h } _ { t , i } )$ by $h _ { t , i }$ in the above set of equations, we get the loss functions of IPPO.

## C Preliminary Results

Lemma C.1. Given a xed state $s \in S ,$ each state-specific cooperative game $v _ { s } \in \mathcal G ( s )$ can be uniquely represented by: $\begin{array} { r } { \boldsymbol { v } _ { s } = \sum _ { \emptyset \neq C \subseteq M } \boldsymbol { k } _ { C } ^ { \prime } \cdot \boldsymbol { v } _ { C , s } ^ { z } , } \end{array}$ where $\begin{array} { r } { k _ { C } ^ { \prime } = \frac { k _ { C } } { z } a n d z \neq 0 } \end{array}$

Proof. We can represent each $k _ { C }$ in an equivalent form as $\frac { k _ { C } \cdot z } { z }$ , where $z \neq 0$ . Substituting this term into the formula $\begin{array} { r } { v _ { s } = \sum _ { \emptyset \neq C \subseteq M } k _ { C } v _ { C , s } ^ { 1 } } \end{array}$ in Definition 3.1, we can get the following formula such that:

$$
v _ { s } = \sum _ { \emptyset \neq C \subseteq M } \frac { k _ { C } \cdot z } { z } \cdot v _ { C , s } ^ { 1 } .
$$

Since $v _ { C , s } ^ { z } = z \cdot v _ { C , s } ^ { 1 }$ by Definition 3.1, we have the following formula such that:

$$
v _ { s } = \sum _ { \emptyset \neq C \subseteq M } \frac { k _ { C } } { z } \cdot v _ { C , s } ^ { z } .
$$

Proposition C.2. For the class of superadditive games formed by a set of state-specific basis games $\{ v _ { C , s } ^ { z ^ { \star } } | \emptyset \neq C \subseteq { \mathcal { M } } \}$ , such that: $\begin{array} { r } { \dot { v } _ { s } \stackrel { - } { = } \sum _ { \emptyset \neq C \subseteq M } \bar { k } _ { C } ^ { \prime } \cdot v _ { C , s } ^ { \dot { z } } , } \end{array}$ it holds that $\dot { k _ { C } ^ { \prime } } \geq 0 ,$ for all $\emptyset \neq C \subseteq M$

Proof. We first consider a cooperative game space $\mathcal { G }$ (a broader space of superadditive games). As per Dubey [1975], a game $v : \bar { \mathcal { G } } \to \mathbb { R } _ { \geq 0 }$ belonging to $\mathcal { G }$ can be uniquely represented as follows:

$$
v = \sum _ { \emptyset \neq C \subseteq \tilde { N } } k _ { C } v _ { C } ^ { 1 } .
$$

The analytic form of $k _ { C }$ under $\mathcal { G }$ is represented as follows:

$$
k _ { C } = \sum _ { T \subseteq C } ( - 1 ) ^ { | C | - | T | } v ( T ) .
$$

The condition for v to be a superadditive game is as follows:

$$
v ( T _ { 1 } \cup T _ { 2 } ) \geq v ( T _ { 1 } ) + v ( T _ { 2 } ) , \qquad T _ { 1 } , T _ { 2 } \subseteq C .
$$

Following the result from Dubey [1975], if the above condition holds, $k _ { C } \geq 0$ has to be satisfied.8

We now extend the $\mathcal { G }$ to a state-specific cooperative game space $\mathcal G ( s )$ , for a fixed state $s \in S$ . Since the $\mathcal { G } ( s )$ is isomorphic to a game space ${ \mathcal { G } } ,$ the characteristics satisfied in the $\mathcal { G }$ also holds in the $\mathcal G ( s )$ . Therefore, $v _ { s } : \mathcal { G } ( s )  \mathbb { R } _ { \geq 0 }$ can be uniquely represented as $\begin{array} { r } { v _ { s } = \sum _ { \emptyset \neq C \subseteq M } k _ { C } v _ { C , s } ^ { 1 } } \end{array}$ . As we consider $\mathcal G ( s )$ as a space of superadditive games, $k _ { C } \geq 0$ holds.

By Lemma C.1, we have $\begin{array} { r } { \boldsymbol { v } _ { s } = \sum _ { \emptyset \neq C \subset \mathcal { M } } \boldsymbol { k } _ { C } ^ { \prime } \cdot \boldsymbol { v } _ { C , s } ^ { z } , } \end{array}$ where $\begin{array} { r } { k _ { C } ^ { \prime } = \frac { k _ { C } } { z } } \end{array}$ and $z \neq 0$ . This implies that any set of basis games in the form $\{ v _ { C , s } ^ { z ^ { - } } | \emptyset \neq C \subseteq { \mathcal { M } } \}$ can form a state-specific cooperative game belonging to superadditive games. The proof is completed. □

Remark C.3. Following discussion in Section 2.2, we aim to restrict the state-specific cooperative game class to the superadditive game. By Proposition C.2, we can conclude that $k _ { C } ^ { \bar { \prime } } \geq 0$ is a sufficient condition for $v _ { s }$ belonging to the superadditive game class. In other words, $k _ { C } ^ { \prime } \geq 0$ is a key condition for reaching cooperation within a team of agents. However, $k _ { C } ^ { \prime } = 0$ is a singularity condition, which leads to 0 for all state-specific cooperative games in values, with no actual meaning in practice. For this reason, we only choose $k _ { C } ^ { \prime } > 0$ as the condition in our work.

## D Introducing Markovian Dynamics and Rewards to Game Structures

Based on the game structure for the NAHT in Definition 3.2, we now introduce Markovian dynamics to establish the link between the state-specific cooperative games of consecutive states, aligned with the environmental setting in Dec-POMDPs [Oliehoek et al., 2016].

The only postulate is that each state-specific cooperative game can be represented in the form of rewards: $v _ { s _ { t } } ( \mathcal { M } ) = \mathbb { E } _ { \pi } [ \sum _ { \tau = 0 } ^ { T - t - 1 } \gamma ^ { \tau } R _ { t + \tau } \ | \ s _ { t } ] \in \mathcal { G } ( s _ { t } )$ . By the convention of RL [Sutton and Barto, 2018][Chap. 7], it is feasibie to use the n-step return to express the $v _ { s _ { t } } ( \mathcal { M } )$ under a fixed horizon ñ to define the transformation, such that:

$$
v _ { s _ { t } } ( \mathcal { M } ) = \mathbb { E } _ { \pi } \left[ \sum _ { \tau = 0 } ^ { \bar { n } - 1 } \gamma ^ { \tau } R _ { t + \tau } + \gamma ^ { \bar { n } } v _ { s _ { t + \bar { n } } } ( \mathcal { M } ) ~ \Bigg | ~ s _ { t } \right] : = \mathbb { E } _ { \pi } \left[ G _ { t : t + \bar { n } } ~ \middle | ~ s _ { t } \right] .\tag{18}
$$

For brevity, we rewrite $v _ { C , s } ^ { v _ { s } ( \mathcal { M } ) } ( \mathcal { M } )$ as $V ( C , s )$ for all $C \subseteq { \mathcal { M } }$ . Then, we get the following fact: Fact D.1. $V ( C _ { 1 } , s ) = V ( C _ { 2 } , s ) = \cdots = V ( \mathcal { M } , s ) = v _ { s } ( \mathcal { M } )$ for all $C _ { i } \in \mathcal P _ { + } ( \mathcal M )$ , as per Definition 2.1 and ${ \mathcal { M } } \supseteq C _ { i }$ , where $\mathcal { P } _ { + } ( \mathcal { M } )$ indicates the powerset of M excluding the empty set.

Definition D.2 (Coalition Ordering by Size and Horizon Assignment). We first set $m : = 2 ^ { M } - 1$ Given a coalition enumeration $( C _ { 1 } , \bar { . . . } , C _ { m } )$ satisfying: $( l ) \left| C _ { i } \right| \leq \left| C _ { i + 1 } \right| f o r$ all i (non-decreasing by size); (2) within each size, ties are broken by a fixed total order $( e . g .$ , lexicographic ordering over agent indices), we can deine a sequence of horizons $( n _ { 1 } , \ldots , n _ { m } )$ such that $n _ { i } : = n _ { i - 1 } + 1$ for all $i \geq 2$ and $n _ { 1 } : = 1$ . Accordingly, we represent the basis games $\dot { V } ( C _ { i } , s )$ as their corresponding expected n-step returns:

$$
V ( C _ { i } , s ) : = \mathbb { E } _ { \pi } { \big [ } G _ { t : t + n _ { i } } \mid s _ { t } = s { \big ] } , \qquad ( i = 1 , \ldots , m ) .
$$

We justify Definition D.2 in Appendix E by illustrating the alignment between the variance ordering of $V ( C _ { i } , s )$ and that of $E _ { \pi } [ \tilde { G } _ { t : t + n _ { i } } \ \vert \ s _ { t } = s ]$ This definition concerns the following situation: The number of agents increases when more exploration is conducted $( \mathrm { i } . \mathrm { e } . , C _ { i }$ becomes larger in sizes), so more diverse agent types and behaviours emerge, resulting in that the estimation for the corresponding $V ( C _ { i } , s )$ has a higher variance. The expected n-step return as an estimation model for $V ( C _ { i } , s )$ can well adapt to this situation, given some mathematically motivated assumptions. To justify the necessity of Definition D.2, we have shown in experiments that the Linearity axiom derived from Proposition D.3 that is building on Definition D.2 can improve the algorithmic performance by deciding the horizon m in Eq. (6).

Proposition D.3. By Definition D.2, for every $s \in S _ { : }$

$$
\bigl ( V ( C _ { 1 } , s ) , \ldots , V ( C _ { m } , s ) \bigr ) \ = \ \bigl ( \mathbb { E } _ { \pi } [ G _ { t : t + n _ { 1 } } \mid s ] , \ldots , \mathbb { E } _ { \pi } [ G _ { t : t + n _ { m } } \mid s ] \bigr ) ,
$$

and, each $V ( C _ { i } , s )$ is equal to $V ( \mathcal { M } , s )$ in value.

Proof. The result can be immediately obtained from the definition of $V ( C _ { i } , s )$ in Definition D.2, and the statement that each $V ( C _ { i } , s )$ is equal to $V ( { \mathcal { M } } , s )$ , directly following Fact D.1. □

## E Justification of Coalition Ordering by Size and Horizon Assignment

## E.1 Sketch of Justification Procedure

In this section, we will justify the insight behind the coalition ordering by size and horizon assignment, as shown in Definition D.2. In the beginning, we define a discrete-time stochastic process of TD error terms $( Z _ { n } ) _ { n \in \mathbb { N } _ { 0 } }$ , where $v _ { \pi } ( s _ { t } )$ denotes a generic state value function:

$$
Z _ { n + 1 } : = R _ { t + n } + \gamma v _ { \pi } ( s _ { t + n + 1 } ) - v _ { \pi } ( s _ { t + n } ) .
$$

As a result, the n-step return $G _ { t : t + n }$ has the following recursive relationship:

$$
G _ { t : t + n + 1 } = G _ { t : t + n } + \gamma ^ { n } Z _ { n + 1 } .
$$

We first show in Lemma E.2 that, given $s _ { t } = s ,$ the variance of the n-step return is non-decreasing with the return horizon when the TD-error sequence $\left( Z _ { n } \right)$ has non-negative autocovariances. Specifically, extending the horizon from n $\tan n + 1$ adds the discounted TD-error term $\gamma ^ { n } Z _ { n + 1 }$ , whose covariance with the existing return is non-negative. The return variance therefore increases strictly when this additional term has positive conditional variance and remains unchanged when it has zero conditional variance. Then, we make the coalition-variance monotonicity assumption in Assumption E.3, which can be justified by Remark E.4 with evidence from the definition of the game structure for the NAHT and a verification experiment that partially supports Assumption E.3. In summary, Assumption E.3 concerns the following situation: The number of agents increases when more exploration is conducted $( \mathrm { i . e . , } C _ { i }$ becomes larger in sizes), so more diverse agent types and behaviours emerge, resulting in that the estimation for the corresponding $V ( C _ { i } , s )$ has a higher variance.

By Lemma E.2 and Assumption E.3, we prove in Proposition E.5 that for the coalition enumeration {i} there exist non-decreasing orderings of variances for both the sequence of basis games of coalitions non-decreasing by sizes and the sequence of n-step returns, under the horizon schedule $n _ { i } = n _ { i - 1 } + 1$ . This motivates us to define the following term under this monotone-variance regularity:

$$
V ( C _ { i } , s ) : = \mathbb { E } _ { \pi } [ G _ { t : t + n _ { i } } \mid s _ { t } = s ] , \qquad ( i = 1 , \ldots , m ) .
$$

From the perspective of the coalition enumeration, the horizon schedule controls how the number of agents increases. The justification of Definition D.2 is completed.

Remark E.1. Among all sequences $\left( \tilde { n } _ { i } \right)$ with $\tilde { n } _ { 1 } = 1 , \tilde { n } _ { i } > \tilde { n } _ { i - 1 } ,$ and $\tilde { n } _ { i } \geq | C _ { i } |$ for all i, the schedule in Definition D.2 is pointwise minimal: $\tilde { n } _ { i } \geq n _ { i } f o r e \nu e r y i .$

Remark E.1 highlights that alternative horizon schedules $( \mathrm { e . g . } , n _ { i } = n _ { i - 1 } + k , n _ { 1 } = 1 , k > 0 )$ exist, which offers additional design possibilities within our framework.

## E.2 Mathematical Results Supporting for Justification

Lemma E.2. $H ( Z _ { n } )$ has non-negative autocovariances given $s _ { t } = s ,$ then $n \mapsto \operatorname { V a r } ( G _ { t : t + n } \mid s )$ is non-decreasing, with Var $\cdot ( G _ { t : t + n + 1 } \mid s ) = \operatorname { V a r } ( G _ { t : t + n } \mid { \overset { } { s } } )$ if and only $i f \mathrm { V a r } ( Z _ { n + 1 } \mid s ) = 0 .$

Proof. Use $G _ { t : t + n + 1 } = G _ { t : t + n } + \gamma ^ { n } Z _ { n + 1 }$ and expand the variance conditioning on $s _ { t } = s , \mathrm { w e }$ can get the following formula:

$$
\mathrm { V a r } ( G _ { t ; t + n + 1 } \mid s ) = \mathrm { V a r } ( G _ { t ; t + n } \mid s ) + 2 \gamma ^ { n } \mathrm { C o v } \big ( G _ { t ; t + n } , Z _ { n + 1 } \mid s \big ) + \gamma ^ { 2 n } \mathrm { V a r } ( Z _ { n + 1 } \mid s ) .
$$

We first prove that $\textbf { i f } \left( Z _ { n } \right)$ has non-negative autocovariances given $\begin{array} { r } { s _ { t } = s , } \end{array}$ then $n \mapsto \operatorname { V a r } ( G _ { t : t + n } \mid s )$ is non-decreasing, as follows:

Using the sum of TD errors to represent the Monte Carlo error [Sutton and Barto, 2018][Chap. 6], we have the following formula:

$$
G _ { t : t + n } - v _ { \pi } ( s _ { t } ) = \sum _ { k = 0 } ^ { n - 1 } \gamma ^ { k } Z _ { k + 1 } .
$$

Given the linear property of covariance and a fact that the covariance of a constant is zero, we can get the following equation:

$$
\operatorname { C o v } { \bigl ( } G _ { t : t + n } , Z _ { n + 1 } \mid s { \bigr ) } = \sum _ { k = 0 } ^ { n - 1 } \gamma ^ { k } \operatorname { C o v } { \bigl ( } Z _ { k + 1 } , Z _ { n + 1 } \mid s { \bigr ) } .
$$

Therefore, if all autocovariances $\operatorname { C o v } ( Z _ { i } , Z _ { j } \mid s ) \ge 0$ , then $\mathrm { C o v } \bigl ( G _ { t : t + n } , Z _ { n + 1 } \mid s \bigr ) \geq 0$

Since $\operatorname { V a r } ( Z _ { n + 1 } \mid s ) \geq 0$ must hold, we can directly get:

$$
\operatorname { V a r } ( G _ { t : t + n + 1 } \mid s ) \geq \operatorname { V a r } ( G _ { t : t + n } \mid s ) .
$$

That is, $n \mapsto \operatorname { V a r } ( G _ { t : t + n } \mid s )$ is non-decreasing.

We next prove Var $\left( G _ { t : t + n + 1 } \mid s \right) = \mathbf { V a r } ( G _ { t : t + n } \mid s )$ if and only ${ \mathbf i } \mathbf { f } \mathbf { V } \mathbf { a r } ( Z _ { n + 1 } \mid s ) = \mathbf { 0 } ,$ as follows:

Given a fact that $\operatorname { V a r } ( G _ { t : t + n + 1 } \mid s ) = \operatorname { V a r } ( G _ { t : t + n } \mid s )$ if and only if the following condition holds:

$$
\mathrm { V a r } ( Z _ { n + 1 } \mid s ) = 0 , \qquad \mathrm { C o v } \bigl ( G _ { t : t + n } , Z _ { n + 1 } \mid s \bigr ) = 0 ,
$$

we now aim to prove that $\operatorname { V a r } ( Z _ { n + 1 } \mid s ) = 0$ is sufficient for $\mathrm { C o v } ( G _ { t : t + n } , Z _ { n + 1 } \mid s ) = 0$ , and the proof is completed:

If a non-negative random variable (r.v.) $W \geq 0$ satisfies $\mathbb { E } \left[ W \mid s \right] = 0$ , then $W = 0$ almost surely (a.s.) given s [Williams, 1991][Chap. 9].

Apply this with $W = ( Z _ { n + 1 } - \mathbb { E } \left[ Z _ { n + 1 } \mid s \right] ) ^ { 2 }$ . If $\operatorname { V a r } ( Z _ { n + 1 } \mid s ) = 0$ then

$$
\begin{array} { r } { \mathbb { E } \left[ ( Z _ { n + 1 } - \mathbb { E } \left[ Z _ { n + 1 } \mid s \right] ) ^ { 2 } \mid s \right] = 0 \Rightarrow ( Z _ { n + 1 } - \mathbb { E } \left[ Z _ { n + 1 } \mid s \right] ) ^ { 2 } = 0 \mathrm { a } . s . \Rightarrow Z _ { n + 1 } = \mathbb { E } \left[ Z _ { n + 1 } \mid s \right] \mid \infty . } \end{array}
$$

The covariance $\operatorname { C o v } ( G _ { t : t + n } , Z _ { n + 1 } \mid s )$ can be expressed as the following formula:

$$
\operatorname { C o v } \bigl ( G _ { t : t + n } , Z _ { n + 1 } \mid s \bigr ) = \mathbb { E } \left[ \bigl ( G _ { t : t + n } - \mathbb { E } [ G _ { t : t + n } \mid s ] \bigr ) \bigl ( Z _ { n + 1 } - \mathbb { E } [ Z _ { n + 1 } \mid s ] \bigr ) \mid s \right] .
$$

Since $Z _ { n + 1 } - \mathbb { E } \left[ Z _ { n + 1 } \mid s \right] = 0 \mathrm { a . s }$ ., we get the following result:

$$
\operatorname { C o v } ( G _ { t : t + n } , Z _ { n + 1 } \mid s ) = 0 .
$$

Assumption E.3 (Coalition-Variance Monotonicity). Let $\widehat { V } ( C , s )$ be an unbiased estimator of $v _ { \pi } ( s )$ from on-policy data restricted to agents in C. $\operatorname { V a r } ( \widehat { V } ( C , s ) )$ is non-decreasing in |C|.

Remark E.4 (Explanation and Verification for Assumption E.3). Recall from Section C that $v _ { C , s } ^ { v _ { s } ( { \mathcal { M } } ) } = v _ { s } ( { \mathcal { M } } ) v _ { C , s } ^ { 1 }$ . Following Definition 3.1, we can express $v _ { C , s } ^ { v _ { s } ( { \mathcal { M } } ) }$ as follows:

$$
v _ { C , s } ^ { v _ { s } ( { \cal M } ) } ( D ) = \left\{ \begin{array} { l l } { v _ { s } ( { \cal M } ) } & { i f C \subseteq D , } \\ { 0 } & { o t h e r w i s e . } \end{array} \right.
$$

As described in Definition 3.2, a basis game $v _ { C , s } ^ { v _ { s } ( \mathcal { M } ) } ( D )$ can effectively evaluate the performance of a coalition of agents $C ^ { 9 }$ with an agent-type composition denoted by $\times _ { i \in D } T _ { i }$ , where $T _ { i }$ indicates the agent i's type. Therefore, we can get the following result: The larger the size of coalition $C ,$ the more agent-type compositions are required to be explored during agent interactions, so that the value $o f V ( C , s ) : = v _ { C , s } ^ { v _ { s } ( { \mathcal { M } } ) } ( { \mathcal { M } } )$ can predict $v _ { s } ( \mathcal { M } )$ . In other words, as the size of coalition C increases, higher variance of the value prediction is incurred to get $v _ { s } ( \mathcal { M } )$ , due to the growing agent-type composition space. Similarly, if the sizes of two coalitions are equal, their variances are equal.

To make Assumption E.3 more convincing, we additionally provide a new proof-of-concept experiment to demonstrate its reasonableness. The key idea behind Assumption E.3 is that a larger coalition $C _ { i }$ is associated with more complicated interactions among various types of agents, so there is a larger functional class to describe the $V ( C _ { i } , s )$ and therefore the variance during training would be larger (for searching the appropriate function).

Because we are not allowed to change the number of agents in the existing environment (which is fixed with pre-trained teammates), we partially prove Assumption E.3 by showing the relation between the agent-type compositions and variances: We would like to show that when the number of agent-type compositions increases, the variance will not decrease. We design three settings denoted as Exp A, Exp B and Exp C to capture different numbers of agent-type compositions. The number of agent-type compositions in descent would be: $E x p A \geq E x p B \geq E x p C ,$ and the variance should be in the same order in expectation.10

9For any team of agents $D \supset C .$ the basis game would be the same as $v _ { C , s } ^ { v _ { s } ( \mathcal { M } ) } ( C )$ , so it is ineffective to evaluate the performance of the D with the larger size than $\bar { C } .$

![](images/713aa1374c657bdb91c4147f1d0029d6133895de952840356b5e2e4130898854.jpg)  
Figure 7: Empirical verification for Assumption E.3.

In Figure 7, it can be observed that the relation between variances of three setups above is almost always as the following order: Exp A ≥ Exp $B \geq E x p C ,$ over the whole training procedure $( i . e .$ Exp A bounded above by Exp B, and Exp B bounded above by Exp $C ) . ^ { 1 1 }$ This result conforms to our expectation above and thus partially veriies the reasonableness of Assumption E.3.

Proposition E.5 (Variance Monotonicity along the Coalition Enumeration). $F i x \ : s \in { \mathcal { S } }$ and the sequence $( C _ { i } , n _ { i } )$ of Deinition D.2. Under the non-negative autocovariance condition of Lemma E.2 and Assumption E.3, the sequences

$$
i \mapsto \operatorname { V a r } ( { \widehat { V } } ( C _ { i } , s ) ) \quad a n d \quad i \mapsto \operatorname { V a r } ( G _ { t : t + n _ { i } } \mid s )
$$

are both non-decreasing in i. Moreover, $i \mapsto \operatorname { V a r } ( G _ { t : t + n _ { i } } \mid s )$ is strictly increasing at every index where Var $\cdot ( Z _ { n _ { i } + 1 } \mid s ) \stackrel { - } { > } 0 .$

Proof. Since $n _ { i } = n _ { i - 1 } + 1$ is strictly increasing, Lemma E.2 yields non-decreasing variance in i, with strictness when the added step contributes nonzero variance. Assumption E.3 gives non-decreasing $\operatorname { V a r } ( \widehat { V } ( C _ { i } , s ) )$ as |Ci] (hence i) increases. □

## F Proof for Results of Incorporating Linearity Axiom

Proposition F.1. Given the expression of state-specific basis games in Eq. (4), applying the Linearity axiom to state-specific cooperative games expressed as Eq. (2) can lead to the following equation:

$$
\phi ( V ( \mathcal { M } , s _ { t } ) ) = \mathbb { E } _ { \boldsymbol { \pi } } \left[ \sum _ { n = 1 } ^ { m } k _ { C _ { n } } ^ { \prime } \cdot \sum _ { \tau = 0 } ^ { n - 1 } \gamma ^ { \tau } \phi ( R _ { t + \tau } ) + \gamma ^ { n } \phi ( V ( \mathcal { M } , s _ { t + n } ) ) \middle | s _ { t } \right] .
$$

Proof. By Lemma C.1, a state-specific cooperative game expressed as $\operatorname { E q . } \left( 2 \right)$ can be equivalently represented by the form: $\begin{array} { r } { \boldsymbol { v } _ { s } = \bar { \sum } _ { \emptyset \neq C \subseteq \mathcal { M } } \boldsymbol { \bar { k } } _ { C } ^ { \prime } \cdot \boldsymbol { v } _ { C , s } ^ { v _ { s } ( \bar { \mathcal { M } } ) } } \end{array}$ with setting $z = v _ { s } ( { \mathcal { M } } )$ and $\begin{array} { r } { k _ { C } ^ { \prime } = \frac { k _ { C } } { v _ { s } ( \mathcal { M } ) } } \end{array}$ and $v _ { s } ( \mathcal { M } ) \neq 0 . ^ { 1 2 }$ Herein, the set of $v _ { C , s } ^ { v _ { s } ( { \mathcal { M } } ) }$ become a set of new state-specific basis games and $k _ { C } ^ { \prime }$ are their corresponding weightings.

Introducing a linear multidimensional operator $\phi : { \mathcal { G } } ( s ) \to \mathbb { R } ^ { M } ~ ( { \mathrm { i . e . } }$ , supposing that φ satisfies the Linearity axiom) on a state-specific cooperative game $v _ { s }$ defined by Lemma C.1, we derive the following formula:

$$
\phi ( v _ { s } ) = \sum _ { \emptyset \neq C \subseteq M } k _ { C } ^ { \prime } \cdot \phi \left( v _ { C , s } ^ { v _ { s } ( { \cal M } ) } \right) .\tag{19}
$$

Recall that $v _ { C _ { n } , s _ { t } } ^ { v _ { s _ { t } } ( \mathcal { M } ) } ( \mathcal { M } )$ is denoted by $V ( C _ { n } , s _ { t } )$ , and $v _ { s _ { t } } ( \mathcal { M } )$ is denoted by $V ( \mathcal { M } , s _ { t } )$ . Substituting Eq. (4) into the Linearity axiom in Eq. (19) and applying linearity of the expectation operator, we obtain the following formula:

$$
\phi ( V ( \mathcal { M } , s _ { t } ) ) = \mathbb { E } _ { \boldsymbol { \pi } } \left[ \sum _ { n = 1 } ^ { m } k _ { C _ { n } } ^ { \prime } \cdot \sum _ { \tau = 0 } ^ { n - 1 } \gamma ^ { \tau } \phi ( R _ { t + \tau } ) + \gamma ^ { n } \phi ( V ( \mathcal { M } , s _ { t + n } ) ) \middle | s _ { t } \right] .
$$

## G Proof for Results of Incorporating Efficiency and Symmetry Axioms

Proposition G.1 (Representation of Transformed Rewards). Given the condition that $\begin{array} { r } { \sum _ { i = 1 } ^ { \bar { M } } \phi _ { i } ( R _ { t } ) = R _ { t } } \end{array}$ , each agent's payoff allocation defined on rewards $R _ { t } , i . e . , \phi _ { i } ( R _ { t } )$ can be expressed as:

$$
\phi _ { i } ( R _ { t } ) : = R _ { t } - \sum _ { j \neq i } \left( \phi _ { j } ( V ( \mathcal { M } , s _ { t } ) ) - \gamma \phi _ { j } ( V ( \mathcal { M } , s _ { t + 1 } ) ) \right) .
$$

Proof. Recall that $\boldsymbol { \phi } ( \cdot ) \in \mathbb { R } ^ { M }$ is a multidimensional linear transformation,13 where M is the number of agents. We now express $\phi ( R _ { t } )$ according to the Efficiency axiom.

To satisfy the Efficiency axiom such that $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \phi _ { i } ( V ( \mathcal { M } , s _ { t } ) ) = V ( \mathcal { M } , s _ { t } ) } \end{array}$ , it is reasonable to assume that $\textstyle \sum _ { i = 1 } ^ { M } \phi _ { i } ( R _ { t } ) = R _ { t }$ . In other words, each agent's value expansion can be expressed independently in its own $\dot { \phi } _ { i } ( R _ { t } )$ , justified by Theorem 4.5.

Next, we aim to show how each $\phi _ { i } ( R _ { t } )$ is expressed in terms of $\phi ( V ( \mathcal { M } , s _ { t } ) )$ 1

It is not difficult to observe that for each agent $i \in { \mathcal { M } }$ , we have

$$
\phi _ { i } ( V ( \boldsymbol { \mathcal { M } } , s _ { t } ) ) = \phi _ { i } ( R _ { t } ) + \gamma \phi _ { i } ( V ( \boldsymbol { \mathcal { M } } , s _ { t + 1 } ) ) .
$$

By the condition $\begin{array} { r } { \sum _ { i = 1 } ^ { M } \phi _ { i } ( R _ { t } ) = R _ { t } } \end{array}$ , we can derive a formula such that:

$$
\phi _ { i } ( R _ { t } ) = R _ { t } - \sum _ { j \neq i } \phi _ { j } ( R _ { t } ) .
$$

Since $\phi _ { j } ( R _ { t } ) = \phi _ { j } ( V ( \mathcal { M } , s _ { t } ) ) - \gamma \phi _ { j } ( V ( \mathcal { M } , s _ { t + 1 } ) )$ , we can express $\phi _ { i } ( R _ { t } )$ as follows:

$$
\phi _ { i } ( R _ { t } ) : = R _ { t } - \sum _ { j \neq i } \left( \ \phi _ { j } ( V ( \mathcal { M } , s _ { t } ) ) - \gamma \phi _ { j } ( V ( \mathcal { M } , s _ { t + 1 } ) ) \ ) . \right.
$$

Theorem G.2. The payoff allocation operator satisfying permutation-equivariance is a sufficient condition for the Symmetry axiom.

Proof. Let $\mathcal { M } = \{ 1 , \dots , M \}$ . Let G be the set of cooperative games $v : 2 ^ { \mathcal { M } } \to \mathbb { R }$ , and $S _ { C }$ be a permutation group acting on $\dot { C } \subseteq { \mathcal { M } }$

Let $S _ { C }$ act on an arbitrary $C \subseteq M$ by exchanging agents:

$$
( \sigma \cdot v ) ( C ) : = v ( \sigma ^ { - 1 } ( C ) ) , \qquad \sigma \in S _ { \mathcal { M } } , C \subseteq \mathcal { M } .
$$

Let a payoff allocation operator $F : \mathcal { G } \to \mathbb { R } ^ { M }$ output payoffs $F ( v ) = ( F _ { 1 } ( v ) , \ldots , F _ { n } ( v ) )$

Let $S _ { \mathcal { M } }$ act on $\mathbb { R } ^ { M }$ by permuting coordinates: $( \sigma \cdot x ) _ { i } : = x _ { \sigma ^ { - 1 } ( i ) }$

F is permutation-equivariant if

$$
{ \cal F } ( \sigma \cdot v ) = \sigma \cdot { \cal F } ( v ) , \qquad \forall \sigma \in S _ { \cal M } , v \in { \cal G } .\tag{20}
$$

Recall that agents i and j are symmetric in v if

$$
v ( C \cup \{ i \} ) = v ( C \cup \{ j \} ) , \quad \forall C \subseteq { \mathcal { M } } \setminus \{ i , j \} .
$$

Equivalently, the transposition $\tau = ( i j )$ leaves the game v invariant:

$$
\tau \cdot v \ = \ v .\tag{21}
$$

Next, we aim to show $F _ { i } ( v ) = F _ { j } ( v )$ to prove the statement in the theorem.

By Eqs. (20) and (21), we have

$$
F ( v ) = F ( \tau { \cdot } v ) = \tau { \cdot } F ( v ) .
$$

Note that $\tau { \cdot } F ( v )$ is just $F ( v )$ with coordinates i and j swapped. Therefore, $F ( v ) = \tau { \cdot } F ( v )$ implies the i-th and j-th entries are equal:

$$
F _ { i } ( v ) = F _ { j } ( v ) .
$$

Theorem G.3. Shapley Machine is an algorithm enforcing $V _ { i }$ to fulfil Efficiency, Symmetry and Linearity, so the $V _ { i }$ is the Shapley value for Dec-POMDPs.

Proof. By construction, the Shapley Machine enforces the individual value functions Vi (payoff allocation functions) to satisfy the axioms of Efficiency, Symmetry and Linearity. Each state-specific cooperative game space $\mathcal G ( s )$ that aims to meet the requirement of Dec-POMDPs is isomorphic to the canonical cooperative game space G. Hence, the properties of payoff allocation functions that hold on G also hold on $\mathcal { G } ( s )$

Now, recall that the Shapley value is uniquely characterized as the value function that satisfies the axioms of Efficiency, Symmetry and Additivity (Theorem 2.2). Since Linearity is a stronger condition than Additivity (indeed, setting all $\alpha _ { i } = 1$ in the definition of Linearity yields Additivity), any payoff allocation function that is Efficient, Symmetric, and Linear must coincide with the Shapley value. Therefore, the individual value functions $V _ { i }$ produced by the Shapley Machine are exactly the Shapley values for Dec-POMDPs. □

## H Experimental Details

## H.1 Implementation Details

Our algorithm is built upon POAM and IPPO. All loss functions for training the encoder-decoder model, policy networks and value networks have been remained. Please refer to Appendix B.4 for details. For conciseness, we only list the novel loss functions proposed in this paper.

## H.1.1 Shapley Machine

We now describe the details about implementing our proposed algorithm, referred to as Shapley Machine. In general, our algorithm is established based on the base algorithms POAM and IPPO, with modification to fulfil all the three axioms of the Shapley value: Efficiency, Linearity and Symmetry Since the Symmetry axiom has been realised in POAM and IPPO, we only need to fulfil the Efficiency and Linearity axioms.

Implementing the Linearity Axiom. In general, both POAM and IPPO have implemented TD(λ) which does not strictly conform to the principle of the Linearity axiom. To address this problem, we change the TD(λ) to TTD(λ), where the number of n-step return components is equal to the number of the non-empty coalition basis games, as informed by our theory. Note that in some scenarios the episode length is smaller than the number of non-empty coalition basis games. In these cases, the TTD(λ) can be reduced to the TD(λ) for the finite-horizon tasks with the episode length as T, equivalently, the TTD(λ) with $m = T$ . Alternatively, we can select a value of m for each task based on extra conditions, which is left for the future work. In this paper, we set $m = T$ for the scenarios 8v9 and 10v11. For the 8v9 scenario, the episode length as 120 is not too far from the number of non-empty coalition basis games as 127. For the 10v11 scenario, the episode length is 150, while the number of non-empty coalition basis games is 511.

Implementing the Efficiency Axiom. By introducing the partial observation in Dec-POMDPs, the Eq. (8) we have derived the condition for realizing the Efficiency axiom is transformed as follows:

$$
\phi _ { i } ( R _ { t } ) = R _ { t } - \sum _ { j \neq i } \left( V _ { j } ( h _ { t , j } ) - \gamma V _ { j } ( h _ { t + 1 , j } ) \right) ,
$$

where $V _ { j } ( h _ { t , j } )$ indicates an agent $j ^ { \circ } \mathbf { s }$ non-parameterized individual value. During the practical training procedure, the $V _ { j } ( h _ { t , j } )$ generated by the individual value network could be severely inaccurate in the beginning, which may result in the instability of learning. To mitigate this issue, we add an extra coefficient α $\in ( 0 , 1 )$ to the term $\begin{array} { r } { \sum _ { j \neq i } \big ( V _ { j } ( h _ { t , j } \mathbf { \bar { \rho } } ) - \gamma V _ { j } ( h _ { t + 1 , j } ^ { - } ) \big ) } \end{array}$ , such that:

$$
\phi _ { i } ( R _ { t } ) = R _ { t } - \alpha \sum _ { j \neq i } \left( V _ { j } ( h _ { t , j } ) - \gamma V _ { j } ( h _ { t + 1 , j } ) \right) .\tag{22}
$$

This coefficient α can be either manually set up as a fixed value, or implemented by a scheduler starting from 0 to some preset upper limit. The $R _ { t }$ in $L _ { \theta } ( h _ { t , i } , a _ { t , i } )$ and $L _ { \theta ^ { c } } ( h _ { t , i } )$ is replaced by the above $\phi _ { i } ( R _ { t } )$ . To clarify this change, the two new losses are expressed as: $\hat { L } _ { \theta } ( h _ { t , i } , a _ { t , i } )$ and $\hat { L } _ { \theta ^ { c } } ( h _ { t , i } )$

Furthermore, it is needed to search the underlying factorization scheme of the transition function in Dec-POMDPs, according to Theorem 4.5. To implement this requirement, we need to fulfil the following condition:

$$
\sum _ { i = 1 } ^ { M } V _ { i } ( h _ { t , i } ) = V ( \mathcal { M } , h _ { t } ) .\tag{23}
$$

The above equality is implemented as a regularization term during training.

To maintain consistency with the critic losses, we consider to use the λ-return denoted by $\hat { G } _ { t } ^ { \lambda }$ to represent $V ( \mathcal { M } , h _ { t } )$ . In turn, the above equation can be expressed as:

$$
V ( \mathcal { M } , h _ { t } ) = \mathbb { E } _ { \pi } \left[ G _ { t } ^ { \lambda } \vert h _ { t } \right] ,
$$

where each $\hat { G } _ { t : t + n }$ contributing to the $\hat { G } _ { t } ^ { \lambda }$ represented in TTD(λ) is expressed as follows:

$$
\begin{array} { r l } { \hat { G } _ { t : t + n } = R _ { t } + \gamma R _ { t + 1 } + \dots + \gamma ^ { n - 1 } R _ { t + n - 1 } + \gamma ^ { n } V ( \mathcal { M } , h _ { t + n } ) } & { } \\ { = R _ { t } + \gamma R _ { t + 1 } + \dots + \gamma ^ { n - 1 } R _ { t + n - 1 } + \gamma ^ { n } \displaystyle \sum _ { i = 1 } ^ { M } V _ { i } ( h _ { t + n , i } ) } & { ( \mathrm { B y ~ E q . ~ } ( 2 3 ) ) . } \end{array}
$$

Substituting the above formula into Eq. (23), we can obtain the regularization term, referred to as the efficiency loss for one timestep t, as follows:

$$
L _ { \theta ^ { c } } ^ { e } ( h _ { t , i } ) = \frac { 1 } { 2 } \left( \hat { G } _ { t } ^ { \lambda } - \sum _ { i = 1 } ^ { M } V _ { i } ^ { \theta ^ { c } } ( h _ { t , i } ) \right) ^ { 2 } .
$$

In summary, the total loss function of the Shapley Machine for POAM (SM-POAM) is as follows:

$$
L _ { \mathrm { S U - p o u n } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \sum _ { i \in \mathcal { C } _ { t } } \hat { L } _ { \theta } ( h _ { t , i } , a _ { t , i } ) + \beta _ { 1 } \sum _ { i \in \mathcal { M } } \hat { L } _ { \theta ^ { c } } ( h _ { t , i } ) + \beta _ { 2 } L _ { \theta ^ { c } } ^ { e } ( h _ { t , i } ) + L _ { \theta ^ { c } , \theta ^ { o } , \theta ^ { a } } ( h _ { t , i } , o _ { t , - i } , a _ { t , - i } ) \right) ,
$$

where $T$ is the episode length; $\beta _ { 1 } , \beta _ { 2 } \in ( 0 , 1 )$ are two coefficients to control the importance of the two losses; $\mathcal { C } _ { t }$ indicates the controlled agent set at timestep t and M indicates the ad hoc team following the convention in Wang et al. [2025].

Similarly, the total loss function of the Shapley Machine for IPPO (SM-IPPO) is as follows:

$$
L _ { \mathtt { S M - I P P 0 } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \sum _ { i \in \mathcal { C } _ { t } } \hat { L } _ { \theta } ( h _ { t , i } , a _ { t , i } ) + \beta _ { 1 } \sum _ { i \in \mathcal { M } } \hat { L } _ { \theta ^ { c } } ( h _ { t , i } ) + \beta _ { 2 } L _ { \theta ^ { c } } ^ { e } ( h _ { t , i } ) \right) .
$$

Implementation of $k _ { C } ^ { \prime } > 0$ . As mentioned in Section C, it is necessary to fulfil $k _ { C } ^ { \prime } > 0$ for reaching cooperation, which can be implemented following the convention of TD(λ) in the RL [Sutton, 1988]. Specifically, the weightings for m basis games $( k _ { C _ { 1 } } ^ { \prime } , k _ { C _ { 2 } } ^ { \prime } , \cdot \cdot \cdot , k _ { C _ { m } } ^ { \prime } )$ are generated using a geometric distribution $P _ { \lambda }$ with the parameter $0 < \lambda < 1$ , such that $k _ { C _ { n } } ^ { \prime } = \overset { \cdot } { P _ { \lambda } } ( n )$ , resulting a tuple $( ( 1 - \lambda ) , ( 1 - \lambda ) \lambda , \cdot \cdot \cdot , ( 1 - \lambda ) \lambda ^ { m - 1 } , \lambda ^ { m } )$ . With this condition, Eq. (6) becomes the well-known truncated TD(λ), shortened as TTD(λ) [Cichosz, 1994].

Partial Observations. In practice, a controlled agent is only able to receive an observation, following the settings of Dec-POMDPs. Therefore, an agent is required to infer the state of the environment as an individual hidden state, through the history of observations. To this end, the policy and individual networks are realized by recurrent neural networks (RNNs) (e.g., GRUs [Chung et al., 2014]), where observations or representations transformed from observations are used as inputs.

Implementation of the Symmetry Axiom. It can be observed that both POAM and IPPO are implemented by the sharing parameters, with agent ID to differentiate agent identities. Given an agent ID, its input is either individual observations or individual observations + teammate embeddings. where the teammate embeddings are transformed from individual observations through an encoder network.

Estimation of Uncontrolled Agent Individual Values. We maintain all possible controlled agent policy and individual value networks as implemented by Wang et al. [2025], given that the maximum number of controlled agents is known in the experimental settings of the NAHT, but only part of controlled agents in an episode can make decision. Thanks to the parameter sharing technique, this can be simply implemented by maintaining only a policy and/or an individual value network, with a set of agent IDs to represent and differentiate all possible controlled agents. During the training phase, the uncontrolled agents' individual values are also required to implement the Efficiency axiom, as informed in Eq. (9), but they are unknown to the controlled agents; all training paradigms that we design should be based on the view of the controlled agents. To address this problem, the uncontrolled agents' individual values are approximated by the maintained individual value function with the agent IDs of those inactive controlled agents, given that the observations of the uncontrolled agents are available for controlled agents during training.

Example H.1 (Estimating an Uncontrolled Agent's Individual Value). Consider a team $\mathcal { M } =$ {c1, c2, u} consisting of two controlled agents, $\breve { \mathscr { C } } = \{ c _ { 1 } , c _ { 2 } \}$ , and one uncontrolled agent, $\mathcal { U } = \{ \boldsymbol { u } \}$ Suppose the implementation maintains a shared policy network and a shared individual value network, with three available controlled-agent IDs: {1, 2, 3}. In an episode, $c _ { 1 }$ and $c _ { 2 }$ use IDs 1 and 2, respectively, leaving ID 3 unused.

Let $V _ { \psi } ( h , i )$ denote the shared individual value network evaluated using observation history h and agent $\dot { I D i }$ . Since the uncontrolled agent's observations are available during training, we approximate its individual value by supplying its observation history to the shared network with the unused ID:

$$
\widehat { V } _ { u } ( h _ { t , u } ) : = V _ { \psi } ( h _ { t , u } , 3 ) .
$$

As a result, the controlled agent associated with ID 3 does not participate in the episode, but its ID is used to obtain a proxy value estimate for u.

## H.1.2 POAM and IPPO

The implementations of POAM and IPPO have been detailed in Appendix B.4.

In summary, the total loss function of POAM is as follows:

$$
L _ { \mathtt { P 0 A M } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \sum _ { i \in \mathcal { C } _ { t } } L _ { \theta } ( h _ { t , i } , a _ { t , i } ) + \beta _ { 1 } \sum _ { i \in \mathcal { M } } L _ { \theta ^ { c } } ( h _ { t , i } ) + L _ { \theta ^ { c } , \theta ^ { c } , \theta ^ { a } } ( h _ { t , i } , o _ { t , - i } , a _ { t , - i } ) \right) .
$$

The total loss function of IPPO is as follows:

$$
L _ { \tt I P P 0 } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \sum _ { i \in \mathcal { C } _ { t } } L _ { \theta } ( h _ { t , i } , a _ { t , i } ) + \beta _ { 1 } \sum _ { i \in \mathcal { M } } L _ { \theta ^ { c } } ( h _ { t , i } ) \right) .
$$

## H.1.3 Banzhaf Machine

The overall loss function and implementation of the Banzhaf Machine closely follow those of POAM and IPPO. The only distinction is that Banzhaf Machine employs the TTD(λ) in place of the TD(λ) that is applied in POAM and IPPO.

## H.2 Experimental Domains

We now briefly introduce the experimental domains for running experiments. If one would like to know more about details, please refer to Wang et al. [2025].

## H.2.1 MPE Predator Prey (MPE-PP)

The MPE Predator Prey (MPE-PP) environment is a predator-prey task implemented within the Multi-Agent Particle Environment (MPE) framework. It simulates interactions within a two-dimensional space populated by two static obstacles, where three pursuer agents must cooperate to capture a single adversarial evader. A successful capture is defined as at least two pursuers simultaneously colliding with the evader, upon which the pursuers receive a positive reward of +1. If the capture is unsuccessful, no reward is granted. This environment is designed to test the ability of agents to coordinate under spatial and dynamic constraints. The length of an episode is T = 100,

## H.2.2 The StarCraft Multi-Agent Challenge (SMAC)

The StarCraft Multi-Agent Challenge (SMAC) serves as a benchmark suite for evaluating CMARL algorithms in partially observable settings. Built on top of the StarCraft II game engine, SMAC presents a variety of micromanagement tasks where each agent (e.g., a Marine or Stalker) operates based on limited local observations and must coordinate actions with teammates to overcome enemy units. In this work, we focus on four specific SMAC scenarios: 5v6: Five allied Marines versus six enemy Marines, 8v9: Eight allied Marines versus nine enemy Marines, 10v11: Ten allied Marines versus eleven enemy Marines, 3s5z: Three allied Stalkers versus five enemy Zealots. At each timestep, agents receive a shaped reward proportional to the damage they inflict on opponents, along with bonus rewards of 10 points for each enemy defeated and 200 points for achieving overall victory by eliminating all adversaries. The total return is normalized so that the maximum achievable return in each scenario is 20. The action space in SMAC is discrete, enabling each agent to choose actions such as attacking a particular enemy, moving in a specific direction, or remaining idle. The length of an episode varies across different scenarios: 3sv5z with T = 250, 5v6 with $\bar { T } = 7 0$ , 8v9 with $T = 1 2 0$ and 10v11 with $T = 1 5 0$

## H.3 Evaluation Metrics

For in-distribution evaluation, the mean return (or winrate) is computed over E randomly sampled episodes. In each episode, we form a joint policy $\pi ^ { ( \mathcal { M } ) }$ by sampling N agents uniformly from C and the remaining M — N agents from U. For the out-of-distribution (OOD) evaluation, the M–N score [Wang et al., 2025] is applied. Given a set of controllable agents C and uncontrolled agents U, the M–N score measures how well mixed teams of these agents cooperate in the NAHT setting. The score is computed exhaustively by changing the number of controllable agents included in the team. Specifically, for each $N \in \{ 1 , \ldots , M - 1 \}$ , we form a joint policy $\pi ^ { ( \bar { \mathcal { M } } ) }$ by sampling N agents uniformly from C and the remaining M — N agents from U. The mixed teams are then evaluated on the task for E episodes. This procedure yields (M — 1) · E returns (or winrates) in total, whose average defines the M–N score.

In this experiment, each convention of an uncontrolled agent is a random seed that pretrains it. For the OOD evaluation on the unseen conventions, the M–N score is computed by averaging the perconvention M–N scores across 4 other conventions for each agent type. For the OOD evaluation on the unseen agent types, the per-convention M–N score is computed by averaging the per-convention M–N score across $^ { 5 }$ conventions for each unseen agent type. All results are obtained by first computing each metric with 128 episodes and then averaging these per-seed metrics across 5 random seeds, with 95% confidence intervals.

## H.4 Experimental Settings

The training procedure of the NAHT process is briefly introduced here. For more details, please refer to Wang et al. [2025]. For each scenario (e.g. MPE, 3sv5z, 5v6, 8v9 and 10v11), there are five groups of pretrained agents (e.g. IQL, MAPPO, VDN, QMIX and IPPO) acting as the uncontrolled agents. Given that each task specifies a fixed total number of agents M, the number of controlled agents is N. For each episode evaluation, the number of uncontrolled agents M — N is sampled, and then a group of pretrained agents is sampled.14 The distribution for all sampling procedures are modelled as the multinational distribution with no replacement. Each uncontrolled agent executes the greedy policy in both training and testing procedures. In contrast, each controlled agent executes the on-policy sampling via the parameterized policy during the training procedure, while the greedy policy during the testing procedure.

## H.5 Hyperparameter Settings

Since our algorithm is established based on POAM and IPPO, most hyperparameter settings follow that in Wang et al. [2025]. First, the actors and critics are implemented in recurrent neural networks, with full parameter sharing. Specifically, they are implemented by two fully connected layers followed by a GRU layer and an output layer. Each layer has 64 dimensions with a ReLU activation function, and layer normalization is applied. The encoder-decoder networks for inferring agent characteristics are also implemented in parameter sharing. The encoder is implemented by a GRU layer, followed by a fully connected layer with a ReLU activation function and an output layer. The decoder is implemented by two fully connected layers with ReLU activation functions, followed by an output layer. Adam Optimizer is used to train all models. The detailed hyperparameters for experiments are shown in Tables 1 and 2. For the scenarios such as MPE, 3sv5z and 5v6, the values of m are set as the number of non-empty coalition basis games. For the scenarios such as 8v9 and 10v11, the values of m are simply set as the length of an episode (see Appendix H.1 for more details). Note that the term $\begin{array} { r } { \sum _ { j \neq i } \left( V _ { j } \hat ( s _ { t } \right) - \gamma V _ { j } ( s _ { t + 1 } ) \hat ) } \end{array}$ in shaped rewards is standardised to match the scales of standardised rewards. To stabilize learning in large-scale scenarios, we add a value loss clip to Shapley Machine and Banzhaf Machine for POAM on both 8v9 and 10v11, and for IPPO on 10v11 only.

Table 1: Key hyperparameters of Shapley and Banzhaf Machine: λ indicates the parameter for the weighting functions of TTD(λ), m indicates the number of basis games, α controls the importance of the term $\begin{array} { r } { \sum _ { j \neq i } \left( V _ { j } ( s _ { t } ) - \gamma V _ { j } ( s _ { t + 1 } ) \right) } \end{array}$ in Eq. (22), $\beta _ { 1 }$ controls the importance of the critic loss $\textstyle \sum _ { i \in { \mathcal { M } } } { \hat { L } } _ { \theta ^ { c } } ( h _ { t , { \frac { i } { 2 } } } )$ , and $\beta _ { 2 }$ controls the importance of the efficiency loss $L _ { \theta ^ { c } } ^ { e } ( h _ { t , i } )$
<table><tr><td>Algorithm</td><td>Task</td><td> $\bar { \lambda }$ </td><td>m</td><td>α</td><td> $\overline { { \beta _ { 1 } } }$ </td><td> $\overline { { \beta _ { 2 } } }$ </td></tr><tr><td rowspan="3">Shapley Machine</td><td rowspan="3">MPE-PP 3sv5z 5v6 8v9</td><td>0.85</td><td>7</td><td>0.01</td><td>0.5</td><td>0.01</td></tr><tr><td>0.85</td><td>7</td><td>0.01</td><td>0.5</td><td>0.01</td></tr><tr><td>0.85 0.95</td><td>31 120</td><td>0.01 0.01</td><td>0.5 0.5</td><td>0.001 0.001</td></tr><tr><td rowspan="5">Banzhaf Machine</td><td>10v11</td><td>0.95</td><td>150</td><td>0.01</td><td>0.5</td><td>0.001</td></tr><tr><td>MPE-PP</td><td>0.85</td><td>7</td><td>N/A</td><td>0.5</td><td>N/A</td></tr><tr><td>3sv5z</td><td>0.85</td><td>7</td><td>N/A</td><td>0.5</td><td>N/A</td></tr><tr><td>5v6</td><td>0.85</td><td>31</td><td>N/A</td><td>0.5</td><td>N/A</td></tr><tr><td>8v9</td><td>0.95</td><td>120</td><td>N/A</td><td>0.5</td><td>N/A</td></tr><tr><td></td><td>10v11</td><td>0.95</td><td>150</td><td>N/A</td><td>0.5</td><td>N/A</td></tr></table>

Table 2: Common hyperparameters of RL settings.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>LR</td><td>0.0005</td></tr><tr><td>Epochs</td><td>5</td></tr><tr><td>Minibatches</td><td>1</td></tr><tr><td>Buffer size</td><td>256</td></tr><tr><td>Entropy coefficient</td><td>0.05</td></tr><tr><td>Clip</td><td>0.2</td></tr><tr><td>ED LR</td><td>0.0005</td></tr><tr><td>ED epochs</td><td>1</td></tr><tr><td>ED Minibatches</td><td>1</td></tr><tr><td>Optim_alpha (Adam)</td><td>0.99</td></tr><tr><td>Optim_eps (Adam)</td><td>0.00001</td></tr><tr><td>Use_obs_norm</td><td>True</td></tr><tr><td>Use_orthogonal_init</td><td>True</td></tr><tr><td>Use_adv_std</td><td>True</td></tr><tr><td>Standardise_rewards</td><td>True</td></tr><tr><td>num_parallel_envs</td><td>8</td></tr></table>

## H.6 Computational Resources

All experiments are conducted on Intel Xeon Gold 6230 CPUs and Nvidia V100-SXM2 GPUs. Each experiment run on MPE takes approximately 7 hours, utilizing 20 CPU cores and 1 GPU. Each experiment run on SMAC takes between 8 and 19 hours, utilizing 30 CPU cores and 1 GPU. All experiments are trained with 20M timesteps. The post-training evaluation for each scenario takes between 6 and 16 hours, utilizing 8 CPU cores and 1 GPU.

## I Additional Experiments

## I.1 Empirical Evidence for the Strength of the Axiomatic Framework

We now discuss the phenomenon observed from the performance under stochastic policy exploration during training. Discussing this phenomenon is meaningful: When a machine learning algorithm is deployed and has to adapt to real-world environments through online learning due to the mismatch between simulators and real-world environments, it is expected that random exploration diminishes rapidly over time to avoid the damage on real-world environments (e.g. physical systems).

Recall that the PPO family algorithms implement exploration by the natural stochastic policy sampling from the learned policy distribution and maximizing policy entropy. As seen from Figure 8, either Shapley Machine or Banzhaf Machine can outperform their base algorithm counterparts with a large margin. The reasons are as follows: As shown in Figure 9, policy entropy decreases rapidly over the course of training (except for the IPPO implementation on 3sv5z, which underperforms due to incompatibility between IPPO and the task). This indicates that the learned policies become more deterministic than those of the base algorithms, despite all methods being optimized with an entropy-maximization objective. A possible explanation for this phenomenon is illustrated in Figure 10: The critic losses of Shapley and Banzhaf Machines decrease faster than that of their base algorithm counterparts. This provides evidence for the advantage of structured individual value functions under our proposed axiomatic framework.

## I.2 Analyzing Effects of Efficiency Axioms

We now demonstrate the association between the changing of the shaping rewards and the response of corresponding testing returns. As shown in Figure 13, the shaped reward increases in the beginning and then gradually reaches a plateau. The efficiency loss also exhibits the similar phenomenon. As shown in Figure 14, it can be observed that when instability happens during training as evidenced by the sudden changes of test returns, the shaped reward can consistently manifest this situation (as highlighted in red vertical dashed lines). This feature is crucial in the NAHT, since it could be a

![](images/703cb44f4984cb3cff0a85787fa7dd2cb21dbac0b8b5f333a3a9a2a9ab921daf.jpg)

![](images/f6f4924b2ddde04eff8bf288c363976979028b277745aa22b993cc8a1b0830b9.jpg)

![](images/fbbd821ce03205df8850e3a38a9951606abce3c7b540ab0aa380dcdbb56d3363.jpg)

![](images/21a1cc48fb78769e787f2c562471167ff857b84a563a345da2d373104f636867.jpg)  
Figure 8: Performance under the stochastic exploration policy during training in NAHT.

![](images/d9fde039540c956adcbd4a9c6d7d41972a24f731e20b3472340791f2c410ca8b.jpg)

![](images/d0317810bc8f69da1edb2a542d2f94dd2714aa2c9df1d417020253f4c1d0731a.jpg)

![](images/d57cc142c6eaabe0c300a227fff647d0a15c4b9ffe08c6da9460c2418dcc3133.jpg)

![](images/5090db0c164f757d68b8f8776c588d1fbc2af699add4e27a33ccbb01f5732440.jpg)  
Figure 9: Policy entropy across scenarios during training in NAHT.

![](images/09195a097346eb1214fab9cffb753c0da1b59eba5c1976c0c44914afd7c993df.jpg)

![](images/f39a5b96a16c99c78a432c124655bfefab804aaf2fc90430115aecb0e377c84e.jpg)

![](images/55711cfc81dc875af9b4552625d7dd61ca15ac63d49f861452b581447757c993.jpg)

![](images/e7b03e6a25df43aa011e73395ef86c97a37d2913c1bc8e61d04fd0cfc6d271e8.jpg)

![](images/5f0ff103e695a755aa97dddc289d7cc0603b05db609e56b9b906863f48d01ec5.jpg)

![](images/02367d14f280705323add92f6647001b8d7df05a36e8fe8ff359b3266435fbe1.jpg)  
Figure 10: Critic loss across scenarios during training in NAHT.

![](images/8362c89e81f11d6f8ba99debe5ffd63bf62724d3590daa289fb45bfd4466e881.jpg)

![](images/da281f40df53f6f8b132f6f44d597996c3afd67b6e4b636e9bd7ef4fa7082534.jpg)

![](images/085a47a7db769cfcdde4f9ee32e3f1ae0e2dd3141ad5839dceb63b45393e27bb.jpg)

![](images/c94d590eaa3492aab683b515f17b31b7c52b1769dff1beaeab8e8692f50d26c5.jpg)

![](images/39a312bbe462a3166eb50dd4f8cf95c55796e2681511354d6e6e642a9918ec9f.jpg)

![](images/6b7194608477441eeb6e1330054bec11b3b85ea7a74a515480f10483e3dc4693.jpg)

(a) Evaluation on unseen conventions (identical algorithms with different random seeds). The star denotes the in-convention baseline, and the histograms show performance on unseen conventions.  
![](images/6378dabea623621b99731d3bd039df0aa06e3563feec673c9d5c066df01c5efb.jpg)

![](images/4bba8610dddea1e0950d917848cd365d1f84402957db08511e757b41eff1cc76.jpg)

![](images/b8821518ff4204f90da1c337812e39909ad4cced2273f9ca80c7f5909ca5c026.jpg)

![](images/434047d1a4a31c987cb57da5b15a11ca9c0194c0c35fe2780683f621dacf8c3b.jpg)  
(b) Evaluation on unseen agent types. Train set: MAPPO, QMIX, IQL; test set: IPPO, VDN.

![](images/0f1a7522f403956a10cb7d2da1fa3f79b8a08759a9adc2dbdb85243a1dbd52d6.jpg)

Figure 11: Out-of-distribution (OOD) evaluation after training, assessing unseen conventions and unseen agent types during training, between learned agents (IPPO-NAHT, SM-IPPO and BM-IPPO) and uncontrolled agent types (IPPO, VDN, IQL, QMIX and MAPPO). The performance is evaluated by averaging all pairings of N controlled and M — N uncontrolled agents and their corresponding random seeds, called M–N score [Wang et al., 2025].

frequent and typical case when an unseen agent appears in the environment, which may perturb the training stability. This justifies the effectiveness of the shaping rewards $\phi _ { i } ( R _ { t } )$ we propose to fulfil

![](images/35859e516261e0f01db11a4582fad6cbacb7162c6f540bfe91e02b4ba17a7677.jpg)

![](images/77ba457f884b3bbd5a8f8a51deff1ea0a31c364fa80fe3102b2502f2de78b672.jpg)

![](images/04dd5300098c885fccf4baa639a8aa03031c6cc0e94a5662cca2c1cf6a5dfd7c.jpg)  
(a) Evaluation on unseen agent types between POAM and IPPO (IPPO-NAHT).

![](images/8036ab48d0544b374e09244d3d47f09f3eb45a9ab8050709f5f1fdb23926a9ad.jpg)

![](images/fcfcdaeb44f724f8302e79f560b67069db5b02f33e1631b0bdc40ad077cb4f5e.jpg)

![](images/467638908198b46ea8d2a836066a60355a3cfeb013c501bf1e1e3a16dfee5d07.jpg)

![](images/217d2d7c60f44cb8ff2f70733b0669fcde377b2af02e9c1c225e4a46a0b0a8aa.jpg)

![](images/3cabd0357578dd7f001976613ed52dd42a6144646959b6b7c003d9745440951c.jpg)

![](images/99d54fdabbdc0d25a15f60513dbb7e90f3e267d11efcdc0c2b6c4215de94bd7a.jpg)

![](images/e64401a51246f6ba7b513233d625f97f1c04d011f4f502346264873b530fb0c5.jpg)

(b) Evaluation on unseen agent types between SM-POAM and SM-IPPO.  
![](images/ed4726acb5bd808fdb1d90466c79ee2a86f48bc27a654306552596ca2a6fb174.jpg)

![](images/b1d3b049852bd4ee4e810f56cea432c81dba6d046ca6be2c5af1e137727f97a1.jpg)

![](images/d46de46b3432e9a39517661f871ed809b28e4773ff576dae22cfcc0c2595e5ac.jpg)  
(c) Evaluation on unseen agent types between BM-POAM and BM-IPPO. Since the implementation of Banzhaf Machine is equivalent to its counterpart baseline algorithms for 8v9 and 10v11 due to approximating the number of non-empty coalition basis games as the episode length, we ignore their evaluation here.

Figure 12: Out-of-distribution (OOD) evaluation after training, assessing unseen conventions during training, between learned agents (IPPO-NAHT, SM-IPPO, and BM-IPPO) and uncontrolled agent types (IPPO, VDN, IQL, QMIX and MAPPO). The performance is evaluated by averaging all pairings of N controlled and M — N uncontrolled agents and their corresponding random seeds, called M–N score [Wang et al., 2025].

![](images/35949d005918b58bb55a01f3eb171bf01c0ed572d60e7b1193c517d313897a13.jpg)

![](images/ab286e701b5d826ee6cf60836b48dd02e0218ce7ec05f0368eb8101b9b5ea841.jpg)

![](images/b0efb4b4aa783fea4f99bcb1f497f7d06b08808b5d46bf88e0e206d3109dd2f1.jpg)  
Figure 13: One run of MPE to show the variation of test return, shaped reward and efficiency loss.

the Efficiency axiom. More importantly, this highlights the importance and necessity of the Efficiency axiom to perceive and measure the changing of environments.

## I.3 Extra Results of OOD Evaluation

We also conducted experiments on evaluating the OOD performance of Shapley Machine and Banzhaf Machine on the base algorithm IPPO (IPPO-NAHT). As seen from Figure 11a, Banzhaf Machine is competitive to Shapley Machine on dealing with unseen conventions. Although it seems that the result is opposite to the conclusion drawn from the POAM related results, the underlying reason could be the general weaker performance of IPPO than POAM due to the lack of agent modelling, as shown in Figure 15. This could inhibit the full capability of Shapley Machine. As seen from Figure 11b, the general performance of Banzhaf Machine is better than Shapley Machine. This is consistent with the conclusion drawn from the POAM related results. In Figure 12, we show additional comparisons between algorithms: POAM vs. IPPO, SM-POAM vs. SM-IPPO and BM-POAM vs. BM-IPPO. We find no strong evidence that the axioms are sensitive to the choice of base algorithm.

![](images/fe6ea82e41e19869d0ef5665e3320751d014ba895f6ea21d19d429023c3eec2f.jpg)  
Figure 14: One run of 5v6 to show the variation of test return, shaped reward and efficiency loss

![](images/860160e5cc2aac7d1366844f3b16903e51dff4e2c8eba59a538964ae547f0a98.jpg)  
Figure 15: Average evaluation performance on unseen conventions across 4 SMAC tasks. The star denotes the in-convention baseline, and the histograms denotes the unseen conventions.

## I.4 Potential Instability for Large-Scale Scenarios

During our repeated trials, we observe that the performance of Shapley Machine is not stable in large-scale scenarios (e.g. 8v9 and 10v11), compared with in small-scale scenarios (e.g. MPE-PP, 3sv5z and 5v6). We now analyze the potential reason behind this phenomenon. As observed from Figures 16 and 17, it can be confirmed that the instability of shaped rewards is the key reason to cause the instability of learning procedure for large-scale scenarios. According to the functionality of shaping rewards discussed in Appendix I.2, we hypothesize that the instability could be caused by the unstable changing of value difference in the shaping reward: $\begin{array} { r } { \sum _ { j \neq i } ( V _ { j } \hat ( s _ { t } ) - \gamma V _ { j } ( s _ { t + 1 } ) ) } \end{array}$ Specifically, when the number of agents increases, the accumulating prediction error of individual value functions will be amplified.

To mitigate the above issue, we have added the value loss clip to stabilize the learning procedure of individual value functions for the large-scale scenarios such as 8v9 and 10v11. As seen from Figure 3, even with the value loss clip the instability of SM-IPPO is still out of control. This could be the cascading effect caused by the lack of agent modelling in contrast to SM-POAM, the instability of which is far beyond the capability of value loss clip. This reflects the value of agent modelling in POAM for tackling the large-scale multi-agent scenarios.

As seen from Figure 18, the learning procedures with the value loss clip for the SM-POAM apparently perform more stably than those without. This verifies our initial hypothesis. Because this strategy will slow down and even hinder the performance for the other three scenarios, we have not posted this as a common strategy. We believe this deserves further investigation in the future before any claims can be made about its general performance.

![](images/3c5475856f6351e7adcb871ee4b52ef2b602750133dbdc648a988e3d1ffab61c.jpg)

![](images/1a4fe68165f6ba2a0119a52674c61926efeaabf46bf378e0b47b11882554e780.jpg)  
Figure 16: A case of 8v9 with SM-POAM to show the instability of learning progress via the mean shaped rewards. The range bounded by two red dashed lines shows the fluctuation of the learning process.

![](images/35c81f21761caafacb748cb8f7f5214f70ca16ab726358345af152a4a54d118d.jpg)

![](images/b610c89f5ecd9468e23eaaa9dc38fddc756007d478043cf0acf2e08c12df5d9c.jpg)  
Figure 17: A case of 10v11 with SM-POAM to show the instability of learning progress via the mean shaped rewards. The range bounded by two red dashed lines shows the fluctuation of the learning procedure.

![](images/1b5c30d7c8270dbd4bdae45f617170e16f51d9c4dd93ee19bdfaf69847acb89f.jpg)

![](images/f18de9dc59ad756a299e17168c9df3106727dbd8df651fd2a2dbbce38f96ffd0.jpg)  
Figure 18: Comparison between the learning procedure of SM-POAM with and without the value loss clip in the 8v9 and 10v11 scenarios.

## I.5 OOD Evaluation Across Different Numbers of Controlled Agents

We report the results on unseen conventions and agent types for each specific number of controlled agents N in Figures 19-22, by fixing the N value in the M–N score. We would like to highlight that the NAHT problem is a far more difficult task to deal with than the CMARL problem. As a result, our work is only a starting point in this direction, and we believe that the potential stability issue is due to the lack of more exact axioms tailored to the concrete task to face. However, following the framework and principle we develop in this work, this can be improved by considering more axioms that we do not cover in this work, from either the game-theoretical perspective or the physical world.

![](images/f893be0b9479d795c8f7f89e9ac0a17758f80cadec9fa3230148d61d65afc3fa.jpg)  
Figure 19: Out-of-distribution (OOD) evaluation after training, assessing unseen conventions during training, between learned agents (IPPO-NAHT, SM-IPPO and BM-IPPO) and uncontrolled agent types (IPPO, VDN, IQL, QMIX and MAPPO).

Target Algorithm: ippo  
![](images/cbf08b3e92a500d3688418830886730eea87b4cad3431c2989141347280797ac.jpg)  
Figure 20: Out-of-distribution (OOD) evaluation after training, assessing unseen agent types during training, between learned agents (IPPO-NAHT, SM-IPPO and BM-IPPO) and unseen uncontrolled agent types (IPPO and VDN).

Target Algorithm: ippo  
![](images/68a73076c75db25253047731489f90f484b041548240290f63e52d4862738845.jpg)  
Figure 21: Out-of-distribution (OOD) evaluation after training, assessing unseen conventions during training, between learned agents (POAM, SM-POAM, and BM-POAM) and uncontrolled agent types (IPPO, VDN, IQL, QMIX and MAPPO).

![](images/1d317001cb9b31911ab4a776b45dcb77b9d40086d0331b5d8ba34493e7833b40.jpg)  
Figure 22: Out-of-distribution (OOD) evaluation after training, assessing unseen agent types during training, between learned agents (POAM, SM-POAM and BM-POAM) and unseen uncontrolled agent types (IPPO and VDN).

## J Broader Impact

This paper introduces an axiomatic framework for learning in open multi-agent systems, with potential applications in domains such as smart grids, transportation networks, and multi-robot systems. By providing a principled foundation for coordinating with partially controllable and potentially unknown agents, the proposed approach may improve the reliability, adaptability, and generalisation of multi-agent systems deployed in real-world environments.

## J.1 Positive Impacts

A key potential benefit lies in infrastructure systems, such as power distribution networks, where coordination among heterogeneous and partially controllable agents (e.g., distributed energy resources, consumer devices) is essential. More robust and generalisable learning algorithms could improve efficiency, reduce energy waste, and enhance grid stability. Similarly, in autonomous transportation and logistics, improved coordination with unknown or non-cooperative agents may lead to safer and more efficient traffic management and resource allocation.

The proposed axiomatic perspective may also contribute to trustworthy AI, as it provides interpretable structural constraints on learning algorithms. By grounding value function design in well-established principles from cooperative game theory, the framework may improve transparency and facilitate reasoning about agent behaviour in safety-critical settings.

## J.2 Potential Risks and Negative Impacts

Despite these benefits, the deployment of open multi-agent systems introduces several risks. First, improved coordination capabilities could be applied in strategic or adversarial settings, such as automated market manipulation or competitive multi-agent environments, where agents learn to exploit or collude with others. The ability to generalise to unseen agent behaviours may amplify such risks.

Second, systems trained under the proposed framework may exhibit unintended coordination behaviours when interacting with human users or legacy systems, especially if the underlying assumptions about cooperation do not hold. In safety-critical domains, such behaviours could lead to undesirable or unpredictable outcomes.

Third, while the framework introduces structural constraints, it does not inherently guarantee fairness or alignment with human values. The allocation of value among agents, even when principled from a game-theoretic perspective, may not correspond to socially desirable outcomes in all contexts.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All key contributions covered in the abstract and introduction can be well matched to the theoretical results described in Sections 3 and 4, as well as experimental results as shown in Section 5. The scope of this paper is restricted to n-agent ad hoc teamwork, an open multi-agent system problem.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We have discussed the key limitation of this work in the Section 6. The key limitation lies in a parameter of our algorithm that could not well adapt to large-scale agent teams, but it can be improved by relaxing the assumption of our theoretical framework.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations" section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All theoretical claims are numbered and we have clarified clearly the corresponding sections in Appendix that include their proofs. In short words, all proofs and justifications for assumptions are given in Appendices C to G.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: All implementation details, evaluation metrics and hyperparameters for reproducing results are disclosed in Appendices H.1, H.3 and H.5.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The opensource code is uploaded as supplemental materials, along with instructions to reproduce key results.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https: //neurips . cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: All training and testing details are shown in Appendices H.3 and H.5.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All experimental results are demonstrated following mean with 95% CIs over 5 random seeds

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: All computer resources for experiments are disclosed in Appendix H.6.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: This paper does not violate any term in the code of ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We have clearly discussed both positive and potentially negative impacts in Appendix J.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: Our work is mainly about theoretical aspects, with no relation to such issues. Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All baseline algorithm (codes) we used for experiments have been credited by citing the corresponding work.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode. com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The code for training has been with a clear documentation.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: Our work has no relation to this term.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: This paper is with no relation to this term.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core part of research does not involve LLMs.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.