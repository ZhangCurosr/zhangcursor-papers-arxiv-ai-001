# A Definition of Catastrophic Performance

Henrik Marklund, Alex Infanger, Benjamin Van Roy

## Abstract

A central challenge in AI is specifying objectives that align with human preferences. While existing theoretical results elucidate how misspecification impacts performance, they typically do not distinguish between merely suboptimal and catastrophic outcomes.

To enable rigorous analysis of catastrophic risk from AI misalignment, we introduce a formal definition of catastrophic performance. Our definition is based on two reference values. The first, baseline value, captures the value attained if no agent is deployed. The second, orthogonal value, captures the value of a trajectory that is unrelated to human preferences, in a sense we describe. Orthogonal value is typically far worse than baseline value.

We interpret these reference values through examples in which past human activity has altered the state of the environment to human benefit. Starting at the resulting state, an AI agent is deployed with the goal of accelerating further progress. Catastrophic performance represents this agent’s undoing of benefits aforded by human activity.

## 1 Introduction

It is widely believed that superintelligent agents — systems that outperform human experts across virtually all intellectual tasks — will eventually be developed. This raises the concern that humans may lose control over them, leading to unintended and potentially catastrophic outcomes.

Theoretical work that studies this possibility quite often characterizes performance relative to optimal performance (see Section 5). And reinforcement learning theory, more generally, tends to do so. But, suboptimal performance need not be detrimental, let alone catastrophic. Thus, theoretical results that characterize performance relative to optimal cannot distinguish between beneficial and detrimental outcomes. This suggests a need for a diferent reference value than optimal performance.

Some works also characterize performance relative to variations of worst-case value, representing what a capable but malicious agent would achieve. But the central concern in the alignment literature is not that the AI is malicious and pursues the opposite of what humans want. Rather the concern is that the AI pursues a goal that does not capture human intentions suficiently well and causes a catastrophe as a byproduct (see e.g., [Yudkowsky, 2008, Omohundro, 2018, Russell, 2019]). While catastrophic, such an outcome ought to still be better than an outcome produced by an equally capable, but malicious agent. This suggests a need for a diferent reference value than worst-case value.

To address these needs, we propose two alternative reference values in this paper. We use these reference values to provide a formal definition of catastrophic performance. Our intention is to enable rigorous analysis focused on catastrophic outcomes due to misaligned AI. It should facilitate analysis focused on catastrophic rather than merely suboptimal outcomes. And it should facilitate analysis focused on AI that is not malicious but still misaligned.

We will refer to the state of the environment when the agent becomes available as the contemporary state. The first reference value is baseline value: the value attained when the environment starts in the contemporary state, and no agent is deployed. Baseline value provides a threshold below which performance is considered detrimental. Any meaningful definition of catastrophic performance must therefore require performance to fall below baseline value.

However, baseline value alone is not suficient for defining catastrophic performance. Consider a robot that accidentally knocks over a vase. The attained value is then below baseline value, and hence detrimental, but it is hardly catastrophic. To capture catastrophe, we need a notion of degree.

We express degree relative to the diference between baseline value and a second baseline, called orthogonal value. Orthogonal value represents the performance of a trajectory that is unrelated to human preferences, in a sense we describe in Section 4. To intuitively understand this baseline, suppose that a human operates in the environment for some time before they have the opportunity to deploy an agent. During that time, the human acts to improve the state of the environment according to their preferences. Orthogonal value reflects value realized in the absence of such human activity. Although we explained orthogonal value by referencing benefits arising from human activity, the notion itself does not depend on where the benefits originate from.

The degree of catastrophe is determined by where performance lies on the spectrum between the baseline value and orthogonal value. This can be interpreted as the extent to which the agent undoes the benefits of human activity or other desirable conditions.

The paper is structured as follows. Section 2 provides preliminary formulations and notation. Section 3.1 discuss the limitations of two commonly used reference values, best-case and worst-case performance. Section 3.2 formally introduces baseline and orthogonal value and our definition of catastrophic performance. Section 3.4 discusses our definitions in the context of a concrete mathematical example. Section 4 explains why the drop from baseline to orthogonal value is intuitively catastrophic in the context of the computer game Minecraft. Section 5 reviews reference values that have been considered in previous literature.

## 2 Preliminaries

In our formulation, there is a designer, an agent, and an environment. The designer has preferences over trajectories of the environment, which are represented by a reward function and discount factor. The designer produces and deploys an agent in the environment. In this section, we elaborate on this formulation with formal mathematical notation.

To simplify exposition, in the main text we will leverage the formalism of fully observable Markov decision processes (MDPs). The definitions we provide are generalized in Appendix C to partially observable Markov decision processes (POMDPs).

## 2.1 Markov Decision Processes (MDPs)

An MDP is a tuple $( S , \mathcal { A } , P , s _ { 0 } , r , \gamma )$ comprised of state space $s ,$ action space A, transition kernel $P _ { \mathrm { : } }$ start state $s _ { 0 } \in { \mathcal { S } }$ , reward function $r : \mathcal { S }  \mathbb { R }$ , and discount factor $\gamma \in [ 0 , 1 )$ . The transition kernel specifies, for each $a \in A .$ , a transition matrix $P _ { a }$ . Each element $P _ { a s s ^ { \prime } }$ is the probability of transitioning to $s ^ { \prime }$ from s upon executing action a.

## The No-Agent Transition Matrix

For the purposes of this paper, we extend the transition kernel to include an additional transition matrix $P _ { \varnothing }$ . The subscript ∅ indicates a no-agent, and we use this matrix to model dynamics when the agent is not active. Absent the agent’s influence, the state transitions according to $P _ { \varnothing }$ . On the other hand, with an active agent that selects an action $a \in { \mathcal { A } }$ , the state transitions according to $P _ { a }$

## 2.2 The Prior

The true MDP is a random tuple $( S , \mathcal { A } , P ^ { * } , S _ { 0 } , r ^ { * } , \gamma )$ . Here, $( \cal { S } , \cal { A } , \gamma )$ is deterministic, but $( P ^ { * } , S _ { 0 } , r ^ { * } )$ is random<sup>1</sup>. As mentioned earlier, the kernel $P ^ { * }$ includes transition matrices $( P _ { a } ^ { * } : a \in { \mathcal { A } } )$ and $P _ { \varnothing } ^ { * } .$ We refer to $P ^ { * } , S _ { 0 }$ , and $r ^ { * }$ as the true transition kernel, the contemporary state, and the true reward function, respectively.

We treat $( P ^ { * } , S _ { 0 } , r ^ { * } )$ as random in order to model uncertainty. This uncertainty is characterized by the probability distribution of $( P ^ { * } , S _ { 0 } , r ^ { * } )$ , which we refer to as the prior. But whose uncertainty does this express?

It is common to interpret the prior as expressing the agent’s initial uncertainty (see e.g., Hadfield-Menell et al. [2016]). We choose to focus on a diferent interpretation of the prior as the designer’s uncertainty (see discussion of why in Appendix B). The designer’s uncertainty stems from two sources.

To explain these two sources of uncertain ${ \mathrm { , y , } }$ let us first discuss them in the context of the transition kernel $P ^ { * }$ . First, the designer may not have full knowledge about $P ^ { * }$ . Following Garrabrant et al. [2016], we refer to this source as empirical uncertainty. Second, even in the absence of empirical uncertainty, the designer may be uncertain due to insuficient time and efort applied to specifying $P ^ { * }$ . We now explain this second form of uncertainty in more depth.

## Specification Uncertainty

When we say that the designer has no empirical uncertainty about $P ^ { * }$ , we mean that in the following sense. Given suficient time, the designer could, in principle, specify the transition kernel in a programming language such as Python. Writing such a specification is nontrivial. As a result, before the designer begins, they are efectively uncertain about which specification implements $P ^ { * }$ . This uncertainty induces a distribution over $P ^ { * }$ . We refer to this as specification uncertain $\therefore t y .$ Randomness in $P ^ { * }$ can be due to both empirical uncertainty and specification uncertainty. The same is true for $S _ { 0 }$

In contrast, randomness in $r ^ { * }$ is solely due to specification uncertainty. Given suficient time, we assume that the human can, in principle, specify their own preferences. But when given nearly no time to specify those preferences, they are uncertain about which specification represents their preferences. This uncertainty induces a distribution over $r ^ { * }$

## 2.3 Trajectory, Policy, and Return

A state trajectory is an infinite sequence $h = ( s _ { 0 } , s _ { 1 } , \ldots )$ of states. We will refer to this as a trajectory. For a given pair $( r , \gamma )$ , the return of a trajectory h is $\textstyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t + 1 } )$ . We write $\begin{array} { r } { r _ { \gamma } ( h ) = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t + 1 } ) } \end{array}$ to express the trajectory’s return.

The return function $r _ { \gamma }$ represents preferences between infinite trajectories in the following sense. For trajectories h and $h ^ { \prime } , \dot { h }$ is preferred over $h ^ { \prime }$ if $r _ { \gamma } ( h ) > r _ { \gamma } ( h ^ { \prime } )$ . In fact, $r _ { \gamma }$ represents not only preferences between trajectories, but preferences between trajectory distributions, where one distribution is preferred if the expected return is higher. The true return function $r _ { \gamma } ^ { \ast }$ , which represents the designer’s preferences, is random because it depends on $r ^ { * }$

A finite state-action trajectory is a finite sequence $h _ { t } = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , \ldots , s _ { t } )$ of states and actions. A policy is a function $\pi$ that maps each finite state-action trajectory $h _ { t }$ to an action distribution $\pi ( \cdot | h _ { t } ) \in \Delta _ { \ A }$ Let Π be the set of all policies. Let $H ^ { \pi } = ( S _ { 0 } , S _ { 1 } , \ldots )$ be the random state trajectory generated under policy $\pi .$

The expected return under a policy π conditioned on $( P ^ { * } , S _ { 0 } , r ^ { * } )$ is

$$
\mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \mid P ^ { * } , S _ { 0 } , r ^ { * } \right] .
$$

This conditional expectation represents the expected return as a function of the transition dynamics, contemporary state, and reward function. It is a random variable because of its dependence on $( P ^ { * } , S _ { 0 } , r ^ { * } )$ which is a random variable. Averaging over realizations of $( P ^ { * } , S _ { 0 } , r ^ { * } )$ leads to the expected return

$$
V ^ { \pi } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \right] = \mathbb { E } \left[ \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \ | \ P ^ { * } , S _ { 0 } , r ^ { * } \right] \right] ,
$$

which is deterministic.

## 2.4 Agent Design and Performance

An agent implements a policy $\pi _ { \mathrm { a g e n t } }$ . We treat this policy as a random variable to model its possible dependence on $P ^ { * } , S _ { 0 }$ , and $r ^ { * }$ . In particular, when designing the agent, the designer typically incorporates some knowledge about their preferences, as well as the state and dynamics of the environment.

We define performance to be the expected return

$$
V ^ { \mathrm { a g e n t } } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi _ { \mathrm { a g e n t } } } ) \right] = \mathbb { E } \left[ \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi _ { \mathrm { a g e n t } } } ) \mid P ^ { * } , S _ { 0 } , r ^ { * } , \pi _ { \mathrm { a g e n t } } \right] \right] ,
$$

which averages over realizations of $\pi _ { \mathrm { a g e n t } }$ as well as $( P ^ { * } , S _ { 0 } , r ^ { * } )$ . Note that this measure of performance pertains to the agent design process rather than the realized agent policy $\pi _ { \mathrm { a g e n t } }$ . In particular, it integrates over specification uncertainty and represents an expectation determined prior to the human’s deliberation in designing the agent.

## 2.5 Complexity of States, Functions and Distributions

Our definition of catastrophic performance will involve a notion of complexity. Here we introduce a notion of complexity of state and complexity of function and trajectory distribution.

## Programming Language

We will formalize the notion of a programming language with a partial function. If X and Y are sets, a partial function $f : X \to Y$ is a function from X to Y except f may be undefined on some elements in X. Thus, a partial function is a generalization of a function.

Let Σ be a finite set referred to as program symbols equipped with an ordering. Let X be a countable set referred to as program inputs and outputs. Let $U : \Sigma ^ { * } \times \mathcal { X } \stackrel { } {  } \mathcal { X }$ be a partial function, referred to as an interpreter. The interpreter takes two arguments, the first referred to as the program, and the second as the input to the program. For all $p \in \Sigma ^ { * }$ and X, we interpret $U ( p , x )$ as the interpreter executing program p with input x.

As mentioned, a partial function is not necessarily defined on all elements in the domain. Elements in $\Sigma ^ { * } \times \mathcal { X }$ for which U is not defined, you can interpret as either invalid program-input pairs or programinput pairs for which the interpreter does not halt.

We take the set X to be the union of the following objects:

1. finite state trajectories $S ^ { * }$ 2

2. rational numbers $\mathbb { Q }$

3. the empty input denoted by ϵ

Importantly, X is countable, and we assume it comes equipped with an enumeration.

This means that a program p can take as input a tuple of states, or a rational number, or another program as input.

Finally, we will assume that U is a so-called computable and universal function. Roughly speaking, that U is computable means that it can be implemented on a computer, and that it is universal means that it can simulate any computable function. Further discussion and a formal definition of the computability and universality of U can be found in Appendix A. Going forward, it may be helpful think of U as a simplified variant of the $\mathrm { P y }$ thon interpreter installed on your computer. The specific $\mathrm { P y }$ thon version does not matter much for discussion. However, to make sure we roughly have the same interpreter in mind, let us assume that you are not allowed import any packages. Assume the program can only take inputs and generate outputs in X . You can think of the input to the program as stored in a file and the output printed in another file. We assume an idealized version of Python where interpreter cannot run out of memory.

## Complexity of States

The complexity of a state s is the length of a shortest program p such that $U ( p , \epsilon ) = s$ . We denote the complexity of a state s by $C ( s )$

## Complexity of a Function

A partial function $f : \mathcal { X } \to \mathcal { X }$ is computable if there exists a program $p \in \Sigma ^ { * }$ such that

$$
U ( p , \cdot ) = f ( \cdot ) .
$$

The complexity of a computable function f is the length of a shortest program p such that $U ( p , \cdot ) = f ( \cdot )$

## Complexity of Trajectory Distributions

A distribution d over infinite trajectories is computable if there exists a program $p \in \Sigma ^ { * }$ such that for all finite state trajectories $h \in S ^ { * }$ that have support under d and states $s \in S$ ，

$$
U ( p , ( h , s ) ) = \mathbb { P } ( S _ { t + 1 } = s | H _ { t } = h ) ,\tag{1}
$$

where t is the length of h and $\mathbb { P } ( S _ { t + 1 } = \cdot , | H _ { t } = h )$ is the is distributed according to d. The complexity of trajectory distribution d is the length of a shortest program such that the condition above holds. We denote the complexity of a distribution d by $C ( d )$

## 3 A Definition of Catastrophic Performance

In this section, we define catastrophic performance in terms of two new reference values: baseline value and orthogonal value. To motivate these reference values, we will first discuss two reference values commonly used in the literature: best-case value and worst-case value.

## 3.1 Common Reference Values: Best and Worst Case

The problem of choosing a policy $\pi ^ { * }$ to maximize expected return given knowledge of $( P ^ { * } , S _ { 0 } , r ^ { * } )$ can be written as

$$
\operatorname* { m a x } _ { \pi \in \Pi } \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \mid P ^ { * } , S _ { 0 } , r ^ { * } \right] .
$$

Because $( P ^ { * } , S _ { 0 } , r ^ { * } )$ is a random variable, so is max<sub>π∈Π</sub> $\mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \mid P ^ { * } , S _ { 0 } , r ^ { * } \right]$ . We define best-case value to be the expectation of this maximum.

Definition 1. Best-case value is

$$
V ^ { + } = \mathbb { E } \left[ \operatorname* { m a x } _ { \pi \in \Pi } \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \mid P ^ { * } , S _ { 0 } , r ^ { * } \right] \right] ,
$$

where $H ^ { \pi }$ is the state trajectory generated by $\pi$

The outer expectation averages over the randomness in $( P ^ { * } , S _ { 0 } , r ^ { * } )$

Worst-case value $V ^ { - }$ is defined analogously with $V ^ { + }$ , minimizing instead of maximizing.

Definition 2. Worst-case value is

$$
V ^ { - } = \mathbb { E } \left[ \operatorname* { m i n } _ { \pi \in \Pi } \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \pi } ) \mid P ^ { * } , S _ { 0 } , r ^ { * } \right] \right] ,
$$

where $H ^ { \pi }$ is the state trajectory generated by $\pi .$

Theoretical papers often characterize performance relative to $V ^ { + }$ , or to similar notions of best-case performance (see Section 5). However, performance below best-case can still be beneficial. As such, theoretical results that characterize performance relative to best-case value cannot distinguish between beneficial and detrimental outcomes. This suggests a need for a diferent reference value. As an alternative, we will in Section 3.2 propose a reference value that represents the threshold between beneficial and detrimental performance.

Some previous papers also use $V ^ { - }$ , or similar notions of worst-case value, when characterizing performance (see $\mathrm { e . g . }$ , Fluri et al. [2025], Peng et al. [2018a]). This reference value can be interpreted as the value achieved by a malicious and knowledgeable agent. But the central concern in the alignment literature, and of this paper, is not that the AI system is malicious and seeks to minimize performance according to $\boldsymbol { r } _ { \gamma } ^ { * }$ (see e.g., [Yudkowsky, 2008, Omohundro, 2018, Russell, 2019]). Rather, the concern is that the agent optimizes an objective that is misspecified or unrelated to human preferences and, as a byproduct, causes a catastrophe. A common analogy is that of humans cutting down a rainforest. Humans are not intending to harm animals in the ecosystem but do so while pursuing other goals. While catastrophic for the animals, it is arguably much better than if humans were intentionally torturing the animals. This diference suggests a need for a diferent reference value than worst-case value. In Section 3.2, we will propose an alternative that better captures the nature and magnitude of misalignment risk that motivates this paper.

## 3.2 New Reference Values: Baseline and Orthogonal Value

In this section, we introduce new reference values as alternatives to best-case value and worst-case value.   
We will define catastrophic performance in terms of these reference values.

Let $H ^ { \varnothing }$ be the state trajectory generated according to the transition matrix $P _ { \varnothing }$ . We refer to $H ^ { \varnothing }$ as the baseline trajectory.

Definition 3. The baseline value is

$$
V ^ { \varnothing } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \varnothing } ) \right] ,
$$

where $H ^ { \varnothing }$ is the baseline trajectory.

Baseline value serves as the threshold between beneficial and detrimental performance. Any meaningful definition of catastrophic performance must therefore require performance to fall below baseline value.

But baseline value alone is not suficient to define a meaningful notion of catastrophic performance. To see this, consider a robot that knocks over a vase. Its performance is below baseline value, and hence detrimental, but hardly catastrophic. To distinguish a range of bad outcomes, we introduce a notion of degree. For that, we will define a second reference value. A potential candidate is worst-case value but, as discussed in Section 3.1, this reference value is inadequate. As an alternative, we propose orthogonal value.

To understand what orthogonal value represents, let us first discuss a key fact about the baseline trajectory. This trajectory typically conveys information about $r ^ { * }$ . We discuss three important reasons. First, past human activity may have improved the state of the environment, making $S _ { 0 }$ informative about $r ^ { * }$ [Shah et al., 2019]. Second, continued human activity may induce a drift towards desirable states making $( S _ { 1 } , S _ { 2 } , \ldots )$ informative about $r ^ { * }$ . Third, the preferences represented by $r ^ { * }$ may be the result of evolution. For instance, as discussed in [Shah et al., 2019, Section 6], on Earth, there is a match between human desire for oxygen and the presence of oxygen. This is not because humans chose the oxygen level but because humans evolved on Earth. As a contrast to the baseline trajectory, we will consider a trajectory that conveys no information about $r ^ { * }$ . Orthogonal value, defined next, will be the value of such a trajectory.

Definition 4. The orthogonal value is

$$
V ^ { \perp } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \perp } ) \right] ,
$$

where $H ^ { \perp }$ is a random trajectory of states that is independent of $r ^ { * }$ and has the simplest distribution.

First, recall that the simplest distribution is the distribution with minimum description length. If there are multiple such distributions, we mean the first one in lexicographical order. As a technical matter, because the distribution of $H ^ { \perp }$ is uniquely defined and $H ^ { \perp }$ is independent of $r ^ { * }$ , the distribution of $r _ { \gamma } ^ { * } ( H ^ { \perp } )$ is uniquely defined, and therefore, so is $V ^ { \perp }$

Second, the fact that $H ^ { \perp }$ has the simplest distribution is not important for the interpretation of $H ^ { \perp }$ What is important is that $H ^ { \perp }$ has a very simple distribution. To gain intuition for what a simple distribution is, it is helpful to note that in Python, two simple distributions are the uniform distribution and the singleton distribution placing all its mass on an easy-to-describe state.

As is clear from the discussion in Section 4, the random trajectory $H ^ { \perp }$ and $r ^ { * }$ are unrelated in an intuitive sense. It is for this reason we refer to $H ^ { \perp }$ as the orthogonal trajectory. It is orthogonal to human preferences.

We will dedicate a substantial part of this paper to interpreting orthogonal value. For now, we ofer only a high-level interpretation. Suppose that the contemporary state $S _ { 0 }$ is the result of past human activity that improves the state with respect to $r ^ { * }$ and that $H ^ { \varnothing }$ represents continued human activity. Let us consider what an orthogonal trajectory $H ^ { \perp }$ might look like. Suppose the simplest possible distribution is the uniform distribution: at each timestep, the state is sampled uniformly from $s$ . In many environments, $S _ { 0 } ^ { \perp }$ likely resembles an arbitrary configuration of the environment rather than one tailored to the human’s preferences. The subsequent evolution of the environment $( S _ { 1 } ^ { \perp } , S _ { 2 } ^ { \perp } , \ldots )$ is likewise unrelated to $r ^ { * }$ . Future changes to the environment are unlikely to make things significantly better or worse relative to $S _ { 0 } ^ { \perp }$

The orthogonal value is the expected return of such a trajectory. For the above example, it seems natural to say that the orthogonal value is the value of a trajectory where we have removed the benefits of the contemporary state and its subsequent evolution.

In subsequent sections, we will elaborate on interpretations of orthogonal value. In Section 3.4, we do so through a didactic example. In Section 4, we explain how orthogonal value coincides with an intuitive notion of catastrophe in many complex environments. Before discussing orthogonal value further, we provide a definition of catastrophic performance.

## 3.3 Catastrophic Performance

Our notion of catastrophic performance is ex ante and measures proximity to orthogonal versus baseline value, with degree expressed by a parameter η.

Definition 5. (η−catastrophic) For all $\eta \in [ 0 , 1 ]$ , performance $V \in \mathbb { R }$ is η-catastrophic if

$$
V \leq \eta V ^ { \perp } + ( 1 - \eta ) V ^ { \varnothing } .
$$

This definition is only meaningful if the baseline value $V ^ { \emptyset }$ exceeds the orthogonal value $V ^ { \perp }$ . When this is the case, rearranging terms to write the inequality as $V \leq V ^ { \varnothing } - \eta ( V ^ { \varnothing } - { \bar { V } } ^ { \perp } )$ , we can interpret the diference $V ^ { \emptyset } - V ^ { \bot }$ as a catastrophic drop in performance. In Section 3.4, we discuss via a didactic example why $V ^ { \emptyset }$ typically exceeds $V ^ { \perp }$

Our definition implies that we consider a performance drop $V ^ { \emptyset } - V ^ { \bot }$ to be catastrophic. We will illustrate in Section 4 through the computer game Minecraft why in complex environments a designer ought to consider the drop ${ \overset { \smile } { V } } ^ { \emptyset } - V ^ { \bot }$ to be catastrophic.

## 3.4 Example: Gridworld

According to our definition, degrees of catastrophic performance are determined by baseline and orthogonal value. In order to ofer a concrete illustration of these abstract reference values, we characterize them for a didactic example. The example is a toy environment designed to facilitate exact calculation of orthogonal value, which turns out to be zero, and to see that baseline value is positive and thus exceeds orthogonal value.

## 3.4.1 MDP Formulation

We consider an MDP $( \boldsymbol { S } , \boldsymbol { \mathcal { A } } , \boldsymbol { P } ^ { * } , \boldsymbol { S } _ { 0 } , \boldsymbol { r } ^ { * } , \boldsymbol { \gamma } )$ that models a two-dimensional gridworld, as illustrated in Figure $1 ( \mathrm { a } )$ . The state space $S = \{ 1 , \dots , n \} \times \{ 1 , \dots , n \}$ is a grid. At each time, an action is selected from A = {stay, left, up, right, down}. The no-agent transition matrix $P _ { \varnothing } ^ { * }$ encodes transition probabilities of the environment when the agent is not deployed. The state either stays constant or moves left, up, right, down. The gridworld wraps around in both dimensions. For instance, if the current state is at the left edge, and action left is taken, then the next state location is on the right edge. These transition dynamics are encoded in the kernel $P ^ { * }$

Let $S ^ { + }$ and $S ^ { - }$ be distinguished states, which we will refer to as seek and avoid states, and let $\sigma > 0$ be a fixed scalar parameter. For each $s \in S$ , reward takes the form

$$
r ^ { * } ( s ) = e ^ { - d ^ { 2 } ( S ^ { + } , s ) / 2 \sigma ^ { 2 } } - e ^ { - d ^ { 2 } ( S ^ { - } , s ) / 2 \sigma ^ { 2 } } ,
$$

where $d : \mathcal { S } \times \mathcal { S } \to \mathbb { R } _ { + }$ denotes the modular L1 distance function, which accounts for wrap-around. For example, if s and $s ^ { \prime }$ are at the upper-right and lower-left corners then $d ( s , s ^ { \prime } ) = 2$ . If $d ( S ^ { + } , S ^ { - } )$ is large relative to $\sigma$ then the reward $r ^ { * } ( s )$ is highest near $S ^ { + }$ and lowest near $S ^ { - }$ . A particular instance of $r ^ { * }$ is illustrated in Figure 1(b).

## 3.4.2 The Prior

Our reference values depend on the prior distribution of $( P ^ { * } , S _ { 0 } , r ^ { * } )$ , which we now define.

## Reward Function

We take the seek and avoid states $S ^ { + }$ and $S ^ { - }$ to be random variables. The prior distribution of $r ^ { * }$ is induced by this randomness. The seek and avoid states, $S ^ { + }$ and $S ^ { - }$ are both uniformly distributed over $s$ and independent of each other.

Because $S ^ { - }$ and $S ^ { + }$ are identically distributed, for all $s \in \mathcal { S } , \mathbb { E } [ e ^ { - d ^ { 2 } ( S ^ { - } , s ) / 2 \sigma ^ { 2 } } ] = \mathbb { E } [ e ^ { - d ^ { 2 } ( S ^ { + } , s ) / 2 \sigma ^ { 2 } } ] $ and, therefore,

$$
\begin{array} { r } { \mathbb { E } \left[ r ^ { * } ( s ) \right] = \mathbb { E } \left[ e ^ { - d ^ { 2 } ( S ^ { + } , s ) / 2 \sigma ^ { 2 } } - e ^ { - d ^ { 2 } ( S ^ { - } , s ) / 2 \sigma ^ { 2 } } \right] = 0 . } \end{array}
$$

![](images/14161e3df1029f014c71d426cedd03c7ba5d0d9f43f96ace40ba2d3a87027324.jpg)  
(a)

![](images/75e48a83d4a888b4af3a1b046fe44f948803456af46f47489948941352896697.jpg)  
(b)  
Figure 1: 2D Gridworld Example. (a) Transition dynamics if no agent is deployed. At each timestep, the state moves left, right, up, down, or remains the same. When the state moves into a wall, it wraps around to the other side of the grid. (b) Reward landscape for a particular realization of $r ^ { * }$ . The contemporary state $S _ { 0 }$ is the environment state when the AI assistant is introduced into the environment if it is deployed. The figure was made with the choices $n = 4 8$ and $\sigma = 5$

## Contemporary State

It is natural for the prior over $S _ { 0 }$ to satisfy two informal properties:

1. the reward $r ^ { * } ( S _ { 0 } )$ is likely quite good due to past human activity,

2. it is very uncertain what $S _ { 0 }$ is due to specification uncertainty.

We now provide a prior that satisfies the two properties above. Our exact choice of distribution is quite arbitrary, but is made for the sake of concreteness. Let $p \in ( 0 , 1 )$ be a fixed parameter and sample $S _ { 0 }$ through the following procedure.

$$
\begin{array} { r l } & { M \sim \mathrm { B i n o m } ( n ; p ) } \\ & { S _ { 0 } \sim \mathrm { U n i f o r m } ( \{ s \in { \mathcal { S } } : d ( s , S ^ { + } ) \leq M \} ) } \end{array}
$$

The parameter $p$ quantifies how far away $S _ { 0 }$ is likely to be from $S ^ { + }$ . If p is close to zero, then $S _ { 0 }$ will likely be close to $S ^ { + }$ . Note that $S _ { 0 }$ depends on $r ^ { * }$ (via dependence on $S ^ { + } )$ . It is easy to see that

$$
\mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] > 0 .
$$

See proof in Appendix D. Intuitively, the inequality reflects the following. By the time the human is considering deploying the agent, the state of the environment is quite good due to past human activity in the environment.

## Dynamics

It is natural that the prior over $P _ { \varnothing } ^ { * }$ satisfies the following two informal properties:

1. the reward tends to improve over time due to human activity if no agent is deployed,

2. it is uncertain what $P _ { \varnothing } ^ { * }$ is due to specification uncertainty (and possibly empirical uncertainty).

We now provide a prior that satisfies the two properties above. Our exact choice of distribution is quite arbitrary, but was made for the sake of concreteness. Let $P _ { \varnothing } ^ { * }$ be defined as follows. Let $q \in [ 0 , 1 )$ be a parameter. For all states s, with probability $q ,$ the state remains the same. With probability $1 - q$ , the state moves one step closer to $S ^ { + }$ by moving to its neighbor closest to $S ^ { + }$ . If multiple neighbors are tied to be the closest to $S ^ { + }$ , then the next state is sampled uniformly amongst those neighbors. If there are no such neighbors, that is, $s = S ^ { + }$ , then the state remains the same. We note that there is some chance that state transitions make things worse since the state may move closer to $S ^ { - }$ sometimes.

Note that $P _ { \varnothing } ^ { * }$ depends on $r ^ { * }$ (via dependence on $S ^ { + } )$ . This is intuitive: if human activity is expected to increase reward, then that activity ought to be informed by reward. For all $t \geq 0$ , we have that

$$
\mathbb { E } [ r ^ { * } ( S _ { t + 1 } ^ { \varnothing } ) ] > \mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] ,
$$

where $S _ { t + 1 } ^ { \varnothing }$ is the state at timestep $t + 1$ of the baseline trajectory. See proof in Appendix D. This reflects that the human is acting in the environment trying to improve the state.

## 3.4.3 Orthogonal Value

By definition, the orthogonal trajectory $H ^ { \perp } = ( S _ { 0 } ^ { \perp } , S _ { 1 } ^ { \perp } , \ldots )$ is independent of $r ^ { * }$ . Recall that for all $s \in \mathcal { S } , \mathbb { E } [ r ^ { * } ( s ) ] = 0$ . Because $S _ { t } ^ { \perp }$ is independent of $r ^ { * }$ , it follows that $\mathbb { E } [ r ^ { * } ( S _ { t } ^ { \bot } ) ] = 0$ . Thus, the orthogonal value is

$$
V ^ { \perp } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \perp } ) \right] = \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ^ { * } ( S _ { t + 1 } ^ { \perp } ) \right] = 0 .
$$

Because $\mathbb { E } [ r ^ { * } ( s ) ]$ is constant across all $s \in S$ , any trajectory $\widetilde { H }$ that is independent of $r ^ { * }$ has the same value. Thus, in this example, the fact that $H ^ { \perp }$ has the simplest distribution did not play any role.

## 3.4.4 Baseline Value

Recall that, along the baseline trajectory, reward tends to improve over the initial state: $\mathbb { E } [ r ^ { * } ( S _ { t + 1 } ^ { \varnothing } ) ] >$ $\mathbb { E } [ r ^ { * } ( S _ { 0 } ) ]$ for all $t \geq 0$ , and therefore, the baseline value satisfies

$$
V ^ { \emptyset } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \emptyset } ) \right] = \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ^ { * } ( S _ { t + 1 } ^ { \emptyset } ) \right] > \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ^ { * } ( S _ { 0 } ) \right] = \frac { \mathbb { E } \left[ r ^ { * } ( S _ { 0 } ) \right] } { 1 - \gamma } > 0 = V ^ { \bot } .
$$

Thus, baseline value exceeds orthogonal value. There are two inequalities among the above relations, each indicating a source of diference between $V ^ { \emptyset }$ and $V ^ { \perp }$ . One captures the extent to which the contemporary state is expected to be favorable, and the other captures the degree to which future human activity is expected to improve reward.

To make things even more concrete, let us assume the following parameter values: $n = 4 8 , \sigma = 5$ $\gamma = 0 . 9 , q = p = 0 . 2$ . Then, we have that

$$
V _ { 0 } = \underbrace { \frac { \mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] } { 1 - \gamma } } _ { \approx 4 . 1 } + \underbrace { \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } ( r ^ { * } ( S _ { t + 1 } ) - r ^ { * } ( S _ { 0 } ) ) \right] } _ { \approx 3 . 7 } \approx 7 . 8 > 0 = V ^ { \perp } .
$$

We expect $V ^ { \emptyset }$ to exceed $V ^ { \perp }$ not only in our gridworld but also across realistic examples. But it is worth recognizing that there exist contrived examples where the gap can be negative. For instance, suppose that $S _ { 0 }$ is sampled to be close to $S ^ { - }$ rather than $S ^ { + }$ and that human activity (for some contrived reason) tends to decrease rather than increase reward. Then, the baseline value $V ^ { \tilde { \varnothing } }$ would be negative.

## 4 How Bad is the Drop?

When we say that a trajectory is catastrophic, we mean it is catastrophic from the designer’s perspective. Definition 5 interprets a performance drop ${ V ^ { 0 } } - { V ^ { \perp } }$ from baseline to orthogonal value as catastrophic. In our gridworld example, it is not clear, in any intuitive sense, why a designer ought to consider this drop to be catastrophic. In fact, it is not clear that a human designer would recognize any trajectory through the grid as a catastrophe.

In the real world, on the other hand, there are trajectories that almost anyone would agree are catastrophic. For example, losing the fruits of human activity accumulated over history would be catastrophic. To make our discussion more concrete, we focus on the computer game Minecraft, for which it is simpler to precisely define state. Within this context, we will explain why the drop $V ^ { \emptyset } - V ^ { \bot }$ is catastrophic in an intuitive sense. This indicates that our abstract definition of catastrophic performance is consistent with an intuitive notion of catastrophic performance.

![](images/6edd21684224c9fdd874c2813269aca5e04728c627513e44d069f0e57e319683.jpg)  
Figure 2: Illustration of Minecraft example with human preoptimization. Left: the primordial state is the state of the environment prior to any human optimization. Middle left: During an extended period, the human acts in the environment shaping it according to their preferences (that is, r<sup>∗</sup>). Middle right: the contemporary state S is the state of the environment when the AI agent is introduced. Far right: AI agent acts alongside human in the environment. Images are not actual images from the game, but generated with ChatGPT.

This section is structured as follows. We first introduce our Minecraft example. We will then present an informal argument for why orthogonal value is intuitively catastrophic in Minecraft. Our argument is made in the context of Python. We conclude the section with a brief discussion of how dependent our notion of catastrophe is upon the choice of programming language.

## 4.1 Example: Minecraft

Our example, based on Minecraft, is chosen to share important features with the real world. In Minecraft, a player works possibly over a very long time to build up desirable structures. It is natural to interpret the destruction of those structures as catastrophic. We will show that this is consistent with our formal definition of catastrophic performance.

## States and Actions

In Minecraft, one or more players are placed in an open-ended three-dimensional world where they can gather resources, construct buildings, and even develop entire cities. We model the game world as a finite three-dimensional grid. Each cell can take on hundreds of values corresponding to its contents (e.g., rock, wood, fence, or empty space)<sup>2</sup>. At any point in time, each player occupies a cell in the grid. The state space therefore consists of all possible configurations of the grid together with the orientation and locations of the players. To simplify our discussion, we assume that each player fully observes the state.

The action space of each player includes moving (forward, backward, left, right), changing orientation (turn left, turn right, look up, look down), doing nothing, breaking items, and a number of more advanced alternatives.

## Reward Function

We imagine a designer who wishes to build a city. The designer has a particular city in mind, consisting of diferent types of buildings, parks, and pathways. There are also cities that the designer detests the designer prefers no city to those cities. Let $S ^ { + }$ and $S ^ { - }$ denote states corresponding to cities the designer most desires and most detests: $r ^ { * } ( S ^ { + } )$ is maximum reward, and $r ^ { * } ( S ^ { - } )$ is minimum reward.

While it may be natural to imagine that the designer has an ideal city in mind, one might wonder why there ought to be a bˆete noire, far worse than having no city at all. As an example, the designer might have a phobia of spiders and regard the worst possible city as one in which every structure resembles a giant spider.

```python
def p(h , s ):
return 1 / 440**10**9
```  
(a) The uniform distribution. This example assumes that $| \boldsymbol { \mathcal { S } } | = 4 4 0 ^ { 1 0 ^ { 9 } }$

```python
def p(h, s):
S = (0 ,) * 10 ** 9 # block 0 repeated
return int (s == S)
```  
(b) The simple singleton distribution. It places all probability mass on a state with the block 0 (air) repeated throughout the whole grid. It assumes that the grid size is $1 0 ^ { 9 }$  
Figure 3: Two prototypically simple distributions over Minecraft trajectories. Each is implemented as a Python program that takes a partial state trajectory $h \in S ^ { * }$ and state $s \in { \mathcal { S } }$ and returns a number in [0, 1]. Both programs are short, yet they sit at opposite ends of the difuse–concentrated spectrum.

## Human Activity Before and After AI Deployment

There will be two players, a human and an AI. Before the AI is introduced into the Minecraft world, a human player interacts with the world for an extended period of time, gradually improving the state of the environment according to $r ^ { * }$

After this period of human activity, the designer has the option to deploy an agent. The designer’s intention with deploying an agent is for it to assist in building the ideal city. Let $S _ { 0 }$ denote the state of the environment at the time of its potential introduction. See Figure 2 for a visual summary of how the contemporary state comes about. If the AI is not deployed, then the state of the environment will tend to continue to improve according to $r ^ { * }$ , as a consequence of continued human activity.

As discussed in Section 2.1, our MDP model includes a no-agent transition matrix $P _ { \varnothing }$ . In Minecraft, that corresponds to no AI agent being deployed.

## 4.2 Why Simple Distributions are Intuitively Catastrophic in Minecraft

In this subsection, we present an informal argument for why if a trajectory is sampled from a simple distribution, and independently of $r ^ { * }$ , then it should be catastrophic in an intuitive sense in Minecraft. To make the discussion more concrete, we first discuss a prototypically simple distribution — the uniform distribution — and why that is intuitively catastrophic. We then explain why all distributions simpler than the uniform distribution ought to be catastrophic in Minecraft.

## 4.2.1 The Uniform Distribution is Simple and Intuitively Catastrophic in Minecraft

Suppose that at each timestep $S _ { t } ^ { \perp }$ is sampled uniformly from S in Minecraft. Assuming the state space size |S| has a short description, the uniform distribution is very simple to describe in Python. In Figure $\mathrm { 3 ( a ) }$ , we give an example program when the state space has size $4 4 0 ^ { 1 0 ^ { 9 } }$ . This is the case when the grid size is 1000 × 1000 × 1000 and there are 440 block types. Since most states in the game of Minecraft do not contain any desirable structures such as sophisticated buildings or beautiful parks, a uniform sample from Minecraft states, independent of $r ^ { * }$ , is very unlikely to contain such structures. This is in stark contrast with the baseline trajectory which likely contains many structures the human desires. Indeed, past and future human activity ought to give rise to a highly desirable and sophisticated city. Therefore, relative to $V ^ { \emptyset }$ , the value $V ^ { \perp }$ is intuitively catastrophic. While the value of $V ^ { \perp }$ is intuitively catastrophic it is still far better than worst-case value: in Minecraft, $S _ { t } ^ { \perp }$ is very unlikely to contain detestable structures such as spider buildings.

To give the reader a richer understanding of what a simple distribution might look like, we include another example in Figure 3 (b). In contrast to the uniform distribution discussed above, this other example places all of its probability mass on a single simple state. This emphasizes that a simple distribution need not be difuse like the uniform distribution, but can be extremely concentrated.

## 4.2.2 Why All Simple Distributions Are Intuitively Catastrophic in Minecraft

In the preceding subsection, we explained why the uniform distribution is intuitively catastrophic in Minecraft. But perhaps there exist other simple distributions that are not catastrophic. In this section, we argue informally that this is not the case: all distributions as simple as or simpler than the uniform distribution are intuitively catastrophic in Minecraft. That will imply that orthogonal value is intuitively catastrophic.

Let $L _ { \mathrm { u n i f o r m } }$ be the length of the program in Figure $\mathrm { 3 ( a ) }$ , which implements the uniform distribution for the case when $\vert \boldsymbol { S } \vert = 4 4 0 ^ { 1 0 ^ { 9 } }$ . We assume that all valid programs start with the code snippet def $\mathtt { p ( h , s ) }$ : and end with the code snippet return followed by a value, and so we ignore these code snippets when computing the length of a program. In this case, we have

$$
L _ { \mathrm { u n i f o r m } } = 1 2 .
$$

Now the key question is: Is it possible to construct another distribution with 12 symbols or less that is not catastrophic? It seems very likely that the answer is no. If a distribution is to put high probability on non-catastrophic states, it seems very likely that the program actually needs to describe features that the human desires, and that this would require more than 12 symbols.

To understand this idea better, it is helpful to consider the special case of singleton distributions. Suppose one wanted to construct a distribution that places all its probability mass on a noncatastrophic state. Then, that state would need to be described in the Python program, and presumably describing such a state (that may involve beautiful parks, or a sophisticated building) would require more than 12 symbols.

While simple distributions are intuitively catastrophic, they ofer far more than worst-case value: it seems very unlikely a program with 12 symbols or less can implement a distribution placing significant probability mass on detestable states (those that include spider-buildings). For if a program did place much probability mass on such states, the program would presumably need to include some description of those spider-buildings that would likely require more than 12 symbols.

All in all, it seems highly likely that all distributions with 12 symbols or less are intuitively catastrophic in Minecraft, while far better than worst-case. It follows that the orthogonal value, which is the value of the simplest distribution is also intuitively catastrophic, while far better than worst-case value.

## 4.2.3 Beyond Python

What constitutes the simplest distribution depends on the programming language. While our discussion has focused on Python, one may wonder how far the ideas extend to other languages. We briefly discuss this now.

We believe that our argument extends naturally to all commonly used programming languages: if a distribution is suficiently simple then it will be intuitively catastrophic in Minecraft. Thus, we expect the choice of programming language not to impact our interpretation of orthogonal value.

More broadly, and as explained in Appendix E, there is a formal sense in which the simpler a distribution is, the less it can place mass on complex elements. This holds for all languages. But these formal results are directional and do not rule out languages in which a simple distribution concentrates on specific complex states. Furthermore, what constitutes a complex state is language-dependent. In principle, one can construct programming languages in which the best possible state in Minecraft is the simplest state. In that case, with a natural technical assumption, there is a simple singleton that places all its mass on the best possible state. But we consider such cases pathological.

This raises an interesting question for future research: is there a fundamental reason why simple distributions in commonly used languages ought to be catastrophic? Relatedly, is there a fundamental reason why very simple states (in commonly used languages) are unlikely to be desirable?

## 5 Related Work

In developing our definition of catastrophic performance, we considered various reference values used in the previous literature. In this section, we situate our work by discussing these reference values and why we did not use them. We hope this will make it easier for the reader to assess the strengths and weaknesses of our definition of catastrophic performance.

## 5.1 Best-Case Value

Best-case value is a reference value that is ubiquitous in the reinforcement learning literature. Indeed, most notions of regret are defined with respect to best-case value or variants thereof. Here, we just mention a few examples that touch on AI misalignment.

Everitt et al. [2017] characterize performance relative to best-case in their study of “corrupt reward MDPs”. Similarly, in their introduction of a distance function on reward functions, Gleave et al. [2021] derive a bound on regret relative to optimal. Fluri et al. [2025] use the normalized regret – namely, the diference from best-case value, divided by the diference between best-case and worst-case value – as the performance metric in their theory on the perils of optimizing learned reward functions. Brown et al. [2021] say a policy is ϵ-value aligned if its value function is uniformly within ϵ of optimal and study the problem of verifying ϵ-value alignment.

As discussed in Section 3.1, characterizing performance relative to best-case does not distinguish between performance that is beneficial and detrimental, let alone catastrophic.

## 5.2 Worst-Case Value

Sometimes, performance is also characterized using worst-case value, or a variant thereof. For instance, as mentioned above, Fluri et al. [2025] characterizes performance relative to the performance of a best-case and a worst-case policy.

Another variant of worst-case value is worst-case return possible based on the interface, even if not achievable by any policy. This is often used as the zero-point when reporting performance scores in reinforcement learning. See, for example, Peng et al. [2018a,b], Abdulhai et al. [2025].

Finally, we mention Cross et al. [2026] which define what they refer to as partial catastrophe. In particular, they say that a performance level V is ν-catastrophic if $V \leq \nu .$ Thus, ν-catastrophic means “performance is worse or equal to ν”. Thus, their notion is not defined relative to any reference values at all. However, they do define “catastrophe” as a performance level equal to worst-case performance.

As discussed in Section 3.1, worst-case value does not capture the nature and magnitude of the central concern in the alignment literature.

## 5.3 Inaction Value

Some works define a baseline state or trajectory corresponding to an agent executing a “no-op” action or an agent that performs “no optimization”. Such reference values are akin to our definition of baseline value.

El-Mhamdi and Hoang [2024] and Zhuang and Hadfield-Menell [2020] develop models of how overoptimization can lead to detrimental outcomes. In their plots, both characterize performance relative to the performance achieved when there is no optimization at all.

The side-efects literature (see e.g., Armstrong and Levinstein [2017], Krakovna et al. [2018], and Turner et al. [2020a,b]) develops methods that incentivize an agent to avoid causing unnecessary changes to the environment. These approaches typically define a baseline state or trajectory, often corresponding to a scenario in which the agent takes no action, and penalize deviations from that baseline. In the side-efects literature, the baseline serves as a reference point for measuring impact and constraining behavior, rather than a reference value against which to evaluate performance.

In a blog post, Kosoy [2021] proposes a closely related reference value. The setup considered is the so-called delegative RL setup in which at every timestep the agent can choose to delegate to the human. The baseline proposed is the expected performance of an agent that always delegates to the human.

## 5.4 Random Action Value

A relatively common reference value is the value of an agent that samples actions uniformly at random at each timestep. For instance, in Mnih et al. [2015], they normalize performance such that 0 is the performance of an agent that sampled actions uniformly at random, and 1 is the performance of a human.

In environments where random actions tend not to impact the environment much, this baseline represents performance similar to, but slightly worse than, baseline value. For instance, consider deploying an agent in Minecraft that randomizes uniformly across actions. Such an agent would almost certainly not impact the environment much at all. Expected performance would be slightly below baseline value, but much better than orthogonal value.

It is worth noting that there can, in principle, be environments in which random actions are likely to radically change the state of the environment and generate a very detrimental outcome. In that case, the random-action baseline may represent performance that is more akin to our definition of orthogonal value.

Because the magnitude of the random action value can vary across environments in this way, we think it is an inappropriate alternative for either baseline or orthogonal value.

## 5.5 Unrelated Goal Value

In writing this paper, before arriving at our definition of orthogonal value, we first worked with another definition. Loosely speaking, it was “the expected return of a maximally competent agent optimizing a reward function unrelated to human preferences”. We leave out how we formalized “maximally competent” and “unrelated”. We were motivated by the intuition that a maximally competent agent that optimizes an “arbitrary” reward function ought to be catastrophic in many contexts. We used such a definition in an early version of a diferent paper of ours [Marklund et al., 2026].

That definition is problematic because the validity of the definition relies on nonobvious facts about the environment. In particular, it relies on what an agent optimizing an unrelated reward function is likely to achieve. We think it is better to use a definition whose validity does not rely on such properties.

## 5.6 Other Reference values

Laidlaw et al. [2024] provide a definition of reward hacking using a baseline policy: a policy hacks ˆr relative to a chosen reference policy $\pi _ { \mathrm { r e f } }$ if it ˆr-dominates $\pi _ { \mathrm { r e f } }$ but is $r ^ { * }$ -dominated by it. Here, poor performance corresponds to performing worse than the reference policy.

In their framework, $\pi _ { \mathrm { r e f } }$ is a free parameter. Thus, performing worse than $\pi _ { \mathrm { r e f } }$ does not necessarily mean detrimental performance since $\pi _ { \mathrm { r e f } }$ may be a relatively good policy. Also, their definition of reward hacking is relative to the single baseline $\pi _ { \mathrm { r e f } }$ . Using a single baseline cannot provide a scale: their theory does not distinguish between a policy that performs just barely worse than the reference policy and one that performs significantly worse.

## 5.7 Other Relevant Work

Reward hacking: The reward hacking literature studies what happens when an agent optimizes a proxy reward ˆr that difers from the true reward $r ^ { * } ;$ see, for instance, Karwowski et al. [2023], Manheim and Garrabrant [2019]. At a high level, reward hacking happens when the agent performs well according to $\hat { r }$ and poorly according to $r ^ { * }$ . Some papers in the reward hacking literature have already been discussed above.

Skalse et al. [2022] define a reward function $r ^ { * }$ and its proxy ˆr to be hackable if there is a pair of policies for which the proxy strictly prefers a strictly $r ^ { * }$ -inferior alternative. But the fact that $r ^ { * }$ and $\hat { r }$ are hackable does not tell us whether an induced policy ˆπ produces beneficial or detrimental, let alone catastrophic performance.

Goal-misgeneralization: This body of work studies how deployed agents may pursue goals at odds with the goals encoded by the training objective. Di Langosco et al. [2022] provide a semi-formal definition of goal misgeneralization. They say a policy goal-misgeneralizes if its trajectories are better explained as those of a policy with “goal-directedness” instead of a policy without goal-directedness, and it receives low reward according to $r ^ { * }$ . They formally define goal-directedness but do not formally define what counts as “low reward”. Shah et al. [2022] also study goal misgeneralization but do not formally specify any reference value in their study (although in their Monster Gridworld and Cultural Transmission examples, they do show the performance of a random policy for comparison). Abdel Sadek et al. [2025] compare expected reward maximization with a regret minimization method for mitigating goal-misgeneralization. They use regret (relative to an optimal policy) as their performance metric.

## 6 Conclusion

Previous theoretical works often characterize agent performance relative to the best-case and, sometimes, worst-case value. This paper introduced two alternative reference values, baseline and orthogonal value, in order to define a meaningful notion of catastrophic performance. Unlike best-case value, baseline value provides a way to distinguish between detrimental performance and mere suboptimality. Compared to worst-case value, orthogonal value more accurately reflects the nature and magnitude of the central concern of the alignment community. We believe these new reference values will prove more useful in analyzing catastrophic risk from misaligned AI.

## Acknowledgments

This work was supported by grants from the Huxley Foundation and the UK AI Security Institute. We thank Jonathan Carr, Ryan Carey, Alex Cloud, Michael Cohen, John Duchi, Blake Elias, Benlin Gan, Stephane Hatgis-Kessell, Ramesh Johari, Cameron Jordan, Saurabh Kumar, Smitha Milli, Stuart Russell, Karim Abdel Sadek, Stewy Slocum, Eric Tang, Bryce Woodworth, Renyuan Xu, Wanqiao Xu, and Yifan Zhu for helpful comments and discussions. We also thank students of the MS&E 338/CS 338 course for valuable discussions. Finally, we used AI tools to check and improve the writing of the paper and find relevant previous work. It also assisted in the writing of proofs in the appendix, and provided key ideas for the proof of Proposition 3 in the appendix.

## References

Karim Abdel Sadek, Matthew Farrugia-Roberts, Usman Anwar, Hannah Erlebach, Christian Schroeder de Witt, David Krueger, and Michael Dennis. Mitigating goal misgeneralization via minimax regret. In Reinforcement Learning Conference (RLC), 2025. arXiv:2507.03068.

Marwa Abdulhai, Isadora White, Charlie Victor Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. LMRL gym: Benchmarks for multi-turn reinforcement learning with language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=hmGhP5DO2W.

Stuart Armstrong and Benjamin Levinstein. Low impact artificial intelligences. arXiv preprint arXiv:1705.10720, 2017.

George S. Boolos, John P. Burgess, and Richard C. Jefrey. Computability and Logic. Cambridge University Press, Cambridge, 5 edition, 2007. ISBN 978-0-521-87752-7. doi: 10.1017/ CBO9780511804076. URL https://www.cambridge.org/core/books/computability-and-logic/ 440B4178B7CBF1C241694233716AB271.

Daniel S. Brown, Jordan Schneider, Anca Dragan, and Scott Niekum. Value Alignment Verification. In Proceedings of the 38th International Conference on Machine Learning, pages 1105–1115. PMLR, July 2021. URL https://proceedings.mlr.press/v139/brown21a.html.

Winter Cross, L´eo Cymbalista, Alfred Harwood, and Jose Faustino. Fragility of value under imperfect alignment, 2026. URL https://arxiv.org/abs/2607.28881.

Lauro Langosco Di Langosco, Jack Koch, Lee D Sharkey, Jacob Pfau, and David Krueger. Goal misgeneralization in deep reinforcement learning. In International Conference on Machine Learning, pages 12004–12019. PMLR, 2022.

El-Mahdi El-Mhamdi and Lˆe-Nguyˆen Hoang. On Goodhart’s law, with an application to value alignment, 2024. URL https://arxiv.org/abs/2410.09638.

Tom Everitt, Victoria Krakovna, Laurent Orseau, and Shane Legg. Reinforcement Learning with a Corrupted Reward Channel. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, pages 4705–4713, Melbourne, Australia, August 2017. International Joint Conferences on Artificial Intelligence Organization. ISBN 978-0-9992411-0-3. doi: 10.24963/ijcai.2017/656. URL https://www.ijcai.org/proceedings/2017/656.

Lukas Fluri, Leon Lang, Alessandro Abate, Patrick Forr´e, David Krueger, and Joar Max Viktor Skalse. The perils of optimizing learned reward functions: Low training error does not guarantee low regret. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 17306–17377. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/fluri25a.html.

Scott Garrabrant, Tsvi Benson-Tilsen, Andrew Critch, Nate Soares, and Jessica Taylor. Logical induction. arXiv preprint arXiv:1609.03543, 2016.

Adam Gleave, Michael D Dennis, Shane Legg, Stuart Russell, and Jan Leike. Quantifying diferences in reward functions. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=LwEQnp6CYev.

Dylan Hadfield-Menell, Stuart J Russell, Pieter Abbeel, and Anca Dragan. Cooperative inverse reinforcement learning. Advances in neural information processing systems, 29, 2016.

Jacek Karwowski, Oliver Hayman, Xingjian Bai, Klaus Kiendlhofer, Charlie Grifin, and Joar Skalse. Goodhart’s law in reinforcement learning. arXiv preprint arXiv:2310.09144, 2023.

Vanessa Kosoy. The hippocratic principle. AI Alignment Forum, March 2021. URL https://www.alignmentforum.org/posts/dPmmuaz9szk26BkmD/vanessa-kosoy-s-shortform? commentId=jMitxvhFceaheD5zb. Last edited March 17, 2021.

Victoria Krakovna, Laurent Orseau, Ramana Kumar, Miljan Martic, and Shane Legg. Measuring and avoiding side efects using relative reachability. arXiv preprint arXiv:1806.01186, 2018.

Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Correlated proxies: A new definition and improved mitigation for reward hacking. arXiv preprint arXiv:2403.03185, 2024.

Ming Li and Paul Vit´anyi. An Introduction to Kolmogorov Complexity and Its Applications. Texts in Computer Science. Springer, New York, NY, 2008. ISBN 978-0-387-33998-6 978-0-387-49820-1. doi: 10.1007/978-0-387-49820-1. URL https://link.springer.com/10.1007/978-0-387-49820-1.

David Manheim and Scott Garrabrant. Categorizing variants of Goodhart’s law, 2019. URL https: //arxiv.org/abs/1803.04585.

Henrik Marklund, Alex Infanger, and Benjamin Van Roy. Consequentialist objectives and catastrophe. arXiv preprint arXiv:2603.15017, 2026.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

Stephen M Omohundro. The basic AI drives. In Artificial intelligence safety and security, pages 47–55. Chapman and Hall/CRC, 2018.

Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. ACM Transactions On Graphics (TOG), 37(4):1–14, 2018a.

Xue Bin Peng, Angjoo Kanazawa, Jitendra Malik, Pieter Abbeel, and Sergey Levine. Sfv: reinforcement learning of physical skills from videos. ACM Trans. Graph., 37(6), December 2018b. ISSN 0730-0301. doi: 10.1145/3272127.3275014. URL https://doi.org/10.1145/3272127.3275014.

Stuart Russell. Human compatible: AI and the problem of control. Penguin UK, 2019.

Rohin Shah, Dmitrii Krasheninnikov, Jordan Alexander, Pieter Abbeel, and Anca Dragan. Preferences implicit in the state of the world. arXiv preprint arXiv:1902.04198, 2019.

Rohin Shah, Vikrant Varma, Ramana Kumar, Mary Phuong, Victoria Krakovna, Jonathan Uesato, and Zac Kenton. Goal misgeneralization: Why correct specifications aren’t enough for correct goals. arXiv preprint arXiv:2210.01790, 2022.

A. Shen and N. Vereshchagin. Computable Functions, volume 19 of The Student Mathematical Library. American Mathematical Society, Providence, Rhode Island, December 2002. ISBN 978-1-4704-2133-5 978-0-8218-2732-1. doi: 10.1090/stml/019. URL https://www.ams.org/stml/019.

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

Alexander Matt Turner, Dylan Hadfield-Menell, and Prasad Tadepalli. Conservative agency via attainable utility preservation. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, pages 385–391, 2020a.

Alexander Matt Turner, Neale Ratzlaf, and Prasad Tadepalli. Avoiding side efects by considering future tasks. In Advances in Neural Information Processing Systems, 2020b.

Eliezer Yudkowsky. Artificial intelligence as a positive and negative factor in global risk. Global catastrophic risks, 1(303):184, 2008.

Simon Zhuang and Dylan Hadfield-Menell. Consequences of misaligned AI. Advances in Neural Information Processing Systems, 33:15763–15773, 2020.

## A Algorithmic Information Theory Background

The goal of this section is to formally define the computability and universality of the interpreter $U :$ $\Sigma ^ { * } \times \mathcal { X } \stackrel { } {  } \mathcal { X }$ described in Section 2.5. To this end, we will first fix a model of computation based on a multi-tape Turing machine.

## A.1 Multitape Turing Machine

We will take as our model of computation a Turing machine with four tapes:

1. read-only tape I,

2. read-only tape II,

3. memory tape,

4. append-only output tape.

Recall that Σ is a finite set whose elements we refer to as program symbols. All tapes have Σ as their alphabet.

We choose this model to match a standard $\mathrm { P y }$ thon or Java interpreter of the following sort. The $\mathrm { P y }$ thon interpreter takes an input program (read-only tape I), and may take an additional input string (read-only tape II). As the interpreter runs the program, it uses memory to facilitate computation (memory tape) and writes output to the user (append-only output tape).

We will refer to a Turing machine M as a single-input tape Turing machine if its operation is independent of the second input tape.

## A.2 Computable Functions and the Interpreter

Given our multitape Turing machine, we are almost in position to define computability and universality of the interpreter U from Section 2.5. Let us begin with computability. A function is typically called computable if some Turing machine computes it. The interpreter U, however, has domain $\Sigma ^ { * } \times \mathcal { X }$ and codomain X, whereas our Turing machine only reads and write elements in $\Sigma ^ { * }$

To bridge this gap, we will make use of a bijective decoder $d _ { \mathcal { X } } : \Sigma ^ { * } \to \mathcal { X }$ . Given this decoder, our path to defining the computability and universality of U is as follows. We will first define computability for partial functions $f : \Sigma ^ { * } \times \Sigma ^ { * } \to \Sigma ^ { * }$ . Then, we will say the interpreter $U$ is computable if the induced partial function $\tilde { U } : \Sigma ^ { * } \times \Sigma ^ { * } \to \Sigma ^ { * }$ with $\tilde { U } ( p , x ) = d _ { \mathcal { X } } ^ { - 1 } \big ( I ( p , d _ { \mathcal { X } } ( x ) ) \big )$ (with $\tilde { U } ( p , x )$ undefined whenever $U ( p , d _ { \mathcal { X } } ( x ) )$ is undefined), is computable.

We will similarly first define universality on computable partial functions from $\Sigma ^ { * } \times \Sigma ^ { * } \to \Sigma ^ { * }$ and then extend it to functions from $\Sigma ^ { * } \times \mathcal { X } \stackrel { } {  } \mathcal { X }$ using the decoder $d _ { \mathcal { X } }$

We now formally develop the theory outlined above.

Definition 6. The partial function $f : \Sigma ^ { * } \times \Sigma ^ { * } \to \Sigma ^ { * }$ is said to be implemented by the Turing machine M if for all $x , x ^ { \prime } \in \Sigma ^ { * }$

1. M halts on $( x , x ^ { \prime } )$ if and only if f is defined on $( x , x ^ { \prime } )$ ，

2. and if, M halts on $( x , x ^ { \prime } )$ , the output is $f ( x , x ^ { \prime } )$

Similarly, a partial function $h : \Sigma ^ { * } \to \Sigma ^ { * }$ is implemented by a single-input tape Turing machine if analogous conditions hold.

Definition 7. A partial function $f : \Sigma ^ { * } \times \Sigma ^ { * } \to \Sigma ^ { * }$ is computable $i f$ there exists a Turing machine that implements it. Similarly, a partial function $h : \Sigma ^ { * } \to \Sigma ^ { * }$ is computable if a single-input tape Turing machine implements it.

Definition 8. Let $f : \Sigma ^ { * } \times \Sigma ^ { * } \to \Sigma ^ { * }$ be a computable function. If for all computable functions $h : \Sigma ^ { * } \to$ $\Sigma ^ { * }$ , there exists a $p \in \Sigma ^ { * }$ such that

$$
f ( p , \cdot ) = h ( \cdot ) ,
$$

then f is called universal.

Remark 1. Universality is sometimes used to describe Turing machines as opposed to computable functions $( s e e , \ e . g .$ , Li and Vit´anyi [2008]). Our approach of using it as a modifier on computable functions is also seen in the literature (see, e.g., Shen and Vereshchagin [2002] and Boolos et al. [2007]).

We will now extend our definitions of computability and universality to functions which have domain $\Sigma ^ { * } \times \mathcal { X }$ and codomain X . Recall that our decoder $d _ { \mathcal { X } } : \Sigma ^ { * } \to \mathcal { X }$ is a bijective map.

Definition 9. The partial function $g : \Sigma ^ { * } \times \mathcal { X } \stackrel { } {  } \mathcal { X }$ is computable if the partial function $f : \Sigma ^ { * } \times \Sigma ^ { * }  \Sigma ^ { * }$ defined by

$$
f ( x , y ) = \left\{ \begin{array} { l l } { d _ { \mathcal { X } } ^ { - 1 } ( g ( x , d _ { \mathcal { X } } ( y ) ) ) } & { ( x , d _ { \mathcal { X } } ( y ) ) \in d o m ( g ) } \\ { u n d e f t n e d } & { o t h e r w i s e . } \end{array} \right.
$$

is computable. Further, g is called universal if f is universal.

Definition 10. An interpreter is a universal computable partial function $U : \Sigma ^ { * } \times \mathcal { X } \stackrel { } {  } \mathcal { X }$

## B The Prior: The Agent’s or the Designer’s Uncertainty?

It is quite common in previous work to treat the prior as representing the agent’s beliefs. In particular, it is common to take the prior to be a probability distribution that the designer equips the agent with.

In this case, the prior is simply an algorithmic component chosen by the designer. And since the prior is part of the algorithm that induces the agent policy, it no longer makes much sense to evaluate the agent policy relative to the prior. Instead, in this case, the prior should be regarded as part of the object being evaluated.

So what should the prior represent? Ultimately, we want to enable theoretical analysis of the conditions that give rise to misaligned AI agents producing catastrophic outcomes. For that purpose, we want to evaluate an agent policy relative to the beliefs of the designer (or the beliefs of some third party) and not the beliefs of the agent. That is why we take the prior to encode uncertainty stemming from the designer.

## C The POMDP Formulation

Our definitions of baseline value, orthogonal value, and η-catastrophic performance extend naturally to more general settings. In this section, we illustrate this by extending the framework to partially observable environments using the partially observed MDP (POMDP) formalism.

## C.1 Preliminaries

A partially observed MDP is a tuple $( S , \mathcal { A } , \mathcal { O } , P , F , s _ { 0 } , \gamma , r )$ , consisting of an $\mathrm { M D P \ } ( S , \mathcal { A } , P , s _ { 0 } , \gamma , r )$ equipped with an observation space O and observation kernel $F : { \mathcal { S } }  \Delta _ { \mathcal { O } }$

As before, we extend the transition kernel to include an additional transition matrix $P _ { \varnothing }$ . The subscript ∅ indicates that no agent is deployed and we use this matrix to model dynamics when the agent is not deployed. Absent the agent’s influence, the state transitions according to $P _ { \varnothing }$ . On the other hand, with an active agent that selects an action $a \in { \mathcal { A } }$ , the state transitions according to $P _ { a }$

The true POMDP is a random tuple $( S , A , \mathcal { O } , P ^ { * } , F ^ { * } , S _ { 0 } , r ^ { * } , \gamma )$ . Here, $( \cal { S } , \cal { A } , \cal { O } , \gamma )$ is deterministic, but $( P ^ { * } , F ^ { * } , S _ { 0 } , r ^ { * } )$ is random.

## C.2 Baseline and Orthogonal Value

The definitions are exactly as in the main text. For extra clarity, we write them out. Let $H ^ { \varnothing }$ be the state trajectory generated according to the transition matrix $P _ { \varnothing } ^ { * }$ . As before, we refer to $H ^ { \varnothing }$ as the baseline trajectory.

Definition 11. The baseline value is

$$
V ^ { \varnothing } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \varnothing } ) \right] ,
$$

where $H ^ { \varnothing }$ is the baseline trajectory.

Definition 12. The orthogonal value is

$$
V ^ { \perp } = \mathbb { E } \left[ r _ { \gamma } ^ { * } ( H ^ { \perp } ) \right] ,
$$

where $H ^ { \perp }$ is a random trajectory of states that is independent of $r ^ { * }$ and has the simplest distribution.

## D Proofs for Section 3.4 (Gridworld Example)

For these proofs, it will be useful to extend the definition of M and $S _ { 0 }$ to allow for the case $p = 0$ and $p = 1$

Lemma 1. If $p = 1$ , then

$$
\mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] = 0 .
$$

Proof. Since $p = 1$ , we have that $M = n$ with probability 1 and M is thus deterministic. Therefore, $S _ { 0 } \perp r ^ { * }$ . Since $\mathbb { E } [ r ^ { * } ( s ) ] = 0$ for all $s \in S .$ it follows that $\mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] = 0$ □

Lemma 2. Let $S _ { 0 } ^ { \prime }$ be a state sampled independently of $S _ { 0 }$ , in the same way as $S _ { 0 }$ but with parameter $p ^ { \prime } \in [ 0 , 1 ]$ instead. ${ \cal I } f p < p ^ { \prime }$ , then

$$
\mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] > \mathbb { E } [ r ^ { * } ( S _ { 0 } ^ { \prime } ) ] .
$$

Proof. Let $M ^ { \prime }$ be the random distance used for sampling $S _ { 0 } ^ { \prime }$

First note that the distribution of $M ^ { \prime }$ stochastically dominates M. Therefore,

$$
\mathbb { E } \left[ e ^ { - d ^ { 2 } ( S ^ { + } , S _ { 0 } ) / 2 \sigma ^ { 2 } } \right] > \mathbb { E } \left[ e ^ { - d ^ { 2 } ( S ^ { + } , S _ { 0 } ^ { \prime } ) / 2 \sigma ^ { 2 } } \right] .
$$

Since $S ^ { - }$ is uniform on $s$ and the boundary wraps around, $e ^ { - d ^ { 2 } ( S ^ { - } , s ) / 2 \sigma ^ { 2 } }$ has the same distribution for all states $s \in S$

Since $S _ { 0 }$ and $S _ { 0 } ^ { \prime }$ are both independent of $S ^ { - }$ , we therefore have that

$$
\begin{array} { r } { \mathbb { E } \left[ e ^ { - d ^ { 2 } ( S ^ { - } , S _ { 0 } ) / 2 \sigma ^ { 2 } } \right] = \mathbb { E } \left[ e ^ { - d ^ { 2 } ( S ^ { - } , S _ { 0 } ^ { \prime } ) / 2 \sigma ^ { 2 } } \right] . } \end{array}
$$

The result follows.

Proposition 1. $I f p \in [ 0 , 1 )$ , we have that

$$
\mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] > 0
$$

Proof. The result follows immediately from the preceding lemmas.

Proposition 2. Suppose $q \in [ 0 , 1 )$ . Then

$$
\mathbb { E } [ r ^ { * } ( S _ { t + 1 } ^ { \varnothing } ) ] > \mathbb { E } [ r ^ { * } ( S _ { 0 } ) ] .
$$

Proof. Let $r _ { - } ^ { * }$ and $r _ { + } ^ { * }$ be defined for all s as follows:

$$
r _ { - } ^ { * } ( s ) = e ^ { - d ^ { 2 } ( S ^ { - } , s ) / 2 \sigma ^ { 2 } }
$$

$$
r _ { + } ^ { * } ( s ) = e ^ { - d ^ { 2 } ( S ^ { + } , s ) / 2 \sigma ^ { 2 } }
$$

Then, for all t we have that

$$
\mathbb { E } [ r _ { - } ^ { * } ( S _ { 0 } ) ] = \mathbb { E } [ r _ { - } ^ { * } ( S _ { t + 1 } ^ { \varnothing } ) ] .
$$

And for all t

$$
\mathbb { E } [ r _ { + } ^ { * } ( S _ { 0 } ) ] < \mathbb { E } [ r _ { + } ^ { * } ( S _ { t + 1 } ^ { \varnothing } ) ] .
$$

The result follows since $r ^ { \ast } ( s ) = r _ { + } ^ { \ast } ( s ) - r _ { - } ^ { \ast } ( s )$

## E A Simple Distribution Cannot Concentrate on Rare Elements

## E.1 Preliminaries

In the preliminaries, we defined what a computable function is. We now define what it means for a distribution $p \in \Delta _ { \mathcal { X } }$ to be computable. To do so, let us treat $p$ as a function from X to [0, 1]. We say that p is computable if the function is computable.

We make a weak assumption about the programming language.

Assumption 1. For all computable functions $f : \mathcal { X }  \mathcal { X }$ , there exists $c _ { f } \geq 0$ such that for all $x \in \mathcal { X }$

$$
C ( f ( x ) ) \leq C ( x ) + c _ { f } .
$$

Recall that X comes equipped with a computable enumeration. This allows us to define a computable ordering on X based on which element comes first in the enumeration.

In what follows, we will make use of the following notion of density of a subset X of X .

Definition 13. The program density of $X \subseteq { \mathcal { X } }$ at program length $k \in \mathbb { Z } _ { > 0 }$ is

$$
\phi _ { k } ( X ) = { \frac { | \{ p \in \Sigma ^ { \leq k } : U ( p , \epsilon ) \in X \} | } { | \Sigma ^ { \leq k } | } } .
$$

In the main text preliminaries, we defined what a computable function is. We now define what it means for a distribution $p \in \Delta _ { \mathcal { X } }$ on X to be computable. To do so, let us treat $p$ as a function from X to [0, 1]. We say that p is computable if the function is computable.

Definition 14. For all computable $p \in \Delta _ { \mathcal { X } }$ , let

$$
\tilde { c } ( p ) = C ( p ) + 2 \log C ( p ) + c ,
$$

where c is the constant from Lemma 3.

In our lemma and proofs we will write log to indicate the logarithm in base |Σ|.

## E.2 Lemma: Complex States Must Come Late in the Probability Ordering

Definition 15. (probability index) For a computable probability distribution $p \in \Delta _ { \mathcal { X } }$ , the index function index $\mathfrak { c } _ { p } : \operatorname { s u p p } ( p ) \to \mathbb { Z } _ { \geq 1 }$ is defined such that for all $x \in s u p p ( p )$ , index (x) returns the index of x in a list of supp(p) sorted by probabilities under p in descending order, with ties broken by the original ordering that X comes equipped with.

Lemma 3. There exists $c \geq 0$ such that for all computable $p \in \Delta _ { \mathcal { X } }$ and all $x \in \operatorname { s u p p } ( p )$

$$
C ( x ) \leq { \mathrm { l o g ~ i n d e x } } _ { p } ( x ) + C ( p ) + 2 \log C ( p ) + c .
$$

Proof sketch. Let $x \in \operatorname { s u p p } ( p )$ . We will give a program to specify x with length at most log $\operatorname { i n d e x } _ { p } ( x ) +$ $C ( p ) + 2 \log C ( p ) + c .$ The result will follow.

The program will include $p$ as a subroutine and inde $\mathrm { x } _ { p } ( \boldsymbol { x } )$ as a hard-coded constant. The program will enumerate the ordered elements of $p$ and pick the $\operatorname { i n d e x } _ { p } ( x ) \nmid$ th one. The length L of such a program will be upperbounded by

$$
L \leq C ( p ) + \lceil \log \operatorname { i n d e x } _ { p } ( x ) \rceil + 2 \log ( C ( p ) ) + c ^ { \prime } .
$$

Let us explain each term in turn. The term $C ( \boldsymbol p )$ shows up because that number of symbols is suficient to specify the subroutine for $p .$ The term $\lceil \log \operatorname { i n d e x } _ { p } ( x ) \rceil$ shows up because that number of symbols is suficient to hard-code $\operatorname { i n d e x } _ { p } ( x )$ . The term $2 \log ( \bar { C } ( p ) )$ is an upperbound bound on the number of symbols necessary to delimit when the subroutine p ends and harded coded $\operatorname { i n d e x } _ { p } ( x )$ begins. Finally, the constant $c ^ { \prime }$ corresponds to overhead in the enumeration of ordered elements that is independent of $p$ and inde $\mathrm { x } _ { p } ( x )$ . We let $c = c ^ { \prime } + 1$ so we can ignore the ceiling in the complexity of $\operatorname { i n d e x } _ { p } ( x )$ . The result follows.

Because $p$ may have infinite support, the enumeration and selection procedure above needs to be handled carefully. We spell out the exact algorithm in Appendix E.4. □

Remark 2. The preceding lemma implies that complex elements must have high probability index.

Next, we state a corollary that we don’t actually use but is helpful for building intuition.

Corollary 1. There exists $c \geq 0$ such that for all computable $p \in \Delta _ { \mathcal { X } }$ and all $x \in \operatorname { s u p p } ( p )$

$$
p ( x ) \leq { \frac { 1 } { | \Sigma | ^ { C ( x ) - C ( p ) - 2 \log C ( p ) - c } } } ,
$$

where c is the constant from the preceding lemma.

Proof. Rearranging terms we have the following:

$$
\begin{array} { r l } & { \ \log \operatorname { i n d e x } _ { p } ( x ) \geq C ( x ) - C ( p ) - 2 \log C ( p ) - c } \\ & { \implies \operatorname { i n d e x } _ { p } ( x ) \geq | \Sigma | ^ { C ( x ) - C ( p ) - 2 \log C ( p ) - c } } \\ & { \qquad \implies p ( x ) \leq \frac { 1 } { | \Sigma | ^ { C ( x ) - C ( p ) - 2 \log C ( p ) - c } } . } \end{array}
$$

The last step follows because if an element x has $\operatorname { i n d e x } _ { p } ( x )$ then its probability must be less than $1 / \mathrm { i n d e x } _ { p } ( x )$ □

## E.3 Main Result

Proposition 3. Let $X \subseteq { \mathcal { X } }$ be a random set and Z an X-measurable random variable. There exists   
$c \geq 0$ such that for all computable distributions $p \in \Delta _ { \mathcal { X } }$

$$
\begin{array} { r } { \mathbb { P } \left( \widetilde { Y } \in X \middle | Z \right) \leq \underset { k } { \operatorname* { s u p } } \mathbb { E } [ \phi _ { k } ( X ) \middle | Z ] | \Sigma | ^ { C ( p ) + 2 \log C ( p ) + c } , } \end{array}
$$

with probability $^ { 1 , }$ where $\widetilde { Y }$ is sampled from p independently of X. The constant c is the constant from Lemma 3 plus 1.

Proof. Let $n _ { p } = | \operatorname { s u p p } ( p ) | \in \mathbb { Z } _ { \ge 0 } \cup \{ \infty \}$ . For all $n \in \{ 1 , 2 , . . . , n _ { p } \}$ (interpreted as $\mathbb { Z } _ { \geq 1 }$ if $n _ { p } = \infty )$ , let $P _ { n } = \{ x : \operatorname { i n d e x } _ { p } ( x ) \leq n \}$ be the first n elements according to the probability ordering induced by $p .$ By Lemma $^ { 3 , }$ we have that for each element $x \in P _ { n } .$ the complexity is bounded as $C ( x ) \leq \log n + \tilde { c } ( p )$ 2 where $\tilde { c } ( p ) = C ( p ) + 2 \log C ( p ) + c$ . Thus,

$$
P _ { n } \subseteq \mathcal { X } _ { \lfloor \log n + \tilde { c } ( p ) \rfloor } ,
$$

where $\mathcal { X } _ { k }$ is the set of elements of X with complexity $k$ or less.

What fraction of elements in $P _ { n }$ are also in X? We compute a bound

$$
\begin{array} { r l } { \frac { \displaystyle | P _ { n } \cap X | } { \displaystyle | P _ { n } | } \stackrel { ( a ) } { \leq } \frac { \displaystyle | \mathcal { X } _ { \lfloor \log n + \tilde { c } ( p ) \rfloor } \cap X | } { \displaystyle | P _ { n } | } } & { } \\ & { \stackrel { ( b ) } { \leq } \frac { \displaystyle \phi _ { \lfloor \log n + \tilde { c } ( p ) \rfloor } ( X ) | \Sigma ^ { \leq \lfloor \log n + \tilde { c } ( p ) \rfloor } } { \displaystyle | P _ { n } | } } \\ & { \stackrel { ( c ) } { \leq } \phi _ { \lfloor \log n + \tilde { c } ( p ) \rfloor } ( X ) \frac { \displaystyle | \Sigma | ^ { \log n + \tilde { c } ( p ) + 1 } } { n } } \\ & { = \phi _ { \lfloor \log n + \tilde { c } ( p ) \rfloor } ( X ) | \Sigma | ^ { \tilde { c } ( p ) + 1 } } \end{array}\tag{2}
$$

Step (a) follows since $P _ { n } \subseteq \mathcal { X } _ { \lfloor \log n + \tilde { c } ( p ) \rfloor }$ . Step (b) follows since each element in $\mathcal { X } _ { \lfloor \log n + \tilde { c } ( p ) \rfloor }$ as at least one program associated with it in $\sum \bar { \lfloor \log n + \tilde { c } ( p ) \rfloor }$ and distinct elements have distinct programs associated with them. Step (c) follows since $\begin{array} { r } { | \Sigma ^ { \leq k } | = \sum _ { j = 0 } ^ { k } | \Sigma | ^ { j } \leq | \Sigma | ^ { k + 1 } } \end{array}$ , and since $\left| P _ { n } \right| = n$

Finally, we will refer to each of $P _ { 1 } , P _ { 2 }$ , and so on, as prefixes. It is a known fact that a distribution is equivalent to a mixture of uniform distributions on these prefixes. That is, $\tilde { Y }$ can be written as

$$
\tilde { Y } = U _ { W }
$$

where W is a random variable with support $\{ 1 , 2 , . . . , n _ { p } \}$ and $( U _ { n } : n \in \{ 1 , . . . , n _ { p } \} )$ is a collection of uniform random variables with $U _ { n }$ uniform on $P _ { n }$ . Here, the $U _ { n }$ and W are mutually independent. Note that the $U _ { n }$ and $W$ are also independent of X. It follows that,

$$
\begin{array} { r l } { \| V ( \vec { Y } \in X | Z ) = } & { \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \| r ( w - w | Z ) \| r ( \theta _ { 1 } \in X | Z , W - w ) } \\ & { = \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \| r ( w - w ) \| \nabla [ ( ( \theta _ { 1 } \in X ) | Z , W - n ] } \\ & { = \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \| r ( w - w ) \mathbb { E } [ ( ( \theta _ { 1 } \in X ) | Z , W - n ] } \\ & { = \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \mathbb { E } [ ( W - w ) \mathbb { E } [ ( \mathbb { I } _ { n } \in X ) | Z ] } \\ & { = \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \mathbb { E } [ ( W - w ) \mathbb { E } [ \mathbb { E } [ ( \mathbb { I } _ { n } \in X ) | X , Z ] | Z ] } \\ & { \stackrel { ( a ) } { \leq } \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \mathbb { E } [ ( W - w ) \mathbb { E } [ \mathbb { I } _ { n } ( w + w ) \mathbb { E } ] ) ( \mathbb { I } _ { n } \| \mathbb { Z } ^ { ( b ) ( 1 ) } | Z ) } \\ & { \leq \displaystyle \sum _ { w \in \mathrm { a r o w } , W _ { 1 } } \mathbb { E } [ ( \mathbb { I } _ { n } \log _ { b \in \mathbb { I } _ { n } ( X ) \cap J } ( X ) | Z ) + \mathbb { I } _ { n } ( Z ) | Z ] } \\ & { \leq \displaystyle \operatorname* { m a x } _ { w \in \mathrm { a r o w } , W _ { 1 } } \mathbb { E } [ ( \mathbb { I } _ { n } \log _ { b \in \mathbb { I } _ { n } ( X ) \cap J } ( X ) | Z ) ] ^ { w _ { 1 } ^ { \mathbb { I } } ( \mathbb { I } _ { n } ) + 1 } } \\ &  \leq \displaystyle \operatorname* { m a x } _ { w \in \mathrm { a r o w } , W _ { 1 } } \mathbb { E } [ ( \mathbb { I } _  \end{array}
$$

where (a) follows from (2).

Proposition 4. Let $X \subseteq { \mathcal { X } }$ be a random set. There exists $c \geq 0$ such that for all computable distributions $p \in \Delta _ { \mathcal { X } }$ ，

$$
\begin{array} { r } { \mathbb { P } \left( \widetilde { Y } \in X \right) \le \underset { k } { \operatorname* { s u p } } \mathbb { E } [ \phi _ { k } ( X ) ] \vert \Sigma \vert ^ { C ( p ) + 2 \log C ( p ) + c } , } \end{array}
$$

where $\widetilde { Y }$ is sampled from p independently of X. The constant c is the constant from Lemma 3 plus 1.

Proof. Follows immediately from our previous result.

## E.4 Proof Details

Proof details for Lemma 3. The proof sketch left out the details of the enumeration and selection algorithm. We discuss these details now.

We loop through all elements $x _ { 1 } , x _ { 2 } , \ldots$ using the computable enumeration of X and compute $p ( x _ { 1 } ) , p ( x _ { 2 } )$ (this is possible because we assume $p ( z )$ is rational for all $z \in \mathcal { X } )$ . At each step k, we compute an ordered

list $( y _ { 1 } ^ { ( k ) } , . . . , y _ { k } ^ { ( k ) } )$ by sorting $( x _ { 1 } , . . . , x _ { k } )$ by their probabilities, breaking ties via the original enumeration on X. We continue this process and stop once index $\boldsymbol { \cdot } _ { p } ( \boldsymbol { x } ) \in \{ 1 , 2 , . . . , k \}$ and

$$
\sum _ { i = 1 } ^ { k } p ( y _ { i } ^ { ( k ) } ) > 1 - p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( k ) } ) .
$$

This happens eventually because the partial sums converge to 1 and $p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( k ) } )$ is a non-decreasing sequence that is eventually positive. The reason $p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( k ) } )$ is eventually positive is because if $\begin{array} { r } { p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( k ) } ) = } \end{array}$ 0 for all k, then the support of p has at most $\mathrm { i n d e x } _ { p } ( x ) - 1$ elements, a contradiction.

Suppose the process stops at $k = n$ . When the process stops, we have

$$
\{ y \in { \mathcal { X } } : p ( y ) \geq p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( n ) } ) \} = \{ y \in \{ x _ { 1 } , . . . , x _ { n } \} : p ( y ) \geq p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( n ) } ) \} ,
$$

because all elements outside of $\{ x _ { 1 } , . . . , x _ { n } \}$ must have probability less than $p ( y _ { \mathrm { i n d e x } _ { p } ( x ) } )$ by the stopping criterion. It follows that $y _ { \mathrm { i n d e x } _ { p } ( x ) } ^ { ( n ) } = x .$ □