# Propose to Learn, Learn to Propose: Evaluability-Aware Assistance under Bounded Rationality

Yifan Zhu<sup>1,2</sup> Sammie Katt<sup>1,2</sup> Samuel Kaski<sup>1,2,3</sup>

<sup>1</sup>ELLIS Institute Finland

<sup>2</sup>Department of Computer Science, Aalto University, Finland <sup>3</sup>Department of Computer Science, University of Manchester, United Kingdom {yifan.zhu, sammie.katt, samuel.kaski}@aalto.fi

## Abstract

AI assistants often collaborate by proposing candidate edits, plans, or designs that users evaluate before adoption. Existing assistance methods focus on proposal quality or user-goal inference, often assuming that the user can reliably evaluate any proposal, which can fail in practice because of bounded rationality. We study evaluability-aware proposal planning, where proposals serve both as task interventions and as probes for learning latent preferences and evaluation constraints, where the resulting belief updates then guide later proposals. We formalise this setting as ProSE, a hidden-parameter sequential assistance problem, and instantiate it with a KL-regularised bounded-rational binary response model in which acceptance trades off value gain against a distance-dependent evaluability penalty. Analysing the planning consequence of this likelihood reveals that likely accepted proposals and informative probes need not coincide, which explains why planners that only pursue acceptance systematically underperform. We operationalise ProSE with PROSE-PLAN, a depth-2 Bayes-adaptive planner that scores proposals by possible responses and response-induced posterior beliefs. In controlled graph simulations, PROSE-PLAN improves over evaluability-unaware and myopic baselines when evaluation cost is the bottleneck, and a probe-commit ablation confirms that our approach selects informative proposals that simpler methods miss. Our results thus identify user evaluability as a planning-relevant dimension of AI assistance, complementary to generation quality and preference inference.

## 1 Introduction

In recent years, a proposal-evaluation paradigm has been widely applied in AI-assisted decisionmaking, where the assistant suggests a proposal, and the user decides whether to incorporate it through evaluation. This is common in coding, writing, and design tasks, where assistance should preserve user control by proposing evaluable next states rather than acting autonomously.

Effective assistance in this paradigm requires more than generating high-quality proposals. Existing assistance primarily models what the user wants, or how to act under uncertainty about the user’s goals [1–5]. This addresses a central aspect of assistance, but neglects whether the user can evaluate a particular proposal from the current state. This matters because human decision-making is bounded by cognitive and computational constraints [6–10]. A sweeping code refactor may improve the codebase in principle, yet be too costly to inspect, while a smaller patch may be less globally optimal, but easier to verify, and therefore could be useful. Empirical work on AI-assisted programming has exposed consistent concerns that users can struggle with proposals that are too complex, and excessive AI information can degrade rather than improve decision quality [11, 12]. While prior work has addressed aspects of this problem, such as bounded-rational user modelling [3, 4], query difficulty in preference learning [13, 14], and display timing [15] (discussed further in Appendix A), how evaluability shapes the user’s response and what the assistant can learn from it has received little attention in sequential assistance.

We formalise this problem as proposal-based sequential assistance (ProSE). In ProSE, each proposal has a dual role: it is a candidate intervention on the task and an observation probe for learning about the user. The assistant must model the user’s response both to predict what will happen if a proposal is shown and to update its belief about the user’s latent parameters. A high-value proposal may be rejected because it is difficult to evaluate, while a more moderate proposal may produce a response that improves future assistance. Thus, the assistant should reason about how possible responses change future proposal choice, rather than only maximising immediate acceptance.

Within ProSE, we instantiate a tractable binary response model motivated by information-theoretic bounded rationality [16, 17]. The response likelihood trades off user-value gain against a distancedependent evaluability penalty, so the same proposal can be rejected either because it is low-value or too costly to evaluate from the current state. We then analyse its structural properties and consequences for proposal planning. The model induces an acceptancefrontier as the response decision boundary, and an informationfrontier, characterising which proposal distances are informative about the user’s evaluability parameter. Our main result shows that the most informative proposal about evaluability can lie on the expected-rejection side of the acceptance frontier. Thus, likely-to-be-accepted proposals and informative proposals need not coincide: rejection can be useful evidence for future assistance, not merely failed assistance.

We operationalise the framework with PROSE-PLAN, a depth-2 Bayes-adaptive planner that scores proposals by possible responses and response-induced posterior beliefs. We evaluate it in two controlled graph tasks: a branching corridor for end-to-end performance when high-value proposals are difficult to evaluate, and a probe-commit task isolating response-dependent belief updates. The results show that evaluability-aware planning helps when evaluation cost is the bottleneck, and that anticipating what a response reveals about the user leads to better subsequent proposals. Our contributions are to formulate evaluability-aware proposal planning, instantiate it with a boundedrational response model and frontier analysis, and validate a minimal Bayes-adaptive planner in controlled simulations. Together, these results identify user evaluability as a planning-relevant dimension of AI assistance, complementary to generation quality and preference inference.

## 2 Preliminaries

Our work utilises two important concepts: Bayesian decision-making under latent model uncertainty and information-theoretic bounded rationality. This section briefly introduces both.

Bayesian decision-making. Sequential decision-making is typically formalised as a Markov decision process (MDP) $( S , { \mathcal { A } } , { \mathcal { T } } , { \mathcal { R } } , { \gamma } )$ , where $s$ is a state space, $\mathcal { A }$ is an action space, ${ \mathcal { T } } ( s ^ { \prime } \mid s , a ) : S \times { \mathcal { A } } \to$ $\Delta ( S )$ is the transition dynamics, $\mathcal { R } ( s , a , s ^ { \prime } ) : \mathcal { S } \times \mathcal { \bar { A } } \times \mathcal { S } $ R is the reward function, and $\gamma \in [ 0 , 1 ]$ is a discount factor. The objective is to find a policy $\pi : S  \Delta ( { \mathcal { A } } )$ that maximises expected return: $\pi ^ { \star } \in$ arg ma $\begin{array} { r } { \mathrm { x } _ { \pi } \operatorname { \mathbb { E } } _ { \tau \sim ( \pi , \mathcal { T } ) } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } \mathcal { R } \big ( s _ { t } , a _ { t } , s _ { t + 1 } \big ) \right] } \end{array}$ , where $a _ { t } \sim \pi ( \cdot \mid s _ { t } ) , s _ { t + 1 } \sim \mathcal { T } ( \cdot \mid s _ { t } , a _ { t } )$ and T is the planning horizon [18].

In many settings, the transition or reward model is not known a priori, often parameterised by latent parameters z and learned from experiences, which is costly in real-world applications. The Bayesian alternative is to assume a prior $p _ { 0 } ( z )$ and maintain a belief $b _ { t } ( z ) = p ( z \mid \bar { h _ { t } } )$ over the latent parameter given the interaction history $h _ { t } = ( s _ { 0 } , a _ { 0 } , \ldots , a _ { t - 1 } , s _ { t } )$ , so that the agent is able to make decisions that are optimal with respect to evolving belief over the world. Specifically, after observing a transition $\left( { { s _ { t } } , { a _ { t } } , { s _ { t + 1 } } } \right)$ , the belief is updated by Bayes’ rule:

$$
b _ { t + 1 } ( z ) = p ( z \mid h _ { t } , a _ { t } , s _ { t + 1 } ) \propto { \mathcal { T } } _ { z } ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) b _ { t } ( z ) .\tag{1}
$$

The optimal solution is then the policy that optimises future reward with respect to this belief. The Bayes-adaptive MDP (BA-MDP) formalises this idea by casting planning under latent model uncertainty in an augmented belief state $\left( s , b \right) \left[ 1 9 \right]$ . Typical solutions to (BA-)MDPs compute and maximise the Q-value: the expected value of taking an action in a state, which can be computed recursively with the Bellman equation with $h$ steps of lookahead remaining:

$$
Q _ { h } ^ { \star } ( s , b , a ) = \int _ { z } b ( z ) \sum _ { s ^ { \prime } } \mathcal { T } _ { z } ( s ^ { \prime } \mid s , a ) \left[ \mathcal { R } _ { z } ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q _ { h - 1 } ^ { \star } ( s ^ { \prime } , b ^ { a , s ^ { \prime } } , a ^ { \prime } ) \right] d z ,\tag{2}
$$

where $Q _ { 0 } ( \cdot )$ is typically defined as $0 , b ^ { a , s ^ { \prime } } ( z )$ is the posterior belief after observing $s ^ { \prime } .$

Information-Theoretic Bounded Rationality. Bounded rationality refers to the observation that agents operate under internal cognitive and computational constraints that prevent them from fully optimising their decisions [6, 7]. The computational-rationality perspective treats such deviations from perfect rationality as a result of agents making the best use of their limited cognitive resources under subjective utility [8, 20, 9]. In particular, information-theoretic bounded rationality [16, 17] formalises user behaviour as solving a one-shot KL-regularized decision problem, where the probability of picking an action $a \in { \mathcal { A } }$ is described as maximizing the trade-off between utility $\bar { U : A  }$ R and information cost in the form of deviating from the prior policy $q \in \Delta ( \mathcal { A } )$

$$
p ^ { \mathrm { I T B R } } = \underset { p \in \Delta ( A ) } { \arg \operatorname* { m a x } } \left[ \sum _ { a \in \cal A } p ( a ) U ( a ) - \frac { 1 } { \kappa } D _ { \mathrm { K L } } ( p  { | | } \ q ) \right]\tag{3}
$$

where $\kappa > 0$ is an inverse-temperature parameter that controls the user’s cognitive effort. A larger κ indicates the user invests more effort in deviating from the default to pursue higher utility, while $\kappa  0$ recovers the default action regardless of utility. Since the objective in Eq. (3) is strictly concave in $p$ whenever $q ( a ) > 0$ for all $^ { a , }$ it admits a unique closed-form solution (Appendix B.1 or [17] for details). Solving the simplex-constrained optimization gives the Gibbs policy:

$$
p ^ { \mathrm { I T B R } } ( a ) \propto q ( a ) \exp { \left( \kappa U ( a ) \right) } .\tag{4}
$$

Eq. (4) shows that the optimal action is the default distribution $q$ tilted toward higher-utility actions, with the degree of tilting controlled by $\kappa .$

## 3 Problem Setup: Proposal-based Sequential Assistance

We study a sequential design task family where the user iteratively revises the current design (state) toward a satisfactory outcome. In this interaction paradigm, the assistant proposes candidate next states or edits, which the user then evaluates and responds to (e.g., accept or reject), and the realised next state is then determined by the current state, the proposal, and the user’s response. The assistant’s primary goal is to help the user achieve a better final artefact through proposals, and the main difficulty is that good proposals depend on the user’s likelihood of accepting them. As we’ve motivated above, this likelihood depends not only on its quality, i.e., how much it aligns with the user’s preferences, but also on whether a user can (and wants to) evaluate it. In this section, we formalise this problem setting as Proposal-based Sequential assistancE (ProSE).

Definition 3.1 (ProSE process). A ProSE process consists of a task state space $s ,$ a candidate proposal space C with ${ \bar { \mathcal { C } } } ( s ) \subseteq S .$ , a user response space $\mathcal { V } _ { \mathrm { ~ ~ } }$ , a latent user parameter space $\mathcal { Z } = \Phi \times \Theta$ with prior $p _ { 0 }$ characterising preferences (Φ) and evaluability (Θ) respectively, a user response model family $\begin{array} { r } { P ( \boldsymbol { \bar { y } } \mid s , \boldsymbol { \tilde { s } } , z ) : \bar { \mathcal { S } } \stackrel { \left. \right.} { \times } \mathcal { C } ( s ) \times \mathcal { Z }  \Delta ( \mathcal { V } ) } \end{array}$ ), response-mediated transition dynamics $\tau ^ { y } ( s ^ { \prime } \mid$ $\begin{array} { r } { s , \tilde { s } , y ) : { S } \times \mathcal { C } ( s ) \times \mathcal { Y }  \Delta ( { S } ) } \end{array}$ , and a family of user value functions $\mathcal { V } _ { \Phi } = \{ V _ { \phi } : \stackrel { \cdot } { S }  \mathbb { R } \} _ { \phi \in \Phi }$

In ProSE, the assistant’s action space is a set of proposals $\mathcal C ( s )$ , which we will refer to as $\tilde { s }$ (as opposed to action a). At each time step, given a selected proposal ${ \tilde { s } } ,$ the user responds according to $y \sim P ( \cdot \mid s , \tilde { s } , z )$ , and the task state evolves according to $\bar { s } ^ { \prime } \sim \mathcal { T } ^ { y } ( \cdot \mid s , \tilde { s } , y )$ . In design tasks, the dynamics are typically deterministic given user actions, which we encode with the deterministic transition function $\mathcal { T } ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , y )$ . For example, the user’s reply $y$ could be to ignore the proposal and apply some edit themself to change the state.

ProSE is a specific instantiation of the Bayes-adaptive MDP. Specifically, given user parameters $z ,$ the problem can be fully formalised and solved as an MDP, with two notable details: (1) the transition model is defined by the user model, and (ii) the reward function is zero $\mathscr { R } ( s _ { t } , a , s _ { t + 1 } ) = 0$ except for the last state $s _ { T }$ , of which the value is determined by the user $\mathscr { R } ( \cdot , \cdot , s _ { T } ) = V _ { \phi } ( \dot { s } _ { T } )$ . Of course, the user parameters are not known. This is resolved by assuming a prior $p _ { 0 } ( z ) = { \dot { p } } _ { 0 } ( \theta , \phi )$ over the parameters and casting the problem as a Bayes-adaptive MDP as discussed in Section 2. Here, the transitions are fully specified by z and the belief over $\tau$ can be summarized with a belief over z:

$$
b _ { t } ( z ) = p ( z \mid h _ { t } ) \propto P ( y _ { t - 1 } \mid s _ { t - 1 } , \tilde { s } _ { t - 1 } , z ) b _ { t - 1 } ( z ) .\tag{5}
$$

The optimal solution also follows from the Bayes-adaptive formulation (Eq. 2), with the notable adaptation that the value of the last step is now determined by the user: $Q _ { 0 } = V _ { \phi }$

## 4 Bounded-Rational Binary User Response Model

In Section 3, we reduced the assistant’s problem to belief-state planning over a response likelihood $P ( y \mid s , \tilde { s } , z )$ . Here, we derive a concrete binary model for users in design tasks where their response is either to accept or reject the AI’s proposal $\dot { ( \mathcal { Y } ) } = \lbrace \mathrm { a c c , r e j } \rbrace )$ . This results in a process that either transitions to the proposed state ${ \tilde { s } } ,$ if the user accepts, or leaves the state s unchanged if rejected:

$$
T ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , \operatorname { a c c } ) = \mathbf { 1 } \{ s ^ { \prime } = \tilde { s } \} \mathrm { a n d } T ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , \mathbf { r e j } ) = \mathbf { 1 } \{ s ^ { \prime } = s \} .
$$

This setting models significant parts of interactions with LLMs in coding and writing, while giving a minimal model in which evaluability can affect both task progress and user-model learning, demonstrating interesting behaviour and insights as we show in the analysis in Section 4.2.

## 4.1 A Tractable Evaluability-Aware Response Model

Here, we derive the binary user model from first principles. By applying Eq. (4) to proposal responses in ProSE, we have $P ( y \mid \mid s , \tilde { s } , z ) \propto q _ { \theta } ( y \mid s , \tilde { s } )$ exp $\left( \kappa U ( y ; s , \tilde { s } , \phi ) \right)$ , where $q _ { \theta }$ is the user’s low-effort default response, and $U ( y ; s , \tilde { s } , \phi )$ is the utility of response y. For binary responses, this gives

$$
P ( \operatorname { a c c } \mid s , \tilde { s } , z ) = \sigma \left( \kappa \Delta U _ { \phi } ( s , \tilde { s } ) + \lambda _ { \theta } ( s , \tilde { s } ) \right) ,\tag{6}
$$

where $\Delta U _ { \phi } ( s , \tilde { s } ) = U ( \operatorname { a c c } ; s , \tilde { s } , \phi ) - U ( \operatorname { r e j } ; s , \tilde { s } , \phi )$ and $\begin{array} { r } { \lambda _ { \theta } ( s , \tilde { s } ) = \log \frac { q _ { \theta } ( \operatorname { a c c } \lvert s , \tilde { s } ) } { q _ { \theta } ( \operatorname { r e j } \lvert s , \tilde { s } ) } } \end{array}$ (see Appendix B.2 for details). The response utility U is induced by the user’s latent value function. For a general response $y ,$ define $\hat { U ( y ; s , s , \phi ) } \stackrel { * } { = } \mathbb { E } _ { s ^ { \prime } \sim \hat { T } ^ { y } ( \cdot | s , \tilde { s } , y ) } [ \hat { V } _ { \phi } ( s ^ { \prime } ) ]$ . Under the deterministic binary transition above, $U ( \mathrm { a c c } ; s , \tilde { s } , \phi ) = V _ { \phi } ( \tilde { s } )$ and $U ( { \mathrm { r e j } } ; s , \tilde { s } , \phi ) = V _ { \phi } ( s )$ . The utility difference therefore becomes the value gain, quantifying the difference in user preference between the proposed and current state:

$$
\Delta U _ { \phi } ( s , \tilde { s } ) = V _ { \phi } ( \tilde { s } ) - V _ { \phi } ( s ) = : \Delta V _ { \phi } ( s , \tilde { s } ) .\tag{7}
$$

As for the default response log-odds $\lambda _ { \theta } ( s , \tilde { s } )$ , instead of assuming or estimating an arbitrary default policy, we parameterise it directly to represent evaluability burden. Here, we make a modelling choice that the default log-odds decreases with proposal distance, to capture proposal-dependent evaluability, where a proposal farther from the current state requires more cognitive effort to evaluate and therefore faces a stronger default toward rejection. This choice is motivated by evidence that status-quo bias scales with the magnitude of the proposed change [21] and that cognitive load increases with the complexity of the evaluation task [22]. Thus, the default log-odds is instantiated as

$$
\lambda _ { \theta } ( s , \tilde { s } ) = - \alpha g \big ( d ( s , \tilde { s } ) \big ) ,\tag{8}
$$

where $d ( s , \tilde { s } ) \geq 0$ measures proposal distance on the state space, $\alpha \geq 0$ controls the overall strength of the distance penalty, and g is a non-decreasing function with $g ( 0 ) = 0$ that maps distance to evaluation burden. We use a transform g rather than raw distance because evaluation difficulty need not scale proportionally with distance [21]: assessing a small edit may be easy, while a moderately larger change can be disproportionately harder to evaluate.

Substituting Eqs. (7) and (8) into Eq. (6) gives a tractable predictive likelihood:

$$
P ( \operatorname { a c c } \mid s , \tilde { s } , z ) = \sigma ( \kappa [ \Delta V _ { \phi } ( s , \tilde { s } ) - \rho g ( d ( s , \tilde { s } ) ) ] ) ,\tag{9}
$$

where $\rho : = \alpha / \kappa$ . This instantiates the user parameters $z = ( \phi , \theta )$ from Section 3 as $z = ( \phi , \rho , \kappa )$ . In this response model, the preference parameter ϕ determines the user’s preference over states, while the evaluability slope $\rho \ge 0$ determines how much proposal distance shifts the acceptance threshold, and the response sharpness $\kappa > 0$ controls how decisively the user responds to the value-cost difference.

## 4.2 User Model Analysis: Acceptance and Information Frontiers

We now analyse the response model in Eq. (9) to identify the structural properties and consequences it induces for proposal planning. We first derive the acceptance frontier, which characterises when a proposal is more likely to be accepted than rejected, and shows that distance raises the value gain required for acceptance. We then derive an information frontier, which characterises which proposal distances make the response most informative about the user’s evaluability slope $\rho .$ These frontiers give two planning lessons. First, likely-to-be-accepted proposals and informative proposals need not coincide, so maximising immediate acceptance can miss useful probes. Second, the most informative proposals about evaluability can lie on the expected-rejection side of the acceptance frontier, rather than on the frontier where the response uncertainty is maximal. This indicates that rejection is not always merely failed assistance, but can be an informative response that improves future proposal choice through user-model learning. The theoretical results thus motivate planning over response-dependent belief updates rather than myopic acceptance maximisation.

(a) Acceptance frontier: $\Delta V _ { \phi } = \rho g ( d )$  
![](images/47a88075a6b7bacbd91d4663fd12663bab8c14f94aa3c4392abe2edd5ed74e67.jpg)

![](images/9608db48c8b197670c731f0bb5802b4b8a61d82b2a02f7e7a791bb0ba6b693dd.jpg)  
Figure 1: Acceptance and information frontiers. (a) The acceptance frontier $\Delta V _ { \phi } = \rho g ( d )$ separates likely acceptance and rejection. (b) With $g ( t ) = t ^ { 2 } , \kappa = 1 , \rho = 0 . 5 ,$ , and $\Delta V _ { \phi } ( t ) = t ,$ Fisher information about $\rho$ peaks beyond the acceptance frontier, where rejection is more probable.

Acceptance Frontier. In Eq. (9), acceptance is more likely when the user-value gain $\Delta V _ { \phi }$ exceeds the evaluability penalty $\rho g ( d )$ .The boundary where these two terms balance,

$$
\Delta V _ { \phi } ( s , \tilde { s } ) = \rho g ( d ( s , \tilde { s } ) ) ,\tag{10}
$$

defines the acceptancefrontier. It is the curve in the (distance, value gain) plane separating proposals that are more likely to be accepted from those that are more likely to be rejected, as illustrated in Figure 1(a). The frontier captures the immediate effect of evaluability: farther proposals require larger value gains to remain acceptable.

Information Frontier. The acceptance frontier characterises immediate acceptance. For sequential proposal planning, the assistant also needs to know which responses are useful for learning the latent user parameters. We therefore analyse which proposal distances make a single accept/reject response informative about the evaluability slope $\rho .$

To isolate the structural effect of proposal distance, we analyse the response likelihood conditional on a latent user-parameter $z = ( \phi , \rho , \kappa )$ This does not assume that the assistant knows z, but characterises the likelihood term that the assistant later integrates under its posterior belief. Consider a one-dimensional proposal path $\{ \tilde { s } ( t ) \} _ { t \geq 0 }$ from the current state s, parameterised by distance so that $\tilde { s } ( 0 ) = s$ and $d ( s , { \tilde { s } } ( t ) ) = t .$ . Along this path, define

$$
\eta _ { z } ( t ) = \kappa [ \Delta V _ { \phi } ( t ) - \rho g ( t ) ] , \qquad \Delta V _ { \phi } ( t ) : = V _ { \phi } ( \tilde { s } ( t ) ) - V _ { \phi } ( s ) ,\tag{11}
$$

and let $p _ { z } ( t ) = \sigma ( \eta _ { z } ( t ) )$ be the acceptance probability.

Since the response is binary, observing the user’s response gives one Bernoulli observation with parameter $p _ { z } ( t )$ . We measure how informative this observation is about $\rho$ using Fisher information, a local sensitivity measure of the likelihood with respect to a parameter [23, 24]. For the Bernoulli likelihood induced by Eq. (9), the Fisher information about $\rho$ is (see Appendix C.1 for details)

$$
I _ { z } ^ { \rho } ( t ) = \kappa ^ { 2 } g ( t ) ^ { 2 } p _ { z } ( t ) ( 1 - p _ { z } ( t ) ) .\tag{12}
$$

Eq. (12) contains two competing factors. The term $g ( t ) ^ { 2 }$ grows with proposal distance and captures sensitivity to $\rho \colon$ when a proposal is very close to the current state, changing $\rho$ has little effect on the response probability. The term $p _ { z } ( \dot { t } ) ( 1 - p _ { z } ( t ) )$ is the Bernoulli response variance, which is largest when acceptance and rejection are equally likely and vanishes when the response is almost deterministic. Thus, very local proposals are weakly informative because they are insensitive to $\rho ,$ while very distant proposals are weakly informative because they are almost surely rejected. Therefore, Eq. (12) indicates that information about $\rho$ must peak at some intermediate distance.

Now, we identify on which side of the acceptance frontier the peak lies with the following proposition. Proposition 4.1 (Information frontier conditional on user parameters). For any latent user-parameter vector $z = ( \phi , \rho , \kappa )$ with $\rho > 0$ and $\kappa > 0 ,$ assume $\Delta V _ { \phi } ( \dot { t } )$ is continuously differentiable, $\dot { \Delta } V _ { \phi } ( t ) =$ $o ( g ( t ) )$ as $t \to \infty$ , and $g$ satisfies $g ( 0 ) = 0 , g ( t ) > \dot { 0 } , g ^ { \prime } ( t ) > 0$ for all $t > 0$ , and $g ( t )  \infty$ Then $I _ { z } ^ { \rho } ( t )$ in Eq. (12) satisfies: $\mathrm { ( i ) } \ I _ { z } ^ { \rho } \mathrm { ( 0 ) } = 0 ; \mathrm { ( i i ) } \ I _ { z } ^ { \rho } \mathrm { ( } t \mathrm { ) } \to 0$ as $t \to \infty ;$ (iii) $I _ { z } ^ { \rho } ( t )$ attains a global maximum at some $t ^ { \star } \in ( 0 , \infty ) ;$ ; and (iv) every global maximiser satisfies $p _ { z } ( t ^ { \star } ) < 1 / 2$

Proof provided in Appendix C. Figure 1(b) illustrates this effect for $g ( t ) = t ^ { 2 } , \kappa = 1 , \rho = 0 . 5$ , and $\Delta V _ { \phi } ( t ) = t \colon$ the information maximum lies beyond the acceptance frontier, where rejection is more likely than acceptance. At the frontier, response uncertainty is maximal, but the sensitivity factor $g ( t ) ^ { \dot { 2 } }$ is still increasing, pushing the information maximum to the expected-rejection side.

This separates two notions that a myopic assistant can conflate. A proposal can be easy to accept but uninformative about evaluability, or likely to be rejected but useful for locating the user’s evaluability boundary. The information frontier is therefore not a new planning objective, but a diagnostic of the response model: it shows why a sequential assistant should reason about how each possible response changes its belief over the user, rather than only maximising immediate acceptance. This is the motivation for the response-dependent belief updates used by our planner in Section 5.

## 5 PROSE-PLAN: Evaluability-Aware Proposal Planning

The ProSE problem in Section 3 is a Bayes-adaptive belief-state planning problem which, in principle, can be solved optimally with standard RL solutions once the response likelihood, transition model, and value family are specified. Our goal in this section is not to contribute a novel general solution, but to introduce a minimal planner, PROSE-PLAN, to analyse solutions to our problem setting.

Belief Tracking We track the posterior belief over the latent user parameters $p ( z )$ with $z = ( \phi , \rho , \kappa )$ according to Eq. 5. For tractability, we represent z on a finite grid and maintain the posterior exactly on that grid. More sophisticated Bayesian approaches are available for large parameter spaces.

Proposal Selection We propose a depth-2 look-ahead search which scores each proposal by the 2-step Q-value $Q _ { h }$ (recall $\operatorname { E q . } \left( 2 \right)$ in Section 3). This means, for every possible proposal $\tilde { s } \in \operatorname { \mathcal { C } } ( s )$ we compute the next state likelihood given our belief of the user accepting it:

$$
p ( s ^ { \prime } \mid b , s , \tilde { s } ) = \sum _ { y } \mathbb { E } _ { z \sim b } \big [ \mathcal { T } ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , y ) P ( y \mid s , \tilde { s } , z ) \big ]\tag{13}
$$

Then, for each of those hypothetical state transitions, the corresponding new hypothetical belief $b _ { s ^ { \prime } }$ is computed (Alg 1 line 5 in Appendix D), and this response-dependent beliefupdate is repeated for a second step to compute the likelihood for all hypothetical future trajectories of depth 2. Then, the Q-value of each node in the graph is computed, where the value of the last step is the expectation over the user preferences: $\breve { Q _ { 0 } } ( \dot { b } , s ) = \mathbb { E } _ { \phi \sim b } ^ { \cdot } \big \lceil V _ { \phi } ( s ) \big \rceil$ (recall Eq. 2). Ultimately, the proposal with the highest expected value $Q _ { 2 }$ is chosen. Deeper lookahead and Monte Carlo tree search [25] are promising improvements, but this solution is sufficient for our purposes.

Under the binary deterministic transition in Section 4.1, the expectations over future states collapse to the accepted or rejected next state. With maximum candidate-set size $| { \mathcal { C } } |$ , response-space size $| y |$ and grid size $| \mathcal { Z } _ { \mathrm { g r i d } } |$ |, exhaustive depth-2 evaluation costs $O ( | \mathcal { C } | ^ { 2 } | \mathcal { V } | ^ { 2 } | \mathcal { Z } _ { \mathrm { g r i d } } ^ { ' } | )$ per planning step.

## 6 Experiments

In this section, we use controlled graph-based simulations to empirically test whether the response model from Section 4 has planning consequences in finite-horizon sequential tasks. Specifically, we ask the following two research questions:

Q1 End-to-end payoff. Does evaluability-aware planning improve final outcomes over methods that ignore evaluability, user-specific evaluability, or non-myopic proposal planning?

![](images/f4830e99fb10190c8a4d3695fb40260b7c90de1839ffbbdd383dbefbe53241a8.jpg)  
Figure 2: Illustrations for experiment settings.

Q2 Mechanism ablation. Does the gain of $\mathrm { P R O S E - P L A N }$ come from depth-2 lookahead alone, or from response-dependent belief updates inside the lookahead tree?

We evaluate Q1 in a branching-corridor task and Q2 in a minimal probe-commit task, and leave the main sensitivity analysis to Appendix E.

## 6.1 Experimental Setup

We evaluate PROSE-PLAN in controlled finite-graph proposal tasks. In each task, states are graph nodes, the assistant proposes a candidate node s˜, and evaluation difficulty is represented by $d ( s , { \tilde { s } } )$ the unweighted shortest-path distance between two nodes. User responses follow the bounded-rational response model in Eq. (9), accepting a proposal moves the state to s˜, while rejecting it leaves the state unchanged. The task terminates if the user reaches the preferred goal or the horizon is reached. The latent user parameter $z = ( \phi , \rho , \kappa )$ determines the user’s preferred goal, evaluability cost, and response sharpness. To avoid confounding the evaluation of planning with approximate inference errors, the assistant maintains an exact discrete posterior over z. We report success rate, defined as the fraction of episodes ending at the user’s preferred goal state, and mean terminal user value $V _ { \phi } ( s _ { T } )$ , averaged over 200 random seeds per condition.

We consider the following baselines and ablation. RANDOM selects proposal uniformly. VALUE-GREEDY ignores evaluability and selects the proposal with the highest value. Inspired by the satisficing alignment principle [26], THRESHOLD greedily proposes from a candidate set constrained by a distance threshold. Both POPULATION-MYOPIC and PERSONALISED-MYOPIC are myopic planners, while the former uses population-level evaluability parameters and the latter infers the full latent user state. BELIEF-FROZEN DEPTH-2 is a mechanism ablation, which has the same depth-2 horizon as PROSE-PLAN, but evaluates future proposals under the current belief rather than the posterior induced by hypothetical first-step responses. ORACLE DEPTH-2 knows the true user parameter and serves as an upper-bound reference. All methods infer ϕ except RANDOM and ORACLE DEPTH-2. Full descriptions are provided in Appendix E.1.

## 6.2 End-to-End Payoff in a Branching Corridor

We construct a branching-corridor task to test whether evaluability-aware planning improves endto-end outcomes when the assistant must trade off easy generic progress against harder but more personalised proposals. The branching corridor (Figure 2a) is a parametric graph-based sequential task family, where the state space is an undirected tree with a start node $s _ { 0 } ,$ , a shared corridor $\{ c _ { 1 } , \ldots , c _ { C } \}$ and $\check { K }$ branches of length $L$ attached at $c _ { C } { \mathrm { . } }$ , with branch nodes $\{ b _ { k , 1 } , \dotsc , b _ { k , L } \}$ for $k \in \{ 1 , \ldots , K \}$ The user has a hidden preferred branch $\phi \in \{ 1 , \ldots , K \}$ with goal leaf $b _ { \phi , L }$ . The user’s value function family combines shared corridor and branch progress: $V _ { \phi } ( c _ { j } ) = \alpha _ { \mathrm { e n v } } w _ { c } j , V _ { \phi } ( b _ { \phi , j } ) =$ $\alpha _ { \mathrm { e n v } } w _ { c } C + w _ { b } j , V _ { \phi } ( b _ { k , j } ) = \alpha _ { \mathrm { e n v } } w _ { c } C - w _ { p } j$ for $k \neq \phi .$ , where $\bar { V _ { \phi } ( s _ { 0 } ) } = 0 , \alpha _ { \mathrm { e n v } } \in [ 0 , 1 ]$ is a task-structure parameter controlling the relative value between common progress and personalised commitment. At each step $t = 0 , \ldots , T - 1$ , the assistant proposes a node $\tilde { s } _ { t } \neq s _ { t } ,$ , the user responds with $y _ { t } \in \{ \mathrm { a c c } , \mathrm { r e j } \}$ via Eq. (9) under latent ${ z } = ( \phi , \rho , \kappa )$ , and the state moves to $\tilde { s } _ { t } \mathrm { i f } y _ { t } = \mathrm { a c c }$ and stays at $s _ { t }$ otherwise. We define success in this case as reaching $b _ { \phi , L }$ within the horizon T. In experiments, we use $K = 4 , C = 2 , L = 4 , T = 5 , ( w _ { c } , w _ { b } , w _ { p } ) = ( 3 , 2 , 3 )$ as defaults, consider a quadratic transform $g ( d ) = d ^ { 2 }$ , and ${ \mathcal { C } } ( s ) = S \setminus \{ s \}$ . More details are provided in Appendix E.2.1.

![](images/6ef323825162bee1f715e14b5078f1de8fc6f2d45942a5af5373e09a3f9fea6d.jpg)  
(a) Success vs. ρ<sub>true</sub>.

![](images/c9426de159555e961b1cdc0edb57116c579579f442739795399a613db003bf9e.jpg)  
(b) Gain vs. $\alpha _ { \mathrm { e n v } } .$  
Figure 3: Branching-corridor results, mean ± standard error over 200 seeds. (a) Success rate versus true evaluability cost at $\alpha _ { \mathrm { e n v } } = 0 . 2 5$ . (b) Success-rate gain of PROSE-PLAN over the stronger myopic baseline. Gains appear only when the evaluation cost is high and shared-corridor value is low.

Main results (Q1). Figure 3(a) shows that PROSE-PLAN improves end-to-end success when proposal evaluability becomes the bottleneck. At low $\rho _ { \mathrm { t r u e } } ,$ distant proposals are easy to evaluate, so the strongest baselines, especially POPULATION-MYOPIC and PERSONALISED-MYOPIC, achieve nearceiling success. $\mathrm { A s } \ \rho _ { \mathrm { t r u e } }$ increases, these baselines collapse because they continue to propose distant branch states whose value is high under some preferences, but whose acceptance probability becomes low under high evaluation cost. In contrast, PROSE-PLAN degrades more gracefully and becomes the best non-oracle method in the high-cost setting by selecting closer, more evaluable proposals. Table 1 gives the default high-cost comparison at $\alpha _ { \mathrm { e n v } } = 0 . 2 5 , \rho _ { \mathrm { t r u e } } = 0 . 3 0 \mathrm { { z } }$ : PROSE-PLAN reaches success 0.515, more than twice PERSONALISED-MYOPIC (0.215) and POPULATION-MYOPIC (0.200). Aggregated proposal behaviour and representative trajectories in Appendix E.2.3 and E.2.4 show the corresponding behavioural pattern. Thus, the branching-corridor result supports Q1: evaluabilityaware proposal planning improves final outcomes when high-value personalised proposals are difficult to evaluate.

Sensitivity to task structure. Figure 3(b) shows that the gain of PROSE-PLAN depends on the task structure. At high evaluation cost $( \rho _ { \mathrm { t r u e } } = 0 . 3 0 )$ , PROSE-PLAN improves over the stronger myopic baseline when $\alpha _ { \mathrm { e n v } }$ is small, with the largest gain at $\alpha _ { \mathrm { e n v } } = 0 . 2 5 . \ \mathrm { A s } \ \alpha _ { \mathrm { e n v } }$ increases, shared-corridor states already provide enough value, so simple myopic baselines become competitive. At lower evaluation cost $( \rho _ { \mathrm { t r u e } } = 0 . 1 8 )$ , the gain is consistently non-positive because distant proposals remain sufficiently evaluable. Thus, evaluability-aware planning helps most when generic local suggestions are easy to accept but insufficient for reaching the user’s intended outcome, while personalised high-value proposals are costly to evaluate.

## 6.3 Ablation: Response-dependent Lookahead

We construct a two-step probe-commit task (Figure 2b) to isolate whether the advantage of PROSE-PLAN comes from lookahead alone or from conditioning the second proposal on the belief induced by a possible first response. The state space contains a start node $s _ { 0 }$ , two probe nodes $\{ p _ { 1 } , p _ { 2 } \}$ and two goal nodes $\left\{ g _ { 1 } , g _ { 2 } \right\}$ , where $p _ { k }$ lies on the path from $s _ { 0 }$ to $g _ { k }$ . The user has a hidden preference $\phi \in \{ 1 , 2 \}$ , with preferred goal $g _ { \phi }$ . As in the corridor task, acceptance moves the state to the proposal, and rejection leaves the state unchanged, and we set the horizon to $T = 2$ We design the user value function such that, for the preferred probe and goal node, the value is $w _ { p , + }$ and $w _ { g , + }$ respectively, otherwise, the value is $w _ { p , }$ <sub>−</sub> and $w _ { g , }$ <sub>−</sub> respectively. By default we use $( \bar { w } _ { p , + } , w _ { p , - } , w _ { g , + } , w _ { g , - } ) = ( 1 , - 3 , 5 , 4 )$ ). Achieving a higher success rate in this design essentially requires the assistant to learn about ϕ: goal proposals have high value under both preferences, so a direct commit is immediately attractive but not informative of ϕ, while probe proposals have low immediate value, but their responses are more informative because a probe on the preferred branch is likely to be accepted, while a probe on the wrong branch is not. Thus, probing can improve terminal success only if the planner anticipates that the first response will update the posterior and then uses that posterior to choose the second proposal. A myopic planner has no incentive to make this sacrifice, and a belief-frozen depth-2 planner has the same horizon as $\mathrm { P R O S E - P L A N }$ but evaluates second-step continuations under the current belief rather than the posterior induced by hypothetical first-step responses. This makes the task a direct test of response-dependent lookahead.

Table 1: Results of the Branching Corridor and Probe-Commit tasks. Branching-Corridor uses $\alpha _ { \mathrm { e n v } } ~ = ~ 0 . 2 5$ and $\rho _ { \mathrm { t r u e } } ~ = ~ 0 . 3 0 ;$ ; Probe-Commit uses w<sub>probe,mismatch</sub> $= - 3 ,$ $\rho _ { \mathrm { t r u e } } ~ = ~ 0 . 5$ , and $\kappa _ { \mathrm { t r u e } } = 1 . 0$ . Values are reported as mean ± standard error over 200 seeds.
<table><tr><td rowspan="2">Method</td><td colspan="2">Branching Corridor</td><td colspan="3">Probe-Commit</td></tr><tr><td> $V _ { \phi } ( s _ { T } )$ </td><td>Success rate</td><td> $V _ { \phi } ( s _ { T } )$ </td><td>Success rate</td><td> $P ( { \mathrm { p r o b e @ 0 } } )$ </td></tr><tr><td>Random</td><td> $3 . 5 1 \pm 0 . 2 4$ </td><td> $0 . 0 9 5 \pm 0 . 0 2 1$ </td><td> $3 . 5 0 \pm 0 . 1 4$ </td><td> $0 . 4 2 5 \pm 0 . 0 3 5$ </td><td> $0 . 4 6 0 \pm 0 . 0 3 5$ </td></tr><tr><td>Value-greedy</td><td> $1 . 2 9 \pm 0 . 0 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td></td><td></td></tr><tr><td>Threshold</td><td> $1 . 2 9 \pm 0 . 0 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td></td><td></td><td></td></tr><tr><td>Population-myopic</td><td> $1 . 9 0 \pm 0 . 2 7$ </td><td> $0 . 2 0 0 \pm 0 . 0 2 8$ </td><td></td><td></td><td></td></tr><tr><td>Personalised-myopic</td><td> $2 . 8 4 \pm 0 . 2 8$ </td><td> $0 . 2 1 5 \pm 0 . 0 2 9$ </td><td> $4 . 5 3 \pm 0 . 0 4$ </td><td> $0 . 5 5 0 \pm 0 . 0 3 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Belief-frozen depth-2</td><td></td><td></td><td> $4 . 5 4 \pm 0 . 0 4$ </td><td> $0 . 5 5 5 \pm 0 . 0 3 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>PROSE-PLAN</td><td> ${ \bf 5 . 6 2 \pm 0 . 2 9 }$ </td><td> ${ \bf 0 . 5 1 5 \pm 0 . 0 3 5 }$ </td><td> ${ \bf 4 . 7 3 \pm 0 . 0 6 }$ </td><td> ${ \bf 0 . 8 1 5 \pm 0 . 0 2 8 }$ </td><td> $\mathbf { 1 . 0 0 0 \mathop { = } 0 . 0 0 0 }$ </td></tr><tr><td>Oracle depth-2</td><td> $9 . 4 1 \pm 0 . 0 7$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 7$ </td><td> $4 . 9 8 \pm 0 . 0 3$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

Main results (Q2). Table 1 shows that PROSE-PLAN reaches 0.815 success, compared with 0.550 for PERSONALISED-MYOPIC and 0.555 for BELIEF-FROZEN DEPTH-2. The behavioural signature is sharper: PROSE-PLAN probes in every episode, while both baselines directly commit to high-value goals. The probe response updates the posterior over ϕ, making the second proposal more targeted and improving success by 26 percentage points. Thus, the mechanism is not lookahead alone, but planning over how user responses update later proposal decisions.

## 7 Discussion and Conclusion

This work identifies evaluability as a relevant dimension of proposal-based AI assistance: a good proposal not only has high value under the user’s preferences but is also evaluable, and rejection may not mean the proposal is poor but can be information about the user’s evaluation capabilities. ProSE formalises this by treating each proposal as both a solution and an observation about the user. Our bounded-rational response model makes this coupling explicit, and analysis of its acceptance frontier shows a concrete consequence: proposals that are likely to be accepted and those informative about evaluability need not coincide. In particular, the most informative proposals can lie on the expected-rejection side of the acceptance frontier. Our controlled experiments also empirically support our central claim that user evaluability is complementary to preference inference and proposal quality. Alongside these contributions, we observe notable avenues for future work:

First, our main instantiation uses a deliberately minimal binary accept/reject model, and it isolates the proposal-evaluation mechanism and matches common interfaces such as accepting or dismissing a patch, edit, or revision. However, ProSE itself is formally more general, and generalising to richer response spaces is important for more sophisticated applications. This is possible in principle by generalising the sigmoid (binary) with a softmax (categorical decision), but future work must investigate the implications of this. Second, the proposed evaluability metric captures the idea that larger or more distant changes are harder to inspect, but the right choice is most likely task-dependent. In coding, for example, evaluability may depend on dependency structure or locality. Learning task-specific evaluability metrics is a key next step. Lastly, the scalability of PROSE-PLAN is limited in that it is a minimal depth-2 Bayes-adaptive planner with exact grid inference. This keeps the mechanism transparent and avoids confounding planning with approximate inference, but it does not directly scale to open-ended design spaces. Real-world applications will require candidate generation, approximate Bayesian inference, and scalable tree search or amortized planning built upon our work.

## Acknowledgments and Disclosure of Funding

This work was supported by the Research Council of Finland (Flagship programme: Finnish Center for Artificial Intelligence FCAI, Grant 359207), ELISE Networks of Excellence Centres (EU Horizon:2020 grant agreement 951847), and UKRI Turing AI World-Leading Researcher Fellowship (EP/W002973/1). We acknowledge the research environment provided by ELLIS Institute Finland. We also acknowledge the computational resources provided by the Aalto Science-IT Project from Computer Science IT and CSC–IT Center for Science, Finland.

## References

[1] Alan Fern, Sriraam Natarajan, Kshitij Judah, and Prasad Tadepalli. A decision-theoretic model of assistance. Journal ofArtificial Intelligence Research, 50:71–104, 2014.

[2] Dylan Hadfield-Menell, Stuart J Russell, Pieter Abbeel, and Anca Dragan. Cooperative inverse reinforcement learning. Advances in neural information processing systems, 29, 2016.

[3] Mustafa Mert Çelikok, Frans A. Oliehoek, and Samuel Kaski. Best-response bayesian reinforcement learning with bayes-adaptive pomdps for centaurs. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’22, page 235–243, Richland, SC, 2022. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 9781450392136.

[4] Sebastiaan De Peuter and Samuel Kaski. Zero-shot assistance in sequential decision problems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11551– 11559, 2023.

[5] Cassidy Laidlaw, Eli Bronstein, Timothy Guo, Dylan Feng, Lukas Berglund, Justin Svegliato, Stuart Russell, and Anca Dragan. Assistancezero: Scalably solving assistance games. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=b9hVMJi0t2.

[6] Herbert A. Simon. A behavioral model of rational choice. Quarterly Journal of Economics, 69: 99–118, 1955. URL https://api.semanticscholar.org/CorpusID:18410595.

[7] Herbert A. Simon. Rational choice and the structure of the environment. Psychological review, 63 2:129–38, 1956. URL https://api.semanticscholar.org/CorpusID:8503301.

[8] Richard L Lewis, Andrew Howes, and Satinder Singh. Computational rationality: Linking mechanism and behavior through bounded utility maximization. Topics in cognitive science, 6 (2):279–311, 2014.

[9] Antti Oulasvirta, Jussi PP Jokinen, and Andrew Howes. Computational rationality as a theory of interaction. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pages 1–14, 2022.

[10] Andrew Howes, Jussi PP Jokinen, and Antti Oulasvirta. Towards machines that understand people. AI Magazine, 44(3):312–327, 2023.

[11] Sarah Fakhoury, Aaditya Naik, Georgios Sakkas, Saikat Chakraborty, and Shuvendu K Lahiri. Llm-based test-driven interactive code generation: User study and empirical evaluation. IEEE Transactions on Software Engineering, 50(9):2254–2268, 2024.

[12] Adam Alami and Neil Ernst. Human and machine: How software engineers perceive and engage with ai-assisted code reviews compared to their peers. In 2025 IEEE/ACM 18th International Conference on Cooperative and Human Aspects of Software Engineering (CHASE), pages 63–74. IEEE, 2025.

[13] Erdem Biyik, Malayandi Palan, Nicholas C. Landolfi, Dylan P. Losey, and Dorsa Sadigh. Asking easy questions: A user-friendly approach to active reward learning. ArXiv, abs/1910.04365, 2019. URL https://api.semanticscholar.org/CorpusID:204008187.

[14] Xuening Feng, Zhaohui Jiang, Timo Kaufmann, Eyke Hüllermeier, Paul Weng, and Yifei Zhu. Comparing comparisons: informative and easy human feedback with distinguishability queries. In Proceedings ofthe 42nd International Conference on Machine Learning, ICML’25. JMLR.org, 2025.

[15] Hussein Mozannar, Gagan Bansal, Adam Fourney, and Eric Horvitz. When to show a suggestion? integrating human feedback in ai-assisted programming. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 10137–10144, 2024.

[16] Pedro A Ortega and Daniel A Braun. Thermodynamics as a theory of decision-making with information-processing costs. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 469(2153), 2013.

[17] Pedro A Ortega, Daniel A Braun, Justin Dyer, Kee-Eung Kim, and Naftali Tishby. Informationtheoretic bounded rationality. arXiv preprint arXiv:1512.06789, 2015.

[18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. A Bradford Book, Cambridge, MA, USA, 2018. ISBN 0262039249.

[19] Michael O’Gordon Duff. Optimal Learning: Computational proceduresfor Bayes-adaptive Markov decision processes. University of Massachusetts Amherst, 2002.

[20] Samuel J Gershman, Eric J Horvitz, and Joshua B Tenenbaum. Computational rationality: A converging paradigm for intelligence in brains, minds, and machines. Science, 349(6245): 273–278, 2015.

[21] William Samuelson and Richard Zeckhauser. Status quo bias in decision making. Journal of risk and uncertainty, 1(1):7–59, 1988.

[22] John Sweller. Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2):257–285, 1988. doi: https://doi.org/10.1207/s15516709cog1202\_4. URL https:// onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1202\_4.

[23] Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. Statistical Science, 10:273–304, 1995. URL https://api.semanticscholar.org/CorpusID: 13676847.

[24] David J. C. MacKay. Information-based objective functions for active data selection. Neural Computation, 4(4):590–604, 1992. doi: 10.1162/neco.1992.4.4.590.

[25] David Silver and Joel Veness. Monte-carlo planning in large pomdps. Advances in neural information processing systems, 23, 2010.

[26] Mohamad Chehade, Soumya Suvra Ghosal, Souradip Chakraborty, Avinash Reddy, Dinesh Manocha, Hao Zhu, and Amrit Singh Bedi. Bounded rationality for llms: satisficing alignment at inference-time. In Proceedings ofthe 42nd International Conference on Machine Learning, ICML’25. JMLR.org, 2025.

[27] Siddharth Reddy, Anca D Dragan, and Sergey Levine. Shared autonomy via deep reinforcement learning. arXiv preprint arXiv:1802.01744, 2018.

[28] Scott Emmons, Caspar Oesterheld, Vincent Conitzer, and Stuart Russell. Observation interference in partially observable assistance games. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=rjZ2SWjwwB.

[29] Ali Khoshvishkaie, Petrus Mikkola, Pierre-Alexandre Murena, and Samuel Kaski. Cooperative bayesian optimization for imperfect agents. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 475–490. Springer, 2023.

[30] Frederick Callaway, Mathew Hardy, and Thomas L Griffiths. Optimal nudging for cognitively bounded agents: A framework for modeling, predicting, and controlling the effects of choice architectures. Psychological Review, 130(6):1457, 2023.

[31] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

[32] Julien Gori, Aurelien Nioche, Christoph A. Johns, and Antti Oulasvirta. A decision-theoretic representation of assistive interfaces. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems, CHI ’26, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400722783. doi: 10.1145/3772318.3791819. URL https://doi.org/ 10.1145/3772318.3791819.

[33] Mustafa Mert Çelikok, Pierre-Alexandre Murena, and Samuel Kaski. Modeling needs user modeling. Frontiers in Artificial Intelligence, 6:1097891, 2023.

[34] Falk Lieder and Thomas L Griffiths. Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources. Behavioral and brain sciences, 43:e1, 2020.

[35] Yifan Zhu, Sammie Katt, and Samuel Kaski. More than irrational: Modeling belief-biased agents. Proceedings ofthe AAAI Conference on Artificial Intelligence, 40(35):29948–29956, Mar. 2026. doi: 10.1609/aaai.v40i35.40242. URL https://ojs.aaai.org/index.php/ AAAI/article/view/40242.

[36] Tan Zhi-Xuan, Jordyn Mann, Tom Silver, Josh Tenenbaum, and Vikash Mansinghka. Online bayesian goal inference for boundedly rational planning agents. Advances in neural information processing systems, 33:19238–19250, 2020.

[37] Arwa Alanqary, Gloria Z Lin, Joie Le, Tan Zhi-Xuan, Vikash K Mansinghka, and Joshua B Tenenbaum. Modeling the mistakes of boundedly rational agents within a bayesian theory of mind. arXiv preprint arXiv:2106.13249, 2021.

[38] Athul Paul Jacob, Abhishek Gupta, and Jacob Andreas. Modeling boundedly rational agents with latent inference budgets. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=W3VsHuga3j.

[39] Owain Evans, Andreas Stuhlmüller, and Noah Goodman. Learning the preferences of ignorant, inconsistent agents. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 30, 2016.

[40] Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, and Sanjit A. Seshia. Active preferencebased learning of reward functions. In Robotics: Science and Systems, 2017. URL https: //api.semanticscholar.org/CorpusID:12226563.

[41] Belinda Z Li, Alex Tamkin, Noah Goodman, and Jacob Andreas. Eliciting human preferences with language models. arXiv preprint arXiv:2310.11589, 2023.

[42] Sebastiaan De Peuter, Shibei Zhu, Yujia Guo, Andrew Howes, and Samuel Kaski. Preference learning of latent decision utilities with a human-like model of preferential choice. Advances in Neural Information Processing Systems, 37:123608–123636, 2024.

[43] Mark Steyvers and Aakriti Kumar. Three challenges for ai-assisted decision-making. Perspectives on Psychological Science, 19(5):722–734, 2024.

[44] Shraddha Barke, Michael B James, and Nadia Polikarpova. Grounded copilot: How programmers interact with code-generating models. Proceedings of the ACM on Programming Languages, 7(OOPSLA1):85–111, 2023.

[45] Jenny T Liang, Chenyang Yang, and Brad A Myers. A large-scale survey on the usability of ai programming assistants: Successes and challenges. In Proceedings ofthe 46th IEEE/ACM international conference on software engineering, pages 1–13, 2024.

[46] Renze Lou, Hanzi Xu, Sijia Wang, Jiangshu Du, Ryo Kamoi, Xiaoxin Lu, Jian Xie, Yuxuan Sun, Yusen Zhang, Jihyun Janice Ahn, Hongchao Fang, Zhuoyang Zou, Wenchao Ma, Xi Li, Kai Zhang, Congying Xia, Lifu Huang, and Wenpeng Yin. Aaar-1.0: assessing ai’s potential to assist research. In Proceedings of the 42nd International Conference on Machine Learning, ICML’25. JMLR.org, 2025.

[47] Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, and Aviv Tamar. Bayesian reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 8(5-6):359–483, 2015.

[48] Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

[49] Stéphane Ross, Joelle Pineau, Brahim Chaib-Draa, and Pierre Kreitmann. A bayesian approach for learning and planning in partially observable markov decision processes. Journal ofMachine Learning Research, 12(5), 2011.

## A Related Work

In this section, we discuss the positioning, novelty, as well as significance of our work with related work from sequential AI assistance and assistance games, bounded and computational rationality, learning preferences and rewards from human feedback, and empirical studies of AI-assisted decisionmaking. Table 2 summarises how representative methods relate to our method. The main takeaway from our work is that the user’s response to a proposal depends on what the user can reliably evaluate from the current state, and that this evaluability is itself a latent quantity the assistant should infer and plan with.

Table 2: Where our work sits relative to representative prior work. Seq. is short for sequential assistance; Control means the human remains in control of the task; Pref. is short for latent preference or reward inference; Bounded is for bounded-rational or cognitively constrained user model; Eval. cost is short for the response or feedback model includes a proposal evaluation cost; Info plan represents planning uses the information value of the user response; Eval.-aware is for proposal selection itself accounts for evaluability. A dash indicates Not Applicable.
<table><tr><td>Method</td><td>Seq.</td><td>Control</td><td>Pref.</td><td>Bounded</td><td>Eval. cost</td><td>Info plan</td><td>Eval.-aware</td></tr><tr><td>Decision-theoretic assistance [1]</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>CIRL [2]</td><td>√</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Shared autonomy [27]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Centaur [3]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>X</td></tr><tr><td>Zero-shot assistance [4]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>AssistanceZero [5]</td><td>√</td><td>x</td><td>√</td><td>X</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Observation interference [28]</td><td>√</td><td>x</td><td>√</td><td>X</td><td>x</td><td>√</td><td>X</td></tr><tr><td>Cooperative BO [29]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Optimal nudging [30]</td><td>X</td><td>√</td><td>X</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>RLHF [31]</td><td>x</td><td></td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Easy queries [13]</td><td>√</td><td></td><td>√</td><td>X</td><td>√</td><td>√</td><td>V</td></tr><tr><td>Distinguishability queries [14]</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>When-to-show [15]</td><td>√</td><td>√</td><td>x</td><td>x</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Our work</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Sequential AI assistance and assistance games. A standard line of work treats AI assistance as a multi-agent decision process with hidden user goals or preferences. Decision-theoretic assistance [1] and cooperative inverse reinforcement learning [2] establish the basic problem of acting under uncertainty about the user’s reward. Subsequent work extends the setting to shared autonomy with deep reinforcement learning [27], advisory assistance for bounded-rational users [3, 4], large-scale assistance games solved with deep RL [5], partially observable assistance games where assistant actions interfere with what is learned about the human [28], and cooperative Bayesian optimisation with imperfect agents [29]. Gori et al. [32] provides a unifying decision-theoretic vocabulary for assistive interfaces, and Çelikok et al. [33] argues that user modelling is itself a load-bearing modelling task. ProSE inherits the Bayes-adaptive structure of this line and adds proposal-dependent evaluability as a latent factor that simultaneously shapes adoption and the informativeness of the user’s response. The planner therefore has to reason about both within the same belief update, which earlier formulations do not.

Bounded and computational rationality. Bounded rationality [6, 7] models human behaviour as approximate optimisation under cognitive constraints. The computational and resource-rational theoretical frameworks refine this view by deriving behaviour from rational use of limited resources [8, 20, 34, 9, 10]. Information-theoretic bounded rationality [16, 17] formalises bounded choice as a KL-regularised problem and provides the response-model backbone we instantiate in Section 4. Other work targets specific bounds, such as memory [35], planning depth [36–38], or one-shot choice architectures for bounded agents [30]. These models are mostly used for behavioural prediction, generation, or for one-shot intervention design as user behaviour likelihood. We embed the same family of models inside a sequential assistance loop, where the assistant’s proposal is the intervention and the user’s bounded response is the observation that drives belief updates. This turns evaluation cost from a descriptive concept into a planning quantity.

Learning preferences and rewards from human feedback. Reward learning, RLHF, and active preference learning aim to recover the user’s latent objective from feedback [2, 31, 39, 40, 13, 41, 42]. Evans et al. [39] consider ignorant or inconsistent agents, Biyik et al. [13] and Feng et al. [14] explicitly select queries that are easier or more distinguishable for the user, and De Peuter et al. [42] models preferential choice with a human-like computational rationality choice model. These methods focus on the inference goal and primarily treat human feedback as a query channel that does not have an impact on the transition dynamics of the task. In ProSE, the proposal is both a candidate next state of the task and a probe of the user’s latent parameters, and its evaluability shapes the response distribution. This coupling of inference and control is what differentiates evaluability-aware proposal planning from active reward learning.

Empirical evidence on user evaluation cost. Empirical work on AI-assisted decision-making and AI-assisted programming consistently reports that users struggle with proposals that are too large or too hard to verify [43, 11, 12, 44, 45]. Mozannar et al. [15] model when an AI programming assistant should show a suggestion, treating the display itself as a decision under uncertainty about user reaction. Benchmark work on AI for research tasks finds that novel experiment proposals from LLMs often lack feasibility, making it difficult for researchers to evaluate their necessity [46]. These findings motivate the central modelling choice in this paper, that proposal complexity acts as an evaluation cost in the user’s response, and that an assistant ignoring this tends to produce proposals the user cannot meaningfully act upon.

Bayes-adaptive planning under hidden user parameters. Methodologically, PROSE-PLAN is a Bayes-adaptive planner [19, 47] for an assistance setting with a non-trivial response likelihood. Bayes-adaptive MDPs and partially observable MDPs are standard frameworks for online inference and planning under unknown dynamics [48, 49, 25], and depth-limited Bayes-adaptive lookahead has been used in human-AI assistance for tractability [3, 4]. Our depth-2 planner stays in this style but uses the response model in two roles within the same lookahead. First, the response model scores each candidate proposal under the current belief. Second, it updates the belief hypothetically over the user’s latent parameters before scoring the follow-up proposal, so that proposals are evaluated on how their possible responses would change subsequent decisions. This is what makes the planner sensitive to evaluability rather than only to expected acceptance.

## B Details of Evaluability-Aware Response Model

## B.1 Derivation of the Gibbs policy

We derive the closed-form solution of the information-theoretic bounded-rational decision problem in Eq. (3). Let A be a finite action set, let $q \in \Delta ( \mathcal { A } )$ be a full-support prior policy, and let $\bar { U } : A $ R be a utility function. For $\kappa > 0$ , define

$$
J ( p ) = \sum _ { a \in \mathcal { A } } p ( a ) U ( a ) - \frac { 1 } { \kappa } \sum _ { a \in \mathcal { A } } p ( a ) \log \frac { p ( a ) } { q ( a ) } .\tag{14}
$$

Since $q ( a ) > 0$ for all $a \in { \mathcal { A } } ,$ , the KL term is finite for all $p \in \Delta ( \mathcal { A } )$ . Moreover, $D _ { \mathrm { K L } } ( p \Vert q )$ is strictly convex in $p ,$ so $J ( p )$ is strictly concave on the simplex for $\kappa > 0$ . Therefore, the maximiser is unique.

To solve for it, introduce a Lagrange multiplier λ for the simplex constraint $\textstyle \sum _ { a \in { \mathcal { A } } } p ( a ) = 1$

$$
{ \mathcal { L } } ( p , \lambda ) = \sum _ { a \in { \mathcal { A } } } p ( a ) U ( a ) - { \frac { 1 } { \kappa } } \sum _ { a \in { \mathcal { A } } } p ( a ) \log { \frac { p ( a ) } { q ( a ) } } - \lambda ( \sum _ { a \in { \mathcal { A } } } p ( a ) - 1 ) .\tag{15}
$$

For each $a \in { \mathcal { A } }$ , the first-order condition is

$$
{ \frac { \partial { \mathcal { L } } } { \partial p ( a ) } } = U ( a ) - { \frac { 1 } { \kappa } } ( \log { \frac { p ( a ) } { q ( a ) } } + 1 ) - \lambda = 0 .\tag{16}
$$

Rearranging gives

$$
\log \frac { p ( a ) } { q ( a ) } = \kappa U ( a ) - \kappa \lambda - 1 .\tag{17}
$$

Exponentiating both sides,

$$
p ( a ) = q ( a ) \exp ( \kappa U ( a ) - \kappa \lambda - 1 ) = q ( a ) \exp ( \kappa U ( a ) ) \exp ( - \kappa \lambda - 1 ) .\tag{18}
$$

The factor $\exp ( - \kappa \lambda - 1 )$ does not depend on a, so it is fixed by normalisation. Enforcing $\textstyle \sum _ { a \in { \mathcal { A } } } p ( a ) = { \bar { 1 } }$ yields

$$
Z _ { \kappa } = \sum _ { a ^ { \prime } \in \mathcal { A } } q ( a ^ { \prime } ) \exp ( \kappa U ( a ^ { \prime } ) ) ,\tag{19}
$$

and therefore

$$
p ^ { \mathrm { I T B R } } ( a ) = \frac { q ( a ) \exp ( \kappa U ( a ) ) } { Z _ { \kappa } } .\tag{20}
$$

This is the Gibbs policy used in Eq. (4). It can be read as the prior policy q tilted toward higher-utility actions by the factor $\exp ( \kappa U ( a ) )$ ).

Finally, the default-policy limit follows directly from the same expression:

$$
\operatorname* { l i m } _ { \kappa \to 0 ^ { + } } p ^ { \mathrm { I T B R } } ( a ) = \frac { q ( a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } q ( a ^ { \prime } ) } = q ( a ) .\tag{21}
$$

Thus, when the information-processing parameter vanishes, the agent does not move away from the prior policy.

## B.2 Reduction to a binary sigmoid response

We now specialise the general Gibbs policy to the binary response space used in this work $\mathcal { V } =$ {acc, rej}. Suppressing the conditioning variables for readability, the acceptance probability is

$$
p ^ { \star } ( \mathrm { a c c } ) = \frac { q ( \mathrm { a c c } ) \exp ( \kappa U ( \mathrm { a c c } ) ) } { q ( \mathrm { a c c } ) \exp ( \kappa U ( \mathrm { a c c } ) ) + q ( \mathrm { r e j } ) \exp ( \kappa U ( \mathrm { r e j } ) ) } .\tag{22}
$$

Dividing the Gibbs weights for acceptance and rejection yields

$$
\frac { p ^ { \star } ( \mathrm { a c c } ) } { p ^ { \star } ( \mathrm { r e j } ) } = \frac { q ( \mathrm { a c c } ) } { q ( \mathrm { r e j } ) } \exp ( \kappa [ U ( \mathrm { a c c } ) - U ( \mathrm { r e j } ) ] ) .\tag{23}
$$

Let $\begin{array} { r } { \Delta U = U ( \mathrm { a c c } ) - U ( \mathrm { r e j } ) \mathrm { a n d } \lambda = \log \frac { q ( \mathrm { a c c } ) } { q ( \mathrm { r e j } ) } } \end{array}$ , we thus obtain Eq. (6):

$$
p ^ { \star } ( \mathrm { a c c } ) = \sigma ( \kappa \Delta U + \lambda ) .\tag{24}
$$

## B.3 Response utilities

Fix a current state s, proposal ${ \tilde { s } } ,$ and preference parameter $\phi .$ For a general response $y \in \mathcal { V }$ , define response utility as the expected value of the realised next state:

$$
U ( y ; s , \tilde { s } , \phi ) = \mathbb { E } _ { s ^ { \prime } \sim T ^ { y } ( \cdot \vert s , \tilde { s } , y ) } [ V _ { \phi } ( s ^ { \prime } ) ] .\tag{25}
$$

In the binary deterministic instantiation,

$$
T ^ { y } ( s ^ { \prime } \mid s , { \tilde { s } } , \operatorname { a c c } ) = \mathbf { 1 } \{ s ^ { \prime } = { \tilde { s } } \} , \qquad T ^ { y } ( s ^ { \prime } \mid s , { \tilde { s } } , \operatorname { r e j } ) = \mathbf { 1 } \{ s ^ { \prime } = s \} .\tag{26}
$$

Therefore

$$
U ( \mathrm { a c c } ; s , \tilde { s } , \phi ) = V _ { \phi } ( \tilde { s } ) , \qquad U ( \mathrm { r e j } ; s , \tilde { s } , \phi ) = V _ { \phi } ( s ) ,\tag{27}
$$

and

$$
\Delta U _ { \phi } ( s , \tilde { s } ) = V _ { \phi } ( \tilde { s } ) - V _ { \phi } ( s ) = \Delta V _ { \phi } ( s , \tilde { s } ) .\tag{28}
$$

## B.4 Distance-dependent default response

The binary default policy is fully determined by its log-odds

$$
\lambda _ { \theta } ( s , \tilde { s } ) = \log \frac { q _ { \theta } ( \operatorname { a c c } \mid s , \tilde { s } ) } { q _ { \theta } ( \operatorname { r e j } \mid s , \tilde { s } ) } .\tag{29}
$$

To encode proposal-dependent evaluability, we use a monotone distance penalty:

$$
\lambda _ { \theta } ( s , \tilde { s } ) = - \alpha g ( d ( s , \tilde { s } ) ) ,\tag{30}
$$

where $d ( s , \tilde { s } ) \geq 0$ is proposal distance, and $g$ is a nondecreasing burden transform with $g ( 0 ) = 0$

## C Proofs and Details for the Information Frontier

## C.1 Information frontier conditional on user parameters

We prove Proposition 4.1. The result is conditional on a latent user-parameter $z = ( \phi , \rho , \kappa )$ with $\rho > 0$ and $\kappa > 0$ . This does not assume that the assistant knows z, but analyses the response likelihood pointwise in z; the assistant later integrates the same likelihood under its posterior belief.

Consider a proposal path $\{ \tilde { s } ( t ) \} _ { t \geq 0 }$ from current state $s ,$ parameterised by distance so that $\tilde { s } ( 0 ) = s$ and $d ( s , \tilde { s } ( \bar { t } ) ) \bar { = } t .$ Along this path, define

$$
\eta _ { z } ( t ) = \kappa [ \Delta V _ { \phi } ( t ) - \rho g ( t ) ] , \qquad \Delta V _ { \phi } ( t ) : = V _ { \phi } ( \tilde { s } ( t ) ) - V _ { \phi } ( s ) ,\tag{31}
$$

and let

$$
p _ { z } ( t ) = \sigma ( \eta _ { z } ( t ) )\tag{32}
$$

be the acceptance probability at distance t.

Deriving the Fisher information. For fixed t and z, the binary response can be written as $Y _ { t } \in$ {0, 1}, where $Y _ { t } = 1$ denotes acceptance and $Y _ { t } = 0$ denotes rejection. Thus

$$
Y _ { t } \sim \mathrm { B e r n o u l l i } ( p _ { z } ( t ) ) .\tag{33}
$$

To derive the Fisher information about $\rho ,$ suppress the dependence on t and z for readability and write $p = p _ { z } ( t )$ . The Bernoulli likelihood is

$$
L ( \rho ; Y ) = p ^ { Y } ( 1 - p ) ^ { 1 - Y } ,\tag{34}
$$

and the log-likelihood is

$$
\ell ( \rho ; Y ) = Y \log p + ( 1 - Y ) \log ( 1 - p ) .\tag{35}
$$

Differentiating with respect to $\rho$ gives the score

$$
\frac { \partial \ell } { \partial \rho } = Y \frac { 1 } { p } \frac { \partial p } { \partial \rho } - ( 1 - Y ) \frac { 1 } { 1 - p } \frac { \partial p } { \partial \rho }\tag{36}
$$

$$
= \frac { Y - p } { p ( 1 - p ) } \frac { \partial p } { \partial \rho } .\tag{37}
$$

The Fisher information is the expected squared score:

$$
I _ { z } ^ { \rho } ( t ) = \mathbb { E } _ { Y } [ ( \frac { \partial \ell } { \partial \rho } ) ^ { 2 } ] .\tag{38}
$$

Using Eq. (37),

$$
I _ { z } ^ { \rho } ( t ) = \mathbb { E } _ { Y } [ \frac { ( Y - p ) ^ { 2 } } { p ^ { 2 } ( 1 - p ) ^ { 2 } } ] \left( \frac { \partial p } { \partial \rho } \right) ^ { 2 }\tag{39}
$$

$$
= { \frac { \operatorname { V a r } ( Y ) } { p ^ { 2 } ( 1 - p ) ^ { 2 } } } \left( { \frac { \partial p } { \partial \rho } } \right) ^ { 2 }\tag{40}
$$

$$
= \frac { \left( \frac { \partial p } { \partial \rho } \right) ^ { 2 } } { p ( 1 - p ) } ,\tag{41}
$$

because $\operatorname { V a r } ( Y ) = p ( 1 - p )$ for a Bernoulli random variable.

Since $p _ { z } ( t ) = \sigma ( \eta _ { z } ( t ) )$ and $\sigma ^ { \prime } ( \eta ) = \sigma ( \eta ) ( 1 - \sigma ( \eta ) )$

$$
\frac { \partial p _ { z } ( t ) } { \partial \rho } = \sigma ^ { \prime } ( \eta _ { z } ( t ) ) \frac { \partial \eta _ { z } ( t ) } { \partial \rho }\tag{42}
$$

$$
= p _ { z } ( t ) ( 1 - p _ { z } ( t ) ) \left[ - \kappa g ( t ) \right] .\tag{43}
$$

Substituting Eq. (43) into Eq. (41) gives Eq. (12):

$$
I _ { z } ^ { \rho } ( t ) = \kappa ^ { 2 } g ( t ) ^ { 2 } p _ { z } ( t ) ( 1 - p _ { z } ( t ) ) .\tag{44}
$$

Interpretation. Equation (44) separates informativeness into two factors. The term $g ( t ) ^ { 2 }$ is the squared sensitivity of the logit $\mathbf { t o } \rho .$ It is small near the current state because a small-distance proposal is almost insensitive to the evaluability slope. The term $p _ { z } ( t ) ( 1 - p _ { z } ( t ) )$ is the Bernoulli response variance. It is large when acceptance and rejection are both plausible and small when the response is nearly deterministic. The information frontier arises from the product of these two terms.

Boundary behaviour. $\mathrm { A t } \ i = 0 , g ( 0 ) = 0 .$ , so Eq. (44) gives

$$
I _ { z } ^ { \rho } ( 0 ) = 0 .\tag{45}
$$

For the tail, assume $\Delta V _ { \phi } ( t ) = o ( g ( t ) )$ as $t \to \infty$ . Then for any $\epsilon > 0$ , there exists $T$ such that $\Delta V _ { \phi } ( t ) \le \epsilon g ( t )$ for all $t > T$ . Choose $\epsilon = \rho / 2$ . Then for all $t > T .$

$$
\eta _ { z } ( t ) = \kappa [ \Delta V _ { \phi } ( t ) - \rho g ( t ) ]\tag{46}
$$

$$
\leq - \frac { \kappa \rho } { 2 } g ( t ) .\tag{47}
$$

For $\eta \leq 0$ , we have $\sigma ( \eta ) ( 1 - \sigma ( \eta ) ) \leq \sigma ( \eta ) \leq e ^ { \eta }$ . Thus,

$$
I _ { z } ^ { \rho } ( t ) = \kappa ^ { 2 } g ( t ) ^ { 2 } \sigma ( \eta _ { z } ( t ) ) [ 1 - \sigma ( \eta _ { z } ( t ) ) ]\tag{48}
$$

$$
\leq \kappa ^ { 2 } g ( t ) ^ { 2 } \exp \left( - \frac { \kappa \rho } { 2 } g ( t ) \right) .\tag{49}
$$

Since $g ( t )  \infty$ and $x ^ { 2 } e ^ { - c x }  0$ for any $c > 0 ,$ , the right-hand side tends to zero. Therefore

$$
I _ { z } ^ { \rho } ( t ) \to 0 \qquad \mathrm { a s } \qquad t \to \infty .\tag{50}
$$

Interior maximum. The function $I _ { z } ^ { \rho } ( t )$ is continuous because g and $\Delta V _ { \phi }$ are continuous. For any $t _ { 0 } > 0$ , we have $g ( t _ { 0 } ) > 0$ and $p _ { z } ( t _ { 0 } ) ( 1 - p _ { z } ( t _ { 0 } ) ) > 0$ , so $I _ { z } ^ { \rho } ( t _ { 0 } ) > 0$ . Since $I _ { z } ^ { \rho } ( t ) \to 0$ as $t \to \infty$ , there exists $R > t _ { 0 }$ such that $I _ { z } ^ { \rho } ( t ) < I _ { z } ^ { \rho } ( t _ { 0 } ) / 2$ for all $t > R$ . By continuity, $I _ { z } ^ { \rho }$ attains a maximum on the compact interval $[ 0 , \tilde { R ] }$ . Since $\tilde { I } _ { z } ^ { \dot { \rho } } ( \ r _ { 0 } ) = 0 < I _ { z } ^ { \rho } ( t _ { 0 } )$ , this maximum is attained at some $t ^ { \star } \in ( 0 , R ]$ , and the tail bound makes it a global maximum over $[ 0 , \infty )$

Rejection-side characterization. Now we show that every global maximiser satisfies $\eta _ { z } ( t ^ { \star } ) < 0$ equivalently $p _ { z } ( t ^ { \star } ) < 1 / 2$ with Proof by Elimination.

First suppose $\eta _ { z } ( t ^ { \star } ) > 0$ . Because $\Delta V _ { \phi } ( t ) = o ( g ( t ) )$ and $\rho > 0$ , we have $\eta _ { z } ( t )  - \infty \mathrm { a s } t  \infty$ By continuity, there exists $t _ { F } > t ^ { \star }$ such that $\eta _ { z } ( t _ { F } ) = 0$ . This $t _ { F }$ is the acceptance-frontier distance along the path. Since $g$ is strictly increasing, $g ( t _ { F } ) > g ( t ^ { \star } )$ . At $t _ { F }$ , the Bernoulli variance is maximal:

$$
p _ { z } ( t _ { F } ) ( 1 - p _ { z } ( t _ { F } ) ) = \frac { 1 } { 4 } .\tag{51}
$$

At $t ^ { \star } ,$ , since $\eta _ { z } ( t ^ { \star } ) \neq 0$ , we have

$$
p _ { z } ( t ^ { \star } ) ( 1 - p _ { z } ( t ^ { \star } ) ) < \frac { 1 } { 4 } .\tag{52}
$$

Therefore

$$
I _ { z } ^ { \rho } ( t _ { F } ) = \kappa ^ { 2 } g ( t _ { F } ) ^ { 2 } \frac { 1 } { 4 }\tag{53}
$$

$$
> \kappa ^ { 2 } g ( t ^ { \star } ) ^ { 2 } \frac { 1 } { 4 }\tag{54}
$$

$$
> \kappa ^ { 2 } g ( t ^ { \star } ) ^ { 2 } p _ { z } ( t ^ { \star } ) ( 1 - p _ { z } ( t ^ { \star } ) )
$$

$$
{ \bf \tau } = I _ { z } ^ { \rho } ( t ^ { \star } ) ,\tag{55}
$$

(56)

contradicting the global optimality of $t ^ { \star }$

Now suppose $\eta _ { z } ( t ^ { \star } ) = 0$ . Since $t ^ { \star } > 0$ , we have $g ( t ^ { \star } ) > 0$ . Let

$$
v ( \eta ) = \sigma ( \eta ) ( 1 - \sigma ( \eta ) ) .\tag{57}
$$

Then

$$
I _ { z } ^ { \rho } ( t ) = \kappa ^ { 2 } g ( t ) ^ { 2 } v ( \eta _ { z } ( t ) ) .\tag{58}
$$

Differentiating with respect to t gives

$$
\frac { d } { d t } I _ { z } ^ { \rho } ( t ) = \kappa ^ { 2 } \left[ 2 g ( t ) g ^ { \prime } ( t ) v ( \eta _ { z } ( t ) ) + g ( t ) ^ { 2 } v ^ { \prime } ( \eta _ { z } ( t ) ) \eta _ { z } ^ { \prime } ( t ) \right] .\tag{59}
$$

The derivative of the Bernoulli variance is

$$
v ^ { \prime } ( \eta ) = \sigma ( \eta ) ( 1 - \sigma ( \eta ) ) ( 1 - 2 \sigma ( \eta ) ) .\tag{60}
$$

At η = 0, σ(0) = 1/2, so $v ^ { \prime } ( 0 ) = 0$ and $v ( 0 ) = 1 / 4$ . Therefore, at $t ^ { \star }$ ,

$$
\frac { d } { d t } I _ { z } ^ { \rho } ( t ^ { \star } ) = \kappa ^ { 2 } \cdot 2 g ( t ^ { \star } ) g ^ { \prime } ( t ^ { \star } ) \cdot \frac { 1 } { 4 } > 0 ,\tag{61}
$$

where the strict inequality uses $g ( t ^ { \star } ) > 0$ and $g ^ { \prime } ( t ^ { \star } ) > 0$ . Thus $t ^ { \star }$ cannot be a local maximum, again a contradiction.

Both cases lead to contradictions, so every global maximiser satisfies $\eta _ { z } ( t ^ { \star } ) < 0$ . Equivalently,

$$
p _ { z } ( t ^ { \star } ) = \sigma ( \eta _ { z } ( t ^ { \star } ) ) < \frac { 1 } { 2 } .\tag{62}
$$

This proves Proposition 4.1.

## C.2 Relation to mutual information and value of information

The $\rho \mathrm { - }$ Fisher information in Proposition 4.1 is a local analytic proxy for how proposal distance affects learning about evaluability. A more general Bayesian measure of response informativeness is the mutual information between the latent user parameter and the response:

$$
I _ { b } ( z ; y \mid s , \tilde { s } ) = \mathbb { E } _ { y } [ D _ { \mathrm { K L } } ( b ^ { y } \| b ) ] ,\tag{63}
$$

where $b ^ { y }$ is the posterior after observing response y to proposal s˜.

This quantity is closer to the Bayesian planner’s actual information gain, because it measures how much the response changes the assistant’s belief over the whole latent parameter vector z. The Fisher information used in the main analysis is narrower: it asks how sensitive one binary response is to the evaluability slope $\rho$ at a given parameter vector. We use Fisher information in the theorem because it gives a closed-form local diagnostic of how proposal distance affects learning.

To connect mutual information to downstream planning value, suppose the continuation value function is M-Lipschitz in the belief under total variation distance. Then the downstream value change induced by the belief update satisfies

$$
\lvert \mathrm { V o I } ( s , \tilde { s } , b ) \rvert \leq M \mathbb { E } _ { y } [ \lVert b ^ { y } - b \rVert _ { \mathrm { T V } } ] .\tag{64}
$$

By Pinsker’s inequality,

$$
\| b ^ { y } - b \| _ { \mathrm { T V } } \leq \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( b ^ { y } \| b ) } .\tag{65}
$$

Applying Jensen’s inequality gives

$$
\left| \mathrm { V o I } ( s , \tilde { s } , b ) \right| \leq M \sqrt { \frac { 1 } { 2 } I _ { b } ( z ; y \mid s , \tilde { s } ) } .\tag{66}
$$

Thus mutual information controls the possible downstream value of belief updates, while $\rho \mathrm { \cdot }$ -Fisher information gives a tractable local view of how proposal distance affects learning about evaluability.

## D Planner Details

## D.1 Finite-grid belief tracking

The planner maintains a belief over latent user parameters $z = ( \phi , \rho , \kappa )$ . In the main experiments, we approximate the latent parameter space by a finite grid ${ \mathcal { Z } } _ { \mathrm { g r i d } }$ , and maintain the posterior exactly on this grid. For a current belief $b ,$ state s, proposal s˜, and observed response $y ,$ the grid posterior is

$$
b _ { s , \tilde { s } } ^ { y } ( z ) = \frac { P ( y \mid s , \tilde { s } , z ) b ( z ) } { \sum _ { z ^ { \prime } \in \mathcal { Z } _ { \mathrm { g r i d } } } P ( y \mid s , \tilde { s } , z ^ { \prime } ) b ( z ^ { \prime } ) } .\tag{67}
$$

The corresponding predictive response distribution is

$$
p _ { b } ( y \mid s , \tilde { s } ) = \sum _ { z \in \mathcal { Z } _ { \mathrm { g r i d } } } P ( y \mid s , \tilde { s } , z ) b ( z ) .\tag{68}
$$

Thus the finite-grid approximation only concerns the representation of $b ;$ once the grid is fixed, the Bayesian update is exact on that grid.

## D.2 Response-branch and transition-branch views

The main text presents PROSE-PLAN in terms of hypothetical next states $s ^ { \prime } ,$ while the belief update in Eq. (5) is written in terms of the observed response y. These are two equivalent views in the binary deterministic instantiation.

In the general ProSE process, for fixed user parameter z, the response model and response-mediated transition induce an assistant-side transition kernel

$$
\mathcal { T } _ { z } ^ { \mathrm { A I } } ( s ^ { \prime } \mid s , \tilde { s } ) = \sum _ { y \in \mathcal { V } } P ( y \mid s , \tilde { s } , z ) \mathcal { T } ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , y ) .\tag{69}
$$

Under belief b, the predictive next-state distribution is

$$
p ( s ^ { \prime } \mid b , s , \tilde { s } ) = \sum _ { z \in \mathcal { Z } _ { \mathrm { g r i d } } } b ( z ) \mathcal { T } _ { z } ^ { \mathrm { A I } } ( s ^ { \prime } \mid s , \tilde { s } ) = \sum _ { y \in \mathcal { Y } } p _ { b } ( y \mid s , \tilde { s } ) \mathcal { T } ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , y ) .\tag{70}
$$

This is the quantity used in Eq. (13).

If the response y is observed, the posterior is Eq. (67). If only the realised next state $s ^ { \prime }$ is observed, the posterior can instead be written as

$$
b _ { s , \tilde { s } } ^ { s ^ { \prime } } ( z ) = \frac { \mathcal { T } _ { z } ^ { \mathrm { A I } } ( s ^ { \prime } \mid s , \tilde { s } ) b ( z ) } { \sum _ { z ^ { \prime } \in \mathcal { Z } _ { \mathrm { g r i d } } } \mathcal { T } _ { z ^ { \prime } } ^ { \mathrm { A I } } ( s ^ { \prime } \mid s , \tilde { s } ) b ( z ^ { \prime } ) } .\tag{71}
$$

In the binary deterministic response model,

$$
\mathcal { V } = \{ \mathrm { a c c , r e j } \} , \qquad s ^ { \mathrm { a c c } } = \tilde { s } , \qquad s ^ { \mathrm { r e j } } = s .\tag{72}
$$

Thus, when $\tilde { s } \neq s ,$ , observing $s ^ { \prime } = \tilde { s }$ is equivalent to observing $y = \operatorname { a c c }$ , and observing $s ^ { \prime } = s$ is equivalent to observing $y = \mathrm { r e j }$ . Therefore,

$$
b _ { s , \tilde { s } } ^ { \tilde { s } } = b _ { s , \tilde { s } } ^ { \mathrm { a c c } } , \qquad b _ { s , \tilde { s } } ^ { s } = b _ { s , \tilde { s } } ^ { \mathrm { r e j } } .\tag{73}
$$

This is why Algorithm 1 can branch over the two next-state outcomes $( s , \tilde { s } )$ , while still implementing response-dependent belief updates. If self-proposals $\tilde { s } = s$ are allowed, the next state alone no longer identifies the response; in that case the planner should branch directly on $y ,$ or the response should be treated as observed.

## D.3 Depth-2 expansion

We now spell out the depth-2 score used by PROSE-PLAN. Define the belief-averaged terminal value

$$
H ( s , b ) = \sum _ { z \in \mathcal { Z } _ { \mathrm { g r i d } } } b ( z ) V _ { \phi } ( s ) .\tag{74}
$$

Algorithm 1 PROSE-PLAN: depth-2 proposal selection   
Require: State s, belief $^ { \dag } b ,$ candidate map C, transition $\tau$   
1: for each candidate proposal ${ \tilde { s } } \in { \mathcal { C } } ( s { \bar { ) } }$ do   
2: for both outcomes $\bar { s } ^ { \prime } \in ( s , \tilde { s } )$ do   
3: $p _ { s ^ { \prime } } \gets \mathrm { E } _ { z \sim b } \big [ \mathcal { T } ( s ^ { \prime } \mid s , \tilde { s } ) \big ]$  ▷ Eq.(13)   
4: $b _ { s ^ { \prime } } \gets$ BAYESUPDATE $( \bar { b ; } \ s , \tilde { s } , s ^ { \prime } )$ ▷ Eq. (5)   
5: $\begin{array} { r } { Q _ { 2 } ^ { s ^ { \prime } } ( s , b , \tilde { s } )  \operatorname* { m a x } _ { \tilde { s } ^ { \prime } \in \mathcal { C } ( s ^ { \prime } ) } Q _ { 1 } ^ { * } ( s ^ { \prime } , b _ { s ^ { \prime } } , \tilde { s } ^ { \prime } ) } \end{array}$ ▷ Eq. (2), where $Q _ { 0 } = V _ { \phi }$   
6: end for   
7: $\begin{array} { r } { Q _ { 2 } ( s , b , \tilde { s } ) = \sum _ { s ^ { \prime } } p _ { s ^ { \prime } } \times Q _ { 2 } ^ { s ^ { \prime } } ( s , b , \tilde { s } ) } \end{array}$   
8: end for   
9: return arg $\mathrm { m a x } _ { \tilde { s } \in \mathcal { C } ( s ) } Q _ { 2 } ( \tilde { s } )$

This is the precise meaning of the terminal condition used in the planner: at a leaf node, the planner evaluates the state by averaging the user’s latent value $V _ { \phi } ( s )$ under the current belief.

For a one-step proposal s˜ from state s under belief $b ,$ define

$$
Q _ { 1 } ( s , b , \tilde { s } ) = \sum _ { y \in \mathcal { V } } p _ { b } ( y \mid s , \tilde { s } ) \sum _ { s ^ { \prime } \in \mathcal { S } } \mathcal { T } ^ { y } ( s ^ { \prime } \mid s , \tilde { s } , y ) H ( s ^ { \prime } , b _ { s , \tilde { s } } ^ { y } ) .\tag{75}
$$

Under the binary deterministic transition, this simplifies to

$$
Q _ { 1 } ( s , b , \tilde { s } ) = p _ { b } ( \operatorname { a c c } \mid s , \tilde { s } ) H ( \tilde { s } , b _ { s , \tilde { s } } ^ { \operatorname { a c c } } ) + p _ { b } ( \operatorname { r e j } \mid s , \tilde { s } ) H ( s , b _ { s , \tilde { s } } ^ { \operatorname { r e j } } ) .\tag{76}
$$

The depth-2 score of a first proposal s˜ is obtained by considering each possible first response, updating the belief under that response, and choosing the best second proposal:

$$
Q _ { 2 } ( \boldsymbol { s } , \boldsymbol { b } , \tilde { \boldsymbol { s } } ) = \sum _ { y \in \mathcal { V } } p _ { b } ( y \mid \boldsymbol { s } , \tilde { \boldsymbol { s } } ) \sum _ { \boldsymbol { s } ^ { \prime } \in S } \mathcal { T } ^ { y } ( \boldsymbol { s } ^ { \prime } \mid \boldsymbol { s } , \tilde { \boldsymbol { s } } , y ) \operatorname* { m a x } _ { \tilde { \boldsymbol { s } } ^ { \prime } \in \mathcal { C } ( \boldsymbol { s } ^ { \prime } ) } Q _ { 1 } ( \boldsymbol { s } ^ { \prime } , \boldsymbol { b } _ { \boldsymbol { s } , \tilde { \boldsymbol { s } } } ^ { y } , \tilde { \boldsymbol { s } } ^ { \prime } ) .\tag{77}
$$

In the binary deterministic case, Eq. (77) becomes

$$
\begin{array} { r l } & { Q _ { 2 } ( s , b , \tilde { s } ) = p _ { b } ( \mathrm { a c c } \mid s , \tilde { s } ) \underset { \tilde { s } ^ { \prime } \in \mathcal { C } ( \tilde { s } ) } { \operatorname* { m a x } } Q _ { 1 } ( \tilde { s } , b _ { s , \tilde { s } } ^ { \mathrm { a c c } } , \tilde { s } ^ { \prime } ) } \\ & { \qquad + p _ { b } ( \mathrm { r e j } \mid s , \tilde { s } ) \underset { \tilde { s } ^ { \prime } \in \mathcal { C } ( s ) } { \operatorname* { m a x } } Q _ { 1 } ( s , b _ { s , \tilde { s } } ^ { \mathrm { r e j } } , \tilde { s } ^ { \prime } ) . } \end{array}\tag{78}
$$

Algorithm 1 is an implementation of Eq. (78), written in terms of the two possible next states $s ^ { \prime } = \tilde { s }$ and $s ^ { \prime } = s$

## D.4 Relation to the Bayes-adaptive recursion

Eq. (77) is the $h = 2$ finite-lookahead instantiation of the Bayes-adaptive recursion in Eq. (2). The generic action a in Eq. (2) is a proposal $\tilde { s } \in \mathcal { C } ( s )$ in ProSE. The generic latent model parameter is the latent user parameter $z = ( \phi , \rho , \kappa )$ . The transition likelihood in the generic BA-MDP is replaced by the proposal-induced transition kernel $\mathcal { T } _ { z } ^ { \mathrm { A I } }$ in Eq. (69), or equivalently by the response likelihood $\bar { P ( y \mid s , \tilde { s } , z ) }$ when the response is observed. The terminal value is $\tilde { H ( s , b ) }$ in Eq. (74), rather than an additional immediate reward at each step, because our experiments use a terminal-artefact objective.

## D.5 Computational complexity

Let |C| be the maximum candidate-set size, |Y| the number of possible responses, and $| \mathcal { Z } _ { \mathrm { g r i d } } |$ the number of grid points in the belief. Computing either $p _ { b } ( y \mid s , \tilde { s } )$ or $H ( s , b )$ requires summing over the grid, and therefore costs $O ( | \mathcal { Z } _ { \mathrm { g r i d } } | )$ . For each first-step proposal, PROSE-PLAN considers |Y| first responses; for each response, it evaluates up to $| { \mathcal { C } } |$ second proposals; and each second proposal considers |Y| responses. Thus exhaustive depth-2 evaluation costs

$$
O ( | \mathcal { C } | ^ { 2 } | \mathcal { V } | ^ { 2 } | \mathcal { Z } _ { \mathrm { g r i d } } | )
$$

per planning step under deterministic response-mediated transitions. In our experiments, $| { \mathcal { N } } | = 2 ,$ , so the main scaling factors are the candidate-set size and the grid size.

Table 3: Baseline methods. Each method removes one or more components of PROSE-PLAN.
<table><tr><td>Method</td><td>Role</td><td>Preference</td><td>Evaluability</td><td>Personal. ρ, k</td><td>Depth-2 lookahead</td><td>Resp.-dep. lookahead update</td></tr><tr><td>Random</td><td>baseline</td><td>none</td><td>none</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Value-greedy</td><td>baseline</td><td>posterior</td><td>none</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Threshold [26]</td><td>baseline</td><td>posterior</td><td>fixed threshold</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Population-myopic</td><td>baseline</td><td>posterior</td><td>fixed  $( \bar { \rho } , \bar { \kappa } )$ </td><td>x</td><td>x</td><td>x</td></tr><tr><td>Personalized-myopic Belief-frozen</td><td>baseline</td><td>posterior</td><td>posterior</td><td>√</td><td>X</td><td>x</td></tr><tr><td>depth-2</td><td>ablation</td><td>posterior</td><td>posterior</td><td>√</td><td>√</td><td>x</td></tr><tr><td>PROSE-PLAN</td><td>full method</td><td>posterior</td><td>posterior</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Oracle depth-2</td><td>reference</td><td>true z</td><td>true z</td><td></td><td>√</td><td></td></tr></table>

## E Experiment Details

This appendix provides implementation details, full results, and sensitivity checks for the experiments in Section 6.

## E.1 Baseline Definitions

All methods operate on the same candidate set $\mathscr { C } ( s _ { t } )$ unless otherwise stated. For a belief $b ,$ define the belief-averaged state value

$$
H ( s , b ) = \sum _ { z \in \mathcal { Z } _ { \mathrm { g r i d } } } b ( z ) V _ { \phi } ( s ) .
$$

For a proposal s˜ from state s, define the belief-predictive response probability

$$
p _ { b } ( y \mid s , \tilde { s } ) = \sum _ { z \in \mathcal { Z } _ { \mathrm { g r i d } } } P ( y \mid s , \tilde { s } , z ) b ( z ) ,
$$

and let $b _ { s , \tilde { s } } ^ { y }$ denote the posterior after hypothetically observing response $y .$ Under the binary deterministic transition, write

$$
s ^ { y } = \left\{ { \begin{array} { l l } { { \tilde { s } } , } & { { y = \mathrm { a c c } , } } \\ { { s , } } & { { y = \mathrm { r e j } . } } \end{array} } \right.
$$

## Random.

$$
{ \tilde { s } } \sim U ( { \mathcal { C } } ( s ) ) .
$$

The random baseline samples uniformly from the candidate set and does not use the value function, response model, belief state, proposal distance, or planning horizon. It serves as a null policy for checking that the task is nontrivial and that successful performance is not an artifact of candidate-set construction or graph topology alone. Observed responses are still generated by the same user model as in all other conditions, but the random planner does not use those responses to choose future proposals.

Value-greedy.

$$
\widetilde { s } _ { * } = \arg \operatorname* { m a x } _ { \widetilde { s } \in \mathcal { C } ( s ) } \mathbb { E } _ { \phi \sim b _ { t } } [ V _ { \phi } ( \widetilde { s } ) ] = \arg \operatorname* { m a x } _ { \widetilde { s } \in \mathcal { C } ( s _ { t } ) } \sum _ { \phi } b _ { t } ( \phi ) V _ { \phi } ( \widetilde { s } ) .
$$

The value-greedy baseline selects the candidate with the highest posterior-expected user value under the current belief over the preferred branch. This baseline tests whether the task can be solved by preference learning and value maximisation alone, without modelling evaluability constraints. It ignores proposal distance in the planning objective and uses a fixed response model with $\rho = 0$ and $\kappa = 1$ ; therefore, rejections are interpreted as evidence about $\phi$ rather than evidence that the proposal was too difficult to evaluate.

Threshold.

$$
\widetilde s _ { * } = \arg \operatorname* { m a x } _ { \widetilde { s } \in \mathcal { C } _ { \tau } ( s _ { t } ) } \mathbb { E } _ { \phi \sim b _ { t } } [ V _ { \phi } ( \widetilde { s } ) ] = \arg \operatorname* { m a x } _ { \widetilde { s } \in \mathcal { C } _ { \tau } ( s _ { t } ) } \sum _ { \phi } b _ { t } ( \phi ) V _ { \phi } ( \widetilde { s } ) .
$$

where $\mathcal { C } _ { \tau } ( s _ { t } ) = \{ s \in \mathcal { C } ( s _ { t } ) : d ( s _ { t } , s ) \leq \tau \}$ . This baseline is a distance-constrained version of the Value-greedy planner. It represents a simple fixed-locality heuristic for evaluability. Rather than modelling or inferring the user’s distance sensitivity, the assistant removes proposals that are farther than a global threshold and then applies the same value-greedy rule as Value-greedy.

The baseline tests whether the benefits of evaluability-aware planning can be explained by a handtuned locality constraint alone, without personalised inference over $\rho$ or response-contingent lookahead. As with Value-greedy, it updates the belief over $\phi$ from observed responses but uses a fixed response model with $\rho = 0$ and $\kappa = 1$ . Because τ is a free heuristic parameter rather than a parameter learned from the response model, we tune it via grid search on held-out validation seeds. For the corridor experiments, we evaluate $\tau \in \{ 1 , \dots , 8 \}$ on validation seeds 0–49, across $\rho _ { \mathrm { t r u e } } \in \{ 0 . 0 4 , 0 . 0 8 , 0 . 1 8 , \bar { 0 } . 3 0 , 0 . 3 6 \} , \alpha _ { \mathrm { e n v } } \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 \bar { 5 } , 1 \}$ , and $\kappa _ { \mathrm { t r u e } } = 1 . 0$ . We choose the threshold with the highest mean terminal value, breaking ties by success rate and then by the smaller threshold, so equally good thresholds favour the more local and conservative rule. This procedure selects $\tau = 4 . 0 ;$ ; in the validation run, $\tau = 4 , \dots , 8$ tie in mean terminal value and success rate, so the smallest tied value is used. If no candidate satisfies the cutoff, the method falls back to the nearest available candidate.

Population myopic.

$$
\tilde { s } _ { * } = \arg \operatorname* { m a x } _ { \tilde { s } \in \mathcal { C } ( s _ { t } ) } \sum _ { \phi } b _ { t } ( \phi ) \Big [ P _ { \bar { \rho } , \bar { \kappa } } \big ( \mathrm { a c c } \mid s _ { t } , \tilde { s } , \phi \big ) V _ { \phi } ( \tilde { s } ) + \big ( 1 - P _ { \bar { \rho } , \bar { \kappa } } \big ( \mathrm { a c c } \mid s _ { t } , \tilde { s } , \phi \big ) \big ) V _ { \phi } ( s _ { t } ) \Big ] ,
$$

where

$$
P _ { \bar { \rho } , \bar { \kappa } } ( \mathrm { a c c } \mid s _ { t } , \tilde { s } , \phi ) = \sigma \big ( \bar { \kappa } \left[ V _ { \phi } ( \tilde { s } ) - V _ { \phi } ( s _ { t } ) - \bar { \rho } d ^ { 2 } ( s _ { t } , \tilde { s } ) \right] \big ) .
$$

The population-myopic baseline is an evaluability-aware one-step planner with fixed populationlevel response parameters $( \bar { \rho } , \bar { \kappa } )$ . It tests whether a generic evaluability model is sufficient, without personalising the user’s distance sensitivity or response sharpness from interaction. Unlike Valuegreedy and Threshold, it uses proposal distance through the predicted acceptance probability, so distant proposals can be down-weighted even when their expected value is high. Unlike PROSE-PLAN and the personalised myopic baseline below, it updates only the belief over $\phi$ from observed responses; $\bar { \rho }$ and κ¯ remain fixed throughout the episode. The method is also myopic: it scores only the expected value after the current accept/reject response and does not plan for how that response would change future proposals. For the main corridor experiments, we set $( \bar { \rho } , \bar { \kappa } ) = ( 0 . 1 \bar { 8 } , 1 . 0 )$ These values are fixed a priori, not tuned on validation seeds, because the purpose of this baseline is to represent a generic typical-user model rather than the best validation-optimised fixed-response model. The value $\bar { \rho } = 0 . 1 8$ is the central value of the synthetic corridor evaluability range used in the main sweep, which spans low-cost users through $\rho _ { \mathrm { t r u e } } = 0 . 3 6$ . Thus, the baseline is deliberately well matched to moderate-cost users and is expected to be competitive near $\rho _ { \mathrm { t r u e } } = 0 . 1 8$ .The value $\bar { \kappa } = 1 . 0$ matches the response sharpness used to generate the main corridor data, giving this baseline the correct response-noise scale and isolating the effect of using a non-personalised evaluability slope.

Personalised-myopic.

$$
\tilde { s } _ { * } = \arg \operatorname* { m a x } _ { \tilde { s } \in \mathcal { C } ( s _ { t } ) } \sum _ { z } b _ { t } ( z ) \Big [ P ( \mathrm { a c c } \mid s _ { t } , \tilde { s } , z ) V _ { \phi } ( \tilde { s } ) + \big ( 1 - P ( \mathrm { a c c } \mid s _ { t } , \tilde { s } , z ) \big ) V _ { \phi } ( s _ { t } ) \Big ] ,
$$

where

$$
P ( \operatorname { a c c } \mid s _ { t } , { \tilde { s } } , z ) = \sigma ( \kappa [ V _ { \phi } ( { \tilde { s } } ) - V _ { \phi } ( s _ { t } ) - \rho d ^ { 2 } ( s _ { t } , { \tilde { s } } ) ] ) .
$$

The personalised-myopic baseline is the one-step version of the evaluability-aware planner. It uses the full posterior belief over $z = ( \phi , \rho , \kappa )$ to predict acceptance probabilities and to score the immediate accept/reject outcome of each candidate. Unlike population-myopic, it personalises the evaluability slope and response sharpness from observed responses by updating the posterior over $\rho$ and κ as well as ϕ. However, the proposal score is still myopic: it includes only the value of the state reached after the current response, and does not include the downstream value of how that response would change future proposals. This baseline therefore isolates the value of personalised evaluability inference without response-contingent lookahead.

Belief-frozen depth-2.

where

$$
\begin{array} { r l } & { \widetilde s _ { * } = \arg \underset { \widetilde { s } \in \mathcal { C } ( s _ { t } ) } { \operatorname* { m a x } } \underset { y \in \{ \mathrm { a c c } , \mathrm { r e j } \} } { \sum } p _ { b _ { t } } ( y \mid s _ { t } , \widetilde { s } ) \underset { \widetilde { s } ^ { \prime } \in \mathcal { C } ( s _ { t } ^ { y } ) } { \operatorname* { m a x } } Q _ { 1 } ^ { \mathrm { f r o z e n } } ( s _ { t } ^ { y } , b _ { t } , \widetilde { s } ^ { \prime } ) , } \\ & { \qquad Q _ { 1 } ^ { \mathrm { f r o z e n } } ( s , b , \widetilde { s } ^ { \prime } ) = \underset { y ^ { \prime } \in \{ \mathrm { a c c } , \mathrm { r e j } \} } { \sum } p _ { b } ( y ^ { \prime } \mid s , \widetilde { s } ^ { \prime } ) H ( s ^ { y ^ { \prime } } , b ) . } \end{array}
$$

This ablation performs depth-2 lookahead with the same horizon as PROSE-PLAN, but it holds the belief fixed at $b _ { t }$ when evaluating the second-step proposal in each hypothetical first-response branch. Equivalently, it asks: if the first response changed the task state but did not teach the assistant anything about the user, which of the first proposals would look best? The method still updates the belief after the actually observed response during execution; the “frozen” assumption applies only inside the planning tree used to score hypothetical continuations. This ablation isolates the value of using a hypothetical response as a learning signal before scoring the follow-up proposal, beyond the value of simply looking two steps ahead in the task state.

Oracle depth-2.

$$
\tilde { s } _ { * } = \arg \operatorname* { m a x } _ { \tilde { s } \in \mathcal { C } ( s _ { t } ) } \sum _ { y \in \{ \mathrm { a c c } , \mathrm { r e j } \} } P ( y \mid s _ { t } , \tilde { s } , z _ { \mathrm { t r u e } } ) \underset { \tilde { s } ^ { \prime } \in \mathcal { C } ( s _ { t } ^ { y } ) } { \operatorname* { m a x } } Q _ { 1 } ( s _ { t } ^ { y } , \delta _ { z _ { \mathrm { t r u e } } } , \tilde { s } ^ { \prime } ) ,
$$

where $\delta _ { z _ { \mathrm { t r u e } } }$ is a point-mass belief on the true latent parameters $z _ { \mathrm { t r u e } } = ( \phi _ { \mathrm { t r u e } } , \rho _ { \mathrm { t r u e } } , \kappa _ { \mathrm { t r u e } } )$ . Oracle depth-2 runs the same depth-2 planner as PROSE-PLAN, but removes all user-model uncertainty by giving the assistant the true preference, evaluability slope, and response sharpness. It is therefore a reference ceiling, which measures how well a depth-2 planner could perform if inference over z were solved perfectly. Because user responses are still sampled from the stochastic response model, even the true best proposal can be rejected, oracle knowledge does not guarantee success in every finite-horizon episode.

## E.2 Branching-Corridor Additional Results

## E.2.1 Full experimental setup

This subsection gives the full instantiation of the Branching-Corridor task used in Section 6.2. The environment is a finite tree with one shared corridor and multiple preference-specific branches. Each episode samples a hidden preferred branch ϕ, which determines the user’s goal state and state-value function. At each step, the assistant observes the current realised state and may propose any non-current node. Acceptance moves the state to the proposed node, while rejection leaves the state unchanged. For belief-based planners, the assistant maintains an exact discrete posterior over ${ z } = ( \phi , \rho , \kappa )$ after observing accept/reject responses. The proposal distance $d ( s , \tilde { s } )$ is the shortest-path distance on the tree, and the main experiments use the quadratic evaluability transform $g ( d ) = \grave { d } ^ { 2 }$

Table 4 summarises the graph structure, value function, response-model parameters, and sweep ranges. The default high-cost condition reported in Table 1 uses $\alpha _ { \mathrm { e n v } } = 0 . 2 5 , \rho _ { \mathrm { t r u e } } = 0 . 3 0$ , and $\kappa _ { \mathrm { t r u e } } = 1 . 0$ . Main sweeps use 200 seeds per condition; robustness and ablation checks use 50 seeds unless otherwise stated.

## E.2.2 Additional Branching-Corridor Ablation

The main Branching-Corridor comparison in Section 6.2 focuses on end-to-end baselines that remove evaluability awareness, personalisation, or non-myopic proposal planning. Here we report an additional belief-frozen depth-2 ablation under the same default high-cost condition. This ablation has the same depth-2 horizon as PROSE-PLAN, but evaluates second-step continuations under the current belief $b _ { t } .$ , rather than under the posterior induced by hypothetical first-step user responses. During actual execution, however, it still updates its belief after observing real user responses.

Table 5 shows that belief-frozen depth-2 outperforms PROSE-PLAN in this environment. This does not contradict the main Q1 result: the frozen planner is not a non-adaptive baseline, but an internal mechanism ablation that shares the same response model, personalised belief state, and depth-2 horizon. Rather, this result indicates that the branching corridor is not a clean isolation test for response-dependent belief updates during lookahead. Branch proposals already generate informative real accept/reject outcomes during execution, so a planner can benefit from the resulting posterior updates even if it did not explicitly value hypothetical belief updates inside the lookahead tree. In this environment, anticipating the belief update during planning is therefore partly redundant. For this reason, we use the Probe-Commit task in Section 6.3 to isolate whether the advantage of PROSE-PLAN comes from depth-2 lookahead alone or from response-dependent information probing.

Table 4: Branching-Corridor graph, value function, and response-model setup.
<table><tr><td>Component</td><td>Notation</td><td>Value</td></tr><tr><td>Branches</td><td>K</td><td>4</td></tr><tr><td>Corridor length</td><td>C</td><td>2</td></tr><tr><td>Branch length</td><td> $L$ </td><td>4</td></tr><tr><td>State space</td><td> $\{ s _ { 0 } , c _ { 1 } , c _ { 2 } \} \cup \{ b _ { k , j } \}$ </td><td> $k \in \{ 1 , \ldots , 4 \} , j \in \{ 1 , \ldots , 4 \}$ </td></tr><tr><td>Start state</td><td>S0</td><td>fixed</td></tr><tr><td>Hidden preference</td><td> $\phi$ </td><td> $1 , 2 , 3 , 4$ </td></tr><tr><td>Goal state</td><td> $g _ { \phi }$ </td><td> $b _ { \phi , 4 }$ </td></tr><tr><td>Success</td><td> $1 \{ s _ { T } = g _ { \phi } \}$ </td><td>reach preferred leaf</td></tr><tr><td>Horizon</td><td> $T$ </td><td>5 proposals</td></tr><tr><td>Candidate set</td><td> $\mathscr { C } ( s _ { t } )$ </td><td>all states except current state</td></tr><tr><td>Corridor value scale</td><td> $w _ { c }$ </td><td>3.0</td></tr><tr><td>Preferred branch reward</td><td> $w _ { b }$ </td><td>2.0 per depth</td></tr><tr><td>Wrong branch penalty</td><td> $w _ { p }$ </td><td>3.0 per depth</td></tr><tr><td>Corridor weight</td><td> $\alpha _ { \mathrm { e n v } }$ </td><td>main: 0.25; alignment sweep: {0, 0.25, 0.5, 0.75, 1}</td></tr><tr><td>Value at start</td><td> $V _ { \phi } ( s _ { 0 } )$ </td><td>0</td></tr><tr><td>Corridor value</td><td> $V _ { \phi } ( c _ { j } )$ </td><td> $\alpha _ { \mathrm { e n v } } w _ { c } j$ </td></tr><tr><td>Preferred branch value</td><td> $V _ { \phi } ( b _ { \phi , j } )$ </td><td> $\alpha _ { \mathrm { e n v } } w _ { c } C + w _ { b } j$ </td></tr><tr><td>Wrong branch value</td><td> $\dot { V _ { \phi } ( b _ { k , j } ) } , k \neq \phi$ </td><td> $\alpha _ { \mathrm { e n v } } w _ { c } C - w _ { p } j$ </td></tr><tr><td>Distance transform</td><td> $g ( d )$ </td><td>main:  $d ^ { 2 } ;$  robustness:  $\{ d , d ^ { 1 . 5 } , d ^ { 2 } , d ^ { 2 . 5 } \}$ </td></tr><tr><td>True evaluability cost</td><td> $\rho _ { \mathrm { t r u e } }$ </td><td>main sweep:  $\{ 0 . 0 4 , 0 . { \dot { 0 } } 8 , 0 . 1 2 , 0 . 1 8 , 0 . { \dot { 2 } } 4 , 0 . 3 0 , 0 . 3 6 \}$ </td></tr><tr><td>True inverse temperature</td><td> $\kappa _ { \mathrm { t r u e } }$ </td><td>1.0</td></tr><tr><td>Lapse rate Default belief grid</td><td> $\epsilon$ </td><td>main: 0; robustness: {0, 0.05, 0.10}</td></tr><tr><td></td><td> $\rho$ </td><td>36 linearly spaced points in [0.01, 0.36]</td></tr><tr><td>Default belief grid</td><td> $\kappa$ </td><td>8 geometrically spaced points in [0.5, 4.0]</td></tr><tr><td>Preference prior</td><td> $p ( \phi )$ </td><td>uniform over branches</td></tr></table>

Table 5: Default-condition Branching-Corridor ablation including belief-frozen depth-2. The setting is $\alpha _ { \mathrm { e n v } } = 0 . 2 5$ $\rho _ { \mathrm { t r u e } } = 0 . 3 0$ , and $\kappa _ { \mathrm { t r u e } } = 1 . 0$ . Values are reported as mean ± standard error over 200 seeds.
<table><tr><td>Method</td><td> $V _ { \phi } ( s _ { T } )$ </td><td>Success rate</td></tr><tr><td>Population-myopic</td><td> $1 . 9 0 \pm 0 . 2 7$ </td><td> $0 . 2 0 0 \pm 0 . 0 2 8$ </td></tr><tr><td>Personalised-myopic</td><td> $2 . 8 4 \pm 0 . 2 8$ </td><td> $0 . 2 1 5 \pm 0 . 0 2 9$ </td></tr><tr><td>Frozen depth-2</td><td> ${ \bf 7 . 6 0 \pm 0 . 2 4 }$ </td><td> $\mathbf { 0 . 7 6 5 \pm 0 . 0 3 0 }$ </td></tr><tr><td>PROSE-PLAN</td><td> $5 . 6 2 \pm 0 . 2 9$ </td><td> $0 . 5 1 5 \pm 0 . 0 3 5$ </td></tr><tr><td>Oracle depth-2</td><td> $9 . 4 1 \pm 0 . 0 7$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 7$ </td></tr></table>

## E.2.3 Aggregate Proposal behaviour

To understand the behavioural difference behind the Branching-Corridor result, we measure how far each planner proposes from the current state and how often those proposals are accepted. For each proposal step t, we report the mean proposal distance $d ( s _ { t } , \tilde { s } _ { t } )$ and the empirical acceptance rate over episodes that remain active at that step.

Table 6 and Figure 4 show a clear behavioural separation between PROSE-PLAN and the myopic baselines. POPULATION-MYOPIC repeatedly proposes maximal-distance branch states with distance

Table 6: Aggregate proposal behaviour at the default high-cost Branching-Corridor condition $( \alpha _ { \mathrm { e n v } } =$ 0.25, $\rho _ { \mathrm { t r u e } } = 0 . 3 0 , \kappa _ { \mathrm { t r u e } } = 1 . 0 )$ . Mean proposal distance and empirical acceptance rate are computed over episodes active at each proposal step. Values are mean ± standard error over 200 seeds.
<table><tr><td colspan="6">Mean proposal distance  $d ( s _ { t } , \tilde { s } _ { t } )$ </td></tr><tr><td>Method</td><td> $t = 0$ </td><td> $t = 1$ </td><td> $t = 2$ </td><td> $t = 3$ </td><td> $t = 4$ </td></tr><tr><td>Random</td><td> $3 . 9 4 \pm 0 . 1 0$ </td><td> $4 . 0 0 \pm 0 . 1 0$ </td><td> $3 . 7 5 \pm 0 . 1 0$ </td><td> $3 . 7 3 \pm 0 . 1 1$ </td><td> $3 . 7 0 \pm 0 . 1 2$ </td></tr><tr><td>Population-myopic</td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Personalised-myopic</td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>PROSE-PLAN</td><td> $4 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 6 5 \pm 0 . 0 5$ </td><td> $3 . 5 4 \pm 0 . 0 7$ </td><td> $2 . 0 5 \pm 0 . 0 2$ </td><td> $3 . 3 3 \pm 0 . 1 1$ </td></tr><tr><td>Oracle depth-2</td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td colspan="6">Acceptance rate</td></tr><tr><td>Method</td><td> $t = 0$ </td><td> $t = 1$ </td><td> $t = 2$ </td><td> $t = 3$ </td><td>t = 4</td></tr><tr><td>Random</td><td> $0 . 1 9 0 \pm 0 . 0 2 8$ </td><td> $0 . 1 8 1 \pm 0 . 0 2 7$ </td><td> $0 . 1 7 9 \pm 0 . 0 2 8$ </td><td> $0 . 1 2 1 \pm 0 . 0 2 4$ </td><td> $0 . 2 4 7 \pm 0 . 0 3 2$ </td></tr><tr><td>Population-myopic</td><td> $0 . 0 3 5 \pm 0 . 0 1 3$ </td><td> $0 . 0 4 7 \pm 0 . 0 1 5$ </td><td> $0 . 0 4 3 \pm 0 . 0 1 5$ </td><td> $0 . 0 3 4 \pm 0 . 0 1 4$ </td><td> $0 . 0 5 9 \pm 0 . 0 1 8$ </td></tr><tr><td>Personalised-myopic</td><td> $0 . 0 5 0 \pm 0 . 0 1 5$ </td><td> $0 . 0 7 4 \pm 0 . 0 1 9$ </td><td> $0 . 0 5 1 \pm 0 . 0 1 7$ </td><td> $0 . 0 6 0 \pm 0 . 0 1 8$ </td><td> $0 . 1 8 5 \pm 0 . 0 3 1$ </td></tr><tr><td>PROSE-PLAN</td><td> $0 . 1 7 5 \pm 0 . 0 2 7$ </td><td> $0 . 3 3 0 \pm 0 . 0 3 3$ </td><td> $0 . 3 5 5 \pm 0 . 0 3 7$ </td><td> $0 . 6 6 2 \pm 0 . 0 4 0$ </td><td> $0 . 3 5 5 \pm 0 . 0 4 6$ </td></tr><tr><td>Oracle depth-2</td><td> $0 . 7 0 0 \pm 0 . 0 3 2$ </td><td> $0 . 8 8 5 \pm 0 . 0 2 3$ </td><td> $0 . 8 9 4 \pm 0 . 0 3 8$ </td><td> $0 . 8 3 3 \pm 0 . 0 9 0$ </td><td> $0 . 7 1 4 \pm 0 . 1 8 4$ </td></tr></table>

Proposal-Distance Diagnostics $( \alpha _ { \mathrm { e n v } } = 0 . 2 5 , \rho _ { \mathrm { t r u e } } = 0 . 3 0 )$  
![](images/30e936fd58a209a2a88da27cf51d9d99379d1ed9be0fa27524a7ee18d6678b1f.jpg)  
Figure 4: Aggregate proposal behaviour at the default high-cost Branching-Corridor condition. Compared with myopic baselines, PROSE-PLAN avoids maximal-distance branch jumps and obtains substantially higher acceptance rates, consistent with its use of more evaluable intermediate proposals.

6, leading to acceptance rates below 0.06 throughout the episode. PERSONALISED-MYOPIC behaves similarly for the first four proposal steps and only shortens its proposal at the final step, after most opportunities for successful progress have already been lost. In contrast, PROSE-PLAN avoids these maximal-distance jumps: its mean proposal distance decreases from 4.00 at t = 0 to 2.05 at t = 3, while its acceptance rate increases from 0.175 to 0.662. At the final step, the mean distance increases again as remaining active episodes require more decisive branch proposals, but the acceptance rate remains substantially above the myopic baselines.

This analysis supports our interpretation of the main result that, PROSE-PLAN does not improve merely by proposing lower-value local moves, but by selecting proposals that better balance value gain against evaluability.

## E.2.4 Representative Proposal Trajectories

Figure 5 shows representative trajectories of all methods in the default high-cost Branching-Corridor condition $( \alpha _ { \mathrm { e n v } } = 0 . 2 5 , \ \rho _ { \mathrm { t r u e } } = 0 . 3 0 , \ \kappa _ { \mathrm { t r u e } } = 1 . 0 )$ , using the same seed for all methods (seed = 113); the true branch is BR2. This figure shows that, VALUE-GREEDY and THRESHOLD remain in the shared corridor, proposing generic states that are easy to evaluate but insufficient for identifying the preferred branch. POPULATION-MYOPIC and PERSONALISED-MYOPIC mostly propose distant leaf states directly; under high evaluability cost, these proposals are typically rejected, so progress is limited. PROSE-PLAN instead proposes an intermediate true-branch state b2-2 before committing to b2-4, reaching the preferred goal in two accepted steps. This example is consistent with our aggregate results, where PROSE-PLAN succeeds by using evaluable stepping stones rather than repeated long-distance branch jumps.

![](images/89e3b6b36c18f65f7ea446fd9db34299eb7b5c1260d20713a589d93965a5ae26.jpg)  
Figure 5: Representative proposal trajectories in the default high-cost Branching-Corridor condition $( \alpha _ { \mathrm { e n v } } = 0 . 2 5$ $\rho _ { \mathrm { t r u e } } = 0 . 3 0$ $\kappa _ { \mathrm { t r u e } } = 1 . 0 )$ , using the same episode seed across all methods (seed $= 1 1 3 ) $ ; the true branch is BR2. Each box shows one proposal. Dark borders indicate accepted proposals, bold text marks proposals on the true branch, and “done” indicates that the episode has already terminated. The figure qualitatively illustrates three behaviours: conservative corridor proposals (VALUE-GREEDY, THRESHOLD), over-ambitious direct branch jumps (POPULATION-MYOPIC, PERSONALISED-MYOPIC), and evaluability-aware stepping-stone proposals (PROSE-PLAN).

## E.3 Probe-Commit Additional Results

## E.3.1 Full experimental setup

Table 7 summarises the full setup for the Probe-Commit task used in Section 6.3. The task is intentionally minimal: there are two possible user types, two low-value probe states, and two high-value goal states. A goal proposal gives high immediate value under both preferences but is weakly diagnostic of the hidden preference $\phi .$ A probe proposal has lower immediate value, but its accept/reject response is more informative about ϕ. With horizon $T = 2$ , this structure makes probing useful only if the planner uses the first response to update its belief before choosing the second proposal. As in the Branching-Corridor task, the assistant may propose any non-current state, proposal distance is the shortest-path distance on the graph, acceptance moves the state to the proposal, and rejection leaves the state unchanged. The default condition used in Table 1 sets $w _ { p , - } = - 3 . 0 , \rho _ { \mathrm { t r u e } } = 0 . 5 ,$ , and $\kappa _ { \mathrm { t r u e } } = 1 . 0$ . Unless otherwise stated, results are averaged over 200 random seeds.

## E.3.2 Sensitivity to Probe Diagnosticity

We vary the mismatched-probe value $w _ { p , - } ,$ which controls how informative a probe response is about ϕ. More negative $w _ { p , - }$ makes wrong-branch probes easier to reject, so observing accept/reject at a probe more strongly identifies the preferred branch.

Figure 6 shows that PROSE-PLAN probes only in the diagnostic regime. At $w _ { p , - } = - 1$ , it does not probe and its success is close to the direct-commit baselines. For $w _ { p , - } \leq - 2 .$ , it probes in every episode and achieves substantially higher success, while PERSONALISED-MYOPIC and BELIEF-FROZEN DEPTH-2 never probe. Random probes about half the time but remain much worse, showing that probe selection must be paired with posterior-dependent continuation planning. We thus conclude that PROSE-PLAN does not blindly probe, rather, it probes only when probe responses are sufficiently informative for improving the second proposal.

Table 7: Probe-Commit environment setup.
<table><tr><td>Component</td><td>Notation</td><td>Value</td></tr><tr><td>Arms</td><td>K</td><td>2</td></tr><tr><td>State space</td><td>S</td><td> $\left\{ s _ { 0 } , p _ { 1 } , p _ { 2 } , g _ { 1 } , g _ { 2 } \right\}$ </td></tr><tr><td>Start state</td><td>S0</td><td>fixed</td></tr><tr><td>Hidden preference</td><td> $\phi$ </td><td> $\{ 1 , 2 \}$ </td></tr><tr><td>Preferred goal</td><td> $g _ { \phi }$ </td><td> ${ \dot { g } } _ { 1 } { \mathrm { i f } } { \dot { \phi } } = 1 , g _ { 2 } { \mathrm { i f } } \phi = 2$ </td></tr><tr><td>Success</td><td> $\mathbf { 1 } \{ s _ { T } = g _ { \phi } \}$ </td><td>reach preferred goal</td></tr><tr><td>Horizon</td><td> $T$ </td><td> $2$ </td></tr><tr><td>Candidate set</td><td> $\mathcal C ( s _ { t } )$ </td><td> $\mathrm { a l l }$  states except current state</td></tr><tr><td>Transition</td><td> $s _ { t + 1 }$ </td><td> $\tilde { s } _ { t }$  if accepted,  $s _ { t }$  if rejected</td></tr><tr><td>Value at start</td><td> $V _ { \phi } ( s _ { 0 } )$ </td><td>0</td></tr><tr><td>Matched probe value</td><td> $V _ { \phi } ( p _ { \phi } )$ </td><td> ${ w _ { p , + } } = 1 . 0$ </td></tr><tr><td>Mismatched probe value</td><td> $V _ { \phi } ( p _ { k } ) , k \neq \phi$ </td><td> $w _ { p , - } = - 3 .$  0 default</td></tr><tr><td>Matched goal value</td><td> $V _ { \phi } ( g _ { \phi } )$ </td><td> $w _ { g , + } = 5 . 0$ </td></tr><tr><td>Mismatched goal value</td><td> $V _ { \phi } ( g _ { k } ) , k \ne \phi$ </td><td> $w _ { g , - } = 4 . 0$ </td></tr><tr><td>Distance transform</td><td> $g$ </td><td>main:  $d ^ { 2 } i$  linear ablation: d</td></tr><tr><td>Default evaluability slope</td><td> $\rho _ { \mathrm { t r u e } }$ </td><td>0.5</td></tr><tr><td>Default response sharpness</td><td> $\kappa _ { \mathrm { t r u e } }$ </td><td>1.0</td></tr><tr><td>Evaluability range</td><td> $\rho _ { \mathrm { t r u e } }$ </td><td>{0.2, 0.5, 0.8, 1.0}</td></tr><tr><td>Response-sharpness range</td><td> $\kappa _ { \mathrm { t r u e } }$ </td><td>{0.3, 0.7, 1.0, 2.0}</td></tr><tr><td>Preference prior</td><td> $p ( \phi )$ </td><td>uniform over two arms</td></tr><tr><td>Belief grid</td><td> $\rho$ </td><td>21 linearly spaced points in [0.01, 1.0]</td></tr><tr><td>Belief grid</td><td>κ</td><td>8 geometrically spaced points in [0.3, 4.0]</td></tr></table>

![](images/aa8812d776d4221069518445b0b9dc04c21988c0e899c71d390e7a329aed638c.jpg)  
Figure $_ { 6 ; }$ Sensitivity to probe diagnosticity in the Probe-Commit task. We vary the wrong-probe value $w _ { p , - } ,$ where more negative values make wrong-branch probes easier to reject and therefore more informative about $\phi .$ PROSE-PLAN probes only once probe responses become diagnostic, while myopic and belief-frozen depth-2 planners continue to commit directly.

## E.3.3 Belief Updates Induced by Probes

We further check whether probing actually provides information about the hidden preference. For episodes in which PROSE-PLAN probes at $t = 0$ , Table 8 reports the posterior change after observing the first response. The prior over ϕ is uniform, so before the first response the assistant assigns probability 0.5 to the true branch.

When $w _ { p , - } = - 1$ , PROSE-PLAN does not probe, so probe-conditioned posterior statistics are undefined. For $w _ { p , - } \leq - 2$ , the first response to a probe substantially concentrates the posterior, which confirms that probes selected by PROSE-PLAN do provide information that can guide the second-step commit.

Table 8: Belief update induced by first-step probes in the Probe-Commit task. Rows condition on PROSE-PLAN proposals at $t = 0 ;$ when $w _ { p , - } = - 1$ , PROSE-PLAN does not probe, so probeconditioned quantities are not defined. Entropy reduction is $H ( b _ { 0 } ) - H ( b _ { 1 } )$ , and MAP correctness is $\mathbf { 1 } \{ \arg \operatorname* { m a x } _ { \phi } { \bar { b } } _ { 1 } ( \phi ) = \phi _ { \mathrm { t r u e } } \}$ . Values are mean ± standard error over first-step probe episodes.
<table><tr><td> $w _ { p , - }$ </td><td>Probe at  $t = 0$ </td><td>Probe n</td><td> $H ( b _ { 0 } ) - H ( b _ { 1 } )$ </td><td>MAP correct</td><td> $b _ { 1 } \big ( \phi _ { \mathrm { t r u e } } \big )$ </td></tr><tr><td>-1</td><td>0.000</td><td>0</td><td>1</td><td></td><td>1</td></tr><tr><td>-2</td><td>1.000</td><td>200</td><td> $0 . 2 5 8 \pm 0 . 0 1 1$ </td><td>一  $0 . 8 1 0 \pm 0 . 0 2 8$ </td><td> $0 . 6 9 1 \pm 0 . 0 1 7$ </td></tr><tr><td>-3</td><td>1.000</td><td>200</td><td> $0 . 2 8 6 \pm 0 . 0 1 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 2 6$ </td><td> $0 . 7 2 9 \pm 0 . 0 1 7$ </td></tr><tr><td>-4</td><td>1.000</td><td>200</td><td> $0 . 3 1 0 \pm 0 . 0 1 5$ </td><td> $0 . 8 1 0 \pm 0 . 0 2 8$ </td><td> $0 . 7 2 9 \pm 0 . 0 1 8$ </td></tr></table>

![](images/74a61ad4a81713c49cfc370f87bde1f164252b66a2ce54d950a1b04c9a1162d1.jpg)  
Figure 7: Same-seed Probe-Commit proposal trajectories under the default condition $( w _ { p , - } =$ $- 3 , \rho _ { \mathrm { t r u e } } = 0 . 5 , \kappa _ { \mathrm { t r u e } } = 1 . 0 )$ . The selected episode uses seed 88, true branch BR1, and is automatically chosen from seeds matching the main qualitative gap. Dark borders indicate accepted proposals; bold labels mark proposals on the preferred branch. PROSE-PLAN uses a rejected probe to identify the preferred branch before committing, while myopic and belief-frozen planners commit to the wrong goal and fail to recover.

## E.3.4 Illustrative Proposal Trajectories

Figure 7 shows selected proposal trajectories under the default Probe-Commit condition $( w _ { p , - } =$ $- 3 , \rho _ { \mathrm { t r u e } } = 0 . 5 , \kappa _ { \mathrm { t r u e } } = 1 . 0 )$ for intuitions on the assistance pattern of each planner. The trajectories show why direct commitment can fail even when the first proposal is accepted. Both PERSONALISED-MYOPIC and BELIEF-FROZEN DEPTH-2 first commit to the wrong goal $g _ { 2 }$ , which can be accepted because the wrong goal still has high value $w _ { g , - } = 4$ . However, once the realised state moves to $g _ { 2 } .$ correcting to the preferred goal $g _ { 1 }$ requires a long-distance proposal, and under high evaluation cost, this correction is hard to evaluate and is likely to be rejected, so both planners remain at the wrong goal and fail.

PROSE-PLAN behaves differently in that it first proposes the wrong-branch probe $p _ { 2 } .$ , which is rejected but informative about $\phi .$ After updating its belief toward $\phi = 1$ , PROSE-PLAN proposes $g _ { 1 } .$ which is now both preference-aligned and reachable from the start state, and the episode succeeds. The oracle commits directly to $g _ { 1 }$ because it already knows the preferred branch. Thus, the trajectory illustrates the mechanism isolated by the Probe-Commit task, which is that the advantage of PROSE-PLAN comes from using the first response to change the second proposal, rather than from depth-2 lookahead alone.