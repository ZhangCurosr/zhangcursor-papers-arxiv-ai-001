# AI agents in Algorithmic Electricity Markets: On the Emergence of Tacit Collusion

Jakub Seredynski, Georgios Tsaousoglou´

Abstract—As electricity market participants increasingly adopt learning-based agents for their bidding strategies, electricity markets are becoming algorithmic. Evidence from algorithmic markets in other domains shows that tacit collusion can arise purely through independent learning. Moreover, electricity markets are typically oligopolistic and feature repeated interaction among a small number of participants, making them structurally susceptible to non-competitive behavior. In the face of these observations, this paper investigates the hypothesis that tacit collusion may emerge in electricity markets where participants’ actions are controlled by autonomous learning-based algorithms. We model strategic bidding as a repeated game with imperfect public monitoring, and model the participants’ emergent behavior using multi-agent reinforcement learning. We propose a multidimensional set of criteria (going beyond profit comparisons against Nash equilibria) to assess whether the resulting behavior constitutes tacit collusion. Our experimental results showcase that such a danger is realistic for electricity markets: there are cases where agents do learn to sustain supra-competitive outcomes that are supportive of tacit collusion indicators, even though the agents were never instructed to collude.

Index Terms—Electricity Markets, Multi-agent Reinforcement Learning, Tacit Collusion, Strategic Behavior.

## I. INTRODUCTION

## A. Motivation and Research Question

towards avoiding competition and increasing their profits. It is severely detrimental, not only to consumers who face supra-competitive prices, but also to the market’s economic efficiency in general. While collusion is associated with cartellike practices and is thereby strictly prohibited by law, the more challenging case to deal with is when collusion is tacit. Definition: Tacit Collusion refers to a group of oligopolists ability to coordinate, even in the absence of explicit agreement, in order to raise prices or, more generally, increase profits at the detriment of consumers [1]. □

Importantly, the absence of a need for an explicit agreement makes it possible for Tacit Collusion to emerge “autonomously”. This is directly relevant to the new reality of electricity markets where participants are increasingly outsourcing their market participation strategies to autonomous learning-based algorithms, generally referred to as Artificial Intelligence (AI) agents. Experience from other domains has already established that AI agents, particularly Reinforcement Learning (RL), often exhibit the ability to maximize their prescribed reward (read: profit) in ways unforeseen by their designers. The study in [2] demonstrates that independently designed reinforcement-learning pricing algorithms can indeed learn to sustain supracompetitive prices, in general contexts. In the authors’ own words: “We find that the algorithms consistently learn to charge supracompetitive prices, without communicating with one another [...] The algorithms learn these strategies purely by trial and error. They are not designed or instructed to collude, they do not communicate with one another, and they have no prior knowledge of the environment”.

Taken to electricity markets, such agents could give rise to Tacit Collusion phenomena, even without the firms’ intention or awareness. Moreover, the fact that electricity markets historically tend to be oligopolistic while also featuring repeated interaction among participants, makes them particularly vulnerable to Tacit Collusion emergence. Motivated by the possibility and implications of this phenomenon in electricity markets, we put forward the following hypothesis:

Hypothesis: In an electricity market where the participants’ actions are controlled by autonomous learning-based algorithms, Tacit Collusion may emerge.

## B. Related Work

An electricity market comprises participants, each of which is interested in optimizing its own profit. Thereby, studying possibilities over electricity market outcomes involves modeling the bidding strategy optimization problem of individual participants, in an agent-based modeling fashion [3]. In the earlier literature, a participant’s bidding strategy was predominantly modeled through bi-level programming (e.g. [4]), giving rise to mixed-complementarity problems modeling the multiparticipant interactive decision-making [5], [6]. The recent literature, however, has substantially pivoted toward learningbased approaches, for at least three reasons:

• Learning algorithms are able to handle realistic aspects of electricity markets such as complexity, partial observability, and stochasticity, departing from simplifying assumptions arising from computational constraints of the model [7].

• The model-free nature of learning algorithms makes them a generic modeling tool which, taken to a sufficient level of richness, can in principle reproduce a wide range of different intelligent strategies [8], [9].

• Real market participants increasingly experiment with learning-based bidding strategies [10]. This motivates the

RL framework, in particular, to be leveraged as a suitable modeling framework that captures the exploration aspect of modern algorithmic trading.

RL has attracted special interest as a generic model for a participant’s bidding strategy optimization problem [11], [12], [13]. From a system perspective, when it comes to modeling the interactive learning among several strategic participants and the emergent outcomes thereof, the concept naturally evolves into multi-agent RL (MARL) [14]. To that end, [15] argues for MARL-based fit-for-purpose simulators that retain the market and physical features relevant to the research question at hand, while [16] formulates day-ahead bidding with multiple strategic generators as algorithmic traders and applies a MARL algorithm to approximate Nash-equilibrium bidding. Recent work on generator modeling further shows that simplified bidding and marginal-cost representations can affect conclusions in market-mechanism simulations, especially when renewable integration changes the operating regimes of thermal generators [17].

The potential emergence of Tacit Collusion has been validated by research in other domains, namely algorithmic pricing, where RL agents have been shown to sustain supracompetitive prices without communication or explicit collusive instructions [2]. This result has been extended to sequential pricing environments, where Q-learning algorithms may converge either to collusive equilibria or to supra-competitive price cycles [18]. Recent experimental evidence further compares algorithmic and human collusion in the same market environments, showing that self-learning pricing algorithms can generate more collusive outcomes than human decisionmakers in some oligopoly settings [19].

Taken to the electricity-market setting, it is particularly timely to understand the implications of algorithmic trading and, in particular, whether Tacit Collusion can emerge endogenously from the interaction of autonomous learning agents. To that end, a MARL agent-based model is employed in [3], where the authors investigate how the agents’ discount factors affect the resulting market outcomes. More recently, [20] explicitly investigates the emergence of Tacit Collusion and define a collusive state as one in which the profit of each participant exceeds that attained under all Nash equilibria. While this provides a useful benchmark for identifying supracompetitive outcomes, establishing such a condition requires the computation of the Nash equilibria of the underlying market game. This requirement substantially constrains the complexity of the market model and makes the approach difficult to extend to richer electricity-market settings.

## C. Research Gap and Contributions

The emergence of algorithmic trading, and the risks it poses for electricity markets, points to a major research need around the emergence of Tacit Collusion. To our knowledge, this phenomenon has been addressed by only a handful of studies (namely [20] and [3]). These studies provide important evidence that intelligent agents may learn supra-competitive bidding strategies. Their identification of Tacit Collusion, however, is closely tied to the structure of the underlying market game. In particular, [20] identifies collusion by comparing the agents’ profits with those attained at all Nash equilibria. Such an approach is informative in a stylized market model whose equilibria can be fully characterized, but becomes difficult to apply once the market model incorporates the structural features of actual electricity markets.

More fundamentally, supra-competitive profits alone do not establish Tacit Collusion: elevated profits can arise for reasons unrelated to collusive coordination. Conversely, because realistic market models rarely converge to a stage-game Nash equilibrium, the absence of such convergence is, on its own, equally uninformative about whether collusion has occurred. In fact, in our numerical results in this paper, we present examples on both ends: cases where supra-competitive outcomes emerge which are, or are not necessarily, supportive of a case for Tacit Collusion. Tacit Collusion is thus a subtle phenomenon that resists identification through any single criterion and calls instead for a multi-dimensional examination. This motivates the contributions of this paper, positioned towards investigating the Hypothesis defined in Section I-A:

1) We adopt Tacit Collusion criteria (beyond merely supracompetitive outcomes) from other domains, with richer experience in this context, and adapt them to the electricity market setting. We also propose a novel criterion which quantifies the central characteristic of collusion as defined in its Definition (cf Section I-A), in the context of electricity markets.

2) We model emergent bidding profiles using a MARL framework and examine their collusive properties across multiple such criteria.

3) We experimentally investigate which market characteristics (e.g. grid constraints’ tightness, demand level, marginal costs’ variance) enhance the possibility of emergent collusion.

The remainder of the paper is organized as follows. Section II presents the system model. Section III defines the different Tacit Collusion criteria of our study. Section IV presents the experimental setting, including the details of the MARL algorithm. Section V presents our experimental results, while Section VI concludes the paper.

## II. SYSTEM MODEL

## A. Electricity Market

We consider a single-timeslot electricity market, operating over a set N of buses, which is cleared by solving the standard DC optimal power flow (OPF) problem. Specifically, a bus $n \in \mathcal N$ features an inelastic demand $\mathrm { D } _ { n }$ and a set $\mathcal { I } _ { n }$ of energy-providing resources. The resources can generally be of various types and technologies (including conventional power plants, renewable energy sources, storage, flexible demand, etc) but we will simply call them “generators” to simplify the exposition. We denote the superset of generators (across all nodes) by $\textstyle { \mathcal { T } } = \bigcup _ { n \in { \mathcal { N } } } { \mathcal { T } } _ { n }$

The active power flow from bus n to bus m is denoted as $f _ { n m }$ and is subject to line capacity bounds per

$$
- \mathrm { F } _ { n m } \le f _ { n m } \le \mathrm { F } _ { n m } , \quad \forall ( n , m ) \in \mathcal { N } ^ { 2 } ,\tag{1}
$$

where, for pairs of buses that are not connected by a line, we simply set $\mathrm { F } _ { n m } = 0$ . Denoting the output of a generator $i \in \mathcal { Z } _ { n }$ as $q _ { i }$ , the power balance constraint at a node reads:

$$
\sum _ { i \in \mathbb { Z } _ { n } } q _ { i } - \sum _ { m \in \mathcal { N } } f _ { n m } = \mathrm { D } _ { n } , \quad \forall n \in \mathcal { N } .\tag{2}
$$

The phase angle at a bus n is denoted as $\phi _ { n }$ and is subject to safety limits per

$$
\underline { { { \phi } } } _ { n } \leq \phi _ { n } \leq \overline { { { \phi } } } _ { n } , \quad \forall n \in \mathcal { N } .\tag{3}
$$

The phase angles and the power flows are linked by the standard DC power flow model:

$$
f _ { n m } = \mathrm { B } _ { n m } ( \phi _ { n } - \phi _ { m } ) , \quad \forall ( n , m ) \in \mathcal { N } ^ { 2 } ,\tag{4}
$$

where $\mathrm { B } _ { n m }$ is the susceptance of transmission line nm.

The output of a generator $i \in \mathcal { T }$ is partitioned into a set $s _ { i }$ of segments, as in:

$$
q _ { i } = \sum _ { s \in S _ { i } } q _ { i , s } , \quad \forall i \in \mathcal { T } ,\tag{5}
$$

where the output $q _ { i , s }$ of a segment is bounded by

$$
0 \leq q _ { i , s } \leq \mathrm { Q } _ { s } , \quad \forall s \in S _ { i } , i \in \mathcal { I } .\tag{6}
$$

Finally, each generator declares a per-unit energy cost $\mathrm { c } _ { i , s }$ for each of its segments $s \in S _ { i }$ , such that the operator receives a piecewise linear cost function for each generator. The market is cleared by solving the standard DC-OPF problem:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \boldsymbol { q } _ { i } , \boldsymbol { q } _ { i } , \boldsymbol { s } , f _ { n m } , \boldsymbol { \phi } _ { n } } \left\{ \sum _ { i \in \mathcal { T } } \sum _ { \boldsymbol { s } \in S _ { i } } \mathbf { c } _ { i , \boldsymbol { s } } \boldsymbol { q } _ { i , \boldsymbol { s } } \right\} } \\ { \displaystyle \quad \mathrm { s u b j e c t ~ t o } } \\ { \displaystyle \quad ( 1 ) - ( 4 ) : \mathrm { N e t w o r k ~ c o n s t r a i n t s } } \\ { \displaystyle \quad ( 5 ) - ( 6 ) : \mathrm { G e n e r a t o r s } ^ { \prime } \mathrm { ~ c o n s t r a i n t s } . } \end{array}\tag{7}
$$

The optimal dual variable $\lambda _ { n }$ of the power balance constraint (2) instantiates the Locational Marginal Price (LMP) of node n. The market-clearing problem (7) is implemented sequentially for a set of instances (timeslots) $\tau$ where in each instance $t \in \mathcal T$ the generators can submit different bids $\mathrm { c } _ { i , s } [ t ]$ resulting in timeslot-specific solutions $( ( q _ { i } ^ { * } [ t ] ) _ { i \in \mathcal { T } } , ( \lambda _ { n } [ t ] ) _ { n \in \mathcal { N } } )$

## B. Agent-based Market Participation Strategies

A generator’s market participation strategy is instantiated by an agent. We use the terms “agent” and “generator” interchangeably and we index either by i. After each marketclearing instance $t ,$ the stage profit $\pi _ { i } [ t ]$ of i is

$$
\pi _ { i } [ t ] = \lambda _ { n ( i ) } [ t ] q _ { i } ^ { * } [ t ] - C _ { i } ( q _ { i } ^ { * } [ t ] ) .\tag{8}
$$

The first term is the generator’s revenue, with $\lambda _ { n ( i ) } [ t ]$ being the LMP at bus $n ( i )$ where i is located. The second term is the generator’s real cost for generating its dispatched quantity $q _ { i } ^ { * } [ t ]$ , which is given by the generator’s cost function $C _ { i } : \mathbb { R }  \mathbb { R }$

The values for $q _ { i } ^ { * } [ t ]$ and $\lambda _ { n ( i ) } [ t ]$ are given from problem (7), and are therefore affected by the actions, i.e. bids, $( \mathrm { c } _ { i , s } [ t ] ) _ { s \in \mathcal { S } _ { i } }$ of the focal agent i, as well as by those of the other agents. This directly leads to the definition of the stage game $\mathcal { G } \colon$

• Players: $i \in \mathcal { T }$

• Actions: $\pmb { c } _ { i } [ t ] = ( \mathrm { c } _ { i , s } [ t ] ) _ { s \in S _ { i } }$

• Payoffs: $\pi _ { i } [ t ]$ as in Eq. (8)

We consider agents that learn to improve their strategies through experience. After each instance, an agent gets to observe its own dispatch and profit, as well as the resulting LMP at every node of the network:

$$
o _ { i } [ t ] = \big ( q _ { i } ^ { * } [ t ] , \pi _ { i } [ t ] , ( \lambda _ { n } [ t ] ) _ { n \in \mathcal { N } } \big ) .\tag{9}
$$

This gives rise to a repeated game with imperfect monitoring, since an agent does not observe the actions or profits of other agents but only a public signal (i.e. the LMPs) from which the others’ actions and profits cannot be inferred. Furthermore, we do not assume that agents know the structure of the network or even the market mechanism and consider agents that learn purely through experience. To that end, each agent accumulates a private history of its actions and observations (up to ζ episodes behind):

$$
h _ { i } [ t ] = \left( c _ { i } [ t - 1 ] , o _ { i } [ t - 1 ] , c _ { i } [ t - 2 ] , o _ { i } [ t - 2 ] , . . . , c _ { i } [ t - \zeta ] , o _ { i } [ t - \zeta ] \right)\tag{10}
$$

and selects its bid at each stage t according to a strategy $\sigma _ { i }$ that maps the current history $h _ { i } [ t ]$ to its decision $\mathbf { } c _ { i } [ t ]$ . We denote the joint strategy as ${ \pmb { \sigma } } = ( \sigma _ { i } ) _ { i \in { \pmb { \mathbb { Z } } } }$ . Based on the above, we can define the objective of an agent as a search for a strategy that maximizes its cumulative profits over time:

$$
\operatorname* { m a x } _ { \sigma _ { i } } \left\{ \mathbb { E } _ { \sigma } \sum _ { t = 0 } ^ { \infty } \pi _ { i } [ t ] \right\}\tag{11}
$$

subject to

(8) : Profits depend on actions,

$\pmb { c } _ { i } [ t ] = \sigma _ { i } ( h _ { i } [ t ] )$ : Actions are given by strategy and history

(10) : The history stacks part actions and observations

(9) : Observations are: dispatch, profit, and LMPs.

Problem (11) is intractable in general: each agent optimizes its own strategy while the strategies of all other agents are unknown and simultaneously evolving, making the environment non-stationary from any single agent’s perspective. In line with the learning-through-experience consideration, we adopt Multi-Agent Reinforcement Learning (MARL) as a model of how AI agents learn and play in practice. At each instance t, every agent i selects a bid $\pmb { c } _ { i } [ t ] = \sigma _ { i } ( h _ { i } [ t ] )$ , the market clears, and each agent receives the observation $o _ { i } [ t ]$ and uses it to update its strategy $\sigma _ { i }$ . This process is iterated over successive instances, and the agents’ strategies co-evolve. The details of the MARL algorithm will be presented in Section IV-C.

The output of this process is a joint strategy profile $\pmb { \sigma } ^ { \ast } =$ $( \sigma _ { i } ^ { * } ) _ { i \in \mathbb { Z } } .$ , representing the emergent bidding behaviour that arises when self-interested agents learn to bid. The paper’s Hypothesis thereby refers to analyzing the emergent joint strategies $\sigma ^ { * }$ in terms of indications for Tacit Collusion. The next section formalizes the Tacit Collusion indices to be used.

## III. TACIT COLLUSION

In this section, we define three criteria that constitute indicators of the emergent joint strategy $\pmb { \sigma } ^ { * }$ being tacitly collusive: Punishment of Deviation, Unsustainability under Shortsight, and Short-term Profitability of Deviations. In the three subsection below, we describe how each criterion is instantiated in our electricity market context.

## A. Punishment of Deviation Criterion

This criterion checks whether an agent that deviates from the joint strategy $\pmb { \sigma } ^ { * }$ and reverts to an explicitly competitive strategy will face a punitive response by other agents.

A deviator reverting to a competitive strategy means that it starts to optimize its short-term profits aggressively. In our context, we can model such a situation by choosing one agent j as the deviator and, after the MARL strategies have converged to $\sigma ^ { * }$ , switching the deviator’s strategy to the one that optimizes its stage profit under the assumption that other agents $i \neq j$ continue to play their previous strategies (specifically, their most recently observed ones). To implement this, at each after-convergence stage t, we force $j ^ { \circ } \mathbf { s }$ strategy to the solution of the following bi-level optimization problem:

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { \boldsymbol { c } _ { j } [ t ] } \left\{ \lambda _ { n ( j ) } [ t ] \boldsymbol { q } _ { j } ^ { * } [ t ] - C _ { j } \big ( \boldsymbol { q } _ { j } ^ { * } [ t ] \big ) \right\} } \\ { \mathrm { s u b j e c t ~ t o } } \\ { \displaystyle q _ { j } ^ { * } [ t ] , \lambda _ { n ( j ) } \in ( 7 ) } \end{array}\tag{12}
$$

where the constraint specifies that the dispatch and LMP result through the market clearing problem (7). Problem (12) can be solved by casting the lower-level problem (7) as a set of KKT conditions and introducing auxiliary binary variables to handle the resulting complementarity constraints. This results in a single-level mixed-integer linear program which can be tackled by commercial solvers. The full derivation is ommited here as it is fairly standard in the literature.

While the deviator is using its aggressive profit-maximizing strategy (12), the other agents continue to play by the MARL regime. A punitive response is considered to be one that has other agents lowering their mark-ups in a way that reduces the deviator’s profit. The test is considered positive if the postdeviation dynamics display the qualitative pattern of punishand-forgive strategies:

• Immediately after the deviation, competitors temporarily switch to more competitive (i.e. lower) bids, reducing prices and the deviator’s market share;

• After a punishment phase, competitors gradually return to their pre-deviation strategies, while prices recover.

## B. Unsustainability under Shortsight Criterion

Two elements that generally facilitate Tacit Collusion are:

• agents playing the long game, by adopting a high discount factor which increases the importance of future payoffs over immediate ones;

• agents having memory of past actions and rewards, which allows them to condition their behaviour on history;

The rationale of this criterion is that reducing or eliminating these elements should make it substantially more difficult for agents to learn coordinated strategies. In practice, this test is implemented by executing the MARL algorithm under a very low discount factor while also severely truncating the memory of past actions and rewards. Compared to the baseline case, this run would presumably result in more competitive strategies, i.e. lower prices and profits.

## C. Short-term Profitability of Deviations Criterion

Given that collusive outcomes are not Nash equilibria of the stage game, one or more agents do have a profitable unilateral strategy deviation. Yet the collusion is sustained when agents refrain from switching to opportunistic strategies, despite this being profitable in the short-run, presumably because they understand that those would lead to down-the-road less profitable outcomes. This criterion checks the emergent joint strategy $\sigma ^ { * }$ precisely for these two conditions: agents having profitable unrealized deviations from it, and whether those, if realized, would trigger joint adaptation dynamics that indeed lead to less profitable trajectories.

Similarly to the first criterion, we instantiate profitable deviations by solving the bi-level problem (12). This time, however, we solve problem (12) iteratively and for every agent in turn, thereby simulating a relapse into competitive (noncollusive) strategy adaptation. The relevant algorithm is called “Iterative Best Response” (IBR) or “diagonalization” and it is widely used in the electricity markets’ literature (see e.g. [21] and references therein) to approximate Nash Equilibria under the so-called Equilibrium-Problem with Equilibrium-Constraints (EPEC) model. The criterion considers that $\sigma ^ { * }$ passes the collusion test if, upon switching from MARL to the IBR regime, the profits of the first deviating agents:

• increase at first, and

• they eventually drop below those under $\sigma ^ { * }$ , as the IBR process continues towards a Nash Equilbirium.

The rationale is that, if both of these conditions hold true, it means that each agent has a profitable unilateral deviation which it doesn’t realize, possibly because it foresees that it is not profitable under the other agents’ corresponding competitive response, thereby staying true to a coordinated (collusive) strategy.

## IV. EXPERIMENTAL SETTING

## A. Electricity Market Setup

We conduct the experiments on a stylised seven-bus electricity market adapted from the case study of [20]. The original system is an extended version of the well-known Pennsylvania–New-Jersey–Maryland (PJM) five-node test system and was designed as a compact networked market in which strong collusive equilibria can arise. We use it here as a controlled benchmark that preserves the elements needed for our hypothesis: an oligopolistic supply side, inelastic nodal demand, transmission constraints, and locational marginal prices.

The market contains four strategic generators located at buses $N _ { 1 } , N _ { 2 } , N _ { 5 }$ , and $N _ { 6 }$ . Nodal demand is perfectly inelastic and the market is cleared, at each instance, by the DC-OPF problem (7). The resulting dispatch and nodal prices determine each generator’s stage profit according to (8), where revenues are computed at the LMP of the generator’s bus and costs are computed using the generator’s true marginal cost. The main generation and cost parameters used in the experiments are reported in Table I.

TABLE I  
STRATEGIC GENERATORS AND MARGINAL-COST CASES.
<table><tr><td>Generator</td><td>Bus</td><td> $P _ { i } ^ { \mathrm { m a x } }$  [MW]</td><td colspan="3">Marginal cost [EUR/MWh]</td></tr><tr><td></td><td></td><td></td><td>Low</td><td>Medium</td><td> $\mathrm { H i g h }$ </td></tr><tr><td> $G _ { 1 }$ </td><td> $N _ { 1 }$ </td><td>42</td><td>20</td><td>20</td><td>20</td></tr><tr><td> $G _ { 2 }$ </td><td> $N _ { 2 }$ </td><td>35</td><td>20</td><td>20</td><td>20</td></tr><tr><td> $G _ { 5 }$ </td><td> $N _ { 5 }$ </td><td>40</td><td>20</td><td>30</td><td>40</td></tr><tr><td> $G _ { 6 }$ </td><td> $N _ { 6 }$ </td><td>40</td><td>20</td><td>10</td><td>1</td></tr></table>

All exogenous fundamentals are kept fixed within a scenario. In particular, nodal demand, marginal costs, and network parameters do not vary over time. The three demand cases, indexed by buses $( N _ { 1 } , \ldots , N _ { 7 } )$ , are $\begin{array} { r l } { D ^ { L } } & { { } = } \end{array}$ (17, 10, 8, 6, 13, 0, 12), $D ^ { M } \ = \ ( 2 2 , 1 4 , 1 0 , 8 , 1 8 , 0 , 1 3 )$ , and $D ^ { H } = ( 2 6 , 1 5 , 1 2 , 9 , 2 0 , 0 , 1 4 )$ . Thus, the temporal dynamics observed in the simulations are induced by the repeated interaction of bidding strategies and by the feedback of the market-clearing mechanism, rather than by exogenous factors.

To isolate the role of transmission constraints, we consider two network representations. In the Grid case, the network constraints (1)–(4) are enforced using the topology and line limits of the test system. In the NoGrid case, the same demand and generator data are retained, but line capacities are set sufficiently high so that the bounds in (1) never become binding, while the corresponding admittance values are chosen so that the DC flow relation (4) and phase-angle bounds (3) do not restrict the optimal dispatch. Operationally, this produces a copper-plate counterfactual. Mathematically, for all solved instances, the NoGrid case is equivalent to relaxing the network-induced restrictions in (1), (3), and (4), so that problem (7) clears the market as a single unconstrained zone.

The experimental design is a full factorial combination of three axes: network representation $\mathtt { G r i d / N o G r i d } ,$ demand level Low/Medium/High, and marginal-cost heterogeneity Low/Medium/High. This gives $2 \times 3 \times 3 = 1 8$ market environments. The selected scenario dimensions correspond to market characteristics that are widely discussed in the Tacit Collusion literature as factors influencing the emergence and sustainability of collusive outcomes, namely network topology, demand conditions, and cost heterogeneity [1], [22]. All scenarios are trained and screened, while the behavioural collusion tests are applied only to the most informative cases selected according to the procedure described next.

## B. Scenario Selection Methodology

Instead of running our tests in randomly selected market instances, we identify instances that are most suspicious for potential Tacit Collusion emergence. In this subsection, we describe the methodology that we use to select which market instances will be tested for the criteria of Section III.

Our starting point is the 18 scenarios of the full factorial design described in the previous subsection. For each of them, we compute screening indicators over the final evaluation window of each training run, after exploratory noise has decayed. The competitive benchmark is obtained by solving the same market-clearing problem (7) under marginal-cost bidding, i.e. with all generators bidding their true marginal costs. Denoting the average realised price and total profit by $\bar { \lambda } ^ { o b s }$ and $\Pi ^ { o b s }$ , and their competitive counterparts by $\bar { \lambda } ^ { c o m p }$ and Π<sup>comp</sup>, we define

TABLE II  
SCENARIOS SELECTED FOR BEHAVIOURAL VALIDATION.
<table><tr><td>Scenario</td><td>Demand</td><td>Cost diff.</td><td>Rationale</td></tr><tr><td>Grid-LowDemand- HighCostDiff</td><td>Low</td><td>High</td><td>Structured non-stationary regime with non-trivial de-</td></tr><tr><td>Grid-LowDemand-</td><td>Low</td><td>Medium</td><td>viation incentives. Reference low-demand</td></tr><tr><td>MediumCostDiff Grid-MediumDemand-</td><td>Medium</td><td>High</td><td>case with a more stationary learned profile. Higher-demand case that</td></tr></table>

$$
M ^ { \lambda } = \frac { \bar { \lambda } ^ { o b s } - \bar { \lambda } ^ { c o m p } } { \bar { \lambda } ^ { c o m p } } ,\tag{13}
$$

$$
{ \cal M } ^ { \Pi } = \frac { \Pi ^ { o b s } - \Pi ^ { c o m p } } { \operatorname* { m a x } \{ \Pi ^ { c o m p } , \epsilon _ { \Pi } \} } ,\tag{14}
$$

where $\textstyle \Pi = \sum _ { i \in I } \pi _ { i }$ denotes aggregate generator profit and $\epsilon _ { \Pi }$ is a small denominator floor used to avoid degenerate ratios when competitive profits are close to zero.

To separate high-price outcomes from coordinated behaviour, the markup indicators are complemented by three dynamic indicators. First, price variance measures whether the learned market regime is stable enough for an intervention test to be meaningful. Second, action correlation and action-change correlation measure, respectively, whether agents maintain similar bidding levels or move their bidding strategies together over time. Third, Best Response Deviation Gain (BRDG) measures the short-run profitability of unilateral deviations. For agent i, it is defined as

$$
\mathrm { B R D G } _ { i } = \frac { \pi _ { i } ( c _ { i } ^ { B R } , \bar { c } _ { - i } ) - \pi _ { i } ( \bar { c } _ { i } , \bar { c } _ { - i } ) } { \pi _ { i } ( \bar { c } _ { i } , \bar { c } _ { - i } ) } ,\tag{15}
$$

where $\bar { c } _ { i }$ denotes the bid prescribed by the learned policy of agent $i , \ \bar { c } _ { - i }$ denotes the learned bids of all other agents, and $\mathbf { \bar { \Phi } } _ { c _ { i } ^ { B R } }$ is the one-shot best response to $\bar { c } _ { - i }$ . A positive BRDG therefore indicates that the learned profile contains an unrealised short-run deviation incentive.

The selection favours cases that satisfy three practical requirements: (i) the learned regime is sufficiently stable to be meaningfully perturbed, (ii) the outcome is not a degenerate boundary solution in which all agents simply bid at the action-space limit, and (iii) there is a non-trivial unilateral deviation incentive that can be tested through the criteria of Section III. The selected cases are then evaluated using the Punishment of Deviation Criterion, the Unsustainability under Shortsight Criterion, and the Short-term Profitability of Deviations Criterion described in Section III.

## C. MARL implementation

The action of agent i at each instance t is to select a bid vector $\pmb { c } _ { i } [ t ] = ( c _ { i , s } [ t ] ) _ { s \in S _ { i } }$ , i.e. one declared marginal cost per segment s. This renders the action space of each agent to be of dimension |S<sub>i</sub>|. To reduce this dimensionality and facilitate learning, we parameterize the bid vector through a supply function. Specifically, each agent i maintains a linear supply function

$$
f ( q _ { i } ; \beta _ { i } ) = \alpha _ { i } + \beta _ { i } \cdot \frac { q _ { i } } { \sum _ { s \in S _ { i } } \mathrm { Q } _ { s } } ,\tag{16}
$$

which maps an output level $q _ { i , s }$ to a declared marginal cost (bid) $\mathrm { c } _ { i , s }$ . Note that this function is generally different than the agent’s true cost function (defined in the previous subsection); it is an auxiliary bidding curve used internally to generate a consistent set of segment bids. The bid $\mathrm { c } _ { i , s }$ for segment s is obtained by evaluating f at the midpoint $\begin{array} { r } { \bar { q } _ { i , s } = \sum _ { s ^ { \prime } \in \{ 0 , 1 , \ldots s - 1 \} } Q _ { s ^ { \prime } } + \frac { \mathrm { Q } _ { s } } { 2 } } \end{array}$ of the segment’s capacity range:

$$
c _ { i , s } [ t ] = f _ { i } ( \bar { q } _ { i , s } ; \beta _ { i } ) = \alpha _ { i } + \beta _ { i } \cdot \frac { \bar { q } _ { i , s } } { \sum _ { s \in S _ { i } } \mathrm { Q } _ { s } } , \quad s \in \mathcal { S } _ { i } ,\tag{17}
$$

where $\alpha _ { i }$ is set to the agent’s true marginal cost and $\beta _ { i }$ is the parameter to be learned by the agent. This defines a deterministic mapping $\beta _ { i } [ t ] \mapsto c _ { i } [ t ]$ , so that the full bid vector is determined by a single scalar parameter. The agent’s internal action at each instance t is therefore the choice of $\beta _ { i } [ t ] \in \mathbb { R } _ { \geq 0 }$ which is then translated into the game action $c _ { i } [ t ]$ via (17).

Remark 1: This parameterization reduces the dimensionality of an agent’s action space from |S | to 1. This is a choice that only makes the validation of our Hypothesis more difficult: by reducing the agents’ degrees of freedom, we reduce the richness of their policies, possibly excluding instances of collusive joint policies. Thus, if collusive policies are not discovered with this model, it does not necessarily mean that they don’t exist. But if collusive policies are discovered with this model, then the validation of the Hypothesis holds also for the general model where each agent’s action comprises an arbitrary choice for each segment’s bid. □

In the experiments, the supply function of each generator is discretised into 12 equal-capacity bid segments. The bidding parameter $\beta _ { i }$ is constrained to the interval [0, 4000], where the upper bound corresponds to the maximum admissible value of the slope parameter. The upper bound is consistent with the market price cap of 4000 EUR/MWh.

The agents are trained using a multi-agent implementation of TD3 under the centralized-training/decentralized-execution paradigm. Each generator is represented by a deterministic actor that maps its private history to the bidding parameter used to construct its offer. During training, critics are allowed to condition on joint market information in order to stabilize learning in the non-stationary multi-agent environment. During execution and evaluation, however, only the decentralized actors are used. The agents therefore do not communicate when submitting bids.

Each of the 18 scenarios is trained for 100,000 market instances, organised as 50 episodes of 2,000 instances. The first 4,000 instances are used as a warm-up phase, during which agents execute randomly generated actions to populate the replay buffer. Exploration is introduced through action noise and is gradually removed over the course of training, so that the final evaluation window reflects the learned deterministic policies rather than exploratory behaviour. All screening indicators reported in the previous subsection are computed over the final 10,000 market instances. The main hyperparameters are reported in Table III.

TABLE III MAIN MARL HYPERPARAMETERS.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Algorithm</td><td>Multi-agent TD3</td></tr><tr><td>Bid segments</td><td>12</td></tr><tr><td>Bidding parameter bounds</td><td>[0,4000]</td></tr><tr><td>Training length</td><td>100,000 market instances</td></tr><tr><td>Episodes / horizon</td><td>50 × 2, 000</td></tr><tr><td>Evaluation window</td><td>Final 10,000 instances</td></tr><tr><td>Warm-up phase</td><td>4,000 instances</td></tr><tr><td>Discount factor</td><td>0.99</td></tr><tr><td>Replay buffer size</td><td>50,000</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Actor / critic learning rate</td><td> $1 0 _ { . } ^ { - 4 } \mathrm { ~ / ~ 3 ~ } \cdot 1 0 _ { . } ^ { - 4 }$ </td></tr><tr><td>Hidden layers</td><td>[256, 256]</td></tr><tr><td>Exploration decay</td><td>90,000 instances</td></tr></table>

## V. RESULTS

## A. Screening Outcomes

Among the screening indicators introduced in Section IV, Fig. 1 reports the price markup across the full set of 18 market environments. Markups are generally modest in the NoGrid cases and do not exhibit a clear ordering across demand levels or cost heterogeneity. In contrast, the Grid cases produce substantially higher supra-competitive markups and display a systematic pattern in which higher demand and lower marginal-cost heterogeneity are associated with higher markups. This indicates that transmission constraints materially amplify market power and make the resulting market outcomes more sensitive to the underlying demand and cost structure. At the same time, elevated markups alone cannot distinguish coordinated behaviour from market power arising directly from network constraints.

Applying the screening procedure described in Section IV leaves three non-boundary Grid cases for behavioural validation: Grid-LowDemand-HighCostDiff, Grid-LowDemand-MediumCostDiff, and Grid-MediumDemand-HighCostDiff. These cases all exhibit supra-competitive outcomes while differing in demand conditions, cost asymmetry, and learned strategic dynamics.

## B. Punishment of Deviations

Figure 2 shows the bidding responses to the forced deviation in the three selected scenarios. In Grid-LowDemand-HighCostDiff, the deviation of $G _ { 1 }$ triggers a pronounced response from the remaining agents, which temporarily lower their bidding slopes and subsequently return towards their pre-deviation strategies, producing a clear punish-and-forgive pattern. A similar pattern emerges in Grid-LowDemand-MediumCostDiff, although the response is asymmetric: $G _ { 5 }$ and $G _ { 6 }$ react strongly, while $G _ { 2 }$ remains close to its predeviation strategy. A plausible explanation for this difference is the change in the residual-supply structure induced by the different relative competitiveness of $G _ { 5 }$ and $G _ { 6 } ,$ , rather than any change in $G _ { 2 } \mathrm { { ' } s }$ own marginal cost, which is identical in the two scenarios. Under HighCostDiff, the very competitive $G _ { 6 }$ is strongly constrained by its network position while $G _ { 5 }$ is a relatively costly substitute, increasing the marginal value of $G _ { 2 } \mathrm { { ' } s }$ available capacity and making its participation in the punishment more important. Under MediumCostDiff, $G _ { 5 }$ becomes more competitive and $G _ { 6 }$ retains more marginal supply headroom, so their combined response can substitute more effectively for $G _ { 2 } .$ , reducing the marginal value of its participation and creating a natural free-riding incentive. In Grid-MediumDemand-HighCostDiff, by contrast, the deviation of $G _ { 5 }$ does not induce a comparable coordinated reduction in competitors’ bidding slopes, and no clear punishand-forgive pattern is observed.

![](images/0107576ca32416bea5b6b750896f1bbaba8e3e789ceadf33a0ca5a008c2a3d1b.jpg)  
(a)

![](images/7674008e7cd913e403ecb2aeffa2e2ad26a3c9b56ce7e5cef375c18c4e249524.jpg)  
(b)

Fig. 1. Price markups across the nine demand and marginal-cost heterogeneity combinations: (a) NoGrid scenarios without active transmission constraints and (b) Grid scenarios with active transmission constraints.  
![](images/5b28a83b066a50fdb6b2fe9cfe0037254d4ed8b1f5f954b37ccf2c8a97f5665a.jpg)  
Fig. 2. Agents’ bidding actions during the Punishment of Deviation test for the three selected scenarios. The shaded region denotes the forced-deviation window, and the black line identifies the deviating agent.

The corresponding reward trajectories in Fig. 3 show whether these strategic responses impose an economic penalty on the deviating agent. In Grid-LowDemand-

HighCostDiff, the response of the other agents reduces the profitability of $G _ { 1 } \mathrm { { ' } s }$ deviation, with rewards subsequently recovering as the agents return towards the previous regime. In Grid-LowDemand-MediumCostDiff, the response of $G _ { 5 }$ and $G _ { 6 }$ is likewise sufficient to penalize the deviator despite the weak reaction of $G _ { 2 } ,$ , supporting the interpretation that $G _ { 2 }$ is not pivotal to punishment in this case. In Grid-MediumDemand-HighCostDiff, the absence of a comparable strategic response is reflected in the deviator retaining the benefit of its deviation, providing no evidence of an effective punishment mechanism. The Punishment of Deviation criterion is therefore supported in both LowDemand scenarios, although with an asymmetric response in the MediumCostDiff case, and is not supported in Grid-MediumDemand-HighCostDiff.

![](images/a5fcace4818d2e8b142025bc91b1dd3f4593f1740102ebc51e7e9d7721c4e0df.jpg)  
Fig. 3. Agents’ rewards during the Punishment of Deviation test for the three selected scenarios. The shaded region denotes the forced-deviation window, and the black line identifies the deviating agent.

## C. Shortsight Ablation

Figure 4 compares the evolution of episode-average bidding slopes under the baseline MARL configuration and the shortsight ablation. Contrary to a simple collapse-to-competition interpretation, removing memory and long-horizon incentives does not systematically reduce the learned bidding slopes: the magnitude and direction of the changes vary across agents and scenarios, while the Grid-MediumDemand-HighCostDiff case remains particularly close to the baseline. A plausible explanation is that a substantial part of the on-path supra-competitive outcome is supported by static, network-induced market power. Transmission constraints, local scarcity, and generator pivotality remain unchanged after the ablation, so high bidding slopes may remain profitable even for nearly myopic agents. The stationarity of demand, costs, and network parameters further allows the learned policies to stabilize around similar actions without relying on long histories of past market outcomes. The ablated runs also converge more smoothly, consistent with weaker inter-agent feedback loops. Hence, the similarity of the on-path outcomes does not imply that the underlying strategic mechanism is unchanged; the relevant distinction is between the level of the learned regime and the agents’ ability to condition their behaviour on deviations from it.

This distinction becomes apparent when the forceddeviation experiment is repeated under the shortsight ablation, as shown in Fig. 5. In both LowDemand scenarios, the pronounced punishment-and-forgiveness responses observed under the baseline disappear, with competitors’ bidding slopes remaining largely unchanged following the forced deviation. Removing history eliminates the informational basis for conditioning current actions on a previous deviation, while the near-myopic discount factor removes much of the incentive to incur a short-run punishment cost in order to restore a more profitable future regime. In Grid-MediumDemand-HighCostDiff, where no clear punishment response was present under the baseline, the ablation likewise produces no evidence of an enforcement mechanism. The shortsight ablation therefore affects primarily the off-path enforcement mechanism rather than necessarily the on-path level of bids. It consequently provides supporting evidence for a repeatedgame component in the two LowDemand scenarios through the disappearance of punishment, while the persistence of similar on-path bidding levels indicates that their supracompetitive outcomes are also partly supported by static market power; no comparable evidence is obtained for Grid-MediumDemand-HighCostDiff.

## D. Iterative Best Responses

Figure 6 shows the reward trajectories obtained when the learned MARL outcomes are used as the starting point for Iterative Best Response. In Grid-LowDemand-HighCostDiff, unilateral best responses provide short-run profit opportunities, but the subsequent IBR dynamics move the agents towards reward levels that are generally below those sustained under MARL. The pattern is even clearer in Grid-LowDemand-MediumCostDiff, where the initially profitable deviations are followed by lower reward levels for all agents relative to their MARL benchmarks. These two cases are therefore consistent with a regime in which unilateral deviation is attractive in the short run, while the competitive adaptation induced by continued best responses is less profitable. In Grid-MediumDemand-HighCostDiff, by contrast, the IBR trajectories do not exhibit the same combination of profitable short-run deviations and a systematically less profitable subsequent regime. The Short-term Profitability of Deviations criterion is therefore supported in the two LowDemand scenarios and not supported in Grid-MediumDemand-HighCostDiff.

![](images/2c99aadc79c917dfdb03558eb29b02f99327bbfe590e359ef0bb0000cfaf96e5.jpg)  
Fig. 4. Evolution of episode-average bidding slopes under the baseline MARL configuration and the shortsight ablation for the three selected scenarios. Solid lines denote the baseline agents and dashed lines the ablated agents.

![](images/110541906f864eb569f543e780e67d5fec32ac28c0ef9facb353b51818618b66.jpg)  
Fig. 5. Agents’ bidding actions during the Punishment of Deviation test under the shortsight ablation for the three selected scenarios. The shaded region denotes the forced-deviation window, and the black line identifies the deviating agent.

TABLE IV  
SUMMARY OF BEHAVIOURAL EVIDENCE FOR TACIT COLLUSION.
<table><tr><td>Scenario</td><td>Punish.</td><td>Shortsight</td><td>IBR</td><td>Interpretation</td></tr><tr><td>Grid-LowDemand- HighCostDiff</td><td></td><td></td><td></td><td>Supportive Supportive Supportive Collusion-consistent</td></tr><tr><td>Grid-LowDemand- MediumCostDiff</td><td></td><td></td><td></td><td>Supportive Supportive Supportive Collusion-consistent</td></tr><tr><td>Grid-</td><td></td><td></td><td></td><td>Not supp. Not supp. Not supp. Structural market</td></tr><tr><td>MediumDemand- HighCostDiff</td><td></td><td></td><td></td><td>power</td></tr></table>

## E. Synthesis of Behavioural Evidence

Table IV summarizes the evidence obtained from the three behavioural criteria. Taken together, the three behavioural tests provide mutually consistent evidence that the two LowDemand learned regimes contain a repeated-game component consistent with Tacit Collusion. The persistence of supra-competitive on-path bidding after the shortsight ablation nevertheless indicates that these outcomes are not driven by repeated-game incentives alone, but are also supported by structural, network-induced market power. Grid-

MediumDemand-HighCostDiff provides the contrasting case: despite exhibiting a supra-competitive market outcome, it shows neither a punishment response, an ablation-sensitive enforcement mechanism, nor an IBR pattern consistent with the collusion criteria. Its elevated markup is therefore more plausibly attributed to structural market power than to Tacit Collusion. Overall, the results show that similar supra-competitive outcomes can arise from qualitatively different strategic mechanisms, which cannot be distinguished from price levels alone.

## VI. CONCLUSIONS

This paper investigated whether Tacit Collusion can emerge in electricity markets whose participants delegate their bidding strategies to autonomous, learning-based algorithms. Strategic bidding was modeled as a repeated game with imperfect public monitoring, with agents’ learning-based policies modeled via multi-agent reinforcement learning. Different cases of market characteristics were filtered based on susceptibility for Tacit Collusion and the most suspicious ones were tested against three Tacit Collusion criteria. Our experimental results show that the danger of Tacit Collusion in electricity markets is realistic: particularly under binding network constraints, agents learn to sustain supra-competitive outcomes that satisfy several independent indicators of Tacit Collusion, despite never being instructed or designed to collude. Our results should not be taken as conclusive but rather as an alarm bell that encourages extensive research on this topic. Specifically, investigating the learning mechanics that make Tacit Collusion emergent and investigating market-design countermeasures constitute important future work directions.

![](images/8d9dd04ad75c029bff75dec6e1d201b755b9b0744a563df8f07e8199137b7bd0.jpg)  
Fig. 6. Agent reward trajectories under Iterative Best Response (IBR), initialized from the learned MARL outcomes. Iteration 0 corresponds to the MARL endpoint, while the horizontal dashed lines indicate the corresponding MARL reward levels.

## REFERENCES

[1] M. Ivaldi, B. Jullien, P. Rey, P. Seabright, and J. Tirole, The Economics of Tacit Collusion: Implications for Merger Control. Elsevier, 2007, p. 217–239. [Online]. Available: http://dx.doi.org/10.1016/S0573-8555(06) 82008-0

[2] E. Calvano, G. Calzolari, V. Denicolo, and S. Pastorello, “Artificial\` intelligence, algorithmic pricing, and collusion,” American Economic Review, vol. 110, no. 10, p. 3267–3297, Oct. 2020. [Online]. Available: http://dx.doi.org/10.1257/aer.20190623

[3] Y. Liang, C. Guo, Z. Ding, and H. Hua, “Agent-based modeling in electricity market using deep deterministic policy gradient algorithm,” IEEE transactions on power systems, vol. 35, no. 6, pp. 4180–4192, 2020.

[4] E. G. Kardakos, C. K. Simoglou, and A. G. Bakirtzis, “Optimal bidding strategy in transmission-constrained electricity markets,” Electric Power Systems Research, vol. 109, pp. 141–149, 2014.

[5] X. Hu and D. Ralph, “Using epecs to model bilevel games in restructured electricity markets with locational prices,” Operations research, vol. 55, no. 5, pp. 809–827, 2007.

[6] S. A. Gabriel, A. J. Conejo, J. D. Fuller, and B. F. Hobbs, Complementarity modeling in energy markets. Springer New York, NY, 2013.

[7] G. Tsaousoglou, J. S. Giraldo, and N. G. Paterakis, “Market mechanisms for local electricity markets: A review of models, solution concepts and algorithmic techniques,” Renewable and Sustainable Energy Reviews, vol. 156, p. 111890, 2022.

[8] Q. Tang, H. Guo, K. Zheng, and Q. Chen, “Forecasting individual bids in real electricity markets through machine learning framework,” Applied Energy, vol. 363, p. 123053, 2024.

[9] S. Baltaoglu, L. Tong, and Q. Zhao, “Algorithmic bidding for virtual trading in electricity markets,” IEEE Transactions on Power Systems, vol. 34, no. 1, pp. 535–543, 2018.

[10] N. Eschenbaum, “Shared bidding algorithms and competition: Evidence from electricity markets,” 2026. [Online]. Available: https://arxiv.org/ abs/2607.13002

[11] Y. Ye, D. Qiu, M. Sun, D. Papadaskalopoulos, and G. Strbac, “Deep reinforcement learning for strategic bidding in electricity markets,” IEEE Transactions on Smart Grid, vol. 11, no. 2, pp. 1343–1355, 2019.

[12] Z. Zhu, Z. Hu, K. W. Chan, S. Bu, B. Zhou, and S. Xia, “Reinforcement learning in deregulated energy market: A comprehensive review,” Applied Energy, vol. 329, p. 120212, 2023. [Online]. Available: https://doi.org/10.1016/j.apenergy.2022.120212

[13] F. Hu, Y. Zhao, Y. Yu, C. Zhang, Y. Lian, C. Huang, and Y. Li, “Strategic bidding with price-quantity pairs based on deep reinforcement learning considering competitors’ behaviors,” Applied Energy, vol. 391, p. 125874, 2025. [Online]. Available: https://doi.org/10.1016/j.apenergy.2025.125874

[14] H. Zhang, G. Tsaousoglou, S. Zhan, K. Kok, and N. G. Paterakis, “Taming deep reinforcement learning agents with pricing mechanism: Validation in power distribution systems,” Energy and AI, p. 100635, 2025.

[15] N. Harder, R. Qussous, and A. Weidlich, “Fit for purpose: Modeling wholesale electricity markets realistically with multi-agent deep reinforcement learning,” Energy and AI, vol. 14, p. 100295, 2023. [Online]. Available: https://doi.org/10.1016/j.egyai.2023.100295

[16] Y. Du, F. Li, H. Zandi, and Y. Xue, “Approximating nash equilibrium in day-ahead electricity market bidding with multi-agent deep reinforcement learning,” Journal of Modern Power Systems and Clean Energy, vol. 9, no. 3, pp. 534–544, 2021. [Online]. Available: https://doi.org/10.35833/MPCE.2020.000502

[17] Z. Pan and Z. Jing, “Decision-making and cost models of generation company agents for supporting future electricity market mechanism design based on agent-based simulation,” Applied Energy, vol. 391, p. 125881, 2025. [Online]. Available: https://doi.org/10.1016/j.apenergy. 2025.125881

[18] T. Klein, “Autonomous algorithmic collusion: Q-learning under sequential pricing,” The RAND Journal of Economics, vol. 52, no. 3, p. 538–558, Aug. 2021. [Online]. Available: http://dx.doi.org/10.1111/ 1756-2171.12383

[19] T. Werner, “Algorithmic and human collusion,” SSRN Electronic Journal, 2021. [Online]. Available: http://dx.doi.org/10.2139/ssrn.3960738

[20] D. Esmaeili Aliabadi and K. Chan, “The emerging threat of artificial intelligence on competition in liberalized electricity markets: A deep q-network approach,” Applied Energy, vol. 325, p. 119813, 2022. [Online]. Available: https://doi.org/10.1016/j.apenergy.2022.119813

[21] M. S. Avila, R. Ebrahimy, and G. Tsaousoglou, “How inefficient can<sup>\`</sup> an electricity market be?” in 2025 IEEE Kiel PowerTech. IEEE, 2025, pp. 1–6.

[22] J. Miklos-Thal, “Optimal collusion under cost asymmetry,”´ Economic Theory, vol. 46, no. 1, pp. 99–125, 2011. [Online]. Available: http://www.jstor.org/stable/41485808