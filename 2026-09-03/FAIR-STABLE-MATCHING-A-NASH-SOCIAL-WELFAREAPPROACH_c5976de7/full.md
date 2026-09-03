# FAIR STABLE MATCHING: A NASH SOCIAL WELFAREAPPROACH

A PREPRINT

Rasheed Machine Learning Lab IIIT Hyderabad Hyderabad, India mohammad.ahmed@research.iiit.ac.in

Ganesh Ghalme Department of AI IIT Hyderabad Hyderabad, India ganeshghalme@ai.iith.ac.in

Parth Desai Machine Learning Lab IIIT Hyderabad Hyderabad, India parth.desai@research.iiit.ac.in

Sujt Gujar   
Machine Learning Lab   
IIIT Hyderabad   
Hyderabad, India   
sujit.gujar@iiit.ac.in

September 3, 2026

## ABSTRACT

While traditional stable matching algorithms, such as the Gale-Shapley algorithm, prioritize stability, they may fall short of achieving equitable outcomes among participants. We study the role of Nash social welfare (NSW) as a fairness objective in the classic stable marriage problem. We develop SNSW-Alg that finds a stable matching that maximizes Nash social welfare under rank-induced utilities in $\tilde { \mathcal { O } } ( n ^ { 4 } )$ time, where n is the number of men or women. We demonstrate that SNSW-Alg balances equity while preserving stability. We empirically evaluate our methods across diverse preference distributions, demonstrating significant gains in fairness without substantial losses in other key measures such as regret, egalitarian criterion, and sex equality. Our findings suggest that the stable matching produced by SNSW-Alg is statistically Pareto-undominated by stable matchings based on other fairness measures - regret, egalitarian, and sex equality. This study offers compelling insights for designing fair-stable matching.

## 1 Introduction

Matching markets are economic settings where agents are paired based on preferences and compatibility. Unlike many economic markets, these do not rely on price signals for allocation. Instead, structured mechanisms determine who matches with whom, taking into account the preferences of both sides (e.g., students and colleges [Gale and Shapley, 1962], workers and firms [Revilla, 2007], organ donors and recipients [Roth et al., 2004]). Matching theory has two main branches: (i) Matching with transferable utility, where utility is transferable among agents (usually involving monetary transfers), and (ii) Matching with non-transferable utility, where transfers aren’t possible (e.g., school admissions, organ donations). In this paper, we study the stable marriage problem which falls under the non-transferable utility setting.

Fairness in Matchings. The stable marriage problem involves two (most often equally sized) groups - typically referred to as men and women - in which each individual seeks to match with a member of the opposite group based on personal preferences. A central concept in the stable marriage problem is stability. A matching is stable if there are no blocking pairs - that is, no unmatched man and woman who would both prefer each other over their assigned partners. Stability is critical because unstable matchings are inherently fragile and prone to collapse, as participants may have incentives to deviate from the outcome. The celebrated Gale-Shapley algorithm [Gale and Shapley, 1962] outputs a stable matching which is optimal for the proposing side.

The male-optimal matching gives every man the best partner he could have in any stable matching. However, this often comes at the expense of the other side; it is highly unfair to women, who receive their least-preferred partners amongst all stable matchings. Conversely, running the algorithm with women proposing yields the female-optimal and male-pessimal outcome. This inherent asymmetry highlights a tension between stability and fairness, sparking a vast literature on improving equity and efficiency in practical matching mechanisms

Existing approaches for Fair Matching. There have been studies on various alternative criteria for improving fairness and social welfare in stable matchings. The egalitarian stable matching [Irving et al., 1987] minimizes the total sum of ranks (i.e., how far down each partner is in their preference list), promoting overall satisfaction. As we show later, the egalitarian approach may sometimes offer a very good match to the majority at the cost of some agents receiving very poor matches. To balance such situations, the minimum-regret stable matching [Gusfield, 1987] minimizes the worst individual dissatisfaction, ensuring that the least-satisfied participant is as well off as possible. The sex equal stable matching [Kato, 1993] minimizes the difference between the sum of men’s ranks and the sum of women’s ranks, focusing on minimizing the disparity across the two sides. These refinements offer ways to balance efficiency and equity while maintaining stability. However, each measure optimizes some aspect of the agents; overall, to improve fairness across all agents, we believe a new approach is warranted.

Our Approach. We propose to maximize a suitably defined Nash social welfare (NSW) objective over the set of all stable matchings. Note that, if we ignore stability and capacity constraints, maximizing NSW is, in general, an NP-hard problem [Jain and Vaish, 2024]. However, under stability and capacity constraints (where each agent wants to match with exactly one agent), the problem assumes a special structure. By building upon the rotation poset framework introduced in [Irving et al., 1987], we can design an efficient algorithm to optimize NSW while preserving stability. Irving et al. [1987] constructs a capacitated directed graph based on the preferences and rotation posets of stable matchings for optimizing the additive egalitarian criterion. A challenge in our case was addressing the gap between the non-linear NSW criterion and the additive rotation poset framework. We bridge this gap by converting the NSW criterion into an equivalent additive objective. Our key idea is to appropriately adjust the edge weights in the graph to maximize NSW. However, it poses another challenge - we need a max-flow algorithm, and our weights are no longer integral, as in [Irving et al., 1987]. We then propose using Dinic’s algorithm [Dinitz, 2006] with dynamic trees by [Sleator and Tarjan, 1981] rather than Ford-Fulkerson, as proposed by Irving et al. [1987]. We prove that our updated formulation enables SNSW-Alg to compute a stable matching that maximizes NSW in polynomial time, specifically within a time complexity of $\tilde { \mathcal { O } } ( n ^ { 4 } )$ . To evaluate the effectiveness of our approach in achieving fairness, we conduct an empirical study comparing three stable matching algorithms: (i) one that minimizes the sum of ranks (egalitarian), (ii) one that minimizes the worst rank (minimum regret), and (iii) one that minimizes the differences across the two sides (sex equal). Across various distributions of preference profiles, our experiments demonstrate that SNSW-Alg is statistically Pareto undominated among the four methods in most measured dimensions, highlighting its strong performance in balancing fairness and efficiency.

Our Contributions. Our contributions in this work are as follows.

(i) To the best of our knowledge, we are first to propose a novel objective for the stable marriage problem: maximizing Nash social welfare (NSW) over the space of stable matchings, balancing fairness and efficiency.

(ii) Toward this, we propose a polynomial-time algorithm SNSW-Alg.

(iii) We conduct an extensive empirical evaluation across diverse preference distributions, comparing four stable matching objectives.

## 2 Model and Background

In this section, we first explain our model and notation (Section 2.1) and then the required prerequisites (Section 2.2).   
We show the need for a newer fairness measure by example.

## 2.1 Model

We consider a two-sided market N consisting of two types of agents: one side, called men $M ,$ , and the other, women W. Each agent is interested in matching with exactly one agent on the other side and has a preference over the other side, denoted by $\succ _ { a }$ for an agent $a \in N$ . We assume (i) both sides have an equal number of agents, $n ,$ (ii) no agent prefers to be unmatched over any possible match from the other side, and (iii) the preferences are strict. We refer to $\succ = ( ( \succ _ { a } ) _ { a \in N } )$ as the original preference list. A matching $\mathcal { M } ( \succ )$ outputs a match for each agent. Let $r ( m _ { i } , w _ { j } )$ (or, $r ( w _ { k } , m _ { l } ) )$ represent the rank of agent $w _ { j } \ ( \mathrm { o r } , m _ { l } ) \mathrm { i n } \succ _ { m _ { i } } ( \mathrm { o r } , \succ _ { w _ { k } } )$ . We assumed a 0-indexed ranking, that is, the best possible match has rank 0. Motivated by Budish and Cantillon [2009], we consider $u r ( m _ { i } , w _ { j } ) = n - r ( m _ { i } , w _ { j } )$ (or, $u r ( w _ { k } , m _ { l } ) = n - r ( w _ { k } , m _ { l } ) )$ to be the utility of man $m _ { i }$ (or, woman $w _ { k } )$ on being matched to woman $w _ { j }$ (or, man m ). Let Σ denote the set of all possible matchings.

## 2.2 Preliminaries

We denote match of an agent $a \in N$ under matching M as $\mathcal { M } _ { a } .$ . In a matching M, i $\textrm { f } r ( m _ { i } , w _ { j } ) < r ( m _ { i } , { \mathcal { M } } _ { m _ { i } } )$ and $r ( w _ { j } , m _ { i } ) < r ( w _ { j } , \mathcal { M } _ { w _ { i } } \bar { ) }$ , then $( m _ { i } , w _ { j } )$ would prefer to match with each other over the matching M. Such a pair is called a blocking pair [Gale and Shapley, 1962]. A matching M is said to be a stable matching if it does not contain any blocking pairs [Gale and Shapley, 1962]. It is desirable to obtain a stable matching; otherwise, agents in blocking pairs may prefer to match with each other outside the market. It is a well-known result that for any arbitrary instance of $\succ ,$ a stable matching always exists [Gale and Shapley, 1962]. Let Π be the set of all possible stable matchings when the agents’ original preference list is ≻.

A carefully constructed instance (≻) may yield an exponential number of stable matchings (in n) [Knuth, 1997]. The Gale-Shapley algorithm<sup>1</sup> [Gale and Shapley, 1962] outputs one such stable matching $\mathcal { M } ^ { o } ( \succ )$ (assuming men propose to women). Similarly, it would output $\mathcal { M } ^ { \bar { z } } ( \succ$ if women proposed to men. The stable matching $\mathcal { M } ^ { o } ( \succ )$ is characteristic in that it is “man-optimal”, i.e., each man gets the best possible match possible under the constraint of stability. In other words, $\forall m _ { i } \in M , r ( m _ { i } , \mathcal { M } ^ { o } { } _ { m _ { i } } ) = \operatorname* { m i n } _ { \mathcal { M } \in \Pi _ { \sim } } r ( m _ { i } , \mathcal { M } _ { m _ { i } } )$ . The stable matching $\mathcal { M } ^ { o }$ is not only “man-optimal” but also “women-pessimal”, i.e., $\begin{array} { r } { \forall w _ { j } \in W , r ( w _ { j } , \ M ^ { o } { } _ { w _ { j } } ) = \operatorname* { m a x } _ { M \in \Pi _ { \sim } } r ( w _ { j } , \ M _ { w _ { j } } ) } \end{array}$ . Similarly, $\mathcal { M } ^ { \bar { z } }$ is “woman-optimal” and “men-pessimal”. Though $\dot { \mathcal { M } } ^ { o } \left( \mathrm { o r } , \mathcal { M } ^ { z } \right)$ satisfies the criterion of stability, it is an unfair matching for W (or, M). This concern has motivated research beyond the Gale-Shapley algorithm to introduce fairness in matching.

Fairness in Matching. The set of all stable matchings forms a lattice [Knuth, 1997]. It is desirable to determine a matching that is not only stable but also fair to all the agents. Towards this, there exist multiple optimal stable matchings in the literature - egalitarian stable matching [Irving et al., 1987], minimum-regret stable matching [Gusfield, 1987], and sex equal stable matching [Kato, 1993]. These stable matchings are optimal with respect to three fairness measures - egalitarian, regret, and sex equality.

Egalitarian criterion is defined as the total sum of ranks of all matches.

$$
\mu ^ { e } ( \mathcal { M } ) = \sum _ { m _ { i } \in M } r ( m _ { i } , \mathcal { M } _ { m _ { i } } ) + \sum _ { w _ { j } \in W } r ( w _ { j } , \mathcal { M } _ { w _ { j } } )
$$

Regret is defined as the worst individual dissatisfaction across men and women.

$$
\mu ^ { r } ( \mathcal { M } ) = \operatorname* { m a x } \Big ( \operatorname* { m a x } _ { m _ { i } \in M } r ( m _ { i } , \mathcal { M } _ { m _ { i } } ) , \operatorname* { m a x } _ { w _ { j } \in W } r ( w _ { j } , \mathcal { M } _ { w _ { j } } ) \Big )
$$

Sex-equality is defined as the absolute difference between the total satisfaction of men and women.

$$
\mu ^ { d } ( \mathcal { M } ) = \Big | \sum _ { m _ { i } \in M } r ( m _ { i } , \mathcal { M } _ { m _ { i } } ) - \sum _ { w _ { j } \in W } r ( w _ { j } , \mathcal { M } _ { w _ { j } } ) \Big |
$$

Definition 1 (Egalitarian Stable Matching). [Irving et al., 1987] A stable matching $\mathcal { M } ^ { e } ( \succ )$ is egalitarian if it minimizes the egalitarian criterion across all stable matchings.

$$
\mathcal { M } ^ { e } ( \succ ) = \arg \operatorname* { m i n } _ { \mathcal { M } \in \Pi _ { \succ } } \mu ^ { e } ( \mathcal { M } )
$$

Definition 2 (Minimum-Regret Stable Matching). [Gusfield, 1987] A stable matching $\boldsymbol { \mathcal { M } ^ { r } } ( \succ )$ is minimum-regret if it minimizes the regret across all stable matchings.

$$
\mathcal { M } ^ { r } ( \succ ) = \arg \operatorname* { m i n } _ { \mathcal { M } \in \Pi _ { \succ } } \mu ^ { r } ( \mathcal { M } )
$$

Definition 3 (Sex-Equal Stable Matching). [Kato, 1993] A stable matching $\mathcal { M } ^ { d } ( \succ )$ is sex equal ifit minimizes the sex equality across all stable matchings.

$$
\mathcal { M } ^ { d } ( \succ ) = \arg \operatorname* { m i n } _ { \mathcal { M } \in \Pi _ { \succ } } \mu ^ { d } ( \mathcal { M } )
$$

Irving [1985] utilize the concept of rotations and propose an $\mathcal { O } ( n ^ { 4 } )$ algorithm that finds the egalitarian stable matching $\mathcal { M } ^ { e } ( \succ )$ . We refer to their algorithm as $E g a l i t a r i a n \mathrm { - } A \bar { \it { 1 } } g$ (more details on this in Section 3.1). Gusfield [1987] provides an algorithm to find $\displaystyle { \mathcal { M } } ^ { r } ( \succ )$ . However, finding $\mathcal { M } ^ { d } ( \succ )$ is NP-Hard [Kato, 1993]. Section A in the Appendix summarizes all the notation used in the paper.

Need for an improved approach. Consider the preference instance ≻ in Table 1. The egalitarian stable matching $\mathcal { M } ^ { e } = \{ ( 1 , 4 ) , ( \hat { 2 } , 1 ) , ( 3 , \hat { 3 } \hat { ) } , ( 4 , 5 ) , ( 5 , 2 ) \}$ } performs poorly on $\mu ^ { r } = 4$ . On the other hand, the minimum-regret stable matching $\mathcal { M } ^ { \acute { r } } = \{ ( \bar { 1 } , \bar { 4 } ) , ( \bar { 2 } , \bar { 2 } ) , ( \bar { 3 } , \bar { 1 } ) , ( \bar { 4 } , \bar { 5 } ) , ( 5 , 3 ) \}$ performs poorly on $\mu ^ { e } = 1 1 = 3 + 8$

Table 1: Example $\mathcal { M } ^ { e }$ dominate $\mathcal { M } ^ { r }$ on $\mu ^ { e }$ and $\mathcal { M } ^ { r }$ dominate $\mathcal { M } ^ { e }$ on $\mu ^ { r }$
<table><tr><td>M</td><td colspan="4">W</td></tr><tr><td>1:</td><td>4 3 </td><td>2</td><td></td><td>5</td></tr><tr><td>2:</td><td>5 4</td><td>2</td><td>3</td><td>1</td></tr><tr><td>3:</td><td>5 1</td><td>3</td><td>4</td><td>2</td></tr><tr><td>4:</td><td>5 2</td><td>1</td><td>4</td><td>3</td></tr><tr><td>5:</td><td>3 2</td><td>4</td><td>5</td><td></td></tr></table>

<table><tr><td>W</td><td colspan="4">M</td></tr><tr><td>1:</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>2:</td><td>5 1 </td><td>2</td><td>3</td><td>4</td></tr><tr><td>3:</td><td>3 1</td><td>5 </td><td>4</td><td>2</td></tr><tr><td>4:</td><td>5 4</td><td>1</td><td>2</td><td>3</td></tr><tr><td>5:</td><td>4 1</td><td>3</td><td>2</td><td>5</td></tr></table>

These gaps point to various limitations that we address in this paper. Our empirical analysis also shows that such examples shown above are not rare. Statistically, each of the three fairness measures above performs poorly on the others; we need a better measure that performs better overall. Towards this, we make use of the well-known measure of Nash social welfare (NSW) [Nash et al., 1950] to address these limitations. Nash social welfare. Nash social welfare (NSW) may be seen as fair in multiple ways in many other domains [Nash et al., 1950, Kaneko and Nakamura, 1979, Moulin, 2004, Caragiannis et al., 2019]. Moulin [2004] shows that NSW is the only fair measure that satisfies many reasonable fairness axioms under general technical conditions. While the Nash bargaining solution maximizes the product of gains over a disagreement point, the NSW maximizes the product of utilities, which is equivalent to Nash bargaining with a disagreement point equal to zero. Accordingly, our formulation is as follows:

Nash social welfare (NSW) is the geometric mean of the utilities of men and women.

$$
\mu ^ { n s w } ( \mathcal { M } ) = \Bigg ( \prod _ { m _ { i } \in M } u r ( m _ { i } , \mathcal { M } _ { m _ { i } } ) \times \prod _ { w _ { j } \in W } u r ( w _ { j } , \mathcal { M } _ { w _ { j } } ) \Bigg ) ^ { \frac { 1 } { 2 n } }\tag{1}
$$

In two-sided market literature. For instance, Jain and Vaish [2024] work with agents’ utilities for the matching and maximize Nash social welfare. We refer to such matching as NSW matching.

Definition 4. A matching $\mathcal { M } ^ { n s w } ( \succ )$ is Nash welfare optimal if it maximizes the Nash social welfare of all agents.

$$
\mathcal { M } ^ { n s w } ( \succ ) = \arg \operatorname* { m a x } _ { \mathcal { M } \in \Sigma } \mu ^ { n s w } ( \mathcal { M } )
$$

The matching $\mathcal { M } ^ { n s w } ( \succ )$ can be computed in $\mathcal { O } ( n ^ { 3 } )$ time using maximum-weight matching algorithms in bipartite graphs. The algorithm indeed outputs a matching that could be considered fair according to the aforementioned measures $( \mu ^ { e } , \bar { \mu ^ { r } } , \mu ^ { d } )$ , owing to the larger search space. Anyhow, the matching it computes need not be stable. Hence, we propose to maximize NSW amongst all possible stable matchings. Consider $\succ _ { 2 }$ given in Table 2. The

Table 2: ≻<sub>2</sub>: Nash social welfare performs well on all measures
<table><tr><td>M</td><td colspan="4">W</td></tr><tr><td>1:</td><td>3</td><td>2 5</td><td>1 </td><td>4</td></tr><tr><td>2:</td><td>1</td><td>5</td><td>2 4</td><td>3</td></tr><tr><td>3:</td><td>4</td><td>3 1</td><td>2</td><td>5</td></tr><tr><td>4:</td><td>2</td><td>3 4</td><td>1</td><td>5</td></tr><tr><td>5:</td><td>2</td><td>3 5</td><td>1</td><td>4</td></tr></table>

<table><tr><td>W</td><td colspan="4">M</td></tr><tr><td>1:</td><td>3 1</td><td>5</td><td>2</td><td>4</td></tr><tr><td>2:</td><td>1 3</td><td>4</td><td>2</td><td>5</td></tr><tr><td>3:</td><td>4 1</td><td>2</td><td>5</td><td>3</td></tr><tr><td>4:</td><td>4 1</td><td>2</td><td>5</td><td>3</td></tr><tr><td>5:</td><td>3 4</td><td>1</td><td>2</td><td>5</td></tr></table>

stable matching $\mathcal { M } ^ { e } ( \succ ) = \{ ( 1 , 2 ) , ( 2 , 1 ) , ( 3 , 4 ) , ( 4 , 3 ) , ( 5 , 5 ) \}$ has $\mu ^ { e } = 1 5 $ , but it performs poorly on $\mu ^ { d } = 7$ . The stable matching $\mathcal { M } ^ { d } ( \succ ) = \{ ( 1 , 3 ) , ( 2 , 5 ) , ( 3 , 1 ) , ( 4 , 2 ) , ( 5 , 4 ) \}$ has $\mu ^ { d } = 2 $ , but its $\mu ^ { e } = 1 6$ . The stable matching $\mathcal { M } = \left\{ ( 1 , 2 ) , ( 2 , 5 ) , ( 3 , 4 ) , ( 4 , 3 ) , ( 5 , 1 ) \right\}$ } performs well on both measures: $\mu ^ { e } = 1 5$ and $\mu ^ { d } = 3$ . This particular stable matching maximizes the Nash social welfare across all stable matchings.

## 3 Our Approach

## 3.1 SNSW-Alg

First, we define an NSW optimal stable matching.

Definition 5. A stable matching $\mathcal { M } ^ { s n s w } ( \succ )$ is Nash welfare optimal if it maximizes the Nash social welfare of all agents.

$$
\mathcal { M } ^ { s n s w } ( \succ ) = \arg \operatorname* { m a x } _ { \mathcal { M } \in \Pi _ { \succ } } \mu ^ { n s w } ( \mathcal { M } )
$$

Note the difference in definitions of $\mathcal { M } ^ { n s w } ( \succ )$ and $\mathcal { M } ^ { s n s w } ( \succ ) ;$ ; the former is over all possible matching Σ and the latter is over $\Pi _ { \succ }$ . Finding a matching $\mathcal { M } ^ { n s w } ( \succ )$ is the same as maximum weight matching in a bipartite graph [Jain and Vaish, $2 0 2 4 ]$ . However, because it does not need to be stable, we cannot leverage existing max-flow or matching algorithms to design an algorithm $- \mathrm { ~ S N S W - A l 9 }$ that finds $\mathcal { M } ^ { s n s w }$ . On the other hand, $\mathtt { E q a l i t a r i a n - A l g }$ in Irving et al. [1987] determines a stable matching that minimizes $\mu ^ { e }$ , that is, outputs $\mathcal { M } ^ { e } ( \succ )$ . We build upon $\mathtt { E q a l i t a r i a n - A l g }$ . To this end, we present a short description of Ega $1 \mathtt { i } \mathtt { t a } \mathtt { r i a } \mathtt { n } \mathtt { - a } \mathtt { l } \mathtt { q }$ in the following paragraph.

Egalitarian-Alg. $\mathtt { E q a l i t a r i a n - A l g }$ begins with a stable matching (typically obtained via the Gale-Shapley algorithm) and identifies rotations, following the framework of [Irving, 1985].

Definition 6. A rotation $\rho ,$

$$
\rho = ( m _ { 0 } , w _ { 0 } ) , \ldots , ( m _ { r - 1 } , w _ { r - 1 } ) ,
$$

is a cyclic sequence ofmatched pairs, such that $\forall i \in [ 0 , r - 1 ] , w _ { i } \succ _ { m _ { i } } w _ { i + 1 }$ and $m _ { i - 1 } \succ _ { w _ { i } } m _ { i }$ , where $( i + 1 ) , ( i - 1 )$ is taken (mod r). Rotation (say ρ) is said to be exposed with respect to a reduced preference list $( s a y \mathcal { S } ) i f w _ { i } =$ $f i r s t _ { \mathcal { S } } ( m _ { i } ) = s e c o n d _ { \mathcal { S } } ( m _ { i - 1 } )$

Successive elimination of rotations produces “reduced preference ${ \mathrm { l i s t s } } ^ { \prime 3 } \mathcal { S } .$ . A rotation $\rho$ is said to be eliminated if, $\forall i \in [ 0 , r - 1 ] , \forall j : m _ { j } \succ _ { w _ { i } } m _ { i - 1 } , m _ { j } \not \in \mathcal { S } _ { w _ { i } } , w _ { i } \not \in \mathcal { S } _ { m _ { j } }$ . If a rotation $\rho$ cannot be exposed before another rotation π is eliminated, then π is said to precede $\rho .$ The resulting precedence structure defines the rotation poset $P ,$ , whose vertices $( V ( P ) )$ correspond to rotations and whose directed edges $( E ( P ) )$ represent precedence relations [Irving, 1985]. We define $p r e d _ { P } ( \rho )$ as all predecessors of rotation $\rho$ in $P .$

A sparse subgraph $P ^ { \prime }$ of $P$ is constructed using the following rules:

R1 If $( m _ { i ^ { * } } , w _ { j ^ { * } } )$ belongs to rotation $\pi ,$ and $w _ { j ^ { \prime } }$ is the first woman below $w _ { j ^ { \ast } }$ in $m _ { i ^ { * } } { \mathrm {  ~ s ~ } }$ preference list such that $( m _ { i ^ { * } } , w _ { j ^ { \prime } } )$ belongs to rotation $\rho ,$ then add edge $( \pi , \rho )$

R2 If $( m _ { i ^ { * } } , w _ { j ^ { \prime } } )$ does not belong to any rotation but is eliminated by rotation $\pi ,$ and $w _ { j ^ { \ast } }$ ∗ is the first woman above $w _ { j ^ { \prime } }$ such that $( m _ { i ^ { * } } , w _ { j ^ { * } } )$ belongs to rotation $\rho ,$ then add edge $( \pi , \rho )$

The transitive closure of $P ^ { \prime }$ equals $P ;$ consequently, $P ^ { \prime }$ preserves all closed subsets and therefore all stable matchings [Irving and Leather, 1986]. Specifically, eliminating the rotations of a closed subset yields a reduced preference structure $\breve { \mathscr { S } }$ , and the stable matching $( m , f i r s t _ { \mathcal { S } } ( m ) )$ corresponds uniquely to that subset.

Given $P ^ { \prime } ,$ a capacitated s-t network $P ^ { \prime } ( s , t )$ is constructed. A source s is connected to each negative-weight rotation $\rho$ with capacity $| w ( \rho ) |$ , while each positive-weight rotation is connected to sink t with capacity $w ( \rho )$ . All original edges of $P ^ { \prime }$ are assigned infinite capacity. The following result links minimum cuts and maximum-weight closed subsets.

Theorem 1 ([Irving et al., 1987]). Let X be a minimum s-t cut in $P ^ { \prime } ( s , t )$ . The positive rotations whose edges to t remain uncut, together with all their predecessors in $P ^ { \prime } ,$ ,form a maximum-weight closed subset of $P ^ { \prime }$

Since the number of rotations and edges in $P ^ { \prime }$ are each bounded by $\mathcal { O } ( n ^ { 2 } )$ , both the network size and maximum flow value are $\mathcal { O } ( n ^ { 2 } )$ . Therefore, applying Ford-Fulkerson yields the maximum-weight closed subset—and hence the egalitarian-optimal stable matching—in $\breve { \mathcal { O } } ( | K | | E | ) = \mathcal { O } ( n ^ { \mathbf { \bar { 4 } } } )$ time.

Adapting Egalitarian-Alg to build SNSW-Alg. Note that Egalitarian-Alg assigns a weight $w ( \rho )$ (Eqn. 2) to a rotation $\rho = \{ ( m _ { 0 } , w _ { 0 } ) , ( m _ { 1 } , w _ { 1 } ) , \dots , ( m _ { r - 1 } , w _ { r - 1 } ) \}$ in poset $P$ . With this, the algorithm actually generates a new graph $\scriptstyle \dot { P } ^ { \prime }$ . The goal in the algorithm is to minimize $\bar { \mu } ^ { e } ( M )$ .

$$
w ( \rho ) = \sum _ { i = 0 } ^ { r - 1 } u r ( m _ { i } , \rho _ { m _ { i + 1 } } ) - u r ( m _ { i } , \rho _ { m _ { i } } ) + \sum _ { i = 0 } ^ { r - 1 } u r ( w _ { i } , \rho _ { w _ { i - 1 } } ) - u r ( w _ { i } , \rho _ { w _ { i } } )\tag{2}
$$

Our goal is to maximize Nash social welfare, which is the product of utilities. Using logarithmic transformations, we can reduce the problem to maximizing the sum of the logarithms of utilities. Hence, we propose to use weights of rotations as defined in Definition 7.

Definition 7. Given a rotation $\rho = \{ ( m _ { 0 } , w _ { 0 } ) , ( m _ { 1 } , w _ { 1 } ) , \dots , ( m _ { r - 1 } , w _ { r - 1 } ) \}$ in a stable marriage instance, we define the weight w $' ( \rho )$ of that rotation by

$$
w ^ { \prime } ( \rho ) = \sum _ { i = 0 } ^ { r - 1 } \log u r ( m _ { i } , \rho _ { m _ { i + 1 } } ) - \log u r ( m _ { i } , \rho _ { m _ { i } } ) + \sum _ { i = 0 } ^ { r - 1 } \log u r ( w _ { i } , \rho _ { w _ { i - 1 } } ) - \log u r ( w _ { i } , \rho _ { w _ { i } } )\tag{3}
$$

Determining $\mathcal { M } ^ { s n s w } ( \succ )$ reduces to determining $\mathcal { M } ^ { e } ( \succ )$ by plugging the weights in Equation 3 in Egalitarian-Alg.

In Egalitarian-Alg, the edge weights of $P ^ { \prime } ( s , t )$ were integers. Hence, they could apply the Ford-Fulkerson algorithm to find the max-flow in $\mathcal { O } ( C \cdot E )$ time. In $\mathrm { S N S W - \bar { A } \bar { 1 } \mathrm { g } , }$ , edge weights need not be integers. Therefore, we employ Dinic’s maximum flow algorithm [Dinitz, 2006] with dynamic trees improvement by Sleator and Tarjan [1981] to find the max-flow in $\mathcal { O } ( | V | \cdot | \bar { E } | \cdot \log | V | )$ time.

Thus, Egalitarian-Alg adapted to our settings is described as follows.

Algorithm 1 SNSW-Alg   
1: Input: $\succ = ( ( \succ _ { a } ) _ { a \in N } )$   
2: Output: $\mathcal { M } ^ { \mathit { \acute { s n s w } } } ( \succ )$   
3: Procedure:   
4: $\mathcal { M } ^ { o } ( \succ _ { m } , \succ _ { w } ) = \mathtt { G a l e } \mathrm { - S h a p 1 }$ ey(≻)   
5: Determine the vertex set $V ( P )$   
6: Initialize $w ( \rho ) , \forall \rho \in V ( P )$ using Equation 3   
7: Construct subgraph $P ^ { \prime }$ via rules in Section 3.1   
8: $\forall e \in E ( P ^ { \prime } ) , \check { w } ( e ) \gets \infty ,$ , Construct $P ^ { \prime } ( s , t )$ as follows:   
9: $\forall \rho \in V ( P ^ { \prime } ) : ( \dot { i } ) w ( \rho ) < 0 , e : s \to \rho , w ( e ) : = | w ( \rho ) | , ( i i ) w ( \rho ) > 0 , e : \rho \to t , w ( e ) : = w ( \rho )$   
10: $X = m i n - c u t ( P ^ { \prime } ( s , t ) )$ (Dinic’s algorithm [Dinitz, 2006] with dynamic trees improvement by Sleator and Tarjan   
[1981])   
11: $\mathbf { \bar { \chi } } ^ { + } : = \{ \rho \in P ^ { \prime } ( s , t ) : e ( \rho , t )$ uncut by $X \} , \mathfrak { S } : = X ^ { + } \cup p r e d _ { P } ( X ^ { + } )$   
12: for all $\rho \in \mathfrak { S }$ do   
13: Eliminate $\rho ,$ and Update $\mathcal { S }$ as explained in Section 3.1   
14: end for   
15: return $\mathcal { M } ^ { \mathrm { S N S W - A 1 g } } ( \succ ) = \{ ( m _ { i } , f i r s t _ { \mathcal { S } } ( m _ { i } ) ) \} _ { m _ { i } \in M }$

Now we prove the correctness and the time complexity of SNSW-Alg.

## 3.2 Theoretical Analysis of SNSW-Alg

We prove the correctness of SNSW-Alg as follows.

Proposition 1. Let $\mathcal { M } ^ { \prime }$ be the stable matching obtained on eliminating rotations $\rho _ { 1 } , \rho _ { 2 } , \ldots , \rho _ { t } .$ from the man-oriented shortlist, then

$$
\mu ^ { n s w } ( \mathcal { M } ^ { \prime } ) = \mu ^ { n s w } ( \mathcal { M } ^ { o } ) \ast \left( \prod _ { i = 1 } ^ { t } \exp \frac { w ^ { \prime } ( \rho _ { i } ) } { 2 n } \right)
$$

Proof. Consider a stable matching $\mathcal { M } ^ { s } .$ , with rotation $\rho$ exposed in its preference lists, and let $\mathcal { M }$ be the stable matching obtained upon eliminating $\rho$ from the reduced preference lists associated with $\mathcal { M } ^ { s }$ . Thus, the following follows

naturally.

$$
\begin{array} { l l } { { w ^ { \prime } ( \rho ) = \displaystyle \sum _ { i = 0 } ^ { r - 1 } \log u r ( m _ { i } , \rho _ { m _ { i + 1 } } ) - \log u r ( m _ { i } , \rho _ { m _ { i } } ) + \displaystyle \sum _ { i = 0 } ^ { r - 1 } \log u r ( w _ { i } , \rho _ { w _ { i - 1 } } ) - \log u r ( w _ { i } , \rho _ { w _ { i } } ) ~ } } & { { ~ \mathrm { U s i n g ~ 3 ~ } } } \\ { { \displaystyle \exp w ^ { \prime } ( \rho ) = \displaystyle \prod _ { i = 0 } ^ { r - 1 } \frac { u r ( m _ { i } , \rho _ { m _ { i + 1 } } ) \cdot u r ( w _ { i } , \rho _ { w _ { i - 1 } } ) } { u r ( m _ { i } , \rho _ { m _ { i } } ) \cdot u r ( w _ { i } , \rho _ { w _ { i } } ) } } } & { { ~ } } \end{array}
$$

Multiply numerator and denominator by the utility of all man-woman pairs who remain matched after elimination of $\rho$

$$
\exp w ^ { \prime } ( \rho ) = \frac { \Bigg ( \prod _ { m _ { i } \in M } u r ( m _ { i } , \mathcal { M } _ { m _ { i } } ) \times \prod _ { w _ { j } \in W } u r ( w _ { j } , \mathcal { M } _ { w _ { j } } ) \Bigg ) } { \Bigg ( \prod _ { m _ { i } \in M } u r ( m _ { i } , \mathcal { M } _ { m _ { i } } ^ { s } ) \times \prod _ { w _ { j } \in W } u r ( w _ { j } , \mathcal { M } _ { w _ { j } } ^ { s } ) \Bigg ) } = \left( \frac { \mu ^ { n s w } ( \mathcal { M } ) } { \mu ^ { n s w } ( \mathcal { M } ^ { s } ) } \right) ^ { 2 n } ,
$$

Since $\mathcal { M } ^ { \prime }$ is obtained upon the successive elimination of $\rho _ { 1 } , \ldots , \rho _ { t }$ from shortlists associated with $\mathcal { M } ^ { o }$ , the equation associated with each rotation may be multiplied together to prove the proposition. □

Theorem 2. For every preference profile ≻, SNSW-Alg outputs the stable matching that maximizes the Nash social welfare, i.e., SNSW-Alg returns $\bar { \mathcal { M } } ^ { s \bar { n } s w } ( \succ$

Proof. SNSW-Alg outputs the maximum-weight closed subset S in the weighted rotation poset $P$ using $w ( \rho )$ from Def. 7.

$$
\mu ^ { n s w } ( \mathcal { M } ^ { \mathrm { S N S W - A l g } } ) = \mu ^ { n s w } ( \mathcal { M } ^ { o } ) * \left( \prod _ { \rho _ { i } \in \mathfrak { S } } \exp \frac { w ^ { \prime } ( \rho _ { i } ) } { 2 n } \right)
$$

With this, we need that $\mathcal { M } ^ { s n s w } ( \succ )$ is obtained on eliminating the rotations in S starting from the man-oriented shortlists to complete the proof. It follows from the Theorem 1 and Definition 5 that $\mathcal { M } ^ { \mathrm { S N S W - A \breve { l } g } } ( \succ ) = \mathcal { M } ^ { s n s w } ( \succ )$ □

We will now present the proof of the running time complexity of $\mathrm { S N S W - A l \mathrm { g } } ,$

Theorem 3. The running time complexity of $S N S W { - } \bar { A } \bar { \mathcal { I } } g i s \mathcal { O } ( n ^ { 4 } \log n )$

Proof. Note that $P ^ { \prime }$ is a subgraph of the rotation poset $P .$ Thus, $V ( P ^ { \prime } ) \subseteq V ( P )$ . Since there exist at most $\mathcal { O } ( n ^ { 2 } )$ man-woman pairs, and each pair is part of at most one rotation, we have $\left| V ( P ^ { \prime } ) \right| \le \left| V ( P ) \right| = \mathcal { O } ( n ^ { 2 } )$

Observation 1. $| V |$ in $P ^ { \prime } ( s , t )$ is bounded by $\mathcal { O } ( n ^ { 2 } )$

Next, we observe that the rules R1 and R2 used to create a subgraph $P ^ { \prime }$ of $P$ associate each edge with a man-woman pair. Furthermore, each pair is associated with the creation of an edge at most once. Therefore, $| \mathbf { \bar { { E } } } ( P ^ { \prime } ) | \leq n * n = \mathcal { O } ( \mathbf { \bar { { n } } } ^ { 2 } )$ We formalize this in the next observation.

Observation $2 . \ | E |$ in $P ^ { \prime } ( s , t )$ is bounded by $\mathcal { O } ( n ^ { 2 } )$

Finally, we observe that the graph $P ^ { \prime } ( s , t )$ as adopted in SNSW-Alg has positive edge weights, and the time complexity of its construction is O. This observation directly follows from the construction of $\bar { P ^ { \prime } } ( s , t )$ from $P ^ { \prime }$ . All edges from the source s to the negative weight rotation nodes $( \rho )$ have capacities $| w ( \rho )$ |. All edges from positive weight rotation nodes (π) to the sink node t have capacities $w ( \pi )$ . All other edges in $P ^ { \prime } ( s , \ddot { t } )$ borrowed from $P ^ { \prime }$ have capacities set to infinity. Therefore, all edges have positive edge weights. Moreover, since the number of vertices in $P ^ { \prime }$ is upper-bounded by $O ( n ^ { 2 } )$ from Observation 1.

Observation 3. The graph $P ^ { \prime } ( s , t )$ as adopted in SNSW-Alg has positive edge weights, and its construction is $O ( n ^ { 2 } )$ time complexity.

Everything put together: Finding the max-flow using Dinic’s algorithm [Dinitz, 2006] with dynamic trees improvement by Sleator and Tarjan [1981] requires $\mathcal { O } ( | V | \cdot | E | \cdot \log | V | )$ time. Based on Observations 1, 2, and 3, this time complexity reduces to $\mathcal { O } ( n ^ { 4 } \log n ) , \mathrm { o r } \tilde { \mathcal { O } } ( n ^ { 4 } )$ □

## 4 Empirical Evaluation

We empirically evaluate the fairness of our proposed algorithm, SNSW-Alg, on randomly generated data. The goal of our experiments is to demonstrate the fairness of our approach relative to existing baselines across multiple dimensions. We begin by describing the experimental setup, followed by how we generate our data, the baseline algorithms we compare SNSW-Alg against, and the fairness metrics used to evaluate the matching. Finally, we describe the experiments performed and the observations.

## 4.1 Setup

Preference Generation. We study four distributions over the space of preference profiles $\succ$ , each capturing different structural assumptions about the input. We may categorize them into two groups:

i Uniform Distribution $\mathrm { D } _ { 1 } \colon \succ _ { m } \sim \mathcal { U } [ n ]$ and $\succ _ { w } \sim \mathcal { U } [ n ]$ , that is, preference is any permutation drawn uniformly at random.

ii Popularity-based Distributions [Zou et al., 2010]. Here, we initialize a popularity profile $p _ { a }$ of all $a \in N$ according to

a $\mathrm { P } _ { \mathcal { U } } .$ , where $p _ { a } \sim \mathcal { U } [ 0 , 1 ]$

b $\mathrm { P } _ { \mathcal { T } }$ , where $p _ { a } \sim \mathcal { T } [ 0 , 1 ]$ is triangular with peak at 0.5.

c $\mathrm { P } _ { \mathcal { N } } .$ where $p _ { a } \sim \mathcal { N } ( \mu = 0 , \sigma = 1 )$ is half-normal.

Next, we sample the first preference of $m _ { i } ~ ( w _ { j } )$ from W (M) as $\begin{array} { r } { \mathbb { P } _ { w _ { x } } ~ = ~ \frac { p ( w _ { x } ) } { \sum _ { w _ { z } \in W } p ( w _ { z } ) } ~ \Bigg ( \mathbb { P } ( m _ { k } ) ~ = ~ } \end{array}$ $\begin{array} { r } { \frac { p ( m _ { k } ) } { \sum _ { m _ { l } \in M } p ( m _ { l } ) } \bigg ) } \end{array}$ . Once the sampled woman (man) appends to $\succ _ { m } \left( \succ _ { w } \right)$ , we remove it from $W \left( M \right)$ and resample the next preference. This is performed iteratively till $\succ _ { a }$ is completed for agent a.

Benchmarks. We consider the following baselines to compare with SNSW-Alg.

i B<sub>1</sub>: Egalitarian-Alg [Irving et al., 1987]

ii $B _ { 2 } \colon { \mathrm { M i n - R e g r e t - A 1 g } }$ [Gusfield, 1987]

iii $B _ { 3 } \colon { \tt S e x - e q u a l - A l g }$ iterates over all possible stable matchings and determines the stable matching with optimal sex equality measure since the sex equal stable matching problem is NP-Hard [Kato, 1993]. The distributions we considered, on average, did not yield an exponential number of stable matchings (|Π|). Thus, we were able to complete all experiments within 6.5 hours, using a 13th Gen Intel i7-1355U (2.3 GHz) with 16GB of RAM.

Evaluation Metrics. We observe that while Baselines $B _ { 1 } , B _ { 2 }$ , and $B _ { 3 }$ achieve strong performance on their respective objectives, their performance deteriorates considerably on the other fairness metrics. By comparison, SNSW-Alg achieves a more balanced performance profile across all four metrics. There exist instances of ≻ such that SNSW-Alg outputs $\mathcal { M } ^ { s n s w }$ that performs relatively well across all four measures $\mu ^ { e } , \mu ^ { r } , \mu ^ { d } , \mu ^ { n s w }$ . For example, for ≻<sub>2</sub> given in Table $2 , \mathcal { M } ^ { s n s w } ( \succ )$ performs particularly well. While such examples are of theoretical importance, it is a worst-case example. Hence, instead of evaluating algorithms on worst-case performance, we employ statistical versions 4. We compute the empirical mean measures of all baselines across 100K instances.

$$
\begin{array} { r l } { \tilde { \mu } ^ { e , A L G } = \frac { \sum _ { i = 1 } ^ { K } \mu ^ { e } \left( \mathcal { M } _ { i } ^ { A L G } \right) } { K } } & { \tilde { \mu } ^ { r , A L G } = \frac { \sum _ { i = 1 } ^ { K } \mu ^ { r } \left( \mathcal { M } _ { i } ^ { A L G } \right) } { K } } \\ { \tilde { \mu } ^ { d , A L G } = \frac { \sum _ { i = 1 } ^ { K } \mu ^ { d } \left( \mathcal { M } _ { i } ^ { A L G } \right) } { K } } & { \tilde { \mu } ^ { n s w , A L G } = \frac { \sum _ { i = 1 } ^ { K } \mu ^ { n s w } \left( \mathcal { M } _ { i } ^ { A L G } \right) } { K } } \end{array}\tag{4}
$$

Setup We use $K ~ = ~ 1 0 0$ , 000 instances for $n \ = \ 2 5$ using all four distributions $\mathcal { U } , \mathrm { P } _ { \mathcal { U } } , \mathrm { P } _ { \mathcal { T } } , \mathrm { P } _ { \mathcal { N } }$ and compare $\tilde { \mu } ^ { e , A \tilde { L } G } , \tilde { \mu } ^ { r , A L G } , \tilde { \mu } ^ { d , A L G } , \tilde { \mu } ^ { n s w , A L G }$ . To compare these statistical versions uniformly, we normalize them to [0, 1], 0 being the best.

Consider the stable matching produced by algorithm ALG. Let the statistical regret measured be $\tilde { \mu } ^ { r , A L G }$ . Let the maximum and minimum values of $\tilde { \mu } ^ { r }$ measured across all baselines be $r _ { m a x } , r _ { m i n }$ . Then we normalize $\tilde { \mu } ^ { r , A L G }$ to $\frac { \tilde { \mu } ^ { r , A L G } - r _ { m i n } } { r _ { m a x } - r _ { m i n } }$ . We normalize $\begin{array} { r } { \tilde { \mu } ^ { n s w , A L G } \mathrm { t o } \frac { r _ { m a x } - \tilde { \mu } ^ { n s w , A L G } } { r _ { m a x } - r _ { m i n } } } \end{array}$ . Each baseline would achieve 0 on the measure it optimizes. The smaller the triangle’s area, the fairer the algorithm’s matching is.

Figure 1: Circular Plots for $n = 2 5$  
![](images/b1dea4cdc930a62fd73ad2561bb6f5e16ade3689bf53e72bb8c9123f6656da21.jpg)

Table 3: CoV across different Measures
<table><tr><td>Agents</td><td>Metric</td><td>Regret</td><td>Egalitarian</td><td>Disparity</td><td>SNSW</td></tr><tr><td rowspan="4">25</td><td> $\mu ^ { r }$ </td><td>0.060</td><td>0.055</td><td>0.055</td><td>0.058</td></tr><tr><td> $\mu ^ { e }$ </td><td>0.094</td><td>0.113</td><td>0.095</td><td>0.113</td></tr><tr><td> $\mu ^ { d }$ </td><td>0.518</td><td>0.521</td><td>0.597</td><td>0.523</td></tr><tr><td> $\dot { \mu } ^ { n s w }$ </td><td>0.062</td><td>0.061</td><td>0.061</td><td>0.062</td></tr><tr><td rowspan="4">50</td><td> $\mu ^ { r }$ </td><td>0.035</td><td>0.032</td><td>0.032</td><td>0.034</td></tr><tr><td> $\mu ^ { e }$ </td><td>0.082</td><td>0.093</td><td>0.082</td><td>0.092</td></tr><tr><td> $\mu ^ { d }$ </td><td>0.587</td><td>0.597</td><td>0.696</td><td>0.600</td></tr><tr><td> $\mu ^ { n s w }$ </td><td>0.045</td><td>0.045</td><td>0.044</td><td>0.045</td></tr></table>

Figure 2: Pareto Undominance of SNSW-Alg against other baselines  
![](images/3981dcaf1f1b989afc9ab35484f5af1d12cde4fc94dc7bc10fc5f34e2f33d054.jpg)

![](images/4415f011449bfb6fcb7c3ca103b0128c455b549cf01a7697539d0134b7b4ffee.jpg)

![](images/f92ecc9f20a0afdacbe4e4e06d45888b9832991f0bd0bf997c707c762c977c2d.jpg)

![](images/baeb7b74533e9a5c4f461ee43e1de1210af18cbea81ae41744ff350c40845d10.jpg)

![](images/920e9e8522bf7803d7eaf16177bd4525e739877a4c05b46d6ad40820efb702c5.jpg)

![](images/576e4a668fe2684d5f0a4d60ec207134a9a81f1b0e010c409f092c4f17131192.jpg)

## 4.2 Observations

We first consider SNSW-Alg against baselines for all 6 possible combinations of two fairness measures at a time. We observe in Figure 2 that $\mathsf { S N S W - A l g }$ is Pareto-undominated in such pairwise measure comparison for $n = 2 5$ . We measure the value of Vargha-Delaney $A ^ { 1 2 }$ measure comparing SNSW-Alg’s output with the baseline algorithms on all measures other than those it optimizes over. We observe that the measure is strictly greater than 0.5 for all such combinations.

We also compare all four measures simultaneously. We plot $\tilde { \mu } \mathrm { { s } }$ of all the algorithms on a circle (Figure 1). We observe that SNSW-Alg has the least area: at least 93.5% smaller than all benchmarks, demonstrating the statistical significance of SNSW-Alg in achieving fairness across multiple criteria

## 5 Related Work

Matching Theory. Gale and Shapley [1962] proposed an algorithm for assigning students to colleges (or residents to hospitals) along with preserving the stability of the matching. Roth et al. [2004] proposed a design to build upon kidney exchange practices, taking into account properties of efficiency and incentive compatibility. Delacrétaz et al. [2016] proposed several refugee resettlement mechanisms that improve match efficiency as well as incentivize refugees to report their choices, and respect the priorities of local areas. More recently, Aziz et al. [2022] authored a survey on many-to-one matching under various market constraints, such as lower quotas, regional constraints, diversity constraints, multidimensional capacity constraints, and hereditary constraints. Kamiyama [2020] studied the problem of popular matching under matroidal constraints, and proposed a polynomial-time algorithm to find the largest size popular matching under the given constraints.

Fairness in Stable Marriage Problem. Irving et al. [1987] established a fairness criterion to measure how equitable any given stable matching is in the context of the stable marriage problem, and proposed an $\mathcal { O } ( n ^ { 4 } )$ algorithm to find such an optimal (egalitarian) stable matching. Gusfield [1987] proposed another fairness metric for the stable marriage problem called regret and proposed an $\mathcal { O } ( \bar { n } ^ { 2 } )$ algorithm to find the minimum-regret stable matching. Gusfield and Irving [1989] studied the parametric stable marriage problem and analyzed the nature of the parametrized measure and time complexity of finding the optimal solution to the parametric stable marriage problem. Kato [1993] studied the complexity of another problem - sex equal stable marriage, which aimed to minimize the net disparity between the assigned matches to the two sides, concluding that this problem is $_ { \mathrm { N P - H a r d } }$ . More recently, Alkan and Yildiz [2022] introduced “modular stable matching rules” as a framework for equity in stable matchings, establishing an equivalence between modular optimization and an ordinal condition called convexity. They also propose a new equity criterion called “equity undominance” and characterize the modular rules that satisfy it.

Nash social welfare. Brânzei et al. [2017] study the question of (approximately) implementing the objective of Nash social welfare for fair allocation of resources in the presence of strategic agents in relation to one-sided markets. Barman et al. [2019] propose algorithms for fair allocation of indivisible goods to strategic agents in single-parameter environments along with theoretical Nash welfare guarantees. Barman et al. [2022] utilizes the Nash social welfare function to study fairness and efficiency in coverage problems. Jain and Vaish [2024] studied the complexity of problems in many-to-one matching (not necessarily stable) that aim to maximize Nash social welfare in two-sided markets. Garg et al. [2025] study the interplay between competitive equilibrium and Nash welfare maximizing allocations in the context of the divisible items allocation problem. To summarize, Nash social welfare has been studied extensively in one-sided markets. As per our knowledge, the work of Jain and Vaish [2024] is the sole work on the application of Nash social welfare in two-sided markets.

## 6 Conclusion

In this paper, we studied the problem of obtaining a fair and stable matching in two-sided markets. Matching produced to optimize any of the existing measures performs poorly on other measures. To this end, we proposed optimizing Nash social welfare under rank-induced utilities while ensuring stability. We then designed SNSW-Alg to output the NSW optimal stable matching. We then proved the correctness of SNSW-Alg and showed its running time complexity is $\tilde { \mathcal { O } } ( n ^ { 4 } )$ . We then empirically study the fairness of matching with different approaches. Empirically, we demonstrate on 100K randomly generated instances per setting, across uniform and popularity-based preference distributions, that, overall, the NSW-optimal stable matching statistically outperforms egalitarian, regret, or sex-equality. Our study shows that the proposed measure $\mu ^ { n s w }$ , coupled with stability, yields a fair outcome. We leave further theoretical investigation into its guarantees relative to other measures as future work

## References

Ahmet Alkan and Kemal Yildiz. Equitable stable matchings under modular assessment. Available at SSRN 4030050, 2022.

Haris Aziz, Péter Biró, and Makoto Yokoo. Matching market design with constraints. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 12308–12316, 2022.

Siddharth Barman, Ganesh Ghalme, Shweta Jain, Pooja Kulkarni, and Shivika Narang. Fair division of indivisible goods among strategic agents. arXiv preprint arXiv:1901.09427, 2019.

Siddharth Barman, Anand Krishna, Y Narahari, and Soumyarup Sadhukhan. Nash welfare guarantees for fair and efficient coverage. In International Conference on Web and Internet Economics, pages 256–272. Springer, 2022.

Simina Brânzei, Vasilis Gkatzelis, and Ruta Mehta. Nash social welfare approximation for strategic agents. In Proceedings of the 2017 ACM Conference on Economics and Computation, pages 611–628, 2017.

Eric Budish and Estelle Cantillon. Strategic behavior in multi-unit assignment problems: Lessons for market design. Technical report, Working Paper, 2009.

Ioannis Caragiannis, David Kurokawa, Hervé Moulin, Ariel D Procaccia, Nisarg Shah, and Junxing Wang. The unreasonable fairness of maximum nash welfare. ACM Transactions on Economics and Computation (TEAC), 7(3): 1–32, 2019.

David Delacrétaz, Scott Duke Kominers, and Alexander Teytelboym. Refugee resettlement. University of Oxford Department ofEconomics Working Paper, 2016.

Yefim Dinitz. Dinitz’algorithm: The original version and even’s version. In Theoretical Computer Science: Essays in Memory of Shimon Even, pages 218–240. Springer, 2006.

David Gale and Lloyd S Shapley. College admissions and the stability of marriage. The American mathematical monthly, 69(1):9–15, 1962.

Jugal Garg, Yixin Tao, and László A Végh. Approximating competitive equilibrium by nash welfare. In Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2538–2559. SIAM, 2025.

Dan Gusfield. Three fast algorithms for four problems in stable marriage. SIAM Journal on Computing, 16(1):111–128, 1987.

Dan Gusfield and Robert W Irving. Parametric stable marriage and minimum cuts. Information Processing Letters, 30 (5):255–259, 1989.

Robert W Irving. An efficient algorithm for the “stable roommates” problem. Journal ofAlgorithms, 6(4):577–595, 1985.

Robert W Irving and Paul Leather. The complexity of counting stable marriages. SIAM Journal on Computing, 15(3): 655–667, 1986.

Robert W Irving, Paul Leather, and Dan Gusfield. An efficient algorithm for the “optimal” stable marriage. Journal of the ACM (JACM), 34(3):532–543, 1987.

Pallavi Jain and Rohit Vaish. Maximizing nash social welfare under two-sided preferences. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 9798–9806, 2024.

Naoyuki Kamiyama. Popular matchings with two-sided preference lists and matroid constraints. Theoretical Computer Science, 809:265–276, 2020.

Mamoru Kaneko and Kenjiro Nakamura. The nash social welfare function. Econometrica: Journal ofthe Econometric Society, pages 423–435, 1979.

Akiko Kato. Complexity of the sex-equal stable marriage problem. Japan Journal of Industrial and Applied Mathematics, 10:1–19, 1993.

Donald E. Knuth. Stable Marriage and Its Relation to Other Combinatorial Problems: An Introduction to the Mathematical Analysis ofAlgorithms, volume 10. American Mathematical Society, Providence, RI, 1997.

Hervé Moulin. Fair division and collective welfare. MIT press, 2004.

John F Nash et al. The bargaining problem. Econometrica, 18(2):155–162, 1950.

Pablo Revilla. Many-to-one matching when colleagues matter. SSRN, 2007.

Alvin E Roth, Tayfun Sönmez, and M Utku Ünver. Kidney exchange. The Quarterly journal ofeconomics, 119(2): 457–488, 2004.

Daniel D Sleator and Robert Endre Tarjan. A data structure for dynamic trees. In Proceedings ofthe thirteenth annual ACM symposium on Theory ofcomputing, pages 114–122, 1981.

James Zou, Sujit Gujar, and David Parkes. Tolerable manipulability in dynamic assignment without money. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 24, pages 947–952, 2010.

## A Notation

<table><tr><td rowspan=1 colspan=1> $\overline { { M } }$ </td><td rowspan=1 colspan=1>Men side of market: $\overline { { \{ m _ { 1 } , m _ { 2 } , \ldots , m _ { n } \} } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { W }$ </td><td rowspan=1 colspan=1>Women side of market: $\{ w _ { 1 } , w _ { 2 } , \ldots , w _ { n } \}$ </td></tr><tr><td rowspan=1 colspan=1> $n$ </td><td rowspan=1 colspan=1>Number of men and women  ${ \overline { { M \left| = \right| } } } \ W \left| = n \right.$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { N } }$ </td><td rowspan=1 colspan=1>Set of all men and women: M ∪ W</td></tr><tr><td rowspan=1 colspan=1> $\overline { { r ( a , b ) } }$ </td><td rowspan=1 colspan=1>Rank of agent $b \operatorname { i n } \succ _ { a }$ </td></tr><tr><td rowspan=1 colspan=1> $u r ( m _ { i } , w _ { j } )$ </td><td rowspan=1 colspan=1>Utility of man $m _ { i }$ on match with woman $w _ { j }$ </td></tr><tr><td rowspan=1 colspan=1> $u r ( w _ { k } , m _ { l } )$ </td><td rowspan=1 colspan=1>Utility of woman $w _ { k }$ on match with man $m _ { l }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { M } } }$ </td><td rowspan=1 colspan=1>Any stable matching between M and W</td></tr><tr><td rowspan=1 colspan=1> ${ \overline { { \mathcal { M } ^ { o } } } }$ </td><td rowspan=1 colspan=1>Man-optimal stable matching</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { M } ^ { z } } }$ </td><td rowspan=1 colspan=1>Woman-optimal stable matching</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { M } ^ { r } } }$ </td><td rowspan=1 colspan=1>Min-Regret stable matching</td></tr><tr><td rowspan=1 colspan=1> $\textstyle { \overline { { \mathcal { M } } } } ^ { e }$ </td><td rowspan=1 colspan=1>Egalitarian stable matching</td></tr><tr><td rowspan=1 colspan=1> ${ \overline { { \mathcal { M } ^ { d } } } }$ </td><td rowspan=1 colspan=1>Sex-equal stable matching</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { M } ^ { n s w } } }$ </td><td rowspan=1 colspan=1>Nash optimal matching</td></tr><tr><td rowspan=1 colspan=1> $\mathscr { M } ^ { s n s w }$ </td><td rowspan=1 colspan=1>Nash stable matching</td></tr></table>

## B Experimental Evaluations

![](images/31c87942b665a4d5da9b7ef6abe0377ce5028ecfc30c30ff3fe71b3497320244.jpg)  
Figure 3: Circular Plot for Uniform Distribution

![](images/dd26b6cf92b2e12bfce504c2cefb2c154edf189fe53ce85c38a19f96e4622c1f.jpg)  
Figure 4: Circular Plot for Triangular Distribution

![](images/3bdce9603ecce2b865e17a45d9a20386c8aa3429d1a43d9eefca857291acddbd.jpg)  
Figure 5: Circular Plot for Normal Distribution

![](images/0b82221291c9ffca81542c978b063620a813edcecf7d5a6dd687ecadb08dcb3c.jpg)  
Figure 6: Circular Plot for Uniform Default Distribution