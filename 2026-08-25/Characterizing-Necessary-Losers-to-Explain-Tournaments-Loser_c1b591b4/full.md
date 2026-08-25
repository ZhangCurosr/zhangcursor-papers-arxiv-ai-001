# Characterizing Necessary Losers to Explain Tournaments Losers

Clément Contet<sup>1,2</sup>, Umberto Grandi<sup>1,3</sup>, and Jérôme Mengin<sup>1,2</sup>

<sup>1</sup> Institut de Recherche en Informatique de Toulouse (IRIT)

<sup>2</sup> Université de Toulouse

3 Université Toulouse Capitole

Abstract. We study the problem of formally explaining why a candidate was not selected by a given tournament rule, by identifying subtournaments in which the candidate loses independently of how the rest of the tournament is completed. We define destructive minimal supports as any minimal sub-tournaments satisfying this property, which in formal explainable artificial intelligence correspond to abductive explanations for the question “Why does the loser lose the tournament?”. For six common tournament solutions (maximin, uncovered set and its weighted variant, top-cycle, Copeland, and Borda) we provide characterizations of when a candidate is either a necessary loser or a possible winner, we determine the size of the smallest destructive minimal supports, complemented by polynomial-time algorithms for their computation except for the case of the Borda rule which is suspected to be NP-complete.

Keywords: Computational Social Choice · Voting · Explainability.

## 1 Introduction

Why do people accept the outcome of decision making processes? If it seems natural that people follow decisions which are favorable to them, it is less clear to see why they would conform to disadvantageous outcomes. Procedural justice theory, developed in the late 1990s and early 2000s by Tyler [29,30] and Hibbing and Theiss-Morse [21], claims that the willingness to accept and to obey the outcome of a decision making process not only depends on the outcome per se but also on the process that leads to it. If the primary influence on the individual position toward an outcome is its favorability, a process perceived with a higher procedural fairness, i.e., which treats all stakeholders fairly, increases the legitimacy and trustworthiness of the decision, the process in itself and more generally the responsible entity. This in turn leads to more acceptance and conformance with the decision. More recently, Carman [7] showed that this theory holds for participatory democracy, in particular for petitioning systems. It is important to realize that it is all about perceived fairness, as Tyler states: “When authorities are presenting their decisions to the people influenced by them, they need to make clear that they have listened to and considered the arguments made. They can do so by accounting for their decisions.” [30]. A practical example of this is the work of Suryanarayana et al. [28] establishing that providing explanations for election outcomes could increase satisfaction and acceptance of voters, especially for those with less favorable outcomes. These results as well as numerous voices in the computational social choice community [19] have called for eforts to make collective decision-making processes more accessible to a wider audience by developing tools to explain their inner workings. More generally, this movement is in line with the field of Explainable Artificial Intelligence (XAI) which aims at making algorithms more understandable to human users (see Miller [24] for an introduction).

Building on previous work on transparency [11], de Fine Licht and de Fine Licht [15] and Grimmelikhuijsen [18] showed that from a social sciences perspective, the basic transparency approach consisting of simply making public all the information to enable anyone to recompute and verify algorithmic decisions is insuficient to foster trust.<sup>1</sup> For collective decisions, this implies that simply publishing the individual vote counts and the rule used may not be enough.

In the computational social choice literature, multiple methods to specifically explain or justify the outcome of an election have been proposed. In a classical social choice approach, a first line of works unfolds some logical reasonings based on desirable axiomatic properties to justify the election of one alternative [2,6,25,26]. A second method presents the stakeholders with manually or automatically curated statistics on the expressed preferences supporting a specific outcome [28]. Finally, a third line of work [9,10] uses abductive reasoning to explain why an alternative is in the winning set. All approaches have their strengths and weaknesses, however, each only allows to generate explanations supporting why an alternative was selected and not why an alternative was not selected. We see this as an important drawback as procedural justice theory has shown that explanations of decisions are particularly important to maintain trust and legitimacy in the case of an adverse outcome [5,8].

Inspired by the literature on control and bribery [13,14] which distinguishes between constructive bribery where the goal is to change votes to make a losing candidate elected, and destructive bribery where an elected candidate becomes unelected, we introduce the notion of destructive minimal supports to answer the question “Why was this candidate not selected?” with abductive reasoning. To produce compact explanations for the election outcome, our approach uses tournaments which are a classical structure in social choice to compactly represent pairwise comparisons over a set of possible alternatives such as contending teams in sports, preferences expressed over candidates in an election or possible outcomes in a decision making problem (for an introduction see [3,16]).

Destructive minimal supports are partial sub-tournaments of the initial tournament in which the scrutinized losing candidate loses in all possible completions. That is, the candidate is a necessary loser of the sub-tournament, or, equivalently, not a possible winner. This problem can be seen as the dual of the Margin of Victory (MoV) problem [4,12] where one is looking for minimal changes in the tournament to change the outcome of the election. More generally, this work fits in the literature on the problem of finding necessary and possible winners in election with partial information [1,22,23,27,31].

Paper structure. In Section 2, we recall classical notions on tournaments. In Section 3, we define destructive minimal supports for the losing candidate l for a tournament solution S in a tournament G as the set of inclusion minimal subtournaments of G where l is a necessary loser, i.e., loses in all completions of those sub-tournaments. In Section 4, we characterize partial tournaments where a specific candidate is a necessary loser, or, equivalently, not a possible winner for the following six tournament solutions : the maximin rule, the uncovered set and its weighted variant, the top-cycle, the Copeland rule, and the Borda rule. Finally, in Section 5 we study the smallest destructive minimal supports that is the destructive minimal supports comprised of the smallest amount of pairwise comparisons, we provide explicit formulas and bounds for their size, and we show how to obtain them eficiently. See Table 1 for a preview of our results. All missing proofs are available in the Appendix.

## 2 Preliminaries

Let the number of voters n be a strictly positive integer. Aziz et al. [1] define a partial n-weighted tournament as a pair $G = ( \mathcal { C } , \mu )$ where C is a nonempty finite set of candidates and $\mu : { \mathcal { C } } \times { \mathcal { C } } \to \{ 0 , \dots , n \}$ a weight function such that for all distinct $x , y \in { \mathcal { C } } , \mu ( x , y ) + \mu ( y , x ) \leq$ n and $\mu ( x , x ) = 0 . \mathrm { ~ A ~ }$ (complete) n-weighted tournament satisfies for all distinct $x , y \in \mathcal { C } , \mu ( x , y ) + \mu ( y , x ) = n$ . Naturally, an (unweighted) tournament is a 1-weighted tournament.

Example 1. In Figure 1a, the edge of G between the pair of alternatives a and b shows that a is preferred to b. In Figure 1b, the edge from a to b with a weight of 2 in $G _ { w }$ shows that a is preferred to b by 2 voters.

Given a tournament G it is then possible to select a set of desirable candidates using a tournament solution S (also called tournaments rule) which is a function taking a tournament as input and returning the winners set as a nonempty subset of candidates noted S(G). We focus on the following set of classical tournament solutions.

– The top cycle (TC) is the unique minimal dominant nonempty subset of candidates of an unweighted tournament $G ,$ where a nonempty subset of candidates $A \subseteq { \mathcal { C } }$ is called dominant in $G = ( { \mathcal C } , \mu )$ if for each candidates $x \in A$ and $y \in \mathcal { C } \setminus A , \mu ( x , y ) = 1$ . Alternatively, TC can also be defined as the set of candidates that can reach every other candidate via a directed path in $G \mathrm { { ~ o r ~ } }$ , in terms of graph theory, as the unique (since G is complete) top strongly connected component of $G .$

<table><tr><td>Tournament Solution</td><td>Necessary Loser Characterization</td><td>Upper bound on SdMS size</td></tr><tr><td>Top Cycle (TC)</td><td> $\exists K \subseteq { \mathcal { C } } \setminus \{ l \} \ { \mathrm { s . t . } } \ K \neq \emptyset ,$   $\forall c , c ^ { \prime } \in K \times ( \mathcal { C } \setminus K ) , \mu ( c , c ^ { \prime } ) = 1 .$  (Theorem 1)</td><td> $\left\lceil { \frac { m } { 2 } } \right\rceil \left\lfloor { \frac { m } { 2 } } \right\rfloor$  (Theorem 5)</td></tr><tr><td>Borda (BO)</td><td> $\exists K \subseteq { \mathcal { C } } \setminus \{ l \} \mathrm { ~ s . t . }$   $\operatorname* { m i n } _ { G ^ { \prime } \in [ G ] } \mathbb { E } _ { c \in K } \sigma _ { \mathrm { B O } } ( c , G ^ { \prime } ) > \operatorname* { m a x } _ { G ^ { \prime } \in [ G ] } \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) .$   $( \mathrm { T h e o r e m ~ 2 } , \mathrm { S c h w a r t z ~ [ 2 7 ] } )$ </td><td> $n ( m - 1 ) + 1$  (Theorem 6)</td></tr><tr><td>Copeland (CO)</td><td> $\exists K \subseteq { \mathcal { C } } \setminus \{ l \} \mathrm { ~ s . t . }$   $\operatorname* { m i n } _ { G ^ { \prime } \in [ G ] } \mathbb { E } _ { c \in K } \sigma _ { \mathrm { C O } } ( c , G ^ { \prime } ) > \operatorname* { m a x } _ { G ^ { \prime } \in [ G ] } \sigma _ { \mathrm { C O } } ( l , G ^ { \prime } ) .$  (Corollary of Theorem 2)</td><td>m (Theorem 7)</td></tr><tr><td>Maximin (MM)</td><td> $\exists \mathrm { ~ a ~ t r e e ~ } T = ( K , E ) \mathrm { ~ s . t . ~ } K \subseteq \mathcal { C } \setminus \{ l \} ,$   $\forall c , c ^ { \prime } \in K \times ( \mathcal { C } \setminus \{ l , c \} ) , \{ c , c ^ { \prime } \} \not \in E$   $\begin{array} { r } { \implies \mu ( c , c ^ { \prime } ) > n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in \mathcal { C } } \mu ( c ^ { \prime \prime } , l ) . } \end{array}$  (Theorem 3)</td><td> $\left\lceil { \frac { n + 1 } { 2 } } \right\rceil ( m - 2 ) + n + 1$  (Theorem 8)</td></tr><tr><td>Weighted Uncovered Set (wUC)</td><td>∃ a tree  $T = ( K , E ) { \mathrm { ~ s . t . ~ } } K \subseteq { \mathcal { C } } \setminus \{ l \} ,$   $\begin{array} { r } { \forall c , c ^ { \prime } \in K \times ( \mathcal { C } \setminus \{ l , c \} ) , \mu ( c , l ) \geq \frac { n } { 2 } , } \end{array}$   $\{ c , c ^ { \prime } \} \not \in E \implies \mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) \geq \stackrel {  } { n . ^ { * } }$  (Theorem 4)</td><td> $n ( m - 2 ) + \lceil \frac { n + 1 } { 2 } \rceil$  (Theorem 9)</td></tr><tr><td>Uncovered Set (UC)</td><td> $\exists \mathrm { ~ a ~ t r e e ~ } T = ( K , E ) \mathrm { ~ s . t . ~ } K \subseteq \mathcal { C } \setminus \{ l \} ,$   $\forall c , c ^ { \prime } \in K \times ( \mathcal { C } \setminus \{ l , c \} ) , \mu ( c , l ) = 1 ,$   $\{ c , c ^ { \prime } \} \not \in E \implies \mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) \geq 1 . ^ { * }$  (Corollary of Theorem 4)</td><td>m − 1 (Corollary 1)</td></tr></table>

\* At least one of all the inequalities has to be strict.  
Table 1: Overview of our results for a tournament $G = ( \mathcal { C } , \mu )$ with n voters, m candidates in ${ \mathcal { C } } ,$ and l the losing candidate considered. $\sigma _ { \mathrm { B O } } ( l , G )$ and $\sigma _ { \mathrm { C O } } ( l , G )$ are the Borda and Copeland scores of l in G. [G] is the set of all completions of G. All bounds are tight.

![](images/9bd908242015687f816619f8829a82f00d636d7f24e67d15a7869d185cc9b1cf.jpg)  
(a) G

![](images/e0dc87bc6481814818f664f02ff70a3a87188acf547a76e3176967a5ed1a5e2d.jpg)  
(b) $G _ { w }$

![](images/ee332d9ee41b701e1f9c7a4d71c7b99ce21ee9f1ad4646c135a0307f129c1c34.jpg)  
(c) $G _ { p }$  
Fig. 1: An (unweighted) tournament G (a), a 5-weighted tournament $G _ { w } ( \mathbf { b } )$ , and an (unweighted) partial tournament $G _ { p } \ ( \mathrm { c } )$ . Throughout this paper, edges with a weight of 0 are not shown and edge labels are omitted in the unweighted case.

– The weighted uncovered set (wUC) is the nonempty subset of candidates that are not strictly weighted covered by any other candidate in a weighted tournament $G = ( { \mathcal C } , \mu )$ where a candidate $x \in { \mathcal { C } }$ is said to strictly cover another candidate $y \in { \mathcal { C } }$ if for all $z \in \mathcal { C } , \mu ( x , z ) \geq \mu ( y , z )$ and at least one inequality is strict. For unweighted tournaments, we simply talk about uncovered set (UC).

– The Borda score of a candidate $c \in { \mathcal { C } }$ in a weighted tournament $G = ( \mathcal { C } , \mu )$ is $\begin{array} { r } { \sigma _ { \mathrm { B O } } ( c , G ) = \sum _ { c ^ { \prime } \in \mathcal { C } } \mu ( c , c ^ { \prime } ) } \end{array}$ . The Borda rule (BO) selects candidates that have a maximal Borda score.

The Copeland score of a candidate $c \in { \mathcal { C } }$ in an unweighted tournament $G = ( { \mathcal C } , \mu )$ is $\sigma _ { \mathrm { C O } } ( c , G ) = | \{ c ^ { \prime } : c ^ { \prime } \in \mathcal { C } , \mu ( c , c ^ { \prime } ) = 1 \} |$ . The Copeland rule (CO) selects candidates that have a maximal Copeland score. It can be seen as the restriction of BO to unweighted tournaments.

– The maximin score of a candidate $c \in { \mathcal { C } }$ in a weighted tournament $G =$ $( \mathcal { C } , \mu )$ is $\begin{array} { r } { \sigma _ { \mathrm { M M } } ( c , G ) = \operatorname* { m i n } _ { c ^ { \prime } \in \mathcal { C } \setminus \{ c \} } \mu ( c , c ^ { \prime } ) } \end{array}$ . The maximin rule (MM) selects candidates that have a maximal maximin score.

## 3 (Smallest) Destructive Minimal Supports for Tournaments

To explain why a candidate c loses a tournament $G ,$ abductive reasoning extracts a subset minimal set of features of the tournament G ensuring that the scrutinized candidate loses. We call destructive minimal supports such minimal sub-tournaments of G where c loses independently of the rest of the tournament. Given two partial tournaments $G = ( { \mathcal C } , \mu )$ and $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ , we say that $G ^ { \prime }$ is an extension of G denoted $G \subseteq G ^ { \prime }$ if for all distinct $x , y \in \mathcal { C } , \mu ( x , y ) \leq \mu ^ { \prime } ( x , y )$ We refer to [G] as the set of all complete extensions of G.

We introduce the notion of necessary losers to formalize this idea of a candidate losing in all completions of a partial tournament. This definition can be viewed as a parallel to the notion of necessary winners, i.e., candidates winning no matter what, and is equivalent to not being a possible winner or a candidate that can win in at least one completion (see Konczak and Lang [22]).

Definition 1. Given n voters, a partial n-weighted tournament G, a tournament solution S and a candidate $c \in { \mathcal { C } } ,$ , c is a necessary loser for $G \ ( w . r . t \ S )$ if for every completion $G ^ { \prime } \in [ G ]$ we have that $c \notin S ( G ^ { \prime } )$ . We write $c \in N L _ { S } ( G )$

Example 2. Consider the partial tournament in Figure 1c and the Copeland rule. Since a already secured 3 wins, c and d cannot catch back as their Copeland score is respectively 1 and 0 and they can only improve it by 1 by being preferred to e. On the contrary, a can win, typically if e is never preferred. b can win if e is preferred to a and b is preferred to e. e can win as it could be a Condorcet winner for instance. Hence, c and d are the necessary Copeland-losers for G.

We now define destructive minimal supports for tournaments.

Definition 2. Given n voters, an n-weighted tournament $G = ( { \mathcal C } , \mu )$ , a tournament solution S and a losing candidate $l \in \mathcal { C } \setminus S ( G )$ , a destructive minimal support (dMS) for $l \not \in S ( G )$ is a partial tournament $G ^ { \prime } \subseteq G$ such that:

(a) $l \in N L _ { S } ( G ^ { \prime } )$

(b) $G ^ { \prime } \mathit { i s } \subseteq - m i n i m a l ,$ i.e., all partial tournaments $G ^ { \prime \prime } \subsetneq G ^ { \prime }$ are such that $l \notin$ $N L _ { S } ( G ^ { \prime \prime } )$

![](images/4bfc449a08cabd0d33d8125c7ce1a200638bcc8dc4db220af036d80bd9febdda.jpg)  
(a) G

![](images/8798db8bc3a4f9c50c78654823e2481019ae9659f4561c38d6e70191ad04c875.jpg)  
(b) X

![](images/312dea9cbe3c72e3ec7a486060054e7c91ed0436144672b350d15b8689e7bfb7.jpg)  
(c) Y  
Fig. 2: A tournament G (a), and X and Y, two dMSs for d $\not \in \mathrm { C O } ( G )$ (b) and (c).

Example 3. Consider the Copeland rule and the tournament G in Figure 2a. Clearly, $c \not \in \mathrm { C O } ( G )$ . To produce a dMS for d $\not \in \mathrm { C O } ( G )$ like X in Figure 2b, one has to find a partial sub-tournament of G such that independently of the way it is completed, there always exists a candidate with a better Copeland score than d. In X, this is achieved by using a mix of pairwise comparisons lost by d and won by a stronger candidate a. Indeed, at best d can achieve a score of 2 while a will at least score 3. This sub-tournament is subset minimal as removing a defeat from d breaks the previous reasoning. The case of the dMS for d $\not \in { \mathrm { C O } } ( G )$ in Figure 2c is similar. However, here it is suficient to attribute 3 losses to d as it is impossible to complete the rest of the partial tournament without giving at least a score of 2 to another candidate. d loses a strict majority of its comparisons and there always exists a candidate in the tournament that wins a majority of them by the pigeonhole principle.

As seen in the previous example, there can exist multiple dMSs for a given losing candidate. We focus on the simpler ones in line with Grice’s manner criterion [17] which suggests to pick the briefest explanation. To measure the dMSs size, we use total number of pairwise comparisons in the partial tournament like in Contet et al. [10] and the microbribery setting of Faliszewski et al. [14].

Definition 3. Given n voters, a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ , and a losing candidate $l \in \mathcal { C } \setminus S ( G )$ , we define the size of a dMS ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ for $l \ \not \in \ S ( G )$ as $\begin{array} { r } { | \mathcal { X } | = \sum _ { ( c , c ^ { \prime } ) \in \mathcal { C } ^ { 2 } } \mu _ { \mathcal { X } } ( c , c ^ { \prime } ) } \end{array}$ . X is a smallest destructive minimal support (SdMS) $f o r \ l \ \not \in { \cal S } ( G )$ if and only if for all dMSs Y for $l \not \in S ( G )$ , we have $| \mathcal { X } | \leq | \mathcal { V } |$

## 4 Characterizing Necessary Losers

## 4.1 Top Cycle

The characterization of top cycle is relatively straightforward. Essentially, a candidate l can be a necessary loser if and only if there always exists a strongly connected component inaccessible from l in each completion. For it to be the case, a “one-way frontier” has to split the initial partial tournament in two.

Theorem 1. Given a partial tournament $G = ( { \mathcal C } , \mu )$ , a candidate $l \in \mathcal { C }$ is a necessary TC-loser for G if and only if exists $K \subseteq { \mathcal { C } } \setminus \{ l \}$ with $K \neq \emptyset$ such that for each pair $( c , c ^ { \prime } ) \in K \times ( \mathcal { C } \setminus K ) , \mu ( c , c ^ { \prime } ) = 1$

## 4.2 Borda

Schwartz proved the characterization for Borda in the 1960s [27]. It was originally expressed with flow networks so we translate it in our framework.

Theorem 2. Schwartz $I 2 \eta I .$ Given n voters and a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , a candidate $l \in \mathcal { C }$ is a necessary Borda loser for G if and only if exists $K \subseteq \mathcal { C } \setminus \{ l \}$ such that

$$
\operatorname* { m i n } _ { G ^ { \prime } \in [ G ] } \mathbb { E } _ { c \in K } \sigma _ { \mathrm { B O } } ( c , G ^ { \prime } ) > \operatorname* { m a x } _ { G ^ { \prime } \in [ G ] } \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) .
$$

## 4.3 Maximin and Weighted Uncovered Set

In this section we provide characterizations of partial tournaments where a specific candidate l is a necessary loser for the maximin rule and the weighted uncovered set rule. We proceed by reducing the problem of determining if l is a necessary loser in a tournament $G$ to the existence of a perfect matching in a specific bipartite graph that we call the l-deficient incidence graph of G. This approach generalizes an idea used by Aziz et al. to prove that the possible winner problem for maximin is in P (Theorem 11 in [1]). We then use Hall’s mariage theorem [20] which characterizes bipartite graphs admitting perfect matchings to characterize partial tournaments where l is a necessary loser.

Let $B = ( X , Y , Z )$ be a bipartite graph where X and Y are vertex sets and Z the edge set. For each subset $W \subseteq X$ , the neighborhood of W in $B ,$ noted $N _ { B } ( W )$ is the subset of vertices in Y adjacent in B to vertices in W. An X-perfect matching (or simply a perfect matching) is a subset of edges $M \subseteq Z$ such that each vertex of X is adjacent to exactly one edge in M.

We distinguish the directed edge from x to $y ,$ noted with parentheses $( x , y )$ from the (undirected) edge between x and y, noted with braces $\{ x , y \}$

Definition 4. Given a directed graph $G = ( V , E )$ where V is the set of vertices, E the set of edges and a vertex v $\in V .$ , we call the v-deficient incidence graph of G the bipartite graph $I _ { G } ^ { v } = ( { \cal { X } } , { \cal { Y } } , Z )$ where $X = V \setminus \{ v \} , Y = \{ \{ x , y \} : x , y \in$ $V ^ { 2 } , ( x , y ) \in E \}$ and $Z = \{ \{ x , \{ x , y \} \} : x , y \in V ^ { 2 } , ( x , y ) \in E \}$ , i.e., a vertex $x \neq v$ is adjacent to an edge e in $I _ { G } ^ { v } \mathrm { \Delta } i f$ and only if e is an edge leaving x in G.

![](images/fa44867bc8cd46521335ebf1416359686f1b88d4d6dbd4a08b2a862d5269bacf.jpg)  
Fig. 3: A directed graph G (a) and its j-deficient incidence graph $I _ { G } ^ { j } \ ( \mathrm { b ) }$

Example 4. Consider the graph $G = ( V , E )$ in Figure 3a. To build its j-deficient incidence graph $I _ { G } ^ { j } \ = \ ( X , Y , Z )$ in Figure 3b simply take the original set of vertices without $\boldsymbol { j } , \boldsymbol { \overset { \cdot } { X } } = V \setminus \{ \boldsymbol { j } \} = \{ a , b , c , d , e , f , g , h , i \}$ , add each pair of vertices containing an edge in the original graph $Y = \{ \{ x , y \} : x , y \in V ^ { 2 } , ( x , y ) \in E \}$ and then link vertices with their out-going edges in G.

We identify graphs whose deficient incidence graph has no perfect matching.

Lemma 1. Given a directed graph $G = ( V , E )$ where V is the set of vertices and E the set of edges, and a vertex $v \in V$ , the v-deficient incidence graph of G, $I _ { G } ^ { v } = ( { \cal { X } } , { \cal { Y } } , Z )$ , does not have an X-perfect matching if and only if there exists a tree $T = ( K , E _ { T } )$ with $K \subseteq V \setminus \{ v \}$ such that for each pair $( c , c ^ { \prime } ) \in K \times V$ with $c \neq c ^ { \prime } , \ i f \ \{ c , c ^ { \prime } \} \ \notin E _ { T }$ then $( c , c ^ { \prime } ) \notin E$

Proof. Given a directed graph $G = ( V , E )$ where V is the set of vertices and E the set of edges, a vertex $v \in V$ and the v-deficient incidence graph of G $I _ { G } ^ { v } = ( { \cal { X } } , { \cal { Y } } , Z )$ . Given a subgraph $S \subseteq G$ , let $V _ { S }$ be the set of vertices contained in S and $E _ { S }$ the set of edges. We define the undirected version of $S , \widetilde { S } = ( V _ { S } , \widetilde { E _ { S } } )$ where $\widetilde { E _ { S } } = \left\{ \left\{ c , c ^ { \prime } \right\} : c , c ^ { \prime } \in V _ { S } ^ { 2 } , ( c , c ^ { \prime } ) \in E _ { S } \right\}$ . To prove Lemma 1, we use a classical result from matching theory.

Hall’s marriage theorem [20]. Let $G = ( X , Y , Z )$ be a finite bipartite graph with bipartite sets X and Y and edge set Z. An X-perfect matching exists if and only if for each subset $W \subseteq X , | W | \leq | N _ { G } ( W )$ |.

$( \implies )$ Suppose $I _ { G } ^ { v }$ has no X-perfect matching. Then, according to Hall’s marriage theorem, there exists a subset $W \subseteq X , | W | > | N _ { I _ { \mathcal { G } } ^ { v } } ( W ) |$ . Given $T =$ $( W , N _ { I _ { G } ^ { v } } ( W ) )$ , if no connected component of $T$ is a tree then each connected component has more edges than vertices and $| W | \leq | N _ { I _ { G } ^ { v } } ( W ) |$ . This is not the case so exists $W ^ { \prime } \subseteq W$ such that $T ^ { \prime } = ( W ^ { \prime } , N _ { I _ { C } ^ { v } } ( \dot { W } ^ { \prime } ) )$ is a tree. Since $W ^ { \prime } \subseteq W$ ， and $v \not \in W , v \not \in W ^ { \prime }$ . By definition of $I _ { G } ^ { v }$ , for each pair $( c , c ^ { \prime } ) \in W ^ { \prime } \times V$ with $c \neq c ^ { \prime } ,$ if $\{ c , c ^ { \prime } \} \not \in N _ { I _ { G } ^ { v } } ( W ^ { \prime } )$ then, since $c \in W ^ { \prime } , ( c , c ^ { \prime } ) \notin E$

$( \Leftarrow )$ Suppose there exists a tree $T = \left( K , E _ { K } \right)$ with $K \subseteq V \setminus \{ v \}$ such that for each pair $( c , c ^ { \prime } ) \in K \times V$ with $c \neq c ^ { \prime } ,$ if {c, c<sup>′</sup>} ̸∈ E<sub>T</sub> then $( c , c ^ { \prime } ) \notin E . \mathrm { \ A s \ } T$ is acyclic $| K | = \widetilde { { E } _ { K } } | + 1$ and since $v \not \in K$ and T is connected, $\widetilde { E _ { K } } = N _ { I _ { G } ^ { v } } ( K )$ Thus, $| K | = N _ { I _ { G } ^ { v } } ( K ) + 1$ . Additionally, since $v \not \in S , K \subseteq X$ . Hence according to Hall’s marriage theorem, there is no X-perfect matching in $I _ { G } ^ { v }$ □

To illustrate this result we revisit the previous example to see why it does not admit an X-perfect matching.

Example 5. Looking back at the previous graph $G = ( V , E )$ in Figure 3a and its j-deficient incidence graph $I _ { G } ^ { j } = ( { \cal { X } } , { \cal { Y } } , Z )$ in Figure 3b, by considering each connected component of $G ,$ it becomes clear that no X-perfect matching exists because of the strongly connected component $\{ d , e , f , g \}$ which has strictly less edges than vertices since it is a tree. Note that even if the connected component $\{ h , i , j \}$ is also a tree, $j$ brings an additional possible edge to match without adding a vertex in the bipartite graph.

Using the previous result we can show that a candidate is a necessary maximin loser if and only if there exists a specific tree in the partial tournament.

Theorem 3. Given n voters and a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ a candidate $l \in \mathcal { C }$ is a necessary MM-loser for G if and only if there exists a tree $T = ( K , E )$ with $K \subseteq { \mathcal { C } } \setminus \{ l \}$ such that for each pair $( c , c ^ { \prime } ) \in K \times \mathcal { C }$ with $c \neq c ^ { \prime }$ $i f \left\{ c , c ^ { \prime } \right\} \not \in E$ then $\begin{array} { r } { \mu ( c , c ^ { \prime } ) > n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in { \mathcal C } } \mu ( c ^ { \prime \prime } , l ) } \end{array}$

Proof. Given n voters, a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m .$ and a candidate $l \in \mathcal { C }$ . If l wins all its unspecified comparisons in $G , l$ achieves a maximin score of $t = n - \operatorname* { m a x } _ { c \in \mathcal { C } } \mu ( c , l )$

$\begin{array} { r } { \mathrm { ~ I f ~ } t \geq \left\lceil \frac { n } { 2 } \right\rceil } \end{array}$ , by winning all its unspecified comparisons in $G , l$ can become a Condorcet winner and hence a maximin winner. Thus, l is not a necessary loser. And since for each $c \in \mathcal { C } , \mu ( c , l ) \leq \textstyle \left\lceil { \frac { n } { 2 } } \right\rceil$ while $t \geq \left\lceil { \frac { n } { 2 } } \right\rceil$ , hence $\mu ( c , l ) \leq$ $n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in { \mathcal C } } \mu ( c ^ { \prime \prime } , l )$ . Hence $K \subseteq { \mathcal { C } } \setminus \{ l \}$ and there exists no candidate satisfying the condition in the theorem and thus no tree.

In the rest of this proof we assume $t < \left\lceil { \frac { n } { 2 } } \right\rceil$ . Let the graph $H = ( V _ { H } , E _ { H } )$ be such that $V _ { H } = { \mathcal { C } }$ and $E _ { H } = \{ ( c , c ^ { \prime } ) : c , c ^ { \prime } \in \hat { \mathcal { C } } ^ { 2 }$ , c ̸= l, c ̸= c<sup>′</sup>, µ(c, c<sup>′</sup>) ≤ t}, and $I _ { H } ^ { l } = ( X , Y , Z )$ be the l-deficient incidence graph of $H$

If c is matched with $\{ c , c ^ { \prime } \}$ in $I _ { H } ^ { l }$ , then $G$ can be completed in $G ^ { \prime }$ where $\mu ^ { \prime } ( c , c ^ { \prime } ) \leq t$ which ensures that the maximin score of c is less than t. Aziz et al. $[ 1 ]$ showed that l is a not necessary loser if and only if there exists an X-perfect matching in $I _ { H } ^ { l }$ , we prove it here for completeness.

( =⇒ ) Suppose that $l \not \in \mathrm { N L } _ { \mathrm { M M } } ( G )$ , then there exists a completion $G ^ { \prime } =$ $( \mathcal { C } , \mu ^ { \prime } )$ of G where l is a MM-winner, i.e., where l has the maximal MM score. Thus, in $G ^ { \prime }$ , for all candidate c distinct from l, there exists $c ^ { \prime } \neq c$ such that $\begin{array} { r } { \mu ^ { \prime } ( c , c ^ { \prime } ) \leq n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in \mathcal { C } } \mu ^ { \prime } ( c ^ { \prime \prime } , l ) \leq n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in \mathcal { C } } \mu ( c ^ { \prime \prime } , l ) = t . } \end{array}$ Additionally, since $\textstyle t < { \Big \lceil } { \frac { n } { 2 } } { \Big \rceil } , 2 t < n$ and $\{ c , c ^ { \prime } \} \in Y$ cannot force both the maximin score of c and $c ^ { \prime }$ to be below t. Hence, in ${ \cal I } _ { H } ^ { l } .$ , each vertex in Y can only be matched to at most one vertex of X. This results in an X-perfect matching.

$( \Leftarrow )$ Suppose there exists an X-perfect matching in $I _ { H } ^ { l }$ , then for each candidate $c \in X$ , exists $c ^ { \prime } \in \mathcal { C }$ such that c is matched with $\{ c , c ^ { \prime } \}$ . Now, let $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ be a completion of G where l wins all its remaining comparisons, i.e., such that for all $c \in \mathcal { C } , \mu ^ { \prime } ( l , c ) = n - \mu ( c , l )$ , where $\begin{array} { r } { \mu ^ { \prime } ( c , c ^ { \prime } ) = n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in { \mathcal C } } \mu ( c ^ { \prime \prime } , l ) \mathrm { ~ i f ~ } \{ c , \{ c , c ^ { \prime } \} \} } \end{array}$ is part of the matching and where the rest of G is completed arbitrarily. Each candidate distinct from l has a maximin score of at most $n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in { \mathcal C } } \mu ( c ^ { \prime \prime } , l )$ and l has a maximin score of min<sub>c</sub>′′<sub>∈C</sub> $\begin{array} { r } { \mu ^ { \prime } ( l , c ^ { \prime \prime } ) = \operatorname* { m i n } _ { c ^ { \prime \prime } \in { \mathcal C } } ( n - \mu ( c ^ { \prime \prime } , l ) ) = n - } \end{array}$ 1 $\mathrm { n a x } _ { c ^ { \prime \prime } \in c } \mu ( c ^ { \prime \prime } , l )$ . Hence, $l \in \mathrm { M M } ( G ^ { \prime } )$ and $l \not \in \mathrm { N L } _ { \mathrm { M M } } ( G )$

Applying Lemma 1, we have that l is not a necessary MM-loser of G if and only if there exists a tree $T = ( K , E _ { T } )$ with $K \subseteq V \setminus$ {v} such that for each pair $( c , c ^ { \prime } ) \in K \times V$ with $c \neq c ^ { \prime } , \mathrm { i f } \ \{ c , c ^ { \prime } \} \ \notin E _ { T }$ then $( c , c ^ { \prime } ) \notin E _ { H }$ . Finally, by construction, such tree exists if and only if there exists a tree $T = ( K , E )$ in G with $K \subseteq { \mathcal { C } } \setminus \{ l \}$ such that for each pair $( c , c ^ { \prime } ) \in K \times \mathcal { C }$ with $c \neq c ^ { \prime } , \mathrm { i f } \ \{ c , c ^ { \prime } \} \ \notin E$ then $\mu ( c , c ^ { \prime } ) > n - \operatorname* { m a x } _ { c ^ { \prime \prime } \in { \mathcal { C } } } ( c ^ { \prime \prime } , l )$ □

![](images/b0dae79217f87e8365060841ba065af7579efa48648b8ecc7973c8819199e27c.jpg)  
(a) G

![](images/55c171dbdc10558d02c59a0ad0718961bf6aea3aac7c16eaf9187fdb6d8b472d.jpg)  
(b) H

![](images/8ccf9d0170bdb23450faba2a6b6094514b77d1d5aed7880268d78add2f3f7f1a.jpg)  
(c) $I _ { H } ^ { b }$

![](images/44bd823a505a5552122535d68d3589ba2dcf5c3eaf0da35ac7d155b32c181dfd.jpg)  
(d) $G ^ { \prime }$  
Fig. 4: A partial 5-weighted tournament $G \ ( \mathrm { a } )$ , its resulting H graph (see proof of Theorem 3 and Example 6) (b), the b-deficient incidence graph of H $I _ { H } ^ { b }$ (c), and $G ^ { \prime }$ , a completion of G where b is a MM-winner (d).

Example 6. Consider, in Figure 4, the partial 5-weighted tournament G. H is obtained from G by only keeping the edges with a weight smaller than $t \ =$ $n - \operatorname* { m a x } _ { x \in \mathcal { C } } \mu ( x , b ) = 5 - 3 = 2$ . Edges in H are edges which can potentially be used to limit the maximin score of the competitors of b. We then try to build a perfect matching in the b-deficient incidence graph of H. We show one with the thick edges. Finally, from this matching we produce $G ^ { \prime }$ , a completion of G where b is a MM-winner, hence not a necessary loser in G.

We proceed with the same idea for the weighted uncovered set.

Theorem 4. Given n voters and a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ , a candidate $l \in \mathcal { C }$ is a necessary wUC-loser for G if and only if there exists a tree $T = ( K , E )$ with $K \subseteq \mathcal { C } \setminus \{ l \}$ such that for each $\textstyle c \in K , \mu ( c , l ) \geq { \frac { n } { 2 } }$ and for each $c ^ { \prime } \in \mathcal { C } \setminus \{ l , c \} , \ i f \left\{ c , c ^ { \prime } \right\} \notin E$ then $\mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) \geq n$ . Additionally, at least one of all the previous inequalities has to be strict.

Besides the strict inequality constraint, the proof is similar to the proof of Theorem 3 with $H = ( V _ { H } , E _ { H } )$ such that $V _ { H } = { \mathcal { C } }$ and $E _ { H } = \{ ( c , c ^ { \prime } ) : c , c ^ { \prime } \in$ $\begin{array} { r } { \mathcal { C } ^ { 2 } , c \neq l , c \neq c ^ { \prime } , \mu ( c , c ^ { \prime } ) < n - \mu ( c ^ { \prime } , l ) \} \cup \{ ( c , l ) : c \in \mathcal { C } , c \neq l , \mu ( c , l ) < \left\lceil \frac { n } { 2 } \right\rceil \} } \end{array}$

## 5 Smallest Destructive Minimal Supports

In this section we continue our work with the analysis of the smallest destructive minimal supports. Proofs in this section all follow the same structure. First, we show how to build small destructive minimal supports in each sub-case and then we prove they match the lower bounds, hence that they are the smallest dMSs.

Top Cycle. To guarantee a necessary loser, we separate the top cycle and the loser. Such separation can only occurs between strongly connected components of a tournament. Since the order between the strongly connected components of a tournament is transitive and complete, it is linear. Additionally, to split m candidates in two groups of sizes k and $m - k , m ( m - k )$ comparisons are required. By concavity, the smallest frontier is either right after the top cycle (which is the top component) or right before the loser’s component.

Theorem 5. Given a complete tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \mathrm { T C } ( G )$ , for each SdMS X $f o r \ l \ \not \in \ \mathrm { T C } ( G )$ , we have $| { \mathcal { X } } | = \operatorname* { m i n } _ { t \in \{ \alpha , \beta \} } t ( m - t )$ where $\alpha = | \operatorname { T C } ( G ) |$ | and $\beta = | \{ c : c \in { \mathcal { C } } ,$ l can reach c in $G \}$ An SdMS X can be computed in polynomial time, and $\left| { \mathcal { X } } \right| \leq \left\lceil { \frac { m } { 2 } } \right\rceil \left\lfloor { \frac { m } { 2 } } \right\rfloor$

Borda. Because there always exists a candidate at the average or above, when possible, it is enough to show that the loser performs strictly below average (case i.). Else we have to present a coalition with a stronger average Borda score (case ii.). The coalition of size 1, which always exists, gives the upper bound. We suspect that the decision problem of finding an SdMS for Borda is NP-complete.

Theorem 6. Given n voters, a complete n-weighted tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , a losing candidate $l \in \mathcal { C } \setminus \mathrm { B O } ( G )$ with Borda score $\sigma _ { \mathrm { B O } } ( l )$ , for each SdMS X for $l \not \in \mathrm { B O } ( G )$ :

$$
\begin{array} { r } { i . \ i f \sigma _ { \mathrm { B O } } ( l ) \leq \left| \frac { n ( m - 1 ) - 1 } { 2 } \right| \ t h e n \ | { \mathcal X } | = \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil } \end{array}
$$

ii. else $\begin{array} { r } { \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil < | \mathcal { X } | \leq n ( m - 1 ) + 1 - \operatorname* { m a x } _ { c \in A } \mu ( c , l ) } \end{array}$ where $A = \{ c : c \in$ $\mathcal { C } , \sigma _ { \mathrm { B O } } ( c ) > \sigma _ { \mathrm { B O } } ( l ) \}$

For each SdMS $\mathcal { X } , | \mathcal { X } | \leq n ( m - 1 ) + 1$

Copeland. This result is derived from the Theorem 6 as Copeland agrees with Borda when $n = 1$ . Except for case i., a coalition of size 1 is always optimal.

Theorem 7. Given a complete tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \mathrm { C O } ( G )$ with Copeland score $\sigma _ { \mathrm { C O } } ( l )$ , for each SdMS X for $l \not \in \mathrm { C O } ( G )$ :

i. $\begin{array} { r } { i f \sigma _ { \mathrm { C O } } ( l ) < \lfloor \frac { m - 1 } { 2 } \rfloor } \end{array}$ then $\left| { \mathcal { X } } \right| = \left\lceil { \frac { m } { 2 } } \right\rceil$

ii. else if exists $c \in { \mathcal { C } }$ such that $\mu ( c , l ) = 1$ and $\sigma _ { \mathrm { C O } } ( c ) > \sigma _ { \mathrm { C O } } ( l )$ then $\scriptstyle | { \mathcal { X } } | = m - 1$ iii. else $| { \mathcal { X } } | { = } m$

An SdMS X can be computed in polynomial time, and $| { \mathcal { X } } | \leq m$

Maximin. Here, the idea is to bound the maximin score of the losing candidate by keeping comparisons where he loses against one candidate while ensuring there exists a candidate with a higher maximin score by keeping enough comparisons where he wins against each other candidate. Case ii. of Theorem 8 is when the two sets of comparisons overlap and case iii. is when they do not. An SdMS fo maximin relies on a tree (in the sense of Theorem 3) reduced to one candidate.

Theorem 8. Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { M M } ( G )$ with maximin score $\sigma _ { \mathrm { M M } } ( l )$ , for each SdMS X for $l \not \in \mathrm { M M } ( G )$

i. if $m = 2$ then $\textstyle | { \mathcal { X } } | = \left\lceil { \frac { n + 1 } { 2 } } \right\rceil$

ii. else if exists $c \in { \mathcal { C } }$ such that $\sigma _ { \mathrm { M M } } ( c ) > \sigma _ { \mathrm { M M } } ( l )$ and $\mu ( c , l ) \ge n - \sigma _ { \mathrm { M M } } ( l )$ then $\vert \mathcal { X } \vert = ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 3 ) + n + 1$

iii. else $\vert \mathcal { X } \vert = ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 2 ) + n + 1$

An SdMS X can be computed in polynomial time, and $| { \mathcal { X } } | \leq \lceil { \frac { n + 1 } { 2 } } \rceil ( m - 2 ) + n + 1$

Weighted Uncovered Set. For a candidate to be a necessary loser with the weighted uncovered set, there need to exist a candidate covering him in each possible completion. We can have one specific candidate to cover the losing candidate in all completions. Concretely, with l the losing candidate and c<sub>0</sub> the “necessary” covering candidate, we ensure that for each candidate $c , \mu ( c _ { 0 } , c ) \geq$ $n - \mu ( c , l )$ by fixing high enough $\mu ( c _ { 0 } , c )$ and $\mu ( c , l )$ (case ii. in Theorem 9, dMS X in Example 7). The tree of Theorem 4 then contains one candidate. However, sometimes a tree with two candidates yields a smaller dMS (case i., dMS Y).

Theorem 9. Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , a losing candidate $l \in \mathcal { C } \ \backslash$ wUC(G), for each SdMS X for $l \not \in \mathrm { w U C } ( G )$

i. $\begin{array} { r } { i f p = \operatorname* { m a x } _ { c _ { 0 } , c _ { 1 } } \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) - n ( m - 3 ) + \left\lfloor \frac { n } { 2 } \right\rfloor > 0 } \end{array}$ with $c _ { 0 } , c _ { 1 } \in ( \mathcal { C } \setminus \{ l \} ) ^ { 2 }$ such that c<sub>0</sub> ̸= c<sub>1</sub>, for i ∈ {0, 1}, $\textstyle \mu ( c _ { i } , l ) \geq { \frac { n } { 2 } }$ , for all $c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \}$ $\mu ( c _ { i } , c ) + \mu ( c , l ) \geq n ,$ , and at least one of the inequalities is strict,

$$
t h e n \ | \mathcal { X } | = n ( m - 2 ) + \lceil \frac { n + 1 } { 2 } \rceil - p
$$

ii. else $\begin{array} { r } { | \mathcal { X } | { = } n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil } \end{array}$

An SdMS X can be computed in polynomial time, and $\begin{array} { r } { | \mathcal { X } | \le n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil } \end{array}$

We illustrate the non-trivial case i. with the following example.

Example 7. Consider the 5-weighted tournament G in Figure 5a. In the case of X in Figure 5b, a weighted covers c across all completions. However, we can allow the covering candidate to vary dependently of the completion as in the dMS Y in Figure 5c. If a majority of voters prefers a to b then a covers c else b covers c. Here, $\begin{array} { r } { p = \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) - n ( m - 3 ) + \left\lfloor \frac { n } { 2 } \right\rfloor = 0 + 0 + 2 } \end{array}$ . Hence Y contains 2 less comparisons than $\mathcal { X } \left( \left| \mathcal { X } \right| = 8 , \left| \mathcal { V } \right| = 6 \right)$

The analogous result for the uncovered set is simpler since the first case cannot occur as $\begin{array} { r } { \sum _ { c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) \le m - 3 } \end{array}$ and $p \leq 0$

Corollary 1. Given a complete tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { U C } ( G )$ , for each SdMS X $f o r \ l \ \not \in \ \mathrm { U C } ( G )$ , we have $\scriptstyle | { \mathcal { X } } | = m - 1$

## 6 Conclusion

First, in contrast with the results of Contet et al. on constructive SMSs [10], for the weighted uncovered set, smallest destructive MSs can be computed in polynomial time while the decision problem associated to finding the smallest constructive MS is NP-complete. Additionally, we do not know if computing an SdMS for Borda is in P. Furthermore, with m the number of candidate and n the number of voters, the size of the SdMSs is in $\mathcal { O } ( m ^ { 2 } )$ for the top cycle rule and in $\mathcal { O } ( n m )$ for the remaining rules. Hence, even in the less favorable instances, only a fraction of the tournament is required. For Borda in the constructive case, almost the whole tournament was necessary when all candidates are tied.

In this paper, to find simple explanations, we put the emphasis on minimizing the size of the dMSs, i.e., the total number of pairwise comparisons. If the simplicity of the structure of dMSs is often correlated with a short number of pairwise comparisons, when looking specifically for the smallest dMSs, it can occur that we end up with a more complex structure. This suggests that depending on the specific application one might optimize for diferent metrics on dMSs. Further work is needed to investigate which explanations are preferred by users, especially empirical studies.

![](images/86498ed028a5b00f1e2b19893754bdd1709d7c6b3668eb283fb61af695de6c48.jpg)  
(a) G

![](images/d7330bce40cca5ce714b78d0142a9078d631d6984f4df63aa14b0c0d8d1aa0d1.jpg)  
(b) X

![](images/e8ab5e8b56f48d5ca2ad878bcf8b86efbedb37f4ec3f3c2ba5ff6e8985db2a5a.jpg)  
(c) Y  
Fig. 5: A 5-weighted tournament G (a), and X and Y, two dMSs for $c \not \in { \mathrm { w U C } } ( G )$ (b) and (c).

## Acknowledgments

The authors thank the reviewers of ADT26 for their constructive comments and suggestions, which helped improve this paper.

This work is funded by the European Union. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. This work is supported by ERC grant 101166894 “Advancing Digital Democratic Innovation” (ADDI).

![](images/172e88902bc52813b09a350e8eb4ef37d924ee25fe25cbdfdb4f056940a5ebce.jpg)

## References

1. Aziz, H., Brill, M., Fischer, F., Harrenstein, P., Lang, J., Seedig, H.G.: Possible and necessary winners of partial tournaments. Journal of Artificial Intelligence Research 54, 493–534 (2015)

2. Boixel, A., Endriss, U., de Haan, R.: A calculus for computing structured justifications for election outcomes. In: Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI) (2022)

3. Brandt, F., Brill, M., Harrenstein, P.: Tournament solutions. In: Brandt, F., Conitzer, V., Endriss, U., Lang, J., Procaccia, A.D. (eds.) Handbook of Computational Social Choice, pp. 57–84. Cambridge University Press (2016)

4. Brill, M., Schmidt-Kraepelin, U., Suksompong, W.: Refining tournament solutions via margin of victory. In: Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI) (2020)

5. Brockner, J., Wiesenfeld, B.M.: An integrative framework for explaining reactions to decisions: interactive efects of outcomes and procedures. Psychological bulletin 120(2), 189 (1996)

6. Cailloux, O., Endriss, U.: Arguing about voting rules. In: Proceedings of the 15th International Conference on Autonomous Agents and Multiagent Systems (AA-MAS) (2016)

7. Carman, C.: The process is the reality: Perceptions of procedural fairness and participatory democracy. Political Studies 58(4), 731–751 (2010)

8. Colquitt, J.A., Chertkof, J.M.: Explaining injustice: The interactive efect of explanation and outcome on fairness perceptions and task motivation. Journal of Management 28(5), 591–610 (2002)

9. Contet, C., Grandi, U., Mengin, J.: Abductive and contrastive explanations for scoring rules in voting. In: Proceedings of the 27th European Conference on Artificial Intelligence (ECAI) (2024)

10. Contet, C., Grandi, U., Mengin, J.: Explaining tournament solutions with minimal supports. In: Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI) (2026)

11. De Fine Licht, J., Naurin, D., Esaiasson, P., Gilljam, M.: When does transparency generate legitimacy? experimenting on a context-bound relationship. Governance 27(1), 111–134 (2014)

12. Döring, M., Peters, J.: Margin of victory for weighted tournament solutions. In: Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems (AAMAS) (2023)

13. Elkind, E., Faliszewski, P., Slinko, A.: Swap bribery. In: Proceedings of the 2nd International Symposium on Algorithmic Game Theory (SAGT) (2009)

14. Faliszewski, P., Hemaspaandra, E., Hemaspaandra, L.A., Rothe, J.: Llull and copeland voting computationally resist bribery and constructive control. Journal of Artificial Intelligence Research 35, 275–341 (2009)

15. de Fine Licht, K., de Fine Licht, J.: Artificial intelligence, transparency, and public decision-making: Why explanations are key when trying to produce perceived legitimacy. AI & society 35(4), 917–926 (2020)

16. Fischer, F., Hudry, O., Niedermeier, R.: Weighted tournament solutions. In: Brandt, F., Conitzer, V., Endriss, U., Lang, J., Procaccia, A.D. (eds.) Handbook of Computational Social Choice, pp. 85–102. Cambridge University Press (2016)

17. Grice, H.P.: Logic and conversation. In: Speech acts, pp. 41–58. Brill (1975)

18. Grimmelikhuijsen, S.: Explaining why the computer says no: Algorithmic transparency afects the perceived trustworthiness of automated decision-making. Public Administration Review 83(2), 241–262 (2023)

19. Grossi, D., Hahn, U., Mäs, M., Nitsche, A., Behrens, J., Boehmer, N., Brill, M., Endriss, U., Grandi, U., Haret, A., Heitzig, J., Janssens, N., Jonker, C.M., Keijzer, M.A., Kistner, A., Lackner, M., Lieben, A., Mikhaylovskaya, A., Murukannaiah, P.K., Proietti, C., Revel, M., Élise Rouméas, Shapiro, E., Sreedurga, G., Swierczek, B., Talmon, N., Turrini, P., Terzopoulou, Z., Putte, F.V.D.: Enabling the digital democratic revival: A research program for digital democracy (2024), https:// arxiv.org/abs/2401.16863

20. Hall, P.: On representatives of subsets. Journal of the London Mathematical Society 1(1), 26–30 (1935)

21. Hibbing, J.R.: Process preferences and american politics: What the people want government to be. American Political Science Review 95(1), 145–153 (2001)

22. Konczak, K., Lang, J.: Voting procedures with incomplete preferences. In: Proceedings of the Multidisciplinary IJCAI Workshop on Advances in Preference Handling (M-PREF) (2005)

23. Lang, J.: Collective decision making under incomplete knowledge: possible and necessary solutions. In: Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI) (2020)

24. Miller, T.: Explainable ai is dead, long live explainable ai! hypothesis-driven decision support using evaluative ai. In: Proceedings of the 2023 ACM conference on fairness, accountability, and transparency. pp. 333–342 (2023)

25. Nardi, O., Boixel, A., Endriss, U.: A graph-based algorithm for the automated justification of collective decisions. In: Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS) (2022)

26. Peters, D., Procaccia, A.D., Psomas, A., Zhou, Z.: Explainable voting. Proceedings of the 33rd Advances in Neural Information Processing Systems (NeurIPS) (2020)

27. Schwartz, B.L.: Possible winners in partially completed tournaments. SIAM Review 8(3), 302–308 (1966)

28. Suryanarayana, S.A., Sarne, D., Kraus, S.: Justifying social-choice mechanism outcome for improving participant satisfaction. In: Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS) (2022)

29. Tyler, T.R.: Governing amid diversity: The efect of fair decisionmaking procedures on the legitimacy of government. Law & Society Review 28(4), 809–831 (1994)

30. Tyler, T.R.: Social justice: Outcome and procedure. International Journal of Psychology 35(2), 117–125 (2000)

31. Xia, L., Conitzer, V.: Determining possible and necessary winners given partial orders. Journal of Artificial Intelligence Research (JAIR) 41, 25–67 (2011)

## Appendix

## A Proofs for Section 4

## A.1 Proofs for Section 4.1

Theorem 1. Given a partial tournament $G = ( { \mathcal C } , \mu )$ , a candidate $l \in \mathcal { C }$ is a necessary TC-loser for G if and only if exists $K \subseteq { \mathcal { C } } \setminus \{ l \}$ with K $\neq \emptyset$ such that for each pair $( c , c ^ { \prime } ) \in K \times ( \mathcal { C } \setminus K ) , \mu ( c , c ^ { \prime } ) = 1$

Proof. Given a partial tournament $G = ( \mathcal { C } , \mu )$ , a candidate $l \in \mathcal { C }$

$( \implies )$ We proceed by contrapose. Suppose for all $K \subseteq \mathcal { C } \setminus \{ l \}$ with $K \neq$ $\mathcal { O } ,$ exists a pair $( c , c ^ { \prime } ) \in K \times \mathcal { C } \setminus K , \mu ( c , c ^ { \prime } ) = 0 . \mathrm { ~ I f ~ } l \in \mathrm { N L } _ { \mathrm { T C } } ( G )$ , for each completion X of $G ,$ let $A _ { \mathcal { X } } = \{ c : c \in \mathcal { C } , l$ cannot reach c in X} be the set of candidates l cannot reach in $\mathcal { X } , ~ A _ { \mathcal { X } } \neq \emptyset$ . Let ${ \mathcal { X } } _ { 0 } = ( { \mathcal { C } } , \mu _ { 0 } )$ be such that $\mathcal { X } _ { 0 } = \mathrm { a r g m i n } _ { \chi \in [ G ] } | A _ { \mathcal { X } } |$ . Since $A _ { { \mathcal { X } } _ { 0 } } \subseteq { \mathcal { C } } \setminus \{ l \}$ and $A _ { \mathcal { X } _ { 0 } } \neq \emptyset$ , exists a pair $( c , c ^ { \prime } ) \in$ $A _ { { \mathcal { X } } _ { 0 } } \times { \mathcal { C } } \setminus A _ { { \mathcal { X } } _ { 0 } } , { \ddot { \mu ( c , c ^ { \prime } ) } } = 0$ . Let $\mathcal { X } _ { 1 } = ( \mathcal { C } , \mu _ { 1 } )$ be a completion of $G$ such that $\mu _ { 1 } ( c , c ^ { \prime } ) = 0 , \mu _ { 1 } ( c ^ { \prime } , c ) = 1$ and $\mu _ { 1 } ~ = ~ \mu _ { 0 }$ everywhere else. Clearly, for all $c \in$ $\mathcal { C } \setminus A _ { \mathcal { X } _ { 0 } } , c \in \mathcal { C } \setminus A _ { \mathcal { X } _ { 1 } }$ In particular, since $c ^ { \prime } \in \mathcal { C } \setminus A _ { \mathcal { X } _ { 0 } } , c ^ { \prime } \in \mathcal { C } \setminus A _ { \mathcal { X } _ { 1 } }$ , i.e. l can reach $c ^ { \prime }$ in $\mathcal { X } _ { 1 }$ . Additionally, $\mu _ { 1 } ( c ^ { \prime } , c ) = 1$ . Hence, l can reach $c , \mathrm { i . e . , } c \in \mathcal { C } \setminus A _ { \mathcal { X } _ { 1 } }$ Hence $( { \mathcal { C } } \setminus A _ { { \mathcal { X } } _ { 0 } } ) \subsetneq ( { \mathcal { C } } \setminus A _ { { \mathcal { X } } _ { 1 } } )$ or equivalently, $A _ { \mathcal { X } _ { 1 } } \subsetneq A _ { \mathcal { X } _ { 0 } }$ . Contradiction. Thus $l \not \in \mathrm { N L } _ { \mathrm { T C } } ( G )$

$( \iff )$ Suppose exists $K \subseteq \mathcal { C } \setminus \{ l \}$ with $K \neq \emptyset$ such that for each pair $( c , c ^ { \prime } ) \in K \times \mathcal { C } \setminus K , \mu ( c , c ^ { \prime } ) = 1$ then clearly for each completion X of $G , l$ cannot reach candidates in K thus $l \not \in \mathrm { T C } ( \mathcal { X } )$ and $l \in \mathrm { N L } _ { \mathrm { T C } } ( G )$ □

## A.2 Proofs for Section 4.2

Theorem 2. Schwartz [27]. Given n voters and a partial n-weighted tournament $G = ( { \mathcal { C } } , \mu )$ with $| { \mathcal { C } } | = m$ , a candidate $l \in \mathcal { C }$ is a necessary Borda loser for G if and only if exists $K \subseteq \mathcal { C } \setminus \{ l \}$ such that

$$
\operatorname* { m i n } _ { G ^ { \prime } \in [ G ] } \mathbb { E } _ { c \in K } \sigma _ { \mathrm { B O } } ( c , G ^ { \prime } ) > \operatorname* { m a x } _ { G ^ { \prime } \in [ G ] } \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) .
$$

Proof. Schwartz studied this problem from the angle of network flow theory. The result was initially stated as follows in [27].

Given a partial tournament, for any candidate $l \in \mathcal { C }$ and subset $K \subseteq { \mathcal { C } } \setminus \{ l \}$ 2 let $N ( l , K )$ be the algebraic sum of half game behind by which l trails each candidate of $K , P ( l )$ be the number of games l has still to play and $P ( K )$ be the number of games the candidates of K still have to play, omitting any in which they face one another. We have that for l to be eliminated it is necessary and suficient that exists a subset K such that

$$
N ( l , K ) - k P ( l ) - P ( K ) > 0 .
$$

We now show how to reach the new formulation from the original one. Given n voters and a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , the half game behind score of a candidate c is defined as $\begin{array} { r } { H ( c ) = n ( m - 1 ) + \sum _ { c ^ { \prime } \in \mathcal { C } } \mu ( c , c ^ { \prime } ) - } \end{array}$ $\textstyle \sum _ { c ^ { \prime } \in c } \mu ( c ^ { \prime } , c )$ . Each of the $n ( m - 1 )$ games that c will have to play grants by default one half game, if c wins the game then it grants a total of two half games (one half game plus one half game) and if c loses that game grants no half game at all (one half game minus one half game). Then we simply have $N ( l , K ) =$ $\begin{array} { r } { \sum _ { c \in K } ( H ( c ) - H ( l ) ) } \end{array}$ . The number of game c still has to play is naturally the total number of games c has to play minus those c already played, i.e., won or lost, hence $\begin{array} { r } { P ( \bar { c } ) = n ( m - 1 ) - \sum _ { c ^ { \prime } \in \mathcal { C } } ( \mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , c ) ) } \end{array}$ . It is the same idea for $P ( K )$ but we only consider games against candidates not in $K , P ( K ) =$ $\begin{array} { r } { \sum _ { c \in K } \sum _ { c ^ { \prime } \notin K } ( n - \mu ( c , c ^ { \prime } ) - \mu ( c ^ { \prime } , c ) ) } \end{array}$

$$
\begin{array} { r l } & { \quad \lambda ( 1 / \lambda ) \dot { S } - \lambda ( \mathcal { R } ^ { \lambda } ) - \mathcal { R } ^ { \lambda } \mathfrak { S } ^ { - \lambda } } \\ & { = \operatorname* { P r } ( \mathcal { R } ( \phi , - \delta \mathfrak { R } ^ { \lambda } ) - \mathcal { R } ^ { \lambda } \mathfrak { S } ^ { - \lambda } ) } \\ & { \quad - \operatorname* { P r } ( \mathcal { R } ^ { \lambda } ( \phi , - \delta \mathfrak { R } ^ { \lambda } ) - \mathcal { R } ^ { \lambda } \mathfrak { S } ^ { - \lambda } ) } \\ & { \quad - \operatorname* { P r } \sum _ { \alpha \in \mathcal { R } ^ { \lambda } } ( \mathfrak { r } - \mathcal { R } ^ { \lambda } ( \phi , - \delta \mathfrak { R } ^ { \lambda } ) - \mathcal { R } ^ { \lambda } \mathfrak { S } ^ { - \lambda } ) } \\ & { \quad - \operatorname* { P r } \sum _ { \alpha \in \mathcal { R } ^ { \lambda } } \operatorname* { P r } ( - \mathcal { R } ^ { \lambda } ( \phi , - \delta \mathfrak { S } ^ { \lambda } ) - \mathcal { R } ^ { \lambda } \mathfrak { S } ^ { - \lambda } ) } \\ & { = \operatorname* { P r } \sum _ { \alpha \in \mathcal { R } ^ { \lambda } } ( \sum _ { \alpha \in \mathcal { R } ^ { \lambda } } \operatorname* { P r } _ { \alpha \in \mathcal { R } ^ { \lambda } } ) - \sum _ { \alpha \in \mathcal { R } ^ { \lambda } } \operatorname* { P r } ( \mathcal { R } ^ { \lambda } ( \phi , - \delta \mathfrak { R } ^ { \lambda } , \mathfrak { S } ^ { \lambda } ) ) } \\ & { \quad - \operatorname* { P r } ( \mathcal { R } ^ { \lambda } ( \phi , - \delta \mathfrak { S } ^ { \lambda } ) - \sum _ { \alpha \in \mathcal { R } ^ { \lambda } } \operatorname* { P r } _ { \alpha \in \mathcal { R } ^ { \lambda } } ) } \\ &  \quad - \operatorname* { P r } ( \mathcal { R } ^ { \lambda } ( \phi , - \delta \mathfrak { S } ^ { \lambda } ) - \sum _  \alpha \in \mathcal { R } ^   \end{array}
$$

In the first double sum, observe that if $c ^ { \prime } \in K$ , then both c and c<sup>′</sup> belong to K and both $\mu ( c , c ^ { \prime } ) - \mu ( c ^ { \prime } , c )$ and $\mu ( c ^ { \prime } , c ) - \mu ( c , c ^ { \prime } )$ appear in the sum. Additionally, remember that $\mu ( c , c ) = 0$ so the case where $c = c ^ { \prime }$ can be ignored. Hence, when c<sup>′</sup> is in K, the pairs cancel in the sum and we have:

$$
\begin{array} { r l } & { \quad \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } ( g ( c , \varepsilon ^ { 2 } ) - p ( c ^ { 2 } , \varepsilon ^ { 3 } ) + 2 k ) \underset { c \in \kappa } { \sum } \mu ( c , \lambda ) - n k ( 2 n - 1 - k ) } \\ & { \quad + \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) + \mu ( c , \varepsilon ^ { 3 } ) ) > 0 } \\ & { = \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) - \mu ( c ^ { 2 } , \varepsilon ^ { 3 } ) ) > 0 } \\ & { = \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) - \mu ( c , \varepsilon ^ { 3 } ) + \frac { 2 k } { 2 } \underset { c \in \kappa } { \sum } p ( c , \lambda ) - n k ( 2 n - 1 - k ) } \\ & { \quad + \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \left( \mu ( c , \varepsilon ^ { 3 } ) + \mu ( c , \varepsilon ^ { 3 } ) \right) > 0 } \\ & { = 2 \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) + 2 k \underset { c \in \kappa } { \sum } p ( c , \lambda ) - n k ( 2 n - 1 - k ) > 0 } \\ & { \Leftrightarrow \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) + k \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) \underset { c \in \kappa } { \sum } \left( \frac { 1 } { 2 } n \right) } \\ &  \Leftrightarrow \underset { c \in \kappa } { \sum } \underset { c \in \kappa } { \sum } \mu ( c , \varepsilon ^ { 3 } ) + \underset { b \in \kappa } { \sum } \mu ( c , \varepsilon ^ \end{array}
$$

Additionally, observe that since the maximum total number of pairwise comparisons a candidate can win in any completion $G ^ { \prime } \in [ G ]$ is the total number of times it is compared less the number of pairwise comparisons it loses in $G$ we have:

$$
\operatorname* { m a x } _ { G ^ { \prime } \in [ G ] } \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) = n ( m - 1 ) - \sum _ { c \in \mathcal { C } } \mu ( c , l ) .
$$

Similarly, the minimum total number of pairwise comparisons a candidate can win in any completion $G ^ { \prime } \in [ G ]$ is the number of comparisons it wins in $G .$ Moreover, independently of the outcome, a comparison between to candidate in a coalition always represent both a win and a loss. Only wins from members of the coalition over non-members increase the average Borda score. Hence we have:

$$
\operatorname* { m i n } _ { G ^ { \prime } \in [ G ] } \mathbb { E } _ { c \in K } \sigma _ { \mathrm { B O } } ( c , G ^ { \prime } ) = n \frac { k - 1 } { 2 } + \frac { 1 } { k } \sum _ { c \in K } \sum _ { c ^ { \prime } \notin K } \mu ( c , c ^ { \prime } ) .
$$

Finally we have:

$$
N ( l , K ) - k P ( l ) - P ( K ) > 0 \iff \operatorname* { m i n } _ { G ^ { \prime } \in [ G ] } \underset { c \in K } { \mathbb { E } } \underset { \mathrm { B O } } { \mathbb { E } } ( c , G ^ { \prime } ) > \operatorname* { m a x } _ { G ^ { \prime } \in [ G ] } \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) .
$$

## A.3 Proofs for Section 4.3

The proof of Theorem 4 follows a similar approach to the one of Theorem 3. We show that a candidate is not weighted covered by any other one if and only if there exists a perfect matching in a specific bipartite graph. Note that this is not suficient as a weighted covered candidate can still be in the uncovered set as long as it is not strictly weighted covered. Typically, when two candidates weighted cover each other.

Theorem 4. Given n voters and a partial n-weighted tournament $G = ( \mathcal { C } , \mu )$ , a candidate $l \in \mathcal { C }$ is a necessary wUC-loser for G if and only if there exists a tree $T = ( K , E )$ ) with $K \subseteq { \mathcal { C } } \setminus \{ l \}$ such that for each $\textstyle c \in K , \mu ( c , l ) \geq { \frac { n } { 2 } }$ and for each $c ^ { \prime } \in \mathcal { C } \setminus \{ l , c \} , \ i f \left\{ c , c ^ { \prime } \right\} \notin E$ then $\mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) \geq n$ . Additionally, at least one of all the previous inequalities has to be strict.

Proof. We proceed in a similar fashion to the proof of Theorem 3, we reduce the problem of determining if a candidate is weighted covered by a candidate in any completion to the problem of finding a perfect matching of a specific undirected unweighted bipartite graph.

Given n voters, a partial n-weighted tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ and a candidate $l \in \mathcal { C }$

Given a completion $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ of $G , l$ is a wUC-winner if and only if it is not strictly weighted covered that is that there is no $c _ { 0 } \in { \mathcal { C } }$ such that $\mu ^ { \prime } ( c _ { 0 } , l ) \geq \mu ^ { \prime } ( l , c _ { 0 } )$ and for all $c \in { \mathcal { C } } \setminus \{ l , c _ { 0 } \} \ \mu ^ { \prime } ( c _ { 0 } , c ) \geq \mu ^ { \prime } ( l , c )$ and at least one of this inequalities is strict.

We start by identifying the relaxed case where l is (simply) weighted covered in each completion $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ of $G , { \mathrm { i . e . } }$ , where for each candidate $c \in \mathcal { C } \setminus \{ l \}$ we either have $\mu ^ { \prime } ( c , l ) < \mu ^ { \prime } ( l , c )$ or a distinct candidate $c ^ { \prime } \in \mathcal { C } \setminus \{ l , c \}$ such that $\mu ^ { \prime } ( c , c ^ { \prime } ) < \mu ^ { \prime } ( l , c ^ { \prime } )$

Let the graph $H = ( V _ { H } , E _ { H } )$ be such that $V _ { H } ~ = ~ { \mathcal { C } }$ and ${ \cal E } _ { H } = \{ ( c , c ^ { \prime } )$ $\forall ( c , c ^ { \prime } ) \in ( \mathcal { C } \setminus \{ l \} ) ^ { 2 } , c \neq c ^ { \prime } \wedge \mu ( c , c ^ { \prime } ) < n - \mu ( c ^ { \prime } , l ) \} \cup \{ ( c , l ) : \forall c \in \mathcal { C } \setminus \{ l \} , \mu ( c , l ) \ < 0 \} .$ $\left\lceil { \frac { n } { 2 } } \right\rceil \}$ and $I _ { H } ^ { l } = ( X , Y , Z )$ be the l-deficient incidence graph of H.

If c is matched with $\{ c , c ^ { \prime } \}$ in $I _ { H } ^ { l }$ then if $c ^ { \prime } = l$ then G is completed in $G ^ { \prime }$ such that $\textstyle \mu ^ { \prime } ( l , c ) \geq \lceil \frac { n + \bar { 1 } } { 2 } \rceil$ which ensures that l is strictly preferred to c else when $c ^ { \prime } \neq l , G$ is completed such that $\mu ^ { \prime } ( l , c ^ { \prime } ) > n - \mu ^ { \prime } ( c ^ { \prime } , c )$ which ensures that l is strictly more preferred to $c ^ { \prime }$ than c is preferred to $c ^ { \prime }$ . In both cases, l is not weighted covered by c.

We now show that l is not weighted covered by a candidate in any completion of G if and only if there exists an X-perfect matching of , i.e., a matching of cardinality |X| of X in $I _ { H } ^ { l }$

$( \implies )$ Suppose that l is a not weighted covered by a candidate in each completion of G, then there exists a completion $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ of G where l is not weighted covered by any other candidate. Thus, in $G ^ { \prime }$ , for all candidate c distinct from l, either $\begin{array} { r } { \mu ^ { \prime } ( l , c ) \geq \left\lceil \frac { n } { 2 } \right\rceil \mathrm { { t h u s } } \mu ^ { \prime } ( c , l ) = n - \mu ^ { \prime } ( l , c ) \leq n - \mu ( l , c ) \leq n - \left\lceil \frac { n } { 2 } \right\rceil < } \end{array}$ $\left\lceil { \frac { n } { 2 } } \right\rceil$ and c is matched with $\{ c , l \}$ in H or there exists c<sup>′</sup> ∈ C \ {l, c} such that $\bar { \mu ^ { \prime } } ( i , c ^ { \prime } ) > \mu ^ { \prime } ( c , c ^ { \prime } )$ which gives us $n - \mu ( l , c ^ { \prime } ) \geq n - \mu ^ { \prime } ( l , c ^ { \prime } ) > \mu ^ { \prime } ( c , c ^ { \prime } ) \geq \mu ( c , c ^ { \prime } )$ and c is matched with $\{ c , c ^ { \prime } \}$ in H. Suppose $\{ c , c ^ { \prime } \}$ has been matched both to c and $c ^ { \prime } ,$ then we have that $\mu ^ { \prime } ( l , c ^ { \prime } ) > \mu ^ { \prime } ( c , c ^ { \prime } )$ and $\mu ^ { \prime } ( l , c ) > \mu ^ { \prime } ( c ^ { \prime } , c )$ . Hence, $\mu ^ { \prime } ( l , c ^ { \prime } ) + \mu ^ { \prime } ( l , c ) > \mu ^ { \prime } ( c , c ^ { \prime } ) + \mu ^ { \prime } ( c ^ { \prime } , c ) = n$ and $\mu ^ { \prime } ( l , c ) \geq \lceil \frac { n } { 2 } \rceil$ or $\mu ^ { \prime } ( l , c ^ { \prime } ) \geq \lceil \frac { n } { 2 } \rceil$ Without loss of generality, suppose $\mu ^ { \prime } ( l , c ) \geq \lceil \frac { n } { 2 } \rceil$ . Then $\mu ( c , l ) \leq \mu ^ { \prime } ( c , l ) =$ $\begin{array} { r } { n - \mu { ' } ( l , c ) \leq n - \left\lceil \frac { n } { 2 } \right\rceil < \left\lceil \frac { n } { 2 } \right\rceil } \end{array}$ . Thus $\{ c , \{ c , l \} \} \in E _ { H }$ which means that we can unmatch c with $\{ c , c ^ { \prime } \}$ and match it with its corresponding edge shared with l. This result in a X-perfect matching in $I _ { H } ^ { l }$

$( \iff )$ Suppose there exists a X-perfect matching in ${ \cal I } _ { H } ^ { l } .$ , then for each candidate $c \in X$ , exists $c ^ { \prime } \in { \mathcal { C } }$ such that c is matched with $\{ c , c ^ { \prime } \}$ . Now, let $G ^ { \prime } ~ = ~ ( { \mathcal { C } } , \mu ^ { \prime } )$ be a completion of G where l wins all its remaining comparisons, $\mathrm { i . e . }$ , such that for all $c \in \mathcal { C } , \mu ^ { \prime } ( l , c ) = n - \mu ( c , l ) \geq \mu ( l , c )$ , and where $\mu ^ { \prime } ( c , c ^ { \prime } ) = \mu ( c , c ^ { \prime } ) { \mathrm { ~ i f ~ } } \{ c , \{ c , c ^ { \prime } \} \}$ is part of the matching and where the rest of G is completed arbitrarily. For each candidate c distinct from l, if $\{ c , \{ c , l \} \}$ is part of the matching then $\{ c , \{ c , l \} \} \in E _ { H }$ and $\mu ( c , l ) < \lceil \frac { n } { 2 } \rceil$ else else exists $c ^ { \prime } \neq l$ such that $\{ c , \{ c , c ^ { \prime } \} \}$ is part of the matching then $\{ c , \bar { \{ c , c ^ { \prime } \} } \} \in E _ { H }$ and $\mu ^ { \prime } ( c , c ^ { \prime } ) = \mu ( c , c ^ { \prime } ) < \mu ( l , c ^ { \prime } ) \leq \mu ^ { \prime } ( l , c )$ , i.e., in both case l is not weighted covered by c. Hence l is not weighted covered by any other candidate in $G ^ { \prime }$ which is a completion of G.

Applying Lemma 1, we have that l is a not weighted covered by a candidate in each completion G if and only if H has an acyclic connected component which does not contain l. Finally, by construction, H has an acyclic connected component which does not contain l if and only if exists a tree $T = ( K , E )$ with $K \subseteq \mathcal { C } \setminus \{ l \}$ such that for each $c \in K , \mu ( c , l ) \geq \lceil \frac { n } { 2 } \rceil$ and for each pair $( c , c ^ { \prime } ) \in K \times \mathcal { C } \setminus \{ l \}$ with $c \neq c ^ { \prime }$ , if $\{ c , c ^ { \prime } \} \not \in E$ then $\mu ( c , c ^ { \prime } ) ^ { - } + \mu ( c ^ { \prime } , l ) \geq n$

Recall that $l \in \mathrm { N L } _ { \mathrm { w U C } } ( G )$ when l is strictly weighted covered in each completion of G. Since we know when l is (simply) weighted covered in each completion of $G ,$ , we now need to identify the sub-cases in which it is always strictly weighted covered.

We show that it is necessary and suficient to simply add the constraint that at least one of inequalities in the tree of the previous characterization has to be strict.

Suppose that no inequality is strict. Then for each $\textstyle c \in K , \mu ( l , c ) = { \frac { n } { 2 } }$ . Let $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ be a completion of G where l wins all its remaining comparisons, i.e., such that for all $c \in \mathcal { C } , \mu ^ { \prime } ( l , c ) = n - \mu ( c , l ) \geq \mu ( l , c )$ , and where

If we require at least one equality to be strict for each member of the tree, it is clear that in each completion, a member of the tree that weighted covers l also strictly weighted covers l.

Given a tree $T = ( K , E )$ with $K \subseteq { \mathcal { C } } \backslash \{ l \}$ such that for each $\textstyle c \in K , \mu ( c , l ) \geq { \frac { n } { 2 } }$ and for each $c ^ { \prime } \in \mathcal { C } \setminus \{ l , c \}$ , if $\{ c , c ^ { \prime } \} \not \in E$ then $\mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) \geq n .$ suppose exists $c _ { 0 } \in K$ such that $\textstyle \mu ( c , l ) = { \frac { n } { 2 } }$ and for each $c ^ { \prime } \in \mathcal { C } \setminus \{ l , c \} , \mathrm { i f } \left\{ c , c ^ { \prime } \right\} \notin E$ then $\mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) = n$

Let $G ^ { \prime } = ( \mathcal { C } , \mu ^ { \prime } )$ be a completion of G where l wins all its remaining comparisons, i.e., such that for all $\begin{array} { r } { c \in \mathcal { C } , \mu ^ { \prime } ( l , c ) = \mu ( l , c ) = \frac { n } { 2 } } \end{array}$ , and where for each pair $( c , c ^ { \prime } ) \in K \times \mathcal { C } \setminus \{ l \}$ with $c \neq c ^ { \prime }$ , if $\{ c , c ^ { \prime } \} \in E$ then $\begin{array} { r } { \bar { \mu } ^ { \prime } ( c , c ^ { \prime } ) = \mu ( c , c ^ { \prime } ) = \frac { n } { 2 } } \end{array}$ else $\begin{array} { r } { \mu ( c , c ^ { \prime } ) = n - \mu ( c ^ { \prime } , l ) = \frac { n } { 2 } } \end{array}$ . It is clear that l is not strictly weighted covered by any candidate in K.

Now suppose that one of the inequalities in $T _ { 0 } = ( K _ { 0 } , E _ { 0 } )$ is strict. Let us assume that it is associated to candidate $c _ { 0 } \in K$ . If $T _ { 0 }$ only contains one vertex, then $c _ { 0 }$ strictly weighted covers l. Else for l not to be weighted covered by $c _ { 0 } .$

G has to be extended in another partial tournament $G _ { 1 } = ( \mathcal { C } , \mu _ { 1 } )$ where exists $c _ { 1 } \in K _ { 0 } , c _ { 1 } \neq c _ { 0 }$ such that $\{ c _ { 0 } , c _ { 1 } \} \in E _ { 0 }$ where $\mu _ { 1 } ( c _ { 0 } , c _ { 1 } ) < \mu _ { 1 } ( l , c _ { 1 } )$ . Thus $\mu _ { 1 } ( c _ { 1 } , c _ { 0 } ) \geq n - \mu _ { 1 } ( c _ { 0 } , c _ { 1 } ) > n - \mu _ { 1 } ( l , c _ { 1 } )$ and we have a new strict inequality $\mu _ { 1 } ( c _ { 1 } , c _ { 0 } ) + \mu _ { 1 } ( l , c _ { 1 } ) > n$ . This means that exists $T _ { 1 } = ( K _ { 1 } , E _ { 1 } )$ a sub-tree of $T _ { 0 }$ with $K _ { 1 } \subseteq K _ { 0 } \setminus \{ c _ { 0 } \}$ and $E _ { 1 } \subsetneq E _ { 0 }$ such that for each $\textstyle c \in K _ { 1 } , \mu _ { 1 } ( c , l ) \geq { \frac { n } { 2 } }$ and for each $c ^ { \prime } \in \mathcal { C } \setminus \{ l , c _ { 0 } , c \}$ , if $\{ c , c ^ { \prime } \} \not \in E _ { 1 }$ then $\mu ( c , c ^ { \prime } ) + \mu ( c ^ { \prime } , l ) \geq n$ where exists a strict inequality. This way we build a strictly decreasing sequence of trees with at least one strict inequality. Since we are working with a finite number of candidate, we end up with a tree restricted to one vertex which strictly weighted covers l in each completion. □

## B Proofs for Section 5

Theorem 5. Given a complete tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \mathrm { T C } ( G )$ , for each SdMS X $f o r \ l \ \not \in \ \mathrm { T C } ( G )$ , we have $\vert { \mathcal { X } } \vert { = \operatorname* { m i n } _ { t \in \{ \alpha , \beta \} } t ( m { - } t ) }$ where $\alpha = | \operatorname { T C } ( G ) |$ | and $\beta = | \{ c : c \in \mathcal { C } , i$ l can reach c in $G \}$

An SdMS X can be computed in polynomial time, and $\left| { \mathcal { X } } \right| \leq \left\lceil { \frac { m } { 2 } } \right\rceil \left\lfloor { \frac { m } { 2 } } \right\rfloor$

Proof. Given a directed graph $G = ( { \mathcal C } , \mu )$ . A subset of vertices $K \in C$ is a strongly connected component if: (i) for each pair $( x , y ) \in K ^ { 2 }$ with $x \neq y .$ , there exist a path from x to y and (ii) K is maximal, in the sense that no vertex can be added without violating condition (i).

Given a complete tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \mathrm { T C } ( G )$ , Let ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ be a dMS for $l \not \in \mathrm { T C } ( G )$

Let ∼ be the equivalence relation on C defined by $x \sim y$ if and only if there are directed paths from x to y and y to x. The equivalence classes of ∼ are the strongly connected components of G. We define the condensation graph $\widetilde { G } = ( \widetilde { c } , \widetilde { \mu } ) , \mathrm { i . e . }$ ., the quotient graph of G by ∼ with $\widetilde { \mathcal { C } } = \mathcal { C } / \sim$ and for $( x , y ) \in \widetilde { \mathcal { C } } ^ { 2 }$ with x $\mathbf {  { \neq } } y ,  { \widetilde { \mu } } ( x , y ) = 1$ if and only if exist $( u , v ) \in x \times y$ such that $\mu ( u , v ) = 1 . \widetilde { G }$ is naturally a directed acyclic graph. However, since G is complete, Ge is complete. Hence ≻, the order induced by $\widetilde { G }$ on the strongly connected components of G is total.

Let p be the number of strongly connected components of G. Since $l \not \in \mathrm { T C } ( G )$ ， l cannot reach all candidates in G and $p \geq 2$ . Let $\{ A _ { 1 } , A _ { 2 } , \ldots , A _ { p } \}$ be the set of strongly connected components of G such that $A _ { 1 } \succ A _ { 2 } \succ \dotsb \succ A _ { p }$ . Let $2 \leq i _ { l } \leq p$ be such that $l \in A _ { i _ { l } }$

According to Theorem 1, exists $K \subseteq \mathcal { C } \setminus \{ l \}$ with $K \neq \emptyset$ such that for each pair $( c , c ^ { \prime } ) \in K \times \mathcal { C } \setminus K , \mu ( c , c ^ { \prime } ) = 1$ . But for each strongly connected component A, either $A \subseteq K$ or $A \cap K = \emptyset$ otherwise some candidates in A could not reach other candidates in A. And for $1 \leq i \leq p _ { ; }$ , if $A _ { i } \subseteq K$ then for all $1 \leq j \leq i$ $A _ { j } \subseteq K$ else there would exist $( A , A ^ { \prime } ) \in \{ A _ { 1 } , A _ { 2 } , . . . , A _ { p } \} ^ { 2 }$ with $A \succ A ^ { \prime }$ such that exists $( x , y ) \in A \times A ^ { \prime }$ such that $\mu ( y , x ) = 1$ . Hence, the only possibilities for K are $\textstyle \bigcup _ { j = 1 } ^ { i } A _ { j }$ with $1 \leq i \leq i _ { l } - 1$

Additionally, to split G between K and its complementary $| K | ( | { \mathcal { C } } | - | K | )$ pairwise comparisons are needed. Let the function $f : x \mapsto x ( m - x )$ be defined on $[ | A _ { 1 } | , | \cup _ { j = 1 } ^ { i _ { l } - 1 } A _ { j } | ]$ . f is clearly concave and thus reaches its minimum either in $x = | A _ { 1 } |$ or $x = | \cup _ { j = 1 } ^ { i _ { l } - 1 } A _ { j } |$ . Since $A _ { 1 } = \mathrm { T C } ( G ) , \cup _ { j = i _ { l } } ^ { p } A _ { j } = m - \bigcup _ { j = 1 } ^ { i _ { l } - 1 } A _ { j }$ and $\bigcup _ { j = i _ { l } } ^ { p } A _ { j } = \{ c : c \in { \mathcal { C } } , l$ can reach c in $G \} , \ | \mathcal { X } | \overset { \cdot } { = } \operatorname* { m i n } _ { t \in \{ \alpha , \beta \} } t ( m - t )$ where $\alpha = | \operatorname { T C } ( G ) |$ and $\beta = | \{ c : c \in \mathcal { C } , l$ l can reach c in $G \}$

$\begin{array} { r } { \operatorname { I f } | A _ { 1 } | ( m - | A _ { 1 } | ) \leq | \bigcup _ { j = 1 } ^ { i _ { l } - 1 } A _ { j } | ( m - | \bigcup _ { j = 1 } ^ { i _ { l } - 1 } A _ { j } | ) } \end{array}$ then let ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ be partial sub-tournament of $G$ such that for all $( \overset { \vartriangle } { c } , \overset { \vartriangle } { c ^ { \prime } } ) \in A _ { 1 } \times ( \mathcal { C } \setminus A _ { 1 } ) , \mu _ { \mathcal { X } } ( \boldsymbol { c } , \boldsymbol { c } ^ { \prime } ) = 1$ and $\mu _ { \mathcal { X } }$ is null everywhere else. According to Theorem $1 , l \in \operatorname { N L } _ { \mathrm { T C } } ( G )$ . Since $| { \mathcal { X } } | =$ $| A _ { 1 } | ( m - | A _ { 1 } | )$ matches the lower bound, X is a smallest dMS for $l \not \in \mathrm { T C } ( G )$ The same reasoning holds when $\begin{array} { r } { | A _ { 1 } | ( m - | A _ { 1 } | ) \geq | \bigcup _ { i = 1 } ^ { i _ { l } - 1 } A _ { j } | ( m - | \bigcup _ { i = 1 } ^ { i _ { l } - 1 } A _ { j } | ) } \end{array}$

It naturally follows from the beginning of the proof, that an SdMS can be computed in polynomial time. Additionally, by concavity and symmetry, the worst case is reached when candidates are split in two equal parts, i.e., when $\begin{array} { r } { | T C ( G ) | = \left\lceil \frac { m } { 2 } \right\rceil \mathrm { o r } | T C ( G ) | = \left\lfloor \frac { m } { 2 } \right\rfloor } \end{array}$ and the losing candidate is in $A _ { 2 }$ . In that case, $\left| { \mathcal { X } } \right| = \left\lceil { \frac { m } { 2 } } \right\rceil \left\lfloor { \frac { m } { 2 } } \right\rfloor$ □

Theorem 6. Given n voters, a complete n-weighted tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , a losing candidate $l \in \mathcal { C } \setminus \mathrm { B O } ( G )$ with Borda score $\sigma _ { \mathrm { B O } } ( l )$ , for each SdMS X for $l \not \in \mathrm { B O } ( G )$

i. if σ<sub>BO</sub>(l) ≤ n(m−1)−1 then |X |= n(m−1)+1   
ii. else $\begin{array} { r } { \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil < | \mathcal { X } | \leq n ( m - 1 ) + 1 - \operatorname* { m a x } _ { c \in A } \mu ( c , l ) } \end{array}$ where $A = \{ c : c \in$ $\mathcal { C } , \sigma _ { \mathrm { B O } } ( c ) > \sigma _ { \mathrm { B O } } ( l ) \}$

For each SdMS $\mathcal { X } , | \mathcal { X } | \leq n ( m - 1 ) + 1$

Proof. We start by proving that $\left\lceil { \frac { n ( m - 1 ) + 1 } { 2 } } \right\rceil$ comparisons are enough when l has below average performances, i.e., $\begin{array} { r } { \sigma _ { \mathrm { B O } } ( l ) \leq \left| \frac { n ( m - 1 ) - 1 } { 2 } \right| } \end{array}$

Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ a losing candidate $l \in \mathcal { C } \setminus \mathrm { B O } ( G )$ . Suppose that $\sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) ~ \leq ~ \left\lfloor \frac { n ( m - 1 ) - 1 } { 2 } \right\rfloor$ then $\begin{array} { r } { \sum _ { c \neq l } \mu _ { G } ( c , l ) = n ( m - 1 ) - \sigma _ { \mathrm { B O } } ( l , G ) \geq \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil } \end{array}$ . Let ${ \mathcal { X } } \subseteq G$ such that $\begin{array} { r } { \sum _ { c \neq l } \mu _ { \mathcal { X } } ( c , l ) \ = \ \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil } \end{array}$ and $\mu _ { \mathcal { X } }$ is null otherwise. Clearly, $| { \mathcal { X } } | =$ $\left\lceil { \frac { n ( m - 1 ) + 1 } { 2 } } \right\rceil$ . Then for all complete tournaments $G ^ { \prime }$ completing X we have $\begin{array} { r } { n ( m - 1 ) - \sum _ { c \neq l } \mu _ { G ^ { \prime } } ( c , l ) \leq n ( m - 1 ) - \sum _ { c \neq l } \mu _ { \mathcal { X } } ( c , l ) = n ( m - 1 ) - \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil = \frac { \sum _ { c \neq l } \mu _ { \mathcal { X } } ( c , l ) } { 2 } . } \end{array}$ $\left| { \frac { n ( m - 1 ) - 1 } { 2 } } \right|$ . Moreover, $\begin{array} { r } { \sum _ { c \neq l } \sigma _ { \mathrm { B O } } ( c , G ^ { \prime } ) = n \frac { m ( m - 1 ) } { 2 } - \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) = n \frac { m ( m - 1 ) } { 2 } - } \end{array}$ 1 $\begin{array} { r } { \left| { \frac { n ( m - 1 ) - 1 } { 2 } } \right| = n { \frac { ( m - 1 ) ( m - 1 ) } { 2 } } + \left\lceil { \frac { 1 } { 2 } } \right\rceil = n { \frac { ( m - 1 ) ( m - 1 ) } { 2 } } + 1 } \end{array}$ and by the pigeonhole principle, among the $m - 1$ candidates distinct from l, there exists $c _ { 0 }$ such that $\begin{array} { r } { \sigma _ { \mathrm { B O } } ( c _ { 0 } , G ^ { \prime } ) \geq \left\lceil \frac { \sum _ { c \neq l } \sigma _ { \mathrm { B O } } ( c , G ^ { \prime } ) } { m - 1 } \right\rceil = \left\lceil \frac { n ( m - 1 ) } { 2 } + \frac { 1 } { m - 1 } \right\rceil } \end{array}$ . Hence, $\sigma _ { \mathrm { B O } } ( c _ { 0 } , G ^ { \prime } ) >$ $\sigma _ { \mathrm { B O } } ( l , G ^ { \prime } )$ and l is a necessary loser for X and X contains a dMS.

$$
\sigma _ { \mathrm { B O } } ( l , G ^ { \prime } ) =
$$

Let us now prove that $n ( m - 1 ) + 1 - \operatorname* { m a x } _ { c \in A } \mu ( c , l )$ comparisons are always enough.

Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ a losing candidate $l \in { \mathcal { C } } \backslash \mathrm { B O } ( G )$ . Let $A = \{ c : c \in \mathcal { C } , \sigma _ { \mathrm { B O } } ( c , G ) > \sigma _ { \mathrm { B O } } ( l , G ) \}$ and $c _ { 0 } = \operatorname { a r g m a x } _ { c \in A } \mu _ { G } ( c , l )$ . Then, $\begin{array} { r } { \sum _ { c \neq c _ { 0 } } \mu _ { G } ( c _ { 0 } , c ) + \sum _ { c \neq l } \mu _ { G } ( c , l ) = \sigma _ { \mathrm { B O } } ( c _ { 0 } , G ) + } \end{array}$ $\begin{array} { r } { n ( m - 1 ) - \sigma _ { \mathrm { B O } } ( l , G ) \geq \sigma _ { \mathrm { B O } } ( l , G ) + \check { 1 } + n ( m - 1 ) - \check { \sigma } _ { \mathrm { B O } } ( l , G ) = n ( m - 1 ) + 1 } \end{array}$ Let $\begin{array} { r } { \mathcal { X } \subseteq G \mathrm { ~ b e ~ s u c h ~ t h a t ~ } \mu _ { \mathcal { X } } ( c _ { 0 } , l ) = \mu _ { G } ( c _ { 0 } , l ) , \sum _ { c \neq l } \mu _ { \mathcal { X } } ( c _ { 0 } , c ) + \sum _ { c \neq l } \mu _ { \mathcal { X } } ( c , l ) = } \end{array}$ $n ( m - 1 ) + 1$ and $\mu _ { \mathcal { X } }$ is null otherwise. $| \mathcal { X } | = n ( m - 1 ) + 1 - ^ { ' } \mu _ { \mathcal { X } } ( c _ { 0 } , l ) =$ $n ( m - 1 ) + 1 - \operatorname* { m a x } _ { c \in A } \mu ( c , l )$ . Then for all complete tournaments $G ^ { \prime }$ completing X we have $\begin{array} { r } { \sum _ { c \neq l } \mu _ { G ^ { \prime } } ( c _ { 0 } , c ) + \sum _ { c \neq l } \mu _ { G ^ { \prime } } ( c , l ) \geq n ( m - 1 ) + 1 \implies \sigma _ { \mathrm { B O } } ( c _ { 0 } , G ^ { \prime } ) + } \end{array}$ $n ( m - 1 ) - \sigma _ { \mathrm { B O } } ^ { \prime } ( l , G ^ { \prime } ) \geq n ( m - 1 ) + 1 \implies \sigma _ { \mathrm { B O } } ( c _ { 0 } , G ^ { \prime } ) > \sigma _ { \mathrm { B O } } ( l , G ^ { \prime } )$ and l is a necessary loser for X and X contains a dMS.

We now move on the lower bounds.

Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ a losing candidate $l \in \mathcal { C } \setminus \mathrm { B O } ( G )$ , let ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ be a dMS for $l \not \in \mathrm { B O } ( G )$ According to Theorem 2, exists $K \subseteq { \mathcal { C } } \setminus \{ l \}$ with $| K | = k$ such that

$$
\sum _ { c \in K } \sum _ { c ^ { \prime } \notin K } \mu ( c , c ^ { \prime } ) + k \sum _ { c \in \mathcal { C } } \mu ( c , l ) > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } .
$$

We have that $| \mathcal { X } | \geq \sum _ { c \in K } \sum _ { c ^ { \prime } \notin K } \mu ( c , c ^ { \prime } ) \mathrm { ~ a n d ~ } | \mathcal { X } | \geq \sum _ { c \in \mathcal { C } } \mu ( c , l )$ . Hence,

$$
\begin{array} { r l } & { \qquad | \mathcal { X } | + k | \mathcal { X } | > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } } \\ & { \Longleftrightarrow ( k + 1 ) | \mathcal { X } | > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } } \\ & { \Longleftrightarrow | \mathcal { X } | > \frac { 1 } { k + 1 } \left( k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } \right) } \\ & { \Longleftrightarrow | \mathcal { X } | > \left( 1 - \frac { 1 } { k + 1 } \right) \left( n ( m - 1 ) - n \frac { k - 1 } { 2 } \right) } \\ & { \Longleftrightarrow | \mathcal { X } | > \left( 1 - \frac { 1 } { k + 1 } \right) \left( n m - n \frac { k + 1 } { 2 } \right) } \end{array}
$$

Let f be the function $\begin{array} { r } { x \mapsto \left( 1 - \frac { 1 } { x + 1 } \right) \left( n m - n \frac { x + 1 } { 2 } \right) } \end{array}$ defined on $[ 1 ; m - 1 ] $ R. f is twice diferentiable on its domain. $f ^ { \prime } : x \mapsto { \frac { 1 } { ( x + 1 ) ^ { 2 } } } \left( n m - n { \frac { x + 1 } { 2 } } \right) -$ $\left( 1 - { \frac { 1 } { x + 1 } } \right) { \frac { n } { 2 } }$ and $\begin{array} { r } { f ^ { \prime \prime } : x \mapsto { \frac { - 2 } { ( x + 1 ) ^ { 3 } } } \left( n m - n { \frac { x + 1 } { 2 } } \right) - { \frac { 1 } { ( x + 1 ) ^ { 2 } } } { \frac { n } { 2 } } - { \frac { 1 } { ( x + 1 ) ^ { 2 } } } { \frac { n } { 2 } } = { \frac { - 2 n m } { ( x + 1 ) ^ { 3 } } } . } \end{array}$ ${ \dot { f } } ^ { \prime \prime }$ is clearly negative on the interval $[ 1 ; m - 1 ]$ , thus f is concave and minimal at $x = 1 { \mathrm { ~ o r ~ } } x = m - 1$

$$
\begin{array} { r } { f ( 1 ) = \left( 1 - \frac { 1 } { 1 + 1 } \right) \left( n m - n \frac { 1 + 1 } { 2 } \right) = \frac { n ( m - 1 ) } { 2 } . } \end{array}
$$

$$
\begin{array} { r } { f ( m - 1 ) = \left( 1 - \frac { 1 } { m - 1 + 1 } \right) \left( n m - n \frac { m - 1 + 1 } { 2 } \right) = \frac { n ( m - 1 ) } { 2 } . } \end{array}
$$

Hence, $\textstyle | { \mathcal { X } } | > { \frac { n ( m - 1 ) } { 2 } }$ . Since $| \mathcal { X } |$ is an integer we have $\begin{array} { r } { | \mathcal { X } | \ge \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil } \end{array}$

Suppose now that $\sigma _ { \mathrm { B O } } ( l ) ~ > ~ \left| ~ \frac { n ( m - 1 ) - 1 } { 2 } ~ \right|$ . We have that $\begin{array} { r } { \sum _ { c \in \mathcal { C } } \mu _ { \mathcal { X } } ( c , l ) \ \leq } \end{array}$ $\begin{array} { r } { \sum _ { c \in \mathcal { C } } \mu ( c , l ) = n ( m - 1 ) - \sigma _ { \mathrm { B O } } ( l ) < n ( m - 1 ) - \left\lfloor \frac { n ( m - 1 ) - 1 } { 2 } \right\rfloor = \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil . } \end{array}$

Thus, $\begin{array} { r } { \sum _ { c \in \mathcal { C } } \mu _ { \mathcal { X } } ( c , l ) \leq \left\lceil \frac { n ( m - 1 ) - 1 } { 2 } \right\rceil } \end{array}$ , it is not possible to take $K = \mathcal { C } \setminus \{ l \}$ anymore and $\begin{array} { r } { | \mathcal { X } | > \left\lceil \frac { n ( m - 1 ) + 1 } { 2 } \right\rceil } \end{array}$

If we look at

$$
\sum _ { c \in K } \sum _ { c ^ { \prime } \notin K } \mu ( c , c ^ { \prime } ) + k \sum _ { c \in \mathcal { C } } \mu ( c , l ) > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 }
$$

it is clear that:

– for $c \in K \ \mu ( c , l )$ contributes $k + 1$ times in the left hand

– for $c , c ^ { \prime } \in K \times ( \mathcal { C } \setminus \{ l \} ) \mu ( c , c ^ { \prime } )$ contributes 1

– for $c \notin K \ \mu ( c , l ) \ \mathrm { c o n t r i b u t e s } \ k$

– for $c , c ^ { \prime } \in ( \mathcal { C } \setminus K ) \times ( \mathcal { C } \setminus \{ l \} ) \ \mu ( c , c ^ { \prime } )$ contributes 0.

Hence, maximizing $\textstyle \sum _ { c \in c } \mu ( c , l )$ is never damaging to obtain smallest dMSs and we can take it to its maximal value, $n ( m - 1 ) - \sigma _ { \mathrm { B O } } ( l , G )$

We have:

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { c \in K } \displaystyle \sum _ { c ^ { \prime } \in K } \mu ( c , c ^ { \prime } ) + k \sum _ { c \in \mathcal { C } } \mu ( c , l ) > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } } \\ & { \Longrightarrow \displaystyle \sum _ { c \in K } \displaystyle \sum _ { c ^ { \prime } \in \mathcal { C } \backslash ( K \cup \{ l \} ) } \mu ( c , c ^ { \prime } ) + \sum _ { c \in K } \mu ( c , l ) + k \sum _ { c \in \mathcal { C } } \mu ( c , l ) > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } } \\ & { \Longrightarrow \displaystyle \sum _ { c \in K ^ { \prime } \in \mathcal { C } \backslash ( K \cup \{ l \} ) } \mu ( c , c ^ { \prime } ) + \sum _ { c \in K } \mu ( c , l ) + k ( n ( m - 1 ) - \sigma _ { \mathrm { B O } } ( l , G ) ) > k n ( m - 1 ) - n \frac { k ( k - 1 ) } { 2 } } \\ & { \Longrightarrow \displaystyle \sum _ { c \in K } \sum _ { c ^ { \prime } \in \mathcal { C } \backslash ( K \cup \{ l \} ) } \mu ( c , c ^ { \prime } ) > k \sigma _ { \mathrm { B O } } ( l , G ) - n \frac { k ( k - 1 ) } { 2 } - \sum _ { c \in K } \mu ( c , l ) . } \end{array}
$$

With a MILP solver, we can fin $K ^ { * }$ and $\mu ^ { * }$ which minimize $\begin{array} { r } { \sum _ { c \in K } \sum _ { c ^ { \prime } \in { \mathcal C } \setminus ( K \cup \{ l \} ) } \mu ( c , c ^ { \prime } ) } \end{array}$ while satisfying this constraint.

Finally, we can take ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ with $\forall c \in K , c ^ { \prime } \in \mathcal { C } \setminus ( K \cup \{ l \} ) , \mu _ { \mathcal { X } } ( c , c ^ { \prime } ) =$ $\mu ^ { * } ( c , c ^ { \prime } ) , \mu _ { \mathcal { X } } ( c , l ) = \mu ( c , l )$ and $\mu _ { \mathcal { X } }$ null elsewhere. Hence, $\begin{array} { r } { | \mathcal { X } | = \sum _ { c \in K ^ { * } } \sum _ { c ^ { \prime } \in \mathcal { C } \backslash ( K ^ { * } \cup \{ l \} ) } \mu ^ { * } ( c , c ^ { \prime } ) + } \\ { \bigcup \qquad \bigcup } \end{array}$ $n ( m - 1 ) - \sigma _ { \mathrm { B O } } ( l , G )$

Theorem 7. Given a complete tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \mathrm { C O } ( G )$ with Copeland score $\sigma _ { \mathrm { C O } } ( l )$ , for each SdMS X $f o r l \not \in \mathrm { C O } ( G )$ :

$i . \ i f \ \sigma _ { \mathrm { C O } } ( l ) < \lfloor \frac { m - 1 } { 2 } \rfloor$ then $\left| { \mathcal { X } } \right| = \left\lceil { \frac { m } { 2 } } \right\rceil$   
ii. else if exists $c \in { \mathcal { C } }$ such that $\mu ( c , l ) = 1$ and $\sigma _ { \mathrm { C O } } ( c ) > \sigma _ { \mathrm { C O } } ( l )$ then $\scriptstyle | { \mathcal { X } } | = m - 1$   
iii. else $| { \mathcal { X } } | { = } m$

An SdMS X can be computed in polynomial time, and $| { \mathcal { X } } | \leq m$

Theorem 8. Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { M M } ( G )$ with maximin score $\sigma _ { \mathrm { M M } } ( l )$ , for each SdMS X for $l \not \in \mathrm { M M } ( G )$

i. if $m = 2$ then $\textstyle | { \mathcal { X } } | = \left\lceil { \frac { n + 1 } { 2 } } \right\rceil$   
ii. else $i f$ exists $c \in { \mathcal { C } }$ such that $\sigma _ { \mathrm { M M } } ( c ) > \sigma _ { \mathrm { M M } } ( l )$ and $\mu ( c , l ) \ge n - \sigma _ { \mathrm { M M } } ( l )$ ， then $\vert \mathcal { X } \vert = ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 3 ) + n + 1$   
iii. else $\vert \mathcal { X } \vert = ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 2 ) + n + 1$

An SdMS X can be computed in polynomial time, and $| { \mathcal { X } } | \leq \lceil { \frac { n + 1 } { 2 } } \rceil ( m - 2 ) + n + 1$

Proof. In the case $m = 2 ,$ ensuring that a strict majority of voters prefers the other candidate to l is clearly necessary and suficient so we move to the case where $m \geq 3$

We start by proving that $( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 3 ) + n + 1$ and $( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m -$ $2 ) + n + 1$ are enough depending on the cases.

Given n voters, a complete n-weighted tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | =$ $m ,$ a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { M M } ( G )$ . Since $l \not \in { \mathrm { M M } } ( G )$ , exists $c _ { 0 } \in \mathcal { C } \setminus \{ l \}$ such that $\sigma _ { \mathrm { M M } } ( c _ { 0 } ) > \sigma _ { \mathrm { M M } } ( l , G )$ . Additionally, let $c _ { 1 } = \mathrm { a r g m a x } _ { c \neq l } \mu ( c , l )$ , by definition $\mu ( c _ { 1 } , l ) = n - \sigma _ { \mathrm { M M } } ( l , G )$ . Let ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ be a such that for all $c \in \mathcal { C } \setminus \{ c _ { 0 } \} , \mu _ { \mathcal { X } } ( c _ { 0 } , c ) = \sigma _ { \mathrm { M M } } ( l , G ) + 1 , \mu _ { \mathcal { X } } ( c _ { 1 } , l ) = n - \sigma _ { \mathrm { M M } } ( l , G )$ and the rest of $\mu _ { \mathcal { X } }$ is null. Then, for all completion $\begin{array} { r } { G ^ { \prime } \mathrm { \ o f \ } \mathcal { X } , \sigma _ { \mathrm { M M } } ( c _ { 0 } , G ^ { \prime } ) = \operatorname* { m i n } _ { c \neq c _ { 0 } } \mu _ { G ^ { \prime } } ( c _ { 0 } , c ) \geq } \end{array}$ $\begin{array} { r } { \operatorname* { m i n } _ { c \neq c _ { 0 } } \mu _ { \mathcal { X } } ( c _ { 0 } , c ) = \sigma _ { \mathrm { M M } } ( l , G ) + 1 } \end{array}$ and $\sigma _ { \mathrm { M M } } ( l , G ^ { \prime } ) =$ min<sub>c̸=l</sub> $\mu _ { G ^ { \prime } } ( l , c ) = n -$ max<sub>c̸=l</sub> $\begin{array} { r } { \mu _ { G ^ { \prime } } ( c , l ) \geq n - \operatorname* { m a x } _ { c \neq l } \mu _ { \mathcal { X } } ( c , l ) \geq n - \mu _ { \mathcal { X } } ( c _ { 1 } , l ) = n - \left( n - \sigma _ { \mathrm { M M } } ( l , G ) \right) = } \end{array}$ $\sigma _ { \mathrm { M M } } ( l , G )$ . Hence, $\sigma _ { \mathrm { M M } } ( c _ { 0 } , G ^ { \prime } ) > \sigma _ { \mathrm { M M } } ( l , G ^ { \prime } )$ and l is a necessary loser for X and X contains a dMS.

$$
\mathrm { I f } \ c _ { 0 } \ \ne \ c _ { 1 } , \ | X | \ = \ ( m - 1 ) ( \sigma _ { \mathrm { M M } } ( l , G ) + 1 ) + n - \sigma _ { \mathrm { M M } } ( l , G ) \ = \ ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ,
$$

$$
1 ) ( m - 2 ) + n + 1
$$

$$
c _ { 0 } ~ = ~ c _ { 1 }
$$

$$
\mu _ { \mathcal { X } } ( c _ { 0 } , l ) = n - \sigma _ { \mathrm { M M } } ( l , G )
$$

$$
\begin{array} { r } { \mu _ { \mathcal { X } } ( c _ { 0 } , l ) = \sigma _ { \mathrm { M M } } ( l , G ) + 1 ~ ( \sigma _ { \mathrm { M M } } ( l , G ) \leq \left\lfloor \frac { n - 1 } { 2 } \right\rfloor } \end{array}
$$

$$
| X | = ( m - 2 ) ( \sigma _ { \mathrm { M M } } ( l , G ) + 1 ) + n - \sigma _ { \mathrm { M M } } ( l , G ) = ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 3 ) + n + 1
$$

$$
\sigma _ { \mathrm { M M } } ( l , G ) + 1
$$

We now prove the matching lower bounds. Given n voters, a complete nweighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , a losing candidate $l \in { \mathcal { C } } \backslash \mathrm { M M } ( G )$ according to Theorem 3, for l to be the necessary MM-loser of a partial subtournament $G ^ { \prime } \subseteq G , G ^ { \prime }$ needs to contain two elements:

– a pairwise comparison in which l is beaten to limit its maximal possible maximin score,

– a set of candidates arranged in a tree where each candidate in the tree beats the non adjacent candidates in the tree with a strictly stronger weight than l’s possible maximin score.

Observe that if the tree contains $k \geq 1$ candidates, $k ( m - 3 ) + 2$ edges needs to have a stronger weight than $l \mathrm { { s } }$ possible maximin score. Each time the maximal possible maximin score of l decreases by one, we can save $k ( m - 3 ) + 2 \geq 1$ comparisons. Hence, to obtain the smallest dMS, we want the maximal possible maximin score of l to be as low as possible, i.e., fixed to $\sigma _ { \mathrm { M M } } ( l , G )$ . Let $c _ { 1 } =$ argma $\mathrm { x } _ { c \neq l } \mu ( c , l )$ , we can fix $\mu _ { \mathcal { X } } ( c _ { 1 } , l ) = n - \sigma _ { \mathrm { M M } } ( l , G )$ to bound the maximal possible maximin score of l to $\sigma _ { \mathrm { M M } } ( l , G )$ using $n - \sigma _ { \mathrm { M M } } ( l , G )$ comparisons.

Additionally, note that $k ( m - 3 ) + 2$ increases with k so we want as few candidates in tree as possible. Since $l \not \in \operatorname { M M } ( G )$ , exists a candidate $c _ { 0 } \neq l$ such that $\sigma _ { \mathrm { M M } } ( c _ { 0 } ) > \sigma _ { \mathrm { M M } } ( l )$ . Thus, we can take the tree reduced to the single vertex $c _ { 0 }$ as $c _ { 0 }$ beats any other candidate with strictly more than $\sigma _ { \mathrm { M M } } ( l )$ comparisons. This gives us a lower bound of $( m - 1 ) ( \sigma _ { \mathrm { M M } } ( l ) + 1 )$ pairwise comparisons for this constraint.

If for all $c \in { \mathcal { C } }$ such that $\sigma _ { \mathrm { M M } } ( c ) > \sigma _ { \mathrm { M M } } ( l ) , \mu ( c , l ) < n - \sigma _ { \mathrm { M M } } ( l )$ both constraints cannot be combined and both sets of pairwise comparisons are disjoint. Hence, a lower bound of $n - \sigma _ { \mathrm { M M } } ( l , G ) + ( m - 1 ) ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) = ( \sigma _ { \mathrm { M M } } ( l ) + 1 ) $ $1 ) ( m - 2 ) + n + 1$ corresponding to the third case of the theorem. In the other case, we can have $c _ { 0 } ~ = ~ c _ { 1 }$ and by fixing $\mu _ { \mathcal { X } } ( c _ { 1 } , l ) \ : = \ : n - \sigma _ { \mathrm { M M } } ( l , G )$ 2 we have $\mu _ { \mathcal { X } } ( c _ { 1 } , l ) \geq \sigma _ { \mathrm { M M } } ( l , G ) + 1$ without requiring additional comparisons. This allows to save an extra $\sigma _ { \mathrm { M M } } ( l , G ) + 1$ which gives a final lower bound of $( \sigma _ { \mathrm { M M } } ( l ) + 1 ) ( m - 3 ) + n + 1$

It naturally follows from the beginning of the proof, that an SdMS can be computed in polynomial time. Finally, since the score of a losing candidate cannot be greater than $\lfloor { \frac { n - 1 } { 2 } } \rfloor$ otherwise, it is a Condorcet winner and thus a Maximin winner, we have that for each SdMS X for maximin, $( \left\lfloor { \frac { n - 1 } { 2 } } \right\rfloor + 1 ) ( m - 2 ) +$ $\begin{array} { r } { n + 1 = \left\lceil \frac { n + 1 } { 2 } \right\rceil ( m - 2 ) + n + 1 } \end{array}$ □

Theorem 9. Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ , a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { w U C } ( G )$ , for each SdMS X for $l \notin$ wU $\mathrm { { I C } } ( G )$

i. $\begin{array} { r } { i f p = \operatorname* { m a x } _ { c _ { 0 } , c _ { 1 } } \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) - n ( m - 3 ) + \left\lfloor \frac { n } { 2 } \right\rfloor > 0 } \end{array}$ with $c _ { 0 } , c _ { 1 } \in ( \mathcal { C } \setminus \{ l \} ) ^ { 2 }$ such that $c _ { 0 } \neq c _ { 1 } , f o r i \in \{ 0 , 1 \} , \mu ( c _ { i } , l ) \geq \frac { n } { 2 }$ , for all $c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \}$ 2 $\mu ( c _ { i } , c ) + \mu ( c , l ) \geq n _ { i }$ , and at least one of the inequalities is strict,

then $\scriptstyle | { \mathcal { X } } | = n ( m - 2 ) + \lceil { \frac { n + 1 } { 2 } } \rceil - p$

ii. else $\begin{array} { r } { | \mathcal { X } | { = } n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil } \end{array}$

An SdMS X can be computed in polynomial time, and $\begin{array} { r } { | \mathcal { X } | \le n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil } \end{array}$

Proof. We first provide upper bounds on the number of pairwise comparisons required in each case by exhibiting partial sub-tournaments where l is a necessary loser in each case.

Given n voters, a complete n-weighted tournament $G = ( \mathcal { C } , \mu )$ with $| { \mathcal { C } } | = m$ a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { w U C } ( G )$

We start with the second case and we show that there always exists a dMS of size $\textstyle n ( m - 2 ) + { \bigl \lceil } { \frac { n + 1 } { 2 } } { \bigr \rceil }$ . Since l is a losing candidate in $G ,$ exists $c _ { 0 } \in$ ${ \mathcal { C } } \setminus \{ l \}$ such that $c _ { 0 }$ weighted covers l, i.e., for all candidates $c \in \mathcal { C } \setminus \{ c _ { 0 } \}$ , $\mu ( c _ { 0 } , c ) ~ \geq ~ \mu ( l , c )$ . Hence, for all candidates $c \in \mathcal { C } \setminus \{ c _ { 0 } , l \} , \mu ( c _ { 0 } , c ) \geq n -$ $\mu ( c , l )$ or equivalently, $\mu ( c _ { 0 } , c ) + \mu ( c , l ) \ge n$ . Let ${ \mathcal { X } } \subseteq G$ be a partial subtournament of G with ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ and $\textstyle \mu _ { \mathcal { X } } ( c _ { 0 } , l ) ~ = ~ \left\lceil \frac { n } { 2 } \right\rceil$ , for all candidates $c \in \mathcal { C } \setminus \{ c _ { 0 } , l \} , \mu _ { X } ( c _ { 0 } , c ) + \mu _ { X } ( c , l ) \geq n$ . For each completion $\boldsymbol { \mathcal { y } } = ( \mathcal { C } , \mu _ { \mathcal { y } } )$ of $\mathcal { X } , \mu _ { \mathcal { Y } } ( c _ { 0 } , l ) \ge \frac { n } { 2 } \ge \mu _ { \mathcal { Y } } ( l , c _ { 0 } )$ . Additionally, for all candidates $c \in \mathcal { C } \setminus \{ c _ { 0 } , l \}$ $\mu _ { Y } ( c _ { 0 } , c ) + \mu _ { \mathcal { Y } } ( \bar { c _ { , } } l ) \geq \mu _ { X } ( c _ { 0 } , c ) + \mu _ { \mathcal { X } } ( c , l ) \geq n$ . Thus, for all candidates $c \in$ ${ \mathcal { C } } \setminus \{ c _ { 0 } , l \} , \mu _ { Y } ( c _ { 0 } , c ) \geq n - \mu _ { \mathcal { Y } } ( c , l ) = \mu _ { \mathcal { Y } } ( l , c )$ . We have that l is weighted covered by $c _ { 0 }$ in $\mathcal { V }$ and $l \in \mathrm { N L } _ { \mathrm { w U C } } ( \mathcal { X } )$ . Finally, for at least one of the inequalities to be strict we need an additional comparison if n is even and none if n is odd as $\textstyle \mu _ { \mathcal { Y } } ( c _ { 0 } , l ) \geq \frac { n } { 2 }$ is already strict. We encode this by changing the $\left\lceil { \frac { n } { 2 } } \right\rceil$ into $\textstyle \left\lceil { \frac { n + 1 } { 2 } } \right\rceil$

Since, $\begin{array} { r } { | { \mathcal { X } } | = n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil , n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil } \end{array}$ pairwise comparisons are enough.

We now move on to the first case. Suppose $\begin{array} { r } { p = \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) - n ( m - } \end{array}$ $3 ) - \left\lfloor { \frac { n } { 2 } } \right\rfloor > 0$ where $( c _ { 0 } , c _ { 1 } ) \in ( \mathcal { C } \setminus \{ l \} ) ^ { 2 }$ maximizes $\textstyle \sum _ { c \in { \mathcal { C } } \setminus \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l )$ with $c _ { 0 } \neq$ $\begin{array} { r } { c _ { 1 } , \mu ( c _ { i } , l ) \ge \frac { n } { 2 } } \end{array}$ , for all $c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} , \mu ( c _ { i } , c ) + \mu ( c , l ) \overset { \cdot } { \geq } n \mathrm { f o r } i \in \{ 0 , 1 \}$ and at least one of the inequalities is strict. Let $\mathcal { X } \subseteq G$ be a partial sub-tournament of G with $\mathcal { X } = ( \mathcal { C } , \mu _ { \mathcal { X } } )$ and $\textstyle \mu _ { \mathcal { X } } ( c _ { i } , l ) = \left\lceil \frac { n } { 2 } \right\rceil$ , for all $c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} , \mu _ { \mathcal { X } } ( c , l ) = \mu ( c , l )$ and $\mu _ { \mathcal { X } } ( c _ { i } , c ) = n - \mu ( c , l ) \mathrm { ~ f o r ~ } i \in \{ 0 , 1 \}$ . For each completion $\mathcal { V } = ( \mathcal { C } , \mu _ { \mathcal { V } } )$ of $x ,$ since $\begin{array} { r } { \mu y ( c _ { 0 } , c _ { 1 } ) + \mu y ( c _ { 1 } , c _ { 0 } ) = n , \mu y ( c _ { 0 } , c _ { 1 } ) \geq \left\lceil \frac { n } { 2 } \right\rceil \mathrm { o r } \mu y ( c _ { 1 } , c _ { 0 } ) \geq \left\lceil \frac { n } { 2 } \right\rceil } \end{array}$ . Without loss of generality, suppose $\textstyle \mu _ { \mathcal { Y } } ( c _ { 0 } , c _ { 1 } ) \geq \left\lceil \frac { n } { 2 } \right\rceil$ . Since $\begin{array} { r } { \mu _ { \mathcal { X } } ( c _ { 1 } , l ) = \left\lceil \frac { n } { 2 } \right\rceil , \mu _ { \mathcal { Y } } ( l , c _ { 1 } ) = } \end{array}$ $\begin{array} { r } { n - \mu _ { \mathcal { V } } ( c _ { 1 } , l ) \le n - \mu _ { \mathcal { X } } ( c _ { 1 } , l ) = n - \left\lceil \frac { n } { 2 } \right\rceil = \left\lfloor \frac { n } { 2 } \right\rfloor } \end{array}$ . Hence $\mu _ { \mathcal { V } } ( c _ { 0 } , c _ { 1 } ) \geq \mu _ { \mathcal { V } } ( l , c _ { 1 } )$ Additionally, $\textstyle \mu _ { \mathcal { Y } } ( c _ { 0 } , l ) \geq \mu _ { \mathcal { X } } ( c _ { 0 } , l ) \geq \left\lceil \frac { n } { 2 } \right\rceil$ and $\mu _ { \mathcal { V } } ( l , c _ { 0 } ) = n - \mu _ { \mathcal { V } } ( c _ { 0 } , l ) \le n -$ $\left\lceil { \frac { n } { 2 } } \right\rceil ~ \leq ~ \left\lfloor { \frac { n } { 2 } } \right\rfloor$ . Thus $\mu _ { \mathcal { Y } } ( c _ { 0 } , l ) \geq \mu _ { \mathcal { Y } } ( l , c _ { 0 } )$ . Moreover, for all $c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \}$ , $\mu _ { \mathcal { V } } ( c _ { 0 } , c ) + \mu _ { \mathcal { V } } ( c , l ) \geq \mu _ { \mathcal { X } } ( c _ { 0 } , c ) + \mu _ { \mathcal { X } } ( c , l ) = \mu ( c , l ) + n - \mu ( c , l ) = n$ . Thus $\mu _ { \mathcal { V } } ( c _ { 0 } , c ) \geq n - \mu _ { \mathcal { V } } ( c , l ) = \mu _ { \mathcal { V } } ( l , c )$ . Finally,

$$
\begin{array} { l } { { 2 \left\lceil \displaystyle { \frac { n } { 2 } } \right\rceil + \sum _ { \substack { c \in { \mathcal { C } } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } } \mu ( c , l ) + 2 \sum _ { \substack { c \in { \mathcal { C } } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } } \left( n - \mu ( c , l ) \right) } }  \\ { { = 2 \left\lceil \displaystyle { \frac { n } { 2 } } \right\rceil + 2 \sum _ { \substack { c \in { \mathcal { C } } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } } n - \sum _ { \substack { c \in { \mathcal { C } } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } } \left( \mu ( c , l ) \right) } } \\ { { = 2 \left\lceil \displaystyle { \frac { n } { 2 } } \right\rceil + 2 n ( m - 3 ) - \sum _ { \substack { c \in { \mathcal { C } } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } } \left( \mu ( c , l ) \right) } } \\ { { = \left\lceil \displaystyle { \frac { n } { 2 } } \right\rceil + n ( m - 3 ) - p } } \end{array}
$$

Additionally, like in the previous case, for at least one of the inequalities to be strict we need an additional comparison if n is even and none if n is odd as $\textstyle \mu _ { \mathcal { Y } } ( c _ { 0 } , l ) \geq \frac { n } { 2 }$ is already strict. We encode this by changing the $\textstyle { \left\lceil { \frac { n } { 2 } } \right\rceil } { \mathrm { ~ i n t o ~ } } \left\lceil { \frac { n + 1 } { 2 } } \right\rceil$

$$
| { \mathcal { X } } | = \left\lceil { \frac { n + 1 } { 2 } } \right\rceil + n ( m - 3 ) - p < \left\lceil { \frac { n + 1 } { 2 } } \right\rceil + n ( m - 3 )
$$

Since, $\begin{array} { r } { | \mathcal { X } | = n ( m - 3 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil - p , n ( m - 3 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil - p } \end{array}$ pairwise comparisons are enough.

We now prove the matching lower bounds.

Since dMSs are partial tournaments where l is a necessary loser, we use the characterization provided in Theorem 4 to obtain lower bounds on the riquered number of pairwise comparisons. Given n voters, a complete n-weighted tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { w U C } ( G )$ . According to Theorem 4, for each dMS ${ \mathcal { X } } = ( { \mathcal { C } } , \mu _ { \mathcal { X } } )$ for $l \not \in \mathrm { w U C } ( G )$ , we have that exists a tree $T = ( K , E )$ with $K \subseteq \mathcal { C } \setminus \{ l \}$ such that for each $\textstyle c \in K , \mu _ { \mathcal { X } } ( c , l ) \geq \frac { n } { 2 }$ and $\forall c ^ { \prime } \in \mathcal { C } \backslash \{ l \} , \forall c \in K , \mathrm { i f ~ i f ~ } \{ c , c ^ { \prime } \} \not \in \overline { { E } }$ then $\mu _ { \mathcal { X } } ( c , c ^ { \prime } ) + \mu _ { \mathcal { X } } ( c ^ { \prime } , l ) \geq n$ and at least one of the inequalities is strict. In particular, $\forall c ^ { \prime } \in { \mathcal { C } } \setminus \{ l \}$ , unless $\forall c \in K , \{ c , c ^ { \prime } \} \in E .$ $\mathrm { i . e , } T$ is a star graph with center $c ^ { \prime } , \exists c \in K$ such that $\mu _ { \mathcal { X } } ( c , c ^ { \prime } ) + \mu _ { \mathcal { X } } ( c ^ { \prime } , l ) \geq n .$

Note that if $| K | = 1$ and we denote $K = \{ c \}$ , then $T$ is a star graph with center c. $\mathrm { I f ~ } | K | = 2$ and $K = \{ c _ { 0 } , c _ { 1 } \}$ , then $T$ is both a star graph with center $c _ { 0 }$ and a star graph with center $c _ { 1 } . \mathrm { I f } \ | K | \geq 3$ , there exists at most on candidate c such that $T$ is a star graph with center c otherwise $T$ would have a cycle.

Additionally, if T is a star graph with center $^ { c , }$ we have $c \in K$ and $\mu _ { \mathcal { X } } ( c , l ) \geq$ ${ \frac { n } { 2 } } .$

Hence, if $| K | = 1$ or $| K | \geq 3$ , if T is a star graph we have $m - 2$ independent inequalities of the form $\mu _ { \mathcal { X } } ( c , c ^ { \prime } ) + \mu _ { \mathcal { X } } ( c ^ { \prime } , l ) \geq n$ and one of the form $\mu _ { \mathcal { X } } ( c , l ) \geq$ $\frac { n } { 2 }$ , and at least one has to be strict thus $| \mathcal { X } | \ge ( m - 2 ) n + \lceil \frac { n + 1 } { 2 } \rceil$ else we have $m - 1$ independent inequalities of the form $\mu _ { \mathcal { X } } ( c , c ^ { \prime } ) + \mu _ { \mathcal { X } } ( c ^ { \prime } , l ) \geq n$ and $| \mathcal { X } | \geq ( m - 1 ) n \geq ( m - 2 ) n + \lceil \frac { n + 1 } { 2 } \rceil$

If $| K | = 2$ and $K = \{ c _ { 0 } , c _ { 1 } \}$ , our set of constrains is $\textstyle \mu _ { \mathcal { X } } ( c _ { i } , l ) ~ \geq ~ \frac { n } { 2 }$ and for all $c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} , \mu _ { \mathcal { X } } ( c _ { i } , c ) + \mu _ { \mathcal { X } } ( c , l ) \geq n$ for $i \in \{ 0 , 1 \}$ and at least one inequality has to be strict. Since $\mu _ { \mathcal { X } } ( c , l )$ appears both in $\mu _ { \mathcal { X } } ( c _ { 0 } , c ) + \mu _ { \mathcal { X } } ( c , l ) \geq n$ and $\mu _ { \mathcal { X } } ( c _ { 1 } , c ) + \mu _ { \mathcal { X } } ( c , l ) \geq n$ , to obtain the smallest dMS, it is clear that we have to take its highest possible value which is $\mu ( c , l )$ . In that case, if we ignore the strict inequality constraint,

$$
\begin{array} { l } { \displaystyle | \mathcal { X } | = 2 \left\lceil \frac { n } { 2 } \right\rceil + \displaystyle \sum _ { c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) + 2 \sum _ { c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} } \left( n - \mu ( c , l ) \right) } \\ { \displaystyle = 2 \left\lceil \frac { n } { 2 } \right\rceil + 2 n ( m - 3 ) - \displaystyle \sum _ { c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} } ( \mu ( c , l ) ) . } \end{array}
$$

Once again, for at least one of the inequalities to be strict we need an additional comparison we change one $\left\lceil { \frac { n } { 2 } } \right\rceil$ into $\textstyle \left\lceil { \frac { n + 1 } { 2 } } \right\rceil$

If $\begin{array} { r } { p = \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) - n ( m - 3 ) - \left\lfloor \frac { n } { 2 } \right\rfloor > 0 } \end{array}$ as in the first case of the theorem then

$$
\begin{array} { l } { \displaystyle | \mathcal { X } | = \left\lceil \frac { n } { 2 } \right\rceil + \left\lceil \frac { n + 1 } { 2 } \right\rceil + 2 n ( m - 3 ) - \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } ( \mu ( c , l ) ) } \\ { = \displaystyle \left\lceil \frac { n + 1 } { 2 } \right\rceil + n ( m - 3 ) - p } \\ { < \displaystyle \left\lceil \frac { n + 1 } { 2 } \right\rceil + n ( m - 2 ) . } \end{array}
$$

Else taking $| K | = 2$ does not help to obtain smaller dMSs and we go back to the general approach with $| K | = 1$

It naturally follows from the beginning of the proof, that an SdMS can be computed in polynomial time. Finally, since the score of a losing candidate cannot be greater than $\lfloor { \frac { n - 1 } { 2 } } \rfloor$ otherwise, it is a Condorcet winner and thus a Maximin winner, we have that for each SdMS X for the weighted uncovered set, $\begin{array} { r } { | \mathcal { X } | \le n ( m - 2 ) + \left\lceil \frac { n + 1 } { 2 } \right\rceil } \end{array}$ □

Corollary 1. Given a complete tournament $G = ( { \mathcal C } , \mu )$ with $| { \mathcal { C } } | = m$ , and a losing candidate $l \in { \mathcal { C } } \setminus \operatorname { U C } ( G )$ , for each SdMS X $f o r \ l \ \not \in \ \mathrm { U C } ( G )$ , we have $\scriptstyle | { \mathcal { X } } | = m - 1$

Proof. In the unweighted case $\begin{array} { r } { ( n = 1 ) , \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) \leq \sum _ { c \in \mathcal { C } \backslash \{ l , c _ { 0 } , c _ { 1 } \} } 1 \leq } \end{array}$ $m - 3$ . Hence, $\begin{array} { r } { p = \sum _ { c \in \mathcal { C } \setminus \{ l , c _ { 0 } , c _ { 1 } \} } \mu ( c , l ) - ( m - 3 ) - \left\lfloor \frac { 1 } { 2 } \right\rfloor \leq 0 } \end{array}$ and the first case of Theorem 9 never occurs. □