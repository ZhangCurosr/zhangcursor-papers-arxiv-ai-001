# Exact Cut Complexity of Equal-Length Proportional Cake Cutting<sup>∗</sup>

Yasushi Kawase<sup>1</sup>, Bodhayan Roy<sup>2</sup>, and Mohammad Azharuddin Sanpui<sup>2</sup>

<sup>1</sup>Chuo University, Tokyo, Japan

<sup>2</sup> Indian Institute of Technology Kharagpur, Kharagpur, West Bengal, India

## Abstract

We study proportional cake cutting under an equal-length constraint, where each of n agents must receive a bundle of length exactly 1/n that is worth at least 1/n to that agent. Our main result determines the exact worst-case number of cuts needed to guarantee such an allocation, namely 2n − 2 for every n. This improves on the previously known quadratic upper bound. The proof combines the constrained necklace-splitting theorem with a recursive assignment argument based on Hall’s theorem. The same bound extends to the stronger requirement of envy-freeness when n is a prime power, and an $O ( n ^ { 1 . 5 2 5 } )$ upper bound holds for arbitrary n. We then ask whether such allocations can be computed. We first show that no deterministic protocol in the Robertson–Webb model terminates after finitely many queries on every instance and always returns an exact equal-length proportional allocation, even for two agents with identical valuations. For every $\varepsilon > 0$ , however, $O ( n ^ { 2 } / \varepsilon )$ queries sufice to produce an allocation in which every agent receives a bundle of length exactly 1/n and value at least $1 / n - \varepsilon$ , again using at most 2n − 2 cuts. Finally, we augment the Robertson–Webb model with two-knife queries. We show that recursive balanced halving attains the optimal bound of $2 n - 2$ cuts when n is a power of two. For arbitrary n, we give a procedure using O(n log n) cuts.

Keywords: cake cutting, fair division, cut complexity, query complexity, proportionality, envy-freeness, Robertson–Webb model, two-knife queries

## 1 Introduction

Fair division asks how to allocate divisible and indivisible resources among agents with heterogeneous preferences [2, 10, 28]. Cake cutting is its standard model for divisible resources. Proportional and envy-free cake cutting have been studied extensively, with existential, algorithmic, and complexity results for many variants [5, 8, 14, 16, 24, 30].

In many fair-division settings, the size of each share is fixed independently of the agents’ preferences. Appointment slots have prescribed durations, bandwidth blocks have fixed widths, and land plots may have standardized areas. In each case, fairness must be achieved subject to an exogenous size constraint.

We consider the interval cake [0, 1], where each of the n agents must receive a bundle of length exactly 1/n. In the terminology of fair division with market values [7], Lebesgue length is a common objective valuation. We treat this measure as public information and the agents’ valuations as private. Proportionality is imposed with respect to the private valuations. Thus, each agent’s bundle must have private value at least $1 / n$ and length exactly $1 / n$

Existence alone does not control fragmentation. A schedule split among many short intervals can be inconvenient, and a land allotment divided into many patches can be costly to use. On an interval, the natural measure of fragmentation is the number of cuts. We ask for the minimum number of cuts that always sufices to give every agent a bundle of length $1 / n$ and value at least $1 / n$ . Without the equal-length constraint, the classical proportionality guarantee can be attained with connected pieces and $n - 1$ cuts [27, 28]. Our main result shows that imposing equal lengths increases the exact worst-case cut complexity to $2 n - 2$ Throughout this paper, a cut is a distinct boundary point retained in the final allocation and measures fragmentation. It is not a Robertson–Webb Cut query, which is an information-acquisition operation counted separately in Section 6.

Before this work, the only upper bound for all n was quadratic. It follows from the corresponding envyfree problem with market values [7], which imposes the same constraint on the common objective measure as our setting but strengthens proportionality to envy-freeness for the private valuations.

## 1.1 Our Results

Let $\kappa _ { n } ^ { \mathrm { p r o p } }$ denote the least number of cuts that always sufice for equal-length proportional cake cutting with n agents. Our main theorem determines this number exactly $\kappa _ { n } ^ { \mathrm { p r o p } } = 2 n - 2$ for all $n \geq 1$ . An identicalvaluation instance gives the lower bound (Proposition 1). The remainder of Section 4 proves the matching upper bound.

To show this, we use the constrained necklace-splitting theorem of Joji´c, Panina, and Zivaljevi´c [19].<sup>ˇ</sup> For a prime-power divisor r of $n ,$ it produces r equal-length top-level bundles while balancing a fractional assignment of agents to bundles. Hall’s theorem converts the fractional support into r equally sized agent groups, after which we recurse within each bundle. This gives the recurrence in Lemma 7 and the exact bound in Theorem 4.

We also consider the stronger requirement of equal-length envy-freeness. Let $\kappa _ { n } ^ { \mathrm { e f } }$ denote its worst-case cut complexity. When n is a prime power, the constrained necklace-splitting theorem and the proportional lower bound give $\kappa _ { n } ^ { \mathrm { e f } } = 2 n - 2$ (Proposition 2). For arbitrary n, a peeling argument based on the Stromquist– Woodall exact-share theorem gives $\kappa _ { n } ^ { \mathrm { e f } } = O ( n ^ { 1 . 5 2 5 } )$ (Proposition 3). Determining $\kappa _ { n } ^ { \mathrm { e f } }$ exactly for general n remains open.

Existence does not imply finite query complexity. Even with free access to length, no deterministic protocol in the Robertson–Webb model terminates after finitely many queries on every instance and always returns an exact equal-length proportional allocation. This impossibility holds even for two agents with the same valuation (Theorem 8). For every $\varepsilon > 0$ , however, discretization computes an equal-length $\varepsilon -$ proportional allocation using $O ( n ^ { 2 } / \varepsilon )$ queries and at most $2 n - 2$ cuts (Theorem 9). A polynomial-time alternative in the real-RAM oracle model uses $O ( n ^ { 3 } / \varepsilon )$ queries and $O ( n ^ { 2 } / \varepsilon )$ cuts (Proposition 4).

With two-knife queries in addition to Robertson–Webb queries, equal-length proportional allocations can be produced without approximation. When n is a power of two, recursive balanced halving uses Move-Length queries, which move a fixed-length window until a prescribed number of agents value it at least their specified thresholds. The procedure uses at most $n - 1$ two-knife queries and the optimal $2 n - 2$ cuts (Theorem 10). For arbitrary $n ,$ a MoveValue query follows Austin’s two-knife procedure by keeping one agent’s value between the knives fixed and stopping when the portion between them reaches a specified length. Combining this query with a matching-based recursion gives an equal-length proportional allocation using $O ( n ^ { 5 } )$ Robertson–Webb queries, $O ( n ^ { 2 } )$ two-knife queries, and O(n log n) cuts for arbitrary nonatomic valuations (Theorem 11).

## 1.2 Related Work

The most directly related work concerns fair division with market values. Barman, Ebadian, Latifian, and Shah [7] study fair division with both private valuations and a common market valuation. Taking that common valuation to be Lebesgue length gives the constraint considered here. In the cake-cutting setting, their result gives the upper bound $\kappa _ { n } ^ { \mathrm { e f } } \leq n ( n - 1 )$ for the stronger requirement of envy-freeness with respect to both the private valuations and the common market valuation, and their Open Question 5 asks whether $O ( n )$ cuts always sufice. We do not settle that question. In the equal-length setting, Section 5 improves their quadratic bound to $O ( n ^ { 1 . 5 2 5 } )$ . Since envy-freeness implies proportionality, their quadratic bound also applied to the proportional problem, but it did not determine its exact cut complexity.

Equal-size constraints also appear for indivisible goods. Biswas and Barman study cardinality constraints [9], while Kawase and Mahara study balanced allocations, in which all agents receive the same number of items [20]. Related models appear in [11, 18, 22, 23].

Proportional cake cutting with unequal or weighted entitlements also minimizes cuts subject to privatevalue guarantees. In that setting, Segal-Halevi [27] proved a general $O ( n \log n )$ upper bound together with a $2 n - 2$ lower bound, and Crew, Narayanan, and Spirkl [13] improved the upper bound to $3 n - 4$ . These results do not require each allocated bundle to have length exactly $1 / n ,$ or more generally common-measure value exactly $1 / n$ , and therefore do not imply the present theorem.

Exact division and continuous necklace splitting require every agent to value all final bundles equally. This stronger requirement yields quadratic cut bounds, for example via Alon [1]; see also Goldberg and Li [17] for exact divisions into prescribed ratios. Exact division is stronger than equal-length proportionality, which requires only the assigned bundle to be worth at least $1 / n$ to each agent. We instead use the constrained necklace-splitting theorem of Joji´c, Panina, and Zivaljevi´c [19]. Its preference version simultaneously equal-<sup>ˇ</sup> izes length and produces an accepted assignment. For a prime-power number of agents, choosing preferences according to private values directly gives the optimal $2 n - 2$ bound.

Exact fairness can be incompatible with finite Robertson–Webb interaction. Exact division is a classical example [26]. Procaccia and Wang [25] give quantitative lower bounds for approximate equitability and show that exact equitability cannot be computed even without a uniform query bound. Here length is known and only the private valuation is queried. The proof in Section 6 exploits this distinction by perturbing private masses at geometric cuts and geometric positions at value cuts. For unconstrained proportional cake cutting, Arndt, Moseley, Im, and Pruhs [3] recently gave a deterministic algorithm using $O ( n )$ Cut queries and $O ( n ^ { 2 } )$ Eval queries. Their result does not impose equal lengths and counts query operations rather than cuts retained in the final allocation.

Austin’s two-knife procedure assumes that an agent can move two knives continuously while keeping the intervening piece at a fixed value [4]. Our additional query uses this continuous motion and stops when the intervening piece reaches a specified length.

## 2 Preliminaries

We consider allocating the cake [0, 1] among n agents, indexed by $\{ 1 , \ldots , n \}$ . Each agent i has a valuation $v _ { i } ,$ modeled as a nonatomic Borel probability measure on $[ 0 , 1 ]$ . Thus, $v _ { i } ( [ 0 , 1 ] ) = 1$ . Let λ denote Lebesgue measure on [0, 1].

An allocation is a measurable partition $( C _ { 1 } , \ldots , C _ { n } )$ of [0, 1]. We call it equal-length if $\lambda ( C _ { i } ) = 1 / n$ for every $i \in \{ 1 , \ldots , n \}$ We call it proportional if $v _ { i } ( C _ { i } ) \geq 1 / n$ for every $i \in \{ 1 , \ldots , n \}$ , and envy-free if $v _ { i } ( C _ { j } ) \leq v _ { i } ( C _ { i } )$ for every $i , j \in \{ 1 , \ldots , n \}$ . Every envy-free allocation is proportional because $v _ { i } ( C _ { i } )$ is then at least the average of $v _ { i } ( C _ { 1 } ) , \dots , v _ { i } ( C _ { n } )$ , which is $1 / n$

Given q distinct interior cut points, list them as $0 < t _ { 1 } < \cdots < t _ { q } < 1$ , put $t _ { 0 } = 0$ and $t _ { q + 1 } = 1$ , and let $m = q + 1$ . For $q = 0$ , this list is empty and $m = 1$ . For $j \in \{ 1 , \dots , m \}$ , define the induced pieces by

$$
I _ { j } : = { \left\{ \begin{array} { l l } { \left[ t _ { 0 } , t _ { 1 } \right] } & { { \mathrm { i f ~ } } j = 1 , } \\ { \left( t _ { j - 1 } , t _ { j } \right] } & { { \mathrm { i f ~ } } j \geq 2 . } \end{array} \right. }
$$

Thus, every interior cut point is assigned to the piece immediately to its left. We say that an allocation uses at most $q$ cuts if there exist at most $q$ distinct interior cut points such that each bundle is a union of the induced pieces. In configuration spaces below, formal cut coordinates may coincide. Such coincident coordinates create only degenerate intervals and never increase the number of distinct cuts in the original cake. Let $\kappa _ { n } ^ { \mathrm { p r o p } }$ denote the least number of cuts that always sufice for equal-length proportional cake cutting with n agents. Similarly, let $\kappa _ { n } ^ { \mathrm { e f } }$ denote the least number of cuts that always sufice for equal-length envy-free cake cutting with n agents.

The following exact-share theorem is the main tool in the peeling argument of Section 5.

Theorem 1 (Stromquist–Woodall [29]). Let $d \geq 2$ , let $\eta _ { 1 } , \ldots , \eta _ { d }$ be nonatomic probability measures on [0, 1], and let $\alpha \in [ 0 , 1 ]$ . Then there exists a measurable set $A \subseteq [ 0 , 1 ]$ such that $\eta _ { j } ( A ) = \alpha$ for every $j \in \{ 1 , \ldots , d \}$ Moreover, A can be chosen as a union of pieces induced by at most $2 d - 2 \ c u t s .$

Remark 2. Stromquist and Woodall originally state the cut-count guarantee for a circle. In our interval notation, this means viewing [0, 1] as a circle by identifying 0 and 1. The exact-share set can then be chosen as a union of at most $d - 1$ connected pieces of the circle. Opening the circle at the identified endpoint gives a subset of [0, 1] with at most $2 d - 2$ distinct interior boundary points. Since all measures are nonatomic, assigning the opened endpoint to either side does not change any measure.

## 3 The Constrained Necklace-Splitting Theorem

We use an interval form of constrained necklace splitting to balance the fractional assignments in the proportional proof. Its preference form will also give the prime-power envy-free bound in Section 5.

## 3.1 Configuration Space and Theorem Statement

Fix integers $m , r \geq 1$ . Let $\begin{array} { r } { \Delta _ { m } : = \left\{ ( y _ { 1 } , \dotsc , y _ { m } ) \in \mathbb { R } _ { > 0 } ^ { m } : \sum _ { k = 1 } ^ { m } y _ { k } = 1 \right\} } \end{array}$ be the simplex of possible lengths of m consecutive subintervals. For $y \in \Delta _ { m }$ , put $x _ { 0 } ( y ) = 0$ and $\begin{array} { r } { x _ { k } ( y ) = \sum _ { \ell = 1 } ^ { k } y _ { \ell } } \end{array}$ for $k \in \{ 1 , \ldots , m \}$ . These cut positions form m possibly degenerate consecutive intervals. An $( m , r )$ -configuration is an equivalence class $\xi = [ ( y , f ) ]$ , where $y \in \Delta _ { m }$ and $f \colon \{ 1 , \ldots , m \}  \{ 1 , \ldots , r \}$ labels the intervals; two pairs are identified when they difer only on indices k with $y _ { k } = 0$ . Let $\mathcal { C } _ { m , r }$ be this quotient space. It is a compact metric space, being a finite simplicial quotient of $\Delta _ { m } \times \{ 1 , \ldots , r \} ^ { \{ 1 , \ldots , m \} }$

For a label $i ,$ let $V _ { i } ( \xi )$ be the union of the intervals labeled i, understood up to boundary points. The bundles $V _ { 1 } ( \xi ) , \dots , V _ { r } ( \xi )$ partition the cake and together use at most $m - 1$ cuts. The symmetric group $S _ { r }$ acts by relabeling

$$
\sigma \cdot [ ( y , f ) ] = [ ( y , \sigma \circ f ) ] \quad { \mathrm { a n d } } \quad V _ { \sigma ( i ) } ( \sigma \cdot \xi ) = V _ { i } ( \xi )
$$

up to boundaries.

Definition 1. A preference matrix on $\mathcal { C } _ { m , r }$ is a family $\{ A _ { i } ^ { j } \} _ { i , j \in \{ 1 , \dots , r \} }$ of subsets of ${ \mathcal { C } } _ { m , r } ,$ where $A _ { i } ^ { j }$ is the set of configurations at which agent $j$ accepts label i. It is closed $i f$ every $A _ { i } ^ { j }$ is closed, proper $i f \cup _ { i = 1 } ^ { r } A _ { i } ^ { j } = \mathcal { C } _ { m , r }$ for every agent $j ,$ and $S _ { r }$ -equivariant if $\sigma \cdot A _ { i } ^ { j } = A _ { \sigma ( i ) } ^ { j }$ for every $\sigma \in S _ { r }$ and every $i , j \in \{ 1 , \ldots , r \}$ , where $\sigma \cdot A _ { i } ^ { j } : = \{ \sigma \cdot \xi : \xi \in A _ { i } ^ { j } \}$

Thus $A _ { i } ^ { j }$ records when agent $j$ accepts bundle i; closedness preserves acceptance under limits, properness guarantees an accepted label, and equivariance makes acceptance independent of label names.

The following result is Theorem 6.9 of Joji´c, Panina, and Zivaljevi´c [19], translated to our relabeling <sup>ˇ</sup> convention. On an interval, nonatomicity gives the required continuity. For a prime power $r ,$ the theorem partitions the cake into r labeled bundles, equalizes t measures across them, and realizes an accepted assignment with $( r - 1 ) ( t + 1 )$ cuts.

Theorem 3 (Theorem 6.9 of [19]). Let $r \geq 2$ be a prime power, let $t \geq 1$ , and put $q : = ( r - 1 ) ( t + 1 )$ and $m : = q + 1$ . Let $\mu _ { 1 } , \ldots , \mu _ { t }$ be nonatomic probability measures on $[ 0 , 1 ]$ , and let $\{ A _ { i } ^ { \mathcal { I } } \} _ { i , j \in \{ 1 , . . . , r \} }$ be a closed, $S _ { r }$ -equivariant, proper preference matrix on $\mathcal { C } _ { m , r }$ . Then there exist a configuration $\xi \in \mathcal { C } _ { m , r }$ and a permutation $\pi \in S _ { r }$ such that $\xi \in A _ { \pi ( j ) } ^ { j }$ for every $j \in \{ 1 , \ldots , r \}$ , and $\mu _ { \tau } ( V _ { i } ( \xi ) ) = 1 / r$ for every $\tau \in \{ 1 , \ldots , t \}$ and every $i \in \{ 1 , \ldots , r \}$ . In particular, the resulting partition uses at most $q = ( r - 1 ) ( t + 1 )$ cuts.

Corollary 1. Under the assumptions of Theorem ${ \mathcal { B } } ,$ there exists a configuration $\xi \in \mathcal { C } _ { m , r }$ such that $\xi \in A _ { i } ^ { j }$ for every $j \in \{ 1 , \ldots , r \}$ , and $\mu _ { \tau } ( V _ { i } ( \xi ) ) = 1 / r$ for every $\tau \in \{ 1 , \ldots , t \}$ and every $i \in \{ 1 , \ldots , r \}$ . The resulting partition uses at most $q = ( r - 1 ) ( t + 1 )$ cuts.

Proof. Let $( \xi , \pi )$ be given by Theorem 3 and set $\xi ^ { \prime } = \pi ^ { - 1 }$ · ξ. Equivariance sends $A _ { \pi ( j ) } ^ { j }$ to $A _ { j } ^ { j }$ . Thus, $\xi ^ { \prime } \in A _ { i } ^ { j }$ for every j. Relabeling does not change any measure equality or the cut count.

This diagonal form assigns each agent an accepted label while balancing every auxiliary measure. We also need the following continuity fact.

Lemma 1. Let ν be a nonatomic finite measure on [0, 1], and fix a label $i \in \{ 1 , \ldots , r \}$ . Then the map $\xi \mapsto \nu ( V _ { i } ( \xi ) )$ is continuous on $\mathcal { C } _ { m , r } .$

Proof. Fix a label function $f \colon \{ 1 , \ldots , m \}  \{ 1 , \ldots , r \}$ . On the fixed-label subset represented by $( y , f )$ , let $x _ { 0 } ( y ) = 0$ and $\begin{array} { r } { x _ { k } ( y ) = \sum _ { \ell = 1 } ^ { k } y _ { \ell } } \end{array}$ . Under the convention of Section 2, the first piece is $[ x _ { 0 } ( y ) , x _ { 1 } ( y ) ]$ ] and the kth piece is $( x _ { k - 1 } ( y ) , x _ { k } ( y ) ]$ for $k \geq 2$ . Hence $\nu ( V _ { i } ( \xi ) )$ is a finite sum of diferences $F _ { \nu } ( x _ { k } ( y ) ) - F _ { \nu } ( x _ { k - 1 } ( y ) )$ where $F _ { \nu } ( x ) = \nu ( [ 0 , x ] )$ . Since ν is nonatomic, $F _ { \nu }$ is continuous, and hence the map is continuous on each fixed-label subset.

If two representatives difer only by relabeling degenerate pieces, then the corresponding bundle unions difer only by sets of ν-measure zero. Thus the subsetwise formulas agree on overlaps and descend to the quotient $\mathcal { C } _ { m , r }$ . Since the quotient is obtained by gluing finitely many fixed-label subsets, the pasting lemma gives continuity on $\mathcal { C } _ { m , r } .$ □

## 3.2 Assignment Equalization

The transfer argument needs an equal-length configuration that equalizes the coordinates of a continuous equivariant map. The following direct consequence of Corollary 1 supplies it; later, $F _ { i }$ is the total fractional weight assigned to bundle i.

Lemma 2. Let $r \geq 2$ be a prime power. Suppose $F = ( F _ { 1 } , \dots , F _ { r } ) \colon \mathcal { C } _ { 2 r - 1 , r } \  \ \mathbb { R } ^ { r }$ is continuous and satisfies $F _ { \sigma ( i ) } ( \sigma \cdot \xi ) = F _ { i } ( \xi )$ for every $\sigma \in S _ { r }$ and $i \in \{ 1 , \ldots , r \}$ . Then there exists $\xi \in \mathcal { C } _ { 2 r - 1 , r }$ such that $\lambda \left( V _ { 1 } ( \xi ) \right) = \dot { \cdot } \cdot \cdot = \lambda \left( V _ { r } ( \xi ) \right) = 1 / r$ , and $F _ { 1 } ( \xi ) = \cdot \cdot \cdot = F _ { r } ( \xi )$

Proof. For each label $i \in \{ 1 , \ldots , r \}$ , define

$$
\begin{array} { r } { A _ { i } : = \left\{ \xi \in \mathcal { C } _ { 2 r - 1 , r } : F _ { i } ( \xi ) \geq \operatorname* { m a x } _ { i ^ { \prime } \in \{ 1 , \dots , r \} } F _ { i ^ { \prime } } ( \xi ) \right\} . } \end{array}
$$

Let every agent have the same preference family, i.e., $A _ { i } ^ { j } : = A _ { i }$ for all $i , j ~ \in ~ \{ 1 , \ldots , r \}$ . Because $F$ is continuous, each set $A _ { i }$ is closed. Because at least one entry of $F ( \xi )$ is maximal at every $\xi ,$ the matrix $\{ A _ { i } ^ { j } \} _ { i , j \in \{ 1 , . . . , r \} }$ is proper. Because $F _ { \sigma ( i ) } ( \sigma \cdot \xi ) = F _ { i } ( \xi )$ , relabeling labels by any permutation $\sigma \in S _ { r }$ sends maximizing labels at a configuration to maximizing labels at the relabeled configuration. Thus the preference matrix is S -equivariant.

Apply Corollary 1 with the single measure λ and with the preference matrix defined above. We obtain a configuration $\xi \in \mathcal { C } _ { 2 r - 1 , }$ such that $\lambda ( V _ { 1 } ( \xi ) ) = \cdot \cdot \cdot = \lambda ( V _ { r } ( \xi ) ) = 1 / r$ , and $\xi \in A _ { i } ^ { j } = A _ { j }$ for every $j \in \{ 1 , \ldots , r \}$ . Thus every label corresponds to a largest entry of $( F _ { 1 } ( \xi ) , \ldots , F _ { r } ( \xi ) )$ , and therefore $F _ { 1 } ( \xi ) =$ $\cdots = F _ { r } ( \xi )$ □

## 4 Exact Cut Complexity for Proportionality

We prove $\kappa _ { n } ^ { \mathrm { p r o p } } = 2 n - 2$ for every n. After an elementary lower bound, we write $n = r s$ with $r \mathrm { ~ a ~ }$ prime power, partition the agents into r balanced groups, and recurse within the corresponding top-level bundles. The lower-bound instance is analogous to one used in the market-value setting [7].

Proposition 1. For every $n \geq 1$ , we have $\kappa _ { n } ^ { \mathrm { p r o p } } \geq 2 n - 2$

Proof. Fix $n \geq 1$ . The case $n = 1$ is trivial. We may therefore assume $n \geq 2$ . Consider the profile in which every agent i has the same valuation $v _ { i } ( C ^ { \prime } ) = n \cdot \lambda ( C ^ { \prime } \cap [ 0 , 1 / n ] )$ for $C ^ { \prime } \subseteq [ 0 , 1 ]$ . Let $( C _ { 1 } , \ldots , C _ { n } )$ be any equal-length proportional allocation. Since every agent values only the prefix $[ 0 , 1 / n ]$ , each bundle $C _ { i }$ must satisfy $\bar { \lambda } ( C _ { i } \cap [ 0 , 1 / n ] ) \geq 1 / n ^ { 2 }$ . Summing over i shows that equality must hold for every i, namely $\lambda \left( C _ { i } \cap [ 0 , 1 / n ] \right) = 1 / n ^ { 2 }$ and $\lambda \left( C _ { i } \cap \left( 1 / n , 1 \right] \right) = 1 / n - 1 / n ^ { 2 }$ . Therefore every bundle has positive length inside $[ 0 , 1 / n ]$ and positive length inside $( 1 / n , 1 ]$

Let T be the set of distinct interior cut points used by the allocation. The points of $T \cap ( 0 , 1 / n )$ split $[ 0 , 1 / n ]$ into $| T \cap ( 0 , 1 / n ) | + 1$ positive-length intervals, each contained in a single induced piece and therefore assigned to a single bundle. Since every one of the n bundles has positive length in $[ 0 , 1 / n ]$ , these intervals must serve all n bundles, and hence $| T \cap ( 0 , 1 / n ) | + 1 \geq n$ . Thus at least $n - 1$ cuts lie in $( 0 , 1 / n )$ . This argument does not depend on whether $1 / n$ itself is a cut point. If an induced piece crosses $1 / n$ , its left part still belongs to only the one bundle containing that induced piece.

The same reasoning applied to $( 1 / n , 1 ]$ gives $| T \cap ( 1 / n , 1 ) | \geq n - 1$ . The two open intervals are disjoint. Hence every such allocation uses at least $2 n - 2$ distinct interior cuts. □

## 4.1 The Transfer Principle

The first step rounds a fractional assignment with balanced row and column sums, using Hall’s marriage theorem, into an assignment of exactly s agents to each of the r labels.

Lemma 3. Let $M = \left( M _ { i j } \right)$ be an $r \times n$ nonnegative matrix such that every row sums to 1 and every column sums to $1 / s$ , where $n = r s$ and s is a positive integer. Then there exists a map π : $\mathbf { \partial } : \{ 1 , \dots , n \} \to \{ 1 , \dots , r \}$ with the following properties, where $\pi ^ { - 1 } ( i )$ denotes the set of elements mapped to i.

$\textit { 1 . } | \pi ^ { - 1 } ( i ) | = s$ for every $i \in \{ 1 , \ldots , r \}$ , and

2. $M _ { \pi ( j ) , j } > 0$ for every $j \in \{ 1 , \ldots , n \}$

Proof. Consider the bipartite graph with left vertices $\{ 1 , \ldots , r \} \times \{ 1 , \ldots , s \}$ and right vertices $\{ 1 , \ldots , n \}$ where $( i , \ell )$ is adjacent to $j$ if and only if $M _ { i j } > 0$ . We first show that this graph has a perfect matching. The two sides both have $n = r s$ vertices. Hence a matching covering the right side is perfect, and by Hall’s theorem it is enough to show that every subset $S \subseteq \{ 1 , \dots , n \}$ of right vertices has at least |S| neighbors.

Let $T \subseteq \{ 1 , \ldots , r \}$ be the set of row indices that appear among the neighbors of S. Then all positive entries in columns indexed by S lie in rows indexed by T. Summing the column sums over S gives

$$
{ \frac { | S | } { s } } = \sum _ { j \in S } \sum _ { i = 1 } ^ { r } M _ { i j } = \sum _ { j \in S } \sum _ { i \in T } M _ { i j } \leq \sum _ { i \in T } \sum _ { j = 1 } ^ { n } M _ { i j } = | T | .
$$

Therefore, the neighborhood of S in $\{ 1 , \ldots , r \} \times \{ 1 , \ldots , s \}$ has size $s | T | \geq | S |$ . Hall’s theorem gives a perfect matching.

If column j is matched to $( i , \ell )$ , define $\pi ( j ) : = i$ . Because there are exactly s copies of each row index $i ,$ we have $| \pi ^ { - 1 } ( i ) | = s$ . Because edges occur only where $M _ { i j } > 0$ , we also have $M _ { \pi ( j ) , j } > 0$ for every $j . \ \sqsubseteq$

We use the following fractional assignments in the transfer. The entry $M _ { i j } ^ { \varepsilon } ( \xi )$ is the weight that agent j places on label $i ;$ positive weight is restricted to bundles worth more than $\bar { 1 / r } - \varepsilon$ . The slack makes the support condition continuous and will vanish in a compactness argument.

Definition 2. Let $n = r s$ , where s is a positive integer, and $\mathit { f i x } \varepsilon > 0$ . An ε-admissible fractional assignment scheme for an n-agent profile is a continuous map $M ^ { \varepsilon } \colon \mathcal { C } _ { 2 r - 1 , r } \to \mathbb { R } _ { \geq 0 } ^ { r \times n }$ with the following properties.

1. For every configuration $\xi$ and every agent $j \in \{ 1 , \ldots , n \}$ , the column sum satisfies $\begin{array} { r } { \sum _ { i = 1 } ^ { r } M _ { i j } ^ { \varepsilon } ( \xi ) = 1 / s } \end{array}$

2. The vector $\begin{array} { r } { F ^ { \varepsilon } ( \xi ) : = \left( \sum _ { j = 1 } ^ { n } M _ { 1 j } ^ { \varepsilon } ( \xi ) , \dots , \sum _ { j = 1 } ^ { n } M _ { r j } ^ { \varepsilon } ( \xi ) \right) } \end{array}$ satisfies $F _ { \sigma ( i ) } ^ { \varepsilon } ( \sigma \cdot \xi ) = F _ { i } ^ { \varepsilon } ( \xi )$ for every $\sigma \in S _ { r }$ and $i \in \{ 1 , \ldots , r \}$

3. Whenever $M _ { i j } ^ { \varepsilon } ( \xi ) > 0$ , we have $v _ { j } ( V _ { i } ( \xi ) ) > 1 / r - \varepsilon$

The transfer lemma first equalizes the row sums and then applies Hall’s theorem to the positive support. A compactness argument removes the slack ε while keeping the discrete assignment fixed along a subsequence.

Lemma 4. Let $n = r s$ , where $r \geq 2$ is a prime power and s is a positive integer, and let $v _ { 1 } , \ldots , v _ { n }$ be an n-agent profile. Suppose that for every $\varepsilon > 0 \ t h i s$ profile admits an ε-admissible fractional assignment scheme. Then this profile admits a partition $o f \ [ 0 , 1 ]$ into r bundles $C _ { 1 } , \ldots , C _ { r }$ with at most $2 r - 2$ cuts together with a map $\pi \colon \{ 1 , \ldots , n \} \to \{ 1 , \ldots , r \}$ such that

$$
1 . \ \lambda \left( C _ { 1 } \right) = \cdot \cdot \cdot = \lambda \left( C _ { r } \right) = 1 / r ,
$$

2. $| \pi ^ { - 1 } ( i ) | = s$ for all $i \in \{ 1 , \ldots , r \}$ , and

3. $v _ { j } ( C _ { \pi ( j ) } ) \geq 1 / r$ for every agent $j \in \{ 1 , \ldots , n \}$

Proof. Fix $\varepsilon > 0$ , and choose an ε-admissible fractional assignment scheme $M ^ { \varepsilon }$ . Let $F ^ { \varepsilon }$ be its vector of total assignment weights. Since $M ^ { \varepsilon }$ is continuous, $F ^ { \varepsilon }$ is continuous as well. By Lemma 2, there exists $\xi _ { \varepsilon } \in \mathcal { C } _ { 2 r - 1 , r }$ such that $\lambda \left( V _ { 1 } ( \xi _ { \varepsilon } ) \right) = \cdot \cdot \cdot = \lambda \left( V _ { r } ( \xi _ { \varepsilon } ) \right) = 1 / r$ and $F _ { 1 } ^ { \varepsilon } ( \xi _ { \varepsilon } ) = \cdot \cdot \cdot = F _ { r } ^ { \varepsilon } ( \xi _ { \varepsilon } )$

The sum of all entries of $M ^ { \varepsilon } ( \xi _ { \varepsilon } )$ is $n / s = r$ . Since all r row sums are equal, each equals 1.

Applying Lemma 3 to $M ^ { \varepsilon } ( \xi _ { \varepsilon } )$ , we obtain a map $\pi _ { \varepsilon } \colon \{ 1 , \ldots , n \} \to \{ 1 , \ldots , r \}$ such that each label has exactly s preimages and $M _ { \pi _ { \varepsilon } ( j ) , j } ^ { \varepsilon } ( \xi _ { \varepsilon } ) > 0 ( j \in \{ 1 , \ldots , n \} )$ . By admissibility, $v _ { j } ( V _ { \pi _ { \varepsilon } ( j ) } ( \xi _ { \varepsilon } ) ) > 1 / r - \varepsilon$

Let Π be the finite set of maps $\pi \colon \{ 1 , \ldots , n \} \to \{ 1 , \ldots , r \}$ that assign s agents to each label. Choose a sequence $\varepsilon _ { k } > 0$ with $\varepsilon _ { k } \to 0$ , apply the preceding argument to each $\varepsilon _ { k } .$ , and denote the resulting pair by $( \xi _ { k } , \pi _ { k } )$ . Since Π is finite, pass to a subsequence with $\pi _ { k } = \pi$ for all k. Since $\mathcal { C } _ { 2 r - 1 , r }$ is a compact metric space, pass to a further subsequence with $\xi _ { k } \to \xi$

Set $C _ { i } : = V _ { i } ( \xi )$ These sets partition [0, 1] up to boundary points, which we assign according to the half-open convention of Section 2. Nonatomicity ensures that this choice does not afect any length or value. By Lemma 1, $\xi \mapsto v _ { j } ( V _ { \pi ( j ) } ( \xi ) )$ is continuous for each $j .$ . Since $v _ { j } ( V _ { \pi ( j ) } ( \xi _ { k } ) ) > 1 / r - \varepsilon _ { k }$ and $\varepsilon _ { k } \to 0$ , we obtain $v _ { j } ( C _ { \pi ( j ) } ) \geq 1 / r$ . The same continuity lemma applied to λ gives $\lambda ( C _ { i } ) = 1 / r \ ( i \in \{ 1 , \ldots , r \} )$ . Since $\xi \in \mathcal { C } _ { 2 r - 1 , r } .$ it has at most $2 r - 2$ formal interior cut positions. Repeated positions create only degenerate intervals. Thus, the partition uses at most $2 r - 2$ distinct cuts. □

For proportionality, an admissible scheme is obtained by normalizing each agent’s positive excesses above $1 / r - \varepsilon$ . This assigns positive weight only to suficiently valuable bundles.

Lemma 5. Let $n = r s$ , where $r \geq 2$ is a prime power and s is a positive integer, and let $v _ { 1 } , \ldots , v _ { n }$ be an n-agent profile. Then for every $\varepsilon > 0$ , this profile admits an ε-admissible fractional assignment scheme.

Proof. Fix $\varepsilon > 0$ . For each configuration $\xi \in \mathcal { C } _ { 2 r - 1 , r }$ , bundle index $i \in \{ 1 , \ldots , r \}$ , and agent $j \in \{ 1 , \ldots , n \}$ define $g _ { i j } ( \xi ) : = \operatorname* { m a x } \{ 0 , \ v _ { j } ( V _ { i } ( \xi ) ) - ( 1 / r - \varepsilon ) \}$ . The maps $\xi \mapsto v _ { j } ( V _ { i } ( \xi ) )$ are continuous on $\mathcal { C } _ { 2 r - 1 , r }$ by Lemma 1. Hence each $g _ { i j }$ is continuous. For every fixed $j ,$ the bundle values sum to 1. Hence at least one bundle has value at least $1 / r _ { : }$ and therefore $\begin{array} { r } { d _ { j } ( \xi ) : = \sum _ { k = 1 } ^ { r } g _ { k j } ( \xi ) } \end{array}$ is continuous and strictly positive for every ξ.

Now define $M _ { i j } ^ { \varepsilon } ( \xi ) : = ( r / n ) g _ { i j } ( \xi ) / d _ { j } ( \xi )$

This is a well-defined continuous nonnegative map. For each $j \in \{ 1 , \ldots , n \}$ we have $\begin{array} { r } { \sum _ { i = 1 } ^ { r } M _ { i j } ^ { \varepsilon } ( \xi ) = r / n = } \end{array}$ $1 / s$ . If $M _ { i i } ^ { \varepsilon } ( \xi ) > 0$ , then $g _ { i j } ( \xi ) > 0$ , hence $v _ { j } ( V _ { i } ( \xi ) ) > 1 / r - \varepsilon$ . Finally, relabeling sends row i of $M ^ { \varepsilon } ( \xi )$ to row $\sigma ( i )$ of $M ^ { \varepsilon } ( \sigma \cdot \xi )$ because $g _ { \sigma ( i ) , j } ( \sigma \cdot \xi ) = g _ { i j } ( \xi )$ , and therefore $d _ { j } ( { \boldsymbol { \sigma } } \cdot { \boldsymbol { \xi } } ) = d _ { j } ( { \boldsymbol { \xi } } )$ and $M _ { \sigma ( i ) , j } ^ { \varepsilon } ( \sigma \cdot \xi ) = M _ { i j } ^ { \varepsilon } ( \xi )$ Summing over j gives $F _ { \sigma ( i ) } ^ { \varepsilon } ( \sigma \cdot \xi ) = F _ { i } ^ { \varepsilon } ( \xi )$ □

Combining Lemmas 4 and 5 gives r equal-length top-level bundles with at most $2 r - 2$ cuts, exactly s assigned agents per bundle, and value at least $1 / r$ for every assigned agent.

## 4.2 Recurrence and Induction

Each top-level bundle may be disconnected. Concatenating its components lets us solve a normalized problem on [0, 1] and pull the solution back without adding cuts at the component boundaries.

Let $C \subseteq [ 0 , 1 ]$ be a positive-length finite union of intervals whose interior component boundaries have already been counted as cuts. Concatenate its components from left to right and rescale by $\lambda \left( C \right)$ to obtain $\phi \colon C \to [ 0 , 1 ]$ . For every finite measure ν with $\nu ( C ) > 0$ , put ${ \widehat { \nu } } ( Y ) = \nu ( \phi ^ { - 1 } ( Y ) ) / \nu ( C )$ ; a measure vanishing on C imposes no constraint on the subproblem.

Lemma 6. If the normalized subproblem has a solution using at most k cuts, then the pulled-back solution uses at most k new cuts in the original cake inside C. Lengths are multiplied by $\lambda \left( C \right)$ , and values for each measure ν with $\nu ( C ) > 0$ are multiplied by $\nu ( C )$ . Thus equal-length conditions, value inequalities, envy comparisons, and equality constraints in the normalized solution become the corresponding conditions inside C.

Proof. Each normalized cut pulls back either to one new point inside a component of $C$ or to an already counted component boundary. Hence, at most k new cuts are created. Moreover, we have

$$
\lambda ( \phi ^ { - 1 } ( Y ) ) = \lambda \left( C \right) \cdot \lambda ( Y ) \quad { \mathrm { a n d } } \quad \nu ( \phi ^ { - 1 } ( Y ) ) = \nu ( C ) \cdot { \widehat { \nu } } ( Y ) .
$$

The two positive scale factors give all claimed transfers. Endpoint choices are immaterial by nonatomicity.

Applying the top-level partition and assignment together with this lemma gives the recurrence.

Lemma 7. Let $n = r s$ , where $r \geq 2$ is a prime power and s is a positive integer. Then $\kappa _ { n } ^ { \mathrm { p r o p } } \leq ( 2 r - 2 ) +$ $r \kappa _ { s } ^ { \mathrm { p r o p } }$

Proof. Fix an arbitrary n-agent profile. By Lemma 5, it has an admissible scheme for every $\varepsilon > 0$ . Applying Lemma 4, we obtain bundles $C _ { 1 } , \ldots , C _ { r }$ of length 1/r and agent groups $S _ { i } : = \pi ^ { - 1 } ( i ) ~ ( i \in \{ 1 , . . . , r \} )$ , each of size s, such that every $j \in S _ { i }$ satisfies $v _ { j } ( C _ { i } ) \geq 1 / r$

Fix $i \in \{ 1 , \ldots , r \}$ . Restrict and normalize each $v _ { j }$ for $j \in S _ { i }$ on $C _ { i } ;$ this is valid because $v _ { j } ( C _ { i } ) \geq 1 / r > 0$ By the definition of $\kappa _ { s } ^ { \mathrm { p r o p } }$ and Lemma 6, the normalized instance has a pulled-back solution using at most $\kappa _ { s } ^ { \mathrm { p r o p } }$ new cuts, with bundle length $1 / n$ . Each assigned agent $j \in S _ { i }$ receives value at least $v _ { j } ( C _ { i } ) / s \geq$ $( 1 / r ) / s = 1 / n$

Applying this independently inside each bundle gives an equal-length proportional allocation of the whole cake. The top-level partition contributes at most $2 r - 2$ cuts, and the recursive cuts add across the r disjoint bundles. Hence the total number of cuts is at most $( 2 r - 2 ) + r \kappa _ { s } ^ { \mathrm { p r o p } }$ □

Theorem 4. For every $n \geq 1$ , we have $\kappa _ { n } ^ { \mathrm { p r o p } } = 2 n - 2$

Proof. We prove the upper bound $\kappa _ { n } ^ { \mathrm { p r o p } } \leq 2 n - 2$ by strong induction on $n .$

The case $n = 1$ is trivial. Now let $n \geq 2$ , and assume for every $t < n$ that $\kappa _ { t } ^ { \mathrm { p r o p } } \leq 2 t - 2$

Choose any prime-power divisor $r > 1$ of n, and write $n = r s$ . By Lemma 7 and the induction hypothesis applied to $s < n ,$ we obtain $\kappa _ { n } ^ { \mathrm { p r o p } } \leq ( 2 r - 2 ) + r \kappa _ { s } ^ { \mathrm { p r o p } } \leq ( 2 r - 2 ) + r ( 2 s - 2 ) = 2 r s - 2 = 2 n - 2$ . This proves $\kappa _ { n } ^ { \mathrm { p r o p } } \leq 2 n - 2$ for every n. Together with Proposition 1, this gives $\kappa _ { n } ^ { \mathrm { p r o p } } = 2 n - 2$ □

## 5 Cut Complexity for Envy-Freeness

Since envy-freeness implies proportionality, Proposition 1 gives $\kappa _ { n } ^ { \mathrm { e f } } \geq 2 n - 2$ . We match this bound when n is a prime power and prove $\kappa _ { n } ^ { \mathrm { e f } } = O ( n ^ { 1 . 5 \bar { 2 } 5 } )$ in general. For the general upper bound, we repeatedly apply the Stromquist–Woodall exact-share theorem (Theorem 1), treating valuations assigned to the other side of a recursive split as auxiliary measures. These measures control envy across the split.

## 5.1 Prime Powers

For fixed m and valuations $v _ { 1 } , \ldots , v _ { r } $ , define a preference matrix on $\mathcal { C } _ { m , r }$ <sub>,r</sub> by

$$
\begin{array} { r } { A _ { i } ^ { j } : = \bigl \{ \xi \in \mathcal { C } _ { m , r } : v _ { j } ( V _ { i } ( \xi ) ) \geq \operatorname* { m a x } _ { i ^ { \prime } \in \{ 1 , \dots , r \} } v _ { j } ( V _ { i ^ { \prime } } ( \xi ) ) \bigr \} . } \end{array}
$$

The matrix is closed by Lemma 1, equivariant under relabeling, and proper because every agent has a maximizing bundle. Hence Corollary 1 applies.

Proposition 2. $I f r \geq 2$ is a prime power, then $\kappa _ { r } ^ { \mathrm { e f } } = 2 r - 2$

Proof. Apply Corollary 1 with $t = 1$ , measure $\mu _ { 1 } = \lambda$ , and the preference matrix above. The resulting bundles have length $1 / r _ { ; }$ and agent j weakly prefers its assigned bundle $V _ { j }$ to every $V _ { i }$ . Thus the allocation is envy-free and uses at most $2 r - 2$ cuts. Conversely, envy-freeness implies proportionality. Thus, Theorem 4 gives $\kappa _ { r } ^ { \mathrm { e f } } \geq \kappa _ { r } ^ { \mathrm { p r o p } } = 2 r - 2$ □

## 5.2 The Exact-Share Recurrence

For n agents and h additional nonatomic probability measures $\nu _ { 1 } , \ldots , \nu _ { h }$ , let $\kappa _ { n , h } ^ { \mathrm { e f } }$ be the least number of cuts that always sufices for an equal-length envy-free allocation in which each $\nu _ { \ell }$ assigns the same value to all n bundles. The additional measures are auxiliary: they impose equality constraints but receive no bundles. Thus $\kappa _ { n } ^ { \mathrm { e f } } = \kappa _ { n , 0 } ^ { \mathrm { e f } } ,$ and $\kappa _ { 1 , h } ^ { \mathrm { e f } } = 0$ because the unique bundle is the whole cake.

Exact-share splitting yields the following recurrence. We apply Theorem 1 to length, the auxiliary measures, and all agent valuations, obtaining a region to which every measure assigns the prescribed share.

Theorem 5. For every $n \geq 2$ , every $h \geq 0$ , and every $a \in \{ 1 , \dots , n - 1 \}$ ，

$$
\kappa _ { n , h } ^ { \mathrm { e f } } \leq 2 ( n + h ) + \kappa _ { a , h + n - a } ^ { \mathrm { e f } } + \kappa _ { n - a , h + a } ^ { \mathrm { e f } } .
$$

Proof. Let the agent valuations be $\mu _ { 1 } , \ldots , \mu _ { n }$ , let the auxiliary measures be $\nu _ { 1 } , \ldots , \nu _ { h }$ , and let λ denote length. Apply the Stromquist–Woodall theorem, Theorem 1, to the $m = n + h + 1$ probability measures $\lambda , \nu _ { 1 } , \ldots , \nu _ { h } , \mu _ { 1 } , \ldots , \mu _ { n }$ with target ratio $\alpha = a / n$ . We obtain a set $A \subseteq [ 0 , 1 ]$ using at most $2 m - 2 = 2 ( n + h )$ cuts such that $\lambda \left( A \right) = a / n , \nu _ { j } ( A ) = a / n$ for every j, and $\mu _ { i } ( A ) = a / n$ for every i. Let $B : = [ 0 , 1 ] \backslash A$ . Then $\lambda \left( B \right) = ( n - a ) / n , \nu _ { j } ( B ) = ( n - a ) / n$ for every j, and $\mu _ { i } ( B ) = ( n - a ) / n$ for every i.

The component boundaries of A and B are already cut by the initial exact-share split. Thus Lemma 6 applies to both sides, and only the recursive cuts themselves add new cuts in the original cake.

Choose any $S \subseteq \{ 1 , \dots , n \}$ with $| S | = a$ . On A, the agents are $\{ \mu _ { i } : i \in S \}$ , and the auxiliary measures are $\nu _ { 1 } , \ldots , \nu _ { h }$ together with $\{ \mu _ { i } : i \notin S \}$ . After restriction, normalization, and pullback, this subproblem costs at most $\kappa _ { a , h + n - a } ^ { \mathrm { e f } }$ new cuts. Every resulting bundle has length $1 / n$ , and each auxiliary measure—including every $\mu _ { i }$ with $i \not \in S \cdot$ assigns it value $1 / n$

On B, the agents are $\{ \mu _ { i } : i \notin S \}$ , and the auxiliary measures are $\nu _ { 1 } , \ldots , \nu _ { h }$ together with $\{ \mu _ { i } : i \in S \}$ This subproblem costs at most $\kappa _ { n - a , h + a } ^ { \mathrm { e f } }$ new cuts and has the analogous length and value properties.

To verify global envy-freeness, fix ${ \bf \chi } _ { i } \in { \bf \chi } _ { S }$ and let $X _ { i } ~ \subseteq ~ A$ be the bundle assigned to i. Local envyfreeness implies $\mu _ { i } ( X _ { i } ) \geq \mu _ { i } ( A ) / a = 1 / n$ . Since $\mu _ { i }$ is auxiliary on B, every bundle $Y \subseteq B$ has value $\mu _ { i } ( Y ) = 1 / n$ . Thus agent i envies neither a bundle in A nor one in B. The case $i \not \in S$ is symmetric, proving the recurrence. □

## 5.3 Peeling and a Subquadratic Upper Bound

We iterate the exact-share recurrence with $a = 1$ , peeling of one bundle at a time until only a prime-power core remains.

Theorem 6. Let $n \geq 2$ , let $2 \leq r \leq n$ be a prime power, and define $t : = n - r$ . Then, we have $\kappa _ { n } ^ { \mathrm { e f } } \ \leq$ $2 n t + ( r - 1 ) ( t + 2 ) \leq 3 n t + 2 n$

Proof. Using $\kappa _ { 1 , m } ^ { \mathrm { e f } } = 0$ for all $m \geq 0$ , taking $a = 1$ in Theorem 5 gives

$$
\kappa _ { n , h } ^ { \mathrm { e f } } \leq 2 ( n + h ) + \kappa _ { n - 1 , h + 1 } ^ { \mathrm { e f } } .
$$

Starting from $\kappa _ { r + t , 0 } ^ { \mathrm { e f } } ,$ we apply this recurrence repeatedly while decreasing the number of agents by one and increasing the number of auxiliary measures by one. Since $r + t = n .$ , for each $k = 0 , \ldots , t - 1$ we obtain

$$
\kappa _ { n - k , k } ^ { \mathrm { e f } } = \kappa _ { r + t - k , k } ^ { \mathrm { e f } } \leq 2 ( r + t ) + \kappa _ { r + t - k - 1 , k + 1 } ^ { \mathrm { e f } } = 2 n + \kappa _ { n - k - 1 , k + 1 } ^ { \mathrm { e f } } ,
$$

since the sum of the two indices stays equal to n throughout. Summing these t inequalities gives

$$
\kappa _ { n } ^ { \mathrm { e f } } \leq 2 n t + \kappa _ { r , t } ^ { \mathrm { e f } } .
$$

To bound the remaining term, fix any prime-power core instance with r agent valuations and t auxiliary measures $\nu _ { 1 } , \ldots , \nu _ { t }$ . Apply Corollary 1 with the $t + 1$ measures $\lambda , \nu _ { 1 } , \ldots , \nu _ { t }$ , and the preference family induced by the r agent valuations. The length measure makes every core bundle have length $1 / r .$ , each auxiliary measure assigns the same value to all core bundles, and the maximizing preferences make the diagonal assignment envy-free. Since the number of measures is $t + 1$ , the cut bound from Corollary 1 is $( r - 1 ) ( t + 2 )$ . Hence

$$
\kappa _ { r , t } ^ { \mathrm { e f } } \leq ( r - 1 ) ( t + 2 ) .
$$

This proves the first displayed bound. Since $r = n - t ,$ we also obtain

$$
\kappa _ { n } ^ { \mathrm { e f } } \leq 2 n t + ( n - t - 1 ) ( t + 2 ) = 3 n t - t ^ { 2 } - 3 t + 2 n - 2 ,
$$

and dropping the negative terms gives $\kappa _ { n } ^ { \mathrm { e f } } \leq 3 n t + 2 n$

To obtain a subquadratic bound, choose a prime power $r \leq n$ with a small gap $t = n - r$ . Baker, Harman, and Pintz [6] show that $[ x - x ^ { 0 . 5 2 5 } , x ]$ contains a prime for all suficiently large x.

Proposition 3. For all suficiently large $n ,$ we have $\kappa _ { n } ^ { \mathrm { e f } } \leq 3 n ^ { 1 . 5 2 5 } + 2 n$ . In particular, $\kappa _ { n } ^ { \mathrm { e f } } = O ( n ^ { 1 . 5 2 5 } ) =$ $o ( n ^ { 2 } )$

Proof. By Baker, Harman, and Pintz [6], for all suficiently large n the interval $[ n - n ^ { 0 . 5 2 5 } , n ]$ contains a prime r. Thus r is a prime power and $\dot { t } = n - r \le n ^ { 0 . 5 2 5 }$ . Substituting into Theorem 6 yields

$$
\kappa _ { n } ^ { \mathrm { e f } } \leq 3 n \cdot n ^ { 0 . 5 2 5 } + 2 n = 3 n ^ { 1 . 5 2 5 } + 2 n = O ( n ^ { 1 . 5 2 5 } ) .
$$

Remark 7. The proof uses a number-theoretic result only to bound the gap $t = n - r$ . Accordingly, any stronger bound on the distance from n to a prime power improves the all-n bound from Theorem 6. Concretely, if every suficiently large n admits a prime power $r \in [ n - \Delta ( n ) , n ]$ , then $t = n - r \leq \Delta ( n )$ and Theorem 6 yields $\kappa _ { n } ^ { \mathrm { e f } } \leq 3 n \Delta ( n ) + 2 n$ . Under the Riemann hypothesis, Cram´er’s classical prime-gap estimate gives $\Delta ( n ) = O ( { \sqrt { n } } \log n )$ and therefore $\kappa _ { n } ^ { \mathrm { e f } } = O ( n ^ { 3 / 2 } \log n ) \ [ 1 2 , 1 5 ]$ . The scale $( \log n ) ^ { \frac { 1 } { 2 } }$ is a classical conjectural benchmark for prime gaps, originating in Cram´er’s probabilistic model [12]. Under a Cram´er-type hypothesis giving $\Delta ( n ) = O ( ( \log n ) ^ { 2 } )$ , we would obtain $\kappa _ { n } ^ { \mathrm { e f } } = O \bigl ( n ( \log n ) ^ { 2 } \bigr )$

## 6 Query Complexity in the Robertson–Webb Model

In this section, we show that no deterministic protocol in the Robertson–Webb model terminates after finitely many queries on every instance and always returns an exact equal-length proportional allocation. We then show that proportionality can be approximated to arbitrary accuracy with finitely many queries while maintaining exact lengths.

Since length is public, the protocol may place cuts at specified coordinates and perform exact arithmetic on coordinates without querying an agent. Its only access to a private valuation $v _ { i }$ is through $\operatorname { E V A L } _ { i } ( x )$ which returns $v _ { i } ( [ 0 , x ] )$ , and $\operatorname { C U T } _ { i } ( r )$ , which returns the leftmost $x$ satisfying $v _ { i } ( [ 0 , x ] ) = r$ . These queries also give access to arbitrary intervals. The value of $[ a , b ]$ is obtained by subtracting $\mathrm { E V A L } _ { i } ( a )$ from $\operatorname { E V A L } _ { i } ( b )$ To find the leftmost $b \geq$ a satisfying $v _ { i } ( [ a , b ] ) = r$ , the protocol first obtains $v _ { i } ( [ 0 , a ] )$ and then makes the query $\mathrm { C U T } _ { i } \big ( v _ { i } ( [ 0 , a ] ) + r \big )$ . Only queries for which the requested cut exists are made. Protocols output finite unions of intervals whose endpoints are either returned by Cut queries or specified by their coordinates.

## 6.1 Finite-Query Impossibility

For two agents with the same valuation $v ,$ an equal-length allocation has the form $( A , [ 0 , 1 ] \setminus A )$ with $\lambda \left( A \right) = 1 / 2$ . It is proportional if and only if

$$
v ( A ) = v ( [ 0 , 1 ] \setminus A ) = \frac { 1 } { 2 } .
$$

Thus, the problem is to halve simultaneously the known length measure $\lambda$ and an unknown measure v. Exact division of two unknown measures cannot in general be accomplished by a finite algorithm [26, Thm. 2.2]. We adapt their decision-tree argument to show that the impossibility remains when length is known.

Theorem 8. There is no deterministic protocol in the Robertson–Webb model that terminates after finitely many queries on every instance and always returns an exact equal-length proportional allocation. This remains true for two agents with the same nonatomic valuation.

Proof. Suppose that such a protocol exists, and fix it for two agents. We answer its queries adversarially while constructing a nonatomic valuation $v ,$ which will be shared by the two agents. Begin with $\mathcal { P } ^ { 0 } = \{ [ 0 , 1 ] \}$ and assign value 1 to its sole member. After the $j \mathrm { t h }$ query, let $\mathcal { P } ^ { j }$ be the partition obtained from $\mathcal { P } ^ { j - 1 }$ by adding every new endpoint introduced by that query. At every stage, each member of $\mathcal { P } ^ { j }$ has positive value, and no $u \subseteq { \mathcal { P } } ^ { j }$ satisfies $\lambda ( \bigcup _ { U \in \mathcal { U } } U ) = v ( \bigcup _ { U \in \mathcal { U } } U ) = 1 / 2$ . Both properties hold initially.

It sufices to show that both properties are preserved whenever a new endpoint is added to the current partition $\mathcal { P }$ . Suppose that the new endpoint splits a subinterval $I \in \mathcal { P }$ into two subintervals $I ^ { \prime }$ and $I ^ { \prime \prime }$ of positive length, producing ${ \mathcal { P } } ^ { \prime }$ . If the protocol specifies the splitting point, the lengths of $I ^ { \prime }$ and $I ^ { \prime \prime }$ are determined, but the adversary can choose how the value of I is divided between them. If the splitting point is the answer to a Cut query, the query determines how the value is divided, but the adversary can choose the lengths by selecting the point. Write $d = v ( I ) / \lambda ( I )$ for the average density on $I .$ In the first case, the adversary chooses $v ( I ^ { \prime } ) \in ( 0 , v ( I ) )$ and sets $v ( I ^ { \prime \prime } ) = v ( I ) - v ( I ^ { \prime } )$ . In the second case, it chooses $\lambda ( I ^ { \prime } ) \in ( 0 , \lambda ( I ) )$ and sets $\lambda ( I ^ { \prime \prime } ) = \lambda ( I ) - \lambda ( I ^ { \prime } )$ . Fix $\mathcal { U } \subseteq \mathcal { P } ^ { \prime }$ . If $\mathcal { U }$ contains both $I ^ { \prime }$ and $I ^ { \prime \prime } .$ , replacing them by I gives a subcollection of $\mathcal { P }$ whose union has the same length and value. If U contains neither, then $\mathcal { U } \subseteq \mathcal { P }$ . By the second property for ${ \mathcal { P } } _ { : }$ , neither case yields both length and value $1 / 2$ . It remains to consider a subcollection $\mathcal { U }$ containing exactly one of $I ^ { \prime }$ and $I ^ { \prime \prime }$ . If the protocol specified the splitting point, then $\lambda ( \bigcup _ { U \in \mathcal { U } } U )$ is fixed. If this length is not $1 / 2 .$ , the union cannot have both length and value $1 / 2$ for any choice of $v ( I ^ { \prime } )$ . If it is $1 / 2$ , the equation $v ( \bigcup _ { U \in \mathcal { U } } U ) = 1 / 2$ excludes at most one choice of $v ( I ^ { \prime } )$ . If the splitting point answers a Cut query, the same argument applies with length and value interchanged. Thus, only finitely many values of the quantity chosen by the adversary are excluded. We choose this quantity close to the value that would give both new subintervals the same density d as $I .$ At the hth time a new endpoint is added to the partition, we choose $v ( I ^ { \prime } )$ close to $d \lambda ( I ^ { \prime } )$ in the first case and $\lambda ( I ^ { \prime } )$ close to $v ( I ^ { \prime } ) / d$ in the second. We choose it close enough that the density of each new subinterval lies between $2 ^ { - 2 ^ { - h } } d$ and $2 ^ { 2 ^ { - h } } d .$ The target value lies strictly inside the relevant interval, and only finitely many choices are excluded. Thus, such a choice exists. It preserves both properties.

This rule determines the adversary’s answers. For an $\operatorname { E V A L } _ { i } ( x )$ query, split at x if necessary and return the total value of the subintervals to its left. For a $\operatorname { C U T } _ { i } ( q )$ query, return an existing endpoint if the cumulative value there is $q .$ . Otherwise, q lies strictly between the cumulative values at the endpoints of a unique subinterval because every member of the partition has positive value. The query fixes the value of the left part of this subinterval, and the adversary chooses its length, and hence the returned point, by the rule above.

If the protocol announces an output, add its endpoints to the current partition one at a time using the same refinement rule. The values assigned during this construction are consistent because a refinement divides the value of one subinterval without changing the value of any earlier subinterval. If only finitely many new endpoints are added, assign a constant density to each member of the eventual partition. If infinitely many are added, let $f _ { h }$ be the density that is constant on each member of the partition after the hth new endpoint has been added and realizes its assigned value. When the hth new endpoint is added, the density of each new subinterval difers from the density on the subinterval that was split by a factor between $2 ^ { - 2 ^ { - h } }$ and $2 ^ { 2 ^ { - h } }$ . Since $\begin{array} { r } { \sum _ { h > 1 } 2 ^ { - h } = 1 } \end{array}$ , every $f _ { h }$ takes values between $1 / 2$ and 2. For $m > h$ , the ratio $f _ { m } / f _ { h }$ lies uniformly between $2 ^ { - } \textstyle \sum _ { j > h } 2 ^ { - j }$ and $2 \sum _ { j > h } 2 ^ { - j }$ . Hence, $\left( f _ { h } \right)$ is uniformly Cauchy and converges to a density $f$ with the same bounds. Thus, in either case the construction defines a single nonatomic valuation v consistent with every answer given to the protocol. Its positive density also makes every answer to a Cut query the unique leftmost solution.

Because the protocol is deterministic, its run on v reproduces the adversarial answer sequence. If this sequence is infinite, the protocol does not terminate on $v ,$ contradicting the assumption. Otherwise, the protocol produces an output after finitely many queries, and its bundles are unions of members of the partition obtained after adding their endpoints. The second property shows that neither bundle can have both length and value $1 / 2$ . Hence, the output is not an equal-length proportional allocation, a contradiction. □

## 6.2 Approximation with Finitely Many Queries

Although exact computation is impossible, proportionality can be approximated with finitely many queries. For $\varepsilon > 0$ , call an equal-length allocation ε-proportional if $v _ { i } ( C _ { i } ) \geq 1 / n - \varepsilon$ for every agent $i .$

Theorem 9. For every $\varepsilon > 0$ , there is a protocol in the Robertson–Webb model that computes an equal-length ε-proportional allocation using $O ( n ^ { 2 } / \varepsilon )$ queries and at most $2 n - 2$ cuts.

Proof. The case $n = 1$ is trivial. We may therefore assume $n \geq 2 . { \mathrm { ~ I f ~ } } \varepsilon \geq 1 / n .$ , any partition into n intervals of length $1 / n$ sufices without queries. We may therefore assume $0 < \varepsilon < 1 / n$ and put $q = 2 n - 2$ . Set $B = \lceil 2 q / \varepsilon \rceil$ . For each agent i and each $r = 1 , \ldots , B - 1$ , ask $\operatorname { C U T } _ { i } ( r / B )$ . Together with 0 and 1, the returned points partition the cake into $B$ intervals, each of value $1 / B$ to agent i. We call this partition agent i’s grid. Let $G = \{ I _ { 1 } , \ldots , I _ { s } \}$ be the interval partition whose endpoints are all the points in the n agents’ grids, with the intervals listed from left to right. Then $s \leq n ( B - 1 ) + 1 \leq n B$ . Every $I _ { r }$ is contained in one interval of each agent’s grid and therefore has value at most $1 / B$ to every agent.

Enumerate all combinatorial types of allocations with q cuts. Such a type consists of indices $1 \leq r _ { 1 } \leq$ $\cdots \leq r _ { q } \leq s$ and an agent label for each of the $q + 1$ intervals induced by the cuts. It represents allocations whose jth cut lies in the closure of $I _ { r _ { j } }$ . At a grid endpoint, the same placement may therefore be represented by either adjacent interval. Equal consecutive indices are allowed because two cuts may lie in the same interval of $G ,$ and the interval between them may have positive length. Allowing coincident cuts also includes allocations with fewer than $q$ cuts. Once a type is fixed, the cut positions are the only variables. The conditions that the jth cut lies in the closure of $I _ { r _ { i } }$ , that the cuts remain ordered, and that every labeled bundle has length exactly $1 / n$ form a finite system of linear equalities and inequalities.

For any feasible placement $x _ { 1 } \leq \dots \leq x _ { q }$ of a fixed type, let $C _ { i }$ be the union of the intervals labeled i. For each cut $x _ { j }$ in the closure of $I _ { r _ { j } } ,$ agent $i \mathrm { \ ' } _ { \mathrm { S } }$ grid determines an integer $a _ { i j }$ such that $a _ { i j } / B \leq v _ { i } ( [ 0 , x _ { j } ] ) \leq$ $( a _ { i j } + 1 ) / B$ . Let $P _ { h }$ be the piece between $x _ { h }$ and $x _ { h + 1 }$ under the convention of Section $2 ,$ , where $x _ { 0 } = 0$ and $x _ { q + 1 } = 1$ . The value of $C _ { i }$ is the sum of $v _ { i } ( P _ { h } ) = v _ { i } ( [ 0 , x _ { h + 1 } ] ) - v _ { i } ( [ 0 , x _ { h } ] )$ over the pieces $P _ { h }$ labeled i. Replace each $v _ { i } ( [ 0 , x _ { j } ] )$ in this sum by $a _ { i j } / B$ , and denote the result by $\widehat { v } _ { i }$ . This number is determined by the type and the queried grids. Each cut contributes to the error at most once. Hence,

$$
\left| v _ { i } ( C _ { i } ) - { \widehat { v } } _ { i } \right| \leq { \frac { q } { B } } .\tag{1}
$$

By Theorem $^ { 4 , }$ there is an exact equal-length proportional allocation with at most q cuts. Its combinatorial type occurs in the enumeration and, by (1), satisfies $\widehat { v } _ { i } \geq 1 / n - q / B$ for every agent i. Select any such type and output any feasible placement of its cuts. Applying (1) once more gives

$$
v _ { i } ( C _ { i } ) \geq { \frac { 1 } { n } } - { \frac { 2 q } { B } } \geq { \frac { 1 } { n } } - \varepsilon \qquad ( i = 1 , \ldots , n ) .
$$

The protocol uses $n ( B - 1 ) = O ( n ^ { 2 } / \varepsilon )$ queries and at most $q = 2 n - 2$ cuts.

The theorem concerns query complexity only, since enumerating the combinatorial types may take exponential time. Allowing more cuts gives the following polynomial-time alternative.

Proposition 4. For every $\varepsilon > 0$ , an equal-length ε-proportional allocation can be computed in polynomial time in the real-RAM oracle model using $O ( n ^ { 3 } / \varepsilon )$ Robertson–Webb queries and $O ( n ^ { 2 } / \varepsilon )$ cuts.

Proof. The case $n = 1$ is immediate. We may therefore assume $n \geq 2 .$ . If $\varepsilon \geq 1 / n ,$ any partition into n intervals of length $1 / n$ sufices without queries. We may therefore assume $0 < \varepsilon < 1 / n$ . Construct B and $G = \{ I _ { 1 } , \ldots , I _ { s } \}$ as in the proof of Theorem 9, and query every agent’s value for every $I _ { r }$ . This uses $O ( n s ) = O ( n ^ { 3 } / \varepsilon )$ queries. Introduce a nonnegative variable $y _ { i r }$ for the fraction of the length of $I _ { r }$ assigned to agent i. Impose $\textstyle \sum _ { i } y _ { i r } = 1$ for every $\begin{array} { r } { r , \sum _ { r } \lambda \left( I _ { r } \right) y _ { i r } = 1 / n } \end{array}$ for every $i ,$ and $\sum _ { r } v _ { i } ( I _ { r } ) y _ { i r } \geq 1 / n$ for every i. The assignment $y _ { i r } = 1 / n$ is feasible, and a basic feasible solution has at most $s + 2 n$ positive variables. Since every $I _ { r }$ has at least one positive variable, there are at most 2n positive variables beyond the first one for each interval. Subdividing an interval requires one fewer cut than the number of its positive variables. Thus, the total number of additional cuts is at most 2n. Dividing each such interval into subintervals of the prescribed lengths decreases each agent’s value from its fractional value by at most $2 n / B \le \varepsilon$ , because $v _ { i } ( I _ { r } ) \leq 1 / B$ . The allocation uses at most $s - 1 + 2 n = O ( n ^ { 2 } / \varepsilon )$ cuts. □

## 7 Two-Knife Augmentation

We augment the Robertson–Webb model with two-knife queries.

We use the following two additional types of queries and count them separately from Robertson–Webb queries. We assume that the two knives can be moved continuously and monotonically while maintaining the specified length or value. A MoveLength query specifies a fixed length, initial and final knife positions, a value threshold for each of a set of agents, and an integer k. The two knives move continuously and monotonically while the portion of the current subcake between them keeps the specified length. Each agent raises a hand whenever that portion is worth at least the agent’s threshold. If fewer than k hands are raised initially and at least k are raised finally, the query stops at the first time when at least k hands are raised and returns the two knife positions. A MoveValue<sub>i</sub> query specifies a value $\alpha ,$ a target length $\ell ,$ initial positions of both knives at which the intervening portion is worth $\alpha ,$ and a final position for one of the knives. Agent i moves the specified knife continuously and monotonically toward its final position and moves the other knife continuously and monotonically with it so that the intervening portion remains worth $\alpha .$ . If ℓ lies between the initial length and the length when the specified knife reaches its final position, the query stops when the intervening portion first has length ℓ and returns the two knife positions. If a motion or answer is not unique, any one may be used, and the procedure must be correct for every choice. Knife positions and query endpoints count as cuts only when they form boundaries of the final allocation.

Queries may also be applied to a disconnected subcake $C = J _ { 1 } \cup \cdot \cdot \cdot \cup J _ { q }$ of length $L _ { : }$ , where the intervals are pairwise disjoint and listed from left to right. Removing the gaps and concatenating the components gives a copy of C on [0, L]. For $0 \leq x \leq y \leq L$ , let $C \langle x , y \rangle$ be the portion corresponding to $( x , y ]$ in this copy, with the left endpoint included when $x = 0$ , as in the convention of Section 2. This portion has length $y - x$ and at most $q$ components in the original cake. Its value can be computed using $O ( q )$ Eval queries, and given one endpoint and a target value, the other endpoint can be found with $O ( q )$ Eval queries and at most one Cut query. Crossing a gap leaves the corresponding portion unchanged. Thus, a continuous monotone motion on $[ 0 , L ]$ also gives one on the original cake.

## 7.1 Balanced Halving

We show that one MoveLength query splits an even subinstance into two equal subinstances while preserving the proportional guarantee of every agent. Applying this split recursively attains the optimal cut bound when the number of agents is a power of two.

Theorem 10. For every nonatomic valuation profile, if n is a power of two, an equal-length proportional allocation can be produced using at most $n - 1$ two-knife queries and at most $2 n - 2$ cuts.

Proof. We first describe how to split an instance $( C , S )$ with $m = 2 r$ agents into two equal subinstances using one MoveLength query. Write $L = \lambda ( C )$ and, for $0 \leq t \leq L / 2$ , let $A _ { t } = C \langle t , t + L / 2 \rangle$ and $B _ { t } = C \setminus A _ { t }$ Thus, $A _ { t }$ and $B _ { t }$ both have length $L / 2$ . The two knives move from the two boundaries of $A _ { 0 }$ to those of $A _ { L / 2 }$ , and agent i raises a hand when $v _ { i } ( A _ { t } ) \geq v _ { i } ( C ) / 2$ . The values $v _ { i } ( C )$ , and hence the thresholds, can be obtained beforehand by evaluating the components of C. These Robertson–Webb queries are not included in the two-knife query count of the theorem.

If at least r agents weakly prefer each of $A _ { 0 }$ and $B _ { 0 }$ , assign to $A _ { 0 }$ every agent who strictly prefers it, add indiferent agents until the group has size $r ,$ and assign the remaining agents to $B _ { 0 }$ . Otherwise, orient the motion so that fewer than r hands are raised initially. The portions between and outside the knives exchange roles at the end of the motion, where more than r hands are therefore raised. The query stops at the first $t ^ { * }$ at which at least r hands are raised. By continuity, fewer than r agents strictly prefer $A _ { t ^ { * } }$ to $B _ { t ^ { * } } ;$ otherwise, at least r hands would have been raised immediately before $t ^ { * }$ . Assign to $A _ { t } .$ ∗ any r agents with raised hands, including all agents who strictly prefer it, and assign the remaining agents to $B _ { t ^ { * } }$ . The first group values $A _ { t ^ { * } }$ at least $v _ { i } ( C ) / 2$ by the stopping rule, while every agent in the second group weakly prefers $B _ { t }$ ∗ and therefore values it at least $v _ { i } ( C ) / 2$ . This split uses one two-knife query and introduces at most two cut points.

We now recurse on instances $( C , S )$ satisfying $\lambda ( C ) = m / n$ and $v _ { i } ( C ) \geq m / n$ for every $i \in S$ , where $m = | S |$ . Initially $C = [ 0 , 1 ]$ , S is the full set of agents, and $m = n$ . If $m = 1$ , assign $C$ to its only agent. Otherwise m is even, and the split above produces two instances with $m / 2$ agents. Every agent assigned to a child values it at least $v _ { i } ( C ) / 2 \geq m / ( 2 n )$ , and each child has length $m / ( 2 n )$ . Thus, the invariant is preserved.

The recursion is a full binary tree with n leaves and hence $n - 1$ internal nodes. Each internal node uses one two-knife query and introduces at most two cut points. The resulting bounds are therefore at most $n - 1$ two-knife queries and $2 n - 2$ cuts. Together with Proposition 1, this also proves that the cut bound is optimal. □

## 7.2 General Numbers of Agents

In this subsection, we show how to produce an equal-length proportional allocation for arbitrary n. The procedure uses $O ( n ^ { 5 } )$ Robertson–Webb queries, $O ( n ^ { 2 } )$ two-knife queries, and $O ( n \log n )$ cuts. The procedure recursively assigns bundles to a nonempty group of agents and continues with the remaining agents and cake. At each stage, one agent divides the current subcake into one equal-length bundle per remaining agent, with all bundles having the same value to that agent. A matching argument then assigns a nonempty collection of these bundles to the same number of agents, giving every assigned agent value at least $1 / n$ . The lemma below constructs the partition used in this matching argument. In its application, the equal-value condition makes the chosen agent adjacent to every bundle. The bound on new cut points yields the total bound of O(n log n) cuts. The query bounds follow by summing the cost of the construction over the recursive stages.

Lemma 8. Fix an agent i, and let C be a finite union of intervals satisfying $\lambda \left( C \right) = k \ell$ and $v _ { i } ( C ) =$ kα for an integer $k \geq 1$ and some $\ell , \alpha > 0$ . One can partition C into bundles $T _ { 1 } , \dots , T _ { k }$ satisfying $\lambda \left( T _ { j } \right) = \ell$ and $v _ { i } ( T _ { j } ) = \alpha ~ f o r$ every j. At most $4 \lceil \log _ { 2 } k \rceil$ new cut points are introduced in obtaining any one bundle.

Proof. We first show how to extract from a current subcake D a subpiece of specified length and value. Suppose that D is a union of $q _ { D }$ intervals, $\lambda \left( D \right) = t \ell ^ { \prime }$ , and $v _ { i } ( D ) = t \alpha ^ { \prime }$ for an integer $t \geq 2$ . Divide the copy of D constructed above into t consecutive blocks of length $\ell ^ { \prime }$ and evaluate them using ${ \cal O } ( t q _ { D } )$ queries.

Their values have average $\alpha ^ { \prime } .$ . If one block has value $\alpha ^ { \prime }$ , it is the desired subpiece. Otherwise, there are two consecutive blocks $D \langle a , b \rangle$ and $D \langle b , c \rangle$ whose values lie on opposite sides of $\alpha ^ { \prime }$ . Suppose first that $v _ { i } ( D \langle a , b \rangle ) <$ $\alpha ^ { \prime } < v _ { i } ( D \langle b , c \rangle )$ . Place the left knife at a and the right knife at a position $y _ { 0 }$ such that $v _ { i } ( D \langle a , y _ { 0 } \rangle ) = \alpha ^ { \prime }$ . The first inequality implies $y _ { 0 } > b$ . For every $x \in [ a , b ]$ , we have $v _ { i } ( D \langle x , c \rangle ) \geq v _ { i } ( D \langle b , c \rangle ) > \alpha ^ { \prime }$ . Thus, a position of the right knife preserving the value $\alpha ^ { \prime }$ exists throughout the motion. Move the left knife from a to b while agent i moves the right knife continuously and monotonically to keep the intervening value equal $\mathrm { t o } \alpha ^ { \prime }$ . Let $y _ { 1 }$ be the position of the right knife when the left knife reaches b. The second inequality implies $y _ { 1 } < c .$ . Hence, the length of the intervening portion starts above $\ell ^ { \prime }$ and ends below $\ell ^ { \prime }$ . If $v _ { i } ( D \langle a , b \rangle ) > \alpha ^ { \prime } > v _ { i } ( D \langle b , c \rangle )$ place the right knife at b and the left knife at a position $x _ { 0 }$ such that $v _ { i } ( D \langle x _ { 0 } , b \rangle ) = \alpha ^ { \prime }$ . The first inequality implies $x _ { 0 } \ > \ a$ . For every $y \in [ b , c ]$ , we have $v _ { i } ( D \langle a , y \rangle ) \geq v _ { i } ( D \langle a , b \rangle ) > \alpha ^ { \prime } .$ Thus, a position of the left knife preserving the value $\alpha ^ { \prime }$ exists throughout the motion. Move the right knife from b to c while agent i moves the left knife continuously and monotonically to keep the intervening value equal to $\alpha ^ { \prime }$ . Let $x _ { 1 }$ be the position of the left knife when the right knife reaches c. The second inequality implies $x _ { 1 } < b$ . Hence, the length starts below $\ell ^ { \prime }$ and ends above $\ell ^ { \prime }$ . In either case, continuity implies that a MoveValue<sub>i</sub> query returns a subpiece of length $\ell ^ { \prime }$ and value $\alpha ^ { \prime }$ . Thus, the extraction uses ${ \cal O } ( t q _ { D } )$ Robertson–Webb queries and at most one two-knife query.

For a current subcake D, let h be the integer satisfying $\lambda \left( D \right) = h \ell$ and $v _ { i } ( D ) = h \alpha$ . If $h = 1$ , then D is a final bundle. If h is even, use the extraction above with $t = 2$ , target length $h \ell / 2$ , and target value $h \alpha / 2$ The returned subpiece and its complement both have length $h \ell / 2$ and value $h \alpha / 2$

If $h \geq 3$ is odd, first use the extraction with $t = h$ to obtain one final bundle of length ℓ and value α. The residual subcake has length $( h - 1 ) \ell$ and value $( h - 1 ) \alpha$ . Apply the even step to split it into two subcakes, each of length $( h - 1 ) \ell / 2$ and value $( h - 1 ) \alpha / 2$ . Continue recursively until $h = 1$ . Every step replaces an integer $h > 1$ by smaller positive integers whose sum is h. Thus, the procedure terminates with the required k bundles. If h is even, it is halved after one split. If h is odd, it is reduced to $( h - 1 ) / 2$ after two splits. Thus, obtaining any one bundle requires at most $2 \lceil \log _ { 2 } k \rceil$ splits. Each split introduces at most two new cut points, proving the last assertion. □

The construction also gives the complexity bounds used below. Exactly $k - 1$ splits are made, each using at most one MoveValue query. Consequently, if $C$ is a union of q intervals, each bundle is a union of $O ( q + \log k )$ intervals.

Using this partition at each stage gives the following result.

Theorem 11. For every nonatomic valuation profile, there is a procedure in the Robertson–Webb model augmented with two-knife queries that produces an equal-length proportional allocation. It uses $O ( n ^ { 5 } )$ Robertson– Webb queries and $O ( n ^ { 2 } )$ two-knife queries. The final allocation has $O ( n \log n )$ cuts.

Proof. We recurse on subinstances $( C , S )$ , where S is a set of m agents and C is a finite union of intervals, under the invariant $\lambda ( C ) = m / n$ and $v _ { i } ( C ) \geq m / n$ for every $i \in S$ . The initial instance satisfies this invariant.

Choose an agent $a \in S$ and use Eval queries on the components of C to obtain $V = v _ { a } ( C )$ . Apply Lemma 8 with $k = m , \ell = 1 / n$ , and $\alpha = V / m$ to partition C into bundles $T _ { 1 } , \ldots , T _ { m }$ satisfying $\lambda ( T _ { j } ) = 1 / n$ and $v _ { a } ( T _ { j } ) = V / m$ for every j. Since $V \geq m / n$ , agent a values every $T _ { j }$ at least $1 / n$

Form a bipartite graph between S and $\mathcal { T } = \{ T _ { 1 } , \ldots , T _ { m } \}$ , joining i to $T _ { j }$ when $v _ { i } ( T _ { j } ) \geq 1 / n$ . Because the bundles partition C and $v _ { i } ( C ) \geq m / n$ , every agent has a neighbor. Agent a is adjacent to every bundle. The graph is obtained by evaluating every bundle for every agent.

Choose a nonempty $\mathcal { T } ^ { \prime } \subseteq \mathcal { T }$ inclusion-minimal subject to $| N ( \mathscr { T } ^ { \prime } ) | \leq | \mathscr { T } ^ { \prime } |$ , and put $S ^ { \prime } = N ( \mathcal { T } ^ { \prime } )$ . Such a set exists because $\tau$ itself qualifies, and agent a belongs to $S ^ { \prime }$ . We first show that $| S ^ { \prime } | = | T ^ { \prime } |$ . If $\vert S ^ { \prime } \vert < \vert T ^ { \prime } \vert$ then $| \mathcal { T } ^ { \prime } | \geq 2$ because $a \in S ^ { \prime }$ , and for any $T _ { j } \in \mathcal { T } ^ { \prime }$ we have $| N ( { \cal T } ^ { \prime } \backslash \{ T _ { j } \} ) | \le | S ^ { \prime } | \le | { \cal T } ^ { \prime } | - 1$ , contradicting the minimality of $\mathcal { T } ^ { \prime }$ . Moreover, every nonempty proper subset $U \subset \mathcal { T } ^ { \prime }$ satisfies $| N ( U ) | > | U |$ by minimality. Together with $| N ( \mathscr { T } ^ { \prime } ) | = | \mathscr { T } ^ { \prime } |$ , this is Hall’s condition for $G [ S ^ { \prime } , \mathcal { T } ^ { \prime } ]$ . Thus, the graph has a perfect matching. By the definition of $S ^ { \prime }$ , no agent outside $S ^ { \prime }$ is adjacent to a bundle in $\mathcal { T } ^ { \prime }$ . Assign the bundles in $\mathcal { T } ^ { \prime }$ to the agents in $S ^ { \prime }$ according to that perfect matching, and recurse on $S \setminus S ^ { \prime }$ and $C \setminus \textstyle \bigcup _ { T _ { i } \in T ^ { \prime } } T _ { j }$ . Every remaining agent values each removed bundle below $1 / n$ . Since $\vert S ^ { \prime } \vert = \vert \mathcal { T } ^ { \prime } \vert$ , every remaining agent values the residual subcake at least $( m - | S ^ { \prime } | ) / n$ . Thus, the invariant is preserved. Because $S ^ { \prime }$ is nonempty, the recursion terminates with a valid allocation.

There are at most n recursive stages. At a stage with m agents, the construction in the proof of Lemma 8 performs $m - 1$ splits. Consequently, it uses at most $m - 1$ two-knife queries and creates $O ( m )$ temporary boundary points. Over all stages, the procedure uses $O ( n ^ { 2 } )$ two-knife queries and creates $O ( n ^ { 2 } )$ temporary boundary points. Every subcake encountered therefore has ${ \dot { O } } ( n ^ { 2 } )$ components. At one stage, the construction performs at most $n - 1$ extractions. Each extraction considers at most n consecutive blocks of a subcake with $O ( n ^ { 2 } )$ components and therefore uses $O ( n ^ { 3 } )$ Robertson–Webb queries, as shown in the proof of Lemma 8. Thus, constructing the partition at one stage uses $O ( n ^ { 4 } )$ Robertson–Webb queries. Evaluating its at most n bundles for at most n agents also uses $O ( n ^ { 4 } )$ queries, because each bundle has $O ( n ^ { 2 } )$ components. The total over at most n stages is therefore $O ( n ^ { 5 } )$ Robertson–Webb queries.

At each stage, keep only the boundaries created while obtaining the bundles assigned at that stage. Any other boundary separates only bundles left for later stages and is internal to the residual subcake. Thus, it can be discarded. Boundaries kept from earlier stages have already been charged to bundles assigned at those stages. By Lemma 8, each assigned bundle requires $O ( \log n )$ new cut points, and exactly n bundles are assigned over the whole recursion. Hence, the final allocation uses $O ( n \log n )$ cuts. □

## 8 Conclusion

We proved that equal-length proportionality has exact cut complexity $2 n - 2$ for every n. For the stronger equal-length envy-free problem, the same bound is optimal when n is a prime power, and the general upper bound is ${ \bar { O } } ( n ^ { 1 . 5 \dot { 2 } 5 } )$ ).

No deterministic protocol in the Robertson–Webb model terminates after finitely many queries on every instance and always returns an exact equal-length proportional allocation. For every $\varepsilon > 0$ , however, an equal-length ε-proportional allocation with at most $2 n - 2$ cuts can be computed using $O ( n ^ { 2 } / \varepsilon )$ queries. A polynomial-time alternative in the real-RAM oracle model uses $O ( n ^ { 3 } / \varepsilon )$ queries and $O ( n ^ { 2 } / \varepsilon )$ cuts. In the Robertson–Webb model augmented with two-knife queries, the optimal cut bound is attained for powers of two under arbitrary nonatomic valuations. For general n and arbitrary nonatomic valuations, the augmented model produces an equal-length proportional allocation using $O ( n ^ { 5 } )$ Robertson–Webb queries, $O ( n ^ { 2 } )$ twoknife queries, and $O ( n \log n )$ cuts. Two questions remain open: whether equal-length envy-freeness always admits $2 n - 2$ cuts, and whether a protocol in this augmented model attains $2 n - 2$ cuts for every n.

## Acknowledgement

This work was partially supported by JST ERATO Grant Number JPMJER2301, JST LOTUS Programme Grant Number JPMJ25152681, JSPS KAKENHI Grant Number JP25K00137, and ANRF MATRICS Grant Number MTR/2021/000474.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

No data was used for the research described in the article.

## Declaration of AI assistance

The authors used OpenAI’s ChatGPT and Codex to assist with language editing, manuscript organization, and the presentation of proofs and related work. The authors reviewed and verified all mathematical content and citations. The authors assume responsibility for all content.

## References

[1] N. Alon. Splitting necklaces. Advances in Mathematics, 63(3):247–253, 1987. doi: 10.1016/ 0001-8708(87)90055-7.

[2] G. Amanatidis, H. Aziz, G. Birmpas, A. Filos-Ratsikas, B. Li, H. Moulin, A. A. Voudouris, and X. Wu. Fair division of indivisible goods: Recent progress and open questions. Artificial Intelligence, 322: 103965, 2023. doi: 10.1016/j.artint.2023.103965.

[3] S. Arndt, B. Moseley, S. Im, and K. Pruhs. No, cake cutting really is a piece of cake, 2026. arXiv:2606.07238.

[4] A. K. Austin. Sharing a cake. The Mathematical Gazette, 66(437):212–215, 1982. doi: 10.2307/3616548.

[5] H. Aziz and S. Mackenzie. A discrete and bounded envy-free cake cutting protocol for any number of agents. In 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS), pages 416–427, 2016. doi: 10.1109/FOCS.2016.52.

[6] R. C. Baker, G. Harman, and J. Pintz. The diference between consecutive primes, II. Proceedings of the London Mathematical Society, 83(3):532–562, 2001. doi: 10.1112/plms/83.3.532.

[7] S. Barman, S. Ebadian, M. Latifian, and N. Shah. Fair division with market values. Proceedings of the AAAI Conference on Artificial Intelligence, 39(13):13589–13596, 2025. doi: 10.1609/aaai.v39i13.33484.

[8] X. Bei, N. Chen, X. Hua, B. Tao, and E. Yang. Optimal proportional cake cutting with connected pieces. In Proceedings of the Twenty-Sixth AAAI Conference on Artificial Intelligence (AAAI 2012), pages 1263–1269, 2012. doi: 10.1609/aaai.v26i1.8243.

[9] A. Biswas and S. Barman. Fair division under cardinality constraints. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI 2018), pages 91–97, 2018. doi: 10.24963/ijcai.2018/13.

[10] S. Bouveret, Y. Chevaleyre, and N. Maudet. Fair allocation of indivisible goods. In F. Brandt, V. Conitzer, U. Endriss, J. Lang, and A. D. Procaccia, editors, Handbook of Computational Social Choice, pages 284–310. Cambridge University Press, 2016. doi: 10.1017/CBO9781107446984.013.

[11] X. Bu, Z. Li, S. Liu, J. Song, and B. Tao. Fair division with allocator’s preference. In Proceedings of the 19th Conference on Web and Internet Economics (WINE 2023), pages 77–94, 2023. doi: 10.1007/ 978-3-031-48974-7 5.

[12] H. Cram´er. On the order of magnitude of the diference between consecutive prime numbers. Acta Arithmetica, 2:23–46, 1936.

[13] T. Crew, B. Narayanan, and S. Spirkl. Disproportionate division. Bulletin of the London Mathematical Society, 52(5):885–890, 2020. doi: 10.1112/blms.12368.

[14] A. Cseh and T. Fleiner. The complexity of cake cutting with unequal shares. <sup>´</sup> ACM Transactions on Algorithms, 16(3):1–21, 2020. doi: 10.1145/3380742.

[15] A. W. Dudek. On the Riemann hypothesis and the diference between primes. International Journal of Number Theory, 11(3):771–778, 2015. doi: 10.1142/S1793042115500426.

[16] J. Edmonds and K. Pruhs. Cake cutting really is not a piece of cake. ACM Transactions on Algorithms, 7(4):51:1–51:12, 2011. doi: 10.1145/2000807.2000819.

[17] P. Goldberg and J. Li. Consensus Division in an Arbitrary Ratio. In 14th Innovations in Theoretical Computer Science Conference (ITCS 2023), pages 57:1–57:18, 2023. doi: 10.4230/LIPIcs.ITCS.2023.57.

[18] P. G¨olz and H. Yaghoubizade. Fair division among couples and small groups. Proceedings of the AAAI Conference on Artificial Intelligence, 40(20):16963–16970, 2026. doi: 10.1609/aaai.v40i20.38743.

[19] D. Joji´c, G. Panina, and R. T. Zivaljevi´c. Splitting necklaces, with constraints. <sup>ˇ</sup> SIAM Journal on Discrete Mathematics, 35(2):1268–1286, 2021. doi: 10.1137/20M1331949.

[20] Y. Kawase and R. Mahara. Fair and eficient balanced allocation for indivisible goods. Proceedings of the AAAI Conference on Artificial Intelligence, 40(20):17067–17075, 2026. doi: 10.1609/aaai.v40i20.38755.

[21] Y. Kawase and M. A. Sanpui. Exact cut complexity of equal-length proportional cake cutting. In 51st International Symposium on Mathematical Foundations of Computer Science (MFCS 2026), volume 386 of Leibniz International Proceedings in Informatics (LIPIcs), pages 34:1–34:16. Schloss Dagstuhl– Leibniz-Zentrum f¨ur Informatik, 2026. doi: 10.4230/LIPIcs.MFCS.2026.34.

[22] Y. Kawase, B. Roy, and M. A. Sanpui. Simultaneously Fair Allocation of Indivisible Items Across Multiple Dimensions. In 45th IARCS Annual Conference on Foundations of Software Technology and Theoretical Computer Science (FSTTCS 2025), pages 41:1–41:19, 2025. doi: 10.4230/LIPIcs.FSTTCS. 2025.41.

[23] M. D. la Tour. Bad news for couples: Tight lower bounds for fair division of indivisible items. arXiv preprint arXiv:2601.01012, 2026. doi: 10.48550/arXiv.2601.01012.

[24] O. Pikhurko. On envy-free cake division. The American Mathematical Monthly, 107(8):736–738, 2000. doi: 10.2307/2695471.

[25] A. D. Procaccia and J. Wang. A lower bound for equitable cake cutting. In Proceedings of the 2017 ACM Conference on Economics and Computation, pages 479–495, 2017. doi: 10.1145/3033274.3085107.

[26] J. M. Robertson and W. A. Webb. Near exact and envy free cake division. Ars Combinatoria, 45: 97–108, 1997.

[27] E. Segal-Halevi. Cake-cutting with diferent entitlements: How many cuts are needed? Journal of Mathematical Analysis and Applications, 480(1):123382, 2019. doi: 10.1016/j.jmaa.2019.123382.

[28] H. Steinhaus. Sur la division pragmatique. Econometrica, 17:315–319, 1949. doi: 10.2307/1907319.

[29] W. Stromquist and D. R. Woodall. Sets on which several measures agree. Journal of Mathematical Analysis and Applications, 108(1):241–248, 1985. doi: 10.1016/0022-247X(85)90021-6.

[30] F. E. Su. Rental harmony: Sperner’s lemma in fair division. The American Mathematical Monthly, 106 (10):930–942, 1999. doi: 10.2307/2589747.