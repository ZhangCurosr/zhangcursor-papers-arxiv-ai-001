# Dec-BFTRL: Squre-Root Regret for Decentralized Online Upper-Linearizable Optimization under Separation Access with Application to Continuous Submodular Maximization

Yiyang Lu

yiyanglu@purdue.edu

Purdue University, West Lafayette, IN, USA

Mohammad Pedramfar

mohammad.pedramfar@mila.quebec

Mila - Quebec AI Institute/McGill University, Montreal, QC, Canada

Vaneet Aggarwal

Purdue University, West Lafayette, IN, USA

vaneet@purdue.edu

## Abstract

We study decentralized online optimization of upper-linearizable payofs over an action set under eficient separation access, with applications to online continuous diminishing-return (DR) submodular maximization. We propose Decentralized Barrier Follow-the-Regularized-Leader (Dec-BFTRL), and evaluate each agent’s played action against the average of all local objectives. Each agent maps an internal iterate to a feasible action through an approximate gauge projection, communicates only a cumulative surrogate-gradient dual state, and invokes the local HybridNewton procedure to approximately minimize its post-communication BFTRL potential. For every agent, we achieve expected network-aggregate regret of $\widetilde { O } ( \sqrt { T } )$ Over T rounds, each agent uses T neighbor-mixing steps and $\widetilde O ( T )$ separation-oracle calls. We give four wrapper instantiations covering three DR-submodular maximization problems.

## 1 Introduction

The demand for real-time decision-making in large-scale systems, such as autonomous robotic swarms and networked sensor arrays, has driven the rapid development of decentralized online optimization (Li et al., 2002; Xiao et al., 2007; Mokhtari et al., 2018). In these distributed frameworks, individual agents must collaboratively process continuous data streams and adapt to evolving environments without relying on a central coordinating server. A fundamental theoretical and practical problem arises when the time-varying global objectives are highly non-convex. Fortunately, a wide array of critical network applications, including dynamic pricing, recommendation algorithms, inventory routing, mean-field variational inference, and power grid reconfiguration (Bian et al., 2019; Aldrighetti et al., 2021; Ito and Fujimaki, 2016; Hassani et al., 2017; Mitra et al., 2021; Gu et al., 2023; Mishra et al., 2017), exhibit structural regularities, most notably the economic property of diminishing returns (DR). In these scenarios, the marginal benefit of an agent’s action naturally decreases as the global system approaches optimality, making cooperative online learning essential. Although much of the literature models such objectives through continuous DR-submodularity or weak up-concavity, upper-linearizability provides a broader abstraction for designing unified algorithms across non-concave function classes.

Problem Setting and Metric. We consider N agents connected by a fixed, connected, undirected communication network. The agents share a compact convex action set K, which is accessed through a separation oracle and contains a known interior point. Before the game begins, an oblivious adversary fixes local payof functions $\{ f _ { t , j } \in \mathcal { F } : t \in [ T ] , j \in [ N ] \}$ , where j indexes the owner of a local payof. At round t, each agent a selects an action $x _ { a , t } \in \mathcal { K }$ , receives feedback about its locally owned payof $f _ { t , a }$ (which is not necessarily incurred at $x _ { a , t } )$ , and exchanges information once with its neighbors before the next round.

Table 1: Summary of Decentralized Online Continuous DR-Submodular Maximization (D-OCSM) Algorithms. Let $\underline { { x } } \in \mathcal { K }$ denote the fixed feasible anchor used by Corollary 3, with $\| \underline { { x } } \| _ { \infty } < 1$ . ‘(Smooth)’ means additional smoothness condition on the objective functions. Oracle-call counts follow the access model of each cited result, and SO and LOO calls should not be treated as equal-cost operations. The notation $\widetilde { \cal O } ( T )$ suppresses logarithmic dependence on T and the set-conditioning parameters.
<table><tr><td>Setting/Function Class</td><td>Reference/Algorithm</td><td>Approx. Ratio (α)</td><td>logT(α-Regret)</td><td>logT(Comm.)</td><td>Oracle Type</td><td>Total Oracle Calls</td></tr><tr><td rowspan="9">Monotone,  $0 \in \mathcal { K } ^ { \mathrm { ~ 1 ~ } }$ </td><td>DMFW (Zhu et al., 2021)</td><td> $1 - e ^ { - 1 }$ </td><td>1/2</td><td> $5 / 2$ </td><td>LOO</td><td> ${ \cal O } ( T ^ { 5 / 2 } )$ </td></tr><tr><td>Mono-DMFW (Zhang et al., 2023)</td><td> $1 - e ^ { - 1 }$ </td><td>4/5</td><td>1</td><td>LOO</td><td>O(T)</td></tr><tr><td>DOBGA (Zhang et al., 2023)</td><td> $1 - e ^ { - 1 }$ </td><td>1/2</td><td>1</td><td>Projection</td><td></td></tr><tr><td>DPOBGA (Liao et al., 2023)</td><td> $1 - e ^ { - 1 }$ </td><td>3/4</td><td> $1 / 2$ </td><td>LOO</td><td>O(T)</td></tr><tr><td>DROCULO  $( \theta = 1 ) \ \mathrm { ( L u \ e t \ a l . , 2 0 2 5 ) }$ </td><td> $1 - e ^ { - 1 }$ </td><td>1/2</td><td> $^ 1$ </td><td>LOO</td><td>O(T2)</td></tr><tr><td>DROCULO (θ = 1/2) (Lu et al., 2025)</td><td> $1 - e ^ { - 1 }$ </td><td>3/4</td><td> $1 / 2$ </td><td>LOO</td><td>O(T)</td></tr><tr><td>Corollary 2 (this work)</td><td> $1 - e ^ { - 1 }$ </td><td>1/2</td><td>1</td><td>SO</td><td>ō(T)</td></tr><tr><td>DROCULO (θ = 1) (Lu et al., 2025)</td><td> $\scriptstyle { \frac { 1 - \| \mathbf { x } \| _ { \infty } } { a } }$ </td><td>1/2</td><td>1</td><td>LOO</td><td>O(T2)</td></tr><tr><td>DROCULO (θ = 1/2) Lu et al. (2025)</td><td> $\frac { 1 - \| \mathbf { x } \| _ { \infty } } { \mathbf { \epsilon } }$   $\frac { 1 - \| \mathbf { x } \| _ { \infty } } { \mathbf { \sigma } }$ </td><td>3/4</td><td> $1 / 2$ </td><td>LOO</td><td>O(T)</td></tr><tr><td rowspan="3"></td><td>Wan et al. (2026)</td><td></td><td>3/4</td><td>1</td><td>LOO</td><td>O(T)</td></tr><tr><td>Wan et al. (2026) (Smooth)</td><td> $\frac { 1 - \| \mathbf { x } \| _ { \infty } } { A }$ </td><td>2/3</td><td>1</td><td>LOO</td><td>O(T)</td></tr><tr><td>Corollary 3 (this work)</td><td> $\frac { 1 - \| \mathbf { \underline { { x } } } \| _ { \infty } } { 4 }$ </td><td>1/2</td><td>1</td><td>SO</td><td>Ö(T)</td></tr><tr><td rowspan="3">Non-monotone, Down-closed K</td><td>Wan et al. (2026)</td><td> $1 / e$ </td><td>3/4</td><td>1</td><td>LOO</td><td>O(T)</td></tr><tr><td>Wan et al. (2026) (Smooth)</td><td>1/e</td><td>2/3</td><td>1</td><td>LOO</td><td>O(T)</td></tr><tr><td>Corollary 4 (this work)</td><td>1/e</td><td>1/2</td><td>1</td><td>SO</td><td>Ö(T)</td></tr></table>

Define the network-aggregate payof $\begin{array} { r } { \mathsf { F } _ { t } ( x ) : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } f _ { t , j } ( x ) } \end{array}$ . Although each agent receives only local feedback, its performance is measured using the aggregate payof. For $\alpha \in ( 0 , 1 ]$ , the per-agent network-aggregate expected α-regret is

$$
\mathfrak { R } _ { \alpha } ^ { a } ( T ) : = \mathbb { E } \left[ \alpha \operatorname* { m a x } _ { x \in K } \sum _ { t = 1 } ^ { T } \mathsf { F } _ { t } ( x ) - \sum _ { t = 1 } ^ { T } \mathsf { F } _ { t } ( x _ { a , t } ) \right] .
$$

The expectation is over the randomness of the algorithm and its feedback. Section 2 specifies the feedback interface, action-set geometry, and network assumptions.

Related Work. Recent development in decentralized online continuous submodular maximization has produced several distinct regret-communication-computation tradeofs (Zhu et al., 2021; Zhang et al., 2023; Liao et al., 2023; Lu et al., 2025; Wan et al., 2026). Early algorithms focused on monotone continuous DR-submodular rewards, while Lu et al. (2025) provided a decentralized framework for the broader upperlinearizable interface. Recent reductions sharpen several application-specific results for non-monotone continuous DR-submodular rewards (Wan et al., 2026), specifically for smooth objectives. In the centralized setting, upper-linearizable algorithms and uniform wrappers provide the translation from online linear regret to a range of non-concave payof models (Pedramfar and Aggarwal, 2024; Pedramfar et al., 2025), and are recently extended to down-closed DR-submodular maximization (Lu et al., 2026). For online convex optimization (OCO) over dificult action sets, another line of work replaces Euclidean projection with weaker action-set access (Hazan and Kale, 2012; Garber and Hazan, 2016; Levy and Krause, 2019; Mhammedi, 2022; Garber and Kretzu, 2022; Mhammedi, 2025) and proposed projection-free algorithms. Of particular relevance, Mhammedi (2025) combine a separation-to-gauge reduction with a centralized Taylor Barrier-ONS analysis. This naturally motivates an oracle-specific question:

What performance can decentralized online upper-linearizable optimization achieve under separation access, and what is the resulting application in Decentralized Online Continuous Submodular Maximization?

We attain $\widetilde { O } ( \sqrt { T } )$ regret, $O ( T )$ communication, and $\widetilde O ( T )$ separation-oracle calls per agent under the assumptions made precise in Section 2. Table 1 reports the regret, communication, and action-set-oracle calls of the relevant methods under their respective access models. Because SO and LOO calls are distinct operations, the table compares asymptotic resource profiles rather than equal-cost operations or wall-clock performance. For a detailed discussion on oracle eficiency we refer to Remark 1 and the relevant projection-free OCO literature.

Contributions. We introduce Dec-BFTRL, a decentralized barrier-FTRL algorithm with a first-order dual-state communication layer for online upper-linearizable maximization under separation access, and prove $\widetilde { O } ( \sqrt { T } )$ per-agent network-aggregated expected α-regret. Over T rounds, each agent uses T neighbor-mixing operations, communicates one d-dimensional state per operation, and makes ${ \cal O } ( T \log ( \kappa T ) )$ calls to the action-set separation oracle. The number of HybridNewton updates is explicitly bounded for the controlled accuracy schedule. The general payof theorem yields square-root-T α-regret guarantees for three well known DR-submodular maximization problems.

Technical Novelties. Our main challenges and technical novelties are as follow:

1. A first-order communication interface for a curved local map. The barrier-FTRL map (4) has agent-dependent curvature, so directly coordinating Newton updates would require more than first-order messages (e.g., the Hessian). Our design instead communicates only the dual state through (5), and the exact-average identity (8) makes this state suficient to define the hypothetical network average problem, while every Hessian and Newton step remains local.

2. Coupling row-wise network mixing with barrier-FTRL stability. The network-aggregate metric evaluates every payof owner’s function $f _ { t , j }$ at one specified agent a’s action, whereas the surrogate analysis initially controls each payof at its owner’s iterate. To bridge this gap, we retain the cumulative row-wise mixing coeficient $M _ { a } ( W )$ (Lemma 2) and use it to control the specified agent’s dual-state disagreement (Lemma 3). Strong convexity of the barrier-FTRL potentials and stability of the composed payofs $f _ { t , j } \circ h$ then convert this row-wise dual control into feasible-action disagreement and bound the resulting owner-action cross terms (Lemma 4, 7 and (19)). Without this row-wise refinement, a direct global- $\cdot \ell _ { 2 }$ consensus bound would yield square-root regret with a $\sqrt { N }$ factor in place of $1 + \log N$

3. Controlled approximate local barrier-FTRL potential solver: Exact local minimization is unrealistic, but uncontrolled errors could accumulate linearly with the horizon. The HybridNewton subroutine uses the self-concordant stopping rule (22), and its computable Newton-decrement test guarantees the Euclidean error bound (Lemma 11). The accuracy schedule in Lemma 6 keeps the cumulative error bounded by $E _ { T } \leq R$ , while the warm-start analysis yields the finite iteration bound given by Proposition 1.

## 2 Preliminaries

## 2.1 Notations

Let $\mathsf { G } = ( \nu , \mathcal { E } )$ denote an undirected communication network of N agents, where $\mathcal { V } = \{ 1 , \ldots , N \}$ is the set of agents and E is the set of communication links. We associate the network with a mixing matrix $W \in \mathbb { R } ^ { N \times N }$ specified formally in Assumption 2.

We denote the Euclidean norm by $\| \cdot \| _ { 2 } .$ . For a positive definite matrix $\Sigma \in \mathbb { R } ^ { d \times d }$ , the induced local norm is $\| \boldsymbol { x } \| _ { \Sigma } : = \sqrt { \boldsymbol { x } ^ { \top } \Sigma \boldsymbol { x } }$ . We write $\mathbb { B } _ { 2 } : = \left\{ u \in \mathbb { R } ^ { d } : \| u \| _ { 2 } \leq 1 \right\}$ for the closed Euclidean unit ball.

## 2.2 Definitions

Separation oracle access and gauge projection. Given a closed convex set $\mathcal { K } \subseteq \mathbb { R } ^ { d }$ and a query point $x \in \mathbb { R } ^ { d }$ , a separation oracle $( \mathrm { S O } ) \ \mathrm { S e p } _ { \kappa } ( x )$ either certifies that $x \in \kappa$ or returns a nonzero normal vector $s \in \mathbb { R } ^ { d }$ such that $\langle s , x - z \rangle > 0 , \forall z \in K$ , and we call s a hyperplane separating x from K. We say an algorithm is under separation access when the algorithm can access the action set via a separation oracle, and when such separation oracle is indeed eficient over the action set.

Remark 1 (Two access oracle models). A separation oracle (SO) and a linear optimization oracle (LOO) expose complementary forms of action-set access, so their call counts do not by themselves determine computational cost. Separation is particularly natural for sets described by convex inequalities or intersections of simple constraints, where an infeasible query can be separated by returning a violated constraint and its normal, whereas linear optimization requires optimizing over the full intersection. For example, for a full-dimensional packing region $\mathcal { K } = x \in [ 0 , 1 ] ^ { d } \leq b , A \geq 0$ , with a known strictly feasible point, an SO scans the box and resource constraints and returns a violated inequality, while an LOO solves the associated packing linear program. In matrix domains, separation over the spectral-norm ball requires only a leading singular-vector pair, whereas linear optimization generally requires a full-rank SVD; the ordering reverses for the nuclear-norm ball (Garber and Kretzu, 2022). Thus neither oracle model uniformly dominates the other.

Under Assumption 1, a separation oracle for K also gives one for the translated set $c = \kappa - x _ { 0 }$ by translation. For the closed convex set C containing the origin in its interior, the Minkowski gauge function is $\gamma c ( u ) : = \operatorname* { i n f } \{ a > 0 : u \in a { \mathcal { C } } \}$ , and gauge distance is defined as $S c ( u ) : = \operatorname* { m a x } \{ 0 , \gamma c ( u ) - 1 \}$ . Using the gauge function and gauge distance, the exact gauge projection scales an infeasible point along the ray from the origin:

$$
p _ { \mathcal { C } } ( u ) : = \frac { u } { 1 + S _ { \mathcal { C } } ( u ) } \in \mathcal { C } , \quad w _ { \mathcal { K } } ( u ) : = x _ { 0 } + p _ { \mathcal { C } } ( u ) \in \mathcal { K } .
$$

Thus a feasible point is unchanged, while an infeasible point is scaled to the boundary. Specifically, we use the SO-based gauge subroutine of Mhammedi (2025), GaugeDist, which uses an approximate gauge projection controlled by a precision parameter $\varepsilon _ { \mathrm { g a u } } ,$ rather than the exact gauge projection. For completeness, we include the implementation along with useful properties and SO-complexity in Appendix A, and discuss how it interacts with our main algorithm in Section 3.1.

Linearizable Framework and Uniform Wrappers. Pedramfar and Aggarwal (2024) first develops the linearizable framework in centralized setting, providing a clean reduction from online linearizable optimization to online linear optimization. Pedramfar et al. (2025) further refines the reduction process by formulating the uniform wrappers for action, query and function. Pedramfar and Aggarwal (2024) also identifies two well known problems, namely monotone and non-monotone DR-submodular functions over general convex set as application of linearizable framework, which is later extended to non-monotone DR-submodular maximization over down-closed convex set with better approximation ratio and regret guarantee (Lu et al., 2026). Moving beyond centralized setting, Lu et al. (2025) uses linearizable framework as engine for decentralized setting and obtain results for the former two classes of DR-submodular maximization problems. Similarly, this work leverages it as a powerful interface so that as more function classes being added to the linearizable framework (Lu et al., 2026), this will automatically create a compounding impact with our meta-algorithm to unlock online decentralized optimization of such functions. We formally introduce the definition of upper-linearizable functions.

Definition 1 (Upper-linearizable function). A function class F over $\kappa$ is upper-linearizable with structural parameters $( \alpha , \beta ) \in ( 0 , 1 ] \times ( 0 , \infty )$ and action map $h : \mathcal { K } \to \mathcal { K }$ if there exists an linearization proxy ${ \mathfrak { g } } : { \mathcal { F } } \times { \mathcal { K } } \to \mathbb { R } ^ { d }$ such that, for every $f \in { \mathcal { F } }$ and $w , y \in { \mathcal { K } }$

$$
\alpha f ( y ) - f ( h ( w ) ) \leq \beta \langle { \mathfrak { g } } ( f , w ) , y - w \rangle .\tag{1}
$$

Following Pedramfar et al. (2025), we associate the upper-linearizable representation and its base oracle with a fixed abstract uniform wrapper $\mathcal { W } = ( \mathcal { W } ^ { \mathrm { a c t } } , \mathcal { W } ^ { \mathrm { q r y } } )$ . For each $f \in { \mathcal { F } }$ , let $\mathcal { O } _ { f }$ denote the base oracle<sup>2</sup> through which f is accessed. The action component is $w ^ { \mathrm { a c t } } = h$ , while the query component transforms $\mathcal { O } _ { f }$ into a wrapped vector-valued oracle $\widehat { \mathcal { O } } _ { f } : = \mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { f } )$ . When invoked at an internal point $w ,$ the wrapped oracle may use auxiliary randomness, query $\mathcal { O } _ { f }$ at one or more feasible points potentially diferent from both w and $h ( w )$ , and transforms the resulting responses into a feedback vector. The wrapper pair associated with each function class and base oracle combination is specified in the applications.

Online Protocol. We consider an oblivious adversarial model. Before the game begins, the adversary fixes a sequence of local payof functions $\{ f _ { t , j } \in \mathcal { F } : t \in [ T ] , j \in [ N ] \}$ , where $j$ indexes the owner of a local payof. The future payof functions and their oracle responses are not revealed to the agents in advance. At round $t ,$ each agent $a \in [ N ]$ , using only the information available before the current feedback is observed, computes an internal feasible point $w _ { a , t } \in \mathcal { K }$ and plays $\mathcal { W } ^ { \mathrm { a c t } } ( w _ { a , t } ) = h ( w _ { a , t } )$ . The agent then obtains access to the base oracle $\mathcal { O } _ { t , a } : = \mathcal { O } _ { f _ { t , a } ; }$ invokes the corresponding wrapped oracle at $w _ { a , t }$ , and receives $q _ { a , t } \sim \widehat { \mathcal { O } } _ { t , a } ( w _ { a , t } )$ , where $\widehat { \mathcal { O } } _ { t , a } : = \mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { t , a } )$ . The randomness in ${ { q } _ { a , t } }$ may arise from the base oracle, the Query Wrapper, or both. Its required conditional mean and boundedness properties are given in Assumption 3.

Regret. Under such protocol, the action $x _ { a , t }$ appearing in the regret definition from the Introduction is instantiated as $x _ { a , t } = \mathscr { W } ^ { \mathrm { a c t } } ( w _ { a , t } ) = h ( w _ { a , t } )$ , and the $\scriptstyle \alpha - \mathrm { r e g r e t } ^ { 3 }$ of agent a becomes

$$
\mathfrak { R } _ { \alpha } ^ { a } ( T ) : = \mathbb { E } \left[ \alpha \operatorname* { m a x } _ { y \in { \mathcal { K } } } \sum _ { t = 1 } ^ { T } \mathsf { F } _ { t } ( y ) - \sum _ { t = 1 } ^ { T } \mathsf { F } _ { t } ( h ( w _ { a , t } ) ) \right] .\tag{2}
$$

where $\begin{array} { r } { \mathsf { F } _ { t } ( \cdot ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } f _ { t , j } ( \cdot ) } \end{array}$ , so the action-agent index a is held fixed while $j$ ranges over all local-payof owners, including $j \neq a$

Remark 2 (Role of the approximation coeficient). Even in the centralized ofline setup, DR-submodular maximization could be NP-hard. As an example, Bian et al. (2017) shows that for monotone DR-submodular functions f over convex set containing the origin $\kappa ,$ it is NP-hard to find any point any point $x \in \kappa$ such that $f ( x )$ is at least $( 1 - e ^ { - 1 } + \epsilon )$ times the optimal value for any ϵ. There are polynomial times algorithms that achieve the ratio of $1 - e ^ { - 1 }$ , and such a number is referred to as the approximation coeficient, indicating the proportion of the optimal value we can achieve with polynomial time algorithms.

## 2.3 Assumptions

Assumption 1 (Translated geometry and separation access). The feasible set $\mathcal { K } \subseteq \mathbb { R } ^ { d }$ is compact and convex. A point $x _ { 0 } \in$ int(K) and radii $0 < r \le R$ are known such that, with $c : = \kappa - x _ { 0 }$

$$
r \mathbb { B } _ { 2 } \subseteq \mathcal { C } \subseteq R \mathbb { B } _ { 2 } .
$$

The set is accessed through a separation oracle for K and hence, by translation, for C. We set $\kappa : = R / r$

Assumption 2 (Network topology and communication matrix). The communication network $\mathsf { G } = ( \nu , \mathcal { E } )$ is undirected and connected. Agents form convex combinations of their local states using a matrix $W \in \mathbb { R } ^ { N \times N }$ satisfying: 1) Graph compatibility: $W _ { i j } = 0$ whenever $i \neq j$ and $\{ i , j \} \not \in \mathcal { E } ; 2 )$ Symmetry and double stochasticity: $W \overset {  } { = } W ^ { \top } , \bar { W } _ { i j } \geq 0$ , and $\bar { W } \mathbf { 1 } = \mathbf { 1 } ; 3 )$ Contraction: with $J : = \mathbf { 1 1 } ^ { \top } / N , \rho : = \| W - J \| _ { 2 } < 1$ Thus $\rho$ is the disagreement contraction factor and $1 - \rho$ is the absolute spectral gap.

Assumption 3 (Conditionally unbiased bounded feedback). Let $\mathcal { F } _ { t } ^ { - }$ denote the sigma-field generated by the fixed payof sequence and all algorithmic and oracle randomness revealed before the round-t wrapped-oracle responses are generated, including the current wrapper inputs $\{ w _ { a , t } \} _ { a = 1 } ^ { N }$ . For every agent a and round t, the wrapped feedback vector satisfies

$$
\begin{array} { r } { { \mathbb E } [ q _ { a , t } \ | \ { \mathcal F } _ { t } ^ { - } ] = \mathfrak { g } ( f _ { t , a } , w _ { a , t } ) , \qquad \| q _ { a , t } \| _ { 2 } \le G . } \end{array}
$$

Assumption 4 (Uniform composed-payof stability). There is a horizon-independent constant $L _ { \phi } \ge 0$ such that, almost surely, for every round t, payof owner $j ,$ and $w , v \in \mathcal { K }$

$$
| f _ { t , j } ( h ( w ) ) - f _ { t , j } ( h ( v ) ) | \leq L _ { \phi } \| w - v \| _ { 2 } .\tag{3}
$$

Equivalently, each composed payof $f _ { t , j } \circ h$ is uniformly $L _ { \phi ^ { - 1 } }$ Lipschitz on $\kappa$

## 3 Decentralized Barrier Follow the Regularized Leader (Dec-BFTRL)

Having specified the online protocol and assumptions, we now present Dec-BFTRL and its main guarantee. The method combines three components: GaugeDist uses separation access to form a feasible wrapper input and the associated gauge correction; the agents convert their wrapped feedback into local surrogate losses and exchange only a single dual state; and HybridNewton locally computes an approximate minimizer of the resulting barrier-FTRL potential. Thus all feasibility and second-order computations remain local, while the network interaction is confined to one dual-state exchange per round. Subsection 3.1 gives a round-by-round description of the method and its subroutines, while Subsection 3.2 states the main regret guarantee and provides a proof sketch. The supporting intermediate results and complete proof are deferred to Section 4.

## 3.1 Main Algorithm

At the beginning of round $t ,$ each agent a holds two local states. The dual state $\theta _ { a , t } \in \mathbb { R } ^ { d }$ contains a mixed history of earlier surrogate losses and is the only state communicated to neighboring agents. The primal state $u _ { a , t } \in \mathrm { i n t } ( R \mathbb { B } _ { 2 } )$ is kept locally and approximately minimizes the BFTRL potential

$$
\Psi _ { a , t } ( u ) : = - \nu \log \bigl ( R ^ { 2 } - \| u \| _ { 2 } ^ { 2 } \bigr ) + \frac { \mu _ { T } } { 2 } \| u \| _ { 2 } ^ { 2 } + \langle \theta _ { a , t } , u \rangle .\tag{4}
$$

and we denote its unique exact minimizer by $\begin{array} { r } { u _ { a , t } ^ { \star } : = \arg \operatorname* { m i n } _ { u \in \mathrm { i n t } ( R \mathbb { B } _ { 2 } ) } \Psi _ { a , t } ( u ) } \end{array}$ . The barrier and quadratic terms together make $\Psi _ { a , t }$ strongly convex, with parameter $\begin{array} { r } { m _ { T } : = \mu _ { T } + \frac { 2 \nu } { R ^ { 2 } } } \end{array}$ . Initially, $\theta _ { a , 1 } = 0$ and $u _ { a , 1 } = 0 = u _ { a , 1 } ^ { \star }$ More generally, $( \theta _ { a , t } , u _ { a , t } )$ depends only on feedback from rounds preceding t. Round t produces $\left( \theta _ { a , t + 1 } , u _ { a , t + 1 } \right)$ which is used at round t + 1. Algorithm 1 summarizes this recursion.

Algorithm 1 Dec-BFTRL: decentralized barrier-FTRL under separation access   
Require: Horizon T; separation-oracle access to $\textstyle \overbrace { \kappa ; }$ known $x _ { 0 } , r , R ;$ symmetric doubly stochastic mixing   
matrix $W ;$ abstract uniform wrapper $\mathcal { W } = ( \mathcal { W } ^ { \mathrm { a c t } } , \mathcal { W } ^ { \mathrm { q r y } } )$ ; parameters $\nu \geq 1 , \mu _ { T } > 0 ;$ local solve tolerances   
$\left\{ \varepsilon _ { a , t } \right\} _ { a \in [ N ] , t = 2 , \ldots , T }$   
1: Set $c \gets \dot { \kappa } - x _ { 0 }$ and $\varepsilon _ { \mathrm { g a u } }  1 / T$   
2: For every agent $a \in [ N ] .$ , initialize $\theta _ { a , 1 } \gets 0$ and $u _ { a , 1 } \gets 0$   
3: for $t = 1 , \dots , T$ do   
4: for every agent $a \in [ N ]$ in parallel do   
5: $( S _ { a , t } , s _ { a , t } ) \gets \mathrm { G A U G E D I S T } ( \mathcal { C } , u _ { a , t } , \varepsilon _ { \mathrm { g a u } } , r )$   
6: $p _ { a , t } \gets u _ { a , t } / ( 1 + S _ { a , t } )$ and $w _ { a , t } \gets x _ { 0 } + p _ { a , t }$   
7: Play $x _ { a , t } \gets \mathcal { W } ^ { \mathrm { a c t } } ( w _ { a , t } ) = h ( w _ { a , t } ) ;$   
8: Invoke the wrapped oracle $\mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { t , a } )$ at $w _ { a , t }$ and receive ${ { q } _ { a , t } }$   
9: Let $\ell _ { a , t } \gets - q _ { a , t }$ and $\widetilde { \ell } _ { a , t } \gets \ell _ { a , t } - \mathbf { 1 } \{ \langle \ell _ { a , t } , u _ { a , t } \rangle < 0 \} \langle \ell _ { a , t } , p _ { a , t } \rangle s _ { a , t }$   
10: Exchange the dual state $\theta _ { a , t }$ with neighbors and update $\theta _ { a , t + 1 } \gets \sum _ { b = 1 } ^ { N } W _ { a b } \theta _ { b , t } + \widetilde { \ell } _ { a , t }$   
11: $\mathbf { i f } \ t < T$ then   
12: Form $\Psi _ { a , t + 1 }$ from (4)   
13: $u _ { a , t + 1 } \gets$ HybridNewton $\left( \Psi _ { a , t + 1 } , \boldsymbol { u } _ { a , t } , \varepsilon _ { a , t + 1 } \right)$   
14: end if   
15: end for   
16: end for

Gauge projection. The primal state $u _ { a , t }$ lies in the simple barrier domain $R \mathbb { B } _ { 2 } .$ , but it need not belong to the translated feasible set ${ \mathcal C } = { \mathcal K } - x _ { 0 }$ . The call to GaugeDist returns an approximate gauge distance $S _ { a , t }$ and an approximate subgradient $s _ { a , t } .$ moderated by the gauge precision parameter $\varepsilon _ { \mathrm { g a u } } .$ Because $S _ { a , t }$ upper bounds the true gauge distance, the radial scaling $\begin{array} { r } { p _ { a , t } = \frac { u _ { a , t } } { 1 + S _ { a , t } } } \end{array}$ belongs to ${ \mathcal { C } } ,$ , and therefore $\boldsymbol { w } _ { a , t } = \boldsymbol { x } _ { 0 } + \boldsymbol { p } _ { a , t }$ belongs to $\kappa .$ . The vector $s _ { a , t }$ is retained for the surrogate-loss correction in the next step. This construction allows the barrier-FTRL update to operate on a Euclidean ball while producing a feasible input for the Action Wrapper. The corresponding feasibility, approximation, and separation-oracle guarantees are established in Section 4.1.

Feedback and surrogate loss. After constructing $w _ { a , t }$ , the agent plays $x _ { a , t } = \mathscr { W } ^ { \mathrm { a c t } } ( w _ { a , t } ) = h ( w _ { a , t } )$ and invokes the wrapped oracle $\mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { t , a } )$ at $w _ { a , t } ,$ receiving the payof-side vector $q _ { a , t } .$ Because the barrier-FTRL update is written as a loss-minimization procedure, the agent first sets $\ell _ { a , t } = - q _ { a , t }$ . It then combines $\ell _ { a , t }$ with the gauge information ${ p } _ { a , t }$ and $s _ { a , t }$ to form the corrected surrogate loss $\widetilde { \ell } _ { a , t } .$ , and the correction is activated only when $\langle \ell _ { a , t } , u _ { a , t } \rangle < 0$ . Its role is to relate linear loss at the feasible point $w _ { a , t }$ to a bounded linear surrogate evaluated at the internal state $u _ { a , t }$ . The precise comparison inequality and norm bound for $\widetilde { \ell } _ { a , t }$ are given in Section 4.1. The implementation and query cost of the wrapped oracle are specified separately for each application

Communication and dual update. Once the current surrogate has been formed, each agent exchanges only its dual state $\theta _ { a , t }$ with its neighbors. Although this exchange appears after the feedback step in Algorithm 1, the communicated vector is the pre-update state and therefore contains only information from earlier rounds. Agent a first mixes these dual states and then injects its current local surrogate:

$$
\theta _ { a , t + 1 } = \sum _ { b = 1 } ^ { N } W _ { a b } \theta _ { b , t } + \widetilde { \ell } _ { a , t } .\tag{5}
$$

Thus the feedback received at round t afects the primal state used at round $t + 1$ , not the action already played at round t. Graph compatibility of W makes the displayed sum implementable through neighbor communication. No primal iterate, feasible point, Hessian, or Newton information is exchanged. Section 4.2 analyzes the evolution of the average dual state and controls disagreement through the network spectral gap.

HybridNewton update. The new dual state $\theta _ { a , t + 1 }$ defines the next local potential $\Psi _ { a , t + 1 }$ . Agent a warm-starts HybridNewton from $u _ { a , t }$ and computes an interior point satisfying

$$
\lVert u _ { a , t + 1 } - u _ { a , t + 1 } ^ { \star } \rVert _ { 2 } \leq \varepsilon _ { a , t + 1 } .
$$

The logarithmic barrier keeps the Newton iterates inside $R \mathbb { B } _ { 2 }$ , while strong convexity converts the solver’s Newton-decrement stopping condition into the stated Euclidean accuracy. The solver first uses damped Newton steps and then switches to full Newton steps once it enters the local convergence region. This computation is entirely local, and the returned point $u _ { a , t + 1 }$ begins the next round. The post-round solve is omitted when $t = T$ , since no subsequent action is needed. Section 4.3 establishes the accuracy guarantee and iteration complexity. Each online round requires one exchange of a d-dimensional dual state per agent, and all gauge computations, wrapped-oracle operations, and HybridNewton updates remain local.

## 3.2 Main Guarantee

Let $\delta : = 1 - \rho , \widetilde G : = 2 \kappa G$ . For main guarantee, we set ν = 1, $\begin{array} { r } { \mu _ { T } = \frac { \widetilde { G } } { R } \sqrt { 2 T \left( 1 + \frac { 1 } { \delta } \right) } } \end{array}$ , and $\begin{array} { r } { m _ { T } = \mu _ { T } + \frac { 2 } { R ^ { 2 } } } \end{array}$ . We use gauge precision $\varepsilon _ { \mathrm { g a u } } = 1 / T$ and local solve inaccuracy tolerances $\varepsilon _ { a , 1 } : = 0$ , and $\begin{array} { r } { \varepsilon _ { a , t } : = \frac { R } { T } } \end{array}$ for $2 \leq t \leq T$ For each agent $a \in [ N ]$ , define its row-mixing coeficient

$$
M _ { a } ( W ) : = \sum _ { k = 0 } ^ { \infty } \sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| .\tag{6}
$$

Assumption 2 ensures that $M _ { a } ( W ) < \infty$ . Define also

$$
\mathcal { B } _ { T } : = \beta \left[ 3 G R + \widetilde { G } R + \log T + \widetilde { G } R \sqrt { 2 T \left( 1 + \frac { 1 } { \delta } \right) } \right] .
$$

Theorem 1 (Dec-BFTRL: per-agent regret and total SO calls). Let $\begin{array} { r } { \mathsf { F } _ { t } ( \cdot ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } f _ { t , j } ( \cdot ) } \end{array}$ . Suppose every $f _ { t , j }$ belongs to the common (α, β)-upper-linearizable class of Definition 1, equipped with the fixed Action and Query Wrappers used by Algorithm 1. Suppose Assumptions $\begin{array} { r } { 1 , \ 2 , \ 3 , } \end{array}$ and 4 hold with constants $G > 0$ and $L _ { \phi } \geq 0$ . For every agent $a \in [ N ]$ , running Algorithm 1 with the above parameters choices ensures

$$
\mathbb { E } \left[ \alpha \operatorname* { m a x } _ { y \in \mathcal { K } } \sum _ { t = 1 } ^ { T } \mathsf { F } _ { t } ( y ) - \sum _ { t = 1 } ^ { T } \mathsf { F } _ { t } ( h ( w _ { a , t } ) ) \right] \le B _ { T } + 2 L _ { \phi } R ( 2 + \kappa ) + L _ { \phi } ( 1 + \kappa ) R \sqrt { \frac { T \delta } { 2 ( 1 + \delta ) } } \left( M _ { a } ( W ) + \frac { 1 } { \delta } \right)\tag{7}
$$

and each agent makes at most $T \left( 1 + \left\lceil \log _ { 2 } ( 4 \kappa ^ { 2 } T ) \right\rceil \right)$ total calls to the action-set separation oracle. This count excludes any action-set-oracle calls, if any, made by an application-specific wrapper.

Section 4.2 shows that $\begin{array} { r } { M _ { a } ( W ) = O \left( \frac { 1 + \log N } { 1 - \rho } \right) } \end{array}$ . Therefore, Theorem 1 implies for every agent a

$$
\Re _ { \alpha } ^ { a } ( T ) = O \left( \beta \log T + [ \beta \kappa G R + L _ { \phi } ( 1 + \kappa ) R ( 1 + \log N ) ] \sqrt { \frac { T } { 1 - \rho } } \right) = \widetilde { O } ( \sqrt { T } )
$$

## 4 Analysis

Throughout this section, we work under the hypotheses and parameter choices of Theorem 1. All pathwise statements are understood on the probability-one event on which the almost-sure feedback bounds in Assumption 3 hold simultaneously for all agents and rounds. In the Analysis, we extend the parameter setup and notations used in section 3.2. We defer the detailed proof of each Lemma in this section to the Appendices.

## 4.1 Gauge Projection and Surrogate Loss

Lemma 1 provides all the necessary information we need from the gauge projection subroutine, and its proof utilizes Lemma 8 and Lemma 9 in Appendix A.

Lemma 1 (Gauge projection and surrogate bound). For every payof owner j and round t, the point constructed by Algorithm 1 satisfies $w _ { j , t } \in \mathcal { K }$ , and $\Vert \widetilde { \ell } _ { j , t } \Vert _ { 2 } \leq \widetilde { G }$ . Moreover, for every $v \in { \mathcal { C } }$

$$
\langle \ell _ { j , t } , w _ { j , t } - ( x _ { 0 } + v ) \rangle \leq \langle \widetilde { \ell } _ { j , t } , u _ { j , t } - v \rangle + \frac { 2 G R } { T } .
$$

## 4.2 Network Mixing and Dual States

Recall from (6), $\begin{array} { r } { M _ { a } ( W ) = \sum _ { k = 0 } ^ { \infty } \sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| } \end{array}$ . Lemma 2 is a useful technical lemma for the row mixing operation of the communication matrix and does not concern the dual state. Lemma 2 (Row mixing). For every agent $a \in [ N ]$ and $k \geq 0$

$$
\sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| \leq \operatorname* { m i n } \{ 2 , \sqrt { N } \rho ^ { k } \} .
$$

Consequently, $M _ { a } ( W ) < \infty$ . Moreover, $\begin{array} { r } { M _ { a } ( W ) = O \left( \frac { 1 + \log N } { 1 - \rho } \right) } \end{array}$

At round t, define the hypothetical network-averaged dual state as

$$
\bar { \theta } _ { t } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \theta _ { j , t }
$$

for $1 \leq t \leq T + 1$ . For $1 \leq t \leq T$ define the hypothetical network-averaged surrogate loss

$$
\bar { \ell } _ { t } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \widetilde { \ell } _ { j , t } .
$$

Also denote the norm of dual disagreement for agent a as

$$
d _ { a , t } : = \lVert { \boldsymbol { \theta } } _ { a , t } - { \boldsymbol { \bar { \theta } } } _ { t } \rVert _ { 2 } .
$$

Lemma 3 analyzes the disagreement in dual space, Lemma 4 builds a bridge between primal space and dua space, and Lemma 5 provides the surrogate regret bound for hypothetical network average BFTRL primal problem.

Lemma 3 (Dual-state averages and disagreement). For every $1 \leq t \leq T + 1$ ，

$$
\bar { \theta } _ { t } = \sum _ { s < t } \bar { \ell } _ { s } .\tag{8}
$$

Moreover,

$$
\left( \sum _ { b = 1 } ^ { N } \| \theta _ { b , t } - \bar { \theta } _ { t } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } \leq \frac { \sqrt { N } \widetilde { G } ( 1 - \rho ^ { t - 1 } ) } { \delta } .\tag{9}
$$

For every agent $a ,$

$$
\sum _ { t = 1 } ^ { T } d _ { a , t } \leq T \widetilde { G } M _ { a } ( W ) ,\tag{10}
$$

The agent-average disagreement also satisfies

$$
\sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { b = 1 } ^ { N } d _ { b , t } \leq \frac { T \widetilde { G } } { \delta } .\tag{11}
$$

With the hypothetical network average dual state ${ \bar { \theta } } _ { t }$ , define the hypothetical network average potential

$$
\bar { \Psi } _ { t } ( u ) : = - \nu \log ( R ^ { 2 } - \| u \| _ { 2 } ^ { 2 } ) + \frac { \mu _ { T } } { 2 } \| u \| _ { 2 } ^ { 2 } + \langle \bar { \theta } _ { t } , u \rangle ,\tag{12}
$$

and the minimizer of the average potential $\begin{array} { r } { \bar { u } _ { t } ^ { \star } : = \arg \operatorname* { m i n } _ { u \in \mathrm { i n t } ( R \mathbb { B } _ { 2 } ) } \bar { \Psi } _ { t } ( u ) } \end{array}$

Lemma 4 (Minimizer disagreement). For every agent a and round t, $\begin{array} { r } { \| u _ { a , t } ^ { \star } - \bar { u } _ { t } ^ { \star } \| _ { 2 } \leq \frac { \| \theta _ { a , t } - \bar { \theta } _ { t } \| _ { 2 } } { m _ { T } } } \end{array}$

Lemma 5 (Network-average BFTRL regret). $\begin{array} { r } { \forall v \in \operatorname { i n t } ( R \mathbb { B } _ { 2 } ) , \sum _ { t = 1 } ^ { T } \langle \bar { \ell } _ { t } , \bar { u } _ { t } ^ { \star } - v \rangle \leq \bar { \Psi } _ { 1 } ( v ) - \bar { \Psi } _ { 1 } ( 0 ) + \frac { T \widetilde G ^ { 2 } } { m _ { T } } . } \end{array}$

## 4.3 Local Solve Inaccuracy Tolerance

Given the local solve inaccuracy tolerances $\left\{ \varepsilon _ { a , t } \right\} _ { a \in [ N ] , t = 2 , \dots , T }$ , we define the total tolerance for agent a over horizon T as $\begin{array} { r } { E _ { a , T } : = \sum _ { t = 1 } ^ { T } \varepsilon _ { a , t } } \end{array}$ , and the average per-agent tolerance over horizon T as $\begin{array} { r } { E _ { T } : = \frac { 1 } { N } \sum _ { a = 1 } ^ { N } E _ { a , T } . } \end{array}$ Lemma 6 (Local solve tolerance). Given the tolerance schedule in Subsection 3.2, every call to Hybrid-Newton terminates after finitely many Newton steps and returns an interior point satisfying

$$
\| u _ { a , t } - u _ { a , t } ^ { \star } \| _ { 2 } \leq \varepsilon _ { a , t } .\tag{13}
$$

Moreover,

$$
E _ { a , T } = { \frac { ( T - 1 ) R } { T } } < R , \qquad E _ { T } < R .\tag{14}
$$

We define the common feasible point $z _ { t } : = x _ { 0 } + p c ( \bar { u } _ { t } ^ { \star } )$ , where $p _ { \mathcal { C } }$ is the exact gauge projection from Section 2, and $\bar { u } _ { t } ^ { \star }$ is the minimizer of hypothetical average potential (12).

Lemma 7 (Feasible-point disagreement). For every agent a and round t, the average internal primal state disagreement is bounded by

$$
\sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| w _ { a , t } - w _ { j , t } \| _ { 2 } \leq 2 R + ( 1 + \kappa ) ( E _ { a , T } + E _ { T } ) + \frac { ( 1 + \kappa ) { \tilde { G } } T } { m _ { T } } \left( M _ { a } ( W ) + \frac { 1 } { \delta } \right) .
$$

## 4.4 Proof of the Main Theorem

Proof of Theorem 1. Define the owner-local payof term

$$
D _ { \alpha } ( T ; y ) : = \mathbb { E } \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } ( \alpha f _ { t , j } ( y ) - f _ { t , j } ( h ( w _ { j , t } ) ) ) \right] .
$$

Lemma 1 gives $w _ { j , t } \in \mathcal { K }$ . By the definition of $F _ { t } ^ { - } , w _ { j , t }$ is $\mathcal { F } _ { t } ^ { - }$ -measurable, while $f _ { t , j }$ and y are fixed. Moreover, $q _ { j , t }$ , and hence $\ell _ { j , t } = - q _ { j , t } .$ is integrable by Assumption 3. Therefore, conditional on $\mathcal { F } _ { t } ^ { - }$ , Definition 1 and Assumption 3 imply

$$
\begin{array} { r l } & { \alpha f _ { t , j } ( y ) - f _ { t , j } ( h ( w _ { j , t } ) ) \leq \beta \langle \mathfrak { g } ( f _ { t , j } , w _ { j , t } ) , y - w _ { j , t } \rangle } \\ & { \qquad = \beta \mathbb { E } \left[ \langle \ell _ { j , t } , w _ { j , t } - y \rangle \mid \mathcal { F } _ { t } ^ { - } \right] . } \end{array}
$$

Taking total expectations yields

$$
D _ { \alpha } ( T ; y ) \leq \frac { \beta } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \ell _ { j , t } , w _ { j , t } - y \rangle ,
$$

and we call $\begin{array} { r } { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \ell _ { j , t } , w _ { j , t } - y \rangle } \end{array}$ on the right hand side the linear surrogate regret. Note that compared with the network aggregate regret, the owner-local payof terms use the action for the agent of the payof function, while the network regret fix the agent of the action. The gap between these two will be bounded in the second step after bounding the linear surrogate regret. After that, we will present parameter tuning to obtain best result, and analyze the total SO calls.

Linear Surrogate regret. Fix $y \in { \cal K } ,$ let $v _ { 0 } : = y - x _ { 0 } \in \mathcal { C }$ , and set $\begin{array} { r } { v : = \left( 1 - \frac { 1 } { T } \right) v _ { 0 } } \end{array}$ . By Assumption 1, $v \in \mathcal { C } \cap \mathrm { i n t } ( R \mathbb { B } _ { 2 } )$ and $\| x _ { 0 } + v - y \| _ { 2 } \leq R / T$ . Since $\| \ell _ { j , t } \| _ { 2 } = \| q _ { j , t } \| _ { 2 } \leq G$ by Assumption 3,

$$
\frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \ell _ { j , t } , x _ { 0 } + v - y \rangle \leq G R .
$$

Applying Lemma 1 and summing its $2 G R / T$ error over t gives

$$
\frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \ell _ { j , t } , w _ { j , t } - y \rangle \leq 3 G R + \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \widetilde { \ell } _ { j , t } , u _ { j , t } - v \rangle .\tag{15}
$$

The remaining term has the exact decomposition

$$
\frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \widetilde { \ell } _ { j , t } , u _ { j , t } - v \rangle = \frac { 1 } { N } \sum _ { j , t } \langle \widetilde { \ell } _ { j , t } , u _ { j , t } - u _ { j , t } ^ { \star } \rangle + \frac { 1 } { N } \sum _ { j , t } \langle \widetilde { \ell } _ { j , t } , u _ { j , t } ^ { \star } - \bar { u } _ { t } ^ { \star } \rangle + \sum _ { t = 1 } ^ { T } \langle \bar { \ell } _ { t } , \bar { u } _ { t } ^ { \star } - v \rangle .\tag{16}
$$

The first term on the right is bounded by

$$
\frac { 1 } { N } \sum _ { j , t } \langle \widetilde { \ell } _ { j , t } , u _ { j , t } - u _ { j , t } ^ { \star } \rangle \leq \frac { 1 } { N } \sum _ { j , t } \Vert \widetilde { \ell } _ { j , t } \Vert _ { 2 } \Vert u _ { a , t } - u _ { a , t } ^ { \star } \Vert _ { 2 } = \widetilde { G } \frac { 1 } { N } \sum _ { j , t } \Vert u _ { a , t } - u _ { a , t } ^ { \star } \Vert _ { 2 } = \widetilde { G } E _ { T } ,
$$

where the first inequality is due to cauchy-shwarz inequality, the second equation comes from Lemmas 1 and the third equation comes from Lemma 6.

The second term is bounded by

$$
\frac { 1 } { N } \sum _ { j , t } \langle \widetilde { \ell } _ { j , t } , u _ { j , t } ^ { \star } - \bar { u } _ { t } ^ { \star } \rangle \leq \frac { 1 } { N } \sum _ { j , t } \| \widetilde { \ell } _ { j , t } \| _ { 2 } \| u _ { a , t } ^ { \star } - \bar { u } _ { t } ^ { \star } \| _ { 2 } \leq \frac { \widetilde { G } } { m _ { T } } \sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } d _ { j , t } \leq \frac { T \widetilde { G } ^ { 2 } } { m _ { T } \delta } .
$$

where the first inequality is due to cauchy-shwarz inequality, the second inequality comes from Lemma 1 and Lemma 4, the third and forth inequality come from Lemma 3.

The third term is directed bounded by Lemma 5 with

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \langle \bar { \ell } _ { t } , \bar { u } _ { t } ^ { \star } - v \rangle \leq \bar { \Psi } _ { 1 } ( v ) - \bar { \Psi } _ { 1 } ( 0 ) + \frac { T \widetilde { G } ^ { 2 } } { m _ { T } } } \\ & { \quad \quad \quad = - \nu \log \bigg ( 1 - \frac { \| v \| _ { 2 } ^ { 2 } } { R ^ { 2 } } \bigg ) + \frac { \mu _ { T } } { 2 } \| v \| _ { 2 } ^ { 2 } + \frac { T \widetilde { G } ^ { 2 } } { m _ { T } } } \\ & { \quad \quad \quad \leq \nu \log T + \frac { \mu _ { T } R ^ { 2 } } { 2 } + \frac { T \widetilde { G } ^ { 2 } } { m _ { T } } } \end{array}
$$

where the first inequality is from Lemma 5, the second equation is because (8) gives $\bar { \theta } _ { 1 } = 0$ , and the third inequality is because $\| v \| _ { 2 } \leq ( 1 - 1 / T ) R$ and $1 - ( 1 - 1 / T ) ^ { 2 } \geq 1 / T$

Substitution of the three partial bounds back into (15) bounds the linear surrogate regret:

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \langle \ell _ { j , t } , w _ { j , t } - y \rangle \leq B _ { \operatorname* { l i n } } : = 3 G R + \nu \log T + \frac { \mu _ { T } R ^ { 2 } } { 2 } + \widetilde G E _ { T } + \frac { T \widetilde G ^ { 2 } } { m _ { T } } \left( 1 + \frac { 1 } { \delta } \right) .\tag{17}
$$

Thus, we have $D _ { \alpha } ( T ; y ) \le \beta B _ { \mathrm { l i n } }$

Network-aggregate regret. For a fixed comparator $y \in \kappa$ , write

$$
\mathfrak { R } _ { \alpha } ^ { a } ( T ; y ) : = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( \alpha \mathsf { F } _ { t } ( y ) - \mathsf { F } _ { t } ( h ( w _ { a , t } ) ) ) \right] ,
$$

where $\begin{array} { r } { \mathsf { F } _ { t } ( y ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } f _ { t , j } ( y ) } \end{array}$ and let $\phi _ { t , j } : = f _ { t , j } \circ h$ . Expanding the aggregate payof gives the exact identity

$$
\begin{array} { l } { \displaystyle \mathfrak { R } _ { \alpha } ^ { \alpha } ( T ; y ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( \alpha \mathsf { F } _ { t } ( y ) - \mathsf { F } _ { t } ( h ( w _ { a , t } ) ) ) \right] = \mathbb { E } \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } ( \alpha f _ { t , j } ( y ) - f _ { t , j } ( h ( w _ { a , t } ) ) ) \right] } \\ { \displaystyle = \mathbb { E } \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } ( \alpha f _ { t , j } ( y ) - f _ { t , j } ( h ( w _ { j , t } ) ) ) \right] + \mathbb { E } \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } ( f _ { t , j } ( h ( w _ { j , t } ) ) - f _ { t , j } ( h ( w _ { a , t } ) ) ) \right] } \\ { \displaystyle = D _ { \alpha } ( T ; y ) + \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( \phi _ { t , j } ( w _ { j , t } ) - \phi _ { t , j } ( w _ { a , t } ) ) \right] . } \end{array}\tag{18}
$$

Recall Lemma 1 ensures that both $w _ { j , t }$ and $w _ { a , t }$ belong to $\kappa ,$ so Assumption 4 applies. Since $\phi _ { t , j } = f _ { t , j } \circ h$

$$
\phi _ { t , j } ( w _ { j , t } ) - \phi _ { t , j } ( w _ { a , t } ) \leq | \phi _ { t , j } ( w _ { j , t } ) - \phi _ { t , j } ( w _ { a , t } ) | \leq L _ { \phi } \| w _ { j , t } - w _ { a , t } \| _ { 2 } .
$$

Thus, substituting the linear surrogate regret of the local-owner payofs $D _ { \alpha } ( T ; y ) \le \beta B _ { \mathrm { l i n } }$ and the above inequality into (18) we have

$$
\begin{array} { r l r } {  { \mathfrak { R } _ { \alpha } ^ { a } ( T ; y ) \le \beta B _ { \mathrm { l i n } } + L _ { \phi } \mathbb { E } [ \sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| w _ { j , t } - w _ { a , t } \| _ { 2 } ] } } \\ & { } & { \leq \beta B _ { \mathrm { l i n } } + L _ { \phi } [ 2 R + ( 1 + \kappa ) ( E _ { a , T } + E _ { T } ) + \frac { ( 1 + \kappa ) \widetilde { G } T } { m _ { T } } ( M _ { a } ( W ) + \frac { 1 } { \delta } ) ] . } \end{array}\tag{19}
$$

where the second inequality comes from Lemma 7. The payof sequence is fixed before the learner’s randomness, and the hindsight maximum is attained. We may therefore select $\begin{array} { r } { y ^ { \star } \in \arg \operatorname* { m a x } _ { y \in { \mathcal { K } } } \sum _ { t = 1 } ^ { T } { \sf F } _ { t } ( y ) } \end{array}$ then $\Re _ { \alpha } ^ { a } ( T ) = \Re _ { \alpha } ^ { a } ( T ; y ^ { \star } )$

Parameter choices and separation-oracle calls. Lemma 6 gives $E _ { a , T } , E _ { T } \leq R$ . Since $m _ { T } \geq \mu _ { T }$

$$
\frac { \mu _ { T } R ^ { 2 } } { 2 } + \frac { T \widetilde { G } ^ { 2 } } { m _ { T } } \left( 1 + \frac { 1 } { \delta } \right) \leq \frac { \mu _ { T } R ^ { 2 } } { 2 } + \frac { T \widetilde { G } ^ { 2 } } { \mu _ { T } } \left( 1 + \frac { 1 } { \delta } \right) = \widetilde { G } R \sqrt { 2 T \left( 1 + \frac { 1 } { \delta } \right) } .
$$

Using $\nu = 1$ , it follows that $\beta B _ { \mathrm { l i n } } \le B _ { T }$ . Furthermore, $\begin{array} { r } { \frac { \widetilde G T } { m _ { T } } \leq \frac { \widetilde G T } { \mu _ { T } } = R \sqrt { \frac { T \delta } { 2 ( 1 + \delta ) } } } \end{array}$ . Substituting these bounds back to (19), and observing that $2 R + ( 1 + \kappa ) ( E _ { a , T } + E _ { T } ) \leq 2 R ( 2 + \kappa )$ , we have (7):

$$
\mathfrak { R } _ { \alpha } ^ { a } ( T ) \le \mathfrak { B } _ { T } + 2 L _ { \phi } R ( 2 + \kappa ) + L _ { \phi } ( 1 + \kappa ) R \sqrt { \frac { T \delta } { 2 ( 1 + \delta ) } } \left( M _ { a } ( W ) + \frac 1 \delta \right) .
$$

$$
\mathrm { w h e r e } \ B _ { T } = \beta \left[ 3 G R + \widetilde { G } R + \log T + \widetilde { G } R \sqrt { 2 T \left( 1 + \frac { 1 } { \delta } \right) } \right] .
$$

Finally, $u _ { a , 1 } = 0 \mathrm { ~ , ~ }$ , and Lemma 6 ensures that $u _ { a , t } \in$ int $\left( R \mathbb { B } _ { 2 } \right)$ for $2 \leq t \leq T$ . Thus every GaugeDist query satisfies the domain hypothesis of Lemma 9. Since $\varepsilon _ { \mathrm { g a u } } = 1 / T \in ( 0 , 1 ]$ , that lemma gives at most $1 + \left\lceil \log _ { 2 } ( 4 \kappa ^ { 2 } T ) \right\rceil$ separation-oracle calls per invocation. There is one invocation per agent-round. By translation, each call to $\operatorname { S e p } _ { \mathit { c } }$ is implemented by one call to $\mathrm { S e p } _ { \kappa } ,$ proving the stated per-agent GaugeDist total. □

## 5 Conclusion, Limitations, and Future Works

We proposed Dec-BFTRL, for decentralized online upper-linearizable optimization under separation-oracle access. The algorithm attains agent-average expected approximate regret $\widetilde { O } \left( \sqrt { T } \right)$ with linear communication in horizon and ${ \cal O } ( T \log ( \kappa T ) )$ total separation-oracle calls. Our analysis avoids distributed curvature tracking. The agents mix a single first-order dual state, and minimizer sensitivity converts disagreement in that state into a primal regret term. A controlled approximate hybrid Newton procedure supplies summable local error without adding communication.

The guarantee in this work is specific to domains with a known full-dimensional interior point and economical separation access. It does not impose a universal computational ordering between separation and linearoptimization oracles. Extension to relative-interior geometries such as matroid basis polytopes remains open, and it is necessary before incorporating application to other classes such as the one-sided smooth functions. Separately, for the applications to DR-submodular classes, this work only discusses static regret under the natural first-order feedback setting. Extensions to limited feedback setting, including zeroth-order and bandit feedback, and the investigation of non-stationary environment like dynamic and adaptive regret, remain open.

## References

Riccardo Aldrighetti, Daria Battini, Dmitry Ivanov, and Ilenia Zennaro. Costs of resilience and disruptions in supply chain network design models: a review and future research directions. International Journal of Production Economics, 235:108103, 2021.

Andrew An Bian, Baharan Mirzasoleiman, Joachim Buhmann, and Andreas Krause. Guaranteed Nonconvex Optimization: Submodular Maximization over Continuous Domains. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, April 2017.

Yatao Bian, Joachim Buhmann, and Andreas Krause. Optimal continuous DR-submodular maximization and applications to provable mean field inference. In Proceedings of the 36th International Conference on Machine Learning, June 2019.

Dan Garber and Elad Hazan. A linearly convergent variant of the conditional gradient algorithm under strong convexity, with applications to online and stochastic optimization. SIAM Journal on Optimization, 26(3):1493–1528, 2016.

Dan Garber and Ben Kretzu. New projection-free algorithms for online convex optimization with adaptive regret guarantees. In Conference on Learning Theory, pages 2326–2359. PMLR, 2022.

Shuyang Gu, Chuangen Gao, Jun Huang, and Weili Wu. Profit maximization in social networks and non-monotone DR-submodular maximization. Theoretical Computer Science, 957:113847, 2023.

Hamed Hassani, Mahdi Soltanolkotabi, and Amin Karbasi. Gradient methods for submodular maximization. In Advances in Neural Information Processing Systems, 2017.

Elad Hazan and Satyen Kale. Projection-free online learning. In Proceedings of the 29th International Coference on Machine Learning, 2012.

Shinji Ito and Ryohei Fujimaki. Large-scale price optimization via network flow. In Advances in Neural Information Processing Systems, 2016.

Kfir Y Levy and Andreas Krause. Projection free online learning over smooth sets. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1458–1466. PMLR, 2019.

Dan Li, Kerry D Wong, Yu Hen Hu, and Akbar M Sayeed. Detection, classification, and tracking of targets. IEEE signal processing magazine, 19(2):17–29, 2002.

Yucheng Liao, Yuanyu Wan, Chang Yao, and Mingli Song. Improved projection-free online continuous submodular maximization. arXiv preprint arXiv:2305.18442, 2023.

Yiyang Lu, Mohammad Pedramfar, and Vaneet Aggarwal. Decentralized projection-free online upperlinearizable optimization with applications to DR-submodular optimization. Transactions on Machine Learning Research, 2025.

Yiyang Lu, Haresh Jadav, Mohammad Pedramfar, Ranveer Singh, and Vaneet Aggarwal. Upper-linearizability of online non-monotone dr-submodular maximization over down-closed convex sets. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Zakaria Mhammedi. Eficient projection-free online convex optimization with membership oracle. In Conference on Learning Theory, pages 5314–5390. PMLR, 2022.

Zakaria Mhammedi. Online convex optimization with a separation oracle. In Proceedings of the 38th Annual Conference on Learning Theory, pages 1–45, 2025.

Sivkumar Mishra, Debapriya Das, and Subrata Paul. A comprehensive review on power distribution network reconfiguration. Energy Systems, 8:227–284, 2017.

Siddharth Mitra, Moran Feldman, and Amin Karbasi. Submodular+ concave. In Advances in Neural Information Processing Systems, 2021.

Aryan Mokhtari, Hamed Hassani, and Amin Karbasi. Decentralized submodular maximization: Bridging discrete and continuous settings. In International conference on machine learning, pages 3616–3625, 2018.

Arkadi S Nemirovski and Michael J Todd. Interior-point methods for optimization. Acta Numerica, 17: 191–234, 2008.

Mohammad Pedramfar and Vaneet Aggarwal. From linear to linearizable optimization: A novel framework with applications to stationary and non-stationary dr-submodular optimization. In Advances in Neural Information Processing Systems, 2024.

Mohammad Pedramfar, Christopher Quinn, and Vaneet Aggarwal. Uniform wrappers: Bridging concave to quadratizable functions in online optimization. In Advances in Neural Information Processing Systems, 2025.

Yuanyu Wan, Yu Shen, Dingzhi Yu, Bo Xue, and Mingli Song. Improved approximate regret for decentralized online continuous submodular maximization via reductions. arXiv preprint arXiv:2602.09502, 2026.

Lin Xiao, Stephen Boyd, and Seung-Jean Kim. Distributed average consensus with least-mean-square deviation. Journal of parallel and distributed computing, 67(1):33–46, 2007.

Qixin Zhang, Zengde Deng, Xiangru Jian, Zaiyi Chen, Haoyuan Hu, and Yu Yang. Communication-eficient decentralized online continuous dr-submodular maximization. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 3330–3339, 2023.

Junlong Zhu, Qingtao Wu, Mingchuan Zhang, Ruijuan Zheng, and Keqin Li. Projection-free decentralized online learning for submodular maximization over time-varying networks. Journal of Machine Learning Research, 22(51):1–42, 2021.

## A Approximate Gauge Projection

This appendix records the separation-based approximate gauge projection subroutine (Algorithm 2) used by the main algorithm, whose construction and guarantees are illustrated by Mhammedi (2025), and no novelty is claimed for this subroutine. We state it for the translated set $\mathcal { C } = \mathcal { K } - x _ { 0 }$ , for which rB $\subseteq { \mathcal { C } } \subseteq R \mathbb { B } _ { 2 }$ Lemma 8 describes useful properties of the gauge function and gauge distance, and Lemma 9 describes the feasibility guarantee and budget of calls to the SO.

```latex
Algorithm 2 GaugeDist: approximate gauge distance and subgradient
Require: Separation oracle Sep<sub>C</sub>, query $u \in \mathbb { R } ^ { d }$ , precision $\varepsilon _ { \mathrm { g a u } } \in ( 0 , 1 ] .$ , and inner radius r
1: $( b , v )  \mathrm { S e p } _ { \mathit { c } } ( u )$
2: if b = 1 then
3: return $( S , s ) = ( 0 , 0 )$
4: end if
5: α $ 0 , \beta  1 , \zeta  ( \alpha + \beta ) / 2$
6: while $\beta - \alpha > r ^ { 2 } \varepsilon _ { \mathrm { g a u } } / ( 2 \Vert u \Vert _ { 2 } ^ { 2 } )$ do
7: $( b , v _ { \mathrm { n e w } } ) \gets \mathrm { S e p } _ { \mathscr C } ( \zeta u )$
8: if b = 1 then
9: $\alpha  \zeta$
10: else
11: $\beta \gets \zeta$ and $v  v _ { \mathrm { n e w } }$
12: end if
13: $\zeta \gets ( \alpha + \beta ) / 2$
14: end while
15: $S \gets \alpha ^ { - 1 } - 1$ and $s \gets v / ( \beta \langle v , u \rangle )$
16: return (S, s)
```

As introduced in Preliminaries, this algorithm is built on gauge function and gauge distance

Lemma 8 (Gauge properties). The function $\gamma _ { \mathcal { C } }$ is convex and positively homogeneous. Moreover,

$$
\gamma _ { \mathcal { C } } ( u ) \leq \frac { \| u \| _ { 2 } } { r } , \qquad S _ { \mathcal { C } } ( u ) = \operatorname* { m a x } \{ 0 , \gamma _ { \mathcal { C } } ( u ) - 1 \} .
$$

Every subgradient s of γ<sub>C</sub> has Euclidean norm at most $1 / r$

Proof. These are the standard gauge properties collected in Mhammedi (2025, Lemma 2.1 and Appendix G). They apply to C because the translated geometry places the origin in its interior and contains $r \mathbb { B } _ { 2 } .$ □

Lemma 9 (Approximate gauge guarantee and SO call budget). For any $u \in R \mathbb { B } _ { 2 }$ and precision $ { \varepsilon } _ { \mathrm { g a u } } \in ( 0 , 1 ]$ Algorithm 2 returns (S, s) satisfying

$$
S _ { C } ( u ) \leq S \leq S _ { C } ( u ) + \varepsilon _ { \mathrm { g a u } } , \quad a n d \quad \| s \| _ { 2 } \leq { \frac { 1 } { r } } , \quad a n d \quad S _ { C } ( v ) \geq S _ { C } ( u ) + \langle s , v - u \rangle - \varepsilon _ { \mathrm { g a u } } , \forall v \in \mathbb { R } ^ { d } .
$$

If $u \in { \mathcal { C } }$ , the algorithm terminates after one separation-oracle call. If u /∈ C, it makes at most

$$
1 + \left\lceil \log _ { 2 } \left( \frac { 4 \| u \| _ { 2 } ^ { 2 } } { r ^ { 2 } \varepsilon _ { \mathrm { g a u } } } \right) \right\rceil \leq 1 + \left\lceil \log _ { 2 } \left( \frac { 4 \kappa ^ { 2 } } { \varepsilon _ { \mathrm { g a u } } } \right) \right\rceil\tag{20}
$$

separation-oracle calls. In particular, the right hand side expression is a uniform upper bound in either case.

Proof. The three approximation properties and the exterior-query call count are exactly Mhammedi (2025, Lemma 2.2), applied to C. An interior query returns immediately. For an exterior query, $\| u \| _ { 2 } > r ,$ so the displayed logarithm is positive; its second inequality follows from $\| u \| _ { 2 } \leq R$ and $\kappa = R / r$ □

At the precision $\varepsilon _ { \mathrm { g a u } } = 1 / T$ used by $\mathrm { D e c - B F T R L }$ , the per-agent budget over the horizon is therefore

$$
T \left( 1 + \left\lceil \log _ { 2 } ( 4 \kappa ^ { 2 } T ) \right\rceil \right) = O ( T \log ( \kappa T ) ) .
$$

The network-wide budget is N times this quantity.

## B HybridNewton: Controlled Approximate Local Barrier-FTRL Solver

This appendix defines and analyzes HybridNewton, the local subroutine used by Dec-BFTRL to approximately minimize a local barrier-FTRL potential. The subroutine is a damped/full Newton implementation equipped with control over a computable Euclidean-accuracy. Its role here is to provide the local-solve guarantee required by the regret analysis, and it uses no communication, feedback-oracle calls, or action-set-oracle calls.

For a target dual vector $\theta \in \mathbb { R } ^ { d }$ , regularization parameters $\mu > 0$ and $\nu \geq 1$ , and radius $R > 0$ , define the generic barrier-FTRL potential for u $\in \operatorname { i n t } ( R \mathbb { B } _ { 2 } )$ by

$$
F ( u ) : = - \nu \log ( R ^ { 2 } - \| u \| _ { 2 } ^ { 2 } ) + \frac { \mu } { 2 } \| u \| _ { 2 } ^ { 2 } + \langle \theta , u \rangle .\tag{21}
$$

Set $m : = \mu + 2 \nu / R ^ { 2 }$ . Then $\nabla ^ { 2 } F ( u ) \succeq m I$ . Since F diverges at the boundary, it has a unique interior minimizer

$$
u ^ { \star } : = \arg \operatorname* { m i n } _ { u \in \mathrm { i n t } ( R \mathbb { B } _ { 2 } ) } F ( u ) .
$$

For an interior point $u ,$ define the Newton decrement of $F$ at u by

$$
\lambda ( u ) : = \| \nabla F ( u ) \| _ { [ \nabla ^ { 2 } F ( u ) ] ^ { - 1 } } .
$$

Given a prescribed Euclidean accuracy $\varepsilon > 0$ , define the stopping threshold

$$
\tau _ { \varepsilon } : = \frac { \sqrt { m } \varepsilon } { 1 + \sqrt { m } \varepsilon } .\tag{22}
$$

We record the standard self-concordant facts used in the solver analysis. For $H ( u ) : = \nabla ^ { 2 } F ( u )$ , write

$$
\| h \| _ { \boldsymbol { u } } : = \sqrt { h ^ { \top } H ( \boldsymbol { u } ) h } , \qquad \| \boldsymbol { z } \| _ { \boldsymbol { u } , * } : = \sqrt { \boldsymbol { z } ^ { \top } H ( \boldsymbol { u } ) ^ { - 1 } \boldsymbol { z } } .
$$

Thus $\lambda ( u ) = \| \nabla F ( u ) \| _ { u , * }$ , and the unit Dikin ellipsoid at u is given by

$$
\mathcal { E } _ { u } : = \{ u + h : \| h \| _ { u } < 1 \} .
$$

For $\nu \geq 1$ , the logarithmic term in (21) is self-concordant on int $\left( R \mathbb { B } _ { 2 } \right)$ . Adding the convex quadratic and linear terms preserves self-concordance on the same open domain, and the resulting F still diverges at the boundary (Nemirovski and Todd, 2008).

Standard self-concordant analysis gives $\mathcal { E } _ { u } \subseteq$ dom F. If $\lambda ( u ) < 1$ , then

$$
\Vert u - u ^ { \star } \Vert _ { u } \leq \frac { \lambda ( u ) } { 1 - \lambda ( u ) } .\tag{23}
$$

Moreover, the damped update in (26) satisfies

$$
F ( u ^ { + } ) \leq F ( u ) - \big ( \lambda ( u ) - \log ( 1 + \lambda ( u ) ) \big ) ,\tag{24}
$$

and the full update in (27), whenever $\lambda ( u ) < 1$ , satisfies

$$
\lambda ( u ^ { + } ) \leq \left( \frac { \lambda ( u ) } { 1 - \lambda ( u ) } \right) ^ { 2 } .\tag{25}
$$

These are cited standard properties rather than new claims of this paper. The computable stopping test $\lambda ( u ) \le \tau _ { \varepsilon }$ guarantees the requested Euclidean accuracy, as shown below.

Algorithm 3 HybridNewton: Local Barrier-FTRL Solver with Guaranteed Accuracy   
Require: Potential F of the form (21), interior initialization $u _ { 0 }$ , and accuracy $\varepsilon > 0$   
Ensure: $u ^ { + }$ satisfying $\lambda ( u ^ { + } ) \leq \tau _ { \varepsilon }$ and hence $\| u ^ { + } - u ^ { \star } \| _ { 2 } \leq \varepsilon$   
1: $u  u _ { 0 }$   
2: $\tau _ { \varepsilon } \gets \sqrt { m } \varepsilon / ( 1 + \sqrt { m } \varepsilon )$   
3: while $\lambda ( u ) > 1 / 4$ do   
4: $\begin{array} { r } { u  u - \frac { 1 } { 1 + \lambda ( u ) } [ \nabla ^ { 2 } F ( u ) ] ^ { - 1 } \nabla F ( u ) } \end{array}$   
5: end while   
6: while $\lambda ( u ) > \tau _ { \varepsilon }$ do   
$\begin{array} { r l } { \mathrm { ~  ~ \omega ~ } _ { 7 : } } & { { } \mathrm { ~  ~ \psi ~ } _ { u } \gets u - [ \nabla ^ { 2 } F ( u ) ] ^ { - 1 } \nabla F ( u ) } \end{array}$   
8: end while   
9: $u ^ { + }  u$   
10: return $u ^ { + }$

The next two lemmas isolate the safety and accuracy facts used in the main analysis. Proposition 1 then summarizes finite termination and the guaranteed output of the complete subroutine. The final lemma records the cost of one exact Newton iteration.

Lemma 10 (Interior safety). Suppose $\nu \geq 1$ and $u \in$ int $\left( R \mathbb { B } _ { 2 } \right)$ . The damped Newton update

$$
u ^ { + } = u - \frac { [ \nabla ^ { 2 } F ( u ) ] ^ { - 1 } \nabla F ( u ) } { 1 + \lambda ( u ) }\tag{26}
$$

remains in int $\left( R \mathbb { B } _ { 2 } \right)$ . The full Newton update

$$
\boldsymbol { u } ^ { + } = \boldsymbol { u } - [ \nabla ^ { 2 } F ( \boldsymbol { u } ) ] ^ { - 1 } \nabla F ( \boldsymbol { u } )\tag{27}
$$

also remains in the domain whenever $\lambda ( u ) < 1$

Proof. If d is the damped displacement, then

$$
\| d \| _ { \nabla ^ { 2 } F ( u ) } = \frac { \lambda ( u ) } { 1 + \lambda ( u ) } < 1 .
$$

Hence $u + d \in \mathcal { E } _ { u } \subseteq \mathrm { d o m } F$ . For the full step, the displacement has local norm $\lambda ( u ) < 1$ , and the same argument applies. □

Lemma 11 (Euclidean accuracy from the decrement). If u is interior and $\lambda ( u ) \leq \delta < 1$ , then

$$
\lVert u - u ^ { \star } \rVert _ { 2 } \leq \frac { \delta } { ( 1 - \delta ) \sqrt { m } } .
$$

In particular, setting $\delta = \tau _ { \varepsilon }$ from (22) gives

$$
\| u ^ { + } - u ^ { \star } \| _ { 2 } \leq \varepsilon .\tag{28}
$$

Proof. The bound (23) and monotonicity of $x / ( 1 - x )$ on [0, 1) give $\| u - u ^ { \star } \| _ { \nabla ^ { 2 } F ( u ) } \le \delta / ( 1 - \delta )$ . Since $\nabla ^ { 2 } F ( u ) \succeq m I$ , this implies the displayed Euclidean bound. Substitution of (22) gives (28). □

Proposition 1 (Guaranteed finite termination). For every potential F of the form (21), every interior initialization $u _ { 0 . }$ , and every accuracy $\varepsilon > 0 _ { ; }$ , Algorithm 3 is well defined and terminates after finitely many Newton updates. Every iterate remains in int $\left( R \mathbb { B } _ { 2 } \right)$ , and its output satisfies

$$
\lambda ( u ^ { + } ) \leq \tau _ { \varepsilon } , \qquad \| u ^ { + } - u ^ { \star } \| _ { 2 } \leq \varepsilon .
$$

Proof. Let $\omega ( s ) : = s - \log ( 1 + s )$ . During the first loop, Lemma 10 makes every damped update well defined and keeps the new iterate interior. By (24), each such update with $\lambda ( u ) > 1 / 4$ decreases F by at least $\omega ( 1 / 4 ) > 0$ . Since $F$ is bounded below by $F ( u ^ { \star } )$ , the first loop terminates after finitely many updates with $\lambda ( u ) \leq 1 / 4$

If $\tau _ { \varepsilon } \geq 1 / 4$ , the second loop is skipped. Otherwise, every full update in that loop remains interior by Lemma 10, and (25) yields

$$
\lambda ( u ^ { + } ) \leq \left( { \frac { \lambda ( u ) } { 1 - \lambda ( u ) } } \right) ^ { 2 } \leq 2 \lambda ( u ) ^ { 2 } \qquad { \mathrm { w h e n e v e r ~ } } \lambda ( u ) \leq { \frac { 1 } { 4 } } .
$$

Writing $b _ { k } : = 2 \lambda ( u _ { k } )$ for the full-step iterates gives $b _ { k + 1 } \leq b _ { k } ^ { 2 }$ and $b _ { 0 } \leq 1 / 2 ,$ , hence $b _ { k } \leq 2 ^ { - 2 ^ { k } }$ . Therefore the positive threshold $\tau _ { \varepsilon }$ is reached after finitely many full steps. The stopping test gives $\lambda ( u ^ { + } ) \leq \tau _ { \varepsilon }$ , and Lemma 11 gives $\| u ^ { + } - u ^ { \star } \| _ { 2 } \leq \varepsilon$ □

For the calls made by Dec-BFTRL, take $F = \Psi _ { a , t } , \nu = 1 , \mu = \mu _ { T } , m = m _ { T } , \theta = \theta _ { a , t } , u _ { 0 } = u _ { a , t - 1 }$ , and $\varepsilon = R / T$ for $a \in \left\lceil N \right\rceil$ and $t = 2 , \ldots , T$ . The convention $\varepsilon _ { a , 1 } = 0$ is only bookkeeping: no solver call is made at $t = 1$ , where $u _ { a , 1 } = 0$ is the exact minimizer of the initial potential. Thus each agent invokes HybridNewton exactly $T - 1$ times.

Lemma 12 (Structured Newton direction). For $u \in \mathrm { i n t } ( R \mathbb { B } _ { 2 } )$ , set

$$
D ( u ) : = R ^ { 2 } - \| u \| _ { 2 } ^ { 2 } , a ( u ) : = \mu + \frac { 2 \nu } { D ( u ) } , b ( u ) : = \frac { 4 \nu } { D ( u ) ^ { 2 } } .
$$

Then

$$
\boldsymbol { \nabla } F ( \boldsymbol { u } ) = \boldsymbol { a } ( \boldsymbol { u } ) \boldsymbol { u } + \boldsymbol { \theta } , \qquad \boldsymbol { \nabla } ^ { 2 } F ( \boldsymbol { u } ) = \boldsymbol { a } ( \boldsymbol { u } ) \boldsymbol { I } + \boldsymbol { b } ( \boldsymbol { u } ) \boldsymbol { u } \boldsymbol { u } ^ { \intercal } ,
$$

and, for every $v \in \mathbb { R } ^ { d }$

$$
[ \nabla ^ { 2 } F ( u ) ] ^ { - 1 } v = \frac { v } { a ( u ) } - \frac { b ( u ) u \langle u , v \rangle } { a ( u ) \big ( a ( u ) + b ( u ) \| u \| _ { 2 } ^ { 2 } \big ) } .
$$

Taking $v = \nabla F ( u )$ , both the Newton direction and $\lambda ( u ) ^ { 2 } = \nabla F ( u ) ^ { \top } [ \nabla ^ { 2 } F ( u ) ] ^ { - 1 } \nabla F ( u )$ use $O ( d )$ arithmetic and $O ( d )$ working memory. Thus one exact HybridNewton iteration has those costs.

Proof. Diferentiate the potential in (21). The inverse formula is the Sherman-Morrison identity applied to $a ( u ) I + b ( u ) u u ^ { \top }$ □

## C Proof of Lemma 1

Proof of Lemma 1. Suppress $j , t .$ , and write $u , S , s , p , \ell$ for the corresponding quantities. Lemma 8 and 9 give, by positive homogeneity,

$$
\gamma c ( p ) = \frac { \gamma c ( u ) } { 1 + S } \leq \frac { 1 + S c ( u ) } { 1 + S } \leq 1 .
$$

Thus $p \in \mathcal { C } , w = x _ { 0 } + p \in \mathcal { K }$ , and, by Assumption $1 , \| p \| _ { 2 } \leq R$ . Moreover, $\| \ell \| _ { 2 } = \| q \| _ { 2 } \leq G$ by Assumption 3. Let $\chi : = \mathbf { 1 } \{ \langle \ell , u \rangle < 0 \}$ . Because p is a positive multiple of $u , - \chi \langle \ell , p \rangle \geq 0$ . Hence

$$
\| \widetilde { \ell } \| _ { 2 } \leq \| \ell \| _ { 2 } + | \langle \ell , p \rangle | \| s \| _ { 2 } \leq G + \frac { G R } { r } \leq 2 \kappa G ,
$$

where $\| s \| _ { 2 } \leq 1 / r$ follows from Lemma 9, and $r \leq R$ follows from Assumption 1.

Define $L ( z ) : = \langle \ell , z \rangle - \chi \langle \ell , p \rangle S c ( z )$ . Since $S c ( v ) = 0$ for $v \in { \mathcal { C } }$ , the approximate subgradient inequality for s in Lemma 9 implies

$$
L ( u ) - L ( v ) \leq \langle \widetilde { \ell } , u - v \rangle + \frac { | \langle \ell , p \rangle | } { T } .\tag{29}
$$

We also have

$$
L ( u ) \geq \langle \ell , p \rangle - \frac { G R } { T } .\tag{30}
$$

because if $S = 0$ , then $p = u$ . If $S > 0$ and $\chi = 0$ , then $\langle \ell , u \rangle \geq 0$ and $\langle \ell , p \rangle \leq L ( u )$ . If $S > 0$ and $\chi = 1$ , then $\begin{array} { r } { L ( u ) - \langle \ell , p \rangle = \langle \ell , p \rangle \big ( S - S _ { \mathcal { C } } ( u ) \big ) \geq - \frac { G R } { T } } \end{array}$ . Combining (29) and (30), using $| \langle \ell , p \rangle | \leq G R$ and $w - ( x _ { 0 } + v ) = p - v$ , we have

$$
\left. \ell , w - ( x _ { 0 } + v ) \right. = \left. \ell , p - v \right. \leq \left. \widetilde { \ell } , u - v \right. + \frac { 2 G R } { T } .
$$

## D Proof of Lemma 2

Proof of Lemma 2. By Assumption 2, the a-th rows of $W ^ { k }$ and J are probability vectors, so their ℓ<sub>1</sub>-distance is at most 2. Also, by Cauchy–Schwarz,

$$
\sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| \leq \sqrt { N } \left( \sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| ^ { 2 } \right) ^ { 1 / 2 } \leq \sqrt { N } \| W ^ { k } - J \| _ { 2 } ,
$$

where $J : = \mathbf { 1 1 } ^ { \top } / N$ . For $k = 0 , \| W ^ { 0 } - J \| _ { 2 } \leq 1$ . For $k \geq 1$ , double stochasticity gives $W J = J W = J ,$ and hence

$$
W ^ { k } - J = ( W - J ) ^ { k } , \qquad \| W ^ { k } - J \| _ { 2 } \leq \rho ^ { k } .
$$

Combining the two conditions of k,

$$
\sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| \leq \operatorname* { m i n } \{ 2 , \sqrt { N } \rho ^ { k } \}
$$

For $0 < \rho < 1$ , split the series in (6) at $\begin{array} { r } { k _ { 0 } : = \left\lceil \frac { [ \log ( \sqrt { N } / 2 ) ] + } { - \log \rho } \right\rceil } \end{array}$ to obtain

$$
M _ { a } ( W ) \leq 2 k _ { 0 } + \sqrt { N } \sum _ { k = k _ { 0 } } ^ { \infty } \rho ^ { k } \leq 2 k _ { 0 } + \frac { 2 } { 1 - \rho } .
$$

If $\rho = 0$ , then $W = J$ , so only $k = 0$ contributes and

$$
M _ { a } ( W ) = \sum _ { b = 1 } ^ { N } \left| \mathbf { 1 } \{ a = b \} - { \frac { 1 } { N } } \right| = 2 \left( 1 - { \frac { 1 } { N } } \right) .
$$

Finally, − log $\rho \ge 1 - \rho$ for $0 < \rho < 1$ , which gives the stated order bound together with the separate $\rho = 0$ case. □

## E Proof of Lemma 3

Proof of Lemma 3. Double stochasticity in Assumption 2 and (5) give $\bar { \theta } _ { t + 1 } = \bar { \theta } _ { t } + \bar { \ell } _ { t }$ . The zero initialization proves (8).

Stack $\theta _ { b , t } ^ { \top }$ as the rows of Θ , and stack $\widetilde { \ell } _ { b , t } ^ { \top }$ as the rows of $\scriptstyle { L _ { t } }$ . With $J = \mathbf { 1 1 } ^ { \top } / N$ , Lemma 1 gives $\Vert \mathbf { \cal L } _ { t } \Vert _ { \mathrm { F } } \leq \sqrt { N } \widetilde { G }$ Moreover, $W J = J W = J$ by Assumption 2, so

$$
( I - J ) \Theta _ { t + 1 } = ( W - J ) ( I - J ) \Theta _ { t } + ( I - J ) L _ { t } .
$$

Therefore

$$
\Vert ( I - J ) \Theta _ { t + 1 } \Vert _ { \mathrm { F } } \leq \rho \Vert ( I - J ) \Theta _ { t } \Vert _ { \mathrm { F } } + \sqrt { N } \widetilde { G } .
$$

Unrolling from $\Theta _ { 1 } = 0$ proves (9).

The same recursion gives

$$
\Theta _ { t } = \sum _ { s < t } W ^ { t - 1 - s } \pmb { L } _ { s } , \qquad J \Theta _ { t } = \sum _ { s < t } J \pmb { L } _ { s } ,
$$

and therefore

$$
\theta _ { a , t } - \bar { \theta } _ { t } = \sum _ { s < t } \sum _ { b = 1 } ^ { N } \left( ( W ^ { t - 1 - s } ) _ { a b } - \frac { 1 } { N } \right) \widetilde { \ell } _ { b , s } .
$$

Using $\| \widetilde { \ell } _ { b , s } \| _ { 2 } \leq \widetilde { G }$ , summing over t, and changing the order of summation give

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } d _ { a , t } \leq \tilde { G } \sum _ { t = 2 } ^ { T } \sum _ { k = 0 } ^ { t - 2 } \sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| } \\ & { \quad \quad \quad \quad = \tilde { G } \displaystyle \sum _ { k = 0 } ^ { T - 2 } ( T - 1 - k ) \sum _ { b = 1 } ^ { N } \left| ( W ^ { k } ) _ { a b } - \frac { 1 } { N } \right| } \\ & { \quad \quad \quad \leq T \tilde { G } M _ { a } ( W ) , } \end{array}
$$

which proves (10). By Cauchy–Schwarz and (9),

$$
\frac { 1 } { N } \sum _ { b = 1 } ^ { N } d _ { b , t } \leq \frac { \widetilde { G } ( 1 - \rho ^ { t - 1 } ) } { \delta } ,
$$

and summing over t proves (11).

## F Proof of Lemma 4 and Lemma 5

Proof of Lemma $\it 4 .$ The local and network-average potentials have Hessians bounded below by m I, and their gradients difer only by $\theta _ { a , t } - \bar { \theta } _ { t }$ . Strong monotonicity of $\nabla \Psi _ { a , t }$ , together with the first-order conditions at $u _ { a , t } ^ { \star }$ and $\bar { u } _ { t } ^ { \star }$ , gives

$$
m _ { T } \| u _ { a , t } ^ { \star } - \bar { u } _ { t } ^ { \star } \| _ { 2 } \leq \| \nabla \Psi _ { a , t } ( \bar { u } _ { t } ^ { \star } ) - \nabla \Psi _ { a , t } ( u _ { a , t } ^ { \star } ) \| _ { 2 } = \| \theta _ { a , t } - \bar { \theta } _ { t } \| _ { 2 } .
$$

Proof of Lemma ${ 5 . }$ Equations (8) and (12) give $\bar { \theta } _ { 1 } = 0$ and $\bar { u } _ { 1 } ^ { \star } = 0$ , and make $\bar { u } _ { t } ^ { \star }$ the FTRL iterate generated by the regularizer $\Psi _ { 1 }$ and the past vectors $\bar { \ell } _ { 1 } , \dots , \bar { \ell } _ { t - 1 }$ . The be-the-leader inequality gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \langle \bar { \ell } _ { t } , \bar { u } _ { t + 1 } ^ { \star } - v \rangle \leq } \end{array}$ $\bar { \Psi } _ { 1 } ( v ) - \bar { \Psi } _ { 1 } ( 0 )$ . The same strong-monotonicity argument as in Lemma 4 yields $\begin{array} { r } { \| \bar { u } _ { t } ^ { \star } - \bar { u } _ { t + 1 } ^ { \star } \| _ { 2 } \leq \frac { \| \bar { \ell } _ { t } \| _ { 2 } } { m _ { T } } } \end{array}$ . Lemma 1 gives $\begin{array} { r } { \| \bar { \ell } _ { t } \| _ { 2 } \leq \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| \widetilde { \ell } _ { j , t } \| _ { 2 } \leq \widetilde { G } } \end{array}$ . Therefore, $\begin{array} { r } { \langle \bar { \ell } _ { t } , \bar { u } _ { t } ^ { \star } - \bar { u } _ { t + 1 } ^ { \star } \rangle \leq \frac { \widetilde { G } ^ { 2 } } { m _ { T } } } \end{array}$ . Adding these inequalities proves the lemma. □

## G Proof of Lemma 6

Proof of Lemma 6. ${ \mathrm { A t ~ } } t = 1 , \theta _ { a , 1 } = 0$ and $u _ { a , 1 } = 0 = u _ { a , 1 } ^ { \star }$ . Consider a call at $t \geq 2$ , and instantiate Appendix B with $F = \Psi _ { a , t } , \nu = 1 , \mu = \mu _ { T } , m = m _ { T } , \theta = \theta _ { a , t } .$ , and $\varepsilon = \varepsilon _ { a , t }$ . Write λ for the corresponding Newton decrement. Its stopping threshold (22) is

$$
\tau _ { a , t } : = \frac { \sqrt { m _ { T } } \varepsilon _ { a , t } } { 1 + \sqrt { m _ { T } } \varepsilon _ { a , t } } \in ( 0 , 1 ) .
$$

By induction, the warm start $u _ { a , t - 1 }$ is interior. While $\lambda > 1 / 4$ , Lemma 10 keeps each damped step interior, and (24), together with monotonicity of $\omega ( s ) : = s - \log ( 1 + s )$ , decreases the fixed potential $\Psi _ { a , t }$ by at least

$\omega ( 1 / 4 ) > 0$ . Since $\Psi _ { a , t }$ is bounded below by its minimum, this phase is finite. Once $\lambda \le 1 / 4$ , Lemma 10 keeps every full step interior, and (25) gives

$$
\lambda ^ { + } \leq \left( \frac { \lambda } { 1 - \lambda } \right) ^ { 2 } \leq 2 \lambda ^ { 2 } .
$$

If $\tau _ { a , t } \geq 1 / 4$ , the stopping condition already holds. Otherwise, for the full-step decrements set $b _ { k } : = 2 \lambda _ { k }$ Then $b _ { k + 1 } \leq b _ { k } ^ { 2 }$ and $b _ { 0 } \leq 1 / 2 \quad$ , so $b _ { k } \leq 2 ^ { - 2 ^ { k } }$ ; hence $\lambda _ { k } \le \tau _ { a , t }$ <sub>t</sub> after finitely many steps. Lemma 11, applied with $m = m _ { T }$ and its threshold variable $\delta = \tau _ { a , t }$ , now gives (13). The returned point is interior and is therefore a valid warm start for the next call

Finally, $\varepsilon _ { a , 1 } = 0$ and $\varepsilon _ { a , t } = R / T$ for $2 \leq t \leq T$ , so

$$
E _ { a , T } = ( T - 1 ) \frac { R } { T } < R .
$$

Averaging over agents proves the claim for $E _ { T }$

## H Proof of Lemma 7

Proof of Lemma 7. We first derive the two properties of the exact radial projection needed below. By Lemma $^ { 8 , }$ the inclusion $r \mathbb { B } _ { 2 } \subseteq \mathcal { C }$ in Assumption 1 makes $\gamma _ { \mathcal { C } }$ , and hence $S c , 1 / r \mathrm { - I }$ Lipschitz. For $u , v \in R \mathbb { B } _ { 2 }$ , set

$$
A : = 1 + S c ( u ) , \qquad B : = 1 + S c ( v ) .
$$

Since $p c ( v ) \in \mathcal { C } \subseteq R \mathbb { B } _ { 2 }$ 2

$$
\| p _ { C } ( u ) - p _ { C } ( v ) \| _ { 2 } \leq \frac { \| u - v \| _ { 2 } } { A } + \frac { \| v \| _ { 2 } | A - B | } { A B } \leq ( 1 + \kappa ) \| u - v \| _ { 2 } .\tag{31}
$$

By initialization and Lemma $6 , u _ { a , t } \in R \mathbb { B } _ { 2 }$ . Therefore, Lemma 9 with $\varepsilon _ { \mathrm { g a u } } = 1 / T$ gives

$$
\| p _ { a , t } - p _ { \mathcal { C } } ( u _ { a , t } ) \| _ { 2 } = \frac { \| u _ { a , t } \| _ { 2 } | S _ { a , t } - S _ { \mathcal { C } } ( u _ { a , t } ) | } { ( 1 + S _ { a , t } ) ( 1 + S _ { \mathcal { C } } ( u _ { a , t } ) ) } \le \frac { R } { T } .\tag{32}
$$

Equations (31) and (32), followed by Lemmas 6 and 4, imply

$$
\begin{array} { l } { \displaystyle \| w _ { a , t } - z _ { t } \| _ { 2 } \leq \| p _ { a , t } - p _ { \mathcal { C } } ( u _ { a , t } ) \| _ { 2 } + \| p _ { \mathcal { C } } ( u _ { a , t } ) - p _ { \mathcal { C } } ( \bar { u } _ { t } ^ { \star } ) \| _ { 2 } } \\ { \displaystyle \leq \frac { R } { T } + ( 1 + \kappa ) \left( \varepsilon _ { a , t } + \frac { d _ { a , t } } { m _ { T } } \right) . } \end{array}
$$

Applying the triangle inequality through $z _ { t } ,$ averaging over payof owners $j ,$ and summing over t gives

$$
\sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| w _ { a , t } - w _ { j , t } \| _ { 2 } \leq 2 R + ( 1 + \kappa ) ( E _ { a , T } + E _ { T } ) + \frac { 1 + \kappa } { m _ { T } } \left( \sum _ { t = 1 } ^ { T } d _ { a , t } + \sum _ { t = 1 } ^ { T } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } d _ { j , t } \right) .
$$

Using Equations (10) and (11) to bound $\textstyle \sum _ { t = 1 } ^ { T } d _ { a , t }$ and $\textstyle \sum _ { t = 1 } ^ { T } { \frac { 1 } { N } } \sum _ { j = 1 } ^ { N } d _ { j , t }$ respectively to obtain the final bound as stated. □

## I Applications to Up-Concave and DR-Submodular Maximization

We instantiate Theorem 1 through four wrapper pairs covering three standard application families. The upper-linearization inequalities are inherited from prior work; our consequence is their combination with the per-agent decentralized separation-oracle guarantee. Throughout this section, the action set also satisfies Assumption 1: for a known $x _ { 0 } \in \operatorname { i n t } ( K )$ ，

$$
{ \mathcal { C } } = { \mathcal { K } } - x _ { 0 } , \qquad r { \mathbb { B } } _ { 2 } \subseteq { \mathcal { C } } \subseteq R { \mathbb { B } } _ { 2 } , \qquad \kappa = R / r .
$$

Thus an application-specific requirement that $\mathbf { 0 } \in \mathcal { K }$ does not replace the interior center $x _ { 0 }$ used by the gauge algorithm.

Function classes. For vectors, $x \leq y$ denotes coordinatewise order, and $\| x \| _ { \infty } : = \operatorname* { m a x } _ { k } \left| x _ { k } \right|$ . A diferentiable function $f$ is γ-weakly up-concave, for $\gamma \in ( 0 , 1 ]$ , if for every $x \leq y$ in its domain,

$$
\gamma \langle \nabla f ( y ) , y - x \rangle \leq f ( y ) - f ( x ) \leq { \frac { 1 } { \gamma } } \langle \nabla f ( x ) , y - x \rangle .
$$

We use up-concave when $\gamma = 1$ . For a continuous monotone function, its curvature is the smallest $c \in [ 0 , 1 ]$ such that

$$
f ( y + z ) - f ( y ) \geq ( 1 - c ) { \big ( } f ( x + z ) - f ( x ) { \big ) }
$$

whenever $x , y , z \ \geq \mathbf { 0 }$ and all four displayed arguments belong to the domain. These conventions follow Pedramfar and Aggarwal (2024). A diferentiable function is continuous DR-submodular if $\nabla f ( x ) \geq \nabla f ( y )$ coordinatewise whenever $x \leq y$ . A set $\mathcal { K } \subseteq \mathbb { R } _ { \geq 0 } ^ { d }$ is down-closed if $x \in \kappa$ and $\mathbf { 0 } \leq y \leq x$ imply $y \in \kappa$

Assumption 5 (One-query first-order base oracle). There exists $L > 0$ such that, for every diferentiable payof $f$ and query point $z \in \kappa$ , the base oracle $\mathcal { O } _ { f } ( z )$ returns a random vector $V _ { f } ( z )$ satisfying, conditiona on the previously revealed information and on the chosen query point,

$$
\begin{array} { r } { \mathbb { E } [ V _ { f } ( z ) \mid \mathrm { p a s t ~ a n d } \ z ] = \nabla f ( z ) , \qquad \| V _ { f } ( z ) \| _ { 2 } \leq L . } \end{array}
$$

The auxiliary randomness used by the Query Wrapper is sampled as part of the wrapped-response generation, after conditioning on $\mathcal { F } _ { t } ^ { - }$ , and independently of the subsequent base-oracle noise.

Each wrapper below invokes its base oracle exactly once. Following the feedback terminology of Pedramfar et al. (2025) and Lu et al. (2025), the feedback is first-order semi-bandit when the base-oracle query point equals the played action $\mathcal { W } ^ { \mathrm { a c t } } ( w )$ , and first-order full-information when the query point may difer from that action. Here full-information means that one first-order query may be placed at another feasible point; it does not mean that the entire payof function is revealed. We do not use blocking, smoothing, zeroth-order estimators, or other limited-feedback conversions.

For a wrapper $\mathcal { W } = ( \mathcal { W } ^ { \mathrm { a c t } } , \mathcal { W } ^ { \mathrm { q r y } } )$ , write

$$
\widehat { \mathcal { O } } _ { f } : = \mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { f } ) , \qquad q \sim \widehat { \mathcal { O } } _ { f } ( w ) .
$$

When the wrapper is applied to $f _ { t , j }$ at $w _ { j , t }$ , its proxy is

$$
\mathfrak { g } ( f _ { t , j } , w _ { j , t } ) : = \mathbb { E } [ q _ { j , t } \ | \ \mathcal { F } _ { t } ^ { - } ] .
$$

The tower property over the wrapper and base-oracle randomness verifies the conditional-mean requirement in Assumption 3. All four wrapped outputs satisfy $\| q \| _ { 2 } \leq L .$ , so $G = L$ . Moreover,

$$
\| \nabla f ( z ) \| _ { 2 } = \| \mathbb { E } [ V _ { f } ( z ) \mid { \mathrm { p a s t ~ a n d ~ } } z ] \| _ { 2 } \leq L .
$$

Hence $f$ is L-Lipschitz on the convex action set. If the Action Wrapper $h = \mathcal { W } ^ { \mathrm { a c t } }$ is $L _ { h } { \mathrm { - L i p s c h i t z } }$ , then

$$
| f ( h ( w ) ) - f ( h ( v ) ) | \leq L L _ { h } \| w - v \| _ { 2 } .\tag{33}
$$

so Assumption 4 holds with $L _ { \phi } = L L _ { h }$ . Substituting $G = L$ and $L _ { \phi } = L L _ { h }$ into Theorem 1 gives, for every agent $a _ { \mathrm { : } }$

$$
\Re _ { \alpha } ^ { a } ( T ) = \widetilde { O } \left( \beta + L R \left[ \beta \kappa + L _ { h } ( 1 + \kappa ) ( 1 + \log N ) \right] \sqrt { \frac { T } { 1 - \rho } } \right) .\tag{34}
$$

In each corollary below, every local payof $f _ { t , j }$ is assumed to satisfy the corresponding proposition with common class and oracle constants, and the same wrapper is used by every agent and round. In the anchored non-monotone case, the anchor $\underline { { x } }$ is also fixed and shared. Every wrapper makes one local payof-oracle query and no action-set-oracle call. Thus the application instantiations retain one neighbor exchange per round and the per-agent ${ \cal O } ( T \log ( \kappa T ) )$ separation-oracle count of Theorem 1.

## I.1 Monotone Up-Concave Functions over General Convex Sets

Proposition 2 (Identity wrapper for monotone weakly up-concave functions; adapted from Pedramfar and Aggarwal (2024)). Let $f : [ 0 , 1 ] ^ { d } \to { \mathbb { R } } _ { \geq 0 }$ be diferentiable, monotone, and γ-weakly up-concave $f o r \gamma \in ( 0 , 1 ]$ with curvature at most $c \in [ 0 , 1 ]$ , and let ${ \mathcal { K } } \subseteq [ 0 , 1 ] ^ { d }$ be convex. Define

$$
{ \mathcal W } ^ { \mathrm { a c t } } ( w ) = h ( w ) = w , \qquad \widehat { \mathcal O } _ { f } ( w ) = { \mathcal W } ^ { \mathrm { q r y } } ( { \mathcal O } _ { f } ) ( w ) : = { \mathcal O } _ { f } ( w ) .
$$

Thus $q = V _ { f } ( w )$ and $\mathfrak { g } ( f , w ) = \nabla f ( w )$ . The base-oracle query equals the played action, so this is first-order semi-bandit feedback. For all $w , y \in { \mathcal { K } }$

$$
\frac { \gamma ^ { 2 } } { 1 + c \gamma ^ { 2 } } f ( y ) - f ( h ( w ) ) \leq \frac { \gamma } { 1 + c \gamma ^ { 2 } } \langle \mathfrak { g } ( f , w ) , y - w \rangle .
$$

Consequently, this wrapper instantiates

$$
( \alpha , \beta , G , L _ { \phi } ) = \left( \frac { \gamma ^ { 2 } } { 1 + c \gamma ^ { 2 } } , \frac { \gamma } { 1 + c \gamma ^ { 2 } } , L , L \right) .
$$

Proof. The action and query point both equal $w \in { \mathcal { K } }$ . Assumption 5 gives $\mathfrak { g } ( f , w ) = \nabla f ( w )$ and $\| q \| _ { 2 } \leq L$ The upper-linearization inequality, including its curvature-dependent constants, is Pedramfar and Aggarwal (2024, Lemma 1, specialized to $\mu = 0 )$ . Finally, the identity Action Wrapper is 1-Lipschitz, so (33) gives $L _ { \phi } = L$ □

Corollary 1 (Monotone up-concave regret on a general set). Under Proposition 2 and the remaining assumptions of Theorem 1, every agent $a \in [ N ]$ satisfies

$$
\mathfrak { R } _ { \gamma ^ { 2 } / ( 1 + c \gamma ^ { 2 } ) } ^ { a } ( T ) = \widetilde { O } \left( \frac { \gamma } { 1 + c \gamma ^ { 2 } } + L R \left[ \frac { \gamma \kappa } { 1 + c \gamma ^ { 2 } } + ( 1 + \kappa ) ( 1 + \log N ) \right] \sqrt { \frac { T } { 1 - \rho } } \right) .
$$

Proof. Apply (34) with the tuple in Proposition 2 and $L _ { h } = 1$

## I.2 Monotone Up-Concave Functions over Convex Sets Containing the Origin

When $\mathbf { 0 } \in \mathcal { K }$ , a randomized Query Wrapper yields the curvature-independent coeficient $1 - e ^ { - \gamma }$

Proposition 3 (Boosted wrapper for monotone weakly up-concave functions; adapted from Pedramfar and Aggarwal (2024)). Let $f : [ 0 , 1 ] ^ { d } \to { \mathbb { R } } _ { \geq 0 }$ be diferentiable, monotone, and γ-weakly up-concave $f o r \gamma \in ( 0 , 1 ]$ and let ${ \mathcal { K } } \subseteq [ 0 , 1 ] ^ { d }$ be convex with $\mathbf { 0 } \in \mathcal { K }$ . Define $\mathcal { W } ^ { \mathrm { a c t } } ( w ) = h ( w ) = w$ . The Query Wrapper draws $Z \in [ 0 , 1 ]$ with density

$$
p _ { \gamma } ( z ) = { \frac { \gamma e ^ { \gamma ( z - 1 ) } } { 1 - e ^ { - \gamma } } }
$$

and returns

$$
\begin{array} { r } { q \sim \widehat { \mathcal { O } } _ { f } ( w ) : = \mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { f } ) ( w ) , \qquad q = V _ { f } ( Z w ) . } \end{array}
$$

Then

$$
{ \mathfrak { g } } ( f , w ) = \int _ { 0 } ^ { 1 } p _ { \gamma } ( z ) \nabla f ( z w ) d z .
$$

The query Zw generally difers from the played action w, so this is first-order full-information feedback. For all $w , y \in { \mathcal { K } }$

$$
( 1 - e ^ { - \gamma } ) f ( y ) - f ( h ( w ) ) \leq \frac { 1 - e ^ { - \gamma } } { \gamma } \langle { \mathfrak { g } } ( f , w ) , y - w \rangle .
$$

Consequently,

$$
( \alpha , \beta , G , L _ { \phi } ) = \left( 1 - e ^ { - \gamma } , \frac { 1 - e ^ { - \gamma } } { \gamma } , L , L \right) .
$$

Proof. Convexity and $\mathbf { 0 } \in \mathcal { K }$ imply $Z w \in \mathcal { K }$ . Conditional unbiasedness of the base oracle followed by the tower property over $Z$ gives the displayed proxy, while $\| q \| _ { 2 } \leq L$ . The boosted upper-linearization inequality is Pedramfar and Aggarwal (2024, Lemma 2 and Algorithm 2). The identity Action Wrapper has $L _ { h } = 1$ , so $L _ { \phi } = L$ □

Corollary 2 (Monotone up-concave regret when $\mathbf { 0 } \in \kappa )$ . Under Proposition 3 and the remaining assumptions of Theorem 1, every agent $a \in [ N ]$ satisfies

$$
\mathfrak { R } _ { 1 - e ^ { - \gamma } } ^ { a } ( T ) = \widetilde { O } \left( \frac { 1 - e ^ { - \gamma } } { \gamma } + L R \left[ \frac { ( 1 - e ^ { - \gamma } ) \kappa } { \gamma } + ( 1 + \kappa ) ( 1 + \log N ) \right] \sqrt { \frac { T } { 1 - \rho } } \right) .
$$

Proof. Apply (34) with the tuple in Proposition 3 and $L _ { h } = 1$

## I.3 Non-Monotone Up-Concave Functions over General Convex Sets

For a non-monotone payof, the Action Wrapper contracts the internal point toward a fixed feasible anchor. Proposition 4 (Anchored wrapper for non-monotone up-concave functions; adapted from Pedramfar and Aggarwal (2024)). Let $f : [ 0 , 1 ] ^ { d } \to { \mathbb { R } } _ { \geq 0 }$ be diferentiable and up-concave, without a monotonicity assumption, and let ${ \mathcal { K } } \subseteq [ 0 , 1 ] ^ { d }$ be convex. Fix a known $\underline { { x } } \in \mathcal { K }$ with $\| \underline { { x } } \| _ { \infty } < 1$ and define

$$
{ \mathcal W } ^ { \mathrm { a c t } } ( w ) = h ( w ) = \frac { w + \underline { x } } { 2 } .
$$

The Query Wrapper draws $Z \in [ 0 , 1 ]$ with density

$$
p _ { \mathrm { n m } } ( z ) = \frac { 1 } { 3 ( 1 - z / 2 ) ^ { 3 } }
$$

and returns

$$
q \sim \widehat { \mathcal { O } } _ { f } ( w ) : = \mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { f } ) ( w ) , \qquad q = V _ { f } \left( \underline { { x } } + \frac { Z } { 2 } ( w - \underline { { x } } ) \right) .
$$

Then

$$
{ \mathfrak { g } } ( f , w ) = \int _ { 0 } ^ { 1 } p _ { \mathrm { n m } } ( z ) \nabla f { \biggl ( } { \underline { { x } } } + { \frac { z } { 2 } } ( w - { \underline { { x } } } ) { \biggr ) } \ d z .
$$

The query point generally difers from the played action $( w + \underline { { x } } ) / 2$ , so this is first-order full-information feedback. For all $w , y \in { \mathcal { K } }$

$$
\frac { 1 - \| \underline { { x } } \| _ { \infty } } { 4 } f ( y ) - f ( h ( w ) ) \leq \frac { 3 } { 8 } \langle \mathfrak { g } ( f , w ) , y - w \rangle .
$$

Consequently,

$$
\left( \alpha , \beta , G , L _ { \phi } \right) = \left( \frac { 1 - \| \underline { { { x } } } \| _ { \infty } } { 4 } , \frac { 3 } { 8 } , L , \frac { L } { 2 } \right) .
$$

Under Assumption 1, the known interior point $x _ { 0 }$ is an admissible choice of $\underline { { x } } .$

Proof. The played action and the query point are convex combinations of w and $\underline { { x } } ,$ hence both belong to $\kappa .$ The density is normalized because

$$
\int _ { 0 } ^ { 1 } { \frac { d z } { 3 ( 1 - z / 2 ) ^ { 3 } } } = 1 .
$$

Conditional unbiasedness followed by the tower property over $Z$ gives the displayed proxy without increasing the almost-sure norm bound. The structural inequality is Pedramfar and Aggarwal (2024, Lemma 3 and Algorithm 3). The Action Wrapper is $1 / 2 \mathrm { - L i p s c h i t z } .$ , so $L _ { \phi } = L / 2$ . Finally, because ${ \mathcal { K } } \subseteq [ 0 , 1 ] ^ { d }$ is fulldimensional and $x _ { 0 } \in \operatorname { i n t } ( K )$ , every coordinate of $x _ { 0 }$ is strictly below 1, and hence $\| x _ { 0 } \| _ { \infty } < 1$ □

Corollary 3 (Non-monotone up-concave regret on a general set). Under Proposition $\it 4$ and the remaining assumptions of Theorem 1, every agent $a \in [ N ]$ satisfies

$$
\mathfrak { R } _ { ( 1 - \| \underline { { x } } \| _ { \infty } ) / 4 } ^ { a } ( T ) = \widetilde { O } \left( \frac { 3 } { 8 } + L R \left[ \frac { 3 \kappa } { 8 } + \frac { ( 1 + \kappa ) ( 1 + \log N ) } { 2 } \right] \sqrt { \frac { T } { 1 - \rho } } \right) .
$$

Proof. Apply (34) with the tuple in Proposition 4 and $L _ { h } = 1 / 2$

## I.4 Non-Monotone DR-Submodular Functions on Down-Closed Sets

Proposition 5 (Exponential wrapper for non-monotone DR-submodular functions; adapted from Lu et al. (2026)). Let $f : [ 0 , 1 ] ^ { d } \to { \mathbb { R } } _ { \geq 0 }$ be diferentiable, non-monotone, and DR-submodular, and let ${ \mathcal { K } } \subseteq [ 0 , 1 ] ^ { d }$ be convex, down-closed, and contain the origin. Define

$$
\mathcal { W } ^ { \mathrm { a c t } } ( w ) = h ( w ) = \mathbf { 1 } - e ^ { - w } .
$$

The Query Wrapper draws $Z \in [ 0 , 1 ]$ with density

$$
p ( z ) = { \frac { e ^ { z - 1 } } { 1 - e ^ { - 1 } } } ,
$$

sets $z ^ { \circ } = \mathbf { 1 } - e ^ { - Z w }$ , and returns

$$
\begin{array} { r } { q \sim \widehat { \mathcal { O } } _ { f } ( w ) : = \mathcal { W } ^ { \mathrm { q r y } } ( \mathcal { O } _ { f } ) ( w ) , \qquad q = V _ { f } ( z ^ { \circ } ) \odot e ^ { - Z w } . } \end{array}
$$

Define

$$
F _ { f } ( w ) = \int _ { 0 } ^ { 1 } { \frac { e ^ { z - 1 } } { ( 1 - e ^ { - 1 } ) z } } { \big ( } f ( \mathbf { 1 } - e ^ { - z w } ) - f ( \mathbf { 0 } ) { \big ) } d z .
$$

The integrand at $z = 0$ is interpreted by its continuous extension. Then

$$
\mathfrak { g } ( f , w ) = \mathbb { E } _ { Z } \big [ \nabla f ( \mathbf { 1 } - e ^ { - Z w } ) \odot e ^ { - Z w } \big ] = \nabla F _ { f } ( w ) .
$$

The query $z ^ { \circ }$ generally difers from the played action ${ \bf 1 } - e ^ { - w }$ , so this is first-order full-information feedback. For all $w , y \in { \mathcal { K } }$

$$
\frac { 1 } { e } f ( y ) - f ( h ( w ) ) \leq ( 1 - e ^ { - 1 } ) \langle \mathfrak { g } ( f , w ) , y - w \rangle .
$$

Consequently,

$$
\left( \alpha , \beta , G , L _ { \phi } \right) = \left( { \frac { 1 } { e } } , 1 - e ^ { - 1 } , L , L \right) .
$$

Proof. For $w \in [ 0 , 1 ] ^ { d }$ and $Z \in [ 0 , 1 ]$ 2

$$
\mathbf { 1 } - e ^ { - Z w } \leq Z w \leq w , \qquad \mathbf { 1 } - e ^ { - w } \leq w
$$

coordinatewise. Down-closedness therefore gives $z ^ { \circ } , h ( w ) \in \mathcal { K }$ . Since the coordinates of $e ^ { - Z w }$ lie in (0, 1],

$$
\| q \| _ { 2 } \leq \| V _ { f } ( z ^ { \circ } ) \| _ { 2 } \leq L .
$$

Conditional unbiasedness and the tower property over $Z$ give the displayed proxy. The bounded-gradient condition supplies an integrable dominating function, so diferentiation under the integral gives $\gimel ( f , w ) = \nabla F _ { f } ( w )$ The $1 / e$ upper-linearization inequality is Lu et al. (2026, Theorem 1), and its expectation representation and one-query estimator appear in that paper’s Remark 3, Algorithm 1, and Lemma 2. Finally, the Jacobian of $h ( w ) = \mathbf { 1 } - e ^ { - w }$ is diagonal with operator norm at most 1 on $[ 0 , 1 ] ^ { d }$ , so $L _ { \phi } = L$ is valid. □

Corollary 4 (Non-monotone DR-Submodular over Down-closed Set). Under Proposition 5 and the remaining assumptions of Theorem 1, every agent $a \in [ N ]$ satisfies

$$
\mathfrak { R } _ { 1 / e } ^ { a } ( T ) = \widetilde { O } \left( 1 - e ^ { - 1 } + L R \left[ ( 1 - e ^ { - 1 } ) \kappa + ( 1 + \kappa ) ( 1 + \log N ) \right] \sqrt { \frac { T } { 1 - \rho } } \right) .
$$

Proof. Apply (34) with the tuple in Proposition 5 and $L _ { h } = 1$

The four wrappers above are not new approximation reductions. Their role here is to provide the exact Action and Query Wrappers, verify the feedback and payof-stability assumptions of Theorem 1, and obtain the corresponding decentralized guarantees. This work does not develop results for limited-feedback reductions and applications to one-sided-smooth objectives, approximately convex objectives, and relative-interior domains such as matroid basis polytopes.