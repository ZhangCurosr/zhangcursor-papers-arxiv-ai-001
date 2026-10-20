# LOCU: Löwdin-Orthogonalized Constraint Updates for Multi-Constraint Policy Optimization

Joonyoung Lim Department of Information Convergence Engineering Pusan National University ingyuh1015@pusan.ac.kr

Younghwan Yoo<sup>∗</sup> School of Computer Science and Engineering Pusan National University ymomo@pusan.ac.kr

## Abstract

We investigate multi-constraint policy optimization in constrained Markov decision processes (CMDPs), where interactions among constraint gradients often yield ill-conditioned update directions, causing numerically fragile steps and unintended cancellation or redundant overlap in constraint corrections. To address this, we propose LOCU, which mitigates numerical instability arising from constraint interactions by applying symmetric Löwdin orthogonalization to the natural gradients of the reward and constraints under the Fisher–Rao metric. Combined with Fisher-geometric screening and near-collinear compression, LOCU yields a compact reduced space in which the trust region becomes a Euclidean ball and constraints are handled symmetrically through a low-dimensional active-set solve. For infeasible iterates, LOCU constructs a Pareto-descent direction which preserves the local feasibility of satisfied constraints within the same reduced formulation. Experiments show that LOCU remains stable under high constraint coupling, narrow feasible regions, and infeasible starts, and generalizes across different cost critic architectures. Code is available at: GitHub Repository.

## 1 Introduction

Reinforcement learning has achieved notable success in sequential decision making, but deployment in safety-critical domains requires policies to additionally satisfy explicit operational constraints rather than maximize reward alone [20, 21, 1, 2]. Constrained Markov decision processes (CMDPs) formalize this requirement by optimizing return subject to cost limits. In practical scenario, agents could face several constraints simultaneously, such as velocity, collision, or resource limit in autonomous driving or robotic control. Existing safe policy-optimization methods commonly proposed and evaluated in single-constraint regimes, and their behavior can degrade when multiple constraints interact.

The core difficulty with multi-constraint is not merely computational overhead, but the interaction geometry among constraints [18]. Existing approaches expose one or more of following failures (i) unstable updates under near-dependent constraint gradients or severe scale disparity, (ii) weakened effective updates under constraint interaction, caused by cancellation, dominance by a single direction, or projection-order dependence, and (iii) inefficient feasibility recovery after constraint violations, as noted in prior work [14].The problem worsens under many or near-parallel constraints, making stability depend on hyperparameter tuning rather than geometry-consistent structure.

A geometric perspective clarifies why these issues arise and how to address them. Policy optimization can be analyzed on a Riemannian manifold whose metric arises as the second-order local approximation of the KL divergence, known as the Fisher–Rao metric [4, 3]. Reward and constraint objectives define tangent directions in natural-gradient coordinates [12], and each linearized constraint induces a halfspace normal to its constraint gradient. With multiple constraints, feasible updates lie in the intersection of these halfspaces and the local Fisher-metric ellipsoid [1, 28]. Redundant or ill-conditioned constraint directions under this metric make both first- and second-order updates fragile and conservative, an effect amplified by finite-sample noise or critic error.

Building on this geometric insight, we introduce the Löwdin Orthogonalized Constraint Update (LOCU), a second-order framework for multi-constraint RL that mitigates constraint coupling in Fisher-consistent coordinates. LOCU constructs a Fisher-orthonormal basis from the natural gradients of the reward and constraints via symmetric Löwdin orthogonalization [17]. This transformation maps the Fisher-metric ellipsoid into a spherical trust region in a reduced Euclidean space, making constraint interactions explicit and numerically stable. For large constraint sets, LOCU employs Fisher-geometric screening to prune non-binding or redundant constraints. Furthermore, it handles infeasibility with a geometry-aligned recovery step that seeks to reduce violations while preserving satisfied constraints in linearized local model. The update operates purely at the policy-optimization layer, ensuring compatibility with various critic architectures.

We conduct experiments with multi-constrained tasks of [8, 15] applying the constraint configurations introduced by [29]. Evaluations under narrow feasible regions and strong constraint coupling, LOCU satisfies all constraints while limiting return drops to around 21% in both settings, compared to 26–49% and 13–32% drops with frequent violations in baselines, performing consistently from infeasible starts and generalizing across different cost critic architectures. These results demonstrate the effectiveness of the proposed method in maintaining stable policy optimization by symmetrically handling constraint interactions under the Fisher–Rao geometry.

## 2 Related Work

Existing constrained RL methods differ in how they incorporate multiple constraints into policy optimization and maintain feasibility during actor updates. These design choices become tightly coupled with the representation of constraint interactions. When constraint interactions are not handled in a geometry-consistent manner under the Fisher–Rao metric, stable learning often relies heavily on environment-specific tuning of numerous algorithmic choices and hyperparameters.

Primal-dual Methods. Lagrangian-based methods adopt a primal-dual formulation where constraints are managed via Lagrange multipliers co-optimized with policy parameters [23]. However, with multi-constraint, these methods aggregate costs from multi constraint into a scalar, obscuring the directional structure of individual constraint gradients. Depending on alignment, constraint gradients may cancel and suppress effective update, or reinforce and destabilize multiplier dynamics, causing oscillations and training instability. As the number of constraints increases, performance becomes increasingly sensitive to multiplier initialization and tuning can be “as hard as solving the RL problem itself” [19].

Recent work has attempted to address gradient-level interference in multi-constraint settings by analyzing directional interactions and applying shaping rules to improve update consistency [29]. While this mitigates some interference issues within the Lagrangian framework, its reliance on Euclidean angle-based similarity may not faithfully reflect the true correlations under the Fisher– Rao geometry [12]. Furthermore, it does not alleviate the burden of tuning an expanding set of hyperparameters as constraints grow.

A separate line of works handle dual variables by solving a quadratic program (QP) or projection defined by linearized constraints within a KL-based trust region [1]. In multi-constraint regimes, feasibility recovery is either not explicitly addressed once iterates leave the local feasible region [1], or handled via repeated constraint-wise projections that shrink the effective reward step and introduce imbalances as the number of constraints grows [28]. Moreover, the active set is unknown a priori, these updates rely on generic QP solvers. Under linear dependence or near-collinearity among constraint gradients, the resulting $\mathrm { Q P }$ can be ill-conditioned, leading to solver failures or high-variance solutions where it is difficult to distinguish intrinsic constraint degeneracy from solverinduced numerical errors.

Recent works in risk-constraint RL proposed gradient integration mechanisms to manage infeasibility start and recovery in multi-constraint settings and further extend these ideas toward global convergence guarantees [14, 15]. Despite these advances, general applicability across constrained RL is largely limited to gradient integration, while the absence of a geometry-consistent formulation and continued dependence on generic solvers hinder scalability and stability as the number of constraints and interaction grows.

Primal Methods. Primal methods can be naturally extended to multi-constraint settings by switching between reward optimization and constraint minimization [26]. However, as the number of constraints increases, handling one constraint at a time leads to slower convergence and inconsistent updates. Furthermore, substantial correlations between adaptive weights and stochastic gradients are known to bias the update step and cause convergence failure [31]. To address such instabilities, recent work proposes a soft-switching strategy [9], which reduces oscillations and accelerates convergence by removing conflicting components between reward and constraint gradients. While it improves stability in single-constraint scenarios, effectiveness in multi-constraint settings remains unverified.

For a geometric discussion of multi-constraint update pathologies in representative primal–dual and projection-based methods, see Appendix C.1.

## 3 Preliminaries

## 3.1 Constrained Markov Decision Processes

A CMDP is defined by $( \mathcal { S } , \mathcal { A } , P , r , \{ c _ { k } \} _ { k = 1 } ^ { K } , \rho , \gamma )$ , where S and A are the state and action spaces, $\textstyle P ( \cdot \mid s , a )$ is the transition probability, $r : S \times \mathcal { A }  \mathbb { R }$ is the reward function, and $c _ { k } : S \times \mathcal { A }  \mathbb { R } _ { \geq 0 }$ are K cost functions that quantify constraint violations [2]. The initial state is $s _ { 0 } \sim \rho$ and the discount factor is $\gamma \in ( 0 , 1 )$ . A stochastic policy $\pi ( a \mid s )$ induces a trajectory distribution $\mathbb { P } _ { \pi }$ over $( s _ { 0 } , a _ { 0 } , s _ { 1 } , \dots )$ . Given a policy π, its discounted state–action distribution is

$$
d ^ { \pi } ( s , a ) : = ( 1 - \gamma ) \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbb { P } _ { \pi } ( s _ { t } = s , a _ { t } = a ) .\tag{1}
$$

Using this distribution, define the discounted return $\begin{array} { r } { J _ { R } ( \pi ) : = \mathbb { E } _ { s _ { 0 } \sim \rho } [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) ] } \end{array}$ , and for each cost k define $\begin{array} { r } { J _ { C _ { k } } ( \pi ) : = \mathbb { E } _ { s _ { 0 } \sim \rho } [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } c _ { k } ( s _ { t } , a _ { t } ) ] } \end{array}$ . The CMDP objective is

$$
\operatorname* { m a x } _ { \pi } J _ { R } ( \pi ) \quad { \mathrm { s . t . } } \quad J _ { C _ { k } } ( \pi ) \leq b _ { k } , \forall k .\tag{2}
$$

Assumption 3.1 (Slater condition). There exists a policy π¯ in the considered policy class such that

$$
J _ { C _ { k } } ( \bar { \pi } ) < b _ { k } , \qquad \forall k \in \{ 1 , \ldots , K \} .\tag{3}
$$

This feasibility condition is standard in constrained policy optimization [1].

## 3.2 Policy Gradients for Reward and Constraints

Consider a differentiable policy π<sub>θ</sub> parameterized by $\boldsymbol \theta \in \mathbb { R } ^ { n }$ . Let $A _ { R } ^ { \pi } ( s , a ) : = Q _ { R } ^ { \pi } ( s , a ) - V _ { R } ^ { \pi } ( s )$ The policy gradient for the reward objective is

$$
\begin{array} { r } { \nabla _ { \theta } J _ { R } ( \pi _ { \theta } ) = \frac { 1 } { 1 - { \gamma } } \mathbb { E } _ { ( s , a ) \sim d ^ { \pi _ { \theta } } } \bigl [ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) A _ { R } ^ { \pi _ { \theta } } ( s , a ) \bigr ] . } \end{array}\tag{4}
$$

Similarly, each cost objective admits a policy gradient obtained by replacing r with $c _ { k }$ . We denote $g _ { r } : = \nabla _ { \theta } J _ { R } ( \pi _ { \theta } )$ and $g _ { c _ { k } } : = \nabla _ { \theta } J _ { C _ { k } } ( \pi _ { \theta } )$ .

## 3.3 Trust-Region Geometry and Natural Gradient

Trust-region updates are measured by the average KL divergence under the discounted state visitation distribution,

$$
\bar { D } _ { \mathrm { K L } } ( \pi _ { \theta } , \pi _ { \theta ^ { \prime } } ) : = \mathbb { E } _ { s \sim d ^ { \pi _ { \theta } } } [ \mathrm { K L } ( \pi _ { \theta } ( \cdot  { \left| \begin{array} { l } { s } \end{array} \right| } | \pi _ { \theta ^ { \prime } } ( \cdot  { \left| \begin{array} { l } { s } \end{array} \right) } ) ] .\tag{5}
$$

For a small displacement $\Delta \theta : = \theta ^ { \prime } - \theta$ , a second-order Taylor expansion gives

$$
\bar { D } _ { \mathrm { K L } } ( \pi _ { \theta } , \pi _ { \theta + \Delta \theta } ) = \frac { 1 } { 2 } \Delta \theta ^ { \top } F \Delta \theta + o ( \| \Delta \theta \| ^ { 2 } ) ,\tag{6}
$$

where $F$ is the Fisher information matrix (FIM),

$$
F : = \mathbb { E } _ { ( s , a ) \sim d ^ { \pi _ { \theta } } } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) ^ { \top } \right] .\tag{7}
$$

Equation (6) induces the Fisher–Rao geometry, where the FIM F defines the inner product $\langle u , v \rangle _ { F } : =$ $u ^ { \top } F v$ and the corresponding norm $\| u \| _ { F } : = \sqrt { u ^ { \top } F u }$ . Since the FIM F is always positive semidefinite, we assume it to be positive definite $( F \succ 0 )$ in what follows so that $F ^ { - 1 }$ and the Fisher inner product are well-defined (in practice, this can be enforced by diagonal damping).

Under this geometry, the natural gradient is defined as the steepest-ascent direction with respect to the Fisher–Rao metric:

$$
{ \tilde { g } } : = F ^ { - 1 } g .
$$

We denote natural gradients for the reward and costs as $\tilde { g } _ { r } : = F ^ { - 1 } g _ { \ i }$ <sub>r</sub> and $\tilde { g } _ { c _ { k } } : = F ^ { - 1 } g _ { c _ { k } }$

(8)

## 3.4 Local Multi-Constraint Trust-Region Subproblem

Under a KL-based trust region, Eq. (2) is locally approximated by first-order expansions of $J _ { R }$ and $J _ { C _ { k } }$ at θ together with the Fisher quadratic approximation (6). Define the current constraint slacks as $\Delta _ { k } : = b _ { k } - J _ { C _ { k } } ( \pi _ { \theta } )$ . The local trust-region CMDP step is

$$
\begin{array} { r l } { \underset { \Delta \theta \in \mathbb { R } ^ { n } } { \operatorname* { m a x } } } & { g _ { r } ^ { \top } \Delta \theta } \\ { \mathrm { s . t . } } & { g _ { c _ { k } } ^ { \top } \Delta \theta \leq \Delta _ { k } , \quad k = 1 , \ldots , K , } \\ & { \frac { 1 } { 2 } \left\| \Delta \theta \right\| _ { F } ^ { 2 } \leq \delta . } \end{array}\tag{9}
$$

Problem (9) is a convex program. When the current iterate is strictly feasible, $\mathrm { i } . \mathrm { e } . , \Delta _ { k } > 0$ for all k and $\delta > 0 _ { : }$ , the point $\Delta \theta = 0$ lies in the interior of the feasible set. In this case, strong duality holds and the optimum is characterized by the Karush–Kuhn–Tucker (KKT) conditions and the associated dual variables [6].

Lemma 3.2. Problem (9) admits an optimal solution $\Delta \theta ^ { \star }$ such that $\Delta \theta ^ { \star } \in \mathrm { s p a n } \{ \tilde { g } _ { r } , \tilde { g } _ { c 1 } , . . . , \tilde { g } _ { c { \scriptscriptstyle K } } \}$

Proof. See Appendix A.2.1.

## 3.5 Symmetric Löwdin orthogonalization

When working with multiple update directions, it is often useful to replace a correlated basis by an orthonormal one in a way that is symmetric and insensitive to the ordering of directions. A classical construction is the symmetric orthogonalization of Löwdin [17].

Let $S = [ s _ { 1 } , \ldots , s _ { p } ] \in \mathbb { R } ^ { n \times p }$ be a matrix whose columns are linearly independent directions. Define the Gram matrix $\begin{array} { r } { \dot { G } : = S ^ { \top } S \in \mathbb { R } ^ { p \times p } } \end{array}$ . Since rank $( S ) = p .$ , we have $G \succ 0$ and the principal inverse square root $G ^ { - 1 / 2 }$ is well-defined. The symmetric Löwdin orthogonalization constructs

$$
E : = S G ^ { - 1 / 2 } \in \mathbb { R } ^ { n \times p } , \qquad E ^ { \top } E = G ^ { - 1 / 2 } S ^ { \top } S G ^ { - 1 / 2 } = I .\tag{10}
$$

so the columns of E are orthonormal and span $. ( E ) = \operatorname { s p a n } ( S )$

Unlike Gram–Schmidt, the Löwdin construction is symmetric: it depends only on $G = S ^ { \top } S$ and does not privilege any particular column order. This order-invariant structure is the property used by LOCU when constructing reduced coordinates from reward and constraint directions.

## 4 Proposed Method

We focus on the local multi-constraint trust-region subproblem in (9). By Lemma 3.2, it suffices to search over the subspace spanned by the reward and constraint natural-gradient directions. We construct Fisher-orthonormal reduced coordinates for this subspace and solve an equivalent lowdimensional convex program; the remainder of this section provides the coordinate construction, the scalable handling of many constraints, and the feasibility-recovery variant. Practical implementation details are provided in Appendix A.4.

## 4.1 Fisher-orthonormal reduced coordinates via Löwdin orthogonalization

Motivated by the symmetric Löwdin orthogonalization introduced in Section 3.5, we apply the same construction to the reward and constraint natural-gradient directions under the Fisher–Rao metric.

Let $\tilde { g } _ { r } : = F ^ { - 1 } g _ { r }$ and $\tilde { g } _ { c _ { k } } : = F ^ { - 1 } g _ { c _ { k } }$ denote the reward and cost natural gradients. Fix any set of m representative constraint indices $\{ i _ { 1 } , \dotsc , i _ { m } \} \subseteq \{ 1 , \dotsc , K \}$ , and form the direction matrix

$$
S : = \left[ \tilde { g } _ { r } \quad \tilde { g } _ { c _ { i _ { 1 } } } \quad \cdot \cdot \cdot \quad \tilde { g } _ { c _ { i _ { m } } } \right] \in \mathbb { R } ^ { n \times ( m + 1 ) } .\tag{11}
$$

Define the Gram matrix under the Fisher inner product, which we call the Fisher Gram matrix for convenience:

$$
O : = { S ^ { \top } F S \in \mathbb { R } ^ { ( m + 1 ) \times ( m + 1 ) } } .\tag{12}
$$

We construct the symmetric Löwdin basis

$$
E : = S O ^ { - 1 / 2 } \in \mathbb { R } ^ { n \times ( m + 1 ) } .\tag{13}
$$

Lemma 4.1. $H O \sim 0 ( o r i f O$ is regularized to be positive definite), then

$$
E ^ { \top } F E = I .\tag{14}
$$

Proof is in Appendix A.2.2.

Lemma 4.1 implies that the columns of E form an orthonormal basis under the Fisher inner product. We restrict the update to span(S) and write $\Delta \theta = E \quad$ w with $w \in \mathbb { R } ^ { m + 1 }$ . Let $R : = \sqrt { 2 \delta }$ . Since $E ^ { \top } F E = I$ by (14), the quadratic KL trust region reduces to

$$
\begin{array} { r } { \frac 1 2 \Delta \boldsymbol { \theta } ^ { \top } \boldsymbol { F } \Delta \boldsymbol { \theta } = \frac 1 2 \boldsymbol { w } ^ { \top } \boldsymbol { w } = \frac 1 2 \| \boldsymbol { w } \| _ { 2 } ^ { 2 } , } \end{array}\tag{15}
$$

and hence becomes the Euclidean ball $\| w \| _ { 2 } \leq R$ in the reduced coordinates.

Define the reduced constraint normals and the reduced-space objective coefficients in the Löwdin basis by

$$
\begin{array} { r } { u : = E ^ { \top } g _ { r } , \qquad C _ { k } ^ { \top } : = E ^ { \top } g _ { c _ { k } } \in \mathbb R ^ { m + 1 } . } \end{array}\tag{16}
$$

Here, we use the symbol u as the objective coefficient vector of the reduced program. In the feasible phase, $u = E ^ { \top } g _ { r }$ as above; in the feasibility-recovery phase (Section 4.4), we simply replace u by the recovery direction constructed directly in the same reduced space.

Let ${ \mathcal { K } } \subseteq \{ 1 , \ldots , K \}$ be the set of constraints enforced in the current update, and collect the corresponding rows into $C \in \mathbb { R } ^ { | K | \times ( m + 1 ) }$ and $\Delta \in \mathbb { R } ^ { | K | }$ with $\Delta _ { k } = b _ { k } - J _ { C _ { k } } ( \pi _ { \theta } )$ . Then the reduced program is

$$
\operatorname* { m a x } _ { w \in \mathbb { R } ^ { m + 1 } } \ u ^ { \top } w \quad \mathrm { s . t . } \quad \| w \| _ { 2 } \leq R , \ C w \leq \Delta .\tag{17}
$$

Proposition 4.2. Let $e _ { j }$ be the jth canonical basis vector in $\mathbb { R } ^ { m + 1 }$ , and let the columns of S be ordered as in (11). Then

$$
E ^ { \top } g _ { r } = O ^ { 1 / 2 } e _ { 1 } , \quad E ^ { \top } g _ { c _ { i _ { j } } } = O ^ { 1 / 2 } e _ { j + 1 } , \quad j = 1 , \ldots , m .\tag{18}
$$

Proof is in Appendix A.2.3.

Theorem 4.3. Assume $O \succ 0 ( o r O$ is regularized so that $O \succ 0 ) .$ . Fix any enforced constraint index set K and any representative set $\{ i _ { 1 } , \ldots , i _ { m } \}$ used toform $S , O , E$ as in (11)–(13). Then the change of variables ∆θ = Ew gives a one-to-one correspondence between span(S) and $\mathbb { R } ^ { m + 1 }$ and transforms the trust-region CMDP subproblem (9) restricted to $\Delta \theta \in \operatorname { s p a n } ( S )$ and constraints $k \in \mathcal { K }$ into the reduced program (17). In particular, any optimizer $w ^ { \star }$ of (17) yields a valid update $\Delta \theta ^ { \star } = E w ^ { \star }$ in the original parameter space.

## Proof is in Appendix A.2.4.

Unlike Gram–Schmidt orthogonalization, which propagates and accumulates perturbations through a sequential process, Löwdin orthogonalization in Proposition 4.2 distributes perturbations symmetrically across the eigenspace. We analyze these perturbation structures in Appendix C.2.1.

## 4.2 Constraint screening and basis compression

However, directly working with all constraint directions can be expensive at scale and can be numerically unstable when constraint directions are strongly aligned in Fisher–Rao geometry. Therefore, LOCU screens locally non-binding/degenerate constraints and compresses near-collinear directions for basis construction, while still enforcing every retained constraint in the reduced optimization.

With Eqs. (11) and (12), we start from full direction with reward and K constraints $S _ { \mathrm { f u l l } } \in \mathbb { R } ^ { n \times ( K + 1 ) }$ and its Fisher Gram matrix $O _ { \mathrm { f u l l } } \in \mathbb { R } ^ { ( K + 1 ) \times ( K + 1 ) }$ . For each constraint $k ,$ define the Fisher norm and normalized slack

$$
\nu _ { k } : = \sqrt { ( O _ { \mathrm { f u l l } } ) _ { k + 1 , k + 1 } } , \qquad \mu _ { k } : = \Delta _ { k } / \nu _ { k } ,\tag{19}
$$

and for each pair $( i , j )$ define the Fisher cosine similarity among constraint natural gradients

$$
\varrho _ { i j } : = \frac { ( O _ { \mathrm { f u l l } } ) _ { i + 1 , j + 1 } } { \nu _ { i } \nu _ { j } } .\tag{20}
$$

Equivalently, with Fisher-unit directions $u _ { k } : = \tilde { g } _ { c _ { k } } / \nu _ { k }$ (so that $\| u _ { k } \| _ { F } = 1 )$ , we have $\varrho _ { i j } = \langle u _ { i } , u _ { j } \rangle _ { F }$ and the kth linearized constraint is $\langle u _ { k } , \Delta \theta \rangle _ { F } \leq \mu _ { k }$

## 4.2.1 Constraint screening

We perform worst-case Fisher screening and exclude non-binding and near-degenerate constraints. Lemma 4.4. For any $\Delta \theta$ satisfying $\| \Delta \theta \| _ { F } \le R$

$$
g _ { c _ { k } } ^ { \top } \Delta \theta \leq \nu _ { k } R .\tag{21}
$$

Proof is in Appendix A.2.5.

Proposition 4.5. If $\Delta _ { k } \geq \nu _ { k } R ,$ , then constraint k cannot be active inside the trust region $\begin{array} { r } { \frac 1 2 \Delta \hat { \theta } ^ { \top } \boldsymbol { F } \Delta \theta \leq \delta } \end{array}$ and can be excluded from $\kappa$ without changing feasibility within that trust region.

Proof is in Appendix A.2.6.

In addition, constraints with $\nu _ { k } \le \varepsilon _ { \mathrm { n z } }$ are excluded from K to avoid unstable normalization in (19), since $\tilde { g } _ { c _ { k } }$ is nearly degenerate in Fisher–Rao geometry.

## 4.2.2 Basis clustering

We compress near-collinear constraint directions only to stabilize basis construction; all constraints in $\kappa$ remain enforced in the reduced program.

We first build an undirected graph on $\kappa$ by connecting each pair $( i , j )$ that satisfies $| \varrho _ { i j } | \geq 1 - \varepsilon _ { \mathrm { d u p } }$ Let $\{ \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { m } \}$ be the connected components of this graph. Then, for each group $\mathcal { G } \ell ,$ , we select a representative index by $i \ell \in$ arg min<sub>k∈G</sub> µ<sub>k</sub>. If all $\mu _ { k }$ in the group are nonnegative, this rule selects the tightest constraint in Fisher-normalized units. If some $\mu _ { k }$ are negative, it selects the most violated constraint in the same units. The reduced basis matrix S in (11) is then formed using $\{ i _ { 1 } , \ldots , i _ { m } \}$

Lemma 4.6. Let $u _ { i } , u _ { j }$ be Fisher-unit vectors with $\langle u _ { i } , u _ { j } \rangle _ { F } = \rho \in [ - 1 , 1 ]$ . Thenfor any $\Delta \theta$ with $\| \Delta \theta \| _ { F } \le R$

$$
\operatorname* { m i n } _ { s \in \{ + 1 , - 1 \} } \left| \langle u _ { i } , \Delta \theta \rangle _ { F } - s \langle u _ { j } , \Delta \theta \rangle _ { F } \right| \leq \sqrt { 2 ( 1 - | \rho | ) } R .\tag{22}
$$

In particular, $i f \vert \rho \vert \geq 1 - \varepsilon _ { \mathrm { d u p } } ,$ the discrepancy is at most $\sqrt { 2 \varepsilon _ { \mathrm { d u p } } } R$ . One may take $s = + 1$ when $\rho \geq 0$ and $s = - 1$ when $\rho < 0 .$

## Proof is in Appendix A.2.7.

Lemma 4.6 formalizes why compressing near-collinear directions in the basis construction has negligible local effect: within the trust region, constraints in the same group respond similarly in Fisher-normalized units up to a sign. Importantly, all constraints in $\kappa$ remain explicitly enforced in (17) via the rows of $C .$

## 4.3 Solving the reduced program

We solve (17) in the reduced coordinates. In the feasible case (all slacks in $\Delta$ are nonnegative), we set $u = E ^ { \top } g _ { r }$ . If the halfspaces are inactive, the maximizer over the ball has the closed form

$$
w _ { 0 } = R \frac { u } { \| u \| _ { 2 } } ,\tag{23}
$$

whenever $\| u \| _ { 2 } \neq 0 .$ . If $C w _ { 0 } \le \Delta$ , we accept $w ^ { \star } = w _ { 0 }$ . Otherwise, we use KKT conditions. There exist $\lambda \in \mathbb { R } ^ { | K | }$ and $\beta \in \mathbb { R }$ such that

$$
- u + C ^ { \top } \lambda + 2 \beta w = 0 , \qquad \lambda \geq 0 , \qquad \beta \geq 0 ,\tag{24}
$$

together with primal feasibility, complementary slackness, and $\| w \| _ { 2 } ~ \le ~ R$ . For an active set $\mathcal { T } \subseteq \{ 1 , \dots , | \dot { \mathcal { K } } | \}$ with $\beta > 0$ , stationarity yields

$$
w = \frac { 1 } { 2 \beta } \big ( u - C _ { \mathcal { T } } ^ { \top } \lambda _ { \mathcal { T } } \big ) ,\tag{25}
$$

and enforcing $C _ { \mathcal { I } } w = \Delta _ { \mathcal { I } }$ gives the reduced linear system

$$
C _ { \mathcal { T } } C _ { \mathcal { I } } ^ { \top } \lambda _ { \mathcal { I } } = C _ { \mathcal { I } } u - 2 \beta \Delta _ { \mathcal { I } } .\tag{26}
$$

All matrices in (25)–(26) have dimension at most $( m + 1 )$ and are independent of the parameter dimension n. The active-set routine and the one-dimensional search for $\beta$ are given in Appendix A.3.

## 4.4 Feasibility recovery via Pareto descent in the orthonormal space

When some constraints are violated $( \Delta _ { k } < 0$ for at least one $k \in \mathcal { K } )$ , we use a recovery objective that prioritizes feasibility improvement in the linearized model while preventing already satisfied constraints from deteriorating in the same local step.

Let $\mathcal { V } : = \left\{ i \in \left\{ 1 , \dots , | \mathcal { K } | \right\} | \Delta _ { i } < 0 \right\}$ and $\mathcal { H } : = \{ i \in \{ 1 , . . . , | \mathcal { K } | \} \mid \Delta _ { i } \geq 0 \}$ , with indices defined according to the row ordering of $C$ induced by $\kappa .$ We compute the simplex weights by solving the following problem:

$$
\operatorname* { m i n } _ { \alpha \in \mathbb { R } ^ { | \mathcal { V } | } } \| C _ { \mathcal { V } } ^ { \top } \alpha \| _ { 2 } ^ { 2 } \quad \mathrm { s . t . } \quad \alpha \geq 0 , \mathbf { 1 } ^ { \top } \alpha = 1 .\tag{27}
$$

and define the Pareto descent direction

$$
\bar { g } _ { p } : = - C _ { \nu } ^ { \top } \alpha .\tag{28}
$$

Lemma 4.7. $I f \bar { g } _ { p } \ne 0$ defined by (27)–(28), thenfor every $i \in \mathcal V ,$

$$
{ \cal C } _ { i } ^ { \top } \bar { g } _ { p } < 0 .\tag{29}
$$

Proof is in Appendix A.2.8.

We set the recovery objective to $u = \bar { g } _ { p }$ and solve

$$
\operatorname* { m a x } _ { w \in \mathbb { R } ^ { m + 1 } } \bar { g } _ { p } ^ { \top } w \quad \mathrm { s . t . } \quad \| w \| _ { 2 } \leq R , \quad C _ { \mathcal { H } } w \leq \Delta _ { \mathcal { H } } .\tag{30}
$$

The point $w = 0$ is feasible since $\Delta _ { \mathcal { H } } \geq 0 ,$ , so (30) cannot worsen already satisfied constraints in the linearized local model.

Computational complexity. The proposed method starts by computing natural gradients for the reward and constraints. Each gradient computation requires one backpropagation over the entire batch. Computing a natural gradient with conjugate gradient (CG) further requires $T _ { \mathrm { c g } }$ iterations per gradient, each consisting of one or two backpropagations depending on the implementation. Therefore, the computational cost can be written as $\mathcal { O } \big ( ( \bar { K } + 1 ) \cdot \bar { T _ { \mathrm { c g } } } \cdot \mathtt { B a c k P r o p } \big )$

After obtaining the natural gradients, the Fisher Gram matrix is constructed from $K ^ { 2 }$ pairwise inner products between them, resulting in a cost of $\mathcal { O } ( K ^ { 2 }$ · InnerProd). Screening and clustering are performed at a cost of $\mathcal { O } ( K + | \dot { K | } ^ { 2 } )$ , followed by Löwdin orthogonalization of the m representative directions at a cost of $\mathcal { O } ( m ^ { 2 }$ · InnerProd + m<sup>3</sup>). Solving the reduced program with the active-set method requires $\mathcal { O } ( T _ { \mathrm { a c t i v e } } \cdot m ^ { 3 } )$ operations, where $T _ { \mathrm { a c t i v e } }$ is the number of active-set iterations, while an additional simplex QP over the violated constraints is solved in the recovery case. Since $m \leq | { \boldsymbol { \kappa } } | \leq$ K, the computational cost of LOCU procedure is bounded by $\mathcal { O } ( K ^ { 2 }$ · InnerProd $+ T _ { \mathrm { a c t i v e } } \cdot { \dot { K } } ^ { 3 } )$ which simplifies to $\mathcal { O } ( K ^ { 2 }$ · InnerProd) since the second term is negligible relative to the parameterdimensional inner products.

![](images/3f5cf48a12bf6e6b1ee2b8930fbc49cd7917cc53a44451b9e61134424b2a360e.jpg)  
Figure 1: Final evaluation on the v2 Circle tasks, with high-velocity and boundary constraints.

Both back-propagation and inner product scale linearly with the size of policy parameters n. However, back-propagation involves forward and backward passes over the entire batch, whereas inner product requires only element-wise multiplication and summation. Consequently, the overall computational cost is dominated by natural-gradient computation, which is also required by other second-order multi-constraint methods, leaving LOCU with only marginal additional overhead. Further discussion related to computational cost is provided in Appendix C.3.1.

## 5 Evaluation and Results

## 5.1 Tasks and Baselines

Tasks We evaluated on two settings. First, we use Circle tasks from Bullet-Safety-Gym [8] with Ball, Car, and Drone agents, denoted as BC, CC, and DC, under the multi-constraint setting of [29]. The task is to follow a circular trajectory, where costs are incurred for leaving the boundary region, violating upper and lower velocity limits. We vary constraint thresholds and extend with redundancy variants [29] to control feasible-set geometry and induce near-collinear constraint directions.

Second, we evaluate legged locomotion tasks introduced by [15] on the bipedal and quadrupedal robots Cassie (Agility Robotics) and Laikago (Unitree), respectively. The objective is to track a commanded velocity while enforcing three constraints on body balance, height, and foot contact timing. To further test heterogeneous constraint handling, we add a goal-navigation variant that retains the three locomotion-stability constraints and introduces three qualitatively different constraints on hazard contact, speed, and energy which was originally on the reward side. This variant require the agent to reach a goal while avoiding hazards and satisfying operational limits. Both versions are initialized from infeasible states with simultaneous constraint violations under high-dimensional observations, making these tasks suitable for assessing feasibility recovery and compatibility with alternative critic types. For locomotion tasks, all methods use the same quantile-based cost critic.

Baselines The baselines include the actor update of SDAC [14], denoted as SDAC(A), which uses a second-order trust-region QP update with gradient integration for recovery, while all other components are aligned with our setup. We also evaluate two primal methods, PCRPO [9] and CRPO [26], alongside PPO and TRPO Lagrangian with PID-style multiplier updates [23], each with and without gradient shaping (GS) [29], denoted as PPOL-GS, PPOL-V, TRPOL-GS, and TRPOL-V, respectively. For the final task, we further apply Clarabel, a modern interior-point conic solver [7], to both LOCU’s reduced QP and the full-dimensional QP.

Detailed descriptions for both task and baselines are provided in Appendix B.

## 5.2 Results

Results are averaged over five training seeds and 100 evaluation episodes with 10 rollout seeds. For the Circle tasks, we additionally reduce evaluation bias by averaging over the five checkpoints at the end of each training run. All costs are normalized by their corresponding limits; values above 1 indicate constraint violations. In the bar plots, violating cost bars and the corresponding reward bar are rendered with reduced opacity and hatching. Error bars indicate standard deviation. Additional results of ablation studies and full learning curves are provided in Appendix C.4 and C.5.

![](images/e6da9f0ac67cbba647332f5d30ba35f6380ae5cc7c571d7b9022476e55cd6956.jpg)

![](images/0763db4c218510b601dd0e1db0ebcb68070f58e698831322ed6a2749bca0f9e2.jpg)

![](images/6e23a0374453d51e69366a41ea7bdbfb2e08a6482e3739eb2dd1915dd3d9f432.jpg)

![](images/742c2f44657c934cafb790d4ebe05f27f0f4b3f4612321bdde03868916997bf2.jpg)

![](images/879856b7afd3e3d450160ac422ef4b14be1716357c56330346848e4c0451f359.jpg)

![](images/fd912e83aaec95ddfc0a423b57a18c10fc8217feb20491922c5b98b1d8ed8018.jpg)

![](images/2565b75691d1ea68d7b3384eb9aeaa8b487a422aab5956e3d72375d98f89835d.jpg)

![](images/7cae5c040272ff5c49e9210c81bac7fb57237701ccee32c080ca05efb448730b.jpg)  
LOCU(Ours) SDAC(A) PCRPO CRPC \_TRPOL-GS TRPOL-V  
Figure 2: Final evaluation results on BC-v3 variants. Top: the low-velocity requirement $\kappa _ { \mathrm { l o } } ~ \in$ {0.0, 0.4, 0.6, 0.8}. Bottom: redundancy variants with increasing numbers of same-type boundary and velocity constraints. The suffix $\mathrm { \ddot { \Omega } - } \mathrm { a } ^ { \mathrm { \check { \Omega } } }$ marks the tightest constraint among the same type.

![](images/f38725eec47c3d8db14e362e908df520f32950aee2f5556b8580d584bbdde77a.jpg)

![](images/0be96e205b325adb39c7000f22c40e23be93c6d6aaf4be1b8801da55ee0db460.jpg)

![](images/fa8b6dc260f43d93af2c986eab33e15dfccda9349ad3a547d2ce86916d07590d.jpg)

![](images/76794451017c15e7d5e944518d1b81f2911adf644f6094a6600b959e8ac3a65b.jpg)  
Figure 3: Final evaluation on the locomotion tasks Cassie and Laikago with three simultaneous constraints. Costs 0, 1, and 2 denote the balance, height, and foot timing constraints. Mean–std constraints use $\alpha _ { \mathrm { l o } } \in \{ 0 . 0 5 , 0 . 2 5 \}$ . Cost axes are shown on a log scale.

Fig. 1 summarizes v2 results on BC, CC, and DC. LOCU consistently satisfies all constraint limits across agents while achieving the highest returns on BC and DC among methods satisfying all constraints, and remaining competitive on CC. The PPOL variants exhibit instability in this multiconstraint setting, where the trained policy fails to reliably track the circular objective. Therefore, we omit PPOL-GS and -V from subsequent experiments.

Fig. 2 summarizes the BC-v3 results under feasibility-tightening and redundancy variants. Feasibility tightening includes low-velocity constraints, which shrink the feasible set and reshape early learning by forcing the agent to increase speed before stable circle tracking, thereby delaying reward improvement. This effect is stronger when train starts from infeasible policies: trust-region and primal methods prioritize feasibility recovery for longer as thresholds tighten, while Lagrangian methods may preserve higher return through multiplier adaptation but show less stable constraint satisfaction when aggressively growing multipliers destabilize optimization. The redundancy variants build on the base v3 setting with a single low-velocity coefficient of $0 . 6 ,$ simultaneously shrinking the feasible set and inducing near-collinear constraint directions. Across both tightening and redundancy variants, LOCU maintains stable constraint satisfaction while limiting return degradation relative to the baselines.

Figure 3 summarizes the results on Cassie and Laikago with three simultaneous constraints. LOCU and $\mathrm { S D A C ( A ) }$ achieve the best returns while consistently satisfying all constraint limits, whereas PCRPO does not reliably converge in our multi-constraint locomotion setup. Since PCRPO reduces oscillations through soft-switching, we interpret this result as indicating that additional stabilization may be required in coupled multi-constraint regimes with intermittent violations and cross-constrain scale mismatch.

![](images/d1750a57b91aec8255776c9736044d35320aa2d1133c9b4451a5366488f25731.jpg)  
Figure 4: Final evaluation on the Cassie goal task with six constraints. Costs 0, 1, and 2 denote the balance, height, and foot timing constraints. And costs 3, 4 and 5 denote speed, hazard contact, energy constraints. Cost axes are shown on a log scale and 0.25 is used for α<sub>lo</sub>.

Additional results on the goal-navigation task, including two variants using Clarabel, are summarized in Figure 4. Here, SDAC(A) misallocates recovery weights as the QP becomes severely ill-conditioned. Unlike the original locomotion task, where the diagonal terms are comparable, body tilt and height constraints violate their limits by roughly 30 times, resulting in much larger diagonal terms than weak-gradient constraints such as energy or obstacle contact. The dual update aims to reduce all active constraints without considering the violation levels, leading to large recovery weights with little correction and negligible weights on the constraints that require correction most. Full QP variants which using the same set of natural gradients exhibit similar behavior, although solving with Clarabel partially improves recovery. Notably, solver flagged only 2 of approximately 24,900 updates across five seeds as inaccurate, indicating that the remaining degradation stems from severe ill-conditioning of the QP itself rather than from frequent solver failures. LOCU avoids this by screening out inactive or degenerate constraints, and constructing an orthogonal basis of the retained gradients. Despite some constraint violations, the Reduced QP variants which use LOCU’s orthogonal basis perform closest to LOCU in both return and constraint satisfaction. This suggests that LOCU’s orthogonal basis construction can be integrated into existing algorithms to enhance numerical stability, while its proposed active-set solver can be replaced with alternative solvers depending on computational and real-time requirements. We provide the additional robustness test on goal-object task with redundancy in Appendix C.4.3.

## 6 Limitations and future works

LOCU provides local feasibility control but no global convergence guarantee as other second order trust region methods [1]. A possible approach is to reinterpret KL-induced Fisher–Rao geometry through a Bregman-proximal mirror-descent perspective over the convex occupancy-measure CMDP, which may provide a route to primal-dual convergence analysis. Also, computing natural-gradient directions and the Fisher Gram matrix for all constraints may raise computational-efficiency concerns. However, as discussed in Appendix C.3.1, parallelized constraint processing can mitigate this issue by better utilizing GPU throughput. It may also be useful to extend natural gradient calculation with curvature approximations to further reduce computation of CG iters and Fisher-related functions.

## 7 Conclusion

We introduced LOCU, a second-order framework for multi-constraint policy optimization that mitigates constraint coupling in a geometry consistent manner. By applying symmetric Löwdin orthogonalization to natural gradients under the Fisher–Rao metric, LOCU yields an orthonormal reduced coordinate system in which the KL trust region becomes a Euclidean ball and constraint interactions are handled in a numerically stable way. Across multi-constraint benchmarks with tight feasible regions, redundant constraints, and infeasible starts, LOCU achieves stable constraint handling with competitive returns across the starting feasibility and critic architectures.

## Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) through the Information Technology Research Center program (IITP-2026-RS-2023-00260098) and the Artificial Intelligence Convergence Innovation Human Resources Development program (IITP-2026-RS-2023-00254177).

We thank Google TPU Research Cloud (TRC) for granting us access to TPUs. Additionally, J.L.   
thanks Nixon Son for the discussion that inspired the initial idea behind this work.

## References

[1] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pages 22–31. PMLR, 2017.

[2] Eitan Altman. Constrained Markov decision processes. Routledge, 2021.

[3] Shun-ichi Amari. Information geometry and its applications, volume 194. Springer, 2016.

[4] Nihat Ay, Jürgen Jost, Hông Vân Lê, and Lorenz Schwachhöfer. Information geometry, volume 64. Springer, 2017.

[5] Lukas Biewald. Experiment tracking with weights and biases, 2020. URL https://www.wandb.com/. Software available from wandb.com.

[6] Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

[7] Paul J Goulart and Yuwen Chen. Clarabel: An interior-point solver for conic programs with quadratic objectives (2024). arXiv preprint arXiv:2405.12762.

[8] Sven Gronauer. Bullet-safety-gym: A framework for constrained reinforcement learning. Technical report, mediaTUM, 2022.

[9] Shangding Gu, Bilgehan Sel, Yuhao Ding, Lu Wang, Qingwei Lin, Ming Jin, and Alois Knoll. Balance reward and safety optimization for safe reinforcement learning: A perspective of gradient manipulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21099–21106, 2024.

[10] Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, and Yaodong Yang. Safety gymnasium: A unified safe reinforcement learning benchmark. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=WZmlxIuIGR.

[11] Jiaming Ji, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, and Yaodong Yang. Omnisafe: An infrastructure for accelerating safe reinforcement learning research. Journal ofMachine Learning Research, 25(285):1–6, 2024.

[12] Sham M Kakade. A natural policy gradient. Advances in neural information processing systems, 14, 2001.

[13] Danial Kamran, Thiago D. Simão, Qisong Yang, Canmanie T. Ponnambalam, Johannes Fischer, Matthijs T. J. Spaan, and Martin Lauer. A modern perspective on safe automated driving for different traffic dynamics using constrained reinforcement learning. In ITSC, pages 4017–4023. IEEE, 2022. doi: 10.1109/ITSC55140.2022.9921907.

[14] Dohyeong Kim, Kyungjae Lee, and Songhwai Oh. Trust region-based safe distributional reinforcement learning for multiple constraints. Advances in neural information processing systems, 36:19908–19939, 2023.

[15] Dohyeong Kim, Taehyun Cho, Seungyub Han, Hojun Chung, Kyungjae Lee, and Songhwai Oh. Spectralrisk safe reinforcement learning with convergence guarantees. Advances in Neural Information Processing Systems, 37:92465–92498, 2024.

[16] Zuxin Liu, Zijian Guo, Haohong Lin, Yihang Yao, Jiacheng Zhu, Zhepeng Cen, Hanjiang Hu, Wenhao Yu, Tingnan Zhang, Jie Tan, and Ding Zhao. Datasets and benchmarks for offline safe reinforcement learning. Journal ofData-centric Machine Learning Research, 2024.

[17] Per-Olov Löwdin. On the non-orthogonality problem connected with the use of atomic wave functions in the theory of molecules and crystals. The Journal ofChemical Physics, 18(3):365–375, 1950.

[18] Jorge Nocedal and Stephen J Wright. Numerical optimization. Springer, 2006.

[19] Santiago Paternain, Luiz Chamon, Miguel Calvo-Fullana, and Alejandro Ribeiro. Constrained reinforcement learning has zero duality gap. Advances in Neural Information Processing Systems, 32, 2019.

[20] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

[21] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[22] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd. OSQP: an operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4):637–672, 2020. doi: 10.1007/ s12532-020-00179-2. URL https://doi.org/10.1007/s12532-020-00179-2.

[23] Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive safety in reinforcement learning by pid lagrangian methods. In International Conference on Machine Learning, pages 9133–9143. PMLR, 2020.

[24] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.

[25] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.

[26] Tengyu Xu, Yingbin Liang, and Guanghui Lan. Crpo: A new approach for safe reinforcement learning with convergence guarantee. In International Conference on Machine Learning, pages 11480–11491. PMLR, 2021.

[27] Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li, Yaodong Yang, and Gang Pan. Constrained update projection approach to safe policy optimization. Advances in Neural Information Processing Systems, 35:9111–9124, 2022.

[28] Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, and Peter J Ramadge. Projection-based constrained policy optimization. arXiv preprint arXiv:2010.03152, 2020.

[29] Yihang Yao, Zuxin Liu, Zhepeng Cen, Peide Huang, Tingnan Zhang, Wenhao Yu, and Ding Zhao. Gradient shaping for multi-constraint safe reinforcement learning. In 6th Annual Learning for Dynamics & Control Conference, pages 25–39. PMLR, 2024.

[30] Yiming Zhang, Quan Vuong, and Keith Ross. First order constrained optimization in policy space. Advances in Neural Information Processing Systems, 33:15338–15349, 2020.

[31] Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu. On the convergence of stochastic multi-objective gradient manipulation and beyond. Advances in Neural Information Processing Systems, 35:38103–38115, 2022.

## A Supplementary proofs and implementation details

This appendix provides (i) proofs omitted in the main text and (ii) additional derivations and implementation notes. To keep the section hierarchy consistent, all omitted proofs are grouped under Section A.2. The remaining sections provide supporting material (reduced-space KKT details, screening notes, recovery mechanics, empirical KL control, and projection-pathology analysis).

## A.1 Notation and shapes

Let K be the enforced constraint index set after screening, and write $\bar { K } : = | K |$ . Let $\{ \mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { m } \}$ be the near-collinear clusters and $\{ i _ { 1 } , \ldots , i _ { m } \}$ their representatives. Define

$$
\begin{array} { r } { S \in \mathbb R ^ { n \times ( m + 1 ) } , \quad O : = S ^ { \top } F S \in \mathbb R ^ { ( m + 1 ) \times ( m + 1 ) } , \quad E : = S O ^ { - 1 / 2 } \in \mathbb R ^ { n \times ( m + 1 ) } . } \end{array}
$$

Throughout the appendix we work with the generic reduced program

$$
\operatorname* { m a x } _ { w \in \mathbb { R } ^ { m + 1 } } u ^ { \top } w \quad \mathrm { s . t . } \quad \| w \| _ { 2 } \leq R , \quad C w \leq \Delta ,
$$

where $u \in \mathbb { R } ^ { m + 1 }$ is the objective coefficient vector in reduced coordinates. In LOCU, u is set depending on the update mode:

$$
u : = { \left\{ \begin{array} { l l } { E ^ { \top } g _ { r } } & { { \mathrm { ( f e a s i b l e ~ c a s e ) } } , } \\ { { \bar { g } } _ { p } } & { { \mathrm { ( i n f e a s i b l e ~ c a s e ; ~ S e c t i o n ~ } } 4 . 4 { \mathrm { ) } } . } \end{array} \right. }
$$

For each enforced constraint index $k \in { \mathcal { K } } ,$ , define the reduced normal

$$
C _ { k } ^ { \top } : = E ^ { \top } g _ { c _ { k } } \in \mathbb { R } ^ { m + 1 } .
$$

After fixing an ordering of $\kappa ,$ we assemble

$$
C \in \mathbb { R } ^ { \bar { K } \times ( m + 1 ) } , \qquad \Delta \in \mathbb { R } ^ { \bar { K } } , \qquad R = \sqrt { 2 \delta } .
$$

We use the Fisher inner product $\langle x , y \rangle _ { F } : = x ^ { \top } F y$ and norm $\| x \| _ { F } : = { \sqrt { x ^ { \top } F x } }$

## A.2 Proofs omitted in the main text

## A.2.1 Lemma 3.2

Lemma 3.2. Problem (9) admits an optimal solution $\Delta \theta ^ { \star }$ such that $\Delta \theta ^ { \star } \in \mathrm { s p a n } \{ \tilde { g } _ { r } , \tilde { g } _ { c _ { 1 } } , . . . , \tilde { g } _ { c _ { K } } \}$

Proof. Let

$$
\mathcal { U } : = \operatorname { s p a n } \{ \tilde { g } _ { r } , \tilde { g } _ { c _ { 1 } } , \dots , \tilde { g } _ { c _ { K } } \} .
$$

Take any $\Delta \theta \in \mathbb { R } ^ { n }$ and decompose it as

$$
\Delta \theta = \Delta \theta _ { \parallel } + \Delta \theta _ { \perp } , \qquad \Delta \theta _ { \parallel } \in \mathcal { U } , \qquad \Delta \theta _ { \perp } \in \mathcal { U } ^ { \perp _ { F } } ,
$$

where $\mathcal { U } ^ { \perp _ { F } } : = \{ v : \langle u , v \rangle _ { F } = 0 , \forall u \in \mathcal { U } \}$

First, the linearized reward objective depends only on $\Delta \theta _ { \parallel }$ :

$$
\begin{array} { r } { g _ { r } ^ { \top } \Delta \theta = ( F \tilde { g } _ { r } ) ^ { \top } \Delta \theta = \tilde { g } _ { r } ^ { \top } F \Delta \theta = \langle \tilde { g } _ { r } , \Delta \theta _ { \parallel } + \Delta \theta _ { \perp } \rangle _ { F } = \langle \tilde { g } _ { r } , \Delta \theta _ { \parallel } \rangle _ { F } , } \end{array}
$$

because $\tilde { g } _ { r } \in \mathcal { U }$ and $\Delta \theta _ { \perp } \in \mathcal { U } ^ { \perp _ { F } }$

Similarly, each constraint depends only on $\Delta \theta _ { \parallel } \colon$ for any $k ,$

$$
\begin{array} { r } { g _ { c _ { k } } ^ { \top } \Delta \theta = ( F \tilde { g } _ { c _ { k } } ) ^ { \top } \Delta \theta = \tilde { g } _ { c _ { k } } ^ { \top } F \Delta \theta = \langle \tilde { g } _ { c _ { k } } , \Delta \theta _ { \parallel } + \Delta \theta _ { \perp } \rangle _ { F } = \langle \tilde { g } _ { c _ { k } } , \Delta \theta _ { \parallel } \rangle _ { F } , } \end{array}
$$

since $\tilde { g } _ { c _ { k } } \in \mathcal { U }$

Finally, the trust-region quadratic term satisfies the Pythagorean decomposition:

$$
\begin{array} { r } { \| \Delta \theta \| _ { F } ^ { 2 } = \langle \Delta \theta _ { \parallel } + \Delta \theta _ { \perp } , \Delta \theta _ { \parallel } + \Delta \theta _ { \perp } \rangle _ { F } = \| \Delta \theta _ { \parallel } \| _ { F } ^ { 2 } + \| \Delta \theta _ { \perp } \| _ { F } ^ { 2 } \geq \| \Delta \theta _ { \parallel } \| _ { F } ^ { 2 } . } \end{array}
$$

Therefore, if ∆θ is feasible for (9), then $\Delta \theta _ { \parallel }$ is also feasible (it does not increase the KL quadratic term and keeps all linear terms unchanged) and achieves the same objective value. Hence an optimal solution exists in U. □

## A.2.2 Lemma 4.1

Lemma 4.1. $I f O \succ 0$ (or ifO is regularized to be positive definite), then

$$
E ^ { \top } F E = I .
$$

Proof. Recall the definitions $E = S O ^ { - 1 / 2 }$ and ${ \cal O } = { \cal S } ^ { \top } { \cal F S }$ . Then

$$
E ^ { \top } F E = \left( S O ^ { - 1 / 2 } \right) ^ { \top } F \left( S O ^ { - 1 / 2 } \right) = O ^ { - 1 / 2 } S ^ { \top } F S O ^ { - 1 / 2 } = O ^ { - 1 / 2 } O O ^ { - 1 / 2 } = I .
$$

## A.2.3 Proposition 4.2

Proposition 4.2. Let $e _ { j }$ be the jth canonical basis vector in $\mathbb { R } ^ { m + 1 }$ , and let the columns of S be ordered as in (11). Then

$$
E ^ { \top } g _ { r } = O ^ { 1 / 2 } e _ { 1 } , \qquad E ^ { \top } g _ { c _ { i } } = O ^ { 1 / 2 } e _ { j + 1 } , \quad j = 1 , \dots , m .
$$

Proof. Let $e _ { 1 }$ be the first canonical vector in $\mathbb { R } ^ { m + 1 }$ . By construction of S in (11), we have $S e _ { 1 } = \tilde { g } _ { r }$ and $g _ { r } = F \tilde { g } _ { r }$ . Compute

$$
S ^ { \top } g _ { r } = S ^ { \top } F \tilde { g } _ { r } = S ^ { \top } F ( S e _ { 1 } ) = ( S ^ { \top } F S ) e _ { 1 } = O e _ { 1 } .
$$

Therefore,

$$
E ^ { \top } g _ { r } = ( S O ^ { - 1 / 2 } ) ^ { \top } g _ { r } = O ^ { - 1 / 2 } S ^ { \top } g _ { r } = O ^ { - 1 / 2 } ( O e _ { 1 } ) = O ^ { 1 / 2 } e _ { 1 } .
$$

For the representative constraint $i _ { j }$ , the $( j + 1 )$ th column of $S$ equals $\tilde { g } _ { c _ { i _ { j } } } , \mathrm { i } . \mathrm { e } . , S e _ { j + 1 } = \tilde { g } _ { c _ { i _ { j } } }$ , and $g _ { c _ { i _ { j } } } = F \tilde { g } _ { c _ { i _ { j } } }$ . Repeating the same steps gives

$$
E ^ { \top } g _ { c _ { i _ { j } } } = O ^ { 1 / 2 } e _ { j + 1 } .
$$

## A.2.4 Theorem 4.3

Theorem 4.3. Assume $O \succ 0 ( o r O$ is regularized so that $O \succ 0 ) .$ . Fix any enforced constraint index set K and any representative set $\{ i _ { 1 } , \ldots , i _ { m } \}$ used toform $S , O , E$ as in (11)–(13). Then the change of variables $\Delta \theta = E w$ gives a one-to-one correspondence between span(S) and $\mathbb { R } ^ { m + 1 }$ and transforms the trust-region CMDP subproblem (9) restricted to $\Delta \theta \in \operatorname { s p a n } ( \mathbf { \hat { \boldsymbol { S } } } )$ and constraints $k \in \mathcal { K }$ into the reduced program (17). In particular, any optimizer w<sup>⋆</sup> of (17) yields a valid update $\Delta \theta ^ { \star } = E w ^ { \star }$ in the original parameter space.

Proof. Since $O \succ 0$ , the matrix $O ^ { - 1 / 2 }$ exists and is invertible. Hence $E = S O ^ { - 1 / 2 }$ has full column rank, and the map $w \mapsto \Delta \theta = E _ { 1 }$ w is a bijection between $\mathbb { R } ^ { m + 1 }$ and $\operatorname { s p a n } ( E ) = \operatorname { s p a n } ( S )$

We now rewrite each term in (9) under the substitution $\Delta \theta = E w$

Objective. We have

$$
\begin{array} { r } { g _ { r } ^ { \top } \Delta \theta = g _ { r } ^ { \top } E w = ( E ^ { \top } g _ { r } ) ^ { \top } w = u ^ { \top } w , } \end{array}
$$

with $u = E ^ { \top } g _ { r }$ in the feasible mode.

Constraints. For any enforced constraint $k \in \mathcal { K }$

$$
\begin{array} { r } { g _ { c _ { k } } ^ { \top } \Delta \theta = g _ { c _ { k } } ^ { \top } E w = ( E ^ { \top } g _ { c _ { k } } ) ^ { \top } w = C _ { k } ^ { \top } w . } \end{array}
$$

Collecting all such constraints yields $C w \leq \Delta$

Trust region. Using Lemma 4.1,

$$
\begin{array} { r } { \frac 1 2 \Delta \theta ^ { \top } F \Delta \theta = \frac 1 2 ( E w ) ^ { \top } F ( E w ) = \frac 1 2 w ^ { \top } ( E ^ { \top } F E ) w = \frac 1 2 \| w \| _ { 2 } ^ { 2 } . } \end{array}
$$

Thus $\begin{array} { r } { \frac { 1 } { 2 } \Delta \theta ^ { \top } F \Delta \theta \leq \delta } \end{array}$ is equivalent to $\| w \| _ { 2 } \leq R .$

Combining these identities shows that (9) restricted to $\Delta \theta \in \operatorname { s p a n } ( S )$ is exactly (17) in the variable w. Optimal solutions correspond under the bijection $\Delta \theta = E \bar { w }$ □

## A.2.5 Lemma 4.4

Lemma 4.4. For any $\Delta \theta$ satisfying $\| \Delta \theta \| _ { F } \le R$

$$
g _ { c _ { k } } ^ { \top } \Delta \theta \leq \nu _ { k } R .
$$

Proof. Using $g _ { c _ { k } } = F \tilde { g } _ { c _ { k } }$

$$
\begin{array} { r } { \boldsymbol { g } _ { c _ { k } } ^ { \top } \Delta \boldsymbol { \theta } = ( F \tilde { g } _ { c _ { k } } ) ^ { \top } \Delta \boldsymbol { \theta } = \tilde { g } _ { c _ { k } } ^ { \top } F \Delta \boldsymbol { \theta } = \langle \tilde { g } _ { c _ { k } } , \Delta \boldsymbol { \theta } \rangle _ { F } . } \end{array}
$$

By Cauchy–Schwarz in the Fisher inner product,

$$
\langle \tilde { g } _ { c _ { k } } , \Delta \theta \rangle _ { F } \leq \| \tilde { g } _ { c _ { k } } \| _ { F } \| \Delta \theta \| _ { F } .
$$

If $\| \Delta \theta \| _ { F } \le R ,$ , then

$$
g _ { c _ { k } } ^ { \top } \Delta \theta \leq \| \tilde { g } _ { c _ { k } } \| _ { F } R .
$$

Finally,

$$
\| \tilde { g } _ { c _ { k } } \| _ { F } ^ { 2 } = \tilde { g } _ { c _ { k } } ^ { \top } F \tilde { g } _ { c _ { k } } = ( O _ { \mathrm { f u l l } } ) _ { k + 1 , k + 1 } = \nu _ { k } ^ { 2 } ,
$$

so $\| \tilde { g } _ { c _ { k } } \| _ { F } = \nu _ { k }$

## A.2.6 Proposition 4.5

Proposition 4.5. $H f \Delta _ { k } \ \geq \ \nu _ { k } R ,$ , then constraint k cannot be active inside the trust region $\begin{array} { r } { \frac 1 2 \Delta \hat { \theta } ^ { \top } \boldsymbol { F } \Delta \theta \leq \delta } \end{array}$ and can be excluded from K without changing feasibility within that trust region.

Proof. Inside $\begin{array} { r } { \frac { 1 } { 2 } \Delta \theta ^ { \top } F \Delta \theta \leq \delta } \end{array}$ , we have $\| \Delta \theta \| _ { F } \le \sqrt { 2 \delta } = R$ . Lemma 4.4 gives, for all such $\Delta \theta ,$

$$
g _ { c _ { k } } ^ { \top } \Delta \theta \leq \nu _ { k } R .
$$

If $\Delta _ { k } \geq \nu _ { k } R .$ , then $g _ { c _ { k } } ^ { \top } \Delta \theta \leq \Delta _ { k }$ holds automatically for every point in the trust region. Hence constraint k cannot cut the feasible set within the trust region and can be removed without changing feasibility there. □

## A.2.7 Lemma 4.6

Lemma 4.6. Let $u _ { i } , u _ { j }$ be Fisher-unit vectors with $\langle u _ { i } , u _ { j } \rangle _ { F } = \rho \in [ - 1 , 1 ]$ . Then for any $\Delta \theta$ with $\| \Delta \theta \| _ { F } \le R$

$$
\operatorname* { m i n } _ { s \in \{ + 1 , - 1 \} } \left| \langle u _ { i } , \Delta \theta \rangle _ { F } - s \langle u _ { j } , \Delta \theta \rangle _ { F } \right| \leq \sqrt { 2 ( 1 - | \rho | ) } R .
$$

In particular, $i f \vert \rho \vert \geq 1 - \varepsilon _ { \mathrm { d u p } } ,$ , the discrepancy is at most $\sqrt { 2 \varepsilon _ { \mathrm { d u p } } } R$ . One may take $s = + 1$ when $\rho \geq 0$ and $s = - 1$ when $\rho < 0$

Proof. Let $u _ { i } , u _ { j }$ satisfy $\| u _ { i } \| _ { F } = \| u _ { j } \| _ { F } = 1$ and $\langle u _ { i } , u _ { j } \rangle _ { F } = \rho \in [ - 1 , 1 ]$ . For any $s \in \{ + 1 , - 1 \}$

$$
\begin{array} { r l } & { \| u _ { i } - s u _ { j } \| _ { F } ^ { 2 } = \langle u _ { i } - s u _ { j } , u _ { i } - s u _ { j } \rangle _ { F } } \\ & { \qquad = \| u _ { i } \| _ { F } ^ { 2 } + \| u _ { j } \| _ { F } ^ { 2 } - 2 s \langle u _ { i } , u _ { j } \rangle _ { F } } \\ & { \qquad = 2 ( 1 - s \rho ) . } \end{array}
$$

Choose $s = + 1$ when $\rho \geq 0$ and $s = - 1$ when $\rho < 0 ,$ , so that $s \rho = | \rho |$ and hence

$$
\| u _ { i } - s u _ { j } \| _ { F } = \sqrt { 2 ( 1 - | \rho | ) } .
$$

Then for any $\Delta \theta ,$

$$
\left| \langle u _ { i } , \Delta \theta \rangle _ { F } - s \langle u _ { j } , \Delta \theta \rangle _ { F } \right| = \left| \langle u _ { i } - s u _ { j } , \Delta \theta \rangle _ { F } \right|
$$

$$
\leq \| u _ { i } - s u _ { j } \| _ { F } \| \Delta \theta \| _ { F } \leq \sqrt { 2 ( 1 - | \rho | ) } R ,
$$

whenever $\| \Delta \theta \| _ { F } \le R$ . Taking the minimum over $s \in \{ + 1 , - 1 \}$ yields the claim.

## A.2.8 Lemma 4.7

Lemma 4.7. $I f \bar { g } _ { p } \ne 0$ defined by (27)–(28), then for every $i \in \mathcal { V } ,$

$$
{ \cal C } _ { i } ^ { \top } \bar { g } _ { p } < 0 .
$$

Proof. Let the violated reduced normals be $\{ C _ { i } \} _ { i \in \mathcal { V } }$ (rows of C indexed by V). Define the convex hull

$$
\mathcal { Q } : = \mathrm { c o n v } \{ C _ { i } : i \in \mathcal { V } \} = \left\{ \sum _ { i \in \mathcal { V } } \alpha _ { i } C _ { i } : \alpha \geq 0 , \mathbf { 1 } ^ { \top } \alpha = 1 \right\} .
$$

Problem (27) computes

$$
g ^ { \star } = \arg \operatorname* { m i n } _ { g \in \mathcal { Q } } \| g \| _ { 2 } ^ { 2 } , \qquad g ^ { \star } = C _ { \mathcal { V } } ^ { \top } \alpha ^ { \star } , \qquad \bar { g } _ { p } = - g ^ { \star } .
$$

Since $g ^ { \star }$ is the Euclidean projection of 0 onto the closed convex set $\mathcal { Q } ,$ the projection optimality condition implies that for all $g \in \mathcal { Q }$

$$
\left( g - g ^ { \star } \right) ^ { \top } ( 0 - g ^ { \star } ) \leq 0 .
$$

Rearranging gives

$$
g ^ { \top } g ^ { \star } \geq \| g ^ { \star } \| _ { 2 } ^ { 2 } .
$$

Taking $g = C _ { i }$ (note that each $C _ { i } \in \mathcal { Q } )$ yields, for every $i \in \mathcal V$

$$
C _ { i } ^ { \top } g ^ { \star } \geq \| g ^ { \star } \| _ { 2 } ^ { 2 } .
$$

If $\bar { g } _ { p } \neq 0$ , then $g ^ { \star } \neq 0$ and $\| g ^ { \star } \| _ { 2 } ^ { 2 } > 0$ . Therefore,

$$
C _ { i } ^ { \top } \bar { g } _ { p } = - C _ { i } ^ { \top } g ^ { \star } \le - \| g ^ { \star } \| _ { 2 } ^ { 2 } < 0 .
$$

## A.3 Reduced QP derivation and KKT solver

Here, we provide a detailed KKT-based solution procedure for the reduced program (17). The presentation parallels the style of the analytical LQCLP derivation in CPO ([1], Thm. 2), but here we treat the multi-constraint case and exploit the fact that LOCU’s Löwdin basis makes the trust region Euclidean $( \mathrm { i . e . }$ , the quadratic form is I in reduced coordinates). In particular, when $| { \mathcal { I } } | = 1$ , the formulas below reduce exactly to the CPO single-constraint LQCLP solution specialized to $H = I$

Problem statement. Recall the reduced step subproblem (with $p : = m + 1 )$

$$
\begin{array} { r l } { \underset { w \in \mathbb { R } ^ { p } } { \operatorname* { m a x } } } & { u ^ { \top } w } \\ { \mathrm { s . t . } } & { \| w \| _ { 2 } \leq R , } \\ & { C w \leq \Delta , } \end{array}\tag{31}
$$

where $C \in \mathbb { R } ^ { \bar { K } \times p }$ collects the reduced constraint normals and $\Delta \in \mathbb { R } ^ { \bar { K } }$ collects the corresponding slacks. In LOCU, u is either $E ^ { \top } g _ { r }$ (feasible mode) or the recovery direction $\bar { g } _ { p }$ (feasibility-recovery mode), but the solver below is identical.

Convexity and dual attainability. (31) is a linear maximization over a closed convex set (intersection of a Euclidean ball and halfspaces), hence an optimum exists whenever the feasible set is nonempty. In LOCU, nonemptiness is ensured in the recovery mode because we only constrain already-satisfied constraints and thus $w = 0$ is feasible; in the reward mode, $w = 0$ is feasible whenever $\Delta \geq 0$

## A.3.1 KKT conditions

Strong duality and KKT sufficiency. (31) is a convex SOCP (linear objective with a Euclidean ball and halfspace constraints). Whenever the reduced feasible set has a nonempty relative interior $( \mathrm { e . g . }$ ., there exists w such that $\| w \| _ { 2 } < R$ and $C w < \Delta$ for the effective inequality set), strong duality holds and KKT conditions are necessary and sufficient for optimality [6]. If the problem is (nearly) degenerate in practice, our implementation handles it using regularization/tolerances in the active-set routine described below.

Write (31) as a minimization

$$
\operatorname* { m i n } _ { w } \ - u ^ { \top } w \quad \mathrm { s . t . } \quad C w - \Delta \leq 0 , \qquad \| w \| _ { 2 } ^ { 2 } - R ^ { 2 } \leq 0 .
$$

The Lagrangian is

$$
\mathcal { L } ( w , \lambda , \beta ) = - u ^ { \top } w + \lambda ^ { \top } ( C w - \Delta ) + \beta ( \| w \| _ { 2 } ^ { 2 } - R ^ { 2 } ) ,
$$

where $\lambda \in \mathbb { R } ^ { \bar { K } }$ and $\beta \in \mathbb { R }$ satisfy $\lambda \ge 0 , \beta \ge 0$

Stationarity:

$$
- u + C ^ { \top } \lambda + 2 \beta w = 0 .\tag{32}
$$

Primal feasibility:

$$
C w \leq \Delta , \qquad \| w \| _ { 2 } \leq R .\tag{33}
$$

Dual feasibility:

$$
\lambda \ge 0 , \qquad \beta \ge 0 .\tag{34}
$$

Complementary slackness:

$$
\lambda _ { i } \big ( C _ { i } ^ { \top } w - \Delta _ { i } \big ) = 0 \ \forall i , \qquad \beta \big ( \| w \| _ { 2 } ^ { 2 } - R ^ { 2 } \big ) = 0 .\tag{35}
$$

Basic structure. If $u \neq 0$ , the optimum typically lies on the ball boundary, i.e., $\| w ^ { \star } \| _ { 2 } = R$ and thus $\beta ^ { \star } > 0$ . (If the halfspaces alone already form a bounded polytope that contains the maximizer strictly inside the ball, then $\beta ^ { \star } = 0$ is possible; we include this case below for completeness.)

## A.3.2 Closed form for a fixed active set

Let $\mathcal { I } \subseteq \{ 1 , \ldots , \bar { K } \}$ denote a candidate active set (constraints that are tight at optimum), and write $C _ { \mathcal { I } } \in \mathbb { R } ^ { | \mathcal { I } | \times p }$ and $\Delta _ { \mathcal { I } } \in \mathbb { R } ^ { | \mathcal { I } | }$ for the corresponding rows/entries. In KKT form,

$$
C _ { \mathcal { T } } w = \Delta _ { \mathcal { T } } , \qquad \lambda _ { \mathcal { T } } \geq 0 , \qquad \lambda _ { \bar { \mathcal { T } } } = 0 .
$$

Eliminating w in terms of $( \beta , \lambda _ { \mathcal { I } } )$ Assume first that $\beta > 0$ . From (32),

$$
w ( \beta , \lambda _ { \mathcal { I } } ) = \frac { 1 } { 2 \beta } \Big ( u - C _ { \mathcal { I } } ^ { \top } \lambda _ { \mathcal { I } } \Big ) .\tag{36}
$$

Imposing the active equalities $C _ { \mathcal { I } } w = \Delta _ { \mathcal { I } }$ yields the reduced linear system

$$
( C _ { \mathcal { I } } C _ { \mathcal { I } } ^ { \top } ) \lambda _ { \mathcal { I } } = C _ { \mathcal { I } } u - 2 \beta \Delta _ { \mathcal { I } } .\tag{37}
$$

This is exactly the multi-constraint analogue of the scalar equations in CPO’s single-constraint LQCLP derivation: there, $| \mathcal { I } | = 1$ and $C _ { \mathcal { I } } \breve { C } _ { \mathcal { I } } ^ { \top }$ is a scalar, leading to a fully analytical solution.

Rank and regularization. If $C _ { \mathcal { I } } C _ { \mathcal { I } } ^ { \top }$ is ill-conditioned, we solve

$$
\left( C _ { \mathcal { I } } C _ { \mathcal { I } } ^ { \top } + \varepsilon I \right) \lambda _ { \mathcal { I } } = C _ { \mathcal { I } } u - 2 \beta \Delta _ { \mathcal { I } } ,
$$

with a small $\varepsilon > 0$ (or use a pseudoinverse). In LOCU, near-collinear compression makes this case rare.

```latex
Algorithm 1 Reduced-space active-set KKT solver for (31)
Require: $u \in \mathbb { R } ^ { p } , C \in \mathbb { R } ^ { \bar { K } \times p } , \Delta \in \mathbb { R } ^ { \bar { K } }$ , radius $R > 0$ , tolerances $\tau _ { \mathrm { f e a s } } , \tau _ { \mathrm { d u a l } }$ , regularization $\varepsilon > 0$
Ensure: $w ^ { \star }$
1: If $\| u \| _ { 2 } \leq \tau _ { \mathrm { d u a l } } ,$ return $w ^ { \star }  0$
2: $w  R u / \| u \| _ { 2 }$ (unconstrained ball maximizer)
3: if $C w \leq \Delta + \tau _ { \mathrm { f e a s } } \mathbf { 1 }$ then
4: return $w ^ { \star }  w$
5: end if
6: Initialize active set $\mathcal { I }  \{ \arg \operatorname* { m a x } _ { i } ( C _ { i } ^ { \top } w - \Delta _ { i } ) \}$ (most violated)
7: for $t = 1 , 2 , \dots$ until convergence do
8: Form $C _ { \mathcal { I } } , \Delta _ { \mathcal { I } } ,$ , compute $\Breve { M } _ { \mathcal { I } } = C _ { \mathcal { I } } C _ { \mathcal { I } } ^ { \top }$
9: Solve $\begin{array} { r } { \left( M _ { \mathcal { I } } + \varepsilon I \right) A _ { \mathcal { I } } ^ { \top } = I \left( \mathrm { i . e . , } A _ { \mathcal { I } } = ( M _ { \mathcal { I } } + \varepsilon I ) ^ { - 1 } \right) } \end{array}$
10: $w _ { p } \gets C _ { \mathcal { T } } ^ { \top } A _ { \mathcal { T } } \Delta _ { \mathcal { I } }$
11: $d  ( I - C _ { \mathcal { T } } ^ { \top } A _ { \mathcal { T } } C _ { \mathcal { T } } ) u$
12: if $\lVert d \rVert _ { 2 } \leq \tau _ { \mathrm { d u a l } }$ then
13: $w  w _ { p } , \beta  0$
14: else
15: Compute $( a , b , c )$ via (41) and set $t ^ { \star }$ by (42)
16: $w  \bar { w } _ { p } + t ^ { \star } d , \beta  1 / ( 2 t ^ { \star } )$
17: end if
18: $\lambda _ { \mathcal { I } }  A _ { \mathcal { I } } \big ( C _ { \mathcal { I } } u - 2 \beta \Delta _ { \mathcal { I } } \big ) , \lambda _ { \bar { \mathcal { I } } }  0$
19: if mi $\boldsymbol { 1 } _ { j \in \mathcal { T } } \boldsymbol { \lambda } _ { j } < - \boldsymbol { \tau } _ { \mathrm { d u a l } }$ then
20: Remove $\begin{array} { r } { \dot { j ^ { \star } } \in \arg \operatorname* { m i n } _ { j \in \mathcal { I } } \lambda _ { j } } \end{array}$ from $\mathcal { I }$ (restore dual feasibility)
21: continue
22: end if
23: if max $( C _ { i } ^ { \top } w - \Delta _ { i } ) > \underline { { \tau } } _ { \mathrm { f e a s } }$ then
24: Add $i ^ { \star } \in$ arg max $\phantom { } _ { ; } ( C _ { i } ^ { \top } w - \Delta _ { i } )$ to $\mathcal { I }$ (restore primal feasibility)
25: continue
26: end if
27: return $w ^ { \star }  w$
28: end for
```

Explicit decomposition using a projector Define

$$
\begin{array} { r } { M _ { \mathcal { I } } : = C _ { \mathcal { I } } C _ { \mathcal { I } } ^ { \top } \in \mathbb { R } ^ { | \mathcal { I } | \times | \mathcal { I } | } , \qquad A _ { \mathcal { I } } : = M _ { \mathcal { I } } ^ { - 1 } , \qquad P _ { \mathcal { I } } : = I - C _ { \mathcal { I } } ^ { \top } A _ { \mathcal { I } } C _ { \mathcal { I } } \in \mathbb { R } ^ { p \times p } . } \end{array}
$$

$P _ { \mathcal { I } }$ is the (Euclidean) orthogonal projector onto the nullspace of $C _ { \mathcal { I } }$ . Also define the minimum-norm particular solution of the active equalities:

$$
w _ { p } : = C _ { \mathcal { I } } ^ { \top } A _ { \mathcal { I } } \Delta _ { \mathcal { I } } \quad \mathrm { s o t h a t } \quad C _ { \mathcal { I } } w _ { p } = \Delta _ { \mathcal { I } } .
$$

Solving (37) and substituting into (36) yields the closed form

$$
w ( \beta ) = w _ { p } + \frac { 1 } { 2 \beta } P \mathcal { I } u .\tag{38}
$$

Thus the solution for a fixed $\mathcal { I }$ lies on the affine space $C _ { \mathcal { I } } w = \Delta _ { \mathcal { I } }$ and moves only along the nullspace direction $P _ { \mathcal { I } } u$ as $\beta$ varies.

Determining $\beta$ from the ball constraint I $\because \beta > 0$ , complementary slackness enforces $\| w ( \beta ) \| _ { 2 } =$ R. Let

$$
d : = P _ { \mathcal { T } } u , \qquad t : = \frac { 1 } { 2 \beta } ( > 0 ) .
$$

Then (38) becomes

$$
w ( t ) = w _ { p } + t d ,\tag{39}
$$

and the ball boundary condition is

$$
\| w _ { p } + t d \| _ { 2 } ^ { 2 } = R ^ { 2 } .\tag{40}
$$

Expanding gives a quadratic equation in t:

$$
a t ^ { 2 } + 2 b t + c = 0 , \qquad a : = \| d \| _ { 2 } ^ { 2 } , \ b : = w _ { p } ^ { \top } d , \ c : = \| w _ { p } \| _ { 2 } ^ { 2 } - R ^ { 2 } .\tag{41}
$$

Case 1: $d \neq 0$ (generic). Then $a > 0$ . Feasibility of the active-equality set inside the ball implies the discriminant is nonnegative:

$$
b ^ { 2 } - a c \geq 0 .
$$

Among the two roots,

$$
t _ { \pm } = { \frac { - b \pm { \sqrt { b ^ { 2 } - a c } } } { a } } ,
$$

the objective increases with t because

$$
\begin{array} { r } { \boldsymbol { u } ^ { \top } \boldsymbol { w } ( t ) = \boldsymbol { u } ^ { \top } \boldsymbol { w } _ { p } + t \boldsymbol { u } ^ { \top } \boldsymbol { d } = \boldsymbol { u } ^ { \top } \boldsymbol { w } _ { p } + t \left. \boldsymbol { d } \right. _ { 2 } ^ { 2 } , } \end{array}
$$

using $u ^ { \top } d = u ^ { \top } P _ { \mathcal { I } } u = \| P _ { \mathcal { I } } u \| _ { 2 } ^ { 2 } = \| d \| _ { 2 } ^ { 2 }$ . Hence the maximizer for the fixed active set uses the larger root:

$$
t ^ { \star } = \frac { - b + \sqrt { b ^ { 2 } - a c } } { a } , \qquad \beta ^ { \star } = \frac { 1 } { 2 t ^ { \star } } , \qquad w ^ { \star } = w _ { p } + t ^ { \star } d .\tag{42}
$$

Case 2: $d = 0$ (degenerate/projection case). Then $P g u = 0$ , meaning $u \in \mathrm { r a n g e } ( C _ { \mathcal { I } } ^ { \top } )$ and the objective is constant over the affine set $C _ { \mathcal { T } } w = \Delta _ { \mathcal { T } } \mathrm { : i f } u = C _ { . 7 } ^ { \top } y$ , then $u ^ { \top } w = y ^ { \top } C _ { \mathcal { T } } w = y ^ { \top } \Delta _ { \mathcal { I } }$ In this case, any feasible w satisfying the active equalities is optimal; we pick the minimum-norm choice $w ^ { \star } = w _ { p } . \mathrm { ~ I f ~ } \| w _ { p } \| _ { 2 } < R .$ , one can set $\beta ^ { \star } = 0 ; \operatorname { i f } \| w _ { p } \| _ { 2 } = R .$ , any $\beta ^ { \star } \geq 0$ satisfies the ball slackness.

Recovering multipliers $\lambda _ { \mathcal { I } }$ Once $t ^ { \star }$ (equivalently $\beta ^ { \star } )$ is determined, the multipliers on the active constraints follow from $( 3 7 ) \colon$

$$
\lambda _ { \mathcal { I } } ^ { \star } = A _ { \mathcal { I } } \Bigl ( C _ { \mathcal { I } } u - 2 \beta ^ { \star } \Delta _ { \mathcal { I } } \Bigr ) = A _ { \mathcal { I } } \Bigl ( C _ { \mathcal { I } } u - \frac { 1 } { t ^ { \star } } \Delta _ { \mathcal { I } } \Bigr ) , \qquad \lambda _ { \mathcal { \bar { I } } } ^ { \star } = 0 .\tag{43}
$$

By construction, $( w ^ { \star } , \lambda ^ { \star } , \beta ^ { \star } )$ satisfies stationarity, the active equalities, and the ball boundary. The remaining KKT requirements are: $( \mathrm { i } ) \lambda _ { \mathcal { I } } ^ { \star } \geq 0$ and (ii) all inactive constraints are satisfied strictly (or at least not violated), i.e., $C w ^ { \star } \leq \Delta$

## A.3.3 Active-set search over $\mathcal { I }$

The closed form above turns the multi-constraint problem into a small-dimensional active-set search: for a candidate $\mathcal { I }$ , we can compute a unique KKT candidate $( w ^ { \star } , \lambda ^ { \star } , \beta ^ { \star } )$ (up to degenerate cases) via (42) and (43), and then verify feasibility and sign conditions. Because the reduced dimension $p = m + 1$ is small, this procedure is inexpensive.

We use a primal active-set loop: start from the unconstrained ball maximizer, add violated constraints until feasibility is reached, and drop constraints with negative multipliers until dual feasibility is restored.

Remarks. (i) The loop terminates when both primal feasibility $( C w \le \Delta )$ and dual feasibility $( \lambda \geq 0 )$ are satisfied; then KKT holds because stationarity and complementarity are enforced by construction. (ii) The only matrix inverse is $( C _ { \mathcal { I } } C _ { . 7 } ^ { \top } + \varepsilon I ) ^ { - 1 }$ with size $| { \mathcal { I } } | \leq p$ . (iii) When $| \mathcal { I } | = \mathrm { i }$ the computation of $t ^ { \star }$ and $\lambda _ { \mathcal { I } } ^ { \star }$ reduces to the familiar two-constraint (linear + quadratic) LQCLP algebra in CPO, with the additional simplification that $H = I$ in LOCU’s reduced coordinates.

Optional: one-dimensional search for $\beta$ (implementation alternative) Instead of solving (41) in closed form, one may determine $\beta$ by a monotone 1D search. For fixed ${ \mathcal { I } } , ( 3 8 )$ implies $\lVert w ( \beta ) \rVert _ { 2 }$ is nonincreasing in $\beta$ whenever $d = \dot { P } _ { \mathcal { T } } u \neq 0$ . Therefore, one can bracket $\beta$ and apply bisection to solve $\lVert w ( \beta ) \rVert _ { 2 } = R$ . This produces the same $w ^ { \star }$ as (42) up to numerical tolerance.

Local approximation error. The preceding LQCLP derivation relies on the local linearization used in CPO [1, Section 6.1, Eq. (11)]. To bound the resulting approximation error, hold the current Fisher matrix $F$ fixed and assume each reward or cost objective $\mathsf { \bar { J } } _ { i } , i \in \{ R , C _ { 1 } , \ldots , C _ { K } \}$ , is twice continuously differentiable near the update segment and satisfies

$$
\operatorname* { s u p } _ { \tau \in [ 0 , 1 ] } \Big \Vert F ^ { - 1 / 2 } \nabla _ { \theta } ^ { 2 } J _ { i } \big ( \pi _ { \theta + \tau \Delta \theta } \big ) F ^ { - 1 / 2 } \Big \Vert _ { 2 } \leq L _ { i } , \qquad L _ { i } \geq 0 .\tag{44}
$$

For any update within the trust region of (9), Taylor’s theorem and the fundamental theorem of calculus give

$$
\begin{array} { r } { \left| J _ { i } \big ( \pi _ { \theta + \Delta \theta } \big ) - J _ { i } \big ( \pi _ { \theta } \big ) - \nabla _ { \theta } J _ { i } ( \pi _ { \theta } ) ^ { \top } \Delta \theta \right| \le L _ { i } \delta , } \end{array}\tag{45}
$$

$$
\begin{array} { r } { \left. F ^ { - 1 / 2 } \bigl ( \nabla _ { \theta } J _ { i } \bigl ( \pi _ { \theta + \Delta \theta } \bigr ) - \nabla _ { \theta } J _ { i } ( \pi _ { \theta } ) \bigr ) \right. _ { 2 } \leq L _ { i } \sqrt { 2 \delta } . } \end{array}\tag{46}
$$

Consequently, any constraint whose linearized inequality is satisfied obeys

$$
\begin{array} { r } { g _ { c _ { k } } ^ { \top } \Delta \theta \leq \Delta _ { k } \quad \implies \quad J _ { C _ { k } } ( \pi _ { \theta + \Delta \theta } ) \leq b _ { k } + L _ { C _ { k } } \delta . } \end{array}\tag{47}
$$

Let $\phi _ { k }$ be the angle between ${ \cal F } ^ { - 1 / 2 } g _ { c _ { k } }$ and $F ^ { - 1 / 2 } \nabla _ { \theta } J _ { C _ { k } } ( \pi _ { \theta + \Delta \theta } )$ , with $\nu _ { k }$ defined in (19). The gradient bound implies

$$
\phi _ { k } \leq \arcsin \left( \frac { L _ { C _ { k } } \sqrt { 2 \delta } } { \nu _ { k } } \right) \quad { \mathrm { i f ~ } } L _ { C _ { k } } \sqrt { 2 \delta } < \nu _ { k } .\tag{48}
$$

Higher local curvature or a smaller original gradient norm weakens this directional bound.

## A.4 Practical implementation

Fisher-vector products and natural gradients. Fisher-vector products are computed using Hessian–vector products of the batch-average KL divergence between the cached old policy and the current policy,

$$
\begin{array} { r } { \bar { D } _ { \mathrm { K L } } ( \pi _ { \theta _ { \mathrm { o l d } } } \parallel \pi _ { \theta } ) : = \mathbb { E } _ { s \sim \mathcal { B } } \Bigl [ \mathrm { K L } \bigl ( \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert s ) \bigr \Vert \pi _ { \theta } ( \cdot \vert s ) \bigr ) \Bigr ] , \qquad F v \approx \nabla _ { \theta } ^ { 2 } \bar { D } _ { \mathrm { K L } } ( \pi _ { \theta _ { \mathrm { o l d } } } \parallel \pi _ { \theta } ) v . } \end{array}\tag{49}
$$

Natural-gradient directions for the reward and constraints are obtained by conjugate-gradient (CG) solves with Tikhonov damping. We use a fixed number of CG iterations and add a diagonal damping term to the implicit Fisher operator to stabilize the solves under finite-sample noise.

Fisher Gram matrix and Löwdin basis construction. Let S stack the reward direction and the representative constraint directions used for basis construction. We form the Fisher Gram matrix

$$
O : = S ^ { \top } F S \in \mathbb { R } ^ { ( m + 1 ) \times ( m + 1 ) } .
$$

Before computing the inverse square root, we symmetrize and regularize O as

$$
\begin{array} { r } { O  \frac { 1 } { 2 } ( O + O ^ { \top } ) + \varepsilon _ { \mathrm { r e g } } I . } \end{array}
$$

We compute $O ^ { - 1 / 2 }$ via eigendecomposition,

$$
O = U \Lambda U ^ { \top } , \qquad O ^ { - 1 / 2 } = U \Lambda ^ { - 1 / 2 } U ^ { \top } ,
$$

where eigenvalue flooring is applied by replacing each $\lambda _ { j }$ with max $\{ \lambda _ { j } , \lambda _ { \operatorname* { m i n } } \}$ prior to inversion. This prevents numerical instability when the Gram matrix is ill-conditioned. The Löwdin basis is then constructed as

$$
E : = S O ^ { - 1 / 2 } .
$$

Reduced quantities and parameter update. Reduced-space quantities are obtained by projection onto the Löwdin basis:

$$
u : = E ^ { \top } g _ { r } , \qquad C _ { k } ^ { \top } : = E ^ { \top } g _ { c _ { k } } , \quad k \in \mathcal { K } .
$$

The trust-region radius is set to $R = { \sqrt { 2 \delta } }$ . After solving the reduced program for $w ^ { \star }$ , the actor update in parameter space is given by

$$
\Delta \theta ^ { \star } = E w ^ { \star } .
$$

Reward-aligned constraint filtering (optional). During basis construction, constraints whose Fisher cosine similarity with the reward direction exceeds a fixed threshold are optionally excluded from basis candidates. This heuristic avoids introducing nearly collinear reward–constraint directions into the reduced basis. In our benchmark environments, strong alignment between reward and constraint gradients is rarely observed in practice, and we found that disabling this filter does not measurably affect performance. Accordingly, we retain this step as an optional numerical safeguard rather than a core assumption of the method.

Algorithm 2 LOCU one actor update   
Require: policy $\pi _ { \theta } .$ , trust region δ, limits $\{ b _ { k } \} _ { k = 1 } ^ { K } ,$ , thresholds $\varepsilon _ { \mathrm { n { z } } } , \varepsilon _ { \mathrm { d u p } } ,$ regularization $\varepsilon _ { \mathrm { r e g } }$   
1: Estimate reward and cost gradients $g _ { r } , \{ g _ { c _ { k } } \} _ { k = 1 } ^ { K }$   
2: Estimate Fisher-vector products for the FIM F   
3: Compute slacks $\Delta _ { k } = b _ { k } - J _ { C _ { k } } ( \pi _ { \theta } ) _ { \mathrm { \Delta } }$ and set $R \gets \sqrt { 2 \delta }$   
4: Compute natural gradients $\tilde { g } _ { r } = \warrow \mathrm { ~ } \tilde { F } ^ { - 1 } g _ { r }$ and $\tilde { g } _ { \underline { { c } } _ { k } } = F ^ { - 1 } g _ { c _ { k } }$ (with damping if needed)   
5: Form $S _ { \mathrm { f u l l } } = [ \tilde { g } _ { r } , \tilde { g } _ { c _ { 1 } } , \dots , \tilde { g } _ { c _ { K } } ]$ and $O _ { \mathrm { f u l l } } = S _ { \mathrm { f u l l } } ^ { \top } F S _ { \mathrm { f u l l } }$   
6: Compute $\nu _ { k } , \mu _ { k } , \varrho _ { i j }$ via (19)–(20)   
7: Build enforced set K by excluding (i) non-binding constraints (Prop. 4.5) and (ii) near-zero   
constraints $\nu _ { k } \le \varepsilon _ { \mathrm { n z } }$   
8: Build near-collinear groups on $\kappa$ using edges $| \varrho _ { i j } | \geq 1 - \varepsilon _ { \mathrm { d u p } }$   
9: Set $i _ { \ell } \in$ arg min $k { \in } \mathcal { G } _ { \ell } ~ \mu _ { k }$ for each group $\mathcal { G } _ { \ell }$   
10: Form $S = [ \tilde { g } _ { r } , \tilde { g } _ { c _ { i _ { 1 } } } , \dots , \tilde { g } _ { c _ { i _ { m } } } ]$ and ${ \bf \bar { \Lambda } } O = S ^ { \top } F S ;$ regularize O if needed   
11: Compute Löwdin basis $E = S O ^ { - 1 / 2 }$   
12: Build reduced quantities $u = E ^ { \top } g _ { r }$ and $C _ { k } ^ { \top } = E ^ { \top } g _ { c _ { k } }$ for all $k \in \mathcal { K }$   
13: if all retained slacks satisfy $\Delta _ { k } \geq 0$ then   
14: Solve the reduced program (17) for $w ^ { \star }$   
15: else   
16: Compute Pareto descent direction $\bar { g } _ { p }$ via (27)–(28)   
17: Set ${ \boldsymbol { \iota } } \gets \bar { \boldsymbol { g } } _ { p }$ and solve the recovery program (30) for $w ^ { \star }$   
18: end if   
19: Set $\Delta \theta ^ { \star }  E w ^ { \star }$   
20: Update $\theta \gets \theta + \Delta \theta ^ { \star }$

Numerical tolerances. All linear solves in the reduced space use small ridge regularization. Feasibility and dual checks in the active-set routine are evaluated with fixed tolerances. If the reduced objective norm falls below a small threshold, the update defaults to the zero step.

## B Experimental Details

## B.1 Task details

## B.1.1 Circle Task from Bullet–safety–gym

We use the Circle tasks from Bullet-Safety-Gym [8] with Ball, Car, and Drone agents and the multi-constraint extension from prior work [29]. The reward of Circle task is defined as

$$
r _ { t } = 0 . 1 \frac { p _ { t } ^ { \top } ( J v _ { t } ) } { 1 + \left. \left. p _ { t } \right. _ { 2 } - r _ { \mathrm { c i r c } } \right. } + 0 . 0 1 r _ { t } ^ { \mathrm { s p e c } } , \qquad J = \left[ 0 \begin{array} { c c } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} \right] .\tag{50}
$$

This term captures the signed tangential velocity around the origin, yielding positive values for clockwise motion along the circular trajectory. The normalization factor attenuates the reward as the agent deviates from the target radius $r _ { \mathrm { c i r c } } ,$ so the reward is maximized near $\| p _ { t } \| _ { 2 } \approx r _ { \mathrm { c i r c } }$ under sustained clockwise rotation.

The agent-specific reward component $r _ { t } ^ { \mathrm { s p e c } }$ follows

$$
r _ { t } ^ { \mathrm { s p e c } } = \left\{ \begin{array} { l l } { - 0 . 5 \left. a _ { t } \right. _ { 2 } , } & { \mathrm { B a l l } , } \\ { 0 , } & { \mathrm { C a r } , } \\ { r _ { t } ^ { \mathrm { a l i v e } } - 0 . 0 1 \left. \omega _ { t } \right. _ { 2 } ^ { 2 } - c _ { t } ^ { \mathrm { g r o u n d } } , } & { \mathrm { D r o n e } . } \end{array} \right.\tag{51}
$$

Here $a _ { t }$ denotes the clipped action and $\omega _ { t }$ the angular velocity. The term $c _ { t } ^ { \mathrm { g r o u n d } }$ denotes the ground-contact penalty used by the environment. This shaping term provides mild agent-dependent regularization without altering the task objective. For the Ball agent, the $\ell _ { 2 }$ action penalty regularizes control effort. The Car agent uses no additional shaping beyond the task reward. For the Drone agent, $r _ { t } ^ { \mathrm { a l i v e } }$ provides a positive reward when the agent remains in a valid flying state and a negative reward otherwise, while the remaining terms discourage aggressive rotations and unsafe ground contacts.

![](images/cc7aa3d5d2ce555f8bbe16cc9598877e142ab0ce175548d6d391a32f263e49bb.jpg)  
Ball

![](images/443c7a4eaa53c0e7edcf14c57bda6c6ef726ec7434de22a8b306c4a615e3aaa9.jpg)  
Car

![](images/410c733bf71532465d17eed9b6191d1c45d6407c00eb7ed4234f84097d4c8cda.jpg)

![](images/3a19783002f8f3a6963e85421e831c97e9ce45e158ca5978f039403160aa2e40.jpg)  
Cassie

![](images/97c10ea4abd4fe7ad82837626bf71e979292d395207dcfe6fdce9c9abf599afc.jpg)  
Laikago  
Figure 5: Agents used in the multi-constraint evaluation. Ball, Car, and Drone are BulletSafetyGym agents, while Cassie and Laikago are bipedal and quadrupedal robots, respectively.

Table 1: BC-v3 stress-test configurations used in Figure 2.
<table><tr><td>Setting</td><td>Low-velocity coefficients  $\kappa _ { \mathrm { l o } }$ </td><td>Boundary  $\mathrm { t h r e s h o l d s } \ x _ { \mathrm { b n d } }$ </td><td>High-velocity coefficients  $\kappa _ { \mathrm { h i } }$ </td></tr><tr><td>Base (v3)</td><td> $\{ \mathbf { 0 . 6 } ^ { ( - a ) } \}$ </td><td> $\left\{ { \bf 6 . 0 } ^ { \left( - a \right) } \right\}$ </td><td> $\{ \mathbf { 1 . 0 } ^ { ( - a ) } \}$ </td></tr><tr><td>Redundant-2</td><td> $\{ \mathbf { 0 . 6 } ^ { ( - a ) } , 0 . 6 5 \}$ </td><td> $\{ { \bf 6 . 0 } ^ { ( - a ) } , 6 . 5 \}$ </td><td> $\{ \mathbf { 1 . 0 } ^ { ( - a ) } , 0 . 9 5 \}$ </td></tr><tr><td>Redundant-3</td><td> $\{ \mathbf { 0 . 6 } ^ { ( - a ) } , 0 . 6 5 , 0 . 7 \}$ </td><td> $\{ 6 . 0 ^ { ( - a ) } , 6 . 5 , 7 . 0 \}$ </td><td> $\{ { \bf 1 . 0 } ^ { ( - a ) } , 0 . 9 5 , 0 . 9 \}$ </td></tr></table>

Constraints. We use three per-step binary costs: a boundary cost, a high-velocity cost, and a low-velocity cost. Let $x _ { t }$ denote the agent position along the x-axis and $v _ { t }$ its planar speed. With boundary threshold $x _ { \mathrm { b n d } } > 0$ , base velocity limit $v _ { \mathrm { m a x } }$ , and scaling coefficients $\kappa _ { \mathrm { h i } } \in ( 0 , 1 ]$ and $\kappa _ { \mathrm { l o } } \in ( 0 , 1 )$ , we define

$$
c _ { \mathrm { b d r y } , t } = \mathbb { I } \{ | x _ { t } | > x _ { \mathrm { b n d } } \} ,\tag{52}
$$

$$
c _ { \mathrm { h i } , t } = \mathbb { I } \{ v _ { t } > \kappa _ { \mathrm { h i } } v _ { \mathrm { m a x } } \} ,\tag{53}
$$

$$
c _ { \mathrm { l o } , t } = \mathbb { I } \{ v _ { t } < \kappa _ { \mathrm { l o } } v _ { \mathrm { m a x } } \} .\tag{54}
$$

The two-constraint setting $\{ c _ { \mathrm { b d r y } } , c _ { \mathrm { h i } } \}$ is referred to as $\mathbf { v } 2 ,$ and the three-constraint setting $\{ c _ { \mathrm { b d r y } } , c _ { \mathrm { h i } } , c _ { \mathrm { l o } } \}$ as v3. Unless otherwise stated, the base v3 configuration uses $x _ { \mathrm { b n d } } = 6 . 0 , \kappa _ { \mathrm { h i } } = 1 . 0$ and $\kappa _ { \mathrm { l o } } = 0 . 6$

BC-v3 Variants. Figure 2 reports results on several BC-v3 variants obtained by tightening and duplicating constraints on top of the base v3 setting. All variants share the same task dynamics and reward, and only differ in the constraint coefficients summarized in Table 1. Boldfaced coefficients (marked with $( - a ) _ { } )$ correspond to the constraints denoted as $\ddot { \mathbf { \omega } } - \dot { \mathbf { a } } ^ { \dagger }$ in the results and learning-curve figures.

As redundancy increases, the low-velocity coefficient $\kappa _ { \mathrm { l o } }$ increases, raising the minimum admissible speed. This tightens the low-velocity constraint and prolongs the recovery phase early in training, since the agent must first escape low-speed states before achieving sustained motion along the circle. In parallel, the high-velocity coefficient $\kappa _ { \mathrm { h i } }$ decreases, tightening the upper speed limit. Because the task reward is proportional to tangential velocity, this directly reduces the achievable return by limiting the speed at which the agent can traverse the trajectory.

The tightest boundary coefficient remains fixed across all variants. Therefore, additional boundary constraints do not further restrict the feasible set, but only introduce near-collinear constraint directions, isolating the effect of constraint redundancy on optimization behavior.

## B.1.2 Locomotion tasks

In locomotion tasks the agent tracks velocity commands on flat terrain. At the beginning of each episode, a command vector

$$
( v _ { x } ^ { \mathrm { c m d } } , v _ { y } ^ { \mathrm { c m d } } , \omega _ { z } ^ { \mathrm { c m d } } )
$$

is sampled, where $v _ { y } ^ { \mathrm { c m d } }$ is fixed to 0 to simplify the task. In the implementation, $( v _ { x } ^ { \mathrm { c m d } } , \omega _ { z } ^ { \mathrm { c m d } } )$ are sampled uniformly from robot-specific ranges: Cassie uses $v _ { x } ^ { \mathrm { c m d } } \sim \mathcal { U } [ - 1 , 1 ]$ , while Laikago uses $v _ { x } ^ { \mathrm { c m d } } \sim \mathcal { U } [ - 1 , 2 ]$ , and both use $\omega _ { z } ^ { \mathrm { \bar { c } m d } } \sim \mathcal { U } [ - 0 . 5 , 0 . 5 ]$ . The reward penalizes squared tracking error between commanded and realized base velocities and adds a small power regularization term.

Constraint Setting At time step t, let $\theta _ { t }$ denote the angle between the robot base z-axis and the world z-axis, and let $a > 0$ be a fixed balance threshold. Let $h _ { t }$ denote the center-of-mass (CoM) height with a robot-specific minimum threshold $h _ { \operatorname* { m i n } } > 0$ . Let $\hat { \xi } _ { t } \in \{ - 1 , 1 \} ^ { n _ { \mathrm { l e g s } } }$ be the measured foot contact vector and $\xi _ { t } \in \{ - 1 , 1 \} ^ { n _ { \mathrm { l e g s } } }$ the desired periodic contact pattern derived from phase variables.

We define the following per-step costs:

$$
c _ { \mathrm { b a l } , t } = \mathbb { I } \{ \theta _ { t } > a \} ,\tag{55}
$$

$$
c _ { \mathrm { h g t } , t } = \mathbb { I } \{ h _ { t } < h _ { \operatorname* { m i n } } \} ,\tag{56}
$$

$$
{ c } _ { \mathrm { f o o t } , t } = \frac { 1 } { n _ { \mathrm { l e g s } } } \sum _ { i = 1 } ^ { n _ { \mathrm { l e g s } } } \mathbb { I } \{ \hat { \xi } _ { t , i } \neq \xi _ { t , i } \} .\tag{57}
$$

For the balance constraint, we use $a = 1 5 ^ { \circ }$ . For the height constraint, we set $h _ { \operatorname* { m i n } } = 0 . 3 5$ for Laikago and $h _ { \mathrm { m i n } } = 0 . 7$ for Cassie. The desired contact pattern $\xi _ { t }$ is defined using per-leg phase variables with a fixed frequency, following the benchmark specification.

Constraint limits are computed from target per-step cost rates by scaling with $1 / ( 1 - \gamma )$ , following the standard rule for per-step costs bounded in [0, 1]. The foot timing constraint is assigned a looser limit than the balance and height constraints, as it primarily serves as a motion regularizer rather than a strict safety constraint.

Episode length and termination handling. Episodes run for a fixed horizon of T steps and are not terminated early upon falling. When a fall is detected, the state and reward are held fixed until the horizon ends, preventing termination effects from confounding constraint evaluation.

## B.1.3 Bipedal Navigation task

Objective. To increase task diversity beyond velocity-tracking locomotion, we construct a Cassie goal-navigation task with obstacle avoidance, denoted as CassieGoalHazard-v0. The task extends the Cassie locomotion environment by replacing the commanded-velocity reward with a planar goalreaching objective and by adding hazard-avoidance constraints adapted from Safety-Gymnasium [10]. This setting is designed to couple legged locomotion stability with state-dependent navigation constraints while preserving a feasible but nontrivial multi-constraint control problem.

At the beginning of each episode, eight circular hazards and one goal are sampled in a square map with extents $[ - 3 . 5 , 3 . 5 ] \times [ - 3 . 5 , 3 . 5 ]$ . The goal radius is $r _ { \mathrm { g o a l } } = 0 . 5$ , and each hazard has radius $r _ { \mathrm { h a z } } = 0 . 3 5$ . The observation augments the Cassie proprioceptive state with 16-bin pseudo-LiDAR measurements for the goal and hazards, together with the goal distance.

Let $d _ { t }$ denote the planar distance between the robot base and the current goal at time step t. The reward is the distance reduction toward the goal plus a sparse goal bonus:

$$
r _ { t } = d _ { t - 1 } - d _ { t } + 0 . 2 \mathbb { I } \{ d _ { t } \leq r _ { \mathrm { g o a l } } \} .\tag{58}
$$

When the goal is reached, a new goal is sampled while the hazard layout remains fixed. Thus, the task rewards repeated navigation progress rather than a single terminal success event.

Constraint setting. The task retains the three Cassie locomotion-stability costs from Section B.1.2 and adds three navigation-specific costs. The full cost vector is ordered $\mathrm { a s } \big ( c _ { \mathrm { b a l } } , c _ { \mathrm { h g t } } , c _ { \mathrm { f o o t } } , c _ { \mathrm { s p d } } , c _ { \mathrm { h a z } } , c _ { \mathrm { e n g } } \big )$ . The first three costs are the balance, height, and foot-timing costs used in the locomotion task, with the Cassie height threshold set to $h _ { \operatorname* { m i n } } \stackrel { - } { = } 0 . 7$

For the additional costs, let $v _ { x , t }$ denote the forward base velocity, let $\mathcal { P } _ { t }$ be the set of floor-contact points of the robot at time t, and let H be the set of hazard centers. We define

$$
c _ { \mathrm { s p d } , t } = \mathbb { I } \{ | v _ { x , t } | > 1 . 0 \} ,\tag{59}
$$

$$
c _ { \mathrm { h a z } , t } = \mathbb { I } \{ \exists p \in \mathcal { P } _ { t } , \exists h \in \mathcal { H } : \| p - h \| _ { 2 } \leq r _ { \mathrm { h a z } } \} ,\tag{60}
$$

$$
c _ { \mathrm { e n g } , t } = \mathbb { I } \{ e _ { t } > 1 3 . 0 \} ,\tag{61}
$$

where $e _ { t }$ is the average actuator-power proxy computed from the absolute actuator force times joint velocity over simulation substeps.

Table 2: Actor/critic architectures and training hyperparameters (shared across methods unless noted).
<table><tr><td>Parameter LOCU</td><td>SDAC(A) PCRPO / CRPO TRPOL / GS</td><td>PPOL / GS</td></tr><tr><td>Actor MLP (hidden, act.) Policy distribution</td><td>(128, 128) ReLU / (512, 512) LeakyReLU Diag. Gaussian / squashed Gaussian</td><td>(128, 128) ReLU / – Diag. Gaussian / –</td></tr><tr><td>Action bounding Critic architecture Distributional atoms (M/M′)</td><td>clip / tanh-squash scalar V-critic / distributional Q-critic</td><td>clip / – scalar V-critic / – -1-</td></tr><tr><td></td><td>− / (25/50)</td><td></td></tr><tr><td>Target KL</td><td>0.001</td><td>0.02 / -</td></tr><tr><td>Critic optimization iters</td><td>20 / 40</td><td></td></tr><tr><td>Critic learning rate</td><td>1e-4 / 3e-4</td><td>1e-4 / –</td></tr><tr><td>L2 regularization</td><td>0.001</td><td></td></tr><tr><td>Damping coefficient</td><td>0.1 / 0.01</td><td></td></tr><tr><td>CG iters</td><td>10</td><td></td></tr><tr><td>Batch size (rollout)</td><td>10000</td><td></td></tr><tr><td>Minibatch size</td><td></td><td>256 /-</td></tr><tr><td>Discount factor γ</td><td></td><td></td></tr><tr><td></td><td>0.99</td><td></td></tr><tr><td>GAE / trace-decay λ</td><td>0.95 / 0.97</td><td></td></tr><tr><td>Advantage normalization</td><td>True</td><td></td></tr><tr><td>PID gains  $\left( k _ { p } , k _ { i } , k _ { d } \right)$ </td><td>(0.2, 0.002, 0.1)</td><td>(0.15, 0.005, 0.1)</td></tr></table>

Entries $a / b$ denote Bullet-Safety-Gym / Locomotion, respectively.  
“–” indicates not applicable or not evaluated; PPOL/GS was not run on locomotion tasks.

Constraint limits are converted by the same $1 / ( 1 - \gamma )$ scaling used for the locomotion tasks. These thresholds were selected through pilot calibration to keep simultaneous satisfaction feasible while avoiding overly loose constraints.

This task introduces qualitatively heterogeneous constraints. Goal-reaching encourages faster progress, while hazard avoidance can require detours, speed regulation can restrict aggressive motion, and energy control penalizes high-actuation strategies. The resulting environment therefore tests whether an algorithm can recover feasibility from infeasible initialization while still incorporating reward-improving directions once the active constraints are controlled. All experiments use the same fixed horizon of T = 500 steps.

## B.2 Implementation details

## B.2.1 Baseline Implementation

SDAC, PID Lagrangian, and Gradient Shaping (GS). For SDAC, we use the official implementation released by the original authors. For PID Lagrangian and Gradient Shaping (GS), we adopt the implementation built on top of the FSRL framework [16], as released by the authors of GS. Across these baselines, we align common experimental settings and remaining algorithm-specific components, including multiplier dynamics, and shaping rules, are kept identical to the respective authors’ implementations without modification.

CRPO. We build CRPO on top of the CRPO implementation provided by OmniSafe [11] and extend it to the multi-constraint setting. When multiple constraints are violated simultaneously, we sample a single constraint to update the policy at each iteration. The sampling probability is proportional to the magnitude of constraint violation, so that more severely violated constraints are prioritized during training.

PCRPO. To the best of our knowledge, no official public implementation of PCRPO is available, although the original paper reports that PCRPO is implemented and evaluated using the OmniSafe framework. We therefore implement PCRPO directly following the algorithmic description and the configuration settings reported in the paper. In particular, the appendix provides a configuration table that lists shared hyperparameter settings for PCRPO, including the use of CG iterations. Based on this specification, we implement PCRPO as a second-order primal method that computes natural-gradient style updates using a CG with its soft-switching method.

PCRPO uses two slack thresholds per constraint, denoted as an upper slack and a lower slack, to determine which objective is optimized at each update. We set these thresholds by scaling the corresponding constraint limit $b _ { k }$ . In Bullet-Safety-Gym, we use an upper slack of $1 . 2 5 b _ { k }$ and a lower slack of $0 . 7 5 b _ { k }$ . In locomotion tasks, we use an upper slack of $1 . 0 b _ { k }$ and a lower slack of $0 . 7 5 b _ { k }$ , applied per constraint.

At each iteration, constraints are categorized into three groups based on their current costs relative to the slack thresholds: (i) constraints with cost above the upper slack, (ii) constraints with cost between the upper and lower slacks, and (iii) constraints with cost below the lower slack. If any constraint falls into group (i), one violated constraint is sampled from this group and a constraint-reduction update is performed. Otherwise, if group (ii) is non-empty, one constraint is sampled and a reward update is applied together with the gradient-manipulation rule specified by PCRPO to mitigate interference between reward and constraint gradients. If all constraints belong to group (iii), a feasible update is performed.

Clarabel. We use Clarabel to solve the local constrained policy-update problem in two forms: full QP and reduced QP. The full QP optimizes the coefficients of the natural reward and cost gradients, while the reduced QP optimizes the update in LOCU’s Löwdin coordinates, including the orthogonal reward direction. For constraint recovery, the full QP minimizes total constraint slack when the reward-maximization problem is infeasible. The reduced QP minimizes slack for currently violated constraints while enforcing the other constraints.

## B.2.2 Mean–std constraint and distributional cost critics

SDAC adopts the mean–std safety measure $\begin{array} { r } { F ( Z ; \alpha _ { \mathrm { l o } } ) = \mathbb { E } [ Z ] + \frac { \phi ( \Phi ^ { - 1 } ( \alpha _ { \mathrm { l o } } ) ) } { \alpha _ { 1 } } \mathrm { S t d } [ Z ] } \end{array}$ motivated as a practical surrogate via the Gaussian–CVaR connection. Following the discussion in SDAC, we treat mean–std as an uncertainty-weighted surrogate rather than an exact CVaR constraint [14].

For locomotion tasks, we use a distributional cost critic and denote the predicted quantile samples for constraint k at $( s , a )$ by $\{ z _ { k } ^ { ( m ) } ( s , a ) \} _ { m = 1 } ^ { M }$ . In our locomotion experiments, we adopt the same task definitions and training protocol as SRCPO [15]. We keep the distributional critic training unchanged, including the quantile regression objective and the construction of bootstrapped target quantiles, and modify only the aggregation used to compute the mean–std surrogate from predicted quantiles.

The original SDAC aggregation estimates the mean–std surrogate by mixing quantile samples across the batch and then taking a single global standard deviation:

$$
\begin{array} { r l } & { \widehat { F } _ { k } ^ { \mathrm { m i x e d } } = \displaystyle \mathrm { M e a n } \Big ( \{ z _ { k } ^ { ( m ) } ( s , a ) \} _ { ( s , a ) \in \mathcal { B } , m } \Big ) + \sigma ( \alpha _ { \mathrm { l o } } ) \mathrm { S t d } \Big ( \{ z _ { k } ^ { ( m ) } ( s , a ) \} _ { ( s , a ) \in \mathcal { B } , m } \Big ) , } \\ & { \sigma ( \alpha _ { \mathrm { l o } } ) = \displaystyle \frac { \phi ( \Phi ^ { - 1 } ( \alpha _ { \mathrm { l o } } ) ) } { \alpha _ { \mathrm { l o } } } . } \end{array}\tag{62}
$$

In our locomotion experiments, this batch mixed aggregation can yield unstable updates and, in some runs, prevent convergence. We therefore compute the statistics per state–action pair first and then average across the batch:

$$
\begin{array} { r } { \widehat { F } _ { k } ^ { \mathrm { c o n d } } = \operatorname { M e a n } _ { ( s , a ) \in \mathcal { B } } \Big ( \operatorname { M e a n } _ { m } [ z _ { k } ^ { ( m ) } ( s , a ) ] + \sigma ( \alpha _ { \mathrm { l o } } ) \operatorname { S t d } _ { m } [ z _ { k } ^ { ( m ) } ( s , a ) ] \Big ) . } \end{array}\tag{63}
$$

This modification reduces overly conservative behavior caused by batch heterogeneity by ensuring that the standard deviation term reflects within-sample distributional uncertainty rather than variation across states within the batch.

## C Supplementary Analysis and Results

## C.1 Multi-constraint update behaviors of existing Primal-dual algorithms

We summarize multi-constraint update behaviors in representative safe policy optimization algorithms by grouping methods by how they incorporate multiple constraints into a single actor step: (i) sequential constraint projection onto linearized halfspaces and (ii) scalarized constraint handling via a weighted aggregation of constraint signals. We additionally discuss practical hyperparameter tuning difficulties that arise in multi-constraint settings.

![](images/966fb96b63fbe6639fa6c9f54aca2879958a8c5ae9a8dd0214bd6b36821f1436.jpg)  
Figure 6: Geometric comparison between LOCU and PCPO updates. LOCU solves a reduced QP and produces an order-invariant update under the local model. PCPO applies sequential projections onto constraint halfspaces, which can produce order-dependent update directions.

## C.1.1 Sequential constraint projection

A common projection-based approach is to first compute a reward-improving step and then enforce constraint satisfaction by projecting it onto a linearized constraint halfspace under a chosen metric. A representative instance is PCPO [28], which in the single-constraint case solves the projection

$$
\Delta \theta ^ { + } = \arg \operatorname* { m i n } _ { \Delta \theta \in \mathbb { R } ^ { n } } \frac { 1 } { 2 } ( \Delta \theta - \Delta \theta ^ { ( 0 ) } ) ^ { \top } L \left( \Delta \theta - \Delta \theta ^ { ( 0 ) } \right) \quad \mathrm { s . t . } \quad g _ { c } ^ { \top } \Delta \theta \leq \Delta ,\tag{64}
$$

where L is the projection metric. PCPO considers $L = I$ for $\ell _ { 2 }$ projection and $L = F$ for Fishermetric projection. The solution is the closed-form correction along $\bar { L ^ { - 1 } } g _ { c } \colon$

$$
\Delta \theta ^ { + } = \Delta \theta ^ { ( 0 ) } - \alpha L ^ { - 1 } g _ { c } , \qquad \alpha = \operatorname * { m a x } \left\{ 0 , \ : \frac { g _ { c } ^ { \top } \Delta \theta ^ { ( 0 ) } - \Delta } { g _ { c } ^ { \top } L ^ { - 1 } g _ { c } } \right\} .\tag{65}
$$

For $L = F .$ , define $u _ { k } = \tilde { g } _ { c _ { k } } / \| \tilde { g } _ { c _ { k } } \| _ { F }$ and $\mu _ { k } = \Delta _ { k } / \left. \tilde { g } _ { c _ { k } } \right. _ { F }$ . The Fisher-metric projection onto the halfspace $\langle u _ { k } , \Delta \theta \rangle _ { F } \leq \mu _ { k }$ is

$$
P _ { k } ( x ) : = \mathop { \operatorname { a r g } } _ { y : \langle u _ { k } , y \rangle _ { F } \leq \mu _ { k } } \frac { 1 } { 2 } \left\| y - x \right\| _ { F } ^ { 2 } = x - \left[ \langle u _ { k } , x \rangle _ { F } - \mu _ { k } \right] _ { + } u _ { k } ,\tag{66}
$$

where $[ t ] _ { + } : = \operatorname* { m a x } \{ t , 0 \}$

As discussed in PCPO [28], in the multi-constraint setting PCPO can apply these halfspace projections sequentially across constraints. However, this sequential projection procedure can exhibit order dependence and step shrink when constraint normals are correlated. Figure 6 visualizes the local model and contrasts LOCU with sequential-projection updates. The left panel corresponds to LOCU’s single reduced-space solve. The right panel depicts sequential halfspace projections, where order dependence is shown in (68)–(69) and step shrink is quantified by (70).

Consider Fisher halfspaces $\mathcal { H } _ { k } : = \{ \Delta \theta : \langle u _ { k } , \Delta \theta \rangle _ { F } \leq \mu _ { k } \}$ . Starting from the reward-only step $\Delta \theta ^ { ( 0 ) }$ , a sequential projection update applies

$$
\Delta \theta ^ { ( t ) } = P _ { \sigma _ { t } } \bigl ( \Delta \theta ^ { ( t - 1 ) } \bigr ) , \qquad t = 1 , 2 , \ldots , m ,\tag{67}
$$

for some ordering σ.

For two constraints i and $j ,$ define

$$
\begin{array} { r } { s _ { i } : = \langle u _ { i } , x \rangle _ { F } , \quad s _ { j } : = \langle u _ { j } , x \rangle _ { F } , \quad t _ { i } : = [ s _ { i } - \mu _ { i } ] _ { + } , \quad t _ { j } : = [ s _ { j } - \mu _ { j } ] _ { + } , \quad \rho _ { i j } : = \langle u _ { i } , u _ { j } \rangle _ { F } . } \end{array}
$$

Then

$$
P _ { j } ( P _ { i } ( x ) ) = x - t _ { i } u _ { i } - \Big [ s _ { j } - \rho _ { i j } t _ { i } - \mu _ { j } \Big ] _ { + } u _ { j } ,\tag{68}
$$

$$
P _ { i } ( P _ { j } ( x ) ) = x - t _ { j } u _ { j } - \Big [ s _ { i } - \rho _ { i j } t _ { j } - \mu _ { i } \Big ] _ { + } u _ { i } .\tag{69}
$$

If both constraints are active and $\rho _ { i j } \neq 0$ , the correction magnitudes differ and the final update depends on the order.

Sequential projections also reduce the Fisher step length. Let $t _ { k } : = [ \langle u _ { k } , x \rangle _ { F } - \mu _ { k } ] _ { + }$ . Using $\| u _ { k } \| _ { F } = 1$ ，

$$
\left. P _ { k } ( x ) \right. _ { F } ^ { 2 } = \left. x \right. _ { F } ^ { 2 } - t _ { k } ( t _ { k } + 2 \mu _ { k } ) .\tag{70}
$$

When $\mu _ { k } \geq 0$ , any active projection $t _ { k } > 0$ strictly decreases $\big \| \cdot \big \| _ { F } , \operatorname { s o } \big \| \Delta \theta ^ { ( t ) } \big \| _ { F }$ is non-increasing along (67).

The reward improvement in the local model is attenuated in proportion to Fisher correlations. Let $\left. u _ { r } = \tilde { g } _ { r } / \left. \tilde { g } _ { r } \right. _ { F } \right]$ . Then

$$
\langle u _ { r } , P _ { k } ( x ) \rangle _ { F } = \langle u _ { r } , x \rangle _ { F } - t _ { k } \langle u _ { r } , u _ { k } \rangle _ { F } .\tag{71}
$$

If $\langle u _ { r } , u _ { k } \rangle _ { F }$ is not small across multiple active constraints, sequential projections repeatedly subtract reward-aligned components and can produce overly conservative updates.

## C.1.2 Scalarized constraint handling

Another prevalent design is to reduce a vector-valued constraint problem to a single scalar surrogate by aggregating multiple constraint signals through a weighted sum (typically via dual variables). This avoids explicit geometric reasoning about how constraint normals interact; instead, interactions appear only through the aggregate scalar and the resulting single update direction.

Constrained Update Projection (CUP) CUP [27] explicitly separates an intermediate performanceimprovement iterate $\pi _ { \theta _ { k + \frac { 1 } { 2 } } }$ from a projection-to-safety step. In the projection stage, CUP solves a constrained KL projection in policy space:

$$
\pi _ { \boldsymbol { \theta } _ { k + 1 } } = \arg \operatorname* { m i n } _ { \pi _ { \boldsymbol { \theta } } \in \Pi _ { \boldsymbol { \theta } } } \bar { D } _ { \mathrm { K L } } \Big ( \pi _ { \boldsymbol { \theta } _ { k + \frac { 1 } { 2 } } } , \pi _ { \boldsymbol { \theta } } \Big ) \quad \mathrm { s . t . } \quad \hat { C } _ { \pi _ { \boldsymbol { \theta } _ { k } } } ( \pi _ { \boldsymbol { \theta } } , \beta _ { k } ) \leq \boldsymbol { b } ,\tag{72}
$$

where $\hat { C }$ denotes a surrogate cost and $\beta _ { k }$ is a penalty coefficient ensuring the theoretical safety bound. The optimization is implemented via a primal–dual routine with Lagrange multipliers associated with constraints. In the multi-constraint case, multipliers are updated per-constraint based on their individual violations, and the resulting policy update is driven by the aggregated penalty direction induced by the current multiplier vector.

First Order Constraint Optimization on Policy Space (FOCOPS) FOCOPS first solves a constrained optimization in policy space with a KL trust region, obtaining a non-parametric optimal policy (shown in [30], Thm. 1):

$$
\pi ^ { * } ( a \mid s ) \propto \pi _ { \theta _ { k } } ( a \mid s ) \exp \left( \frac { 1 } { \lambda } \Big ( A _ { R } ^ { \pi _ { \theta _ { k } } } ( s , a ) - \sum _ { k } \nu _ { k } A _ { C _ { k } } ^ { \pi _ { \theta _ { k } } } ( s , a ) \Big ) \right) ,\tag{73}
$$

where $\lambda > 0$ and $\{ \nu _ { k } \ge 0 \}$ are dual variables. FOCOPS then projects back to the parametric class by minimizing $\mathrm { K L }$ :

$$
\theta _ { k + 1 } = \arg \operatorname* { m i n } _ { \theta } ~ \mathbb { E } _ { s \sim d ^ { \pi _ { \theta _ { k } } } } \left[ \mathrm { K L } ( \pi ^ { * } ( \cdot \mid s ) \| \pi _ { \theta } ( \cdot \mid s ) ) \right] .\tag{74}
$$

Overlap and cancellation under scalarization. Both CUP and FOCOPS rely on a scalarized aggregate of multiple constraints, e.g., the aggregated advantage in (73)

$$
\hat { A } ( s , a ) : = A _ { R } ( s , a ) - \sum _ { k = 1 } ^ { K } \nu _ { k } A _ { C _ { k } } ( s , a ) ,
$$

or, in parameter space, an aggregated penalty direction of the form

$$
\tilde { g } _ { \mathrm { e f f } } = \tilde { g } _ { r } - \sum _ { k = 1 } ^ { K } \lambda _ { k } \tilde { g } _ { c _ { k } } .
$$

Table 3: PID coefficient candidate values used in our experiments.
<table><tr><td>Coefficient</td><td>Candidate values</td></tr><tr><td> $K _ { P }$ </td><td>{0.25, 1}</td></tr><tr><td>KI</td><td> $\mathring { \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 \} }$ </td></tr><tr><td>KD</td><td>{0.01, 0.1, 1}</td></tr></table>

In multi-constraint regimes, the behavior of the aggregate is governed by pairwise correlations among constraint directions. In Fisher geometry, the penalty direction satisfies

$$
\left\| \sum _ { k = 1 } ^ { K } \lambda _ { k } \tilde { g } _ { c _ { k } } \right\| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { K } \lambda _ { i } ^ { 2 } \left\| \tilde { g } _ { c _ { i } } \right\| _ { F } ^ { 2 } + 2 \sum _ { 1 \leq i < j \leq K } \lambda _ { i } \lambda _ { j } \left. \tilde { g } _ { c _ { i } } , \tilde { g } _ { c _ { j } } \right. _ { F } .\tag{75}
$$

The cross terms encode interference: if $\langle \tilde { g } _ { c _ { i } } , \tilde { g } _ { c _ { j } } \rangle _ { F } < 0$ (opposing directions), the aggregate can exhibit partial cancellation, weakening constraint correction even when individual constraints are violated; if $\langle \tilde { g } _ { c _ { i } } , \tilde { g } _ { c _ { i } } \rangle _ { F } > 0$ (overlapping directions), the aggregate can over-emphasize a redundant direction, effectively reducing the rank of meaningful constraint information and making the update sensitive to the instantaneous multiplier vector. Since scalarization does not explicitly model the constraint geometry, trade-offs among conflicting constraints are resolved implicitly through multiplier weights rather than through an explicit reduced-space analysis of active constraints.

## C.1.3 Hyperparameter tuning difficulties

Primal–dual Lagrangian methods handle constraints by introducing multipliers and optimizing

$$
\operatorname* { m a x } _ { \theta } \operatorname* { m i n } _ { \lambda \geq 0 } J _ { R } ( \pi _ { \theta } ) - \sum _ { k = 1 } ^ { K } \lambda _ { k } \big ( J _ { C _ { k } } ( \pi _ { \theta } ) - b _ { k } \big ) ,\tag{76}
$$

so the actor update direction is $\begin{array} { r } { g _ { r } - \sum _ { k } \lambda _ { k } g _ { c _ { k } } } \end{array}$ . In practice, the training behavior can be dominated by the dual feedback loop and becomes sensitive to the initialization of λ and the relative step sizes of primal and dual updates, which often leads to substantial hyperparameter tuning and unstable learning [19, 13].

PID Lagrangian replaces simple dual ascent with a PID-style controller over the constraint error $e _ { k } : = J _ { C _ { k } } ( \pi _ { \theta } ) - b _ { k }$ [23],

$$
\lambda _ { k } \gets \Big [ \lambda _ { k } + K _ { P } e _ { k } + K _ { I } \sum _ { \tau \leq t } e _ { k , \tau } + K _ { D } ( e _ { k } - e _ { k - 1 } ) \Big ] _ { + } ,\tag{77}
$$

However, selecting $K _ { P } , K _ { I }$ , and $K _ { D }$ remains a practical bottleneck.

We evaluated PID PPO-Lagrangian with gradient shaping [29] under different PID coefficient choices. We used similar PID coefficient sets used for the coefficient learning-curve comparisons in Appendix C of the PID-Lagrangian paper [23], summarized in Table 3. All combinations were evaluated with five random seeds. In Figure 7, each curve was obtained by averaging all runs that shared the coefficient value of interest over the remaining two PID coefficients and seeds, and the shaded region denotes the standard deviation over the same set of runs.

Gradient shaping mitigates cancellation and overlap among constraint gradients in the actor update [29], yet sensitivity of the multiplier dynamics from (77) still remains a challenge. Across the tested settings, only a narrow regime such as $K _ { I } = 1 0 ^ { - 3 }$ yields relatively stable boundary tracking in this task, suggesting that the robustness reported in single-constraint PID Lagrangian analyses [23] can be weakened when multiple constraints are coupled through a policy update.

## C.2 Perturbation sensitivity of Löwdin and Gram-Schmidt coefficients

We compare the perturbation sensitivity of Löwdin and Gram–Schmidt orthogonalization through theoretical analysis and empirical evaluation, focusing on how stochastic noise propagates through their respective orthogonalization procedures.

![](images/06a36c69dd6f5ced96f6904b7ace241b96aac1b4d795ba4651c903345bd8757d.jpg)  
Figure 7: Learning-curve comparison of PID PPO-Lagrangian under P I D coefficient variations on the Bullet Safety Gym BC-v3 task. Dashed lines in the cost plots indicate the corresponding limits.

## C.2.1 Theoretical analysis

Using the notation in Sections 3.2, 3.3, and 4.1, let ${ \cal { S } } = [ \tilde { g } _ { r } , \tilde { g } _ { c _ { i _ { 1 } } } , \dots , \tilde { g } _ { c _ { i _ { m } } } ]$ and ${ \cal O } = { \cal S } ^ { \top } { \cal F } { \cal S }$ Suppose O can have small eigenvalues due to near-collinear directions. Under stochastic gradient noise Ξ, where ${ \widehat { S } } : = S + \Xi$ , the perturbed Fisher Gram matrix is $\widehat { O } : = \widehat { S } ^ { \top } F \widehat { S }$ . Then

$$
\Delta O : = \widehat { O } - O = S ^ { \top } F \Xi + \Xi ^ { \top } F S + \Xi ^ { \top } F \Xi .\tag{78}
$$

Its norm is bounded by

$$
\begin{array} { r } { \| \Delta O \| _ { 2 } \leq 2 \| F ^ { 1 / 2 } S \| _ { 2 } \| F ^ { 1 / 2 } \Xi \| _ { 2 } + \| F ^ { 1 / 2 } \Xi \| _ { 2 } ^ { 2 } . } \end{array}\tag{79}
$$

For LOCU, since Proposition 4.2 gives $E ^ { \top } g _ { r } = O ^ { 1 / 2 } e _ { 1 }$ and $E ^ { \top } g _ { c _ { i _ { j } } } = O ^ { 1 / 2 } e _ { j + 1 }$ , the sensitivity depends on the noise of the square root.

For Löwdin Orthogonalization. Let $O = Q \Lambda Q ^ { \top }$ , with Q the eigenvector matrix so that $Q ^ { \top } ( \cdot ) Q$ represents a change to the eigenbasis, and $\boldsymbol { \Lambda } = \operatorname { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { m + 1 } )$ . Then, defining $\Delta _ { L } : = \widehat { O } ^ { 1 / 2 } -$ $O ^ { 1 / 2 }$ , the first-order term satisfies the Sylvester equation:

$$
O ^ { 1 / 2 } \Delta _ { L } + \Delta _ { L } O ^ { 1 / 2 } = \Delta O + \mathcal { O } ( \| \Delta O \| _ { 2 } ^ { 2 } ) .\tag{80}
$$

So in the eigenbasis of O,

$$
Q ^ { \top } \Delta _ { L } Q = \left[ \frac { ( Q ^ { \top } \Delta O Q ) _ { a b } } { \sqrt { \lambda _ { a } } + \sqrt { \lambda _ { b } } } \right] _ { a , b } + { \mathcal O } ( \| \Delta O \| _ { 2 } ^ { 2 } ) , \qquad a , b \in \{ 1 , \ldots , m + 1 \} .\tag{81}
$$

where $a , b \in \{ 1 , \dots , m + 1 \}$ index the coordinates in that eigenbasis.

This shows the symmetric structure of Löwdin. The sensitivity to near-collinearity depends on the denominator in (81), so the effect of the noise is distributed across the spectrum of the whole retained block.

For Gram–Schmidt Orthogonalization. Let $O = T ^ { \top } T$ and $T = \mathrm { c h o l } ( O )$ , the Cholesky factorization of O; this is equivalent to Fisher Gram–Schmidt applied to the columns of S, where T collects the resulting projection and normalization coefficients, so the corresponding reduced coefficients are encoded by the upper-triangular factor T. If $o _ { a b }$ and $t _ { a b }$ denote the entries of O and T, then the Cholesky/Gram–Schmidt recursion is

$$
t _ { k k } ^ { 2 } = o _ { k k } - \sum _ { s < k } t _ { s k } ^ { 2 } , \qquad t _ { k j } = \frac { o _ { k j } - \sum _ { s < k } t _ { s k } t _ { s j } } { t _ { k k } } , \qquad j > k .\tag{82}
$$

Using the first-order Taylor expansion of these relations gives

$$
\Delta t _ { k k } = \frac { \Delta o _ { k k } - 2 \sum _ { s < k } t _ { s k } \Delta t _ { s k } } { 2 t _ { k k } } + \mathcal { O } ( \| \Delta O \| _ { 2 } ^ { 2 } ) ,\tag{83}
$$

and

$$
\Delta t _ { k j } = \frac { \Delta o _ { k j } - \sum _ { s < k } \left( \Delta t _ { s k } t _ { s j } + t _ { s k } \Delta t _ { s j } \right) - t _ { k j } \Delta t _ { k k } } { t _ { k k } } + \mathcal { O } ( \| \Delta O \| _ { 2 } ^ { 2 } ) , \qquad j > k .\tag{84}
$$

This shows the sequential noise structure of Gram–Schmidt: the noise at stage k is divided by the current pivot $t _ { k k }$ , and later quantities inherit earlier noise through the recursion.

Comparison in update. Fundamentally, both orthogonalization methods suffer under severe collinearity. LOCU addresses this via pairwise screening and compression, $| \varrho _ { i j } | \geq 1 - \varepsilon _ { \mathrm { d u p } } .$ However, this cannot fully prevent span-level near-collinearity, where a retained vector can align with the span of others much more tightly than the threshold $\varepsilon _ { \mathrm { d u p } }$

In worst-case scenarios, the Gram matrix O becomes near-singular, destabilizing both methods. For Löwdin, this forces the minimum eigenvalue $\lambda _ { \operatorname* { m i n } } ( O ) \approx 0$ , driving the denominator in (81) to approach zero. For Gram–Schmidt, the instability manifests in the sequential pivots. Rewriting (82) in Schur-complement form yields

$$
\begin{array} { r } { t _ { k k } ^ { 2 } = o _ { k k } - O _ { 1 : k - 1 , k } ^ { \top } O _ { 1 : k - 1 , 1 : k - 1 } ^ { - 1 } O _ { 1 : k - 1 , k } = o _ { k k } \big ( 1 - \rho _ { k | 1 : k - 1 } ^ { 2 } \big ) . } \end{array}\tag{85}
$$

Thus, span-level near-collinearity, $\rho _ { k | 1 : k - 1 } ^ { 2 } \approx 1$ , drives the normalized residual $t _ { k k } ^ { 2 } / o _ { k k }$ to near zero.

In such cases, both methods require regularization/damping to avoid division by near zero. However, their noise and bias propagation differ fundamentally as shown earlier. In Gram–Schmidt, the sequential structure causes both stochastic noise and the damping bias introduced at step k to be inherited and compounded by all subsequent steps $j > k$ , leading to cumulative distortion in the resulting basis geometry. Conversely, Löwdin applies symmetric matrix-level regularization, e.g., $O + \varepsilon I ,$ , which distributes both perturbation and damping effects across the entire eigenbasis without order-dependent accumulation. By treating all vectors simultaneously, Löwdin preserves the mutual symmetry and geometric integrity of the basis far more effectively than the inherently hierarchical Gram–Schmidt approach.

## C.2.2 Empirical comparison

We evaluated LOCU on the BC-v3 Redun : 3 task by replacing Löwdin orthogonalization with Gram–Schmidt orthogonalization under three gradient orderings (original, random, and reverse), while injecting observation noise into each training batch following [16].

As shown in Figure 8, Löwdin orthogonalization generally performs below Gram–Schmidt index but above random and reverse, attributed to its symmetric distribution of perturbation-induced errors across gradient directions. In this task, the last three constraints in the indexed order are the loosest low-velocity constraints, and the differences among the Gram–Schmidt variants reflect their inherent ordering dependence. Under the indexed order, accumulated errors primarily propagate to these loose constraints, whereas under the reverse order, they affect the reward gradient and the tighter high-velocity constraints. The noise implementation of [16] adds perturbations of a fixed magnitude without accounting the relative scale of each observation. At the highest noise level of 0.25, all methods degrade substantially, while the pronounced failure of Gram–Schmidt-random suggests that Gram–Schmidt orthogonalization can be more vulnerable than Löwdin when large stochastic perturbations distort the gradient geometry.

![](images/b01be4fc779708ce54361e3bde57b219e6fdc53980df05cfa381fddca043809c.jpg)

![](images/48e1d956455e32356cf864a51cbd58cba7d1df844565a2410eaae1031ceb58d8.jpg)

![](images/baa06474991a8ca3bfc91e5e18363edbac86e63ddcac818dcb7074c388e26c05.jpg)  
Figure 8: Performance comparison of Löwdin and Gram–Schmidt orthogonalization under observation noise on BC-v3 Redun 3.

## C.3 Computational Complexity

## C.3.1 Wall clock time results

Table 4 reports wall clock training time on BC-v2 and redun 3 variants of BC-v3, measured on a single machine with an NVIDIA RTX 4070 GPU, an Intel i7-13700K CPU, and 64GB RAM. PPOL-V and PPOL-GS are excluded from the redundancy results in the main text due to unstable learning in earlier stages, but we additionally trained them solely to compare wall clock time and include their times in the table. Wall clock time is logged with Weights and Biases [5].

Table 4: Wall-clock time on the main circle task.
<table><tr><td>Task</td><td>LOCU</td><td>SDAC(A)</td><td>PCRPO</td><td>CRPO</td><td>TRPOL -GS</td><td>TRPOL -V</td><td>PPOL -V</td><td>PPOL -GS</td></tr><tr><td>BC v2</td><td>17:08</td><td>18:20</td><td>17:30</td><td>17:23</td><td>16:08</td><td>16:27</td><td>17:11</td><td>18:29</td></tr><tr><td>BC v3 Redun 3</td><td>25:02</td><td>25:40</td><td>23:00</td><td>22:53</td><td>17:02</td><td>17:05</td><td>22:28</td><td>23:40</td></tr></table>

Table 5: Wall-clock time on Circle tasks with parallel processing.
<table><tr><td>Task</td><td>Redun</td><td>LOCU</td><td>SDAC(A)</td><td>CRPO</td><td>TRPOL</td><td>PPOL (bs=64)</td><td>PPOL (bs=128)</td></tr><tr><td rowspan="3">PointCircle 3</td><td>2</td><td>10:19</td><td>10:55</td><td>10:18</td><td>10:22</td><td>21:08</td><td>15:37</td></tr><tr><td></td><td>10:31</td><td>11:08</td><td>10:19</td><td>10:25</td><td>21:13</td><td>15:40</td></tr><tr><td>5</td><td>10:44</td><td>11:19</td><td>10:22</td><td>10:29</td><td>21:20</td><td>15:45</td></tr></table>

With implementation for main text experiment, LOCU computes natural-gradient directions for all constraints through CG then solves a QP, so its cost can grow with the number of constraints. However, this overhead is not always dominant. In BC-v2, where only a small number of constraints are used, PPO-style baselines often spend more time on repeated mini-batch actor optimization and multiple constraint-critic updates, which can outweigh CG solves.

In BC-v3 Redun 3, where the number of constraints increases to nine, wall-clock time increases for most methods, with the largest increases observed for LOCU and SDAC(A). This reflects the cumulative cost of computing multiple constraint natural-gradient directions before the constrained update. The comparison should also be interpreted with implementation details: SDAC(A) uses OSQP and SciPy backends for its QP [22, 25], whereas LOCU implementation are in Python-level.

## C.3.2 Parallel processing

To examine whether this bottleneck can be reduced, we construct a MuJoCo-JAX (MJX) [24] version of the CircleRun-v3 Redun task and implement LOCU, SDAC(A), CRPO, PPOLag, and TRPOLag with vectorized constraint processing and end-to-end XLA compilation. This targets a common limitation of Python-based safe RL implementations, where most computations are executed sequentially and therefore use only part of the available hardware throughput. Table 5 reports the wall-clock time of the MJX implementation on PointCircle with redundancy levels 2, 3, and 5. The results show that full-batch second-order actor updates can be faster than minibatch-based first-order updates with parallel processing. LOCU, SDAC(A), CRPO, and TRPOL complete each run in roughly 10–11 minutes, whereas PPOLag takes 15–21 minutes depending on the minibatch size, denoted as bs in the table.

As expected, LOCU and SDAC(A), which compute natural-gradient directions for all constraints, show a larger redundancy-dependent increase than CRPO and TRPOL, which only compute single natural-gradient. Under the MJX implementation, however, this increase is small: from redundancy level 2 to 5, LOCU increases from 10:19 to 10:44 and SDAC(A) from 10:55 to 11:19, i.e., less than 30 seconds while the number of constraints grows from 6 to 15. By contrast, in the Python-based implementation in Table 4, increasing from BC-v2 with two constraints to BC-v3 Redun 3 with nine constraints increases LOCU and SDAC(A) by roughly 7–8 minutes. Compared with the substantially larger cost of PPOLag, the remaining difference between LOCU and the single-natural-gradient methods can be considered minor.

Overall, the results indicate that parallelized constraint processing substantially reduces wall-clock sensitivity to the number of constraints compared with sequential Python-based implementations. The same implementation strategy may therefore be useful for other multi-constraint safe RL methods or multi-objective settings. More broadly, the results weaken the view that second-order safe RL methods are impractical solely due to computational burden. When multiple constraints must be handled and constraint satisfaction is central to the task, LOCU or other trust-region-based second-order methods can remain practical options under parallel processing.

## C.3.3 Inverse fisher preconditioner

Natural gradients are typically computed using conjugate gradient (CG), which needs to be applied separately to each gradient, the computational cost increases linearly with the number of constraints. An alternative option can be using a shared approximate Fisher inverse or preconditioner across all gradients. Since the FIM is defined solely by the outer product of policy score functions, independent of advantage estimates, layer-wise curvature approximations such as K-FAC and its variants can provide a single approximate inverse applicable to all reward and constraint gradients. Although these approaches can be more computationally efficient than using CG iterations depending on the network architecture and the number of constraints, further investigation is required to assess its approximation accuracy, numerical stability, and overall effectiveness in multi-constraint policy optimization.

## C.4 Additional Experiments

We conduct additional experiments to evaluate LOCU from three perspectives. First, we examine the contribution of its algorithmic components through ablation studies. Second, we investigate hyperparameter sensitivity by varying the clustering threshold and KL limit. Finally, we evaluate robustness to redundant constraints by increasing the number of constraints while preserving the feasible region.

## C.4.1 Component Analysis

We compare LOCU against two ablated variants and a CPO baseline. The two ablated variants in Figure 9 are labeled as $\pi / \circ$ Lowdin and $\mathtt { w } / \mathtt { o }$ Lowdin, Screening & compression. $\pi / \circ$ Lowdin removes Löwdin orthogonalization and only normalizes each direction by its Fisher norm before applying the same reduced update. $\mathtt { w } / \mathtt { o }$ Lowdin, Screening & compression further removes screening and clustering and uses all normalized constraint directions in the same update procedure. For the CPO baseline, we follow the original implementation where infeasibility and degenerate cases use a recovery direction formed by summing the gradients of currently violated constraints, and the resulting $\dot { \mathrm { Q P } }$ is solved with OSQP [22]. Removing Löwdin slows convergence relative to CPO and only recovers return late in training. When both Löwdin and screening with compression are removed, the early stage becomes highly seed dependent under simultaneous violations, and some runs finish with less than half of LOCU’s final return. This is consistent with redundant and correlated constraint directions producing ill conditioned reduced systems and noisy recovery when they are treated as distinct correction directions without orthogonalization or reduction.

![](images/fc291d90ef6831e758aa3bf38bdb98de3feb09989a2b62b47305fe1268b0b3c3.jpg)

![](images/4eeb2368f06a72b94c0cbcef6cfe302cb65d55839ad8dbea108879e8733672a0.jpg)

![](images/8027e5fa7139e883d3f3d4ec45cbe2c1dd3ade3290658f516c0e5973f5483b25.jpg)  
T LOCU —CPO →w/o Lowdin w/o Lowdin, Screening & Compression

![](images/8c5d74377abeafce7a27d2692ae00ac7a1341b795b0122ce72c32eef95d1b54f.jpg)

Figure 9: Learning curves for w/o Lowdin and w/o Lowdin, Screening & compression, compared with the CPO baseline.  
![](images/ae369969b9fe1ed40cd387900b365de102ca99cf6d52a53d2a67cd977b0272ad.jpg)

![](images/c5b5ba6a1147be3ccb6e297d4143fa14973bce51fd40eac3c8c9427567bec789.jpg)

![](images/5ab30aebbdaef64e84a2d9cd9434216cbef06a21a49e0815e29c5dc24db527ad.jpg)  
εdup = 0.05 εdup = 0.1 εdup = 0.2

![](images/4ff80aa316a2546bc7245249087dc8d2690eac7c8ec9b052a55f966e07748397.jpg)

Figure 10: Ablation on the clustering threshold $\varepsilon _ { \mathrm { d u p } }$ for near-collinear constraint grouping on BC-v3 Redun : 3.  
![](images/1a72cd2d7502b60835df1c0e24494f2133054d3abda974368d5e57ac2b255840.jpg)

![](images/405c3a20ab03f594e90cf5280eeb56f91d8cea83f4dadce07a19e3ce17862f3c.jpg)

![](images/f08584c903da0c6ae65420cfb0aa4a478dd868e35ecccf7867cf72cfe50d04b1.jpg)

![](images/604fa88427b4bed605c9f65b91f189b3b477368022d874fdcd481a2ec7c3522d.jpg)  
Figure 11: Sensitivity analysis on the KL divergence limit compared against SDAC(A) on BC-v3.

## C.4.2 Hyperparameter Sensitivity

We test the cosine similarity threshold $\varepsilon _ { \mathrm { d u p } }$ that controls near-collinear clustering in Section 4.2. Since $\varepsilon _ { \mathrm { d u p } }$ is used to decide whether constraint gradients are treated as redundant, we evaluate it on BC-v3 Redun : 3, where redundancy is strongest. Figure 10 compares threshold values 0.05, 0.1, and 0.2. Differences appear mainly in the early recovery stage, while final returns remain very similar across all tested values. These gaps are small relative to typical across-seed variation and may narrow with additional seeds. Normalized costs converge below the limits in all cases, indicating that LOCU is not particularly sensitive to the exact choice of $\varepsilon _ { \mathrm { d u p } }$ within this range and does not require careful angle tuning to avoid ill-conditioned updates.

Next, we evaluate sensitivity to the trust region radius against SDAC(A) on the BC-v3 task. As the limit size decreases, a smaller KL limit generally reduces approximation error but yields less policy improvement per update. Since our task starts from an infeasible state, a smaller radius requires more updates to reach the feasible region, leaving fewer remaining updates for reward optimization. Both algorithms exhibit similar performance degradation in reward, while showing no further sensitivity in cost satisfaction, as shown in Figure 11.

Finally, we examine numerical hyperparameter sensitivity on the BC-v3 Redun : 3 task by simultaneously assigning $1 0 ^ { - 4 }$ and $1 0 ^ { - 6 }$ to the near-zero threshold, Fisher damping, and CG tolerance. The baseline setup uses default values $1 0 ^ { - 2 }$ for the near-zero threshold, $1 0 ^ { - 1 }$ for Fisher damping, and $1 0 ^ { - 8 }$ for CG tolerance. As illustrated in Figure 12, smaller sweep values yield a small increase in reward while causing larger variance across runs. This behavior is attributed to reduced Fisher damping, which produces a closer approximation to the true natural gradient but increases sensitivity to near-zero eigenvalues of the Fisher matrix, making individual updates more accurate yet overall training less stable.

![](images/5d9f7b87b90a2b8ef47a858cc3fe48231d492ae6a8f896fb2523b05ac2ec1114.jpg)  
+1e-4 +1e-6 + 1e-8  
Figure 12: Ablation on threshold, damping, and CG tolerance values on BC-v3 Redun : 3.

![](images/c7d836f92c91cb4b0f2c7cde9684e92edbb298c2c8295032e4fa575f5168835b.jpg)  
Figure 13: Final evaluation comparison between the original 6-constraint Cassie task and the redundant 18-constraint version.

## C.4.3 Robustness to Redundant Constraints

To evaluate robustness to a larger number of constraints without changing the feasible region, we introduce redundant constraint variants to the goal-object Cassie task. For each base constraint, we keep the original tight threshold and add two looser variants, increasing the total number of constraints from 6 to 18. For example, we use thresholds of $1 5 ^ { \circ } / 2 0 ^ { \circ } / 2 5 ^ { \circ }$ for body tilt, $1 3 / 1 5 / 1 8$ W for energy, and $0 . 0 / 0 . 0 5 / 0 . \dot { . }$ 1 m for the hazard contact margin. This creates a setting with substantial redundancy and heterogeneity across constraint gradients.

Figure 13 compares the final evaluation results between the original 6-constraint Cassie task and the redundant 18-constraint version. LOCU preserves reward under 3× constraint scaling while satisfying all constraints at convergence. Figure 20 further shows that the learning dynamics remain stable after adding redundant constraints, indicating that the additional constraints do not substantially degrade optimization.

Table 6 reports the screening statistics during training in the 18-constraint setting. Even with 18 nominal constraints, only a small subset is active at each update, and redundancy compression further reduces the basis dimension. By the end of training, the average active set size is 4.9 and the average basis dimension is 2.8, so the practical optimization remains compact.

## C.5 Learning curves

We report learning curves for the experiments in the main text and Appendix. Costs are normalized by their corresponding limits, so values above 1 indicate constraint violations.

Table 6: Screening statistics during training in the 18-constraint redundant Cassie task.
<table><tr><td>Phase</td><td>Basis dim.</td><td>Active set</td><td>Redun.</td><td>Inactive</td><td>Zero-grad</td></tr><tr><td>0-20%</td><td> $5 . 7 \pm 0 . 9$ </td><td> $9 . 5 \pm 1 . 6$ </td><td> $3 . 8 \pm 1 . 0$ </td><td> $8 . 5 \pm 1 . 6$ </td><td> $2 . 5 \pm 1 . 2$ </td></tr><tr><td>20-40%</td><td> $3 . 8 \pm 1 . 5$ </td><td> $6 . 5 \pm 3 . 1 $ </td><td> $2 . 7 \pm 1 . 6$ </td><td> $1 1 . 5 \pm 3 . 1$ </td><td> $3 . 2 \pm 1 . 7$ </td></tr><tr><td>40-60%</td><td> $3 . 9 \pm 1 . 7$ </td><td> $6 . 6 \pm 4 . 0$ </td><td> $2 . 7 \pm 1 . 8$ </td><td> $1 1 . 4 \pm 4 . 0$ </td><td> $1 . 6 \pm 1 . 1$ </td></tr><tr><td>60-80%</td><td> $3 . 1 \pm 1 . 5$ </td><td> $5 . 5 \pm 4 . 2$ </td><td> $2 . 4 \pm 1 . 7$ </td><td> $1 2 . 5 \pm 4 . 2$ </td><td> $1 . 2 \pm 0 . 5$ </td></tr><tr><td>80-100%</td><td> $2 . 8 \pm 1 . 4$ </td><td> $4 . 9 \pm 4 . 0$ </td><td> $2 . 1 \pm 1 . 5$ </td><td> $1 3 . 1 \pm 4 . 0$ </td><td> $0 . 9 \pm 0 . 4$ </td></tr></table>

![](images/82aa93485c5195770006695d6decbbf8b6e69b77cef75a35409e2f7659f7f747.jpg)  
LOCU(Ours)SDAC(A)PCRPOCRPOTRPOL-GSTRPOL-VPPOL-GSPPOL-V  
Figure 14: Learning curves on the Bullet Safety Gym v2 suite for $\mathrm { B C , C C , }$ and DC.

![](images/7609c52e71fb05775162c3a725e099710b5299e2e9d882f41645033a456013a0.jpg)  
Figure 15: Learning curves on BC-v3 under the low velocity sweep from the $\mathbf { v } 2$ case to $\kappa _ { \mathrm { l o } } ~ \in$ {0.4, 0.6, 0.8}.

![](images/feb71cca8724b668d78d4eec8189894cb176543f28d7d64375f79f5ba7c2ae67.jpg)  
Figure 16: Learning curves on BC-v3 under redundancy variants redun : 1, redun : 2, and redun : 3. The "-a" denotes the tightest redundant constraint.

![](images/e1d51d5235883301b30206fcf3be0e8c68787fc7f947b0074fbd483d6e523f3f.jpg)  
Figure 17: Learning curves of LOCU with different hyperparameters on BC-v3 under redundancy variants redun : 1 and redun : 3.

![](images/c8e2f7bfda35fa0a499ea9c6608456cfe6de2df7560383e53eb2edf2800faed0.jpg)  
Figure 18: Learning curves on the locomotion tasks with Cassie and Laikago under three simultaneous constraints.

![](images/3b25d19f2e7941736046dfbd69f42afdb3c716d72135b458772ad268cb882d79.jpg)  
Figure 19: Learning curves on the goal-object task with Cassie under additional three simultaneous constraints.

![](images/9343ed4cb3256529d40ff02b46c81c08baead1a60ade185ab07b1fbc12f11b86.jpg)  
Figure 20: Learning curves for the goal-objective task with Cassie and the redundant 18-constraint version.

![](images/9a299d3fcb7f7c8c7467111dd114a0482d257d269e129ede0ba24174ffc4e423.jpg)  
Figure 21: Learning curves for the mjx implementation of circle tasks.

![](images/c5c207441bf6fba4abd836756edb57bc0a11fb5c471715574478ed92d0a31e56.jpg)  
Figure 22: Learning curves for the orthogonalization comparison.