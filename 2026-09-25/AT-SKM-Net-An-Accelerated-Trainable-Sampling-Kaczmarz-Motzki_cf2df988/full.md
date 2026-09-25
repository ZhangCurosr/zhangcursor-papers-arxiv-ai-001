# AT-SKM-Net: An Accelerated Trainable Sampling Kaczmarz-Motzkin Framework for Linear Hard-Constraint Feasibility on Dynamic Graphs

Xiaochen Zhang Haoyu Zhu Yao Zhang Qingchun Hou<sup>∗</sup>

Zhejiang University

xiaochen.23@intl.zju.edu.cn haoyu.22@intl.zju.edu.cn eezhangyao@zju.edu.cn houqingchun@zju.edu.cn

## Abstract

Graph-structured optimization with linear constraints is fundamental to critical infrastructure but faces scalability limits due to massive strict hard constraints and high dimensionality. While recent projection-based methods such as Trainable Sampling Kaczmarz-Motzkin Net (T-SKM-Net) guarantee feasibility, they face high computational costs in dynamic environments by processing the entire constraint set and requiring expensive matrix factorizations. To bridge this gap, we propose the Accelerated Trainable-SKM (AT-SKM) Net framework. To concentrate computation on the active constraints and eliminate redundant calculations, we introduce a hybrid sampling strategy guided by a topology-aware heterogeneous GNN model. To efficiently handle topological shifts in graph-based constraints, we employ a Cholesky Update mechanism that theoretically reduces the equality projection complexity from (N<sup>3</sup>) to (N<sup>2</sup>) under low-rank perturbations. Experiments on random geometric graphs, N-1 Security-Constrained DC-OPF, and minimum-cost gas transport problem demonstrate that AT-SKM reduces iteration counts by up to 85% and achieves 2.95 -7.29 SKM layer speedups, while maintaining zero constraint violations.

## 1 Introduction

Constrained optimization is essential for decision-making in a wide range of critical engineering systems [1]. In particular, graph-based formulations are particularly critical in infrastructure domains, such as power system dispatch and transportation network flow control [2]. These applications typically face a dual challenge: on one hand, decisions must strictly satisfy hard constraints imposed by physical laws or operational requirements, where even minor violations are unacceptable; on the other hand, real-world environments are highly dynamic, for instance, sudden line failures in power grids cause instantaneous changes to the underlying graph topology [3]. This demands that solvers adapt to topological changes and produce feasible solutions within milliseconds, yet traditional optimization algorithms struggle to meet real-time control requirements when confronted with large-scale networks and frequent topological perturbations.

In recent years, neural network has been widely employed to approximate solutions for various optimization problems, due to its powerful function approximation capabilities [4, 5]. However, standard neural networks cannot inherently guarantee the feasibility of their outputs. Existing approaches to address this limitation fall into two categories. Soft constraint methods suppress violations through penalty terms but lack theoretical safety guarantees, posing significant risks in safety-critical applications [6–8]. Hard constraint methods, such as differentiable optimization layers [9–11] and projection-based approaches [12, 13], can enforce feasibility but often require solving complex embedded optimization subproblems during inference, facing severe computational scalability bottlenecks that hinder their application to large-scale graph data with thousands of nodes [14].

Recently, Zhu et al. [15] proposed a Trainable Sampling Kaczmarz-Motzkin Net (T-SKM-Net) framework which successfully enables end-to-end training while guaranteeing linear hard constraint satisfaction by incorporating the SKM algorithm. However, T-SKM-Net relies on SVD to handle equality constraint projection, and topological changes necessitate expensive recomputation. Moreover, in practical optimization problems, typically only a few constraints are active. Consequently, blind uniform sampling wastes substantial computation budget on redundant constraints, then limiting convergence speed.

To address these challenges, we propose AT-SKM-Net, a trainable hard-constraint correction framework for dynamic graph-structured optimization. Instead of simply treating feasibility correction as a post-processing step, AT-SKM-Net aligns the correction procedure with two structural properties of graph-based constrained problems, low-rank topology-induced perturbations and sparse active constraints. Our main contributions are as follows:

• We formulate a graph structure-aware trainable framework that preserves the hard-feasibility guarantee of projection-based SKM while solving the real-time scalability bottlenecks caused by large-scale constraints and topological perturbations.

• We accelerate the equality projection and inequality iteration bottleneck stages of SKMbased correction. For equality projection, we introduce Cholesky updates, reducing the adaptation complexity from $\mathcal { O } \dot { (} N ^ { 3 } )$ to $\mathcal { O } ( N ^ { 2 } )$ under low-rank topological perturbations. For inequality iteration, we introduce active-set-guided hybrid sampling to focus computation on likely binding constraints.

• We develop a heterogeneous graph neural network (HGNN) tailored to dynamic topologies to provide both warm-start candidates and active-set scores for AT-SKM-Net. Experiments demonstrate that AT-SKM-Net achieves substantial speedups while maintaining zero equality and inequality constraint violations.

## 2 Related Work

Existing work has made significant progress in improving the feasibility of solutions produced by neural solvers and traditional optimization methods. The literature relevant to our work can be broadly categorized into three distinct streams: neural methods for hard constraint satisfaction, acceleration strategies for the SKM algorithm, and applications of active-set prediction.

## 2.1 Neural Methods for Hard Constraint Satisfaction

Differentiable Optimization Layer-Based Methods. These approaches integrate optimization problems or iterative operations directly into the neural inference pipeline. While exact solvers like OptNet [9] and CvxPyLayers [10] enable differentiability for QP and general convex problems, their high computational complexity limits scalability. SCQPTH [16] accelerates solving via first-order ADMM, it remains restricted to convex QP formulations. Similarly, iterative projection methods such as LinSATNet [17] and the entropy-regularized GLinSAT [4] achieve constraint satisfaction but incur high computational costs due to the necessity of extensive iterations.

Feasible Set Parameterization Methods. These methods map the latent space onto the feasible region. CP [18] parameterizes solutions for linear inequalities but struggles with high dimensionality. While Rayen [19] and scaling-based approaches [20] extend support to quadratic and other convex constraints, they rely on a pre-identified strictly feasible interior point, rendering them ill-suited for constraints that vary dynamically with inputs. Safety decision rules [14] ensure feasibility via convex combinations of a solver and a safety network, but they require convex constraint sets and inherently compromise optimality.

Correction and Projection Methods. These methods correct infeasible predictions through postprocessing or iterative updates. DC3 [21] employs equality completion and inequality correction but strictly guarantees only equality satisfaction. ProjectNet [22] utilizes alternating projections but is limited to simple non-negative constraints. Rashwan et al. [23] enforce hard linear constraints via the

Component-Averaged Dykstra algorithm. T-SKM-Net [15] achieves rigorous satisfaction of both equality and inequality constraints via null-space projection and SKM iterations. However, the use of SVD factorization and uniform constraint sampling may not fully utilize the structure of sparse active constraint sets and constraint matrix, limiting inference efficiency.

## 2.2 Acceleration Methods for SKM

Among efforts to accelerate Kaczmarz-type iterative methods, the Greedy Kaczmarz algorithm [24] has been proven to achieve the theoretically optimal convergence rate. However, each iteration requires scanning all constraint rows to find the maximum residual, resulting in prohibitively high perstep computational cost that hinders application to large-scale problems. While Randomized Block SKM [25] achieves faster convergence rates through the parallel iteration of multiple constraints, it is plagued by the difficulty of tuning parameters like sampling scale and block size. Probably Accelerated SKM (PASKM) [26] incorporates Nesterov momentum into the SKM framework, leveraging historical gradient information to accelerate convergence toward the feasible region. Nevertheless, this method still employs uniform sampling and does not fundamentally address the inefficiency of constraint sampling.

## 2.3 Application of Active-Set Prediction

Xiang et al. [27] demonstrated that active set prediction can effectively reduce both the computational time and memory usage required for solving Lasso problems. Similarly, Misra et al. [28] leveraged active set prediction to mitigate the complexity of constrained optimization; however, their approach necessitates solving reduced KKT systems for all identified candidate active sets, which can lead to substantial computational overhead if the prediction accuracy is insufficient. Ng et al. [29] propose ensemble-based active set prediction, predicting active constraints at the optimal solution based on input parameters. However, this method assumes the constraint system itself remains unchanged and cannot handle active set shifts caused by topological variations. Schmidtobreick et al. [30] proposed using GNN to predict active sets for warm-starting conventional solvers.

## 3 Preliminaries

Given a graph , consider the objective function F and the feasibility of the state vector $\mathbf { x } \in \mathbb { R } ^ { n }$ subject to linear constraints:

$$
\begin{array} { r l } { \underset { \mathbf { x } } { \operatorname* { m i n } } } & { { } F ( \mathcal { G } , \mathbf { x } ) } \\ { \mathrm { s . t . } } & { { } A ( \mathcal { G } ) \mathbf { x } \leq \mathbf { b } , C ( \mathcal { G } ) \mathbf { x } = \mathbf { d } } \end{array}\tag{1}
$$

where the constraint matrices $A ( \mathcal { G } ) \in \mathbb { R } ^ { n _ { \mathrm { i n e q } } \times n }$ and $C ( \mathcal G ) \in \mathbb { R } ^ { n _ { \mathrm { e q } } \times n }$ , with b $\in \mathbb { R } ^ { n _ { \mathrm { i n e q } } }$ and d $\in \mathbb { R } ^ { n _ { \mathrm { e q } } }$ representing $n _ { \mathrm { i n e q } }$ inequality constraints and $n _ { \mathrm { e q } }$ equality constraints, respectively.

The SKM with null space projection method first obtains an equality-feasible point by projecting an initial guess $\mathbf { x } _ { \mathrm { 0 } }$ onto the equality-constraint hyperplane via: [15]

$$
\mathbf { x } _ { \mathrm { e q } } = \mathbf { x } _ { 0 } - C ^ { \dagger } ( C \mathbf { x } _ { 0 } - \mathbf { d } ) .\tag{2}
$$

Let $N \in \mathbb { R } ^ { n \times r }$ be a basis for null(C). Any equality-feasible iterate admits ${ \bf x } = { \bf x } _ { \mathrm { e q } } + N \omega$ with $\boldsymbol \omega \in \mathbb { R } ^ { r }$

Let $\tilde { A } = A N$ with rows $\tilde { \mathbf { a } } _ { i } ^ { \top }$ , and $\tilde { \mathbf { b } } = \mathbf { b } - A \mathbf { x } _ { \mathrm { e q } }$ . Using a sampled constraint set $S _ { k } \subseteq \{ 1 , \dots , n _ { \mathrm { i n e q } } \}$ and the hinge operator $[ t ] _ { + } \triangleq \operatorname* { m a x } \{ t , 0 \}$ , the SKM update on the null-space coefficients is:

$$
\begin{array} { r } { \begin{array} { r l } & { i _ { k } \in \arg \operatorname* { m a x } _ { i \in \mathcal { S } _ { k } } \left[ \tilde { \mathbf { a } } _ { i } ^ { \top } \boldsymbol { \omega } _ { k } - \tilde { b } _ { i } \right] _ { + } , } \\ & { } \\ & { \omega _ { k + 1 } = \omega _ { k } - \alpha \frac { \left[ \tilde { \mathbf { a } } _ { i _ { k } } ^ { \top } \boldsymbol { \omega } _ { k } - \tilde { b } _ { i _ { k } } \right] _ { + } } { \| \tilde { \mathbf { a } } _ { i _ { k } } \| _ { 2 } ^ { 2 } } \tilde { \mathbf { a } } _ { i _ { k } } , } \\ & { \mathbf { x } _ { k + 1 } = \mathbf { x } _ { \mathrm { e q } } + N \omega _ { k + 1 } , \quad \omega _ { 0 } = \mathbf { 0 } , } \end{array} } \end{array}\tag{3}
$$

where $\alpha \in ( 0 , 2 )$ is the step-size. Equation (3) enforces $C \mathbf { x } _ { k } = \mathbf { d }$ while progressively reducing violations of the transformed inequalities.

## 4 Accelerated Trainable-SKM Framework

We propose the Accelerated Trainable Sampling Kaczmarz-Motzkin Net (AT-SKM-Net) framework to resolve the conflict between strict feasibility guarantees and real-time scalability in dynamic environments. As shown in Figure 1, given a graph instance with updated topology and constraint features, the HGNN first produces a warm-start candidate and active-set scores. The warm-start solution is then projected onto the equality-constraint manifold, and SKM iterations are performed in the null space to remove the remaining inequality violations. The complete inference procedure is summarized in Appendix A.1. The framework aims to accelerate these two stages by using complementary mechanisms for equality projection and inequality iteration to achieve end-to-end speedup:

![](images/24870958f5b4f2d1f505e18c9af34e889794b67d2249f9c0a259d3faed8446a7.jpg)  
Figure 1: Flowchart of the AT-SKM-Net.

Inequality SKM Iteration: Sampling Acceleration. To overcome the inefficiency of uniform sampling, we introduce a hybrid strategy guided by Active-Set Prediction in Sec. 4.1. By leveraging topological priors to identify binding constraints, we effectively shrink the search space from the massive global set to the local active manifold. We provide the theoretical analysis in Sec. 4.2.

Equality Projection: Computing Acceleration. To address the bottleneck of SVD refactorization under low-rank constraint matrix perturbations, we propose a Cholesky Update mechanism in Sec. 4.3. This utilizes the low-rank nature of constraint matrix to theoretically reduce the equality projection complexity from $\mathcal { O } ( N ^ { 3 } )$ to $\mathcal { O } ( N ^ { 2 } )$ .

These modules are powered by our proposed Heterogeneous GNN in Sec. 4.4, which encodes the dynamic graph structure to warm-start the solver and guide the optimization process.

## 4.1 Accelerated Sampling for SKM Iteration: Active-Set Prediction

We address the structural inefficiency of standard SKM methods arising from the mismatch between constraints sparsity and sampling uniformity. The geometry of the feasible region is typically characterized by a sparse set of active constraints, while the vast majority of constraints are redundant at the solution. However, standard SKM remains oblivious to this structure and samples uniformly from the total constraint set $m .$ . This kind of blind search leads to inefficient sampling, where the likelihood of selecting an informative constraint is significantly diminished by the large number of inactive constraints.

To bridge this gap, we propose a neural-network-based hybrid sampling strategy that concentrates sampling probability on the active constraints, theoretically shifting the convergence dependence from the total number of constraints m and the global Hoffman constant to the reduced active set size $| \hat { \cal A } |$ and the local condition number.

## 4.1.1 Hybrid Sampling Strategy via Predicted Active-sets

Consider the linear inequality constraints over the polyhedron:

$$
P = \{ \mathbf { x } \in \mathbb { R } ^ { n } \mid A \mathbf { x } \leq \mathbf { b } \} ,\tag{4}
$$

where $A \in \mathbb { R } ^ { m \times n }$ and $\mathbf { a } _ { i } ^ { \top }$ denotes the i-th row of A. Let $\mathbf { x } ^ { * }$ be the targeted feasible solution.

$\mathbf { A } \mathbf { t } \ \mathbf { x } ^ { * }$ , the active constraint set is defined as

$$
\mathcal { A } ^ { * } ( \mathbf { x } ^ { * } ) = \{ i \in \{ 1 , . . . , m \} \mid \mathbf { a } _ { i } ^ { \top } \mathbf { x } ^ { * } = b _ { i } \} .\tag{5}
$$

In a neighborhood of $\mathbf { x } ^ { * }$ , the feasible region is locally characterized by the affine manifold $A _ { \bf \nabla } A ^ { * } { \bf x } =$ $\mathbf { b } _ { \mathbf { \mathcal { A } } ^ { \ast } }$ since other inequality constraints are still slack.

To address the structural sampling inefficiency, we propose a hybrid sampling strategy guided by neural network predictions. This strategy aims to concentrate computational resources on constraints

with high active probability, while retaining a component of uniform sampling to guarantee theoretical global convergence.

Let $\phi ( \mathcal G )$ be the neural network model. For an input optimization instance ${ \mathcal { G } } ,$ the model outputs unnormalized logits $\mathbf { z } \in \mathbb { R } ^ { m }$ . We transform the predicted active scores into a sampling distribution p via global normalization:

$$
p _ { i } = \frac { \sigma ( z _ { i } ) } { \sum _ { j = 1 } ^ { m } \sigma ( z _ { j } ) } , \quad \forall i \in \{ 1 , \ldots , m \} .\tag{6}
$$

In each SKM iteration $k ,$ with a total batch size $\beta$ and a mixing ratio $\rho \in ( 0 , 1 ]$ , the sampled index set $\tau _ { k }$ is constructed from two components:

Prediction-Guided Exploitation $\left( \tau _ { \mathbf { l e a r n } } \right)$ . We select $\beta _ { 1 } = \lfloor \rho \cdot \beta \rfloor$ indices by weighted sampling based on distribution $\mathbf { p } .$ This step mimics sampling from the predicted active set ${ \hat { A } } ,$ aiming to accelerate convergence by significantly reducing the effective sampling space dimension from m to approximately $| \hat { \cal A } |$

Uniform Exploration $( \tau _ { \mathrm { u n i f } } )$ . We uniformly sample $\beta _ { 2 } = \beta - \beta _ { 1 }$ indices from the remaining constraint set. This step acts as a safeguard, preventing the omission of true active constraints due to prediction errors and ensuring the global robustness of the algorithm.

The final sampled constraint subset for iteration k is $\tau _ { k } = \tau _ { \mathrm { l e a r n } } \cup \tau _ { \mathrm { u n i f } }$ . This hybrid approach effectively leverages prior geometric information while overcoming the potential blind spots of purely prediction-based methods.

## 4.2 Theoretical Analysis

To quantify the acceleration achieved by dimension reduction, we introduce the following assumptions regarding the local geometry and the predictor’s capability:

Assumption 1 (Local Active-Set Stability). There exists a neighborhood $\mathcal { U }$ of $\mathbf { x } ^ { * }$ such that for any iterate $\mathbf { x } \in \mathcal { U }$ , the set of active constraints remains invariant at the projection, which means that all constraints inactive at $\mathbf { x } ^ { * }$ remain slack at $\Pi _ { P } ( \mathbf { x } )$ , and $\Pi _ { P } ( \mathbf { x } )$ lies on the affine subspace defined by $\boldsymbol { \mathcal { A } } ^ { * } ( \mathbf { x } ^ { * } )$ . That is, $\mathcal { A } ( \Pi _ { P } ( \mathbf { x } ) ) = \mathcal { A } ^ { * } ( \mathbf { x } ^ { * } )$

Assumption 2 (Effective Active-Set Covering). We assume the predictor identifies a candidate set $\hat { A }$ that satisfies two conditions:

• Coverage: It includes the true active set, $\mathrm { i . e . , } A ^ { \ast } \subseteq \hat { A } .$

• Sparsity: Its size is significantly smaller than the total number of constraints, i.e., $| { \hat { A } } | \ll m$

We provide a unified theoretical analysis of the acceleration mechanism. We first establish the global convergence guarantee of the algorithm, and then quantify its acceleration relative to standard SKM in the local linear regime.

Theorem 1 (Global Convergence and Robustness). Provided that the mixing ratio satisfies $\rho < 1$ the proposed algorithm maintains global linear convergence in expectation toward thefeasible set $P .$

Proof Sketch. The proof relies on the positive sampling probabilities. Since $\rho < 1$ , every constraint retains a strictly positive probability of being sampled. This satisfies the sufficient condition for global linear convergence established in standard randomized Kaczmarz theory [31, 32], ensuring robustness against prediction errors. Complete proof in Appendix A.2. □

Theorem 2 (Efficiency Gain via Dimension Reduction). Consider the iterate in the local linear regime from Assumption 1 with row-normalized constraints. Comparing uniform sampling over $\{ 1 , \ldots , m \}$ and our proposed uniform sampling over the predicted set $\hat { \mathcal { A } } ,$ under Assumption 2, the expected one-step error contraction rates are:

• Standard SKM (Uniform on m):

$$
\mathbb { E } [ \| \mathbf { e } _ { k + 1 } \| ^ { 2 } ] \leq \left( 1 - { \frac { 1 } { m \cdot \mathcal { H } ^ { 2 } ( A ) } } \right) \| \mathbf { e } _ { k } \| ^ { 2 }\tag{7}
$$

• Accelerated SKM (Uniform on ${ \hat { \mathcal { A } } } ) .$

$$
\mathbb { E } [ \| \mathbf { e } _ { k + 1 } \| ^ { 2 } ] \leq \left( 1 - \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } ( A _ { A ^ { * } } ) } { | \hat { A } | } \right) \| \mathbf { e } _ { k } \| ^ { 2 }\tag{8}
$$

where $\mathcal { H } ( A )$ is the global Hoffman constant, and $\sigma _ { \mathrm { m i n } } ( A _ { \mathcal { A } ^ { \ast } } )$ denotes the smallest non-zero singular value ofthe true active submatrix.

Proof Sketch. The acceleration stems from the error contraction behavior in the local linear regime by Assumption 1. In neighborhood $u ,$ , the hinge loss for any inactive constraint has $[ a _ { i } ^ { \top } x - b _ { i } ] _ { + } = 0$ Consequently, the summation term in the expected update collapses from the full constraint set to the active set $\mathcal { A } ^ { * }$ . Complete proof in Appendix A.2. □

This theorem shows the acceleration through two complementary mechanisms: sampling efficiency and geometric regularization. First, the reduction in the denominator from $m$ to $| \hat { A } |$ reflects the compression of the sampling space, yielding significant acceleration whenever $| { \hat { A } } | \ll m$ . Second, the convergence rate is governed by the singular value $\sigma _ { \mathrm { m i n } }$ of the true active set $\ b { A } ^ { * }$ rather than the Hoffman constant $\mathcal { H } ,$ which accounts for the worst-case conditioning among all submatrices [33].

## 4.3 Accelerated Computing for Equality Projection: Cholesky Update

In T-SKM-Net, the SKM layer first projects the neural prediction $\mathbf { x } _ { \mathrm { 0 } }$ onto the equality manifold $\mathbf { C x } = \mathbf { d }$ and then performs inequality iterations in the null space of C. This equality projection is essential for preserving feasibility, but it becomes a major bottleneck in dynamic graph problems. When the graph topology changes, the equality matrix C also changes, and the SVD-based projection used in T-SKM-Net has to be recomputed under an expensive $\breve { \mathcal { O } ( n ^ { 3 } ) }$ computational cost for each perturbed instance [34].

However, mild topological changes, such as line outages or reconnections, typically induce localized modifications in the adjacency matrix or the graph Laplacian. Motivated by this observation, we propose an incremental Cholesky update framework that exploits such structured changes to accelerate the equality-constraint projection phase.

Given a row full-rank matrix $\mathbf { C } \in \mathbb { R } ^ { m \times n }$ , a rank-1 perturbation defined by $\tilde { \mathbf { C } } = \mathbf { C } + \mathbf { u v } ^ { \top }$ induces a rank-2 modification in its corresponding Gram matrix $\mathbf { G } = \mathbf { C } \mathbf { C } ^ { \top }$ . To handle this efficiently, we bypass computationally expensive SVD re-computations in favor of a generic incremental Cholesky update strategy. By sequentially applying one rank-1 update and one rank-1 downdate to the original $\mathbf { G } = \mathbf { L } \mathbf { L } ^ { \top }$ <sub>, we can directly evolve the triangular factor L</sub>˜ <sub>to adapt to the new topology without</sub> reconstructing the matrix from scratch.

Based on the updated factor $\tilde { \mathbf { L } } ,$ we transform the explicit matrix inversion into two efficient triangular matrix solves. For the constraint residual $\mathbf { r } = \tilde { \mathbf { C } } \mathbf { x } _ { 0 } - \mathbf { d }$ , the projected solution can be expressed as:

$$
\tilde { \mathbf { x } } _ { \mathrm { e q } } = \mathbf { x } _ { 0 } - \tilde { \mathbf { C } } ^ { \top } ( \tilde { \mathbf { L } } ^ { - \top } ( \tilde { \mathbf { L } } ^ { - 1 } \mathbf { r } ) )\tag{9}
$$

The null-space representation used by the subsequent SKM iterations is updated consistently with the perturbed equality constraints. Instead of reconstructing the whole null-space basis from scratch, we reuse the previous basis and correct only the low-dimensional subspace affected by the perturbation [35]. Intuitively, a low-rank change in C only alters a small number of directions in the null space, while the remaining directions can be retained after an orthogonal alignment. The explicit construction of this basis update is provided in Appendix B.1.

Computational Complexity. While the SVD approach necessitates a full re-factorization with cubic complexity $\mathcal { O } ( \bar { N } ^ { 3 } )$ , our proposed framework significantly reduces the computational burden by exploiting the low-rank perturbations of constraint matrix. By replacing global decomposition with localized Cholesky rank-1 updates and efficient basis rotations, we effectively decouple the high-dimensional problem into sequential vector operations and triangular solves. This strategy lowers the overall complexity to $\mathcal { \dot { O } } ( N ^ { 2 } )$ , offering a theoretical speedup of order N compared to standard methods. Description with more details is in Appendix B.2.

## 4.4 Topology-Aware Heterogeneous Graph Neural Network for Active-Set Prediction

The hybrid sampling strategy proposed in Section 4.1 relies on precise estimation of active probabilities for individual constraints, which poses two key challenges: capturing the heterogeneity of constraints, and generalizing across varying topological scenarios. To address such challenges, we propose a heterogeneous graph neural network (HGNN) architecture that aligns with the structure of constraints and topology of graph-constrained optimization problems.

## 4.4.1 Heterogeneous Graph Representation

In graph-based optimization problems, constraints are naturally partitioned into node-level constraints and edge-level constraints. Homogeneous GNNs treat edges as message-passing channels, which tends to smooth edge-specific information into node representations [36]. This smoothing effect hinders the extraction of distinct active probabilities for edge-based couplings.

To address this, we adopt a heterogeneous bipartite graph architecture where both original nodes and edges are modeled as distinct computational entities. This design establishes a structural correspondence between neural network embeddings and the optimization constraint set. Let $\mathbf { h } _ { u } ^ { ( l ) }$ and $\mathbf { h } _ { e } ^ { ( l ) }$ denote the latent representations of node u and edge $\boldsymbol { e } = \left( u , v \right)$ at layer l. The heterogeneous message passing updates are:

$$
\mathbf { h } _ { e } ^ { ( l + 1 ) } = \phi _ { \mathrm { e d g e } } \left( \mathbf { h } _ { e } ^ { ( l ) } , \mathbf { h } _ { u } ^ { ( l ) } , \mathbf { h } _ { v } ^ { ( l ) } \right) , \qquad \mathbf { h } _ { u } ^ { ( l + 1 ) } = \phi _ { \mathrm { n o d e } } \left( \mathbf { h } _ { u } ^ { ( l ) } , \sum _ { e \in \mathcal { N } ( u ) } \mathbf { h } _ { e } ^ { ( l + 1 ) } \right) ,\tag{10}
$$

where $\phi _ { \mathrm { e d g e } }$ and $\phi _ { \mathrm { n o d e } }$ are nonlinear transition functions. This formulation preserves the representation $\mathbf { h } _ { e }$ of each coupling constraint independently throughout the propagation depth, rather than absorbing it into nodal states.

## 4.4.2 Static-Dynamic Feature Separation

When the graph topology changes, the neural model needs to generalize across these perturbations. We observe that problem inputs can be decomposed into two categories: static geometric parameters $\mathbf { z } _ { \mathrm { s t a t } }$ that define the constraint coefficient matrix A, and dynamic instance conditions ${ \bf z } _ { \mathrm { d y n } }$ that define the boundary vector b.

To prevent static structural information from being diluted by dynamic variations during deep message passing, we employ a structure-aware normalization mechanism. The mechanism first concatenates the dynamic hidden features $\mathbf { h } ^ { ( l ) }$ with static physical features $\mathbf { z } _ { \mathrm { s t a t } }$ , then generates the normalization parameters through graph convolution:

$$
\mathbf { h } ^ { ( l + 1 ) } = \gamma ( \mathbf { h } ^ { ( l ) } , \mathbf { z } _ { \mathrm { s t a t } } ) \odot \frac { \mathbf { h } ^ { ( l ) } - \mu } { \sigma } + \beta ( \mathbf { h } ^ { ( l ) } , \mathbf { z } _ { \mathrm { s t a t } } ) ,\tag{11}
$$

where the affine parameters $\gamma$ and $\beta$ are generated from the concatenation $[ { \bf h } ^ { ( l ) } \rVert _ { \bf z _ { \mathrm { s t a t } } } ]$ . This formulation is related to adaptive normalization method proposed by [37], but instead of injecting random noise, we condition on the global static topology of the graph. This helps ensure that predictions remain consistent with constraints of $\mathcal { G }$ when its topology changes.

## 4.4.3 Output Heads

The HGNN serves as a generator for the accelerated solver. Its final embeddings are fed into two output heads. A regression head predicts the primal variable candidate $\hat { \mathbf { x } } _ { 0 }$ to provide a warm start for the SKM layer. A classification head generates active scores for both node and edge constraints. The sampling distribution p required by the hybrid strategy is obtained via global normalization:

$$
p _ { k } = \frac { \sigma ( \mathbf { M L P } ( \mathbf { h } _ { k } ) ) } { \sum _ { j \in \mathcal { V } \cup \mathcal { E } } \sigma ( \mathbf { M L P } ( \mathbf { h } _ { j } ) ) } ,\tag{12}
$$

where k indexes over all node and edge constraints.

## 5 Experiments

We evaluate the proposed AT-SKM method on random geometric graphs and N-1 security-constrained DC optimal power flow problem, and provide an additional validation on the minimum-cost gas transport problem in Appendix C. All experiments are conducted on an Apple M4 chip.

## 5.1 Equality Projection on Random Geometric Graphs

We validate the asymptotic computational complexity of the proposed Cholesky update method for equality-constraint projection using random geometric graphs (RGGs) of varying sizes. In an RGG, nodes are uniformly distributed within the unit square, with edges connecting pairs whose Euclidean distance falls below a threshold r; this local connectivity structure closely resembles real-world networks where physical proximity constrains connections.

Figure 2 compares the computation time of the Cholesky Update method, which incrementally updates the initial factorization, against SVD recomputation that performs a full decomposition for each topology change. On a log-log scale, the SVD computation time closely follows the $\mathcal { O } ( N ^ { 3 } )$ reference curve as the number of nodes increases. Meanwhile, the empirical runtime of our proposed method aligns well with $\mathcal { O } ( N ^ { 2 } )$ . Notably, when the graph size reaches approximately 2000 nodes, SVD requires seconds to complete, which is prohibitive for safety-critical real-time applications such as power system dispatch. In contrast, the Cholesky update maintains computation times on the order of $1 0 ^ { - 2 }$ seconds, demonstrating a substantial efficiency advantage. A detailed breakdown of these computational overheads is provided in Appendix B.3.

![](images/9bf334b8c2674f3a864530fe882d9093f50a23dc57ae30b9ae12b0e70c3b4696.jpg)  
Figure 2: Computational complexity comparison between SVD recomputation and Cholesky update on RGGs. SVD exhibits $\mathcal { O } ( n ^ { 3 } )$ scaling while Cholesky update achieves $\mathcal { O } ( \dot { n } ^ { 2 } )$

## 5.2 N-1 Security-Constrained DC Optimal Power Flow

We evaluate the performance of our proposed method on the N-1 Security-Constrained DC Optimal Power Flow (N-1 SC DC-OPF) problem. This problem stands as a fundamental challenge in critical power system infrastructure and represents a distinct class of graph-based constrained optimization characterized by strict safety requirements and dynamic network topologies [38].

The primary objective is to minimize costs subject to physical constraints, including power balance equations, generator output limits, and transmission line thermal limits. The rigorous N-1 security criterion mandates that the system must maintain feasibility following the loss of a single transmission component [39]. The detailed mathematical formulation is provided in Appendix D.1.

The experiment is based on IEEE 57-bus, 118-bus, and 300-bus systems. While we consider all feasible N-1 contingencies, we include only a subset of these scenarios in the training set to assess the model’s adaptability and generalization to unseen topologies. The testing set evaluates the solver’s performance in terms of cost optimization and constraint satisfaction across both seen and unseen topologies. The ground truth solutions for all instances are generated using the PyPower library [40].

We benchmark our method against: (1) PyPower: A traditional interior-point solver serving as the ground truth; (2) NN / HGNN: End-to-end learning baselines without explicit constraint enforcement; (3) DC3: A differentiable optimization method guaranteeing equality constraints via projection; and (4) T-SKM-Net: The original Trainable SKM framework. To decouple the contributions of our core modules, we evaluate three variants: AT-SKM-A (Active-set prediction only), AT-SKM-C (Cholesky update only), and AT-SKM-AC (Full).

As shown in Table 1, our proposed method achieves significant speedups over the T-SKM-Net baseline across all system sizes, while strictly satisfying all equality and inequality constraints without compromising solution quality. Regarding solution quality, although unconstrained baselines

Table 1: Performance comparison across different methods. The Optimality Gap and Constraint Violations $( \mathrm { t o l } { = } 1 0 ^ { - 5 } )$ are formatted as “Seen / Unseen”. The “Iter. Num.” column reports the mean (max) number of inequality iterations. The SKM-related metrics are reported on unseen scenarios to demonstrate robust acceleration.
<table><tr><td>METHOD</td><td>TOT. TIME (MS)</td><td>ITER. NUM.</td><td>SKM TIME</td><td>SKM SPEEDUP</td><td>OPT. GAP (%)</td><td>MAX EQ. VIO.</td><td>MAX INEQ. VIO.</td></tr><tr><td colspan="8"> $\mathrm { I E E E ~ } 5 7 \mathrm { - B U S ~ S Y S T E M } \colon n = 6 4 , n _ { \mathrm { E 9 } } = 5 8 , n _ { \mathrm { I N E 9 } } = 1 7 4$ </td></tr><tr><td>PYPOWER</td><td>14.785</td><td></td><td></td><td></td><td>0 / 0</td><td>0 / 0</td><td>0 / 0</td></tr><tr><td>NN</td><td>0.048</td><td></td><td></td><td></td><td>0.173 / 5.379</td><td>24.77 / 53.07</td><td>0.101 / 192.71</td></tr><tr><td>HGNN</td><td>4.735</td><td></td><td></td><td></td><td>0.071 / 0.061</td><td>17.52 / 5.163</td><td>0.061 / 0.041</td></tr><tr><td>DC3</td><td>6.278</td><td></td><td></td><td></td><td>0.013 / 0.013</td><td>0 / 0</td><td>0.067 / 0.051</td></tr><tr><td>T-SKM-NET</td><td>5.831</td><td>6.93 (148)</td><td>1.096</td><td>1×</td><td>0.008 / 0.008</td><td>0 / 0</td><td>0 / 0</td></tr><tr><td>AT-SKM-A</td><td>5.202</td><td>1.42 (11)</td><td>0.467</td><td>2.35×</td><td>0.008 / 0.008</td><td>0 / 0</td><td>0/0</td></tr><tr><td>AT-SKM-C</td><td>5.353</td><td>6.93 (148)</td><td>0.618</td><td>1.77×</td><td>0.008 / 0.008</td><td>0 / 0</td><td>0 /0</td></tr><tr><td>AT-SKM-AC</td><td>5.009</td><td>1.42 (11)</td><td>0.274</td><td>4.00×</td><td>0.008 / 0.008</td><td>0 /0</td><td>0 / 0</td></tr><tr><td colspan="8"> $\mathrm { I E E E 1 1 8 - B U S ~ S Y S T E M : } \ n = 1 7 2 , n _ { \mathrm { E 9 } } = 1 1 9 , n _ { \mathrm { I N E Q } } = 4 8 0$ </td></tr><tr><td>PYPOWER</td><td>31.557</td><td></td><td></td><td></td><td>0 / 0</td><td>0 / 0</td><td>0 / 0</td></tr><tr><td>NN</td><td>0.057</td><td></td><td></td><td></td><td>0.142 / 1.481</td><td>72.59 / 93.21</td><td>0.031 / 664.11</td></tr><tr><td>HGNN</td><td>6.938</td><td></td><td></td><td></td><td>0.017 / 0.016</td><td>61.06 / 67.36</td><td>0.010 /0.047</td></tr><tr><td>DC3</td><td>8.040</td><td></td><td></td><td></td><td>0.019 / 0.019</td><td>0 / 0</td><td>0.011 / 0.009</td></tr><tr><td>T-SKM-NET</td><td>10.251</td><td>34.60 (141)</td><td>3.313</td><td>1×</td><td>0.026 / 0.025</td><td>0 /0</td><td>0 / 0</td></tr><tr><td>AT-SKM-A</td><td>9.349</td><td>15.68 (57)</td><td>2.411</td><td>1.37×</td><td>0.026 / 0.025</td><td>0 / 0</td><td>0/0</td></tr><tr><td>AT-SKM-C</td><td>8.942</td><td>34.60 (141)</td><td>2.004</td><td>1.65×</td><td>0.026 / 0.025</td><td>0 / 0</td><td>0/0</td></tr><tr><td>AT-SKM-AC</td><td>8.060</td><td>15.68 (57)</td><td>1.122</td><td>2.95×</td><td>0.026 / 0.025</td><td>0 / 0</td><td>0/0</td></tr><tr><td colspan="8"> $\mathrm { I E E E } 3 0 0 \mathrm { - B U S } \mathrm { S Y S T E M } \mathrm { : } n = 3 6 9 , n _ { \mathrm { E Q } } = 3 0 1 , n _ { \mathrm { I N E Q } } = 9 6 0$ </td></tr><tr><td>PYPOWER</td><td>64.398</td><td></td><td></td><td></td><td>0 /0</td><td>0 /0</td><td>0 / 0</td></tr><tr><td>NN</td><td>0.078</td><td></td><td></td><td></td><td>0.057 / 0.367</td><td></td><td>2.00E+3 / 9.44E+3 1.53E+3 / 8.52E+3</td></tr><tr><td>HGNN</td><td>9.390</td><td></td><td></td><td></td><td>0.068 / 0.068</td><td>64.61 / 62.36</td><td>537.66 / 0.086</td></tr><tr><td>DC3</td><td>16.138</td><td></td><td></td><td></td><td>0.102 / 0.102</td><td>0 / 0</td><td>0.471 / 0.465</td></tr><tr><td>T-SKM-NET</td><td>18.382</td><td>24.72 (285)</td><td>8.992</td><td>1×</td><td>0.108 / 0.108</td><td>0 /0</td><td>0 / 0</td></tr><tr><td>AT-SKM-A</td><td>17.264</td><td>3.68 (9)</td><td>7.874</td><td>1.14×</td><td>0.108 / 0.108</td><td>0 / 0</td><td>0/0</td></tr><tr><td>AT-SKM-C</td><td>12.179</td><td>24.72 (285)</td><td>2.789</td><td>3.22×</td><td>0.110/0.110</td><td>0 /0</td><td>0 / 0</td></tr><tr><td>AT-SKM-AC</td><td>10.623</td><td>3.69 (9)</td><td>1.233</td><td>7.29×</td><td>0.110 /0.110</td><td>0 / 0</td><td>0 / 0</td></tr></table>

NN and HGNN achieve small optimality gaps, they exhibit varying degrees of constraint violations, rendering them unsafe for real-world operations. Moreover, the NN shows poor generalization on unseen topologies. While DC3 satisfies equality constraints, it fails to eliminate inequality violations.  
We further analyze the isolated impacts of active-set prediction and Cholesky updates:  
Active-Set Prediction: AT-SKM-A reduces the average iteration count by 80%, 55%, and 85% on the 57, 118, and 300-bus systems, respectively. This confirms that our hybrid sampling strategy effectively concentrates computational resources on the sparse subset of active constraints.

Cholesky Update: AT-SKM-C significantly accelerates the projection phase. In the 300-bus system, although the results indicate that the Cholesky update does not affect the inequality iteration rate, AT-SKM-C reduces the SKM runtime from 8.99ms to 2.79ms (3.22 speedup), whereas AT-SKM-A reduces iterations by 85% but only yields a 1.14 speedup. This reveals that in large-scale systems, the computational bottleneck shifts from the iterative search to linear algebraic operations, highlighting the ${ \dot { \mathcal { O } } } ( n ^ { 2 } )$ advantage of our Cholesky update.

Interestingly, we observe a synergistic effect over these two methods. The total speedup of AT-SKM-AC approximates or even exceeds the product of the individual speedups. For instance, on the 57-bus system, $S _ { A } ( 2 . 3 5 \times ) \times S _ { C } ( 1 . 7 7 \times ) \approx \stackrel { . } { S } _ { A C } ( 4 . 0 0 \times )$ , and on the 300-bus system, the combined speedup (7.29 ) significantly surpasses the product of individual gains $S _ { A } ( 1 . 1 \dot { 4 } \times ) \times S _ { C } ( 3 . 2 2 \times ) \ll$ $S _ { A C } ( 7 . 2 9 \times )$ , suggesting a super-linear efficiency improvement.

Further experiments are provided in the appendix: hybrid sampling sensitivity (D.4), detailed runtime breakdown (D.5), high-load robustness (D.6), active-set prediction quality (D.7), active-set perturbation (D.8), and transfer learning with pseudo-label active-set training (D.9).

## 6 Conclusion

In this paper, we propose the AT-SKM framework to effectively address the trade-off between strict feasibility and real-time scalability in graph-structured optimization. By combining neural network active-set predictions with efficient Cholesky incremental updates, AT-SKM-Net overcomes the computational bottlenecks that limit the original projection-based T-SKM-Net in dynamic environments.

## References

[1] James Kotary, Ferdinando Fioretto, Pascal Van Hentenryck, and Bryan Wilder. End-to-end constrained optimization learning: A survey. In International Joint Conference on Artificial Intelligence, 2021.

[2] Wei Wei, WU Danman, WU Qiuwei, Miadreza Shafie-Khah, and João PS Catalão. Interdependence between transportation system and power distribution system: A comprehensive review on models and applications. Journal of Modern Power Systems and Clean Energy, 7(3): 433–448, 2019.

[3] Mojtaba Khanabadi, Hassan Ghasemi, and Meysam Doostizadeh. Optimal transmission switching considering voltage security and n-1 contingency analysis. IEEE Transactions on Power Systems, 28(1):542–550, 2013. doi: 10.1109/TPWRS.2012.2207464.

[4] Hongtai Zeng, Chao Yang, Yanzhen Zhou, Cheng Yang, and Qinglai Guo. Glinsat: The general linear satisfiability neural network layer by accelerated gradient descent. Advances in Neural Information Processing Systems, 37:122584–122615, 2024.

[5] Utkarsh Utkarsh, Danielle C. Maddix, Ruijun Ma, Michael W. Mahoney, and Yuyang Wang. End-to-end probabilistic framework for learning with hard constraints. arXiv preprint arXiv:2506.07003, 2025.

[6] Ferdinando Fioretto, Pascal Van Hentenryck, Terrence WK Mak, Cuong Tran, Federico Baldo, and Michele Lombardi. Lagrangian duality for constrained deep learning. In Joint European conference on machine learning and knowledge discovery in databases, pages 118–135. Springer, 2020.

[7] Luiz FO Chamon, Santiago Paternain, Miguel Calvo-Fullana, and Alejandro Ribeiro. Constrained learning with non-convex losses. IEEE Transactions on Information Theory, 69(3): 1739–1760, 2022.

[8] James Kotary and Ferdinando Fioretto. Learning constrained optimization with deep augmented lagrangian methods. arXiv preprint arXiv:2403.03454, 2024.

[9] Brandon Amos and J Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks. In International conference on machine learning, pages 136–145. PMLR, 2017.

[10] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J Zico Kolter. Differentiable convex optimization layers. Advances in neural information processing systems, 32, 2019.

[11] Youngjae Min and Navid Azizan. Hardnet: Hard-constrained neural networks with universal approximation guarantees. arXiv preprint arXiv:2410.10807, 2024.

[12] Enming Liang, Minghua Chen, and Steven H. Low. Low complexity homeomorphic projection to ensure neural-network solution feasibility for optimization over (non-)convex set. In Proceedings ofthe 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

[13] Xiang Pan, Tianyu Zhao, Minghua Chen, and Shengyu Zhang. Deepopf: A deep neural network approach for security-constrained dc optimal power flow. IEEE Transactions on Power Systems, 36(3):1725–1735, 2020.

[14] Gonzalo E Constante-Flores, Hao Chen, and Can Li. Enforcing hard linear constraints in deep learning models with decision rules. arXiv preprint arXiv:2505.13858, 2025.

[15] Haoyu Zhu, Yao Zhang, Jiashen Ren, and Qingchun Hou. T-skm-net: Trainable neural network framework for linear constraint satisfaction via sampling kaczmarz-motzkin method. arXiv preprint arXiv:2512.10461, 2025.

[16] Andrew Butler. Scqpth: an efficient differentiable splitting method for convex quadratic programming. arXiv preprint arXiv:2308.08232, 2023.

[17] Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang, and Junchi Yan. Linsatnet: the positive linear satisfiability neural networks. In International Conference on Machine Learning, pages 36605–36625. PMLR, 2023.

[18] Thomas Frerix, Matthias Nießner, and Daniel Cremers. Homogeneous linear inequality constraints for neural network activations. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 748–749, 2020.

[19] Jesus Tordesillas, Jonathan P How, and Marco Hutter. Rayen: Imposition of hard convex constraints on neural networks. arXiv preprint arXiv:2307.08336, 2023.

[20] Andrei V Konstantinov and Lev V Utkin. A new computationally simple approach for implementing neural networks with output hard constraints. In Doklady Mathematics, volume 108, pages S233–S241. Springer, 2023.

[21] Priya Donti, David Rolnick, and J Zico Kolter. Dc3: A learning method for optimization with hard constraints. In International Conference on Learning Representations, 2021.

[22] Rares Cristian, Pavithra Harsha, Georgia Perakis, Brian L Quanz, and Ioannis Spantidakis. End-to-end learning for optimization via constraint-enforcing approximators. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 7253–7260, 2023.

[23] Ahmed Rashwan, Keith Briggs, Chris Budd, and Lisa Kreusser. Enforcing convex constraints in graph neural networks. arXiv preprint arXiv:2510.11227, 2025.

[24] Julie Nutini, Behrooz Sepehry, Issam Laradji, Mark Schmidt, Hoyt Koepke, and Alim Virani. Convergence rates for greedy kaczmarz algorithms, and faster randomized kaczmarz rules using the orthogonality graph. arXiv preprint arXiv:1612.07838, 2016.

[25] Yanjun Zhang and Hanyu Li. Randomized block subsampling kaczmarz-motzkin method. Linear Algebra and its Applications, 667:133–150, 2023.

[26] Md Sarowar Morshed, Md Saiful Islam, and Md Noor-E-Alam. Sampling kaczmarz-motzkin method for linear feasibility problems: generalization and acceleration. Mathematical Programming, 194(1):719–779, 2022.

[27] Zhen James Xiang, Yun Wang, and Peter J Ramadge. Screening tests for lasso problems. IEEE transactions on pattern analysis and machine intelligence, 39(5):1008–1027, 2016.

[28] Sidhant Misra, Line Roald, and Yeesian Ng. Learning for constrained optimization: Identifying optimal active constraint sets. INFORMS Journal on Computing, 34(1):463–480, 2022.

[29] Yeesian Ng, Sidhant Misra, Line A. Roald, and Scott Backhaus. Statistical learning for dc optimal power flow. In 2018 Power Systems Computation Conference (PSCC), pages 1–7, 2018. doi: 10.23919/PSCC.2018.8442859.

[30] Ella J Schmidtobreick, Daniel Arnström, Paul Häusner, and Jens Sjölund. Warm-starting active-set solvers using graph neural networks. arXiv preprint arXiv:2511.13174, 2025.

[31] Jamie Haddock and Anna Ma. Greed works: An improved analysis of sampling kaczmarz– motzkin. SIAM Journal on Mathematics of Data Science, 3(1):342–368, 2021.

[32] Jesus A De Loera, Jamie Haddock, and Deanna Needell. A sampling kaczmarz–motzkin algorithm for linear feasibility. SIAM Journal on Scientific Computing, 39(5):S66–S87, 2017.

[33] Diethard Klatte and Gisbert Thiere. Error bounds for solutions of linear equations and inequalities. Zeitschriftfür Operations Research, 41(2):191–214, 1995.

[34] Nisheeth K Vishnoi. Laplacian solvers and their algorithmic applications. Foundations and Trends® in Theoretical Computer Science, 8(1-2):1–145, 2010.

[35] Jorge Nocedal. Numerical optimization. Springer Ser. Oper. Res. Financ. Eng./Springer, 2006.

[36] Salah Ghamizi, Aoxiang Ma, Jun Cao, and Pedro Rodriguez Cortes. Opf-hgnn: Generalizable heterogeneous graph neural networks for ac optimal power flow. In 2024 IEEE Power & Energy Society General Meeting (PESGM), pages 1–5. IEEE, 2024.

[37] Moshe Eliasof, Beatrice Bevilacqua, Carola-Bibiane Schönlieb, and Haggai Maron. Granola: Adaptive normalization for graph neural networks. Advances in Neural Information Processing Systems, 37:90514–90551, 2024.

[38] Xiaochen Zhang, Kaijie Xu, Shengchen Liao, Lin Qiu, Chengjin Ye, and Youtong Fang. Disturbed security-constrained and time-variant optimal power flow for dynamic power system based on chaotic-genetic-centroid puffin optimization. Applied Energy, 397:126287, 2025.

[39] Min Zhou, Chensheng Liu, Amir Abiri Jahromi, Deepa Kundur, Jing Wu, and Chengnian Long. Revealing vulnerability of n-1 secure power systems to coordinated cyber-physical attacks. IEEE Transactions on Power Systems, 38(2):1044–1057, 2023. doi: 10.1109/TPWRS.2022.3169482.

[40] Richard Lincoln. PYPOWER. https://pypi.org/project/PYPOWER/5.1.19/, 2025. Version 5.1.19; released July 10, 2025; accessed January 1, 2026.

[41] Dennis Leventhal and Adrian S Lewis. Randomized methods for linear constraints: convergence rates and conditioning. Mathematics of Operations Research, 35(3):641–654, 2010.

[42] Alston S Householder. Unitary triangularization of a nonsymmetric matrix. Journal ofthe ACM (JACM), 5(4):339–342, 1958.

[43] Martin Schmidt, Denis Aßmann, Robert Burlacu, Jesco Humpola, Imke Joormann, Nikolaos Kanelakis, Thorsten Koch, Djamal Oucherif, Marc E. Pfetsch, Lars Schewe, et al. Gaslib—a library of gas network instances. Data, 2(4):40, 2017.

[44] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

# A AT-SKM-Net Algorithm Flow and Theoretical Proofs

## A.1 AT-SKM-Net Algorithm Flow

Algorithm 1 Pseudo-code of the proposed AT-SKM-Net Framework.   
Require: Constraint data $( A , b , C , d )$ , predictor ϕ, sampling batch size β, mixing ratio ρ, maximum iterations K   
Ensure: Feasible solution x<sub>K</sub>   
1: (x , s) ← ϕ(A, b, C, d)   
2: $p _ { i } \gets \sigma ( s _ { i } ) / \sum _ { j = 1 } ^ { m } \sigma ( s _ { j } ) , i = 1 , \ldots , m$   
3: Equality projection stage.   
4: L ← CHOLESKYUPDATE(C) such that $L L ^ { \top } = C C ^ { \top }$   
5: Solve $L L ^ { \top } y = C x _ { 0 } - d$ and set $x _ { \mathrm { e q } } = x _ { 0 } - C ^ { \top } y$   
6: N ← NULLSPACEUPDATE(C, L), CN = 0   
7: $\widetilde { A }  A N , \quad \widetilde { b }  b - A x _ { \mathrm { e q } } , \quad \omega _ { 0 }  0$   
8: Inequality iteration stage.   
9: for $\bar { k } = \dot { 0 , } 1 , \dots , K - \tilde { 1 }$ do   
10: $\beta _ { \mathrm { a c t } }  \lfloor \rho \beta \rfloor , \quad \beta _ { \mathrm { u n i f } }  \beta - \beta _ { \mathrm { a c t } }$   
11: Sample τ<sub>act</sub> from {1, . . . , m} according to p with size $\beta _ { \mathrm { a c t } }$   
12: Uniformly sample $\tau _ { \mathrm { u n i f } }$ from $\{ 1 , \ldots , m \}$ with size $\beta _ { \mathrm { u n i f } }$   
13: $\tau _ { k }  \tau _ { \mathrm { a c t } } \cup \tau _ { \mathrm { u n i f } }$   
14: $\begin{array} { r } { i _ { k } \gets \arg \operatorname* { m a x } _ { i \in \tau _ { k } } [ \widetilde { a } _ { i } ^ { \top } \omega _ { k } - \widetilde { b } _ { i } ] _ { + } } \end{array}$   
15: $\mathbf { i f } \widetilde { a } _ { i _ { k } } ^ { \top } \omega _ { k } > \widetilde { b } _ { i _ { k } }$ then   
16: $\begin{array} { r } { \omega _ { k + 1 } = \omega _ { k } - \alpha \frac { \widetilde { a } _ { i _ { k } } ^ { \top } \omega _ { k } - \widetilde { b } _ { i _ { k } } } { \| \widetilde { a } _ { i _ { k } } \| _ { 2 } ^ { 2 } } \widetilde { a } _ { i _ { k } } , \qquad \alpha \in ( 0 , 2 ) . } \end{array}$   
17: else   
18: ω<sub>k+1</sub> ← ω<sub>k</sub>   
19: end if   
20: end for   
21: x<sub>K</sub> ← x<sub>eq</sub> + Nω<sub>K</sub>   
22:   
23: return $x _ { K }$

## A.2 Accelerated Sampling by Active-Set Prediction

Theorem 1 (Global Convergence and Robustness). Provided that the mixing ratio satisfies $\rho < 1$ the proposed algorithm maintains global linear convergence in expectation toward thefeasible set $P .$

Proof. The AT-SKM algorithm iterates on the null-space coefficients ω. Let $\Omega = \{ \omega \in \mathbb { R } ^ { r } \mid \tilde { A } \omega \leq \tilde { b } \}$ be the feasible set in the null-space representation, where $\tilde { A } = A N$ and $\tilde { b } = b - A x _ { e q }$ . Let $\omega ^ { * }$ be the projection of the current iterate $\omega _ { k }$ onto Ω. We define the error vector as $e _ { k } = \omega _ { k } - \omega ^ { * }$

Consider the k-th iteration. Let $\begin{array} { r } { g _ { j } ( \omega _ { k } ) = \frac { [ \tilde { a } _ { j } ^ { \top } \omega _ { k } - \tilde { b } _ { j } ] _ { + } ^ { 2 } } { \| \tilde { a } _ { j } \| ^ { 2 } } = [ \tilde { a } _ { j } ^ { \top } \omega _ { k } - \tilde { b } _ { j } ] _ { + } ^ { 2 } } \end{array}$ denote the squared normalized violation of the row-normalized j-th constraint. The algorithm selects the active constraint index $i _ { k }$ from a sampled batch $\boldsymbol { S } _ { k }$ using the greedy criterion:

$$
i _ { k } = \underset { j \in { \cal S } _ { k } } { \arg \operatorname* { m a x } } g _ { j } ( \omega _ { k } ) .\tag{13}
$$

The standard Kaczmarz update with step size $\alpha \in ( 0 , 2 )$ yields the following error contraction recurrence [32]:

$$
\begin{array} { r } { \| e _ { k + 1 } \| ^ { 2 } \leq \| e _ { k } \| ^ { 2 } - \alpha ( 2 - \alpha ) g _ { i _ { k } } ( \omega _ { k } ) . } \end{array}\tag{14}
$$

The hybrid sampling strategy ensures that the batch $S _ { k }$ contains a subset $\tau _ { \mathrm { u n i f } }$ of size $\beta _ { \mathrm { u n i f } } \geq 1 .$ , which is sampled uniformly from all m constraints, which is guaranteed by $\rho < 1$ . The maximum violation in the full batch is bounded below by the maximum violation in the uniform subset, which is further bounded below by the average violation in that subset:

$$
g _ { i _ { k } } ( \omega _ { k } ) = \operatorname* { m a x } _ { j \in \mathcal { S } _ { k } } g _ { j } ( \omega _ { k } ) \ge \operatorname* { m a x } _ { j \in \tau _ { \mathrm { u n i f } } } g _ { j } ( \omega _ { k } ) \ge \frac { 1 } { \beta _ { \mathrm { u n i f } } } \sum _ { j \in \tau _ { \mathrm { u n i f } } } g _ { j } ( \omega _ { k } ) .\tag{15}
$$

We take the expectation with respect to the random sampling of the batch. Since $\tau _ { \mathrm { u n i f } }$ is drawn uniformly from $\{ 1 , \ldots , m \}$ , the probability that any specific constraint l is included in $\tau _ { \mathrm { u n i f } }$ is $\beta _ { \mathrm { u n i f } } / m$ . By the linearity of expectation:

$$
\begin{array} { r l } & { \mathbb { E } \big [ g _ { i _ { k } } ( \omega _ { k } ) \big ] \geq \mathbb { E } \left[ \displaystyle \frac { 1 } { \beta _ { \mathrm { u n i f } } } \sum _ { j \in \tau _ { \mathrm { u n f } } } g _ { j } ( \omega _ { k } ) \right] } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { \beta _ { \mathrm { u n i f } } } \sum _ { l = 1 } ^ { m } \frac { \beta _ { \mathrm { u n i f } } } { m } g _ { l } ( \omega _ { k } ) } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { m } \sum _ { l = 1 } ^ { m } g _ { l } ( \omega _ { k } ) . } \end{array}\tag{16}
$$

The summation term corresponds to the squared norm of the residual vector $\| [ \tilde { A } \omega _ { k } - \tilde { b } ] _ { + } \| ^ { 2 }$ . By the global Hoffman error bound theorem, there exists a constant $\mathcal { H } ( \tilde { \boldsymbol { A } } )$ such that:

$$
\sum _ { l = 1 } ^ { m } g _ { l } ( \omega _ { k } ) = \| [ \tilde { A } \omega _ { k } - \tilde { b } ] _ { + } \| ^ { 2 } \geq \frac { 1 } { \mathcal { H } ^ { 2 } ( \tilde { A } ) } \| e _ { k } \| ^ { 2 } .\tag{17}
$$

Substituting this back into the expectation of the error recurrence:

$$
\mathbb { E } [ \| e _ { k + 1 } \| ^ { 2 } ] \le \left( 1 - \frac { \alpha ( 2 - \alpha ) } { m \cdot \mathcal { H } ^ { 2 } ( \tilde { A } ) } \right) \| e _ { k } \| ^ { 2 } = \left( 1 - \frac { 1 } { m \cdot \mathcal { H } ^ { 2 } ( \tilde { A } ) } \right) \| e _ { k } \| ^ { 2 } .\tag{18}
$$

Under $\alpha = 1 \in ( 0 , 2 )$ and $\mathcal { H } ( \tilde { A } ) < \infty$ , the algorithm converges linearly in expectation.

Theorem 2 (Efficiency Gain via Dimension Reduction). Consider the iterate in the local linear regime from Assumption 1 with row-normalized constraints. Comparing uniform sampling over $\{ 1 , \ldots , m \}$ and our proposed uniform sampling over the predicted set $\hat { \mathcal { A } } ,$ under Assumption 2, the expected one-step error contraction rates are:

• Standard SKM (Uniform on m):

$$
\mathbb { E } [ \| \mathbf { e } _ { k + 1 } \| ^ { 2 } ] \leq \left( 1 - { \frac { 1 } { m \cdot \mathcal { H } ^ { 2 } ( A ) } } \right) \| \mathbf { e } _ { k } \| ^ { 2 }\tag{7}
$$

• Accelerated SKM (Uniform on $\hat { \mathcal { A } } )$ :

$$
\mathbb { E } [ \| \mathbf { e } _ { k + 1 } \| ^ { 2 } ] \leq \left( 1 - \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } ( A _ { { A ^ { * } } } ) } { | \hat { \mathcal { A } } | } \right) \| \mathbf { e } _ { k } \| ^ { 2 }\tag{8}
$$

where $\mathcal { H } ( A )$ is the global Hoffman constant, and $\sigma _ { \mathrm { m i n } } ( A _ { \mathcal { A } ^ { \ast } } )$ denotes the smallest non-zero singular value ofthe true active submatrix.

Proof. Let $\mathbf { e } _ { k } = \mathbf { x } _ { k } - \mathbf { x } ^ { * }$ . Given row normalization, the Kaczmarz update with hinge loss implies $\| \mathbf { e } _ { k + 1 } \| ^ { 2 } = \| \mathbf { e } _ { k } \| ^ { 2 } - [ \mathbf { a } _ { i } ^ { \top } \mathbf { x } _ { k } - b _ { i } ] _ { + } ^ { 2 }$

Taking the expectation over the sampled index i:

$$
\mathbb { E } [ \| \mathbf { e } _ { k + 1 } \| ^ { 2 } ] = \| \mathbf { e } _ { k } \| ^ { 2 } - \sum _ { i \in \mathcal { S } } p _ { i } [ \mathbf { a } _ { i } ^ { \top } \mathbf { x } _ { k } - b _ { i } ] _ { + } ^ { 2 } .\tag{19}
$$

For Standard SKM, $\mathcal { S } = \{ 1 , \dots , m \}$ and $p _ { i } = 1 / m$ . Using the global Hoffman bound on the residual vector $[ A { \bf x } _ { k } - { \bf b } ] _ { + }$ , we have $\begin{array} { r } { \frac { 1 } { m } \| [ A \dot { \mathbf { x } } _ { k } - \mathbf { b } ] _ { + } ^ { \prime } \| ^ { 2 } \geq \frac { \smile } { m \cdot \mathcal { H } ^ { 2 } ( A ) } \| \mathbf { \bar { e } } _ { k } \| ^ { 2 } [ 4 1 ] } \end{array}$

For Accelerated SKM, $s = { \hat { A } }$ and $p _ { i } = 1 / | \hat { A } |$ . Crucially, under Assumption 1, for any inactive constraint $i \notin { \mathcal { A } } ^ { * }$ , we have $\mathbf { a } _ { i } ^ { \top } \mathbf { x } _ { k } < b _ { i }$ , implying $[ { \bf a } _ { i } ^ { \top } { \bf x } _ { k } - b _ { i } ] _ { + } = 0$ . Thus, the summation vanishes for all inactive constraints and retains only the contribution from the true active set $\ b { A } ^ { * }$ . For $i \in \mathcal { A } ^ { * }$ since $\mathbf { a } _ { i } ^ { \top } \mathbf { x } ^ { * } = b _ { i }$ , the term simplifies to $[ \bar { \mathbf { a } } _ { i } ^ { \top } \mathbf { x } _ { k } - b _ { i } ] _ { + } ^ { 2 } = \lVert \mathbf { a } _ { i } ^ { \top } \mathbf { e } _ { k } \rVert ^ { 2 } .$

$$
\begin{array} { r } { \displaystyle \sum _ { i \in \hat { \mathcal { A } } } \frac { 1 } { | \hat { \mathcal { A } } | } [ \mathbf { a } _ { i } ^ { \top } \mathbf { x } _ { k } - b _ { i } ] _ { + } ^ { 2 } = \frac { 1 } { | \hat { \mathcal { A } } | } \displaystyle \sum _ { i \in \mathcal { A } ^ { * } } \| \mathbf { a } _ { i } ^ { \top } \mathbf { e } _ { k } \| ^ { 2 } } \\ { = \frac { 1 } { | \hat { \mathcal { A } } | } \| \boldsymbol { A } _ { \mathcal { A } ^ { * } } \mathbf { e } _ { k } \| ^ { 2 } . } \end{array}\tag{20}
$$

Finally we can apply the spectral bound $\| A _ { A ^ { * } } \mathbf { e } _ { k } \| ^ { 2 } \geq \sigma _ { \operatorname* { m i n } } ^ { 2 } ( A _ { A ^ { * } } ) \| \mathbf { e } _ { k } \| ^ { 2 }$ to yield the accelerated rate. □

## B Computational Complexity of Proposed Cholesky Update Method

## B.1 Implementation Details of the Cholesky Update

The key of our proposed Cholesky update framework is to utilize the properties of low-rank perturbations, allowing us to reuse updated Cholesky factors. This avoids the high computational cost of Singular Value Decomposition while efficiently maintaining both the projection operator and the null-space basis.

Let $\mathbf { G } = \mathbf { C } \mathbf { C } ^ { \top }$ be the original Gram matrix. The perturbed Gram matrix $\tilde { \mathbf { G } }$ is derived as:

$$
\begin{array} { r l } & { \tilde { \mathbf { G } } = ( \mathbf { C } + \mathbf { u v } ^ { \top } ) ( \mathbf { C } + \mathbf { u v } ^ { \top } ) ^ { \top } } \\ & { \quad = \mathbf { C C } ^ { \top } + \mathbf { C v } \mathbf { u } ^ { \top } + \mathbf { u v } ^ { \top } \mathbf { C } ^ { \top } + \mathbf { u } ( \mathbf { v } ^ { \top } \mathbf { v } ) \mathbf { u } ^ { \top } } \\ & { \quad = \mathbf { G } + ( \mathbf { C v } ) \mathbf { u } ^ { \top } + \mathbf { u } ( \mathbf { C v } ) ^ { \top } + \| \mathbf { v } \| ^ { 2 } \mathbf { u } \mathbf { u } ^ { \top } } \end{array}\tag{21}
$$

Let $\mathbf { h } = \mathbf { C } \mathbf { v }$ and scalar $\lambda = \| \mathbf { v } \| ^ { 2 }$ , we rewrite the expression as:

$$
\tilde { \mathbf { G } } = \mathbf { G } + \mathbf { h } \mathbf { u } ^ { \top } + \mathbf { u } \mathbf { h } ^ { \top } + \lambda \mathbf { u } \mathbf { u } ^ { \top }\tag{22}
$$

We define an auxiliary vector $\begin{array} { r } { \mathbf { m } = \mathbf { h } + \frac { \lambda } { 2 } \mathbf { \imath } } \end{array}$ u, then we have:

$$
\mathbf { h } \mathbf { u } ^ { \top } + \mathbf { u } \mathbf { h } ^ { \top } + \lambda \mathbf { u } \mathbf { u } ^ { \top } = \mathbf { u } \mathbf { m } ^ { \top } + \mathbf { m } \mathbf { u } ^ { \top } = { \frac { 1 } { 2 } } ( \mathbf { u } + \mathbf { m } ) ( \mathbf { u } + \mathbf { m } ) ^ { \top } - { \frac { 1 } { 2 } } ( \mathbf { u } - \mathbf { m } ) ( \mathbf { u } - \mathbf { m } ) ^ { \top }\tag{23}
$$

Substituting this back into the Gram matrix expression yields:

$$
\tilde { \mathbf { G } } = \mathbf { G } + \underbrace { \frac { 1 } { 2 } ( \mathbf { u } + \mathbf { m } ) ( \mathbf { u } + \mathbf { m } ) } _ { \mathrm { U p d a t e } } ^ { \tilde { \mathbf { G } } = } - \underbrace { \frac { 1 } { 2 } ( \mathbf { u } - \mathbf { m } ) ( \mathbf { u } - \mathbf { m } ) ^ { \top } } _ { \mathrm { D o w n d a t e } }\tag{24}
$$

Consequently, we do not need to explicitly reconstruct $\tilde { \mathbf { G } }$ . Using standard chol\_update and chol\_downdate routines, we apply one rank-1 updates and one rank-1 downdate to the original factor $\mathbf { L } ,$ directly obtaining $\tilde { \bf L }$ such that $\tilde { \bf L } \tilde { \bf L } ^ { \top } = \tilde { \bf G }$

With the updated factor $\tilde { \mathbf { L } } ,$ we define the constraint residual $\mathbf { r } = \tilde { \mathbf { C } } \mathbf { x } _ { 0 } - \mathbf { d }$ . The projection formula is given by:

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { e q } } = \mathbf { x } _ { 0 } - \tilde { \mathbf { C } } ^ { \top } ( ( \tilde { \mathbf { L } } ^ { \top } ) ^ { - 1 } ( \tilde { \mathbf { L } } ^ { - 1 } \mathbf { r } ) ) } \end{array}\tag{25}
$$

Using $\tilde { \mathbf { L } } ,$ we solve this via two triangular substitutions:

1. Forward Substitution: $\tilde { \mathbf { L } } \mathbf { k } = \mathbf { r } ;$

2. Backward Substitution: $\tilde { \mathbf { L } } ^ { \top } \mathbf { y } = \mathbf { k }$

The final projected solution is obtained by:

$$
\tilde { \mathbf { x } } _ { \mathrm { e q } } = \mathbf { x } _ { 0 } - \tilde { \mathbf { C } } ^ { \top } \mathbf { y }\tag{26}
$$

Next, we construct the new null-space basis $\tilde { \mathbf { Z } }$ such that $\tilde { \mathbf { C } } \tilde { \mathbf { Z } } = \mathbf { 0 } ,$ , employing a rotation-correction strategy. First, we compute the projection of the perturbation onto the original null space, $\mathbf { w } = \mathbf { Z } ^ { \top } \mathbf { v }$ To concentrate the perturbation, we construct a Householder reflection $\begin{array} { r } { \mathbf { H } = \mathbf { I } - 2 \frac { \mathbf { q } \mathbf { q } ^ { \mathrm { ~ \tiny ~ 1 ~ } } } { | | \mathbf { q } | | ^ { 2 } } } \end{array}$ with $\mathbf { q } =$ $\mathbf { w } - \| \mathbf { w } \| \mathbf { e } _ { 1 \cdot } \mathsf { A }$ pplying this to the original basis yields the transition basis [42]:

$$
\mathbf { Z } ^ { \prime } = \mathbf { Z } \mathbf { H } = \left[ \mathbf { z } _ { 1 } ^ { \prime } , \mathbf { z } _ { 2 } ^ { \prime } , \ldots , \mathbf { z } _ { n - m } ^ { \prime } \right]\tag{27}
$$

By properties of the Householder transformation, columns 2 through $n - m$ of $\mathbf { Z } ^ { \prime }$ satisfy $\mathbf { v } ^ { \top } \mathbf { z } _ { j } ^ { \prime } = 0$ Thus, for $j \geq 2 $

$$
\tilde { \mathbf { C } } \mathbf { z } _ { j } ^ { \prime } = ( \mathbf { C } + \mathbf { u } \mathbf { v } ^ { \top } ) \mathbf { z } _ { j } ^ { \prime } = \mathbf { C } \mathbf { z } _ { j } ^ { \prime } + \mathbf { u } ( \mathbf { v } ^ { \top } \mathbf { z } _ { j } ^ { \prime } ) = \mathbf { 0 }\tag{28}
$$

This indicates that $\mathbf { Z } _ { : , 2 : \mathrm { e n d } } ^ { \prime }$ directly forms the majority of the new null space. The first column $\mathbf { z } _ { 1 } ^ { \prime }$ produces a non-zero residual $\tilde { \mathbf { C } } \mathbf { z } _ { 1 } ^ { \prime } = \mathbf { u } ( \mathbf { v } ^ { \top } \mathbf { z } _ { 1 } ^ { \prime } )$ . To eliminate this, we seek a vector p satisfying $\tilde { \mathbf { C } } \mathbf { p } = \mathbf { u }$ . Using the maintained $\tilde { \mathbf { L } } _ { : }$ , we solve $\tilde { \mathbf { C } } \tilde { \mathbf { C } } ^ { \top } \mathbf { y } = \mathbf { u }$ via triangular substitution and set $\mathbf { p } = \tilde { \mathbf { C } } ^ { \top } \mathbf { y }$ Finally, adding the correction term to $\mathbf { z } _ { 1 } ^ { \prime }$ yields the complete new basis:

$$
\tilde { \mathbf { Z } } = \left[ \mathbf { z } _ { 1 } ^ { \prime } - \mathbf { v } ^ { \top } \mathbf { z } _ { 1 } ^ { \prime } \mathbf { p } , \quad \mathbf { Z } _ { : , 2 : \mathrm { e n d } } ^ { \prime } \right]\tag{29}
$$

## B.2 Complexity Analysis

Let $\mathbf { C } \in \mathbb { R } ^ { n _ { \mathrm { e q } } \times n }$ denote the row full-rank equality-constraint matrix, and let $r = n - n _ { \mathrm { e q } }$ be the dimension of its null space. The $\operatorname { s v p }$ approach requires a complete factorization of the constraint matrix, incurring a time complexity of $\bar { \mathcal { O } } ( { n ^ { 3 } } )$ . In contrast, the proposed framework exploits the lowrank nature of topological perturbations, effectively decoupling the high-dimensional problem into sequential vector-level operations. The computational cost of one complete iteration is decomposed as follows:

• Factor Maintenance: We perform one rank-1 update and one rank-1 downdate on the Cholesky factor L of $\mathbf { G } = \bar { \mathbf { C } } \mathbf { C } ^ { \top } \in \mathbb { R } ^ { n _ { \mathrm { e q } } \times n _ { \mathrm { e q } } }$ . This step operates purely on the triangular factor, reducing the complexity to $\mathcal { O } ( n _ { \mathrm { e q } } ^ { 2 } )$ , where $n _ { \mathrm { e q } }$ is the number of equality constraints.

• Equality Projection: Computing the projected solution involves matrix-vector multiplica-<sub>tions and triangular substitutions using the updated L</sub>˜<sub>. The dominant cost is the matrix-vector</sub> product, scaling as $\mathcal { O } ( n _ { \mathrm { e q } } \cdot n )$

• Null Space Update: The subspace rotation involves Householder transformations on the basis matrix $\bar { \mathbf Z } \in \mathbb R ^ { n \times r }$ , costing $\mathcal { O } ( n \cdot r )$ , and the correction requires solving for vector p, adding a $\mathcal { O } ( n _ { \mathrm { e q } } \cdot n )$ cost regardless of the null-space dimension.

Since both $n _ { \mathrm { e q } }$ and $r$ are bounded by $n ,$ the overall asymptotic complexity of our framework is $O ( n ^ { 2 } )$ . This represents a theoretical speedup of order n compared to standard SVD-based methods, transforming the expensive topological adaptation into a computationally lightweight process which is suitable for real-time applications.

## B.3 Detailed Analysis for Results on RGGs

To validate the theoretical complexity analysis, we performed tests on Random Geometric Graphs of varying scales. Table 2 presents the overall performance comparison.

Table 2 highlights the significant scalability gap between the SVD approach and our proposed Cholesky Update method. In the small-scale group from 100 to 1000, we observe empirical complexities of $\dot { \mathcal { O } } ( n ^ { 1 . 9 9 } )$ for SVD and $\mathcal { O } ( n ^ { 1 . 4 6 } )$ for the Cholesky Update. These growth rates are much lower than their theoretical asymptotic bounds, which can be attributed to specific optimizations in underlying linear algebra libraries that maximize cache efficiency and instruction throughput for smaller matrices. However, even with these library-level enhancements benefiting the baseline, ou method maintains superior performance, delivering substantial speedups ranging from 7.36 to 23.35 .

As the number of nodes increases from 1500 to 5000, the SVD computation time exhibits a supercubic growth trend with a fitted exponent of 3.49 in the large-scale group, reflecting the heavy burden of computation and memory data movement. In contrast, the proposed Cholesky Update maintain a near-quadratic growth rate of approximately 2.28, which closely matches the theoretical lower bound. A remaining limitation is that large-scale matrix operations can become memory-bound, as the increasing matrix size and data movement overhead can partially offset the arithmetic savings and push the observed scaling above the ideal quadratic rate. Nevertheless, the method still maintains a clear efficiency advantage over SVD, achieving over 168 times acceleration at 5000 nodes.

Table 2: Computation time comparison between SVD and Cholesky Update on random geometric graphs. The reported complexity values are empirical scaling exponents obtained by least-squares linear regression in log–log space.
<table><tr><td colspan="5">SMALL-SCALE (N = 100 ~ 1000)</td><td colspan="5">LARGE-SCALE (N = 1500 ~ 5000)</td></tr><tr><td>N</td><td>K</td><td>SVD (MS)</td><td>CHOL. (MS)</td><td>SPEEDUP</td><td>N</td><td>K</td><td>SVD (MS)</td><td>CHOL. (MS)</td><td>SPEEDUP</td></tr><tr><td>100</td><td>10</td><td>0.874</td><td>0.119</td><td>7.36×</td><td>1500</td><td>38</td><td>307.441</td><td>8.187</td><td>37.55×</td></tr><tr><td>200</td><td>14</td><td>2.906</td><td>0.290</td><td>10.02×</td><td>2000</td><td>44</td><td>995.480</td><td>14.754</td><td>67.47×</td></tr><tr><td>400</td><td>20</td><td>11.106</td><td>0.575</td><td>19.33×</td><td>2500</td><td>50</td><td>2276.658</td><td>23.450</td><td>97.09×</td></tr><tr><td>800</td><td>28</td><td>49.833</td><td>2.257</td><td>22.08×</td><td>3000</td><td>54</td><td>4595.625</td><td>40.505</td><td>113.46×</td></tr><tr><td>1000</td><td>31</td><td>83.666</td><td>3.583</td><td>23.35×</td><td>5000</td><td>70</td><td>20596.582</td><td>122.586</td><td>168.02×</td></tr><tr><td>COMPLEX.</td><td>=</td><td> $\mathcal { O } ( n ^ { 1 . 9 9 } )$ </td><td> $\mathcal { O } ( n ^ { 1 . 4 6 } )$ </td><td></td><td>COMPLEX.</td><td>-</td><td> $\mathcal { O } ( n ^ { 3 . 4 9 } )$ </td><td> $\mathcal { O } ( n ^ { 2 . 2 8 } )$ </td><td></td></tr></table>

## B.4 Numerical Stability under Higher-Rank Contingencies

Sequential Cholesky updates may accumulate numerical errors under finite precision, especially when multiple topology changes are applied consecutively. This issue becomes more visible for higher-rank contingencies and larger systems, where the update and downdate vectors can contain entries with very different magnitudes. To address this numerical concern, we use an adaptive rescaling strategy that preserves the exact low-rank perturbation while improving the conditioning of intermediate update vectors.

For a rank-one perturbation $\Delta \mathbf { C } = \mathbf { u } \mathbf { v } ^ { \top }$ , the representation is not unique. For any scalar $\alpha > 0$ , we have

$$
\Delta \mathbf { C } = \mathbf { u } \mathbf { v } ^ { \top } = ( \alpha \mathbf { u } ) ( \mathbf { v } / \alpha ) ^ { \top } .\tag{30}
$$

In the Gram-matrix update derivation in Appendix B.1, we define $\mathbf { h } \ = \ \mathbf { C } \mathbf { v } , \ \lambda \ = \ \| \mathbf { v } \| _ { 2 } ^ { 2 }$ , and $\mathbf { m } = \mathbf { h } + \textstyle { \frac { \lambda } { 2 } } \mathbf { u }$ . After applying the above rescaling, the corresponding pair becomes αu and m/α. Therefore, the exact perturbation remains unchanged, but the numerical magnitudes of the two vectors entering the Cholesky update and downdate can be balanced. In practice, we choose

$$
\alpha = \sqrt { \frac { \| { \bf m } \| _ { 2 } } { \| { \bf u } \| _ { 2 } } } ,\tag{31}
$$

which equalizes the norms of αu and m $/ \alpha$ . This adaptive rescaling reduces extreme intermediate values, for example, on the IEEE 300-bus system, the largest intermediate magnitude is reduced from roughly $1 0 ^ { 1 0 }$ to roughly 10<sup>5</sup>.

We evaluate both the original and adaptive Cholesky updates under fp32 and fp64 precision. The experiments cover all N-1 and N-2 contingencies, together with a subset of $N { - } 3$ contingencies, on the IEEE 57-, 118-, and 300-bus systems. A scenario is counted as a failure if the sequential Cholesky update cannot be completed in the corresponding floating-point precision.

As shown in Table 3, the original sequential update is highly unstable in fp32 and becomes almost unusable for N-1 and N-2 contingencies. Although fp64 substantially improves stability, the original update still produces a small number of failures on the IEEE 300-bus system. In contrast, the adaptive rescaling removes all observed fp64 failures across all evaluated systems and contingency levels. Under fp32, it also eliminates all failures on the 57- and 118-bus systems and leaves only a small failure rate on the larger 300-bus system. These results indicate that the numerical stability issue of sequential Cholesky updates can be effectively controlled by adaptive rescaling, while the exact low-rank perturbation and the theoretical update structure remain unchanged.

Table 3: Failure counts of original and adaptive Cholesky updates under fp32 and fp64 precision. The percentage in parentheses denotes the failure rate among evaluated contingency scenarios.
<table><tr><td rowspan="2">SYSTEM</td><td rowspan="2">CONTINGENCY</td><td rowspan="2"># SCENARIOS</td><td colspan="2">FP32 FAILURES</td><td colspan="2">FP64 FAILURES</td></tr><tr><td>ORIGINAL</td><td>ADAPTIVE</td><td>ORIGINAL</td><td>ADAPTIVE</td></tr><tr><td>57-BUS</td><td>N-1</td><td>79</td><td>68 (86.1%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td></tr><tr><td>57-BUS</td><td>N-2</td><td>3024</td><td>2973 (98.3%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td></tr><tr><td>57-BUS</td><td>N-3</td><td>6048</td><td></td><td>0 (0.0%)</td><td></td><td>0 (0.0%)</td></tr><tr><td>118-BUS</td><td>N-1</td><td>177</td><td>174 (98.3%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td></tr><tr><td>118-BUS</td><td>N-2</td><td>15502</td><td>15501 (99.9%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td></tr><tr><td>118-BUS</td><td>N-3</td><td>31004</td><td></td><td>0 (0.0%)</td><td></td><td>0 (0.0%)</td></tr><tr><td>300-BUS</td><td>N-1</td><td>322</td><td>303 (94.1%)</td><td>1 (0.3%)</td><td>2 (0.6%)</td><td>0 (0.0%)</td></tr><tr><td>300-BUS</td><td>N-2</td><td>51559</td><td>51405 (99.7%)</td><td>279 (0.5%)</td><td>641 (1.2%)</td><td>0 (0.0%)</td></tr><tr><td>300-BUS</td><td>N-3</td><td>103118</td><td></td><td>558 (0.5%)</td><td></td><td>0 (0.0%)</td></tr></table>

## C Minimum-Cost Gas Transport Problem

We further evaluate the proposed method on a minimum-cost gas transport problem using the GasLib-135 benchmark [43]. Given a gas network $G = ( V , E )$ , each node $v \in V$ represents a supply or demand point, and each edge $e \in E$ is associated with capacity limits and transportation costs. The goal is to find a minimum-cost flow allocation while satisfying nodal flow-balance constraints and edge-capacity constraints. Each test instance corresponds to an operating scenario with different demand levels and possible topology changes, such as edge removals or failure cases. We evaluate both seen and unseen scenarios to assess generalization across network conditions. The optimization problem is formulated as follows:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { S } } \displaystyle \operatorname* { m a x } _ { s \in \mathcal { S } } \left( \displaystyle \sum _ { \ell \in \mathcal { L } ^ { ( k ) } } \sum _ { \ell \in \mathcal { L } ^ { ( k ) } } \rho _ { \ell } \Big | f _ { \ell } ^ { ( k ) } \Big | \right. } & { } \\ { \displaystyle \mathrm { s . t . } \ } & { \displaystyle \sum _ { s \in \mathcal { S } } g _ { s } ^ { ( k ) } + \displaystyle \sum _ { \ell \in \mathcal { L } ^ { ( k ) } } A _ { \ell \ell } ^ { ( k ) } f _ { \ell } ^ { ( k ) } = d _ { v } , \quad \forall v \in \mathcal { V } \setminus \{ v _ { \mathrm { r e f } } \} , } \\ & { \displaystyle \sum _ { s \in \mathcal { S } } g _ { s } ^ { ( k ) } = \sum _ { \ell \in \mathcal { R } } d _ { v } , } \\ & { \displaystyle \left. 0 \leq g _ { s } ^ { ( k ) } \leq \overline { { g } } _ { s } , \quad \forall s \in \mathcal { S } , \right. } \\ & { \displaystyle - \overline { { f } } _ { \ell } \leq f _ { \ell } ^ { ( k ) } \leq \overline { { f } } _ { \ell } , \quad \forall \ell \in \mathcal { L } ^ { ( k ) } , } \\ & { \displaystyle k \in \{ 0 \} \cup \mathcal { K } , } \\ & { \displaystyle \mathcal { L } ^ { ( 0 ) } = \mathcal { L } , \quad \mathcal { L } ^ { ( k ) } = \mathcal { L } \setminus \{ k \} , \quad k \in \mathcal { K } . } \end{array}\tag{32}
$$

where

Table 4: HGNN-only prediction quality on the minimum-cost gas transport problem. Metrics except forward time are formatted as “Seen / Unseen”.
<table><tr><td>METHOD</td><td>FORWARD TIME (MS)</td><td>MAX EQ. VIOL.</td><td>MAX INEQ. VIOL.</td><td>OPT. GAP (%)</td></tr><tr><td>HGNN</td><td> $3 . 8 1 0 \pm 2 . 0 7 4$ </td><td> $4 7 3 . 6 / 2 3 3 7 . 0$ </td><td> $1 0 0 . 5 / 4 1 6 . 8 $ </td><td>0.008/0.009</td></tr></table>

Table 5: SKM-layer performance on the minimum-cost gas transport problem. The optimality gap is reported as mean value in the format “Seen / Unseen”.
<table><tr><td>METHOD</td><td>EQ. PROJ.</td><td> $\mathrm { E Q . } \ \mathrm { S P D . }$ </td><td>INEQ. ITER.</td><td>INEQ. TIME</td><td>INEQ. SPD.</td><td>SKM TIME</td><td>SKM SPD.</td><td>OPT. GAP (%)</td></tr><tr><td>T-SKM</td><td> $2 . 1 9 \pm 1 . 0 6$ </td><td>1.00×</td><td>15.21 (128)</td><td> $0 . 9 6 \pm 1 . 2 2$ </td><td>1.00×</td><td>3.15</td><td>1.00×</td><td>0.006/0.019</td></tr><tr><td>AT-SKM</td><td> $0 . 5 3 \pm 0 . 3 5$ </td><td>4.13×</td><td>5.33 (78)</td><td> $0 . 3 7 \pm 0 . 7 1$ </td><td>2.59×</td><td>0.90</td><td>3.50×</td><td>0.006/0.019</td></tr></table>

Table 4 shows that the HGNN-only predictor can obtain small mean optimality gaps, but it suffers from substantial worst-case equality and inequality constraint violations, especially on unseen scenarios. This indicates that direct neural prediction alone is insufficient for safety-critical constrained optimization.

In contrast, Table 5 shows that both T-SKM-Net and AT-SKM-Net achieve zero equality and inequality constraint violations on both seen and unseen scenarios. Compared with T-SKM-Net, AT-SKM-Net reduces the equality projection time from 2.19 ms to 0.53 ms, yielding a 4.13 speedup, and reduces the inequality iteration time from 0.96 ms to 0.37 ms, yielding a 2.59 speedup. Overall, AT-SKM Net reduces the total SKM computation time from 3.15 ms to 0.90 ms, corresponding to a 3.50 acceleration while preserving strict feasibility and maintaining comparable mean optimality gaps.

## D Experimental Settings and Additional Results for N-1 Security-Constrained DC-OPF

## D.1 Mathematical Formulation of N-1 Security-Constrained DC-OPF

The N-1 Security-Constrained DC Optimal Power Flow problem aims to determine the optimal generation dispatch that minimizes total operating costs while ensuring system feasibility under both normal conditions and any single transmission line outage. Let denote the set of all considered scenarios, where $k = 0$ represents the base case and each remaining $k \in \mathcal { K }$ corresponds to a specific line failure contingency. Under a given scenario $k ,$ the network topology changes accordingly; we define $\mathcal { L } ^ { ( k ) }$ as the set of remaining active lines and $\mathcal { N } ^ { ( k ) } ( i )$ as the set of neighboring buses for bus i in that topology. The optimization problem is formulated as follows:

$$
\begin{array} { r l } { \underset { { \bf { P } } _ { \sigma } , \delta } { \operatorname* { m i n } } } & { \displaystyle \sum _ { i \in \mathcal { G } } \bigg ( \frac { 1 } { 2 } c _ { i } P _ { G , i } ^ { 2 } + b _ { i } P _ { G , i } \bigg ) } \\ { \mathrm { s . t . } } & { P _ { G , i } - P _ { D , i } = \displaystyle \sum _ { j \in \mathcal { N } ^ { ( k ) } ( i ) } B _ { i j } ( \delta _ { i } ^ { ( k ) } - \delta _ { j } ^ { ( k ) } ) , \quad \forall i \in \mathcal { B } , \forall k \in \{ 0 \} \cup \mathcal { K } } \\ & { P _ { G , i } ^ { \operatorname* { m i n } } \le P _ { G , i } \le P _ { G , i } ^ { \operatorname* { m a x } } , \quad \forall i \in \mathcal { G } } \\ & { - P _ { i j } ^ { \operatorname* { m a x } } \le B _ { i j } ( \delta _ { i } ^ { ( k ) } - \delta _ { j } ^ { ( k ) } ) \le P _ { i j } ^ { \operatorname* { m a x } } , \quad \forall ( i , j ) \in \mathcal { L } ^ { ( k ) } , \forall k \in \{ 0 \} \cup \mathcal { K } } \end{array}\tag{33}
$$

where the sets $B , { \mathcal { G } } \subseteq B ,$ , and  represent buses, generators, and transmission lines, respectively. The decision variables consist of the active power outputs $\mathbf { P } _ { \mathcal { G } } = \{ P _ { G , i } \} _ { i \in \mathcal { G } }$ and the voltage phase angles $\delta ^ { ( k ) } = \{ \delta _ { i } ^ { ( k ) } \} _ { i \in \mathcal { B } }$ for each scenario. It is important to note that the generation dispatch $P _ { G }$ is preventive and shared across all scenarios, whereas the system state $\delta ^ { ( k ) }$ adapts to each specific topology. In the objective function, $c _ { i }$ and $b _ { i }$ denote the quadratic and linear cost coefficients for generator i. The parameter $P _ { D , : }$ <sub>i</sub> represents the active power demand at bus i, and $B _ { i j }$ is the line susceptance defined as a positive value $1 / X _ { i j }$ . For non-generator buses $i \in B \backslash { \mathcal { G } }$ , we set $P _ { G , i } = 0$ The terms $P _ { G , i } ^ { \operatorname* { m i n } / \operatorname* { m a x } }$ and $P _ { i j } ^ { \mathrm { m a x } }$ define the generator capacity limits and transmission line thermal limits, respectively.

## D.2 N-1 Security-Constrainted DC-OPF Dataset Generation

To validate the effectiveness of the proposed method, we employ the IEEE 57, 118, and 300-bus systems as experimental benchmarks. In constructing the N-1 contingency dataset, we first screen all potential single-line outage scenarios, retaining only those that are physically feasible. Here, a feasible scenario is defined as one where the post-contingency network topology remains connected without forming islands, and the OPF solver successfully converges to a feasible solution. Table 6 summarizes the statistics for each test system, including the number of buses, total branches, and the count of valid N-1 contingencies with the original scenario after screening.

For each selected topological scenario, we generate 100 independent operating instances by introducing random perturbations to the nodal loads. Specifically, the load vectors are obtained by uniformly scaling the baseline load profile:

$$
{ \cal P } _ { D } ^ { ( i ) } = \alpha _ { i } \cdot { \cal P } _ { D } ^ { \mathrm { b a s e } } , \quad \alpha _ { i } \sim \mathcal { U } ( 0 . 8 , 1 . 2 ) ,\tag{34}
$$

where $P _ { D } ^ { \mathrm { b a s e } }$ represents the baseline load values, and $\alpha _ { i }$ is a random scaling factor drawn from a uniform distribution over the interval [0.8, 1.2]. This generation procedure ensures that the dataset encompasses a wide range of load fluctuations.

Regarding dataset partitioning, to evaluate the model’s generalization capability to unseen contingencies, we randomly divide all valid N-1 scenarios into two subsets: 70% are designated as seen scenarios for the training phase, while the remaining 30% serve as unseen scenarios. The training set consists exclusively of samples generated from the seen scenarios, whereas the test set comprehensively covers instances from both seen and unseen topologies.

Table 6: Settings of N-1 contingency Dataset and Hyperparameters of SKM Sampling
<table><tr><td>TEST SYSTEM</td><td>BUSES</td><td>BRANCHES</td><td>REMOVABLE</td><td>SCENARIOS</td><td>β</td><td>ρ</td></tr><tr><td>IEEE 57-BUS</td><td>57</td><td>80</td><td>79</td><td>80</td><td>10</td><td>0.8</td></tr><tr><td>IEEE 118-BUS</td><td>118</td><td>186</td><td>177</td><td>178</td><td>50</td><td>0.8</td></tr><tr><td>IEEE 300-BUS</td><td>300</td><td>411</td><td>322</td><td>323</td><td>60</td><td>0.8</td></tr></table>

## D.3 Experimental Hyperparameters and Loss Function Configuration

In terms of model architecture and training configuration, we employ a Heterogeneous Graph Neural Network (HGNN) consisting of two message-passing layers, with the hidden dimension uniformly set to 128. We utilize the Adam optimizer for training, initialized with a learning rate of 5e 4. A StepLR scheduler is applied to implement stepwise decay:

$$
\eta _ { t } = \eta _ { 0 } \cdot \gamma ^ { \lfloor t / S \rfloor } ,\tag{35}
$$

where $\eta _ { 0 }$ denotes the initial learning rate, $S = 2 5$ represents the step size, and $\gamma = 0 . 8$ is the decay factor. The batch size is set to 64 during the training phase. Conversely, the testing phase utilizes a batch size of 1 to simulate real-world scenarios requiring real-time inference for individual samples.

The composite loss function comprises two components: regression loss and classification loss. Following the methodology of T-SKM [15], the regression loss adopts the Mean Squared Error (MSE) formulation, calculated as a weighted sum of the pre-projection and post-projection losses:

$$
L _ { r e g } = 0 . 9 \cdot \mathrm { M S E } ( \hat { \bf y } _ { p r e } , { \bf y } ) + 0 . 1 \cdot \mathrm { M S E } ( \hat { \bf y } _ { p r o j } , { \bf y } ) ,\tag{36}
$$

where $\hat { \mathbf { y } } _ { p r e }$ represents the raw model output and $\hat { \mathbf { y } } _ { p r o j }$ denotes the output after constraint projection. This combination enables the model to learn the primary feature mapping while simultaneously encouraging satisfaction of physical constraints.

For the active-set prediction task, we employ the Binary Focal Loss to address the issue of positivenegative sample imbalance [44]:

$$
L _ { \mathrm { f o c a l } } ( p _ { t } ) = - \alpha _ { t } ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) ,\tag{37}
$$

where $p _ { t }$ is the model’s estimated probability for the ground-truth class. The focusing parameter $\gamma = 2 . 0$ down-weights easy-to-classify examples, while the balancing parameter $\alpha = 0 . 5$ adjusts the relative importance of positive and negative samples. The total loss function is expressed as:

$$
L _ { \mathrm { t o t a l } } = L _ { \mathrm { r e g } } + \lambda ( t ) \cdot L _ { \mathrm { c l s } } .\tag{38}
$$

Here, $\lambda ( t )$ follows a linear warmup schedule, increasing linearly from 0 to 1 over the first 40 training epochs. This strategy allows the model to prioritize the regression task during the initial training phase before gradually integrating the classification objective.

Active-set labels are generated using a relative slack criterion. For a two-sided inequality $l _ { i } \ \leq$ $g _ { i } ( x ) \leq u _ { i }$ , the constraint is labeled active if its slack to either bound is less than $1 \%$ of the feasible interval $u _ { i } - l _ { i }$ . This threshold is used only for training the active-set predictor. In each iteration, the number of sampled constraints corresponds to approximately 10% of the total constraint set. The specific configurations for the sampling batch size $\beta$ and the hybrid sampling ratio $\rho$ across different test systems are detailed in Table 6.

## D.4 Hyperparameter Sensitivity of Hybrid Sampling Strategy

To analyze the sensitivity of our proposed active-set predict method to the batch size parameter, we conducted a controlled experiment using ground truth data. In this setup, we assume the availability of a perfect predictor by directly utilizing the actual active sets derived from the optimal solutions to guide the sampling process. We compared the convergence performance of the standard uniform sampling against the ground truth guided hybrid sampling across varying sampling ratios on IEEE 118 bus systems.

![](images/718ed82b640d1bf7459d1d1b6358a3476cfc33e8c2481da56c18b42af2702cee.jpg)  
(a) IEEE 118 Mean Iterations

![](images/f7c4a41fa81e60aaa9dcbe5fd922da7ace599231f6d2a3a4b8d78e591863fedf.jpg)  
(b) IEEE 118 Max Iterations

![](images/9de8cb1b66832a327a69762b0c538f98a6918b56684e1f92accd9fb97d80a427.jpg)  
(c) IEEE 300 Mean Iterations

![](images/22fbc311678cc87866673b1c507f75bc7cb0aede92b6d72ed96c654fb68dca80.jpg)  
(d) IEEE 300 Max Iterations  
Figure 3: Sensitivity analysis of sampling strategies with a perfect predictor on IEEE 118-bus and 300-bus systems. Comparison between uniform sampling and ground truth guided hybrid sampling across varying sampling ratios.

Figure 3 presents the sensitivity analysis where the SKM iterative process is initialized using the solution predicted by the HGNN.

For the IEEE 118 bus system shown in Figure 3a and Figure 3b, the results demonstrate that the hybrid sampling strategy exhibits remarkable stability when the active set is accurately identified. Regardless of the proportion of additional random sampling, both the mean and maximum iteration counts remain at a consistently low level. This implies that once the true active constraints are covered, the algorithm is effectively decoupled from the interference of massive inactive constraints.

In the larger IEEE 300 bus system shown in Figure 3c and Figure 3d, the hybrid strategy shows a significantly sharper improvement in convergence speed compared to uniform sampling. With only a minimal amount of random sampling included, the hybrid approach allows the iteration count to drop rapidly to its optimal level. Specifically, Figure 3d reveals that the hybrid strategy effectively suppresses worst case scenarios, achieving performance with about a tenth of the sampling effort that would otherwise require full set scanning in the uniform approach.

## D.5 Detailed Time Breakdown for AT-SKM-Net on SC DC-OPF Problems

Using the same models and experimental settings as in the main SC DC-OPF experiments, we provide a detailed runtime breakdown of T-SKM-Net and AT-SKM-Net. Specifically, we separately measure the time spent in the equality projection stage and the inequality SKM iteration stage, which correspond to the two main computational components accelerated by the proposed Cholesky update and active-set-guided sampling strategy, respectively.

Table 7: Detailed runtime breakdown of T-SKM-Net and AT-SKM-Net on SC DC-OPF problems. AT-SKM-AC denotes the full AT-SKM-Net with both active-set-guided sampling and Cholesky update. Runtime values are reported in milliseconds.
<table><tr><td>SYSTEM</td><td>METHOD</td><td>EQ. PROJ. TIME</td><td>EQ. PROJ. SPD.</td><td>INEQ. ITER. TIME</td><td>INEQ. ITER. SPD.</td><td>TOTAL SKM SPD.</td></tr><tr><td>57-bus</td><td>T-SKM-Net</td><td> $0 . 5 6 \pm 0 . 3 4$ </td><td>1.00×</td><td> $0 . 4 1 \pm 1 . 2 0$ </td><td>1.00×</td><td>1.00×</td></tr><tr><td>57-bus</td><td>AT-SKM-AC</td><td> $0 . 1 8 \pm 0 . 0 2$ </td><td>3.11×</td><td> $0 . 0 6 \pm 0 . 0 7$ </td><td>6.83×</td><td>4.04×</td></tr><tr><td>118-bus</td><td>T-SKM-Net</td><td> $1 . 5 6 \pm 0 . 0 6$ </td><td>1.00×</td><td> $2 . 1 3 \pm 2 . 5 1$ </td><td>1.00×</td><td>1.00×</td></tr><tr><td>118-bus</td><td>AT-SKM-AC</td><td> $0 . 3 0 \pm 0 . 0 3$ </td><td>5.20×</td><td> $1 . 0 1 \pm 1 . 3 8$ </td><td>2.11×</td><td>2.81×</td></tr><tr><td>300-bus</td><td>T-SKM-Net</td><td> $7 . 6 5 \pm 1 . 5 4$ </td><td>1.00×</td><td> $2 . 0 7 \pm 4 . 5 9$ </td><td>1.00×</td><td>1.00×</td></tr><tr><td>300-bus</td><td>AT-SKM-AC</td><td> $0 . 7 0 \pm 0 . 0 2$ </td><td>10.92×</td><td> $0 . 3 2 \pm 0 . 2 0$ </td><td>6.46×</td><td>9.53×</td></tr></table>

As shown in Table 7, AT-SKM-Net substantially reduces the equality projection time compared with T-SKM-Net. Moreover, the acceleration becomes more pronounced as the network size increases:

the equality projection speedup grows from 3.11 on the 57-bus system to 10.92 on the 300-bus system. This trend is consistent with the theoretical motivation of the Cholesky update, whose advantage becomes more significant when the equality projection involves larger constraint matrices.

The inequality SKM iteration stage is also accelerated. Compared with T-SKM-Net, AT-SKM-Net reduces the inequality iteration time by 6.83 , 2.11 , and 6.46 on the 57-, 118-, and 300-bus systems, respectively. These reductions reflect the benefit of active-set-guided sampling, which avoids spending excessive iterations on inactive or redundant constraints. Overall, AT-SKM-Net achieves total SKM speedups of $4 . 0 4 \times , 2 . 8 1 \times$ , and 9.53 across the three systems, which is consistent with the main-paper conclusion that the proposed framework accelerates both equality projection and inequality iteration while preserving strict feasibility.

## D.6 Robustness under High-Load Operating Conditions

To further evaluate the robustness of proposed AT-SKM-Net, we construct an additional high-load test regime for the SC DC-OPF problem. In the main experiments, load perturbations are sampled from the range [0.8, 1.2] around the nominal demand profile. Here, we shift the load range to [1.0, 1.2], thereby concentrating the test instances in more heavily loaded operating conditions. Such cases are more challenging because line-flow and generation-capacity constraints are more likely to approach their limits, making the active-set pattern more sensitive to both load variations and topology changes.

Table 8: Performance of T-SKM-Net and AT-SKM-Net under high-load SC DC-OPF conditions. Load perturbations are sampled from [1.0, 1.2]. Runtime values are reported in milliseconds. The inequality iteration column reports the mean number of iterations, with the maximum shown in parentheses.
<table><tr><td>SYSTEM</td><td>METHOD</td><td>EQ. PROJ. TIME EQ. SPD.</td><td></td><td>INEQ. ITER.</td><td>INEQ. TIME</td><td></td><td></td><td>INEQ. SPD. SKM SPD. MEAN GAP (%)</td></tr><tr><td>57-BUS</td><td>T-SKM-NET</td><td> $0 . 4 2 \pm 0 . 1 4$ </td><td>1.00×</td><td>16.00 (133)</td><td> $0 . 7 0 \pm 1 . 2 2$ </td><td>1.00×</td><td>1.00×</td><td>0.014</td></tr><tr><td>57-BUS</td><td>AT-SKM-AC</td><td> $0 . 1 8 \pm 0 . 0 4$ </td><td>2.33×</td><td>2.75 (12)</td><td> $0 . 1 3 \pm 0 . 1 4$ </td><td>5.38×</td><td>3.61×</td><td>0.010</td></tr><tr><td></td><td>118-BUS T-SKM-NET</td><td> $1 . 8 1 \pm 0 . 5 9$ </td><td>1.00×</td><td>18.59 (160)</td><td> $1 . 4 5 \pm 2 . 8 0$ </td><td>1.00×</td><td>1.00×</td><td>0.004</td></tr><tr><td></td><td>118-BUS AT-SKM-AC</td><td> $0 . 3 7 \pm 0 . 2 2$ </td><td>4.89×</td><td>9.21 (56)</td><td> $0 . 8 2 \pm 1 . 5 6$ </td><td>1.77×</td><td>2.74×</td><td>0.004</td></tr><tr><td>300-BUS T-SKM-NET</td><td></td><td> $7 . 0 5 \pm 0 . 5 1$ </td><td>1.00×</td><td>25.49 (248)</td><td> $1 . 7 8 \pm 3 . 0 2$ </td><td>1.00×</td><td>1.00×</td><td>0.136</td></tr><tr><td></td><td>300-BUS AT-SKM-AC</td><td> $0 . 7 1 \pm 0 . 0 2$ </td><td>9.93×</td><td>5.34 (28)</td><td> $0 . 4 2 \pm 0 . 5 8$ </td><td>4.24×</td><td>7.81×</td><td>0.136</td></tr></table>

As shown in Table 8, AT-SKM-Net continues to achieve clear acceleration over T-SKM-Net under the high-load regime. The equality projection stage is consistently accelerated, with the speedup increasing from 2.33 on the 57-bus system to 9.93 on the 300-bus system. This again confirms that the Cholesky update becomes increasingly beneficial as the system size grows.

The inequality iteration stage also remains robust under heavier loading. Compared with T-SKM-Net, AT-SKM-Net reduces the mean number of inequality iterations from 16.00 to 2.75 on the 57-bus system, from 18.59 to 9.21 on the 118-bus system, and from 25.49 to 5.34 on the 300-bus system. The maximum iteration count is also substantially reduced across all systems, indicating that activeset-guided sampling mitigates difficult tail cases even when more constraints are close to becoming active.

These results suggest that the HGNN predictor remains effective in more stressed operating conditions. The warm-start prediction continues to provide useful initialization, while the active-set predictor still concentrates sampling on the most relevant constraints. As a result, AT-SKM-Net achieves total SKM speedups of 3.61 , 2.74 , and 7.81 on the 57-, 118-, and 300-bus systems, respectively, while maintaining comparable mean optimality gaps to T-SKM-Net.

## D.7 Active-Set Prediction Quality

Table 9: Active set prediction performance on test “seen / unseen” scenarios.
<table><tr><td>SYSTEM</td><td>ACCURACY</td><td>PRECISION</td><td>RECALL</td></tr><tr><td>IEEE 57-BUS</td><td>0.9995 / 0.9995</td><td>0.9814 / 0.9802</td><td>0.9804 / 0.9815</td></tr><tr><td>IEEE 118-BUS</td><td>0.9995 / 0.9995</td><td>0.9988 / 0.9989</td><td>0.9948 / 0.9940</td></tr><tr><td>IEEE 300-BUS</td><td>0.9991 /0.9991</td><td>0.9647 / 0.9649</td><td>0.9834 / 0.9834</td></tr></table>

To validate that the theoretical acceleration potential established in the sensitivity analysis is achievable in practice, we evaluate the actual classification performance of the proposed HGNN predictor.

Table 9 details the Accuracy, Precision, and Recall metrics across three test systems. We explicitly distinguish between “seen” and “unseen” scenarios to assess the model’s generalization capability.

Given that active constraints constitute only a tiny fraction of the total set yet strictly define the optimal solution, missing even a single binding constraint can severely stall convergence. Consequently, the Recall rate is as the most paramount metric. Table 9 shows that the HGNN consistently achieves a Recall exceeding 98% across all systems. This high coverage guarantees that the predictor captures nearly all binding constraints required to define the active manifold, allowing SKM to focus on the feasible boundary and minimizing the reliance on random exploration to recover missed information. These recall values provide empirical support for the coverage condition in Assumption 2, namely ${ \mathcal { A } } ^ { * } \subseteq { \hat { \mathcal { A } } } ,$ up to the small prediction error observed in finite test samples.

In addition to reliability, the method maintains consistently high precision from 96.5% to 99.9%, ensuring that the predicted candidate sets are compact with minimal redundancy. Such compactness is essential for efficiency as it prevents the dilution of sampling probability. Moreover, the framework demonstrates that HGNN remains robustness to topological changes. The negligible performance gap between seen and unseen scenarios indicates that the HGNN has captured the underlying physical laws governing constraint activation, rather than overfitting to specific patterns. This robustness ensures that AT-SKM sustains its acceleration performance even under unexpected line failures.

## D.8 Robustness to Active-Set Perturbations

We further analyze the robustness of the proposed active-set-guided sampling strategy when the predicted active set is imperfect. In practice, the prediction errors of an active-set predictor can be divided into two types: false negatives, where truly active constraints are dropped from the predicted pool, and false positives, where inactive constraints are added to the predicted pool. Both types of errors weaken the sampling guidance, but their effects are not symmetric.

To isolate these two effects, we conduct a two-dimensional perturbation experiment using the oracle active set. Starting from the true active set, we randomly drop a prescribed fraction of true active constraints and add a prescribed number of inactive constraints. Both the add and drop ratios are normalized by the size of the true active set. We then run the SKM layer with the perturbed active-set pool and report the mean number of inequality iterations.

Table 10: Mean number of SKM inequality iterations under active-set perturbations on the IEEE 118-bus system. Rows denote the ratio of added inactive constraints, and columns denote the ratio of dropped true active constraints. Both ratios are normalized by the true active-set size.
<table><tr><td rowspan="2">ADD RATIO</td><td colspan="7">DROP RATIO</td></tr><tr><td>0.00</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.50</td></tr><tr><td>0.0</td><td>15.68</td><td>18.04</td><td>20.02</td><td>22.71</td><td>24.41</td><td>24.71</td><td>29.61</td></tr><tr><td>0.5</td><td>15.85</td><td>20.64</td><td>29.60</td><td>33.44</td><td>39.12</td><td>39.71</td><td>42.53</td></tr><tr><td>1.0</td><td>16.37</td><td>21.41</td><td>29.04</td><td>35.27</td><td>37.91</td><td>42.79</td><td>64.02</td></tr><tr><td>1.5</td><td>17.04</td><td>21.34</td><td>27.89</td><td>35.81</td><td>39.81</td><td>41.90</td><td>63.72</td></tr><tr><td>2.0</td><td>17.64</td><td>22.10</td><td>30.32</td><td>36.28</td><td>41.06</td><td>42.42</td><td>63.08</td></tr><tr><td>3.0</td><td>19.12</td><td>24.32</td><td>31.16</td><td>36.93</td><td>41.80</td><td>44.11</td><td>65.27</td></tr></table>

Tables 10 and 11 show that both false positives and false negatives degrade the efficiency of active-setguided sampling. However, dropping true active constraints is generally more harmful than adding inactive constraints. On the IEEE 118-bus system, when no inactive constraints are added, increasing the drop ratio from 0 to 0.5 increases the mean iteration count from 15.68 to 29.61. In contrast, when no true active constraints are dropped, increasing the add ratio from 0 to 3.0 only increases the mean iteration count from 15.68 to 19.12. A similar trend appears on the IEEE 300-bus system, where the iteration count increases from 3.69 to 19.24 as the drop ratio grows from 0 to 0.5, but only from 3.69 to 4.04 as the add ratio grows from 0 to 3.0 without dropping true active constraints. For reference, the original T-SKM-Net requires 34.60 inequality iterations on the IEEE 118-bus system and 24.72 iterations on the IEEE 300-bus system.

Table 11: Mean number of SKM inequality iterations under active-set perturbations on the IEEE 300-bus system. Rows denote the ratio of added inactive constraints, and columns denote the ratio of dropped true active constraints.
<table><tr><td rowspan="2">ADD RATIO</td><td colspan="7">DROP RATIO</td></tr><tr><td>0.00</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.50</td></tr><tr><td>0.0</td><td>3.69</td><td>4.15</td><td>7.68</td><td>8.88</td><td>10.99</td><td>13.96</td><td>19.24</td></tr><tr><td>0.5</td><td>3.69</td><td>4.51</td><td>9.65</td><td>10.39</td><td>14.43</td><td>16.55</td><td>23.97</td></tr><tr><td>1.0</td><td>3.72</td><td>4.78</td><td>11.53</td><td>12.50</td><td>17.17</td><td>23.93</td><td>33.14</td></tr><tr><td>1.5</td><td>3.77</td><td>5.25</td><td>12.03</td><td>13.27</td><td>20.15</td><td>24.80</td><td>39.51</td></tr><tr><td>2.0</td><td>3.80</td><td>4.79</td><td>13.36</td><td>15.24</td><td>23.41</td><td>28.90</td><td>43.07</td></tr><tr><td>3.0</td><td>4.04</td><td>5.79</td><td>17.90</td><td>21.70</td><td>34.99</td><td>40.32</td><td>69.75</td></tr></table>

This behavior is consistent with the SKM update rule. In each iteration, SKM samples a subset of constraints and projects onto the most violated constraint within that sampled subset. Therefore, inactive false-positive constraints do not necessarily cause harmful projections; their main effect is to dilute the probability of sampling truly active constraints. In contrast, false negatives remove truly active constraints from the guided sampling pool, forcing them to be recovered only through the uniform sampling branch.

Let m denote the total number of constraints, s the size of the predicted active-set pool, and $\beta _ { g }$ and $\beta _ { u }$ the guided and uniform sample sizes, respectively. For a true active constraint that remains in the predicted pool, its probability of being sampled is

$$
p _ { \mathrm { k e e p } } = \frac { \beta _ { g } } { s } + \left( 1 - \frac { \beta _ { g } } { s } \right) \frac { \beta _ { u } } { m - \beta _ { g } } > \frac { \beta _ { g } } { s } .\tag{39}
$$

If this active constraint is mistakenly dropped from the predicted pool, it can only be selected by the uniform branch:

$$
p _ { \mathrm { d r o p } } = \frac { \beta _ { u } } { m - \beta _ { g } } .\tag{40}
$$

Thus, adding ∆ inactive constraints mainly dilutes the guided sampling probability from $\beta _ { g } / s$ to $\beta _ { g } / ( s + \Delta )$ . By contrast, dropping a true active constraint moves it from the guided branch to the uniform branch, reducing its sampling probability from approximately $\beta _ { g } / s \mathrm { t o } \mathrm { \Delta } \bar { \beta } _ { u } / ( m - \beta _ { g } )$ . Since in our setting the predicted active-set pool is much smaller than the full constraint set and the guided sampling ratio satisfies $\rho > 1 / 2$ , we have

$$
\frac { \beta _ { u } / ( m - \beta _ { g } ) } { \beta _ { g } / ( s + \Delta ) } = \frac { 1 - \rho } { \rho } \cdot \frac { s + \Delta } { m - \beta _ { g } } \ll 1 .\tag{41}
$$

This explains why dropping true active constraints is more damaging than adding inactive constraints. At the same time, the perturbation results show that AT-SKM-Net remains robust under mild active-set prediction errors, and that the guided sampling strategy continues to provide acceleration even when the predicted pool contains moderate false positives.

## D.9 Transfer Learning and Pseudo-Label Active-Set Training

Although the experiments in the main paper use supervised training with solver-generated reference solutions, AT-SKM-Net does not fundamentally require exact optimal solutions for all training instances. The framework relies on two acceleration components: a warm-start candidate for equality projection and SKM iteration, and active-set scores for guided inequality sampling. These components can be trained or adapted with weaker supervision.

For the warm-start branch, prior work on T-SKM-Net has shown that the predictor can be trained without solver-generated ground-truth solutions by directly optimizing the objective together with constraint-violation penalties. For the active-set prediction branch, exact solver-derived activeset labels are also not strictly necessary. Since this branch is only used to bias the SKM sampling distribution toward likely active constraints, pseudo labels can be constructed from corrected solutions produced by the model itself. Specifically, for each target sample, the transferred model first produces a warm-start solution, the SKM layer then corrects it to a feasible solution, and the active constraints of this corrected solution are used as pseudo labels for retraining the active-set heads.

To evaluate this possibility, we consider two transfer settings: from the IEEE 57-bus system to the IEEE 118-bus system, and from the IEEE 118-bus system to the IEEE 300-bus system. In each setting, we first train a source model on the smaller system and transfer it to the larger target system. We then freeze the shared HGNN backbone and the warm-start output branch, and retrain only the active-set prediction heads using 5% of the target N-1 scenarios. Importantly, these active-set heads are not trained with solver-derived active-set labels; instead, they use pseudo labels generated from the model’s own corrected SKM outputs.

Table 12: Transfer learning results with pseudo-label active-set training. Runtime values are reported in milliseconds. The notation 57 118 denotes transfer from the IEEE 57-bus system to the IEEE 118-bus system.
<table><tr><td>TRANSFER</td><td>METHOD</td><td>EQ. PROJ.</td><td>EQ. SPD.</td><td>INEQ. ITER.</td><td>INEQ. TIME</td><td>INEQ. SPD.</td><td>SKM SPD.</td><td>GAP (%)</td></tr><tr><td> $5 7 \to 1 1 8$ </td><td>T-SKM-NET</td><td> $1 . 5 4 \pm 0 . 0 2$ </td><td>1.00×</td><td>57.31 (148)</td><td> $5 . 9 3 \pm 7 . 0 7$ </td><td>1.00×</td><td>1.00×</td><td>0.009</td></tr><tr><td> $5 7 \to 1 1 8$ </td><td>AT-SKM-AC</td><td> $0 . 2 8 \pm 0 . 0 2$ </td><td>5.50×</td><td>29.28 (59)</td><td> $1 . 9 6 \pm 1 . 5 8$ </td><td>3.02×</td><td>3.33×</td><td>0.009</td></tr><tr><td>118→300</td><td>T-SKM-NET</td><td> $7 . 0 1 \pm 0 . 0 8$ </td><td>1.00×</td><td>30.83 (218)</td><td> $2 . 0 6 \pm 2 . 5 3$ </td><td>1.00×</td><td>1.00×</td><td>0.618</td></tr><tr><td>118→300</td><td>AT-SKM-AC</td><td> $0 . 7 4 \pm 0 . 0 5$ </td><td>9.47×</td><td>6.30 (18)</td><td> $0 . 5 1 \pm 0 . 5 4$ </td><td>4.03×</td><td>7.26×</td><td>0.590</td></tr></table>

As shown in Table 12, AT-SKM-AC retains substantial acceleration even when the active-set heads are trained only with pseudo labels. In the 57 118 transfer setting, the average number of inequality iterations decreases from 57.31 to 29.28, and the worst-case iteration count decreases from 148 to 59. In the more challenging 118 300 setting, the average iteration count decreases from 30.83 to 6.30, and the worst-case iteration count decreases from 218 to 18. These reductions show that pseudo-label active-set learning still provides useful sampling guidance after transfer to a larger system.

The runtime results show the same trend. In the 57 118 setting, AT-SKM-AC achieves a 5.50 speedup in equality projection, a 3.02 speedup in inequality iteration, and a 3.33 total SKM speedup. In the 118 300 setting, the corresponding speedups are 9.47 , 4.03 , and 7.26 . Meanwhile, the optimality gap remains comparable to T-SKM-Net: 0.009% versus 0.009% for 57 118, and 0.590% versus 0.618% for 118 300. Since the final outputs are corrected by the projection and SKM layers, both methods preserve strict constraint feasibility after correction.

These results suggest that supervised ground-truth solutions are a training choice in the current implementation rather than a fundamental requirement of AT-SKM-Net. When exact solver-generated labels are expensive or insufficient on larger systems, the model can still transfer from a smaller system and use self-generated pseudo labels to train the active-set predictor, thereby retaining much of the acceleration benefit while preserving strict feasibility.