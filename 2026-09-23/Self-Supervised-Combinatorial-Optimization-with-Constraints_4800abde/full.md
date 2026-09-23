# Self-Supervised Combinatorial Optimization with Constraints via Frank–Wolfe

Akbar Rafiey<sup>∗</sup> NYU ar9530@nyu.edu

Yifei Xu<sup>∗</sup>   
NYU   
yx3590@nyu.edu

Nikolaos Karalias MIT stalence@mit.edu

## Abstract

Self-supervised learning for combinatorial optimization has emerged as a promising paradigm for solving discrete optimization problems with neural networks, but a central challenge remains: handling hard combinatorial constraints within continuous, gradient-based training. Continuously extending combinatorial objectives to convex domains is a powerful technique, yet existing approaches often require projection steps that constrain neural network outputs to lie inside the feasible polytope and rely on ad-hoc and problem-specific constructions. We propose a general framework in which the neural network is allowed to predict arbitrary continuous vectors that could potentially lie outside of the feasible polytope. These predictions are then approximated by sparse convex combinations of feasible solutions using a geometric decomposition algorithm based on Frank–Wolfe methods and approximate Carathéodory results. This decomposition induces an a.e.-differentiable, self-supervised loss defined as the expected value of the discrete objective. The same procedure provides an automatic rounding guarantee at inference time. We demonstrate strong empirical performance across multiple combinatorial problems, including the Quadratic Assignment Problem, Maximum Coverage, and the Traveling Salesperson Problem.

## 1 Introduction

Combinatorial optimization (CO) forms a central pillar of optimization theory and practice, encompassing a broad class of problems in which one seeks to optimize an objective over a discrete, often exponentially large, feasible set. Such problems arise ubiquitously across science and engineering, from routing and resource allocation to learning and inference. Canonical examples include the Traveling Salesperson Problem (TSP), Maximum Coverage (MC), and Quadratic Assignment Problem (QAP), among many others. What unifies these problems is not only their computational hardness, but also the rich algorithmic structure induced by discrete constraints that describe large-scale discrete combinatorial spaces of configurations.

Despite this shared structure, successful approaches to CO have historically been case-specific, often reflecting deep problem-dependent insights; "stroke-of-genius" heuristics. A common algorithmic template, especially in approximation algorithms, relaxes the discrete problem into a continuous or convex surrogate, solves for a fractional solution, and then applies a bespoke rounding or improvement procedure to recover feasibility. While this relax-and-round paradigm has led to remarkable practical results and theoretical guarantees, adapting this approach to gradient-based data-driven settings can be challenging. This limitation and challenge is especially pronounced when problem instances are drawn from structured distributions, increasingly common in real world applications. Self-supervised learning (SSL) approaches for combinatorial optimization offer an appealing alternative: unlike static approximation algorithms and heuristics, they can leverage latent patterns in data, train efficiently without requiring large amounts of labeled solutions, and incorporate algorithmic priors that facilitate learning and generalization.

In this work, we bridge the gap between discrete optimization and deep learning by building on work that used continuous extensions of discrete functions as losses for neural CO [23, 24, 50]. The central idea in this line of work is to use extensions to embed discrete constraints and objectives directly into a learning pipeline, allowing learning and optimization to proceed end-to-end. By smoothing the discrete landscape, these approaches enable an unsupervised learning paradigm where models learn to exploit patterns and discover high-quality solutions without requiring expensive labels.

More specifically, we propose a generic learning-based pipeline that integrates hard combinatorial constraints directly into training and inference. The neural network is allowed to output an arbitrary continuous vector, without any feasibility requirement. This output is then passed through a geometric decomposition algorithm inspired by approximate Carathéodory and Frank–Wolfe methods [4, 46, 12], which produces a sparse convex combination of feasible solutions. The resulting decomposition may differ substantially from the network output but is always expressed entirely in terms of feasible points. This effectively shifts the burden of constraint enforcement from tuning penalty terms to capturing constraints algorithmically through our decomposition algorithm. The decomposition induces a (a.e. differentiable) distribution over feasible solutions, and the expected value of the discrete objective under this distribution is used directly as a self-supervised training loss. In addition, the discrepancy between the network output and its decomposition can be quantified and included as an auxiliary regularization term, with an associated weight. While not required by the framework, we find empirically that including this term improves optimization stability and solution quality. Gradients propagate through this objective using automatic differentiation. At inference time, the same procedure yields a small set of feasible candidate solutions from which the best one is selected. Figure 1 summarizes the pipeline.

Our pipeline is projection-free. The decomposition in [24] requires the neural prediction to lie inside the feasible polytope, with feasibility ensured separately through problem-specific constructions. For example, it uses distinct projections for the hypersimplex (Proposition 4.4) and the spanning-tree polytope (Theorem E.7). Our method removes the need for projection and accepts arbitrary ambientspace outputs and uses a Frank–Wolfe decomposition to construct a sparse convex combination of feasible vertices.

Moreover, our decomposition has an explicit run time certificate and is more efficient in terms of the number of Linear Maximization Oracle (LMO) calls. It requires exactly one exact or approximate LMO call per iteration, so T iterations use exactly T LMO calls. In contrast, each iteration of the generic GLS (Groetschel/Lovasz/Schrijver) decomposition in [24] invokes minimal-face and ray-boundary optimization routines that themselves require solving convex optimization problems using the oracle; see the proof of Theorem 4.1. Consequently, a single GLS iteration may require many LMO calls. For example, a standard ellipsoid-based implementation requires approximately $O ( n ^ { 2 } L )$ oracle calls per iteration and $O ( n ^ { 3 } L )$ calls overall, where L is a precision parameter.

Although specialized GLS decompositions can be more efficient for particular polytopes, deriving them requires nontrivial, problem-specific techniques. Given an efficient exact or approximate LMO, our procedure applies unchanged across different polytopes, with only the LMO changing. It thus provides a unified oracle-based method for hypersimplex, Birkhoff, spanning-tree, and matching polytopes, subsuming the specialized settings considered in [24, 50].

## Contributions. We make the following contributions:

• A projection-free, self-supervised framework for CO. We introduce a unified learning framework that enables neural networks to optimize combinatorial objectives under hard discrete constraints by operating over convex polytopes of feasible solutions. Unlike prior geometric-extension methods, our framework does not require network outputs to lie in the polytope during training and provides a common formulation across a broad class of combinatorial structures.

• An oracle-based extension layer using Frank–Wolfe decomposition. To realize this framework, we propose a Frank–Wolfe–style extension layer that maps an arbitrary continuous prediction to a sparse convex combination of feasible solutions, yielding a distribution over feasible discrete solutions. Given an efficient exact or approximate linear maximization oracle (LMO), the same procedure applies across different polytopes; only the implementation of the LMO changes. It requires one LMO call per decomposition iteration and avoids problem-specific projections and specialized decomposition algorithms.

![](images/cb6bb3737bc7ca1d4758d71c1a920aec8ccc5ac46d9a9aec122cf6ffc9e8d30b.jpg)  
Figure 1: Overview of our framework. During training, the model with parameters θ outputs a point $\mathbf { x } _ { \theta }$ , which may lie outside ${ \mathcal P } ,$ the convex hull of feasible solutions. Our Frank–Wolfe decomposition layer constructs a feasible proxy u as a convex combination of at most T feasible vertices $\mathbf { v } _ { t } ,$ with weights $\alpha _ { t } ( \mathbf { x } _ { \theta } )$ . The self-supervised loss $\mathcal { L } _ { \mathrm { S S L } }$ is computed from the expected discrete objective f and a reconstruction term weighted by $\lambda \geq 0$ . The weights $\alpha _ { t } ( \mathbf { x } _ { \theta } )$ are a.e. differentiable w.r.t x<sub>θ</sub>, allowing gradients from the loss to propagate through the decomposition layer back to the neural network. During inference (green box), the same procedure generates feasible candidates, and the best solution $\mathbf { v } ^ { * }$ is selected. Solid arrows indicate the forward pass, and the dashed arrow indicates backpropagation.

• A differentiable, self-supervised loss with built-in rounding guarantees. The induced convex decomposition allows us to define a differentiable loss as the expected discrete objective, with approximation error explicitly incorporated. At inference time, selecting the best solution in the support yields a feasible solution whose objective value is no worse than the expectation. This provides a direct rounding guarantee and unifies learning and solution recovery within a single pipeline.

• Empirical validation across diverse combinatorial domains. We demonstrate the effectiveness of our framework on a range of classical CO problems—including MC, TSP, and QAP—showing strong empirical performance and broad applicability across diverse constraint families.

## 2 Related Work

Extensions and optimization. A long line of work in CO views discrete problems through the lens of continuous optimization by embedding feasible solutions into a convex polytope and optimizing over this relaxation. This perspective is deeply rooted in algorithm design and polyhedral geometry, where constructing convex or concave extensions with favorable optimization properties has been a central theme [43, 13, 48, 60]. In particular, the convex closure provides the tightest convex extension of a discrete set function and underlies many relaxation-based methods [16, 60]. Classical solvers and approximation algorithms routinely exploit this framework: prominent examples include LP and SDP relaxations for problems such as Max Cut and TSP, with solvers like Concorde and Gurobi relying on large-scale LP and cutting-plane techniques [19, 67, 14, 2]. While these approaches yield strong theoretical guarantees, they typically rely on problem-specific, rounding procedures that are non-differentiable and difficult to integrate into modern, data-driven learning pipelines.

Neural CO. Recent work has explored unsupervised and self-supervised learning approaches for CO that optimize continuous surrogates of discrete objectives, typically induced through probabilistic relaxations and expected-value formulations [1, 22, 61, 63]. Compared to reinforcement learning or supervised methods [28, 31, 62], these approaches avoid labeled data and often exhibit more stable training. Unsupervised CO pipelines involve several crucial components, including neural network architecture [45, 62, 54] and the role of input features for well-known classes of models [40, 27]. In this work, our focus is on loss function design and rounding. A common strategy in self-supervised CO is to parameterize a distribution over discrete solutions and use the expected value of the discrete objective as a differentiable training loss [22, 63, 7], enabling smoother optimization but raising challenges related to constraint enforcement. A major consideration in these approaches is enforcing constraints on the outputs of neural networks. Techniques in the literature typically involve projecting the network output onto the feasible set. This has been explored for cardinality constraints using ideas from optimal transport [66], and for general linear constraints using gradient based methods [15, 69]. This also includes Sinkhorn-style projection methods and their extensions [56, 65]. In the continuous (and potentially non-convex) optimization setting, several projection techniques have also been developed [38, 39, 37, 36]. Other approaches add penalty terms or improve training dynamics through annealing or physics-inspired formulations [57, 55, 47]. Our method differs from these approaches by incorporating feasibility directly through geometric decomposition. We define a distribution supported entirely on feasible solutions, allowing constraints to be built into the distribution itself. Thus, the training loss focuses solely on the discrete objective while remaining a.e. differentiable. A closely related line of work develops neural CO methods based on geometric relaxations of feasibility polytopes. Karalias et al. [23, 24] propose differentiable extensions for set-function optimization under various constraints, while Nerem et al. [50] study CO problems over permutations. These approaches typically require the neural network output to lie within the feasibility polytope. In contrast, our approach allows arbitrary continuous network outputs and applies a general-purpose geometric decomposition across a broad class of combinatorial constraints.

Approximate Carathéodory and Frank-Wolfe. Our approach builds on results from convex geometry and projection-free optimization. A central component is the Frank–Wolfe algorithm [17, 9], a first-order method for constrained optimization. Several variants with improved convergence have been studied [21, 34], and Frank–Wolfe has been widely applied to structured and combinatorial polytopes [18, 35, 32]. Approximate variants of Frank–Wolfe with provable guarantees have also been proposed [42, 70]. From a geometric perspective, the Carathéodory theorem [10] states that any point in a polytope $P \subseteq \mathbb { R } ^ { d }$ can be expressed as a convex combination of at most $d + 1$ corners of the polytope. Approximate versions provide dimension-independent bounds on sparse approximations [4, 46]. Obtaining such sparse convex combinations of polytope vertices is essential for our framework, and Frank–Wolfe-type algorithms provide a natural algorithmic mechanism to achieve this [46, 12].

## 3 Problem Formulation and Learning Setup

We consider CO problems defined over a discrete feasible set. Each problem instance is specified by a collection of variables $V ~ = ~ \{ v _ { 1 } , \ldots , v _ { n } \}$ , each assigned a value from a discrete domain $\dot { D } = \{ 1 , \ldots , d \}$ , a finite set of feasible solutions ${ \mathcal { C } } \subseteq D ^ { n }$ , and a real-valued objective function $f : { \mathcal { C } } \to \mathbb { R }$ . The goal is to find an optimal feasible solution $\mathbf { x } ^ { \ast } \in \mathcal { C }$ that optimizes (either minimizes or maximizes) the objective function:

$$
\operatorname* { m a x } f ( \mathbf { x } ) , \quad { \mathrm { s . t . } } \quad \mathbf { x } \in { \mathcal { C } } .\tag{1}
$$

This abstract formulation accommodates a broad class of CO problems, including those defined over binary vectors, general discrete assignments, and permutations.

Many examples in this paper focus on problems with Boolean decision variables, where solutions can be represented by indicator vectors encoding subset or structure selection under constraints. Two representative running examples are Maximum Coverage, which selects a subset of elements subject to a budget constraint, and the TSP, where binary variables indicate whether edges are included in a Hamiltonian tour. Our framework also applies to CO problems over permutations, e.g., the QAP, and to non-Boolean discrete domains, e.g., k-submodular maximization.

A standard geometric perspective on CO is to consider the convex hull of all feasible solutions,

$$
{ \mathcal { P } } = \operatorname { c o n v } ( { \mathcal { C } } ) .
$$

For CO problems with Boolean domain, we have $\mathcal { P } = \mathrm { c o n v } ( \mathcal { C } ) \subseteq [ 0 , 1 ] ^ { n }$ . As an example, when C consists of all subsets of size $k , \mathcal { P }$ is the associated cardinality polytope, a.k.a hypersimplex. Each $S \in { \mathcal { C } }$ is represented by its indicator vector $\mathbf { 1 } _ { S } \in \{ 0 , 1 \} ^ { n }$ , where $( \mathbf { 1 } _ { S } ) _ { i } = 1$ if element i belongs to S and $( { \bf 1 } _ { S } ) _ { i } = 0$ otherwise. The polytope associated with C is then $\mathcal { P } : = \mathrm { c o n v } \{ \mathbf { 1 } _ { S } : S \in \mathcal { C } \} \subseteq [ 0 , 1 ] ^ { n }$ whose vertices, i.e., extreme points $\mathrm { e x t } ( \mathcal { P } )$ , coincide with the indicator vectors of subsets of size k.

Note that although P may have an exponentially large facet description, many important combinatorial polytopes admit efficient linear optimization via polynomial-time separation or optimization oracles.

## 3.1 Self-Supervised Learning for CO

In the self-supervised learning setting, each problem instance $\mathcal { T }$ is described by input features $\mathbf { Z } _ { \mathcal { I } }$ possibly together with instance-specific structure. These features are processed by a neural network $\mathrm { N N } _ { \theta }$ , which outputs a representation $\mathbf { x } _ { \theta }$ . Importantly, this output is not required to lie in the convex polytope of feasible solutions.

Rather than training the network to imitate optimal solutions, learning is guided by a loss function defined on an a.e. differentiable extension of the combinatorial objective. This extension is constructed using the geometry of the polytope ${ \mathcal P } ,$ , allowing the objective to be meaningfully evaluated on fractional solutions and optimized using gradient-based methods.

The network is trained over a collection of instances without access to ground-truth optimal solutions, making the design of the loss function central. At inference time, the continuous output x is converted into a feasible discrete solution using a principled, geometry-based procedure, rather than heuristic rounding. Moreover, because no labeled data is required, the same objective can be optimized directly on test instances, allowing additional computation at test time to further improve solution quality.

## 4 Proposed Method

We now describe the overall pipeline. Consider a polytope $\mathcal { P } = \mathrm { c o n v } ( \mathcal { C } )$ , where $\mathcal { C } \subset \mathbb { R } ^ { n }$ denotes the set of feasible solutions for a CO problem. Let ex $( { \mathcal { P } } ) \subseteq { \mathcal { C } }$ denote the set of extreme points i.e., corners, of $\mathcal { P }$ which correspond to feasible combinatorial solutions.

Given a target point $\mathbf { x } \in \mathbb { R } ^ { n }$ , which can be thought of as the output of a neural network, our goal is to find a point u $\in \mathcal { P }$ that serves as a good proxy for x and satisfies desirable properties. In particular, we want u to be a sparse convex combination of $T$ points $\mathbf { v } _ { 0 } , \ldots , \mathbf { v } _ { T - 1 } \in \operatorname { e x t } ( \mathcal { P } )$ . That is, for a small $T ,$ we have

$$
\begin{array}{c} \mathbf { u } ( \mathbf { x } ) = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } ( \mathbf { x } ) \mathbf { v } _ { t } \in \mathcal { P } \quad \mathrm { w i t h } \quad \left\{ \mathbf { v } _ { t } \in \mathrm { e x t } ( \mathcal { P } ) , \right. \qquad \\ { \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } ( \mathbf { x } ) = 1 ; \ \alpha _ { t } \geq 0 . \qquad } \end{array}\tag{2}
$$

The coefficients $\alpha _ { t } ( \mathbf { x } )$ naturally define a distribution $\mathcal { D } ( \mathbf { x } )$ over feasible solutions. We define the FW-induced objective

$$
F ( \mathbf { x } ) = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } ( \mathbf { x } ) f ( \mathbf { v } _ { t } ) = \mathbb { E } _ { \mathbf { v } \sim \mathcal { D } ( \mathbf { x } ) } [ f ( \mathbf { v } ) ] .\tag{3}
$$

Training. In the learning pipeline, let $\mathbf { x } _ { \theta }$ denote the prediction of a neural network with parameters θ. For a maximization problem (1), we train the network by minimizing the self-supervised loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S S L } } ( \theta ) : = - F ( \mathbf { x } _ { \theta } ) + \lambda \| \mathbf { x } _ { \theta } - \mathbf { u } ( \mathbf { x } _ { \theta } ) \| _ { 2 } ^ { 2 } , \qquad \lambda \geq 0 . } \end{array}\tag{4}
$$

In order to optimize the loss function using gradient-based methods, the weights $\alpha _ { t } ( \mathbf { x } _ { \theta } )$ must be differentiable functions of $\mathbf { x } _ { \theta }$ . Under this condition, derivatives of the loss with respect to the network parameters can be computed via the chain rule: $\begin{array} { r } { \frac { \partial F } { \partial \boldsymbol { \theta } } = \frac { \partial F } { \partial \boldsymbol { \alpha } ( \mathbf x _ { \boldsymbol { \theta } } ) } \cdot \frac { \partial \boldsymbol { \alpha } ( \mathbf x _ { \boldsymbol { \theta } } ) } { \partial \mathbf x _ { \boldsymbol { \theta } } } \cdot \frac { \partial \mathbf x _ { \boldsymbol { \theta } } } { \partial \boldsymbol { \theta } } } \end{array}$ , where $\frac { \partial F } { \partial \alpha _ { t } ( \mathbf { x } _ { \theta } ) } = f ( \mathbf { v } _ { t } )$ and $\frac { \partial \pmb { \alpha } ( \mathbf { x } _ { \theta } ) } { \partial \mathbf { x } _ { \theta } }$ comes from differentiating the steps our FW decomposition algorithm, Algorithm 1. Note that under the stable-oracle decision assumption introduced in Section 4.1, the selected vertices $\mathbf { v } _ { t }$ are locally constant functions of $\mathbf { x } _ { \theta }$ almost everywhere. Hence, $\begin{array} { r } { \frac { \partial \mathbf { v } _ { t } } { \partial \mathbf { x } _ { \theta } } = 0 } \end{array}$ wherever the derivative exists, and the gradient propagates only through the coefficients. The gradient of the term $\lambda \| \mathbf { x } _ { \theta } - \mathbf { u } ( \mathbf { x } _ { \theta } ) \| _ { 2 } ^ { 2 }$ is computed similarly by differentiating through $\begin{array} { r } { \mathbf { u } ( \mathbf { x } _ { \theta } ) = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } ( \mathbf { x } _ { \theta } ) \mathbf { v } _ { t } } \end{array}$

In Section 4.1, we present our algorithm, which given x computes $\mathbf { u } ( \mathbf { x } ) \in \mathcal { P }$ satisfying the properties in (2). Moreover, we show that the weights $\alpha _ { t } ( \mathbf { x } )$ returned by our algorithm are a.e. differentiable w.r.t. x and provide a convergence rate in $T .$ , the number of iterations of the algorithm. For notational simplicity, throughout Section 4.1 we suppress the dependence on the target point x, writing u and $\alpha _ { t }$ in place of $\mathbf { u } ( \mathbf { x } )$ and $\alpha _ { t } ( \mathbf { x } )$

Remark 4.1 (Extension property). The FW-induced objective F defines a surrogate on $\mathbb { R } ^ { n }$ . For $F$ to be an extension of $f : { \bar { \mathcal { C } } }  \mathbb { R }$ , it must agree with f on C. We therefore adopt the convention that

Algorithm 1 FW DECOMPOSITION $( \mathbf { x } , \mathcal { P } , T ; \mathrm { L M O } )$   
1: Input: Target $\mathbf { x } \in \mathbb { R } ^ { n }$ , polytope ${ \mathcal P } ,$ support budget $T ,$ and an exact or approximate LMO.   
2: Initialize: $\begin{array} { r } { \mathbf { v } _ { 0 } = \mathbf { u } _ { 0 } = \arg \operatorname* { m a x } _ { \mathbf { v } \in \mathcal { P } } \langle \mathbf { x } , \mathbf { v } \rangle , \alpha _ { 0 } \gets 1 } \end{array}$   
3: for t = 1 to T − 1 do   
4: $\mathbf { r } _ { t } \gets \mathbf { x } - \mathbf { u } _ { t - 1 }$ {Residual}   
5: v<sub>t</sub> ← arg ma $\mathrm { x } _ { \mathbf { v } \in \mathcal { P } } \langle \mathbf { r } _ { t } , \mathbf { v } \rangle$ {Find a corner via LMO}   
6: $\mathbf { d } _ { t }  \mathbf { v } _ { t } - \mathbf { u } _ { t - 1 }$ {Descent direction}   
7: $\gamma _ { t } \gets \mathrm { C l i p } ( 0 , 1 , \langle \bar { \mathbf { r } } _ { t } , \mathbf { d } _ { t } \rangle / \| \mathbf { d } _ { t } \| _ { 2 } ^ { 2 } )$ {Line search step}   
8: $\alpha _ { j } \gets ( 1 - \gamma _ { t } ) \alpha _ { j }$ for all existing weight $\alpha _ { j } , j \in \bar { \{ 0 , \ldots , t - 1 \} }$   
9: α<sub>t</sub> ← γ<sub>t</sub>   
10: $\mathbf { u } _ { t } \gets ( 1 - \gamma _ { t } ) \mathbf { u } _ { t - 1 } + \gamma _ { t } \mathbf { v } _ { t }$   
11: end for   
12: Output: Weights and vertices $\{ ( \alpha _ { t } , \mathbf v _ { t } ) \} _ { t = 0 } ^ { T - 1 }$ such that $\begin{array} { r } { \mathbf { u } _ { T - 1 } = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } \mathbf { v } _ { t } . } \end{array}$

Algorithm 1 returns the singleton decomposition $( 1 , \mathbf { x } )$ whenever $\mathbf { x } \in { \mathcal { C } }$ . Consequently, $F ( \mathbf { c } ) = f ( \mathbf { c } )$ for every $c \in { \mathcal { C } }$ . Since $\mathcal { C }$ is finite, this convention modifies $F$ only on a set of Lebesgue measure zero and therefore does not affect its almost-everywhere differentiability.

## 4.1 From NN Output to Polytope: Frank-Wolfe Decomposition

In this section, we describe our algorithm presented in Algorithm 1. Given x, define the following quadratic objective:

$$
\begin{array} { r } { J ( \mathbf { u } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { u } \| _ { 2 } ^ { 2 } , \qquad \mathbf { u } \in \mathcal { P } . } \end{array}\tag{5}
$$

Starting at an initial point $\mathbf { u } _ { 0 }$ in the polytope our goal is to iteratively move towards $\mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } , . . , \mathbf { u } _ { T - 1 }$ so that at each iteration we decrease ${ \cal J } ( { \bf u } _ { t } )$ , and at the end ${ \bf u } _ { T - 1 }$ has the desired aforementioned properties. More specifically, we start with an initial point $\mathbf { u } _ { 0 }$ which is a corner of the polytope $\mathcal { P }$ i.e. $\mathbf { v } _ { 0 } = \mathbf { u } _ { 0 } \in \mathrm { e x t } ( \mathcal { P } )$ . Then, in the first iteration, we find another corner of the polytope ${ \mathcal P } ,$ say $\mathbf { v } _ { 1 }$ and set $\mathbf { u } _ { 1 } = ( 1 - \gamma _ { 1 } ) \mathbf { u } _ { 0 } + \gamma _ { 1 } \mathbf { v } _ { 1 }$ with an appropriate value $\gamma _ { 1 } \in [ 0 , 1 ]$ , to be determined later. By construction, $\mathbf { u } _ { 1 } \in \mathcal { P }$ as it is a convex combination of two corners of the polytope. This iterative process is repeated and at every iteration t we have

$$
\mathbf { u } _ { t } = ( 1 - \gamma _ { t } ) \mathbf { u } _ { t - 1 } + \gamma _ { t } \mathbf { v } _ { t }\tag{6}
$$

with an appropriate value for $\gamma _ { t } \in [ 0 , 1 ]$ and $\mathbf { v } _ { t } \in \mathop { \mathrm { e x t } } ( \mathcal { P } )$ . The algorithm terminates after $T - 1$ iterations and returns ${ \bf u } _ { T - 1 }$ , which will lie in $\mathcal { P }$ by construction. At this point, ${ \bf u } _ { T - 1 }$ is a convex combination of $\mathbf { u } _ { 0 } = \mathbf { v } _ { 0 } , \ldots , \mathbf { v } _ { T - 1 }$ with weights $\alpha _ { 0 } , \ldots , \alpha _ { T - 1 }$ that are calculated recursively with respect to $\gamma _ { t } \mathbf { S } .$ . Specifically, $\textstyle \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } = 1$ , with $\begin{array} { r } { \alpha _ { 0 } = \prod _ { i = 1 } ^ { T - 1 } ( 1 - \gamma _ { i } ) } \end{array}$ and $\begin{array} { r } { \alpha _ { t } = \gamma _ { t } \prod _ { i = t + 1 } ^ { T - 1 } ( 1 - \gamma _ { i } ) } \end{array}$ The main two points to address here are which corner $\mathbf { v } _ { t }$ to choose and what is a good value for step size $\gamma _ { t }$

Note that the function $J ( \mathbf { u } )$ is convex and its gradient is $\nabla J ( \mathbf { u } ) = \mathbf { u } - \mathbf { x }$ . We define the residual at iteration t as $\mathbf { r } _ { t } : = \mathbf { x } - \mathbf { u } _ { t - 1 } = - \nabla J ( \mathbf { u } _ { t - 1 } )$ , which points from our current estimate toward the target. Frank–Wolfe is a "projection-free" method. At every iteration, instead of moving toward the target in the gradient direction and then projecting back into $\mathcal { P }$ (which can be computationally expensive), it finds the vertex $\mathbf { v } _ { t }$ of the polytope that lies furthest in the direction of the residual $\mathbf { r } _ { t }$ . It then takes a small step $\gamma _ { t }$ from the current point $\mathbf { u } _ { t - 1 }$ toward that vertex $\mathbf { v } _ { t }$ . Since both $\mathbf { u } _ { t - 1 }$ and $\mathbf { v } _ { t }$ are in ${ \mathcal { P } } _ { \mathrm { { : } } }$ their convex combination is guaranteed to remain inside $\mathcal { P } _ { \cdot }$ . The exact procedure is shown in Algorithm 1, and in order to find $\mathbf { v } _ { t }$ we require what is known as LMO.

Definition 4.2 (Linear Maximization Oracle (LMO)). Given a residual $\mathbf { r } \in \mathbb { R } ^ { n }$ , an exact LMO returns a vertex $\mathbf { v } ^ { \star }$ that maximizes the alignment with r:

$$
\mathbf { v } ^ { \star } = \arg \operatorname* { m a x } \langle \mathbf { r } , \mathbf { v } \rangle \quad \mathrm { s . t . } \quad \mathbf { v } \in \mathcal { P }\tag{7}
$$

A fundamental property of linear optimization is that a linear objective over a polytope attains an optimum at a vertex. Hence, an LMO returns a (integral) vertex of the polytope. When exact maximization is difficult, the convergence statement below uses a relative Frank–Wolfe gap approximation [21, 35, 42, 70]. Formally, at iteration t, a δ-relative $\mathrm { F W - g a p }$ oracle returns $\mathbf { v } _ { t } \in \bar { \mathrm { e x t } } ( \bar { \mathcal { P } } )$ satisfying

$$
\langle \mathbf { r } _ { t } , \mathbf { v } _ { t } - \mathbf { u } _ { t - 1 } \rangle \geq \delta \operatorname* { m a x } _ { \mathbf { v } \in \mathcal { P } } \langle \mathbf { r } _ { t } , \mathbf { v } - \mathbf { u } _ { t - 1 } \rangle ,\tag{8}
$$

where $\mathbf { r } _ { t } = \mathbf { x } - \mathbf { u } _ { t - 1 }$ and $\delta \in ( 0 , 1 ]$ This is the approximation notion used in our convergence statement. Exact LMOs satisfy (8) with $\delta = 1$

For the step size $\gamma _ { t } ,$ , several choices are possible. We compute it using a simple line-search:

$$
\gamma _ { t } = \mathrm { C l i p } ( 0 , 1 , \langle \mathbf { r } _ { t } , \mathbf { d } _ { t } \rangle / \| \mathbf { d } _ { t } \| _ { 2 } ^ { 2 } ) ,
$$

with the convention $\gamma _ { t } ~ = ~ 0$ when $\mathbf { d } _ { t } \ = \ 0$ . This allows for efficient computation and supports branchwise differentiation of the weights $\left\{ \alpha _ { t } \right\}$

Assumption (stable oracle decisions). Fix the support budget T. We assume that the deterministic oracle and tie-breaking rule do not switch pathologically. The set of inputs x at which an arbitrarily small perturbation can change one of the selected vertices, or change whether a line-search step is clipped, has zero volume. Equivalently, for almost every input x, there is a small neighborhood around x on which Algorithm 1 selects the same vertices $\mathbf { v } _ { 0 } , \ldots , \mathbf { v } _ { T - 1 }$ and uses the same linesearch clipping cases. This assumption rules out artificial rules that switch between equally valid approximate vertices on dense sets. It is satisfied by the standard comparison-based oracles used in combinatorial optimization, including top-k selection, matroid greedy algorithms such as Kruskal’s algorithm, and exact assignment or matching oracles with fixed tie-breaking.

All these results together yield the following theorem. The convergence guarantee is the standard exact Frank–Wolfe rate, and the a.e. differentiability proof is given in Section A.

Theorem 4.3 (Exact Frank–Wolfe Decomposition). Let $\mathbf { x } \in \mathbb { R } ^ { n }$ , and let $\mathcal { P } \subset \mathbb { R } ^ { n }$ be a polytope with diameter $\begin{array} { r } { D : = \mathrm { { \ d i a m } } ( \mathcal { P } ) \ = \ \operatorname* { m a x } _ { \mathbf { u } , \mathbf { v } \in \mathcal { P } } \| \mathbf { u } - \mathbf { v } \| _ { 2 } } \end{array}$ Run Algorithm 1 with an exact LMO and support budget $T \geq 2 ,$ so that it performs T − 1 Frank–Wolfe updates. Then the algorithm outputs $\begin{array} { r } { \mathbf { u } _ { T - 1 } = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } \mathbf { v } _ { t } \in \mathcal { P } } \end{array}$ , where $\mathbf { v } _ { t } \in \mathrm { e x t } ( \mathcal { P } ) , \alpha _ { t } \geq 0 ,$ , and $\textstyle \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } = 1$ . Moreover, if $\begin{array} { r } { \mathbf { u } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { u } \in \mathcal { P } } J ( \mathbf { u } ) } \end{array}$ , then

$$
J ( { \bf u } _ { T - 1 } ) - J ( { \bf u } ^ { \star } ) \leq \frac { 2 D ^ { 2 } } { T + 1 } .
$$

The coefficients $\alpha _ { t }$ are almost everywhere differentiable functions of x.

Corollary 4.4 (Approximate Frank–Wolfe Decomposition). If the exact LMO in Theorem 4.3 is replaced by a δ-relative FW-gap oracle satisfying (8), then the same convex-combination conclusion holds. Let $\begin{array} { r } { D = \mathrm { d i a m } ( \mathcal { P } ) , \mathbf { u } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { u } \in \mathcal { P } } J ( \mathbf { u } ) } \end{array}$ , and $h _ { 0 } = J ( \mathbf { u } _ { 0 } ) - J ( \mathbf { u } ^ { \star } )$ . Then

$$
J ( { \bf u } _ { T - 1 } ) - J ( { \bf u } ^ { \star } ) \leq \frac { 2 \left( { D ^ { 2 } } / { \delta + h _ { 0 } } \right) } { \delta ( T - 1 ) + 2 } .
$$

The coefficient map remains almost everywhere differentiable.

Discussion on projection onto the polytope. In the above discussion we did not require neural network output to lie in the polytope, yielding a generic projection-free framework. Nevertheless, when this occurs—either by design of the neural network or because it is empirically advantageous— our framework aligns with approximate Carathéodory results. When $\mathbf { x } \in \mathcal { P }$ , minimizing ${ \cal J } ( { \bf u } ) =$ $\frac { 1 } { 2 } \| \mathbf { x } - \mathbf { u } \| _ { 2 } ^ { 2 }$ via Algorithm 1 produces a sparse convex combination of vertices of P approximating x. Constructing such sparse convex combinations is the goal of approximate Carathéodory results. Unlike the exact Carathéodory theorem, the approximate variants relax exact representation in favor of an additive error to achieve sparser convex combinations [51, 4, 46]. In the exact-LMO setting, Theorem 4.3 yields the approximate Carathéodory bound $\begin{array} { r } { \| \mathbf { x } - \mathbf { u } _ { T - 1 } \| _ { 2 } ^ { 2 } \leq \frac { 4 D ^ { 2 } } { T + 1 } } \end{array}$

## 4.2 Examples with Efficient (Approximate) LMO

We briefly discuss how our framework extends to structured polytopes; details are in Section G.

Matroid polytopes. Matroid base polytopes capture many combinatorial constraints (e.g., cardinality, partition, spanning trees) and admit efficient LMOs via greedy maximum-weight base algorithms. Since our method requires only an efficient LMO, it applies uniformly across matroids without modification, unlike [24], which requires constraint-specific projections and decompositions.

Birkhoff and matching polytopes. The Birkhoff and (perfect) matching polytopes admit LMOs via maximum-weight matching, allowing direct optimization without projections or penalties. This is unlike prior work $( \mathrm { e . g . , [ 5 0 ] ) }$ , which relies on specialized decompositions and projection. Although exact LMOs may be costly at scale, we find that simple greedy approximations suffice empirically.

Table 1: MC performance comparison across three datasets, reporting mean inference time and coverage across k. Among neural methods, ■ indicates ranking the 1st, ■ the 2nd in each column.
<table><tr><td rowspan="3"></td><td colspan="4">Random500</td><td colspan="4">Random1000</td><td colspan="4">Rail</td></tr><tr><td colspan="2"> $\overline { { k = 1 0 } }$ </td><td colspan="2"> $k = 5 0$ </td><td colspan="2"> $k = 2 0$ </td><td colspan="2"> $k = 1 0 0$ </td><td colspan="2"> $\overline { { k = 2 0 } }$ </td><td colspan="2">k = 50</td></tr><tr><td>Time (s)</td><td>Coverage</td><td>Time (s)</td><td>Coverage</td><td>Time (s)</td><td>Coverage</td><td>Time (s)</td><td>Coverage</td><td>Time (s)</td><td>Coverage</td><td>Time (s)</td><td>Coverage</td></tr><tr><td>Random (240s)</td><td>240.00</td><td>13372.76</td><td>240.00</td><td>36786.89</td><td>240.00</td><td>24133.50</td><td>240.00</td><td>70527.31</td><td>240.00</td><td>5291.67</td><td>240.00</td><td>7367.00</td></tr><tr><td>Gurobi (120s)</td><td>11.574</td><td>15714.90</td><td>120.065</td><td>44880.59</td><td>37.298</td><td>31347.62</td><td>120.139</td><td>89696.83</td><td>121.059</td><td>5631.67</td><td>121.033</td><td>7604.67</td></tr><tr><td>Greedy</td><td>0.057</td><td>15640.99</td><td>0.121</td><td>44597.56</td><td>0.190</td><td>31105.89</td><td>0.460</td><td>88685.40</td><td>0.727</td><td>5617.00</td><td>1.320</td><td>7630.00</td></tr><tr><td>Ucom2-short</td><td>1.041</td><td>15253.35</td><td>0.739</td><td>44208.95</td><td>1.734</td><td>29791.66</td><td>1.621</td><td>88472.61</td><td>2.299</td><td>5512.67</td><td>2.454</td><td>7594.67</td></tr><tr><td>GeoNCO</td><td>0.579</td><td>15177.77</td><td>1.108</td><td>41247.80</td><td>0.825</td><td>29810.12</td><td>1.908</td><td>81357.29</td><td>1.856</td><td>5343.00</td><td>2.084</td><td>7411.67</td></tr><tr><td>CardNN-noTTO-S</td><td>1.688</td><td>9231.54</td><td>1.721</td><td>33055.87</td><td>1.869</td><td>18458.92</td><td>1.881</td><td>65793.40</td><td>1.929</td><td>5074.33</td><td>2.189</td><td>7193.00</td></tr><tr><td>EGN-naive</td><td>53.041</td><td>15262.76</td><td>120.136</td><td>41272.68</td><td>120.010</td><td>29968.04</td><td>120.356</td><td>81166.12</td><td>120.676</td><td>5234.67</td><td>121.335</td><td>7408.33</td></tr><tr><td>RL (GNN+Actor)</td><td>0.069</td><td>14741.26</td><td>0.247</td><td>39510.96</td><td>0.102</td><td>29912.14</td><td>0.465</td><td>81158.54</td><td>0.219</td><td>5137.00</td><td>0.219</td><td>7456.67</td></tr><tr><td>FWNCO (ours)</td><td>0.426</td><td>15192.64</td><td>0.555</td><td>42172.44</td><td>0.647</td><td>29926.63</td><td>0.913</td><td>83559.22</td><td>1.432</td><td>5344.67</td><td>1.931</td><td>7393.00</td></tr></table>

## 5 Applications and Experiments

We evaluate our framework on three core CO problems—MC, QAP, and TSP—and show that the same decomposition and extension apply across all cases. Our method is competitive with and often outperforms SOTA approaches. Compared to SOTA approaches, which rely on problem-specific and novel engineered constructions, this generality is a key advantage. For testing methods and choice of baselines, we keep a strict one-shot setting (e.g., without Test Time Optimization (TTO)) to ensure fairness, detailed in Section E. Code and data are available in link.

## 5.1 Maximum Coverage

Let U be a ground set of elements with weights $w _ { u } ~ \ge ~ 0$ for $u \in \mathcal { U } .$ , and let $\{ S _ { 1 } , \ldots , S _ { n } \}$ be subsets $S _ { i } \subseteq { \mathcal { U } } .$ . The value of $\mathbf { x } \in \{ 0 , 1 \} ^ { n }$ is the total weight of covered elements, $\begin{array} { r l } { f _ { \mathrm { c o v } } ( \mathbf { x } ) } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { u \in \mathcal { U } } w _ { u } \mathbf { 1 } [ \exists i \in [ n ] } \end{array}$ with $u \in S _ { i }$ and $\mathbf { \bar { x } } ( i ) = 1 ]$ . The MC problem under cardinality constraint is $\mathrm { m a x } _ { \mathbf { x } \in \{ 0 , 1 \} ^ { n } } f _ { \mathrm { c o v } } ( \mathbf { x } )$ subject to $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \mathbf { x } ( i ) = k } \end{array}$ , where k denotes the prescribed cardinality.

Neural model and continuous targets. This problem can naturally be represented as a bipartite graph where the goal is to select k nodes from one part so that we maximize the total number of their neighbors in the other part. We apply a GAT-based encoder to obtain node embeddings. This $\mathbf { x } _ { \theta }$ is the input to our FW DECOMPOSITION algorithm to obtain a small list of subsets $\{ S ^ { ( t ) } \} _ { t = 0 } ^ { T - 1 }$ of size k, and coefficients $\{ \alpha _ { t } \} _ { t = 0 } ^ { T - 1 }$ . For training, we minimize the differentiable surrogate loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { C O V } } = - \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } f _ { \mathrm { c o v } } \big ( \mathbf { 1 } _ { S ^ { ( t ) } } \big ) + \lambda \| \mathbf { x } _ { \theta } - \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } \mathbf { 1 } _ { S ^ { ( t ) } } \| _ { 2 } ^ { 2 } } \end{array}$ and backpropagate through the FW steps into the GAT parameters. Following [24] (GeoNCO), we train on synthetic random graphs and test on real-world graphs. For evaluation, Table 1, we report max<sub>t</sub> $f _ { \mathrm { c o v } } \bigl ( \mathbf { 1 } _ { S ^ { ( t ) } } \bigr )$ without TTO and compare with the baselines without TTO. Our method is faster than GeoNCO and it often outperforms the learning baselines. Specifically, it consistently stays on the pareto front over the learning baselines and is also competitive against non-learning baselines considering the quality-efficiency tradeoff. (See Section B.4 for adversarial instances where greedy fails.)

Ablation on projection for MC. Table 4 shows that removing the polytope projection preserves solution quality while improving runtime. Although the model converges to a point outside the hypersimplex, the solution quality from our decomposition is slightly better than the projected version. This suggests the projection-free variant allows greater exploration by avoiding hard constraints.

## 5.2 Quadratic Assignment Problem

There are n facilities and n locations. Distances between locations are given by a matrix $\mathbf { B } \in \mathbb { R } ^ { n \times n }$ and flows between facilities by a matrix $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ . The goal is to assign each facility to a distinct location so as to minimize the total flow-weighted distance. Let $\Pi _ { n }$ denote the set of n×n permutation matrices. The objective is mi $\operatorname { n } _ { \mathbf { P } \in \Pi _ { n } } f _ { \mathrm { q a p } } ( \mathbf { \bar { P } } )$ where $f _ { \mathrm { q a p } } ( \mathbf { P } ) = \mathrm { t r a c e } ( \mathbf { A P B P } ^ { \top } )$ for all $\mathbf { \hat { P } } \in \Pi _ { n }$

Neural model and continuous targets. Given the output of encoder network, GraphSAGE, $\mathbf { X } _ { \theta } \in$ $[ 0 , 1 ] ^ { n \times n }$ , we apply our FW DECOMPOSITION algorithm using a deterministic greedy algorithm for weighted maximum matching. After $T$ iterations, it returns a short list of permutation matrices $\{ \mathbf { P } _ { t } \} _ { t = 0 } ^ { T - 1 }$ and coefficients $\{ \alpha _ { t } \} _ { t = 0 } ^ { T - 1 }$ . We train directly against the discrete QAP objective and using the following $\begin{array} { r } { \mathcal { L } _ { \mathrm { Q A P } } ( \mathbf { X } _ { \theta } ) = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } f _ { \mathrm { q a p } } ( \mathbf { P } _ { t } ) + \lambda \Vert \mathbf { X } _ { \theta } - \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } \mathbf { P } _ { t } \Vert _ { F } ^ { 2 } } \end{array}$ . Gradients are backpropagated through the FW steps into the parameters that produce $\mathbf { X } _ { \theta } .$ At evaluation we report min<sub>t</sub> $f _ { \mathrm { q a p } } ( \mathbf { P } _ { t } )$ .

Table 2: Mean gaps and inference time (s) for models pre-trained on QAP32 and applied to QAPLIB instances. Std. is the standard deviation of the class-wise mean gaps across the 14 data classes (bur–wil). In each column, ■ indicates ranking the 1st, ■ the 2nd.
<table><tr><td>Method</td><td>bur</td><td>chr</td><td>esc</td><td>had</td><td>kra</td><td>lipa</td><td>nug</td><td>rou</td><td>scr</td><td>sko</td><td>ste</td><td>tai</td><td>tho</td><td>wil</td><td>Avg.</td><td>Std.</td><td>|Time (s)</td></tr><tr><td>SM</td><td>22.3</td><td>460.1</td><td>301.6</td><td>17.4</td><td>65.3</td><td>19.0</td><td>45.5</td><td>35.8</td><td>123.4</td><td>29.0</td><td>475.5</td><td>180.5</td><td>55.0</td><td>13.8</td><td>181.2</td><td>163.3</td><td>0.01</td></tr><tr><td>RRWM</td><td>23.1</td><td>616.0</td><td>63.9</td><td>25.1</td><td>58.8</td><td>20.9</td><td>67.8</td><td>51.2</td><td>173.5</td><td>48.5</td><td>539.4</td><td>197.2</td><td>80.6</td><td>18.2</td><td>169.5</td><td>192.9</td><td>0.15</td></tr><tr><td>SK-JA</td><td>4.7</td><td>38.5</td><td>364.8</td><td>25.8</td><td>41.4</td><td>0.0</td><td>25.3</td><td>13.7</td><td>48.6</td><td>18.3</td><td>120.4</td><td>25.2</td><td>32.9</td><td>8.8</td><td>93.2</td><td>93.9</td><td>563.4</td></tr><tr><td>NGM</td><td>3.4</td><td>121.3</td><td>126.7</td><td>8.2</td><td>31.6</td><td>16.2</td><td>21.0</td><td>30.9</td><td>55.5</td><td>25.2</td><td>101.7</td><td>61.4</td><td>27.5</td><td>10.8</td><td>62.4</td><td>41.9</td><td>15.72</td></tr><tr><td>RGM</td><td>7.1</td><td>112.4</td><td>32.8</td><td>6.2</td><td>15.0</td><td>13.3</td><td>9.7</td><td>13.4</td><td>45.5</td><td>10.6</td><td>134.1</td><td>17.3</td><td>20.7</td><td>8.1</td><td>35.8</td><td>40.4</td><td>75.53</td></tr><tr><td>SAWT</td><td>2.8</td><td>110.7</td><td>13.5</td><td>3.8</td><td>30.1</td><td>0.4</td><td>9.2</td><td>10.8</td><td>28.5</td><td>17.7</td><td>93.5</td><td>16.5</td><td>24.8</td><td>8.1</td><td>26.8</td><td>33.5</td><td>12.11</td></tr><tr><td>FWNCO (ours)</td><td>2.3</td><td>97.7</td><td>19.0</td><td>2.8</td><td></td><td>37.0 13.5 11.5</td><td></td><td>10.9</td><td>23.6</td><td>15.0</td><td>83.5</td><td>20.7</td><td>23.0</td><td>9.2</td><td>26.4</td><td>28.8</td><td>1.56</td></tr></table>

We train on synthetic instances following [59]. For size n, we sample locations $\mathbf { L } \sim \mathrm { U n i f o r m } ( 0 , 1 ) ^ { 2 }$ and a symmetric flow matrix $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ with zero diagonal, upper-triangular entries sampled from [0, 1], and entries independently set to zero with probability $p$ (symmetrically). Testing uses the standard QAPLIB benchmark [8] (134 instances, 15 classes; see Section C.1).

Consistent with prior studies, we report per-class and aggregated gap statistics on instances of size 12–64, where baseline methods are tractable. Recent learning-based QAP methods, such as RGM [41] and SAWT [59], rely on iterative Learn-to-Construct or Learn-to-Improve pipelines which could be time consuming and prevent scalability. Our approach, on the other hand, produces a permutation in a single forward pass, which compared to the learning baselines, yields better efficiency, while achieving a better out of distribution generalization (OOD) quality.

In our experiments, Table 2, our method achieves the best overall performance among all baselines, including both learning-based and classical solvers. The consistently low standard deviation of the average optimality gap across instance classes indicates stable behavior and good generalization across distributions. Moreover, our approach is the fastest neural method, running noticeably faster than other neural baselines. The only two methods with lower runtime, SM and RRWM, are purely traditional heuristics, which come at a substantial loss in solution quality. While Table 2 reports sizes 12–64, our method scales to larger instances $( > 6 4 )$ with similar average gaps (Table 8). Most prior methods do not scale to such sizes with reasonable compute resources.

Ablation on projection for QAP. Algorithm 1 does not theoretically require its input to be in the Birkhoff polytope. However, we observed a weaker OOD generalization results for QAP due to the strong heterogeneity and distribution shift present in QAPLIB. Unconstrained outputs, i.e., "far from the polytope", cause the decomposition algorithm to return low quality permutations under drastic distributional shift. We therefore include a lightweight Sinkhorn projection layer that enforces approximate doubly-stochasticity, improving generalization. This step is introduced for practical considerations, not theoretical necessity.

## 5.3 TSP via Spanning Trees and Learned Matchings

We consider spanning tree polytopes, where $\mathcal { C }$ is the set of spanning trees of a graph. For each $T \in { \mathcal { C } }$ , the objective $\bar { f } _ { \mathrm { T S P } } ( T )$ is the cost of the TSP tour obtained deterministically via Christofides’s algorithm [11]. Training learns a distribution over spanning trees optimized for this objective.

Algorithm-guided neural optimization for TSP. One of the appeals of our approach is that it allows us to infuse learnable NN modules into an algorithmic framework. We adopt an algorithmically aligned, data-driven approach that integrates learning into the classical Christofides–Serdyukov algorithmic framework. Christofides’s algorithm builds a TSP solution from a minimum spanning tree, whereas recent advances (e.g., [3, 25, 26]) sample from spanning tree distributions constructed from a subtour LP solution before applying variants of the Christofides pipeline. Our approach can be viewed as a learning-based analogue of this paradigm: instead of constructing such distributions analytically, we learn a distribution over spanning trees in an end-to-end manner, optimized directly for the final TSP objective. Moreover, we extend this idea further by incorporating a learning module for the matching step, effectively learning over matching polytopes as well. Given the neural network output, FW DECOMPOSITION returns a sparse distribution over spanning trees using Kruskal’s algorithm as LMO. We sample trees from this distribution, and recover tours by applying minimumweight perfect matching on odd degree nodes followed by shortcutting. This pipeline defines a differentiable surrogate objective that mirrors the Christofides algorithm end to end. This also allows the perfect-matching step itself to be learned jointly, improving solution quality.

Table 3: Performance comparison on TSP benchmarks across different scales. Results include optimality gap (%) and inference time. Methods are categorized as Unsupervised Learning (UL), Exact, Heuristics, Reinforcement Learning (RL), and Supervised Learning (SL). Among neural methods, ■ indicates ranking the 1st, ■ the 2nd in each column.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="2">TSP-50</td><td colspan="2">TSP-100</td><td colspan="2">TSP-500</td><td colspan="2">TSP-1000</td></tr><tr><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td><td>Gap</td><td>Time</td></tr><tr><td>Concorde (Optimal) Gurobi (Optimal)</td><td>Exact</td><td>opt</td><td>0.04s</td><td>opt</td><td>0.21s</td><td>opt</td><td>17.91s</td><td>opt</td><td>435.61s</td></tr><tr><td>LKH3 (500)</td><td>Exact</td><td>opt 0.01%</td><td>0.27s 0.03s</td><td>opt 0.01%</td><td>1.14s 0.05s</td><td>N/A 0.93%</td><td>N/A 0.32s</td><td>N/A 1.30%</td><td>N/A</td></tr><tr><td>DIFUSCO</td><td>Heuristics</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.06s</td></tr><tr><td>COExpander  $( S = 1 , D _ { s } = 3 , I _ { s } = 5 )$ </td><td>SL SL</td><td>0.53% 0.02%</td><td>0.43s 0.11s</td><td>1.24% 0.07%</td><td>0.66s 0.20s</td><td>9.79% 3.80%</td><td>2.19s 0.69s</td><td>10.90% 6.53%</td><td>7.68s 2.50s</td></tr><tr><td>DIMES</td><td>RL</td><td>13.71%</td><td>0.05s</td><td>11.60%</td><td>0.08s</td><td>15.34%</td><td>0.39s</td><td>13.97%</td><td>0.79s</td></tr><tr><td>RL4CO (Sym-NCO)</td><td>RL</td><td>1.23%</td><td>0.36s</td><td>13.78%</td><td>0.66s</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>FWNCO (ours)</td><td>UL</td><td>1.81%</td><td>0.23s</td><td>3.48%</td><td>0.26s</td><td>7.30%</td><td>1.27s</td><td>8.27%</td><td>4.26s</td></tr><tr><td>FWNCO+learned matching (Ours)</td><td>UL</td><td>1.40%</td><td>0.52s</td><td>2.84%</td><td>0.74s</td><td>6.90%</td><td>2.77s</td><td>7.43%</td><td>9.16s</td></tr></table>

In our experiments, we consider Euclidean TSP instances $( G = ( V , E ) , D )$ , where V is a set of points drawn uniformly at random from the unit square [0, 1]<sup>2</sup>, E is the complete edge set on V, and D is the corresponding symmetric distance matrix induced by Euclidean distances, which satisfies the triangle inequality. A tour is defined as a Hamiltonian cycle that visits each node exactly once, and its cost is given by the sum of the lengths of its constituent edges under D. The objective is to find a tour of minimum total cost.

Neural TSP methods commonly fall into two paradigms [44]: Learning-to-Construct (LC), which builds TSP tours sequentially with hard feasibility checks but suffers from high inference latency on large instances due to its inherently sequential decoding, and Global Prediction (GP), which predicts a full structure (e.g., a heatmap) and relies on post-processing, yielding fast inference but strong sensitivity to recovery heuristics. COExpander [44] combines LC and GP via Adaptive Expansion (AE), but still depends on heatmap recovery, which still is a combinatorial optimization subroutine, inheriting GP’s core limitation. Motivated by critiques of heatmap-guided post-hoc search [68], our method directly optimizes feasible tours end-to-end by differentiating through the selected FW branch, reducing reliance on hand-crafted post-processing.

We compare against four baseline categories: (i) exact solvers (Concorde, Gurobi); (ii) classical heuristics (LKH3 [20], 500 trials); (iii) neural LC methods (Sym-NCO [29]), implemented in the RL4CO library [5]; and (iv) neural GP/AE and heatmap-based methods across RL and supervised settings, including DIMES [52], DIFUSCO [58], and COExpander, a SOTA AE method. In Section E, we provide discussion regarding approaches that use inference optimization techniques such as Gradient-based Active Search, and Search-based Refinement such as MCTS and 2-opt.

Our results, Table 3, show that our method consistently outperforms unsupervised neural baselines including RL. On larger instances, it also surpasses DIFUSCO, a strong diffusion-based approach trained with supervision. While we are still behind the strongest supervised baselines, e.g. COExpander, our approach is more resource-efficient. In particular, DIFUSCO and COExpander rely on up to ∼1.5M labeled instances (depending on the instance size) and require extensive training time on good hardware, whereas our model can be trained in under one hour on a single outdated GPU.

Ablation for TSP. We examine the effect of projecting the neural network output onto the spanning tree polytope and find that removing projection slightly improves performance on both TSP-500 and TSP-1000 (Table 13). We also study generalization across instance sizes and observe strong cross-size generalization (Table 12). Our model uses subtour LP solutions [14] as features, obtained via the QSopt LP solver, also used by Concorde. Despite this shared solver, our learning pipeline is significantly faster on large instances. Our ablation study, Section D.7, confirm that these gains arise from learned structures rather than arbitrary perturbations of the LP relaxation.

## 6 Conclusion

We propose a general framework for self-supervised CO under constraints. Our approach is flexible in that it does not require the neural network output to lie in a feasible polytope. It uses a generic procedure, to decompose the output and obtain a sparse distribution over feasible solutions, effectively enabling SSL. Our method shows strong empirical performance compared to neural baselines on several CO problems. We present an example of infusing learned modules within an algorithm in an end-to-end pipeline, showcasing this approach on TSP using the Christofides algorithm. We believe this paves the way toward new NN-infused algorithms that can effectively tackle complex CO problems. We note that model architecture can affect performance; we do not study this here and leave it for future work.

## References

[1] Saeed Amizadeh, Sergiy Matusevych, and Markus Weimer. Learning to solve circuit-sat: An unsupervised differentiable approach. In International Conference on Learning Representations (ICLR), 2018.

[2] David L Applegate, Robert E Bixby, Vašek Chvátal, and William J Cook. The traveling salesman problem: a computational study. In The Traveling Salesman Problem. Princeton university press, 2011.

[3] Arash Asadpour, Michel X Goemans, Aleksander Madry, Shayan Oveis Gharan, and Amin Saberi. An o (log n/log log n)-approximation algorithm for the asymmetric traveling salesman problem. Operations Research, 65(4):1043–1061, 2017.

[4] Siddharth Barman. Approximating nash equilibria and dense bipartite subgraphs via an approximate version of caratheodory’s theorem. In Proceedings of the forty-seventh annual ACM symposium on Theory of computing, pages 361–369, 2015.

[5] Federico Berto, Chuanbo Hua, Junyoung Park, Laurin Luttmann, Yining Ma, Fanchen Bu, Jiarui Wang, Haoran Ye, Minsu Kim, Sanghyeok Choi, et al. Rl4co: an extensive reinforcement learning for combinatorial optimization benchmark. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 5278–5289, 2025.

[6] Maximilian Böther, Otto Kißig, Martin Taraz, Sarel Cohen, Karen Seidel, and Tobias Friedrich. What’s wrong with deep learning in tree search for combinatorial optimization. In International Conference on Learning Representations (ICLR), 2022.

[7] Fanchen Bu, Hyeonsoo Jo, Soo Yong Lee, Sungsoo Ahn, and Kijung Shin. Tackling prevalent conditions in unsupervised combinatorial optimization: Cardinality, minimum, covering, and more. In 41st International Conference on Machine Learning (ICML), pages 4696–4729, 2024.

[8] Rainer E Burkard, Stefan E Karisch, and Franz Rendl. Qaplib–a quadratic assignment problem library. Journal ofGlobal optimization, 10(4):391–403, 1997.

[9] M. D. Canon and C. D. Cullum. A tight upper bound on the rate of convergence of Frank–Wolfe algorithm. SIAM Journal on Control, 6(4):509–516, 1968.

[10] Constantin Carathéodory. Über den variabilitätsbereich der koeffizienten von potenzreihen, die gegebene werte nicht annehmen. Mathematische Annalen, 64(1):95–115, 1907.

[11] Nicos Christofides. Worst-case analysis of a new heuristic for the travelling salesman problem. Technical report, 1976.

[12] Cyrille W. Combettes and Sebastian Pokutta. Revisiting the approximate Carathéodory problem via the Frank-Wolfe algorithm. Mathematical Programming, 197(1):191–214, January 2023. ISSN 1436-4646.

[13] Yves Crama. Concave extensions for nonlinear 0–1 maximization problems. Mathematical Programming, 61(1):53–60, 1993.

[14] George Dantzig, Ray Fulkerson, and Selmer Johnson. Solution of a large-scale travelingsalesman problem. Journal of the operations research society of America, 2(4):393–410, 1954.

[15] Priya L. Donti, David Rolnick, and J Zico Kolter. DC3: A learning method for optimization with hard constraints. In International Conference on Learning Representations (ICLR), 2021.

[16] James E Falk and Karla R Hoffman. A successive underestimation method for concave minimization problems. Mathematics of operations research, 1(3):251–259, 1976.

[17] Marguerite Frank, Philip Wolfe, et al. An algorithm for quadratic programming. Naval research logistics quarterly, 3(1-2):95–110, 1956.

[18] Dan Garber and Ofer Meshi. Linear-memory and decomposition-invariant linearly convergent conditional gradient algorithm for structured polytopes. Advances in neural information processing systems, 29, 2016.

[19] Michel X Goemans and David P Williamson. Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming. Journal ofthe ACM (JACM), 42(6):1115–1145, 1995.

[20] Keld Helsgaun. An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems. Roskilde: Roskilde University, 12:966–980, 2017.

[21] Martin Jaggi. Revisiting Frank–Wolfe: Projection-free sparse convex optimization. In Proceedings of the 30th International Conference on Machine Learning, volume 28, pages 427–435. PMLR, 2013.

[22] Nikolaos Karalias and Andreas Loukas. Erdos goes neural: an unsupervised learning framework for combinatorial optimization on graphs. Advances in Neural Information Processing Systems, 33:6659–6672, 2020.

[23] Nikolaos Karalias, Joshua Robinson, Andreas Loukas, and Stefanie Jegelka. Neural set function extensions: Learning with discrete functions in high dimensions. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[24] Nikolaos Karalias, Akbar Rafiey, Yifei Xu, Zhishang Luo, Behrooz Tahmasebi, Connie Jiang, and Stefanie Jegelka. Geometric algorithms for neural combinatorial optimization with constraints. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[25] Anna R Karlin, Nathan Klein, and Shayan Oveis Gharan. A (slightly) improved approximation algorithm for metric tsp. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory ofComputing, pages 32–45, 2021.

[26] Anna R Karlin, Nathan Klein, and Shayan Oveis Gharan. A deterministic better-than-3/2 approximation algorithm for metric tsp. In International Conference on Integer Programming and Combinatorial Optimization (IPCO), pages 261–274. Springer, 2023.

[27] Nicolas Keriven and Samuel Vaiter. What functions can graph neural networks compute on random graphs? the role of positional encoding. Advances in Neural Information Processing Systems (NeurIPS), 36:11823–11849, 2023.

[28] Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. In Advances in Neural Information Processing Systems (NeurIPS), pages 6348–6358, 2017.

[29] Minsu Kim, Junyoung Park, and Jinkyoo Park. Sym-nco: Leveraging symmetricity for neural combinatorial optimization. Advances in Neural Information Processing Systems, 35:1936– 1949, 2022.

[30] Vijay Konda and John Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems (NeurIPS), volume 12. MIT Press, 1999.

[31] Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations (ICLR), 2019.

[32] Rahul G Krishnan, Simon Lacoste-Julien, and David Sontag. Barrier frank-wolfe for marginal inference. Advances in Neural Information Processing Systems, 28, 2015.

[33] Yam Kushinsky, Haggai Maron, Nadav Dym, and Yaron Lipman. Sinkhorn algorithm for lifted assignment problems. SIAM Journal on Imaging Sciences, 12(2):716–735, 2019.

[34] Simon Lacoste-Julien and Martin Jaggi. On the global linear convergence of frank-wolfe optimization variants. Advances in Neural Information Processing Systems (NeurIPS), 28, 2015.

[35] Simon Lacoste-Julien, Martin Jaggi, Mark Schmidt, and Patrick Pletscher. Block-coordinate Frank-Wolfe optimization for structural SVMs. In Proceedings of the 30th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research. PMLR, 2013.

[36] Xinpeng Li, Enming Liang, and Minghua Chen. Gauge flow matching for efficient constrained generative modeling over general convex set. In ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2025.

[37] Enming Liang and Minghua Chen. Efficient bisection projection to ensure nn solution feasibility for optimization over general set.

[38] Enming Liang, Minghua Chen, and Steven H Low. Low complexity homeomorphic projection to ensure neural-network solution feasibility for optimization over (non-) convex set. In International Conference on Machine Learning (ICML), pages 20623–20649, 2023.

[39] Enming Liang, Minghua Chen, and Steven H Low. Homeomorphic projection to ensure neural-network solution feasibility for constrained optimization. Journal of Machine Learning Research, 25(329):1–55, 2024.

[40] Derek Lim, Joshua David Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie Jegelka. Sign and basis invariant networks for spectral graph representation learning. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

[41] Chang Liu, Zetian Jiang, Runzhong Wang, Lingxiao Huang, Pinyan Lu, and Junchi Yan. Revocable deep reinforcement learning with affinity regularization for outlier-robust graph matching. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

[42] Francesco Locatello, Rajiv Khanna, Michael Tschannen, and Martin Jaggi. A unified optimization view on generalized matching pursuit and frank-wolfe. In Artificial intelligence and statistics (AISTAT), pages 860–868. PMLR, 2017.

[43] László Lovász. Submodular functions and convexity. In Mathematical Programming The State ofthe Art: Bonn 1982, pages 235–257. Springer, 1983.

[44] Jiale Ma, Wenzheng Pan, Yang Li, and Junchi Yan. Coexpander: Adaptive solution expansion for combinatorial optimization. In International Conference on Machine Learning, 2025.

[45] Yimeng Min, Frederik Wenkel, Michael Perlmutter, and Guy Wolf. Can hybrid geometric scattering networks help solve the maximum clique problem? Advances in Neural Information Processing Systems (NeurIPS), 35:22713–22724, 2022.

[46] Vahab Mirrokni, Renato Paes Leme, Adrian Vladu, and Sam Chiu wai Wong. Tight bounds for approximate Carathéodory and beyond. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70, pages 2440–2448. PMLR, 2017.

[47] Konrad Mundinger, Max Zimmer, Aldo Kiem, Christoph Spiegel, and Sebastian Pokutta. Neural discovery in mathematics: Do machines dream of colored planes? In Forty-second International Conference on Machine Learning (ICML), 2025.

[48] Kazuo Murota. Discrete convex analysis. Mathematical Programming, 83(1):313–371, 1998.

[49] George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions - I. Math. Program., 14(1):265–294, 1978.

[50] Robert R Nerem, Zhishang Luo, Akbar Rafiey, and Yusu Wang. Differentiable extensions with rounding guarantees for combinatorial optimization over permutations. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[51] Gilles Pisier. Remarques sur un résultat non publié de b. maurey. Séminaire d’Analyse fonctionnelle (dit" Maurey-Schwartz"), pages 1–12, 1981.

[52] Ruizhong Qiu, Zhiqing Sun, and Yiming Yang. Dimes: A differentiable meta solver for combinatorial optimization problems. Advances in Neural Information Processing Systems (NeurIPS), 35:25531–25546, 2022.

[53] Gerhard Reinelt. Tsplib—a traveling salesman problem library. ORSA journal on computing, 3 (4):376–384, 1991.

[54] Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Approximation ratios of graph neural networks for combinatorial problems. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

[55] Martin JA Schuetz, J Kyle Brubaker, and Helmut G Katzgraber. Combinatorial optimization with physics-inspired graph neural networks. Nature Machine Intelligence, 4(4):367–377, 2022.

[56] Richard Sinkhorn. A relationship between arbitrary positive matrices and doubly stochastic matrices. The annals ofmathematical statistics, 35(2):876–879, 1964.

[57] Haoran Sun, Etash K Guha, and Hanjun Dai. Annealed training for combinatorial optimization on graphs. arXiv preprint arXiv:2207.11542, 2022.

[58] Zhiqing Sun and Yiming Yang. Difusco: Graph-based diffusion solvers for combinatorial optimization. Advances in Neural Information Processing Systems (NeurIPS), 36:3706–3731, 2023.

[59] Zhentao Tan and Yadong Mu. Learning solution-aware transformers for efficiently solving quadratic assignment problem. In Proceedings of the 41st International Conference on Machine Learning (ICML), pages 47627–47648, 2024.

[60] Mohit Tawarmalani and Nikolaos V Sahinidis. Convex extensions and envelopes of lower semi-continuous functions. Mathematical programming, 93(2):247–263, 2002.

[61] Jan Toenshoff, Martin Ritzert, Hinrikus Wolf, and Martin Grohe. Graph neural networks for maximum constraint satisfaction. Frontiers in artificial intelligence, 3:580607, 2021.

[62] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. Pointer networks. In Advances in Neural Information Processing Systems (NeurIPS), pages 2692–2700, 2015.

[63] Haoyu Peter Wang, Nan Wu, Hang Yang, Cong Hao, and Pan Li. Unsupervised learning for combinatorial optimization with principled objective relaxation. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[64] Runzhong Wang, Junchi Yan, and Xiaokang Yang. Neural graph matching network: Learning lawler’s quadratic assignment problem with extension to hypergraph and multiple-graph matching. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5261–5279, 2021.

[65] Runzhong Wang, Li Shen, Yiting Chen, Xiaokang Yang, Dacheng Tao, and Junchi Yan. Towards one-shot neural combinatorial solvers: Theoretical and empirical notes on the cardinalityconstrained case. In The Eleventh International Conference on Learning Representations (ICLR), 2022.

[66] Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang, and Junchi Yan. Linsatnet: the positive linear satisfiability neural networks. In International Conference on Machine Learning (ICML), pages 36605–36625. PMLR, 2023.

[67] David P Williamson and David B Shmoys. The design of approximation algorithms. Cambridge university press, 2011.

[68] Yifan Xia, Xianliang Yang, Zichuan Liu, Zhihao Liu, Lei Song, and Jiang Bian. Position: Rethinking post-hoc search-based neural approaches for solving large-scale traveling salesman problems. In International Conference on Machine Learning (ICML), pages 54178–54190. PMLR, 2024.

[69] Hongtai Zeng, Chao Yang, Yanzhen Zhou, Cheng Yang, and Qinglai Guo. Glinsat: The general linear satisfiability neural network layer by accelerated gradient descent. Advances in Neural Information Processing Systems (NeurIPS), 37:122584–122615, 2024.

[70] Baojian Zhou and Yifan Sun. Approximate Frank–Wolfe algorithms over graph-structured support sets. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pages 27303–27337. PMLR, 2022.

## A Proofs for Theorem 4.3 and Corollary 4.4

Proof of Theorem 4.3. The convex-combination structure follows directly from the update $\mathbf { u } _ { t } ~ =$ $( 1 - \gamma _ { t } ) \mathbf { u } _ { t - 1 } + \gamma _ { t } \mathbf { v } _ { t }$ with $\gamma _ { t } \in [ 0 , 1 ]$ and the initialization $\mathbf { u } _ { 0 } = \mathbf { v } _ { 0 } \in \mathrm { e x t } ( \mathcal { P } )$ . Inductively,

$$
\mathbf { u } _ { T - 1 } = \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } \mathbf { v } _ { t } , \qquad \mathbf { v } _ { t } \in \mathrm { e x t } ( \mathcal { P } ) , \qquad \alpha _ { t } \geq 0 , \qquad \sum _ { t = 0 } ^ { T - 1 } \alpha _ { t } = 1 .
$$

For the convergence rate, let $D = \mathrm { d i a m } ( \mathcal { P } )$ and $\begin{array} { r } { J ( \mathbf { u } ) = \frac { 1 } { 2 } \Vert \mathbf { x } - \mathbf { u } \Vert _ { 2 } ^ { 2 } } \end{array}$ . The gradient of J is 1-Lipschitz, so the Frank–Wolfe curvature of $J$ over $\mathcal { P }$ is bounded by $D ^ { 2 }$ . The standard Frank–Wolfe rate with exact linear minimization/maximization oracles and line search gives, after k updates,

$$
J ( \mathbf { u } _ { k } ) - J ( \mathbf { u } ^ { \star } ) \leq \frac { 2 D ^ { 2 } } { k + 2 } , \qquad \mathbf { u } ^ { \star } \in \arg \operatorname* { m i n } _ { \mathbf { u } \in \mathcal { P } } J ( \mathbf { u } ) .
$$

Setting $k = T - 1$ yields

$$
J ( { \bf u } _ { T - 1 } ) - J ( { \bf u } ^ { \star } ) \leq \frac { 2 D ^ { 2 } } { T + 1 } .
$$

It remains to justify the almost-everywhere differentiability claim under the stable oracle decision assumption. For almost every $\mathbf { x } ,$ the assumption gives a neighborhood on which all selected vertices and line-search clipping regimes are fixed. Fix such an input and neighborhood. The initialization $\mathbf { u } _ { 0 } = \mathbf { v } _ { 0 }$ is constant on this neighborhood. Recursively, if $\mathbf { u } _ { t - 1 } ( \mathbf { x } )$ is smooth there, then

$$
\mathbf { r } _ { t } ( \mathbf { x } ) = \mathbf { x } - \mathbf { u } _ { t - 1 } ( \mathbf { x } ) , \qquad \mathbf { d } _ { t } ( \mathbf { x } ) = \mathbf { v } _ { t } - \mathbf { u } _ { t - 1 } ( \mathbf { x } )
$$

are smooth because $\mathbf { v } _ { t }$ is fixed. When $\mathbf { d } _ { t } ( \mathbf { x } ) \neq 0$ , the unclipped line-search value

$$
\widetilde { \gamma } _ { t } ( \mathbf { x } ) = \frac { \langle \mathbf { r } _ { t } ( \mathbf { x } ) , \mathbf { d } _ { t } ( \mathbf { x } ) \rangle } { \| \mathbf { d } _ { t } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } }
$$

is a smooth algebraic function on a possibly smaller neighborhood with nonzero denominator. When $\mathbf { d } _ { t } = 0$ , we use the algorithmic convention $\gamma _ { t } = 0$ , and the update is locally constant in that direction. Since the clipping regime is fixed on the neighborhood, $\gamma _ { t }$ is either 0, 1, or $\tilde { \gamma } _ { t }$ , and is therefore differentiable there. Hence

$$
\mathbf { u } _ { t } = ( 1 - \gamma _ { t } ) \mathbf { u } _ { t - 1 } + \gamma _ { t } \mathbf { v } _ { t }
$$

involves only differentiable algebraic operations. The same recursive update shows that each coefficient $\alpha _ { t } ( \mathbf { x } )$ is differentiable on the neighborhood. The only excluded inputs lie in the zero-volume switching set from the stable oracle decision assumption, so the coefficient map $\mathbf { x } \mapsto \pmb { \alpha } ( \mathbf { x } )$ is differentiable almost everywhere. □

ProofofCorollary 4.4. The convex-combination conclusion is unchanged because the update stil uses $\gamma _ { t } \in [ 0 , 1 ]$ and vertices of $\mathcal { P } _ { \cdot }$ . The relative FW-gap condition

$$
\langle \mathbf { r } _ { t } , \mathbf { v } _ { t } - \mathbf { u } _ { t - 1 } \rangle \geq \delta \operatorname* { m a x } _ { \mathbf { v } \in \mathcal { P } } \langle \mathbf { r } _ { t } , \mathbf { v } - \mathbf { u } _ { t - 1 } \rangle
$$

is the maximization form of the standard approximate Frank–Wolfe oracle condition. Applying the approximate Frank–Wolfe theorem of Locatello et al. [42] to $\begin{array} { r } { J ( \mathbf { u } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { u } \| _ { 2 } ^ { 2 } } \end{array}$ , with Lipschitz constant 1 and diameter $D ,$ , gives after $T - 1$ updates

$$
J ( \mathbf { u } _ { T - 1 } ) - J ( \mathbf { u } ^ { \star } ) \leq \frac { 2 \left( D ^ { 2 } / \delta + h _ { 0 } \right) } { \delta ( T - 1 ) + 2 } , \qquad h _ { 0 } = J ( \mathbf { u } _ { 0 } ) - J ( \mathbf { u } ^ { \star } ) .
$$

The almost-everywhere differentiability statement follows from the same stable oracle decision argument used above. □

## B Details and Further Experiments for Maximum Coverage

Hardware. All experiments, including training and inference on Maximum Coverage, TSP and QAP, are done on 16 cores (32 threads) of Intel(R) Xeon(R) Platinum 8268 CPU (24 cores, 48 threads in total), 32 GB DDR4 ram, with a single Nvidia RTX8000 48GB GPU.

## B.1 Experiment Setup

Datasets. Following prior work [7, 65, 24], we evaluate on synthetic and real-world bipartite graphs $( U , V , E )$ , where $V$ is the ground set and the task is to select k nodes from V to maximize the total weight of covered nodes in U. For synthetic data, we use the Random Uniform dataset under two scales, Random500 and Random1000. Each setting contains a training dataset of 100 independently generated instances with $| V | = 5 0 0 , | U | = 1 0 \mathrm { \bar { 0 } { 0 } }$ (Random500) or $\mathsf { \bar { \Pi } }$ (Random1000). Each $u \in U$ is assigned an integer weight sampled uniformly from $\{ 1 , \ldots , 1 0 0 \}$ , and each $v \in V$ covers a random subset of U whose cardinality is sampled uniformly from $\{ 1 0 , \ldots , 3 0 \}$ Testing are conducted on datasets sampled from the same distribution. For real-world testing, we additionally evaluate on three Railway instances derived from Italian railway crew assignment graphs: rail507 $( | V | = 5 0 7 , | U | = \dot { 6 } 3 0 0 9 )$ $\mathbf { r a i 1 5 1 6 } \left( \lvert V \rvert = 5 1 6 , \lvert U \rvert = 4 7 \dot { 3 } 1 1 \right)$ , and rail582 $\bar { ( } \lvert V \rvert = 5 8 2 , \lvert U \rvert = \dot { 5 } 5 \dot { 5 } 1 5 )$

Training. In general, we use the same training setup as [24], using a 3-layer GNN with residual connection. To enhance optimization stability and solution quality, we incorporate specific regularization strategies to promote exploration within the hypersimplex. An entropy regularization term is applied with a cosine-decayed weight $\lambda _ { t }$ , which is initialized at 0.05 and annealed to zero by the 30th epoch. A dropout rate of 0.1 is also applied exclusively during the training phase.

The model is trained for a total of 80 epochs using the AdamW optimizer with a weight decay of $1 \times 1 0 ^ { - 4 }$ . We employ a batch size of 4 and a fixed random seed of 42 to ensure reproducibility. The learning rate follows a warmup\_cosine scheduler, which includes 50 warmup epochs and decays from a peak of $5 \times 1 0 ^ { - 3 }$ to a minimum of $5 \times 1 0 ^ { - 5 }$ . Data loading and preprocessing are handled efficiently by 16 CPU worker threads.

Inference. In inference, we run our FW DECOMPOSITION for 50 iterations for each instance, and select the best result out of the decomposed sets. We do not use any randomized or local improvment method.

## B.2 Baselines

We compare against three classes of baselines that span exact solvers, classical heuristics, and neural approaches.

(i) Exact solvers (non-learning). We employ Gurobi to obtain exact MIP-based solutions, imposing a time limit of 120 seconds for each graph.

(ii) Classical heuristics (non-learning). We include a Random baseline, which samples subsets of size k uniformly at random over multiple trials and selects the best solution found within 240 seconds. We also report the Greedy algorithm [49], which iteratively adds the element with the largest marginal gain to achieve the well-known $\begin{array} { r } { ( 1 - \frac { 1 } { \mathrm { e } } ) } \end{array}$ )-approximation guarantee. Additionally, we classify UCOM2 [7] as a non-learning baseline due to its use of incremental greedy de-randomization, and we report results for its short variant.

(iii) Neural baselines. We evaluate a diverse set of neural methods. We include CardNN [65] variants (without Test-Time Optimization, following [7]) and a naive version of EGN [22] with naive probabilistic objective construction and iterative rounding. We also compare against GeoNCO [24], which employs the same Carathéodory decomposition framework as our method but lacks our proposed enhancements. Finally, we include a Reinforcement Learning (RL) baseline following [24] trained with the Actor-Critic [30] algorithm on the same problem instances.

## B.3 Ablation study on projection

We ablate the effect of explicitly projecting the network output back into the feasible solution polytope before running FW DECOMPOSITION.We use the same projection method as [24], while they rely on such a projection to enforce feasibility, our framework does not require it since feasibility is already guaranteed by the decomposition.

Table 4 shows that removing projection preserves solution quality and in fact yields slightly better coverage across both rand500 and rand1000 settings, with slightly lower wall-clock inference time. We attribute this to the fact that the projection is a hard, piecewise operator that can compress the effective search region and reduce gradient signal diversity; in our pipeline, it becomes largely redundant because feasibility is enforced by the decomposition itself. By contrast, allowing unconstrained continuous scores retains a richer exploration space for the model, while FW DECOMPOSITION still produces feasible candidates, leading to steadier optimization and marginally improved performance in practice.

Table 4: Ablation study results. Grouped by dataset size.
<table><tr><td rowspan="2">Method</td><td colspan="2">rand500</td><td colspan="2">rand1000</td></tr><tr><td>k = 10</td><td>k = 50</td><td>k = 20</td><td>k = 100</td></tr><tr><td rowspan="2">With Projection</td><td>15190.10</td><td>42138.45</td><td>29906.24</td><td>83474.98</td></tr><tr><td>0.4635s</td><td>0.5513s</td><td>0.6423s</td><td>0.9940s</td></tr><tr><td rowspan="2">No Projection</td><td>15192.64</td><td>42172.44</td><td>29926.33</td><td>83559.22</td></tr><tr><td>0.4261s</td><td>0.5378s</td><td>0.6472s</td><td>0.9135s</td></tr></table>

## B.4 Ablation on Hard Instances for Greedy

Hard example for greedy. The greedy algorithm performs well, especially on random graphs, for the MC problem because the coverage function is monotone and submodular. To show our method works even when greedy fails, we construct the GREEDYTRAP500 and GREEDYTRAP1000 datasets for MC. In this construction, the greedy algorithm is pushed near its theoretical 1 − 1/e = 0.63212 lower bound and attracted by sets with large initial marginal coverage, falling significantly short of the optimal solution.

The construction is as follows. We first choose a coverable subset of elements from U and partition it into k disjoint parts. For each part, we add one set that covers the entire part; selecting these k sets covers all coverable elements. We then add k greedy-trap sets. Each trap set covers slightly more than a 1/k fraction of every part, so these sets have large marginal gain at the beginning and are selected by greedy. The remaining sets are low-degree filler sets. For GREEDYTRAP500, we use 500 sets and 1000 elements; for GREEDYTRAP1000, we use 1000 sets and 2000 elements. In testing, we report the coverage ratio over all elements.

Table 5: Performance comparison on the GREEDYTRAP datasets. Values represent coverage ratios in percentage averaged across 64 test instances.
<table><tr><td rowspan="2">Method</td><td colspan="2">GreedyTrap500</td><td colspan="2">GreedyTrap1000</td></tr><tr><td>k = 10</td><td>k = 50</td><td>k = 20</td><td>k = 100</td></tr><tr><td>Greedy</td><td>46.30%</td><td>66.50%</td><td>45.80%</td><td>70.60%</td></tr><tr><td>FWNCO(ours)</td><td>59.90%</td><td>78.40%</td><td>55.60%</td><td>82.30%</td></tr><tr><td>OPT</td><td>60.10%</td><td>81.90%</td><td>58.60%</td><td>86.70%</td></tr></table>

As shown in Table 5, greedy falls significantly short of OPT on these hard instances, whereas our method consistently outperforms greedy and achieves coverage much closer to OPT. This shows that our method can still learn useful problem structure and maintain strong performance on data distributions where the standard greedy heuristic fails.

## B.5 Effect of the Frank–Wolfe Iteration Budget

We investigate the effect of the number of Frank–Wolfe decomposition iterations T on solution quality and runtime for Maximum Coverage. Each iteration of Algorithm 1 performs one LMO call and a closed-form line-search step. Note that the runtime is linear in T.

Table 6 reports results for four configurations, with solution quality normalized by the weighted coverage obtained at $T = 2 0 0$ . Increasing T consistently improves quality, with diminishing gains at larger budgets. At $T = 5 0$ , the method already achieves 99.391–99.652% of the reference quality across all four configurations. Increasing the budget to $T = 2 0 0$ improves relative quality by only 0.348–0.609 percentage points, while increasing the reported runtime by approximately $3 . 7 \mathrm { - 3 . 8 \times }$ These results support our choice of $T = 5 0$ as a practical balance between solution quality and computational cost.

Table 6: Maximum coverage quality and inference time under different FW iteration budgets. Quality is normalized by the weighted coverage obtained at $T = 2 0 0$ and reported as a percentage. Time recorded on an NVIDIA GeForce RTX 4090 GPU and not directly comparable with other results.
<table><tr><td rowspan="2">T</td><td colspan="2">n=500, k=10</td><td colspan="2">n=500, k=50</td><td colspan="2">n=1000, k=20</td><td colspan="2">n=1000, k=100</td></tr><tr><td>Quality (%)</td><td>Time (s)</td><td>Quality (%)</td><td>Time (s)</td><td>Quality (%)</td><td>Time (s)</td><td>Quality (%)</td><td>Time (s)</td></tr><tr><td>10</td><td>98.197</td><td>0.0802</td><td>99.230</td><td>0.1153</td><td>98.712</td><td>0.1418</td><td>99.464</td><td>0.1965</td></tr><tr><td>30</td><td>98.942</td><td>0.2046</td><td>99.376</td><td>0.2853</td><td>99.296</td><td>0.3556</td><td>99.548</td><td>0.4920</td></tr><tr><td>50</td><td>99.391</td><td>0.3295</td><td>99.494</td><td>0.4554</td><td>99.501</td><td>0.5669</td><td>99.652</td><td>0.7852</td></tr><tr><td>70</td><td>99.470</td><td>0.4516</td><td>99.630</td><td>0.6246</td><td>99.601</td><td>0.7784</td><td>99.755</td><td>1.0770</td></tr><tr><td>100</td><td>99.659</td><td>0.6333</td><td>99.771</td><td>0.8775</td><td>99.695</td><td>1.0889</td><td>99.861</td><td>1.5150</td></tr><tr><td>150</td><td>99.859</td><td>0.9345</td><td>99.909</td><td>1.2994</td><td>99.848</td><td>1.5979</td><td>99.930</td><td>2.2447</td></tr><tr><td>200</td><td>100.000</td><td>1.2365</td><td>100.000</td><td>1.7197</td><td>100.000</td><td>2.1068</td><td>100.000</td><td>2.9653</td></tr></table>

## C Details and Further Experiments on QAP

In this section, we provide a detailed discussion of the numerical results for QAP. The full experimental results are reported in Table 7.

## C.1 Experiment Setup

Instance distribution and data splits. For synthetic training, we follow the data generation method of [59] and generate a synthetic training pool with 5120 instances, $n = 3 2$ and $p = 0 . 3$ . During training, each gradient step samples a mini-batch uniformly from the training pool.

For test, we use the QAPLIB dataset to demostrate the generalization ability of our method. QAPLIB [8] is a widely used benchmark for QAP, collecting real-world and synthetic instances from multiple sources. Each instance is specified by a flow matrix and a distance matrix, and the naming convention uses a class prefix (e.g., bur, chr, tai) together with the problem size. The distributions and structural properties of the flow/distance matrices vary substantially across classes, making QAPLIB a challenging testbed for cross-class generalization.

Input representation. Given A and B, we build a node embedding for each matrix separately using per-node statistics (diagonal entry and row/column aggregations) together with random-projection features computed with a fixed Gaussian projection matrix R. Features are computed after perinstance magnitude normalization (i.e., dividing by ma $_ { \mathrm { X } _ { i , j } } \left| M _ { i j } \right| )$ to stabilize training across varying scales.

Table 7: Complete experiment results on QAPLIB. The mean/max/min gaps are reported for each category. The mean performance over all categories and the inference time (s) per instance are reported.
<table><tr><td rowspan="2"></td><td colspan="2">bur (26)</td><td rowspan="2"></td><td colspan="3">chr(12-25)</td><td colspan="2">esc (16-64)</td><td colspan="3">had (12 - 20)</td></tr><tr><td>Mean↓</td><td>Min↓ Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean</td><td>Min↓ Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td></tr><tr><td>SM</td><td>22.3</td><td>20.3</td><td>24.9</td><td>460.1</td><td>144.6</td><td>869.1</td><td>301.6 0.0</td><td>3300.0</td><td>17.4</td><td>14.7</td><td>21.5</td></tr><tr><td>RRWM</td><td>23.1</td><td>19.3</td><td>27.3</td><td>616.0</td><td>120.5</td><td>1346.3</td><td>63.9 0.0</td><td>200.0</td><td>25.1</td><td>22.1</td><td>28.3</td></tr><tr><td>SK-JA</td><td>4.7</td><td>2.8</td><td>6.2</td><td>38.5</td><td>0.0</td><td>186.1</td><td>364.8 0.0</td><td>2200.0</td><td>25.8</td><td>6.9</td><td>100.0</td></tr><tr><td>NGM</td><td>3.4</td><td>2.8</td><td>4.4</td><td>121.3</td><td>45.4</td><td>251.9</td><td>126.7 0.0</td><td>200.0</td><td>8.2</td><td>6.0</td><td>11.6</td></tr><tr><td>RGM</td><td>7.1</td><td>4.5</td><td>9.0</td><td>112.4</td><td>23.4</td><td>361.4</td><td>32.8 0.0</td><td>141.5</td><td>6.2</td><td>1.9</td><td>9.0</td></tr><tr><td>SAWT</td><td>2.8</td><td>2.2</td><td>3.4</td><td>110.7</td><td>8.6</td><td>201.2</td><td>13.5 0.0</td><td>63.7</td><td>3.8</td><td>1.6</td><td>6.5</td></tr><tr><td>FWNCO (ours)</td><td>2.3</td><td>1.9</td><td>2.6</td><td>97.7</td><td>28.8</td><td>209.9</td><td>19.1 0.0</td><td>119.1</td><td>2.8</td><td>1.2</td><td>3.7</td></tr><tr><td></td><td colspan="3">kra (30-32)</td><td colspan="3">lipa (20-60)</td><td colspan="2">nug (12-30)</td><td colspan="3">rou (12-30)</td></tr><tr><td></td><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean↓ Min↓</td><td>Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td></tr><tr><td>SM</td><td>65.3</td><td>63.8</td><td>67.3</td><td>19.0</td><td>3.8</td><td>34.8</td><td>45.5 34.2</td><td>64.0</td><td>35.8</td><td>30.9</td><td>38.2</td></tr><tr><td>RRWM</td><td>58.8</td><td>53.9</td><td>67.7</td><td>20.9</td><td>3.6</td><td>41.2</td><td>67.8 52.6</td><td>79.6</td><td>51.2</td><td>39.3</td><td>60.1</td></tr><tr><td>SK-JA</td><td>41.4</td><td>38.9</td><td>44.4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>25.3 10.9</td><td>100.0</td><td>13.7</td><td>10.3</td><td>17.4</td></tr><tr><td>NGM</td><td>31.6</td><td>28.7</td><td>36.8</td><td>16.2</td><td>3.6</td><td>29.4</td><td>21.0 14.0</td><td>28.5</td><td>30.9</td><td>23.7</td><td>36.3</td></tr><tr><td>RGM</td><td>15.0</td><td>10.4</td><td>20.6</td><td>13.3</td><td>3.0</td><td>23.8</td><td>9.7 6.1</td><td>12.9</td><td>13.4</td><td>7.1</td><td>16.7</td></tr><tr><td>SAWT</td><td>30.1</td><td>28.1</td><td>34.2</td><td>0.4</td><td>0.0</td><td>2.6</td><td>9.2 4.1</td><td>13.1</td><td>10.8</td><td>8.2</td><td>13.1</td></tr><tr><td>FWNCO (ours)</td><td>37.0</td><td>34.8</td><td>39.2</td><td>13.5</td><td>2.2</td><td>26.3</td><td>11.5 7.3</td><td>14.7</td><td>10.9</td><td>9.7</td><td>12.1</td></tr><tr><td rowspan="2"></td><td colspan="3">scr (12-20)</td><td colspan="3">sko(42-64)</td><td colspan="2">ste(36)</td><td colspan="3">tai(12-64)</td></tr><tr><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean↓</td><td>Min↓ Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td></tr><tr><td>SM</td><td>123.4</td><td>104.0</td><td>139.1</td><td>29.0</td><td>26.6</td><td>31.4</td><td>475.5 197.7</td><td>1013.6</td><td>180.5</td><td>21.6</td><td>1257.9</td></tr><tr><td>RRWM</td><td>173.5</td><td>98.9</td><td>218.6</td><td>48.5</td><td>47.7</td><td>49.3</td><td>539.4 249.5</td><td>1117.8</td><td>197.2</td><td>26.8</td><td>1256.7</td></tr><tr><td>SK-JA</td><td>48.6</td><td>44.3</td><td>55.7</td><td>18.3</td><td>16.1</td><td>20.5</td><td>120.4 72.5</td><td>200.4</td><td>25.2</td><td>1.6</td><td>107.1</td></tr><tr><td>NGM</td><td>55.5</td><td>41.4</td><td>66.2</td><td>25.2</td><td>22.8</td><td>27.7</td><td>101.7 57.6</td><td>172.8</td><td>61.4</td><td>18.7</td><td>352.1</td></tr><tr><td>RGM</td><td>45.5</td><td>30.2</td><td>56.1</td><td>10.6</td><td>9.9</td><td>11.2</td><td>134.1 69.9</td><td>237.0</td><td>17.3</td><td>11.4</td><td>28.6</td></tr><tr><td>SAWT</td><td>28.5</td><td>10.2</td><td>48.0</td><td>17.7</td><td>16.7</td><td>18.7</td><td>93.5 46.7</td><td>170.2</td><td>16.5</td><td>0.5</td><td>48.7</td></tr><tr><td>FWNCO (ours)</td><td>23.6</td><td>13.6</td><td>39.8</td><td>15.0</td><td>13.8</td><td>15.6</td><td>83.5 58.5</td><td>130.2</td><td>20.7</td><td>1.7</td><td>43.0</td></tr><tr><td rowspan="2"></td><td colspan="3">tho (30-40)</td><td colspan="3">wil(50)</td><td colspan="3">Average(12-64)</td><td colspan="2">Time per instance</td></tr><tr><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>Mean↓</td><td>Min↓</td><td>Max↓</td><td>(in seconds)</td><td></td></tr><tr><td>SM</td><td>55.0</td><td>54.0</td><td>56.0</td><td>13.8</td><td>11.7</td><td>15.9</td><td>181.2</td><td>46.9 949.9</td><td></td><td>0.01</td><td></td></tr><tr><td>RRWM</td><td>80.6</td><td>78.2</td><td>83.0</td><td>18.2</td><td>12.5</td><td>23.8</td><td>169.5</td><td>49.5</td><td>432.9</td><td>0.15</td><td></td></tr><tr><td>SK-JA</td><td>32.9</td><td>30.6</td><td>35.3</td><td>8.8</td><td>6.7</td><td>10.7</td><td>93.2</td><td>9.0</td><td>497.9</td><td>563.4</td><td></td></tr><tr><td>NGM</td><td>27.5</td><td>24.8</td><td>30.2</td><td>10.8</td><td>8.2</td><td>11.1</td><td>62.4</td><td>17.8</td><td>129.7 101.1</td><td>15.72 75.53</td><td></td></tr><tr><td>RGM SAWT</td><td>20.7 24.8</td><td>12.7 23.2</td><td>28.6 26.4</td><td>8.1 8.1</td><td>7.9 7.6</td><td>8.4 8.6</td><td>35.8 10.7 26.8 11.2</td></table>

Model, optimizer, and training schedule. We use a two-layer GraphSAGE aggregator with a residual skip connection and hidden size model\_dim=128, applied independently to A and B. The neural network outputs a matrix followed by a Sinkhorn projection layer. We train with AdamW (learning rate 3 · 10<sup>−4</sup>, weight decay 10<sup>−4</sup>) and gradient clipping at norm 1.0. Training runs for 100 epochs; each epoch consists of 100 instances randomly sampled from the training pool. A single global RNG seed (0) is used for both training and inference.

Sinkhorn as a practical projection layer. Although our FW-based pipeline does not theoretically require the network output to lie inside the Birkhoff polytope (the FW DECOMPOSITION can be applied to any real matrix), in QAP this relaxation is empirically fragile because QAP instances are substantially more heterogeneous than the other tasks: QAPLIB in particular exhibits highly dispersed scales, sparsity patterns, and instance-generation mechanisms across data classes, leading to large distribution gaps relative to synthetic pretraining. In this regime, leaving the model output unconstrained often yields score matrices whose magnitude and row/column mass distribution are poorly calibrated, which in turn destabilizes the downstream greedy steps and makes the distribution returned by FW DECOMPOSITION less informative.

To mitigate this, we insert a Sinkhorn [56] projection layer with 20 iterations as a soft projection that maps raw assignment logits to an approximately doubly-stochastic matrix. This operation enforces nonnegativity and approximately unit row/column sums, placing X, the output of neural network, closer to the Birkhoff polytope. It serves as a lightweight output-range calibration mechanism that significantly improves numerical stability under the large inter-class distribution shift in QAPLIB. We emphasize that this Sinkhorn step is used for practical robustness, not as a theoretical requirement of the FW DECOMPOSITION itself.

Decomposition settings. We run our FW DECOMPOSITION algorithm over permutation matrices with a iteration budget T = 2000 in training and T = 3000 in inference.

At each iteration, the LMO is instantiated by a greedy maximum-weight assignment routine (no Hungarian), applied to the current residual matrix. The resulting decomposition yields a sequence of permutations and their weights; in inference, we select the minimum-cost permutation under the QAP objective for reporting.

Table 8: QAPLIB results for instances with n > 64.
<table><tr><td>Instance</td><td>pred_best</td><td>best_known</td><td>gap</td></tr><tr><td>lipa70a</td><td>172973</td><td>169755</td><td>1.90%</td></tr><tr><td>lipa70b</td><td>5872840</td><td>4603200</td><td>27.58%</td></tr><tr><td>sko72</td><td>75952</td><td>66256</td><td>14.63%</td></tr><tr><td>lipa80a</td><td>257769</td><td>253195</td><td>1.81%</td></tr><tr><td>lipa80b</td><td>9983972</td><td>7763962</td><td>28.59%</td></tr><tr><td>tai80b</td><td>1057836224</td><td>818415043</td><td>29.25%</td></tr><tr><td>tai80a</td><td>15311664</td><td>13499184</td><td>13.43%</td></tr><tr><td>sko81</td><td>104214</td><td>90998</td><td>14.52%</td></tr><tr><td>lipa90a</td><td>366329</td><td>360630</td><td>1.58%</td></tr><tr><td>lipa90b</td><td>16017197</td><td>12490441</td><td>28.24%</td></tr><tr><td>sko90</td><td>132078</td><td>115534</td><td>14.32%</td></tr><tr><td>sko100a</td><td>174044</td><td>152002</td><td>14.50%</td></tr><tr><td>sko100b</td><td>174378</td><td>153890</td><td>13.31%</td></tr><tr><td>sko100c</td><td>170276</td><td>147862</td><td>15.16%</td></tr><tr><td>sko100d</td><td>170574</td><td>149576</td><td>14.04%</td></tr><tr><td>sko100e</td><td>170746</td><td>149150</td><td>14.48%</td></tr><tr><td>sko100f</td><td>169558</td><td>149036</td><td>13.77%</td></tr><tr><td>tai100a</td><td>23638054</td><td>21052466</td><td>12.28%</td></tr><tr><td>tai100b</td><td>1498196608</td><td>1185996137</td><td>26.32%</td></tr><tr><td>wil100</td><td>296136</td><td>273038</td><td>8.46%</td></tr><tr><td>esc128</td><td>244</td><td>64</td><td>281.25%</td></tr><tr><td>tai150b</td><td>619556288</td><td>498896643</td><td>24.19%</td></tr><tr><td>tho150</td><td>9579958</td><td>8133398</td><td>17.79%</td></tr><tr><td>tai256c</td><td>50753096</td><td>44759294</td><td>13.39%</td></tr><tr><td></td><td></td><td>Average gap</td><td>26.87%</td></tr></table>

QAPLIB evaluation. Following previous approaches [59, 41], We report per-class gap statistics and the aggregate summary as in Table 7 for instances of size 12-64 since many baseline methods will run out of time on very large instances. However, our method is also capable of solving instances with larger sizes, as listed below in Table 8.

## C.2 Related Work (QAP / Matching).

QAP and matching are among the most challenging CO problems for learning-based methods, largely due to severe distribution shifts and outliers such as those in QAPLIB. NGM [64] was one of the earliest attempts to tackle QAP/graph matching with neural models. Subsequent work proposed RGM [41], which introduced a learn-to-construct (L2C) pipeline for QAP, building a permutation by making a sequence of constructive decisions, following that, SAWT [59] introduced a learnto-improve (L2I) pipeline, which starts from an initial permutation and repeatedly applies local modifications to improve it.

Both L2C and L2I are inherently iterative: they start from an empty/partial solution or an initial solution and improve it step by step. This makes inference naturally less efficient and less friendly to large instances, and robustness to outliers is achieved through relatively complex neural architectures and nontrivial inference procedures.

Our general framework applies to QAP in a single-pass direct-output manner: we produce assignment logits once and deterministically output a permutation without repeated optimization loops, yielding an immediate speed advantage. Moreover, with only a lightweight Sinkhorn normalization layer added to the base framework, we obtain outlier-friendly behavior at very low additional cost.

## C.3 Baselines

We compare against two classes of baselines that cover both classical matching solvers and representative neural methods on QAPLIB.

(i) Classical solvers (non-learning). We include SM, RRWM, and SK-JA [33] as standard nonlearning baselines for graph matching / QAP.

(ii) Neural methods. We evaluate NGM, RGM, and SAWT as representative learning-based baselines on QAPLIB.

We follow the same per-class reporting protocol as prior QAPLIB evaluations, and report the mean/min/max optimality gaps for each QAPLIB class together with the average per-instance inference time. As the public releases of both SAWT and RGM do not include a complete evaluation setup and code for QAPLIB, we are unable to reproduce their experiments locally. Instead, we directly cite their reported timing results. Consequently, due to hardware disparities, these runtime comparisons should be interpreted as indicative references rather than strictly controlled benchmarks. We also note an additional caveat in their QAPLIB results: for the WIL class they report multiple results, whereas within range $( n \in [ 1 2 , 6 4 ] )$ ) where they reported, WIL contains only a single instance (WIL50). Since we cannot verify the exact evaluation protocol (e.g., whether additional variants or sizes were included), we therefore quote their reported numbers in full without further reinterpretation.

## C.4 Effect of the Frank–Wolfe Iteration Budget

We evaluate the sensitivity of QAP solution quality and runtime to the Frank–Wolfe decomposition budget (T) on QAPLIB. Table 9 reports relative quality normalized by the weighted solution obtained at $T = 1 0 0 0 0$ , together with runtime for each budget.

Solution quality improves substantially as the budget increases from $T = 5 0 \ : \mathrm { t o } \ : T = 1 0 0 0$ , rising from 94.247% to 99.237% of the reference quality. Further iterations yield progressively smaller improvements. At our chosen budget of T = 3000, relative quality reaches 99.837%, with a runtime of 4.410 seconds. Increasing the budget to T = 10000 adds only 0.163 percentage points of relative quality while increasing runtime to 15.798 seconds, approximately 3.6× higher. We therefore use $\bar { T } = \dot { 3 } 0 0 0$ to balance solution quality and runtime.

Compared with Maximum Coverage, QAP empirically benefits from a substantially larger decomposition budget. The approximation guarantee in Theorem 4.3 provides qualitative context: the squared approximation error is bounded by $4 D ^ { 2 } / ( T + 1 )$ , where D is the diameter of the feasible polytope. Our choice of T is guided by this and the observed quality–runtime trade-off.

Table 9: QAP solution quality under different FW iteration budgets on QAPLIB. Results are normalized by the weighted solution obtained at $\mathrm { T } = 1 0 0 0 0$ . Time recorded on an NVIDIA GeForce RTX 4090 GPU and not directly comparable with other results.
<table><tr><td>FW budget T</td><td>Relative quality (%)</td><td>Time (s)</td></tr><tr><td>50</td><td>94.247</td><td>0.054</td></tr><tr><td>100</td><td>95.632</td><td>0.105</td></tr><tr><td>200</td><td>96.825</td><td>0.216</td></tr><tr><td>500</td><td>98.595</td><td>0.594</td></tr><tr><td>1000</td><td>99.237</td><td>1.304</td></tr><tr><td>2000</td><td>99.672</td><td>2.828</td></tr><tr><td>3000</td><td>99.837</td><td>4.410</td></tr><tr><td>4000</td><td>99.893</td><td>6.017</td></tr><tr><td>5000</td><td>99.898</td><td>7.639</td></tr><tr><td>7500</td><td>99.921</td><td>11.707</td></tr><tr><td>10000</td><td>100.000</td><td>15.798</td></tr></table>

## D Details and Further Experiments on TSP

## D.1 Experiment Setup

Instance distribution and data splits. We train and evaluate on the 2D Euclidean TSP, where each instance consists of n points in [0, 1]<sup>2</sup> and tour length is computed under standard Euclidean distances. We use the training and evaluation data from COExpander [44]. During training, for a given n, at each epoch, 64 instances are sampled from the training pool, and different data are used across epochs. The test split has 1280 instances for TSP-50 and TSP-100, 128 instances for TSP-500 and 32 instances for TSP-1000, so that most methods could complete the evaluation in reasonable time frame.

Graph construction and input features. For each instance we construct the complete graph on n nodes (undirected edges enumerated with $u < v )$ , and represent it in PyG using the standard bidirected encoding (each undirected edge appears in both directions in edge\_index). Node features are the raw coordinates augmented with a Fourier positional encoding (with $F { = } 4$ frequencies) passed through a small MLP before message passing. Edge features consist of a single scalar channel: the LP relaxation value $x _ { \mathrm { L P } } ( e )$ for the undirected edge, duplicated across both directed copies.

LP relaxation feature computation. We compute $x _ { \mathrm { L P } }$ by solving the subtour-elimination LP using the QSopt LP solver, which is also the LP solver used by Concorde. Edge costs are obtained from Euclidean distances and scaled to integer weights using a fixed coordinate scale of $1 0 ^ { 6 }$ for the QSopt interface. LP solves are performed on CPU for every instance in both train and test splits.

Model, optimizer, and training schedule. The encoder is a GATv2 model with residual blocks: hidden size 128, 3 message-passing layers, and 4 attention heads per layer. We train with AdamW with a learning rate of $1 0 ^ { - 3 }$ and a weight decay of $1 0 ^ { - 4 }$ for 50 epochs, using a single global RNG seed (0) for reproducibility. The best model is saved and used for inference.

Decomposition hyperparameters used in training and inference. We use a fixed decomposition budget of $T = 6 4$ iterations in both training and testing, and a max-entropy regularization with temperature fw\_entropy\_tau= 2.0 is added to the decomposition. All reported TSP test results use the same hyperparameters as training and the reported result is the best permutation from the decomposition.

## D.2 Related Work: Neural Methods for TSP

Learning paradigms: LC vs. GP. Neural approaches for TSP are often categorized by the granularity at which they make decisions. According to [44], the Learning-to-Construct (LC) paradigm builds a tour sequentially, repeatedly selecting the next decision conditioned on the current partial solution. A key strength of LC is that feasibility can be hard-enforced during decoding (e.g., by imposed masks that rule out invalid moves), which provides a transparent and reliable path to valid tours. However, this same sequential structure typically incurs high inference latency on large instances, since the model must execute many dependent steps.

In contrast, the Global Prediction (GP) paradigm predicts a globally complete structure in one (or a few) forward passes, commonly as an edge-probability heatmap, and then applies a separate recovery procedure to obtain an integer-feasible tour. GP can be substantially faster at inference and has become a dominant template for large-scale settings, including diffusion- and consistency-based solvers that generate global solutions through iterative denoising. At the same time, GP shifts difficulty into the heatmap-to-solution stage: the final performance can depend heavily on the specific decoding/recovery algorithm $( \mathrm { e . g . }$ , Parallel Sampling, Greedy, Monte Carlo Tree Search (MCTS)) and additional improvement heuristics (e.g., 2-opt), making the pipeline sensitive to post-processing choices and sometimes blurring what is being evaluated (the neural predictor vs. the downstream solver), since heatmap recovery is itself a combinatorial optimization subroutine.

Adaptive Expansion (AE) and the remaining heatmap bottleneck. COExpander introduces Adaptive Expansion (AE) as an intermediate paradigm intended to combine the best of both worlds: it uses a global predictor to produce a heatmap under partial-solution prompting, and then expands the determined set via a determination/decoding operator, effectively adapting the decision granularity across iterations. While AE improves the efficiency–feasibility trade-off relative to pure LC or pure GP, it still inherits a core limitation of GP-style pipelines: it relies on predicting a heatmap and then recovering an integer tour via post-processing. So overall success remains coupled to the robustness and tuning of heatmap-based recovery.

Toward end-to-end pipelines with reduced problem-specific setup. Xia et al. [68] critically examine the prevalent heatmap-guided post-hoc search paradigm for large-scale TSP (notably heatmap-guided MCTS), questioning the practical value of learning heatmaps that ultimately require substantial downstream search and advocating a shift toward more autonomous and generalizable ML pipelines with less hand-crafted machinery and stronger guarantees. Motivated by this perspective, our method aims to retain LC’s feasibility discipline and GP’s efficiency while avoiding the heatmap bottleneck: we adopt an end-to-end pipeline that differentiates through the selected FW branch, with the optimization objective aligned with the prediction target—we do not learn a surrogate heatmap and then rely on a separate recovery/search step, but instead directly optimize for improved feasible tours through a structured, deterministic procedure. This design reduces reliance on hand-crafted post-processing, improves robustness across instance distributions, and enables stronger theoretical characterization of the resulting algorithmic guarantees.

## D.3 Baselines

We compare against four classes of baselines that span both classical optimization and representative neural paradigms under different supervision regimes.

(i) Exact solvers (non-learning). We report Concorde as an exact TSP solver, and the Gurobi optimizer as an exact baseline on small instances. For larger instances, exact results are omitted when the solver fails to finish within the runtime cutoff.

(ii) Classical heuristics (non-learning). We include LKH3 [20] as a strong hand-engineered heuristic, run for 500 trials.

(iii) Neural LC baselines. We evaluate RL4CO (SymNCO) [5] as a representative Learningto-Construct (LC) method trained with reinforcement learning. LC-style solvers typically decode solutions sequentially and rely on imposed masks (or equivalent constraints) to hard-guarantee feasibility during construction.

(iv) Neural GP/AE baselines. We test heatmap-driven pipelines in both RL-style and supervised regimes. DIMES [52] is included as a representative GP-style method trained via meta-RL. On the supervised side, we include DIFUSCO [58], a diffusion-based GP approach, and COExpander, a heatmap-based Adaptive Expansion (AE) method that achieves state-of-the-art performance.

Overall, under a fixed compute budget, we aim to test the landscape as comprehensively as possible across (a) paradigms (LC vs. GP vs. AE) and (b) supervision regimes (RL / meta-RL vs. supervised learning), while transparently reporting any omissions caused by solver timeouts or training-cost constraints.

## D.4 Detailed Results

Runtime breakdown. In Table 11, we measure the average per-instance wall-clock cost of each stage in the TSP pipeline . The Subtour-Elimination LP and FW DECOMPOSITION dominates the runtime; in the learned matching version, the matching takes up a large portion of the overall runtime.

Generalization Experiments (Across Instance Sizes). We further evaluate how well a model trained on one problem size transfers to other sizes. Specifically, we train three separate models on TSP-100, TSP-500, and TSP-1000, respectively, and test each model on the same evaluation splits at sizes {100, 500, 1000}. Table 12 reports the resulting tour lengths.

Overall, we observe strong cross-size generalization: performance is largely stable when transferring between sizes, with only modest degradation when training on smaller instances and testing on larger ones. For example, a model trained on TSP-100 remains competitive on TSP-500/1000, albeit with a small gap relative to models trained on the target size. Conversely, models trained on larger instances transfer well to smaller instances, with only minor differences on TSP-100. These results suggest that the learned scoring function captures size-agnostic geometric regularities of Euclidean TSP, enabling effective transfer across instance scales without retraining on each target size.

Table 10: Raw performance comparison on TSP instances. We report the objective value (Obj.) and inference time (s).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="2">TSP-50</td><td colspan="2">TSP-100</td><td colspan="2">TSP-500</td><td colspan="2">TSP-1000</td></tr><tr><td>Obj.</td><td>Time</td><td>Obj.</td><td>Time</td><td>Obj.</td><td>Time</td><td>Obj.</td><td>Time</td></tr><tr><td>Concorde (Optimal)</td><td>Exact</td><td>5.69</td><td>0.04</td><td>7.76</td><td>0.21</td><td>16.55</td><td>17.91</td><td>23.12</td><td>435.61</td></tr><tr><td>Gurobi (Optimal)*</td><td>Exact</td><td>5.69</td><td>0.27</td><td>7.76</td><td>1.14</td><td></td><td></td><td></td><td></td></tr><tr><td>LKH3 (500)</td><td>Heuristics</td><td>5.69</td><td>0.03</td><td>7.76</td><td>0.05</td><td>16.70</td><td>0.32</td><td>23.41</td><td>1.06</td></tr><tr><td>DIMES (RL+S)</td><td>RL</td><td>6.47</td><td>0.05</td><td>8.66</td><td>0.08</td><td>19.09</td><td>0.39</td><td>26.35</td><td>0.79</td></tr><tr><td>RL4CO (Sym-NCO)†</td><td>RL</td><td>5.76</td><td>0.36</td><td>8.83</td><td>0.66</td><td></td><td></td><td></td><td></td></tr><tr><td>DIFUSCO (S=1, I=50)</td><td>SL</td><td>5.72</td><td>0.43</td><td>7.86</td><td>0.66</td><td>18.17</td><td>2.19</td><td>25.64</td><td>7.68</td></tr><tr><td>COExpander  $( \mathrm { S } { = } 1 , \mathrm { D } \mathrm { s } { = } 3 , \mathrm { I s } { = } 5 )$ </td><td>SL</td><td>5.69</td><td>0.11</td><td>7.76</td><td>0.20</td><td>17.17</td><td>0.69</td><td>24.63</td><td>2.50</td></tr><tr><td>Ours-lp</td><td>UL</td><td>5.79</td><td>0.23</td><td>8.03</td><td>0.26</td><td>17.76</td><td>1.27</td><td>25.03</td><td>4.26</td></tr><tr><td>Ours-lp+learned matching</td><td>UL</td><td>5.77</td><td>0.52</td><td>7.98</td><td>0.74</td><td>17.70</td><td>2.77</td><td>24.84</td><td>9.16</td></tr></table>

<sup>∗</sup> Results on larger instances are omitted since it was unable to finish within the cutoff time.  
<sup>†</sup> Results are omitted due to prohibitive computational costs during training on large datasets.

<table><tr><td>Stage Time (s/inst)</td></tr><tr><td>LP (Subtour-Elimination) Neural Network</td></tr><tr><td>3.09 0.06</td></tr><tr><td>FW over trees 0.93</td></tr><tr><td>Greedy matching 0.18</td></tr><tr><td>Learned Matching 4.23</td></tr></table>

Table 11: Average per-instance time on TSP1000.

Table 12: Generalization performance across different instance sizes.
<table><tr><td rowspan="2">Training Set</td><td colspan="3">Testing Set</td></tr><tr><td>TSP-100</td><td>TSP-500</td><td>TSP-1000</td></tr><tr><td>TSP-100</td><td>8.03</td><td>17.85</td><td>25.28</td></tr><tr><td>TSP-500</td><td>8.09</td><td>17.76</td><td>25.06</td></tr><tr><td>TSP-1000</td><td>8.16</td><td>17.81</td><td>25.03</td></tr></table>

## D.5 Discussion on Learned Matching

Motivation and positioning. In our TSP pipeline, each candidate spanning tree produced by the FW decomposition is rounded into a tour via the Christofides procedure. A key step in Christofides is computing a perfect matching on the odd-degree vertex set induced by the tree. Our default implementation uses a greedy matching routine for efficiency, while the learned-matching variant replaces this step with a trainable module that is optimized end-to-end for the TSP objective, rather than supervised to recover an exact minimum-weight perfect matching.

Matching as another structured polytope component. For a given tree, let O be its odd-degree vertex set (with |O| even), and let $E _ { O }$ denote all unordered pairs in O, which forms a structured polytope. The matching network outputs a vector $S \in ( 0 , 1 ) ^ { | E _ { O } | }$ over $E _ { O }$ . We then run our FW DECOMPOSITION procedure on the matching polytope over O to obtain a short convex decomposition A<sub>match</sub> $\approx \textstyle \sum _ { k } \beta _ { k } { \bar { M } } _ { k }$ , where each $M _ { k }$ is an integral perfect matching (an extreme point of the polytope), $\beta _ { k } \ge 0 ,$ , and $\textstyle \sum _ { k } \beta _ { k } = 1$

Joint objective (learning matchings for tours). Training couples the spanning-tree and matching components through the downstream tour length. Concretely, the TSP model produces X which is then decomposed through FW DECOMPOSITION over the spanning-tree polytope, and that yields a short list of trees with weights {α }. For each tree t, we form its odd set $O _ { t } ,$ which then goes through the neural network which outputs $S _ { t }$ on $E _ { O _ { t } }$ , and we follow that by applying FW to obtain matchings $\{ M _ { t , k } \}$ with weights $\{ \beta _ { t , k } \bar  \}$ . We evaluate the Christofides tour length for each (t, k) pair and minimize the mixture objective

$$
\mathcal { L } _ { \mathrm { t o u r } } = \sum _ { t } \alpha _ { t } \sum _ { k } \beta _ { t , k } \left( \mathrm { C H R I S T O F I D E S } ( T _ { t } , M _ { t , k } ) \right) ,
$$

where CHRISTOFIDES $( T _ { t } , M _ { t , k } )$ represents the length of the TSP tour generated by the given parameters. So the learned matchings are directly trained to be usefulfor producing short tours under our pipeline. We also include reconstruction loss in the same way as with other problems, in order to keep X close or inside the polytope. This training also does not require ground-truth matchings; gradients are driven by the downstream TSP objective through the Christofides rounding.

Matching network architecture. The learned matching utilizes a lightweight 2 layer MLP structure to perform fast inferences, it encodes nodes using Fourier positional encoding of the 2D coordinates as input features. The matching FW EXTENSION uses a small iteration budget $T _ { \mathrm { m a t c h } } = 8 .$ The TSP neural network working on the spanning tree polytope follows the same setup as Section D.1.

Empirically, the learned matching variant improves objective values over the greedy-matching version (Table 10) at the cost of higher inference time, and the runtime breakdown (Table 11) shows that matching can dominate the additional overhead on large instances.

## D.6 Ablation Study and Discussion on Projection

Following the experiments for Maximum Coverage, we ablate the effect of projecting the output of the neural network back into the feasible polytope. Table 13 shows that, for TSP, removing projection yields slightly better performance on both TSP-500 and TSP-1000.

We attribute this to a structural property of (2D) Euclidean TSP. Unlike problems with highly heterogeneous instance (e.g., QAP), Euclidean TSP is constrained by a low-dimensional geometric embedding and metric distances, which makes extremely “hard” out-of-distribution instances comparatively rare—both for synthetic generators and for real-world benchmarks that remain within the same problem family. To support this claim empirically, we further test the model trained on TSP-1000 on 2D Euclidean instances from TSPLIB [53] with 51 to 1002 nodes. As shown in Table 14, our method remains robust even on this substantially broader collection.

These results suggest that, for problems like Euclidean TSP, explicit projection is not only unnecessary but can be counterproductive, while the no-projection design is fully viable and can even yield better performance in practice.

Table 13: Ablation results with and without projection
<table><tr><td>Dataset</td><td>With Projection</td><td>No Projection</td></tr><tr><td>TSP-500</td><td>18.06</td><td>17.76</td></tr><tr><td>TSP-1000</td><td>25.47</td><td>25.03</td></tr></table>

Table 14: Performance on TSPLIB of the model trained on TSP-1000.
<table><tr><td>Train Set</td><td>Test Set</td><td>Avg. Pred</td><td>Avg. Opt</td><td>Gap Avg. Time (s)</td></tr><tr><td>TSP-1000</td><td>TSPLIB</td><td>8.848170</td><td>8.061844</td><td>9.75% 1.703</td></tr></table>

## D.7 Ablation Study: LP + Noise

To test whether the gains come from learned structure rather than arbitrary perturbations of the LP relaxation, we construct a randomized baseline by adding i.i.d. Gaussian noise to the subtour-LP solution: $\tilde { x } = x _ { \mathrm { L P } } + \epsilon$ with $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and $\sigma = 0 . 2$ . We then run the same FW decomposition and downstream rounding pipeline using x˜ as the input weights. For each instance we repeat this procedure for 10 independent noise draws and report the best (shortest) tour length, which is a favorable “best-of-10” setting for the noise baseline. As shown in Table 15, this randomized perturbation is unable to match our method, with the gap becoming especially pronounced on larger instances (TSP500–TSP1000). This indicates that our network learns instance-dependent signals that improve the pipeline beyond what can be obtained by injecting random noise into $x _ { \mathrm { L P } }$

<table><tr><td>LP + Gaussian noise+FW (σ=0.2, best of 10)</td><td></td><td>Ours</td></tr><tr><td>TSP50</td><td>5.93</td><td>5.79</td></tr><tr><td>TSP100</td><td>8.86</td><td>8.03</td></tr><tr><td>TSP500</td><td>32.83</td><td>17.76</td></tr><tr><td>TSP1000</td><td>65.51</td><td>25.03</td></tr></table>

Table 15: Ablation results on replacing the learned edge weights with random perturbations of the LP solution. Lower is better.

## E Further Discussion on Inference Settings

In the field of Neural Combinatorial Optimization (NCO), strategies to refine solution quality during inference are widely adopted. While terminology varies across domains—often referred to as Test-Time Optimization (TTO) or Local Search/Sampling—these procedures share a common goal: to improve the solution beyond a single neural forward pass. Broadly, they fall into three categories:

1. Gradient-based Active Search: Fine-tuning model parameters on individual test instances (e.g., active search).

2. Search-based Refinement: Applying discrete search algorithms, such as beam search, MCTS, or heuristic local search (e.g., 2-opt), to the model’s output.

3. Parallel Sampling: Running multiple inference trajectories, such as multiple diffusion noise seeds, batched augmentations, or independent neural forward passes, to enlarge the candidate pool and select the best solution. We distinguish this from native candidate-set decoding, where a single neural forward pass deterministically or procedurally produces several candidates as part of the method’s prescribed decoder.

Decoupling Learned Priors from Search Strategies. While these inference-time techniques significantly enhance performance, they introduce a critical challenge in evaluation: they often obscure the intrinsic quality of the learned neural model.

Recent studies have critically examined this phenomenon. [6] demonstrated that in tree-search-based neural methods, the search algorithm frequently performs the majority of the "heavy lifting," rendering the contribution of the learned neural guidance marginal or, in some cases, redundant. [68] finds that this is also true for tree-search based heatmap recovery. They further point out that this inconsistency is particularly concerning because the search algorithm, which is critical for determining the final solution, is not integrated during neural network training. This confounding effect where the solver relies more on inference time compute than on the learned structural prior has also been noted in recent works such as UCOM2 [7] and GeoNCO [24], which sets a clear difference between TTO and non-TTO methods and make no comparison between them.

Evaluation Protocol: The "One-Shot" Setting. In this work, our primary goal is to evaluate the intrinsic constructive capability of the proposed framework. We aim to assess how well the model aligns with the problem structure in a minimal-latency setting, without relying on extensive post-hoc search. Consequently, we strictly standardize the evaluation to a non-TTO context:

• For Maximum Coverage baselines (e.g., CardNN, UCOM2, GeoNCO), we utilize their "short" inference modes which comes without gradient-based finetune on inference or iterative refinement steps.

• For TSP baselines (e.g., DIMES, DIFUSCO, COExpander), which often rely on active search, parallelized diffusion steps or local search to refine results, we restrict them to their greedy or native candidate-set decoding equivalents. We exclude parallel sampling strategies that enlarge the candidate pool through repeated inference, such as running diffusion-based solvers with multiple noise seeds. However, when a method’s native decoder generates multiple feasible candidates from one neural network pass and selects the best one, we still regard it as one-shot. For example, DIMES (RL+S) uses a number of samples generated from a single neural network pass, so this is treated as native candidate-set decoding rather than parallel sampling. Our FW decomposition follows the same principle: one neural output is decomposed into a sparse set of feasible vertices, and inference reports the best vertex among them.

By isolating the raw performance of the underlying neural architectures, we provide a clearer benchmark of their structural learning capabilities, ensuring that improvements are attributed to better representation learning rather than increased inference computation. This protocol also keeps the comparison consistent with the decoding budgets used by strong baselines, where candidate-set selection is often part of the native decoder, while excluding additional inference-time computation whose main effect is to enlarge the search pool.

## F GPU Memory Usage

We report the GPU memory footprint of FWNCO across all training configurations. Table 16 reports peak CUDA allocated memory during training. These measurements describe allocated GPU memory, rather than total device-memory consumption, which can additionally include reserved but unused memory and CUDA runtime overhead.

Table 16: Peak CUDA allocated memory during FWNCO training.
<table><tr><td>Problem</td><td>Dataset / k</td><td>Peak memory (GiB)</td></tr><tr><td>MC</td><td> $\mathrm { R a n d o m } 5 0 0 , k = 1 0$ </td><td>0.135</td></tr><tr><td>MC</td><td> $\mathrm { R a n d o m } 5 0 0 , k = 5 0$ </td><td>0.135</td></tr><tr><td>MC</td><td> $\mathrm { R a n d o m } 1 0 0 0 , k = 2 0$ </td><td>0.194</td></tr><tr><td>MC</td><td> $\mathrm { R a n d o m } 1 0 0 0 , k = 1 0 0$ </td><td>0.194</td></tr><tr><td>TSP</td><td>TSP50</td><td>0.043</td></tr><tr><td>TSP</td><td>TSP100</td><td>0.106</td></tr><tr><td>TSP</td><td>TSP500</td><td>2.012</td></tr><tr><td>TSP</td><td>TSP1000</td><td>7.708</td></tr><tr><td>QAP</td><td>QAP32</td><td>5.168</td></tr></table>

Overall, the framework has a relatively modest memory footprint of below 8 GiB for every configuration. These results indicate that FWNCO does not require the latest high-end GPUs for the evaluated configurations and suggest potential for scaling to substantially larger instances.

## G Polytopes with Efficient (Approximate) LMO

Matroid polytope Matroid polytopes provide a unified geometric framework for a broad class of combinatorial constraints. Given a matroid $\mathcal { M } = ( V , \mathcal { T } )$ with rank function $r ( \cdot )$ , Edmonds’ classical formulation characterizes the associated matroid base polytope as

$$
\mathcal { P } _ { \mathrm { b a s e } } = \left\{ \mathbf { x } \in \mathbb { R } _ { \geq 0 } ^ { n } \bigg | \sum _ { i \in A } x _ { i } \leq r ( A ) \forall A \subseteq V , \sum _ { i \in V } x _ { i } = r ( V ) \right\} .
$$

A key property of this polytope is that it admits an efficient LMO: optimizing a linear function over $\mathcal { P } _ { \mathrm { b a s e } }$ reduces to finding a maximum-weight base of the matroid, which can be solved in polynomial time via the greedy algorithm. This class of polytopes subsumes several fundamental constraints as special cases:

• Cardinality constraints (uniform matroid). The ground set is $V = \{ 1 , \ldots , n \}$ with rank function $r ( A ) = \operatorname* { m i n } \{ | A | , k \}$ . The corresponding base polytope is the hypersimplex $\begin{array} { r } { \Delta _ { n , k } = \left\{ \mathbf { x } \in \mathbb { R } _ { \geq 0 } ^ { n } \ \middle | \ \sum _ { i = 1 } ^ { n } x _ { i } = k , \ 0 \leq x _ { i } \leq 1 \right\} } \end{array}$

• Partition matroids. Let $\textstyle V = \bigcup _ { j = 1 } ^ { c } V _ { j }$ be a partition of the ground set with associated budgets $k _ { 1 } , \ldots , k _ { c }$ . The rank function is $r ( A ) = \textstyle \sum _ { j = 1 } ^ { c }$ min $\{ | A \cap V _ { j } | , k _ { j } \}$ , and the base polytope is given by $\left\{ \mathbf { x } \in \mathbb { R } _ { \ge 0 } ^ { n } \Big | \sum _ { i \in V _ { j } } x _ { i } = k _ { j } \forall j \right\}$

• Spanning tree constraints (graphic matroid). Given an undirected graph $G = ( V , E )$ the ground set consists of edges and the rank function is $r ( A ) = | V | - \bar { \kappa ( A ) }$ , where $\kappa ( A )$ denotes the number of connected components in $( V , A )$ . The corresponding base polytope is $\begin{array} { r } { \left\{ \mathbf { x } \in \mathbb { R } _ { \geq 0 } ^ { | E | } \ \Big | \ \sum _ { e \in E } x _ { e } = | V | - 1 , \ \sum _ { e \in A } x _ { e } \leq | V | - \kappa ( A ) \forall A \subseteq E \right\} } \end{array}$

Our setup naturally covers all of these cases within a single algorithmic framework. In contrast, Karalias et al. [24] study these settings in a case-specific manner, requiring the design of separate projection operators onto each polytope as well as bespoke decomposition procedures tailored to each constraint. By relying solely on the availability of an efficient LMO, our results apply uniformly across all these matroidal constraints without modifying the algorithm.

Birkhoff polytope $\mathbf { A } \mathbf { n } \ : n \times n$ matrix is called doubly stochastic if all of its entries are nonnegative and the sum of the entries in each row and each column equals one. Permutation matrices form a special subclass of doubly stochastic matrices, characterized by having exactly one entry equal to one in each row and column and zeros elsewhere. The set of all n × n doubly stochastic matrices constitutes a convex polytope known as the $B i r k h o f f p o l y t o p e \ B _ { n } .$ . The Birkhoff polytope lies in an $( n - 1 ) ^ { 2 }$ -dimensional affine subspace of $n ^ { 2 } .$ -dimensional Euclidean space defined by $2 n - 1$ independent linear constraints specifying that the row and column sums all equal 1. The extreme points of Birkhoff polytope $B _ { n }$ are exactly the permutation matrices. Given a residual matrix $\mathbf { \bar { \boldsymbol { R } } } \in \mathbb { R } ^ { n \times n }$ , the LMO over $B _ { n }$ solves

$$
P ^ { \star } \ \in \ \operatorname { a r g m a x } _ { P \in \mathrm { e x t } ( \mathcal { B } _ { n } ) } \langle R , P \rangle = \sum _ { i } \sum _ { j } R ( i , j ) P ( i , j )
$$

This problem is equivalent to finding a maximum-weight perfect matching in a complete bipartite graph $G = ( U , V , \mathbf { \bar { } { E } } )$ , where $U$ and V index the rows and columns of $R ,$ respectively, and each edge $( i , j ) \in E$ has weight $R ( i , j )$ . Hence the linear maximization problem over Birkhoff polytope can be solved using the Hungarian algorithm in time $O ( n ^ { 3 } )$ . Hence, our approach subsumes the setting considered in [50] without requiring any projection onto the Birkhoff polytope. In contrast, the authors of [50] develop a specialized decomposition algorithm and introduce an additional loss term to penalize neural network outputs that lie outside the polytope.

DMO: greedy bipartite matching. Note that a simple greedy algorithm produces a 0.5- approximation to maximum weight matching for any (positively) weighted graph. This still outputs a rather good matching while being significantly faster than running an exact Hungarian algorithm oracle at every iteration. Empirically, we observe that the greedy DMO does not sacrifice quality.

Matching polytope Building on the discussion of the Birkhoff polytope, our approach also applies to matching and perfect matching polytopes.