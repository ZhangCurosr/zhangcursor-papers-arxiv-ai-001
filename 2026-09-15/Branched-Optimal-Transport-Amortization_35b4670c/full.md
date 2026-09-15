# Branched Optimal Transport Amortization

Semyon Semenov<sup>1</sup> Viktor Kovalchuk<sup>1</sup> Meir Roketlishvili<sup>1</sup> Albert Baichorov<sup>1</sup>

Fakhri Karray<sup>1</sup>

Martin Takácˇ<sup>1</sup>

Arip Asadulaev<sup>1</sup>

## Abstract

Methods of Branched Optimal Transport (BOT) mimic the economy and efficiency of natural tree-like structures, such as those found in rivers and biological systems. These methods are widely applicable for designing efficient networks in society, from river basins and blood vessels to mail and gas distribution systems. However, they remain understudied in the context of designing deep generative models, particularly at a large scale. Standard continuous-time generative models, such as the flow matching approach, fail to capture the inherent hierarchical and branching patterns present in real-world data. Current models provide no mechanism for flows to merge or share pathways to minimize total transport cost. Inspired by the "economy of scale" principle in BOT, we introduce a novel, scalable branched flow-matching algorithm designed to solve the branched optimal transport problem in high dimensions. Our method adapts the Benamou-Brenier continuous-time optimal transport formulation to learn branched generative flows. These flows allow probability mass to aggregate along common pathways before branching out to diverse targets. Parametrized by neural networks, our method effectively learns complex branched generative processes. We demonstrate its effectiveness on challenging high-dimensional tasks in biology and image generation.

## 1 Introduction

Deep generative modeling has advanced rapidly, with diffusion models and simulation-free flow–based methods (e.g., flow matching and rectified flows) now supporting state-of-the-art synthesis, fast sampling, and scalable training. These approaches learn vector fields (or stochastic dynamics) that transport a simple base distribution to complex data distributions, and many of them explicitly encourage straight transport trajectories between coupled samples—mirroring the displacement interpolation paradigm in classical optimal transport (OT) [10–12, 18].

However, straight-line, independent motions of mass are often a poor fit for data with hierarchical or multi-modal organization. ImageNet, for

![](images/6d64826004d6fd92faabba626406632d75f43ad3abd244aba367262bb96bbbf2.jpg)  
Figure 1: BOTA Gaussian results for α = 0.5.

example, is built on the WordNet taxonomy: “golden retriever” and “labrador” are siblings under “dog,” which is nested under broader synsets such as “canine” and “mammal.” It is natural to expect that the generative process should share substantial computation along the path common to “dogs”

before specializing to particular breeds. Such structures arise even more naturally and meaningfully in biological processes, which can also be observed as empirical distributions. Modeling such shared structure is difficult.

Branched optimal transport generalizes classical OT by introducing economies of scale: moving two units of mass together along the same route is cheaper than moving them separately. Formally, BOT minimizes energies in which the cost along an edge is subadditive in the mass (often proportional to $m ^ { \alpha }$ with $0 < \alpha < 1 )$ , which causes optimal solutions to coalesce into tree- or network-like structures rather than independent straight paths. This mechanism closely matches how efficient networks form in nature and society, from river basins to mail and gas distribution systems [2, 5, 20].

The BOT perspective is powerful for generative modeling because it encodes shared prefixes of computation: samples that belong to nearby leaves in a taxonomy naturally travel together for most of the journey, branching only when necessary. In contrast with Euclidean quadratic-cost OT—where displacement interpolation moves particles independently along straight segments defined by a Brenier map—BOT explicitly rewards branching, aligning the transport with hierarchical data.

Mathematically, BOT has a mature foundation: it has been studied via discrete “irrigation” models, continuum formulations, and relaxations [3]. Beyond the modeling appeal, BOT solutions satisfy rich structural properties (e.g., optimal junctions have bounded degree), reflecting the network nature of the minimizers [3, 14, 16]. Despite its appeal, BOT is algorithmically challenging. Even in planar settings with finitely many sources/sinks, optimizing BOT networks is NP-hard, and require solving large PDE systems, which becomes prohibitive at modern data scales. This has kept BOT largely confined to small or low-dimensional instances.

At the same time, contemporary generative models operate at unprecedented scale. Flow matching and rectified-flow variants are trained on web-scale datasets and large backbones, and they benefit from transport formulations that are differentiable end-to-end and hardware-accelerated. A scalable BOT solver would let us replace independent straight-path assumptions by economy-of-scale transport that respects data hierarchies practical training pipelines.

Neural parameterizations are a natural fit. In standard OT, neural solvers already amortize transport maps or potentials and integrate cleanly with deep learning. Extending this spirit to BOT promises (i) amortized inference of branched routes that can be reused across batches and tasks,

Contribution: We introduce Branched Optimal Transport Amortization (BOTA), a continuous and scalable solver for branched optimal transport. Our core contribution is a novel optimization problem that we adapt into a generative algorithm, which is efficiently parameterized by neural networks using a flow-matching objective. We demonstrate the effectiveness of our method on challenging biological and image generation tasks, showing that it successfully learns meaningful hierarchical generative processes that reflect the underlying structure of the data.

## 2 Background

Notation. Let $\Omega \subset \mathbb { R } ^ { d }$ be a compact and convex domain. We denote by $\mathscr { P } _ { 2 } ( \Omega )$ the space of probability measures on Ω with a finite second moment. When a measure $\mu$ admits a density relative to the Lebesgue measure, we denote its density by $\rho \left( \mathrm { i . e . , } d \mu ( x ) = d \rho ( x ) d x \right)$ . Vectors are column vectors, $\| \cdot \|$ is the standard Euclidean norm, and $\nabla \cdot$ is the divergence operator and $\| \cdot \| _ { F }$ is the Frobenius norm.

Continuous Normalizing Flow A Continuous Normalizing Flow (CNF) [6] is a generative model that defines a probability density path through a Neural Ordinary Differential Equation (ODE):

$$
\frac { d } { d t } x _ { t } = v _ { \theta } ( x _ { t } , t ) , \qquad x _ { t = 0 } \sim \mu _ { 0 } .\tag{1}
$$

Let $\Phi _ { t }$ be the flow map associated with this ODE, which transports a particle from its initial condition at time 0 to its location at time t. The pushforward density $\rho _ { t } = ( \Phi _ { t } ) _ { \# } \rho _ { 0 }$ evolving under this dynamics necessarily satisfies the continuity equation (2) with the parameterized velocity field $v _ { \theta }$ Where continuity equation encodes mass conservation:

$$
\partial _ { t } \rho _ { t } +  { \nabla } \cdot ( \rho _ { t } v _ { t } ) = 0 \quad \mathrm { o n } \ \Omega \times ( 0 , 1 ) , \quad \rho _ { t = 0 } = \rho _ { 0 } , \quad \rho _ { t = 1 } = \rho _ { 1 } .
$$

A key result is the change of variables formula, which describes how the log density evolves:

$$
\frac { d } { d t } \log \rho _ { t } ( x _ { t } ) = - \nabla \cdot v _ { \theta } ( x _ { t } , t ) .\tag{2}
$$

This allows for a likelihood calculation by integrating this quantity over time. Training can be done by directly maximizing likelihood (integrating (2)).

Flow Matching. The core idea of Flow Matching (FM)[11, 18] is to train a CNF by directly regressing its velocity field $v _ { \theta }$ toward a target vector field $u _ { t }$ that generates a desired probability path. The Flow Matching objective is: ${ \mathcal { L } } _ { \mathrm { F M } } ( \theta ) { \stackrel { - } { = } } \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , \boldsymbol { x } \sim \rho _ { t } } \left[ \Vert \boldsymbol { v } _ { \theta } ( { \boldsymbol { \bar { x } } } , t ) - \boldsymbol { u } _ { t } ( \boldsymbol { x } ) \Vert ^ { 2 } \right]$

A critical challenge is that sampling $x \sim \rho _ { t }$ from the marginal path at arbitrary times is typically intractable. Conditional Flow Matching (CFM) [11] provides a solution by constructing the marginal path as a mixture of simpler and tractable conditional paths. Let z be a conditioning variable with distribution $q ( z )$ . We define the marginal path as follows: $\begin{array} { r } { \rho _ { t } ( x ) = \int \rho _ { t } ( x \mid z ) q ( z ) } \end{array}$ dz, where each conditional path $\rho _ { t } ( x | z )$ is generated by a corresponding conditional vector field $\boldsymbol { u } _ { t } ( \boldsymbol { x } \mid z )$ . The marginal field $u _ { t } ( x )$ that generates $\rho _ { t }$ is then given by:

$$
u _ { t } ( x ) = \mathbb { E } _ { z \sim q } \left[ \frac { \rho _ { t } ( x \mid z ) } { \rho _ { t } ( x ) } u _ { t } ( x \mid z ) \right] ,\tag{3}
$$

where $q ( z \mid x )$ is the posterior. The key theorem [11] is to minimize the Conditional Flow Matching:

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , z \sim q , x \sim \rho _ { t } ( \cdot \vert z ) } \left[ \Vert v _ { \theta } ( x , t ) - u _ { t } ( x \mid z ) \Vert ^ { 2 } \right]\tag{4}
$$

with $t \sim \mathcal { U } [ 0 , 1 ]$ yields the same gradient for θ as minimizing the intractable ${ \mathcal { L } } _ { \mathrm { F M } } ( \theta )$ in (4). This makes CFM a practical objective, as it only requires sampling from conditional paths $\rho _ { t } ( x | z )$ and knowing their closed-form drifts $u _ { t } ( x | z )$ ).

The flexibility of CFM lies in the choice of conditional paths. The coupling $q ( z )$ is the independent joint distribution $q ( x _ { 0 } ) q ( x _ { 1 } ) , \mathrm { s o } z = ( x _ { 0 } , x _ { 1 } )$ . A common conditional path is a Gaussian bridge: $\dot { \rho } _ { t } ( x \mid z ) = \mathcal { N } ( x \mid \mu _ { t } , \dot { \sigma } ^ { 2 } I )$ , where $\mu _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ is a linear interpolation. The simple constant drift that generates this path is $u _ { t } ( x \mid z ) = x _ { 1 } - x _ { 0 }$

Monge–Kantorovich Optimal Transport (OT) seeks a way to morph one probability distribution with minimal effort, as quantified by a cost function $c ( x , y )$ The Monge formulation seeks a deterministic map $T : \Omega \to \Omega$ that pushes $\mu _ { 0 } ~ \mathrm { t o } ~ \mu _ { 1 }$ and minimizes the total cost $\textstyle \int c ( x , T ( x ) ) d \mu _ { 0 } ( x )$ [19]. This problem can be ill-posed, its relaxation, the Kantorovich problem, searches over couplings (joint distributions) $\gamma \in \Gamma ( \mu _ { 0 } , \mu _ { 1 } )$ with marginals $\mu _ { 0 }$ and $\mu _ { 1 } { : }$

$$
\operatorname* { i n f } _ { \gamma \in \Gamma ( \mu _ { 0 } , \mu _ { 1 } ) } \int _ { \Omega \times \Omega } c ( x , y ) d \gamma ( x , y ) .\tag{5}
$$

For cost $c ( x , y ) = \| x - y \| ^ { 2 }$ , the square root of the solution is the Wasserstein-2 distance. The flow matching approach discussed in the section above can be connected with the flow matching to create the optimal transport CFM (OT-CFM) [18] method uses an optimal coupling to define conditionals. Here, ${ z = \left( x _ { 0 } , x _ { 1 } \right) }$ is sampled from an OT plan γ between $\mu _ { 0 }$ and $\mu _ { 1 } 5 .$ . In practice, the OT plan $\gamma$ is efficiently approximated using mini-batch OT by Sinkhorn algorithm [7].

Benamou–Brenier OT frames transportation as a continuous-time problem. For flow $\Phi _ { t }$ , we have a pushforward density $\rho _ { t } = ( \Phi _ { t } ) _ { \# } \rho _ { 0 }$ evolving under dynamics that satisfies continuity equation 2 with the velocity field v. The Benamou–Brenier theorem states that the squared Wasserstein distance equals the minimal kinetic energy of such a flow:

$$
\operatorname* { i n f } _ { \rho _ { t } , v _ { t } } \left\{ \int _ { 0 } ^ { 1 } \int _ { \Omega } \| v _ { t } ( x ) \| ^ { 2 } \rho _ { t } ( x ) d x d t \quad { \mathrm { s u b j e c t ~ t o ~ } } ( 2 ) \right\} .\tag{6}
$$

The minimizers of this problem are constant-speed geodesics in the Wasserstein space. When an optimal transport map $\dot { T }$ exists for the static problem $( \mathrm { e . g . }$ , when $\mu _ { 0 }$ is absolutely continuous), the geodesic is given by interpolation: $\rho _ { t } = ( ( 1 - t ) \mathrm { I d } + t T ) _ { \# } \rho _ { 0 }$ . The corresponding velocity field is constant along the trajectories, $v _ { t } ( x _ { t } ) = T ( x _ { 0 } ) - x _ { 0 }$ , and the particle dynamics is linear: $x _ { t } = ( 1 - t ) x _ { 0 } + t T \bar { ( x _ { 0 } ) }$

## 3 Branched Benamou–Brenier OT

Branched Benamou–Brenier formulation studied by [4], which is also equivalent to the formulation of [20]. The idea is to consider the classical Benamou–Brenier optimal transport problem 6, but with a concave cost function. The boundary conditions remain the same as in standard optimal transport: $\rho ( 0 , \cdot ) = \mu _ { 0 }$ and $\rho ( 1 , \cdot ) = \mu _ { 1 }$ . With $\alpha \in ( 0 , 1 )$ as the branching parameter, we define the cost that measures the work of moving a mass m over a distance $\ell ,$ simply written as $m ^ { \alpha } \ell . \mathrm { H } \alpha = 1$ , the cost becomes linear in the mass $( m ^ { 1 } \ell )$ , recovering the classical OT setting 2. By identifying the length ℓ with the velocity v and the mass m with the time-dependent probability density $\rho _ { t }$ , the dynamic branched optimal transport can be formulated as:

$$
B _ { \alpha } ( \mu _ { 0 } , \mu _ { 1 } ) \ = \ \operatorname* { i n f } _ { \rho ( t , \cdot ) , \upsilon ( t , \cdot ) } \int _ { 0 } ^ { 1 } \sum _ { i \in I _ { t } } \lvert v _ { t , i } \rvert ( \rho _ { t , i } ) ^ { \alpha } d t .\tag{7}
$$

What the reader can notice is that for the cost (7) need to be finite, the measure $\rho ( t , \cdot )$ must be purely atomic for (almost every) time t! This means that the mass is concentrated at a points $\{ x _ { t , i } \} _ { i \in I _ { t } } \colon$

$$
\rho ( t , \cdot ) = \sum _ { i \in I _ { t } } \rho _ { t , i } \delta _ { x _ { t , i } } , \quad \mathrm { w h e r e } \ \rho _ { t , i } = \rho ( t , \{ x _ { t , i } \} ) .
$$

The associated momentum is then $\begin{array} { r } { q \ = \ \rho v = \sum _ { i \in I _ { t } } v _ { t , i } \rho _ { t , i } \delta _ { x _ { t , i } : } } \end{array}$ , where $v _ { t , i } = v ( t , x _ { t , i } )$ is the velocity of the atom $x _ { t , i } .$ This formulation is computationally intensive, which motivates our development of a more efficient relaxation.

## 4 Branched Benamou Brenier Neural OT

To build a Benamou Brenier style Branced OT flows we need to develop a flow algorithms that meet a few constraints:

• The boundary constraint: we need a flow whose starting and end points are equal to the given distributions: $\rho ( 0 , \cdot ) = \mu _ { 0 } \rho ( 1 , \cdot ) = \mu _ { 1 }$

• Mass-dependent cost: We need to have a time-dependent density function that we are using as part of the cost function.

• Atomic mass constraint: Our density function must be atomic almost every time $t ,$ so the mass needs to be concentrated in a countable set of points.

To meet these constraints, it is convenient to draw an analogy between branched optimal transport and continuous normalizing flows. Having a continued vector field, we can derive a time continuous probability density function that plays a central role in the formulation of branched optimal transport. Also, CNFs by design meet the the boundary constraint 2. So, first, let us parameterize the velocity field by a continuous normalizing flow.

$$
{ \dot { x } } _ { t } = v _ { \theta } ( x _ { t } , t ) , \qquad t \in [ t _ { 0 } , t _ { 1 } ] ,
$$

whose density $p _ { \theta } ( x , t )$ evolves under the instantaneous change–of–variables, CNF formula:

$$
\log p _ { \theta } ( x _ { t _ { 1 } } , t _ { 1 } ) = \log p _ { \theta } ( x _ { t _ { 0 } } , t _ { 0 } ) - \int _ { t _ { 0 } } ^ { t _ { 1 } } \mathrm { t r } \big ( \partial _ { x } v _ { \theta } ( x _ { t } , t ) \big ) d t .\tag{8}
$$

This enforces the continuity equation by construction, making CNFs natural for dynamic OT objectives. To capture branched transport, we concavify the kinetic action by the density, mirroring the BOT cost $m ^ { \alpha } \ell$ with $\alpha \in ( 0 , 1 )$ . Our learning objective over terminal samples is

$$
\mathcal { L } ^ { \alpha } ( \theta ) = \int _ { t _ { 0 } } ^ { t _ { 1 } } \int _ { \Omega } \underbrace { \left\| v _ { \theta } ( x , t ) \right\| p _ { \theta } ( x , t ) ^ { \alpha } d x d t } _ { \mathrm { b r a n c h e d c o s t } } - \lambda \underbrace { \mathbb { E } _ { x _ { t _ { 1 } } \sim \mu _ { 1 } } \bigl [ \log p _ { \theta } ( x _ { t _ { 1 } } , t _ { 1 } ) \bigr ] } _ { \mathrm { m a t c h d a t a t } t _ { 1 } } .\tag{9}
$$

The second term is the CNF maximum-likelihood fit to $\mu _ { 1 }$ at $t _ { 1 }$ . The first term is a BOT-style dynamic action: it penalizes kinetic effort $\| v _ { \theta } \|$ but discounts it by $p _ { \theta } ^ { \alpha }$ with $\alpha < 1$ . Our method is a concave reweighting that implements the BOT principle move together, then split, while keeping the CNF machinery and training recipe.

A challenge in implementing the BOT objective (9) within a CNF framework is the atomic mass constraint. The theoretical BOT formulation requires the evolving measure $\rho ( t , \cdot )$ to be a sum of discrete point masses for the cost to be finite. However, CNFs naturally define a continuous density field $p _ { \theta } ( x , t )$ , making these two concepts fundamentally incompatible for optimization.

To bridge this gap, we introduce a differentiable relaxation of the atomic constraint, which we term the soft-atomic cost. In a computational setting, we cannot handle continuous measures or an infinite number of atoms. We make two approximations. Instead of forcing the continuous density to collapse into discrete points, this objective encourages the density to cluster around a predefined set of $K$ proxy atoms. This is achieved by softly assigning the mass of particles from the flow to these centers based on proximity.

At any given time t, for a batch of particles $x _ { t }$ sampled from the flow density $p _ { \theta } ( \cdot , t )$ , we define the soft mass $m _ { k } ( t )$ and the average velocity $v _ { k } ( t )$ associated with center $c _ { k }$ and where $w _ { k } ( x )$ is a soft-assignment weight, typically a Gaussian kernel that measures the influence of center $c _ { k }$ on a particle at position x:

$$
w _ { k } ( \boldsymbol { x } ) = \frac { \exp ( - \frac { \| \boldsymbol { x } - \boldsymbol { c } _ { k } \| ^ { 2 } } { 2 \sigma ^ { 2 } } ) } { \sum _ { j = 1 } ^ { K } \exp ( - \frac { \| \boldsymbol { x } - \boldsymbol { c } _ { j } \| ^ { 2 } } { 2 \sigma ^ { 2 } } ) } , \quad m _ { k } ( t ) = \mathbb { E } _ { \boldsymbol { x } _ { t } \sim p _ { \theta } ( \cdot , t ) } \left[ w _ { k } ( \boldsymbol { x } _ { t } ) \right] ,\tag{10}
$$

$$
v _ { k } ( t ) = \frac { \mathbb { E } _ { x _ { t } \sim p _ { \theta } ( \cdot , t ) } \left[ v _ { \theta } ( x _ { t } , t ) w _ { k } ( x _ { t } ) \right] } { m _ { k } ( t ) } .\tag{11}
$$

The instantaneous soft-atomic cost is then formulated as a differentiable proxy to the theoretical BOT cost. By integrating soft cost over time, we reformulate the learning objective as:

$$
\mathcal { L } ^ { \alpha } ( \theta ) = \int _ { t _ { 0 } } ^ { t _ { 1 } } \sum _ { k = 1 } ^ { K } \underbrace { v _ { k } ( t ) m _ { k } ( t ) ^ { \alpha } d t } _ { \mathrm { s o f t b r a n c h e d ~ c o s t } } - \lambda \underbrace { \mathbb { E } _ { x _ { t _ { 1 } } \sim \mu _ { 1 } } \bigl [ \log p _ { \theta } ( x _ { t _ { 1 } } , t _ { 1 } ) \bigr ] } _ { \mathrm { m a t c h ~ d a t } _ { 1 } } .\tag{12}
$$

Lemma 1 (Differentiable Approximation of the Instantaneous Branched Transport Cost) Let $\hat { \rho } _ { t } =$ $\begin{array} { r } { \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \delta _ { x _ { j } ( t ) } } \end{array}$ be an empirical probability measure representing a distribution at time t with particles at positions $\{ x _ { j } \} _ { j = 1 } ^ { B }$ moving with velocities $\{ v _ { j } \} _ { j = 1 } ^ { B }$ . Let $\{ c _ { k } \} _ { k = 1 } ^ { K }$ be a set offixed centers in $\mathbb { R } ^ { d }$ The function soft atomic cost, defined as:

$$
C _ { s o f t } ( \{ x _ { j } \} , \{ v _ { j } \} ) = \sum _ { k = 1 } ^ { K } | v _ { k } | m _ { k } ^ { \alpha } ,
$$

wherefor each center $c _ { k } .$

$$
m _ { k } = { \frac { 1 } { B } } \sum _ { j = 1 } ^ { B } w _ { j k } \quad ( S o f t M a s s ) \qquad | v _ { k } | = { \frac { \sum _ { j = 1 } ^ { B } w _ { j k } | v _ { j } | } { \sum _ { j = 1 } ^ { B } w _ { j k } } } \quad ( S o f t A \nu e r a g e S p e e d )
$$

with weights $w _ { j k } = s o f t m a x ( - \lvert \lvert x _ { j } - c _ { k } \rvert \rvert ^ { 2 } / 2 \sigma ^ { 2 } ) _ { k } ,$ , is a differentiable approximation of the true instantaneous branched transport cost functional for the discretized measure. In the limit as the kernel width $\sigma  0 ,$ , this cost converges to the cost ofa hard assignment ofparticles to their nearest centers. Please see proofs in Appendix.

## 5 Branched Flow Matching via BOT Amortization

The primary challenge of branched optimal transport, as formulated in Sec.3, is the difficulty of directly optimizing the branched Benamou-Brenier objective (9) over the space of continuous-time neural velocity fields. The objective is non-trivial, and the "atomic" constraint is inherently discrete.

To bridge this, we introduce a practical two-stage method, Branched Optimal Transport Amortization (BOTA) also we call it Branched Flow Matching, which first solves a discrete-time version of the our proposed BOT soft-atomic relaxation for a fixed batch and then amortizes this solution into a continuous-time neural network using a flow matching objective. This approach is analogous to standard OT-CFM (9), but we replace the simple, straight-line (Euclidean) optimal transport plan Sec. 2 with a more complex, branched transport plan.

## 5.1 Discrete Branched Transport Solver

First, we compute an exemplar branched flow for a single, fixed batch of $N$ source samples $\{ x _ { 0 } ^ { i } \} _ { i = 1 } ^ { N } \sim$ $\mu _ { 0 }$ and $N$ target samples $\mathsf { \bar { \{ } }  x _ { 1 } ^ { i } \} _ { i = 1 } ^ { N } \sim \mu _ { 1 }$ . We discretize the time interval $[ 0 , 1 ]$ into $T$ steps of size $\Delta t = 1 / T$ . Our goal is to find an optimal discrete velocity field $V \in \mathbb { R } ^ { T \times N \times \bar { D } }$ that transports the batch $\{ \stackrel { ' } { x _ { 0 } ^ { i } } \} \{ 0 \{ \stackrel { \smile } { x _ { 1 } ^ { i } } \}$ while minimizing a discrete analogue of the branched transport cost. The particle trajectories $\{ X _ { k } \} _ { k = 0 } ^ { T } .$ , where $X _ { k } = \breve { \{ } x _ { k } ^ { i } \} _ { i = 1 } ^ { N }$ , are computed via standard Euler integration:

$$
\begin{array} { r } { x _ { k + 1 } ^ { i } = x _ { k } ^ { i } + V _ { k , i } \cdot \Delta t , \quad \mathrm { w i t h } \ x _ { 0 } ^ { i } \mathrm { g i v e n } . } \end{array}
$$

Here, $V _ { k , i } \in \mathbb { R } ^ { D }$ is the velocity of particle i at discrete time step $k .$

We optimize V by minimizing a loss function that combines a terminal matching cost with the time-integrated soft-atomic energy:

$$
\mathcal { L } _ { \mathrm { d i s c r e t e } } ( V ) = \sum _ { k = 0 } ^ { T - 1 } C _ { \mathrm { s o f t } } ( X _ { k } , V _ { k } ) \Delta t + \lambda \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| x _ { T } ^ { i } - x _ { 1 } ^ { i } \| ^ { 2 } .
$$

The first term is a standard mean-squared error that forces the final positions $x _ { T } ^ { i }$ to match the target positions $x _ { 1 } ^ { i }$ . The second term is the discrete integral of the instantaneous branched transport cost, $C _ { \mathrm { s o f t } }$ , which is defined precisely in Lemma 1 (Eq. 14) as:

$$
C _ { \mathrm { s o f t } } ( X _ { k } , V _ { k } ) = \sum _ { j = 1 } ^ { K } | v _ { j } | _ { k } ( m _ { j } ) _ { k } ^ { \alpha } ,
$$

where $( m _ { j } ) _ { k }$ and $( | v _ { j } | ) _ { k }$ are the soft mass and soft average speed, respectively, associated with the j-th proxy atom $c _ { j }$ at time step k.

It is important to note that our terminal matching term, $\begin{array} { r } { \lambda \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| x _ { T } ^ { i } - x _ { 1 } ^ { i } \| ^ { 2 } } \end{array}$ , is a simple Mean Squared Error (MSE). This is a deliberate choice for computational efficiency. Since this discrete solver operates on a fixed, finite batch, we are computing a deterministic, point-to-point map for these specific $N$ samples. We are not training a generative model to match the full distribution $\mu _ { 1 }$ at this stage. Therefore, a direct MSE is the most appropriate and efficient loss to enforce the terminal constraint, in contrast to a heavier probabilistic loss (like a CNF log-likelihood) which would be unnecessary for this exemplar-finding step. By solving this optimization problem $( \mathrm { e . g . }$ , via gradient descent on $V )$ , we obtain an optimal discrete velocity field $\mathbf { \Psi } _ { V ^ { * } } ^ { \mathbf { 1 } } \in \mathbb { R } ^ { T \times N \times D }$ and its corresponding optimal trajectory $X ^ { * } \in \mathbb { R } ^ { ( T + 1 ) \times N \times D }$ . This pair $( X ^ { * } , V ^ { * } )$ represents a single, discrete branched transport path for the given batch.

## 5.2 Amortization via Flow Matching

The discrete solution $( X ^ { * } , V ^ { * } )$ is only valid for the specific batch it was trained on. To create a generative model, we must learn a continuous-time neural velocity field $v _ { \theta } ( x , t )$ that can generalize this branched behavior. We achieve this by "distillin $1 \mathrm { g } "$ the discrete solution into $v _ { \theta }$ using a flow matching objective. This step reframes the problem as a standard supervised regression task. We treat the discrete solution $( X ^ { * } , V ^ { * } )$ as the ground truth. We construct a continuous-time target vector field, $u _ { t } ( x )$ , by linearly interpolating the discrete solution. For continuous time $t \in [ 0 , 1 ]$ , we find its corresponding discrete time index $\check { k } = \lfloor t / \Delta t \rfloor$ (clamped to $[ 0 , T - 1 ] )$ and the interpolation weight $\beta = ( \bar { t } - k \Delta \bar { t } ) / \Delta t \in [ 0 , 1 ]$ . For each particle i, the target position $ { \boldsymbol { { x } } } _ { t } ^ { \imath }$ and target velocity $u _ { t } ( x _ { t } ^ { i } )$ at time t are defined by linear interpolation:

$$
x _ { t } ^ { i } = ( 1 - \beta ) x _ { k } ^ { * , i } + \beta x _ { k + 1 } ^ { * , i } , \qquad u _ { t } ( x _ { t } ^ { i } ) = ( 1 - \beta ) V _ { k , i } ^ { * } + \beta V _ { k ^ { \prime } , i } ^ { * } ,
$$

where $k ^ { \prime } = \operatorname* { m i n } ( k + 1 , T - 1 )$ is the next velocity index.

The Branched Flow Matching (BFM) objective is then to regress the neural network $v _ { \theta }$ onto this continuous, interpolated target field $u _ { t } \colon$

$$
\mathcal { L } _ { \mathrm { B F M } } ( \theta ) = \mathbb { E } _ { i \sim \mathcal { U } [ 1 , N ] , t \sim \mathcal { U } [ 0 , 1 ] } \left[ \left. v _ { \theta } ( x _ { t } ^ { i } , t ) - u _ { t } ( x _ { t } ^ { i } ) \right. ^ { 2 } \right] .
$$

By minimizing this objective, the network $v _ { \theta }$ learns to approximate the complex, branched velocity field derived from the discrete BOT solver. Once trained, $v _ { \theta }$ can be used as a generative model to transport new samples from $\mu _ { 0 }$ to $\mu _ { 1 }$ by solving the continuous-time ODE $\dot { x } = v _ { \theta } ( x , t )$ , effectively amortizing the expensive discrete optimization.

![](images/150808461d3b72095e7826ebb8d018f9834a8ee47c15b5bf039e2cfda8ed01e9.jpg)  
(a) 3 (Flow Matching)

![](images/7b27663bb3d5e0ffa797cff8c89b9b2faceedeb9bc72f7350d4e19a9aa6d6ca5.jpg)  
(b) 6 (Flow Matching)

![](images/df7cc48efe5be31fd6d5cdd23d80e4efbe8119bb74d4e03fa99a4837965856d4.jpg)  
(c) 18 (Flow Matching)

![](images/10228edb4e29b399f057a346757e203b3b5d6ec499dab399f0e98ccbbc1d30d8.jpg)  
(d) 3 (BOTA)

![](images/20ec2830a8f45e6aeb2e076bfd2a538e381267682bd945e6063fc296a5831f61.jpg)  
(e) 6 (BOTA)

![](images/737981d7b784cc20a865deef7db9d6e232de9936d050276126179143619e89ad.jpg)  
(f) 18 (BOTA)  
Figure 2: BOTA Gaussian mixture results for increasing number of Gaussians. Top row: Flow Matching. Bottom row: BOTA (α = 0.5).

## 6 Experiments

## 6.1 Gaussian Mixtures

We compare how two flow-based methods transport mass from a single source distribution to a highly multi-modal target. The target is a mixture of $\dot { K }$ clusters (“branches”) arranged at the top; the source is a cluster (or mix of two clusters) at the bottom. We train the same-size models with (i) standard FM and (ii) BOTA. Each curve shows a sample’s trajectory from source (blue) to its destination cluster (purple). As K grows $( 3  6  1 8 )$ , FM learns many independent, fan-out paths that crowd and cross, offering little shared routing.

BOTA instead discovers a shared “trunk” that later splits into branches, yielding short, structured, tree-like transport. The qualitative gap widens with more branches. Both use models the same timeconditioned MLP $v _ { \theta } ( x , t )$ : 64-d time embedding → 3×256 SiLU layers → 2-D output; integrated with 10 fixed Euler steps over $t \in [ 0 , 1 ]$ . Training is identical: Adam $( l r = 1 0 ^ { - 3 }$ , batch 256, 10k iterations, seed 42. FM uses the standard flow-matching objective along linear source–target couplings. For the results, please see Figure 2. As you can notice, FM produces independent straight-line arms; BOTA discover shared trunks before splitting.

Table 1: Comparison of methods on the Tedsim dataset (50D PCA); mean ± std over 5 seeds.
<table><tr><td>Metric</td><td>BSBM</td><td>CNF</td><td>CFM</td><td>FM</td><td>BOTA</td></tr><tr><td> $W _ { 1 }$ </td><td> $1 7 . 7 2 \pm 0 . 1 2$ </td><td> $1 3 . 7 6 \pm 0 . 1 0$ </td><td> $1 2 . 0 1 \pm 0 . 0 5$ </td><td> $1 2 . 0 3 \pm 0 . 1 3$ </td><td> ${ \bf 1 1 . 8 6 \pm 0 . 0 7 }$ </td></tr><tr><td> $W _ { 2 }$ </td><td> $1 7 . 9 6 \pm 0 . 1 5$ </td><td> $1 3 . 8 1 \pm 0 . 1 1$ </td><td> $1 2 . 1 2 \pm 0 . 0 4$ </td><td> $1 2 . 1 5 \pm 0 . 1 6$ </td><td> ${ \bf 1 2 . 0 2 \pm 0 . 0 9 }$ </td></tr><tr><td>RBF-MMD</td><td> $0 . 6 3 \pm 0 . 0 0 7$ </td><td> $0 . 5 1 \pm 0 . 0 0 5$ </td><td> $0 . 1 1 \pm 0 . 0 0 2$ </td><td> $0 . 1 5 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 1 1 \pm 0 . 0 0 3 }$ </td></tr></table>

## 6.2 Biological data

To further validate our method, we use the Tedsim dataset ([15]) as a controlled reference point with known ground truth dynamics. Tedsim simulates a cellular differentiation process by modeling cell division from a root cell, generating both gene expression profiles and heritable lineage barcodes. This provides a complete record of a simple, branching differentiation process.

We apply our method to a specific Tedsim scenario, modeling the transport from a single progenitor cell to two distinct terminal states. This controlled experiment allows us to quantitatively assess our method’s ability to accurately reconstruct branching trajectories where the true pathways are known beforehand. Please see Figure 3 and Table 1. We use the same architecture for fair comparison for all types of models except for Mean-Flows which requires additional input r to encoding the start of sampling trajectory.

Comparison baselines: For image data Flow Matching (FM)[11] and Mean Flows (MFs)[9] Branched Schrödinger Bridge Matching (BSBM)[17] and CNF[6] are used as baselines in our work. All methods are trained with exactly the same amount of iteration 100k, learning rate 1e-4 and optimizer Adam. The experiments highlight our model’s effectiveness in reconstructing branching trajectories. Our method produces the best results among all the models compared. By leveraging the natural branching structure inherent in the data, our approach achieves the highest performance score while maintaining computational efficiency. The parameter α ablation please see in Appendix.

![](images/879bd06bf283a9f16e784f169a92aad448d98c9a7bc437ab34ee635cb1e5f8eb.jpg)  
Figure 3: BOTA results on complicated biology Tedsim dataset with $\alpha = 0 . 5$

## 6.3 Image data

![](images/add0e78d69b561b6c96642392e774b2124bec2d6815f361321d65a3638e1362f.jpg)  
Figure 4: PCA of the learned flow for generating classes 0 and 1.

MNIST. For our proof of concept with image data, we conducted experiments using the classic MNIST dataset [8]. We designed two experiments: first, mapping digits into the two classes 0 and 1 to examine branching structures in the data, and second, mapping the entire dataset from the Gaussian distribution. As shown in Figure 4, our method successfully reveals branching structures in this image dataset as well. The hyperparameters setting was set equal to the one chosen for the biology experiment.

FFHQ. All experiments are conducted in the latent space of a pretrained ALAE model on FFHQ 1024x1024 dataset. The ALAE is used only for decoding latents to images during evaluation and remains frozen at all times. We load latent vectors paired with gender labels and split them into 60,000 training and 10,000 test samples. Let D denote the latent dimensionality inferred from the data. The goal of this experiment is to show that our method can produce a smooth and hierarchical interpolation between two classes. We learn a BOTA from source-class latents to target-class latents. (etc. female → male); mini-batches for $x _ { 0 }$ and $x _ { 1 }$ are sampled from the respective class-specific training subsets. We amortize model a time-dependent vector field $v _ { \theta } ( t , x )$ that defines the ODE $\begin{array} { r } { \frac { d } { d t } x ( t ) = v _ { \theta } \big ( t , x ( t ) \big ) } \end{array}$ integrated from $t = 0 \mathrm { t o } t = 1$

![](images/c29aeeec7fe2dbcac42af174380f0076362ae14f0b629a71a0b02c046514cf79.jpg)  
Figure 5: Branched flow interpolation on FFHQ dataset From left to right over the timestamps. Female → Male task.

Architecture: v<sub>θ</sub> is an MLP with Tanh activations and hidden width 1024. A scalar time embedding (linear layer) is concatenated to x before the MLP; the output is D-dimensional. For the results please see Figure 5.

## 7 Related Work

Branched optimal transport can be formalized in various ways, such as for traffic plans, irrigation flows, and branched networks. Some details on rectified flows, action matching, and generative flow networks are also provided. For example in [13], with a single source supply $\mu .$ The authors modeled the transportation network as a set of “fibers” $\chi ( p , \cdot )$ , where $\chi ( p , t )$ represents the location of a particle $p \in \Omega$ in time t, and (Ω is an abstract probability space) [2]. Such approaches provide insight into how continuous mass might cluster around “branch nodes,” but they remain computationally intractable for high-dimensional data.

In parallel, the deep generative modeling community [1, 17] proposed Y-shaped and branched Generative Flows, a continuous-time flow framework that encourages branching of paths. However, these models do not explicitly solve the BOT optimization problem; they embed a BOT-inspired inductive bias in the model but do not guaranty a globally optimal branched transport plan for the given data. In other words, Y-shaped flows learn a hierarchy of transport heuristically within a single-stage network training, rather than computing or supervising with the ground-truth BOT.

## 8 Limitations and Broader Impact

Our method relies on a differentiable soft-atomic relaxation whose geometry depends on choices such as proxy atoms and kernel bandwidth may require task-specific tuning. The two-stage discrete optimization and neural amortization procedure also adds design complexity. This work may support applications involving structured branching processes, such as cellular differentiation and other hierarchical systems, and improves the expressivity of generative models. However, it inherits standard generative-modeling risks, including misuse for synthetic data generation and amplification of dataset biases, especially in sensitive domains.

## 9 Conclusion

We introduce Branched Optimal Transport Amortization (BOTA), a scalable generative algorithm designed to overcome the inability of standard continuous-time models to capture hierarchical, treelike data structures. By adapting the Benamou-Brenier formulation with a concave "economy of scale" cost function, BOTA encourages probability mass to aggregate along shared "trunks" before branching out to specific targets. To make this computationally feasible, the method utilizes a novel differentiable "soft-atomic" relaxation and amortizes the solution into a continuous neural vector field via flow matching. Experiments demonstrate that BOTA effectively recovers underlying branching geometries and outperforms baseline methods in reconstructing complex, multi-modal distributions.

## References

[1] Arip Asadulaev, Semyon Semenov, Abduragim Shtanchaev, Eric Moulines, Fakhri Karray, and Martin Takac. Y-shaped generative flows. arXiv preprint arXiv:2510.11955, 2025.

[2] Marc Bernot, Vicent Caselles, and Jean-Michel Morel. Traffic plans. Publicacions Matemà- tiques, pages 417–451, 2005.

[3] Marc Bernot, Vicent Caselles, and Jean-Michel Morel. Optimal transportation networks: models and theory. Springer, 2009.

[4] Lorenzo Brasco, Giuseppe Buttazzo, and Filippo Santambrogio. A benamou–brenier approach to branched transport. SIAMjournal on mathematical analysis, 43(2):1023–1040, 2011.

[5] Giuseppe Buttazzo and Eugene Stepanov. Optimal transportation networks as free dirichlet regions for the monge-kantorovich problem. Annali della Scuola Normale Superiore di Pisa-Classe di Scienze, 2(4):631–678, 2003.

[6] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.

[7] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in neural information processing systems, pages 2292–2300, 2013.

[8] Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE signal processing magazine, 29(6):141–142, 2012.

[9] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[11] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[12] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[13] Francesco Maddalena, Giovanni Taglialatela, and Jean-Michel Morel. A variational model of irrigation patterns. Interfaces and Free Boundaries, 5(4):391–416, 2003.

[14] Edouard Oudet and Filippo Santambrogio. A modica-mortola approximation for branched transport and applications. Archivefor rational mechanics and analysis, 201(1):115–142, 2011.

[15] Xiaoyu Pan, Hongyu Fan, Wei Li, Haochen Shen, Rui Wang, and Xiaobo Zhou. Tedsim: A simulation framework for single-cell rna sequencing data. Bioinformatics, 38(8):2208–2214, 2022.

[16] Filippo Santambrogio. Optimal transport for applied mathematicians. Birkäuser, NY, 55(58-63): 94, 2015.

[17] Sophia Tang, Yinuo Zhang, Alexander Tong, and Pranam Chatterjee. Branched schr\" odinger bridge matching. arXiv preprint arXiv:2506.09007, 2025.

[18] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482, 2023.

[19] Cédric Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media, 2008.

[20] Qinglan Xia. Optimal paths related to transport problems. Communications in Contemporary Mathematics, 5(02):251–279, 2003.

## Extended Analysis

$$
\mathcal { L } ^ { \alpha } ( \theta ) = \int _ { t _ { 0 } } ^ { t _ { 1 } } \sum _ { k = 1 } ^ { K } \underbrace { \Vert \bar { v } _ { k } ( t ) \Vert m _ { k } ( t ) ^ { \alpha } } _ { \mathrm { s o f t b r a n c h e d c o s t } } d t - \lambda \underbrace { \mathbb { E } _ { x _ { t _ { 1 } } \sim \mu _ { 1 } } \big [ \log p _ { \theta } ( x _ { t _ { 1 } } , t _ { 1 } ) \big ] } _ { \mathrm { m a t c h d a t a t } t _ { 1 } } .\tag{13}
$$

Lemma 1 (Differentiable Approximation of the Instantaneous Branched Transport Cost). Let $\begin{array} { r } { \hat { \rho } _ { t } = \sum _ { j = 1 } ^ { B } \rho _ { j } \delta _ { x _ { j } ( t ) } } \end{array}$ be an empirical measure representing a particle approximation ofa distribution at time t, with particle positions $\{ x _ { j } ( t ) \} _ { j = 1 } ^ { B } \subset \mathbb { R } ^ { d }$ and velocities $\{ v _ { j } ( t ) \} _ { j = 1 } ^ { B } \subset \mathbb { R } ^ { d } .$ . Let $\{ c _ { k } \} _ { k = 1 } ^ { K }$ be a set of fixed centers. Define the soft\_atomic\_cost as

$$
C _ { s o f t } ( \{ x _ { j } \} , \{ v _ { j } \} ) = \sum _ { k = 1 } ^ { K } \| { \bar { v } } _ { k } \| m _ { k } ^ { \alpha } ,
$$

wherefor each center $c _ { k } .$

$$
\begin{array} { l l } { { m _ { k } = \displaystyle \sum _ { j = 1 } ^ { B } w _ { j k } \rho _ { j } } } & { { \qquad ( S o f t M a s s ) , } } \\ { { { } } } & { { { } } } \\ { { \bar { v } _ { k } = \displaystyle \frac { \sum _ { j = 1 } ^ { B } w _ { j k } v _ { j } } { \sum _ { j = 1 } ^ { B } w _ { j k } + \varepsilon } } } & { { \qquad ( S o f t A v e r a g e d V e l o c i t y ) , } } \end{array}
$$

with smooth weights

$$
w _ { j k } = \frac { \exp \left( - \| x _ { j } - c _ { k } \| ^ { 2 } / 2 \sigma ^ { 2 } \right) } { \sum _ { \ell = 1 } ^ { K } \exp \left( - \| x _ { j } - c _ { \ell } \| ^ { 2 } / 2 \sigma ^ { 2 } \right) } .
$$

Then $C _ { s o f t }$ is differentiable in $\{ x _ { j } , v _ { j } \}$ for any $\sigma > 0 ,$ , and as $\sigma  0 ,$ , it converges to the hardassignment cost where each particle is assigned to its nearest center.

Proof. The instantaneous branched transport cost for a purely atomic measure $\begin{array} { r } { \rho _ { t } = \sum _ { i } \rho _ { t , i } \delta _ { x _ { i } } } \end{array}$ [4] is given by

$$
\mathcal { F } ( \rho _ { t } ) = \sum _ { i } \| v _ { t , i } \| \rho _ { t , i } ^ { \alpha } .
$$

To approximate this functional with a differentiable surrogate, we discretize space into $K$ centers $\left\{ c _ { k } \right\}$ and replace the discontinuous Voronoi assignment with a soft weighting.

In the hard, non-differentiable case, the partition of space into Voronoi cells $\{ V _ { k } \}$ yields

$$
\begin{array} { l } { m _ { k } ^ { \mathrm { h a r d } } = \displaystyle \sum _ { j = 1 } ^ { B } \rho _ { j } \ \mathbf { 1 } _ { x _ { j } \in V _ { k } } , } \\ { \bar { v } _ { k } ^ { \mathrm { h a r d } } = \displaystyle \frac { \sum _ { j = 1 } ^ { B } \rho _ { j } \ v _ { j } \ \mathbf { 1 } _ { x _ { j } \in V _ { k } } } { \sum _ { j = 1 } ^ { B } \rho _ { j } \ \mathbf { 1 } _ { x _ { j } \in V _ { k } } } . } \end{array}
$$

This construction is non-differentiable because the indicator $\mathbf { 1 } _ { x _ { j } \in V _ { k } }$ is discontinuous.

We therefore replace it with smooth, normalized kernel weights $w _ { j k }$ as defined above. The Gaussian kernel ensures that $w _ { j k }$ is infinitely differentiable in $x _ { j }$ . As $\sigma  0$ , the kernel becomes sharply peaked, so that

$$
w _ { j k } \ \to \ \mathbf { 1 } \biggl [ k = \arg \operatorname* { m i n } _ { \ell } \left\| x _ { j } - c _ { \ell } \right\| \biggr ] .
$$

Consequently,

$$
m _ { k } \ \to \ m _ { k } ^ { \mathrm { h a r d } } , \bar { v } _ { k } \ \to \ \bar { v } _ { k } ^ { \mathrm { h a r d } } .
$$

Substituting these quantities into the definition of $C _ { \mathrm { s o f t } }$ gives

$$
C _ { \mathrm { s o f t } } = \sum _ { k = 1 } ^ { K } \left\| \bar { v } _ { k } \right\| m _ { k } ^ { \alpha } = \sum _ { k = 1 } ^ { K } \left\| \frac { \sum _ { j } w _ { j k } v _ { j } } { \sum _ { j } w _ { j k } } \right\| \left( \sum _ { j } w _ { j k } \rho _ { j } \right) ^ { \alpha } .
$$

Each operation involved (sums, norms, powers, and exponentials) is differentiable for $\sigma > 0$ and $\varepsilon > 0 .$ , ensuring that $C _ { \mathrm { s o f t } }$ is a smooth, differentiable approximation of the branched transport cost.

In the limit $\sigma  0$ , the soft assignments converge to hard nearest-center assignments, and hence

$$
\operatorname* { l i m } _ { \sigma \to 0 } C _ { \mathrm { s o f t } } = \sum _ { k = 1 } ^ { K } \| \bar { v } _ { k } ^ { \mathrm { h a r d } } \| ( m _ { k } ^ { \mathrm { h a r d } } ) ^ { \alpha } .
$$

If the centers coincide with the atom positions $( c _ { k } = x _ { k } )$ and $K = B$ , this expression recovers exactly

$$
\mathcal { F } ( \rho _ { t } ) = \sum _ { i = 1 } ^ { B } \| v _ { i } \| \rho _ { i } ^ { \alpha } .
$$

Thus, the proposed $C _ { \mathrm { s o f t } }$ provides a differentiable surrogate that converges to the atomic branched transport cost in the small-σ limit. □

This differentiable reformulation replaces the hard, non-differentiable atomic constraint with a smooth objective that still promotes the desired clustering and branching behavior. It enables gradient-based learning of velocity fields that generate transport maps with the geometric structure characteristic of branched optimal transport, thereby providing a practical and effective objective for solving the branched optimal transport problem.

## Gaussian Data

In this section, we further analyze the role of the branching parameter α in shaping transport geometry through a series of Gaussian ablations (Fig. 6). These experiments provide both qualitative and structural insight into how trajectory behavior evolves across regimes. For larger values of α, trajectories become increasingly independent, closely resembling classical optimal transport solutions. In contrast, smaller values of α encourage mass to follow shared pathways, giving rise to pronounced trunk-and-branch structures. This transition highlights how α governs the balance between indepen dent transport and cooperative routing, ultimately controlling the geometric organization of mass flow.

## Image Data

In this supplementary section, we extend the analysis presented in the main paper with additional qualitative and quantitative results on MNIST and FFHQ.

For MNIST, Figure 7 provides additional samples generated by BOTA. These results illustrate that the model produces clean, well-formed digits across classes, with consistent stroke thickness and minimal visual artifacts. This complements the main paper by showing that BOTA maintains stable trajectories even in discrete, multidimensional datasets, resulting in reliable class-conditional generations.

![](images/f4b0cbb59b511de3b68db769f05f3f8a7e204a0468c9f7963ab13efcf5c23c35.jpg)  
Figure 7: Examples of MNIST digits generated by BOTA.

Table 2: FID scores comparison on the FFHQ dataset.
<table><tr><td>Method</td><td>FID</td></tr><tr><td>FM</td><td> $\overline { { 1 0 . 6 1 \pm 1 . 3 9 } }$ </td></tr><tr><td>BOTA</td><td> $\mathbf { 1 0 . 4 6 \pm 1 . 8 0 }$ </td></tr></table>

For high-resolution natural images, Table 2 re-

ports the full FID scores on FFHQ. Also BOTA achieves a lower FID compared to Flow Matching, indicating that the learned transport field aligns more accurately with the underlying data distribution. The improvement in FID is consistent across seeds and corroborates the qualitative differences shown in the main paper.

![](images/8adfee502af82ddde5ed1b5397c81810322884021afcda4a6726cf023d9e8dbe.jpg)  
(a) 3 Gaussians (α = 0.1)

![](images/11c43359d8cc323a3eb9b0e0f2558865f586955761c1939ff75324a25ae753af.jpg)  
(b) 6 Gaussians (α = 0.1)

![](images/f6eb74faaf6e511665e403fd5ec9c0434bfbc20c6187bb7596fb2ad2398fe00c.jpg)  
(c) 18 Gaussians (α = 0.1)

![](images/b48e98db2ef22b43257b03e73070d00a41b4c9bc9017c50b34c64d0239b9d26c.jpg)  
(d) 3 Gaussians (α = 0.5)

![](images/e8bd73a59464ce988f8fb97c9ec283a7505e3f1dec2bbd63b18b362a0d498223.jpg)  
(e) 6 Gaussians (α = 0.5)

![](images/17a6c1a6f80a02d7baa2828267a84b0e575c1f937f1d4df97d5b30f9d17f5b9e.jpg)  
(f) 18 Gaussians (α = 0.5)

![](images/5681b11d7b78c21f1c826194e2172ec30c5440214d706f78d62e9e1c694aae73.jpg)  
(g) 3 Gaussians (α = 0.9)

![](images/f6c9739dfa11c73df148a7a9e763491391ac8b466531d23b434c24d6a07690fc.jpg)  
(h) 6 Gaussians (α = 0.9)

![](images/5d260964c9db5f7a5cc00dcdba2ac6d60dbfb012b18e721832a63ccadc7ede19.jpg)  
(i) 18 Gaussians (α = 0.9)  
Figure 6: BOTA Gaussian mixture results for increasing number of Gaussians (3, 6, 18) and different branching parameters α. Top row: α = 0.1. Middle row: α = 0.5. Bottom row: α = 0.9.

## Biological Data

We evaluate BOTA on the Tedsim dataset, a controlled benchmark with known branching differentiation dynamics. Tedsim simulates cellular development from a single progenitor into multiple terminal states, providing both gene expression data and lineage information, making it well-suited for assessing the recovery of branching trajectories.

As shown in Fig. 8, BOTA accurately reconstructs the underlying bifurcation structure, producing smooth trajectories that align with the true developmental paths. The model captures a shared progenitor trunk followed by coherent lineage splits, without mode collapse or spurious branching. Quantitatively, BOTA achieves substantially lower soft-atomic cost and competitive distributional

![](images/92a5e9db0a4d076f9ba0966613cd6c71575cb96c132362f616b77a5a6b33e29a.jpg)

<table><tr><td>Method</td><td>Cost↓</td></tr><tr><td>FM</td><td> $4 0 5 . 0 8 \pm 1 . 0 0$ </td></tr><tr><td>CFM</td><td> $4 0 8 . 8 8 \pm 1 . 0 0$ </td></tr><tr><td>BOTA</td><td> $\mathbf { 1 7 4 . 8 1 \pm 5 . 0 2 }$ </td></tr></table>

(a) Learned branched transport on Tedsim dataset.  
(b) Soft-atomic cost comparison.  
![](images/41224da2420fb26fd010e617d8b045724964bc6279c5e3d7d6905b47685aae94.jpg)  
(c) Quantitative metrics $( W _ { 1 } , W _ { 2 } ,$ RBF-MMD) with $\alpha = 0 . 5 .$  
Figure 8: BOTA results on the Tedsim dataset combining qualitative transport structure and quantitative evaluation.  
metrics $( W _ { 1 } , W _ { 2 }$ , RBF-MMD), confirming that the learned transport is both structurally meaningful and statistically consistent.

Runtime and complexity. The branched machinery adds a bounded, one-off training overhead and no sampling overhead. Stage 1 optimizes the discrete velocity tensor $\dot { V } \in \mathbb { R } ^ { T \times N \times D }$ directly, with no neural network: each iteration costs $O ( T \cdot N \cdot K \cdot D )$ , linear in the number of particles N, with no $N ^ { 2 }$ term (unlike an OT or Sinkhorn pairing, which costs $O ( N ^ { 2 } D )$ before training begins). Stage 2 is unmodified flow matching, so its cost equals that of the FM baseline. Table 3 reports wall-clock training times on Tedsim (50-D PCA) under the identical training protocol of Table 1. In total, BOTA trains in roughly $2 \times$ the time of FM, paid once offline, while sampling cost is identical to FM since inference integrates a single velocity field of the same architecture.

Table 3: Training cost on Tedsim (50-D PCA). The discrete solver stage involves no network training; the amortization stage is standard flow matching. Sampling cost is identical for all methods.
<table><tr><td></td><td>FM</td><td>CFM</td><td>BOTA</td></tr><tr><td>Solver stage</td><td></td><td></td><td>425 s (no network)</td></tr><tr><td>Network training</td><td>360 s</td><td>480 s</td><td>360 s (unmodified FM)</td></tr><tr><td>Total training</td><td>1× FM</td><td>1.3× FM</td><td>≈ 2× FM</td></tr><tr><td>Per-iteration cost in N</td><td>O(N)</td><td> $O ( N ^ { 2 } )$  if OT-paired</td><td> $O ( T \cdot N \cdot K \cdot D )$  , linear in N</td></tr><tr><td>Sampling cost</td><td>1×</td><td>1×</td><td>1×</td></tr></table>