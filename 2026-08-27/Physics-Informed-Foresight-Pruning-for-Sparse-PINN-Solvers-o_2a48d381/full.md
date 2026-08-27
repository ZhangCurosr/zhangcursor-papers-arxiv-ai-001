# Physics-Informed Foresight Pruning for Sparse PINN Solvers of Nonlinear PDEs

Ahmad Ishaque Karimi

Uvini Balasuriya Mudiyanselage

Kookjin Lee

School of Computing and

School of Computing and

School of Computing and

Augmented Intelligence

Augmented Intelligence

Augmented Intelligence

Arizona State University

Arizona State University

Tempe, USA

Arizona State University

aikarimi@asu.edu

Tempe, USA

ubalasur@asu.edu

Tempe, USA

Kookjin.Lee@asu.edu

Abstract—Physics-informed neural networks (PINNs) often rely on over-parameterized models to optimize coupled solution and differential-residual objectives, leaving unclear how much capacity is necessary and what pruning should preserve. We study foresight pruning at initialization for sparse PirateNet PDE solvers. Standard neural tangent kernel spectrum-aware pruning (NTK-SAP) aims to preserve output-side training dynamics but may overlook parameters whose main influence arises through derivatives in the governing equations. We introduce physics-informed spectrum-aware pruning (PI-SAP), which assigns saliency using sensitivity of the PDE residual. Experiments on the Gray–Scott equations, complex Ginzburg–Landau equation, Burgers’ equation, and linear convection equation show that PI-SAP more consistently preserves Gray–Scott residual fidelity and is competitive under aggressive sparsity. However, no criterion is uniformly optimal across equations or sparsity levels. Small-batch PINN-NTK diagnostics further show that residual fidelity, solution accuracy, and kernel conditioning are distinct objectives, motivating pruning methods that explicitly balance solution-side and residual-side training dynamics during optimization.

Index Terms—physics-informed neural networks, PirateNets, pruning, neural tangent kernel, scientific machine learning, sparsity

## I. INTRODUCTION

Physics-informed neural networks (PINNs) approximate the solution of partial differential equations (PDEs) by combining data, initial or boundary conditions, and the governing residual in a differentiable training objective [1]. They are attractive for scientific machine learning because they avoid an explicit mesh at training time and represent the learned solution as a continuous function. However, the same features that make PINNs flexible also make them difficult to train. Deep coordinate networks can favor low-frequency modes [3], derivative networks can be poorly conditioned at initialization, and the competing initial-condition and residual losses can create gradient imbalance [4], [5]. Recent convergence analyses also emphasize that residual minimization alone is not always sufficient to guarantee reliable approximation of the target physical solution [6].

These difficulties are amplified in nonlinear PDEs with sharp gradients or coupled fields. Reaction–diffusion and dispersive systems require a network to represent not only solution values but also the spatial and temporal derivatives appearing in the residual [1], [4]. Modern PINN architectures therefore use substantial model capacity to make optimization feasible; residual-adaptive PirateNets are one recent example [7]. That capacity, however, increases memory and training cost. This computational burden is further amplified in parametric settings, where multi-query scenarios require resolving the PDE across a range of parameter configurations [8], [9]. Foresight pruning [10]–[12], which removes weights at initialization before training begins, offers a way to reduce this redundancy while keeping the training protocol fixed. The central question is therefore not simply whether a PINN can be pruned, but which notion of weight importance should be preserved.

Neural tangent kernel spectrum-aware pruning (NTK-SAP) provides a natural starting point for this question. It is motivated by neural tangent kernel (NTK) theory, where the spectrum of the NTK controls gradient-descent training dynamics in wide networks [2], [12]. For ordinary supervised learning, preserving the output-side NTK spectrum is a principled way to preserve the dense model’s training behavior under pruning. For PINNs, however, output-side preservation may be incomplete. Let θ denote the trainable network parameters. A parameter with limited influence on the predicted fields [u<sub>θ</sub>, v<sub>θ</sub>] can still strongly affect $\left[ \partial _ { t } u _ { \theta } , \Delta u _ { \theta } , \partial _ { t } v _ { \theta } , \Delta v _ { \theta } \right]$ and therefore the PDE residual. This observation motivates physics-informed spectrum-aware pruning (PI-SAP), a saliency rule that scores weights through the residual network rather than through the raw output map.

We evaluate this idea using residual-adaptive PirateNets [7]. PirateNets initialize as shallow stable models and progressively open deeper nonlinear paths during training, making them a strong architecture for stiff PINN problems. Our most complete experiment is the two-dimensional Gray–Scott reaction–diffusion system, whose stripe regime contains winding, derivative-sensitive interfaces. We also report the complex Ginzburg–Landau equation, Burgers’ equation, and convection equation experiments to test whether the observed behavior is specific to the Gray–Scott equations or reflects a broader cross-PDE pruning pattern.

The experiments reveal a qualified advantage rather than a universal winner. PI-SAP improves residual fidelity of the Gray–Scott equations throughout the sparsity sweep and is particularly competitive at high sparsity, whereas NTK-SAP can be stronger at intermediate sparsity. The two criteria therefore protect complementary aspects of training: outputside dynamics and residual-sensitive physics.

The PINN-NTK perspective formalizes this distinction because solution and residual losses evolve through different kernel blocks and can converge at different rates [14]. This motivates two diagnostic extensions. Conditioning-aware NTK-SAP tests whether avoiding small eigenvalues and excessive spectral spread improves difficult modes, while PINN-blockaware SAP separately inspects small-batch solution and residual kernels. Together, they test whether a useful sparse model must balance both dynamics rather than optimize either one in isolation.

The paper makes three contributions. First, we benchmark NTK-SAP and PI-SAP across the Gray–Scott equations, complex Ginzburg–Landau equation, Burgers’ equation, and convection equation PDEs to characterize when physics-informed saliency helps and when output-spectrum pruning remains competitive. Second, we show that PI-SAP improves residual fidelity of the Gray–Scott equations at every pruning level and substantially reduces high-frequency error at 70% pruning, supporting the claim that residual-aware saliency protects derivative-sensitive structure. Third, through diagnostic experiments at 70% pruning on the Gray–Scott equations, we show that improving NTK conditioning or preserving residual-side dynamics does not necessarily improve solution accuracy. This reveals a trade-off among solution accuracy, physics-residual fidelity, and optimization dynamics, motivating pruning criteria that explicitly balance these objectives.

## II. TECHNICAL BACKGROUND

## A. PINN Objective

A PINN can use the same feed-forward architecture as a conventional supervised neural network: coordinates enter the network and predicted fields leave it. The difference is the training signal. A supervised network primarily compares its outputs with labeled targets, whereas a PINN also differentiates its outputs with respect to the input coordinates, substitutes those derivatives into the governing PDE, and minimizes the resulting residual together with initial or boundary errors. Thus, the physics constraints supplement or replace much of the labeled supervision; they do not require a fundamentally different network architecture.

Let $z \ = \ ( t , \mathbf { x } ) \ \in \ [ 0 , T ] \times \Omega$ denote a space–time coordinate and let $q _ { \theta } : [ 0 , T ] \times \Omega \to \mathbb { R } ^ { d _ { q } }$ be a neural solution parameterized by $\boldsymbol { \theta } \in \mathbb { R } ^ { \bar { P } }$ . Here $\mathbf x = x$ for a one-dimensional spatial domain and $\mathbf { x } = ( x , y )$ for a two-dimensional domain; consequently, the implementation for the Gray–Scott equations uses $z = ( t , x , y )$ . For a governing differential operator ${ \mathcal { N } } ,$ define the pointwise physics residual as $r _ { \theta } ( z ) = \mathcal { N } [ q _ { \theta } ] ( z )$ . A typical PINN objective combines an initial/boundary constraint loss with an interior residual loss,

$$
\mathcal { L } ( \theta ) = \lambda _ { \mathrm { i c } } \mathcal { L } _ { \mathrm { i c } } ( \theta ) + \lambda _ { \mathrm { r } } \mathcal { L } _ { \mathrm { r } } ( \theta ) ,\tag{1}
$$

where

$$
\mathcal { L } _ { \mathrm { { r } } } ( \theta ) = \frac { 1 } { N _ { r } } \sum _ { i = 1 } ^ { N _ { r } } \left\| \mathcal { N } [ q _ { \theta } ] ( t _ { i } , \mathbf { x } _ { i } ) \right\| _ { 2 } ^ { 2 } .\tag{2}
$$

Here $\lambda _ { \mathrm { i c } } , \lambda _ { \mathrm { r } } \geq 0$ are scalar loss weights, ${ \mathcal { L } } _ { \mathrm { i c } }$ enforces the prescribed initial and boundary data, and $\{ ( t _ { i } , \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N _ { r } }$ are the $N _ { r }$ interior collocation points. For coupled systems such as the Gray–Scott equations or the complex Ginzburg–Landau equation, $q _ { \theta } = [ u _ { \theta } , v _ { \theta } ]$ and $r _ { \theta } = [ r _ { u } , r _ { v } ]$ contains one residual component for each field. The residual depends on automatic differentiation through the network, so a pruning mask can change not only the predicted fields but also their derivatives.

## B. PirateNet Architecture and Pruning Scope

We use a physics-informed residual adaptive network (PirateNet) as the backbone of each PINN solver [7]. PirateNets are designed to avoid unstable initialization of deep PDE residual networks by beginning as shallow mappings and progressively introducing nonlinear depth during training.

For the two-dimensional benchmarks, the input coordinate $\boldsymbol { z } = ( t , x , y )$ is first transformed using fixed periodic encodings of x and y, followed by a trainable Fourier-feature embedding of dimension 256. Two parallel dense transformations of this embedding produce auxiliary feature streams U and V . The embedded representation then passes through three residual-adaptive blocks of width 256 with Swish activations. For the one-dimensional benchmarks, the coordinate input and activation function follow the corresponding benchmark configuration; periodic encoding is used only when specified.

Figure 1 summarizes the PirateNet backbone and distinguishes parameters eligible for pruning from those kept dense. Within each residual-adaptive block, three dense transformations are interleaved with two modulation operations that incorporate the auxiliary feature streams. If $h _ { \ell }$ denotes the input to block ℓ and $H _ { \ell }$ denotes its nonlinear transformation, the block output is

$$
h _ { \ell + 1 } = \alpha _ { \ell } H _ { \ell } + ( 1 - \alpha _ { \ell } ) h _ { \ell } ,\tag{3}
$$

where $\alpha _ { \ell }$ is a trainable scalar gate. Each gate is initialized as $\alpha _ { \ell } \ = \ 0 ,$ making every block an identity mapping at initialization. As the gates evolve during optimization, the nonlinear blocks are progressively introduced. A final linear projection maps the learned representation to the predicted PDE fields, such as $q _ { \theta } ( z ) = [ u _ { \theta } ( z ) , v _ { \theta } ( z ) ]$ for the coupled Gray–Scott equations. Its coefficients are initialized through a least-squares fit to the initial state of each temporal window.

Pruning is applied globally only to trainable parameter leaves stored as kernels. This eligible set includes the Fourierfeature kernel, the two auxiliary-stream kernels, and the kernels within the three residual-adaptive blocks. Because the dense layers use random weight factorization, both factors in their kernel parameterization are included in the eligible set.

![](images/918d9d5dfac7967946ba99d23778f4ae92e5e2c1bd1b1ab05c06a3d506637719.jpg)  
Fig. 1. PirateNet backbone and pruning scope used in this study. The upper view summarizes the coordinate embedding, auxiliary feature streams, adaptiv blocks, and output projection, while the lower view details one adaptive block. Blue components contain kernel parameters eligible for global pruning by NTK-SAP or PI-SAP. Adaptive gates $\alpha _ { \ell } ,$ biases, and physics-informed output coefficients remain dense. The architecture follows the PirateNets design of Wang et al. [7]; the pruning-scope representation is specific to our implementation.

The reported pruning percentage is computed globally over these eligible parameters. Biases, adaptive gates $\alpha _ { \ell } ,$ and the physics-informed output coefficients are kept dense.

After saliency scoring, eligible parameters below the global threshold are set to zero. The resulting binary mask is applied to both their gradients and parameter values after every Adam update, preventing pruned parameters from regrowing through optimizer momentum. NTK-SAP and PI-SAP therefore use the same PirateNet architecture, eligible parameter set, and maskenforcement procedure; they differ only in the quantity used to calculate parameter saliency.

## C. NTK View

Let $f _ { \theta } ~ : ~ \mathbb { R } ^ { d _ { \mathrm { i n } } } ~  ~ \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ be a differentiable feed-forward network; the PINN solution $q _ { \theta }$ is one such map. For an input batch X, let $f _ { \theta } ( X )$ denote its stacked output vector. The Jacobian $J _ { \theta } ( X ) = \partial f _ { \theta } ( X ) / \partial \theta$ contains the derivative of every stacked output with respect to every trainable parameter. The empirical neural tangent kernel (NTK) is

$$
K ( X , X ) = J _ { \theta } ( X ) J _ { \theta } ( X ) ^ { \top } , \qquad J _ { \theta } ( X ) = { \frac { \partial f _ { \theta } ( X ) } { \partial \theta } } .\tag{4}
$$

NTK theory connects the eigenspectrum of K to gradientdescent training dynamics [2]. In the linearized NTK regime for squared loss, error components aligned with an eigenvector of K decay at a rate controlled by the corresponding eigenvalue. Small eigenvalues therefore create slow modes, while a wide eigenvalue spread creates uneven learning across modes. In this paper, “conditioning” refers to this eigenspectral conditioning of the empirical kernel, for example through the condition number $\kappa ( K ) = \lambda _ { \mathrm { m a x } } ( K ) / \lambda _ { \mathrm { m i n } } ( K )$ , not to input normalization or an optimizer preconditioner. This motivates NTK-preserving compression: if a sparse model preserves the relevant NTK spectrum, it should preserve important aspects of the dense model’s optimization behavior. NTK-SAP follows this principle by pruning connections that have little influence on an NTK-spectrum proxy [12]. Related high-dimensional NTK compression theory also supports the idea that spectral equivalence can preserve convergence and generalization behavior under compression [13].

The PINN-NTK analysis of Wang et al. [14] shows that PINN training can be understood through NTK blocks associated with different loss components, and that solution and residual terms can converge at different rates. Thus, matching only the output-side dynamics is not guaranteed to preserve residual-side dynamics. This motivates a pruning criterion that sees the differential operator.

## D. NTK-SAP Baseline

NTK-SAP is a foresight pruning method: the sparse mask is selected before training. The original method avoids forming a full NTK eigenspectrum by using a tractable trace/nuclearnorm proxy. In our PINN implementation, the mask is computed at initialization and then enforced throughout training by masking both gradients and parameters, preventing pruned connections from regrowing through optimizer momentum.

At a high level, NTK-SAP introduces a mask $m \in [ 0 , 1 ] ^ { P }$ and scores each mask variable $m _ { j }$ using the sensitivity of a perturbed output difference,

$$
S _ { \mathrm { N T K } } ( m _ { j } ) = \left| \frac { \partial } { \partial m _ { j } } \left. f _ { \boldsymbol { \theta } \odot m } ( Z ) - f _ { ( \boldsymbol { \theta } + \Delta \boldsymbol { \theta } ) \odot m } ( Z ) \right. _ { 2 } ^ { 2 } \right| ,\tag{5}
$$

where $Z$ is a pruning input batch, ∆θ is a small parameter perturbation, and ⊙ denotes the elementwise (Hadamard) product. Only ordinary kernel weights are eligible for pruning;

biases, PirateNet α gates, and other initialization coefficients retain mask value one. For target sparsity s, a global threshold is set at the 100s-th percentile of all eligible scores: kernel weights at or below the threshold are assigned mask value zero, while higher-scoring weights are retained. Thus, the rule removes the eligible weights judged least influential to the output-dynamics proxy.

## III. PROPOSED METHOD

## A. PI-SAP

For a coupled PDE residual $r _ { \theta } = [ r _ { u } , r _ { v } ]$ evaluated on a residual collocation batch $X _ { r }$ , PI-SAP scores

$$
S _ { \mathrm { P I } } ( m _ { j } ) = \left| \frac { \partial } { \partial m _ { j } } \left. r _ { \theta \odot m } ( X _ { r } ) - r _ { ( \theta + \Delta \theta ) \odot m } ( X _ { r } ) \right. _ { 2 } ^ { 2 } \right| .\tag{6}
$$

PI-SAP uses the same global ranking and target sparsity s as NTK-SAP. The practical intent is simple: protect the connections that most affect satisfaction of the governing PDE. For reaction–diffusion systems, this means preserving weights that influence the derivative-sensitive diffusion and reaction terms, not merely weights that influence the raw output magnitude.

## B. Diagnostic Extensions

Beyond residual saliency, we consider two diagnostic objectives that use small-batch empirical kernels during mask selection. Their definitions are independent of the governing PDE and can be applied whenever differentiable solution and residual maps are available. The first, conditioning-aware NTK-SAP, examines the empirical kernel of a selected network output map $f _ { \theta }$ . Instead of only preserving a dense-model spectral proxy, it favors masks whose small-batch empirical NTK has larger average scale and less spectral collapse. This is not an architectural preconditioner and does not form the full training-set NTK. For a small pruning batch $Z ,$ it computes $J = \partial f _ { \theta } ( Z ) / \partial \theta$ and forms

$$
K = J J ^ { \top }\tag{7}
$$

as a compact diagnostic/objective. For eigenvalues $\lambda _ { i } ^ { \epsilon } ~ =$ $\operatorname* { m a x } ( \lambda _ { i } , \epsilon )$ , the implemented objective has the form

$$
\phi ( K ) = \log \bar { \lambda } ^ { \epsilon } - \alpha _ { c } ( \log \lambda _ { \operatorname* { m a x } } ^ { \epsilon } - \log \lambda _ { \operatorname* { m i n } } ^ { \epsilon } ) - \alpha _ { s } \operatorname { s t d } ( \log \lambda _ { i } ^ { \epsilon } ) ,\tag{8}
$$

where $\epsilon > 0$ is an eigenvalue floor, $\bar { \lambda } ^ { \epsilon }$ is the mean clipped eigenvalue, and $\alpha _ { c } , \alpha _ { s } \ge 0$ weight the condition-number and log-spectrum-spread penalties, respectively. The mask saliency is computed from $| \theta _ { j } \partial \phi / \partial \theta _ { j }$ .

The second, PINN-block-aware SAP, separates solutionside and residual-side empirical kernels. For the shared small batch X used by the current implementation, define $J _ { \mathrm { s o l } } =$ $\partial q _ { \theta } ( X ) / \partial \theta$ and $J _ { \mathrm { r e s } } = \partial r _ { \theta } ( X ) / \partial \theta ;$ ; for a two-field system these stack the component Jacobians $[ J _ { u } ; J _ { v } ]$ and $[ J _ { r _ { u } } ; J _ { r _ { v } } ]$ respectively. The corresponding kernel blocks are

$$
K _ { \mathrm { s o l } } = J _ { \mathrm { s o l } } J _ { \mathrm { s o l } } ^ { \top } ,\tag{9}
$$

$$
K _ { \mathrm { r e s } } = J _ { \mathrm { r e s } } J _ { \mathrm { r e s } } ^ { \top } .\tag{10}
$$

These are practical small-batch analogs of the $K _ { u u }$ and $K _ { r r }$ blocks used in the PINN-NTK view. The block-aware objective combines $\phi ( K _ { \mathrm { s o l } } )$ and $\phi ( K _ { \mathrm { r e s } } )$ and penalizes mismatch in their mean log-eigenvalue scales. Cross terms $K _ { u r } = J _ { \mathrm { s o l } } J _ { \mathrm { r e s } } ^ { \top }$ and $K _ { r u } = K _ { u r } ^ { \top }$ are left for future work.

## IV. EXPERIMENTAL SETUP

The main experiments on the Gray–Scott equations use the scaled system

$$
u _ { t } = \epsilon _ { 1 } \Delta u + b _ { 1 } ( 1 - u ) - c _ { 1 } u v ^ { 2 } ,\tag{11}
$$

$$
v _ { t } = \epsilon _ { 2 } \Delta v - b _ { 2 } v + c _ { 2 } u v ^ { 2 } .\tag{12}
$$

The domain is $\Omega ~ = ~ [ - 1 , 1 ] ^ { 2 }$ with $t ~ \in ~ [ 0 , 2 ]$ , 101 saved snapshots, and a $2 0 0 \times 2 0 0$ reference grid. The parameters are $\epsilon _ { 1 } = 0 . 2 , \epsilon _ { 2 } = 0 . 1 , b _ { 1 } = 4 0 , b _ { 2 } = 1 0 0$ , and $c _ { 1 } = c _ { 2 } = 1 0 0 0$ With the code scaling, this corresponds approximately to the standard feed/kill values for the Gray–Scott equations, $F = 0 . 0 4$ and $k ~ = ~ 0 . 0 6$ because $b _ { 1 } ~ = ~ 1 0 0 0 F$ and $b _ { 2 } \ =$ 1000( $F + k )$ .

All models for the Gray–Scott equations use PirateNet with 3 residual adaptive blocks, width 256, Swish activation, Fourier features of dimension 256, Adam optimization, Grad-Norm weighting, and causal time marching over 10 temporal windows. The completed sparsity sweep for the Gray–Scott equations uses five pruning levels: 10%, 30%, 50%, 70%, and 90%. The diagnostic extension runs use the same stripe-like regime at 70% pruning with a shorter budget of 60,000 steps per window.

For the complex Ginzburg–Landau equation, we report dense, NTK-SAP, and PI-SAP runs across the same five pruning levels. For Burgers’ equation and convection equation, we use prior cross-PDE studies averaged over seeds 0–4 with 50k Adam steps. These latter experiments are not as deep as the Gray–Scott study, but they are useful for testing whether the observed sparsity trends are equation-specific.

We report relative $L ^ { 2 }$ error for each field and the mean solution error $( e _ { u } + e _ { v } ) / 2$ when two fields are present. For the Gray–Scott equations and the Ginzburg–Landau equation, we also report mean PDE residual $( \ell _ { r _ { u } } + \ell _ { r _ { v } } ) / 2$ . Runtime is reported as wall-clock training time for the current masked implementation. Because the implementation masks dense arrays rather than invoking sparse kernels, pruning is not expected to produce proportional wall-clock speedups.

## V. RESULTS

## A. Gray–Scott Equations: Main Case Study

Table I is the main result. PI-SAP has lower mean solution error at 10%, 30%, 70%, and 90% pruning, while NTK-SAP is slightly better at 50%. The larger distinction appears in the physics residual: PI-SAP has lower mean PDE residual at every pruning level.

At 10% pruning, PI-SAP reduces mean error by about 34.1% relative to NTK-SAP. At 70%, it reduces mean error by about 28.8%. At 90%, PI-SAP still improves mean error by about 12.9%, even though both methods have entered a degraded high-sparsity regime. The 50% exception is important: residual-aware saliency is not automatically better for every sparsity. However, the high-sparsity behavior supports the main hypothesis that physics-informed saliency becomes more valuable when the sparse model has fewer redundant paths.

TABLE I  
RESULTS FOR THE GRAY–SCOTT EQUATIONS IN THE STRIPE-FORMING REGIME. ∆ DENOTES THE PI-SAP MEAN-ERROR CHANGE RELATIVE TO NTK-SAP; NEGATIVE VALUES INDICATE IMPROVEMENT.
<table><tr><td>Prune</td><td>NTK err.</td><td>PI err.</td><td>∆</td><td>NTK res.</td><td>PI res.</td></tr><tr><td>10%</td><td> $1 . 1 4 3 9 \times 1 0 ^ { - 2 }$ </td><td> $7 . 5 3 4 3 \times 1 0 ^ { - 3 }$ </td><td>-34.1%</td><td> $1 . 2 9 9 6 \times 1 0 ^ { - 5 }$ </td><td> $1 . 1 4 0 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>30%</td><td> $1 . 7 8 9 6 \times 1 0 ^ { - 2 }$ </td><td> $1 . 7 5 5 5 \times 1 0 ^ { - 2 }$ </td><td>-1.9%</td><td> $1 . 5 3 4 5 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 2 0 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>50%</td><td> $4 . 3 0 6 7 \times 1 0 ^ { - 2 }$ </td><td> $4 . 4 7 2 7 \times 1 0 ^ { - 2 }$ </td><td> $+ 3 . 9 \%$ </td><td> $2 . 6 9 5 3 \times 1 0 ^ { - 5 }$ </td><td> $2 . 1 3 1 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>70%</td><td> $2 . 7 2 9 1 \times 1 0 ^ { - 1 }$ </td><td> $1 . 9 4 3 0 \times { { 1 0 } ^ { - 1 } }$ </td><td>-28.8%</td><td> $5 . 1 4 8 2 \times { { 1 0 } ^ { - 5 } }$ </td><td> $4 . 8 4 1 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>90%</td><td> $4 . 3 4 9 3 \times 1 0 ^ { - }$  -1</td><td> $3 . 7 8 6 6 \times 1 0 ^ { - }$  -1</td><td>-12.9%</td><td> $3 . 8 9 2 1 \times 1 0 ^ { - }$  -4</td><td> $3 . 8 7 1 3 \times 1 0 ^ { - }$  4</td></tr></table>

TABLE II

TERMINAL HIGH-FREQUENCY ERRORS FOR THE GRAY–SCOTT EQUATIONS AT 70% PRUNING.
<table><tr><td>Method</td><td>u high-freq.</td><td>v high-freq.</td></tr><tr><td>Dense PirateNet</td><td>0.0227</td><td>0.0277</td></tr><tr><td>NTK-SAP 70%</td><td>0.7238</td><td>0.7749</td></tr><tr><td>PI-SAP 70%</td><td>0.3591</td><td>0.4132</td></tr></table>

The residual trend is more consistent. PI-SAP improves the mean PDE residual at every pruning level, with the largest residual reductions at 10% and 50%. This does not mean lower residual always implies lower solution error, but it does show that residual-sensitive pruning better preserves sampled physics consistency across the full sparsity sweep for the Gray–Scott equations. Both methods exhibit a sharp jump in solution error between 50% and 70%, suggesting that the practical accuracy threshold for this PirateNet configuration lies in that interval. Runtime stays in the 16.4–18.0 hour range for these runs of the Gray–Scott equations, with no systematic wall-clock advantage from masking.

## B. High-Frequency Diagnostic for the Gray–Scott Equations

The stripe regime of the Gray–Scott equations is visually governed by sharp spatial interfaces. To check whether the L2 trends reflect loss of high-frequency content, we computed terminal-time Fourier errors at 70% pruning from saved checkpoints. The high-frequency mask keeps spatial modes whose radial frequency is at least 0.35 of the maximum discrete frequency.

PI-SAP roughly halves the terminal high-frequency error relative to NTK-SAP for both species. This strengthens the interpretation that residual-aware saliency better protects the derivative-sensitive connections needed to resolve stripe interfaces.

## C. NTK-Block Diagnostics for the Gray–Scott Equations

Table III summarizes the diagnostic extensions for the Gray–Scott equations at 70% pruning. These runs use a shorter

TABLE III  
DIAGNOSTIC EXTENSIONS FOR THE GRAY–SCOTT EQUATIONS AT 70% PRUNING AND 600K TOTAL STEPS.
<table><tr><td>Method</td><td>u err.</td><td>v err.</td><td> $r _ { u } \ 1 0 5 8$ </td><td> $r _ { v }$  loss</td></tr><tr><td>Dense</td><td>0.0264</td><td>0.0489</td><td> $2 . 2 6 2 \times 1 0 ^ { - 4 }$ </td><td> $8 . 2 1 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>NTK-SAP</td><td>0.1786</td><td>0.3229</td><td> $1 . 9 7 3 \times 1 0 ^ { - 4 }$ </td><td> $6 . 9 7 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Cond. NTK-SAP</td><td>0.1966</td><td>0.3516</td><td> $2 . 5 8 9 \times { { 1 0 } ^ { - 4 } }$ </td><td> $9 . 2 7 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>PI-SAP</td><td>0.2006</td><td>0.3578</td><td> $6 . 6 6 7 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 8 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>PINN-block SAP</td><td>0.2249</td><td>0.4021</td><td> $1 . 6 0 5 \times 1 0 ^ { - 4 }$ </td><td> $5 . 6 7 1 \times 1 0 ^ { - 5 }$ </td></tr></table>

TABLE IV

GINZBURG–LANDAU SUMMARY. DENSE MEAN ERROR IS $2 . 8 0 1 7 \times 1 0 ^ { - 2 }$ AND DENSE RESIDUAL IS 1.4557 10<sup>−5</sup>.
<table><tr><td>Prune</td><td>NTK err.</td><td>PI err.</td><td>Better</td><td>NTK res.</td><td>PI res.</td></tr><tr><td>10%</td><td> $2 . 6 0 6 2 \times 1 0 ^ { - }$  -2</td><td> $2 . 5 3 8 4 \times 1 0 ^ { - }$  -2</td><td>PI</td><td> $1 . 0 8 3 1 \times 1 0 ^ { - }$  -5</td><td> $1 . 1 4 0 1 \times 1 0 ^ { - }$  -5</td></tr><tr><td>30%</td><td> $2 . 8 7 6 9 \times 1 0 ^ { - 2 }$ </td><td> $3 . 4 2 5 7 \times 1 0 ^ { - 2 }$ </td><td>NTK</td><td> $1 . 4 9 0 1 \times 1 0 ^ { - 5 }$ </td><td> $1 . 4 7 0 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>50%</td><td> $4 . 6 6 3 4 \times 1 0 ^ { - 2 }$ </td><td> $5 . 4 2 4 4 \times 1 0 ^ { - 2 }$ </td><td>NTK</td><td> $2 . 7 3 5 9 \times 1 0 ^ { - 5 }$ </td><td> $2 . 6 4 8 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>70%</td><td> $6 . 5 6 1 5 \times 1 0 ^ { - 2 }$ </td><td> $5 . 8 0 7 8 \times 1 0 ^ { - 2 }$ </td><td>PI</td><td> $7 . 9 3 5 0 \times 1 0 ^ { - 5 }$ </td><td> $8 . 2 1 8 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>90%</td><td> $1 . 8 8 3 6 \times 1 0 ^ { - 1 }$ </td><td> $1 . 7 4 3 4 \times 1 0 ^ { - 1 }$ </td><td>PI</td><td> $4 . 7 9 2 4 \times 1 0 ^ { - 3 }$ </td><td> $3 . 7 9 2 4 \times 1 0 ^ { - 3 }$ </td></tr></table>

600k-step budget and should not replace the completed sweep in Table I.

The diagnostic result is intentionally nuanced. Under this shorter budget, original NTK-SAP has the best solution accuracy among pruned models. PINN-block-aware SAP has the lowest residual losses among pruned models but the worst solution error. The conditioning-aware NTK-SAP variant also does not improve over the original NTK-SAP result in this setting. Thus, these extensions are best interpreted as diagnostic probes: block-level conditioning terms can improve one target, such as sampled residual loss, without preserving the full solution field. This is the central reason for treating sparse PINN pruning as a multi-objective design problem rather than selecting a single saliency score.

## D. Complex Ginzburg–Landau Equation

Table IV reports the Ginzburg–Landau comparison, including the dense baseline. The result is not a duplicate of Gray– Scott. Both pruning methods improve over dense at 10%, NTK-SAP is better at 30% and 50%, and PI-SAP is better at 70% and 90%.

The Ginzburg–Landau results support a more careful claim than “PI-SAP always wins.” PI-SAP improves mean error by about 11.5% relative to NTK-SAP at 70% pruning and by about 7.4% at 90%, but NTK-SAP is better at 30% and 50%. This agrees with the trend observed for the Gray–Scott equations that residual-aware saliency is most valuable when pruning pressure is aggressive, while also showing that the balance between output and residual saliency depends on the PDE and sparsity level. The runtime remains near 10 hours for all runs, again indicating that mask-based pruning does not automatically translate to wall-clock acceleration without sparse kernels.

## E. Burgers’ Equation and Convection Equation

The Burgers’ equation and convection equation studies provide additional cross-PDE context. They were run over seeds 0–4 for 50k Adam steps. Table V reports the best pruned Burgers’ equation result for each network width, while Table VI reports the best pruned convection equation result for each β–width setting. In both tables, we report only the best pruned result rather than every sparsity level to keep the comparison compact.

TABLE V  
BEST PRUNED RESULTS FOR BURGERS’ EQUATION ACROSS NETWORK WIDTHS.
<table><tr><td>Width</td><td>Dense</td><td>Best pruned</td><td>Method</td></tr><tr><td>128</td><td> $1 . 0 8 4 9 \times 1 0 ^ { - 2 }$ </td><td> $3 . 7 5 3 2 \times 1 0 ^ { - 3 }$ </td><td>PI 30%</td></tr><tr><td>256</td><td> $3 . 1 4 8 5 \times 1 0 ^ { - 2 }$  -1</td><td> $5 . 4 1 6 0 \times 1 0 ^ { - 3 }$ </td><td>PI 30%</td></tr><tr><td>512</td><td> $3 . 0 0 5 1 \times 1 0 ^ { - }$ </td><td> $2 . 5 3 2 4 \times 1 0 ^ { - 3 }$ </td><td> $\mathrm { N T K } 9 0 \%$ </td></tr></table>

TABLE VI

BEST PRUNED RESULTS FOR THE CONVECTION EQUATION ACROSS TRANSPORT PARAMETER β AND NETWORK WIDTH.
<table><tr><td> $\beta$ </td><td>Width</td><td>Dense</td><td>Best pruned</td><td>Method</td></tr><tr><td>1</td><td>128</td><td> $1 . 2 4 9 5 \times 1 0$  -2</td><td> $5 . 2 0 8 0 \times 1 0 ^ { - }$  -3</td><td>PI 30%</td></tr><tr><td></td><td>256</td><td> $1 . 1 2 7 1 \times 1 0 ^ { - 2 }$ </td><td> $5 . 0 3 1 0 \times 1 0 ^ { - }$  -3</td><td> $\mathrm { P I } 7 0 \%$ </td></tr><tr><td></td><td>512</td><td> $1 . 3 2 2 5 \times 1 0 ^ { - 1 }$ </td><td> $7 . 2 1 6 0 \times 1 0 ^ { - 3 }$ </td><td>NTK 90%</td></tr><tr><td> $^ { 5 }$ </td><td>128</td><td> $1 . 6 9 4 8 \times { { 1 0 } ^ { - 2 } }$ </td><td> $6 . 8 2 4 0 \times 1 0 ^ { - 3 }$ </td><td>NTK 50%</td></tr><tr><td></td><td>256</td><td> $9 . 2 2 4 0 \times 1 0 ^ { - 3 }$ </td><td> $9 . 4 5 9 0 \times 1 0 ^ { - 3 }$ </td><td>PI 30%</td></tr><tr><td></td><td>512</td><td> $9 . 6 9 7 0 \times 1 0 ^ { - 3 }$ </td><td> $7 . 4 7 7 0 \times 1 0 ^ { - 3 }$ </td><td>PI 70%</td></tr><tr><td>10</td><td>128</td><td> $2 . 4 6 4 5 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 0 8 9 \times 1 0 ^ { - 2 }$ </td><td>PI 50%</td></tr><tr><td></td><td>256</td><td> $1 . 8 3 3 4 \times 1 0 ^ { - 2 }$ </td><td> $1 . 0 1 2 4 \times 1 0 ^ { - 2 }$ </td><td> $\mathrm { N T K } \ 1 0 \%$ </td></tr><tr><td></td><td>512</td><td> $3 . 6 6 4 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 7 2 0 2 \times 1 0 ^ { - 2 }$ </td><td>PI 30%</td></tr></table>

These supporting results show two patterns. First, pruning can act as structural regularization: the best pruned models for Burgers’ equation outperform the dense baselines at all three widths, and the effect is especially strong at width 512. Second, the best saliency rule is not fixed. PI-SAP is best for Burgers’ equation at widths 128 and 256, while NTK-SAP is best at width 512. For the convection equation, the preferred method also varies with both $\beta$ and network width: PI-SAP is best in several settings, while NTK-SAP is selected in others, such as $\beta = 1$ at width 512, $\beta = 5$ at width 128, and $\beta = 1 0$ at width 256.

As an additional high-β stress test at width 128, we also evaluate $\beta ~ = ~ 1 5$ and $\beta ~ = ~ 2 0$ . In both cases, the best pruned models still improve over the dense baselines: NTK-SAP at 70% sparsity reduces the mean relative $L ^ { 2 }$ error from $5 . 3 2 9 8 \times 1 0 ^ { - 2 }$ to $2 . 5 0 1 0 \times 1 0 ^ { - 2 }$ for $\beta = 1 5$ , and $\mathrm { N T K } { \cdot } \mathrm { S A P }$ at 50% sparsity reduces it from $1 . 3 3 0 2 \times 1 0 ^ { - 1 }$ to $7 . 9 8 8 6 \times 1 0 ^ { - 2 }$ for $\beta = 2 0$ . Overall, these results reinforce the main paper message: physics-aware pruning is useful, but sparse PINN selection should be evaluated across equation type, physical regime, sparsity, width, and training stiffness.

## VI. DISCUSSION

Across the four PDE families, the results support a consistent but nontrivial conclusion. Pruning is viable for PINN solvers, but the useful pruning criterion depends on what part of the physics-constrained training dynamics is under pressure. NTK-SAP is a strong output-dynamics baseline because it is tied to spectral preservation. PI-SAP becomes attractive when residual-sensitive structures matter, especially in highsparsity settings for the Gray–Scott equations and the complex Ginzburg–Landau equation. The results for Burgers’ equation and the convection equation show that this is not a universal dominance claim: the best method can shift with width, advection strength, and sparsity.

The most important observation for the Gray–Scott equations is the separation between solution error, residual error, and high-frequency error. PI-SAP lowers the mean PDE residual across the complete sparsity sweep for the Gray–Scott equations and reduces high-frequency error at 70%. However, the 70% diagnostic extensions show that optimizing residualside or kernel-conditioning criteria alone is not sufficient. A sparse PINN can achieve a lower residual loss on sampled collocation points while having worse relative solution error. This is consistent with the PINN-NTK view that different loss blocks can evolve at different rates [14].

For scientific machine learning, this suggests a practical evaluation standard. Sparse PINN solvers should be judged by at least three quantities: solution error, physics residual, and a task-aware spectral or morphological diagnostic. For reaction–diffusion systems, high-frequency spatial error is an effective diagnostic because pattern fidelity is lost first at the interfaces. For advection-dominated systems, stability across seeds and aggressive sparsity may be more informative. This multi-metric view is also aligned with the workshop goal of efficient and trustworthy AI for scientific applications: efficiency without physical fidelity is not enough, and residual reduction without solution fidelity can be misleading.

## VII. LIMITATIONS AND REPRODUCIBILITY

The experiments should be interpreted as evidence for a pruning principle, not as a final sparse-kernel acceleration result. The implementation enforces masks on dense JAX arrays, so the reported wall-clock times measure training stability and overhead under fixed software infrastructure rather than hardware-level sparse speedup. This is why the runtimes for the Gray–Scott equations and the complex Ginzburg–Landau equation remain nearly constant across sparsity levels. Actual acceleration would require sparse matrix kernels, structured sparsity, or compiler support that exploits the mask pattern during training and inference. For the present study, this design choice is intentional: it isolates the numerical effect of the pruning criterion from the engineering effect of a sparse backend.

The second limitation is statistical. The studies on Burgers’ equation and the convection equation are averaged over five seeds, but the results for the Gray–Scott equations and the complex Ginzburg–Landau equation are treated primarily as completed benchmark runs rather than large multi-seed sweeps. The Gray–Scott equations are expensive because each run uses 10 causal windows and long training budgets, and the purpose of the study is to compare pruning behavior under a fixed, reproducible solver setup. Future work should repeat the high-sparsity benchmark settings for the Gray–Scott equations and the complex Ginzburg–Landau equation across seeds, since the most interesting differences occur near the accuracydegradation threshold where initialization effects may matter.

The third limitation concerns the PINN-block diagnostics. The small-batch kernels $K _ { \mathrm { s o l } }$ and $K _ { \mathrm { r e s } }$ are practical approximations, not full training-set NTKs. This is necessary because explicitly constructing full NTK blocks for every collocation point and every residual component would be computationally prohibitive. The diagnostic coefficients were not extensively tuned, so these runs should be read as a first stress test of the objective design rather than a final optimized method. However, the diagnostic result is still useful: it shows that improving the residual-block diagnostic can lower sampled residual loss while worsening solution error. The next pruning objective should therefore condition both blocks jointly, potentially including cross terms $K _ { u r }$ and $K _ { r u }$ , rather than optimizing either output or residual dynamics in isolation.

## VIII. CONCLUSION

This paper studies physics-informed foresight pruning for sparse PINN solvers across nonlinear PDEs, with the Gray– Scott equations as the most complete benchmark and the complex Ginzburg–Landau equation, Burgers’ equation, and the convection equation as cross-PDE validation. The results show that sparse PirateNet solvers can retain useful accuracy under substantial pruning, and that residual-aware PI-SAP improves physics residuals for the Gray–Scott equations across all tested sparsities while providing the largest solution benefits in high-sparsity regimes. Cross-PDE experiments confirm that the advantage is not uniform: NTK-SAP remains competitive and sometimes superior, particularly at intermediate sparsity or in specific width/parameter settings. The NTK-block diagnostics further show why future pruning objectives should balance solution-side and residual-side dynamics instead of optimizing either one in isolation. The resulting direction is a conditioning- and physics-aware pruning framework for scientific ML models that reduces parameter count while remaining interpretable through NTK diagnostics and validated by PDEspecific fidelity metrics.

## REFERENCES

[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019.

[2] A. Jacot, F. Gabriel, and C. Hongler, “Neural tangent kernel: Convergence and generalization in neural networks,” in Advances in Neural Information Processing Systems, 2018.

[3] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. A. Hamprecht, Y. Bengio, and A. Courville, “On the spectral bias of neural networks,” in Proceedings of the 36th International Conference on Machine Learning, 2019.

[4] S. Wang, Y. Teng, and P. Perdikaris, “Understanding and mitigating gradient flow pathologies in physics-informed neural networks,” SIAM Journal on Scientific Computing, vol. 43, no. 5, pp. A3055–A3081, 2021.

[5] J. Kim, K. Lee, D. Lee, S. Y. Jhin, and N. Park, “DPM: A novel training method for physics-informed neural networks in extrapolation,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 35, no. 9, pp. 8146–8154, 2021.

[6] N. Doumeche, G. Biau, and C. Boyer, “On the convergence of PINNs,” arXiv:2305.01240v2, 2026.

[7] S. Wang, B. Li, Y. Chen, and P. Perdikaris, “PirateNets: Physics-informed deep learning with residual adaptive networks,” arXiv:2402.00326, 2024.

[8] W. Cho, K. Lee, D. Rim, and N. Park, “Hypernetwork-based metalearning for low-rank physics-informed neural networks,” in Advances in Neural Information Processing Systems, vol. 36, pp. 11219–11231, 2023.

[9] W. Cho, M. Jo, H. Lim, K. Lee, D. Lee, S. Hong, and N. Park, “Parameterized physics-informed neural networks for parameterized PDEs,” in Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

[10] N. Lee, T. Ajanthan, and P. H. S. Torr, “SNIP: Single-shot network pruning based on connection sensitivity,” in International Conference on Learning Representations, 2019.

[11] C. Wang, G. Zhang, and R. Grosse, “Picking winning tickets before training by preserving gradient flow,” in International Conference on Learning Representations, 2020.

[12] Y. Wang, D. Li, and R. Sun, “NTK-SAP: Improving neural network pruning by aligning training dynamics,” in International Conference on Learning Representations, 2023.

[13] L. Gu, Y. Du, Y. Zhang, D. Xie, S. Pu, R. C. Qiu, and Z. Liao, “Lossless compression of deep neural networks: A high-dimensional neural tangent kernel approach,” arXiv:2403.00258, 2024.

[14] S. Wang, X. Yu, and P. Perdikaris, “When and why PINNs fail to train: A neural tangent kernel perspective,” Journal of Computational Physics, vol. 449, 110768, 2022.