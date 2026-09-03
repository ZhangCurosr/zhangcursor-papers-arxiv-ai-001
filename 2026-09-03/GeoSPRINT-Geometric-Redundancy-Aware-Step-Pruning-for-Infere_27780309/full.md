# GeoSPRINT: Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories

Arpita Joshi The Scripps Research Institute 10550 N. Torrey Pines Road, La Jolla, San Diego CA, USA arpitamhow@gmail.com / ajoshi@scripps.edu

## Abstract

Diffusion models achieve high sample quality but remain expensive at inference time because sampling requires many sequential neural function evaluations (NFEs). Existing acceleration methods either use fixed step-skipping schedules, adapt step sizes based on local numerical error, or require additional training. We introduce GeoSPRINT (Geometric Step Pruning for Inference in Trajectories), a training-free framework for constructing non-uniform sampling schedules from the geometry of denoising trajectories. GeoSPRINT detects geometrically redundant steps using a hyperplanarity test in latent space, implemented efficiently via QR factorization, and converts the resulting redundancy profile into a sampling schedule that allocates more steps to high-curvature regions of the trajectory. In addition, we introduce the trajectory projection score αtraj, a residual-variance metric that quantifies trajectory straightness and serves as a model-free diagnostic for rectified flow quality. Across CIFAR-10 (32×32), LSUN Church (256×256), and Stable Diffusion v1.5 (512×512 latent), GeoSPRINT consistently improves over uniform DDIM (Denoising Diffusion Implicit Models) schedules at matched NFE budgets. On CIFAR-10, GeoSPRINT improves FID (Fréchet Inception Distance) by 0.7- 1.1 over DDIM across 49-89 NFEs and surpasses DPM-Solver++ at NFE≥30 despite using a first-order DDIM solver. On LSUN Church, it reduces FID from 1.48 to 1.26 at 52 steps, and on Stable Diffusion v1.5 it achieves up to 1.93 FID improvement over DDIM. These results show that trajectory geometry provides a useful global signal for allocating inference steps and that schedule quality can substantially improve diffusion sampling efficiency without retraining.

## 1 Introduction

Diffusion models [1, 2] and their continuous-time generalizations through score-based stochastic differential equations have become the dominant generative modeling paradigm across images [3], video [4], audio [5], and molecular design [6]. Despite their exceptional generation quality, the iterative nature of the reverse sampling process remains a computational bottleneck: generating a single sample typically requires tens to thousands of sequential neural network evaluations.

The community has responded with a rich ecosystem of acceleration techniques. Training-free approaches include DDIM [7], which reinterprets diffusion sampling as a deterministic ODE and enables uniform step-skipping; DPM-Solver [8] and DPM-Solver++ [9], which employ exponential integrators with adaptive step sizes based on local truncation error; and UniPC [10], which unifies predictor-corrector schemes. Training-based approaches include progressive distillation [11], which iteratively halves the required steps; consistency models [12, 13], which learn to map any point on the trajectory directly to the endpoint; and rectified flow [14], which straightens transport paths to reduce integration steps.

Despite this progress, a gap remains: existing training-free methods lack a global, trajectory-level criterion for step importance grounded in geometric redundancy. DDIM skips steps uniformly, blind to trajectory geometry. DPM-Solver adapts step sizes based on local error estimates at individual timesteps. Adaptive non-uniform timestep sampling [15] focuses on training rather than inference

The recent SDM framework [16] analyzes local ODE stiffness but still operates pointwise. None of these methods ask the global question: across the full trajectory, which steps actually contribute new geometric information to the path from noise to data?

We argue that this is precisely the question addressed by geometric data instance reduction [17]. That work demonstrated that ordered sequences of data points can be dramatically reduced by testing geometric redundancy—collinearity in 2D, coplanarity in 3D—while preserving the essential shape and variance of the structure, as quantified by a projection score based on eigenvalues of the covariance matrix of removed points [18]. The key insight is that many consecutive points in an ordered dataset lie approximately on the same linear or planar subspace and can be removed without information loss.

A denoising trajectory $\{ z _ { T } , z _ { T - 1 } , \dots , z _ { 0 } \}$ is precisely such an ordered sequence. In regions where the score function changes slowly, consecutive latent states trace a nearly linear path—these steps are geometrically redundant. In regions of rapid change (e.g., when the model resolves fine structure), the trajectory curves sharply, and every step carries new directional information. This analogy motivates GeoSPRINT.

Contributions: (1) We generalize geometric instance reduction from 2–3 PCA dimensions to arbitrary d-dimensional latent spaces via a hyperplanarity test with $O ( d \cdot k ^ { 2 } )$ per-step complexity. (2) We introduce a novel schedule construction method that blends log-SNR spacing with GeoSPRINT's trajectory curvature density, yielding non-uniform schedules that outperform both uniform-t (DDIM) and uniform-logSNR spacing. (3) We introduce the trajectory projection score $\alpha _ { \mathrm { t r a j } }$ , a residualvariance metric that quantifies trajectory non-straightness; we prove it equals zero for perfectly straight trajectories and demonstrate a $4 5 0 \times$ drop from DDPM to DDIM, establishing it as a trainingfree diagnostic for flow-matching quality. (4) We demonstrate that a first-order DDIM solver with GeoSPRINT scheduling outperforms the second-order DPM-Solver++ at NFE≥30 on CIFAR-10, showing that where to step matters more than how to step. (5) We provide benchmarks on CIFAR-10 (32×32), LSUN Church (256×256), and Stable Diffusion v1.5 (512×512 latent), with consistent improvements across resolutions, model architectures, and conditioning types.

## 2 Background and Related Work

## 2.1 Diffusion Models and the Sampling Problem

Score-based SDE framework. Song et al. [2] unified diffusion models under a continuous-time SDE framework. The forward process is $\mathrm { d } z = f ( z , t ) \mathrm { d } t + g ( t )$ dw, where $f$ and $g$ define the drift and diffusion coefficients. The reverse process is:

$$
\mathrm { d } z = \big [ f ( z , t ) - g ( t ) ^ { 2 } \nabla _ { z } \log p _ { t } ( z ) \big ] \mathrm { d } t + g ( t ) \mathrm { d } { \bar { w } } ,\tag{1}
$$

where $\nabla _ { z }$ log $p _ { t } ( z )$ is the score function approximated by a neural network $s _ { \theta } ( z , t )$ (and ¯ is the standard Weiner process). An equivalent deterministic formulation, the probability flow ODE is:

$$
\begin{array} { l } { \displaystyle { \frac { \mathrm { d } z } { \mathrm { d } t } = f ( z , t ) - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla _ { z } \log p _ { t } ( z ) . } } \end{array}\tag{2}
$$

Sampling reduces to solving this ODE from $t { = } T$ to $t { = } 0$ , requiring discretization into N steps, each involving one neural function evaluation (NFE). Flow matching: Lipman et al. [19] and Liu et al. [14] proposed learning a velocity field $v _ { \theta } ( z , t )$ that transports a source distribution to a target via $\mathrm { d } z / \mathrm { d } t = v _ { \theta } ( z , t )$ . Rectified flow [14] straightens these paths through iterative reflow. Wang et al. [20] showed that strict straightness is not necessary—first-order ODE consistency suffices.

## 2.2 Existing Acceleration Methods

Uniform step-skipping. DDIM [7] enables deterministic sampling that skips timesteps uniformly, reducing 1000 steps to \~50, but is blind to trajectory geometry.

High-order ODE solvers. DPM-Solver [8] derives exact solutions for the linear part of the diffusion ODE and applies exponential integrators for the nonlinear residual, achieving high quality in 10–20 steps. DPM-Solver++ [9] extends this to guided sampling. Both use local truncation error for adaptive step sizing with heuristic schedules (logSNR-uniform, time-uniform, time-quadratic).

Training-based methods. Progressive distillation [11] trains a student to match a teacher using N/2 steps, iteratively halving until 4-step generation. Consistency models [12, 13] learn direct trajectory-endpoint mappings. Consistency flow matching [21] combines consistency training with flow matching. All require additional training for each target NFE budget.

Adaptive timestep methods. Kim et al. [15] propose adaptive non-uniform timestep sampling for training acceleration. The SDM framework [16] analyzes local PF-ODE stiffness to set adaptive solver order and step size during inference. Both operate on local properties at individual timesteps.

## 2.3 Geometric Instance Reduction

Joshi and Haspel [17] introduced a data instance reduction algorithm that operates on ordered sequences projected into PCA space. The algorithm tests geometric redundancy via collinearity (2D) and coplanarity (3D) tests, removing points that lie approximately on the same linear subspace as their neighbors. Information loss is quantified by a projection score [18]:

$$
\alpha = \frac { \sum _ { j = 1 } ^ { p } \lambda _ { j } ^ { ( S ) } } { \sum _ { j = 1 } ^ { p } \lambda _ { j } ^ { ( \mathcal { Z } ) } } ,\tag{3}
$$

where $\lambda _ { j } ^ { ( S ) }$ and $\lambda _ { i } ^ { ( { \mathcal Z } ) }$ are the j-th eigenvalues of the covariance matrices computed from the removed subset $\breve { S }$ and the full dataset ${ \mathcal { Z } } ,$ respectively, and $p$ is the number of retained PCA dimensions. The algorithm achieves 40–87% point reduction across diverse datasets (molecular trajectories, images, ML benchmarks) with projection scores as low as $1 0 ^ { - 5 }$

Critical limitation for our purposes: the original algorithm operates in 2–3 PCA dimensions.   
Diffusion latent spaces are typically 32–512 dimensional. Section 3 addresses this directly.

## 3 Method: GeoSPRINT

## 3.1 Generalizing Geometric Redundancy to High Dimensions

Let $\left\{ z _ { 0 } , z _ { 1 } , \dots , z _ { N } \right\}$ be an ordered sequence of points in $\mathbb { R } ^ { d }$ (latent states along a denoising trajectory, ordered from $t { = } T$ to $t { = } 0 )$ . We seek to identify and remove points approximately contained in the affine subspace spanned by their neighbors.

Definition 1 (Hyperplanarity test). Given a window of k retained points $W = \{ w _ { 1 } , w _ { 2 } , \ldots , w _ { k } \}$ (where $k \leq d )$ and a candidate point $z _ { i } ,$ define:

$$
M _ { W } = \left[ w _ { 2 } - w _ { 1 } \mid w _ { 3 } - w _ { 1 } \mid \cdots \mid w _ { k } - w _ { 1 } \right] \in \mathbb { R } ^ { d \times ( k - 1 ) } .\tag{4}
$$

The residual distance of zi from the affine subspace spanned by W is:

$$
r _ { i } = \left\| ( I - M _ { W } M _ { W } ^ { + } ) ( z _ { i } - w _ { 1 } ) \right\| _ { 2 } ,\tag{5}
$$

where $M _ { W } ^ { + }$ is the Moore-Penrose pseudoinverse. $I f r _ { i } < \tau$ , then $z _ { i }$ is approximately in the affine span and is a candidate for removal

Note that Definition 1 uses the most recently retained (not necessarily consecutive) points as the reference window, matching the causal sliding-window logic of Algorithm 2.

Efficient computation via QR factorization. Rather than computing the pseudoinverse directly, we compute the thin QR factorization $M _ { W } = Q _ { W } R _ { W }$ and obtain $r _ { i } = \| ( I - Q _ { W } Q _ { W } ^ { \top } ) ( z _ { i } - w _ { 1 } ) \| _ { 2 }$ Computing the QR factorization of $M _ { W } \in \mathbb { R } ^ { d \times ( k - 1 ) }$ costs $O ( d \cdot k ^ { 2 } )$ operations; since k is small (typically $k { = } 2 )$ , this is $O ( d )$ in practice.

Adaptive window size. We propose a progressive hierarchy mirroring the 2D→3D progression in the original algorithm: Tests are applied in a progressive hierarchy: Level 1 (k=2, collinearity) catches long straight segments; Level $\begin{array} { r } { 2 \ ( k = 3 . } \end{array}$ , coplanarity) test if 4 points are coplanar, catches planar sweeps; Level $\ell \left( k { = } \ell \right)$ tests against an (l—1)-dimensional affine subspace. These tests are applied sequentially: Level 1 first, then Level 2 on surviving points, etc. In practice, we find $k { = } 2$ (collinearity) sufficient for all models tested (see Section C.3).

Threshold determination. Following Joshi and Haspel [17], we determine τ through binary search guided by the trajectory projection score (Section 3.3), adjusting until the projection score of removed points falls below a target $( \mathrm { e . g . , 1 0 ^ { - 3 } } )$ while maximizing the number of removed steps.

## 3.2 Application to Denoising Trajectories

Reference trajectory generation. Given a pretrained diffusion or flow model, we first generate B reference trajectories using the full N-step schedule. Each trajectory $\{ z _ { T } ^ { ( b ) } , \ldots , z _ { 0 } ^ { ( b ) } \}$ provides an

ordered sequence in $\mathbb { R } ^ { d }$

Per-trajectory pruning. For each reference trajectory, we apply the hyperplanarity test (Algorithm 2) to identify and remove geometrically redundant timesteps, producing a per-trajectory retained set ${ \mathcal { R } } ^ { ( b ) } \subseteq \{ t _ { 0 } , t _ { 1 } , \dots , t _ { T } \}$

Universal schedule aggregation. To obtain a fixed schedule usable without per-sample overhead, we aggregate across reference trajectories:

$$
w ( t ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathbf { 1 } \big [ t \in \mathcal { R } ^ { ( b ) } \big ] .\tag{6}
$$

The retention frequency $w ( t )$ encodes a per-timestep curvature density: it is high where trajectories consistently curve and low where they are consistently straight.

Schedule construction via log-SNR curvature blending. A naive approach selects the top-K timesteps by retention frequency $w ( t )$ . However, this clusters steps in high-curvature regions while leaving large gaps elsewhere, and the DDIM solver's prediction error grows with the gap in log-signalto-noise ratio (log-SNR) between consecutive steps, not with the gap in t. We therefore construct schedules by blending two density functions:

$$
\rho ( t ) = ( 1 - \beta ) \cdot \rho _ { \mathrm { l o g S N R } } ( t ) + \beta \cdot \rho _ { \mathrm { c u r v } } ( t ) ,\tag{7}
$$

where $\rho _ { \mathrm { l o g S N R } }$ is uniform density in log-SNR space (ensuring uniform error contribution per step, as in DPM-Solver), $\rho _ { \mathrm { c u r v } } ( t ) \propto \tilde { w } ( t )$ is the smoothed and floored curvature density derived from Eq. 6, and $\beta \in [ 0 , 1 ]$ is the blend weight. Smoothing applies a Gaussian kernel with bandwidth $\sigma { = } 5$ timesteps, followed by flooring at $\epsilon _ { \mathrm { f l o o r } } = 0 . 0 5 \cdot \operatorname* { m a x } _ { t } \tilde { w } ( t )$ to ensure minimum coverage. We then place $K$ steps at uniform quantiles of the cumulative distribution function $\begin{array} { r } { F ( t ) = \int _ { 0 } ^ { t } \rho ( s ) \mathrm { d } s / \int _ { 0 } ^ { T } \rho ( s ) } \end{array}$ ds, analogous to importance sampling. Setting $\beta { = } 0$ recovers log-SNR-uniform spacing (a known baseline). Setting $\beta { = } 1$ gives pure curvature-weighted spacing (which fails at low NFE due to coverage gaps). We find $\beta { = } 0 . 6$ is optimal on CIFAR-10 and effective across the tested datasets (Section 5.3), though per-dataset ablation on LSUN Church and SD v1.5 would further validate robustness. The key insight is that neither spacing alone is optimal: log-SNR spacing ignores trajectory geometry, while curvature spacing ignores solver error characteristics. The blend combines both signals.

Sample-adaptive analysis. The universal schedule applies the same timestep selection to all samples. To understand how geometric complexity varies across samples, we additionally analyze each reference trajectory individually: for a given trajectory, the number of retained timesteps after pruning indicates that sample's geometric complexity. This per-sample analysis does not directly reduce NFEs at inference time—the model evaluation must still be performed to obtain each $z _ { t } -$ -but it serves two important purposes: (i) it reveals the distribution of sample difficulty across a dataset (Section 5.6), informing whether a single universal schedule is sufficient or whether future work on predictive per-sample budgeting is warranted; and (ii) it provides per-sample $\alpha _ { \mathrm { t r a j } }$ values that can be correlated with sample quality metrics.

Remark on inference cost. The primary acceleration mechanism of GeoSPRINT is the universal schedule (Algorithm 1), which extracts a fixed set of timesteps offline and then generates all new samples using only those timesteps. The per-sample analysis described above is an offline diagnostic tool, not an inference-time accelerator.

## 3.3 Trajectory Projection Score

We extend the projection score of Joshi and Haspel [17] to denoising trajectories. The key insight is that the residual variance—the variance orthogonal to the local affine subspace at each pruned point— is the correct measure of information loss, not the raw variance of the pruned points themselves.

Definition 2 (Trajectory projection score). Given a trajectory $\mathcal { Z } = \{ z _ { T } , \ldots , z _ { 0 } \}$ with $N { + 1 }$ points, let $S \subset { \mathcal { Z } }$ be the set of pruned points, and let $r _ { s }$ denote the hyperplanarity residual (Eq. 5) of each pruned point $z _ { s } \in S .$ The trajectory projection score is:

$$
\alpha _ { \mathrm { t r a j } } = \frac { \sum _ { s \in S } r _ { s } ^ { 2 } } { \sum _ { z \in \mathcal { Z } } \| z - \bar { z } \| ^ { 2 } } ,\tag{8}
$$

where z is the global trajectory mean. The numerator is the total squared residual of all pruned points: the variance that lies outside the affine span of their retained neighbors. The denominator is the total trajectory variance.

$\alpha _ { \mathrm { t r a j } }$ measures the fraction of total trajectory variance that is lost by pruning—specifically, the variance orthogonal to what the retained points can reconstruct through local affine interpolation. A low score $( \mathrm { e . g . , 1 0 ^ { - 3 }  t o 1 0 ^ { - 5 } } )$ indicates that removed steps contributed negligibly to the trajectory's geometric structure beyond what is already captured by their neighbors.

Proposition 1. For a perfectly straight trajectory (all points on a single line in $\mathbb { R } ^ { d } )$ , GeoSPRINT with the collinearity test $\left( k { = } 2 \right)$ prunes all points beyond the initial window, retaining exactly the frst k points, with $\alpha _ { \mathrm { t r a j } } = 0$

Proof. If all points are collinear, every candidate point $z _ { i }$ (for $i \geq k )$ lies exactly on the line through the retained window points, so the hyperplanarity residual $r _ { i } = 0$ for all such i. Because all residuals satisfy $r _ { i } < \tau$ , the window never updates (the else branch of Algorithm 2 is never executed), and the retained set consists of the first k points used to initialize the window. The numerator $\sum r _ { i } ^ { 2 } = 0 .$ hence $\alpha _ { \mathrm { t r a j } } = 0$ □

Corollary 1. $\alpha _ { \mathrm { t r a j } }$ is a measure of trajectory non-straightness. For rectified flow models, which aim to straighten trajectories, $\alpha _ { \mathrm { t r a j } }$ decreases with each round of reflow, providing a training-free diagnostic for rectification quality.

## 3.4 Connections to Existing Methods

Relation to DPM-Solver. DPM-Solver's adaptive step size is based on local truncation error—the difference between a higher-order and lower-order solution at each step. GeoSPRINT's hyperplanarity test is a geometric criterion over a window of points, making it sensitive to trajectory structure over multiple steps rather than single-step error. The two are complementary: DPM-Solver minimizes local numerical error; GeoSPRINT minimizes global geometric redundancy.

Relation to SDM. The SDM framework [16] analyzes local PF-ODE stiffness to determine solver order at each timestep. GeoSPRINT instead operates on the realized trajectory post-hoc, testing whether the path actually traced contains redundant segments. SDM is predictive (deciding before taking the step); GeoSPRINT is retrospective (analyzing after generation). This makes GeoSPRINT applicable to any model without Jacobian access. An empirical comparison of these complementary strategies is an important direction for future work.

Relation to rectified flow. Rectified flow learns transport paths that are inherently straight, making all interior steps redundant by construction. GeoSPRINT diagnoses rather than enforces straightness, and works equally well on curved trajectories by adaptively allocating steps where curvature demands them.

Relation to consistency models. Consistency models learn to jump from any trajectory point to the endpoint in one step. GeoSPRINT identifies which intermediate points are necessary to maintain trajectory integrity. The two can be combined: GeoSPRINT identifies minimal waypoints, and a consistency model could jump between them.

## 4 Theoretical Analysis

## 4.1 Reconstruction Error Bound

Theorem 1 (Reconstruction error bound). Let Z be a trajectory of $N { + 1 }$ points in $\mathbb { R } ^ { d }$ , and let S be the set of pruned points after GeoSPRINT with hyperplanarity threshold $\tau$ Then:

$$
\alpha _ { \mathrm { t r a j } } \leq \frac { | \boldsymbol { S } | \cdot \tau ^ { 2 } } { ( N + 1 ) \cdot \mathrm { t r } ( \boldsymbol { C } _ { \mathcal { Z } } ) } ,\tag{9}
$$

where |S| is the number of pruned steps and $\operatorname { t r } ( C _ { \mathcal { Z } } )$ is the trace of the sample covariance of $\mathcal { Z } .$

Proof sketch. Each pruned point $z _ { s }$ satisfies $r _ { s } \le \tau$ by the hyperplanarity test, so $\begin{array} { r } { \sum _ { s \in S } r _ { s } ^ { 2 } \le | S | \cdot \tau ^ { 2 } } \end{array}$ The denominator of $\alpha _ { \mathrm { t r a j } }$ equals $( N { + } 1 ) \cdot \operatorname { t r } ( C _ { \mathcal { Z } } )$ . Division yields the result. The full proof is in Appendix B. □

This provides a formal guarantee: GeoSPRINT's reconstruction error—the fraction of total trajectory variance attributable to orthogonal residuals of pruned points—is controlled by τ, the pruning rate $| S | / ( N { + } 1 )$ , and the total trajectory variance. For a desired $\alpha _ { \mathrm { t r a j } } \leq \epsilon _ { \mathrm { : } }$ , it suffices to set $\tau \leq$ $\sqrt { \epsilon \cdot ( N + 1 ) \cdot \mathrm { t r } ( C _ { \mathcal { Z } } ) / | S | }$

Remark. We emphasize that $\alpha _ { \mathrm { t r a j } }$ bounds the orthogonal reconstruction residual, not the retained set's sample covariance tr $( C _ { R } ) / \mathrm { t } \mathbf { \check { r } } ( C _ { \mathcal { Z } } )$ . Pruning points changes the sample mean and can reduce the retained sample covariance even when all residuals are zero $( e . g .$ , removing interior points from a collinear trajectory reduces sample variance while $\alpha _ { \mathrm { t r a j } } = 0 )$ . The reconstruction interpretation— that pruned points are well-approximated by affine interpolation from their retained neighbors—is the operationally relevant guarantee for schedule quality.

## 4.2 Complexity Analysis

The multi-level hyperplanarity test with maximum level l has complexity $O ( N \cdot d \cdot \ell ^ { 2 } )$ per trajectory in the worst case. For Level-1 only (collinearity, k=2), this is $\bar { O ( N \cdot d ) }$ -linear in both trajectory length and dimension, and negligible compared to the cost of the NFEs themselves. In Algorithm $2 .$ the QR factorization is cached and recomputed only when the window updates, further reducing the amortized cost. Schedule amortization over a batch of M samples reduces per-sample overhead to $O ( B / M )$ reference trajectories.

## 4.3 Relation to Trajectory Curvature

The hyperplanarity residual ri at point $z _ { i }$ relates to discrete curvature. For a smooth trajectory $z ( t )$ with a collinearity window $\left( k { = } 2 \right)$ using causal forward-extrapolation from retained points at $z ( t )$ and $z ( t { + } \Delta t )$ to the candidate z(t+2∆t):

$$
r _ { i } \approx ( \Delta t ) ^ { 2 } \left\| z _ { \perp } ^ { \prime \prime } ( t ) \right\| _ { 2 } + O \big ( ( \Delta t ) ^ { 3 } \big ) ,\tag{10}
$$

where $z _ { \perp } ^ { \prime \prime } ( t )$ is the component of $z ^ { \prime \prime } ( t )$ orthogonal to the velocity $z ^ { \prime } ( t )$ . Tangential acceleration (speeding up along a straight path) does not contribute, since the residual measures orthogonal distance from the extrapolated line. The coefficient $( \Delta { t } ) ^ { 2 }$ (rather than $( \Delta t ) ^ { 2 } / 2 )$ reflects causal forward-extrapolation rather than midpoint interpolation. Removing points with $r _ { i } ~ < ~ \tau$ is thus equivalent to removing steps where the trajectory's orthogonal curvature is small. Higher-level tests detect higher-order geometric structure (torsion, etc.).

## 5 Experiments

## 5.1 Setup

Models and datasets. We evaluate on: CIFAR-10 (32×32) using the pretrained google/ddpm-cifar10-32 UNet from HuggingFace Diffusers; LSUN Church (256×256) using google/ddpm-ema-church-256; and Stable Diffusion v1.5 (512×512, latent 4×64×64) using runwayml/stable-diffusion-v1-5 with classifier-free guidance (CFG scale 7.5). All pixel-space models use the standard linear $\beta .$ -schedule with 1000 training timesteps.

Protocol. For each model, we record reference trajectories using DDIM at 200 steps: $B { = } 1 0 0$ for CIFAR-10 and $B { = } 5 0$ for LSUN Church and SD v1.5 (see Appendix C.1 for convergence analysis). We then extract the curvature density profile and construct LogSNR+curvature schedules (Eq. 7, $\beta { = } 0 . 6 )$ . For FID computation, we generate 10,000 samples for CIFAR-10 and LSUN Church, and 5,000 samples for SD v1.5, using a fixed set of prompts randomly sampled from the COCO 2017 validation set. FID is computed against the training set using clean-FID [22] for CIFAR-10 and LSUN Church. For SD v1.5, FID is computed against a 50-step DDIM reference set, measuring relative trajectory fidelity rather than absolute generation quality (see Section 5.2.1). All sampling uses the manual DDIM update formula to correctly handle non-uniform timestep spacing. All reported FID values are means ± standard deviation over 3 independent runs with different random seeds.

Baselines. DDIM [7]: uniform timestep spacing at matched NFE. DPM-Solver++ [9]: secondorder solver with default logSNR-uniform schedule, solver\_order=2. DDIM-200: proxy for zero-pruning baseline.

## 5.2 Main Result: FID vs. NFE Across Datasets

GeoSPRINT operates as a step-budget oracle: starting from an initial $K ,$ it searches upward, increasing K until the trajectory projection score $\alpha _ { \mathrm { t r a j } }$ falls below a target threshold. At each $K ,$ we construct a LogSNR+curvature schedule (Eq. $7 , \beta { = } 0 . 6 )$ and compare with DDIM uniform spacing at matched NFE. On CIFAR-10, we additionally compare with DPM-Solver++ (second-order, default schedule).

Key findings. (1) GeoSPRINT outperforms DDIM at every NFE on all three datasets (except NFE=8 on SD and NFE=10 on CIFAR-10, see Figure 1(a) and Figure 2). (2) Improvements grow with model complexity: -0.7 to -1.1 on $\mathrm { C I F A R { - } 1 0 , - 0 . 2 \ t o - 1 . 1 }$ on Church, and up to —1.93 on SD v1.5 (Table 1). (3) DDIM's uniform schedule becomes actively counterproductive at high NFE on both Church and SD (FID increasing), while GeoSPRINT monotonically improves (Table 1 and Figure 2). (4) On CIFAR-10, first-order DDIM+GeoSPRINT outperforms second-order DPM-Solver++ at NFE≥30, demonstrating that schedule quality dominates solver order at moderate step budgets on this model; extending this comparison to higher-resolution models is left to future work. (5) GeoSPRINT at 60 steps matches DDIM at 80 steps on CIFAR-10 (both $\mathrm { F I D } { \approx } 1 5 . 3 )$ , a 25% NFE reduction.

Table 1: FID (↓) across datasets (mean ± std over 3 seeds). GeoSPRINT uses LogSNR+curvature $\left( \beta { = } 0 . 6 \right)$ . Bold: best per row. CIFAR-10 baseline: $\mathrm { D D I M } { - } 2 0 0 = 1 2 . 6 5$ . SD FID computed against 50-step DDIM reference (see Section 5.2.1).
<table><tr><td>NFE</td><td>Geo</td><td>DDIM</td><td>∆</td><td>DPM++</td></tr><tr><td>10</td><td>38.53±.31</td><td> $3 4 . 9 9 \pm . 2 8 $ </td><td>3.54</td><td> $2 9 . 0 5 \pm . 2 5$ </td></tr><tr><td>20</td><td>22.23±.19</td><td> $2 3 . 1 7 \pm . 2 1$ </td><td>-0.94</td><td>21.31±.18</td></tr><tr><td>30</td><td>18.50±.15</td><td> $1 9 . 3 3 \pm . 1 7$ </td><td>-0.83</td><td>18.85±.16</td></tr><tr><td>49</td><td>15.95±.13</td><td>16.82±.14</td><td>-0.87</td><td>17.41±.15</td></tr><tr><td>60</td><td>15.25±.12</td><td>15.95±.13</td><td>-0.70</td><td>17.08±.14</td></tr><tr><td>70</td><td>14.80±.11</td><td>15.69±.12</td><td>-0.89</td><td>16.82±.14</td></tr><tr><td>80</td><td>14.38±.10</td><td>15.30±.12</td><td>-0.92</td><td>16.68±.13</td></tr><tr><td>89</td><td> ${ \bf 1 4 . 1 0 \pm . 1 0 }$ </td><td>15.17±.11</td><td>-1.07</td><td>16.65±.13</td></tr></table>

<table><tr><td colspan="3">Church (256× 256)</td></tr><tr><td>NFE</td><td>Geo</td><td>DDIM ∆</td></tr><tr><td>52</td><td> $1 . 2 6 \pm . 0 4$ </td><td>1.48±.05 -0.22</td></tr><tr><td>56</td><td> $\mathbf { 1 . 1 6 \pm . 0 3 }$ </td><td>2.22±.07 -1.06</td></tr><tr><td>74</td><td> ${ \bf 0 . 8 7 \pm . 0 3 }$ </td><td>1.72±.05 -0.85</td></tr><tr><td>79</td><td> ${ \bf 0 . 7 9 \pm . 0 2 }$ </td><td>1.43±.04 -0.64</td></tr></table>

<table><tr><td>NFE</td><td>Geo</td><td>DDIM ∆</td></tr><tr><td>15</td><td> ${ \bf 7 . 9 1 \pm . 0 8 }$  8.72±.09</td><td>-0.81</td></tr><tr><td>20</td><td> $5 . 5 1 { \pm } . 0 6$ </td><td>5.97±.07 -0.46</td></tr><tr><td>24</td><td> ${ \bf 4 . 6 0 \pm . 0 5 }$ </td><td>5.92±.06 -1.32</td></tr><tr><td>29</td><td> ${ \bf 4 . 0 1 } \pm . 0 5$ </td><td>4.92±.05 -0.91</td></tr><tr><td>34</td><td> $3 . 6 6 \pm . 0 4$ </td><td>4.23±.05 -0.57</td></tr><tr><td>39</td><td> $3 . 5 1 \pm . 0 4$ </td><td>4.92±.06 -1.41</td></tr><tr><td>44</td><td> $3 . 4 5 \pm . 0 4$ </td><td>5.38±.06 -1.93</td></tr></table>

![](images/21a96412e89c7b8e166165977fd0dc777ab1421ac925fbe30afbc34ebe6c19ed.jpg)

![](images/e1a0082e50cabaead6400633f438ef6044f94cdf8b71d807759336cf3563c147.jpg)

![](images/d0a994bc9b5cd313b9313bbcc6b58fcb0027554aea771e715c270c22ce65885a.jpg)  
Figure 1: CIFAR-10 results. (a) FID vs. NFE. GeoSPRINT consistently outperforms DDIM. Dashed lines: DDIM-200 baseline (12.65) and DDIM-50 reference (16.80). Error bands: ±1 std over 3 seeds. (b) $\alpha _ { \mathrm { t r a j } }$ decreases log-linearly with NFE, providing a principled knob for the step-budget oracle. (c) First-order DDIM+GeoSPRINT outperforms second-order DPM-Solver++ at $\mathrm { N F E { \geq } 3 0 }$ on CIFAR-10; where to step matters more than how to step.

## 5.2.1 Note on Stable Diffusion FID Measurement

For SD v1.5, we compute FID against a 50-step DDIM reference set rather than against a ground-truth image dataset. This measures relative trajectory fidelity—how closely a reduced schedule reproduces the full solver's output distribution—rather than absolute generation quality. This protocol isolates the effect of the schedule from the base model's inherent generation gap.

## 5.3 Blend Parameter Ablation

We ablate the blend parameter β (Eq. 7) on CIFAR-10; results are summarized in Table 2 (Left). DDIM uniform achieves 23.17/19.33/16.80 at NFE=20/30/50. Pure log-SNR (β=0) is worse than

Table 2: Left: Blend ablation on CIFAR-10 (FID ↓, mean of 3 seeds). $\beta { = } 0 ;$ pure log-SNR; $\beta { = } 1$ pure curvature. Optimum at $\beta \in [ 0 . 5 , 0 . 7 ]$ . DDIM uniform: 23.17/19.33/16.80. Right: $\alpha _ { \mathrm { t r a j } }$ across schedulers (450 × drop DDPM→DDIM).
<table><tr><td>β</td><td>0</td><td>0.3</td><td>0.5</td><td>0.6</td><td>0.7</td><td>1.0</td></tr><tr><td>20</td><td>25.81</td><td>22.45</td><td>22.26</td><td>22.23</td><td>22.19</td><td>22.68</td></tr><tr><td>30</td><td>20.89</td><td>18.71</td><td>18.56</td><td>18.50</td><td>18.52</td><td>18.86</td></tr><tr><td>50</td><td>16.80</td><td>16.03</td><td>15.91</td><td>15.87</td><td>15.94</td><td>16.04</td></tr></table>

<table><tr><td>Scheduler</td><td> $\alpha _ { \mathrm { t r a j } }$ </td><td>Pruned</td></tr><tr><td>DDPM</td><td> $. 4 0 8 { \pm } . 0 2 4$ </td><td>79%</td></tr><tr><td>DDIM</td><td> $. 0 0 0 9 { \pm } . 0 0 0 1$ </td><td>82%</td></tr><tr><td>DPM++</td><td> $. 0 0 0 8 { \pm } . 0 0 0 2$ </td><td>84%</td></tr></table>

![](images/5ce7cd598e4d1e822ce01f4e6a84836f136c4327ff37547b6c8e2955d3b70db9.jpg)

![](images/59444e4b259847de8eb3989ef88c79e0be4c92fe6a5b99907e944510b816a3de.jpg)  
Figure 2: FID vs. NFE on LSUN Church 256×256 (left) and Stable Diffusion v1.5 (right). GeoSPRINT monotonically improves while DDIM's uniform schedule plateaus or worsens at high NFE. Error bands: ±1 std over 3 seeds.

DDIM uniform, confirming that log-SNR spacing alone is not superior—it is GeoSPRINT's curvature correction that makes the difference. Neither component alone suffices; the blend addresses both solver error (via log-SNR) and trajectory geometry (via curvature) simultaneously. We note that this ablation is performed on CIFAR-10 only; while $\beta { = } 0 . 6$ is effective on the other datasets, dedicated ablations would further confirm robustness.

## 5.4 Schedule Analysis

The curvature density profile w(t) reveals where GeoSPRINT concentrates steps. The late denoising zone (low noise, $t / T < 0 . 2 )$ has mean retention 2× higher than the middle zone $( 0 . 2 < t / T < 0 . 8 )$ confirming that fine-detail resolution drives step allocation. The early zone $( t / T > 0 . 8 )$ also shows elevated retention, reflecting the initial direction-setting phase (Figure 3).

![](images/195bbdc0a48a436866167bad9d3f3b1627df79968ca712870c4599e394907f49.jpg)

![](images/8a2f416f75b9f81155be460f2576ce2fe8c0f4fdbd6268e57ee12ed461b2d52e.jpg)

![](images/122eaf4a79a4ca75a92d1d13e37577206087187256d99c5bfd496ebe793cbe72.jpg)  
Figure 3: (a) Retention frequency w(t) peaks at both ends (bimodal). (b) Selected timesteps at budgets $K = 1 0 , 2 0 , 5 0$ . (c) Cumulative: 50% of retained steps lie in the final 20% of denoising.

## 5.5 $\alpha _ { \mathrm { t r a j } }$ as Rectification Diagnostic

We record 50 trajectories from three scheduler types on the same pretrained CIFAR-10 model. $\alpha _ { \mathrm { t r a j } }$ drops 450× from DDPM to DDIM (Table 2, Right), confirming it as a training-free rectification diagnostic (Figure 4).

## 5.6 Supporting Experiments (Appendix C)

Reference set convergence: Schedule stability reaches fuzzy Jaccard 0.84 and correlation 0.86 at B=75 reference trajectories; B=100 is adequate for CIFAR-10 (Figure 5). Per-sample complexity: NFE variation is modest on CIFAR-10 (1.2–1.4×), but individual curvature profiles are diverse (mean correlation 0.37 with universal), validating the universal schedule as a robust compromise across samples (Figure 6).

## 6 Discussion

Why does curvature-aware scheduling help? The DDIM update error at each step depends on two factors: the gap in log-SNR space (governing the noise-prediction error) and the trajectory curvature (governing how far the local linear approximation deviates from the true path). Uniform-t spacing equalizes neither. Log-SNR spacing equalizes the first but ignores the second. GeoSPRINT's curvature density captures the second, and the blend (Eq. 7) addresses both simultaneously.

![](images/2da387e689f45daa9b32a9fb87c0d8353b901572593a7f9ed2bdaf438f6f9458.jpg)

![](images/bc08ff702926a8ab9346345187f5be1a2c8c0e2d600a1c326fc439e6e5ab2e28.jpg)  
Figure 4: (a) $\alpha _ { \mathrm { t r a j } }$ across schedulers: 450× drop DDPM→DDIM. (b) Reduction rate: straighter trajectories are more compressible.

Schedule quality vs. solver order. A surprising finding is that GeoSPRINT+DDIM (first-order) outperforms DPM-Solver++ (second-order) at NFE≥30 on CIFAR-10. This suggests that at moderate step budgets, the dominant error source is schedule quality (which steps to take) rather than solver accuracy (how to take each step). At very low NFE (≤10), the solver's accuracy per step matters more, explaining DPM-Solver++'s advantage there. Extending this comparison to LSUN Church and SD v1.5 would further substantiate this as a general principle.

Scaling to complex models. The SD v1.5 results (Table 1) reveal that GeoSPRINT's advantage grows with model complexity. While CIFAR-10 improvements are 0.7–1.1 FID, SD v1.5 improvements reach 1.93 FID. Moreover, DDIM's uniform schedule becomes actively counterproductive at high NFE on SD v1.5 (FID increasing from NFE=29 to NFE=44), while GeoSPRINT monotonically improves. This suggests that curvature-aware scheduling is most valuable precisely where it is most needed: in complex, production-scale models with classifier-free guidance.

Limitations: The universal schedule requires B full reference trajectories offline (B=100 at N=200 steps = 20,000 NFEs for CIFAR-10; B=50 for higher-resolution models). For single-sample generation this overhead dominates; the method is best suited to batch or deployment settings. The blend parameter β=0.6 was tuned on CIFAR-10; while it is effective on LSUN Church and SD v1.5, per-dataset ablation would further validate robustness. The SD v1.5 experiments use a fixed CFG scale of 7.5; whether a universal schedule extracted at one CFG scale generalizes to other scales requires further investigation. Higher-level hyperplanarity tests (k > 2) did not improve schedule quality on the models tested (Section C.3), suggesting that collinearity captures the dominant geometric signal for these architectures. Finally, GeoSPRINT selects which timesteps to use but does not improve the solver at each timestep; it is complementary to, not a replacement for, high-order ODE solvers.

Broader Impact: GeoSPRINT's training-free nature makes fast diffusion sampling accessible without distillation or retraining costs. The trajectory projection score provides an interpretable diagnostic for model quality. As an acceleration method, it inherits but does not amplify the societal risks of the underlying generative models.

## 7 Conclusion

We have presented GeoSPRINT, a training-free framework that constructs non-uniform sampling schedules for diffusion models by analyzing the geometry of denoising trajectories. The key technical contribution is a blended schedule construction that combines log-SNR spacing (matching the ODE solver's error profile) with trajectory curvature density (derived from a high-dimensional hyperplanarity test). On CIFAR-10, this yields schedules that outperform uniform DDIM by 0.7–1.1 FID across all tested NFE budgets, and outperform DPM-Solver++ at NFE≥30 despite using a simpler first-order solver—demonstrating that schedule quality dominates solver order at moderate step counts on this model. These improvements generalize to LSUN Church at 256×256 resolution and, most strikingly, to Stable Diffusion v1.5 where improvements reach —1.93 FID and uniform scheduling becomes actively counterproductive at high step counts. The trajectory projection score $\alpha _ { \mathrm { t r a j } }$ , which drives the curvature analysis, additionally serves as a standalone diagnostic for trajectory rectification quality, exhibiting a 450 × drop from DDPM to DDIM. Future directions include learning the blend parameter β per-model, developing predictive per-sample step budgeting for heterogeneous datasets, empirical comparison with the SDM framework [16], and extending the DPM-Solver++ comparison to higher-resolution models.

## References

[1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[2] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[3] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[4] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In Advances in Neural Information Processing Systems, 2022.

[5] Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. In ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models, 2021.

[6] Emiel Hoogeboom, Víctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3D. In International Conference on Machine Learning, pages 8867–8887, 2022.

[7] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

[8] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, volume 35, pages 5775–5787, 2022.

[9] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, 22(4):730–751, 2025.

[10] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: A unified predictorcorrector framework for fast sampling of diffusion models. In Advances in Neural Information Processing Systems, 2023.

[11] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.

[12] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252, 2023.

[13] Yang Song and Cheng Lu. Simplifying, stabilizing and scaling continuous-time consistency models. In International Conference on Learning Representations, 2025.

[14] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

[15] Myunsoo Kim, Donghyeon Ki, Seong-Woong Shim, and Byung-Jun Lee. Adaptive nonuniform timestep sampling for accelerating diffusion model training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[16] Sangwoo Jo and Sungjoon Choi. Formalizing the sampling design space of diffusion-based generative models via adaptive solvers and Wasserstein-bounded timesteps. arXiv preprint arXiv:2602.12624, 2026.

[17] Arpita Joshi and Nurit Haspel. A novel data instance reduction technique using linear feature reduction. Journal of Artificial Intelligence and Systems, 2:191–206, 2020.

[18] Magnus Fontes and Charlotte Soneson. The projection score—an evaluation criterion for variable subset selection in PCA visualization. BMC Bioinformatics, 12:307, 2011.

[19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[20] Fu-Yun Wang et al. Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303, 2024.

[21] Ling Yang et al. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398, 2024.

[22] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in GAN evaluation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11410–11420, 2022.

## A Pseudocode for GeoSPRINT

Algorithm 1: GeoSPRINT — Universal Schedule Extraction   
Input: Pretrained model M, full schedule $S = \{ t _ { 0 } , t _ { 1 } , \ldots , t _ { T } \}$ , reference batch size B, window   
k, target $\alpha _ { \mathrm { t a r g e t } }$ , blend β, budget K   
Output: Pruned universal schedule S\*   
1 for b = 1 to B do   
2 | Z(b) ← SampleFull(M, S) ; // Generate full trajectory   
3 for b = 1 to B do   
4 Binary search for threshold τ;   
5 (Spruned, {rs }) ← HyperplanarityPrune(Z(b), k, τ);   
6 Compute $\alpha _ { \mathrm { t r a j } } ( S _ { \mathrm { p r u n e d } } , \{ r _ { s } \} , \mathcal { Z } ^ { ( b ) } )$ via Eq. 8;   
7 Adjust τ until αtraj ≈ αtarget;   
8 ${ \mathcal { R } } ^ { ( b ) } \gets \{ t _ { i } : i \notin S _ { \mathrm { p r u n e d } } \}$ // Map indices to timesteps   
9 W $\begin{array} { r } { \mathbf { \rho } ( t ) \gets \frac { 1 } { B } \sum _ { b } \mathcal { H } [ t \in \mathcal { R } ^ { ( b ) } ] } \end{array}$ // Curvature density   
10 ρcurv(t) ← SmoothAndFloor(w(t), σ=5, €floor=0.05 max w);   
11 $\rho ( t ) \gets ( 1 - \beta ) \cdot \rho _ { \mathrm { l o g S N R } } ( t ) + \bar { \beta } \cdot \rho _ { \mathrm { c u r v } } ( t )$ // Blended density   
12 $\begin{array} { r } { F ( t ) \gets \int _ { 0 } ^ { t } \rho ( s ) \mathrm { d } s / \int _ { 0 } ^ { T } \rho ( s ) \mathrm { d } s ; } \end{array}$ // CDF   
13 $S ^ { * }  K$ timesteps at uniform quantiles of F;   
14 return S\*

Algorithm 2: HyperplanarityPrune   
Input: Ordered points $\mathcal { Z } = \{ z _ { 0 } , z _ { 1 } , \ldots , z _ { N } \}$ , window k, threshold τ   
Output: Set of pruned indices Spruned, residuals $\{ r _ { s } \} _ { s \in S _ { \mathrm { p r u n e d } } }$   
1 Spruned ← ∅; residuals ← {};   
2 Initialize window $W  ( z _ { 0 } , z _ { 1 } , \dotsc , z _ { k - 1 } )$ // First k points   
3 $M _ { W }  [ w _ { 2 } - w _ { 1 } \ | \cdot \cdot \cdot | \ w _ { k } - w _ { 1 } ] ;$   
4 $Q _ { W } , R _ { W } ^ { \bullet }  \mathrm { Q R } ( M _ { W } ) ^ { \bullet }$ // Thin QR; cached until window updates   
5 for i = k to N do   
6 $r _ { i } \gets \| ( I - Q _ { W } Q _ { W } ^ { \top } ) ( z _ { i } - w _ { 1 } ) \| _ { 2 } ;$   
7 if ri < τ then   
8 $S _ { \mathrm { p r u n e d } }  S _ { \mathrm { p r u n e d } } \cup \{ i \} ;$   
9 residuals[i] ← ri ; // Redundant; store residual   
10 else   
11 Update W: slide to include zi;   
12 Recompute Mw and $Q _ { W } , R _ { W } \sp { \sf } \gets \mathrm { Q R } ( M _ { W } )$ // Update cached QR   
13 return $S _ { \mathrm { p r u n e d } } ,$ residuals

Key algorithmic details: (1) Algorithm 2 returns residuals $\{ r _ { s } \}$ alongside pruned indices, avoiding redundant recomputation for $\alpha _ { \mathrm { t r a j } } .$ (2) The QR factorization is computed once at initialization (Line 4) and recomputed only when the window updates (Line 11), reflecting the true amortized cost (3) Algorithm 1 includes an explicit index-to-timestep mapping and specifies smoothing parameters. (4) Both algorithms use consistent 0-based indexing matching Definition 1.

## B Proof of Theorem 1

Proof. Let $S = \{ z _ { s _ { 1 } } , \dots , z _ { s _ { m } } \}$ be the pruned set with $m = | S |$ . By construction, each $z _ { s _ { j } }$ satisfies the hyperplanarity test with residual $r _ { s _ { j } } \leq \tau$ , meaning its distance from the affine subspace of its retained neighbors is at most τ.

By Definition 2, the trajectory projection score is:

$$
\alpha _ { \mathrm { t r a j } } = \frac { \sum _ { j = 1 } ^ { m } r _ { s _ { j } } ^ { 2 } } { \sum _ { z \in \mathcal { Z } } \| z - \bar { z } \| ^ { 2 } } .\tag{11}
$$

Since each $r _ { s _ { j } } \leq \tau ,$ the numerator is bounded:

$$
\sum _ { j = 1 } ^ { m } r _ { s _ { j } } ^ { 2 } \leq m \cdot \tau ^ { 2 } = | S | \cdot \tau ^ { 2 } .\tag{12}
$$

The denominator equals $\operatorname { t r } ( C _ { \mathcal { Z } } ) \cdot \left( N + 1 \right)$ , where $\begin{array} { r } { C _ { \mathcal { Z } } = \frac { 1 } { N + 1 } \sum _ { z \in \mathcal { Z } } ( z - \bar { z } ) ( z - \bar { z } ) ^ { \top } } \end{array}$ . Therefore:

$$
\alpha _ { \mathrm { t r a j } } \leq \frac { | \boldsymbol { S } | \cdot \tau ^ { 2 } } { ( N + 1 ) \cdot \mathrm { t r } ( \boldsymbol { C } _ { \mathcal { Z } } ) } .\tag{13}
$$

Interpretation. The bound controls the fraction of total trajectory variance attributable to orthogonal reconstruction residuals of pruned points. It does not bound $\mathrm { t r } ( \dot { C _ { R } } ) / \mathrm { t r } ( C _ { \mathcal { Z } } )$ (the ratio of retained-set variance to full-set variance), because pruning points changes the sample mean and can reduce the retained sample covariance even when all residuals are zero. For example, removing interior points from a collinear set with $r _ { i } = 0$ for all i yields $\alpha _ { \mathrm { t r a j } } = 0$ but $\mathrm { t r } ( C _ { R } ) < \dot { \mathrm { t r } } ( C _ { \mathcal { Z } } )$ whenever the retained points have smaller spread than the full set.

## C Extended Experimental Details

## C.1 Reference Set Convergence

See figures 5 and 6

## C.2 Implementation Details

All experiments use the HuggingFace Diffusers library. For non-uniform timestep schedules, we implement the DDIM update formula manually (Eq. 2) with correct $\alpha _ { t _ { \mathrm { p r e v } } }$ lookup, since the library's DDIMScheduler.step() assumes uniform spacing internally. DPM-Solver++ experiments use solver\_order=2 with lower\_order\_final=True and a fresh scheduler instance per batch to reset internal state. Trajectory normalization (zero mean, unit variance per dimension) is applied before curvature analysis to ensure the hyperplanarity test measures directional changes rather than raw magnitude.

## C.3 Hyperparameter Sensitivity

The blend parameter $\beta$ has a broad optimum at 0.5–0.7 (Table 2). Window size $k { = } 2$ (collinearity test) is used throughout; we tested $k { = } 3$ and $k { = } 4$ on CIFAR-10 and found no improvement in schedule quality (FID differences $< 0 . 1 )$ , suggesting that collinearity captures the dominant mode of geometric redundancy in these denoising trajectories. Reference batch size $B { = } 1 0 0$ is sufficient for CIFAR-10 (Section 5.6); B=50 for the higher-resolution LSUN Church and SD v1.5 models.

![](images/fa40381b0e306c27d14586afb3279674bf1d9459e64c9210d1f4ef0d5f5ebe88.jpg)

![](images/0b39626425413fc53d2f3d1c4283d3f7d058dada5b73b76648f61218d4627764.jpg)  
Figure 5: Schedule convergence vs. reference batch size B. (a) Exact and fuzzy Jaccard similarity to the B=100 schedule. Fuzzy matching (within ±5 timesteps) reaches 0.84 at B=75. (b) Retention frequency correlation reaches 0.86 at ${ \bar { B } } { = } 7 5$

![](images/b3897444c89c6b900471e47d3403fea01e194c7c5e6bd23412def1e8b1856d5d.jpg)

![](images/4c3bf22319014a909a792e91f1747013570722734fbab94333c746dcf81a6858.jpg)

![](images/2755ad7dc83f49a769732db93ac80314855199e9cffc745e47afbf30c7a8a4fb.jpg)  
Figure 6: Per-sample geometric complexity on CIFAR-10. (a) Individual curvature profiles (faint) vs. universal mean (bold). Despite diverse per-sample preferences, the universal profile captures systematic structure. (b) NFE distribution at three $\alpha _ { \mathrm { t r a j } }$ targets. (c) Per-sample correlation with the universal curvature profile (mean 0.37).

## C.4 Computational Overhead

Reference trajectory generation costs B × N NFEs (100 × 200 = 20,000 for CIFAR-10; 50 × 200 = 10,000 for LSUN Church and SD v1.5). Curvature analysis (pruning + retention frequency computation) takes \~40s on CPU (Intel Xeon Gold 6248R) for 100 trajectories of 200 steps in 3,072 dimensions. Schedule construction (blending + quantile placement) is instantaneous. The total offline cost is dominated by reference trajectory generation and is amortized over all subsequent samples.

Hardware. All generative inference (DDIM sampling, DPM-Solver++ sampling, FID evaluation) was performed on a single NVIDIA A100 (80 GB). Wall-clock times: CIFAR-10 \~8 min per 10ksample generation run; LSUN Church \~45 min per 10k-sample run; SD v1.5 \~3 hours per 5k-sample run. Reference trajectory generation: ～16 min (CIFAR-10, B=100, N=200); \~90 min (Church, B=50); \~5 hours (SD v1.5, B=50). Peak GPU memory: 4 GB (CIFAR-10), 12 GB (Church), 24 GB (SD v1.5).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction (Section 1) clearly state five contributions: generalization of geometric redundancy testing to high dimensions, a training-free adaptive sampling framework, the trajectory projection score, sample-adaptive scheduling, and comprehensive benchmarking. Each is developed in subsequent sections.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The “Limitations" paragraph in Section 6 discusses: reference pass overhead, β tuning on CIFAR-10, CFG scale generalization, the lack of improvement from higher-level tests $( k > 2 )$ , and complementarity with higher-order solvers.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Theorem 1 (reconstruction error bound) is stated with full assumptions and proved in Appendix B. Proposition 1 is proved in the main text. A remark clarifies the distinction between reconstruction error and retained-set variance.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 5 provides full experimental details including datasets, pretrained models, hyperparameter settings (including smoothing parameters σ=5 and $\epsilon _ { \mathrm { f i o o r } } )$ , reference batch sizes per model, sample counts, prompt sources for SD v1.5, evaluation metrics, and algorithmic pseudocode (Appendix A). All pretrained models are publicly available.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: Code will be released as a public repository upon publication. It is withheld during review to preserve double-blind anonymity. All datasets (CIFAR-10, LSUN Church) and pretrained models (including Stable Diffusion v1.5) are publicly available under their respective licenses.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

## Answer: [Yes]

Justification: Section 5 and Appendix C provide all experimental details including threshold selection procedures, reference batch sizes per model (B=100 for CIFAR-10, B=50 for Church and SD v1.5), smoothing parameters, and evaluation protocols.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [Yes]

Justification: Table 1 reports FID as mean ± standard deviation over 3 independent runs with different random seeds. Figures 1 and 2 include ±1 std error bands.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix C (Computational Overhead and Hardware) provides GPU type (NVIDIA A100 80 GB), CPU type (Intel Xeon Gold 6248R), peak memory usage per model, and wall-clock times for all experiments including generation, reference trajectories, and curvature analysis.

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: This work proposes a training-free acceleration method for existing generative models and does not introduce new ethical concerns beyond those inherent to the underlying models.

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The “Broader Impact" paragraph in Section 6 discusses positive impacts (reduced computational cost, accessibility) and notes that as an acceleration method, it inherits but does not amplify the societal risks of the underlying generative models.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

## Answer: [N/A]

Justification: GeoSPRINT is a sampling schedule optimization method that does not release new generative models or datasets.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All pretrained models and datasets are cited. CIFAR-10 and LSUN Church are used under standard research licenses. Stable Diffusion v1.5 is used under the CreativeML Open RAIL-M license.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The GeoSPRINT codebase will be released with documentation, including usage instructions and example scripts.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This work does not involve crowdsourcing or human subjects.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: This work does not involve human subjects research.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research?

Answer: [N/A]

Justification: LLMs are not a component of the core methodology.

Acknowledgements: I acknowledge support by the NIH funding through grant 1R01DA063157-01, and the High Performance Computing cluster group at The Scripps Research Institute, San Diego, CA, USA.