## Highlights

SUMNO: Spatially Unrolled Multigrid Neural Operator for Fluid Flow Surrogates

Yingqi Hu, Nan Deng, Changxiao Shao, Xiaozhou He

• Multigrid-aligned macro-architecture for fluid flow surrogates

• General shifted-window attention operator handling periodicity independently per axis

• Evaluated on wake, convection, vortex-sheet and cavity flows, unsteady and steady

• Best on 5 of 7 settings, about 30% lower rollout error, with 9.4M parameters

• Parameter-partitioned evaluation protocol and reproducible pipeline

# SUMNO: Spatially Unrolled Multigrid Neural Operator for Fluid Flow Surrogates

Yingqi Hu<sup>a,b</sup>, Nan Deng<sup>a,b</sup>, Changxiao Shao<sup>a,b</sup>, Xiaozhou He<sup>a,b</sup>

<sup>a</sup>College of Artificial Intelligence, Harbin Institute of Technology,

Shenzhen, 518055, People’s Republic of China

<sup>b</sup>Institute of Low-Altitude Science and Technology, Harbin Institute of Technology, Shenzhen, 518055, People’s Republic of China

## Abstract

Data-driven surrogate modeling has emerged as a powerful paradigm for accelerating computational fluid dynamics, and hierarchical U-shaped networks, which resemble the structure of classical Geometric Multigrid (GMG) methods, have become prevalent designs. Modern surrogates built on these designs achieve competitive accuracy at a fraction of the cost of numerical solvers; however, accurate prediction of complex flows remains challenging, and no existing architecture leads across them. In such networks, the multigrid analogy mainly concerns the multi-resolution topology, while their components largely follow common vision practice. To address this, this study introduces SUMNO, a multigrid neural operator for fluid flow fields on regular grids that spatially unrolls a V-cycle into a feed-forward Ushaped network. Parameter-free down- and upsampling stand in for restriction and prolongation, and additive skip connections for coarse-grid correction, with normalization-free transfer and skip pathways. Within this macroarchitecture, convolutions are used at the fine and intermediate resolutions and shifted-window attention at the coarsest, whose window partitioning is generalized to handle periodicity independently per axis. Evaluated on unseen test configurations across four unsteady and steady flow benchmarks against five representative baselines, SUMNO is best in 5 of 7 settings and ranks second in the remaining two. It reduces the 10-step rollout MSE by about 30% relative to the strongest baseline, and outperforms three baselines with over 130 million parameters using 9.4 million parameters.

Keywords: Neural surrogate, Multigrid, U-Net, Shifted-window attention, Periodic boundary conditions, Autoregressive rollout, Fluid dynamics

## 1. Introduction

High-fidelity numerical simulations of complex fluid flows, such as direct numerical simulation (DNS) and large eddy simulation (LES), impose prohibitive computational demands due to the broad range of active spatial and temporal scales governed by the Navier–Stokes equations (Pope, 2000). In response, data-driven surrogate modeling and scientific machine learning (SciML) have emerged as promising alternatives to accelerate computational fluid dynamics (CFD) (Thuerey et al., 2020; Kochkov et al., 2021). Because discretized Eulerian flow fields on regular grids mirror multi-channel image tensors, adapting architectures from computer vision (CV), from early convolutional networks (Ovadia et al., 2025; Bhatnagar et al., 2019) to recent attention-based and hybrid neural operators (Herde et al., 2024; Holzschuh et al., 2025; Li and Salim, 2026; Feng et al., 2026), has become a natural choice. These models achieve competitive accuracy at a fraction of the cost of numerical solvers.

Many leading surrogates adopt a hierarchical U-shaped design, whether as classical convolutional U-Nets (Ronneberger et al., 2015), attention-based operators such as ScOT (Herde et al., 2024) and PDE Transformer (Holzschuh et al., 2025), or frequency-coupled models such as DRIFT-Net (Li and Salim, 2026). This layout resembles classical Geometric Multigrid (GMG) methods (He and Xu, 2019; Briggs et al., 2000; Trottenberg et al., 2000), in which computation is arranged over a hierarchy of grid levels. The empirical success of these networks underscores the utility of such hierarchical arrangements as an efective structural template for fluid flow modeling. However, in these networks the multigrid analogy mainly concerns the multi-resolution topology, while the components within it, such as resampling, skip fusion and normalization, are largely carried over from vision practice rather than designed around fluid flow fields. Meanwhile, accurate prediction of complex flows, such as cylinder wakes and thermal convection, remains challenging, and the strongest baseline changes from one setting to the next (Table 4).

Vision practice was developed for natural images, which exhibit substantial redundancy (Liu et al., 2022). Fluid flow fields are diferent: they carry spatial diferential constraints at every coordinate, where local velocity gradients, strain rates and vorticity, defined by diferences between adjacent grid points, dictate the turbulent energy cascade (Pope, 2000). Fine-scale local continuity therefore has to be preserved at high resolution, whereas patchifying merges neighboring points into one token before any local operation acts on them. Alongside these local constraints, flow fields also carry global couplings, such as the instantaneous pressure coupling between distant vortices in incompressible flow (Warsi, 2005). Boundaries difer as well. An image border is an edge, but the boundary conditions of a flow field are set by the geometry of the domain: a periodic axis wraps around, so opposite borders are physical neighbors, and a domain may be periodic along one axis but bounded along the other. Windowed operators designed for images partition the grid with hard borders and do not connect tokens across them (Liu et al., 2021), so such periodic neighbors are separated into unrelated windows, and a single operator cannot serve periodic, partially periodic and bounded domains alike.

Motivated by these insights, this study introduces SUMNO (Spatially Unrolled Multigrid Neural Operator), a multigrid neural operator for fluid flow fields on regular grids. SUMNO spatially unrolls a multigrid V-cycle (Briggs et al., 2000) into a feed-forward U-shaped network, and its resampling, skip fusion and normalization each follow a multigrid counterpart. Parameter-free downsampling via BlurPool (Zhang, 2019) and bilinear upsampling stand in for restriction and prolongation; additive skip connections scaled by learnable relaxation factors stand in for coarse-grid correction; normalization is omitted from the transfer and skip pathways; and the network is asymmetric, with a capacity-rich encoder and a slim decoder, reminiscent of the light postsmoothing in multigrid cycles. At the operator level, convolutional blocks at the fine and intermediate resolutions preserve local continuity for gradient modeling (Xiao et al., 2021; Li et al., 2023a), and a shifted-window attention block at the coarsest resolution provides global coupling. The attention operator is generalized to handle periodicity independently per axis (TopoSwin, Section 3.3), so that tokens on opposite sides of a periodic border fall into the same window. Together with circular or reflection padding in convolutions, this allows a single network to cover doubly periodic, cylindrical and non-periodic domains.

SUMNO is evaluated on four benchmark flows: three unsteady dynamical systems, namely the doubly periodic Navier–Stokes sinusoidal vortex sheet (NS-SVS) (Herde et al., 2024), the open-boundary wake of actuated fluidic pinballs (Pinball) (Deng et al., 2025), and horizontally periodic Rayleigh– Bénard convection (RBC) (Ohana et al., 2024), and one steady boundaryvalue problem, the lid-driven cavity flow with internal obstacles (LDC) (Rabeh et al., 2025). Models are trained on single-step transitions and evaluated by autoregressive rollout at horizons of 1 and 14 steps on NS-SVS and 1 and 10 steps on Pinball and RBC, with LDC at steady state, which gives seven settings in total. Under a parameter-partitioned protocol (Table 1), the training, validation and test sets are disjoint at the trajectory level (the geometry level for LDC), with held-out initial vorticity fields, actuation controls, initial buoyancy fields or obstacle geometries.

On the two complex flows, Pinball and RBC, SUMNO is best in all four settings under both metrics: it roughly halves the single-step MSE of the strongest baseline and reduces the 10-step rollout MSE by about 30% on both (Table 4). With 9.4M parameters, SUMNO outperforms ScOT-B (Herde et al., 2024), PDET-B (Holzschuh et al., 2025) and DRIFT-Net (Li and Salim, 2026), each with over 130M parameters, in all seven settings (Table 2). SUMNO is also best on NS-SVS at one step, and ranks second on the remaining two settings, NS-SVS at 14 steps and LDC. Qualitatively, the baselines’ errors appear as grid-like textures, fine-scale ripples or smoothed fine structures, whereas SUMNO produces milder errors and retains more fine-scale structure (Fig. 5).

The remainder of this paper is organized as follows: Section 3 formalizes the multigrid-inspired formulation and details the architectural components; Section 4 outlines the benchmark configurations, baselines, model complexity and training specifications; Section 5 presents quantitative accuracy evaluations, a step-wise analysis of rollout errors, and qualitative comparisons of the predicted flow fields and their error distributions; and Section 6 concludes the paper.

## 2. Related Work

## 2.1. Neural Surrogates for Fluid Dynamics

Data-driven surrogates learn the mapping from the current flow state to the next state or to a steady solution, a formulation known as operator learning (Chen and Chen, 1995; Lu et al., 2021; Kovachki et al., 2023). Existing models difer in their overall layout and in the local operator used to mix spatial information.

Early surrogates applied convolutional networks to regular-grid fields, mostly within U-shaped encoder–decoder hierarchies (Ronneberger et al., 2015; Zhu and Zabaras, 2018), for steady flow prediction, aerodynamics, and time-dependent simulation (Guo et al., 2016; Thuerey et al., 2020; Bhatnagar et al., 2019; Rabeh et al., 2025; Geneva and Zabaras, 2020; Kochkov et al., 2021; Stachenfeld et al., 2022). Difusion-based refinement has been used to improve long rollouts of such backbones (Lippe et al., 2023). Fourier Neural Operators (FNO) (Li et al., 2021) instead stack spectral convolutions without a multi-resolution hierarchy, parameterizing integral kernels in the Fourier domain; they are resolution-independent, and extensions address irregular geometries and non-periodic domains (Li et al., 2023b; Tripura and Chakraborty, 2023). In these models, spatial information is mixed by convolutional or spectral operators.

A second line adopts attention as the local operator. Galerkin Transformer (Cao, 2021) applied self-attention to operator learning, interpreting a linearized variant as a Petrov–Galerkin projection. Subsequent designs use attention to handle irregular meshes, arbitrary query locations, or multiple input functions, making few assumptions on the sampling grid (Kissas et al., 2022; Li et al., 2023c; Hao et al., 2023). For regular grids, ScOT (Herde et al., 2024) and PDE Transformer (Holzschuh et al., 2025) follow vision transformers (Dosovitskiy et al., 2021), returning to a U-shaped hierarchy and using shifted-window attention (Swin) (Liu et al., 2021) as the operator at all resolutions, and are trained at scale on diverse datasets. DRIFT-Net (Li and Salim, 2026) retains the ScOT backbone but replaces window attention with parallel convolutional, MLP, and Fourier-domain branches. Thus, recent models share a U-shaped hierarchy and difer in the local operator and training scale. SUMNO adopts the same U-shaped template, but its inter-grid transfer, skip fusion and normalization follow multigrid counterparts (Section 3), with convolutions at fine and intermediate resolutions and shifted-window attention only at the coarsest scale. Unlike resolutionindependent or mesh-flexible operators, SUMNO targets fixed regular grids.

## 2.2. Multiscale Hierarchies and Multigrid

The U-shaped hierarchy resembles the structure of classical Geometric Multigrid (GMG) methods (Briggs et al., 2000; Hackbusch, 1985; Trottenberg et al., 2000), in which smoothing on fine grids is combined with restriction, coarse-grid correction, and prolongation. This correspondence has been made explicit in deep learning: MgNet (He and Xu, 2019) maps residual convolutional networks to multigrid cycles for elliptic PDEs, and subsequent studies learn relaxation, restriction and prolongation operators or neural preconditioners to accelerate linear solvers (Greenfeld et al., 2019; Margenberg et al.,

2022; Chen et al., 2022). These works target solvers, whereas SUMNO uses the multigrid cycle as the macro-architecture of a surrogate for flow fields, with parameter-free inter-grid transfer in place of learned transfer operators.

## 2.3. Boundary Handling and Topology

Grid-based networks handle boundaries in two places: convolutions and windowed attention. In convolutions, padding determines how fields are extended beyond the domain, with zero, reflection, and circular padding being common choices; circular padding corresponds to periodic domains and underlies the equivalence between convolution and Fourier-domain filtering (Gupta and Brandstetter, 2023), and is employed for periodic flows (Koehler et al., 2024).

In window-based Transformers, Swin (Liu et al., 2021) shifts windows by cyclically rolling the feature map and masks attention across the rolled seam. PDE Transformer (Holzschuh et al., 2025) supports fully periodic and fully non-periodic domains. In this work, the mask is derived from the periodicity of each axis rather than selected from these two cases, which recovers both and further covers mixed configurations such as cylinders.

## 2.4. Benchmarks and Evaluation Protocols

Several benchmark collections have been proposed for learning PDE solutions (Takamoto et al., 2022; Ohana et al., 2024; Koehler et al., 2024), ofering a wide range of metrics, from RMSE and its normalized or variance-scaled variants to maximum error, conservation, Fourier-band, $H ^ { 1 }$ and correlationbased measures.

Recent models are likewise evaluated with diferent metrics. ScOT (Herde et al., 2024) and DRIFT-Net (Li and Salim, 2026) report the relative $L _ { 1 }$ error at the final time, whereas PDE Transformer (Holzschuh et al., 2025) reports a normalized RMSE, which is equivalent to the relative $L _ { 2 }$ error. This work reports the mean squared error (MSE) and the relative $L _ { 2 }$ error $\left( \mathrm { R e l } { - } L _ { 2 } \right)$ ; the latter coincides with the metric of PDE Transformer and belongs to the same family of relative norm errors as that of ScOT and DRIFT-Net. All models are evaluated under a common protocol with shared metrics, rollout settings, mask handling, and validation-based model selection.

## 3. Methods

## 3.1. Problem Definition

This work investigates the data-driven surrogate modeling of two-dimensional incompressible fluid flows across two problem paradigms: unsteady temporal evolutionary dynamics and stationary boundary-value mappings.

For unsteady flow systems, the dynamic progression is formulated as a first-order Markovian transition:

$$
\hat { \mathbf { u } } _ { t + \Delta t } = \mathcal G _ { \theta } ( \mathbf u _ { t } ) ,\tag{1}
$$

where $\mathbf { u } _ { t } \in \mathbb { R } ^ { H \times W \times C }$ denotes the multi-channel physical state snapshot discretized on a Cartesian domain at time t. The parameterized surrogate $\mathcal { G } _ { \theta }$ maps the current state to the subsequent state $\hat { \mathbf { u } } _ { t + \Delta t }$ advanced by a discrete temporal increment $\Delta t$ . To evaluate long-term temporal dynamics, multistep predictions are obtained recursively via autoregressive rollout over L successive horizons:

$$
\begin{array} { r } { \hat { \mathbf { u } } _ { t + k \Delta t } = \mathcal G _ { \theta } \big ( \hat { \mathbf { u } } _ { t + ( k - 1 ) \Delta t } \big ) , \quad k = 1 , 2 , \ldots , L , } \end{array}\tag{2}
$$

initialized with the ground-truth state ${ \bf { u } } _ { t } .$ , such that $\hat { \mathbf { u } } _ { t } = \mathbf { u } _ { t }$

For steady flow problems, the task is formulated as mapping spatial geometry and governing parameters to the converged steady-state field:

$$
\hat { \mathbf { u } } _ { \infty } = \mathcal { G } _ { \boldsymbol { \theta } } ( \mathbf { g } , \pmb { \mu } ) ,\tag{3}
$$

where g denotes the spatial domain or boundary representation, µ represents the governing physical parameters such as the Reynolds number, and $\hat { \mathbf { u } } _ { \infty } \in$ $\mathbb { R } ^ { H \times W \times C }$ is the predicted steady-state field.

The optimization objective across both paradigms is to minimize the expected loss $\mathcal { L }$ between predictions and references:

$$
\operatorname* { m i n } _ { \theta } \ \mathbb { E } \left[ \mathcal { L } ( \hat { \mathbf { u } } , \mathbf { u } ) \right] ,\tag{4}
$$

where (uˆ, u) denotes $\left( \hat { \mathbf { u } } _ { t + \Delta t } , \mathbf { u } _ { t + \Delta t } \right)$ for unsteady evolutions and $\left( \hat { \mathbf { u } } _ { \infty } , \mathbf { u } _ { \infty } \right)$ for steady-state mappings.

![](images/9132541ab4b3a628415fb37b706fa72bc0a0523e8f8ad7959ed18a6ad84824e0.jpg)  
Figure 1: Overview of the SUMNO architecture (exemplified for $M = 4 \mathrm { s t a g e s } )$ . Left: multigrid-like U-shaped hierarchy across grid levels $\Omega _ { h } \to \cdots \to \Omega _ { 8 h }$ , with an asymmetric encoder–decoder, parameter-free resampling, and additive fusion with learnable relaxation factors. Right: the DoubleConv block used at fine and intermediate resolutions, and the TopoSwin block at the coarsest level, whose window attention (TopoWin-SA) follows the periodicity of each axis (highlighted). $\otimes ( \gamma )$ denotes LayerScale. Normalization layers appear only inside sub-blocks; no normalization is applied in the inter-grid transition and fusion paths. See main text for details.

## 3.2. Model Architecture

The surrogate network adopts a U-shaped multiscale topology organized across M stages spanning successive spatial grid levels $\Omega _ { h }  \Omega _ { 2 h }  \cdot \cdot \cdot $ $\Omega _ { 2 ^ { M - 1 } h }$ . This layout resembles a single V-cycle of geometric multigrid (GMG) methods (Trottenberg et al., 2000) unrolled in space: the descent to coarser grids forms the encoder stages, and the ascent back to finer grids forms the decoder stages. An overview of the architecture (instantiated with $M = 4$ stages) is illustrated in Figure 1.

Across this hierarchy, the forward pipeline is recursively structured. On the downward path for stage $m \in \{ 1 , \ldots , M - 1 \}$ , the encoding stage first applies a DoubleConv block, analogous to pre-smoothing, followed by a restriction operator mapping the latent fields to the next coarser grid $\Omega _ { 2 ^ { m } h }$ At the coarsest level $\Omega _ { 2 ^ { M - 1 } h }$ , a bottleneck module, analogous to a coarse-grid solver, processes domain-wide interactions. On the upward path, for stage m, a prolongation operator projects coarse representations back to the fine grid $\Omega _ { 2 ^ { m - 1 } h }$ , where the resulting coarse-grid corrections are combined with the lateral feature fields via fusion blocks, prior to the decoding stage, whose DoubleConv block is analogous to post-smoothing. At the finest grid level $\Omega _ { h }$ , an input stem and an output projection layer map the physical state fields to and from the latent variable space.

Optionally, for time-dependent problems, a global identity shortcut (indicated by the dashed path in Figure 1) can bypass the multiscale hierarchy to perform a discrete forward Euler update:

$$
\hat { \mathbf { u } } _ { t + \Delta t } = \mathbf { u } _ { t } + \Delta \hat { \mathbf { u } } ,\tag{5}
$$

where the backbone network directly predicts the temporal increment $\Delta \hat { \mathbf { u } }$ Otherwise, the output projection layer directly yields the target physical fields.

The individual operators constituting this architecture are detailed in the following sub-sections.

## 3.2.1. DownSampling and UpSampling Operators

For down-sampling, a Full Weighting (FW) type restriction operator $R : \Omega _ { h }  \Omega _ { 2 h }$ is adopted, the nine-point stencil that is the scaled adjoint of bilinear interpolation. In signal processing terms, it is the low-pass antialiasing filter of BlurPool (Zhang, 2019), built from the three-tap binomial filter applied along each axis, and is implemented as a fixed depthwise convolution with stride 2 and the kernel

$$
K _ { \mathrm { F W } } = \frac { 1 } { 1 6 } \left[ \begin{array} { l l l } { 1 } & { 2 } & { 1 } \\ { 2 } & { 4 } & { 2 } \\ { 1 } & { 2 } & { 1 } \end{array} \right] .\tag{6}
$$

The kernel is registered as a non-trainable bufer. Circular padding is used on periodic axes, so that the stencil wraps across the domain border, and reflection padding on non-periodic axes.

For up-sampling, a bilinear interpolation prolongation operator $P : \Omega _ { 2 h } $ $\Omega _ { h }$ is used, implemented as parameter-free cell-centered bilinear interpolation with scale factor 2. Border handling follows a pad-interpolate-crop scheme: circular padding on periodic axes, so that interpolation wraps across the border, and replicate padding on non-periodic axes.

When latent channel dimensions vary across grid levels, a pointwise $( 1 \times 1 )$ convolution is applied prior to the spatial inter-grid transfer operators; it acts on channels only, and the spatial operators R and P remain parameter-free. No normalization layer is applied in these transition stages.

## 3.2.2. Encoder and Decoder Blocks

Feature representations at each spatial scale are processed by Double Convolution (DoubleConv) blocks, analogous to multigrid pre- and postsmoothing operators. As illustrated in Figure 1, each DoubleConv block encapsulates two consecutive convolutional units wrapped by a single identity residual connection. A convolutional unit is defined as:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { c o n v } } ( \cdot ) = \mathrm { G E L U } \Big ( \mathrm { C o n v } _ { 3 \times 3 } \big ( \mathrm { G N } ( \cdot ) \big ) \Big ) , } \end{array}\tag{7}
$$

incorporating Group Normalization (GN) (Wu and He, 2018), a standard $3 \times 3$ convolution, and a GELU (Hendrycks and Gimpel, 2016) activation. The residual branch is scaled by LayerScale (Touvron et al., 2021), yielding the overall DoubleConv output:

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { o u t } } = \mathbf { x } + \gamma \odot ( \mathcal { T } _ { \mathrm { c o n v } } \circ \mathcal { T } _ { \mathrm { c o n v } } ) ( \mathbf { x } ) , } \end{array}\tag{8}
$$

where $\gamma \in \mathbb { R } ^ { C }$ is a learnable channel-wise scaling vector initialized to $1 0 ^ { - 6 }$ 2 and ⊙ denotes element-wise multiplication along the channel dimension.

## 3.2.3. Bottleneck Transformer

At the coarsest grid level $\Omega _ { 2 ^ { M - 1 } h }$ , a stack of shifted-window Transformer blocks (Liu et al., 2021) serves as the coarse-grid solver and couples the whole domain. Within each block, self-attention acts inside non-overlapping windows, and consecutive blocks shift the window grid by a preset ofset so that neighboring windows exchange information. Periodicity is handled independently per axis by the window attention (TopoWin-SA, Section 3.3); the resulting block is referred to as TopoSwin. Each TopoSwin block comprises two residual sub-layers, window self-attention and an MLP, each preceded by Layer Normalization (LN) (Ba et al., 2016).

The window size and shift are set automatically from the feature-map size, so that the same block can be placed at any grid level and on any aspect ratio. Given preset maximum window sizes $( K _ { h } ^ { \mathrm { m a x } } , K _ { w } ^ { \mathrm { m a x } } )$ that divide the feature-map dimensions $( H _ { \mathrm { i n } } , W _ { \mathrm { i n } } )$ , the efective window size contracts when the grid is smaller than the preset:

$$
K _ { h } = \operatorname* { m i n } ( K _ { h } ^ { \operatorname* { m a x } } , H _ { \mathrm { i n } } ) , \quad K _ { w } = \operatorname* { m i n } ( K _ { w } ^ { \operatorname* { m a x } } , W _ { \mathrm { i n } } ) .\tag{9}
$$

The shift along an axis is disabled when that axis holds a single window, since shifting would only fragment the field:

$$
S _ { i } = \left\{ \begin{array} { l l } { S _ { i } ^ { \operatorname* { m a x } } \bmod K _ { i } , } & { \mathrm { i f ~ } D _ { i } > K _ { i } , } \\ { 0 , } & { \mathrm { i f ~ } D _ { i } = K _ { i } , } \end{array} \right. \mathrm { ~ f o r ~ } i \in \{ h , w \} ,\tag{10}
$$

where $D _ { h } = H _ { \mathrm { i n } } , D _ { w } = W _ { \mathrm { i n } } .$ , and $S _ { i } ^ { \mathrm { m a x } }$ is the preset shift.

Queries and keys are normalized with RMS-Norm (Zhang and Sennrich, 2019) before the attention logits are computed (QK-Norm (Dehghani et al., 2023)). The MLP inserts a depthwise $3 \times 3$ convolution between its two linear layers, as in (Wu et al., 2021):

$$
\mathrm { C o n v M L P } ( \mathbf { z } ) = \mathbf { W } _ { \mathrm { 2 } } \mathrm { G E L U } \Big ( \mathrm { D W C o n v } _ { 3 \times 3 } \big ( \mathrm { G E L U } ( \mathbf { W } _ { 1 } \mathbf { z } + \mathbf { b } _ { 1 } ) \big ) \Big ) + \mathbf { b } _ { 2 } .\tag{11}
$$

Each residual branch is scaled by a learnable LayerScale vector $\gamma _ { 1 } , \gamma _ { 2 } \in \mathbb { R } ^ { C }$ 2 so that a TopoSwin block computes

$$
\mathbf { z } ^ { \prime } = \mathbf { z } + \gamma _ { 1 } \odot \mathrm { T o p o W i n } \mathrm { - S A } \left( \mathrm { L N } ( \mathbf { z } ) \right) ,\tag{12}
$$

$$
\begin{array} { r } { { \bf x } _ { \mathrm { o u t } } = { \bf z } ^ { \prime } + \gamma _ { 2 } \odot \mathrm { C o n v M L P } \big ( \mathrm { L N } ( { \bf z } ^ { \prime } ) \big ) . } \end{array}\tag{13}
$$

## 3.2.4. Fusion Operator

On the upward path, the coarse-grid correction $\mathbf { z } _ { \mathrm { u p } }$ from the prolongation operator is added to the fine-grid field $\mathbf { z } _ { \mathrm { s k i p } }$ , as in the correction step of multigrid. The two terms are weighted by a learnable channel-wise relaxation factor to control the variance of the sum:

$$
\lambda = \sigma ( \alpha ) ,\tag{14}
$$

$$
\mathbf { z } _ { \mathrm { f u s e d } } = \lambda \odot \mathbf { z } _ { \mathrm { s k i p } } + ( \mathbf { 1 } - \lambda ) \odot \mathbf { z } _ { \mathrm { u p } } ,\tag{15}
$$

where ${ \pmb { \alpha } } \in \mathbb { R } ^ { C }$ is a learnable parameter vector initialized to zero, and $\sigma ( \cdot )$ denotes the Sigmoid function, so that $\pmb { \lambda } \in ( 0 , 1 ) ^ { C }$ . The fused field $\mathbf { z } _ { \mathrm { f u s e d } }$ is subsequently passed to the post-smoothing decoder blocks.

## 3.2.5. Stem and Output Projections

Transitions between physical state fields and latent representations are handled by an input stem and an output projection layer. The input stem maps the discrete physical fields onto the finest latent grid $\Omega _ { h } .$ , and the output projection maps the decoder output on $\Omega _ { h }$ back to the physical state space. Both modules are implemented with convolutional layers, optionally followed by activation functions; their depth is adapted to the task.

## 3.3. Topology-Aware Boundary Handling

Boundary treatments follow domain topology, categorized into periodic and non-periodic regimes. Along periodic axes, spatial convolutions employ circular padding to preserve continuity across cylindrical or toroidal topologies. Along non-periodic axes, spatial convolutions employ reflection padding.

The same distinction applies to the bottleneck window attention (TopoWin-SA). In the standard Swin Transformer (Liu et al., 2021) processing images, a half-window shift splits the windows along both axes into fragments. For flows with a periodic direction, however, the two sides of that border are physical neighbors, so windows across it should remain intact. Figure 2(a) illustrates this on a cylindrical domain: along the periodic axis (horizontal, w), shifting leaves the windows intact, whereas along the non-periodic axis (vertical, h), fragments remain. Figure 2(b) details the resulting window behavior under all four boundary configurations. On a grid of 6 windows, a half-window shift fragments the domain into 12 sub-windows when both axes are non-periodic (top left), which is the behavior of the standard Swin Transformer. Merging the fragments across periodic borders yields 8 windows under h-periodicity and 9 under w-periodicity, and restores all 6 complete windows on a torus (h-w-periodic).

The standard Swin Transformer controls window behavior through an attention mask: tokens that a cyclic shift brings together from opposite borders are prevented from attending to each other. To obtain the window behavior described above, TopoWin-SA reformulates the mask as a single rule parameterized by the periodicity of each axis. The rule assigns each token of the shifted feature map a region tag, and a token attends only to tokens carrying the same tag. The tags are defined by an integer map $\mathbf { C } \in \{ 0 , 1 , 2 , 3 \} ^ { H _ { \mathrm { i n } } \times W _ { \mathrm { i n } } }$ , which encodes the shifted border regions of the two

![](images/11233723e099194ca041a06da960b453dd88b805cd60d293cdf6101505c769a4.jpg)  
Figure 2: Window partitioning across boundary topologies. (a) Window partitioning before (top) and after (bottom) shifting for a w-periodic domain (cylindrical topology). (b) Comparison across four boundary configurations, exemplified on a 6-window grid (dashed lines). Shifting by half a window splits the 6 windows into 12 fragments when both axes are non-periodic (top left). Across periodic boundaries, fragments straddling opposing edges are merged (linked by curved dashed lines, sharing identical colors and indices), yielding 8 windows for h-periodicity (top right) and 9 for w-periodicity (bottom left), and restoring all 6 complete windows on the torus topology (h-w-periodic, bottom right).

axes as two bits:

$$
\begin{array} { r } { \mathbf { C } _ { i , j } = \epsilon _ { h } \mathbb { I } _ { [ H _ { \mathrm { i n } } - S _ { h } , H _ { \mathrm { i n } } ) } ( i ) + 2 \epsilon _ { w } \mathbb { I } _ { [ W _ { \mathrm { i n } } - S _ { w } , W _ { \mathrm { i n } } ) } ( j ) , } \end{array}\tag{16}
$$

where $\mathbb { I } _ { [ \cdot ) } ( \cdot )$ is the interval indicator function, $( S _ { h } , S _ { w } )$ are the directional shift steps defined in Eq. (10), and $\epsilon _ { h } , \epsilon _ { w } \in \{ 0 , 1 \}$ equal 1 when the corresponding axis is non-periodic and 0 otherwise. For a periodic axis, $\epsilon = 0 .$ , so tokens on both sides of its border share the same tag and attend to each other without masking.

Given C, the attention masks are constructed per window. C is partitioned into windows in the same way as the tokens, and the tags of each window are compared pairwise by broadcast subtraction. Writing $\mathbf { c } _ { k }$ for the flattened tags of the k-th window, the mask is

$$
( \mathbf { A } _ { k } ^ { \mathrm { m a s k } } ) _ { p , q } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } ( \mathbf { c } _ { k } ) _ { p } = ( \mathbf { c } _ { k } ) _ { q } , } \\ { - \infty , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{17}
$$

where $p , q$ index the tokens within the window. When both axes are nonperiodic, the mask reduces to that of the standard shifted-window attention; when both are periodic, no token pair is masked; in mixed configurations, only the non-periodic axis contributes to the mask.

## 3.4. Principles of Model Configuration

While Sections 3.2–3.3 formulate the mathematical operators of individual components, this section outlines the guiding principles used to determine the architectural configurations shown in Figure 1. These choices govern two core aspects: the bottom-up determination of the grid hierarchy and stage count, and the computational-budget-driven allocation of channel widths, stage depths, and bottleneck attention windows.

## 3.4.1. Hierarchical Stages via Bottleneck Resolution Limits

In classical geometric multigrid methods, spatial coarsening is bounded by the smallest characteristic length scale of the flow field, as over-coarsening degrades coarse-grid corrections. Guided by this principle, the hierarchy is constructed bottom-up by first determining an irreducible coarse-grid dimension at the bottleneck, defined as $H \times n H$ , where $n = W _ { \mathrm { i n } } / H _ { \mathrm { i n } }$ denotes the domain aspect ratio. Once this minimal resolution is established, the network hierarchy grows outward toward the physical input resolution through successive dyadic refinements. The required stage count M is uniquely determined by the short-side input resolution $H _ { \mathrm { i n } }$

$$
M = \log _ { 2 } \left( \frac { H _ { \mathrm { i n } } } { H } \right) + 1 .\tag{18}
$$

In the canonical configuration, the bottleneck short side is empirically set to $H = 1 6$ . For input fields with $H _ { \mathrm { i n } } = 1 2 8$ , this naturally defines an $M = 4$ hierarchy across grid levels $\Omega _ { h }  \Omega _ { 2 h }  \Omega _ { 4 h }  \Omega _ { 8 h }$ . If higher-resolution inputs are processed, additional fine-grid stages are appended outward, preserving the same bottleneck scale.

## 3.4.2. Channels, Depths and Structural Hyperparameters

Following common practice, the number of channels doubles at each coarser level, [C, 2C, 4C, 8C]. For a $3 \times 3$ DoubleConv block on an $H _ { m } \times W _ { m } =$ n $H _ { m } ^ { 2 }$ grid with $C _ { m }$ channels, the computational cost is:

$$
\mathrm { M A C s } _ { \mathrm { D o u b l e C o n v } } = 2 \cdot 3 ^ { 2 } C _ { m } ^ { 2 } H _ { m } W _ { m } = 1 8 n H _ { m } ^ { 2 } C _ { m } ^ { 2 } .
$$

Under dyadic scaling, the spatial area contracts fourfold $( H _ { m + 1 } W _ { m + 1 } \ =$ $\textstyle { \frac { 1 } { 4 } } H _ { m } W _ { m } )$ precisely as channels double $\left( C _ { m + 1 } = 2 C _ { m } \right)$ . Consequently, the computational footprint of an individual DoubleConv block remains strictly invariant across all hierarchy levels.

This per-block cost invariance enables an asymmetric allocation of stage depths to optimize representational capacity under a constrained computational budget. Specifically, pre-smoothing depths in the encoder are progressively scaled across stages $\Omega _ { h }  \Omega _ { 4 h }$ (configured as [1, 2, 8] blocks with base width $C = 3 2 )$ . Because the per-block cost is the same at every level, stacking blocks at coarser resolutions adds depth at the same per-block cost. Conversely, motivated by the multigrid heuristic that post-smoothing should be kept minimal to prevent excessive smoothing, the decoder depth is uniformly restricted to 1 block per stage.

At the coarsest level $\Omega _ { 8 h } ~ ( H \times n H$ with width 8C), a Swin Transformer block serves as the bottleneck solver. For an attention window of size $K _ { h } \times K _ { w }$ with an FFN expansion ratio $r = 4$ , the theoretical computation per block is evaluated as:

$$
\begin{array} { r } { \mathrm { M A C s } _ { \mathrm { S w i n } } = ( 4 + 2 r ) H W ( 8 C ) ^ { 2 } + 2 K _ { h } K _ { w } H W ( 8 C ) } \\ { = 7 6 8 n H ^ { 2 } C ^ { 2 } + 1 6 n H ^ { 2 } K _ { h } K _ { w } C . \qquad } \end{array}\tag{19}
$$

Enlarging the attention window strengthens spatial representation. Under the configuration $H = 1 6$ and $C = 3 2$ , setting the window dimensions to $( K _ { h } , K _ { w } ) \ : = \ : ( H , H ) \ : = \ : ( 1 6 , 1 6 )$ is cheaper than increasing the number of channels, as the attention term $\left( 1 6 n H ^ { 2 } K _ { h } K _ { w } C \right)$ remains small relative to the channel projections $\left( 7 6 8 n H ^ { 2 } C ^ { 2 } \right)$ . The bottleneck is configured with 8 blocks (4 regular and shifted window pairs) using these full-span (16, 16) windows.

Finally, auxiliary structural hyperparameters are kept uniform across the architecture: the Transformer FFN expansion ratio is $r = 4$ , convolutions use $3 \times 3$ kernels, Group Normalization is configured with 16 groups, and the skip-connection residual scaling factors $\alpha _ { m }$ are initialized to 0.1.

## 4. Experiments

## 4.1. Benchmark Datasets

Four representative fluid flows are investigated in this work, spanning three unsteady dynamic systems and one stationary boundary-value problem under diverse flow configurations. A comprehensive summary of governing physical variables, spatiotemporal discretizations, sample sizes, evaluation rollout horizons, and parameter partitioning protocols is provided in Table 1, while representative flow field snapshots from each benchmark are visualized in Figure 3. The governing equations, simulation settings, and physical characteristics of each benchmark are detailed below, followed by the trajectory-level dataset splitting and parameter partitioning protocols in Section 4.2.

![](images/006a9982f871895ad7a714a8cb9743a5a939e8d6cd69bd9a1ab43d5a7b619f59.jpg)

![](images/28bbca3fafceeca0b0839eb053a03bc0abb4ff12921c56b906ccb20f8fc0ade1.jpg)

![](images/a8068c6ee39b1ffcd26923276ea65b74852f777182e56cc6f41e2883012cc064.jpg)  
Figure 3: Representative flow field snapshots selected from the four benchmark datasets: (a) vorticity field ω showing thermal plume structures in Rayleigh–Bénard convection (RBC, $1 2 8 \times 5 1 2$ $R a = 1 0 ^ { 8 }$ $P r = 0 . 2 )$ ; (b) converged velocity magnitude |u| overlaid with velocity vectors in the lid-driven cavity flow (LDC, 128 × 128, $R e = 6 3 )$ ; (c) vorticity field ω of the vortex-shedding wake behind rotating cylinders in the fluidic pinball benchmark $( 1 2 8 \times 3 8 4 ) $ ; (d) co-evolving vorticity field $\omega$ and passive scalar concentration c in the Navier–Stokes sinusoidal vortex sheet problem (NS-SVS, $1 2 8 \times 1 2 8 )$

The Navier-Stokes Sinusoidal Vortex Sheet (NS-SVS): This dataset is taken from the POSEIDON benchmark suite for fluidic foundation model (Herde et al., 2024). It models 2D incompressible ideal flows on a periodic torus, where initial energy is concentrated along perturbed sinusoidal curves synthesized by superimposing multiple high-frequency modes onto a fundamental wave, followed by divergence-free projection. The temporal dynamics are governed by the 2D Navier-Stokes equations coupled with a passive scalar advection equation. Each simulation trajectory is provided by the benchmark authors as uniform tensor fields across 21 temporal snapshots sampled on a discretized $1 2 8 \times 1 2 8$ Cartesian grid, yielding three physical channels per state: two velocity components $( u , v )$ and the passive tracer concentration c.

Table 1: Benchmark configurations, target physical variables, spatiotemporal dimensions, dataset scales, evaluation rollout horizons, and parameter partitioning protocols. Windows indicates the number of training and evaluation samples (Train / Test), and Horiz. specifies the autoregressive rollout horizon during evaluation.
<table><tr><td>Dataset</td><td>Variables</td><td>Grid  $( H \times W , T )$ </td><td>Windows</td><td></td><td>Horiz. Parameter Partitioning</td></tr><tr><td>NS-SVS</td><td> $( u , v , c )$ </td><td> $1 2 8 \times 1 2 8 , T = 2 1$ </td><td> $3 6 , 0 0 0 \ / \ 2 , 0 4 8$ </td><td>14</td><td>Disjoint initial vorticity  $\left( \omega _ { 0 } \right)$ </td></tr><tr><td>Pinball</td><td> $( u , v , p )$ </td><td> $1 2 8 \times 3 8 4 , T = 1 , 0 0 1$ </td><td> $1 2 , 0 0 0 \ / \ 2 , 0 4 8$ </td><td>10</td><td>Disjoint actuation controls (b)</td></tr><tr><td>RBC</td><td> $( u , v , p , b )$ </td><td> $1 2 8 \times 5 1 2 , T = 1 0 0$ </td><td> $1 2 , 0 0 0 \ / \ 2 , 0 4 8$ </td><td>10</td><td>Disjoint  ${ \bf { B } } _ { 0 }$  (Shared  $R a , P r )$ </td></tr><tr><td>LDC</td><td> $( u , v , p )$ </td><td> $1 2 8 \times 1 2 8 , \mathrm { S t e a d y }$ </td><td> $2 { , } 4 0 0 ~ / ~ 3 0 0$ </td><td></td><td>Disjoint geometries (Shared Re)</td></tr></table>

The Actuated Fluidic Pinballs (Pinball): The dataset is derived from numerical studies on the actuated fluidic pinballs (Deng et al., 2025), in which three rotating cylinders are placed in a triangular configuration. The nonuniform freestream flow impinges on these bluf-body obstacles, giving rise to wake flows exhibiting the Bénard-von Kármán vortex-shedding streets. Each trajectory is parameterized by the rotation speeds and rotational directions of the three cylinders, which yield a rich variety of vortex-shedding patterns and flow dynamics. While the original high-fidelity simulations were performed on a larger computational domain with well-defined boundary conditions, a central sub-region is cropped for neural-network training, resulting in openflow boundary conditions for the cropped dataset. The cropped subdomain is rasterized onto a uniform $1 2 8 \times 3 8 4$ Cartesian grid across all 1,001 available raw simulation snapshots, producing an open-boundary flow sequence containing velocity and pressure fields $( u , v , p )$

Rayleigh-Bénard Convection (RBC): This dataset is adopted from the Well collection (Ohana et al., 2024), where the RBC governing equations are solved via the spectral method. The simulations are conducted on a 2- D horizontally periodic domain with bottom heating. Each simulation run is governed by the non-dimensional Rayleigh number $R a$ , Prandtl number $P r$ , and initial buoyancy distribution $\mathbf { B } _ { 0 }$ . The published benchmark directly provides trajectories containing the velocity, pressure, and buoyancy fields $( u , v , p , b )$ , formatted as uniform $1 2 8 \times 5 1 2$ Cartesian arrays spanning 200 time steps. For this study, snapshots are extracted from the latter segment $( t \in$ [100, 199]) of each run, where the flow has developed into a fully turbulent state.

Lid-Driven Cavity Flow (LDC): This dataset originates from FlowBench (Rabeh et al., 2025; Tali et al., 2025), evaluating steady-state incompressible Navier–Stokes solutions within an enclosed cavity containing arbitrary internal solid obstacles. This work adopts the 2-D isothermal subset pre-rasterized on 128 × 128 Cartesian grids, containing the converged velocity and pressure fields $( u , v , p )$ alongside binary obstacle geometry masks across varied geometries and Reynolds numbers (Re). The benchmark assesses the stationary mapping from obstacle shapes to converged steady-state flow fields, conditioned on prescribed Reynolds numbers.

## 4.2. Data Processing Protocol

## 4.2.1. Dataset Splitting and Parameter Partitioning

To evaluate model generalization across physical parameter configurations while eliminating temporal and spatial data leakage, all datasets are split into mutually disjoint training, validation, and test subsets at the trajectory level (or at the geometry level for the stationary LDC case). Consequently, no temporal sequences or obstacle geometries overlap across splits. An independent validation set is reserved exclusively for hyperparameter tuning and model checkpoint selection. As summarized under Parameter Partitioning in Table 1, the underlying parameter space is categorized into shared and disjoint components:

For NS-SVS, the initial vorticity realizations $\omega _ { 0 }$ are mutually disjoint between splits. For Pinball, each subset is assigned mutually disjoint cylinder actuation controls $b _ { i }$ . For RBC, the global Rayleigh and Prandtl numbers (Ra, Pr) remain identically distributed across all splits, whereas trajectories originate from mutually exclusive initial buoyancy fields $\mathbf { B } _ { 0 }$ . For LDC, the Reynolds number Re distribution is shared across splits, while the immersed obstacle geometries remains disjoint to evaluate zero-shot geometric transfer.

## 4.2.2. Spatiotemporal Windowing and Autoregressive Rollout

Rather than directly ingesting entire simulation trajectories, surrogate models are trained and evaluated on discrete temporal slices, termed spatiotemporal windows. Formally, an individual window extracted from an underlying trajectory is characterized by its temporal origin $t _ { 0 } .$ sequence length $L ,$ and sampling stride $\Delta t _ { \mathrm { s w } }$ , defining a sequence of state snapshots $\{ t _ { 0 } + k \Delta t _ { \mathrm { s w } } \} _ { k = 0 } ^ { L - 1 }$ . During training, the models are optimized on short two-step windows $( L = 2 )$ to learn the one-step forward mapping $( t \to t + \Delta t _ { \mathrm { s w } } )$ . During evaluation, predictions are recursively rolled out over extended forecasting horizons (L = Horiz. in Table 1) to benchmark long-term performance and cumulative error propagation.

The window sampling stride is chosen on a per-dataset basis. For NS-SVS and RBC, consecutive frames are sampled using the temporal spacing as given by the dataset $( \Delta t _ { \mathrm { s w } } = \Delta t _ { \mathrm { f r a m e } } )$ . For Pinball, to capture long timehorizon vortex shedding evolutions while suppressing excessive temporal redundancy across adjacent frames, windows are extracted using an interleaved stride of 10 snapshots $( \Delta t _ { \mathrm { s w } } = 1 0 \Delta t _ { \mathrm { f r a m e } } )$ initiated from staggered time ofsets $t _ { 0 }$ . For stationary LDC, the temporal dimension degenerates $( L = 1 )$ , and each instance corresponds to a single spatial snapshot of the converged steady-state field.

Following window extraction within each trajectory split, all valid instances are pooled and shufled. A fixed quota is uniformly sampled from the respective pools to construct the final training and test sets (summarized under Windows in Table 1).

## 4.3. Baselines

To benchmark SUMNO against current surrogate paradigms, five representative neural operator and deep learning baselines are selected across diverse architectural families. These include transformer-based operators, specifically Scalable Operator Transformer (ScOT-B) (Herde et al., 2024) and PDE Transformer (PDET-B) (Holzschuh et al., 2025); frequency-domain architectures, comprising DRIFT-Net-B (Li and Salim, 2026) and Fourier Neural Operator (FNO) (Li et al., 2021); and the classical convolutional model U-Net (Ronneberger et al., 2015). For baseline architectures providing multiple capacity scales, variants with comparable computational budgets are chosen, denoted by the sufix “-B”. Specifically, FNO is configured with 64 hidden channels and truncated at 32 Fourier modes along each spatial dimension.

To ensure fair and rigorous comparisons across continuous flow fields, several targeted adjustments are implemented. For the convolutional baseline, the original BatchNorm and ReLU layers are replaced with Group Normalization (using 32 groups) (Wu and He, 2018) and GELU activations (Hendrycks and Gimpel, 2016), denoted as U-Net(GN). This adaptation conforms to standard SciML practices by eliminating batch-dimension statistical coupling while keeping the macro-topology identical to the original design. Additionally, because the oficial implementations of ScOT and DRIFT-Net only accommodate square spatial grids, minimal modifications are made to support non-square computational domains without altering their core operator designs or attention mechanics.

The parameter counts and measured computational complexities across all evaluated models are summarized in Table 2. For full reproducibility, all patched baseline implementations, training scripts, and configuration files will be released in an open-source repository.

Table 2: Computational complexity and parameter count across evaluated models. Multiply–accumulate operations (MACs) are measured in gigamacs (G) for a representative single forward pass, and model sizes are reported in millions (M) of parameters.
<table><tr><td>Models</td><td>Measured MACs (G)</td><td>Parameters (M)</td></tr><tr><td>ScOT-B</td><td>9.31</td><td>157.63</td></tr><tr><td>PDET-B</td><td>19.37</td><td>130.65</td></tr><tr><td>DRIFTNet-B</td><td>7.74</td><td>132.42</td></tr><tr><td>FNO</td><td>0.90</td><td>8.96</td></tr><tr><td>U-Net(GN)</td><td>12.08</td><td>31.04</td></tr><tr><td>Ours</td><td>6.31</td><td>9.40</td></tr></table>

## 4.4. Evaluation Protocol and Model Selection

For unsteady flows, both the single-step and the multi-step rollout performance are evaluated, while for the steady flow, only the final steady results are evaluated. Following prior works of ScOT and PDET, the NS-SVS and RBC are set to a rollout of 14 and 10 steps respectively. To maintain alignment, the Pinball is also evaluated on a rollout of 10 steps. For benchmarks with immersed solid bodies (i.e., the Pinball), the prediction inside the solid region is additionally reset to zero at every rollout step before being fed back as the next input, during both validation and testing. This reset is applied identically to all models, since omitting it (i.e., masking only the loss) was observed to degrade the rollout accuracy of the baselines.

Two metrics are reported for each evaluation: Mean Squared Error (MSE) and Relative $L _ { 2 }$ Error (RelL2). Errors of a single sample are calculated as:

$$
l _ { \mathrm { M S E } } = \frac { \Vert \mathbf { u } - \hat { \mathbf { u } } \Vert _ { 2 } ^ { 2 } } { \mathrm { d i m } ( \mathbf { u } ) }\tag{20}
$$

$$
l _ { \mathrm { R e l - } L _ { 2 } } = \frac { \Vert \mathbf { u } - \hat { \mathbf { u } } \Vert _ { 2 } } { \Vert \mathbf { u } \Vert _ { 2 } } ,\tag{21}
$$

where uˆ, $\textbf { u } \in \ \mathbb { R } ^ { d }$ denote the flattened prediction and ground truth fields respectively. The dimension d equals to the total amount of valid fluid pixels across all channels, excluding the solid interior regions if geometric obstacle masks exist (e.g., in the Pinball and LDC benchmarks). For a whole dataset the final testing metric is the averaged error of all samples:

$$
\mathcal { L } _ { \mathrm { t e s t } } = \frac { 1 } { N _ { \mathrm { t e } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e } } } l _ { i } ,\tag{22}
$$

where $N _ { \mathrm { t e } }$ is the number of test samples (windows) of a certain task and $l _ { i }$ is the corresponding sample-wise metric of sample i.

During training, a multi-step rollout evaluation identical to the test protocol is periodically conducted on the validation set, whereas single-step errors are additionally logged every epoch as diagnostics only. While models are optimized using single-step transitions, the final checkpoint for each architecture is selected based on the lowest multi-step rollout error on the validation set before being evaluated on the test set.

## 4.5. Training Details

All models are trained for 300 epochs using the AdamW optimizer (Loshchilov and Hutter, 2019) with a per-step cosine annealing schedule and a global batch size of 256. As summarized in Table 3, training objectives follow established baseline conventions: FNO is trained with the channel-wise relative $L _ { 2 }$ loss aligned with the oficial implementation, DRIFT-Net optimizes an $L _ { 1 }$ loss combined with the Frequency-Weighted Loss (FWL) (Li and Salim, 2026), and all other models (including SUMNO) are trained using the mean squared error (MSE). For benchmarks featuring immersed solid bodies (i.e., the Pinball and steady-state LDC datasets), training losses are computed exclusively over the fluid domain by masking out the interior obstacle region.

Key optimization hyperparametersprimarily peak learning rates, gradient clipping thresholds, and dataset-specific weight decaysare selected based on preliminary pilot runs to ensure stable optimization and proper convergence across diferent flow regimes. Peak learning rates and gradient clipping norms are paired with linear warmup schedules across all architectures. Weight decay is generally set to zero, as empirical sweeps indicated that non-zero $L _ { 2 }$ penalties blurred fine-scale flow structures without conferring rollout stability benefits. The exceptions are FNO, which adopts its literature-standard canonical value of $1 \times 1 0 ^ { - 4 }$ (Li et al., 2021), and U-Net(GN), which applies

Table 3: Detailed hyperparameter configurations across models and benchmark datasets. Dataset-specific values are ordered as (NS-SVS / Pinball / RBC / LDC). Columns list training objective (Loss), peak learning rate (LR), maximum gradient norm for clipping (Clip), number of linear warmup epochs (Warmup), weight decay (WD), and floating-point precision (Dtype).
<table><tr><td>Model</td><td>Loss</td><td>LR (×10−3)</td><td> $\mathbf { C l i p } \ ( \times 1 0 ^ { - 2 } )$ </td><td>Warmup</td><td>WD</td><td>Dtype</td></tr><tr><td>ScOT</td><td>MSE</td><td> $0 . 5 / 1 / 1 / 1$ </td><td>0.05/0.05/0.02/0.02</td><td> $3 / 6 / 6 / 5 0$ </td><td>0</td><td>BF16</td></tr><tr><td>PDET</td><td>MSE</td><td>0.2/0.2/0.2/1</td><td>0.02/0.02/0.02/0.02</td><td>3/6/6/50</td><td>0</td><td>BF16</td></tr><tr><td>DRIFT-Net</td><td>L1+FWL</td><td> $1 / 2 / 2 / 1$ </td><td>0.05/0.05/0.05/0.02</td><td>3/6/6/50</td><td>0</td><td>BF16</td></tr><tr><td>FNO</td><td>Rel L2</td><td>8</td><td>0.30/0.30/0.30/0.10</td><td>3/6/6/50</td><td> $1 0 ^ { - 4 }$ </td><td>FP32</td></tr><tr><td>U-Net(GN)</td><td>MSE</td><td>1</td><td>0.05</td><td>3/6/6/50</td><td> $0 / 0 / 0 / 1 0 ^ { - 4 }$ </td><td>BF16</td></tr><tr><td>Ours</td><td>MSE</td><td>1</td><td> $0 . 0 5 / 0 . 0 5 / 0 . 0 5 / 0 . 0 2$ </td><td>7/14/14/50</td><td>0</td><td>BF16</td></tr></table>

$1 \times 1 0 ^ { - 4 } \mathrm { o n }$ the LDC dataset to mitigate over-parameterization. For numerical precision, most models are trained in bfloat16 (BF16), whereas FNO uses single precision (FP32) due to implementation constraints with its spectral transform operations.

For SUMNO, Group Normalization is configured with 16 groups, and the stability constant ϵ for both normalization layers and the AdamW optimizer is set to $1 0 ^ { - 1 2 }$ , which was observed to improve training stability. Baseline models otherwise follow their standard implementations and default settings.

## 5. Results

## 5.1. Quantitative Evaluation against Baselines

Table 4 presents the quantitative performance of the proposed architecture alongside established neural surrogate baselines across four fluid dynamics benchmarks, spanning both transient rollouts and steady-state fields.

On the three unsteady benchmarks (NS-SVS, Pinball, and RBC), SUMNO achieves the lowest error in all single-step settings under both metrics. On NS-SVS, it reduces the single-step MSE by approximately 23% relative to the nearest baseline (ScOT-B). On Pinball and RBC, it roughly halves the single-step MSE of the strongest baseline (PDET-B, $5 . 2 4  2 . 5 9 \times 1 0 ^ { - 5 }$ ; $\mathrm { U - N e t ( G N ) , ~ 1 . 2 0 ~  ~ 0 . 6 0 ~ \times ~ 1 0 ^ { - 4 } ) }$ and reaches single-step $\mathrm { R e l } { - } L _ { 2 }$ errors of $0 . 7 1 \times 1 0 ^ { - 2 }$ and $1 . 9 4 \times 1 0 ^ { - 2 }$ , respectively.

Under extended autoregressive rollouts, SUMNO attains the lowest error on Pinball@10 $( 1 . 2 4 \times 1 0 ^ { - 3 } ~ \mathrm { M S E } , \ : 4 . 6 8 \times 1 0 ^ { - 2 } ~ \mathrm { R e l } { - } L _ { 2 } )$ and RBC@10 (1.36 × $1 0 ^ { - 3 } ~ \mathrm { M S E } , ~ 8 . 2 3 \times 1 0 ^ { - 2 } ~ \mathrm { R e l } { - } L _ { 2 } )$ . Relative to the strongest baseline in MSE, this corresponds to reductions of approximately 30% on both, against U-Net(GN) on Pinball and DRIFT-Net on RBC. The three baselines with over 130M parameters (ScOT-B, PDET-B and DRIFT-Net) are all outperformed by SUMNO, with 9.4M parameters, in every setting (Table 2).

Table 4: Quantitative evaluation of SUMNO and baseline architectures across physical benchmarks. Metrics include Mean Squared Error (MSE) and Relative $L _ { 2 }$ error $\left( \mathrm { R e l }  – L _ { 2 } \right)$ evaluated at single-step transitions (@1) and multi-step autoregressive rollout horizons (@10, @14). The best and second-best results are highlighted in bold and underlined, respectively. Scale multipliers for each column are specified in parentheses.
<table><tr><td>Model</td><td>SVS@1</td><td>Pinball@1</td><td>RBC@1</td><td>SVS@14</td><td>Pinball@10</td><td>RBC@10</td><td>LDC</td></tr><tr><td>MSE (↓)</td><td> $( \times 1 0 ^ { - 7 } )$ </td><td> $( \times 1 0 ^ { - 5 } )$ </td><td> $( \times 1 0 ^ { - 4 } )$ </td><td> $( \times 1 0 ^ { - 5 } )$ </td><td> $( \times 1 0 ^ { - 3 } )$ </td><td>(×10−3)</td><td> $( \times 1 0 ^ { - 4 } )$ </td></tr><tr><td>ScOT-B†</td><td>4.36</td><td>10.92</td><td>1.24</td><td>5.26</td><td>3.04</td><td>3.02</td><td>1.44</td></tr><tr><td>PDET-B</td><td>7.68</td><td>5.24</td><td>1.97</td><td>6.08</td><td>1.87</td><td>2.29</td><td>1.95</td></tr><tr><td>DRIFT-Net†</td><td>7.47</td><td>9.40</td><td>1.26</td><td>5.81</td><td>2.66</td><td>1.93</td><td>15.70</td></tr><tr><td>FNO</td><td>4.39</td><td>8.89</td><td>4.22</td><td>2.35</td><td>2.51</td><td>3.31</td><td>57.72</td></tr><tr><td>U-Net(GN)‡</td><td>6.93</td><td>6.13</td><td>1.20</td><td>15.77</td><td>1.76</td><td>3.06</td><td>0.27</td></tr><tr><td>SUMNO</td><td>3.35</td><td>2.59</td><td>0.60</td><td>2.45</td><td>1.24</td><td>1.36</td><td>0.64</td></tr><tr><td>Rel-L2 (↓)</td><td>(×10−3)</td><td>(×10−2)</td><td>(×10−2)</td><td>(×10−2)</td><td>(×10−2)</td><td>(×10−2)</td><td>(×10−2)</td></tr><tr><td>ScOT-B</td><td>2.45</td><td>1.49</td><td>2.86</td><td>2.72</td><td>7.53</td><td>14.99</td><td>4.09</td></tr><tr><td>PDET-B</td><td>3.28</td><td>1.05</td><td>3.65</td><td>2.99</td><td>6.03</td><td>11.59</td><td>5.51</td></tr><tr><td>DRIFT-Net</td><td>3.18</td><td>1.41</td><td>3.01</td><td>2.82</td><td>7.08</td><td>10.57</td><td>7.27</td></tr><tr><td>FNO</td><td>2.34</td><td>1.32</td><td>5.78</td><td>1.80</td><td>6.66</td><td>15.72</td><td>24.77</td></tr><tr><td>U-Net(GN)</td><td>3.09</td><td>1.07</td><td>2.75</td><td>4.85</td><td>5.56</td><td>15.45</td><td>1.92</td></tr><tr><td>SUMNO</td><td>2.05</td><td>0.71</td><td>1.94</td><td>1.88</td><td>4.68</td><td>8.23</td><td>3.54</td></tr></table>

†Original implementations only support square inputs; adaptations were introduced in the present work to accommodate non-square computational domains. ‡GN: GroupNorm and GELU replace the BatchNorm and ReLU of the canonical U-Net.

SUMNO is second on the remaining two settings, NS-SVS@14 and LDC, which favor the leading baselines. On NS-SVS@14, FNO attains the lowest error $( 2 . 3 5 { \times } 1 0 ^ { - 5 } \ \mathrm { M S E } , 1 . 8 0 { \times } 1 0 ^ { - 2 } \ \mathrm { R e l } { - } L _ { 2 } )$ , with SUMNO second $( 2 . 4 5 { \times } 1 0 ^ { - 5 }$ $1 . 8 8 \times 1 0 ^ { - 2 } )$ and the remaining baselines more than twice as high in MSE. On the steady-state lid-driven cavity (LDC) benchmark, which maps internal obstacle geometry masks to converged flow fields, U-Net(GN) attains the lowest error $( \mathrm { R e l } – L _ { 2 } = 1 . 9 2 \times 1 0 ^ { - 2 } )$ , and SUMNO is second $( 3 . 5 4 \times 1 0 ^ { - 2 } )$ ， ahead of ScOT-B, PDET-B, DRIFT-Net and FNO.

These two settings are consistent with the strengths of the leading baselines. NS-SVS is an idealized doubly periodic flow on a torus, which matches the Fourier basis of FNO. LDC is a static mapping from obstacle geometry to a steady field rather than flow-to-flow prediction, an image-to-image task suited to convolutional U-Nets. The ranking is mirrored across the two: FNO is best on NS-SVS@14 but worst on LDC, and U-Net(GN) is best on LDC but worst on NS-SVS@14.

## 5.2. Rollout Error over Steps

Table 4 reports only the first and last step of each rollout. Figure 4 shows the test-set average $\mathrm { R e l } { - } L _ { 2 }$ at every step. Its top row (a–c) plots the error of all six models against the rollout step, its middle row (d–f) compares SUMNO with the best baseline at each step, and its bottom row (g–i) shows how fast each model’s error grows from one step to the next.

![](images/b27fbd06726e42980ff0b93a9728c4835cbbc89fa83ab6218aa84d31b9bdac7c.jpg)  
Figure 4: Rollout error analysis on the three unsteady benchmarks. Columns correspond to NS-SVS (14 steps), Pinball (10 steps) and RBC (10 steps); LDC is omitted because it is a steady-state mapping without rollout. All quantities are computed from the test-set average $\mathrm { R e l } { - } L _ { 2 }$ at each step, and the six models share one legend. $\mathrm { ( a - c ) \ R e l - } L _ { 2 }$ error (in %) at each rollout step, on a logarithmic axis. (d–f) Ratio of the lowest Rel- $L _ { 2 }$ among the five baselines at that step to that of SUMNO; values above 1 indicate lower error for SUMNO, and the dashed line marks a ratio of 1 (equal error). (g–i) Step-to-step growth factor $g ( k ) = \mathrm { R e l } { - } L _ { 2 } ( k ) / \mathrm { R e l } { - } L _ { 2 } ( k - 1 )$ for $k \geq 2 .$ , with the dashed line marking $g = 1$ (no growth). Axis ranges are set independently for each panel.

The error of every model rises with the rollout step (Fig. 4a–c). This follows from the protocol shared by ScOT, PDE Transformer and DRIFT-Net (Herde et al., 2024; Holzschuh et al., 2025; Li and Salim, 2026): models are trained on single-step transitions and then fed their own predictions at inference, so an error made at one step is inherited by the next. Strategies such as pushforward training or noise injection can reduce this growth (Brandstetter et al., 2022; Sanchez-Gonzalez et al., 2020; Lippe et al., 2023), but they also mask diferences in what the networks themselves represent. The plain single-step protocol avoids this and directly exposes the representation ability and robustness of each architecture under the same error growth.

On Pinball and RBC, the curve of SUMNO lies below all baselines from the first step to the last, while the baselines cross one another and change rank along the rollout (Fig. 4b,c). On NS-SVS, SUMNO and FNO are the two lowest curves, clearly below the other four, and U-Net is the highest (Fig. 4a).

The middle row makes the comparison with the best baseline explicit. On Pinball and RBC the ratio stays above 1 at every step, so SUMNO leads throughout (Fig. 4e,f). The lead is largest at the first step, where the best baseline has about 1.4 to 1.5 times the error of SUMNO, and narrows to 1.19 on Pinball and 1.28 on RBC at the tenth step. On NS-SVS the ratio crosses 1 at the tenth step: SUMNO is lower than FNO for the first nine steps, and FNO is lower afterwards, ending about 4% below SUMNO at the fourteenth step (Fig. 4d). The two are close over the whole rollout, and averaged over all fourteen steps they difer by less than 1%.

The bottom row shows why the gaps are mostly set early. The growth factor of every model is largest in the first few steps and then settles at about 1.1 per step, with the curves of all models nearly overlapping late in the rollout (Fig. 4g–i). After the first few steps, all models therefore carry their accumulated error forward at similar rates, and the gap between them changes only slowly. SUMNO’s growth in the early steps is not lower than that of the baselines (on Pinball it is the highest of the six models at the second step), so its lead on Pinball and RBC originates in a lower error at the start of the rollout, which is then carried forward. On NS-SVS the late growth factors of SUMNO and FNO are practically identical, and FNO’s crossing comes from its slower growth in the first steps.

## 5.3. Qualitative Comparison of Predicted Flow Fields

Figure 5 compares the predicted flow fields of all models with the reference solution on a sampled test instance from each benchmark. Diferences among the architectures are most noticeable on Pinball@10 and RBC@10, whereas on SVS@14 and LDC all models reproduce the main structures with little visible deviation.

![](images/b432fbf66910e086412897b5bb0d83868cb6b51c075acb75ce1cd29203e37573.jpg)  
Figure 5: Qualitative comparison of predicted flow fields across the four benchmarks. Columns display the velocity component v for SVS@14, the vorticity field $\omega$ for Pinball@10, the buoyancy field b for RBC@10, and the velocity component v for steadystate LDC. Rows present the Ground Truth (GT) alongside predictions from SUMNO and five baseline models.

On Pinball@10, evaluating the derived vorticity field $\omega = \partial _ { x } v - \partial _ { y } u$ exposes fine spatial diferences. Under multi-step autoregressive rollouts, ScOT exhibits grid-like textures and fragmented wake patterns across the domain, and DRIFT-Net shows fainter grid-like textures in the wake. FNO displays fine-scale ripples of alternating sign that span the entire wake region. U-Net(GN) and PDET capture the global vortex layout with smooth interfaces between positive and negative vorticity regions, but over-smooth the intensity variations inside individual vortices. SUMNO retains the elongated shear layers and distinct downstream vortex cores; recurring multi-step errors nevertheless introduce localized noise and a slight bifurcation near the middle vortex core. All models, including the proposed one, show a slight phase lag: the leading vortex core has not advanced as far downstream as that in the ground truth. This lag can be related to training on single-step transitions while inference is autoregressive.

The buoyancy field of turbulent Rayleigh–Bénard convection (RBC@10), with heating at the bottom and cooling at the top, features a thin highbuoyancy layer (dark red) along the bottom boundary, from which slender plume stems rise into the interior and curl at their tips into mushroomlike heads. Between the stems, the interior is filled with thin, continuous filaments that wind and bend repeatedly. FNO is the blurriest: it smooths the field strongly and does not resolve the slender plume stems, which blend into broad, difuse regions. DRIFT-Net, PDET and ScOT are visibly similar to one another, recovering the main convective fronts and plume locations while blurring the thin filaments to a comparable degree. U-Net(GN) is sharper than these three and keeps continuous contours, but some of the finest filaments are still smeared out. SUMNO preserves more of the fine filaments and their winding paths than all baselines.

On SVS@14 and the steady-state LDC benchmark, visible diferences are small: all models reproduce the primary velocity structures, and in LDC all models recover the global recirculation pattern.

## 5.4. Qualitative Comparison of Spatial Error Distributions

Figure 6 shows the point-wise absolute errors of the same test instances as Figure 5. As single samples, their visual error levels may difer from the test-set averages in Table 4. The color scale is shared across models within each benchmark but not across benchmarks.

On SVS@14, the color scale spans only 0 to $9 . 0 \times 1 0 ^ { - 3 }$ , so absolute errors are small for all models. U-Net shows elevated error across the entire field. ScOT-B, PDET-B and DRIFT-Net show similar, scattered bright patches over the domain. SUMNO shows large bright patches concentrated at the vortex cores, which suggests a misalignment of core positions. FNO has the dimmest error map overall and also locates the vortex cores accurately.

![](images/2f689530869e7df79f8cdaf544b867285f1ad2fe05d78b31cbc68dd5151bdd0f.jpg)  
Figure 6: Point-wise absolute error distributions across the four benchmarks. Columns display the absolute error fields $\left| e _ { v } \right|$ for SVS@14, $| e _ { \omega } |$ for Pinball@10, $\left| e _ { b } \right|$ for RBC@10, and $\left| e _ { v } \right|$ for steady-state LDC, computed on the same sampled test instances as Figure 5. Rows compare SUMNO against five baseline architectures. For each benchmark, the color scale is shared across all models and capped at the largest per-model 98th percentile error.

On Pinball@10, all models show large bright regions around the first shed vortex behind the cylinders, which is the phase lag discussed in Section 5.3, whereas the downstream wake is comparatively dark. The absolute error scale (0 to 2.2) matches the range of the vorticity field itself, so phasematching errors in this region are of the same order as the signal. Together with the SVS@14 results, this suggests that the models difer in the intensity of the error, while phase matching is dificult for all of them under single-step training and autoregressive inference.

On RBC@10, bright regions follow the plumes and filaments, in contrast to the extended bright patches of the other benchmarks. FNO is brightest over the largest area, and U-Net is bright across the whole field, both along and between the filaments. ScOT-B, PDET-B and DRIFT-Net show weaker error than these two. The error of SUMNO is confined mainly to the filaments, with the regions between them close to dark, and both lateral borders also stay dark. No phase lag is evident in this sample.

On LDC, FNO has the brightest error, with large bright clumps spreading across the field and concentrated around the internal obstacle. U-Net and DRIFT-Net show dim error maps overall. ScOT-B, PDET-B and DRIFT-Net show grid-like fabric textures in their errors, whereas SUMNO shows a smooth, blurred error distribution at an overall level comparable to ScOT-B.

## 6. Conclusion

This paper presented SUMNO, a multigrid neural operator for fluid flow fields on regular grids, which spatially unrolls a multigrid V-cycle into a feedforward U-shaped network. Its resampling, skip fusion, and normalization each follow a multigrid counterpart, hierarchically integrating convolutional processing at fine and intermediate resolutions with a shifted-window attention bottleneck at the coarsest scale. The window partitioning is generalized to handle periodicity independently per spatial axis, allowing doubly periodic, cylindrical, and non-periodic domains to be modeled within a unified rule. Evaluated across four unsteady and steady benchmarks under held-out physical configurations, SUMNO ranks first in all four settings on the complex Pinball and RBC flows, cutting 10-step rollout MSE by about 30% on both over the strongest baselines, and leads in 5 of 7 settings overall. Operating with just 9.4M parameters, it outperforms models with over 130M parameters (ScOT-B, PDET-B, and DRIFT-Net). The remaining two settings are second place: NS-SVS at 14 steps behind FNO, and LDC behind U-Net(GN).

Limitations remain. Predicted fields still deviate visibly from the ground truth over extended rollouts; for instance, on the cylinder wake, the predicted vortices lag behind their true positions. SUMNO is designed for regular grids, and, as with data-driven surrogates in general, it does not provide formal accuracy or stability guarantees.

Future work includes extending the approach beyond regular grids to irregular or unstructured meshes and to three-dimensional flows, and enforcing exact physical invariants such as mass or energy conservation to improve the reliability of neural surrogates in practical simulations.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (Grant Nos. 12672294, 52676139, 12372216, W2541002, and 52276151), the Guangdong Basic and Applied Basic Research Foundation (Grant Nos. 2026A1515011445 and 2025A1515060008), and the Science, Technology and Innovation Commission of Shenzhen Municipality (Grant Nos. GXWD20220818113020001, GXWD20231130201948001, JCYJ20240813104853070, and JCYJ20240813105134045). The authors appreciate the valuable discussions and constructive suggestions with Luc R. Pastur.

## Declaration of competing interest

The authors declare that there is no conflict of interest regarding the publication of this paper.

## Code Availability

The implementation code, preprocessing scripts, configuration files and trained model checkpoints required to reproduce the results in this paper will be made publicly available upon peer-reviewed publication.

## Declaration of generative AI in scientific writing

During the preparation of this work, the authors used Doubao and Chat-GPT to improve the readability and language of the manuscript. After using this tool, the authors reviewed and edited the content as needed and assume full responsibility for the content of the published article.

## References

Ba, J.L., Kiros, J.R., Hinton, G.E., 2016. Layer normalization.

Bhatnagar, S., Afshar, Y., Pan, S., Duraisamy, K., Kaushik, S., 2019. Prediction of aerodynamic flow fields using convolutional neural networks. Computational Mechanics 64, 525–545. doi:10.1007/s00466-019-01740-0.

Brandstetter, J., Worrall, D.E., Welling, M., 2022. Message passing neural PDE solvers, in: International Conference on Learning Representations.

Briggs, W.L., Henson, V.E., McCormick, S.F., 2000. A Multigrid Tutorial. Second ed., Society for Industrial and Applied Mathematics, Philadelphia, PA.

Cao, S., 2021. Choose a transformer: Fourier or galerkin, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 24924–24940.

Chen, T., Chen, H., 1995. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. IEEE Transactions on Neural Networks 6, 911–917.

Chen, Y., Dong, B., Xu, J., 2022. Meta-mgnet: Meta multigrid networks for solving parameterized partial diferential equations. Journal of Computational Physics 455, 110996.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A.P., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer, L., Tschannen, M., Arnab, A., Wang, X., Riquelme Ruiz, C., Minderer, M., Puigcerver, J., Evci, U., Kumar, M., Steenkiste, S.V., Elsayed, G.F., Mahendran, A., Yu, F., Oliver, A., Huot, F., Bastings, J., Collier, M., Gritsenko, A.A., Birodkar, V., Vasconcelos, C.N., Tay, Y., Mensink, T., Kolesnikov, A., Pavetic, F., Tran, D., Kipf, T., Lucic, M., Zhai, X., Keysers, D., Harmsen, J.J., Houlsby, N., 2023. Scaling vision transformers to 22 billion parameters, in: Proceedings of the 40th International Conference on Machine Learning, PMLR. pp. 7480–7512.

Deng, N., Noack, B.R., Cornejo Maceda, G.Y., Pastur, L.R., 2025. Nonlinear dynamics of the actuated fluidic pinball steady, periodic, and chaotic regimes. Chaos, Solitons & Fractals 193, 116075.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2021. An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations (ICLR). arXiv:2010.11929.

Feng, Z., Tariq, Z., Zhang, Z., Zhao, P., Zhao, R., Wang, W., Huang, X., Yan, B., Shen, X., Zhang, F., 2026. Coswinnet: A conditional swin transformer

multimodal surrogate model for subsurface multiphase flow. Fuel 411, 138067.

Geneva, N., Zabaras, N., 2020. Modeling the dynamics of PDE systems with physics-constrained deep auto-regressive networks. Journal of Computational Physics 403, 109056.

Greenfeld, D., Galun, M., Basri, R., Yavneh, I., Kimmel, R., 2019. Learning to optimize multigrid PDE solvers, in: International Conference on Machine Learning (ICML), pp. 2415–2423.

Guo, X., Li, W., Iorio, F., 2016. Convolutional neural networks for steady flow approximation, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Association for Computing Machinery, New York, NY, USA. pp. 481–490. doi:10.1145/2939672.2939738.

Gupta, J.K., Brandstetter, J., 2023. Towards Multi-Spatiotemporal-Scale Generalized PDE Modeling. Transactions on Machine Learning Research

Hackbusch, W., 1985. Multi-Grid Methods and Applications. volume 4 of Springer Series in Computational Mathematics. Springer, Berlin, Heidelberg. 2013 softcover reprint, Springer Berlin Heidelberg.

Hao, Z., Wang, Z., Su, H., Ying, C., Dong, Y., Liu, S., Cheng, Z., Song, J., Zhu, J., 2023. GNOT: A general neural operator transformer for operator learning, in: International Conference on Machine Learning (ICML), pp. 12556–12569.

He, J., Xu, J., 2019. MgNet: A unified framework of multigrid and convolutional neural network. Science China Mathematics 62, 1331–1354. doi:10.1007/s11425-019-9547-2.

Hendrycks, D., Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 .

Herde, M., Raonić, B., Rohner, T., Käppeli, R., Molinaro, R., de Bézenac, E., Mishra, S., 2024. Poseidon: Eficient foundation models for pdes, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 72525–72624.

Holzschuh, B., Liu, Q., Kohl, G., Thuerey, N., 2025. PDE-transformer: Eficient and versatile transformers for physics simulations, in: Proceedings of the 42nd International Conference on Machine Learning, PMLR. pp. 23562–23602.

Kissas, G., Seidman, J.H., Guilhoto, L.F., Preciado, V.M., Pappas, G.J., Perdikaris, P., 2022. Learning operators with coupled attention. Journal of Machine Learning Research 23, 1–63.

Kochkov, D., Smith, J.A., Alieva, A., Wang, Q., Brenner, M.P., Hoyer, S., 2021. Machine learning–accelerated computational fluid dynamics. Proceedings of the National Academy of Sciences 118, e2101784118. doi:10.1073/pnas.2101784118.

Koehler, F., Niedermayr, S., Westermann, R., Thuerey, N., 2024. Apebench: A benchmark for autoregressive neural emulators of pdes, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 120252–120310.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., Anandkumar, A., 2023. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research 24, 1–97.

Li, J., Salim, F., 2026. Drift-net: A spectral-coupled neural operator for pdes learning, in: International Conference on Learning Representations, pp. 123958–123977.

Li, K., Wang, Y., Zhang, J., Gao, P., Song, G., Liu, Y., Li, H., Qiao, Y., 2023a. Uniformer: Unifying convolution and self-attention for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 12581–12600.

Li, Z., Huang, D.Z., Liu, B., Anandkumar, A., 2023b. Fourier neural operator with learned deformations for PDEs on general geometries. Journal of Machine Learning Research 24, 1–26.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A., 2021. Fourier neural operator for parametric partial diferential equations, in: International Conference on Learning Representations (ICLR). arXiv:2010.08895.

Li, Z., Meidani, K., Farimani, A.B., 2023c. Transformer for partial diferential equations’ operator learning. Transactions on Machine Learning Research

Lippe, P., Veeling, B.S., Perdikaris, P., Turner, R.E., Brandstetter, J., 2023. PDE-refiner: Achieving accurate long rollouts with neural PDE solvers, in: Advances in Neural Information Processing Systems (NeurIPS), pp. 67398–67433.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10012–10022.

Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022. A convnet for the 2020s, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986.

Loshchilov, I., Hutter, F., 2019. Decoupled Weight Decay Regularization, in: International Conference on Learning Representations.

Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G.E., 2021. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 3, 218–229.

Margenberg, N., Hartmann, D., Lessig, C., Richter, T., 2022. A neural network multigrid solver for the Navier-Stokes equations. Journal of Computational Physics 460, 110983.

Ohana, R., McCabe, M., Meyer, L., Morel, R., Agocs, F.J., Beneitez, M., Berger, M., Burkhart, B., Dalziel, S.B., Fielding, D.B., Fortunato, D., Goldberg, J.A., Hirashima, K., Jiang, Y.F., Kerswell, R.R., Maddu, S., Miller, J., Mukhopadhyay, P., Nixon, S.S., Shen, J., Watteaux, R., Blancard, B.R.S., Rozet, F., Parker, L., Cranmer, M., Ho, S., 2024. The well: a large-scale collection of diverse physics simulations for machine learning, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 44989–45037.

Ovadia, O., Oommen, V., Kahana, A., Peyvan, A., Turkel, E., Karniadakis, G.E., 2025. Real-time inference and extrapolation with time-conditioned

unet: Applications in hypersonic flows, incompressible flows, and global temperature forecasting. Computer Methods in Applied Mechanics and Engineering 441, 117982.

Pope, S.B., 2000. Turbulent Flows. Cambridge University Press, Cambridge,UK.

Rabeh, A., Herron, E., Balu, A., Sarkar, S., Hegde, C., Krishnamurthy, A., Ganapathysubramanian, B., 2025. Benchmarking scientific machinelearning approaches for flow prediction around complex geometries. Communications Engineering 4, 182.

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation, in: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Springer International Publishing, Cham. pp. 234–241.

Sanchez-Gonzalez, A., Godwin, J., Pfaf, T., Ying, R., Leskovec, J., Battaglia, P., 2020. Learning to simulate complex physics with graph networks, in: Proceedings of the 37th International Conference on Machine Learning, PMLR. pp. 8459–8468.

Stachenfeld, K., Fielding, D.B., Kochkov, D., Cranmer, M., Pfaf, T., Godwin, J., Cui, C., Ho, S., Battaglia, P., Sanchez-Gonzalez, A., 2022. Learned coarse models for eficient turbulence simulation, in: International Conference on Learning Representations (ICLR).

Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., Niepert, M., 2022. Pdebench: An extensive benchmark for scientific machine learning, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 1596–1611.

Tali, R., Rabeh, A., Yang, C.H., Shadkhah, M., Karki, S., Upadhyaya, A., Dhakshinamoorthy, S., Saadati, M., Sarkar, S., Krishnamurthy, A., Hegde, C., Balu, A., Ganapathysubramanian, B., 2025. Flowbench: A large scale benchmark for flow simulation over complex geometries. Journal of Datacentric Machine Learning Research 2, 1–35.

Thuerey, N., Weißenow, K., Prantl, L., Hu, X., 2020. Deep learning methods for reynolds-averaged navier–stokes simulations of airfoil flows. AIAA Journal 58, 25–36.

Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., Jégou, H., 2021. Going deeper with image transformers, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 32–42.

Tripura, T., Chakraborty, S., 2023. Wavelet neural operator for solving parametric partial diferential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering 404, 115783.

Trottenberg, U., Oosterlee, C.W., Schüller, A., 2000. Multigrid. Academic Press, San Diego, CA.

Warsi, Z.U.A., 2005. Fluid Dynamics: Theoretical and Computational Approaches. 3rd ed., CRC Press. doi:10.1201/9781420057881.

Wu, H., Xiao, B., Codella, N., Liu, M., Dai, X., Yuan, L., Zhang, L., 2021. Cvt: Introducing convolutions to vision transformers, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22–31.

Wu, Y., He, K., 2018. Group normalization, in: Computer Vision – ECCV 2018, Springer International Publishing, Cham. pp. 3–19.

Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollar, P., Girshick, R., 2021. Early convolutions help transformers see better, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 30392–30400.

Zhang, B., Sennrich, R., 2019. Root mean square layer normalization, in: Advances in Neural Information Processing Systems, Curran Associates, Inc.. pp. 12360–12371.

Zhang, R., 2019. Making convolutional networks shift-invariant again, in: Proceedings of the 36th International Conference on Machine Learning, PMLR. pp. 7324–7334.

Zhu, Y., Zabaras, N., 2018. Bayesian deep convolutional encoderdecoder networks for surrogate modeling and uncertainty quantification. Journal of Computational Physics 366, 415–447.