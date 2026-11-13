# PISim: A Unified Photonic Inference Simulator for Hardware-Aware Cross-Layer Design-Space Exploration

Andreas Falkenberg

QPI Laboratories, Inc.

Escondido, CA, USA

andreas@qpilabs.com

Abstract—Photonic neural accelerators offer ultra-low latency linear algebra, yet their usable numerical precision, sustainable throughput, and application accuracy remain tightly bounded by surrounding electro-optical interfaces and physical non-idealities. Pre-fabrication design-space exploration (DSE) requires a crosslayer electronic design automation (EDA) framework capable of evaluating disparate optical topologies under identical neural workloads and physical assumptions. We present PISim, a unified Photonic Inference Simulator that couples pretrained deep neural network execution with physics-based models of six accelerator architecture families, six electro-optic modulator technologies, four optical source organizations, and direct or coherent readout schemes. PISim embeds stage-resolved impairment operators— covering control and analog-to-digital quantization, finite extinction ratio, shot noise, correlated relative-intensity noise (RIN), thermal drift, propagation loss, and channel crosstalk— alongside an analytical cycle-level interface timing model to expose electronic-optical bottlenecks. Using simulated MNIST multilayer perceptron (MLP) benchmarks, we demonstrate how PISim identifies optimal precision operating points under multiimpairment stress (peaking at 6 bits for Mach–Zehnder and microring modulators), quantifies a 52.5 percentage-point accuracy gap between VCSEL-array and parallel-fanout laser architectures arising, under the modeled conditions, from power partitioning and correlated source noise, and models interfacelimited throughput scaling across emitter-array widths from 32 to 128 channels. This comprehensive framework establishes an extensible, reproducible foundation for accuracy-constrained Pareto optimization prior to photonic tape-out.

Index Terms—Electronic design automation, photonic neural accelerators, hardware-aware simulation, design-space exploration, cross-layer co-design, Photonic Inference Simulator, PISim.

## I. INTRODUCTION

Photonic integrated circuits (PICs) and free-space optical processors have emerged as compelling platforms for accelerating matrix-vector multiplication (MVM) in deep neural networks (DNNs) [3]–[6], [11]. By exploiting wave propagation, interference, and wavelength-division multiplexing (WDM), optical cores execute linear transformations at the speed of light with minimal dynamic energy dissipation inside the passive optical fabric. However, photonic neural accelerators expose an unusually coupled cross-layer design problem: while the optical core computes a linear transform passively, its effective bit precision, scheduling rate, and endto-end task accuracy are governed by the surrounding electrooptical system [11], [12]. Digital-to-analog converters (DACs), electro-optic modulators, optical power distribution networks, photodetectors, transimpedance amplifiers (TIAs), and analogto-digital converters (ADCs) impose strict bandwidth, dynamic range, and noise constraints that directly impact inference quality.

Consequently, selecting an accelerator topology, modulator technology, source organization, readout scheme, and tile geometry cannot be decoupled from the target neural workload [12], [13]. Surveys and reviews of photonic matrix multiplication highlight distinct implementation paradigms—including plane-light conversion, coherent interferometer meshes, microring weight banks, and electro-optic multiply–accumulate (MAC) units [3], [11]. Existing photonic design tools address valuable subsets of this stack: for example, L2ight focuses on scalable mapping, calibration, and in-situ subspace learning for coherent meshes [1], LoCI models optical loss and coherent crosstalk in silicon-photonic networks [2], and system-level frameworks examine physical-to-algorithm co-exploration and modular scale-out bottlenecks [12], [13]. Yet architects lack a unified pre-fabrication EDA environment that holds pretrained neural workloads fixed while systematically sweeping across disparate optical architecture families, device transfer curves, correlated noise mechanisms, and electronic interface schedules.

To address this gap, we introduce the Photonic Inference Simulator (PISim), an integrated cross-layer simulation framework designed for hardware-aware design-space exploration (DSE). PISim enables architects to screen candidate photonic accelerator configurations prior to fabrication while preserving explicit traceability between physical component parameters and application-level inference behavior. Specifically, the contributions of this work are fourfold:

1) Cross-Architecture Unified Modeling: A modular simulation engine supporting six photonic accelerator families, six modulator transfer models, four optical source organizations, and both incoherent (direct) and coherent (homodyne) detection paths within a single workload execution contract.

2) Stage-Resolved Physical Impairment Composition: Mathematical and stochastic formulations of hardware non-idealities—including pre- and post-transfer quantization, finite extinction ratio, signal-dependent shot noise, source-correlated relative-intensity noise (RIN), static fabrication offsets, slow thermal drift, insertion loss, and spatial/spectral crosstalk—injected at their exact physical stages in the forward optical path.

3) Interface-Aware Timing and Bottleneck Accounting: An analytical cycle-level throughput and latency model separating optical propagation from weight loading, DAC/ADC conversion, integration, and digital postprocessing to identify binding electronic-interface bottlenecks across varying array widths and tile counts.

4) Paired Diagnostics and Multi-Objective Exploration: Repeatable random-stream management enabling paired ideal-versus-photonic confusion matrices, per-class error diagnostics, multi-parameter impairment sweeps, and accuracy-constrained Pareto frontier generation.

The remainder of this extended manuscript is organized as follows. Section II details the unified simulator architecture, taxonomy of supported photonic topologies, device transfer functions, and source/readout models. Section III formalizes the forward optical operators, signed arithmetic encodings, and stage-resolved impairment models. Section IV presents the analytical interface-aware timing, useful-work accounting, and Pareto DSE methodology. Section V evaluates PISim across five exploratory case studies (E1–E5) examining modulator sensitivity, multi-impairment precision scaling, source organization trade-offs, and modeled array-width throughput scaling. Section VI discusses validation priorities and roadmap extensions, and Section VII concludes the paper.

## II. UNIFIED ARCHITECTURE AND PHYSICAL DEVICE MODELS

## A. Cross-Layer Workflow and Execution Contract

PISim bridges high-level deep learning frameworks and low-level photonic physics through a deterministic workload mapping contract. Given a pretrained neural network checkpoint (such as MLPs, ResNet-8, VGG-8, or Mixer-Tiny evaluated on MNIST, Fashion-MNIST, or CIFAR-10) and an input dataset, the simulator constructs a dual-path execution graph. Every input sample $x \in \mathbb { R } ^ { N _ { \mathrm { i n } } }$ is propagated simultaneously through an ideal floating-point reference path and a configurable photonic physical path using identical preprocessing and sample ordering.

To prevent ambiguous comparisons between partial optical acceleration and full-system execution, PISim enforces an explicit operator-mapping contract. Linear layers and lowered convolutional layers (via im2col matrix expansion) are partitioned into hardware tiles of dimension $W _ { \mathrm { a r r a y } } \times H _ { \mathrm { a r r a y } }$

![](images/5b569f5af1033b2bf5d6677395c0d8fc829faacafefc3d4e91770d9af5a1989d.jpg)  
Fig. 1. Overview of the PISim unified photonic inference simulation workflow, coupling neural network layer lowering with stage-resolved physical device models, stochastic impairments, and interface timing analysis.

Non-linear activations, batch normalization, residual additions, and pooling operations are explicitly assigned either to digital electronic post-processing or to optical nonlinear elements (when supported by the chosen topology). For each mapped tile, the simulator tracks:

• Weight Persistence Policy: Whether weights are programmed statically once per inference run (e.g., nonvolatile phase-change cells or thermally tuned meshes), reloaded per tile fold, or streamed dynamically per input sample.

• Active vs. Padded Arithmetic: Useful MAC operations are counted strictly over active matrix entries, separating zero-padded array channels and redundant multirail encoding lanes from true algorithmic work.

## B. Supported Photonic Accelerator Families

Photonic accelerators execute linear transforms using fundamentally different physical field representations. A coherent interferometer mesh manipulates complex optical field amplitudes and phases, whereas an incoherent WDM weight bank sums optical intensities across distinct wavelengths. PISim maintains strict architectural neutrality by decoupling the compute topology from the underlying electro-optic device technology. Table I summarizes the six supported accelerator families, their required physical state representations, and their primary EDA exploration variables.

## C. Electro-Optic Modulator Transfer Models

PISim parameterizes six distinct electro-optic modulator technologies, mapping normalized digital control drives $u \in$ (or signed drives $u \in [ - 1 , 1 ] )$ to optical field transmission t(u) or intensity transmission $T ( u ) ~ = ~ | t ( u ) | ^ { 2 }$ . Including finite extinction ratio $\mathrm { E R } _ { \mathrm { d B } }$ (where $\epsilon = 1 0 ^ { - \mathrm { E R } _ { \mathrm { d B } } / 1 0 }$ denotes the residual transmission floor), the six device models are formulated as follows:

TABLE I  
TAXONOMY OF SUPPORTED PHOTONIC ACCELERATOR FAMILIES AND EDA DESIGN VARIABLES IN PISIM
<table><tr><td>Architecture Family</td><td>Physical Representation &amp; Forward Operator</td><td>Primary EDA Design Variables</td></tr><tr><td>Diffractive Optical Networks [6]</td><td>Free-space or slab spatial complex field propagation across phase/amplitude diffractive masks onto detector regions.</td><td>Mask resolution, layer axial spacing, diffraction efficiency, transverse misalignment, and detector pixel binning.</td></tr><tr><td>Mach-Zehnder Interferometer (MZI) Meshes [4], [7]</td><td>Coherent unitary/arbitrary matrix decomposition (UΣV†) via cascaded  $2 \times 2$  beamsplitters and thermo-/electro-optic phase shifters.</td><td>Mesh decomposition topology  $( \mathrm { e . g . }$  , Clements [7]), phase bit precision, calibration error, and path-dependent insertion loss.</td></tr><tr><td>Electro-Optic MAC Units [5]</td><td>Point-to-point or broadcast-and-weight intensity/field modulation followed by spatial or temporal photodetector accumulation.</td><td>Modulator transfer curve, signed encoding convention, DAC/ADC sampling rates, and matrix tiling factor.</td></tr><tr><td>Resonator-Based Inference [8]</td><td>WDM spectral routing through cascaded microring resonator (MRR) weight banks with differential balanced readout.</td><td>Resonance detuning, WDM channel spacing, thermal drift sensitivity weight state resolution, and inter-channel crosstalk.</td></tr><tr><td>VCSEL-Array Processors</td><td>Spatially parallel vertical-cavity surface-emitting laser arrays with direct current modulation and optical fan-in/fan-out.</td><td>Emitter array width  $( N _ { \mathrm { v c s e l } } ) .$  inter-emitter coupling, nonlinear L-I dynamics, and detector integration window.</td></tr><tr><td>Free-Space Optical Accelerators [9]</td><td>Spatial light modulation (SLM/DMD) with cylindrical/Fourier lens optical matrix-vector projection onto camera/photodiode arrays.</td><td>Spatial pixel encoding, optical point-spread function, SLM frame refresh rate, and sensor readout schedule.</td></tr></table>

1) Thin-Film $\mathbf { L i N b O _ { 3 } }$ Mach–Zehnder Modulator (MZM): Biased at the quadrature point $( \phi _ { 0 } ~ = ~ \pi / 2 )$ with push-pull phase shift $\begin{array} { r } { \Delta \phi ( u ) = \frac { \pi } { 2 } u } \end{array}$ for $u \in [ - 1 , 1 ] .$ the normalized intensity transmission follows a raised-cosine characteristic:

$$
\begin{array} { l } { \displaystyle { T _ { \mathrm { M Z M } } ( u ) = \epsilon + ( 1 - \epsilon ) \sin ^ { 2 } \left( \frac { \pi } { 4 } ( 1 + u ) \right) } } \\ { \displaystyle { \quad = \epsilon + \frac { 1 - \epsilon } { 2 } \left( 1 + \sin \left( \frac { \pi } { 2 } u \right) \right) . } } \end{array}\tag{1}
$$

In its linearized pre-distorted mode, PISim applies an inverse arcsine lookup $\textstyle u _ { \mathrm { l i n } } ~ = ~ { \frac { 2 } { \pi } }$ arcsin(2˜u − 1) prior to DAC quantization so that DAC discrete levels map linearly onto optical intensity.

2) Silicon Microring Resonator (Ring): Modeled via an all-pass Lorentzian transfer function governed by resonance wavelength detuning $\delta \lambda ( u )$ normalized to the full-width at half-maximum linewidth ∆λ<sub>FWHM</sub>:

$$
T _ { \mathrm { R i n g } } ( u ) = \epsilon + ( 1 - \epsilon ) \frac { \left( \frac { 2 ( \delta \lambda _ { 0 } + \kappa _ { \lambda } u ) } { \Delta \lambda _ { \mathrm { F W H M } } } \right) ^ { 2 } } { 1 + \left( \frac { 2 ( \delta \lambda _ { 0 } + \kappa _ { \lambda } u ) } { \Delta \lambda _ { \mathrm { F W H M } } } \right) ^ { 2 } } ,\tag{2}
$$

where $\delta \lambda _ { 0 }$ is the static bias offset and $\kappa _ { \lambda }$ is the tuning efficiency. Thermal perturbations shift $\delta \lambda _ { 0 } .$ , introducing nonlinear weight distortion.

3) InP Quantum-Confined Stark Effect Electro-Absorption Modulator (InP-QCSE EAM): Modeled

via voltage-dependent exponential optical absorption $\alpha ( u )$ over active waveguide length $L _ { \mathrm { w g } } \mathrm { : }$

$$
T _ { \mathrm { E A M } } ( u ) = \epsilon + ( 1 - \epsilon ) \exp \left( - \alpha _ { 0 } L _ { \mathrm { w g } } \left( \frac { u } { u _ { \mathrm { m a x } } } \right) ^ { \gamma } \right)\tag{3}
$$

with optional phase-amplitude chirp coupling $\begin{array} { c c l } { \Delta \phi _ { \mathrm { c h i r p } } ( u ) } & { = } & { \frac { \alpha _ { H } } { 2 } \ln T _ { \mathrm { E A M } } ( u ) } \end{array}$ parameterized by the Henry linewidth enhancement factor $\alpha _ { H }$ when used in coherent paths.

4) Phase-Change Material $\begin{array} { r l r } { { \bf C e l l } } & { { } } & { ( { \bf G e } _ { 2 } { \bf S } { \bf b } _ { 2 } { \bf T e } _ { 5 } , } \end{array}$ PCM): Represents nonvolatile multi-level weights programmed into M discrete crystalline-amorphous conductance/attenuation states $\left\{ s _ { 0 } , s _ { 1 } , \ldots , s _ { M - 1 } \right\}$

$$
T _ { \mathrm { P C M } } ( m ) = \epsilon + ( 1 - \epsilon ) T _ { \mathrm { s t a t e } } ( s _ { m } ) , \quad m \in \{ 0 , \ldots , M - 1 \} ,\tag{4}
$$

capturing non-uniform state spacing and stochastic programming variability without static holding power.

5) Digital Micromirror Device (DLP / DMD): Operates as a binary spatial reflector $b _ { k } \in \{ 0 , 1 \}$ modulated via pulse-width temporal grayscale integration across B bitplanes within integration window $T _ { \mathrm { i n t } } \mathrm { : }$

$$
T _ { \mathrm { D M D } } ( u ) = \epsilon + ( 1 - \epsilon ) \frac { 1 } { 2 ^ { B } - 1 } \sum _ { k = 0 } ^ { B - 1 } 2 ^ { k } b _ { k } ( u ) .\tag{5}
$$

6) Surface-Normal QCSE Micro-Pad: Models verticalincidence electro-absorption pixels subject to electrical RC bandwidth roll-off and photo-generated carrier screening at high optical intensities:

$$
\begin{array} { r l } & { T _ { \mathrm { m i c r o - p a d } } ( u , I _ { \mathrm { o p t } } ) = T _ { \mathrm { Q C S E } } \left( u _ { \mathrm { e f f } } ( t ) - \beta _ { \mathrm { s c r } } I _ { \mathrm { o p t } } \right) , } \\ & { \qquad \quad \tau _ { R C } \frac { d u _ { \mathrm { e f f } } } { d t } + u _ { \mathrm { e f f } } = u ( t ) , } \end{array}\tag{6}
$$

where $\tau _ { R C }$ is the pad charging time constant and $\beta _ { \mathrm { s c r } }$ captures saturation due to carrier screening.

## D. Optical Source Organizations and Detection Paths

PISim separates optical source architecture from matrix compute topology across four configurations:

• VCSEL Array: $N _ { \mathrm { v c s e l } }$ independent emitters drive parallel spatial or spectral channels. Each emitter carries independent relative-intensity noise and thermal fluctuations, avoiding $1 / N$ passive splitting loss at the expense of peremitter power and wavelength calibration.

• Single Laser with Parallel Fanout: A master continuous-wave (CW) laser of power $P _ { 0 }$ is split across N channels via a $1 \times N$ star coupler or splitter tree. Each channel receives $P _ { \mathrm { c h } } = \eta _ { \mathrm { s p l i t } } P _ { 0 } / N$ , and master laser RIN propagates as a spatially correlated common-mode fluctuation across all N channels simultaneously.

• Time-Serialized Single Laser: Full laser power $P _ { 0 }$ is directed sequentially through a single high-speed modulator-detector channel over N time slots, trading temporal latency for maximum per-sample optical signalto-noise ratio (SNR) and zero spatial fanout loss.

• LED Micro-Display: Incoherent broad-spectrum emitter array restricted strictly to intensity-based direct detection without phase interference.

For readout, PISim supports direct incoherent detection, where photodiode current $I _ { \mathrm { P D } } ~ = ~ { \mathcal R } P _ { \mathrm { o p t } }$ is proportional to incident optical power (R is responsivity in A/W), and coherent homodyne detection, where the signal field $E _ { \mathrm { s i g } }$ is mixed with a phase-locked local oscillator $E _ { \mathrm { L O } }$ in a balanced receiver yielding differential photocurrent $I _ { \mathrm { b a l } }$ ∝ 2R $\sqrt { P _ { \mathrm { s i g } } P _ { \mathrm { L O } } } \cos ( \Delta \theta )$

## III. FORWARD OPERATORS AND STAGE-RESOLVEDIMPAIRMENTS

## A. Signed Neural Arithmetic Encoding

Because neural network weights $W \in \mathbb { R } ^ { M \times N }$ and activations $\boldsymbol { x } \in \mathbb { R } ^ { N }$ take signed values while direct photodetection measures non-negative optical power $P _ { \mathrm { o p t } } \geq 0$ , PISim enforces explicit signed encoding contracts:

• Differential Dual-Rail Encoding: Each signed weight $w _ { i j }$ is decomposed into positive and negative components $w _ { i j } = w _ { i j } ^ { + } - w _ { i j } ^ { - }$ with $w _ { i j } ^ { + } , w _ { i j } ^ { - } \geq 0$ , routed to balanced photodetector pairs. While doubling hardware rails or time slots, balanced subtraction cancels common-mode intensity offsets and residual extinction floors ϵ.

• Biased Offset Encoding: Weights and inputs are shifted to non-negative optical intensities $w _ { i j } ^ { \mathrm { o p t } } = w _ { i j } + w _ { \mathrm { b i a s } }$ and ${ x _ { j } ^ { \mathrm { { o p t } } } = x _ { j } + x _ { \mathrm { { b i a s } } } }$ . The accumulated photocurrent contains cross-terms $\begin{array} { r } { \sum _ { j } ( w _ { i j } x _ { j } + w _ { i j } x _ { \mathrm { b i a s } } + w _ { \mathrm { b i a s } } x _ { j } + w _ { \mathrm { b i a s } } x _ { \mathrm { b i a s } } ) } \end{array}$ that are subtracted digitally or via reference channels. Unlike differential encoding, biased offset schemes suffer directly from uncancelled finite extinction floors ϵ.

• Coherent Field Encoding: Signed values are encoded directly into optical field amplitude and π-phase shifts $( E _ { j } \propto x _ { j } , W _ { i j } \propto w _ { i j } )$ and recovered via homodyne mixing with an external phase reference [14].

## B. Stage-Resolved Physical Impairment Composition

Rather than applying an undifferentiated Gaussian perturbation to layer outputs, PISim injects each non-ideality at its exact physical stage along the forward signal chain [11], [12]:

$$
\begin{array} { r } { y _ { \mathrm { d i g } } = \mathcal { Q } _ { \mathrm { A D C } } \Big ( \mathcal { D } \Big ( \mathcal { H } _ { \mathrm { p r o p } } \big ( \mathcal { M } \big ( \mathcal { Q } _ { \mathrm { D A C } } ( x ) , \mathcal { Q } _ { \mathrm { c t r l } } ( W ) \big ) \big ) \Big ) + n _ { \mathrm { d e t } } \Big ) , } \end{array}\tag{7}
$$

where $\mathcal { Q } _ { \mathrm { D A C } }$ and $\mathcal { Q } _ { \mathrm { c t r l } }$ denote input and weight control quantization, $\mathcal { M } ( \cdot )$ applies nonlinear modulator transfer and finite extinction ratio, $\mathcal { H } _ { \mathrm { p r o p } } ( \cdot )$ applies optical loss, crosstalk, and phase/thermal drift, $\mathcal { D } ( \cdot )$ models square-law or homodyne photodetection, $n _ { \mathrm { d e t } }$ injects shot and thermal noise, and $\mathcal { Q } _ { \mathrm { A D C } }$ digitizes the electrical readout.

1) Multi-Stage Quantization: PISim separates checkpoint weight precision from $b _ { \mathrm { c t r l } } { - } \mathrm { b i t }$ DAC/modulator control quantization and $b _ { \mathrm { A D C } }$ -bit receiver digitization [15]. A normalized control signal $u \in [ u _ { \mathrm { m i n } } , u _ { \mathrm { m a x } } ]$ is discretized onto 2<sup>bctrl</sup> uniform steps:

$$
\mathcal { Q } _ { b } ( u ) = \Delta _ { b } \cdot \mathrm { c l i p } \left( \left\lfloor \frac { u - u _ { \operatorname* { m i n } } } { \Delta _ { b } } + \frac { 1 } { 2 } \right\rfloor , 0 , 2 ^ { b } - 1 \right) + u _ { \operatorname* { m i n } } ,\tag{8}
$$

where $\Delta _ { b } = ( u _ { \operatorname* { m a x } } - u _ { \operatorname* { m i n } } ) / ( 2 ^ { b } - 1 )$ . Applying $\mathcal { Q } _ { b _ { \mathrm { c u r l } } }$ prior to nonlinear device transfer (such as Eq. 1 or Eq. 2) accurately captures non-uniform optical transmission spacing caused by device nonlinearity [10], [15].

2) Correlated Source RIN and Signal-Dependent Shot Noise: For a source channel with nominal optical power $P _ { \mathrm { s r c } } ,$ relative-intensity noise is modeled as a stochastic multiplicative fluctuation $\delta _ { \mathrm { R I N } } ( t ) \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { R I N } } ^ { 2 } )$ . PISim distinguishes between physical RIN calibration and normalized sensitivity parameters: in physical device mode, $\sigma _ { \mathrm { R I N } } = \sqrt { \mathrm { R I N } \cdot B _ { \mathrm { e f f } } }$ is derived from the physical laser relative-intensity noise power spectral density (RIN in $\mathrm { H z ^ { - 1 } }$ , typically expressed in dB/Hz) integrated over the effective receiver bandwidth $B _ { \mathrm { e f f } } .$ whereas in normalized sensitivity mode, σ<sub>RIN</sub> is specified directly as a dimensionless fractional standard deviation in normalized simulator control units. Crucially, PISim preserves spatial correlation across source organizations [10]:

$$
P _ { \mathrm { c h } , k } ( t ) = \frac { \eta _ { \mathrm { s p l i t } } P _ { \mathrm { s r c } } } { N _ { \mathrm { f a n o u t } } } \left( 1 + \delta _ { \mathrm { R I N } , k } ( t ) \right) ,\tag{9}
$$

where $\delta _ { \mathrm { R I N , 1 } } ( t ) = \delta _ { \mathrm { R I N , 2 } } ( t ) = \cdot \cdot \cdot = \delta _ { \mathrm { R I N , } N } ( t )$ for a single laser with parallel fanout, whereas $\delta _ { \mathrm { R I N } , k } ( t )$ are drawn independently across $k \in \{ 1 , \ldots , N _ { \mathrm { v c s e l } } \}$ for a VCSEL array.

At the photodetector, incident optical power $P _ { \mathrm { d e t } , i }$ integrated over window $T _ { \mathrm { i n t } }$ (bandwidth $B \ = \ 1 / ( 2 T _ { \mathrm { i n t } } ) )$ ) experiences signal-dependent quantum shot noise:

$$
I _ { \mathrm { P D } , i } = \mathcal { R } P _ { \mathrm { d e t } , i } + n _ { \mathrm { s h o t } , i } + n _ { \mathrm { t h } , i } , \quad \sigma _ { \mathrm { s h o t } , i } ^ { 2 } = 2 q \mathcal { R } P _ { \mathrm { d e t } , i } B ,\tag{10}
$$

where $q$ is the elementary charge and $n _ { \mathrm { t h } , i }$ represents additive receiver thermal/TIA noise. In normalized sensitivity mode, shot perturbations scale proportionally to $\sigma _ { \mathrm { s h o t } } \sqrt { P _ { \mathrm { d e t } , i } }$

3) Optical Loss, Thermal Drift, and Channel Crosstalk: Total optical path attenuation $L _ { \mathrm { d B } } = L _ { \mathrm { s p l i t } } + L _ { \mathrm { m o d } } + L _ { \mathrm { p r o p } } + L _ { \mathrm { c o u p } }$ reduces detected power by $\eta _ { \mathrm { o p t } } = 1 0 ^ { - \dot { L } _ { \mathrm { d B } } / 1 0 }$ , directly degrading SNR in Eq. (10) [2], [13]. Thermal drift and static fabrication variations are separated by stochastic lifetime: static fabrication offsets $\Delta \phi _ { \mathrm { f a b } , i }$ are sampled once per simulated chip instance and held fixed across all dataset samples, whereas slow thermal drift $\Delta \theta _ { \mathrm { t h } } ( t ) \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { d r i f t } } ^ { 2 } )$ evolves across sample batches, perturbing resonator detuning $( \delta \lambda _ { 0 } \gets \delta \lambda _ { 0 } { + } \Delta \lambda _ { \mathrm { t h } } )$ or interferometer phase $( \phi  \phi + \Delta \phi _ { \mathrm { t h } } )$ [10], [14]. Finally, inter-channel crosstalk (from adjacent WDM spectral leakage or waveguide directional coupling) is applied via a sparse coupling matrix $\mathbf { C } \in \mathbb { R } ^ { N \times N } \ [ \bar { 2 } ] \mathrm { : }$

$$
\mathbf { P } _ { \mathrm { o u t } } = \mathbf { C } \mathbf { P } _ { \mathrm { i n } } , \quad C _ { i i } = 1 - \sum _ { j \neq i } \kappa _ { i j } , \quad C _ { i j } = \kappa _ { i j } \ ( i \neq j ) ,\tag{11}
$$

where $\kappa _ { i j }$ denotes the power (or field) coupling coefficient between channels i and j.

## IV. INTERFACE-AWARE TIMING AND DESIGN-SPACE EXPLORATION

## A. Cycle-Level Throughput and Bottleneck Accounting

A central pitfall in photonic accelerator evaluation is quoting peak optical propagation throughput $( f _ { \mathrm { o p t } } ~ = ~ c / ( n _ { \mathrm { e f f } } L _ { \mathrm { c o r e } } ) )$ while ignoring electronic conversion and memory feed bottlenecks [11], [13]. PISim implements an analytical schedule accounting model that decomposes end-to-end inference into five explicit stages for a matrix-vector tile of size $W \times H$ mapped onto an array of physical emitter width $N _ { w }$ and $N _ { \mathrm { t i l e s } }$ parallel tiles [13], [15]:

1) Weight Loading $( T _ { \mathbf { w t } } ) \colon$ Time required to program $W \times$ H weight elements across $N _ { \mathrm { p r o g } }$ programming lanes at update rate $f _ { \mathrm { w t } }$ , amortized over $N _ { \mathrm { r e u s e } }$ input vectors:

$$
T _ { \mathrm { w t } } = \frac { \lceil ( W \cdot H ) / N _ { \mathrm { p r o g } } \rceil } { N _ { \mathrm { r e u s e } } \cdot f _ { \mathrm { w t } } } .\tag{12}
$$

For stationary weights held across an entire batch $( N _ { \mathrm { r e u s e } } \gg 1 ) , T _ { \mathrm { w t } }  0 [ 1 2 ]$

2) Input Conversion & Modulation $( T _ { \mathbf { i n } } )$ : Time to stream N input activations through $N _ { \mathrm { D A C } }$ converters operating at sampling rate f<sub>DAC</sub> (accounting for $S _ { \mathrm { s e r } } ~ = ~ N / N _ { w }$ serialization folds):

$$
T _ { \mathrm { i n } } = \left\lceil \frac { N } { N _ { \mathrm { D A C } } } \right\rceil \frac { 1 } { f _ { \mathrm { D A C } } } .\tag{13}
$$

For a time-serialized single laser $( N _ { \mathrm { D A C } } = 1 , N _ { w } = 1 )$ $T _ { \mathrm { i n } }$ scales linearly with vector dimension $N$

3) Optical Propagation & Integration $( T _ { \mathbf { o p t } } )$ : Physical time-of-flight $T _ { \mathrm { t o f } }$ plus photodetector integration window $T _ { \mathrm { i n t } }$ required to meet target SNR:

$$
T _ { \mathrm { o p t } } = T _ { \mathrm { t o f } } + T _ { \mathrm { i n t } } .\tag{14}
$$

4) Output Digitization $\mathbf { ( } T _ { \mathbf { o u t } } ) \colon$ Time to convert M accumulated output channels across $N _ { \mathrm { A D C } }$ receivers at sampling rate f<sub>ADC</sub>:

$$
T _ { \mathrm { o u t } } = \left\lceil \frac { M \cdot r _ { \mathrm { r a i l } } } { N _ { \mathrm { A D C } } } \right\rceil \frac { 1 } { f _ { \mathrm { A D C } } } ,\tag{15}
$$

where $r _ { \mathrm { r a i l } } \ \in \ \{ 1 , 2 \}$ accounts for single-ended versus differential dual-rail readout.

5) Digital Post-Processing $( T _ { \mathbf { d i g } } ) { \ : \mathbf { : } }$ Partial-sum accumulation across tiles, bias subtraction, scaling, and nonlinear activation execution in the digital host/ASIC [15].

Under pipelined streaming execution, the amortized persample latency $T _ { \mathrm { s a m p l e } }$ and sustainable inference throughput $\Theta _ { \mathrm { i n f } }$ (in inferences/s) are bounded by the slowest stage:

$$
T _ { \mathrm { s a m p l e } } = \mathrm { m a x } \left( T _ { \mathrm { w t } } , T _ { \mathrm { i n } } , T _ { \mathrm { o p t } } , T _ { \mathrm { o u t } } , T _ { \mathrm { d i g } } \right) , \quad \Theta _ { \mathrm { i n f } } = \frac { 1 } { T _ { \mathrm { s a m p l e } } } ,\tag{16}
$$

whereas single-request unpipelined latency equals the serial sum $T _ { \mathrm { l a t } } = T _ { \mathrm { w t } } + T _ { \mathrm { i n } } + T _ { \mathrm { o p t } } + T _ { \mathrm { o u t } } + T _ { \mathrm { d i g } } .$ . Useful arithmetic throughput (in MAC/s or OP/s, with 1 $\mathsf { M A C } = 2 ~ \mathrm { O P s } )$ is computed strictly over active algorithmic operations $\mathbf { M A C } _ { \mathrm { u s e f u l } } \colon$

$$
\Theta _ { \mathrm { O P } } = \frac { 2 \cdot \mathrm { M A C _ { u s e f u l } } } { T _ { \mathrm { s a m p l e } } } .\tag{17}
$$

By inspecting which term dominates Eq. (16), PISim automatically flags whether a candidate design is DAC-bound, serialization-bound, integration/SNR-bound, or ADC-bound [11].

## B. Paired Reproducibility and Pareto Optimization

To improve statistical comparability across design sweeps, PISim partitions pseudo-random number generator (PRNG) streams by physical mechanism and stochastic lifetime:

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { S } _ { \mathrm { r u n } } = \Big ( S _ { \mathrm { f a b } } ( \mathrm { c h i p \_ i d } ) , ~ \mathcal { S } _ { \mathrm { d r i f t } } ( \mathrm { b a t c h \_ i d } ) , } \\ { ~ \mathcal { S } _ { \mathrm { f a s t } } ( \mathrm { s a m p l e \_ i d } , \mathrm { l a y e r \_ i d } ) \Big ) . } \end{array} } \end{array}\tag{18}
$$

Because every configuration evaluates identical input samples with matched stream partitioning, PISim computes both aggregate top-1 accuracy $\mathbf { A c c _ { p h o t } }$ and paired label agreement $\begin{array} { r } { \mathrm { A g r } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { I } ( \hat { y } _ { \mathrm { p h o t } } ^ { ( k ) } = \hat { y } _ { \mathrm { i d e a l } } ^ { ( k ) } ) } \end{array}$ , alongside paired confusion matrices that expose class-specific bias induced by extinction floors or drift [10].

For multi-objective DSE, let X denote the discretecontinuous configuration space spanning architecture family, modulator device, source organization, bit precision $b ,$ array width $N _ { w }$ , and tile count $N _ { \mathrm { t i l e s } } \ [ 1 1 ]$ , [12]. Given an application accuracy constraint $\operatorname { A c c } _ { \mathrm { p h o t } } ( \mathbf { x } ) \geq \operatorname { A c c } _ { \mathrm { m i n } }$ and a cost/latency objective vector $\begin{array} { r l r } { { \bf f } ( { \bf x } ) } & { { } \stackrel {  } { = } } & { \big ( T _ { \mathrm { s a m p l e } } ( { \bf x } ) , \ : N _ { \mathrm { d e t } } ( { \bf x } ) , C _ { \mathrm { p r o x y } } ( { \bf x } ) \big ) } \end{array}$ PISim extracts the Pareto-optimal frontier ${ \mathcal { P } } ^ { * }$ :

$$
\begin{array} { r l } & { \mathcal { P } ^ { * } = \Big \{ \mathbf { x } \in \mathcal { X } _ { \mathrm { f e a s } } \Big | \ l \mathbf { x } ^ { \prime } \in \mathcal { X } _ { \mathrm { f e a s } } \ \mathrm { s . t . } } \\ & { \qquad \quad \mathbf { f } ( \mathbf { x } ^ { \prime } ) \preceq \mathbf { f } ( \mathbf { x } ) \wedge \mathbf { f } ( \mathbf { x } ^ { \prime } ) \neq \mathbf { f } ( \mathbf { x } ) \Big \} . } \end{array}\tag{19}
$$

V. EXPERIMENTAL EVALUATION AND DSE CASE STUDIES

## A. Experimental Setup

Although PISim implements the broader model taxonomy described in Section II, the experiments reported here exercise a representative subset selected to isolate device-, source-, precision-, and interface-scaling effects. Specifically, we evaluate PISim using a pretrained 3-layer MNIST MLP (784 →

TABLE II  
DEFAULT SIMULATION AND EXPERIMENTAL CONFIGURATION
<table><tr><td>Parameter</td><td>Documented Value / Setting</td></tr><tr><td>Dataset</td><td>MNIST (test split, indices 0–999)</td></tr><tr><td>Sample Count</td><td>1,000 samples</td></tr><tr><td>Network Topology</td><td>3-layer  $\mathrm { M L P \ ( 7 8 4 \to 6 4 \to 1 0 ; 5 0 , 8 1 6 }$  weight bytes)</td></tr><tr><td>Random Seed</td><td>33 (deterministic PRNG)</td></tr><tr><td>Reference Accuracy</td><td>97.1% (ideal FP32)</td></tr><tr><td>Array Width  $( N _ { w } ) ^ { \ }$ </td><td>{32, 64, 128} channels (E5)</td></tr><tr><td>DAC / Control Precision</td><td>{4, 6, 8} bits (E2–E3); 8 bits (E4)</td></tr><tr><td>Optical Source</td><td>Single-laser parallel fanout (E1-E3); VCSEL, serialized (E4–E5)</td></tr><tr><td>Source Power  $( P _ { \mathrm { s r c } } )$ </td><td>1.0 (normalized, E2–E3)</td></tr><tr><td>Detection Mode</td><td>Incoherent direct detection (E2-E4)</td></tr><tr><td>Laser RIN (σRIN)</td><td>0 (E1–E2); 0.01 (normalized, E3–E4)</td></tr><tr><td>Shot Noise  $( \sigma _ { \mathrm { { s h o t } } } )$ </td><td>0 (E1–E2); 0.01 (normalized, E3–E4)</td></tr><tr><td>Thermal Drift  $( \sigma _ { \mathrm { d r i f t } } )$ </td><td>0 (E1–E2); 0.01 (normalized, E3–E4)</td></tr><tr><td>Extinction Ratio (ER)</td><td>30 dB  $( \epsilon = 1 0 ^ { - 3 } , \mathtt { E 3 - E 4 } )$ </td></tr><tr><td>Insertion Loss  $( L _ { \mathrm { i n s } } )$ </td><td>0 dB (E1–E2); 3 dB (E3–E4)</td></tr></table>

64 → 10, 50,816 weight bytes) across 1,000 test samples (indices 0–999) with deterministic PRNG seed 33. The ideal floating-point reference checkpoint achieves 97.1% classification accuracy on this evaluation split. Table II summarizes the documented experimental and physical simulation parameters across all case studies. Across five structured experiments (E1–E5), we investigate modulator equivalence under ideal baselines, precision scaling under multi-impairment stress, optical source organization trade-offs, and modeled arraywidth throughput scaling.

## B. E1: Clean-Reference Modulator Comparison

In Experiment E1, we compare three distinct electrooptic modulator technologies—Thin-Film $\mathrm { L i N b O _ { 3 } }$ MZM, Silicon Microring Resonator (Ring), and Surface-Normal QCSE Micro-Pad—under clean operating conditions (all stochastic noise, drift, loss, and crosstalk controls disabled) using singlelaser parallel fanout. All three modulator architectures achieve 97.1% photonic inference accuracy, matching the 97.1% ideal floating-point reference with zero degradation. This confirms that when calibrated to their linear operating regimes without physical impairments, distinct modulator transfer functions preserve full network fidelity.

## C. E2–E3: Precision Sweeps Under Clean and Multi-Impairment Stress

To quantify how control bit precision interacts with hardware non-idealities, Experiments E2 and E3 sweep quant\_ $\mathsf { b i t s } \in \{ 4 , 6 , 8 \}$ across MZM, Ring, and QCSE configurations using single-laser parallel fanout, incoherent detection, and normalized source power $P _ { \mathrm { s r c } } = 1 . 0 $

• E2 (Clean Quantization Baseline): All five physical impairment controls are disabled.

• E3 (Combined Multi-Impairment Stress): Five simultaneous non-idealities are enabled in normalized sensitivity mode: shot noise $( \sigma _ { \mathrm { s h o t } } ~ = ~ 0 . 0 1 )$ , laser RIN $( \sigma _ { \mathrm { R I N } } = 0 . 0 1 )$ , thermal drift $( \sigma _ { \mathrm { d r i f t } } ~ = ~ 0 . 0 1 )$ , finite extinction ratio $( \mathrm { E R } = 3 0 ~ \mathrm { d B } , ~ \epsilon = 1 0 ^ { - 3 } )$ , and optical insertion loss $( L _ { \mathrm { i n s } } = 3 ~ \mathrm { d B ) }$ . Here, normalized parameters such as $\sigma _ { \mathrm { R I N } } ~ = ~ 0 . 0 1$ (as well as $\sigma _ { \mathrm { s h o t } } ~ = ~ 0 . 0 1$ and $\sigma _ { \mathrm { d r i f t } } ~ = ~ 0 . 0 1 )$ are controlled, dimensionless simulation sensitivity parameters and should not be interpreted as calibrated physical measurements from a specific laser or photonic device. Crosstalk, nonlinear optical effects, speckle, and phase noise remain disabled.

TABLE III  
E2–E3 PER-BIT PHOTONIC INFERENCE ACCURACY (%) ON MNIST MLP (IDEAL REFERENCE = 97.1%)
<table><tr><td>Precision (Bits)</td><td>E2 Clean (All Three)</td><td>E3 Stress MZM</td><td>E3 Stress Ring</td><td>E3 Stress QCSE</td></tr><tr><td>4 bits</td><td>95.9%</td><td>35.7%</td><td>35.7%</td><td>24.6%</td></tr><tr><td>6 bits</td><td>97.0%</td><td>46.4%</td><td>46.4%</td><td>15.9%</td></tr><tr><td>8 bits</td><td>97.1%</td><td>44.5%</td><td>44.5%</td><td>19.0%</td></tr></table>

Table III summarizes the numerical accuracy across precision levels, revealing two critical EDA insights:

1) Clean Precision Saturation (E2): Under impairmentfree conditions, all three modulators exhibit identical monotonic accuracy recovery: 95.9% at 4 bits (−1.2% drop), 97.0% at 6 bits (−0.1% drop), and full recovery to 97.1% at 8 bits. Thus, 6-bit control precision suffices for nearly lossless representation in the absence of analog noise.

2) Non-Monotonic Precision Behavior Under Stress (E3): Under heavy combined noise $( \sigma = 0 . 0 1 )$ and 30 dB extinction ratio, accuracy degrades severely across all modulators, and scaling to higher bit-widths no longer guarantees monotonic improvement [10]. Both MZM and Ring peak at 46.4% at 6 bits before dropping slightly to 44.5% at 8 bits, as finer quantization steps fall below the analog noise floor $( \Delta _ { 8 } \approx 0 . 0 0 3 9 < \sigma _ { \mathrm { n o i s e } } )$ and capture stochastic perturbations without improving signal fidelity. QCSE exhibits substantially higher sensitivity under identical stress settings, peaking at 24.6% at 4 bits and dropping to 15.9% at 6 bits and 19.0% at 8 bits due to its distinct transfer sensitivity.

## D. E4: Impact of Optical Source Organization Under Stress

Experiment E4 isolates the architectural impact of optical source organization by holding the workload (MNIST MLP), modulator (MZM), detection mode (incoherent), precision (8 bits), and combined E3 stress profile $( \sigma _ { \mathrm { s h o t } } = \sigma _ { \mathrm { R I N } } = \sigma _ { \mathrm { d r i f t } } =$ 0.01, ER = 30 dB, $L _ { \mathrm { i n s } } \ = \ 3 \ \mathrm { d B } )$ completely fixed while varying only the optical source topology.

As shown in Fig. 2, source organization produces dramatic accuracy divergence despite identical device parameters:

• VCSEL Array (97.0%): Achieves near-ideal accuracy (within 0.1% of the 97.1% reference) even under full E3 stress. Because each channel is driven by a dedicated emitter without passive $1 / N$ optical splitting loss and without spatially correlated common-mode RIN across the entire layer vector, per-channel SNR remains well above the noise floor.

![](images/6fae66618715f1588898dd1b1e0602d1f12985e9560d58e76967dd6c591f505a.jpg)  
Fig. 2. Photonic inference accuracy across optical source organizations under matched 8-bit MZM multi-impairment stress (E4). Eliminating $1 / N$ fanout splitting loss and spatially correlated RIN restores accuracy from 44.5% to 97.0%.

TABLE IV  
E5 MODELED INFERENCE THROUGHPUT (MILLION INFERENCES/S) AND PEAK AMORTIZED LATENCY ACROSS ARRAY WIDTHS
<table><tr><td>Array Width (Nw)</td><td>VCSEL / MZM</td><td>Parallel Laser / MZM</td><td>Serialized Laser / MZM</td></tr><tr><td>32</td><td>5.563 M/s</td><td>5.563 M/s</td><td>12.399 M/s</td></tr><tr><td>64</td><td>10.710 M/s</td><td>10.710 M/s</td><td>12.496 M/s</td></tr><tr><td>128</td><td>19.584 M/s</td><td>19.584 M/s</td><td>12.545 M/s</td></tr><tr><td>Min Latency</td><td>51.06 ns</td><td>51.06 ns</td><td>79.71 ns</td></tr></table>

• Single-Laser Time-Serialized (77.0%): Outperforms parallel fanout by +32.5% because full master laser power $P _ { 0 }$ is delivered to a single active channel per time slot without 1/N power division, mitigating shot-noise degradation while still experiencing temporal RIN and drift across sequential steps.

• Single-Laser Parallel Fanout (44.5%): Suffers an accuracy penalty of 52.5 percentage points relative to the VCSEL array arising, under the modeled simulation conditions, from the combined effect of 1/N optical power partitioning (reducing detected signal power relative to shot noise and extinction floor) and spatially correlated RIN perturbing all vector elements coherently.

This finding underscores a core cross-layer EDA trade-off: parallel fanout from a single laser simplifies emitter packaging but requires substantially higher launch power or differential noise cancellation to match VCSEL-array robustness.

## E. E5: Array-Width Throughput and Latency Scaling

Experiment E5 evaluates analytically modeled interfacelimited system throughput across emitter-array widths $N _ { w } \in$ {32, 64, 128} for the MNIST MLP (50,816 bytes weight memory) across VCSEL/MZM, single-laser parallel/MZM, and single-laser serialized/MZM configurations. Table IV reports modeled inference throughput (in million inferences/s) and amortized per-sample latency at peak throughput.

The modeled scaling curves in Table IV expose a clear architectural crossover governed by electronic-interface scheduling:

• Spatial Parallel Scaling (VCSEL & Parallel Laser): Both parallel architectures exhibit identical modeled throughput, scaling nearly linearly with array width in the analytical schedule model from 5.563 M inferences/s at $N _ { w } = 3 2$ to 10.710 M/s at $N _ { w } = 6 4 ~ ( 1 . 9 3 { \times } )$ and 19.584 M/s at $N _ { w } = 1 2 8 \ ( 3 . 5 2 \times )$ , reaching a modeled amortized per-sample latency of 51.06 ns. Increasing $N _ { w }$ reduces the number of sequential matrix-vector folding cycles required to process the 784-element input layer.

• Time-Serialized Saturation: The serialized single-laser configuration maintains nearly flat modeled throughput across array widths (12.399 M/s at $N _ { w } \ = \ 3 2$ to 12.545 M/s at $N _ { w } = 1 2 8$ , with a modeled peak amortized latency of 79.71 ns) because simulated execution is bottlenecked by single-channel sequential modulation/digitization rather than spatial emitter width.

• Crossover and Joint Accuracy-Throughput Selection: At narrow array widths $( N _ { w } \in \{ 3 2 , 6 4 \} )$ ), the high-speed serialized laser achieves 1.17× to 2.23× higher modeled throughput than spatial arrays. However, at $N _ { w } = 1 2 8 .$ parallel arrays surpass the serialized bottleneck in the timing model by 1.56×. Combining E4 and E5 reveals that while Parallel Laser/MZM and VCSEL/MZM achieve identical modeled peak throughput (19.584 M/s at $N _ { w } = 1 2 8 )$ , only VCSEL/MZM simultaneously satisfies high accuracy constraints (97.0% vs. 44.5% under E3 stress), positioning it on the simulated accuracyconstrained Pareto frontier.

## VI. DISCUSSION AND ROADMAP FOR HARDWARE VALIDATION

The exploratory evaluations in Section V demonstrate how PISim exposes non-obvious interactions between device physics, noise correlation, and interface scheduling prior to fabrication. Follow-on multi-dimensional perturbation studies have already extended these device models to deeper CIFAR-10 convolutional workloads (a compact four-layer CNN and ResNet-8), mapping two-dimensional RIN-versusdrift accuracy response surfaces for Mach–Zehnder and microring matrix multipliers [10]. To transition from normalized pre-fabrication screening to silicon-calibrated sign-off, three extensions are prioritized:

1) Physical Parameter Calibration and Multi-Seed Confidence: Replacing normalized simulator sigma controls with foundry-calibrated PDK distributions (linking σ<sub>RIN</sub>, $\tau _ { R C } , \beta _ { \mathrm { s c r } }$ , and thermal drift directly to physical dB/Hz and Kelvin metrics) [11] and running multi-seed Monte Carlo ensembles to establish statistical confidence intervals on Pareto frontiers.

2) Joint Accuracy-Throughput-Energy Sweeps (E6): Extending E5 to evaluate matched paired accuracy at every array width $N _ { w }$ and tile count $N _ { \mathrm { t i l e s } }$ across multicore scale-out topologies [12], [13], while incorporating a validated system-level energy accounting boundary covering laser wall-plug efficiency, modulator drivers,

thermal heaters, TIAs, and DAC/ADC conversion energy [11], [15].

3) Experimental Silicon Correlation: Benchmarking PISim forward predictions against fabricated photonic test chips across deeper vision and sequence workloads (ResNet-8, VGG-8, and Mixer-Tiny) [10], [12].

## VII. CONCLUSION

We presented the Photonic Inference Simulator (PISim), a unified electronic design automation framework for prefabrication design-space exploration of photonic neural accelerators. By integrating six optical architecture families, six parameterized electro-optic modulator transfer models, four source organizations, and stage-resolved physical impairment operators with an analytical interface timing model, PISim enables architects to evaluate end-to-end neural network accuracy and system bottlenecks on equal footing. Case studies on MNIST inference reveal critical cross-layer design rules: precision scaling under multi-impairment stress exhibits nonmonotonic optima where finer quantization falls below analog noise floors; VCSEL arrays preserve 97.0% accuracy under stress profiles that degrade single-laser parallel fanout to 44.5%; and, under the analytical interface timing model, spatial parallel arrays overtake time-serialized lasers in throughput at width $N _ { w } = 1 2 8$ (19.58 M inferences/s, 51.06 ns modeled latency). PISim provides a rigorous, reproducible foundation for accuracy-constrained photonic accelerator co-design.

## REFERENCES

[1] J. Gu, H. Zhu, C. Feng, Z. Jiang, R. Chen, and D. Pan, “L2ight: Enabling On-Chip Learning for Optical Neural Networks via Efficient in-situ Subspace Optimization,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021.

[2] A. Shafiee, S. Banerjee, K. Chakrabarty, S. Pasricha, and M. Nikdast, “LoCI: An Analysis of the Impact of Optical Loss and Crosstalk Noise in Integrated Silicon-Photonic Neural Networks,” in Proc. Great Lakes Symposium on VLSI (GLSVLSI), 2022, pp. 343–348.

[3] H. Zhou et al., “Photonic matrix multiplication lights up photonic accelerator and beyond,” Light: Science & Applications, vol. 11, art. 30, 2022.

[4] Y. Shen et al., “Deep learning with coherent nanophotonic circuits,” Nature Photonics, vol. 11, no. 7, pp. 441–446, 2017.

[5] J. Feldmann et al., “Parallel convolutional processing using an integrated photonic tensor core,” Nature, vol. 589, no. 7840, pp. 52–58, 2021.

[6] X. Lin et al., “All-optical machine learning using diffractive deep neural networks,” Science, vol. 361, no. 6406, pp. 1004–1008, 2018.

[7] W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, and I. A. Walmsley, “Optimal design for universal multiport interferometers,” Optica, vol. 3, no. 12, pp. 1460–1465, 2016.

[8] A. N. Tait et al., “Neuromorphic photonic networks using silicon photonic weight banks,” Scientific Reports, vol. 7, art. 7430, 2017.

[9] G. Wetzstein et al., “Inference in artificial intelligence with deep optics and photonics,” Nature, vol. 588, no. 7836, pp. 39–47, 2020.

[10] A. Falkenberg, “Comparative Robustness of Mach-Zehnder and Microring Photonic Matrix Multipliers Under Laser RIN and Thermal Drift,” arXiv preprint arXiv:2610.15400, 2026.

[11] H. Zhu, S. Ning, H. Zhou, Z. Yin, R. T. Chen, J. Gu, and D. Z. Pan, “Harnessing Photonics for Machine Intelligence,” arXiv preprint arXiv:2604.10841, 2026.

[12] Z. Yin, H. Zhou, N. Gangi, M. Zhang, J. Zhang, Z. R. Huang, and J. Gu, “Toward Large-Scale Photonics-Empowered AI Systems: From Physical Design Automation to System-Algorithm Co-Exploration,” arXiv preprint arXiv:2601.00129, 2026.

[13] B. Jahannia, A. Amirany, and H. Dalir, “Towards Topology-Aware Very Large-Scale Photonic AI Accelerators,” arXiv preprint arXiv:2604.26966, 2026.

[14] S. A. Fldzhyan, S. S. Straupe, and M. Yu. Saygin, “Perturbative photonic matrix-vector multiplication with reduced phase-shift range,” arXiv preprint arXiv:2606.02451, 2026.

[15] H. Gong, H. Zhang, A. S. Barnard, M. Hassan, M. Woolley, and R. Buyya, “LightMat-HP: A Photonic-Electronic System for Accelerating General Matrix Multiplication With Configurable Precision,” arXiv preprint arXiv:2604.12278, 2026.