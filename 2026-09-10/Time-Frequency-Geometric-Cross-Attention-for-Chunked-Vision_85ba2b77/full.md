# Time–Frequency Geometric Cross-Attention for Chunked Vision–Language–Action Models

Shengye Dong, Haochen Niu, Hao Liu, Peiwen Lin, Chuang Wang, Shanmin Pang

## Abstract

Modern vision–language–action (VLA) policies predict a whole chunk of actions: one to two seconds of coordinated motion emitted in a single forward pass. Yet an action chunk is essentially a short multivariate trajectory, and inside these models it is represented as a sequence of generic per-timestep hidden tokens and decoded by a linear head. This representation tends to under-serve two structures of motion. The first is frequency: a chunk superimposes a smooth global trend and fine corrective motion across several time scales, and a single token entangles them. The second is cross-phase geometric relationships: within a chunk, the motions of different phases (reach, contact, grasp adjustment, settling) unfold along very diferent directions—near-orthogonal in the representation space, yet tightly related for the task, and arising across the time axis (between phases) rather than at the same instant. Dot-product attention scores alignment by an inner product, so it favors similar (aligned) tokens and is least sensitive exactly near orthogonality, leaving such cross-phase relationships to be recovered by the network through a detour. We introduce Time–Frequency Geometric Cross-Attention (TFGCA), a drop-in module that repairs both blind spots. TFGCA uses a per-dimension learnable stationary wavelet transform (SWT) to decompose the action chunk into time– frequency tokens, and each time token then retrieves information from them through a cross-attention that fuses the dot product (similarity) with the wedge-product magnitude (sensitive to near-orthogonality) via a learnable weight. The module uses a zero-initialized residual, so it reproduces the base behavior exactly at initialization and can be dropped onto a pretrained VLA and fine-tuned jointly with it. Relative to the same-source base, TFGCA improves the near-saturated in-distribution LIBERO by +1.5 on average, the OOD bench mark LIBERO-Plus by +6.3, and the randomized average under RoboTwin domain randomization by +28.5, and raises the overall success rate on three real-robot AgiBot A2 tasks by +11.67 percentage points—with the gains larger under out-of-distribution conditions.

## 1 Introduction

Vision–language–action models have converged on a common output interface: rather than emitting one action per step, they predict a chunk of T future actions at once and execute some prefix of it before re-planning. Flow-matching and diffusion policies such as π<sub>0</sub> (Black et al. 2024), π<sub>0.5</sub> (Physical Intelligence et al. 2025), RDT-1B (Liu et al. 2025), and the action-chunking transformer ACT (Zhao et al. 2023) all commit to a T × D trajectory in a single forward pass, where T is the chunk horizon and D the number of action dimensions. Chunked prediction improves temporal consistency and lets a policy plan short-horizon coordination instead of reacting one step at a time.

The chunk is therefore a structured object: a short multivariate trajectory whose value lies in how its dimensions move together over time. Current VLAs, however, treat it as an unstructured one. The action tokens leaving the transformer are generic hidden vectors, and a single linear head maps each to an action. Two properties of motion that a trajectory model should exploit are left implicit.

Frequency structure. A manipulation chunk overlays motion at several time scales. A slow transport component carries the end-efector across the workspace, while faster components perform contact-time corrections, grasp adjustments, and settling. Packed into one hidden token per timestep, these scales are entangled, and the model has no explicit handle on “the trend” versus “the correction.”

Cross-phase geometric relationships. Skilled manipulation unfolds over time into several phases—reach, contact, grasp adjustment, settling—that are related for the task yet can act along very diferent directions in the representation space, near-orthogonal in the limit. Where does this near-orthogonal “division of labor” occur? Our analysis of RoboTwin 2.0 bimanual data shows that it hardly appears at the same instant, but unfolds sequentially along the time axis—trajectories successively occupy a set of nearorthogonal subspaces in ordered temporal phases, which we call Temporal Orthogonal Division-of-Labor (TO-DoL) (metrics and robustness in Appendix H). Dot-product attention, however, measures only alignment and is least sensitive exactly near orthogonality, so it must spend extra capacity to re-encode such cross-phase relationships as similarity. This motivates an inductive hypothesis: giving attention an extra scoring channel sensitive to near-orthogonal components (the wedge product) may let each time token retrieve such cross-phase relationships more directly from the whole chunk’s time–frequency evidence (Section 4).

Frequency structure and geometric attention have precedents in time-series forecasting (e.g., compact forecasters that pair a wavelet transform with a geometric attention).

The VLA setting is diferent in kind: an action chunk is a short, non-periodic trajectory over actual control degrees of freedom, and any added module rides on a pretrained multi-billion-parameter backbone whose learned weights it must not disturb at initialization. We therefore design a time– frequency geometric attention module, TFGCA, purposebuilt for VLA action prediction; its design parts and contributions follow.

Contributions. We introduce TFGCA, a time–frequency geometric cross-attention module for VLA action prediction that supplies chunked policies with two motion structures they previously ignored: multi-scale frequency content, and cross-phase geometric relationships within the chunk that fall along near-orthogonal directions. Its two core parts are as follows (full method in Section 4).

• (C1, frequency) Per-control-dimension action-space wavelet tokenization. We project the hidden action tokens to action space and apply a per-dimension learnable stationary wavelet transform (SWT), turning each control dimension’s trajectory into length-preserving, multiresolution frequency tokens—supplying the multi-scale frequency structure a base VLA leaves implicit.

• (C2, geometry) A time-to-time–frequency geometric cross-attention. Each time token retrieves information from the whole chunk’s time–frequency tokens through a learnable blend of the dot product (similarity) and the wedge product. Proposition 1 shows the wedge is sensitive to near-orthogonal directions in the representation space and can realize orderings the dot channel alone cannot; from this we propose a geometry-sensitive inductive hypothesis—that cross-phase, cross-time-scale correlations within a chunk often fall along dot-insensitive near-orthogonal directions, for which the wedge channel can provide a direct score—whose real-task payof is answered empirically in Section 5.

## 2 Related Work

Vision–language–action models and action chunking. VLAs map instructions and observations to robot actions by adapting large pretrained vision–language backbones. Discrete-token approaches such as RT-2 (Brohan et al. 2023) and OpenVLA (Kim et al. 2024) autoregress action tokens, while Octo (Octo Model Team 2024), difusion policy (Chi et al. 2023), π (Black et al. 2024), π (Physical Intelligence et al. 2025), and RDT-1B (Liu et al. 2025) predict continuous action chunks with difusion or flow matching (Lipman et al. 2023). Action chunking, introduced for imitation with ACT (Zhao et al. 2023), predicts a block of future actions and executes a prefix, improving temporal coherence, often with temporal ensembling across overlapping chunks to reduce discontinuities. TFGCA operates inside this chunkedprediction interface: it is a representation of the chunk placed between the backbone and the action head, orthogonal to how the chunk is supervised (flow matching, difusion, or regression).

Frequency-domain methods for sequences and actions. Frequency-domain inductive biases are well studied in forecasting: Autoformer (Wu et al. 2021) and FEDformer (Zhou et al. 2022) inject seasonal–trend decomposition and Fourier attention, FreTS (Yi et al. 2023) learns in the frequency domain, and SimpleTM (Chen et al. 2025a) couples a stationary wavelet transform with a geometric attention. Within VLAs, FAST (Pertsch et al. 2025) applies a DCT to action chunks to produce compact discrete tokens for autoregressive policies, using frequency for tokenization and compression. Our use is diferent: we treat frequency as a multi-resolution, time-localized feature that attention consults, on a continuous flow-matching policy; the stationary (non-decimating) wavelet preserves sequence length, so every band stays aligned with the original timesteps and can serve directly as attention keys.

Geometric structure in attention. Geometric-algebra networks (the Geometric Algebra Transformer (Brehmer et al. 2023); Cliford neural layers (Brandstetter et al. 2023)) represent data as multivectors and act on them equivariantly. We borrow only a single scalar—the wedge-product magnitude—and use it as an attention score alongside the inner product, keeping the module lightweight and free of multivector bookkeeping while capturing the one property we need: sensitivity to orthogonality. Unlike iTransformer (Liu et al. 2024), which treats variables (dimensions) as attention tokens, our frequency tokens jointly encode the multidimensional action at each time–band position, and the crossattention retrieves along time–frequency rather than attending between dimensions.

## 3 Preliminaries

VLA action prediction as trajectory forecasting. We build on a flow-matching VLA $( \pi _ { 0 . 5 } )$ . Given observations (images and a language instruction) and proprioceptive state, the policy predicts an action chunk $\boldsymbol { a } \in \mathbb { R } ^ { \dot { T } \times D }$ , where $T$ is the chunk horizon and D the number of action dimensions. Flow matching trains a velocity field: sample noise ϵ and a time $\tau \in [ 0 , 1 \big ]$ , form the interpolant $x _ { \tau } = \tau \epsilon + \left( 1 - \tau \right) a ,$ and regress the network output $v _ { \theta } ( x _ { \tau } , \cdot )$ toward the target velocity $u = \epsilon - a$ with the loss

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { \tau , \epsilon } \left. v _ { \theta } ( x _ { \tau } , \cdot ) - ( \epsilon - a ) \right. ^ { 2 } .\tag{1}
$$

The backbone is a decoder transformer that jointly attends over a multimodal prefix (image and language tokens) and an action sufix; the last $\dot { T }$ sufix positions produce hidden action tokens $H \in \mathbb { R } ^ { T \times d }$ (with d the model width), which a linear action head maps to $v _ { \theta }$ . TFGCA is a transformation $H \mapsto { \tilde { H } }$ inserted just before this head, applied identically in training and in each denoising step at inference.

Dot-product attention prefers similarity. For queries $Q$ and keys K split into heads of dimension $d _ { h }$ , attention weights come from $q _ { i } ^ { \top } k _ { j } / \sqrt { d _ { h } } .$ , largest when $q _ { i }$ and $k _ { j }$ are aligned and vanishing when orthogonal, so in a given representation two orthogonal tokens receive only a small direct score. A deep network can of course relearn features that reencode orthogonal relationships as aligned and route them through the similarity channel (which is why existing dotproduct policies work); but that spends representational capacity, and an explicit orthogonality-sensitive signal removes the detour (formalized in the scope remark of Proposition 1).

![](images/a37ee921752845da8ce24c0d6d8066a67ef1c8f60cb7c3f5a058f6ab04eff926.jpg)  
Figure 1: TFGCA architecture: a drop-in module between the VLA transformer output and the action head. The frequency branch (action-space projection → per-dimension SWT → frequency tokens) supplies the keys/values for the geometric cross-attention, whose queries are the time tokens; its output is written back residually and passed through the action head to give the velocity ffi<sub>v . (Blue: data tensors; amber: new modules; gray: backbone.)</sub>

The wedge product measures orthogonality. The wedgeproduct magnitude is the area of the parallelogram q, k span, with a closed form via the Cauchy–Schwarz identity (no antisymmetric-tensor construction):

$$
\| q \land k \| ^ { 2 } = \| q \| ^ { 2 } \| k \| ^ { 2 } - ( q ^ { \top } k ) ^ { 2 } .\tag{2}
$$

It is geometrically complementary to the inner product: zero when $q \parallel k$ , maximal (for fixed norms) when $q \perp k$ . Using both scores gives attention access to alignment and orthogonality at once.

## 4 Method

TFGCA transforms the hidden action tokens $H \in \mathbb { R } ^ { T \times d }$ into refined tokens $\tilde { H } \in \mathbb { R } ^ { T \times d }$ in three stages (overall structure in Figure 1): a projection to a per-dimension control space (Section 4.1), a per-dimension wavelet tokenization (Section 4.2), and a geometric cross-attention that fuses the two (Section 4.3). Section 4.4 covers the identity-at-initialization ⋯construction that makes the module safe to attach to a pretrained policy, and Section 4.5 gives the wedge’s theoretical properties and the mechanism explanation.

## 4.1 Action-Space Projection

The transformer’s action tokens live in an abstract $d -$ dimensional space with no explicit per-control-dimension meaning, whereas the structure we want to model—the frequency content of each action dimension and their coordination—is defined in the action coordinates the policy actually outputs. We therefore first project each token to a per-dimension control-space view,

$$
Z ^ { \mathrm { a c t } } = \mathrm { A c t i o n P r o j } ( H ) \in \mathbb { R } ^ { T \times D } ,\tag{3}
$$

with a dedicated linear map. The physical meaning of the D output channels depends on the embodiment’s action parameterization: in RoboTwin 2.0 and AgiBot A2 they correspond to joint-space actions; in LIBERO, the 7 channels correspond to Cartesian end-efector increments (3-D translation, 3-D rotation) and a 1-D binary gripper command, not 7 joint angles. The projection is initialized so that $Z ^ { \mathrm { a c t } }$ is nonzero from the first step, which matters for gradient flow (Section 4.4). To give this projection explicit physical grounding, we add an auxiliary alignment loss that supervises $Z ^ { \mathrm { a c t } }$ toward the stop-gradient ground-truth action chunk $a ,$

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \big \| Z ^ { \mathrm { a c t } } - \mathrm { s g } ( a ) \big \| ^ { 2 } ,\tag{4}
$$

where $\operatorname { s g } ( \cdot )$ is the stop-gradient (only the projection is updated, no gradient flows to the target); training minimizes $\mathcal { L } = \mathcal { L } _ { \mathrm { F M } } \bar { + } \lambda \mathcal { L } _ { \mathrm { a l i g n } }$ for a small weight λ (per-benchmark values in Appendix B). This loss is part of the full model; removing it is the “w/o alignment” ablation (Section 5.1).

## 4.2 Per-Dimension Learnable SWT Tokenization

Given the control-space view $Z ^ { \mathrm { a c t } }$ , we decompose each action dimension’s length-T sequence into multiple frequency bands with a stationary wavelet transform (SWT; Figure 2). Unlike the standard (decimated) wavelet transform, the SWT does not downsample: every band has length T, so bands stay aligned with the original timesteps and can serve as attention keys. One level splits an input approximation $a ^ { ( j ) }$ into a coarser approximation and a detail band using a lowpass/high-pass filter pair $( h , g )$ applied at dilation $2 ^ { \bar { j } }$

$$
a _ { t } ^ { ( j + 1 ) } = \sum _ { k } { h _ { k } a _ { t + 2 ^ { j } k } ^ { ( j ) } } , \quad d _ { t } ^ { ( j + 1 ) } = \sum _ { k } { g _ { k } a _ { t + 2 ^ { j } k } ^ { ( j ) } } ,\tag{5}
$$

recursing on the approximation branch. With J levels this yields J detail bands and one final approximation, $\{ d ^ { ( 1 ) } , \dots , d ^ { ( J ) } , a ^ { ( J ) } \}$ , ordered high to low frequency. We make three design choices specific to robot actions.

Per-dimension learnable filters. Each action dimension gets its own filter pair, initialized to Daubechies-2 (db2, length 4; Haar is an option). db2’s two vanishing moments make its detail coeficients cancel locally linear motion and respond only to curvature, suiting the long, smooth segments common in manipulation; filters are learnable, so each dimension specializes a task-relevant decomposition on top of this strong prior.

Per-dimension DC removal. A smooth demonstration’s energy sits mostly at zero frequency (a near-constant ofset) and would swamp the detail bands, so we subtract each dimension’s mean over time before decomposition. We deliberately do not divide by the standard deviation: the amplitude contrast between high-motion moments (contact, grasp) and stationary segments is exactly the signal cross-attention should key on, and normalizing it away would amplify noise in still segments.

Optional derivative pre-processing. Because action sequences can be low-frequency-dominated even after DC removal, the view may be finite-diferenced before the SWT to move task-relevant energy into higher bands (the order used per benchmark is in Appendix B).

At each (time, band) position, a per-band linear map embeds the whole D-dimensional action vector at that position into the model width d $( \mathbb { R } ^ { D }  \mathbb { R } ^ { d } )$ ), plus a band-type embedding and a time-position embedding. This step jointly encodes the multiple action dimensions at the same time– band position into a single token—cross-dimension structure is carried as token content rather than paired dimension-bydimension by the subsequent attention. Stacking the $J { \ + } 1$ bands over $\dot { T }$ timesteps and flattening (band index fastest) gives a sequence of frequency tokens

$$
F \in \mathbb { R } ^ { T ( J + 1 ) \times d } .\tag{6}
$$

## 4.3 Dot-Plus-Wedge Cross-Attention

TFGCA’s attention is a cross-attention: queries are the time tokens H, and keys/values are the frequency tokens $F .$ . Each time token thus retrieves evidence from the whole chunk’s time–frequency tokens (each token jointly encoding multidimensional action content at one time–band position). With multi-head projections $Q = W _ { Q } H , K = W \dot { \bar { \kappa } } F , V = W _ { V } F$ (head dimension $d _ { h } .$ , scale $s = 1 / \sqrt { d _ { h } } )$ , we form two score matrices,

$$
S _ { i j } ^ { \mathrm { d o t } } = s ( q _ { i } ^ { \top } k _ { j } ) ,\tag{7}
$$

$$
S _ { i j } ^ { \mathrm { w e d } } = s \sqrt { \| q _ { i } \| ^ { 2 } \| k _ { j } \| ^ { 2 } - ( q _ { i } ^ { \top } k _ { j } ) ^ { 2 } } .\tag{8}
$$

![](images/74eae5926a83cf3e7e830673e13a38cbbac132075f4fd33f090dad89117be9ee.jpg)  
Figure 2: A per-dimension SWT (db2) decomposes a single action-dimension sequence (top, DC-removed per dimension) into multi-resolution bands. High-frequency jitter at the contact moment (shaded) concentrates in $d ^ { ( 1 ) }$ , while the final approximation $\grave { a } ^ { ( 2 ) }$ retains the smooth trend—trend and correction are decoupled into diferent bands.

We convert each score matrix to attention weights with its own softmax and blend them with a learnable scalar. Let $\beta = \sigma ( \ell )$ be a global learnable mixing weight (a single scalar logit ℓ, initialized so $\beta = 0 . 5 )$ . Then

$$
A = ( 1 - \beta ) \operatorname { s o f t m a x } ( S ^ { \operatorname { d o t } } ) + \beta \operatorname { s o f t m a x } ( S ^ { \operatorname { w e d } } ) .\tag{9}
$$

This two-softmax-then-blend form (rather than one softmax over a pre-summed score) keeps each channel a proper distribution and lets the model weight alignment (dot) against orthogonality (wedge) with the learnable $\beta .$ The attended value is projected and added residually,

$$
\tilde { H } = H + W _ { O } ( A V ) .\tag{10}
$$

## 4.4 Identity at Initialization and Integration

The output projection $W _ { O }$ is zero-initialized, so at initialization ${ \tilde { H } } = H$ and the module reproduces the base prediction exactly on attachment. This is the standard zero-initialized residual paradigm (Zhang, Rao, and Agrawala 2023; Alayrac et al. 2022; Hu et al. 2022; Bachlechner et al. 2021). So that the module can leave the identity map, we zero only $W _ { O }$ and keep the value path (band embeddings and action-space projection) nonzero: the initial gradient $\nabla _ { W _ { O } } { \mathcal { L } } = \delta { \dot { \left( A V \right) } } ^ { \top }$ is then generically nonzero, whereas zeroing $V$ too would trap the module at the identity (formal statement in Appendix I). This preserves function at initialization only—it eases dropin but does not guarantee the fine-tuned model beats the base (Section 5.3 and Section 6). The added parameters are negligible (about 4.23M, roughly 0.10% of the $\pi _ { 0 . 5 }$ backbone; breakdown in Appendix G.2).

## 4.5 Theoretical Properties and Mechanism

We record the property that formalizes the wedge channel;   
its proof is short and follows from the definitions.

Proposition 1 (Alignment–orthogonality decomposition and order separation). For any nonzero $q , \dot { k } \in \mathbb { R } ^ { d _ { h } } , ( q ^ { \top } k ) ^ { 2 } +$ $\| q \land k \| ^ { 2 } = \| q \| ^ { 2 } \| k \| ^ { 2 } ,$ ; after norm normalization, $\begin{array} { r l } { \hat { s } } & { { } = } \end{array}$ $q ^ { \top } k / ( \| q \| \| k \| )$ and $\ddot { w } = \| q \rangle \langle k \| / ( \| q \| \| k \| )$ ) satisfy $\hat { s } ^ { 2 } + \hat { w } ^ { 2 } =$ 1 (with sˆ = cos θ and $\hat { w } = | \sin \theta | f o r \theta$ the angle between q and k). Thus the dot and wedge scores are two orthogonal coordinates on the unit circle ofpairwise relatedness—the dot product reads only sˆ and is blind to the wˆ axis along which near-orthogonal relationships vary. Consequently there is an order separation: fix $q \ne 0$ and two equal-norm keys $\| k _ { a } \| = \| k _ { b } \| = \kappa$ with $q ^ { \top } k _ { a } > q ^ { \top } k _ { b } \ge 0 ;$ then

• the dot channel under any monotone softmax has $A ^ { \mathrm { d o t } } ( k _ { a } ) > A ^ { \mathrm { d o t } } ( k _ { b } ) - i t$ must place more weight on the more-aligned $k _ { a }$ andcan neverfavor the more-orthogonal $k _ { b } ,$

• the wedge channel has $\| q \land k _ { a } \| < \| q \land k _ { b } \|$ and hence $A ^ { \mathrm { w e d } } ( \tilde { k _ { b } } ) > A ^ { \mathrm { w e d } } ( k _ { a } )$

Therefore, once $\beta$ exceeds a threshold $\beta ^ { \star } = \Delta ^ { \mathrm { d o t } } / ( \Delta ^ { \mathrm { d o t } } +$ $\Delta ^ { \mathrm { w e d } } ) \in ( 0 , 1 )$ (with ∆<sup>dot</sup> = A<sup>dot</sup> $( \dot { k } _ { a } ) - A ^ { \mathrm { d o t } } \dot { ( k _ { b } ) } > 0 ,$ $\Delta ^ { \mathrm { w e d } } = \dot { A } ^ { \mathrm { w e d } } ( k _ { b } ) - A ^ { \mathrm { w e d } } ( k _ { a } ) > 0 )$ , the blend $A = ( 1 -$ $\beta ) A ^ { \mathrm { { d o t } } } + \beta A ^ { \mathrm { { w e d } } }$ favors the near-orthogonal key $( A ( k _ { b } ) >$ $A ( k _ { a } ) ) - a$ ranking no dot-product score can realize. Such a $\beta$ exists and is reachable by the learnable weight; not every $\beta > 0$ flips the order.

Proof. The first identity is the Lagrange (Cauchy–Schwarz) identity; dividing by $\| q \| ^ { 2 } \| k \| ^ { 2 }$ gives $\hat { s } ^ { 2 } + \hat { w } ^ { 2 } = 1$ . The two per-channel orderings follow from monotonicity of the softmax and, for the wedge channel, from ∥q $\wedge \dot { k } \| ^ { 2 } \ =$ $\| q \| ^ { 2 } \kappa ^ { 2 } - ( q ^ { \top } k ) ^ { 2 }$ with $( q ^ { \top } k _ { a } ) ^ { 2 } > ( q ^ { \top } k _ { b } ) ^ { 2 }$ . For the blend, $\ddot { A } ( \dot { k } _ { b } ) - A \dot { ( k _ { a } ) } = \beta \Delta ^ { \mathrm { w e d } } - ( \mathrm { i } - \beta ) \bar { \Delta } ^ { \mathrm { d o t } }$ , which is positive if $\ddot { \beta } > \Delta ^ { \mathrm { d o t } } / ( \Delta ^ { \mathrm { d o t } } + \Delta ^ { \mathrm { w e d } } ) = \dot { \beta } ^ { \star } ;$ since $\Delta ^ { \mathrm { d o t } } , \Delta ^ { \mathrm { \scriptsize { s e d } } } > 0$ we have $\beta ^ { \star } \in ( 0 , 1 )$ , and at $\beta = 1$ the blend equals the wedge channel, which already flips the order. □

Representational-complexity view. $\| q \land k \| ^ { 2 }$ is a quadratic form in $( q , k )$ ; a dot product can reproduce it only after a quadratic feature lift $\dot { \phi } ( x ) = \mathrm { v e c } ( x \dot { x ^ { \top } } )$ of dimension up to $\hat { O } ( d _ { h } ^ { 2 } )$ , whereas the wedge computes it in closed form at $O ( d _ { h } )$ ) via Cauchy–Schwarz.

Scope of the claim. This separation holds at afixed representation. Because Q, K come from learnable projections, a deep network can in principle relearn features that re-encode orthogonal relationships as aligned ones and route them through the dot channel, which is why existing dot-product policies work. Proposition 1 is therefore not an impossibility result for dot-product models; it states that the wedge supplies this ordering directly, without relearning features or a quadratic $l i f t$ , a low-cost inductive bias whose payof on real tasks is answered empirically in Section 5 (Figure 3 gives a $d _ { h } { = } 2$ instance).

Mechanism. Figure 3 visualizes Proposition 1 on a threephase toy example: after the softmax the dot and wedge channels put their weight on opposite (aligned vs. orthogonal) keys, and the learnable $\beta$ interpolates between them. From this we hypothesize that cross-phase correlations within a chunk often fall along near-orthogonal directions the dot product misses and that the wedge can score them directly— supported by the TO-DoL analysis of Section 1. Its real benefit is evaluated empirically (Section 5); the causal attribution $( Q / K$ routing analysis and OOD ablations) is left to future work (Section 6).

![](images/40be2752f01617ce2095408eef1685e543f8f1b1154a8990d2073734e67e8350.jpg)  
Figure 3: Toy example $( d _ { h } = 2 ,$ , q along axis A). (a) the three keys make angles $0 ^ { \circ } / 4 5 ^ { \circ } / 9 0 ^ { \circ }$ with $q _ { 0 } ; \mathbf { ( b ) }$ normalized scores versus angle—dot ∝ cos θ (falls with angle), wedge ∝ sin θ (rises with angle), crossing at $4 5 ^ { \circ }$ , exactly the $\hat { s } ^ { \breve { 2 } } + \hat { w } ^ { 2 } =$ 1 of Proposition 1; (c) after the softmax the two attention distributions are flipped: the dot channel gives its weight to the aligned $k _ { 0 } ~ ( 0 . 4 3 4 )$ ), the wedge channel to the orthogonal $k _ { 2 } ( 0 . \bar { 4 } 3 4 )$

## 5 Experiments

We evaluate TFGCA in four settings: LIBERO (Liu et al. 2023) (Section 5.1, full-suite in-distribution performance and component ablations), LIBERO-Plus (Fei et al. 2025) (Section 5.2, OOD generalization), RoboTwin 2.0 (Chen et al. 2025b) (Section 5.3, single-task, clean and domainrandomized), and the AgiBot A2 real robot (Section 5.4). Dataset sources, per-benchmark training and evaluation hyperparameters (optimizer, learning rate, chunk length, SWT levels, alignment-loss weight, etc.), and the training hardware and wall-clock times are all listed in Appendices A, B, and G; on each benchmark TFGCA and its same-source reproduced $\pi _ { 0 . 5 }$ base share consistent data, training budget, and evaluation, difering only in the TFGCA-specific SWT/geometry configuration.

The empirical emphasis across these four settings is TFGCA’s robustness under out-of-distribution (OOD) conditions: beyond the near-saturated in-distribution LIBERO, evaluation covers the unseen perturbations of LIBERO-Plus, RoboTwin domain randomization, and the real AgiBot A2 robot. We report each setting separately.

## 5.1 LIBERO: Full-Suite Experiments

Setup. We evaluate TFGCA attached to $\pi _ { 0 . 5 }$ on the four LIBERO task suites (Spatial, Object, Goal, Long/L10), against the $\pi _ { 0 . 5 }$ base (reported value and our reproduction); a broader comparison with external methods is deferred to Appendix C. Because the base success rate is already high (∼97% average), LIBERO is a near-saturated in-distribution setting. LIBERO is also single-arm (end-efector-delta control), so the specific bimanual time-axis division-of-labor of TO-DoL (Section 1) does not arise here; even so, the wedge remains a general scoring channel sensitive to the near-orthogonal components the dot product misses, and can still supply complementary information in-distribution. The full model is weakly but consistently best on the four-suite average below, though at LIBERO’s saturation these gaps are small and we do not read them as isolating the wedge’s efect. Both the ablations and the full TFGCA are trained on our reproduced $\pi _ { 0 . 5 }$ . Table 1 reports the full TFGCA together with per-component ablations: “w/o SWT” removes the wavelet frequency branch (C1); “w/o geometry” removes the wedge channel, reducing the attention to pure dot product $( \mathrm { C } 2 ) ; \mathrm { \Lambda ^ { * } w / o }$ alignment” disables the optional alignment loss of Section 4.1.

<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td> $\operatorname { A v g }$ </td></tr><tr><td>π0.5 (reported)</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>π0.5 (repro.)</td><td>95.2</td><td>99.6</td><td>97.2</td><td>94.6</td><td>96.7</td></tr><tr><td>TFGCÁ (full)</td><td>98.5</td><td>99.4</td><td>97.7</td><td>97.0</td><td>98.2</td></tr><tr><td>w/o SWT</td><td>96.8</td><td>99.1</td><td>97.1</td><td>97.8</td><td>97.7</td></tr><tr><td>w/o geometry</td><td>97.2</td><td>99.7</td><td>97.6</td><td>96.5</td><td>97.8</td></tr><tr><td>w/o alignment</td><td>94.9</td><td>99.5</td><td>98.5</td><td>95.5</td><td>97.1</td></tr></table>

Table 1: LIBERO success rate (SR%) on the four suites and the average. TFGCA and the reproduced $\pi _ { 0 . 5 }$ base are run under the same protocol; the $\pi _ { 0 . 5 }$ reference is the reported value (Physical Intelligence et al. 2025). A broader comparison against 11 external methods (DP, OpenVLA, SpatialVLA, CoT-VLA, π<sub>0</sub>/-FAST, GR00T, OpenVLA-OFT, Fast-WAM, X-VLA) is given in Appendix C.

On near-saturated LIBERO, TFGCA still delivers a consistent average gain. Relative to our reproduced $\pi _ { 0 . 5 }$ base (96.7% average), full TFGCA lifts the four-suite average to $9 8 . 2 \% - \mathrm { a \ + 1 . 5 }$ net gain despite only ∼3 points of headroom, and it raises the hardest suite (Long) from 94.6 to 97.0; the primary evidence for OOD robustness comes from the later evaluations. The component ablation illustrates the joint necessity of the three components in-distribution: removing the wavelet frequency branch (w/o SWT), the wedge channel (w/o geometry), or the alignment loss (w/o alignment) each lowers the four-suite average (to 97.7, 97.8, and 97.1 respectively), with w/o alignment dropping the most.

## 5.2 LIBERO-Plus: OOD Generalization

Setup. LIBERO-Plus applies seven families of out-ofdistribution perturbations to LIBERO tasks—camera viewpoint (Camera), robot initialization (Robot), language rephrasing (Language), lighting (Light), background (Background), sensor noise (Noise), and object layout (Layout)— reporting a success rate per family and a Total. All policies are trained only on clean (unperturbed) LIBERO and then tested under each perturbation, so LIBERO-Plus measures OOD generalization to unseen perturbations. We compare TFGCA attached to $\pi _ { 0 . 5 }$ against the same-source reproduced base, both self-tested under the same protocol (Table 2).

On LIBERO-Plus, TFGCA improves broadly over its same-source base. Under the same protocol, TFGCA beats the reproduced base (66.7) on six of the seven perturbations, and the Total rises from 66.7 to 73.0 (+6.3), with the gains concentrated on the base’s hardest families—Noise, Robot, and Camera (per-family in Table 2; the full per-suite × per-perturbation results are in Appendix E). The component ablation mirrors LIBERO but with a diferent ordering under OOD: removing the wedge channel (w/o geometry) drops the Total the most (to $6 9 . 7 , - 3 . 3 )$ , ahead of w/o alignment (71.9) and w/o SWT (72.1)—the reverse of the nearsaturated in-distribution case (Section 5.1), where alignment mattered most, consistent with the geometry channel contributing more precisely under out-of-distribution perturbations.

## 5.3 RoboTwin: Single-Task (Clean and Randomized)

Setup. All policies are trained only on RoboTwin 2.0 clean (no domain randomization) demonstrations for 30k steps, then tested separately in clean and randomized environments. Since randomization is never seen during training, the randomized column measures zero-shot generalization to unseen perturbations. Table 3 compares TFGCA (attached to $\pi _ { 0 . 5 } )$ against the $\pi _ { 0 . 5 }$ base under the same protocol; a full comparison with RoboTwin leaderboard baselines (DP, ACT, DP3, RDT, π ) is in Appendix D.

TFGCA improves the base on both clean and randomized, and more so under OOD. Relative to $\pi _ { 0 . 5 } .$ TFGCA improves the six-task mean by +3.7 on clean and +28.5 on randomized (randomization is never trained on, so the randomized column measures zero-shot generalization). Its randomized average of 42.7 is also the highest among all methods in the Appendix D comparison—about 24 points above the next best—while non-pretrained baselines collapse almost entirely under randomization. The most dramatic are click\_bell and stack\_bowls\_three: $\pi _ { 0 . 5 }$ nearly collapses under randomization (≈6%), whereas TFGCA holds 86% and 59%.

The largest randomized gains are on contact/multistage tasks—a correlation, not causal proof. These tasks (click\_bell, stack\_bowls\_three, turn\_switch) are consistent in direction with the Section 4.5 motivation, but this is only a task-level correlation—handover\_block dips from 13 to 11 on randomized, within the binomial confidence interval at n=100 and better read as noise. The only two clean regressions (open\_microwave, stack\_bowls\_three) also occur on near-saturated tasks and still show clear randomized gains, so they look like saturation noise. Component net contributions are assessed by the Section 5.1 ablation.

## 5.4 AgiBot A2: Real-Robot Validation

We further conduct a real-robot evaluation on AgiBot $\mathbf { A } 2 ,$ comparing TFGCA against the $\pi _ { 0 . 5 }$ base on three manipulation tasks (Table 4). Each task is run 20 times, for 60 trials in total; the phase decomposition, success criteria, and test protocol of each task are given in Appendix F.

Across the three real-robot tasks, TFGCA attains an overall success rate of 61.67%, above the $\pi _ { 0 . 5 }$ base’s 50.00%, an improvement of 11.67 percentage points. All three tasks improve, with the largest gain on the pull-a-tissue-from-thebox task (+15 points). This indicates that TFGCA improves multi-stage manipulation and object interaction on a real robot, beyond the simulation benchmarks.

<table><tr><td>Method</td><td>Camera</td><td>Robot</td><td>Language</td><td>Light</td><td>Background</td><td>Noise</td><td>Layout</td><td>Total</td></tr><tr><td>π0.5 (our reproduced base)</td><td>42.1</td><td>62.9</td><td>77.2</td><td>96.3</td><td>88.4</td><td>38.4</td><td>79.6</td><td>66.7</td></tr><tr><td>TFGCA (ours, 30k)</td><td>51.5</td><td>73.4</td><td>81.8</td><td>96.4</td><td>87.8</td><td>52.3</td><td>81.7</td><td>73.0</td></tr><tr><td>w/o SWT</td><td>46.7</td><td>74.2</td><td>80.7</td><td>93.8</td><td>87.4</td><td>54.2</td><td>82.4</td><td>72.1</td></tr><tr><td>w/o geometry</td><td>41.1</td><td>67.9</td><td>81.0</td><td>94.6</td><td>87.1</td><td>49.8</td><td>83.4</td><td>69.7</td></tr><tr><td>w/o alignment</td><td>49.6</td><td>70.5</td><td>79.1</td><td>97.4</td><td>89.1</td><td>51.1</td><td>82.7</td><td>71.9</td></tr></table>

Table 2: LIBERO-Plus success rate (%) across seven perturbation families and the Total. All rows are self-tested under the same protocol: our reproduced same-source base, the TFGCA-augmented model, and its three component ablations (w/o SWT, w/o geometry, w/o alignment; cf. Table 1).

<table><tr><td>Task</td><td>π0.5</td><td>TFGCA (ours)</td></tr><tr><td>open_microwave</td><td>93 / 27</td><td>88 / 36</td></tr><tr><td>stamp_seal</td><td>15 /5</td><td>19 / 3</td></tr><tr><td>handover_block</td><td>57 / 13</td><td>72 / 11</td></tr><tr><td>turn_switch</td><td>41 / 28</td><td>54 / 61</td></tr><tr><td>stack_bowls_three</td><td>83 / 6</td><td>74/59</td></tr><tr><td>click_bell</td><td>79/6</td><td>83 / 86</td></tr><tr><td>Average</td><td>61.3 / 14.2</td><td>65.0 / 42.7</td></tr></table>

Table 3: RoboTwin 2.0 success rate (%); each cell is clean / randomized. $\pi _ { 0 . 5 }$ and TFGCA (ours) are run under the same protocol (Aloha-AgileX, 30k steps, 100 evaluations each). The last row is the 6-task average. A full comparison with leaderboard baselines (DP, ACT, DP3, RDT, π<sub>0</sub>) is given in Appendix D.
<table><tr><td>Task</td><td>TFGCA</td><td>π0.5</td><td>∆</td></tr><tr><td>Soap into box</td><td>6/20 (30%)</td><td>4/20 (20%)</td><td>+10 pts</td></tr><tr><td>Plush toys into basket</td><td>19/20 (95%)</td><td>17/20 (85%)</td><td>+10 pts</td></tr><tr><td>Pull a tissue</td><td>12/20 (60%)</td><td>9/20 (45%)</td><td>+15 pts</td></tr><tr><td>Overall</td><td colspan="3">37/60 (61.67%) 30/60 (50.00%) +11.67 pts</td></tr></table>

Table 4: AgiBot A2 real-robot success rate.

## 6 Discussion and Limitations

Boundary of the evidence. Identity initialization (Section 4.4) only guarantees that the module matches the base pointwise at initialization—a statement about attachment safety, not a theoretical guarantee that the jointly fine-tuned model beats the base. Efectiveness is instead established empirically: across all four settings TFGCA improves over the same-source base in a consistent direction (Section 5), and the occasional small regressions on near-saturated tasks (Section 5.3) do not change this overall conclusion.

Limitations. The frequency framing is weaker at short horizons: LIBERO’s $T = 1 0$ and single SWT level make the decomposition close to a single trend/detail split, so the frequency narrative is more solid on the $T = 5 0$ , two-level, second-order-diference RoboTwin setting.

Future work. First, linking TO-DoL to the wedge channel: a Q/K routing analysis can test whether the wedge really moves attention mass onto the cross-phase token pairs with “small dot but large wedge,” and, paired with a dot-only ablation under OOD, could upgrade the mechanism from a motivating hypothesis to evidence. Second, task-dependent inference frequency: how often a chunked policy should replan appears to interact with a task’s reactivity, and the module’s frequency view may help predict a good setting. Third, chunk-boundary diagnostics: chunked policies jump where consecutive chunks disagree, and a boundary-discontinuity metric (e.g., spectral arc length in kinematics (Balasubramanian et al. 2015)) would make this observable and connect the frequency view to a concrete smoothness quantity.

## 7 Conclusion

Chunked VLA policies under-use the structure of the trajectories they emit, missing both the multi-scale frequency content of motion and the cross-phase, near-orthogonal relationships within a chunk that dot-product attention can retrieve only indirectly (it is least sensitive exactly near orthogonality and must re-encode such relationships as similarity). Time–Frequency Geometric Cross-Attention addresses both with a single drop-in module: a per-dimension control-space wavelet tokenization of the action chunk (jointly encoding the multi-dimensional action content at each time–band position), and a cross-attention in which each time token retrieves information from the whole chunk’s time–frequency tokens through a learnable blend of similarity and geometric diference. The module uses a zero-initialized residual, so it reproduces the base output exactly at initialization, easing drop-in onto a pretrained policy and joint fine-tuning with it. Empirically, TFGCA raises overall success on RoboTwin and improves more under domain randomization; on LIBERO it yields an average gain on a near-saturated benchmark, and on its OOD variant LIBERO-Plus it improves the Total by +6.3 over the same-source base; on the real AgiBot A2 robot the overall success rate rises from 50.00% to 61.67%. These gains are consistent in direction under varying degrees of out-of-distribution conditions.

## References

Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022. Flamingo: A Visual Language Model for Few-Shot Learning. In Advances in Neural Information Processing Systems (NeurIPS).

Bachlechner, T.; Majumder, B. P.; Mao, H. H.; Cottrell, G. W.; and McAuley, J. 2021. ReZero is All You Need: Fast Convergence at Large Depth. In Conference on Uncertainty in Artificial Intelligence (UAI).

Balasubramanian, S.; Melendez-Calderon, A.; Roby-Brami, A.; and Burdet, E. 2015. On the Analysis of Movement Smoothness. Journal of NeuroEngineering and Rehabilitation, 12(112).

Black, K.; et al. 2024. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164.

Brandstetter, J.; van den Berg, R.; Welling, M.; and Gupta, J. K. 2023. Cliford Neural Layers for PDE Modeling. In International Conference on Learning Representations (ICLR).

Brehmer, J.; de Haan, P.; Behrends, S.; and Cohen, T. 2023. Geometric Algebra Transformer. In Advances in Neural Information Processing Systems (NeurIPS).

Brohan, A.; et al. 2023. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. In Conference on Robot Learning (CoRL).

Chen, H.; Luong, V.; Mukherjee, L.; and Singh, V. 2025a. SimpleTM: A Simple Baseline for Multivariate Time Series Forecasting. In International Conference on Learning Representations (ICLR).

Chen, T.; et al. 2025b. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation. arXiv preprint arXiv:2506.18088.

Chi, C.; Xu, Z.; Feng, S.; Cousineau, E.; Du, Y.; Burchfiel, B.; Tedrake, R.; and Song, S. 2023. Difusion Policy: Visuomotor Policy Learning via Action Difusion. In Robotics: Science and Systems (RSS). Extended in International Journal of Robotics Research 44(10–11):1684–1704, 2025.

Fei, S.; Wang, S.; Shi, J.; Dai, Z.; Cai, J.; Qian, P.; Ji, L.; He, X.; Zhang, S.; Fei, Z.; Fu, J.; Gong, J.; and Qiu, X. 2025. LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models. arXiv preprint arXiv:2510.13626.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR).

Kim, M. J.; Finn, C.; and Liang, P. 2025. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. In Robotics: Science and Systems (RSS).

Kim, M. J.; et al. 2024. OpenVLA: An Open-Source Vision-Language-Action Model. In Conference on Robot Learning (CoRL).

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling.

In International Conference on Learning Representations (ICLR).

Liu, B.; Zhu, Y.; Gao, C.; Feng, Y.; Liu, Q.; Zhu, Y.; and Stone, P. 2023. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track.

Liu, S.; et al. 2025. RDT-1B: A Difusion Foundation Model for Bimanual Manipulation. In International Conference on Learning Representations (ICLR).

Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and Long, M. 2024. iTransformer: Inverted Transformers Are Efective for Time Series Forecasting. In International Conference on Learning Representations (ICLR).

NVIDIA. 2025. GR00T N1.6: An Improved Open Foundation Model for Generalist Humanoid Robots. https: //research.nvidia.com/labs/gear/gr00t-n1\_6/.

NVIDIA; Bjorck, J.; Castañeda, F.; Cherniadev, N.; Da, X.; Ding, R.; Fan, L.; Fang, Y.; Fox, D.; et al. 2025. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv preprint arXiv:2503.14734.

Octo Model Team. 2024. Octo: An Open-Source Generalist Robot Policy. In Robotics: Science and Systems (RSS).

Pertsch, K.; Stachowicz, K.; Ichter, B.; Driess, D.; Nair, S.; Vuong, Q.; Mees, O.; Finn, C.; and Levine, S. 2025. FAST: Eficient Action Tokenization for Vision-Language-Action Models. arXiv preprint arXiv:2501.09747.

Physical Intelligence; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; et al. 2025. π<sub>0.5</sub>: A Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054.

Qu, D.; Song, H.; Chen, Q.; Yao, Y.; Ye, X.; Ding, Y.; Wang, Z.; Gu, J.; Zhao, B.; Wang, D.; et al. 2025. SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model. In Robotics: Science and Systems (RSS).

Wu, H.; Xu, J.; Wang, J.; and Long, M. 2021. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. In Advances in Neural Information Processing Systems (NeurIPS).

Yi, K.; et al. 2023. Frequency-Domain MLPs Are More Efective Learners in Time Series Forecasting. In Advances in Neural Information Processing Systems (NeurIPS).

Yuan, T.; Dong, Z.; Liu, Y.; and Zhao, H. 2026. Fast-WAM: Do World Action Models Need Test-Time Future Imagination? arXiv preprint arXiv:2603.16666.

Ze, Y.; Zhang, G.; Zhang, K.; Hu, C.; Wang, M.; and Xu, H. 2024. 3D Difusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations. In Robotics: Science and Systems (RSS).

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding Conditional Control to Text-to-Image Difusion Models. In IEEE/CVF International Conference on Computer Vision (ICCV).

Zhao, Q.; Lu, Y.; Kim, M. J.; Fu, Z.; Zhang, Z.; Wu, Y.; Li, Z.; Ma, Q.; Han, S.; Finn, C.; et al. 2025. CoT-VLA:

Visual Chain-of-Thought Reasoning for Vision-Language-Action Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Zhao, T. Z.; Kumar, V.; Levine, S.; and Finn, C. 2023. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Robotics: Science and Systems (RSS).

Zheng, J.; Li, J.; Wang, Z.; Liu, D.; Kang, X.; Feng, Y.; Zheng, Y.; Zou, J.; Chen, Y.; Zeng, J.; et al. 2026. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model. In International Conference on Learning Representations (ICLR).

Zhou, T.; Ma, Z.; Wen, Q.; Wang, X.; Sun, L.; and Jin, R. 2022. FEDformer: Frequency Enhanced Decomposed Transformer for Long-Term Series Forecasting. In International Conference on Machine Learning (ICML).

This document provides the dataset configurations, training and evaluation settings, resource usage, and the Temporal Orthogonal Division-of-Labor (TO-DoL) analysis that are omitted from the main paper for space. Unless otherwise noted, the simulated experiments use LeRobot as the common infrastructure for data organization, policy training, and rollout evaluation; TFGCA and the corresponding $\pi _ { 0 . 5 }$ base share data processing, training budget, and evaluation conditions. Method definitions, theoretical properties, and the main results are given in the main paper and are not repeated here.

## A Datasets

LIBERO. The LIBERO experiments use the LeRobot Hugging Face dataset lerobot/libero, read and organized in the LeRobot data format. Evaluation covers the four standard suites LIBERO-Long, LIBERO-Goal, LIBERO-Object, and LIBERO-Spatial. Each sample contains two 256 × 256 RGB images, an 8-dimensional robot state, and a 7-dimensional action; that action is not seven joint angles but a Cartesian end-efector increment (3-D translation, 3-D rotation) plus a 1-D binary gripper command. Images use identity normalization; states and actions use quantile normalization. Training and evaluation episodes follow the LeRobot dataset documentation and the standard LIBERO benchmark conventions.

RoboTwin 2.0. The RoboTwin 2.0 experiments use the LeRobot Hugging Face dataset lerobot/robotwin\_unified, organized in the LeRobot format. Each sample contains three 480×640 RGB images, a 14-dimensional robot state, and a 14-dimensional joint-space action (dual-arm qpos). Policies are trained only on clean demonstrations and evaluated separately in clean and randomized environments, to probe in-distribution performance and generalization to unseen environmental perturbations. Episode setup follows the LeRobot dataset documentation and the RoboTwin benchmark conventions.

AgiBot A2. The real-robot data were collected in-house and cover three tasks: placing soap into a soap box, placing two plush toys into a basket, and pulling a tissue from a tissue box. Owing to confidentiality constraints, we do not release the raw data or the collection procedure; public documentation of the A2 platform’s software/hardware and developer interfaces is available from AgiBot’s oficial developer materials.

## B Training and Evaluation Configuration

Common implementation. All simulated training and evaluation are run in LeRobot, including the policy interface, data loading, feature normalization, action post-processing, and environment rollout. TFGCA is attached between the VLA transformer output and the action head, with base model lerobot/pi05\_base. Apart from TFGCA-specific settings, the augmented model and its $\pi _ { 0 . 5 }$ base use the same data, training budget, and evaluation pipeline. The common training configuration is given in Table 5.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Base model</td><td>lerobot/pi05_base</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Initial learning rate</td><td>2.5 × 10−5</td></tr><tr><td>Adam betas</td><td>(0.9, 0.95)</td></tr><tr><td>Adam epsilon</td><td> $\mathrm { i } 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Warmup</td><td>1,000 steps</td></tr><tr><td>LR schedule</td><td>cosine to  $\mathrm { { 2 . 5 \times 1 0 ^ { - 6 } } }$  at 30k steps</td></tr><tr><td>Numerical precision</td><td>bfloat16</td></tr><tr><td>Memory optimization</td><td>gradient checkpointing</td></tr><tr><td>Random seed</td><td>1000</td></tr><tr><td>FM time sampling</td><td>Beta(1.5, 1.0), scale 0.999, shift 0.001</td></tr><tr><td>Inference denoising steps</td><td>10</td></tr><tr><td>Checkpoint interval</td><td>1,000 steps</td></tr></table>

Table 5: Common training configuration shared by TFGCA and the $\pi _ { 0 . 5 }$ base.

Per-benchmark configuration. The training and evaluation configuration for the three simulation benchmarks (LIBERO, LIBERO-Plus, RoboTwin 2.0) is given in Table 6; other common hyperparameters are in Table 5. On each benchmark TFGCA and the corresponding reproduced π base share consistent data, training budget, and evaluation, difering only in the TFGCA-specific SWT/geometry configuration. The action-space alignment loss uses the groundtruth action as its target with a stop-gradient on the target branch. The diference order counts the per-dimension sequence out of action\_proj as order 0: LIBERO and LIBERO-Plus feed the SWT directly, while RoboTwin 2.0 applies two finite diferences before the SWT. Success rates are computed as the fraction of rollouts that strictly complete the task. LIBERO-Plus is not trained separately: the LIBERO-trained models are evaluated on its out-ofdistribution perturbations to measure zero-shot generalization; the construction, parameter ranges, and success criteria of the seven perturbation families (camera viewpoint, robot initialization, language, lighting, background, sensor noise, object layout) follow the oficial LIBERO-Plus setting (Fei et al. 2025).

How the TFGCA-specific settings are chosen. We do not run a large hyperparameter sweep; the few TFGCA-specific settings in Table 6 are fixed from the data and the base configuration rather than tuned per benchmark. The chunk and execution length follow the LeRobot base configuration for each benchmark (10 for LIBERO/LIBERO-Plus, 50 for RoboTwin). The number of SWT levels is tied to the chunk length—one level for the short LIBERO chunk, two for the longer RoboTwin chunk, so that the coarsest band still covers a meaningful fraction of the chunk. The diference order before the SWT is set from the energy distribution of the action sequence: LIBERO/LIBERO-Plus end-efector deltas are already near-stationary and are fed to the SWT directly (order 0), whereas RoboTwin joint-space qpos carries a strong low-frequency drift, so we apply two finite diferences (order 2) to move its energy into the detail bands. The alignment-loss weight is 0.01 in the multi-task simulation settings (LIBERO, LIBERO-Plus) and 0.1 in the single-task RoboTwin setting, where a stronger alignment target is affordable without competing across tasks. The remaining values (optimizer, learning rate, schedule, batch size, denoising steps, seed) are shared with the $\pi _ { 0 . 5 }$ base and are not re-tuned for TFGCA.

<table><tr><td>Setting</td><td>LIBERO</td><td>LIBERO-Plus</td><td>RoboTwin</td></tr><tr><td>Training data</td><td>libero</td><td>reuse</td><td>robotwin</td></tr><tr><td>Training steps Batch (per GPU /</td><td>30k 32 / 64</td><td></td><td>30k/task 32 / 64</td></tr><tr><td>global)</td><td></td><td></td><td></td></tr><tr><td>Chunk 1 exec. length</td><td>10 / 10</td><td>10 / 10</td><td>50 / 50</td></tr><tr><td>SWT levels Diff. order before</td><td>1 0</td><td>1 0</td><td>2 2</td></tr><tr><td>SWT</td><td></td><td></td><td></td></tr><tr><td>Mother wavelet</td><td>db2</td><td>db2</td><td>db2</td></tr><tr><td>Attention heads</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Alignment-loss weight</td><td>0.01</td><td>0.01</td><td>0.1</td></tr><tr><td>Rollouts / condi- tion</td><td>conv.</td><td>official</td><td>100 each</td></tr></table>

Table 6: Per-benchmark training and evaluation configuration. Filters are learnable in all cases (initialized to db2). “reuse” means LIBERO-Plus reuses the LIBEROtrained model without separate training; “30k/task” means RoboTwin is trained 30k steps per task; “conv.” follows the LIBERO benchmark convention for rollouts per condition.

## C LIBERO: Extended Comparison

Table 1 of the main paper reports TFGCA, the reproduced $\pi _ { 0 . 5 }$ base, and the ablations under a single protocol. Table 7 places these alongside external methods for context. As these values come from the original papers or recent public comparisons under possibly diferent evaluation protocols, they are provided for reference only and should not be read as a strict ranking against TFGCA.

## D RoboTwin 2.0: Full Comparison

Table 3 of the main paper compares TFGCA against the $\pi _ { 0 . 5 }$ base under a single protocol. Table 8 places these alongside the RoboTwin 2.0 oficial leaderboard baselines. $\pi _ { 0 . 5 }$ and TFGCA (ours) are run under the same protocol (Aloha-AgileX, 30k steps, 100 evaluations each) and are directly comparable; DP, ACT, DP3, RDT, and $\pi _ { 0 }$ are the leaderboard-reported values. DP3 uses point-cloud input, a diferent modality from the other RGB methods.

## E LIBERO-Plus: Per-Suite Full Results

Table 2 ofthe main paper reports only the average ofthe seven LIBERO-Plus perturbations across the four suites. Table 9 gives TFGCA’s full per-perturbation success rate (%) on each of the four suites (Spatial, Object, Goal, Long); each cell is TFGCA, with the $\pi _ { 0 . 5 }$ base and the delta relative to it in parentheses. The final Avg row is the four-suite average and corresponds to the two LIBERO-Plus rows of main-paper Table 2. Both models are self-tested under the same protocol.

![](images/79dbb501bcbae3b1e0bd19c52f337390420a794e480b70233964bad4258ba477.jpg)  
Figure 4: Put the soap into the soap box. Four-stage keyframes: grasp the soap → move above the box → release into the box → reset.

![](images/c761f7a6d478d497db5482a8a76d5deba4c2de591cd6502634b09ca6c6188af4.jpg)  
Figure 5: Pull a tissue from the box. Four-stage keyframes: the near hand picks up the tissue box → the other hand pulls a tissue → put down the box → put down the tissue.

By suite, TFGCA’s gains remain concentrated on the perturbations the base handles worst—Noise and Camera (e.g., Spatial-Noise +19.1, Long-Noise +18.5, Goal-Camera +15.7)—while on the near-saturated Light/Background it is roughly flat or slightly regresses, within the evaluation sampling variance. This is consistent with the average-level conclusion of Section 5.2 in the main paper.

## F Real-Robot Task Decomposition (AgiBot A2)

AgiBot A2 protocol. TFGCA and the $\pi _ { 0 . 5 }$ base use the same training data, number of steps, task initialization, and testing conditions. Each model is tested 20 times per task. Evaluation is strict: a single continuous execution counts as a success only if the goal state is fully reached; human intervention, mid-task retries, dropped objects, or the failure of any required sub-step count as failures.

Figures 4–6 show a stage-wise keyframe decomposition of the three AgiBot A2 real-robot tasks. Keyframes are taken from a representative rollout and correspond one-to-one with Table 4 of the main paper.

## G Resource Usage

## G.1 Training Hardware and Wall-Clock

The resource records in Table 10 correspond to each training job rather than the cumulative cost of a whole benchmark. Beyond the measured training hardware and approximate wall-clock, we do not report peak memory, training throughput, end-to-end inference latency, or other quantities that were not independently measured, so as to avoid inferring real running costs from configuration files or theoretical values. Exact software versions were not recorded in a single verifiable log and are therefore not stated speculatively; the AgiBot A2 platform’s internal resource configuration is also withheld for confidentiality reasons.

<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg</td></tr><tr><td>Diffusion Policy (Chi et al. 2023)</td><td>78.5</td><td>87.5</td><td>73.5</td><td>64.8</td><td>76.1</td></tr><tr><td>OpenVLA (Kim et al. 2024)</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>SpatialVLA (Qu et al. 2025)</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>78.1</td></tr><tr><td>CoT-VLA (Zhao et al. 2025)</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>83.9</td></tr><tr><td>π₀-FAST (Pertsch et al. 2025)</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>85.5</td></tr><tr><td>GR00T-N1 (NVIDIA et al. 2025)</td><td>94.4</td><td>97.6</td><td>93.0</td><td>90.6</td><td>93.9</td></tr><tr><td> $\pi _ { 0 }$  (Black et al. 2024)</td><td>98.0</td><td>96.8</td><td>94.4</td><td>88.4</td><td>94.4</td></tr><tr><td> $\pi _ { 0 . 5 }$  (Physical Intelligence et al. 2025)</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>GR00T-N1.6 (NVIDÍA 2025)</td><td>97.7</td><td>98.5</td><td>97.5</td><td>94.4</td><td>97.0</td></tr><tr><td>OpenVLA-OFT (Kim, Finn, and Liang 2025)</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td>Fast-WAM (Yuan et al. 2026)</td><td>98.2</td><td>100.0</td><td>97.0</td><td>95.2</td><td>97.6</td></tr><tr><td>X-VLA (Zheng et al. 2026)</td><td>98.2</td><td>98.6</td><td>97.8</td><td>97.6</td><td>98.1</td></tr><tr><td>π0.5 (our repro.)</td><td>95.2</td><td>99.6</td><td>97.2</td><td>94.6</td><td>96.7</td></tr><tr><td>TFGCA (full)</td><td>98.5</td><td>99.4</td><td>97.7</td><td>97.0</td><td>98.2</td></tr><tr><td>TFGCA – SWT</td><td>96.8</td><td>99.1</td><td>97.1</td><td>97.8</td><td>97.7</td></tr><tr><td>TFGCA – geometry</td><td>97.2</td><td>99.7</td><td>97.6</td><td>96.5</td><td>97.8</td></tr><tr><td>TFGCA — alignment</td><td>94.9</td><td>99.5</td><td>98.5</td><td>95.5</td><td>97.1</td></tr></table>

Table 7: LIBERO success rate (SR%), TFGCA and $\pi _ { 0 . 5 }$ (our reproduction and ablations, same protocol) alongside external methods (upper block, reference only—protocols may difer).
<table><tr><td>Task</td><td>DP</td><td>ACT</td><td>DP3</td><td>RDT</td><td>π0</td><td>π0.5</td><td>TFGCA (ours)</td></tr><tr><td>open_microwave</td><td>5/0</td><td>86 / 0</td><td>61 / 22</td><td>37 / 20</td><td>80 / 50</td><td>93 / 27</td><td>88/ 36</td></tr><tr><td>stamp_seal</td><td>2/0</td><td>2/0</td><td>18/0</td><td>1/0</td><td>3/4</td><td>15/ 5</td><td>19 /3</td></tr><tr><td>handover_block</td><td>10 /0</td><td>42/0</td><td>70/0</td><td>45 / 14</td><td>45/8</td><td>57 / 13</td><td>72 / 11</td></tr><tr><td>turn_switch</td><td>36/1</td><td>5/2</td><td>46/8</td><td>35 / 15</td><td>27 / 23</td><td>41 /28</td><td>54/61</td></tr><tr><td>stack_bowls_three</td><td>63 / 0</td><td>48 / 0</td><td>57/5</td><td>51 / 17</td><td>66 / 24</td><td>83 / 6</td><td>74/59</td></tr><tr><td>click_bell</td><td>54/0</td><td>58/3</td><td>90/0</td><td>80/9</td><td>44/3</td><td>79/6</td><td>83 /86</td></tr><tr><td>Average</td><td>28.3 / 0.2</td><td>40.2 / 0.8</td><td>57.0/5.8</td><td>41.5 / 12.5</td><td>44.2 / 18.7</td><td>61.3 / 14.2</td><td>65.0 / 42.7</td></tr></table>

Table 8: RoboTwin 2.0 success rate (%); each cell is clean / randomized (the leaderboard’s Easy / Hard). Values for DP (Chi et al. 2023), ACT (Zhao et al. 2023), DP3 (Ze et al. 2024), RDT (Liu et al. 2025), and $\pi _ { 0 }$ (Black et al. 2024) are from the RoboTwin 2.0 oficial leaderboard; $\pi _ { 0 . 5 }$ and TFGCA (ours) are run under the same protocol and directly comparable. The last row is the 6-task average.

![](images/ff140e8c7f110c487fdc4b84ef9bf1a96ad0008f350adcb7a14e387f44930604.jpg)  
Figure 6: Put the two plush toys into the basket. Keyframes: pick up a toy → place it in the basket, then repeat for the second toy.

## G.2 Added Parameter Count

The parameters TFGCA adds over the $\pi _ { 0 . 5 }$ base are fully determined by the module structure and can be computed exactly from the architecture (no independent measurement needed). Taking model width d = 1024 and LIBERO’s action dimension D = 7 (6-D Cartesian end-efector increment + 1-D binary gripper), the per-module counts are given in Table 11.

The added parameters are dominated by the geometric cross-attention’s Q/K/V/output projections (four 1024 × 1024 matrices), which do not exist in the $\pi _ { 0 . 5 }$ base and are the genuinely new part of TFGCA; the perdimension SWT encoder and the action-space projection (PerDimensionSWTEncoder and action\_proj) are almost negligible. The total ofabout 4.23M is roughly 0.10% of the ∼4B-parameter $\pi _ { 0 . 5 }$ backbone, consistent with the main paper’s positioning of TFGCA as a lightweight dropin module. This count is an architecture-determined exact value and does not change with training data or steps; when the action dimension D varies with the embodiment (e.g., D = 14 for RoboTwin and AgiBot), only action\_proj scales linearly in d × D, with negligible efect on the total.

## H Temporal Orthogonal Division-of-Labor (TO-DoL): Definition and Data Analysis

Sections 1 and 4.5 of the main paper use the claim that “the near-orthogonal structure of division-of-labor coordination lives along the time axis” as data support for the design motivation. This section gives the full definition, metrics, and robustness analysis. The analysis covers RoboTwin 2.0’s handover\_block and stack\_bowls\_three tasks, 50 demonstrations each, over the aloha-agilex and arx-x5 bimanual embodiments. It is based on existing demonstration trajectories and involves no new training or evaluation.

<table><tr><td>Suite</td><td>Camera</td><td>Robot</td><td>Language</td><td>Light</td><td>Background</td><td>Noise</td><td>Layout</td><td>Total</td></tr><tr><td>Spatial</td><td> $6 8 . 3 \ : ( 5 9 . 3 , + 9 . 0 )$ </td><td> $8 8 . 6 \ : ( 8 0 . 9 , + 7 . 7 )$ </td><td> $8 7 . 7 \ ( 8 1 . 8 , + 5 . 9 )$ </td><td> $9 9 . 0 \left( 9 7 . 3 , + 1 . 7 \right)$ </td><td>99.2 (94.2, +5.0)</td><td> $6 5 . 8 \ : ( 4 6 . 7 , + 1 9 . 1 )$ </td><td> $9 7 . 4 \ : ( 9 5 . 1 , + 2 . 3 )$ </td><td> $8 5 . 8 \ : ( 7 8 . 3 , + 7 . 4 )$ </td></tr><tr><td>Object</td><td> $4 1 . 7 \ : ( 3 3 . 6 , + 8 . 1 )$ </td><td> $7 0 . 8 \ : ( 5 8 . 5 , + 1 2 . 3 )$ </td><td> $9 4 . 1 \ ( 8 4 . 5 , + 9 . 6 )$ </td><td> $9 6 . 6 \left( 1 0 0 . 0 , - 3 . 4 \right)$ </td><td> $9 6 . 4 \ : ( 9 7 . 6 , - 1 . 2 )$ </td><td> $4 9 . 3 \ : ( 4 0 . 8 , + 8 . 5 )$ </td><td> $8 2 . 6 \ : ( 8 2 . 9 , - 0 . 2 )$ </td><td> $7 3 . 3 \ : ( 6 7 . 9 , + 5 . 4 )$ </td></tr><tr><td>Goal</td><td> $6 0 . 8 \ : ( 4 5 . 1 , + 1 5 . 7 )$ </td><td> $6 7 . 0 \ : ( 5 8 . 4 , + 8 . 5 )$ </td><td> $5 3 . 2 ( 5 8 . 3 , - 5 . 1 )$ </td><td> $9 7 . 5 \ : ( 9 5 . 7 , + 1 . 8 )$ </td><td> $7 5 . 1 \ : ( 7 7 . 2 , - 2 . 1 )$ </td><td> $4 8 . 8 \ : ( 3 9 . 3 , + 9 . 5 )$ </td><td> $6 4 . 2 \ : ( 6 0 . 7 , + 3 . 5 )$ </td><td> $6 4 . 9 \ : ( 5 9 . 9 , + 4 . 9 )$ </td></tr><tr><td>Long</td><td> $3 5 . 1 \ : ( 3 0 . 3 , + 4 . 8 )$ </td><td> $6 7 . 2 \ : ( 5 3 . 9 , + 1 3 . 3 )$ </td><td> $9 2 . 2 \ : ( 8 4 . 1 , + 8 . 1 )$ </td><td> $9 2 . 7 \ : ( 9 2 . 3 , + 0 . 4 )$ </td><td>80.6 (84.8, −4.2)</td><td> $4 5 . 2 \ : ( 2 6 . 7 , + 1 8 . 5 )$ </td><td> $8 2 . 4 \left( 7 9 . 8 , + 2 . 6 \right)$ </td><td> $6 7 . 9 \ : ( 6 0 . 7 , + 7 . 2 )$ </td></tr><tr><td>Avg</td><td> ${ \bf 5 1 . 5 } \left( { \bf 4 2 . 1 } , { \bf + 9 . 4 } \right)$ </td><td> $7 3 . 4 \ : ( 6 2 . 9 , + 1 0 . 5 )$ </td><td>81.8 (77.2, +4.6)</td><td>96.4 (96.3, +0.1)</td><td> ${ \mathbf { 8 7 . 8 \ : ( 8 8 . 4 , - 0 . 6 ) } }$ </td><td> $5 2 . 3 \ : ( 3 8 . 4 , + 1 3 . 9 )$ </td><td> $\mathbf { 8 1 . 7 \ ( 7 9 . 6 , + 2 . 1 ) }$ </td><td> $7 3 . 0 \ : ( 6 6 . 7 , + 6 . 3 )$ </td></tr></table>

Table 9: LIBERO-Plus per-suite, per-perturbation success rate (%). Each cell is TFGCA $( \pi _ { 0 . 5 }$ base, ∆); the Avg row is the four-suite average and corresponds to the two LIBERO-Plus rows of Table 2 in the main paper.

<table><tr><td>Experiment</td><td>GPU</td><td>Wall-clock</td></tr><tr><td>LIBERO</td><td>2× NVIDIA A800 80GB</td><td>~26 hours</td></tr><tr><td>RoboTwin 2.0</td><td>2× NVIDIA A800 80GB</td><td>~28 hours</td></tr></table>

Table 10: Resource usage per training job (not the cumulative cost of a full benchmark).
<table><tr><td>Module</td><td>Parameters</td></tr><tr><td>CrossGeometricAttention</td><td>4,198,403 (4.20M)</td></tr><tr><td>PerDimensionSWTEncoder</td><td>28,728 (0.029M)</td></tr><tr><td>action_proj</td><td>7,168 (0.007M)</td></tr><tr><td>Total</td><td>4,234,299 (4.23M)</td></tr></table>

Table 11: Added parameters of TFGCA at $d = 1 0 2 4 , D = 7 .$

## H.1 Motivation: Two Notions of Orthogonality

A natural but untested hypothesis is a same-instant version of orthogonality: at some frame t, two division-of-labor subactions A and B are simultaneously active with joint directions $v _ { A } ( t ) \perp v _ { B } ( t )$ . The data nearly falsify this hypothesis (Table 12): same-instant orthogonal coupling is essentially absent. The same data, however, exhibit very strong positive structure along the time axis—this is TO-DoL.

## H.2 Definition

Let each trajectory e have per-framejoint velocity $v ( t ) \in \mathbb { R } ^ { 1 4 }$ and active-frame direction $u ( t ) = v ( t ) / \| v ( t ) \|$ . Segment the trajectory by its dominant active subspace into ordered phases $P _ { 1 } \prec P _ { 2 } \prec \dots \prec P _ { K }$ , where phase k’s principal direction u¯<sub>k</sub> is the principal component of u(t) within that phase. The trajectory has Temporal Orthogonal Divisionof-Labor (TO-DoL) if the following three conditions hold simultaneously:

• (C1) Inter-phase orthogonality: max $_ { i \neq j } | \langle { \bar { u } } _ { i } , { \bar { u } } _ { j } \rangle | \leq \varepsilon ,$ i.e., diferent temporal phases occupy near-orthogonal joint directions.

• (C2) Non-trivial temporal order: the phase order $( P _ { 1 } \prec$ $\cdots \prec P _ { K } )$ and relative onset times are stable across trials and significantly better than a null control.

• (C3) Within-phase coherence: within a single phase, the direction u(t) is stable (low variance), so the phase genuinely “occupies” a subspace rather than wandering.

<table><tr><td>Test</td><td>Observation</td><td>Verdict</td></tr><tr><td> $\geq ~ 3$  subspaces co-active within a 10-frame chunk</td><td>0.0% (all datasets)</td><td>four almost never &quot;many things</td></tr><tr><td>Cross-arm co-activation 14% / 13%</td><td></td><td> $\mathrm { a t \ o n c e ^ { \gamma } }$  below ran-</td></tr><tr><td>rate</td><td></td><td>dom control (27–30%)</td></tr><tr><td>Left/right direction corr. in —0.24 to —0.37 active frames (handover)</td><td></td><td>one arm moves, the other pauses</td></tr></table>

Table 12: Same-instant orthogonal coupling is near-zero.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Left→right phase-order rate (handover)</td><td>100% of episodes</td></tr><tr><td>Gripper-event order (L-grasp &lt; R-grasp &lt; L-release) Grasp→release interval</td><td>100% (both em- bodiments) 22 frames, std =</td></tr><tr><td></td><td>0 3.5-5×</td></tr><tr><td>Envelope-preserving,</td><td>order- real</td></tr><tr><td></td><td></td></tr><tr><td>shuffled clean control</td><td></td></tr><tr><td></td><td>tighter</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>than</td></tr><tr><td></td><td>control</td></tr></table>

Table 13: C2 evidence: the temporal order is stable and beats the clean control.

C2 is the crux. In high dimensions any two sparse directions with disjoint support are already near-orthogonal, so C1 is nearly geometrically trivial and is not evidence on its own; what genuinely needs data support is C2—the ordering must beat controls that “shufle the phase order” and “randomly pair across trials”.

## H.3 Condition-by-Condition Verification

C1 (inter-phase orthogonality) — holds, but is a trivial baseline. The 14-D direction cosine between the leftdominant and right-dominant segments: aloha-agilex |cos| = 0.0013 (p90 ≈ 0.0017), arx-x5 = 0.0014 (p90 ≈ 0.002). Disjoint coordinate blocks are almost exactly orthogonal by geometry, so the weight of evidence rests on C2.

C2 (non-trivial temporal order) — holds strongly (Table 13). Two honest corrections must be stated. (1) The efect size is 3.5–5×, not higher. A “random-phase” control destroys the entire motion envelope (not just the ordering), inflating the efect size to $9 { - } 3 7 { \times } ;$ using an envelopepreserving, order-only-shufled clean control, the real order structure is $3 . 5 \mathrm { - 5 } \times$ tighter, and both the main paper and this section quote the latter. (2) The coupling sits at the “population-template” level, not per-trial feedback. Randomly pairing across episodes (a left-arm trajectory with any right-arm trajectory) almost perfectly reproduces the ordering coupling $( \mathrm { r a t i o } \approx 1 . 0 )$ . This means each arm follows its own fixed “temporal script” and the two scripts happen to be staggered—a shared sequential template rather than one arm closed-loop-adjusting to the other in real time. C2 still holds (the order is significant, stable, and beats the clean control), but is more accurately described as a population-level, script-level temporal coupling.

C3 (within-phase coherence) — essentially holds. Each arm’s direction is stable within its dominant phase; the handover left arm is bimodal (grasp at t ≈ 0, release at $t \approx 0 . 8 )$ but both peaks lie in the left-arm subspace and do not break C3—they merely split the left-arm phase into two segments.

## H.4 Robustness

• Across embodiments: aloha-agilex and arx-x5 give same-order numbers and consistent conclusions (C1 |cos| 0.0013 vs 0.0014; C2 order 100% of episodes; deterministic gripper timing lock).

• Across tasks: handover (sequential hand-of) strongly satisfies C2 (clear ordering); stack (left–right symmetric, no strict order, 48–50%) weakens C2 but still satisfies C1/C3. The “order” strength of C2 is thus task-dependent—the main paper states C2 as “a stable temporal-phase separation exists”, not universally “strict order”.

• Across datasets: LIBERO (end-efector delta, single arm) has no left/right-arm structure and TO-DoL does not apply; but it corroborates the triviality of “same-instant orthogonality”, which does not conflict with this definition.

## I Identity Initialization: Formal Properties

Section 4.4 of the main paper constructs the identity initialization from a zero-initialized residual; here we give its formal statement. Let $f _ { \theta }$ be the base policy and $f _ { \theta , \phi }$ its TFGCA-augmented version with module parameters ϕ and output projection $W _ { O } = 0$ . Then $\tilde { H } = H + W _ { O } ( A V ) = H$ giving two complementary properties.

• (a) Safe (non-decreasing capacity). $f _ { \theta , \phi } = f _ { \theta }$ pointwise, so the base policy lies in the augmented hypothesis class and min $1 _ { \phi } \bar { \mathcal { L } ( } f _ { \theta , \phi } \bar { ) } \le \bar { \mathcal { L } ( } f _ { \theta } )$ : attaching TFGCA cannot worsen the best attainable fit, and it changes no output at initialization.

• (b) Trainable (escaping the identity). The initial gradient of the loss with respect to $W _ { O }$ is $\mathbf { \bar { \nabla } } \nabla _ { W _ { \mathcal { O } } } \mathcal { L } = \delta ( \bar { A } V ) ^ { \top }$ where δ is the upstream gradient at the module output. As long as the value tokens satisfy $V \neq 0$ (and $\delta \neq 0 )$ , this gradient is generically nonzero, so $W _ { O }$ leaves zero and the module begins to act; if $V = 0$ then $A V = 0$ forces $\nabla _ { W _ { O } } { \mathcal { L } } = 0$ , and the gradient to the upstream wavelet parameters also vanishes through $W _ { O } = 0 ,$ , trapping the module at the identity. This is why we initialize the subband embeddings and the action-space projection to be nonzero and zero only $W _ { O }$

Proof. With $W _ { O } = 0$ we have $\tilde { H } = H$ and the head and backbone are unchanged; (a) follows by evaluating the augmented class at $W _ { O } = 0$ , and (b) follows by diferentiating $\tilde { H } = H + W _ { O } ( A V )$ with respect to $W _ { O } . \boxed { 1 }$

Safety and efectiveness are two diferent things. These properties guarantee function preservation at the moment of initialization—the module reproduces the base output pointwise on attachment and does not decrease capacity, a statement about attachment risk. They neither guarantee that the jointly fine-tuned behavior is no worse than the base (there are indeed a few per-task regressions on clean, Section 5.3 of the main paper) nor constitute evidence that the module is effective; efectiveness can only be supported by post-training empirical results (Section $5 ^ { \circ }$ of the main paper). The two should not be conflated.