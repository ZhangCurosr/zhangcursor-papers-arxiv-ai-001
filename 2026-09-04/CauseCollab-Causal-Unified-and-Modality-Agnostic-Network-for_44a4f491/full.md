# CauseCollab: Causal Unified and Modality-Agnostic Network for Heterogeneous Collaborative Perception

Weize Li <sup>\*</sup> <sup>1</sup> Yang Li <sup>\*</sup> <sup>1</sup> Quan Yuan <sup>1</sup> Xiaoyuan Fu <sup>1</sup> Guiyang Luo <sup>1</sup> Jinglin Li <sup>1</sup>

## Abstract

Collaborative perception enhances environment understanding through multi-agent information sharing, but its performance in real-world scenarios is constrained by heterogeneous sensor modalities and model architectures. Recent protocol-based two-stage methods alleviate this problem by mapping heterogeneous features into a shared protocol space; however, independently trained modality-specific converters often generate modality-specific pseudo-protocol distributions, leading to semantic inconsistency and error accumulation, which is particularly pronounced in scenarios with large modality discrepancies. To address this issue, we propose CauseCollab, a causal unified and modality-agnostic network. CauseCollab formulates representation learning in the protocol space from a causal perspective, explicitly disentangling semantic factors from modality-specific statistical confounders via causal metric learning. Meanwhile, CauseCollab adopts context-guided Unified Converter for heterogeneous modalities to ensure cross-modal semantic consistency. In addition, integrating new modalities only requires training adapters with minimal parameters. Extensive experiments on the OPV2V and DAIR-V2X datasets demonstrate that CauseCollab achieves state-of-the-art performance, with more significant gains in scenarios involving large modality gaps.

## 1. Introduction

Collaborative perception shares intermediate features among multiple agents through communication, enabling them to complement each other’s perceptual blind spots, and is widely regarded as a key technique for improving the robustness of autonomous driving systems (Arnold et al., 2022; Han et al., 2023; Liu et al., 2023). In recent years, extensive research has focused on enhancing the performance of homogeneous collaboration (Liu et al., 2020; Hu et al., 2022; Xu et al., 2022b). However, in real-world deployments, collaborative agents often originate from different manufacturers and platforms, exhibiting substantial differences in sensor types, voxel sizes, and network architectures. It makes intermediate features difficult to fuse and limits the practical effectiveness of collaborative perception.

To address heterogeneity in collaborative perception, prior work has explored feature transformation modules that map heterogeneous modality features from neighboring agents into the local space of the ego agent (Zhao et al., 2023; Yin et al., 2024). Existing approaches can be broadly categorized into two classes. The first class (Xu et al., 2023; Lu et al., 2024) learns a dedicated mapping for each pair of heterogeneous modalities to achieve one-stage alignment, but suffers from poor scalability as the training cost grows rapidly with the number of modalities. The second (Luo et al., 2024; Gao et al., 2025) class introduces a unified protocol modality as an intermediate representation space, and trains a converter and a reconstructor for each modality to map heterogeneous features into this protocol space, achieving improved scalability while maintaining strong performance.

Although two-stage heterogeneous adaptation methods based on a protocol space exhibit favorable scalability, their performance is often constrained by error accumulation across stages. When the converters are trained independently for each modality, each modality tends to learn a locally self-consistent mapping between its local representation and the protocol representation. This behavior leads to inconsistent pseudo-protocol distributions in the protocol space. As a result, even after all features are projected into a unified protocol space, significant discrepancies in semantic responses and statistical distributions may persist across modalities, which weakens the effectiveness of local semantic reconstruction.

This issue becomes particularly pronounced when the modality gap is large, where pseudo-protocol distributions often introduces noisy responses and spurious activations in background regions. The root cause is that existing methods typically model the protocol space merely as a distributionalignment intermediary, failing to explicitly characterize which semantic factors should be shared across modalities and which modality-specific statistical confounders should be disentangled.

To address this problem, we propose a unified two-stage framework from a causal perspective. Unlike previous methods that treat the protocol space as a reversible intermediate and emphasize distribution matching, our framework focuses on semantic extraction and modal factor disentanglement in Stage-1, followed by modality-specific local semantic reconstruction in Stage-2.

Specifically, in Stage-1, we adopt a unified converter composed of a semantic context extractor (SCE) and a contextguided dynamic refiner (CGDR) for multi-modality mixed training. By taking global semantics as an anchor to constrain local feature refinement, we reduce backgroundinduced misalignment. Meanwhile, we construct a structural causal model (SCM) (Scholkopf¨ , 2022), depicted in Figure 2, to characterize the relationships among heterogeneous modalities in terms of semantic components. Through a counterfactual intervention strategy based on causal metric learning, we explicitly drive the protocol representations of each modality to approximate the weakly observed protocol modality. This suppresses pseudo-protocol shifts induced by residual modal biases and background noise, thus achieving cross-modal unified semantic alignment. We design the Mask-Guided Intervention via SPD Feature Geometry (MGI-SPD) module to enforce causal interventions. Based on the semantically consistent protocol space, Stage-2 independently trains a local semantic reconstructor for each modality to restore feature semantics to the local modality. When a new modality is introduced, we freeze the backbone of the unified converter and only train lightweight adapters to project the new modality into the semantically consistent protocol space, thus achieving expansion at the cost of minimal parameters.

• We propose CauseCollab, a causal unified and modalityagnostic network for heterogeneous collaborative perception(HCP). It disentangles modality-specific statistical confounders during the conversion, and realizes the learning of a semantically consistent protocol space based on causal metric learning. For intermediate features, we design a module called Mask-Guided Intervention via SPD Feature Geometry.

• We design a unified converter composed of a semantic context extractor and a context-guided dynamic refiner to ensure cross-modal semantic consistency. A lightweight adapter is incorporated to enable the adaptation of new modalities at the cost of minimal parameters.

• Extensive experiments on the OPV2V and DAIR-V2X datasets demonstrate that CauseCollab achieves state-ofthe-art (SOTA) performance among existing heterogeneous collaborative perception methods, with substantial improvements in perception accuracy in scenarios with larger modality gaps.

## 2. Related Work

## 2.1. Collaborative Perception

Collaborative perception allows multiple agents to share and fuse multi-view information, mitigating the inherent view limitations and occlusions of single-agent perception. Intermediate fusion (Li et al., 2021; Xu et al., 2022a) has become the dominant paradigm in collaborative perception, as it preserves rich intermediate semantics while avoiding the high bandwidth requirements and strict synchronization constraints of early fusion. Methods such as Where2comm (Hu et al., 2022) and CodeFilling (Hu et al., 2024) address communication bottlenecks through structured feature compression, while CoAlign (Lu et al., 2023), CoBEVFlow (Wei et al., 2023), and CoDiff (Huang et al., 2025) focus on mitigating feature misalignment caused by localization errors and temporal delays.

Existing collaborative perception methods excel in homogeneous multi-agent settings, yet real-world deployments inevitably require heterogeneous agents. Recently, several works (Xiang et al., 2023; Shao et al., 2024) have attempted to address heterogeneity in collaborative perception. MPDA (Xu et al., 2023) and PnPDA (Luo et al., 2024) employ one-stage and two-stage neighbor-to-ego translators to map features into the ego agent’s semantic space. HEAL (Lu et al., 2024) performs one-to-one mappings via backward alignment, while STAMP (Gao et al., 2025) introduces a protocol space with reversible mappings between local and protocol representations. NegoCollab (Shao et al., 2025) learns a general representation through knowledge distillation. In contrast to prior approaches, our method aims to eliminate pseudo-protocol distributions induced by heterogeneous modalities within the protocol space and to learn a semantically consistent unified protocol representation.

## 2.2. Causal Inference

Causal inference aims to distinguish true causal factors from spurious correlations, characterize causal relationships among variables, and estimate intervention effects that are free from confounding factors. In recent years, causal inference has been introduced into computer vision tasks (Wang et al., 2024; Li et al., 2024) to improve model robustness. CIIM (Yan et al., 2024) explores causally invariant interactions to learn modality-consistent embeddings. CWNet (Zhang et al., 2025) applies causal reasoning to low-light image enhancement, emphasizing semantic information during the enhancement process. MSFT (Qiao et al., 2025) incorporates causal intervention into time-series forecasting to address confounding effects induced by varying temporal scales. We introduce structural causal models into collaborative perception and design a novel intervention module operating on intermediate features, enabling more robust cross-modal semantic alignment.

![](images/b95e3877c415621262a3fe41798f7f5e0272ef4be1b8083599a2d68ea0aadb93.jpg)  
Figure 1. The overall architecture of CauseCollab. CauseCollab consists of a modality-agnostic protocol semantics conversion stage via causal modeling (Stage-1) and a modality-specific local semantic reconstruction stage (Stage-2). Unified Converter incorporates SCE and CGDR modules to derive semantically consistent representations.

## 3. Method

## 3.1. Overview Pipeline

The overall architecture of CauseCollab is illustrated in Figure 1.We focus on the heterogeneous collaborative perception setting, where the ego agent ε receives observations from multiple collaborative agents $\mathcal { V } = \{ 1 , \ldots , N \}$ during perception. Since collaborative agents may belong to different sensor modalities, the modality set is denoted as M, and directly performing feature fusion suffers from significant modality discrepancies.

For any collaborative agent $i \in \mathcal V$ , its raw observation is denoted as $x _ { i } ^ { m _ { i } } \in \mathcal { X } ^ { m _ { i } }$ , where $m _ { i } ~ \in ~ { \mathcal { M } }$ indicates the sensor modality of agent i. The ego agent observation is denoted as $x _ { \varepsilon } ^ { m _ { \varepsilon } }$ . For each modality $m ,$ , we introduce a modality-specific encoder $E ^ { m } ( \cdot )$ to map raw inputs into

feature representations:

$$
f _ { i } = E ^ { m _ { i } } ( x _ { i } ^ { m _ { i } } ) , \quad f _ { \varepsilon } = E ^ { m _ { \varepsilon } } ( x _ { \varepsilon } ^ { m _ { \varepsilon } } ) ,\tag{1}
$$

where $f _ { i } , f _ { \varepsilon } \in \mathbb { R } ^ { C \times H \times W }$ denote the modality features of collaborative agents and the ego agent, respectively.

In Stage-1, we introduce a shared protocol space $\mathcal { P }$ and project collaborative agent features into this space to obtain unified semantic representations, where each collaborative feature is transformed into its protocol representation $p _ { i } \in$ $\mathcal { P } _ { \cdot }$

In Stage-2, the ego agent employs a reconstructor $R ^ { m _ { \varepsilon } } ( \cdot )$ to map protocol features back to local space ${ \mathcal { L } } ,$ yielding reconstructed collaborative features $\tilde { f } _ { i } \in \mathcal { L } .$ The ego feature $f _ { \varepsilon }$ and all reconstructed collaborative features $\{ \bar { \tilde { f } _ { i } } \} _ { i = 1 } ^ { N }$ are then fused by a pyramid fusion module $\Phi _ { \varepsilon } ( \cdot )$ to obtain multi-scale fused feature:

$$
F _ { \mathrm { p y r a m i d } } = \Phi _ { \varepsilon } ( f _ { \varepsilon } , \tilde { f } _ { 1 } , \dots , \tilde { f } _ { N } ) .\tag{2}
$$

Finally, the fused feature are fed into the task head $H _ { \varepsilon } ( \cdot )$ to produce the final detection output:

$$
z = H _ { \varepsilon } ( F _ { \mathrm { p y r a m i d } } ) .\tag{3}
$$

![](images/a72e3c1dd5f54625f7dea124b008619e0cba431555362bf463c574faf9b7cebc.jpg)  
Figure 2. Representation-level Structural Causal Model(SCM) for Two-Stage Heterogeneous Collaborative Perception. Black edges denote causal paths, while red edges denote spurious correlation paths. Solid edges represent direct effects, and dashed edges indicate indirect effects on the protocol-space representation.

## 3.2. Stage-1: Protocol Semantic Conversion via Causal Modeling

Structural Causal Modeling. To distinguish modalityinvariant semantic factors from modality-specific statistical confounders that should be suppressed, we introduce Structural Causal Modeling (SCM) in Stage-1 to explicitly characterize the causal generation process of representations. As illustrated in Figure 2, the variables include the protocolmodality representation $P ,$ , the heterogeneous modality representation $F _ { m }$ , the protocol-space heterogeneous representation $F _ { m  p } ,$ the locally reconstructed representation $F _ { m  p  m } ,$ , the true semantic factor S, the protocol-modality statistical noise $N _ { p }$ , and the heterogeneous-modality statistical noise $N _ { m }$

Without proper constraints, the statistical noises $N _ { p }$ and $N _ { m }$ can affect $F _ { m }$ and P through the encoder, thereby inducing spurious correlation paths in the protocol space that are irrelevant to the true semantics S. Consequently, the model may exploit modality-style differences as discriminative cues, leading to modality-specific spurious semantic clusters in $F _ { m  p } .$ To obtain a semantically consistent protocol space, we naturally regard S as the causal factor, while treating $N _ { p }$ and $N _ { m }$ as confounders.

Based on the above SCM, the learning objective of Stage-1 can be stated as follows: while maintaining high sensitivity of $F _ { m  p }$ to the semantic factor S, we reduce its dependence on modality statistical noises $N _ { p }$ and $N _ { m }$ . Since S is not directly observable, we use the protocol-modality representation P as a weak observation during training, and introduce SPD mask-guided intervention together with causal metric learning to achieve the above causal objective.

Mask-Guided Intervention via SPD Feature Geometry. After heterogeneous features are transformed into the protocol space, they still retain significant modality-specific statistical shifts, which are mainly reflected in the discrepancies of first-order means and second-order correlation structures of channel distributions (Huang & Belongie, 2017; Li et al., 2017). To address this issue, we design Mask-Guided Intervention via SPD Feature Geometry, and realize the separation of confounders through causal metric learning. Specifically, we regard Bird’s-Eye View (BEV) features as a collection of high-dimensional random vectors sampled from various spatial locations, and construct closed-form statistical mappings on the geometry of symmetric positive definite (SPD) covariance matrices (Huang & Gool, 2017); meanwhile, we introduce semantic mask-guided spatially adaptive strength modulation to perform differentiated statistical suppression and statistical injection on foreground and background regions, thereby implementing counterfactual intervention on $N _ { p }$ and $N _ { m }$ without disrupting the feature semantic factor S.

Given a protocol feature map $p _ { i } \in \mathbb { R } ^ { C \times H \times W }$ , we reshape it into $\bar { X _ { i } } \in \mathbb { R } ^ { C \times N }$ with $N = H \times W$ , where each column vector $x _ { n } \in \mathbb { R } ^ { C }$ represents the channel feature at the corresponding spatial location. Under this representation, modality style is modeled by the first-order and secondorder statistics $( \mu , \Sigma )$ of the feature set X.

To characterize the discrepancy in semantic contributions between foreground and background regions, we introduce a semantic mask $m _ { i } \in [ 0 , 1 ] ^ { \bar { 1 } \times N }$ generated by inputting heterogeneous features into a mask generator $g ( \cdot )$ , and construct non-negative weights w based on this mask to define weighted statistics:

$$
\mu = \frac { \sum _ { n = 1 } ^ { N } w _ { n } x _ { n } } { \sum _ { n = 1 } ^ { N } w _ { n } } ,\tag{4}
$$

$$
\Sigma = \frac { \sum _ { n = 1 } ^ { N } w _ { n } ( x _ { n } - \mu ) ( x _ { n } - \mu ) ^ { \top } } { \sum _ { n = 1 } ^ { N } w _ { n } } + \varepsilon I ,\tag{5}
$$

where $\Sigma \in \mathcal { S } _ { + } ^ { C }$ lies on the SPD geometry, and εI is added to ensure numerical stability.

On the SPD geometry, we adopt a whitening–coloring linear mapping to realize a closed-form transformation from source statistics $\left( \mu _ { s } , \Sigma _ { s } \right)$ to target statistics $\left( \mu _ { t } , \Sigma _ { t } \right)$ :

$$
T _ { s  t } ( X ) = \mu _ { t } + \Sigma _ { t } ^ { \frac { 1 } { 2 } } \Sigma _ { s } ^ { - \frac { 1 } { 2 } } ( X - \mu _ { s } ) ,\tag{6}
$$

where $\Sigma ^ { 1 / 2 }$ and $\Sigma ^ { - 1 / 2 }$ are computed via eigendecomposition of SPD matrices to ensure numerical stability and preserve positive definiteness. The mapping performs a linear adjustment of the feature mean and correlation structure along the channel dimension only, without altering the spatial structure of the feature map, thereby guaranteeing that the BEV semantics are not compromised.

Under the above formulation, we apply counterfactual intervention to the protocol feature $f _ { p }$ and construct a pair of positive and negative samples for causal metric learning: the demodalized protocol feature $p _ { i } ^ { + }$ serves as the positive sample, and the confounded protocol feature $\left. p _ { i } ^ { - } \right.$ acts as the negative sample. Specifically, $p _ { i } ^ { + }$ is obtained by mapping the channel statistics $\left( \mu _ { c } , \Sigma _ { c } \right)$ of $f _ { p }$ to a neutral target distribution $\left( \mu _ { 0 } , \Sigma _ { 0 } \right)$ via whitening coloring transformation combined with semantic mask-guided spatial interpolation, thereby suppressing modality-specific statistics while preserving the spatial semantic structure; in contrast, after completing the aforementioned suppression, $\boldsymbol { p } _ { i } ^ { - }$ is generated by introducing controlled injection of heterogeneous modality statistics $\left( \mu _ { h } , \Sigma _ { h } \right)$ through isomorphic statistical mapping and spatial modulation, such that modality-induced statistical perturbations are mixed in while maintaining semantic consistency. Through this intervention construction based on SPD statistical geometry, $p _ { i } ^ { + }$ provides a demodalized semantic positive sample, while $\boldsymbol { p } _ { i } ^ { - }$ acts as a hard negative sample with controlled modality shifts introduced, jointly driving the unified converter to learn modality-agnostic protocol representations. It is worth noting that MGI-SPD is used only during Stage-1 training to remove modality-specific confounding factors; it is not involved during inference. More details can be found in Appendix B.1.

Semantic Context Extractor. To learn representations with stronger semantic consistency in the protocol space, we design a semantic context extractor $G ( \cdot )$ for extracting highlevel semantic information. This module takes the modality feature $f _ { i } ~ \in ~ \mathbb { R } ^ { C \times H \times W }$ generated by the heterogeneous modality encoders as input, and outputs the global semantic context feature $c _ { i } \in \mathbb { R } ^ { \bar { C } _ { g } \times H _ { g } \times W _ { g } }$

G(·) adopts a hierarchical structure, and gradually enhances the global semantic modeling capability through two-stage downsampling and semantic extraction. Specifically, we construct two levels of semantic representations:

$$
\begin{array} { r } { \boldsymbol { x } ^ { ( 1 ) } = \mathcal { B } _ { 1 } ( \mathcal { D } _ { 1 } ( f _ { i } ) ) , \quad \boldsymbol { x } ^ { ( 2 ) } = \mathcal { B } _ { 2 } ( \mathcal { D } _ { 2 } ( \boldsymbol { x } ^ { ( 1 ) } ) ) , } \end{array}\tag{7}
$$

and fuse the two-level features to obtain the context representation:

$$
c _ { i } = \mathrm { F u s e } ( x ^ { ( 1 ) } , x ^ { ( 2 ) } ) .\tag{8}
$$

where $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 }$ denote downsampling convolutions, $B _ { 1 } , B _ { 2 }$ are semantic extraction blocks, and Fuse(·) is the feature fusion operator. Specifically, Fuse(·) first aligns the channel dimensions using $1 \times 1$ convolutions, performs spatial alignment via interpolation, and then applies weighted concatenation followed by a depth-wise $3 \times 3$ convolution for spatial mixing. Finally, we apply channel recalibration, implemented by the classical Squeeze-and-Excitation module, and normalization to $c _ { i }$ at the output stage to enhance task-relevant semantic responses and stabilize the feature distribution, thereby providing reliable global semantic context for subsequent dynamic refinement.

Context-Guided Dynamic Refiner. In Stage-1, we introduce the Context-Guided Dynamic Refiner $T ( \cdot )$ . This module performs layer-wise semantic correction on local modality features under semantic context constraints, thereby generating protocol-space representations with stronger semantic consistency.

Given the local modality feature $f _ { i } \in \mathbb { R } ^ { C \times H \times W }$ of the i-th agent and the semantic context feature $c _ { i }$ extracted by the Semantic Context Extractor (SCE), we first align the context feature in both channel and spatial dimensions to obtain the conditional context representation:

$$
{ \hat { c } } _ { i } = U { \left( \phi ( c _ { i } ) \right) } ,\tag{9}
$$

where $\phi ( \cdot )$ denotes a $1 \times 1$ convolution and $U ( \cdot )$ represents linear interpolation. Subsequently, the refiner takes $( f _ { i } , \hat { c } _ { i } )$ as joint input and progressively generates protocol-space features through a stack of context-conditioned transformations:

$$
p _ { i } = T ( f _ { i } , { \hat { c } } _ { i } ) .\tag{10}
$$

In practice, the refiner is composed of multiple dynamic refinement blocks. For the ℓ-th layer, the update of local modality features is conditioned on the aligned context representation, which is formulated as:

$$
f _ { i } ^ { ( \ell + 1 ) } = T _ { \ell } \big ( f _ { i } ^ { ( \ell ) } , \hat { c } _ { i } \big ) .\tag{11}
$$

Within each dynamic refinement block, local features are first modulated by FiLM-style multi-head feature modulation under the guidance of semantic context, and then filtered by a gating function gate(·), implemented as a $1 \times 1$ convolution followed by SiLU activation, to suppress noninformative regions and noise responses, thereby stabilizing the feature transformation process.

By continuously enforcing semantic context constraints throughout the transformation, the resulting protocol representation $p _ { i }$ not only preserves local discriminative details but also anchors to consistent global semantics, providing reliable inputs for local semantic reconstruction and task prediction in Stage-2.

Training Strategy. Stage-1 aims to learn semantically consistent protocol representations: this stage enhances the unified semantic factor S across modalities while suppressing the interference of modality-specific statistical noise $N _ { p }$ and $N _ { m }$ , thereby reducing pseudo-protocol shifts and erroneous activations in background regions.

The overall objective function of Stage-1 is defined as fol-

Table 1. Performance comparison on the OPV2V dataset using AP@30/AP@50, where the ego agent is $L _ { \mathrm { p p 4 } }$ . ALL denotes the scenario with all collaborative agents.
<table><tr><td rowspan="2">Method</td><td colspan="4"> $\mathbf { A P } @ 3 \mathbf { 0 } \uparrow$ </td><td colspan="4">AP@50 ↑</td></tr><tr><td> $+ C _ { \mathrm { E f f } }$ </td><td> $+ L _ { s d 1 }$ </td><td> $+ C _ { \mathrm { R e s } }$ </td><td>ALL</td><td> $+ C _ { \mathrm { E f f } }$ </td><td> $+ \cal { L } _ { s d 1 }$ </td><td> $+ C _ { \mathrm { R e s } }$ </td><td>ALL</td></tr><tr><td>MPDA</td><td>0.947</td><td>0.967</td><td>0.967</td><td>0.969</td><td>0.939</td><td>0.962</td><td>0.962</td><td>0.966</td></tr><tr><td>PnPDA</td><td>0.936</td><td>0.965</td><td>0.966</td><td>0.970</td><td>0.931</td><td>0.962</td><td>0.963</td><td>0.967</td></tr><tr><td>HEAL</td><td>0.941</td><td>0.958</td><td>0.959</td><td>0.963</td><td>0.935</td><td>0.954</td><td>0.955</td><td>0.960</td></tr><tr><td>STAMP</td><td>0.945</td><td>0.967</td><td>0.968</td><td>0.972</td><td>0.940</td><td>0.964</td><td>0.965</td><td>0.969</td></tr><tr><td>CauseCollab</td><td>0.952</td><td>0.972</td><td>0.973</td><td>0.978</td><td>0.944</td><td>0.968</td><td>0.969</td><td>0.975</td></tr></table>

lows:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { S t a g e - 1 } } = \lambda _ { \mathrm { c o n } } { \mathcal { L } } _ { \mathrm { c o n t r a s t } } + \lambda _ { \mathrm { c t x } } \big ( { \mathcal { L } } _ { \mathrm { c t x - s s i m } } + { \mathcal { L } } _ { \mathrm { c t x - c o s } } \big ) } \\ & { ~ + \lambda _ { \mathrm { s s i m } } { \mathcal { L } } _ { \mathrm { s s i m } } + \lambda _ { \mathrm { t a s k } } ^ { ( 1 ) } { \mathcal { L } } _ { \mathrm { t a s k } } ^ { p } , } \end{array}\tag{12}
$$

where $\lambda _ { \mathrm { c o n } } , \lambda _ { \mathrm { c t x } } , \lambda _ { \mathrm { s s i m } }$ and $\lambda _ { \mathrm { t a s k } } ^ { ( 1 ) }$ are weighting coefficients.

Among these loss terms, ${ \mathcal { L } } _ { \mathrm { c o n t r a s t } }$ is an InfoNCE-style contrastive loss (van den Oord et al., 2019) for causal metric learning, where $p _ { i }$ is used as the anchor, the demodalized intervention feature $p _ { i } ^ { + }$ serves as the positive sample, and the confounded intervention feature $\mathrm { \Delta } p _ { i } ^ { - }$ serves as the negative sample; $\mathcal { L } _ { \mathrm { c t x - s s i m } }$ and $\mathcal { L } _ { \mathrm { c t x - c o s } }$ are semantic context consistency constraints applied between the semantic context feature $c _ { i }$ and the downsampled positive intervention feature $p _ { i } ^ { + }$ , aligning high-dimensional semantics from the perspectives of spatial structure and channel response, respectively; $\mathcal { L } _ { \mathrm { s s i m } }$ enforces semantic consistency between the protocol-modality feature $f _ { p }$ and the converted protocol representation $p _ { i } ,$ constraining dynamic refinement to achieve protocol mapping while preserving spatial semantic structure; $\mathcal { L } _ { \mathrm { t a s k } } ^ { p }$ is the protocol-head detection loss, implemented as the focal loss plus the smooth $\mathcal { L } _ { 1 }$ loss, ensuring protocol representations have both semantic consistency and sufficient discriminability for downstream detection tasks.

## 3.3. Stage-2: Local Semantic Reconstruction

After Stage-1, features from different modalities are mapped into a semantically consistent protocol space. Since heterogeneous modalities differ in resolution and representation characteristics, Stage-2 reconstructs protocol back to each modality-specific local space to fit modality-specific heads.

With the unified converter in Stage-1 frozen, we train a modality-specific local reconstructor $R ^ { m } ( \cdot )$ for each modality m to map features back to the local feature space:

$$
{ \tilde { f } } _ { i } = R ^ { m } ( p _ { i } ) ,\tag{13}
$$

where $p _ { i }$ denotes the protocol-space feature produced by Stage-1. The reconstructor is designed following the ConvNeXt (Liu et al., 2022) architecture to ensure stable optimization.

For modality m, Stage-2 is optimized with task supervision and consistency constraints. The overall objective is:

$$
\mathcal { L } _ { \mathrm { S t a g e \it { - 2 } } } = \lambda _ { \mathrm { t a s k } } ^ { ( 2 ) } \mathcal { L } _ { \mathrm { t a s k } } ^ { \epsilon } + \lambda _ { \mathrm { m s e } } \mathcal { L } _ { m s e } + \lambda _ { \mathrm { c o n s } } \mathcal { L } _ { c o n s } ,\tag{14}
$$

where $\lambda _ { \mathrm { t a s k } } ^ { ( 2 ) } , \lambda _ { \mathrm { m s e } }$ , and $\lambda _ { \mathrm { c o n s } }$ are weighting coefficients. $\mathcal { L } _ { \mathrm { t a s k } } ^ { \epsilon }$ is the modality-specific task loss for downstream task supervision; $\mathcal { L } _ { m s e }$ is the MSE loss between reconstructed and original local features; $\mathcal { L } _ { c o n s }$ is the consistency loss to stabilize the mapping between protocol and local feature spaces.

## 3.4. Generalization

The initial collaborative agents have learned modalityagnostic causal semantics with consistent representations in the protocol space. For a new modality, we freeze the backbone of the unified converter in Stage-1 and only employ lightweight adapters to align the new modality to the established protocol-consistent space; in Stage-2, we further train a local semantic reconstructor to recover local space.

The lightweight adaptation in Stage-1 is achieved by inserting a small number of bottleneck adapters at key locations in the unified converter, while keeping the backbone parameters frozen. The adapters are placed after the downsampling operations in the SCE and between modules of the CGDR, so that the new modality is progressively aligned to modality-invariant semantics through scale transitions and fine-grained conversions. Each adapter follows a low-rank bottleneck residual design and is initialized with a scaling coefficient $\alpha = 0$ , enabling effective compensation for the new modality with negligible parameter overhead while preserving the original protocol semantic consistency.

## 4. Experiments

## 4.1. Settings

Datasets. We conduct experiments on OPV2V (Xu et al., 2022c) and DAIR-V2X (Yu et al., 2022). OPV2V is a simulator-collected multi-agent dataset tailored for heterogeneous collaborative perception. DAIR-V2X is a real-world V2X dataset collected in traffic environments, consisting of both agent-side and roadside LiDAR and camera data.

Table 2. Experiments of different methods on the OPV2V dataset under the setting of larger domain gap, along with the parameters required for new modality adaptation. The notation A + B indicates that the ego agent is configured with modality A, while all neighbo agents are configured with modality B. Model parameters are measured in megabytes (MB).
<table><tr><td rowspan="2">Method</td><td colspan="4">AP@30/AP@50↑</td><td rowspan="2">#Parmas (New Modality)</td></tr><tr><td> $L _ { p p 4 } + C _ { \mathrm { E f f } }$ </td><td> $C _ { \mathrm { E f f } } + C _ { \mathrm { R e s } }$ </td><td> $L _ { p p 4 } + L _ { s d 2 } \mathrm { { n e w } }$ </td><td> $L _ { s d 2 ^ { \mathrm { n e w } } } + C _ { \mathrm { E f f B 1 ^ { \mathrm { n e w } } } }$ </td></tr><tr><td>MPDA</td><td>0.967 / 0.961</td><td>0.711 / 0.597</td><td>0.949 / 0.943</td><td>0.941 / 0.937</td><td>5.00</td></tr><tr><td>PnPDA</td><td>0.966 / 0.962</td><td>0.654 / 0.597</td><td>0.980 / 0.979</td><td>0.951 / 0.946</td><td>4.37</td></tr><tr><td>HEAL</td><td>0.960 / 0.956</td><td>0.627 / 0.550</td><td>0.980 / 0.978</td><td>0.936 / 0.932</td><td>17.10</td></tr><tr><td>STAMP</td><td>0.961 / 0.957</td><td>0.730 / 0.634</td><td>0.981 / 0.980</td><td>0.958 / 0.952</td><td>1.18</td></tr><tr><td>CauseCollab</td><td>0.971 / 0.966</td><td>0.772 / 0.670</td><td>0.983 / 0.980</td><td>0.966 / 0.959</td><td>0.68</td></tr></table>

Table 3. Performance comparison of heterogeneous collaboration on real-world datasets DAIR-V2X.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=2>AP@30/AP@50↑ $L _ { \mathrm { p p 4 } } + C _ { \mathrm { E f f } }$ </td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>MPDA</td><td rowspan=1 colspan=1>0.792 / 0.685</td><td rowspan=1 colspan=1>0.405 / 0.228</td></tr><tr><td rowspan=1 colspan=1>PnPDA</td><td rowspan=1 colspan=1>0.790 / 0.738</td><td rowspan=1 colspan=1>0.408 / 0.235</td></tr><tr><td rowspan=1 colspan=1>HEAL</td><td rowspan=1 colspan=1>0.738 / 0.692</td><td rowspan=1 colspan=1>0.410 / 0.253</td></tr><tr><td rowspan=1 colspan=1>STAMP</td><td rowspan=1 colspan=1>0.782 / 0.731</td><td rowspan=1 colspan=1>0.394 / 0.249</td></tr><tr><td rowspan=1 colspan=1>CauseCollab</td><td rowspan=1 colspan=1>0.795 / 0.733</td><td rowspan=1 colspan=1>0.647 / 0.464</td></tr></table>

Experimental Details. We adopt heterogeneous feature encoders spanning different sensor types and network architectures. For LiDAR, we use PointPillar (Lang et al., 2019) and SECOND (Yan et al., 2018) as two representative encoders. For cameras, we use an image encoder based on the Lift-Splat-Shoot (LSS) (Philion & Fidler, 2020) framework, which employs ResNet (He et al., 2015) and EfficientNet (Tan & Le, 2019) as the backbone networks. Specifically, we construct four heterogeneous modality configurations, denoted as $L _ { \mathrm { p p 4 } } , C _ { \mathrm { E f f } } , L _ { \mathrm { s d 1 } }$ , and $C _ { \mathrm { R e s } }$ . These configurations differ in sensor types, voxel size, and backbone design. Detailed settings are provided in Table 6.

We employ a two-stage training scheme. In Stage-1, features extracted by the four heterogeneous encoders are mixed and fed into shared converter. The converter is trained with multi-modal mixed batches to align heterogeneous modality spaces into a unified protocol space. In Stage-2, we train a modality-specific local semantic reconstructor for each modality to recover collaborator features in its local representation space. When integrating a new modality, we freeze the shared converter backbone and only perform lightweight adaptation, together with training the local reconstructor for the new modality. Notably, during adaptation, the encoders, fusion modules, and classification heads of previously deployed modalities remain frozen and inaccessible, which satisfies practical deployment requirements on privacy preservation and non-modifiability of existing modalities.

![](images/9e65e500bff68a041068d6fabee8b9bbdae6cc57871261f4cea6fbde120c4a53.jpg)

![](images/26bf8ac19c009dab7a6ec254895d75c90475393b0ab0274d7bd9bfd08d0e1e7f.jpg)  
Figure 3. Robustness comparison of different methods under pose noise. We evaluate the AP@0.3 and AP@0.5 performance as the pose noise standard deviation increases from 0.0 to 0.5.

## 4.2. Performance Comparison

Collaboration on OPV2V. We evaluate our method on the 3D object detection task and compare it with prior approaches. We select two one-stage methods, MPDA (Xu et al., 2023) and HEAL (Lu et al., 2024), and two twostage methods, PnPDA (Luo et al., 2024) and STAMP (Gao et al., 2025). For a fair comparison, we adopt the same pyramid fusion strategy used in prior works (Lu et al., 2024; Gao et al., 2025). Table 1 reports the results on OPV2V under a general heterogeneous collaboration setting. We set the ego agent to the $L _ { \mathrm { p p 4 } }$ modality and progressively add three heterogeneous-modality neighbors for collaboration. CauseCollab achieves the best performance under both AP@0.3 and AP@0.5 metrics after each heterogeneous modality agent is progressively added.

Furthermore, we evaluate the robustness of the model against localization errors by injecting Gaussian noise with standard deviation σ ranging from 0.0 to 0.5 into agent poses, and the experimental results are shown in Figure 3. The results demonstrate that the performance of CauseCol-

(a) $L _ { \mathrm { p p 4 } }$ Feature

Table 4. Ablation study: Evaluating the impact of individual components on the performance of CauseCollab. Experiments are conducted under the modality configuration of $L _ { \mathrm { p p 4 } } + C _ { \mathrm { E f f } } +$ $L _ { \mathrm { s d 1 } } + C _ { \mathrm { R e s } . }$
<table><tr><td></td><td>AP@30↑</td><td>AP@50↑</td></tr><tr><td>CauseCollab</td><td>0.978</td><td>0.975</td></tr><tr><td>-w/o Causal Intervention</td><td>0.969</td><td>0.965</td></tr><tr><td>-w/o Mask Generator</td><td>0.972</td><td>0.970</td></tr><tr><td>-w/o Semantic Context</td><td>0.837</td><td>0.834</td></tr><tr><td>-w/o CGDR</td><td>0.972</td><td>0.969</td></tr><tr><td>-w/o SCE + CGDR</td><td>0.968</td><td>0.966</td></tr><tr><td>-w/o Lightweight Adapters</td><td>0.909</td><td>0.904</td></tr></table>

lab is better than or comparable to that of other methods at most noise levels.

To validate semantic consistency in the protocol space, we set the ego agent to one modality and neighbor agents to a single different modality, which emulates a larger modality gap for heterogeneous collaboration. We then evaluate the model’s robustness in this challenging scenario, with results provided in Table 2. When the ego is $C _ { \mathrm { E f f } }$ and the neighbor is $C _ { \mathrm { R e s } }$ , our method improves AP by 4.2% at IoU=0.3 and by 3.6% at IoU=0.5 compared to four other methods. Similarly, when the ego is $L _ { \mathrm { p p 4 } }$ and the neighbor is $C _ { \mathrm { E f f } } .$ , the AP gains are 0.5% and 0.4% under the two thresholds, with improvements even for the higher-performance combination. These results confirm that our method preserves more semantically consistent information during collaboration.

Collaboration on DAIR-V2X. We further conduct performance comparisons on the real-world DAIR-V2X dataset. Since DAIR-V2X only provides two agent perspectives for the same scene, we set the Ego agent to the $L _ { p p 4 }$ and C<sub>Ef</sub> modalities respectively for collaboration. As shown in Table 3, our method still outperforms all previous methods overall.

Scalability. Heterogeneous collaboration methods should be scalable to incorporate novel modalities. Starting from a unified converter trained on the four heterogeneous modalities above, we adapt it to two novel modalities, $L _ { \mathrm { s d } 2 }$ and $\mathrm { C _ { E f f B 1 } }$ , via fine-tuning. We then use the novel modality as the neighbor agent for collaboration, as shown in Table 2. We observe that, for novel modalities, our method achieves higher average precision with fewer trainable parameters, indicating that the unified converter enables semantic consistency and parameter reuse.

## 4.3. Ablation Study

We conducted a comprehensive ablation study on the OPV2V dataset. In Stage-1, we ablate each component of the model individually and evaluate the heterogeneous collaboration performance under the general collaboration setting. As shown in Table 4, removing any single component consistently leads to performance degradation.

![](images/c3b7965e091c65c528e34dd158c5768c8ed68983db9a21645d1a3a3c887fc251.jpg)

![](images/7bfae801239710baa39c1a5e5b0976512bd438dcb13a95db0f4278a0c04face7.jpg)

![](images/47b77b143bbf0cea02eafd39e0b011f58e1e42f32a3dfe5b959f5ed2fd5896ef.jpg)  
(c) C<sub>Ef</sub> Feature in Protocol Space (STAMP)

![](images/62ad85e7f871c1b48d1ca221a1973bec76733b323c40039a3034745796ff9f45.jpg)  
(d) C<sub>Ef</sub> Feature in Protocol Space (CauseCollab)  
Figure 4. Visualization of intermediate features before and after conversion.

Specifically, after removing the causal intervention module, the AP metrics at IoU thresholds of 0.3 and 0.5 decrease significantly, respectively, highlighting the critical role of our causal intervention in improving the semantic consistency of the protocol space. In addition, removing the intervention mask also causes performance degradation, verifying the effectiveness of the mask-guided intervention mechanism.

We directly remove the semantic context guidance, which results in a substantial performance drop, demonstrating the necessity of semantic context for the unified converter. Replacing CGDR and SCE + CGDR with a single-tower ConvNeXt of the same number of layers, respectively, leads to additional performance degradation, and this downward trend demonstrates the necessity of the coexistence of SCE and CGDR. Finally, removing the adapter for new modalities causes performance degradation, which demonstrates the value of adapters in incorporating new modalities.

## 4.4. Qualitative Evaluation

To provide more intuitive evidence that CauseCollab learns a semantically consistent protocol space for heterogeneous collaboration, we visualize the intermediate features of Li-DAR and camera modalities for the same scene in Figure 4, as well as the protocol space features converted from the camera modality by STAMP and our method, respectively. As shown in Figure 4(a) and (b), a significant semantic gap exists between the two modalities’ intermediate features. STAMP introduces semantically irrelevant noise in background regions (Figure 4(c)), whereas our method achieves cleaner and more effective protocol semantics conversion, better preserving cross-modality semantic consistency (Figure 4(d)).

## 5. Conclusion

In this paper, we propose a causal unified and modalityagnostic network called CauseCollab, to address the issues of semantic inconsistency and modality confounding in the protocol space of heterogeneous collaborative perception. CauseCollab achieves modality-agnostic consistent semantics via causal metric learning, and enables the extension of new modalities to the semantically consistent protocol space by training only lightweight adapters with minimal parameters. Extensive experiments demonstrate the effectiveness of our proposed method.

## Acknowledgements

This work was supported in part by the Natural Science Foundation of China under Grant 62272053 and Grant 62472048, in part by the Beijing Nova Program under Grant 20230484364, in part by Beijing Natural Science Foundation under Grant L242081, and in part by BUPT Excellent Ph.D. Students Foundation.

## Impact Statement

This work advances heterogeneous collaborative perception by addressing semantic inconsistency and modality confounding through a causally unified, modality-agnostic framework. CauseCollab enables seamless integration of new sensing modalities with minimal computational overhead, accelerating the deployment of multi-modal collaborative systems in autonomous driving and intelligent transportation. Potential risks may arise from safety-critical deployment and privacy concerns in multi-agent perception systems. However, our framework mitigates these risks to some extent by improving cross-modal semantic consistency and enabling adaptation without accessing or modifying previously deployed modality-specific encoders, fusion modules, and detection heads.

## References

Arnold, E., Dianati, M., Temple, R. d., and Fallah, S. Cooperative perception for 3d object detection in driving scenarios using infrastructure sensors. 23(3):1852– 1864, 2022. doi: 10.1109/TITS.2020.3028424. URL http://arxiv.org/abs/1912.12147.

Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., and Koltun, V. Carla: An open urban driving simulator, 2017.

URL https://arxiv.org/abs/1711.03938.

Gao, X., Xu, R., Li, J., Wang, Z., Fan, Z., and Tu, Z. Stamp: Scalable task and model-agnostic collaborative perception. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2501.18616.

Han, Y., Zhang, H., Li, H., Jin, Y., Lang, C., and Li, Y. Collaborative perception in autonomous driving: Methods, datasets and challenges. IEEE Intelligent Transportation Systems Magazine, 15(6):131–151, November 2023. doi: 10.1109/MITS.2023.3298534. URL https://arxiv.org/abs/2301.06262.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition, 2015. URL https:// arxiv.org/abs/1512.03385.

Hu, J., Shen, L., Albanie, S., Sun, G., and Wu, E. Squeezeand-excitation networks, 2019. URL https://arxiv. org/abs/1709.01507. CVPR 2019.

Hu, Y., Fang, S., Lei, Z., Zhong, Y., and Chen, S. Where2comm: Communication-efficient collaborative perception via spatial confidence maps. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2209.12836.

Hu, Y., Peng, J., Liu, S., Ge, J., Liu, S., and Chen, S. Communication-efficient collaborative perception via information filling with codebook. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL https://arxiv. org/abs/2405.04966.

Huang, X. and Belongie, S. Arbitrary style transfer in realtime with adaptive instance normalization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017. URL https://arxiv.org/ abs/1703.06868.

Huang, Z. and Gool, L. V. A riemannian network for spd matrix learning. In Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI), 2017. URL https://arxiv.org/abs/1608.04233.

Huang, Z., Wang, S., Wang, Y., and Wang, L. Codiff: Conditional diffusion model for collaborative 3d object detection, 2025. URL https://arxiv.org/abs/ 2502.14891.

Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., and Beijbom, O. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. URL https://arxiv.org/abs/ 1812.05784.

Li, M., Lin, H., Qiu, L., Liang, X., Chen, L., Elsaddik, A., and Chang, X. Contrastive learning with counterfactual explanations for radiology report generation. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. URL https://arxiv.org/ abs/2407.14474.

Li, Y., Fang, C., Yang, J., Wang, Z., Lu, X., and Yang, M.-H. Universal style transfer via feature transforms. In Advances in Neural Information Processing Systems (NeurIPS), 2017. URL https://arxiv.org/abs/ 1705.08086.

Li, Y., Ren, S., Wu, P., Chen, S., Feng, C., and Zhang, W. Learning distilled collaboration graph for multiagent perception. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 2021. URL https://arxiv.org/abs/2111.00643.

Liu, S., Gao, C., Chen, Y., Peng, X., Kong, X., Wang, K., Xu, R., Jiang, W., Xiang, H., Ma, J., and Wang, M. Towards vehicle-to-everything autonomous driving: A survey on collaborative perception, 2023. URL https: //arxiv.org/abs/2308.16714.

Liu, Y.-C., Tian, J., Glaser, N., and Kira, Z. When2com: Multi-agent perception via communication graph grouping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. URL https://arxiv.org/abs/2006.00176.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s, 2022. URL https://arxiv.org/abs/2201.03545. Published in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Lu, Y., Li, Q., Liu, B., Dianati, M., Feng, C., Chen, S., and Wang, Y. Robust collaborative 3d object detection in presence of pose errors. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2023. URL https://arxiv.org/abs/ 2211.07214.

Lu, Y., Hu, Y., Zhong, Y., Wang, D., Chen, S., and Wang, Y. An extensible framework for open heterogeneous collaborative perception. In International Conference on Learning Representations (ICLR), 2024.

Luo, T., Yuan, Q., Luo, G., Xia, Y., Yang, Y., and Li, J. Plug and Play: A representation enhanced domain adapter for collaborative perception. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXXXI, pp. 287–303, Berlin, Heidelberg, 2024. Springer-Verlag. ISBN 978-3-031-73003-0. doi: 10.1007/978-3-031-73004-7 17. URL https://doi. org/10.1007/978-3-031-73004-7\_17.

Philion, J. and Fidler, S. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In Proceedings of the European Conference on Computer Vision (ECCV), 2020. URL https: //arxiv.org/abs/2008.05711.

Qiao, Z., Liu, C., Zhang, Y., Jin, M., Pham, Q., Wen, Q., Suganthan, P. N., Jiang, X., and Ramasamy, S. Multiscale finetuning for encoder-based time series foundation models. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv. org/abs/2506.14087.

Scholkopf, B. ¨ Causality for Machine Learning, pp. 765–804. ACM, February 2022. ISBN 9781450395861. doi: 10.1145/3501714.3501755. URL http://dx.doi. org/10.1145/3501714.3501755.

Shao, C., Luo, G., Yuan, Q., Chen, Y., Liu, Y., Gong, K., and Li, J. Hetecooper: Feature collaboration graph for heterogeneous collaborative perception. In European Conference on Computer Vision, 2024. URL https://api.semanticscholar. org/CorpusID:272350894.

Shao, C., Yuan, Q., Luo, G., Hu, Y., Wang, D., Liu, Y., Pan, R., Chen, B., and Li, J. Negocollab: A common representation negotiation approach for heterogeneous collaborative perception. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2510.27647.

Tan, M. and Le, Q. V. Efficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings of the 36th International Conference on Machine Learning (ICML), 2019. URL https://arxiv.org/ abs/1905.11946.

van den Oord, A., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748.

Wang, L., He, Z., Dang, R., Shen, M., Liu, C., and Chen, Q. Vision-and-language navigation via causal learning, 2024. URL https://arxiv.org/abs/2404.10241.

Wei, S., Wei, Y., Hu, Y., Lu, Y., Zhong, Y., Chen, S., and Zhang, Y. Asynchrony-robust collaborative perception via bird’s eye view flow. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2309.16940.

Xiang, H., Xu, R., and Ma, J. Hm-vit: Hetero-modal vehicle-to-vehicle cooperative perception with vision transformer, 2023. URL https://arxiv.org/ abs/2304.10628.

Xu, R., Guo, Y., Han, X., Xia, X., Xiang, H., and Ma, J. Opencda:an open cooperative driving automation framework integrated with co-simulation, 2021. URL https://arxiv.org/abs/2107.06260.

Xu, R., Tu, Z., Xiang, H., Shao, W., Zhou, B., and Ma, J. Cobevt: Cooperative bird’s eye view semantic segmentation with sparse transformers. In Proceedings of the 6th Conference on Robot Learning (CoRL), 2022a. URL https://arxiv.org/abs/2207.02202.

Xu, R., Xiang, H., Tu, Z., Xia, X., Yang, M.-H., and Ma, J. V2x-vit: Vehicle-to-everything cooperative perception with vision transformer. In Avidan, S., Brostow, G., Cisse,´ M., Farinella, G. M., and Hassner, T. (eds.), Computer Vision – ECCV 2022, volume 13699 of Lecture Notes in Computer Science, pp. 107–124. Springer, Cham, 2022b. doi: 10.1007/978-3-031-19842-7 7.

Xu, R., Xiang, H., Xia, X., Han, X., Li, J., and Ma, J. Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication. In Proceedings ofthe IEEE International Conference on Robotics and Automation (ICRA), 2022c. URL https: //arxiv.org/abs/2109.07644.

Xu, R., Li, J., Dong, X., Yu, H., and Ma, J. Bridging the domain gap for multi-agent perception. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2023. URL https://arxiv.org/ abs/2210.08451.

Yan, J., Deng, C., Huang, H., and Liu, W. Causalityinvariant interactive mining for cross-modal similarity learning. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 46(9):6216–6230, 2024. doi: 10.1109/TPAMI.2024.3379752. URL https:// doi.org/10.1109/TPAMI.2024.3379752.

Yan, Y., Mao, Y., and Li, B. Second: Sparsely embedded convolutional detection. Sensors, 18(10):3337, 2018.

Yin, H., Tian, D., Lin, C., Duan, X., Zhou, J., Zhao, D., and Cao, D. V2vformer++: Multi-modal vehicle-to-vehicle cooperative perception via global-local transformer. IEEE Transactions on Intelligent Transportation Systems, 25 (2):2153–2166, 2024. doi: 10.1109/TITS.2023.3314919.

Yu, H., Luo, Y., Shu, M., Huo, Y., Yang, Z., Shi, Y., Guo, Z., Li, H., Hu, X., Yuan, J., and Nie, Z. Dair-v2x: A large-scale dataset for vehicle-infrastructure cooperative 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. URL https://arxiv.org/abs/ 2204.05575.

Zhang, T., Liu, P., Lu, Y., Cai, M., Zhang, Z., Zhang, Z., and Zhou, Q. Cwnet: Causal wavelet network for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. URL https://arxiv.org/abs/2507.10689.

Zhao, B., Zhang, W., and Zou, Z. Bm2cp: Efficient collaborative perception with lidar-camera modalities. In Proceedings of the 7th Conference on Robot Learning (CoRL), 2023. URL https://arxiv.org/abs/ 2310.14702.

Zhou, Y. and Tuzel, O. Voxelnet: End-to-end learning for point cloud based 3d object detection, 2017. URL https://arxiv.org/abs/1711.06396.

## A. Details of Experiment

## A.1. Dataset

OPV2V. OPV2V (Xu et al., 2022c) is a large-scale public dataset for multi-agent collaborative perception research. Generated by the CARLA (Dosovitskiy et al., 2017) simulation platform combined with the OpenCDA (Xu et al., 2021) co-simulation framework, it contains over 70 scenarios with a total of 11,464 frames, covering diverse urban scenarios and various road structures. Each agent is equipped with one 16-channel, one 32-channel, and one 64-channel LiDAR, along with 4 RGB cameras and 4 depth cameras.

DAIR-V2X. DAIR-V2X (Yu et al., 2022) is a real-world dataset dedicated to vehicle-infrastructure collaborative perception. The dataset scenarios consist of one vehicle and one Road Side Unit (RSU). The agent is equipped with a 40-channel LiDAR and a 1920×1080 resolution camera, while the RSU is equipped with a 300-channel LiDAR and a camera of the same specification.

## A.2. Training Details

To ensure fair comparisons, we use identical encoder weights across different methods, such that the modality encoders achieve the same detection performance in homogeneous settings. All LiDAR encoders are trained with the perception range of $[ - 1 0 2 . 4 , - 5 1 . 2 , 1 0 2 . 4 , 5 1 . 2 ]$ , while all camera encoders are trained with [−51.2, −51.2, 51.2, 51.2]. All heterogeneous cooperative perception networks are trained and evaluated within [−51.2, −51.2, 51.2, 51.2]. Our training is conducted on NVIDIA RTX 4090 GPUs. We use the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ . The unified converter is trained for 10 epochs. Adapting to a new modality only requires 6 epochs to reach aligned semantics. Each modality-specific local semantic reconstructor is trained for 5 epochs. The hyperparameters used during training are shown in Table 5.

Table 5. Hyperparameters in experiments.
<table><tr><td></td><td colspan="4">Stage-1</td><td colspan="3">Stage-2</td></tr><tr><td>Loss Weight</td><td> $\lambda _ { \mathrm { c o n } }$ </td><td> $\lambda _ { \mathrm { c t x } }$ </td><td> $\lambda _ { \mathrm { s s i m } }$ </td><td> $\lambda _ { \mathrm { t a s k } } ^ { ( 1 ) }$ </td><td> $\lambda _ { \mathrm { t a s k } } ^ { ( 2 ) }$ </td><td> $\lambda _ { \mathrm { m s e } }$ </td><td> $\lambda _ { \mathrm { c o n s } }$ </td></tr><tr><td>Value</td><td>2.0</td><td>0.5</td><td>1.0</td><td>0.5</td><td>1.0</td><td>5.0</td><td>5.0</td></tr></table>

## A.3. Detailed Configuration of Agents

Our experiments involve eight agent modalities and a protocol modality, whose perception configurations and performance in homogeneous scenarios are shown in Table 6. Here, L denotes the LiDAR modality, and C denotes the Camera modality In the supplementary experiments, $V _ { v n 4 }$ (Zhou & Tuzel, 2017) and $C _ { \mathrm { R e s 3 4 } }$ (He et al., 2015) are additionally included.

Table 6. Configuration and performance of agents
<table><tr><td>Agent</td><td>Encoder</td><td>Voxel Size</td><td>Camera-Encoder</td><td>AP@0.3 / AP@0.5 ↑</td></tr><tr><td>Protocol</td><td>PointPillar</td><td>0.4, 0.4, 4</td><td></td><td>0.982 / 0.980</td></tr><tr><td> $L _ { p p 4 }$ </td><td>PointPillar</td><td>0.4, 0.4, 4</td><td></td><td>0.972 / 0.968</td></tr><tr><td> $C _ { \mathrm { E f f } }$ </td><td>Lift-Splat-Shoot</td><td></td><td>EfficientNet</td><td>0.790 / 0.706</td></tr><tr><td> $L _ { s d 1 }$ </td><td>SECOND</td><td>0.1, 0.1, 0.1</td><td></td><td>0.975 / 0.971</td></tr><tr><td> $C _ { \mathrm { R e s } }$ </td><td>Lift-Splat-Shoot</td><td></td><td>ResNet101(Output Channel = 512)</td><td>0.729 / 0.625</td></tr><tr><td> $L _ { s d 2 }$ </td><td>SECOND</td><td>0.2, 0.2, 0.2</td><td></td><td>0.973 / 0.968</td></tr><tr><td> $C _ { \mathrm { E f f B 1 } }$ </td><td>Lift-Splat-Shoot</td><td></td><td>EfficientNetB1</td><td>0.652 / 0.579</td></tr><tr><td> $L _ { v n 4 }$ </td><td>VoxelNet</td><td>0.4, 0.4, 0.4</td><td></td><td>0.960 / 0.958</td></tr><tr><td> $C _ { \mathrm { R e s 3 4 } }$ </td><td>Lift-Splat-Shoot</td><td></td><td>ResNet34(Output Channel = 128)</td><td>0.619 / 0.546</td></tr></table>

## B. Implementation Details

## B.1. Intervention via SPD Feature Geometry

All operations are performed on the reshaped protocol feature

$$
p \in \mathbb { R } ^ { C \times H \times W }  X \in \mathbb { R } ^ { C \times N } , \quad N = H \cdot W ,
$$

where the n-th column $x _ { n } \in \mathbb { R } ^ { C }$ denotes the C-channel feature vector at spatial position n.

We represent modality style using the weighted first- and second-order statistics $( \mu , \Sigma )$ computed from X, with $\Sigma \in S _ { + } ^ { C }$ A semantic mask $m \in [ 0 , 1 ] ^ { 1 \times N }$ , produced by a mask generator $g ( \cdot )$ , is required to define spatially varying non-negative weights, thereby modulating foreground and background contributions. The Mask Generator applies the classification head from single-modality training to local features to produce a confidence map, which is filtered by a Gaussian kernel $( k = 5 , \sigma = 1 . 0 )$ , and thresholded at 0.01 to obtain a binary mask.

Notation and weighted statistics. Given $X = [ x _ { 1 } , \dots , x _ { N } ]$ and non-negative weights $w = [ w _ { 1 } , \dots , w _ { N } ]$ , we compute

$$
\mu = \frac { \sum _ { n = 1 } ^ { N } w _ { n } x _ { n } } { \sum _ { n = 1 } ^ { N } w _ { n } } ,\tag{15}
$$

$$
\Sigma = \frac { \sum _ { n = 1 } ^ { N } w _ { n } ( x _ { n } - \mu ) ( x _ { n } - \mu ) ^ { \top } } { \sum _ { n = 1 } ^ { N } w _ { n } } + \varepsilon I , \qquad \varepsilon > 0 ,\tag{16}
$$

where $\varepsilon I$ ensures numerical stability. In practice, the weights are parameterized using the semantic mask as

$$
w = \eta + \rho m , \qquad \eta > 0 , \rho \geq 0 ,\tag{17}
$$

which guarantees strictly positive weights and allows the mask to control contribution strength.

SPD and whitening–coloring transform. For any SPD matrix $\Sigma ,$ let $\Sigma = U \Lambda U ^ { \top }$ be its eigendecomposition and define

$$
\Sigma ^ { \frac { 1 } { 2 } } = U \Lambda ^ { \frac { 1 } { 2 } } U ^ { \top } , \qquad \Sigma ^ { - \frac { 1 } { 2 } } = U ( \Lambda + \varepsilon I ) ^ { - \frac { 1 } { 2 } } U ^ { \top } ,
$$

where $\Lambda ^ { \frac { 1 } { 2 } }$ is applied elementwise and $\varepsilon$ is added to the eigenvalues for stability. Given source statistics $\left( \mu _ { s } , \Sigma _ { s } \right)$ and target statistics $\left( \mu _ { t } , \Sigma _ { t } \right)$ , the closed-form whitening–coloring transform applied to X is

$$
T _ { s  t } ( X ) = \mu _ { t } + \Sigma _ { t } ^ { \frac { 1 } { 2 } } \Sigma _ { s } ^ { - \frac { 1 } { 2 } } ( X - \mu _ { s } ) .\tag{18}
$$

This linear mapping preserves the spatial arrangement of columns in $X$ while modifying only channel statistics.

## MASK-AWARE CANONICAL SUPPRESSION (MCS)

The objective of MCS is to demodalize a protocol feature by mapping its channel statistics to a neutral canonical distribution $\left( \mu _ { 0 } , \Sigma _ { 0 } \right)$ , while preserving spatial semantics through a mask-guided residual update.

Algorithm 1 Mask-aware Canonical Suppression (MCS)   
Require: $\boldsymbol { X } \in \mathbb { R } ^ { C \times N }$ (protocol feature), confidence map M   
Ensure: $X _ { 0 }$ (modality-agnostic feature)   
1: $m \gets g ( M )$ {semantic mask, $1 \times N \}$   
2: w $ \eta + \rho m$ {weights as in (17)}   
3: Compute $\left( \mu _ { c } , \Sigma _ { c } \right)$ from X using $( 1 5 ) \AA - ( 1 6 )$   
4: Set the canonical target to $( \mu _ { 0 } , \Sigma _ { 0 } ) = ( \mathbf { 0 } , I )$   
5: $\tilde { X } \gets T _ { c \to 0 } ( X )$ using (18)   
6: Compute strength map $\beta ( m ) \in [ 0 , 1 ] ^ { 1 \times N }$   
7: $X _ { 0 } \gets X + ( \beta ( m ) \odot \mathbf { 1 } _ { C } ) \odot ( \tilde { X } - \tilde { X } )$   
8: Return: $X _ { 0 }$

Note: All eigendecompositions apply a small eigenvalue regularizer to ensure $\Sigma ^ { - \frac { 1 } { 2 } }$ is well-defined.

## CANONICAL SUPPRESS-THEN-INJECT (STI)

To create hard negative samples and controlled modality perturbations, we first suppress modality bias via MCS and then inject heterogeneous modality statistics $\left( \mu _ { s } , \Sigma _ { s } \right)$ computed from a style feature $X _ { s }$ . The same semantic mask m (and associated weights w) is used so that injection respects the original spatial semantics.

Algorithm 2 Canonical Suppress-Then-Inject (STI)   
Require: X (content protocol feature), X<sub>s</sub> (heterogeneous style feature), confidence map M   
Ensure: $X ^ { + }$ (modality-agnostic positive), $X ^ { - }$ (injected hard negative)   
1: $X _ { 0 } \gets \mathbf { M C S } ( X , M )$   
2: m $ g ( M )$ ), w $w \gets \eta + \rho m$   
3: Compute $\left( \mu _ { 0 } , \Sigma _ { 0 } \right)$ from $X _ { 0 }$ and $( \mu _ { s } , \Sigma _ { s } )$ from $X _ { s }$   
4: $\hat { X } \left. T _ { 0 \right. s } ( X _ { 0 } )$   
5: Compute injection strength map $\beta _ { . } ^ { \mathrm { i n j } } ( m )$   
6: $X ^ { - }  X _ { 0 } + ( \beta ^ { \mathrm { i n j } } ( m ) \odot \mathbf { 1 } _ { C } ) \odot ( \hat { X } - X _ { 0 } )$   
7: $X ^ { + }  X _ { 0 }$   
8: Return: $X ^ { + } , X ^ { - }$

The positive sample $p ^ { + }$ is obtained by reshaping $X ^ { + }$ back to the original spatial layout and serves as the modality-agnostic anchor for metric learning. The negative sample $p ^ { - }$ is the injected feature $X ^ { - }$ , which is a hard negative created by re-introducing controlled heterogeneous modality statistics while preserving consistent semantics.

## B.2. SCE Architecture

The architecture of the SCE is illustrated in Figure 5, which gradually enhances the global semantic modeling capability through two-step downsampling and ConvNeXt (Liu et al., 2022).

![](images/f23d7fa008078a8e48ad67d23b36bd5e1fe2ab626e2febdd4b7eddf66054463f.jpg)  
Figure 5. Detailed structure of the Semantic Context Extractor

## B.3. CGDR Architecture

The architecture of the CGDR is illustrated in Figure 6. This module takes the local modality feature $f _ { i } \in \mathbb { R } ^ { C \times H \times W }$ and the context feature $\hat { c } _ { i } \in \mathbb { R } ^ { C _ { c t x } \times H \times W }$ as inputs. After the context feature is processed by a $1 \times 1$ Project Conv, it is concatenated with $f _ { i }$ and fed into the Joint Fusion module, which consists of a depth-wise convolution and LayerNorm, for information integration.

The fused feature is modulated by the Context Guide module (FiLM-style multi-head feature modulation) and recalibrated in channel dimension via the $S E$ module (Hu et al., 2019), then filtered by the Gate module (1x1 convolution followed by SiLU) to suppress noise. Subsequently, a residual connection with Layer Scale stabilizes the gradient, followed by transformation through an $M L P$ and another residual connection with Layer Scale 2 to preserve detailed features. Finally, the output is split into an updated local feature and a context feature for subsequent layers, which balances local discriminability and global semantic consistency.

![](images/84138b8c4f15f08f262e0bcce147a88680b210fec07a7881d16dde23a0e9cdbe.jpg)  
Figure 6. Detailed structure of the Context-Guided Dynamic Refiner

## C. More Experiments

## C.1. Comparison between CauseCollab and Late Fusion

Late fusion directly integrates the 3D detection boxes perceived by each agent to enable information sharing.We compare the performance of our method with that of late fusion on the OPV2V dataset under different pose error σ values. As shown in Table 7, ALL denotes the collaboration of the four modalities. We additionally set two modality configurations for Ego-Neighbor pairs with large modality gaps. Our method significantly outperforms the late fusion method across all pose error levels and collaboration combinations, demonstrating the superiority of our method in real-world application scenarios.

Table 7. Performance comparison of CauseCollab and Late Fusion under varying pose error standard deviations
<table><tr><td rowspan=1 colspan=1>σ</td><td rowspan=1 colspan=4>AP@30Method</td><td rowspan=1 colspan=3>AP@50</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ALL</td><td rowspan=1 colspan=1> $L _ { p p 4 } + C _ { \mathrm { E f f } }$ </td><td rowspan=1 colspan=1> $C _ { \mathrm { E f f } } + C _ { \mathrm { R e s } }$ </td><td rowspan=1 colspan=1>ALL</td><td rowspan=1 colspan=1> $L _ { p p 4 }$ + CEff</td><td rowspan=1 colspan=1> $C _ { \mathrm { E f f } } + C _ { \mathrm { R e s } }$ </td></tr><tr><td rowspan=2 colspan=1>0.0</td><td rowspan=2 colspan=1>Late FusionCauseCollab</td><td rowspan=2 colspan=1>0.9570.978</td><td rowspan=1 colspan=1>0.956</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.943</td><td rowspan=2 colspan=1>0.9300.966</td><td rowspan=2 colspan=1>0.6300.670</td></tr><tr><td rowspan=1 colspan=1>0.971</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.975</td></tr><tr><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>Late FusionCauseCollab</td><td rowspan=1 colspan=1>0.9520.976</td><td rowspan=1 colspan=1>0.9510.970</td><td rowspan=1 colspan=1>0.7340.762</td><td rowspan=1 colspan=1>0.8480.968</td><td rowspan=1 colspan=1>0.8510.960</td><td rowspan=1 colspan=1>0.5340.625</td></tr><tr><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>Late FusionCauseCollab</td><td rowspan=1 colspan=1>0.8300.964</td><td rowspan=1 colspan=1>0.8490.960</td><td rowspan=1 colspan=1>0.6240.698</td><td rowspan=1 colspan=1>0.5630.916</td><td rowspan=1 colspan=1>0.6320.922</td><td rowspan=1 colspan=1>0.3320.487</td></tr><tr><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>Late FusionCauseCollab</td><td rowspan=1 colspan=1>0.6790.928</td><td rowspan=1 colspan=1>0.7300.932</td><td rowspan=1 colspan=1>0.4840.590</td><td rowspan=1 colspan=1>0.4370.859</td><td rowspan=1 colspan=1>0.5380.882</td><td rowspan=1 colspan=1>0.2350.381</td></tr></table>

## C.2. Analysis of Different Protocol Modalities

The alignment effectiveness of protocol-based heterogeneous collaborative perception depends on the affinity between the protocol modality and heterogeneous modalities. To analyze this effect, we conduct supplementary experiments using either LiDAR $( L _ { p p 4 } )$ or Camera $( C _ { E f f } )$ as the protocol modality. The results are shown in Table 8.

Table 8. Performance comparison under different protocol modalities.
<table><tr><td rowspan=1 colspan=1>Protocol Modality</td><td rowspan=1 colspan=1>ALL</td><td rowspan=1 colspan=1> $L _ { \mathrm { p p 4 } } + C _ { \mathrm { E f f } }$ </td><td rowspan=1 colspan=1> $C _ { \mathrm { E f f } } + L _ { \mathrm { p p 4 } }$ </td></tr><tr><td rowspan=1 colspan=1>LiDAR (PointPillar)</td><td rowspan=1 colspan=1>0.978 / 0.9751</td><td rowspan=1 colspan=1>0.971 / 0.966</td><td rowspan=1 colspan=1>0.867 / 0.842</td></tr><tr><td rowspan=1 colspan=1>Camera(Lift-Splat-Shoot)</td><td rowspan=1 colspan=1>0.975 / 0.973</td><td rowspan=1 colspan=1>0.968 / 0.964</td><td rowspan=1 colspan=1>0.875 / 0.847</td></tr></table>

The results show that CauseCollab achieves comparable collaborative perception performance under both LiDAR-based and Camera-based protocol modalities. This indicates that the proposed causal intervention effectively reduces protocol-modality asymmetry and improves the robustness of the learned protocol space.

## C.3. Comparison between CauseCollab and NegoCollab

NegoCollab (Shao et al., 2025) mitigates the domain gap by negotiating a common representation, whereas our method explicitly disentangles semantic factors from modal entanglement to learn a modality-agnostic protocol space. Experimental results demonstrate that our method outperforms NegoCollab in both general and novel-modality scenarios. ALL represents the general multi-agent collaboration scenarios under the same experimental settings as in Table 1, while $L _ { s d 2 ^ { \mathrm { n e w } } } + C _ { \mathrm { E f f B 1 ^ { \mathrm { n e w } } } }$ verifies the performance on the new modality.

Table 9. Performance and parameter comparison of CauseCollab and NegoCollab
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>ALL</td><td rowspan=1 colspan=1> $L _ { s d 2 ^ { \mathrm { n e w } } } + C _ { \mathrm { E f f B 1 ^ { \mathrm { n e w } } } }$ </td><td rowspan=1 colspan=1>#Params(MB)</td></tr><tr><td rowspan=1 colspan=1>CauseCollab</td><td rowspan=1 colspan=1>0.978/0.975</td><td rowspan=1 colspan=1>0.966/0.959</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>NegoCollab</td><td rowspan=1 colspan=1>0.974/0.972</td><td rowspan=1 colspan=1>0.958/0.953</td><td rowspan=1 colspan=1>0.98</td></tr></table>

## C.4. Robustness Evaluation on More Novel Modality Adaptation

To further validate the robustness of the CauseCollab protocol space for novel modality adaptation, we introduce two additional novel modalities: $L _ { \mathrm { v n 4 } } \mathrm { n e w }$ and $C _ { \mathrm { R e s 3 4 } } ^ { \mathrm { n e w } }$ . Specifically, $L _ { \mathrm { v n 4 ^ { n e w } } }$ denotes a LiDAR modality encoded by VoxelNet (Zhou & Tuzel, 2017), while $C _ { \mathrm { R e s 3 4 ^ { n e w } } }$ denotes a Camera modality based on ResNet34 with a reduced output channel size of 128. Together with $L _ { \mathrm { s d 2 ^ { n e w } } }$ and C new , we evaluate a collaborative perception scenario involving four novel modalities: $L _ { \mathrm { v n 4 ^ { n e w } } } + C _ { \mathrm { R e s 3 4 ^ { n e w } } } + L _ { \mathrm { s d 2 ^ { n e w } } } + C _ { \mathrm { E f f B 1 ^ { n e w } } }$ . The results are reported in Table 10.

Table 10. Robustness evaluation under four novel modalities.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>AP@30</td><td rowspan=1 colspan=1>AP@50</td></tr><tr><td rowspan=1 colspan=1>MPDA</td><td rowspan=1 colspan=1>0.970</td><td rowspan=1 colspan=1>0.968</td></tr><tr><td rowspan=1 colspan=1>PnPDA    一</td><td rowspan=1 colspan=1>0.968   一</td><td rowspan=1 colspan=1>0.965</td></tr><tr><td rowspan=1 colspan=1>HEAL    /</td><td rowspan=1 colspan=1>0.919   一</td><td rowspan=1 colspan=1>0.917</td></tr><tr><td rowspan=1 colspan=1>STAMP</td><td rowspan=1 colspan=1>0.963   </td><td rowspan=1 colspan=1>0.961</td></tr><tr><td rowspan=1 colspan=1>CauseCollab</td><td rowspan=1 colspan=1>0.974</td><td rowspan=1 colspan=1>0.971</td></tr></table>

## C.5. Additional qualitative experiments

Figure 7 presents additional visualization experiments. To highlight the visual differences, we specifically select $C _ { \mathrm { E f f } }$ and $C _ { \mathrm { R e s } }$ for collaboration due to their larger modality gaps. The visualization intuitively demonstrates the robustness of our method even in such scenarios with larger domain gaps.

![](images/b66232f98163e7252a3113504844c9f1e045616fa68440d66a6cfffb45bcfe93.jpg)  
Figure 7. Visual comparison of detection results across different methods, using $C _ { \mathrm { E f f } }$ and $C _ { \mathrm { R e s } }$