# DISAGREEMENT-AWARE DISTRIBUTION MATCHING DISTILLATION FOR FEW-STEP DIFFUSION MODELS

Zikai Zhou<sup>1</sup>, Zhengpeng Xie<sup>1</sup>, Shitong Shao<sup>1</sup>, Lichen Bai<sup>1</sup>, Tianhao Zhang<sup>3</sup> Bojun Cheng<sup>1</sup>, Zeke Xie<sup>1,2∗</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup> The Hong Kong University of Science and Technology <sup>3</sup>Catnip AI

zzhou440@connect.hkust-gz.edu.cn

## ABSTRACT

Distribution Matching Distillation (DMD) enables strong few-step diffusion generation by matching the student distribution to a pretrained teacher. Existing improvements to DMD often augment its supervision or modify its optimization machinery. We ask a complementary question: can DMD diagnose its own updates using signals already produced during training? DMD inherently computes the discrepancy between a pretrained teacher and a fake model tracking the student distribution, but conventionally uses it primarily to determine the distribution matching direction. We find that the magnitude of this teacher–fake disagreement provides additional diagnostic information. Across extensive experiments, large disagreement in the high-noise regime consistently coincides with substantially higher-energy DMD updates. This reveals a dual role of disagreement: it determines not only where the student should move, but also provides an intrinsic signal of how aggressively the update should be applied. Based on this observation, we introduce Disagreement-Aware Distribution Matching Distillation (DA-DMD), which activates only in the high-noise regime and continuously modulates the update strength according to teacher–fake disagreement. DA-DMD reuses quantities already available during DMD training and requires no additional model, evaluator, or inference-time modification. We show that it exactly recovers vanilla DMD outside the selected region, reduces update energy only through selected samples, and exhibits redescending influence under extreme disagreement. Across multiple diffusion backbones, DA-DMD consistently improves few-step generation at the same inference cost. On SD3.5-L, DA-DMD improves HPSv2 from 28.23 to 30.93, while achieving a 64.4% average win rate over DMD2.

## 1 INTRODUCTION

Diffusion models have become a dominant paradigm for high-fidelity image generation, but their iterative sampling procedure often requires many denoising evaluations at inference time (Ho et al., 2020; Song et al., 2020; Karras et al., 2022; Liu et al., 2026; Shao et al., 2026). This has motivated extensive research on compressing pretrained diffusion models into few-step generators, including progressive distillation, consistency-based approaches, and distribution matching methods (Salimans & Ho, 2022; Song et al., 2023; Luo et al., 2023; Yin et al., 2024b;a). Among them, Distribution Matching Distillation (DMD) directly aligns the distribution induced by a few-step student with that of a pretrained diffusion teacher, using a frozen teacher and a fake model that tracks the evolving student distribution (Yin et al., 2024b;a). This distribution-level formulation has enabled strong few-step generation across a range of diffusion backbones.

Recent work has improved DMD through better fake-model optimization, auxiliary objectives, reward-guided optimization, and adaptive teacher mechanisms (Yin et al., 2024a; Luo et al., 2026; Bai et al., 2026; Jiang et al., 2025; Wu et al., 2026a;b; Zhou et al., 2024). We take a complementary perspective: can DMD make better use of the information it already produces during training? Every DMD update compares the predictions of a pretrained teacher and a fake model at the same noisy sample. This teacher–fake discrepancy determines the direction of distribution matching, but its magnitude varies substantially across samples and noise levels. We therefore investigate whether this native variation also carries information about the behavior of the resulting update.

![](images/df47b859640cc0c49841e0bbd81b3fd301db8b313d1040071a1a15b7d2e86c8e.jpg)  
Normalized teacher-fake disagreement r

![](images/caba5ccd1b3daf0a606ab0532bb6eb1dda84173e6e60448ffe56abadd6c509b7.jpg)  
Normalized teacher-fake disagreement r

![](images/0a2dd12c5fadb172dfb8c5462388054baacd14984c1d317df37c8f029da55ad0.jpg)  
Normalized teacher-fake disagreement  
Figure 1: Teacher–fake disagreement diagnoses DMD update aggressiveness. Within the predefined high-noise region H, larger normalized disagreement is consistently associated with higherenergy raw DMD updates across DIGITS, CIFAR-10, and CIFAR-100. DA-DMD continuously attenuates these updates according to disagreement. Shaded regions highlight the high-disagreement tail for visualization only and do not correspond to an algorithmic threshold.

Our analysis reveals two complementary patterns. First, on real-world data from DIGITS (Alpaydin & Kaynak, 1998), CIFAR-10 (Krizhevsky et al., 2009), and CIFAR-100 (Krizhevsky et al., 2009), larger normalized teacher–fake disagreement within the high-noise regime consistently coincides with higher-energy raw DMD updates (Fig. 1). Second, the same high-energy behavior persists at scale: during SD3.5-M (Esser et al., 2024) distillation, unusually high-energy raw DMD updates become increasingly concentrated in the high-noise portion of the trajectory (Fig. 2). Importantly, these observations do not imply that high-noise or large-disagreement updates are inherently incorrect. Instead, together they suggest a simple optimization principle: use noise level to localize where aggressive DMD behavior tends to emerge, and use teacher–fake disagreement to continuously determine how strongly the update should be moderated.

Motivated by these insights, we introduce Disagreement-Aware Distribution Matching Distillation (DA-DMD). Rather than introducing a new source of guidance, DA-DMD reuses the teacher–fake discrepancy already computed by standard DMD. It activates only in a predefined high-noise region and continuously adjusts the DMD update strength according to normalized teacher–fake disagreement: updates outside this region remain exactly unchanged, while larger disagreement within the region induces stronger attenuation. DA-DMD therefore turns a native DMD signal that originally defines where the student should move into an additional control signal for how con-

![](images/89258d210c3f1315ab3d2b7498d3de305e900d2bcb719c204e5b180218f02bdb.jpg)  
Figure 2: High-energy DMD updates concentrate at high noise. During SD3.5-M distillation, raw DMD update energy rises sharply in the high-noise region H, where DA-DMD activates disagreement-aware attenuation.

servatively that move should be applied. Our ablations further show that applying the same correction globally or in lower-noise regions is less effective, supporting the selective high-noise design.

DA-DMD is theoretically attractive because its robustness is selective rather than global. Outside the active high-noise region, it exactly recovers vanilla DMD, ensuring that ordinary distribution matching supervision is not weakened. Within the active region, disagreement-adaptive weighting reduces update energy only through the modified samples, while its redescending influence prevents extreme teacher–fake disagreement from dominating optimization. Moreover, DA-DMD preserves the ideal fixed point of DMD, modifying the optimization dynamics rather than the target distribution. We conduct extensive experiments across multiple diffusion backbones, including SD3, SD3.5 (Esser et al., 2024), SANA (Xie et al., 2024), and SANA-1.5 (Xie et al., 2025), and evaluate on four complementary benchmarks: DrawBench (Saharia et al., 2022), PartiPrompts (Yu et al., 2022), GenEval2 (Kamath et al., 2025), and TIIF-Bench (Wei et al., 2025). Across model families and evaluation suites, DA-DMD consistently improves the corresponding DMD baselines, with particularly clear gains on challenging compositional and fine-grained generation tasks. These improvements are achieved without changing the inference procedure or sampling cost, while qualitative comparisons further demonstrate better visual fidelity. Our contributions are threefold:

1. We uncover an overlooked role of teacher–fake disagreement, which contains previously underused diagnostic information: in the high-noise regime, larger disagreement is consistently associated with higher-energy DMD updates.

2. We introduce Disagreement-Aware Distribution Matching Distillation (DA-DMD), which reuses this native signal to continuously modulate DMD updates in a predefined high-noise region while exactly preserving vanilla DMD elsewhere.

3. We provide a theoretical analysis of disagreement-aware attenuation. Under standard smoothness and alignment assumptions, we show that adaptive attenuation yields a tighter one-step optimization bound when DMD update energy is sufficiently large relative to its descent signal, providing a principled explanation for selective intervention in the high-noise regime. We further establish exact preservation outside the active region and redescending influence under extreme disagreement, and extensively validate DA-DMD across multiple diffusion backbones and benchmarks.

## 2 METHOD

Our goal is to make better use of a signal that DMD already computes: the teacher–fake discrepancy. We first revisit how this discrepancy forms the standard DMD update. We then derive an adaptive reweighting from a simple teacher–fake mixture construction and turn it into a practical disagreement-aware update rule. Finally, we provide a theoretical analysis explaining when and why selective attenuation can be preferable to vanilla DMD.

## 2.1 DISTRIBUTION MATCHING DISTILLATION

Distribution Matching Distillation (DMD) (Yin et al., 2024b;a) trains a few-step generator by matching the noisy marginal distribution induced by the student to that of a pretrained diffusion teacher. Let $G _ { \theta }$ denote the student generator and $T$ the frozen teacher. Given condition c and initial noise $z ,$ the student generates $x _ { 0 } ^ { \theta } = { \cal G } _ { \theta } ( z , c )$ , which is perturbed according to

$$
\begin{array} { r } { x _ { t } = \alpha _ { t } x _ { 0 } ^ { \theta } + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{1}
$$

We denote the resulting student marginal by $q _ { \theta , t } ( x _ { t } \mid c )$ and the corresponding teacher marginal by $p _ { T , t } ( x _ { t } \mid c )$ . DMD minimizes

$$
{ \mathcal { L } } _ { \mathrm { D M D } } ( \theta ) = \mathbb { E } _ { t , c } \left[ \lambda _ { t } D _ { \mathrm { K L } } \left( q _ { \theta , t } ( \cdot \mid c ) \parallel p _ { T , t } ( \cdot \mid c ) \right) \right] ,\tag{2}
$$

where $\lambda _ { t }$ is a timestep-dependent weighting coefficient.

The score of the evolving student marginal is not directly available, so DMD maintains a fake model $F _ { \phi }$ trained on student-generated samples to track the current student distribution. Let $p _ { F , t }$ denote the noisy marginal represented by the fake model, and define $s _ { T } = \nabla _ { x _ { 1 } }$ log $p _ { T , t } ( x _ { t } \mid c )$ and $s _ { F } =$ $\nabla _ { x _ { t } } \log \dot { p } _ { F , t } ( \bar { x } _ { t } \mid c )$ as the teacher and fake scores, respectively. Absorbing timestep-dependent coefficients into $w _ { t }$ , the sample-wise DMD update can be written as

$$
g _ { \mathrm { D M D } } = w _ { t } J _ { \theta } ^ { \top } \left( s _ { F } - s _ { T } \right) ,\tag{3}
$$

where $J _ { \theta } = \partial x _ { 0 } ^ { \theta } / \partial \theta .$

To connect Eq. 3 with the clean predictions directly produced by diffusion models, we use the Gaussian perturbation identity

$$
s _ { t } ( x _ { t } , t , c ) = \frac { \alpha _ { t } \mathbb { E } [ x _ { 0 } \mid x _ { t } , c ] - x _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{4}
$$

For an x<sub>0</sub>-prediction model trained with squared-error denoising, $\hat { x } _ { 0 } ( x _ { t } , t , c ) \approx \mathbb { E } [ x _ { 0 } \mid x _ { t } , c ]$ , and therefore

$$
s _ { t } ( x _ { t } , t , c ) \approx \frac { \alpha _ { t } \hat { x } _ { 0 } ( x _ { t } , t , c ) - x _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{5}
$$

(a) DA-DMD Paradigm  
![](images/008f18af0f1cca764ed08c68e5548735ab16f062e8a2a072b6dfd548c215ee30.jpg)

![](images/837cf02d2c091e8723c52b881d9c30e2f018d687b1b8d066949a6bb5a4196bf5.jpg)

(c) How disagreement-aware attenuation helps  
![](images/5cac51ff2868fe6183421acc35548cc4d746a5903fb946e49b6e3ef3d5f3f0aa.jpg)  
Figure 3: Overview of DA-DMD. (a) In the active high-noise region, DA-DMD replaces the raw teacher-directed correction with a disagreement-aware proximal interpolation between the fake and teacher predictions, yielding an attenuated DMD update with multiplier $\omega _ { \alpha } ( \boldsymbol r )$ . (b) Large teacher– fake disagreement can induce disproportionately large raw DMD corrections. (c) DA-DMD shortens these corrections toward an adaptive proximal target, reducing the influence of high-disagreement updates while preserving lower-disagreement updates.

A complete derivation of Eq. 4 from Fisher’s identity is provided in Appendix C.1.

Since the teacher and fake model are evaluated at the same noisy sample $x _ { t }$ , the shared term $- x _ { t } / \sigma _ { t } ^ { 2 }$ cancels, giving

$$
s _ { F } - s _ { T } \approx \frac { \alpha _ { t } } { \sigma _ { t } ^ { 2 } } \left( \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } \right) .\tag{6}
$$

Substituting Eq. 6 into Eq. 3 yields

$$
g _ { \mathrm { D M D } } \approx \gamma _ { t } J _ { \theta } ^ { \top } \left( \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } \right) ,\tag{7}
$$

where $\gamma _ { t } = w _ { t } \alpha _ { t } / \sigma _ { t } ^ { 2 }$ collects the timestep- and objective-dependent scalar factors.

Eq. 7 exposes the central signal driving DMD: the discrepancy between the teacher and fake predictions. Its direction determines the orientation of the distribution-matching correction, while its magnitude directly affects the resulting update strength. Importantly, this discrepancy varies substantially across samples and noise levels, as observed in Sec. 1. This raises a complementary question: can the teacher–fake discrepancy be used not only to construct the DMD correction, but also to determine how strongly that correction should be applied?

## 2.2 FROM DISTRIBUTION MIXTURE TO DISAGREEMENT-AWARE DMD

From distribution mixture to adaptive DMD. To derive an adaptive DMD correction, we begin with a simple auxiliary mixture of the teacher and fake distributions. For clarity, we omit the conditioning variable c. At noise level t, define

$$
p _ { \alpha , t } ( \boldsymbol { x } _ { t } ) = \alpha p _ { T , t } ( \boldsymbol { x } _ { t } ) + ( 1 - \alpha ) p _ { F , t } ( \boldsymbol { x } _ { t } ) , \qquad \alpha \in ( 0 , 1 ) .\tag{8}
$$

Although α is fixed at the distribution level, the corresponding score mixture is sample dependent. Let $\begin{array} { r } { \Delta _ { t } ( x _ { t } ) = \log \frac { p _ { T , t } ( x _ { t } ) } { p _ { F , t } ( x _ { t } ) } } \end{array}$ , and define the generic mixture weighting function

$$
\mathcal { W } _ { \alpha } ( u ) = \frac { \alpha e ^ { u } } { \alpha e ^ { u } + 1 - \alpha } .\tag{9}
$$

Using $p _ { \alpha , t } ( x _ { t } ) = p _ { F , t } ( x _ { t } ) [ \alpha e ^ { \Delta _ { t } ( x _ { t } ) } + 1 - \alpha ]$ and $\nabla _ { x _ { t } } \Delta _ { t } = s _ { T } - s _ { F }$ , the score of the auxiliary mixture becomes

$$
s _ { \alpha } = \mathcal { W } _ { \alpha } ( \Delta _ { t } ) s _ { T } + \left[ 1 - \mathcal { W } _ { \alpha } ( \Delta _ { t } ) \right] s _ { F } .\tag{10}
$$

Therefore, $s _ { F } - s _ { \alpha } = \mathcal { W } _ { \alpha } ( \Delta _ { t } ) ( s _ { F } - s _ { T } )$ , which induces the adaptive DMD correction

$$
g _ { \alpha } = \mathcal { W } _ { \alpha } ( \Delta _ { t } ) g _ { \mathrm { D M D } } .\tag{11}
$$

This sample adaptivity is specific to the arithmetic mixture. By contrast, the geometric interpolation $p _ { \alpha , t } ^ { \mathrm { g e o } } \propto p _ { T , t } ^ { \dot { \alpha } } p _ { F , t } ^ { 1 - \dot { \alpha } }$ yields $s _ { \alpha } ^ { \mathrm { g e o } } = \alpha s _ { T } + ( 1 - \alpha ) s _ { F }$ , and therefore only a uniform rescaling of the DMD correction by α. We therefore adopt the arithmetic mixture to obtain sample-dependent weighting; a detailed comparison is provided in Appendix C.6.

From density ratio to native disagreement. The exact adaptive weight in Eq. 10 depends on the sample-wise log-density ratio $\Delta _ { t } ( x _ { t } )$ , which is unavailable in standard diffusion models. We therefore instantiate the same weighting principle using a tractable signal already produced during DMD training. From Eq. 6, the teacher–fake clean-prediction discrepancy is directly related to the corresponding score discrepancy. We define its noise-normalized magnitude as

$$
r ^ { 2 } = \mathrm { m e a n } \left[ \left( \frac { \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } } { \sigma _ { t } } \right) ^ { 2 } \right] ,\tag{12}
$$

where the mean is taken over the non-batch latent dimensions. By Eq. $6 , ( \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } ) / \sigma _ { t }$ ≈ $( \sigma _ { t } / \alpha _ { t } ) ( s _ { F } - s _ { T } )$ , so r measures the magnitude of the same native teacher–fake discrepancy that drives the DMD correction, after accounting for its noise-dependent scale.

To convert this discrepancy into an adaptive weight, we use the monotone quadratic-energy surrogate $\widetilde { \Delta } ( r ) = - r ^ { 2 } / 2$ for the inaccessible sample-wise density evidence. Importantly, $\widetilde { \Delta } ( r )$ is not intended to estimate the true log-density ratio $\Delta _ { t } ( x _ { t } )$ ; it provides a conservative, disagreement-dependent surrogate within the same mixture-weighting form. The practical DA-DMD weight is therefore

$$
\omega _ { \alpha } ( r ) : = \mathcal { W } _ { \alpha } \left( - \frac { r ^ { 2 } } { 2 } \right) = \frac { \alpha e ^ { - r ^ { 2 } / 2 } } { \alpha e ^ { - r ^ { 2 } / 2 } + 1 - \alpha } .\tag{13}
$$

The weight decreases monotonically with disagreement, satisfying $\omega _ { \alpha } ( 0 ) = \alpha$ and $\omega _ { \alpha } ( \boldsymbol r ) \to 0$ as $r  \infty$ . Hence, the same coefficient α that defines the auxiliary mixture also determines the maximum retained DMD update strength in the practical weighting rule, while disagreement introduces additional sample-adaptive attenuation.

Selective high-noise intervention. The analysis in Sec. 1 shows that unusually high-energy DMD updates concentrate in the high-noise regime. We therefore restrict disagreement-aware attenuation to a predefined high-noise region H. Let $\chi = \mathbf { 1 } [ t \in \mathcal { H } ]$ . The final DA-DMD update is

$$
g _ { \mathrm { D A - D M D } } = \mathrm { s g } \left[ 1 - \chi + \chi \omega _ { \alpha } ( r ) \right] g _ { \mathrm { D M D } } ,\tag{14}
$$

where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. Equivalently,

$$
g _ { \mathrm { D A - D M D } } = \left\{ \begin{array} { l l } { g _ { \mathrm { D M D } } , } & { t \notin \mathcal { H } , } \\ { \omega _ { \alpha } ( r ) g _ { \mathrm { D M D } } , } & { t \in \mathcal { H } . } \end{array} \right.
$$

Eq. 14 separates the roles of noise level and disagreement: the high-noise gate determines where DA DMD intervenes, while normalized teacher–fake disagreement determines how strongly the update is attenuated. Appendix Fig. 7 visualizes this weighting behavior, including the exact preservation outside H and the progressively stronger attenuation within the active region.

The resulting effect on optimization is illustrated in Appendix Fig. 8, and Fig. 9 and 10. In the controlled example, high-disagreement samples are associated with larger raw DMD updates, whereas DA-DMD reduces their influence while retaining the underlying DMD correction direction. This provides an intuitive view of how disagreement-aware attenuation can prevent a small subset of aggressive updates from disproportionately shaping the optimization trajectory.

All quantities required by DA-DMD are already available during standard DMD training. In all experiments, both our DMD2 baseline and DA-DMD use only the score-based distribution-matching branch, without the real-data adversarial objective. DA-DMD therefore modifies only this DMD update through disagreement-aware reweighting, while the student rollout, teacher query, fake-model optimization, and inference procedure remain unchanged.

```latex
Algorithm 1 DA-DMD
Require: Teacher $T ,$ , student $G _ { \theta } ,$ fake model $F _ { \phi } ;$ weighting coefficient $\alpha \in ( 0 , 1 )$ ; noise schedule
$\{ \sigma _ { t } \} ;$ ; high-noise region H
1: for each training iteration do
2: Obtain $x _ { 0 } ^ { \theta } , \bar { x _ { t } } , \hat { x } _ { 0 , T } , \hat { x } _ { 0 , F } ,$ , and $g _ { \mathrm { D M D } }$ as in DMD2 ▷ standard DMD
3: $\begin{array} { r } { r ^ { 2 } \gets \mathrm { m e a n } \Bigg \lceil \left( \frac { \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } } { \sigma _ { t } } \right) ^ { 2 } \Bigg \rceil } \end{array}$ ▷ measure normalized disagreement
4: $\chi  \mathbf { 1 } [ t \in \bar { \mathcal { H } } ]$ ▷ activate in high-noise region
$\alpha e ^ { - r ^ { 2 } / 2 }$
5: $\omega _ { \alpha } ( r ) \gets \frac { \mathbf { u } \mathbf { c } } { \alpha e ^ { - r ^ { 2 } / 2 } + 1 - \alpha }$ ▷ compute adaptive retention
6: $g _ { \mathrm { D A - D M D } } \gets \mathrm { s g } [ 1 - \chi + \chi \omega _ { \alpha } ( r ) ]$ g<sub>DMD</sub> ▷ selectively reweight DMD
7: Update $G _ { \theta }$ with $g _ { \mathrm { D A - D M D } } ;$ update $F _ { \phi }$ as in DMD2
8: end for
```

## 2.3 DA-DMD AS SELECTIVE ADAPTIVE STEP CONTROL

The mixture construction in Sec. 2.2 explains how the adaptive weighting rule arises. We now turn to a complementary question: when is attenuating a DMD update actually preferable to applying it at full strength? Importantly, DA-DMD does not interpret high disagreement as evidence of an incorrect update. Instead, our analysis considers a subtler regime in which the update remains aligned with a useful descent direction but its magnitude becomes overly aggressive.

Let $g = g _ { \mathrm { D M D } }$ , and consider a generic attenuated step $\theta ^ { + } = \theta - \eta a g$ , where $a \in [ 0 , 1 ]$ . Assume that the DMD objective $\mathcal { L } ( \boldsymbol { \theta } )$ is β-smooth and that the conditional DMD update has positive descent alignment. Conditioned on noise level t and disagreement r, define $\bar { A ( t , r ) } = \bar { \mathbb { E } } [ \langle \nabla \mathcal { L } ( \theta ) , g \rangle$ $t , { \check { r } } ] > 0$ as the useful descent alignment and $B ( t , r ) { \overset {  } { = } } \mathbb { E } [ \| g \| ^ { 2 } \mid t , r ] > 0$ as the update energy. The standard smoothness inequality gives

$$
\mathbb { E } \left[ \mathcal { L } ( { \theta } ^ { + } ) \mid t , r \right] \leq \mathcal { L } ( { \theta } ) - { \eta } a A ( t , r ) + \frac { \beta { \eta } ^ { 2 } a ^ { 2 } } { 2 } B ( t , r ) .\tag{15}
$$

Minimizing the right-hand side over $a \in [ 0 , 1 ]$ yields

$$
a ^ { \star } ( t , r ) = \operatorname* { m i n } \left\{ 1 , \frac { A ( t , r ) } { \beta \eta B ( t , r ) } \right\} .\tag{16}
$$

Eq. 16 exposes the key optimization principle behind DA-DMD: the preferred update strength decreases when update energy becomes large relative to useful descent alignment. Thus, a DMD update can remain positively aligned with the descent direction while still being too aggressive to apply at full strength. In other words, a useful update direction does not necessarily imply an appropriate update magnitude. More explicitly, for any fixed attenuation factor $a \in \mathsf { \Gamma } ( 0 , 1 )$ , the correspond ing conditional one-step bound is tighter than that of vanilla DMD whenever $\bar { B ( t , r ) / A ( t , r ) } >$ $2 / [ \beta \eta ( 1 + a ) ]$ ]. The complete derivation is provided in Appendix C.2.

The analysis further connects the empirical disagreement signal to adaptive step control. Within the active high-noise region $\mathcal { H } ,$ suppose that the conditional energy-to-alignment ratio $B ( t , r ) / A ( t , r )$ is nondecreasing with disagreement r. Eq. 16 then implies

$$
r \uparrow \implies a ^ { \star } ( t , r ) \downarrow .
$$

Such behavior is consistent with the practical DA-DMD weight $\omega _ { \alpha } ( \boldsymbol r )$ in Eq. 13, which assigns stronger attenuation to larger disagreement. Under this assumption, teacher–fake disagreement therefore serves as an intrinsic control signal that follows the same monotonic step-control principle suggested by the smoothness analysis.

DA-DMD further localizes this robustification to the region where aggressive updates empirically concentrate. Outside the active region ${ \mathcal { H } } ,$ vanilla DMD is recovered exactly, while any induced reduction in update energy is confined to the high-noise regime. In addition, the disagreementdependent influence is redescending:

$$
t \not \in \mathcal { H } \implies g _ { \mathrm { D A - D M D } } = g _ { \mathrm { D M D } } , \qquad \operatorname* { l i m } _ { r \to \infty } r \omega _ { \alpha } ( r ) = 0 .
$$

Table 1: Quantitative comparison on DrawBench. DA-DMD provides broad improvements over DMD2 across multiple diffusion backbones and evaluation metrics.
<table><tr><td>Backbone</td><td>Method</td><td>PickScore↑</td><td>ImageReward↑</td><td>HPSv2↑</td><td>AES↑</td><td>VQ-R1↑</td><td>Avg. WR↑</td></tr><tr><td rowspan="2">SANA</td><td>DMD2 (Yin et al., 2024a)</td><td>22.53</td><td>95.05</td><td>29.67</td><td>5.820</td><td>4.580</td><td></td></tr><tr><td>DA-DMD (Ours)</td><td>22.56 +0.03</td><td>101.58 +6.53</td><td>29.87 +0.20</td><td>5.945 +0.125</td><td>4.643 +0.063</td><td>56.0%</td></tr><tr><td rowspan="2">SANA-1.5</td><td>DMD2 (Yin et al., 2024a)</td><td>22.53</td><td>98.57</td><td>29.06</td><td>5.858</td><td>4.525</td><td></td></tr><tr><td>DA-DMD (Ours)</td><td>22.58 +0.05</td><td>102.09 +3.52</td><td>29.35 +0.29</td><td>5.916 +0.058</td><td>4.549 +0.024</td><td>57.0%</td></tr><tr><td rowspan="7">SD3-M</td><td>DMD2 (Yin et al., 2024a)</td><td>22.33</td><td>98.56</td><td>30.00</td><td>5.463</td><td>4.534</td><td></td></tr><tr><td>Hyper-SD3 (Ren et al., 2024)</td><td>21.99</td><td>67.67</td><td>26.30</td><td>5.305</td><td>4.292</td><td>28.8%</td></tr><tr><td>SiD-DiT (Zhou et al., 2024)</td><td>22.46</td><td>91.31</td><td>28.01</td><td>5.287</td><td>4.599</td><td>39.8%</td></tr><tr><td>SiD-Adv (Zhou et al., 2024)</td><td>22.53</td><td>90.67</td><td>29.62</td><td>5.560</td><td>4.624</td><td>50.2%</td></tr><tr><td>TDM (Luo et al., 2025)</td><td>22.49</td><td>101.13</td><td>29.81</td><td>5.474</td><td>4.693</td><td>52.2%</td></tr><tr><td>DA-DMD (Ours)</td><td>22.56</td><td>110.72</td><td>30.39</td><td>5.556</td><td>4.649</td><td>61.2%</td></tr><tr><td></td><td>+0.23</td><td>+12.16</td><td>+0.39</td><td>+0.093</td><td>+0.115</td><td></td></tr><tr><td rowspan="6">SD3.5-M</td><td>DMD2 (Yin et al., 2024a)</td><td>22.49</td><td>97.82</td><td>28.70</td><td>5.366</td><td>4.647</td><td></td></tr><tr><td>SiD-DiT (Zhou et al., 2024)</td><td>22.57</td><td>91.15</td><td>28.92</td><td>5.463</td><td>4.682</td><td>48.0%</td></tr><tr><td>SiD-Adv (Zhou et al., 2024)</td><td>22.50</td><td>93.75</td><td>28.97</td><td>5.574</td><td>4.558</td><td>48.8%</td></tr><tr><td>DP-DMD (Wu et al., 2026a)</td><td>22.32</td><td>91.37</td><td>28.08</td><td>5.299</td><td>4.553</td><td>39.4%</td></tr><tr><td>SenseFlow (Ge et al., 2025)</td><td>22.42</td><td>89.37</td><td>29.43</td><td>5.564</td><td>4.648</td><td>56.8%</td></tr><tr><td>DA-DMD (Ours)</td><td>22.51 +0.02</td><td>104.20 +6.38</td><td>29.57 +0.87</td><td>5.474 +0.108</td><td>4.700 +0.053</td><td>59.5%</td></tr><tr><td rowspan="6">SD3.5-L</td><td>DMD2 (Yin et al., 2024a)</td><td>22.56</td><td>94.33</td><td>28.23</td><td>5.378</td><td>4.550</td><td></td></tr><tr><td>SiD-DiT (Zhou et al., 2024)</td><td>22.67</td><td>101.77</td><td>29.35</td><td>5.543</td><td>4.566</td><td>53.8%</td></tr><tr><td>SiD-Adv (Zhou et al., 2024)</td><td>22.69</td><td>112.92</td><td>30.52</td><td>5.743</td><td>4.638</td><td>65.6%</td></tr><tr><td>SD3.5-L Turbo (Sauer et al., 2024)</td><td>22.65</td><td>91.07</td><td>28.47</td><td>5.631</td><td>4.608</td><td>54.4%</td></tr><tr><td>SenseFlow (Ge et al., 2025)</td><td>22.70</td><td>112.52</td><td>30.18</td><td>5.421</td><td>4.652</td><td>62.2%</td></tr><tr><td>DA-DMD (Ours)</td><td>22.73 +0.17</td><td>113.88 +19.55</td><td>30.93 +2.70</td><td>5.506 +0.128</td><td>4.587 +0.037</td><td>64.4%</td></tr></table>

Consequently, DA-DMD moderates aggressive high-noise updates without globally weakening the original distribution-matching signal, while progressively reducing the influence of extreme teacher– fake disagreement. Complete proofs of these properties, together with complementary bias–variance and proximal interpretations, are provided in Appendix C.3.

## 3 EXPERIMENTS

In this section, we conduct extensive experiments to validate DA-DMD. Following the framework developed in Secs. 2.2 and 2.3, these experiments primarily serve to verify the effectiveness of disagreement-aware update control and the rationality of our selective high-noise design across multiple diffusion backbones and text-to-image benchmarks.

## 3.1 EXPERIMENTAL SETUP

We evaluate DA-DMD on 5 text-to-image diffusion backbones, including SANA (Xie et al., 2024), SANA-1.5 (Xie et al., 2025), SD3-M, SD3.5-M, and SD3.5-L (Esser et al., 2024). Our evaluation covers 4 complementary benchmarks: DrawBench (Saharia et al., 2022), PartiPrompts (Yu et al., 2022), GenEval2 (Kamath et al., 2025), and TIIF-Bench (Wei et al., 2025), spanning general text-to-image generation, compositional fidelity, and fine-grained instruction following. We report PickScore (Kirstain et al., 2023), ImageReward (Xu et al., 2023), HPSv2 (Wu et al., 2023), AES (Schuhmann, 2022), VisualQuality-R1 (Wu et al., 2026c), and average pairwise win rate as complementary measures of preference alignment and visual quality. DMD2 (Yin et al., 2024a) serves as our primary baseline, together with representative few-step distillation and acceleration methods where applicable. Detailed benchmark descriptions, evaluation protocols, model configurations, and implementation settings are provided in the Appendix Sec. A.

## 3.2 MAIN RESULTS

Quantitative comparison. Tab. 1 and Appendix Tab. 3 report the main results on DrawBench and PartiPrompts. Across all 5 backbones, DA-DMD consistently improves the corresponding DMD2 baseline on both benchmarks, demonstrating that the benefit of disagreement-aware update control generalizes across different model families and scales. The gains are particularly pronounced on the SD3/SD3.5 family. On DrawBench, DA-DMD achieves average pairwise win rates of 61.2%, 59.5%, and 64.4% on SD3-M, SD3.5-M, and SD3.5-L, respectively, while improving multiple preference- and quality-oriented metrics. Similar trends are observed on PartiPrompts, where DA-DMD reaches 61.2% and 61.4% average win rates on SD3.5-M and SD3.5-L. Compared with representative few-step distillation and acceleration baselines, DA-DMD remains competitive or superior across most evaluation dimensions while preserving the original inference procedure and sampling cost. Additional results on GenEval2 and TIIF-Bench in the Appendix Tab. 4, 5, 6, 7 further show that these improvements extend to compositional generation and fine-grained instruction following.

![](images/4b1b2e50881d2fe0a7e3c6357c96e377d771b43ce651c0474bbe5fc3eb082fed.jpg)

![](images/5b12f84fd9704003436a9976d18d4e882da089b30ea9aa5dc89062498475c854.jpg)

![](images/ccb973bda81cb51a87cee04a68f007b2e71b347785b0d8949bd630450d1e25a1.jpg)

![](images/55a2057a0c5d6a1012f0822b25c91866f82e81fe4da2f6573f4e76cc8189ac98.jpg)

![](images/8569029695aa0e7213233a0042ddfe1a180a33e4d938a255c81e1b9e247ac5cc.jpg)

![](images/7c963a99b41f214fc5d7a52530f11be45c6a7d76dd8b23c062e22b722da4d7cd.jpg)  
Figure 4: Sensitivity to the weighting coefficient α on SD3.5-M. Moderate weighting, particularly $\alpha = 0 . 3$ , provides the best overall trade-off across metrics on DrawBench and PartiPrompts.

Diagnosing DMD updates. Beyond generation quality, we examine whether teacher–fake disagreement provides a meaningful diagnostic of DMD optimization. Across DIGITS, CIFAR-10, and CIFAR-100, high-disagreement samples produce larger raw DMD updates and can disproportionately steer the optimization trajectory, whereas DA-DMD attenuates these corrections and recovers target-mode occupancy more faithfully (Figs. 8, 9, and 10). Consistently, larger normalized disagreement within the high-noise regime coincides with higher-energy raw DMD updates across all three datasets (Fig. 1). At model scale, SD3.5-M further shows that high-energy updates concentrate in the high-noise portion of the trajectory (Fig. 2). Together, these observations support the design of DA-DMD: noise level determines where to intervene, while disagreement determines how strongly.

## 3.3 ABLATION STUDY

Where should DA-DMD intervene? We first investigate where disagreement-aware attenuation should be applied. As shown in Fig. 6, restricting DA-DMD to the high-noise region provides the strongest overall trade-off across DrawBench and PartiPrompts, with particularly clear gains in average pairwise win rate and most preference- and quality-oriented metrics. In contrast, global, middle-noise, and low-noise intervention is generally less effective and can underperform vanilla DMD on several metrics. These results indicate that disagreement-aware weighting is not uniformly beneficial across the diffusion trajectory, supporting our selective high-noise design.

Sensitivity to the weighting coefficient. We next study the effect of the weighting coefficient α, which controls the maximum retained update strength in the active region through $\omega _ { \alpha } ( 0 ) = \alpha$ Fig. 4 shows that moderate weighting consistently provides the best overall trade-off across evaluation metrics, with $\alpha = 0 . 3$ performing particularly well on both DrawBench and PartiPrompts. Smaller values tend to over-attenuate useful updates, whereas larger values provide weaker correction. Overall, DA-DMD exhibits a clear and interpretable sensitivity to α, favoring moderate intervention over either extreme.

## 3.4 QUALITATIVE RESULTS

Fig. 5 and Appendix Fig. 11 compare DA-DMD with representative few-step distillation and acceleration baselines on SD3.5-M. Across challenging prompts involving counting, text rendering, attribute binding, and object identity, DA-DMD produces generations that more faithfully satisfy the specified visual constraints while maintaining strong perceptual quality. These qualitative improvements are consistent with the quantitative gains reported in Sec. 3.2.

![](images/3969b2f17b597956e797381b9b39173c84d69650a63d631a26935ab3599e44cf.jpg)  
Figure 5: Qualitative comparison with representative distillation baselines on SD3.5-M.

We further compare DA-DMD directly against DMD2 across all evaluated backbones. As shown in Appendix Fig. 12, 13, 14, 15, and 16, DA-DMD consistently improves fine-grained prompt adherence across diverse compositional challenges, including object counting, spatial relations, relative attributes, color control, and structured multi-object generation. Together, these examples suggest that disagreement-aware update control improves not only automatic evaluation scores but also visually observable compositional fidelity.

![](images/8a9f1602c1d1dd6ea5b64a32e29d406bd593117d6ecbc4f8249a66b486188a5a.jpg)  
Figure 6: Effect of the intervention region on SD3.5-M. High-noise intervention provides the strongest overall trade-off across DrawBench and PartiPrompts, while global, middle-noise, and low-noise attenuation are generally less effective.

Limitations. DA-DMD currently relies on a predefined high-noise intervention region and a handdesigned disagreement-to-weight mapping. While these choices are effective across the evaluated settings, the preferred intervention strength can vary across model families. A promising direction for future work is therefore to make both components more adaptive, allowing the model to determine when intervention is beneficial and how strongly the DMD correction should be modulated.

Conclusion. DA-DMD reuses the native teacher–fake discrepancy in DMD as an intrinsic signal for adaptive update control. Our key observation is that, beyond forming the distribution-matching correction itself, the magnitude of this discrepancy also provides useful information about the appropriate update strength, particularly in the high-noise regime where aggressive corrections tend to concentrate. Based on this observation, DA-DMD selectively attenuates high-noise updates according to disagreement while exactly preserving vanilla DMD outside the active region. Our analysis interprets this mechanism as adaptive step control, providing a principled explanation for why a directionally useful DMD correction can still benefit from a reduced step magnitude. Extensive experiments across multiple diffusion backbones and text-to-image benchmarks demonstrate broad improvements over DMD2 without introducing additional supervision or inference-time overhead. More broadly, our results suggest that native signals already present in distribution matching can be leveraged not only to construct the correction, but also to control how strongly it is applied.

## AI USE STATEMENT

Generative AI tools were used to assist with research code implementation, manuscript organization, language polishing, and the presentation of mathematical derivations. All AI-assisted content was independently reviewed and verified by the authors, including manual inspection of generated code, mathematical arguments, experimental analyses, and manuscript revisions. The final research decisions, experimental design, technical claims, and scientific conclusions were made and validated by the authors, who take full responsibility for the content of this work.

## ETHICS STATEMENT

This work studies efficient distillation of text-to-image diffusion models and aims to reduce the computational cost of high-quality image generation. As with other generative modeling techniques, accelerated generation may inherit broader risks associated with image synthesis, including misuse for deceptive or inappropriate content. DA-DMD does not introduce new data collection, user profiling, or additional inference-time models, and is intended for responsible research and legitimate generative applications.

## REPRODUCIBILITY STATEMENT

We provide the formulation and training procedure of DA-DMD, including the disagreement measure, adaptive weighting rule, high-noise intervention strategy, and complete training pseudocode. We evaluate the method across multiple diffusion backbones and benchmarks, with experimental settings, evaluation protocols, ablations, and additional theoretical derivations provided in the Appendix. These details are intended to support reproducibility and facilitate comparison with existing DMD-based distillation methods.

## REFERENCES

E. Alpaydin and C. Kaynak. Optical Recognition of Handwritten Digits. UCI Machine Learning Repository, 1998.

Lichen Bai, Zikai Zhou, Shitong Shao, Wenliang Zhong, Shuo Yang, Shuo Chen, Bojun Chen, and Zeke Xie. Optimizing few-step generation with adaptive matching distillation. arXiv preprint arXiv:2602.07345, 2026.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow Transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Xingtong Ge, Xin Zhang, Tongda Xu, Yi Zhang, Xinjie Zhang, Yan Wang, and Jun Zhang. Senseflow: Scaling distribution matching for flow-based text-to-image distillation. arXiv preprint arXiv:2506.00523, 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 2020.

Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Liuzhuozheng Li, Hengzhuang Li, Xin Jin, David Liu, Changsheng Lu, Zhen Li, et al. Distribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649, 2025.

Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in neural information processing systems, 2022.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 2023.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Buhua Liu, Shitong Shao, Bao Li, Lichen Bai, Zhiqiang Xu, Haoyi Xiong, James T. Kwok, Sumi Helal, and Zeke Xie. Alignment of diffusion models: Fundamentals, challenges, and future. 2026.

Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Hongsheng Li, et al. Decoupled DMD: CFG augmentation as the spear, distribution matching as the shield. arXiv preprint arXiv:2511.22677, 2025.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025.

Yihong Luo, Tianyang Hu, Weijian Luo, and Jing Tang. TDM-R1: Reinforcing few-step diffusion models with non-differentiable reward. arXiv preprint arXiv:2603.07700, 2026.

Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-SD: Trajectory segmented consistency model for efficient image synthesis. Advances in neural information processing systems, 2024.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXXXVI, 2024.

Christoph Schuhmann. Improved aesthetic predictor. https://github.com/ christophschuhmann/improved-aesthetic-predictor, 2022. GitHub repository.

Shitong Shao, Lichen Bai, Pengfei Wan, James Kwok, and Zeke Xie. Efficient video diffusion models: Advancements and challenges, 2026. URL https://arxiv.org/abs/2604.15911.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, 2023.

Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. TIIF-Bench: How does your T2I model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.

Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. arXiv preprint arXiv:2602.03139, 2026a.

Tianhe Wu, Kun Yan, Zikai Zhou, Lihan Jiang, Jiahao Li, Jie Zhang, Kaiyuan Gao, Ningyuan Tang, Shengming Yin, Xiaoyue Chen, Xiao Xu, Yilei Chen, Yuxiang Chen, Yan Shu, Yixian Xu, Yanran Zhang, Zihao Liu, Zhendong Wang, Zekai Zhang, Deqing Li, Liang Peng, Yi Wang, Jingren Zhou, and Chenfei Wu. Qwen-Image-Flash: Beyond objective design, 2026b. URL https://arxiv.org/abs/2606.03746.

Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang, and Kede Ma. Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank. Advances in Neural Information Processing Systems, 2026c.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human Preference Score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.

Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inferencetime compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 2023.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and ¨ Bill Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, pp. 47455–47487, 2024a.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman,¨ and Taesung Park. One-step diffusion with distribution matching distillation. In IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.

Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.

## A IMPLEMENTATION DETAILS

In this section, we provide an overview of the benchmarks, evaluation metrics, diffusion models and the hyperparameter settings in our main paper.

## A.1 BENCHMARK

DrawBench. We evaluate general text-to-image generation capabilities using DrawBench (Saharia et al., 2022), a challenging benchmark designed to probe a broad range of semantic and compositional skills. DrawBench contains 200 manually curated prompts organized into 11 categories, covering challenging aspects such as color and attribute binding, counting, spatial relations, text rendering, rare concepts, misspellings, and unusual object interactions. It further includes complex and long-form descriptions that require models to correctly compose multiple visual concepts. The diversity and compact size of DrawBench make it particularly suitable for fine-grained evaluation of both prompt alignment and perceptual generation quality.

PartiPrompts. We further evaluate on PartiPrompts (P2) (Yu et al., 2022), a large-scale and diverse benchmark introduced for evaluating content-rich text-to-image generation. PartiPrompts consists of 1,632 English prompts spanning 12 semantic categories and 11 challenge aspects. The semantic categories cover a broad range of visual concepts, including people, animals, scenes, artifacts, abstract concepts, and world knowledge, while the challenge aspects range from basic generation and simple details to fine-grained attributes, quantity, spatial positioning, linguistic structures, writing and symbols, and complex compositions. This structured organization enables evaluation across both prompt content and generation difficulty, making PartiPrompts particularly useful for assessing generalization to diverse and compositionally demanding instructions.

TIIF-Bench. To evaluate fine-grained instruction-following ability, we adopt TIIF-Bench (Wei et al., 2025), which is specifically designed to assess how accurately modern text-to-image models interpret and execute complex user instructions. TIIF-Bench contains 5,000 diverse prompts organized across multiple dimensions and three levels of difficulty and complexity. For each instruction, the benchmark provides both a concise version and a rhetorically enriched long-form version with the same underlying semantics, allowing systematic evaluation of robustness to prompt length and linguistic complexity. Beyond conventional compositional requirements, TIIF-Bench also covers challenging capabilities such as text rendering, style control, and designer-level generation tasks. These characteristics make it well suited for measuring fine-grained instruction adherence under realistic and complex generation settings.

GenEval 2. We additionally employ GenEval 2 (Kamath et al., 2025), an updated compositional text-to-image benchmark designed to address the increasing saturation and benchmark drift of earlier automatic evaluation suites. GenEval 2 contains 800 prompts with substantially improved coverage of primitive visual concepts and higher degrees of compositionality, including objects, attributes, spatial relations, and counting. Each prompt is decomposed into a set of atomic visual requirements, enabling evaluation not only at the prompt level but also at the level of individual compositional skills. GenEval 2 is evaluated using Soft-TIFA, a VQA-based evaluation framework that aggregates soft judgments over these visual primitives and is designed to better align automatic evaluation with human judgment. The benchmark therefore provides a fine-grained diagnostic of whether a model can simultaneously satisfy multiple visual constraints in increasingly complex prompts.

## A.2 EVALUATION METRICS

PickScore. PickScore (Kirstain et al., 2023) is a CLIP-based human-preference metric trained on the Pick-a-Pic dataset, which contains pairwise preferences collected from real users of text-toimage systems. It jointly considers the input prompt and generated image to estimate human preference, providing an automatic measure of both semantic alignment and overall generation quality.

ImageReward. ImageReward (Xu et al., 2023) is a general-purpose text-to-image reward model trained on expert-annotated image comparisons. It is designed to capture human preferences ove generated images while accounting for both text–image alignment and perceptual quality, and has been widely adopted for automatic evaluation of text-to-image generation.

HPS v2. Human Preference Score v2 (HPS v2) (Wu et al., 2023) is a CLIP-based preference model trained on the Human Preference Dataset v2 (HPD v2), which contains large-scale pairwise human preference annotations. HPS v2 is designed to provide robust preference prediction across diverse image distributions and serves as a complementary measure of human-aligned generation quality.

AES. We use the Aesthetic Score (AES) (Schuhmann, 2022) to evaluate the perceptual attractiveness of generated images. AES is predicted from CLIP image representations using an aesthetic predictor trained on human-rated images, providing a prompt-independent measure of visual aesthetics. Unlike preference metrics that jointly consider text and image, AES primarily reflects the overall aesthetic quality of the generated image.

VisualQuality-R1. We further employ VisualQuality-R1 (Wu et al., 2026c), a reasoningenhanced no-reference image quality assessment model trained with reinforcement learning to rank. VisualQuality-R1 evaluates perceptual image quality without requiring a reference image and provides a complementary assessment of visual fidelity beyond text–image preference metrics.

## A.3 MODEL BACKBONES

We evaluate DA-DMD on five text-to-image diffusion backbones from two representative model families: SANA, SANA-1.5, Stable Diffusion 3 Medium (SD3-M), Stable Diffusion 3.5 Medium (SD3.5-M), and Stable Diffusion 3.5 Large (SD3.5-L). These models cover different architectures and model scales, allowing us to evaluate whether the proposed disagreement-aware optimization generalizes beyond a particular backbone.

SANA and SANA-1.5. SANA (Xie et al., 2024) is an efficient text-to-image framework based on a linear Diffusion Transformer architecture, designed to achieve high-resolution image generation with substantially reduced computational cost. SANA-1.5 (Xie et al., 2025) further scales the SANA family through improved training-time and inference-time scaling strategies. We include both generations of SANA to evaluate DA-DMD across closely related but differently scaled diffu sion backbones.

Stable Diffusion 3 and 3.5. Stable Diffusion 3 (Esser et al., 2024) introduces the Multimodal Diffusion Transformer (MMDiT) architecture for text-to-image generation. We use SD3-M together with the subsequent SD3.5-M and SD3.5-L variants, covering both medium- and large-scale models. Evaluating DA-DMD across these three backbones allows us to study its behavior under different model capacities within the same general architecture family.

## A.4 HYPERPARAMETER SETTINGS

Unless otherwise specified, DA-DMD follows the same training recipe as the corresponding DMD2 baseline. Following the multi-step distribution-matching and fake-model optimization framework of DMD2, we use only the score-based distribution-matching objective in all experiments and disable the real-data adversarial objective for both DMD2 and DA-DMD. Therefore, the only difference between the two methods is the disagreement-aware reweighting applied to the DMD update.

All models are trained at 1024 × 1024 resolution with a 4-step student. The teacher is frozen, while both the student and fake model are initialized from the corresponding pretrained backbone. We use AdamW with a learning rate of $5 \times 1 0 ^ { - 6 }$ for both the student and fake model, and perform five fake-model optimization steps per student update. For DA-DMD, the active high-noise region corresponds to the first 30% of the noise trajectory, i.e., normalized noise level $\sigma _ { t } ^ { \mathrm { n o r m } } \geq 0 . 7 ;$ outside this region, the original DMD update is left unchanged. The main model-specific settings are summarized in Tab. 2.

For all benchmarks, distilled models are evaluated using 4 sampling steps at 1024 × 1024 resolution with guidance scale 1.0. We use the same evaluation seed and metric implementation for DMD2 and DA-DMD. The weighting coefficient α and intervention region are further analyzed in Sec. 3.3.

Table 2: Main training hyperparameters. DA-DMD uses the same optimization and sampling configuration as DMD2, with model-specific teacher guidance and disagreement weighting coefficient α.
<table><tr><td>Backbone</td><td>Teacher NFE</td><td>Teacher CFG</td><td>α</td></tr><tr><td>SANA</td><td>20</td><td>4.5</td><td>0.2</td></tr><tr><td>SANA-1.5</td><td>20</td><td>4.5</td><td>0.2</td></tr><tr><td>SD3-M</td><td>30</td><td>7.0</td><td>0.3</td></tr><tr><td>SD3.5-M</td><td>30</td><td>3.5</td><td>0.3</td></tr><tr><td>SD3.5-L</td><td>30</td><td>3.5</td><td>0.7</td></tr></table>

The coefficient α controls the maximum retained DMD update strength within the active high-noise region through $\omega _ { \alpha } ( 0 ) = \alpha$ . We do not assume a universal value across model families. Instead, α is selected independently for each backbone from a small candidate set using the same tuning protocol. Our ablations show that performance is sensitive to the amount of attenuation and that moderate retention generally performs well, while the preferred value can shift across architectures. We therefore treat α as a model-dependent optimization hyperparameter rather than attributing its variation to model scale or parameter count.

## B RELATED WORK

Few-step diffusion distillation. Diffusion and flow-based generative models achieve strong synthesis quality but typically require iterative sampling, motivating extensive work on reducing the number of function evaluations at inference time. Early approaches compress multi-step teachers through progressive distillation (Salimans & Ho, 2022), while consistency-based methods learn mappings that directly connect noisy states to clean samples or enforce consistency along probability-flow trajectories (Song et al., 2023; Luo et al., 2023). More recent methods explore trajectory-, transition-, and output-level distillation objectives to further reduce sampling steps while preserving generation quality (Luo et al., 2026; Zhou et al., 2024). These approaches primarily constrain the student through teacher-generated targets or trajectories. In contrast, distribution matching methods optimize the student at the level of its induced marginal distribution, allowing the student trajectory to deviate from the teacher’s original sampling path.

Distribution matching distillation. Distribution Matching Distillation (DMD) (Yin et al., 2024b) introduced a distribution-level approach to few-step generation by minimizing an approximate KL divergence between teacher and student distributions. Its gradient is estimated through the difference between a pretrained teacher score and a “fake” score model trained to track the evolving student distribution. DMD2 (Yin et al., 2024a) substantially improves this framework by removing the original regression-data requirement, stabilizing fake-model estimation with a two-time-scale update rule, incorporating real-data adversarial supervision, and extending DMD to multi-step generation. Subsequent work has improved different aspects of the DMD framework. Diversity-Preserved DMD (Wu et al., 2026a) separates the roles of distilled steps to mitigate diversity collapse, while Decoupled DMD (Liu et al., 2025) decomposes the effects of classifier-free guidance augmentation and distribution matching and uses this decomposition to redesign the distillation schedule. Other studies investigate broader design choices surrounding distribution matching, including training recipes, teacher guidance, and data composition (Wu et al., 2026b). Together, these works demonstrate that both the objective and the surrounding optimization procedure are critical to successful few-step distribution matching.

Adaptive optimization of DMD. A growing line of work further studies when standard DMD supervision may become ineffective or unstable. Adaptive Matching Distillation (AMD) (Bai et al., 2026), for example, identifies failure regions in the DMD optimization landscape and employs reward-based proxies together with adaptive correction mechanisms to steer the student away from such regions. Other approaches introduce auxiliary perceptual, adversarial, reward, or task-specific signals to improve fidelity, diversity, or training stability (Jiang et al., 2025; Wu et al., 2026a). These methods highlight the importance of adapting DMD optimization, but typically do so by augmenting the supervision, modifying individual objective components, or introducing additional mechanisms.

Our work is complementary to these directions. Rather than adding a new source of supervision or explicitly detecting failure states with an external proxy, we ask whether DMD’s native teacher– fake discrepancy can itself diagnose the aggressiveness of its updates. We find that disagreement magnitude contains such information and use it as a continuous control signal for selective highnoise attenuation. DA-DMD therefore modifies how strongly existing DMD supervision is applied, rather than introducing a new target for the student.

## C ADDITIONAL THEORETICAL ANALYSIS

This section provides the complete derivations and complementary interpretations underlying the theoretical results in Secs. 2.1–2.3. We first derive the score–clean prediction identity used to express the DMD update in terms of teacher–fake clean predictions. We then analyze disagreementaware attenuation from optimization, robustness, and gradient-estimation perspectives, followed by proximal and probabilistic interpretations of the practical weighting rule.

## C.1 SCORE–CLEAN PREDICTION IDENTITY

We first derive Eq. 4. Consider the Gaussian perturbation process

$$
\begin{array} { r } { x _ { t } = \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{17}
$$

which induces the conditional transition

$$
\begin{array} { r } { q _ { t } ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } \left( x _ { t } ; \alpha _ { t } x _ { 0 } , \sigma _ { t } ^ { 2 } I \right) . } \end{array}\tag{18}
$$

For a conditional clean-data distribution $p _ { 0 } ( x _ { 0 } \mid c )$ , the noisy marginal is

$$
p _ { t } ( x _ { t } \mid c ) = \int p _ { 0 } ( x _ { 0 } \mid c ) q _ { t } ( x _ { t } \mid x _ { 0 } ) d x _ { 0 } .\tag{19}
$$

Applying Fisher’s identity gives

$$
\begin{array} { r l } & { s _ { t } ( x _ { t } , c ) = \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } \mid c ) } \\ & { ~ = \operatorname { \mathbb { E } } \left[ \nabla _ { x _ { t } } \log q _ { t } ( x _ { t } \mid x _ { 0 } ) \mid x _ { t } , c \right] . } \end{array}\tag{20}
$$

For the Gaussian transition in Eq. 18,

$$
\nabla _ { x _ { t } } \log { q _ { t } ( x _ { t } \mid x _ { 0 } ) } = - { \frac { x _ { t } - \alpha _ { t } x _ { 0 } } { \sigma _ { t } ^ { 2 } } } .\tag{21}
$$

Substituting Eq. 21 into Eq. 20 yields

$$
s _ { t } ( x _ { t } , c ) = { \frac { \alpha _ { t } \mathbb { E } [ x _ { 0 } \mid x _ { t } , c ] - x _ { t } } { \sigma _ { t } ^ { 2 } } } ,\tag{22}
$$

which recovers Eq. 4.

For an $x _ { 0 }$ -prediction model trained with squared-error denoising, the population-optimal predictor is

$$
\hat { x } _ { 0 } ^ { \star } ( x _ { t } , t , c ) = \mathbb { E } [ x _ { 0 } \mid x _ { t } , c ] .\tag{23}
$$

Hence, a trained model satisfies $\hat { x } _ { 0 } ( x _ { t } , t , c ) \approx \mathbb { E } [ x _ { 0 } \mid x _ { t } , c ]$ , and Eq. 22 gives

$$
s _ { t } ( x _ { t } , t , c ) \approx \frac { \alpha _ { t } \hat { x } _ { 0 } ( x _ { t } , t , c ) - x _ { t } } { \sigma _ { t } ^ { 2 } } .\tag{24}
$$

Since the teacher and fake model are evaluated at the same noisy sample $x _ { t } .$ , the shared term $- x _ { t } / \sigma _ { t } ^ { 2 }$ cancels:

$$
s _ { F } - s _ { T } \approx \frac { \alpha _ { t } } { \sigma _ { t } ^ { 2 } } \left( \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } \right) .\tag{25}
$$

Thus, the clean-prediction disagreement used by DA-DMD is directly proportional, up to timestepdependent scaling, to the native teacher–fake score discrepancy driving the DMD correction.

## C.2 ONE-STEP ADAPTIVE ATTENUATION

We now provide the complete derivation of the adaptive step-control result summarized in Sec. 2.3. Let $g = g _ { \mathrm { D M D } }$ , and consider a scaled update

$$
\theta ^ { + } = \theta - \eta a g , \qquad a \in [ 0 , 1 ] .\tag{26}
$$

Assume that the DMD objective $\mathcal { L } ( \boldsymbol { \theta } )$ is $\beta \mathrm { . }$ -smooth. Then

$$
\mathcal { L } ( \theta ^ { + } ) \leq \mathcal { L } ( \theta ) - \eta a \langle \nabla \mathcal { L } ( \theta ) , g \rangle + \frac { \beta \eta ^ { 2 } a ^ { 2 } } { 2 } \| g \| ^ { 2 } .\tag{27}
$$

Conditioned on noise level t and disagreement r, define

$$
\begin{array} { r l } & { A ( t , r ) = \mathbb { E } \left[ \langle \nabla \mathcal { L } ( \theta ) , g \rangle \mid t , r \right] > 0 , } \\ & { B ( t , r ) = \mathbb { E } \left[ \Vert g \Vert ^ { 2 } \mid t , r \right] > 0 . } \end{array}\tag{28}
$$

Here, $A ( t , \boldsymbol { r } )$ measures useful descent alignment, while $B ( t , \boldsymbol { r } )$ measures update energy. Taking the conditional expectation of Eq. 27 gives

$$
\mathbb { E } \left[ \mathcal { L } ( { \theta } ^ { + } ) \mid t , r \right] \leq \mathcal { L } ( { \theta } ) - { \eta } a A ( t , r ) + \frac { \beta { \eta } ^ { 2 } a ^ { 2 } } { 2 } B ( t , r ) .\tag{29}
$$

Optimal retained update strength. The a-dependent part of the bound is $Q ( a ) \ = \ - \eta a A +$ ${ \frac { \bar { \beta } \bar { \eta } ^ { 2 } a ^ { 2 } } { \mathrm { ~ \alpha ~ } } } B .$ Differentiating gives $\partial Q / \partial a = - \eta A + \beta \eta ^ { 2 } a B$ , whose unconstrained minimizer is $A / ( \beta \eta B )$ . Enforcing $a \in [ \bar { 0 } , 1 ]$ therefore yields

$$
a ^ { \star } ( t , r ) = \operatorname* { m i n } \left\{ 1 , \frac { A ( t , r ) } { \beta \eta B ( t , r ) } \right\} .\tag{30}
$$

Equivalently,

$$
\frac { B ( t , r ) } { A ( t , r ) } \uparrow \implies a ^ { \star } ( t , r ) \downarrow .\tag{31}
$$

Eq. 31 formalizes the distinction between update direction and update magnitude: a DMD update may remain positively aligned with the descent direction while still being too aggressive to apply at full strength.

When is attenuation preferable to vanilla DMD? For a fixed attenuation $a \in \mathsf { \Gamma } ( 0 , 1 )$ , let $U _ { a }$ denote the right-hand side of Eq. 29, and let $U _ { 1 }$ denote the vanilla-DMD bound. Their difference is

$$
\begin{array} { c } { { U _ { a } - U _ { 1 } = \eta ( 1 - a ) A - \frac { \beta \eta ^ { 2 } } { 2 } ( 1 - a ^ { 2 } ) B } } \\ { { = \eta ( 1 - a ) \left[ A - \frac { \beta \eta } { 2 } ( 1 + a ) B \right] . } } \end{array}\tag{32}
$$

Since $a \in ( 0 , 1 )$ , we obtain

$$
U _ { a } < U _ { 1 } \quad \Longleftrightarrow \quad { \frac { B ( t , r ) } { A ( t , r ) } } > { \frac { 2 } { \beta \eta ( 1 + a ) } } .\tag{33}
$$

Therefore, attenuation admits a strictly tighter one-step smoothness bound whenever update energy is sufficiently large relative to useful descent alignment.

Connection to disagreement. Suppose that within the active high-noise region H, the conditional energy-to-alignment ratio

$$
\rho ( t , r ) = \frac { B ( t , r ) } { A ( t , r ) }\tag{34}
$$

is nondecreasing with disagreement r. Combining this assumption with Eq. 30 gives

$$
r \uparrow \implies a ^ { \star } ( t , r ) \downarrow .\tag{35}
$$

This monotonic trend is consistent with the practical DA-DMD weighting rule in Eq. 13, where larger teacher–fake disagreement induces stronger attenuation. Importantly, this argument does not require high-disagreement updates to be incorrect; it only requires their update energy to grow sufficiently relative to useful descent alignment.

## C.3 SELECTIVE ENERGY CONTROL AND REDESCENDING INFLUENCE

We next establish the selective nature of DA-DMD and characterize its response to extreme disagreement.

Selective update-energy control. Ignoring the stop-gradient operator, which does not change the forward value of the multiplier, Eq. 14 gives

$$
g _ { \mathrm { D A - D M D } } = [ 1 - \chi + \chi \omega _ { \alpha } ( r ) ] g _ { \mathrm { D M D } } , \chi = { \bf 1 } [ t \in \mathcal { H } ] .\tag{36}
$$

Since $\chi \in \{ 0 , 1 \}$

$$
\left[ 1 - \chi + \chi \omega _ { \alpha } ( r ) \right] ^ { 2 } = 1 - \chi \left[ 1 - \omega _ { \alpha } ^ { 2 } ( r ) \right] .\tag{37}
$$

Combining Eqs. 36 and 37 yields

$$
\begin{array} { r l } & { \mathbb { E } \| g _ { \mathrm { D A - D M D } } \| ^ { 2 } = \mathbb { E } \| g _ { \mathrm { D M D } } \| ^ { 2 } } \\ & { \qquad - \mathbb { E } \left[ { \bf 1 } [ t \in \mathcal { H } ] \left( 1 - \omega _ { \alpha } ^ { 2 } ( r ) \right) \| g _ { \mathrm { D M D } } \| ^ { 2 } \right] . } \end{array}\tag{38}
$$

Hence, the instantaneous update-energy reduction induced by DA-DMD is localized entirely to the active region. In particular,

$$
t \notin \mathcal { H } \quad \Longrightarrow \quad g _ { \mathrm { D A - D M D } } = g _ { \mathrm { D M D } } .\tag{39}
$$

Monotonic and redescending influence. Recall the practical weighting rule

$$
\omega _ { \alpha } ( r ) = \frac { \alpha e ^ { - r ^ { 2 } / 2 } } { \alpha e ^ { - r ^ { 2 } / 2 } + 1 - \alpha } .\tag{40}
$$

Its derivative is

$$
\frac { d \omega _ { \alpha } } { d r } = - r \omega _ { \alpha } ( r ) \left[ 1 - \omega _ { \alpha } ( r ) \right] ,\tag{41}
$$

which is strictly negative for $r > 0$ . Thus, larger disagreement always induces stronger attenuation. Moreover,

$$
\omega _ { \alpha } ( r ) \leq \frac { \alpha } { 1 - \alpha } e ^ { - r ^ { 2 } / 2 } .\tag{42}
$$

Multiplying both sides by $r \geq 0$ gives

$$
0 \leq r \omega _ { \alpha } ( r ) \leq \frac { \alpha } { 1 - \alpha } r e ^ { - r ^ { 2 } / 2 } .\tag{43}
$$

Since $r e ^ { - r ^ { 2 } / 2 } \to 0$ as $r  \infty$

$$
\boxed { \operatorname* { l i m } _ { r  \infty } r \omega _ { \alpha } ( r ) = 0 . }\tag{44}
$$

The disagreement-dependent correction therefore exhibits redescending influence: increasingly extreme teacher–fake disagreement receives progressively diminishing effective weight instead of unbounded influence.

Ideal fixed-point preservation. At the ideal distribution-matching solution, the student matches the teacher and the fake model perfectly tracks the student. Hence $s _ { F } = s _ { T }$ , which implies $g _ { \mathrm { D M D } } =$ 0. Eq. 14 therefore gives

$$
g _ { \mathrm { D M D } } = 0 \quad \Longrightarrow \quad g _ { \mathrm { D A - D M D } } = 0 .\tag{45}
$$

Thus, DA-DMD modifies the optimization dynamics while preserving the ideal distributionmatching solution. We do not require or claim that all non-ideal stationary points of DA-DMD and vanilla DMD are identical.

## C.4 BIAS–VARIANCE INTERPRETATION

The smoothness analysis above studies the quality of a single optimization step. A complementary view interprets disagreement-aware attenuation as conditional shrinkage of a stochastic DMD update.

Suppose

$$
g = g ^ { \star } + \epsilon , \qquad \mathbb { E } [ \epsilon \mid t , r ] = 0 ,\tag{46}
$$

where $g ^ { \star }$ denotes the conditional mean update. Define the conditional stochastic variation as

$$
V ( t , r ) = \mathbb { E } \left[ \Vert \epsilon \Vert ^ { 2 } \mid t , r \right] .\tag{47}
$$

The vanilla estimator therefore has $\mathrm { M S E } _ { \mathrm { D M D } } = V$

For a shrunk estimator $g _ { a } = a g , a \in ( 0 , 1 ) , g _ { a } - g ^ { \star } = ( a - 1 ) g ^ { \star } + a \epsilon$ . Using the conditional zero-mean assumption gives

$$
\begin{array} { r } { \mathrm { M S E } ( { a } ) = ( 1 - { a } ) ^ { 2 } \| \boldsymbol { g } ^ { \star } \| ^ { 2 } + { a } ^ { 2 } V . } \end{array}\tag{48}
$$

The first term is the bias introduced by shrinkage, while the second is the remaining stochastic variation.

Comparing Eq. 48 with the vanilla estimator yields

$$
\mathrm { M S E } ( a ) < \mathrm { M S E _ { D M D } } \quad \Longleftrightarrow \quad V > \frac { 1 - a } { 1 + a } \| g ^ { \star } \| ^ { 2 } .\tag{49}
$$

Minimizing Eq. 48 with respect to a gives

$$
a _ { \mathrm { M S E } } ^ { \star } = \frac { \| g ^ { \star } \| ^ { 2 } } { \| g ^ { \star } \| ^ { 2 } + V } .\tag{50}
$$

Therefore,

$$
\frac { \cal { V } } { \| g ^ { \star } \| ^ { 2 } } \uparrow \implies a _ { \mathrm { M S E } } ^ { \star } \downarrow .\tag{51}
$$

This bias–variance view complements the smoothness analysis in Sec. C.2: both favor more conservative updates when the stochastic DMD correction becomes large or uncertain relative to its useful signal.

We emphasize that this interpretation does not assume that disagreement r directly estimates $V ( t , r ) ;$ rather, it provides an independent statistical explanation for why conditional shrinkage can improve a noisy update estimator.

## C.5 PROXIMAL AND ENERGY-BASED INTERPRETATIONS

We finally provide two complementary interpretations of the practical disagreement-aware weighting rule.

Proximal interpretation. Within the active region $t \in \mathcal { H }$ , DA-DMD scales the original teacherdirected correction by $\omega _ { \alpha } ( \boldsymbol r )$ ). Define the adaptive proximal target

$$
\hat { x } _ { 0 , P } = \left[ 1 - \omega _ { \alpha } ( r ) \right] \hat { x } _ { 0 , F } + \omega _ { \alpha } ( r ) \hat { x } _ { 0 , T } .\tag{52}
$$

Then

$$
\hat { x } _ { 0 , F } - \hat { x } _ { 0 , P } = \omega _ { \alpha } ( r ) \left( \hat { x } _ { 0 , F } - \hat { x } _ { 0 , T } \right) .\tag{53}
$$

Thus, scaling the DMD correction by $\omega _ { \alpha } ( \boldsymbol r )$ is equivalent to replacing the full teacher-directed correction with a shorter correction toward an adaptive point between the fake and teacher predictions. As disagreement increases, $\omega _ { \alpha } ( \boldsymbol r )$ decreases and the effective target moves toward ${ \hat { x } } _ { 0 , F } ,$ , yielding a more conservative update. This is the geometric interpretation illustrated in Fig. 3.

Energy-based responsibility interpretation. The same weighting rule also admits an interpretation in terms of normalized energy-based responsibilities. Consider two disagreement-dependent compatibility potentials associated with the teacher-directed and fake-centered components:

$$
\psi _ { T } ( r ) = e ^ { - r ^ { 2 } / 2 } , \qquad \psi _ { F } ( r ) = 1 .\tag{54}
$$

Together with mixture coefficients $\pi _ { T } = \alpha$ and $\tau _ { F } = 1 - \alpha$ , we define the normalized responsibility of the teacher-directed component as

$$
\rho _ { T } ( r ) = { \frac { \pi _ { T } \psi _ { T } ( r ) } { \pi _ { T } \psi _ { T } ( r ) + \pi _ { F } \psi _ { F } ( r ) } } = { \frac { \alpha e ^ { - r ^ { 2 } / 2 } } { \alpha e ^ { - r ^ { 2 } / 2 } + 1 - \alpha } } = \omega _ { \alpha } ( r ) .\tag{55}
$$

Accordingly, $\rho _ { F } ( r ) = 1 - \rho _ { T } ( r )$ . As disagreement increases, the compatibility of the teacherdirected component decreases relative to the fake-centered component, resulting in a progressively smaller teacher responsibility.

Using these responsibilities to interpolate between the two predictions gives

$$
\rho _ { T } ( r ) \hat { x } _ { 0 , T } + \rho _ { F } ( r ) \hat { x } _ { 0 , F } = \omega _ { \alpha } ( r ) \hat { x } _ { 0 , T } + \left[ 1 - \omega _ { \alpha } ( r ) \right] \hat { x } _ { 0 , F } ,\tag{56}
$$

which exactly recovers the proximal target in Eq. 52.

Importantly, ψ<sub>T</sub> and ψ<sub>F</sub> are unnormalized compatibility potentials rather than probability densities over r. Therefore, Eq. 55 should be interpreted as an energy-based normalization of relative responsibilities, rather than as an estimate of the true teacher–fake density ratio.

## C.6 ARITHMETIC VERSUS GEOMETRIC MIXTURES

We briefly compare two natural interpolations between the teacher and fake distributions and explain why DA-DMD adopts the arithmetic mixture.

Arithmetic mixture. Consider

$$
p _ { \alpha , t } ^ { \mathrm { a r i t h } } ( x _ { t } ) = \alpha p _ { T , t } ( x _ { t } ) + ( 1 - \alpha ) p _ { F , t } ( x _ { t } ) .\tag{57}
$$

Its score is

$$
s _ { \alpha } ^ { \mathrm { a r i t h } } = \rho _ { \alpha } ( { x } _ { t } ) s _ { T } + \left[ 1 - \rho _ { \alpha } ( { x } _ { t } ) \right] s _ { F } ,\tag{58}
$$

where

$$
\rho _ { \alpha } ( x _ { t } ) = \frac { \alpha p _ { T , t } ( x _ { t } ) } { \alpha p _ { T , t } ( x _ { t } ) + ( 1 - \alpha ) p _ { F , t } ( x _ { t } ) } = \mathcal { W } _ { \alpha } ( \Delta _ { t } ( x _ { t } ) ) .\tag{59}
$$

Therefore,

$$
s _ { F } - s _ { \alpha } ^ { \mathrm { a r i t h } } = \rho _ { \alpha } ( x _ { t } ) ( s _ { F } - s _ { T } ) .\tag{60}
$$

Although the mixture coefficient α is fixed globally, the effective score-space coefficient $\rho _ { \alpha } ( x _ { t } )$ depends on the sample. The arithmetic mixture therefore naturally induces sample-adaptive modulation of the original DMD correction.

Geometric mixture. Alternatively, consider the normalized geometric interpolation

$$
p _ { \alpha , t } ^ { \mathrm { g e o } } ( x _ { t } ) = \frac { p _ { T , t } ( x _ { t } ) ^ { \alpha } p _ { F , t } ( x _ { t } ) ^ { 1 - \alpha } } { Z _ { \alpha } } ,\tag{61}
$$

where $Z _ { \alpha }$ is the normalization constant. Since $Z _ { \alpha }$ does not depend on $x _ { t }$ ,

$$
\begin{array} { r } { s _ { \alpha } ^ { \mathrm { g e o } } = \nabla _ { x _ { t } } \log p _ { \alpha , t } ^ { \mathrm { g e o } } ( x _ { t } ) } \\ { = \alpha s _ { T } + ( 1 - \alpha ) s _ { F } . } \end{array}\tag{62}
$$

Hence,

$$
s _ { F } - s _ { \alpha } ^ { \mathrm { g e o } } = \alpha \left( s _ { F } - s _ { T } \right) .\tag{63}
$$

The geometric mixture therefore preserves the DMD correction direction but only rescales it by the constant factor α, independent of the sample.

Implication for DA-DMD. Both constructions produce intermediate distributions between the teacher and fake models, but they differ critically in score space. The geometric mixture yields uniform attenuation, whereas the arithmetic mixture converts a fixed distribution-level mixture co efficient into a sample-dependent score-space responsibility. Since our goal is to adapt the update strength according to sample-level training signals while preserving the original DMD correction direction, the arithmetic mixture provides the more appropriate structural template.

We emphasize that this comparison motivates theform of sample-adaptive weighting. The practical DA-DMD weight further replaces the unavailable density-ratio-dependent responsibility with the disagreement-based control rule introduced in Sec. 2.2.

## D ADDITIONAL EXPERIMENTS

We provide additional experiments complementing the main-paper evaluation, including controlled analyses of DMD optimization, complete quantitative results on additional benchmarks, and quali tative comparisons across all evaluated backbones.

![](images/a340743d0fe346013ed92e47877f7c0cff4d90ecf6cb9060ff39001d282c7018.jpg)

![](images/09699548ce42a12fe466d0507d62bf025c4b03a97931e3ba69e7aad823c74190.jpg)  
Figure 7: Selective disagreement-aware weighting in DA-DMD. (a) Outside the predefined highnoise region H, DA-DMD exactly preserves the vanilla DMD update. Within H, the retained update strength decreases continuously with normalized teacher–fake disagreement r, with α controlling its maximum value. (b) The resulting weighting leaves low-/mid-noise updates unchanged while substantially attenuating the high-energy updates concentrated in the high-noise regime.

## D.1 DIAGNOSTIC ANALYSIS

We first provide controlled analyses to further understand the relationship between teacher–fake disagreement and DMD optimization. Figs. 8, 9, and 10 show results on DIGITS, CIFAR-10, and CIFAR-100, respectively. Starting from the same initialization, high-disagreement samples consistently exhibit larger raw DMD updates and can disproportionately influence the optimization trajectory, leading to biased target-mode occupancy. DA-DMD attenuates these aggressive corrections and recovers the target distribution more faithfully across all three settings.

Fig. 7 further illustrates the selective weighting mechanism of DA-DMD. Updates outside the predefined high-noise region remain exactly unchanged, whereas disagreement continuously controls the retained update strength within the active region. Together with the large-scale SD3.5-M analysis in Fig. 2, these results consistently support the central design principle of DA-DMD: noise level determines where intervention is useful, while teacher–fake disagreement determines how strongly the update should be moderated.

## D.2 ADDITIONAL QUANTITATIVE RESULTS

PartiPrompts. Tab. 3 reports the complete results on PartiPrompts. DA-DMD achieves higher average pairwise win rates than DMD2 across all five backbones, demonstrating that its improvements extend beyond DrawBench to a substantially larger and more diverse prompt set. The gains are particularly pronounced on SD3.5-M and SD3.5-L, where DA-DMD reaches average win rates of 61.2% and 61.4%, respectively.

GenEval2. Tabs. 4 and 5 evaluate compositional generation from complementary perspectives. DA-DMD improves the overall GenEval2 score across the evaluated backbones, with the largest gain of +6.81 on SD3.5-L. Improvements become especially pronounced on more compositionally demanding prompts: on SD3.5-L, DA-DMD improves the 7- and 8-atom scores by +13.04 and +9.56, respectively. These results suggest that disagreement-aware update control is particularly beneficial when multiple visual constraints must be satisfied simultaneously.

TIIF-Bench. Tabs. 6 and 7 provide a detailed evaluation of fine-grained instruction following. DA-DMD improves the overall TIIF-Bench score across all five evaluated backbones, with the largest gain of +2.44 on SD3.5-L. Category-level performance varies across individual skills, but the aggregate results show broad improvements on both short- and long-form instructions, including advanced compositional settings.

## D.3 ADDITIONAL QUALITATIVE RESULTS

We further compare DA-DMD with DMD2 across all evaluated model families. Figs. 12, 13, 14, 15, and 16 present additional examples on SD3-M, SD3.5-M, SD3.5-L, SANA, and SANA-1.5, respectively. Across diverse prompts involving counting, spatial relations, relative attributes, color binding, text rendering, and structured multi-object composition, DA-DMD more faithfully satisfies the specified visual constraints while maintaining comparable perceptual quality. These examples complement the quantitative results and show that the improvements produced by disagreementaware update control are also reflected in visually observable prompt adherence.

![](images/fe13d9d85173af118edb93f28eddaffe42f09a2f9d6be98d0c9a240b75393e5b.jpg)

![](images/fce7320df0857c04cb71f77b8970f7418086a3886aadb1efa19e27186e2884d8.jpg)

![](images/201b6ffe21e0db3cb772ecc1b7bc31dfba207807313ef0d994dfe7cb597f6609.jpg)

![](images/8971d3051d824d791fbe8fe7f7610fe7434d0b231a0ef7594ac03ada0b515fb0.jpg)

![](images/dca256ddfcd681a78620f62db152d1140083607a9691433b1b5baec50b849632.jpg)  
Figure 8: Controlled analysis of disagreement-aware DMD optimization on CIFAR-10. Starting from the same initialization, raw DMD can be dominated by high-disagreement samples with large update magnitudes, leading to an imbalanced optimization trajectory. DA-DMD selectively attenuates these updates while retaining lower-disagreement corrections, resulting in improved targetmode matching. The bottom panels visualize the evolution of target-mode occupancy and the concentration of large raw DMD updates among high-disagreement samples.

![](images/d90d7028749011790888349a72445dbd881be2023aa8bddc33b1e3f21e300375.jpg)

![](images/4cf317e3baa60eabbc99fa89e64363b1ad59579bf283d89918086c089b630aae.jpg)

![](images/991ec50af50cc7219651c6c9585d401e0baa5578e289f4f96b25eb26b8424bde.jpg)

![](images/9f75a8e642778c4bb1db41b17b255de25efe64e18567c2c8af3267ac06d0fd78.jpg)

![](images/a892fbc1bde85be46f0013a1111f4a218f46fa101ec036f2c1f5d7fe026db610.jpg)  
Figure 9: Controlled analysis of disagreement-aware DMD optimization on DIGITS. Starting from the same initialization, raw DMD can be dominated by high-disagreement samples with large update magnitudes, leading to an imbalanced optimization trajectory. DA-DMD selectively attenuates these updates while retaining lower-disagreement corrections, resulting in improved target mode matching. The bottom panels visualize the evolution of target-mode occupancy and the concentration of large raw DMD updates among high-disagreement samples.

![](images/1f4d95f7e9ed7cfd7a7ab4597bb083b9bf4cb742670ee63e38dc711087e07ee6.jpg)

![](images/2c8b9e5996d2b979e830e47608c8d0604b9558c44f2ba858b844bb7a59ec2ec8.jpg)

![](images/8b7bf87d7632c558ede2bf9d835307bcd9e1c99d16bfd20e5706a175e2e830e9.jpg)

![](images/782ac3194942e3ed06eef689d22e2dc27967139cb8eefae9df78266b2aa361b3.jpg)

![](images/8d1b1c7b0a5b3e3a99e2f3799163b5ccafcff0b68e39c7a9f655fffd71c55707.jpg)  
Figure 10: Controlled analysis of disagreement-aware DMD optimization on CIFAR-100. Starting from the same initialization, raw DMD can be dominated by high-disagreement samples with large update magnitudes, leading to an imbalanced optimization trajectory. DA-DMD selectively attenuates these updates while retaining lower-disagreement corrections, resulting in improved targetmode matching. The bottom panels visualize the evolution of target-mode occupancy and the con centration of large raw DMD updates among high-disagreement samples.

Table 3: Quantitative comparison on PartiPrompts. DA-DMD yields broad improvements over the corresponding DMD2 baselines across different model families.
<table><tr><td>Backbone</td><td>Method</td><td>PickScore↑</td><td>ImageReward↑</td><td>HPSv2↑</td><td>AES↑</td><td>VQ-R1↑</td><td>Avg. WR↑</td></tr><tr><td rowspan="2">SANA</td><td>DMD2</td><td>22.74</td><td>117.28</td><td>30.57</td><td>5.940</td><td>4.617</td><td></td></tr><tr><td>DA-DMD (Ours)</td><td>22.77 +0.03</td><td>119.58 +2.30</td><td>30.57 +0.00</td><td>5.956 +0.016</td><td>4.638 +0.020</td><td>53.6%</td></tr><tr><td rowspan="2">SANA-1.5</td><td>DMD2</td><td>22.85</td><td>122.79</td><td>30.31</td><td>6.022</td><td>4.553</td><td>=</td></tr><tr><td>DA-DMD (Ours)</td><td>22.90 +0.05</td><td>122.29 -0.50</td><td>30.44 +0.13</td><td>6.050 +0.029</td><td>4.575 +0.023</td><td>54.6%</td></tr><tr><td rowspan="6">SD3-M</td><td>DMD2</td><td>22.39</td><td>116.74</td><td>30.44</td><td>5.697</td><td>4.571</td><td></td></tr><tr><td>Hyper-SD3</td><td>22.24</td><td>92.07</td><td>27.87</td><td>5.530</td><td>4.418</td><td>30.0%</td></tr><tr><td>SiD-DiT</td><td>22.56</td><td>109.77</td><td>28.98</td><td>5.499</td><td>4.617</td><td>38.8%</td></tr><tr><td>SiD-Adv</td><td>22.53</td><td>104.79</td><td>29.86</td><td>5.716</td><td>4.583</td><td>44.4%</td></tr><tr><td>TDM</td><td>22.54</td><td>117.95</td><td>30.38</td><td>5.647</td><td>4.648</td><td>50.2%</td></tr><tr><td>DA-DMD (Ours)</td><td>22.57 +0.17</td><td>113.96 -2.78</td><td>30.84</td><td>5.726</td><td>4.678</td><td>57.2%</td></tr><tr><td rowspan="6">SD3.5-M</td><td>DMD2</td><td>22.69</td><td>119.55</td><td>+0.40 29.56</td><td>+0.030 5.543</td><td>+0.107 4.611</td><td></td></tr><tr><td>SiD-DiT</td><td>22.71</td><td>114.97</td><td>29.87</td><td>5.661</td><td>4.661</td><td>49.0%</td></tr><tr><td>SiD-Adv</td><td>22.67</td><td>114.72</td><td>29.45</td><td>5.816</td><td>4.553</td><td>48.4%</td></tr><tr><td>DP-DMD</td><td>22.48</td><td>114.12</td><td>28.83</td><td>5.486</td><td>4.560</td><td>36.6%</td></tr><tr><td>SenseFlow</td><td>22.50</td><td>104.05</td><td>30.18</td><td>5.714</td><td>4.674</td><td>56.2%</td></tr><tr><td>DA-DMD (Ours)</td><td>22.75 +0.06</td><td>123.73</td><td>30.33</td><td>5.659</td><td>4.692</td><td>61.2%</td></tr><tr><td rowspan="7">SD3.5-L</td><td>DMD2</td><td>22.69</td><td>+4.18 116.32</td><td>+0.77 29.14</td><td>+0.116 5.595</td><td>+0.081</td><td></td></tr><tr><td>SiD-DiT</td><td>22.81</td><td>112.08</td><td>29.95</td><td>5.683</td><td>4.591 4.621</td><td>54.0%</td></tr><tr><td>SiD-Adv</td><td>22.98</td><td>118.46</td><td>30.01</td><td>5.849</td><td>4.628</td><td>61.2%</td></tr><tr><td>SD3.5-L Turbo</td><td>22.85</td><td>110.84</td><td>29.14</td><td>5.755</td><td>4.616</td><td>52.4%</td></tr><tr><td>SenseFlow</td><td>22.89</td><td>119.52</td><td>30.48</td><td>5.639</td><td>4.671</td><td>60.8%</td></tr><tr><td></td><td>23.10</td><td>121.78</td><td>30.53</td><td>5.694</td><td>4.588</td><td></td></tr><tr><td>DA-DMD (Ours)</td><td>+0.41</td><td>+5.46</td><td>+1.39</td><td>+0.099</td><td>-0.003</td><td>61.4%</td></tr></table>

Table 4: GenEval2 results across backbones. ∆ denotes the absolute improvement over DMD2. WR is the pairwise win rate of DA-DMD against DMD2.
<table><tr><td rowspan="2">Backbone</td><td colspan="3">Overall ↑</td><td rowspan="2">WR (%) ↑</td></tr><tr><td>DMD2</td><td>DA-DMD</td><td>Δ</td></tr><tr><td>SANA</td><td>14.68</td><td>15.89</td><td>+1.21</td><td>55.0</td></tr><tr><td>SD3.5-M</td><td>20.17</td><td>21.74</td><td>+1.57</td><td>55.4</td></tr><tr><td>SD3.5-L</td><td>15.93</td><td>22.74</td><td>+6.81</td><td>61.9</td></tr></table>

Table 5: GenEval2 results across prompt complexity. DA-DMD provides broad gains across compositional complexity levels, with particularly strong improvements on challenging high-atom prompts, e.g., +13.04 and +9.56 on 7- and 8-atom prompts with SD3.5-L. Colored values denote absolute changes over DMD2.
<table><tr><td>Backbone</td><td>Method</td><td>3↑</td><td>4↑</td><td>5↑</td><td>6个</td><td>7个</td><td>8↑</td><td>9↑</td><td>10↑</td></tr><tr><td rowspan="2">SANA</td><td>DMD2</td><td>42.03</td><td>32.25</td><td>14.39</td><td>11.27</td><td>6.48</td><td>5.53</td><td>3.22</td><td>2.23</td></tr><tr><td>DA-DMD</td><td>44.68 +2.65</td><td>32.11 -0.14</td><td>13.89 -0.50</td><td>13.08 +1.81</td><td>10.41 +3.93</td><td>7.08 +1.55</td><td>4.22 +1.00</td><td>3.37 +1.14</td></tr><tr><td rowspan="2">SANA-1.5</td><td>DMD2</td><td>53.69</td><td>43.07</td><td>20.28</td><td>18.02</td><td>6.95</td><td>9.27</td><td>5.63</td><td>3.43</td></tr><tr><td>DA-DMD</td><td>54.66 +0.97</td><td>44.20 +1.13</td><td>22.62 +2.34</td><td>17.97 -0.05</td><td>6.59 -0.36</td><td>8.84 -0.43</td><td>6.73 +1.10</td><td>3.98 +0.55</td></tr><tr><td rowspan="2">SD3.5-M</td><td>DMD2</td><td>50.01</td><td>39.16</td><td>26.65</td><td>18.27</td><td>8.44</td><td>7.75</td><td>7.90</td><td>3.14</td></tr><tr><td>DA-DMD</td><td>53.47 +3.46</td><td>36.63 -2.53</td><td>27.68 +1.03</td><td>19.05 +0.78</td><td>15.15 +6.71</td><td>9.02 +1.27</td><td>8.58 +0.68</td><td>4.32 +1.18</td></tr><tr><td rowspan="2">SD3.5-L</td><td>DMD2</td><td>39.87</td><td>31.37</td><td>20.27</td><td>14.98</td><td>6.92</td><td>5.83</td><td>5.16</td><td>3.02</td></tr><tr><td>DA-DMD</td><td>45.83 +5.96</td><td>42.11 +10.74</td><td>26.81 +6.54</td><td>20.10 +5.12</td><td>19.96 +13.04</td><td>15.39 +9.56</td><td>7.81 +2.65</td><td>3.89 +0.87</td></tr></table>

Table 6: TIIF-Bench overall and basic-category results. DA-DMD improves the overall instruction-following score across all five evaluated backbones, with the largest gain of +2.44 on SD3.5-L, while showing broadly positive improvements across short- and long-form categories.
<table><tr><td>Backbone</td><td>Method</td><td>Overall↑</td><td>Overall S↑</td><td>Overall L↑</td><td>Basic S↑</td><td>Basic L↑</td><td>Attr. S↑</td><td>Attr. L↑</td><td>Rela. S↑</td><td>Rela. L↑</td><td>Reason S↑</td><td>Reason L↑</td></tr><tr><td rowspan="2">SANA</td><td>DMD2</td><td>60.72</td><td>60.1</td><td>61.3</td><td>75.8</td><td>77.3</td><td>79.3</td><td>79.6</td><td>84.1</td><td>84.6</td><td>63.4</td><td>67.6</td></tr><tr><td>DA-DMD</td><td>61.12 +0.40</td><td>60.4 +0.3</td><td>61.8 +0.5</td><td>75.7 -0.1</td><td>77.1 -0.2</td><td>79.1 -0.2</td><td>79.8 +0.2</td><td>84.2 +0.1</td><td>84.3 -0.3</td><td>63.4 +0.0</td><td>66.9 -0.7</td></tr><tr><td rowspan="2">SANA-1.5</td><td>DMD2</td><td>62.47</td><td>63.2</td><td>61.8</td><td>77.1</td><td>76.4</td><td>81.2</td><td>79.2</td><td>84.5</td><td>83.9</td><td>65.1</td><td>65.7</td></tr><tr><td>DA-DMD</td><td>62.84 +0.37</td><td>63.5 +0.3</td><td>62.2 +0.4</td><td>77.5 +0.4</td><td>76.6 +0.2</td><td>80.8 -0.4</td><td>79.1 -0.1</td><td>84.8 +0.3</td><td>84.7 +0.8</td><td>66.3 +1.2</td><td>66.1 +0.4</td></tr><tr><td>SD3-M</td><td>DMD2</td><td>63.33</td><td>63.8</td><td>62.8</td><td>77.8</td><td>77.8</td><td>83.5</td><td>82.5</td><td>82.3</td><td>83.1</td><td>66.4</td><td>66.9</td></tr><tr><td rowspan="2">SD3.5-M</td><td>DA-DMD</td><td>64.23 +0.90</td><td>64.5 +0.7</td><td>63.9 +1.1</td><td>78.0 +0.2</td><td>78.4 +0.6</td><td>83.9 +0.4</td><td>85.1 +2.6</td><td>82.0 -0.3</td><td>81.9 -1.2</td><td>66.9</td><td>66.7</td></tr><tr><td>DMD2</td><td>66.77</td><td>67.9</td><td>65.6</td><td>79.5</td><td>79.3</td><td>85.9</td><td>84.0</td><td>84.4</td><td>85.5</td><td>+0.5 66.8</td><td>-0.2 67.6</td></tr><tr><td></td><td>DA-DMD</td><td>66.98 +0.21</td><td>67.5 -0.4</td><td>66.5 +0.9</td><td>78.7 -0.8</td><td>79.8 +0.5</td><td>84.3 -1.6</td><td>85.2 +1.2</td><td>85.2 +0.8</td><td>85.4 -0.1</td><td>65.7 -1.1</td><td>68.0 +0.4</td></tr><tr><td rowspan="2">SD3.5-L</td><td>DMD2</td><td>65.82</td><td>68.4</td><td>63.3</td><td>77.8</td><td>75.6</td><td>84.6</td><td>80.2</td><td>83.3</td><td>81.4</td><td>64.2</td><td>64.3</td></tr><tr><td>DA-DMD</td><td>68.26 +2.44</td><td>70.4 +2.0</td><td>66.1 +2.8</td><td>79.7 +1.9</td><td>77.8 +2.2</td><td>86.5 +1.9</td><td>82.2 +2.0</td><td>84.4 +1.1</td><td>83.1 +1.7</td><td>66.7 +2.5</td><td>67.1 +2.8</td></tr></table>

Table 7: TIIF-Bench advanced compositional results. DA-DMD generally improves complex compositional instruction following across model backbones, with particularly strong gains on SD3.5-L, including +2.4 and +3.3 on advanced short- and long-form prompts. Colored values denote absolute changes over DMD2.
<table><tr><td>Backbone</td><td>Method</td><td>Adv. S↑</td><td>Adv. L↑</td><td>Attr.+Rela. S↑</td><td>Attr.+Rela. L↑</td><td>Attr.+Rea. S↑</td><td>Attr.+Rea. L↑</td><td>Rela.+Rea. S↑</td><td>Rela.+Rea. L↑</td></tr><tr><td rowspan="2">SANA</td><td>DMD2</td><td>51.8</td><td>52.8</td><td>78.0</td><td>76.3</td><td>63.3</td><td>66.9</td><td>66.0</td><td>68.7</td></tr><tr><td>DA-DMD</td><td>52.3 +0.5</td><td>53.8 +1.0</td><td>78.2 +0.2</td><td>78.2 +1.9</td><td>63.5 +0.2</td><td>66.9 +0.0</td><td>67.0 +1.0</td><td>71.0 +2.3</td></tr><tr><td rowspan="2">SANA-1.5</td><td>DMD2</td><td>56.1</td><td>54.0</td><td>80.4</td><td>76.7</td><td>68.7</td><td>65.9</td><td>68.8</td><td>69.4</td></tr><tr><td>DA-DMD</td><td>56.3 +0.2</td><td>54.7 +0.7</td><td>80.6 +0.2</td><td>76.8 +0.1</td><td>69.8 +1.1</td><td>67.2 +1.3</td><td>68.4 -0.4</td><td>68.6 -0.8</td></tr><tr><td rowspan="2">SD3-M</td><td>DMD2</td><td>57.1</td><td>55.4</td><td>78.0</td><td>78.7</td><td>63.9</td><td>65.7</td><td>63.2</td><td>66.9</td></tr><tr><td>DA-DMD</td><td>58.2 +1.1</td><td>57.0 +1.6</td><td>78.8 +0.8</td><td>79.2 +0.5</td><td>65.5 +1.6</td><td>66.9 +1.2</td><td>65.2 +2.0</td><td>68.6 +1.7</td></tr><tr><td rowspan="2">SD3.5-M</td><td>DMD2</td><td>62.5</td><td>58.9</td><td>80.9</td><td>78.7</td><td>68.2</td><td>67.4</td><td>66.4</td><td>67.6</td></tr><tr><td>DA-DMD</td><td>61.9 -0.6</td><td>59.7 +0.8</td><td>79.9 -1.0</td><td>78.8</td><td>67.0</td><td>67.1</td><td>66.8</td><td>68.8</td></tr><tr><td rowspan="2">SD3.5-L</td><td>DMD2</td><td>64.0</td><td>57.0</td><td>80.1</td><td>+0.1 75.6</td><td>-1.2 65.1</td><td>-0.3</td><td>+0.4</td><td>+1.2</td></tr><tr><td>DA-DMD</td><td>66.4 +2.4</td><td>60.3 +3.3</td><td>80.8 +0.7</td><td>78.5 +2.9</td><td>68.0 +2.9</td><td>64.4 65.8</td><td>63.5 66.6</td><td>62.8 65.4</td></tr></table>

![](images/a5a3ec74f897218aa4ff3df4f3c0a404f8da8c093a330ed3871a4f5c333e8111.jpg)  
Figure 11: Qualitative comparison with representative distillation baselines on SD3.5-M. DA-DMD demonstrates stronger prompt adherence and visual consistency across challenging cases involving counting, text rendering, fine-grained attributes, and object identity, compared with DMD, SiD-Adv, SiD-DiT, SenseFlow, and DP-DMD.

A giraffe.

![](images/01628bae782e006ab9fa72d95d2e6a534037f0a6ea78f8b7ff87e076f8fbffaf.jpg)  
A green pepper to the left of a red pepper.  
Four owls standing on a telephone wire.  
A stone bagel on top of a backpack.  
A vibrant blue rose blooms beside a flourishing bush of lush, red roses

Figure 12: Qualitative comparison on SD3-M. DA-DMD improves prompt adherence over DMD2 across diverse compositional challenges, including object counting, spatial placement, relative relations, color control, and structured multi-object generation.  
![](images/59144f247f9c5f47e4edcaae10faabb03b2800c5ba01fe83f8f854e1b0fd5e92.jpg)  
Figure 13: Qualitative comparison on SD3.5-M. DA-DMD improves prompt adherence over DMD2 across diverse compositional challenges, including object counting, spatial placement, relative relations, color control, and structured multi-object generation.

![](images/43ea36e42b072a438ed3a70823bbc88ecc952c4096303a0ae118c3351df5e6e0.jpg)  
Figure 14: Qualitative comparison on SD3.5-L. DA-DMD improves prompt adherence over DMD2 across diverse compositional challenges, including object counting, spatial placement, relative relations, color control, and structured multi-object generation.

![](images/03a25380f33d88050d7495ed33c9be91a826d6a68b405fcc24eea13313b253fb.jpg)  
Figure 15: Qualitative comparison on SANA. DA-DMD improves prompt adherence over DMD2 across diverse compositional challenges, including object counting, spatial placement, relative relations, color control, and structured multi-object generation.

![](images/1e831d21e1936d660d5510fed9ead45158d8fc1dfbb7751c9839671ccfbd3dd9.jpg)  
A pair of headphones on a statue of a horse.

![](images/219e6fe19d96bcb422b6098c3b5210d10fe0ce80df08a49f5003f45e8c1bbffa.jpg)  
A green apple and a black backpack.

![](images/ceef00b83c2cc99b932f16a6a910aa4febcd5c858e65e10831a37888f0e1d6e3.jpg)  
A cat on the right of a tennis racket.

![](images/9be2491a324a30f307099536dac6ad926f7e2229c3df07573066b77c60dd7868.jpg)  
A pony with a shooting star on its flank.

![](images/b2663a24ae094713168eabd015712690007bfaa762c2c4d13946f80e6c4cd62a.jpg)  
A sidewalk next to a wooden post with a blue '5' painted on top

Figure 16: Qualitative comparison on SANA-1.5. DA-DMD improves prompt adherence over DMD2 across diverse compositional challenges, including object counting, spatial placement, relative relations, color control, and structured multi-object generation.