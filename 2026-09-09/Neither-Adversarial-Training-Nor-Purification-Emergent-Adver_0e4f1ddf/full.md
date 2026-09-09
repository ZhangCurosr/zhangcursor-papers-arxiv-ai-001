# Neither Adversarial Training Nor Purification: Emergent Adversarial Robustness from Oscillatory Predictive Learning

Mohammed-Yassine Habibi<sup>1,2∗</sup> <sup>†</sup>, Klea Ziu<sup>3∗</sup>, Martin Takácˇ<sup>3</sup>, Makoto Yamada<sup>4</sup> <sup>1</sup> Ecole polytechnique, IPP, France

<sup>2</sup> Ecole Nationale des Ponts et Chaussées, IPP, France <sup>3</sup> Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE <sup>4</sup> Okinawa Institute of Science and Technology, Okinawa, Japan

## Abstract

Adversarial robustness in computer vision is still largely achieved through adversarial training or test-time adversarial purification, both of which introduce significant computational overhead by generating adversarial examples during training or performing iterative denoising at test time. We study whether empirical robustness can instead emerge from architectural and representation-learning inductive biases. We introduce Oscillatory Predictive Learning (OPL), a two-stage framework that combines Artificial Kuramoto Oscillatory Neurons (AKOrN) with predictive self-supervised pretraining using X-PhiNet. Because our default checkpoint uses randomized initial oscillator states, we compare it with other randomized adversarial defense methods that provide precise, reproducible, and strong attack protocols. Experiments on CIFAR-10 and CIFAR-100, with additional corruption evaluation on CIFAR-10-C, demonstrate that our method achieves competitive results under the AutoAttack-rand evaluation protocol. On CIFAR-10 and CIFAR-100, OPL attains 76.63±0.76% and 50.44% robust accuracy, respectively, under $\ell _ { \infty } , \epsilon = 8 / 2 5 5$ , AutoAttack-rand with EoT K = 20. In addition, we conduct several diagnostics for attack failure and gradient masking via black-box attacks, transfer attacks, and attack-strength sweeps. Ablation studies show that oscillatory dynamics provide a key robustness-inducing inductive bias, which is significantly amplified by predictive pretraining. We further find that robustness is concentrated in a narrow dynamical regime, with changes in oscillator dimension sharply reducing robust accuracy despite improving clean accuracy. Our results suggest that architectural and self-supervised predictive biases can offer an efficient and complementary pathway to empirical adversarial robustness, challenging the view that high robustness necessarily requires adversarial-example generation during training or iterative purification at test time.

## 1 Introduction

Deep vision models remain vulnerable to small adversarial perturbations [24, 5], and the strongest widely adopted robustness results are still dominated by adversarial training [1, 5, 19] and adversarial purification [23, 11]. Indeed, there are no strong baselines for vision models that rely on neither adversarial training nor adversarial purifying [10, 11]. Although effective, those methods substantially increase training costs because they require generating adversarial examples throughout optimization or performing iterative denoising at test time, often at scale. This computational burden is not merely an engineering inconvenience: recent scaling analyses argue that, under current adversarial-training paradigms, simply increasing compute is unlikely to close the gap to human-level robustness, and that meaningful progress will require more efficient training algorithms and improved architectures rather than scale alone [5]. This motivates the question explored: can adversarial robustness emerge from architectural and representation-learning inductive biases rather than from repeated exposure to adversarial examples?

In this paper, we test a concrete hypothesis: a predictive self-supervised objective combined with oscillatory encoder dynamics can yield strong robustness under standardized adversarial evaluation, without adversarial training. We study this hypothesis through Oscillatory Predictive Learning, a framework that combines an Artificial Kuramoto Oscillatory Neurons (AKOrN) encoder [22] with predictive self-supervised learning via X-PhiNet [18]. Oscillatory coupling imposes structured consensus dynamics that can reduce sensitivity to local perturbations, while predictive pretraining encourages stable and globally consistent representations in this dynamical setting. The biological analogy motivates this design, but the contribution we evaluate is algorithmic: the robustness arises from this interaction, and the combination of AKOrN and X-PhiNet substantially outperforms either component alone under standardized attacks.

To assess whether the robustness gain is specific to X-PhiNet or reflects a broader benefit of predictive non-contrastive pretraining, we compare several SSL objectives on the same AKOrN backbone in Appendix D. In these preliminary experiments, X-PhiNet gives the strongest robust accuracy among the tested methods, while SimSiam [7] achieves a similar but slightly lower result, and BYOL [15] is less stable. We therefore use X-PhiNet as the main pretraining objective in the rest of the paper. We do not claim that the effect automatically transfers to all predictive SSL frameworks; evaluating JEPA-style [14, 3] objectives and larger predictive architectures remains future work.

Empirically, this is what we observe. On CIFAR-10, our method, Oscillatory Predictive Learning, achieves $7 6 . 6 3 \pm 0 . 7 6 \%$ robust accuracy under $\ell _ { \infty } , \epsilon = 8 / 2 5 5$ , AutoAttack-rand with EoT $K = 2 0$ [13, 11], outperforming prior stochastic defenses evaluated under the same AutoAttack-rand protocol that has been recommended by [11, 21] (DiffPure: 71.29%, [23]) . In the same evaluation, AKOrN alone reaches 64.98% AutoAttack-rand robustness, while X-PhiNet alone and a standard ResNet baseline remain almost entirely non-robust. This pattern supports a complementary view of the two components: oscillatory encoder dynamics provide a robustness-supporting inductive bias, and predictive pretraining substantially amplifies it.

A central outcome of this discovery is computational. Because our method does not rely on adversarial training, it avoids the dominant source of cost in current robust-learning pipelines. As a result, it offers a substantially lower-compute route to robustness and a more promising starting point for scaling than methods whose training procedure already embeds adversarial-example generation at every iteration. In our current experiments, the robustness is achieved at the cost of lower clean accuracy compared to high-capacity adversarially trained models. Beyond aggregate performance, we find a sharp sensitivity to oscillator dimension. In our sweep, robustness is concentrated in the $N = 2$ family, while increasing the oscillator dimension to $N = 4$ substantially reduces robust accuracy despite improving clean accuracy. Increasing encoder flexibility or optimizing for clean accuracy does not monotonically improve adversarial robustness; in several settings, it actively degrades it. This suggests that robustness in oscillatory models is not simply a byproduct of larger capacity or better standard representations, but depends on maintaining specific dynamical constraints.

Together, these findings point to a complementary route to adversarial robustness: instead of explicitly fitting adversarial perturbations during training, one can induce robustness through the interaction between structured encoder dynamics and predictive learning. We therefore propose Oscillatory Predictive Learning as a framework for studying robustness from inductive bias, and we evaluate it through standardized attacks, controlled ablations, and dynamical hyperparameter sweeps that expose a reproducible robust regime.

Contributions In summary, our work provides a highly efficient alternative to adversarial optimization through the following principal contributions:

• We propose Oscillatory Predictive Learning, a framework combining AKOrN encoders with X-PhiNet predictive pretraining, achieving 76.63% and 50.44% AutoAttack-rand robustness on CIFAR-10 and CIFAR-100, respectively, entirely without adversarial training or adversarial purification.

• We report OPL in a substantially lower measured training-FLOP regime than representative adversarially trained CIFAR models. Because the compared systems differ in architecture, data, training procedure, and evaluation protocol, we treat these results as contextual rather than as a protocol-matched efficiency comparison.

• We provide controlled evidence that oscillatory dynamics and predictive pretraining play complementary roles: X-PhiNet provides a substantial +11.65 percentage point robustness amplification when applied to AKOrN (from 64.98% to 76.63%), whereas applying X-PhiNet to a standard ResNet yields near-zero robustness.

• We identify a sharp sensitivity to oscillator dimensionality driven by dynamical hyperparameters, demonstrating that robustness collapses near-discontinuously when increasing oscillator dimensionality (e.g., from N = 2 to N = 4), revealing that clean and robust accuracy are not monotonically aligned.

## 2 Background and Motivation

Standardized evaluation of adversarial robustness. Empirical robustness claims depend strongly on the attack used for evaluation. Deterministic CIFAR classifiers are commonly evaluated with standard AutoAttack and reported on RobustBench. Since OPL uses randomized initial oscillator states, we instead evaluate it as a stochastic classifier using AutoAttack-rand with EoT. We therefore report deterministic RobustBench-style results only as contextual references, not as protocol-matched comparisons.[10, 13]. We report additional $l _ { \infty }$ and $\ell _ { 2 }$ robust accuracy and corruption/noise evaluations as complementary stress.

Adversarial training at scale. One of the two strongest CIFAR robustness methods is adversarial training, often combined with large synthetic datasets and high-capacity backbones [5, 1]. This line of work is highly effective, but its cost profile is increasingly extreme. As emphasized in [5], adversarially trained CIFAR-10 models now operate in a regime ranging from tens of millions of synthetic examples to hundreds of millions, culminating in a 300M-unique-sample / 500M-sample training pipeline and over $1 0 ^ { 2 1 }$ training FLOPs [5, 1].

Dynamical models for robustness. Prior work has linked robustness to dynamical stability. Stable neural ODEs impose Lyapunov-stable equilibrium behavior so that small perturbations are driven toward similar trajectories [19]. AKOrN extends this dynamical viewpoint by replacing static threshold units with coupled Kuramoto-style oscillator updates [22]. In this paper, we treat oscillatory dynamics as a robustness-supporting inductive bias.

Stochastic and purification-based defenses. OPL belongs to a broader class of stochastic defenses that leverage randomness at inference time to improve adversarial robustness. This includes both randomized smoothing methods [9], which inject noise in input space to obtain certified guarantees, and adversarial purification approaches, such as diffusion-based defenses, which iteratively denoise perturbed inputs through stochastic generative processes before classification. These methods typically require adaptive evaluation via Expectation over Transformation (EoT) [4] to account for inference-time randomness [11]. In contrast, OPL does not rely on explicit input perturbation or generative purification; instead, stochasticity arises intrinsically from the random initialization of the internal oscillators. This yields strong empirical robustness without adversarial training, noise augmentation, or iterative test-time preprocessing.

Self-supervised approaches to robustness. Prior work have demonstrated that self-supervised learning can improve model robustness and uncertainty estimation [16], particularly as a pre-training step for adversarial fine-tuning [6]. Predictive learning serves as a general non-contrastive SSL framework, prominently featured in methods like SimSiam [7] and BYOL [15], which learn stable representations by predicting augmented views. While these methods rely on general architectural constraints to prevent collapse, X-PhiNet [18] makes an explicit link to biological temporal prediction. We build upon this foundation, showing that predictive SSL without adversarial fine-tuning yields state-of-the-art robustness when paired with the right dynamical architecture.

Recurrent and iterative-inference defenses. Iterative inference mechanisms, such as deep equilibrium networks and recurrent reasoning steps, naturally resist adversarial perturbations by driving activations toward stable attractors [8]. OPL extends this philosophy by operating as a continuous-time dynamical system via Kuramoto coupling [22].

Our work sits at the intersection of these directions. Unlike prior top-performing defenses, we do not use adversarial examples during training or adversarial purification during test-time inference. Unlike prior dynamical or self-supervised methods considered in isolation, we study robustness as an interaction between architecture and objective: the encoder provides the oscillatory inductive bias while the specific SSL framework provides the predictive learning signal, and their combination yields strong standardized robustness in a narrow dynamical regime.

## 3 Preliminaries: Oscillatory Encoders and Predictive Pretraining

AKOrN as a coupled oscillatory encoder. Let $\boldsymbol { x } ~ \in ~ \mathbb { R } ^ { 3 \times H _ { 0 } \times W _ { 0 } }$ denote an input image. A convolutional stem maps x to a feature map $X \in \mathbb { R } ^ { C \times H \times W }$ with $C = K N$ channels. We reshape this tensor as $X \in \mathbb { R } ^ { K \times N \times H \times W }$ so that each spatial location $( h , w )$ contains K oscillator states $\mathbf { x } _ { k , h , w } \in \mathbb { S } ^ { N - 1 }$ , where

$$
\mathbb { S } ^ { N - 1 } : = \{ \mathbf { u } \in \mathbb { R } ^ { N } : \| \mathbf { u } \| _ { 2 } = 1 \} .
$$

Here, N is the oscillator dimension (the number of rotating dimensions), and $K = C / N$ is the number of oscillators per spatial location.

A Kuramoto layer updates each oscillator through a coupled dynamical system:

$$
\dot { \mathbf { x } } _ { k , h , w } = \Omega _ { k , h , w } \mathbf { x } _ { k , h , w } + \mathrm { P r o j } _ { \mathbf { x } _ { k , h , w } } \left( \mathbf { y } _ { k , h , w } + \sum _ { k ^ { \prime } , \Delta h , \Delta w } \mathbf { J } _ { k , k ^ { \prime } , \Delta h , \Delta w } \mathbf { x } _ { k ^ { \prime } , h + \Delta h , w + \Delta w } \right) ,\tag{1}
$$

where $\Omega _ { k , h , w } \in \mathbb { R } ^ { N \times N }$ is a learned antisymmetric matrix controlling intrinsic rotation, $\mathbf { J } _ { k , k ^ { \prime } , \Delta h , \Delta u }$ are learned coupling weights, and ${ \bf y } _ { k , h , w }$ is an input-dependent drive term. Because we use an implementation of the Kuramoto dynamical system with convolutions in our experiments, $\mathbf { J } _ { k , k ^ { \prime } , \Delta h , \Delta } .$ w will be the learned connectivity weights of a convolution kernel and $k , k ^ { \prime }$ stand for output and input channels indexes. The projection operator

$$
\operatorname { P r o j } _ { \mathbf { x } } ( \mathbf { y } ) = \mathbf { y } - \langle \mathbf { y } , \mathbf { x } \rangle \mathbf { x }
$$

ensures the velocity vector of an oscillator x effectively lives in the tangent plane of $\mathbf { S } ^ { N - 1 }$ at x.

The coupling term is the key inductive bias of AKOrN. Each oscillator is updated using both its own state and the states of neighboring oscillators, which encourages locally consistent configurations in feature space. Intuitively, if a perturbation affects only a subset of local states, the coupled dynamics need not amplify those deviations independently; instead, neighboring oscillators can pull the representation back toward a consistent configuration.

In practice, Eq. (1) is implemented by unrolling the dynamics for T discrete steps within each of L Oscillatory Neurons Units. Full derivations and implementation details are deferred to Appendix A.

Predictive self-supervised pretraining. X-PhiNet provides the pretraining objective used to shape the encoder before supervised fine-tuning. At a high level, X-PhiNet is a non-contrastive learning protocol that learns by predicting one transformed view from another, rather than by contrasting each sample against large sets of negatives [18]. Its neuroscience motivation comes from the temporal prediction hypothesis: a CA3-like predictor introduces a short delay between signals, and a CA1-like predictor learns to compensate that delay. In self-supervised learning terms, this can be viewed as encouraging representations that remain predictive across nearby observations of the same underlying scene while preserving semantically relevant structure.

This predictive objective is also a natural candidate for robustness-oriented representation shaping. Predictive self-supervision encourages invariances across transformed views of the same input and can promote stable latent structure, which may reduce sensitivity to small perturbations, although it does not guarantee adversarial robustness by itself. In our setting, predictive pretraining therefore acts as a complementary inductive bias to oscillatory dynamics. AKOrN constrains how features evolve through the encoder, while X-PhiNet constrains which representations are learned. Our central question is whether robustness emerges specifically from this interaction, rather than from either ingredient alone.

As shown in Figure 1 below, X-PhiNet, a single input image x is augmented in two different ways, resulting in three inputs $x , x ^ { ( 1 ) } , x ^ { ( 2 ) }$ that are processed by three parallel streams. The first two streams each contain an encoder f while the last one contains a different encoder $f _ { l o n g }$

The outputs of these streams are compared at different depths by computing their similarities. The model is trained by jointly minimizing two losses: the first one, Sim-1, is a symmetric negative cosine loss function for the hippocampus model, and the second one, Sim-2, is a Mean Squared Error loss, which is a slow-learning loss for the neocortex model. Those two losses are combined to update the encoder f by backpropagation. The third stream is designed to model the neocortex by $f _ { l o n g } ,$ , which corresponds to a stable encoder that further improves slow learning with an ability to maintain longterm signals. In our experiments, we extract the $f _ { l o n g }$ once the pretraining is complete.

![](images/84558e9749113fb0e9e5fd20ebd394907e68080f052daff325d88b77ab7703fa.jpg)  
Figure 1: X-PhiNet Temporal Prediction Framework.

We give the exact loss used in our implementation in Appendix A and summarize the overall training pipeline in Section 5.

## 4 Oscillatory Predictive Learning

Overview. The Oscillatory Predictive Learning framework explored here combines an AKOrN encoder with X-PhiNet pretraining in a two-stage pipeline.

In the first stage, we pretrain the AKOrN encoder using the X-PhiNet objective to learn viewconsistent oscillatory representations. In the second stage, we replace the pretraining heads with a classification head and fine-tune the model for supervised image classification. This design lets us test the central hypothesis of the paper directly: whether predictive self-supervision amplifies the robustness bias induced by oscillatory encoder dynamics.

Architecture. The encoder consists of a convolutional stem followed by L Oscillatory Neurons Units blocks, each unrolled for T discrete dynamics steps. Let C denote the total channel width and N the oscillator dimension; then each spatial location contains $K = C / N$ oscillators. After the final Oscillatory Neurons Unit, global pooling produces a representation that is fed either to the X-PhiNet pretraining heads or to a supervised classification head. The initial oscillatory state $X ^ { i n }$ for the first layer of the model can be chosen randomly or deterministically. We evaluate both stochastic and deterministic variants of OPL. For stochastic models, we use AutoAttack-rand with Expectation over Transformation (EoT) to correctly estimate gradients under randomness. For deterministic models, the forward pass is fixed, and EoT has no effect. [4, 13].

![](images/3b3548d5edacf562f81d602fa49fa61f285be834433660c4f6c504f874f8ebf8.jpg)  
Figure 2: Stream of a single Oscillatory Neurons Unit used in the encoder architecture. The inputs are an initial oscillatory state $\breve { X } ^ { i n } \in \mathbb { R } ^ { K \times N \times H \times W }$ , and a conditional stimuli (in practice the input image) $Y ^ { i n } \in \mathbb { R } ^ { K \times N \times H \times \check { W } }$ . The outputs are the final oscillatory state $X ^ { o u t }$ and the extracted features $Y ^ { o u t }$

Training pipeline. During pretraining, we optimize the X-PhiNet objective using the AKOrN encoder and the corresponding pretraining heads required by the predictive learning setup. After pretraining, these heads are discarded and replaced with a classification head. We then fine-tune the resulting model on labeled training data using standard cross-entropy loss. Unless otherwise stated, the encoder architecture is unchanged between the two stages.

Notation. We denote a model configuration by $( C , N , T , L )$ , where C is the channel width, N is the oscillator dimension, T is the number of unrolled dynamics steps per block, and L is the number of Oscillatory Neurons Units. The default configuration used in our experiments is reported in Section 5.

## 5 Experiments

We evaluate OPL on CIFAR-10, CIFAR-10-C and CIFAR-100 under $\ell _ { \infty }$ and $\ell _ { 2 }$ threat models. Because the checkpoint studied in this paper uses randomized initial oscillator states, our primary robustness evaluation uses AutoAttack-rand with Expectation over Transformation (EoT). We primarily compare our result with other SOTA methods reporting AutoAttack-rand with Expectation over Transformation (EoT) results. We additionally report PGD, Square Attack, transfer attacks, and Gaussian-noise robustness as complementary diagnostics.

## 5.1 Experimental Setup

Architecture and training. Our default OPL encoder uses $C = 1 2 8$ channels, $N = 2$ oscillator dimensions, L = 3 Oscillatory Neurons Units, and $T = 3$ integration steps per unit. We pre-train the encoder with the X-PhiNet temporal prediction framework for 400 epochs and fine-tune it for 400 epochs on the same dataset (denoted 400/400). Unless otherwise stated, all OPL results in this section use this configuration. Full implementation details, including optimizer, learning-rate schedule, and EMA coefficients, are provided in Appendix C and D.

Attack suite. We report robustness under five evaluation protocols.

• AutoAttack-rand [13]: our primary evaluation for the randomized OPL checkpoint. We use $\ell _ { \infty }$ attacks with $\varepsilon = 8 / 2 5 5$ and EoT with $K = 2 0$ samples. When available, we report mean ± std over five independently trained models.

• PGD $( \ell _ { \infty }$ and $\ell _ { 2 } ) \colon$ we use $\varepsilon = 8 / 2 5 5$ and $\varepsilon = 0 . 5 .$ , respectively, with 5 random restarts and a step-count sweep $\{ 1 0 , 2 0 , 5 0 , \mathrm { 1 0 0 } \}$ . For randomized checkpoints, PGD uses EoT with $K = 5$ . We use these attacks as stronger first-order diagnostics rather than as a replacement for AutoAttack.

• Square Attack [2]: a black-box, score-based attack used as a complementary diagnostic that does not rely on backpropagated gradients.

• Transfer attacks: PGD adversarial examples are generated on a surrogate ResNet-50 and then evaluated on the target model.

• Gaussian-noise robustness: we measure accuracy under increasing additive Gaussian noise to assess stability to non-adversarial corruptions.

All $\ell _ { \infty }$ evaluations use $\varepsilon = 8 / 2 5 5$ on CIFAR-10/100. Where relevant, we include published RobustBench numbers only as context, because those references are reported under the standard AutoAttack protocol and are not directly protocol-matched to randomized OPL.

## 5.2 Comprehensive Adversarial Evaluation

Main result. Table 1 reports the adaptive robustness of randomized OPL under AutoAttack-rand with EoT $( K = 2 0 )$ . Without adversarial training, OPL reaches $7 6 . 6 3 \% \pm 0 . 7 6 \%$ robust accuracy on CIFAR-10 and 50.44% on CIFAR-100. Table 2 provides contextual comparison with prior stochastic or purification-based defenses evaluated using EoT-style adaptive protocols. Because these methods differ in architecture, test-time computation, and exact attack implementation, we use this comparison to position OPL. Best reported robust accuracy on stochastic adversarial defense models on CIFAR-10 under the same AutoAttack-rand protocol is 71.29%, meaning we claim +5.34%. All OPL results in this subsection use the same 400/400 training setup described in Section 5.1.

Table 1: Adaptive white-box evaluation of randomized OPL. OPL is evaluated with AutoAttack-rand using EoT $( K = 2 0 )$
<table><tr><td>Dataset</td><td>Evaluation protocol</td><td>Robust accuracy (%)</td></tr><tr><td>CIFAR-10</td><td>AA-rand,  $\ell _ { \infty } , \varepsilon = 8 / 2 5 5 .$  EoT  $K = 2 0$ </td><td>76.63± 0.76</td></tr><tr><td>CIFAR-100</td><td>AA-rand,  $\ell _ { \infty } , \varepsilon = 8 / 2 5 5$  EoT  $K = 2 0$ </td><td>50.44</td></tr></table>

Table 2: Robustness to AutoAttack-rand adversarial attack Contextual comparison with stochastic or adaptive defenses on CIFAR-10 under $\ell _ { \infty } , \epsilon = 8 / 2 5 5$ . OPL is evaluated with AutoAttack-rand and EoT K = 20 (Standard [10] evaluation scheme).
<table><tr><td>Model</td><td>AutoAttack-rand robust acc. (%)</td></tr><tr><td>Nie et al. (2022) [23]</td><td>71.29</td></tr><tr><td>Lee et al. (2023) [21]</td><td>70.47</td></tr><tr><td>AKOrN [22]</td><td>64.98</td></tr><tr><td>OPL (ours)</td><td> ${ \bf 7 6 . 6 3 \pm 0 . 7 6 }$ </td></tr></table>

## 5.3 Adaptive Evaluation Diagnostics

Stochastic defenses are vulnerable to overestimated robustness if attacks do not properly account for randomness or iterative computation. We therefore report several diagnostics: AutoAttack-rand with EoT, PGD attack-strength sweeps, Square Attack, and transfer attacks. These diagnostics test several common failure modes.

PGD robustness. Table 3 reports PGD accuracy as a function of attack iterations. Robust accuracy decreases monotonically as the attack budget increases, indicating that stronger first-order attacks continue to find additional adversarial examples rather than plateauing prematurely. Under $\ell _ { \infty }$ PGD-100, OPL reaches 76.10% robust accuracy; under $\ell _ { 2 }$ PGD-100, it reaches 82.11%.

Table 3: PGD robustness on randomized OPL for CIFAR-10. Clean accuracy: 86.81%. PGD uses 5 restarts and EoT (K = 5). The monotonic decrease with attack strength is consistent with informative gradients.
<table><tr><td>Attack</td><td>10</td><td>20</td><td>50</td><td>100</td></tr><tr><td> $\ell _ { 2 } ( \varepsilon = 0 . 5 )$ </td><td>82.32</td><td>82.28</td><td>82.26</td><td>82.11</td></tr><tr><td> $\ell _ { \infty } \left( \varepsilon = 8 / 2 5 5 \right)$ </td><td>79.80</td><td>77.90</td><td>76.51</td><td>76.10</td></tr></table>

Square Attack. Under Square Attack (black-box, score-based, $\ell _ { \infty } )$ , OPL achieves 81.81% robust accuracy from a clean accuracy of 86.81%. Because Square Attack does not rely on backpropagated gradients, it provides a complementary diagnostic to the white-box evaluations. Notably, Square Attack achieves higher accuracy (81.81%) than AutoAttack-rand (76.63%), which is the expected ordering when gradients are informative; the opposite ordering is a canonical signature of gradient masking.

Transfer attacks. Table 4 reports robustness to PGD adversarial examples crafted on a ResNet-50 surrogate. We treat transfer attacks as a compementary evaluation to adaptive white-box evaluations. OPL is more resistant than AKOrN alone under both $\ell _ { \infty }$ and $\ell _ { 2 }$ transfer attacks, suggesting that predictive pretraining improves off-model stability. However, the $\ell _ { \infty }$ transfer result for OPL is lower than the AutoAttack-rand result. We therefore do not interpret AutoAttack-rand as a worst-case lower bound across all possible attacks; rather, transfer attacks provide an additional diagnostic indicating that stronger adaptive attacks may further reduce robust accuracy.

Table 4: Transfer attack (ResNet-50 → target) on CIFAR-10. PGD-100 adversarial examples are generated on the surrogate model and evaluated on the target model. The $\ell _ { \infty }$ and $\ell _ { 2 }$ columns use $\varepsilon = 8 / 2 5 5$ and ε = 0.5, respectively.
<table><tr><td>Model</td><td>Clean (%)</td><td> $\mathrm { P G D } { \cdot } { \ell } _ { \infty } \left( \% \right)$ </td><td>PGD-l2 (%)</td></tr><tr><td>AKOrN</td><td>84.99</td><td>67.05</td><td>76.32</td></tr><tr><td>OPL</td><td>86.81</td><td>70.59</td><td>78.52</td></tr></table>

## 5.4 Ablation Studies

We use ablations to answer two questions: (i) does robustness primarily come from the oscillatory encoder or from predictive pretraining, and (ii) which dynamical hyperparameters place the model in a robust or collapsed regime? Unless otherwise stated, robust accuracy in this subsection is measured with AA-rand $( \ell _ { \infty } , \varepsilon = 8 / 2 5 5$ , EoT $K = 2 0 )$ , matching the primary evaluation in Section 5.2. We report PGD diagnostics for the full model separately in Table 3 and avoid repeating them here.

## 5.4.1 Architecture vs. Predictive Pretraining

Table 5 isolates four conditions: (i) a standard ResNet-50 baseline, (ii) X-PhiNet pretraining on ResNet-50, (iii) AKOrN without predictive pretraining, and (iv) the full OPL model. The pattern is clear. Applying X-PhiNet to a standard ResNet-50 encoder does not yield measurable adversarial robustness on its own (0.31%), whereas AKOrN without predictive pretraining is already substantially robust (64.98%). Within the tested model family, this indicates that oscillatory dynamics are the primary robustness-supporting component, while predictive pretraining alone is insufficient on a standard ResNet encoder. Moreover, adding the X-PhiNet predictive pretraining pipeline on top of AKOrN increases robust accuracy to $7 6 . 6 3 \overset { \_ } { \pm } 0 . 7 6 \%$ , a gain of 11.65 percentage points over AKOrN alone. This result suggests that predictive pretraining substantially strengthens the robustness bias induced by the oscillatory encoder AKOrN.

Table 5: Disentangling architecture and predictive pretraining on CIFAR-10. All models are evaluated under the same randomized white-box protocol: AA-rand $( \ell _ { \infty } , \varepsilon = 8 / 2 5 5$ , EoT $K = 2 0 )$ . The table isolates the contributions of a standard encoder, predictive pretraining, oscillatory dynamics, and their combination.
<table><tr><td>Model</td><td>Oscillatory encoder</td><td>Predictive SSL</td><td>Clean (%)</td><td> $\mathbf { A A - r a n d - } \ell _ { \infty } \ ( \% )$ </td></tr><tr><td>ResNet-50</td><td>No</td><td>No</td><td>82.17</td><td>0.31</td></tr><tr><td> $\mathrm { X } { \mathrm { - P h i N e t } } + \mathrm { R e s N e t } { - } 5 0$ </td><td>No</td><td>Yes</td><td>82.04</td><td>0.17</td></tr><tr><td>AKOrN</td><td>Yes</td><td>No</td><td>84.99</td><td>64.98</td></tr><tr><td>OPL</td><td>Yes</td><td>Yes</td><td>86.81</td><td> ${ \bf 7 6 . 6 3 \pm 0 . 7 6 }$ </td></tr></table>

Gaussian-noise robustness. Figure 3 shows that the same ordering extends beyond adversarial perturbations. AKOrN is consistently more stable than the ResNet-50 and X-PhiNet-only controls across the full noise range, and OPL performs best throughout. This result does not replace adversarial evaluation, but it supports the broader picture that oscillatory dynamics improve stability to both worst-case and stochastic perturbations.

## 5.4.2 Hyperparameter

## sensitivity and the robust regime.

![](images/9e93112a6edd201ed4e046a8457c8ebb95e5c4a676f5e9ff9656fa88bf03f83c.jpg)  
Figure 3: Robustness to additive Gaussian noise on CIFAR-10. We report mean top-1 test accuracy as a function of the Gaussian noise standard deviation σ. Each point is averaged over three independently sampled Gaussian-noise draws on the full 10,000-image test set; $\sigma = 0$ corresponds to clean accuracy. OPL maintains the highest accuracy across the evaluated noise range, while AKOrN alone remains more stable than the X-PhiNet-only and ResNet-50 controls.

Table 6 summarizes the sweep over oscillator dimension N, integration time T, and training schedule. Two regimes emerge. The N=2 family contains all robust configurations, with the 400/400, T=3 model yielding the strongest overall tradeoff. Within this family, increasing T from 3 to 5 sharply reduces clean accuracy while also reducing robust accuracy, so longer unrolling does not improve robustness in our setting. By contrast, increasing oscillator dimension from N=2 to N=4 raises clean accuracy toward 92% but collapses AA-rand robustness to near zero. This non-monotonic pattern argues against a simple capacity explanation: robustness appears only in a narrow dynamical regime rather than improving monotonically with model flexibility.

## 6 Discussion, Trade-offs, and Limitations

Mechanistic hypotheses. The clearest evidence that robustness is not explained by capacity alone comes from the N=4 family. Across the sweep in Table 6, increasing oscillator dimension from $N { = } 2$ to N=4 pushes clean accuracy as high as 92% while collapsing AA-rand robustness to near zero. This non-monotonic behavior suggests that OPL’s robustness depends on a narrow dynamical regime rather than on simply increasing model flexibility. We conjecture that the $N { = } 2$ regime enforces strict degree-1 phase consensus (equivalent to a 2D rotation), creating a tightly coupled dynamical system in which individual features cannot independently align with adversarial gradient vectors. Increasing to N=4 provides enough degrees of freedom for the network to embed adversarial perturbations orthogonally to the consensus manifold, bypassing the structural defense. We predict that analogous low-dimensional couplings in other iterative architectures (such as complex-valued networks or strict 2D Lie group convolutions) should exhibit similar robustness phase transitions. Testing this hypothesis in other dynamical or iterative models is an important direction for future work.

Table 6: Hyperparameter sensitivity on CIFAR-10. One representative result is shown for each unique configuration in the current sweep. All robust accuracies are measured with AA-rand $( \ell _ { \infty } , \varepsilon = 8 / 2 5 5$ , EoT $K = 2 0 )$ . Channel width is fixed at $C = 1 2 8$ throughout. Robustness is concentrated in the N=2 family; increasing oscillator dimension to N=4 improves clean accuracy but collapses robust accuracy.
<table><tr><td>Pre.</td><td>Fine.</td><td>N</td><td>T</td><td>L</td><td>Clean (%) – Robust (%)</td></tr><tr><td>400</td><td>400</td><td>2</td><td>3</td><td>3</td><td>86.81 - 76.63</td></tr><tr><td>200</td><td>400</td><td>2</td><td>3</td><td>3</td><td>84.68–69.56</td></tr><tr><td>400</td><td>400</td><td>2</td><td>5</td><td>3</td><td>65.36-54.05</td></tr><tr><td>50</td><td>50</td><td>2</td><td>5</td><td>3</td><td>53.56–46.54</td></tr><tr><td>400</td><td>400</td><td>4</td><td>3</td><td>3</td><td>92.06–0.03</td></tr><tr><td>400</td><td>100</td><td>4</td><td>3</td><td>3</td><td>81.65-4.01</td></tr><tr><td>400</td><td>400</td><td>4</td><td>5</td><td>3</td><td>92.10–0.02</td></tr><tr><td>50</td><td>50</td><td>4</td><td>5</td><td>3</td><td>85.56-0.04</td></tr><tr><td>50</td><td>50</td><td>4</td><td>3</td><td>3</td><td>87.35-0.15</td></tr><tr><td>0</td><td>500</td><td>4</td><td>3</td><td>3</td><td>91.85–0.05</td></tr></table>

Interpreting the clean–robust trade-off. OPL operates under a different operating regime than large adversarially trained CIFAR models [5, 1]. Our best randomized checkpoint reaches 86.81% clean accuracy and 76.63% AA-rand robustness on CIFAR-10 without adversarial-example generation during training. We do not interpret this as a matched-capacity or matched-compute comparison to adversarial training, since the compared methods differ in architecture, training procedure, and evaluation protocol. Instead, the main empirical result is narrower: substantial adversarial robustness can emerge without adversarial training, but in our current setup, this robustness is associated with lower clean accuracy than the highest-capacity adversarially trained baselines.

Limitations. Our work should be considered with the following limitations in mind.

• Our experiments are currently limited to CIFAR-10, CIFAR-10-C (cf. Appendix D) and CIFAR-100; and do not include higher-resolution datasets such as ImageNet. Evaluating this dataset would require implementing oscillatory neurons in model architectures that perform well in those settings, as done in the literature [10].

• Although the contrast between the N=2 and N=4 regimes suggests that oscillatory dynamics matter, we do not yet provide a mechanistic theory or diagnostic that predicts in advance when a given configuration will be robust.

## 7 Broader Impacts

A positive implication of this work is that robustness without adversarial training could significantly reduce the computational cost of deploying robust models, making them accessible in low-resource or on-device settings such as medical imaging or edge AI. However, the same efficiency gains could also lower the barrier for deploying robust models in surveillance, filtering, or adversarial settings where resistance to attacks may be misused. As such, improvements in robustness should be considered in the broader context of dual-use machine learning technologies.

## References

[1] Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, and Amir Houmansadr. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsifi cation, June 2024.

[2] Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square attack: A query-efficient black-box adversarial attack via random search. In European Conference on Computer Vision (ECCV), 2020.

[3] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture, April 2023. arXiv:2301.08243 [cs].

[4] Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. In Proceedings of the 35th International Conference on Machine Learning, pages 274–283. PMLR, July 2018.

[5] Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, and Bhavya Kailkhura. Adversarial robustness limits via scaling-law and human-alignment studies. In 2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ICML 2024), 2024.

[6] Tianlong Chen, Sijia Liu, Shiyu Chang, Yu Cheng, Lisa Amini, and Zhangyang Wang. Adversarial Robustness: From Self-Supervised Pre-Training to Fine-Tuning. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 696–705, Seattle, WA, USA, June 2020. IEEE.

[7] Xinlei Chen and Kaiming He. Exploring Simple Siamese Representation Learning. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15745– 15753, June 2021. ISSN: 2575-7075.

[8] Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao, and Yuto Miyatake. Lyapunov-stable deep equilibrium models. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, volume 38 of AAAI’24/IAAI’24/EAAI’24, pages 11615–11623. AAAI Press, 2024.

[9] Jeremy Cohen, Elan Rosenfeld, and J. Zico Kolter. Certified adversarial robustness via randomized smoothing. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 1310–1320. PMLR, 2019.

[10] Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

[11] Francesco Croce, Sven Gowal, Thomas Brunner, Evan Shelhamer, Matthias Hein, and Taylan Cemgil. Evaluating the Adversarial Robustness of Adaptive Test-time Defenses, July 2022. arXiv:2202.13711 [cs].

[12] Francesco Croce and Matthias Hein. Minimally distorted adversarial examples with a fast adaptive boundary attack. In International Conference on Machine Learning (ICML), 2020.

[13] Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In Proceedings ofthe 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.

[14] Matthieu Destrade, Oumayma Bounou, Quentin Le Lidec, Jean Ponce, and Yann LeCun. Value-guided action planning with JEPA world models, December 2025. arXiv:2601.00844 [cs].

[15] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent a new approach to self-supervised learning. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, pages 21271–21284, Red Hook, NY, USA, 2020. Curran Associates Inc.

[16] Dan Hendrycks, Mantas Mazeika, Saurav Kadavath, and Dawn Song. Using self-supervised learning can improve model robustness and uncertainty. In Advances in Neural Information Processing Systems, volume 32, 2019.

[17] Daniel Hendrycks. Cifar-10-c and cifar-10-p, January 2019.

[18] Satoki Ishikawa, Makoto Yamada, Han Bao, and Yuki Takezawa. Phinets: Brain-inspired noncontrastive learning based on temporal prediction hypothesis. In The Thirteenth International Conference on Learning Representations, 2025.

[19] Qiyu Kang, Yang Song, Qinxu Ding, and Wee Peng Tay. Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 14925–14937. Curran Associates, Inc., 2021.

[20] Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial examples in the physical world. CoRR, abs/1607.02533, 2016.

[21] Minjong Lee and Dongwoo Kim. Robust Evaluation of Diffusion-Based Adversarial Purification, December 2023. arXiv:2303.09051 [cs].

[22] Takeru Miyato, Sindy Löwe, Andreas Geiger, and Max Welling. Artificial kuramoto oscillatory neurons. In The Thirteenth International Conference on Learning Representations, 2025.

[23] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, and Anima Anandkumar. Diffusion models for adversarial purification. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 16805–16827. PMLR, 2022.

[24] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

## A Mathematical Derivations: Full Kuramoto ODE, projection operators, X-PhiNet loss functions.

## A.1 Kuramoto layer

The Kuramoto layers are the layers that implement the Kuramoto dynamics between oscillators and update the input image according to this dynamic. We will describe in detail here the calculations made by a Kuramoto layer in an AKOrN encoder.

A Kuramoto layer consists in a single layer of neurons that receives an initial state $X \in \mathbb { R } ^ { C \times H \times W }$ and an initial conditional stimuli $\overset { \smile } { Y } \in \mathbb { R } ^ { \sp { C } \times H \times W }$ as input. We assume that $C = K N$ , so that each spatial location $( h , w )$ of X contains K oscillator states $\mathbf { x } _ { k , h , w } \in \mathbb { R } ^ { N }$

A single Kuramoto layer will apply $T \in \mathbb { N } ^ { * }$ discrete updates to the initial state of the oscillator system $X ^ { ( 0 ) } : = X$ by applying the following Kuramoto scheme. For $t = 0 , \ldots , T - 1$

$$
\begin{array} { r l } & { \Delta \mathbf { x } _ { k , h , w } ^ { ( t ) } = \Omega _ { k , h , w } \mathbf { x } _ { k , h , w } ^ { ( t ) } + \mathrm { P r o j } _ { \mathbf { x } _ { k , h , w } ^ { ( t ) } } \left( \mathbf { y } _ { k , h , w } + \displaystyle \sum _ { k ^ { \prime } , \Delta h , \Delta w } \mathbf { J } _ { k , k ^ { \prime } , \Delta h , \Delta w } \mathbf { x } _ { k ^ { \prime } , h + \Delta h , w + \Delta w } ^ { ( t ) } \right) , } \\ & { \mathbf { x } _ { k , h , w } ^ { ( t + 1 ) } = \Pi \left( \mathbf { x } _ { k , h , w } ^ { ( t ) } + \gamma \Delta \mathbf { x } _ { k , h , w } ^ { ( t ) } \right) } \end{array}\tag{2}
$$

where $\Omega _ { k , h , w } \in \mathbb { R } ^ { N \times N }$ is a learned antisymmetric matrix controlling intrinsic rotation, $\Pi : \mathbf { x } \mapsto$ $\begin{array} { r } { \frac { \mathbf { x } } { \| \mathbf { x } \| _ { 2 } } , \gamma > 0 , \mathbf { J } _ { k , k ^ { \prime } , \Delta h , \Delta w } } \end{array}$ are learned coupling weights, and ${ \bf y } _ { k , h , w }$ is an input-dependent drive term. Because we use an implementation of the Kuramoto dynamical system with convolutions in our experiments, $\mathbf { J } _ { k , k ^ { \prime } , \Delta h , \Delta w }$ will be the learned connectivity weights of a convolution kernel and $k , k ^ { \prime }$ stand for output and input channels indexes. The projection operator

$$
\operatorname { P r o j } _ { \mathbf { x } } ( \mathbf { y } ) = \mathbf { y } - \langle \mathbf { y } , \mathbf { x } \rangle \mathbf { x }
$$

ensures the velocity vector of an oscillator x effectively lives in the tangent plane of $\mathbf { S } ^ { N - 1 }$ at x.

The output of this layer is the final state of the system of oscillators after the updates : $\mathbf { X } ^ { ( T ) }$

In order to extract relevant features from this final oscillatory state, we will rely on a phase-invariant feature extractor module introduced in [22]. We will motivate the use of this feature extractor below.

## A.2 Feature Extractor

The feature extractor module, called the readout module $\mathcal { R M } : \mathbb { R } ^ { K \times N \times H \times W } \longrightarrow \mathbb { R } ^ { K \times N \times H \times W }$ was introduced by [22] and is specifically designed to leverage the symmetries and properties of the oscillators to extract meaningful features. The oscillators are constrained to live on the unit hyper-sphere of $\mathbb { R } ^ { N }$ due to the projector operator Π. As a result, all information and patterns are encoded in the relative directions of the oscillators rather than their absolute positions.

Consequently, the readout module $\mathcal { R M }$ is phase-invariant: adding a constant phase to all oscillators does not change the output of the module. Formally, for $X \in \mathbb { R } ^ { K \times N \times H \times \mathbf { \bar { W } } }$ and a rotation R : $\mathbb { R } ^ { N } \longrightarrow \mathbb { R } ^ { N }$ , if we denote by

$$
R ( X ) : = \left( R ( \mathbf { x } _ { i } ) \right) _ { i \in [ K ] \times [ H ] \times [ W ] }
$$

then the readout satisfies

$$
\mathcal { R M } \big ( R ( { \mathbf { X } } ) \big ) = \mathcal { R M } ( { \mathbf { X } } ) .
$$

This formulation ensures that $\mathcal { R M }$ depends only on the relative orientations of the oscillators, preserving the intrinsic structure of the data encoded on the hypersphere.

In practice, we choose the simple following readout module

$$
\mathcal { R M } ( X ) = g \left( \left( \left. \sum _ { i } \mathbf { U } _ { j i } \mathbf { x } _ { i } \right. _ { 2 } \right) _ { j \in [ K ] \times [ N ] \times [ H ] \times [ W ] } \right)
$$

with $g \ : \ \mathbb { R } ^ { K \times N \times H \times W } \longrightarrow \ \mathbb { R } ^ { K \times N \times H \times W }$ a linear layer (that can be also set to identity) and $\mathbf { U } _ { j i } \in \mathbb { R } ^ { N \times N }$ learned weight matrices.

## A.3 Oscillatory Neurons Unit

To create a neural network from those Kuramoto layers, one needs to stack multiple Kuramoto layers and to add a readout module between two succcessive kuramoto layers that aims at extracting features from the final oscillatory states to create a new conditional stimuli C.

## A.4 X-PhiNet Predictive learning setup

The X-PhiNet pre-training architecture we implemented is trained by jointly minimizing two losses : the first one, Sim-1 for the hippocampus model, and the second one, Sim-2 for the neocortex model. Those two losses are combined to update the encoder f by backpropagation.

Sim-1. This hippocampus loss is a symmetric negative cosine loss function, exactly as in the SimSiam architecture [7] :

$$
L _ { \mathrm { C o s } } ( \theta ) = - \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } \frac { ( \boldsymbol { h } _ { i } ^ { ( 1 ) } ) ^ { \top } \boldsymbol { s } \boldsymbol { g } ( \boldsymbol { z } _ { i } ^ { ( 2 ) } ) } { \| \boldsymbol { h } _ { i } ^ { ( 1 ) } \| _ { 2 } \| \boldsymbol { s } \boldsymbol { g } ( \boldsymbol { z } _ { i } ^ { ( 2 ) } ) \| _ { 2 } } - \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } \frac { s g ( \boldsymbol { z } _ { i } ^ { ( 1 ) } ) ^ { \top } \boldsymbol { h } _ { i } ^ { ( 2 ) } } { \| \boldsymbol { s } \boldsymbol { g } ( \boldsymbol { z } _ { i } ^ { ( 1 ) } ) \| _ { 2 } \| \boldsymbol { h } _ { i } ^ { ( 2 ) } \| _ { 2 } }
$$

This loss is appropriate because it becomes minimal when the predicted representation from CA3, $h ^ { ( 1 ) }$ , is perfectly aligned with the temporally distant signal $z ^ { ( 2 ) }$

Next, we examine how information transfer from the hippocampus to the neocortex is captured during the model’s training.

Sim-2. In the neocortex, this loss is defined as an MSE loss between the CA1 output and the representation $z = f ( x )$ from the entorhinal cortex (EC):

$$
L _ { \mathrm { N C } } ( \theta ) = \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } \left\| y _ { i } ^ { ( 1 ) } - s g ( z _ { i } ) \right\| _ { 2 } ^ { 2 } + \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } \left\| y _ { i } ^ { ( 2 ) } - s g ( z _ { i } ) \right\| _ { 2 } ^ { 2 }
$$

The final loss for X-PhiNet is the sum of Sim-1 and Sim-2. Sim-2 is considered a slow-learning loss.

Stable encoder. To incorporate long-term memory, a different encoder $f _ { l o n g }$ has been added to X-PhiNet’s third stream. This stream is designed to model the neocortex and $f _ { l o n g }$ corresponds to a stable encoder that further improve slow learning with an ability to maintain long-term signals. The weights of $f _ { l o n g }$ are updated by using an exponential moving average of the model parameters of f and $f _ { l o n g }$ . It is similar to techniques used by other self-supervised methods as BYOL architecture [15]. If we denote by ξ and $\xi _ { l o n g }$ the respective parameters of f and $f _ { l o n g }$ we have the following update scheme :

$$
\xi _ { l o n g } \gets \beta \xi _ { l o n g } + ( 1 - \beta ) \xi
$$

with $\beta \in [ 0 , 1 ]$ a hyperparameter usually chosen close to 1.

The benefit of maintaining a separate encoder whose weights are updated via an exponential moving average (EMA) scheme is that it yields a representation network with more stable parameters during training, while simultaneously mitigating the risk of representation collapse.

PhiNet has demonstrated several advantages for encoder pre-training compared to other state-of-theart architectures such as SimSiam [7] and BYOL [15]. Among these benefits are its robustness to the choice of the weight decay hyperparameter and its superior performance in online learning settings.

Having introduced self-supervised learning and the neuroscience-inspired models that guide our approach, we now turn to the task at hand. Our goal is to leverage the strengths of AKOrN and X-PhiNet to push beyond current state-of-the-art performance in term of robustness to adversarial attacks.

## B Adversarial Exemples and Adversarial Attacks

While some works report only test-set accuracies, this evaluation alone may not be sufficient to faithfully assess the performance of a model. Indeed, it is common for a model to achieve high test accuracy while failing to learn robust and generalizable features, effectively overfitting to the data distribution. This issue becomes apparent when the model is attacked: adversarial examples—inputs that are very close to correctly classified evaluation samples—can highlight those weakness in a model, revealing its vulnerability.

![](images/eb8f9714db0ac0cd424b041a1648d2c13c42b5f53d21aea13fc6577e9f299c3a.jpg)  
Figure 4: Adversarial perturbation illustration

Many different methods have been proposed to generate adversarial examples and reliably evaluate a model’s robustness. Recently, one evaluation method has gained widespread popularity for its reliability. Known as AutoAttack [13], it is described by its authors as a “parameter-free, computationally affordable, and user-independent ensemble of attacks to test adversarial robustness.” AutoAttack has been tested on more than 50 different models and is currently the standard tool used by the reference benchmark for adversarial robustness, RobustBench [10].

Let’s see what are the main ideas used by the AutoAttack to realize those adversarial attacks.

## B.1 Adversarial attack

First, we formally define what constitutes an adversarial example for a K-class classifier $g : { \mathcal { D } } \subset$ $\mathbb { R } ^ { d } \to \mathbb { R } ^ { K }$ , which assigns labels according to argmax $\tau _ { k } g _ { k } ( \cdot )$ . Let $x _ { \mathrm { o r g } } \in \mathbb { R } ^ { d }$ be a point correctly classified by $g$ as class $c .$ Given a distance metric $\bar { d } ( \cdot , \cdot )$ and a perturbation budget $\epsilon > 0$ , the feasible set of the attack at $x _ { o r g }$ is defined as

$$
\mathcal { B } _ { \epsilon } ( x _ { \mathrm { o r g } } ) = \{ z \in \mathcal { D } \mid d ( x _ { \mathrm { o r g } } , z ) < \epsilon \} .
$$

The most popular attacks rely on $l _ { p }$ distances $d : ( x , y ) \in \mathbb { R } ^ { 2 d } \longmapsto \| x - y \| _ { r }$ with $p \in \{ 2 , \infty \}$ . This choice is made for practical reasons as it will be explained below.   
An adversarial example for $g$ at $x _ { \mathrm { o r g } }$ is then any input such that

$$
\arg \operatorname* { m a x } _ { k = 1 , \ldots , K } g _ { k } ( z ) \neq c \quad \mathrm { a n d } \quad z \in \mathcal { B } _ { \epsilon } ( x _ { \mathrm { o r g } } ) .
$$

Intuitively, it is an input that is almost indistinguishable from $x _ { \mathrm { o r g } }$ to a human observer, but is misclassified by the model.

To find an adversarial example $z ,$ it is common to solve a constrained optimization problem

$$
\operatorname* { m a x } _ { z \in \mathcal { D } } L ( g ( z ) , c ) \quad \mathrm { s u c h t h a t } \quad d ( x _ { \mathrm { o r i g } } , z ) \leq \epsilon , \ z \in \mathcal { D } .\tag{3}
$$

where L is a function that enforces z into not being assigned to class c.   
Let’s explore a popular algorithm used in adversarial attack to solve this kind of problem.

## B.1.1 Auto-PGD

In PGD-attack [20], one of the most popular white-box attack, used by AutoAttack, the function L used is the cross-entropy $- \log ( q _ { c } ( g ( \bar { z } ) )$ where $q _ { c } ( g ( z ) )$ is the predicted probability assigned to the correct class c.

To solve this optimization problem (3), [13] introduced Auto-PGD, a variant of the Projected Gradient Descet (PGD) algorithm that solves issues one may face while using PGD for solving this optimization problem.

Let’s recall first the regular PGD algorithm. For a given $f : \mathbb { R } ^ { d } \longrightarrow \mathbb { R } , s \subset \mathbb { R } ^ { d }$ and the problem $\operatorname* { m a x } _ { x \in S } f ( x )$ , the PGD is an iterative algorithm. For $k = 1 , \ldots , N _ { i t e r }$

$$
x ^ { ( k + 1 ) } = P _ { S } \left( x ^ { ( k ) } + \eta ^ { ( k ) } \nabla f ( x ^ { ( k ) } ) \right) ,
$$

where $\eta ^ { ( k ) }$ is the step size at iteration k and $P _ { S }$ is the projection onto $s$ and the initial point $x ^ { ( 0 ) }$ is either $x _ { \mathrm { o r g } } \ \mathrm { o r } \ x _ { \mathrm { o r g } } + \zeta$ with ζ a random vector such that $( \bar { x } _ { \mathrm { o r g } } + \zeta ) \in \mathcal { S }$ . We can notice that Problem 3 can be effectively solved for $d ( \cdot , \cdot )$ being an $l _ { 2 }$ or an $l _ { \infty }$ distance.

In the original PGD attack, the step size is fixed, i.e. $\eta ^ { ( k ) } = \eta$ for every iteration k. This choice is suboptimal and does not guarantee convergence. In contrast, Auto-PGD divides the total number of iterations $N _ { \mathrm { i t e r } }$ into an exploration phase and an exploitation phase, during which the step size $\eta$ is adaptively updated according to different criteria.

Let’s present the ideas of Auto-PGD in four steps.

Gradient step. Auto-PGD updates use a step size $\eta ^ { ( k ) }$ at iteration k and incorporates a momentum term, which is absent in the original PGD algorithm. The initial steps of APGD are typically large, so the momentum helps to regularize the updates by incorporating information from previous iterations, smoothing the optimization trajectory. The update step corresponds to the following :

$$
\begin{array} { r l } & { z ^ { ( k + 1 ) } = P _ { S } \left( x ^ { ( k ) } + \eta ^ { ( k ) } \nabla f ( x ^ { ( k ) } ) \right) } \\ & { x ^ { ( k + 1 ) } = P _ { S } \left( x ^ { ( k ) } + \alpha \cdot ( z ^ { ( k + 1 ) } - x ^ { ( k ) } ) + ( 1 - \alpha ) \cdot ( x ^ { ( k ) } - x ^ { ( k - 1 ) } ) \right) } \end{array}
$$

where $\alpha \in [ 0 , 1 ]$ regulates the influence of the momentum term. $\alpha = 0 . 7 5$ is used in practice.

Step size selection. The initial step size at iteration 0 is chosen such that $\eta ^ { ( 0 ) } = 2 \epsilon$ . Then the algorithm will decide weather or not the step size will be halved at fixed chosen checkpoints $w _ { 0 } = 0 < w _ { 1 } , \cdot \cdot \cdot < w _ { n } < N _ { \mathrm { i t e r } }$ . To update the step size, one of the two following conditions has to be true:

$$
\begin{array} { r l } & { 1 . \quad \displaystyle \sum _ { i = w _ { j - 1 } } ^ { w _ { j } - 1 } \mathbf { 1 } _ { f ( x ^ { ( i + 1 ) } ) > f ( x ^ { ( i ) } ) } < \rho \cdot ( w _ { j } - w _ { j - 1 } ) , } \\ & { 2 . \quad \displaystyle \eta ^ { ( w _ { j - 1 } ) } \equiv \eta ^ { ( w _ { j } ) } \quad \mathrm { a n d } \quad f _ { \mathrm { m a x } } ^ { ( w _ { j - 1 } ) } \equiv f _ { \mathrm { m a x } } ^ { ( w _ { j } ) } , } \end{array}
$$

• 1. Insufficient Improvement: The step size is halved if the proportion of iterations that increased the objective function $f$ since the prior checkpoint, $w _ { j - 1 }$ , is below a threshold $\rho$ (typically $\rho = 0 . 7 5 )$ .

• 2. Stagnation and Cycling Prevention: The step size is also halved if it remained unchanged at the previous checkpoint $( w _ { j - 1 } )$ and the maximum value of the objective function f has not improved since that checkpoint. This rule helps the algorithm escape cycles.

Checkpoints restart. At each checkpoint $w _ { j }$ where the step size has been halved, the algorithm restarts from the best point found so far. It means in that case that we fix $x ^ { w _ { j } + 1 } : = x _ { \operatorname* { m a x } }$ with $x _ { \mathrm { m a x } } : = \operatorname { a r g m a x } _ { k \leq w _ { j } } f ( x ^ { ( k ) } )$ . The idea is to refine the research of an adversarial example around the neighborhood of the current best candidate solution.

Exploration vs Exploitation. The choice of the checkpoints $w _ { j }$ at which the step size can be reduced is important to define a transition between an initial exploration phase exploring the whole feasible set S and an exploitation phase where the step size is reduced more often leading to more frequent improvements of the objective but with smaller magnitude.

The checkpoints are defined as follow:

$$
w _ { j } : = \lceil p _ { j } N _ { \mathrm { i t e r } } \rceil \leq N _ { \mathrm { i t e r } }
$$

with $p _ { j } \in [ 0 , 1 ]$ defined as

$$
\left\{ { \begin{array} { l } { p _ { 0 } = 0 } \\ { p _ { 1 } = 0 . 2 2 } \\ { p _ { j + 1 } = p _ { j } + \operatorname* { m a x } \{ p _ { j } - p _ { j - 1 } - 0 . 0 3 , 0 . 0 6 \} } \end{array} } \right.
$$

The only remaining free parameter in the implementation of AutoAttack is the budget $N _ { \mathrm { i t e r } }$ . By construction, the resulting checkpoints are increasingly dense as the iterations progress.

In conclusion, the ensemble AutoAttack consists in combination of two parameter-free versions of PGD, APGD with cross-entropy loss, APGD with Difference of Logits Ratio Loss [13], and two existing complementary attacks, FAB [12] and Square Attack [2].

The adversarial accuracy of a model is obtained by evaluating it under an adversarial attack. It corresponds to the number of correctly classified samples that cannot be successfully perturbed into adversarial examples by the attacker (here, AutoAttack) divided by the number of samples in total. It is this adversarial accuracy that serves as the principal metric highlighted in robustness benchmark for comparing model robustness.

In the next part, we will focus on the current state of the research in adversarial accuracy. We will rely on RobustBench [10], a standardized online benchmark for adversarial robustness.

## C Experimental Settings and Reproducibility Details

All experiments follow a two-stage training pipeline consisting of self-supervised pretraining followed by supervised fine-tuning. Unless otherwise stated, the results reported for the OPL CIFAR-10 setting use the configuration described below.

Training pipeline. We first pretrain the model using a self-supervised objective and then fine-tune the pretrained backbone for downstream image classification. For the experiments in the main text, the self-supervised method is PhiNet and the backbone is AKOrN. Fine-tuning is initialized from the pretrained checkpoint.

Dataset and data loading. We use CIFAR-10, which contains 10 classes. The default data root is data/. During training, dataloaders use shuffling for the training split and no shuffling for the test split. All dataloaders use num\_workers=1 and pin\_memory=True.

For pretraining, augmentations are determined by the SSL method, and in the X-PhiNet setting we use the PhiNet augmentation pipeline. For fine-tuning, we use the finetuning augmentation strategy implemented by augmentation\_strong(...). For evaluation, we use a minimal test-time transform consisting of ToTensor() only.

AutoAttack. We use AutoAttack to compare with the state-of-the-art adversarial purification methods. To make a fair comparison, we uses their codebase: https://github.com/RobustBench/ with default hyperparameters for evaluation. Similarly, we set $\epsilon = 8 / 2 5 5$ for AutoAttack $l _ { \infty } ,$ on CIFAR-10 and CIFAR-100.

There are two versions of AutoAttack: (i) the Standard version, which contains four attacks: APGD-CE, APGD-T, FAB-T and Square, and is mainly used for evaluating deterministic defense methods, and (ii) the Rand version, which contains two attacks: APGD-CE and APGD-DLR, and is used for evaluating stochastic defense methods. Because our method is stochastic, we use the Rand version and set EoT=20 by default.

Optimization. Both pretraining and fine-tuning use the Adam optimizer. Pretraining is run for 700 epochs with batch size 512, learning rate $1 \times 1 0 ^ { - 4 }$ , and weight decay $1 \times 1 0 ^ { - 5 }$ . Fine-tuning is run for 400 epochs with batch size 512, learning rate $1 \times 1 0 ^ { - 4 }$ , and weight decay 0.0. For single-run ablations, we use seed 0. For the main CIFAR-10 result, we report the mean and standard deviation over five independently trained runs. Training is run on CUDA if available, otherwise CPU.

PhiNet configuration. For X-PhiNet, we use a projection output dimension of 2048. The X-PhiNet loss uses an MSE loss ratio of 0.5, an orientation loss ratio of 0.1, and an EMA coefficient β = 0.99 for the slow encoder update.

AKOrN backbone configuration. For the AKOrN backbone, we use 128 channels, oscillator dimensionality n = 2, T = 3 recurrent steps, and L = 3 layers. The interaction operator is convolutional (J="conv"), with kernel sizes [9, 7, 5]. We use reorientation kernel size 3 and reorientation order 2. Normalization settings are batch normalization $( \mathtt { n o r m } \mathtt { = } " \mathtt { b n } " )$ and group normalization (c\_norm="gn"). Additional AKOrN settings are gamma=1.0, use\_omega=True, init\_omg=1.0, global\_omg=True, learn\_omg=True, and ensemble=1. Randomness in the AKOrN forward pass is enabled.

Checkpointing and logging. During pretraining, if the SSL model exposes a slow encoder, we save the slow encoder backbone; otherwise, we save the backbone weights directly. During fine-tuning, we save both the final checkpoint.

Exact configuration used. Table 7 summarizes the exact hyperparameter configuration corresponding to the training script used in our experiments.

Command used. For completeness, the following command was used for the reported configuration:

```shell
python3 train.py \
--dataset cifar10 \
--run_pretraining \
--run_finetuning \
--load_from_pretrain \
--ssl_method X-PhiNet \
--backbone akorn \
--pretrain_epochs 400 \
--finetune_epochs 400 \
--pretrain_bs 512 \
--finetune_bs 512 \
--pretrain_lr 1e-4 \
--finetune_lr 1e-4 \
--out_dim 2048 \
--ch 128 \
--randomness True \
--n 2 \
--T 3 \
--L 3 \
--use_wandb \
--mse_loss_ratio 0.5 \
--ori_loss_ratio 0.1 \
--pretrain_weight_decay 1e-5
```

## D Further Experimental Results

OPL with different predictive learning Frameworks Table 8 displays initial experiments conducted with different predictive learning frameworks before we selected X-PhiNet for later experiments and work.

Table 7: Experimental settings for reproducibility for the OPL CIFAR-10 experiments.
<table><tr><td>Category</td><td>Setting</td><td>Value</td></tr><tr><td>Training pipeline</td><td>Phases</td><td>Self-supervised pretraining + supervised fine-</td></tr><tr><td>SSL method</td><td>Method</td><td>tuning X-PhiNet</td></tr><tr><td>Backbone</td><td>Architecture</td><td>AKOrN</td></tr><tr><td>Dataset</td><td>Dataset</td><td>CIFAR-10</td></tr><tr><td>Dataset</td><td>Number of classes</td><td>10</td></tr><tr><td>Random seed</td><td>Seed</td><td>0</td></tr><tr><td>Device</td><td>Hardware selection</td><td>CUDA if available, else CPU</td></tr><tr><td>Pretraining</td><td>Epochs</td><td>400</td></tr><tr><td>Fine-tuning</td><td>Epochs</td><td>400</td></tr><tr><td>Pretraining</td><td>Batch size</td><td>512</td></tr><tr><td>Fine-tuning</td><td>Batch size</td><td>512</td></tr><tr><td>Pretraining</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Fine-tuning</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Pretraining</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Fine-tuning</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Pretraining</td><td>Weight decay</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Fine-tuning</td><td>Weight decay</td><td>0.0</td></tr><tr><td>SSL head</td><td>Projection dimension</td><td>2048</td></tr><tr><td>AKOrN</td><td>Channels (ch)</td><td>128</td></tr><tr><td>AKOrN</td><td>Oscillator dimensions (n)</td><td>2</td></tr><tr><td>AKOrN AKOrN</td><td>Time steps (T)</td><td>3</td></tr><tr><td></td><td>Number of layers (L)</td><td>3</td></tr><tr><td>AKOrN</td><td>Interaction operator (J)</td><td>conv</td></tr><tr><td>AKOrN AKOrN</td><td>Kernel sizes</td><td>[9,7,5]</td></tr><tr><td>AKOrN</td><td>Reorientation kernel size</td><td>3</td></tr><tr><td>AKOrN</td><td>Reorientation order (roN)</td><td>2</td></tr><tr><td>AKOrN</td><td>Normalization</td><td>bn</td></tr><tr><td>AKOrN</td><td>Channel normalization</td><td>gn</td></tr><tr><td>AKOrN</td><td>Gamma</td><td>1.0</td></tr><tr><td>AKOrN</td><td>Use omega</td><td>True</td></tr><tr><td>AKOrN</td><td>Initial omega</td><td>1.0</td></tr><tr><td>AKOrN</td><td>Global omega</td><td>True</td></tr><tr><td>AKOrN</td><td>Learn omega</td><td>True</td></tr><tr><td></td><td>Ensemble size</td><td>1</td></tr><tr><td>AKOrN</td><td>Randomness in forward pass</td><td>True</td></tr><tr><td>PhiNet</td><td>MSE loss ratio</td><td>0.5</td></tr><tr><td>PhiNet</td><td>Orientation loss ratio</td><td>0.1</td></tr><tr><td>PhiNet Pretraining augmentation</td><td>EMA beta</td><td>0.99</td></tr><tr><td>Fine-tuning augmentation</td><td>Strategy</td><td>X-PhiNet</td></tr><tr><td>Evaluation augmentation</td><td>Strategy</td><td>finetuning</td></tr><tr><td>Data loading</td><td>Test transform</td><td>ToTensor()</td></tr><tr><td></td><td>Train shuffle</td><td>True</td></tr><tr><td>Data loading</td><td>Test shuffle</td><td>False</td></tr><tr><td>Data loading</td><td>Num. workers</td><td>1</td></tr><tr><td>Data loading</td><td>Pin memory</td><td>True</td></tr><tr><td>Logging</td><td>W&amp;B</td><td>Enabled</td></tr><tr><td>Logging</td><td>TensorBoard</td><td>Enabled for fine-tuning</td></tr><tr><td>Checkpointing</td><td>Pretraining save target</td><td>Slow encoder if available; otherwise backbone</td></tr><tr><td>Checkpointing</td><td>Fine-tuning save target</td><td>Last and best checkpoints</td></tr></table>

Training compute. Table 9 reports absolute training FLOPs for OPL and representative adversari ally trained reference models. FLOPs for OPL and AKOrN are estimated using a PyTorch FLOP counter. Because OPL does not generate adversarial examples during training, its overall training cost is 358 times lower than the current SOTA models. Results are not directly comparable as the adversarially trained models are evaluated under standard AutoAttack.

CIFAR-10-C robustness. We further evaluate robustness to natural distribution shifts using CIFAR-10-C [17], which contains 15 corruption types at five severity levels. Table 10 reports corruption-wise accuracy averaged over severities. OPL improves the mean corruption accuracy from 80.69% to 82.61%, yielding a +1.92 percentage point gain over AKOrN. Equivalently, OPL reduces the mean corruption error from 19.31% to 17.39%, corresponding to a 9.9% relative error reduction. Notably, the improvement is consistent across all 15 corruption types, including noise, blur, weather, and digital artifacts. These results indicate that the robustness benefits of OPL are not limited to adversarial perturbations, but also transfer to common natural corruptions.

Table 8: Adversarial attack robustness on CIFAR-10 for different predictive learning frameworks.
<table><tr><td rowspan="2">Method</td><td colspan="2">Pretraining: 200 epochs</td><td colspan="2">Pretraining: 400 epochs</td></tr><tr><td>Clean Acc</td><td>Robust Acc</td><td>Clean Acc</td><td>Robust Acc</td></tr><tr><td>BYOL</td><td>82.71</td><td>58.57</td><td>76.05</td><td>46.15</td></tr><tr><td>SimSiam</td><td>84.22</td><td>68.89</td><td>83.50</td><td>74.80</td></tr><tr><td>Phinet</td><td>84.36</td><td>68.79</td><td>84.35</td><td>76.56</td></tr></table>

Table 9: Robustness to adversarial examples by AutoAttack and absolute training-cost. Models marked with <sup>∗</sup> use stochastic forward passes and are evaluated with AutoAttack-rand (EoT). The top two methods are selected from the highestranked methods on https://robustbench.github.io
<table><tr><td>Model</td><td>Reported robust acc.</td><td>Protocol</td><td>Training FLOPs</td></tr><tr><td>Bartoldson et al. (2024) [5]</td><td>73.71</td><td>standard AA</td><td> $1 . 4 3 \times 1 0 ^ { 2 1 }$ </td></tr><tr><td>Amini et al. (2024) [1]</td><td>75.28</td><td>standard AA</td><td> $> 1 0 ^ { 2 1 }$ </td></tr><tr><td>AKOrN* [22]</td><td>64.98</td><td>AA-rand, EoT K = 20</td><td> $1 . 1 9 \times 1 0 ^ { 1 8 }$ </td></tr><tr><td>OPL (ours)*</td><td> ${ \bf 7 6 . 6 3 \pm 0 . 7 6 }$ </td><td>AA-rand, EoT  $K = 2 0$ </td><td> $\mathbf { 3 . 9 9 \times 1 0 ^ { 1 8 } }$ </td></tr></table>

Extended hyperparameter sweeps. To better understand the sensitivity of OPL to optimization and architectural choices, we performed additional hyperparameter sweeps over the use of AKOrN forward-pass randomness, the oscillator dimensionality N, the pretraining duration, the X-PhiNet loss weights, the EMA coefficient $\beta ,$ and pretraining weight decay. Unless otherwise stated, all runs use the same base setting as the main experiments: CIFAR-10, AKOrN backbone, ch = 128, T = 3, L = 3, 400 fine-tuning epochs, and the same evaluation protocol as in the main text.

Table 11 summarizes these exploratory runs. In the table, C-R reports the pair clean accuracy / robust accuracy. Here, robust accuracy denotes the adversarial accuracy under the robustness evaluation protocol used in the main paper. When the final column contains entries such as 0.99 / 1e-5, the first value denotes the EMA coefficient $\beta$ and the second denotes the pretraining weight decay. Thus, 1e-5 refers to pretraining weight decay, not to $\beta .$

Table 10: CIFAR-10-C robustness. Accuracy is averaged over the five severity levels for each corruption type. OPL improves over AKOrN for every corruption.
<table><tr><td>Corruption</td><td>AKOrN</td><td>OPL</td><td>Gain</td></tr><tr><td>Brightness Contrast</td><td>83.81 78.36</td><td>86.20 80.80</td><td>+2.39 +2.44</td></tr><tr><td>Elastic Transform</td><td>79.94</td><td>82.02</td><td>+2.08</td></tr><tr><td>Pixelate</td><td>83.59</td><td>85.90</td><td>+2.31</td></tr><tr><td>JPEG Compression</td><td>83.12</td><td>84.74</td><td>+1.62</td></tr><tr><td>Gaussian Noise</td><td>82.12</td><td>82.86</td><td>+0.74</td></tr><tr><td>Shot Noise</td><td>82.94</td><td>83.88</td><td></td></tr><tr><td>Impulse Noise</td><td>78.16</td><td></td><td>+0.94</td></tr><tr><td></td><td></td><td>78.59</td><td>+0.43</td></tr><tr><td>Defocus Blur</td><td>82.93</td><td>84.79</td><td>+1.86</td></tr><tr><td>Glass Blur</td><td>77.47</td><td>79.30</td><td>+1.83</td></tr><tr><td>Motion Blur</td><td>79.69</td><td>81.37</td><td>+1.68</td></tr><tr><td>Zoom Blur</td><td>83.59</td><td>86.06</td><td>+2.47</td></tr><tr><td>Snow</td><td>78.38</td><td>80.83</td><td>+2.45</td></tr><tr><td>Frost</td><td>80.71</td><td>84.19</td><td>+3.48</td></tr><tr><td></td><td>75.60</td><td>77.59</td><td>+1.99</td></tr><tr><td>Fog</td><td></td><td></td><td></td></tr><tr><td>Mean</td><td>80.69</td><td>82.61</td><td>+1.92</td></tr></table>

Effect of randomness and oscillator dimensionality. We first studied the effect of enabling randomness in the AKOrN forward pass and varying the oscillator dimensionality N. Increasing N from 2 to 4 while keeping randomness enabled led to a severe collapse in robust accuracy despite strong clean accuracy, indicating that this setting is unstable from a robustness perspective. In contrast, disabling randomness with N = 2 yielded a much stronger clean/robust tradeoff than the corresponding randomized setting with orientation loss disabled.

Effect of X-PhiNet loss weights. We also varied the MSE and orientation loss ratios used in X-PhiNet pretraining. Across these runs, moderate MSE weighting produced the strongest overall tradeoff, while adding a nonzero orientation term generally improved robustness relative to the zeroorientation baseline with randomness enabled. However, the results also indicate some sensitivity to the precise weighting, suggesting that the method benefits from careful balancing of representation alignment and orientation regularization.

Effect of pretraining duration. Increasing pretraining from 400 to 700 epochs improved robust accuracy in our sweep while preserving strong clean performance. This suggests that longer selfsupervised training can continue to improve the learned representation for downstream robustness, although the gains are not strictly monotonic across all settings.

Effect of EMA coefficient and pretraining weight decay. We additionally explored the role of the EMA coefficient β and pretraining weight decay. In these experiments, the notation β / wd is used, where wd denotes the pretraining weight decay. We found that introducing a small amount of pretraining weight decay (e.g., 10<sup>−5</sup>) was often beneficial for stabilizing training, while the bestperforming runs continued to use β values close to 1.0. These observations motivated the default setting used in our main experiments.

Table 11: Extended hyperparameter sweeps for X-PhiNet+AKOrN on CIFAR-10. C-R denotes clean accuracy / robust accuracy. Pret. wd denotes pretraining weight decay.
<table><tr><td>Rand</td><td>Pret.</td><td>Fint.</td><td>ch</td><td>N</td><td>T</td><td>L</td><td>C-R (%)</td><td>MSE</td><td>Ori</td><td>β</td><td>Pret. wd</td></tr><tr><td>yes</td><td>400</td><td>400</td><td>128</td><td>4</td><td>3</td><td>3</td><td>88.65 / 0.62</td><td>0.5</td><td>0.0</td><td>0.995</td><td>0</td></tr><tr><td>no</td><td>400</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>85.25 / 76.81</td><td>0.5</td><td>0.0</td><td>0.995</td><td>0</td></tr><tr><td>yes</td><td>400</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>84.60 / 76.08</td><td>0.25</td><td>0.1</td><td>0.99</td><td>0</td></tr><tr><td>yes</td><td>400</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>87.20 / 74.20</td><td>0.5</td><td>0.1</td><td>0.99</td><td>1e-5</td></tr><tr><td>yes</td><td>700</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>86.81 / 77.66</td><td>0.5</td><td>0.1</td><td>0.99</td><td>1e-5</td></tr><tr><td>yes</td><td>400</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>87.91 / 75.85</td><td>0.25</td><td>0.1</td><td>0.99</td><td>1e-5</td></tr><tr><td>yes</td><td>400</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>87.79 / 75.72</td><td>0.1</td><td>0.1</td><td>0.99</td><td>1e-5</td></tr><tr><td>yes</td><td>400</td><td>400</td><td>128</td><td>2</td><td>3</td><td>3</td><td>89.61 / 30.13</td><td>0.25</td><td>0.1</td><td>0.99</td><td>1e-5</td></tr></table>

## D.1 Compute resources

All experiments reported in this paper were conducted on a single NVIDIA RTX A6000 GPU with 48GB of GPU memory. We did not use multi-GPU training, TPUs, or distributed execution. CPU resources were used only for standard data loading, preprocessing, checkpointing, and logging. All reported experiments use CIFAR-scale datasets, so storage requirements are modest and dominated by saved checkpoints and logging files rather than by dataset size. Robustness evaluations were run on the same single-GPU setup. EoT-based attacks are substantially more expensive than clean evaluation because they require multiple stochastic forward/backward passes per attack step.

Table 12 summarizes the compute resources used for the main experiment classes. Training FLOPs are estimated with a PyTorch FLOP counter and include the reported training stages, but exclude logging overhead, checkpoint I/O, preliminary failed runs, and adversarial evaluation cost. The full research project required additional compute for exploratory sweeps over oscillator dimension, randomness, SSL objective, pretraining length, loss weights, EMA coefficient, and weight decay; these exploratory runs are reported separately in Appendix D.

We did not systematically record wall-clock time for every exploratory run. We therefore report hardware type, GPU memory, number of GPUs, training schedules, and estimated FLOPs, but not a complete wall-clock accounting for every experiment.

Table 12: Compute resources used for the main experiment classes. All experiments were run on a single NVIDIA RTX A6000 GPU with 48GB of memory. FLOP estimates exclude logging, checkpoint I/O, preliminary failed runs, and adversarial evaluation cost.
<table><tr><td>Experiment class</td><td>GPUs</td><td>Training schedule</td><td>Estimated training FLOPs</td></tr><tr><td>AKOrN baseline</td><td>1</td><td>Supervised training on CIFAR-10.</td><td> $1 . 1 9 \times 1 0 ^ { 1 8 }$ </td></tr><tr><td>OPL main run</td><td>1</td><td>X-PhiNet self-supervised pretrain- ing followed by supervised fine- tuning.</td><td> $3 . 9 9 \times 1 0 ^ { 1 8 }$ </td></tr><tr><td>SSL objective ablations</td><td>1</td><td>Same AKOrN backbone with alter- Not separately recorded native predictive SSL objectives.</td><td></td></tr><tr><td>Hyperparameter sweeps</td><td>1</td><td>Sweeps over oscillator dimension, randomness, pretraining length, loss weights, EMA coefficient, and weight decay.</td><td>Not separately recorded</td></tr><tr><td>CIFAR-10-C evaluation</td><td>1</td><td>Evaluation only; no additional train- Not applicable ing.</td><td></td></tr><tr><td>Adversarial evaluation</td><td>1</td><td>AutoAttack-rand with EoT, PGD with EoT, Square Attack, and trans- fer attacks.</td><td>Not included in training FLOPs</td></tr></table>

## D.2 Code Availability

For reproducibility, we provide an anonymous repository containing the OPL pipeline and the training and evaluation scripts used in this work: https://anonymous.4open.science/r/OPL-72D2