# COMPACT ROBOT POLICIES NEED FINE-GRAINED VISUAL REPRESENTATIONS

Nanhe Chen<sup>†</sup> Runqiu Yang<sup>†</sup> Jiawei Tang<sup>†</sup> Sichao Liu<sup>‡∗</sup> Yuquan Wang<sup>†∗</sup> † X Square Robot ‡ KTH

## ABSTRACT

Multi-task manipulation policies differ in architecture, scale, and pretrained priors all at once, so published comparisons cannot attribute performance to any single component. We argue that most of it comes from the visual representation, and that parameter scale and generative priors are largely incidental. To test this, we build CoRP (Compressed Representation Policy), a deliberately compact policy (48.9M parameters, no vision-language model and no video-generative prior) that factorizes into a representation extractor and a flow-matching action generator. It reaches 97.0% on LIBERO and 75.78%/73.36% on RoboTwin 2.0 Clean/Randomized, matching systems 40.9–163.6× larger. Holding the action generator fixed, we then vary one extractor property at a time. Pretrained initialization is decisive: a random ViT-S/14 drops to 78.1% and an ImageNet ResNet-34 to 74.5% on LIBERO. Pretraining alone is not enough, as freezing the encoder costs 19.8 points. Compression matters as much: resampling each view to 48 tokens beats passing all patch tokens (97.0% vs 83.2%), and a variational information bottleneck over those tokens is worse than a hard token budget, cutting LIBERO-Goal from 95.8% to 33.0% by suppressing the instruction-dependent token selection the policy relies on. Language conditioning contributes only where the observation leaves the goal ambiguous (LIBERO-Goal: 9.2% to 95.8%), while on RoboTwin 2.0, where observations are unambiguous, removing it slightly improves success. Therefore, we argue that a compact policy works when its representation is pretrained, task-adapted, and compressed. Project page: https://corp-policy.github.io/

## 1 INTRODUCTION

Vision-language-action models (VLAs) have become the dominant paradigm for multi-task robotic manipulation, mapping images and language instructions directly to robot actions (Zitkovich et al., 2023; Kim et al., 2025b; Black et al., 2025b). These systems typically couple a pretrained vision–language model (VLM) with a generative action head, carrying billions of parameters to generalize across tasks (Kim et al., 2025b; Black et al., 2025b). More recently, world-action models (WAMs) go further, adding video-generative priors that jointly predict future observations and robot actions (Ye et al., 2026). Because these systems differ in architecture, scale, pretraining, data, and training recipes, we cannot attribute their success to any single component.

We factor a manipulation policy into representation extraction and conditional action generation:

$$
\begin{array} { r } { \mathbf { z } _ { t } = f _ { \phi } ( \mathbf { o } _ { t } , \mathbf { s } _ { t } , \ell ) , \qquad \mathbf { a } _ { t : t + H - 1 } \sim p _ { \theta } ( \cdot \mid \mathbf { z } _ { t } ) , } \end{array}\tag{1}
$$

where t is the control step, the representation extractor $f _ { \phi }$ (parameters $\phi )$ maps the observation $\mathbf { o } _ { t } ,$ state $\mathbf { s } _ { t } ,$ and instruction ℓ to the conditioning representation $\mathbf { z } _ { t } .$ , and the action generator $p _ { \theta }$ (parameters θ), a DiT-style model (Peebles & Xie, 2023) trained with flow matching (Lipman et al., 2023), samples an action chunk $\mathbf { a } _ { t : t + H - 1 }$ of H steps. The two lines pursue different kinds of generalization: VLAs seek semantic and task generalization through pretrained vision–language backbones and heterogeneous co-training (Kim et al., 2025b; Black et al., 2025a), while WAMs seek temporal and physical generalization through pretrained video priors (Pai et al., 2026; Ye et al., 2026).

Yet both ultimately rely on stronger pretrained representations, and neither isolates how much of its performance the representation alone accounts for. We argue that, in multi-task manipulation, most of this performance comesfrom the visual representation, while parameter scale and generative priors are largely incidental. We therefore ask a narrower question: how much of a policy’s performance depends on its representation, and which of its properties matter? To answer it, we hold the action generator fixed and vary one property of $f _ { \phi }$ at a time: how the visual encoder is initialized, whether it is adapted during training, how many visual tokens reach the action generator, and whether the instruction is available.

We instantiate the factorization as the Compressed Representation Policy (CoRP), a deliberately compact and modular policy. CoRP has only 48.9M parameters and four standard components: a DINOv2-initialized ViT-S/14 visual encoder (Oquab et al., 2024), a T5-Efficient-Tiny language encoder with a lightweight adapter (Raffel et al., 2020; Tay et al., 2022), a learned-query crossattention resampler that compresses each camera view into 48 tokens, and a single-block DiT action generator trained with flow matching. With few interacting parts, each intervention changes exactly one factor, so we can directly attribute its effect. Despite having neither a VLM nor a video-generative prior, CoRP reaches 97.0% average success on LIBERO (Liu et al., 2023), matching VLA and WAM systems 40–160× larger, and 75.8%/73.4% on RoboTwin 2.0 Clean/Randomized (Chen et al., 2026), surpassing π , π , and X-VLA while remaining below video-pretrained WAMs (Tables 1 and 2). This result is our starting point, not our contribution: our controlled interventions locate its source.

Our interventions identify the representation extractor as the main source of this performance and show what makes it fine-grained enough for control: the representation must be pretrained, taskadapted, and compressed. Our study is restricted to closed-set benchmarks; we do not evaluate the open-world semantic and physical generalization that VLAs and WAMs target (Black et al., 2025a; Ye et al., 2026).

We summarize our contributions as follows:

• Compressed Representation Policy (CoRP), a 48.9M-parameter policy without any VLM or video-generative prior that is competitive with systems up to 163× larger on LIBERO and RoboTwin 2.0, and serves as a controlled testbed for isolating the role of representation.

• A controlled ablation study shows that effective compact policies require visual representations that are pretrained, task-adapted, and compressed, and that language helps only when the goal is visually ambiguous.

• Real-robot validation on three bimanual tasks (closed-loop adaptation to object displacement, coordinated two-arm placement, and precision plug insertion), providing qualitative evidence that CoRP, pretrained on our large-scale real-world corpus, transfers to physical hardware.

## 2 RELATED WORK

Generative and Pretrained Policies. Diffusion Policy (Chi et al., 2023) and ACT (Zhao et al., 2023) established generative action heads as strong baselines for visuomotor control, but each model trains on a single task. Octo (Ghosh et al., 2024) scaled such policies across tasks and embodiments by pretraining on large multi-robot datasets, but learns its visual tokenizer from robot data rather than initializing it from a large-scale pretrained vision model, a choice our encoder ablation examines directly (Section 4.3). VLAs instead build the policy on a pretrained vision–language model to inherit its richer visual and semantic representations (Zitkovich et al., 2023; Kim et al., 2025b; Black et al., 2025b;a). World-action models add video-generative priors to target temporal and physical generalization, either alongside a vision–language understanding module (Bi et al., 2026) or in place of it (Yuan et al., 2026). Across both lines, however, the visual pathway changes alongside the language model, the generative prior, and the parameter count, so no single factor can be credited with the gains. In contrast, we hold a small flow-matching action generator fixed and vary only the representation extractor, testing how far a compact policy can go without either prior.

Visual Representation Learning. Supervised pretraining on large labeled datasets established convolutional (He et al., 2016) and patch-based Transformer (Dosovitskiy et al., 2021) backbones as sources of transferable visual features, and our ImageNet ResNet-34 ablation follows this recipe. Self-supervised methods remove the need for labels: the DINO series (Caron et al., 2021; Oquab et al., 2024; Simeoni et al., 2026) yields the fine-grained patch features our resampler consumes.´

Recent work pushes further toward dense geometric structure, for example through masked boundary modeling (Fu et al., 2026). Another line pretrains representations specifically for manipulation from human video, masked image modeling, or sensorimotor prediction (Nair et al., 2023; Radosavovic et al., 2023; Majumdar et al., 2023), and finds that the benefit of pretrained features varies substantially across tasks and evaluation settings (Burns et al., 2025; Hansen et al., 2023). However, these studies mostly evaluate frozen features with single-task or lightweight policy heads, leaving open how initialization, fine-tuning, and token compression jointly shape a multi-task, language-conditioned generative policy. We address this gap by comparing a DINOv2-initialized ViT-S/14 against random and ImageNet-supervised alternatives, frozen and fine-tuned, under an identical action generator (Sections 4.3 and 4.4).

## 3 METHOD

## 3.1 POLICY ARCHITECTURE

CoRP instantiates Eq. (1) in two stages (Figure 1(a)): the representation extractor $f _ { \phi }$ maps the multi-view observation $\mathbf { o } _ { t }$ , proprioceptive state $\mathbf { s } _ { t } ,$ and instruction ℓ to a conditioning sequence $\mathbf { z } _ { t }$ , and the action generator $p _ { \theta }$ , a DiT-style Transformer trained with flow matching, models the distribution of $\mathbf { a } _ { t : t + H - 1 } \mathrm { \ g i v e n \ z } _ { t } .$

Visual Encoder. Let $\mathbf { o } _ { t } ~ = ~ \{ \mathbf { o } _ { t } ^ { v } \} _ { v = 1 } ^ { V }$ denote the V camera views. Each view is encoded independently by a DINOv2-initialized ViT-S/14 (Oquab et al., 2024) with weights shared across views, yielding $P \ = \ 2 5 6$ patch tokens at 224 × 224 resolution. We discard the class token, pass the patch tokens to the

(b)  
![](images/573d4797053990d1afef4d0dcdbac9bc770d72b357199f2fb41bd6bb174b307d.jpg)

![](images/3f6ac4d7e642785ac2b11d6220ff231302400642c3db3b2ac1a4928d854f4154.jpg)  
LIBERO SpatialGoalAverage ObjectLong

![](images/ce5d4414a2bd57f39a9f4f15f22d1583d8409b28313a7bf76894972efbed4e76.jpg)

![](images/7abdf460e0de1731cb0ebe24d609d50ab368d7e5d8382376e01dabdcc32d4849.jpg)  
Figure 1: (a) $\mathbf { C o R P ^ { * } s }$ architecture. The representation extractor (DINOv2, T5, resampler, and state MLP) outputs $\mathbf { z } _ { t } .$ which conditions the action generator to produce an action chunk. (b) Performance overview. CoRP is competitive with much larger systems while using only 0.049B parameters.

resampler, and fine-tune the encoder jointly with the policy.

Language Encoder. We encode the instruction ℓ with a T5-Efficient-Tiny encoder (Raffel et al., 2020; Tay et al., 2022), chosen because the instructions are short and drawn from a closed set, and fine-tune it jointly with the policy. A LayerNorm and a linear adapter map its outputs to the policy width $d ,$ giving $\dot { \mathbf { u } } \in \mathbb { R } ^ { M \times d }$ , where M is the padded instruction length. A masked mean over the non-padding tokens gives the pooled embedding $\bar { \mathbf { u } } \in \mathbb { R } ^ { d }$ , which conditions both the resampler and the action generator.

Vision–Language Fusion. Because the P patch tokens per view are far more than the action generator needs, we compress them into $K \ll P$ tokens with a learned-query cross-attention resampler. We linearly project the patch tokens to the policy width, giving $\dot { \mathbf { X } _ { t } ^ { v } } \in \mathbb { R } ^ { P \times d }$ , and give each view its own trainable queries $\mathbf { Q } ^ { v } \in \mathbb { R } ^ { K \times d } ( K \stackrel {  } { = } 4 8$ by default). Before cross-attention, u¯ modulates the queries through a FiLM-style scale and shift (Perez et al., 2018):

$$
\widetilde { \mathbf { Q } } ^ { v } = \mathbf { Q } ^ { v } \odot \big ( 1 + \gamma ( \bar { \mathbf { u } } ) \big ) + \eta ( \bar { \mathbf { u } } ) , \qquad \mathbf { r } _ { t } ^ { v } = \mathrm { C r o s s A t t n } \big ( \widetilde { \mathbf { Q } } ^ { v } , \mathbf { X } _ { t } ^ { v } \big ) ,\tag{2}
$$

where CrossAttn is multi-head cross-attention (queries first, then keys and values) followed by LayerNorm, $\mathbf { r } _ { t } ^ { v } \in \mathbb { R } ^ { K \times d }$ are the K resampled tokens, and $\gamma , \eta : \mathbb { R } ^ { d } \overset { \bullet } {  } \mathbb { R } ^ { d }$ are learned mappings whose outputs are broadcast across the K queries.

A learned M $\boldsymbol { \mathrm { ~ P ~ } } _ { \boldsymbol { \mathrm { { c } } } _ { s } }$ maps the proprioceptive state to a single token $\widetilde { \mathbf { s } } _ { t } = e _ { s } ( \mathbf { s } _ { t } ) \in \mathbb { R } ^ { d }$ . Concatenating the resampled visual tokens in view order, the language tokens, and the state token gives $\mathbf { z } _ { t } ~ \in$ ${ \mathbb R } ^ { ( V K + M + 1 ) \times d }$

$$
\mathbf { z } _ { t } = f _ { \phi } ( \mathbf { o } _ { t } , \mathbf { s } _ { t } , \ell ) = \left[ \mathbf { r } _ { t } ^ { 1 } ; \ldots ; \mathbf { r } _ { t } ^ { V } ; \mathbf { u } ; \widetilde { \mathbf { s } } _ { t } \right] .\tag{3}
$$

The first V K tokens form the visual prefix, the only tokens of $\mathbf { z } _ { t }$ that carry image information at the input to the action generator. Keeping K small therefore forces all visual information through a fixed-capacity bottleneck, which we test in Section 4.6.

Action Generator. The action generator is an L-block, width-d DiT-style Transformer $( L = 1$ by default). It linearly projects the noisy action chunk into H action tokens with learned positional embeddings. The interpolation time τ is embedded with sinusoidal encoding and an MLP, then added to every action token. $\mathbf { z } _ { t }$ is prepended to the action tokens as a prefix; the attention mask lets prefix tokens attend only to the prefix and action tokens attend to the prefix and all action tokens, keeping $\mathbf { z } _ { t }$ independent of the noisy actions. Each block also applies an adaptive LayerNorm (adaLN) (Peebles & Xie, 2023) with scale and shift computed from u¯, so the instruction conditions every layer rather than only the resampler. The DiT then predicts a velocity for each action token (Section 3.2).

## 3.2 TRAINING OBJECTIVE

Flow-Matching Loss. Let $( \mathbf { o } _ { t } , \mathbf { s } _ { t } , \ell , \mathbf { a } ) \sim \mathcal { D }$ be a training tuple from the demonstration dataset D, where $\mathbf { a } \equiv \mathbf { a } _ { t : t + H - 1 } \in \mathbb { R } ^ { H \times D _ { a } }$ is the ground-truth action chunk and $D _ { a }$ is the per-step action dimension. We independently sample noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ of the same shape as a and an interpolation time $\tau \sim \mathrm { B e t a } ( 1 . 0 , 1 . 5 )$ . Here $\tau \in [ 0 , 1 ]$ , distinct from the control step $t ,$ runs from noise $( \tau = 0 )$ to data $( \tau = 1 )$ . We form the point at time τ on the straight path and its constant target velocity:

$$
{ \bf a } ^ { \tau } = ( 1 - \tau ) \epsilon + \tau { \bf a } , \qquad { \bf v } ^ { \star } = { \bf a } - \epsilon .\tag{4}
$$

The action generator learns to predict this velocity, $\mathbf { v } _ { \theta } ,$ , by minimizing

$$
\mathcal { L } _ { \mathrm { f m } } = \mathbb { E } \bigg [ \frac { 1 } { H D _ { a } } \left\| \mathbf { v } _ { \theta } \big ( \mathbf { a } ^ { \tau } , \tau \mid \mathbf { z } _ { t } \big ) - \mathbf { v } ^ { \star } \right\| _ { F } ^ { 2 } \bigg ] ,\tag{5}
$$

where the expectation is over training tuples, ϵ, and τ, and we exclude chunk positions padded beyond the end of an episode from both the norm and the normalization (Appendix A.3).

At inference, we draw $\mathbf { a } ^ { \tau _ { 0 } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and integrate with $N = 1 0$ Euler steps on the uniform grid $\tau _ { n } = n / N ;$

$$
\mathbf { a } ^ { \tau _ { n + 1 } } = \mathbf { a } ^ { \tau _ { n } } + \frac { 1 } { N } \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { a } ^ { \tau _ { n } } , \tau _ { n } \mid \mathbf { z } _ { t } ) , \qquad n = 0 , \ldots , N - 1 .\tag{6}
$$

The final sample $\mathbf { a } ^ { \tau _ { N } }$ is the predicted chunk $\hat { \mathbf { a } } _ { t : t + H - 1 }$ , whose H actions are all executed before replanning $( \bar { H _ { \mathrm { ~ } } } = 1 2 $ on LIBERO and $H = 5 0$ on RoboTwin $2 . 0 ;$ Appendix A.3).

Task-Classification Loss. The flow-matching loss supervises only the actions, so it shapes the conditioning representation only indirectly, through gradients that reach $f _ { \phi }$ via the action generator. We therefore add an auxiliary task-classification loss that directly encourages the language-conditioned visual prefix to retain task-relevant information. Let $c \in \{ 1 , \ldots , C \}$ be the label of the current task and $\mathbf { h } ^ { ( j ) } \in \mathbb { R } ^ { V K \times d }$ the visual-prefix hidden states after the j-th DiT block, with $j = 0$ denoting the input prefix. For each depth $j$ in a set ${ \mathcal { I } } ,$ a task head $g _ { \psi }$ (an MLP shared across depths) applies LayerNorm to each token, averages over the $V K$ tokens, and produces logits

$$
\pmb { \xi } ^ { ( j ) } = g _ { \psi } \left( \mathrm { m e a n } \big ( \mathrm { L a y e r N o r m } ( \mathbf { h } ^ { ( j ) } ) \big ) \right) \in \mathbb { R } ^ { C } ,\tag{7}
$$

and $\mathcal { L } _ { \mathrm { t a s k } }$ averages the softmax cross-entropy CE over these depths:

$$
\mathcal { L } _ { \mathrm { t a s k } } = \frac { 1 } { \left| \mathcal { T } \right| } \sum _ { j \in \mathcal { I } } \mathrm { C E } \big ( \pmb { \xi } ^ { ( j ) } , c \big ) .\tag{8}
$$

By default $( L = 1 ) , \mathcal { T } = \{ 1 \}$ . The task head is used only during training: neither c nor its logits condition action generation at inference.

Overall Objective. The overall training loss weights the two terms by $\lambda \ge 0 ( \lambda = 0 . 0 1$ by default):

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f m } } + \lambda \mathcal { L } _ { \mathrm { t a s k } } . } \end{array}\tag{9}
$$

We minimize L jointly over $\phi , \theta ,$ and ψ with a single AdamW optimizer (Loshchilov & Hutter, 2019) (Appendix A.3), training all modules end-to-end, including the pretrained visual and language encoders (Section 4.4 studies freezing the visual encoder).

## 4 EXPERIMENTS

Section 4.2 shows CoRP is competitive with much larger systems, Sections 4.3–4.6 isolate why by varying one property of $f _ { \phi }$ at a time, and Section 4.7 evaluates CoRP on real robots.

## 4.1 SETUP

Benchmarks. We evaluate on two simulation benchmarks, plus a precision peg-insertion task.

LIBERO (Liu et al., 2023) is a single-arm benchmark with four 10-task suites: Spatial, Object, Goal, and Long. Its evaluation follows the demonstration distribution, so we use it for controlled comparisons (Sections 4.3–4.6), where differences reflect representation rather than robustness.

RoboTwin 2.0 (Chen et al., 2026) is a 50-task bimanual benchmark (Aloha-AgileX) with Clean and Randomized settings; following Bi et al. (2026), we train on 2,500 clean and 25,000 randomized demonstrations, so Randomized tests in-distribution visual variation rather than zero-shot transfer. We use it to test whether our findings extend to bimanual control.

Implementation. Unless stated otherwise, every variant uses the default configuration of Section 3 $( K = 4 8 , L = 1 , d = 7 6 8 , \lambda = 0 . 0 1 )$ ) and the same training recipe, changing only the factor under study. Appendix A.3 lists the detailed training schedules and compute, using random seed 42 during training.

Table 1: Success rates (%) on LIBERO.
<table><tr><td>Method</td><td>Params (B)</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td colspan="7">With Vision-Language Model</td></tr><tr><td>SmolVLA (Shukor et al., 2025)</td><td>2.25</td><td>93.0</td><td>94.0</td><td>91.0</td><td>77.0</td><td>88.8</td></tr><tr><td>π0 (Black et al., 2025b)</td><td>3.3</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.1</td></tr><tr><td>π0.5 (Black et al., 2025a)</td><td>3.3</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>OpenVLA (Kim et al., 2025b)</td><td>7.0</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025a)</td><td>7.0</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td></tr><tr><td colspan="7">With Video Generation Model</td></tr><tr><td>Cosmos-Policy (Kim et al., 2026a)</td><td>2.0</td><td>98.1</td><td>100.0</td><td>98.2</td><td>97.6</td><td>98.5</td></tr><tr><td>Fast-WAM-Joint (Yuan et al., 2026)</td><td>6.0</td><td>99.6</td><td>99.4</td><td>98.2</td><td>96.8</td><td>98.5</td></tr><tr><td>Motus (Bi et al., 2026)</td><td>8.0</td><td>96.8</td><td>99.8</td><td>96.6</td><td>97.6</td><td>97.7</td></tr><tr><td colspan="7">Without Both</td></tr><tr><td>ACT (Zhao et al., 2023)</td><td>0.084</td><td>82.0</td><td>78.8</td><td>66.1</td><td>44.0</td><td>67.7</td></tr><tr><td>Diffusion Policy (Chi et al., 2023)</td><td>0.157</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td></tr><tr><td>Octo (Ghosh et al., 2024)</td><td>0.093</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>75.1</td></tr><tr><td>Prog VLA (Kim et al., 2026b)</td><td>0.109</td><td>87.6</td><td>96.0</td><td>92.0</td><td>88.6</td><td>91.1</td></tr><tr><td>Ours (L = 1, d = 180)</td><td>0.037</td><td>97.6</td><td>99.4</td><td>93.2</td><td>85.4</td><td>93.9</td></tr><tr><td>Ours (L = 1, d = 768, default)</td><td>0.049</td><td>98.2</td><td>99.4</td><td>95.8</td><td>94.4</td><td>97.0</td></tr><tr><td>Ours (L = 3, d = 768)</td><td>0.063</td><td>98.2</td><td>99.6</td><td>98.8</td><td>97.0</td><td>98.4</td></tr></table>

Metrics. Our primary metric is task success rate (%), following each benchmark’s protocol (Liu et al., 2023; Chen et al., 2026): 50 trials per task on LIBERO, reported per suite and averaged over 40 tasks; 100 trials per task on RoboTwin 2.0, averaged over 50 tasks per setting (per-task results in Table 8). We also report task-head accuracy (%, Eq. (7)) on each evaluation episode’s first frame, measuring how well the visual prefix identifies the task. To measure action-relevant information without closed-loop execution, we freeze $f _ { \phi }$ and train a lightweight probe to predict action chunks from the visual prefix (Radosavovic et al., 2023; Nikulin et al., 2025), reporting its NMAE (per-dimension, normalized by the 1st–99th percentile range; lower is better) and $\scriptstyle { \dot { R } } ^ { 2 }$ (higher is better) on a held-out split (Eq. (12), details in A.1).

## 4.2 CAN A COMPACT POLICY MATCH MUCH LARGER SYSTEMS?

The answer is YES on LIBERO, and partly on RoboTwin 2.0.

Tables 1 and 2 compare CoRP with VLA, WAM, and compact policies that use neither prior; baseline results are taken from (Chen et al., 2026; Giga-World Team et al., 2026; Bi et al., 2026; Shukor et al., 2025; Kim et al., 2025b;a; Yuan et al., 2026; Kim et al., 2026b;a) (Appendix A.4). On LIBERO, the default CoRP (0.049B) reaches 97.0%, on par with $\pi _ { 0 . 5 }$ (96.9%, 3.3B) and OpenVLA-OFT (97.1%, 7.0B), and 5.9 points above ProgVLA (91.1%), the strongest prior-free compact policy. Performance also increases with capacity, from 93.9% (d = 180, 0.037B) to 98.4% $( L \ = \ 3 ; \ 0 . 0 6 3 \mathbf { B } )$ , matching Fast-WAM-Joint and Cosmos-Policy

Table 2: Success rates (%) on RoboTwin 2.0.
<table><tr><td colspan="3">Method Params (B) Clean Rand.</td></tr><tr><td>With Vision-Language Model</td><td></td><td></td></tr><tr><td>X-VLA (Zheng et al., 2026)</td><td>0.9</td><td>72.9 72.8 46.4</td></tr><tr><td> $\pi _ { 0 }$  (Black et al., 2025b)  $\pi _ { 0 . 5 }$  (Black et al., 2025a)</td><td>3.3 3.3 43.0</td><td>16.3 43.8</td></tr><tr><td>With Video Generation Model</td><td></td><td></td></tr><tr><td>Motus (Bi et al., 2026)</td><td>8.0</td><td>88.7 87.0</td></tr><tr><td>w/o pretraining GigaWorld-</td><td>8.0 77.6</td><td>77.0</td></tr><tr><td>(GigaWorld Team et al., 2026)</td><td>11.4 86.4</td><td>85.0</td></tr><tr><td>Without Both</td><td></td><td></td></tr><tr><td>ACT (Zhao et al., 2023)</td><td>0.084 29.7</td><td>1.7</td></tr><tr><td>Diffusion Policy (Chi et al., 2023)</td><td>0.157 28.0</td><td>0.6</td></tr><tr><td>Ours</td><td>0.049 75.8</td><td>73.4</td></tr></table>

(98.5%); we nonetheless use the one-block default for ablations to keep them cheap and comparable.

On RoboTwin 2.0 (Clean/Randomized), CoRP (75.8%/73.4%) outperforms X-VLA (72.9%/72.8%) and $\pi _ { 0 . 5 }$ (43.0%/43.8%) and approaches Motus without video pretraining (77.6%/77.0%) with about 160× fewer parameters, but stays below pretrained Motus (88.7%/87.0%) and GigaWorld-Policy (86.4%/85.0%). Notably, Diffusion Policy and ACT, whose visual encoders are learned without a large-scale pretrained vision model, collapse in the Randomized setting (0.6% and 1.7%), wherea CoRP loses only 2.4 points, an early sign that the visual representation drives robustness.

## 4.3 DOES A PRETRAINED VISUAL ENCODER HELP?

## The answer is YES: the DINOv2-initialized ViT-S/14 outperforms every other encoder variant by at least 14.9 points in average LIBERO success.

To separate the effect of pretraining from that of architecture, we cross two initializations, pretrained or random, with two parameter-matched architectures in a $2 \times 2$ design. The ViT-S/14 (Dosovitskiy et al., 2021) (22.1M, default) is pretrained from a DINOv2 checkpoint (Oquab et al., 2024) and the ResNet-34 (21.5M) from ImageNet-supervised weights (He et al., 2016; Deng et al., 2009); the ResNet’s final feature map provides 49 spatial tokens per view to the resampler.

Table 3: Ablation study on visual encoder initialization and architecture.
<table><tr><td>Method</td><td>Spatial (%)</td><td>Object (%)</td><td>Goal (%)</td><td>Long (%)</td><td>Average (%)</td><td>NMAE</td><td> $R ^ { 2 }$ </td></tr><tr><td>Random ResNet-34</td><td>91.2</td><td>97.8</td><td>72.8</td><td>66.6</td><td>82.1</td><td>0.12</td><td>0.54</td></tr><tr><td>Pretrained ResNet-34</td><td>85.4</td><td>98.8</td><td>37.0</td><td>76.8</td><td>74.5</td><td>0.09</td><td>0.73</td></tr><tr><td>Random ViT-S/14</td><td>84.4</td><td>95.6</td><td>86.0</td><td>46.4</td><td>78.1</td><td>0.11</td><td>0.59</td></tr><tr><td>Pretrained ViT-S/14</td><td>98.2</td><td>99.4</td><td>95.8</td><td>94.4</td><td>97.0</td><td>0.07</td><td>0.81</td></tr></table>

Table 3 shows that the pretrained ViT achieves the highest success rate on every suite (97.0% on average) and the most action-decodable visual prefix (NMAE 0.07, $R ^ { 2 } = 0 . 8 1 )$ . Neither factor alone explains this gain: with random initialization, the ViT is no better than the ResNet (78.1% vs. 82.1%), and ImageNet pretraining even lowers the ResNet’s success (74.5% vs. 82.1%). The gain therefore requires combining the two, consistent with DINOv2’s patch-level pretraining providing fine-grained spatial features for the resampler to select from. Separating this from the benefit of any pretrained ViT would require a supervised-pretrained ViT control, which we leave to future work. All four variants identify the task with 100% accuracy, yet the pretrained ResNet reaches only a 37.0% success rate on LIBERO-Goal: these policies know which task to perform but lack the fine-grained visual information needed to do so.

## 4.4 IS PRETRAINING ENOUGH WITHOUT ADAPTATION?

The answer is NO: freezing the pretrained encoder lowers average LIBERO success from 97.0% to 77.2%. If DINOv2 features were already sufficient for control, freezing them should cost little. We test this by comparing two otherwise identical policies:

• Frozen: the DINOv2 encoder is fixed, and only the resampler, language adapter, and action generator are trained.

• Fine-tuned (default): all modules, including the encoder, are trained jointly.

The frozen policy (77.2%) performs no better than a randomly initialized ViT trained end-toend (78.1%, Table 3), so pretraining pays off only when the features adapt to the task. The probe shows why: freezing reduces $R ^ { 2 }$ from 0.81 to 0.24 and doubles NMAE (0.07 to 0.16), indicating that frozen features retain much less action-relevant detail, and t-SNE analysis (van der Maaten & Hinton, 2008) (Figure 2) shows that, after fine-tuning, representations at similar episode stages cluster more closely. They are also more clearly ordered from early to late stages.

![](images/12eb51170d9c34f9c9843a937fd6273fc67a8161c01c19a5b4316959c61f716b.jpg)  
Figure 2: t-SNE of visual-prefix features with and without fine-tuning. Color indicates episode progress (purple: early, yellow: late). Fine-tuning groups features by stage and yields a clearer early-to-late progression.

## 4.5 DO WE NEED LANGUAGE CONDITIONING?

The answer is: it depends. Language is needed only when the observation does not uniquely specify the goal κ, i.e., which task is consistent with the current scene.

We compare the default policy with one that removes the language encoder and tokens u, with FiLM and adaLN receiving a learned constant conditioning in place of u¯.

Table 4: Ablation study on whether to fine-tune the visual encoder.
<table><tr><td>Method</td><td>Average (%)</td><td>NMAE</td><td> $R ^ { 2 }$ </td></tr><tr><td>Frozen</td><td>77.2</td><td>0.16</td><td>0.24</td></tr><tr><td>Fine-tuned</td><td>97.0</td><td>0.07</td><td>0.81</td></tr></table>

Removing language is catastrophic only on LIBERO-Goal, where success drops from 95.8% to 9.2%, while the other suites lose at most 11.2 points (Ta-

Table 5: Language-conditioning ablation on LIBERO (%).
<table><tr><td></td><td colspan="2">Spatial</td><td colspan="2">Object</td><td colspan="2">Goal</td><td colspan="2">Long</td><td colspan="2">Average</td></tr><tr><td>Method</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td></tr><tr><td>Without Lang.</td><td>90.4</td><td>87.0</td><td>100.0</td><td>100.0</td><td>8.6</td><td>9.2</td><td>89.0</td><td>83.4</td><td>72.0</td><td>69.9</td></tr><tr><td>With Lang.</td><td>100.0</td><td>98.2</td><td>100.0</td><td>99.4</td><td>100.0</td><td>95.8</td><td>100.0</td><td>94.4</td><td>100.0</td><td>97.0</td></tr></table>

ble 5). Without the instruction, the policy can only model the goal-marginal distribution $p ( \mathbf { a } _ { t }$ $\begin{array} { r } { \mathbf { o } _ { t } , \mathbf { s } _ { t } \big ) = \sum _ { { \boldsymbol { \kappa } } } p \big ( \mathbf { a } _ { t } \mid \mathbf { o } _ { t } , \mathbf { s } _ { t } , { \boldsymbol { \kappa } } \big ) p \big ( { \boldsymbol { \kappa } } \mid \mathbf { o } _ { t } , \mathbf { s } _ { t } \big ) ; } \end{array}$ when the scene is shared across goals, as in LIBERO-Goal, $p ( \boldsymbol { \kappa } \mid \mathbf { o } _ { t } , \mathbf { s } _ { t } )$ is nearly uniform, and the policy effectively commits to a random goal.

The no-language policy’s task head confirms this: its firstframe accuracy closely tracks its success rate on every suite (Table $5 ) ,$ and on Goal both are near the 10% chance level for a 10-task suite (8.6% and 9.2%). With language, the head reaches 100.0% on every suite, since the instruction, injected into the visual prefix through FiLM, resolves the goal the scene leaves ambiguous. The t-SNE analysis (Appendix Figure 7) illustrates that, without language conditioning, the Goal tasks collapse into a single cluster, conditioning, the Goal tasks collapse into a single cluster, whereas other tasks remain separable.

Table 6: Language-conditioning ablation on RoboTwin 2.0 (%).
<table><tr><td></td><td colspan="2">Clean</td><td colspan="2">Randomized</td></tr><tr><td>Method</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td></tr><tr><td>Without Lang.</td><td>99.96</td><td>78.50</td><td>99.92</td><td>76.06</td></tr><tr><td>With Lang.</td><td></td><td></td><td>99.78 75.78 99.58</td><td>373.36</td></tr></table>

Conversely, on RoboTwin 2.0, where each scene identifies its task, removing language raises success by about 2.7 points in both settings, with task accuracy near 100% either way (Table 6); here the instruction adds no information. Language thus acts as a disambiguator rather than a general source of task knowledge: it is essential when the scene leaves the goal ambiguous, and unnecessary or even slightly harmful when the scene determines it.

![](images/5c4a4085cde039879afb4e16322aca8759b28f4a18a19c84542488b4d3ff55f5.jpg)

## 4.6 DOES THE INFORMATION BOTTLENECK HELP?

The answer is yes, but not every bottleneck formulation helps: a hard, instruction-aware token budget raises average LIBERO success from 83.2% to 97.0%, whereas a variational bottleneck lowers success with or without it. The information bottleneck (IB) principle (Tishby et al., 1999) seeks a representation $Z$ of the input X that is maximally informative about a target $Y$ while discarding the rest of X, i.e., ma $\mathrm { x } _ { p ( Z | X ) } I ( Z ; Y ) - \beta _ { \mathrm { I B } } I ( Z ; \dot { X } )$ with $\beta _ { \mathrm { I B } } > 0$ ; here, $\breve { X }$ is the visual input, Z the visual prefix, and $\dot { Y }$ the action chunk. A common tractable surrogate is the deep variational information bottleneck (VIB) (Alemi et al., 2017), which adds $\beta _ { \mathrm { I B } } D _ { \mathrm { K L } } ( q ( Z \mid X ) \parallel \hat { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) )$ ) to a prediction loss, where $q ( Z \mid X )$ is a Gaussian encoder. In our VIB variants, we add this KL term on each visual token to our training loss (Eq. (9)), with $\beta _ { \mathrm { I B } } = 4 \times 1 0 ^ { - 8 }$ (Appendix A.1). However, the KL penalty is task-agnostic: it compresses every token toward the same prior, regardless of which visual details the current instruction requires. Our resampler instead imposes a hard, instruction-aware bottleneck: capacity is limited to $K \ll P$ tokens but allocated according to the instruction through the FiLM-modulated queries. We cross the two choices in a $2 \times 2$ design: with or without the resampler, and with or without VIB. Without the resampler (All Tokens), all $P \stackrel { - } { = } 2 5 6$ projected patch tokens per view enter the visual prefix. In this case, u¯ directly modulates every patch token.

Table 7: Ablation study on information bottleneck. (Acc.: task-head accuracy; S.R.: success rate.)
<table><tr><td></td><td colspan="2">Spatial</td><td colspan="2">Object</td><td colspan="2">Goal</td><td colspan="2">Long</td><td colspan="2">Average</td></tr><tr><td>Method</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td><td>Acc.</td><td>S.R.</td></tr><tr><td>All Tokens</td><td>93.8</td><td>89.6</td><td>100.0</td><td>99.6</td><td>91.4</td><td>55.8</td><td>100.0</td><td>87.6</td><td>96.3</td><td>83.2</td></tr><tr><td>All Tokens + VIB</td><td>99.8</td><td>87.8</td><td>100.0</td><td>99.4</td><td>99.9</td><td>33.6</td><td>100.0</td><td>90.2</td><td>99.9</td><td>77.8</td></tr><tr><td>Partial Tokens</td><td>100.0</td><td>98.2</td><td>100.0</td><td>99.4</td><td>100.0</td><td>95.8</td><td>100.0</td><td>94.4</td><td>100.0</td><td>97.0</td></tr><tr><td>Partial Tokens + VIB</td><td>91.6</td><td>87.8</td><td>99.9</td><td>99.2</td><td>69.6</td><td>33.0</td><td>95.5</td><td>83.6</td><td>89.2</td><td>75.9</td></tr></table>

The gain over All Tokens is concentrated on LIBERO-Goal (55.8% to 95.8%, Table 7), the suite where the instruction alone specifies the goal. To quantify how strongly the visual prefix depends on the instruction, we keep the observation fixed, swap the instruction for another task’s, and measure the prefix’s relative change $( \Delta _ { p r e f i x } ,$ Appendix A.1): 49.7% for Resampled but 5.4% for All Tokens. Thus, the resampler selects instruction-specific visual information, whereas passing all tokens leaves the visual prefix nearly instruction-invariant and forces the action generator to resolve the goal on its own.

Adding VIB undoes this selection: on the resampled prefix, it reduces the instruction-induced change from 49.7% to 13.6% and Goal success from 95.8% to 33.0%; on All Tokens, it lowers average success from 83.2% to 77.8%. Notably, All Tokens + VIB still identifies the Goal task with 99.9% accuracy yet succeeds only 33.6% of the time: the KL penalty preserves which task to perform but suppresses the instruction-specific visual detail needed to perform it. A tight, instruction-aware budget therefore outperforms a soft, task-agnostic one.

## 4.7 CAN CORP WORK ON REAL ROBOTS?

Figure 3: Dynamic adjustment in the pickcup task. The robot adapts to cup and plate displacements during execution.

Finally, we test whether CoRP transfers to physical hardware on three bimanual tasks that probe closed-loop adaptation, arm coordination, and precision. Unlike in simulation, we pretrain the unchanged 48.9M-parameter CoRP on more than 10,000 hours of real-world data covering 5,887 tasks on our dual-arm robots, then fine-tune it on 10 tasks (Appendix A.5). We evaluate three of these tasks over 10 trials each with randomized initial object placements. The policy uses three cameras, one on the head and one on each arm, and runs synchronously at about 7 ms per chunk.

Pick cup. The robot grasps a cup by its thin wall and places it on a plate, both at random positions on the table. CoRP succeeds in 12/12 trials. To test closed-loop adaptation, we manually move the cup

or plate during execution in 8 trials. As Figure 3 shows, the policy re-targets the displaced object within the next action chunk and completes the task in 8/8 perturbed trials. This shows that CoRP reacts to new observations rather than replaying a memorized trajectory.

![](images/d904f1d1e8e652f250ea317e54e5555ae458f38bff3bc1eb1628ef6f4c44a8c3.jpg)

Place spoon. The robot picks up a spoon from the table and places it in a bowl. Because the bowl may start outside the spoon-holding arm’s reach, the other arm must first move it

Figure 4: Bimanual coordination in the place-spoon task, shown from three initial configurations.

into reach. Across 10 initial configurations (three shown in Figure 4), CoRP succeeds 9/10.

Plug insertion. The robot unplugs a charger from the right socket, transfers it, and inserts it into the left socket (Figure 5). Successful insertion requires aligning the pins with the socket openings in both position and orientation and maintaining that alignment as the pins enter. It thus tests precision manipulation under geometric constraints, complementing the simulated peg-insertion task in Appendix A.3 (8.0 mm peg, 8.1 mm hole), where CoRP reaches 84% success. CoRP completes the insertion in 2 trials. Since the architecture is unchanged from simulation, these results suggest that the same compact, representationcentered design scales to real-world pretraining and deployment.

![](images/68637101165f32f6bd37cc6a9d08fd1198269aeff21ff142b24c985d00776a8a.jpg)  
Figure 5: Precision manipulation in the plug-insertion task. The robot unplugs a charger from the right socket and inserts it into the left socket.

## 5 CONCLUSION AND LIMITATIONS

We presented CoRP, a 48.9M-parameter manipulation policy without a VLM or video-generative prior, and used it as a controlled testbed to ask where a multi-task policy’s performance comes from. By holding the action generator fixed and varying one property of the representation extractor at a time, we find that a compact policy works when its visual representation is pretrained, task-adapted, and compressed: removing any one of these properties costs 14–20 points on LIBERO, while language matters only when the scene leaves the goal ambiguous. With these properties, CoRP is competitive with systems up to 163× larger on LIBERO and RoboTwin 2.0 and transfers unchanged to real bimanual robots. Across our ablations, the weaker variants still identify the task but fail to perform it, suggesting that compact policies lack not task knowledge but fine-grained, instruction-specific visual detail. Our study has several limitations that are worth exploring in future work.

Closed-set evaluation. Our benchmarks measure in-distribution control on closed task sets; we do not evaluate the open-world semantic and physical generalization that VLAs and WAMs target. Whether pretrained, task-adapted, and compressed representations remain sufficient in that regime, or must be combined with such priors, is an open question.

Attribution of the pretraining benefit. Our encoder ablation lacks a supervised-pretrained ViT, so it cannot separate the benefit of DINOv2’s patch-level pretraining from that of any pretrained ViT; adding this control is the next direct step.

Real-world evidence. Our real-robot experiments cover three tasks, include no real-robot baselines, and follow large-scale pretraining; they show that CoRP transfers to hardware but do not isolate the role of the representation as our simulation ablations do. Repeating the controlled interventions on real hardware would close this gap.

## AI USE STATEMENT

In compliance with the ICLR 2027 policy, we acknowledge using large language models (LLMs) as assistive tools in preparing this paper. We used LLMs to polish the writing and help develop the software, including debugging and generating scripts for data preprocessing and visualization. The authors carefully reviewed and edited all final text to ensure it accurately reflects our research and contributors. The authors also developed the core algorithmic implementation of our proposed method. Overall, the LLMs served mainly as a productivity and quality-enhancement tool.

## REPRODUCIBILITY STATEMENT

To support reproducibility, Section 3 describes CoRP’s architecture and training objectives, while Section. 4 specifies the benchmarks and evaluation metrics. Sections 4.3–4.6 present controlled ablations on visual encoder initialization, fine-tuning, language conditioning, and representation compression. Our simulation experiments use the publicly available LIBERO and RoboTwin 2.0 datasets. The real-world training datasets are proprietary and cannot be released due to commercial restrictions, which limits independent reproduction of the real-world results. We will release the source code and configuration files for the simulation experiments with the camera-ready version to facilitate their reproduction.

## REFERENCES

Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information bottleneck. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=HyxQzBceg.

Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 35101–35113, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Bi\_ Motus\_A\_Unified\_Latent\_Action\_World\_Model\_CVPR\_2026\_paper.html.

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π<sub>0.5</sub>: A vision-languageaction model with open-world generalization. In Proceedings of the 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 17–40. PMLR, 2025a. URL https://proceedings.mlr.press/v305/black25a.html.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π : A vision-language-action flow model for general robot control. In Proceedings ofRobotics: Science and Systems, 2025b. doi: 10.15607/RSS.2025.XXI.010. URL https://roboticsproceedings.org/rss21/ p010.html.

Kaylee Burns, Zach Witzel, Jubayer Ibn Hamid, Tianhe Yu, Chelsea Finn, and Karol Hausman. What makes pre-trained visual representations successful for robust manipulation? In Proceedings ofthe 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 4525–4545. PMLR, 2025. URL https://proceedings.mlr.press/v270/burns25a. html.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, 2021.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2506.18088.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin C. M. Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems, 2023. doi: 10.15607/RSS.2023.XIX.026. URL https: //www.roboticsproceedings.org/rss19/p026.html.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848. URL https://www. image-net.org/static\_files/papers/imagenet\_cvpr09.pdf.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=YicbFdNTTy.

Zelin Fu, Bin Tan, Changjiang Sun, Shaohui Liu, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Vision pretraining for dense spatial perception. arXiv preprint arXiv:2607.05247, 2026.

Dibya Ghosh, Homer Rich Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, et al. Octo: An open-source generalist robot policy. In Proceedings ofRobotics: Science and Systems, 2024. doi: 10.15607/RSS.2024.XX.090. URL https://roboticsproceedings.org/rss20/p090.html.

GigaWorld Team, Angen Ye, Angyuan Ma, Boyuan Wang, Chaojun Ni, Fangzheng Ye, Guan Huang, Guo Li, Guosheng Zhao, Haodong Yan, et al. GigaWorld-Policy-0.5: A faster and stronger WAM empowered by AutoResearch. arXiv preprint arXiv:2607.13960, 2026. URL https://arxiv.org/abs/2607.13960.

Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, and Xiaolong Wang. On pre-training for visuo-motor control: Revisiting a learning-fromscratch baseline. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 12511–12526. PMLR, 2023. URL https://proceedings.mlr.press/v202/hansen23c.html.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016. URL https://openaccess.thecvf.com/content\_cvpr\_2016/ html/He\_Deep\_Residual\_Learning\_CVPR\_2016\_paper.html.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In Proceedings of Robotics: Science and Systems, 2025a. doi: 10.15607/ RSS.2025.XXI.017. URL https://www.roboticsproceedings.org/rss21/p017. html.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An opensource vision-language-action model. In Proceedings ofthe 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pp. 2679–2713. PMLR, 2025b. URL https://proceedings.mlr.press/v270/kim25c.html.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. In International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=wPEIStHxYH.

Seungsu Kim, Jinyoung Choi, Seungmin Baek, and Jean-Michel Renders. ProgVLA: Progress-aware robot manipulation skill learning, 2026b. URL https://arxiv.org/abs/2605.28231.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, volume 36, pp. 44776–44791, 2023. doi: 10.52202/075280-1939. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets\_and\_ Benchmarks.html.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, and Franziska Meier. Where are we in the search for an artificial visual cortex for embodied intelligence? In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 655–677. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-0031. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/022ca1bed6b574b962c48a2856eb207b-Paper-Conference.pdf.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3M: A universal visual representation for robot manipulation. In Proceedings of the 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pp. 892–909. PMLR, 2023. URL https://proceedings.mlr.press/v205/nair23a.html.

Alexander Nikulin, Ilya Zisman, Denis Tarasov, Nikita Lyubaykin, Andrei Polubarov, Igor Kiselev, and Vladislav Kurenkov. Latent action learning requires supervision in the presence of distractors. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 46427–46447, 2025. URL https://proceedings.mlr.press/v267/nikulin25a.html.

Michael Noseworthy, Bingjie Tang, Bowen Wen, Ankur Handa, Chad Kessens, Nicholas Roy, Dieter Fox, Fabio Ramos, Yashraj Narang, and Iretiayo Akinola. FORGE: Force-guided exploration for robust contact-rich manipulation under uncertainty. IEEE Robotics and Automation Letters, 10(5): 4436–4443, 2025. doi: 10.1109/LRA.2025.3551637. URL https://doi.org/10.1109/ LRA.2025.3551637.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khali-´ dov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou, Julien´ Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=a68SUt6zFt.

Jonas Pai, Liam Achenbach, Oliver Sanchez, Stefanos Charalambous, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond VLAs. In Proceedings of Robotics: Science and Systems, 2026. doi: 10.15607/RSS.2026.XXII.077. URL https://www.roboticsproceedings.org/ rss22/p077.html.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Peebles\_ Scalable\_Diffusion\_Models\_with\_Transformers\_ICCV\_2023\_paper.html.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32, 2018. doi: 10.1609/aaai.v32i1.11671. URL https://ojs.aaai. org/index.php/AAAI/article/view/11671.

Ilija Radosavovic, Baifeng Shi, Letian Fu, Ken Goldberg, Trevor Darrell, and Jitendra Malik. Robot learning with sensorimotor pre-training. In Proceedings ofthe 7th Conference on Robot Learning, volume 229 of Proceedings ofMachine Learning Research, pp. 683–693, 2023. URL https: //proceedings.mlr.press/v229/radosavovic23a.html.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL https://jmlr.org/papers/v21/20-074.html.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. SmolVLA: A vision-language-action model for affordable and efficient robotics, 2025. URL https://arxiv.org/abs/2506.01844.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve´ Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3.´ Transactions on Machine Learning Research, 2026. URL https://arxiv.org/abs/2508.10104. To appear.

Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar, Hyung Won Chung, Sharan Narang, Dani Yogatama, Ashish Vaswani, and Donald Metzler. Scale efficiently: Insights from pre training and fine-tuning transformers. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2109.10686.

Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In Proceedings ofthe 37th Annual Allerton Conference on Communication, Control, and Computing, pp. 368–377, 1999. URL https://arxiv.org/abs/physics/0004057.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9:2579–2605, 2008. URL https://jmlr.org/papers/v9/ vandermaaten08a.html.

Seonghyeon Ye et al. World action models are zero-shot policies, 2026. URL https://arxiv. org/abs/2602.15922.

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination?, 2026. URL https://arxiv.org/abs/2603.16666.

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Proceedings of Robotics: Science and Systems, 2023. doi: 10.15607/RSS.2023.XIX.016. URL https://roboticsproceedings.org/rss19/ p016.html.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-VLA: Soft-prompted transformer as scalable cross-embodiment visionlanguage-action model. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=kt51kZH4aG.

Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. URL https://zhouyisjtu.github.io/project\_ rotation/rotation.html.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pp. 2165–2183, 2023. URL https: //proceedings.mlr.press/v229/zitkovich23a.html.

## A ADDITIONAL EXPERIMENTAL DETAILS

We provide additional experimental details for all experiments in this paper.

## A.1 POLICY ARCHITECTURE

Visual encoder. The visual backbone loaded from DINOv2 ViT-S/14. At 224 × 224, the 14-pixel patch size yields a $1 6 \times 1 6 = 2 5 6$ patch-token grid per camera, each 384 wide. The class token and any register tokens are not passed to the policy. The same backbone weights are shared across all views; the resampler distinguishes views using learned view embeddings. The default policy fine-tunes the backbone. The image encoder is not a video encoder and receives no frame history in this paper.

Language encoder. We load the language backbone from Google T5-Efficient-Tiny. We only load transformers.T5TokenizerFast and transformers.T5EncoderModel and drop the decoder part. Tokenization uses padding to a fixed maximum length of 32 and truncation. The attention mask is used for the masked mean, and the projected padded token rows are set to zero. The encoder has a model width of 256, a feed-forward width of 1024, four encoder layers, four attention heads, and $d _ { k v } = 6 4$ . A LayerNorm and a 256 → 768 linear adapter produce the policy-width language tokens.

Resampler and FiLM. Each view is projected from 384 to 768 and compressed to 48 learned query tokens. The queries use eight attention heads in one MultiheadAttention layer, followed by LayerNorm. This resampler implementation has no separate FFN or dropout and no explicit resampler residual branch. Query parameters are separate for each view unless a view-role alias is supplied. The number of visual prefixes equals the number of views times 48 learned query tokens.

To measure the resampled visual prefix’s sensitivity to the language instruction, we introduce $\Delta _ { p r e f i x } ,$ which we use in Section 4.6. Let $\mathbf { R _ { i } } = \left\lceil \mathbf { r } _ { t } ^ { 1 } ; \ldots ; \mathbf { r } _ { t } ^ { V } \right\rceil = R ( \mathbf { o } _ { i } , \mathbf { s } _ { i } , \ell _ { i } )$ denote a visual prefix produced by the resampler using the original instruction $\ell _ { i }$ and $\tilde { \bf R } _ { i } = R ( { \bf o } _ { i } , { \bf s } _ { i } , \tilde { \ell } _ { i } )$ denote the visual prefix after replacing the instruction with one from a different instruction in dataset. The relative change is defined as:

$$
\Delta _ { p r e f i x } = \sqrt { \frac { \sum _ { i = 1 } ^ { N } | | \tilde { \mathbf { R } } _ { i } - \mathbf { R } _ { i } | | _ { F } ^ { 2 } } { \sum _ { i = 1 } ^ { N } | | \mathbf { R } _ { i } | | _ { F } ^ { 2 } } } ,\tag{10}
$$

where $N = 8 0 0 0$ . For fairness, we use the same instruction-swap pairs for all variants in the experiment.

Action generator. We use a $D _ { a } \to 7 6 8$ projection to transform noisy actions into tokens. We follow the widely used sinusoidal time embedding to inject interpolation time. Let $P$ be the complete conditioning prefix (visual tokens, language tokens, and the state token) and let A be the noisy action suffix. Prefix query rows can attend to every valid prefix key but cannot attend to any noisy action key. Every action query row can attend to every valid prefix key and to every action key, including the other action tokens. Thus, action tokens are bidirectional, not causal. The language mask removes padding from the key side. The task head pools only the visual-prefix positions: at layer 0 this is direct visual information. After a transformer block, the prefix attention rule allows those visual positions to carry information from the language and state tokens as well.

Variational information bottleneck. The VIB is an archived ablation, not a module in the current policy. We apply VIB after the visual resampler, before concatenating language tokens. For each visual token x,

$$
\mu = W _ { \mu } { \mathbf x } + b _ { \mu } , \quad \log \sigma ^ { 2 } = W _ { \sigma } { \mathbf x } + b _ { \sigma } , \quad { \mathbf z } = \mu + \epsilon \odot \exp ( \frac { 1 } { 2 } \log \sigma ^ { 2 } ) .\tag{11}
$$

We adopt a standard normal Gaussian as the prior, and $\beta _ { I B } = 4 \times 1 0 ^ { - 8 }$ in experiments.

Action probe. To quantify whether a representation is useful for action prediction, we freeze each model’s representation extractor $f _ { \phi }$ and train an action probe, following prior robot representation studies (Radosavovic et al., 2023; Nikulin et al., 2025). The probe is trained mainly to evaluate representation quality rather than to control robots. Given a visual prefix, the probe predicts the robot’s next action chunk. We make a separate train/test split of the LIBERO demonstrations. For each task, we train the probe on 80% of the episodes and test it on the remaining 20%. This tests whether the frozen representation contains action information that can be read out on episodes in the test set. More accurate predictions indicate that the representation preserves higher-quality information for manipulation. We report NMAE and $R ^ { 2 }$ as the metrics:

$$
N M A E = m e a n \left( \frac { | \hat { \mathbf { a } } - \mathbf { a } | } { \mathbf { q } _ { 9 9 } - \mathbf { q } _ { 0 1 } } \right) , \quad R ^ { 2 } = 1 - \frac { | | \hat { \mathbf { a } } - \mathbf { a } | | _ { 2 } ^ { 2 } } { | | \mathbf { a } - \bar { \mathbf { a } } | | _ { 2 } ^ { 2 } } ,\tag{12}
$$

where a is the ground-truth action chunk, aˆ is the probe prediction, a¯ is the mean action chunk, and q<sub>01</sub> and q<sub>99</sub> are the per-dimension action percentiles. The action probe’s training target approximates the full $\bar { H } \times D _ { a }$ action chunk.

## A.2 INFERENCE COST

Time cost. We tested the inference time on an Nvidia 5090 GPU. The full inference time cost for the default configuration without any optimization is 7.01 ms, where the representation extractor $f _ { \phi }$ costs about 3.43 ms, while the action generator costs about 3.08 ms. We also tried to accelerate inference using CUDA Graph capture and replay to reduce CPU-side kernel-launch overhead. After acceleration, we reach 5.79 ms for inference. All results in this paper are inferred by the policy without acceleration, since it is already fast enough.

Memory cost. The policy’s model weights occupy a total of 95.4 MB of GPU memory, with DINOv2-S, T5, and DiT accounting for 42.1 MB, 23.7 MB, and 23.7 MB, respectively, and the remaining components contributing 5.9 MB. All model parameters are stored in BF16 precision.

## A.3 SIMULATION

Implementation Details. We resize all images to 224 × 224 in both training and evaluation. Our visual encoder is DINOv2 ViT-S/14 (Oquab et al., 2024) (22.1M parameters), and we fine-tune it by default. A cross-attention resampler maps its patch features to 48 tokens per camera, each with width 768. We encode instructions with the Google T5-Efficient-Tiny language encoder (Raffel et al., 2020; Tay et al., 2022), which has 11.4M parameters and is fine-tuned with the policy. A trainable LayerNorm and linear layer map its output to width 768. Both the visual and language encoders are the smallest in their respective families.

The default action generator has one DiT-style block (Peebles & Xie, 2023) with width 768 and about 12.2M parameters. A small task head maps the mean visual token to discrete task types. We train it with cross-entropy weight λ = 0.01. The complete policy has about 48.9M parameters.

We train for 80k steps on LIBERO and 120k steps on RoboTwin 2.0. We train directly on the original LIBERO dataset, while following Motus’s (Bi et al., 2026) recipe to use 2,500 demonstrations collected in clean scenes (50 per task) plus 25,000 demonstrations collected in randomized scenes (500 per task). Both settings use AdamW (Loshchilov & Hutter, 2019) with a learning rate of $5 \times 1 0 ^ { - 5 }$ , batch size 392, BF16 training, gradient clipping at 1.0, and an EMA decay of 0.999. For flow-matching training (Lipman et al., 2023), we sample times from Beta(1.0, 1.5). We use 2× A800 for training; LIBERO costs about 8 hours, while RoboTwin 2.0 costs about 13 hours. At test time, we generate each action chunk with 10 Euler steps.

Figure 6 compares the training loss curves across the ablation studies. Lower training loss does not necessarily translate into better policy performance. Notably, the best-performing configurations often exhibit the highest training losses. We do not report validation losses because we train on the full dataset without a held-out validation split.

LIBERO. We train a single joint model across four suites with 40 tasks. We do not split the dataset for validation, so we uniformly sample all data for training. We use the chunk length $H = 1 2$ for LIBERO. If fewer than H actions remain, we repeat the final action and mark the repeated positions as invalid. Invalid positions do not contribute to the flow-matching loss $\mathcal { L } _ { f m }$ Figure 7 shows t-SNE analysis for all 40 tasks, with and without language conditioning. We follow the original LIBERO evaluation protocol to identify whether the task is successful. We repeated every task 50 times with randomized initial conditions, for a total of 2,000 trials. Figures 8 and 9 record the manipulation process.

(a) Visual initialization / architecture  
![](images/f17777340f05ebc460d26b722fdee4692353a9a6a72823bb0ae239f0deabd82b.jpg)

(b) Visual encoder adaptation  
![](images/d7f4c7a8f88f19a3ec0ded1ec26352c59e580a40ca226f175b5218e06e629a91.jpg)

(c) Language: LIBERO  
![](images/b748effb227737b5dfd6e66b0cd90c1429647f1ed01b95f880dc23c80a8397e7.jpg)  
(e) Language: RoboTwin 2.0

(d) Information bottleneck  
![](images/84e3fb2622b701055bf2ae59ea847a5a31b21ed199c4c98d982f463e1f58f60c.jpg)

![](images/d6f0bdd493a0f62297273af1a0e21d0d5db86d4b4a4be066434fad9e0fb00800.jpg)  
Figure 6: Training loss does not predict policy performance. Training loss (log scale) for the ablations in Sections 4.3 (a), 4.4 (b), 4.5 (c: LIBERO, e: RoboTwin 2.0), and 4.6 (d). In (a), (c), and (d), the best-performing configuration (dark blue) reaches the highest loss; freezing the encoder (b) is the exception, raising loss and lowering success. With and without language, the RoboTwin 2.0 curves (e) nearly coincide.

RoboTwin 2.0. In all experiments, we report a model trained on 50 tasks. We use the chunk length H = 50 for RoboTwin 2.0. We also have no validation data, and we feed all data into the training process.

We report detailed per-task evaluation on the RoboTwin 2.0 benchmark in Clean and Randomized settings. These two settings differ only in evaluation conditions, while the training setup remains identical. We follow the original evaluation protocol to assess task success. We repeated every task 100 times for evaluation, for a total of 10, 000 trials. The rollouts for different tasks are presented in Figures 10– 19

![](images/28b41915bfa55d5da1006b4c54fc69254b7c4a176f520764a26d706d761cda06.jpg)  
Figure 7: t-SNE of visual-prefix representations on LIBERO, with and without language. Colors denote tasks. With language (top), each suite forms ten well-separated task clusters. Without language (bottom), tasks stay separable only where the scene identifies them: Object remains separable, Spatial and Long each merge a few tasks (red dashed boxes), and Goal, whose tasks share one scene, collapses into a single cloud, mirroring the task-head accuracy in Table 5.

Forge (Noseworthy et al., 2025). We have an additional simulation for precision manipulation. We choose the task peg insertion and adopt only visual information. The task is difficult because it requires precise peg manipulation. The hole has an internal diameter of 8.1 ms, while the peg has a diameter of 8.0 ms. We train only 30k steps from scratch for this task. However, we achieved an 84% success rate over 50 trials, similar to the best result reported in (Noseworthy et al., 2025). As visualized in Figure 20, the robot smoothly inserts the peg into the hole from different initial conditions. In this task, we use H = 12, and the control frequency is 15 Hz. We keep the other configurations the same as in the aforementioned simulations.

## A.4 BASELINE POLICIES

We summarize the baseline policies compared in Tables 1 and 2. We organize them into three groups according to their use of pretrained vision–language or video-generation backbones. We include methods that combine both types of components in the video-model-based group. This grouping concerns backbone initialization rather than whether a policy accepts language inputs or uses pretraining in general.

## A.4.1 POLICIES BASED ON PRETRAINED VISION–LANGUAGE MODELS

OpenVLA (Kim et al., 2025b) adapts the vision–language backbone for robot action prediction. It combines a Llama 2 language model with visual features from DINOv2 and SigLIP, and is trained on heterogeneous robot demonstrations. Each continuous action dimension is quantized into discrete tokens, allowing the model to predict robot commands autoregressively from image observations and language instructions using a next-token prediction objective.

SmolVLA (Shukor et al., 2025) combines a SmolVLM-2 backbone with a flow-matching action expert. The expert alternates self-attention over action tokens with cross-attention to multimodal features to generate continuous action chunks. Visual-token reduction and intermediate VLM layers reduce computational cost, while an asynchronous inference scheme can decouple action generation from execution. Our LIBERO comparison includes the 2.25B configuration reported in the original paper.

π<sub>0</sub> (Black et al., 2025b) couples a pretrained PaliGemma vision–language model with a dedicated action expert trained using flow matching. Images, language instructions, and proprioceptive states condition the generation of continuous action chunks, avoiding the need to discretize control outputs.

![](images/d2eddef8963433b0721c4a77dac67111e295c08ceac254dd4c65bd18d87bea05.jpg)  
pick up the alphabet soup and place it in the basket  
pick up the cream cheese and place it in the basket

## Object

![](images/98feb0d2b2fac09ac295ab16cb590b410b070f85c92b75c6505fd33f1e8e0608.jpg)  
pick up the salad dressing and place it in the basket

![](images/74aa36e7c8b08ee53c1a223905b846d299a4df596876f5ee3692cc280970763a.jpg)  
pick up the bbq sauce and place it in the basket

![](images/c2251caeaab2e4931a4cfc1d9b08d03c5b0b4fda621141caffb92dbe45dfd7be.jpg)  
pick up the ketchup and place it in the basket

![](images/4520ef5b2f6a17459df7b0dbfa1b7515b35dc3b6419a99faa388b2be828042ff.jpg)  
pick up the tomato sauce and place it in the basket

![](images/5e854a7d20d4fedd15c63fd02e99589666a3d027318654a2d90065a82848e18c.jpg)  
pick up the butter and place it in the basket

![](images/3de730ec02591ec0a3474fd53b04f92bd7c0b0e6af55417c603ffc500290fbb0.jpg)  
pick up the milk and place it in the basket

![](images/73e4da2509bf62824d714b668dbd2a07c6a2f083709a9afb26dc6cc962f5b586.jpg)  
pick up the chocolate pudding and place it in the basket  
pick up the black bowl between the plate and the ramekin and place it on the plate

## Spatial

![](images/6bf3e997c1745c16ef6b2c80eea920cc394169a149abe482a20ae0bd9a3c2382.jpg)

![](images/a1f2f4ca7ae7208c30d34c258fcdf90fc0ebdfb6fe4ed0166c0a35bce640300c.jpg)

![](images/cb3b4dcc1660074d19a3944ce3bc63535c4fd62d0b69609608b88f0a96521516.jpg)

![](images/caa2ca1c1d5ef38bae5b96c2d03a7e4b5257ba1aa7b6d08fd20f05eb6c8f27b0.jpg)  
pick up the orange juice and place it in the basket

![](images/f8bfe351f598ed7bc55f6a7b913af3a80ad10358febdd035a3e504bbea22322e.jpg)

![](images/896e11d497aaa7623e3197eab1608ddcdd18adc3f591e1fd59ad7a4e7278d158.jpg)

![](images/c4c8dd07fbda74ad654b98dc848c509c09f144fae73d28b300c16f589adc7cde.jpg)  
pick up the black bowl next to the ramekin and place it on the plate

![](images/b5ab96b83efb12ea4f34ef53aa22378be97989ac2945b18ec849a72cda4d732d.jpg)

![](images/fffcf73f29858186897644f2f61cc7541e1e62ee83a680f8e84154237d966c85.jpg)

![](images/b01a10f64842896fc1bd5483241bd9f40fe24fdc843ad241bebe4afaf7dfa9ea.jpg)

![](images/e5b0b111b74858e8b983cada00cb6bba8b4ec634098a46ce80b581539fa623d4.jpg)

![](images/c5c34f26a742062be088f0d0ff1b2a6ef2141c9747811aa556d6c28a52920e32.jpg)

![](images/199bcd7691324597e8344ccbcbdc6c82a2d0a9fb72692eeccc384573dbaf6754.jpg)  
pick up the black bowl from table center and place it on the plate

![](images/8d9ba182ec869c6cb1b506909b837e50624d49e657e95c774c90964098ea0e41.jpg)

![](images/6a6ffb6554da26044ed6b73d7816df6e48664904b02d9ae1ba4cb3caa4ef53c9.jpg)

![](images/5e7ab3c748eb159b98165c9526e1cfb0f617bed68275e28f4edd04db43b01158.jpg)

![](images/524a7c1dd864dbd79af9763a4f5004e130e3f15ed9fb102ca1830c56d0d6c949.jpg)

![](images/5c69fa120f8830ef2475f13ea3e778d4d7a98eba13a6c3475ff9641060caeab9.jpg)

![](images/cc51b2646f66ea552282aa65dbb31f938871f004b66f29fa4e6e0e245e1041cd.jpg)  
pick up the black bowl on the cookie box and place it on the plate

![](images/dd9384050ff87cbde13c2963ce8c62ec800b102ff4b8609ad9276e7642cb1244.jpg)

![](images/b1c4866fce731b6ee6e4c1a15dc417f46a4bc1c77aac398ce23725d93bedada7.jpg)

![](images/f87fd45adc2e2c5442773dd7da3e795e50b2ab4f561aade8ce7cfa48953d27bd.jpg)

![](images/330505a755aaa570cb6248820c6977b3412c1100108839ba228f11fbf76f87cd.jpg)

![](images/e989dfb565dbf7947062303012037e349759475680cc8bb2c8c3a8b60c0481f4.jpg)  
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate

![](images/b82bfd839541893c3405cddfc4e6fa9ae53c4c36845867e7cc072d929cca3c33.jpg)

![](images/ddc49563a3e7338f6c942c9829f8157608807168c3a44db6b67227e4ce5827d5.jpg)

![](images/1e1b444aa6b3909ba85ee93beced209c61302a7e6850a4131c661abf26b0aaf4.jpg)  
pick up the black bowl on the ramekin and place it on the plate

![](images/fef4f46f16b9a3c278ca26ad6c611fc04799f3e8267ab9fcbad27636fd1922c0.jpg)  
pick up the black bowl next to the cookie box and place it on the plate

![](images/e24b7e6df9ce51405b37485e698cf0e8c28d0101f5770b82aa9c887227cf8874.jpg)

![](images/e444d6c8bbe6a940b495a85fb17fdf7cdc9808e69e57bab1b99c5a24b86ab8de.jpg)

![](images/765026345f5b0e508fc1191e37d43dc0b222e3077138dd468787043952687577.jpg)

![](images/ae56068934a761def588d1fbc35ed0653e752c34d88682e17b1c2df40bda0af3.jpg)

![](images/446eb68994d14d6f21ceac211baccb06ba1e6097eaec8d28dce29e4884f11dd0.jpg)

![](images/cf2c35f7b736eaac6996c0b85889c08935acbb12eef8e50e3ed0c3e898bf2a3e.jpg)  
pick up the black bowl on the stove and place it on the plate

![](images/6f7e6581764c609be6c49cb24d33d8b4b6b03d32a19180fe158c1de8e1ec22e0.jpg)  
pick up the black bowl next to the plate and place it on the plate

![](images/f824f2dc30be2807f9302c012d1c225cb2a4b0c2603896350149cf19d063925b.jpg)  
pick up the black bowl on the wooden cabinet and place it on the plate

![](images/244b17c9dd1de9a740ecdfbc9c9a56c1f662949f1e827aa18f73298659be8cb1.jpg)  
Figure 8: Example CoRP rollouts on LIBERO-Object and LIBERO-Spatial. Each row shows six frames from one rollout of the task named above it, from left to right. Object tasks require picking the named item and placing it in the basket; Spatial tasks move the same black bowl to the plate from different starting locations.

Cross-embodiment robot pretraining combines general visual–language representations with motor supervision, after which the policy can be adapted to downstream manipulation tasks through taskspecific post-training.

Goal

open the middle drawer of the cabinet

![](images/749fb2a6fd7bb9cc3c8f679885c359560521fba7d28e20ff4e22c075715ec8d7.jpg)  
Long  
put both the alphabet soup and the tomato sauce in the basket  
put the bowl on the stove

![](images/4dbd9089d23863765ab589c01158ebda2b795e3faaa45a0bbbefcf211555ea71.jpg)  
put both the cream cheese box and the butter in the basket

![](images/c7668dcd752d1ef9eb305c3c8622d68cc5c0682f2f422e87ec692e579850b651.jpg)

![](images/c6c6d3ea85f385fba02417545a1bc6aa88eb0715d8269a429bb0ef340bff7632.jpg)  
put the wine bottle on top of the cabinet  
turn on the stove and put the moka pot on it

![](images/0f82667b484686123fc668fad9581a132e9d571406ce760b50d1d473a6818744.jpg)  
open the top drawer and put the bowl inside

![](images/19cbcb5a89527f24214deaf3ce314be2d605558c5f195c6d96e0428d2e3112fb.jpg)

![](images/3a6e9596803683164ed81d57c0bf32bb5ed4c20888a813a1198669afdd027501.jpg)  
put the black bowl in the bottom drawer of the cabinet and close it  
put the bowl on top of the cabinet

![](images/b3c5658484e21b7d129d528dd903d8b4ff0f9ee858872ce67e55edab1ecbbd1a.jpg)  
put the white mug on the left plate and put the yellow and white mug on the right plate

![](images/9f1792d28727f20c9ce867ef0c466f2910d99d1c474543d0d2f32c6f662337aa.jpg)  
push the plate to the front of the stove

![](images/97d16aac0b8a3379bac4f09831def870bb40101b3f719430b1a11c7e587c934e.jpg)

![](images/16e1229c1869cd962a12ea6eff55a8bf1c2b019faa4c18d8ed754536354148f5.jpg)  
pick up the book and place it in the back compartment of the caddy  
put the cream cheese in the bowl

![](images/2b826742a2c55ed2a32544856c745290fc8be053d10f0c7974d5e3296ff7219d.jpg)

![](images/8e45a6ee6b36d614705682f4036029a1bd4599234a67d97377e3fc64dd7dab9c.jpg)  
put the white mug on the plate and put the chocolate pudding to the right of the plate  
turn on the stove

![](images/14649a2c781f8f8ae83288157771dcef77a8e0b489abfe1c6712e99c835cec07.jpg)  
put both the alphabet soup and the cream cheese box in the basket

![](images/fa7c2e6d20ce87b162971ded387bc9a47516be65869ec5928635fcca2e6353e4.jpg)  
put the bowl on the plate

![](images/5695048bc3c5f56cb2233d9e67359fe968b47aa3f697d9d84b4652078aa0ecff.jpg)  
put both moka pots on the stove

![](images/dc3b84ae91df463ee8a7468bdb429d6f10940d5302d4c62666062ab624c2d1c4.jpg)  
put the wine bottle on the rack

![](images/7fbebb3f2c28afeb74feab18350d306a6661d1efcb061bff5c18f8f5175745d6.jpg)

![](images/1cde48c517badfe487d871569a1fb7269ea865ff09d4cbb2e8d8564a223246ae.jpg)  
put the yellow and white mug in the microwave and close it

![](images/b4afa0d2c0ed493580ff895593a85158b6cec4a735f4d2d0620fc8de85c05201.jpg)  
Figure 9: Example CoRP rollouts on LIBERO-Goal and LIBERO-Long. Each row shows six frames from one rollout of the task named above it, from left to right. Goal tasks share a single scene and differ only in the instruction; Long tasks are multi-step, most chaining two subgoals.

$\pi _ { 0 . 5 }$ (Black et al., 2025a) extends the $\pi _ { 0 }$ framework with a broader training mixture that includes robot demonstrations, semantic prediction tasks, and non-robot vision–language data. Its hybrid training recipe uses discrete action tokens during pretraining and introduces a flow-matching action expert during post-training for continuous control. In its original formulation, the model can also predict intermediate language subtasks and condition action generation on them, connecting high-level task decomposition with low-level execution.

![](images/0af34300e6a31db3bf0e57be122cb9e7afde50d4106c68494672e1661d3b2c09.jpg)  
Figure 10: Example CoRP rollouts on RoboTwin 2.0 (Clean), tasks 1–10. Each row shows six frames from one rollout, from left to right in time; each frame combines the head-camera view (top) with the two wrist-camera views (bottom). Figures 10–14 cover all 50 tasks in alphabetical order.

![](images/d72d6344115aefce9b4d92349f7a2805b29511d4a9cac9f0578d337afaca399a.jpg)  
Figure 11: Example CoRP rollouts on RoboTwin 2.0 (Clean), tasks 11–20. Layout as in Figure 10.

![](images/4723343545f32a5f9463f6eea272a29d243856f58abf09a3187c3d11c521de6d.jpg)  
Figure 12: Example CoRP rollouts on RoboTwin 2.0 (Clean), tasks 21–30. Layout as in Figure 10.

![](images/64872c00298cdeeace678f7d410e8f9b211c407ef0ea5205fb268da200cf94cd.jpg)  
Figure 13: Example CoRP rollouts on RoboTwin 2.0 (Clean), tasks 31–40. Layout as in Figure 10.

![](images/ae39000057c4cba76a52850a66b89322544b29b92bc61fdbcb435c146e69da9a.jpg)  
Figure 14: Example CoRP rollouts on RoboTwin 2.0 (Clean), tasks 41–50. Layout as in Figure 10.

![](images/f154ed47ee66fe7b32cfd682d6f6c9afe4337bdb965fb84e9497df68c7149452.jpg)  
Figure 15: Example CoRP rollouts on RoboTwin 2.0 (Randomized), tasks 1–10. Same tasks and layout as Figure 10, under randomized distractors, textures, and lighting. Figures 15–19 cover all 50 tasks in alphabetical order.

![](images/a7d18b0f48f0ee1fd05d891aab8484b2c07c80000a901639ff026bc98007d82b.jpg)  
Figure 16: Example CoRP rollouts on RoboTwin 2.0 (Randomized), tasks 11–20. Same tasks and layout as Figure 11.

![](images/91ac63ce5340b7592f31329d0e1c7ab37dd4fb48002f6f5ac6d2767439363ce8.jpg)  
Figure 17: Example CoRP rollouts on RoboTwin 2.0 (Randomized), tasks 21–30. Same tasks and layout as Figure 12.

![](images/f1f13b12389adff6fcdbad71cbec9532218e11c982f7a4587d0ab2accebbda1a.jpg)  
Figure 18: Example CoRP rollouts on RoboTwin 2.0 (Randomized), tasks 31–40. Same tasks and layout as Figure 13.

![](images/d24715e194c81b6d39979ebcba1a55d29781d1efb823b7e99574de840fa2e3ab.jpg)  
Figure 19: Example CoRP rollouts on RoboTwin 2.0 (Randomized), tasks 41–50. Same tasks and layout as Figure 14.

Table 8: Task-wise success rates (%) on RoboTwin 2.0. Best results are in bold for each task and setting, including ties.
<table><tr><td rowspan=2 colspan=26>π0        $\pi _ { 0 . 5 }$      X-VLA    Motus     GWP      DP     ACT     OursTask                  C   R   C   R   C   R   C   R   C   R                             R</td></tr><tr><td rowspan=1 colspan=6>C RC RCR</td></tr><tr><td rowspan=1 colspan=20>Adjust Bottle           90  56  79  83  100  99  89  93  100 100</td><td rowspan=1 colspan=6>97  0  97  23  98  98</td></tr><tr><td rowspan=1 colspan=2>Beat Block Hammer</td><td rowspan=1 colspan=6>43  21  63  50</td><td rowspan=1 colspan=8>92  88  95  88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>86  86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>42  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>56  3  97  89</td></tr><tr><td rowspan=1 colspan=2>Blocks Ranking RGB</td><td rowspan=1 colspan=6>19   5   43  35</td><td rowspan=1 colspan=8>83  83  99  97</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>92  96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1   0  55  61</td></tr><tr><td rowspan=1 colspan=2>Blocks Ranking Size</td><td rowspan=1 colspan=6>7   1   8   14</td><td rowspan=1 colspan=9>67  74  75  63</td><td rowspan=1 colspan=3>44  48</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0   0  48  42</td></tr><tr><td rowspan=1 colspan=2>Click Alarm clock</td><td rowspan=1 colspan=6>63  11  97  93</td><td rowspan=1 colspan=9>99  99  100 100</td><td rowspan=1 colspan=3>100 100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>61   5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>32  4  97  92</td></tr><tr><td rowspan=1 colspan=2>Click Bell</td><td rowspan=1 colspan=6>44   3   75  76</td><td rowspan=1 colspan=4>100 100</td><td rowspan=1 colspan=5>100 100</td><td rowspan=1 colspan=3>100 100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>54  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>58  3  100  99</td></tr><tr><td rowspan=1 colspan=2>Dump Bin Bigbin</td><td rowspan=1 colspan=6>83  24  30  42</td><td rowspan=1 colspan=4>79  77</td><td rowspan=1 colspan=11>95  91  92  100  49   0</td><td rowspan=1 colspan=3>68   1  70  70</td></tr><tr><td rowspan=1 colspan=2>Grab Roller</td><td rowspan=1 colspan=6>96  80  90  89</td><td rowspan=1 colspan=4>100  100</td><td rowspan=1 colspan=11>100  100  100  100  98   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>94  25  100  99</td></tr><tr><td rowspan=1 colspan=2>Handover Block</td><td rowspan=1 colspan=3>45   8</td><td rowspan=1 colspan=3>18  19</td><td rowspan=1 colspan=4>73  37</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>86  73</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>80  80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>10   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>42  0  79  50</td></tr><tr><td rowspan=1 colspan=2>Handover Mic</td><td rowspan=1 colspan=3>98  13</td><td rowspan=1 colspan=3>28  18</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>78  63</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>72  72</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>53  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>85   0   99  96</td></tr><tr><td rowspan=1 colspan=2>Hanging Mug</td><td rowspan=1 colspan=3>11   3</td><td rowspan=1 colspan=3>3   3</td><td rowspan=1 colspan=2>23</td><td rowspan=1 colspan=2>27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>38  38</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>16  12</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7   0   1   8</td></tr><tr><td rowspan=1 colspan=2>Lift Pot</td><td rowspan=1 colspan=3>84  36</td><td rowspan=1 colspan=3>0   0</td><td rowspan=1 colspan=2>99</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=2>99</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>98  98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>39   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>88   0  100  98</td></tr><tr><td rowspan=1 colspan=2>Move Can Pot</td><td rowspan=1 colspan=2>58  21</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>29  27</td><td rowspan=1 colspan=2>89</td><td rowspan=1 colspan=2>86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=2>74</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>76  78</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>39  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>22  4  85  75</td></tr><tr><td rowspan=1 colspan=2>Move Pillbottle Pad</td><td rowspan=1 colspan=2>21   1</td><td rowspan=1 colspan=4>33  29</td><td rowspan=1 colspan=2>73</td><td rowspan=1 colspan=2>71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>90  90</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0   0   88  74</td></tr><tr><td rowspan=1 colspan=2>Move Playingcard Away</td><td rowspan=1 colspan=2>53  22</td><td rowspan=1 colspan=4>59  67</td><td rowspan=1 colspan=2>93</td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>78  72</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>47  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36  0</td><td rowspan=1 colspan=1>96  96</td></tr><tr><td rowspan=1 colspan=2>Move Stapler Pad</td><td rowspan=1 colspan=6>0   2   16  18</td><td rowspan=1 colspan=2>78</td><td rowspan=1 colspan=2>73</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=2>85</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>92  82</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0   0</td><td rowspan=1 colspan=1>42  37</td></tr><tr><td rowspan=1 colspan=2>Open Laptop</td><td rowspan=1 colspan=6>85  46  19  35</td><td rowspan=1 colspan=4>93  100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=2>91</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>96  98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>49  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Open Mícrowave</td><td rowspan=1 colspan=6>80  50  35  37</td><td rowspan=1 colspan=2>79</td><td rowspan=1 colspan=2>71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=2>91</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=2>66</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Pick Diverse Bottles</td><td rowspan=1 colspan=6>27   6   5   3</td><td rowspan=1 colspan=2>58</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=2>91</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=2>70</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Pick Dual Bottles</td><td rowspan=1 colspan=6>57   12  10   6</td><td rowspan=1 colspan=2>47</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=2>90</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=2>86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>24  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=2>Place A2b Left</td><td rowspan=1 colspan=6>31   1   62  60</td><td rowspan=1 colspan=2>48</td><td rowspan=1 colspan=2>49</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=2>79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>94  88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>2   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Place A</td><td rowspan=1 colspan=1>2b Right</td><td rowspan=1 colspan=6>27   6   62  57</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=2>87</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>90  92</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>13  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0   63  67</td></tr><tr><td rowspan=1 colspan=1>Place B</td><td rowspan=1 colspan=1>read Basket</td><td rowspan=1 colspan=4>17   4   48</td><td rowspan=1 colspan=2>56</td><td rowspan=1 colspan=2>81</td><td rowspan=1 colspan=2>71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=2>94</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>82  82</td><td rowspan=1 colspan=2>14  0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0   78  78</td></tr><tr><td rowspan=1 colspan=1>Place B</td><td rowspan=1 colspan=1>read Skillet</td><td rowspan=1 colspan=4>23   1   38</td><td rowspan=1 colspan=2>46</td><td rowspan=1 colspan=2>77</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>94</td><td rowspan=1 colspan=2>90</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0   92  95</td></tr><tr><td rowspan=1 colspan=1>Place B</td><td rowspan=1 colspan=1>urger Fries</td><td rowspan=1 colspan=4>80   4   66</td><td rowspan=1 colspan=2>70</td><td rowspan=1 colspan=2>94</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>0   96  97</td></tr><tr><td rowspan=1 colspan=1>Place C</td><td rowspan=1 colspan=1>an Basket</td><td rowspan=1 colspan=4>41        19</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=2>49</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>78</td><td rowspan=1 colspan=2>74</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>70  64</td></tr><tr><td rowspan=1 colspan=1>Place C</td><td rowspan=1 colspan=1>ans Plastic box</td><td rowspan=1 colspan=3>34</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=2>47</td><td rowspan=1 colspan=2>97</td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=3>100</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>97  95</td></tr><tr><td rowspan=1 colspan=1>Place C</td><td rowspan=1 colspan=1>ontainer Plate</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=2>45</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=2>78</td><td rowspan=1 colspan=2>97</td><td rowspan=1 colspan=2>95</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>1   85  85</td></tr><tr><td rowspan=1 colspan=1>Place D</td><td rowspan=1 colspan=1>ual Shoes</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=2>7</td><td rowspan=1 colspan=2>79</td><td rowspan=1 colspan=2>88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>93</td><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=2>84</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0   77  81</td></tr><tr><td rowspan=1 colspan=1>Place</td><td rowspan=1 colspan=1>Empty Cup</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=2>11</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=2>86</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>99</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>90</td><td rowspan=1 colspan=2>90</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>0   91   86</td></tr><tr><td rowspan=1 colspan=1>Place</td><td rowspan=1 colspan=1>Fan</td><td rowspan=1 colspan=3>20   10</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=2>80</td><td rowspan=1 colspan=2>75</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>91</td><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=3>92</td><td rowspan=1 colspan=2>94</td><td rowspan=1 colspan=2>3   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0   79   78</td></tr><tr><td rowspan=1 colspan=1>Place</td><td rowspan=1 colspan=1>Mouse Pad</td><td rowspan=1 colspan=3>7   1</td><td rowspan=1 colspan=3>21  26</td><td rowspan=1 colspan=5>70  70</td><td rowspan=1 colspan=2>66</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=3>88</td><td rowspan=1 colspan=2>90</td><td rowspan=1 colspan=2>0   0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0   57  58</td></tr><tr><td rowspan=1 colspan=1>Place O</td><td rowspan=1 colspan=1>bject Basket</td><td rowspan=1 colspan=3>16   2</td><td rowspan=1 colspan=3>43  36</td><td rowspan=1 colspan=2>44</td><td rowspan=1 colspan=3>39</td><td rowspan=1 colspan=2>81</td><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=3>90</td><td rowspan=1 colspan=2>92</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0   79  69</td></tr><tr><td rowspan=1 colspan=1>Place O</td><td rowspan=1 colspan=1>bject Scale</td><td rowspan=1 colspan=3>10   0</td><td rowspan=1 colspan=3>40  49</td><td rowspan=1 colspan=2>52</td><td rowspan=1 colspan=2>74</td><td rowspan=1 colspan=4>88  85</td><td rowspan=1 colspan=3>88</td><td rowspan=1 colspan=2>80</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>82  70</td></tr><tr><td rowspan=1 colspan=1>Place O</td><td rowspan=1 colspan=1>bject Stand</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=2>11</td><td rowspan=1 colspan=3>74  65</td><td rowspan=1 colspan=2>86</td><td rowspan=1 colspan=2>88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>98  97</td><td rowspan=1 colspan=3>100</td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=2>1</td><td rowspan=1 colspan=1>0   85  83</td></tr><tr><td rowspan=1 colspan=1>Place P</td><td rowspan=1 colspan=1>hone Stand</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=2>7</td><td rowspan=1 colspan=3>49  53</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>87  86</td><td rowspan=1 colspan=2>82</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>72</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>0   80  88</td></tr><tr><td rowspan=1 colspan=1>Place</td><td rowspan=1 colspan=1>Shoe</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=2>6</td><td rowspan=1 colspan=3>57  61</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>99</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=2>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=2>5</td><td rowspan=1 colspan=1>93  95</td></tr><tr><td rowspan=1 colspan=1>Press</td><td rowspan=1 colspan=1>Stapler</td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=2>29</td><td rowspan=1 colspan=2>80</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>93</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=2>6   0</td><td rowspan=1 colspan=2>31</td><td rowspan=1 colspan=1>6   96  97</td></tr><tr><td rowspan=1 colspan=1>Put Bot</td><td rowspan=1 colspan=1>tles in Dustbin</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=2>13</td><td rowspan=1 colspan=2>12</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>77</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>81</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=2>72</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>70</td><td rowspan=1 colspan=2>22  0</td><td rowspan=1 colspan=2>27</td><td rowspan=1 colspan=1>1   20  14</td></tr><tr><td rowspan=1 colspan=2>Put Object Cabinet</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=2>18</td><td rowspan=1 colspan=2>24</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=2>46</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>88  71</td><td rowspan=1 colspan=2>74</td><td rowspan=1 colspan=5>74  42   0</td><td rowspan=1 colspan=2>15</td><td rowspan=1 colspan=1>0   53  57</td></tr><tr><td rowspan=1 colspan=2>Rotate QR code</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=2>15</td><td rowspan=1 colspan=2>47</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=2>34</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=8>89  73   90   84</td><td rowspan=1 colspan=2>13  0</td><td rowspan=1 colspan=2>1</td><td rowspan=1 colspan=1>0   82  85</td></tr><tr><td rowspan=1 colspan=2>Scan Object</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=2>1</td><td rowspan=1 colspan=3>42  38</td><td rowspan=1 colspan=2>14</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=8>67  66   60  64</td><td rowspan=1 colspan=2>9   0</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>85  80</td></tr><tr><td rowspan=1 colspan=2>Shake Bottle</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=5>60  91  100</td><td rowspan=1 colspan=2>99</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=9>100  97  100 100</td><td rowspan=1 colspan=2>65  8</td><td rowspan=1 colspan=2>74</td><td rowspan=1 colspan=1>99  99</td></tr><tr><td rowspan=1 colspan=2>Shake Bottle Horizontally</td><td rowspan=1 colspan=6>99  51  96  100</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=3>100</td><td rowspan=1 colspan=8>98  100  98  59  18</td><td rowspan=1 colspan=2>63</td><td rowspan=1 colspan=1>4  97  97</td></tr><tr><td rowspan=1 colspan=2>Stack Blocks Three</td><td rowspan=1 colspan=6>17   0   15  16</td><td rowspan=1 colspan=2>6</td><td rowspan=1 colspan=2>10</td><td rowspan=1 colspan=3>91</td><td rowspan=1 colspan=4>95  70</td><td rowspan=1 colspan=4>78   0   0</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>16  30</td></tr><tr><td rowspan=1 colspan=2>Stack Blocks Two</td><td rowspan=1 colspan=6>42   1   48  56</td><td rowspan=1 colspan=2>92</td><td rowspan=1 colspan=2>87</td><td rowspan=1 colspan=3>99</td><td rowspan=1 colspan=4>98  100</td><td rowspan=1 colspan=4>94   7   0</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>71  71</td></tr><tr><td rowspan=1 colspan=2>Stack Bowls Three</td><td rowspan=1 colspan=6>66  24  33  35</td><td rowspan=1 colspan=2>76</td><td rowspan=1 colspan=2>86</td><td rowspan=1 colspan=7>79  87  70</td><td rowspan=1 colspan=4>72  63  0</td><td rowspan=2 colspan=3>48  0  53  5282  0  71  78</td></tr><tr><td rowspan=1 colspan=2>Stack Bowls Two</td><td rowspan=1 colspan=6>91  41  78  66</td><td rowspan=1 colspan=2>96</td><td rowspan=1 colspan=2>93</td><td rowspan=1 colspan=11>98  98  96  92  61   0</td></tr><tr><td rowspan=1 colspan=2>Stamp Seal</td><td rowspan=1 colspan=6>3   4   36  23</td><td rowspan=1 colspan=15>76  82  93  92  96  98   2   0</td><td rowspan=1 colspan=3>2   0  82  55</td></tr><tr><td rowspan=1 colspan=26>Turn Switch            27  23   5   6   40  61  84  78  82  84  36  1   5   2  53  54</td></tr><tr><td rowspan=1 colspan=26>Average              46.4216.3442.9843.8472.8872.8488.6486.9686.3685.0428.040.6429.741.7475.7873.36</td></tr></table>

OpenVLA-OFT (Kim et al., 2025a) applies an optimized fine-tuning recipe to OpenVLA. Instead of autoregressively decoding discretized actions, it predicts continuous action chunks in parallel and trains the action outputs with an $L _ { 1 }$ regression objective. The resulting policy also accommodates additional inputs such as multiple camera views and proprioceptive states. It therefore provides a comparison to VLM-based control in which action decoding and fine-tuning are optimized without introducing an iterative diffusion or flow-matching action head.

X-VLA (Zheng et al., 2026) uses learnable domain-specific soft prompts to accommodate differences in robot embodiments and data-collection setups. A Florence-2 encoder processes the main image–language stream, while auxiliary camera views are encoded separately. These representations are combined with proprioceptive inputs and soft prompts in a shared Transformer for flow-matching action generation. The prompts provide domain-dependent conditioning without requiring a separate policy backbone for every embodiment.

## A.4.2 POLICIES BASED ON PRETRAINED VIDEO-GENERATION MODELS

Motus (Bi et al., 2026) integrates understanding, video-generation, and action experts through a Mixture-of-Transformers architecture. It learns latent actions from optical flow to capture motion information that can transfer across embodiments. Modality-specific diffusion timesteps allow the model to switch among action prediction, video prediction, inverse dynamics, and joint video–action modeling. Its staged training transfers general multimodal priors and motion knowledge from heterogeneous datasets to robot control.

Fast-WAM-Joint (Yuan et al., 2026) is the joint-generation variant investigated within the Fast-WAM framework. It couples a pretrained Wan2.2 video diffusion Transformer with an action expert and allows attention between future-video and action tokens during joint denoising. Unlike the main Fast-WAM model, this variant retains future-video generation at inference time. It therefore represents coupled video–action prediction rather than the action-only inference procedure used by Fast-WAM.

Cosmos-Policy (Kim et al., 2026a) adapts a pretrained video diffusion model to robot control through latent-frame injection. Robot states, action chunks, and value targets are represented in the existing latent-frame format, allowing the denoising backbone to learn action prediction, futurestate prediction, and value estimation without a separate action-specific network. The framework supports direct policy execution and, after refinement using rollout data, value-guided action selection. The LIBERO results in our comparison correspond to direct policy evaluation rather than test-time planning.

GigaWorld-Policy (GigaWorld Team et al., 2026) initializes a unified policy from a pretrained video-generation backbone and trains it to predict action chunks together with action-conditioned future observations. Its blockwise causal attention mask prevents action tokens from attending to future-video tokens, while allowing future-video prediction to depend on the actions. This asymmetric information flow makes future prediction an auxiliary training objective rather than a prerequisite for action generation, allowing future-video synthesis to be omitted during inference.

## A.4.3 POLICIES WITHOUT EITHER PRETRAINED BACKBONE

ACT (Zhao et al., 2023) is a Transformer-based imitation learning method that models action chunks using a conditional variational autoencoder. Given camera observations and robot joint states, it predicts a sequence of future actions instead of a single command. A latent variable captures variation in demonstrations during training, while temporal ensembling can combine overlapping predictions during execution to produce smoother control.

Diffusion Policy (Chi et al., 2023) models a visuomotor policy as a conditional denoising process over action sequences. Starting from Gaussian noise, it iteratively refines an action chunk conditioned on observation features. Receding-horizon execution incorporates updated observations into subsequent predictions, while the generative formulation accommodates multimodal demonstration behaviors. The diffusion process operates in action space and does not require a pretrained video-generation backbone.

Octo (Ghosh et al., 2024) is a generalist robot policy pretrained on heterogeneous multi-robot demonstrations. Modality-specific tokenizers encode observations and task specifications, a Transformer fuses the resulting tokens, and a diffusion action head produces continuous action chunks. Tasks can be specified through language instructions or goal images, and the modular design supports adaptation to new observation and action spaces. Although Octo uses a pretrained T5 text encoder, its policy backbone is not initialized from a pretrained VLM.

ProgVLA (Kim et al., 2026b) combines separate pretrained visual and language encoders with a compact flow-matching action expert. Two stages of Perceiver resampling, placed before and after multimodal fusion, compress observation and instruction features into a small set of context tokens. Auxiliary progress heads, trained with offline reinforcement-learning objectives, estimate task progress and provide advantage- and success-based weights for the imitation objective. This combines representation compression with progress-aware policy training without relying on a pretrained VLM or video-generation backbone. We only consider it in LIBERO, as it does not currently release code.

## A.5 REAL-WORLD

Robot platform. We conduct real-world experiments on two dual-arm robots. The policy observes the scene through three cameras (Figure 21): one mounted on the head and one on each wrist.

Implementation Details. We make no further architectural changes in real-world experiments. The main difference lies in the training process. We first pretrain on diverse data for more than 10,000 hours, covering 5887 different tasks collected in our real-world embodiments. We train 400k pretraining steps with a batch size of 4096. Then we fine-tune the pretrained model on 10 tasks for 30k steps with a batch size of 512. Both processes adopt exactly same loss function $( \mathcal { L } _ { f m }$ and $\mathcal { L } _ { t a s k } )$ and hyper parameters $( \lambda = 0 . 0 1$ , learning rate $5 \times 1 0 ^ { - 5 }$ , and EMA decay 0.999).

The action space is designed as:

$$
\mathbf { a } _ { t } = ( \mathbf { p } _ { l } , \mathbf { R } _ { l } , g _ { l } , \mathbf { p } _ { r } , \mathbf { R } _ { r } , g _ { r } ) \in \mathbb { R } ^ { 2 0 } ,\tag{13}
$$

where $\mathbf { p } _ { l }$ and $\mathbf { p } _ { r }$ denote the positions of the left and right end effectors, respectively, ${ \bf R } _ { l }$ and ${ \bf R } _ { r }$ denote the first two rows of the rotation matrix (Zhou et al., 2019) for the left and right end effectors, and $g _ { l }$ and $g _ { r }$ denote the gripper state. We represent actions as delta states, where each action within an action chunk denotes the relative state difference between the corresponding timestep and the chunk’s initial state. In particular, for rotational states, we compute the relative rotation on the SO(3) manifold rather than using a Euclidean representation.

For inference, we use synchronous execution, which is the simplest approach. Because our model is small enough to require about 20ms for inference, execution latency is low. The input instructions for the three tasks are directly pick cup, place the spoon inside the bowl, and plug insert, respectively.

![](images/8cc2d69a6482c408035063a6e164c0ffbedb628b5896182d0a90f063449751f0.jpg)  
Figure 20: Example CoRP rollouts on the Forge peg-insertion task. Each row shows six frames from one rollout with a different initial condition, from left to right in time; Each frame combines the main view (top) with two auxiliary views (bottom). The policy uses vision only to insert an 8.0 mm peg into an 8.1 mm hole, reaching 84% success over 50 trials.

![](images/5971a2a5df8eb97fbe1caf25920ecd30b32204d50dcd02b42929dbecc4a99d57.jpg)  
Figure 21: Example CoRP rollouts on real-world bimanual tasks. Rows show place spoon (top), pick cup (middle), and plug insertion (bottom), with six frames per rollout from left to right in time; each frame combines the head-camera view (top) with the two wrist-camera views (bottom).