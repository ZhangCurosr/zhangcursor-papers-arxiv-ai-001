. <sub>RESEARCH PAPER</sub> .

# PhysMLLMs: Spatial Priors for Unified Referring Segmentation and Grounded Reasoning of Images and Videos

Siyao Yan<sup>1</sup>, Bo Han<sup>1</sup>, Jisheng Dang<sup>1\*</sup>, Bimei Wang<sup>1</sup>, Shude Wang<sup>1</sup>, Hong Peng<sup>1\*</sup>, Yulan Guo<sup>2</sup>, Jianhuang Lai<sup>2</sup>, Bin Hu<sup>1\*</sup> & Tat-SengChua<sup>3</sup>

<sup>1</sup>School of Information Science and Engineering, Lanzhou University, Lanzhou, 730000, China <sup>2</sup>School of Electronics and Communication Engineering, Sun Yat-sen University, Shenzhen 510006, China <sup>3</sup>School of Computing, National University of Singapore, Singapore, 119077, Singapore

Abstract Video multimodal large language models support language guided video segmentation, but they often show spatio temporal inconsistencies, e.g., jitter, drift, and identity switches. These failures are more common when targets are partly hidden or when similar objects appear nearby.One likely reason is that current training lacks explicit spatial priors, which makes it dificult to maintain stable spatial identity and shape over time. We present PhysMLLMs, a training-stage prior injection architecture that injects physics-inspired spatial continuity priors into Video MLLMs. PhysMLLMs is designed to encourage more stable object-centered representations by aligning the student global visual representation with a frozen teacher model during training. Our core mechanism, Global Representation Prior Alignment (REPA-Global), distills global visual representations from a frozen DINOv2 teacher using an ofline embedding cache and a scheduled distillation plan. This design keeps inference unchanged and does not add inference time cost. Across multiple video benchmarks, PhysMLLMs improves video segmentation mask quality and cross-frame consistency, with larger gains on challenging cases involving small targets, fast motion, occlusion, distractors, and reasoning queries. On single-frame referring image segmentation and representative general VLM benchmarks, PhysMLLMs maintains comparable performance, demonstrating that the injected spatial prior improves video consistency without compromising image-level grounding or general multimodal capability. These results suggest that physics-inspired spatial prior injection can improve temporal stability while preserving general capability. The code is available at https://github.com/tusu-code/20260121-icml2026-2.git.

Keywords Language-guided video segmentation, spatiotemporal consistency, representation distillation, knowledge distillation, prior injection, parameter-eficient fine-tuning

Citation Title for citation. Sci China Inf Sci, for review

## 1 Introduction

Video multimodal large language models, or Video MLLMs, have recently shown strong semantic grounding through large scale image text alignment and instruction tuning. When combined with strong mask priors and lightweight decoders, they provide a practical backbone for language guided video segmentation [1, 2]. However, the main training signals give limited direct guidance for keeping predictions consistent over time. As a result, masks that look reasonable in each frame can still become unstable across frames, especially under occlusion, distractors, and appearance changes. Figure 1 shows a key pattern. Many failures in video segmentation come from unstable feature changes over time rather than from isolated errors in the mask decoder. This observation motivates a training strategy that guides representation dynamics toward stable object centered features, instead of relying only on stronger heads or test time rules.

We focus on three common failure patterns in practice. They include temporal jitter, drifting in identity or shape, and identity switches after occlusion. These issues reflect missing spatial priors, which weakens spatial object permanence and spatial continuity across frames. To make this failure mode more concrete, we further examine the oficial Sa2VA-InternVL3-2B baseline on the ReVOS validation split. Its overall J&F is 54.61, but the score drops to 33.65 on small-target cases and 31.29 on occlusion or disappearance cases. These decreases of 20.96 and 23.32 points show that the main dificulty is not only frame-level mask prediction, but also the preservation of stable object identity and spatial continuity under challenging video conditions. This observation motivates a training-time prior that regularizes visual representations toward temporally coherent object-centric features. We do not rely on explicit physics-based simulation or hand-crafted physical equations. Instead, we use a physics-inspired spatial consistency prior: once an object is identified, its representation should remain coherent over time unless the video provides clear evidence of a real physical change. This prior reflects object permanence, spatial continuity, and temporal coherence in the physical world. This perspective suggests that higher per frame accuracy alone is often not enough. Memory based association is a common remedy in video segmentation [3–5], but it may not fully fix instability when upstream Video MLLM features are not well regularized. In our experiments, unstable masks can remain even with memory, which suggests that the main bottleneck often lies in the representation space.

![](images/fb850c0db3ebc99764f43b9c020a58048fe8d3243429aa08270aa0e892c9d1f5.jpg)  
Figure 1 Motivation and overview of spatial prior injection for video segmentation. Unstable representations can lead to temporally inconsistent masks (left). We inject a continuity prior by distilling global visual representations from a frozen DINOv2 teacher using an ofline teacher cache and scheduled distillation weight (middle). The resulting model yields more consistent segmentations with zero inference-time overhead, since the teacher is removed at test time (right).

We propose a training-stage prior injection strategy that transfers object centered and geometry aware priors from a strong self supervised teacher, DINOv2 [6], into a Video MLLM through representation distillation [7]. Our core mechanism, REPA-Global, aligns the student global visual representation with cached DINOv2 global embeddings. This provides a representation-level regularizer that is expected to reduce abrupt cross-frame shifts while keeping inference unchanged. To make distillation scalable, we precompute teacher embeddings for each frame ofline and distill from the cache during training, which avoids repeated teacher forward passes. To reduce interference with general multimodal capability and to stabilize training, we combine a scheduled distillation weight with calibrated parameter eficient fine tuning, including Vision LoRA [8] and MaskDecoder only calibration. This design routes most injected gradients into the visual and decoding pathway while keeping the core vision language alignment largely intact.

Our contributions are summarized below.

• We propose PhysMLLMs, a training-stage prior injection strategy that injects physics-inspired spatial continuity priors into Video MLLMs to improve temporal stability for language-guided video segmentation and support more stable video reasoning behavior, while preserving the general grounding ability of the underlying vision-language model.

• We develop REPA-Global, which aligns the student global embedding with a frozen DINOv2 teacher to inject a physics-inspired spatial continuity prior that improves cross-frame consistency with no added inference-time cost.

• We make the training strategy practical at scale by combining an ofline teacher cache, a scheduled distillation plan, and a calibrated parameter eficient fine tuning recipe, which improves eficiency and reduces the risk of general capability drop during specialization.

• Extensive experiments show that PhysMLLMs consistently improves language-guided video segmentation on ReVOS, MeVIS, and Ref-DAVIS17, especially in challenging video scenarios with small targets, fast motion, occlusion, distractors, and reasoning queries. Meanwhile, it preserves competitive single-frame referring segmentation performance on RefCOCO, RefCOCO+, and RefCOCOg, and shows no clear degradation on representative general VLM benchmarks.

## 2 Related Work

## 2.1 Referring Video Object Segmentation

Referring video object segmentation extends video object segmentation by adding language guidance. It requires accurate grounding of free form expressions and masks that remain consistent across frames. Compared with class agnostic tracking, the model must resolve language ambiguity, handle multiple similar instances, and keep the target identity stable over time. Common evaluation suites include DAVIS [9] and YouTube VOS [10], as well as language guided datasets such as Ref YouTube VOS [11] and MeVIS [12]. These benchmarks cover diverse scenes and motion patterns, and they often stress occlusion, viewpoint change, and appearance variation, which can expose long range instability.

Earlier systems often improve temporal association with memory based designs, e.g., STM [3], STCN [4], and XMem [5]. Memory helps reuse past cues and reduce short term flicker, but it can still struggle when the upstream representation drifts or when the query target is visually similar to distractors. More recently, strong mask priors and large vision language models have become widely used. Segment Anything provides open world mask priors [2], and vision language models, e.g., LLaVA, improve semantic grounding [1]. In practice, these components can improve per frame mask quality and language alignment, yet long range stability remains dificult under occlusion and appearance changes. Predictions may show jitter, drift, and identity switches over time. Our PhysMLLMs architecture addresses these consistency problems at the representation level and uses a training stage prior injection strategy to improve temporal stability in video segmentation. Classical memory-based video object segmentation methods, such as STCN and XMem, provide strong temporal association when an object mask or reliable object cue is available. However, they are not directly comparable to language-guided Video-MLLM systems because they do not resolve free-form referring expressions or generate masks from open-ended language queries. We therefore discuss them as representative temporal association methods, while the main quantitative comparison focuses on language-guided Video-MLLM systems that share a closer input-output protocol.

## 2.2 Self-Supervised Spatial Priors and Geometric Structure

Self supervised pretraining can learn transferable visual priors from large scale unlabeled data. These priors are useful as teacher signals when dense video mask labels are limited and when the training distribution is diverse. Contrastive and bootstrap methods learn invariances and object centered cues [13–16]. Masked prediction methods provide complementary supervision for dense representations [17–20]. DINO style self distillation learns strong semantic features and often improves sensitivity to shape and geometry [21]. DINOv2 further strengthens these properties through scaling and data curation [6]. These properties make such teachers attractive for transferring stable global structure without requiring extra manual labels.

For video, masked autoencoding, e.g., VideoMAE, naturally uses temporal structure and can capture motion related cues [22]. Other self supervised video objectives also encourage temporal invariance, which can support smoother feature evolution even when the pixel content changes quickly. These priors provide a practical way to encourage temporally coherent features beyond frame by frame segmentation supervision. They also relate to spatial awareness because stable object identity over time depends on spatial continuity priors that standard segmentation losses do not always enforce, especially when the target becomes small, blurred, or partly hidden

## 2.3 Consistency Oriented Distillation and Representation Alignment

Knowledge distillation has moved beyond logit matching and now often uses feature level alignment and geometry preserving objectives. In this setting, the teacher provides structural guidance for student representations [23]. Supervision on intermediate features can provide steadier signals and improve spatial selectivity [24]. Relation aware and contrastive variants can also preserve the geometry of the feature space [25]. These methods are often useful when the task needs stable structure and when the student model must retain general ability while specializing.

For dense prediction, structured distillation can be more reliable than direct pixel matching when global layout and long range dependencies matter [26]. Global alignment can also reduce sensitivity to local noise, which is important when distractors trigger large false masks. These findings motivate our REPA-Global design within the PhysMLLMs architecture. We align global visual representations to transfer object centered priors that support persistent identity and coherent geometry over time [27]. Because overly strong alignment may suppress valid changes, we treat the distillation weight as a training control variable rather than a fixed constant [28]. This design supports physics-inspired spatial continuity priors, such as object permanence and temporal coherence, while avoiding an overly rigid representation.

![](images/e32768b2384a4fe0a89a028748b6bd818b306e177d7eb0722f9dc3457eb3de49.jpg)  
Figure 2 PhysMLLMs overview. During training, REPA-Global injects a physics-inspired spatial continuity prior by aligning the student global visual embedding z<sub>s</sub> with a frozen DINOv2 teacher embedding z<sub>t</sub> retrieved from an ofline cache. Text embeddings condition the LLM and downstream mask prediction, but they do not enter the global visual pooling branch before teacher alignment. At inference, the teacher branch and cache are removed.

## 2.4 Parameter Eficient Fine Tuning and Capability Preservation

Parameter eficient fine tuning adapts pretrained models while limiting representational drift, which helps preserve multimodal abilities [29]. This is important for Video MLLMs because the same model is often expected to support both segmentation and general language based reasoning. Adapter based methods and composition strategies can reduce forgetting and support safer task updates [30]. Low rank and prompt based tuning further constrains the learnable space and often improves eficiency with reduced interference [8, 31, 32]. These approaches also make it easier to control where the model changes, which matters when adding new training signals.

In our setting, Vision LoRA and MaskDecoder only calibration act as practical guardrails. They direct most injected gradients into the visual and decoding pathways and reduce unnecessary changes to vision language align ment. Capacity remains an important factor. If the parameter eficient budget is too small, the model may not absorb the injected priors well, especially when the training data contains long occlusions or fast motion. This makes module placement and rank important for performance.

## 3 Method

## 3.1 Overview

We present PhysMLLMs, a transferable prior injection architecture for large video segmentation models. PhysM-LLMs uses a training stage strategy that injects spatially grounded consistency through representation distillation. Our goal is not to rely on test time heuristics, e.g., optical flow or post processing. Instead, we guide how visual representations evolve over time during training. This design supports deployment because the teacher is used only during training and is removed at inference, so runtime cost remains unchanged, as shown in Fig.2.

In this work, the term physics-inspired prior refers to a soft constraint derived from object permanence, spatial continuity, and temporal coherence, rather than an explicit physical simulator. We implement this prior through representation-level distillation, so that the student representation is encouraged to evolve smoothly across frames while still allowing real appearance and pose changes. Given a training sample, the student predicts segmentation masks and we optimize it with the standard segmentation loss $\mathcal { L } _ { \mathrm { s e g } }$ . We further add a representation-level distillation term ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ to encode spatial priors, e.g., spatial persistence and geometric continuity of the target. We observe that temporally implausible outputs, e.g., jitter, drift, and identity switches after occlusion, can happen even when each frame looks locally correct. This often correlates with unstable feature changes across frames. We therefore impose a consistency constraint in representation space rather than using pixel level smoothing, which can depend strongly on the decoder design.

The overall objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda ( t ) \mathcal { L } _ { \mathrm { d i s t i l l } } , } \end{array}\tag{1}
$$

where t denotes the training step and $\lambda ( t )$ λ(t) controls the strength of the injected prior. We treat $\lambda ( t )$ as a training control signal rather than a fixed scalar. If the constraint becomes too strong too early, it can weaken grounding and localization and may lead to over smoothing. We therefore use a progressive schedule as part of the training strategy, described in Sec. 3.4. We implement the student on Sa2VA [33]. A pretrained multimodal backbone produces visual tokens, and a segmentation decoder predicts masks conditioned on the language query. For each frame x, we extract student visual tokens $\mathbf { S } ( x )$ from the visual encoder, pool them into a student global embedding in Eq. 5, and map it through a lightweight projector for teacher alignment in Sec. 3.2. We compute the distillation loss per frame and aggregate it over frames within a batch. This provides a representation level regularizer that complements the mask loss.

## 3.2 REPA-Global Distillation

We instantiate $\mathcal { L } _ { \mathrm { d i s t i l l } }$ with a global prior alignment term, denoted as REPA-Global. We use DINOv2 [6] with ViT B 14 as the frozen vision teacher. For an input frame x, the teacher produces patch tokens

$$
\mathbf { T } ( x ) \in \mathbb { R } ^ { N \times D } , \quad D = 7 6 8 ,\tag{2}
$$

and we compute the teacher global embedding by mean pooling

$$
\mathbf { z } _ { \mathrm { t } } ( x ) = \mathrm { M e a n P o o l } ( \mathbf { T } ( x ) ) \in \mathbb { R } ^ { D } .\tag{3}
$$

The student provides visual tokens

$$
\mathbf { S } ( x ) \in \mathbb { R } ^ { M \times C } ,\tag{4}
$$

and we obtain a student global embedding by pooling

$$
\mathbf { z } _ { \mathrm { s } } ( x ) = \mathrm { P o o l } ( \mathbf { S } ( x ) ) \in \mathbb { R } ^ { C } .\tag{5}
$$

The student embedding ${ \bf z } _ { \mathrm { s } } ( x )$ is extracted from visual tokens before text-conditioned mask prediction. Text condi tioning occurs later through the LLM-generated segmentation token and the language-conditioned mask decoder. Thus, REPA-Global injects a training-time visual prior rather than distilling a text-conditioned multimodal hidden state. We use global alignment because the targeted failures, such as jitter, drift, and identity switches, often reflect unstable object-level trajectories rather than isolated patch errors. Dense token-level imitation may also over-constrain query-irrelevant regions, so we treat token-level alignment as an ablation. Our default choice is mean pooling over the visual tokens, and we treat CLS pooling as an optional variant. Since $C \neq D ,$ , we use a lightweight projector $P ( \cdot )$ , implemented as a linear layer, to map the student embedding to the teacher dimension. We then align normalized embeddings using cosine distance:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } ^ { \mathrm { g l o b a l } } = \mathbb { E } _ { x } \left[ 1 - \cos \left( \widehat { P ( \mathbf { z } _ { \mathrm { s } } ( x ) ) } , \widehat { \mathbf { z } _ { \mathrm { t } } ( x ) } \right) \right] ,\tag{6}
$$

where both $P ( \mathbf { z } _ { \mathrm { s } } ( x ) )$ and $\mathbf { z } _ { \mathrm { t } } ( x )$ are L2-normalized before computing the cosine distance. We use cosine alignment for scale robustness, which often improves optimization stability.

As an optional variant, we also evaluate MSE on normalized embeddings. REPA-Global provides a training-time representation-level regularization signal that encourages physics-inspired object-centered features without tying the training strategy to a specific mask head.

The novelty is not the use of DINOv2 alone, but its integration with a cache-enabled training workflow, scheduled prior injection, and calibrated PEFT. This makes the teacher a training-only physics-inspired prior provider while keeping the inference graph identical to the original student.

Algorithm 1 End-to-end workflow for cache-enabled prior injection   
Require: Training frames $\{ x \}$ ; frozen teacher $\mathbf { T } ;$ student $\mathbf { S } _ { \boldsymbol { \theta } } ;$ schedule $\lambda ( t )$ ; max weight $\lambda _ { \mathrm { m a x } }$   
Ensure: Trained student parameters θ for deployment   
1: Cache generation (ofline)   
2: for all training frame x do   
3: compute teacher global embedding $\mathbf { z } _ { \mathrm { t } } ( x )$   
4: store $\mathbf { z } _ { \mathrm { t } } ( x )$ in the teacher cache   
5: end for   
6: Warmup   
7: for training step $t < t _ { \mathrm { w } }$ do   
8: set $\lambda ( t ) \gets 0$   
9: update θ by minimizing $\mathcal { L } _ { \mathrm { s e g } }$   
10: end for   
11: Ramp and hold   
12: for training step $t \geqslant t _ { \mathrm { w } }$ do   
13: read cached $\mathbf { z } _ { \mathrm { t } } ( x )$ for frames in the current batch   
14: compute $\mathcal { L } _ { \mathfrak { c } }$ distill   
15: optimize $\mathcal { L } _ { \mathrm { s e g } } + \lambda ( t ) \mathcal { L } _ { \mathrm { d i s t i l l } }$   
16: update only calibrated PEFT parameters   
17: end for   
18: Deployment   
19: discard the teacher and cache; run inference with the student only

## 3.3 Calibrated PEFT for Prior Injection

Injecting priors through distillation can interact with multimodal alignment. To limit unwanted drift and keep deployment simple, we use a parameter eficient tuning strategy. We freeze the language model and most pretrained modules, and we update only two parts. We enable Vision LoRA [8] in the visual backbone and we calibrate only the mask decoder on the segmentation side.

For a selected linear weight matrix $\mathbf { W } _ { 0 }$ , LoRA parameterizes the update as

$$
\mathbf { W } = \mathbf { W } _ { 0 } + \Delta \mathbf { W } , \quad \Delta \mathbf { W } = \mathbf { B } \mathbf { A } .\tag{7}
$$

In this setup, the base matrix $\mathbf { W } _ { 0 }$ stays frozen and the model learns only the low rank factors A and B. This constrained update helps preserve semantic grounding while allowing the injected prior to shape visual features that afect temporal stability. A practical limitation is capacity. If the parameter eficient budget is too small, the model may not absorb the spatial priors well, so module placement and rank become important hyperparameters.

## 3.4 Overall Training Workflow for Cache, Scheduling, and Deployment

Stable prior injection requires consistent teacher supervision and controlled regularization strength. As summarized in Algorithm 1, we first precompute the teacher global embedding $\mathbf { z } _ { \mathrm { t } } ( x )$ for each training frame and store it in an ofline cache. During training, the student reads cached teacher embeddings to compute ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ , which avoids repeated teacher forward passes and keeps the distillation target deterministic.

We treat λ(t) as a control variable and use a warmup-ramp-hold schedule:

$$
\begin{array} { r } { \lambda ( t ) = \left\{ \begin{array} { l l } { 0 , } & { t < t _ { \mathrm { w } } , } \\ { \lambda _ { \mathrm { m a x } } \cdot \frac { t - t _ { \mathrm { w } } } { t _ { \mathrm { r } } - t _ { \mathrm { w } } } , } & { t _ { \mathrm { w } } \leqslant t < t _ { \mathrm { r } } , } \\ { \lambda _ { \mathrm { m a x } } , } & { t \geqslant t _ { \mathrm { r } } , } \end{array} \right. } \end{array}\tag{8}
$$

where $t _ { \mathrm { w } }$ and $t _ { \mathrm { r } }$ denote the end of warmup and ramp. The warmup stage optimizes only the segmentation objective, while the scheduled stage introduces the physics-inspired prior through ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ and updates only calibrated PEFT parameters. At inference, the teacher and cache are removed, and only the student model is used.

Table 1 Comparison with recent Video-MLLM systems on language-guided video segmentation. We report DAVIS-style J&F on MeVIS U, ReVOS, and Ref-DAVIS17 because most published Video-MLLM papers report aggregate J&F but do not consistently provide separate $\mathcal { T }$ and F values across all datasets. Higher is better. “–” means not reported or not available under a comparable protocol.
<table><tr><td>Method</td><td>MeVIS U</td><td>ReVOS</td><td>Ref-DAVIS17</td></tr><tr><td>PG-Video-LLaVA [34]</td><td>18.9</td><td></td><td></td></tr><tr><td>GLaMM + SAM2 [35]</td><td>38.7</td><td></td><td></td></tr><tr><td>VideoGLaMM-3.8B [35]</td><td>45.2</td><td></td><td></td></tr><tr><td>VISA-13B [36]</td><td>44.5</td><td>50.9</td><td>70.4</td></tr><tr><td>VideoLISA-3B [37]</td><td>44.4</td><td></td><td>68.8</td></tr><tr><td>HyperSeg-3B [38]</td><td></td><td>55.7</td><td>71.2</td></tr><tr><td>InstructSeg [39]</td><td></td><td>54.5</td><td>71.1</td></tr><tr><td>ViLLa-InternVideo2-6B [40]</td><td>49.4</td><td>57.0</td><td>74.3</td></tr><tr><td>GLUS-7B/GLUS-ED [41]</td><td>51.3</td><td>54.9</td><td></td></tr><tr><td>Sa2VA-8B [33]</td><td>46.2</td><td>53.6</td><td>73.8</td></tr><tr><td>Sa2VA-InternVL3-2B [33]</td><td>53.9</td><td>56.2</td><td>74.5</td></tr><tr><td>PhysMLLMs dw012</td><td>55.1</td><td>57.4</td><td>76.0</td></tr></table>

## 4 Experiments

We evaluate whether PhysMLLMs improves language-guided video segmentation while preserving image-level grounding and general VLM capability. We first clarify the setup and evaluation protocol, then compare with recent Video-MLLMs, analyze the efect of prior injection, and summarize capability-preservation results.

## 4.1 Experimental Setup and Evaluation Protocol

Unless otherwise stated, PhysMLLMs is built on Sa2VA-InternVL3-2B, which combines an InternVL3-2B multimodal backbone with the SAM2 segmentation module. The full student model contains approximately 2B parameters. All controlled comparisons and ablations use this same 2B backbone, so the observed diferences come from REPA-Global rather than model scale. Table 1 includes Sa2VA-8B only as an external reference.

We use a frozen DINOv2 ViT-B/14 teacher and precompute frame-level teacher embeddings into an ofline cache. During inference, the teacher and cache are removed. To preserve vision-language alignment, we freeze the language model and most pretrained modules, and update only Vision-LoRA and mask-decoder calibration parameters. The settings dw008, dw012, and dw020 denote peak distillation weights $\lambda _ { \operatorname* { m a x } } = 0 . 0 0 8 , 0 . 0 1 2$ , and 0.020 under the same warmup-ramp-hold schedule.

For video segmentation, we report DAVIS-style J, F, and J&F when available, and use $\mathcal { T } \& \mathcal { F }$ as the main metric for cross-method comparison because many published Video-MLLM systems do not report separate $\mathcal { I }$ and F. For image grounding, we report RefCOCO-series accuracy as a non-degradation check. For general VLM capability, we evaluate MMBench, MME, POPE, and TextVQA, with detailed results provided in the appendix.

## 4.2 Main Comparison with Recent Video-MLLM Systems

Table 1 positions PhysMLLMs relative to representative recent Video-MLLM systems on MeVIS U, ReVOS, and Ref-DAVIS17, when the corresponding numbers are available. To reflect recent progress, we include recently reported Video-MLLM segmentation systems such as HyperSeg-3B and InstructSeg when their published results are available. The comparison covers a range of model sizes and training strategies, and it includes the Sa2VA baseline that our method builds upon. PhysMLLMs improves consistently across the reported benchmarks, indicating that representation-level prior injection is not tied to a single dataset or evaluation setup.

It is worth noting that Table 1 contains models with diferent backbone capacities, including Sa2VA-8B and Sa2VA-InternVL3-2B. Therefore, this table is used to position PhysMLLMs among recent Video-MLLM systems, rather than to serve as the only evidence for a controlled capacity-matched comparison. The controlled comparisons are conducted under the same Sa2VA-InternVL3-2B + SAM2 backbone in the following ablation and capabilitycheck analyses.Some compared methods report results on only one or two datasets. We keep these entries as “–” rather than re-estimating them under unmatched settings, because diferences in model size, training data, and evaluation protocol may otherwise confound the comparison. This reporting follows the published availability of each method and avoids introducing non-comparable numbers.

We use the aggregate J&F score in Table 1 to maximize coverage across published Video-MLLM systems. Separate $\mathcal { I }$ and F values are reported in controlled internal analyses when they are available, such as Table 2.

Table 2 Ablation of global and token-level DINOv2 alignment on ReVOS under the same Sa2VA-InternVL3-2B + SAM2 backbone. Globalonly REPA-Global outperforms token-only and global+token variants, suggesting that dense patch-level constraints may interfere with languageconditioned target selection.
<table><tr><td>Setting</td><td>ReVOS J</td><td>ReVOS F</td><td>ReVOS J&amp;F</td></tr><tr><td>Global-only dw012</td><td>54.31</td><td>60.51</td><td>57.41</td></tr><tr><td>DINOv2 token-only</td><td>53.32</td><td>59.21</td><td>56.26</td></tr><tr><td>DINOv2 global+token</td><td>53.01</td><td>59.12</td><td>56.07</td></tr></table>

Table 3 Ablation of teacher-prior injection under the same Sa2VA-InternVL3-2B + SAM2 backbone. The symbols dw008, dw012, and dw020 denote peak distillation weights $\lambda _ { \operatorname* { m a x } } = 0 . 0 0 8$ , 0.012, and 0.020 under the same warmup-ramp-hold schedule. Higher J&F is better.
<table><tr><td>Setting</td><td>KD</td><td>ReVOS J&amp;F</td><td>MeVIS U J&amp;F</td></tr><tr><td>baseline</td><td>w/o</td><td>56.06</td><td>53.92</td></tr><tr><td>dw008</td><td>w/</td><td>56.88</td><td>54.51</td></tr><tr><td>dw008</td><td>w/o</td><td>56.46</td><td>54.42</td></tr><tr><td>dw012</td><td>w/</td><td>57.41</td><td>55.12</td></tr><tr><td>dw012</td><td>w/o</td><td>56.76</td><td>54.45</td></tr><tr><td>dw020</td><td>w/</td><td>56.87</td><td>55.24</td></tr><tr><td>dw020</td><td>w/o</td><td>56.66</td><td>54.32</td></tr></table>

Table 4 Complexity-stratified analysis on ReVOS. Here, n denotes the number of ReVOS validation expressions in each stratum, rather than videos or frames. We report the oficial Sa2VA-InternVL3-2B result, the PhysMLLMs dw012 result, and the absolute J&F improvement. “Hard union” includes expressions satisfying at least one challenging condition.
<table><tr><td>Stratum</td><td>n</td><td>Official J&amp;F</td><td>Ours dw012</td><td>∆</td></tr><tr><td>All</td><td>5822</td><td>54.61</td><td>55.91</td><td>+1.30</td></tr><tr><td>Small target</td><td>1459</td><td>33.65</td><td>35.91</td><td>+2.26</td></tr><tr><td>Fast motion</td><td>1457</td><td>49.57</td><td>50.94</td><td>+1.37</td></tr><tr><td>Occlusion or disappearance</td><td>1459</td><td>31.29</td><td>33.22</td><td>+1.92</td></tr><tr><td>Distractor-heavy</td><td>1460</td><td>55.95</td><td>57.97</td><td>+2.02</td></tr><tr><td>Reasoning query</td><td>2475</td><td>51.74</td><td>53.50</td><td>+1.76</td></tr><tr><td>Hard union</td><td>4699</td><td>50.47</td><td>52.07</td><td>+1.59</td></tr></table>

## 4.3 Efect of Prior Injection

Teacher-prior injection is the dominant source of gain. Table 3 isolates the efect of the teacher spatial prior by toggling distillation while keeping the rest of the training recipe unchanged. Across the tested settings, enabling distillation improves segmentation on ReVOS and transfers to MeVIS U. This pattern suggests that the gains are driven by the injected physics-inspired spatial prior rather than incidental optimization efects. The improvements on MeVIS U are particularly informative because this split difers from the training distribution, indicating that the continuity constraint strengthens generalization in addition to improving in-domain performance.

The gains are concentrated in dificult video scenarios. To better understand why the gains are more pronounced in video segmentation than in single-frame referring tasks, we conduct a complexity-stratified analysis on ReVOS. As shown in Table 4, PhysMLLMs improves over the oficial Sa2VA-InternVL3-2B model by +2.26 J&F on small targets, +1.92 on occlusion or disappearance cases, +2.02 on distractor-heavy cases, +1.37 on fast-motion cases, +1.76 on reasoning queries, and +1.59 on the hard union subset. These scenarios directly stress temporal identity preservation, cross-frame spatial continuity, and robustness to visual distractors. In contrast, RefCOCO is a single-frame benchmark where the oficial Sa2VA-InternVL3-2B baseline is already strong and no temporal identity maintenance is required. We therefore use the RefCOCO-series results mainly as non-degradation checks for image-level grounding, rather than as the primary target of the proposed video-oriented prior. The remaining gap on small-target and occlusion-heavy cases also suggests that extremely small or persistently occluded objects remain challenging, which clarifies the current applicability boundary of PhysMLLMs.

Global alignment is more stable than token-level alignment. We further evaluate whether DINOv2 token-level teacher support can improve fine-grained video segmentation. As shown in Table 2, token-only alignment achieves 56.26 J&F and global+token alignment achieves 56.07 J&F, both lower than the global-only REPA-Global objective with 57.41 J&F. Although DINOv2 provides strong patch-level semantic features, directly imposing dense token-level constraints may over-regularize the student visual tokens and interfere with language-conditioned target selection. Therefore, we adopt global representation alignment as a more stable training-time prior.

The current evidence supports a mechanism-oriented interpretation. Although we do not include a separate feature-trajectory probe, the current evidence is consistent with the intended representation-level mechanism. The KD on/of comparison shows that the gain comes from the teacher-prior term rather than only from the training recipe. The global-versus-token alignment ablation further shows that a compact global constraint is more efective than dense token-level imitation for language-conditioned video segmentation. Together with the temporal stability proxy in the appendix, these results suggest that REPA-Global acts as a representation-level temporal regularizer, rather than only as a mask-head adjustment.

![](images/13e09939dbb5f5f472db1eafb26b5d18889b6f78414fa7370b5e59a5ef44ec67.jpg)  
Figure 3 Qualitative visual video QA examples. Each panel shows the input video, the question, and the model response, illustrating that the injected priors do not noticeably degrade multimodal reasoning on representative cases.

Table 5 Unified evaluation across image grounding, video segmentation, and MMBench under the InternVL3-2B + SAM2 backbone. We report RefCOCO accuracy, ReVOS and MeVIS U J&F, and MMBench score. Higher is better.
<table><tr><td>Run ID</td><td>RefCOCO</td><td>ReVOS</td><td>MeVIS U</td><td>MMBench</td></tr><tr><td>baseline</td><td>81.42%</td><td>56.06</td><td>53.92</td><td>0.7876</td></tr><tr><td>dw012</td><td>81.61%</td><td>57.41</td><td>55.12</td><td>0.7878</td></tr><tr><td>dw008</td><td>81.73%</td><td>56.88</td><td>54.51</td><td>0.7878</td></tr><tr><td>dw020</td><td>81.68%</td><td>56.87</td><td>55.24</td><td>0.7878</td></tr></table>

## 4.4 Default Configuration and General Grounding

Table 5 summarizes the default configuration across image grounding, video segmentation, and MMBench. The default dw012 setting gives the strongest ReVOS performance and competitive MeVIS U transfer, while RefCOCO and MMBench remain close to the baseline. Additional MME, POPE, TextVQA, RefCOCO-series, and distillationweight sensitivity results are reported in the appendix. These results indicate that the physics-inspired prior improves video consistency without causing a clear collapse in image-level grounding or representative general VLM capability.

## 4.5 Limitations and Failure Cases

PhysMLLMs still has several limitations. As shown in the appendix failure cases, dense similar distractors, persistent occlusion, and extremely small targets can still cause incorrect localization or identity ambiguity. Although Table 4 shows gains on small-target and occlusion/disappearance strata, their absolute J&F scores remain much lower than the full validation average. This indicates that these scenarios are not fully solved.

The limitation mainly comes from the global nature of REPA-Global. The injected physics-inspired prior encourages object permanence, spatial continuity, and temporal coherence through a soft representation-level regularizer, but it does not explicitly model physical forces, 3D dynamics, object interactions, or query-specific target identity. Direct feature-trajectory verification also remains future work. These limitations suggest that query-aware or mask-aware prior injection may further improve fine-grained video grounding.

Table 6 General VLM capability check on MME, POPE, and TextVQA. All rows use the same Sa2VA-InternVL3-2B backbone and compare checkpoints from diferent calibration stages rather than diferent model backbones. Higher is better.
<table><tr><td>Checkpoint / training stage</td><td>MME perception</td><td>MME reasoning</td><td>MME total</td><td>POPE F1</td><td>TextVQA val acc</td></tr><tr><td>Sa2VA-InternVL3-2B official</td><td>1630.838</td><td>520.357</td><td>2151.195</td><td>87.421</td><td>76.808</td></tr><tr><td>RefCOCO-ft init</td><td>1626.126</td><td>536.071</td><td>2162.198</td><td>87.405</td><td>76.580</td></tr><tr><td>ReVOS calib it200</td><td>1610.827</td><td>522.500</td><td>2133.327</td><td>87.575</td><td>76.876</td></tr><tr><td>ReVOS calib it500</td><td>1602.206</td><td>527.857</td><td>2130.064</td><td>87.542</td><td>76.796</td></tr><tr><td>ReVOS calib it500 + KD dw012</td><td>1622.166</td><td>530.714</td><td>2152.880</td><td></td><td></td></tr></table>

## 5 Conclusion

PhysMLLMs is a training-stage prior injection strategy for Video-MLLM-based language-guided video segmentation. It targets common temporal failures, including jitter, drift, and identity switches that often appear under occlusion and distractors. The key idea is to inject a physics-inspired spatial continuity prior by aligning the student global visual representation with a frozen DINOv2 teacher through REPA-Global. Teacher supervision is delivered through an ofline embedding cache, which makes distillation scalable and keeps inference unchanged because the teacher is removed at test time. We further combine scheduled distillation weighting with calibrated parameter-eficient fine-tuning, which improves training stability and helps preserve general grounding ability while improving mask quality and cross-frame consistency.

The complexity-stratified ReVOS analysis further shows that the gains are concentrated in dificult video scenar ios, including small targets, fast motion, occlusion or disappearance, distractor-heavy scenes, and reasoning queries. This supports our motivation that the injected prior mainly addresses temporal identity and spatial continuity. Meanwhile, the RefCOCO-series and general VLM results indicate that the proposed training-time prior preserves image-level grounding and representative multimodal capability.

At the same time, the current evidence mainly supports the proposed mechanism through controlled ablations, challenging-scenario analysis, and temporal stability proxies. Direct feature-trajectory analysis remains future work, especially for query-specific cases with persistent occlusion, extremely small targets, or dense similar distractors.

Acknowledgements This work was supported by the National Natural Science Foundation of China (Grants No. 62227807 and U24B20186).   
This work was also supported by the Supercomputing Center of Lanzhou University.

## Appendix A. Additional Experimental Results

This appendix provides supplementary quantitative results that support the main experimental conclusions, including general VLM capability checks, RefCOCO-series referring image segmentation comparisons, and distillationweight sensitivity analysis. These results are moved from the main text to keep the core experimental section concise while preserving the detailed evidence for capability preservation and hyperparameter sensitivity.

## A.1 General VLM Capability Check

All rows in Table 6 are based on the same Sa2VA-InternVL3-2B backbone. The listed entries denote checkpoints obtained at diferent calibration stages, including the oficial checkpoint, the RefCOCO fine-tuned initialization, ReVOS-calibrated checkpoints, and the ReVOS-calibrated checkpoint with REPA-Global KD. Table 6 reports additional general VLM capability checks on MME, POPE, and TextVQA. The ReVOS calib it500 + KD dw012 checkpoint achieves an MME total score of 2152.880, which is comparable to the oficial Sa2VA-InternVL3-2B score of 2151.195. The nearby ReVOS-calibrated checkpoints also remain stable on POPE and TextVQA. These results suggest that the video-oriented calibration path and REPA-Global injection do not cause an obvious collapse in representative general VLM capability.

## A.2 RefCOCO-Series Referring Image Segmentation

Table 7 reports the RefCOCO-series comparison with representative fine-tuned models. PhysMLLMs remains competitive on RefCOCO, RefCOCO+, and RefCOCOg. Since the proposed spatial prior is designed mainly for video consistency, these results are used as non-degradation checks for image-level referring capability.

Table 7 Referring image segmentation comparison with representative fine-tuned models. We report validation accuracy on RefCOCO, RefCOCO+, and RefCOCOg. Higher is better.
<table><tr><td>Model</td><td>RefCOCO</td><td>RefCOCO+</td><td>RefCOCOg</td></tr><tr><td>LAVT [42]</td><td>72.7</td><td>62.1</td><td>61.2</td></tr><tr><td>GLaMM-7B [43]</td><td>79.5</td><td>72.6</td><td>74.2</td></tr><tr><td>OMG-LLaVA-7B [44]</td><td>78.0</td><td>69.1</td><td>72.9</td></tr><tr><td>F-LMM-7B [45]</td><td>76.1</td><td>65.2</td><td>68.5</td></tr><tr><td>Sa2VA-InternVL3-2B [33]</td><td>81.4</td><td>75.7</td><td>80.3</td></tr><tr><td>PhysMLLMs</td><td>81.9</td><td>76.3</td><td>80.4</td></tr></table>

Table 8 Sensitivity to the peak distillation weight under a fixed warmup-ramp-hold schedule. The column “Peak weight $\lambda _ { \mathrm { m a x } } ? $ denotes the maximum scheduled KD weight. Higher tIoU mean and lower tIoU variance indicate stronger temporal stability.
<table><tr><td>Setting</td><td>Peak weight  $\lambda _ { \mathrm { m a x } }$ </td><td>ReVOS J&amp;F ↑</td><td>tIoU mean ↑</td><td>tIoU var ↓</td></tr><tr><td>dw0002</td><td>0.002</td><td>56.64</td><td>0.6461</td><td>0.0841</td></tr><tr><td>dw004</td><td>0.004</td><td>56.53</td><td>0.6452</td><td>0.0843</td></tr><tr><td>dw006</td><td>0.006</td><td>56.59</td><td>0.6396</td><td>0.0854</td></tr><tr><td>dw008</td><td>0.008</td><td>56.88</td><td>0.6476</td><td>0.0839</td></tr><tr><td>dw010</td><td>0.010</td><td>56.66</td><td>0.6514</td><td>0.0824</td></tr><tr><td>dw012</td><td>0.012</td><td>57.41</td><td>0.6419</td><td>0.0854</td></tr><tr><td>dw015</td><td>0.015</td><td>56.82</td><td>0.6451</td><td>0.0868</td></tr><tr><td>dw020</td><td>0.020</td><td>56.87</td><td>0.6505</td><td>0.0817</td></tr></table>

## A.3 Distillation-Weight Sensitivity

Table 8 reports the sensitivity to the peak distillation weight under the same warmup–ramp–hold schedule. Moderate distillation gives the best ReVOS J&F, while stronger distillation tends to improve the temporal stability proxy with a small cost in mask quality. This supports using dw012 as the default setting in the main experiments.

## A.4 Qualitative Success Cases

Figure 4 provides additional qualitative comparisons on LV-VIS. These examples cover human-object interaction, multiple similar instances, small or thin objects, and reasoning-style language queries. The baseline often produces unstable masks, such as over-segmented regions under distractors, missing small objects, or identity drift after partial occlusion. PhysMLLMs produces more coherent masks across frames and more accurate target localization in these examples. This supports the quantitative finding that the physics-inspired prior mainly improves temporal consistency and target identity preservation.

## A.5 Failure Cases

Figure 5 shows representative failure cases of PhysMLLMs. Although the proposed physics-inspired prior improves temporal consistency in many challenging scenarios, it can still fail under dense similar distractors, persistent occlusion, and extremely small targets.

![](images/e52583904cc27cf6d3ea32a359f93e28650396cef41ce3c00ebe5aa8ae7ce2ad.jpg)

![](images/28c4a4bde39b0b89539b45d5ff4cc596950e5441243ca2b8267c83ef734ac740.jpg)

![](images/87127e10114cdf27fa8569806aa4bfdc47f5997623d7e911122603fe5617afa0.jpg)

![](images/1410f301c657952dd803d74dacd9127d70e6bff5dace3a46be7faf7d8315b43f.jpg)

![](images/80fe7a8c562afc5801892c2c3347ee87e3552e2e7b4c00884f242b483c4b0cc5.jpg)

![](images/2a3e6ad5320b99578eee07adaac91d8ee99a7abeff1a152bcd7cac8554181a09.jpg)

![](images/0838ff1f4b7a17d703b0c8dc350ae2985d5445c264ae720dbdc4f1fd17e49e66.jpg)

![](images/95a50af77a25ef3574e0d3b184ee1c7d9265b2cd59520bd1904e5a64729daa3f.jpg)

![](images/95eb32ba89522615628bed430bd3550e7422f68507580797bd36e4ffeebcf97d.jpg)

![](images/118c4b3edffbede5a92aa01c5c7c88fbdec5f3ef26e9bb950e91569a36e675a7.jpg)

![](images/18712f1c869279495a0ffff539d39779d409a0e7c458ff86e0d10d957260b506.jpg)

![](images/582185a8528e8f3e9a37ccc85ec07319bc67a4db374052782dc72662be20ee68.jpg)  
(a)

![](images/9c38215e9b1ad6a548a2f78887146f05d0f9dd85d56869c83032f6741a513c28.jpg)

![](images/54035f2852c0a3df6fa2a31d2000497de048eae72bb3bb249995205bea33b058.jpg)

![](images/6ca0b38effeb9b3f56f76990dd30aa6d4ed45b135ebfea19f4c880aaf67aba1d.jpg)  
(b)

![](images/aaf7956ad4452b47cb2c30a4e9bc57a6af731a3865e8786ac339d4cdf1c2b25c.jpg)  
(c)

![](images/4d16dc1cdb9f72c301cde9bf56f3eb89c6e9051a555946c25cf1f2e0adebd224.jpg)

(d)  
![](images/e0db319b9497af7b141e9a3ddc3bd910e8af46ecb82a9d6e6ca2ea2579dab079.jpg)

![](images/873cc9a62ce89a5de53b16ce489f90bdfe71e91d56e6f3cdea31b218f2b63473.jpg)  
(e)

![](images/90cffa379f76463d013f39520b2c56ada61dcdc95692568ad58798f5d56bf4d9.jpg)

![](images/044c3bf9ffbb89abe074e1ae041dd6c0b17e97a3899b58b09dba954583f4dafd.jpg)  
(f)  
Figure 4 Qualitative comparison on LV-VIS for language-guided video segmentation. The top three rows show the baseline results and the bottom three rows show PhysMLLMs results. Red boxes highlight baseline failures, and blue boxes highlight improved predictions by PhysMLLMs. The segmentation prompts are: (a) “Please segment a person holding a mop.” (b) “Which creatures are primates? Please answer using segmentation masks.” (c) “Please segment the object(s) carrying ammunition.” (d) “Which gray mat(s) have patterns? Please respond with a segmentation mask.” (e) “Please segment the object fanning its wings on the koala’s snout.” (f) “Please segment the aircraft that is most likely to be in a state of lacking fuel.” Overall, PhysMLLMs improves mask quality and cross-frame consistency under occlusion, distractors, and reasoning-style referring expressions.

## Dense Similar Distractors

## Query: Persons sitting down holding phone

image  
![](images/573610b9750f8f815c25bd9b622add842d12913d772232dbe61b71d1994ddf20.jpg)

ground truth  
![](images/2eabaf600515b2e49db7c6e0ad4c38ff045e47123687ab9e20c5ce4a15d91be1.jpg)  
Persistent Occlusion

baseline  
![](images/ade8e567124871a403423e7a08d3f61e73432e1a5ffd015415d188fee424e32d.jpg)

ours + failure  
![](images/adb41d50f81fa2929d058fd5d6eacded00f7a3ef9e5463e5f0e42eede9ed7038.jpg)

![](images/03c7b9393939b2a27cd4d498a1f2f758ad765c037daa1efe205c20529d1cc246.jpg)  
image

![](images/7f6a5d302de728c85f179d0b1d53fa1154a1344829083485aa0fd7f663a3978e.jpg)

ground truth  
![](images/cc71c685e27f2a47ca4b2fd40d25d07832e7b62348883516c1d4a480982e5e1c.jpg)

baseline  
![](images/3220fdcde9677f29db7c517b85def785d510f700d5ace80d4c61484d846f67ad.jpg)

ours + failure  
![](images/f13239e8c784ae7fc1c08300bc6185fda4b241d4925b51f790b4bae9e66ef7a5.jpg)

image  
![](images/afcaa3526758dba94786b5ec1bd5f149377cc0ce0f0d306863fc6045f6a442d6.jpg)

ground truth  
![](images/3710c71863aa9959e81d5c96c292f464ba4b1e10b23003cced3a06dc63191b9c.jpg)

baseline  
![](images/c4ff0d45daaaaa1cb01b0632097a267e7f1145e8c7de47826f2f11df3a8c7855.jpg)

ours + failure  
![](images/e3da33b21333518d0ad63fab3ddc45c2697059d78df2c0a8b87e3e6dc9594741.jpg)  
Red = false positive, purple = false negative.  
Figure 5 Representative failure cases of PhysMLLMs. Each row shows the input image, ground truth, baseline prediction, and our prediction. The three rows correspond to dense similar distractors, persistent occlusion, and small target cases. Red regions indicate false positives, and purple regions indicate false negatives. These examples show that global prior alignment improves temporal stability but may still be insuficient for query-specific identity disambiguation under highly ambiguous visual conditions.

## References

1 Liu H, Li C, Wu Q, et al. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023

2 Kirillov A, Mintun E, Ravi N, et al. Segment anything. In: Proceedings of IEEE/CVF International Conference on Computer Vision, 2023

3 Oh S W, Lee J Y, Xu N, et al. Video object segmentation using space-time memory networks. In: Proceedings of IEEE/CVF International Conference on Computer Vision, 2019

4 Cheng H K, Tai Y W, Tang C K. Rethinking space-time networks with improved memory coverage for eficient video object segmentation. In: Proceedings of IEEE/CVF International Conference on Computer Vision, 2021

5 Cheng H K, Oh S W, Price B, et al. XMem: Long-term video object segmentation with an atkinson-shifrin memory model. In: Proceedings of European Conference on Computer Vision, 2022

6 Oquab M, Darcet T, Moutakanni T, et al. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023

7 Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015

8 Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models. In: Proceedings of International Conference on Learning Representations, 2022

9 Pont-Tuset J, Perazzi F, Caelles S, et al. The 2017 DAVIS challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017

10 Xu N, Yang L, Fan Y, et al. YouTube-VOS: A large-scale video object segmentation benchmark. In: Proceedings of European Conference on Computer Vision, 2018

11 Seo S, Lee J Y, Han B. Ref-YouTube-VOS: A large-scale video object segmentation benchmark for referring expressions. arXiv preprint, 2020

12 Ding H, et al. MeVIS: A large-scale benchmark for video segmentation with motion expressions. arXiv preprint arXiv:2308.03772, 2023

13 Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual representations. In: Proceedings of International Conference on Machine Learning, 2020

14 He K, Fan H, Wu Y, et al. Momentum contrast for unsupervised visual representation learning. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020

15 Grill J B, Strub F, Altche F, et al. Bootstrap your own latent: A new approach to self-supervised learning. In: Proceedings of Advances in Neural Information Processing Systems, 2020

16 Caron M, Misra I, Mairal J, et al. Unsupervised learning of visual features by contrasting cluster assignments. In: Proceedings of Advances in Neural Information Processing Systems, 2020

17 He K, Chen X, Xie S, et al. Masked autoencoders are scalable vision learners. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022

18 Bao H, Dong L, Piao S, et al. BEiT: BERT pre-training of image transformers. In: Proceedings of International Conference on Learning Representations, 2022

19 Zhou J, Wei C, Wang H, et al. iBOT: Image BERT pre-training with online tokenizer. In: Proceedings of International Conference on Learning Representations, 2022

20 Wei C, Xie C, Kong T, et al. Masked feature prediction for self-supervised visual pre-training. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022

21 Caron M, Touvron H, Misra I, et al. Emerging properties in self-supervised vision transformers. In: Proceedings of IEEE/CVF International Conference on Computer Vision, 2021

22 Tong Z, Song Y, Wang J, et al. VideoMAE: Masked autoencoders are data-eficient learners for self-supervised video pre-training. arXiv preprint arXiv:2203.12602, 2022

23 Gou J, Yu B, Maybank S J, et al. Knowledge distillation: A survey. International Journal of Computer Vision, 2021, 129: 1789–1819

24 Romero A, Ballas N, Kahou S E, et al. FitNets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2015

25 Park W, Kim D, Lu Y, et al. Relational knowledge distillation. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019

26 Liu Y, Shun C, Wang J, et al. Structured knowledge distillation for dense prediction. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019

27 Touvron H, Cord M, Douze M, et al. Training data-eficient image transformers and distillation through attention. In: Proceedings of International Conference on Machine Learning, 2021. 10347–10357

28 Han Z, Gao C, Liu J, et al. Parameter-eficient fine-tuning for large models: A comprehensive survey. arXiv preprint arXiv:2403.14608, 2024

29 Houlsby N, Giurgiu A, Jastrzebski S, et al. Parameter-eficient transfer learning for NLP. In: Proceedings of Chaudhuri K, Salakhutdinov R, editors, Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019. 2790–2799

30 Pfeifer J, Kamath A, R¨uckl´e A, et al. Adapterfusion: Non-destructive task composition for transfer learning, 2021

31 Li X L, Liang P. Prefix-tuning: Optimizing continuous prompts for generation. In: Proceedings of Zong C, Xia F, Li W, et al., editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), Online: Association for Computational Linguistics, 2021. 4582–4597

32 Lester B, Al-Rfou R, Constant N. The power of scale for parameter-eficient prompt tuning. In: Proceedings of Moens M F, Huang X, Specia L, et al., editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021. 3045–3059

33 Yuan H, Li X, Zhang T, et al. Sa2VA: Marrying SAM2 with LLaVA for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025

34 Munasinghe S, Thushara R, Maaz M, et al. PG-Video-LLaVA: Pixel grounding large video-language models. arXiv preprint arXiv:2311.13435, 2023

35 Munasinghe S, Gani H, Zhu W, et al. VideoGLaMM: A large multimodal model for pixel-level visual grounding in videos. arXiv preprint arXiv:2411.04923, 2024

36 Yan C, Wang H, Yan S, et al. VISA: Reasoning video object segmentation via large language models. In: Proceedings of European Conference on Computer Vision, 2024

37 Bai Z, He T, Mei H, et al. One token to seg them all: Language instructed reasoning segmentation in videos. In: Proceedings of Advances in Neural Information Processing Systems, 2024

38 Wei C, Zhong Y, Tan H, et al. HyperSeg: Towards universal visual segmentation with large language model. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025

39 Wei C, Zhong Y, Tan H, et al. InstructSeg: Unifying instructed visual segmentation with multi-modal large language models. arXiv preprint arXiv:2412.14006, 2024

40 Zheng R, Qi L, Chen X, et al. Villa: Video reasoning segmentation with large language model. In: Proceedings of IEEE/CVF International Conference on Computer Vision, 2025. 23667–23677

41 Lin L, Yu X, Pang Z, et al. Glus: Global-local reasoning unified into a single large language model for video segmentation. In: Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 8658–8667

42 Yang Z, Wang J, Tang Y, et al. LAVT: Language-aware vision transformer for referring image segmentation. In: Proceedings of IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 2022

43 Rasheed H, Maaz M, Mullappilly S S, et al. GLaMM: Pixel grounding large multimodal model. arXiv preprint arXiv:2311.03356, 2024

44 Zhang T, Li X, Fei H, et al. OMG-LLaVA: Bridging image-level, object-level, pixel-level reasoning and understanding. arXiv preprint arXiv:2406.19389, 2024

45 Wu S, Jin S, Zhang W, et al. F-LMM: Grounding frozen large multimodal models. arXiv preprint arXiv:2406.05821, 2025