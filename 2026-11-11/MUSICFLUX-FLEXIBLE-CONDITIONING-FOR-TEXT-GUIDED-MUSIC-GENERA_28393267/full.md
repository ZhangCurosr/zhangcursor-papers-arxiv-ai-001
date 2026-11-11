# MUSICFLUX: FLEXIBLE CONDITIONING FOR TEXT-GUIDED MUSIC GENERATION AND EDITING

Teysir Baoueb<sup>\*</sup> Leonardo Boulitreau de M. M. Marques<sup>\*</sup> Paraskevas Stamatiadis Gael Richard¨

LTCI, Tel´ ecom Paris, Institut Polytechnique de Paris, France´

## ABSTRACT

Music editing transforms existing recordings while preserving their original attributes. Recent approaches adapt pretrained text-to-music models for tasks like inpainting, event insertion, or restoration by conditioning them with context audio. It was recently proposed for image editing to unify generation and editing by treating the source image as in-context information. However, applying this directly to music is computationally expensive due to long audio sequences. In this work, we develop MusicFlux, a text-guided music editor built on an efficient pretrained flow-matching text-to-audio model. We introduce two conditioning approaches that trade off editing performance and computational efficiency. Experiments show that both are competitive with strong baselines, with one achieving better editing quality and the other preserving the original sequence length and substantially improving computational efficiency.

Index Terms— Music generation and editing, Text-to-music, Flow matching, Diffusion transformer

## 1. INTRODUCTION

Music editing aims to transform an existing recording to meet a userspecified objective while preserving musical attributes outside the intended modification. In contrast to unconditional generation, an editor must therefore jointly satisfy a new condition and remain consistent with the source audio [1]. This concept comprises both global transformations, such as changing instrumentation or timbre, as well as temporally localized operations, including event insertion and removal and inpainting/outpainting [2]. Such capabilities can facilitate interactive composition, music production, and post-production.

Given the high cost of training music generators, editing systems often adapt pretrained text-to-music models by conditioning on the source audio [3]. AIRGen, for instance, combines heterogeneous adapters to MusicGen [4] with masked training to enable arrangement, refinement, and inpainting [5]. Instruct-MusicGen [6] augments MusicGen with audio- and text-fusion modules for instruction-guided stem addition, removal, and extraction. Diffusion Transformer (DiT) [7] approaches have similarly introduced auxiliary controls: in [8], Stable Audio Open [9] is extended with a ControlNet branch conditioned on text and melody, while in Audio ControlNet [10], temporal controls are introduced along with an editor for inserting or removing localized sound events. Collectively, these methods demonstrate that the generative knowledge of a pretrained model can be repurposed for editing, but generally require dedicated architecture or additional attention operations.

A simpler strategy has recently emerged in image editing. FLUX.1 Kontext [11] unifies generation and editing in a single flow-matching Transformer by concatenating the source image with the noisy target, allowing pretrained self-attention layers to jointly model preserved and generated content. This in-context formulation avoids task-specific architectures while supporting both global and local edits, making it a promising basis for extending text-to-music generators to editing.

![](images/cdbec03bb149c13b2bfaad992312d719dd6408ca1cee6b40caf67ea37d85cc72.jpg)  
Fig. 1. MusicFlux formulates text-to-music generation, music inpainting, and SFX restoration under a shared latent conditioning paradigm (the source latent s represents the input audio to be transformed). The source-target latent fusion module combines (s, m, x<sub>t</sub>) to construct the latent trajectory z used as an input to the DiT.

However, directly transferring sequence concatenation to longform audio, thus doubling the sequence length, introduces a substantial computational burden, which reduces the feasible batch size or audio duration and compromises convergence. Thus, although Kontext-style concatenation provides an effective mechanism for source preservation, its direct application to music compromises the efficiency and scalability inherited from the original generator, due to its long-context nature.

In this work (see Fig. 1), we develop a text-guided music editor based on TangoFlux [12], an efficient flow-matching text-togeneral-audio model. We first fine-tune TangoFlux for text-to-music generation. Then, we adapt the model for music editing, in the tasks of inpainting/outpainting and sound-effects restoration by introducing source-audio conditioning without simply concatenating the source latents. Inspired by Condition-Aware Reparameterization for Flow Matching (CAR-Flow) [13], we design lightweight shifting networks to predict additive shifts of the flow endpoints from the editing conditions: source audio, text prompt, and a temporal editing mask. The temporal mask explicitly identifies the regions involved in inpainting or outpainting. By embedding the editing context directly into the source and target distributions, we retain the original sequence length and relieve the flow backbone from simultaneously learning source–target transport and conditional generation.

Our contributions are threefold: 1) We specialize TangoFlux [12] for text-to-music generation by fine-tuning its complete backbone on an open-source, large, and diverse high-fidelity dataset; 2) We extend TangoFlux into a text-guided music-editing framework by introducing joint audio, text, and temporal-mask conditioning in a straightforward approach. The resulting model supports diverse operations, including generation, localized edits such as inpainting and outpainting, and global edits such as sound effects (SFX) degradation restoration; and 3) we introduce a CAR-Flow-inspired [13] formulation based on condition-dependent additive transformations of the flow endpoints. In contrast to Kontext-style concatenation, CAR-Flow preserves the original sequence length, reducing attention computation and memory usage while improving efficiency. To the best of our knowledge, this is the first application of CAR-Flow to audio editing and, more broadly, to time-series data. Our demo page is accessible at https://musicflux.onrender.com/.

## 2. BACKGROUND

## 2.1. Flow Matching

Flow Matching (FM) [14] is a generative framework that learns a time-dependent vector field defining an ODE that continuously transports samples from a simple prior to the target data distribution. Let $\mathbf { x } _ { 0 } \sim p _ { 0 }$ denote a sample from the prior and $\mathbf { x } _ { 1 } \sim p _ { 1 }$ a sample from the data distribution. Given a conditional probability path $p _ { t } ( \mathbf { x } | \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } )$ , FM learns a velocity field $\mathbf { v } _ { \theta } ( \mathbf { x } , t )$ by minimizing

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } [ | | \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) | | _ { 2 } ^ { 2 } ] ,\tag{1}
$$

where $\mathbf { u } _ { t }$ is the velocity field associated with the chosen probability path. At inference, samples are generated by solving the ODE $\mathbf x _ { t } ^ { \prime } = \mathbf v _ { \theta } ( \mathbf x _ { t } , t )$ starting from $\mathbf { x } _ { 0 } \sim p _ { 0 }$ and integrating toward $t = 1$ $\mathbf { A }$ particularly choice is the linear interpolation between paired samples, named rectified flow, ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ , whose conditional velocity is constant, ${ \bf u } _ { t } = { \bf x } _ { 1 } - { \bf x } _ { 0 }$ . The resulting trajectories are substantially simpler than those induced by more general paths, enabling efficient generation with a small number of ODE solver steps.

## 2.2. Condition-Aware Reparameterization for Flow Matching

Standard conditional FM learns a velocity field that transports samples from a condition-independent source distribution, typically $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , to a condition-dependent target distribution. Consequently, the velocity network must simultaneously perform probability transport and incorporate the conditioning information. CAR-Flow [13] addresses this by explicitly reparameterizing the source and target distributions according to the condition, thereby reducing the amount of transport that must be learned by the flow network.

In CAR-Flow, condition-dependent mappings $f ( \cdot , y )$ f(·, and $g ( \cdot , y )$ transform the source and target into a reparameterized space,

$$
{ \bf z } _ { 0 } = f ( { \bf x } _ { 0 } , y ) , \qquad { \bf z } _ { 1 } = g ( { \bf x } _ { 1 } , y ) ,\tag{2}
$$

where $_ y$ denotes the conditioning variable. The flow is then learned between $\mathbf { z } _ { 0 }$ and $\mathbf { z } _ { 1 } .$ , rather than directly between $\mathbf { x } _ { \mathrm { 0 } }$ and $\mathbf { x } _ { 1 }$ . This effectively aligns the endpoints of the probability path with the conditioning information, allowing the velocity network to focus on the remaining transport. To avoid degenerate flow solutions that can arise with unrestricted reparameterizations, CAR-Flow restricts the mappings to additive, condition-dependent shifts:

$$
{ \bf z } _ { 0 } = { \bf x } _ { 0 } + { \pmb \mu } _ { 0 } ( y ) , \qquad { \bf z } _ { 1 } = { \bf x } _ { 1 } + { \pmb \mu } _ { 1 } ( y ) ,\tag{3}
$$

where $\pmb { \mu } _ { 0 }$ and $\pmb { \mu } _ { 1 }$ are lightweight learned networks. Thus, the condition-dependent shifts can relocate either endpoint or both.

## 2.3. TangoFlux

TangoFlux [12] is a fast, high-fidelity text-to-audio model that generates up to 30s of 44.1,kHz audio using a flow-matching-based hybrid DiT backbone. Text is encoded with FLAN-T5 and concatenated with a learned duration embedding, while the flow timestep and pooled text representation form a global conditioning embedding that modulates the transformer through adaptive normalization. Text and duration features interact with the audio latents through joint attention in the MMDiT blocks and are subsequently processed jointly in the single-stream DiT blocks. It achieves strong performance across both objective metrics and human evaluations, while being significantly faster than many prior text-to-audio models.

## 3. PROPOSED METHOD

We consider the problem of fine-tuning a text-to-music model to support both music generation and editing. To this end, we explore two conditioning strategies. The first, based on FLUX.1 Kontext [11] conditions the model by concatenating the target sequence with the contextual input. The second is inspired by CAR-Flow [13], and instead introduces a condition-aware reparameterization of the model.

## 3.1. MusicFlux-Concat: Context Concatenation

Let $\mathbf { x } _ { t } \in \mathbb { R } ^ { B \times L \times C }$ denote the noisy target latent at time t, where B is the batch size, L is the sequence length, and C is the number of latent channels. Our first conditioning strategy directly extends TangoFlux by concatenating the source latent s with the noisy target latent along the sequence dimension (see Fig. $2 \ ( L e f t ) )$ . The source latent provides the contextual information that the model should condition on, such as an audio signal to be fully transformed or an input containing a masked region. In the absence of an audio context, as in text-to-music, the source latent is replaced by a zero tensor.

To explicitly specify how the model should use the context, we append an extra mask channel, yielding a total of $( C + 1 )$ channels:

$$
\begin{array} { r } { \mathbf { z } _ { t } = \mathrm { c o n c a t } \left( [ \mathbf { x } _ { t } ; \mathbf { s } ] , [ \mathbf { 0 } ; \mathbf { m } ] \right) \in \mathbb { R } ^ { B \times 2 L \times ( C + 1 ) } , } \end{array}\tag{4}
$$

where the mask m contains 0 at positions corresponding to known context and 1 at positions to be generated. On text-to-music, m is set entirely to 1. For inpainting/outpainting, it contains 1 within the region to be generated and 0 elsewhere. For SFX restoration, where the complete audio context is provided, m is set entirely to 0.

We assign distinct positional IDs to the source and target tokens, with target tokens occupying positions $0 \ldots L - 1$ and source tokens occupying positions $L \ldots 2 L \not {- 1 }$ , following FLUX.1 Kontext [11]. The DiT backbone then performs full self-attention over the resulting sequence of length 2L, while the training loss is computed only on the target half. Thus, the conditioning mechanism requires only a single additional learnable projection from C to $C + 1$ channels and leaves the DiT backbone architecture unchanged.

## 3.2. MusicFlux-CAR: Condition-Aware Reparameterization

The main drawback of the aforementioned approach is its computational cost since the complexity of self-attention scales with sequence length. Hence, we propose a second approach which eliminates the sequence-length doubling by pre-aligning the flow endpoints with the conditioning signal (see Fig. $2 \left( R i g h t \right) )$ . Inspired by CAR-Flow [13], we introduce two lightweight Context Shift Network modules $\mu _ { \mathrm { s r c } }$ and $\mu _ { \mathrm { t g t } }$ that produce additive latent shifts:

$$
\mathbf { z } _ { 0 } = \epsilon + \mu _ { \mathrm { s r c } } ( \mathbf { s } , \mathbf { e } , \mathbf { m } ) , \qquad \mathbf { z } _ { 1 } = \mathbf { x } _ { 1 } + \mu _ { \mathrm { t g t } } ( \mathbf { s } , \mathbf { e } , \mathbf { m } ) ,\tag{5}
$$

![](images/7301a5d829e88b6bbff245b51dc7e9818e00b1c39f22e39ebe87d69de2ae5c18.jpg)  
Fig. 2. (Left) MusicFlux-Concat: The target latent $x _ { t }$ and source audio latent s are concatenated along the sequence dimension (2L frames), while the binary mask m is concatenated along the channel dimension. (Right) MusicFlux-CAR: Condition-Aware Reparameterization uses lightweight auxiliary networks $( \mu _ { \mathrm { s r c } } , \mu _ { \mathrm { t g t } } )$ to shift the noise start-point z<sub>0</sub> and target endpoint $z _ { 1 }$ . Diffusion operates directly in the shifted space over L frames, reducing complexity.

where $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , e is a pooled text embedding, and m denotes the mask defined in the previous section. The DiT operates on the noisy interpolant $\mathbf { z } _ { t } = ( \hat { 1 } - \sigma _ { t } ) \mathbf { z } _ { 1 } + \sigma _ { t } \mathbf { z } _ { 0 } \in \mathbb { R } ^ { B \times L \hat { \times } C }$ , which has the same length as the input used for standard unconditional text-tomusic generation. The training loss is then given by:

$$
\mathcal { L } _ { \mathrm { C A R - F L o w } } = \mathbb { E } \big [ \| v _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } ) - ( \mathbf { z } _ { 0 } - \mathbf { z } _ { 1 } ) \| _ { 2 } ^ { 2 } \big ] .\tag{6}
$$

At inference, the generated latent is recovered by: $\hat { \mathbf { x } } _ { 1 } = \mathbf { z } _ { \mathrm { f i n a l } } - \mu _ { \mathrm { t g t } }$ Shift Network Architecture. Each shift network $( \mu _ { \mathrm { s r c } } , \mu _ { \mathrm { t g t } } )$ processes the source context latent s $\in \mathbb { R } ^ { B \times C \times L }$ , pooled text embedding e, and binary mask 1−m using two complementary branches: (i) Global branch: computes the spatial mean of s over knowncontext frames, concatenates it with the text embedding e, and passes the vector through a 2-layer MLP (Linear → SiLU → Linear) to yield a per-channel global offset broadcast across sequence length L; (ii) Spatial branch: processes the masked context s ⊙ (1−m) via a 1-D depthwise-separable convolution (kernel size 3) to capture local temporal details, augmented by a learnable positional bias $\textbf { p } \in \mathbb { R } ^ { 1 \times C \times L }$ Features from both branches are concatenated and projected via a zero-initialized linear output layer, ensuring the model begins training with identity shifts $( \mu \mathbf { \Omega } = \mathbf { 0 } )$ corresponding to standard unshifted flow matching endpoints. Both shift networks add only ≈4.6 M parameters $( < ~ 0 . 6 \%$ of the DiT backbone), maintaining negligible computational overhead.

## 3.3. Task-Specific Loss Objectives

All tasks optimize the velocity prediction $v _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } )$ against target flow field: ${ \bf u } _ { t } = { \bf x } _ { 1 } - { \bf x } _ { 0 }$ for MusicFlux-Concat and ${ \bf u } _ { t } = { \bf z } _ { 1 } - { \bf z } _ { 0 }$ for MusicFlux-CAR. For text-to-music generation and SFX restoration, we compute a full-sequence MSE loss $\mathcal { L } _ { \mathrm { f u l l } } = \mathbb { E } \big [ \| v _ { \theta } - \mathbf { u } _ { t } \| ^ { 2 } \big ]$ For inpainting and outpainting, the objective combines two distinct terms over the target reconstruction region m and context region (1 − m), ${ \mathcal { L } } _ { \mathrm { i n p a i n t } } = { \mathcal { L } } _ { \mathrm { m a s k e d } } + \lambda _ { \mathrm { c t x } } { \mathcal { L } } _ { \mathrm { c t x } }$ , where $\mathcal { L } _ { \mathrm { m a s k e d } } ~ =$ $\mathbb { E } \big [ \mathbf { m } \odot \big | \big | \boldsymbol { v } \theta - \dot { \mathbf { u } } _ { t } \big | \big | ^ { 2 } \big ]$ penalizes errors within the generated region, while $\mathcal { L } _ { \mathrm { c t x } } = \mathbb { E } \big [ ( 1 - \mathbf { m } ) \odot \| v _ { \theta } - \mathbf { u } _ { t } \| ^ { 2 } \big ]$ encourages fidelity to the context boundaries. We use a factor $\bar { \lambda } _ { \mathrm { c t x } } \mathrm { ~ \ ' < ~ } 1 . 0$ for the context loss, since reconstructing the observed context is more straightforward.

## 4. EXPERIMENTS

## 4.1. Datasets

JamendoMaxCaps (JMC) [17]: A collection of instrumental audio collected from the Jamendo Music platform paired with metadata and captions generated using Qwen2-Audio [18]. The corpus contains approximately 362K tracks, totaling around 58K hours, with an average duration of 4 minutes and 12 seconds per track. All audio is stereo and standardized at 44.1 kHz. Each 30-second audio segment is captioned, producing 1,812,341 paired audio–caption samples. We split the dataset into 357 samples for validation, 10,000 for testing, and the remainder is allocated for training.

SonicMaster-Test [16]: 1,000 clean 30-s music excerpts and 7,000 corresponding degraded versions, with seven variants generated per reference: four single-, two double-, and one triple-degradation examples. The degradations cover 19 audio effects from five categories: equalization, dynamics, reverb, amplitude, and stereo. Natural language instructions for restoration are given.

Song Describer Dataset (SDD) [19]: 706 openly licensed music recordings paired with 1,106 human-written captions, with 95% of the audio excerpts lasting up to 2 minutes. In our experiments, we use the valid subset, which went through a manual quality-control process and further exclude tracks whose captions indicate the presence of vocals, yielding an instrumental-only evaluation subset.

## 4.2. Setup

We use the JMC training split first fine-tune the TangoFlux model for music generation, and then to train the editing models. These were trained for 4 epochs, on 3 L40S GPUs, with a per-device batch size of 32 for MusicFlux-CAR and 16 for the MusicFlux-Concat version with the Adam optimizer. We set a learning rate of $2 e ^ { - 4 }$ , with a 2k warmup and a cosine-with-restarts scheduler with 2 restarts. We sample all three tasks with equal probability. We use a Transformer with 6 dual-stream and 18 single-stream blocks, 8 heads of dimension 128, and a joint conditioning dimension of 1024. In total, MusicFlux-CAR and MusicFlux-Concat have roughly 859M and 860M parameters, respectively. The model operates on sequences of maximum 645 latents, extracted with Stable Audio Open 1.0 [9], corresponding to 30 seconds. We found that $\lambda _ { \mathrm { c t x } } = 0 . 1$ yielded the best results. We select the checkpoint that performs best jointly on both generation and editing, according to a CAF-Score metric [20] on the JMC validation set. The SonicMaster-Test and the SDD dataset are used as test sets for out-of-domain evaluation of SFX restoration and inpainting/outpainting and generation, respectively. We run inference with 50 integration steps and a CFG scale of 4.5.

## 4.3. Baselines

For music generation and inpainting/outpainting, we use the state-ofthe-art Stable Audio 3.0 (SA3) [15] as a baseline. The model is also based on a DiT encapsulated in a Flow Matching framework. Frozen T5Gemma text embeddings, concatenated with a duration embedding, condition the model through cross-attention. Instead of using CAR-Flows or concatenation, the context audio is linearly projected and added to the hidden states of each transformer block. We use its small-music-base version, as it’s the closest in terms of size to MusicFlux, with 459M parameters. It was trained on a mixture of the private, curated AudioSparx and Freesound datasets, with ≈1.2M tracks. For restoration, we use SonicMaster [16], a text-conditioned rectified-flow model that directly transports degraded audio latents toward clean ones. Unlike MusicFlux, the degraded source is not provided as an additional conditioning signal, but instead serves as the initial state of the flow trajectory.

<table><tr><td rowspan="2"></td><td colspan="4">Generation</td><td colspan="4">Inpainting/Outpainting</td><td colspan="4">Restoration</td></tr><tr><td colspan="2">JMC-Test</td><td colspan="2">SDD</td><td colspan="2">JMC-Test</td><td colspan="2">SDD</td><td colspan="4">SonicMaster-Test</td></tr><tr><td>Model</td><td>CAF↑</td><td>KAD↓</td><td>CAF↑</td><td>KAD↓</td><td>FLEUR*↑</td><td>KAD↓</td><td>FLEUR*↑</td><td>KAD↓</td><td>KAD↓</td><td>KL↓</td><td>SSIM↑</td><td>PQ↑</td></tr><tr><td>Ground-truth (GT)</td><td>0.5102</td><td>0.0000</td><td>0.5704</td><td>0.0000</td><td>0.8592</td><td>0.0000</td><td>0.8834</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>1.000</td><td>7.7951</td></tr><tr><td>Stable Audio 3 [15]</td><td></td><td></td><td>0.5593</td><td>2.8631</td><td></td><td></td><td>0.8248</td><td>1.4794</td><td></td><td></td><td></td><td></td></tr><tr><td>SonicMaster [16]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.1137</td><td>0.0009</td><td>0.5918</td><td>7.5310</td></tr><tr><td>MusicFlux-Concat</td><td>0.5100</td><td>0.6400</td><td>0.5213</td><td>4.5535</td><td>0.8340</td><td>0.0099</td><td>0.8428</td><td>0.0380</td><td>0.0489</td><td>0.0013</td><td>0.5702</td><td>7.5478</td></tr><tr><td>MusicFlux-CAR</td><td>0.5196</td><td>1.1239</td><td>0.4950</td><td>4.6687</td><td>0.8234</td><td>0.2496</td><td>0.8390</td><td>0.4744</td><td>0.2680</td><td>0.0015</td><td>0.5412</td><td>7.4044</td></tr></table>

Table 1. Evaluation results of music generation, inpainting, and restoration. Best results are shown in bold and second best are underlined.

<table><tr><td>Stage</td><td>MusicFlux Method</td><td>Base (MiB)</td><td>Dynamic (MiB)</td><td>RTF</td></tr><tr><td rowspan="2">Inference</td><td>CAR</td><td>3364.19</td><td>75.71</td><td>0.1188</td></tr><tr><td>Concat</td><td>3358.54</td><td>150.11</td><td>0.2203</td></tr><tr><td rowspan="2">Training</td><td>CAR</td><td>3372.48</td><td>1988.10</td><td>一</td></tr><tr><td>Concat</td><td>3366.82</td><td>2748.19</td><td>一</td></tr></table>

Table 2. Computational comparison MusicFlux versions. Measurements use a batch size of 1, a sequence of 645, and FP32 weights.

## 4.4. Evaluation Metrics

For generation we report CAF-Score [20] for text–audio alignment and Kernel Audio Distance (KAD) [21] for distributional audio quality. CAF-Score combines CLAP similarity with FLEUR-based LALM assessment [22]; we use Speech-Music LAION CLAP. KAD computes the MMD between CLAP-LAION music embeddings of generated and pre-editing audio. For inpainting/outpainting, the objective is seamless continuation of the source rather than caption alignment. We therefore replace CAF-Score with FLEUR\*, a taskspecific modification of FLEUR [22]. Given the boundaries of the regenerated region, Qwen3-Omni-30B-A3B-Instruct [23] rates from 0 to 1 how well it fits the recording, considering boundary continuity, timbre, loudness, ambience, rhythm, and audible artifacts. We query three settings: (i) edited audio alone, (ii) edited audio paired with the masked source, and (iii) edited audio paired with the unmasked ground truth. Following FLEUR, ratings are smoothed using score-token probabilities, and FLEUR\* is their average. We additionally report KAD for inpainting/outpainting. For restoration, we use the perceptual fidelity and production-quality metrics adopted by SonicMaster [16]. KL divergence measures distributional discrepancy, SSIM measures structural similarity between 128-bin melspectrograms, and Production Quality (PQ) from Audiobox Aesthetics [24] estimates perceived technical quality in aspects such as clarity, dynamics, balance, and spatialization.

## 5. RESULTS

Table 1 summarizes the results for all metrics used. The first row reports the results on the ground-truth (GT) test set, where each metric is computed by comparing the ground-truth audio or set against itself. For in-domain generation on JMC-Test, MusicFlux-CAR achieves the best CAF-Score (0.5196), just above MusicFlux-Concat and GT, while MusicFlux-Concat obtains the lowest KAD (0.6400).

This shows that both conditioning strategies preserve the generation capabilities of the pretrained model while adapting to new tasks, with CAR favoring text-audio alignment and Concat distributional similarity. On the out-of-domain SDD, SA3 obtains better results in both metrics, showing that the proposed models generalize less to unseen prompts. This could be due to better prompt extraction, given the curated dataset of SA3, or better prompt encoding.

For inpainting/outpainting, MusicFlux-Concat provides the strongest results in both metrics on all in- and out-of-domain scenarios. MusicFlux-CAR also beats the SA3 baseline on the out-ofdomain evaluation. This advantage of MusicFlux-Concat over all other models might come from the direct exposure of the context to the attention layers, whereas in MusicFlux-CAR the context is conveyed through lightweight additive shifts, and in SA3 through a simple projection and sum, both of which could potentially be acting as information bottlenecks for localized reconstruction.

For SFX restoration, MusicFlux-Concat achieves the best KAD and PQ, while SonicMaster performs best in KL divergence and SSIM. Both MusicFlux variants remain competitive with the specialized SonicMaster baseline despite being trained jointly across multiple tasks, with MusicFlux-Concat slightly outperforming MusicFlux-CAR. This advantage may again result from the direct access to source information provided by sequence concatenation, whereas CAR encodes the context indirectly through latent shifts, making fine-grained restoration more challenging.

Lastly, Table 2 shows that CAR substantially improves computational efficiency over Concat. RTF and Dynamic memory (model execution) are reduced by over 50% at inference and 28% at training, while base memory (model instantiation) is kept almost the same due to the lightweight shift networks. This confirms the benefit of preserving the original sequence.

## 6. CONCLUSION

We introduced MusicFlux, a unified framework covering three music editing settings: global text-to-music generation, localized editing through inpainting/outpainting, and global editing through SFX restoration. We explored two conditioning strategies: MusicFlux-Concat, which achieves stronger editing performance, and MusicFlux-CAR, which preserves the original sequence length and reduces inference runtime and dynamic memory by over 50%. Both remain competitive with task-specific baselines despite being single multi-task models trained on open data. The results highlight a trade-off between editing fidelity and efficiency, while limitations in out-of-domain generalization and CAR conditioning motivate richer efficient conditioning in future work.

## 7. ACKNOWLEDGEMENTS

With support from the European Union (ERC, HI-AUDIO, 101052978). Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. This work was performed using HPC resources from GENCI-IDRIS (Grants 2026-AD011018031, 2026-AD011017871, 2026-AD011014567R2, and 2026-AD011014567R3) and was supported by the IDRIS Open Hackathon 2026 program in collaboration with NVIDIA and OpenACC-Standard.org. The authors thank Martin Comminges, Krishna Kalyan and Nemo Bouillon for their´ contributions.

## 8. REFERENCES

[1] Huadai Liu, Jialei Wang, Xiangtai Li, Wen Wang, Qian Chen, Rongjie Huang, Yang Liu, Jiayang Xu, Zhou Zhao, and Wei Xue, “Melodyedit: Zero-shot music editing with disentangled inversion control,” in Proc. ACM International Conference on Multimedia, 2025.

[2] Changhao Pan, Yifei Fan, Fan Zhuo, Yifu Chen, Wenxiang Guo, Yu Zhang, Ruiqi Li, Zhiyuan Zhu, Rui Yang, Shengpeng Ji, et al., “Audio editing in the era of foundation models: A survey,” arXiv preprint arXiv:2606.23139, 2026.

[3] Manos Plitsis, Theodoros Kouzelis, Georgios Paraskevopoulos, Vassilis Katsouros, and Yannis Panagakis, “Investigating personalization methods in text to music generation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.

[4] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Defossez, “Sim-´ ple and controllable music generation,” in Advances in neural information processing systems (NeurIPS), 2023.

[5] Liwei Lin, Gus Xia, Yixiao Zhang, and Junyan Jiang, “Arrange, inpaint, and refine: Steerable long-term music audio generation and editing via content-based controls,” arXiv preprint arXiv:2402.09508, 2024.

[6] Yixiao Zhang, Yukara Ikemiya, Woosung Choi, Naoki Murata, Marco A. Mart´ınez-Ram´ırez, Liwei Lin, Gus Xia, Wei-Hsiang Liao, Yuki Mitsufuji, and Simon Dixon, “Instruct-musicgen: Unlocking text-to-music editing for music language models via instruction tuning,” in International Society for Music Information Retrieval Conference (ISMIR), 2025.

[7] William Peebles and Saining Xie, “Scalable diffusion models with transformers,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[8] Siyuan Hou, Shansong Liu, Ruibin Yuan, Wei Xue, Ying Shan, Mangsuo Zhao, and Chao Zhang, “Editing music with melody and text: Using controlnet for diffusion transformer,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.

[9] Zach Evans, Julian D. Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons, “Stable audio open,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.

[10] Haina Zhu, Yao Xiao, Xiquan Li, Ziyang Ma, Jianwei Yu, Bowen Zhang, Mingqi Yang, and Xie Chen, “Audio controlnet for fine-grained audio generation and editing,” arXiv preprint arXiv:2602.04680, 2026.

[11] Black Forest Labs et al., “Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,” arXiv preprint arXiv:2506.15742, 2025.

[12] Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Zadeh, Chuan Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria, “Tangoflux: Super fast and faithful text to audio generation with flow matching and clapranked preference optimization,” in International Conference on Learning Representations (ICLR), 2026.

[13] Chen Chen, Pengsheng Guo, Liangchen Song, Jiasen Lu, Rui Qian, Tsu-Jui Fu, Xinze Wang, Wei Liu, Yinfei Yang, and Alex Schwing, “CAR-flow: Condition-aware reparameterization aligns source and target for better flow matching,” in Advances in neural information processing systems (NeurIPS), 2025.

[14] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le, “Flow matching for generative modeling,” in International Conference on Learning Representations (ICLR), 2023.

[15] Zach Evans, Julian D. Parker, Matthew Rice, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons, “Stable audio 3,” arXiv preprint arXiv:2605.17991, 2026.

[16] Jan Melechovsky, Ambuj Mehrish, Abhinaba Roy, and Dorien Herremans, “Sonicmaster: Towards controllable all-in-one music restoration and mastering,” in International Conference on Machine Learning (ICML), 2026.

[17] Abhinaba Roy, Renhang Liu, Tongyu Lu, and Dorien Herremans, “Jamendomaxcaps: A large scale music-caption dataset with imputed metadata,” in Proc. International Joint Conference on Neural Networks (IJCNN), 2025.

[18] Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, et al., “Qwen2-audio technical report,” arXiv preprint arXiv:2407.10759, 2024.

[19] Ilaria Manco, Benno Weck, Seungheon Doh, Minz Won, Yixiao Zhang, Dmitry Bogdanov, Yusong Wu, Ke Chen, Philip Tovstogan, Emmanouil Benetos, et al., “The song describer dataset: a corpus of audio captions for music-and-language evaluation,” arXiv preprint arXiv:2311.10057, 2023.

[20] Insung Lee, Taeyoung Jeong, Haejun Yoo, Du-Seong Chang, and Myoung-Wan Koo, “Caf-score: Calibrating clap with lalms for reference-free audio captioning evaluation,” arXiv preprint arXiv:2603.19615, 2026.

[21] Yoonjin Chung, Pilsun Eu, Junwon Lee, Keunwoo Choi, Juhan Nam, and Ben Sangbae Chon, “Kad: No more fad! an effective and efficient evaluation metric for audio generation,” in AI Heard That! ICML Workshop, 2025.

[22] Yebin Lee, Imseong Park, and Myungjoo Kang, “FLEUR: An explainable reference-free evaluation metric for image captioning using a large multimodal model,” in Annual Meeting ofthe Associationfor Computational Linguistics, 2024.

[23] Jin Xu et al., “Qwen3-omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[24] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, Carleigh Wood, Ann Lee, and Wei-Ning Hsu, “Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound,” arXiv preprint arXiv:2502.05139, 2025.