# MURFormer: Attention-Based Lightweight Multi-Scale Uncertainty-Guided Refinement Framework for PPG Denoising

Haram Kwon, Chaeeun Hwang, Seayul Choi, and Sungho Jo, Senior Member, IEEE

Abstract— Photoplethysmography (PPG) is a widely used physiological signal for its convenience but is highly susceptible to motion artifacts that degrade waveform quality and limit downstream analysis. This paper proposes MURFormer, a lightweight multi-scale uncertainty-guided refinement framework for PPG denoising through noisyto-clean PPG mapping. MURFormer employs a newly proposed Multi-Scale Attention-Conv Fusion (MACF) block, which is designed to efficiently capture multi-scale temporal dynamics of PPG, as the fundamental building block and performs uncertainty-guided two-pass refinement prioritizing unreliable regions. The method was evaluated on the publicly available CIME dataset and on a Daily Activities dataset collected in this work, comprising paired noisyclean PPG recordings from 24 subjects. Experimental results show that MURFormer improves both signal quality and physiological fidelity over the raw noisy input, yielding 129% and 105% relative signal-to-noise ratio (SNR) improvements, and 66% and 81% relative reductions in heartrate mean absolute error (MAE), on the Daily Activities and CIME datasets, respectively. MURFormer also outperforms an existing denoising baseline on both datasets with a relatively small parameter count while preserving beat-level timing and waveform morphology. In addition, the model maintains strong denoising performance across varying input window lengths, suggesting that MURFormer is an effective and robust framework for practical PPG denoising. The implementation of MURFormer is publicly available at https://github.com/k-hram/MURFormer.git

Index Terms— photoplethysmography (PPG), denoising, uncertainty estimation, attention

## I. INTRODUCTION

P <sup>HOTOPLETHYSMOGRAPHY</sup> <sup>(PPG)</sup> <sup>is</sup> <sup>a</sup> <sup>simple,</sup> <sup>cost-</sup> effective, and noninvasive optical sensing technique commonly used to detect changes in blood volume in the microvascular bed of tissue [1], [2]. PPG has been extensively adopted in various physiological monitoring tasks such as heart rate estimation, pulse interval analysis [3], and vascular assessments through wearable or clinical monitoring systems, due to its convenient acquisition [4]–[6]. However, its high vulnerability to motion artifacts often limits the practical utility of PPG, particularly in daily-life settings, where signal corruption can distort waveform morphology and degrade the reliability of PPG-based downstream analysis [7], [8].

This vulnerability is particularly problematic in wearable scenarios, where body movement, sensor displacement, and varying contact conditions frequently contaminate the measured signal. Under such conditions, motion artifacts not only affect visual signal quality but can also alter peak timing, local morphology, and pulse-to-pulse consistency, which complicate physiological analysis in PPG.

A large body of prior work has therefore addressed PPG denoising and motion-artifact reduction. Traditional methods often relied on signal processing techniques such as wavelet transform [9], [10], adaptive filtering [11]–[13], and decomposition [14]–[16]. More recent work has explored learning-based approaches such as autoencoder-based models [17], [18], generative adversarial networks (GANs) [19]–[23], mamba [24], transformer [25], learned convolutional sparse coding [26], and diffusion [27], [28]. These studies have demonstrated that learning-based approaches can effectively reduce noise by reconstructing a clean PPG signal from a corrupted input. However, the way in which noisy-clean PPG pairs are constructed remains an important limitation. Several studies have trained denoising models by adding extracted or simulated noise to clean PPG signals [17], [24], [26], using self-supervised erasing of clean segments [18], or adopting unpaired noisy-to-clean translation [21]. Although these strategies reduce the need for paired recordings, they may not faithfully reproduce real motion artifacts arising from sensor displacement, skin deformation, and motion-related physiological changes, potentially limiting generalization to wearable settings.

To address this limitation, recent work by Xia et al. [27] and Liu et al. [28] employ noisy-clean PPG pairs for supervised reconstruction. These studies have demonstrated that paireddata approaches can effectively reduce noise by reconstructing a clean PPG waveform from a corrupted input. Nevertheless, existing paired-data denoisers often rely on relatively complex generative or multi-module pipelines. Therefore, this work aims to retain the benefits of paired noisy-clean supervision while introducing a lightweight and parameter-efficient denois-

ing framework.

This study proposes MURFormer (Multi-Scale Uncertainty-Guided Refinement Former), a lightweight uncertainty-guided two-pass refinement framework trained on paired noisy-clean PPG recordings. Although several prior methods incorporate motion reference signals such as accelerometer or gyroscope measurements [29]–[31], MURFormer uses only PPG as input and does not require additional motion signals to reconstruct clean signals. While excluding IMU signals may remove useful motion-reference information, this design makes the framework applicable to PPG-only recordings and avoids additional synchronization requirements. The model combines the newly proposed Multi-Scale Attention-Convolution Fusion (MACF) block with a three-stage training strategy for reconstruction, uncertainty estimation, and refinement. Unlike prior artifact-detection-and-reconstruction frameworks that use detection mainly for artifact removal preprocessing [18], [21], [28], MURFormer uses uncertainty to further refine the signal by prioritizing regions that remain unreliable after initial reconstruction.

Experimental results on both datasets show that MUR-Former substantially improves signal quality and physiological fidelity over noisy inputs, outperforms a recent diffusion-based baseline [27] with fewer parameters, and maintains robust performance across varying test-time input lengths.

In summary, the main contributions of this work are as follows:

1) We propose MURFormer, a lightweight PPG denoising framework that achieves strong denoising performance.

2) We introduce the MACF block, a parameter-efficient encoder designed to capture local and multi-scale temporal characteristics of PPG.

3) We present an uncertainty-guided two-pass refinement strategy that improves reconstruction quality.

## II. METHODS

## A. Model Overview

MURFormer is designed to denoise motion-corrupted PPG signals using paired noisy-clean recordings. Given a noisy PPG segment $\mathbf { x } _ { n o i s y } \in \mathbb { R } ^ { T }$ , the model reconstructs its clean counterpart $\mathbf { y } _ { c l e a n } \in \mathbb { R } ^ { T }$ through a two-pass framework composed of first-pass reconstruction, patch-level uncertainty estimation, and second-pass refinement.

The architecture of MURFormer is shown in Fig. 2c and Fig. 1a. The pass1 reconstruction path, consisting of a patchifier, MACF blocks, and a pass1 reconstruction head, converts the noisy input PPG into an initial denoised signal. The uncertainty head predicts a patch-level uncertainty map that indicates which temporal regions of the first-pass output are likely to remain unreliable. The pass2 refinement path, consisting of MACF blocks and a pass2 reconstruction head, then takes a masked version of the pass1 latent representation and prioritizes the refinement of unreliable patches to produce the final denoised output. The remainder of this section describes the MACF block, the three-stage training procedure, and the two-pass inference workflow in detail.

## B. MACFBlock

Fig. 1b illustrates the proposed Multi-Scale Attention-Convolution Fusion (MACF) block, which serves as the basic encoder block of MURFormer. The residual-style structure of an MACF block resembles that of a standard Transformer encoder block [32], but differs in that the input is processed through a triple-branch structure, intended to capture both local waveform morphology and temporal dependencies at multiple scales. Each branch is assigned to a different operation: local convolution, small-scale attention, and large-scale attention. This design enables the block to model complementary temporal patterns while using fewer parameters than applying attention to the full embedding dimension.

Let the input sequence be $\bar { \mathbf { x } } \in \mathbb { R } ^ { B \times P \times D }$ , where $B , P ,$ and D denote the batch size, the number of patch tokens, and the embedding dimension, respectively. The overall flow of the MACF block can be written as

$$
[ \mathbf { x } _ { c } , \mathbf { x } _ { s } , \mathbf { x } _ { l } ] = \mathrm { S p l i t } ( \mathrm { L i n e a r } ( \mathrm { L N } ( \mathbf { x } ) ) ) ,\tag{1}
$$

$$
\tilde { \mathbf { x } } = \mathrm { C o n c a t } ( \mathrm { C o n v } ( \mathbf { x } _ { c } ) , \mathrm { A t t n } _ { s } ( \mathbf { x } _ { s } ) , \mathrm { A t t n } _ { l } ( \mathbf { x } _ { l } ) ) ,\tag{2}
$$

$$
\mathbf { z } = \mathbf { x } + \mathrm { D r o p o u t } ( \tilde { \mathbf { x } } ) ,\tag{3}
$$

$$
\mathbf { y } = \mathbf { z } + \mathrm { D r o p o u t } ( \mathrm { F F N } ( \mathbf { z } ) ) ,\tag{4}
$$

where LN(·) denotes layer normalization, Linear(·) denotes a linear projection, Split(·) denotes channel-wise feature splitting, Conv(·) denotes the convolution branch, $\mathrm { A t t n } _ { s } ( \cdot )$ and $\mathrm { A t t n } _ { l } ( \cdot )$ denote small-scale and large-scale attention branches, respectively, Concat(·) denotes channel-wise concatenation, and FFN(·) denotes the position-wise feed-forward network with an expansion ratio of 2.

As shown in Fig. 1b, an MACF block input sequence $\mathbf { x } ~ \in ~ \mathbb { R } ^ { B \times P \times D }$ is first linearly projected while preserving its shape, and the resulting sequence is then split along the channel dimension into

$$
\begin{array} { r } { \mathbf { x } _ { c } \in \mathbb { R } ^ { B \times P \times ( D / 2 ) } , \mathbf { x } _ { s } \in \mathbb { R } ^ { B \times P \times ( D / 4 ) } , \mathbf { x } _ { l } \in \mathbb { R } ^ { B \times P \times ( D / 4 ) } . } \end{array}\tag{5}
$$

The first branch is a convolution branch that takes $\mathbf { x } _ { c }$ and applies 1-D convolution of kernel size 5 along the temporal axis while preserving the input shape through padding, producing $\tilde { \mathbf { x } _ { c } }$

The second branch is a small-scale attention branch, in which temporal partition, multi-head attention, and reverse partition are applied sequentially. For temporal partition, the input $\mathbf { x } _ { s }$ is partitioned along the temporal dimension using partition size $p _ { s }$ (Eq. (6)), a small-scale partition size, followed by batch-wise concatenation (Eq. (7)). Thus,

$$
\mathbf { x } _ { s } \in \mathbb { R } ^ { B \times P \times ( D / 4 ) } \to \mathbb { R } ^ { B \times ( P / p _ { s } ) \times p _ { s } \times ( D / 4 ) }\tag{6}
$$

$$
\mathbb { R } ^ { ( B \times p _ { s } ) \times ( P / p _ { s } ) \times ( D / 4 ) } .\tag{7}
$$

Multi-head attention is then applied to the partitioned sequence, followed by a reverse partition, which restores the original sequence layout and produces $\tilde { \mathbf { x } _ { s } }$

$$
\mathbb { R } ^ { ( B \times p _ { s } ) \times ( P / p _ { s } ) \times ( D / 4 ) }  \mathbb { R } ^ { B \times ( P / p _ { s } ) \times p _ { s } \times ( D / 4 ) }\tag{8}
$$

$$
\tilde { \mathbf { x } _ { s } } \in \mathbb { R } ^ { B \times P \times ( D / 4 ) } .\tag{9}
$$

![](images/54b3dff144b1369bb328b091f2b46826916e030e790e3ad2ccf766e696793caa.jpg)  
Fig. 1: (a) MURFormer architecture and inference workflow. (b) MACF block architecture. The top panel presents the overall architecture of an MACF block, and the bottom panel illustrates the multi-scale attention mechanism in its triple-branch structure.

The third branch is a large-scale attention branch that follows the same procedure as the second branch, but uses a different partition size, $p _ { l }$ , and a separate multi-head attention module of the same structure, producing $\tilde { \mathbf { x } } _ { l }$

The outputs of three branches $\tilde { \mathbf { x } _ { c } } , \tilde { \mathbf { x } _ { s } } ,$ , and $\tilde { \mathbf { x } } _ { l }$ are concatenated along the channel dimension to recover the original embedding size, yielding a fused representation $\tilde { \mathbf { x } } \in \bar { \mathbb { R } ^ { B \times P \times D } }$ Dropout is then applied followed by residual addition with the block input. The resulting feature is further processed by a position-wise feed-forward network, with another dropout and residual addition following to produce the output y.

Since attention is applied only to partial channel subspaces, the MACF block uses fewer attention parameters than a standard Transformer encoder with the same input shape. With the adopted patch configuration, each token corresponds to 0.25 s of the input signal. The value of $p _ { s }$ is set to 5, aggregating information over 1.25 s of the input. This temporal span is slightly longer than a single pulse interval at 60 bpm, allowing the small-scale attention branch to apply multi-head attention over approximately one to two cardiac cycles. We further set the value of p<sub>l</sub> to 20, so that the large-scale attention branch covers 5 s, and captures dependencies across multiple pulses.

## C. Three-Stage Training Strategy

A three-stage training strategy is designed to train MUR-Former to first learn a reliable reconstruction path, then to estimate a patch-wise reconstruction uncertainty, and to guide refinement toward unreliable regions. At each stage, only modules that are relevant to the current objective remain unfrozen in order to allow the model to learn each function progressively while preserving previously learned components. An overview of the training strategy is shown in Fig. 2.

Stage 1: Direct Noisy-to-Clean Reconstruction: In Stage 1, the noisy input $\mathbf { x } _ { n o i s y }$ is converted into patch tokens ${ \bf P } _ { n o i s y }$

Then it is processed by the MACF backbone and the pass1 reconstruction head, producing the first-pass denoised signal. The objective of this stage is to minimize the reconstruction loss ${ \mathcal L } _ { \mathrm { S 1 } }$

The Stage 1 reconstruction loss is defined as

$$
\mathcal { L } _ { \mathrm { S 1 } } = \mathcal { L } _ { \mathrm { c h a r } } + \alpha \mathcal { L } _ { \mathrm { s p e c } } + \beta \mathcal { L } _ { \nabla } + \gamma \mathcal { L } _ { \nabla ^ { 2 } } ,\tag{10}
$$

where $\mathcal { L } _ { \mathrm { c h a r } }$ is the Charbonnier loss, $\mathcal { L } _ { \mathrm { s p e c } }$ is the spectral loss that penalizes mismatches in the frequency domain, and $\mathcal { L } _ { \nabla }$ and ${ \mathcal { L } } _ { \nabla ^ { 2 } }$ enforce consistency in the first and second-order temporal derivatives $( \alpha , \beta , \gamma \in \mathbb { R } )$ . These loss functions are expressed as

$$
\mathcal { L } _ { \mathrm { c h a r } } = \frac { 1 } { B T } \sum _ { b , t } \sqrt { \left( \left( \hat { \mathbf { y } } _ { R 1 } \right) _ { b , t } - \left( \mathbf { y } _ { c l e a n } \right) _ { b , t } \right) ^ { 2 } + \epsilon ^ { 2 } } ,\tag{11}
$$

$$
\mathcal { L } _ { \mathrm { s p e c } } = \frac { \Vert | \mathrm { r F F T } ( \hat { \mathbf { y } } _ { R 1 } ) | - | \mathrm { r F F T } ( \mathbf { y } _ { c l e a n } ) | \Vert _ { 2 } } { N _ { \mathrm { f f t } } } ,\tag{12}
$$

$$
\mathcal { L } _ { \nabla } = \frac { 1 } { B T } \sum _ { b , t } \sqrt { \left( \nabla ( \hat { \mathbf { y } } _ { R 1 } ) _ { b , t } - \nabla ( \mathbf { y } _ { c l e a n } ) _ { b , t } \right) ^ { 2 } + \epsilon ^ { 2 } } ,\tag{13}
$$

$$
\mathcal { L } _ { \nabla ^ { 2 } } = \frac { 1 } { B T } \sum _ { b , t } \sqrt { \left( \nabla ^ { 2 } ( \hat { \mathbf { y } } _ { R 1 } ) _ { b , t } - \nabla ^ { 2 } ( \mathbf { y } _ { c l e a n } ) _ { b , t } \right) ^ { 2 } + \epsilon ^ { 2 } } .\tag{14}
$$

Here, B and $T$ denote the batch size and sequence length, respectively. $\hat { \bf y } _ { R 1 }$ and $\mathbf { y } _ { c l e a n }$ are the reconstructed and target clean reference signal, and ϵ is a small constant in the Charbonnier term. In $\mathcal { L } _ { \mathrm { s p e c } } ,$ real-valued fast Fourier transform (rFFT) is applied along the time axis of each sample. It is then normalized by $N _ { \mathrm { { f f t } } } = \Omega B$ , where Ω denotes the number of non-redundant frequency bins produced by the rFFT.

Only the parameters of the patchifier, the pass1 MACF backbone, and the pass1 reconstruction head are updated in this stage.

Stage 2: Patch-Level Uncertainty Estimation: In Stage 2, the pass1 reconstruction path trained during Stage 1 is frozen, and only the uncertainty head is trained. For uncertainty estimation, $\hat { \bf y } _ { R 1 }$ is first passed through the Stage 1 patchifier to obtain its patch-level representation $\mathbf { P } _ { R 1 } . \mathbf { P } _ { R 1 }$ and $\mathbf { P } _ { \mathrm { n o i s y } }$ from Stage 1 are concatenated along the feature dimension to form the input to the uncertainty head. The uncertainty head consists of two linear layers with LayerNorm and ReLU activation between them, followed by a Softplus activation.

![](images/984258822c1ebc2fd27be8a3313b180865f50a0e4eac17254237e30f31b5b521.jpg)  
Fig. 2: Three-stage training strategy and the model overview. Frozen layers at each stage are overlaid with gray squares. (a) Stage 1 overview, only the parameters of the pass1 reconstruction path are updated. (b) Stage 2, where only the parameters of the uncertainty head are updated. (c) Stage 3, only the parameters of the pass2 refinement path are updated.

Stage 2 uses Gaussian negative log-likelihood loss to train the model to estimate patch-level aleatoric uncertainty [33]. Stage 2 loss can be expressed as

$$
e _ { b , p } = \frac { 1 } { s } \sum _ { t \in \mathcal { P } _ { p } } \left( ( \hat { \mathbf { y } } _ { R 1 } ) _ { b , t } - ( \mathbf { y } _ { c l e a n } ) _ { b , t } \right) ^ { 2 } ,\tag{15}
$$

$$
\mathcal { L } _ { \mathrm { S 2 } } = \frac { 1 } { B P } \sum _ { b , p } \left( \frac { e _ { b , p } } { 2 U _ { b , p } } + \frac { 1 } { 2 } \log U _ { b , p } \right) .\tag{16}
$$

where B, P, and s denote the batch size, the number of temporal patches per sample, and the number of time points in each patch, respectively. $( \hat { \mathbf { y } } _ { R 1 } ) _ { b , t }$ and $( \mathbf { y } _ { \mathrm { c l e a n } } ) _ { b , t }$ denote the values at time point t of the b-th first-pass denoised output and the corresponding clean reference signal, respectively. $e _ { b , p }$ is the patch-level MSE for patch $p ,$ and $U _ { b , p }$ denotes the $( b , p )$ -th element of the predicted uncertainty map U, with $U _ { b , p } > 0$

Stage 3: Uncertainty-Guided Refinement Learning: In Stage 3, the pass1 reconstruction path and the uncertainty head are frozen, and only the parameters of the pass2 refinement path are updated. Stage 3 leverages the outputs of both Stage 1 and Stage 2. It first takes the uncertainty map $\mathbf { U } \in \mathbb { R } ^ { \bar { \boldsymbol { B } } \times \boldsymbol { P } }$ produced by the uncertainty head. Based on U, a core binary mask I is obtained by thresholding the uncertainty values with the threshold τ as

$$
I _ { b , p } = \mathbf { 1 } ( U _ { b , p } > \tau ) , \qquad \mathbf { I } \in \{ 0 , 1 \} ^ { B \times P } .\tag{17}
$$

Here, b and $p$ denote batch and patch indices, respectively. Patches whose uncertainty exceeds the threshold are treated as unreliable.

Masking is applied to $\mathbf { Z } _ { R 1 }$ , the first-pass latent representation, in two ways. The first approach is neighbor-expanded uncertainty masking, which serves as the primary masking

policy used consistently during both training and inference. In this masking approach, core mask $I _ { b , p }$ is expanded to 1- hop temporal neighbors as shown in Eq. (18).

$$
M _ { b , p } = I _ { b , p - 1 } \vee I _ { b , p } \vee I _ { b , p + 1 } , \qquad { \bf M } \in \{ 0 , 1 \} ^ { B \times P } .\tag{18}
$$

The purpose of this masking expansion is to prevent a single patch masking from creating sharp transitions at patch boundaries. The second masking approach is random block masking, which is used only during training. For each sample, a target mask ratio is first sampled, and contiguous patch blocks with random start positions and lengths are then masked until the target masked-patch budget is reached. This strategy mainly serves as a fallback when the uncertainty-driven mask is empty.

Once $\mathbf { Z } _ { R 1 }$ is masked, it is processed by the pass2 MACF backbone and the pass2 reconstruction head to produce the second-pass denoised signal. The objective of the Stage 3 refinement path is to first mask unreliable patches and refine such regions. Thus, the Stage 3 loss is defined as

$$
\mathcal { L } _ { \mathrm { m a s k e d - M S E } } = \frac { \sum _ { b , p } \sum _ { t \in \mathcal { P } _ { p } } M _ { b , p } ( ( \hat { \mathbf { y } } _ { R 2 } ) _ { b , t } - ( \mathbf { y } _ { c l e a n } ) _ { b , t } ) ^ { 2 } } { s \sum _ { b , p } M _ { b , p } } ,\tag{19}
$$

$$
\mathcal { L } _ { \mathrm { g l o b a l - M S E } } = \frac { 1 } { B T } \sum _ { b , t } \left( ( \hat { \mathbf { y } } _ { R 2 } ) _ { b , t } - ( \mathbf { y } _ { c l e a n } ) _ { b , t } \right) ^ { 2 } ,\tag{20}
$$

$$
\mathcal { L } _ { \mathrm { S 3 } } = \alpha _ { m } \mathcal { L } _ { \mathrm { m a s k e d - M S E } } + \mathcal { L } _ { \mathrm { g l o b a l - M S E } } .\tag{21}
$$

where $\mathcal { P } _ { p }$ denotes the timepoints that belong to p-th patch, M the binary mask, and $\hat { \bf y } _ { R 2 }$ the pass2 denoised output.

$\mathcal { L } _ { \mathrm { m a s k e d - M S E } }$ concentrates supervision on patches selected for refinement, while $\mathcal { L } _ { \mathrm { g l o b a l - M S E } }$ provides a global constraint on the full waveform to prevent discontinuities at the boundaries of masked and unmasked regions. $\mathcal { L } _ { \mathrm { m a s k e d - M S E } }$ and $\mathcal { L } _ { \mathrm { g l o b a l - M S E } }$ are used together, with a larger weight assigned to $\mathcal { L } _ { \mathrm { m a s k e d - M S E } }$ $( \alpha _ { m } = 5$ in our experiments) to balance targeted refinement and overall signal fidelity.

## D. Two-Pass Inference

Fig. 1a depicts the inference workflow of MURFormer. The noisy input is first processed by the pass1 reconstruction path to obtain the first-pass denoised output $\hat { \bf y } _ { R 1 }$ . A patch-level uncertainty map is then predicted and thresholded to identify unreliable regions as described in Section II-C and Eq. (17). If no patch exceeds the uncertainty threshold τ, the first-pass output $\hat { \bf y } _ { R 1 }$ is used as the final denoised signal. Otherwise, the refinement mask is applied to the first-pass latent sequence $\mathbf { Z } _ { R 1 }$ . The masked latent sequence is subsequently processed by the pass2 refinement path and its output $\hat { \bf y } _ { R 2 }$ is used as the final denoised signal.

## III. EXPERIMENTAL SETUP

## A. Dataset

We used the publicly available CIME dataset and a Daily Activities dataset collected in this study. These datasets include noisy-clean PPG pairs which are suitable for training MURFormer. Although the IEEE Signal Processing Cup 2015 Database (ISPC) contains noisy-clean PPG pairs, it was not used due to its limited size [2].

1) CIME Dataset: For training and evaluation, we used the public dataset CIME [34], [35], which contains paired noisy and clean finger PPG recordings from 48 individuals. The signals were acquired simultaneously from the left and right index fingertips, with the left finger kept still and the right finger performing movements.

2) Daily Activities Dataset: We collected a paired noisyclean PPG dataset, referred to in this paper as the Daily Activities dataset, to support both supervised noisy-to-clean PPG denoising and evaluation under realistic daily-life motion conditions. Signals were recorded at 40 Hz using a custom watch-type device built with an Arduino ESP32-based TinyPICO board and the Gravity: Heart Rate Monitor Sensor for Arduino.

During recording, each subject wore the devices on both wrists. The dominant hand performed the assigned task while the non-dominant hand was kept still to provide a relatively clean reference signal. Thus, PPG obtained from the dominant hand was used as the noisy signal and the PPG obtained from the non-dominant hand was used as the clean signal.

PPG was collected from a total of 24 participants, all of whom provided informed consent prior to participating in the data collection. Each subject participated in two recording sessions conducted on different days. In each session, eight tasks were performed in the fixed order of rest, writing, typing, gaming, walking, exercise, eating, and cleaning. Each task lasted 5 minutes, and a 1 minute rest period was provided between tasks. Most tasks were performed in a seated position, whereas walking and exercise were performed while standing. The tasks were designed to induce motion artifacts representative of daily activities while keeping the non-dominant hand as still as possible. The activity tasks included writing, onehanded typing, touchscreen gaming, walking in place with dominant-arm swing, one-sided stretching exercise, eating with chopsticks, and cleaning-related object manipulation.

The study protocol was approved by the Institutional Review Board of Korea Advanced Institute of Science and Technology (KAIST) under Application No. KAISTIRB-2025-279 on November 25, 2025.

## B. Data Preprocessing

The CIME and Daily Activities datasets are used throughout the experiments for training and evaluation. For the Daily Activities dataset, peak alignment is performed before the main preprocessing pipeline. For each session, we estimate the average peak time offset between the band-pass filtered clean and noisy PPG recorded during the resting task. The offset is then used to temporally align each signal pair by shifting one signal relative to the other.

After peak alignment, both CIME and Daily Activities datasets are processed using the same preprocessing pipeline. Signals are segmented into overlapping 10-second windows with a stride of 0.25 s. A band-pass filter of 0.5–4.0 Hz is then applied. Next, peak rejection is performed on the clean windows to exclude low-quality segments, including those that may still contain residual motion artifacts despite being recorded from the relatively still hand. In the peak rejection step, peaks are detected from the clean PPG, and each window is retained only if both signal-quality criteria are satisfied, with the coefficient of variation of RR intervals limited to 0.1 or less and the coefficient of variation of peak amplitudes limited to 0.3 or less. Lastly, sample-wise z-score normalization is applied to each window.

## C. Experimental Settings

All experiments use the same optimization and model settings unless otherwise stated. For each experiment, the compared models are trained and evaluated separately on the Daily Activities dataset and the CIME dataset. MURFormer is trained with the AdamW optimizer using a learning rate of $4 \times 1 0 ^ { - 4 }$ and a batch size of 256 for all three stages. Early stopping is applied with a patience of 20 epochs. The patch size of the patchifier is set to 10 for the Daily Activities dataset and 25 for the CIME dataset. The embedding dimension is 64, with 4 attention heads and a dropout rate of 0.1. The pass1 MACF backbone contains 8 layers, whereas the pass2 MACF backbone contains 4 layers. In the MACF block, the convolution kernel size is 5, the small-scale and large-scale partition sizes are 5 and 20, respectively.

Stages 1, 2, and 3 are trained using losses defined in Eqs.(10), (16), and (21), respectively. For the Stage 1 and Stage 3 losses from Eq.(10) and Eq.(21), $\alpha , ~ \beta , ~ \gamma .$ , and $\alpha _ { m }$ are set to 5. For Stage 1, the best model is selected based on the validation reconstruction error. For Stage 2, the best model is selected based on the validation AUROC for detecting higherror patches. For Stage 3, the best model is selected based on the validation loss. The uncertainty threshold τ used in Eq.(17) is determined from the validation set, where the threshold is set to the 95th percentile of validation uncertainty values.

For the Daily Activities dataset, we adopt a leave-onesubject-out (LOSO) setting. In each fold, one subject is reserved for testing, and the remaining subjects are used for training and validation. Specifically, Session 1 data from the remaining subjects are used for training and Session 2 data are used for validation. For the CIME dataset, we use data from 10 subjects for training and the remaining 38 subjects for testing following the original split [34], [35].

## D. Experiment 1: Transformer Encoder vs MACF Encoder

The first experiment compares the proposed MACF block with a standard Transformer encoder under the same reconstruction framework. Both models use the same patchifier– encoder–reconstruction head architecture as the pass1 reconstruction path of MURFormer, differing only in the encoder block, which is implemented as either a standard Transformer encoder or the proposed MACF block. For the MACF-based model, we use the Stage 1 model from Experiment 2 (Section III-E), which corresponds to the first-pass reconstruction path of MURFormer trained for direct reconstruction. The encoder depth is fixed to 8 layers in both models. This experiment evaluates whether MACF provides a more effective encoder for direct noisy-to-clean PPG reconstruction.

## E. Experiment 2: Direct Reconstruction vs Three-Stage Training

The second experiment is designed to justify the proposed three-stage training strategy and uncertainty-guided two-pass inference. We compare the Stage 1 model with the final model obtained after Stage 3 training. The Stage 1 model performs direct reconstruction using only the first-pass reconstruction path, whereas the final model applies two-pass inference after the full three-stage training.

We also validate if the uncertainty head identifies unreliable regions accurately by treating it as a patch-level detector of high-error regions. On the held-out test set, patches whose Stage 1 reconstruction MSE falls in the top 10% (q=10%) are labeled as positives, and AUROC/AUPRC are computed over all decision thresholds using predicted uncertainty as the score. For the Daily Activities dataset, AUROC and AUPRC are computed for each held-out subject and then averaged across all 24 folds.

## F. Experiment 3: Baseline Comparison

Finally, the third experiment compares MURFormer with the diffusion-based denoising model proposed by Xia et al. [27], which was developed for the same task. The goal of this experiment is to directly compare the denoising performance of MURFormer against a recent baseline. Among recent learning-based PPG denoising methods, this model was selected as the primary baseline because it is reproducible and can be adapted to our experimental setting with clearly specified implementation details. Other recent methods [23], [28] are discussed as related work, but were not included as direct baselines due to limited reproducibility under the same training and evaluation protocol. In this experiment, maxabsolute scaling which rescales the maximum absolute value to 1 is used instead of z-score normalization, following the original design of the diffusion baseline. This is because the diffusion model assumes bounded signal amplitudes during preprocessing and sampling, including clipping. Accordingly, to ensure a fair comparison, the input to MURFormer is also normalized using max-absolute scaling.

In addition, to adapt the diffusion baseline to the datasets in the experiment, we adjusted the spectrogram configuration according to the sampling rate and window length while preserving a temporal resolution close to that of the original signal. For the Daily Activities dataset, we use a hop size of 10 samples, 40 mel frames, an FFT size of 64, 8 mel bands, and a maximum mel frequency of 5 Hz. For the CIME dataset, we use a hop size of 25 samples and an FFT size of 128, while keeping the remaining settings identical to those used for the Daily Activities dataset. All other preprocessing steps follow the settings stated in Section III-B. The training settings of the diffusion baseline follow those of the original work, with a batch size of 128, a learning rate of 0.001, and 100,000 training iterations. MURFormer is trained using the same settings described in Section III-C.

## IV. RESULTS AND DISCUSSION

In this section, we demonstrate that the proposed MACF block and uncertainty-guided two-pass framework enable MURFormer to effectively denoise motion-corrupted signals and improve denoising performance across the evaluated settings. We evaluate and compare denoising performance using signal-to-noise ratio (SNR) and physiological metrics derived from the reconstructed PPG. In addition to heart rate (HR), we report peak-to-peak interval (PPI), standard deviation of RR intervals (SDNN), and root mean square of successive differences (RMSSD) to assess whether the denoised PPG preserves beat-to-beat timing and heart-rate-variability (HRV) measures beyond waveform-level signal quality. These metrics are extracted from the denoised PPG and compared with the corresponding values obtained from the paired clean reference PPG. For HR, PPI, SDNN, and RMSSD, both mean absolute error (MAE) and root mean square error (RMSE) are reported. HR-related errors are expressed in beats per minute, whereas PPI, SDNN, and RMSSD-related errors are expressed in milliseconds.

## A. Experiment 1 Results: Transformer Encoder vs MACF Encoder

Table I compares the direct reconstruction performance of the Transformer-based denoiser and the MACF-based denoiser, where the latter corresponds to the pass1 reconstruction path of MURFormer. On the Daily Activities dataset, the MACF-based model outperformed the Transformer-based model across all reported metrics, with statistically significant improvements in both SNR and HR MAE across 24 subjects according to the Wilcoxon signed-rank test(SNR, p = 1.19 × $1 0 ^ { - 7 } ; \mathrm { H R } , p = 4 . 3 0 \times 1 0 ^ { - 4 } )$ , as well as substantially lower PPI and HRV-related errors. Similar improvements were observed on the CIME dataset, where the MACF-based model increased SNR from 10.81 to 11.12 dB, reduced HR MAE from 0.96 to 0.91 bpm, and showed consistently lowered PPI and HRVrelated errors. This indicates that the proposed MACF block provides a more effective encoder for direct noisy-to-clean reconstruction. Notably, this gain was achieved with approximately 15% fewer parameters than the Transformer-based denoiser as shown in Table II. This reduction in overall model size is mainly attributed to the MACF block, which is more parameter-efficient than a standard Transformer encoder under the same embedding dimension. These results support the effectiveness of MACF as a parameter-efficient encoder for noisy-to-clean PPG reconstruction.

TABLE I: Experiment 1: Comparison between the Transformer-based denoiser and the MACF-based denoiser (MURFormer pass1). Bold values indicate the better result within each dataset.
<table><tr><td>Dataset</td><td>Model</td><td>SNR Mean</td><td>HR MAE</td><td>HR RMSE</td><td>PPI MAE</td><td>PPI RMSE</td><td>SDNN MAE</td><td>SDNN RMSE</td><td>RMSSD MAE</td><td>RMSSD RMSE</td></tr><tr><td rowspan="2">Daily Activities</td><td>Transformer-based</td><td>7.29</td><td>1.88</td><td>4.60</td><td>15.93</td><td>38.58</td><td>35.76</td><td>64.40</td><td>54.97</td><td>94.73</td></tr><tr><td>MACF-based</td><td>8.45</td><td>1.52</td><td>4.17</td><td>12.34</td><td>32.67</td><td>14.28</td><td>30.21</td><td>18.54</td><td>36.51</td></tr><tr><td rowspan="2">CIME</td><td>Transformer-based</td><td>10.81</td><td>0.96</td><td>3.14</td><td>9.36</td><td>29.94</td><td>15.64</td><td>34.35</td><td>22.62</td><td>46.69</td></tr><tr><td>MACF-based</td><td>11.12</td><td>0.91</td><td>3.25</td><td>8.79</td><td>30.20</td><td>10.61</td><td>24.53</td><td>13.76</td><td>29.00</td></tr></table>

TABLE II: Parameter comparison between Transformer-based and MACF-based designs under the Daily Activities setting.
<table><tr><td>Component</td><td>Transformer-based</td><td>MACF-based</td><td>Reduction (%)</td></tr><tr><td>Encoder block</td><td>33,472</td><td>28,320</td><td>15.4%</td></tr><tr><td>Full model</td><td>273,601</td><td>232,321</td><td>15.1%</td></tr></table>

![](images/4dd6b9f9f9297abd604a04d841818798d9c5091a11dc8e5c55e13c90a2610fb5.jpg)  
Fig. 3: Comparison of Stage 1 and Stage 3 outputs. Gray regions indicate patches exceeding the uncertainty threshold, highlighting unreliable regions of Stage 1 identified by the uncertainty head.

## B. Experiment 2 Results: Direct Reconstruction vs Three-Stage Training

Table III presents the comparison between the Stage 1 direct-reconstruction model with the final Stage 3 model of MURFormer, which applies the full uncertainty-guided twopass framework after three-stage training.

The final Stage 3 model consistently improved upon the Stage 1 direct-reconstruction model on both datasets. Although the absolute improvement was modest, the improvements in SNR and HR MAE on the Daily Activities dataset were statistically significant according to the Wilcoxon signed-rank test (SNR, $p \ = \ 0 . 0 2 9 ;$ HR, $p = 6 . 5 0 \times 1 0 ^ { - 4 } )$ . The Stage 3 model outperformed the Stage 1 model across all reported metrics on both datasets except for a slight increase in PPI RMSE on CIME, supporting the role of the uncertainty-guided second-pass refinement.

To verify the usefulness of the uncertainty-guided twopass approach in denoising, Table IV reports the high-error patch detection performance on the test set. The uncertainty head achieved an AUROC of 0.79 on both datasets, with AUPRC values of 0.33 and 0.36 on the Daily Activities and CIME datasets, respectively. Since the positive class ratio is 10%, these AUPRC values are substantially above the random baseline, indicating that the predicted uncertainty is reasonably aligned with the actual reconstruction error after the first pass, which supports its use in Stage 3 refinement.

Fig. 3 shows a representative example of how the proposed method refines the initial denoised output for a single noisy sample. In the figure, the blue curve denotes the Stage 1 output and the purple curve denotes the Stage 3 output. Gray regions correspond to patches whose uncertainty values exceed the uncertainty threshold according to the uncertainty map predicted by the uncertainty head. Although the Stage 1 reconstruction output captures the overall waveform, some local regions remain imperfectly restored. The uncertainty head identifies these regions and the Stage 3 output further corrects them.

These results suggest that the first-pass MACF reconstruction removes most artifacts, while the uncertainty-guided second pass provides additional localized correction with the help of the uncertainty head.

## C. Experiment 3 Results: Baseline Comparison

Table V compares the denoising performance of MUR-Former and the diffusion-based baseline [27] under the same max-absolute scaling setting. Both models improved substantially over the noisy input, but MURFormer achieved better overall performance on both datasets. On the Daily Activities dataset, MURFormer yielded higher SNR, lower HR, PPI, and RMSSD errors than the diffusion baseline. On the CIME dataset, MURFormer outperformed the diffusion baseline across all reported metrics, significantly increasing SNR from 9.60 to 10.72 dB and reducing HR MAE from 1.41 to 0.82 bpm (SNR, $p = 3 . 6 6 \times 1 0 ^ { - 5 } ;$ HR, $p = 0 . 0 4 2 )$ It also achieved lower errors for PPI, SDNN, and RMSSD, indicating improved preservation of both beat-to-beat timing and HRV-related characteristics.

Fig. 4 further supports this result by showing Pearson correlation plots for HR and PPI on the CIME dataset. Compared with the diffusion-denoised output, the MURFormer-denoised output showed stronger agreement with the clean PPG-derived HR and PPI reference values. MURFormer achieved higher correlations for both HR and PPI (Fig. 4c, $r = 0 . 9 7 5 5$ and $r = 0 . 9 7 6 0 )$ than the diffusion baseline (Fig. 4b, $r = 0 . 9 4 1 6$ and $r ~ = ~ 0 . 9 4 6 0 )$ . The MURFormer estimates were more tightly concentrated around the identity line.

TABLE III: Experiment 2: Comparison among the noisy input, Stage 1 direct reconstruction, and the final Stage 3 model of MURFormer. Bold values indicate the better result within each dataset.
<table><tr><td>Dataset</td><td>Model</td><td>SNR Mean</td><td>HR MAE</td><td>HR RMSE</td><td>PPI MAE</td><td>PPI RMSE</td><td>SDNN MAE</td><td>SDNN RMSE</td><td>RMSSD MAE</td><td>RMSSD RMSE</td></tr><tr><td rowspan="3">Daily Activities</td><td>Noisy</td><td>3.72</td><td>4.02</td><td>7.47</td><td>34.49</td><td>65.95</td><td>79.77</td><td>115.50</td><td>116.85</td><td>166.66</td></tr><tr><td>Stage 1</td><td>8.45</td><td>1.52</td><td>4.17</td><td>12.34</td><td>32.67</td><td>14.28</td><td>30.21</td><td>18.54</td><td>36.51</td></tr><tr><td>Stage 3</td><td>8.51</td><td>1.38</td><td>3.97</td><td>11.31</td><td>31.50</td><td>10.94</td><td>23.41</td><td>13.86</td><td>26.27</td></tr><tr><td rowspan="3">CIME</td><td>Noisy</td><td>5.44</td><td>4.53</td><td>9.07</td><td>41.98</td><td>81.71</td><td>60.65</td><td>90.54</td><td>87.52</td><td>127.70</td></tr><tr><td>Stage 1</td><td>11.12</td><td>0.91</td><td>3.25</td><td>8.79</td><td>30.20</td><td>10.61</td><td>24.53</td><td>13.76</td><td>29.00</td></tr><tr><td>Stage 3</td><td>11.21</td><td>0.88</td><td>3.23</td><td>8.58</td><td>30.28</td><td>9.52</td><td>22.36</td><td>11.99</td><td>25.00</td></tr></table>

TABLE IV: Uncertainty validation results on test set. Higherror patch detection AUROC/AUPRC (q=10%)
<table><tr><td>Dataset</td><td>AUROC(q=10%)</td><td>AUPRC(q=10%)</td></tr><tr><td>Daily Activities</td><td>0.79</td><td>0.33</td></tr><tr><td>CIME</td><td>0.79</td><td>0.36</td></tr></table>

![](images/17922bfe87ed00e6c50beedb3a7b552907b7e5b6c1c4dabaf8e3d8d982ae59ba.jpg)

![](images/db27a2450d71c802ad727e736b538a411b7bb1131c14b103f1716a9fa0cb9404.jpg)  
(a)

![](images/035cf879221dac61cac4b5c474f3469e2b2fed670eba3f958ae3b272457f13af.jpg)

![](images/e8943733634a7bd1780fac1e6126478d6acb2399291905c200f66c72390d0c4b.jpg)  
(b)

![](images/eab42283b85adcff23ac3272057fd3297d62fa761881886114da656535764d44.jpg)

![](images/d9a646c9ace9d6236234326ffc0791b203d1e696096841d70472b41762d67f4c.jpg)  
(c)  
Fig. 4: Pearson correlation plots of HR and PPI estimates on the CIME dataset. The plots compare estimates derived from (a) noisy PPG, (b) diffusion-denoised PPG, and (c) MURFormer-denoised PPG against the clean PPG-derived reference values.

## D. Comparison with Noisy Input

We next examine the improvement of MURFormer over the original noisy input using the standard preprocessing setting in

![](images/f7211710e5eeb4aea66ab3ee404e166a07705cf6a15987f05c5f4b51aac74694.jpg)  
Fig. 5: Comparison of noisy PPG, denoised output, and clean PPG.

Table III. On the Daily Activities dataset, MURFormer (Stage 3) increased SNR from 3.72 to 8.51 dB and reduced HR MAE from 4.02 to 1.38 bpm. On the CIME dataset, SNR increased from 5.44 to 11.21 dB while HR MAE decreased from 4.53 to 0.88 bpm. These improvements were accompanied by consistent reductions in PPI, SDNN, and RMSSD errors, indicating that MURFormer substantially improves waveformlevel signal quality and beat timing and HRV-related physiological fidelity.

The correlation plots in Fig. 4 also show that MURFormerdenoised estimates (Fig. 4c) align more closely with the clean reference than estimates derived from the noisy PPG (Fig. 4a).

A qualitative example shown in Fig. 5 illustrates how MURFormer recovers the PPG waveform even when the input is heavily corrupted. Notably, the model recovers not only the major systolic peaks but also the smaller diastolic peaks, suggesting that the proposed framework can preserve the finegrained morphological structures of the PPG waveform.

## E. MURFormer on Varying Input Window Sizes

MURFormer is designed to accept variable-length input sequences. We further evaluate the robustness of MURFormer to varying input lengths by using input windows of different durations at test time. The models trained using 10-s windows from a previous experiment in Section IV-B are directly applied to test inputs of 5 s, 10 s, 30 s, and 60 s. For the Daily Activities dataset, models trained under the leave-one-subjectout protocol were reused, and each model was evaluated on its corresponding held-out subject using different input window sizes. For CIME, the 60-s setting was not considered because each trial is 30 s long.

Fig. 6 illustrates the SNR, HR, PPI, SDNN, and RMSSD MAEs of the noisy PPG signals and the MURFormer-denoised outputs on each dataset across different input window sizes. The results show that MURFormer maintains strong denoising performance across all tested input lengths. For example, on the Daily Activities dataset, HR MAE decreased from 4.70 to 1.95 bpm at 5 s and from 3.21 to 0.84 bpm at 60 s. On CIME, HR MAE decreased from 4.67 to 1.18 bpm at 5 s and from 3.16 to 0.47 bpm at 30 s. Consistent trends were observed across the remaining metrics, as MURFormer improved SNR and reduced PPI, SDNN, and RMSSD errors in all input lengths tested.

TABLE V: Experiment 3: Comparison among the noisy input, the diffusion-based baseline, and MURFormer with max-absolute scaling. Bold values indicate the best result within each dataset.
<table><tr><td>Dataset</td><td>Model</td><td>SNR Mean</td><td>HR MAE</td><td>HR RMSE</td><td>PPI MAE</td><td>PPI RMSE</td><td>SDNN MAE</td><td>SDNN RMSE</td><td>RMSSD MAE</td><td>RMSSD RMSE</td></tr><tr><td>Daily Activities</td><td>Noisy</td><td>4.04</td><td>4.02</td><td>7.47</td><td>34.49</td><td>65.95</td><td>79.77</td><td>115.50</td><td>116.85</td><td>166.66</td></tr><tr><td></td><td>Diffusion</td><td>7.47</td><td>1.64</td><td>4.77</td><td>13.67</td><td>39.56</td><td>10.61</td><td>22.35</td><td>14.23</td><td>28.41</td></tr><tr><td></td><td>MURFormer</td><td>8.16</td><td>1.37</td><td>3.92</td><td>11.20</td><td>31.09</td><td>11.07</td><td>23.66</td><td>14.04</td><td>26.89</td></tr><tr><td>CIME</td><td>Noisy</td><td>5.53</td><td>4.53</td><td>9.07</td><td>41.99</td><td>81.71</td><td>60.65</td><td>90.54</td><td>87.52</td><td>127.70</td></tr><tr><td></td><td>Diffusion</td><td>9.60</td><td>1.41</td><td>4.47</td><td>13.58</td><td>42.10</td><td>11.76</td><td>24.90</td><td>14.42</td><td>27.63</td></tr><tr><td></td><td>MURFormer</td><td>10.72</td><td>0.82</td><td>2.96</td><td>7.99</td><td>28.03</td><td>9.16</td><td>21.04</td><td>11.42</td><td>22.83</td></tr></table>

TABLE VI: Comparison of the number of model parameters among recent PPG denoising models and MURFormer.
<table><tr><td>Model</td><td>Number of Parameters</td></tr><tr><td>FlexiMA [28]</td><td>109.8M</td></tr><tr><td>CycleGAN [21]</td><td>94.35M</td></tr><tr><td>Diffusion [27]</td><td>2.3M</td></tr><tr><td>MURFormer</td><td>0.36M</td></tr></table>

MURFormer Performance Across Input Window Sizes

![](images/100dfe75f5865f757734d06b457d4eba5c22354c0727ae4c2874a7b2d2c956ac.jpg)  
Fig. 6: Comparison of noisy PPG and denoised output of MURFormer across different input window sizes at test time.

These results indicate that MURFormer generalizes well to a range of input lengths at inference time even when it was trained using 10-s windows. While the best SNR is obtained near the original training scale, longer input windows tend to provide lower HR and HRV-related errors, likely because they provide more temporal context for stable physiological estimation. Overall, the proposed method shows robust denoising performance under varying input window sizes, suggesting that the learned representation is not tightly restricted to a single input duration.

## F. Model Size

Table VI compares the number of parameters of MUR-Former with those of recent learning-based PPG denoising models. For the diffusion-based model [27], the parameter count was estimated from the open-source implementation used in our baseline experiment. For CycleGAN [21] and FlexiMA [28], whose implementations are not publicly available, we report the parameter counts stated in the paper. MURFormer uses approximately 6.4 times fewer parameters than the diffusion baseline and has a substantially smaller parameter count than FlexiMA and CycleGAN. This result highlights the compactness of MURFormer relative to recent learning-based PPG denoising models.

## V. CONCLUSION

This study proposes MURFormer, a lightweight uncertaintyguided two-pass framework for PPG denoising. Experimental results on the self-collected Daily Activities dataset under a cross-subject evaluation protocol and on the CIME dataset demonstrate that MURFormer improves signal quality and physiological fidelity while outperforming the diffusion-based baseline. The model also maintained strong denoising performance across varying input window lengths. These gains were achieved with a substantially smaller number of parameters than recent models, supporting its suitability for practical deployment. Moreover, the proposed MACF block was designed to reflect the temporal and multi-scale dependencies of PPG, and its favorable balance between parameter efficiency and reconstruction performance in PPG denoising suggests that it may serve as a useful basic building block for broader PPGrelated tasks beyond denoising.

## CONFLICT OF INTEREST STATEMENT

The authors declare that they have no known financial or personal conflicts of interest that could have influenced the work reported in this paper.

## REFERENCES

[1] J. Allen, “Photoplethysmography and its application in clinical physiological measurement,” Physiological measurement, vol. 28, no. 3, pp. R1–R39, 2007.

[2] Z. Zhang, Z. Pi, and B. Liu, “Troika: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise,” IEEE Transactions on biomedical engineering, vol. 62, no. 2, pp. 522–531, 2014.

[3] P. H. Charlton, J. Allen, R. Bailon, S. Baker, J. A. Behar, F. Chen, G. D.´ Clifford, D. A. Clifton, H. J. Davies, C. Ding et al., “The 2023 wearable photoplethysmography roadmap,” Physiological measurement, vol. 44, no. 11, p. 111001, 2023.

[4] D. Castaneda, A. Esparza, M. Ghamari, C. Soltanpur, and H. Nazeran, “A review on wearable photoplethysmography sensors and their potential future applications in health care,” International journal of biosensors & bioelectronics, vol. 4, no. 4, p. 195, 2018.

[5] P. H. Charlton, P. A. Kyriacou, J. Mant, V. Marozas, P. Chowienczyk, and J. Alastruey, “Wearable photoplethysmography for cardiovascular monitoring,” Proceedings of the IEEE, vol. 110, no. 3, pp. 355–381, 2022.

[6] R. Ferizoli, P. Karimpour, J. M. May, and P. A. Kyriacou, “Arterial stiffness assessment using ppg feature extraction and significance testing in an in vitro cardiovascular system,” Scientific reports, vol. 14, no. 1, p. 2024, 2024.

[7] W. He, Y. Ye, L. Lu, Y. Cheng, Y. Li, and Z. Wang, “Robust heart rate monitoring for quasi-periodic motions by wrist-type ppg signals,” IEEE journal of biomedical and health informatics, vol. 24, no. 3, pp. 636–648, 2019.

[8] A. K. Maity, A. Veeraraghavan, and A. Sabharwal, “Ppgmotion: Modelbased detection of motion artifacts in photoplethysmography signals,” Biomedical Signal Processing and Control, vol. 75, p. 103632, 2022.

[9] G. Joseph, A. Joseph, G. Titus, R. M. Thomas, and D. Jose, “Photoplethysmogram (ppg) signal analysis and wavelet de-noising,” in 2014 annual international conference on emerging research areas: Magnetics machines and drives (AICERA/iCMMD). IEEE, 2014, pp. 1–5.

[10] R. Ahmed, A. Mehmood, M. M. U. Rahman, and O. A. Dobre, “A deep learning and fast wavelet transform-based hybrid approach for denoising of ppg signals,” IEEE sensors letters, vol. 7, no. 7, pp. 1–4, 2023.

[11] M. R. Ram, K. V. Madhav, E. H. Krishna, N. R. Komalla, and K. A. Reddy, “A novel approach for motion artifact reduction in ppg signals based on as-lms adaptive filter,” IEEE Transactions on Instrumentation and Measurement, vol. 61, no. 5, pp. 1445–1457, 2011.

[12] M. Raghuram, K. Sivani, and K. A. Reddy, “Use of complex emd generated noise reference for adaptive reduction of motion artifacts from ppg signals,” in 2016 international conference on electrical, electronics, and optimization techniques (ICEEOT). IEEE, 2016, pp. 1816–1820.

[13] K. T. Tanweer, S. R. Hasan, and A. M. Kamboh, “Motion artifact reduction from ppg signals during intense exercise using filtered x-lms,” in 2017 IEEE international symposium on circuits and systems (ISCAS). IEEE, 2017, pp. 1–4.

[14] Y. Ye, Y. Cheng, W. He, M. Hou, and Z. Zhang, “Combining nonlinear adaptive filtering and signal decomposition for motion artifact removal in wearable photoplethysmography,” IEEE Sensors Journal, vol. 16, no. 19, pp. 7133–7141, 2016.

[15] Y. Ye, W. He, Y. Cheng, W. Huang, and Z. Zhang, “A robust random forest-based approach for heart rate monitoring using photoplethysmography signal contaminated by intense motion artifacts,” Sensors, vol. 17, no. 2, p. 385, 2017.

[16] S. Gupta, A. Singh, A. Sharma, and R. K. Tripathy, “Exploiting tunable q-factor wavelet transform domain sparsity to denoise wrist ppg signals,” IEEE transactions on instrumentation and measurement, vol. 72, pp. 1– 12, 2023.

[17] F. Mohagheghian, D. Han, O. Ghetia, A. Peitzsch, N. Nishita, M. P. S. Nejad, E. Y. Ding, K. Noorishirazi, A. Hamel, E. M. Otabil et al., “Noise reduction in photoplethysmography signals using a convolutional denoising autoencoder with unconventional training scheme,” IEEE Transactions on Biomedical Engineering, vol. 71, no. 2, pp. 456–466, 2023.

[18] P. Jain, C. Ding, C. Rudin, and X. Hu, “A self-supervised algorithm for denoising photoplethysmography signals for heart rate estimation from wearables,” arXiv preprint arXiv:2307.05339, 2023.

[19] Y. Wang, I. Azimi, K. Kazemi, A. M. Rahmani, and P. Liljeberg, “Ppg signal reconstruction using deep convolutional generative adversarial network,” in 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2022, pp. 3387–3391.

[20] X. Zheng, M. Derakhshani, L. A. Barrett, V. M. Dwyer, and S. Hu, “Ppggan: An adversarial network to de-noise ppg signals during physical activity,” in 2022 IEEE International Conference on E-health Networking, Application & Services (HealthCom). IEEE, 2022, pp. 63–68.

[21] A. H. Afandizadeh Zargari, S. A. H. Aqajari, H. Khodabandeh, A. Rahmani, and F. Kurdahi, “An accurate non-accelerometer-based ppg motion artifact removal technique using cyclegan,” ACM Transactions on Computing for Healthcare, vol. 4, no. 1, pp. 1–14, 2023.

[22] P. Sawangjai and T. Wilaiprasitporn, “Ppganet: Removal of motion artifacts from the ppg signal using generative adversarial networks,” in 2023 IEEE Biomedical Circuits and Systems Conference (BioCAS). IEEE, 2023, pp. 1–5.

[23] P. Sawangjai, N. Seesawad, and T. Wilaiprasitporn, “Removal of motion artifacts from the ppg signal using attentive generative adversarial networks with dual discriminator,” IEEE Transactions on Instrumentation and Measurement, vol. 74, pp. 1–10, 2025.

[24] I. Chiu, Y.-T. Liu, K.-C. Wang, H.-Y. Wei, and Y. Tsao, “Robust photoplethysmography signal denoising via mamba networks,” arXiv preprint arXiv:2510.11058, 2025.

[25] J. H. Kwon, S. E. Kim, N. H. Kim, E. C. Lee, and J. H. Lee, “Preeminently robust neural ppg denoiser,” Sensors, vol. 22, no. 6, p. 2082, 2022.

[26] G. Basso, X. Long, R. Haakma, and R. Vullings, “Reduction of motion artifacts from photoplethysmography signals using learned convolutional sparse coding,” Physiological Measurement, vol. 47, no. 2, p. 025001, 2026.

[27] Z. Xia, Z. Luo, C.-H. Chen, and X. Shen, “An effective photoplethysmography denosing method based on diffusion probabilistic model,” IEEE Journal of Biomedical and Health Informatics, 2025.

[28] Q. Liu, H. Zou, Y. Wu, Y. Chen, H. Ma, and B. Wang, “Flexima: Robust motion artifact removal in ppg signals using an adversarial variational autoencoder,” IEEE Internet of Things Journal, 2026.

[29] C.-C. Wu, I.-W. Chen, and W.-C. Fang, “An implementation of motion artifacts elimination for ppg signal processing based on recursive least squares adaptive filter,” in 2017 IEEE biomedical circuits and systems conference (BioCAS). IEEE, 2017, pp. 1–4.

[30] H. Lee, H. Chung, and J. Lee, “Motion artifact cancellation in wearable photoplethysmography using gyroscope,” IEEE Sensors Journal, vol. 19, no. 3, pp. 1166–1175, 2018.

[31] P. Kasnesis, L. Toumanidis, A. Burrello, C. Chatzigeorgiou, and C. Z. Patrikakis, “Feature-level cross-attentional ppg and motion signal fusion for heart rate estimation,” in 2023 IEEE 47th Annual Computers, Software, and Applications Conference (COMPSAC). IEEE, 2023, pp. 1731–1736.

[32] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[33] A. Kendall and Y. Gal, “What uncertainties do we need in bayesian deep learning for computer vision?” Advances in neural information processing systems, vol. 30, 2017.

[34] K. Xu, X. Jiang, and W. Chen, “Photoplethysmography motion artifacts removal based on signal-noise interaction modeling utilizing envelope filtering and time-delay neural network,” IEEE Sensors Journal, vol. 20, no. 7, pp. 3732–3744, 2019.

[35] K. Xu, X. Jiang, S. Lin, C. Dai, and W. Chen, “Stochastic modeling based nonlinear bayesian filtering for photoplethysmography denoising in wearable devices,” IEEE Transactions on Industrial Informatics, vol. 16, no. 11, pp. 7219–7230, 2020.