# DR-net-Mamba: Selective State-Space Modeling for Long-Range ECG Time-Series Denoising

Basile Morel, Samuel Ruipérez-Campillo, Andreas P. Streich, Julia E. Vogt, Thomas Hofmann

Abstract— Electrocardiogram (ECG) recordings are corrupted by non-stationary noise sources that degrade diagnostic reliability, particularly in ambulatory and longduration recordings. Deep learning denoisers exist, but convolutional architectures are limited by their receptive field, transformer-based models scale quadratically with sequence length, and diffusion-based approaches incur prohibitive inference cost. We propose a Mambaaugmented model that inserts selective state-space blocks at the convolutional bottleneck, combining local feature extraction with long-range temporal modeling at linear complexity. We comprehensively evaluate the proposed model with respect to reconstruction fidelity, noise robustness, recording-length scaling, and downstream diagnostic classification across over 40 pathology classes. On synthetic and real datasets, our model achieves the highest SNR and lowest RMSE, with the Mamba advantage increasing with sequence length and in low-SNR regimes. On classification with two independent classifiers, the proposed Mambabased models achieve the best macro AUROC among all denoisers and improve over their convolutional base models. Calibration is more nuanced and classifier-dependent: denoising improves Binary Cross-Entropy and Brier score on Inception1D but often fails to beat the noisy input on ResNet1D-Wang, and the lead-specific Mamba variant is the only denoiser to improve both calibration metrics over the noisy baseline on both classifiers. Per-class analysis reveals a morphology-dependent benefit: Mamba substantially improves ST/T-change diagnoses, which depend on broad, context-sensitive waveforms.

Index Terms— Electrocardiogram, selective state-space models, Mamba, deep learning, physiological time series, signal theory, long-range temporal modeling.

## I. INTRODUCTION

Cardiovascular diseases remain the leading cause of mortality worldwide [1], motivating the development of scalable, automated screening tools. The electrocardiogram (ECG) is non-invasive, inexpensive, and widely deployed across clinical and ambulatory settings [2]. Recent work has shown that deep learning (DL) applied to ECG signals can aid in the detection and risk assessment of multiple cardiovascular conditions [3], [4], including structural abnormalities [5], ventricular dysfunction in adult [6] and pediatric populations [7], [8], ischemic disease [9], cardiomyopathies [10], arrhythmias [11], [12], or primary prevention [13]. However, downstream analyses depend critically on signal quality, which is often degraded by non-stationary, unstructured noise sources that evolve unpredictably over long recordings. When multiple contaminations overlap, the signal-to-noise ratio (SNR) can drop to levels at which both human readers and automated classifiers fail.

Traditional denoising methods based on filtering, wavelets, and signal decomposition can be effective in controlled settings but require expert-driven parameter tuning and degrade under realistic multi-source noise [14], [15]. DL has advanced the field substantially: convolutional autoencoders [16], [17], recurrent networks [18], GANs [19], [20], and diffusion models [21] all improve reconstruction fidelity over classical pipelines. Yet, supervised methods suffer from the scarcity of clean ground truth in clinical datasets [16], Transformer-based architectures face quadratic compute and memory scaling with sequence length [22], [23], and diffusion models incur high inference costs from iterative sampling [24]. These constraints are especially problematic for long ambulatory recordings, where denoising must operate over tens of seconds of continuous signal at low latency.

Selective state-space models, and Mamba in particular [25], offer a compelling alternative. By making the state-transition parameters input-dependent, Mamba combines long-range temporal modeling with linear-time complexity and efficient recurrent-style inference. Mamba has been explored for ECG classification [26], [27] and short-segment enhancement [28], yet its potential for ECG denoising under diverse noise conditions and recording lengths remains unexplored.

In this work, we propose DR-net-Mamba, a two-stage architecture that inserts selective state-space blocks at the bottleneck of a convolutional UNet encoder–decoder, extending the effective receptive field to the full sequence length while adding only linear computational overhead, with a compact architecture suited for deployment. A learnable log-compression layer normalizes dynamic range. We evaluate DR-net-Mamba across four complementary axes: (i) reconstruction fidelity on synthetic, clinical, and ambulatory datasets under simultaneous four-source noise at clinically calibrated levels; (ii) noise robustness across clinically representative SNR regimes and noise conditions; (iii) recording-length scaling, characterizing how the selective state-space bottleneck leverages increasing temporal context to improve sequence-to-sequence modeling; and (iv) downstream diagnostic classification of 44 PTB-XL pathology classes through two independent classifiers, evaluating whether denoising gains preserve or improve diagnostic accuracy in a clinically actionable setting. To the best of our knowledge, this is the first evaluation framework spanning reconstruction, noise robustness, length scaling, and clinical utility within a single study for ECG denoising. Our open code is publicly available at https://github.com/....

## II. RELATED WORK

a) Traditional ECG denoising.: Classical approaches to ECG denoising rely on frequency-domain filtering, adaptive and Kalman-based methods [29], empirical mode decomposition [30], independent component analysis [31], and wavelet transforms [32]. These methods require expert-driven parameter selection and degrade with non-stationary artifacts [33], motivating data-driven alternatives.

b) Deep learning for ECG denoising.: While DL has recently been applied to denoising across multiple cardiac signal domains [34], [35], [36], [37], the ECG is the most extensively studied, with data-driven methods learning denoising priors directly from examples rather than relying on hand-crafted filter design. Convolutional architectures range from fully convolutional autoencoders [16] to multi-scale UNet variants with channel attention and detail-restoration stages [17]. Recurrent approaches, including LSTM-based denoising autoencoders [18] and deeper recurrent–convolutional hybrids [38], model temporal dependencies explicitly but scale poorly to long sequences. Multi-scale filtering designs [39] and transformer-based denoisers [40] further improve fidelity, though transformers incur quadratic complexity in sequence length [22]. On the generative side, GAN-based frameworks [19], [20] and score-based diffusion models [21] achieve strong reconstruction quality and can support unpaired training via cycle-consistency [41], but diffusion models face high inference costs from iterative sampling [24]. A persistent challenge across supervised methods is the scarcity of clean ground truth in clinical datasets, where residual artifacts survive preprocessing and become part of the reconstruction target [16], [17].

c) Synthetic ECG generation for evaluation.: The absence of clean clinical references motivates the use of synthetic signals for controlled evaluation: we cannot access the true latent signal and its unknown generative process beneath observed noisy measurements. The dynamical model of [42] generates realistic PQRST morphology from a three-variable ODE, producing signals of arbitrary length without segment stitching and requiring no training data. Data-driven generators based on GANs [43], [44] and VAEs [45] can capture richer morphological distributions but inherit residual noise from their training corpora, limiting their utility as denoising benchmarks. We adopt the ODE-based generator for our synthetic experiments, as it provides verified clean ground truth and continuous-time integration.

d) Selective state-space models and Mamba.: Structured state-space models (SSMs) offer linear-time sequence modeling as an alternative to attention [25]. Mamba extends SSMs with input-dependent selectivity: the state-transition parameters (B, C, ∆) are computed from each input token, enabling the model to selectively propagate or forget information along the sequence. This yields time-varying dynamics with efficient hardware-aware implementations and recurrent-style inference, and can be interpreted through an implicit attention viewpoint [46].

Interest in Mamba for physiological signals has grown rapidly [47], and existing ECG applications fall into two groups. For classification, ECGMamba [26] replaces attention with stacked selective state-space blocks to categorize rhythms and morphologies, while MambaCapsule [27] couples a Mamba backbone with a capsule network for interpretable disease diagnosis; both target discrete labels rather than waveform reconstruction. For enhancement, MECGE [28] is, to our knowledge, the only prior Mamba-based ECG denoiser: it transforms the signal to the time–frequency domain via the STFT and applies bidirectional Mamba blocks along both the time and frequency axes to enhance the resulting spectrogram. This performs well on short segments, but its per-frame frequency sequences cause memory to scale with recording length, limiting use on long ambulatory data. Our work differs from these prior ECG applications of Mamba in three respects: (i) we operate directly in the time domain rather than on spectrograms, (ii) we use Mamba as a bottleneck augmentation to a convolutional backbone rather than as a standalone architecture, and (iii) we evaluate across reconstruction, robustness, length scaling, and downstream classification within a unified framework.

## III. METHODS

Let $\textbf { x } \in \ \mathbb { R } ^ { T }$ denote a clean single-lead ECG signal of T time steps. The observed signal $\tilde { \textbf { x } } = \textbf { x } + \textbf { v }$ is corrupted by zero-mean noise v comprising baseline wander, muscle artifact, electrode motion, and additive white Gaussian noise. An offline denoising model $\varphi$ produces the estimate $\hat { \textbf { x } } =$ $\varphi ( \tilde { \mathbf { x } } ) \approx \mathbf { x }$ from the full observed sequence.

## A. Architecture

1) Baseline architectures.: We benchmark against six models spanning the major architectural families. UNet [17]: a three-stage convolutional encoder–decoder adapted from the original 2D U-Net [48] for 1D signals, with decreasing kernel sizes to separate low- and high-frequency components. IMUNet [17]: extends UNet with deeper convolutional blocks, channel attention, additional skip connections, and a dilated context-contrast bottleneck. DR-net-UNet and DR-net-IMUNet [17]: two-stage pipelines that cascade either backbone with a multi-branch detail-restoration network operating at full resolution to recover fine waveform features. DRNN [18]: an LSTM-based denoising autoencoder that processes the signal sample-by-sample. DAE [16]: a fully convolutional autoencoder with 13 symmetric layers and stride-based downsampling. MECGE [28]: a Mamba-based enhancer operating in the STFT domain (described above). These baselines span convolutional, recurrent, and state-space paradigms, providing a comprehensive reference for evaluating the proposed architecture.

2) Convolutional Backbone and its Receptive-Field Limitation: We adopt a 1D UNet encoder–decoder [17], [48] as our backbone. The encoder consists of three stages with convolutional kernel sizes decreasing from 25 to 3, interleaved with pooling layers (factors $1 / 5 , 1 / 2 , 1 / 2 )$ that progressively down-sample the input by a factor of 20. Skip connections link each encoder stage to the corresponding decoder stage, and bilinear upsampling restores the original resolution. While effective at extracting local features, the convolutional receptive field is bounded: a layer-by-layer analysis (Supplementary Material A) yields the receptive field $\begin{array} { r c l } { \dot { r } _ { 0 } } & { = } & { \dot { \sum _ { l = 1 } ^ { L } } \bigl ( ( k _ { l } } \stackrel { \cdot } { - }  \end{array}$ $\textstyle 1 ) \prod _ { i = 1 } ^ { l - 1 } s _ { i } ) + 1 \ = \ 5 4 2$ time steps—approximately 1.5 s at 360 Hz.

![](images/aaf62684da49d23f0673f13bca9600e48f1aa554f46941a5bf3aa7a91f56ffce.jpg)  
Fig. 1. Overall Pipeline: clean ECG records (a) are corrupted with noise (b), denoised (c), and classified (d) to assess whether reconstruction gains translate to diagnostic accuracy (e).

3) Selective State-Space Bottleneck: To extend the effective receptive field to the full sequence length, we insert three Mamba blocks [25] at the UNet bottleneck. At this stage, the signal has been down-sampled by 20×, so the blocks operate on a 1D sequence of length $\lfloor { T } / { 2 0 } \rfloor$ with 48 channels. Each block applies, in sequence: (i) layer normalization over the 48-dimensional feature vector at each time step; (ii) Mamba-1 layer [25], which (a) projects the input from dimension d=48 to $d _ { E } { = } 1 9 2$ via two parallel linear maps (main and gate branches) with expansion factor E=4, (b) applies a 1D depthwise convolution of width $d _ { \mathrm { c o n v } } { = } 4$ on the main branch for local context, (c) runs a selective state-space model with state dimension $d _ { \mathrm { s t a t e } } = 2 5 6 .$ , whose transition parameters (B, C, ∆) are computed from each input token, enabling timevarying propagation or forgetting of information along the sequence, (d) element-wise multiplies the SSM output with the SiLU-activated gate branch, and (e) projects back to 48 dimensions; and (iii) residual addition of the block input to the output. The optional MLP sub-layer $( d _ { \mathrm { i n t e r m e d i a t e } } { = } 0 )$ is omitted, as preliminary experiments showed no improvement in reconstruction while adding parameters. Similar experiments also showed that 3 Mamba blocks yield the best results; we denote the concrete models as UNet-Mamba1-3B. See Figure 2 for an overview over the architecture.

4) Two-Stage Detail Restoration: Following [17], we optionally cascade the UNet-Mamba with a detail restoration network (DR-net). The DR-net receives a two-channel input— the concatenation of the noisy ECG x˜ and the first-stage estimate xˆ—and produces a refined single-channel output. It comprises four parallel branches (kernel sizes 3, 5, 13, 15) with residual blocks and channel attention, operating at full resolution to preserve sharp features that may be attenuated by the bottleneck. We denote the full two-stage model as DR-net-Mamba, and the specific variant with 3 Mamba-Blocks (found to yield optimal performance in preliminary experiments) as DR-net-Mamba1-3B.

5) Learnable Signal Compression: We apply a learnable log-compression to the network input: $f _ { \alpha } ( x ) ~ = ~ \mathrm { s i g n } ( x )$ $\log ( 1 + \alpha \left| x \right| )$ , where $\alpha > 0$ is a trainable scalar. The inverse transform $f _ { \alpha } ^ { - 1 }$ is applied to the network output to recover the original amplitude scale. By factoring out the dynamic range, the compression layer allows the model to devote capacity to morphological structure rather than amplitude variation. Further, compressing the input range may stabilize gradient flow through the recurrent Mamba layers since learning long-range dependencies is prone to exploding and vanishing gradients even with input-dependent recurrence matrices [49], [50]. See Supplementary Material B (Figure S1) for ablations exploring this benefit.

6) ParameterCount and Memory Scaling: The three Mamba blocks add a total of 533,088 parameters to the 227Kparameter UNet backbone, yielding a total of 759,826 parameters, which is still smaller than our biggest baseline (MECGE, with 819K parameters). Critically, Mamba is applied only along the time axis after 20× down-sampling, keeping the effective batch size constant regardless of input length. By contrast, MECGE’s frequency-axis Mamba branch creates one sequence per time frame, causing effective batch size to scale linearly with signal length and making long-recording training prohibitively expensive (see Supplementary Material C, Figure S2).

![](images/3fd25f3350204e0fa2ddebe5c4b2d9a3a51c581a9b1a1c1c35e994170089ac7f.jpg)  
Fig. 2. Architecture of UNet-Mamba1-3B. Three Mamba blocks, each preceded by layer normalization, go into the convolutional bottleneck to capture temporal dependencies beyond the ∼1.5 s receptive field of the encoder. The Mamba layer follows [25].

## B. Training

All models are trained with MSE loss (batch size 32), normalized per lead using the median and interquartile range, with early stopping on validation loss. Details on training for baseline models are in Supplementary Material D (Table S2); convergence is confirmed in Supplementary Material F (Figure S3).

For the 12-lead PTB-XL dataset, we deploy UNet-Mamba in two modes: a single lead-agnostic model trained across all leads, and 12 separate lead-specific (or LS, for short) instances, one per lead. The lead-agnostic variant enables direct comparison with prior work, which is overwhelmingly lead-agnostic by design [16], [17], [18]. The single lead is selected by retaining the channel with the fewest detected peaks, as leads with fewer peaks are less likely to be corrupted by noise [51]. The lead-specific variant allows each instance to specialize to the distinct morphological characteristics of its respective lead—e.g. the dominant R-wave often in $\mathrm { { V } _ { 5 } }$ versus the predominantly negative QRS complex in aVR [52], [53].

## IV. COHORT

## A. Clinical Cohorts

PTB-XL [54] is a large publicly available dataset of 21,799 clinical 12-lead ECGs from 18,869 patients, each 10 s long at a native sampling frequency of 500 Hz. Records flagged with baseline drift, static noise, burst noise, or electrode problems are excluded following [51], yielding 13,480 (from the original 17,418) training and 1,639 (from 2,183) test segments.

Each record carries a subset of 44 diagnostic statements provided by up to two cardiologists, which can be aggregated into five superdiagnostic classes: NORM (normal), MI (myocardial infarction), STTC (ST/T change), CD (conduction disturbance), and HYP (hypertrophy). We leverage these annotations for downstream evaluation of diagnostic classification. Folds 1–8 are used for training, fold 9 for validation, and fold 10 for testing. Further cohort demographics and acquisition details are provided in [54] and Supplementary Material G.

European ST-T Database [55] consists of 90 annotated excerpts of two-hour ambulatory ECG recordings from 79 subjects in whom myocardial ischemia was diagnosed or suspected. Each record contains two channels sampled at 250 Hz. The 90 records are split into 54 training, 18 test, and 18 evaluation records. Within each record, segments are extracted at evenly spaced positions using non-overlapping windows: 2,048 segments for training and 256 segments for testing and evaluation.

## B. Synthetic Cohort

We generate single-channel synthetic ECGs segments using the dynamical ODE model of [42]. Heart rate is drawn uniformly from [60, 80] bpm for each segment, and the PQRST morphology parameters (amplitudes $a _ { i }$ and widths $b _ { i } )$ are sampled uniformly around reference values (Supplementary Material H) to introduce inter-segment variability. Each segment is simulated for 45 s at 360 Hz; the first 5 s are discarded to eliminate transient warm-up artifacts, yielding 40 s effective signals. We generate 1,024 training, 256 test and 256 validation segments. Deterministic random seeds per split ensure reproducibility and non-overlapping populations.

## C. Data Extraction and Preprocessing

All clinical recordings are resampled to 360 Hz via polyphase resampling to match the frequency expected by the denoising models, then bandpass-filtered (1–45 Hz), retaining the clinically relevant ECG frequency content—the bulk of QRS spectral energy lies below 40 Hz [56]—while attenuating baseline wander below 1 Hz and suppressing powerline interference and high-frequency noise above 45 Hz. Signals are then robustly normalized using the median and interquartile range computed on the training set.

## D. Noise Model

a) Noise source.: Realistic noise signals are sourced from the MIT-BIH Noise Stress Test Database (NSTDB) [57], which provides three half-hour, two-channel recordings of noise typical in ambulatory ECG settings: baseline wander (BW), muscle/EMG artifact (MA), and electrode motion artifact (EM). The first channel of each record is split as: 0–15 mins for training, 15–22.5 for testing, and 22.5–30 for evaluation.

b) Online noise application.: During training, all four noise types (BW, MA, EM, and additive white Gaussian noise) are applied simultaneously to each clean ECG segment. For each structured noise type, a random excerpt of the required length is extracted from the corresponding noise bank and scaled to a prescribed SNR. AWGN is generated independently and scaled likewise. Following the ranges recommended by [58], we set SNR levels to 2.5 dB for BW, 7.5 dB for MA, 12.5 dB for EM, and 22.5 dB for AWGN. The simultaneous application of all four noise sources ensures that the denoiser is trained under realistic multi-source contamination rather than idealized single-noise conditions. The noise pipeline is validated by comparing empirical SNR distributions against theoretical expectations (Supplementary Material I, Table S4, Figure S4).

## V. RESULTS

## A. Experiments and Evaluation

a) Evaluation Metrics.: We quantify reconstruction fidelity using SNR as $\begin{array} { r } { \mathrm { S N R } ( \hat { \mathbf { x } } , \mathbf { x } ) = \hat { 1 0 } \log _ { 1 0 } \biggl ( \frac { \vert \vert \mathbf { x } \vert \vert _ { 2 } ^ { 2 } } { \vert \vert \mathbf { x } - \hat { \mathbf { x } } \vert \vert _ { 2 } ^ { 2 } } \biggr ) } \end{array}$ , and RMSE as $\begin{array} { r l r } { \mathrm { R M S E } ( \hat { \bf x } , { \bf x } ) } & { { } = } & { \sqrt { \frac { 1 } { T } \left\| \hat { \bf x } - { \bf x } \right\| _ { 2 } ^ { 2 } } } \end{array}$ , where x is the clean signal, xˆ the denoised estimate, and T the number of time steps. Each metric is computed per sequence and averaged across all test sequences. For downstream classification we report macro-averaged AUROC (ranking-based), BCE and Brier score (calibration-sensitive, after temperature scaling [59]), and macro sensitivity, specificity, and F1 (threshold 0.5). For each record and class, we compute a per-entry score (the BCE term, or the squared error for Brier) on temperature-scaled probabilities, then take the flat mean over all entries. These are per-element averages, not summed or prevalence-weighted (micro-averaging).

All metrics use 95% bootstrap confidence intervals (1,000 resamples, percentile method) following [60].

b) Downstream Evaluation Pipeline.: For reconstruction (SNR/RMSE), one lead per record is selected (fewest detected peaks [51]) and the denoiser is trained and evaluated on these single-lead signals. The lead-agnostic model pools all leads and is lead-blind, while the lead-specific variant trains 12 separate lead-aware instances. To assess clinical utility, clean 12-lead PTB-XL ECGs are separated into individual leads, independently normalized and corrupted with noise, denoised by each model under test, de-normalized, and reassembled into 12-lead signals. Two independent classifiers— Inception1D [61] and ResNet1D-Wang [62] (Supplementary Material J)—trained exclusively on clean data then classify the reconstructed signals across 44 diagnostic classes. We reuse the architectures, hyper-parameters, and training scripts from the PTB-XL benchmark of [60] (https://github.com/ helme/ecg\_ptbxl\_benchmarking) without modification, including the standard fold split and multi-label BCEwith-logits objective. We add temperature scaling ([59]): a single global scalar T is fit by minimizing BCE on the cleancondition logits and labels of the validation fold and reused for noisy and denoised conditions. This pipeline is illustrated in Figure 1.

c) Inference Cost.: On a single 10 s ECG window at 360 Hz (batch=1, NVIDIA RTX A5000, PyTorch $2 . 9 \ +$ CUDA 12.8, mean of 50 forwards after 10 warmups), DR-net Mamba1-3B runs in 11.9 ms with 15.6 MB peak GPU memory, faster than DR-net-IMUNet (16.0 ms) at comparable memory (Table V). All evaluated models process 10 s of signal in under 50 ms, comfortably real-time for clinical deployment.

## B. ECG Reconstruction

Figures 3 and S8 report SNR and RMSE, respectively, across all three datasets under the medium noise setting (combined input $\mathrm { S N R } \approx 0 . 9 6 \mathrm { d B } ;$ see Supplementary Material I). DR-net-Mamba achieves the highest SNR and lowest RMSE on all three datasets. On the synthetic benchmark (Figure 3a), UNet-Mamba and DR-net-Mamba obtain approximately 2.5– 3 dB higher SNR than their base models (UNet and DR-net-UNet), while DRNN, DAE, and MECGE consistently yield the weakest performance.

Performance rankings on PTB-XL (Figure 3b) and European ST-T (Figure 3c) are largely consistent with the synthetic benchmark. On clinical data, the performance gap between Stage 1 and Stage 2 models is more pronounced. The relationship between training data volume and reconstruction performance is explored in Supplementary Material L.

To confirm the reported improvements are not artifacts of overlapping per-model CIs, we ran paired bootstrap tests of the mean difference $\Delta = \bar { x } _ { A } - \bar { x } _ { B }$ between every baseline and our models for each of the three datasets (for MECGE, the 1800-sample windows were stitched back into the 14400- sample recordings before computing the metric). For each pair, recording indices were resampled with replacement $B = 1 0 0 0$ times, using the same indices for both models per iteration to preserve pairing, and 95% percentile confidence intervals of $\Delta$ are reported. A CI excluding 0 indicates a significant difference at $\alpha = 0 . 0 5$

We find that every comparison of our models against their respective baseline yields a significant difference (at $\alpha =$ 0.05) on each of the three datasets: For the synthetic test set (n = 256 source recordings), the smallest gain is +0.29 dB SNR over DR-net-IMUNet with a CI of [+0.15, +0.44]. On PTB-XL (n = 1,639 recordings, 10 s each at 360 Hz), DR-net Mamba1-3B outperforms every baseline (smallest gain +0.83 dB SNR over DR-net-IMUNet, CI $[ + 0 . 7 8 , + 0 . 8 8 ] )$ , while on the European ST-T database (n = 256 recordings, 40 s) the smallest gain of +0.34 dB SNR, $\operatorname { C I } \left[ + 0 . 2 0 , + 0 . 4 9 \right]$ is achieved over DR-net-IMUNet. The individual results are given in Tables S7, S8 and S9 in Section N in the Supplementary Material.

The compression layer is a potential confounder. We have therefore run ablation studies to separate the effects of

![](images/e3c3b2001c93827175e4aa1bceea02a86435b8655328a2ad93b24d3ade5f1910.jpg)

![](images/330d02ba785d3b125c1e3cb7ed0653e14c2120e0b36be54f11bf8d91361a11e1.jpg)

![](images/555ae27127fe04d46ba58c89403dd6f98bee28d9017b79335b29106b90fdec5e.jpg)  
Fig. 3. Reconstruction SNR (dB) on synthetic (a), PTB-XL (b), and European ST-T (c) datasets. Empirical noisy SNR = 0.96 dB (medium noise setting, see Supplementary Material I). Analogous RMSE results are reported in Figure S8 in Supplementary Material M.

Mamba, DR-net, and log compression. We isolated the three components by evaluating a 6-cell grid on European ST-T (UNet baseline, with/without DR-net, with/without Mamba, with/without learnable compression) on the same $N = 2 5 6$ test recordings. All pairwise SNR differences are computed with a paired bootstrap: recording indices are resampled with replacement $B ~ = ~ 1 0 0 0$ times using the same indices for both models per iteration, yielding the 95% CI of $\Delta \ =$ ${ \bar { x } } _ { A } - { \bar { x } } _ { B }$ . Mamba and DR-net each contribute large, significant gains; learnable compression adds +0.9 dB SNR on top of single-stage Mamba (CI [+0.7, +1.0]) but its effect vanishes once DR-net is added (-0.0 dB, CI [-0.1, +0.1]), indicating that compression and the two-stage DR-net design capture overlapping benefits. The detailed numbers are tabulated in Table S10 in the Supplementary Material.

## C. Noise Robustness

Figure 4 reports reconstruction SNR as a function of input SNR for each noise type individually and for the combined condition on the European ST-T dataset, restricted to the bestperforming models for clarity. Grey-shaded regions indicate input SNR values outside the plausible clinical operating ranges as reported by [58]. DR-net-Mamba achieves the highest output SNR across all noise types and clinically relevant input SNR levels. The advantage over purely convolutional models is most pronounced under heavy AWGN. MECGE shows notable robustness under heavy muscle artifact and AWGN. Under the combined noise condition, both DR-net-Mamba and MECGE degrade more gracefully than DR-net-UNet and DR-net-IMUNet.

## D. Impact of Recording Length

Figure 5 compares UNet and UNet-Mamba (without log compression, to isolate the bottleneck contribution) as a function of recording length using a curriculum-based protocol. Models are trained in four stages of increasing sequence length $( 5 \mathrm { s } \to 1 0 \mathrm { s } \to 2 0 \mathrm { s } \to 4 0 \mathrm { s } )$ , with the original 40 s long records being cut into shorter windows of 7,200, 3,600, and 1,800 samples (20 s, 10 s, and 5 s), respectively, alongside the full 40 s. Each model is thus trained on the same total recording length, but a varying maximal length. In the training, each model is initialized using the weights from the previous stage, with compression disabled so the two architectures differ only by the presence of the Mamba bottleneck. On the synthetic dataset (Figure 5a), the two models achieve comparable SNR at 5 s, but UNet-Mamba consistently outperforms UNet for longer recordings, with the gain growing from approximately 1 dB at 10 s to roughly 2.5 dB at 40 s. On the European ST-T dataset (Figure 5b), UNet-Mamba already outperforms UNet by approximately 1 dB at 5 s. The improvement peaks around 10 s and then diminishes for longer sequences, in contrast to the monotonic growth observed on synthetic data. Training details for each curriculum stage are reported in Supplementary Material K (Tables S5–S6).

## E. Clinical Translation: Downstream Disease Prediction

1) Aggregated Evaluation: Figure 6 summarizes downstream classification performance on PTB-XL under the strong noise setting (BW 0dB; MA 5dB; EM 10dB; AWGN 20dB) using the ResNet1D-Wang backbone–see Figure S10 in Supplementary Material R for the Inception1D backbone. Results under a lower noise setting are reported in Supplementary Material P. Noise reduces AUROC by 0.069 for Inception1D and 0.055 for ResNet1D-Wang. The Mamba-based models achieve the best overall AUROC across both classifiers and consistently improve over their base models. The lead-specific UNet-Mamba (Section III-B) further improves AUROC on ResNet1D-Wang.

The calibration picture is more nuanced and backbonedependent. Among the denoisers, the Mamba-based models attain the best BCE and Brier scores on both classifiers, but whether denoising improves calibration over the noisy input depends on the classifier. On Inception1D, denoising broadly helps: every model that improves AUROC also lowers BCE and Brier relative to the noisy baseline, with the lead-specific UNet-Mamba best on both (BCE 6.89 and Brier 1.82 vs. noisy 7.85/2.02). On ResNet1D-Wang the effect is weaker: several denoisers—including some Mamba variants—sit at or slightly above the noisy baseline on Brier even when they improve BCE, and only the lead-specific UNet-Mamba clearly improves both BCE and Brier over the noisy input (BCE 7.10 and Brier 1.89 vs. noisy 7.71/2.01). Thus the leadspecific variant is the only denoiser to improve calibration over the noisy baseline on both classifiers, while other models trade improved discrimination against degraded or unchanged calibration. We return to this discrimination–calibration gap in Section VI-B.

![](images/2b4e5fe67b033f4c94b71a14a19cf8834efd617d6dd957b47321318298d4e5a7.jpg)  
Fig. 4. Reconstruction SNR (dB) as a function of input SNR for each noise type (EM, BW, MA, AWGN) and the combined condition on European ST-T. Grey-shaded areas denote input SNR values outside clinically plausible ranges [58]. Shaded bands indicate 95% bootstrap CIs.

To understand the clinical relevance of these results, Table II reports macro-averaged metrics for Inception1D (strong noise, bold/underlined: best/second best). The noisy signal achieves higher sensitivity than the clean baseline (0.852 vs. 0.815) at the cost of substantially lower specificity (0.697 vs. 0.863). The three Mamba variants rank first through third on F1, combining specificity of 0.839–0.841 with sensitivity of 0.789– 0.808. Similar trends hold for ResNet1D-Wang (Supplementary Material Q, Table S14).

2) Evaluation by Superdiagnostic Class: Table II additionally breaks down AUROC by superdiagnostic class. The Mamba-based models achieve the highest AUROC across all five classes. The lead-specific UNet-Mamba leads in HYP, MI, NORM, and STTC, while DR-net-Mamba scores second-best in CD, HYP, MI, and STTC. Results under a lower noise setting (Tables S12–S13) and per-class sensitivity, specificity, and F1 (Tables S15–S16) are reported in Appendices P and S.

3) Evaluation by Diagnostic Class: Figure 7a isolates the effect of adding a Mamba bottleneck to the UNet while keeping all other design choices fixed. UNet-Mamba consistently outperforms UNet on STTC (ST/T change) and its subclasses, whereas UNet achieves comparable or slightly better AUROC on CD (conduction disturbance) and most of its subclasses.

Figure 7b shows the absolute AUROC of the lead-specific

UNet-Mamba per diagnostic class, with color encoding improvement over the noisy baseline. For all diagnostic classes with more than 18 segments, denoising either improves or preserves classification performance. The corresponding breakdowns for ResNet1D-Wang are reported in Supplementary Material T (Figure S11) and show similar trends.

A natural alternative to the denoise-plus-clean-classifier pipeline is to train the classifier directly on noise-corrupted ECGs. As shown in Table I (strong noise, SNR ≈ −1.5 dB) , this noise-aware classifier provides a strong baseline, achieving an AUROC of 0.923 compared with 0.909 for the denoiseplus-clean-classifier pipeline, which therefore does not outperform direct noise-aware training in downstream discrimination. This comparison, however, favors the noise-aware classifier in an important respect: it is trained and evaluated using the same noise family and noise setting, such that the noisy test data remain in-distribution. Moreover, this strategy requires the classifier to be retrained for each noise condition and each downstream classification task. In contrast, our approach decouples signal restoration from downstream prediction: the denoiser is trained once for the target noise setting and can subsequently be reused with different off-the-shelf classifiers trained on clean ECGs and for different downstream tasks. Importantly, the denoiser also reconstructs the underlying ECG signal itself, rather than learning only a noise-robust decision boundary for a specific set of labels.

a. Record-Level SNR (Synthetic Dataset)  
![](images/cba8b5f5b00e016a96097ec602fc67e4c3038c3ede7c61ee4abf6e52a1b9f0c4.jpg)  
b. Record-Level SNR (European ST-T)

![](images/fcbb57aa7cdaaeadc5a5cec761f3a9b00473f1930cf80e34f68af299e0d34d5d.jpg)  
Fig. 5. Reconstruction SNR (dB) as a function of recording length for UNet and UNet-Mamba (without signal compression), on synthetic (a) and European ST-T (b). Shaded bands indicate 95% bootstrap CIs.

TABLE I  
DOWNSTREAM CLASSIFIER AUROC UNDER STRONG NOISE.
<table><tr><td>Classifier trained on</td><td>Test input</td><td>AUC</td><td></td></tr><tr><td>Clean</td><td>Clean</td><td></td><td>0.925 [0.916, 0.932]</td></tr><tr><td>Clean</td><td>Noisy</td><td></td><td>0.879 [0.865, 0.892]</td></tr><tr><td>Noisy</td><td>Noisy</td><td></td><td>0.923 [0.913, 0.930]</td></tr><tr><td>Clean</td><td>Denoised (Mamba1-3B Lead Aware)</td><td></td><td>0.909 [0.901, 0.918]</td></tr></table>

## VI. DISCUSSION

The central finding of this work is that augmenting a convolutional encoder–decoder with a selective state-space bottleneck consistently improves reconstruction fidelity and noise robustness, and improves downstream diagnostic discrimination (AUROC), while its effect on calibration (BCE, Brier) is smaller and classifier-dependent—and that the nature of these effects is informative about when and why long-range temporal context matters for physiological signal processing.

## A. Long-Range Context as an Inductive Bias for Physiological Denoising

The recording-length experiment (Figure 5) provides a clear direct evidence of the above. On synthetic data, where every beat shares the same parametric PQRST morphology [42], the Mamba advantage grows monotonically with sequence length because the state of distant beats is genuinely predictive of the current one. The UNet’s convolutional receptive field saturates at ${ \sim } 1 . 5 \mathrm { s } ,$ so additional temporal context cannot be exploited by the bottleneck convolutions, whereas the Mamba blocks propagate information across the full downsampled sequence. On clinical data, the Mamba advantage peaks around 10 s and then diminishes, reflecting the lower temporal regularity of real ECGs: inter-beat variability, heart rate variability, and pathological waveform heterogeneity make distant beats less predictive than in the stationary synthetic case. This contrast offers a practical guideline: selective state-space bottlenecks are most beneficial when the signal exhibits structured temporal regularity beyond the convolutional receptive field, a condition that holds for many physiological recordings (e.g., respiratory signals, EEG rhythms, continuous glucose monitoring) but whose degree is application-specific.

The noise robustness results (Figure 4) reinforce this interpretation. The Mamba advantage over purely convolutional models is most pronounced under heavy AWGN. Structured noise sources (BW, MA, EM) carry recognizable temporal patterns that convolutional models can learn to suppress within their receptive field [63]. AWGN is sample-wise independent and offers no such local cues; as its intensity grows, reconstruction must rely on temporal predictability across beats—precisely the form of context the Mamba bottleneck provides. Under the combined noise condition, both DR-net-Mamba and MECGE degrade more gracefully than purely convolutional models, suggesting that global context—whether temporal (Mamba) or spectral (MECGE; [28])—is beneficial when multiple noise sources interact simultaneously.

## B. Clinical Translation: When Long-Range Context Helps Most

The downstream evaluation confirms that Mamba-based denoising improves diagnostic performance across both classifiers and nearly all pathology classes. Crucially, the perclass analysis (Figure 7a, Table S18) reveals that this benefit is morphology-dependent, offering actionable insight into which clinical scenarios gain most from long-range temporal modeling. The Mamba bottleneck consistently and substantially improves AUROC on STTC diagnoses—slow, broad waveforms (200–400 ms) whose interpretation depends on the surrounding baseline [64] and that are easily confounded with structured noise such as baseline wander. For conduction disturbances, diagnosed from QRS-internal intervals such as the R-wave peak time (45–60 ms) [65], performance matches the convolutional baseline: with 20× downsampling these span barely one bottleneck time step, leaving little room for improvement. Table S18 summarizes this resolution-versuscontext trade-off. Importantly, for all diagnostic classes with more than 18 segments, the lead-specific Mamba model either improves or preserves AUROC relative to the noisy baseline (Figure 7b), confirming that denoising does not harm downstream classification at the individual-diagnosis level.

The calibration results further complicate the reconstruction-to-utility mapping, and they differ sharply between the two classifiers. On Inception1D, denoising generally improves calibration: models that raise AUROC also lower BCE and Brier relative to the noisy baseline.

![](images/732c9a09bcf0b7625c6edb22f246df2923da98f2a12f8aaaa1edf96d944ebfec.jpg)  
Fig. 6. Downstream diagnostic classification (macro and superdiagnostic AUROC) on PTB-XL using ResNet1D-Wang, comparing denoised signals against clean and noisy baselines (strong noise setting).

TABLE II  
PER-SUPERDIAGNOSTIC-CLASS AUROC AND OVERALL MACRO-AVERAGED SENSITIVITY, SPECIFICITY, AND F1 ON PTB-XL
<table><tr><td></td><td>CD</td><td>HYP</td><td></td><td>MI</td><td>NORM</td><td></td><td>STTC</td><td></td><td></td><td>Overall</td><td></td><td></td><td></td></tr><tr><td></td><td colspan="7">AUROC</td><td colspan="2">Sensitivity</td><td colspan="2">Specificity</td><td colspan="2">F1</td></tr><tr><td>UNet-Mamba1-3B</td><td> $\mathbf { 9 2 5 } \pm \mathbf { . 0 1 }$ </td><td> $. 8 8 3 \pm . 0 2$ </td><td></td><td> $. 9 0 9 \pm . 0 2$ </td><td></td><td> $. 9 2 6 \pm . 0 1$ </td><td> $. 8 8 0 \pm . 0 2$ </td><td></td><td> $. 7 8 9 \pm . 0 2$ </td><td></td><td> $. 8 4 0 \pm . 0 1$ </td><td></td><td> $. 7 0 3 \pm . 0 2$ </td></tr><tr><td>DR-net-Mamba1-3B</td><td> $. 9 2 1 \pm . 0 1$ </td><td> $. 8 8 7 \pm . 0 2$ </td><td></td><td> $. 9 0 9 \pm . 0 2$ </td><td></td><td> $. 9 2 6 \pm . 0 1$ </td><td> $. 8 8 1 \pm . 0 2$ </td><td></td><td> $. 7 9 4 \pm . 0 2$ </td><td></td><td> $\overline { { { \bf 8 4 1 } \pm { \bf . 0 1 } } }$ </td><td></td><td> $. 7 0 6 \pm . 0 2$ </td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td> $\overline { { 9 2 0 \pm . 0 1 } }$ </td><td> $\overline { { { . 8 8 7 } \pm { . 0 2 } } }$ </td><td></td><td> $\mathbf { 9 1 0 } \pm \mathbf { \delta . 0 1 }$ </td><td> $\mathbf { 9 2 8 } \pm \mathbf { . 0 1 }$ </td><td></td><td> $\overline { { { . 8 8 7 \pm . 0 2 } } }$ </td><td></td><td> $\overline { { { \bf 8 0 8 } \pm { \bf . 0 2 } } }$ </td><td></td><td> $. 8 3 9 \pm . 0 1$ </td><td></td><td> $\overline { { . 7 1 5 \pm . 0 2 } }$ </td></tr><tr><td>DRNN</td><td> $. 9 0 0 \pm . 0 2$ </td><td> $. 8 8 5 \pm . 0 2$ </td><td></td><td> $. 8 9 5 \pm . 0 2$ </td><td></td><td> $. 9 1 0 \pm . 0 2$ </td><td> $. 8 2 4 \pm . 0 2$ </td><td></td><td> $. 7 8 3 \pm . 0 2$ </td><td></td><td> $. 7 8 0 \pm . 0 1$ </td><td></td><td> $. 6 5 1 \pm . 0 2$ </td></tr><tr><td>DAE</td><td> $. 8 6 0 \pm . 0 2$ </td><td> $. 8 5 4 \pm . 0 3$ </td><td></td><td> $. 8 7 4 \pm . 0 1$ </td><td> $. 9 0 7 \pm . 0 1$ </td><td></td><td> $. 8 4 8 \pm . 0 2$ </td><td></td><td> $. 7 0 9 \pm . 0 2$ </td><td></td><td> $. 8 1 9 \pm . 0 1$ </td><td></td><td> $. 6 3 8 \pm . 0 2$ </td></tr><tr><td>UNet</td><td> $. 9 1 4 \pm . 0 2$ </td><td> $. 8 6 4 \pm . 0 3$ </td><td></td><td> $. 9 0 4 \pm . 0 1$ </td><td> $. 9 2 1 \pm . 0 1$ </td><td></td><td> $. 8 6 5 \pm . 0 2$ </td><td></td><td> $. 7 6 4 \pm . 0 1$ </td><td></td><td> $. 8 2 2 \pm . 0 1$ </td><td></td><td> $. 6 7 2 \pm . 0 2$ </td></tr><tr><td>DR-net-UNet</td><td> $. 9 1 4 \pm . 0 1$ </td><td> $. 8 7 5 \pm . 0 3$ </td><td></td><td> $. 9 0 4 \pm . 0 1$ </td><td> $. 9 2 1 \pm . 0 1$ </td><td></td><td> $. 8 6 6 \pm . 0 2$ </td><td></td><td> $. 7 8 4 \pm . 0 2$ </td><td></td><td> $. 8 2 8 \pm . 0 1$ </td><td></td><td> $. 6 9 0 \pm . 0 1$ </td></tr><tr><td>IMUNet</td><td> $. 9 1 4 \pm . 0 1$ </td><td> $. 8 7 5 \pm . 0 2$ </td><td></td><td> $. 9 0 1 \pm . 0 1$ </td><td></td><td> $. 9 2 5 \pm . 0 1$ </td><td> $. 8 7 9 \pm . 0 2$ </td><td></td><td> $. 7 6 7 \pm . 0 2$ </td><td></td><td> $. 8 4 0 \pm . 0 1$ </td><td></td><td> $. 6 9 1 \pm . 0 2$ </td></tr><tr><td>DR-net-IMUNet</td><td> $. 9 1 2 \pm . 0 1$ </td><td> $. 8 8 3 \pm . 0 3$ </td><td></td><td> $. 9 0 0 \pm . 0 2$ </td><td> $. 9 2 7 \pm . 0 1$ </td><td></td><td> $. 8 7 9 \pm . 0 2$ </td><td></td><td> $. 7 8 5 \pm . 0 2$ </td><td></td><td> $\overline { { { \bf 8 4 1 } \pm { \bf . 0 1 } } }$ </td><td></td><td> $. 6 9 9 \pm . 0 2$ </td></tr><tr><td>MECGE</td><td> $. 9 0 5 \pm . 0 1$ </td><td> $. 8 7 0 \pm . 0 3$ </td><td></td><td> $. 8 8 7 \pm . 0 2$ </td><td> $. 9 1 7 \pm . 0 1$ </td><td></td><td> $. 8 7 2 \pm . 0 2$ </td><td></td><td> $. 7 9 0 \pm . 0 2$ </td><td></td><td> $. 8 0 8 \pm . 0 1$ </td><td></td><td> $. 6 8 0 \pm . 0 2$ </td></tr><tr><td>clean</td><td> $. 9 3 2 \pm . 0 1$ </td><td> $. 9 0 6 \pm . 0 2$ </td><td></td><td> $. 9 2 5 \pm . 0 2$ </td><td></td><td> $. 9 3 8 \pm . 0 1$ </td><td> $. 9 1 4 \pm . 0 2$ </td><td></td><td> $. 8 1 5 \pm . 0 2$ </td><td></td><td> $. 8 6 3 \pm . 0 1$ </td><td></td><td> $. 7 3 8 \pm . 0 1$ </td></tr><tr><td>noisy</td><td> $. 9 0 3 \pm . 0 2$ </td><td> $. 8 7 2 \pm . 0 3$ </td><td></td><td> $. 8 7 5 \pm . 0 2$ </td><td></td><td> $. 9 1 1 \pm . 0 2$ </td><td> $. 8 6 3 \pm . 0 2$ </td><td></td><td> $. 8 5 2 \pm . 0 2$ </td><td></td><td> $. 6 9 7 \pm . 0 1$ </td><td></td><td> $. 6 3 2 \pm . 0 2$ </td></tr></table>

## C. The Sensitivity–Specificity Shift Induced by Noise

On ResNet1D-Wang, however, this coupling breaks down— several denoisers improve AUROC yet sit at or above the noisy baseline on Brier, and some worsen both BCE and Brier. A plausible explanation is that on ResNet1D-Wang denoising helps the classifier on borderline cases (improving rankings and thus AUROC) but reduces confidence on non-borderline segments, where the denoised signal may differ subtly from what the classifier was trained on; because BCE and Brier directly penalize reduced confidence on every segment, this loss can outweigh the gains on the borderline subset. Across both classifiers, the lead-specific Mamba variant is the only denoiser that improves both BCE and Brier over the noisy baseline, likely because per-lead specialization yields denoised signals closer to the clean morphology on which the classifiers were trained. This contrast highlights the importance of evaluating denoising not only through discrimination metrics (AUROC) but also through calibration metrics (BCE, Brier)—and of doing so per classifier—when the downstream application involves clinical decision-making with probabilistic outputs.

The observation that noisy signals yield higher sensitivity than clean baselines (0.852 vs. 0.815 for Inception1D) while substantially degrading specificity (0.697 vs. 0.863) reveals a clinically important operating-point shift. Noise-corrupted morphology makes many negative examples resemble positive ones, pushing the classifier toward a higher false-positive rate. Denoising partially reverses this shift: all models reduce sensitivity relative to the noisy input while recovering specificity. Mamba models achieve the best balance, with the highest F1 scores and specificity closest to the clean baseline. For clinical deployment, the specificity recovery offered by denoising may be more valuable than the raw AUROC improvement, particularly in screening settings where false positives trigger costly follow-up.

## D. DR-net-Mamba in the Broader Denoising Landscape

Prior ECG denoising studies [16], [17], [18] each use different clean-signal sources, noise protocols, segment lengths, and metric definitions, making cross-study comparison of absolute values unreliable. Our noise pipeline differs fundamentally: each source is scaled to per-type SNR targets drawn from clinically calibrated ranges [58], and all four types are applied simultaneously, yielding evaluation conditions more challenging than those in the baseline studies. Rather than comparing absolute metric values across incompatible setups, we focus on within-study rankings using a unified noise protocol applied consistently to all models. The absence of a standardized ECG denoising benchmark remains an open problem for the field.

![](images/ee0f802ed7b2eb5d1aad1e63724e15e8c82d0c385ced054e47120f6d263d45db.jpg)  
Fig. 7. Per-diagnostic-class downstream AUROC (Inception1D). (a) AUROC delta when adding a Mamba bottleneck to the UNet, all other design choices held constant. (b) Absolute AUROC of the lead-specific UNet-Mamba; color indicates improvement over the noisy baseline. In both panels, diagnostic classes with fewer than five segments are omitted. See Supplementary Material U for abbreviations.

## Further Machine Learning in Healthcare Insights

This work offers three insights that extend beyond ECG denoising. First, we demonstrate that inserting selective statespace blocks at the bottleneck of a convolutional encoder– decoder is a parameter-efficient strategy for injecting longrange temporal context into physiological signal models, with the benefit growing with recording length. Second, we show that reconstruction gains do not automatically transfer to downstream clinical utility: the relationship between denoising quality and diagnostic accuracy is morphology-dependent, with context-sensitive waveforms (e.g., ST/T changes) benefiting most from long-range modeling while sharp features (e.g., QRS complexes) can be harmed by bottleneck over-smoothing. Third, we find that lead-specific denoising—training separate models per ECG lead—can preserve calibration where leadagnostic models degrade it, highlighting that evaluation beyond discrimination is essential for clinical deployment.

## E. Limitations

Various limitations arise in this work: (i) Ground-truth quality. Neither PTB-XL nor European ST-T provides truly clean references. Residual artifacts survive preprocessing and become part of the reconstruction target through the MSE loss. Reconstruction metrics therefore partially measure reproduction of these artifacts rather than genuine signal recovery, and Stage 2 improvements on clinical data should be interpreted with this caveat. (ii) Single-lead pipeline. Each lead is denoised independently and reassembled for downstream classification. Inter-lead spatial correlations in the 12-lead ECG are not exploited. A multi-lead architecture could improve both reconstruction and diagnostic performance, and would avoid the 12× training cost of the lead-specific variant. (iii) Noise model scope. All models are trained and evaluated with four noise types at prescribed SNR levels. Other clinically relevant artifact types (e.g., lead misplacement, pacemaker spikes, motion artifacts in wearable devices) and adaptive or non-stationary noise levels are not covered. (iv) Downstream evaluation scope. The downstream pipeline uses classifiers trained on clean data and evaluated on denoised data. In practice, classifiers may be trained on noisy data or jointly optimized with the denoiser. Our pipeline isolates the effect of denoising but does not capture potential benefits from end-to-end training. (v) Recording-length scaling on clinical data. The Mamba advantage peaks around 10 s on clinical recordings and then diminishes, unlike the monotonic growth on synthetic data. Whether this reflects intrinsic limits of longrange context in irregular clinical signals or artifacts of the training protocol (e.g., boundary effects [66]) warrants further investigation.

## CONFLICTS OF INTEREST

S. R.-C. reports equity, consulting, and intellectual property with Physcade Inc. Disclosures are unrelated to this work. All other authors have nothing to disclose.

## REFERENCES

[1] B. Chong et al., “Global burden of cardiovascular diseases: Projections from 2025 to 2050,” European journal of preventive cardiology, vol. 32, no. 11, pp. 1001–1015, 2025.

[2] E. J. Rowin et al., “Extended ambulatory ecg monitoring enhances identification of higher-risk ventricular tachyarrhythmias in patients with hypertrophic cardiomyopathy,” Heart Rhythm, vol. 22, no. 7, pp. 1696–1704, 2025.

[3] Y. Tian et al., “Foundation model of ecg diagnosis: Diagnostics and explanations of any form and rhythm on ecg,” Cell Reports Medicine, vol. 5, no. 12, 2024.

[4] J. Li et al., “An electrocardiogram foundation model built on over 10 million recordings,” NEJM AI, vol. 2, no. 7, 2025.

[5] T. J. Poterucha et al., “Detecting structural heart disease from electrocardiograms using ai,” Nature, vol. 644, no. 8075, pp. 221–230, 2025.

[6] A. Vaid et al., “Using deep-learning algorithms to simultaneously identify right and left ventricular dysfunction from the electrocardiogram,” Cardiovascular Imaging, vol. 15, no. 3, pp. 395–410, 2022.

[7] J. Mayourian et al., “Pediatric ecg-based deep learning to predict left ventricular dysfunction and remodeling,” Circulation, vol. 149, no. 12, pp. 917–931, 2024.

[8] J. Mayourian et al., “Electrocardiogram-based deep learning to predict left ventricular systolic dysfunction in paediatric and adult congenital heart disease in the usa: A multicentre modelling study,” The Lancet Digital Health, vol. 7, no. 4, e264–e274, 2025.

[9] C.-C. Yu et al., “Ecg-based machine learning model for af identification in patients with first ischemic stroke,” International Journal of Stroke, vol. 20, no. 4, pp. 411–418, 2025.

[10] V. Sangha et al., “Identification of hypertrophic cardiomyopathy on electrocardiographic images with deep learning,” Nature Cardiovascular Research, vol. 4, no. 8, pp. 991–1000, 2025.

[11] M. Z. Kolk et al., “Dynamic prediction of malignant ventricular arrhythmias using neural networks in patients with an implantable cardioverter-defibrillator,” EBioMedicine, vol. 99, 2024.

[12] S. Ruipérez-Campillo, M. Reiss, E. Ramírez, A. Cebrián, J. Millet, and F. Castells, “Clustering and machine learning framework for medical time series classification,” Biocybernetics and Biomedical Engineering, vol. 44, no. 3, pp. 521– 533, 2024.

[13] M. Z. Kolk et al., “Optimizing patient selection for primary prevention implantable cardioverter-defibrillator implantation: Utilizing multimodal machine learning to assess risk of implantable cardioverter-defibrillator non-benefit,” Europace, vol. 25, no. 9, euad271, 2023.

[14] S. Chatterjee, R. S. Thakur, R. N. Yadav, L. Gupta, and D. K. Raghuvanshi, “Review of noise removal techniques in ecg signals,” IET Signal Processing, vol. 14, no. 9, pp. 569–590, 2020.

[15] Y. Jia et al., “Preprocessing and denoising techniques for electrocardiography and magnetocardiography: A review,” Bioengineering, vol. 11, no. 11, p. 1109, 2024.

[16] H.-T. Chiang, Y.-Y. Hsieh, S.-W. Fu, K.-H. Hung, Y. Tsao, and S.-Y. Chien, “Noise reduction in ecg signals using fully convolutional denoising autoencoders,” IEEE Access, vol. 7, pp. 60 806–60 813, 2019.

[17] L. Qiu, W. Cai, M. Zhang, W. Zhu, and L. Wang, “Two-stage ecg signal denoising based on deep convolutional network,” Physiological Measurement, vol. 42, no. 11, 2021.

[18] K. Antczak, Deep recurrent neural networks for ecg signal denoising, 2019. arXiv: 1807.11551 [cs.NE].

[19] X. Wang et al., “An ecg signal denoising method using conditional generative adversarial net,” IEEE Journal ofBiomedical and Health Informatics, vol. 26, no. 7, pp. 2929–2940, 2022.

[20] P. Singh and G. Pradhan, “A new ecg denoising framework using generative adversarial network,” IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 18, pp. 759–764, Mar. 2021.

[21] H. Li, G. Ditzler, J. Roveda, and A. Li, “Descod-ecg: Deep score-based diffusion model for ecg baseline wander and noise removal,” IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 9, pp. 5081–5091, Sep. 2024, ISSN: 2168-2208.

[22] A. Vaswani et al., “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[23] Y. Tay, M. Dehghani, D. Bahri, and D. Metzler, “Efficient transformers: A survey,” ACM Computing Surveys, vol. 55, no. 6, pp. 1–28, 2022.

[24] J. Song, C. Meng, and S. Ermon, Denoising diffusion implicit models, 2022. arXiv: 2010.02502 [cs.LG].

[25] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in First conference on language modeling, 2024.

[26] Y. Qiang, X. Dong, X. Liu, Y. Yang, F. Hu, and R. Wang, “Ecgmamba: Towards ecg classification with state space models,” in 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), IEEE, 2024, pp. 6498–6505.

[27] Y. Xu et al., Mambacapsule: Towards transparent cardiac disease diagnosis with electrocardiography using mamba capsule network, 2024. arXiv: 2407.20893 [eess.SP].

[28] K.-H. Hung et al., Mecg-e: Mamba-based ecg enhancer for baseline wander removal, 2024. arXiv: 2409 . 18828 [eess.SP].

[29] R. Vullings, B. De Vries, and J. W. Bergmans, “An adaptive kalman filter for ecg signal enhancement,” IEEE transactions on biomedical engineering, vol. 58, no. 4, pp. 1094–1103, 2010.

[30] M. Blanco-Velasco, B. Weng, and K. E. Barner, “Ecg signal denoising and baseline wander correction based on the

empirical mode decomposition,” Computers in biology and medicine, vol. 38, no. 1, 2008.

[31] J.-N. Lee and K.-C. Kwak, “ECG-based biometrics using a deep network based on independent component analysis,” IEEE Access, vol. 10, 2022.

[32] K. Yu et al., “Accurate wavelet thresholding method for ecg signals,” Computers in Biology and Medicine, vol. 169, p. 107 835, 2024.

[33] R. Holgado-Cuadrado, C. Plaza-Seco, L. Lovisolo, and M. Blanco-Velasco, “Characterization of noise in long-term ecg monitoring with machine learning based on clinical criteria,” Medical & Biological Engineering & Computing, vol. 61, no. 9, pp. 2227–2240, 2023.

[34] W. Yang, B. Cao, J. Wu, et al., “Ecgdedrdnet: A deep learning-based method for electrocardiogram noise removal using a double recurrent dense network,” arXiv preprint arXiv:2505.05477, 2025.

[35] I. A. Avila Castro et al., “Generative adversarial networks with fully connected layers to denoise ppg signals,” Physiological measurement, vol. 46, no. 2, p. 025 008, 2025.

[36] S. Ruipérez-Campillo et al., “Reducing diverse sources of noise in ventricular electrical signals using variational autoencoders,” Expert Systems with Applications, vol. 300, p. 130 185, 2026.

[37] S. Ruipérez-Campillo et al., “Antithetic sampling enhanced probabilistic diffusion for denoising cardiac time series,” IEEE journal of biomedical and health informatics, 2026.

[38] Y. Hou, R. Liu, M. Shu, X. Xie, and C. Chen, “Deep neural network denoising model based on sparse representation algorithm for ecg signal,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–11, 2023.

[39] F. P. Romero, D. C. Piñol, and C. R. Vázquez-Seisdedos, “Deepfilter: An ecg baseline wander removal filter using deep learning techniques,” Biomedical Signal Processing and Control, vol. 70, p. 102 992, 2021.

[40] D. Zhu, V. K. Chhabra, and M. M. Khalili, “Ecg signal denoising using multi-scale patch embedding and transformers,” in ICML 2024 Next Generation of Sequence Modeling Architectures Workshop, 2024.

[41] S. Kiranyaz et al., Blind ECG restoration by operational cycle-GANs, 2022. arXiv: 2202.00589 [eess.SP].

[42] P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith, “A dynamical model for generating synthetic electrocardiogram signals,” IEEE transactions on biomedical engineering, vol. 50, no. 3, 2003.

[43] A. M. Delaney, E. Brophy, T. E. Ward, C. Hennigan, and S. McLoone, “Synthesis of realistic ECG using generative adversarial networks,” in Proceedings of the 27th European Signal Processing Conference (EUSIPCO), 2019, pp. 1–5.

[44] F. Zhu, F. Ye, Y. Fu, Q. Liu, and B. Shen, “Electrocardiogram generation with a bidirectional LSTM-CNN generative adversarial network,” Scientific Reports, vol. 9, no. 1, p. 6734, 2019.

[45] V. V. Kuznetsov, V. A. Moskalenko, D. V. Gribanov, and N. Y. Zolotykh, “Interpretable feature generation in ECG using a variational autoencoder,” Frontiers in Genetics, vol. 12, 2021, ISSN: 1664-8021.

[46] A. A. Ali, I. Zimerman, and L. Wolf, “The hidden attention of mamba models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025, pp. 1516–1534.

[47] A. Z. Sellam, I. Benaissa, A. Taleb-Ahmed, L. Patrono, and C. Distante, “Mamba adaptive anomaly transformer with association discrepancy for time series,” Engineering Applications of Artificial Intelligence, vol. 160, p. 11 685, 2025.

[48] O. Ronneberger, P. Fischer, and T. Brox, U-net: Convolutional networks for biomedical image segmentation, 2015. arXiv: 1505.04597 [cs.CV].

[49] Y. Bengio, P. Simard, and P. Frasconi, “Learning long-term dependencies with gradient descent is difficult,” IEEE transactions on neural networks, vol. 5, no. 2, pp. 157–166, 1994.

[50] R. Pascanu, T. Mikolov, and Y. Bengio, On the difficulty of training recurrent neural networks, 2013. arXiv: 1211.5063 [cs.LG].

[51] M. Dias, P. Probst, L. Silva, and H. Gamboa, “Cleaning ECG with deep learning: A denoiser based on gated recurrent units,” in Jun. 2023, pp. 149–160, ISBN: 978-3-031-36006-0.

[52] A. L. Goldberger, Z. D. Goldberger, and A. Shvilkin, Goldberger’s Clinical Electrocardiography: A Simplified Approach, 10th ed. Philadelphia: Elsevier, 2024, ISBN: 978-0-323-82475- 0.

[53] E. Ramirez et al., “The art of selecting the ecg input in neural networks to classify heart diseases: A dual focus on maximizing information and reducing redundancy,” Frontiers in physiology, vol. 15, p. 1 452 829, 2024.

[54] P. Wagner et al., “Ptb-xl, a large publicly available electrocardiography dataset,” Scientific data, vol. 7, no. 1, pp. 1–15, 2020.

[55] A. Taddei et al., “The european st-t database: Standard for evaluating systems for the analysis of st-t changes in ambulatory electrocardiography,” European heart journal, vol. 13, no. 9, pp. 1164–1172, 1992.

[56] N. V. Thakor, J. G. Webster, and W. J. Tompkins, “Estimation of QRS complex power spectra for design of a QRS filter,” IEEE Transactions on Biomedical Engineering, vol. BME-31, no. 11, pp. 702–706, 1984.

[57] G. B. Moody, W. E. Muldrow, and R. G. Mark, “A noise stress test for arrhythmia detectors,” in Computers in Cardiology, vol. 11, 1984, pp. 381–384.

[58] L. Hu, W. Cai, Z. Chen, and M. Wang, “A lightweight u-net model for denoising and noise localization of ecg signals,” Biomedical Signal Processing and Control, vol. 88, p. 105 504, 2024, ISSN: 1746-8094.

[59] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning, D. Precup and Y. W. Teh, Eds., ser. Proceedings of Machine Learning Research, vol. 70, PMLR, 2017, pp. 1321–1330.

[60] N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, “Deep learning for ECG analysis: Benchmarks and insights from PTB-XL,” IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 5, pp. 1519–1528, 2021.

[61] H. Ismail Fawaz et al., “Inceptiontime: Finding alexnet for time series classification,” Data Mining and Knowledge Discovery, vol. 34, no. 6, pp. 1936–1962, Sep. 2020, ISSN: 1573- 756X.

[62] Z. Wang, W. Yan, and T. Oates, Time series classification from scratch with deep neural networks: A strong baseline, 2016. arXiv: 1611.06455 [cs.LG].

[63] P. Kumar and V. K. Sharma, “Detection and classification of ECG noises using decomposition on mixed codebook for quality analysis,” Healthcare Technology Letters, vol. 7, no. 1, pp. 18–24, 2020.

[64] B. Surawicz and T. K. Knilans, Chou’s Electrocardiography in Clinical Practice: Adult and Pediatric, 6th. Philadelphia: Saunders/Elsevier, 2008.

[65] B. Surawicz et al., “AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram: Part III: Intraventricular conduction disturbances,” Journal of the American College of Cardiology, vol. 53, no. 11, pp. 976–981, 2009.

[66] N. Pielawski and C. Wählby, “Introducing Hann windows for reducing edge-effects in patch-based image segmentation,” PLoS ONE, vol. 15, no. 3, e0229839, 2020.

<table><tr><td rowspan=10 colspan=5>A       Receptive Field Computation                    13B       Signal Compression Ablation                    13C       Memory Scaling Comparison                    13D       Training Details for Baseline Models            13E       Training Hyperparameters                       13Training Convergence                            14PTB-XL                                         14Synthetic ECG Parameters                       14I       Noise Pipeline Validation                        14J       Downstream Classifier Architectures             15K      Recording-Length Experiment: Training Details 15L       Effect of Training Data Volume</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>H</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>Recording-Length</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Effect of Training</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>M</td><td></td><td></td><td rowspan=4 colspan=1>151516</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>N</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>P</td><td></td><td></td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=2>Q</td><td></td><td></td><td rowspan=1 colspan=1>17</td></tr><tr><td rowspan=1 colspan=2>R</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>S</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>T</td><td></td><td></td><td></td></tr></table>

## A. Receptive Field Computation

The receptive field of a chain of L convolutions is the number of input samples that can influence a single output sample. For a network where layer l has kernel size $k _ { l }$ and stride $s _ { l } \colon$

$$
r _ { 0 } = \sum _ { l = 1 } ^ { L } \left( ( k _ { l } - 1 ) \prod _ { i = 1 } ^ { l - 1 } s _ { i } \right) + 1 .\tag{1}
$$

Pooling layers are treated identically, using their pool size as $k _ { l }$ and pool stride as $s _ { l }$ . Table S1 details the per-layer contribution for the UNet encoder.

## B. Signal Compression Ablation

Figure S1 compares the training dynamics of UNet-Mamba with and without the learnable log-compression layer (Section III-A.5). Without compression, the model converges faster initially—consistent with easily fitting the dominant scale component—but plateaus at a higher error, indicating that capacity is spent on amplitude scale at the expense of morphological structure.

## C. Memory Scaling Comparison

MECGE [28] applies bidirectional Mamba along both time and frequency axes in the STFT domain. Its Frequency Bi-Mamba branch creates one sequence per time frame, so the effective batch size scales as ${ \mathcal { O } } ( B \cdot T / h )$ where h is the STFT hop size. For a 40 s recording at 360 Hz with hop size $h { = } 8$ this yields an effective batch of $B \times 1 { , } 8 0 0$ in the frequency branch alone. By contrast, UNet-Mamba applies Mamba only along the time axis after 20× down-sampling, keeping the effective batch size at B and the sequence length at $T / 2 0$ Figure S2 compares peak GPU memory allocation.

TABLE S1 PER-LAYER RECEPTIVE FIELD COMPUTATION FOR THE UNET ENCODER.
<table><tr><td>l</td><td>Layer Type</td><td>sl</td><td> $k _ { l }$ </td><td>l-1  $\prod { s _ { i } }$  i=1</td><td>l-1  $( k _ { l } - 1 ) \prod s _ { i }$  i=1</td></tr><tr><td>1</td><td>Conv</td><td>1</td><td>25</td><td>1</td><td>24</td></tr><tr><td>2</td><td>Conv</td><td>1</td><td>25</td><td>1</td><td>24</td></tr><tr><td>3</td><td>Conv</td><td>1</td><td>25</td><td>1</td><td>24</td></tr><tr><td>4</td><td>AvgPool</td><td>5</td><td>5</td><td>1</td><td>4</td></tr><tr><td>5</td><td>Conv</td><td>1</td><td>15</td><td>5</td><td>70</td></tr><tr><td>6</td><td>Conv</td><td>1</td><td>15</td><td>5</td><td>70</td></tr><tr><td>7</td><td>Conv</td><td>1</td><td>15</td><td>5</td><td>70</td></tr><tr><td>8</td><td>AvgPool</td><td>2</td><td>2</td><td>5</td><td>5</td></tr><tr><td>9</td><td>Conv</td><td>1</td><td>5</td><td>10</td><td>40</td></tr><tr><td>10</td><td>Conv</td><td>1</td><td>5</td><td>10</td><td>40</td></tr><tr><td>11</td><td>Conv</td><td>1</td><td>5</td><td>10</td><td>40</td></tr><tr><td>12</td><td>AvgPool</td><td>2</td><td>2</td><td>10</td><td>10</td></tr><tr><td>13</td><td>Conv</td><td>1</td><td>3</td><td>20</td><td>40</td></tr><tr><td>14</td><td>Conv</td><td>1</td><td>3</td><td>20</td><td>40</td></tr><tr><td>15</td><td>Conv</td><td>1</td><td>3</td><td>20</td><td>40</td></tr><tr><td colspan="5">Sum: Receptive Field  $( r _ { 0 } = \mathrm { S u m } + 1 ) \colon$ </td><td>541 542</td></tr></table>

![](images/a85cc4c97a4f2b3bcb15f58a53fb8d3fbce2c95e0b8629640bd3e74a88bde483.jpg)  
Fig. S1. Test RMSE over training steps for UNet-Mamba with and without the learnable log-compression layer. Removing compression yields faster initial convergence but a worse final optimum.

## D. Training Details for Baseline Models

For the baseline models (UNet, IMUNet, DAE, DRNN, DRnet variants), the optimizer and scheduler follow the respective original publications, using Adam with learning rate $1 0 ^ { - 3 }$ and ReduceLROnPlateau. MECGE uses AdamW with learning rate $1 0 ^ { - 4 }$ and exponential decay. For UNet-Mamba, we use Adam with $\eta _ { \mathrm { m a x } } = 8 \times 1 0 ^ { - 4 }$ and cosine annealing with 10-epoch linear warm-up from $\eta _ { \mathrm { m a x } } \times 0 . 0 1$ to $\eta _ { \mathrm { m a x } } .$ , followed by cosine decay over 400 cycles to $\eta _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ , following [25]. Full hyperparameters are listed in Supplementary Material E.

## E. Training Hyperparameters

Table S2 summarizes the training hyperparameters for all models. All models use MSE loss with batch size 32.

![](images/33df787f156d7c4b972f9fee16cedbfb122286e708172e543e0e3aa1f3a073a0.jpg)  
Fig. S2. Peak GPU memory allocation as a function of ECG signal length (1–5 s at 360 Hz, batch size 2). A: inference (forward pass). B: training (forward + backward). MECGE training requires substantially more memory than UNet-Mamba due to STFT-domain processing.

TABLE S2  
TRAINING HYPERPARAMETERS FOR BASELINE AND PROPOSED MODELS.
<table><tr><td colspan="2">UNet / IMUNet / DAE / DRNN / DR-net-S2</td><td>MECGE</td><td>UNet-Mamba (ours)</td></tr><tr><td>Optimizer</td><td>Adam</td><td>AdamW</td><td>Adam</td></tr><tr><td>Learning rate</td><td>10−3</td><td> $1 0 ^ { - 4 }$ </td><td> $8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR scheduler Warmup epochs</td><td>ReduceLROnPlateau</td><td>Exp. decay</td><td>Cosine + warmup</td></tr><tr><td>Tmax (cosine)</td><td></td><td></td><td>10 400</td></tr><tr><td></td><td></td><td></td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>ηmin Max epochs</td><td>500</td><td>120</td><td>500</td></tr><tr><td></td><td>120-240</td><td>15</td><td>120-240</td></tr><tr><td>ES patience</td><td></td><td></td><td></td></tr></table>

## F. Training Convergence

All models converge well before the maximum number of epochs, with test SNR curves reaching a stable plateau. Figure S3 shows the train loss and test SNR across all three datasets.

## G. PTB-XL

The waveform data underlying PTB-XL were collected with devices from Schiller AG at the Physikalisch-Technische Bundesanstalt over nearly seven years (October 1989–June 1996). The cohort is 52% male and 48% female, with ages spanning 0–95 years (median $6 2 .$ interquartile range 22). Signals are stored in WFDB format at 16-bit precision with a resolution of $1 \mu \mathrm { V } / \mathrm { L S B }$ . Each record was annotated with a free-text report converted into standardized SCP-ECG statements; a large fraction was additionally validated by a second cardiologist. The recommended 10-fold splits are stratified by patient. Records in folds 9 and 10 underwent at least one human evaluation and are therefore of particularly high label quality. For full details, see [54].

## H. Synthetic ECG Parameters

Table S3 lists the morphology parameters used for synthetic ECG generation, following the notation of [42]. For each waveform event $i ~ \in ~ \{ P , Q , R , S , T \}$ , the amplitude $a _ { i }$ is sampled uniformly from $\left[ { { c } _ { { { a } _ { i } } } } \mathrm { ~ - ~ } { { \delta } _ { { { a } _ { i } } } } \mathrm { , ~ } { { c } _ { { { a } _ { i } } } } \mathrm { ~ + ~ } { { \delta } _ { { { a } _ { i } } } } \right]$ and the width $b _ { i }$ from $\left[ c _ { b _ { i } } - \delta _ { b _ { i } } , c _ { b _ { i } } + \delta _ { b _ { i } } \right]$

![](images/59a672ee6017c6ec395b24624ac4181d519740afbffa58bea7e0bb437c96ac80.jpg)  
Fig. S3. Training loss (top row) and test SNR (bottom row) over training steps across all three datasets: (A) European ST-T, (B) synthetic, (C) PTB-XL. Stage 1 models on the left of each panel; Stage 2 (DR-net) on the right.

TABLE S3  
SYNTHETIC ECG SIMULATION PARAMETERS FOR THE PQRST MORPHOLOGY.
<table><tr><td></td><td>P</td><td>Q</td><td>R</td><td>S</td><td>T</td></tr><tr><td> $c _ { a _ { i } }$ </td><td>1.2</td><td>-5.0</td><td>30.0</td><td>-7.5</td><td>0.75</td></tr><tr><td> $\delta _ { a _ { i } }$ </td><td>0.6</td><td>0.2</td><td>0.0</td><td>1.0</td><td>0.35</td></tr><tr><td> $c _ { b _ { i } }$ </td><td>0.25</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.4</td></tr><tr><td> $\delta _ { b _ { i } }$ </td><td>0.1</td><td>0.1</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

## I. Noise Pipeline Validation

We validated the noise injection pipeline by loading 1,000 preprocessed PTB-XL segments and applying the full noise pipeline under three configurations (light, medium, strong), corresponding to the upper bound, midpoint, and lower bound of the per-type SNR ranges recommended by [58]. The theoretical combined SNR when K independent noise sources are added simultaneously is:

$$
\mathrm { S N R } _ { \mathrm { c o m b i n e d } } = 1 0 \log _ { 1 0 } \left( \frac { 1 } { \sum _ { k = 1 } ^ { K } 1 0 ^ { - \mathrm { S N R } _ { k } / 1 0 } } \right) .\tag{2}
$$

Table S4 summarizes the per-type SNR values and predicted combined SNR for each configuration. Figure S4 shows that the empirical medians closely match the theoretical predictions and the ordering across configurations is preserved.

TABLE S4  
NOISE CONFIGURATIONS. PER-TYPE SNR VALUES (DB) ARE DRAWN FROM RANGES RECOMMENDED BY [58].
<table><tr><td>Config</td><td>BW</td><td>MA</td><td>EM</td><td>AWGN</td><td>Combined SNR</td></tr><tr><td>Light</td><td>5.0</td><td>10.0</td><td>15.0</td><td>25.0</td><td>3.46</td></tr><tr><td>Medium</td><td>2.5</td><td>7.5</td><td>12.5</td><td>22.5</td><td>0.96</td></tr><tr><td>Strong</td><td>0.0</td><td>5.0</td><td>10.0</td><td>20.0</td><td>-1.54</td></tr></table>

## J. Downstream Classifier Architectures

The two downstream classification models are:

Inception1D: a 1D adaptation of InceptionTime [61], stacking six inception blocks with multi-scale parallel convolutions and residual shortcuts every three blocks, followed by adaptive concatenation pooling and a linear head.

ResNet1D-Wang: a 1D residual network following [62], with three single-block residual stages using kernel sizes 5 and 3, stride-1 convolutions, and the same adaptive concatenation pooling and linear head as Inception1D.

Both models are trained using binary cross-entropy loss on clean data from the PTB-XL training folds and accept 12-channel inputs with an independent binary prediction per diagnostic class.

## K. Recording-Length Experiment: Training Details

Tables S5 and S6 summarize the number of epochs and wall-clock training time per curriculum stage on the synthetic and European ST-T datasets, respectively. Figures S5 and S6 show the corresponding train loss curves.

TABLE S5  
TRAINING DETAILS FOR THE SYNTHETIC DATASET ACROSS SPLITLENGTHS. TRAINED ON A SINGLE NVIDIA RTX A5000.
<table><tr><td>Method</td><td>Split Length</td><td>Epochs</td><td>Runtime</td></tr><tr><td>UNet-Mamba1-3B</td><td>14400</td><td>180</td><td>41m 45s</td></tr><tr><td>UNet-Mamba1-3B</td><td>7200</td><td>256</td><td>1h 4m 52s</td></tr><tr><td>UNet-Mamba1-3B</td><td>3600</td><td>250</td><td>1h 50m 39s</td></tr><tr><td>UNet-Mamba1-3B</td><td>1800</td><td>286</td><td>7h 18m 38s</td></tr><tr><td>UNet</td><td>14400</td><td>102</td><td>12m 33s</td></tr><tr><td>UNet</td><td>7200</td><td>105</td><td>21m 22s</td></tr><tr><td>UNet</td><td>3600</td><td>108</td><td>38m 26s</td></tr><tr><td>UNet</td><td>1800</td><td>245</td><td>5h 42m 15s</td></tr></table>

## L. Effect of Training Data Volume

Figure S7 shows reconstruction SNR and RMSE on PTB-XL as a function of the number of training folds. Mambabased models improve monotonically up to 8 folds, whereas UNet and IMUNet peak at 6 folds and slightly degrade with additional data, suggesting that the higher capacity of the Mamba bottleneck allows it to continue benefiting from additional training data where purely convolutional models saturate earlier.

TABLE S6  
TRAINING DETAILS FOR THE EUROPEAN ST-T DATASET ACROSS SPLITLENGTHS. TRAINED ON A SINGLE NVIDIA RTX A5000.
<table><tr><td>Method</td><td>Split Length</td><td>Epochs</td><td>Runtime</td></tr><tr><td>UNet-Mamba1-3B</td><td>14400</td><td>263</td><td>32m 13s</td></tr><tr><td>UNet-Mamba1-3B</td><td>7200</td><td>285</td><td>39m 44s</td></tr><tr><td>UNet-Mamba1-3B</td><td>3600</td><td>274</td><td>1h 7m 7s</td></tr><tr><td>UNet-Mamba1-3B</td><td>1800</td><td>113</td><td>1h 34m 27s</td></tr><tr><td>UNet</td><td>14400</td><td>257</td><td>17m 13s</td></tr><tr><td>UNet</td><td>7200</td><td>206</td><td>21m 56s</td></tr><tr><td>UNet</td><td>3600</td><td>334</td><td>1h 4m</td></tr><tr><td>UNet</td><td>1800</td><td>306</td><td>3h 32m 46s</td></tr></table>

## M. Reconstruction RMSE

Figure S8 reports reconstruction RMSE across all three datasets, supplementing the SNR results in Figure 3.

## N. Pairwise Tests for Reconstruction SNR

Table S7 shows the pairwise comparison between the proposed models and each baseline model on the synthetic test set along with the 95% confidence intervals. Table S8 gives the results of the same evaluation on the PTB-XL dataset, and Table S9 on the ST-T dataset. On all three datasets, the confidence intervals do not contain 0, implying a significant difference (at α = 0.05 level) between the baseline models and the proposed models.

TABLE S7  
PAIRWISE COMPARISON WITH 95% CONFIDENCE INTERVALS FOR THE SYNTHETIC TEST SET (n = 256 SOURCE RECORDINGS). DR-NET MAMBA1-3B OUTPERFORMS EVERY BASELINE (SMALLEST GAIN +0.92 DB SNR OVER DR-NET-IMUNET, CI [+0.79, +1.06]). SEE FIGURE 3A.
<table><tr><td>Baseline</td><td colspan="2">UNet Mamba1-3B</td><td colspan="2">DR-net Mamba1-3B</td></tr><tr><td>DRNN</td><td></td><td>+11.922 [+11.761, +12.098]</td><td>+12.546</td><td>[+12.355, +12.720]</td></tr><tr><td>DAE</td><td>+6.657</td><td>[+6.492, +6.811]</td><td>+7.281</td><td>[+7.116,+7.440]</td></tr><tr><td>UNet</td><td>+2.551</td><td>[+2.414, +2.699]</td><td>+3.175</td><td>[+3.025, +3.332</td></tr><tr><td>DR-net-UNet</td><td>+2.538</td><td>[+2.386, +2.682</td><td>+3.162</td><td>[+3.015, +3.316</td></tr><tr><td>IMUNet</td><td>+1.172</td><td>[+1.019, +1.309]</td><td>+1.796</td><td>[+1.646, +1.954]</td></tr><tr><td>DR-net-IMUNet</td><td>+0.294</td><td>[+0.148, +0.440]</td><td>+0.918</td><td>[+0.785, +1.060]</td></tr><tr><td>MECGE</td><td>+8.194</td><td>[+8.002, +8.400]</td><td>+8.818</td><td>+8.623, +9.049]</td></tr><tr><td>UNet Mamba1-3B</td><td></td><td></td><td>+0.624</td><td>[+0.573, +0.674]</td></tr><tr><td>DR-net Mamba1-3B</td><td></td><td>-0.624 [−0.674, −0.573]</td><td></td><td></td></tr></table>

TABLE S8

PAIRWISE COMPARISON WITH 95% CONFIDENCE INTERVALS FOR PTB-XL (n = 1,639 RECORDINGS, 10 S EACH AT 360 HZ). DR-NET MAMBA1-3B OUTPERFORMS EVERY BASELINE (SMALLEST GAIN +0.83 DB SNR OVER DR-NET-IMUNET, CI [+0.78, +0.88]). SEE FIGURE 3B.
<table><tr><td>Baseline</td><td>UNet Mamba1-3B</td><td></td><td>DR-net Mamba1-3B</td></tr><tr><td>DRNN</td><td>+3.955</td><td>[+3.889, +4.019]</td><td>+4.854 [+4.785, +4.926]</td></tr><tr><td>DAE</td><td>+4.385</td><td>[+4.297, +4.473]</td><td>+5.284 [+5.193, +5.382]</td></tr><tr><td>UNet</td><td>+2.409</td><td>[+2.359, +2.460]</td><td>+3.308 [+3.244, +3.366]</td></tr><tr><td>DR-net-UNet</td><td>+0.206</td><td>[+0.162, +0.252]</td><td>+1.105 [+1.062, +1.148]</td></tr><tr><td>IMUNet</td><td>+1.063</td><td>[+1.020,+1.104]</td><td>+1.962 [+1.911,+2.016]</td></tr><tr><td>DR-net-IMUNet</td><td>-0.070</td><td>[−0.116, −0.022]</td><td>+0.828 [+0.778, +0.882]</td></tr><tr><td>MECGE</td><td>+1.485</td><td>[+1.395, +1.579]</td><td>+2.384 [+2.299, +2.475]</td></tr><tr><td>UNet Mamba1-3B</td><td></td><td></td><td>+0.899 [+0.868, +0.930]</td></tr><tr><td>DR-net Mamba1-3B</td><td></td><td>-0.899 [−0.930, -0.868]</td><td></td></tr></table>

A. Empirical vs Theoretical SNR  
![](images/8f6f9946bb19bb3f3e71124913f772f56f1d72ad7c7ec065a608636d1cc35d8f.jpg)  
Noise Configuration

B. Empirical vs Theoretical RMSE  
![](images/2d7628893627d9909b34b5a871196921e9477d8dbeabf939d56ab05ed5bacc28.jpg)  
Noise Configuration

Fig. S4. Empirical vs. theoretical SNR (left) and RMSE (right) across three noise configurations. Boxplots show per-segment distributions; dashed lines indicate theoretical expected values.  
![](images/9c6d4e0dd7501f4c5060eebbcde464419055715b024e2a243c24867aa6099ffa.jpg)

![](images/7796ab0e939cf636dac185bb3e6c9ad3fd5427e6977869b58158d73a2ad4251b.jpg)

Fig. S5. Train loss curves for the recording-length experiment on the synthetic dataset.  
![](images/8a702db0215087259a5744aa96936b960e176d4fc5c2f4abd17eae1b6c56495f.jpg)

![](images/0586026cd8f0d7564ac6a8f2b50db0c14aba13e1291c54facffe49357d075c26.jpg)  
Fig. S6. Train loss curves for the recording-length experiment on the European ST-T dataset.

TABLE S9  
PAIRWISE COMPARISON WITH 95% CONFIDENCE INTERVALS FOR THE ST-T DATABASE (n = 256 RECORDINGS, 40 S). DR-NET MAMBA1-3B OUTPERFORMS EVERY BASELINE (SMALLEST GAIN +0.34 DB SNR OVER DR-NET-IMUNET, CI [+0.20, +0.49]). SEE FIGURE 3C.
<table><tr><td>Baseline</td><td colspan="2">UNet Mamba1-3B</td><td colspan="2">DR-net Mamba1-3B</td></tr><tr><td>DRNN</td><td>+3.650</td><td>[+3.489, +3.813]</td><td>+4.878</td><td>[+4.712, +5.055]</td></tr><tr><td>DAE</td><td>+4.966</td><td>[+4.801, +5.113]</td><td>+6.194</td><td>[+6.014, +6.380]</td></tr><tr><td>UNet</td><td>+2.078</td><td>[+1.940, +2.226]</td><td>+3.307</td><td>[+3.157, +3.453]</td></tr><tr><td>DR-net-UNet</td><td>-0.784</td><td>[−0.930, -0.622]</td><td>+0.445</td><td>[+0.315, +0.592]</td></tr><tr><td>IMUNet</td><td>+1.391</td><td>[+1.232, +1.550]</td><td>+2.619</td><td>[+2.428, +2.815]</td></tr><tr><td>DR-net-IMUNet</td><td>-0.887</td><td>[−1.045, −0.713]</td><td>+0.342</td><td>[+0.198, +0.493]</td></tr><tr><td>MECGE</td><td></td><td>+0.085 [−0.085, +0.261]</td><td>+1.313</td><td>[+1.138, +1.472]</td></tr><tr><td>UNet Mamba1-3B</td><td></td><td></td><td>+1.229</td><td>[+1.130, +1.331]</td></tr><tr><td>DR-net Mamba1-3B</td><td></td><td>-1.229 [−1.331, -1.130]</td><td></td><td></td></tr></table>

TABLE S10  
ABLATION STUDY SEPARATING THE EFFECTS OF MAMBA, DR-NET, AND LOG COMPRESSION. MEAN DIFFERENCES AND 95% CONFIDENCE INTERVALS ARE REPORTED FROM PAIRED BOOTSTRAP ANALYSIS WITH B = 1000 RESAMPLES WITH REPLACEMENT.

## O. Impact of Compression Layer in Reconstruction Accuracy

## P. Downstream Classification under Lower Noise

Table S11 reports the per-type SNR used for the lower noise setting. Figure S9 shows the corresponding downstream classification results. Table S12 provides the per-superdiagnosticclass AUROC breakdown, and Table S13 reports macro sensitivity, specificity, and F1 for Inception1D. Relative model rankings are consistent with the strong noise setting.

<table><tr><td>Baseline</td><td>UNet</td><td>DR-net- UNet</td><td>UNet Mamba1-3B (No Comp)</td><td>DR-net Mamba1-3B (No Comp)</td><td>UNet Mamba1-3B</td><td>DR-net Mamba1-3B</td></tr><tr><td>UNet</td><td></td><td>+2.9</td><td>+1.2</td><td>+3.3</td><td>+2.1</td><td>+3.3 [+2.7, +3.0] [+1.1, +1.3] [+3.2, +3.4] [+1.9, +2.2] [+3.2, +3.5]</td></tr><tr><td>DR-net- UNet</td><td></td><td></td><td>-1.6 [−1.8, −1.5] [+0.3, +0.6] [−0.9, −0.6] [+0.3, +0.6]</td><td>+0.5</td><td>-0.8</td><td>+0.4</td></tr><tr><td>UNet Mamba1-3B</td><td></td><td></td><td></td><td>+2.1 [+2.0, +2.2] [+0.7, +1.0] [+1.9, +2.2]</td><td>+0.9</td><td>+2.1</td></tr><tr><td>(No Comp) DR-net Mamba1-3B</td><td></td><td></td><td></td><td></td><td>-1.2</td><td>-0.0</td></tr><tr><td>(No Comp) UNet</td><td></td><td></td><td></td><td></td><td></td><td>[−1.4, −1.1] [−0.1, +0.1] +1.2</td></tr></table>

![](images/898aa5fe7c13ae22b66bb41baa13dceecfee4175a75fd5138238457900ceb3d8.jpg)  
Fig. S7. Reconstruction SNR (top) and RMSE (bottom) on PTB-XL as a function of the number of training folds.

![](images/e08b139313c126373ed03d6fb3a17da48c47f11d9050fbd3920bec7dc3bd1031.jpg)

B. Denoising PTB-XL  
![](images/837533f1e8fcac6c8495b18290babaab56b14fe16d601952324c91fc12ced897.jpg)

C. European ST-T  
![](images/66b4e6711dcb04ffb56192a992ec54ae697a420b4783cd3868850815b8e41255.jpg)  
Fig. S8. Reconstruction RMSE on synthetic (A), PTB-XL (B), and European ST-T (C) datasets. Empirical noisy SNR = 0.96 dB (medium noise setting, see Supplementary Material I). This figure supplements Figure 3.

TABLE S11  
PER-TYPE SNR (DB) FOR THE LOWER (MEDIUM) DOWNSTREAM NOISE SETTING.
<table><tr><td>BW</td><td>MA</td><td>EM</td><td>AWGN</td></tr><tr><td>2.5</td><td>7.5</td><td>12.5</td><td>22.5</td></tr></table>

## Q. Downstream Macro Results for ResNet1D-Wang

Table S14 reports macro-averaged sensitivity, specificity, and F1 for the ResNet1D-Wang classifier under the strong noise setting, complementing the Inception1D results in the

main text.

## R. Downstream Macro Results for Inception1D

Figure S10 summarizes downstream classification performance on PTB-XL under the strong noise setting (BW 0dB; MA 5dB; EM 10dB; AWGN 20dB) using the Inception1D backbone.

## S. Superdiagnostic Sensitivity, Specificity, and F1

Tables S15 and S16 report per-superdiagnostic-class sensitivity, specificity, and F1 under the strong noise setting for Inception1D and ResNet1D-Wang, respectively.

TABLE S12
<table><tr><td></td><td>CD</td><td>HYP</td><td>MI</td><td>NORM</td><td>STTC</td></tr><tr><td>UNet Mamba1-3B</td><td> $. 9 2 6 \pm . 0 2$ </td><td> $. 8 8 8 \pm . 0 2$ </td><td> $. 9 1 6 \pm . 0 1$ </td><td> $. 9 3 1 \pm . 0 1$ </td><td> $. 8 8 8 \pm . 0 2$ </td></tr><tr><td>DRNET Mamba1-3B</td><td> $\overline { { { \bf { 9 } } 2 7 \pm { \bf { \Omega } } . { \bf { 0 } } 2 } }$ </td><td> $\mathbf { 8 9 5 \pm { \sigma . 0 2 } }$ </td><td> $\overline { { 9 1 5 \pm . 0 2 } }$ </td><td> $. 9 3 3 \pm . 0 1$ </td><td> $. 8 8 9 \pm . 0 2$ </td></tr><tr><td>UNet Mamba1-3B (LS)</td><td> $. 9 2 4 \pm . 0 1$ </td><td> $. 8 9 4 \pm . 0 2$ </td><td> $\mathbf { 9 1 8 } \pm \mathbf { . 0 2 }$ </td><td> $\mathbf { 9 3 5 \pm . 0 1 }$ </td><td> $\overline { { { \bf 8 9 9 } \pm { \bf 8 2 } } }$ </td></tr><tr><td>DRNN</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DAE</td><td> $. 9 1 1 \pm . 0 1$ </td><td>.887 ± .02</td><td> $. 9 0 3 \pm . 0 1$ </td><td> $. 9 1 9 \pm . 0 1$ </td><td> $. 8 5 7 \pm . 0 2$ </td></tr><tr><td>UNet</td><td> $. 8 6 3 \pm . 0 2$ </td><td>.853 ± .02</td><td> $. 8 8 6 \pm . 0 2$ </td><td> $. 9 0 9 \pm . 0 1$ </td><td> $. 8 5 9 \pm . 0 2$ </td></tr><tr><td>DRNET-UNet</td><td> $. 9 1 9 \pm . 0 1$ </td><td>.877 ± .02</td><td> $. 9 1 3 \pm . 0 1$ </td><td> $. 9 2 6 \pm . 0 1$ </td><td>.881 ± .02</td></tr><tr><td>IMUNet</td><td> $. 9 1 7 \pm . 0 2$ </td><td>.886 ± .02</td><td> $. 9 0 9 \pm . 0 2$   $. 9 1 0 \pm . 0 1$ </td><td> $. 9 2 7 \pm . 0 1$ </td><td>.881 ± .02  $. 8 8 6 \pm . 0 2$ </td></tr><tr><td>DRNET-IMUNet</td><td> $. 9 2 2 \pm . 0 2$   $. 9 1 9 \pm . 0 1$ </td><td>.883 ± .02  $. 8 9 2 \pm . 0 2$ </td><td> $. 9 0 9 \pm . 0 2$ </td><td> $. 9 2 9 \pm . 0 1$   $. 9 2 8 \pm . 0 1$ </td><td> $. 8 8 7 \pm . 0 2$ </td></tr><tr><td>MECGE</td><td> $. 9 1 6 \pm . 0 2$ </td><td> $. 8 8 3 \pm . 0 3$ </td><td> $. 9 0 3 \pm . 0 1$ </td><td> $. 9 2 4 \pm . 0 1$ </td><td> $. 8 8 8 \pm . 0 2$ </td></tr><tr><td>clean</td><td> $. 9 3 2 \pm . 0 1$ </td><td> $. 9 0 6 \pm . 0 2$ </td><td> $. 9 2 5 \pm . 0 2$ </td><td> $. 9 3 8 \pm . 0 1$ </td><td> $. 9 1 4 \pm . 0 2$ </td></tr><tr><td>noisy</td><td></td><td> $. 8 8 1 \pm . 0 2$ </td><td> $. 8 8 7 \pm . 0 2$ </td><td> $. 9 2 3 \pm . 0 1$ </td><td> $. 8 9 6 \pm . 0 2$ </td></tr><tr><td></td><td> $. 9 1 9 \pm . 0 2$ </td><td></td><td></td><td></td><td></td></tr></table>

TABLE S13  
TABLE S15

MACRO-AVERAGED SENSITIVITY, SPECIFICITY, AND F1 ON PTB-XL (LOWER NOISE, INCEPTION1D). BOLD/UNDERLINED: BEST/SECOND-BEST.  
PER-SUPERDIAGNOSTIC-CLASS SENSITIVITY, SPECIFICITY, AND F1 ON PTB-XL (STRONG NOISE, INCEPTION1D). BOLD/UNDERLINED: BEST/SECOND-BEST.
<table><tr><td></td><td>Sensitivity</td><td>Specificity</td><td>F1</td></tr><tr><td>UNet Mamba1-3B (ours)</td><td> $. 7 8 5 \pm . 0 2$ </td><td> $. 8 6 2 \pm . 0 1$ </td><td>.723 ± .01</td></tr><tr><td>DRNET Mamba1-3B (ours)</td><td> $\underline { { 7 8 8 \pm . 0 2 } }$ </td><td> $\mathbf { 8 6 5 \pm . 0 1 }$ </td><td> $. 7 2 7 \pm . 0 1$ </td></tr><tr><td>UNet Mamba1-3B (Lead aware) (ours)</td><td> $\overline { { 7 9 1 \pm . 0 2 } }$ </td><td> $. 8 6 1 \pm . 0 1$ </td><td> $\underline { { 7 2 5 \pm . 0 1 } }$ </td></tr><tr><td>DAE</td><td> $. 6 9 8 \pm . 0 2$ </td><td> $. 8 4 6 \pm . 0 1$ </td><td> $. 6 5 3 \pm . 0 2$ </td></tr><tr><td>DRNN</td><td>.784 ± .02</td><td> $. 8 1 9 \pm . 0 1$ </td><td> $. 6 8 2 \pm . 0 2$ </td></tr><tr><td>MECGE</td><td> $. 7 6 0 \pm . 0 2$ </td><td> $. 8 4 6 \pm . 0 1$ </td><td> $. 6 9 3 \pm . 0 2$ </td></tr><tr><td>UNet</td><td> $. 7 4 8 \pm . 0 2$ </td><td> $. 8 5 4 \pm . 0 1$ </td><td> $. 6 9 1 \pm . 0 2$ </td></tr><tr><td>DRNET-UNet</td><td> $. 7 7 4 \pm . 0 2$ </td><td> $. 8 5 9 \pm . 0 1$ </td><td> $. 7 1 1 \pm . 0 2$ </td></tr><tr><td>IMUNet</td><td> $. 7 5 3 \pm . 0 2$ </td><td> $. 8 6 4 \pm . 0 1$ </td><td> $. 7 0 3 \pm . 0 2$ </td></tr><tr><td>DRNET-IMUNet</td><td> $. 7 7 7 \pm . 0 2$ </td><td> $. 8 6 4 \pm . 0 1$ </td><td> $. 7 1 7 \pm . 0 2$ </td></tr><tr><td>noisy</td><td></td><td></td><td></td></tr><tr><td>clean</td><td> $. 8 5 7 \pm . 0 1$   $. 8 1 5 \pm . 0 2$ </td><td> $. 7 5 7 \pm . 0 1$   $. 8 6 3 \pm . 0 1$ </td><td> $. 6 7 0 \pm . 0 2$   $. 7 3 8 \pm . 0 1$ </td></tr></table>

TABLE S14

MACRO-AVERAGED SENSITIVITY, SPECIFICITY, AND F1 ON PTB-XL (STRONG NOISE, RESNET1D-WANG). BOLD/UNDERLINED: BEST/SECOND-BEST.
<table><tr><td></td><td>Sensitivity</td><td>Specificity</td><td>F1</td></tr><tr><td>UNet Mamba1-3B (ours)</td><td> $. 7 9 7 \pm . 0 2$ </td><td> $. 8 5 3 \pm . 0 1$ </td><td> $. 7 2 2 \pm . 0 1$ </td></tr><tr><td>DRNET Mamba1-3B (ours)</td><td> ${ \bf 8 0 0 \pm 0 2 }$ </td><td> ${ \bf 8 5 8 \pm . 0 1 }$ </td><td> $. 7 2 7 \pm . 0 2$ </td></tr><tr><td>UNet Mamba1-3B (Lead Specific) (ours)</td><td>.798 ± .02</td><td> $8 5 8 \pm . 0 1$ </td><td> $\overline { { . 7 2 8 \pm . 0 2 } }$ </td></tr><tr><td>DAE</td><td> $. 7 3 3 \pm . 0 2$ </td><td> $. 8 3 1 \pm . 0 1$ </td><td> $. 6 6 8 \pm . 0 2$ </td></tr><tr><td>DRNN</td><td> $. 7 8 7 \pm . 0 2$ </td><td> $. 8 0 8 \pm . 0 1$ </td><td> $. 6 7 9 \pm . 0 2$ </td></tr><tr><td>MECGE</td><td> $. 7 8 6 \pm . 0 2$ </td><td> $. 8 2 6 \pm . 0 1$ </td><td> $. 6 9 1 \pm . 0 2$ </td></tr><tr><td>UNet</td><td> $. 7 5 7 \pm . 0 2$ </td><td> $. 8 3 9 \pm . 0 1$ </td><td> $. 6 8 1 \pm . 0 2$ </td></tr><tr><td>DRNET-UNet</td><td> $. 7 7 2 \pm . 0 2$ </td><td> $. 8 4 1 \pm . 0 1$ </td><td> $. 6 9 3 \pm . 0 2$ </td></tr><tr><td>IMUNet</td><td> $. 7 7 1 \pm . 0 2$ </td><td> $. 8 5 2 \pm . 0 1$ </td><td> $. 7 0 7 \pm . 0 2$ </td></tr><tr><td>DRNET-IMUNet</td><td> $. 7 8 1 \pm . 0 2$ </td><td> $. 8 5 6 \pm . 0 1$ </td><td> $. 7 1 3 \pm . 0 2$ </td></tr><tr><td>noisy</td><td> $. 8 3 1 \pm . 0 2$ </td><td> $. 7 4 1 \pm . 0 1$ </td><td> $. 6 5 2 \pm . 0 2$ </td></tr><tr><td>clean</td><td> $. 8 0 8 \pm . 0 2$ </td><td> $. 8 7 5 \pm . 0 1$ </td><td> $. 7 4 6 \pm . 0 2$ </td></tr></table>

<table><tr><td></td><td>CD</td><td>HYP</td><td></td><td>MI</td><td>NORM</td><td></td><td>STTC</td></tr><tr><td>Sensitivity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNet-Mamba1-3B</td><td>.822 ± .03</td><td></td><td>.526 ± .05</td><td>.865 ± .03</td><td></td><td> $. 9 1 8 \pm . 0 2$ </td><td>.792 ± .03</td></tr><tr><td>DR-net-Mamba1-3B</td><td>.834 ± .04</td><td></td><td>.534 ± .05</td><td>.854 ± .04</td><td> $. 9 1 7 \pm . 0 2$ </td><td></td><td>.801 ± .04</td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td>.816 ± .03</td><td></td><td>.556 ± .05</td><td> $. 8 1 5 \pm . 0 3$ </td><td> $\mathbf { 9 4 3 } \pm \mathbf { . 0 1 }$ </td><td></td><td> $. 8 2 4 \pm . 0 4$ </td></tr><tr><td>DAE</td><td>.640 ± .05</td><td></td><td>.299 ± .07</td><td> $\mathbf { 8 8 9 } \pm \mathbf { . 0 3 }$ </td><td>.891 ± .02</td><td></td><td>.769 ± .04</td></tr><tr><td>DRNN</td><td>.822 ± .03</td><td></td><td> $. 4 8 9 \pm . 0 6$ </td><td> $. 8 8 7 \pm . 0 3$ </td><td> $. 8 6 3 \pm . 0 3$ </td><td></td><td> $\mathbf { 8 6 0 \pm { \sigma . 0 3 } }$ </td></tr><tr><td>MECGE</td><td>.796 ± .03</td><td></td><td> $. 4 4 0 \pm . 0 5$ </td><td> $8 5 0 \pm . 0 3$ </td><td> $. 9 2 0 \pm . 0 2$ </td><td></td><td> $. 7 9 2 \pm . 0 4$ </td></tr><tr><td>UNet</td><td> $. 7 7 8 \pm . 0 3$ </td><td></td><td> $. 3 5 1 \pm . 0 6$ </td><td> $. 8 7 2 \pm . 0 3$ </td><td> $. 9 2 0 \pm . 0 2$ </td><td></td><td> $. 8 1 8 \pm . 0 4$ </td></tr><tr><td>DR-net-UNet</td><td> $. 8 1 6 \pm . 0 3$ </td><td></td><td> $. 4 9 6 \pm . 0 6$ </td><td> $. 8 3 5 \pm . 0 4$ </td><td> $\overline { { . 8 8 7 \pm . 0 2 } }$ </td><td></td><td> $. 8 3 3 \pm . 0 3$ </td></tr><tr><td>IMUNet</td><td> $. 7 9 4 \pm . 0 3$ </td><td></td><td> $. 4 0 7 \pm . 0 5$ </td><td> $. 8 7 4 \pm . 0 3$ </td><td> $. 9 1 6 \pm . 0 2$ </td><td></td><td> $. 7 7 3 \pm . 0 4$ </td></tr><tr><td>DR-net-IMUNet</td><td> $. 8 1 4 \pm . 0 3$ </td><td></td><td> $. 5 2 6 \pm . 0 5$ </td><td> $. 8 4 1 \pm . 0 4$ </td><td> $. 8 9 5 \pm . 0 3$ </td><td></td><td> $. 8 0 9 \pm . 0 4$ </td></tr><tr><td>noisy</td><td> $. 8 7 1 \pm . 0 3$ </td><td></td><td> $. 6 4 2 \pm . 0 6$ </td><td> $. 8 9 1 \pm . 0 2$ </td><td> $. 9 3 9 \pm . 0 2$ </td><td></td><td> $. 9 4 1 \pm . 0 2$ </td></tr><tr><td>clean</td><td> $. 8 4 6 \pm . 0 3$ </td><td></td><td> $. 6 2 7 \pm . 0 5$ </td><td> $. 8 0 7 \pm . 0 4$ </td><td></td><td> $. 9 4 6 \pm . 0 1$ </td><td> $. 8 4 7 \pm . 0 3$ </td></tr><tr><td>Specificity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNet-Mamba1-3B</td><td> $. 9 0 2 \pm . 0 1$ </td><td></td><td> $. 9 6 4 \pm . 0 1$ </td><td> $. 8 0 3 \pm . 0 2$ </td><td></td><td> $. 7 8 3 \pm . 0 3$ </td><td> $. 8 6 0 \pm . 0 2$ </td></tr><tr><td>DR-net-Mamba1-3B</td><td> $. 9 0 5 \pm . 0 1$ </td><td></td><td> $. 9 6 3 \pm . 0 1$ </td><td> $. 8 1 9 \pm . 0 2$ </td><td> $. 7 8 5 \pm . 0 3$ </td><td></td><td> $\overline { { . 8 5 1 \pm . 0 2 } }$ </td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td> $. 8 9 9 \pm . 0 1$ </td><td></td><td> $. 9 5 6 \pm . 0 1$ </td><td> $. 8 4 7 \pm . 0 2$ </td><td> $. 7 6 3 \pm . 0 3$ </td><td></td><td> $. 8 4 0 \pm . 0 2$ </td></tr><tr><td>DAE</td><td>.945 ± .01</td><td></td><td>.990 ± .00</td><td>.728 ± .02</td><td></td><td>.757 ± .03</td><td>.809 ± .02</td></tr><tr><td>DRNN</td><td> $. 8 7 3 \pm . 0 1$ </td><td></td><td>.969 ± .01</td><td>.730 ± .03</td><td>.825 ± .02</td><td></td><td> $. 6 9 7 \pm . 0 2$ </td></tr><tr><td>MECGE</td><td> $. 8 8 9 \pm . 0 2$ </td><td></td><td>.972 ± .01</td><td>.778 ± .02</td><td>.767 ± .03</td><td></td><td> $. 8 2 4 \pm . 0 2$ </td></tr><tr><td>UNet</td><td>.927 ± .01</td><td></td><td>.984 ± .01</td><td>.796 ± .02</td><td>.767 ± .02</td><td></td><td> $. 7 9 4 \pm . 0 2$ </td></tr><tr><td>DR-net-UNet</td><td>.901 ± .01</td><td></td><td>.968 ± .01</td><td>.826 ± .02</td><td> $. 8 0 9 \pm . 0 2$ </td><td></td><td> $. 7 8 9 \pm . 0 2$ </td></tr><tr><td>IMUNet</td><td> $9 1 6 \pm . 0 1$ </td><td></td><td> $. 9 7 2 \pm . 0 1$ </td><td> $7 9 0 \pm . 0 2$ </td><td> $. 7 7 4 \pm . 0 2$ </td><td></td><td> $\mathbf { 8 6 9 } \pm \mathbf { . 0 2 }$ </td></tr><tr><td>DR-net-IMUNet</td><td>.898 ± .02</td><td></td><td> $. 9 6 5 \pm . 0 1$ </td><td> $. 8 1 4 \pm . 0 2$ </td><td> $\underline { { 8 1 5 \pm . 0 2 } }$ </td><td></td><td> $. 8 2 9 \pm . 0 2$ </td></tr><tr><td>noisy</td><td> $. 8 1 9 \pm . 0 2$ </td><td></td><td>.915 ± .02</td><td> $. 6 8 7 \pm . 0 2$ </td><td> $. 7 2 4 \pm . 0 3$ </td><td></td><td> $. 6 4 2 \pm . 0 2$ </td></tr><tr><td>clean</td><td> $. 8 8 7 \pm . 0 2$ </td><td></td><td> $. 9 3 6 \pm . 0 1$ </td><td> $. 8 8 1 \pm . 0 1$ </td><td> $. 7 6 5 \pm . 0 3$ </td><td></td><td> $. 8 4 6 \pm . 0 2$ </td></tr><tr><td>F1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNet-Mamba1-3B</td><td> $. 7 6 5 \pm . 0 3$ </td><td></td><td> $. 5 9 1 \pm . 0 4$ </td><td> $. 7 0 5 \pm . 0 3$ </td><td></td><td> $. 8 3 9 \pm . 0 2$ </td><td> $. 7 1 3 \pm . 0 3$ </td></tr><tr><td>DR-net-Mamba1-3B</td><td>.776 ± .03</td><td></td><td> ${ \underline { { 5 9 6 \pm . 0 4 } } }$ </td><td> $. 7 1 4 \pm . 0 3$ </td><td></td><td> $. 8 3 9 \pm . 0 2$ </td><td> $. 7 1 0 \pm . 0 3$ </td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td> $7 5 8 \pm . 0 3$ </td><td></td><td> $\overline { { { . 5 9 6 \pm . 0 5 } } }$ </td><td> $\overline { { . 7 1 8 \pm . 0 3 } }$ </td><td></td><td> ${ \bf 8 4 3 \pm . 0 2 }$ </td><td> $. 7 1 2 \pm . 0 4$ </td></tr><tr><td>DAE</td><td> $. 7 0 3 \pm . 0 4$ </td><td></td><td> $. 4 3 6 \pm . 0 7$ </td><td> $. 6 5 9 \pm . 0 3$ </td><td></td><td> $. 8 1 2 \pm . 0 2$ </td><td> $. 6 5 3 \pm . 0 3$ </td></tr><tr><td>DRNN</td><td>.733 ± .03</td><td></td><td> $. 5 7 2 \pm . 0 5$ </td><td> $. 6 5 9 \pm . 0 3$ </td><td></td><td> $. 8 2 9 \pm . 0 2$ </td><td> $. 6 1 7 \pm . 0 3$ </td></tr><tr><td>MECGE</td><td>.735 ± .03</td><td></td><td> $. 5 3 9 \pm . 0 5$ </td><td> $. 6 7 7 \pm . 0 3$ </td><td></td><td> $. 8 3 3 \pm . 0 2$ </td><td> $. 6 8 0 \pm . 0 3$ </td></tr><tr><td>UNet</td><td> $. 7 7 0 \pm . 0 3$ </td><td></td><td> $. 4 8 0 \pm . 0 6$ </td><td> $. 7 0 4 \pm . 0 3$ </td><td> $. 8 3 2 \pm . 0 2$ </td><td></td><td> $. 6 6 8 \pm . 0 3$ </td></tr><tr><td>DR-net-UNet</td><td>.761 ± .03</td><td></td><td> $. 5 7 7 \pm . 0 5$ </td><td> $. 7 1 0 \pm . 0 3$ </td><td></td><td> $. 8 3 5 \pm . 0 2$ </td><td> $. 6 7 2 \pm . 0 3$ </td></tr><tr><td>IMUNet</td><td> $. 7 6 5 \pm . 0 3$ </td><td></td><td> $. 5 0 8 \pm . 0 5$ </td><td> $. 6 9 9 \pm . 0 3$ </td><td></td><td>.834 ± .02</td><td>.711 ± .04</td></tr><tr><td>DR-net-IMUNet</td><td>.755 ± .03</td><td></td><td> $. 5 9 4 \pm . 0 5$ </td><td> $. 7 0 3 \pm . 0 3$ </td><td></td><td> $\underline { { . 8 4 2 \pm . 0 2 } }$ </td><td> $. 6 9 3 \pm . 0 3$ </td></tr><tr><td>noisy</td><td> $. 7 0 4 \pm . 0 4$ </td><td></td><td> $. 5 7 4 \pm . 0 5$ </td><td> $. 6 3 1 \pm . 0 3$ </td><td></td><td> $. 8 2 3 \pm . 0 2$ </td><td> $. 6 1 9 \pm . 0 3$ </td></tr><tr><td>clean</td><td> $. 7 6 1 \pm . 0 3$ </td><td></td><td> $. 6 0 4 \pm . 0 5$ </td><td> $. 7 4 7 \pm . 0 3$ </td><td></td><td> $. 8 4 5 \pm . 0 2$ </td><td> $. 7 3 0 \pm . 0 3$ </td></tr></table>

## T. Per-Class Downstream Classification (ResNet1D-Wang)

## U. Diagnostic Abbreviations

The diagnostic class abbreviations used in Figures 7 and S11 are listed in Table S17.

Figure S11 shows the per-diagnostic-class AUROC breakdown using ResNet1D-Wang, analogous to Figure 7 in the main text. Panel (A) shows the AUROC delta when adding a Mamba bottleneck; panel (B) shows absolute AUROC of the lead-specific UNet-Mamba.

## V. STTC and CD diagnostic morphologies and The Mamba Bottleneck

Table S18 shows a comparison of STTC and CD diagnostic morphologies and their interaction with the Mamba bottleneck.

A. Downstream Classification (Inception1D)  
B. Downstream Classification (ResNet1D-Wang)  
![](images/f5d95f221747400735d473444fabcc3117ccf8848c1245d8c6d5b7c22ece1e12.jpg)

Fig. S9. Downstream diagnostic classification (macro and superdiagnostic AUROC) on PTB-XL under the lower noise setting, using Inception1D and ResNet1D-Wang.  
TABLE S16  
PER-SUPERDIAGNOSTIC-CLASS SENSITIVITY, SPECIFICITY, AND F1 ON PTB-XL (STRONG NOISE, RESNET1D-WANG). BOLD/UNDERLINED: BEST/SECOND-BEST.
<table><tr><td></td><td>CD</td><td></td><td>HYP</td><td>MI</td><td></td><td>NORM</td><td>STTC</td></tr><tr><td>Sensitivity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNet-Mamba1-3B</td><td>.784 ± .03</td><td></td><td>.515 ± .06</td><td>.898 ± .02</td><td></td><td>.942 ± .02</td><td>.786 ± .03</td></tr><tr><td>DR-net-Mamba1-3B</td><td>.784 ± .03</td><td></td><td>.522 ± .07</td><td>.870 ± .03</td><td></td><td>.931 ± .02</td><td>.794 ± .04</td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td>.780 ± .03</td><td></td><td>.541 ± .06</td><td>.841 ± .03</td><td>.961 ± .01</td><td></td><td>.803 ± .04</td></tr><tr><td>DAE</td><td>.667 ± .04</td><td></td><td>.347 ± .06</td><td>.894 ± .03</td><td></td><td>.915 ± .02</td><td>.756 ± .04</td></tr><tr><td>DRNN</td><td>.762 ± .03</td><td></td><td>.507 ± .06</td><td>.902 ± .02</td><td></td><td>.898 ± .03</td><td>.850 ± .03</td></tr><tr><td>MECGE</td><td>.774 ± .03</td><td></td><td>.440 ± .05</td><td>.896 ± .03</td><td></td><td>.942 ± .02</td><td>.780 ± .04</td></tr><tr><td>UNet</td><td>.760 ± .03</td><td></td><td>.362 ± .06</td><td>.906 ± .03</td><td></td><td>.931 ± .02</td><td>.822 ± .03</td></tr><tr><td>DR-net-UNet</td><td>.766 ± .04</td><td></td><td>.451 ± .06</td><td>.887 ± .04</td><td></td><td>.907 ± .02</td><td>.807 ± .04</td></tr><tr><td>IMUNet</td><td>.743 ± .03</td><td></td><td>.429 ± .06</td><td>.904 ± .02</td><td></td><td>.932 ± .02</td><td>.771 ± .04</td></tr><tr><td>DR-net-IMUNet</td><td>.764 ± .03</td><td></td><td>.478 ± .06</td><td>.874 ± .04</td><td></td><td>.929 ± .02</td><td>.786 ± .04</td></tr><tr><td>noisy</td><td>.808 ± .03</td><td></td><td>.627 ± .05</td><td>.896 ± .03</td><td></td><td>.939 ± .01</td><td>.915 ± .03</td></tr><tr><td>clean</td><td>.820 ± .03</td><td></td><td>.604 ± .06</td><td>.807 ± .04</td><td></td><td>.959 ± .02</td><td>.848 ± .04</td></tr><tr><td>Specificity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNet-Mamba1-3B</td><td>.924 ± .01</td><td></td><td>.968 ± .01</td><td>.794 ± .02</td><td></td><td>.788 ± .02</td><td>.885 ± .02</td></tr><tr><td>DR-net-Mamba1-3B</td><td>.924 ± .01</td><td></td><td>.964 ± .01</td><td>.812 ± .02</td><td></td><td>.793 ± .03</td><td>.875 ± .02</td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td>.925 ± .01</td><td></td><td>.960 ± .01</td><td>.845 ± .02</td><td></td><td>.767 ± .03</td><td>.874 ± .02</td></tr><tr><td>DAE</td><td>.938 ± .01</td><td></td><td>.988 ± .00</td><td>.699 ± .02</td><td></td><td>.766 ± .03</td><td>.855 ± .02</td></tr><tr><td>DRNN</td><td>.925 ± .01</td><td></td><td>.970 ± .01</td><td>.751 ± .02</td><td></td><td>.824 ± .02</td><td>.756 ± .03</td></tr><tr><td>MECGE</td><td>.905 ± .01</td><td></td><td>.974 ± .01</td><td>.757 ± .02</td><td></td><td>.768 ± .02</td><td>.869 ± .02</td></tr><tr><td>UNet</td><td>.935 ± .01</td><td></td><td>.989 ± .01</td><td>.766 ± .02</td><td></td><td>.788 ± .02</td><td>.840 ± .02</td></tr><tr><td>DR-net-UNet</td><td>.932 ± .01</td><td></td><td>.972 ± .01</td><td>.770 ± .02</td><td></td><td>.825 ± .02</td><td>.835 ± .02</td></tr><tr><td>IMUNet</td><td>.930 ± .01</td><td></td><td>.978 ± .01</td><td>.763 ± .02</td><td></td><td>.800 ± .02</td><td>.890 ± .01</td></tr><tr><td>DR-net-IMUNet</td><td>.921 ± .01</td><td></td><td>.971 ± .01</td><td>.792 ± .02</td><td></td><td>.821 ± .02</td><td>.872 ± .02</td></tr><tr><td>noisy</td><td>.879 ± .02</td><td></td><td></td><td></td><td></td><td></td><td>.693 ± .02</td></tr><tr><td>clean</td><td>.911 ± .01</td><td></td><td>.934 ± .01 .943 ± .01</td><td>.693 ± .02 .881 ± .01</td><td></td><td>.767 ± .02 .776 ± .02</td><td>.865 ± .02</td></tr><tr><td>F1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UNet-Mamba1-3B</td><td>.769 ± .03</td><td></td><td>.591 ± .06</td><td>.715 ± .03</td><td></td><td>.854 ± .02</td><td>.735 ± .03</td></tr><tr><td>DR-net-Mamba1-3B</td><td>.770 ± .03</td><td></td><td>.588 ± .06</td><td>.716 ± .03</td><td></td><td>.850 ± .02</td><td>.729 ± .03</td></tr><tr><td>UNet-Mamba1-3B (LS)</td><td>.769 ± .03</td><td></td><td>.593 ± .05</td><td>.730 ± .03</td><td></td><td>.854 ± .02</td><td>.734 ± .03</td></tr><tr><td>DAE</td><td>.712 ± .03</td><td></td><td>.486 ± .07</td><td>.641 ± .03</td><td></td><td>.829 ± .02</td><td>.687 ± .03</td></tr><tr><td>DRNN</td><td>.758 ± .03</td><td></td><td>.591 ± .05</td><td>.683 ± .03</td><td></td><td>.848 ± .02</td><td>.655 ± .03</td></tr><tr><td>MECGE</td><td>.740 ± .04</td><td></td><td>.544 ± .05</td><td>.684 ± .03</td><td></td><td>.845 ± .02</td><td>.715 ± .04</td></tr><tr><td>UNet</td><td>.768 ± .04</td><td></td><td>.503 ± .05</td><td>.696 ± .03</td><td></td><td>.848 ± .02</td><td>.711 ± .03</td></tr><tr><td>DR-net-UNet</td><td>.769 ± .03</td><td></td><td>.548 ± .06</td><td>.690 ± .03</td><td></td><td>.853 ± .02</td><td>.698 ± .04</td></tr><tr><td>IMUNet</td><td>.753 ± .03</td><td></td><td>.542 ± .05</td><td>.693 ± .03</td><td></td><td>.855 ± .01</td><td>.731 ± .04</td></tr><tr><td>DR-net-IMUNet</td><td>.754 ± .03</td><td></td><td>.568 ± .06</td><td>.701 ± .03</td><td></td><td>.863 ± .02</td><td>.722 ± .03</td></tr><tr><td>noisy</td><td>.731 ± .03</td><td></td><td>.600 ± .05</td><td>.638 ± .03</td><td></td><td>.843 ± .02</td><td>.641 ± .04</td></tr><tr><td>clean</td><td>.775 ± .03</td><td></td><td>.603 ± .05</td><td>.747 ± .03</td><td></td><td>.857 ± .02</td><td>.750 ± .04</td></tr></table>

![](images/ceff4c22aae9aa11b012547b7302e4ead0796de20693dc07a858b0d8d56bdb51.jpg)  
Fig. S10. Downstream diagnostic classification (macro and superdiagnostic AUROC) on PTB-XL using Inception1D, comparing denoised signals against clean and noisy baselines (strong noise setting).

TABLE S17 ECG FORM STATEMENT LABELS
<table><tr><td>Abbreviation</td><td>Description</td></tr><tr><td>NDT</td><td>non-diagnostic T abnormalities</td></tr><tr><td>NST</td><td>non-specific ST changes</td></tr><tr><td>DIG</td><td>digitalis-effect</td></tr><tr><td>LNGQT</td><td>long QT-interval</td></tr><tr><td>NORM</td><td>normal ECG</td></tr><tr><td>IMI</td><td>inferior myocardial infarction</td></tr><tr><td>ASMI</td><td>anteroseptal myocardial infarction</td></tr><tr><td>LVH</td><td>left ventricular hypertrophy</td></tr><tr><td>LAFB</td><td>left anterior fascicular block</td></tr><tr><td>ISC</td><td>non-specific ischemic</td></tr><tr><td>IRBBB</td><td>incomplete right bundle branch block</td></tr><tr><td>1AVB</td><td>first degree AV block</td></tr><tr><td>IVCD</td><td>non-specific intraventricular conduction disturbance (block)</td></tr><tr><td>ISCAL</td><td>ischemic in anterolateral leads</td></tr><tr><td>CRBBB</td><td>complete right bundle branch block</td></tr><tr><td>CLBBB</td><td>complete left bundle branch block</td></tr><tr><td>ILMI</td><td>inferolateral myocardial infarction</td></tr><tr><td>LAO/LAE</td><td>left atrial overload/enlargement</td></tr><tr><td>AMI</td><td>anterior myocardial infarction</td></tr><tr><td>ALMI</td><td>anterolateral myocardial infarction</td></tr><tr><td>ISCIN</td><td>ischemic in inferior leads</td></tr><tr><td>INJAS</td><td>subendocardial injury in anteroseptal leads</td></tr><tr><td>LMI</td><td>lateral myocardial infarction</td></tr><tr><td>ISCIL</td><td>ischemic in inferolateral leads</td></tr><tr><td>LPFB</td><td>left posterior fascicular block</td></tr><tr><td>ISCAS</td><td>ischemic in anteroseptal leads</td></tr><tr><td>INJAL</td><td>subendocardial injury in anterolateral leads</td></tr><tr><td>ISCLA</td><td>ischemic in lateral leads</td></tr><tr><td>RVH</td><td>right ventricular hypertrophy</td></tr><tr><td>ANEUR</td><td>ST-T changes compatible with ventricular aneurysm</td></tr><tr><td>RAO/RAE</td><td>right atrial overload/enlargement</td></tr><tr><td>EL</td><td>electrolytic disturbance or drug (former EDIS)</td></tr><tr><td>WPW</td><td>Wolff-Parkinson-White syndrome</td></tr><tr><td>ILBBB</td><td>incomplete left bundle branch block</td></tr><tr><td>IPLMI</td><td>inferoposterolateral myocardial infarction</td></tr><tr><td>ISCAN</td><td>ischemic in anterior leads</td></tr><tr><td>IPMI</td><td>inferoposterior myocardial infarction</td></tr><tr><td>SEHYP</td><td>septal hypertrophy</td></tr><tr><td>INJIN</td><td>subendocardial injury in inferior leads</td></tr><tr><td>INJLA</td><td>subendocardial injury in lateral leads</td></tr><tr><td>PMI</td><td>posterior myocardial infarction</td></tr><tr><td>3AVB</td><td>third degree AV block</td></tr><tr><td>INJIL</td><td>subendocardial injury in inferolateral leads</td></tr><tr><td>2AVB</td><td>second degree AV block</td></tr></table>

A. UNet Mamba1-3B vs. UNet [Difference]  
B. UNet Mamba1-3B LS vs. Noisy [Absolute]  
![](images/dd6cbe024f9367f427cbfaf9c960797b86d195970fda87b45ef64dffce39f723.jpg)  
Fig. S11. Per-diagnostic-class downstream AUROC (ResNet1D-Wang). (A) AUROC delta when adding a Mamba bottleneck to the UNet, all other design choices held constant. (B) Absolute AUROC of the lead-specific UNet-Mamba; color indicates improvement over the noisy baseline. In both panels, diagnostic classes with fewer than five segments are omitted. See Table S17 for abbreviations.

TABLE S18  
COMPARISON OF STTC AND CD DIAGNOSTIC MORPHOLOGIES AND THEIR INTERACTION WITH THE MAMBA BOTTLENECK.
<table><tr><td></td><td>STTC</td><td>CD</td></tr><tr><td>Temporal scale</td><td>Broad (200–400 ms)</td><td>Narrow (R-peak time 45–60 ms)</td></tr><tr><td>Bottleneck steps spanned</td><td>~4-7</td><td>~1</td></tr><tr><td>Frequency content</td><td>Low</td><td>High</td></tr><tr><td>Context needed</td><td>Global (baseline-relative)</td><td>Local (QRS-internal)</td></tr><tr><td>Mamba effect on denoising</td><td>Better noise/signal separation</td><td>Over-smoothing of sharp features</td></tr></table>