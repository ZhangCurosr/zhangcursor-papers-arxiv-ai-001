# REVE: EFFICIENT HALLUCINATION CORRECTION FOR LARGE AUDIO-LANGUAGE MODELS VIA REUSED ENCODER STATES

Hongjin Song, Jiasheng Kuang, Xinyu Yang, Qiuyu Fang, Ziyu Wu, Guowu Tan, Xiang Xie<sup>\*</sup>

Beijing Institute of Technology, Zhuhai, Guangdong, China

Corresponding author: Xiang Xie

## ABSTRACT

Large audio-language models may mention acoustic events that are absent from the input. A separate audio event detector can verify these mentions, but doing so requires a second audio encoder and a separate forward pass. We propose Reused Encoder States for Verifying Events (REVE), a lightweight method that uses states already computed by the target model. One readout summarizes class scores across audio frames, while another uses pooled states from four consecutive frame intervals. Class-aware score fusion combines their outputs to verify generated event mentions without encoding the audio again. On AudioSet, REVE removes 92.9% of label-unsupported mentions under a faithful-mention recall constraint. With fewer added parameters and no second audio-encoding pass, REVE achieves a reduction comparable to those of CED-Tiny and CED-Base. Its complete verification latency is about 1/18 of the CED-Base path. Results on controlled DESED mixtures and different target-model architectures further confirm the effectiveness of encoder-state reuse.

Index Terms— large audio-language models, hallucination, representation probing, inference-time correction

## 1. INTRODUCTION

Large audio-language models (LALMs) usually connect an audio encoder and a large language model through a modality projector. This design supports tasks such as audio question answering and open-ended audio captioning. For example, Qwen2-Audio [1] first uses a Whisper-based audio encoder [2] to extract acoustic representations. A modality projector then maps these representations into audio embeddings that are compatible with the input space of the language model. The language model combines the audio embeddings with a text instruction to generate an answer or caption. Despite their strong audio understanding, LALMs may still describe acoustic events that are absent from the input. For example, a model may mention thunder in a rain-only clip or music in a speech recording. Such descriptions are often plausible. It is therefore difficult to verify them using only the generated text or token probabilities. Prior work reduces audio hallucinations through audio-aware decoding [3], activation steering [4], preference alignment [5], and broader evaluation frameworks [6]. However, caption hallucinations often occur at the event-mention level. A caption-level score cannot locate the exact unsupported text span. Changes to model training or decoding may also be difficult to apply to a deployed model.

A direct post-hoc solution first extracts event mentions from the generated caption, maps them to classes in an audio event ontology, and then queries a separate detector such as CED [7]. This process provides class-level acoustic evidence and links each detection result to a specific caption span. It can therefore retain supported event mentions and remove unsupported ones. However, CED-Base introduces a high parameter cost and requires a second audio-encoder forward pass. CED-Tiny substantially reduces the detector size, but it still extracts acoustic features and encodes the waveform again. This additional path increases verification latency. Since the target model has already encoded the audio before generating the caption, we ask whether its existing encoder representations can provide event-presence evidence without running another audio model.

To this end, we propose Reused Encoder States for Verifying Events (REVE). It uses two readouts: one summarizes frame-level class scores from the pre-projector states, and the other reads out the means of four contiguous, equal-length frame intervals. A classaware calibrator fuses the two scores and produces the final eventpresence score for each ontology class. REVE maps event spans in the generated caption to ontology classes and uses these scores to verify the corresponding events.

Our contributions are threefold. First, we formulate post-hoc audio caption correction as ontology-aligned event verification and introduce a two-scale verifier that reuses encoder states. Second, we evaluate hallucination reduction under a faithful-mention recall constraint and report the added computation and storage costs of the verification module. Third, source-disjoint controlled DESED experiments and cross-model experiments show that REVE generalizes across acoustic domains and LALM architectures.

## 2. METHOD

Fig. 1 shows the full caption generation and event verification process. REVE reuses frame-level encoder states before the modality projector as acoustic evidence. It maps event spans in the generated caption to an audio event ontology and retains only mentions supported by the corresponding class scores.

## 2.1. Encoder State Reuse

Let � denote the input waveform and � the captioning instruction. Let �, �, and � denote the audio encoder, modality projector, and language-model decoder of the target LALM, respectively. The audio encoder first extracts frame-level states from a log-Mel spectrogram:

$$
H = E { \big ( } { \mathrm { L o g M e l } } ( x ) { \big ) } = [ h _ { 1 } , \dots , h _ { T } ] ^ { \top } \in \mathbb { R } ^ { T \times d } .\tag{1}
$$

where � is the number of encoder frames, � is the dimension of each encoder state, and $h _ { t } \in \mathbb { R } ^ { d }$ is the state at frame �. The modality projector then converts � into audio tokens that are compatible with the language model:

$$
Z = P ( { \cal H } ) , \qquad Z \in \mathbb { R } ^ { L \times d _ { G } } .\tag{2}
$$

![](images/66120854f553cfe3b1dafb446ebf81d80ccf4f01edf5fa89d3467f5ebf97f7fd.jpg)  
Fig. 1. Caption generation and event verification. (a) The audio encoder extracts frame-level states �. (b) The target LALM performs one standard multimodal decoding pass. (c) REVE combines frame-level score statistics with four temporal segment representations, fuses the two signals in a class-aware manner, and removes unsupported event mentions.

where $L$ is the number of audio tokens and $d _ { G }$ is the token dimension. The language model generates caption � from � and instruction �:

$$
C = G ( Z , q ) .\tag{3}
$$

As shown in Fig. 1(a)–(b), the original caption generation path and REVE share the same encoder states �. REVE can therefore extract acoustic evidence directly from the existing states without reloading or re-encoding the input audio.

## 2.2. Ontology Mapping

The language model uses free-form event descriptions, whereas the readouts use fixed classes. REVE derives matching forms from the official AudioSet class names by normalizing case and separators and removing parenthetical qualifiers. Whole-word matching identifies event spans in caption �:

$$
\begin{array} { r } { \mathcal { M } ( \boldsymbol { C } ) = \{ ( s _ { j } , c _ { j } ) \} _ { j = 1 } ^ { J } , } \end{array}\tag{4}
$$

where $s _ { j }$ is an event span and $c _ { j } \in \{ 1 , \ldots , K \}$ is its ontology class. A matched node maps directly to a readout class or to its nearest represented ancestor. Unmapped spans are not edited. Reference labels use the same mapping.

## 2.3. Two-Scale Event Scoring

Mean pooling gives a compact clip-level representation but may weaken evidence for short events. To capture both global score distributions and local temporal information, REVE combines framelevel score statistics with four contiguous, equal-length temporal segments. We first mean-pool all encoder frames to obtain the base clip representation:

$$
e = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } h _ { t } , \qquad e \in \mathbb { R } ^ { d } .\tag{5}
$$

A multi-label readout defines the base class logit $g _ { c } = w _ { c } ^ { \top } e + b _ { c }$ and the frame-level logit $a _ { t , c } = w _ { c } ^ { \top } h _ { t } + b _ { c }$ . For each mentioned class, we summarize the temporal distribution of its frame-level scores as

$$
q _ { c } = [ g _ { c } , \mathrm { s t d } _ { t } a _ { t , c } , \operatorname* { m a x } _ { t } a _ { t , c } ] ,\tag{6}
$$

and obtain $s _ { c } ^ { \mathrm { s t a t } } = \sigma ( \alpha _ { c } ^ { \top } q _ { c } + \beta _ { c } )$ . In parallel, we divide the $T$ encoder frames into four contiguous temporal segments $\mathcal { T } _ { l }$ and retain the mean representation of each segment:

$$
e _ { l } = | \mathcal { T } _ { l } | ^ { - 1 } \sum _ { t \in \mathcal { Z } _ { l } } h _ { t } , \quad l = 1 , \dots , 4 ,\tag{7}
$$

$$
s ^ { \mathrm { s e g } } = \sigma ( W _ { 4 } [ e _ { 1 } ; . . . ; e _ { 4 } ] + b _ { 4 } ) .\tag{8}
$$

Both readouts are trained on AudioSet-balanced-train with classbalanced binary cross-entropy. Let $\ell ( \cdot )$ denote the logit function and define $z _ { c } = [ \dot { \ell ( s _ { c } ^ { \mathrm { s t a t } } ) } , \ell ( s _ { c } ^ { \mathrm { s e } } ) \dot { ] } ^ { \intercal }$ . The final event-presence score is

$$
p _ { c } = \sigma \bigg ( \theta ^ { \top } z _ { c } + \delta _ { c } + \gamma _ { c } \frac { \mathbf { 1 } ^ { \top } z _ { c } } { 2 } + b \bigg ) .\tag{9}
$$

Here, � and � are shared across classes, while $\delta _ { c }$ and $\gamma _ { c }$ are the class-specific offset and slope. This regularized class-aware calibrator adds only 1,057 coefficients. All parameters of the target LALM remain frozen during training. The original readout that uses only one mean-pooled representation serves as the Mean Readout ablation.

## 2.4. Event Mention Correction

After ontology mapping, REVE scores only the classes mentioned in the caption. It produces score $p _ { c _ { \mathcal { I } } }$ for each event span $s _ { j }$ . This candidate-sparse implementation is equivalent to scoring all � classes. The threshold decision is defined as

$$
r _ { j } ( \tau ) = \mathbb { I } \left[ p _ { c _ { j } } \geq \tau \right] .\tag{10}
$$

where I[·] is the indicator function and � is the event-presence threshold. The event span is retained when $r _ { j } = 1$ and removed when $r _ { j } = 0$ . The corrected caption is

$$
C ^ { \prime } = \operatorname { E d i t } \left( C , \{ s _ { j } \mid r _ { j } ( \tau ) = 0 \} \right) .\tag{11}
$$

REVE edits only event spans with an ontology class score. All other text remains unchanged.

## 3. EXPERIMENTAL SETUP

## 3.1. Datasets and Models

AudioSet. We train the two readouts on AudioSet-balanced-train [8]. For evaluation, we divide 1,000 AudioSet-eval clips into nonoverlapping development and test sets of 500 clips each. Only the development set is used to select the class-aware calibrator and threshold. All methods use the same captions, event spans, ontology mappings, and reference labels.

DESED mixtures. We create 1,000 ten-second mixtures from isolated DESED foreground events [9]. They cover eight event families and contain one to three distinct events without background noise. Development and test sets contain 500 mixtures each and use disjoint source waveforms. The mixing records provide exact event-presence labels.

Models. Our main target is Qwen2-Audio-7B-Instruct [1]. For each ten-second clip, its encoder produces $H \in \mathbb { R } ^ { 7 5 0 \times 1 2 8 0 }$ and the projector produces 250 audio tokens. Captions use greedy decoding. We also test Qwen2.5-Omni-7B and SALMONN-13B [10, 11].

## 3.2. Baselines

We compare signals from generation confidence, internal targetmodel states, and external audio models. Token Confidence uses probabilities aligned with each event mention. Decoder Probe uses the last-layer language-model states. Mean Readout uses one meanpooled encoder representation. External baselines are CLAP [12], AST [13], PANN Cnn14 [14], and the CED-Tiny and CED-Base audio taggers [7]. These models reprocess the audio, whereas REVE reuses target-model states.

## 3.3. Evaluation Protocol

Let ℐ be the set of evaluated event mentions in the original captions. Let $y _ { i } ~ \in ~ \{ 0 , 1 \}$ indicate whether mention � is supported by the reference labels. For scoring method � and threshold �, let $r _ { i } ^ { ( m ) } ( \tau ) \in \{ 0 , 1 \}$ indicate whether the mention is retained after pruning. We define residual false-mention density and faithful-mention recall as

$$
D _ { m } ( \tau ) = \frac { \sum _ { i \in \mathcal { I } } ( 1 - y _ { i } ) r _ { i } ^ { ( m ) } ( \tau ) } { | \mathcal { I } | } ,\tag{12}
$$

$$
R _ { m } ( \tau ) = \frac { \sum _ { i \in \mathcal { T } } y _ { i } r _ { i } ^ { ( m ) } ( \tau ) } { \sum _ { i \in \mathcal { T } } y _ { i } } .\tag{13}
$$

$D _ { m }$ is the fraction of all original evaluated mentions that are false but remain after pruning. Lower values are better. $R _ { m }$ is the fraction of reference-supported mentions that remain after pruning.

We report relative hallucination reduction as

$$
Q _ { m } ( \tau ) = 1 - \frac { D _ { m } ( \tau ) } { D _ { \mathrm { n o \ p r u n i n g } } } .\tag{14}
$$

The reduction directly measures the fraction of original false mentions that are removed. However, it must be reported together with faithful-mention recall, because deleting every mention would otherwise give a 100% reduction. We therefore maximize the reduction under the development-set recall constraint below.

For each score-based method, we first obtain the set of development thresholds that satisfy the recall constraint:

$$
\mathcal { F } _ { m } = \{ \tau \in \mathcal { T } _ { m } : R _ { m } ^ { \mathrm { d e v } } ( \tau ) \geq 0 . 7 5 \} ,\tag{15}
$$

$$
\tau _ { m } ^ { * } = \operatorname* { m a x } { \mathcal { F } _ { m } } .\tag{16}
$$

Class-balanced training makes the acoustic outputs decision scores rather than calibrated probabilities. We therefore select � on the development set instead of fixing it at 0.5. REVE calibration and threshold selection use only grouped out-of-fold predictions from the development set. The complexity analysis includes only the added verification cost for a ten-second clip and excludes the shared target LALM. The data-processing, evaluation, and complexity-profiling scripts will be released on GitHub.

## 4. RESULTS

## 4.1. AudioSet Results

Table 1 reports results under the development-set recall constraint. REVE lowers residual density from 0.844 to 0.060 and removes 92.9% of false mentions at 0.756 recall. This is comparable to the 92.7% and 93.8% reductions of CED-Tiny and CED-Base, without a separate audio-encoder pass.

Table 1. AudioSet results at a development-set recall target of 0.75. REVE is shown in bold; underlining marks a best result only when it is achieved by another method.
<table><tr><td></td><td colspan="3">Residual</td></tr><tr><td>Method</td><td>Recall ↑</td><td></td><td>density ↓ Reduction ↑</td></tr><tr><td>No Correction</td><td>1.000</td><td>0.844</td><td>0.0%</td></tr><tr><td>Token Confidence</td><td>0.810</td><td>0.563</td><td>33.3%</td></tr><tr><td>Decoder Probe</td><td>0.747</td><td>0.207</td><td>75.5%</td></tr><tr><td>Mean Readout</td><td>0.774</td><td>0.103</td><td>87.8%</td></tr><tr><td>CLAP</td><td>0.785</td><td>0.301</td><td>64.3%</td></tr><tr><td>PANN Cnn14</td><td>0.791</td><td>0.077</td><td>90.9%</td></tr><tr><td>AST</td><td>0.802</td><td>0.068</td><td>91.9%</td></tr><tr><td>CED-Tiny</td><td>0.786</td><td>0.061</td><td>92.7%</td></tr><tr><td>CED-Base</td><td>0.758</td><td>0.052</td><td>93.8%</td></tr><tr><td>REVE</td><td>0.756</td><td>0.060</td><td>92.9%</td></tr></table>

Table 2 compares the added deployment cost with CED. CED-Tiny reduces CED-Base from 85.71M to 5.50M parameters and from 10.32 to 8.27 ms complete latency. Both still extract features and encode the waveform again. REVE adds 3.38M parameters. Its GPU computation takes 0.29 ms and its synchronized complete latency is 0.57 ms, about 1/18 of the CED-Base path.

(a) AudioSet efficiency landscape  
![](images/117034ade24828592c083b113529ad3ed302d8ac7aa47aa9cd12dae95e78d686.jpg)

![](images/510f8b3d54fced21aab5984e8534fafd0381083e601f4f0f9adab7a7c59f4993.jpg)  
Fig. 2. Efficiency and controlled-mixture results. (a) AudioSet reduction versus complete added latency and parameters (log scales). Markers are measurements; the smoothed surface is a visual guide. (b) AUROC across controlled DESED polyphony levels.

Table 2. Incremental cost for a 10-s clip on an NVIDIA L20Y in FP32. GPU forward uses CUDA-event timing; CED complete latency also includes feature extraction and transfer. REVE is shown in bold.
<table><tr><td></td><td colspan="3">GPU forward Complete path</td></tr><tr><td>Method</td><td>Parameters</td><td>(ms)</td><td>(ms)</td></tr><tr><td>CED-Tiny</td><td>5.50M</td><td>4.20</td><td>8.27</td></tr><tr><td>CED-Base</td><td>85.71M</td><td>5.46</td><td>10.32</td></tr><tr><td>REVE</td><td>3.38M</td><td>0.29</td><td>0.57</td></tr></table>

Fig. 2(a) shows that REVE reaches a 92.9% reduction with only 0.57 ms of added latency, while PANN, CED, and CLAP are slower.

## 4.2. Ablation Study

Table 3 shows that the two readouts provide complementary evidence. Shared fusion raises mention AUROC from about 0.91 to 0.931 and lowers residual density to 0.082. Class-aware calibration keeps AU-ROC nearly unchanged but lowers residual density to 0.060, which supports more accurate decisions with one threshold across classes.

Table 3. Component ablation on AudioSet. Residual density is measured at the development-selected recall target of 0.75. REVE is shown in bold; underlining marks a best result only when it is achieved by another variant.
<table><tr><td>Variant</td><td>Mention AUROC ↑ density ↓</td><td>Residual</td></tr><tr><td>Mean Readout</td><td>0.905</td><td>0.103</td></tr><tr><td>Statistics only</td><td>0.907</td><td>0.095</td></tr><tr><td>Segments only</td><td>0.909</td><td>0.095</td></tr><tr><td>Shared fusion</td><td>0.931</td><td>0.082</td></tr><tr><td>REVE</td><td>0.930</td><td>0.060</td></tr></table>

## 4.3. Controlled DESED Results

We use controlled DESED mixtures to avoid the missing-label ambiguity of AudioSet. The mixing records give exact presence labels for all eight event families.

Table 4 reports 0.959 AUROC and 0.915 AP for REVE. The strongest external detector, CED-Base, reaches 0.901/0.791.

Table 4. Source-presence diagnostic on controlled DESED mixtures. REVE is shown in bold.
<table><tr><td>Verification method</td><td>Micro-AUROC ↑ Micro-AP ↑</td><td></td></tr><tr><td>CLAP</td><td>0.821</td><td>0.644</td></tr><tr><td>PANN Cnn14</td><td>0.847</td><td>0.707</td></tr><tr><td>AST</td><td>0.887</td><td>0.761</td></tr><tr><td>CED-Tiny</td><td>0.896</td><td>0.779</td></tr><tr><td>CED-Base</td><td>0.901</td><td>0.791</td></tr><tr><td>REVE (DESED)</td><td>0.959</td><td>0.915</td></tr></table>

In Fig. 2(b), REVE obtains AUROCs of 0.996, 0.964, and 0.932 for one to three sources. It remains above both CED variants at every source count.

## 4.4. Cross-Model Results

We also evaluate REVE on Qwen2.5-Omni-7B and SALMONN-13B [10, 11] to test different target-model architectures.

Table 5. Complete REVE across target LALMs at the development recall target of 0.75.
<table><tr><td rowspan="2">Target LALM</td><td rowspan="2">Mention</td><td colspan="2">Density</td></tr><tr><td>AUROC ↑ before → after Reduction</td><td></td></tr><tr><td>Qwen2-Audio-7B</td><td>0.930</td><td>0.844 → 0.060</td><td>92.9%</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>0.865</td><td>0.797 → 0.158</td><td>80.1%</td></tr><tr><td>SALMONN-13B</td><td>0.841</td><td>0.804 → 0.207</td><td>74.3%</td></tr></table>

REVE reduces residual density to 0.158 on Qwen2.5-Omni and 0.207 on SALMONN, or by 80.1% and 74.3%, respectively.

## 5. CONCLUSION

REVE corrects LALM captions using encoder states already available after caption generation. Its statistical and segment readouts remove 92.9% of false mentions under the recall constraint, matching CED-Tiny and CED-Base. REVE uses fewer parameters, avoids another audio encoding pass, and adds about 1/18 of CED-Base’s complete verification latency. DESED and cross-model results confirm the effectiveness of encoder-state reuse.

## 6. REFERENCES

[1] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, C. Zhou, and J. Zhou, “Qwen2-Audio technical report,” arXiv preprint arXiv:2407.10759, 2024.

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023.

[3] T.-W. Hsu, K.-H. Lu, C.-H. Chiang, and H.-Y. Lee, “Reducing object hallucination in large audio-language models via audioaware decoding,” in Proc. IEEE ASRU, 2025.

[4] T.-E. Lin, K.-Y. Lee, and H.-Y. Lee, “Adaptive vector steering: A training-free, layer-wise intervention for hallucination mitigation in large audio and multimodal models,” arXiv preprint arXiv:2510.12851, 2025.

[5] Y. Chen, W. Zhu, X. Chen, Z. Wang, X. Li, P. Qiu, H. Wang, X. Dong, Y. Xiong, A. Schneider, Y. Nevmyvaka, and Y. Wang, “AHA: Aligning large audio-language models for reasoning hallucinations via counterfactual hard negatives,” in Findings of ACL, 2026, pp. 29 294–29 306.

[6] F. Zhao, Y. Chen, W. Lu, D. Zhang, X. Yue, and J. Wei, “Hallu-Audio: A comprehensive benchmark for hallucination detection in large audio-language models,” in Proc. ACL, 2026, pp. 38 797–38 816.

[7] H. Dinkel, Y. Wang, Z. Yan, J. Zhang, and Y. Wang, “CED: Consistent ensemble distillation for audio tagging,” in Proc. ICASSP, 2024, pp. 291–295.

[8] J. F. Gemmeke, D. P. W. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “AudioSet: An ontology and human-labeled dataset for audio events,” in Proc. ICASSP, 2017, pp. 776–780.

[9] N. Turpault, R. Serizel, S. Wisdom, H. Erdogan, J. R. Hershey, E. Fonseca, P. Seetharaman, and J. Salamon, “Sound event detection and separation: A benchmark on DESED synthetic soundscapes,” in Proc. ICASSP, 2021, pp. 840–844.

[10] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, B. Zhang, X. Wang, Y. Chu, and J. Lin, “Qwen2.5-Omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[11] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. Ma, and C. Zhang, “SALMONN: Towards generic hearing abilities for large language models,” in Proc. ICLR, 2024.

[12] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in Proc. ICASSP, 2023, pp. 1–5.

[13] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio spectrogram transformer,” in Proc. Interspeech, 2021, pp. 571–575.

[14] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “PANNs: Large-scale pretrained audio neural networks for audio pattern recognition,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 28, pp. 2880–2894, 2020.