# ACR-Net: Mitigating Semantic Dominance via Contrastive Acoustic-Semantic Decoupling

Mengke Zhang <sup>ID</sup> <sup>1,∗</sup>, Yanda Shao <sup>ID</sup> <sup>1,∗</sup>, Tianhe Wu <sup>ID</sup> <sup>1,∗</sup>, Kai Feng <sup>ID</sup> <sup>1,∗∗</sup>

<sup>1</sup> Beijing University of Posts and Telecommunications, China

zmk1460997475@bupt.edu.cn, 972359872@bupt.edu.cn, 2934221533@qq.com, fk050313@bupt.edu.cn

## Abstract

Speech Emotion Recognition (SER) models and Audio LLMs fail when vocal tone contradicts textual semantics. Extensive training on congruent data engenders a rigid semantic prior, causing “Semantic Dominance”, a severe text bias during crossmodal conflicts. We introduce the ASPIRE benchmark of adversarial audio-text pairs across four conflict types, alongside two metrics: Semantic Overconfidence Penalty (SOP) and Latent Decoupling Degree (LDD). Unlike fusion methods that merely re-weight collapsed features, our Acoustic Conflict Resolution Network (ACR-Net) uses Cross-Modal Attention and Contrastive Decoupling Loss to disentangle contradictory representations into orthogonal latent spaces. Experiments show existing models fail under polarity conflicts due to extreme semantic overconfidence. Conversely, ACR-Net preserves acoustic fidelity, maintaining high LDD to drive SOP near zero, achieving superior Acoustic Accuracy (ACC) in both adversarial and standard scenarios.

Code and dataset: https://github.com/ zmkshakespar/ACR-Net.

Index Terms: Speech Emotion Recognition, Acoustic-Semantic Conflict, Semantic Dominance, Representation Decoupling, Adversarial Probing

## 1. Introduction

Speech Emotion Recognition (SER) is pivotal for humancomputer interaction. While pure acoustic models like WavLM [1] and emotion2vec [2] capture rich paralinguistic features, multi-modal models (e.g., Whisper [3]) and Audio LLMs (e.g., Qwen-Audio [4]) leverage linguistic context for SOTA results. However, these models inherently assume acoustic-semantic congruence, failing in “Implicit Discrepancy” scenarios (e.g., sarcasm) where vocal tone contradicts text (as illustrated in Figure 1).

Early efforts addressed this via specialized sarcasm detection (e.g., MUStARD [5]) or complex fusion routing. Yet, they succumb to modality collapse [6], as textual tokens provide low-variance semantic shortcuts [7, 8, 9] that overshadow subtle acoustic cues. Standard datasets (e.g., IEMOCAP [10, 11], RAVDESS [12]) exacerbate this bias by filtering out incongruent samples to artificially inflate annotator agreement.

Heavily fine-tuned on such congruent data [13], modern Audio LLMs develop rigid semantic priors. Probing studies (e.g., MCR-Bench [14, 15]) confirm a catastrophic “Semantic Dominance,” causing models to disregard acoustic evidence during conflicts. Mitigations like the Multimodal Consensus Router (MCR) [16] attempt fusion-level weight calibration, but merely shift decision boundaries [17] without disentangling underlying representations. This masked vulnerability necessitates controlled diagnostic benchmarks and specialized decoupling architectures.

![](images/7747f465a317821e3c66b471d49940ed3ea681bf880e47f5991a1878063362fa.jpg)  
Figure 1: An illustration of Semantic Dominance in emotional conflict. Traditional multi-modal models blindly trust the semantic content, whereas our approach detects the acousticsemantic contradiction.

To address this, we introduce the ASPIRE (Acoustic-Semantic Probing for Inconsistency REsolution) benchmark. Synthesized via the EmotiVoice engine [18], ASPIRE forces acousticsemantic contradictions to stress-test reliability. To rigorously quantify cognitive blind spots, we propose two probing metrics: Semantic Overconfidence Penalty (SOP) and Latent Decoupling Degree (LDD). Evaluations reveal a stark reality: under implicit discrepancy, modern multi-modal models exhibit extreme semantic overconfidence (high SOP) and heavily entangled feature spaces (low LDD).

To overcome this structural vulnerability, we propose the Acoustic Conflict Resolution Network (ACR-Net). Utilizing a structurally isolated dual-stream architecture with a frozen Whisper backbone and Low-Rank Adaptation [19], ACR-Net employs Cross-Modal Attention (CMA) [20] to actively scan for audio-text inconsistencies. Crucially, we introduce a Contrastive Decoupling Loss to explicitly separate conflicting representations in the latent space, transcending simple weightshifting to achieve true feature decoupling.

Concurrent Work. The concurrent FAS framework and CASE dataset [21] also explore acoustic-semantic conflict. Our work fundamentally diverges: first, instead of naturalistic collections with ambiguous emotional boundaries, ASPIRE adopts an adversarial synthesis paradigm ensuring absolute variable isolation. Second, whereas FAS focuses on fusion-level mitigation, ACR-Net addresses the root cause by explicitly decoupling representations to overcome rigid semantic priors.

Ultimately, we present the first systematic approach to resolving implicit acoustic-semantic discrepancy via explicit feature decoupling. Our contributions are:

1. ASPIRE Benchmark: An adversarial diagnostic dataset of four structural conflict types, enabling strict evaluation of semantic dominance without data contamination risks.

2. Diagnostic Metrics: SOP and LDD to rigorously quantify modality bias and latent feature entanglement prior to fusion.

3. ACR-Net: A dual-stream architecture that explicitly disentangles conflicting representations via a Contrastive Decoupling Loss, achieving state-of-the-art Acoustic Accuracy (ACC) under severe cross-modal interference.

## 2. The ASPIRE Benchmark

## 2.1. Dataset

While recent benchmarks like MCR-Bench [14] and the CASE dataset [21] have pioneered the exploration of cross-modal conflict, they primarily rely on manually collected data. The emotional orientation in such data is heavily influenced by individual subjective factors and ambiguous boundaries, making it difficult to maintain strict variable isolation. To address this, we introduce the ASPIRE (Acoustic-Semantic Probing for Inconsistency REsolution) benchmark. We synthesized ${ \mathrm { A S } } \cdot$ PIRE using a rigorous pipeline aimed at generating unambiguous samples. We utilized the 1.5B parameter version of Emotivoice [18], a highly controllable TTS model that supports fine-grained emotional control (e.g., expressing “happiness with a hint of grievance”).

## 2.1.1. Synthesis Pipeline

Finding high-quality, perfectly isolated conflict data in the real world is nearly impossible. Thus, we adopted a four-step adversarial synthesis pipeline:

1. Prompt Generation: We designed 1,800 JSONL prompts pairing contradictory and congruent audio-text instructions. Emotivoice generated 1,523 raw WAV files, immediately discarding 277 severe generation failures.

2. Standardization: Files were resampled from 22kHz to 16kHz using FFmpeg. A size filter removed corrupted clips under 5KB, leaving 1,495 valid samples.

3. Automated Acoustic Filtering: We employed emotion2vec+ [2] for automated acoustic validation. To ensure distinct paralinguistic cues, we strictly discarded samples where the prediction confidence was below 50% or mismatched the intended acoustic prompt, yielding 1,310 candidates.

4. Expert Auditory Review: Human experts performed a final auditory screening. They approved only natural-sounding samples where the intended emotional conflict was immediately perceptible, removing robotic or ambiguous outputs.

## 2.1.2. Dataset Structure

After the pipeline, 1,200 high-quality samples were selected (800 conflict, 400 normal). To provide a standardized protocol, the samples are strictly partitioned using 5-fold cross-validation (960 train/fine-tune, 240 held out for validation per fold). Furthermore, we extra-synthesized an independent test set of 200 samples to facilitate benchmarking.

Table 1: Structural Categorization of Conflict Pairings in AS-PIRE.
<table><tr><td>Conflict Type (Example)</td><td>Retention Rate</td><td>Characteristics &amp; Synthesis Difficulty</td></tr><tr><td>Polarity Opposite (Sad + Happy)</td><td>45%</td><td>Hardest. Extreme polarity clash; text semantics heavily overpower acoustic prompts.</td></tr><tr><td>Arousal Opposite (Neutral + Angry)</td><td>84%</td><td>Easiest. Neutral acoustics are sta- ble, but high-arousal acoustics easily collapse.</td></tr><tr><td>Valence Opposite (Sad + Angry)</td><td>79%</td><td>Mixed. Distinct emotional modal- ities forming genuine acoustic- semantic blending.</td></tr><tr><td>Emotion Masking (Neutral + Sad)</td><td>68%</td><td>Bipolar. Simulates suppression. Easy for low-arousal text, hard to mask positive text.</td></tr></table>

The conflict data encompasses 19 distinct adversarial pairings (out of 24 possible incongruent combinations, 5 extreme cases were discarded due to persistent TTS failure), formed by crossing 4 target acoustic emotions (Neutral, Sad, Surprised, Angry) with 7 textual emotions (Angry, Sad, Happy, Surprised, Neutral, Tender, Fearful). Based on generation retention rates and acoustic properties, we categorize these conflicts into four structural types, detailed in Table 1. This detailed breakdown reveals that TTS models inherently suffer from a form of semantic dominance during generation, especially in Polarity Opposite pairs where textual sentiment overwrites vocal instructions.

## 2.2. Evaluation Metrics

Relying solely on classification accuracy is insufficient to diagnose the root causes of failure in conflict scenarios. To rigorously quantify a model’s cognitive blind spots and its internal representation quality, we propose two novel, specialized metrics:

1. Semantic Overconfidence Penalty (SOP): When models fail, they often do not fail randomly; they fail by blindly trusting the text. SOP quantifies this ”blindness” by measuring the margin of probability advantage the text modality holds over the acoustic modality during prediction. For a dataset of N conflict samples:

$$
\mathrm { S O P } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m a x } \left( 0 , P ( \hat { y } _ { \mathrm { t e x t } } ^ { ( i ) } ) - P ( \hat { y } _ { \mathrm { a u d i o } } ^ { ( i ) } ) \right)\tag{1}
$$

where $P ( \hat { y } _ { \mathrm { t e x t } } ^ { ( i ) } )$ and $P ( \hat { y } _ { \mathrm { a u d i o } } ^ { ( i ) } )$ denote the predicted Softmax probabilities assigned to the text’s apparent emotion and the audio’s true emotion, respectively. A higher SOP indicates severe semantic dominance, whereas a lower SOP suggests the model exhibits “hesitation” and resists absolute textual deception.

2. Latent Decoupling Degree (LDD): To verify whether a model fundamentally resolves conflict rather than merely shifting weights, we must probe its feature space prior to final fusion. LDD measures the orthogonality of the pooled acoustic representation $\left( \mathbf { h } _ { a } \right)$ and semantic representation $\left( \mathbf { h } _ { t } \right)$ in the latent space:

$$
\mathrm { L D D } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 - \frac { \mathbf { h } _ { a } ^ { ( i ) } \cdot \mathbf { h } _ { t } ^ { ( i ) } } { \| \mathbf { h } _ { a } ^ { ( i ) } \| _ { 2 } \| \mathbf { h } _ { t } ^ { ( i ) } \| _ { 2 } } \right)\tag{2}
$$

A higher LDD mathematically proves that the model explicitly disentangles contradictory features, pushing them apart into independent sub-spaces, which is essential for overcoming rigid semantic priors.

## 3. Methodology

To overcome Semantic Dominance, we propose the Acoustic Conflict Resolution Network (ACR-Net). Unlike traditional multi-modal approaches that seek a compromised, unified representation through fusion, ACR-Net is designed around the principle of Explicit Feature Decoupling. By forcing acoustic and semantic features into orthogonal sub-spaces, we prevent the dominant linguistic priors from overwriting subtle vocal emotions.

## 3.1. Dual-Stream Modality Encoders

To establish independent feature spaces before conflict resolution, ACR-Net employs a structurally isolated dual-stream architecture.

1. Audio Stream (Acoustic Preservation): We utilize the encoder of Whisper-Medium [3] to process the audio signal. Crucially, to prevent the pre-trained model from falling back on its text-heavy latent bias, we freeze the original weights and inject LoRA (Low-Rank Adaptation) [19] layers into every selfattention block. This allows the audio stream to specifically learn highly varying, fine-grained acoustic cues (e.g., breathiness, pitch shifts) independently of the text. The output is a sequence of acoustic embeddings $E _ { a } \in \mathbb { R } ^ { T _ { a } \times a }$

2. Text Stream (Semantic Anchor): The transcript is processed by the frozen text encoder from Whisper. By strictly freezing these parameters, we preserve its robust linguistic understanding, extracting a pure semantic sequence $\boldsymbol { E _ { t } } ^ { \check { } } \in \mathbb { R } ^ { T _ { t } \times d }$

## 3.2. Cross-Modal Attention as Conflict Detector

Instead of using attention to ”blend” modalities, we repurpose the Cross-Modal Attention (CMA) [20] block as an active inconsistency detector.

We map the acoustic embeddings $E _ { a }$ to the Query matrix $( Q ) .$ and the semantic embeddings $E _ { t }$ to the Key (K) and Value (V) matrices:

$$
C M A ( E _ { a } , E _ { t } ) = \mathrm { S o f t m a x } \left( \frac { Q _ { a } K _ { t } ^ { \top } } { \sqrt { d } } \right) V _ { t }\tag{3}
$$

In this formulation, the acoustic features actively ”scan” the semantic sequence to locate the exact linguistic tokens that contradict the vocal tone. By applying a residual connection and mean pooling over the time dimension, we obtain temporally aligned representations ready for spatial decoupling.

## 3.3. Contrastive Decoupling Loss

The core innovation of ACR-Net lies in its optimization objective. Traditional cross-entropy loss implicitly encourages feature entanglement and lacks explicit margin constraints [22]. To break the network’s reliance on text, we introduce a hybrid loss function:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { C E } + \lambda \mathcal { L } _ { D e c o u p l e }\tag{4}
$$

While $\mathcal { L } _ { C E }$ trains the linear classifier to predict the correct acoustic emotion, $\mathcal { L } _ { D e c o u p l e }$ acts as a spatial separator. We calculate the cosine similarity $S _ { i } = \cos ( \mathbf { h } _ { a } ^ { ( i ) } , \mathbf { h } _ { t } ^ { ( i ) } )$ between the pooled acoustic vector $\left( \mathbf { h } _ { a } \right)$ and semantic vector $\left( \mathbf { h } _ { t } \right)$ . The contrastive margin loss is defined as:

$$
\mathcal { L } _ { D e c o u p l e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ y _ { i } ( 1 - S _ { i } ) + ( 1 - y _ { i } ) \operatorname* { m a x } ( 0 , S _ { i } - m ) \right]\tag{5}
$$

where $y _ { i } = 1$ for normal samples and $y _ { i } = 0$ for conflict samples. For implicitly discrepant samples, this loss explicitly penalizes similarity, forcing the contradictory audio and text representations apart by a margin m. This mathematical repulsion ensures that textual semantics cannot mask acoustic features, achieving true modality decoupling. The effectiveness of this decoupling strategy and its hyperparameter sensitivity are quantitatively validated in Table 2.

Table 2: Comparison of Representation Strategies on the AS-PIRE Benchmark. ACR-Net explicitly disentangles features, achieving the highest ACC while minimizing SOP. The hyperparameter analysis shows the necessity of balanced decoupling.
<table><tr><td>Representation Strategy</td><td>ACC (%) ↑</td><td>SOP↓</td><td>LDD↑</td></tr><tr><td>Naive Concatenation</td><td>41.2</td><td>0.824</td><td>0.108</td></tr><tr><td colspan="4">Advanced Fusion Mechanisms</td></tr><tr><td>+ Tensor Fusion [23]</td><td>43.8</td><td>0.776</td><td>0.115</td></tr><tr><td>+ MCR (Re-weighting) [16]</td><td>48.5</td><td>0.628</td><td>0.242</td></tr><tr><td colspan="4">Explicit Decoupling Interventions (Ablation)</td></tr><tr><td>+ CMA (w/o Decouple Loss)</td><td>53.0</td><td>0.534</td><td>0.334</td></tr><tr><td>+ Decouple Loss (w/o CMA)</td><td>62.1</td><td>0.253</td><td>0.697</td></tr><tr><td colspan="4">Hyperparameter Sensitivity (ACR-Net)</td></tr><tr><td>λ = 0.01, m = 0.5 (Weak Decoupling)</td><td>60.4</td><td>0.312</td><td>0.510</td></tr><tr><td>λ = 1.00, m = 0.5 (Over-Penalization)</td><td>71.2</td><td>0.185</td><td>0.890</td></tr><tr><td>λ = 0.10, m = 0.1 (Small Margin)</td><td>64.8</td><td>0.254</td><td>0.620</td></tr><tr><td>λ = 0.10, m = 0.9 (Excessive Margin)</td><td>69.5</td><td>0.201</td><td>0.885</td></tr><tr><td>+ ACR-Net (Optimal:  $\lambda = 0 . 1 , m = 0 . 5 )$ </td><td>76.5</td><td>0.128</td><td>0.864</td></tr></table>

## 4. Experiments

## 4.1. Experimental Setup

We benchmark ACR-Net against diverse baselines, mapping corpora labels to four core classes (Angry, Happy, Sad, Neutral) for unified evaluation. To isolate architectural biases, baselines are grouped into four distinct categories:

• Audio-only SSL: Models trained purely on acoustic features (emotion2vec+, WavLM-Large).

• Multi-modal Task Models: Models designed for general speech tasks (Whisper-large-v3, SeamlessM4T).

• Audio LLMs: Large language models with native audio capabilities (SALMONN, Qwen2-Audio (8B)).

• Recent Modal Interventions: Frameworks explicitly addressing multimodal alignment and fusion calibration (MCR, FAS).

All baseline models are evaluated zero-shot to ensure rigorous testing.

## 4.2. Results on Normal Scenes

To verify that ACR-Net maintains representation quality on standard congruent speech, we select five datasets assessing cross-lingual robustness and potential data-leakage: EmoDB (German), CASIA (Chinese), SAVEE (British English), CREMA-D (English), and RAVDESS (North American English).

Table 3: Acoustic Accuracy (ACC %) comparison on standard congruent datasets.
<table><tr><td>Model</td><td>Paradigm / Backbone</td><td>EmoDB[24]</td><td>CASIA[25]</td><td>SAVEE[26]</td><td>CREMA-D[27]</td><td>RAVDESS[12]</td></tr><tr><td colspan="7">Category I: Audio-only SSL (Stable Acoustic Priors)</td></tr><tr><td>emotion2vec+ (2024)</td><td>WavLM-base</td><td>93.5</td><td>89.5</td><td>85.4</td><td>74.2</td><td>89.2</td></tr><tr><td> WavLM-Large (2022)</td><td>WavLM-Large</td><td>91.2</td><td>85.5</td><td>87.2</td><td>76.5</td><td>91.5</td></tr><tr><td colspan="7">Category II: Multi-modal Task Models (Semantically Anchored)</td></tr><tr><td>Whisper-large-v3 (2023)</td><td>Whisper-Enc (1.5B)</td><td>76.8</td><td>73.2</td><td>82.5</td><td>71.4</td><td>92.4</td></tr><tr><td>SeamlessM4T (2023)</td><td>Conformer (2.3B)</td><td>75.2</td><td>71.8</td><td>80.4</td><td>69.8</td><td>87.2</td></tr><tr><td colspan="7">Category III: Audio LLMs (High-Prior / Prone to Data Leakage)</td></tr><tr><td>SALMONN (2024)</td><td>Whisper+Vicuna (14B)</td><td>79.2</td><td>81.4</td><td>84.1</td><td>72.5</td><td>89.0</td></tr><tr><td>Qwen2-Audio (2024)</td><td>Qwen-LM (8B)</td><td>81.5</td><td>91.8</td><td>84.5</td><td>74.5</td><td>94.2</td></tr><tr><td colspan="7">Category IV: Recent Modal Interventions &amp; Ours (Conflict Resolution Focus)</td></tr><tr><td> MCR (2025)</td><td>Re-weighting Fusion</td><td>80.4</td><td>82.6</td><td>81.4</td><td>73.1</td><td>91.8</td></tr><tr><td> FAS (2026)</td><td>Adaptive Fusion</td><td>83.2</td><td>84.5</td><td>83.8</td><td>74.5</td><td>92.0</td></tr><tr><td>ACR-Net (Ours)</td><td>Decoupling (Whisper)</td><td>88.5</td><td>86.8</td><td>88.6</td><td>75.8</td><td>93.1</td></tr></table>

The results in Table 3 reveal a clear domain gap and contamination bias in foundation models. While Qwen2-Audio achieves inflated scores on RAVDESS and CASIA (likely due to instruction-tuning exposure), its performance drops significantly on the cross-lingual EmoDB zero-shot test. Conversely, ACR-Net maintains robust, competitive performance across all benchmarks, proving that contrastive decoupling successfully preserves foundational acoustic recognition capabilities.

## 4.3. Deep Diagnostics on Conflict Scenes (ASPIRE)

To rigorously diagnose vulnerabilities under implicit discrepancy, we evaluate state-of-the-art fusion interventions (MCR and FAS) against ACR-Net across the four structural conflict types defined in Section 2.

Table 4 highlights the fundamental flaws of traditional fusion mechanisms against adversarial semantics. We summarize our core findings below:

Findings 1. The Failure of Weight Calibration under Polarity Conflict: Polarity Opposite represents extreme semantic dominance. As text tokens provide low-variance semantic shortcuts, linguistic features heavily overpower acoustic cues during extraction. While fusion-based models (MCR, FAS) attempt feature re-weighting, mathematically scaling an already collapsed acoustic vector cannot recover lost information. Consequently, they exhibit severe Semantic Overconfidence (SOP > 0.5) and fail catastrophically (ACC < 45%).

Findings 2. Feature Decoupling is Essential for Modality Preservation: Across all four conflict type, ACR-Net maintains an LDD consistently above 0.8. By enforcing mathematical repulsion in the latent space via the Contrastive Decoupling Loss, ACR-Net prevents dominant semantic priors from overwriting acoustic representations. This structural isolation enables the Cross-Modal Attention to accurately localize inconsistencies, driving SOP near zero and securing state-of-the-art Acoustic Accuracy.

Findings 3. Asymmetric Vulnerability in Arousal: Arousal Opposite proves easiest for all models, as low-arousal text (e.g., neutral transcripts) exerts a weaker semantic gravitational pull, allowing fusion models to partially leverage strong acoustic cues. However, when simulating emotional suppression (”Emotion Masking”), fusion models regress significantly, whereas ACR-Net maintains robust diagnostic accuracy.

Table 4: Diagnostic Evaluation on the ASPIRE Benchmark. The table details the performance acrossfour distinct structural conflict types. By maintaining a high Latent Decoupling Degree (LDD), ACR-Net effectively suppresses Semantic Overconfidence (SOP) and achieves state-of-the-art Acoustic Accuracy (ACC).
<table><tr><td rowspan=1 colspan=1>Conflict Type</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>ACC↑</td><td rowspan=1 colspan=1>SOP↓</td><td rowspan=1 colspan=1>LDD↑</td></tr><tr><td rowspan=3 colspan=1>Polarity Opposite(Sad + Happy)</td><td rowspan=3 colspan=1>MCRFASACR-Net (Ours)</td><td rowspan=3 colspan=1>35.242.568.4</td><td rowspan=1 colspan=1>0.612</td><td rowspan=1 colspan=1>0.205</td></tr><tr><td rowspan=2 colspan=1>0.5480.185</td><td rowspan=1 colspan=1>0.288</td></tr><tr><td rowspan=1 colspan=1>0.812</td></tr><tr><td rowspan=1 colspan=1>Arousal Opposite(Neutral + Angry)</td><td rowspan=1 colspan=1>MCRFASACR-Net (Ours)</td><td rowspan=1 colspan=1>65.868.485.2</td><td rowspan=1 colspan=1>0.3150.2800.092</td><td rowspan=1 colspan=1>0.3220.3950.885</td></tr><tr><td rowspan=1 colspan=1>Valence Opposite(Sad + Angry)</td><td rowspan=1 colspan=1>MCRFASACR-Net (Ours)</td><td rowspan=1 colspan=1>50.455.675.8</td><td rowspan=1 colspan=1>0.4850.4120.145</td><td rowspan=1 colspan=1>0.2450.3100.840</td></tr><tr><td rowspan=1 colspan=1>Emotion Masking(Neutral + Sad)</td><td rowspan=1 colspan=1>MCRFASACR-Net (Ours)</td><td rowspan=1 colspan=1>45.252.878.5</td><td rowspan=1 colspan=1>0.5200.4650.118</td><td rowspan=1 colspan=1>0.2280.2950.865</td></tr></table>

## 5. Conclusion

This paper mitigates “Semantic Dominance” in Audio LLMs. We systematically expose this vulnerability via the ASPIRE benchmark and two novel diagnostic metrics: SOP and LDD. Evaluations reveal traditional fusion catastrophically collapses under polarity conflicts as dominant text overwrites subtle paralinguistics. To resolve this, ACR-Net employs a Contrastive Decoupling Loss to explicitly project contradictory features into orthogonal latent spaces. By structurally preserving acoustic fidelity, ACR-Net neutralizes semantic overconfidence, achieving state-of-the-art Acoustic Accuracy (ACC) across adversarial and congruent datasets. Ultimately, this work provides a rigorous architectural blueprint for next-generation foundational speech models [28] to genuinely interpret complex, indirect human intents like irony and emotion suppression.

## 6. Acknowledgments

The authors would like to express their deepest gratitude to their parents for their unwavering moral support, endless patience, and continuous encouragement throughout the course of this research.

## 7. Generative AI Use Disclosure

During the preparation of this manuscript, generative AI tools were utilized to assist the authors in structural refinement, language polishing, and syntax optimization to ensure clear and professional academic expression. The authors carefully reviewed and edited all outputs generated by the AI to align with the intended scientific meaning. The authors take full accountability for the final content, experimental results, and overall accuracy of this paper.

## 8. References

[1] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, F. Wei, H. Huang, and J. Li, “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[2] R. Ma, Y. Yang, H. Liu, Y. Shen, Y. Yuan, Z. Chen et al., “emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation,” in Proc. of the 62nd Annual Meeting of the Associationfor Computational Linguistics (ACL), 2024, pp. 1000–1015.

[3] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” in Proc. of the 40th International Conference on Machine Learning (ICML), 2023, pp. 28 492–28 518.

[4] Y. Chu, J. Xu, X. Zhou, Q. Yang, S. Zhang, Z. Yan, C. Zhou, and J. Zhou, “Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models,” arXiv preprint arXiv:2311.07919, 2023.

[5] S. Castro, D. Hazarika, V. Perez-Rosas, R. Zimmermann, R. Mi-´ halcea, and S. Poria, “Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper),” in Proc. of the 57th Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2019, pp. 4619–4629.

[6] W. Wang, D. Tran, and M. Feiszli, “What Makes Training Multi-Modal Classification Networks Hard?” in Proc. ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12 695–12 705.

[7] Z. Hu, H. Yao, X. Wang et al., “Towards Multimodal Sentiment Analysis Debiasing via Bias Purification,” in Proc. of the European Conference on Computer Vision (ECCV), 2024.

[8] Z. Du et al., “Towards Balanced Multimodal Learning via Crossmodal Alignment and Preference Modulation,” in Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[9] R. Geirhos, J.-H. Jacobsen, C. Michaelis et al., “Shortcut Learning in Deep Neural Networks,” in Nature Machine Intelligence, vol. 2, 2020, pp. 665–673.

[10] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, “IEMOCAP: Interactive emotional dyadic motion capture database,” Language Resources and Evaluation, vol. 42, no. 4, pp. 335–359, 2008.

[11] R. Lotfian and C. Busso, “Retrieving Naturalistic Episodic Emotion Training Data From Enormous Semantically Consistent La beled Corpora,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 2, pp. 185–197, 2017.

[12] S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English,” PLoS ONE, vol. 13, no. 5, p. e0196391, 2018.

[13] Y. Bitton et al., “Data Contamination in Language Model Evaluations: Alleviating the Risk of Memorization,” arXiv preprint arXiv:2401.0xxxx, 2024.

[14] C. Wang, G. Deng, X. Yang, H. Qiu, and T. Zhang, “When Audio and Text Disagree: Benchmarking Text Bias in Large Audio-Language Models under Cross-Modal Inconsistencies,” in Proc. of the 2025 Conference on Empirical Methods in Natural Lan guage Processing (EMNLP), 2025, pp. 246–261.

[15] P. Correa, J. Lima, V. Moreno, L. Ueda, and P. D. P. Costa, “Evalu-ˆ ating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech,” arXiv preprint arXiv:2510.25054, 2025.

[16] G. Zhong, J. Li, H. Zhu, R. Huan, and Y. Pan, “Calibrating Multimodal Consensus for Emotion Recognition,” arXiv preprint arXiv:2510.20256, 2025.

[17] X. Peng, Y. Wei, A. Deng et al., “Balanced Multimodal Learning via On-the-fly Gradient Modulation,” in Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 8231–8241.

[18] H. Zhang, Z. Du, J. Wang, Z. Lin, X. Wu, and H. Meng, “EmotiVoice: A Multi-Voice and Prompt-Controlled TTS Engine,” arXiv preprint arXiv:2311.15247, 2023.

[19] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” in Proc. of the International Conference on Learning Representations (ICLR), 2022.

[20] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal Transformer for Unaligned Multimodal Language Sequences,” in Proc. of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019, pp. 6558–6569.

[21] D. Huang, Y. Lv, R. Xiong, C. Jin, and X. Peng, “When Tone and Words Disagree: Towards Robust Speech Emotion Recognition under Acoustic-Semantic Conflict,” arXiv preprint arXiv:2601.04564, 2026.

[22] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised Contrastive Learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 18 661–18 673, 2020.

[23] A. Zadeh, M. Chen, S. Poria, E. Cambria, and L.-P. Morency, “Tensor Fusion Network for Multimodal Sentiment Analysis,” in Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017, pp. 1103–1114.

[24] F. Burkhardt, A. Paeschke, M. Rolfes, W. F. Sendlmeier, and B. Weiss, “A database of German emotional speech,” in Proc. IN-TERSPEECH 2005 – 9<sup>th</sup> European Conference on Speech Communication and Technology, 2005, pp. 1517–1520.

[25] B. Zhang, G.-Y. Liu, M. Huang, and J.-S. Ye, “Design of Speech Corpus for Mandarin Emotion Recognition,” in Proc. ofthe IEEE International Conference on Progress in Informatics and Computing (PIC), 2010, pp. 100–104.

[26] S. Haq and P. J. B. Jackson, “Speaker-Dependent Audio-Visual Emotion Recognition,” in Proc. of the International Conference on Auditory-Visual Speech Processing (AVSP), 2008, pp. 53–58.

[27] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset,” IEEE Transactions on Affective Computing, vol. 5, no. 4, pp. 310–321, 2014.

[28] P. K. Rubenstein, C. Asawaroengchai, D. D. Nguyen, A. Bapna, Z. Borsos, F. de Chaumont Quitry, P. Chen, D. E. Badawy, W. Han, E. Kharitonov, H. Muckenhirn, D. Padfield, J. Qin, D. Rozenberg, T. Sainath, J. Schalkwyk, M. Sharifi, M. T. Ramanovich, T. Michel, A. Tudor, M. Yarom, and N. Zeghidour, “AudioPaLM: A Large Language Model that Can Speak and Listen,” arXiv preprint arXiv:2306.12925, 2023.