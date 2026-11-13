# MusicRWKV: A Multimodal RWKV Framework for Music Representation Learning

## Abstract

Recent music foundation models have achieved remarkable progress in music representation learning, yet most existing approaches rely on Transformer architectures whose quadratic attention complexity introduces substantial computational and memory overhead when modeling long-duration music recordings. In this work, we propose MusicRWKV, a multimodal music representation learning framework built upon the RWKV architecture. MusicRWKV jointly models waveform, spectrogram, and metadata information within a unified self-supervised learning framework. To improve efficiency for long musical sequences, we introduce an invertible compression module that reduces sequence length while preserving reconstruction capability. A FiLM-based conditioning strategy is employed to facilitate modality-aware representation learning without introducing expensive cross-attention operations. We further investigate the integration of Soft ROSA as a differentiable retrieval mechanism for neural-symbolic interaction. Experimental evaluation across a diverse collection of Music Information Retrieval (MIR) benchmarks demonstrates that the proposed RWKV-based architecture can be effectively adapted to heterogeneous music understanding tasks. A controlled pretraining ablation shows that Soft ROSA improves validation performance, while layer-wise telemetry analysis reveals a progressive transition from distributed retrieval to selective retrieval in deeper network layers. These results suggest that RWKV provides a promising architectural alternative to Transformer-based models for multimodal music understanding.

## 1 Introduction

Learning music representation has become a fundamental problem in Music Information Retrieval (MIR). Recent advances in self-supervised learning (SSL) have enabled the development of large-scale music foundation models that learn transferable representations from unlabeled audio data. Models such as MERT [1], MusicFM [2], and MuQ [3] have demonstrated strong generalization ability across a diverse range of downstream MIR tasks, including music tagging, genre classification, key estimation, emotion recognition, and instrument classification. These models are typically pretrained on large collections of music recordings and later evaluated through linear probing or lightweight task-specific adaptation.

Despite their success, existing music representation learning frameworks, including MERT, MusicFM, and MuQ, are predominantly based on Transformer architectures. While Transformers provide powerful long-range sequence modeling capabilities, their quadratic attention complexity introduces substantial computational overhead when modeling longduration music recordings. In addition, Transformer inference commonly relies on Key-Value (KV) caching, whose memory consumption grows with sequence length. For music recordings that may span several minutes, high-resolution audio tokenizers may produce extremely long token sequences. In such scenarios, the accumulation of KV caches can become a substantial memory bottleneck during both training and deployment.

Recently, a new generation of efficient foundation model architectures has emerged beyond the conventional Transformer paradigm. Among them, state-space and recurrent architectures have attracted increasing attention due to their ability to model long-context sequences with substantially lower computational complexity. The RWKV architecture [4] is one representative example of this trend, combining the parallelizable training characteristics of Transformers with the linear-time inference complexity of recurrent neural networks. Recent generations of RWKV have demonstrated competitive performance against Transformer-based large language models while offering superior memory efficiency and scalability for longcontext modeling. These advantages make RWKV particularly appealing for music understanding, where audio recordings often contain long temporal dependencies and rich hierarchical structures spanning rhythm, timbre, harmony, and semantics.

Furthermore, recent work has explored RWKV for symbolic music generation through the MIDI-RWKV [5] framework, demonstrating the effectiveness of RWKV for long-context symbolic music modeling. However, existing RWKV-based music studies mainly focus on symbolic sequence generation and controllable music creation rather than learning transferable audio representations for downstream music understanding tasks. To the best of our knowledge, the potential of RWKV as a backbone architecture for self-supervised music representation learning remains largely unexplored.

Another challenge in music representation learning lies in the heterogeneous nature of musical information. Music understanding often requires integrating information from multiple modalities, including raw audio waveforms, spectral representations, and semantic metadata. Existing Transformer-based music foundation models frequently employ cross-attention mechanisms to facilitate modality interaction. In contrast, RWKV does not inherently provide a cross-attention framework for multimodal fusion, making effective integration of heterogeneous musical representations an open problem.

To address these limitations, we propose MusicRWKV<sup>1</sup>, a multimodal RWKV framework for music representation learning. The proposed framework jointly models waveform tokens, spectrogram tokens, and metadata tokens within a unified RWKV architecture. To efficiently handle long musical sequences, we introduce an invertible compression module that reduces sequence length while preserving reconstruction capability. A FiLM-based [6] conditioning strategy is adopted to provide modality-aware representation learning without introducing expensive cross-attention operations. Furthermore, we investigate the effect of differentiable neural-symbolic interaction by incorporating the Soft ROSA [7] mechanism into RWKV blocks as an optional enhancement for representation learning.

To evaluate the applicability of the proposed architecture across diverse MIR scenarios, we conduct downstream experiments on music tagging, genre classification, key estimation, emotion recognition, instrument classification, and singing voice analysis. For each task, we train a task-specific MusicRWKV model from scratch using a shared backbone design, while adapting the input modalities and output heads to the corresponding task.

The main contributions of this work are summarized as follows:

1. To the best of our knowledge, MusicRWKV is the first multimodal RWKV framework specifically designed for audio-based music representation learning.

2. We demonstrate that RWKV can serve as an effective alternative to Transformer-based architectures for self-supervised music representation learning while retaining efficient long-sequence modeling capability.

3. We construct a unified multimodal pretraining framework that jointly models waveform, spectrogram, and metadata information for music understanding.

4. We provide an empirical analysis of Soft ROSA through ablation experiments and layerwise telemetry statistics.

## 2 Related Work

## 2.1 Music Representation Learning

Music representation learning aims to learn transferable semantic representations from largescale unlabeled music data and has become a fundamental research direction in Music Information Retrieval (MIR). Early studies commonly adopted self-supervised learning paradigms based on contrastive learning. Representative approaches such as CLMR [8] and MULE [9] demonstrated the effectiveness of contrastive objectives for learning reusable music embeddings and established the foundation of self-supervised music representation learning.

With the development of foundation models, recent research has increasingly focused on largescale self-supervised pretraining. Among recent music foundation models, MERT introduced a masked prediction framework combining acoustic and musical teachers to learn transferable music representations. MusicFM further explored a unified foundation model for music informatics tasks, while MuQ adopted Mel Residual Vector Quantization (Mel-RVQ) as a selfsupervised target and achieved state-of-the-art performance across multiple MIR benchmarks. More recently, multitask self-supervised learning frameworks such as MT2 [10] demonstrated that combining multiple pretext objectives can further improve representation quality and transferability across heterogeneous music understanding tasks.

Despite their differences in architecture design and training objectives, most contemporary music representation learning frameworks remain based on Transformer architecture. Consequently, the potential of alternative foundation model architectures for music representation learning remains insufficiently explored.

## 2.2 Multimodal Music Understanding

Music understanding is inherently multimodal. Beyond raw audio signals, musical information can be represented through multiple complementary modalities, including waveform representations, spectrogram representations, and semantic metadata. Each modality captures different aspects of musical content. Waveforms preserve fine-grained temporal information, spectrograms emphasize frequency-domain structures such as timbre and harmony, while metadata provides high-level semantic descriptions regarding genre, mood, instrumentation, and contextual information.

Recent studies have demonstrated that integrating multiple modalities can improve the quality of learned music representations. In addition, several works have explored structured representation learning from multiple views. For example, MARL [11] investigates disentangled music representations by separating shared and view-specific information across different audio views. Such studies highlight the importance of exploiting heterogeneous musical information for robust representation learning. However, existing multimodal music learning frameworks are predominantly designed around attention-based architectures, leaving multimodal learning in RWKV-based systems largely unexplored.

## 2.3 Efficient Foundation Model Architectures

Transformer architecture has become the dominant paradigm for foundation models due to their strong sequence modeling capability. However, the quadratic computational complexity of selfattention introduces substantial memory and computational overhead when processing longcontext sequences. This limitation becomes increasingly problematic in music understanding scenarios, where audio recordings often span several minutes and contain rich hierarchical structures.

To address this challenge, recent studies have explored more efficient foundation model architectures, including state-space models and recurrent architectures. These approaches aim to retain long-range dependency modeling capabilities while reducing computational complexity. Among them, RWKV has emerged as a promising foundation model architecture that combines Transformer-style parallel training with recurrent inference. By maintaining linear complexity with respect to sequence length during inference, RWKV provides a scalable solution for long-context sequence modeling while achieving competitive performance against Transformer-based architectures.

## 2.4 RWKV for Music Modeling

Although RWKV has attracted considerable attention in language modeling, its application to music remains relatively limited. Existing music-related RWKV research primarily focuses on symbolic music modeling. The most representative example is Adaptable Symbolic Music Infilling with MIDI-RWKV, which employs the RWKV-7 [12] architecture for long-context symbolic music infilling and controllable music generation. Experimental results demonstrate that RWKV can effectively model long-range symbolic musical structures and generate coherent musical sequences.

However, symbolic music generation and audio representation learning address fundamentally different objectives. Symbolic generation focuses on autoregressive modeling of discrete musical events, whereas music representation learning aims to learn transferable features that generalize across diverse downstream MIR tasks. To the best of our knowledge, existing RWKV-based music studies have not systematically investigated RWKV as a backbone architecture for multimodal self-supervised music representation learning. In contrast, our work explores RWKV for audio-based multimodal music representation learning and evaluates the applicability of the proposed backbone architecture across a broad collection of MIR benchmarks.

## 3 Methodology

Before describing each component in detail, we first provide an overview of the data flow in MusicRWKV. Waveform, spectrogram, and metadata tokens are processed by modalityspecific branches, fused into a shared representation space, and optimized through selfsupervised objectives. The following subsections describe each component in detail.

## 3.1 Overview

![](images/49ce9665975b80520ab2d5cbe7c00c932171da1c2604569f1bcc9b0bc533bf94.jpg)  
Figure 1: Illustration of MusicRWKV Framework

Figure 1 illustrates the overall architecture of MusicRWKV, a multimodal framework for music representation learning built upon the RWKV architecture. The framework jointly models three complementary modalities of musical information, including waveform tokens, spectrogram tokens, and semantic metadata.

To efficiently process long-duration music recordings, waveform and spectrogram token sequences are first passed through an invertible audio compression module that reduces sequence length while preserving reconstruction capability. This design reduces the effective sequence length presented to the RWKV encoders, which is particularly important for longduration music recordings. Metadata tokens are directly forwarded to the representation encoder without compression. The resulting modality-specific token sequences are then processed by independent RWKV encoders to capture temporal and semantic dependencies within each modality.

Since RWKV does not naturally provide a cross-attention mechanism for multimodal interaction, we adopt a FiLM-based conditioning strategy to perform modality-aware feature modulation. The conditioned representations from different modalities are subsequently fused into a unified joint representation space through multimodal fusion. During pretraining, the shared representation is optimized through multimodal self-supervised objectives. For downstream evaluation, the same backbone design is adapted to each task and trained from scratch using task-specific input modalities and prediction heads.

In addition, we investigate an enhanced variant of MusicRWKV by incorporating Soft ROSA into the RWKV encoders. Soft ROSA introduces differentiable neural-symbolic interactions through structured retrieval and aggregation over historical candidate representations. From a complementary perspective, Soft ROSA can be viewed as a router-free expert selection mechanism, where a subset of highly relevant candidates is selectively emphasized without requiring explicit routing networks. We compare the standard MusicRWKV and the Soft ROSA enhanced variant through a controlled pretraining ablation and analyze the retrieval behavior of Soft ROSA using layer-wise telemetry statistics.

Formally, given waveform, spectrogram, and metadata token sequences

$$
X = \{ X _ { w } , X _ { s } , X _ { m } \}
$$

MusicRWKV learns a unified representation

$$
H = f _ { \mathrm { M u s i c R W K V } } ( X _ { w } , X _ { s } , X _ { m } )
$$

where � is optimized through multimodal self-supervised objectives. For downstream tasks, the MusicRWKV backbone architecture is reused with task-specific modality configurations and output heads, while model parameters are initialized and trained from scratch.

## 3.2 Multimodal Input Representation

Music contains heterogeneous information spanning acoustic, spectral, and semantic levels. To capture complementary musical characteristics, MusicRWKV jointly models waveform, spectrogram, and metadata modalities. These modalities are converted into token sequences and subsequently processed by modality-specific branches.

For waveform modeling, raw audio recordings are tokenized using the Jukebox [13] tokenizer, which converts continuous audio signals into discrete codec tokens while preserving finegrained temporal structures. To provide complementary spectral information, we further extract spectrogram tokens using the MuQ Mel-RVQ tokenizer. Compared with waveform tokens, Mel-RVQ tokens emphasize timbral and harmonic characteristics that are important for music understanding tasks.

In addition to acoustic representations, we incorporate semantic metadata including genre, artist, and lyrics. These metadata tokens provide high-level contextual information that may not be directly observable from audio signals alone.

Among the three modalities, waveform and spectrogram sequences are typically much longer than metadata sequences. To improve computational efficiency, waveform and spectrogram tokens are first passed through an invertible audio compression module before representation learning, while metadata tokens are directly forwarded to the corresponding encoder. The compression module groups adjacent tokens into fixed-length segments and projects them into a latent representation, while an inverse projection module reconstructs the original token sequence during training. The resulting compressed and uncompressed token sequences serve as inputs to the MusicRWKV architecture.

## 3.3 Architecture

Given the multimodal inputs $X = \{ X _ { w } , X _ { s } , X _ { m } \}$ , MusicRWKV learns modality-aware music representations through compression, RWKV encoding, conditioning, and multimodal fusion, as illustrated in Figure 1.

Since waveform and spectrogram token sequences are considerably longer than metadata sequences, directly processing them may introduce unnecessary computational overhead. To improve efficiency, waveform and spectrogram tokens are first passed through an invertible audio compression module, while metadata tokens are directly forwarded to the corresponding encoder. Let

$$
Z _ { w } = C ( X _ { w } ) \qquad Z _ { \mathit { s } } = C ( X _ { \mathit { s } } )
$$

where �(∙) denotes the compression function. The compressed representations preserve essential musical information while reducing sequence length for subsequent representation learning.

The resulting modality-specific token sequences are then processed by independent RWKV encoders. Unlike Transformer architectures that rely on self-attention, RWKV combines Transformer-style parallel training with recurrent inference, enabling efficient long-context modeling with linear inference complexity. Such characteristics are particularly desirable for music understanding, where recordings often contain long temporal dependencies spanning rhythm, timbre, harmony, and structural information.

Let $f _ { \mathrm { r w k v } }$ denote a modality-specific RWKV encoder. The encoded representations can be written as

$$
\begin{array} { r } { H _ { w } = f _ { \mathrm { r w k v } } ( Z _ { w } ) } \\ { H _ { s } = f _ { \mathrm { r w k v } } ( Z _ { s } ) } \\ { H _ { m } = f _ { \mathrm { r w k v } } ( X _ { m } ) } \end{array}
$$

Since RWKV does not naturally provide a cross-attention mechanism for multimodal interaction, we employ a FiLM-based conditioning strategy to inject modality-specific information into the learned representations. Following the FiLM formulation, feature modulation is performed through learnable affine transformations,

$$
F i L M ( H ) = \gamma \odot H + \beta
$$

where $\gamma$ and $\beta$ are conditioning parameters determined by the corresponding modality. The conditioned representations are subsequently fused to obtain a unified music representation,

$$
H _ { j o i n t } = f u s i o n ( H _ { w } , H _ { s } , H _ { m } )
$$

where ������ denotes the multimodal fusion operation. In our implementation, the fusion operation is realized by concatenating modality-specific representations followed by a shared RWKV encoder.

## 3.4 Soft ROSA Enhanced Variant

To further investigate the effect of neural-symbolic interaction in music representation learning, we introduce an enhanced variant of MusicRWKV by incorporating the Soft ROSA mechanism into modality-specific RWKV encoders. Unlike the standard MusicRWKV framework, the enhanced variant augments representation learning through differentiable retrieval and aggregation over historical candidate representations while preserving end-to-end differentiability.

Soft ROSA operates on intermediate hidden representations. Given query, key, and value representations �, �, and �, the Soft ROSA module performs structured candidate matching and aggregation to produce enriched representations

$$
H ^ { \prime } = S o f t R O S A ( Q , K , V )
$$

which are subsequently propagated through the remaining RWKV layers.

From a representation learning perspective, Soft ROSA can be interpreted as a differentiable neural-symbolic interaction mechanism that enables hidden representations to selectively retrieve information from relevant historical contexts. Such interactions provide an additional structured retrieval process beyond the recurrent dynamics of RWKV and may facilitate the modeling of high-level musical concepts and long-range dependencies.

From a complementary perspective, Soft ROSA can also be viewed as a router-free expert selection mechanism. Instead of dispatching tokens through explicit routing networks, the model progressively emphasizes a subset of highly relevant historical candidates through differentiable matching and weighting operations.

In this work, we employ Soft ROSA as an optional enhancement to MusicRWKV and investigate its influence on multimodal pretraining through a controlled ablation study.

## 3.5 Training Objective

MusicRWKV is trained through multimodal self-supervised objectives. Given waveform, spectrogram, and metadata inputs, the model learns modality-specific and joint representations through modality-aware encoding and fusion.

The primary training objective consists of modality-level cross entropy losses over waveform, spectrogram, and metadata representations,

$$
L _ { m a i n } = L _ { w a v e } + L _ { s p e c } + L _ { m e t a }
$$

To mitigate information loss introduced by compression and modality fusion, we introduce a lightweight reconstruction regularization term. Specifically, the reconstructed waveform and spectrogram representations are encouraged to remain close to their original inputs,

$$
L _ { r e c o n } = \left\| X _ { w } - \hat { X } _ { w } \right\| _ { 2 } ^ { 2 } + \left\| X _ { s } - \hat { X } _ { s } \right\| _ { 2 } ^ { 2 }
$$

Unlike modality-level prediction objectives, the reconstruction loss does not serve as an independent learning target. Instead, it acts as an information-preserving regularization that encourages reconstructed representations to remain close to their original embedded representations.

The final training objective is defined as

$$
L = L _ { m a i n } + \lambda L _ { r e c o n }
$$

where � controls the contribution of the reconstruction regularization term.

## 4 Experiments

## 4.1 Experimental Setup

## Pretraining Configuration

MusicRWKV was pretrained on approximately 25,000 publicly accessible full-length music recordings collected from Internet sources. To encourage stylistic diversity, the corpus was assembled with approximately balanced genre coverage whenever possible. The dataset was partitioned into training and validation subsets using an 80/20 genre-stratified split. To prevent evaluation leakage, recordings overlapping with any downstream benchmark datasets were excluded from the pretraining corpus.

Following the architecture described in Section 3, raw audio recordings were converted into discrete token sequences using the Jukebox tokenizer and the MuQ Mel-RVQ tokenizer. Waveform tokens preserve fine-grained temporal information, whereas spectrogram tokens provide complementary spectral characteristics such as timbre and harmonic structure. Metadata information, including artist, genre, and lyrics when available, was converted into textual token sequences and jointly modeled with audio representations. The maximum sequence lengths were set to 20,000 waveform tokens, 10,000 spectrogram tokens, and 500 metadata tokens. Such sequence lengths highlight the long-context nature of multimodal music modeling and motivate the use of RWKV as the backbone architecture.

The model was pretrained using the multimodal self-supervised objectives described in Section 3.4, consisting of waveform, spectrogram, and metadata prediction losses together with a lightweight reconstruction regularization term. Unless otherwise specified, both MusicRWKV and the Soft ROSA enhanced variant were trained under identical optimization settings, with the inclusion of the Soft ROSA module being the only architectural difference between the two models. Detailed optimization settings are provided in Appendix C.

Table 1. Architecture Configuration of the Pretraining MusicRWKV Model
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td></td><td></td></tr><tr><td>Hidden Dimension</td><td>1024</td></tr><tr><td>Waveform RWKV Layers</td><td>6</td></tr><tr><td>Spectrogram RWKV Layers</td><td>6</td></tr><tr><td>Metadata RWKV Layers</td><td>6</td></tr><tr><td>Fusion RWKV Layers</td><td>4</td></tr><tr><td>Compression Stride</td><td>10</td></tr><tr><td>Trainable Parameters</td><td>883M</td></tr></table>

The configuration in Table 1 corresponds to the multimodal pretraining model. The downstream models use smaller task-specific configurations with reduced model capacity and hidden dimensionality, while retaining the same general RWKV backbone design. The parameter count includes modality-specific embedding layers, invertible compression modules, RWKV encoders, FiLM conditioning layers, and output prediction heads. A substantial portion of the parameters originates from the modality-specific compression modules introduced for invertible sequence compression.

## Downstream Tasks

To evaluate the applicability of the proposed RWKV-based architecture, we conduct experiments on a diverse collection of downstream Music Information Retrieval (MIR) tasks covering music tagging, genre recognition, musical attribute prediction, and singing voice analysis.

Specifically, we evaluate MusicRWKV on the following benchmark datasets:

⚫ MagnaTagATune (MTT) [14] for automatic music tagging, where the objective is to predict semantic tags describing musical content.

⚫ GTZAN [15] for music genre classification.

⚫ GiantSteps Key [16] for musical key estimation.

⚫ EMO [17] for music emotion recognition.

⚫ NSynth [18] for instrument classification and pitch classification.

VocalSet [19] for singing voice analysis, including singer identification and vocal technique classification.

These benchmarks cover a broad range of MIR scenarios, ranging from high-level semantic understanding to low-level acoustic attribute prediction. Consequently, they provide a broad evaluation of the applicability of the proposed architecture across heterogeneous MIR tasks.

For downstream evaluation, we reuse the general MusicRWKV backbone design but train each task-specific model from scratch without loading the pretrained MusicRWKV parameters. Compared with the pretraining model, the downstream configurations use fewer parameters and lower-dimensional hidden representations to reduce overfitting on relatively small, labeled datasets. The input modalities and output heads are adapted to the characteristics of each task. Accordingly, the downstream experiments evaluate the applicability of the MusicRWKV architecture across heterogeneous MIR tasks rather than a conventional frozen-probing or finetuning protocol.

Detailed input modalities, hidden dimensions, backbone depths, parameter counts, and prediction heads for individual downstream tasks are provided in Appendix B.

## Baseline Models

We compare MusicRWKV against representative self-supervised music representation learning models spanning contrastive learning, masked prediction, multitask learning, and large-scale music foundation model paradigms. Specifically, we include CLMR and MULE as early selfsupervised music representation learning baselines, MERT, MusicFM, and MuQ as representative Transformer-based music foundation models, MT2 as a recent multitask selfsupervised learning framework, and AudioMAE++ [20] as a strong reconstruction-based audio foundation model. In addition, the effect of Soft ROSA is evaluated separately through a controlled pretraining ablation and layer-wise telemetry analysis.

Table 2. Representative Baseline Models and Their Learning Paradigms
<table><tr><td>Category</td><td>Models</td></tr><tr><td>Contrastive SSL</td><td>CLMR, MULE</td></tr><tr><td>Masked Prediction</td><td>MERT, MusicFM</td></tr><tr><td>Token Modeling</td><td>MuQ</td></tr><tr><td>Multitask SSL</td><td>MT2</td></tr><tr><td>Reconstruction-based SSL</td><td>AudioMAE++</td></tr></table>

## Evaluation Metrics

We follow the standard evaluation protocols adopted by each benchmark. Depending on the task, performance is measured using ROC-AUC, Average Precision (AP), Refined Accuracy, Accuracy, or $R ^ { 2 }$ . Higher values indicate better performance.

## 4.2 Main Results

Tables 3 and 4 summarize the downstream evaluation results of MusicRWKV on a diverse collection of MIR benchmarks. Overall, MusicRWKV achieves competitive performance across both general music understanding and fine-grained musical attribute prediction tasks, demonstrating the broad applicability of the proposed backbone design. Particularly strong performance is observed on music tagging average precision, musical key estimation, and vocal technique classification, indicating that task-specific variants of the proposed backbone can capture both high-level semantic information and fine-grained expressive musical characteristics.

Table 3. Performance Comparison on General MIR Benchmarks
<table><tr><td>Dataset</td><td colspan="2">MTT</td><td>GS</td><td>GTZAN</td><td colspan="2">EMO</td></tr><tr><td>Task</td><td colspan="2">tagging</td><td>key</td><td>genre</td><td colspan="2">emotion</td></tr><tr><td>Metrics</td><td>ROC</td><td>AP</td><td> $\mathtt { A c c } ^ { \mathrm { r e f i n e d } }$ </td><td>Acc</td><td>valence/R²</td><td>arousal/R²</td></tr><tr><td>CLMR</td><td>89.4</td><td>36.1†</td><td>14.9</td><td>68.6</td><td>45.8</td><td>67.8</td></tr><tr><td>MULE</td><td>91.4+</td><td>40.4</td><td>66.7</td><td>73.5</td><td>57.7</td><td>70.0</td></tr><tr><td>MERT</td><td>90.6</td><td>37.90</td><td>64.1</td><td>77.6</td><td>56.7</td><td>76.1</td></tr><tr><td>MERT2-30s</td><td>91.9‡</td><td>41.3</td><td>66.9</td><td>91.7</td><td>63.2</td><td>80.0</td></tr><tr><td>MERT2-FS</td><td>91.7</td><td>41.2</td><td>67.1</td><td>90.7</td><td>63.5</td><td>78.1</td></tr><tr><td>MusicFM</td><td>90.9</td><td>38.3</td><td>63.0</td><td>84.1</td><td>57.2</td><td>74.4</td></tr><tr><td>MuQ</td><td>90.5</td><td>38.5</td><td>63.2</td><td>83.8</td><td>58.3</td><td>76.4</td></tr><tr><td>MT2</td><td>88.4</td><td>38.8†</td><td></td><td></td><td></td><td></td></tr><tr><td>AudioMAE++</td><td> $9 1 . 2 ^ { \ddagger }$ </td><td> $3 9 . 5 ^ { \ddagger }$ </td><td> $6 1 . 7 ^ { \ddagger }$ </td><td> $8 0 . 3 ^ { \ddagger }$ </td><td> $5 9 . 0 ^ { \ddagger }$ </td><td> $7 5 . 7 \ddagger$ </td></tr><tr><td>MusicRWKV</td><td>88.6</td><td>46.1</td><td>70.3</td><td>86.6</td><td>62.1</td><td>75.1</td></tr></table>

Table 4. Performance on Fine-Grained Musical Attribute and Singing Voice Tasks
<table><tr><td>Dataset</td><td colspan="2">Nsynth</td><td colspan="2">VocalSet</td></tr><tr><td>Task</td><td>instrument</td><td>pitch</td><td>technique</td><td>singer</td></tr><tr><td>Metrics</td><td>Acc</td><td>Acc</td><td>Acc</td><td>Acc</td></tr><tr><td>CLMR</td><td> $6 7 . 9 ^ { \dagger }$ </td><td>47.0</td><td>58.1†</td><td>49.9</td></tr><tr><td>MULE</td><td>74.0</td><td>89.2</td><td>75.5</td><td>87.5</td></tr><tr><td>MERT</td><td>72.6</td><td>94.4</td><td>76.9+</td><td>87.1</td></tr><tr><td>MusicRWKV</td><td>73.3</td><td>87.4</td><td>89.1</td><td>83.0</td></tr></table>

† Results reported in the original publication.

‡ Results obtained from the public MARBLE benchmark.

MusicRWKV results are obtained by training task-specific models from scratch using the proposed backbone architecture. Baseline results are taken from original publications or the public MARBLE benchmark and may follow pretrained probing protocols. Therefore, these results should be interpreted as reference comparisons rather than strictly controlled evaluations under an identical training paradigm.

An important practical observation is that directly transferring the large pretrained MusicRWKV model to downstream benchmarks frequently resulted in substantial overfitting. This effect was particularly evident for datasets with limited labeled samples and appears to arise from the mismatch between the capacity and high-dimensional representation space of the pretrained model and the scale of downstream supervision. In preliminary experiments, reducing both model capacity and representation dimensionality resulted in more stable optimization and improved generalization. We therefore adopted smaller task-specific configurations and trained all downstream models from scratch.

Another notable finding is that downstream performance depends strongly on task-specific modality and representation choices. The evaluated tasks emphasize different musical factors, including harmonic structure, timbre, pitch, vocal expression, and performer identity. MusicRWKV performs particularly well on key estimation and vocal technique classification, whereas pitch classification and singer identification remain more challenging. These results suggest that MIR tasks may benefit from task-aware modality configurations and representation designs rather than a single fixed input pipeline.

## 4.3 Pretraining Ablation Study

To evaluate the impact of Soft ROSA on music representation learning, we conduct a pretraining ablation study by comparing the standard MusicRWKV framework with its Soft ROSA enhanced variant. Both models are trained under identical settings, with the only difference being the inclusion of the Soft ROSA module within the RWKV encoder.

Table 5. Effect of Soft ROSA on Pretraining Performance
<table><tr><td>Model</td><td>Params (M)</td><td>Peak Memory (GiB)</td><td>Time / Sample</td><td>Validation Accuracy (%)</td></tr><tr><td>MusicRWKV</td><td>883.00</td><td>64.90</td><td>3.32 s</td><td>93.30</td></tr><tr><td>MusicRWKV + Soft ROSA</td><td>952.66</td><td>69.80</td><td>9.49 s</td><td>94.82</td></tr></table>

Both models were trained using identical data splits, initialization seeds, and optimization settings. The reported validation accuracies correspond to the selected checkpoint with the lowest validation loss for each model.

As shown in Table 5, incorporating Soft ROSA improves validation accuracy from 93.3% to 94.8% under identical training settings. Since both models were trained using the same data split, random seed, and optimization configuration, the observed improvement is consistent with the inclusion of the Soft ROSA retrieval mechanism.

To better understand the source of this improvement, we further analyze the behavior of Soft ROSA through layer-wise telemetry statistics in the next section.

## 4.4 Empirical Analysis of Soft ROSA

## Layer-wise Telemetry

Telemetry statistics are computed from the normalized routing distributions produced by Soft ROSA and averaged across all batch elements, heads, and token positions. Top Probability denotes the maximum routing probability, Entropy corresponds to the normalized Shannon entropy of the routing distribution, and Candidate Mass denotes the total probability assigned to retrieved candidates (equivalently, one minus the sink probability). Table 6 summarizes the resulting layer-wise statistics.

$$
p _ { t o p } = \operatorname* { m a x } _ { i } p _ { i }
$$

$$
M _ { c } = \sum _ { i } p _ { i } = 1 - p _ { s i n k }
$$

$$
H _ { n o r m } = { \frac { - \sum _ { i } p _ { i } \log p _ { i } } { \log K } }
$$

where $\setminus p _ { i }$ denotes the normalized routing probability, $p _ { s i n k }$ denotes the sink-route probability, and � is the number of available routes. The entropy metric is normalized to the range [0,1].

Table 6. Layer-wise Telemetry Statistics of Soft ROSA
<table><tr><td>Layer</td><td>Top Probability</td><td>Entropy</td><td>Candidate Mass</td></tr><tr><td>0</td><td>0.31</td><td>0.73</td><td>0.86</td></tr><tr><td>1</td><td>0.38</td><td>0.65</td><td>0.82</td></tr><tr><td>2</td><td>0.47</td><td>0.56</td><td>0.79</td></tr><tr><td>3</td><td>0.58</td><td>0.45</td><td>0.73</td></tr><tr><td>4</td><td>0.71</td><td>0.31</td><td>0.66</td></tr><tr><td>5</td><td>0.84</td><td>0.18</td><td>0.58</td></tr></table>

As layer depth increases, the top probability consistently rises from 0.31 to 0.84, while routing entropy decreases from 0.73 to 0.18. In addition, candidate mass gradually decreases from 0.86 to 0.58. In our experiments, these trends indicate that retrieval distributions become progressively more concentrated in deeper layers. Rather than distributing attention broadly across many candidates, later layers increasingly focus on a smaller subset of highly relevant historical candidates.

## Retrieval Behavior Discussion

The telemetry analysis suggests that Soft ROSA exhibits a hierarchical retrieval behavior across network depth. Early layers maintain relatively distributed routing patterns, allowing representations to aggregate information from multiple historical candidates. As representations become progressively refined, deeper layers increasingly emphasize a smaller subset of highly relevant candidates.

This behavior is consistent with the interpretation of Soft ROSA as a differentiable retrieval mechanism operating over historical candidate representations. Rather than relying on explicit routing networks, Soft ROSA gradually concentrates probability mass onto selected candidates through differentiable matching and aggregation. Such behavior aligns with the router-free expert-selection perspective introduced in Section 3 and may contribute to the improved pretraining performance observed in the ablation study.

Additional qualitative visualization of the learned representation space using UMAP is provided in Appendix A.

## 5 Conclusion

In this work, we presented MusicRWKV, a multimodal RWKV framework for music representation learning. Unlike existing music foundation models that primarily rely on Transformer architectures, MusicRWKV adopts RWKV as the representation backbone and jointly models waveform, spectrogram, and metadata information within a unified multimodal framework. To facilitate multimodal interaction without introducing expensive cross-attention operations, we employed FiLM-based conditioning and investigated the integration of Soft ROSA as an optional differentiable retrieval mechanism.

Experimental results across a diverse collection of MIR benchmarks demonstrate that the proposed RWKV-based backbone can be effectively adapted to heterogeneous music understanding tasks. These downstream models share the same general backbone design but are trained from scratch using task-specific input modalities and output heads. In the pretraining study, incorporating Soft ROSA improves validation performance, while layer-wise telemetry analysis reveals a progressive transition from distributed retrieval to selective retrieval.

Several limitations remain. First, the retrieval mechanism introduced by Soft ROSA incurs additional computational and memory overhead, which may limit scalability for long-context music modeling. Second, the present pretraining study evaluates a single large-scale MusicRWKV configuration and does not systematically explore scaling behavior across different pretraining model sizes. Understanding how representation quality varies with model scale therefore remains an important direction for future work. Finally, our results suggest that representation suitability may vary across downstream tasks, indicating that a unified multimodal representation may not be optimal for all MIR scenarios.

Future work will investigate more efficient retrieval mechanisms for Soft ROSA, larger-scale pretraining strategies, and adaptive modality selection methods tailored to different music understanding tasks. We believe that these directions will further improve the scalability and cross-task applicability of RWKV-based music models.

## References

[1] Li, Y., “MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training”, arXiv eprints, Art. no. arXiv:2306.00107, 2023. doi:10.48550/arXiv.2306.00107.

[2] Won, M., Hung, Y.-N., and Le, D., “A Foundation Model for Music Informatics”, in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, Art. no. 254. doi:10.1109/ICASSP48485.2024.10448314.

[3] Zhu, H., “MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization”, IEEE Transactions on Audio, Speech and Language Processing, vol. 33, IEEE, pp. 3653–3664, 2025. doi:10.1109/TASLPRO.2025.3602320.

[4] Datta, A., “The Evolution of RWKV: Advancements in Efficient Language Modeling”, arXiv e-prints, Art. no. arXiv:2411.02795, 2024. doi:10.48550/arXiv.2411.02795.

[5] Zhou-Zheng, C. and Pasquier, P., “Adaptable Symbolic Music Infilling with MIDI-RWKV”, arXiv e-prints, Art. no. arXiv:2506.13001, 2025. doi:10.48550/arXiv.2506.13001.

[6] Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A., “FiLM: Visual Reasoning with a General Conditioning Layer”, arXiv e-prints, Art. no. arXiv:1709.07871, 2017. doi:10.48550/arXiv.1709.07871.

[7] W. Jie, “rosa soft: Differentiable Soft ROSA implementation,” GitHub repository. Available: https://github.com/wjie98/rosa\_soft.

[8] Spijkervet, J. and Burgoyne, J. A., “Contrastive Learning of Musical Representations”, arXiv e-prints, Art. no. arXiv:2103.09410, 2021. doi:10.48550/arXiv.2103.09410.

[9] McCallum, M. C., Korzeniowski, F., Oramas, S., Gouyon, F., and Ehmann, A. F., “Supervised and Unsupervised Learning of Audio Representations for Music Understanding”, arXiv e-prints, Art. no. arXiv:2210.03799, 2022. doi:10.48550/arXiv.2210.03799.

[10] Kong, Y., Lostanlen, V., Hennequin, R., Lagrange, M., and Meseguer-Brocal, G., “Multi-Class-Token Transformer for Multitask Self-supervised Music Information Retrieval”, arXiv e-prints, Art. no. arXiv:2507.12996, 2025. doi:10.48550/arXiv.2507.12996.

[11] Wilkins, J., Ding, S., Fuentes, M., and Bello, J. P., “Self-Supervised Multi-View Learning for Disentangled Music Audio Representations”, arXiv e-prints, Art. no. arXiv:2411.02711, 2024. doi:10.48550/arXiv.2411.02711.

[12] Peng, B., “RWKV-7 "Goose" with Expressive Dynamic State Evolution”, arXiv e-prints, Art. no. arXiv:2503.14456, 2025. doi:10.48550/arXiv.2503.14456.

[13] Dhariwal, P., Jun, H., Payne, C., Kim, J. W., Radford, A., and Sutskever, I., “Jukebox: A Generative Model for Music”, arXiv e-prints, Art. no. arXiv:2005.00341, 2020. doi:10.48550/arXiv.2005.00341.

[14] E. Law, K. West, M. I. Mandel, M. Bay, and J. S. Downie, "Evaluation of Algorithms Using Games: The Case of Music Tagging," in Proc. 10th Int. Soc. Music Inf. Retr. Conf., Kobe, Japan, 2009, pp. 387–392.

[15] Tzanetakis, G. and Cook, P., “Musical Genre Classification of Audio Signals”, IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, p. 293, 2002. doi:10.1109/TSA.2002.800560.

[16] P. Knees, Á. Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger, and M. Le Goff, "Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections," in Proc. 16th Int. Soc. Music Inf. Retr. Conf., 2015, pp. 387–392.

[17] M. Soleymani, M. N. Caro, E. M. Schmidt, C.-Y. Sha, and Y.-H. Yang, "1000 songs for emotional analysis of music," in Proc. 2nd ACM Int. Workshop Crowdsourcing Multimedia (CrowdMM' 13), New York, NY, USA, 2013, pp. 1–6. doi: 10.1145/2506364.2506365.

[18] Engel, J., “Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders”, arXiv e-prints, Art. no. arXiv:1704.01279, 2017. doi:10.48550/arXiv.1704.01279.

[19] J. Wilkins, Prem Seetharaman, Alison Wahl, and Bryan Pardo, “VocalSet: A Singing Voice Dataset”. Zenodo, Mar. 08, 2018. doi: 10.5281/zenodo.1193957.

[20] Yadav, S., Theodoridis, S., and Tan, Z.-H., “AudioMAE++: learning better masked audio representations with SwiGLU FFNs”, arXiv e-prints, Art. no. arXiv:2507.10464, 2025. doi:10.48550/arXiv.2507.10464.

## Appendix A

## Representation Space Visualization

To provide additional qualitative insight into the learned representation space, we visualize MusicRWKV embeddings using UMAP. Figure A1 shows UMAP visualizations of VocalSet representations colored by singer identity and vocal technique.

![](images/0814c222691f5a474b91cce3f15e2518f477302bcaee4ab4ee9b8748c6dfd679.jpg)  
(a) colored by singer identity

![](images/ce4ec6c3ca32038d6764960cdd20b3abf7fb19c4092bb226eff7260bf250f584.jpg)  
(b) colored by vocal technique  
Figure A1. UMAP visualization of VocalSet representations

The visualizations suggest slightly stronger local organization with respect to vocal techniques than singer identities, which is broadly consistent with the downstream classification results reported in Table 4. While the overall clustering structure remains diffuse, the embeddings reveal meaningful organization related to vocal expression.

## Appendix B

## Downstream Task Configurations

All downstream models reuse the general MusicRWKV backbone design but are trained from scratch without loading pretrained MusicRWKV parameters. Task-specific configurations differ in input modality selection, backbone depth, parameter count, and prediction heads according to the characteristics of each benchmark.

Table B1. Task-Specific Downstream Model Configurations
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Task</td><td rowspan="2">Input Modalities</td><td rowspan="2">Dim</td><td rowspan="2">Hidden RWKV Params Layers</td><td rowspan="2">(M)</td><td rowspan="2">Output Head Initialization</td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>MTT</td><td>Tagging</td><td>Waveform + Spectrogram</td><td>512</td><td>6</td><td>21.20</td><td>Multi-label Classification</td><td>From Scratch</td></tr><tr><td>GiantSteps</td><td>Key</td><td>Harmonic Mel Spectrogram</td><td>512</td><td>2</td><td>19.70</td><td></td><td>Classification From Scratch</td></tr><tr><td>GTZAN</td><td>Genre</td><td>Waveform + Spectrogram</td><td>512</td><td>6</td><td>21.33</td><td></td><td>Classification From Scratch</td></tr><tr><td>EMO</td><td>Emotion</td><td>Waveform + Spectrogram</td><td>512</td><td>6</td><td>21.36</td><td>Regression</td><td>From Scratch</td></tr><tr><td></td><td>NSynth Instrument </td><td>Spectrogram</td><td>512</td><td>6</td><td>21.54</td><td></td><td>Classification From Scratch</td></tr><tr><td>NSynth</td><td>Pitch</td><td>Harmonic CQT</td><td>512</td><td>6</td><td>21.54</td><td>Classification From Scratch</td><td></td></tr><tr><td></td><td>VocalSet Technique</td><td>CREPE Pitch Embeddings</td><td>512</td><td>6</td><td>20.35</td><td>Classification From Scratch</td><td></td></tr><tr><td>VocalSet</td><td>Singer</td><td>WavLM Speaker Embeddings</td><td>512</td><td>6</td><td>20.35</td><td>Classification From Scratch</td><td></td></tr></table>

## Appendix C

## Pretraining Optimization Details

Table C1. Pretraining Dataset and Optimization Settings
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Dataset Size</td><td>~25,000 recordings</td></tr><tr><td>Train / Validation Split</td><td>80/ 20</td></tr><tr><td>Split Strategy</td><td>Genre-stratified</td></tr><tr><td>Leakage Prevention</td><td>Downstream benchmark</td></tr><tr><td>Max Waveform Length</td><td>recordings excluded</td></tr><tr><td>Max Spectrogram Length</td><td>20,000</td></tr><tr><td>Max Metadata Length</td><td>10,000</td></tr><tr><td>Optimizer</td><td>500</td></tr><tr><td>Initial Learning Rate</td><td>Adam</td></tr><tr><td></td><td>1e-3</td></tr><tr><td>Batch Size</td><td>4</td></tr><tr><td>Epochs</td><td>400</td></tr><tr><td>Scheduler</td><td>ReduceLROnPlateau</td></tr><tr><td>Scheduler Factor</td><td>0.5</td></tr><tr><td>Scheduler Patience</td><td>3</td></tr><tr><td>Minimum Learning Rate</td><td>1e-6</td></tr><tr><td>Checkpoint Selection</td><td>Lowest validation loss</td></tr></table>