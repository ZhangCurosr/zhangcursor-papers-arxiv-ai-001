# On the role of the tokenizer in ECG transformer models

Jiawei Li<sup>1</sup>, Fabio Bonassi<sup>1</sup>, Johan Sundstrom¨ <sup>1</sup>, Thomas B. Schon¨ <sup>1</sup>, Antonio H. Ribeiro<sup>1,2</sup>

<sup>1</sup> Uppsala University, Uppsala, Sweden Scilifelab, Uppsala, Sweden

## Abstract

Tokenization determines both the physiological content presented to an ECG Transformer and the sequence over which attention operates. We compare eight tokenization strategies across Transformer, Informer, Reformer, and FEDformer on the nine-label CPSC2018 classification task. The input projection and principal backbone capacity are controlled to isolate the effect of token construction. Median-beat and HeartLang tokenization achieve mean macro-AUCs of 0.893 and 0.889 across the four backbones, compared with 0.822 and 0.824 for point-wise and patch-wise tokenization. Pooling the two physiology-aware representations yields an 8.2% relative improvement in macro-AUC. They also reduce mean sequence length from 1,250 to 158 tokens and mean peak training memory from 5.21 to 0.27 GB. The results show that aligning tokens with ECG morphology can improve both predictive performance and memory efficiency without increasing backbone capacity. The source code is available on https://github.com/ LeeJarvis996/ecg\_tokenizer.

## 1. Introduction

Transformer architectures have achieved strong performance in general time-series modelling and are increasingly being adopted for electrocardiogram (ECG) analysis. Nevertheless, their performance on ECG classification often remains suboptimal, usually falling behind stateof-the-art convolutional neural networks and state-space models [1]. While previous work has primarily focused on improving attention mechanisms [2] or scaling model size [3], we argue that tokenization is also an important design choice. An ECG is a quasi-periodic, multi-lead signal whose diagnostic information is encoded in the morphology of the P–QRS–T complex, beat-to-beat rhythm, and spatial relationships among leads. Point-wise tokenization preserves the original sampling resolution but produces long sequences, whereas fixed-length patches may divide physiologically coherent waveforms across token boundaries. Token count also directly affects computational cost, with full self-attention scaling quadratically with sequence length.

In this study, we examine the role of tokenization in ECG Transformer models by evaluating eight strategies across four representative backbones, Transformer [4], Informer [5], Reformer [6], and FEDformer [7], on the CPSC2018 classification task. The evaluated representations span point-wise, lead-wise, fixed-length patch, stochastic segment, BIOT-inspired [8] segment, ECG-Byte [9], HeartLang heartbeat [10], and median-beat tokenization, while the input projection is unified and the principal backbone hyperparameters are held constant across experiments. This design allows us to investigate how token definition and sequence length influence predictive performance and computational efficiency.

Averaged across the four backbones, median-beat and HeartLang tokenization achieve macro-AUCs of 0.893 and 0.889, respectively. Pooling these two physiology-aware tokenizers gives a mean macro-AUC of 0.891. Conventional point-wise and patch-wise tokenization achieve corresponding means of 0.822 and 0.824, with a pooled mean of 0.823. The pooled improvement is 8.2% relative. Compared with the pooled point-wise and patch-wise group, the physiology-aware group reduces mean sequence length from 1,250 to 158 tokens and mean peak training VRAM from 5.21 to 0.27 GB, corresponding to reductions of 87.4% and 94.8%, respectively. Although our evaluation is limited to a single dataset, the results suggest that preserving physiologically meaningful structure is crucial and that a simple tokenizer choice can yield substantial classification performance and efficiency gains.

## 2. Methods

## 2.1. Tokenization Definition

For one ECG record, let $\mathbf { X } \in \mathbb { R } ^ { \mathrm { T } \times \mathrm { C } }$ denote T samples from C leads. A tokenizer τ transforms the signal into an ordered sequence of token units,

$$
\mathcal { U } _ { \tau } ( \mathbf { X } ) = ( \mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { \mathrm { L } } ) , \qquad \mathbf { u } _ { \mathrm { k } } \in \mathbb { R } ^ { \mathrm { P } } ,\tag{1}
$$

Table 1. Token units for a 10 s , 100 Hz, 12-lead ECG.
<table><tr><td>Tokenizer</td><td>L</td><td>P</td><td>Physiological meaning of one unit</td></tr><tr><td>Point-wise</td><td>1000</td><td>12</td><td>Simultaneous sample across all leads</td></tr><tr><td>Lead-wise</td><td>12</td><td>1000</td><td>Complete waveform from one lead</td></tr><tr><td>Patching</td><td>1500</td><td>16</td><td>Overlapping 0.16s segment from one lead</td></tr><tr><td>Stochastic</td><td>192</td><td>128</td><td>Resampled 0.8 s-1.6 s segment from one lead</td></tr><tr><td>BIOT-inspired</td><td>228</td><td>100</td><td>Fixed 1 s segment from one lead</td></tr><tr><td>ECG-Byte</td><td>1020</td><td>ID</td><td>BPE unit over quantized ECG sym- bols</td></tr><tr><td>HeartLang Median beat</td><td>256</td><td>96</td><td>One heartbeat from one lead</td></tr><tr><td></td><td>60</td><td>12</td><td>One relative sample of a represen- tative multi-lead beat</td></tr></table>

where $\mathbf { u } _ { \mathrm { k } }$ is the content of the k-th token unit, L is the sequence length, and P is its tokenizer-specific dimension. The construction of $\mathbf { u } _ { \mathrm { k } }$ determines its physiological meaning, while the ordered sequence of L units defines the axis along which the Transformer models token-to-token relationships.

Each token unit is converted to the common model dimension D by an input projection $\phi _ { \tau } .$ , and optional lead and positional information is then added. We define the resulting token embedding, i.e., the vector actually passed to the Transformer, as

$$
\mathbf { e } _ { \mathrm { k } } = \phi _ { \tau } ( \mathbf { u } _ { \mathrm { k } } ) + \mathbf { e } _ { \mathrm { l e a d } } ( \mathbf { k } ) + \mathbf { e } _ { \mathrm { t e m p } } ( \mathbf { k } ) \in \mathbb { R } ^ { \mathrm { D } } ,\tag{2}
$$

where $\mathbf { e } _ { \mathrm { l e a d } } ( \mathbf { k } )$ and $\mathbf { e } _ { \mathrm { t e m p } } ( \mathbf { k } )$ denote the lead embedding and the temporal embedding associated with the k-th token unit, respectively. The embedding sequence ${ \textbf { E } } =$ $( \mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { \mathrm { { L } } } ) \bar { \in } \mathbb { R } ^ { \mathrm { { L } } \times \mathrm { { D } } }$ is then processed by a Transformer.

## 2.2. Tokenizer Architectures

Let us now briefly introduce the eight tokenizer baselines that we will compare. All dimensions below refer to our 100 Hz, 10 s, 12-lead ECG input $( T = 1 0 0 0 , C = 1 2 )$ Here, k follows the token-unit index introduced in Section 2.1; for lead-specific tokenizers, the lead index c and within-lead index n are flattened into k in lead-major order.

Point-wise. Each simultaneous multi-lead sample is one token unit, $\mathbf { u } _ { \mathrm { k } } = \mathbf { X } _ { \mathrm { k } } , $ for $k = 1 , \dots , T$ . Hence, $L = T$ and $P = C$ , and attention models temporal relations at the original sampling resolution. Since each unit mixes all leads, it has no unique lead identity.

Lead-wise. Inspired by iTransformer, the input axes are inverted so that one complete lead is one token unit, $\mathbf { u } _ { \mathbf { k } } =$ $\mathbf { X } _ { : , \mathrm { k } }$ for $k = 1 , \dots , C$ . Hence, $L = C$ and $P = T .$ , and attention operates across leads rather than time points.

Patching. Each lead is divided into overlapping windows of length p with stride $s ,$ with one stride replicated at the right boundary. Let $N _ { p } = \lfloor ( T + s - p ) / s \rfloor + 1$ . The unit indexed by $k = ( c - \bar { 1 } ) N _ { p } + n$ is the n-th patch from lead c, giving $L = C N _ { p }$ . We choose $p = 1 6$ and $s = 8$ via grid search.

Stochastic segments. For each lead, up to K intervals $[ a _ { n } , b _ { n } ]$ are sampled, where the segment length satisfies $b _ { n } \mathrm { ~ - ~ } a _ { n } \in \mathsf { \Gamma } [ \ell _ { \operatorname* { m i n } } , \ell _ { \operatorname* { m a x } } ]$ and the start-point increment satisfies $a _ { n + 1 } - a _ { n } \in [ s _ { \operatorname* { m i n } } , s _ { \operatorname* { m a x } } ]$ . The unit u , indexed by $k = ( c - 1 ) K + n$ , is obtained by linearly resampling $\mathbf { X } _ { \mathrm { a _ { n } : b _ { n } , c } }$ to $P$ samples. A seeded template fixes the same boundaries across leads and throughout a run. The sequence contains at most CK units and is padded to $L _ { \mathrm { m a x } } = C K$ for batching. We choose $K = 1 6 , P = 1 2 8$ $[ \ell _ { \mathrm { m i n } } , \ell _ { \mathrm { m a x } } ] = [ 8 0 , 1 6 0 ]$ , and $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ] = [ 5 0 , 7 0 ]$ via grid search.

BIOT-inspired segments. Following [8], each lead is divided into fixed windows of length $p = 1 0 0$ with stride $s = 5 0$ . This gives $N _ { b } = \lfloor ( T - p ) / s \rfloor + 1 = 1 9$ windows per lead. The index $k = ( c - 1 ) N _ { b } + n$ denotes the n-th window from lead c, yielding $L = C N _ { b } = 2 2 8$

ECG-Byte. The central 2s interval is resampled from 100 to 250 Hz, normalized using training-set percentiles, and quantized into 26 amplitude symbols. The lead-major symbol stream is compressed using the pretrained 3,500- entry byte-pair encoding vocabulary [9]. Each $\mathbf { u } _ { \mathrm { k } }$ is a discrete vocabulary index. The resulting sequence is capped at $L _ { \mathrm { m a x } } = 1 0 2 0$ and padded when shorter.

HeartLang. Following [10], QRS complexes are detected once on the lead II, and the resulting beat boundaries are shared across all leads. Let $\mathbf { b } _ { \mathrm { c , n } } \in \mathbb { R } ^ { 9 6 }$ denote the nth heartbeat from lead c after cropping or zero-padding. These lead–beat units are ordered lead-major such that $\mathbf { u _ { k } } = \mathbf { b } _ { \mathrm { c , n } }$ and the variable-length sequence is capped and padded to $L _ { \operatorname* { m a x } } = 2 5 6$

Median beat. R peaks are detected on lead II, and R-centred multi-lead beats of length W are extracted. Their element-wise median yields a representative heartbeat $ { \widetilde { \mathbf { X } } } \in \mathbb { R } ^ {  { \mathrm { L } } \times  { \mathbf { C } } }$ . Each relative time point constitutes one token unit, $\mathbf { u } _ { \mathrm { k } } = \widetilde { \mathbf { X } } _ { \mathrm { k } } ,$ <sub>,:</sub> for $k = 1 , \dots , L$ . Attention therefore models morphological relationships within a representative heartbeat rather than relationships among individual beats. As in point-wise tokenization, each unit contains measurements from all leads and consequently has no unique lead identity. The beat length was set to $L = 6 0$ samples based on a grid search, and we found that longer windows did not improve performance.

## 3. Results

## 3.1. Experiment Setup

Baselines. We evaluate four Transformer backbones: Transformer [4], Informer [5], Reformer [6], and FEDformer [7]. For tokenizer-independent context, we also consider the convolutional XResNet1d101 and recurrent bidirectional LSTM baselines.

![](images/09bdf118238a54c4f22a3d3d0bb1494c7b53a743d648519a2e8ca20f0c9cf413.jpg)  
Figure 1. CPSC2018 test macro-AUC under eight tokenizer configurations. The tokenizer-independent convolutional and recurrent baselines are shown separately.

CPSC2018 dataset. We downsample each recording to 100 Hz and retain or resample it to 10 s. The task predicts nine diagnostic labels: AFIB, VPC, NORM, 1AVB, CRBBB, STE, PAC, CLBBB, and STD. The dataset was divided into training, validation, and test sets in a patientstratified 8:1:1 ratio. A global standardizer is fitted on the training set and then applied to all splits.

Implementations. For a controlled comparison, every Transformer backbone uses embedding dimension D = 128, three encoder layers, and eight attention heads. Continuous token units are mapped from P to D by a linear input projection, whereas ECG-Byte uses a learned vocabulary embedding. Following HeartLang, lead-specific tokens receive a learned embedding indexed by lead identity. The positional term combines a learned absolute tokenposition vector with a learned embedding of the temporal index. Models are optimized with RAdam and binary cross-entropy at a learning rate of $1 0 ^ { - 3 }$ for at most 300 epochs, with early stopping after 15 epochs without validation macro-AUC improvement. All experiments are conducted on a single NVIDIA A40 GPU.

## 3.2. Main Results

Figure 1 reports the best test macro-AUC for each available backbone–tokenizer pair. The CNN and recurrent baselines reach 0.958–0.961 macro-AUC and outperform all evaluated Transformer combinations. Among the latter, the strongest combination is Reformer with HeartLang, reaching 0.932 macro-AUC and 0.607 macro-F1. Reformer with patching ranks second in macro-AUC (0.920), while Informer with HeartLang obtains 0.913 macro-AUC and the highest macro-F1. HeartLang is the best tokenizer for Transformer, Reformer, and Informer.

Median-beat tokenization is the most consistent compact representation. It is second best for Transformer and Informer, remains competitive for Reformer, and is the strongest FEDformer input by a large margin, improving its macro-AUC from 0.745 with point-wise tokens to 0.882 with only 60 tokens. Patching is highly backbonedependent, which performs poorly with FEDformer. ECG-Byte is less competitive, which may reflect both its restriction to the central 2 s and the observed truncation at its 1,020-token cap. Lead-wise inversion performs worst or near-worst across all four backbones, contrasting with its strong performance in general time-series forecasting.

## 3.3. Analysing efficiency

Tables 2 and 3 separate complete model cost from tokenizer-only latency. Measurements use an NVIDIA A40, batch size 32, three warm-up iterations, and ten timed repetitions. Patching produces 1,500 tokens and requires 18.86 GB with Transformer, whereas lead-wise tokenization requires at most 0.06 GB across the four backbones. The compact physiological representations also substantially reduce the memory requirements. Median-beat tokenization requires only 0.08–0.16 GB and has an endto-end latency of 3.87–4.64 ms per ECG, while Heart-Lang requires 0.22–0.66 GB and 4.59–5.22 ms per ECG. Their tokenizer-only costs are already 3.81 and 4.37 ms per ECG, respectively, indicating that heartbeat extraction dominates their wall-clock latency. This cost accompanies strong predictive performance. ECG-Byte is the slowest tokenizer at 10.36 ms per ECG, making its endto-end latency approximately 11–12 ms across all backbones. BIOT-inspired tokenization is comparatively efficient, while the median beat provides a favourable predictive performance.

Table 2. Complete model efficiency results. Each entry reports end-to-end inference latency in ms/ECG followed by peak allocated training VRAM in GB (latency/VRAM). The lowest value is bold and the second lowest is underlined. Latency is averaged over ten repetitions; VRAM is measured for a complete training step.
<table><tr><td>Tokenizer</td><td>Transformer</td><td>Reformer</td><td>Informer</td><td>FEDformer</td></tr><tr><td>Point-wise</td><td>1.35/8.52</td><td>1.95/3.41</td><td>0.57/1.08</td><td>0.86/0.60</td></tr><tr><td>Lead-wise</td><td>0.06/0.03</td><td>0.15/0.06</td><td>0.10/0.03</td><td>0.21/0.04</td></tr><tr><td>Patching</td><td>3.11/18.86</td><td>3.78/6.59</td><td>0.89/1.75</td><td>0.90/0.87</td></tr><tr><td>Stochastic</td><td>4.17/0.40</td><td>4.26/0.45</td><td>4.26/0.21</td><td>4.90/0.18</td></tr><tr><td>BIOT-inspired</td><td>0.12/0.54</td><td>0.29/0.54</td><td>0.16/0.25</td><td>0.84/0.20</td></tr><tr><td>ECG-Byte</td><td>11.79/8.87</td><td>12.18/3.56</td><td>11.01/1.10</td><td>11.31/0.61</td></tr><tr><td>HeartLang</td><td>4.59/0.66</td><td>4.67/0.59</td><td>4.62/0.27</td><td>5.22/0.22</td></tr><tr><td>Median beat</td><td>3.87/0.08</td><td>3.96/0.16</td><td>3.95/0.08</td><td>4.64/0.10</td></tr></table>

Table 3. Tokenizer-only efficiency on an NVIDIA A40 at batch size 32. Latency is the mean time per ECG over ten repetitions after three warm-up iterations.
<table><tr><td>Tokenizer</td><td>L</td><td>Tokenization (ms/ECG)</td></tr><tr><td>Point-wise</td><td>1000</td><td>&lt; 0.01</td></tr><tr><td>Lead-wise</td><td>12</td><td>&lt; 0.01</td></tr><tr><td>Patching</td><td>1500</td><td>&lt; 0.01</td></tr><tr><td>Stochastic</td><td>192</td><td>4.07</td></tr><tr><td>BIOT-inspired</td><td>228</td><td>0.01</td></tr><tr><td>ECG-Byte</td><td>1020</td><td>10.36</td></tr><tr><td>HeartLang</td><td>256</td><td>4.37</td></tr><tr><td>Median beat</td><td>60</td><td>3.81</td></tr></table>

## 4. Conclusion

We found that tokenization is an important determinant of the ECG Transformer performance. Across four backbones, median-beat and HeartLang representations improve pooled mean macro-AUC from 0.823 to 0.891 relative to conventional point-wise and patch-wise inputs, while reducing mean peak training VRAM from 5.21 to 0.27 GB. The median beat offers the best overall performance–memory trade-off, whereas HeartLang with Reformer gives the best individual Transformer result. These findings show the value of preserving physiologically coherent heartbeat morphology for ECG tokenizers.

## Acknowledgments

Jiawei Li and Antonio Horta Ribeiro are financially sup-ˆ ported by the eSSENCE and SciLifeLab, with the project ”Digital Biomarkers from the Electrocardiogram using Artificial Intelligence”; and, by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by Knut and Alice Wallenberg Foundation. This research was partially supported by Kjell och Marta Bei-¨ jer Foundation. This project has received funding from the European Research Council (ERC) under the European Union’s Horizon Europe research and innovation programme through grant agreement no. 101054643. Computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.

## References

[1] Al-Masud M, Lopez Alcaraz JM, Strodthoff N. Benchmarking ECG FMs: A reality check across clinical tasks. In International Conference on Learning Representations (ICLR). 2026;

[2] Guoqi Y, Wang J, Yang C, Qin J, Aviles-Rivero A, Wang S. Decentralized attention fails centralized signals: Rethinking transformers for medical time series. In International Conference on Learning Representations (ICLR). 2026;

[3] Gu X, Tang W, Han J, Sangha V, Liu F, Gowda SN, Ribeiro AH, Schwab P, Branson K, Clifton L, et al. Cardiac health assessment across scenarios and devices using a multimodal foundation model pretrained on data from 1.7 million individuals. Nature Machine Intelligence 2026;8(2):220–233.

[4] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I. Attention is all you need. In Advances in Neural Information Processing Systems. 2017;

[5] Zhou H, Zhang S, Peng J, Zhang S, Li J, Xiong H, Zhang W. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Conference on Artificial Intelligence (AAAI). 2021;

[6] Kitaev N, Kaiser L, Levskaya A. Reformer: The efficient transformer. International Conference on Learning Representations (ICLR). 2020; .

[7] Zhou T, Ma Z, Wen Q, Wang X, Sun L, Jin R. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning (ICML), volume 162. 2022;

[8] Yang C, Westover MB, Sun J. Biot: biosignal transformer for cross-data learning in the wild. In Advances in Neural Information Processing Systems. 2023; .

[9] Han W, Duan C, Rosenberg MA, Liu E, Zhao D. Ecgbyte: A tokenizer for end-to-end generative electrocardiogram language modeling, 2024.

[10] Jin J, Wang H, Li H, Li J, Pan J, Hong S. Reading your heart: Learning ECG words and sentences via pretraining ECG language model. InInternational Conference on Learning Representations (ICLR). 2025;