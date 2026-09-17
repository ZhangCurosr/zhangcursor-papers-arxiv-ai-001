# TRIPROBE: Probing Task Separability Beyond Classification for XAI

1<sup>st</sup> Amirhossein Sadough

Machine Learning and Neural Computing

2<sup>nd</sup> Freek Hens

3<sup>rd</sup> Aleksa Bokšan

Machien Learning and Neural Computing

Delft University of Technology

Radboud University

Radboud University

Delft, Netherlands

Nijmegen, Netherlands

Nijmegen, Netherlands

ORCID:0009-0003-3099-4472

ORCID:0009-0005-5647-2888

ORCID:0009-0005-0405-0560

4<sup>th</sup> Mohammad Mahdi Dehshibi

Unconventional Computing Lab

University of the West of England (UWE)

Bristol, United Kingdom

ORCID: 0000-0001-8112-5419

5<sup>th</sup> Mahyar Shahsavari

Machine Learning and Neural Computing

Radboud University

Nijmegen, Netherlands

ORCID:0000-0002-9671-0917

Abstract—Modern evaluation of learning pipelines often reduces to downstream accuracy, leaving open the question of why tasks succeed or fail. TriProbe addresses this gap with a multi-level probing framework for explainable diagnosis of task separability. Rather than treating models as black boxes, TriProbe traces how separability evolves across inputs, learned features, and final classifiers. It decomposes multi-task problems into binary subtasks and applies three complementary probes: a Foundational Probe on input spaces, a Latent Probe on feature representations, and a Final Probe on classifier outputs. Using Maximum Fisher’s Discriminant Ratio as a principled separability metric, TriProbe identifies bottlenecks and affected task pairs. Experiments on the Roshambo sEMG benchmark show how TriProbe reveals hidden breakdowns, guiding data collection, validation, and architecture design.

Index Terms—Explainable AI, task separability, probing framework, Fisher’s discriminant ratio, learning pipeline

## I. INTRODUCTION

Understanding why learning models succeed or fail in distinguishing between tasks remains a fundamental challenge in machine learning and signal processing. Across diverse application areas, including speech, EEG/MEG, and sEMG, task separability is affected not only by intrinsic data complexity but also by architectural choices across the learning pipeline [1]. Existing evaluation methods typically focus on downstream accuracy, providing little insight into where separability bottlenecks emerge [2]–[4]. This lack of interpretability complicates both data collection and model design, motivating the need for systematic and explainable tools [5], [6].

Model probing has emerged as a promising methodology for understanding how representations evolve in modern learning pipelines. Early work introduced diagnostic classifiers to test whether embeddings captured linguistic or structural informa tion [7]–[9]. Complementary explainable AI approaches, such as Shapley values and integrated gradients, provide attribution scores but typically stop short of tracing separability across multiple stages of the pipeline [10], [11]. In parallel, studies in physiological signals, such as sEMG gesture recognition, have emphasized the importance of transfer learning and representation quality for task separability [12], [13]. However, these methods primarily aim at boosting accuracy rather than systematically diagnosing where separability is gained or lost across stages.

To address this gap, we introduce TriProbe, a multi-level probing framework that explains the root causes of task separability difficulties by tracing them across successive stages of learning. The central premise of TriProbe is that class separability may either deteriorate or improve throughout different stages of the learning pipeline. Understanding these dynamics requires not only identifying where representational bottlenecks arise but also determining which task pairs are most affected.

TriProbe decomposes a multi-task classification problem into a set of binary sub-problems and systematically analyzes separability across the processing hierarchy. The framework first examines the discriminative potential of the original input data and handcrafted feature representations through a Foundational Probe. It then evaluates the quality of the learned latent representations produced by the feature extractor using a Latent Probe. Finally, a Final Probe assesses the separability achieved in the network’s output space, enabling a comprehensive diagnosis of how discriminative information evolves from the input to the final decision layer.

The contributions of this work are threefold. First, we propose TriProbe as a stage-consistent diagnostic framework for explainable evaluation of task separability across learning stages. Second, we show how it provides actionable insights by diagnosing task difficulty, localizing bottlenecks, and revealing raw-level data complexity that informs both collection strategies and interpretation of results. Third, we demonstrate TriProbe on an sEMG gesture recognition task, where it uncovers bottlenecks consistent with downstream performance and prior studies, while exposing how separability evolves across stages.

![](images/0fcce666bb8f5fcfdcad2045ad8bb0aba239da0b016669a9aa0e8816b0892cb7.jpg)  
Fig. 1: TriProbe: a framework for explainable separability analysis across learning stages.

## II. PROPOSED MULTI-LEVEL PROBING

This work introduces TriProbe, a multi-level probing framework for diagnosing the root causes of task separability difficulty across successive stages of learning. The key idea is that separability may degrade or improve at different points in the pipeline, so identifying bottlenecks requires both localizing the stage and pinpointing the most affected class pairs. TriProbe decomposes a multi-task problem into binary sub-tasks, preventing models from exploiting indirect cues from unrelated classes and ensuring that analysis reflects the intrinsic difficulty of each pair. As illustrated in Figure 1, TriProbe employs three complementary probes : (i) a Foundational Probe to assess raw data and hand-crafted features, (ii) a Latent Probe to evaluate feature extractor representations, and (iii) a Final Probe to analyze final-level separability. By combining binary decomposition with multi-level probing, TriProbe offers a principled and interpretable diagnosis of whether limitations arise from data, representation, or final stages.

## A. Foundational Probe

Let $\mathbf { X } \in \mathbb { R } ^ { n \times m }$ denote the raw multi-channel input data, where n is the number of time samples and m the number of input channels. For each class $c _ { i } .$ , let $\mathbf { X } _ { c _ { i } } \in \mathbb { R } ^ { n _ { i } \times m }$ represent the set of raw samples belonging to that class, with $n _ { i }$ denoting the number of samples. In addition to the raw space, we also consider a hand-crafted feature space. A feature extraction function $\Phi ( \cdot )$ maps each raw input $\mathbf { x } \in \mathbb { R } ^ { n \times m }$ into a feature vector $\mathbf { f } = \Phi ( \mathbf { x } ) \in \mathbb { R } ^ { d }$ , where d is the number of engineered features (e.g., entropy measures, zero-crossings, RMS, etc.). Collecting across all samples of class $c _ { i }$ yields the feature matrix $\mathbf { F } _ { c _ { i } } \in \mathbb { R } ^ { n _ { i } \times d }$

Maximum Fisher’s Discriminant Ratio (F1): A central analytical tool in our framework is Maximum Fisher’s Discriminant Ratio (F1), drawn from data complexity analysis [14]. This metric quantifies separability between two distributions along individual feature dimensions and identifies the single most discriminative feature for distinguishing a target class from a reference. In the one-vs-rest setting, where a target class $c _ { i }$ is compared against all others, the Fisher score of feature k is defined as

$$
\mathrm { F } 1 _ { k } ( c _ { i } ) = \frac { ( \mu _ { i , k } - \mu _ { \mathrm { r e s t } , k } ) ^ { 2 } } { \sigma _ { i , k } ^ { 2 } + \sigma _ { \mathrm { r e s t } , k } ^ { 2 } } ,
$$

and the overall score is $\operatorname { F 1 } ( c _ { i } ) = \operatorname* { m a x } _ { k } \operatorname { F 1 } _ { k } ( c _ { i } )$ . For pairwise analysis between two classes $( c _ { i } , c _ { j } )$ , the same formulation applies by replacing the “rest” statistics with those of class $c _ { j } .$ Higher F1 values indicate stronger separability, while low values suggest intrinsic overlap. This metric is particularly useful because it highlights the single most discriminative feature dimension, allowing us to assess whether poor separability stems from intrinsic data overlap at the raw level.

## B. Latent Probe

This probe evaluates task separability in the learned representation space. The feature extractor can be any contemporary architecture, which makes the approach general. In this work, we employ an autoencoder (AE) and use the encoder output after ensuring satisfactory reconstruction quality, while discard ing the decoder. This choice avoids bias from downstream task objectives such as classification, while still capturing rich datadriven features. We train a dedicated AE for each class, enabling fine-grained diagnosis and preventing the AE to exploit indirect cues from unrelated classes during reconstruction. This ensures that the encoder learns features solely from its own class distribution, thereby preserving class-specific representation quality. Concretely, the encoder maps each input sample x into a latent feature vector $\textbf { f } \in \mathbb { R } ^ { d }$ . Collecting samples of class $c _ { i }$ forms a feature matrix $\mathbf { F } _ { c _ { i } } ~ \in ~ \mathbb { R } ^ { n _ { i } \times d }$ , and similarly $\mathbf { F } _ { c _ { j } }$ for class $c _ { j }$ , which are then compared to measure pairwise separability in the latent space.

## C. Final Probe

This probe is placed immediately before the final separation stage (e.g., the last classifier layer). It aims to evaluate how effectively the body network prepares discriminative representations for the final decision. To maintain generality, we employ a simple multilayer perceptron (MLP) as the classifier body, positioned after AE’s latent representation (encoder output). The MLP is chosen deliberately as a generic architecture, free from domain-specific inductive biases. Therefore, the probe reflects the intrinsic quality of the representations rather than advantages of a specialized classifier. For each input sample, the probe captures the representation vector at the penultimate layer (i.e., the input to the final layer). These vectors form the basis for evaluating task separability at the separation boundary. This enables assessment of whether difficulty arises from insufficiently discriminative representations at the body level, as opposed to intrinsic data limitations (Foundational Probe) or bottlenecks in feature extraction (Latent Probe).

## D. Interpretation

TriProbe provides a stage-wise view of how task separability evolves through the learning pipeline. Low scores at the Foundational Probe signal intrinsic data complexity, while discrepancies between later probes expose architectural limitations. A key guideline is that if separability is strong in the latent space but weak at the final stage, the issue lies in the intervening architecture rather than the data. Thus, TriProbe not only diagnoses task difficulty via binary decomposition and localizes bottlenecks, but also informs data quality, guides architectural refinement, and supports the rational interpretation of results. By pinpointing where separability is lost or preserved, TriProbe advances the goals of Explainable AI with a practical tool for both analysis and design.

## III. EXPERIMENT

## A. Setup

We evaluate the proposed TriProbe framework on the Roshambo dataset [15], a benchmark known for its low signal-to-noise ratio. The dataset contains recordings from ten participants using a Myo armband with eight sEMG channels sampled at 200 Hz. Participants performed three gestures — Rock (R), Paper (P), Scissors (S) — plus a control (Rest). Each participant completed three sessions, with five trials of 3 seconds (s) per gesture, yielding 450 trials in total. To isolate steady-state activity, the first and last 600 ms of each trial were discarded. A sliding window of 400 samples (2 s) with 50% overlap was then applied, yielding 296 samples per class of size 8×400 (channels × time).

Foundational Probe: Each 8 × 400 sample was transformed into a 72-D feature vector (8 channels × 9 features) using a modality-agnostic set of hand-crafted features including Shannon entropy, sample entropy, zero crossings, waveform length, root mean square (RMS), slope sign changes, median frequency, wavelet energy, and fractal dimension. This yields $\mathbf { F } _ { c _ { i } } ~ \in ~ \mathbb { R } ^ { 2 9 6 \times 7 2 }$ per class, capturing temporal, spectral, and complexity characteristics while avoiding domain-specific bias.

Latent Probe: Three class-specific AE (R-only, P-only, S-only) were trained to construct latent spaces for pairwise separability analysis. When referring to an AE in the experiments, we specifically use the architecture detailed in Table I. The encoder output, once achieving satisfactory reconstruction quality, was taken as the latent representation. Each class $c _ { i }$ produces $\mathbf { F } _ { c _ { i } } \in \mathbb { R } ^ { 2 9 6 \times 3 9 2 }$ , with 392 being the latent dimension. Training used 300 epochs with Huber loss $( \delta = 0 . 2 5 )$ , Adam optimizer $( \mathrm { l r } = 5 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 } )$ , and all available samples, aiming the robust representation learning rather than downstream classification.

Final Probe: For each binary sub-task (R–P, R–S, P–S), we trained a pairwise AE whose encoder output was fed into a simple multilayer perceptron (MLP). To avoid task-objective bias, the AE was kept frozen (decoder discarded, encoder weights fixed), and only the MLP was trained. The initial MLP layers are treated as the body, while the last layer defines the final separator, with the probe placed at their interface. This design preserves architectural neutrality while enabling the probe to identify whether separability limitations arise from insufficient body representations or from the final decision rule.

## B. Results

Figure 2 illustrates the application of TriProbe on the Roshambo dataset. We first validate the use of F1 for task separability evaluation and show its consistency with both downstream performance and prior results on this dataset. This establishes the analytical core of TriProbe as a robust evaluator, capable of supporting multiple interpretations when deployed at different stages of the learning pipeline. Second, we report the experiment specific findings.

F1 Validation: F1 applied at the Foundational probe shows that the P–S sub-task has the weakest separability, with substantial overlap in both raw data and hand-crafted feature assessments. In contrast, R–S and R–P consistently display stronger separability. A closer look reveals a minor discrepancy: raw data suggests R–P is slightly easier than R–S, while handcrafted features suggest the opposite. At the Latent probe, F1 results align with the raw-level findings, and the Final probe largely preserves this pattern, except for a noticeable degradation in R–S. Confusion matrices further confirm that P–S is the hardest pair to discriminate, consistent with both our probes and prior results from [16] (see Fig. 3). Interestingly, their two model variants diverge slightly: one favors R–S, the other R–P, echoing our observation that such small gaps are sensitive to the feature space on which the learner operates.

Two key insights emerge: (i) inherent data complexity observed at the raw level propagates through the pipeline, meaning downstream stages should not contradict these trends, and (ii) F1 can flag sub-task separability difficulties early, even at the data collection stage, making it a practical diagnostic tool. Taken together, these results validate F1 as an effective probe of task separability, allowing researchers to peak into the black box.

TriProbe Analysis: With F1 validated as a reliable separabil ity evaluator, we now leverage it to analyze different stages of the learning pipeline. By decomposing the multi-task problem into binary sub-tasks, TriProbe enables pairwise separability analysis and reveals which sub-tasks act as bottlenecks from raw data through to downstream performance. Our results consistently indicate that separability is most challenging for P–S, as reflected in both our binary confusion matrices and prior multi-task evaluations in [16]. This shows that TriProbe can diagnose the root cause of multi-task difficulty already at the raw data level. Accordingly, P–S emerges as the primary bottleneck pair, drawing attention to where learning pipeline design should be strengthened, especially in multi-task settings. Beyond bottleneck detection, TriProbe also reveals how separability evolves across stages, supporting architectural search. For instance, R–S shows good separation in the Latent probe but degrades at the Final probe, suggesting ineffective learning in the MLP body. Redesigning this stage could mitigate the loss. Similarly, P–S not only appears as the hardest pair overall but also undergoes degradation from Foundational to Latent probe, underscoring the need for stronger feature extraction. Thus, TriProbe not only diagnoses bottlenecks but also tracks the propagation of separability across the pipeline, enabling principled evaluation of whether architectural modifications improve or degrade performance relative to a baseline.

TABLE I: The utilized AE. k: kernel, s: stride, BN: BatchNorm.
<table><tr><td>Layer Config</td><td colspan="2"></td><td>Output</td></tr><tr><td>Input</td><td>一</td><td></td><td>1 × 8 × 400</td></tr><tr><td>Conv1</td><td>1  $\overline { { { \mathrm { \Large ~  \sum B , ~ } k = 3 \times 3 , ~ s = 1 , ~ \mathrm { \tiny ~ R e L U } , } } }$  BN, Dropout</td><td></td><td>128 × 6 × 398</td></tr><tr><td>Conv2</td><td>128 → 256, k = 3 × 3, s =1, Leaky-ReLU, BN, Dropout</td><td></td><td> $2 5 6 \times 4 \times 3 9 6$ </td></tr><tr><td>Conv3</td><td> $2 5 6 \to 5 1 2 , k = 3 \times 3 , s = 1 , \mathrm { E L U } ,$ </td><td>BN, Dropout</td><td> $5 1 2 \times 2 \times 3 9 4$ </td></tr><tr><td>Conv4</td><td>512→1, k=2×3, s=1</td><td></td><td> $1 \times 1 \times 3 9 2$ </td></tr><tr><td>Latent</td><td>一</td><td></td><td>392</td></tr><tr><td>Deconv1</td><td> $1  5 1 2 , k = 2 \times 3 , s = 1 , \mathrm { R e L U } ,$ </td><td>BN, Dropout</td><td> $\overline { { 5 1 2 \times 2 \times 3 9 4 } }$ </td></tr><tr><td>Deconv2</td><td> ${ 5 1 2 } {  } 2 5 6 ,  { k } { = } 3 { \times } 3 ,  { s } { = } 1 , \ { \mathrm { S o f t p l u s } } ,$ </td><td>BN, Dropout</td><td> $2 5 6 \times 4 \times 3 9 6$ </td></tr><tr><td>Deconv3</td><td> $2 5 6  1 2 8 , k = 3 \times 3 , s = 1 .$  ELU,</td><td>BN, Dropout</td><td> $1 2 8 \times 6 \times 3 9 8$ </td></tr><tr><td>Deconv4</td><td>128→1,  $k = 3 \times 3 , s = 1$ </td><td></td><td> $1 \times 8 \times 4 0 0$ </td></tr><tr><td>Output</td><td>一</td><td></td><td> $\overline { { 1 \times 8 \times 4 0 0 } }$ </td></tr></table>

![](images/fbdcededaa2d06476fe66feaea2a5785bcc4f362c7fdf1e06f2ea6f3efd7ef8a.jpg)  
Fig. 2: Experimenting the proposed TriProbe framework on the Roshambo dataset, an sEMG-based gesture recognition task with three classes (Rock, Paper, and Scissors). The framework decomposes the problem into binary sub-tasks (Rock–Paper, Rock–Scissors, Paper–Scissors) and evaluates separability across three levels: (i) Foundational Probe measuring raw data and hand-crafted feature discriminability using F1; (ii) Latent Probe assessing latent representations with F1 and visualized using UMAP scatter plots, with covariance ellipses summarizing class spread; and (iii) Final Probe analyzing separability at the penultimate classifier layer. Evaluation confusion matrices are shown for each binary sub-task.

![](images/5d140f335fd5da03632204bdb5379c66477d42782a56519b29249581ef78469f.jpg)  
Fig. 3: Normalized confusion matrices of two Roshambo model variants from [16].

## IV. DISCUSSION

TriProbe provides a structured methodology for diagnosing task separability across the full learning pipeline. Unlike existing probing approaches that focus on isolated representations, TriProbe localizes where separability is gained or lost throughout the entire learning pipeline. While validated here on an sEMG gesture recognition dataset, its formulation is architecture-agnostic and task-independent, suggesting broad applicability across domains such as speech, EEG/MEG, and sensor networks.

While TriProbe builds upon established components such as Fisher’s Discriminant Ratio, autoencoders, and probing concepts, its novelty lies in their integration into a unified diagnostic framework for explainable task separability analysis. Rather than introducing a new classifier or separability metric, TriProbe provides a stage-consistent methodology that systematically traces how discriminative information evolves from the raw input space, through learned latent representations, to the final decision space. By combining binary task decomposition with a common separability criterion across all stages, the framework enables direct localization of representational bottlenecks and distinguishes whether limitations originate from the data itself, the feature extractor, or the downstream classifier. This unified perspective is not provided by existing probing or attribution methods, which typically analyze only a single stage or explain individual predictions.

A promising use case for TriProbe is early-stage evaluation during data collection. By probing separability at the raw and hand-crafted feature level, TriProbe serves as an early warning system, allowing pilot studies and assess data quality before committing to large-scale collection or model training. This is particularly valuable in domains where acquisition is costly. Another application lies in verifying the rationality of experimental findings. If probe results at early stages contradict downstream model performance, TriProbe can highlight potential issues such as overfitting, data leakage, or ineffective architectural choices. In this way, TriProbe complements traditional performance metrics by providing an interpretable explanation of where separability is lost or preserved across stages.

Beyond these general applications, our findings also suggest dataset-specific improvements. For example, the observed degradation of the R–S pair between the latent and final probes indicates that the current MLP body may be ineffective, motivating experiments with alternative architectures. Such targeted refinements illustrate how TriProbe can guide iterative design by localizing weaknesses in the pipeline. Extending this idea, future work should evaluate TriProbe across different datasets and modalities, not only to confirm its generalizability but also to explore how separability evolves under varying data complexities and learning architectures.

## V. CONCLUSION

In conclusion, TriProbe contributes to Explainable AI by providing a unified framework for tracing the root causes of task separability difficulty across raw data, learned representations, and decision spaces. Rather than relying solely on downstream performance metrics, it offers stage-wise insights into how discriminative information evolves throughout the learning pipeline, enabling the localization of bottlenecks and the identification of challenging class pairs. Consequently, TriProbe serves not only as an early diagnostic tool for assessing data quality, but also as a means of validating experimental findings and guiding architecture refinement through interpretable separability analysis. Although we demonstrated TriProbe here on an sEMG gesture recognition task, the framework is designed to be architecture-agnostic and readily applicable to other machine learning domains. Future work will evaluate TriProbe across additional modalities and datasets, investigate alternative separability measures, and integrate the framework into adaptive learning pipelines, with the long-term goal of establishing a general-purpose, separability-driven methodology for interpretable and reliable AI.

## REFERENCES

[1] Shimon Fridkin and Michael Bendersky, “Interpretable machine learning: A comprehensive review of foundations, methods, and the path forward,” WIREs Data Mining and Knowledge Discovery, vol. 16, no. 1, pp. e70075, 2026.

[2] Ricards Marcinkeviˇ cs and Julia E. Vogt, “Interpretable and explainableˇ machine learning: A methods-centric overview with concrete examples,” WIREs Data Mining and Knowledge Discovery, vol. 13, no. 3, pp. e1493, 2023.

[3] Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frederic Precioso, Stefano Melacci, Adrian Weller, Pietro Lio, and Mateja Jamnik, “Concept embedding models: beyond the accuracy-explainability trade-off,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2022, NIPS ’22, Curran Associates Inc.

[4] Pantelis Linardatos, Vasilis Papastefanopoulos, and Sotiris Kotsiantis, “Explainable AI: A review of machine learning interpretability methods,” Entropy, vol. 23, no. 1, 2021.

[5] Kim Huat Goh, Le Wang, Adrian Yong Kwang Yeow, Hermione Poh, Ke Li, Joannas Jie Lin Yeow, and Gamaliel Yu Heng Tan, “Artificial intelligence in sepsis early prediction and diagnosis using unstructured data in healthcare,” Nature Communications, vol. 12, no. 1, pp. 711, 2021.

[6] Meike Nauta, Jan Trienes, Shreyasi Pathak, Elisa Nguyen, Michelle Peters, Yasmin Schmitt, Jörg Schlötterer, Maurice van Keulen, and Christin Seifert, “From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI,” ACM Comput. Surv., vol. 55, no. 13s, July 2023.

[7] Guillaume Alain and Yoshua Bengio, “Understanding intermediate layers using linear classifier probes,” arXiv preprint arXiv:1610.01644, 2016.

[8] John Hewitt and Christopher D Manning, “A structural probe for finding syntax in word representations,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 4129–4138.

[9] Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni, “What you can cram into a single vector: Probing sentence embeddings for linguistic properties,” 2018.

[10] Mukund Sundararajan and Amir Najmi, “The many shapley values for model explanation,” in International conference on machine learning. PMLR, 2020, pp. 9269–9278.

[11] Mukund Sundararajan, Ankur Taly, and Qiqi Yan, “Axiomatic attribution for deep networks,” in International conference on machine learning. PMLR, 2017, pp. 3319–3328.

[12] Ulysse Côté-Allard, Cheikh Latyr Fall, Alexandre Drouin, Alexandre Campeau-Lecours, Clément Gosselin, Kyrre Glette, François Laviolette, and Benoit Gosselin, “Deep learning for electromyographic hand gesture signal classification using transfer learning,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 27, no. 4, pp. 760–771, 2019.

[13] Mr. Amol Pandurang Yadav and Dr. Sandip.R. Patil, ““optimizing semg gesture recognition with stacked autoencoder neural network for bionic hand”,” MethodsX, vol. 14, pp. 103207, 2025.

[14] Tin Kam Ho and M. Basu, “Complexity measures of supervised classification problems,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 3, pp. 289–300, 2002.

[15] Elisa Donati, “EMG from forearm datasets for hand gestures recognition,” May 2019.

[16] Nikhil Garg, Ismael Balafrej, Yann Beilliard, Dominique Drouin, Fabien Alibart, and Jean Rouat, “Signals to Spikes for Neuromorphic Regulated Reservoir Computing and EMG Hand Gesture Recognition,” in International Conference on Neuromorphic Systems 2021. 2021, Association for Computing Machinery.