# EEG-Xplain: Decoding Neural Black-Boxes of EEG Foundation Models

Hansong Ma, Junxiao Wang

Abstract—EEG foundation models such as BIOT, LaBraM, and EEGMamba have achieved remarkable performance in neural signal decoding, but their black-box nature limits clinical trust and neuroscientific validation. We propose a unified attribution framework for interpreting EEG foundation models across heterogeneous architectures. The framework integrates gradient-, perturbation-, and activation-based explanation methods to analyze model behavior in spatial, temporal, and frequency dimensions. Spatially, it identifies critical EEG channels and visualizes their distributions using topographic maps. Temporally, it highlights decision-relevant signa segments through attribution heatmaps. In the frequency domain, it quantifies the contributions of canonical EEG rhythms via spectra perturbation analysis. To assess explanation reliability, we introduce a population-level evaluation combining Area Over the Perturbation Curve (AOPC) and cross-method consistency analysis. The framework further leverages Large Language Models (LLMs) to transform structured attribution outputs into natural-language reports, bridging low-level neural representations and high-leve semantic reasoning. Experiments on benchmark datasets, including Mumtaz2016 and TUAB, demonstrate that the generated explanations are consistent with established neurophysiological markers, validating meaningful neural representations while exposing potential dependencies on artifacts and spurious patterns. The proposed framework provides a standardized approach for evaluating the interpretability, reliability, and physiological plausibility of EEG foundation models.

Index Terms—Brain-Computer Interface, EEG, Foundation Model, Explainability.

## 1 INTRODUCTION

Background. Electroencephalography (EEG), serving as a core non-invasive technique for monitoring the dynamic electrophysiological activity of the human brain, holds irreplaceable value in the fields of neuroscience research, clinical auxiliary diagnosis, and Brain-Computer Interfaces (BCI). Early deep learning research primarily focused on lightweight architectures designed for specific tasks, such as EEGNet [1], proposed by Lawhern et al., and other variants based on Convolutional Neural Networks (CNNs). While these models successfully captured transient spatiotemporal patterns through local convolutional operators, their representational capacity was constrained by the scale of labeled data and the task-specific nature of their design. In recent years, the field of EEG decoding has undergone a significant paradigm shift, entering the era of Foundation Models. Models exemplified by BIOT [2], EEGPT [3](based on the Transformer architecture), and EEGMamba [4] (based on selective state-space models) have shattered performance records across various clinical benchmark tasks, such as the Mumtaz2016 depression triage task and the TUAB clinical abnormality detection task . These achievements were realized through large-scale self-supervised pre-training (e.g., masked autoencoding or contrastive learning) conducted on tens of thousands of hours of unlabeled EEG data. These models have demonstrated that large-scale representation learning can capture deeper, latent neurodynamic features than traditional architectures, possessing immense potential for generalization across different datasets.

Motivation. Despite the breakthroughs achieved by foundation models in predictive accuracy, their massive parameter counts and complex internal logic (such as global attention mechanisms or linear state transitions) have resulted in decision-making processes that exhibit severe “black-box” characteristics [5]. This lack of transparency creates a significant trust gap in high-stakes application scenarios, such as medical diagnosis. Currently, research into the interpretability of EEG foundation models still faces the following three core bottlenecks: First, there exists a challenging problem regarding the misalignment of latent space representations. EEG foundation models (such as LaBraM) map raw signals into a discrete latent space using a Neural Tokenizer. This implies that the models no longer process physical waveforms directly, but rather operate on high-dimensional combinations of symbolic tokens. The core challenge lies in this: although the model achieves accurate classification, there is a lack of a “Mapping Protocol” to bridge the features learned internally by the model with established neurophysiological principles (such as specific patterns of rhythmic evolution). Existing attribution methods can only provide local sensitivity maps; they are unable to distill the model’s decision paths within the latent space into clinically interpretable evidence through dimensionality reduction.

Secondly, the risk of spurious causality is a pervasive issue within the learning pathways of foundation models. Foundation models, trained on massive and heterogeneous datasets, are highly prone to acquiring statistical shortcuts that span across different clinical centers and recording devices [6], [7]. A critical challenge lies in determining whether an EEG foundation model bases its judgments on genuine physiological significance; we need a framework capable of simultaneously identifying the regions the model deems critical and evaluating whether the model made the correct decision based on the right reasons. This is particularly critical in high-stakes scenarios such as medical diagnosis, as erroneous decisions could hinder the clinical deployment of the model. For instance, if a model achieves accurate classification solely by detecting electrode impedance imbalance( a type of physical artifact), it would lack clinical credibility [8].

Furthermore, there exists a distinct logical disconnect between perceptual maps and semantic reasoning. Fundamentally, current Explainable AI (XAI) results remain confined to the realm of visual features rather than constituting knowledge-based conclusions. The key requirement is that neuroscientists and clinicians need a summary of the causal chain; for example, a diagnosis of a high level of attention is inferred from the suppression of alpha waves in the occipital lobe. Existing attribution algorithms fail to bridge the gap between gradient-based weights and symbolic reasoning, thereby creating a significant barrier to effective interaction between AI-driven decision-making and the clinical logic employed by human experts.

Our contributions. Addressing the aforementioned challenges, this paper proposes a unified attribution framework that integrates heterogeneous foundation models, multifaceted attribution algorithms, and Large Language Models. The objective of this framework is to distill highdimensional latent representations into low-dimensional physical evidence, ultimately translating this evidence into semantic reports that are readily interpretable by clinical experts. The main contributions of this study are summarized as follows:

Integration of a Unified Cross-Architecture Attribution Framework. We have built a generalpurpose analysis platform compatible with various advanced foundation model architectures (such as BIOT, LaBraM, EEGPT, and EEGMamba) and multiple types of post-hoc attribution algorithms, including input gradient-based, perturbation-based, and feature activation-based methods. Through this platform, we are able to systematically compare the impact of different pre-training strategies on feature capture preferences (e.g., spatial localization accuracy and time-frequency sensitivity).

Quantified Fidelity Analysis and Model Diagnosis Paradigm. We introduced a population-level fidelity analysis paradigm based on Progressive Occlusion and the Area Over Perturbation Curve (AOPC) [9]. By quantitatively comparing the rate and significance of the decline in model confidence following the removal of the top-k core features, we validate the causal consistency between attributional evidence and the model’s underlying logic. This process reveals the neurophysiologically aligned features learned by the model, as well as any potential dependencies on erroneous patterns.

Preliminary Exploration of Neuro-Symbolic Semantic Reasoning Pathways. We innovatively utilized the structured evidence (in JSON or image format) generated by multi-dimensional attribution as prompts for a Large Language Model (LLM). Leveraging the LLM’s medical knowledge base, we achieved an automated translation from low-level physical representations to high-level clinical semantics, thereby providing neuroscientists with intuitive and transparent decision-making reports .

Our code is available on Github<sup>1</sup>.

## 2 RELATED WORK

## 2.1 Evolution of EEG Foundation Models

Early research on automated Electroencephalography (EEG) decoding primarily relied on supervised learning architectures designed for specific tasks. Convolutional Neural Networks (CNNs), such as EEGNet [10] (proposed by Lawhern et al.) and DeepConvNet [11] (proposed by Schirrmeister et al.), achieve efficient extraction of spatiotemporal features by introducing local constraints across spatial and temporal dimensions. However, these models were typically constrained by the limited scale of specific datasets, making it difficult to overcome the challenges posed by the extreme inter-individual variability and non-stationarity inherent in EEG signals.

In recent years, inspired by the paradigm of large-scale pre-training, EEG Foundation Models (FMs) have emerged at the forefront of the field. BENDR [12], proposed by Kostas et al., marked the inception of large-scale self-supervised pre-training for EEG. Subsequently, Yang et al. [2] introduced a channel-level spectral tokenization mechanism through BIOT, enabling a unified representation of heterogeneous lead data. Inspired by the Masked Autoencoder (MAE) framework, LaBraM [13] utilizes a neural tokenizer to map continuous signals into a discrete latent space, thereby significantly improving the modeling of long-range neurodynamic features. Furthermore, EEGMamba [4], based on Selective State Space Models, further optimized the computational efficiency of sequence modeling. Although Foundation Models have achieved State-of-the-Art (SOTA) performance across various tasks, their massive parameter counts and patch-based discretization logic endow their internal knowledge representations with highly “black-box” characteristics; consequently, the alignment between their decision-making logic and established neurophysiological principles remains an area requiring systematic investigation.

## 2.2 Post-hoc Attribution and XAI Applications in EEG

Applying post-hoc attribution algorithms to EEG models to identify decision-relevant features across spatial and temporal dimensions has become a focal point of research. Sturm et al. [14] were the first to propose integrating Deep Neural Networks (DNNs) with Layer-wise Relevance Propagation (LRP) for EEG data analysis, successfully revealing feature patterns that align with neurophysiological principles. Schirrmeister et al. [11], through systematic feature visualization and perturbation analysis, revealed the heavy reliance of CNNs on specific frequency bands (such as Alpha and Beta rhythms) during the decoding process. This finding established the efficacy of deep learning models in extracting features within the frequency domain.

With the evolution of algorithms, SHAP (Shapley Additive Explanations) as proposed by Lundberg et al. [15] has been widely adopted to interpret complex neurophysiological decision-making processes, largely due to its robust foundation in game theory. Shawly et al. [16] integrated novel attention modules into CNNs and utilized SHAP to introduce quantitative interpretability metrics, thereby refining the logic behind the extraction of EEG signal features. Islam et al. [17] employed the LIME algorithm to interpret the behavior of a stroke prediction model, successfully identifying biological features that contribute significantly to clinical diagnosis. In the realm of clinical applications, Khan et al. [18] proposed a framework for epilepsy detection that combines machine learning with XAI techniques, thereby significantly enhancing the system’s credibility in supporting clinical diagnosis. Addressing emerging foundation model architectures, Madsen et al. [19] introduced the Concept Activation Vectors (CAV) technique into the field of EEG analysis; by leveraging external labeled data and anatomical definitions to construct conceptual representations, they successfully improved the interpretability of large-scale Transformer models such as BENDR. However, most existing studies focus on single-model designs tailored to specific tasks. They lack a standardized auditing framework capable of universally adapting to heterogeneous foundation models and translating weight distributions into physiologically meaningful evidence.

## 2.3 Neuro-Symbolic Integration and Semantic Reasoning via LLMs

Translating low-level attribution maps into high-level semantic logic is crucial to bridge the interpretability gap. With the maturation of LLMs, their potential to assist in medical decision-making and scientific logical reasoning has been widely demonstrated. A study by Singhal et al. [20] demonstrates that LLMs possess extensive clinical medical knowledge and exhibit exceptional potential for diagnostic reasoning. In the field of EEG, Thapa et al. [21] introduced EEG-GPT, conducting a preliminary exploration into the feasibility of using LLMs to interpret EEG feature descriptors. A current limitation is that, while post-hoc attribution techniques can generate high-quality spatial topological maps and time-frequency evidence spectra, deriving logically consistent diagnostic conclusions from these visualizations remains a significant cognitive challenge for non-clinician experts. This paper aims to address this challenge through neuro-symbolic integration: first, relevant physical evidence (such as specific spatial polarities and time-frequency patterns) is tokenized into structured JSON data or direct input; subsequently, leveraging the medical knowledge base of LLM, this data is transformed into an expert report in natural language. This closed-loop process not only enhances the interpretability of the underlying model but also offers a novel theoretical paradigm for constructing end-to-end, auditable EEG AI systems.

## 3 METHOD

## 3.1 Preliminaries

EEG Signal Modeling and Characteristics of Foundation Model Architecture. Let $\textbf { X } \in \ \mathbb { R } ^ { C \times T }$ represent a multichannel EEG recording, where $C$ is the number of electrodes and $T$ is the number of time samples. An EEG foundation model (FM) is a pre-trained backbone $f _ { \theta }$ that maps X into a high-dimensional representation space H. In downstream applications $( \mathrm { e . g . }$ , Mumtaz2016 for depression detection or TUAB for abnormality screening), the model is typically appended with a task-specific head $h _ { \phi }$ to produce a prediction $y = h _ { \phi } ( f _ { \theta } ( \mathbf { X } ) ) ,$ ). For classification, y denotes the class probability after softmax normalization. Unlike traditional end-to-end CNNs, current FMs for EEG employ large-scale self-supervised pre-training and incorporate the following three key architectural features:

Tokenization and Patching: These models utilize patchbased representations, segmenting the raw signal X into patches along the time axis and learning features at the patch level.

Encoder Architectures: FMs primarily leverage Transformer encoders to capture global attention across tokens, or Selective State Space Models (SSMs) like EEGMamba to model long-range dependencies with linear complexity.

Latent Space Representation: The decision-making process of these models relies on high-dimensional, abstract representations within the encoder’s hidden layers; consequently, the specific contributions of input channels and time segments to the prediction results cannot be directly interpreted.

Paradigms of Post-hoc Attribution. To systematically audit the decision-making process of the EEG foundation models, we integrate six representative algorithms, categorized into three mathematical paradigms:

(1) Input Gradient-based Paradigm (IG, Gradient-SHAP): These methods quantify feature importance by backpropagating the model’s prediction to the input space. The core idea is to measure the sensitivity of the output f(X) via gradients.

Integrated Gradients (IG) calculates the integral of gradients along a straight-line path from a baseline $\mathbf { \bar { X } ^ { \prime } }$ to the input $\mathbf { X } ,$ ensuring the Completeness axiom:

$$
\phi _ { I G } ( \mathbf { X } ) = ( \mathbf { X } - \mathbf { X ^ { \prime } } ) \times \int _ { \alpha = 0 } ^ { 1 } \frac { \partial f ( \mathbf { X ^ { \prime } } + \boldsymbol { \alpha } ( \mathbf { X } - \mathbf { X ^ { \prime } } ) ) } { \partial \mathbf { X } } d c\tag{1}
$$

Gradient-SHAP extends IG by sampling multiple interpolated paths between the baseline and input, averaging gradients across these samples to reduce variance and provide a smoother, more stable attribution map for high-dimensional EEG patches.

(2) Perturbation-based Paradigm (Occlusion, SHAP, LIME): These are model-agnostic approaches that treat the FM as a black box, estimating importance by observing the response to input variations.

Occlusion systematically masks temporal patches or channels to measure the direct drop in confidence $\Delta y = f ( \mathbf { X } ) - f ( \mathbf { X } _ { m a s k e d } )$

![](images/9cf60cbe50ffdbf61c65003a7185c9315a77fd16b5819d99f8a6af51c32ad4da.jpg)  
Fig. 1: Overview of our proposed EEG-Xplain framework. It comprises four components: (a) a unified interface for various models and tasks to generate diverse types of model prediction attributions, with an extensible list of supported models and tasks; (b) projects attribution results onto three distinct dimensions: spatial, temporal, and frequency-band; (c) supports both single-sample and population-level analyses to verify the reliability of the attribution results; and (d) integrates the multidimensional analysis and validation results as joint inputs, leveraging the medical knowledge of large-scale models to align the attributions and analyses with physiological significance, ultimately generating a structured report for expert review.

SHAP leverages cooperative game theory to assign each feature a value representing its marginal contribution across all possible feature combinations:

$$
\phi _ { i } = \sum _ { S \subseteq \mathcal { F } \setminus \{ i \} } w ( | S | ) [ f ( S \cup \{ i \} ) - f ( S ) ]\tag{2}
$$

where $S \subseteq { \mathcal { F } } \backslash \{ i \}$ denotes all feature subsets excluding $i , w ( | S | )$ is a weighting function, and $f ( S \cup i ) -$ $f ( S )$ measures the marginal contribution of adding feature i. We use KernelSHAP for efficient approximation.

LIME approximates the FM locally by training an interpretable surrogate model $g \left( \mathrm { e . g . } \right.$ , a linear regressor) on perturbed samples in the neighborhood of X.

(3) Feature Activation-based Paradigm (Grad-CAM): Targeting models with convolutional or self-attention layers (e.g., EEG-Conformer), Grad-CAM computes a coarse localization map via a weighted combination of feature maps, where the weights are derived from the globally pooled gradients of the final bottleneck layer:

$$
{ \cal L } _ { G r a d - C A M } ^ { c } = \mathrm { R e L U } \left( \sum _ { k } \alpha _ { k } ^ { c } { \bf A } ^ { k } \right)\tag{3}
$$

where $\alpha _ { k } ^ { c }$ denotes the importance weight of feature map k for class $c ,$ computed via global average pooling of the gradients, and $\mathbf { A } ^ { \dot { k } }$ represents the activation of the k-th feature map.

## 3.2 System Architecture and Design Philosophy

The proposed framework, as illustrated in Fig 1, aims to bridge the interpretability gap between the highdimensional latent representations of EEG foundation models and tangible neurophysiological evidence. The core design philosophy is based on the concept of Causal Distillation: reducing millions of neural parameters into a few key spatial-temporal-frequency signatures that an expert can audit. The framework follows a hierarchical pipeline: (1) Unified Model-Task Adaptation and Explanation Interface: An abstraction layer encapsulates model-specific differences to allow uniform processing of diverse architectures within a shared attribution space. It exposes a standardized attribution API, providing model-agnostic input-output specifications for downstream interpretability methods.

(2) Spatiotemporal-Frequency Attribution Analysis: Building upon the unified interface, the framework integrates multiple post-hoc attribution algorithms (based on input gradients, perturbations, and feature activations) to generate attribution results in parallel across spatial, temporal, and frequency dimensions. The spatial dimension identifies top-k channels and topological maps; the temporal dimension produces importance heatmaps overlaid on raw waveforms; and the frequency dimension quantifies the relative contributions of classic EEG rhythms via frequencydomain perturbations. This constructs a hierarchical chain of evidence ranging from coarse-grained brain regions to fine-grained spatio-temporal-frequency features. (3) Fidelity Evaluation: To ensure reliability, the framework introduces a population-level evaluation mechanism, such as the Area Over the Perturbation Curve (AOPC), to quantify confidence decline during progressive masking. This mechanism provides a rigorous basis for validating attribution outcomes. (4) LLM-assisted Clinical Interpretation: Structured attribution results (key channels, time intervals, frequency bands, and their relative contributions) and assessment metrics are injected into a LLM via structured prompts. The LLM automatically generates natural language interpretations consistent with neurophysiological contexts. By translating abstract attribution values into readable evidence statements for clinical experts, it completes the final distillation loop from model decisions to expert-reviewable diagnostic hypotheses.

## 3.3 Unified Model-Task Adaptation and Explanation Interface

Deep learning models for EEG data typically exhibit significant heterogeneity in their input representations; for instance, different models or tasks may employ varying numbers of channels, temporal segmentation strategies, positional encoding schemes, and feature extraction layer architectures. Directly adapting specific attribution methods to each model or task not only introduces immense implementation complexity but also compromises the comparability of the resulting explanations. To address this challenge, this paper unifies the modeling of task configurations, model adaptations, and attribution interfaces, thereby enabling the interpretation processes for diverse models to be executed within a single, consistent abstraction layer. Let the EEG input be denoted as

$$
\mathbf { X } \in \mathbb { R } ^ { C \times T } ,\tag{4}
$$

where $C$ represents the number of channels and $T$ represents the temporal length. For models employing a blockbased representation, this can be further expressed as

$$
\mathbf { X } _ { p } \in \mathbb { R } ^ { C \times P \times L } ,\tag{5}
$$

where $P$ is the number of temporal blocks and $L$ is the length of each block, satisfying $\boldsymbol { \bar { T } } = \boldsymbol { P } \cdot \boldsymbol { L }$ . The core concept of unified adaptation is to define a mapping from the original input space to the model input space:

$$
\phi : \mathbb { R } ^ { C \times T }  \mathbb { R } ^ { C \times P \times L } .\tag{6}
$$

Based on this foundation, the models can be uniformly represented as

$$
\hat { \mathbf { y } } = f ( \phi ( \mathbf { X } ) ) ,\tag{7}
$$

where $f ( \cdot )$ denotes the deep EEG network under interpretation. For a given sample and target class, various attribution methods are ultimately constrained to produce a unified joint attribution matrix

$$
\mathbf { A } \in \mathbb { R } ^ { C \times P } ,\tag{8}
$$

Within the unified output space, spatial importance and temporal importance are derived through marginal aggregation, defined as follows:

$$
s _ { c } = \sum _ { p = 1 } ^ { P } \mathbf { A } _ { c , p } , \quad t _ { p } = \sum _ { c = 1 } ^ { C } \mathbf { A } _ { c , p } , \quad c = 1 , \ldots , C , p = 1 , \ldots , P .\tag{9}
$$

Here, $s _ { c }$ quantifies the overall contribution strength of channel c to the current prediction, while $t _ { p }$ characterizes the significance of information surrounding time block $p$ to the model’s discriminative decision. By employing this unified aggregation scheme, comparisons and statistical analyses can be conducted across identical channel and temporal dimensions, even when the underlying attribution methods differ.

## 3.4 Spatiotemporal-Frequency Attribution Analysis

Upon obtaining the unified attribution matrix, the framework conducts an in-depth analysis of the model’s discriminative basis across three dimensions: space, time, and frequency. The primary objective of this module is to transform raw, high-dimensional, and method-specific attribution results into multi-view representations with neurophysiological interpretability, thereby identifying the key brain regions, time segments, and frequency components upon which the model relies.

## 3.4.1 Spatial Attribution Analysis

Spatial dimension analysis is grounded in the channel importance vector $\mathbf { s } = [ s _ { 1 } , \ldots , s _ { C } ]$ , which serves to characterize the specific regions on the scalp topography to which the model directs its attention. Since EEG channels correspond to fixed electrode positions, the vector s can be mapped onto a standard scalp coordinate system to form a continuous topological distribution. Figure 2 illustrates the electrode distribution for a monopolar montage based on the standard 10-10 system; it comprises approximately 60 electrode locations, categorized into five anatomical brain regions: frontal, central, temporal, parietal, and occipital. For group-level analysis: Assuming that there are N samples in total, the importance of group-level channel can be constructed by calculating the sample mean:

$$
\bar { s } _ { c } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { c } ^ { ( n ) } .\tag{10}
$$

Furthermore, these results can be aggregated at the level of anatomical regions. If $\Omega _ { r }$ denotes the set of channels corresponding to the r-th brain region, the regional contribution is defined as:

$$
S _ { r } = \sum _ { c \in \Omega _ { r } } \bar { s } _ { c } .\tag{11}
$$

This statistic can be utilized to compare the relative contributions of different brain regions to the model’s decisionmaking process, and it also facilitates the analysis of hemispheric lateralization.

Standard 10-10 Electrode Layout (Colored by Brain Region)

![](images/1eaf177240ad71b1443c21e819c0178fff9e2b697d1d63547e4a5360fc208309.jpg)  
Fig. 2: Standard 10-10 electrode layout colored by brain region.

## 3.4.2 Temporal Attribution Analysis

Analysis along the temporal dimension is based on the temporal structure of the attribution matrix.

At the sample level, each row of the attribution matrix $\textbf { A } \in \mathbb { R } ^ { C \times P }$ contains $P$ patch-level attribution scores for the respective channel. Each patch p corresponds to the time interval $[ ( p - 1 ) \cdot S , ( p \bar { \textrm { - } } 1 ) \cdot S + w ]$ on the raw waveform time axis (where w is the patch width and $S$ is the stride). By mapping these patch-level scores back to their corresponding time intervals, we enable patch-aligned temporal attribution visualization at the channel level.

At the population level, for event-related tasks where temporal structure is preserved across samples, we compute temporal importance by aggregating attribution values across both channels and samples:

$$
\bar { t } _ { p } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } t _ { p } ^ { ( n ) } , \quad t _ { p } ^ { ( n ) } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } A _ { c , p } ^ { ( n ) }\tag{12}
$$

This yields a population-level temporal importance vector $\bar { \textbf { t } } \in \mathbf { \widetilde { R } } ^ { P }$ , characterizing the model’s average attention pattern along the time axis.

Temporal attribution reveals whether the model has learned to selectively focus on specific time intervals. For event-based tasks (e.g., epileptic discharge detection), highcontribution patches should cluster within time windows containing sharp waves or spike-and-slow-wave complexes. Conversely, for resting-state tasks $( \mathrm { e . g . , }$ normal vs. abnormal classification), temporal contributions are typically more uniform, reflecting the model’s reliance on global rhythmic features rather than transient events. Figure 3 presents the results obtained from a sample taken in a healthy subject from the Mumtaz2016 dataset: the topk high-contribution patches for channels O1 and O2 are evenly distributed along the time axis, indicating that the model’s decision for this sample does not rely heavily on any single local time segment.

![](images/800a3852c3a0a99c97d2d4ce75bb097c832e73940d50c71eeb0fd5027ff2a92d.jpg)  
Fig. 3: Visualization of temporal attribution for a TP sample from a healthy subject in the Mumtaz2016 dataset.

## 3.4.3 Frequency Band Attribution Analysis

Frequency band attribution aims to investigate which neural oscillation rhythms the model relies on. We define the frequency bands $\overset { \mathcal { B } } { B } = \left\{ \delta , \theta , \alpha , \beta , \gamma \right\} ( 0 . 5 \mathrm { - } 4 , 4 \mathrm { - } 8 , 8 \mathrm { - } 1 3 , 1 3 \mathrm { - } 3 0 ,$ 30–45 Hz respectively). The framework quantifies the causal contribution of each frequency band through band ablation.

For each channel c and frequency band b, a band-ablated baseline signal $\boldsymbol { x } _ { c , b } ^ { \prime }$ is constructed: a band-stop filter is applied to channel c to remove the energy components within band b while preserving the remaining frequency components. Taking Integrated Gradients as an example, the attribution is computed via path integration from this baseline to the original signal:

$$
\mathrm { B A } _ { c , b } = \sum _ { t } ( \boldsymbol { x } _ { c , t } - \boldsymbol { x } _ { c , b , t } ^ { \prime } ) \cdot \int _ { 0 } ^ { 1 } \frac { \partial F _ { k } } { \partial \boldsymbol { x } _ { c , t } } ( \boldsymbol { x } _ { c , b } ^ { \prime } + \alpha ( \boldsymbol { x } _ { c } - \boldsymbol { x } _ { c , b } ^ { \prime } ) )\tag{dα}
$$

(13)

where t indexes all time points of channel c. This carries a clear causal interpretation: $\mathtt { B A } _ { c , b }$ quantifies the change in the model’s prediction confidence for the target class if the oscillatory activity of band b were removed from channel c. By aggregating across samples, population-level bandchannel attribution is obtained:

$$
\overline { { \mathbf { B } \mathbf { A } } } _ { c , b } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { B } \mathbf { A } _ { c , b } ^ { ( n ) }\tag{14}
$$

This produces a population-level band-channel attribution matrix $\overline { { \mathbf { B A } } } \in \mathbb { R } ^ { C \times | \mathbf { \vec { B } } | }$ . It allows visualization of spatial distribution patterns of various rhythms through band-specific topographic maps. For example, if the model relies on alpha rhythms in posterior regions for classification, $\overline { { \mathbf { B A } } } _ { : , \alpha }$ should exhibit significant positive values in occipital channels (O1, $\scriptstyle \mathrm { O } 2 , \mathrm { O z } )$

## 3.5 Fidelity Evaluation

Does the importance ranking generated by the attribution method faithfully reflect the model’s internal decisionmaking logic? The framework employs an interventionbased fidelity evaluation paradigm to verify this. The core idea is that if the attribution correctly identifies features causally contributing to the decision, then systematically removing high-attribution features should cause a more significant drop in performance than random removal.

## 3.5.1 Perturbation-Based Faithfulness

The framework conducts cumulative masking experiments on the attributed features. Let $\pi$ denote the ranking of features by attribution value in descending order. We progressively eliminate information from the top k features and observe the trajectory of the model’s output logit:

$$
F ^ { ( k ) } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } F _ { k } \big ( \tilde { \mathbf { x } } _ { \pi _ { 1 : k } } ^ { ( n ) } \big )\tag{15}
$$

where $\tilde { \mathbf { x } } _ { \pi _ { 1 : k } } ^ { ( n ) }$ represents the input after masking the top k features. For masking in the spatial dimension, channel-wise mean replacement is employed, whereas Gaussian noise with matched mean and variance is used for the temporal dimension. The confidence curve $\{ F ^ { ( 0 ) } , F ^ { ( 1 ) } , \ldots , F ^ { ( \dot { K } ) } \}$ intuitively illustrates the causal validity of the attribution ranking: a steeper decline indicates that the attribution more precisely identifies the features critical to the decision.

Based on this, AOPC is used to quantify overall faithfulness:

$$
\mathrm { A O P C } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left[ F ^ { ( 0 ) } - F ^ { ( k ) } \right]\tag{16}
$$

Additionally, random-order masking serves as a baseline, with $\mathrm { A O P C } _ { \mathrm { g a i n } } = \mathrm { A O P C } _ { \mathrm { a t t r } } - \mathrm { A O P C } _ { \mathrm { r a n d } }$ quantifying the gain of the attribution ranking relative to the random baseline.

AOPC evaluates the quality of the global ranking under cumulative masking. To complement this, the framework incorporates feature-wise independent intervention validation: independent single-feature masking is performed for each feature dimension (channel c or temporal patch p) to calculate the actual logit drop $\Delta _ { i }$

$$
\Delta _ { i } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left[ f ( x ^ { ( n ) } ) - f ( \tilde { x } _ { i } ^ { ( n ) } ) \right]\tag{17}
$$

where $\tilde { x } _ { i } ^ { ( n ) }$ denotes sample n with feature i masked.

Spearman rank correlation is then computed between the attribution vector and the intervention impact vector:

$$
\rho _ { \mathrm { s p a t i a l } } = r _ { s } ( { \bar { \mathbf { s } } } , \ \Delta ) , \quad \rho _ { \mathrm { t e m p o r a l } } = r _ { s } ( { \bar { \mathbf { t } } } , \ \Delta )\tag{18}
$$

where ¯s represents population-level channel importance and <sup>¯</sup>t represents population-level temporal importance. A value of $\rho$ significantly greater than zero indicates monotonic consistency between the attribution ranking and the model’s true feature dependencies. This metric enables cross-method comparisons, facilitating the recommendation of the most faithful attribution method for each model-task combination.

## 3.5.2 Cross-Method Consistency

For the same model-task combination, discrepancies may arise between the population attribution vectors produced by different attribution methods. This reflects the inherent assumption biases of the attribution methods themselves [22]. The framework introduces cross-method consistency as a supplementary metric for attribution robustness by calculating the Spearman rank correlation between all pairs of methods:

$$
\rho _ { i j } = r _ { s } \big ( \bar { \mathbf { s } } ^ { ( i ) } , \ \bar { \mathbf { s } } ^ { ( j ) } \big ) , \quad \bar { \boldsymbol { \rho } } = \frac { 2 } { L ( L - 1 ) } \sum _ { i < j } \rho _ { i j }\tag{19}
$$

where $\bar { \mathbf { s } } ^ { ( i ) }$ is the population-level channel importance vector produced by the i-th attribution method, and L is the number of attribution methods. A high consistency score $\bar { \rho }$ indicates that the attribution conclusions are insensitive to the choice of method, implying strong robustness of the results. In cases where discrepancies arise among different methods, the aforementioned AOPC and Spearman-based faithfulness metrics can serve as references to select the method with the best faithfulness as the recommended explanation for the specific model-task combination.

## 3.6 LLM-assisted Clinical Interpretation

The aforementioned analysis distills model decisions into low-dimensional attribution maps. However, as attribution patterns are merely numerical rankings, their neurophysiological significance requires interpretation through domain knowledge. To address this, the framework incorporates an LLM-assisted interpretation module. It leverages the medical and neuroscience knowledge internalized by the LLM to assess the physiological plausibility of attribution results and align them with clinical semantics. Attribution outputs are presented to the LLM as either structured data or visualizations, accompanied by a consistent set of domainspecific prompt constraints. These constraints require the LLM to perform reasoning within a neuroelectrophysiological framework, cross-referencing attribution patterns with the functional specialization of brain regions and the physiological significance of neural rhythms. Simultaneously, the LLM must integrate information across dimensions and assign confidence scores to uncertain inferences.

This design aims to bridge the semantic gap between numerical attribution and clinical understanding, allowing clinicians without computational backgrounds to comprehend the rationale behind model decisions. Simultaneously, physiological alignment assessments grounded in domain knowledge serve as supplementary evidence for the validity of the attributions. The output of this interpretive overlay does not influence fidelity assessments or alter attribution rankings. Instead, it functions as a downstream semantic interface within the attribution pipeline.

## 4 EXPERIMENTAL SETUP

To validate the effectiveness of the proposed multidimensional EEG attribution analysis framework, we conducted systematic experiments on different models ,tasks and methods.

Base Models. As shown in Table 1, we selected five representative baseline EEG models for interpretability analysis, comprising four Transformer-based architectures (CBraMod [23], EEGPT, LaBraM, and BIOT) and one SSMbased architecture (EEGMamba). All models were finetuned on downstream tasks using their respective official pre-trained weights.

Datasets and Tasks. As shown in Table 2, we selected three datasets covering state recognition and event classification tasks:

(1) Temple University Hospital Abnormal EEG Corpus (TUAB) [24]: A binary classification task (normal vs. abnormal) using resting-state clinical EEG recordings across 2,993 sessions; (2)Mumtaz2016 [25]: A classification task distinguishing healthy controls (HC, $n = 3 0 )$ from patients with major depressive disorder (MDD, $n ~ = ~ 3 4 ) .$ , utilizing 19-channel resting-state EEG data from 64 subjects in total; (3) Temple University Hospital EEG Events Corpus (TUEV) [24]: A six-class classification task for fine-grained EEG micro-event detection, with annotations covering specific electrographic events such as spike-and-wave (SPSW) complexes, generalized periodic epileptiform discharges (GPEDs), and artifacts.

We demonstrate the various processes of the framework using Mumtaz, while conducting quantitative analyses using TUAB and TUEV. After undergoing model-specific preprocessing workflows, these datasets are used for model fine-tuning and inference.

Attribution Methods. The framework integrates six post-hoc attribution methods (Integrated Gradients (IG) [26], SHAP [15], GradientSHAP [15], [27], LIME [28], Occlusion [29], and GradCAM [30])across three paradigms: input gradient-based, perturbation-based, and feature activation-based approaches.

Evaluation Protocol. All analyses are based on population-level statistics. Specifically, evaluation metrics are calculated by averaging the attributions of true positive (TP) samples(defined as those with a model output probability of at least 0.7 for the target class). Spatial fidelity is assessed by computing Spearman’s $\rho$ between attribution rankings and single-channel masking impact. Temporal fidelity is evaluated similarly by masking time segments sequentially. Frequency-band attribution is performed by attributing power within specific bands to generate joint spatial-frequency attribution maps. Consistency across different methods is quantified using the average Spearman rank correlation coefficient of the attribution rankings.

TABLE 1: Overview of EEG foundation models supported by the proposed framework.
<table><tr><td>Model</td><td>Architecture</td><td>Pre-training Strategy</td></tr><tr><td>CBraMod</td><td>Transformer</td><td>Masked EEG Modeling</td></tr><tr><td>LaBraM</td><td>Transformer</td><td>Masked EEG Modeling</td></tr><tr><td>EEGPT</td><td>Transformer</td><td>Autoregressive Pre-training</td></tr><tr><td>EEGMamba</td><td>SSM (Mamba)</td><td>Masked EEG Modeling</td></tr><tr><td>BIOT</td><td>Transformer</td><td>Multi-dataset Pre-training</td></tr></table>

Note: The framework provides a unified adapter interface, supporting attribution analysis across heterogeneous architectures.

## 4.1 Resting-State Task: Spatial and Frequency-Band Attribution Analysis

## 4.1.1 Channel-Level Spatial Fidelity

Tables 3 and 4 present the channel-wise spatial fidelity of various model-method combinations on the Mumtaz and TUAB datasets, respectively.

Mumtaz Dataset. On the Mumtaz dataset (Table 3), CBraMod’s attributions for the HC class consistently pointed to the occipital regions O1, O2, and Pz (IG: $\rho =$ $0 . 9 8 1 , p < . 0 0 1 )$ , while for the MDD class, they focused on the frontal regions F3 and F8 and the central region C4. Figure 4 displays the group-averaged attribution topographies and faithfulness curve for the two categories using the IG method. Differences in lateralization are evident between the groups: the HC (Healthy Control) group exhibits strong positive attribution in the occipital region, whereas the MDD (Major Depressive Disorder) group shows strong positive attribution in the fronto-central region.The attributionguided curves show a steeper decline than random baselines across both classes, which indicates that the attribution ranking effectively captures feature importance patterns aligned with the model’s decision process.

From the perspective of clinical interpretability, the occipital attribution pattern observed in the healthy control (HC) group aligns with the physiological characteristics of resting-state EEG in healthy individuals, where the occipital lobe serves as a core region of activation [31]. Meanwhile, the prominent frontal lobe attribution features in the major depressive disorder (MDD) group are consistent with previous findings regarding abnormal activity and impaired neural function in the frontal brain regions of patients with depression [32]. EEGMamba also demonstrated good spatial fidelity on the Mumtaz dataset (HC class Occlusion: $\rho = 0 . 9 1 1$ ; MDD class LIME: $\rho = 0 . 9 3 0 )$ . Furthermore, its top-5 channels for the MDD class included frontal-temporal channels such as F8, Fp1, and T5, further corroborating the clinical consistency of the attributions across models.

TUAB Dataset. On the TUAB dataset (Table 4), CBraMod and EEGMamba demonstrate high attribution fidelity. Taking CBraMod as an example, both IG and GradientSHAP achieve $\rho = 0 . 9 7 9 \ ( p < . 0 0 1 )$ for the abnormal class. The top-5 channels (F8-T4, FP2-F8, T3-T5, T5-O1, T6- O2) are highly concentrated in the temporal region, consistent with the temporal lobe abnormal discharge patterns characteristic of abnormal EEG [33]. The top-5 channels identified by five of the methods (excluding GradCAM) are identical, indicating high cross-method consistency in attribution results. For the normal class, the models focus on channels in the central-parieto-occipital regions (C3-P3, P4-O2), aligning with the dominance of posterior alpha rhythms in normal resting-state EEG [34].

TABLE 2: Summary of the benchmark datasets used for explainability evaluation.
<table><tr><td>Dataset</td><td>Type</td><td>Description</td><td>Classes</td></tr><tr><td>Mumtaz2016</td><td>State recognition</td><td>Binary classification of clinical EEG into major depressive disor- der (MDD) vs. healthy control (HC) states</td><td>0: HC 1: MDD</td></tr><tr><td>TUAB</td><td>State recognition</td><td>Binary classification of clinical EEG into normal vs. pathological states.</td><td>0: Normal 1: Abnormal</td></tr><tr><td>TUEV</td><td>Event detection</td><td>Segment-level classification of clinical EEG into six neurological event types.</td><td>0: SPSW 1: GPED 2: PLED 3: EYEM 4: ARTF 5: BCKG</td></tr></table>

Note: SPSW = Spike and Sharp Wave; GPED = Generalized Periodic Epileptiform Discharges; PLED = Periodic Lateralized Epileptiform Discharges; EYEM = Eye Movement; ARTF = Artifact; BCKG = Background.

CBRAMOD | mumtaz | class 0 (TP) | IG | n=100 samples conf 0.7 Red=positive attribution, Blue=negative attribution  
![](images/9899a14df6090aac9a8998acddb06a5befc0fc5b9014e601c00dcac6cc738de9.jpg)

CBRAMOD | mumtaz | class 1 (TP) | IG | n=100 samples conf 0.7 Red=positive attribution, Blue=negative attribution  
![](images/7877f4c1de7565e3ea67ea0c5d7c059c73872addc717bfd0cc4fa01c429a2fdc.jpg)

(a) Class 0 (Control) – Channel Attribution  
![](images/03a2b9b2de34fc9a7d12a6ff3777b363c0748894f09c85558015e2ae7d5aea1e.jpg)  
(c) Class 0 (Control) – Spatial Faithfulness

(b) Class 1 (MDD) – Channel Attribution  
![](images/b74b2c7aa015497a23954066a054c14770435618fd844d8f29654678a667a6ce.jpg)  
(d) Class 1 (MDD) – Spatial Faithfulness  
Fig. 4: Channel-level attribution analysis on the Mumtaz dataset using IG with CBraMod. Top row: topographic maps of channel importance. Bottom row: spatial faithfulness evaluation.

TABLE 3: Attribution method comparison on mumtaz2016 (MDD task), true positive samples. Boldface: channels in top-5 of ${ \ge } 4 / 5$ methods (excl. GradCAM). Underline: clinically relevant channels (HC: posterior alpha $- { \mathrm { O 1 } } / { \mathrm { O 2 } } / { \mathrm { P z } } ;$ MDD: frontal asymmetry — F3/F4/F7/F8/Fp1/Fp2). <sup>⋆</sup>Recommended. $^ { * * * } p < . 0 0 1 , ^ { * * } p < . 0 1 , ^ { * } p < . 0 \dot { 5 }$
<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">Class 0 — HC (N=100)</td><td colspan="3">Class 1 — MDD (N=100)</td></tr><tr><td>ρ</td><td>Consist.</td><td>Top-5 Channels</td><td>ρ</td><td>Consist.</td><td>Top-5 Channels</td></tr><tr><td rowspan="6">CBraMod (Transformer)</td><td>IG*</td><td> $0 . 9 8 1 ^ { * * * }$ </td><td>0.712</td><td> $\mathbf { O 1 } , \mathbf { O 2 } , \mathbf { P z } , \mathrm { T 3 } , \mathrm { T 6 }$ </td><td> $0 . 9 7 2 ^ { * * * }$ </td><td>0.902</td><td>C3, C4, F3, F4, F8</td></tr><tr><td>GradientSHAP</td><td> $0 . 9 7 9 ^ { * * * }$ </td><td>0.709</td><td>O1, O2, Pz, T3, T6</td><td> $0 . 9 6 8 ^ { * * * \star }$ </td><td>0.907</td><td>C3, C4, F3, F4, F8</td></tr><tr><td>SHAP</td><td> $0 . 9 6 1 ^ { * * * }$ </td><td>0.702</td><td>O1, O2, Pz, T3, T6</td><td> $0 . 9 3 3 ^ { \ast \ast \ast }$ </td><td>0.901</td><td>C4, Cz, F3, F4, F8</td></tr><tr><td>LIME</td><td> $0 . 9 5 3 ^ { * * * }$ </td><td>0.695</td><td>O1, O2, Pz, T3, T6</td><td> $0 . 9 4 9 ^ { * * * }$ </td><td>0.874</td><td>C4, F3, F8, Fp1, Fp2</td></tr><tr><td>Occlusion</td><td> $0 . 9 4 7 ^ { * * * }$ </td><td>0.700</td><td>O1, O2, T3, T4, T6</td><td> $0 . 8 7 7 ^ { * * * }$ </td><td>0.839</td><td>C4, Cz, F3, Fp1, Fp2</td></tr><tr><td>GradCAM</td><td>-0.339</td><td>-0.355</td><td>C3, F4, F8, Pz, T3</td><td> $0 . 7 0 2 ^ { * * * }$ </td><td>0.728</td><td>C4, Cz, F3, F4, Fz</td></tr><tr><td rowspan="6">EEGMamba (SSM)</td><td>IG</td><td>0.532*</td><td>0.289</td><td>F7, Fp1, Fp2, T3, T4</td><td> $0 . 8 8 4 ^ { * * * }$ </td><td>0.756</td><td>C4, F3, F8, Fp1, T5</td></tr><tr><td>GradientSHAP</td><td> $0 . 7 9 3 ^ { * * * }$ </td><td>0.576</td><td>C4, F7, F8, O1, O2</td><td> $0 . 9 1 9 ^ { * * * }$ </td><td>0.795</td><td>C4, F8, Fp1, Fp2, T5</td></tr><tr><td>SHAP</td><td> $0 . 8 0 2 ^ { * * * }$ </td><td>0.559</td><td>C4, F8, Fp1, O1, O2</td><td> $0 . 9 4 6 ^ { * * * }$ </td><td>0.801</td><td>C4, F8, Fp1, Fp2, T5</td></tr><tr><td>LIME</td><td> $0 . 8 4 4 ^ { * * * }$ </td><td>0.584</td><td>C4, F8, O1, O2, Pz</td><td> $0 . 9 3 0 ^ { \ast \ast \ast \star }$ </td><td>0.806</td><td>C4, F8, Fp1, P4, T5</td></tr><tr><td>Occlusion*</td><td> $0 . 9 1 1 ^ { * * * }$ </td><td>0.568</td><td>C4, F8, O1, O2, Pz</td><td> $0 . 8 6 5 ^ { * * * }$ </td><td>0.757</td><td>C4, Cz, F8, Fp1, P4</td></tr><tr><td>GradCAM</td><td>-0.332</td><td>-0.266</td><td>C3, F3, F4, Fz, Pz</td><td>0.204</td><td>0.216</td><td>Cz, P3, P4, Pz, T4</td></tr></table>

In contrast, LaBraM and BIOT generally exhibited lower spatial fidelity (with most ρ values failing to reach statistical significance). This may be linked to differences in downstream task performance on TUAB, as attribution fidelity presupposes that the model has actually learned meaningful spatial patterns. Among all models, GradCAM performed the worst. It stems from the fact that GradCAM was originally designed for CNN spatial feature maps and lacks a direct spatial correspondence when applied to Transformer architectures based on patch embeddings.

## 4.1.2 Frequency-Band-Level Attribution Analysis

Figure 5 displays the joint spatial-frequency attribution topographic maps for the CBraMod model on the Mumtaz dataset, derived using Integrated Gradients (IG). This analysis, conducted by applying IG to the power features of five standard frequency bands, reveals the model’s reliance on specific frequency bands across different brain regions.

HC Class (Figure 5, top): For healthy controls (Class 0), discriminative contributions in the alpha (8–13 Hz) and lowbeta bands were concentrated in occipito-parietal regions, with relatively negative patterns in frontal areas. This aligns with the posterior alpha dominance typically observed in healthy subjects at rest. It is also consistent with MDDrelated patterns, particularly the attenuation of posterior alpha activity and alterations in frontal midline slow-waves, both of which have been repeatedly validated in the Mumtaz dataset [35]. The gamma (30–45 Hz) band exhibited diffuse and relatively high-intensity attribution. Given that scalp gamma activity is susceptible to interference from temporalis/frontalis electromyographic (EMG) signals and microsaccade-locked components under standard preprocessing conditions, the signal likely contains non-neural components.

MDD Class (Figure 5, bottom): Patients with depression (Class 1) exhibited positive contributions from frontotemporal regions in the ${ \bar { \delta } } / \theta$ bands and formed significant hotspots in the fronto-central region within the β band. These findings align with conclusions from recent machine learning-based EEG meta-analyses of major depressive disorder (MDD), which identified “elevated frontal slow-wave power” and “altered β-band powe” as robust and discriminative physiological markers [36], [37]. The attribution maps showed near-symmetrical patterns for frontal α activity, without explicitly displaying the Frontal Alpha Asymmetry (FAA) pattern [36]. This phenomenon suggests that the model relies on distributed power combinations across electrodes and frequency bands for classification. It likely implicitly integrates information related to FAA, rather than highlighting it as a distinct asymmetry metric at the attribution level.

Overall, both types of IG attribution across dominant frequency bands and spatial distributions align with mainstream findings from the Mumtaz dataset, validating the physiological interpretability of the model’s decision.

The value of frequency-band-level attribution analysis lies in its ability to go beyond simple channel ranking and reveal the spectral basis underlying the model’s decisions. While channel-level analysis uncovers where the model focuses, frequency-band-level analysis further elucidates what the model specifically perceives, thereby establishing a correspondence between the internal representations of “blackbox” models and specific electrophysiological mechanisms.

## 4.2 Event-based Tasks: Temporal Faithfulness Analysis

In the Table 5, ρ denotes the Spearman correlation coefficient between the ranking of segment attributions and the ranking of confidence drops following occlusion. A positive value close to 1 indicates that the temporal segments the model actually relies on align well with the attribution annotations. Since the number of patches $( n ~ = ~ 5 / 9 / 1 5 )$ varies across models, the absolute values of ρ cannot be directly compared. Larger values of n imply finer granularity in event localization and a relatively lower significance threshold. Therefore, the conclusions are drawn based on statistical significance.

TABLE 4: Channel-level explainability comparison on TUAB (Class 0: subject aaaaaiwz, Normal; Class 1: subject aaaaamft, Abnormal). Boldface: channels in top-5 of ≥4/5 methods (excl. GradCAM). Underline: clinically relevant channels (Normal: posterior alpha — O1/O2, P3/P4; Abnormal: temporal focus — T1/T2/T3/T4/T5/T6, F7/F8). ⋆ Recommended. $^ { * * * } p < . 0 \hat { 0 } 1 , ^ { * * } p < . 0 \hat { 1 } , ^ { * } p < . 0 5 .$
<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">Class 0 — Normal</td><td colspan="3">Class 1 — Abnormal</td></tr><tr><td>ρ</td><td>Consist.</td><td>Top-5 Channels</td><td>ρ</td><td>Consist.</td><td>Top-5 Channels</td></tr><tr><td rowspan="6">CBraMod (116/272)</td><td>IG*</td><td>0.712**</td><td>0.643</td><td>C3-P3, FP2-F4, P4-O2, T3-T5, T6-O2</td><td>0.979***</td><td>0.872</td><td>F8-T4, FP2-F8, T3-T5, T5-O1, T6-O2</td></tr><tr><td>SHAP</td><td>0.691**</td><td>0.639</td><td>C3-P3, F3-C3, F7-T3, FP2-F8, P4-O2</td><td>0.956***</td><td>0.872</td><td>F8-T4, FP2-F8, P3-O1, T5-O1, T6-O2</td></tr><tr><td>GradientShap</td><td>0.712**</td><td>0.641</td><td>C3-P3, FP2-F4, P4-O2, T3-T5, T6-O2</td><td>0.979***</td><td>0.872</td><td>F8-T4, FP2-F8, T3-T5, T5-O1, T6-O2</td></tr><tr><td>LIME</td><td>0.279</td><td>0.431</td><td>F7-T3, FP2-F4, P4-O2, T3-T5, T5-O1</td><td>0.979***</td><td>0.872</td><td>F8-T4, FP2-F8, T3-T5, T5-O1, T6-O2</td></tr><tr><td>Occlusion GradCAM</td><td>0.594* 0.121</td><td>0.562</td><td>C3-P3, FP2-F8, P4-O2, T3-T5, T4-T6</td><td>0.991***</td><td>0.872</td><td>F8-T4, FP2-F8, T3-T5, T5-O1, T6-O2</td></tr><tr><td></td><td></td><td>-0.013</td><td>C3-P3, C4-P4, F4-C4, T3-T5, T5-O1</td><td>0.465</td><td>0.475</td><td>F8-T4, FP2-F8, P3-O1, T4-T6, T6-O2</td></tr><tr><td rowspan="6">EEGMamba (116/270)</td><td>LIME*</td><td>0.976***</td><td>0.714</td><td>Fp1-F3, Fp2-F4, P3-O1, T3-T5, T4-T6</td><td>0.738**</td><td>0.516</td><td>C4-P4, F3-C3, F8-T4, Fp1-F7, P4-O2</td></tr><tr><td>IG</td><td>0.591*</td><td>0.493</td><td>F8-T4, Fp1-F3, T3-T5, T4-T6, T6-O2</td><td>0.712**</td><td>0.542</td><td>F8-T4, Fp1-F3, P3-O1, P4-O2, T4-T6</td></tr><tr><td>SHAP</td><td>0.856***</td><td>0.678</td><td>Fp1-F3, Fp2-F4, P4-O2, T3-T5, T4-T6</td><td>0.859***</td><td>0.632</td><td>F3-C3, F4-C4, F7-T3, F8-T4, Fp2-F4</td></tr><tr><td>GradientShap</td><td>0.832***</td><td>0.654</td><td>F3-C3, Fp1-F3, Fp2-F4, T3-T5, T4-T6</td><td>0.868***</td><td>0.671</td><td>F8-T4, Fp1-F3, Fp2-F8, P4-O2, T6-O2</td></tr><tr><td>Occlusion</td><td>0.991***</td><td>0.690</td><td>F8-T4, Fp2-F4, P3-O1, T3-T5, T4-T6</td><td>0.900***</td><td>0.542</td><td>F7-T3, F8-T4, Fp1-F3, Fp1-F7, Fp2-F8</td></tr><tr><td>GradCAM</td><td>0.238</td><td>0.038</td><td>F3-C3, F7-T3, F8-T4, Fp1-F7, T4-T6</td><td>0.185</td><td>0.115</td><td>F8-T4, Fp1-F3, Fp1-F7, Fp2-F4, T4-T6</td></tr><tr><td rowspan="6">LaBraM (116/272)</td><td>Occlusion*</td><td>0.720***</td><td>0.359</td><td>C4, F3, F4, FZ, T2</td><td>0.627**</td><td>0.238</td><td>CZ, F4, FP1, P3, T2</td></tr><tr><td>IG</td><td>-0.146</td><td>0.152</td><td>A2, C4, O2, P3, T6</td><td>0.551**</td><td>0.176</td><td>F3, F4, F7, P3, T5</td></tr><tr><td>SHAP</td><td>0.231</td><td>0.187</td><td>FP1, O2, T1, T2, T3</td><td>0.360</td><td>0.202</td><td>A1, A2, F3, F8, PZ</td></tr><tr><td>GradientShap</td><td>0.325</td><td>0.345</td><td>A2, O2, P4, T2, T6</td><td>0.316</td><td>0.314</td><td>A1, A2, T3, T4, T5</td></tr><tr><td>LIME</td><td>0.027</td><td>0.371</td><td>FZ, P3, P4, T2, T3</td><td>0.223</td><td>0.173</td><td>A1, CZ, F4, FP2, T3</td></tr><tr><td>GradCAM</td><td>0.384</td><td>0.102</td><td>A2, FP1, FP2, P4, T2</td><td>0.256</td><td>0.254</td><td>F7, FP1, P4, T3, T4</td></tr><tr><td rowspan="6">EEGPT (115/264)</td><td>Occlusion*</td><td>0.717***</td><td>0.312</td><td>A2, CZ, FP1, T3, T6</td><td>0.670***</td><td>0.125</td><td>F8, T2, T3, T4, T6</td></tr><tr><td>GradientShap</td><td>0.628**</td><td>0.274</td><td>A2, F8, FP1, O2, T5</td><td>0.589**</td><td>0.561</td><td>A2, O1, O2, T2, T4</td></tr><tr><td>LIME</td><td>0.297</td><td>0.407</td><td>A2, C3, O1, O2, T6</td><td>0.656***</td><td>0.420</td><td>CZ, O1, O2, T2, T4</td></tr><tr><td>IG</td><td>0.058</td><td>0.245</td><td>A2, O1, O2, T3, T4</td><td>0.490*</td><td>0.500</td><td>A2, O1, O2, T2, T4</td></tr><tr><td>SHAP</td><td>0.008</td><td>0.226</td><td>A2, F7, O1, T3, T6</td><td>0.226</td><td>0.299</td><td>FP1, T3, T4, T5, T6</td></tr><tr><td>GradCAM</td><td></td><td></td><td>F3, F4, F7, FP2, T2</td><td></td><td></td><td>F3, F4, FZ, FP2, T2</td></tr><tr><td rowspan="6">BIOT (75/270)</td><td>Occlusion*</td><td>-0.412</td><td>0.285</td><td>C3-P3, C4-P4, F3-C3, F8-T4, T4-T6</td><td>0.762***</td><td>0.331</td><td>P4-O2, T3-T5, T4-T6, T5-O1, T6-O2</td></tr><tr><td>GradCAM</td><td>0.653**</td><td></td><td>C4-P4, P3-O1, P4-O2, T5-O1, T6-O2</td><td>0.450</td><td></td><td>F7-T3, FP1-F3, FP1-F7, FP2-F8, T5-O1</td></tr><tr><td>LIME</td><td>0.235</td><td>0.358</td><td>F3-C3, FP1-F7, P3-O1, T3-T5, T6-O2</td><td>0.268</td><td>0.642</td><td>C3-P3, FP1-F3, P4-O2, T3-T5, T6-O2</td></tr><tr><td>IG</td><td>-0.350</td><td>0.490</td><td>C4-P4, FP1-F3, FP1-F7, P3-O1, P4-O2</td><td>0.094</td><td>0.624</td><td>C3-P3, FP1-F3, P4-O2, T5-O1, T6-O2</td></tr><tr><td>GradientShap</td><td>-0.374</td><td>0.418</td><td>C4-P4, FP1-F3, FP1-F7, P4-O2, T6-O2</td><td>-0.021</td><td>0.675</td><td>F4-C4, FP1-F3, P3-O1, P4-O2, T5-O1</td></tr><tr><td>SHAP</td><td>-0.235</td><td>0.401</td><td>C3-P3, FP1-F7, P3-O1, T4-T6, T6-O2</td><td>-0.018</td><td>0.586</td><td>FP1-F3, P4-O2, T4-T6, T5-O1, T6-O2</td></tr></table>

Band Topomap IG | baseline=zero | n=100  
![](images/7c48908f9bf31313b087d165d1a288587277224daf7384ed0b66fe02af419ac9.jpg)

Band Topomap IG | baseline=zero | n=100  
![](images/58e743a79b4e63e1c383b6f84bcda936f4f455b32e60be05cbf93dee2c432d95.jpg)  
Fig. 5: Band-level population attribution on the Mumtaz dataset using IG with CBraMod. Each row shows the topographic distribution of attribution across five frequency bands for one class.

TABLE 5: Temporal faithfulness on TUEV Class 1 (GPED). ρ: Spearman correlation between patch attribution rank and confidence drop rank. ⋆ Recommended (highest $\rho > 0$ with $p \ < \ . 0 5 )$ $^ { * * * } p ^ { ^ { \circ } } < . 0 0 1 , ^ { * * } p < . 0 1 , ^ { * } p < . 0 5 . - !$ not computable (constant attribution). Note: The patch granularities for the models are: CBraMod / EEGMamba / LaBraM = 5, $\mathrm { B I O T } = 9 $ , and $\operatorname { E E G P T } = 1 5$ . The absolute magnitudes of $\rho$ are not directly comparable; a larger n imposes stricter requirements on the method’s event localization accuracy but simultaneously results in a lower significance threshold.

<table><tr><td>Model</td><td>Method</td><td>ρ</td></tr><tr><td rowspan="6">CBraMod</td><td>LIME IG</td><td>0.800 0.800</td></tr><tr><td></td><td></td></tr><tr><td>SHAP</td><td>0.800</td></tr><tr><td>GradientSHAP</td><td>0.800</td></tr><tr><td>Occlusion</td><td>-0.700</td></tr><tr><td>GradCAM</td><td>-0.900*</td></tr><tr><td rowspan="6">EEGMamba</td><td>LIME</td><td>-0.300</td></tr><tr><td>IG</td><td>0.600</td></tr><tr><td>SHAP*</td><td>1.000***</td></tr><tr><td>GradientSHAP</td><td>0.800</td></tr><tr><td>Occlusion</td><td>-0.400</td></tr><tr><td>GradCAM</td><td>0.300</td></tr><tr><td rowspan="6">EEGPT</td><td>LIME</td><td>0.132</td></tr><tr><td>IG</td><td>0.064</td></tr><tr><td>SHAP</td><td>0.204</td></tr><tr><td>GradientSHAP</td><td>0.543*</td></tr><tr><td>Occlusion*</td><td>0.686**</td></tr><tr><td>GradCAM</td><td></td></tr><tr><td rowspan="6">LaBraM</td><td>LIME</td><td>0.100</td></tr><tr><td>IG</td><td>0.300</td></tr><tr><td>SHAP</td><td>0.100</td></tr><tr><td>GradientSHAP</td><td>0.700</td></tr><tr><td>Occlusion</td><td>0.200</td></tr><tr><td>GradCAM</td><td>-0.100</td></tr><tr><td rowspan="6">BIOT</td><td>LIME</td><td>0.771</td></tr><tr><td>IG*</td><td>0.700*</td></tr><tr><td>SHAP</td><td>0.029</td></tr><tr><td>GradientSHAP</td><td>0.636*</td></tr><tr><td>Occlusion</td><td>0.657</td></tr><tr><td>GradCAM</td><td>0.536</td></tr></table>

At finer granularity, EEGPT $( n ~ = ~ 1 5 )$ achieved $\rho ~ =$ $0 . 6 8 6 ^ { \ast \ast }$ with Occlusion and $\rho = 0 . 5 4 3 ^ { \ast }$ with GradientSHAP, demonstrating strong evidence of event anchoring. BIOT $( n = 9 )$ showed consistently positive values across all six methods, with IG and GradientSHAP reaching significance, indicating clear cross-method convergence.

CBraMod $( n = 5 )$ consistently yielded $\rho = 0 . 8 0 0$ across LIME, IG, SHAP, and GradientSHAP. While it did not reach statistical significance due to sample size limitations,this cross-method consistency offers moderate evidence for attribution reliability. Krishnadisagreement et al. [22], through a systematic comparison of various attribution methods, point out that in practice, consistency among different methods serves as a common heuristic for assessing the reliability of attributions.

For EEGMamba $( n = 5 )$ , SHAP achieved $\rho = 1 . 0 0 0 ^ { \ast \ast * }$ whereas LIME and Occlusion showed inverse correlations, illustrating a typical case of inter-method disagreement [22]. LaBraM $( \bar { n } = 5 )$ exhibited low and non-significant $\rho$ values across methods, reflecting overall weakness in patch-level temporal faithfulness. Bjelogrlic et al. [38] analyzed similar low-faithfulness phenomena in XAI benchmarks for timeseries classification, identifying common causes such as the use of distributed representations with cross-segment redundancy, high feature correlation leading to contextual compensation of perturbations, and method-architecture coupling.

Across the different models, GradientSHAP consistently maintained a positive $\rho$ and frequently approached the bestperforming results for each model, demonstrating relative robustness in the current evaluation. While SHAP exhibited outstanding local performance, it showed high variance across models. LIME and Occlusion showed inconsistent directional behavior across different models, reflecting the limited stability of these methods when applied to highdimensional, highly autocorrelated signals such as EEG data.

## 4.3 Discussion

Experimental results validate the effectiveness of the proposed framework from multiple dimensions:

Clinical consistency of attribution results. In the spatial dimension, various attribution methods consistently highlight brain regions that align with clinical knowledge: the TUAB abnormality category corresponds to temporal lobe regions, the Mumtaz HC (Healthy Control) category to posterior occipital regions, and the MDD (Major Depressive Disorder) category to prefrontal regions. Frequency band analysis further reveals the electrophysiological basis of model decisions, thereby establishing an interpretable link between “black-box” predictions and physiological mechanisms.

Necessity of multi-method cross-validation. The radar chart (Fig. 6) clearly demonstrates that no single attribution method performs optimally across all model-task combinations. For instance, the Occlusion method achieves a high correlation of $\rho = 0 . 9 9 1$ on EEGMamba (TUAB) but only −0.412 on BIOT; the IG method performs stably on CBraMod but fails almost completely on BIOT (TUAB). These results justify the framework’s design choice to integrate multiple methods and provide a recommendation mechanism.

Spatial fidelity generally exceeds temporal fidelity. Overall, the radar radii for TUAB are significantly larger than those for TUEV, indicating that current EEG foundation models exhibit higher consistency in event anchoring along the channel/spatial dimension than along the temporal dimension. Attribution fidelity in the temporal dimension is generally low across most model-method combinations, reflecting that the stable characterization of patch-level temporal fidelity requires further fine-grained methodological evaluation.

## 5 CONCLUSION

This paper proposes a unified ex post-hoc interpretability framework for EEG basic models. This framework achieves multi-dimensional attribution analysis independent of specific model architectures by abstracting different model architectures and integrating various attribution methods.

Through population-level experiments on EEG tasks, we found that: (1) there is no single attribution method that is optimal across models and dimensions; method selection is highly coupled with model architecture and task dimensions; (2) the attribution consistency of current EEG basic models is generally higher in the spatial dimension than in the temporal dimension, and a more granular methodological evaluation is needed to stably characterize patch-level temporal fidelity; (3) cross-model comparisons for specific tasks reveal both common neurobiomarkers and modelspecific biases.

![](images/4eb6c560a8d9abbd4d6a4f4b03fc575bf45a059fa7053fcdc29fba3cf10987ef.jpg)  
Fig. 6: Radar chart of attribution method faithfulness across models. Left: spatial faithfulness (ρ) on TUAB (averaged over Normal/Abnormal classes). Right: temporal faithfulness (ρ) on TUEV Class 1 (GPED). Negative ρ values are clipped to 0.

## REFERENCES

[1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. Lance, “Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces,” Journal of Neural Engineering, vol. 15, 2016. [Online]. Available: https://api.semanticscholar.org/CorpusID:3849381

[2] C. Yang, M. Westover, and J. Sun, “Biot: Biosignal transformer for cross-data learning in the wild,” Advances in Neural Information Processing Systems, vol. 36, pp. 78 240–78 260, 2023.

[3] G. Wang, W. Liu, Y. He, C. Xu, L. Ma, and H. Li, “Eegpt: Pretrained transformer for universal and reliable representation of eeg signals,” Advances in Neural Information Processing Systems, vol. 37, pp. 39 249–39 280, 2024.

[4] Y. Gui, M. Chen, Y. Su, G. Luo, and Y. Yang, “Eegmamba: Bidirectional state space model with mixture of experts for eeg multi-task classification,” arXiv preprint arXiv:2407.20254, 2024.

[5] G. Kuruppu, N. Wagh, V. Kremen, and Y. Varatharajah, “Eeg foundation models: a critical review of current progress and future directions,” Journal of neural engineering, vol. 23, no. 2, p. 021001, 2026.

[6] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al., “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021.

[7] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, “Shortcut learning in deep neural networks,” Nature Machine Intelligence, vol. 2, no. 11, pp. 665–673, 2020.

[8] M. Ghassemi, L. Oakden-Rayner, and A. L. Beam, “The false hope of current approaches to explainable artificial intelligence in health care,” The lancet digital health, vol. 3, no. 11, pp. e745–e750, 2021.

[9] W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Muller, “Evaluating the visualization of what a deep neural net-¨ work has learned,” IEEE transactions on neural networks and learning systems, vol. 28, no. 11, pp. 2660–2673, 2016.

[10] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces,” Journal of neural engineering, vol. 15, no. 5, p. 056013, 2018.

[11] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, “Deep learning with convolutional neural networks for eeg decoding and visualization,” Human brain mapping, vol. 38, no. 11, pp. 5391–5420, 2017.

[12] D. Kostas, S. Aroca-Ouellette, and F. Rudzicz, “Bendr: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of eeg data,” Frontiers in Human Neuroscience, vol. 15, p. 653659, 2021.

[13] W.-B. Jiang, L. Zhao, and B.-L. Lu, “Large brain model for learning generic representations with tremendous eeg data in bci,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 16 405–16 426.

[14] I. Sturm, S. Lapuschkin, W. Samek, and K.-R. Muller, “Inter-¨ pretable deep neural networks for single-trial eeg classification,” Journal of neuroscience methods, vol. 274, pp. 141–145, 2016.

[15] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems, vol. 30, 2017.

[16] T. Shawly and A. A. Alsheikhy, “Eeg-based detection of epileptic seizures in patients with disabilities using a novel attention-driven deep learning framework with shap interpretability,” Egyptian Informatics Journal, vol. 31, p. 100734, 2025.

[17] M. S. Islam, I. Hussain, M. M. Rahman, S. J. Park, and M. A. Hossain, “Explainable artificial intelligence model for stroke prediction using eeg signal,” Sensors, vol. 22, no. 24, p. 9859, 2022.

[18] F. A. Khan, Z. Umar, A. Jolfaei, and M. Tariq, “Explainable ai for epileptic seizure detection in internet of medical things,” Digital Communications and Networks, vol. 11, no. 3, pp. 587–593, 2025.

[19] A. G. Madsen, W. T. Lehn-Schiøler, A. J<sup>´</sup> onsd´ ottir, B. Arnard´ ottir,´ and L. K. Hansen, “Concept-based explainability for an eeg transformer model,” in 2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2023, pp. 1–6.

[20] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl et al., “Large language models encode clinical knowledge,” Nature, vol. 620, no. 7972, pp. 172–180, 2023.

[21] J. W. Kim, A. Alaa, and D. Bernardo, “Eeg-gpt: exploring capabilities of large language models for eeg classification and interpretation,” arXiv preprint arXiv:2401.18006, 2024.

[22] S. Krishna, T. Han, A. Gu, S. Wu, S. Jabbari, and H. Lakkaraju, “The disagreement problem in explainable machine learning: A practitioner’s perspective,” Transactions on Machine Learning Research.

[23] J. Wang, S. Zhao, Z. Luo, Y. Zhou, H. Jiang, S. Li, T. Li, and G. Pan, “Cbramod: A criss-cross brain foundation model for eeg decoding,” in International conference on learning representations, vol. 2025, 2025, pp. 75 310–75 346.

[24] I. Obeid and J. Picone, “The temple university hospital eeg data corpus,” Frontiers in neuroscience, vol. 10, p. 196, 2016.

[25] W. Mumtaz, “Mdd patients and healthy controls eeg data (new),” figshare, Dataset, 2016.

[26] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in International conference on machine learning. PMLR, 2017, pp. 3319–3328.

[27] G. Erion, J. D. Janizek, P. Sturmfels, S. M. Lundberg, and S.-I. Lee, “Improving performance of deep learning models with axiomatic attribution priors and expected gradients,” Nature machine intelligence, vol. 3, no. 7, pp. 620–631, 2021.

[28] M. T. Ribeiro, S. Singh, and C. Guestrin, “” why should i trust you?” explaining the predictions of any classifier,” in Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 2016, pp. 1135–1144.

[29] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in European conference on computer vision. Springer, 2014, pp. 818–833.

[30] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 618–626.

[31] E. Niedermeyer and F. H. Lopes da Silva, Electroencephalography: Basic Principles, Clinical Applications, and Related Fields, 5th ed. Philadelphia: Lippincott Williams & Wilkins, 2005.

[32] D. A. Pizzagalli, “Frontocingulate dysfunction in depression: toward biomarkers of treatment response,” Neuropsychopharmacology, vol. 36, no. 1, pp. 183–206, 2011.

[33] W. O. Tatum IV, Handbook ofEEG Interpretation, 3rd ed. New York: Demos Medical Publishing, 2021.

[34] A. Umemoto, L. Y. Panier, S. L. Cole, J. Kayser, D. A. Pizzagalli, and R. P. Auerbach, “Resting posterior alpha power and adolescent major depressive disorder,” Journal of psychiatric research, vol. 141, pp. 233–240, 2021.

[35] R. A. Movahed, G. P. Jahromi, S. Shahyad, and G. H. Meftahi, “A major depressive disorder classification framework based on eeg signals using statistical, spectral, wavelet, functional connectivity, and nonlinear analysis,” Journal of Neuroscience Methods, vol. 358, p. 109209, 2021.

[36] D. Watts, R. F. Pulice, J. Reilly, A. R. Brunoni, F. Kapczinski, and I. C. Passos, “Predicting treatment response using eeg in major depressive disorder: A machine-learning meta-analysis,” Translational psychiatry, vol. 12, no. 1, p. 332, 2022.

[37] A. Khosla, P. Khandnor, and T. Chand, “Automated diagnosis of depression from eeg signals using traditional and deep learning approaches: A comparative analysis,” Biocybernetics and Biomedical Engineering, vol. 42, no. 1, pp. 108–142, 2022.

[38] H. Turbe, M. Bjelogrlic, C. Lovis, and G. Mengaldo, “Evaluation´ of post-hoc interpretability methods in time-series classification,” Nature Machine Intelligence, vol. 5, no. 3, pp. 250–260, 2023.