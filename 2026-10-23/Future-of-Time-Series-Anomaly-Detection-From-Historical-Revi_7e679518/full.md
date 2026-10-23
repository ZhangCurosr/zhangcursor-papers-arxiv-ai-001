# Future of Time Series Anomaly Detection: From Historical Review to Unified Outlook

Zhijie Zhong, Zhiwen Yu<sup>∗</sup>, Senior Member, IEEE, Pengyu Li, Liujian Zhang, Qiaoli Ni, Jiang Jun, Jianming Lv, Member, IEEE, Kaixiang Yang, Member, IEEE, and C. L. Philip Chen, Fellow, IEEE

Abstract—Time series anomaly detection (TSAD) is a foundational problem in data mining, aiming to identify observations that deviate significantly from nominal temporal patterns. Driven by escalating data volumes and application complexity, recent research has yielded diverse architectures, including diffusion models, state space models, and large language models (LLMs). However, existing surveys provide limited coverage of these emerging designs and lack a unified treatment of multivariate anomalies, cross-domain unified modeling, metric reliability, and contemporary evaluation challenges. To bridge these gaps, this survey delivers three core contributions: (1) We establish a refined anomaly categorization and a structured taxonomy of paradigms reflecting modern advances, comparing detection mechanisms and charting developmental trajectories across paradigms. (2) We provide an extensive analysis of evaluation methodologies and conduct comprehensive experiments to assess metric fidelity, complemented by an open-source benchmark repository encompassing algorithms, datasets, and evaluation tools. (3) We outline key open challenges, offering forward-looking perspectives on dynamic thresholding, model interpretability, zeroshot transfer, multi-modal alignment, and foundational models for time series. By addressing existing theoretical and empirical limitations, this survey establishes a coherent framework to guide future TSAD research and practical deployments.

Impact Statement—Detecting anomalies in time series data is essential for safeguarding critical infrastructure across healthcare, finance, transportation, and manufacturing. Failures in detection can result in undetected equipment malfunctions, financial fraud, and security breaches, carrying significant economic and societal consequences. Despite rapid algorithmic progress, practitioners face a fragmented research landscape with inconsistent taxonomies and unreliable evaluation standards, making dependable method selection difficult. This survey addresses this challenge by establishing a unified paradigm taxonomy that separates detection principles from model architectures, empirically exposing the

inflation bias of widely adopted metrics, and providing actionable algorithm selection guidelines tailored to operational constraints such as interpretability, latency, and zero-shot capability. By further charting open frontiers in dynamic thresholding, model interpretability, multi-modal perception, and foundational models, this work offers a structured roadmap to advance both research and real-world deployment of time series anomaly detection.

Index Terms—Time series anomaly detection, data mining, large language models, evaluation metrics, anomaly taxonomy

## I. INTRODUCTION

time series is a sequence of data points indexed in chronological order, with extensive applications across diverse domains, including finance, healthcare, industrial monitoring, and cybersecurity [1, 2, 3, 4]. Time series anomaly detection (TSAD), a fundamental area of temporal data analytics, aims to identify observations or segments that deviate significantly from nominal patterns [5, 6, 7]. By uncovering early indicators of equipment malfunctions, financial fraud, and malicious network intrusions, timely anomaly detection safeguards system reliability and prevents catastrophic operational failures across complex infrastructures [8, 9, 10].

The field of TSAD has evolved rapidly, transitioning from early statistical and classical machine learning methods to advanced deep learning architectures. To quantitatively contextualize this evolution, we analyzed publication trends over the decade spanning 2015 to 2025 across different algorithmic paradigms and architectures using data from Google Scholar, as illustrated in Fig. 1. This analysis reveals several key shifts: while research into traditional forecasting and statistical paradigms has gradually decelerated since 2022, generative, similarity-based, and reconstruction paradigms remain highly active. Concurrently, large language model (LLM)-based methods have exhibited a marked surge in publication volume since 2024, emerging as the fastest-expanding research area. Detailed search queries and raw publication data are documented in Section ?? of the Supplementary Material (SM).

However, this rapid expansion has also exposed notable limitations in existing surveys. As detailed in our related work comparison, previous surveys typically focus on specific application domains or earlier model families, leaving a marked gap in the coverage of recent architectural innovations, such as Transformers [7, 11, 5], Diffusion Models [12, 13], State Space Models (SSMs) [14, 15, 16], and Large Language Models (LLMs) [17, 18, 19, 20]. In parallel, existing surveys lack a unified taxonomy to categorize these emerging paradigms, while critical methodological issues, such as anomaly classifi cation for multivariate series and the empirical reliability of evaluation metrics, remain insufficiently examined. In particular, ambiguous anomaly definitions in multivariate time series continue to hinder reproducible research progress.

![](images/0da88960d4a6ee86ad49db58ec3c572852dc0239a9895ed6378e5f0d1faa0186.jpg)  
Fig. 1: Publication trends of TSAD algorithms from 2015 to 2025. The top panel shows a statistical breakdown based on the taxonomy of algorithmic paradigms, while the bottom panel delineates publications across different model architectures.

To address these limitations, this survey presents a comprehensive and unified perspective on TSAD. Specifically, we introduce a refined classification for time series anomalies, establish a structured taxonomy of algorithmic paradigms grounded in core detection principles, and thoroughly examine representative deep learning architectures. In addition, we discuss community-wide challenges and outline promising future directions to guide subsequent research. The organizational framework of this paper is illustrated in Fig. 2.

In summary, our key contributions are:

1) Refined Anomaly Taxonomy (Sec. III): We formalize the concept of channel-correlation anomalies, establishing a unified perspective on multivariate anomaly patterns and providing a rigorous framework for analyzing complex cross-channel dependencies.

2) Structured Taxonomy of Paradigms (Sec. IV): We establish a structured taxonomy that categorizes mainstream algorithms according to their core detection assumptions and structural characteristics, clarifying the fundamental mechanisms governing their detection capabilities.

3) Comprehensive Review of Emerging Architectures (SM ??): We provide an in-depth overview of modern algorithm designs, with dedicated analyses of diffusion models, state space models, prototype-based networks, and LLM-driven detectors that are largely omitted from previous surveys.

4) Critical Evaluation of Metrics and Benchmark Toolkit (SM ?? and Sec. V): We critically evaluate existing evaluation metrics, exposing methodological pitfalls such as metric inflation bias and offering actionable selection guidelines. To facilitate reproducible research, we compile and publicly release an open-source benchmark repository covering representative algorithms, datasets, and standardized evaluation tools.

5) Open Challenges and Future Directions (Secs. V and VI): We highlight pivotal open challenges, including dynamic thresholding, model interpretability, zero-shot transfer, multi-modal alignment, and foundational models, outlining a structured roadmap for future research and deployment.

The remainder of this survey is structured as follows. Sec. II reviews related literature and defines essential terminology. Sec. III presents our refined anomaly taxonomy, followed by the structured taxonomy of algorithmic paradigms in Sec. IV. Secs. V and VI provide in-depth discussions on operational challenges and outline prospective research opportunities. Finally, Sec. VII concludes the survey.

To foster reproducible research and community engagement, all resources mentioned in this survey, including supplemental materials, benchmark code, and curated datasets, are openly accessible at https://emorzz1g.github.io/FTSAD/.

## II. PRELIMINARIES

This section positions our survey within the broader context of TSAD research through a comparative analysis of related literature. We then formalize core mathematical notations and problem definitions, establishing the core concepts required for the anomaly taxonomy in Sec. III and the paradigm classification in Sec. IV.

## A. Related Works

To delineate the historical and technical trajectory of the field, we categorize the evolution of TSAD algorithms into four chronological phases governed by dominant architectural and modeling principles.

The first phase focused on classical machine learning algorithms. During this period, general-purpose anomaly detectors (such as tree-based ensembles and empirical distribution estimators) were either applied directly to temporal sequences or augmented with handcrafted features [21, 22, 23]. Although computationally lightweight and inherently interpretable, these methods frequently struggle with the scale and high dimensionality of modern datasets, as their underlying assumptions of linearity or low-rank structures fail to capture complex, nonlinear, and dynamic temporal dependencies.

The second phase introduced lightweight deep learning models for TSAD. These architectures primarily encompass three families: Recurrent Neural Networks (RNNs) [24, 25], Temporal Convolutional Networks (TCNs) [26, 27], and Graph Neural Networks (GNNs) [28, 29, 30]. Concurrently, prototypebased models [31, 32, 33] emerged as a distinct subcategory within the similarity-based paradigm. Unlike standard autoencoding architectures that frequently suffer from overgeneralization, such as inadvertently reconstructing anomalous inputs, prototype models explicitly learn a discrete dictionary of representative memory prototypes to encapsulate nominal temporal dynamics. Because existing surveys have largely overlooked prototype-based algorithms, we thoroughly analyze their formulations and architectural characteristics.

![](images/c358f5ed850b9145a92e3c065967143605b68f6c2428f42fb257dcb9c88d4ec3.jpg)  
Fig. 2: The organizational framework of this paper.

The third phase centers on large-parameter deep learning models for TSAD, characterized by expressive backbone architectures designed to capture extended temporal contexts. Transformer-based models exploit multi-head self-attention to model long-range sequence correlations [7, 11], whereas diffusion-based models estimate underlying data distributions through iterative denoising to isolate low-probability events [13, 34]. More recently, state space models (SSMs) [14, 16], particularly selective scan architectures such as Mamba, achieve linear computational complexity with respect to sequence length, demonstrating substantial potential for scalable sequence analysis. Given that prior surveys have focused predominantly on Transformers while giving insufficient treatment to diffusion models and SSMs, we review both categories in detail to address this gap.

Finally, the fourth phase explores future-oriented deep TSAD algorithms, primarily encompassing cross-domain unified architectures and large language model (LLM)-adapted detectors. Rather than training isolated models from scratch for each dataset, current research emphasizes zero-shot transfer, multimodal signal alignment (integrating numeric sequences with textual logs and visual indicators), and foundational models for time series. We examine these emerging paradigms and their operational constraints in Secs. V and VI. Because existing surveys rarely explore LLM-based approaches or unified multidomain architectures, our survey provides a timely roadmap for these frontiers.

Table I compares our work with recent key related surveys to position our contributions within the existing literature. A detailed comparison of all related surveys is provided in SM Table ??. As the table illustrates, our survey is distinguished from prior studies in three key aspects:

• Broad Coverage of Emerging Architectures: Existing surveys rarely cover the full spectrum of modern deep learning architectures, such as prototype networks, state space models, diffusion models, and LLMs (as detailed in the Novel Algorithms column of Table I). Cross-domain unified TSAD models also remain largely unexplored. This review provides a detailed overview of these cutting-edge algorithms and their operational characteristics.

• Principled Taxonomy of Paradigms: To accommodate the continuous influx of new models, we formalize a structured taxonomy that classifies algorithms according to their core detection assumptions rather than superficial backbone implementations, with developmental trajectories detailed in Section ?? of the Supplementary Material.

• Refined Multivariate Anomaly Taxonomy: While prior surveys propose anomaly taxonomies [40, 41, 42, 43], they remain largely confined to univariate formulations. Because multivariate detection must explicitly account for inter-channel relationships [44, 5, 45], we formalize channel-correlation anomalies to provide a unified framework for complex cross-channel interactions.

## B. Definitions

This subsection establishes formal mathematical notations and problem definitions for time series anomaly detection and multi-domain unified modeling, providing the foundation for the taxonomies introduced in Secs. III and IV.

1) Definition of Problems: Time series data comprises a sequence of observations ordered chronologically. Formally, a time series is represented as $\pmb { t } = [ \pmb { t } _ { 1 } , \pmb { t } _ { 2 } , \pmb { \bar { \ } } . \pmb { \cdot } , \pmb { t } _ { N } ] ^ { \top } \in \mathbb { R } ^ { \tilde { N } \times C }$ where $\pmb { t } _ { i } ~ \in ~ \mathbb { R } ^ { \bar { C } }$ denotes the observation vector at timestamp i, and N signifies the total sequence length. Each observation encompasses C distinct channels or variables. The univariate sequence for the c-th channel is denoted as $\pmb { t } _ { : , c } = [ t _ { 1 , c } , t _ { 2 , c } , \ldots , t _ { N , c } ] ^ { \top } \in \mathbb { R } ^ { N }$ . A sequence with C > 1 is termed a multivariate time series, whereas C = 1 defines a univariate time series.

While time series analysis conventionally assumes temporal continuity and strong auto-correlation, empirical datasets frequently deviate from ideal regularities. In practical deployments, detectors must contend with irregularities such as non-uniform sampling intervals [46] and missing values [47], which require specialized pre-processing or robust architectural accommoda tions.

TABLE I: Overview of Recent Key Time Series Anomaly Detection Surveys.
<table><tr><td rowspan="2">Citation</td><td rowspan="2">Application</td><td rowspan="2">Taxonomy of Paradigms</td><td rowspan="2">Anomaly Taxonomy</td><td colspan="2">Novel Algorithms</td><td rowspan="2">Library</td><td rowspan="2">Datasets</td><td rowspan="2"></td><td colspan="3">Discussion</td></tr><tr><td>Prot/SSM/Diff</td><td>LLM</td><td>Metrics Interpretability</td><td>Multi-Modality</td><td>Multi-Domain</td></tr><tr><td>Choi et al. [1] (2021)</td><td>V</td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Schmidl et al. [35] (2022)</td><td></td><td>√</td><td>√</td><td></td><td></td><td></td><td>√</td><td>√</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wen et al. [36] (2023)</td><td></td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yan et al. [37] (2023)</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ye et al. [38] (2024) Su et al. [39] (2024)</td><td></td><td></td><td></td><td></td><td>√ √</td><td></td><td>√</td><td>V</td><td></td><td>√</td><td>√</td></tr><tr><td>Zamanzadeh Darban et al. [6] (2024)</td><td>√</td><td>1</td><td>√</td><td>1</td><td></td><td></td><td></td><td>√</td><td>V</td><td></td><td></td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Each timestamp is associated with a ground-truth binary label $y _ { i } \in \{ 0 , 1 \}$ , where $y _ { i } = 1$ denotes an anomaly and $y _ { i } = 0$ indicates nominal behavior, yielding a label sequence ${ \textbf { \em y } } =$ $[ y _ { 1 } , y _ { 2 } , \dotsc , y _ { N } ] ^ { \top } \in \{ 0 , 1 \} ^ { N }$ . Because anomalies are inherently rare in real-world systems, anomalous observations are heavily outnumbered by nominal points. The objective of a TSAD algorithm is to estimate an anomaly score sequence and predict the binary label sequence y from the input observations t.

Anomaly Length: A point anomaly manifests at index i when $y _ { i } = 1$ , where all existing immediate adjacent timestamps of i are normal. For boundary positions at the start or end of the entire sequence, only the single existing neighbor needs to be normal to form a point anomaly. A segment anomaly (or sequence anomaly) spans a contiguous interval from i to j where $y _ { i : j } ~ = ~ 1 ~ [ 6 ]$ . This definition does not require the timestamps immediately before i or after j to be normal, allowing distinct anomalies to exist in close proximity. The anomaly length is defined as $L = j - i + 1$ . Accounting for anomaly length is essential, as differing durations often reflect distinct operational severity levels [48].

2) Multi-Domain Learning: In time series analysis, multidomain learning involves extracting shared representations across diverse temporal domains simultaneously [49, 50]. Formally, a multi-domain temporal dataset is defined as $\mathcal { D } \ = \ \{ \mathcal { T } , \mathcal { Y } , \mathcal { S } \}$ , where T, Y, and S represent the collections of time series observations, corresponding groundtruth labels, and domain identifiers, respectively. Here, $\mathcal { T } =$ $\{ T ^ { 1 } , T ^ { 2 } , \dots , T ^ { N _ { d } } \}$ comprises data from $N _ { d }$ distinct domains, with each $\pmb { T } ^ { i } \in \mathbb { R } ^ { N _ { i } \times C _ { i } }$ encapsulating the observations specific to domain i. Correspondingly, the label set is $y =$ $\left\{ { \pmb y } _ { 1 } , { \pmb y } _ { 2 } , \ldots , { \pmb y } _ { N _ { d } } \right\}$ , where each domain’s label sequence is given by $\pmb { y _ { i } } = \{ y _ { 1 } , y _ { 2 } , \dotsc , y _ { N _ { i } } \} \in \{ 0 , 1 \} ^ { N _ { i } }$ . Distinct domains commonly present heterogeneous sequence lengths $N _ { i }$ and channel dimensions $C _ { i }$ , such that $N _ { i } \neq N _ { j }$ or $C _ { i } \neq C _ { j }$ holds for arbitrary domain pairs $i \neq j$ . Models designed for multidomain settings must dynamically accommodate this structural heterogeneity while leveraging invariant cross-domain temporal knowledge.

Unified TSAD models are designed to detect anomalies across heterogeneous temporal domains through a single shared network [51]. Formally, given a multi-domain dataset $\mathcal { D }$ , a unified detector M maps temporal inputs T to anomaly scores A and domain-aware latent representations $\mathcal { Z }$

$$
m ( \mathcal { T } ; \Theta , \Phi ) \to ( \mathcal { A } , \mathcal { Z } ) ,
$$

where Θ denotes domain-shared parameters that encode universal temporal features, and $\Phi = \{ \phi _ { 1 } , \phi _ { 2 } , . . . , \phi _ { N _ { d } } \}$ represents optional domain-specific adapters. The inclusion of Φ depends on whether target domains are known during training.

An effective unified architecture realizes dual capabilities: 1) Parameter Sharing: The core backbone M relies primarily on the unified parameter set Θ across all domains in D, minimizing parameter redundancy and training overhead [52]. 2) Dual-Level Generalization: The model achieves both intra-domain robustness (detecting anomalies under temporal distribution shifts within known domains) and cross-domain transferability (generalizing to unseen target domains with zero or fewshot adaptation). By harmonizing heterogeneous inputs into a shared representation space while maintaining high parameter efficiency, unified architectures provide a scalable foundation for modern industrial monitoring [52].

## III. REFINED ANOMALY TAXONOMY

An anomaly in temporal data represents a significant deviation from the underlying generative distribution, manifesting either as an isolated point observation or as a contiguous subsequence departing from nominal behavior [53].

Historically, literature in time series analysis [54, 55, 53, 56] has categorized anomalies into global, contextual, and structural forms. However, this classical taxonomy is predominantly tailored for univariate sequences and fails to articulate how anomalies manifest across interdependent variables. This section first reviews these three classical univariate anomaly types and then, in Sec. III-D, formalizes the taxonomy of channelcorrelation (CC) anomalies to capture the intricate inter-channel dynamics governing multivariate systems.

## A. Global Anomalies

Global anomalies represent extreme deviations within a time series, frequently manifesting as sudden spikes or abrupt level shifts that fall far outside the nominal distribution bounds [54, 6], as illustrated in Fig. 3(a). Formally, a global anomaly at timestamp i for channel c is evaluated as:

$$
\mathcal { F } _ { g a } = | t _ { i , c } - { \hat { t } } _ { i , c } | > \delta _ { g a } ,\tag{1}
$$

where $t _ { i , c }$ denotes the observed value at timestamp i for channel $c , \hat { t } _ { i , c }$ is the expected or reconstructed baseline value, $\mathcal { F } _ { g a }$ denotes the residual-based anomaly indicator, and $\delta _ { g a }$ is the corresponding detection threshold. An observation is flagged as a global anomaly whenever its absolute deviation surpasses $\delta _ { g a }$

![](images/06d61a5d5a9748534630e43a7ecd54449ca40dadc9f0fe4591f80891d289c962.jpg)  
Fig. 3: Illustration of different types of anomalies. The red box indicates the area of anomaly.

## B. Contextual Anomalies

Contextual anomalies manifest as observations whose values fall within the global range of the series but deviate significantly when evaluated against their localized temporal neighborhood [57, 6, 54], as shown in Fig. 3(b). Contextual detection relies on surrounding temporal structure and is formally expressed as:

$$
\mathcal { F } _ { c a } = | \sigma ( \pmb { t } _ { ( i - m + 1 ) : i , c } ) - \sigma ( \hat { \pmb { t } } _ { ( i - m + 1 ) : i , c } ) | > \delta _ { c a } ,\tag{2}
$$

where $\pmb { t } _ { ( i - m + 1 ) : i , c }$ denotes the sequence of channel c over a temporal window of length m ending at index i, and the function $\sigma ( \cdot )$ extracts a localized contextual feature, such as local variance or neighborhood density.

## C. Structural Anomalies

Unlike point-level irregularities, structural anomalies disrupt the underlying morphological or generative patterns of a time series across an extended temporal window, as illustrated in Figs. 3(c–e). These anomalies generally encompass three primary forms:

• Trend anomalies represent unexpected regime shifts or persistent drift in the low-frequency trajectory of a sequence [54] (Fig. 3(c));

• Shape anomalies occur when local temporal waveforms violate expected morphological patterns, even while point values remain within nominal bounds (Fig. 3(d));

• Seasonal anomalies involve deviations from regular periodic cycles, where timing or phase relationships are distorted despite normal cyclical amplitudes (Fig. 3(e)).

These structural variations can be unified under a generalized evaluation formula:

$$
\mathcal { F } _ { s a } = \mathrm { S t r u c F u n c } _ { \mathrm { t y p e } } ( \pmb { t } _ { ( i - m + 1 ) : i , c } , ~ \hat { \pmb { t } } _ { ( i - m + 1 ) : i , c } ) > \delta _ { s a } ,\tag{3}
$$

where $\mathrm { S t r u c F u n c } _ { \mathrm { t y p e } }$ evaluates deviations across decomposed representations, such as trend trajectories, seasonal profiles, or shapelet distances, against detection threshold $\delta _ { s a }$

## D. Channel-Correlation Anomalies

While previous literature has established diverse anomaly taxonomies [54, 55, 53, 56], existing studies rarely formalize the boundary between univariate and multivariate anomalies, nor do they quantify how cross-channel dependencies govern the composite anomaly score. Many existing methods [53, 7,

44] adopt the channel-independence assumption, detailed in Section ?? of the Supplementary Material, which treats each variable as an isolated series. However, this assumption remains valid only when inter-channel correlations are demonstrably negligible [6].

Building upon the concept of correlation anomalies introduced by Wenig et al. [57], we formalize a unified taxonomy: Channel-correlation (CC) anomalies originate from synchronous or coupled abnormal behaviors across multiple channels, manifesting as disruptions in the joint distribution rather than extreme marginal values in individual variables. As demonstrated by Roy et al. [58], subtle multivariate anomalies often elude single-channel inspection because indi vidual variables appear statistically nominal when viewed in isolation. Such events become identifiable only through crosschannel joint modeling. Consequently, contemporary research increasingly deploys joint models that explicitly capture intervariable dependencies.

Figs. 3(f–j) illustrate five distinct classes of channelcorrelation anomalies: CC-Global, CC-Contextual, CC-Trend, CC-Shape, and CC-Seasonal. In Fig. 3(g), although the peaks in the two channels individually surpass their contextual thresholds, it is their synchronous co-occurrence that constitutes the multivariate anomaly. In Fig. 3(i), the individual shapes of both channels adhere largely to historical profiles, leading marginal detectors to misclassify the interval as nominal. Under the CC-Shape formulation, however, the coupled shape distortion across both channels collectively flags the anomaly.

We formalize the anomaly score for a single channel c as $\mathcal { F } _ { \mathrm { t y p e } } ^ { c }$ . Channel-correlation anomalies can be defined as follows:

$$
\begin{array} { l } { \displaystyle \mathcal { F } _ { \mathrm { c c a } } = \sum _ { c = 1 } ^ { C } \mathcal { F } _ { \mathrm { t y p e } } ^ { c } > \sigma _ { \mathrm { c c a } } } \\ { \displaystyle \mathrm { s . t . ~ } \forall \mathrm { t y p e } \in \{ \mathrm { g a , c a , s a } \} , \forall c , \quad \mathcal { F } _ { \mathrm { t y p e } } ^ { c } > \sigma _ { \mathrm { t y p e } } . } \end{array}\tag{4}
$$

This formulation flags a channel-correlation anomaly when every channel-specific score $\mathcal { F } _ { \mathrm { t y p e } } ^ { c }$ exceeds its individual threshold $\sigma _ { \mathrm { t y p e } }$ and their aggregate sum simultaneously surpasses the joint threshold $\sigma _ { \mathrm { c c a } }$ . This concurrent violation captures anomalous dynamics unique to coupled multivariate sequences.

In practical deployments, isolating whether a multivariate event is driven by a specific underlying anomaly sub-type per channel is often intractable. A generalized and widely adopted formulation evaluates each channel’s weighted contribution to

the joint anomaly score:

$$
\begin{array} { l } { { \displaystyle { \mathcal { F } } _ { \mathrm { c c a } } = \sum _ { c = 1 } ^ { C } \alpha _ { c } { \mathcal { F } } ^ { c } > \sigma _ { \mathrm { c c a } } } , } \\ { { \displaystyle \mathrm { w i t h } \forall c , \mathcal { F } ^ { c } > \sigma _ { c } } , } \end{array}\tag{5}
$$

where $\alpha ~ = ~ [ \alpha _ { 1 } , \ldots , \alpha _ { C } ]$ denotes predefined channel importance weights, and ${ \mathcal { F } } ^ { c }$ represents the aggregate anomaly score for channel c. The weights $\alpha _ { c }$ enable uninformative or decoupled channels to be masked by setting $\alpha _ { c } = 0$ when an anomaly is confined to a specific functional subsystem.

While (4) and (5) define scoring rules to detect cross-channel synchrony, they serve as empirical identification criteria rather than explicit generative models. In operational environments, multivariate anomalies emerge from diverse inter-channel dynamics; our formulation prioritizes observable synchrony to ensure robust practical detection.

In critical cyber-physical systems and autonomous platforms, multivariate time series are additionally governed by physical and kinematic constraints [59]. Whereas classical modelbased detectors naturally incorporate kinematic equations to identify physically inadmissible trajectories, data-driven modelfree methods require explicit inductive biases or structured embeddings. Recent architectures address this requirement through physics-informed deep learning, such as Pi-Transformer [60], which embeds conservation laws directly into the attention mechanism to ensure that detected deviations reflect genuine physical faults rather than unconstrained modeling artifacts.

## IV. TAXONOMY OF PARADIGMS

In this section, we categorize existing TSAD algorithms according to their core operational paradigms, outlining their fundamental methodologies and formal structures.

To ensure unambiguous paradigm assignment, the taxonomy is grounded in a three-tier classification framework. Detailed decision criteria, a boundary disambiguation table, and illustrative case analyses are provided in SM ??.

1) LLM Priority Rule. The emergence of foundational models represents a paradigm shift from learning taskspecific distributions from scratch to leveraging pretrained cross-domain knowledge for zero-shot generalization. Accordingly, methods whose detection capability fundamentally relies on such pre-trained representations are assigned to the LLM-based Paradigm, irrespective of the mathematical form of their anomaly score.

2) Core Detection Assumption. All other methods are classified by their core detection assumption, which defines the fundamental criterion for identifying anomalies across Sec. IV-A through Sec. IV-F.

3) Dominant Mechanism Rule. When a method employs multiple scoring signals, it is assigned to the paradigm of its primary anomaly score; hybrid cases are noted in SM ??.

Fig. 4 illustrates the conceptual detection principles of each paradigm. Complementarily, Table II compiles representative literature across paradigms, neural backbones, and unified models, with detailed algorithmic treatments in SM ??. The alignment between architectural backbones and detection paradigms is further analyzed in Sec. V-B and illustrated in Fig. 5.

TABLE II: Summary of algorithmic innovations. Detailed descriptions and references are available in SM ??.
<table><tr><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=1>|Category</td><td rowspan=1 colspan=1>Representative Algorithms</td></tr><tr><td rowspan=6 colspan=1>Detection Paradigms(SM ??)</td><td rowspan=1 colspan=1>Reconstruction</td><td rowspan=1 colspan=1>| EncDec-AD, MSCRED, UAE, G-LSTM-AE, USAD, RE-ADTS, NPSR</td></tr><tr><td rowspan=1 colspan=1>Forecasting</td><td rowspan=1 colspan=1>ARIMA, FEDformer, Autoformer, PatchTST, DLinear</td></tr><tr><td rowspan=1 colspan=1>Generative</td><td rowspan=1 colspan=1>LSTM-VAE, InterFusion, MUTANT, FCVAE, FGANomaly, TadGAN,AnoFormer, BeatGAN, UAC-AD, D3R, ImDiffusion, DDMT, TimeADDM,DDTAD</td></tr><tr><td rowspan=1 colspan=1>Similarity-based</td><td rowspan=1 colspan=1>KNN, DTW, PSOM, LOF, HBOS, ECOD, OC-TCN, COCA, COUTA,THOC, PatchAD, CARLA, SimAD, TimeAutoAD, USD, BroadCAE.MEMTO, AMSL, AdaMemBLS, OCSVM, EB-OCSVM, Isolation Forest</td></tr><tr><td rowspan=1 colspan=1>Statistical</td><td rowspan=1 colspan=1> Z-Scores, 3-Sigma, IQR</td></tr><tr><td rowspan=1 colspan=1>LLM-based</td><td rowspan=1 colspan=1>TS-Bert, GPT4TS, aLLM4TS, Timer, AnomalyLLM, LLMAD, SIGLLM,ParInfoGPT, Holmes-VAD, MERIT</td></tr><tr><td rowspan=6 colspan=1>RepresentationArchitectures (SM ??)</td><td rowspan=1 colspan=1>RNN Backbone</td><td rowspan=1 colspan=1>| OmniAnomaly, CAE-AD, MSDMM, LSGA</td></tr><tr><td rowspan=1 colspan=1>GNN Backbone</td><td rowspan=1 colspan=1> STADN, DGINet, DPGLAD, SGTrans</td></tr><tr><td rowspan=1 colspan=1>TCN Backbone</td><td rowspan=1 colspan=1>| TCN-AE, WaRTEm-AD, TFAD</td></tr><tr><td rowspan=1 colspan=1>State Space Model</td><td rowspan=1 colspan=1>| S-Mamba, DMamba, C-Mamba, Mambaformer, SGFM, TSAD-C</td></tr><tr><td rowspan=1 colspan=1>Transformer Backbone</td><td rowspan=1 colspan=1>Anomaly Transformer, TransAnomaly, CauseFormer, TiSAT, TranAD,AnomalyBERT, DCdetector, D3TN</td></tr><tr><td rowspan=1 colspan=1>Others</td><td rowspan=1 colspan=1>| Graph-Augmented Normalizing Flows, PeFAD</td></tr><tr><td rowspan=1 colspan=1>Unified TSAD Models(SM ??)</td><td rowspan=1 colspan=1>Cross-Domain Models</td><td rowspan=1 colspan=1>DADA, DACAD, ContexTDA, FS-ADAPT, VANDA, UNITS</td></tr></table>

## A. Reconstruction Paradigm

As illustrated in Fig. 4(a), the reconstruction paradigm exploits the intrinsic structural regularities of time series, focusing on temporal continuity and inter-channel correlations. The core premise posits that normal temporal patterns can be faithfully reconstructed from compressed latent representations, whereas substantial reconstruction errors signal potential abnormalities [6, 7, 5].

Formally, for a given time sequence t, this paradigm employs an autoencoder framework defined as $f _ { \mathrm { r e c } } = \{ f _ { \mathrm { e n c } } , f _ { \mathrm { d e c } } \}$ . The encoder $f _ { \mathrm { e n c } } ~ : ~ \mathbb { R } ^ { m \times C } ~ \to ~ \mathbb { R } ^ { d }$ maps an input time window $\pmb { t } _ { ( i - m + 1 ) : i }$ of length m into a d-dimensional latent feature z:

$$
\begin{array} { r } { z = f _ { \mathrm { e n c } } ( \boldsymbol { t } _ { ( i - m + 1 ) : i } ) . } \end{array}\tag{6}
$$

Subsequently, the decoder $f _ { \mathrm { d e c } } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m \times C }$ reconstructs the original temporal window from this latent representation:

$$
\begin{array} { r } { \hat { \pmb { t } } _ { ( i - m + 1 ) : i } = f _ { \mathrm { d e c } } ( \pmb { z } ) . } \end{array}\tag{7}
$$

The reconstruction error, typically quantified via Mean Squared Error (MSE) [61, 62], serves as the anomaly score:

$$
\mathcal { F } _ { i } ^ { \mathrm { r e c } } = \frac { 1 } { m } \sum _ { k = i - m + 1 } ^ { i } \Vert \pmb { t } _ { k } - \hat { \pmb { t } } _ { k } \Vert _ { 2 } ^ { 2 } .\tag{8}
$$

Anomalies are flagged when $\mathcal { F } _ { i } ^ { \mathrm { r e c } }$ exceeds a predefined threshold δ.

## B. Forecasting Paradigm

As illustrated in Fig. 4(b), the forecasting paradigm prioritizes predicting forthcoming temporal values rather than reconstructing historical observations [27, 52]. In the literature, this approach is alternatively termed predictability modeling or prediction-based detection [63, 6, 56]. This methodology posits that predictive models capture normal evolutionary dynamics, such that unexpected anomalies manifest as prominent discrepancies between forecasted trajectories and actual observations.

![](images/1420069d3d0c8492a71c15e78f26f32faad0db0e7204c1f300fab8040a3bf2be.jpg)  
Fig. 4: Diagram of different paradigms of TSAD.

Formally, a forecasting model $f _ { \mathrm { p r e d } } : \mathbb { R } ^ { m \times C }  \mathbb { R } ^ { p \times C }$ takes a historical observation window $\pmb { t } _ { ( i - m + 1 ) : i }$ of length m to predict the subsequent p time steps:

$$
\begin{array} { r } { \hat { \pmb { t } } _ { ( i + 1 ) : ( i + p ) } = f _ { \mathrm { p r e d } } ( \pmb { t } _ { ( i - m + 1 ) : i } ) . } \end{array}\tag{9}
$$

The prediction error, typically quantified using MSE, acts as the anomaly score for the forecasted horizon:

$$
\mathcal { F } _ { i } ^ { \mathrm { p r e d } } = \frac { 1 } { p } \sum _ { k = i + 1 } ^ { i + p } \| \boldsymbol { t } _ { k } - \boldsymbol { \hat { t } } _ { k } \| _ { 2 } ^ { 2 } .\tag{10}
$$

An anomaly is triggered when this discrepancy exceeds a designated threshold.

Recent empirical evaluations [64, 45, 7, 65] demonstrate that architectures originally designed for pure forecasting can effectively perform TSAD out-of-the-box, provided their intermediate layers extract stable and consistent temporal representations.

## C. Generative Paradigm

As depicted in Fig. 4(c), the generative paradigm models the underlying probabilistic generation process of normal sequences. It fits a generative model to the training set and flags observations exhibiting unusually low likelihoods or substantial deviations from synthetic counterparts [66, 67, 68, 69]. While often incorporating reconstruction-like mechanisms during processing, its core detection principle fundamentally relies on stochastic likelihood or density estimation under a learned data distribution, rather than deterministic point-wise feature mapping. In addition, generative approaches uniquely possess the capability to explicitly sample and synthesize new data from the learned continuous distribution, distinguishing them from strictly discriminative reconstruction models.

Formally, a generative model $G : \mathbb { R } ^ { d }  \mathbb { R } ^ { C }$ (or across a full temporal window $\mathbb { R } ^ { m \times C } )$ maps a latent variable z to the observation space to produce synthetic data t<sup>ˆ</sup>. During training, the parameters θ are optimized to approximate the true data distribution, frequently through maximum likelihood estimation or equivalent objective functions:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { t \sim p _ { \mathrm { d a t a } } , z \sim p ( z ) } \left[ \mathcal { D } \big ( \pmb { t } , G ( z ; \theta ) \big ) \right] ,\tag{11}
$$

where $p _ { \mathrm { d a t a } }$ represents the empirical data distribution and $p ( z )$ is the latent prior.

During inference, for an observed data point $t _ { i } ,$ a corresponding latent representation $z _ { i }$ is inferred or sampled, and the synthetic observation $\hat { \pmb { t } } _ { i } = G ( z _ { i } ; \theta )$ is generated. The anomaly score is determined by a discrepancy metric D:

$$
\mathcal { F } _ { i } ^ { \mathrm { g e n } } = \mathcal { D } ( t _ { i } , \hat { t } _ { i } ) .\tag{12}
$$

Common choices for D include MSE or Kullback-Leibler (KL) divergence [70]. Alternatively, adversarial frameworks such as Generative Adversarial Networks (GANs) may employ a dedicated discriminator network to directly evaluate sample authenticity [71].

This paradigm is distinguished from the Reconstruction Paradigm by its probabilistic training objective, optimizing stochastic objectives such as the evidence lower bound (ELBO), adversarial losses, or score-matching criteria rather than deterministic point-wise residuals alone. It is further distinguished from the Statistical Paradigm in that the data distribution is implicitly captured through neural network parameterization rather than specified in closed analytical form, as detailed in SM ??.

## D. Similarity-based Paradigm

As illustrated in Fig. 4(d), the similarity-based paradigm identifies anomalies by explicitly computing geometric or statistical proximity measures across data samples.

Formally, given a time series t, individual time points or local windows are treated as distinct samples. A generalized similarity function sim(·, ·) is defined to quantify pairwise proximity. Depending on the underlying mathematical formulation, these comparative strategies fall into four primary operational mechanisms, all positing that anomalies exhibit low similarity to the predominant data distribution:

1) Distance-based Evaluation: Anomalies are isolated points situated at substantial distances from the majority cluster. Employing metrics such as Euclidean distance or Dynamic Time Warping (DTW), models flag points that violate designated proximity thresholds.

2) Density-based Evaluation: Normal observations populate high-density regions, whereas anomalies reside in sparse neighborhoods. Algorithms such as Local Outlier Factor (LOF) compute local density estimates, assigning elevated anomaly scores to points lacking dense neighbor clusters.

3) Classification-based Evaluation: Anomaly detection operates as an unsupervised boundary-learning task. Methods such as One-Class Support Vector Machines (OCSVM) construct high-dimensional hyperspheres encapsulating normal data, classifying points lying outside this boundary as anomalies.

4) Tree-structured Implementation: Anomalies are inherently susceptible to rapid isolation due to their lowdensity nature. Architectures such as Isolation Forest recursively partition the feature space; anomalous points require fewer splits, yielding shorter tree path lengths that directly reflect their low similarity to normal dense clusters.

Density-based methods, such as LOF and HBOS, rely on non-parametric neighborhood geometry, distinguishing them from the Statistical Paradigm which fits closed-form distributional parameters. Contrastive and prototype-based approaches, though employing neural feature extractors, are assigned here because their anomaly score derives from geometric distance in embedding space rather than from the backbone architecture.

## E. LLM-based Paradigm

Distinct from classical statistical or connectionist models that learn task-specific error distributions from scratch, the LLMbased paradigm (Fig. 4(e)) introduces a fundamentally distinct detection principle: knowledge-driven semantic inference. By leveraging pre-trained cross-domain historical contexts and zero-shot reasoning capabilities, this paradigm transitions the detection focus from computing traditional mathematical error thresholds to formulating semantic, context-aware judgments [18, 19, 72, 20, 73, 74, 75]. Capitalizing on the autoregressive nature of foundational models, adaptations for TSAD generally align representations and capture temporal context via three primary strategies:

1) Injecting External Knowledge: For zero-shot TSAD tasks, external domain knowledge is directly injected into frozen LLMs via advanced prompt engineering or Retrieval-Augmented Generation (RAG) [76, 18]. Because LLMs map text → text, numerical time series undergo a serialization process Inject : time series → text that transforms raw arrays into descriptive text. For example, a numerical sequence [1, 3, 5, 10] is translated into the sentence: “The series increases from 1 to 10 via intermediate values 3 and 5.” This serialized representation is then paired with a task-oriented system prompt instructing the model to analyze sequence patterns, identify anomalous intervals, and generate diagnostic explanations. Guided by these prompts, the LLM leverages its natural language generation capabilities to pinpoint and explain anomalies through semantic alignment [18].

2) Parameter-Efficient Fine-Tuning Strategies: To enable few-shot adaptation, this strategy freezes core LLM layers, such as multi-head self-attention and feedforward networks containing extensive pre-trained knowledge, fine-tuning only positional embeddings and layer normalization [20]. This adapts the input embedding layer to project numerical sequences into tokenized representations, effectively balancing broad pre-trained cross-domain representations with dataset-specific alignment.

3) Multimodal and Agentic Extensions: Moving beyond strictly sequential or tokenized adaptations, recent innovations harness multimodal large language models (MLLMs) and agentic workflows to enrich the representation space of TSAD. For instance, vision-language models can interpret time series data converted into visual charts or plots, thereby exploiting their mature, pre-trained pattern-recognition capabilities [77]. In complex industrial or surveillance contexts, MLLMs jointly align and analyze heterogeneous streams, including numerical sensors, textual logs, and visual imagery [78, 79]. Furthermore, conceptualizing LLMs as reasoning engines enables multiagent collaborative frameworks, where specialized agents interact iteratively to refine unsupervised representations, handle domain-specific abstractions, and execute complex TSAD tasks [80].

Under this classification, methods such as GPT4TS [20] and Timer [81], whose anomaly scores take the form of prediction errors but whose detection capability derives from pre-trained cross-domain representations, are assigned here rather than to the Forecasting Paradigm.

## F. Statistical Paradigm

As illustrated in Fig. 4(f), the statistical paradigm identifies anomalies by fitting explicit probability distributions or descriptive summary statistics directly to observed temporal sequences. A defining characteristic of this paradigm is that the distributional model takes a closed-form analytical expression whose parameters are estimated without neural optimization, distinguishing it from the Generative Paradigm where the distribution is implicitly encoded in network weights. This typically involves decoupling the periodicity and trend information in time series sequences. Traditional parametric techniques, such as box plots and Z-scores, evaluate anomalies by calculating summary statistics such as the mean and standard deviation. Moving beyond simple distance-based thresholds, log-probability serves as a more expressive metric within this paradigm. In parametric density-based models, such as multivariate Gaussian distributions or Gaussian mixture models, the detector estimates the joint probability distribution of the sequence, flagging observations that exhibit unusually low log density as anomalies.

## V. DISCUSSIONS

This section examines evaluation methodologies, architectural selection criteria, and foundational inductive biases in TSAD, identifying critical bottlenecks and outlining actionable research directions.

## A. Discussions on Evaluations

Evaluating TSAD algorithms requires addressing two interdependent mechanisms: threshold determination and performance metric selection. We subsequently present an experimental analysis assessing the discriminability and stability of diverse evaluation metrics.

1) Anomaly Threshold: Thresholds demarcate nominal observations from anomalous deviations. Conventional approaches predominantly rely on static thresholds established as invariant classification boundaries. While computationally straightforward, static cutoffs degrade rapidly under temporal nonstationarity. In high-frequency operational or financial monitoring, for example, fixed thresholds fail to accommodate structural behavioral shifts triggered by external regime changes. Although input normalization standardizes the marginal distribution P(X), it cannot ensure invariant scoring behavior. Because anomaly scores depend on the learned conditional mapping P(Y|X) [82], distributional shifts in this conditional mapping induced by evolving operational contexts alter score distributions, rendering static thresholds brittle.

This vulnerability underscores the necessity of dynamic thresholding mechanisms capable of continuous adaptation. However, dynamic calibration remains substantially underexplored in TSAD literature. Future investigations should prioritize end-to-end learnable thresholds that infer scenariospecific decision boundaries directly from operational context, eliminating arbitrary heuristic cutoffs.

2) Flawed Metrics: The reliability of comparative evaluations hinges on the choice of scoring metrics. While earlier benchmarks relied primarily on the point-adjustment F1 (F1- PA) score in conjunction with AUC-ROC and AUC-PR, recent theoretical and empirical analyses [83, 84, 85] have exposed the severe inflation bias inherent to F1-PA. Specifically, under point adjustment, correctly detecting a single timestamp within an anomalous segment automatically labels the entire segment as successfully identified. This mechanism artificially inflates precision and recall, allowing even random score generators to achieve deceptively competitive performance on segment anomalies.

Although the community has explored range-aware alternatives, a universally accepted evaluation standard remains elusive. Algorithms frequently optimize for specific metric dimensions without demonstrating consistent superiority across all evaluation axes, precluding rigorous cross-study comparability.

To quantitatively demonstrate the inflation bias of F1-PA against alternative protocols, we present a synthetic case study in SM ??. We strongly advise against relying on F1- PA in empirical benchmarks, as range-wise protocols provide substantially higher evaluation fidelity [86, 87]. Crucially, our experimental findings detailed in Sec. V-A3 confirm that no individual metric encapsulates performance across heterogeneous anomaly densities and sequence horizons [5, 88, 65, 89]. Robust benchmarking necessitates multi-metric evaluations accompanied by open-source implementations and pre-trained checkpoints to guarantee scientific reproducibility.

3) Analysis of Metrics: In this subsection, we experimentally evaluate the benchmark metrics summarized in Table ??, focusing on discriminative power, computational latency, and distributional sensitivity across synthetic and real-world bench marks.

Detailed experimental configurations and sensitivity evaluations across varied scoring backbones, sequence lengths, and segment durations are provided in SM ??, corresponding to Figs. ??–?? and Table ??. These empirical analyses establish four primary findings:

• Robustness and Variance: Point-wise F1 and F1-PA exhibit high vulnerability to score perturbations. F1-PA exhibits substantial score variance and unstable baseline bias, rendering it unsuited for rigorous performance differentiation.

• Metric Discriminability: Range-based metrics display markedly disparate discriminative capacities. Reduced-F1, R-based F1, TaPR, eTaPR, PATE, and AUC-PR frequently fail to separate detectors of distinct competency. Further more, R-based F1 and TaPR are structurally bounded below their theoretical maximums even under perfect detection.

• Computational Latency: Evaluation overhead scales steeply with sequence length and segment density. Because PATE incurs quadratic computational complexity, its deployment becomes computationally prohibitive on datasets exceeding 100,000 timestamps or containing over 50 anomalous segments.

• Sensitivity to Segment Length: Segment duration directly influences metric bias. For brief anomalies with minimal duration variance, metrics such as F1, PA%K, Reduced-F1, TaPR, and AUC-PR present low distortion. Conversely, affiliation-based metrics (NAff-F1, UAff-F1) and volume-under-surface measures (VUS-ROC, VUS-PR) provide superior evaluation fidelity on extended anomalous intervals and high-variance regimes.

Actionable protocols for task-specific metric selection are consolidated in SM ??.

## B. Discussions on Algorithm Architectures

![](images/5c8e7c0925326042e6c101a81b287846d661387bd2066123a499c019f29b3039.jpg)  
Fig. 5: The mapping relationship between algorithm architec tures (categorized by model capability) and the taxonomy of paradigms.

Fig. 5 illustrates the bipartite mapping between detection paradigms and representation architectures. The Taxonomy of Paradigms (left) formalizes mathematical assumptions and anomaly scoring principles, whereas the Algorithm Architecture (right) organizes backbones along a spectrum of representational capacity, spanning closed-form statistical estimators to high-capacity foundational models capable of zero-shot reasoning. Color gradients in Fig. 5 signify this capacity progression, with directional edges marking predominant architectural implementations across paradigms. Detailed algorithmic treatments appear in SM ??.

To delineate architectural trade-offs, Table III summarizes the representative strengths and limitations of each model class. Table IV further translates these trade-offs into practical selection guidelines across operational constraints. Following the color coding of Fig. 5, neutral gray denotes classical machine learning and statistical methods, while pink highlights high-capacity unified and LLM-based architectures. For each decision criterion, candidate architectures are categorized into recommended, secondary, and unsuitable tiers.

To guide practitioners in selecting appropriate detection backbones, we outline six structured recommendations:

1) Multimodal and Zero-Shot Generalization: Prioritize unified and LLM-based architectures when tasks demand cross-domain knowledge transfer, multimodal contextual reasoning, or zero-shot inference across unseen domains.

2) Complex Temporal and Channel Dynamics: Prioritize specialized deep representation architectures, such as Transformer or recurrent backbones [90], over classical machine learning models when capturing intricate longrange temporal dependencies and inter-variable interactions is essential.

3) Mathematical Transparency and Interpretability: Deploy rigorous statistical models and analytical score indicators where decision transparency is non-negotiable. While ensemble methods provide post-hoc feature importance, statistical formulations offer direct, closed-form interpretability.

4) Resource-Constrained and Real-Time Streaming: Favor lightweight statistical techniques and classical machine learning algorithms under stringent computational or latency budgets. Small-scale recurrent or temporal convolutional networks serve as viable deep learning alternatives balancing accuracy and throughput.

5) Peak Empirical Accuracy: Adopt Transformer-based architectures when raw detection precision is the overriding benchmark criterion.

6) Baseline Selection Considerations: Exercise caution when deploying diffusion, state space, or foundational models as baseline comparators, as their substantial training overhead and evolving design spaces make them less suited for minimal reference benchmarks.

## C. Discussions on General Technologies

This subsection highlights core principles governing two widely used methodological strategies in TSAD: time series data augmentation and channel independence. A comprehensive literature taxonomy for both techniques is provided in SM ??.

Time Series Data Augmentation: Data augmentation protocols, including noise injection, synthetic anomaly generation, and sample interpolation, serve to bolster representation robust ness under self-supervised objectives [91, 33, 51]. However, indiscriminate application of naive mixing techniques such as Mixup risks disrupting temporal causality and phase align ment. Practitioners should favor graph-guided or adversarially constrained perturbations that preserve underlying physical constraints while generating informative boundary-case nominal profiles.

Channel Independence: Decomposing multivariate sequences into isolated univariate channels significantly mitigates overfitting and enhances parameter sharing across heterogeneous datasets [92]. Nonetheless, because structural faults and channel-correlation anomalies fundamentally manifest through coordinated inter-sensor couplings, strict channel independence inevitably blinds detectors to multi-channel dynamics. Emerging architectures must harmonize univariate temporal modeling with adaptive cross-channel message passing, leveraging lightweight cross-attention or topological graph layers to dynamically capture synchronized inter-variable anomalies.

## VI. FUTURE AND OPPORTUNITY

This section highlights critical bottlenecks in contemporary TSAD methodologies and outlines prospective directions for future research.

## A. Interpretability

A persistent limitation in TSAD research is the disproportionate focus on benchmark detection accuracy at the expense of operational transparency [86, 89, 65]. In mission-critical deployments, interpretability is as essential as raw accuracy, as transparent decision boundaries allow domain operators to validate alarms and establish operational trust. Existing explainable TSAD models [93, 94, 95] typically rely on heuristic attributions, such as ranking channel correlation coefficients or highlighting localized attention weights. However, such correlational signals fail to uncover underlying causal triggers, phase delay progressions, or physical feedback mechanisms [96, 97].

To provide actionable insights beyond correlational heuristics, meaningful TSAD interpretability must address three core dimensions:

• Temporal and Channel Specificity: Detectors must accurately isolate the exact temporal boundaries of an anomaly while identifying the specific multivariate channels driving the anomalous departure.

• Root Cause Analysis: Beyond merely flagging anomalous occurrences, models should trace causal propagation pathways, identifying how initial variations in specific root cause variables trigger downstream disturbances across interconnected channels over time.

• Contextual Grounding: Models should contextualize anomalous behaviors against nominal dynamics by isolating specific subsequences or shapelets responsible for the divergence, thereby allowing domain experts to inspect and verify the physical faults.

Establishing rigorous evaluation frameworks is equally vital for advancing explainable TSAD. Future progress depends on four methodological initiatives:

• Constructing Interpretable Datasets: Controlled benchmark suites should incorporate verified ground-truth explanations. For instance, synthetic industrial datasets can simulate explicit multivariate interactions, such as coincident spikes in vibration and temperature, providing verifiable benchmarks for cross-channel attribution.

• Structured Anomaly Annotation: Benchmark annotations must advance beyond binary anomaly tags. Datasets should categorize anomaly taxonomies and explicitly link affected time points to their driving root causes, providing supervisory signals for explanatory models.

TABLE III: Comparative analysis of strengths and limitations across representation architectures.
<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>Advantages</td><td rowspan=1 colspan=1>Disadvantages</td></tr><tr><td rowspan=1 colspan=1>Unified TSAD ModelExploring LLM-Powered ModelPredictive LLM-based Model</td><td rowspan=1 colspan=1>①High representational capacity providing a versatileframework for cross-domain and multi-task learning.②Seamless integration with heterogeneous modalities(such as vision and text) via multimodal LLMs for context-rich reasoning.③Facilitates collaborative multi-agent architectures thatleverage generative capabilities for automated diagnosticreasoning.</td><td rowspan=1 colspan=1>①Substantial parameter footprint, demanding memoryoverhead, and elevated inference latency.②Evolving research domain without consistent empiricalmargins over specialized architectures.③Heavy reliance on textual tokenization or 2D imagerendering, causing token bloat in high-frequency numeri-cal time series.</td></tr><tr><td rowspan=1 colspan=1>Diffusion-based Model</td><td rowspan=1 colspan=1>①Exceptional generative capacity for accurately capturingcomplex high-dimensional distributions.②High modeling flexibility with minimal parametricassumptions on data distributions.</td><td rowspan=1 colspan=1>①Elevated computational overhead from iterative reversedenoising steps during both training and inference, limit-ing real-time utility.②High data dependency requiring substantial trainingvolumes to faithfully learn normal temporal baselines.</td></tr><tr><td rowspan=1 colspan=1>Transformer-based Model</td><td rowspan=1 colspan=1>①Highly effective at modeling non-local associations andpairwise variable relationships across extended sequences.②Fully parallelized training over entire temporal se-quences, accelerating model convergence.</td><td rowspan=1 colspan=1>①Quadratic computational and memory complexity withrespect to sequence length, creating scalability bottle-necks.</td></tr><tr><td rowspan=1 colspan=1>State Space Model</td><td rowspan=1 colspan=1>①Linear computational complexity with sequence length,enabling efficient long-horizon temporal processing.②Favorable efficiency-performance trade-off comparedto dense self-attention under equivalent parameter scales.</td><td rowspan=1 colspan=1>①Relatively nascent architectural ecosystem with limitedoff-the-shelf tooling and pre-trained baselines.②Empirical advantages over long horizons require carefulrecurrent state parameterization and domain tuning.</td></tr><tr><td rowspan=1 colspan=1>Recurrent Neural Network ModelGraph Neural Network ModelTemporal Convolution NetworkModel</td><td rowspan=1 colspan=1>①Intrinsic inductive biases tailored for causal sequentialdependencies.②Moderate computational footprint suitable for deploy-ment under tight edge resources.③Explicit structural modeling of inter-variable topologicalgraphs when paired with graph convolutions.</td><td rowspan=1 colspan=1>①Susceptible to gradient vanishing and memory bottle-necks over long temporal windows.©Constrained transferability compared to attention-basedarchitectures, limiting cross-domain generalization.</td></tr><tr><td rowspan=1 colspan=1>Prototype-based Model</td><td rowspan=1 colspan=1>①Explicitly characterizes normal manifolds using com-pact sets of learnable prototypes.②Modular framework easily integrated into diverse metricand representation backbones.</td><td rowspan=1 colspan=1>①Prototype representations require careful regularizationto ensure geometric diversity and prevent representationcollapse.②Scaling to large numbers of prototypes substantiallyincreases optimization difficulty and inference latency.</td></tr><tr><td rowspan=1 colspan=1>Classical Machine Learning</td><td rowspan=1 colspan=1>①High operational transparency; tree ensembles provideintuitive decision pathways and feature importance rank-ings.②Well-suited for low-sample regimes with modest featuredimensionality.</td><td rowspan=1 colspan=1>①Limited representational capacity for capturing com-plex non-linear temporal evolutions or cross-variablecouplings.②Susceptible to the curse of dimensionality, leading toperformance deterioration in high-channel settings.③Requires extensive manual feature engineering.</td></tr><tr><td rowspan=1 colspan=1>Statistical Method</td><td rowspan=1 colspan=1>①Near-instantaneous execution speed, ideal for ultra-high-frequency online streaming.②Rigorous mathematical foundation providing exactprobabilistic interpretation through analytical log-density.③Computationally scalable when formulated with struc-tured marginal or low-rank joint covariance matrices.</td><td rowspan=1 colspan=1>①Rigid parametric assumptions (such as Gaussianity andlinearity) that frequently fail in real-world time series.②Incapable of modeling non-stationary drift and high-order temporal dynamics.</td></tr></table>

TABLE IV: The algorithm selection suggestions under different selection criteria.
<table><tr><td rowspan=1 colspan=1>Criterion</td><td rowspan=1 colspan=1>Recommended</td><td rowspan=1 colspan=1>Suitable</td><td rowspan=1 colspan=1>Not Recommended</td></tr><tr><td rowspan=1 colspan=1>Interpretability</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Long-range Dependencies</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Multi-modality / Multi-domain</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Zero-shot</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Real-time</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>□</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Channel Interaction</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Computational Resources</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Resource &amp; Tool Maturity</td><td rowspan=1 colspan=1>日</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>Classical ML    RNN, GNNSSM     Diff, Trans    Unified, LLM&amp; Stats     TCN, Prot</td></tr></table>

• Developing a Formalized Language for Root Causes: Comparing explainability across diverse models requires a structured descriptive grammar. This representation should systematically record anomaly categories, temporal boundaries, root-cause channels, key subsequences, and secondary cascading effects. Compiling such standardized descriptions into a unified evaluation repository will facilitate rigorous, cross-model benchmarking.

• Advancing Explanatory Evaluation Metrics: Quantitative criteria are needed to evaluate explanation fidelity, particularly explanation sparsity, defined as the minimal feature subset required to justify an alarm, and explanation stability, measured by attribution consistency under minor observational noise [98, 99, 100].

Integrating large language models also offers a promising direction for text-based interpretability [101]. Although LLMs operate with black-box architectures, their natural language generation capabilities enable translating technical attribution maps and multivariate statistical discrepancies into coherent, operator-friendly summaries [73, 20, 52]. This capability can bridge the gap between complex model outputs and operational human decision-making.

## B. Zero-Shot Learning

Zero-shot learning (ZSL) enables detectors to identify previously unseen anomaly patterns without requiring targetdomain training data [64]. Driven by advancements in selfsupervised pre-training, there is growing interest in building unified models that generalize across diverse time series domains [51, 64, 20]. However, achieving reliable zero-shot detection in TSAD involves four primary challenges:

First, data scarcity and annotation bias limit model training. True anomalies are inherently rare in practical operational data, and existing benchmarks rely almost exclusively on binary tags that lack granular descriptions of fault mechanisms. Injecting structured domain knowledge or synthetic perturbations offers a potential avenue to enrich training variety. Second, distributional and behavioral divergence complicates generalization. Anomalies manifest through varied characteristics across domains, ranging from subtle trend shifts in environmental monitoring to sudden transient spikes in industrial machinery. A model trained on a specific set of dynamics often fails when deployed in an unfamiliar regime. Third, structural heterogeneity poses architectural constraints. Real-world applications feature sequences of arbitrary lengths, irregular sampling rates, and variable channel dimensions [52]. Detectors with rigid input layers cannot process these heterogeneous inputs without lossy interpolation or channel selection. Fourth, a clear performance gap remains between few-shot adaptation and true zero-shot inference. Current largescale time series models frequently require fine-tuning on a small fraction of domain-specific samples (such as 5% of the target dataset) to yield acceptable performance [52, 20].

Closing this gap is essential for practical deployment across diverse industries. A truly capable zero-shot detector should accurately identify anomalies upon initial deployment, avoiding the high costs and operational latency associated with targetspecific data collection and retraining.

## C. Multi-Modality Learning

Classical machine learning and deep learning TSAD methods typically focus solely on sequential data. In contrast, LLMbased unified models often incorporate both sequential and textual modalities. One common approach involves projecting sequential data into vector embeddings and feeding them into LLMs. To align these embedding spaces, researchers usually redesign the input layer and fine-tune it on downstream time series datasets. An alternative approach uses prompt learning to format sequential data as text prompts, although directly representing numerical values as text often degrades their intrinsic mathematical semantics.

In parallel, converting time series data into visual representations allows models to leverage the rich priors embedded in pre-trained vision-language models [17]. For example, Insight Miner [102] fine-tunes LLaVA to translate windowed time series images into descriptive language. Similarly, CLIP-LSTM converts stock market data into textual and visual representations, utilizing CLIP features for downstream tasks. Despite these early attempts, research on multi-modal TSAD remains scarce. Given the growing emphasis on multi-modal perception across AI domains, effectively integrating diverse modalities and aligning their embedding spaces represents a critical frontier for TSAD.

Despite these developments, current multimodal TSAD frameworks remain largely limited to shallow feature concatenation [17, 102, 52]. In these setups, auxiliary modalities are merely appended as side inputs, without deep cross-modal reasoning. To establish a more tightly integrated multimodal framework, future research should explore three key opportunities:

First, integrating contextual metadata throughout training: Time series signals in complex engineering systems are accompanied by equipment maintenance logs, sensor topology descriptions, and operational manuals. Incorporating these textual sources directly into the representation space enables models to distinguish benign operating adjustments from genuine equipment faults. Second, multiround conversational feedback: To facilitate collaborative diagnostics, TSAD systems require an interactive loop where human feedback refines model outputs. When operators correct a false alarm through natural language, the system should dynamically update its confidence scores and adjust future predictions. Third, interactive multimodal prompting: Practical diagnostics benefit from flexible operator inputs, combining natural language queries with visual annotations (such as bounding boxes on sensor charts) and temporal window selections. Compiling visual prompts into spatial attention masks and text instructions into logical constraints provides an intuitive interface for root cause investigation.

## D. Multi-Domain Learning

Unlike conventional domain-specific models that require separate training for data from each domain, a unified TSAD model should share parameters across multiple domains and handle various tasks simultaneously. It should exhibit key characteristics including: (1) structural agnosticism, allowing the model to process time series data of different lengths and variables without modifying its architecture, and (2) parameter sharing, enabling the model to utilize shared representations for concurrent task handling [52].

Considering previous discussions, we anticipate that future unified TSAD models need the following characteristics: 1) Zero-shot capability: Current large-scale time series models often require fine-tuning with few-shot data when switching scenarios. Making TSAD models commercially viable will depend on enabling them to perform anomaly detection without any fine-tuning. 2) Dynamic threshold selection: Given the inherent differences in data across domains, the model should adaptively select thresholds based on the current scenario. As discussed, using dynamic thresholds is essential for the current model design. For TSAD models that learn across multiple domains, it becomes even more important—not just to handle fixed thresholds failing over time, but also to deal with the added uncertainty of thresholds varying across domains. 3) Interpretability: Identifying anomalies in time series data is only the initial step. In practical applications, early alerts are essential, but the model must also explain the anomalies to facilitate subsequent tasks, such as identifying root causes and suggesting fixes.

## E. Foundational Models

The emergence of foundational models presents a transformative opportunity for time series anomaly detection, offering potential to handle diverse operational tasks within a unified architecture. Current research directions generally fall into two categories: native time series foundational models and adapted LLM-based architectures.

Native time series foundational models are pre-trained directly on extensive collections of time series data to learn crossdomain temporal dynamics (as detailed in SM ??). A common implementation strategy decomposes multivariate sequences into independent univariate channels, which facilitates zeroshot transfer across datasets with varying channel counts. However, the limited availability of high-quality, open-source time series archives constrains the scale and generalization capabilities of these models compared to their vision and language counterparts. Additionally, relying solely on numerical sequences prevents these models from leveraging external contextual knowledge across domains.

Adapted LLM-based models explore transferring linguistic representations to sequential numerical inputs (as discussed in SMs ?? and ??). While pre-trained LLMs offer strong contextual reasoning and text generation capabilities, they are not inherently designed for continuous numerical signals. Standard text tokenizers are optimized for discrete linguistic tokens, often struggling to capture high-frequency oscillations, exact numerical magnitudes, and temporal phase relationships without precision loss.

Looking forward, a promising trajectory lies in developing unified multimodal foundational models that jointly process numerical time series, structural system topologies, and textual documentation. Such architectures can integrate the numeric precision of native time series encoders with the contextual reasoning of language backbones, providing a robust and generalizable foundation for next-generation anomaly detection.

## VII. CONCLUSION

This survey provides a comprehensive review of advances in time series anomaly detection by establishing a rigorous taxonomy, categorizing modern algorithmic paradigms, and conducting an empirical evaluation of benchmark protocols. By refining anomaly definitions and methodology classifications, this work establishes a structured taxonomy aligned with contemporary research. We evaluate representative approaches, analyzing their operational strengths, computational constraints, and practical suitability while offering concrete algorithmic selection criteria. Additionally, this survey examines modern architectural designs introduced to time series anomaly detection, including diffusion formulations, state space models, large language models, and unified cross-domain architectures. In addition, we categorize evaluation protocols and present empirical analyses addressing latency, stability, and evaluation bias. Finally, we delineate persistent challenges and open frontiers across causal interpretability, zero-shot transfer, multimodal integration, adaptive thresholding, and foundational models, offering a coherent roadmap for future research and operational deployment.

## REFERENCES

[1] K. Choi, J. Yi, C. Park, and S. Yoon, “Deep Learning for Anomaly Detection in Time-Series Data: Review, Analysis, and Guidelines,” IEEE Access, vol. 9, pp. 120 043–120 065, 2021.

[2] H. Alimohammadi and S. Nancy Chen, “Performance evaluation of outlier detection techniques in production timeseries: A systematic review and meta-analysis,” Expert Systems with Applications, vol. 191, p. 116371, 2022.

[3] H. Wang, Z. Luo, J. W. Yip, C. Ye, and M. Zhang, “Ecggan: A framework for effective and interpretable electrocardiogram anomaly detection,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 5071–5081.

[4] G. Zhong, J. Liu, F. Liu, J. Jiang, B. Wang, F. Zhang, and C. L. P. Chen, “Unsupervised textless prompt with llms for multivariate time series anomaly detection,” IEEE Transactions on Computational Social Systems, pp. 1–13, 2026.

[5] Z. Zhong, Z. Yu, X. Xi, Y. Xu, W. Cao, Y. Yang, K. Yang, and J. You, “Simad: A simple dissimilaritybased approach for time-series anomaly detection,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–12, 2025.

[6] Z. Zamanzadeh Darban, G. I. Webb, S. Pan, C. Aggarwal, and M. Salehi, “Deep learning for time series anomaly detection: A survey,” ACM Comput. Surv., vol. 57, no. 1, Oct. 2024.

[7] Y. Yang, C. Zhang, T. Zhou, Q. Wen, and L. Sun, “Dcdetector: Dual attention contrastive representation learning for time series anomaly detection,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 3033– 3045.

[8] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A dataset to support research in the design of secure water treatment systems,” in Critical Information Infrastructures Security, G. Havarneanu, R. Setola, H. Nassopoulos, and S. Wolthusen, Eds. Cham: Springer International Publishing, 2017, pp. 88–99.

[9] J. Jiang, Q. Tang, B. Wang, X. Tang, F. Zhu, K. Yadav, A. Filali, and A. V. Vasilakos, “Deal: Dynamic ensemble algorithm for imbalanced streaming data in uav-assisted consumer iot,” IEEE Transactions on Consumer Electronics, vol. 72, no. 3, pp. 6516–6525, 2026.

[10] S. Ahmad, A. Lavin, S. Purdy, and Z. Agha, “Unsupervised real-time anomaly detection for streaming data,” Neurocomputing, vol. 262, pp. 134–147, 2017.

[11] C. Wang, S. Xing, R. Gao, L. Yan, N. Xiong, and R. Wang, “Disentangled Dynamic Deviation Transformer Networks for Multivariate Time Series Anomaly Detection,” Sensors, vol. 23, no. 3, p. 1104, 2023.

[12] Y. Chen, C. Zhang, M. Ma, Y. Liu, R. Ding, B. Li, S. He, S. Rajmohan, Q. Lin, and D. Zhang, “Imdiffusion: Imputed diffusion models for multivariate time series anomaly detection,” pp. 359–372, 2023.

[13] R. Hu, X. Yuan, Y. Qiao, B. Zhang, and P. Zhao, “Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 9606–9610.

[14] M. A. Ahamed and Q. Cheng, “TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting,” 2024.

[15] A. Zhang and C. Li, “Adaptive state-space mamba for real-time sensor data anomaly detection,” 2025. [Online]. Available: https://arxiv.org/abs/2503.22743

[16] G. Li, M. Ge, J. Wan, D. Han, M. Li, and M. Zhou, “Memmambaad: Memory-augmented state space model for multivariate time series anomaly detection,” Engineering Applications of Artificial Intelligence, vol. 158, p. 111308, 2025.

[17] X. Zhang, R. R. Chowdhury, R. K. Gupta, and J. Shang, “Large language models for time series: A survey,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, K. Lar son, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2024, pp. 8335–8343, survey Track.

[18] Y. Bian, X. Ju, J. Li, Z. Xu, D. Cheng, and Q. Xu, “Multi-patch prediction: Adapting llms for time series representation learning,” International Conference on Machine Learning (ICML), pp. 1–14, 2024.

[19] C. Liu, S. He, Q. Zhou, S. Li, and W. Meng, “Large language model guided knowledge distillation for time series anomaly detection,” in Proceedings of the Thirty Third International Joint Conference on Artificial Intel ligence, ser. IJCAI ’24, 2024.

[20] T. Zhou, P. Niu, L. Sun, R. Jin et al., “One fits all: Power general time series analysis by pretrained lm,” Advances in neural information processing systems, vol. 36, pp. 43 322–43 355, 2023.

[21] Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu, and G. H. Chen, “Ecod: Unsupervised outlier detection using empirical cumulative distribution functions,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 12, pp. 12 181–12 193, 2022.

[22] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 eighth ieee international conference on data mining. IEEE, 2008, pp. 413–422.

[23] J. Jiang, F. Liu, W. W. Ng, Q. Tang, G. Zhong, X. Tang, and B. Wang, “Aerf: Adaptive ensemble random fuzzy algorithm for anomaly detection in cloud computing,” Computer Communications, vol. 200, pp. 86–94, 2023.

[24] Y. Su, R. Liu, Y. Zhao, W. Sun, C. Niu, and D. Pei, “Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network,” Kdd, vol. 1485, pp. 2828–2837, 2019.

[25] H. Zhou, K. Yu, X. Zhang, G. Wu, and A. Yazidi, “Contrastive autoencoder for anomaly detection in multi

variate time series,” Information Sciences, vol. 610, pp. 266–280, 2022.

[26] S. Abilasha, S. Bhadra, P. Deepak, and A. Mathew, “Warping resilient scalable anomaly detection in time series,” Neurocomputing, vol. 511, pp. 22–33, 2022.

[27] D. Huang, L. Shen, Z. Yu, Z. Zheng, M. Huang, and Q. Ma, “Efficient time series anomaly detection by multiresolution self-supervised discriminative network,” Neurocomputing, vol. 491, pp. 261–272, 2022.

[28] Q. Yang, J. Zhang, J. Zhang, C. Sun, S. Xie, S. Liu, and Y. Ji, “Graph transformer network incorporating sparse representation for multivariate time series anomaly detection,” Electronics, vol. 13, no. 11, p. 2032, Jan. 2024.

[29] S. He, G. Li, J. Wang, K. Xie, and P. K. Sharma, “Unidirectional graph structure learning-based multivariate time series anomaly detection with dynamic prior knowl edge,” International Journal of Machine Learning and Cybernetics, May 2024.

[30] X. Huang, N. Chen, Z. Deng, and S. Huang, “Multivariate time series anomaly detection via dynamic graph attention network and informer,” Applied Intelligence, Jun. 2024.

[31] J. Song, K. Kim, J. Oh, and S. Cho, “Memto: Memory guided transformer for multivariate time series anomaly detection,” Advances in Neural Information Processing Systems, vol. 36, pp. 57 947–57 963, Dec. 2023.

[32] Y. Zhang, J. Wang, Y. Chen, H. Yu, and T. Qin, “Adaptive Memory Networks with Self-supervised Learning for Unsupervised Anomaly Detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 4347, no. c, pp. 1–13, 2022.

[33] Z. Zhong, Z. Yu, Z. Fan, C. L. P. Chen, and K. Yang, “Adaptive memory broad learning system for unsupervised time series anomaly detection,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–15, 2024.

[34] I. Pintilie, A. Manolache, and F. Brad, “Time Series Anomaly Detection using Diffusion-based Models,” in 2023 IEEE International Conference on Data Mining Workshops (ICDMW), 2023, pp. 570–578.

[35] S. Schmidl, P. Wenig, and T. Papenbrock, “Anomaly detection in time series: A comprehensive evaluation,” Proceedings of the VLDB Endowment, vol. 15, no. 9, pp. 1779–1797, 2022.

[36] Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, and L. Sun, “Transformers in time series: A survey,” in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China. ijcai.org, 2023, pp. 6778–6786.

[37] P. Yan, A. Abdulkadir, P.-P. Luley, M. Rosenthal, G. A. Schatte, B. F. Grewe, and T. Stadelmann, “A Comprehensive Survey of Deep Transfer Learning for Anomaly Detection in Industrial Time Series: Methods, Applications, and Directions,” IEEE Access, vol. 12, pp. 3768–3789, 2024.

[38] J. Ye, W. Zhang, K. Yi, Y. Yu, Z. Li, J. Li, and

F. Tsung, “A Survey of Time Series Foundation Models: Generalizing Time Series Representation with Large Language Model,” 2024.

[39] J. Su, C. Jiang, X. Jin, Y. Qiao, T. Xiao, H. Ma, R. Wei, Z. Jing, J. Xu, and J. Lin, “Large language models for forecasting and anomaly detection: A systematic literature review,” arXiv preprint arXiv:2402.10350, 2024.

[40] A. A. Cook, G. Misirli, and Z. Fan, “Anomaly Detection for IoT Time-Series Data: A Survey,” IEEE Internet of Things Journal, vol. 7, no. 7, pp. 6481–6494, 2020.

[41] A. Blazquez-Garc´ ´ıa, A. Conde, U. Mori, and J. A. Lozano, “A review on outlier/anomaly detection in time series data,” 2020.

[42] K. Shaukat, T. M. Alam, S. Luo, S. Shabbir, I. A. Hameed, J. Li, S. K. Abbas, and U. Javed, “A Review of Time-Series Anomaly Detection Techniques: A Step to Future Perspectives,” in Advances in Information and Communication, K. Arai, Ed. Cham: Springer International Publishing, 2021, vol. 1363, pp. 865–877.

[43] A. Garg, W. Zhang, J. Samaran, R. Savitha, and C.-S. Foo, “An evaluation of anomaly detection and diagnosis in multivariate time series,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 6, pp. 2508– 2517, 2021.

[44] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, “USAD: UnSupervised Anomaly Detection on Multivariate Time Series,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. Virtual Event CA USA: ACM, 2020, pp. 3395–3404.

[45] Z. Zhong, Z. Yu, Y. Yang, W. Wang, K. Yang, and C. L. P. Chen, “Patchad: A lightweight patch-based mlp-mixer for time series anomaly detection,” IEEE Transactions on Big Data, pp. 1–15, 2025.

[46] Y. Fu, M. Xi, Z. Li, X. Xu, Y. Li, K. Ma, X. Pan, Y. Yang, and J. Yin, “Hadn: A hybrid anomaly detection method for non-uniform multivariate time series,” in International Conference on Service Science. Springer, 2025, pp. 319–336.

[47] B. Yang, W. Long, Y. Zhang, Z. Xi, J. Jiao, and Y. Li, “Multivariate time series anomaly detection: Missing data handling and feature collaborative analysis in robot joint data,” Journal of Manufacturing Systems, vol. 75, pp. 132–149, 2024.

[48] W.-S. Hwang, J.-H. Yun, J. Kim, and B. G. Min, “”do you know existing accuracy metrics overrate timeseries anomaly detections?”,” in Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing, ser. SAC ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 403–412.

[49] W. Lu, J. Wang, X. Sun, Y. Chen, and X. Xie, “Outof-distribution representation learning for time series classification,” in ICLR, 2023, pp. 1–14.

[50] R. Shi, H. Huang, K. Yin, W. Zhou, and H. Jin, “Orthogonality Matters: Invariant Time Series Representation for Out-of-distribution Classification,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge

Discovery and Data Mining, Aug. 2024, pp. 2674–2685.

[51] Q. Shentu, B. Li, K. Zhao, Y. Shu, Z. Rao, L. Pan, B. Yang, and C. Guo, “Towards a general time series anomaly detector with adaptive bottlenecks and dual adversarial decoders,” 2025.

[52] S. Gao, T. Koker, O. Queen, T. Hartvigsen, T. Tsiligkaridis, and M. Zitnik, “Units: A unified multitask time series model,” Advances in Neural Information Processing Systems, vol. 37, pp. 140 589–140 631, 2025.

[53] K.-H. Lai, D. Zha, J. Xu, Y. Zhao, G. Wang, and X. Hu, “Revisiting time series outlier detection: Definitions and benchmarks,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, J. Vanschoren and S. Yeung, Eds., vol. 1, 2021.

[54] M. Braei and S. Wagner, “Anomaly Detection in Univariate Time-series: A Survey on the State-of-the-Art,” 2020.

[55] G. Li and J. J. Jung, “Deep learning for anomaly detection in multivariate time series: Approaches, applications, and challenges,” Information Fusion, vol. 91, pp. 93–102, 2023.

[56] J. E. Zhang, D. Wu, and B. Boulet, “Time Series Anomaly Detection for Smart Grids: A Survey,” 2021.

[57] P. Wenig, S. Schmidl, and T. Papenbrock, “Anomaly detectors for multivariate time series: The proof of the pudding is in the eating,” in 2024 IEEE 40th International Conference on Data Engineering Workshops (ICDEW), 2024, pp. 96–101.

[58] P. Roy, A. Boker, and L. Mili, “Beyond marginals: Learning joint spatio-temporal patterns for multivariate anomaly detection,” arXiv preprint arXiv:2509.15033, 2025.

[59] H. Meng, S. Luo, Z. Liang, Q. Huang, A. Khazraei, and M. Pajic, “Mars: Defending unmanned aerial vehicles from attacks on inertial sensors with modelbased anomaly detection and recovery,” arXiv preprint arXiv:2505.00924, 2025.

[60] S. Maleki and N. Pourmoazemi, “Pi-transformer: A physics-informed attention mechanism for time series anomaly detection,” arXiv preprint arXiv:2509.19985, 2025.

[61] C.-Y. A. Lai, F.-K. Sun, Z. Gao, J. H. Lang, and D. Boning, “Nominality score conditioned time series anomaly detection by point/sequential reconstruction,” Advances in Neural Information Processing Systems, vol. 36, pp. 76 637–76 655, Dec. 2023.

[62] T. Amarbayasgalan, V. H. Pham, N. Theera-Umpon, and K. H. Ryu, “Unsupervised Anomaly Detection Approach for Time-Series in Multi-Domains Using Deep Reconstruction Error,” Symmetry, vol. 12, no. 8, p. 1251, 2020.

[63] A. Garg, W. Zhang, J. Samaran, S. Ramasamy, and C.-S. Foo, “An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 6, pp. 2508–2517, 2022.

[64] S. Wang, J. LI, X. Shi, Z. Ye, B. Mo, W. Lin, J. Shengtong, Z. Chu, and M. Jin, “Timemixer++: A general time

series pattern machine for universal predictive analysis,” in The Thirteenth International Conference on Learning Representations, 2025.

[65] Q. Liu and J. Paparrizos, “The elephant in the room: Towards a reliable time-series anomaly detection bench mark,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 108 231–108 261.

[66] B. Du, X. Sun, J. Ye, K. Cheng, J. Wang, and L. Sun, “Gan-based anomaly detection for multivariate time series using polluted training set,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 12, pp. 12 208–12 219, 2023.

[67] A. Geiger, D. Liu, S. Alnegheimish, A. Cuesta-Infante, and K. Veeramachaneni, “Tadgan: Time series anomaly detection using generative adversarial networks,” 2020 ieee international conference on big data (big data), pp. 33–43, 2020.

[68] S. Liu, B. Zhou, Q. Ding, B. Hooi, Z. Zhang, H. Shen, and X. Cheng, “Time Series Anomaly Detection With Adversarial Reconstruction Networks,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 4, pp. 4293–4306, 2023.

[69] F. Zeng, M. Chen, C. Qian, Y. Wang, Y. Zhou, and W. Tang, “Multivariate time series anomaly detection with adversarial transformer architecture in the Internet of Things,” Future Generation Computer Systems, vol. 144, pp. 244–255, 2023.

[70] A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sengupta, and A. A. Bharath, “Generative adversarial networks: An overview,” IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 53–65, 2018.

[71] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.

[72] K.-H. Lai, L. Wang, H. Chen, K. Zhou, F. Wang, H. Yang, and X. Hu, “Context-aware domain adaptation for time series anomaly detection,” in Proceedings of the 2023 SIAM International Conference on Data Mining (SDM). SIAM, 2023, pp. 676–684.

[73] J. Liu, C. Zhang, J. Qian, M. Ma, S. Qin, C. Bansal, Q. Lin, S. Rajmohan, and D. Zhang, “Large language models can deliver accurate and interpretable time series anomaly detection,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, ser. KDD ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 4623–4634.

[74] Z. Z. Darban, Y. Yang, G. I. Webb, C. C. Aggarwal, Q. Wen, S. Pan, and M. Salehi, “Dacad: Domain adaptation contrastive learning for anomaly detection in multivariate time series,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4485–4496, 2025.

[75] Z. Pang, Y. Luan, J. Chen, and T. Li, “Parinfogpt: An llmbased two-stage framework for reliability assessment of

rotating machine under partial information,” Reliability Engineering & System Safety, p. 110312, 2024.

[76] W. Dang, B. Zhou, L. Wei, W. Zhang, Z. Yang, and S. Hu, “Ts-bert: Time series anomaly detection via pretraining model bert,” in Computational Science–ICCS 2021: 21st International Conference, Krakow, Poland, June 16–18, 2021, Proceedings, Part II 21. Springer, 2021, pp. 209–223.

[77] Z. He, S. Alnegheimish, and M. Reimherr, “Harnessing vision-language models for time series anomaly detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 26, 2026, pp. 21 690– 21 698.

[78] S. Mokhtar, A. Mousakhan, S. Galesso, J. Tayyub, and T. Brox, “Detect, classify, act: Categorizing industrial anomalies with multi-modal large language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 4058–4067.

[79] H. Zhang, X. Xu, X. Wang, J. Zuo, C. Han, X. Huang, C. Gao, Y. Wang, and N. Sang, “Holmes-vad: Towards unbiased and explainable video anomaly detection via multi-modal llm,” arXiv preprint arXiv:2406.12235, 2024.

[80] S. Zhou, Y. Xuan, Y. Ao, X. Wang, T. Fan, and H. Wang, “MERIT: Multi-Agent Collaboration for Unsupervised Time Series Representation Learning,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 24 011–24 028.

[81] Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang, and M. Long, “Timer: Generative pre-trained transformers are large time series models,” in Forty-first International Conference on Machine Learning, 2024.

[82] M. Long, J. Wang, G. Ding, J. Sun, and P. S. Yu, “Transfer feature learning with joint distribution adaptation,” in Proceedings of the IEEE international conference on computer vision, 2013, pp. 2200–2207.

[83] J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, and M. J. Franklin, “Volume under the surface: a new accuracy evaluation measure for timeseries anomaly detection,” Proceedings of the VLDB Endowment, vol. 15, no. 11, pp. 2774–2787, 2022.

[84] A. Huet, J. M. Navarro, and D. Rossi, “Local evaluation of time series anomaly detection algorithms,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 635– 645.

[85] Z. Zhong, Z. Yu, K. Yang, Y. Liu, J. Jiang, and C. L. P. Chen, “Labels Matter More Than Models: Rethinking the Unsupervised Paradigm in Time Series Anomaly Detection,” arXiv e-prints, p. arXiv:2511.16145, Nov. 2025.

[86] M. S. Sarfraz, M.-Y. Chen, L. Layer, K. Peng, and M. Koulakis, “Position: quo vadis, unsupervised time series anomaly detection?” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[87] Z. Zhong, Z. Yu, Y.-m. Cheung, and K. Yang, “Cce: Confidence-consistency evaluation for time series anomaly detection,” arXiv preprint arXiv:2509.01098, 2025.

[88] J. Paparrizos, Y. Kang, P. Boniol, R. S. Tsay, T. Palpanas, and M. J. Franklin, “Tsb-uad: an end-to-end benchmark suite for univariate time-series anomaly detection,” Proceedings of the VLDB Endowment, vol. 15, no. 8, pp. 1697–1711, 2022.

[89] R. Wu and E. J. Keogh, “Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 3, pp. 2421–2429, 2023.

[90] P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, “LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection,” ArXiv preprint, vol. abs/1607.00148, 2016.

[91] H. Xu, Y. Wang, S. Jian, Q. Liao, Y. Wang, and G. Pang, “Calibrated one-class classification for unsupervised time series anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, pp. 1–14, 2024.

[92] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in The Eleventh International Con ference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[93] C. Tang, L. Xu, B. Yang, Y. Tang, and D. Zhao, “Grubased interpretable multivariate time series anomaly detection in industrial control system,” Computers & Security, vol. 127, p. 103094, 2023.

[94] H. Zhu, C. Yi, S. Rho, S. Liu, and F. Jiang, “An interpretable multivariate time-series anomaly detection method in cyber–physical systems based on adaptive mask,” IEEE Internet of Things Journal, vol. 11, no. 2, pp. 2728–2740, 2024.

[95] X. Han, S. Absar, L. Zhang, and S. Yuan, “Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery,” in The Thirteenth International Conference on Learning Representations, Oct. 2024.

[96] A. N. Gummadi, J. C. Napier, and M. Abdallah, “Xai iot: an explainable ai framework for enhancing anomaly detection in iot systems,” IEEE Access, vol. 12, pp. 71 024–71 054, 2024.

[97] J. Sipple and A. Youssef, “A general-purpose method for applying explainable ai for anomaly detection,” in International Symposium on Methodologies for Intelligent Systems. Springer, 2022, pp. 162–174.

[98] V. Jacob, F. Song, A. Stiegler, B. Rad, Y. Diao, and N. Tatbul, “Exathlon: a benchmark for explainable anomaly detection over time series,” Proceedings of the VLDB Endowment, vol. 14, no. 11, pp. 2613–2626, 2021.

[99] Y. Wen, T. Ma, R. Luss, D. Bhattacharjya, A. Fokoue, and A. A. Julius, “Shedding Light on Time Series Classification using Interpretability Gated Networks,” in The Thirteenth International Conference on Learning

Representations, Oct. 2024.

[100] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017.

[101] E. Fons, R. Kaur, S. Palande, Z. Zeng, T. Balch, M. Veloso, and S. Vyetrenko, “Evaluating large language models on time series feature understanding: A comprehensive taxonomy and benchmark,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 21 598– 21 634.

[102] S. Moon, A. Madotto, Z. Lin, T. Nagarajan, M. Smith, S. Jain, C.-F. Yeh, P. Murugesan, P. Heidari, Y. Liu, K. Srinet, B. Damavandi, and A. Kumar, “AnyMAL: An efficient and scalable any-modality augmented language model,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, F. Dernoncourt, D. Preot¸iuc-Pietro, and A. Shimorina, Eds. Miami, Florida, US: Association for Computational Linguistics, Nov. 2024, pp. 1314–1332.

![](images/e66909ca3b7accbad80337c889ed9bed9228d676e6866ee1396b50b0a4eac277.jpg)  
Zhijie Zhong received the B.S. degree in 2022 from the Harbin Engineering University, Harbin, China and he is currently pursuing the Ph.D. degree in the School of Future Technology, South China University of Technology, Guangzhou, China. His research interests include data mining, machine learning, time series analysis, anomaly detection, and large language model (LLM).

![](images/4c8fb648b8e51b73f2822667af09c6427f6a195ca8e4c80a7f1538faadcf8c0c.jpg)

Zhiwen Yu received the Ph.D. degree from the City University of Hong Kong, Hong Kong, in 2008. He is a Professor with the School of Computer Science and Engineering, South China University of Technology, China. He has authored or co-authored more than 200 refereed journal articles and international conference papers, including more than 90 articles in the journals of IEEE Transactions, with an H-index of 56 and Google citations of 16000. He is an Associate Editor of IEEE Transactions on Systems, Man, and Cybernetics: Systems and IEEE Transactions on

Neural Networks and Learning Systems. Ranked among the top 2% of scientists worldwide in Stanford University’s 2024 World’s Top 2% Scientists List.

![](images/52a5b45097c31025825ab527b7e5af312e83e21a7f59d046f4ef72ab65bc0449.jpg)

Pengyu Li received the B.S. degree in 2024 from Nanchang University, Jiangxi, China, and he is currently pursuing the M.S. degree at South China University of Technology, Guangzhou, China. His research interests include machine learning, time series anomaly detection, and large language models (LLMs).

![](images/fb46c7f11cdc1e7eb3709b877b26b16ccdad20bfa48871c2513dbf36c6e31bc6.jpg)  
Liujian Zhang received the B.S. degree from the School of Computer Science and Engineering, South China University of Technology, Guangzhou, China in 2022. He is currently pursuing the Ph.D. degree in the School of Future Technology, South China University of Technology, Guangzhou, China. His research interests include machine learning and transfer learning.

Qiaoli Ni is with HUAWEI Technologies Co., Ltd., Shenzhen, Guangdong 518066, China.

![](images/77ca928fd72bf6c67e1b879451398d6d310921b86648481fd8b84d4f3a9747f4.jpg)

Jun Jiang is currently a post-doctoral researcher with the Department of New Networks, Pengcheng Labora tory, Shenzhen, China. He received his Ph.D. degree in computer science and technology from the South China University of Technology, Guangzhou, China, in 2022. He was a visiting student at Pengcheng Laboratory, Shenzhen, China, from March 2022 to April 2023. He worked as a lecturer with the College of Information Science and Technology, Nanjing Forestry University, Nanjing, China, from 2023 to 2024. His research interests include data stream classification, anomaly detection, cloud computing, and the Internet of Things.

![](images/45b3d821a77e39fa408c721ce55843d59663dc3befe477d09a6c79ce48675908.jpg)

Jianming Lv received the B.S. degree in computer science from Sun Yat-sen University, Guangzhou, China, in 2002, and the Ph.D. degree from the Insti tute of Computing Technology, University of Chinese Academy of Sciences, Beijing, China, in 2008. He is currently an Associate Professor with the South China University of Technology, Guangzhou, China. He has authored or coauthored more than 30 papers with prestigious venues, including IEEE Transactions on Pattern Analysis and Machine Intelligence, the IEEE Transactions on Services Computing, ACM

TOMMCAP, Computer Networks, CVPR, and CIKM. His research interests include data mining, computer vision, and distributed computing and privacy. He is a member the ACM and the CCF.

![](images/f141fa3a939c245fa38b002a8f0bb3ebfa6d6e6a2de15a8743c7e6d18158f716.jpg)

Kaixiang Yang (M’21) received the B.S. degree from the University of Electronic Science and Technology of China, Chengdu, China, in 2012, the M.S. degree from the Harbin Institute of Technology, Harbin, China, in 2015, and the Ph.D. degree from the School of Computer Science and Engineering, South China University of Technology, Guangzhou, China, in 2020. He was a Research Engineer with 7th Research Institute, China Electronics Technology Group Corporation, Guangzhou, China, from 2015 to 2017 and Postdoctoral Researcher with Zhejiang

University from 2021 to 2023. He is currently an Associate Professor with the School of Computer Science and Engineering, South China University of Technology. His research interests include pattern recognition, machine learning, and industrial data intelligence. He is an Associate Editor for Pattern Recognition. He is a senior member of China Computer Federation

![](images/75eed2b4959fe5aefaa7bfde6d246d1f1ac5355f46cf1403de6893ee7ca5d675.jpg)

C. L. Philip Chen (S’88-M’88-SM’94-F’07) gradu ated from the University of Michigan at Ann Arbor, Ann Arbor, MI, USA, in 1985, and the Ph.D. degree in electrical engineering from Purdue University, West Lafaette, Indiana, USA, in 1988.,He is currently the Chair Professor and the Dean of the College of Computer Science and Engineering, South China University of Technology, Guangzhou, China. Being a Program Evaluator of the Accreditation Board of Engineering and Technology Education in the USA, for computer engineering, electrical engineering, and

software engineering programs, he successfully architects the University of Macau’s Engineering and Computer Science programs receiving accreditations from Washington/Seoul Accord through the Hong Kong Institute of Engineers, of which is considered as his utmost contribution in engineering/computer science education for Macau as the former Dean of the Faculty of Science and Technology. His current research interests include cybernetics, systems, and computational intelligence.,Prof. Chen is a Fellow of the American Association for the Advancement of Science, the International Association for Pattern Recognition, the Chinese Association of Automation (CAA), and the Hong Kong Institution of Engineers. He is a member of Academia Europaea, the European Academy of Sciences and Arts, and the International Academy of Systems and Cybernetics Science. He received the IEEE Norbert Wiener Award in 2018 for his contribution in systems and cybernetics, and machine learnings. He was also a highly cited researcher by Clarivate Analytics in 2018 and 2019. He was a recipient of the 2016 Outstanding Electrical and Computer Engineers Award from his Alma Mater, Purdue University, West Lafayette, IN, USA, in 1988. He was the President of the IEEE Systems, Man, and Cybernetics Society from 2012 to 2013. He was the Editor-in-Chief for IEEE Transactions on Systems, Man, and Cybernetics: Systems from 2014 to 2019 and IEEE Transactions on Cybernetics from 2019 to 2021). He was an Associate Editor for IEEE Transactions on Artificial Intelligence and IEEE Transactions on Fuzzy Systems. He was the Chair of TC 9.1 Economic and Business Systems of the International Federation of Automatic Control from 2015 to 2017. He is a Vice-President of the CAA.