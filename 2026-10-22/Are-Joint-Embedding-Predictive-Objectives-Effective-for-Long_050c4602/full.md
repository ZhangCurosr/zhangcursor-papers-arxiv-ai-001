# Are Joint-Embedding Predictive Objectives Effective for Longitudinal EHRs?

Ashvin Gupta<sup>1</sup> Brendan Delaney<sup>1</sup> Alessandra Russo<sup>1</sup>

<sup>1</sup>Imperial College London

## Abstract

Masked language modelling (MLM) is the dominant pretraining objective for transformer encoders applied to longitudinal electronic health records (EHRs). However, reconstructing individual events may overemphasise noisy or weakly informative observations and does not explicitly encourage learning higher-level patient states. Joint-Embedding Predictive Architectures (JEPAs) instead learn by predicting representations in latent space and have shown strong performance in a variety of domains. In this paper, we investigate to what extent JEPA can be adapted to EHR sequence representation. Our empirical investigation using MIMIC-IV shows that although all three of our JEPA formulations train stably without representational collapse, they consistently underperform standard MLM. We find that target size, not target heterogeneity, increases prediction error, yet reducing the target size does not recover downstream performance. This suggests that stable latent prediction alone is insufficient for competitive EHR representation learning and highlights the challenge of defining clinically meaningful latent targets for EHR.

## 1 Introduction

Electronic Health Records (EHRs) are longitudinal records of patient-healthcare interactions. Recent work has increasingly focused on EHR foundation models, which aim to learn general-purpose patient representations through self-supervised learning and can be adapted to a range of downstream clinical prediction tasks [1, 2, 3, 4]. Downstream performance therefore hinges on the quality of these representations.

Due to the sequential nature of EHRs, transformer encoders have become a common backbone for EHR representation learning. A dominant objective for pretraining has been BERT-style masked language modelling (MLM), in which the model is trained to reconstruct masked medical events from their surrounding context [5, 6, 7, 8]. However, exact token reconstruction may not provide the ideal learning signal for longitudinal clinical data. EHR sequences contain routine, repeated, administrative, and partially unpredictable events, yet MLM treats reconstruction of each masked token as equally important. This may encourage representations that capture the identity of local events rather than the true patient state.

In contrast, clinicians use their medical knowledge to abstract away irrelevant detail and focus on key events, trends, and interactions to form a higher-level representation of the patient state. This motivation is closely aligned with Joint-Embedding Predictive Architectures (JEPAs), which provide an alternative form of self-supervision [9, 10]. Rather than reconstructing observations in the input space, JEPA trains a predictor to estimate latent representations of unobserved targets from an available context. This approach has shown strong representation-learning performance in vision [10, 11], world modelling [12] and language [13]. Latent prediction is intended to produce semantically meaningful features and avoid unnecessary reconstruction of unpredictable low-level detail. This is particularly appealing for EHRs, where multiple observed event sequences may correspond to similar underlying clinical states and trajectories.

However, defining a meaningful and predictable JEPA target in longitudinal EHRs is non-trivial. Unlike image patches, EHR events are irregularly sampled, vary substantially in density, and already represent semantically meaningful concepts. Temporal proximity therefore does not necessarily imply semantic similarity, and future clinical events may depend not only on patient state but also on healthcare decisions.

In this work, we ask whether the I-JEPA-style latent prediction framework [10] can be transferred effectively to longitudinal EHR sequences. We evaluate multiple target-construction strategies, including span masking,future masking and random masking of events. We investigate (1) whether these JEPA formulations can be trained stably without representational collapse, (2) how the downstream performance compares, and (3) which objective and training characteristics may explain the observed differences.

## 2 Methodology

## 2.1 Medical Event Representation

A patient trajectory is represented as a timeordered sequence of clinical events, $\begin{array} { r l } { X } & { { } = } \end{array}$ $( e _ { 1 } , e _ { 2 } , \ldots , e _ { N } )$ , where each event is a triplet $\boldsymbol { e } _ { i } ~ = ~ \left( c _ { i } , v _ { i } , t _ { i } \right)$ comprising a discrete medical code $c _ { i } \in \mathcal V .$ , an optional continuous value $v _ { i } \in \mathbb { R } .$ and timestamp $t _ { i }$ . Patient age $a _ { i }$ is derived from $t _ { i }$ and date of birth. Representing time and continuous values is an important design choice in EHR modelling [14, 15, 16]. Rather than discretising continuous recordings into tokens, we retain them directly in the event representation using triplet embeddings [14]. The input embedding for event $e _ { i }$ is

$$
\begin{array} { r } { \mathrm { h } _ { \mathrm { i } } = \mathrm { E } _ { \mathrm { c o d e } } ( c _ { i } ) + \mathrm { P r o j } _ { \mathrm { v a l } } ( v _ { i } ) \cdot \mathbb { I } _ { [ v _ { i } \neq \emptyset ] } + \mathrm { P r o j } _ { \mathrm { a g e } } ( a _ { i } ) , } \\ { ( 1 ) \qquad } \end{array}
$$

where $\operatorname { E } _ { \mathrm { c o d e } }$ is a learned code embedding and I indicates value presence. Sequence order is encoded using Rotary Position Embeddings (RoPE). All models use the same transformer backbone with 6 layers, 8 attention heads, hidden dimension 768, and maximum context length $N = 1 0 2 4$ , ensuring that downstream differences reflect the pretraining objective rather than model capacity.

## 2.2 Objectives

JEPA performance depends strongly on context– target design [13], and as such, we evaluate three EHR target constructions reflecting different assumptions about what should be predictable.

Our first objective, Span JEPA, follows the I-JEPA paradigm as directly as possible [10]. We treat the

![](images/2dc2ccea11479fada5a3c4a31e77be16d55f7c4b332763cd4052ef126373b3d5.jpg)  
Figure 1: The context sequence is encoded by the online encoder, while the complete target sequence is processed by an exponential-movingaverage (EMA) target encoder with gradients stopped through the target branch. A predictor combines the context representations with learnable mask embeddings to estimate target latents, which are matched to the EMA target representations using mean squared error. We evaluate three target constructions: contiguous span masking, prospective prediction of future events, and random masking of individual events.

EHR as a one-dimensional sequence of N clinical events. The complete uncorrupted sequence is passed through the exponential moving average (EMA) target encoder to obtain event-level representations $[ { \bf z } _ { y } = { \bf z } _ { y _ { 1 } } , \ldots , { \bf z } _ { y _ { N } } . ]$ . We sample M contiguous target spans from these representations. The (i)-th target span is defined by the set of event positions $B _ { i } { \mathrm { . } }$ , giving the corresponding target representations $\mathbf { z } _ { y _ { j } } , j \in B _ { i }$ . The target events are removed from the input leaving the remaining events as context. The context encoder produces a set of event representations, which are passed to a shallow two-layer transformer predictor together with learnable mask tokens containing positional information for the target locations. As in I-JEPA, the predictor is applied separately for each target span, producing $\hat { \mathbf { z } } _ { y _ { i } } , j \in B _ { i }$ . The prediction objective is the mean squared distance between the predicted and target representations across the M target spans:

Table 1: Performance (AUROC and AUPRC) on downstream clinical prediction tasks on MIMIC-IV for different pretraining objectives and masking strategies.
<table><tr><td></td><td></td><td colspan="2">ICU Mortality</td><td colspan="2">ICU Readmission</td><td colspan="2">Hospital Mortality</td><td colspan="2">Hospital Readmission</td></tr><tr><td>Loss</td><td>Masking</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>MLM</td><td>Random</td><td>0.774</td><td>0.241</td><td>0.695</td><td>0.269</td><td>0.878</td><td>0.266</td><td>0.720</td><td>0.416</td></tr><tr><td>JEPA</td><td>Span</td><td>0.715</td><td>0.172</td><td>0.642</td><td>0.219</td><td>0.845</td><td>0.210</td><td>0.700</td><td>0.401</td></tr><tr><td>JEPA</td><td>Future</td><td>0.757</td><td>0.216</td><td>0.681</td><td>0.259</td><td>0.870</td><td>0.249</td><td>0.708</td><td>0.409</td></tr><tr><td>JEPA</td><td>Random</td><td>0.715</td><td>0.179</td><td>0.638</td><td>0.215</td><td>0.841</td><td>0.214</td><td>0.691</td><td>0.390</td></tr></table>

$$
\mathcal { L } _ { \mathrm { J E P A } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { \lvert B _ { i } \rvert } \sum _ { j \in B _ { i } } \left. \hat { \mathbf { z } } _ { y _ { j } } ^ { ( i ) } - \mathbf { z } _ { y _ { j } } \right. _ { 2 } ^ { 2 } .\tag{2}
$$

While span masking provides the closest analogue to I-JEPA, EHRs are inherently temporal, and clinically meaningful information may instead lie in how the patient evolves over time. We therefore consider a prospective formulation, Future JEPA, motivated by temporally predictive JEPA and world-model approaches [11]. All events observed up to a time point $T$ form the context, while events satisfying $[ T < t _ { j } \leq T + 2 4 $ , h] form a single future target region $B _ { 1 }$ . This is equivalent to setting $M = 1$ in the JEPA objective above. The architecture and loss are otherwise unchanged.

Finally, we evaluate a random-mask JEPA formulation, Random JEPA, that uses the same random sampling strategy as MLM, but optimises the JEPA objective with $M = 1$ . This provides a controlled comparison with MLM under the same corruption scheme, differing only in whether the model predicts discrete medical codes or continuous target representations.

Together, these objectives test three increasingly different assumptions about predictable structure in EHRs: (1) whether latent clinical information can be recovered from neighbouring events, (2) whether it can be predicted prospectively from earlier patient history, and (3) whether latent reconstruction offers an advantage over token reconstruction when the corruption pattern itself is held fixed.

For all JEPA experiments, we use an EMA target encoder with stop-gradient, rather than the sharedencoder SIGReg alternative [17]. In preliminary experiments, SIGReg was unstable and collapsed, with substantially reduced target-embedding variance. The EMA formulation remained stable and non-collapsed, allowing us to assess the JEPA objective without conflating downstream performance with optimisation failure. Standard MLM is retained as the primary baseline.

## 3 Experiments and Results

Each objective is trained for 50 epochs on the full MIMIC-IV training cohort using the same optimisation setup. Our downstream tasks include ICU mortality, ICU readmission, hospital mortality, and hospital readmission using the same 80:10:10 patient-level split as pretraining. Representation quality is assessed using frozen-encoder linear probes with mean pooling, using the checkpoint with the lowest validation pretraining loss, reporting AUROC and AUPRC. The full hyperparameter sweep is provided in Appendix 5.6.

MLM outperformed all JEPA formulations across downstream tasks, with Future JEPA performing closest to MLM, followed by the two others (Table 1). To determine whether this performance gap was caused by representational collapse, we additionally tracked effective rank and feature standard deviation of the target-encoder embeddings throughout pretraining [18]. Both remained stable while training and validation prediction losses decreased (Appendix 5.3, Figure 3), indicating that the JEPA models converged to non-collapsed solutions despite learning representations that were less useful for downstream clinical prediction.

We further investigated which properties of an EHR target were associated with latent-prediction difficulty. We performed a post-hoc multivariable regression on per-target prediction error using target size, unique-code fraction, event-type diversity, target–context code overlap, and code rarity as covariates (Appendix 5.4). Target size, a key design parameter in I-JEPA [10], was the only factor consistently positively associated with prediction error across all three formulations. In contrast, unique-code fraction was negatively associated with prediction error, while event-type diversity had comparatively small effects. To investigate the effect of target size, we varied the masking ratio to alter target size for Span JEPA. Although smaller targets reduced prediction difficulty, this produced no consistent improvement in downstream performance (Figure 2).

![](images/68113f2485bc7b3130ed296cab747844fd46bf8917d40a3ab88ebf075ccc82f3.jpg)  
Figure 2: Sensitivity of Span-JEPA downstream performance to target masking ratio across the four downstream clinical prediction tasks.

## 4 Discussion

Direct JEPA-style latent prediction can be optimised successfully on longitudinal EHRs without representational collapse, yet the resulting representations remain weaker than MLM on the evaluated downstream tasks. The fact that reducing target size also fails to improve downstream performance suggests that the limitation is not simply optimisation difficulty, but may instead lie in how the prediction target is defined.

A likely explanation is a mismatch in the semantic unit of prediction. In I-JEPA, image patches are relatively low-level observations, and predicting larger regions can encourage abstraction toward higher-level structure [10]. However, EHR tokens already have a very specific semantic meaning associated to them. Predicting masked event-level latents requires reconstructing the semantic meaning of each event, rather than definitely encouraging compression into a coherent patient-state representation. In this setting, increasing the amount of masked information does not necessarily increase semantic abstraction in the same way that predicting larger image regions can. By contrast, MLM retains an explicit token-level objective that directly preserves event identity, which itself is clinically informative. This may also explain why recent JEPA-style EHR work retains explicit token-level supervision alongside latent prediction [19].

The stronger performance of Future JEPA suggests that temporal prediction may provide a more appropriate inductive bias than reconstructing arbitrary internal targets. However, future EHR observations are only partially determined by latent patient state and are also shaped by clinical decisions. The target may therefore contain information about how care is delivered as well as how the patient evolves. Irregular sampling further complicates target construction, since a fixed number of events can span very different clinical intervals, while a fixed temporal window can contain widely varying amounts of information. Consequently, neither token contiguity nor temporal proximity necessarily corresponds to a coherent patient-state target.

These observations suggest that an EHR-specific JEPA may need to operate at a higher semantic level. Rather than predicting one latent representation per recorded event, groups of events could first be aggregated into a smaller number of patient-state representations, with the predictive objective modelling transitions between these states over time. Such targets could better separate persistent clinical state from individual observations, potentially matching the abstraction motivating JEPA more closely.

Overall, our findings suggest that the central challenge in transferring JEPA to longitudinal EHRs is not simply preventing collapse or tuning target difficulty, but defining a prediction unit that corresponds to clinically meaningful patient state. In the formulations studied here, successful latent prediction alone is insufficient to outperform token-level reconstruction.

## References

[1] Ethan Steinberg, Ken Jung, Jason A. Fries, Conor K. Corbin, Stephen R. Pfohl, and Nigam H. Shah. Language Models Are An Effective Patient Representation Learning Technique For Electronic Health Record Data. 5 2020.

[2] Pawel Renc, Yugang Jia, Anthony E. Samir, Jaroslaw Was, Quanzheng Li, David W. Bates, and Arkadiusz Sitek. Zero shot health trajectory prediction using transformer. npj Digital Medicine, 7(1):256, 9 2024. ISSN 2398-6352. doi: 10.1038/s41746-024-01235-0.

[3] Artem Shmatko, Alexander Wolfgang Jung, Kumar Gaurav, Søren Brunak, Laust Hvas Mortensen, Ewan Birney, Tom Fitzgerald, and Moritz Gerstung. Learning the natural history of human disease with generative transformers. Nature, 9 2025. ISSN 0028-0836. doi: 10.1038/s41586-025-09529-3.

[4] Haresh Rengaraj Rajamohan, Xiang Gao, Weicheng Zhu, Shih-Lun Huang, Long Chen, Kyunghyun Cho, Cem M. Deniz, and Narges Razavian. Foundation Models for Clinical Records at Health System Scale. 7 2025.

[5] Yikuan Li, Shishir Rao, Jose Roberto Ayala Solares, Abdelaali Hassaine, Dexter Canoy, Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. BEHRT: Transformer for Electronic Health Records. 7 2019.

[6] Yikuan Li, Mohammad Mamouei, Gholamreza Salimi-Khorshidi, Shishir Rao, Abdelaali Hassaine, Dexter Canoy, Thomas Lukasiewicz, and Kazem Rahimi. Hi-BEHRT: Hierarchical Transformer-based model for accurate prediction of clinical events using multimodal longitudi nal electronic health records. 6 2021.

[7] Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. npj Digital Medicine, 4(1):86, 5 2021. ISSN 2398-6352. doi: 10.1038/s41746-021-00455-y.

[8] Chao Pang, Xinzhuo Jiang, Krishna S Kalluri, Matthew Spotnitz, RuiJun Chen, Adler Perotte, and Karthik Natarajan. CEHR-BERT: Incorporating temporal information from structured EHR data to improve prediction tasks. 11 2021.

[9] Yann LeCun. A Path Towards Autonomous Machine Intelligence. OpenReview, 2022. URL https://openreview.net/pdf?id=BZ5a1r-kVsf.

[10] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. 4 2023.

[11] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. 6 2025.

[12] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorld-Model: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. 6 2026.

[13] Hai Huang, Yann LeCun, and Randall Balestriero. LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures. 10 2025.

[14] Rafi Al Attrach, Rajna Fani, David Restrepo, Yugang Jia, and Peter Schüffler. Rethinking Tokenization for Clinical Time Series: When Less is More. 12 2025.

[15] Maria Elkjær Montgomery, Christian Igel, Mikkel Fruelund Odgaard, Martin Sillesen, and Mads Nielsen. How Should Transformers Encode Numeric Values in Electronic Health Records? In Forty-third International Conference on Machine Learning, 2026. URL https://openreview. net/forum?id=YzlscRoNUj.

[16] Lin Lawrence Guo, Santiago Eduardo Arciniegas, Joseph Jihyung Lee, Adam Paul Yan, George Tomlinson, Jason Fries, and Lillian Sung. Tokenization Tradeoffs in Structured EHR Foundation Models. 3 2026.

[17] Randall Balestriero and Yann LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. 11 2025.

[18] Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10929–10974. PMLR, 8 2023. URL https://proceedings.mlr.press/v202/garrido23a.html.

[19] Irsyad Adam, Zekai Chen, David Laprade, Shaun Porwal, David Laub, Erik Reinertsen, Arda Pekis, and Kevin Brown. The Patient is not a Moving Document: A World Model Training Paradigm for Longitudinal EHR. 1 2026.

## 5 Appendix

## 5.1 Responsible Use

This study evaluates JEPA techniques for retrospective prediction from longitudinal EHRs and does not establish suitability for clinical deployment. Experiments are limited to MIMIC-IV, and performance may not generalise across institutions, patient populations, or care settings. Generative predictions may exhibit uncertainty, miscalibration, and clinically consequential errors. Before any clinical application, models would require external validation, subgroup fairness assessment, calibration analysis, and prospective evaluation under appropriate clinical oversight. Outputs should not be used to guide patient care without such validation and human supervision and as such, this work should be used for research purposes only.

## 5.2 Acknowledgements

This work is supported by UK Research and Innovation (UKRI AI Centre for Doctoral Training in Digital Healthcare grant number EP/Y030974/1), NIHR Imperial Biomedical Research Centre and the NIHR Invention for Innovation grant “Artificial Intelligence to support cancer early diagnosis in general practice" (NIHR207533).

## 5.3 Training Dynamics

![](images/c2f5988977114cabb67fa3bdb50f7b7d7b963150f35b4cb3ad2e3b5453cdc4a8.jpg)  
Figure 3: Training dynamics for the three JEPA formulations over 50 epochs. Rows correspond to Span, Future, and Random JEPA, while columns show prediction loss, effective rank (RankMe) of the target-encoder embeddings, and feature standard deviation. Training and validation prediction losses decrease throughout pretraining, while effective rank and feature variance remain non-degenerate, providing evidence that the models do not undergo representational collapse.

## 5.4 Regression

To investigate which properties of an EHR target are associated with latent-prediction difficulty, we perform a post-hoc regression analysis on the validation set for each JEPA formulation. For every target, we record the JEPA mean squared prediction error (MSE) together with structural characteristics of the target and its relationship to the available context. Because the per-target MSE distribution is strongly right-skewed, we use log(1 + MSE) as the regression outcome. All predictor variables are standardised to zero mean and unit variance before fitting, allowing their coefficients to be compared on a common scale. We fit ordinary least-squares regression models using HC3 heteroskedasticity-robust standard errors and report 95% confidence intervals.

Table 2: Detailed regression results for JEPA prediction error. Coefficients are standardised; confidence intervals and p-values use HC3 robust standard errors.
<table><tr><td>Model</td><td>Predictor</td><td>Std. Coef.</td><td>95% CI</td><td>Robust p-value</td><td>VIF</td></tr><tr><td rowspan="5">Span JEPA</td><td>Target size</td><td>0.121</td><td>[0.120, 0.122]</td><td>&lt; 0.001</td><td>3.03</td></tr><tr><td>Unique-code fraction</td><td>-0.013</td><td>[−0.015, -0.012]</td><td>&lt; 0.001</td><td>3.00</td></tr><tr><td>Event-type diversity</td><td>0.001</td><td>[0.001, 0.002]</td><td>&lt; 0.001</td><td>1.56</td></tr><tr><td>Target-context code overlap</td><td>-0.014</td><td>[−0.015, -0.013]</td><td>&lt; 0.001</td><td>2.28</td></tr><tr><td>Code rarity</td><td>-0.012</td><td>[−0.012, -0.011]</td><td>&lt; 0.001</td><td>1.50</td></tr><tr><td rowspan="5">Future JEPA</td><td>Target size</td><td>0.067</td><td>[0.065, 0.068]</td><td>&lt; 0.001</td><td>3.32</td></tr><tr><td>Unique-code fraction</td><td>-0.069</td><td>[−0.071, -0.068]</td><td>&lt; 0.001</td><td>2.91</td></tr><tr><td>Event-type diversity</td><td>-0.001</td><td>[-0.002, 0.001]</td><td>0.379</td><td>1.86</td></tr><tr><td>Target-context code overlap</td><td>-0.021</td><td>[−0.022, -0.020]</td><td>&lt; 0.001</td><td>1.96</td></tr><tr><td>Code rarity</td><td>-0.023</td><td>-0.025, −0.022]</td><td>&lt; 0.001</td><td>1.56</td></tr><tr><td rowspan="5">Random JEPA</td><td>Target size Unique-code fraction</td><td>0.060</td><td>[0.059, 0.062]</td><td>&lt; 0.001</td><td>6.37</td></tr><tr><td></td><td>-0.040</td><td>[−0.041, -0.039]</td><td>&lt; 0.001</td><td>6.56</td></tr><tr><td>Event-type diversity</td><td>-0.011 -0.018</td><td>-0.011, -0.010]</td><td>&lt; 0.001</td><td>1.26</td></tr><tr><td>Target-context code overlap Code rarity</td><td></td><td>-0.018, -0.017]</td><td>&lt; 0.001</td><td>3.11</td></tr><tr><td></td><td>-0.009</td><td>[-0.010, -0.008]</td><td>&lt; 0.001</td><td>1.20</td></tr></table>

The regression covariates include target size, unique-code fraction, event-type diversity, target–context code overlap, and code rarity. Target size is defined as the number of events being predicted. Unique code fraction is the number of unique medical codes in the target divided by the total number of target events. Event-type diversity is measured using normalised entropy over the event types present in the target. Target–context code overlap is the proportion of unique target codes that are also observed in the context, while code rarity is calculated as the mean negative log-frequency of the target codes using frequencies estimated from the pretraining set. We assess multicollinearity using pairwise correlations and variance inflation factors (VIFs). The regression can be written schematically as

$$
\log ( 1 + \mathrm { M S E } _ { i } ) = \beta _ { 0 } + \beta _ { 1 } \log ( 1 + N _ { \mathrm { t a r g e t } , i } ) + \beta _ { 2 } U _ { i } + \beta _ { 3 } H _ { i } + \beta _ { 4 } O _ { i } + \beta _ { 5 } R _ { i } + \epsilon _ { i } ,\tag{3}
$$

where $N _ { \mathrm { t a r g e t } , i }$ is target size, $U _ { i }$ is unique-code fraction, $H _ { i }$ is event-type diversity, $O _ { i }$ is target– context code overlap, $R _ { i }$ is code rarity.

## 5.5 Context and Target selection

For span masking, let N be the number of real (non-padding) events in the training window, and let r be the mask ratio. The masked-token budget is $B = \mathsf { \bar { \Pi } } [ N r ]$ , clipped to $[ 0 , N ]$ . The number of spans is 4 when $B \geq 4 0$ , and max(1, $\lfloor B / 1 0 \rfloor )$ otherwise. Each span has length has a rough length of $\lfloor \bar { B } / S \rfloor$ where S is the number of spans, and the minimum span length is 10 tokens. Its starting position is drawn independently and uniformly among the positions where a block of that length fits, so spans may overlap. Events that fall in no span are the context.

For future prediction, timestamps are measured in hours from the first event in the same window. The cut $T$ is drawn uniformly from event times that leave at least 10 context events at $\mathrm { t i m e s } \le T$ and at least 8 target events in $( T , T + 2 4 \mathrm { h } ]$ . Events after $T + 2 4 \mathrm { { h } }$ are neither context nor target. If no such cut exists, the last 10% of events in the window are the target and the preceding events are the context.

## 5.6 Hyperparameters

Table 3: Hyperparameter sweep used during pretraining for the three JEPA objectives. Transformer and predictor settings were held fixed.
<table><tr><td>Hyperparameter</td><td>Values</td></tr><tr><td>Learning rate Mask ratio Batch size EMA coefficient</td><td> $1 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 \times 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 }$  0.15, 0.30, 0.50 64, 128, 256 0.996</td></tr><tr><td>Transformer (fixed) Hidden size Layers Attention heads Feed-forward width Dropout Context length Normalization</td><td>768 6 8 3072 0.1 1024</td></tr><tr><td>Position encoding Attention Predictor (fixed) Hidden size Layers 2 Attention heads 8 Feed-forward width 3072</td><td>pre-norm, final LayerNorm RoPE bidirectional 768</td></tr></table>

EMA update: $\theta _ { \mathrm { t g t } }  0 . 9 9 6 \theta _ { \mathrm { t g t } } + 0 . 0 0 4 \theta _ { \mathrm { c o n t e x t } }$

## 5.7 MIMIC Dataset Information

Table 4: MIMIC-IV Population and Task-Specific Distribution
<table><tr><td rowspan="3">Metric / Task</td><td colspan="2">Train</td><td colspan="2">Validation</td><td colspan="2">Test</td></tr><tr><td>Count</td><td>%</td><td>Count</td><td>%</td><td>Count</td><td>%</td></tr><tr><td>Unique Patients</td><td>291,702</td><td>80.0%</td><td>36,463</td><td>10.0%</td><td>36,462</td><td>10.0%</td></tr><tr><td>ICU Mortality</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Case (Positive)</td><td>5,881</td><td>7.8%</td><td>705</td><td>7.7%</td><td>715</td><td>7.7%</td></tr><tr><td>Control (Negative)</td><td>69,169</td><td>92.2%</td><td>8,508</td><td>92.3%</td><td>8,628</td><td>92.3%</td></tr><tr><td>ICU Readmission</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Case (Positive)</td><td>10,745</td><td>14.2%</td><td>1,345</td><td>14.5%</td><td>1,350</td><td>14.3%</td></tr><tr><td>Control (Negative)</td><td>64,978</td><td>85.8%</td><td>7,947</td><td>85.5%</td><td>8,079</td><td>85.7%</td></tr><tr><td>Hospital Mortality</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Case (Positive)</td><td>8,061</td><td>2.9%</td><td>1,014</td><td>2.9%</td><td>988</td><td>2.8%</td></tr><tr><td>Control (Negative)</td><td>271,056</td><td>97.1%</td><td>34,271</td><td>97.1%</td><td>33,840</td><td>97.2%</td></tr><tr><td>Hospital Readmission</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Case (Positive)</td><td>86,859</td><td>20.4%</td><td>11,074</td><td>20.6%</td><td>11,480</td><td>21.2%</td></tr><tr><td>Control (Negative)</td><td>339,559</td><td>79.6%</td><td>42,659</td><td>79.4%</td><td>42,676</td><td>78.8%</td></tr></table>

## 5.8 Linear Probing

Table 5: Downstream linear-probe performance on AUROC and AUPRC with brackets denoting 95% percentile bootstrap confidence intervals from 1,000 stratified resamples
<table><tr><td>Model</td><td>ICU Mortality</td><td>ICU Readmission</td><td>Hospital Mortality</td><td>Hospital Readmission</td></tr><tr><td>AUROC</td><td></td><td></td><td></td><td></td></tr><tr><td>MLM</td><td>0.774 [0.757, 0.792]</td><td>0.695 [0.677, 0.712]</td><td>0.878 [0.867, 0.889]</td><td>0.720 [0.711, 0.728]</td></tr><tr><td>Span JEPA</td><td>0.715 [0.696, 0.734]</td><td>0.642 [0.625, 0.658]</td><td>0.845 [0.832, 0.858]</td><td>0.700 [0.690, 0.711]</td></tr><tr><td>Future JEPA</td><td>0.757 [0.739, 0.775]</td><td>0.681 [0.663, 0.697]</td><td>0.870 [0.859, 0.882]</td><td>0.708 [0.698, 0.719]</td></tr><tr><td>Random JEPA</td><td>0.715 [0.696, 0.734]</td><td>0.638 [0.622, 0.653]</td><td>0.841 [0.827, 0.854]</td><td>0.691 [0.681, 0.702]</td></tr><tr><td>AUPRC</td><td></td><td></td><td></td><td></td></tr><tr><td>MLM</td><td>0.241 [0.214, 0.274]</td><td>0.269 [0.246, 0.294]</td><td>0.266 [0.231, 0.306]</td><td>0.416 [0.398, 0.436]</td></tr><tr><td>Span JEPA</td><td>0.172 [0.154, 0.198]</td><td>0.219 [0.202, 0.238]</td><td>0.210 [0.178, 0.249]</td><td>0.401 [0.376, 0.428]</td></tr><tr><td>Future JEPA</td><td>0.216 [0.193, 0.247]</td><td>0.259 [0.238, 0.285]</td><td>0.249 [0.214, 0.289]</td><td>0.409 [0.383, 0.437]</td></tr><tr><td>Random JEPA</td><td>0.179 [0.159, 0.204]</td><td>0.215 [0.199, 0.232]</td><td>0.214 [0.182, 0.252]</td><td>0.390 [0.367, 0.416]</td></tr></table>