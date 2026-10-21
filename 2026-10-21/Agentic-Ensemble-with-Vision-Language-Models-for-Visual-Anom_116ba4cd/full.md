# Agentic Ensemble with Vision-Language Models for Visual Anomaly

Detection

Shangquan Sun, Wenjun Luo, Xu He, Yoonsoo Lee, and Shijian Lu

Abstract—Visual anomaly detection aims to detect abnormal objects, while anomaly samples are often scarce during model training. Assembling multiple pretrained detectors can help by leveraging their transferable image scores and anomaly maps. However, most existing ensemble methods apply fixed fusion rules over saved predictions which struggle when pretrained detectors clearly disagree due to different training objectives, priors, and visual representations. To this end, we design AnomalyAgent, a post-training fusion framework for accurate visual anomaly detection. AnomalyAgent aligns heterogeneous scores and maps, summarizes score- and region-level disagreement, and treats fusion as a finite action-selection problem. It implements two types of actions: 1) score actions that determine how to fuse image-level detector confidence via consensus, robust aggregation, detector weighting, or score–map consistency; 2) map actions that fuse dense anomaly responses via consensus, detector weighting, geometric aggregation, or disagreement correction. A constrained vision-language model (VLM) is introduced to inspect highconflict samples and select valid score action and map action to produce the final prediction. We evaluate AnomalyAgent on MVTec AD and VisA as primary benchmarks, together with seven additional public anomaly benchmarks and Real-IAD for large-scale score-level evaluation. Across these benchmarks, AnomalyAgent consistently improves over equal-weight fusion and achieves the best mean image AUROC and pixel AP among the compared fusion methods on MVTec AD and VisA, while demonstrating robust transfer across additional anomaly domains.

Index Terms—Visual anomaly detection, post-training ensemble, vision-language model, anomaly localization.

## I. INTRODUCTION

Visual anomaly detection (VAD) aims to detect anomalous samples and localize abnormal regions under limited anomaly supervision, where anomalies are often rare, diverse, and difficult to characterize exhaustively. The prevalent approach mitigates the lack of anomaly supervision by leveraging pretrained anomaly detectors that provide transferable imagelevel anomaly scores and pixel-level localization maps. It substantially improves VAD performance by exploiting visionlanguage representations [1]–[3], feature reconstruction [4], memory modeling [5], [6], state-space architectures [7], [8], foundation-model priors [9], etc. However, different detectors, pretrained under different objectives and priors, often exhibit different failure modes: some may be sensitive to harmless texture variations, some may miss small structural anomalies, and some may produce diffused localization [10], [11]. Their very diverse predictions suggest that pretrained detectors learn complementary evidence and that properly assembling them could outperform any individual detector.

Different ensemble approaches have been explored, including classical strategies such as score averaging [12], weighted voting [13]–[15], and stacking [16]. These methods can improve the detection robustness when validation labels or task-specific fitting are available. However, they struggle while handling label-free post-training visual inspection, where anomaly labels are unavailable, base detectors are fixed, and the fusion module must handle both image-level scores and dense anomaly maps without retraining. Under such a scenario, the most accessible post-training baseline is equalweight averaging that normalizes and averages the scores and anomaly maps of all assembled detectors. Despite being simple and deterministic, this strategy assumes that every detector is equally reliable for every image and every spatial region. As a result, an isolated false response from one detector will be aggregated into the fused anomaly map and distort it, and a subtle but valid anomaly detected by only one expert can be diluted by the others’ predictions. The key challenge thus becomes evidence arbitration: an ensemble method should understand which detector to trust, which response to reject, and how to resolve inter-detector prediction conflicts.

To address this problem, we propose AnomalyAgent, a structured post-training fusion framework for visual anomaly detection. Rather than modifying the pretrained detectors or training an additional meta-detector, AnomalyAgent handles VAD after the pretrained detectors have produced their prediction scores and dense anomaly maps. It normalizes heterogeneous outputs, builds a disagreement-aware evidence state, and invokes a constrained vision-language model (VLM) when the conflict score exceeds a predefined threshold. Here the VLM acts as a single-step policy selector over finite fusion actions, instead of a free-form mask generator or a multi-turn toolusing agent—every valid decision remains grounded in the outputs of the pretrained detectors. For image-level prediction, the available score actions include consensus, robust aggregation, detector weighting, and score–map consistency, allowing the VLM to determine how detector confidence should be combined. For pixel-level localization, the map actions include consensus, detector weighting, geometric aggregation, and disagreement correction, enabling the VLM to suppress unreliable responses or preserve complementary spatial evidence. The selected score and map actions are then applied to the fixed detector outputs to produce the final anomaly score and localization map.

Our main contributions are summarized as follows:

• We formulate label-free transductive post-training anomaly-detector fusion as conflict-conditioned selection over deterministic score- and map-level operators, without updating the base detectors.

• We introduce a constrained VLM policy that makes each intervention traceable and prevents free-form anomaly generation through selection over valid detector-grounded fusion actions.

• Across all six MVTec AD and VisA benchmark– pool configurations, AnomalyAgent consistently improves over equal-weight fusion and achieves the best mean image AUROC and pixel AP among the compared fusion methods on both primary benchmarks, together with robust transfer across seven additional public benchmarks and effective large-scale evaluation on Real-IAD.

## II. RELATED WORK

a) Visual anomaly detection.: Visual anomaly detection aims to identify anomalous images and localize abnormal regions under limited anomaly supervision. MVTec AD [17] and VisA [18] are widely used benchmarks, while SDD/KSDD [19], Real-IAD [20], MVTec 3D-AD [21], Kvasir-SEG [22], and endoluminal scene segmentation benchmarks [23] extend evaluation across surface, 3D, medical, and other anomaly settings.

Existing methods are commonly grouped into embedding-, discrimination-, and reconstruction-based approaches [10], [11]. Embedding-based methods model normal feature distributions or memory banks with pretrained representations, including PaDiM [6], PatchCore [5], CS-Flow [24], Pyramid-Flow [25], and VQ-Flow [26]. Discrimination-based methods learn decision boundaries from synthetic or predefined pseudo anomalies, including CutPaste [27], DRAEM [28], SimpleNet [29], open-set supervised anomaly detection [30], and recent open-set visual anomaly recognition methods [31]. Reconstruction- and distillation-based methods rely on reconstruction errors, teacher–student discrepancies, feature reconstruction, or inversion, including autoencoder-based models [32], [33], Uninformed Students [34], Reverse Distillation [35], UniAD [36], FastRecon [37], OneNIP [38], InvAD [39], Dinomaly [4], self-supervised masking [40], PAD-Net [41], and memory-based coarse-to-fine feature fusion [42]. Recent methods further explore state-space architectures, multimodal sensing, and foundation-model priors for stronger multi-class or cross-domain generalization [7]–[9], [43]. Different from these works, we do not design another base detector, but combine image-level scores and dense anomaly maps from fixed pretrained detectors at test time.

b) Vision-language anomaly detectors.: Large visionlanguage models such as CLIP [44] have improved anomaly detection transfer across categories and domains. WinCLIP [1] uses normal/abnormal text prompts and multi-scale window features, while AnomalyCLIP [2], AdaCLIP [45], and VCP-CLIP [3] adapt anomaly, textual, or visual-context prompts for classification and segmentation. Other studies exploit language guidance and vision-language feature interaction [46], [47]. Beyond prompt adaptation, ACR [48] and MuSc [49] use distributional or mutual-scoring mechanisms, while PromptAD [50], one-normal-prompt learning [38], and InCtrl [51] study prompt-based or in-context adaptation with limited normal samples. These methods mainly improve a single detector or adapt a vision-language backbone. In contrast, AnomalyAgent uses a VLM as a constrained selector over multiple detector outputs, choosing only actions grounded in existing anomaly evidence rather than generating anomaly maps freely.

c) Post-training ensemble fusion.: Ensemble learning has been widely studied through averaging, weighted voting, selective ensembles, and stacking [12]–[16], [52]. Recent anomaly-detection systems also explore feature- and modality level fusion, including memory-guided multimodal anomaly detection and coarse-to-fine feature fusion [42], [43]. These approaches combine complementary evidence, but generally fuse representations within jointly designed detectors. Posttraining fusion instead operates on fixed detectors, often without anomaly labels, and must combine both scalar image scores and dense anomaly maps despite image- and regiondependent detector reliability. Equal-weight averaging is a useful baseline, but assumes uniform reliability across images, regions, and output types. AnomalyAgent instead conditions fusion on detector disagreement and structured visual evidence while retaining averaging as a safe reference.

d) Vision-language agents for expert arbitration.: Recent language and vision-language agents reason over observations, intermediate states, and external tools [53]–[58]. Related anomaly-detection work has explored multimodal evidential reasoning and training-free VLM decision processes, including open-world video anomaly detection and uncertainty-guided hierarchical retrieval [59], [60]. Direct VLM mask generation, however, can be unstable or weakly grounded for quantitative dense localization. AnomalyAgent therefore restricts the VLM to selecting score- and map-level actions for high-conflict cases from the image, detector scores and maps, candidate regions, disagreement summaries, and a bounded action set. It cannot directly draw a mask or emit a new anomaly score, so the final prediction remains a reproducible transformation of detector outputs.

## III. METHOD

## A. Post-Training Anomaly-Detector Ensemble

We consider a post-training ensemble of fixed anomaly detectors for visual anomaly detection. For a test image x<sub>i</sub>, detector $k \in \{ 1 , \ldots , K \}$ outputs an image-level anomaly score $s _ { i k } \ \in \ \mathbb { R }$ and a dense anomaly map $\mathbf { \bar { \boldsymbol { M } } } _ { i k } \in \mathbb { R } ^ { H _ { i } \times W _ { i } }$ where $H _ { i }$ and $W _ { i }$ are the map height and width. The detector pool is fixed: AnomalyAgent does not update detector weights, prompts, memory banks, or reconstruction modules. The fusion problem is to map the prediction set

$$
\mathcal { P } _ { i } = \{ ( s _ { i k } , M _ { i k } ) \} _ { k = 1 } ^ { K }\tag{1}
$$

to a final image-level anomaly score $\hat { s } _ { i }$ and pixel-level anomaly map $\hat { M _ { i } }$

Because pretrained detectors may have different score ranges and map calibrations, we first apply group-wise robust normalization:

$$
\tilde { s } _ { i k } = \operatorname { N o r m } _ { g _ { i } } ( s _ { i k } ) , \qquad \tilde { M } _ { i k } = \operatorname { N o r m } _ { g _ { i } } ( M _ { i k } ) ,\tag{2}
$$

![](images/81b1962811684473d398d474fdaa0e49fa11eef2dc700b270cbf1c16169abd78.jpg)  
Fig. 1: AnomalyAgent overview. Pretrained anomaly detectors produce image-level scores and dense anomaly maps. Their outputs are normalized and summarized into a structured evidence state. The VLM is invoked when the conflict score exceeds a fixed threshold and chooses among fixed image- and pixel-level fusion actions.

where $g _ { i }$ denotes the evaluation group of image $x _ { i } ,$ , such as object category, dataset category, or benchmark-defined subset when such metadata is available. $\mathrm { N o r m } _ { g _ { i } }$ denotes percentileclipped min–max scaling within that group. This label-free transductive calibration uses detector outputs and metadata only, and does not require anomaly labels. The standard averaging-based ensemble is

$$
{ \bar { s } } _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { \tilde { s } } _ { i k } , \qquad { \bar { M } } _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { \tilde { M } } _ { i k } .\tag{3}
$$

Although $\bar { s } _ { i }$ and $\bar { M _ { i } }$ provide simple and deterministic references, they assume that all detectors are equally reliable for every image, region, and output type. In contrast, AnomalyAgent replaces this fixed assumption with image-conditioned evidence arbitration.

## B. Structured Evidence State

For each image, AnomalyAgent summarizes both the detector predictions and their internal disagreement into a structured evidence representation. We first characterize pixel-level disagreement using the detector response range:

$$
R _ { i } ( u ) = \operatorname* { m a x } _ { k } \tilde { M } _ { i k } ( u ) - \operatorname* { m i n } _ { k } \tilde { M } _ { i k } ( u ) ,\tag{4}
$$

where u denotes a pixel location. A large $R _ { i } ( u )$ indicates that the detectors produce substantially different responses at that location. However, raw disagreement may be dominated by isolated noisy pixels. To emphasize compact high-response regions, we further define the top-response functional

$$
T _ { q } ( M ) = \frac { 1 } { | \Omega _ { q } ( M ) | } \sum _ { u \in \Omega _ { q } ( M ) } M ( u ) ,\tag{5}
$$

where $\Omega _ { q } ( M )$ denotes the set containing the top-q fraction of pixels ranked by their responses in map M. Compared with the maximum response, $T _ { q }$ is less sensitive to isolated spikes, while remaining more responsive to compact abnormal regions than a global average. Based on these quantities, we construct a label-free collection of derived evidence statistics

$$
\Delta _ { i } = \left( R _ { i } , T _ { q } ( R _ { i } ) , \{ T _ { q } ( \tilde { M } _ { i k } ) \} _ { k = 1 } ^ { K } , T _ { q } ( \bar { M } _ { i } ) , \mathcal { C } _ { i } \right) ,\tag{6}
$$

where $\mathcal { C } _ { i }$ denotes the candidate high-response regions extracted from the fused and detector-specific anomaly maps. Thus, $\Delta _ { i }$ summarizes spatial disagreement, detector-specific high-response statistics, fused-map responses, and candidate regions without using anomaly labels. The complete structured evidence state is then defined as

$$
E _ { i } = \left( x _ { i } , \tilde { s } _ { i 1 : K } , \tilde { M } _ { i 1 : K } , \bar { s } _ { i } , \bar { M } _ { i } , \Delta _ { i } \right) .\tag{7}
$$

Therefore, $E _ { i }$ contains the original image, normalized detector scores and maps, the equal-weight fusion reference, and derived disagreement evidence. This representation provides the information used by the subsequent conflict trigger and constrained VLM fusion policy.

## C. Thresholded High-Conflict Agent Invocation

To improve efficiency, AnomalyAgent invokes the VLM when the pretrained detectors exhibit substantial internal disagreement. Before VLM call, we first compute a deterministic preliminary fusion

$$
( s _ { i } ^ { 0 } , M _ { i } ^ { 0 } ) = \Big ( F _ { s } ( a ^ { s , 0 } ; \tilde { s } _ { i 1 : K } , \tilde { M } _ { i 1 : K } ) , F _ { m } ( a ^ { m , 0 } ; \tilde { M } _ { i 1 : K } ) \Big )\tag{8}
$$

where $F _ { s }$ and $F _ { m }$ denote the image-score and pixel-map fusion operators, respectively, and $a ^ { s , 0 }$ and $a ^ { m , 0 }$ are predefined fallback actions selected from the same finite action space used by the VLM. The preliminary fusion is retained when the VLM is not invoked or when its output cannot be parsed, while equal-weight fusion $( \bar { s } _ { i } , \bar { M } _ { i } )$ serves as a neutral reference for measuring disagreement. We then define a label-free conflict score

$$
\begin{array} { r } { C _ { i } = \lambda _ { s } D _ { s } ( i ) + \lambda _ { m } D _ { m } ( i ) + \lambda _ { p } D _ { p } ( i ) + \lambda _ { r } D _ { r } ( i ) , } \end{array}\tag{9}
$$

TABLE I: Detector-grounded fusion-action families for the constrained VLM policy in image-score and pixel-map arbitration.
<table><tr><td>Family</td><td>Image-score operator  $F _ { s }$ </td><td>Pixel-map operator  $F _ { m }$ </td></tr><tr><td>Consensus</td><td> $\begin{array} { r } { \frac { 1 } { K } \sum _ { k } \tilde { s } _ { i k } } \end{array}$ </td><td> $\begin{array} { r } { \frac { 1 } { K } \sum _ { k } \tilde { M } _ { i k } } \end{array}$ </td></tr><tr><td>Robust consensus</td><td> $\mathrm { \ddot { m e d i a n } } _ { k } ( \tilde { s } _ { i k } )$ </td><td> $\tilde { \mathrm { m e d i a n } } _ { k } ( \tilde { M } _ { i k } )$ </td></tr><tr><td>Detector weighting</td><td> $\textstyle \sum _ { k } w _ { k } \tilde { s } _ { i k }$ </td><td> $\Sigma _ { k } w _ { k } \tilde { M } _ { i k }$ </td></tr><tr><td>Score-map consistency</td><td> $\begin{array} { r } { ( 1 - \alpha ) \sum _ { k } w _ { k } \tilde { s } _ { i k } + \alpha T _ { q } ( M ) } \end{array}$ </td><td></td></tr><tr><td>Disagreement correction</td><td></td><td> $\mathrm { c l i p } ( M - \eta R _ { i } , 0 , 1 ) \mathrm { o r } \mathrm { c l i p } ( M + \eta R _ { i } , 0 , 1 )$ </td></tr><tr><td>Geometric consensus</td><td></td><td> $\begin{array} { r } { \exp \Bigl ( \frac { 1 } { K } \sum _ { k } \log ( \tilde { M } _ { i k } + \epsilon ) \Bigr ) } \end{array}$ </td></tr></table>

where $\lambda _ { s } , \lambda _ { m } , \lambda _ { p } , \lambda _ { r } \geq 0$ are hyperparameters controlling the relative contributions of different disagreement cues. These cues are defined as

$$
\begin{array} { r l r } & { D _ { s } ( i ) = \mathrm { S t d } _ { k } \big ( \tilde { s } _ { i k } \big ) , } & { D _ { m } ( i ) = T _ { q } ( R _ { i } ) , } \\ & { D _ { p } ( i ) = \big | T _ { q } ( M _ { i } ^ { 0 } ) - T _ { q } ( \bar { M } _ { i } ) \big | , } & { D _ { r } ( i ) = \big | s _ { i } ^ { 0 } - \bar { s } _ { i } \big | . } \end{array}\tag{10}
$$

Here, $D _ { s }$ measures image-score disagreement across detectors, $D _ { m }$ summarizes spatial disagreement in high-response regions, $D _ { p }$ measures the deviation of the preliminary fused map from the equal-weight reference, and $D _ { r }$ measures the corresponding deviation at the image-score level. These terms are not intended to estimate anomaly likelihood; instead, they quantify internal inconsistency within the detector ensemble. After that, the VLM is invoked when the conflict score exceeds a threshold:

$$
{ \mathcal { T } } _ { \tau } = \{ i : C _ { i } \geq \tau \} .\tag{11}
$$

Samples in $\mathcal { T } _ { \tau }$ are sent to the VLM for fusion-action selection, while the remaining samples keep the preliminary fusion result. We use a fixed intervention threshold, which does not use anomaly labels.

This conflict-triggered design separates anomaly likelihood from arbitration need: a highly anomalous image does not require additional reasoning if the detectors already agree, whereas even a low-scoring image may require arbitration when the detector outputs are inconsistent. By reserving VLM inference for high-conflict cases, AnomalyAgent avoids unnecessary computation on samples for which the detector ensemble already provides consistent evidence.

## D. Constrained VLM Fusion Policy

We formulate the VLM as an agent that observes a structured evidence state and selects score- and map-level actions from a bounded fusion-action space. For each selected sample $i \in \mathcal { T } _ { \tau }$ , the VLM receives a structured visual input together with a textual evidence summary derived from $E _ { i }$ . The visual input contains the original image, individual detector maps, the equal-weight reference, detector disagreement, and candidateregion overlays under a common normalization. The textual summary provides detector scores, map statistics, conflict cues, candidate regions, and the valid fusion actions.

Rather than generating a new anomaly score or mask, the VLM acts as a constrained selector over two finite action spaces: $\mathcal { A } _ { i } ^ { s }$ for image-score fusion and $\mathcal { A } _ { i } ^ { m }$ for pixel-map fusion. Its policy is written as

$$
\left( a _ { i } ^ { s } , a _ { i } ^ { m } \right) = \pi _ { \mathrm { V L M } } \left( E _ { i } , \boldsymbol { A } _ { i } ^ { s } , \boldsymbol { A } _ { i } ^ { m } \right) ,\tag{12}
$$

where $a _ { i } ^ { s } \in \mathcal { A } _ { i } ^ { s }$ and $a _ { i } ^ { m } \in \mathcal { A } _ { i } ^ { m }$ denote the selected score- and map-level actions, respectively.

The available action families are summarized in Table I. Image-level actions include score consensus, robust aggregation, detector weighting, and score–map consistency, while pixel-level actions include map consensus, robust aggregation, detector weighting, geometric consensus, and disagreement correction. In the score–map consistency family, M denotes the map summary paired with the score action. The weights $w _ { k }$ are nonnegative and sum to one, while $\alpha , \eta ,$ and ϵ control score–map coupling, disagreement correction, and numerical stability, respectively.

The VLM returns only valid action identifiers together with an optional short rationale used for interpretation. The parser accepts only identifiers from the predefined action sets; invalid or incomplete outputs revert to the fallback action. Consequently, the VLM can use the original image and detector evidence to determine how the predictions should be fused, but the final score and anomaly map always remain deterministic transformations of the pretrained detector outputs.

## E. Final Prediction

The final prediction combines the fallback fusion with VLM-selected actions:

$$
\begin{array} { r l } & { \hat { s } _ { i } = \left\{ \begin{array} { l l } { F _ { s } ( a _ { i } ^ { s } ; \tilde { s } _ { i 1 : K } , \tilde { M } _ { i 1 : K } ) , } & { i \in \mathbb { Z } _ { \tau } , } \\ { s _ { i } ^ { 0 } , } & { i \notin \mathbb { Z } _ { \tau } , } \end{array} \right. } \\ & { \hat { M } _ { i } = \left\{ \begin{array} { l l } { F _ { m } ( a _ { i } ^ { m } ; \tilde { M } _ { i 1 : K } ) , } & { i \in \mathbb { Z } _ { \tau } , } \\ { M _ { i } ^ { 0 } , } & { i \notin \mathbb { Z } _ { \tau } . } \end{array} \right. } \end{array}\tag{13}
$$

All components of the fusion protocol, including the fallback actions, VLM prompt, parser, action space, and conflicttriggering rule, are fixed before evaluation. Therefore, AnomalyAgent requires neither anomaly labels nor detector retraining or task-specific meta-training at evaluation time.

## IV. EXPERIMENTS

## A. Experimental Settings

a) Benchmarks and metrics.: We evaluate AnomalyAgent on MVTec AD [17], VisA [18], and seven additional benchmarks: BTAD [62], KSDD/SDD [19], DAGM [63], DTD [64], Magnetic Tile [65], KolektorSDD2 [66], and MVTec LOCO [67]. Real-IAD [20] is additionally used as a large-scale diagnostic for the CLIP-based detector pool. We report macro-averaged image AUROC, image AP, image F1, pixel AUROC, pixel AP, and pixel F1, all in percentage.

Equal-weight

Ours

Ground Truth  
TABLE II: Complete comparison on MVTec AD across the three detector pools. All values are percentages, and the best result within each detector-pool block is boldfaced.
<table><tr><td rowspan="2"></td><td colspan="3">Image-level</td><td colspan="3">Pixel-level</td></tr><tr><td>AUROC</td><td>AP</td><td>F1</td><td>AUROC</td><td>AP</td><td>F1</td></tr><tr><td colspan="7">CLIP-based detector pool</td></tr><tr><td>AdaCLIP [45]</td><td>99.17</td><td>99.66</td><td>98.34</td><td>98.99</td><td>72.31</td><td>71.47</td></tr><tr><td>AnomalyCLIP [2]</td><td>91.56</td><td>96.36</td><td>92.75</td><td>91.05</td><td>31.28</td><td>38.37</td></tr><tr><td>WinCLIP [1]</td><td>93.56</td><td>96.86</td><td>93.52</td><td>96.40</td><td>49.74</td><td>54.21</td></tr><tr><td>Equal-weight [12]</td><td>98.80</td><td>99.50</td><td>97.95</td><td>97.83</td><td>64.43</td><td>64.87</td></tr><tr><td>Inverse-distance [61]</td><td>98.79</td><td>99.43</td><td>97.74</td><td>97.67</td><td>62.18</td><td>62.65</td></tr><tr><td>Voting [13]–[15]</td><td>97.13</td><td>98.68</td><td>96.30</td><td>94.98</td><td>51.06</td><td>57.25</td></tr><tr><td>Stacking [16]</td><td>99.23</td><td>99.68</td><td>98.41</td><td>98.07</td><td>69.03</td><td>69.27</td></tr><tr><td>AnomalyÅgent</td><td>99.25</td><td>99.69</td><td>98.45</td><td>98.99</td><td>72.31</td><td>71.47</td></tr><tr><td colspan="7">Feature-reconstruction detector pool</td></tr><tr><td>MambaAD [7]</td><td>98.54</td><td>99.52</td><td>97.77</td><td>97.64</td><td>51.85</td><td>58.18</td></tr><tr><td>OneNIP [38]</td><td>97.98</td><td>99.35</td><td>97.60</td><td>97.85</td><td>58.47</td><td>60.75</td></tr><tr><td>InvAD [39]</td><td>95.53</td><td>97.91</td><td>94.88</td><td>95.93</td><td>35.47</td><td>43.52</td></tr><tr><td>Equal-weight [12]</td><td>98.98</td><td>99.69</td><td>98.28</td><td>98.18</td><td>59.67</td><td>61.52</td></tr><tr><td>Inverse-distance [61]</td><td>98.93</td><td>99.68</td><td>98.45</td><td>98.16</td><td>58.74</td><td>61.30</td></tr><tr><td>Voting [13]-[15]</td><td>96.30</td><td>98.24</td><td>96.06</td><td>95.92</td><td>44.09</td><td>54.39</td></tr><tr><td>Stacking [16]</td><td>98.99</td><td>99.69</td><td>98.31</td><td>98.08</td><td>57.83</td><td>60.45</td></tr><tr><td>AnomalyAgent</td><td>99.11</td><td>99.74</td><td>98.62</td><td>98.27</td><td>60.99</td><td>62.66</td></tr><tr><td colspan="7">Hybrid detector pool</td></tr><tr><td>OneNIP [38]</td><td>97.98</td><td>99.35</td><td>97.60</td><td>97.85</td><td>58.38</td><td>60.69</td></tr><tr><td>InvAD [39]</td><td>95.53</td><td>97.91</td><td>94.88</td><td>95.93</td><td>35.41</td><td>43.47</td></tr><tr><td>AdaCLIP [45]</td><td>99.17</td><td>99.66</td><td>98.34</td><td>98.99</td><td>72.28</td><td>71.44</td></tr><tr><td>Equal-weight [12]</td><td>99.65</td><td>99.90</td><td>99.04</td><td>98.93</td><td>72.98</td><td>72.04</td></tr><tr><td>Inverse-distance [61]</td><td>99.62</td><td>99.90</td><td>99.09</td><td>98.87</td><td>70.70</td><td>69.29</td></tr><tr><td>Voting [13]–[15]</td><td>97.87</td><td>98.90</td><td>97.55</td><td>93.25</td><td>51.72</td><td>62.17</td></tr><tr><td>Stacking [16]</td><td>99.71</td><td>99.92</td><td>99.21</td><td>98.85</td><td>72.63</td><td>72.13</td></tr><tr><td>AnomalyAgent</td><td>99.72</td><td>99.92</td><td>99.31</td><td>99.13</td><td>74.61</td><td>73.45</td></tr></table>

(a) MVTec AD, CLIP-based detector pool (Detector 1: AdaCLIP, Detector 2: AnomalyCLIP, Detector 3: WinCLIP)  
![](images/4ceb2191ffbb6937e98b2a13ed8be0640462885b848e3d06595668769b2d18da.jpg)  
Input  
Detector 1  
Detector 2  
Detector 3  
Error (Equal-weight)  
Error (Ours)

Fig. 2: Qualitative comparison of detector disagreement and post-training fusion across three representative samples. Each row shows the input image, ground-truth mask, three candidate detector responses, equal-weight fusion, AnomalyAgent, and GT referenced error maps for the two fusion results. The MVTec AD hybrid example exhibits the strongest detector disagreement, while the other two examples show more moderate differences between equal-weight fusion and AnomalyAgent.

b) Detector pools.: On MVTec AD and VisA, we evaluate three heterogeneous three-detector pools. The CLIPbased pool contains AdaCLIP [45], AnomalyCLIP [2], and WinCLIP [1]. The feature-reconstruction pool contains MambaAD [7], OneNIP [38], and InvAD [39]. The hybrid pool contains OneNIP, InvAD, and AdaCLIP. These pools cover CLIP-based anomaly detectors, feature- or reconstructionbased detectors, and cross-paradigm detector combinations.

c) Baselines.: The comparison includes the three individual candidate detectors in each pool, equal-weight fusion [12], inverse-distance fusion, voting-based fusion [13]–[15], and stacking-style fusion [16]. Equal-weight fusion uses the same sample alignment, map resolution, and percentile normalization as AnomalyAgent, and then averages image scores and pixel maps. The other baselines provide training-free alternatives over the same saved detector outputs. The stacking-style baseline is an unsupervised score-statistic heuristic rather than a supervised meta-learner. Box fusion [68] and Vote-NMS [69] are reported separately as diagnostic map-to-box adaptations because they discard dense localization structure.

TABLE III: Complete comparison on VisA across the three detector pools. All values are percentages, and the best result within each detector-pool block is boldfaced.
<table><tr><td></td><td colspan="3">Image-level</td><td colspan="3">Pixel-level</td></tr><tr><td>Method</td><td>AUROC</td><td>AP</td><td>F1</td><td>AUROC</td><td>AP</td><td>F1</td></tr><tr><td colspan="7">CLIP-based detector pool</td></tr><tr><td>AdaCLIP [45]</td><td>93.75</td><td>94.35</td><td>89.78</td><td>95.79</td><td>28.38</td><td>38.39</td></tr><tr><td>AnomalyCLIP [2]</td><td>81.97</td><td>85.34</td><td>80.44</td><td>95.43</td><td>14.14</td><td>23.12</td></tr><tr><td>WinCLIP [1]</td><td>81.67</td><td>83.68</td><td>81.44</td><td>88.77</td><td>15.26</td><td>22.41</td></tr><tr><td>Equal-weight [12]</td><td>91.19</td><td>92.26</td><td>87.38</td><td>97.90</td><td>30.06</td><td>38.56</td></tr><tr><td>Inverse-distance [61]</td><td>91.11</td><td>92.35</td><td>87.22</td><td>97.68</td><td>28.98</td><td>37.95</td></tr><tr><td>Voting [13]–[15]</td><td>88.66</td><td>90.07</td><td>85.58</td><td>93.54</td><td>18.48</td><td>31.43</td></tr><tr><td>Stacking [16] AnomalyÅgent</td><td>93.66 93.88</td><td>94.21 94.43</td><td>89.81</td><td>97.92 97.95</td><td>30.76</td><td>39.15</td></tr><tr><td></td><td></td><td></td><td>90.07</td><td></td><td>31.62</td><td>40.54</td></tr><tr><td colspan="7">Feature-reconstruction detector pool</td></tr><tr><td>MambaAD [7]</td><td>94.64</td><td>95.35</td><td>90.26</td><td>98.38</td><td>26.98</td><td>37.38</td></tr><tr><td>OneNIP [38]</td><td>93.32</td><td>95.22</td><td>90.18</td><td>98.78</td><td>28.80</td><td>38.30</td></tr><tr><td>InvAD [39]</td><td>95.22</td><td>96.55</td><td>91.90</td><td>97.53</td><td>20.46</td><td>30.56</td></tr><tr><td>Equal-weight [12]</td><td>96.83</td><td>97.56</td><td>93.92</td><td>99.13</td><td>37.39</td><td>44.35</td></tr><tr><td>Inverse-distance [61]</td><td>96.78</td><td>97.45</td><td>93.54</td><td>99.15</td><td>37.28</td><td>44.10</td></tr><tr><td>Voting  $[ 1 3 ] \mathrm { - } [ 1 5 ]$ </td><td>81.65</td><td>84.66</td><td>81.04</td><td>96.91</td><td>18.50</td><td>28.57</td></tr><tr><td>Stacking [16]</td><td>96.76</td><td>97.51</td><td>93.97</td><td>98.81</td><td>29.07</td><td>40.03</td></tr><tr><td>AnomalyÅgent</td><td>96.92</td><td>97.62</td><td>94.10</td><td>99.15</td><td>37.69</td><td>44.86</td></tr><tr><td colspan="7">Hybrid detector pool</td></tr><tr><td>OneNIP [38]</td><td>93.32</td><td>95.22</td><td>90.18</td><td>98.63</td><td>30.08</td><td>39.72</td></tr><tr><td>InvAD [39]</td><td>95.22</td><td>96.55</td><td>91.90</td><td>97.46</td><td>21.79</td><td>32.26</td></tr><tr><td>AdaCLIP [45]</td><td>93.75</td><td>94.35</td><td>89.78</td><td>95.62</td><td>30.04</td><td>40.37</td></tr><tr><td>Equal-weight [12]</td><td>98.07</td><td>98.47</td><td>95.16</td><td>99.30</td><td>45.62</td><td>51.72</td></tr><tr><td>Inverse-distance [61]</td><td>97.73</td><td>98.23</td><td>95.02</td><td>99.32</td><td>45.36</td><td>51.76</td></tr><tr><td>Voting [13]-[15]</td><td>92.90</td><td>94.19</td><td>90.09</td><td>97.64</td><td>29.59</td><td>41.66</td></tr><tr><td>Stacking [16]</td><td>98.06</td><td>98.46</td><td>95.03</td><td>99.09</td><td>43.15</td><td>50.12</td></tr><tr><td>AnomalyAgent</td><td>98.48</td><td>98.83</td><td>95.73</td><td>99.33</td><td>45.66</td><td>51.89</td></tr></table>

d) Implementation details.: For each benchmark setting, the conflict score in Eq. (9) is computed for all samples, and we use a fixed intervention threshold of $\tau = 0 . 5 .$ . Samples with conflict scores above the threshold select one imagescore action and one pixel-map action from the predefined action bank, while the remaining samples use a fixed settingspecific fallback action. Invalid VLM outputs revert to the corresponding fallback action. We use LLaVA-7B as the VLM backend with temperature 0.1 and a maximum generation length of 160 tokens. All detector outputs are normalized by 1st–99th percentile scaling within category using the unlabeled evaluation split, and the same normalization is applied to all ensemble baselines. Dense anomaly maps are resized to the ground-truth mask resolution before pixel-level evaluation. The conflict score uses $\lambda _ { s } = \lambda _ { m } = 0 . 3 5$ and $\lambda _ { p } = \lambda _ { r } = 0 . 1 5$ with a top-response fraction of $q = 0 . 0 1$ . The finite action bank uses predefined detector-weighting vectors, score–map coupling coefficients $\alpha \in \{ 0 . 1 5 , 0 . 2 0 , 0 . 4 0 \}$ , disagreementcorrection strengths $\eta \in \{ 0 . 0 2 , 0 . 1 5 \}$ , and $\epsilon = 1 0 ^ { - 6 }$ for geometric aggregation. No anomaly labels are used for evaluationtime normalization, triggering, or per-sample VLM action selection.

e) Constrained agent prompt.: AnomalyAgent uses a constrained prompt interface. The VLM receives detector evidence and selects one image-level and one pixel-level fusion action from the predefined action sets. The prompt is instantiated with the dataset, category, detector scores, conflict statistics, candidate tools, and valid fusion actions.

## V. RESULTS

## A. Primary Benchmark Results

Tables II and III report six-metric comparisons on MVTec AD and VisA across three detector pools. All ensemble methods use the same saved detector outputs, category-wise normalization, and map resolution. The strongest individual detector varies across datasets, pools, and metrics, while fixed fusion rules do not consistently improve both image- and pixellevel performance.

AnomalyAgent improves over equal-weight fusion in all six primary configurations, with the largest gain on MVTec AD using the CLIP-based pool, especially for pixel-level localization. These results show that detector reliability varies across both images and spatial responses, motivating adaptive score- and map-level fusion.

## B. Transfer and Large-Scale Evaluation

We further evaluate on BTAD [62], KSDD/SDD [19], DAGM [63], DTD [64], Magnetic Tile [65], KolektorSDD2 [66], and MVTec LOCO [67]. As shown in Table IV, AnomalyAgent consistently improves over equal-weight fusion and remains competitive with or stronger than other fixed fusion strategies without detector retraining or dataset-specific fitting.

a) Large-scale Real-IAD evaluation.: On the full Real-IAD benchmark [20], Table V shows that AnomalyAgent improves image AUROC/AP/F1 from 72.47/58.85/59.66 to 75.53/62.54/62.49, confirming its effectiveness at larger scale.

TABLE IV: Transfer evaluation on seven public anomaly benchmarks. The compared individual detectors are AdaCLIP [45], AnomalyCLIP [2], and WinCLIP [1]. All values are percentages, and the best results within each benchmark are boldfaced.
<table><tr><td>Metric</td><td>AdaCLIP [45]</td><td>AnomalyCLIP [2]</td><td>WinCLIP [1]</td><td>Equal [12]</td><td>InvDist [61]</td><td>Voting [13]–[15]</td><td>Stacking [16]</td><td>AnomalyAgent</td></tr><tr><td colspan="9">BTAD</td></tr><tr><td>Image AP Image F1 Pixel AUROC Pixel AP Pixel F1</td><td>91.32 94.46 88.92 93.64 41.32 46.56</td><td>89.12 91.05 85.99 93.25 35.59 42.84</td><td>75.55 68.09 67.22 95.81 34.46 42.07</td><td>94.22 95.19 89.78 97.25 49.29 53.09</td><td>93.80 94.70 90.49 96.95 47.93 52.35</td><td>91.15 93.79 88.44 92.96 31.69 38.75</td><td>94.21 95.01 90.12 96.45 47.25 51.33</td><td>95.31 96.08 92.96 97.30 49.87 53.82</td></tr><tr><td colspan="9">KSDD/SDD</td></tr><tr><td>Image AP Image F1 Pixel AUROC Pixel AP Pixel F1</td><td>96.21 90.82 86.24 94.15 14.90 29.34</td><td>96.06 85.47 82.57 97.68 19.79 34.42</td><td>90.83 81.12 78.00 97.72 15.93 29.24</td><td>95.99 90.27 85.71 98.21 34.17 48.97</td><td>94.68 88.12 82.69 98.10 34.02 48.97</td><td>95.40 83.26 85.44 95.16 21.28 39.45</td><td>96.33 91.12 87.13 98.47 33.49 48.46</td><td>96.92 91.77 89.32 98.43 34.65 48.97</td></tr><tr><td colspan="9">DAGM 96.43 99.89</td></tr><tr><td>Image AP Image F1 Pixel AUROC Pixel AP Pixel F1</td><td>99.94 99.83 79.95 0.95 2.72</td><td>92.19 91.68 59.79 0.44 2.09</td><td>92.95 89.99 63.35 0.48 1.66</td><td>99.61 98.79 68.70 0.65 2.85</td><td>99.70 99.08 72.33 0.68 2.85</td><td>99.45 98.74 44.56 0.38 2.80</td><td>99.94 99.83 70.66 0.67 2.73</td><td>99.94 99.89 74.03 0.74 2.86</td></tr><tr><td colspan="9">DTD 94.46 97.26 99.26 99.09 97.70 98.80 99.67 99.56 94.53 96.18 98.58</td></tr><tr><td>Pixel AUROC Pixel AP Pixel F1</td><td>99.48 72.71 76.80</td><td>97.52 44.49 54.94</td><td>98.20 51.61 60.69</td><td>99.11 73.00 74.47 Magnetic Tile</td><td>98.36 99.11 72.41 73.42</td><td>98.39 98.08 61.96 71.20</td><td>99.08 99.10 74.36 76.02</td><td>99.25 99.20 75.12 76.82</td></tr><tr><td colspan="9">67.35 73.87 72.74 50.61 59.42 58.10 49.68 53.71 52.83</td></tr><tr><td>Pixel AUROC Pixel AP Pixel F1 Image AUROC</td><td>75.49 17.07 24.10</td><td>69.50 6.80 12.86 90.81</td><td>72.87 15.49 19.21 93.50</td><td>75.97 14.26 21.53 KolektorSDD2</td><td>75.60 13.00 19.81</td><td>67.77 6.69 12.65 95.43</td><td>75.66 14.42 22.34 95.51</td><td>57.57 76.74 15.61 24.42 96.49</td></tr><tr><td colspan="9">95.61 95.47 75.27 88.98 88.26 71.37 85.02 82.18</td></tr><tr><td>Pixel AUROC Pixel AP Pixel F1</td><td>98.42 69.19 66.65</td><td>97.87 47.70 54.13 61.02</td><td>84.20 5.18 13.04</td><td>98.87 45.81 57.22 MVTec LOCO</td><td>99.03 45.95 57.09</td><td>94.25 31.12 52.83</td><td>98.91 43.65 57.08</td><td>87.44 98.97 46.66 57.75</td></tr><tr><td colspan="9">59.36 63.98 62.79 71.16 75.85 75.29</td></tr><tr><td>Image F1 Pixel AUROC Pixel AP Pixel F1</td><td>76.95 58.86 9.20 16.05</td><td>76.87 64.21 10.22 16.85</td><td>76.96 72.81 15.48 21.41</td><td>77.15 72.35 12.90 21.07</td><td>77.07 71.29 12.37 20.15</td><td>76.63 69.86 12.37 22.83</td><td>72.61 13.11 21.29</td><td>72.81 15.48 21.41</td></tr><tr><td colspan="9">Average over seven benchmarks</td></tr><tr><td></td><td>88.10 86.80 84.49 85.71</td><td>85.57 81.66 79.34 82.83 23.58</td><td>82.90 76.86 75.63 83.57</td><td>88.97 87.00 84.11 87.21</td><td>88.35 86.25 83.24 87.49</td><td>84.19 83.80 80.38</td><td>88.80 87.08 84.90 87.41</td><td>89.91 88.12 86.24 88.21</td></tr></table>

## C. Robustness and Diagnostic Analysis

a) Detector complementarity.: Table VI reports pairwise score correlations and high-response-region overlap for the CLIP-based pool.

The detectors show moderate score agreement but substantially different spatial responses. On VisA, WinCLIP has topresponse IoUs of only 0.048 and 0.042 with AdaCLIP and AnomalyCLIP, respectively, supporting the need for adaptive fusion.

b) Calibration robustness.: Table VII compares default transductive calibration with 1st–99th percentile statistics estimated from 64 stored samples per category. Performance remains comparable or improves, indicating that AnomalyAgent is not critically dependent on the full evaluation split.

c) Calibration-size sensitivity.: We further vary the number of unlabeled calibration samples on MVTec AD.

Performance improves from 16 to 32 samples and remains stable for 32–128 samples, showing that reliable normalization requires only limited calibration data.

TABLE V: Large-scale score-level evaluation on Real-IAD with the CLIP-based detector pool. All values are percentages, and the best result is boldfaced.
<table><tr><td>Method</td><td>AUROC</td><td>AP</td><td>F1</td></tr><tr><td>AdaCLIP [45]</td><td>74.15</td><td>61.44</td><td>60.88</td></tr><tr><td>AnomalyLI [2]</td><td>68.60</td><td>53.28</td><td>56.32</td></tr><tr><td>WinCLIP [1]</td><td>63.45</td><td>45.88</td><td>57.25</td></tr><tr><td>Equal-weight [12]</td><td>72.47</td><td>58.85</td><td>59.66</td></tr><tr><td>AnomalyAgent</td><td>75.53</td><td>62.54</td><td>62.49</td></tr></table>

TABLE VI: Pairwise complementarity of AdaCLIP, AnomalyCLIP, and WinCLIP on MVTec AD and VisA. Score correlation is measured by Spearman correlation, while spatial overlap is measured by the IoU of high-response regions. Lower IoU indicates more diverse spatial responses.
<table><tr><td>Dataset</td><td>Detector Pair</td><td></td><td>Score Spearman Top-response IoU</td></tr><tr><td rowspan="3">MVTec AD</td><td>AdaCLIP-AnomalyCLIP</td><td>0.615</td><td>0.160</td></tr><tr><td>AdaCLIP-WinCLIP</td><td>0.502</td><td>0.163</td></tr><tr><td>AnomalyCLIP-WinCLIP</td><td>0.511</td><td>0.114</td></tr><tr><td rowspan="3">VisA</td><td>AdaCLIP-AnomalyCLIP</td><td>0.761</td><td>0.182</td></tr><tr><td>AdaCLIP-WinCLIP</td><td>0.472</td><td>0.048</td></tr><tr><td>AnomalyCLIP-WinCLIP</td><td>0.436</td><td>0.042</td></tr></table>

d) Action-bank diagnostic.: We progressively enlarge the fusion-action bank on the same replayed triggered samples.

Detector weighting yields a small gain over consensus-only actions, while the full bank provides a further modest improvement; geometric consensus and score–map consistency are not selected in this replay.

e) Selector diagnostic.: Table X compares fallback-only, random, deterministic-rule, and historical VLM selection on VisA.

The deterministic selector improves over fallback and random selection, while the historical VLM result performs best. Because reconstructed trigger indices do not exactly match the original VLM trace, this is a diagnostic replay rather than a strict attribution of gains to VLM reasoning.

f) Setting-level robustness.: Table XI reports descriptive resampling over the six primary benchmark–pool settings; the ranges characterize setting-level rather than image-level variation.

g) Diagnostic box-based ensembles.: We additionally convert dense anomaly maps into boxes and evaluate WBF and Vote-NMS.

Both substantially underperform equal-weight fusion, particularly on pixel-level metrics, indicating that box-based fusion is poorly suited to dense anomaly localization.

h) VLM invocation cost.: Because all methods reuse saved detector predictions, we report only the additional VLM calls introduced by AnomalyAgent. Only high-conflict samples invoke the VLM, while other samples use deterministic fusion and failed calls revert to valid fallback actions.

## D. Qualitative Analysis

Figure 2 shows three representative cases. When detector predictions agree, AnomalyAgent makes only limited changes to equal-weight fusion; under stronger disagreement, it suppresses unsupported diffuse responses while preserving compact anomaly evidence. These examples illustrate how adaptive fusion resolves detector disagreement without generating new anomaly maps.

TABLE VII: Calibration robustness for the CLIP-based detector pool. Fixed-prefix calibration estimates the 1st–99th percentile bounds from 64 stored samples per category without using anomaly labels.
<table><tr><td>Dataset</td><td>Calibration</td><td>Image AUROC</td><td>Pixel AP</td></tr><tr><td rowspan="2">MVTec AD</td><td>Transductive</td><td>98.84</td><td>64.42</td></tr><tr><td>Fixed prefix (64)</td><td>98.75</td><td>64.71</td></tr><tr><td rowspan="2">VisA</td><td>Transductive</td><td>88.19</td><td>30.14</td></tr><tr><td>Fixed prefix (64)</td><td>89.16</td><td>32.86</td></tr></table>

TABLE VIII: Calibration-size sensitivity of AnomalyAgent on the MVTec AD CLIP-based detector pool. Normalization statistics are estimated from a fixed unlabeled prefix of each category and then kept frozen.
<table><tr><td>Size</td><td>Im. AUROC</td><td>Im. AP</td><td>Im. F1</td><td>Pix. AUROC</td><td>Pix. AP</td><td>Pix. F1</td></tr><tr><td>16</td><td>97.77</td><td>98.97</td><td>96.81</td><td>97.98</td><td>63.90</td><td>65.44</td></tr><tr><td>32</td><td>98.28</td><td>99.22</td><td>97.20</td><td>97.92</td><td>64.24</td><td>65.26</td></tr><tr><td>64</td><td>98.75</td><td>99.48</td><td>97.80</td><td>98.05</td><td>64.71</td><td>65.62</td></tr><tr><td>128</td><td>98.80</td><td>99.49</td><td>97.90</td><td>97.84</td><td>64.24</td><td>65.27</td></tr></table>

TABLE IX: Diagnostic action-bank replay on MVTec AD with the CLIP-based detector pool. All variants use the same replayed triggered samples; unavailable historical actions are deterministically remapped to valid actions in the restricted bank.
<table><tr><td>Action Bank</td><td>Pixel AP Pixel F1</td><td></td><td>Newly Used Action</td></tr><tr><td>A1: Equal + Median</td><td>64.433</td><td>64.867</td><td></td></tr><tr><td>A2: + Detector Weighting</td><td>64.471</td><td>64.905</td><td>Weighting</td></tr><tr><td>A3: + Geometric/Consistency</td><td>64.471</td><td>64.905</td><td>None</td></tr><tr><td>A4: Full Bank</td><td>64.485</td><td>64.918</td><td>Disagreement correction</td></tr></table>

## E. Discussion and Limitations

Across detector pools and benchmarks, AnomalyAgent consistently improves over fixed label-free fusion strategies, with particularly strong gains in localization metrics and robust transfer to additional anomaly benchmarks. The method is also insensitive to reduced calibration data and keeps each fusion decision traceable to detector outputs, conflict statistics, and selected actions. Its main limitation is the dependence on the quality and diversity of the candidate detectors; future work may explore broader action spaces and additional VLM backends.

## VI. CONCLUSION

We presented AnomalyAgent, a post-training visionlanguage agent for anomaly-detector fusion. Rather than training an additional detector, AnomalyAgent normalizes the outputs of pretrained detectors, identifies conflicting evidence, and uses a constrained VLM to select among predefined image-score and pixel-map fusion actions. Across MVTec AD,

TABLE X: Diagnostic selector replay on VisA with the CLIP-based detector pool. Fallback-only, random, and deterministic-rule variants use the same reconstructed conflict-trigger set and intervention budget. Random selection reports mean ± standard deviation over eight fixed seeds.
<table><tr><td>Selector</td><td>Image AUROC</td><td>Image AP</td><td>Image F1</td><td>Pixel AUROC</td><td>Pixel AP</td><td>Pixel F1</td></tr><tr><td>Fallback only</td><td>91.26</td><td>92.41</td><td>87.34</td><td>97.84</td><td>30.22</td><td>38.45</td></tr><tr><td>Random action selector</td><td> $9 1 . 2 4 \pm 0 . 0 4$ </td><td> $9 2 . 3 7 \pm 0 . 0 2$ </td><td> $8 7 . 2 1 \pm 0 . 0 5$ </td><td> $9 7 . 7 5 \pm 0 . 1 4$ </td><td> $3 0 . 0 9 \pm 0 . 1 5$ </td><td> $3 8 . 3 4 \pm 0 . 1 2$ </td></tr><tr><td>Deterministic rule selector</td><td>92.72</td><td>93.01</td><td>89.84</td><td>97.90</td><td>30.91</td><td>39.21</td></tr><tr><td>Historical VLM result</td><td>93.88</td><td>94.43</td><td>90.07</td><td>97.95</td><td>31.62</td><td>40.54</td></tr></table>

TABLE XI: Setting-level resampling summary on the six primary MVTec AD / VisA detector-pool settings. Values are percentage-point deltas; brackets show 95% percentile ranges over 10,000 setting-level resamples.
<table><tr><td>Comparison</td><td></td><td>Image AUROC</td><td>Image AP</td><td>Pixel AUROC</td><td>Pixel AP</td><td>Pixel F1</td></tr><tr><td>Direct VLM — Equal</td><td></td><td>+0.627  $\left[ + 0 . 1 3 5 , + 1 . 4 8 9 \right]$ </td><td>+0.447 [+0.033, +1.160]</td><td>+0.244 [+0.040, +0.592]</td><td>+1.981 [+0.548, +4.175]</td><td> $+ 1 . 7 9 3 \ [ + 0 . 6 0 7 , + 3 . 5 6 8 ]$ </td></tr></table>

TABLE XII: Diagnostic box-based fusion baselines averaged over the six MVTec AD and VisA settings. Values denote the performance difference relative to equal-weight fusion in percentage points.
<table><tr><td>Baseline</td><td>Image AUROC</td><td>Image AP</td><td>Image F1</td><td>Pixel AUROC</td><td>Pixel AP</td><td>Pixel F1</td></tr><tr><td>WBF</td><td>-6.556</td><td>-5.488</td><td>-4.861</td><td>-2.789</td><td>-23.498</td><td>-17.579</td></tr><tr><td>Vote-NMS</td><td>-16.447</td><td>-14.258</td><td>-8.519</td><td>-3.817</td><td>-30.400</td><td>-24.512</td></tr></table>

TABLE XIII: Fusion-stage compute accounting. Detector inference is excluded because all fusion methods replay the same saved detector dumps. The VLM backend is LLaVA-7B.
<table><tr><td>Experiment</td><td>Scope</td></tr><tr><td>Main Direct VLM-Agent MVTec AD / VisA primary settings</td><td>489</td></tr><tr><td>Transfer Direct VLM-Agent 7 additional settings</td><td>1286</td></tr></table>

VisA, and seven additional public anomaly benchmarks, the method provides consistent improvements over fixed label-free fusion strategies while preserving the original detector outputs and avoiding detector retraining. These results suggest that post-training anomaly ensembles can benefit not only from stronger base detectors, but also from explicitly resolving heterogeneous confidence scales, spatial responses, and detector disagreement.

## REFERENCES

[1] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “Winclip: Zero-/few-shot anomaly classification and segmentation,” in CVPR, 2023, pp. 19 606–19 616.

[2] Q. Zhou, G. Pang, Y. Tian, S. He, and J. Chen, “Anomalyclip: Object-agnostic prompt learning for zero-shot anomaly detection,” arXiv preprint arXiv:2310.18961, 2023.

[3] Z. Qu, X. Tao, M. Prasad, F. Shen, Z. Zhang, X. Gong, and G. Ding, “Vcp-clip: A visual context prompting model for zero-shot anomaly segmentation,” in ECCV. Springer, 2024, pp. 301–317.

[4] J. Guo, S. Lu, W. Zhang, F. Chen, H. Li, and H. Liao, “Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection,” in CVPR, 2025, pp. 20 405–20 415.

[5] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler,¨ “Towards total recall in industrial anomaly detection,” in CVPR, 2022, pp. 14 318–14 328.

[6] T. Defard, A. Setkov, A. Loesch, and R. Audigier, “Padim: a patch distribution modeling framework for anomaly detection and localization,” in International conference on pattern recognition. Springer, 2021, pp. 475–489.

[7] H. He, Y. Bai, J. Zhang, Q. He, H. Chen, Z. Gan, C. Wang, X. Li, G. Tian, and L. Xie, “Mambaad: Exploring state space models for multiclass unsupervised anomaly detection,” Advances in Neural Information Processing Systems, vol. 37, pp. 71 162–71 187, 2024.

[8] Q. Zhang, M. Shao, X. Chen, X. Lv, and K. Xu, “Wave-mambaad: Wavelet-driven state space model for multi-class unsupervised anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 20 868–20 877.

[9] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[10] J. Liu, G. Xie, J. Wang, S. Li, C. Wang, F. Zheng, and Y. Jin, “Deep industrial image anomaly detection: A survey,” Machine Intelligence Research, vol. 21, no. 1, pp. 104–135, 2024.

[11] Y. Cui, Z. Liu, and S. Lian, “A survey on unsupervised anomaly detection algorithms for industrial images,” IEEE Access, vol. 11, pp. 55 297–55 315, 2023.

[12] T. G. Dietterich, “Ensemble methods in machine learning,” in International workshop on multiple classifier systems. Springer, 2000, pp. 1–15.

[13] C. C. Aggarwal, “Outlier ensembles,” in Outlier Analysis. Springer, 2016, pp. 185–218.

[14] S. Rayana and L. Akoglu, “Less is more: Building selective anomaly ensembles,” Acm transactions on knowledge discovery from data (tkdd), vol. 10, no. 4, pp. 1–33, 2016.

[15] E. A. Boateng et al., “Unsupervised ensemble methods for anomaly detection in plc-based process control,” arXiv preprint arXiv:2302.02097, 2023.

[16] D. H. Wolpert, “Stacked generalization,” Neural networks, vol. 5, no. 2, pp. 241–259, 1992.

[17] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection,” in CVPR, 2019, pp. 9592–9600.

[18] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, “Spot-thedifference self-supervised pre-training for anomaly detection and segmentation,” in ECCV. Springer, 2022, pp. 392–408.

[19] D. Tabernik, S. Sela, J. Skvar<sup>ˇ</sup> c, and D. Skoˇ caj, “Segmentation-basedˇ deep-learning approach for surface-defect detection,” Journal of Intelligent Manufacturing, vol. 31, no. 3, pp. 759–776, 2020.

[20] C. Wang, W. Zhu, B.-B. Gao, Z. Gan, J. Zhang, Z. Gu, S. Qian, M. Chen, and L. Ma, “Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection,” in CVPR, 2024, pp. 22 883– 22 892.

[21] P. Bergmann, X. Jin, D. Sattlegger, and C. Steger, “The mvtec 3d-ad dataset for unsupervised 3d anomaly detection and localization,” arXiv preprint arXiv:2112.09045, 2021.

[22] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. De Lange, D. Johansen, and H. D. Johansen, “Kvasir-seg: A segmented polyp dataset,” in International conference on multimedia modeling. Springer, 2019, pp. 451–462.

[23] D. Vazquez, J. Bernal, F. J. S´ anchez, G. Fern´ andez-Esparrach, A. M.´ Lopez, A. Romero, M. Drozdzal, and A. Courville, “A benchmark for´ endoluminal scene segmentation of colonoscopy images,” Journal of healthcare engineering, vol. 2017, no. 1, p. 4037190, 2017.

[24] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt, “Fully convolutional cross-scale-flows for image-based defect detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 1088–1097.

[25] J. Lei, X. Hu, Y. Wang, and D. Liu, “Pyramidflow: High-resolution defect contrastive localization using pyramid normalizing flow,” in CVPR, 2023, pp. 14 143–14 152.

[26] Y. Zhou, X. Xu, Z. Sun, J. Song, A. Cichocki, and H. T. Shen, “Vqflow: Taming normalizing flows for multi-class anomaly detection via hierarchical vector quantization,” IEEE Transactions on Multimedia, 2025.

[27] C.-L. Li, K. Sohn, J. Yoon, and T. Pfister, “Cutpaste: Self-supervised learning for anomaly detection and localization,” in CVPR, 2021, pp. 9664–9674.

[28] V. Zavrtanik, M. Kristan, and D. Skocaj, “Draem-a discriminativelyˇ trained reconstruction embedding for surface anomaly detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8330–8339.

[29] Z. Liu, Y. Zhou, Y. Xu, and Z. Wang, “Simplenet: A simple network for image anomaly detection and localization,” in CVPR, 2023, pp. 20 402– 20 411.

[30] C. Ding, G. Pang, and C. Shen, “Catching both gray and black swans: Open-set supervised anomaly detection,” in CVPR, 2022, pp. 7388– 7398.

[31] Z. Yang, T. Zheng, X. Ni, Z. Yan, S. Liu, Y. Yu, H. Chen, Y. Wang, L. Fang, and W. Hao, “Open set industrial surface defect recognition with high frequency feature enhancement and class mutual-information constraint,” IEEE Transactions on Multimedia, 2026.

[32] Y. Bengio, L. Yao, G. Alain, and P. Vincent, “Generalized denoising auto-encoders as generative models,” Advances in neural information processing systems, vol. 26, 2013.

[33] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in Proceedings of the 25th international conference on Machine learning, 2008, pp. 1096–1103.

[34] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings,” in CVPR, 2020, pp. 4183–4192.

[35] H. Deng and X. Li, “Anomaly detection via reverse distillation from one-class embedding,” in CVPR, 2022, pp. 9737–9746.

[36] Z. You, L. Cui, Y. Shen, K. Yang, X. Lu, Y. Zheng, and X. Le, “A unified model for multi-class anomaly detection,” Advances in Neural Information Processing Systems, vol. 35, pp. 4571–4584, 2022.

[37] Z. Fang, X. Wang, H. Li, J. Liu, Q. Hu, and J. Xiao, “Fastrecon: Fewshot industrial anomaly detection via fast feature reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17 481–17 490.

[38] B.-B. Gao, “Learning to detect multi-class anomalies with just one normal image prompt,” in ECCV. Springer, 2024, pp. 454–470.

[39] J. Zhang, C. Wang, X. Li, G. Tian, Z. Xue, Y. Liu, G. Pang, and D. Tao, “Learning feature inversion for multi-class anomaly detection under general-purpose coco-ad benchmark,” arXiv preprint arXiv:2404.10760, 2024.

[40] C. Huang, Q. Xu, Y. Wang, Y. Wang, and Y. Zhang, “Self-supervised masking for unsupervised anomaly detection and localization,” IEEE Transactions on Multimedia, vol. 25, pp. 4426–4438, 2022.

[41] F. Yang, P. Jing, W. Wang, F. L. Wang, and Y. Su, “Padnet: Progressivedifference-aware feature reconstruction mechanism for anomaly detection,” IEEE Transactions on Multimedia, 2025.

[42] H. Zhu, Z. Liu, Z. Sun, W. Dong, X. Xiao, Z. Zhang, and Y. Xu, “Anomaly or characteristic: Memory-based coarse-to-fine feature fusion for industrial anomaly detection,” IEEE Transactions on Multimedia, 2026.

[43] S. Liu, X. Luan, and Y. Li, “Multimodal industrial anomaly detection via attention-enhanced memory-guided network,” IEEE Transactions on Multimedia, 2025.

[44] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[45] Y. Cao, J. Zhang, L. Frittoli, Y. Cheng, W. Shen, and G. Boracchi, “Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection,” in ECCV. Springer, 2024, pp. 55–72.

[46] D. Chen, K. Pan, G. Dai, G. Wang, Y. Zhuang, S. Tang, and M. Xu, “Improving vision anomaly detection with the guidance of language modality,” IEEE Transactions on Multimedia, vol. 27, pp. 1410–1419, 2024.

[47] T. Zhang, W. Pang, and X. Lu, “Tunclip: Adaptive multi-scale zero-shot anomaly detection via vision-language feature infiltration and semantic enhancement,” IEEE Transactions on Multimedia, 2026.

[48] A. Li, C. Qiu, M. Kloft, P. Smyth, M. Rudolph, and S. Mandt, “Zeroshot anomaly detection via batch normalization,” Advances in Neural Information Processing Systems, vol. 36, pp. 40 963–40 993, 2023.

[49] X. Li, Z. Huang, F. Xue, and Y. Zhou, “Musc: Zero-shot industrial anomaly classification and segmentation with mutual scoring of the unlabeled images,” in The Twelfth International Conference on Learning Representations, 2024.

[50] X. Li, Z. Zhang, X. Tan, C. Chen, Y. Qu, Y. Xie, and L. Ma, “Promptad: Learning prompts with only normal samples for few-shot anomaly detection,” in CVPR, 2024, pp. 16 838–16 848.

[51] J. Zhu and G. Pang, “Toward generalist anomaly detection via in-context residual learning with few-shot sample prompts,” in CVPR, 2024, pp. 17 826–17 836.

[52] S. Sun, W. Ren, Z. Liu, H. Park, R. Wang, and X. Cao, “Ensir: An ensemble algorithm for image restoration via gaussian mixture models,” Advances in Neural Information Processing Systems, vol. 37, pp. 133 487–133 517, 2024.

[53] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” arXiv preprint arXiv:2210.03629, 2022.

[54] C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan, “Visual chatgpt: Talking, drawing and editing with visual foundation models,” arXiv preprint arXiv:2303.04671, 2023.

[55] D. Sur´ıs, S. Menon, and C. Vondrick, “Vipergpt: Visual inference via python execution for reasoning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 888–11 898.

[56] Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z. Liu, C. Liu, M. Zeng, and L. Wang, “Mm-react: Prompting chatgpt for multimodal reasoning and action,” arXiv preprint arXiv:2303.11381, 2023.

[57] M. Hu, C. Ma, W. Li, W. Xu, J. Wu, J. Hu, T. Li, G. Zhuang, J. Liu, Y. Lu et al., “A survey of scientific large language models: From data foundations to agent frontiers,” arXiv preprint arXiv:2508.21148, 2025.

[58] J. Liu, J. Pei, J. Huang, C. Si, A. Qu, X. Tang, R. Lu, L. Chen, X. Bai, H. Zheng et al., “The last human-written paper: Agent-native research artifacts,” arXiv preprint arXiv:2604.24658, 2026.

[59] C. Huang, W. Huang, Q. Jiang, W. Wang, J. Wen, and B. Zhang, “Multimodal evidential learning for open-world weakly-supervised video anomaly detection,” IEEE Transactions on Multimedia, vol. 27, pp. 3132–3143, 2025.

[60] J. Zheng, M. Mo, J. Leng, M. Tan, Z. Wu, J. Gan, H. Chen, and X. Gao, “Training-free video anomaly detection via uncertainty-guided hierarchical retrieval with vision-language models,” IEEE Transactions on Multimedia, 2026.

[61] D. Shepard, “A two-dimensional interpolation function for irregularlyspaced data,” in Proceedings ofthe 1968 23rd ACM national conference, 1968, pp. 517–524.

[62] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, and G. L. Foresti, “Vtadl: A vision transformer network for image anomaly detection and localization,” arXiv preprint arXiv:2104.10036, 2021.

[63] M. Wieler, T. Hahn, and F. A. Hamprecht, “Weakly supervised learning for industrial optical inspection,” DAGM 2007 Industrial Defect Dataset, 2007.

[64] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi, “Describing textures in the wild,” in CVPR, 2014, pp. 3606–3613.

[65] Y. Huang, C. Qiu, and K. Yuan, “Surface defect saliency of magnetic tile: Y. huang et al.” The Visual Computer, vol. 36, no. 1, pp. 85–96, 2020.

[66] J. Boziˇ c, D. Tabernik, and D. Skoˇ caj, “Mixed supervision for surface-ˇ defect detection: From weakly to fully supervised learning,” Computers in Industry, vol. 129, p. 103459, 2021.

[67] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, and C. Steger, “Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization,” International Journal of Computer Vision, vol. 130, no. 4, pp. 947–969, 2022.

[68] R. Solovyev, W. Wang, and T. Gabruseva, “Weighted boxes fusion: Ensembling boxes from different object detection models,” Image and Vision Computing, vol. 107, p. 104117, 2021.

[69] A. Neubeck and L. Van Gool, “Efficient non-maximum suppression,” in 18th international conference on pattern recognition (ICPR’06), vol. 3. Ieee, 2006, pp. 850–855.