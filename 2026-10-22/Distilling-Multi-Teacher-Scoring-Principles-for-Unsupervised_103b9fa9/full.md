# Distilling Multi-Teacher Scoring Principles for Unsupervised Time Series Anomaly Detection

Dongchan Cho<sup>∗</sup> Jiho Han Juwon Hwang

Changhee Min Minsang Kim Namsoon Jung

Industrial AI Lab, SimPlatform Co. Ltd. Affiliate Research Institute {cdc0213, jihohan, jwhwang, chmin, mskim, nsj}@simplatform.com

## Abstract

Time series anomalies take diverse forms, and individual detectors encode different anomaly scoring principles that rarely cover all anomaly types. Recent unsupervised time-series anomaly detectors provide complementary scoring signals, yet deploying them as an ensemble preserves the inference and maintenance cost of every individual detector. We study whether heterogeneous teacher detectors can be distilled into a single lightweight detector in the unsupervised setting, where no anomaly labels are available to define a shared target. AnoMix addresses this by extracting weak pairwise supervision from the training-set anomaly scores of frozen teachers. It normalizes each teacher score sequence independently and trains the student with CoRank, a multi-teacher ranking-distillation objective that converts teacher score gaps into pairwise supervision through teacher consensus. Across seven benchmark datasets, AnoMix successfully integrates complementary teacher scoring principles into a single deployable model, improving over individual detectors and static score ensembles while using a compact student with under 80k parameters on the benchmark settings. A zero-shot variant further shows that the distilled ranking signal transfers to unseen datasets and can outperform large pretrained time-series foundation models for anomaly detection.

## 1 Introduction

Time series anomaly detection (TSAD) is a critical problem in industrial monitoring [1], cyberphysical systems [2], and sensor analytics [3–5]. Because anomalies are open-set events and anomaly labels are scarce, TSAD is often addressed in an unsupervised setting. In this setting, the anomaly criterion is ambiguous, and a detector must infer which temporal deviations should be treated as abnormal from the training series alone. Recent benchmark studies [6–8] highlight that this ambiguity leads different detectors to define and detect anomalies under their own criteria, with no single detector dominating across datasets. Together, these observations suggest that improving TSAD performance requires using multiple anomaly-scoring principles.

Existing TSAD detectors already provide a rich source of such complementary scoring principles. Reconstruction-based models emphasize reconstruction failure [3, 9], forecasting models emphasize predictive residuals [10, 11], latent-space models emphasize deviation from compact normal representations [12, 13], and association-based models emphasize structural and temporal dependency mismatch [14, 15]. Because their scores reflect different abnormality criteria, these detectors can complement one another [16]. A direct way to exploit this complementarity is to deploy them together and combine their scores. However, such ensembles preserve the inference cost and maintenance overhead of all individual models [17–19]. A more practical goal is to transfer the complementary scoring principles into one lightweight student.

Knowledge distillation appears to be a natural route toward this goal because teachers can be used during training while only the student is deployed at inference [20, 21]. However, applying distillation to unsupervised TSAD raises a more fundamental question than how to compress a model: what exactly should be distilled? In supervised classification, teacher and student models predict probabilities over the same set of classes, and ground-truth labels anchor the meaning of the teacher signal [21]. In unsupervised TSAD, neither condition holds. Although each teacher outputs anomaly scores, the scores are continuous and detector-specific, and the label-free setting provides no direct way to verify whether they are correct. When multiple teachers are used, this mismatch becomes more pronounced because their numerical magnitudes reflect different abnormality criteria and different calibrations. Therefore, direct score regression is structurally mismatched with this setting, as it asks one student to fit multiple teacher-specific score scales as if they formed a single consistent target.

Our key insight is that imperfect teacher scores can still provide useful weak supervision through pairwise anomaly comparisons rather than through pointwise regression targets. Although two teachers may use different score scales and abnormality criteria, they can still agree that one time point appears more anomalous than another. We call this relation a pairwise anomaly preference. Such preferences provide a more stable supervision form than pointwise regression targets because they compare samples within each teacher score sequence instead of requiring a shared calibrated score scale across teachers. Moreover, when multiple heterogeneous teachers support the same pairwise direction, their agreement suggests that the comparison is informative. Thus, unsupervised TSAD distillation becomes feasible by training a single student to learn reliable pairwise anomaly preferences from multi-teacher score sequences computed on the training data.

Building on this insight, we introduce AnoMix, a label-free multi-teacher distillation framework for learning a neural anomaly-scoring function. Unlike conventional TSAD models that derive anomaly scores from reconstruction, forecasting, association, or representation discrepancies, AnoMix trains a student to directly output the anomaly score used at inference. Since no labeled anomaly-score target exists, AnoMix supervises this scalar score through pairwise anomaly preferences derived from frozen teacher score sequences. First, AnoMix normalizes each frozen teacher score sequence independently and then trains a compact student using CoRank, a consensus-ranking objective over mini-batch time-point pairs. CoRank constructs consensus support from teacher-level score gaps, assigns a consensus direction induced by normalized teacher differences, and weights each comparison by consensus confidence. The student is therefore not asked to reproduce teacher-specific pointwise score magnitudes. Instead, it learns a neural anomaly-scoring function aligned with the teacher-consensu ordering induced by CoRank.

Our contributions are as follows:

• We formulate label-free multi-teacher TSAD distillation as direct anomaly-score prediction for deployment, where a lightweight student learns a neural anomaly-scoring function from heterogeneous unsupervised teacher score sequences.

• We propose AnoMix, which turns normalized teacher score sequences computed on the training data into pairwise anomaly-preference supervision, allowing a single student to learn anomaly-scoring principles without regressing to teacher-specific pointwise scores.

• Across TSAD benchmark datasets and the zero-shot cross-dataset setting, AnoMix and AnoMix-ZS achieve strong performance against individual detectors, static score-ensemble baselines, and pretrained foundation-model baselines, demonstrating that pairwise anomaly preferences from multiple teachers can be distilled into a single lightweight student.

## 2 Related Work

Knowledge Distillation. Knowledge distillation (KD) was originally introduced as a way to compress a large teacher into a compact student by matching softened output distributions [21]. Later KD methods broadened the transferred signal from logits to intermediate features [22], contrastive representations [23], and sample relations such as activation similarities, similarity rankings, and relational structures [24–26]. Multi-teacher KD further aggregates supervision from several teachers into a single student [20]. Together, these lines show that KD need not be limited to output matching, but they still typically assume a well-defined transfer target. For anomaly detection, distillation has been explored primarily in the image domain, often with a single teacher and with teacher–student discrepancy itself serving as the anomaly signal [27, 28]. In time-series anomaly detection, distillation remains relatively underexplored. Recent work such as AnomalyLLM studies single-teacher transfer through feature mimicry and discrepancy-based detection [29]. However, prior work does not address the AnoMix setting of label-free transfer from heterogeneous TSAD teachers to a single student.

![](images/208797fef0ddce4117eb631e498f1495f114cf680f941a5b065adc0266b3e85c.jpg)  
Figure 1: Training overview of AnoMix. (a) Frozen teachers $\big ( \ast \big )$ ) score the unlabeled training series, followed by score normalization. (b) CoRank selects informative pairs and derives their consensus direction and confidence. (c) A trainable student ( ) learns the pairwise ordering through a confidence-weighted loss.

Learning-to-Rank Methods. Learning-to-rank methods learn scoring functions from relative ordering constraints rather than calibrated target values. Existing ranking objectives, however, typically assume that the target ordering is already available. Pairwise methods such as RankNet [30] optimize pairwise preferences or teacher-induced orderings, max-margin ranking losses [31] impose ordered margin constraints, and listwise methods such as ListNet [32] optimize an ordering distribution over a set of samples. More recent differentiable sorting and ranking surrogates [33] further make rank-based objectives compatible with gradient optimization. Relation-based distillation similarly transfers sample-level structure rather than only pointwise outputs [24–26]. These methods primarily specify how to optimize a given target ordering or relation. AnoMix addresses a different setting. In our setting, the target relation is not given by labels or by a single calibrated teacher. AnoMix must construct pairwise supervision from heterogeneous teacher score sequences.

## 3 Methodology

Figure 1 follows a concrete pair of endpoints through AnoMix. In panel (a), each teacher $f _ { m }$ produces raw anomaly scores $\mathbf { \bar { \rho } } _ { z _ { i } } ^ { ( m ) } , z _ { i } ^ { ( m ) }$ aligned with endpoints $i , j$ of the unlabeled training series, with its parameters fixed throughout student training. Per-teacher min–max normalization using training statistics yields the scores $\tilde { z } _ { i } ^ { ( m ) } , \tilde { z } _ { i } ^ { ( m ) }$ in row m of the figure. In panel (b), CoRank retains the pair if any teacher provides a sufficiently large score gap, combines the signed gaps into a consensus direction, and measures their agreement as a confidence weight. The illustrated pair remains informative despite one opposing teacher: its consensus favors i over j with confidence $2 / 3$ In panel (c), the same lightweight student processes both windows and learns this preference through a confidence-weighted pairwise loss. At inference, all teachers are discarded, leaving only the student to produce pointwise anomaly scores. A detailed rationale for CoRank-based multi-teacher ranking distillation is provided in Appendix A.

## 3.1 Problem Setup

Let $\mathbf { X } _ { \mathrm { t r a i n } } = ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { T } ) \in \mathbb { R } ^ { T \times D }$ denote an unlabeled multivariate training time series. For each endpoint $t \geq L ,$ we form a sliding-window sample $\mathbf { W } _ { t } = ( \mathbf { x } _ { t - L + 1 } , \ldots , \mathbf { x } _ { t } ) \overset { \sim } { \in } \mathbb { R } ^ { L \times D }$ . We assume access to M frozen pre-trained teacher detectors $\{ f _ { 1 } , \dotsc , f _ { M } \}$ . Each teacher provides an anomaly score $z _ { t } ^ { ( m ) }$ aligned with endpoint $t ,$ and these teacher scores are normalized independently. The student takes the sliding-window sample $\mathbf { W } _ { t }$ as input and outputs a scalar prediction $\hat { z } _ { t }$ for the same endpoint. No anomaly labels are used, and the normalized teacher scores provide the only supervision signal for training. After training, the teacher detectors are discarded, and the student alone processes each test window to produce a scalar anomaly score. The goal is to learn a single deployable student from unlabeled data and frozen teacher score sequences by capturing pairwise anomaly preferences supported by multiple teachers.

## 3.2 Teacher Score Normalization

Each teacher produces anomaly scores on a detector-specific scale. We therefore min-max normalize each teacher score sequence independently using only training-set statistics. This yields normalized teacher scores $\tilde { z } _ { t } ^ { ( m ) } \in [ 0 , 1 ]$ on the training series. This preserves within-teacher ordering while reducing cross-teacher scale mismatch for distillation. Although teacher score normalization does not make teacher scores a reliable shared regression target, it makes teacher-specific score gaps comparable enough for pairwise supervision because their possible ranges are aligned. AnoMix then computes pairwise teacher score gaps from these normalized scores. These gap magnitudes are used later to select informative pairs and weight pairwise comparisons derived from teacher scores.

## 3.3 Student Architecture

We use a lightweight causal temporal convolutional network (TCN) student with one scalar output. This architecture matches the endpoint-based scoring protocol, where each input window is assigned the anomaly score of its final timestep. Given a window $\mathbf { W } _ { t }$ , a linear projection maps it to ${ \bf H } ^ { ( 0 ) } \in$ $\mathbb { R } ^ { L \times H }$ , N causal temporal blocks [34] produce ${ \bf H } ^ { ( N ) }$ , and the endpoint representation $\mathbf { h } _ { t } = \mathbf { H } _ { L } ^ { ( N ) } \in$ $\mathbb { R } ^ { H }$ is mapped to a scalar score $\hat { z } _ { t } = \mathbf { w } ^ { \top } \mathbf { h } _ { t } + b$ . This lightweight design helps separate the effect of AnoMix’s multi-teacher ranking distillation from gains due to a complex student architecture. Architectural details for both the main student and the zero-shot variant are provided in Appendix B.

## 3.4 CoRank Loss

The main challenge is not choosing a ranking loss, but constructing reliable ranking supervision from heterogeneous teachers. CoRank therefore uses a standard pairwise logistic form, but differs from conventional ranking losses in the situation it addresses. RankNet [30], max-margin ranking [31], listwise ranking [32], and differentiable sorting and ranking objectives [33] can be considered when a target ordering is already available. In AnoMix, no anomaly labels are available, and no single teacher provides a reliable target ordering on its own. A straightforward baseline is to regress student outputs directly to teacher scores. In TSAD, however, direct regression can spend student capacity fitting the score scale of many normal or low-score regions, and detector-specific calibration differences remain even after normalization. AnoMix instead learns from pairwise score gaps whose directions are supported by multiple teachers.

Consider a mini-batch of B windows. Let $\tilde { z } _ { i } ^ { ( m ) }$ denote the normalized score assigned by teacher m to sample i, and let $\hat { z } _ { i }$ denote the corresponding student prediction. We denote teacher $m \mathrm { { ^ { * } s } }$ signed score gap by $\Delta _ { i j } ^ { ( m ) } = \tilde { z } _ { i } ^ { ( m ) } - \tilde { z } _ { i } ^ { ( m ) }$ . From the batch, we enumerate all unordered non-identical pairs $( i , j )$ with $i < j$ and retain only those pairs for which at least one teacher exhibits a meaningful score gap:

$$
\mathcal { P } = \left\{ ( i , j ) : 1 \leq i < j \leq B , \operatorname* { m a x } _ { m } \left| \Delta _ { i j } ^ { ( m ) } \right| > \delta \right\} .\tag{1}
$$

The pair selection step in panel (b) of Figure 1 removes pairs whose teacher score gaps are too small to provide meaningful anomaly-preference supervision. The resulting set $\mathcal { P }$ defines the consensus support and contains the meaningful teacher-induced comparisons used for consensus supervision before teacher scores are averaged.

Once a pair $( i , j )$ is retained in ${ \mathcal { P } } _ { : }$ , every teacher contributes its signed score gap $\Delta _ { i i } ^ { ( m ) }$ . Using all teacher score gaps for each retained pair allows the final supervision to reflect the full degree of cross-teacher agreement or disagreement on that pair, rather than only the strongest response. We aggregate these signed gaps to form the teacher consensus

$$
v _ { i j } = \sum _ { m = 1 } ^ { M } \Delta _ { i j } ^ { ( m ) } , \qquad c _ { i j } = \frac { \left| v _ { i j } \right| } { \sum _ { m = 1 } ^ { M } \left| \Delta _ { i j } ^ { ( m ) } \right| } .\tag{2}
$$

Here, $y _ { i j } = \mathrm { s g n } ( v _ { i j } ) \in \{ - 1 , 0 , 1 \}$ defines the consensus direction, where sgn(·) is the sign function: $y _ { i j } = 1$ if v<sub>ij</sub> > 0, y<sub>ij</sub> = −1 if $v _ { i j } < 0$ , and $y _ { i j } = 0 \mathrm { i f } v _ { i j } = 0$ . The quantity $\dot { c } _ { i j } \in [ 0 , 1 \bar { ] }$ quantifies how consistently teachers support that direction, as visualized by the confidence bar in Figure 1. When all nonzero signed gaps point in the same direction, $c _ { i j } = 1$ . When their signed score gaps strongly conflict, $c _ { i j }$ approaches 0. A high confidence weight indicates agreement among teacher score gaps, not correctness of the resulting preference. If $v _ { i j } = 0$ , then $y _ { i j } = 0 , c _ { i j } = 0$ , and the pair contributes no effective supervision.

The student’s pairwise anomaly preference is represented by the score margin $d _ { i j } = \hat { z } _ { i } - \hat { z } _ { j }$ , the signed difference between student predictions for samples i and j. The CoRank loss in panel (c) of Figure 1 penalizes student margins $d _ { i j }$ that disagree with the teacher consensus direction $y _ { i j }$ , with each pair weighted by its consensus confidence $c _ { i j }$ :

$$
\mathcal { L } _ { \mathrm { C o R a n k } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } c _ { i j } \log \Bigl ( 1 + \exp \bigl ( - y _ { i j } d _ { i j } \bigr ) \Bigr ) .\tag{3}
$$

If no valid pair is retained in a mini-batch, we define the batch loss as zero and skip its contribution.

CoRank can be viewed as confidence-weighted distillation of pairwise anomaly preferences induced by normalized teacher scores. It avoids fitting teacher-specific pointwise scores and does not construct pairwise supervision from an averaged teacher score alone. Instead, it first constructs the pair support from teacher-level score gaps, assigns the consensus direction, and then uses consensus confidence to down-weight conflicting comparisons. CoRank contributes a multi-teacher procedure for constructing and weighting pairwise supervision before applying the pairwise logistic objective. This trains the student network to act as a neural anomaly-scoring function whose scalar outputs reflect pairwise anomaly preferences supported by teacher consensus.

## 3.5 Training and Inference

Let $\mathcal { W } _ { \mathrm { t r a i n } } = \{ \mathbf { W } _ { t } : t = L , \dots , T \}$ denote the set of training windows. During training, each mini-batch $B \subset \mathcal { W } _ { \mathrm { t r a i n } }$ is paired with the normalized teacher scores at the corresponding endpoints, and AnoMix optimizes

$$
\theta ^ { \star } = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } _ { \mathcal { B } \sim \mathcal { W } _ { \mathrm { t r a i n } } } \left[ \mathcal { L } _ { \mathrm { C o R a n k } } \left( g _ { \theta } ; \mathcal { B } , \{ \tilde { z } ^ { ( m ) } \} _ { m = 1 } ^ { M } \right) \right] .\tag{4}
$$

At inference time, all teachers are discarded. For each test window, the anomaly score is simply

$$
s _ { t } = g _ { \theta ^ { \star } } ( \mathbf { W } _ { t } ) , \qquad t = L , \dots , T _ { \mathrm { t e s t } } .\tag{5}
$$

Thus, pairwise training still yields a standard pointwise anomaly score at inference time.

## 4 Experiments

## 4.1 Setup

Datasets. We evaluate on seven real-world multivariate TSAD benchmark datasets, including PSM [1], SWaT [2], Exathlon [35], SMD [3], SMAP [4], Daphnet [5], and SWAN [36]. Dataset descriptions and statistics are provided in Appendix F.

Metrics. We report six threshold-independent metrics: VUS-ROC and VUS-PR [37], AUC-ROC [38], AUC-PR [39], and R-AUC-ROC and R-AUC-PR [40]. The main experiments focus on VUS-ROC (V-ROC) and VUS-PR (V-PR). Formal definitions of the reported metrics are provided in Appendix G.

Baselines. Table 1 compares AnoMix against ten representative individual TSAD detectors. These baselines include OmniAnomaly [3], USAD [9], Anomaly Transformer [15], TranAD [10], Times-Net [11], SARAD [41], MOMENT [42], CATCH [43], CrossAD [14], and OracleAD [12], covering the main TSAD paradigms considered in recent benchmarks. Table 2 compares AnoMix against five score-ensemble baselines built from the same three teachers and the same normalized teacher scores. We consider Mean, Max, Median, Rank Fusion, and Reciprocal Rank Fusion (RRF) [44]. For the zero-shot transfer setting in Table 4, we compare AnoMix-ZS against six pretrained time-series foun dation models: MOMENT [42], TimesFM [45], Chronos [46], MOIRAI [47], Lag-Llama [48], and

Table 1: Main results on the seven benchmark datasets (5-seed mean, %). Avg Rank is the per-row average rank across the 14 dataset-metric values, where lower is better. Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td><td rowspan="2">Avg Rank</td></tr><tr><td>|V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>|V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td></tr><tr><td>Omni [3]</td><td>51.59</td><td>22.52</td><td>59.42</td><td>27.32</td><td>99.98</td><td>99.75</td><td>95.40</td><td>69.98</td><td>49.93</td><td>27.68</td><td>78.60</td><td>18.38</td><td>92.06</td><td>91.16</td><td>6.18</td></tr><tr><td>USAD [9]</td><td>52.15</td><td>22.73</td><td>68.92</td><td>33.39</td><td>99.97</td><td>99.63</td><td>95.98</td><td>71.97</td><td>49.80</td><td>27.58</td><td>76.95</td><td>17.15</td><td>89.80</td><td>88.60</td><td>6.36</td></tr><tr><td>A.T [15]</td><td>53.76</td><td>23.13</td><td>56.63</td><td>23.55</td><td>37.70</td><td>24.88</td><td>44.03</td><td>11.80</td><td>40.56</td><td>22.30</td><td>46.28</td><td>10.33</td><td>86.33</td><td>81.07</td><td>10.29</td></tr><tr><td>TranAD [10]</td><td>55.90</td><td>22.25</td><td>77.06</td><td>37.86</td><td>99.50</td><td>93.88</td><td>86.51</td><td>52.05</td><td>71.96</td><td>37.06</td><td>76.71</td><td>17.98</td><td>92.02</td><td>86.48</td><td>6.93</td></tr><tr><td>TimesNet [11]</td><td>57.91</td><td>24.32</td><td>59.29</td><td>24.35</td><td>98.61</td><td>85.33</td><td>93.96</td><td>53.06</td><td>59.22</td><td>23.90</td><td>90.25</td><td>33.97</td><td>93.87</td><td>88.77</td><td>6.71</td></tr><tr><td>SARAD [41] MOMENT [42]</td><td>58.10 42.39</td><td>26.78 16.59</td><td>68.29</td><td>29.82</td><td>99.86</td><td>98.27</td><td>95.40</td><td>62.65</td><td>75.27</td><td>37.33</td><td>89.96</td><td>38.72</td><td>92.11</td><td>90.20</td><td>4.25</td></tr><tr><td></td><td></td><td></td><td>54.95</td><td>20.21</td><td>99.66</td><td>97.17</td><td>57.27</td><td>10.19</td><td>54.72</td><td>27.85</td><td>91.71</td><td>35.10</td><td>92.07</td><td>86.25</td><td>8.18</td></tr><tr><td>CATCH [43]</td><td>53.99</td><td>23.42</td><td>58.82</td><td>24.61</td><td>99.83</td><td>97.79</td><td>96.45</td><td>67.33</td><td>49.69</td><td>27.45</td><td>91.37</td><td>38.86</td><td>92.16</td><td>89.66</td><td>5.86</td></tr><tr><td>CrossAD [14]</td><td>59.02</td><td>27.66</td><td>60.79</td><td>26.33</td><td>99.41</td><td>92.47</td><td>96.49</td><td>72.52</td><td>52.95</td><td>35.62</td><td>89.77</td><td>36.31</td><td>92.91</td><td>91.66</td><td>4.79</td></tr><tr><td>OracleAD [12]</td><td>66.33</td><td>28.68</td><td>59.87</td><td>26.44</td><td>99.66</td><td>97.18</td><td>85.71</td><td>52.69</td><td>79.88</td><td>40.18</td><td>88.46</td><td>34.60</td><td>92.44</td><td>86.30</td><td>5.39</td></tr><tr><td>AnoMix</td><td>70.24</td><td>44.03</td><td>78.56</td><td>42.40</td><td>100.00</td><td>99.97</td><td>98.87</td><td>89.37</td><td>98.38</td><td>97.65</td><td>94.13</td><td>52.37</td><td>93.75</td><td>91.76|</td><td>1.07</td></tr></table>

Table 2: Score-ensemble comparison on the seven benchmark datasets (5-seed mean, %). Avg Rank is the per-row average rank across the 14 dataset-metric values, where lower is better. Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td><td rowspan="2">Avg Rank</td></tr><tr><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td></tr><tr><td>Mean</td><td>67.00</td><td>31.23</td><td>63.80</td><td>27.07</td><td>99.65</td><td>95.82</td><td>97.55</td><td>74.23</td><td>53.50</td><td>34.87</td><td>91.64</td><td>38.10</td><td>93.96</td><td>91.54</td><td>4.00</td></tr><tr><td>Max</td><td>66.81</td><td>31.20</td><td>59.77</td><td>25.78</td><td>99.45</td><td>93.49</td><td>96.46</td><td>68.26</td><td>53.02</td><td>35.59</td><td>89.35</td><td>31.59</td><td>93.98</td><td>91.30</td><td>5.21</td></tr><tr><td>Median</td><td>61.93</td><td>27.28</td><td>65.66</td><td>28.21</td><td>99.83</td><td>97.78</td><td>96.50</td><td>72.13</td><td>66.88</td><td>32.59</td><td>90.24</td><td>38.85</td><td>92.81</td><td>89.47</td><td>4.57</td></tr><tr><td>Rank Fusion</td><td>64.90</td><td>27.81</td><td>65.08</td><td>28.57</td><td>99.97</td><td>99.74</td><td>97.72</td><td>83.70</td><td>70.26</td><td>37.86</td><td>92.71</td><td>46.11</td><td>93.88</td><td>91.69</td><td>2.93</td></tr><tr><td>RRF [44]</td><td>67.37</td><td>31.69</td><td>65.28</td><td>28.55</td><td>99.47</td><td>94.59</td><td>97.73</td><td>79.13</td><td>71.84</td><td>39.62</td><td>91.32</td><td>36.18</td><td>94.24</td><td>91.59</td><td>3.00</td></tr><tr><td>AnoMix</td><td>70.24</td><td>44.03</td><td>78.56</td><td>42.40</td><td>100.00</td><td>99.97</td><td>98.87</td><td>89.37</td><td>98.38</td><td>97.65</td><td>94.13</td><td>52.37</td><td>93.75</td><td>91.76</td><td>1.29</td></tr></table>

DADA [49]. For most deep TSAD and zero-shot baselines, we use implementations from the TSB-AD repository [6]. Baselines not available in TSB-AD are evaluated using their official repositories. Baseline descriptions and the training and evaluation protocol are provided in Appendix H.

Teacher Set. By default, AnoMix uses a fixed three-teacher setting consisting of SARAD, CrossAD, and OracleAD. This teacher set is formed by taking the three most recent state-of-the-art TSAD detectors in our teacher pool, which provides heterogeneous anomaly supervision while keeping the training-time teacher pool compact. For analysis convenience and to avoid post-hoc teacher selection, this set was fixed before evaluating AnoMix on the benchmark datasets. For each teacher, one fixed checkpoint supplies cached training scores shared across the five student seeds; Table 1 separately reports each teacher’s five-seed mean performance. Alternative teacher-set choices, including leaveone-out (LOO) rankings and All-10, are analyzed in Section 4.4. The LOO selection protocol and the full teacher ranking for each target dataset are detailed in Appendix I.

Implementation Details. Retrained models use five random seeds under fixed splits; the zero-shot foundation baselines use fixed released checkpoints. AnoMix uses a single-head TCN student with hidden dimension 64, six causal blocks, kernel size 3, dropout 0.1, and fewer than 80k parameters on the benchmark datasets. The student receptive field is 127 time steps, and most datasets use window size 127 to fully use this temporal context. We train the student for 10 epochs using AdamW with cosine annealing and a score-gap threshold δ = 0.1 for valid-pair selection. Student training uses only the provided training split and no anomaly labels. Full implementation details are provided in Appendix J.

## 4.2 Main Results

Benchmark Results. Table 1 shows that AnoMix achieves the strongest overall average rank across the benchmark. More importantly, strong baselines are not stable across datasets. The second-best results are distributed across reconstruction, forecasting, association-aware, deviation-based, and foundation-model baselines, indicating that each detector family captures only part of the anomaly structure present in the benchmark. AnoMix turns this dataset-dependent complementarity into a single student, rather than committing to one detector family at deployment.

![](images/d62cce7ece7dc6fb6d85128e1365f6578645734acf831555e4dadf66abd4abda.jpg)

![](images/1ea13c1862cde78f183bd334b9b6ed2dc400ead130ef09109c6e2b4f4240c8ab.jpg)

![](images/e3b72ad0d1c17eb3bad46178f795e48395708215c91504342d276d8e0db4d1cb.jpg)

![](images/2b44486e0cbbc0fe9911c9d04c953123359e085f980855cbd0a7bd48390acf8b.jpg)  
Figure 2: Ablation study reporting macro-average V-PR (%) over the seven benchmark datasets. (a) Student backbones: LSTM, Transformer, Dilated-RNN, and TCN. (b) Ranking losses: RankNet, Hinge, ListNet, and Spearman on an averaged teacher-score target, and CoRank. (c) Supervision forms: score regression on the averaged teacher-score target, MeanRank, and CoRank. (d) CoRank decomposition: MeanRank, consensus confidence, consensus support, and full CoRank.

The magnitude of improvement is also non-uniform across datasets and metrics. AnoMix yields its largest gains on PSM, SMD, and especially SMAP, where the best individual detectors still leave substantial room for improvement, whereas the gains on Exathlon are small because several methods already achieve near-ceiling performance. Across datasets, the improvement over the strongest individual baseline is especially pronounced on V-PR and smaller on V-ROC. This suggests that AnoMix does not merely improve coarse ranking quality, but more sharply concentrates high anomaly scores on true anomalous regions. Full PR and ROC metric results are reported in Appendix E.

Comparison with Score-Ensemble Baselines. Table 2 tests whether AnoMix’s gains can be explained by static score aggregation alone. The comparison fixes the teacher pool and changes only how teacher information is combined at inference time. Among the fusion rules, RRF is the strongest overall, but AnoMix achieves a substantially better average rank. The gaps are especially large on PSM, SWaT, SMD, and SMAP. If the gains came merely from merging teacher outputs at test time, stronger fusion rules such as rank fusion or RRF would be expected to close most of the gap. Instead, the gap persists even against rank-based fusion, indicating that AnoMix benefits from learning an input-dependent student scoring function rather than from static score aggregation alone. Teacher scores are used as cross-teacher comparative supervision during training, allowing the student to learn anomaly preferences across teachers while deploying as a single model.

## 4.3 Ablation Studies

Backbone Choice. To assess backbone dependence, Figure 2(a) compares four student backbones under the same CoRank objective. Each backbone is used within the same endpoint-scoring template, producing an endpoint representation that is mapped to a scalar anomaly score. The LSTM, Transformer, and Dilated-RNN variants use their best learning rate on each dataset. TCN achieves the highest macro V-PR (73.93%), outperforming Transformer (62.51%), Dilated-RNN (58.90%), and LSTM (55.63%) by 11–18%pt. With a comparable number of parameters, the result supports choosing a TCN as an efficient lightweight student for distilling pairwise anomaly preferences, without introducing a new architecture design.

Training Objective. To test whether standard ranking losses suffice, Figure 2(b) compares CoRank with four such losses using the same student. The baseline losses rank pairs according to the averaged normalized teacher score, but they do not use CoRank’s consensus support or consensus confidence. CoRank achieves the highest macro V-PR (73.93%), exceeding RankNet [30] (68.01%), Hinge [31] (67.75%), ListNet [32] (64.60%), and Spearman [33] (68.27%). Because the student architecture is held fixed, the 5–10%pt gap is not attributable to model capacity or to the use of ranking supervision alone. It reflects the difference between applying standard ranking surrogates to a single fused target and constructing the pairwise training signal from teacher-level score gaps. In this setting, CoRank’s advantage comes from how pair support and confidence are defined before the logistic ranking surrogate is applied.

Score Supervision Form. To assess the benefit of pairwise supervision, Figure 2(c) contrasts direct score regression, fused-target ranking, and CoRank using the same single-head student. In the scoreregression baseline, the student directly fits the averaged normalized teacher score using a pointwise MSE loss. MeanRank denotes the fused-target ranking baseline that first averages normalized teacher scores and then forms ranking pairs from gaps in the averaged score. This baseline reaches 67.97%,

Table 3: Teacher-set variants on the seven benchmark datasets (5-seed mean, %; standard deviations in Appendix O). LOO-k uses the top-k teachers from the target-excluded teacher ranking, and Sel-3 is the fixed three-teacher setting used in Table 1. Avg Rank is computed over the 14 reported values, where lower is better. Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td><td rowspan="2">Avg Rank</td></tr><tr><td>V-ROC</td><td>V-PR</td><td>|V-ROC</td><td>V-PR</td><td>|V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>|V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>|V-ROC</td><td>V-PR</td></tr><tr><td>L00-1</td><td>48.10</td><td>22.32</td><td>43.48</td><td>25.09</td><td>80.05</td><td>81.58</td><td>59.08</td><td>48.49</td><td>56.59</td><td>56.58</td><td>75.65</td><td>30.98</td><td>90.81</td><td>84.37</td><td>7.00</td></tr><tr><td>L00-2</td><td>69.91</td><td>42.22</td><td>73.28</td><td>38.72</td><td>99.99</td><td>99.96</td><td>98.92</td><td>89.48</td><td>98.37</td><td>97.65</td><td>94.01</td><td>51.50</td><td>92.45</td><td>90.77</td><td>4.61</td></tr><tr><td>L00-3</td><td>69.04</td><td>42.64</td><td>76.82</td><td>43.37</td><td>99.99</td><td>99.96</td><td>98.87</td><td>89.37</td><td>98.40</td><td>97.72</td><td>94.12</td><td>51.96</td><td>93.75</td><td>91.76</td><td>3.61</td></tr><tr><td>L00-4</td><td>67.55</td><td>35.17</td><td>80.98</td><td>44.59</td><td>99.99</td><td>99.96</td><td>98.90</td><td>89.37</td><td>98.41</td><td>97.75</td><td>94.10</td><td>51.94</td><td>93.60</td><td>91.52</td><td>3.68</td></tr><tr><td>L00-5</td><td>67.64</td><td>34.40</td><td>75.49</td><td>41.28</td><td>99.99</td><td>99.96</td><td>98.90</td><td>89.39</td><td>98.42</td><td>97.77</td><td>94.15</td><td>52.25</td><td>93.43</td><td>91.28</td><td>3.68</td></tr><tr><td>Sel-3</td><td>70.24</td><td>44.03</td><td>78.56</td><td>42.40</td><td>100.00</td><td>99.97</td><td>98.87</td><td>89.37</td><td>98.38</td><td>97.65</td><td>94.13</td><td>52.37</td><td>93.75</td><td>91.76</td><td>2.86</td></tr><tr><td>All-10</td><td>64.45</td><td>31.19</td><td>80.47</td><td>45.23</td><td>100.00</td><td>99.97</td><td>98.93</td><td>89.41</td><td>98.43</td><td>97.78</td><td>94.17</td><td>52.34</td><td>93.23</td><td>91.06</td><td>2.57</td></tr></table>

![](images/a26e58e2be52dce61607189c94b4002fb61f83d83b5e5a79519a2522fe02d96f.jpg)  
Figure 3: Mean absolute Spearman correlation with teacher score sequences $( | \rho _ { S } | )$ versus V-PR on five non-saturated datasets. A higher $| \rho _ { S } |$ indicates stronger alignment with teacher score sequences, while a lower $| \rho _ { S } |$ indicates weaker alignment. The blue star denotes AnoMix, and the remaining markers denote the score-ensemble baselines (Mean, Max, Median, Rank Fusion, RRF).

MeanRank reaches 69.46%, and CoRank reaches 73.93%. These results support two conclusions. First, relative supervision is more effective than direct pointwise score matching. Second, ranking the averaged teacher score is not sufficient to recover the full benefit of multi-teacher distillation. This comparison shows that the benefit comes not only from ranking, but also from constructing pairwise supervision from teacher-level score gaps before averaging. Figure 2(d) further decomposes this teacher-level construction. Together with Figure 2(b), these results suggest that AnoMix benefits specifically from consensus-based multi-teacher distillation rather than from ranking alone.

CoRank Component Decomposition. To identify which components drive CoRank’s gains, Figure 2(d) separates the effects of consensus support and consensus confidence while using the same pairwise logistic loss across all variants. MeanRank reaches 69.46% macro V-PR, and adding consensus confidence under the same support from averaged-score gaps changes this only slightly to 69.48%. Using CoRank’s consensus support increases macro V-PR to 73.54%, and adding consensus confidence on top yields full CoRank at 73.93%. This indicates that CoRank’s main gain comes from constructing the pairwise training set from teacher-level score gaps before averaging. Averaging teachers before support construction can weaken comparisons where one detector gives a clear score gap but other detectors give small or neutral gaps, whereas consensus support preserves these comparisons for the student. Consensus confidence then provides a smaller additional gain by reducing the influence of retained pairs with stronger teacher conflict. Additional ablation studies are provided in Appendix M.

## 4.4 Distillation Analysis

Teacher-Set Variants. We test AnoMix’s sensitivity to teacher identity and pool size. For each target dataset, LOO-k selects the top k teachers by average V-PR on the other six datasets; Sel-3 is the fixed three-teacher set chosen before evaluation, and All-10 uses the full teacher pool. Table 3 shows a clear separation between single-teacher and multi-teacher settings. LOO-1 is substantially weaker than all evaluated multi-teacher variants, suggesting that the teacher signals are complementary enough for AnoMix to benefit from combining them rather than relying on a single teacher. This is consistent with the role of CoRank, which aggregates teacher-supported score gaps instead of matching any single teacher score directly.

The same table also shows that AnoMix is not highly sensitive to teacher-set selection. All-10 achieves the best overall average rank, while the fixed three-teacher benchmark setting (Sel-3) remains close to the stronger LOO variants. However, All-10 also degrades PSM substantially, suggesting that broader teacher pools can add complementary training signal while also introducing harmful or redundant comparisons. The similar performance of LOO-3, LOO-4, LOO-5, and Sel-3 suggests that a compact heterogeneous pool captures most of the gain, with LOO selection offering a practical alternative. Appendix I further tests mixed-quality, weak, and correlated teacher pools.

Table 4: Zero-shot transfer results (%). AnoMix-ZS reports five-seed means, with standard deviations in Appendix O; foundation baselines use fixed released checkpoints. AnoMix-ZS is trained on the other six datasets and evaluated on the held-out target. Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td><td rowspan="2"> $\mathbf { \mathbf { \sigma } } _ { \mathbf { A v g \ R a n k } }$ </td></tr><tr><td>|V-ROC V-PR|</td><td></td><td>V-ROC</td><td>V-PR|</td><td></td><td></td><td>|V-ROC V-PR|V-ROC V-PR|V-ROC V-PR|</td><td></td><td></td><td></td><td>|V-ROC V-PR|</td><td></td><td>|V-ROC V-PR|</td><td></td></tr><tr><td>MOMENT [42]</td><td>110M</td><td>41.87</td><td>16.66</td><td>54.62</td><td>20.17</td><td>99.66</td><td>97.18</td><td>58.22</td><td>10.38</td><td>56.03</td><td>28.31</td><td>91.49</td><td>34.49</td><td>92.06</td><td>86.29</td><td>5.57</td></tr><tr><td>TimesFM [45]</td><td>200M</td><td>53.99</td><td>20.14</td><td>49.45</td><td>19.22</td><td>99.19</td><td>96.64</td><td>59.04</td><td>16.19</td><td>94.65</td><td>88.93</td><td>89.97</td><td>34.79</td><td>93.79</td><td>87.66</td><td>4.50</td></tr><tr><td>Chronos [46]</td><td>710M</td><td>52.46</td><td>18.68</td><td>63.11</td><td>23.33</td><td>87.54</td><td>61.15</td><td>89.06</td><td>34.67</td><td>70.54</td><td>30.97</td><td>90.53</td><td>33.50</td><td>93.02</td><td>88.11</td><td>4.71</td></tr><tr><td>MOIRAI [47]</td><td>14M</td><td>54.02</td><td>19.47</td><td>55.17</td><td>20.49</td><td>91.52</td><td>66.86</td><td>60.99</td><td>10.37</td><td>57.98</td><td>21.12</td><td>91.07</td><td>34.11</td><td>92.64</td><td>83.72</td><td>5.43</td></tr><tr><td>Lag-Llama [48]</td><td>10M</td><td>54.63</td><td>20.45</td><td>47.96</td><td>18.56</td><td>89.08</td><td>58.11</td><td>83.10</td><td>33.47</td><td>80.29</td><td>39.28</td><td>92.04</td><td>38.24</td><td>93.11</td><td>87.86</td><td>4.43</td></tr><tr><td>DADA [49]</td><td>1.84M</td><td>57.80</td><td>23.08</td><td>60.82</td><td>25.40</td><td>99.80</td><td>98.72</td><td>98.62</td><td>87.90</td><td>98.67</td><td>95.61</td><td>92.53</td><td>40.29</td><td>93.57</td><td>88.84</td><td>1.71</td></tr><tr><td>AnoMix-ZS</td><td>83k</td><td>58.71</td><td>25.44</td><td>78.97</td><td>39.85</td><td>99.88</td><td>98.19</td><td>98.45</td><td>86.12</td><td>98.19</td><td>93.88</td><td>93.53</td><td>47.73</td><td>92.91</td><td>90.28</td><td>1.64</td></tr></table>

Score Alignment. Figure 3 tests whether AnoMix is merely approximating a static ensemble. Across the non-saturated datasets, AnoMix attains higher V-PR than all score-ensemble baselines while exhibiting lower average Spearman correlation with the teacher score sequences. If the student were simply learning a smoothed version of one teacher or of a fused teacher score, stronger performance would be expected to come with stronger teacher-score alignment. Instead, the opposite pattern is observed: AnoMix is more accurate while being less correlated with teacher outputs. This indicates that the student is not passively inheriting teacher scores, but integrating teacher information into a more effective neural anomaly-scoring function. Taken together with Table 3, this supports the intended role of CoRank as a distillation objective rather than a surrogate ensemble rule. Score visualizations and teacher-alignment diagnostics are provided in Appendices D and N.

## 4.5 Zero-Shot Cross-Dataset Detection

AnoMix’s training procedure suggests a natural zero-shot extension because the student can be distilled from teacher-derived ranking signals on source datasets and then applied to an unseen target dataset. We evaluate this extension under a stricter cross-dataset setting that removes target-dataset supervision entirely. For this evaluation, we use AnoMix-ZS, a compact channel-independent student that can operate on varying input dimensionality. The architecture and protocol are detailed in Appendix B. AnoMix-ZS is trained on six datasets with the same ranking-based distillation objective and then evaluated on the held-out target, without observing the target dataset or teacher scores during training. The six foundation baselines are converted into anomaly scores following their evaluation wrappers: MOMENT uses reconstruction error, TimesFM, Chronos, MOIRAI, and Lag-Llama use one-step forecasting error, and DADA uses mask-reconstruction variance.

Table 4 shows that ranking-based distillation transfers beyond the training benchmarks and remains highly competitive even in the strict zero-shot setting. Among the evaluated models, AnoMix-ZS achieves the best average rank (1.64) with 83k parameters, whereas the pretrained foundation baselines range from 1.84M to 710M parameters. This comparison is informative for two reasons. First, AnoMix-ZS is not merely memorizing source-dataset behavior, but learning anomaly-ranking information that generalizes to unseen datasets. Second, the comparison shows that cross-dataset TSAD-oriented distillation can match a much larger TSAD-specific foundation model (83k vs 1.84M parameters for DADA), and substantially outperforms general-purpose time-series foundation models that range from 10M to 710M parameters. Taken together, these results suggest that TSAD-oriented cross-dataset distillation, as in AnoMix, offers a promising direction for building TSAD-specific foundation models rather than relying solely on general-purpose time-series foundation models.

## 5 Conclusion

We studied whether unsupervised time series anomaly detection can be effectively addressed through knowledge distillation despite the absence of labels and the heterogeneity of teacher anomaly scores.

Our experimental results suggest that it can. We introduced AnoMix, a multi-teacher distillation framework that learns a single lightweight student by transferring pairwise anomaly preferences from multiple heterogeneous TSAD teachers, rather than directly imitating raw teacher scores.

Across benchmarks, the empirical evidence consistently shows that the benefit of multi-teacher TSAD comes from distilling complementary teacher scoring principles into a new student predictor, not merely from combining teacher scores. AnoMix outperforms individual detectors and score-ensemble baselines, while teacher-set and correlation analyses further indicate that it learns beyond static ensemble behavior and remains effective under broader teacher pools. The zero-shot cross-dataset results further show that AnoMix-ZS remains effective on unseen datasets and can compete with much larger pretrained time-series foundation models.

Although teachers use complex, detector-specific models, their scoring principles are ultimately expressed through scalar anomaly scores. Our results suggest that learning pairwise preferences from these scores allows a compact student to integrate complementary scoring principles without reproducing the teachers’ internal models. A more detailed discussion of limitations and future directions is provided in Appendix Q.

## References

[1] Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. Practical approach to asynchronous multivariate time series anomaly detection and localization. In Proceedings ofthe 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2021.

[2] Aditya P. Mathur and Nils Ole Tippenhauer. SWaT: a water treatment testbed for research and training on ICS security. In International Workshop on Cyber-physical Systems for Smart Water Networks, 2016.

[3] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019.

[4] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.

[5] Marc Bächlin, Meir Plotnik, Daniel Roggen, Inbal Maidan, Jeffrey M. Hausdorff, Nir Giladi, and Gerhard Tröster. Wearable Assistant for Parkinson’s Disease Patients With the Freezing of Gait Symptom. IEEE Transactions on Information Technology in Biomedicine, 2010.

[6] Qinghua Liu and John Paparrizos. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. In Advances in Neural Information Processing Systems, 2024.

[7] Xiangfei Qiu, Zhe Li, Wanghui Qiu, Shiyan Hu, Lekui Zhou, Xingjian Wu, Zhengyu Li, Chenjuan Guo, Aoying Zhou, Zhenli Sheng, Jilin Hu, Christian S. Jensen, and Bin Yang. TAB: Unified benchmarking of time series anomaly detection methods. Proceedings of the VLDB Endowment, 2025.

[8] M. Saquib Sarfraz, Mei-Yen Chen, Lukas Layer, Kunyu Peng, and Marios Koulakis. Position: Quo vadis, unsupervised time series anomaly detection? In Proceedings of the 41st International Conference on Machine Learning, 2024.

[9] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A. Zuluaga. USAD: UnSupervised Anomaly Detection on multivariate time series. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020.

[10] Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings. TranAD: deep transformer networks for anomaly detection in multivariate time series data. Proceedings of the VLDB Endowment, 2022.

[11] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023.

[12] Dongchan Cho, Jiho Han, Keumyeong Kang, Minsang Kim, Honggyu Ryu, and Namsoon Jung. Structured temporal causality for interpretable multivariate time series anomaly detection. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[13] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. Deep one-class classification. In Proceedings ofthe International Conference on Machine Learning, 2018.

[14] Beibu Li, Qichao Shentu, Yang Shu, Hui Zhang, Ming Li, Ning Jin, Bin Yang, and Chenjuan Guo. CrossAD: Time series anomaly detection with cross-scale associations and cross-window modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[15] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly Transformer: Time series anomaly detection with association discrepancy. In International Conference on Learning Representations, 2022.

[16] Charu C. Aggarwal and Saket Sathe. Theoretical foundations and algorithms for outlier ensembles. ACM SIGKDD Explorations Newsletter, 2015.

[17] Yue Zhao, Zain Nasrullah, Maciej K. Hryniewicki, and Zheng Li. LSCP: Locally selective combination in parallel outlier ensembles. In Proceedings ofthe SIAM International Conference on Data Mining, 2019.

[18] Yue Zhao, Xiyang Hu, Cheng Cheng, Cong Wang, Changlin Wan, Wen Wang, Jianing Yang, Haoping Bai, Zheng Li, Cao Xiao, Yunlong Wang, Zhi Qiao, Jimeng Sun, and Leman Akoglu. SUOD: Accelerating large-scale unsupervised heterogeneous outlier detection. In Proceedings of Machine Learning and Systems, 2021.

[19] Yue Zhao, Ryan Rossi, and Leman Akoglu. Automatic unsupervised outlier model selection. In Advances in Neural Information Processing Systems, 2021.

[20] Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal ofComputer Vision, 2021.

[21] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[22] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. In International Conference on Learning Representations, 2015.

[23] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. In International Conference on Learning Representations, 2020.

[24] Frederick Tung and Greg Mori. Similarity-preserving knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

[25] Yuntao Chen, Naiyan Wang, and Zhaoxiang Zhang. DarkRank: Accelerating deep metric learning via cross sample similarities transfer. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[26] Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. Relational knowledge distillation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[27] Guodong Wang, Shumin Han, Errui Ding, and Di Huang. Student-teacher feature pyramid matching for anomaly detection. In British Machine Vision Conference, 2021.

[28] Mohammadreza Salehi, Niousha Sadjadi, Soroosh Baselizadeh, Mohammad H. Rohban, and Hamid R. Rabiee. Multiresolution knowledge distillation for anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[29] Chen Liu, Shibo He, Qihang Zhou, Shizhong Li, and Wenchao Meng. Large language model guided knowledge distillation for time series anomaly detection. In Proceedings of the Interna tional Joint Conference on Artificial Intelligence, 2024.

[30] Christopher Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. In Proceedings of the International Conference on Machine Learning, 2005.

[31] Tie-Yan Liu. Learning to rank for information retrieval. Foundations and Trends in Information Retrieval, 2009.

[32] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. Learning to rank: From pairwise approach to listwise approach. In Proceedings of the International Conference on Machine Learning, 2007.

[33] Mathieu Blondel, Olivier Teboul, Quentin Berthet, and Josip Djolonga. Fast differentiable sorting and ranking. In Proceedings of the International Conference on Machine Learning, 2020.

[34] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, 2018.

[35] Vincent Jacob, Fei Song, Arnaud Stiegler, Bijan Rad, Yanlei Diao, and Nesime Tatbul. Exathlon: A benchmark for explainable anomaly detection over time series. Proceedings of the VLDB Endowment, 2021.

[36] Rafal A. Angryk, Petrus C. Martens, Berkay Aydin, Dustin Kempton, Sushant S. Mahajan, Sunitha Basodi, Azim Ahmadzadeh, Xumin Cai, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi, Michael A. Schuh, and Manolis K. Georgoulis. Multivariate time series dataset for space weather data analytics. Scientific Data, 2020.

[37] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay, Aaron Elmore, and Michael J. Franklin. Volume under the surface: A new accuracy evaluation measure for time-series anomaly detection. Proceedings ofthe VLDB Endowment, 2022.

[38] Tom Fawcett. An introduction to ROC analysis. Pattern Recognition Letters, 2006.

[39] Jesse Davis and Mark Goadrich. The relationship between precision-recall and ROC curves. In Proceedings ofthe International Conference on Machine Learning, 2006.

[40] Nesime Tatbul, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. Precision and recall for time series. In Advances in Neural Information Processing Systems, 2018.

[41] Zhihao Dai, Ligang He, Shuang-Hua Yang, and Matthew Leeke. SARAD: Spatial associationaware anomaly detection and diagnosis for multivariate time series. In Advances in Neural Information Processing Systems, 2024.

[42] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. MOMENT: A Family of Open Time-Series Foundation Models. In Proceedings of the International Conference on Machine Learning, 2024.

[43] Xingjian Wu, Xiangfei Qiu, Zhengyu Li, Yihang Wang, Jilin Hu, Chenjuan Guo, Hui Xiong, and Bin Yang. CATCH: Channel-aware multivariate time series anomaly detection via frequency patching. In International Conference on Learning Representations, 2025.

[44] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval, 2009.

[45] Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A Decoder-Only Foundation Model for Time-Series Forecasting. In Proceedings of the 41st International Conference on Machine Learning, 2024.

[46] Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Bernie Wang. Chronos: Learning the Language of Time Series. Transactions on Machine Learning Research, 2024.

[47] Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified Training of Universal Time Series Forecasting Transformers. In Proceedings of the International Conference on Machine Learning, 2024.

[48] Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Biloš, Hena Ghonia, Nadhir Hassen, Anderson Schneider, Sahil Garg, Alexandre Drouin, Nicolas Chapados, Yuriy Nevmyvaka, and Irina Rish. Lag-Llama: Towards Foundation Models for Time Series Forecasting. In NeurIPS 2023 Workshop on Robustness of Few-Shot and Zero-Shot Learning in Large Foundation Models, 2023.

[49] Qichao Shentu, Beibu Li, Kai Zhao, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, and Chenjuan Guo. Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders. In International Conference on Learning Representations, 2025.

[50] Astha Garg, Wenyu Zhang, Jules Samaran, Ramasamy Savitha, and Chuan-Sheng Foo. An evaluation of anomaly detection and diagnosis in multivariate time series. IEEE Transactions on Neural Networks and Learning Systems, 2022.

[51] Alexis Huet, Jose Manuel Navarro, and Dario Rossi. Local evaluation of time series anomaly detection algorithms. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022.

[52] Chengsen Wang, Zirui Zhuang, Qi Qi, Jingyu Wang, Xingyu Wang, Haifeng Sun, and Jianxin Liao. Drift doesn’t matter: Dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection. Advances in neural information processing systems, 2023.

[53] HyunGi Kim, Jisoo Mok, Dongjun Lee, Jaihyun Lew, Sungjae Kim, and Sungroh Yoon. Causality-aware contrastive learning for robust multivariate time-series anomaly detection. In Proceedings of the International Conference on Machine Learning, 2025.

[54] Mononito Goswami, Cristian Ignacio Challu, Laurent Callot, Lenon Minorics, and Andrey Kan. Unsupervised model selection for time series anomaly detection. In International Conference on Learning Representations, 2023.

## A Rationale for Multi-Teacher Ranking Distillation

AnoMix trains a neural anomaly-scoring function $g _ { \theta } : \mathcal { X }  \mathbb { R }$ . For notational simplicity, we write $x _ { i }$ for the training window sample associated with an endpoint. Since the student outputs a scalar score, learning this function does not require a pointwise anomaly-score target. It requires supervision that shapes the ordering induced by $g _ { \theta }$ . In supervised anomaly detection, one could train the student against binary anomaly labels or labeled pairwise preferences. In unsupervised TSAD, however, no such labels are available. The only training signal comes from frozen teacher detectors.

A direct pointwise regression target is not identifiable in this setting. Each teacher produces a scalar anomaly score, but the score is generated from a detector-specific anomaly-scoring principle, such as reconstruction error, forecasting residual, association discrepancy, or representation deviation. Even after per-teacher normalization, these scores do not share a calibrated score scale. Direct regression can therefore spend student capacity fitting teacher-specific score scales rather than learning pairwise anomaly preferences supported by multiple teachers. Moreover, for any strictly increasing transformation $h _ { m }$

$$
s _ { m } ( x _ { i } ) > s _ { m } ( x _ { j } ) \quad \Longleftrightarrow \quad h _ { m } ( s _ { m } ( x _ { i } ) ) > h _ { m } ( s _ { m } ( x _ { j } ) ) .\tag{6}
$$

The teacher-induced ordering is preserved, but the pointwise regression target is changed. Thus, without anomaly labels, the score scale cannot be identified from data alone. AnoMix therefore distills pairwise anomaly preferences from teacher score gaps rather than treating teacher scores as pointwise regression targets.

Let $\tilde { z } _ { i } ^ { ( m ) }$ be the normalized score assigned by teacher m to sample $x _ { i }$ , and define the teacher score-gap vector for a pair (i, j) as

$$
\begin{array} { r } { \mathbf { \Delta } \Delta _ { i j } = \left( \Delta _ { i j } ^ { ( 1 ) } , \ldots , \Delta _ { i j } ^ { ( M ) } \right) , \qquad \Delta _ { i j } ^ { ( m ) } = \tilde { z } _ { i } ^ { ( m ) } - \tilde { z } _ { j } ^ { ( m ) } . } \end{array}\tag{7}
$$

Training a neural anomaly-scoring function from these teacher score gaps requires converting $\Delta _ { i j }$ into pairwise supervision. We write this supervision as

$$
\Gamma ( \Delta _ { i j } ) = ( A _ { i j } , y _ { i j } , c _ { i j } ) ,\tag{8}
$$

where $A _ { i j } \in \{ 0 , 1 \}$ indicates whether the pair is informative enough to use, $y _ { i j } \in \{ - 1 , 0 , + 1 \}$ specifies the consensus pairwise direction, and $c _ { i j } \in [ 0 , 1 ]$ controls the strength of the pair in training. Given such supervision, a natural pairwise learning objective is

$$
\mathcal { L } ( \theta ) = \frac { 1 } { \sum _ { i < j } A _ { i j } } \sum _ { i < j } A _ { i j } c _ { i j } \log \left( 1 + \exp \left( - y _ { i j } \left( g _ { \theta } ( x _ { i } ) - g _ { \theta } ( x _ { j } ) \right) \right) \right) .\tag{9}
$$

If no pair is retained in a mini-batch, the batch contributes zero loss. This objective directly trains the student score margin $g _ { \theta } ( x _ { i } ) - g _ { \theta } ( x _ { j } )$ rather than regressing to teacher-specific pointwise scores.

The label-free multi-teacher setting imposes three requirements on $\Gamma ( \Delta _ { i j } )$ . First, it should select informative pairs using teacher-level score gaps, because averaging teacher scores before pair selection can suppress a clear gap from one detector when other detectors give small or neutral gaps. Second, it should produce a scalar-compatible consensus direction, because the student itself outputs a single scalar anomaly score and can only realize scalar score orderings. Third, it should be conflict-aware, so that pairs with strong opposing teacher score gaps do not contribute as much as pairs with little directional conflict.

CoRank instantiates Γ with the following choices. The support indicator is

$$
A _ { i j } = \mathbf { 1 } \left[ \| \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } _ { i j } \| _ { \infty } > \delta \right] = \mathbf { 1 } \left[ \operatorname* { m a x } _ { m } | \Delta _ { i j } ^ { ( m ) } | > \delta \right] .\tag{10}
$$

This retains a pair whenever at least one teacher observes a meaningful score gap. The condition does not assume that the teacher producing the gap is correct. It only marks the pair as meaningful for consensus supervision before teacher-level contrasts are suppressed by averaging.

The consensus direction is

$$
y _ { i j } = \mathrm { s g n } \left( \mathbf { 1 } ^ { \top } \Delta _ { i j } \right) = \mathrm { s g n } \left( \sum _ { m = 1 } ^ { M } \Delta _ { i j } ^ { ( m ) } \right) .\tag{11}
$$

This direction is equivalent to comparing the averaged normalized teacher score

$$
\mathrm { s g n } \left( \sum _ { m = 1 } ^ { M } \Delta _ { i j } ^ { ( m ) } \right) = \mathrm { s g n } \left( \bar { z } _ { i } - \bar { z } _ { j } \right) , \qquad \bar { z } _ { i } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \tilde { z } _ { i } ^ { ( m ) } .\tag{12}
$$

This equivalence defines only the consensus direction after the support set has been constructed from teacher-level gaps. Thus, CoRank does not ask the student to fit an arbitrary cyclic preference relation. The pairwise direction is induced by a scalar teacher-consensus score and is compatible with the student’s scalar output.

The confidence weight is

$$
c _ { i j } = \frac { \big | \mathbf { 1 } ^ { \top } \mathbf { \Delta } \mathbf { \Delta } _ { i j } \big | } { \| \mathbf { \Delta } \mathbf { { A } } _ { i j } \| _ { 1 } } = \frac { \left| \sum _ { m = 1 } ^ { M } \Delta _ { i j } ^ { ( m ) } \right| } { \sum _ { m = 1 } ^ { M } | \Delta _ { i j } ^ { ( m ) } | } .\tag{13}
$$

By the triangle inequality, $0 \leq c _ { i j } \leq 1$ for every retained pair. This weight measures directional conflict among teacher score gaps. It is large when teacher score gaps provide little opposition to the consensus direction, and small when teacher score gaps strongly conflict. Importantly, $c _ { i j }$ should not be interpreted as an oracle estimate of teacher correctness. It is a label-free conflict weight. If the consensus direction is zero, the numerator is zero and the pair does not affect the update.

Substituting these choices into the general pairwise objective and writing $\mathcal { P } = \{ ( i , j ) : A _ { i j } = 1 \}$ yields

$$
\mathcal { L } _ { \mathrm { C o R a n k } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } c _ { i j } \log \left( 1 + \exp \left( - y _ { i j } \left( g _ { \theta } ( x _ { i } ) - g _ { \theta } ( x _ { j } ) \right) \right) \right) .\tag{14}
$$

Therefore, CoRank provides a concrete way to convert heterogeneous teacher score gaps into pairwise supervision for training a neural anomaly-scoring function. Compared with pointwise regression, it avoids fitting a non-identifiable score scale. Compared with single-teacher ranking, it uses comparative signals from multiple anomaly-scoring principles. Compared with ranking on an averaged teacher score, it selects informative pairs from teacher-level score gaps before aggregation and then discounts directional teacher conflict.

## B Student Architectures

Endpoint Scoring. Both the main AnoMix student and AnoMix-ZS follow the same prediction template. Given an input window ending at time $t ,$ the student first constructs an endpoint-centered representation $\mathbf { h } _ { t } \in \mathbb { R } ^ { H }$ and then maps it to one scalar anomaly score:

$$
\mathbf { h } _ { t } = \boldsymbol { \Psi } ( \mathbf { W } _ { t } ) , \qquad \hat { \boldsymbol { z } } _ { t } = \mathbf { w } _ { o } ^ { \top } \mathbf { h } _ { t } + b _ { o } .\tag{15}
$$

This template matches the supervision used by AnoMix, since teacher scores and CoRank pairs are defined for window endpoints. The main student and the zero-shot student differ only in the representation map $\Psi ( \cdot )$

Fixed-Dimensional Encoder. For the main benchmark setting, all train and test windows within a dataset have the same number of variables $D .$ . We therefore use a compact multivariate causal TCN encoder. Given $\mathbf { W } _ { t } = ( \mathbf { x } _ { t - L + 1 } , \ldots , \mathbf { x } _ { t } ) \in \mathbb { R } ^ { L \times D }$ , each timestep is projected into the hidden dimension:

$$
\mathbf { h } _ { i } ^ { ( 0 ) } = \mathbf { P } ^ { \top } \mathbf { x } _ { i } + \mathbf { b } _ { p } \in \mathbb { R } ^ { H } , \qquad i = t - L + 1 , \ldots , t ,\tag{16}
$$

which defines ${ \bf H } ^ { ( 0 ) } = ( { \bf h } _ { t - L + 1 } ^ { ( 0 ) } , \dots , { \bf h } _ { t } ^ { ( 0 ) } ) \in \mathbb { R } ^ { L \times H }$ . The temporal encoder applies N causal convolutional blocks:

$$
{ \bf H } ^ { ( \ell ) } = \Phi _ { \ell } \Big ( { \bf H } ^ { ( \ell - 1 ) } \Big ) , \qquad \ell = 1 , \dots , N .\tag{17}
$$

Here, $\mathbf { P } \in \mathbb { R } ^ { D \times H }$ and $\mathbf { b } _ { p } \in \mathbb { R } ^ { H }$ define the input projection, and $\Phi _ { \ell }$ denotes a causal temporal block. The endpoint representation used by the shared scoring template is

$$
\begin{array} { r } { \mathbf { h } _ { t } = \mathbf { H } _ { L } ^ { ( N ) } . } \end{array}\tag{18}
$$

In the main paper, we use $N = 6$ blocks with dilations {1, 2, 4, 8, 16, 32} and kernel size $k = 3$ Each block consists of a causal 1D convolution, residual connection, GELU activation, and dropout. The resulting receptive field is 127 when $L = 1 2 7$ , so the endpoint representation can summarize the full input window before the final scalar prediction is made. Under the benchmark settings used in the main paper, this student contains under 80k parameters.

Variable-Dimensional Encoder. In zero-shot transfer, the held-out dataset may have a different number of variables from the source datasets. The main multivariate encoder is tied to a fixed input dimension $D ,$ because its first projection maps $\mathbb { R } ^ { D }$ to $\mathbb { R } ^ { H }$ . AnoMix-ZS keeps the same endpointscoring template but replaces this D-dependent input encoder with a channel-independent encoder. It processes each variable with shared parameters, constructs one endpoint representation per variable, and aggregates those representations into the single $\mathbf { h } _ { t }$ used by the score head.

Let $\mathbf { x } ^ { ( d ) } = ( x _ { t - L + 1 , d } , \ldots , x _ { t , d } ) \in \mathbb { R } ^ { L \times 1 }$ denote the univariate window for variable $d .$ All variables are reshaped into the batch dimension and passed through the same univariate projection:

$$
\mathbf { h } _ { d , i } ^ { ( 0 ) } = \phi ( x _ { i , d } ) \in \mathbb { R } ^ { H } , \qquad d = 1 , \dotsc , D .\tag{19}
$$

A fixed sinusoidal positional encoding is added, followed by a single-layer Pre-LayerNorm Transformer and four causal TCN blocks with dilations {1, 2, 4, 8}. This produces a per-variable endpoint representation $\mathbf { g } _ { t , d } \in \mathbb { R } ^ { H }$ . AnoMix-ZS then aggregates variables with learned attention weights:

$$
a _ { t , d } = { \frac { \exp ( { \bf w } _ { a } ^ { \top } { \bf g } _ { t , d } ) } { \sum _ { r = 1 } ^ { D } \exp ( { \bf w } _ { a } ^ { \top } { \bf g } _ { t , r } ) } } , \qquad { \bf h } _ { t } = \sum _ { d = 1 } ^ { D } a _ { t , d } { \bf g } _ { t , d } .\tag{20}
$$

The same scalar score head then computes $\hat { z } _ { t } = \mathbf { w } _ { o } ^ { \top } \mathbf { h } _ { t } + b _ { o }$ . Thus, AnoMix-ZS changes only the encoder used to obtain the endpoint representation, while preserving the endpoint-centered scoring template and the CoRank objective. With $H = 6 4$ , one Transformer layer, four TCN blocks, and one score head, AnoMix-ZS has 83k parameters independent of $D .$

Zero-Shot Training. AnoMix-ZS uses the same CoRank objective as the main method. For each target dataset $\bar { \mathcal { D } } ^ { \star }$ , the student is trained on the concatenation of training windows from the remaining six benchmark datasets and evaluated on ${ \mathcal { D } } ^ { \star }$ with no target-dataset fine-tuning, no target teacher scores, and no target training data. Teacher scores on the training mixture are obtained from the same stored per-dataset teacher checkpoints used in the main experiments, and CoRank supervision is constructed only within each dataset’s mini-batch so that teacher normalizations remain dataset-consistent.

## C Pseudocode

Algorithm 1 summarizes the full AnoMix pipeline used in the main paper. It separates offline teacher preprocessing, single-head student training with CoRank, and teacher-free inference.

Algorithm 1 AnoMix Training and Inference   
Input: $\mathbf { X } _ { \mathrm { t r a i n } } \in \mathbb { R } ^ { T \times D } , \mathbf { X } _ { \mathrm { t e s t } } \in \mathbb { R } ^ { T ^ { \prime } \times D }$ , teachers $\{ f _ { m } \} _ { m = 1 } ^ { M }$ , window length L, batch size B, $B ,$ epochs $E ,$ learning   
rate $\eta ,$ threshold δ   
Output: Student $g _ { \theta }$ and anomaly scores $\{ s _ { t } \} _ { t = L } ^ { T ^ { \prime } }$   
Stage 1. Offline teacher preprocessing   
1: for $m = 1 , \ldots , M$ do   
2: Compute teacher scores ${ z ^ { ( m ) } \gets f _ { m } ( \mathbf { X } _ { \mathrm { t r a i n } } ) }$   
3: Min-max normalize $z ^ { ( m ) }$ on $\mathbf { X } _ { \mathrm { t r a i n } }$ to obtain $\tilde { z } ^ { ( m ) }$   
4: end for   
Stage 2. Single-head student training with $C o R a n k$   
5: $\mathcal { D }  \{ ( \mathbf { W } _ { t } , \tilde { \mathbf { z } } _ { t } ) \} _ { t = L } ^ { T } ,$ where $\tilde { \mathbf { z } } _ { t } = ( \tilde { z } _ { t } ^ { ( 1 ) } , \dots , \tilde { z } _ { t } ^ { ( M ) } )$   
6: Initialize g<sub>θ</sub> : $\mathbb { R } ^ { L \times \overline { { D } } ^ { * } } $ R as the TCN student   
7: Initialize AdamW with learning rate η and cosine annealing for E epochs   
8: for $e = 1 , \ldots , E$ do   
9: for mini-batch $\mathbf { \nabla } B = \{ ( \mathbf { W } _ { i } , \tilde { \mathbf { z } } _ { i } ) \} _ { i = 1 } ^ { B } \sim \mathcal { D }$ do   
10: $\hat { z } _ { i } \gets g _ { \theta } ( \mathbf { W } _ { i } )$ for all $i \in [ \bar { B } ]$   
11: Enumerate all unordered pairs $( i , j )$ with $i < j$   
12: $\Delta _ { i j }  \tilde { \mathbf { z } } _ { i } - \tilde { \mathbf { z } } _ { j }$ for each pair   
13: $\mathcal { P }  \{ ( i , j ) : i < j ,$ max<sub>m</sub> $| \Delta _ { i j } ^ { ( m ) } | > \delta \}$   
14: if $\mathcal { P } = \emptyset$ then   
15: continue   
16: end if   
17: for each $( i , j ) \in \mathscr { P }$ do   
18: y<sub>ij</sub> ← sgn $\left( \sum _ { m = 1 } ^ { M } \Delta _ { i j } ^ { ( m ) } \right)$   
19: $c _ { i j } \gets \left| \sum _ { m = 1 } ^ { M } \Delta _ { i j } ^ { ( m ) } \right| / \sum _ { m = 1 } ^ { M } | \Delta _ { i j } ^ { ( m ) } |$   
20: $d _ { i j } \gets \dot { { \hat { z } } } _ { i } - { \hat { z } } _ { j }$   
21: end for   
22: $\begin{array} { r } { \mathcal { L } _ { \mathrm { C o R a n k } }  \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } c _ { i j } \log ( 1 + } \end{array}$ exp(− $\boldsymbol { \cdot } y _ { i j } d _ { i j } ) )$   
23: $\theta \gets \theta - \eta \nabla _ { \theta } \dot { \mathcal { L } }$ CoRank   
24: end for   
25: end for   
Stage 3. Teacher-free inference   
26: for $t = L , \ldots , T ^ { \prime }$ do   
27: $s _ { t } \gets g _ { \theta } ( \mathbf { W } _ { t } ^ { \mathrm { t e s t } } )$   
28: end for   
29: return $\{ s _ { t } \} _ { t = I } ^ { T ^ { \prime } }$

## D Visualization

Figure S1 visualizes the test-time anomaly scores produced by the three Sel-3 teachers (SARAD, CrossAD, OracleAD) and AnoMix on the seven benchmark datasets. Each series is min-max normalized to [0, 1] for direct visual comparison, and anomaly segments are shaded in red. Across these datasets, AnoMix can be compared directly with the teacher score sequences and often assigns high scores to labeled anomaly segments without simply copying any single teacher.

![](images/1e27259d184a4470e7392d9e9802d0ff61e0bf438bd923869b83e72c0949cb0c.jpg)  
Figure S1: Test-time anomaly scores. Rows correspond to the seven benchmark datasets and columns to SARAD, CrossAD, OracleAD, and AnoMix. Scores are min-max normalized to [0, 1]. Red shaded bands indicate labeled anomaly segments.

## E Full Results

PR Metrics. Table S1 reports VUS-PR, AUC-PR, and R-AUC-PR (5-seed mean, %).  
Table S1: PR metrics on the seven benchmark datasets (5-seed mean, %). Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td colspan="3">PSM</td><td colspan="3"></td><td colspan="3">Exathlon</td><td colspan="3">SMD</td><td colspan="3">SMAP</td><td colspan="3">Daphnet</td><td colspan="3">SWAN</td></tr><tr><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td></tr><tr><td>Omni</td><td>22.52</td><td>21.43</td><td>24.07</td><td>27.32</td><td>24.03</td><td>32.12</td><td>99.75</td><td>98.88</td><td>99.96</td><td>69.98</td><td>71.68</td><td>75.18</td><td>27.68</td><td>26.97</td><td>28.08</td><td>18.38</td><td>16.97</td><td>19.25</td><td>91.16</td><td>54.34</td><td>91.51</td></tr><tr><td>USAD20</td><td>22.73</td><td>22.52</td><td>24.35</td><td>33.39</td><td>29.60</td><td>38.56</td><td>99.63</td><td>98.47</td><td>99.93</td><td>71.97</td><td>72.42</td><td>77.36</td><td>27.58</td><td>26.86</td><td>27.98</td><td>17.15</td><td>15.24</td><td>18.31</td><td>88.60</td><td>49.50</td><td>87.86</td></tr><tr><td>A.T</td><td>23.13</td><td>21.95</td><td>23.18</td><td>23.55</td><td>20.94</td><td>26.87</td><td>24.88</td><td>24.59</td><td>25.28</td><td>11.80</td><td>9.74</td><td>13.74</td><td>22.30</td><td>21.35</td><td>22.90</td><td>10.33</td><td>9.40</td><td>11.35</td><td>81.07</td><td>36.05</td><td>85.01</td></tr><tr><td>TranAD&#x27;22</td><td>22.25</td><td>21.27</td><td>22.61</td><td>37.86</td><td>31.38</td><td>43.86</td><td>93.88</td><td>93.35</td><td>93.97</td><td>52.05</td><td>46.92</td><td>45.51</td><td>37.06</td><td>35.94</td><td>37.54</td><td>17.98</td><td>16.48</td><td>19.29</td><td>86.48</td><td>41.25</td><td>86.87</td></tr><tr><td>TimesNet23</td><td>24.32</td><td>20.33</td><td>26.19</td><td>24.35</td><td>20.87</td><td>28.20</td><td>85.33</td><td>81.74</td><td>88.15</td><td>53.06</td><td>44.41</td><td>56.79</td><td>23.90</td><td>22.80</td><td>24.62</td><td>33.97</td><td>31.01</td><td>36.82</td><td>88.77</td><td>43.48</td><td>90.36</td></tr><tr><td>SARAD{24</td><td>26.78</td><td>22.28</td><td>31.37</td><td>29.82</td><td>24.90</td><td>34.93</td><td>98.27</td><td>95.47</td><td>99.59</td><td>62.65</td><td>56.33</td><td>70.98</td><td>37.33</td><td>36.45</td><td>37.90</td><td>38.72</td><td>35.11</td><td>41.02</td><td>90.20</td><td>52.56</td><td>89.63</td></tr><tr><td>MOMENT24</td><td>16.59</td><td>15.68</td><td>17.76</td><td>20.21</td><td>17.98</td><td>22.60</td><td>97.17</td><td>93.89</td><td>98.91</td><td>10.19</td><td>8.20</td><td>12.16</td><td>27.85</td><td>27.43</td><td>28.53</td><td>35.10</td><td>31.58</td><td>37.15</td><td>86.25</td><td>44.56</td><td>86.65</td></tr><tr><td>CATCH25</td><td>23.42</td><td>19.76</td><td>27.25</td><td>24.61</td><td>20.83</td><td>28.47</td><td>97.79</td><td>94.52</td><td>99.52</td><td>67.33</td><td>60.17</td><td>76.64</td><td>27.45</td><td>26.74</td><td>27.84</td><td>38.86</td><td>35.82</td><td>40.76</td><td>89.66</td><td>44.65</td><td>91.75</td></tr><tr><td>CrossAD25</td><td>27.66</td><td>24.44</td><td>32.01</td><td>26.33</td><td>23.70</td><td>30.82</td><td>92.47</td><td>89.33</td><td>95.76</td><td>72.52</td><td>72.48</td><td>80.65</td><td>35.62</td><td>34.27</td><td>37.42</td><td>36.31</td><td>32.67</td><td>38.99</td><td>91.66</td><td>49.04</td><td>92.68</td></tr><tr><td>OracleAD&#x27;25</td><td>28.68</td><td>27.92</td><td>29.10</td><td>26.44</td><td>23.54</td><td>29.10</td><td>97.18</td><td>96.95</td><td>97.32</td><td>52.69</td><td>48.61</td><td>55.03</td><td>40.18</td><td>39.61</td><td>40.70</td><td>34.60</td><td>31.77</td><td>37.47</td><td>86.30</td><td>48.78</td><td>86.88</td></tr><tr><td>AnoMix</td><td>44.03</td><td>54.90</td><td>44.75</td><td>42.40</td><td>38.19</td><td>47.25</td><td>99.97</td><td>99.95</td><td>99.98</td><td>89.37</td><td>81.79</td><td>91.99</td><td>97.65</td><td>96.85</td><td>98.40</td><td>52.37</td><td>47.98</td><td>56.29</td><td>91.76</td><td>58.74</td><td>90.59</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

ROC Metrics. Table S2 reports VUS-ROC, AUC-ROC, and R-AUC-ROC (5-seed mean, %).

Table S2: ROC metrics on the seven benchmark datasets (5-seed mean, %). Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td colspan="4">PSM</td><td colspan="3">SWaT</td><td colspan="3">Exathlon</td><td colspan="3">SMD</td><td colspan="3">SMAP</td><td colspan="3">Daphnet</td><td colspan="3">SWAN</td></tr><tr><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td></td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td><td>VUS</td><td>AUC</td><td>R-AUC</td></tr><tr><td>Omni</td><td>51.59</td><td>59.49</td><td>52.74</td><td>59.42</td><td>59.91</td><td>63.64</td><td>99.98</td><td>99.91</td><td>100.00</td><td>95.40</td><td>95.17</td><td>96.34</td><td>49.93</td><td>48.90</td><td>50.83</td><td>78.60</td><td>79.20</td><td></td><td>79.01</td><td>92.06</td><td>63.19</td><td>86.54</td></tr><tr><td>USAD20</td><td>52.15</td><td>58.18</td><td>52.52</td><td>68.92</td><td>70.88</td><td>72.92</td><td>99.97</td><td>99.88</td><td>99.99</td><td>95.98</td><td>95.40</td><td>96.88</td><td></td><td>49.80</td><td>48.77</td><td>50.70</td><td>76.95</td><td>76.13</td><td>78.18</td><td>89.80</td><td>56.15</td><td>81.06</td></tr><tr><td>A.T</td><td>53.76</td><td>54.64</td><td>44.00</td><td>56.63</td><td>54.36</td><td>59.22</td><td>37.70</td><td>37.13</td><td>38.35</td><td>44.03</td><td>42.04</td><td>44.83</td><td>40.56</td><td>39.73</td><td></td><td>41.49</td><td>46.28</td><td>44.92</td><td>47.59</td><td>86.33</td><td>50.95</td><td>78.38</td></tr><tr><td>TranAD22</td><td>55.90</td><td>55.14</td><td>48.22</td><td>77.06</td><td>72.97</td><td>81.54</td><td>99.50</td><td>99.47</td><td>99.50</td><td>86.51</td><td>82.75</td><td>81.84</td><td>71.96</td><td>70.85</td><td>72.81</td><td></td><td>76.71</td><td>73.04</td><td>79.00</td><td>92.02</td><td>52.60</td><td>81.50</td></tr><tr><td>TimesNet23</td><td>57.91</td><td>53.22</td><td>57.87</td><td>59.29</td><td>53.56</td><td>64.78</td><td>98.61</td><td>98.24</td><td>98.88</td><td>93.96</td><td>90.60</td><td>92.45</td><td>59.22</td><td>58.06</td><td>60.13</td><td></td><td>90.25</td><td>88.98</td><td>91.30</td><td>93.87</td><td>56.52</td><td>86.33</td></tr><tr><td>SARAD24</td><td>58.10</td><td>57.44</td><td>62.37</td><td>68.29</td><td>64.29</td><td>72.41</td><td>99.86</td><td>99.67</td><td>99.96</td><td>95.40</td><td>93.58</td><td>96.83</td><td>75.27</td><td>74.43</td><td>75.92</td><td></td><td></td><td>88.91</td><td>90.93</td><td>92.11</td><td>58.13</td><td>84.49</td></tr><tr><td>MOMENT24</td><td>42.39</td><td>44.07</td><td>40.29</td><td>54.95</td><td>50.25</td><td>59.33</td><td>99.66</td><td>99.24</td><td>99.83</td><td>57.27</td><td>49.95</td><td>61.94</td><td>54.72</td><td>53.35</td><td>56.10</td><td></td><td>89.96</td><td></td><td></td><td>92.07</td><td>55.78</td><td>81.22</td></tr><tr><td>CATCH25</td><td>53.99</td><td>51.92</td><td>57.53</td><td>58.82</td><td>53.47</td><td>64.30</td><td>99.83</td><td>99.59</td><td>99.96</td><td>96.45</td><td>94.40</td><td>97.89</td><td>49.69</td><td>48.66</td><td>50.58</td><td>91.37</td><td>91.71</td><td>91.03 90.89</td><td>92.28 91.87</td><td>92.16</td><td>53.23</td><td>88.17</td></tr><tr><td>CrossAD&#x27;25</td><td>59.02</td><td>57.87</td><td>63.33</td><td>60.79</td><td>56.96</td><td>66.17</td><td>99.41</td><td>99.18</td><td>99.65</td><td>96.49</td><td>97.18</td><td>97.38</td><td>52.95</td><td>51.77</td><td>54.24</td><td>89.77</td><td>89.74</td><td></td><td>90.76</td><td>92.91</td><td>59.42</td><td>88.37</td></tr><tr><td>OracleAD25</td><td>66.33</td><td>65.02</td><td>64.19</td><td>59.87</td><td>58.39</td><td>58.64</td><td>99.66</td><td>99.57</td><td>99.69</td><td>85.71</td><td>80.39</td><td>88.50</td><td>79.88</td><td>78.80</td><td>80.69</td><td>88.46</td><td>85.71</td><td></td><td>90.14</td><td>92.44</td><td>60.36</td><td>81.47</td></tr><tr><td>AnoMix</td><td>70.24</td><td>77.01</td><td>69.98</td><td>78.56</td><td>77.80</td><td>80.62</td><td>100.00</td><td>99.99</td><td>100.00</td><td>98.87</td><td>97.00</td><td>98.81</td><td>98.38</td><td>97.63</td><td>99.02</td><td>94.13</td><td></td><td>92.94</td><td>95.07</td><td></td><td></td><td>85.33</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>93.75</td><td>69.06</td><td></td></tr></table>

F1-Based Metrics. Table S3 reports point-wise F1, Event-F1 [50], and Aff-F [51] (5-seed mean, %).

Table S3: F1-based metrics on the seven benchmark datasets (5-seed mean, %). Boldface indicates the best result in each column, and underlined values indicate the second-best result.
<table><tr><td rowspan="2">Method</td><td colspan="3">PSM</td><td colspan="3">SWaT</td><td colspan="3">Exathlon</td><td colspan="3">SMD</td><td colspan="3">SMAP</td><td colspan="3">Daphnet</td><td colspan="3">SWAN</td></tr><tr><td>F1</td><td>Event</td><td>Aff</td><td>F1</td><td>Event</td><td>Aff</td><td>F1</td><td>Event</td><td>Aff</td><td>F1</td><td>Event</td><td>Aff</td><td>F1</td><td>Event</td><td>Aff</td><td>F1</td><td>Event</td><td>Aff</td><td>F1</td><td>Event</td><td>Aff</td></tr><tr><td>Omni</td><td>32.33</td><td>33.23</td><td>71.17</td><td>36.08</td><td>42.27</td><td>71.63</td><td>98.90</td><td>100.00</td><td>99.92</td><td>65.77</td><td>73.18</td><td>83.23</td><td>40.81</td><td>100.00</td><td>94.05</td><td>31.66</td><td>33.39</td><td>86.13</td><td>49.72</td><td>49.17</td><td>71.05</td></tr><tr><td>USAD20</td><td>31.42</td><td>32.11</td><td>71.52</td><td>45.84</td><td>44.86</td><td>74.07</td><td>98.70</td><td>99.85</td><td>99.91</td><td>64.51</td><td>66.15</td><td>83.88</td><td>40.81</td><td>100.00</td><td>94.05</td><td>30.87</td><td>31.11</td><td>85.85</td><td>49.17</td><td>49.17</td><td>71.05</td></tr><tr><td>A.T</td><td>30.13</td><td>28.58</td><td>66.79</td><td>32.80</td><td>42.83</td><td>70.23</td><td>29.57</td><td>78.63</td><td>87.28</td><td>15.07</td><td>35.43</td><td>64.33</td><td>30.92</td><td>57.50</td><td>77.16</td><td>16.68</td><td>46.50</td><td>79.35</td><td>49.17</td><td>32.23</td><td>53.48</td></tr><tr><td>TranAD22</td><td>29.30</td><td>29.20</td><td>69.43</td><td>48.65</td><td>33.39</td><td>71.64</td><td>97.62</td><td>100.00</td><td>99.11</td><td>52.64</td><td>76.91</td><td>91.88</td><td>63.20</td><td>100.00</td><td>93.14</td><td>29.36</td><td>61.49</td><td>88.35</td><td>49.17</td><td>49.17</td><td>71.05</td></tr><tr><td>TimesNet&#x27;23</td><td>29.20</td><td>29.20</td><td>69.43</td><td>30.39</td><td>30.26</td><td>71.63</td><td>90.08</td><td>94.00</td><td>97.45</td><td>51.94</td><td>67.34</td><td>86.40</td><td>40.20</td><td>100.00</td><td>92.58</td><td>40.48</td><td>58.10</td><td>82.44</td><td>49.17</td><td>49.17</td><td>71.05</td></tr><tr><td>SARAD{24</td><td>31.10</td><td>29.20</td><td>69.43</td><td>36.93</td><td>35.53</td><td>71.64</td><td>97.77</td><td>98.92</td><td>99.79</td><td>55.60</td><td>65.40</td><td>89.31</td><td>52.73</td><td>100.00</td><td>94.05</td><td>47.63</td><td>58.84</td><td>90.81</td><td>49.17</td><td>49.17</td><td>71.05</td></tr><tr><td>MOMENT24</td><td>29.20</td><td>29.20</td><td>69.43</td><td>30.05</td><td>49.88</td><td>76.89</td><td>94.78</td><td>98.83</td><td>99.72</td><td>15.09</td><td>22.12</td><td>71.66</td><td>30.90</td><td>100.00</td><td>93.75</td><td>46.99</td><td>53.02</td><td>89.70</td><td>49.17</td><td>49.17</td><td>71.05</td></tr><tr><td>CATCH{25</td><td>29.36</td><td>39.75</td><td>78.87</td><td>31.60</td><td>40.69</td><td>78.71</td><td>98.26</td><td>98.95</td><td>99.81</td><td>57.37</td><td>70.97</td><td>91.43</td><td>40.80</td><td>100.00</td><td>94.05</td><td>46.15</td><td>54.07</td><td>89.46</td><td>49.17</td><td>49.17</td><td>71.05</td></tr><tr><td>CrossAD&#x27;25</td><td>29.83</td><td>43.74</td><td>81.07</td><td>33.26</td><td>37.96</td><td>75.96</td><td>97.71</td><td>97.76</td><td>99.60</td><td>68.21</td><td>67.25</td><td>88.26</td><td>40.82</td><td>100.00</td><td>94.75</td><td>49.03</td><td>52.18</td><td>84.40</td><td>49.18</td><td>49.17</td><td>71.05</td></tr><tr><td>OracleAD25</td><td>35.71</td><td>45.93</td><td>77.66</td><td>35.37</td><td>37.15</td><td>74.98</td><td>95.34</td><td>100.00</td><td>99.59</td><td>48.00</td><td>88.26</td><td>89.88</td><td>56.92</td><td>100.00</td><td>93.42</td><td>41.13</td><td>68.14</td><td>90.11</td><td>50.05</td><td>49.18</td><td>71.05</td></tr><tr><td>AnoMix</td><td>|56.69</td><td>50.93</td><td>70.97</td><td>53.43</td><td>42.89</td><td>74.58</td><td>99.36</td><td>100.00</td><td>100.00</td><td>82.36</td><td>72.85</td><td>85.84</td><td>96.82</td><td>100.00</td><td>99.97</td><td>50.07</td><td>78.21</td><td>94.28</td><td>53.57</td><td>50.13</td><td>71.12</td></tr></table>

## F Datasets

Table S4 summarizes the seven multivariate time series anomaly detection datasets used in the paper. We selected these datasets to evaluate AnoMix across diverse benchmark conditions, including different training-set sizes, anomaly ratios, event counts, and numbers of variates. We treat subsets from the same benchmark source as separate benchmark instances rather than different splits of a single dataset, since they differ in dimensionality and data distribution, following prior works [3, 52, 53]. For datasets with multiple available subsets, the subset was fixed before evaluating AnoMix by choosing among subsets with sufficient anomaly ratios for evaluation. The first six are sourced from TSB-AD [6], which provides standardized preprocessing, train/test splits, and anomaly labels, and SWAN is sourced from TAB [7].

Table S4: Statistics of the seven benchmark datasets.
<table><tr><td>Dataset</td><td>Subset</td><td>Domain</td><td>D</td><td>Train</td><td>Validation</td><td>Test</td><td>Anomaly (%)</td><td>Events</td></tr><tr><td>PSM</td><td>一</td><td>Server monitoring</td><td>25</td><td>67,500</td><td>7,500</td><td>142,624</td><td>11.2</td><td>72</td></tr><tr><td>SWaT</td><td></td><td>Cyber-physical control</td><td>66</td><td>3,374</td><td>375</td><td>11,247</td><td>13.2</td><td>6</td></tr><tr><td>Exathlon</td><td>id22</td><td>Distributed stream processing</td><td>31</td><td>9,689</td><td>1,077</td><td>32,300</td><td>12.6</td><td>3</td></tr><tr><td>SMD</td><td>id6</td><td>Internet-company server</td><td>38</td><td>6,462</td><td>718</td><td>21,542</td><td>6.1</td><td>10</td></tr><tr><td>SMAP</td><td>id1</td><td>Satellite telemetry</td><td>25</td><td>1,846</td><td>206</td><td>6,157</td><td>13.7</td><td>1</td></tr><tr><td>Daphnet</td><td>一</td><td>Wearable accelerometer</td><td>9</td><td>8,723</td><td>970</td><td>29,081</td><td>5.9</td><td>6</td></tr><tr><td>SWAN</td><td>一</td><td>Solar-flare monitoring</td><td>38</td><td>54,000</td><td>6,000</td><td>60,000</td><td>32.6</td><td>6,148</td></tr></table>

• PSM [1]: Server-monitoring metrics collected from eBay’s production infrastructure. The 25 channels cover CPU, memory, disk, and network usage, with anomalies corresponding to real software incidents.

• SWaT [2]: Sensor and actuator readings from the Secure Water Treatment testbed. The 66 channels cover flow, tank-level, valve, and chemical-dosing signals under cyber-physical attacks. We use the processed TSB-AD version, while the original SWaT data are distributed by iTrust and require an access request through the official dataset website before download: https://itrust.sutd.edu.sg/itrust-labs\_datasets/dataset\_info/.

• Exathlon [35]: Application-level metrics from distributed stream-processing jobs running on Apache Spark. The 31 channels include job latency, throughput, CPU utilization, and garbage-collection statistics, and anomalies are caused by resource contention, configuration errors, and infrastructure failures.

• SMD [3]: Server Machine Dataset consisting of 38 monitoring metrics (CPU, memory, disk, network) collected from a large Internet company.

• SMAP [4]: Telemetry data from NASA’s Soil Moisture Active Passive satellite. The 25 channels include sensor readings and actuator commands, with anomalies corresponding to sensor malfunctions and actuator failures.

• Daphnet [5]: Wearable accelerometer data from Parkinson’s disease patients. Three tri-axial accelerometers produce 9 channels, and anomalies correspond to freezing-of-gait episodes.

• SWAN [36]: SHARP magnetic-field metrics from NASA SDO/HMI used for solar flare monitoring. The 38 channels describe magnetic-field properties, region geometry, and X-ray flux history, with anomalies corresponding to active regions associated with elevated flare activity.

## G Metrics

We report threshold-independent PR and ROC metrics in three forms. The threshold-independent metrics capture complementary aspects of TSAD performance, including pointwise discrimination, tolerance to small temporal offsets, and ranking quality across the full score range.

They summarize performance across all possible score thresholds and therefore evaluate the quality of the anomaly scores themselves rather than one specific threshold choice.

• AUC-PR, AUC-ROC [38, 39]: Standard pointwise areas under the Precision-Recall and ROC curves. AUC-PR is especially informative under severe class imbalance, while AUC-ROC summarizes global ranking quality over normal and anomalous points.

• R-AUC-PR, R-AUC-ROC [37, 40]: Range-based variants that relax exact timestamp matching and instead credit predictions that fall within a tolerance window around each anomaly segment. They are therefore more appropriate when slight temporal delays or advances should not be penalized as harshly as pointwise errors.

• VUS-PR, VUS-ROC [37]: Volume Under the Surface extends range-based AUC by integrating not only over score thresholds but also over tolerance-window sizes. VUS therefore provides a broader threshold-free view of time-series detection quality than either standard or fixed-range AUC.

## H Baselines and Experimental Protocol

## Deep TSAD Baselines.

• OmniAnomaly (KDD 2019) [3]: a stochastic reconstruction model that combines recurrent dynamics with a variational latent-variable formulation for sequence anomaly detection.

• USAD (KDD 2020) [9]: an unsupervised adversarial autoencoder that detects anomalies through reconstruction inconsistency under dual decoder objectives.

• Anomaly Transformer (ICLR 2022) [15]: a Transformer-based detector that uses association discrepancy to identify unusual attention structure.

• TranAD (VLDB 2022) [10]: a forecasting-oriented Transformer that predicts future behavior and flags deviations from those predictions.

• TimesNet (ICLR 2023) [11]: a general time-series backbone that models multi-period temporal variation and is used here as a forecasting-style anomaly detector.

• MOMENT (ICML 2024) [42]: a pretrained time-series foundation model used in our targetdataset benchmark as a foundation-model-based anomaly baseline. For Table 1, we follow the TSB-AD wrapper convention [6]: the pretrained MOMENT backbone is kept frozen and a lightweight reconstruction head is fine-tuned for 2 epochs on each target training split.

• SARAD (NeurIPS 2024) [41]: a spatial association-aware TSAD method that detects anomalies through reconstruction in association space.

• CATCH (ICLR 2025) [43]: a frequency-patching TSAD framework that reconstructs spectral patterns while modeling channel interactions.

• CrossAD (NeurIPS 2025) [14]: a multi-scale detector that models cross-scale associations and cross-window temporal context.

• OracleAD (NeurIPS 2025) [12]: a deviation-based detector that scores anomalies by measuring departure from learned normality.

## Score-Ensemble Baselines.

• Mean, Max, and Median: pointwise aggregation of normalized teacher scores using fixed reduction rules.

• Rank Fusion: conversion of each teacher score sequence into percentile ranks, followed by averaging of those ranks.

• RRF (SIGIR 2009) [44]: aggregation of teacher rankings through reciprocal-rank weighting.

## Zero-Shot Transfer Baselines.

• MOMENT (ICML 2024) [42]: evaluated in the zero-shot transfer experiment without fitting any component on the held-out target dataset.

• MOIRAI (ICML 2024) [47]: a universal pretrained forecasting model designed for broad transfer across domains and sampling regimes.

• TimesFM (ICML 2024) [45]: a decoder-style foundation model trained primarily for large-scale forecasting transfer.

• Lag-Llama (NeurIPS 2023 Workshop) [48]: a generative foundation model for probabilistic time-series forecasting.

• Chronos (TMLR 2024) [46]: a model that casts time-series forecasting into a languagemodeling-style token prediction problem.

• DADA (ICLR 2025) [49]: a TSAD-specific foundation model pretrained for zero-shot anomaly detection with adaptive bottlenecks and dual adversarial decoders.

Code Bases.

• TSB-AD: https://github.com/TheDatumOrg/TSB-AD

• CATCH: https://github.com/decisionintelligence/CATCH

• SARAD: https://github.com/daidahao/SARAD

• CrossAD: https://github.com/decisionintelligence/CrossAD

• MOIRAI: https://github.com/SalesforceAIResearch/uni2ts

• Lag-Llama: https://github.com/time-series-foundation-models/lag-llama

• DADA: https://github.com/iambowen/DADA

Shared Data and Evaluation Controls. We reran all ten target-dataset baselines and teachers on the same dataset files and fixed train/test splits; no published benchmark values were reused. Five-run comparisons use seeds {42, 123, 456, 789, 1024}, and training uses no anomaly labels. All scores use the common TSB-AD metric implementation with evaluation window 100 for range-based and VUS metrics, independently of the model’s input-window length.

Model-Specific Settings and Optimization. The TSB-AD implementations and local adapters use recommended baseline configurations without per-dataset hyperparameter searches. Table S5 summarizes the target-dataset benchmark configuration; windowing, normalization, epoch limits, and checkpoint restoration remain implementation-specific. Except for MOMENT’s fixed adaptation schedule, trainable baselines and teachers use the last 10% of the training split for validation and early stopping with patience 3. OmniAnomaly, TimesNet, SARAD, CATCH, and OracleAD share a training loop with at most 10 epochs, patience 3, and restoration of the lowest-validation-loss checkpoint. USAD, Anomaly Transformer, TranAD, and CrossAD use their wrappers’ stopping and checkpoint rules. For Table 1, MOMENT fine-tunes its reconstruction head for a fixed 2 epochs and uses the final checkpoint, without early stopping; its pretrained backbone remains frozen. In Table 4, MOMENT instead uses the pretrained checkpoint without any target-dataset fine-tuning. AnoMix uses the final checkpoint after 10 epochs; Appendix J gives its per-dataset settings and training-loss-based learning-rate choice.

Table S5: Baseline configurations for Table 1.
<table><tr><td>Model</td><td>Window</td><td>Learning rate</td><td>Stopping rule</td></tr><tr><td>OmniAnomaly</td><td>100</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td>Early stopping</td></tr><tr><td>USAD</td><td>100</td><td> $1 0 ^ { - 3 }$ </td><td>Early stopping</td></tr><tr><td>Anomaly Transformer</td><td>50</td><td>10⁻³</td><td>Early stopping</td></tr><tr><td>TranAD</td><td>10</td><td>10⁻3</td><td>Early stopping</td></tr><tr><td>TimesNet</td><td>96</td><td>10⁻⁴</td><td>Early stopping</td></tr><tr><td>SARAD</td><td>100</td><td>7×10⁻⁴</td><td>Early stopping</td></tr><tr><td>MOMENT</td><td>64</td><td>10⁻⁴</td><td>Fixed 2 epochs (head fine-tuning)</td></tr><tr><td>CATCH</td><td>100</td><td>10⁻⁴</td><td>Early stopping</td></tr><tr><td>CrossAD</td><td>192</td><td>10⁻4</td><td>Early stopping</td></tr><tr><td>OracleAD</td><td>10</td><td>10-3</td><td>Early stopping</td></tr></table>

Teacher Checkpoints and Student Seeds. For each dataset and teacher, a seed from the common seed set is fixed in advance, without performance-based seed selection. Its checkpoint supplies cached training scores, min–max normalized using training-score statistics and reused across five student seeds. Student variability in Appendix O is conditional on this fixed supervision and excludes teacher-retraining variability. Each standalone teacher row in Table 1 instead averages five teacher runs.

Fixed-Checkpoint Transfer Baselines. The six foundation baselines in Table 4 use fixed released checkpoints and model-specific preprocessing without target-dataset weight updates. Only AnoMix-ZS is trained on the source datasets with five seeds; Appendix O reports its standard deviations.

## I Teacher Sets and Robustness

## I.1 Teacher-Set Selection

The main teacher-set study considers three types of configuration. Sel-3 is the fixed benchmark teacher set used by AnoMix in the main experiments and consists of SARAD, CrossAD, and OracleAD. LOO Best-3 denotes leave-one-out teacher selection, where the target dataset is excluded and teachers are ranked only by their average performance on the remaining six datasets in the seven-task analysis. All-10 uses the entire teacher pool without selection. The purpose of this comparison is not to derive a universal teacher-selection rule, but to clarify what the fixed benchmark choice represents and how sensitive AnoMix is to alternative teacher pools.

The fixed Sel-3 configuration was chosen before evaluating AnoMix for analysis convenience and is intended as a compact and representative benchmark setting rather than a claimed optimum. It is formed by taking the three most recent state-of-the-art TSAD detectors in the teacher pool and therefore combines SARAD from association-space reconstruction, CrossAD from cross-scale association modeling, and OracleAD from deviation-based scoring. This choice keeps the trainingtime teacher pool small while still covering heterogeneous anomaly-scoring behavior. It also matches the design philosophy of AnoMix, which aims to distill comparative supervision from complementary teachers into a single deployable student.

The leave-one-out analysis is useful because it removes target-dataset information from teacher selection. The resulting per-target ranking is shown in Table S6. Under this protocol, SARAD and CrossAD occupy the top two positions on six of the seven target datasets, indicating that they are the most reliable cross-dataset teachers in the candidate pool. The third position is typically taken by OracleAD or CATCH. This pattern helps explain why the fixed benchmark three-teacher set remains competitive even when compared with teacher sets obtained from target-excluded ranking.

Table S6: LOO teacher ranking on the seven benchmark datasets. For each target dataset, teachers are ranked by average VUS-PR (%) on the other six datasets. Omni denotes OmniAnomaly, and A.T denotes Anomaly Transformer. Boldface indicates the top three teachers for each target dataset.
<table><tr><td>Rank</td><td>PSM</td><td>SWaT</td><td>Exathlon</td><td>SMD</td><td>SMAP</td><td>Daphnet</td><td>SWAN</td></tr><tr><td>1</td><td>SARAD</td><td>CrossAD</td><td>CrossAD</td><td>SARAD</td><td>CrossAD</td><td>CrossAD</td><td>SARAD</td></tr><tr><td>2</td><td>CrossAD</td><td>SARAD</td><td>SARAD</td><td>OracleAD</td><td>SARAD</td><td>SARAD</td><td>CrossAD</td></tr><tr><td>3</td><td>CATCH</td><td>CATCH</td><td>CATCH</td><td>CrossAD</td><td>CATCH</td><td>USAD</td><td>OracleAD</td></tr><tr><td>4</td><td>USAD</td><td>OracleAD</td><td>OracleAD</td><td>CATCH</td><td>USAD</td><td>Omni</td><td>CATCH</td></tr><tr><td>5</td><td>OracleAD</td><td>Omni</td><td>USAD</td><td>TranAD</td><td>Omni</td><td>OracleAD</td><td>USAD</td></tr><tr><td>6</td><td>Omni</td><td>USAD</td><td>Omni</td><td>USAD</td><td>OracleAD</td><td>CATCH</td><td>Omni</td></tr><tr><td>7</td><td>TranAD</td><td>TranAD</td><td>TranAD</td><td>Omni</td><td>TranAD</td><td>TranAD</td><td>TranAD</td></tr><tr><td>8</td><td>TimesNet</td><td>TimesNet</td><td>TimesNet</td><td>MOMENT</td><td>TimesNet</td><td>TimesNet</td><td>TimesNet</td></tr><tr><td>9</td><td>MOMENT</td><td>MOMENT</td><td>MOMENT</td><td>TimesNet</td><td>MOMENT</td><td>MOMENT</td><td>MOMENT</td></tr><tr><td>10</td><td>A.T</td><td>A.T</td><td>A.T</td><td>A.T</td><td>A.T</td><td>A.T</td><td>A.T</td></tr></table>

## I.2 Teacher-Pool Robustness

We stress-test teacher quality and score-sequence homogeneity on all seven datasets using five student seeds, with the same TCN, CoRank objective, and training protocol as the main experiment. HML-3 combines SARAD, OmniAnomaly, and Anomaly Transformer, spanning high, middle, and low performance tiers. Mid-3 contains USAD, OracleAD, and OmniAnomaly, while Bottom-3 contains TimesNet, MOMENT, and Anomaly Transformer. These pools remain fixed across datasets. Corr-3 selects the three-teacher subset with the highest mean pairwise Spearman correlation on each dataset’s unlabeled training-score sequences, using neither anomaly labels nor target test scores. For each pool, we compare AnoMix with the best constituent teacher and the normalized mean-score ensemble built from the same teacher checkpoints.

Table S7 reports macro averages over the seven datasets. The standard deviation is computed across the five seed-level macro averages. Mean Spearman measures pairwise teacher correlation on training scores, and pair support is the fraction of candidate training pairs satisfying ma $\mathrm { x } _ { m } \vert \Delta _ { i j } ^ { ( m ) } \vert > \delta$ . Gains are averaged against each dataset’s best constituent teacher and mean ensemble.

Table S7: Teacher-pool stress test. VUS-PR is reported as mean ± standard deviation (%), and gains are in %pt.
<table><tr><td>Teacher set</td><td>Mean Spearman</td><td></td><td></td><td></td><td>Pair support AnoMix VUS-PR vs. best teacher vs. mean ensemble</td></tr><tr><td>Sel-3</td><td>0.463</td><td>36.8%</td><td> ${ \bf 7 3 . 9 3 \pm 0 . 5 1 }$ </td><td>+15.49</td><td>+16.95</td></tr><tr><td>HML-3</td><td>0.248</td><td>27.1%</td><td> $6 6 . 1 7 \pm 1 . 2 7$ </td><td>+9.65</td><td>+14.11</td></tr><tr><td>Mid-3</td><td>0.607</td><td>33.9%</td><td> $6 8 . 9 4 \pm 0 . 5 7$ </td><td>+10.09</td><td>+15.15</td></tr><tr><td>Bottom-3</td><td>0.024</td><td>3.9%</td><td> $4 8 . 7 4 \pm 3 . 4 0$ </td><td>-2.17</td><td>-1.14</td></tr><tr><td>Corr-3</td><td>0.844</td><td>27.3%</td><td> $6 9 . 2 2 \pm 0 . 7 0$ </td><td>+14.67</td><td>+15.54</td></tr></table>

AnoMix exceeds both comparators with mixed-quality, medium-quality, and highly correlated pools. HML-3 improves on its best teacher by 9.65%pt despite combining detectors from different quality tiers. Bottom-3, however, provides only 3.9% pair support and falls below both comparators, exposing a failure case with weak teachers and sparse pair coverage. Sel-3 also provides broader pair support than Corr-3 and achieves higher student performance. This is consistent with complementary teachers supplying informative gaps on different pairs, although these comparisons do not isolate the effects of teacher quality and coverage. High rank correlation does not imply identical score gaps, so Corr-3 can still supply distinct pairwise signals.

Together with the target-excluded LOO analysis and All-10 control, these results broaden the evidence beyond the fixed Sel-3 pool. The tested pools are not an average over random subsets, and their success does not establish robustness to shared systematic teacher errors. Agreement based only on teacher outputs cannot distinguish unanimously correct preferences from unanimously incorrect ones, as discussed in Appendix Q.

## J Implementation Details

Every reported run uses the single-head TCN student described in Appendix B. The student has hidden dimension $H = 6 4$ , six causal blocks with dilations {1, 2, 4, 8, 16, 32}, kernel size $k = 3 .$ dropout 0.1, and under 80k parameters on the seven benchmark settings. Its receptive field is 127 time steps, and most datasets therefore use a window size of 127 in order to fully use the available temporal context. Student training uses only the training split and no anomaly labels. We train for 10 epochs with AdamW, weight decay $1 0 ^ { - \overline { { 4 } } }$ , cosine annealing, and a CoRank score-gap threshold $\delta { = } 0 . 1$ . Within each mini-batch, CoRank evaluates all unordered non-identical time-point pairs and excludes reverse duplicates. Student results use five random seeds under fixed data splits and fixed teacher-score caches, so their standard deviations measure student optimization variability conditional on the teacher supervision. The zero-shot foundation baselines use fixed released checkpoints rather than five independently trained models. Table S8 lists the remaining per-dataset choices.

We select per-dataset learning rates based on training-loss stability, choosing the rate at which the CoRank loss decreases gradually rather than collapsing to near zero within a few epochs. For SMD and Daphnet, the loss converges rapidly at the default $\mathrm { { 1 0 ^ { - 3 } } }$ , so we use a lower learning rate of $1 0 ^ { - 5 }$

Table S8: Per-dataset student hyperparameters.
<table><tr><td>Dataset</td><td>Learning Rate</td><td>Window Size</td><td>Batch Size</td></tr><tr><td>PSM</td><td> $1 0 ^ { - 3 }$ </td><td>127</td><td>64</td></tr><tr><td>SWaT</td><td> $1 0 ^ { - 3 }$ </td><td>127</td><td>64</td></tr><tr><td>Exathlon</td><td> $1 0 ^ { - 3 }$ </td><td>127</td><td>64</td></tr><tr><td>SMD</td><td> $1 0 ^ { - 5 }$ </td><td>64</td><td>64</td></tr><tr><td>SMAP</td><td> $1 0 ^ { - 3 }$ </td><td>127</td><td>64</td></tr><tr><td>Daphnet</td><td> $1 0 ^ { - 5 }$ </td><td>127</td><td>64</td></tr><tr><td>SWAN</td><td> $1 0 ^ { - 3 }$ </td><td>127</td><td>64</td></tr></table>

Figure S2 reports hyperparameter sensitivity with respect to window size $L ,$ score-gap threshold δ, and batch size. Window size and batch size are largely flat across the moderate values used in practice, but performance collapses at $L = 1 0 ,$ , where the receptive field is too short to reflect meaningful temporal context. The score-gap threshold shows the most informative pattern, with small positive values essentially tied at the optimum, while $\delta \geq 0 . 2$ drops sharply by roughly seven points. This indicates that the threshold acts as a meaningful filter on small teacher score gaps. Small positive thresholds retain informative comparisons while suppressing ambiguous pairs, whereas larger thresholds discard too many teacher-derived comparisons and noticeably weaken supervision.

The reported runs use the dataset-specific defaults in Table S8. For a new dataset, we recommend choosing within the near-optimal ranges $L \in [ 6 4 , 1 2 7 ] , \delta \in [ 0 . 0 5 , 0 . 1 ]$ , and bs $\in [ 3 2 , 6 4 ]$ , where each setting stays within roughly 1%pt of the macro optimum across the seven datasets.

![](images/79068290e00d020550a8312b64bd3c03ac6f780684e800b5a2d058caa3283119.jpg)

![](images/a642b6b066159ad5c1561a13beb2af8360ee431ae5a5cc5f9878a2e6a97e196f.jpg)

![](images/3a09602994017a2d04a23d6a383c39ade2f19ad64df870d778f6bc184ec1df2b.jpg)  
Figure S2: Macro-average hyperparameter sensitivity across the seven datasets. The line denotes the macro mean across datasets, and the shaded band denotes the ±1 standard deviation across five random seeds.

![](images/23ae3655ae95a368da03e3c96ad566c5694e4abe7bed385556b52afb84ddba39.jpg)  
Figure S3: AnoMix CoRank training loss over 10 epochs on each of the six benchmark datasets. The line denotes the five-seed mean, and the shaded region denotes the ±1 standard deviation across five random seeds.

## K Optimization

Figure S3 shows the per-epoch CoRank training loss for each of the original six benchmark datasets, averaged over five random seeds, with the shaded region denoting the ±1 standard deviation across five random seeds. All six curves decrease monotonically under the shared 10-epoch budget, and the across-seed band is narrow on every dataset. Some datasets show faster early decay than others, but the overall training behavior is stable across these tasks.

## L Model Complexity and Inference Efficiency

Table S9 compares model complexity, parameter count, peak GPU memory during single-sample inference, and FLOPs per forward pass on a standardized synthetic input $( L { \dot { = } } 1 0 0 , \bar { D = } 2 0 { \bar { 0 } } )$ . We use a common input size rather than a specific benchmark to isolate architectural cost from per-dataset feature count. Peak memory is measured with torch.cuda.max\_memory\_allocated after warmup. We estimate FLOPs with fvcore when tracing is supported and use ptflops estimates for LSTM modules that fvcore cannot trace. The complexity column drops fixed model hyperparameters such as hidden dimension and reports scaling only in the data-side quantities. AnoMix requires only 8.65M FLOPs per window and 9.59 MB of peak GPU memory at inference, corresponding to 127,000× fewer FLOPs than MOMENT and 93× lower peak memory than OracleAD.

These measurements describe deployment-time inference with the student alone. They exclude the offline costs of training or obtaining the teachers, generating and caching their training scores, and training the student. AnoMix therefore requires offline teacher computation; the reported inference savings do not imply lower end-to-end training cost than training a single detector.

Table S9: Model complexity and single-model inference cost on synthetic input $( L { = } 1 0 0 , D { = } 2 0 0$ batch=1), excluding offline training and teacher scoring. L=window, D=features, P=patch count.
<table><tr><td>Model</td><td>Complexity</td><td>Params</td><td>Memory (MB)</td><td>FLOPs</td></tr><tr><td>SARAD</td><td> $O ( L ^ { 2 } D )$ </td><td>9.66M</td><td>58.04</td><td>2.03G</td></tr><tr><td>CrossAD</td><td> $O \dot { ( } L ^ { 2 } D \dot { ) }$ </td><td>2.64M</td><td>178.49</td><td>8.81G</td></tr><tr><td>OracleAD</td><td> $O ( L D + D ^ { 2 } )$ </td><td>211.72M</td><td>893.94</td><td>21.21G</td></tr><tr><td>CATCH</td><td> $O ( P ^ { 2 } D )$ </td><td>13.52M</td><td>86.17</td><td>1.18G</td></tr><tr><td>Omni</td><td> $\dot { O ( } L D \dot { ) }$ </td><td>314k</td><td>27.55</td><td>31.64M</td></tr><tr><td>USAD</td><td> $O ( L D )$ </td><td>1.00M</td><td>13.82</td><td>1.60M</td></tr><tr><td>A.T</td><td> $O ( \dot { L } ^ { 2 } \dot { D } )$ </td><td>5.16M</td><td>43.10</td><td>545.33M</td></tr><tr><td>TranAD</td><td> $O ( L ^ { 2 } D )$ </td><td>3.33M</td><td>37.50</td><td>237.72M</td></tr><tr><td>TimesNet</td><td> $O ( L \log L + \dot { L } D )$ </td><td>80k</td><td>9.90</td><td>22.90M</td></tr><tr><td>MOMENT</td><td> $O ( L ^ { 2 } D )$ </td><td>109.64M</td><td>1035.50</td><td>1.10T</td></tr><tr><td>AnoMix</td><td>O(LD)</td><td>87k</td><td>9.59</td><td>8.65M</td></tr></table>

All reported efficiency measurements and student training runs were performed on a single local workstation. Table S10 summarizes the system configuration.

Table S10: System configuration used for the reported experiments.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>CPU</td><td>Intel Core Ultra 7 265K, 20 cores</td></tr><tr><td>GPU</td><td>NVIDIA GeForce RTX 5090, 32 GB memory</td></tr><tr><td>System memory</td><td>96 GB RAM</td></tr><tr><td>Operating system</td><td>Ubuntu 24.04.4 LTS, Linux 6.14</td></tr><tr><td>NVIDIA driver</td><td>570.153.02</td></tr><tr><td>Software stack</td><td>Python 3.13, PyTorch 2.7.0+cu128, CUDA runtime 12.8</td></tr></table>

## M Additional Ablations

![](images/dbd9f52256c8ff7320f3e78f2d215778f1fe9020824d023158092143280959de.jpg)  
(b) CoRank Margin Scheme

![](images/1fe58d3698e6c8970d5f4b8a915f3224a1d9585ef8cc64faa8284b778a61a55d.jpg)  
Figure S4: Ablation on teacher score normalization (a) and score-gap threshold selection (b). Each bar is the seven-dataset macro VUS-PR averaged over five seeds. Default settings are colored in dark blue.

This appendix reports two additional ablations on design choices that are fixed in the main experiments.   
Figure S4(a) and Figure S4(b) both report the seven-dataset macro VUS-PR averaged over five seeds.

Teacher Score Normalization. Figure S4(a) compares three ways of normalizing each teacher’s training scores before they enter the CoRank objective. Min-max maps each teacher’s training-set range to [0, 1]. Z-score normalizes by the training mean and standard deviation. Rank converts each score to its empirical CDF position in [0, 1]. Min-max attains the highest macro VUS-PR, with z-score and rank both within roughly one point.

Score-Gap Threshold Selection. CoRank uses the score-gap threshold δ to decide whether a time-point pair is informative enough for training. In the default setting, a pair $( i , j )$ is retained when at least one teacher has a normalized score gap larger than $\delta = 0 . 1$ , namely when max $_ m | \Delta _ { i j } ^ { ( m ) } | > \delta$ This shared threshold is simple, but it can be less appropriate if different teachers have very different score-gap distributions.

Figure S4(b) asks whether replacing the shared threshold with teacher-specific thresholds improves performance. We compare the default Fixed rule with two alternatives computed from training-set score gaps. Teacher Mean sets $\delta _ { m } = \mathbb { E } _ { i , j } [ | \Delta _ { i j } ^ { ( m ) } | ]$ , which filters out pairs below teacher m’s average separation. Quantile sets $\delta _ { m }$ to the $2 5 ^ { \mathrm { t h } }$ percentile of the same gap distribution, so that each teacher filters the same fraction of small-gap pairs. Both alternatives are parameter-free and use no anomaly labels.

On the seven-dataset macro VUS-PR, the fixed scalar matches or slightly exceeds both per-teacher alternatives. This suggests that, after min-max normalization, a shared δ = 0.1 is sufficient in our setting. Additional teacher-specific thresholding therefore provides little benefit in this experiment.

## N Teacher Alignment and Consensus Confidence

We examine where CoRank obtains supervision from the Sel-3 teachers and how teacher agreement relates to student performance. Teacher alignment is defined as $a _ { i j } = ( 1 + c _ { i j } ) / 2$ , ranging from 50% for balanced opposing score-gap magnitudes to 100% when all nonzero gaps share the same direction.

Training-Pair Distribution. We sample two million candidate pairs per dataset and apply the CoRank support rule with $\delta = 0 . 1$ . Table S11 reports confidence distributions conditional on retention. Most retained pairs have high confidence, while SWaT and SMD contain larger shares of conflicting comparisons.

Table S11: Confidence distribution among retained training pairs (%). Rows sum to 100% up to rounding.
<table><tr><td>Dataset</td><td> $c \in [ 0 , 0 . 2 )$ </td><td> $c \in [ 0 . 2 , 0 . 4 )$ </td><td> $c \in [ 0 . 4 , 0 . 6 )$ </td><td> $c \in [ 0 . 6 , 0 . 8 )$ </td><td> $c \in [ 0 . 8 , 1 ]$ </td></tr><tr><td>PSM</td><td>0.04</td><td>0.09</td><td>0.08</td><td>0.40</td><td>99.39</td></tr><tr><td>SWaT</td><td>4.06</td><td>5.16</td><td>7.36</td><td>11.65</td><td>71.76</td></tr><tr><td>Exathlon</td><td>0.01</td><td>0.02</td><td>0.02</td><td>0.05</td><td>99.91</td></tr><tr><td>SMD</td><td>3.72</td><td>4.52</td><td>6.14</td><td>9.85</td><td>75.77</td></tr><tr><td>SMAP</td><td>2.30</td><td>2.02</td><td>2.71</td><td>5.61</td><td>87.36</td></tr><tr><td>Daphnet</td><td>0.40</td><td>0.58</td><td>0.83</td><td>2.22</td><td>95.96</td></tr><tr><td>SWAN</td><td>0.18</td><td>0.41</td><td>1.24</td><td>4.33</td><td>93.84</td></tr></table>

Confidence-Filtered Training. We retrain AnoMix on SWaT with five seeds, keeping the architecture, pair support, optimizer, and training schedule fixed. For each cutoff, loss terms below that confidence receive zero weight, while the original loss denominator is unchanged. Full denotes the matched unfiltered run. Table S12 shows that every cutoff lowers mean VUS-ROC and VUS-PR relative to full CoRank. This supports retaining comparisons with small but nonzero confidence and weighting them softly. The VUS-PR trend is not monotonic, and the variation across seeds does not establish a separate contribution for each interval.

Table S12: Confidence-filtered training on SWaT (5-seed mean ± standard deviation, %).
<table><tr><td>Training confidence range</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><td>[0, 1] (Full)</td><td> $7 8 . 5 5 \pm 4 . 3 2$ </td><td> $4 2 . 3 9 \pm 3 . 3 4$ </td></tr><tr><td>[0.2, 1]</td><td> $7 5 . 9 3 \pm 5 . 7 4$ </td><td> $3 9 . 7 4 \pm 4 . 2 5$ </td></tr><tr><td>[0.4, 1]</td><td> $7 5 . 7 6 \pm 5 . 8 8$ </td><td> $4 1 . 3 9 \pm 4 . 9 4$ </td></tr><tr><td>[0.6, 1]</td><td> $7 3 . 8 4 \pm 6 . 1 0$ </td><td> $3 9 . 5 6 \pm 3 . 8 2$ </td></tr><tr><td>[0.8, 1]</td><td> $7 3 . 1 2 \pm 7 . 3 2$ </td><td> $4 0 . 7 9 \pm 5 . 5 6$ </td></tr></table>

Alignment and Pairwise Accuracy. We sample two million anomaly–normal test pairs per dataset, retain pairs under the same score-gap rule, and group them by teacher alignment using trainingstatistic score normalization. A preference is correct when the anomalous point receives the higher score, with half credit for ties. Table S13 compares the mean individual-teacher accuracy with the seed-42 student’s accuracy, macro-averaged over the seven datasets. AnoMix is more accurate in every interval, including those with strongly conflicting teacher gaps. This is a label-aware post-hoc diagnostic rather than a causal training analysis. Teacher test scores and test labels are used only for this diagnostic and never to train the student.

Table S13: Pairwise accuracy by teacher alignment (seven-dataset macro average, %).
<table><tr><td>Teacher alignment</td><td>Mean teacher</td><td>AnoMix</td></tr><tr><td>[50, 60)%</td><td>58.31</td><td>84.00</td></tr><tr><td>[60, 70)%</td><td>57.44</td><td>84.77</td></tr><tr><td>[70, 80)%</td><td>53.27</td><td>84.98</td></tr><tr><td>[80, 90)%</td><td>51.54</td><td>87.94</td></tr><tr><td>[90, 100]%</td><td>83.28</td><td>91.38</td></tr></table>

## O Statistical Significance

This appendix reports variability across the five random seeds {42, 123, 456, 789, 1024} for Tables 1– 4. Standard deviations use the population convention (divisor 5), matching the experiment summaries, and are reported in %pt. An entry of 0.00 may reflect rounding of a small nonzero deviation.

Table S14: Standard deviation across five random seeds (%) for the main results in Table 1.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td></tr><tr><td>|V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>|V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td></tr><tr><td>Omni [3]</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>USAD [9]</td><td>0.89</td><td>0.57</td><td>1.31</td><td>0.75</td><td>0.00</td><td>0.04</td><td>0.34</td><td>1.31</td><td>0.00</td><td>0.00</td><td>0.84</td><td>0.38</td><td>0.20</td><td>0.29</td></tr><tr><td>A.T [15]</td><td>4.04</td><td>2.59</td><td>10.16</td><td>6.40</td><td>6.57</td><td>6.97</td><td>2.63</td><td>1.03</td><td>7.06</td><td>4.02</td><td>7.63</td><td>1.77</td><td>3.05</td><td>0.96</td></tr><tr><td>TranAD [10]</td><td>0.59</td><td>0.19</td><td>1.25</td><td>0.78</td><td>0.01</td><td>0.36</td><td>0.15</td><td>0.17</td><td>11.31</td><td>3.95</td><td>0.34</td><td>0.25</td><td>0.18</td><td>0.28</td></tr><tr><td>TimesNet [11]</td><td>0.69</td><td>0.58</td><td>1.01</td><td>0.89</td><td>0.11</td><td>0.84</td><td>0.65</td><td>2.05</td><td>4.50</td><td>1.85</td><td>0.15</td><td>0.59</td><td>0.46</td><td>0.90</td></tr><tr><td>MOMENT [42]</td><td>0.13</td><td>0.01</td><td>0.30</td><td>0.24</td><td>0.00</td><td>0.01</td><td>0.59</td><td>0.13</td><td>0.90</td><td>0.48</td><td>0.01</td><td>0.06</td><td>0.04</td><td>0.08</td></tr><tr><td>SARAD [41]</td><td>0.87</td><td>0.70</td><td>1.05</td><td>0.28</td><td>0.02</td><td>0.26</td><td>0.02</td><td>0.10</td><td>0.34</td><td>0.25</td><td>0.61</td><td>0.89</td><td>0.52</td><td>0.95</td></tr><tr><td>CATCH [43]</td><td>0.16</td><td>0.30</td><td>0.03</td><td>0.01</td><td>0.01</td><td>0.11</td><td>0.01</td><td>0.11</td><td>0.00</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.03</td></tr><tr><td>CrossAD [14]</td><td>0.01</td><td>0.01</td><td>0.08</td><td>0.01</td><td>0.00</td><td>0.02</td><td>0.01</td><td>0.06</td><td>0.01</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.00</td><td>0.01</td></tr><tr><td>OracleAD [12]</td><td>2.43</td><td>2.09</td><td>6.69</td><td>3.50</td><td>0.34</td><td>2.97</td><td>1.65</td><td>3.76</td><td>2.32</td><td>4.53</td><td>1.59</td><td>5.60</td><td>1.46</td><td>2.92</td></tr><tr><td>AnoMix</td><td>1.27</td><td>2.14</td><td>4.33</td><td>3.34</td><td>0.00</td><td>0.02</td><td>0.42</td><td>1.29</td><td>0.05</td><td>0.06</td><td>0.28</td><td>1.66</td><td>0.51</td><td>0.67</td></tr></table>

Table S15: Standard deviation across five random seeds (%) for the score-ensemble comparison in Table 2.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td></tr><tr><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td></tr><tr><td>Mean</td><td>2.66</td><td>1.59</td><td>3.54</td><td>1.29</td><td>0.13</td><td>1.76</td><td>0.28</td><td>1.24</td><td>0.58</td><td>0.14</td><td>0.88</td><td>3.92</td><td>0.32</td><td>0.30</td></tr><tr><td>Max</td><td>3.91</td><td>2.46</td><td>6.60</td><td>2.80</td><td>0.18</td><td>2.74</td><td>0.49</td><td>1.42</td><td>0.41</td><td>0.08</td><td>1.46</td><td>3.67</td><td>0.52</td><td>0.14</td></tr><tr><td>Median</td><td>2.01</td><td>0.86</td><td>0.68</td><td>0.31</td><td>0.21</td><td>2.49</td><td>0.34</td><td>1.38</td><td>0.82</td><td>0.41</td><td>0.38</td><td>1.38</td><td>0.89</td><td>1.85</td></tr><tr><td>Rank Fusion</td><td>1.28</td><td>0.86</td><td>2.47</td><td>1.14</td><td>0.01</td><td>0.08</td><td>0.50</td><td>2.13</td><td>0.93</td><td>1.04</td><td>0.44</td><td>2.14</td><td>0.19</td><td>0.34</td></tr><tr><td>RRF [44]</td><td>1.33</td><td>1.45</td><td>2.48</td><td>1.30</td><td>0.12</td><td>1.17</td><td>0.15</td><td>1.32</td><td>1.55</td><td>1.51</td><td>0.77</td><td>2.82</td><td>0.37</td><td>0.91</td></tr><tr><td>AnoMix</td><td>1.27</td><td>2.14</td><td>4.33</td><td>3.34</td><td>0.00</td><td>0.02</td><td>0.42</td><td>1.29</td><td>0.05</td><td>0.06</td><td>0.28</td><td>1.66</td><td>0.51</td><td>0.67</td></tr></table>

Teacher-Set Variants. Table S16 reports per-dataset standard deviations for Table 3, with teacher supervision fixed across student seeds. SWAN LOO-3 and Sel-3 use the same teachers and student runs.

Table S16: Per-dataset standard deviations (%pt) for the teacher sets in Table 3.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td></tr><tr><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td><td>V-ROC</td><td>V-PR</td></tr><tr><td>L00-1</td><td>6.07</td><td>4.27</td><td>22.60</td><td>10.70</td><td>39.37</td><td>36.16</td><td>34.56</td><td>33.08</td><td>39.45</td><td>36.99</td><td>29.93</td><td>13.42</td><td>3.05</td><td>5.10</td></tr><tr><td>L00-2</td><td>2.41</td><td>3.93</td><td>2.14</td><td>2.02</td><td>0.01</td><td>0.03</td><td>0.42</td><td>0.76</td><td>0.06</td><td>0.06</td><td>0.28</td><td>1.76</td><td>0.18</td><td>0.33</td></tr><tr><td>L00-3</td><td>6.25</td><td>5.11</td><td>3.10</td><td>4.75</td><td>0.01</td><td>0.04</td><td>0.42</td><td>1.29</td><td>0.04</td><td>0.05</td><td>0.38</td><td>2.01</td><td>0.51</td><td>0.67</td></tr><tr><td>L00-4</td><td>3.11</td><td>2.55</td><td>3.16</td><td>3.35</td><td>0.00</td><td>0.02</td><td>0.38</td><td>0.99</td><td>0.04</td><td>0.04</td><td>0.38</td><td>1.99</td><td>0.43</td><td>0.80</td></tr><tr><td>L00-5</td><td>3.86</td><td>2.53</td><td>7.15</td><td>5.06</td><td>0.01</td><td>0.04</td><td>0.38</td><td>0.99</td><td>0.04</td><td>0.04</td><td>0.38</td><td>2.01</td><td>0.27</td><td>0.84</td></tr><tr><td>Sel-3</td><td>1.27</td><td>2.14</td><td>4.33</td><td>3.34</td><td>0.00</td><td>0.02</td><td>0.42</td><td>1.29</td><td>0.05</td><td>0.06</td><td>0.28</td><td>1.66</td><td>0.51</td><td>0.67</td></tr><tr><td>All-10</td><td>2.33</td><td>1.74</td><td>3.35</td><td>2.07</td><td>0.00</td><td>0.02</td><td>0.32</td><td>1.14</td><td>0.03</td><td>0.05</td><td>0.37</td><td>2.01</td><td>0.55</td><td>0.74</td></tr></table>

Zero-Shot Transfer. Table S17 reports per-dataset standard deviations for AnoMix-ZS in Table 4. Fixed-checkpoint foundation baselines have no across-training-seed standard deviation.

Table S17: Per-dataset standard deviations (%pt) for AnoMix-ZS in Table 4.
<table><tr><td rowspan="2">Method</td><td colspan="2">PSM</td><td colspan="2">SWaT</td><td colspan="2">Exathlon</td><td colspan="2">SMD</td><td colspan="2">SMAP</td><td colspan="2">Daphnet</td><td colspan="2">SWAN</td></tr><tr><td colspan="2">V-ROC V-PR</td><td colspan="2">V-ROC V-PR</td><td colspan="2">V-ROC V-PR</td><td colspan="2">V-ROC V-PR</td><td colspan="2">V-ROC V-PR</td><td colspan="2">V-ROC V-PR</td><td colspan="2">V-ROC V-PR</td></tr><tr><td>AnoMix-ZS</td><td>1.45</td><td>1.45</td><td>1.46</td><td>2.17</td><td>0.08</td><td>1.38</td><td>0.47</td><td>4.72</td><td>0.75</td><td>2.30</td><td>1.00</td><td>4.63</td><td>0.25</td><td>0.53</td></tr></table>

## P Broader Impact

AnoMix can reduce the practical cost of deploying strong time-series anomaly detection by replacing multi-model teacher ensembles with a single lightweight student. This is especially useful in monitoring settings where inference latency, memory footprint, and maintenance cost matter, such as industrial sensor systems, cyber-physical infrastructure, and edge devices. This makes the framework relevant to on-device AI for self-diagnosis in robots, satellites, and manufacturing systems, where local anomaly detection can support earlier fault detection without continuously running large detector ensembles. By distilling complementary teacher scoring principles into one deployable model, AnoMix can make stronger anomaly detection more accessible in environments where running several detectors in parallel is difficult.

The framework also offers a reusable path for building compact TSAD models from existing detectors. Many application domains already have specialized unsupervised detectors, each capturing different abnormality criteria. AnoMix provides a way to consolidate these signals without requiring anomaly labels, which are often expensive, delayed, or incomplete in real systems. This can support faster deployment cycles and lower the barrier to applying multi-detector anomaly detection in domains with limited labeling resources.

As with any anomaly detector, AnoMix can produce false positives and false negatives, and the impact of these errors depends on the deployment domain. False positives may increase inspection cost or operator fatigue, while false negatives may delay responses to emerging faults. These risks should be addressed through domain-specific validation, threshold calibration, human oversight, and continuous monitoring after deployment. Nevertheless, by improving deployment efficiency while preserving multiple teacher scoring principles, we believe that AnoMix can have a positive impact on industry from an application perspective.

## Q Limitations and Future Work

In this paper, we proposed AnoMix, a new TSAD framework for label-free multi-teacher distillation. Although the experimental results support the proposed methodology, several limitations remain. First, if the teacher pool is weak on the target domain, noisy teacher score sequences can pass unreliable supervision to the student. If several teachers share a correlated failure mode, their score gaps may also align on an incorrect pairwise preference, causing CoRank to assign high confidence to an unreliable consensus. Second, if the training split is too clean or teacher scores are nearly uniform, only a small number of meaningful score gaps may be available, limiting pairwise anomaly-preference supervision. Future work could address these issues through automatic teacher-pool construction, label-free TSAD model-selection criteria [54], diversity-aware teacher selection, source-data selection strategies, and synthetic anomaly injection that enriches the training data with stronger pairwise training signals.