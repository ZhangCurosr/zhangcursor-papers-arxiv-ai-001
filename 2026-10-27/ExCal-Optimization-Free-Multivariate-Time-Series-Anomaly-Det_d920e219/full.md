# ExCal: Optimization-Free Multivariate Time Series Anomaly Detection via Excess Calibration

Kadir-Kaan Ozer <sup>¨</sup> <sup>∗†</sup>, Rene Ebeling ´ <sup>∗</sup>, Markus Enzweiler<sup>†</sup>

Abstract—We introduce EXCAL, a normal-only multivariate time series anomaly detector that combines raw state retrieval, a signal/derivative subspace, and robust statistics across temporal scales. Each expert map receives a high normality floor from the training prefix. During scoring, positive excess is retained, attenuated for broadly firing maps, and combined with a small rank term. The method fits reference memories and projections without gradient optimization or anomaly labels. Prevalence and ranks depend on the complete scored sequence, making inference batch and transductive rather than online. On 180 series from 17 TSB-AD-M families, EXCAL obtains AUC-PR 0.421 ± 0.37 and VUS-PR 0.452 ± 0.37 over three seeds, ranking second among 24 methods. Holm-adjusted paired tests detect no significant difference from AxonAD or StreamVAE, which are the two leading non-pretrained methods on TSB-AD-M, on either primary metric, while finding significant positive differences against the other 21 baselines. A separate single-run probe over three series measures median model-wrapper time of 0.0662 s. Ablations associate the largest primary-metric changes with raw retrieval and excess-magnitude preservation. The evaluated score is decomposable by expert, but its diagnostic utility and online suitability remain untested.

Index Terms—multivariate time series, nearest-neighbor retrieval, robust statistics, self-calibration, subspace residuals, timeseries anomaly detection, training-free detection

## I. INTRODUCTION

Multivariate telemetry may deviate through unusual operating points, broken cross-channel relations, changes in local dynamics, or events visible only at particular durations. With only a normal training prefix, no single representation is assured to expose all of these departures. Neural forecasting, reconstruction, and attention methods offer expressive alternatives [1]–[8], but require optimization and architecture choices. Classical approaches offer transparent normality tests based on isolation, density, projection, or retrieval [9]–[16]. Combining these heterogeneous signals remains a calibration problem because their score scales differ.

We propose EXCAL (Excess Calibration), which derives temporal scales and normality references from the prefix, evaluates absolute raw state, derivative-aware subspace, and multiscale statistic maps, and combines their standardized excess above expert-specific floors. Prevalence damping attenuates maps firing broadly over the scored sequence, and a small rank term preserves sub-floor ordering. An optional context-centered shape branch is evaluated but excluded from the base configuration. Fig. 1 illustrates map calibration and the resulting score, while Fig. 2 details the architecture.

![](images/8db7e7771b0d9436b73bccaaadaba581a729f56e639cf54dfd30771016f8fd99.jpg)  
Fig. 1. Schematic expert-map responses to three distinct event motifs. (A) Sustained raw-retrieval excess, (B) localized subspace excess, and (C) intermittent statistics excess. Dashed lines denote map-specific normalit floors, and the bottom trace illustrates their possible combination. These are not measured responses and do not establish expert specificity or diagnostic utility.

EXCAL is optimization-free: it performs no gradient updates or neural learning, but fits PCA projections, reference banks, and quantiles. The primitives are established. The contribution lies in their explicit excess-preserving calibration and fusion, evaluated using TSB-AD-M [17], range-aware metrics [18], ablations, paired tests, seed repeats, and a small timing probe. The study does not isolate causal component effects or establish generalization beyond this benchmark.

## II. RELATED WORK

## A. Learned temporal detectors

Forecasting and reconstruction methods infer anomaly evidence from the discrepancy between an observed sequence and a learned normal model. LSTM predictors use future prediction error [1]. Donut models seasonal web-service KPIs with a variational autoencoder [2], whereas OmniAnomaly and sequential variational models represent stochastic dynamics and temporal smoothness [3], [4]. USAD combines autoencoding with adversarial training [5]. TranAD uses transformer encoders, self-conditioning, and adversarial learning for multivariate detection [6].

Attention-based methods explicitly model temporal interactions. Anomaly Transformer uses association discrepancy to distinguish anomalous points [7]. TimesNet discovers multiple periods and maps one-dimensional variation into a two-dimensional representation [8]. AxonAD models causal attention-query dynamics [19], and STREAM-VAE separates slow and fast telemetry dynamics with dual-path variational routing [20]. EXCAL shares the objective of covering multiple temporal mechanisms, but replaces end-to-end learned fusion with explicit normality tests and calibration on the training prefix.

![](images/82e4e2a62bbb9727f0c098292199cb4bcb424132f22ae664bc81743d752e78cc.jpg)  
Fig. 2. The EXCAL pipeline. The normal prefix supplies normalization, a temporal scale from the data, exemplar memories, a cross-channel subspace, robust statistic references, and per-map normality floors. The dashed branch is implemented and evaluated as an ablation but is disabled in the current base model The diagram describes the implemented data flow. It does not establish independent branch effects or guarantee a particular detection outcome.

## B. Classical and representation-based detectors

Isolation-based, density-based, histogram-based, and copula-based detectors offer simple alternatives to iterative neural training [9]–[13]. Robust principal component analysis (PCA) provides a low-rank model that tolerates sparse corruption [14]. K-means and nearest-neighbor constructions supply additional geometric views of normal data [15], [21]. PyOD exposes many such detectors through a common interface [22]. EXCAL follows the same transparent geometric spirit but applies its references to overlapping temporal windows and explicitly combines absolute state, derivatives, and local statistics.

## C. Evaluation of time-series anomalies

The TSB-AD benchmark emphasizes that aggregate anomaly-detection results depend on dataset heterogeneity, class imbalance, and the evaluation protocol [17]. Area under the precision–recall curve (AUC-PR) and area under the receiver operating characteristic curve (AUC-ROC) evaluate score ordering without selecting one threshold, while volume under the surface for precision–recall (VUS-PR) and receiver operating characteristic (VUS-ROC) extend this idea to rangeaware behavior across a volume of parameters [18]. We therefore treat AUC-PR and VUS-PR as the primary metrics and use the remaining reported measures as complementary diagnostics.

## III. EXCAL

## A. Problem formulation and prefix normalization

Let a multivariate series be

$$
\mathbf { X } = [ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { L } ] ^ { \mathsf { T } } \in \mathbb { R } ^ { L \times C } ,\tag{1}
$$

where L is the sequence length and C is the number of channels. The detector receives a normal training prefix $\mathbf { X } _ { \mathrm { t r } } =$ $\mathbf { X } _ { 1 : L _ { \mathrm { t r } } }$ and scores the full sequence. Labels are not used when fitting the detector.

Each channel is normalized with statistics computed only on the training prefix:

$$
z _ { t , c } = \frac { x _ { t , c } - \mu _ { c } } { \mathrm { m a x } ( \sigma _ { c } , 1 0 ^ { - 3 } ) } , \qquad \mu _ { c } = \mathrm { m e a n } _ { t \le L _ { \mathrm { t r } } } x _ { t , c } .\tag{2}
$$

This prevents the scored sequence from redefining the scale against which the normality references are calibrated. The standard-deviation floor keeps a training-constant channel numerically finite while allowing a subsequent movement to produce a large standardized deviation.

## B. Temporal scale from the data

The dominant period is estimated from the mean channel autocorrelation of the first min $( L _ { \mathrm { t r } } , 2 0 , 0 0 0 )$ training points. Each centered channel is transformed with a real fast Fourier transform (FFT), the inverse power spectrum produces an autocorrelation estimate, and the channel estimates are averaged. The implementation searches local maxima at lags 4 through min $( 4 0 0 , L _ { \mathrm { t r } } / 3 )$ and accepts the candidate with the largest autocorrelation only when that autocorrelation exceeds 0.1.

When a period $p$ is found, the implementation derives

$$
\begin{array} { l } { { W = \mathrm { c l i p } ( \lfloor 2 . 5 p \rfloor , 3 2 , 5 1 2 ) , ~ H = \mathrm { c l i p } ( \lfloor p / 2 \rfloor , 4 , 6 4 ) , } } \\ { { R = \mathrm { c l i p } ( p , 1 6 , 1 2 8 ) . } } \end{array}\tag{3}
$$

Here W and H define the optional context-centered shape expert, while R defines the longer raw state memory in the base model. If no candidate exceeds the autocorrelation threshold, the fallback is $( W , H , R ) = ( 1 0 0 , 8 , 3 2 )$ . The horizon is shortened when the training prefix is too short. The policy is fixed, but its effective temporal scale is derived from each normal series.

## C. Raw state retrieval

The first active expert tests whether a local joint state resembles a normal state in absolute coordinates. For window length r,

$$
\mathbf { u } _ { t } ^ { ( r ) } = \mathrm { v e c } ( \mathbf { z } _ { t - r + 1 : t } ) \in \mathbb { R } ^ { r C } .\tag{4}
$$

The windows are projected onto a PCA basis retaining at least 95% of sampled training window variance, and u¯<sub>r</sub> in Eq. (5) is the mean of those sampled training-prefix windows. It is a global training reference, not a per-window local centering. The projection reduces the distance computation. Because the raw windows are not centered by their local mean, the representation retains context level. For each query window, the nearest Euclidean distance to a sampled normal exemplar bank is computed. The window score is mapped back to points by taking the maximum over windows covering each point:

$$
m _ { \mathrm { r a w } } ^ { ( r ) } ( t ) = \operatorname* { m a x } _ { \substack { j : t \in [ j , j + r - 1 ] } } \operatorname* { m i n } _ { \bf e \in \mathscr { E } _ { r } } \left\| \mathbf { V } _ { r } ^ { \top } \left( \mathbf { u } _ { j } ^ { ( r ) } - \bar { \mathbf { u } } _ { r } \right) - \mathbf { e } \right\| _ { 2 } .\tag{5}
$$

Two banks are used in the base model: a short window of length 8 and the period-based window R. The short bank is intended to retain brief events, whereas the longer bank represents joint operating states over a broader cycle. During normality fitting, temporally overlapping self-matches are excluded whenever a non-overlapping exemplar exists.

## D. Optional shape retrieval with context centering

For an additional analysis, the implementation also retains a context-centered shape expert, which is disabled in the default configuration. It centers a context and its future by the context mean, projects the context onto a 95%-energy PCA basis, retrieves the three nearest normal contexts by cosine similarity, and averages their stored futures. The continuation error is normalized per channel and aggregated from the three largest channel values. Because centering removes local level, this branch is intended to represent continuation shape rather than absolute operating point. Its ablation is reported in Section V.

## E. Signal/derivative subspace

The implementation represents both values and first differences. EXCAL therefore augments every standardized point with a first difference:

$$
\mathbf { a } _ { t } = [ \mathbf { z } _ { t } , \Delta \mathbf { z } _ { t } ] \in \mathbb { R } ^ { 2 C } , \qquad \Delta \mathbf { z } _ { t } = \mathbf { z } _ { t } - \mathbf { z } _ { t - 1 } .\tag{6}
$$

The covariance of the training vectors is eigendecomposed. The smallest leading basis retaining 90% of the variance, capped at dimension 32, defines the normal subspace. The residual is

$$
{ \bf r } _ { t } = \left| ( { \bf a } _ { t } - { \pmb \mu } _ { a } ) - { \bf V } _ { s } { \bf V } _ { s } ^ { \top } ( { \bf a } _ { t } - { \pmb \mu } _ { a } ) \right| .\tag{7}
$$

Each residual coordinate is robustly standardized with its training median and interquartile range. The map is the mean of the three largest standardized residual coordinates to retain the contribution of a small number of large residuals.

## F. Multi-scale robust statistics

The statistics expert measures local level and variability changes at scales $s \in \{ 4 , 1 6 , 6 4 , 2 5 6 \}$ , subject to the available training length. Centered rolling means $\mu _ { t , c } ^ { ( s ) }$ and standard

deviations $\sigma _ { t , c } ^ { ( s ) }$ are compared with training medians and interquartile ranges:

$$
\begin{array} { r l } & { d _ { \mathrm { m e a n } , c } ^ { ( s ) } ( t ) = \frac { \left| \mu _ { t , c } ^ { ( s ) } - q _ { \mathrm { m e a n } , c , 0 . 5 } ^ { ( s ) } \right| } { \operatorname* { m a x } ( q _ { \mathrm { m e a n } , c , 0 . 7 5 } ^ { ( s ) } - q _ { \mathrm { m e a n } , c , 0 . 2 5 } ^ { ( s ) } , 0 . 0 5 ) } , } \\ & { d _ { \mathrm { s c a l e } , c } ^ { ( s ) } ( t ) = \frac { \left| \log ( \sigma _ { t , c } ^ { ( s ) } + 1 0 ^ { - 3 } ) - q _ { \mathrm { l o g } \sigma , c , 0 . 5 } ^ { ( s ) } \right| } { \operatorname* { m a x } ( q _ { \mathrm { l o g } \sigma , c , 0 . 7 5 } ^ { ( s ) } - q _ { \mathrm { l o g } \sigma , c , 0 . 2 5 } ^ { ( s ) } , 0 . 0 5 ) } . } \end{array}\tag{8}
$$

(9)

The absolute log-scale deviation represents both variance explosions and variance collapses. The mean of the three largest channel values retains the contribution of a small number of affected channels instead of averaging across all coordinates.

## G. Normality floors and excess-preserving fusion

Let $m _ { i } ( t )$ denote one map in the expert bank. During fitting, the map is evaluated on the training sequence itself. Its median and 99.5th percentile define a map-specific normality reference. At inference, only excess over the high floor is retained:

$$
e _ { i } ( t ) = \left[ \frac { m _ { i } ( t ) - q _ { i , 0 . 9 9 5 } } { \operatorname* { m a x } ( q _ { i , 0 . 9 9 5 } - q _ { i , 0 . 5 } , 1 0 ^ { - 6 } ) } \right] _ { + } .\tag{10}
$$

An observation that is ordinary for an expert contributes zero calibrated excess for that expert. This expresses different distance, residual, and statistic scales relative to their trainingprefix upper spread. The floor does not guarantee a 0.5% falsepositive rate on the scored sequence.

Let $f _ { i }$ be the fraction of points of the scored sequence with positive excess for map i. If a map fires broadly, its excess may reflect a broken reference or distribution shift rather than sparse anomalies. The implementation therefore applies

$$
h _ { i } = \sqrt { \frac { \rho } { \operatorname* { m a x } ( f _ { i } , \rho ) } } , \qquad \rho = 0 . 1 0 .\tag{11}
$$

Finally, a small normalized rank term preserves ordering among sub-floor points:

$$
s ( t ) = \sum _ { i } \left[ h _ { i } e _ { i } ( t ) + \tau \ \mathrm { r a n k 0 1 } ( m _ { i } ( t ) ) \right] , \qquad \tau = 0 . 0 2 .\tag{12}
$$

Here rank01 $\left( m _ { i } ( t ) \right)$ denotes the average rank of $m _ { i } ( t )$ among all points of the scored sequence, rescaled to [0, 1] and computed separately for each map. The rank term supplies ordering information below the floor and does not replace excess magnitude.

Both $f _ { i }$ and rank01 are statistics of the complete scored sequence, which under the benchmark protocol includes the training prefix. The final score is therefore a batch, transductive quantity rather than an online or causal one: the score at time t depends on the distribution of map values over the entire sequence being scored, and identical points embedded in different scored sequences can receive different scores. All normal-reference quantities (normalization statistics, temporal scale, exemplar banks, subspace, statistic references, and floors) remain prefix-only. Because the prevalence and

FITTING AND SCORING PROCEDURE FOR THE CURRENT EXCAL MODEL.

1) Normalize $\mathbf { X } _ { \mathrm { t r } }$ and the scored sequence using the training prefix statistics.

2) Estimate the dominant period and derive the short and period-scale windows.

3) Build raw exemplar memories, the $[ \mathbf { x } , \Delta \mathbf { x } ]$ subspace, and multi-scale statistic references.

4) If enabled, build the centered-context memory and replay futures for the optional shape expert.

5) Evaluate every active map on $\mathbf { X } _ { \mathrm { t r } } ,$ suppress overlapping memory self-matches, and store its median and q<sub>0.995</sub> floor.

6) Evaluate the same maps on the scored sequence, convert map values to calibrated excess, and compute the per-map prevalence factor over the scored sequence.

7) Sum damped excess and sub-floor rank terms, with ranks taken over the scored sequence, to return one score per time point.

rank terms read only the unlabeled input sequence, and because every semisupervised method in the benchmark scores the same complete sequence, this is a statement about the deployment form of the detector rather than label leakage in the comparison. What this study does not evaluate is a prefix-only or causal variant of the prevalence factor and rank normalization. Such a variant is a separate design point.

## H. Fitting and scoring procedure

The complete fitting and scoring path is summarized in Table I. All normal-reference quantities are constructed from the training prefix before scoring. The same expert-map code is called for the training and scored sequences. The training-map construction additionally excludes overlapping self-matches. The prevalence factors and rank normalization are then computed over the complete scored sequence, so the returned scores are batch quantities rather than prefix-only or online scores.

## I. Expert score decomposition

The final score is additive over maps, which makes an individual alert inspectable without introducing a second explanation model. For a point t, the contribution of map i is

$$
s _ { i } ( t ) = h _ { i } e _ { i } ( t ) + \tau \ \mathrm { r a n k } 0 1 ( m _ { i } ( t ) ) , \qquad s ( t ) = \sum _ { i } s _ { i } ( t ) .\tag{13}
$$

The map tag identifies whether the evidence came from the short raw bank, the period-scale raw bank, the derivativeaware subspace, a rolling mean or log-scale statistic, or the optional shape branch. The floor and spread used to produce $e _ { i } ( t )$ are also available from the fitted normal reference. The decomposition identifies the normality references associated with a score. It does not establish physical root causes, diagnostic accuracy, or improvements in human operator decisions. Channel-level and domain-level diagnosis remains a separate task.

FIXED IMPLEMENTATION CONSTANTS IN THE SUPPLIED EXCAL CONFIGURATION.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Exemplar and memory cap</td><td>10,000 windows</td></tr><tr><td>PCA sample cap</td><td>4,096 windows</td></tr><tr><td>Distance chunk</td><td>1,024 queries</td></tr><tr><td>Nearest neighbors</td><td> $k = 3$ </td></tr><tr><td>Top coordinates/channels</td><td>3</td></tr><tr><td>Normality floor</td><td>q0.995</td></tr><tr><td>Prevalence knee</td><td> $\rho = 0 . 1 0$ </td></tr><tr><td>Sub-floor rank weight</td><td> $\tau = 0 . 0 2$ </td></tr><tr><td>PCA energy</td><td>95%</td></tr><tr><td>Subspace energy</td><td>90%</td></tr><tr><td>Statistic scales</td><td>{4, 16, 64, 256}</td></tr></table>

## J. Computational profile

The CPU implementation fits FFT-based periods, rolling statistics, PCA and subspaces, and chunked nearest-neighbor banks without backpropagation or an optimizer. Caps in Table II limit reference and query batches, not total latency. Scoring is batch-oriented because prevalence and rank normalization use the full scored sequence. Online use would require prefix-only alternatives, bounded reference updates, and contamination handling. The timing probe in Section IV covers only the current batch implementation.

## IV. EXPERIMENTAL METHODOLOGY

## A. Benchmark and model set

We use the official M/eval split of TSB-AD-M. The target contains 180 multivariate series from 17 dataset families [17]. The benchmark run contains 24 methods: eight unsupervised detectors (IForest, LOF, HBOS, KNN, KMeansAD, COPOD, EIF, and RobustPCA) and sixteen methods in the semisupervised pool (Left STAMPi, CNN, LSTMAD, TranAD, USAD, OmniAnomaly, PatchTST, AnomalyTransformer, TimesNet, FITS, Donut, M2N2, StreamVAE, PaAno PAI, AxonAD, and ExCal). The method names are retained exactly as serialized by the evaluation code.

For semisupervised methods, the training prefix is parsed from the dataset filename, the model is fitted on that prefix, and the complete sequence is scored. ExCal follows this path: only $\mathbf { X } _ { \mathrm { t r } }$ determines its normalization, temporal scale, memories, subspace, statistic references, and normality floors. Its prevalence factors and rank terms are computed over the complete scored sequence, which includes the prefix, so its scores are transductive batch quantities. The benchmark labels are used only by the metric functions after scores have been generated.

## B. Evaluation protocol and metrics

The supplied evaluation code loads each CSV after removing missing rows, uses all feature columns except the final label column, derives the metric sliding window with the benchmark’s window-length routine at rank one on the first channel, and computes the nine reported measures:

• area under the precision–recall and receiver operating characteristic curves (AUC-PR and AUC-ROC).

• volume under the surface for precision–recall and receiver operating characteristic (VUS-PR and VUS-ROC).

• standard F1 score (Standard-F1), point-adjusted F1 (PA-F1), event-based F1 (Event-F1), range-based F1 (R-F1), and affiliation F-score (Aff-F).

The VUS measures follow the benchmark metric implementation and complement conventional AUC measures for range anomalies [18]. The F1-style values are threshold-dependent outputs of the supplied metric routine. They should not be interpreted as performance at a separately selected deployment threshold. We report arithmetic means and standard deviations over valid series-level results rather than pooling point-level predictions.

## C. Seeds, hardware, and aggregation

The benchmark uses seeds 2024, 2025, and 2026 on an Apple M3 Max MacBook Pro with 36 GB of memory. Results pool $1 8 0 \times 3 = 5 4 0$ series–seed observations for ExCal, with the same aggregation for baselines. Standard deviations reflect series heterogeneity, not confidence intervals.

All 24 methods, including every reported baseline, were rerun with the current TSB-AD evaluation code under the official split, metrics, and three-seed protocol, because the benchmark code has changed since the original repository and paper release. The ExCal row in that common rerun predates the change to the current default, in which the shape expert is disabled, so ExCal was rerun once more with the current implementation and only its row was replaced. No baseline score from the common rerun was changed. The paired statistical inputs apply the same replacement only to the ExCal column of the per-dataset matrix averaged over three seeds. This keeps the paired comparisons aligned with the implementation evaluated in this paper.

## D. Implementation and configuration protocol

The constants in Table II were shared across series, seeds, and variants except when a mechanism was disabled. All series-specific parameters derive from the normal prefix without labels. Aggregates, tests, and ablations use retained outputs from the supplied TSB-AD evaluation and metric code.

## E. Runtime benchmark

Runtime measurements were produced by the supplied benchmark timing runner. The runner evaluated all 24 benchmark methods on the first three automatically sorted TSB-AD-M files: one Genesis series with 16,220 rows and 18 channels, and two MSL series with 1,827 and 3,535 rows and 55 channels. The timing seed was 2024, with no warm-up and one measured repetition per method–series pair. The timed region starts immediately before the model-wrapper call and ends after device synchronization. CSV loading, garbage collection, cache clearing, and metric computation are excluded.

For each completed pair, the runner reports elapsed seconds and derives rows per second as the number of scored rows divided by the elapsed time. Section V-B reports the median over successful cases for each method. The median is used because the three cases have different lengths and model startup costs. It is a descriptive comparison on this fixed sample, not an estimate of asymptotic complexity. The probe produced 68 successful and 4 failed records. FITS failed on all three cases because the MPS backend rejected complex-valued linear input. LSTMAD failed on the shorter MSL case. Failed records are retained as failures rather than assigned artificial timings.

## F. Statistical tests

For each metric, we average the three seed results per dataset and perform paired comparisons over datasets. The omnibus test is Friedman’s test across the 24 methods, with Kendall’s W as the effect-size measure. We then compare ExCal with each of the 23 baselines using the two-sided Wilcoxon signed-rank test after removing zero differences. The 23 resulting p-values per metric are adjusted with Holm’s step-down procedure at $\alpha \ = \ 0 . 0 5$ . Affiliation-F has 175 complete datasets in the omnibus comparison because of missing baseline values. The remaining primary metric tests use all 180 datasets.

## V. RESULTS

## A. Main benchmark results

Table III reports results for 24 methods. ExCal in its base configuration obtains AUC-PR 0.421 ± 0.37 and VUS-PR $0 . 4 5 2 \pm 0 . 3 7$ , ranking second on both primary metrics behind AxonAD. It also ranks fourth on AUC-ROC and VUS-ROC, second on Standard-F1 and R-based-F1, and fifth on Affiliation-F. Point-adjusted and event-based F1 produce a different ordering. M2N2 has the highest PA-F1 and CNN has the highest Event-F1 among the listed methods.

Relative to AxonAD, ExCal is lower by 0.031 AUC-PR and 0.041 VUS-PR in the means pooled across seeds. It is higher than StreamVAE by 0.038 and 0.027, higher than PaAno PAI by 0.022 and 0.047, and higher than USAD by 0.058 and 0.049 on AUC-PR and VUS-PR, respectively. The standard deviations across series indicate that these averages combine different dataset behaviors rather than uniform perseries differences.

## B. Runtime and throughput

Fig. 3 reports the per-method results. The three measured ExCal times were 0.2010 s for Genesis, 0.0486 s for the shorter MSL series, and 0.0662 s for the second MSL series. The median was therefore 0.0662 s (66.2 ms), the mean was 0.1052 s, and the median throughput was 53,428 rows/s, or $2 . 0 7 \times 1 0 ^ { 6 }$ values/s. ExCal was the fifth-fastest method among the 22 methods with complete timings. Its measured median was lower than that of every semisupervised comparator with three successful cases in this probe. CNN had the lowest median time among these semisupervised comparators at 0.8560 s, about 12.9× the ExCal median in this probe. These values describe the timed model-wrapper region on the three supplied cases and do not include data loading or metric computation.

TABLE III  
TSB-AD-M RESULTS ON THE OFFICIAL M/EVAL SPLIT. ENTRIES ARE MEAN ± STANDARD DEVIATION OVER SERIES RESULTS POOLED ACROSS SEEDS. THE BEST VALUE IN EACH COLUMN IS IN BOLD. THE SECOND-BEST VALUE IS UNDERLINED. ALL ENTRIES HAVE 540 VALID OBSERVATIONS FOR THE AREA METRICS. AFFILIATION-F HAS 537 VALID OBSERVATIONS FOR LOF AND 528 FOR LEFT STAMPI. METHOD CITATIONS ARE SHOWN BESIDE THE METHOD NAMES AND USE THE SUPPLIED BIBLIOGRAPHY.  
PER-DATASET DISTRIBUTION OF CURRENT EXCAL SCORES AFTER AVERAGING THE THREE SEEDS. RANKS ARE COMPUTED AGAINST ALL 24 METHODS FOR EACH DATASET.  
TABLE IV
<table><tr><td>Metric</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td><td>Median</td><td>Q1-Q3</td><td>Min-Max</td><td>Rank / #1 / Top-3</td></tr><tr><td>AUC-PR</td><td> $0 . 4 2 1 \pm 0 . 3 7$ </td><td>0.344</td><td>[0.061, 0.753][0.003, 1.000] 7.75 / 30 / 71</td><td></td><td></td></tr><tr><td>VUS-PR</td><td> $0 . 4 5 2 \pm 0 . 3 7$ </td><td>0.411</td><td></td><td></td><td>[0.067, 0.845][0.003, 1.000]8.09 / 29 / 74</td></tr></table>

![](images/0b20cae9be249cd744daafd48511237eff03e96648380f46ff1edff0d48f8d5c.jpg)  
Fig. 3. Accuracy-latency trade-off across 23 evaluated detectors on TSB-AD-M. The horizontal axis shows the median model-wrapper runtime per time series on a logarithmic scale, and the vertical axis shows VUS-PR. Solid circles denote semisupervised methods, and hollow squares denote unsupervised methods. The dashed gray line indicates the Pareto front. ExCal (0.0662 s, 0.452 VUS-PR) lies on the Pareto front and executes 99.6× (∼ 100×) faster than AxonAD (6.5936 s, 0.492 VUS-PR).

<table><tr><td rowspan=1 colspan=9>Model                AUC-PR    AUC-ROC    VUS-PR    VUS-ROC     Std-F1       PA-F1      Event-F1      R-F1        Aff-F</td></tr><tr><td rowspan=1 colspan=1>AxonAD [19]        0.451 ± 0.32</td><td rowspan=1 colspan=1>0.828 ± 0.17 0</td><td rowspan=1 colspan=1>.492 ± 0.32</td><td rowspan=1 colspan=1>0.854 ± 0.16</td><td rowspan=1 colspan=1> ${ \bf 0 . 5 3 4 \pm 0 . 2 9 }$ </td><td rowspan=1 colspan=1> $0 . 7 0 9 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 6 1 5 } } \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> ${ \bf 0 . 4 8 3 \pm 0 . 2 9 }$ </td><td rowspan=1 colspan=1> $0 . 8 6 2 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1>EXCAL              $\underline { { 0 . 4 2 1 } } \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 7 7 2 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1>0.452±0.37</td><td rowspan=1 colspan=1> $0 . 7 9 4 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 4 6 6 } } \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 6 3 5 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 5 5 8 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 4 4 1 } } \pm 0 . 3 0$ </td><td rowspan=1 colspan=1> $0 . 8 5 6 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>PaAno_PAI [23]        $0 . 3 9 8 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 7 8 4 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1> $0 . 4 0 4 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1> $0 . 8 0 6 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1> $0 . 4 3 3 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 7 7 3 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1> $0 . 6 1 2 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 3 7 6 \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> ${ \bf 0 . 8 7 0 \pm 0 . 1 1 }$ </td></tr><tr><td rowspan=1 colspan=1>StreamVAE [20]        $0 . 3 8 3 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1> $0 . 7 5 5 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 4 2 4 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 7 7 1 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 4 5 1 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 5 3 3 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 4 6 9 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 4 1 7 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 8 3 0 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { U S A D } ^ { \left[ 5 \right] }$              $0 . 3 6 3 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 7 3 7 \pm 0 . 2 6$ </td><td rowspan=1 colspan=1>0.403 ± 0.35</td><td rowspan=1 colspan=1> $0 . 7 6 5 \pm 0 . 2 5$ </td><td rowspan=1 colspan=1> $0 . 4 3 3 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 6 2 3 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1> $0 . 5 1 9 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 4 2 3 \pm 0 . 2 9$ </td><td rowspan=1 colspan=1> $0 . 8 3 7 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { C N N } ^ { \mathrm { \scriptsize ~ I 2 4 1 } }$              $0 . 3 5 1 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1> $0 . 7 7 4 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1>0.353 ± 0.36</td><td rowspan=1 colspan=1> $\underline { { 0 . 8 0 7 } } \pm 0 . 1 6$ </td><td rowspan=1 colspan=1> $0 . 4 1 3 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 8 2 6 } } \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 6 4 5 \ : \pm { \bf 0 . 3 6 } }$ </td><td rowspan=1 colspan=1> $0 . 3 0 2 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 8 6 7 } } \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>TimesNet [8]           $0 . 3 4 0 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 6 7 1 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1>0.392 ± 0.34</td><td rowspan=1 colspan=1> $0 . 7 1 1 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1> $0 . 4 1 3 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 6 7 2 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 5 3 7 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 4 0 8 \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> $0 . 8 4 9 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>PatchTST [25]         $0 . 3 3 5 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1>0.675 ± 0.30</td><td rowspan=1 colspan=1>0.390 ± 0.33</td><td rowspan=1 colspan=1> $0 . 7 1 7 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1> $0 . 4 0 6 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 7 0 8 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 5 3 7 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 3 7 8 \pm 0 . 2 5$ </td><td rowspan=1 colspan=1> $0 . 8 4 6 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1>M2N2 [26]            $0 . 3 2 1 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1>0.741 ± 0.20</td><td rowspan=1 colspan=1>0.322 ± 0.36</td><td rowspan=1 colspan=1> $0 . 7 7 6 \pm 0 . 1 9$ </td><td rowspan=1 colspan=1> $0 . 3 8 1 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 8 7 4 \pm 0 . 1 9 }$ </td><td rowspan=1 colspan=1> $0 . 6 0 1 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 2 8 2 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 8 5 8 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>TranAD [6]           $0 . 3 1 9 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 7 1 0 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1>0.317 ± 0.36</td><td rowspan=1 colspan=1>0.745 ± 0.21</td><td rowspan=1 colspan=1> $0 . 3 8 4 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 7 8 9 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 6 0 8 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 2 8 4 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 8 4 7 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>OmniAnomaly [3]      $0 . 3 1 5 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 7 0 6 \pm 0 . 2 2$ </td><td rowspan=1 colspan=1>0.316 ± 0.36</td><td rowspan=1 colspan=1>0.744 ± 0.21</td><td rowspan=1 colspan=1> $0 . 3 7 5 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1> $0 . 7 8 7 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 6 0 4 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 2 9 4 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 8 4 9 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1>FITS [27]             $0 . 3 0 2 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 6 5 9 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1>0.360 ± 0.34</td><td rowspan=1 colspan=1> $0 . 7 0 9 \pm 0 . 2 9$ </td><td rowspan=1 colspan=1> $0 . 3 6 9 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 7 2 1 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 5 2 5 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 2 8 0 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> $0 . 8 4 9 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { L e f t \_ S T A M P i } ^ { \left[ 2 8 \right] }$       $0 . 2 5 3 \pm 0 . 2 8$ </td><td rowspan=1 colspan=1> $0 . 7 0 0 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1>0.321 ± 0.32</td><td rowspan=1 colspan=1> $0 . 7 3 5 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1> $0 . 3 1 1 \pm 0 . 2 8$ </td><td rowspan=1 colspan=1> $0 . 5 9 7 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 4 3 9 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 3 0 2 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 8 0 9 \pm 0 . 1 7$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { K M e a n s A D } ^ { [ 2 1 ] }$         $0 . 2 4 9 \pm 0 . 2 6$ </td><td rowspan=1 colspan=1> $0 . 6 8 7 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1>0.287 ± 0.30</td><td rowspan=1 colspan=1> $0 . 7 2 4 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> $0 . 3 0 5 \pm 0 . 2 5$ </td><td rowspan=1 colspan=1> $0 . 6 7 2 \pm 0 . 3 1$ </td><td rowspan=1 colspan=1> $0 . 4 8 4 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 3 2 7 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 8 2 2 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1> $\mathbf { L S T M A D } ^ { [ 1 ] }$           $0 . 2 4 7 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $0 . 5 9 8 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1>0.243 ± 0.33</td><td rowspan=1 colspan=1> $0 . 6 2 3 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 2 9 7 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 6 6 3 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $0 . 5 0 7 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $0 . 2 0 1 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 7 0 2 \pm 0 . 3 5$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { R o b u s t P C A } ^ { [ 1 4 ] }$         $0 . 2 3 9 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1>0.590 ± 0.24</td><td rowspan=1 colspan=1>0.237 ± 0.35</td><td rowspan=1 colspan=1> $0 . 6 1 2 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 2 8 3 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $0 . 5 7 1 \pm 0 . 3 9$ </td><td rowspan=1 colspan=1> $0 . 3 7 9 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 3 3 0 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 7 9 0 \pm 0 . 1 4$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { D o n u t } ^ { [ 2 ] }$              $0 . 2 1 5 \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> $0 . 6 2 1 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1>0.254 ± 0.30</td><td rowspan=1 colspan=1> $0 . 6 8 0 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 2 8 1 \pm 0 . 2 8$ </td><td rowspan=1 colspan=1> $0 . 5 2 7 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $0 . 4 0 0 \pm 0 . 3 8$ </td><td rowspan=1 colspan=1> $0 . 1 7 8 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1>0.767 ± 0.20</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { I F o r e s t } ^ { \left[ 9 \right] }$             $0 . 2 1 1 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 7 0 4 \pm 0 . 1 9$ </td><td rowspan=1 colspan=1>0.249 ± 0.26</td><td rowspan=1 colspan=1> $0 . 7 4 4 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 2 8 1 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 6 5 6 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $0 . 4 0 3 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 2 4 4 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 8 0 1 \pm 0 . 1 1$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { C O P O D } ^ { \mathrm { [ 1 3 ] } }$            $0 . 2 0 5 \pm 0 . 2 9$ </td><td rowspan=1 colspan=1> $0 . 6 5 2 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1>0.202 ± 0.29</td><td rowspan=1 colspan=1> $0 . 6 8 2 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 2 6 9 \pm 0 . 2 9$ </td><td rowspan=1 colspan=1> $0 . 7 1 7 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $0 . 4 1 4 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $0 . 2 4 2 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 7 9 9 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { E I F } ^ { [ 1 0 ] }$               $0 . 1 8 6 \pm 0 . 2 2$ </td><td rowspan=1 colspan=1> $0 . 6 6 9 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1>0.206 ± 0.25</td><td rowspan=1 colspan=1> $0 . 7 0 6 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1> $0 . 2 5 8 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $0 . 7 3 9 \pm 0 . 2 9$ </td><td rowspan=1 colspan=1> $0 . 4 3 1 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 2 5 7 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 8 1 0 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { H B O S } ^ { [ 1 2 ] }$             $0 . 1 6 1 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> $0 . 6 3 3 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1>0.186 ± 0.24</td><td rowspan=1 colspan=1> $0 . 6 6 9 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 2 4 4 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 6 6 7 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $0 . 3 9 9 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1> $0 . 2 4 4 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $0 . 7 9 6 \pm 0 . 1 2$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { K N N } ^ { \mathrm { \scriptsize ~ [ 1 5 ] } }$              $0 . 1 3 3 \pm 0 . 1 6$ </td><td rowspan=1 colspan=1> $0 . 4 9 9 \pm 0 . 2 2$ </td><td rowspan=1 colspan=1>0.170 ± 0.21</td><td rowspan=1 colspan=1> $0 . 5 7 2 \pm 0 . 2 2$ </td><td rowspan=1 colspan=1> $0 . 1 8 4 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1> $0 . 6 8 1 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 4 4 7 \pm 0 . 3 9$ </td><td rowspan=1 colspan=1> $0 . 2 0 5 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1> $0 . 7 9 1 \pm 0 . 1 4$ </td></tr><tr><td rowspan=1 colspan=1> $\mathrm { L O F } ^ { \mathrm { [ I I ] } }$               $0 . 0 9 6 \pm 0 . 0 9$ </td><td rowspan=1 colspan=1> $0 . 5 3 4 \pm 0 . 1 0$ </td><td rowspan=1 colspan=1> $0 . 1 3 2 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 5 9 0 \pm 0 . 1 4$ </td><td rowspan=1 colspan=1> $0 . 1 5 1 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1> $0 . 5 6 3 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 3 2 5 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $0 . 1 4 9 \pm 0 . 1 4$ </td><td rowspan=1 colspan=1> $0 . 7 6 4 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1>AnomalyTransformer [7] $0 . 0 7 4 \pm 0 . 0 7$ </td><td rowspan=1 colspan=1> $0 . 5 0 5 \pm 0 . 0 8$ </td><td rowspan=1 colspan=1> $0 . 1 1 7 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 5 3 7 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1> $0 . 1 1 8 \pm 0 . 1 0$ </td><td rowspan=1 colspan=1> $0 . 6 5 1 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 3 9 6 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $0 . 1 5 2 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1> $0 . 7 5 4 \pm 0 . 1 8$ </td></tr></table>

ranges extend from 0.0029 to 0.9999 for AUC-PR and from 0.0032 to 0.9998 for VUS-PR. This spread reflects the composition of the benchmark and the per-dataset aggregation.

When methods are ranked separately on every dataset, ExCal has mean rank 7.75 on AUC-PR and 8.09 on VUS-PR among the 24 methods. It has rank one on 30 of 180 datasets for AUC-PR and 29 of 180 for VUS-PR, and it is in the top three on 71 and 74 datasets, respectively. The perdataset ranks and the aggregate ranking therefore summarize different aspects of the results. ExCal has a broad but uneven rank distribution across the benchmark.

## D. Pairwise effects by series

## C. Score distributions and ranks by series

Aggregate means do not indicate whether a method performs similarly across series. We therefore average the three executions within each dataset before computing per-series summaries. Table IV summarizes the distribution for ExCal. The AUC-PR median is 0.3440 with an interquartile range of [0.0608, 0.7534], while the VUS-PR median is 0.4108 with an interquartile range of [0.0665, 0.8450]. The corresponding

We report per-dataset deltas between ExCal and selected baselines in addition to aggregate means. The intervals in Table V are nonparametric 95% bootstrap intervals for the mean delta, based on 10,000 resamples of the 180 dataset-level seed means and a fixed bootstrap generator seed of 2026. The win–loss counts and Holm-adjusted Wilcoxon values use the same paired dataset-level observations. The bootstrap interval describes uncertainty around the arithmetic mean effect.

The Wilcoxon test evaluates the signed-rank distribution. A skewed set of differences can therefore produce a positive mean without a significant majority-dataset advantage. The two viewpoints differ concretely for PaAno PAI on AUC-PR: the bootstrap interval contains zero, while the adjusted Wilcoxon p-value is 0.042. The interval reflects uncertainty in the mean difference, whereas the test reflects the signed ranks, which favor ExCal on 106 of 180 datasets.

ExCal has a positive mean delta relative to StreamVAE, PaAno PAI, USAD, CNN, and AnomalyTransformer on both primary metrics. The interval for the AxonAD comparison contains zero on both metrics, and the paired tests are nonsignificant. These results do not establish equivalence. The StreamVAE intervals are positive even though ExCal wins on fewer datasets than it loses. This indicates a skewed effect distribution in which a smaller number of larger gains outweigh more numerous smaller losses. The Wilcoxon test remains nonsignificant after correction, which does not establish equivalence. Against AnomalyTransformer, ExCal has higher scores on 148 of 180 datasets for AUC-PR and 145 for VUS-PR, with positive mean deltas and small adjusted p-values.

## E. Omnibus and paired tests

The Friedman tests reject the null hypothesis of no systematic differences in method ranks for each tested metric. For AUC-PR, the statistic is 890.015 with Kendall’s W = 0.2150. For VUS-PR it is 949.927 with W = 0.2295. For R-based-F1 it is 1081.997 with W = 0.2614. All three omnibus p-values are below 10<sup>−170</sup>. Affiliation-F has a smaller but still significant overall difference, with statistic 492.190, W = 0.1223, and 175 complete datasets.

Table V gives selected paired comparisons. After Holm correction, the ExCal differences from AxonAD and StreamVAE are not significant on either primary metric. The paired differences relative to PaAno PAI, USAD, and AnomalyTransformer are statistically significant on both primary metrics under this benchmark protocol. Across all 23 baselines, 21 comparisons reject the null for AUC-PR and the same 21 reject it for VUS-PR. The Kendall’s W values near 0.22 indicate that the global ordering, although consistent across datasets, is modest in size, so these counts should be read together with the absolute effect sizes in Table V. The tests indicate differences from most baselines. They do not establish superiority over, or equivalence to, the highest-ranked learned comparators.

## F. Component ablations

Table VI reports the base model and seven variants of the current implementation evaluated on the same 180 series and three seeds. Values highlighted in the table follow the reported rounded summaries. Rank-only fusion has the largest decrease relative to the base model among the evaluated variants, with changes of −0.1042 in AUC-PR, −0.1050 in VUS-PR, and −0.1148 in R-based-F1. Removing raw retrieval is associated with decreases of 0.0450 in AUC-PR and 0.0471 in VUS-PR. Removing excess calibration is associated with decreases of 0.0315 and 0.0346 in the two primary metrics, respectively.

Removing multi-scale statistics is associated with decreases of 0.0173 in AUC-PR and 0.0135 in VUS-PR. Removing the derivative subspace changes AUC-PR by only +0.0003 and R-based-F1 by +0.0037, which is consistent with overlap among the evaluated views on this benchmark. These ablations provide little evidence of an independent contribution from the derivative subspace in this configuration. Interactions among experts remain uncontrolled. Removing prevalence damping changes the primary means by less than 0.001 and raises Rbased-F1 by 0.0109. This pattern is consistent with a larger effect on the threshold-dependent score distribution than on area-based ranking metrics, although the ablation does not isolate that mechanism.

Enabling the optional shape expert changes AUC-PR from 0.4206 to 0.4145 and VUS-PR from 0.4516 to 0.4464, while increasing AUC-ROC from 0.7722 to 0.7858 and VUS-ROC from 0.7935 to 0.8038. It also reduces R-based-F1 from 0.4407 to 0.3769. The default configuration therefore keeps shape retrieval disabled. This is an implementation and ablation result, not an assertion that shape information is universally unhelpful.

## G. Ablation results by series

The aggregate ablation differences are also visible at the dataset level. Table VII reports the mean difference between the base model and each variant after averaging seeds within each dataset. The base model has higher AUC-PR than the variant without raw retrieval on 138 of 180 datasets and higher VUS-PR on 136. For rank-only fusion, the corresponding counts are 131 and 125. By contrast, removing the derivative subspace does not produce a one-sided effect. The variant without the subspace is higher on AUC-PR for 102 datasets, while the base model is higher on VUS-PR for 83. Removing prevalence damping yields many ties because the damping factor is inactive for maps below its prevalence knee. Its aggregate effect is therefore concentrated in a smaller set of series.

The shape comparison is mixed on the area metrics. The base model has higher AUC-PR on 102 datasets and higher VUS-PR on 96, while the median per-dataset advantage is close to zero. Its R-based-F1 difference is more often positive, with 114 base wins versus 51 wins for the shape variant. These counts are consistent with using the configuration without the shape expert as the current default. They do not establish that the optional branch cannot be useful on a different data distribution.

## H. Repeatability across seeds

Table VIII shows the current base result separately for each execution seed. Across seeds, the mean AUC-PR varies from 0.420489 to 0.420818 and VUS-PR from 0.451462 to 0.451735. The standard deviation of the three means by seed is $1 . 6 9 \times 1 0 ^ { - 4 }$ for AUC-PR and $1 . 3 9 \times 1 0 ^ { - 4 }$ for VUS-PR. The corresponding 95th-percentile deviations across seeds at the individual-series level are $6 . 0 2 \times 1 0 ^ { - 4 }$ and $9 . 6 7 \times 1 0 ^ { - 4 }$

TABLE V  
PER-DATASET EFFECTS FOR SELECTED BASELINES. ∆ IS EXCAL MINUS THE BASELINE AFTER AVERAGING SEEDS WITHIN EACH DATASET. BOOTSTRAP INTERVALS USE 10,000 RESAMPLES. W–L COUNTS EXCAL WINS AND BASELINE WINS.
<table><tr><td>Baseline</td><td colspan="4">AUC-PR</td><td colspan="4">VUS-PR</td></tr><tr><td></td><td>∆</td><td>95% CI</td><td>W-L</td><td>PHolm</td><td>∆</td><td>95% CI</td><td>W-L</td><td>PHolm</td></tr><tr><td> $\mathbf { A x o n A D } ^ { [ 1 9 ] }$ </td><td>-0.031</td><td> $[ - 0 . 0 7 5 , 0 . 0 1 4 ]$ </td><td>94-86</td><td>0.177</td><td>-0.041</td><td>[-0.084, 0.001]</td><td>92-88</td><td>0.276</td></tr><tr><td>StreamVAE [20]</td><td>+0.038</td><td>[0.010, 0.067]</td><td>88-92</td><td>0.111</td><td>+0.027</td><td>[0.001, 0.055]</td><td>84-96</td><td>0.276</td></tr><tr><td>PaAno_PAI [23]</td><td>+0.022</td><td>[-0.023, 0.066]</td><td>106-74</td><td>0.042</td><td>+0.047</td><td>[0.009, 0.087]</td><td>106-74</td><td>0.016</td></tr><tr><td>USAD [5]</td><td>+0.058</td><td>[0.023, 0.094]</td><td>120-60</td><td>0.001</td><td>+0.049</td><td>[0.014, 0.085]</td><td>122-58</td><td>0.002</td></tr><tr><td>CNN [24]</td><td>+0.070</td><td>[0.020, 0.120]</td><td>114-66</td><td>0.0002</td><td>+0.099</td><td>[0.055, 0.143]</td><td>113-67</td><td> $8 . 9 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>AnomalyTransformer [7]</td><td>+0.346</td><td>[0.297, 0.395]</td><td>148-32</td><td>2.54×10 -23</td><td>+0.335</td><td>[0.285,0.387]</td><td>145-35</td><td> $1 . 1 6 \times { 1 0 } ^ { - 2 1 }$ </td></tr></table>

TABLE VI

EXCAL COMPONENT ABLATIONS ON TSB-AD-M. ENTRIES ARE MEAN ± STANDARD DEVIATION OVER 540 OBSERVATIONS ACROSS SERIES AND SEEDS. THE BEST VALUE IN EACH COLUMN IS IN BOLD. THE SECOND-BEST VALUE IS UNDERLINED.
<table><tr><td>Variant</td><td>AUC-PR</td><td>AUC-ROC</td><td>VUS-PR</td><td>VUS-ROC</td><td>Std-F1</td><td>PA-F1</td><td>Event-F1</td><td>R-F1</td><td>Aff-F</td></tr><tr><td>Base, shape disabled</td><td> $\underline { { 0 . 4 2 1 } } \pm 0 . 3 7$ </td><td> $0 . 7 7 2 \pm 0 . 2 4$ </td><td> ${ \bf 0 . 4 5 2 \pm 0 . 3 7 }$ </td><td> $0 . 7 9 4 \pm 0 . 2 4$ </td><td> $0 . 4 6 6 \pm 0 . 3 4$ </td><td> $0 . 6 3 5 \pm 0 . 3 8$ </td><td> $0 . 5 5 8 \pm 0 . 3 8$ </td><td> $0 . 4 4 1 \pm 0 . 3 0$ </td><td> $0 . 8 5 6 \pm 0 . 1 2$ </td></tr><tr><td>With shape expert</td><td> $0 . 4 1 5 \pm 0 . 3 5$ </td><td> $\mathbf { 0 . 7 8 6 \pm 0 . 2 4 }$ </td><td> $0 . 4 4 6 \pm 0 . 3 6$ </td><td>0.804 ± 0.24 0.469 ± 0.32</td><td></td><td> $0 . 6 3 7 \pm 0 . 3 8$ </td><td> $0 . 5 5 9 \pm 0 . 3 8$ </td><td> $0 . 3 7 7 \pm 0 . 2 8$ </td><td> $0 . 8 4 2 \pm 0 . 1 3$ </td></tr><tr><td>Without raw retrieval</td><td> $0 . 3 7 6 \pm 0 . 3 6$ </td><td> $0 . 7 3 3 \pm 0 . 2 5$ </td><td> $0 . 4 0 5 \pm 0 . 3 7$ </td><td> $0 . 7 6 3 \pm 0 . 2 4$ </td><td> $0 . 4 2 4 \pm 0 . 3 4$ </td><td> $\underline { { 0 . 6 8 7 } } \pm 0 . 3 3$ </td><td> $0 . 5 2 2 \pm 0 . 3 8$ </td><td> $0 . 3 9 7 \pm 0 . 3 0$ </td><td> $0 . 8 4 1 \pm 0 . 1 2$ </td></tr><tr><td>Without derivative subspace</td><td> ${ \bf 0 . 4 2 1 \pm 0 . 3 7 }$ </td><td> $0 . 7 7 1 \pm 0 . 2 4$ </td><td> $0 . 4 5 0 \pm 0 . 3 7$ </td><td> $0 . 7 9 0 \pm 0 . 2 4$ </td><td> $\underline { { 0 . 4 6 8 } } \pm 0 . 3 4$ </td><td> $0 . 6 2 4 \pm 0 . 3 8$ </td><td> $0 . 5 4 1 \pm 0 . 3 8$ </td><td> $\underline { { 0 . 4 4 4 } } \pm 0 . 3 1$ </td><td> $0 . 8 5 3 \pm 0 . 1 2$ </td></tr><tr><td>Without multi-scale statistics</td><td> $0 . 4 0 3 \pm 0 . 3 5$ </td><td> $\underline { { 0 . 7 8 0 } } \pm 0 . 2 2$ </td><td> $0 . 4 3 8 \pm 0 . 3 7$ </td><td> $0 . 8 0 1 \pm 0 . 2 2$ </td><td> $0 . 4 5 7 \pm 0 . 3 2$ </td><td> $0 . 6 6 7 \pm 0 . 3 8$ </td><td> ${ \bf 0 . 6 0 2 \pm 0 . 3 8 }$ </td><td> $0 . 4 1 3 \pm 0 . 2 8$ </td><td> $\mathbf { 0 . 8 6 2 \pm 0 . 1 2 }$ </td></tr><tr><td>Without excess calibration</td><td> $0 . 3 8 9 \pm 0 . 3 4$ </td><td> $0 . 7 5 9 \pm 0 . 2 5$ </td><td> $0 . 4 1 7 \pm 0 . 3 6$ </td><td> $0 . 7 8 0 \pm 0 . 2 5$ </td><td> $0 . 4 3 2 \pm 0 . 3 2$ </td><td> $\mathbf { 0 . 7 5 2 \pm 0 . 3 0 }$ </td><td> $\underline { { 0 . 5 8 8 } } \pm 0 . 3 7$ </td><td> $0 . 4 0 3 \pm 0 . 2 9$ </td><td> $0 . 8 6 1 \pm 0 . 1 1$ </td></tr><tr><td>Without prevalence damping</td><td> $0 . 4 2 0 \pm 0 . 3 7$ </td><td> $0 . 7 7 2 \pm 0 . 2 4$ </td><td> $\underline { { 0 . 4 5 1 } } \pm 0 . 3 7$ </td><td> $0 . 7 9 2 \pm 0 . 2 4$ </td><td> $0 . 4 6 7 \pm 0 . 3 4$ </td><td> $0 . 6 2 8 \pm 0 . 3 9$ </td><td> $0 . 5 5 5 \pm 0 . 3 8$ </td><td> ${ \bf 0 . 4 5 2 \pm 0 . 3 0 }$ </td><td> $0 . 8 5 6 \pm 0 . 1 2$ </td></tr><tr><td>Rank-only fusion</td><td> $0 . 3 1 6 \pm 0 . 3 3$ </td><td> $0 . 7 4 3 \pm 0 . 2 3$ </td><td> $0 . 3 4 7 \pm 0 . 3 4$ </td><td> $0 . 7 7 1 \pm 0 . 2 3$ </td><td> $0 . 3 6 0 \pm 0 . 3 1$ </td><td> $0 . 6 7 1 \pm 0 . 3 4$ </td><td> $0 . 5 0 7 \pm 0 . 3 7$ </td><td> $0 . 3 2 6 \pm 0 . 2 4$ </td><td> $0 . 8 4 4 \pm 0 . 1 1$ </td></tr></table>

TABLE VII

PER-DATASET ABLATION EFFECTS AFTER AVERAGING THE THREE SEEDS. ∆ IS BASE EXCAL MINUS THE ABLATION. W–L COUNTS BASE WINS AND ABLATION WINS, WITH TIES OMITTED.
<table><tr><td>Variant</td><td colspan="2">AUC-PR</td><td colspan="2">VUS-PR</td><td colspan="2">R-based-F1</td></tr><tr><td></td><td>∆</td><td>W-L</td><td>∆</td><td>W-L</td><td>∆</td><td>W-L</td></tr><tr><td>With shape expert</td><td>+0.006</td><td>102-76</td><td>+0.005</td><td>96-81</td><td>+0.064</td><td>114-51</td></tr><tr><td>Without raw retrieval</td><td>+0.045</td><td>138-37</td><td>+0.047</td><td>136-39</td><td>+0.044</td><td>117-29</td></tr><tr><td>Without derivative subspace</td><td>-0.000</td><td>73-102</td><td>+0.001</td><td>83-89</td><td>-0.004</td><td>62-90</td></tr><tr><td>Without multi-scale statistics</td><td>+0.017</td><td>76-104</td><td>+0.014</td><td>70-109</td><td>+0.028</td><td>100-78</td></tr><tr><td>Without excess calibration</td><td>+0.032</td><td>118-61</td><td>+0.035</td><td>114-63</td><td>+0.038</td><td>119-58</td></tr><tr><td>Without prevalence damping</td><td>+0.001</td><td>55-71</td><td>+0.001</td><td>56-66</td><td>-0.011</td><td>42-68</td></tr><tr><td>Rank-only fusion</td><td>+0.104</td><td>131-49</td><td>+0.105</td><td>125-55</td><td>+0.115</td><td>126-53</td></tr></table>

TABLE VIII  
CURRENT EXCAL MEAN SCORES BY EXECUTION SEED.
<table><tr><td>Seed</td><td>AUC-PR VUS-PR AUC-ROC VUS-ROC</td><td></td><td></td><td></td></tr><tr><td>2024</td><td>0.421</td><td>0.452</td><td>0.772</td><td>0.794</td></tr><tr><td>2025</td><td>0.421</td><td>0.452</td><td>0.772</td><td>0.794</td></tr><tr><td>2026</td><td>0.420</td><td>0.451</td><td>0.772</td><td>0.793</td></tr><tr><td>Pooled</td><td>0.421</td><td>0.452</td><td>0.772</td><td>0.794</td></tr></table>

These are repeatability summaries, not uncertainty intervals for the benchmark population.

## I. Results by dataset family

0.37), and SVDB $( 0 . 5 1 5 \pm 0 . 3 0 )$ , but lower on Genesis, GHL, and GECCO. These results are consistent with different performance across families. They do not establish uniform performance across all operating conditions.

The global standard deviation accompanies variation across the 17 source families. Table IX shows current ExCal results by family. AUC-PR is highest on Exathlon (0.944 ± 0.12), SWaT (0.549 ± 0.28), SMAP (0.491 ± 0.36), SMD (0.483 ± 0.27), and SVDB (0.479 ± 0.29). It is lower on GHL $( 0 . 0 1 3 \pm 0 . 0 1 )$ , Genesis (0.004), CreditCard (0.054), and OPPORTUNITY $( 0 . 1 2 1 \pm 0 . 2 5 )$ . VUS-PR is higher on Exathlon $( 0 . 9 1 5 \pm 0 . 1 9 )$ , TAO (0.703±0.24), SMAP (0.519±

## VI. DISCUSSION

Ablations associate raw retrieval and excess magnitude with the largest primary-metric differences. Multiscale statistics have a smaller association, while removing the derivative subspace barely changes the primary metrics. The optional shape branch lowers AUC-PR and VUS-PR in the evaluated configuration. These are comparisons among configurations, not evidence of independent causal contributions. They also do not show that a branch is generally unhelpful.

The second-place aggregate ranking coexists with heterogeneous family results and modest Kendall’s W values. Corrected tests do not distinguish EXCAL from AxonAD or StreamVAE on either primary metric. The score decomposition identifies contributing normality maps, not physical root causes. Batch prevalence and rank statistics and a three-series, single-run timing probe preclude claims of online or real-time readiness.

TABLE IX  
EXCAL FAMILY-LEVEL MACRO RESULTS. THE COUNT IS THE NUMBER OF UNIQUE SERIES. EACH SERIES IS FIRST AVERAGED OVER THE THREE EXECUTION SEEDS, AND THE DISPLAYED STANDARD DEVIATION IS THEN COMPUTED ACROSS SERIES. A DASH DENOTES A FAMILY WITH ONE SERIES
<table><tr><td>Family</td><td>Series</td><td>AUC-PR</td><td>VUS-PR</td><td>AUC-ROC</td></tr><tr><td>CATSv2</td><td></td><td>50.157 ± 0.13</td><td> $0 . 1 0 7 \pm 0 . 0 9$ </td><td> $0 . 6 6 5 \pm 0 . 1 7$ </td></tr><tr><td>CreditCard</td><td>1</td><td>0.054</td><td>0.115</td><td>0.764</td></tr><tr><td>Daphnet</td><td>1</td><td>0.407</td><td>0.425</td><td>0.944</td></tr><tr><td>Exathlon</td><td></td><td>25 0.944 ± 0.12</td><td> $0 . 9 1 5 \pm 0 . 1 9$ </td><td> $0 . 9 6 5 \pm 0 . 0 8$ </td></tr><tr><td>GECCO</td><td>1</td><td>0.274</td><td>0.046</td><td>0.750</td></tr><tr><td>GHL</td><td>23</td><td>0.013 ± 0.01</td><td> $0 . 0 1 4 \pm 0 . 0 1$ </td><td> $0 . 4 5 0 \pm 0 . 2 4$ </td></tr><tr><td>Genesis</td><td>1</td><td>0.004</td><td>0.012</td><td>0.595</td></tr><tr><td>LTDB</td><td>4</td><td> $0 . 3 9 9 \pm 0 . 3 7$ </td><td> $0 . 4 4 3 \pm 0 . 3 8$ </td><td> $0 . 7 7 0 \pm 0 . 2 3$ </td></tr><tr><td>MITDB</td><td>11</td><td> $0 . 3 7 2 \pm 0 . 2 7$ </td><td> $0 . 2 8 8 \pm 0 . 2 3$ </td><td> $0 . 8 1 1 \pm 0 . 1 3$ </td></tr><tr><td>MSL</td><td>14</td><td> $0 . 4 1 2 \pm 0 . 3 9$ </td><td> $0 . 4 4 4 \pm 0 . 4 0$ </td><td> $0 . 8 3 1 \pm 0 . 1 5$ </td></tr><tr><td>OPPORTUNITY</td><td>7</td><td> $0 . 1 2 1 \pm 0 . 2 5$ </td><td> $0 . 1 2 3 \pm 0 . 2 5$ </td><td> $0 . 3 0 0 \pm 0 . 3 1$ </td></tr><tr><td>PSM</td><td>1</td><td>0.212</td><td>0.222</td><td>0.738</td></tr><tr><td>SMAP</td><td>25</td><td> $0 . 4 9 1 \pm 0 . 3 6$ </td><td> $0 . 5 1 9 \pm 0 . 3 7$ </td><td> $0 . 8 6 3 \pm 0 . 1 8$ </td></tr><tr><td>SMD</td><td>20</td><td> $0 . 4 8 3 \pm 0 . 2 7$ </td><td> $0 . 4 4 2 \pm 0 . 2 5$ </td><td> $0 . 8 8 6 \pm 0 . 1 7$ </td></tr><tr><td>SVDB</td><td>28</td><td> $0 . 4 7 9 \pm 0 . 2 9$ </td><td> $0 . 5 1 5 \pm 0 . 3 0$ </td><td> $0 . 8 6 4 \pm 0 . 1 3$ </td></tr><tr><td>SWaT</td><td>2</td><td> $0 . 5 4 9 \pm 0 . 2 8$ </td><td> $0 . 3 7 6 \pm 0 . 0 4$ </td><td> $0 . 8 4 4 \pm 0 . 0 1$ </td></tr><tr><td>TAO</td><td>11</td><td> $0 . 1 2 3 \pm 0 . 0 6$ </td><td> $0 . 7 0 3 \pm 0 . 2 4$ </td><td> $0 . 5 9 3 \pm 0 . 0 3$ </td></tr></table>

## VII. LIMITATIONS AND FUTURE DIRECTIONS

The evaluation covers one TSB-AD-M split, three seeds, and correlated performance metrics. The timing probe includes three series with one measured repetition and excludes loading, metrics, memory, and energy. A clean normal prefix is assumed. Contamination can alter reference banks and floors, while legitimate regime changes can trigger broad excess. A single detected period may miss nonstationary or multiperiodic structure. Moreover, batch prevalence and rank normalization change a point’s score when the surrounding sequence changes, so a causal deployment variant remains unevaluated. Score decomposition has not been tested for operator diagnosis, and benchmark F1 values do not establish deployment thresholds. Relevant next steps include independent telemetry datasets, contamination studies, causal fusion references, threshold and detection-delay evaluation, and larger resource studies.

## VIII. CONCLUSION

EXCAL combines prefix-fitted raw retrieval, signal/derivative subspace residuals, and multiscale robust statistics through magnitude-preserving excess fusion without gradient optimization. Prevalence and rank terms make its inference batch and transductive. Across 180 TSB-AD-M series, the detector ranks second among 24 methods on AUC-PR and VUS-PR, although paired tests find no significant difference from AxonAD or StreamVAE. Ablations associate the largest score changes with raw retrieval and magnitudepreserving fusion. The runtime probe is preliminary, and operational performance and causal scoring remain untested.

## REFERENCES

[1] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, “Long short term memory networks for anomaly detection in time series,” 04 2015.

[2] H. Xu, W. Chen, N. Zhao, Z. Li, J. Bu, Z. Li, Y. Liu, Y. Zhao, D. Pei, Y. Feng, J. Chen, Z. Wang, and H. Qiao, “Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications,” Feb. 2018, arXiv:1802.03903. [Online]. Available: http://arxiv.org/abs/1802.03903

[3] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. Anchorage AK USA: ACM, Jul. 2019, pp. 2828–2837. [Online]. Available: https://dl.acm.org/doi/10.1145/3292500.3330672

[4] L. Li, J. Yan, H. Wang, and Y. Jin, “Anomaly Detection of Time Series With Smoothness-Inducing Sequential Variational Auto-Encoder,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 3, pp. 1177–1191, Mar. 2021. [Online]. Available: https://ieeexplore.ieee.org/document/9064715/

[5] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, “USAD: UnSupervised anomaly detection on multivariate time series,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2020, pp. 3395–3404. [Online]. Available: https://dl.acm.org/doi/10.1145/3394486.3403392

[6] S. Tuli, G. Casale, and N. R. Jennings, “TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data,” May 2022, arXiv:2201.07284. [Online]. Available: http: //arxiv.org/abs/2201.07284

[7] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy,” Jun. 2022, arXiv:2110.02642. [Online]. Available: http://arxiv.org/abs/2110.02642

[8] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis,” Apr. 2023, arXiv:2210.02186. [Online]. Available: http://arxiv.org/abs/2210.02186

[9] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” in 2008 Eighth IEEE International Conference on Data Mining, Dec. 2008, pp. 413–422, iSSN: 2374-8486. [Online]. Available: https://ieeexplore.ieee.org/document/4781136

[10] S. Hariri, M. C. Kind, and R. J. Brunner, “Extended Isolation Forest,” IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 4, pp. 1479–1489, Apr. 2021. [Online]. Available: https://ieeexplore.ieee.org/document/8888179/

[11] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “LOF: identifying density-based local outliers,” vol. 29, no. 2, pp. 93–104, 2000. [Online]. Available: https://dl.acm.org/doi/10.1145/335191.335388

[12] M. Goldstein and A. Dengel, “Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm,” 09 2012.

[13] Z. Li, Y. Zhao, N. Botta, C. Ionescu, and X. Hu, “COPOD: Copula-based outlier detection.” [Online]. Available: http://arxiv.org/abs/2009.09463

[14] E. J. Candes, X. Li, Y. Ma, and J. Wright, “Robust principal component\` analysis?” vol. 58, no. 3, pp. 1–37, 2011. [Online]. Available: https://dl.acm.org/doi/10.1145/1970392.1970395

[15] T. Cover and P. Hart, “Nearest neighbor pattern classification,” IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21–27, January 1967.

[16] A. Mackiewicz and W. Ratajczak, “Principal components analysis´ (pca),” Computers & Geosciences, vol. 19, no. 3, pp. 303–342, 1993. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ 009830049390090R

[17] Q. Liu and J. Paparrizos, “The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark,” in Advances in Neural Information Processing Systems 37. Vancouver, BC, Canada: Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024, pp. 108 231–108 261. [Online]. Available: http: //www.proceedings.com/079017-3437.html

[18] J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, and M. J. Franklin, “Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection,” Proceedings of the VLDB Endowment, vol. 15, no. 11, pp. 2774–2787, Jul. 2022. [Online]. Available: https://dl.acm.org/doi/10.14778/3551793.3551830

[19] K.-K. Ozer, R. Ebeling, and M. Enzweiler, “Surprised by attention:<sup>¨</sup> Predictable query dynamics for time series anomaly detection,” in Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track, Demo Track and Industrial Track. Cham: Springer Nature Switzerland, 2027, pp. 217–234.

[20] K.-K. Ozer, R. Ebeling, and M. Enzweiler, “Stream-vae: Dual-path<sup>¨</sup> routing for slow and fast dynamics in vehicle telemetry anomaly

detection,” in 2026 IEEE Intelligent Vehicles Symposium (IV), 2026, pp. 444–451.

[21] T. Yairi, Y. Kato, and K. Hori, “Fault detection by mining association rules from housekeeping data,” 06 2001.

[22] Y. Zhao, Z. Nasrullah, and Z. Li, “PyOD: A Python Toolbox for Scalable Outlier Detection,” Journal of Machine Learning Research, vol. 20, no. 96, pp. 1–7, 2019. [Online]. Available: http://jmlr.org/papers/v20/19-011.html

[23] J. Park and S. Kang, “Paano: Patch-based representation learning for time-series anomaly detection,” 2026. [Online]. Available: https: //arxiv.org/abs/2602.01359

[24] H. Ren, B. Xu, Y. Wang, C. Yi, C. Huang, X. Kou, T. Xing, M. Yang, J. Tong, and Q. Zhang, “Time-Series Anomaly Detection Service at Microsoft,” Jun. 2019, arXiv:1906.03821. [Online]. Available: http://arxiv.org/abs/1906.03821

[25] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” 2023. [Online]. Available: https://arxiv.org/abs/2211.14730

[26] D. Kim, S. Park, and J. Choo, “When model meets new normals: Test-time adaptation for unsupervised time-series anomaly detection,” 2024. [Online]. Available: https://arxiv.org/abs/2312.11976

[27] Z. Xu, A. Zeng, and Q. Xu, “FITS: Modeling time series with \$10k\$ parameters.” [Online]. Available: http://arxiv.org/abs/2307.03756

[28] C.-C. M. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H. A. Dau, D. F. Silva, A. Mueen, and E. Keogh, “Matrix profile i: All pairs similarity joins for time series: A unifying view that includes motifs, discords and shapelets,” in 2016 IEEE 16th International Conference on Data Mining (ICDM), 2016, pp. 1317–1322.