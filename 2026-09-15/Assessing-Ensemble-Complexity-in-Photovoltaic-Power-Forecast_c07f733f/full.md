# Assessing Ensemble Complexity in Photovoltaic Power Forecasting

Sun Ze<sup>a</sup>, Zhou Liguo<sup>a</sup>, Xu Yuqing<sup>a</sup>, Yu Lei<sup>a</sup> and Jiang Mingming<sup>a</sup>

<sup>a</sup>School ofComputer Science and Technology, Huaibei Normal University, China

## A R T I C L E I N F O

Keywords:   
Photovoltaic power forecasting   
Forecast combination   
Component ablation   
Computational cost   
Chronological evaluation

## A BS T R AC T

An ensemble can improve photovoltaic forecasts while adding components that contribute little or increase computation. We assess these efects through matched comparisons and ablations of a fixed heterogeneous predictor bank. Hourly experiments use GEFCom2014 and three additional public datasets, with chronological partitions and three seeds. Under retrospective ERA5 assistance, static fusion reduces scaled mean absolute error against matched boosting by 1.11%, 4.41%, and 1.63% on PVDAQ, OPSD, and Ausgrid; only OPSD remains supported after multiple-comparison correction. Weather gating ofers no consistent incremental benefit. Exploratory member removals show grouplevel dependence alongside individual redundancy. A separate, previously inspected fifteen-minute case replaces one neural member with a tree predictor: normalized error falls by 1.72%, but measured inference is slower. These findings support component-wise evaluation with explicit limits on weather availability and test-set reuse.

## 1. Introduction

Photovoltaic (PV) power forecasts over the next few hours support the scheduling of generation, storage, and reserve capacity. At this horizon, the daily solar cycle is predictable, whereas passing clouds and site-specific operating conditions can change output rapidly. Forecast errors therefore depend on the prediction horizon, local measurements, and the weather information available when the forecast is issued [2, 16]. These diferences also complicate comparisons: target-period realized weather provides information that a model restricted to past observations would not have.

Forecast combination is an established response to uncertainty about the best predictor [27]. A bank can contain recurrent networks, attention-based models, temporal mixers, and regression predictors, followed by learned weights, sitespecific adjustments, and weather-dependent corrections. Each component adds something to the implementation; it need not add useful predictive information. The practical question is whether the resulting error reduction justifies the added fitting and inference requirements.

Attribution is particularly dificult when an ensemble is compared with a single predictor. A diference can arise from combining forecasts, from extra weather inputs, or from labels used to fit the weights but withheld from the control. An equal epoch count per neural member also does not equalize the total cost of training a bank and training one model. Component ablations and explicit information and fitting-period controls are therefore needed to interpret an aggregate score [12]. The incremental analysis of NWP-topower processing by Mayer et al. [21] motivates a similar question here, although our ensemble members are power predictors rather than weather forecasts.

We study three questions. First, does learned fusion improve on equal averaging and a regression control with matched information and fitting-label access? Second, do site dependence, global shrinkage, and weather gating contribute beyond simpler combinations, and how sensitive is the current combination to its members? Third, what accuracy and inference-cost change follows the replacement of one member in a separately evaluated small ensemble? These questions concern the value of specific components; model count alone is not treated as a methodological contribution.

The primary implementation, denoted V9 in the source project, learns station–horizon convex weights, shrinks them toward a global estimate, and applies a bounded weatherregime adjustment with a chronological confirmation rule. Its archived private score of approximately 91.8% is audited separately. Public experiments train a fourteen-member hourly adaptation on GEFCom2014 [8, 14] and three further sources. The latter experiments use matched regression controls and explicitly retrospective ERA5 weather. We retain failed gate confirmations and perform exploratory fixed-weight removals without selecting a new model from their test results.

A secondary fifteen-minute case compares two fourslot combinations on identical private targets, retaining three neural checkpoints and replacing the fourth member with a tree predictor. It adds a measured inference-cost comparison to the accuracy analysis, but uses an already inspected historical period and is not an independent public validation. The contributions are measured accuracy gains within explicitly controlled comparisons, component-level evidence explaining their limits, and an inference-cost assessment. Member count is a design variable in this assessment; the diferent implementations do not establish a controlled reduction from 82 to 14 to four models.

## 2. Related works

## 2.1. PV forecasting and weather-conditioned models

The survey of Antonanzas et al. [2] provides a foundation for distinguishing PV forecasting tasks by their inputs and horizons. Iheanetu [16] review data-driven PV forecasting procedures, while the more recent deep-learning review of Husein et al. [15] identifies inconsistent datasets, forecasting horizons, and evaluation metrics as obstacles to comparing published models. These observations motivate evaluating all constituents and ensemble variants on a shared target population.

Weather classification has already been investigated as a forecasting mechanism. Amarasinghe et al. [1] combine random forests, support vector regression, and deep belief networks, with cloud-based classification and separate ensembles for diferent subsets. Their study uses 21 German PV facilities at three-hour resolution and a shufled train– test split. Weather-classification-based MARS forecasting provides another example of this research direction [30]. Consequently, weather-conditioned aggregation itself should not be presented as a new idea.

The implementation studied here uses a diferent parameterization: it keeps its predictor bank fixed, first learns local and global convex weights, and then learns bounded regimedependent ofsets at the model-family level. This structure avoids training a separate copy of every model for every weather class. Whether this smaller adjustment is useful remains an empirical question. Published errors from weatherclassification studies are not used as numerical baselines here because their datasets, resolutions, splits, and meteorological information difer.

Forecast meteorology also difers from realized meteorology. Work combining ensemble numerical weather prediction (NWP) with physical model chains explicitly considers forecast-weather uncertainty [20]. An archived dataset from the ECMWF Ensemble Prediction System ofers a further example of data developed specifically for probabilistic solar power forecasting [26]. It is a relevant source for future evaluation, rather than an additional dataset tested in this study. This distinction matters for the private V9 experiment because its weather context is obtained from a historical weather cache at target timestamps. In contrast, the GEFCom2014 solar variables are supplied as NWP predictors, although the processed mirror used here does not retain forecast-issuance timestamps. Both limitations are disclosed in the experimental design.

The choice of processing steps also deserves attention. Mayer et al. [21] compare sixteen workflows for converting ensemble NWP into deterministic PV forecasts at five Hungarian plants. Their analysis shows that some intermediate processing stages add little once the final power forecast is bias-corrected. This finding motivates evaluating the incremental contribution of each stage, rather than attributing the final error to every component in a pipeline. Their ensemble consists of weather-forecast members, whereas the present study combines separately trained power predictors. The datasets, information sets, and error normalization also difer, so their reported scores are not numerical baselines for this study.

## 2.2. Neural and regression forecasting models

Long short-term memory networks model sequential dependencies through gated recurrent states [13]. Temporal Fusion Transformers provide an interpretable multi-horizon forecasting architecture [19]. PatchTST introduces patchbased time-series representations [22], while TiDE uses a dense encoder–decoder with covariates [6]. TSMixer and TimeMixer study mixing operations along feature, temporal, or sampling-scale dimensions [4, 25]. Mamba introduces input-dependent selective state-space dynamics as an alternative sequence-modeling mechanism [10]; its general sequence results do not establish performance on this PV task. These methods motivate architectural diversity, rather than a presumption that one architecture dominates PV forecasting.

The findings of Zeng et al. [29] also motivate including simple models in time-series comparisons. Accordingly, the public experiment includes ridge regression and gradient boosting [9], together with persistence-based baselines. The neural benchmarks are the original project’s implementations of nine architecture families. Several are explicitly simplified or task-adapted: for example, the repository uses a pure-PyTorch Mamba-style block. The benchmark therefore compares this implementation family and does not establish a ranking of the authors’ oficial implementations of all named architectures.

## 2.3. Forecast combinations and evaluation

Forecast combination spans simple averages, learned weighting, and conditional combinations [27]. A stationspecific weight can capture local diferences, while a horizonspecific weight can reflect the changing usefulness of persistence and meteorological predictors. Shrinkage toward global weights is a practical way to reduce the sensitivity of local estimates when few fitting windows are available.

Input-dependent allocation among predictors also has a long history. Adaptive mixtures of local experts learn to assign subsets of examples to expert networks [17], and hierarchical mixtures of experts model both mixture coeficients and component responses within a learned hierarchy [18]. V9 belongs to the broader family of conditional combinations, but the evaluated implementation fits a bounded adjustment to a previously trained predictor bank. It does not jointly train a hierarchical mixture model or use the EM procedure of Jordan and Jacobs [18]. Its usefulness must therefore be assessed through the specific weight parameterization and confirmation protocol, rather than through a claim that conditional expert weighting is new.

Evaluation must separate predictor fitting, hyperparameter selection, ensemble fitting, and final scoring [12]. GEF-Com2014 provides a widely used public setting for renewableenergy forecasting [14]; it has also supported probabilistic generative-model studies [8]. The present task is a deterministic, rolling four-hour adaptation of the solar track. Competition quantile scores and published probabilistic results are therefore contextual references, rather than directly comparable scores.

## 3. Method

## 3.1. Task and information set

Let $p _ { s , t }$ denote the output of site � and let $c _ { s } > 0$ be its normalization scale. The target is $y _ { s , t } = p _ { s , t } / c _ { s }$ . Given � historical observations, calendar variables, and an eligible

meteorological input �, predictor � produces

$$
\widehat { \boldsymbol { y } } _ { s , t + h } ^ { ( m ) } = { f _ { m } } ( \boldsymbol { y } _ { s , t - L + 1 : t } , \boldsymbol { x } , s , h ) , \qquad h = 1 , \ldots , H .\tag{1}
$$

The private task uses five-minute data and $H \ = \ 4 8 ;$ the public adaptation uses hourly data and $L = 2 4 , H = 4$ Both forecast the next four hours. The separate native fifteenminute replacement case is specified in Section 4.12; it is not a further compression of this bank. In a deployed system, � must belong to the information set available at origin �. The private archived result is treated as a retrospective weatherassisted result until this requirement is established for its weather feed.

## 3.2. Frozen heterogeneous predictor bank

The archived V9 bank contains 72 trained predictors and 10 statistical baselines. Its learned predictors span nine neural architecture families and a station-wise ridge component; checkpoints difer in training history and weather usage. The baseline groups include persistence, daily persistence, climatology, state-conditioned climatology, and several historicalanalog configurations. V9 changes the combination weights while leaving these predictors frozen.

The public experiment retrains one implementation from each of the nine neural families per random seed and adds five classical predictors. This gives � = 14 constituents per run. Public sites use new embeddings and newly fitted models; private weights and checkpoints are not transferred to the public test set. Reducing the bank is an explicit experimental adaptation, so the public experiment validates the combination mechanism at this scale rather than reproducing the private 82-constituent system.

## 3.3. Station–horizon weights and shrinkage

Predictor errors can vary by site and lead time. We represent this variation with nonnegative weights, parameterized by logits and normalized across predictors:

$$
w _ { s , m , h } = \frac { \exp ( a _ { s , m , h } ) } { \sum _ { j = 1 } ^ { M } \exp ( a _ { s , j , h } ) } .\tag{2}
$$

The weights minimize mean absolute error on the ensemblefitting partition. Local estimates can be unstable when a site contributes few windows, so a separate global vector $w _ { m , h } ^ { ( g ) }$ is fitted using all sites. Each local estimate is shrunk toward this shared vector:

$$
\bar { w } _ { s , m , h } = \alpha _ { s } w _ { s , m , h } + ( 1 - \alpha _ { s } ) w _ { m , h } ^ { ( g ) } ,\tag{3}
$$

$$
\alpha _ { s } = \frac { n _ { s } } { n _ { s } + \kappa } ,\tag{4}
$$

where $n _ { s }$ counts fitting windows and � is selected on a later chronological partition. Convexity is preserved because both component weight vectors sum to one. The static ensemble is $\begin{array} { r } { \widehat { y } _ { s , t + h } ^ { ( 0 ) } = \sum _ { m } \bar { w } _ { s , m , h } \widehat { y } _ { s , t + h } ^ { ( m ) } . } \end{array}$

## 3.4. Weather regimes

The selected private configuration, physical\_weather3, uses irradiance index �, cloud cover � in percent, precipitation $r ,$ and the absolute one-hour change $d _ { k }$ . The three regimes are assigned by

$$
z = \left\{ \begin{array} { l l } { 0 , } & { k \geq 0 . 7 5 , q \leq 5 0 , r < 0 . 0 5 , d _ { k } < 0 . 1 5 , } \\ { 2 , } & { k < 0 . 3 5 \mathrm { o r } q \geq 7 0 \mathrm { o r } r \geq 0 . 0 5 , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{5}
$$

The adverse regime has precedence in the source implementation. The private context uses the cached clear-sky index and target-time weather. These regime names describe deterministic rules; they are not independently verified weather labels.

GEFCom2014 does not provide site coordinates in the mirror used here. The public adaptation therefore replaces the physical clear-sky index with an empirical radiationenvelope index. For each site, calendar month, and hour, the 95th percentile of training-period forecast surface irradiance defines $e _ { s , m , h } .$ . The index is

$$
k ^ { ( e ) } = \mathrm { c l i p } \left( \frac { \mathrm { S S R D } } { \operatorname* { m a x } ( e _ { s , m , h } , 2 0 ) } , 0 , 2 \right) .\tag{6}
$$

This quantity is not a physical clear-sky index. Its denominator uses only the base-training period. The supplied cloud fraction is multiplied by 100 and hourly precipitation in metres by 1,000 before applying the existing thresholds. An origin-regime ablation repeats the last historical regime over all four horizons while keeping the base forecasts unchanged.

## 3.5. Bounded family-level weather gate

Static weights summarize performance over the fitting period, but they cannot respond to a change in weather regime. The gate adds this response without retraining the predictors. Let �(�) map each predictor to a family. Its weighted contribution and weight mass are

$$
C _ { s , t , g , h } = \sum _ { m : g ( m ) = g } \bar { w } _ { s , m , h } \widehat { y } _ { s , t + h } ^ { ( m ) } ,\tag{7}
$$

$$
B _ { s , g , h } = \sum _ { m : g ( m ) = g } \bar { w } _ { s , m , h } .\tag{8}
$$

For regime $z ,$ horizon $h ,$ and family $^ { g , }$ an unconstrained parameter $b _ { z , h , g }$ is bounded and centered:

$$
u _ { z , h , g } = 0 . 7 5 \operatorname { t a n h } ( b _ { z , h , g } ) ,\tag{9}
$$

$$
\delta _ { z , h , g } = u _ { z , h , g } - G ^ { - 1 } \sum _ { j = 1 } ^ { G } u _ { z , h , j } .\tag{10}
$$

The gated forecast becomes

$$
\hat { y } _ { s , t + h } ^ { ( z ) } = \mathrm { c l i p } \Bigg ( \frac { \sum _ { g } e ^ { \delta _ { z , h , g } } C _ { s , t , g , h } } { \operatorname* { m a x } ( \sum _ { g } e ^ { \delta _ { z , h , g } } B _ { s , g , h } , 1 0 ^ { - 7 } ) } , 0 , 1 . 2 5 \Bigg )\tag{11}
$$

This renormalized form preserves nonnegative constituent weights. A zero gate exactly recovers the static combination for forecasts within the clipping interval; this identity is checked against the saved implementation. The source also clips exponent arguments to [−2, 2] during application. The gate is fitted by minimizing

![](images/f581c6a90bd22633d99951c3864d301c6f7c7e774c2f19de2738d5b1ccb3d6bd.jpg)  
Chronological sequence: base training → checkpoint selection → weight fit → selection → confirmation → test Failed confirmation: retain the static ensemble. Test diagnostics do not trigger another search.  
Figure 1: Forecast combination and evaluation. Base predictors are frozen before weight learning. Weather availability must be established at the forecast origin. The schematic was drawn with reproducible Python code prepared with OpenAI Codex assistance (OpenAI, September 2026).

$$
\mathcal { L } = \frac { 1 } { \vert \Omega \vert } \sum _ { ( s , t , h ) \in \Omega } \vert \widehat { y } _ { s , t + h } ^ { ( z ) } - y _ { s , t + h } \vert + \lambda \operatorname* { m e a n } ( \delta ^ { 2 } ) ,\tag{12}
$$

where Ω contains valid fitting targets. The archived gate has $3 \times 4 8 \times 2 0 = 2 , 8 8 0$ stored coeficients. The public adaptation has $3 \times 4 \times 1 4 = 1 6 8$ because each constituent is its own family in a run.

## 3.6. Selection and confirmation

The gate is accepted only if its improvement persists beyond the period used to select its regularization. To make this check possible, the public pipeline separates base training, checkpoint selection, weight fitting, parameter selection, confirmation, and testing in chronological order. The shrinkage grid is {250, 500, 1000, 2000, 4000} and the gate regularization grid is {0.01, 0.1, 1}. The archived V9 compared three regime definitions; the public experiments fix its retained three-regime definition before testing.

The confirmation period is divided into three consecutive blocks. A candidate passes only if it improves MAE in all three blocks and the mean improvement is at least $1 0 ^ { - 6 } .$ Once choices are frozen, weights are refitted on the combined fitting, selection, and confirmation periods. The primary safeguard falls back to static weights if confirmation fails. For transparency, the experimental tables also report the rejected candidate’s test error as a diagnostic. These diagnostic errors do not trigger additional tuning.

## 3.7. Fixed-weight member-removal diagnostic

To examine dependence on the fitted bank, we remove a member or group � from the static combination and renormalize the remaining station–horizon weights:

$$
\widehat { y } _ { s , t + h } ^ { ( - R ) } = \frac { \sum _ { m \notin R } \bar { w } _ { s , m , h } \widehat { y } _ { s , t + h } ^ { ( m ) } } { \sum _ { m \notin R } \bar { w } _ { s , m , h } } .\tag{13}
$$

The retained weight mass is positive in every evaluated case.   
We report $\Delta _ { R } = \mathrm { s M A E } ( \widehat { y } ^ { ( - R ) } ) - \mathrm { s M A E } ( \widehat { y } ^ { ( 0 ) } )$ , so positive values mean that removal worsens the current combination.   
Base predictions and remaining relative weights are fixed;   
neither refitting nor gating is performed.

The diagnostic covers each of the fourteen members, all nine neural members together, and all five classical members together. The full static reference plus sixteen removals produce seventeen scenarios for each dataset and seed. This analysis was specified after the existing test results had been inspected and is explicitly exploratory. It measures dependence on frozen predictions, not the best performance of a smaller retrained bank, an optimal number of models, or independent evidence for test-driven pruning.

## 4. Experiments

## 4.1. Evidence scope

The evaluation has three distinct settings. The private five-minute V9 archive verifies implementation identity and the reported score, without retraining its full bank. The hourly public study tests the retained combination mechanism on GEFCom2014 and, with stronger information and label controls, PVDAQ, OPSD, and Ausgrid. The added fixedweight removals use those already inspected public tests as exploratory diagnostics. Finally, a separate native fifteenminute private case examines a matched member replacement and its measured inference cost. Absolute scores and member counts are not ranked across these settings because targets, inputs, fitting histories, and normalization difer.

## 4.2. Archived private case study

The retained V9 configuration is identified by the saved summary, the weights array, and its matching training log. Its core source snapshot has SHA-256 prefix ba5c8010b517178e. The weight array has dimensions (28, 82, 48) and the gate array (3, 48, 20). Independent checks verify finite nonnegative weights, normalization, zero-gate identity, and the arithmetic relating nMAE to the reported percentage.

Table 1  
Archived private results on the common cleaned target population. No new private-data rerun is claimed.
<table><tr><td>Configuration</td><td>nMAE</td><td>Score  $( \% )$ </td></tr><tr><td>Frozen reference</td><td>0.08181196</td><td>91.818804</td></tr><tr><td>V9 static control</td><td>0.08178049</td><td>91.821951</td></tr><tr><td>V9 gate (retained)</td><td>0.08173462</td><td>91.826538</td></tr></table>

The archived snapshot is frozen at 25 August 2026 and contains 441,844 training windows, 184,980 validation windows, and 114,262 test windows. All 48 target timestamps of each evaluated window lie between 06:00 and 20:00. The test contains 5,484,576 window–horizon targets, including repeated timestamps arising from overlapping forecasting origins. These are not 5.48 million independent observations.

The V9 gain over its recomputed static control is 0.004587 percentage points in the reported score, corresponding to approximately 0.0561% relative nMAE reduction.

The private results in Table 1 are archived evidence. Neither all 72 base models nor their full private test predictions were regenerated for this manuscript. Earlier rawdata and cleaned-data scores use diferent target sets and are excluded from an improvement calculation. Moreover, several historical versions inspected the same private test period. The private test is consequently not a globally untouched benchmark across the entire development history. Its normalization provenance and target-time historical-weather availability remain additional limits on operational interpretation.

## 4.3. GEFCom2014 dataset and forecasting protocol

The public experiment uses solar\_new.csv from the published research repository accompanying Dumas et al. [8]. The downloaded commit and file hashes are stored with the experiment. The file contains 59,040 complete hourly records across three solar sites, with 19,680 records per site from 2 April 2012 01:00 to 1 July 2014 00:00. Duplicate timestamps and missing numeric values are absent in this file. POWER is already normalized by capacity; the observed value slightly above one at one site is retained.

The source preprocessing diferences accumulated radiation and precipitation fields and shifts the complete site series by ten rows. The present experiment uses the supplied processed timestamps and does not apply another time-zone shift. Calendar-hour filtering therefore refers to the mirror’s clock. The 12 meteorological channels include cloud water and ice, pressure, humidity, cloud cover, wind components, temperature, surface and top-of-atmosphere radiation, and precipitation.

Weather scaling uses training-only means and standard deviations; standardized channels are clipped to [−10, 10]. Neither public power nor its evaluation denominator is rescaled using test observations. A valid forecasting window contains 24 past hours and four subsequent target hours; only the targets must lie within the clock interval [06∶00, 20∶00). Targets crossing a partition boundary are excluded, while earlier observations may legitimately serve as historical inputs for a later forecast.

The final test covers January–June 2014 and contains 5,973 windows, or 23,892 window–horizon targets. Each site contributes 1,991 windows. All methods use the same targets. This is a within-site temporal evaluation across three public sites, not a held-out-site experiment. The public NWP variables are treated as supplied forecast covariates. Because their issue timestamps are absent, the experiment does not independently certify the latest forecast vintage available at every origin.

## 4.4. Baselines and implementation

The nine neural constituents are LSTM, TimeMixer-style, Transformer-style, Mamba-style, PatchTST-style, TiDE-style, TSMixer-style, ModernTCN-style, and TFT-style implementations from the project. Each uses a hidden dimension of 64, dropout 0.1, a 24-hour history, and the available weather and calendar inputs. PatchTST-style inputs use an eighthour patch and a four-hour stride; TimeMixer scales are 1, 2, 4, 8. The original implementations are preserved in the reproducibility package, including their defaults and taskspecific modifications.

For each seed, base models are trained for 12 epochs with AdamW, learning rate 0.002, weight decay $1 0 ^ { - 4 }$ , batch size 256, and gradient-norm clipping at 1. Each epoch visits the available training windows in a randomly permuted order; the configured 16,000-window cap exceeds the 13,992 available windows. Checkpoints are selected by the independent earlystopping period rather than by test error. Seeds 2026, 2027, and 2028 define the three runs, with fixed family-specific ofsets.

The classical constituents are last-value persistence, previous-day same-hour persistence, training-only hourly median climatology, ridge regression with penalty 10, and histogram gradient boosting. Ridge and boosting use the historical power vector, future forecast covariates and calendar features, and site indicators. Four boosting regressors use absolute-error loss, 150 iterations, at most 15 leaves, $L _ { 2 }$ regularization 1, and no internal early-stopping split. These deterministic classical predictions are shared across the three neural seeds. Equal per-member epochs limit neural fitting but do not match total bank training, hyperparameter search, or hardware costs to those of a single regression control. Base, static-ensemble, and gated-ensemble forecasts are clipped to [0, 1.25] consistently where models can exceed this range.

Weight optimization reuses the project’s Adam-based simplex solver for 800 steps at learning rate 0.08. The gate uses the original 600-step routine, learning rate 0.03, and batches of up to 4,096 windows. Training runs locally with Python 3.11.16, PyTorch 2.6.0+cu124, and an NVIDIA GeForce RTX 3060 Laptop GPU.

Experiment and figure scripts were prepared with OpenAI Codex assistance (OpenAI, September 2026) and preserved with software and data hashes. Data plots were generated from saved predictions; independent routines reconstructed input windows, reported errors, and ensemble outputs.

Table 2  
Chronological public partitions. Complete forecast targets must remain inside a partition.
<table><tr><td>Purpose</td><td>Period</td><td>Windows</td></tr><tr><td>Base training</td><td>2 Apr 2012–31 May 2013</td><td>13,992</td></tr><tr><td>Checkpoint selection</td><td>June 2013</td><td>990</td></tr><tr><td>Weight fitting</td><td>July-September 2013</td><td>3,036</td></tr><tr><td>Hyperparameter selection</td><td>October-November 2013</td><td>2,013</td></tr><tr><td>Confirmation</td><td>December 2013</td><td>1,023</td></tr><tr><td>Final test</td><td>January-June 2014</td><td>5,973</td></tr></table>

## 4.5. Metrics and uncertainty

For the common target population Ω, the main metrics are

$$
\mathrm { \ n M A E } = | \Omega | ^ { - 1 } \sum _ { \Omega } | \widehat { y } - y | ,\tag{14}
$$

$$
{ \mathrm { n R M S E } } = { \sqrt { | \Omega | ^ { - 1 } \sum _ { \Omega } ( { \widehat { y } } - y ) ^ { 2 } } } .\tag{15}
$$

The project’s score is $A = 1 0 0 ( 1 \mathrm { ~ - ~ } \mathrm { n M A E } )$ . It is a transformed regression error, not a classification accuracy or the percentage of forecasts within a tolerance. An additional active-output metric evaluates targets with $y > 0 . 0 2$ ; this is a retrospective reporting subset, not a weather classifier or a rule for selecting the main test set.

Tables report means and sample standard deviations over three neural seeds. For the gate–static comparison, losses are averaged over seeds and aggregated by forecast-origin calendar day. A paired bootstrap resamples 181 complete day blocks 2,000 times. Each block retains all sites and overlapping horizons, reducing the false precision of pointwise resampling. The interval is conditional on this test period and the fitted models; it does not fully model dependence across consecutive days or uncertainty from alternative temporal splits.

## 4.6. Original GEFCom2014 comparison results

The static ensemble achieves nMAE 0.062028, compared with 0.066130 for gradient boosting, a relative reduction of 6.20%. The nine neural constituents range from approximately 0.06747 to 0.07217 in mean nMAE. Thus, no individual neural family beats gradient boosting under the fixed budget, while the learned heterogeneous combination improves over both. Equal weighting is weaker than learned static weighting, indicating that diversity alone is insuficient when constituent error levels difer substantially. This is the original benchmark comparison: its regression controls omit historical weather channels and do not receive the later labels used for ensemble fitting. These diferences limit architectural attribution of the reported improvement. The added experiments below use complete-feature, matchedlabel controls.

The target-weather gate has mean nMAE 0.062021 and score 93.797912%. Its improvement over static weights is only 0.00000754 in normalized MAE, or approximately 0.000754 percentage points in the score. The 95% day-block bootstrap interval is [−0.00001381, 0.00002845], which includes zero. In addition, the gate fails confirmation for each of the three training seeds. The predeclared confirmationselected rule consequently uses the static ensemble in every run.

The origin-regime ablation has a slightly lower average test MAE than the target-regime gate. It shares the same NWP-driven base predictions and changes only the gate context. It therefore does not measure the removal of all future weather information. Nor is it promoted as a new winning configuration after observing these test results. Together with the confirmation failures and the bootstrap interval, this ablation indicates that the additional weather-conditioning efect is weak in the present public setting.

## 4.7. Additional datasets and retrospective weather condition

Three further public sources extend the evaluation while retaining the V9 combination mechanism. The Ausgrid release supplies residential generation in Australia [3, 24]; the OPSD household release supplies original German electricitymeter feeds [23, 28]; and PVDAQ supplies measured AC power from system 34 in Las Vegas [7]. These are distinct data sources, rather than partitions of the GEFCom mirror. Site selection uses source identity and data quality before model scoring: three eligible Ausgrid households, three OPSD households with adequate common coverage, and the specified PVDAQ installation. This is temporal evaluation within the selected sites, not a held-out-site transfer experiment.

Ausgrid half-hourly generation energy is converted to hourly mean power only when both measurements are present and the source marks the day as measured. Ambiguous or nonexistent daylight-saving timestamps are excluded. OPSD labels are reconstructed from original cumulative kWh readings, rather than its interpolated hourly release. The energy diference is divided by the actual endpoint duration; endpoints must precede their nominal hour boundaries by no more than 180 seconds, and internal observation gaps above 300 seconds or counter anomalies invalidate the label. Consequently, these are approximately hourly averages whose actual endpoints remain available for audit. PVDAQ uses the oficial power-unit conversion and requires four unique, finite observations at exact quarter-hour clock times. Signed AC readings, including small negative values, are retained.

Table 3  
Public benchmark on the same 5,973 test windows. Mean and sample standard deviation over three neural seeds.
<table><tr><td>Method</td><td>nMAE</td><td>nRMSE</td><td>Active-output nMAE</td><td>Score (%)</td></tr><tr><td>Persistence</td><td> $0 . 2 3 2 8 6 1 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 3 0 6 3 2 4 \pm 0 . 0 0 0 0 0 0$ </td><td>0.242035</td><td>76.7139</td></tr><tr><td>Daily persistence</td><td> $0 . 1 2 5 9 9 3 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 2 0 2 7 6 9 \pm 0 . 0 0 0 0 0 0$ </td><td>0.140266</td><td>87.4007</td></tr><tr><td>Climatology</td><td> $0 . 1 5 3 2 9 3 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 2 1 9 1 3 0 \pm 0 . 0 0 0 0 0 0$ </td><td>0.168680</td><td>84.6707</td></tr><tr><td>Ridge regression</td><td> $0 . 0 8 3 1 0 4 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 1 6 1 2 3 \pm 0 . 0 0 0 0 0 0$ </td><td>0.090445</td><td>91.6896</td></tr><tr><td>Gradient boosting</td><td> $0 . 0 6 6 1 3 0 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 0 6 9 5 1 \pm 0 . 0 0 0 0 0 0$ </td><td>0.073832</td><td>93.3870</td></tr><tr><td>LSTM</td><td> $0 . 0 7 1 0 0 6 \pm 0 . 0 0 0 5 3 7$ </td><td> $0 . 1 1 2 7 9 7 \pm 0 . 0 0 0 4 6 2$ </td><td>0.078321</td><td>92.8994</td></tr><tr><td>TimeMixer-style</td><td> $0 . 0 6 9 7 4 4 \pm 0 . 0 0 0 7 5 2$ </td><td> $0 . 1 1 1 5 2 2 \pm 0 . 0 0 0 8 6 0$ </td><td>0.077052</td><td>93.0256</td></tr><tr><td>Transformer-style</td><td> $0 . 0 7 0 4 4 0 \pm 0 . 0 0 2 1 0 3$ </td><td> $0 . 1 1 2 7 0 1 \pm 0 . 0 0 3 4 4 0$ </td><td>0.078129</td><td>92.9560</td></tr><tr><td>Mamba-style</td><td> $0 . 0 6 7 6 5 6 \pm 0 . 0 0 0 3 1 2$ </td><td> $0 . 1 0 9 4 9 4 \pm 0 . 0 0 0 8 6 8$ </td><td>0.074763</td><td>93.2344</td></tr><tr><td>PatchTST-style</td><td> $0 . 0 7 2 1 6 5 \pm 0 . 0 0 3 4 7 0$ </td><td> $0 . 1 1 1 0 4 2 \pm 0 . 0 0 4 8 4 3$ </td><td>0.079617</td><td>92.7835</td></tr><tr><td>TiDE-style</td><td> $0 . 0 6 7 4 6 9 \pm 0 . 0 0 1 0 1 9$ </td><td> $0 . 1 0 8 0 9 6 \pm 0 . 0 0 2 8 6 4$ </td><td>0.074778</td><td>93.2531</td></tr><tr><td>TSMixer-style</td><td> $0 . 0 6 7 5 3 4 \pm 0 . 0 0 1 5 3 7$ </td><td> $0 . 1 0 8 2 0 3 \pm 0 . 0 0 2 2 2 2$ </td><td>0.074954</td><td>93.2466</td></tr><tr><td>ModernTCN-style</td><td> $0 . 0 6 8 8 1 9 \pm 0 . 0 0 1 1 7 0$ </td><td> $0 . 1 0 9 1 7 1 \pm 0 . 0 0 1 4 7 5$ </td><td>0.076030</td><td>93.1181</td></tr><tr><td>TFT-style</td><td> $0 . 0 6 9 4 9 8 \pm 0 . 0 0 0 8 6 0$ </td><td> $0 . 1 1 2 1 8 5 \pm 0 . 0 0 0 8 0 9$ </td><td>0.077275</td><td>93.0502</td></tr><tr><td>Uniform ensemble</td><td> $0 . 0 6 9 8 3 3 \pm 0 . 0 0 0 1 7 2$ </td><td> $0 . 1 0 4 3 2 8 \pm 0 . 0 0 0 2 5 6$ </td><td>0.075985</td><td>93.0167</td></tr><tr><td>Static ensemble</td><td> $0 . 0 6 2 0 2 8 \pm 0 . 0 0 0 2 6 2$ </td><td> $0 . 1 0 2 1 2 7 \pm 0 . 0 0 0 6 0 0$ </td><td>0.068987</td><td>93.7972</td></tr><tr><td>V9 gate (hourly adaptation)</td><td> $0 . 0 6 2 0 2 1 \pm 0 . 0 0 0 2 6 3$ </td><td> $0 . 1 0 2 1 3 4 \pm 0 . 0 0 0 5 9 7$ </td><td>0.068984</td><td>93.7979</td></tr><tr><td>Origin-regime gate</td><td> $0 . 0 6 2 0 1 2 \pm 0 . 0 0 0 2 6 4$ </td><td> $0 . 1 0 2 1 1 2 \pm 0 . 0 0 0 6 0 3$ </td><td>0.068969</td><td>93.7988</td></tr><tr><td>Confirmation-selected rule</td><td> $0 . 0 6 2 0 2 8 \pm 0 . 0 0 0 2 6 2$ </td><td> $0 . 1 0 2 1 2 7 \pm 0 . 0 0 0 6 0 0$ </td><td>0.068987</td><td>93.7972</td></tr></table>

The style sufix denotes a repository implementation. Classical baselines are deterministic and shared across runs. The candidate gate and origin-regime variant are diagnostics; the confirmation-selected rule uses static weights in all runs.

![](images/cfd0e82b102f4385bbf4b27489a7f6aaa81fb035b9cc798e27b4a8ef20b3ea1d.jpg)

![](images/7cdd8dcccbb68769eef5e0e90828343cc275f986d67b4919584549926a938d9f.jpg)  
Figure 2: Public-data comparison. Left: selected methods’ mean nMAE; error bars show the sample standard deviation across three neural seeds. Deterministic baselines have no seed variation. Right: per-horizon nMAE for the strongest classical constituent and ensemble variants. All curves use identical test windows.

Its instrument averaging alignment and historical daylightsaving behavior are not independently certified.

All three sources are paired with ERA5 reanalysis [5, 11] retrieved through Open-Meteo with the ERA5 model explicitly requested [31]. Eight covariates comprise shortwave radiation, cloud cover, precipitation, temperature, relative humidity, surface pressure, wind speed, and wind direction. Radiation is the preceding-hour mean and precipitation the preceding-hour total; the other fields are instantaneous values. UTC weather labels are aligned to nominal power-interval ends. Ausgrid uses postcode reference coordinates and OPSD a common Konstanz regional point; neither represents a verified rooftop sensor location. PVDAQ uses its published site coordinates. Requests, returned grids, elevation settings, units, responses, and checksums are archived.

These additional experiments are retrospective ERA5- assisted evaluations. Target-period reanalysis is supplied equally to the neural models and complete-feature regression controls, but it is unavailable as such at a real forecast origin. The results therefore test the retained architecture under a disclosed oracle-weather condition; they do not validate an operational NWP forecast pipeline. The observed PVDAQ weather is retained in the source data but is not silently substituted for ERA5 or presented as a forecast.

Table 4  
Gate selection and final test diagnostics. Positive test gain means lower gated nMAE.
<table><tr><td>Seed</td><td>K</td><td>λ</td><td>Static nMAE</td><td>Gated nMAE</td><td>Test gain</td><td>Confirmation</td></tr><tr><td>2026</td><td>1000</td><td>0.1</td><td>0.06174977</td><td>0.06174567</td><td>+0.00000410</td><td>Reject</td></tr><tr><td>2027</td><td>4000</td><td>1.0</td><td>0.06226879</td><td>0.06226920</td><td>-0.00000041</td><td>Reject</td></tr><tr><td>2028</td><td>4000</td><td>0.01</td><td>0.06206670</td><td>0.06204777</td><td>+0.00001893</td><td>Reject</td></tr></table>

Table 5  
Dataset coverage and disjoint chronological evaluation stages.
<table><tr><td>Method quantity</td><td>PVDAQ</td><td>OPSD</td><td>Ausgrid</td></tr><tr><td>Stations</td><td>1</td><td>3</td><td>3</td></tr><tr><td rowspan="2">Hourly end-label range (UTC)</td><td>2011-01-01 09:00</td><td>2016-03-01 01:00</td><td>2011-06-3015:00</td></tr><tr><td>to 2014-01-01 08:00</td><td>to 2018-01-01 00:00</td><td>to 2013-06-30 14:00</td></tr><tr><td>Valid/ total station-hours</td><td>25,405/26,304 (96.58%)</td><td>46,073/48,312 (95.37%)</td><td>52,608/52,632 (99.95%)</td></tr><tr><td>Train windows</td><td>4,343</td><td>9,217</td><td>11,955</td></tr><tr><td>Checkpoint windows</td><td>1,040</td><td>1,596</td><td>2,406</td></tr><tr><td>Weight Fit windows</td><td>1,031</td><td>1,712</td><td>2,376</td></tr><tr><td>Selection windows</td><td>605</td><td>976</td><td>1,188</td></tr><tr><td>Confirmation windows</td><td>526</td><td>979</td><td>1,218</td></tr><tr><td>Test windows</td><td>2,329</td><td>3,516</td><td>4,785</td></tr><tr><td>Test target points</td><td>9,316</td><td>14,064 2017-08-20 05:00</td><td>19,140</td></tr><tr><td rowspan="2">Test target-label range (UTC)</td><td>2013-05-2714:00 to 2014-01-01 04:00</td><td></td><td>2013-02-04 20:00</td></tr><tr><td></td><td>to 2017-12-31 19:00</td><td>to 2013-06-30 10:00</td></tr></table>

All tasks use 24 hourly historical values and four subsequent hourly targets; complete target interval midpoints must be in local [06:00,20:00). The grid retains missing hours; windows containing missing input or target values are excluded. Raw valid-hour counts include signed observations where retained by the source audit. Target stages are disjoint, whereas rolling windows within a stage overlap. The hourly end-label range uses inclusive first and last labels. Normalization uses each station’s training maximum, not installed capacity; weather inputs are retrospective ERA5.

## 4.8. Additional protocol and matched controls

Each dataset uses the same 24-hour history and four-hour target duration. All 28 power observations must be present on the hourly UTC grid; no missing history or target is interpolated. Only the four target interval midpoints must lie in the local clock range [06∶00, 20∶00). A first target ending at � corresponds to an issue time of � − 1 hour. The common time range is divided chronologically into 50% base training, 10% checkpoint selection, 10% weight fitting, 5% parameter selection, 5% confirmation, and 20% test. Complete target intervals must stay within their partition. Earlier observations can enter later rolling histories.

For these sources, $c _ { s }$ is the maximum valid power within the first training partition. Errors normalized by this training scale are denoted sMAE and sRMSE, distinguishing them from the capacity-normalized GEFCom metrics. Neither the denominator nor weather standardization uses later partitions. Labels remain unmodified, while predictions from every constituent and combination are clipped to [0, 1.25]. Physicalunit MAE and RMSE, training scales, and per-site and perhorizon errors are also archived.

The original public adaptation’s nine neural families, five classical constituents, hyperparameters, three seeds, and 12-epoch budget are retained. Weather dimensionality changes from 12 to eight, and site embeddings are fitted anew. The bank’s ridge and gradient-boosting constituents are trained only on the base-training partition, ensuring that their weight-fitting predictions are out of sample. Both now receive the complete flattened historical power/weather, future weather/calendar, and site indicators. Separate matched controls select ridge penalty from {0.1, 1, 10, 100} or boosting iterations from {75, 150, 300} using the checkpointselection partition, then refit on the union of base training and the three ensemble fitting/selection/confirmation partitions. They thereby receive the labels accessible to the final refitted ensemble. These controls are not inserted into the ensemble bank. Oficial-author DLinear code [29], with the same neural epoch budget, provides an additional historical-power-only reference; it does not share the meteorological information set.

The original simplex solvers, shrinkage candidates, bounded gate, regularization candidates, and confirmation rule are unchanged. The empirical radiation envelope uses only training-period monthly/hourly values. Cloud cover is already in percent and precipitation in millimetres, requiring no GEFCom-specific unit multiplier. Confirmation is divided into three chronological folds; the candidate gate must improve in every fold. After the original prescribed refitting, all candidates and the confirmation-selected fallback are scored on the common test set.

Two primary contrasts are fixed for each dataset: static fusion against matched boosting, and gated against static fusion. Losses are averaged across seeds before 10,000 circular seven-calendar-day bootstrap replicates, keeping all sites and overlapping horizons together. We report descriptive 95% intervals and Bonferroni intervals for these six primary contrasts. Additional ablation contrasts are exploratory. All intervals are conditional on these sites, temporal splits, and fitted runs.

Table 6  
Additional-data comparison of the V9-based public-data adaptation.
<table><tr><td>Method / quantity</td><td>PVDAQ</td><td>OPSD</td><td>Ausgrid</td></tr><tr><td>Gradient boosting (matched)</td><td> $0 . 0 4 0 7 0 4 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 5 9 1 5 5 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 6 7 8 7 5 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Ridge (matched)</td><td> $0 . 0 4 7 2 6 9 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 6 8 5 9 4 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 8 0 0 6 7 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Static ensemble</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 4 8 \pm 0 . 0 0 0 3 7 7$ </td><td> $0 . 0 6 6 7 6 7 \pm 0 . 0 0 0 2 8 5$ </td></tr><tr><td>V9 regime gate</td><td> $0 . 0 4 0 1 3 7 \pm 0 . 0 0 0 2 6 9$ </td><td> $0 . 0 5 6 6 1 7 \pm 0 . 0 0 0 4 4 2$ </td><td> $0 . 0 6 6 7 2 4 \pm 0 . 0 0 0 3 0 4$ </td></tr><tr><td>Confirmation-selected rule</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 4 7 \pm 0 . 0 0 0 3 7 7$ </td><td> $0 . 0 6 6 7 6 7 \pm 0 . 0 0 0 2 8 5$ </td></tr></table>

Entries are test sMAE mean ± sample standard deviation across three training seeds. Each station is normalized by its training-period maximum; sMAE is not capacity-normalized nMAE. Lower is better. Future ERA5 is a retrospective oracle input, not an operational weather forecast. Matched $\mathsf { R i d g e / G B }$ use the same complete feature information and final fitting-label access as the ensemble.

## 4.9. Component ablations

The ablations share the same frozen bank and final ensemble-fitting labels. Removing the gate yields the static ensemble. Removing shrinkage retains the fitted local station– horizon weights; removing site dependence uses only global horizon weights; equal weighting removes weight fitting altogether. A final variant supplies the last historical regime to the gate over every horizon, retaining the full gate’s selected regularization and the same ERA5-informed base predictions. It isolates gate context, not removal of all future weather information. Individual bank members provide complementary component comparisons.

PVDAQ contains one site. Its local and global objectives coincide, so the shrinkage and site-dependence ablations are degenerate there and cannot establish a cross-site benefit. Those mechanisms must be assessed on the two multihousehold datasets.

## 4.10. Additional comparison and ablation results

The three ERA5-assisted benchmarks contain 10,630 test windows and 42,520 window–horizon targets. These counts refer to the shared test population, not the number of training seeds. Static fusion reduces sMAE relative to matched gradient boosting by 1.11% on PVDAQ, 4.41% on OPSD, and 1.63% on Ausgrid (Table 6).

The strength of this evidence difers across datasets. After correction for the six primary comparisons, only the OPSD static-fusion contrast excludes zero. The lower mean errors on PVDAQ and Ausgrid therefore do not establish the same level of support. Gating provides no supported incremental improvement on any of the three datasets. It passes the threefold confirmation rule in none of the PVDAQ or Ausgrid runs and in one of the three OPSD runs. Table 6 reports both the candidate gate and the rule that falls back to static fusion when confirmation fails.

Weight learning accounts for a clearer change than weather gating (Table 7). Learned static weights lower mean error relative to uniform averaging on all three datasets. On OPSD, removing shrinkage increases sMAE from 0.056548 to 0.056964, whereas shared global horizon weights give 0.056536. The corresponding Ausgrid errors are 0.066767, 0.066917, and 0.066717. Local weights alone thus perform worse than either shrinkage or global sharing in these two samples. The small diferences between shrinkage and global sharing provide little reason to prefer the more site-specific estimate. These comparisons are exploratory; the six-test correction applies only to the primary contrasts. PVDAQ’s identical local and global rows follow from its single-site design.

Changing the gate context produces smaller and less consistent efects. The target-period context uses an oracle, while the origin-context variant changes only the gate and retains the same oracle-informed base forecasts. This comparison cannot establish the performance of a system restricted to information available in real time. Table 11 reports every model’s sMAE; physical-unit, per-site, and per-horizon results are provided with the experiment records.

## 4.11. Dependence on individual members and groups

Removing all neural members or all classical members increases error on each added dataset (Table 8). Every groupremoval contrast worsens in all three seeds. The mean sMAE increases range from 0.001417 to 0.004949. Thus, both groups contribute to the particular frozen mixture, even though no individual neural model consistently outperforms the matched boosting control.

This group-level result does not imply that all fourteen constituents are indispensable. The bank’s gradient-boosting member has the largest mean single-removal deterioration on each dataset: 0.003721, 0.001827, and 0.001787 on PVDAQ, OPSD, and Ausgrid. In contrast, removing TiDE reduces OPSD sMAE by 0.000177, and removing persistence reduces Ausgrid sMAE by 0.000171. These are small, descriptive changes, not separately tested discoveries. Complete singlemember results appear in Table 12.

Across three datasets, three seeds, and seventeen scenarios, 153 diagnostic evaluations were reconstructed from frozen predictions. An independent complementary-sum implementation agrees in sMAE to within $2 . 8 \times 1 0 ^ { - 1 7 }$ , and all 27 source-file hashes remain unchanged. This verifies the calculation, not the independence of the already inspected test sets. No new member selection or training follows from these results.

Table 7  
Component ablations for the V9-based public-data adaptation.
<table><tr><td>Method / quantity</td><td>PVDAQ</td><td>OPSD</td><td>Ausgrid</td></tr><tr><td>V9 regime gate</td><td> $0 . 0 4 0 1 3 7 \pm 0 . 0 0 0 2 6 9$ </td><td> $0 . 0 5 6 6 1 7 \pm 0 . 0 0 0 4 4 2$ </td><td> $0 . 0 6 6 7 2 4 \pm 0 . 0 0 0 3 0 4$ </td></tr><tr><td>Static ensemble</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 4 8 \pm 0 . 0 0 0 3 7 7$ </td><td> $0 . 0 6 6 7 6 7 \pm 0 . 0 0 0 2 8 5$ </td></tr><tr><td>Station weights, no shrinkage</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 9 6 4 \pm 0 . 0 0 0 4 2 5$ </td><td> $0 . 0 6 6 9 1 7 \pm 0 . 0 0 0 2 5 2$ </td></tr><tr><td>Global horizon weights</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 3 6 \pm 0 . 0 0 0 3 4 3$ </td><td> $0 . 0 6 6 7 1 7 \pm 0 . 0 0 0 2 3 7$ </td></tr><tr><td>Uniform ensemble</td><td> $0 . 0 5 0 0 2 1 \pm 0 . 0 0 0 0 3 6$ </td><td> $0 . 0 6 6 0 5 2 \pm 0 . 0 0 0 3 1 1$ </td><td> $0 . 0 7 6 0 0 1 \pm 0 . 0 0 0 3 0 1$ </td></tr><tr><td>Origin-context gate</td><td> $0 . 0 4 0 3 2 4 \pm 0 . 0 0 0 1 9 9$ </td><td> $0 . 0 5 6 6 0 7 \pm 0 . 0 0 0 4 4 3$ </td><td> $0 . 0 6 6 7 8 0 \pm 0 . 0 0 0 2 6 8$ </td></tr></table>

Test sMAE mean ± three-seed sample standard deviation, using training-maximum normalization. All variants share the frozen predictor bank. Static removes the regime gate; no shrinkage removes the global-weight shrinkage; global horizon weights remove station-specific weighting; uniform uses equal weights. Origin-context gating changes only the gate context and retains oracle-informed base predictions and the main gate’s selected regularization. It is not a full causal-input ablation. These rows are secondary comparisons; the six-primary-contrast family-wise correction must not be generalized to them.

Table 8  
Fixed-weight group removal: mean sMAE over three seeds. Removing either group worsens every seed on each dataset. These are exploratory test diagnostics without refitting.
<table><tr><td>Retained forecasts</td><td>PVDAQ</td><td>OPSD</td><td>Ausgrid</td></tr><tr><td>Full static (14)</td><td>0.040252</td><td>0.056548</td><td>0.066767</td></tr><tr><td>Classical only (5)</td><td>0.041669</td><td>0.059649</td><td>0.070592</td></tr><tr><td>Neural only (9)</td><td>0.045201</td><td>0.059391</td><td>0.068401</td></tr></table>

Table 9

## 4.12. Secondary case: a native member replacement

The native case examines a separate engineering implementation denoted V28. It combines TiDE, PatchTST, LSTM, and HGB16, a histogram-gradient-boosting predictor with sixteen separately fitted horizon heads. HGB16 occupies one fusion slot, so four slots do not mean four equally sized models. The matched control retains the same three neural checkpoints and substitutes the original Transformer for HGB16. Both pipelines use the same development procedure: static convex fusion with regularization 0.1, followed by additive global-horizon and station-intercept median calibration. Horizon residual medians use shrinkage prior 256; station medians, pooled over horizons after the horizon correction, use prior 128. Each term is clipped at ±0.03 normalized units before shrinkage; their sum is clipped at the same bound, and final forecasts at [0, 1.25]. This is not the V9 weather gate.

Each window contains 96 historical fifteen-minute intervals and predicts sixteen subsequent interval-average powers. The historical and future feature arrays have 33 and 45 channels, respectively, including fifteen weather channels and time and availability features. The feed uses an assumed NWP availability delay of 370 minutes after its run time; independently verified operational arrival records are unavailable. The task-specific compact neural backbones use one training run under protocol seed 20260913, fixed model-specific initialization seeds, twenty epochs, and 125,776 training windows. Each HGB head uses 187 summary features, absoluteerror loss, 400 boosting iterations, learning rate 0.05, at most 31 leaves, minimum leaf size $^ { 6 4 , }$ and $L _ { 2 }$ regularization 1. These are matched-input pipeline comparisons, not matched total training budgets.

Native fifteen-minute case on previously inspected private targets. The first pair shares the complete fitting and calibration procedure. Remaining rows omit calibration and are diagnostic comparisons.
<table><tr><td>Configuration</td><td>nMAE</td></tr><tr><td>Four neural, calibrated V28 replacement, calibrated</td><td>0.076767 0.075446</td></tr><tr><td>Four neural, uncalibrated V28 replacement, uncalibrated V28 uniform, uncalibrated</td><td>0.077431 0.076057 0.076254</td></tr></table>

Frozen predictions are evaluated on 24 sites, 16,114 common windows, and 257,824 window–horizon targets from 1–18 July 2026. Both combinations use an identical valid target mask and the same engineering capacity-reference snapshot for nMAE; the snapshot’s nameplate provenance is not independently certified. This period was inspected during prior development; the analysis adds no new training or untouched test data. Paired bootstrap intervals use 20,000 circular three-day resamples, retaining sites and horizons together, with seven-day sensitivity analysis. These intervals describe the fixed historical cohort and cannot remove selection bias.

The calibrated replacement reduces nMAE by 1.72% relative to the matched four-neural pipeline. The transformed score 100(1−nMAE) rises from 92.323336% to 92.455413%, a gain of 0.132077 percentage points. Error falls at 22 sites and increases at two; every horizon aggregate improves. The descriptive 95% interval for the score gain is [0.05868, 0.20675] percentage points; the three-contrast adjusted interval is [0.04289, 0.22231]. Seven-day resampling also retains a positive interval. Eighteen days and one seed do not establish long-term stability. The improvement is also present without calibration, while uniform fusion and HGB16 alone are exploratory, uncalibrated references. An older system archive, V18, has lower same-target nMAE (0.072125), but diferent training and calibration prevent using it as a controlled component contrast. Thus, the replacement has the better score in the matched pair, not the highest score across all archived systems.

Table 10  
Measured inference and saved checkpoints. Latencies are medians in ms per window; batch-256 times are amortized. The two pipelines share hardware and timed inputs.
<table><tr><td>Quantity</td><td>Four neural</td><td>V28</td></tr><tr><td>Batch 1  $( \boldsymbol { \mathrm { m s } } / \boldsymbol { \mathrm { w i n d o w } } )$ </td><td>14.421</td><td>69.597</td></tr><tr><td>Batch 256  $( \mathsf { m s } / \mathsf { w i n d o w } )$ </td><td>0.0922</td><td>0.6533</td></tr><tr><td>Checkpoint files (bytes)</td><td>5,880,746</td><td>15,939,797</td></tr></table>

## 4.13. Measured inference cost of the replacement

Both complete combinations are timed on the same RTX 3060 Laptop GPU and host CPU, with two CPU threads, Python 3.12.13, PyTorch 2.11.0+cu128, and scikit-learn 1.9.0. The fixed sample comprises the first 256 checkpointselection windows, all from one site. Two warm-ups precede seven alternating repetitions. Timings include device copies, neural forwards, HGB feature construction and all sixteen heads, fusion, and calibration; they exclude loading, data retrieval, network service, and training. Batch-one and batch-256 outputs agree with frozen predictions within $2 \times 1 0 ^ { - 6 }$

Table 10 shows that the replacement is slower in this implementation. The median paired latency ratios are 4.71 for individual windows and 7.96 for batches of 256; these are medians of paired ratios, not ratios of the table’s median times. Saved checkpoint bytes also increase. File size includes serialization overhead and is not a measurement of RAM or GPU memory. The timing sample and hardware limit extrapolation to a multi-site server, and no energy or endto-end service measurement is claimed.

## 4.14. Discussion and limits of the evidence

The first question has a qualified positive answer. Learned static fusion lowers mean error relative to uniform averaging on the added datasets, but its advantage over a completefeature, matched-label boosting control survives the sixcomparison correction only on OPSD. The original GEFCom diference is less controlled because its regression baseline has fewer inputs and fitting labels. Neither comparison establishes that a large bank is always preferable to a welltuned single predictor.

For the second question, the added weather gate has no consistent incremental support. Global sharing performs close to or slightly better than local shrinkage in the two multisite additions, while group removals expose dependence on both neural and classical predictions in the current frozen combination. Some individual removals nevertheless reduce error. These observations distinguish the contribution of a group from the necessity of every member. A smaller retrained bank might recover the same information with diferent weights; the present removal analysis cannot answer that question.

The native replacement case answers the third question for one implementation and one previously inspected period. A tree-based member reduces error relative to the matched fourneural pipeline, including before calibration, while increasing measured inference latency and saved checkpoint size. A 69.6 ms single-window latency may still be compatible with a fifteen-minute issue interval. No service deadline, request volume, or monetary value of forecast error is specified, so the measured tradeof does not by itself establish deployment value. Nor were energy use, memory consumption, or total training and search costs measured.

Several limits apply across the study. The public tests use one temporal holdout per dataset and three seeds, with fixedbudget, task-adapted architectures. Matching input and fitting label access does not match all optimization opportunities or total compute. The added sources use training-maximum scaling, regional ERA5, and source-specific time and measurement assumptions. Their target-period reanalysis is a retrospective oracle; the GEFCom mirror also lacks forecastissuance metadata. The private archives have further normalization and development-history limits. The 82-member archive, fourteen-member hourly study, and native four-slot case therefore do not form a common accuracy–complexity curve.

Finally, numerical reproducibility is distinct from independence of evaluation. Hashes and independent recomputation support the reported numbers, but do not undo repeated inspection of a historical test set. The new removals are exploratory and the native bootstrap intervals are conditional on a reused period. Establishing a deployable reduced ensemble requires freezing member selection, calibration, and compute budgets using development data, then evaluating genuinely unexposed dates with verifiable forecast vintages. No such prospective validation is claimed here.

## 5. Conclusion

This study assessed the contribution of ensemble components to four-hour PV forecasting. Its primary public implementation retains the V9 weighting mechanism in a fourteen-member hourly bank, while treating the original 82-member result as an archive audit. Static fusion has corrected positive evidence against matched boosting on OPSD; weather gating does not show a consistent additional benefit. Exploratory removals reveal dependence on both neural and classical groups, without establishing that all fourteen members are necessary.

A separate native fifteen-minute comparison finds a 1.72% relative nMAE reduction after one member replacement, accompanied by slower inference and larger saved checkpoints. Together, these results support reporting incremental predictive efects and measured costs, rather than using ensemble size or a transformed accuracy score as evidence of value.

The weather-oracle condition, limited temporal holdouts, and historical test reuse require further validation before claims of operational generalization or optimal ensemble size can be made.

## Data and code availability

The public sources comprise the GEFCom2014 researcher mirror, Ausgrid generation data, original OPSD household feeds, and NREL PVDAQ system 34. Additional weather is ERA5 served through Open-Meteo. Download scripts preserve attribution, request parameters, response hashes, and preprocessing rules. The existing public reproducibility packages retain model snapshots, trained checkpoints, weights, predictions, ablations, and numerical audits. A separate revision supplement adds the fixed-weight removal analysis, its source hashes, and summarized native accuracy and timing evidence. It depends on the existing public prediction caches and is not a replacement for those packages. Private raw data, predictions, and checkpoints are not redistributed. The native case consequently has auditable local evidence but is not fully reproducible from the public supplement alone.

## A. Reproduction and evidence boundaries

The public runner is run\_public\_benchmark.py; its default settings define the reported three-seed protocol. Result tables are generated from the saved JSON and NPZ outputs. Validation choices are saved before test-loss evaluation. Restarting a completed run loads its stored results instead of searching for a new configuration. A changed protocol should use a new experiment directory and a new evaluation plan.

The retained private gate uses ordinary row-disjoint chronological subdivisions of its validation windows. The source split assertion checks window-row membership, not the independence of all target timestamps across these internal subdivisions. Public subdivisions additionally require complete target intervals to remain within their own boundaries. The private weights’ simplex constraints and zero-gate identity are verified, but these numerical checks do not certify every historical training decision, label-cleaning rule, or meteorological input vintage.

## B. Complete member-removal diagnostics

Table 12 reports every fixed-weight removal. Diferences are descriptive three-seed means on already inspected test periods. Entries displayed as approximately zero have absolute magnitude below 0.0005 in the table’s scaled units. No member is removed from the retained system on this basis.

Table 11  
Complete model comparison on the three additional datasets.
<table><tr><td>Method / quantity</td><td>PVDAQ</td><td>OPSD</td><td>Ausgrid</td></tr><tr><td>Persistence</td><td> $0 . 2 5 6 1 7 9 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 6 5 9 8 1 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 2 5 2 4 8 2 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Daily persistence</td><td> $0 . 0 6 5 7 0 6 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 1 2 6 1 7 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 2 6 3 4 3 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Training climatology</td><td> $0 . 1 1 6 0 9 8 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 6 9 7 4 3 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 1 5 4 3 0 8 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Ridge (bank)</td><td> $0 . 0 4 8 9 1 8 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 6 8 7 6 3 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 7 9 8 3 5 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Gradient boosting (bank)</td><td> $0 . 0 4 1 9 9 4 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 5 8 1 0 5 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 6 8 3 0 0 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Ridge (matched)</td><td> $0 . 0 4 7 2 6 9 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 6 8 5 9 4 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 8 0 0 6 7 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>Gradient boosting (matched)</td><td> $0 . 0 4 0 7 0 4 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 5 9 1 5 5 \pm 0 . 0 0 0 0 0 0$ </td><td> $0 . 0 6 7 8 7 5 \pm 0 . 0 0 0 0 0 0$ </td></tr><tr><td>LSTM</td><td> $0 . 0 5 1 4 1 2 \pm 0 . 0 0 1 4 6 8$ </td><td> $0 . 0 6 7 3 5 1 \pm 0 . 0 0 1 3 2 5$ </td><td> $0 . 0 7 5 9 2 0 \pm 0 . 0 0 0 7 7 9$ </td></tr><tr><td>TimeMixer</td><td> $0 . 0 4 8 1 1 9 \pm 0 . 0 0 1 8 0 6$ </td><td> $0 . 0 6 5 1 8 2 \pm 0 . 0 0 1 8 0 0$ </td><td> $0 . 0 7 1 5 2 8 \pm 0 . 0 0 0 4 4 7$ </td></tr><tr><td>Transformer</td><td> $0 . 0 4 7 4 6 3 \pm 0 . 0 0 1 6 0 1$ </td><td> $0 . 0 6 4 3 5 3 \pm 0 . 0 0 1 3 8 9$ </td><td> $0 . 0 7 3 4 9 2 \pm 0 . 0 0 0 9 4 1$ </td></tr><tr><td>Mamba-style</td><td> $0 . 0 4 8 9 5 8 \pm 0 . 0 0 2 8 4 4$ </td><td> $0 . 0 6 2 8 8 4 \pm 0 . 0 0 0 1 0 9$ </td><td> $0 . 0 7 0 9 2 9 \pm 0 . 0 0 0 8 0 0$ </td></tr><tr><td>PatchTST</td><td> $0 . 0 5 0 8 1 3 \pm 0 . 0 0 1 1 3 9$ </td><td> $0 . 0 7 0 0 0 3 \pm 0 . 0 0 2 8 4 6$ </td><td> $0 . 0 7 8 1 7 6 \pm 0 . 0 0 1 1 8 3$ </td></tr><tr><td>TiDE</td><td> $0 . 0 5 0 2 3 7 \pm 0 . 0 0 1 1 0 1$ </td><td> $0 . 0 6 5 0 8 7 \pm 0 . 0 0 0 9 1 0$ </td><td> $0 . 0 7 1 2 1 4 \pm 0 . 0 0 0 1 6 9$ </td></tr><tr><td>TSMixer</td><td> $0 . 0 4 9 4 3 6 \pm 0 . 0 0 0 6 6 1$ </td><td> $0 . 0 6 2 8 2 8 \pm 0 . 0 0 2 0 2 3$ </td><td> $0 . 0 7 1 8 4 9 \pm 0 . 0 0 2 2 8 7$ </td></tr><tr><td>ModernTCN</td><td> $0 . 0 4 7 1 2 9 \pm 0 . 0 0 0 7 1 0$ </td><td> $0 . 0 6 3 0 0 6 \pm 0 . 0 0 2 1 0 9$ </td><td> $0 . 0 7 0 7 9 1 \pm 0 . 0 0 1 2 8 4$ </td></tr><tr><td>TFT</td><td> $0 . 0 4 9 4 6 4 \pm 0 . 0 0 2 0 2 7$ </td><td> $0 . 0 6 3 5 5 7 \pm 0 . 0 0 2 0 4 2$ </td><td> $0 . 0 7 2 5 2 2 \pm 0 . 0 0 0 5 4 4$ </td></tr><tr><td>DLinear (power only)</td><td> $0 . 0 6 6 4 3 5 \pm 0 . 0 0 0 7 7 8$ </td><td> $0 . 0 9 2 6 8 5 \pm 0 . 0 0 0 3 6 2$ </td><td> $0 . 1 0 4 9 1 9 \pm 0 . 0 0 0 3 6 3$ </td></tr><tr><td>Uniform ensemble</td><td> $0 . 0 5 0 0 2 1 \pm 0 . 0 0 0 0 3 6$ </td><td> $0 . 0 6 6 0 5 2 \pm 0 . 0 0 0 3 1 1$ </td><td> $0 . 0 7 6 0 0 1 \pm 0 . 0 0 0 3 0 1$ </td></tr><tr><td>Global horizon weights</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 3 6 \pm 0 . 0 0 0 3 4 3$ </td><td> $0 . 0 6 6 7 1 7 \pm 0 . 0 0 0 2 3 7$ </td></tr><tr><td>Station weights, no shrinkage</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 9 6 4 \pm 0 . 0 0 0 4 2 5$ </td><td> $0 . 0 6 6 9 1 7 \pm 0 . 0 0 0 2 5 2$ </td></tr><tr><td>Static ensemble</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 4 8 \pm 0 . 0 0 0 3 7 7$ </td><td> $0 . 0 6 6 7 6 7 \pm 0 . 0 0 0 2 8 5$ </td></tr><tr><td>Origin-context gate</td><td> $0 . 0 4 0 3 2 4 \pm 0 . 0 0 0 1 9 9$ </td><td> $0 . 0 5 6 6 0 7 \pm 0 . 0 0 0 4 4 3$ </td><td> $0 . 0 6 6 7 8 0 \pm 0 . 0 0 0 2 6 8$ </td></tr><tr><td>V9 regime gate</td><td> $0 . 0 4 0 1 3 7 \pm 0 . 0 0 0 2 6 9$ </td><td> $0 . 0 5 6 6 1 7 \pm 0 . 0 0 0 4 4 2$ </td><td> $0 . 0 6 6 7 2 4 \pm 0 . 0 0 0 3 0 4$ </td></tr><tr><td>Confirmation-selected rule</td><td> $0 . 0 4 0 2 5 2 \pm 0 . 0 0 0 1 7 8$ </td><td> $0 . 0 5 6 5 4 7 \pm 0 . 0 0 0 3 7 7$ </td><td> $0 . 0 6 6 7 6 7 \pm 0 . 0 0 0 2 8 5$ </td></tr></table>

Entries are test sMAE mean ± sample standard deviation across three training seeds. Each station is normalized by its training-period maximum; sMAE is not capacity-normalized nMAE. Lower is better. Future ERA5 is a retrospective oracle input, not an operational weather forecast. DLinear uses historical power only; its information set difers from the weather-assisted models. The nine neural rows are original-repository model families retrained for the hourly task. Bank $\mathsf { R i d g e / G B }$ use the initial training stage; matched controls additionally fit the ensemble’s meta-stage labels. A zero standard deviation for deterministic controls does not mean zero statistical uncertainty.

## Table 12

Exploratory fixed-weight removal sensitivity on previously inspected test sets. Values are mean changes in sMAE multiplied by 10<sup>3</sup>; positive values indicate deterioration after removal. Remaining station–horizon weights are renormalized without refitting.
<table><tr><td>Removed member(s)</td><td>PVDAQ</td><td>OPSD</td><td>Ausgrid</td></tr><tr><td>LSTM</td><td>+0.006</td><td>+0.001</td><td>≈0</td></tr><tr><td>TimeMixer</td><td>+0.026</td><td>-0.003</td><td>+0.189</td></tr><tr><td>Transformer</td><td>+0.116</td><td>+0.132</td><td>-0.004</td></tr><tr><td>Mamba</td><td>≈0</td><td>-0.095</td><td>+0.015</td></tr><tr><td>PatchTST</td><td>+0.073</td><td>+0.135</td><td>+0.015</td></tr><tr><td>TiDE</td><td>≈0</td><td>-0.177</td><td>+0.119</td></tr><tr><td>TSMixer</td><td>+0.078</td><td>+0.179</td><td>+0.089</td></tr><tr><td>ModernTCN</td><td>-0.011</td><td>-0.007</td><td>+0.029</td></tr><tr><td>TFT</td><td>+0.197</td><td>+0.366</td><td>-0.033</td></tr><tr><td>Persistence</td><td>+0.002</td><td>-0.021</td><td>-0.171</td></tr><tr><td>Daily persistence</td><td>≈0</td><td>-0.003</td><td>≈0</td></tr><tr><td>Climatology</td><td>≈0</td><td>-0.004</td><td>-0.001</td></tr><tr><td>Ridge</td><td>+0.135</td><td>-0.022</td><td>-0.005</td></tr><tr><td>GB (bank)</td><td>+3.721</td><td>+1.827</td><td>+1.787</td></tr><tr><td>All neural (9)</td><td>+1.417</td><td>+3.101</td><td>+3.825</td></tr><tr><td>All classical (5)</td><td>+4.949</td><td>+2.843</td><td>+1.634</td></tr></table>

![](images/db479fe967e2b390cd56290981984b7898bb393f906d49f5310fe12f26052f39.jpg)  
Figure 3: Incremental gate gain relative to static ensembling. Positive values indicate lower gate nMAE. Individual seed gains and the paired calendar-day bootstrap interval show that the small aggregate improvement does not establish a stable benefit.

Positive values favour the candidate; descriptive 95% intervals  
![](images/e788e43c5054e6a4c3717090cd5e82e9ff08677f14e268c64e13abaf43f139cd.jpg)  
Figure 4: Paired efects on the additional datasets. Positive values indicate reduced sMAE for static fusion against matched boosting (left), or for gating against static fusion (right). Bars are descriptive 95% circular seven-day bootstrap intervals, conditional on the three fitted seeds. The separate six-primary-contrast correction is used for the inferential statements in the text. All panels use retrospective ERA5 covariates and training-maximum normalization.

## References

[1] Amarasinghe, P.A.G.M., Abeygunawardana, N.S., Jayasekara, T.N., Edirisinghe, E.A.J.P., Abeygunawardane, S.K., 2020. Ensemble models for solar power forecasting—a weather classification approach. AIMS Energy 8, 252–271. doi:10.3934/energy.2020.2.252.

[2] Antonanzas, J., Osorio, N., Escobar, R., Urraca, R., Martinez-de Pison, F., Antonanzas-Torres, F., 2016. Review of photovoltaic power forecasting. Solar Energy 136, 78–111. doi:10.1016/j.solener.2016. 06.069.

[3] Ausgrid, n.d. Solar home electricity data. Data set. URL: https://data.gov.au/data/dataset/nsw-solar-home-electricty-data. half-hourly gross-metered solar generation; study subset July 2011–June 2013. Publication date and dataset DOI not verified. CC BY 3.0 Australia.

[4] Chen, S.A., Li, C.L., Yoder, N., Arik, S.O., Pfister, T., 2023. Tsmixer: An all-mlp architecture for time series forecasting. doi:10.48550/arXiv. 2303.06053. arXiv preprint.

[5] Copernicus Climate Change Service (C3S), 2018. ERA5 hourly data on single levels from 1940 to present. Data set. URL: https://cds. climate.copernicus.eu/datasets/reanalysis-era5-single-levels, doi:10.24381/cds.adbb2d47. source data-set DOI. Study weather was obtained through Open-Meteo with models=era5, not downloaded directly from CDS.

[6] Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., Yu, R., 2023. Longterm forecasting with tide: Time-series dense encoder. doi:10.48550/ arXiv.2304.08424. arXiv preprint.

[7] Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., Jordan, D., 2021. Photovoltaic data acquisition (PVDAQ) public datasets. Data set. URL: https://data.openei.org/submissions/4568, doi:10.25984/ 1846021. system 34, 2011–2013 subset. CC BY 4.0.

[8] Dumas, J., Wehenkel, A., Lanaspeze, D., Cornélusse, B., Sutera, A., 2022. A deep generative model for probabilistic energy forecasting in power systems: normalizing flows. Applied Energy 305, 117871. doi:10.1016/j.apenergy.2021.117871.

[9] Friedman, J.H., 2001. Greedy function approximation: A gradient boosting machine. The Annals of Statistics 29. doi:10.1214/aos/ 1013203451.

[10] Gu, A., Dao, T., 2023. Mamba: Linear-time sequence modeling with selective state spaces. doi:10.48550/arXiv.2312.00752. arXiv preprint.

[11] Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., Thépaut, J.N., 2020. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146, 1999–2049. URL: https://doi.org/10.1002/qj.3803, doi:10.1002/qj.3803.

[12] Hewamalage, H., Ackermann, K., Bergmeir, C., 2023. Forecast evaluation for data scientists: common pitfalls and best practices. Data Mining and Knowledge Discovery 37, 788–832. doi:10.1007/ s10618-022-00894-5.

[13] Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural Computation 9, 1735–1780. doi:10.1162/neco.1997.9.8.1735.

[14] Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., Hyndman, R.J., 2016. Probabilistic energy forecasting: Global energy forecasting competition 2014 and beyond. International Journal of Forecasting 32, 896–913. doi:10.1016/j.ijforecast.2016.02.001.

[15] Husein, M., Gago, E., Hasan, B., Pegalajar, M., 2024. Towards energy eficiency: A comprehensive review of deep learning-based photovoltaic power forecasting strategies. Heliyon 10, e33419. doi:10. 1016/j.heliyon.2024.e33419.

[16] Iheanetu, K.J., 2022. Solar photovoltaic power forecasting: A review. Sustainability 14, 17005. doi:10.3390/su142417005.

[17] Jacobs, R.A., Jordan, M.I., Nowlan, S.J., Hinton, G.E., 1991. Adaptive mixtures of local experts. Neural Computation 3, 79–87. doi:10.1162/ neco.1991.3.1.79.

[18] Jordan, M.I., Jacobs, R.A., 1994. Hierarchical mixtures of experts and the em algorithm. Neural Computation 6, 181–214. doi:10.1162/neco.

1994.6.2.181.

[19] Lim, B., Arık, S.Ö., Loef, N., Pfister, T., 2021. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting 37, 1748–1764. doi:10.1016/j. ijforecast.2021.03.012.

[20] Mayer, M.J., Yang, D., 2023. Pairing ensemble numerical weather prediction with ensemble physical model chain for probabilistic photovoltaic power forecasting. Renewable and Sustainable Energy Reviews 175, 113171. doi:10.1016/j.rser.2023.113171.

[21] Mayer, M.J., Yang, D., Markovics, D., 2025. The complexity and dimensionality of making deterministic photovoltaic power forecasts from ensemble numerical weather prediction. Energy Conversion and Management 344, 120303. URL: https://doi.org/10.1016/j. enconman.2025.120303, doi:10.1016/j.enconman.2025.120303.

[22] Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J., 2022. A time series is worth 64 words: Long-term forecasting with transformers. doi:10.48550/arXiv.2211.14730. arXiv preprint.

[23] Open Power System Data, 2020. Data package household data. Data set, version 2020-04-15. URL: https://data.open-power-system-data. org/household\_data/2020-04-15/. primary measurement data from CoSSMic. Study labels derived from original meter feeds, not the interpolated hourly release. CC BY 4.0.

[24] Ratnam, E.L., Weller, S.R., Kellett, C.M., Murray, A.T., 2017. Residential load and rooftop PV generation: an australian distribution network dataset. International Journal of Sustainable Energy 36, 787– 806. URL: https://doi.org/10.1080/14786451.2015.1100196, doi:10. 1080/14786451.2015.1100196.

[25] Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., Zhou, J., 2024. Timemixer: Decomposable multiscale mixing for time series forecasting. doi:10.48550/arXiv.2405.14616. arXiv preprint.

[26] Wang, W., Yang, D., Hong, T., Kleissl, J., 2022. An archived dataset from the ecmwf ensemble prediction system for probabilistic solar power forecasting. Solar Energy 248, 64–75. doi:10.1016/j.solener. 2022.10.062.

[27] Wang, X., Hyndman, R.J., Li, F., Kang, Y., 2023. Forecast combinations: An over 50-year review. International Journal of Forecasting 39, 1518–1547. doi:10.1016/j.ijforecast.2022.11.005.

[28] Wiese, F., Schlecht, I., Bunke, W.D., Gerbaulet, C., Hirth, L., Jahn, M., Kunz, F., Lorenz, C., Mühlenpfordt, J., Reimann, J., Schill, W.P., 2019. Open Power System Data – frictionless data for electricity system modelling. Applied Energy 236, 401–409. URL: https://doi.org/10. 1016/j.apenergy.2018.11.097, doi:10.1016/j.apenergy.2018.11.097.

[29] Zeng, A., Chen, M., Zhang, L., Xu, Q., 2023. Are transformers efective for time series forecasting? Proceedings of the AAAI Conference on Artificial Intelligence 37, 11121–11128. doi:10.1609/aaai.v37i9. 26317.

[30] Zhang, X., Fang, F., Liu, J., 2019. Weather-classification-mars-based photovoltaic power forecasting for energy imbalance market. IEEE Transactions on Industrial Electronics 66, 8692–8702. doi:10.1109/ TIE.2018.2889611.

[31] Zippenfenig, P., 2024. Open-Meteo.com weather API. Computer software and API service. URL: https://open-meteo.com/, doi:10. 5281/zenodo.7970649. concept DOI metadata identifies software version 1.4.0, issued 2024-12-31. The hosted API build used for this study is not independently pinned; exact requests and response hashes are archived.