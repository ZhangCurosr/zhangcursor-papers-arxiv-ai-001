# FARE: Factor-Aware Retrieval with Error-Guided Fusion for Retail Forecasting

Chenghao Gu<sup>1∗</sup>, Zhuokang Shen<sup>1∗</sup>, Shenghao Wu<sup>1</sup>, Tingxuan Zhu<sup>1</sup>, Weiyi Luo<sup>2</sup>, Haifeng Tang<sup>2</sup>,

<sup>1</sup> East China Normal University, Shanghai, China <sup>2</sup> China Merchants Bank Credit Card Center, Shanghai, China 10245101426@stu.ecnu.edu.cn, 10235101553@stu.ecnu.edu.cn

Abstract. Retail sales forecasting underpins inventory planning, yet promotions and calendar events reshape short-term demand. Although each factor usually has historical occurrences, the complete factor combination of a target week often has few precedents. The challenge is therefore to retrieve informative partial matches and weight them by their reliability, which varies across series. We propose FARE (Factor-Aware Retrieval with Error-Guided Fusion), which ranks historical weeks separately for each known target-week factor and merges the selected weeks into a factor reference under a fixed budget. Guided by shrunk errors from completed development weeks and the current disagreement among candidates, a first gate mixes this factor reference with a shape reference from similar sales histories, and a second gate controls how strongly the mixture corrects a backbone forecast. Both gates remain fixed during testing. On Favorita, M5, Rossmann, and Rohlik, FARE reduces the weighted absolute percentage error (WAPE) of five backbones by 11.46%, 4.06%, 14.63%, and 10.16% on average. With Chronos-2 frozen, FARE also outperforms conventional and existing enhancement methods, including supervised residual correction and retrieval augmentation. Code is available at https://github.com/HousonGu/FARE.

Keywords: Time series forecasting · Factor-conditioned retrieval · Historical prediction errors · Retail sales forecasting

## 1 Introduction

Retail sales forecasts [8] support inventory replenishment and supply-chain planning, where errors afect product availability and cause excess stock. Promotions and calendar events change product-level sales over short horizons, so their known future schedules are relevant to the forecast. The dificulty sharpens when each factor has occurred historically but their complete combination in the target week has few precedents, and the forecaster must reuse observations whose factors only partially agree with the target week.

Existing methods use these signals in three ways. Covariate models such as TFT [14] and DD-CZFormer [20] condition the forecast on known factors, retrieval methods such as SARAF [28], PFRP [7], PIR [16], and RAF [25] supply observed historical trajectories, and combination methods [2,17,22] weight candidate forecasts by past errors or current features.

![](images/2dbfec4026227921d48823cb3bff1c0da37905312b43ebd137b9ed4b74d1c1ad.jpg)  
Fig. 1. Previous methods versus FARE (illustrative example). FARE retrieves partial precedents for each known target-week factor and fuses candidates with gates guided by their historical errors.

When precedents for complete factor combinations are scarce, retrieval and combination face two linked challenges. First, similar past sales do not guarantee that a retrieved trajectory reflects the promotion or calendar conditions of the target week, yet requiring complete factor agreement discards useful partial precedents. Second, partial references may disagree with one another and with the backbone, and their reliability varies across series and is hard to estimate from limited error records.

We address these challenges with FARE, which couples factor-conditioned reference construction with error-guided two-level fusion. FARE ranks historical weeks separately for each target-week factor by sales-shape distance and agreement on that factor, and merges the selected weeks into a factor reference through alternating selection and deduplication. It then shrinks candidate-specific errors from completed development weeks toward a global mean and combines them with current prediction disagreements as inputs to two gates, which separately set the composition of the reference mixture and its correction strength on the backbone.

As illustrated in Fig. 1, FARE constructs factor references and allocates their contributions. Our contributions are threefold:

1. Factor-conditioned partial references. Separate factor rankings build an eightwindow reference through alternating selection and deduplication under a fixed retrieval budget.

2. Historical error-guided two-level fusion. Shrunk per-candidate error histories and current disagreements guide separate gates for reference composition and correction strength, with convex weights that return continuously to the backbone.

3. Evaluation of reference selection and allocation. Experiments on Favorita, M5, Rossmann, and Rohlik with five backbones assess both designs through ablations and matched-candidate comparisons.

## 2 Related Work

## 2.1 Retail Factors and Historical References

Retail forecasting [8] treats promotions and calendar conditions as demand predictors, and DD-CZFormer [20] learns disentangled drivers for unseen factor combinations. Our setting instead targets scarce precedents for complete combinations and uses separate factor rankings to supply explicit reference trajectories. SARAF [28] combines temporal alignment with stationarity-based selection, PFRP [7] combines forecasts from a global memory and a local model, PIR [16] revises failed forecasts from local and global context, and Naive RAF [25] adds retrieved examples without fine-tuning the forecaster. RAFT [10] adds the futures of retrieved training segments to the input of a forecaster, and TS-RAG [24] mixes retrieved segments into the representations of a time series foundation model for zero-shot forecasting. Unlike these methods, FARE constructs factor-conditioned reference forecasts, learns their contribution to the backbone output, and retains explicit shape and factor references under the same history boundary and retrieval budget.

## 2.2 Adaptive Forecast Combination

Classical combination [2] and the review of Wang et al. [26] show that weighting forecasts by historical performance can help, although estimating the weights carries risks. Diebold and Pauly [6] reduce this risk by shrinking estimated combination weights toward equal weights. FFORMA [22] and retail meta-learning [18] map series features to weights, and TimeFuse [17] learns instance-level fusion from input meta-features. FARE instead shrinks the per-series error history of each candidate toward a global mean before it enters the gates, and its gates separate the composition of the reference forecast from its correction of the backbone. PFRP also learns confidence and output gates, whereas FARE’s two gates separately control how the two references are mixed and how much they contribute in total.

## 3 Problem Definition

Let $y _ { i , d } \ge 0$ denote the sales of series � on day �. At each weekly origin �, the task is to predict $y _ { i , t + h }$ for $h \in \{ 0 , \ldots , 6 \}$ from the 56-day history $\mathbf { x } _ { i , t } = ( y _ { i , t - 5 6 } , \ldots , y _ { i , t - 1 } )$ and the target-week factors � known at the origin, such as the promotion, discount, SNAP, and calendar indicators described in Sect. 5.1.

A historical reference $r = ( j , u )$ pairs the 56 days preceding � in series � with the following seven-day trajectory. Its target week must be fully observed before the query origin, that is, $u + 6 < t$ , and retrieval follows the dataset-specific grouping rules in Sect. 5.1. During rolling evaluation, completed weeks enter the library under this boundary, while gate parameters and historical error statistics stay fixed.

FARE fuses three daily candidates, namely a backbone prediction �, a shape reference � from similar sales histories, and a factor reference � from separate matches on known target-week factors. Their fusion weights depend on historical errors, current disagreements, and origin features, and the output $P _ { i , t , h }$ is a nonnegative point forecast for day $t + h$

## 4 Methodology

As shown in Fig. 2, FARE constructs two references and fuses them in two stages. The first gate combines � and � into �, and the second sets how much � adjusts � to produce �. Here clip $( v , l , u )$ truncates � to [�, �], I[·] is the indicator function, and $\sigma ( v ) = 1 / ( 1 + \exp ( - v ) )$ .

## 4.1 Three Forecast Candidates

All candidates are truncated to nonnegative values, and � can come from any backbone. The main experiments use the median forecast of the frozen Chronos-2 [1], with 56 days of sales and concurrent covariates as input and seven days of known future covariates, and Sect. 6.4 evaluates LightGBM, TFT, PatchTST, and SparseTSF as alternatives. On all datasets, Chronos-2 receives one-hot weekdays and the two dataset-specific factors, to which retrieval matching is also restricted. Each window is scaled by its pre-origin sales level

$$
a _ { i , t } = \operatorname* { m a x } \left( \frac { 1 } { 5 6 } \sum _ { \ell = 1 } ^ { 5 6 } y _ { i , t - \ell } , 1 \right)\tag{1}
$$

which reduces magnitude diferences between trajectories, and the shape distance is

$$
d _ { x } ( ( i , t ) , ( j , u ) ) = \frac { 1 } { 5 6 } \sum _ { \ell = 1 } ^ { 5 6 } \left( \frac { y _ { i , t - \ell } } { a _ { i , t } } - \frac { y _ { j , u - \ell } } { a _ { j , u } } \right) ^ { 2 }\tag{2}
$$

� selects the $K = 8$ nearest eligible windows, with ties broken by a stable sort on origin and series ID.

Favorita [5] encodes promotions as categories (0, 1–5, or at least 6) together with binary holidays, M5 [19] uses binary SNAP and event indicators, Rossmann [9] uses binary promotions and school holidays, and Rohlik [21] uses binary discount and holiday indicators, where the discount indicator is one if any of its seven discount fields is positive. For factor $q \in \{ 1 , 2 \}$ , the target-week mismatch is $\begin{array} { r } { d _ { q } = \frac { 1 } { 7 } \sum _ { h = 0 } ^ { 6 } \mathbb { I } [ c _ { i , t , h } ^ { q } \neq } \end{array}$ $c _ { j , u , h } ^ { q } ]$ . Each branch combines sales shape with one factor by ranking references with $\begin{array} { r } { v _ { q } = \frac { 1 } { 2 } \left( \frac { d _ { x } } { 1 + d _ { x } } + d _ { q } \right) } \end{array}$ , where $d _ { x } / ( 1 + d _ { x } )$ puts the shape distance on a scale comparable to $d _ { q } . \mathrm { A }$ window can thus qualify by matching either factor, possibly for only part of the target week. Alternating selection starts with the promotion branch on Favorita and Rossmann, the SNAP branch on M5, and the discount branch on Rohlik, and it skips duplicates so that each branch yields four windows. Both references average their eight selected windows $N _ { i , t }$ with equal weights.

![](images/87863c72f440dd63cbf273535d18d5c570d46fc5f66c68ef6da43d14c8156a49.jpg)  
Fig. 2. Overview of FARE. (a) A backbone forecast and two historical references provide three candidate forecasts. (b) Historical candidate errors and forecast-origin features form the gating inputs. (c) Two gates combine the references and control their contribution to the final forecast.

$$
\widehat { y } _ { i , t , h } ^ { \mathrm { r e f } } = a _ { i , t } \frac { 1 } { 8 } \sum _ { ( j , u ) \in N _ { i , t } } \frac { y _ { j , u + h } } { a _ { j , u } }\tag{3}
$$

� and � share the same eligible library and reference count. On M5, Rossmann, and Rohlik, groups with fewer than eight eligible windows fall back to the query’s previousweek sales, and the query is retained. All references respect the pre-origin boundary, and the controlled Favorita library also excludes complete target-week combinations of high promotions and holidays. All Favorita results that use � and �, namely FARE, its ablations, TimeFuse, and the fused backbones in Table 4, use this controlled library. The exclusion removes 4,556 rows from the library before development, and 3,719 of the 311,850 test rows fall in such combinations.

## 4.2 Historical Errors Accumulated Forward in Time

For source $k \in \{ B , S , F \}$ , the normalized absolute error on an eligible completed day is

$$
e _ { i , u , h } ^ { k } = \mathrm { c l i p } \left( \frac { | y _ { i , u + h } - \widehat { y } _ { i , u , h } ^ { k } | } { a _ { i , u } } , 0 , 1 0 \right)\tag{4}
$$

Let $\mathcal { H } _ { i , i }$ <sub>�</sub> hold the $n _ { i , t }$ prior completed development forecast-day rows (seven per week) and $\mu _ { t } ^ { k }$ be the mean error over all previously eligible development rows. Errors of series with few observations are shrunk toward this global mean.

$$
\widetilde { e } _ { i , t } ^ { k } = \mathrm { c l i p } \left( \frac { \sum _ { ( u , h ) \in \mathcal { H } _ { i , t } } e _ { i , u , h } ^ { k } + \lambda \mu _ { t } ^ { k } } { n _ { i , t } + \lambda } , 0 , 1 0 \right) , \qquad \lambda = 1 4\tag{5}
$$

The gates take the three shrunk errors and log(1 + �) as inputs, and the pseudo-count $\lambda = 1 4$ equals two weeks. Series without history use the global mean, and errors and support are zero when no global history exists.

Development features use only origins whose seven target days precede the current origin. Evaluation freezes the statistics of the full development period.

## 4.3 Gating Features

The base inputs are log(1 + mean sales) over 56 days, the history standard deviation divided by � and clipped to [0, 10], and the zero-sales fraction. Every dataset adds log(1+library size) and its two daily factor indicators, with Favorita using the promotion term min(onpromotion, $6 ) / 6$ in place of a binary promotion flag. Favorita, Rossmann, and Rohlik also add the branch-wise log(1+mean ranking score) over the four references selected by each branch, which is set to zero on Rossmann and Rohlik when reference construction falls back to previous-week sales. On Favorita, the library size enters once per branch, and the two values coincide because both branches search the same library.

Historical errors describe past accuracy, and current disagreements measure the size of candidate corrections.

$$
\begin{array} { r l } & { \delta _ { i , t , h } ^ { B S } = \mathrm { c l i p } \left( \frac { \left| B _ { i , t , h } - S _ { i , t , h } \right| } { a _ { i , t } } , 0 , 1 0 \right) , } \\ & { \delta _ { i , t , h } ^ { F S } = \mathrm { c l i p } \left( \frac { \left| F _ { i , t , h } - S _ { i , t , h } \right| } { a _ { i , t } } , 0 , 1 0 \right) . } \end{array}\tag{6}
$$

z stacks the base and factor inputs, the two disagreements, and the four historical inputs. Binary indicators keep their values, other continuous inputs are standardized on gating training rows (standard deviation floor $1 0 ^ { - 8 } )$ , and an intercept is added. The complete models have 15, 12, 14, and 14 non-intercept inputs on Favorita, M5, Rossmann, and Rohlik, and they are fitted separately for each dataset and backbone.

## 4.4 Two Bounded Soft Gates

The gate weights are

$$
\begin{array} { r } { \alpha _ { i , t , h } = \sigma ( \theta _ { \alpha } ^ { \top } \mathbf z _ { i , t , h } ) , \qquad \beta _ { i , t , h } = \sigma ( \theta _ { \beta } ^ { \top } \mathbf z _ { i , t , h } ) } \end{array}\tag{7}
$$

With a shared input and separate parameters, the first gate constructs

$$
R _ { i , t , h } = S _ { i , t , h } + \alpha _ { i , t , h } ( F _ { i , t , h } - S _ { i , t , h } )\tag{8}
$$

and the second produces

$$
P _ { i , t , h } = \operatorname* { m a x } \{ 0 , B _ { i , t , h } + \beta _ { i , t , h } ( R _ { i , t , h } - B _ { i , t , h } ) \}\tag{9}
$$

The efective weights of $B , S ,$ , and $F$ are therefore $1 - \beta , \beta ( 1 - \alpha )$ , and $\beta \alpha$ . They are nonnegative and sum to one, so $P$ lies in the convex hull of the three candidates and returns continuously to � as $\beta$ approaches zero. The hierarchy thus separates reference composition $( \alpha )$ from total reference participation $( \beta )$ , which Sect. 6.3 evaluates with matched-candidate controls.

## 4.5 Fitting Objective and Optimization

The gates are fitted on development rows $\mathcal { D } _ { \mathrm { d e v } }$ by minimizing

$$
\begin{array} { r } { \mathcal { L } = \frac { \sum _ { ( i , t , h ) \in \mathcal { D } _ { \mathrm { d e v } } } \sqrt { ( P _ { i , t , h } - y _ { i , t + h } ) ^ { 2 } + 1 } } { \sum _ { ( i , t , h ) \in \mathcal { D } _ { \mathrm { d e v } } } y _ { i , t + h } } } \\ { + 0 . 0 0 1 \left( \| \theta _ { \alpha , \mathrm { n o n b i a s } } \| _ { 2 } ^ { 2 } + \| \theta _ { \beta , \mathrm { n o n b i a s } } \| _ { 2 } ^ { 2 } \right) . } \end{array}\tag{10}
$$

The first term is a smooth surrogate of normalized absolute error with a smoothing constant of one sales unit. The second term regularizes the non-intercept gate parameters, and evaluation uses unsmoothed WAPE and MAE. Zero-initialized parameters are optimized by L-BFGS-B with analytical gradients (at most 250 iterations, $\mathrm { f t o l } = 1 0 ^ { - 9 }$ $\mathrm { g t o l } = 1 0 ^ { - 6 } )$ . Gate fitting and error accumulation both use development labels under the boundary in Sect. 3.

## 5 Experimental Setup

## 5.1 Data and Temporal Protocol

Favorita [5] has 1,782 store-category series with promotion and holiday factors, and its references are grouped by product category. Missing Christmas records are not zerofilled, so origins whose 56-day history or target week includes 25 December are skipped, which removes nine origins from each holiday season. M5 [19] has 30,490 item-store series with SNAP and calendar-event factors, grouped by item. A series enters at its first week with a listed price and needs 56 days of coverage, but price values are not used. M5 series are mostly intermittent, with zero sales on 62% of history days in the average development row, and their WAPE is correspondingly high. Rossmann [9] has 1,115 store series with promotion and school-holiday factors, grouped by store type and assortment. Evaluation covers all days, including 162,990 open-day rows, without forcing closed-day forecasts to zero. Rohlik V2 [21] has 5,390 anonymized product series of availability-adjusted sales in item counts or kilograms, with discount and holiday indicators, grouped by warehouse and level-2 category. On all datasets, each weekly origin uses 56 complete calendar days to forecast the next seven nonoverlapping days, incomplete histories are skipped, and our rolling protocol replaces the original competition horizons and scoring rules.

The fitting, development, and testing periods are listed in Table 1. Chronos-2 [1] stays frozen in all comparisons, whereas LightGBM [13], TFT [14], PatchTST [23], and SparseTSF [15] are trained on the fitting data, with configurations selected on fittingperiod validation origins before retraining on all fitting origins. Correction modules are selected by chronological splits of the development period, whose labels fit FARE’s gates and the LightGBM residual corrector. Test labels are never used for fitting, selection, or error calibration, although completed test weeks may supply history for later origins.

Table 1. Temporal ranges and forecast-day row counts. Fit denotes supervised-backbone fitting and historical references, and Dev. denotes correction-module development.
<table><tr><td>Dataset</td><td>Stage</td><td>Origins</td><td>Origin range</td><td>Forecast-day rows</td></tr><tr><td>Favorita</td><td>Fit</td><td>62</td><td>2015-02-23 to 2016-06-27</td><td>773,388</td></tr><tr><td>Favorita</td><td>Dev.</td><td>24</td><td>2016-07-04 to 2016-12-12</td><td>299,376</td></tr><tr><td>Favorita</td><td>Test</td><td>25</td><td>2017-02-20 to 2017-08-07</td><td>311,850</td></tr><tr><td>M5</td><td>Fit</td><td>165</td><td>2011-03-28 to 2014-05-19</td><td>23,520,098</td></tr><tr><td>M5</td><td>Dev.</td><td>52</td><td>2014-05-26 to 2015-05-18</td><td>10,539,837</td></tr><tr><td>M5</td><td>Test</td><td>52</td><td>2015-05-25 to 2016-05-16</td><td>11,053,322</td></tr><tr><td>Rossmann</td><td>Fit</td><td>73</td><td>2013-03-04 to 2014-07-21</td><td>564,725</td></tr><tr><td>Rossmann</td><td>Dev.</td><td>26</td><td>2014-07-28 to 2015-01-19</td><td>170,170</td></tr><tr><td>Rossmann</td><td>Test</td><td>26</td><td>2015-01-26 to 2015-07-20</td><td>196,630</td></tr><tr><td>Rohlik</td><td>Fit</td><td>140</td><td>2020-09-28 to 2023-05-29</td><td>1,300,544</td></tr><tr><td>Rohlik</td><td>Dev.</td><td>26</td><td>2023-06-05 to 2023-11-27</td><td>321,573</td></tr><tr><td>Rohlik</td><td>Test</td><td>26</td><td>2023-12-04 to 2024-05-27</td><td>283,185</td></tr></table>

## 5.2 Comparisons and Controls

Except for the backbone study in Sect. 6.4, all comparisons and ablations use frozen Chronos-2 as the backbone of FARE, and all contrasts use the evaluation rows of the complete model in each dataset. Input ablations remove the historical errors and support, the disagreements, or the factor branch with its inputs. Component controls substitute joint Top-8 retrieval, three-seed random retrieval, global errors (retaining series-specific support), unshrunk errors, or a learned constant gate. Output-path controls keep all gate inputs. Learned B+S fixes $\alpha = 0$ and learned B+F fixes $\alpha = 1$ , both learning � on development data with � parameters versus 2� in the complete model, while the joint-constant control learns only the two gate intercepts.

TimeFuse [17] combines our three stored candidates through meta-feature weighting. PFRP [7] and its DLinear [27] backbone use sales history, PIR [16] revises frozen Chronos-2 forecasts with deterministic calendar features and cross-series retrieval over fitting windows, and SARAF [28] uses a fitted linear predictor and same-series history. Because these adapted systems difer in backbones, covariates, and libraries, matchedcandidate controls and within-backbone ablations assess allocation more directly. PFRP, PIR, and SARAF report means over three seeds.

RAF and LightGBM residual correction share the frozen Chronos-2 weights, known covariates, and evaluation rows. Our retail adaptation of Naive RAF [25] uses Chronos-2 for embeddings and forecasts with a weekly fitting-only library. It retrieves one 56+7- day example from the legal fitting-period windows of the same series by the summed L2 distance of corresponding tokens, and series without such windows use up to 64 fixed global examples from the last fitting origin, evenly spaced in series-ID order. The example and the 56-day query are normalized separately, and the example is shifted to join the query continuously, giving a 119-day input. Source-date covariates are retained, and forecasts are restored to the query scale.

The LightGBM [13] residual corrector learns $y - B$ from 56-day lags, historical summaries, forecast step, date features, permitted future factors, and the clipped frozen forecast �, returning max $( \boldsymbol { B } + \boldsymbol { \hat { r } } , 0 )$ . Using only development labels, it selects the number of leaves in {15, 31, 63} by corrected-forecast pooled WAPE on the last $\lceil 0 . 2 T _ { \mathrm { d e v } } \rceil$ origins after fitting on earlier ones, trains 300 L1 boosting rounds, and refits 300 rounds on all development origins. We report the mean of three separately scored seeds, not an ensemble. This corrector difers from the LightGBM backbone in Table 4, and Chronos-2 is neither retrained nor fine-tuned.

The conventional baselines are ETS [12], SARIMA [3], Ridge regression [11], and XGBoost [4]. ETS and SARIMA are refit on the latest 56 days at each origin, while Ridge and XGBoost also use permitted future factors and deterministic date features. Settings are selected on the last eight fitting origins, after which Ridge and XGBoost are retrained on all fitting origins. ETS selects AAA on Favorita and Rossmann and ANN on M5 and Rohlik, SARIMA uses $( 1 , 0 , 0 ) \times ( 0 , 1 , 1 ) _ { 7 }$ , Ridge uses a regularization strength of 10,000, and XGBoost uses depth 7, a learning rate of 0.05, and 300 rounds. ETS, SARIMA, and Ridge report one run, and XGBoost reports the mean of three separately scored seeds.

For supervised backbones, candidate errors and disagreements are recomputed and the gates are refit for each backbone, and LightGBM, TFT, PatchTST, and SparseTSF results are means over three seeds on all datasets. PatchTST [23] and SparseTSF [15] use only sales history with the same 56-day input and seven-day horizon, and their settings are selected on the last eight fitting origins before retraining.

## 5.3 Metrics and Implementation

For evaluation rows E, the two metrics are defined as

$$
\mathrm { W A P E } = \frac { \sum _ { r \in \mathcal { E } } \left| y _ { r } - P _ { r } \right| } { \sum _ { r \in \mathcal { E } } y _ { r } } , \qquad \mathrm { M A E } = \frac { \sum _ { r \in \mathcal { E } } \left| y _ { r } - P _ { r } \right| } { \left| \mathcal { E } \right| }\tag{11}
$$

WAPE pools absolute errors and sales across weeks, rather than averaging weekly values, and is reported in percent. MAE is in daily-sales units and equals WAPE times mean sales on the same targets.

Chronos-2 uses the pretrained amazon/chronos-2 model with nonnegative median forecasts and cross-series learning disabled. Each backbone forecast therefore depends only on the history and known factors of its own series, which keeps the forecasts independent of batch composition and leaves retrieval as the only channel through which other series inform FARE. FARE uses sales-shape distances, whereas the RAF adaptation reuses frozen Chronos-2 embeddings. Reference aggregation costs �(��) after selection. Search and sorting, however, scale with library size, which the parameter count does not reflect. On M5, building the references for all 104 development and test origins takes 289 s, and fitting and applying the gates for Chronos-2, LightGBM, and three TFT seeds takes 1,141 s. Including intercepts, the two gates have 32, 26, 30, and 30 trainable parameters on Favorita, M5, Rossmann, and Rohlik, optimized as in Sect. 4.5. All supervised-backbone and fusion runs use an NVIDIA RTX 4090.

## 6 Results and Analysis

## 6.1 Comparison with Existing Methods

As reported in Table 2, FARE with Chronos-2 achieves the lowest WAPE and MAE on all four datasets. On Favorita and Rohlik, the strongest competitor is TimeFuse, which fuses the same three candidates, and FARE lowers its WAPE by a further 6.81% and 6.69%, showing that FARE’s allocation draws more from the same candidates. On M5 and Rossmann, FARE still beats the strongest competitors, LightGBM residual correction and XGBoost, which use the factors directly, by 3.64% and 1.32%, while keeping Chronos-2 frozen and fitting only 26 to 30 gate parameters.

With the same frozen Chronos-2, FARE reduces the backbone WAPE reported in Table 4 by 12.29%, 6.13%, 25.94%, and 9.96% on Favorita, M5, Rossmann, and Rohlik, compared with at most 5.88%, 2.59%, 15.38%, and 3.50% for the other correction modules, and its advantage is largest on Rossmann. Retrieval augmentation and residual correction each fall below the backbone on one dataset, whereas FARE improves it on all four.

## 6.2 Ablation Study

As summarized in Table 3, removing the historical errors, the disagreements, or the factor branch increases WAPE on all four datasets, and every component therefore contributes. Random references are the most damaging variant on every dataset, which makes the retrieved content the foundation of the correction, but the bottleneck beyond it difers by dataset. On Rossmann, removing the factor branch raises WAPE by 18.66%, more than any control except random references, and supplying factor-matched precedents is the bottleneck there. On Favorita, fixing � or $\beta$ or using global errors raises WAPE by 15.14–16.49%, about twice the 7.61–8.21% of removing any single input, which points to candidate weighting as the bottleneck. On M5 and Rohlik, error estimation and retrieval matter more than either gate alone, since global or unshrunk errors and joint Top-8 retrieval raise WAPE by 7.80–9.19% and 8.63–9.98%, whereas fixing � or $\beta$ costs at most 4.43% and 4.53%. Because the bottleneck changes across datasets, FARE keeps both levels, and the complete model outperforms every variant on all four datasets. Under the same eight-reference budget, factor-wise ranking also beats joint Top-8 retrieval on every dataset, because reserving references for each factor keeps partial precedents that a combined score would rank below windows with similar shape but mismatched factors.

Table 2. Comparison with conventional and conference methods. FARE uses frozen Chronos-2 as its backbone, and TimeFuse, PIR, RAF, and LightGBM residual correction build on the same frozen forecasts.
<table><tr><td rowspan="2">Method</td><td colspan="2">Favorita</td><td colspan="2">M5</td><td colspan="2">Rossmann</td><td colspan="2">Rohlik</td></tr><tr><td>WAPE↓ MAE↓ WAPE↓ MAE↓ WAPE↓ MAE↓ WAPE↓ MAE↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Conventional methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ETS [12]</td><td>17.72</td><td>85.4</td><td>75.47</td><td>0.992</td><td>25.07</td><td>1477</td><td>27.00</td><td>37.1</td></tr><tr><td>SARIMA [3]</td><td>15.94</td><td>76.8</td><td>77.85</td><td>1.023</td><td>23.22</td><td>1368</td><td>26.39</td><td>36.3</td></tr><tr><td>Ridge [11]</td><td>18.58</td><td>89.6</td><td>74.99</td><td>0.986</td><td>18.52</td><td>1091</td><td>25.80</td><td>35.4</td></tr><tr><td>XGBoost [4]</td><td>15.63</td><td>75.4</td><td>73.30</td><td>0.964</td><td>15.91</td><td>937</td><td>20.92</td><td>28.7</td></tr><tr><td>LightGBM residual [13]</td><td>15.15</td><td>73.0</td><td>66.54</td><td>0.875</td><td>17.94</td><td>1057</td><td>20.10</td><td>27.6</td></tr><tr><td>Conference methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(ICML 25’) TimeFuse [17]</td><td>14.25</td><td>68.7</td><td>67.48</td><td>0.887</td><td>18.74</td><td>1104</td><td>19.87</td><td>27.3</td></tr><tr><td>(AAAI 26’) PFRP [7]</td><td>17.03</td><td>82.1</td><td>73.44</td><td>0.965</td><td>21.23</td><td>1251</td><td>22.88</td><td>31.4</td></tr><tr><td>(NeurIPS 25’) PIR [16]</td><td>15.58</td><td>75.1</td><td>72.96</td><td>0.959</td><td>20.53</td><td>1210</td><td>20.23</td><td>27.8</td></tr><tr><td>(KDD 26’) SARAF [28]</td><td>21.60</td><td>104.1</td><td>73.94</td><td>0.972</td><td>19.72</td><td>1162</td><td>25.64</td><td>35.2</td></tr><tr><td>(AISTATS 26’) RAF [25]</td><td>14.80</td><td>71.4</td><td>68.70</td><td>0.903</td><td>20.88</td><td>1230</td><td>20.58</td><td>28.3</td></tr><tr><td>FARE (Ours)</td><td>13.28</td><td>64.0</td><td>64.12</td><td>0.843</td><td>15.70</td><td>925</td><td>18.54</td><td>25.5</td></tr></table>

With all gate inputs retained, learned B+F beats learned B+S on all four datasets, which makes � the more informative correction target, and both beat the joint-constant control, while none of the three matches the complete FARE. A fitted $\beta$ does not by itself protect the backbone, as seven variants on Favorita, five on M5, and two on Rohlik fall below it. Fitted on development rows, $\beta$ stays small on test rows only when its inputs separate reliable from unreliable references, and changed references, pooled or unshrunk errors, or restricted gates weaken this signal, much as the development-fitted global weights degrade M5 in Fig. 3a.

## 6.3 Mechanism Analysis of Reference Provision versus Allocation

As illustrated in Fig. 3a, the one-level softmax isolates the two-level structure because it shares the candidates, inputs, loss, and parameter count of FARE. On M5, it degrades the backbone, as do fixed blends, global weights, and historical hard selection, whereas FARE improves it. Convexity alone therefore does not prevent degradation. With the same candidates, inputs, loss, and parameter count, the remaining diference is that FARE separates what to mix (�) from how far to move (�). As shown in Fig. 3b, fixing either gate raises WAPE on all four datasets, which supports learning the two decisions separately.

Table 3. FARE component ablations with Chronos-2.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Favorita</td><td colspan="2">M5</td><td colspan="2">Rossmann</td><td colspan="2">Rohlik</td></tr><tr><td>WAPE↓ MAE↓ WAPE↓ MAE↓ WAPE↓ MAE↓ WAPE↓ MAE↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chronos-2 backbone FARE (Ours)</td><td>15.14 13.28</td><td>73.0 64.0</td><td>68.31 64.12</td><td>0.898 0.843</td><td>21.20 15.70</td><td>1249 925</td><td>20.59 18.54</td><td>28.3 25.5</td></tr><tr><td colspan="9">Input and reference-branch ablations</td></tr><tr><td>w/o historical errors</td><td></td><td></td><td>66.97</td><td>0.880</td><td>16.86</td><td>994</td><td>19.65</td><td>27.0</td></tr><tr><td>w/o factor reference</td><td>14.34 14.37</td><td>69.1 69.3</td><td>66.98</td><td>0.880</td><td>18.63</td><td>1098</td><td>20.11</td><td>27.6</td></tr><tr><td>w/o disagreements</td><td>14.29</td><td>68.9</td><td>67.41</td><td>0.886</td><td>18.00</td><td>1061</td><td>19.65</td><td>27.0</td></tr><tr><td colspan="9">Retrieval, error estimation, and gate ablations</td></tr><tr><td>Constant α</td><td>15.29</td><td>73.7</td><td>66.54</td><td>0.875</td><td>16.60</td><td>978</td><td>19.14</td><td>26.3</td></tr><tr><td>Constant β</td><td>15.33</td><td>73.9</td><td>66.96</td><td>0.880</td><td>16.92</td><td>997</td><td>19.38</td><td>26.6</td></tr><tr><td>Global errors</td><td>15.47</td><td>74.6</td><td>69.61</td><td>0.915</td><td>17.81</td><td>1050</td><td>20.14</td><td>27.7</td></tr><tr><td>w/o shrinkage</td><td>14.88</td><td>71.7</td><td>70.01</td><td>0.920</td><td>18.28</td><td>1077</td><td>20.35</td><td>28.0</td></tr><tr><td>Joint Top-8</td><td>14.97</td><td>72.2</td><td>69.12</td><td>0.909</td><td>17.20</td><td>1014</td><td>20.39</td><td>28.0</td></tr><tr><td>Random references</td><td>16.94</td><td>81.6</td><td>71.82</td><td>0.944</td><td>19.31</td><td>1138</td><td>22.16</td><td>30.4</td></tr><tr><td colspan="9">Candidate-path and joint-constant controls</td></tr><tr><td>Learned B+S</td><td>15.78</td><td>76.0</td><td>67.75</td><td>0.891</td><td>17.44</td><td>1028</td><td>20.10</td><td>27.6</td></tr><tr><td>Learned B+F</td><td>15.62</td><td>75.3</td><td>67.09</td><td>0.882</td><td>17.20</td><td>1014</td><td>19.94</td><td>27.4</td></tr><tr><td>Constant α and β</td><td>16.14</td><td>77.8</td><td>68.46</td><td>0.900</td><td>18.11</td><td>1067</td><td>21.14</td><td>29.0</td></tr></table>

The error history helps only when it is series-specific and shrunk. On all four datasets, global or unshrunk errors are worse than no error history. An inaccurate reliability estimate thus misguides the gates, and shrinkage makes series-level estimates usable from the few development weeks that scarce combinations provide. Disagreements complement this history. As shown in Fig. 3c, removing them hurts every dataset even with the error history and origin features intact, since past errors describe how reliable a candidate has been, whereas disagreements describe how large the current correction would be. Finally, as shown in Fig. 3d, removing the factor branch on Favorita raises WAPE by 2.36 points in the high-promotion and holiday subgroup but by 1.09 points over all test rows, and partial factor precedents therefore matter more where the complete combination is rare.

## 6.4 Complementarity Across Forecasting Backbones

As reported in Table 4, FARE improves all five backbones on all four datasets, reducing WAPE by 11.46%, 4.06%, 14.63%, and 10.16% on average. Each of the 48 fused supervised-backbone runs, across four datasets and three seeds, has lower WAPE than its own unfused backbone, and the largest seed standard deviation of the fused WAPE is 0.87 points. The gain depends more on the dataset than on the backbone. On

![](images/dc65351b82174d4f4e4ac1e76b59dd322bbcc77c29c9220d7becf6ee92959d3d.jpg)

(b) Two allocation decisions  
![](images/d11f96e22e1eb8d7318565f4ff442ee19845957e8a04034ad48bd305509111cd.jpg)

![](images/c1b91cef545e243328f8aaf585cf1a0f6f251ed064e83b1def705543cdaae251.jpg)

(d) Concurrent factors  
![](images/5ad5b09cf00b2a76d798180bd5df8de660fdcdd4ab72dffac5f3713e75ea99a5.jpg)  
Fig. 3. Reference allocation and component efects in FARE on the final test rows. (a) WAPE reduction from the backbone for alternative combination rules. Best 50:50 reports the lower test WAPE of the fixed 50:50 B/S and B/F blends. (b) Efects of fixing reference composition (�) or correction strength (�). (c) WAPE increase over full FARE after changing allocation inputs. (d) Component efects on all Favorita rows and the promotion–holiday subgroup.

Table 4. Backbone forecasts with and without FARE.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">FARE</td><td colspan="2">Favorita</td><td colspan="2">M5</td><td colspan="2">Rossmann</td><td colspan="2">Rohlik</td></tr><tr><td>WAPE↓ MAE↓ WAPE↓ MAE↓ WAPE↓ MAE↓ WAPE↓ MAE↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(arXiv 25’) Chronos-2 [1]</td><td>without</td><td>15.14</td><td>73.0</td><td>68.31</td><td>0.898</td><td>21.20</td><td>1249</td><td>20.59</td><td>28.3</td></tr><tr><td rowspan="2">(NeurIPS 17&#x27;) LightGBM [13]without</td><td>with</td><td>13.28</td><td>64.0</td><td>64.12</td><td>0.843</td><td>15.70</td><td>925</td><td>18.54</td><td>25.5</td></tr><tr><td></td><td>13.70</td><td>66.0</td><td>65.78</td><td>0.865</td><td>13.01</td><td>767</td><td>18.60</td><td>25.5</td></tr><tr><td rowspan="2">(IJF 21’) TFT [14]</td><td>with</td><td>11.98</td><td>57.7</td><td>63.01</td><td>0.828</td><td>11.98</td><td>706</td><td>16.49</td><td>22.7</td></tr><tr><td>without with</td><td>13.91 12.43</td><td>67.1</td><td>64.83</td><td>0.852</td><td>14.57</td><td>859</td><td>18.48</td><td>25.4</td></tr><tr><td rowspan="2">(ICLR 23’) PatchTST [23]</td><td></td><td></td><td>59.9</td><td>62.57</td><td>0.823</td><td>12.80</td><td>754</td><td>16.23</td><td>22.3</td></tr><tr><td>without with</td><td>14.50</td><td>69.9</td><td>66.18</td><td>0.870</td><td>17.73</td><td>1045</td><td>21.21</td><td>29.1</td></tr><tr><td rowspan="2">(ICML 24’) SparseTSF [15]</td><td></td><td>13.16</td><td>63.5</td><td>63.24</td><td>0.831</td><td>15.73</td><td>927</td><td>19.29</td><td>26.5</td></tr><tr><td>without with</td><td>15.37 13.44</td><td>74.1 64.8</td><td>74.03 72.53</td><td>0.973 0.953</td><td>18.79 15.81</td><td>1107 932</td><td>23.22 21.30</td><td>31.9 29.3</td></tr></table>

Favorita and Rohlik, the reductions stay within 9.24–12.56% and 8.27–12.18% across pretrained, gradient-boosted, and deep architectures, indicating that FARE corrects an error component these backbones share. On Rossmann, FARE narrows the WAPE range across backbones from 8.19 to 3.83 points, and the weakest backbone, Chronos-2, gains the most at 25.94%, or 23.90% when the closed days with zero sales are excluded. Because Chronos-2 already receives the target-week factors, this gain indicates that retrieved trajectories add realized sales shapes beyond covariate access, while the salesonly PatchTST and SparseTSF additionally benefit from the factor information that FARE introduces. A strong backbone still matters, as fused SparseTSF trails unfused Chronos-2 on M5.

## 6.5 Temporal Consistency and Remaining Regressions

The gain of FARE is consistent over time. With Chronos-2, it improves 22 of 25 Favorita origins, every M5 and Rohlik origin, and 24 of 26 Rossmann origins, including Rossmann’s open days, so the gain is not confined to promotion or holiday weeks. With LightGBM and TFT, every seed improves at all Rohlik origins.

FARE trails the backbone at three Favorita and two Rossmann origins, yet it beats the one-level softmax, which uses the same convex form, at every Rossmann and Rohlik origin, so the two-level structure sustains the gain.

## 7 Discussion and Limitations

The results point to two design choices for retail retrieval. Known target-week factors belong in the retrieval key, because series with similar histories can face diferent promotions and holidays. Reliability weights work best when per-series error estimates are shrunk toward a pooled mean, and the retrieval and allocation ablations in Sect. 6.2 show which level to improve first on a new dataset. FARE adds 26 to 32 parameters to a frozen backbone and can be attached without retraining it, although its gates are refit for each backbone.

The evidence covers two known factors and seven-day forecasts on four retail datasets. Favorita’s controlled library tests the reuse of partial precedents, but generalization to combinations absent from all training data remains open, and evaluating FARE across levels of combination support would show when partial retrieval helps most. Error statistics and gate parameters stay fixed during testing, yet the average gain per origin is similar in the first and second halves of the test period, at 4.19 and 4.13 points on M5, 4.97 and 6.59 points on Rossmann, and 2.38 and 1.90 points on Rohlik. When all candidates miss the same demand change, convex fusion cannot recover it, and richer candidate forecasts are one remedy.

## 8 Conclusion

To address retail forecasting when the complete factor combination of a target week has few historical precedents, this paper proposes FARE, which retrieves partial precedents separately for each known target-week factor and then fuses them with the backbone forecast through error-guided two-level gates. Across all 20 pairings of five backbones and four benchmarks, FARE lowers both WAPE and MAE, with average relative WAPE reductions of 11.46%, 4.06%, 14.63%, and 10.16% on Favorita, M5, Rossmann, and Rohlik, while adding only a few dozen parameters to a frozen backbone. The experiments support three conclusions. First, retrieval should be conditioned on known future conditions rather than on historical similarity alone. Second, under scarce evidence, candidate reliability should be estimated per series and shrunk toward a pooled mean, in the spirit of empirical Bayes. Third, separating reference composition from correction strength outperforms one-level and fixed allocation of the same candidates.

Disclosure of Interests. The authors declare no competing interests or financial conflicts.

## References

1. Ansari, A.F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., et al.: Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821 (2025). https: //doi.org/10.48550/arXiv.2510.15821

2. Bates, J.M., Granger, C.W.J.: The combination of forecasts. Operational Research Quarterly 20(4), 451–468 (1969). https://doi.org/10.1057/jors.1969.103

3. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., Ljung, G.M.: Time Series Analysis: Forecasting and Control. Wiley, Hoboken, NJ, 5th edn. (2015)

4. Chen, T., Guestrin, C.: XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 785–794. Association for Computing Machinery (2016). https://doi.org/10.1145/2939672. 2939785

5. Cook, A., DanB, inversion, Holbrook, R.: Store sales - time series forecasting. Kaggle (2021), https://www.kaggle.com/competitions/store-sales-time-series-forecasting, dataset. Accessed 20 October 2026

6. Diebold, F.X., Pauly, P.: The use of prior information in forecast combination. International Journal of Forecasting 6, 503–508 (1990)

7. Du, D., Han, T., Guo, S.: Predicting the future by retrieving the past. Proceedings of the AAAI Conference on Artificial Intelligence 40(25), 20896–20904 (2026). https://doi.org/10. 1609/aaai.v40i25.39230

8. Fildes, R., Ma, S., Kolassa, S.: Retail forecasting: Research and practice. International Journal of Forecasting 38(4), 1283–1318 (2022). https://doi.org/10.1016/j.ijforecast.2019.06.004

9. FlorianKnauer, Cukierski, W.: Rossmann store sales. Kaggle (2015), https://www.kaggle. com/competitions/rossmann-store-sales, dataset. Accessed 20 October 2026

10. Han, S., Lee, S., Cha, M., Arik, S.O., Yoon, J.: Retrieval augmented time series forecasting. In: Proceedings of the 42nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 267, pp. 21774–21797. PMLR (2025), https://proceedings. mlr.press/v267/han25d.html

11. Hoerl, A.E., Kennard, R.W.: Ridge regression: Biased estimation for nonorthogonal problems. Technometrics 12(1), 55–67 (1970). https://doi.org/10.1080/00401706.1970.10488634

12. Hyndman, R.J., Koehler, A.B., Snyder, R.D., Grose, S.: A state space framework for automatic forecasting using exponential smoothing methods. International Journal of Forecasting 18(3), 439–454 (2002). https://doi.org/10.1016/S0169-2070(01)00110-8

13. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., et al.: LightGBM: A highly eficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems. vol. 30, pp. 3146–3154 (2017), https://proceedings.neurips.cc/paper\_files/paper/ 2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

14. Lim, B., Arık, S.Ö., Loef, N., Pfister, T.: Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting 37(4), 1748–1764 (2021). https://doi.org/10.1016/j.ijforecast.2021.03.012

15. Lin, S., Lin, W., Wu, W., Chen, H., Yang, J.: SparseTSF: Modeling long-term time series forecasting with 1k parameters. In: Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 30211–30226. PMLR (2024), https://proceedings.mlr.press/v235/lin24n.html

16. Liu, Z., Cheng, M., Zhao, G., Yang, J., Liu, Q., Chen, E.: Improving time series forecasting via instance-aware post-hoc revision. In: Advances in Neural Information Processing Systems. vol. 38, pp. 40040–40065 (2025). https://doi.org/10.52202/085713-1196

17. Liu, Z., Yang, Z., Lin, X., Qiu, R., Wei, T., Zhu, Y., et al.: Breaking silos: Adaptive model fusion unlocks better time series forecasting. In: Proceedings of the 42nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 267, pp. 40022–40042. PMLR (2025), https://proceedings.mlr.press/v267/liu25cm.html

18. Ma, S., Fildes, R.: Retail sales forecasting with meta-learning. European Journal of Operational Research 288(1), 111–128 (2021). https://doi.org/10.1016/j.ejor.2020.05.038

19. Makridakis, S., Spiliotis, E., Assimakopoulos, V.: M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting 38(4), 1346–1364 (2022). https://doi. org/10.1016/j.ijforecast.2021.11.013

20. Mayaluri, Z.L., Mishra, S.: Compositional zero-shot demand forecasting in retail via disentangled demand drivers. International Journal of Forecasting (2026). https://doi.org/10.1016 j.ijforecast.2026.07.010, in press

21. MichalKecera: Rohlik sales forecasting challenge. Kaggle (2024), https://www.kaggle.com/ competitions/rohlik-sales-forecasting-challenge-v2, dataset. Accessed 20 October 2026

22. Montero-Manso, P., Athanasopoulos, G., Hyndman, R.J., Talagala, T.S.: FFORMA: Featurebased forecast model averaging. International Journal of Forecasting 36(1), 86–92 (2020). https://doi.org/10.1016/j.ijforecast.2019.02.011

23. Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J.: A time series is worth 64 words: Longterm forecasting with transformers. In: International Conference on Learning Representations (2023), https://openreview.net/forum?id=Jbdc0vTOcol

24. Ning, K., Pan, Z., Liu, Y., Jiang, Y., Zhang, J., Rasul, K., Schneider, A., Ma, L., Nevmyvaka, Y., Song, D.: TS-RAG: Retrieval-augmented generation based time series foundation models are stronger zero-shot forecaster. In: Advances in Neural Information Processing Systems. vol. 38 (2025)

25. Tire, K., Taga, E.O., Ildiz, M.E., Oymak, S.: Retrieval augmented time series forecasting. In: Proceedings of the 29th International Conference on Artificial Intelligence and Statistics. Proceedings of Machine Learning Research, vol. 300, pp. 4294–4302. PMLR (2026), https: //proceedings.mlr.press/v300/tire26a.html

26. Wang, X., Hyndman, R.J., Li, F., Kang, Y.: Forecast combinations: An over 50-year review. International Journal of Forecasting 39(4), 1518–1547 (2023). https://doi.org/10.1016/ j.ijforecast.2022.11.005

27. Zeng, A., Chen, M., Zhang, L., Xu, Q.: Are transformers efective for time series forecasting? Proceedings of the AAAI Conference on Artificial Intelligence 37(9), 11121–11128 (2023). https://doi.org/10.1609/aaai.v37i9.26317

28. Zhou, S., Schöner, H., Wu, Z., Fouché, E., Wilson, I., Wang, S.: Stationarity-aware retrievalaugmented time series forecasting. In: Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2. pp. 6875–6886. Association for Computing Machinery (2026). https://doi.org/10.1145/3770855.3817813