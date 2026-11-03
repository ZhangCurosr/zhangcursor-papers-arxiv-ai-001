# Evaluating Uncertainty Calibration in Probabilistic Time Series Foundation Models

Thomas Decker<sup>1,2,3</sup>

Volker Tresp<sup>2,3</sup>

<sup>1</sup>Siemens AG <sup>2</sup>LMU Munich <sup>3</sup>Munich Center for Machine Learning (MCML) thomas.decker@siemens.com, volker.tresp@lmu.de

## Abstract

Time series foundation models (TSFMs) are increasingly used for probabilistic forecasting across diverse domains, yet the reliability of their uncertainty estimates remains poorly understood. In this work, we study uncertainty calibration in probabilistic TSFMs from a broader methodological and empirical perspective. We introduce a taxonomy of how TSFMs represent predictive uncertainty and develop a fine-grained evaluation framework for calibration in multi-horizon forecasting that distinguishes pooled, horizon-wise, and stronger conditional notions of calibration. Using this framework, we conduct a comprehensive empirical evaluation of popular TSFMs across diverse datasets and show that current models exhibit distinct and systematic patterns of miscalibration that standard metrics often obscure. We further analyze common post-hoc recalibration methods and characterize which forms of calibration they can improve and which tend to persist. Our results provide practical guidance for evaluating and improving uncertainty calibration in TSFMs and highlight important limitations of current practice.

## 1 Introduction

Time series foundation models (TSFMs) are progressively emerging as a general-purpose approach to forecasting [27]. Pretrained on large and heterogeneous collections of time series, they can often be applied in a zero-shot manner to unseen datasets, frequencies, and domains while remaining competitive with specialized forecasting systems [5, 42, 12]. As their adoption grows in high-stakes applications such as healthcare [21], energy systems [33], and finance [44], it becomes increasingly important that they not only produce accurate forecasts but also communicate uncertainty in a trustworthy manner.

In many of these settings, decisions depend not only on the predicted outcome itself but also on how likely adverse outcomes are. This makes the quality of uncertainty estimates an important practical objective beyond predictive accuracy. However, such estimates are useful only if their stated confidence levels can be interpreted reliably. In particular, a prediction interval reported at 90% confidence for a forecast should contain the realized outcome about 90% of the time. This notion of reliability is commonly captured by calibration, meaning that predicted probabilities agree with observed frequencies [17, 20, 26, 40]. A model might therefore achieve strong forecasting accuracy while still providing uncertainty estimates that are misleading in practice.

Although TSFMs increasingly support uncertainty-aware forecasting, calibration in this setting is still only partially understood. Recent work has begun to examine this question empirically [1], providing an important first view of calibration in modern forecasting models. At the same time, calibration is a subtle property in multi-horizon forecasting, where conclusions can depend on how forecasts are represented, how behavior varies across horizon steps, and how results are aggregated across instances. This creates the need for a more systematic evaluation framework that supports comparison across heterogeneous TSFMs and reveals forms of miscalibration that may be hidden by standard aggregate metrics.

![](images/f60461674b203a0f50b18352065be123cbb2bf5020d18138a4062493ed007345.jpg)

Recalibration (Isotonic recalibration, chronological split)  
![](images/2c816d11c897b6b97e5ad6d53e53ee17fef32b8da33da8d93a2ad9646682f557.jpg)  
Figure 1: Overview of the proposed calibration evaluation framework and main empirical findings. Left: Relationship between $\mathrm { P \bar { C } E _ { p o o l } }$ and $\mathrm { Q C E _ { a v g } }$ across model-dataset pairs. All points lie above the diagonal, consistent with the calibration hierarchy $\mathrm { P C E } _ { \mathrm { p o o l } } \leq \mathrm { Q C E } _ { \mathrm { a v g } } ,$ , showing that pooled probabilistic calibration can substantially understate miscalibration once horizon-specific and quantileconditioned effects are taken into account. Right: Effect of post-hoc isotonic recalibration on $\mathrm { P C E _ { a v g } } ,$ fitted on the earliest 20% of the forecast windows and evaluated on the later ones. Points below the diagonal improve (62% of the model–dataset pairs), and the largest gains occur where the original miscalibration is most severe, indicating that even simple recalibration techniques often improve horizon-wise probabilistic calibration in practice.

This question is also practically important because miscalibration is often addressed after training through post-hoc recalibration methods. Such techniques have been studied extensively in classification [34, 39] and regression [26, 40, 13] as a lightweight way to improve the reliability of model confidence estimates without significantly retraining the underlying predictor. However, for probabilistic TSFMs, their role remains unclear as it is not well understood yet how effective they can be on individual datasets, and which forms of miscalibration they are able to correct.

In this work, we present a systematic study of uncertainty calibration in probabilistic TSFMs. We first introduce a taxonomy of uncertainty quantification mechanisms that organizes TSFMs by the predictive object from which their uncertainty estimates are obtained. Building on this common view, we then develop a calibration evaluation framework for multi-horizon forecasting that goes beyond pooled average coverage and includes stronger quantile-conditioned diagnostics. Finally, we use this framework to analyze calibration and post-hoc recalibration across a broad collection of TSFMs and datasets. Figure 1 summarizes the framework and the main empirical findings.

## Our main contributions are as follows:

• We introduce a unifying taxonomy of uncertainty quantification mechanisms in probabilistic TSFMs, organizing models by the predictive object from which their horizon-wise predictive distributions are obtained.

• We develop a calibration evaluation framework for probabilistic multi-horizon forecasting that applies across these model classes and distinguishes pooled, horizon-wise, and quantile-conditioned notions of calibration.

• We establish theoretical relationships between the resulting calibration metrics, clarifying how pooled and average coverage can hide structured forms of miscalibration, and use these relationships to define diagnostic gap measures for horizon-specific and value-dependent errors.

• We provide a broad empirical study of calibration and post-hoc recalibration across diverse TSFMs and datasets, showing that current models exhibit systematic but heterogeneous miscalibration patterns and that standard recalibration methods often improve average calibration substantially while only partially addressing stronger conditional deficiencies.

## 2 Background and Related Work

Mathematical setup We consider a univariate time series as an ordered sequence of real-valued observations. In forecasting, we observe the most recent $T \in \mathbb { N }$ values, called the context window $\mathbf { x } = ( x _ { 1 } , \dots , x _ { T } ) \in \mathbb { R } _ { \dots } ^ { T }$ , and the goal is to predict the next $H \in \mathbb { N }$ values, called theforecast horizon $\mathbf { y } = ( y _ { 1 } , \dots , y _ { H } ) \in \mathbb { R } ^ { H }$ . Since future values are uncertain, we treat y as a realization of a random vector $\mathbf { Y } = ( Y _ { 1 } , \ldots , Y _ { H } )$ and write $P _ { \mathbf { Y | X = x } }$ for its true conditional distribution given context x, where x is a realization of the random vector $\mathbf { X } = ( X _ { 1 } , \ldots , X _ { T } )$

For scalar-valued targets, uncertainty is often represented through the cumulative distribution function (CDF) and its inverse, the quantile function. For a random variable Y with realization $y \in \mathbb { R }$ , these are defined as

$$
F _ { Y } ( y ) : = P ( Y \leq y ) , \qquad F _ { Y } ^ { - 1 } ( \tau ) : = \operatorname* { i n f } \{ y \in \mathbb { R } : F _ { Y } ( y ) \geq \tau \} ,
$$

where $F _ { Y } ( y )$ gives the probability that $Y$ does not exceed y, and $F _ { Y } ^ { - 1 } ( \tau )$ gives the threshold below which a fraction τ of outcomes fall. Quantiles are especially useful in forecasting because they directly induce prediction intervals. At nominal level $1 - \alpha$ , the interval $[ F _ { Y } ^ { - 1 } ( \alpha / 2 ) , F _ { Y } ^ { - 1 } ( 1 - \overset { . } { \alpha } / 2 ) ]$ will contain the realized outcome with probability $1 - \alpha$ under the true distribution if $F _ { Y }$ is continuous.

The goal of probabilistic forecasting is therefore to estimate, for each context x, the conditional predictive distribution of future values. In this work, we focus on the corresponding horizon-wise marginal predictive CDFs $\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } , h = 1 , \ldots , H$ , as the common probabilistic object that is accessible across the different TSFM families considered below. The mechanism through which these predictive CDFs are represented, however, differs substantially across models.

Uncertainty calibration in forecasting A probabilistic forecast is calibrated if its stated probabilities agree with empirical frequencies [17, 20]. In probabilistic regression, a standard notion is probabilistic calibration [26, 13], which requires predicted quantiles to attain the advertised coverage rate on average across forecast instances. For a one-step forecast and quantile level $\tau \in ( 0 , 1 )$ , this can be written as

$$
\frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { 1 } \Big \{ Y _ { n } \leq \hat { F } _ { Y | \mathbf { X } = \mathbf { x } _ { n } } ^ { - 1 } ( \tau ) \Big \} \approx \tau .
$$

That is, across N forecast instances, the fraction of observed outcomes below the predicted τ-quantile should be close to τ. Averaging the deviation from this identity over a finite set of quantile levels $\mathcal { T } \subset ( 0 , 1 )$ yields the probabilistic calibration error (PCE) [13]:

$$
\mathrm { P C E } = \frac { 1 } { | { \cal T } | } \sum _ { \tau \in { \cal T } } \left| \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { 1 } \Big \{ Y _ { n } \leq \hat { F } _ { Y | \mathbf { X } = \mathbf { x } _ { n } } ^ { - 1 } ( \tau ) \Big \} - \tau \right| .
$$

In multi-horizon forecasting, each prediction produces H forecast steps, and calibration can be summarized across horizons in different ways. The authors of [1] pool all horizon-instance pairs into a single collection, yielding a global calibration measure across the full test set. This gives a useful aggregate view, but it can mask systematic differences across horizons or other subsets of instances. In particular, a model may appear well calibrated under pooled average coverage while still exhibiting structured miscalibration in specific regions of its predictive distribution.

This limitation is closely related to the distinction between average and conditional notions of calibration [17, 18]. Probabilistic calibration only requires coverage to hold on average across all instances, so overconfidence in some regions can be offset by underconfidence in others. Stronger conditional notions ask whether predicted probabilities remain reliable given the model’s own forecasted level, rather than only after averaging over all predictions. Our work builds on this perspective and studies calibration for TSFMs at a finer level than pooled average coverage alone.

Post-hoc recalibration. When predictive uncertainty is miscalibrated, a practical question is whether it can be corrected without retraining the forecasting model. Post-hoc recalibration addresses this by learning a lightweight correction on held-out data and applying it after training. Such methods are particularly attractive for TSFMs, which are large, pretrained, and often used in zero-shot mode.

Popular approaches include isotonic recalibration [26], Platt-style scaling [36], distribution-based recalibration methods [40], and KDE-based methods [13]. While these approaches have shown strong empirical performance in regression settings [13], their applicability and limitations for TSFMs remain largely unexplored.

## 3 A Unifying Taxonomy of Probabilistic TSFMs

Probabilistic TSFMs differ not only in architecture and pretraining data, but also in the predictive object through which they represent forecast uncertainty. To compare calibration across such heterogeneous models, we begin by introducing a unifying perspective on these approaches. What they have in common is that, for each forecast horizon step $h = 1 , \ldots , H$ , every model ultimately provides an estimate of the marginal predictive CDF $\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } .$ The key difference lies in the intermediate representation from which this CDF is obtained and in the training objective used to learn it. Based on this perspective, we distinguish four conceptually different categories.

Category 1: Parametric density models. In this category, TSFMs represent predictive uncertainty through an explicit parametric family. For each horizon step h, the model predicts parameters $\hat { \theta } _ { h }$ of a conditional density $\hat { p } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } ( y ) = p ( y ; \hat { \theta } _ { h } )$ . The predictive CDF is then accessible by integration,

$$
\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } ( y ) = \int _ { - \infty } ^ { y } p ( z ; \hat { \theta } _ { h } ) d z .
$$

For autoregressive heads, later-horizon marginals are obtained by sampling and need not remain in the parametric family. Training in this category is naturally based on negative log-likelihood. At inference time, quantiles and prediction intervals may be available analytically for simple families, but are often obtained numerically or through sampling in practice. This mechanism is, for instance, used by Lag-Llama [37], which employs a Student-t output head, by MOIRAI 1.x and MOIRAI-MoE [42, 30], which use mixture heads, or by Toto 1.0 [11], which uses a Student-t mixture head.

Category 2: Discretized distribution models. In this category, TSFMs represent predictive uncertainty over a finite discretization of the real value axis. Let $b _ { 0 } < b _ { 1 } < \dots < b _ { K }$ denote bin boundaries in R. For each horizon step h, a model in this category predicts probabilities

$$
\hat { p } _ { h , k } = \hat { P } \big ( Y _ { h } \in ( b _ { k - 1 } , b _ { k } ] \mid \mathbf { X } = \mathbf { x } \big ) , \qquad k = 1 , \ldots , K .
$$

This is the mechanism used by Chronos-T5 [5], which quantizes values into tokens and predicts token probabilities autoregressively, and by TabPFN-TS [24], whose bar distribution assigns predictive mass to fixed intervals. For token-based models the predictive CDF is then formed by cumulative summation over the discretized support, yielding a stepwise approximation:

$$
\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } ( y ) = \sum _ { k : b _ { k } \leq y } \hat { p } _ { h , k } .
$$

For bar distributions, the mass is assumed uniform within each bin, so the CDF is not a staircase but increases linearly within each bin. Training is typically carried out with cross-entropy, equivalently a discretized log-likelihood under the induced representation. At inference time, quantiles are obtained by direct inversion of the CDF, whereas autoregressive token-based variants generate forecast values by sampling from the per-step categorical distribution.

Category 3: Direct quantile forecasters. In this category, TSFMs directly predict values of the inverse CDF at a prescribed finite set of probability levels $\tau _ { 1 } , \dots , \tau _ { M } \in ( 0 , 1 )$ . For each horizon step h, the model outputs

$$
\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } ^ { - 1 } ( \tau _ { 1 } ) , \ldots , \hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } ^ { - 1 } ( \tau _ { M } ) .
$$

The model is therefore trained to estimate selected points of the inverse CDF directly rather than a full density or CDF over the outcome space. The standard objective is the pinball loss or weighted variants [10]. At inference time, uncertainty is typically available only at the chosen quantile levels, so its resolution is limited by the quantile grid used during training, although some models additionally offer a continuous quantile head. This is a popular approach used by many recent models such as Chronos-Bolt [6], Chronos-2 [4], MOIRAI 2.0 [29], TiRex [7], TimesFM [12], KAIROS [16], FlowState [19], and YINGLONG [41].

<table><tr><td>#</td><td>Category</td><td>How  $\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } }$  is obtained</td><td>Training loss</td><td></td><td>Representative models</td></tr><tr><td></td><td>Parametric Density</td><td>The model predicts parameters  $\hat { \theta } _ { h }$  ditional density family  $p ( y ; \hat { \theta } _ { h } ) ;$  the pre- dictive CDF may be obtained analytically or via sampling in practice.</td><td>of a con- Negative likelihood</td><td>log-</td><td>Lag-Llama [37],MOIRAI 1.x [42], MOIRAI-MoE [30], Toto 1.0 [11]</td></tr><tr><td></td><td>2 Discretized Distribution</td><td>The model predicts discrete probabilities  $\hat { p } _ { h , k }$  over predefined value intervals; the predictive CDF is obtained by cumulative mass over the discretized support, interpo- lated within bins for bar distributions.</td><td>discretized log- [24] likelihood</td><td></td><td>Cross-entropy / Chronos-T5 [5], TabPFN-TS</td></tr><tr><td>3 Direct</td><td>Quantile</td><td>The model predicts  $\hat { F } ^ { - 1 } ( \tau )$  at finitely many levels  $\tau _ { 1 } , \dots , \tau _ { M } ;$  uncertainty is therefore typically available only at the res- olution of the chosen quantile grid.</td><td>Pinball loss / weighted quan- tile loss</td><td></td><td>Chronos-Bolt [6], Chronos-2 [4], MOIRAI 2.0 [29], TiRex [7], TimesFM [12], KAIROS [16], FlowState [19], YING- LONG [41]</td></tr><tr><td>4</td><td>Implicit Generative Process</td><td>The model generates forecast trajecto- Flow matching ries from an implicit joint predictive law; / diffusion de- marginal predictive CDFs are estimated empirically from Monte Carlo samples.</td><td>noising</td><td></td><td>Sundial [31], TimeDiT [9]</td></tr></table>

Table 1: Taxonomy of uncertainty quantification approaches in TSFMs, organized by the intermediate predictive mechanism learned by the model (density parameters, discretized probability mass, quantile values, or samples), how the predictive CDF is obtained, and the corresponding training objective.

Category 4: Implicit generative process models. The defining feature of this category is that predictive uncertainty is represented implicitly and is only accessible through sampling. Corresponding TSFMs learn a generative process for entire forecast trajectories $\hat { \mathbf { y } } ^ { ( 1 ) } , \hdots , \hat { \mathbf { y } } ^ { ( S ) } , \quad \bar { \hat { \mathbf { y } } } ^ { ( s ) } \in \mathbb { R } ^ { \hat { H } }$ which result as samples from an implicit joint predictive distribution over the full horizon. The horizon-wise predictive CDF is then estimated empirically by:

$$
\hat { F } _ { Y _ { h } | \mathbf { X } = \mathbf { x } } ( y ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathbf { 1 } \Bigl \{ \hat { y } _ { h } ^ { ( s ) } \leq y \Bigr \} .
$$

This mechanism is for instance used by Sundial [31], which learns a conditional generative transport process via flow matching [28], and by TimeDiT [9], which generates trajectories using a diffusionbased denoising objective [22]. In both cases, quantiles and prediction intervals need to be computed from the empirical distribution of sampled paths rather than from an explicit probabilistic output.

This taxonomy separates TSFMs by the intermediate predictive object from which the horizon-wise CDF is obtained, as also summarized in Table 1. This distinction is central for calibration analysis as it clarifies which aspects of the predictive distribution are directly available for evaluation and which calibration diagnostics and post-hoc correction methods naturally apply.

## 4 A Calibration Evaluation Framework for Probabilistic TSFMs

We now introduce a calibration evaluation framework tailored to probabilistic TSFMs. The goal is to evaluate uncertainty estimates in a way that is both comparable across the heterogeneous output types in Section 3 and sufficiently fine-grained for multi-horizon forecasting. To do so, we first refine probabilistic calibration assessment to account for the multi-horizon setting, and then move to a stronger conditional notion of calibration that captures whether uncertainty estimates are also trustworthy at the level of individual predictions. Throughout, we fix a finite set of quantile level $\mathcal { T } = \{ \tau _ { 1 } , . . . , \tau _ { Q } \} \subset ( 0 , 1 )$ and write $\hat { q } _ { h , \tau _ { q } } : = \hat { F } _ { Y _ { h } | \mathbf { X } } ^ { - 1 } ( \tau _ { q } )$ for the grid levels.

Multi-horizon probabilistic calibration. A natural extension of the probabilistic calibration error from Section 2 is to pool all horizon-instance pairs and compare empirical frequencies with nominal quantile levels over the full test set. Following [1], we define

$$
\mathrm { P C E } _ { \mathrm { p o o l } } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \left| \frac { 1 } { H } \sum _ { h = 1 } ^ { H } P \big ( Y _ { h } \leq \hat { q } _ { h , \tau _ { q } } \big ) - \tau _ { q } \right| .
$$

This provides a concise global summary, but it can hide systematic differences across forecast steps because over- and undercoverage at different horizons may cancel after pooling. To avoid this, we evaluate calibration separately at each horizon and aggregate the resulting errors by their mean and maximum. For $h = 1 , \bar { \ldots } , \bar { H }$ we define:

$$
\mathrm { P C E } _ { h } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \left| P \mathopen { } \mathclose \bgroup \left( Y _ { h } \leq \hat { q } _ { h , \tau _ { q } } \aftergroup \egroup \right) - \tau _ { q } \aftergroup \egroup \right|
$$

such that $\begin{array} { r } { \mathrm { P C E } _ { \mathrm { a v g } } ~ = ~ \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { P C E } _ { h } } \end{array}$ summarizes the typical error across the horizon, while $\mathrm { P C E } _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { h = 1 , \ldots , H } \overline { { \mathrm { P C } } } \ddot { \mathrm { E } } _ { h }$ identifies the worst forecast step.

Quantile-conditioned calibration. Even when probabilistic calibration error is small, a model may still be unreliable in a more structured way. Its predictive intervals may be too narrow for some predicted quantile values and too wide for others, with these effects averaging out overall. To capture such value-dependent distortions, we consider the stronger notion of quantile-conditioned calibration [18], which requires calibration to hold conditional on the model’s own predicted quantile value.

Definition 4.1 (Quantile-conditioned calibration). A predictive CDF $\hat { F }$ for a real-valued outcome Y is quantile-conditioned calibrated if

$$
P \Big ( Y \leq \hat { F } ^ { - 1 } ( \tau ) \Big | \hat { F } ^ { - 1 } ( \tau ) \Big ) = \tau \qquad \mathrm { f o r ~ a l l ~ } \tau \in ( 0 , 1 ) , \mathrm { a l m o s t ~ s u r e l y } .
$$

In words, among all instances for which the model predicts similar values for the τ-quantile, the realized outcome should fall below that value a fraction τ of the time. For continuous predictive CDFs, this is the conditional exceedance probability (CEP) calibration discussed in [18]. This notion is stronger than average probabilistic calibration because it conditions on the model output itself rather than averaging over all predictions. At the same time, it remains practically approachable through nonparametric estimation (see Appendix B.3), which makes it useful as a diagnostic even though it is more challenging to estimate than average coverage. We then define

$$
\mathrm { Q C E } _ { h } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \mathbb { E } \big [ \big | P \big ( Y _ { h } \leq \hat { q } _ { h , \tau _ { q } } \bigm | \hat { q } _ { h , \tau _ { q } } \bigm ) - \tau _ { q } \bigm | \big ] , \qquad h = 1 , \ldots , H ,
$$

and aggregate again via

$$
\mathrm { Q C E } _ { \mathrm { a v g } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { Q C E } _ { h } , \qquad \mathrm { Q C E } _ { \mathrm { m a x } } = \operatorname* { m a x } _ { h = 1 , \ldots , H } \mathrm { Q C E } _ { h } .
$$

Thus, $\mathrm { Q C E _ { a v g } }$ and $\mathrm { Q C E } _ { \operatorname* { m a x } }$ are horizon-aware counterparts of PCE based on a stronger calibration requirement that can still be estimated in practice. This is particularly useful for TSFMs because models with similar values of $\mathrm { P C E _ { a v g } }$ can still differ substantially in how reliable their uncertainty estimates are across different kinds of predictions. We estimate it by isotonic regression, whose population target lower-bounds $\mathrm { Q C E } _ { h }$ , with equality for non-decreasing conditional coverage (see Appendix B.3).

A hierarchy of calibration errors. The metrics introduced above form a natural ordering which can be expressed by the following proposition:

Proposition 4.2. For any predictive model and any finite quantile grid $\tau ,$

$$
\mathrm { P C E } _ { \mathrm { p o o l } } \ \leq \ \mathrm { P C E } _ { \mathrm { a v g } } \ \leq \ \mathrm { Q C E } _ { \mathrm { a v g } } \ \leq \ \mathrm { Q C E } _ { \mathrm { m a x } } .
$$

This proposition clarifies the role of the different metrics. $\mathrm { P C E _ { p o o l } }$ is the weakest summary because it aggregates over all horizons and instances at once. $\mathrm { P C E _ { a v g } }$ is stricter because it evaluates each horizon separately before aggregation. $\mathrm { Q C E _ { a v g } }$ is stricter still because it assesses whether calibration remains valid conditional on the model’s own predicted quantile values. Finally, $\mathrm { Q C E } _ { \operatorname* { m a x } }$ upperbounds all of them and highlights the worst-calibrated forecast horizon under the strongest notion considered here.

Definition 4.3 (Horizon gap and conditioning gap). We define the horizon gap $\Delta _ { H }$ and conditioning gap $\Delta _ { C }$ as additional diagnostics by

$$
\Delta _ { H } : = \mathrm { P C E } _ { \mathrm { a v g } } - \mathrm { P C E } _ { \mathrm { p o o l } } \geq 0 , \qquad \Delta _ { C } : = \mathrm { Q C E } _ { \mathrm { a v g } } - \mathrm { P C E } _ { \mathrm { a v g } } \geq 0 .
$$

The horizon gap $\Delta _ { H }$ measures how much calibration error is hidden by pooling across horizons. A large value indicates that reliability differs substantially across forecast steps, even if the pooled summary appears acceptable. The conditioning gap $\Delta _ { C }$ measures how much additional error becomes visible once one conditions on the model’s own predicted quantile values. A large conditioning gap therefore points to structured value-dependent miscalibration that is invisible under average coverage alone. Together, these diagnostics help separate horizon-specific effects from deeper conditional unreliability and will also be useful when interpreting post-hoc recalibration results. Since finitesample estimates of both gaps are positive even when no such structure is present, we accompany them with significance tests (see Appendix B.4).

Implications for post-hoc recalibration. The proposed evaluation framework can be used not only to assess how well TSFMs are calibrated out of the box, but also to analyze which types of miscalibration can be improved by post-hoc correction. Quantile relabeling methods [13] are particularly appealing for probabilistic TSFMs as t hese methods preserve the support and ordering of predicted outcomes and only redistribute probability mass to improve calibration. This makes them lightweight and leaves the pretrained model itself unchanged. Formally, such methods estimate a monotone map $g : [ 0 , 1 ]  { \overline { { [ 0 , 1 ] } } }$ from a small held-out calibration set and redefine

$$
\begin{array} { r } { \hat { F } _ { Y _ { h } | \mathbf { X } } ^ { - 1 , \mathrm { g } } ( \tau ) : = \hat { F } _ { Y _ { h } | \mathbf { X } } ^ { - 1 } \big ( g ( \tau ) \big ) . } \end{array}
$$

For models that output quantiles only on a finite grid, $\hat { F } _ { Y _ { h } | \mathbf { X } } ^ { - 1 }$ is interpolated between the grid levels. To understand how such recalibration interacts with our diagnostic framework, we characterize which forms of miscalibration can and cannot be addressed by a single global relabeling map.

Theorem 4.4. Let $g : [ 0 , 1 ]  [ 0 , 1 ]$ be any relabeling map, and let $\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g } , \mathrm { P C E } _ { \mathrm { a v g } } ^ { g } ,$ and $\mathrm { Q C E } _ { \mathrm { a v g } } ^ { g }$ denote the corresponding calibration errors after recalibration with g. Then thefollowing hold:

(i) If the pooled coverage $\begin{array} { r } { \bar { p } ( t ) : = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } P \big ( Y _ { h } \leq \hat { F } _ { Y _ { h } | \mathbf { X } } ^ { - 1 } ( t ) \big ) } \end{array}$  is continuous in t, there exists a non-decreasing map $g ^ { \star }$ with $\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g ^ { \star } } = 0 .$

(ii) For every g, $\mathrm { P C E } _ { \mathrm { a v g } } ^ { g } \ \geq \ \Delta _ { H } / 2 ,$ , where $\Delta _ { H }$ is the horizon gap of the original forecast. In particular, if the original forecast exhibits a non-zero horizon gap $\Delta _ { H } \ > \ 0$ , then $\mathrm { P \dot { C } E } _ { \mathrm { a v g } } ^ { g } > 0 \mathrm { \dot { \it f o r } }$ every g, including any g with $\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g } = 0 .$

(iii) For $e \nu e r y \varepsilon \in ( 0 , 1 / 2 )$ , there exists a forecasting setup with continuous p¯ in which every g with $\mathrm { P C i E } _ { \mathrm { p o o l } } ^ { g } \dot { = } 0$ satisfies $\mathrm { Q C E _ { a v g } ^ { \it g } - Q C E _ { a v g } ^ { \mathrm { } } > \varepsilon . }$

This theorem highlights key limitations of global relabeling-based recalibration. In principle, relabeling is able to eliminate the pooled probabilistic calibration error entirely, a map that isotonic recalibration approximates by estimating the pooled coverage curve. However, this is the weakest notion in our hierarchy. If miscalibration differs across horizons, then no global relabeling can remove horizon-wise errors simultaneously. At each quantile level, the map moves the coverage of all horizons in the same direction, so it can reduce over-coverage at some horizons only by increasing under-coverage at others. Part (ii) shows that half of the horizon gap is a floor for $\mathrm { P C E _ { a v g } }$ that no global relabeling can go below. Moreover, success under pooled coverage does not imply improvement under the stronger quantile-conditioned notion. Part (iii) shows that there are setups in which every map that removes the pooled error increases $\mathrm { Q C E _ { a v g } }$ by almost $1 / 2$ . This additionally motivates the proposed hierarchy for evaluating recalibration of probabilistic TSFMs as pooled summaries alone may overstate the extent to which uncertainty estimates have truly been improved. All proofs, together with the intuition behind them, are given in Appendix A.

## 5 Experiments

Setup. We now apply the proposed calibration framework to a comprehensive empirical benchmark. We evaluate 14 zero-shot probabilistic TSFMs spanning all four uncertainty quantification

$$
\Delta _ { \mathbf { C } }
$$

<table><tr><td>Cat.</td><td>Model</td><td> $\bf { P C E _ { p o o l } }$ </td><td> $\mathbf { P C E _ { a v g } }$ </td><td> $\mathbf { P C E _ { m a x } }$ </td><td> $\mathbf { Q C E _ { a v g } }$ </td><td> $\mathbf { Q C E _ { m a x } }$ </td><td> $\pmb { \Delta } _ { \mathbf { H } }$ </td><td>raw</td><td>corr.</td><td> $\pmb { \Delta } _ { \mathbf { H } }$ </td><td> $\Delta _ { \mathbf { C } }$ </td></tr><tr><td>1</td><td>Moirai-Base</td><td>0.0536</td><td>0.0829</td><td>0.1579</td><td>0.1215</td><td>0.1862</td><td>0.0293</td><td>0.0386</td><td>0.0199</td><td>76%</td><td>76%</td></tr><tr><td>1</td><td>Moirai-Large</td><td>0.0484</td><td>0.0776</td><td>0.1581</td><td>0.1225</td><td>0.1906</td><td>0.0292</td><td>0.0448</td><td>0.0249</td><td>80%</td><td>82%</td></tr><tr><td>1</td><td>Moirai-MoE-Base</td><td>0.0533</td><td>0.0826</td><td>0.1537</td><td>0.1204</td><td>0.1878</td><td>0.0293</td><td>0.0379</td><td>0.0194</td><td>84%</td><td>76%</td></tr><tr><td>1</td><td>Moirai-MoE-Small</td><td>0.0521</td><td>0.0837</td><td>0.1578</td><td>0.1234</td><td>0.1908</td><td>0.0317</td><td>0.0397</td><td>0.0215</td><td>88%</td><td>76%</td></tr><tr><td>1</td><td>Moirai-Small</td><td>0.0530</td><td>0.0855</td><td>0.1607</td><td>0.1283</td><td>0.1982</td><td>0.0326</td><td>0.0428</td><td>0.0238</td><td>86%</td><td>73%</td></tr><tr><td>2</td><td>Chronos-T5-Base</td><td>0.1024</td><td>0.1240</td><td>0.1939</td><td>0.1491</td><td>0.2159</td><td>0.0216</td><td>0.0251</td><td>0.0075</td><td>84%</td><td>49%</td></tr><tr><td>2</td><td>Chronos-T5-Small</td><td>0.1035</td><td>0.1253</td><td>0.1960</td><td>0.1524</td><td>0.2183</td><td>0.0218</td><td>0.0271</td><td>0.0092</td><td>80%</td><td>53%</td></tr><tr><td>2</td><td>TabPFN-TS</td><td>0.0629</td><td>0.0872</td><td>0.1537</td><td>0.1198</td><td>0.1797</td><td>0.0243</td><td>0.0325</td><td>0.0149</td><td>78%</td><td>57%</td></tr><tr><td>3</td><td>Chronos-2</td><td>0.0641</td><td>0.0919</td><td>0.1653</td><td>0.1229</td><td>0.1881</td><td>0.0278</td><td>0.0310</td><td>0.0115</td><td>78%</td><td>63%</td></tr><tr><td>3</td><td>Chronos-Bolt-Base</td><td>0.0505</td><td>0.0804</td><td>0.1494</td><td>0.1202</td><td>0.1838</td><td>0.0299</td><td>0.0398</td><td>0.0180</td><td>88%</td><td>69%</td></tr><tr><td>3</td><td>Chronos-Bolt-Small</td><td>0.0523</td><td>0.0833</td><td>0.1569</td><td>0.1225</td><td>0.1890</td><td>0.0310</td><td>0.0393</td><td>0.0185</td><td>84%</td><td>67%</td></tr><tr><td>3</td><td>TiRex</td><td>0.0502</td><td>0.0796</td><td>0.1485</td><td>0.1161</td><td>0.1798</td><td>0.0294</td><td>0.0365</td><td>0.0174</td><td>80%</td><td>69%</td></tr><tr><td>3</td><td>TimesFM-2.5-200M</td><td>0.0743</td><td>0.1011</td><td>0.1711</td><td>0.1254</td><td>0.1900</td><td>0.0268</td><td>0.0243</td><td>0.0058</td><td>86%</td><td>45%</td></tr><tr><td>4</td><td>Sundial-Base</td><td>0.1467</td><td>0.1678</td><td>0.2413</td><td>0.1958</td><td>0.2629</td><td>0.0211</td><td>0.0280</td><td>0.0088</td><td>82%</td><td>49%</td></tr></table>

Table 2: Calibration metrics across probabilistic TSFMs, averaged over the 55 dataset configurations. Lower is better for all calibration errors and diagnostics. $\Delta _ { C }$ corr. subtracts the part of the conditioning gap expected from estimation noise alone, and the last two columns give the share of model–dataset pairs whose gap is significant at level 0.05. Both use the 49 configurations with at least two series, for which the resampling tests are defined (Appendix B.4). The category label corresponds to the uncertainty quantification taxonomy in Table 1.

categories from Section 3: five parametric density models, three discretized distribution models, five direct quantile models, and one implicit generative model. Here, zero-shot means that no model is finetuned on the evaluated datasets. Each model is tested on the short-term forecasting task of 55 dataset configurations from the GIFT-Eval benchmark [2] covering diverse domains and sampling frequencies. The recalibration experiments additionally require at least 50 forecast windows per dataset, which leaves 42 configurations. To ensure comparability across model classes, all calibration metrics are computed on the common quantile grid $\mathcal { T } = \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ . For models that do not directly expose these quantiles, they are extracted from the corresponding predictive distribution or empirical samples. Quantile-conditioned calibration errors are estimated nonparametrically by isotonic regression of the coverage indicators on the predicted quantile values, and we test the significance of both diagnostic gaps for every model–dataset pair with at least two series. Full implementation details are provided in Appendix B and accompanying code is available at https://github.com/thomdeck/tsfm-calibration.

Fine-grained calibration analysis. Table 2 reports calibration results aggregated across datasets for all evaluated TSFMs. We highlight three observations. First, $\mathrm { P C E _ { a v g } ^ { - } }$ is consistently larger than $\mathrm { P C E _ { p o o l } }$ (in 98% of the model–dataset pairs, and by 40% on average), showing that pooled coverage systematically understates calibration error by averaging over forecast steps with different reliability profiles. Second, the further increase to $\mathrm { Q C E _ { a v g } }$ is substantial across all model families, indicating that even horizon-wise average coverage misses structured value-dependent errors that only become visible under the stronger quantile-conditioned criterion. The gap diagnostics make this explicit. $\Delta _ { H }$ captures uneven calibration across the horizon, whereas $\Delta _ { C }$ captures residual conditional distortions beyond average coverage. Both gaps are significant for most model–dataset pairs with at least 50 forecast windows $( \Delta _ { H } \colon 8 6 \% , \Delta _ { C } \colon 6 9 \%$ , see also Appendix B.4). Third, parametric and direct quantile models tend to display the strongest calibration profile across the considered metrics, although the differences remain dataset-dependent. Chronos-T5 and Sundial combine large pooled errors with small corrected conditioning gaps, so their miscalibration appears to be mostly marginal. Sundial’s nominal 80% interval, for instance, covers only 40% of the outcomes.

Figure 1(left) complements this aggregate view by showing $\mathrm { P C E _ { p o o l } }$ against $\mathrm { Q C E _ { a v g } }$ for individual model-dataset pairs. Consistent with the calibration hierarchy, all points lie above the diagonal. This gap varies considerably across datasets, indicating that the extent of hidden miscalibration is highly setting-dependent. Further metric pairs are shown in Appendix C.

![](images/b555eb929f19b04c284da35263ae235a24c0180a7c1700b7469b132d569a25b0.jpg)

![](images/25f06aa68649f573cd40f480170fb6a3ab717ba5f5f360df26ae41658f144f3f.jpg)

![](images/ca7256858a0fb9508a8c9bb31c4cd0678da39a2483cc5bf8fc785542d1cfb576.jpg)  
Figure 2: Concrete effect of isotonic recalibration (IR). (a) 90% prediction interval of one forecast before (blue) and after (green) IR; over all test windows, relabeling improves the coverage from 77% to 91%. (b, c) Share of outcomes beyond a predicted quantile and differences per quintile of its predicted value. Despite low PCE scores, the actual coverage varies substantially across value ranges.

Why the quantile-conditioned view matters. Figure 2 illustrates what global recalibration can and cannot correct. Marginal miscalibration can matter greatly for individual forecasts: the 90% intervals of Chronos-2 on LOOP\_SEATTLE/D cover only 77% of the later outcomes, and isotonic recalibration on the earliest windows raises this to 91%. Panels (b) and (c) show two pairs that look well calibrated by standard metrics, but grouping the forecasts by their predicted value tells a different story. When Moirai-Small predicts pollution peaks on kdd\_cup\_2018/H, 27% of the outcomes fall below the predicted 10% quantile. When Chronos-T5-Small predicts congestion on LOOP\_SEATTLE/5T, 29% of the speeds exceed the predicted 90% quantile instead of 10%. These errors have opposite signs for low and high predictions and cancel in marginal coverage, but not in the conditioning gap $( \Delta _ { C } = 0 . 0 3 9$ and 0.044, both significant). Because one relabeling map applies to all forecasts, recalibration shifts all quintiles alike, and the value-dependent pattern persists $( \Delta _ { C } \colon$ $0 . 0 4 1  0 . 0 3 8$ and 0.040 → 0.038 on the test windows). Removing it would require recalibration that conditions on the forecast itself. Appendix C.2 confirms this for further models and datasets.

Post-hoc recalibration effectiveness. We next study how effectively standard post-hoc methods can correct different forms of miscalibration. We evaluate four widely used recalibration methods: isotonic recalibration (IR) [26], Platt scaling (PS) [36], histogram binning (HB) [43], and KDE-based calibration (KDE) [13]. To reflect a deployment-relevant scenario, we use a chronological split per dataset such that the recalibrator is fitted on the earliest 20% of the forecast windows of every series and evaluated on the remaining, later windows. For datasets consisting only of single-window series, 20% of the series are held out for calibration instead (Appendix B.5). Table 3 reports the resulting relative changes of the calibration errors, where negative values are improvements. Histogram binning coincides with IR at the fitted quantile levels and is deferred to Appendix C.4.

Across models, all methods improve calibration to a broadly similar extent. As suggested by the theory, the gains are strongest for $\mathrm { P C E } _ { \mathrm { p o o l } } ( 3 7 \%$ for IR), which is the calibration notion most directly targeted by global relabeling, smaller $\mathrm { f o r } \mathrm { P C E } _ { \mathrm { a v g } }$ (19%), and smaller still for $\mathrm { Q C E _ { a v g } }$ (15%), while the horizon gap even increases by 24% and the conditioning gap is essentially unchanged (see Appendix C.3). This supports the view that post-hoc relabeling is useful for correcting average probabilistic misalignment, but less able to remove deeper value-conditional distortions. Consistently, the models whose miscalibration is mostly marginal benefit most. Chronos-T5, Sundial, and TimesFM lose 51–62% of their pooled error, compared with 9–24% for the Moirai and Chronos-Bolt models.

Figure 1(right) shows that $\mathrm { P C E _ { a v g } }$ improves for 62% of the model–dataset pairs, with the largest gains where the original miscalibration is most severe. Overall, the results indicate that post-hoc recalibration is a practical and data-efficient way to improve the reliability of TSFM uncertainty estimates, but it should be evaluated with more than pooled coverage alone to faithfully judge its effectiveness.

<table><tr><td></td><td colspan="3"> $\mathbf { P C E _ { p o o l } }$ </td><td colspan="3"> $\mathbf { P C E _ { a v g } }$ </td><td colspan="3"> $\mathbf { Q C E _ { a v g } }$ </td></tr><tr><td>Model</td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td></tr><tr><td>Moirai-Base</td><td>-13.36%</td><td>-12.93%</td><td>-12.16%</td><td>-4.77%</td><td>-4.58%</td><td>-4.08%</td><td>-5.79%</td><td>-5.66%</td><td>-4.95%</td></tr><tr><td>Moirai-Large</td><td>-12.44%</td><td>-11.78%</td><td>-9.08%</td><td>-0.52%</td><td>-0.29%</td><td>+0.95%</td><td>-1.57%</td><td>-1.46%</td><td>-0.49%</td></tr><tr><td>Moirai-MoE-Base</td><td>-19.72%</td><td>-19.62%</td><td>-13.91%</td><td>-5.83%</td><td>-5.77%</td><td>-4.09%</td><td>-5.51%</td><td>-5.43%</td><td>-4.21%</td></tr><tr><td>Moirai-MoE-Small</td><td>-18.77%</td><td>-19.05%</td><td>-16.24%</td><td>-7.78%</td><td>-7.82%</td><td>-7.07%</td><td>-5.62%</td><td>-5.58%</td><td>-4.92%</td></tr><tr><td>Moirai-Small</td><td>-9.29%</td><td>-9.45%</td><td>-7.89%</td><td>-3.55%</td><td>-3.44%</td><td>-2.30%</td><td>-2.23%</td><td>-2.32%</td><td>-1.59%</td></tr><tr><td>Chronos-T5-Base</td><td>-62.23%</td><td>-62.12%</td><td>-60.25%</td><td>-39.04%</td><td>-38.96%</td><td>-37.90%</td><td>-32.40%</td><td>-32.30%</td><td>-31.56%</td></tr><tr><td>Chronos-T5-Small</td><td>-60.38%</td><td>-60.29%</td><td>-58.23%</td><td>-38.53%</td><td>-38.48%</td><td>-37.45%</td><td>-31.14%</td><td>-31.06%</td><td>-30.24%</td></tr><tr><td>TabPFN-TS</td><td>-21.20%</td><td>−21.21%</td><td>-17.18%</td><td>-10.67%</td><td>-10.57%</td><td>-9.61%</td><td>-10.01%</td><td>-9.92%</td><td>-9.25%</td></tr><tr><td>Chronos-2</td><td>-40.08%</td><td>-40.17%</td><td>-39.83%</td><td>-17.30%</td><td>-17.17%</td><td>-18.21%</td><td>-14.79%</td><td>-14.66%</td><td>-14.73%</td></tr><tr><td>Chronos-Bolt-Base</td><td>-12.36%</td><td>-11.89%</td><td>-8.42%</td><td>-3.87%</td><td>-3.48%</td><td>-3.17%</td><td>-4.54%</td><td>-4.39%</td><td>-4.16%</td></tr><tr><td>Chronos-Bolt-Small</td><td>-24.45%</td><td>-24.03%</td><td>-19.27%</td><td>-8.93%</td><td>-8.64%</td><td>-7.83%</td><td>-6.79%</td><td>-6.71%</td><td>-5.91%</td></tr><tr><td>TiRex</td><td>-16.31%</td><td>-15.85%</td><td>-14.55%</td><td>-4.82%</td><td>-4.64%</td><td>-5.16%</td><td>-6.46%</td><td>-6.42%</td><td>-6.12%</td></tr><tr><td>TimesFM-2.5-200M</td><td>-51.33%</td><td>-51.14%</td><td>-45.27%</td><td>-29.13%</td><td>-29.00%</td><td>-26.38%</td><td>-26.33%</td><td>-26.21%</td><td>-23.94%</td></tr><tr><td>Sundial-Base</td><td>-57.58%</td><td>-57.60%</td><td>-57.16%</td><td>-39.44%</td><td>-39.44%</td><td>-39.22%</td><td>-33.85%</td><td>-33.86%</td><td>-33.68%</td></tr></table>

Table 3: Relative change of the calibration errors after post-hoc recalibration under the chronological split, i.e., the change of each model’s mean error over the 42 dataset configurations. Negative values are improvements, and bold numbers mark the best recalibration method for each model and metric. With a single exception, all errors decrease, but the gains are largest for $\mathrm { P C E _ { p o o l } }$ and decrease for $\mathrm { P C E _ { a v g } }$ and $\mathrm { Q C E _ { a v g } }$ , highlighting again that recalibration via CDF relabeling mainly targets probabilistic miscalibration.

Additional experiments. The appendix complements these results with several further analyses. Simulations with independent series confirm that the significance tests of both gaps hold their nominal level and reliably separate horizon-dependent from value-dependent miscalibration (Appendix B.4). Further case studies show that the pattern of Figure 2 is no isolated example as across models and datasets, forecasts tend to be too extreme in both directions, with errors that cancel in average coverage. They also illustrate the complementary failure mode, where large coverage errors across the hours of the day cancel under pooling and keep recalibration above the floor of Theorem 4.4(ii) (Appendix C.2). When the calibration windows are drawn at random rather than from the past, recalibration is even more effective, with IR reducing $\mathrm { P C E _ { p o o l } }$ by 67% instead of 37% (Appendix C.3). The recalibration methods perform similarly because they estimate the same empirical coverage curve at the fitted levels, and they differ mainly when extrapolating to tail quantiles, where Platt scaling performs best (Appendix C.4). Finally, our conclusions are robust to tail quantiles, the number of Monte Carlo samples, scale differences between series, and point masses, while longer horizons enlarge the horizon gap (Appendix C.5).

## 6 Discussion and Conclusion

We presented a calibration evaluation framework for probabilistic TSFMs that goes beyond pooled average coverage and enables a more fine-grained diagnosis of uncertainty reliability in multihorizon forecasting. Our results show that standard pooled metrics can systematically understate miscalibration by masking both differences across forecast steps and additional value-dependent errors. Across 14 models and 55 datasets, we find that such miscalibration is widespread in current TSFMs, regardless of how predictive uncertainty is represented. We further studied post-hoc recalibration as a lightweight practical remedy. Both our analysis and experiments show that standard recalibration methods can substantially improve weaker average notions of calibration, also when they are fitted on past data only. They are less effective at removing stronger conditional deficiencies. The main limitations of our study lie in its scope as we focus only on horizon-wise marginal calibration and simple post-hoc recalibration methods. Moreover, our isotonic estimate of quantile-conditioned calibration is conservative, and point masses in the outcomes can inflate coverage errors. Extending the analysis to joint forecast calibration across time [3], more expressive recalibration methods that condition on the forecast horizon or on the forecast itself [40, 25], and training-time approaches for improving calibration [14] are promising directions for future work. Overall, we hope that the framework introduced here helps make calibration evaluation a more standard and informative part of assessing probabilistic TSFMs in practice.

## References

[1] Coen Adler, Yuxin Chang, Felix Draxler, Samar Abdi, and Padhraic Smyth. Beyond accuracy: Are time series foundation models well-calibrated? In International Conference on Learning Representations (ICLR), 2026.

[2] Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Gift-eval: A benchmark for general time series forecasting model evaluation. arXiv preprint arXiv:2410.10393, 2024.

[3] Sam Allen, Johanna Ziegel, and David Ginsbourger. Assessing the calibration of multivariate probabilistic forecasts. Quarterly Journal ofthe Royal Meteorological Society, 150(760):1315– 1335, 2024.

[4] Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting, 2025. arXiv:2510.15821.

[5] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. arXiv:2403.07815.

[6] Abdul Fatir Ansari, Caner Turkmen, Oleksandr Shchur, and Lorenzo Stella. Fast and accurate zero-shot forecasting with Chronos-Bolt and AutoGluon. https://aws.amazon.com/blogs/machine-learning/ fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/, 2024.

[7] Andreas Auer, Patrick Podestà, Daniel Klotz, Sebastian Böck, Günter Klambauer, and Sepp Hochreiter. TiRex: Zero-shot forecasting across long and short horizons with enhanced incontext learning, 2025. arXiv:2505.23719.

[8] Richard E Barlow and Hugh D Brunk. The isotonic regression problem and its dual. Journal of the American Statistical Association, 67(337):140–147, 1972.

[9] Defu Cao, Wen Ye, Yizhou Zhang, and Yan Liu. Timedit: General-purpose diffusion transformers for time series foundation model. arXiv preprint arXiv:2409.02322, 2024.

[10] Youngseog Chung, Willie Neiswanger, Ian Char, and Jeff Schneider. Beyond pinball loss: Quantile methods for calibrated uncertainty quantification. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 10971–10984, 2021.

[11] Ben Cohen, Emaad Khwaja, Youssef Doubli, Salahidine Lemaachi, Chris Lettieri, Charles Masson, Hugo Miccinilli, Elise Ramé, Qiqi Ren, Afshin Rostamizadeh, Jean Ogier du Terrail, Anna-Monica Toon, Kan Wang, Stephan Xie, Zongzhe Xu, Viktoriya Zhukova, David Asker, Ameet Talwalkar, and Othmane Abou-Amal. This time is different: An observability perspective on time series foundation models, 2025. arXiv:2505.14766.

[12] Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), 2024.

[13] Victor Dheur and Souhaib Ben Taieb. A large-scale study of probabilistic calibration in neural network regression. In International Conference on Machine Learning (ICML), pages 7813– 7836. PMLR, 2023.

[14] Victor Dheur and Souhaib Ben Taieb. Probabilistic calibration by design for neural network regression. In International Conference on Artificial Intelligence and Statistics, pages 3133– 3141. PMLR, 2024.

[15] Timo Dimitriadis, Tilmann Gneiting, and Alexander I Jordan. Stable reliability diagrams for probabilistic classifiers. Proceedings of the National Academy of Sciences, 118(8):e2016191118, 2021.

[16] Kun Feng, Shaocheng Lan, Yuchen Fang, Wenchao He, Lintao Ma, Xingyu Lu, and Kan Ren. Kairos: Toward adaptive and parameter-efficient time series foundation models. arXiv preprint arXiv:2509.25826, 2025.

[17] Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E. Raftery. Probabilistic forecasts, calibration and sharpness. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(2):243–268, 2007.

[18] Tilmann Gneiting and Johannes Resin. Regression diagnostics meets forecast evaluation: Conditional calibration, reliability diagrams, and coefficient of determination. Electronic Journal of Statistics, 17(2):3226–3286, 2023.

[19] Lars Graf, Thomas Ortner, Stanisław Wo´zniak, and Angeliki Pantazi. FlowState: Samplingrate-equivariant time-series forecasting. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML), 2026. arXiv:2508.05287.

[20] Sebastian Gruber and Florian Buettner. Better uncertainty calibration via proper scores for classification and beyond. Advances in Neural Information Processing Systems, 35:8618–8632, 2022.

[21] Xiao Gu, Yu Liu, Zaineb Mohsin, Jonathan Bedford, Anshul Thakur, Peter Watkinson, Lei Clifton, Tingting Zhu, and David Clifton. Are time series foundation models ready for vital sign forecasting in healthcare? In Proceedings of the 4th Machine Learning for Health Symposium, volume 259, pages 401–419. PMLR, 2025.

[22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[23] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025.

[24] Shi Bin Hoo, Samuel Müller, David Salinas, and Frank Hutter. From tables to time: Extending TabPFN-v2 to time series forecasting, 2025. arXiv:2501.02945.

[25] Volodymyr Kuleshov and Shachi Deshpande. Calibrated and sharp uncertainties in deep learning via density estimation. In International Conference on Machine Learning, pages 11683–11693. PMLR, 2022.

[26] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon. Accurate uncertainties for deep learning using calibrated regression. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80, pages 2801–2809. PMLR, 2018.

[27] Yuxuan Liang, Haomin Wen, Yuqi Nie, Yushan Jiang, Ming Jin, Dongjin Song, Shirui Pan, and Qingsong Wen. Foundation models for time series analysis: A tutorial and survey. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pages 6555–6565, 2024.

[28] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[29] Chenghao Liu, Taha Aksu, Juncheng Liu, Xu Liu, Hanshu Yan, Quang Pham, Silvio Savarese, Doyen Sahoo, Caiming Xiong, and Junnan Li. Moirai 2.0: When less is more for time series forecasting, 2025. arXiv:2511.11698.

[30] Xu Liu, Juncheng Liu, Gerald Woo, Taha Aksu, Yuxuan Liang, Roger Zimmermann, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Moirai-MoE: Empowering time series foundation models with mixture of experts. arXiv preprint arXiv:2410.10469, 2024.

[31] Yong Liu, Guo Qin, Zhiyuan Shi, Zhe Chen, Chuanyang Yang, Xue Huang, Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time series foundation models. arXiv preprint arXiv:2502.00816, 2025.

[32] Simon J Mason, Jacqueline S Galpin, Lisa Goddard, Nicholas E Graham, and Balakrishnan Rajaratnam. Conditional exceedance probabilities. Monthly Weather Review, 135(2):363–372, 2007.

[33] Marcel Meyer, David Zapata Gonzalez, Sascha Kaltenpoth, and Oliver Müller. Benchmarking time series foundation models for short-term household electricity load forecasting. IEEE Access, 13:218141–218153, 2025.

[34] Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. Advances in Neural Information Processing Systems, 34:15682–15694, 2021.

[35] Belinda Phipson and Gordon K Smyth. Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1), 2010.

[36] John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999.

[37] Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Hena Ghonia, Rishika Bhagwatkar, Arian Khorasani, Mohammad Javad Darvish Bayazi, George Adamopoulos, Roland Riachy, Nadhir Hassan, et al. Lag-Llama: Towards foundation models for probabilistic time series forecasting, 2023. arXiv:2310.08278.

[38] Tim Robertson, F T Wright, and Richard L Dykstra. Order Restricted Statistical Inference. Wiley, Chichester, 1988.

[39] Telmo Silva Filho, Hao Song, Miquel Perello-Nieto, Raul Santos-Rodriguez, Meelis Kull, and Peter Flach. Classifier calibration: a survey on how to assess and improve predicted class probabilities. Machine Learning, 112(9):3211–3260, 2023.

[40] Hao Song, Tom Diethe, Meelis Kull, and Peter A. Flach. Distribution calibration for regression. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97, pages 5897–5906. PMLR, 2019.

[41] Xue Wang, Tian Zhou, Jinyang Gao, Bolin Ding, and Jingren Zhou. Output scaling: Ying-Long – delayed chain of thought in a large pretrained time series forecasting model, 2025. arXiv:2506.11029.

[42] Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), JMLR.org, 2024.

[43] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings ofthe eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699, 2002.

[44] Zhuohang Zhu, Haodong Chen, Qiang Qu, and Vera Chung. Fincast: A foundation model for financial time-series forecasting. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, pages 4539–4549, 2025.

## A Theoretical Results

This appendix proves the calibration hierarchy (Proposition 4.2) and the limits of global relabeling (Theorem 4.4), and characterizes the population target of the isotonic estimator (Lemma A.1). All arguments rest on two elementary observations. First, the absolute value of an average never exceeds the average of absolute values, and the difference measures how much positive and negative terms cancel. Each step of the hierarchy is an instance of this principle, applied to cancellation across horizons or across predicted values. Second, a single relabeling map moves the predicted quantile at a given level in the same direction for every horizon and every forecast. It can therefore remove errors that all horizons and forecasts share, but not errors that differ between them.

Notation. For a level $t \in [ 0 , 1 ]$ , let $\hat { q } _ { h , t } : = \hat { F } _ { Y _ { h } | \mathbf { X } } ^ { - 1 } ( t )$ denote the predicted t-quantile at horizon $h ,$ with the conventions $\hat { q } _ { h , 0 } = - \infty$ and $\hat { q } _ { h , 1 } = + \infty$ . The two central quantities are the coverage at horizon h and the pooled coverage,

$$
p _ { h } ( t ) : = P \big ( Y _ { h } \le \hat { q } _ { h , t } \big ) , \qquad \bar { p } ( t ) : = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } p _ { h } ( t ) .
$$

Since the predicted quantiles are non-decreasing in $t ,$ so are $p _ { h }$ and ${ \bar { p } } ,$ with value 0 at $t = 0$ and 1 at $t = 1 . \ \mathrm { A t }$ a grid level $\tau _ { q } ,$ we write

$$
\begin{array} { r } { r _ { h , q } : = p _ { h } ( \tau _ { q } ) - \tau _ { q } , \quad \quad C _ { h , q } : = P \big ( Y _ { h } \leq \hat { q } _ { h , \tau _ { q } } \mid \hat { q } _ { h , \tau _ { q } } \big ) } \end{array}
$$

for the signed coverage error of horizon $h _ { ; }$ , which is positive for over-coverage and negative for under-coverage, and for the conditional coverage given the predicted value. In this notation, the calibration errors of Section 4 read

$$
\mathrm { P C E } _ { \mathrm { p o o l } } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \biggl | \frac { 1 } { H } \sum _ { h = 1 } ^ { H } r _ { h , q } \biggr | , \qquad \mathrm { P C E } _ { \mathrm { a v g } } = \frac { 1 } { Q H } \sum _ { q = 1 } ^ { Q } \sum _ { h = 1 } ^ { H } \bigl | r _ { h , q } \bigr | ,
$$

$$
\mathrm { Q C E } _ { \mathrm { a v g } } = \frac { 1 } { Q H } \sum _ { q = 1 } ^ { Q } \sum _ { h = 1 } ^ { H } \mathbb { E } | C _ { h , q } - \tau _ { q } | .
$$

Recalibration with a map $g$ replaces $\hat { q } _ { h , \tau } \mathrm { \bf ~ b y } \hat { q } _ { h , g ( \tau ) }$ and keeps the nominal level $\tau$ . The recalibrated errors are therefore obtained by evaluating the coverages at $g ( \tau _ { q } )$ instead of $\tau _ { q } .$

$$
\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \lvert \bar { p } ( g ( \tau _ { q } ) ) - \tau _ { q } \rvert , \qquad \mathrm { P C E } _ { \mathrm { a v g } } ^ { g } = \frac { 1 } { Q H } \sum _ { q = 1 } ^ { Q } \sum _ { h = 1 } ^ { H } \lvert p _ { h } ( g ( \tau _ { q } ) ) - \tau _ { q } \rvert ,
$$

and $\mathrm { Q C E } _ { \mathrm { a v g } } ^ { g }$ is defined analogously with the conditional coverage $C _ { h , q } ^ { g } : = P ( Y _ { h } \leq \hat { q } _ { h , g ( \tau _ { q } ) } |$ $\hat { q } _ { h , g ( \tau _ { q } ) } )$

## A.1 Proof of the calibration error hierarchy

Intuition. Each step of the chain removes one opportunity for errors to cancel. $\mathrm { P C E _ { p o o l } }$ averages the signed errors over horizons before taking absolute values, so over-coverage at one horizon can offset under-coverage at another. $\mathrm { P C E _ { a v g } }$ takes absolute values per horizon and rules out this cancellation, but still averages over all forecasts of a horizon. $\mathrm { Q C E _ { a v g } }$ additionally conditions on the predicted value and thereby also rules out cancellation between forecasts with different predicted values. $\mathrm { Q C E } _ { \operatorname* { m a x } }$ finally replaces the average over horizons by the worst horizon.

Proof of Proposition 4.2. We verify the three inequalities in turn.

Pooling over horizons. By the triangle inequality, at every level

$$
\Big | \frac { 1 } { H } \sum _ { h = 1 } ^ { H } r _ { h , q } \Big | \leq \frac { 1 } { H } \sum _ { h = 1 } ^ { H } | r _ { h , q } | ,
$$

and averaging over q gives $\mathrm { P C E } _ { \mathrm { p o o l } } \leq \mathrm { P C E } _ { \mathrm { a v g } }$

Conditioning on the predicted value. By the tower property, the conditional coverage averages to the marginal coverage, $\mathbb { E } [ C _ { h , q } ] = p _ { h } ( \tau _ { q } )$ . Jensen’s inequality for the convex function $x \mapsto | x - \tau _ { q } |$ therefore gives

$$
| r _ { h , q } | = \bigl | \mathbb { E } [ C _ { h , q } ] - \tau _ { q } \bigr | \le \mathbb { E } | C _ { h , q } - \tau _ { q } | ,
$$

and averaging over q and $h$ gives $\mathrm { P C E _ { a v g } \leq Q C E _ { a v g } }$

A mean over horizons never exceeds their maximum, so $\mathrm { Q C E } _ { \mathrm { a v g } } \leq \mathrm { Q C E } _ { \operatorname* { m a x } } .$

The same chain holds for the estimators. The estimators of Appendix B.3 satisfy the hierarchy exactly on every sample, so both estimated gaps are non-negative. The first and the last step carry over verbatim with the empirical coverages $\hat { p } _ { h }$ in place of $p _ { h }$ . For the middle step, the isotonic fit assigns each block of consecutive forecasts the mean of their coverage indicators, so the fitted values average to the empirical coverage $\hat { p } _ { h } ( \tau _ { q } )$ . This is the empirical counterpart of the tower property, and Jensen’s inequality applies to the empirical distribution.

## A.2 Proof of the limits of quantile relabeling

We prove the three parts of Theorem 4.4 separately and precede each proof by its main idea.

Part (i): pooled calibration can always be restored. To make the pooled coverage at nominal level τ equal to $\tau ,$ it suffices to find a model level t at which the pooled coverage equals $\tau$ and to relabel τ to t. Since $\bar { p }$ goes from 0 to 1, such a level exists whenever $\bar { p }$ has no jumps, and the generalized inverse selects the smallest one. Isotonic recalibration implements this idea by estimating $\bar { p }$ on calibration data.

Proof of Theorem $4 . 4 ( i )$ . We start by defining $g ^ { \star }$ as the generalized inverse of ${ \bar { p } } .$ For $\tau \in ( 0 , 1 )$ let

$$
\begin{array} { r } { g ^ { \star } ( \tau ) : = \bar { p } ^ { - 1 } ( \tau ) = \operatorname* { i n f } \{ t \in [ 0 , 1 ] : \bar { p } ( t ) \geq \tau \} , } \end{array}
$$

and set $g ^ { \star } ( 0 ) = 0$ and $g ^ { \star } ( 1 ) = 1$ . Fix $\tau \in ( 0 , 1 )$ and write $t ^ { \star } : = g ^ { \star } ( \tau )$ . The set $\{ t : \bar { p } ( t ) \geq \tau \}$ contains 1 and is closed because p¯ is continuous, so $\bar { p } ( t ^ { \star } ) \geq \tau$ . Since $\bar { p } ( 0 ) = 0 < \tau$ , we have $t ^ { \star } > 0$ and $\bar { p } ( t ) < \tau$ for all $t < t ^ { \star }$ , and letting $t \uparrow t ^ { \star }$ gives $\bar { p } ( t ^ { \star } ) \leq \tau$ by continuity. Hence

$$
\bar { p } \bigl ( g ^ { \star } ( \tau _ { q } ) \bigr ) = \tau _ { q } \qquad \mathrm { f o r e v e r y ~ g r i d ~ l e v e l } \tau _ { q } ,
$$

and $\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g ^ { \star } } = 0$ . Moreover, $g ^ { \star }$ is non-decreasing because the set shrinks as $\tau$ grows, so the recalibrated quantiles $\hat { q } _ { h , g ^ { \star } ( \tau ) }$ remain ordered in $\tau .$ □

Continuity of $\bar { p }$ is needed only for this part, to solve $\bar { p } ( t ) = \tau _ { q }$ exactly. If $\bar { p }$ jumps over a target level, for example for predictive distributions with point masses or tied quantiles, no deterministic relabeling attains exact pooled coverage at that level. For quantile-only models, the statement refers to the forecast whose quantile function is interpolated between the grid levels. Alternatively, one can keep the predicted values and only relabel their nominal levels as $\bar { p } ( \tau _ { q } )$ , the form used in the experiments (see below). The pooled error is then zero without any continuity assumption, but the levels are no longer the requested ones. Parts (ii) and (iii) hold without this assumption.

Part (ii): cancellation across horizons cannot be undone. At a given level, a global map selects one model level t for all horizons. Raising t raises the coverage of every horizon, and lowering it lowers all of them. If some horizons over-cover while others under-cover, every such shift brings one group closer to the nominal level only at the expense of the other, or leaves it unchanged. The horizon gap measures exactly how much over- and under-coverage coexist at the same level, which yields the floor.

Proof of Theorem $4 . 4 ( i i )$ . For each level, collect the total over- and under-coverage across horizons,

$$
S _ { q } ^ { + } : = \sum _ { h = 1 } ^ { H } \operatorname* { m a x } ( r _ { h , q } , 0 ) , \qquad S _ { q } ^ { - } : = \sum _ { h = 1 } ^ { H } \operatorname* { m a x } ( - r _ { h , q } , 0 ) .
$$

Step 1: the horizon gap measures cancellation. Since $\begin{array} { r } { \sum _ { h } | r _ { h , q } | = S _ { q } ^ { + } + S _ { q } ^ { - } , | \sum _ { h } r _ { h , q } | = | S _ { q } ^ { + } - S _ { q } ^ { - } | , } \end{array}$ and $a + b - | a - b | = 2 \operatorname* { m i n } ( a , b )$

$$
\Delta _ { H } = \mathrm { P C E } _ { \mathrm { a v g } } - \mathrm { P C E } _ { \mathrm { p o o l } } = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \frac { 1 } { H } \Big ( S _ { q } ^ { + } + S _ { q } ^ { - } - | S _ { q } ^ { + } - S _ { q } ^ { - } | \Big ) = \frac { 2 } { Q H } \sum _ { q = 1 } ^ { Q } \operatorname* { m i n } ( S _ { q } ^ { + } , S _ { q } ^ { - } ) .\tag{∗}
$$

Thus $\Delta _ { H }$ is twice the amount of over-coverage that is offset by under-coverage at other horizons, averaged over levels.

Step 2: a global map cannot correct both sides. Fix any map g and a level $q ,$ and let $t : = g ( \tau _ { q } )$ . Since every $p _ { h }$ is non-decreasing,

$$
t \geq \tau _ { q } \implies p _ { h } ( t ) - \tau _ { q } \geq r _ { h , q } \mathrm { f o r } \mathrm { a l l } h , \qquad t \leq \tau _ { q } \implies p _ { h } ( t ) - \tau _ { q } \leq r _ { h , q }
$$

In the first case, every over-covering horizon still over-covers by at least $r _ { h , q } ,$ so $\begin{array} { r } { \sum _ { h } | p _ { h } ( t ) - \tau _ { q } | \ge } \end{array}$ $S _ { q } ^ { + }$ . In the second case, every under-covering horizon still under-covers by at least $| r _ { h , q } |$ , so $\begin{array} { r } { \sum _ { h } ^ { } | p _ { h } ( t ) - \tau _ { q } | \geq S _ { q } ^ { - } } \end{array}$ . In both cases,

$$
\sum _ { h = 1 } ^ { H } \bigl | p _ { h } \bigl ( g ( \tau _ { q } ) \bigr ) - \tau _ { q } \bigr | \geq \operatorname* { m i n } ( S _ { q } ^ { + } , S _ { q } ^ { - } ) .
$$

Step 3: averaging. Dividing by $Q H ,$ , summing over $q ,$ and using (∗) gives $\mathrm { P C E } _ { \mathrm { a v g } } ^ { g } \geq \Delta _ { H } / 2$

Sharpness. Let $H = 2 , Q = 1 , \tau _ { 1 } = \frac { 1 } { 2 }$ , and $a \in ( 0 , \textstyle { \frac { 1 } { 4 } } )$ . Take uniform predictive distributions, $\hat { q } _ { h , t } = t$ , and outcomes $Y _ { h }$ on $[ 0 , 1 ]$ with continuous CDFs $c _ { h } ,$ , so that $p _ { h } = c _ { h }$ . Let $c _ { 1 }$ interpolate $( 0 , 0 ) , ( \textstyle { \frac { 1 } { 2 } } , \textstyle { \frac { 1 } { 2 } } + a ) , ( \textstyle { \frac { 1 } { 2 } } + a , \textstyle { \frac { 1 } { 2 } } + a ) , ( 1 , 1 )$ and $c _ { 2 }$ interpolate $( 0 , 0 ) , ( \textstyle { \frac { 1 } { 2 } } , \frac { 1 } { 2 } - a ) , ( \textstyle { \frac { 1 } { 2 } } + a , \frac { 1 } { 2 } ) , ( 1 , 1 )$ piecewise linearly. Then $r _ { 1 , 1 } = a , r _ { 2 , 1 } = - a$ , and $\Delta _ { H } = a \mathrm { b y } ( * )$ . Relabeling to $\begin{array} { r } { g ( \frac { 1 } { 2 } ) = \frac { 1 } { 2 } + } \end{array}$ a calibrates the second horizon exactly, while the coverage of the first horizon is flat on $[ \mathinner { \frac { 1 } { 2 } } , \mathinner { \frac { 1 } { 2 } } + a ]$ and its error stays at a. Hence $\mathrm { P C E } _ { \mathrm { a v g } } ^ { g } = \frac { 1 } { 2 } ( a + 0 ) = \Delta _ { H } / 2$ , and the bound is attained. □

Forward relabeling, as used in the experiments. In the experiments, recalibrated forecasts are represented in an alternative form: the predicted value $\hat { q } _ { h , \tau _ { q } }$ is kept and receives a new nominal level $\ell _ { q }$ shared by all horizons, with error $\begin{array} { r } { \frac { 1 } { Q H } \sum _ { q , h } | p _ { h } ( \tau _ { q } ) - \tilde { \ell _ { q } } | } \end{array}$ . This avoids inverting a sparse quantile grid, but its averages over a finite grid need not coincide with those of the transformed quantiles $\hat { q } _ { h , g ( \tau _ { q } ) }$ . The bound of part (ii) holds for this form as well, by the same argument with the roles of the two cases exchanged. If $\ell _ { q } \geq \tau _ { q }$ , every under-covering horizon contributes at least $| r _ { h , q } |$ , and otherwise every over-covering horizon contributes at least $r _ { h , q } .$ Because the argument applies to any distribution, including the empirical distribution of a test set, the bound also holds exactly on every sample.

Part (iii): restoring pooled coverage can expose conditional errors. Quantile-conditioned calibration conditions on the predicted quantile. If all forecasts predict the same quantile, conditioning on it has no effect and QCE reduces to the marginal error. The construction exploits this. Two regimes predict the same median, so that their very different calibration is invisible before recalibration. Every map that restores pooled coverage must move to a level at which the two regimes predict different quantiles. The relabeled quantile then reveals the regime, and QCE exposes that the model overcovers in one regime and under-covers in the other. Recalibration thus removes the pooled error while making the conditional error almost as large as possible.

ProofofTheorem 4.4(iii). Fix $\varepsilon \in ( 0 , { \frac { 1 } { 2 } } )$ and $\eta \in ( 0 , \frac { 1 } { 2 } - \varepsilon )$ . Let $\begin{array} { r } { H = Q = 1 , \tau _ { 1 } = \frac { 1 } { 2 } } \end{array}$ , and let a latent regime X be uniform on $\{ A , B \}$ . Table 4 specifies the predictive distributions, their quantile functions $q _ { x } ,$ , the outcome distributions, and the resulting regime-wise coverages $c _ { x } ( t ) : = \mathbf { \dot { P } } ( Y \leq$ $q _ { x } ( t ) \mid X \stackrel { \sim } { = } x )$ . Both coverages are continuous, so $\begin{array} { r } { \bar { p } = \frac { \tilde { \Gamma } } { 2 } ( c _ { A } + c _ { B } ) } \end{array}$ is continuous.

Step 1: before recalibration, the regimes are indistinguishable. Both regimes predict the median $\begin{array} { r } { q _ { A } \dot { ( \frac { 1 } { 2 } ) } = \dot { q } _ { B } ( \frac { 1 } { 2 } ) = 0 } \end{array}$ . The predicted median is therefore constant, its conditional coverage equals the marginal coverage $\begin{array} { r } { \frac { 1 } { 2 } \big ( c _ { A } \big ( \frac { 1 } { 2 } \big ) + c _ { B } \big ( \frac { 1 } { 2 } \big ) \big ) = \frac { 1 } { 2 } \big ( 1 + 2 \eta \big ) } \end{array}$ , and

$$
\mathrm { P C E } _ { \mathrm { p o o l } } = \mathrm { Q C E } _ { \mathrm { a v g } } = \eta .
$$

<table><tr><td></td><td>Regime Predictive distribution Quantile function</td><td> $q _ { x } ( t )$ </td><td>Outcome Y |  $X = x$ </td><td>Coverage  $c _ { x } ( t )$ </td></tr><tr><td>A</td><td> $\scriptstyle { \frac { 1 } { 2 } } \mathrm { U } [ - 2 , 0 ] + { \frac { 1 } { 2 } } \mathrm { U } [ 0 , 1 ]$ </td><td> $\left\{ \begin{array} { l l } { 4 t - 2 , } & { t \leq \frac { 1 } { 2 } } \\ { 2 t - 1 , } & { t > \frac { 1 } { 2 } } \end{array} \right.$ </td><td> $\mathrm { U } [ - 2 , - 1 ]$ </td><td>min(4t, 1)</td></tr><tr><td>B</td><td> $\textstyle \mathrm { \frac { 1 } { 2 } U [ - 1 , 0 ] + \frac { 1 } { 2 } U [ 1 , 2 ] }$ </td><td> $\left\{ { \begin{array} { l l } { 2 t - 1 , } & { t \leq { \frac { 1 } { 2 } } } \\ { 2 t , } & { t > { \frac { 1 } { 2 } } } \end{array} } \right.$ </td><td> $2 \eta \mathrm { U } [ - \textstyle { \frac { 1 } { 4 } } , 0 ]$ </td><td> $\textstyle t \leq { \frac { 3 } { 8 } }$   $2 \eta ( 8 t - 3 ) ,$   $\begin{array} { r } { \frac { 3 } { 8 } < t \leq \frac { 1 } { 2 } } \end{array}$ </td></tr><tr><td></td><td></td><td></td><td> $+ \left( 1 - 2 \eta \right) \mathrm { U } [ 1 , 2 ]$ </td><td> $\lfloor 2 \eta + ( 1 - 2 \eta ) ( 2 t - 1 )$  ,t &gt; 112</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4: Construction for Theorem 4.4(iii). Both regimes predict the median $\begin{array} { r } { q _ { A } \big ( \frac { 1 } { 2 } \big ) = q _ { B } \big ( \frac { 1 } { 2 } \big ) = 0 . } \end{array}$ but the model over-covers in regime A and under-covers in regime $B$

Step 2: restoring pooled coverage forces a level that separates the regimes. The pooled coverage satisfies $\begin{array} { r } { \bar { p } ( t ) = \bar { 2 } \bar { t } < \frac { 1 } { 2 } } \end{array}$ on $[ 0 , \textstyle { \frac { 1 } { 4 } } ) , \bar { p } = \frac { 1 } { 2 }$ on $[ \frac { 1 } { 4 } , \frac { 3 } { 8 } ]$ , and $\bar { p } > \frac { 1 } { 2 }$ beyond. Hence

$$
\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g } = \big | \bar { p } ( g ( \frac { 1 } { 2 } ) ) - \frac { 1 } { 2 } \big | = 0 \quad \Longleftrightarrow \quad t : = g ( \frac { 1 } { 2 } ) \in [ \frac { 1 } { 4 } , \frac { 3 } { 8 } ] ,
$$

and the map of part (i) gives $\begin{array} { r } { t = \frac { 1 } { 4 } } \end{array}$ . For every such t, the relabeled medians $q _ { A } ( t ) = 4 t - 2$ and $q _ { B } ( t ) = 2 t - 1$ differ because $\textstyle t \neq { \frac { 1 } { 2 } }$

Step 3: the relabeled median reveals the regime. Conditional on the relabeled median, the regime is known, and the conditional coverages are $c _ { A } ( t ) = 1$ under A and $c _ { B } ( t ) = 0$ under B. Thus

$$
\begin{array} { r } { \mathrm { Q C E } _ { \mathrm { a v g } } ^ { g } = \frac { 1 } { 2 } { \left| 1 - \frac { 1 } { 2 } \right| } + \frac { 1 } { 2 } { \left| 0 - \frac { 1 } { 2 } \right| } = \frac { 1 } { 2 } } \end{array}
$$

for every g with $\mathrm { P C E } _ { \mathrm { p o o l } } ^ { g } = 0$ , and the increase is $\frac { 1 } { 2 } - \eta > \varepsilon .$

## A.3 Population target of the isotonic estimator

The isotonic estimator of Appendix B.3 does not target the conditional coverage curve itself, but its best non-decreasing approximation. This approximation pools adjacent predicted values into blocks and averages the conditional coverage within each block. Averaging can only pull values towards their mean, so the resulting error lies between the marginal and the full conditional error. If the conditional coverage is already non-decreasing, no pooling is needed and the target equals $\mathrm { Q C E } _ { h }$

Lemma A.1. Fix h and q, write $Z : = \hat { q } _ { h , \tau _ { q } }$ and $c ( z ) : = P ( Y _ { h } \leq Z \mid Z = z )$ , and let $c ^ { \uparrow }$ be the $L ^ { 2 } ( P _ { Z } )$ -projection of c onto non-decreasing functions, i.e., the population counterpart of the isotonic fit in Appendix B.3. Then

$$
\begin{array} { r } { | p _ { h } ( \tau _ { q } ) - \tau _ { q } | \le \mathbb { E } | c ^ { \uparrow } ( Z ) - \tau _ { q } | \le \mathbb { E } | c ( Z ) - \tau _ { q } | = \mathrm { Q C E } _ { h } ( \tau _ { q } ) , } \end{array}
$$

with equality on the right if c is non-decreasing.

Proof. The isotonic projection is the conditional expectation of c given its own level sets [8, 38],

$$
\mathbb { E } { \bigl [ } c ( Z ) \mid c ^ { \uparrow } ( Z ) { \bigr ] } = c ^ { \uparrow } ( Z ) \quad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Taking expectations gives $\mathbb { E } [ c ^ { \uparrow } ( Z ) ] = \mathbb { E } [ c ( Z ) ] = p _ { h } ( \tau _ { q } )$ , and the left inequality is Jensen’s inequality. The right inequality is the conditional Jensen inequality

$$
\begin{array} { r } { | c ^ { \uparrow } ( Z ) - \tau _ { q } | = \left| \mathbb { E } [ c ( Z ) - \tau _ { q } \mid c ^ { \uparrow } ( Z ) ] \right| \leq \mathbb { E } \left[ | c ( Z ) - \tau _ { q } | \ \middle | \ c ^ { \uparrow } ( Z ) \right] , } \end{array}
$$

followed by taking expectations. If c is non-decreasing, then $c ^ { \uparrow } = c$ and equality holds. □

The lower bound can be strict. Let Z take the values −1 and 0 with probability $\frac { 1 } { 2 }$ each, with conditional coverages $c ( - 1 ) = 1$ and $c ( 0 ) = 0$ at level $\begin{array} { r } { \tau _ { q } = \frac { 1 } { 2 } } \end{array}$ . Then $\begin{array} { r } { \mathrm { Q C E } _ { h } ( \tau _ { q } ) = \frac { 1 } { 2 } } \end{array}$ , whereas the projection pools both values into the constant ${ \frac { 1 } { 2 } } ,$ , so the isotonic target is zero, like the marginal error. The diagnostic therefore detects value dependence in which coverage increases with the predicted value. This is the pattern of forecasts that are too extreme in both directions, which dominates our case studies (Appendix C.2), while decreasing or non-monotone patterns may be missed.

## B Experimental Details

## B.1 Models and compute

All model inference was performed on a single NVIDIA RTX A5000 GPU running Linux. Models were loaded via HuggingFace transformers or the model-specific package and evaluated sequentially per dataset with batch size 32. We evaluate 14 foundation models spanning the four uncertainty quantification categories; Table 5 lists the model identifiers. Code implementing all metrics, significance tests, and recalibration methods is released at https://github.com/thomdeck/tsfm-calibration.

Table 5: Models evaluated, grouped by uncertainty quantification category.
<table><tr><td>UQ Category</td><td>Model</td><td>HuggingFace ID / Package</td></tr><tr><td rowspan="5">Cat. 1: Parametric</td><td>Moirai-Small</td><td>Salesforce/moirai-1.1-R-small</td></tr><tr><td>Moirai-Base</td><td>Salesforce/moirai-1.1-R-base</td></tr><tr><td>Moirai-Large</td><td>Salesforce/moirai-1.1-R-large</td></tr><tr><td>Moirai-MoE-Small</td><td>Salesforce/moirai-moe-1.O-R-small</td></tr><tr><td>Moirai-MoE-Base</td><td>Salesforce/moirai-moe-1.0-R-base</td></tr><tr><td rowspan="3">Cat. 2: Discretized</td><td>Chronos-T5-Small</td><td>amazon/chronos-t5-small</td></tr><tr><td>Chronos-T5-Base</td><td>amazon/chronos-t5-base</td></tr><tr><td>TabPFN-TS</td><td>PriorLabs/tabpfn-time-series</td></tr><tr><td rowspan="5">Cat. 3: Direct Quantile</td><td>Chronos-2</td><td>amazon/chronos-2</td></tr><tr><td>Chronos-Bolt-Small</td><td>amazon/chronos-bolt-small</td></tr><tr><td>Chronos-Bolt-Base</td><td>amazon/chronos-bolt-base</td></tr><tr><td>TiRex</td><td>NX-AI/TiRex</td></tr><tr><td>TimesFM-2.5-200M</td><td>google/timesfm-2.5-200m-pytorch</td></tr><tr><td>Cat.4: Generative</td><td>Sundial-Base</td><td>thuml/sundial-base-128m</td></tr></table>

Quantile levels. All probabilistic forecasts are evaluated at the nine levels $\tau \in \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ the intersection of the levels natively supported by all models.

Sampling-based models. For Chronos-T5 (autoregressive categorical sampling over 4096 bins), Moirai 1.1 and Moirai-MoE (a mixture of Student-t, Normal, Negative Binomial, and Log-Normal distributions), and Sundial (flow matching, 128M parameters), we draw S = 100 sample paths per forecast and read off empirical quantiles.

Quantile-output models. TabPFN-TS applies the TabPFN-v2 tabular foundation model [23] to time series by turning forecasting into a tabular regression task. It represents its predictive distribution as a bar distribution over a fixed grid and returns the nine quantiles in a single forward pass. Chronos-2, Chronos-Bolt, TiRex, and TimesFM output the nine quantile levels directly.

## B.2 Data

We use the GIFT-Eval benchmark [2] (Salesforce/gift-eval on HuggingFace), which covers energy, traffic, weather, finance, health, sales, and cloud infrastructure data. The calibration analysis uses the short-term task of all 55 dataset configurations (770 model–dataset pairs). The recalibration experiments require a calibration and a test split, and we restrict them to the 42 configurations with at least 50 forecast windows. Medium- and long-term tasks are used only for the horizon-length analysis in Appendix C.5. All models are evaluated on the same forecast windows of each dataset, so that model comparisons are paired and based on equal sample sizes. Missing target values (present in ten configurations) carry no calibration information and are excluded from all computations; they are never counted as non-covered.

## B.3 Estimating the calibration errors

For forecast window i, horizon $h ,$ and level τ, let $z _ { i , h } ( \tau ) = \mathbf { 1 } \{ y _ { i , h } \ \leq \ \hat { q } _ { i , h , \tau } \}$ be the coverage indicator, and let $\hat { p } _ { h } ( \tau )$ be its mean over the $N _ { h }$ windows with an observed target at horizon h.

Probabilistic calibration errors. The estimators replace $p _ { h }$ by $\hat { p } _ { h }$ in the definitions of Section 4: $\begin{array} { r } { \widehat { \mathrm { P C E } } _ { \mathrm { p o o l } } = \frac { 1 } { Q } \sum _ { q } | \frac { 1 } { H } \sum _ { h } \widehat { p } _ { h } ( \tau _ { q } ) - \tau _ { q } | , \widehat { \mathrm { P C E } } _ { h } = \frac { 1 } { Q } \sum _ { q } | \widehat { p } _ { h } ( \tau _ { q } ) - \tau _ { q } | , \widehat { \mathrm { P C E } } _ { \mathrm { a v g } } = \frac { 1 } { H } \sum _ { h } \widehat { \mathrm { P C E } } _ { h } , } \end{array}$ and ${ \widehat { \mathrm { P C E } } } _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { h } { \widehat { \mathrm { P C E } } } _ { h } , { \mathrm { i . e } }$ ., the worst horizon after averaging over levels.

Quantile-conditioned calibration errors. For each horizon h and level $\tau _ { q } ,$ we estimate the conditional coverage curve $c _ { h , q } ( x ) \ = \ P ( Y _ { h } \ \leq \ \hat { q } _ { h , \tau _ { q } } \ | \ \hat { \ q } _ { h , \tau _ { q } } \ = \ x )$ by least-squares isotonic (nondecreasing) regression of the indicators $z _ { i , h } ( \tau _ { q } )$ on the predicted values $\hat { q } _ { i , h , \tau _ { q } }$ , computed with the pool-adjacent-violators algorithm. Windows with tied predicted values share one fitted value. With the fitted curve $\hat { c } _ { h , q } :$

$$
\widehat { \mathrm { Q C E } } _ { h } ( \tau _ { q } ) = \frac { 1 } { N _ { h } } \sum _ { i } \big | \hat { c } _ { h , q } ( \hat { q } _ { i , h , \tau _ { q } } ) - \tau _ { q } \big | ,
$$

which is averaged over levels and then over horizons $\widehat { ( \mathrm { Q C E } } _ { \mathrm { a v g } } )$ or maximized over horizons $( \widehat { \mathrm { Q C E } } _ { \mathrm { m a x } } )$ . The gaps are $\widehat { \Delta } _ { H } = \widehat { \mathrm { P C E } } _ { \mathrm { a v g } } - \widehat { \mathrm { P C E } } _ { \mathrm { p o o l } }$ and $\widehat { \Delta } _ { C } = \widehat { \mathrm { Q C E } } _ { \mathrm { a v g } } - \widehat { \mathrm { P C E } } _ { \mathrm { a v g } }$ . Each curve is fitted within one dataset and model.

## Why this estimator.

• No tuning. Isotonic regression chooses the grouping of forecasts automatically and requires no bins or bandwidth. This is the principle behind CORP reliability diagrams [15, 18], which use it for the same reason. The same principle underlies the T-PAV decomposition of the quantile score [18], which measures miscalibration in units of the outcome through the pinball loss; QCE measures it in coverage probability and therefore remains comparable across datasets and series of different scale.

• Same units as PCE, exact hierarchy. $\widehat { \mathrm { Q C E } }$ is measured in coverage probability, and the empirical hierarchy $\widehat { \mathrm { P C E } } _ { \mathrm { p o o l } } \leq \widehat { \mathrm { P C E } } _ { \mathrm { a v g } } \leq \widehat { \mathrm { Q C E } } _ { \mathrm { a v g } } \leq \widehat { \mathrm { Q C E } } _ { \mathrm { m a x } }$ holds on every sample (Appendix A.1), so both estimated gaps are non-negative.

• Comparable across datasets. The estimate depends on the data only through the coverage indicators and the ordering of the predicted values. It is therefore unchanged by any strictly increasing transformation applied jointly to outcomes and forecasts (change of units, rescaling, log transform), and every forecast contributes equally regardless of the magnitude of its series. This matters for a benchmark of zero-shot models applied to arbitrary domains.

• Conservative population target. The estimator targets the isotonic projection of $c _ { h , q } . \ \mathrm { ~ I f ~ } c _ { h , q }$ is non-decreasing, this is $\mathrm { Q C E } _ { h } ( \tau _ { q } )$ itself; otherwise it lies between $\mathrm { \tilde { P C E } } _ { h } ( \tau _ { q } )$ and $\mathrm { { \dot { Q } C E } } _ { h } ( \tau _ { q } )$ (Lemma A.1), so the monotonicity constraint never overstates value dependence. Conversely, it can understate decreasing or non-monotone value dependence, which the estimate may therefore miss.

• Finite-sample bias. Fitting and evaluating the curve on the same windows makes $\widehat { \mathrm { Q C E } }$ positive even without value dependence. We quantify this component with the permutation test of Appendix B.4. Neither a small gap nor a non-significant test certifies conditional calibration.

Outcomes with point masses. For continuous predictive CDFs, Definition 4.1 coincides with CEP calibration [32, 18]. Both this definition and probabilistic calibration presuppose that the outcome has no point masses, i.e., no single value that occurs with positive probability. Some datasets violate this, for example through zero demand in intermittent sales or zero solar power at night. If a predicted quantile equals such a value a, every outcome $Y = a$ counts as covered, so coverage exceeds the nominal level even when the forecast equals the true conditional distribution. At night, for instance, the true distribution of solar power is a point mass at zero. All its quantiles are zero and are covered with probability one, so a perfect forecast has coverage error $1 - \tau$ at every level $\tau .$ Average coverage dilutes this effect with the remaining forecasts, whereas quantile conditioning isolates exactly the forecasts whose quantile sits at the point mass, so such datasets produce the largest conditioning gaps. We therefore flag them and check that they do not drive our conclusions (Appendix C.5).

## B.4 Testing the diagnostic gaps

Why a test is needed. Both gaps are estimated and evaluated on the same data, and their estimates are positive even when coverage depends neither on the predicted value nor on the horizon as the isotonic fit always finds some spurious structure, and per-horizon coverages always fluctuate. To decide whether an observed gap reflects real structure, we compare it with the distribution of the same statistic on data from which this structure has been removed while everything else, including the dependence between windows, horizons, and levels, is kept.

Conditioning gap. The null hypothesis is that the coverage indicators of a series are unrelated to its predicted values. We generate null data by randomly reassigning the complete coverage-indicator trajectories (all windows, horizons, and levels of a series) among series with the same number of windows, while keeping all predicted quantiles fixed. This leaves every per-horizon coverage, and hence every PCE, unchanged, preserves the temporal and cross-horizon dependence of the indicators, and removes only the association between coverage and predicted value. Recomputing $\widehat { \Delta } _ { C }$ on $B = 9 9 9$ such data sets gives its null distribution. The test is exact when series are exchangeable under the null hypothesis.

Horizon gap. The null hypothesis is that coverage is the same at all horizons. We resample whole series with replacement (a series-cluster bootstrap, which preserves arbitrary dependence within a series) and shift each bootstrap coverage curve to $\begin{array} { r } { \hat { p _ { h } ^ { * } } ( \tau ) - \hat { p _ { h } } ( \tau ) + \frac { 1 } { H } \sum _ { h ^ { \prime } } \hat { p } _ { h ^ { \prime } } ( \tau ) } \end{array}$ , so that all horizons share the observed pooled coverage. Recomputing $\widehat { \Delta } _ { H }$ on $B =$ 999 bootstrap samples gives an approximate null distribution, which assumes that different series are independent.

What we report. For both gaps, the p-value is $( 1 + \#$ {null values ≥ observed}) $/ ( B + 1 )$ [35], and the corrected gap is the observed gap minus the mean of its null distribution, i.e., the part of the gap not explained by estimation noise. We report the share of model–dataset pairs with $p \leq 0 . 0 5 ;$ if no pair had any structure, about 5% would be significant by chance. A rejection indicates that coverage varies with the predicted value or the horizon in a way that produces cancellation, which is the structure the gaps are designed to reveal. Both tests require at least two series. For datasets consisting of a single series (saugeenday and us\_births, 84 model–dataset pairs), every bootstrap sample reproduces the original data and no nontrivial permutation exists, so we report only their descriptive metrics and exclude them from all significance shares. Over the remaining 686 pairs, 82% of the horizon gaps and 65% of the conditioning gaps are significant.

Size and power. We simulate 40 independent series with 10 windows each and $H = 2 4$ , with Gaussian forecast errors that are AR(1) across horizons with correlation $\rho$ and window-specific location and scale (300 replicates, $B = 1 9 9 )$ . In the calibrated scenario the predictive distribution equals the true one. In the horizon scenario the predictive spread is too wide at short and too narrow at long horizons, and in the value scenario it is too narrow for high and too wide for low predicted values. Table 6 shows that both tests hold their nominal level up to simulation error (the $\Delta _ { H }$ test is slightly liberal under very strong cross-horizon correlation), detect the failure mode they are designed for, and reject the other one only at about the nominal rate, so the two tests separate the two failure modes. The $\Delta _ { C }$ test loses power under very strong cross-horizon correlation $( \rho = 0 . 9 )$ , where the 24 horizons carry little independent information.

<table><tr><td></td><td colspan="3"> $\Delta _ { C }$  permutation test</td><td colspan="3"> $\Delta _ { H }$  bootstrap test</td></tr><tr><td>Scenario</td><td> $\rho = 0$ </td><td>0.5</td><td>0.9</td><td> $\rho = 0$ </td><td>0.5</td><td>0.9</td></tr><tr><td>Calibrated (size)</td><td>0.05</td><td>0.04</td><td>0.04</td><td>0.06</td><td>0.04</td><td>0.08</td></tr><tr><td>Value-dependent spread</td><td>1.00</td><td>0.99</td><td>0.48</td><td>0.03</td><td>0.06</td><td>0.06</td></tr><tr><td>Horizon-dependent spread</td><td>0.03</td><td>0.06</td><td>0.05</td><td>1.00</td><td>1.00</td><td>1.00</td></tr></table>

Table 6: Rejection rates at level 0.05 in the simulation study. The calibrated scenario gives the size of each test; the other rows give the power against the failure mode the test targets and the rejection rate under the other one.

Gaps by dataset size. Table 7 stratifies the model–dataset pairs with at least two series by the number of forecast windows. The estimation-noise component of $\widehat { \Delta } _ { C }$ is large for small datasets (0.043 below 50 windows) and much smaller for larger ones, whereas the corrected gap does not shrink with dataset size and is largest for the largest datasets (0.024), as expected for real structure.

Raw gaps of small datasets should therefore not be interpreted without the test, and the tests have little power there.

<table><tr><td colspan="2"></td><td colspan="3"> $\widehat { \Delta } _ { C }$ </td><td colspan="2">% significant</td></tr><tr><td>Windows</td><td>Pairs</td><td>raw</td><td>noise</td><td>corrected</td><td> $\Delta _ { C }$ </td><td> $\Delta _ { H }$ </td></tr><tr><td>&lt; 50</td><td>98</td><td>0.063</td><td>0.043</td><td>0.020</td><td>37%</td><td>61%</td></tr><tr><td>50-199</td><td>98</td><td>0.023</td><td>0.016</td><td>0.007</td><td>47%</td><td>68%</td></tr><tr><td>200-499</td><td>196</td><td>0.024</td><td>0.010</td><td>0.014</td><td>76%</td><td>94%</td></tr><tr><td>500-999</td><td>182</td><td>0.023</td><td>0.007</td><td>0.015</td><td>68%</td><td>80%</td></tr><tr><td>≥ 1000</td><td>112</td><td>0.035</td><td>0.011</td><td>0.024</td><td>79%</td><td>95%</td></tr></table>

Table 7: Conditioning gap and share of significant gaps $\overline { { ( p \leq 0 . 0 5 ) } }$ by number of forecast windows, over the 686 model–dataset pairs with at least two series. “Noise” is the mean of the permutation null distribution.

## B.5 Recalibration protocol

Calibration and test splits. Each dataset is split into 20% calibration and 80% test windows, and no window is used in both.

• Chronological (main text). For series with $W \geq 2$ windows, the first ⌈0.2W⌉ windows in time are used for calibration and the remaining ones for testing, so that calibration always precedes evaluation. Series with a single window cannot be split in time; for these, a random 20% of the series is used for calibration (a cold-start deployment). Eleven of the 42 configurations consist only of single-window series (m4\_\*, hospital, covid\_deaths, car\_parts, restaurant, solar/W).

• Random (Appendix C.3). A random permutation of the windows, which ignores temporal order and is reported for comparison.

Methods. All four methods fit a single non-decreasing map from nominal to recalibrated levels after pooling the calibration windows and horizons. The predicted quantile values are left unchanged and receive the recalibrated levels, and all metrics after recalibration compare test coverage with these levels (Appendix A.2).

• Isotonic Regression (IR) [26]: a non-decreasing map from each nominal level to its empirical coverage on the calibration windows, interpolated on a 1000-point grid.

• Platt Scaling (PS) [36]: the map $\sigma ( a \log \mathrm { i t } ( \tau ) + b )$ , fitted by least squares to the empirical calibration coverages.

• Histogram Binning (HB) [43]: the empirical CDF of the PIT values evaluated at the edges of B = 20 equal-width bins and interpolated linearly in between.

• KDE Calibration (KDE) [13]: the smoothed PIT CDF $\begin{array} { r } { \hat { F } ( \alpha ) = \frac { 1 } { N } \sum _ { i } \sigma ( \tau _ { \mathrm { b w } } ( \alpha - u _ { i } ) ) } \end{array}$ with $\tau _ { \mathrm { b w } } = 1 0 0$

For HB and KDE, approximate PIT values are obtained by linear interpolation between the nine predicted quantiles of each forecast; values at or below the lowest quantile are set to $1 0 ^ { - 6 }$ and values above the highest to $1 - 1 0 ^ { - 6 }$ . A target equal to a predicted quantile counts as covered, so that $u \leq \tau _ { q }$ holds exactly when $y \le \hat { q } _ { \tau _ { q } }$ . Figure 1 and the case study use IR as a fixed method, chosen before seeing any test data. For the prediction intervals in Figure $2 ( \mathrm { a } ) .$ , IR is fitted on the extended grid of Appendix C.5, and the recalibrated 90% interval spans the predicted quantiles at the levels whose calibration coverage is 5% and 95% (0.021 and 0.975), interpolated linearly between grid levels; the example forecast is the test window with the largest coverage gain.

## C Additional Results

## C.1 Additional calibration errors

Table 8 extends Table 2 with standard errors across the 55 dataset configurations. Because all models are evaluated on the same datasets, comparisons between models should rely on paired differences across datasets rather than on these standard errors.

<table><tr><td></td><td></td><td colspan="3">PCE</td><td colspan="2">QCE</td><td colspan="2">Diag.</td></tr><tr><td>Cat.</td><td>Model</td><td>pool</td><td>avg</td><td>max</td><td>avg</td><td>max</td><td> $\pmb { \triangle } _ { \mathbf { H } }$ </td><td> $\Delta _ { \mathbf { C } }$ </td></tr><tr><td rowspan="2">1</td><td rowspan="2"></td><td>0.0536</td><td>0.0829</td><td>0.1579</td><td>0.1215</td><td>0.1862</td><td>0.0293</td><td>0.0386</td></tr><tr><td>±0.0073</td><td>±0.0085</td><td>±0.0142</td><td>±0.0100</td><td>±0.0144</td><td>±0.0046</td><td>±0.0040</td></tr><tr><td rowspan="2"></td><td rowspan="2">Moirai-Base</td><td>0.0484</td><td>0.0776</td><td>0.1581</td><td>0.1225</td><td>0.1906</td><td>0.0292</td><td>0.0448</td></tr><tr><td>±0.0065</td><td>±0.0080</td><td>±0.0149</td><td>±0.0099</td><td>±0.0151</td><td>±0.0040</td><td>±0.0055</td></tr><tr><td rowspan="2">1</td><td>Moirai-Large</td><td>0.0533</td><td>0.0826</td><td>0.1537</td><td>0.1204</td><td>0.1878</td><td>0.0293</td><td>0.0379</td></tr><tr><td>Moirai-MoE-Base</td><td>±0.0066</td><td>±0.0082</td><td>±0.0145</td><td>±0.0098</td><td>±0.0157</td><td>±0.0041</td><td>±0.0042</td></tr><tr><td rowspan="2">1 1</td><td rowspan="2">Moirai-MoE-Small</td><td>0.0521</td><td>0.0837</td><td>0.1578</td><td>0.1234</td><td>0.1908</td><td>0.0317</td><td>0.0397</td></tr><tr><td>±0.0067</td><td>±0.0082</td><td>±0.0136</td><td>±0.0097</td><td>±0.0148</td><td>±0.0044</td><td>±0.0047</td></tr><tr><td rowspan="2">1</td><td rowspan="2">Moirai-Small</td><td>0.0530</td><td>0.0855</td><td>0.1607</td><td>0.1283</td><td>0.1982</td><td>0.0326</td><td>0.0428</td></tr><tr><td>±0.0070</td><td>±0.0092</td><td>±0.0146</td><td>±0.0101</td><td>±0.0146</td><td>±0.0049</td><td>±0.0049</td></tr><tr><td rowspan="2">2</td><td rowspan="2">Chronos-T5-Base</td><td>0.1024</td><td>0.1240</td><td>0.1939</td><td>0.1491</td><td>0.2159</td><td>0.0216</td><td>0.0251</td></tr><tr><td>±0.0109</td><td>±0.0117</td><td>±0.0174</td><td>±0.0124</td><td>±0.0177</td><td>±0.0041</td><td>±0.0037</td></tr><tr><td rowspan="2">2</td><td rowspan="2">Chronos-T5-Small</td><td>0.1035</td><td>0.1253</td><td>0.1960</td><td>0.1524</td><td>0.2183</td><td>0.0218</td><td>0.0271</td></tr><tr><td>±0.0112</td><td>±0.0115</td><td>±0.0171</td><td>±0.0122</td><td>±0.0173</td><td>±0.0046</td><td>±0.0041</td></tr><tr><td rowspan="2">2</td><td rowspan="2">TabPFN-TS</td><td>0.0629</td><td>0.0872</td><td>0.1537</td><td>0.1198</td><td>0.1797</td><td>0.0243</td><td>0.0325</td></tr><tr><td>±0.0062</td><td>±0.0079</td><td>±0.0137</td><td>±0.0094</td><td>±0.0135</td><td>±0.0045</td><td>±0.0036</td></tr><tr><td rowspan="2">3</td><td rowspan="2">Chronos-2</td><td>0.0641</td><td>0.0919</td><td>0.1653</td><td>0.1229</td><td>0.1881</td><td>0.0278</td><td>0.0310</td></tr><tr><td>±0.0082</td><td>±0.0094</td><td>±0.0147</td><td>±0.0107</td><td>±0.0150</td><td>±0.0043</td><td>±0.0040</td></tr><tr><td rowspan="2">3</td><td rowspan="2">Chronos-Bolt-Base</td><td>0.0505</td><td>0.0804</td><td>0.1494</td><td>0.1202</td><td>0.1838</td><td>0.0299</td><td>0.0398</td></tr><tr><td>±0.0063</td><td>±0.0086</td><td>±0.0147</td><td>±0.0104</td><td>±0.0150</td><td>±0.0044</td><td>±0.0048</td></tr><tr><td rowspan="2">3</td><td rowspan="2">Chronos-Bolt-Small</td><td>0.0523</td><td>0.0833</td><td>0.1569</td><td>0.1225</td><td>0.1890</td><td>0.0310</td><td>0.0393</td></tr><tr><td>±0.0053</td><td>±0.0078</td><td>±0.0135</td><td>±0.0098</td><td>±0.0142</td><td>±0.0042</td><td>±0.0045</td></tr><tr><td rowspan="2">3</td><td rowspan="2">TiRex</td><td>0.0502</td><td>0.0796</td><td>0.1485</td><td>0.1161</td><td>0.1798</td><td>0.0294</td><td>0.0365</td></tr><tr><td>±0.0068</td><td>±0.0084</td><td>±0.0139</td><td>±0.0102</td><td>±0.0150</td><td>±0.0041</td><td>±0.0048</td></tr><tr><td rowspan="2">3</td><td rowspan="2">TimesFM-2.5-200M</td><td>0.0743</td><td>0.1011</td><td>0.1711</td><td>0.1254</td><td>0.1900</td><td>0.0268</td><td>0.0243</td></tr><tr><td>±0.0105</td><td>±0.0109</td><td>±0.0156</td><td>±0.0113</td><td>±0.0158</td><td>±0.0041</td><td>±0.0037</td></tr><tr><td rowspan="2">4</td><td rowspan="2"></td><td>0.1467</td><td>0.1678</td><td>0.2413</td><td>0.1958</td><td>0.2629</td><td>0.0211</td><td>0.0280</td></tr><tr><td>±0.0072</td><td>±0.0092</td><td>±0.0137</td><td>±0.0107</td><td>±0.0143</td><td>±0.0047</td><td>±0.0049</td></tr></table>

Table 8: Calibration metrics across probabilistic TSFMs with standard errors across datasets (mean on the first line, standard error on the second). Lower is better for all calibration errors and diagnostics.  
Figure 3 shows, for each model–dataset pair, how aggregation choices affect the apparent severity of miscalibration and how calibration relates to forecast accuracy.

## C.2 Further case studies

Value dependence across datasets. Table 9 lists further model–dataset pairs with small probabilistic calibration errors but significant conditioning gaps. For each, it reports how often outcomes fall below the predicted 10% quantile and above the predicted 90% quantile when the respective predicted quantile is in its lowest or highest within-horizon quintile (nominal: 10% everywhere). The pattern of the main-text case study recurs in every row: outcomes fall below the lower quantile far more often when the model predicts high values, and above the upper quantile more often when it predicts low values. The forecasts are therefore too extreme in both directions, and these errors cancel in marginal coverage. The last row is the intermittent-demand dataset car\_parts, where 79% of the targets are zero. Its very large conditioning gap is genuine in the sense that the model’s lower quantiles are negative for low-demand items and never covered, but much of it reflects the point mass at zero (Appendix B.3), and it should not be compared directly with the other rows.

Horizon cancellation. Figure 4 shows the complementary failure mode. Moirai-Small on hourly traffic speeds (LOOP\_SEATTLE/H) appears almost perfectly calibrated under pooled coverage $( \mathrm { P C E } _ { \mathrm { p o o l } } = 0 . 0 0 8 )$ . Its coverage, however, oscillates with the time of day being forecast, with errors of up to 38 percentage points that cancel across the 48 horizons $\mathrm { ( P \mathrm { \bar { C } E _ { a v g } } ) } ^ { - } = 0 . 0 9 8$ $\Delta _ { H } = 0 . 0 8 9 )$ . A practitioner who uses the intervals for a particular hour of the next day would face these errors in full. Global recalibration fitted on the earliest windows cannot remove them and even increases the horizon-wise error on the later windows (0.105 → 0.117), which stays above the floor $\Delta _ { H } / 2$ of Theorem 4.4(ii).

![](images/cd64b6b36375ed22dacff869590dcf6e78a152cf1d1eaf16035ad384bce7a981.jpg)

![](images/5bcbe2ce18d17f663fdf5354eb48783d0643b184ad96e8bd89664e50a1cca065.jpg)

![](images/2ab45d8590bce9b021df011dd41e75e7321bf50414684872f89afaf42af1c84c.jpg)

Calibration vs Forecast Quality  
![](images/71d71d13ecb527fe86b6d6c8f20544ca9210d49e3ea909a80afbd66d3c02d305.jpg)  
Figure 3: Comparison of aggregate and disaggregated calibration diagnostics for all model–dataset pairs. Top left: $\mathrm { P C E _ { p o o l } }$ versus $\mathrm { P C E _ { a v g } ; }$ pooling across horizons understates the horizon-wise error. Top right: $\mathrm { P C E _ { a v g } }$ versus $\mathrm { P C E } _ { \operatorname* { m a x } } ;$ moderate average miscalibration can coincide with severe failures at individual horizons. Bottom left: $\mathrm { Q C E _ { a v g } }$ versus $\mathrm { Q C E } _ { \operatorname* { m a x } }$ . Bottom right: $\mathrm { Q C E _ { a v g } }$ versus relative CRPS (CRPS divided by its geometric mean over models on the same dataset); better calibrated models tend to be more accurate, but calibration and accuracy are clearly distinct properties. Colors denote uncertainty representation categories. Because the maxima are taken over horizons, $\mathrm { P C E } _ { \mathrm { m a x } }$ and $\mathrm { Q C E } _ { \operatorname* { m a x } }$ grow with the horizon length and are best compared within a dataset.

## C.3 Additional recalibration results

Worst-horizon errors and gaps. Table 10 reports the relative change of the worst-horizon errors and of both gaps under the chronological split, computed as in Table 3. Worst-horizon errors improve much less than their averages. The horizon gap increases for every model and method: recalibration removes the pooled error, which is the part that $\Delta _ { H }$ does not measure, while the cancellation across horizons remains (Theorem 4.4(ii)). The conditioning gap changes little on average, so in our experiments global relabeling rarely creates or removes value-dependent miscalibration. Its relative increase for Chronos-T5 and Sundial reflects that their raw conditioning gaps are small; once their large marginal errors are removed, the value-dependent part becomes more prominent.

<table><tr><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2">below  $\hat { q } _ { 0 . 1 }$ </td><td colspan="2">above 0.9</td><td></td></tr><tr><td>Model</td><td>Dataset</td><td> $\mathrm { P C E _ { p o o l } }$ </td><td> $\mathrm { P C E _ { a v g } }$ </td><td> $\mathrm { Q C E _ { a v g } }$ </td><td> $\Delta _ { C }$  corr.</td><td>lowest</td><td>highest</td><td></td><td>lowest highest</td><td>point mass</td></tr><tr><td>Chronos-T5-Small</td><td> $\mathtt { L 0 0 P \_ S E A T T L E } / 5 \mathtt { T }$ </td><td>0.031</td><td>0.034</td><td>0.079</td><td>0.030</td><td>15%</td><td>22%</td><td>29%</td><td>4%</td><td>0%</td></tr><tr><td>Moirai-Small</td><td> $\mathtt { k d d \_ c u p \_ 2 0 1 8 / H }$ </td><td>0.015</td><td>0.032</td><td>0.071</td><td>0.028</td><td>2%</td><td>26%</td><td>12%</td><td>9%</td><td>2%</td></tr><tr><td>Moirai-Base</td><td> $\mathtt { e l e c t r i c i t y / D }$ </td><td>0.023</td><td>0.034</td><td>0.075</td><td>0.031</td><td>8%</td><td>17%</td><td>14%</td><td>5%</td><td>0%</td></tr><tr><td>TiRex</td><td> $\mathtt { e l e c t r i c i t y / W }$ </td><td>0.009</td><td>0.039</td><td>0.070</td><td>0.020</td><td>6%</td><td>12%</td><td>11%</td><td>4%</td><td>0%</td></tr><tr><td>TabPFN-TS</td><td> $\mathtt { m 4 \_ h o u r l y }$ </td><td>0.036</td><td>0.050</td><td>0.088</td><td>0.030</td><td>6%</td><td>15%</td><td>17%</td><td>10%</td><td>1%</td></tr><tr><td>Moirai-MoE-Base</td><td>m4_quarterly</td><td>0.016</td><td>0.022</td><td>0.058</td><td>0.025</td><td>6%</td><td>12%</td><td>12%</td><td>6%</td><td>0%</td></tr><tr><td>Moirai-Large</td><td> $\mathtt { c a r \_ p a r t s }$ </td><td>0.038</td><td>0.050</td><td>0.239</td><td>0.182</td><td>0%</td><td>37%</td><td>12%</td><td>3%</td><td>79%</td></tr></table>

Table 9: Model–dataset pairs with value-dependent miscalibration. “lowest”/“highest”: share of outcomes below $\hat { q } _ { 0 . 1 }$ (or above $\hat { q } _ { 0 . 9 } )$ among forecasts whose predicted quantile lies in the lowest or highest within-horizon quintile; the nominal value is 10%. “point mass”: share of targets equal to the most frequent value of the dataset. All conditioning gaps are significant $( p \leq 0 . 0 5 )$

(a) Moirai-Small, LOOP\_SEATTLE/H coverage error per horizon (all windows)  
![](images/70db914d64b1e5cbd0eeb0455b51e2392110f9af59a973a08e92b7f8a73e28ca.jpg)

(b) PCEh on the chronological test windows before and after pooled IR  
![](images/0f0aa36604cf74713fc45f71317618a928687b580e95d5e83534119ad795b2ce.jpg)  
Figure 4: Horizon cancellation for Moirai-Small on LOOP\_SEATTLE/H. (a) Coverage error $\hat { p } _ { h } ( \tau ) - \tau$ per horizon for three levels. (b) $\mathrm { P C E } _ { h }$ on the later 80% of the windows before and after isotonic recalibration fitted on the earliest 20%, with the lower bound $\Delta _ { H } / 2$ of Theorem 4.4(ii) computed on the same windows.

Random split. Table 11 repeats Table 3 for a random split of the forecast windows. The gains are larger (IR: 67% for $\mathrm { P C E _ { p o o l } ^ { - } }$ , 35% for $\mathrm { P C E _ { a v g } , }$ , and $24 \%$ for $\mathrm { Q C E _ { a v g } , }$ , with 80% of the pairs improved) likely because calibration and test windows are interleaved in time, so the calibration data represent the test period better. The ordering across metrics and the near-equivalence of the methods are the same as under the chronological split.

Per-pair effects and method selection. Figure 5 shows the effect of IR on every model–dataset pair. Selecting for each pair the method with the lowest test error, which is not possible in deployment, improves 63% of the pairs under the chronological split (82% under the random split), compared with 62% (80%) for the fixed IR map. A fixed method therefore loses little.

## C.4 Relabeling maps on a quantile grid

All four recalibration methods learn a single non-decreasing map from nominal to recalibrated levels. They differ in what they use from the calibration data (coverages at the grid levels for IR and PS, the distribution of the PIT values for HB and KDE) and in how they behave between and beyond the fitted levels. For quantile-grid forecasts, these differences shrink at the fitted levels: IR and HB coincide there, while KDE and PS smooth or approximate the same empirical coverage curve.

Proposition C.1. Let $\hat { p } ( \tau _ { q } )$ be the pooled empirical coverage ofthe $\tau _ { q } .$ -quantiles on the calibration windows and $\hat { F } _ { U }$ the empirical CDF of the approximate PIT values ofAppendix B.5. Then $\hat { p }$ is non-decreasing in q, the IR map satisfies $r _ { \mathrm { I R } } ( \tau _ { q } ) = \hat { p } ( \tau _ { q } )$ , and $\hat { F } _ { U } ( \tau _ { q } ) = \hat { p } ( \tau _ { q } )$ . Consequently, HB with bin edges that contain the grid satisfies $r _ { \mathrm { H B } } ( \tau _ { q } ) = r _ { \mathrm { I R } } ( \tau _ { q } )$ at every grid level, KDE returns a smoothed version of $\hat { F } _ { U } ( \tau _ { q } )$ , and PS returns the best logit-affine approximation of pˆ.

<table><tr><td>Model</td><td colspan="3"> $\mathbf { P C E _ { m a x } }$ </td><td colspan="3"> $\mathbf { Q C E _ { m a x } }$ </td><td colspan="3"> $\pmb { \Delta } _ { \mathbf { H } }$ </td><td colspan="3"> $\pmb { \Delta } _ { \mathbf { C } }$ </td></tr><tr><td></td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td></tr><tr><td>Moirai-Base</td><td>-1.67%</td><td>-1.53%</td><td>-1.98%</td><td>-1.80%</td><td>-1.75%</td><td>-2.31%</td><td>+9.86%</td><td>+9.64%</td><td>+9.69%</td><td>-8.46%</td><td>-8.48%</td><td>-7.22%</td></tr><tr><td>Moirai-Large</td><td>-0.86%</td><td>-0.73%</td><td>+0.40%</td><td>-0.76%</td><td>-0.76%</td><td>+0.02%</td><td>+17.00%</td><td>+16.62%</td><td>+15.70%</td><td>-3.71%</td><td>-3.87%</td><td>-3.44%</td></tr><tr><td>Moirai-MoE-Base</td><td>-3.43%</td><td>-3.37%</td><td>-3.06%</td><td>-3.00%</td><td>-3.06%</td><td>-2.59%</td><td>+16.67%</td><td>+16.68%</td><td>+11.82%</td><td>-4.68%</td><td>-4.54%</td><td>-4.51%</td></tr><tr><td>Moirai-MoE-Small</td><td>+0.33%</td><td>+0.31%</td><td>+0.20%</td><td>-0.63%</td><td>-0.58%</td><td>-0.58%</td><td>+7.00%</td><td>+7.30%</td><td>+5.27%</td><td>-0.19%</td><td>+0.03%</td><td>+0.50%</td></tr><tr><td>Moirai-Small</td><td>+2.64%</td><td>+2.78%</td><td>+2.13%</td><td>+1.46%</td><td>+1.44%</td><td>+1.04%</td><td>+4.69%</td><td>+5.18%</td><td>+5.72%</td><td>+0.93%</td><td>+0.38%</td><td>+0.11%</td></tr><tr><td>Chronos-T5-Base</td><td>-17.59%</td><td>-17.43%</td><td>-16.97%</td><td>–15.23%</td><td>-15.03%</td><td>-15.02%</td><td>+66.79%</td><td>+66.71%</td><td>+64.09%</td><td>+26.71%</td><td>+27.12%</td><td>+24.91%</td></tr><tr><td>Chronos-T5-Small</td><td>-16.43%</td><td>-16.42%</td><td>-16.69%</td><td>-14.72%</td><td>-14.67%</td><td>-14.93%</td><td>+55.23%</td><td>+55.10%</td><td>+51.71%</td><td>+26.90%</td><td>+27.23%</td><td>+26.33%</td></tr><tr><td>TabPFN-TS</td><td>-3.70%</td><td>-3.78%</td><td>-2.74%</td><td>-6.32%</td><td>-6.36%</td><td>-5.81%</td><td>+9.43%</td><td>+9.73%</td><td>+4.85%</td><td>-7.80%</td><td>-7.72%</td><td>-8.05%</td></tr><tr><td>Chronos-2</td><td>-5.86%</td><td>-5.68%</td><td>-6.32%</td><td>-6.51%</td><td>-6.47%</td><td>-6.83%</td><td>+35.86%</td><td>+36.52%</td><td>+32.24%</td><td>-3.57%</td><td>-3.42%</td><td>+0.86%</td></tr><tr><td>Chronos-Bolt-Base</td><td>+0.81%</td><td>+0.96%</td><td>+1.80%</td><td>-0.40%</td><td>-0.42%</td><td>+0.08%</td><td>+10.03%</td><td>+10.27%</td><td>+5.42%</td><td>-6.29%</td><td>-6.74%</td><td>-6.75%</td></tr><tr><td>Chronos-Bolt-Small</td><td>-4.80%</td><td>-4.51%</td><td>-3.74%</td><td>-4.53%</td><td>-4.45%</td><td>-3.71%</td><td>+18.02%</td><td>+18.10%</td><td>+12.04%</td><td>-0.77%</td><td>-1.27%</td><td>-0.54%</td></tr><tr><td>TiRex</td><td>+1.11%</td><td>+1.20%</td><td>+0.90%</td><td>-2.62%</td><td>-2.71%</td><td>-1.75%</td><td>+15.00%</td><td>+14.70%</td><td>+11.04%</td><td>-10.95%</td><td>-11.30%</td><td>-8.73%</td></tr><tr><td>TimesFM-2.5-200M</td><td>-11.45%</td><td>-11.38%</td><td>-10.69%</td><td>-11.05%</td><td>-10.98%</td><td>-10.32%</td><td>+36.86%</td><td>+36.81%</td><td>+29.80%</td><td>-4.47%</td><td>-4.43%</td><td>-4.97%</td></tr><tr><td>Sundial-Base</td><td>-14.19%</td><td>-14.19%</td><td>-14.43%</td><td>-14.38%</td><td>-14.37%</td><td>-14.46%</td><td>+66.25%</td><td>+66.37%</td><td>+65.29%</td><td>+13.74%</td><td>+13.61%</td><td>+13.43%</td></tr></table>

Table 10: Relative change of worst-horizon errors and diagnostic gaps under the chronological split, computed as in Table 3. Negative values are decreases, positive values increases.
<table><tr><td></td><td colspan="3"> $\mathbf { P C E _ { p o o l } }$ </td><td colspan="3"> $\mathbf { P C E _ { a v g } }$ </td><td colspan="3"> $\mathbf { Q C E _ { a v g } }$ </td></tr><tr><td>Model</td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td><td>IR</td><td>KDE</td><td>PS</td></tr><tr><td>Moirai-Base</td><td>-53.45%</td><td>-53.23%</td><td>-46.30%</td><td>-22.56%</td><td>-22.39%</td><td>-20.86%</td><td>-14.60%</td><td>-14.47%</td><td>-13.40%</td></tr><tr><td>Moirai-Large</td><td>-44.36%</td><td>-44.47%</td><td>-39.68%</td><td>-14.84%</td><td>-14.72%</td><td>-13.80%</td><td>-9.53%</td><td>-9.45%</td><td>-8.76%</td></tr><tr><td>Moirai-MoE-Base</td><td>-57.14%</td><td>-56.55%</td><td>-48.77%</td><td>-22.02%</td><td>-21.82%</td><td>-19.81%</td><td>-14.50%</td><td>-14.30%</td><td>-12.83%</td></tr><tr><td>Moirai-MoE-Small</td><td>-59.97%</td><td>-59.06%</td><td>-48.86%</td><td>-22.22%</td><td>-22.03%</td><td>-18.87%</td><td>-13.64%</td><td>-13.48%</td><td>-11.85%</td></tr><tr><td>Moirai-Small</td><td>-47.60%</td><td>-47.43%</td><td>-38.48%</td><td>-17.46%</td><td>-17.41%</td><td>-14.40%</td><td>-9.14%</td><td>-9.16%</td><td>-7.41%</td></tr><tr><td>Chronos-T5-Base</td><td>-78.28%</td><td>-78.52%</td><td>-74.57%</td><td>-51.31%</td><td>-51.34%</td><td>-48.85%</td><td>—40.97%</td><td>-40.96%</td><td>-38.80%</td></tr><tr><td>Chronos-T5-Small</td><td>-78.34%</td><td>–78.57%</td><td>-69.57%</td><td>-51.26%</td><td>-51.32%</td><td>-47.79%</td><td>-40.00%</td><td>-40.00%</td><td>-36.71%</td></tr><tr><td>TabPFN-TS</td><td>-60.14%</td><td>-59.95%</td><td>-53.30%</td><td>-30.11%</td><td>-29.92%</td><td>-28.97%</td><td>-20.67%</td><td>-20.49%</td><td>-19.99%</td></tr><tr><td>Chronos-2</td><td>-69.14%</td><td>-68.65%</td><td>-63.09%</td><td>-33.86%</td><td>-33.70%</td><td>-32.57%</td><td>-25.35%</td><td>-25.23%</td><td>-24.36%</td></tr><tr><td>Chronos-Bolt-Base</td><td>-52.48%</td><td>–52.53%</td><td>-48.73%</td><td>-21.13%</td><td>-21.03%</td><td>-20.55%</td><td>-11.46%</td><td>-11.47%</td><td>-11.03%</td></tr><tr><td>Chronos-Bolt-Small</td><td>-56.14%</td><td>-55.51%</td><td>-42.30%</td><td>-22.91%</td><td>-22.76%</td><td>-18.28%</td><td>-13.77%</td><td>-13.72%</td><td>-11.08%</td></tr><tr><td>TiRex</td><td>-57.62%</td><td>-57.32%</td><td>-53.61%</td><td>-23.23%</td><td>-23.00%</td><td>-22.70%</td><td>-16.55%</td><td>-16.51%</td><td>-16.07%</td></tr><tr><td>TimesFM-2.5-200M</td><td>-75.56%</td><td>-75.38%</td><td>-68.65%</td><td>-44.49%</td><td>-44.41%</td><td>-41.12%</td><td>-37.66%</td><td>-37.57%</td><td>-34.72%</td></tr><tr><td>Sundial-Base</td><td>-82.96%</td><td>-82.96%</td><td>-82.12%</td><td>-57.77%</td><td>-57.75%</td><td>-57.48%</td><td>-45.64%</td><td>-45.63%</td><td>-45.47%</td></tr></table>

Table 11: Relative change of the calibration errors under a random 20/80 split of the forecast windows; layout as in Table 3.

Proof. The predicted quantiles of each forecast are ordered, so a forecast that covers at level $\tau _ { q }$ also covers at every higher level, and $\hat { p }$ is non-decreasing; isotonic regression leaves a non-decreasing sequence unchanged. By construction of the approximate PIT, $u \leq \tau _ { q }$ holds if and only if $y \leq { \hat { q } } _ { \tau _ { q } } ,$ and averaging gives $\hat { F } _ { U } ( \tau _ { q } ) = \hat { p } ( \tau _ { q } )$ . HB evaluates $\hat { F } _ { U }$ at its bin edges, and the statements for KDE and PS follow from their definitions. □

In our implementation, IR is evaluated through interpolation on a 1000-point grid, so HB and IR differ by at most 0.0001 in any error of any model–dataset pair; we therefore report HB only here. The similarity of the methods in the main experiments is therefore expected: at the fitted levels they estimate the same coverage curve, giving similar but not identical results, and the remaining errors are largely structural (Theorem 4.4) rather than specific to one estimator. The methods do differ when recalibrated levels are needed off the fitted grid. We illustrate this with the extended-grid forecasts of Appendix C.5 (four models, 42 configurations, chronological split). In design A every map is fitted on all thirteen levels {0.01, 0.05, 0.1, . . . , 0.9, 0.95, 0.99}; in design B it is fitted on $\lbrace 0 . 1 , \dots , 0 . 9 \rbrace$ only and then applied to all thirteen, as when a model recalibrated on deciles is used for 1% and 99% quantiles (Table 12). In design A, IR is exact at every fitted level and is best throughout. In design B, IR is constant outside the fitted range, so the 1% quantile receives the level of the 10% quantile and its tail error is no better than without recalibration, whereas PS extrapolates in logit space and more than halves the tail error compared with no recalibration. In this comparison, IR is thus preferable when recalibrated levels are only needed at fitted levels, and PS when they must be extrapolated; neither choice changes the structural limits of global relabeling.

![](images/e2525c5e2378c468b991d63f20743716d31ba9d0fb4c26968fe0b700c9a8763b.jpg)

![](images/067255b5d04017bd7301ab69503968cd3791d3a6c218a461fce6fdd924100953.jpg)

![](images/c8a12192c47e5f2d7d4f775e8cf9c67f32da1fd55d9a08f4d4aeb698037789a0.jpg)

![](images/97beba2d5e54b2244b033b2dace2c964f8054d0fa2756b53de74dadceed4dbb6.jpg)

![](images/e411cc2e74aafdf3bbc5062b74faeba9fcde57850c497943d5a3563c85694c66.jpg)

![](images/633f1150375bac0b29dcf8924e20cd356b1bd8fde2253a042c8e4b5f5af2d4d3.jpg)  
Figure 5: Effect of isotonic recalibration on each model–dataset pair under the chronological split. Each panel compares an error before (x-axis) and after (y-axis) recalibration; points below the diagonal improve. Gains concentrate in $\mathrm { P C E _ { p o o l } }$ and in strongly miscalibrated pairs. Worst-horizon errors and the gaps $\Delta _ { H }$ and $\Delta _ { C }$ improve much less often, and $\Delta _ { H }$ typically increases.

<table><tr><td></td><td></td><td colspan="4">A: fitted on all 13 levels</td><td colspan="4">B: fitted on 0.1–0.9</td></tr><tr><td>T</td><td>None</td><td>IR</td><td>HB</td><td>KDE</td><td>PS</td><td>IR</td><td>HB</td><td>KDE</td><td>PS</td></tr><tr><td>0.01</td><td>0.077</td><td>0.019</td><td>0.061</td><td>0.029</td><td>0.031</td><td>0.103</td><td>0.060</td><td>0.066</td><td>0.039</td></tr><tr><td>0.05</td><td>0.098</td><td>0.030</td><td>0.030</td><td>0.030</td><td>0.034</td><td>0.058</td><td>0.058</td><td>0.057</td><td>0.041</td></tr><tr><td>0.1-0.9</td><td>0.091</td><td>0.047</td><td>0.047</td><td>0.047</td><td>0.050</td><td>0.047</td><td>0.047</td><td>0.047</td><td>0.049</td></tr><tr><td>0.95</td><td>0.078</td><td>0.029</td><td>0.029</td><td>0.031</td><td>0.032</td><td>0.053</td><td>0.053</td><td>0.053</td><td>0.034</td></tr><tr><td>0.99</td><td>0.053</td><td>0.017</td><td>0.041</td><td>0.021</td><td>0.022</td><td>0.106</td><td>0.037</td><td>0.064</td><td>0.027</td></tr><tr><td>Tails: vs. IR</td><td>一</td><td></td><td>+0.016</td><td>+0.004</td><td>+0.006</td><td>一</td><td>-0.028</td><td>-0.020</td><td>-0.045</td></tr></table>

Table 12: Pooled coverage error $| \bar { p } _ { \mathrm { t e s t } } ( \tau ) - r ( \tau ) |$ of the recalibrated levels on the extended grid, chronological split, mean over four models and 42 configurations. Bold marks the best method per design and row. The last row gives the mean difference to IR over the four tail levels. At the fitted levels the methods are similar; they separate off the fitted grid.

## C.5 Robustness analyses

Tail quantiles. Table 13 reports per-level errors on the extended grid $\{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , \ldots , 0 . 9 , 0 . 9 5 , 0 . 9 9 \}$ for one model per category: quantiles of the three sample-based models are computed from $S = 5 0 0$ samples, and Chronos-2 was run with the extended grid. For the better-calibrated models (Moirai-Base, Chronos-2), the coverage errors shrink towards the tails. Chronos-T5 and Sundial remain badly miscalibrated in the tails: at $\tau = 0 . 0 1$ their coverage errors exceed ten times the nominal tail probability. Conclusions drawn from the central grid therefore carry over to the tails.

<table><tr><td rowspan="2">T</td><td colspan="2">Moirai-Base</td><td colspan="2">Chronos-T5-S</td><td colspan="2">Chronos-2</td><td colspan="2">Sundial</td></tr><tr><td> $\mathrm { P C E } _ { h }$ </td><td> $\mathrm { Q C E } _ { h }$ </td><td> $\mathrm { P C E } _ { h }$ </td><td> ${ \mathrm { Q C E } } _ { h }$ </td><td> $\mathrm { P C E } _ { h }$ </td><td> $\mathrm { Q C E } _ { h }$ </td><td> $\mathrm { P C E } _ { h }$ </td><td> ${ \mathrm { Q C E } } _ { h }$ </td></tr><tr><td>0.01</td><td>0.010</td><td>0.015</td><td>0.133</td><td>0.135</td><td>0.010</td><td>0.014</td><td>0.154</td><td>0.156</td></tr><tr><td>0.05</td><td>0.031</td><td>0.046</td><td>0.143</td><td>0.150</td><td>0.033</td><td>0.048</td><td>0.196</td><td>0.203</td></tr><tr><td>0.10</td><td>0.051</td><td>0.074</td><td>0.143</td><td>0.154</td><td>0.058</td><td>0.081</td><td>0.202</td><td>0.214</td></tr><tr><td>0.20</td><td>0.075</td><td>0.108</td><td>0.134</td><td>0.149</td><td>0.087</td><td>0.117</td><td>0.180</td><td>0.201</td></tr><tr><td>0.30</td><td>0.084</td><td>0.124</td><td>0.123</td><td>0.142</td><td>0.101</td><td>0.130</td><td>0.150</td><td>0.177</td></tr><tr><td>0.40</td><td>0.088</td><td>0.129</td><td>0.114</td><td>0.135</td><td>0.105</td><td>0.129</td><td>0.127</td><td>0.158</td></tr><tr><td>0.50</td><td>0.087</td><td>0.125</td><td>0.110</td><td>0.130</td><td>0.102</td><td>0.122</td><td>0.118</td><td>0.150</td></tr><tr><td>0.60</td><td>0.087</td><td>0.116</td><td>0.108</td><td>0.127</td><td>0.094</td><td>0.112</td><td>0.127</td><td>0.154</td></tr><tr><td>0.70</td><td>0.074</td><td>0.100</td><td>0.110</td><td>0.127</td><td>0.082</td><td>0.097</td><td>0.151</td><td>0.168</td></tr><tr><td>0.80</td><td>0.061</td><td>0.082</td><td>0.111</td><td>0.124</td><td>0.068</td><td>0.080</td><td>0.177</td><td>0.186</td></tr><tr><td>0.90</td><td>0.042</td><td>0.057</td><td>0.104</td><td>0.113</td><td>0.046</td><td>0.056</td><td>0.193</td><td>0.198</td></tr><tr><td>0.95</td><td>0.027</td><td>0.037</td><td>0.092</td><td>0.098</td><td>0.029</td><td>0.037</td><td>0.183</td><td>0.186</td></tr><tr><td>0.99</td><td>0.009</td><td>0.013</td><td>0.066</td><td>0.068</td><td>0.011</td><td>0.014</td><td>0.128</td><td>0.130</td></tr></table>

Table 13: Per-level calibration errors on the extended grid, averaged over horizons and the 42 configurations with at least 50 forecast windows.

Number of Monte Carlo samples. Table 14 recomputes all metrics from random subsets of $S \in \{ 2 0 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 \}$ samples for the three sample-based models. All metrics stabilize at $S \approx \mathrm { 1 0 0 }$ , and moving to $S = 5 0 0$ changes them by at most 0.002. The large errors of Chronos-T5 and Sundial are therefore not Monte Carlo artifacts.

Scale differences between series. GIFT-Eval datasets combine series of very different magnitudes, so conditioning on raw predicted values could partly condition on the identity of the series. We therefore recompute QCE after standardizing every predicted quantile by the mean and standard deviation of its context window, which mirrors the instance normalization of TSFMs; coverage indicators are unchanged. Averaged over the 588 model–dataset pairs with at least 50 windows, $\mathrm { Q C E _ { a v g } }$ changes from 0.112 to 0.110 and $\Delta _ { C }$ from 0.026 to 0.023, so value dependence is not an artifact of scale differences.

<table><tr><td>Model</td><td>S</td><td> $\mathrm { P C E _ { p o o l } }$ </td><td> $\mathrm { P C E _ { a v g } }$ </td><td> $\mathrm { Q C E _ { a v g } }$ </td><td> $\Delta { _ { C } }$ </td></tr><tr><td>Moirai-Base</td><td>20</td><td>0.0499</td><td>0.0734</td><td>0.1084</td><td>0.0350</td></tr><tr><td>Moirai-Base</td><td>50</td><td>0.0463</td><td>0.0717</td><td>0.1038</td><td>0.0322</td></tr><tr><td>Moirai-Base</td><td>100</td><td>0.0461</td><td>0.0717</td><td>0.1025</td><td>0.0308</td></tr><tr><td>Moirai-Base</td><td>200</td><td>0.0462</td><td>0.0720</td><td>0.1021</td><td>0.0301</td></tr><tr><td>Moirai-Base</td><td>500</td><td>0.0463</td><td>0.0722</td><td>0.1018</td><td>0.0296</td></tr><tr><td>Chronos-T5-Small Chronos-T5-Small Chronos-T5-Small</td><td>20 50 100</td><td>0.1084 0.1025 0.1005</td><td>0.1249 0.1201 0.1186</td><td>0.1409 0.1361 0.1345</td><td>0.0160 0.0160 0.0158</td></tr><tr><td>Chronos-T5-Small</td><td>200</td><td>0.0995</td><td>0.1180</td><td>0.1338</td><td>0.0158</td></tr><tr><td>Chronos-T5-Small</td><td>500</td><td>0.0989</td><td>0.1175</td><td>0.1333</td><td>0.0158</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sundial-Base</td><td>20</td><td>0.1489</td><td>0.1650</td><td>0.1849</td><td>0.0200</td></tr><tr><td>Sundial-Base</td><td>50</td><td>0.1446</td><td>0.1611</td><td>0.1811</td><td>0.0200</td></tr><tr><td>Sundial-Base</td><td>100</td><td>0.1430</td><td>0.1596</td><td>0.1796</td><td>0.0200</td></tr><tr><td>Sundial-Base</td><td>200</td><td>0.1421</td><td>0.1589</td><td></td><td></td></tr><tr><td>Sundial-Base</td><td>500</td><td>0.1416</td><td>0.1584</td><td>0.1789 0.1784</td><td>0.0200 0.0200</td></tr></table>

Table 14: Sensitivity of calibration metrics to the number of samples S (42 configurations).

Horizon length. Table 15 compares the short-, medium-, and long-term tasks of the 21 configurations that support all three. The worst-horizon error more than doubles from short to long horizons, and the horizon gap nearly triples, so longer horizons contain increasingly heterogeneous step-wise calibration that pooled coverage hides.
<table><tr><td>Task</td><td>avg. H</td><td> $\mathrm { P C E _ { p o o l } }$ </td><td> $\mathrm { P C E _ { a v g } }$ </td><td> $\mathrm { P C E } _ { \mathrm { m a x } }$ </td><td> $\Delta _ { H }$ </td></tr><tr><td>Short</td><td>49</td><td>0.056</td><td>0.073</td><td>0.130</td><td>0.017</td></tr><tr><td>Medium</td><td>491</td><td>0.085</td><td>0.125</td><td>0.260</td><td>0.040</td></tr><tr><td>Long</td><td>737</td><td>0.097</td><td>0.144</td><td>0.297</td><td>0.047</td></tr></table>

Table 15: Calibration errors by forecasting task on the 21 configurations with short-, medium-, and long-term tasks (mean over models and configurations).

Outcomes with point masses. Four configurations have more than 40% of their targets at a single value (car\_parts, solar/10T, solar/H, temperature\_rain). As explained in Appendix B.3, even a perfect forecast can show coverage errors there. These configurations produce the largest raw conditioning gaps, but excluding all configurations with more than 5% of their targets at a single value changes the mean $\Delta _ { C }$ only from 0.035 to 0.033.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes],

Justification: All contributions relate to sections in the theoretical and empirical analysis.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes],

Justification: We discuss the limitation in the final section.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes],

Justification: We provide complete proofs in the appendix.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes],

Justification: We documented detailed experimental details in the appendix and provide supplementary code.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes],

Justification: We provide the underlying code via a public GitHub repository.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes],

Justification: Details can be derived from the appendix and the code.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes],

Justification: We provide tables with standard error when applicable.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The compute ressources used are documented in the appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: NA.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: NA

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: NA

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: NA

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: NA

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: NA

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: NA

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: NA

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.