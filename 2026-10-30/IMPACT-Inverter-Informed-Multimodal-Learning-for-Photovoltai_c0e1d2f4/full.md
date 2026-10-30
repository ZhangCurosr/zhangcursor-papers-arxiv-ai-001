# IMPACT: Inverter-Informed Multimodal Learning for Photovoltaic Generation Forecasting

Anushka Katiyar, Cong Feng, Jing Wang

National Laboratory of the Rockies

Golden, USA

{anushka.katiyar, cong.feng, jing.wang}@nlr.gov

Tristan Liu, Simon Julien, Zachary Jacobs

Latimer Controls

Denver, USA

{tristan, simon, zachary}@latimercontrols.com

Abstract—This work develops a physics-informed multimodal deep learning framework for 10- to 110-minute-ahead probabilistic photovoltaic power forecasting. The forecasting model is built on a four-branch architecture that separately encodes plantlevel temporal history and inverter-level spatial cloud dynamics through a graph attention encoder with a learned adjacency, known-future solar geometry, and the current meteorological state, fused through gated cross-attention. The geometry branch adds two forward-looking physical signals, spatial cloud heterogeneity and inverter ramp-down activity, each empirically weighted toward the horizon where it is most predictive. The decoder anchors its prediction to a smart persistence baseline, learning a horizon-conditioned correction, and the resulting quantile forecasts are calibrated online through a kernel-weighted adaptive conformal calibration step. The proposed model outperforms all 14 baselines, achieving a forecast skill score (FSS) of 0.2950, 0.099 above the next-best model. A branch-composition ablation shows that every branch contributes positively and the model retains substantial skill in challenging overcast and variable sky conditions (FSS 0.198 and 0.255).

Index Terms—Solar power forecasting, Multi-modality, Attention mechanism, Conformal prediction

## I. INTRODUCTION

Intra-hour power forecasts are a critical input for reliable and cost-effective grid operation. They enable better dispatch and reserve procurement decisions, support higher-value market participation by asset owners, and reduce the system-wide cost of managing short-term variability through ancillary services, such as regulation reserves. Photovoltaic (PV) generation is especially difficult to forecast at these horizons because cloud motion and localized atmospheric dynamics can sharply shift the plant output in minutes [1]. Purely data-driven models often lack physical grounding for these dynamics, while physicsbased models struggle to track fast-moving transient cloud conditions.

Probabilistic solar forecasts are increasingly recognized as necessary for the market and for grid operation decisions that a deterministic forecast alone cannot support [2]. Probabilistic forecasts quantify a range of likely outcomes, allowing reserve, bidding, and real-time operating decisions to be sized according to confidence rather than expected output alone by using deterministic forecasting models. This distinction is especially important for services such as PJM’s down-regulation reserve, where a deterministic forecast only indicates the expected average flexibility over the interval, while a probabilistic forecast reveals whether that flexibility is likely to be sustained across the full commitment period.

Recent machine learning approaches have advanced probabilistic solar forecasting along several fronts, but they typically fall short in three specific ways that this work addresses. First, many remain limited in modality: Attention-based models have combined multiscale temporal history with meteorological context [3] but without a spatial, inverter-level view of cloud dynamics across the plant. Second, when multiple data sources are available, they are often combined through a single fusion step rather than staged according to each modality’s structure; one recent approach fuses sky images and time-series data via a single gated-attention module [4], without staging fusion across more than two modalities or input types. Third, inverterlevel telemetry has been used to infer subplant shading and cloud motion but only at second-level granularity using a simple feed-forward network [5], without integration into a broader multimodal, intra-hour probabilistic framework.

This work directly addresses and bridges these three gaps by introducing a novel inverter-informed multimodal transformer model for PV generation forecasting, named IMPACT, which fuses a plant’s inverter-level telemetry, temporal history, solar geometry, and meteorological snapshots through gated crossattention, anchoring its predictions to a smart persistence baseline via horizon-conditioned corrections.

## II. METHODOLOGY

The IMPACT model predicts a multi-quantile representation of site-level AC power at 11 horizons (10–110 minutes, 10- minute resolution) via a horizon- and quantile-conditioned correction on the smart persistence benchmark (Section II-A), followed by online calibration. As shown in Fig. 1, four branch encoders feed the network, described in the following sections, followed by fusion and decoding. Calibration (Section II-E) is applied afterward.

![](images/7bf2e4777723c71830d0708feb8f555d808ea7ffb19efd76d273826b060f0f89.jpg)  
Fig. 1: Overview of IMPACT framework. Four branch encoders are fused via gated cross-attention and decoded into a perhorizon quantile forecast through a residual head.

## A. Data

Experiments use data from a 205-MW utility-scale solar plant at a 10-minute resolution from January 2020 through October 2023, including plant-level and per-inverter AC power (112 inverters) and four meteorological station readings of plane-of-array irradiance, global horizontal irradiance, direct normal irradiance, temperature, and wind speed. Plant power is normalized against a clear-sky reference to obtain the clear-sky index (CSI), built from a pvlib Ineichen clear-sky model propagated through a PySAM PVWatts® model, with a walk-forward multilayer perceptron (MLP) bias correction. CSI defines the smart persistence prediction interval (SP-PI) benchmark, used in the evaluations. Sky conditions are classified as clear, variable, and overcast regimes using global horizontal irradiance-based clearness, and samples are grouped by season. Model development uses three expanding-window splits, each adding 1 year of training data before validating on the following year.

## B. Branch Encoders

Four parallel encoders map heterogeneous input modalities to a shared 96-dimensional embedding space.

Branch 1 (temporal history). An 8-hour look-back (48 steps, 12 plant-level features: CSI, power, irradiance, and spatial/rolling variability metrics) is encoded by a transformer using ProbSparse self-attention [6], which concentrates capacity away from flat nighttime or clear-sky stretches. Convolutional downsampling between the three encoder layers halves the sequence length at each stage, and a depthwise-separable Conv1d sublayer replaces the standard feed-forward block.

Branch 2 (spatial inverter). The plant’s 112 inverters are aggregated into six blocks following the plant’s existing inverter layout in the operational data. Let CSI<sub>i</sub>(t) denote inverter $i \mathrm { \ ' } _ { \mathrm { s } }$ raw CSI at time t. Each inverter’s CSI is first corrected by a fixed per-inverter scale factor, $s _ { i } ,$ estimated from that inverter’s own clear-sky-day median:

$$
c _ { i } ( t ) = \mathrm { C S I } _ { i } ( t ) / s _ { i } .\tag{1}
$$

Block k’s mean corrected CSI is then $b _ { k } ( t )$ , the average of $c _ { i } ( t )$ over the inverters in that block, and the anomaly fed to the encoder is:

$$
a _ { k } ( t ) = b _ { k } ( t ) - \frac { 1 } { 6 } \sum _ { k ^ { \prime } = 1 } ^ { 6 } b _ { k ^ { \prime } } ( t ) .\tag{2}
$$

These block-level anomaly sequences pass through multiscale Conv1d layers and a bidirectional long short-term memory (LSTM), then a two-layer graph attention network [7] whose adjacency combines an unconstrained, fully learned pairwise bias term with a separate adaptive term derived from learned node embeddings. A learned gate, a sigmoid of an affine function of the recent cross-block CSI standard deviation, scales this branch’s contribution.

Branch 3 (horizon covariates). At each horizon, a deterministic geometry and calendar vector (cosine solar zenith, sine and cosine of hour of day and day of year) and the origin-time CSI are combined with two horizon-targeted lead features. Let $\sigma _ { \mathrm { i n v } } ( t )$ denote the standard deviation of the CSI across all N inverters at time t, a measure of how spatially uneven the sky currently looks. This quantity was found to most strongly correlate with plant-level power volatility roughly 20 minutes later, so it is weighted by a Gaussian of width 40 minutes centered at that lead time:

$$
w ( h ; c ) = \exp \bigl ( - ( h - c ) ^ { 2 } / ( 2 \cdot 4 0 ^ { 2 } ) \bigr ) ,\tag{3}
$$

giving the horizon-h feature spatial $\smash { \mathrm { \ l e a d } ( t , h ) = \sigma _ { \mathrm { i n v } } ( t ) }$ $w ( h ; 2 0 )$ ). Separately, let ramp down frac(t) denote the fraction of inverters whose CSI dropped by more than 0.1 over the trailing 20 minutes—a spatial count of local ramps already in progress. This signal was found to lead to a broader ramp-down in the entire plant roughly 10 minutes later, independently of $\sigma _ { \mathrm { i n v } } ( t )$ , so it receives the same Gaussian weighting peaked closer to the present, ramp count $( t , h ) =$ ramp down frac(t) · w(h; 10).

Branch 4 (meteorological snapshot). A three-layer MLP encodes the most recently measured state: module and ambient temperature, wind speed, instantaneous plane-of-array spatial variability, and beam fraction. Three additional descriptors computed from the full inverter matrix are appended: $\sigma _ { \mathrm { i n v } } ( t )$ itself and the 20- and 60-minute changes in block-to-block CSI spread, which track cloud edges entering or leaving the plant. The two spread-change terms are scaled by $\sigma _ { \mathrm { i n v } } ( t )$ , which drives them toward zero during spatially uniform conditions.

## C. Fusion

Branches 1, 2, and 4 are combined through a staged fusion; Branch 3 bypasses fusion. This staged design addresses the single-gate limitation of prior multimodal PV fusion noted in Section I [4] by combining branches according to each one’s input structure rather than through a single fusion step. Branches 1 and 2 are both sequences—over time and over spatial blocks, respectively—so they naturally combine through cross-attention. Branch 4 is a single instantaneous snapshot with no sequence dimension of its own, so it is better suited to modulating the combined signal through a gate. Branch 3 is indexed by forecast horizon and is entirely deferred to the decoder, where it directly initializes the per-horizon query (Section II-D). Let $h _ { 1 }$ denote Branch 1’s temporal sequence and $h _ { 2 }$ Branch 2’s block embeddings. Branch 1 first crossattends to Branch 2:

h<sup>′</sup> = LayerNorm $\left( h _ { 1 } + \mathrm { A t t n } ( Q { = } h _ { 1 } , K { = } h _ { 2 } , V { = } h _ { 2 } ) \right)$ , (4) followed by a position-wise feed-forward layer. Branch 4’s embedding, $h _ { 4 }$ , then produces a gate:

$$
g = \sigma ( W _ { g } h _ { 4 } + b _ { g } ) , \quad g \in ( 0 , 1 ) ^ { d } ,\tag{5}
$$

and the fused representation blends the spatially enriched sequence with the original temporal one:

$$
h _ { \mathrm { f u s e d } } = \mathrm { L a y e r N o r m } \big ( g \odot h _ { 1 } ^ { \prime } + ( 1 - g ) \odot h _ { 1 } \big ) .\tag{6}
$$

## D. Decoding and Probabilistic Output

The decoder predicts a correction on top of the smart persistence benchmark for 99 quantile levels spanning from 0.01 to 0.99. The Branch 3 horizon queries, offset by a learned perhorizon embedding, cross-attend to the fused sequence, receive an additive Branch 4 bias, and are mapped to 99 corrections by a small MLP. A horizon-conditioned gate, shared across quantiles and bounded in (0, 1), scales the correction before being added to the persistence anchor and clipped to the valid power range, after which the 99 outputs are sorted along the quantile axis.

## E. Loss and Online Calibration

The network is trained with an equal blend of pinball loss and a continuous ranked probability score (CRPS) score [8]. Because raw network outputs under-cover relative to their nominal quantile levels, a post hoc calibration step corrects them online, separately for each horizon and quantile. Past residuals are stored in a rolling buffer together with a feature vector describing the current conditions (time of day, solar zenith, recent CSI variability). Each buffered residual is then weighted by its similarity to the current feature vector, x<sub>t</sub>:

$$
w _ { i } = \exp \left( - \| x _ { t } - x _ { i } \| ^ { 2 } / ( 2 \sigma ^ { 2 } ) \right)\tag{7}
$$

with weights normalized to sum to 1, and the correction is the weighted quantile of the buffered residuals at the nominal quantile level. This kernel-weighted residual retrieval builds on sequential predictive conformal inference [9].

## III. EXPERIMENTAL SETUP

## A. Evaluation Metrics

IMPACT’s deterministic accuracy against SP-PI uses normalized root mean square error (nRMSE), normalized mean absolute error (nMAE), and $R ^ { 2 }$ , computed from each model’s median $( q ~ = ~ 0 . 5 0 )$ quantile prediction. Forecast skill relative to SP-PI is $S _ { f } ~ = ~ 1 - \mathrm { R M S E } _ { \mathrm { m o d e l } } / \mathrm { R M S E } _ { \mathrm { S P - P I } }$ . All other comparisons use probabilistic metrics computed postcalibration (Section II-E): CRPS and an analogous forecast skill score, $\mathrm { F S S } ~ = ~ 1 - \mathrm { C R P S } _ { \mathrm { m o d e l } } / \mathrm { C R P S } _ { \mathrm { S P - P I } }$ , both integrated over the full 99-point predicted quantile grid and therefore already reflecting calibration and sharpness across the entire predicted distribution. Prediction interval coverage probability (PICP) and prediction interval normalized average width (PINAW) [10], by contrast, are each defined at a single nominal level; rather than averaging across all 49 central intervals the 99-point grid admits, we evaluate them at four representative levels (80/90/95/98%), which are then aggregated as the mean absolute coverage error (MACE, mean $| \mathrm { P I C P } _ { \ell } - \ell | )$ and PINAW-avg (mean PINAW), both across those levels. $\mathrm { C R P S } _ { \mathrm { S P - P I } }$ is the uncalibrated benchmark of Section II-A, so SP-PI’s calibrated row reflects the skill gained from the calibration alone, not zero. $S _ { f }$ needs no such caveat because it is computed before the calibration.

## B. Baselines

Fourteen baselines are evaluated on identical data splits, preprocessing, and loss functions. The baseline suite is chosen to span the major forecasting paradigms. Smart persistence is the domain-standard floor, statistical and classical methods represent feature-engineered approaches, linear models test whether the attention’s added complexity is required, and deep learning and transformer baselines represent current sequenceforecasting approaches.

• Smart persistence: SP-PI, the primary benchmark for all skill scores

• Statistical: autoregressive integrated moving average (ARIMA) with automatic order selection

• Classical machine learning: support vector regression (SVR), quantile random forest (QRF), XGBoost

• Linear: DLinear, NLinear

• Deep learning: LSTM, GRU, TCN [11], N-BEATS

• Transformers: Informer [6], TFT [12], PatchTST [13]. All deep learning, linear, and transformer baselines use AdamW with standard regularization and early stopping.

## C. Ablation Study

The contribution of the spatial-inverter and horizon-covariate branches is isolated through a cumulative branch-composition

TABLE I: Deterministic, probabilistic, and per-horizon forecast skill comparison, threefold mean, ranked by FSS. Subscript 4 on MACE/PINAW-avg denotes the mean over the four canonical central interval levels (80/90/95/98%).
<table><tr><td rowspan="2">Model</td><td colspan="4">Deterministic</td><td colspan="4">Probabilistic</td><td colspan="10">FSS by horizon (min)</td></tr><tr><td>nRMSE</td><td>nMAE</td><td> $R ^ { 2 }$ </td><td> $S _ { f }$ </td><td>CRPS</td><td>FSS</td><td>MACE4</td><td>PINAW-avg4</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80 90</td><td>100</td><td>110</td></tr><tr><td>IMPACT</td><td>0.1375</td><td>0.0669</td><td>0.7683</td><td>0.1616</td><td>|0.0360</td><td>0.2950</td><td>0.0067</td><td>0.2519</td><td>0.172</td><td>0.221</td><td>0.259</td><td>0.285</td><td>0.304 0.322</td><td></td><td>0.331</td><td>0.338 0.341</td><td>0.339</td><td>0.334</td></tr><tr><td>XGBoost-Q</td><td>0.1452</td><td>0.0738</td><td>0.7445</td><td>0.1146</td><td>0.0398</td><td>0.1962</td><td>0.0072</td><td>0.3114</td><td>-0.242</td><td>0.050</td><td>0.155</td><td>0.204</td><td>0.238</td><td>0.266</td><td>0.281</td><td>0.291 0.299</td><td>0.310</td><td>0.308</td></tr><tr><td>QRF</td><td>0.1515</td><td>0.0803</td><td>0.7212</td><td>0.0762</td><td>0.0422</td><td>0.1405</td><td>0.0072</td><td>0.3747</td><td>-0.392</td><td>-0.038</td><td>0.083</td><td>0.144</td><td>0.188</td><td>0.226</td><td>0.247</td><td>0.259 0.268</td><td>0.280</td><td>0.281</td></tr><tr><td>Informer-Q</td><td>0.1556</td><td></td><td>0.0787 0.7073</td><td>0.0512</td><td>0.0424</td><td>0.1366</td><td>0.0072</td><td>0.2951</td><td>-0.431</td><td>-0.032</td><td>0.110</td><td>0.179</td><td>0.215</td><td>0.238</td><td>0.244</td><td>0.250 0.247</td><td>0.248</td><td>0.235</td></tr><tr><td>TFT-Q</td><td>0.1553</td><td></td><td>0.0817 0.7081</td><td>0.0530</td><td>0.0436</td><td>0.1077</td><td>0.0084</td><td>0.3291</td><td>-0.540</td><td>-0.083</td><td>0.078</td><td>0.163</td><td>0.200 0.225</td><td></td><td>0.237</td><td>0.236 0.233</td><td>0.227</td><td>0.209</td></tr><tr><td>SP-PI</td><td>0.1640</td><td></td><td>0.08580.6674</td><td>0.0000</td><td>0.0461</td><td>0.1058</td><td>0.0195</td><td>0.3597</td><td>0.062</td><td>0.079</td><td>0.092</td><td>0.102</td><td>0.110</td><td>0.114</td><td>0.118</td><td>0.121 0.122</td><td>0.122</td><td>0.121</td></tr><tr><td>LSTM-Q</td><td>0.1593</td><td></td><td>0.0840 0.6893</td><td>0.0287</td><td>0.0442</td><td>0.0975</td><td>0.0128</td><td>0.3126</td><td>-0.519</td><td>-0.076</td><td>0.081</td><td>0.152</td><td>0.192</td><td>0.214</td><td>0.220</td><td>0.219 0.210</td><td>0.199</td><td>0.180</td></tr><tr><td>DLinear-Q</td><td>0.1558</td><td></td><td>0.0847 0.7060</td><td>0.0500</td><td>0.0453</td><td>0.0904</td><td>0.0196</td><td>0.4031</td><td>-0.369</td><td>-0.062</td><td>0.048</td><td>0.098</td><td>0.135</td><td>0.164</td><td>0.181</td><td>0.191 0.197</td><td>0.204</td><td>0.205</td></tr><tr><td>TCN-Q</td><td>0.1566</td><td>0.0827</td><td>0.7043</td><td>0.0451</td><td>0.0445</td><td>0.0819</td><td>0.0080</td><td>0.3111</td><td>-0.621</td><td>-0.138</td><td>0.037</td><td>0.127</td><td>0.178 0.207</td><td></td><td>0.224 0.229</td><td>0.228</td><td>0.222</td><td>0.207</td></tr><tr><td>NLinear-Q</td><td>0.1584</td><td></td><td>0.0855 0.6962</td><td>0.0341</td><td>0.0462</td><td>0.0741</td><td>0.0185</td><td>0.4056</td><td>-0.378</td><td>-0.074</td><td>0.030</td><td>0.082</td><td>0.115 0.145</td><td></td><td>0.162</td><td>0.174 0.180</td><td>0.188</td><td>0.191</td></tr><tr><td>GRU-Q</td><td>0.1570</td><td>0.0857</td><td>0.6998</td><td>0.0427</td><td>0.0453</td><td>0.0695</td><td>0.0092</td><td>0.3095</td><td>-0.585</td><td>-0.118</td><td>0.048</td><td>0.129</td><td>0.170 0.198</td><td></td><td>0.205</td><td>0.2000.191</td><td>0.174</td><td>0.153</td></tr><tr><td>SVR-PI</td><td>0.1702</td><td>0.0931</td><td>0.6481</td><td>-0.0378</td><td>0.0535</td><td>-0.0941</td><td>0.0116</td><td>0.4160</td><td>-0.794</td><td>-0.354</td><td>-0.202</td><td>-0.088</td><td>-0.020</td><td>-0.001</td><td>0.040 0.051</td><td>0.103</td><td>0.116</td><td>0.114</td></tr><tr><td>N-BEATS-Q</td><td>0.1726</td><td>0.1017</td><td>0.6377</td><td>-0.0524</td><td>0.0542</td><td>-0.1162</td><td>0.0093</td><td>0.3511</td><td>-0.952</td><td>-0.343</td><td>-0.132</td><td>-0.030</td><td>0.027</td><td>0.056</td><td>0.064 0.051</td><td>0.026</td><td>-0.003</td><td>-0.043</td></tr><tr><td>ARIMA-PI</td><td>0.1739</td><td>0.1046</td><td>0.6306</td><td>-0.0604</td><td>0.0556</td><td>-0.1205</td><td>0.0085</td><td>0.4679</td><td>-0.620</td><td>-0.321</td><td>-0.213</td><td>-0.149</td><td>-0.098</td><td>-0.049</td><td>-0.014 0.011</td><td>0.028</td><td>0.046</td><td>0.055</td></tr><tr><td>PatchTST-Q</td><td>0.1758</td><td>0.1055</td><td>0.6278</td><td>-0.0720</td><td>0.0551</td><td>-0.1397</td><td>0.0086</td><td>0.3453</td><td>-1.049</td><td>-0.409</td><td>-0.177</td><td>-0.064 -0.004</td><td>0.034</td><td>0.045</td><td></td><td>0.043 0.030</td><td>0.016</td><td>-0.002</td></tr></table>

ablation, adding one branch at a time (B1, B1+B2, B1+B2+B3, full model) across all cross-validation folds, with each added branch retaining its full mechanism, as described in Section II.

## IV. RESULTS

Results are evaluated across multiple angles: skill relative to SP-PI and a broad baseline suite; calibration and sharpness across the full predicted distribution; and the contribution of each architectural component, each further broken down by sky condition, season, and time of day.

## A. Deterministic Forecasting Performance

Table I compares IMPACT’s median forecast against SP-PI and all 13 baselines averaged across all 11 horizons and 3 cross-validation folds. The model reduces nRMSE from SP-PI’s 0.1640 to 0.1375 and nMAE from 0.0858 to 0.0669, increasing $R ^ { 2 }$ from 0.6674 to 0.7683, for an overall forecast skill of $S _ { f } ~ = ~ 0 . 1 6 1 6$ . It achieves the lowest nRMSE and nMAE and the highest $R ^ { 2 }$ of any model evaluated, leading the strongest deterministic baseline, XGBoost-Q, by 0.008 nRMSE and 0.024 $R ^ { 2 }$

## B. Probabilistic Forecasting Performance

Table I ranks IMPACT against all 14 baselines by FSS, leading the strongest (XGBoost-Q) by 0.099 FSS. QRF and Informer-Q form a clear second tier (FSS 0.141 and 0.137), while most of the remaining deep learning and linear baselines cluster more loosely between 0.07 and 0.11 FSS. On MACE, IMPACT (0.0067) is the best-calibrated of all 15 compared models, narrowly ahead of XGBoost-Q and Informer-Q (0.0072 each), and it remains the sharpest on PINAW-avg (0.2519) as well. The classical statistical baselines (ARIMA-PI, SVR-PI) fall below SP-PI, indicating that a poorly calibrated model can be actively worse than doing nothing. At 10 minutes, every learned baseline scores below zero, meaning SP-PI alone beats every learned model at the shortest lead time; most baselines do not turn positive until 30 minutes. IMPACT is the only one with positive skill at every horizon, and it remains at the top of the ranking throughout the full 10- to 110-minute range. Fig. 2 confirms this pattern visually: At 10 minutes, the baselines’ forecasts track actual power far less reliably than IMPACT’s, consistent with their negative FSS at that horizon, while by 60 and 110 minutes, the gap narrows as the baselines catch up. Fig. 3 shows IMPACT’s per-quantile reliability and sharpness remain strong across all horizons.

![](images/70aa93c15fda1307bf56119d14f43b9f15919bc70310de5ac3382d45791ecfc1.jpg)  
Fig. 2: Forecast comparison at three horizons. Each row shows the actual power (black) against IMPACT (green, solid) with four nested prediction intervals in one blue family (80/90/95/98%, darkest to lightest) and four baselines (dotted) at 10, 60, and 110 minutes across three example days spanning clear, partly cloudy, and turbulent conditions.

## C. Sensitivity Analysis

A branch-composition ablation (Table II) isolates the contribution of each branch, adding one at a time with its full mechanism. Branch variants were trained and evaluated at the 9-quantile configuration [0.01, 0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975, 0.99], which contains exactly the levels needed to construct the four operationally relevant prediction intervals (80/90/95/98%) reported throughout this evaluation. Every added branch improves the forecast skill, and the improvement is not evenly distributed: Branch 3’s horizon-targeted spatial feature contributes the largest single gain (+0.038 FSS), with others contributing smaller, still positive increments.

![](images/16e2ed6caad044bbf45b5ec52f5609f41301e2f7c6ac91f6b4837d29e4b716de.jpg)

![](images/50eb820299e91faef918eeeab6f8056ef86dc1395f0a31cc49f1f015e51869df.jpg)  
Fig. 3: Per-quantile reliability and sharpness of IMPACT’s forecasts across all horizons. Left: quantile coverage, with the diagonal denoting perfect calibration. Right: mean PINAW at the four nominal levels (80/90/95/98%).

TABLE II: Branch-composition ablation, threefold mean, evaluated at the 9-quantile configuration.
<table><tr><td rowspan=1 colspan=6>Configuration               FSS</td></tr><tr><td rowspan=2 colspan=6>B1                        0.2735</td></tr><tr><td rowspan=2 colspan=2> $\mathbf { B } 1 + \mathbf { B } 2 + \mathbf { B } 3$ </td><td rowspan=1 colspan=1>0.2920</td></tr><tr><td rowspan=1 colspan=2>2 + B3</td><td rowspan=1 colspan=2>33</td><td></td><td rowspan=1 colspan=1>0.3304</td></tr><tr><td rowspan=1 colspan=6> $\mathbf { B } \mathbf { l } + \mathbf { B } \mathbf { 2 } + \mathbf { B } \mathbf { 3 } + \mathbf { B } \mathbf { 4 }$ (full)  0.3485</td></tr></table>

TABLE III: Stratified analysis, threefold mean.
<table><tr><td>Type</td><td>Stratum</td><td>CRPS</td><td>FSS</td><td>MACE</td><td>PINAW-avg</td></tr><tr><td>Regime</td><td>Clear</td><td>0.0268</td><td>0.337</td><td>0.006</td><td>0.202</td></tr><tr><td>Regime</td><td>Overcast</td><td>0.1103</td><td>0.198</td><td>0.023</td><td>0.646</td></tr><tr><td>Regime</td><td>Variable</td><td>0.1094</td><td>0.255</td><td>0.022</td><td>0.654</td></tr><tr><td>Season</td><td>Summer</td><td>0.0222</td><td>0.366</td><td>0.009</td><td>0.178</td></tr><tr><td>Season</td><td>Fall</td><td>0.0287</td><td>0.291</td><td>0.032</td><td>0.187</td></tr><tr><td>Season</td><td>Winter</td><td>0.0607</td><td>0.280</td><td>0.008</td><td>0.397</td></tr><tr><td>Season</td><td>Spring</td><td>0.0615</td><td>0.280</td><td>0.021</td><td>0.403</td></tr><tr><td>Time of day</td><td>Morning</td><td>0.0466</td><td>0.473</td><td>0.012</td><td>0.342</td></tr><tr><td>Time of day</td><td>Midday</td><td>0.0357</td><td>0.238</td><td>0.007</td><td>0.239</td></tr><tr><td>Time of day</td><td>Evening</td><td>0.0277</td><td>0.274</td><td>0.008</td><td>0.216</td></tr></table>

A stratified analysis (Table III) further shows that the performance systematically varies by sky condition, season, and time of day. By sky regime, skill is highest in clear conditions and lowest in overcast sky, with variable between the two; overcast and variable sky show comparably weak calibration (MACE 0.023 and 0.022) and comparably wide intervals. By season, summer leads clearly, while others are closely grouped behind it. By time of day, both transition windows outperform midday: morning has the widest intervals of the three, while evening is the sharpest; midday is nonetheless the best calibrated (MACE 0.007), narrowly ahead of evening (0.008) and clearly ahead of morning (0.012).

## V. CONCLUSION AND FUTURE WORK

This paper presented IMPACT, a multimodal deep learning framework for probabilistic intra-hour PV power forecasting, anchored to smart persistence and refined through online conformal calibration. IMPACT outperformed all 14 baselines, leading the strongest one by 0.099 FSS. Branch-composition ablation showed every branch contributing positive, nonredundant skill, with the spatial and horizon-covariate branches contributing the most. Performance varied by sky regime, season, and time of day, with overcast conditions the hardest and variable close behind, and unexplained asymmetry in calibration behavior between morning and evening transitions.

Future work includes combining IMPACT’s forecasts with external vendor irradiance products to extend accuracy across the full horizon range. Access to true geographic coordinates for each inverter could introduce a physically grounded adjacency prior, which the current fully data-driven adjacency lacks. The performance gaps and transition window asymmetry noted above also warrant further investigation.

## REFERENCES

[1] F. Barbieri, S. Rajakaruna, and A. Ghosh, “Very short-term photovoltaic power forecasting with cloud modeling: a review,” Renewable and Sustainable Energy Reviews, vol. 75, pp. 242–263, 2017.

[2] B. Li and J. Zhang, “A review on the integration of probabilistic solar forecasting in power systems,” Solar Energy, vol. 210, pp. 68–86, 2020.

[3] G. Wang, Y. Zhou, Y. Yan, Z. Zhou, Z. Yang, L. Dai, and J. Huang, “Probabilistic photovoltaic power forecasting with reliable uncertainty quantification via multi-scale temporal–spatial attention and conformalized quantile regression,” Sustainability, vol. 18, no. 2, p. 739, 2026.

[4] H. Jiang, Z. Yang, Y. Li, R. Wang, J. Shi, and Y. Zhao, “A novel multimodal framework for ultra-short-term photovoltaic power forecasting via fusion of sky images and time-series data,” International Journal of Electrical Power & Energy Systems, vol. 180, p. 112064, 2026.

[5] X. Meng, F. Gao, T. Xu, K. Zhou, W. Li, and Q. Wu, “Inverter-datadriven second-level power forecasting for photovoltaic power plant,” IEEE Transactions on Industrial Electronics, vol. 68, no. 8, pp. 7034– 7044, 2021.

[6] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, 2021, pp. 11 106–11 115.

[7] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li´ o, and\` Y. Bengio, “Graph attention networks,” in International Conference on Learning Representations (ICLR), 2018.

[8] C. Feng, J. Hao, R. Yang, and J. Han, “Deep quantile transformer for probabilistic net load forecasting with extreme solar penetration,” in 2025 IEEE Power & Energy Society General Meeting (PESGM). IEEE, 2025, pp. 1–5.

[9] C. Xu and Y. Xie, “Sequential predictive conformal inference for time series,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 38 707–38 727.

[10] Q. Paletta, G. Terren-Serrano, Y. Nie, B. Li, J. Bieker, W. Zhang,´ L. Dubus, S. Dev, and C. Feng, “Advances in solar forecasting: Computer vision with deep learning,” Advances in Applied Energy, vol. 11, p. 100150, 2023.

[11] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[12] B. Lim, S. O. Arık, N. Loeff, and T. Pfister, “Temporal fusion transform-<sup>¨</sup> ers for interpretable multi-horizon time series forecasting,” International Journal of Forecasting, vol. 37, no. 4, pp. 1748–1764, 2021.

[13] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in International Conference on Learning Representations (ICLR), 2023.