# FARR: FORECAST-AWARE RESIDUAL REFINEMENT WITH MATCHED ENDPOINT EVIDENCE FOR LONG-HORIZON TIME-SERIES FORECASTING

Yiyang Zhang<sup>1</sup>, Yuli Zheng<sup>2</sup>, Bin Li<sup>3</sup>, Meimei Zhang<sup>4</sup>, Qiang Sun<sup>1,\*</sup>

<sup>1</sup>Nantong University, Nantong, China

<sup>2</sup>Faculty of Mathematics and Statistics, University of Ottawa, Ottawa, Canada <sup>3</sup>Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China <sup>4</sup>Amazon.com Inc., Seattle, WA, USA

This work has been submitted to IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

## Abstract

Large language models use auxiliary conditioning to incorporate additional information into timeseries forecasts. However, auxiliary conditioning does not consistently improve accuracy and can increase forecasting error when its influence is poorly matched to the observed temporal context. We propose Forecast-Aware Residual Refinement (FARR), which learns to correct forecasts using endpoint evidence. FARR obtains a valid forecast under the standard condition and a null forecast under a predefined null condition. Their difference reveals how conditioning changes the prediction. A shared residual refiner combines this endpoint evidence with historical features to estimate the remaining error of the valid forecast. Horizon-wise gating then controls the contribution of the predicted residual to the final output. Experiments on four long-horizon forecasting benchmarks show that FARR achieves the best dataset-average MSE and MAE among the compared methods. Code will be released at https://github.com/blossomrichzyy-star/farr-forecast-aware-residual-refinement.

Index Terms— Long-horizon forecasting, endpoint evidence, residual refinement, horizon-wise gating, language-model-based forecasting

## 1 Introduction

Language-model-based forecasting incorporates auxiliary information into time-series prediction through conditioning. Time-LLM adapts numerical inputs to pretrained language models through reprogramming [1]. S<sup>2</sup>IP-LLM learns prompts informed by the language embedding space [2], while Deep TPC introduces temporal information at multiple decoder layers [3]. AutoTimes further adapts autoregressive language modeling to time-series forecasting [4]. These methods differ in how they construct conditioning, whose forecasting benefit depends on the data.

In parallel, forecasting architectures have improved the representation of temporal structure. DLinear demonstrates the strength of simple decomposition-based prediction [5]. Informer reduces the cost of longsequence attention [6], and Autoformer incorporates decomposition and auto-correlation [7]. FEDformer introduces frequency-domain operations [8], whereas PatchTST learns from temporal patches [9]. iTrans former models dependencies between variables [10], and TimeMixer combines information across temporal scales [11]. Periodic modeling in CycleNet [12], efficient attention in Ister [13], and spectral filtering before attention [14] provide further improvements. Large-scale pretraining offers another direction, as illustrated by Chronos [15], TimesFM [16], and Time-MoE [17]; DistilTS transfers forecasting capabilities to smaller models [18].

For language-model-based forecasters, a related line of work improves the interaction between numerical observations and auxiliary representations. CALF uses cross-modal fine-tuning [19], and TimeCMA aligns time-series and language representations [20]. Retrieval augmentation instead supplies relevant historical examples [21]. Nevertheless, stronger conditioning does not imply that every resulting forecast is more accurate. Controlled studies also show that the benefits of language-model components depend on the forecasting setting [22].

Figure 1 illustrates this distinction. A standard pipeline produces a single conditioned forecast, leaving its change relative to a null condition unobserved. Conditioning can increase, barely change, or decrease prediction error. The difference between two forecasts is observable at inference, but its relationship to future error must be learned from training data. Historical context is therefore needed to determine how this difference should influence the final prediction.

![](images/26a3f7598278c47bdce5fd3d518385aee4d24c609e2a4a80e94d0f4045784443.jpg)  
(c) Ours: FARR: Forecast-Aware Residual Refinement  
Fig. 1. Motivation of FARR. (a) A single conditioned forecast does not reveal the prediction change relative to a null condition. (b) Conditioning can have different effects on forecasting error. (c) FARR combines endpoint evidence with historical features for residual refinement and applies horizon-wise gating before adding the residual to the effective valid forecast.

We address this problem through residual refinement. RefineBridge demonstrates that forecast outputs can serve as starting points for a learned refinement process [23]. FARR focuses on the additional information provided by two forecasts generated under different conditions. It combines their difference with historical features to predict a residual relative to the valid forecast. A horizon-wise gate controls how strongly this residual changes the output.

Our contributions are threefold:

• We introduce endpoint evidence from valid and null forecasts, making the prediction change associated with conditioning available to residual refinement.

• We develop a shared residual refiner and horizon-wise gating that use endpoint evidence and historical features to correct the valid forecast.

• We evaluate FARR on four forecasting benchmarks, obtaining the best dataset-average MSE and MAE among the compared methods. Ablations examine endpoint inputs, historical features, residual refinement, and forecast adjustment.

## 2 Method

## 2.1 Problem formulation

Let $X _ { i } \in \mathbb { R } ^ { L \times C }$ denote the historical observations at forecast origin i, where L is the input length and C is the number of channels. The target $Y _ { i } \in \mathbb { R } ^ { H \times C }$ contains the next H observations. A frozen forecaster $F _ { \theta }$ receives the history under a specified conditioning operation. FARR consists of an Endpoint Proposal Module (EPM), a residual refiner, and a Conservative Residual Composer (CRC), as shown in Figure 2. Refinement uses observed history and matched endpoint forecasts, providing a common refinement interface for forecasters with a compatible conditioning pathway.

![](images/3ff5eb06bdbf8f92293bd66526578c4df32aa619b65eeab404763ca98d19ed36.jpg)  
Fig. 2. FARR architecture. EPM obtains matched endpoints, and forecast adjustment yields the effective valid forecast. The residual refiner combines endpoint evidence with historical context in U. CRC applies horizon-wise gating and adds the predicted residual to $\widehat { Y } ^ { v , \mathrm { e f f } }$

## 2.2 Endpoint evidence

For a shared history, EPM evaluates the frozen forecaster under standard conditioning $S _ { i } ^ { v }$ and a fixed reference (null) operation $S _ { i } ^ { 0 }$

$$
\widehat { Y } _ { i } ^ { v } = F _ { \theta } ( X _ { i } , S _ { i } ^ { v } ) , \qquad \widehat { Y } _ { i } ^ { 0 } = F _ { \theta } ( X _ { i } , S _ { i } ^ { 0 } ) .\tag{1}
$$

Both passes share the checkpoint, history, preprocessing, output settings, and sample order; only conditioning changes. Here, “valid” denotes standard conditioning, not the validation split. We define

$$
\Delta \widehat { Y } _ { i } = \widehat { Y } _ { i } ^ { v } - \widehat { Y } _ { i } ^ { 0 } .\tag{2}
$$

The forecasts and their difference constitute endpoint evidence; $\Delta \widehat { Y } _ { i }$ is the endpoint disagreement induced by the reference operation.

The reference operation acts on a designated auxiliary conditioning pathway. The zero symbols in Figure 2 denote this intervention, with historical inputs unchanged. The refiner uses only the resulting forecasts and observed history.

## 2.3 Residual refinement and forecast adjustment

Residual architectures model successive corrections in time-series forecasting [28]. The refiner receives a fixed 18-dimensional feature vector $u _ { i , h , c } .$ . Indices 0–7 describe history: the latest value, mean, scale, slope, first-difference scale, local volatility, motif similarity, and ambiguity. Indices 8–9 are the null and valid forecasts. Indices 10–16 describe their disagreement: signed difference, absolute difference, history-scalenormalized difference, first-order difference, smoothness, relative displacement, and sign consistency. Index 17 is the normalized horizon coordinate. These 18 features are deterministic functions of history, forecasts, and horizon position; the encodings in Figure 2 denote these fixed computations. Their mapping to a residual is learned.

History statistics use the latest 168 observations. The history scale is the population standard deviation, floored at $1 0 ^ { - 4 } \mathrm { ; }$ ; slope is $( x _ { L } - x _ { L - 1 6 7 } ) / 1 6 7$ . The first-difference scale uses the same normalization and floor. Local volatility is the mean absolute first difference. Motif similarity is the cosine similarity between the latest and preceding 24-point windows, representing past-only recurrence in Figure 2. Ambiguity is $\mathrm { c l i p } ( { v } / { \sigma } + 1 - m , 0 , 1 0 )$ , with local volatility $v ,$ history scale $\sigma ,$ and motif similarity m. For endpoint disagreement $\Delta _ { h }$ , the derived quantities are

$$
d _ { h } ^ { ( 1 ) } = \Delta _ { h } - \Delta _ { h - 1 } , \qquad q _ { h } = \mathrm { c l i p } \left( \frac { | d _ { h } ^ { ( 1 ) } | } { | \Delta _ { h } | + 1 0 ^ { - 3 } } , 0 , 1 0 \right) ,\tag{3}
$$

$$
r _ { h } = \frac { | \Delta _ { h } | } { | \hat { Y } _ { h } ^ { 0 } | + 1 0 ^ { - 3 } } , \qquad k = \left| \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \mathrm { s i g n } ( \Delta _ { h } ) \right| ,\tag{4}
$$

where $d _ { 1 } ^ { ( 1 ) } = 0$ . The horizon coordinate is linearly spaced from 0 to 1. Concatenating these features gives $U \in \mathbb { R } ^ { \hat { B } \times H \times 1 8 }$ , with B single-channel windows. One MLP with two hidden layers is shared across prediction steps and channels:

$$
\widehat { R } _ { i , h , c } = \operatorname { t a n h } \bigl ( g _ { \Phi } \bigl ( u _ { i , h , c } \bigr ) \bigr ) , \qquad g _ { \Phi } : 1 8 \to 1 2 8 \to 1 2 8 \to 1 .\tag{5}
$$

Both hidden layers use GELU; the output layer is zero-initialized. Thus $\widehat { R } \in [ - 1 , 1 ]$ in standardized space. Stage 2 applies a forecast adjustment to the valid forecast relative to the null forecast:

$$
\widehat { Y } _ { i , h , c } ^ { v , \mathrm { a d j } } = \widehat { Y } _ { i , h , c } ^ { 0 } + [ 1 + 0 . 1 0 \operatorname { t a n h } ( s _ { h } ) ] \Delta \widehat { Y } _ { i , h , c } + 0 . 1 0 \operatorname { t a n h } ( b _ { h } ) .\tag{6}
$$

The horizon-wise vectors s, $b \in \mathbb { R } ^ { H }$ are zero-initialized and shared across samples and channels. Stage 2 recomputes endpoint-dependent features with $\widehat { Y } ^ { v , \mathrm { a d j } }$ and disagreement $\hat { Y } ^ { v , \mathrm { a d j } } \dot { - } \hat { Y } ^ { 0 }$ , using residual target $Y - \widehat { Y } ^ { v , \mathrm { a d j } }$ . Gradients pass through the adjusted forecast. In Stage 1, the reference is the original valid forecast. We denote this stage-specific reference by the effective validforecast $\widehat { Y } ^ { v , \mathrm { e f f } }$

## 2.4 Horizon-wise gating

CRC controls the predicted residual through a learnable coefficient for each prediction step:

$$
\lambda _ { h } = \lambda _ { \mathrm { m a x } } \mathrm { s i g m o i d } ( a _ { h } ) , \qquad \lambda _ { \mathrm { m a x } } = 0 . 2 0 , \quad h = 1 , \ldots , H .\tag{7}
$$

The logits are initialized at $a _ { h } = - 6$ and shared across samples and channels. The final forecast is

$$
\widehat { Y } _ { i , h , c } ^ { \mathrm { f i n a l } } = \widehat { Y } _ { i , h , c } ^ { v , \mathrm { e f f } } + \lambda _ { h } \widehat { R } _ { i , h , c } .\tag{8}
$$

The gated residual is bounded by [−0.20, 0.20] relative to the effective valid forecast in standardized space. Sample-dependent corrections come from the residual refiner. The refiner has 19,073 parameters; gating and forecast adjustment add 3H, giving 19.4K–21.2K trainable parameters in total.

## 2.5 Training objective

FARR is trained with forecast and residual losses and two residual regularizers:

$$
\begin{array} { r l } & { \mathcal { L } = \mathrm { S m o o t h L 1 } ( \widehat { Y } ^ { \mathrm { f i n a l } } , Y ) + \alpha _ { r } \mathrm { S m o o t h L 1 } ( \widehat { R } , Y - \widehat { Y } ^ { v , \mathrm { e f f } } ) } \\ & { \qquad + \beta _ { s } \mathcal { L } _ { \mathrm { s m o o t h } } + \beta _ { \ell 2 } \mathcal { L } _ { \ell 2 } . } \end{array}\tag{9}
$$

We use SmoothL1 (Huber loss [29] with unit transition), $\alpha _ { r } ~ = ~ 0 . 1 0 .$ , and $\beta _ { s } ~ = ~ \beta _ { \ell 2 } ~ = ~ 0 . 0 1$ . For B single-channel windows ${ j = ( i , c ) }$

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { \sum _ { j , h } ( \widehat { R } _ { j , h + 1 } - \widehat { R } _ { j , h } ) ^ { 2 } } { B ( H - 1 ) } , \qquad \mathcal { L } _ { \ell 2 } = \frac { \sum _ { j , h } \widehat { R } _ { j , h } ^ { 2 } } { B H } .\tag{10}
$$

The smoothness sum uses $h < H$ . Both regularizers act before gating. Stage 1 trains the refiner and gate. Stage 2 starts only if Stage 1 improves both validation MSE and MAE over the valid forecast, and jointly trains the forecast adjustment. Each checkpoint update requires both metrics to strictly improve. Stage-2 candidates must also improve both metrics over the valid forecast; otherwise Stage 1 is retained, with the valid forecast as fallback. All selection uses validation data; the frozen candidate is evaluated on the test split using Eq. (8).

## 3 Experiments

Experimental setup. We evaluate FARR on ETTh1, Weather, Traffic, and Electricity (ECL), using input length L = 672 and prediction horizons $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . A StandardScaler is fitted on the training split and applied unchanged to validation and test data. All locally evaluated MSE and MAE, including the local valid anchor and FARR, are computed in standardized space. Dataset averages weight the four horizons equally. Endpoint forecasts use a frozen local checkpoint based on Qwen2.5-0.5B-Instruct, with label length 576 and token length 96. Table 1 lists the forecasting baselines.

Table 1. Long-horizon forecasting results (MSE/MAE). Red/blue mark the lowest/second-lowest values. Avg. averages the four displayed horizons; – denotes unreported results. ∗ denotes our reproduced results
<table><tr><td rowspan=1 colspan=2>Dataset H</td><td rowspan=1 colspan=9>FARR    SE-LLM TimeMixer++Time-MoE   TQNet(Ours)     [24]       [25]       [17]       [26]</td><td rowspan=1 colspan=6>TimeCMA  LLM-TS  AutoTimes[20]      [27]       [4]</td><td rowspan=1 colspan=1>Time-LLM[1]</td></tr><tr><td rowspan=1 colspan=2>ETTh1 96</td><td rowspan=1 colspan=4>0.353/0.3930.352/0.393</td><td rowspan=3 colspan=11>0.361/0.4030.349/0.3790.371/0.3930.373/0.3910.403/0.4200.360/0.3970.416/0.4410.395/0.4130.428/0.4260.427/0.4210.440/0.4410.391/0.4190.430/0.4340.447/0.4530.476/0.4460.458/0.4480.471/0.4570.408/0.4320.467/0.4510.457/0.4620.487/0.4700.449/0.4600.503/0.4870.429/0.4520.419/0.4320.412/0.4270.441/0.4340.427/0.4300.454/0.4510.397/0.425</td><td rowspan=3 colspan=1>0.362/0.3920.398/0.4180.430/0.4270.442/0.4570.408/0.424</td></tr><tr><td rowspan=2 colspan=2>19233607200Avg.</td><td rowspan=1 colspan=3>0.379/0.4130.385/0.4130</td><td rowspan=2 colspan=1>.378/0.411.391/0.420.402/0.437.381/0.415</td><td rowspan=2 colspan=3></td></tr><tr><td rowspan=1 colspan=3>.392/0.4290.0.377/0.4120</td></tr><tr><td rowspan=1 colspan=2>Weather 96</td><td rowspan=1 colspan=3>0.141/0.1900</td><td rowspan=1 colspan=1>.150/0.200</td><td rowspan=5 colspan=3>0.155/0.2050.201/0.245.248/0.285*0.316/0.332*.230/0.267*</td><td rowspan=5 colspan=1>0.157/0.21100.255/0.29000.405/0.3970</td><td rowspan=5 colspan=1>.157/0.2000.262/0.2870.344/0.3420</td><td rowspan=5 colspan=2>.167/0.2110.270/0.2920.350/0.3480</td><td rowspan=1 colspan=3>.166/0.2170</td><td rowspan=5 colspan=1>.158/0.208.262/0.298.342/0.3530</td><td rowspan=5 colspan=1>0.147/0.2010.189/0.2340.262/0.279.318/0.338*</td></tr><tr><td rowspan=4 colspan=2>336720Avg.</td><td rowspan=1 colspan=1>192</td><td rowspan=1 colspan=2>0.185/0.2340</td><td rowspan=1 colspan=1>.194/0.243</td><td rowspan=3 colspan=3>.278/0.3020.354/0.3510</td></tr><tr><td rowspan=1 colspan=3>0.242/0.2770</td><td rowspan=2 colspan=1>.247/0.28509</td></tr><tr><td rowspan=2 colspan=3>0.221/0.2580</td></tr><tr><td rowspan=1 colspan=1>.229/0.2670</td><td rowspan=1 colspan=1>0.276</td><td rowspan=1 colspan=3>0.257/0.285</td></tr><tr><td rowspan=6 colspan=2>Traffic  96</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=5 colspan=3>0.344/0.2400</td><td rowspan=5 colspan=1>.350/0.243</td><td rowspan=5 colspan=3>0.392/0.253</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=4 colspan=1>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>.413/0.261</td><td rowspan=3 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=2></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=3>0.587/0.31500.612/0.3260</td><td rowspan=2 colspan=1>.369/0.257.394/0.2680</td><td rowspan=7 colspan=1>0.362/0.248.390/0.288*.407/0.299*0.430/0.288.397/0.281*</td></tr><tr><td rowspan=1 colspan=2>1920</td><td rowspan=4 colspan=3>.367/0.2520</td><td rowspan=1 colspan=1>.373/0.254</td><td rowspan=1 colspan=2>0.402/0.258</td><td></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.432/0.271</td></tr><tr><td rowspan=4 colspan=2>332</td><td rowspan=4 colspan=2>0.385/0</td><td rowspan=3 colspan=1>0.390/0.263</td><td rowspan=3 colspan=3></td><td rowspan=4 colspan=2>0.428/0.2630.441/0.282</td><td rowspan=3 colspan=1>0</td><td rowspan=3 colspan=2>.450/0.277</td><td rowspan=3 colspan=3></td><td rowspan=3 colspan=1>0.634/0.3</td></tr><tr><td rowspan=2 colspan=2>4/0.338</td></tr><tr><td rowspan=1 colspan=1>380</td><td rowspan=1 colspan=1>.413/0.2780</td></tr><tr><td rowspan=2 colspan=2>Avg.</td><td rowspan=1 colspan=2>0/0.279</td><td rowspan=1 colspan=1>0.429/0.285</td><td rowspan=1 colspan=2>0.441/0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.486/0.295</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>0.640/0.3510</td><td rowspan=1 colspan=1>.449/0.299</td></tr><tr><td rowspan=1 colspan=3>.0.379/0.2580</td><td rowspan=1 colspan=1>.386/0.261</td><td rowspan=1 colspan=3>0.416/0.264</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.445/0.276</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>0.618/0.3330</td><td rowspan=1 colspan=1>.406/0.2760</td></tr><tr><td rowspan=4 colspan=2>ECL96</td><td rowspan=5 colspan=3>1920</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=3>0.124/0.218</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=3>.136/0.2330</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.130/0.225</td><td rowspan=1 colspan=3>0.135/0.222</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>0.134/0.22900.154/0.2470</td><td rowspan=2 colspan=2>.143/0.2380.161/0.2590</td><td rowspan=2 colspan=3>.167/0.2710.178/0.2800</td><td rowspan=2 colspan=1>.140/0.236.159/0.253</td><td rowspan=2 colspan=1>0.131/0.2240.152/0.241</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>.148/0.242</td><td rowspan=1 colspan=3>0.147/0.235</td></tr><tr><td rowspan=2 colspan=2>3360.</td><td rowspan=2 colspan=3>155/0.2510</td><td rowspan=1 colspan=1>.164/0.259</td><td rowspan=1 colspan=3>0.164/0.245</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.169/0.2640</td><td rowspan=1 colspan=2>.169/0.2610</td><td rowspan=2 colspan=3>.198/0.3020.233/0.3440</td><td rowspan=2 colspan=1>.177/0.270.216/0.303</td><td rowspan=3 colspan=1>0.160/0.2480.192/0.2980.159/0.253</td></tr><tr><td rowspan=2 colspan=2>Avg.0</td><td rowspan=1 colspan=1>7200.193/0.2840.203/0.293</td><td rowspan=1 colspan=3>0.212/0.310</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.201/0.2940</td><td rowspan=1 colspan=2>.219/0.3150</td></tr><tr><td rowspan=1 colspan=3>.152/0.2470</td><td rowspan=1 colspan=1>.161/0.255</td><td rowspan=1 colspan=3>0.165/0.253</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.165/0.2590</td><td rowspan=1 colspan=2>.173/0.2680</td><td rowspan=1 colspan=3>.194/0.2990</td><td rowspan=1 colspan=1>.173/0.266</td></tr></table>

## 3.1 Long-horizon forecasting results

Table 1 compares FARR with the forecasting baselines. Its dataset-average MSE/MAE is 0.3773/0.4120 on ETTh1, 0.2208/0.2578 on Weather, 0.3790/0.2578 on Traffic, and 0.1520/0.2465 on ECL. FARR achieves the lowest dataset-average MSE and MAE among the listed methods on all four datasets. At individual horizons, Time-MoE has lower ETTh1 error at H = 96, and SE-LLM at H = 192. Time-LLM has lower ECL MSE at H = 720. FARR leads on dataset averages while individual-horizon rankings vary.

## 3.2 Component ablations

Table 2 tests the input information and residual refinement mechanism. The local valid anchor is the uncorrected forecast from the evaluated checkpoint; SE-LLM in Table 1 is a separately reported published baseline. Adjustment only applies Eq. (6) without residual refinement. Matched MLP is a parameter-matched dual-endpoint MLP without explicit disagreement features, horizon-wise gating, or Stage 2. History only uses historical features and the horizon coordinate; single endpoint adds the valid forecast; disagreement only uses the disagreement features. Dual endpoint (Stage 1) retains all features without forecast adjustment.

Table 2. Component ablations in standardized space, averaged over four horizons. Best values are bold.
<table><tr><td></td><td colspan="2">ETTh1</td><td colspan="2">Weather</td><td colspan="2">Traffic</td><td colspan="2">ECL</td></tr><tr><td>Variant</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Local valid anchor</td><td>0.3900</td><td>0.4200</td><td>0.2300</td><td>0.2651</td><td>0.3865</td><td>0.2627</td><td>0.1605</td><td>0.2545</td></tr><tr><td>Adjustment only</td><td>0.3810</td><td>0.4151</td><td>0.2280</td><td>0.2661</td><td>0.3848</td><td>0.2609</td><td>0.1609</td><td>0.2540</td></tr><tr><td>Matched MLP</td><td>0.3796</td><td>0.4142</td><td>0.2258</td><td>0.2622</td><td>0.3835</td><td>0.2600</td><td>0.1584</td><td>0.2518</td></tr><tr><td>History only</td><td>0.3834</td><td>0.4170</td><td>0.2286</td><td>0.2668</td><td>0.3881</td><td>0.2635</td><td>0.1615</td><td>0.2549</td></tr><tr><td>Single endpoint</td><td>0.3818</td><td>0.4158</td><td>0.2270</td><td>0.2645</td><td>0.3856</td><td>0.2614</td><td>0.1602</td><td>0.2533</td></tr><tr><td>Disagreement only</td><td>0.3828</td><td>0.4164</td><td>0.2284</td><td>0.2659</td><td>0.3841</td><td>0.2607</td><td>0.1611</td><td>0.2545</td></tr><tr><td>Dual endpoint (Stage 1)</td><td>0.3770</td><td>0.4123</td><td>0.2230</td><td>0.2577</td><td>0.3821</td><td>0.2590</td><td>0.1562</td><td>0.2497</td></tr><tr><td>Full FARR</td><td>0.3773</td><td>0.4120</td><td>0.2208</td><td>0.2578</td><td></td><td>0.37900.2578</td><td></td><td>0.1520 0.2465</td></tr></table>

Dual-endpoint refinement improves both metrics over the single-endpoint and disagreement-only variants on every dataset. On Weather, its MSE is 0.2230, compared with 0.2270 for single-endpoint refinement. Full FARR also outperforms the parameter-matched MLP in all eight dataset–metric comparisons. Against forecast adjustment alone, it reduces ECL MSE from 0.1609 to 0.1520. Stage 2 improves six of eight comparisons over Stage 1; ETTh1 MSE increases by 0.0003 and Weather MAE by 0.0001. The combined features improve refinement, while the additional gain from forecast adjustment varies across datasets.

## 3.3 Implementation details

Table 3 summarizes the settings. We use AdamW [30] with complete single-channel windows as batch samples. Stage 2 initializes from Stage 1 and jointly trains the refiner, horizon-wise gate, and forecast adjustment. Residual regularizers act before gating, whereas weight decay acts on parameters. Both endpoint passes reset seed 2021 and preserve sample order under deterministic inference. The final Stage-2 candidate is selected by joint validation improvement (Section 2.5) before test evaluation.

Table 3. Model hyperparameters and training settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Input / label / token length</td><td>672 / 576 / 96</td></tr><tr><td>Prediction horizon H</td><td>96, 192, 336, 720</td></tr><tr><td>History / recurrence window</td><td>168 /24</td></tr><tr><td>Refiner architecture</td><td> $1 8 \to 1 2 8 \to 1 2 8 \to 1$ </td></tr><tr><td>Residual output</td><td>tanh; [−1, 1]</td></tr><tr><td>Gate cap / initial logit</td><td>0.20 / -6</td></tr><tr><td>SmoothL1 transition  $/ \alpha _ { r }$ </td><td>1.0 /0.10</td></tr><tr><td>Smoothness / residual L2 weights</td><td>0.01 / 0.01</td></tr><tr><td>Adjustment scale / offset bounds</td><td>0.10 / 0.10</td></tr><tr><td>Batch size (train and evaluation)</td><td>8192</td></tr><tr><td>Weight decay / gradient clipping</td><td> $1 0 ^ { - 4 } / 1 . 0$ </td></tr><tr><td>Stage 1/2 learning rates</td><td> $1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>10/5; 2/2</td></tr><tr><td>Stage 1/2 epochs; patience</td><td></td></tr></table>

## 3.4 Hyperparameter sensitivity

We vary one factor at a time around $\lambda _ { \operatorname* { m a x } } = 0 . 2 0 , \alpha _ { r } = 0 . 1 0 .$ , and $\beta _ { s } = \beta _ { \ell 2 } = 0 . 0 1$ . Alternatives are $\{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 4 0 \}$ for $\lambda _ { \operatorname* { m a x } } , \{ 0 , 0 . 0 5 , 0 . 2 0 \}$ for $\alpha _ { r } ,$ , and $\{ 0 , 0 . 0 0 1 , 0 . 0 5 \}$ for each regularizer. Architecture, seed, optimizer, and training budget are fixed. Sensitivity runs use training and validation data only. Their checkpoint rule requires a strict MSE decrease and permits at most a 0.2% MAE increase, unlike the joint strict improvement used for the main experiments.

In Figure 3, disabling residual refinement or increasing regularization produces larger MSE changes. Shading shows variation across the included forecasting settings. The zero-change reference is defined by normalization.

(a) Gate cap λmax  
![](images/d8c7b0050ac05465521121dd08b695543e127f2ae4670963decc24ce71629b77.jpg)

(b) Residual loss $\alpha _ { r }$  
![](images/b6a18fefd93883fcc09af303ac4562bed89591d9c8a79ec4cac463689fc8c3ff.jpg)

(c) Smoothness $\beta _ { s }$  
![](images/88550f3c7b51ac9ab5ddafc6e6ab7b318849dbd26aa64fa11f992168bc844fe6.jpg)

(d) Residual L2 $\beta _ { \ell _ { 2 } }$  
![](images/4c2a6146efb147f7ccc9f4ba65284309a4f79c70f5a0acc2ea51fd882331abb5.jpg)  
Fig. 3. Hyperparameter sensitivity. Lines and shading show mean relative MSE changes and the 25th–75th percentile range across included forecasting settings. Stars mark the normalization reference.

## 3.5 Inference cost

Table 4 reports the ETTh1 latency breakdown. Additional endpoint generation and subsequent correction add 21.01 ms and 1.08 ms, respectively, based on differences between execution-mode means. Full FARR takes 85.12 ms, 35.0% more than the single-endpoint backbone. The standardized MSE reduction is $0 . 3 9 0 0 - 0 . 3 7 7 3 = 0 . 0 1 2 7$ , or 3.3% relative to the local valid anchor, using the reported averages in Table 2.

Table 4. ETTh1 mean inference latency (ms) and standardized MSE reduction relative to the local valid anchor. Latency increments are differences between execution-mode means.
<table><tr><td>Dataset</td><td>Single backbone</td><td>Extra endpoint</td><td>Extra correction</td><td>Full FARR</td><td>MSE reduction</td></tr><tr><td>ETTh1</td><td>63.03</td><td>21.01</td><td>1.08</td><td>85.12</td><td>0.0127</td></tr></table>

The additional endpoint accounts for 95.1% of the 22.09 ms latency increase; subsequent correction contributes 4.9%. The 19.4K–21.2K trainable parameters describe the refinement module, while deployment still requires two backbone passes.

## 4 Conclusion

FARR addresses the variable effects of auxiliary conditioning by learning residual corrections from matched valid and null forecasts and historical context. A lightweight refiner and bounded horizon-wise gate improve predictions while keeping the backbone frozen. Across four benchmarks, FARR achieves the lowest datasetaverage MSE and MAE among the compared methods. Generating the additional endpoint incurs inference overhead. Future work will explore selective endpoint generation and distillation to retain the benefits of endpoint evidence with lower inference cost.

## 5 Compliance with ethical standards

This is a numerical simulation study for which no ethical approval was required.

Acknowledgments. The authors have no relevant financial or nonfinancial interests to disclose.

## 6 References

[1] M. Jin et al., “Time-LLM: Time series forecasting by reprogramming large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024, vol. 2024, pp. 23857–23880.

[2] Z. Pan, Y. Jiang, S. Garg, A. Schneider, Y. Nevmyvaka, and D. Song, “S<sup>2</sup>IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting,” in Proc. Int. Conf. Mach. Learn. (ICML), 2024, vol. 235, pp. 39135–39153.

[3] F. Bellos, N. Premkumar, Y. Avrithis, N. H. Nguyen, and J. J. Corso, “DEEP TPC: Temporal-prior conditioning for time series forecasting,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026, pp. 1331–1335.

[4] Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, “AutoTimes: Autoregressive time series forecasters via large language models,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2024, vol. 37, pp. 122154–122184.

[5] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?,” in Proc. AAAI Conf. Artif. Intell., 2023, vol. 37, pp. 11121–11128.

[6] H. Zhou et al., “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proc. AAAI Conf. Artif. Intell., 2021, vol. 35, pp. 11106–11115.

[7] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, vol. 34, pp. 22419–22430.

[8] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in Proc. Int. Conf. Mach. Learn. (ICML), 2022, vol. 162, pp. 27268–27286.

[9] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[10] Y. Liu et al., “iTransformer: Inverted transformers are effective for time series forecasting,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024, vol. 2024, pp. 11116–11140.

[11] S. Wang et al., “TimeMixer: Decomposable multiscale mixing for time series forecasting,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024, vol. 2024, pp. 38626–38652.

[12] S. Lin, W. Lin, X. Hu, W. Wu, R. Mo, and H. Zhong, “CycleNet: Enhancing time series forecasting through modeling periodic patterns,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2024, vol. 37, pp. 106315–106345.

[13] F. Cao, S. Yang, Z. Chen, Y. Liu, and L. Cui, “Ister: Linear transformer for efficient multivariate time series forecasting,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026, pp. 3571–3575.

[14] E. Dayag, N. T. Tran, and J. Xin, “Filter then attend: Improving attention-based time series forecasting with spectral filtering,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026, pp. 1131–1135.

[15] A. F. Ansari et al., “Chronos: Learning the language of time series,” Trans. Mach. Learn. Res., 2024.

[16] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in Proc. Int. Conf. Mach. Learn. (ICML), 2024, vol. 235, pp. 10148–10167.

[17] X. Shi et al., “Time-MoE: Billion-scale time series foundation models with mixture of experts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025, vol. 2025, pp. 34635–34667.

[18] Y. Li, K. Ding, C. Yang, S.-Y. Chen, and Y. Tian, “Distilling time series foundation models for efficient forecasting,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026, pp. 4631–4635.

[19] P. Liu et al., “CALF: Aligning LLMs for time series forecasting via cross-modal fine-tuning,” in Proc. AAAI Conf. Artif. Intell., 2025, vol. 39, pp. 18915–18923.

[20] C. Liu et al., “TimeCMA: Towards LLM-empowered multivariate time series forecasting via cross-modality alignment,” in Proc. AAAI Conf. Artif. Intell., 2025, vol. 39, pp. 18780–18788.

[21] S. Han, S. Lee, M. Cha, S. O. Arik, and J. Yoon, “Retrieval augmented time series forecasting,” in Proc. Int. Conf. Mach. Learn. (ICML), 2025, vol. 267, pp. 21774–21797.

[22] M. Tan, M. A. Merrill, V. Gupta, T. Althoff, and T. Hartvigsen, “Are language models actually useful for time series forecasting?,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2024, vol. 37, pp. 60162–60191.

[23] A. Bolton, W. Zhou, Z. Chen, G. Iacovides, and D. Mandic, “RefineBridge: Generative bridge models improve financial forecasting by foundation models,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026, pp. 1796–1800.

[24] H. Liu, X. Zhang, C. Yang, and X. Zhu, “Semantic-enhanced time-series forecasting via large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026, vol. 2026, pp. 101404–101425.

[25] S. Wang et al., “TimeMixer++: A general time series pattern machine for universal predictive analysis,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025, vol. 2025, pp. 16980–17016.

[26] S. Lin, H. Chen, H. Wu, C. Qiu, and W. Lin, “Temporal query network for efficient multivariate time series forecasting,” in Proc. Int. Conf. Mach. Learn. (ICML), 2025, vol. 267, pp. 37797–37814.

[27] C. Chen, G. L. Oliveira, H. Sharifi-Noghabi, and T. Sylvain, “LLM-TS Integrator: Integrating LLM for enhanced time series modeling,” Trans. Mach. Learn. Res., 2025.

[28] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, “N-BEATS: Neural basis expansion analysis for interpretable time series forecasting,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[29] P. J. Huber, “Robust estimation of a location parameter,” Ann. Math. Statist., vol. 35, no. 1, pp. 73–101, 1964.

[30] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.