# ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events

Huajian Zhang<sup>1</sup>, Mingyue Cheng<sup>1∗</sup>, Xingpeng Gao<sup>1</sup>, Xiaoyu Tao<sup>1</sup>, Zhiding Liu<sup>1</sup>, Jiqian Yang<sup>2</sup>, Zhi Li<sup>2</sup>

<sup>1</sup>State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China

Hefei, China

<sup>2</sup>Shenzhen International Graduate School, Tsinghua University

Shenzhen, China

zhjustc, gaoxingpeng, txytiny, zhiding, yangjq@mail.ustc.edu.cn, mycheng@ustc.edu.cn, zhilizl@sz.tsinghua.edu.cn

## Abstract

Accurate electricity load forecasting provides critical decision support for energy procurement, generation scheduling, and supplydemand balancing, contributing to the economic eficiency and operational reliability of power systems. However, future electricity demand can be afected by the complex interplay of multiple events described in Web notices, weather alerts, and event schedules. Their efects vary over time and can reinforce or ofset one another, mak ing electricity load dificult to forecast accurately. While existing methods can incorporate context as additional input, how to use the efects of multiple events and their interactions to guide the load forecasting process itself remains to be explored. To address this gap, we propose ElecLoad, a context-aware agent for multi-event electricity load forecasting that adaptively organizes the forecasting process. Specifically, we train ElecLoad to combine historical load with context to analyze each event’s relevance to the target load, the timing of its efects, and potential interactions among events. Guided by this analysis, ElecLoad adaptively chooses to forecast the entire forecast horizon directly or to partition it into segments according to changes in event efects. When the horizon is partitioned, sub-agents independently generate initial forecasts for each segment. The main agent then reviews these forecasts, and the sub-agents perform selective refinement guided by its feedback. Experiments on multiple electricity load datasets show that ElecLoad achieves state-of-the-art performance in most evaluation scenarios. Code is available at https://github.com/zhjai/ElecLoad.

## Keywords

Electricity load forecasting, multi-event interactions, context-aware agents

## ACM Reference Format:

Huajian Zhang<sup>1</sup>, Mingyue Cheng<sup>1∗</sup>, Xingpeng Gao<sup>1</sup>, Xiaoyu Tao<sup>1</sup>, Zhiding Liu<sup>1</sup>, Jiqian Yang<sup>2</sup>, Zhi Li<sup>2</sup> . 2018. ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 13 pages. https://doi.org/ XXXXXXX.XXXXXXX

![](images/7443f7358b57719e541d71486bc5abd18de6ae6fcc906f5422d068155a9d6e82.jpg)  
Figure 1: Conceptual illustration of (a) load changes under multiple events over time and (b) event-guided forecast partitioning in ElecLoad.

## 1 Introduction

To meet residential and industrial electricity demand, power systems need to plan electricity procurement, generation scheduling, and reserve allocation in advance to maintain the balance between electricity supply and demand. These decisions depend on anticipating future electricity demand, and accurate electricity load forecasting can provide valuable support. Forecasting errors can have economic and operational consequences: overestimating demand may lead to unnecessary electricity procurement and resource commitments, whereas underestimating demand may increase the costs of unplanned adjustments and risks to system operation [42]. Moreover, efective power system operation requires not only accurately forecasting future demand levels but also capturing the timing and magnitude of load peaks [32]. Therefore, accurately forecasting future electricity load sequences is important for improving the economic eficiency and operational reliability of power systems. Reliable electricity supply also underpins the data centers on which Web services depend [17].

However, extreme weather, major sporting events, and events related to power system operation can cause electricity load to deviate from regular consumption patterns. Weather alerts, news reports, and operational notices on the Web can provide clues about event details, including relevant times and regions [23]. Using these Web event descriptions for forecasting requires assessing each event’s relevance to the target load, the timing of its efects within the forecast horizon, and potential interactions among events. For example, during a heatwave in the United Kingdom on June 23, 2026, many viewers watched England’s World Cup match with air conditioners and electric fans running. A report on The Guardian’s website cited estimates of additional electricity demand at halftime and after the final whistle [1]. When event efects overlap or occur in sequence, they may reinforce or ofset one another across diferent periods. Load may still be recovering from one event when another begins to afect consumption. At such times, load reflects the remaining efects of the earlier event alongside those of the later one. As illustrated in Figure 1(a), these changing influences can alter load levels and the timing and magnitude of load peaks. Analysis of changing event efects and load recovery must inform hourly numerical load forecasts and guide decisions about which periods to forecast together and which forecasts need refinement.

General forecasting methods already incorporate covariates or textual information [2, 14], and some introduce process mechanisms such as event alignment, planning, and reflection [14, 26, 39]. In electricity load forecasting, MMGPT4LF [13] aligns and fuses textual and time-series representations. ITA-LF [40] analyzes news relevance, efect direction, and duration before selecting forecasting inputs. NACF [6] represents news semantics and publication times together with historical load and covariates, and examines forecasts under altered event conditions. A remaining challenge is to use analysis of how the efects of multiple events change over time to guide an adaptive forecasting process.

To address this challenge, we propose ElecLoad, a context-aware agent for multi-event electricity load forecasting. ElecLoad combines historical load with context to analyze each event’s relevance to the target load, the timing of its efects, and potential interactions among events. Based on this analysis, the main agent adaptively chooses between forecasting the full horizon directly and partitioning it into multiple segments. The partition follows changes in event efects over time, as shown in Figure 1(b). For segmented forecasting, ElecLoad adopts a multi-agent collaboration framework in which the main agent organizes the process and shares the same language model with the sub-agents. Each sub-agent independently generates an initial forecast using the complete original historical load and context, together with the main agent’s analysis of its assigned segment. To address mismatches between predicted load changes and event efects, the main agent checks forecasts against the original inputs and event analysis, selecting segments for feedback and refinement. The corresponding sub-agent returns a refined forecast, and the main agent reassesses the updated sequence to decide whether to continue or end this selective refinement loop. Finally, boundary reconciliation retains or adjusts values near segment boundaries. Our main contributions are summarized as follows:

• We analyze the forecasting requirements arising from overlapping events, sequential event efects, and load recovery, and propose using changes in event efects over time to guide the forecasting process.

• ElecLoad combines historical load and context to analyze each event’s relevance to the target load, the timing of its efects, and potential interactions among events. This analy sis informs numerical load predictions for diferent periods within the forecast horizon.

• We evaluate ElecLoad against forecasting baselines on indomain electricity load datasets and out-of-domain electricity systems, and analyze its performance under diferent event conditions.

## 2 Related Work

## 2.1 Neural Time Series Forecasting

One major line of research in neural time series forecasting improves how models represent the structure of a series [7]. PatchTST [25] encodes local subsequences as patches for Transformer input, preserving local information while supporting longer input histories. ConvTimeNet [9] uses deformable patches and hierarchical convolutions to capture local patterns and their multiscale dependencies. In representation learning, TimeDART [38] combines a causal Transformer with difusion denoising in selfsupervised pretraining to learn global and local representations for downstream forecasting. To account for diferences in periodicity and trends across application domains, OneCast [27] decomposes a series into seasonal and trend components. It models these components separately and combines the resulting predictions to produce the final forecast. Beyond the structure of the series itself, external conditions can also inform forecasting: Chronos-2 [2] and TimesFM-3 [18] both support zero-shot forecasting using historical time series and covariates. EventTSF [14] aligns event embeddings with fixed-length segments and autoregressively predicts subsequent segments from preceding hidden states and event information.

## 2.2 LLM-Based Forecasting

Forecasting with LLMs connects contextual understanding and reasoning to numerical prediction. TokenCast discretizes time series into tokens, aligns them with context tokens, and decodes generated future tokens into numerical values [35]. TimeReasoner performs multistep reasoning, using earlier outputs as context for subsequent forecasts in its rollout [8]. MemCast builds hierarchical memory from training-set prediction results, reasoning trajectories, and temporal features to guide reasoning, trajectory selection, and reflective iteration during inference [33]. For evidence gathering and feedback, AlphaCast uses investigation, generation, and reflection roles to construct context, generate forecasts, and refine them [44]. Subsequently, CastFlow assigns planning and reflection to a frozen language model, while a fine-tuned model produces numerical forecasts from tool evidence and an ensemble forecast baseline [26]. CastFSR combines initial forecasts with contextual reasoning, adaptively retrieves evidence and selects a lookback window, and uses reflection to correct local inconsistencies [34]. MoiraiAgent ofers contextual forecasting and expert selection as independent, composable capabilities. The former adjusts historical windows and forecasts using context, while the latter selects suitable forecasting models [43].

## 2.3 Electricity Load Forecasting

Electricity load forecasting research explores how to incorporate event information. Gaussian-process transfer learning calibrates a prior forecast from a few observations during an anomalous event [45]. News and Load uses news features for day-ahead forecasting [4]; its follow-up combines news with economic indicators for regional load forecasting across multiple horizons [5]. MMGPT4LF encodes textual descriptions of calendar information, temperature, and historical load trends with GPT-2 and fuses them with load sequences through cross-attention [13]. ITA-LF uses an

LLM agent to analyze the relevance of news to load and the direction and duration of its impact, and to reflect on news selection. A fine-tuned LLM then forecasts from selected news and historical load [40]. NACF uses an event-conditioned network to produce factual forecasts from historical load, covariates, and the semantics and publication times of LLM-structured news events [6]. It also produces counterfactual forecasts under a no-news baseline or hypothetical events. AutoLFM uses LLM agents to automate the development of building load predictors and corrects generated code with validation feedback [22]. ElecLoad analyzes how the efects of multiple events on the target load vary over time and, guided by this analysis, adaptively chooses to forecast the full horizon directly or partition the forecast horizon into segments. On the segmented path, the event analysis and forecast checks determine which segments undergo selective refinement.

## 3 Problem Formulation

We study multi-event electricity load forecasting, covering both power system load and household electricity load. Let $t _ { 0 }$ denote the forecast origin and Δ� the sampling interval. The load at time $t _ { 0 } + \tau \Delta t$ is denoted by $y _ { \tau }$ , with $\tau = 0$ marking the first time step to forecast. Given a historical load sequence of length $L , \mathbf { y } _ { - L : - 1 } =$ $\left( y _ { - L } , \dotsc , y _ { - 1 } \right)$ , the goal is to forecast the load sequence ${ \bf y } _ { 0 : H - 1 } =$ $\left( y _ { 0 } , \dots , y _ { H - 1 } \right)$ over the next � steps.

The context $C \ = \ ( \boldsymbol { \chi } , \boldsymbol { \mathcal { E } } )$ consists of auxiliary inputs beyond the historical load sequence, with covariates X including numerical, categorical, and calendar features. Each event in the event set $\mathcal { E } = \{ e _ { m } \} _ { m = 1 } ^ { M }$ is represented by a textual description and any temporal and geographic information supplied by its source. An announcement’s publication time indicates when the information is released. The occurrence or efective period stated by the source provides cues for relating the event to the forecast horizon. Neither the publication time nor the stated period is equivalent to the time range over which the load is afected. The data protocol specifies the conditions under which future covariates are provided. In the experiments reported here, event announcements are available at the corresponding forecast origins.

A predictor � maps the historical load sequence and the available context to a numerical load forecast:

$$
\widehat { \mathbf { y } } _ { 0 : H - 1 } = f ( \mathbf { y } _ { - L : - 1 } , C ) ,\tag{1}
$$

where $\widehat { \mathbf { y } } _ { 0 : H - 1 } \in \mathbb { R } ^ { H }$ is the predicted load sequence covering the entire forecast horizon.

## 4 Method

Figure 2 and Algorithm 1 show how ElecLoad uses multi-event analysis to guide adaptive forecasting. Its agents call four functions: event analysis and forecast planning (T1), numerical forecasting (T2), forecast checks with selective feedback and stopping decisions (T3), and boundary reconciliation (T4). In T1, the main agent analyzes how event efects vary over time and adaptively chooses between direct forecasting and partitioning the forecast horizon into segments. The direct path ends after T2 generates the forecast from the original inputs and T1’s analysis. On the segmented path, sub-agents independently generate initial segment forecasts in T2.

Algorithm 1: ElecLoad inference workflow   
Input: $\mathbf { x } = ( \mathbf { y } _ { - L : - 1 } , C ) .$ , horizon �, refinement budget $R _ { \mathrm { m a x } }$   
Output: Complete load forecast yˆ $\in \mathbb { R } ^ { H }$   
// T1: event-guided planning   
1 (A,�, S) ← T1(x, �)   
2 if � = DIRECT then   
3 return T2(x, A, [0, � ))   
// T2: independent forecasting   
4 foreach $S _ { j } \in S$ do   
5 $\underline { { \hat { \mathbf { \Pi } } } } \hat { \mathbf { y } } _ { S _ { j } } \gets \top 2 ( \mathbf { x } , \mathcal { A } , S _ { j } )$   
6 $\hat { \mathbf { y } } \gets \mathsf { C o n c a t } ( \hat { \mathbf { y } } _ { S _ { 1 } } , \hdots , \hat { \mathbf { y } } _ { S _ { K } } )$   
// T3: selective refinement   
7 for $r = 0 , \ldots , R _ { \mathrm { m a x } }$ do   
8 $( a , j , f ) \gets \top 3 ( \mathbf { x } , \mathcal { A } , S , \hat { \mathbf { y } } , R _ { \operatorname* { m a x } } - r )$   
9 i $\dot { \mathbf { \xi } } a = 5 7 0 \mathsf { P } \ o r r = R _ { \operatorname* { m a x } }$ then   
10 break   
11 $\hat { \mathbf { y } } [ S _ { j } ] \gets \top 2 ( \mathbf { x } , \mathcal { A } , S _ { j } , f )$   
// T4: boundary reconciliation   
12 (�, patch) ← T4(x, A, S, yˆ)   
13 if � = REGENERATE then   
14 yˆ ← ApplyPatch(yˆ, patch)   
15 return yˆ

In T3, the main agent checks them and either stops or provides feedback on a selected segment, which its sub-agent refines in T2. After each refinement, the main agent reassesses the complete forecast in T3 and decides whether to continue within the refinement budget. T4 retains or adjusts values near segment boundaries as needed. All roles share LLM parameters and receive the complete original inputs x on every call. T1’s analysis A comprises impact analysis, overall explanation, and uncertainty assessment. Segment-specific analysis, reports, and role-specific histories are abbreviated in the calls, with details in Sections 4.2 and 4.3.

## 4.1 Event-Guided Forecast Planning

The main agent uses historical load and context to analyze each event’s relevance to the target load, the timing of its efects, and potential interactions among events. Drawing on the activities, regions involved, and timing described in event descriptions, it explains the mechanisms through which an event may alter electricity demand. It infers the direction of each event’s efects and its impact window, defined as the range of forecast steps during which load may be afected. The impact window may include delayed responses or recovery after the event ends. The main agent also examines the order of events and how their efects overlap. For example, it considers whether diferent efects may reinforce or ofset one another, and whether another event may afect load recovery following an earlier event. This impact analysis draws on specific evidence in the inputs and retains uncertainty about the timing of efects and potential interactions.

Based on this impact analysis, the main agent selects one of two forecasting paths: direct forecasting or segmented forecasting. In direct forecasting, T2 generates the load forecast for the entire horizon from the original inputs and T1’s analysis, and the process ends. On the segmented path, it partitions the forecast horizon into segments, inferring their number and boundaries from how event efects change over time. For example, a boundary can mark an efect’s onset or gradual decay, the start of load recovery, or a change in how multiple efects overlap. One event’s efects may span multiple segments, and one segment may contain the efects of multiple events. For a forecast horizon comprising � discrete time steps, the partition is defined as

![](images/33c7001c686378c2f097da3993f43a3910144ebd1ec10e4b9e5452b514b2e7df.jpg)  
Figure 2: ElecLoad uses multi-event analysis to adaptively choose direct or segmented forecasting. In the segmented path, the main agent selects segments for feedback and refinement. The main agent and sub-agents share the same LLM weights.

$$
\begin{array} { r l } & { \boldsymbol { S } = \{ S _ { j } \} _ { j = 1 } ^ { K } , } \\ & { S _ { j } = \{ \tau \in \mathbb { Z } \mid b _ { j - 1 } \leq \tau < b _ { j } \} , } \\ & { 0 = b _ { 0 } < b _ { 1 } < \cdot \cdot \cdot < b _ { K } = H , \quad K \geq 2 , } \end{array}\tag{2}
$$

where the integer boundaries $b _ { j }$ yield chronological segments that cover all forecast steps without gaps or overlaps, within preset limits on segment count and minimum length.

## 4.2 Independent Segment Forecasting

On the segmented path, each sub-agent receives one forecast segment $S _ { j }$ for T2 forecasting. The program selects factors whose impact windows overlap $S _ { j }$ and passes their analysis and potential interactions. The sub-agent receives the complete original historical load and context, along with the main agent’s overall explanation and uncertainty assessment. For its initial forecast, each sub-agent receives no forecasts or reports generated by other sub-agents. This separation is intended to limit error propagation across segments during autoregressive generation.

The sub-agent evaluates the main agent’s analysis against historical load and context, then predicts the load level, curve shape, and timing of changes within its segment. It translates the analyzed persistence, overlap, reinforcement, and ofsetting into numerical values at each forecast step. Full original context supports efects spanning segment boundaries, such as recovery from an earlier event. All segments share the original forecast origin and future time grid, preserving each target step’s lead time.

Each sub-agent returns a complete segment forecast and a report explaining its rationale and uncertainty, with a self-check against the evidence. We denote the complete initial forecast for segment $S _ { j }$ by $\hat { \mathbf { y } } _ { S _ { j } } ^ { ( 0 ) } = ( \hat { y } _ { \tau } ^ { ( 0 ) } ) _ { \tau = b _ { j - 1 } } ^ { b _ { j } - 1 }$ . Concatenating these segment forecasts in chronological order yields the initial forecast sequence covering the entire forecast horizon:

$$
{ \hat { \mathbf { y } } } ^ { ( 0 ) } = \operatorname { C o n c a t } \Bigl ( { \hat { \mathbf { y } } } _ { S _ { 1 } } ^ { ( 0 ) } , \dots , { \hat { \mathbf { y } } } _ { S _ { K } } ^ { ( 0 ) } \Bigr ) = \Bigl ( { \hat { y } } _ { 0 } ^ { ( 0 ) } , \dots , { \hat { y } } _ { H - 1 } ^ { ( 0 ) } \Bigr ) .\tag{3}
$$

## 4.3 Selective Refinement

In T3, the main agent checks the initial forecast sequence $\hat { \mathbf { y } } ^ { ( 0 ) }$ for consistency with the original inputs, impact analysis, and segment reports. It examines whether predicted levels and recovery reflect the reinforcement, ofsetting, and persistence of event efects. Using these checks, the feedback history, and the remaining budget, it selects one segment for refinement or stops. Its diagnostic feedback identifies issues to examine and supporting evidence, optionally highlighting a time interval for closer attention.

In T2, the selected sub-agent reexamines its forecast using the inputs described in Section 4.2, its own previous outputs, and the feedback. It explains whether the feedback warrants changes and returns a complete forecast and updated report for the same segment. The highlighted interval focuses review, while delayed efects or recovery may motivate changes elsewhere within that segment. Let the �-th refinement $( r \geq 1 )$ select segment $S _ { j _ { r } }$ and return the forecast $\tilde { \mathbf { y } } _ { S _ { j _ { r } } } ^ { ( r ) }$ . The forecast sequence is updated as

$$
\hat { y } _ { \tau } ^ { ( r ) } = \left\{ \begin{array} { l l } { \tilde { y } _ { \tau } ^ { ( r ) } , } & { \tau \in S _ { j _ { r } } , } \\ { \hat { y } _ { \tau } ^ { ( r - 1 ) } , } & { \tau \notin S _ { j _ { r } } , } \end{array} \right. \qquad \tau = 0 , \ldots , H - 1 .\tag{4}
$$

The main agent reassesses the complete forecast after each refinement, using all segment results and the refinement history. It stops when no refinement is selected or the budget is exhausted.

Adaptive boundary reconciliation. After refinement, the main agent reconciles forecast values near segment boundaries using the original inputs, planning output, final reports, and refinement history. It checks whether diferences where segments join reflect event onset, decay, or recovery. T4 returns KEEP or a REGENERATE patch that adjusts values only near segment boundaries, keeping the partition fixed.

## 4.4 Model Training

ElecLoad is built on Qwen3.5-4B [29] and is trained in two stages. Reasoning initialization teaches the model basic numerical load forecasting and event identification and understanding. Agent optimization then trains all roles to execute tasks and make decisions within the adaptive forecasting process. Each stage uses supervised fine-tuning (SFT) followed by reinforcement learning (RL), with all model parameters updated in both SFT and RL.

During reasoning initialization, the teacher predicts loads and explains them from historical load and context. It uses training labels to review and retain or rewrite the explanation to support the target answer. The accepted explanation and true load labels supervise the student.

Agent SFT starts from the reasoning RL checkpoint. Demonstrations are generated mainly by the teacher, starting with event analyses and plans. After label-assisted review, accepted plans are executed; demonstrations retain the role explanations, control decisions, and numerical forecasts generated and accepted. Each episode � is a complete forecasting workflow execution. SFT averages valid response-token losses within episodes and across episodes.

Agent RL starts from the agent SFT checkpoint and uses utility

$$
U ( e ) = - \frac { W ( e ) } { 0 . 1 0 } - 0 . 0 1 \frac { C ( e ) } { C _ { \mathrm { r e f } } } ,\tag{5}
$$

where $W ( e )$ is fractional WAPE of the delivered forecast, including fallbacks, and �(�) counts input and output tokens across all calls, including failed attempts. The fixed per-case reference $C _ { \mathrm { r e f } }$ normalizes cost; 0.10 sets the error scale and 0.01 is the cost weight. We sample two independent planning responses per case and continue each accepted plan along both forecasting paths. Paired utilities support path selection. Role-specific credit supplies learning signals for initial forecasting, refinement decisions, and boundary reconcil iation. It distinguishes local forecast quality or improvement, call cost, and output format and structural validity. Appendix B details supervision, credit assignment, and validation scoring.

## 5 Experiments

## 5.1 Experimental Setup

5.1.1 Datasets. We evaluate hourly household, regional, and system load, treating each power system as a domain. In-domain (ID) sources comprise the Global Energy Forecasting Competition 2014 (GEFCom2014) load track [16], national load (Nord) [12], PJM native load and balancing-area demand [28, 37], Southwest Power Pool (SPP) balancing-area demand [37], Low Carbon London (LCL) [36], and iFlex [15]. The Australian Energy Market Operator (AEMO) [3] and Electric Reliability Council of Texas (ERCOT) [11] are out-of domain (OOD) systems reserved for 96-hour multi-day and 24-hour day-ahead testing, respectively. Nord, PJM, and SPP use both 24- and 96-hour forecast horizons, while GEFCom2014 uses only the 96- hour horizon. We use LCL and iFlex only for day-ahead forecasting because the LCL tarif notices and iFlex research price schedules used here cover only the next day. Table 1 lists sources, history and forecast windows, and split sizes.

Table 1: Dataset sizes by forecasting window. �/�: history/forecast hours. ERCOT includes 71 system-level cases for the main evaluation and 14 regional sensitivity cases.
<table><tr><td>Source</td><td> $L / H$ </td><td>Train</td><td>Val.</td><td>ID test</td><td>OOD test</td><td>Total</td></tr><tr><td>GEFCom2014</td><td>96/96</td><td>640</td><td>100</td><td>200</td><td>0</td><td>940</td></tr><tr><td>Nord</td><td>168/24</td><td>288</td><td>48</td><td>96</td><td>0</td><td>432</td></tr><tr><td></td><td>96/96</td><td>72</td><td>12</td><td>24</td><td>0</td><td>108</td></tr><tr><td>PJM</td><td>168/24</td><td>320</td><td>56</td><td>112</td><td>0</td><td>488</td></tr><tr><td></td><td>96/96</td><td>80</td><td>14</td><td>28</td><td>0</td><td>122</td></tr><tr><td>SPP</td><td>168/24</td><td>160</td><td>24</td><td>48</td><td>0</td><td>232</td></tr><tr><td></td><td>96/96</td><td>40</td><td>6</td><td>12</td><td>0</td><td>58</td></tr><tr><td>LCL</td><td>168/24</td><td>120</td><td>24</td><td>48</td><td>0</td><td>192</td></tr><tr><td>iFlex</td><td>168/24</td><td>80</td><td>16</td><td>32</td><td>0</td><td>128</td></tr><tr><td>AEMO</td><td>96/96</td><td>0</td><td>0</td><td>0</td><td>150</td><td>150</td></tr><tr><td>ERCOT</td><td>168/24</td><td>0</td><td>0</td><td>0</td><td>85</td><td>85</td></tr><tr><td>Total</td><td>一</td><td>1,800</td><td>300</td><td>600</td><td>235</td><td>2,935</td></tr></table>

To evaluate forecasting under diferent event conditions, we construct synthetic cases from GEFCom2014 with single or multiple load-relevant events, or only irrelevant events. We also retain the original load sequences without added events. For load-relevant events, we modify the future target load according to the specified magnitudes and timing of their efects. Some events include a recovery phase in which deferred electricity demand returns in later hours. We combine multiple events by summing efects computed separately from the original sequence or applying events sequentially by start time to the already modified sequence. Under the sequential rule, later efects depend on earlier changes. Cases with only irrelevant events retain the original future target load. Appendix A details load and event sources, construction rules, and case proportions.

5.1.2 Evaluation Metrics. We use weighted absolute percentage error (WAPE) [20] as the primary metric. The secondary metric is normalized root mean squared error (NRMSE) [21], which is more sensitive to large deviations. For a true load sequence � and a forecast sequence �ˆ, both of length �, the metrics are defined as

$$
\begin{array} { r l } & { \mathrm { W A P E } = \cfrac { \sum _ { k = 0 } ^ { H - 1 } | \hat { y } _ { k } - y _ { k } | } { \sum _ { k = 0 } ^ { H - 1 } | y _ { k } | } , } \\ & { \mathrm { N R M S E } = \cfrac { \sqrt { H ^ { - 1 } \sum _ { k = 0 } ^ { H - 1 } ( \hat { y } _ { k } - y _ { k } ) ^ { 2 } } } { H ^ { - 1 } \sum _ { k = 0 } ^ { H - 1 } | y _ { k } | } . } \end{array}\tag{6}
$$

(7)

WAPE and NRMSE score the delivered forecasts for all cases, including fallback forecasts after failure, and are reported as percentages. Appendix A.6 details aggregation and near-zero-load handling; Appendix B.4 describes fallback.

5.1.3 Baselines. The statistical baseline is ARIMAX; the tree-based machine learning baseline is LightGBM [19]; deep time series baselines include PatchTST [25], TimeXer [41], and EventTSF [14]. Time series foundation models include TimesFM-3 [18] and Chronos-2 [2]. Large language model forecasting baselines include TimeReasoner [8] and MemCast [33]. We use a DeepSeek-v4-flash [10] implementation of TimeReasoner in our experiments. Agent-based forecasting baselines include CastFSR [34] and AlphaCast [44].

Table 2: Overall performance in multi-day and day-ahead load forecasting. WAPE and NRMSE are reported as percentages. Best results are bold; second-best results are underlined. �/�: history length/forecast horizon in hours.
<table><tr><td></td><td colspan="8">Multi-day forecasting (L = 96, H = 96)</td><td colspan="10">Day-ahead forecasting (L = 168, H = 24)</td></tr><tr><td>Method</td><td colspan="2">GEFCom2014</td><td colspan="2">Nord</td><td colspan="2">PJM</td><td colspan="2">SPP AEMO†</td><td colspan="2">Nord</td><td colspan="2">PJM</td><td colspan="2">SPP</td><td colspan="2">LCL</td><td colspan="2">iFlex</td><td colspan="2">ERCOT†</td></tr><tr><td></td><td>WAPE NRMSE WAPE NRMSE WAPE NRMSE WAPE NRMSE WAPE NRMSE WAPE NRMSE WAPE NRMSE WAPE NRMSE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>WAPE NRMSE WAPE NRMSE WAPE NRMSE</td><td></td><td></td><td></td><td></td></tr><tr><td>ARIMAX</td><td>17.63</td><td>20.91</td><td>7.49</td><td>8.87 10.09</td><td>11.62</td><td>6.09</td><td>7.08</td><td>29.21</td><td>33.83</td><td>5.48</td><td>6.58</td><td>7.01</td><td>8.07 6.25</td><td>7.11</td><td>808.45</td><td>964.88</td><td>30.19</td><td>37.34</td><td>7.44</td><td>8.86</td></tr><tr><td>LightGBM</td><td>9.03</td><td>10.90</td><td>5.93</td><td>6.90 7.07</td><td>8.15</td><td>4.50</td><td>5.51</td><td>15.57</td><td>20.06</td><td>4.84</td><td>5.56</td><td>4.89</td><td>5.75 4.76</td><td>5.64</td><td></td><td>1008.97 1167.11</td><td>34.94</td><td>41.98</td><td>7.55</td><td>8.51</td></tr><tr><td>PatchTST</td><td>13.82</td><td>16.91</td><td>7.97</td><td>9.35 7.52</td><td>8.96</td><td>5.07</td><td>6.23</td><td>18.45</td><td>22.67</td><td>5.22</td><td>6.14</td><td>5.08 6.28</td><td>4.33</td><td>5.46</td><td></td><td>1873.00 2151.25</td><td>31.19</td><td>39.03</td><td>7.00</td><td>8.56</td></tr><tr><td>TimeXer</td><td>13.40</td><td>16.45</td><td>8.18</td><td>9.58 7.84</td><td>9.21</td><td>4.92</td><td>5.96</td><td>15.85</td><td>20.11</td><td>4.96</td><td>5.69</td><td>4.71 5.74</td><td>4.16</td><td>5.01</td><td>958.83</td><td>1120.3030.49</td><td></td><td>38.58</td><td>7.18</td><td>8.47</td></tr><tr><td>EventTSF</td><td>16.57</td><td>20.15</td><td>8.82</td><td>10.15 9.69</td><td>11.67</td><td>7.61</td><td>9.43</td><td>23.62</td><td>29.53</td><td>5.54</td><td>6.35</td><td>5.48 6.55</td><td>5.73</td><td>6.73</td><td>804.16</td><td>1017.90</td><td>27.65</td><td>35.52</td><td>11.77</td><td>13.36</td></tr><tr><td>Chronos-2</td><td>9.92</td><td>12.29</td><td>7.11</td><td>8.22 6.90</td><td>8.33</td><td>4.30</td><td>5.27</td><td>13.99</td><td>18.38</td><td>3.60</td><td>4.22</td><td>3.61 4.40</td><td>2.64</td><td>3.18</td><td>62.39</td><td>99.54</td><td>25.39</td><td>32.73</td><td>4.28</td><td>5.23</td></tr><tr><td>TimesFM-3</td><td>9.40</td><td>11.68</td><td>7.13</td><td>8.45 6.11</td><td>7.48</td><td>4.23</td><td>5.12</td><td>14.75</td><td>19.26</td><td>3.29</td><td>3.91</td><td>3.43 4.23</td><td>2.66</td><td>3.28</td><td>41.87</td><td>73.86</td><td>25.64</td><td>33.02</td><td>3.99</td><td>4.97</td></tr><tr><td>TimeReasoner</td><td>8.61</td><td>10.84</td><td>5.76</td><td>6.76 6.45</td><td>7.78</td><td>3.82</td><td>4.87</td><td>12.61</td><td>17.48</td><td>3.96</td><td>4.55</td><td>5.48</td><td>6.30 2.94</td><td>3.52</td><td>97.70</td><td>179.39</td><td>30.10</td><td>38.72</td><td>12.94</td><td>13.78</td></tr><tr><td>MemCast</td><td>8.18</td><td>10.35</td><td>5.75</td><td>6.78 6.32</td><td>7.63</td><td>3.59</td><td>4.53</td><td>12.78</td><td>17.57</td><td>4.52</td><td>5.26</td><td>3.89</td><td>4.76 5.75</td><td>6.98</td><td>67.93</td><td>142.36</td><td>31.16</td><td>40.31</td><td>7.85</td><td>9.53</td></tr><tr><td>AlphaCast</td><td>12.68</td><td>15.39</td><td>7.93 9.22</td><td>7.61</td><td>9.03</td><td>4.51</td><td>5.51</td><td>14.20</td><td>18.73</td><td>3.96</td><td>4.57</td><td>3.64 4.45</td><td>2.93</td><td>3.48</td><td></td><td>2726.63 2890.32</td><td>30.53</td><td>38.33</td><td>5.02</td><td>6.04</td></tr><tr><td>CastFSR</td><td>9.36</td><td>11.66</td><td>10.66 11.74</td><td>6.66</td><td>7.95</td><td>3.82</td><td>4.80</td><td>13.39</td><td>17.69</td><td>3.61</td><td>4.15</td><td>3.78 4.58</td><td>3.33</td><td>3.87</td><td>86.95</td><td>167.05</td><td>26.88</td><td>34.78</td><td>5.73</td><td>6.66</td></tr><tr><td>MMGPT4LF</td><td>10.86</td><td>13.25</td><td>7.71</td><td>8.88</td><td>8.80 10.11</td><td>6.84</td><td>7.98</td><td>18.44</td><td>23.23</td><td>7.00</td><td>7.91</td><td>7.72</td><td>9.04 7.60</td><td>8.80</td><td>851.35</td><td>938.90</td><td>30.62</td><td>38.23</td><td>13.75</td><td>15.46</td></tr><tr><td>NACF</td><td>14.23</td><td>17.15</td><td>7.88</td><td>9.22 8.81</td><td>10.41</td><td>6.69</td><td>8.07</td><td>17.22</td><td>21.73</td><td>5.96</td><td>7.06</td><td>5.92 7.17</td><td>5.14</td><td>6.42</td><td>1761.87</td><td>2013.86</td><td>33.87</td><td>42.33</td><td>11.19</td><td>13.29</td></tr><tr><td>ITA-LF</td><td>15.94</td><td>19.66</td><td>9.69</td><td>11.55 14.63</td><td>22.94</td><td>9.55</td><td>11.21</td><td>32.68</td><td>48.16</td><td>5.06</td><td>7.18</td><td>4.18 5.13</td><td>4.04</td><td>4.83</td><td>65.19</td><td>107.48</td><td>32.66</td><td>43.73</td><td>6.37</td><td>7.62</td></tr><tr><td>ElecLoad</td><td>6.68</td><td>8.42</td><td>5.43</td><td>6.43</td><td>6.03 7.28</td><td>3.63</td><td>4.13</td><td>13.49</td><td>17.69</td><td>3.52</td><td>4.14</td><td>3.13 3.86</td><td>2.36</td><td>2.92</td><td>41.94</td><td>77.24</td><td>24.32</td><td>31.06</td><td>4.33</td><td>5.20</td></tr></table>

<sup>†</sup> OOD datasets; all other datasets are ID.

Electricity load forecasting baselines include MMGPT4LF [13], NACF [6], and ITA-LF [40]. Appendix B.5 describes these variants.

5.1.4 Implementation Details. ElecLoad uses Qwen3.5-4B as its backbone and is trained on four NVIDIA H100 80GB GPUs. Agent SFT starts from the reasoning RL checkpoint and updates all model parameters for 2 epochs. It uses 11,135 role-specific training examples from 1,800 cases, with a batch size of 16. We use AdamW [24] with a learning rate of $5 \times 1 0 ^ { - 6 }$ and a weight decay of 0.01. Agent RL starts from the resulting SFT checkpoint and also updates all model parameters. It uses a learning rate of $1 \times { 1 0 } ^ { - 6 }$ , 16 forecast origins per batch, and a sampling temperature of 1.0. For segmented forecasting, the implementation allows at most 4 segments, each containing at least 6 hourly steps. Each episode has a budget of 2 refinements, each applied to one selected segment. The main agent can stop before using the full budget. Appendix B.4 lists training configurations and runtime budgets.

## 5.2 Overall Performance

Table 2 compares ElecLoad with 14 external baselines across 22 metric comparisons. ElecLoad ranks first in 13 and second in six, with a top-two rank in all 18 in-domain comparisons. These ID results span 24-hour day-ahead and 96-hour multi-day forecasting and household, regional, and system load sources. On GEFCom2014, ElecLoad reduces WAPE and NRMSE by 18.3% and 18.6%, respectively, relative to the strongest baseline, MemCast.

AEMO and ERCOT test transfer to power systems excluded from task training, at forecast horizons of 96 and 24 hours, respectively. ElecLoad ranks second to fourth across these four OOD metric comparisons among the 15 evaluated methods.

## 5.3 Ablation Study

We use the comparisons in Table 3 to examine how event text and covariates, together with planning, selective refinement, and boundary reconciliation, contribute to ElecLoad’s forecasting accuracy. ElecLoad has the lowest WAPE and NRMSE on all three datasets. Removing event text or covariates raises both errors throughout. Covariate removal has the larger efect on PJM.

The segmented-only variant outperforms the direct-only variant on GEFCom2014, whereas the latter performs better on LCL; ElecLoad has lower mean errors than both. Equal-length segments also raise errors while preserving each case’s segment count, highlighting the contribution of segment placement in planning (T1). Removing T3-guided refinement raises both metrics throughout. Refinement without feedback and random refinement also yield higher errors. Boundary reconciliation (T4) adds a smaller improvement: removing it, or replacing the option to retain the current sequence with forced boundary reconciliation within permitted limits, raises both metrics on every dataset.

## 5.4 Multi-Event Forecasting

Event conditions. Table 4 groups cases by event relevance. Relevant events afect the target load within the forecast horizon, including recovery. Irrelevant descriptions concern events elsewhere that do not afect this target, administrative notices unrelated to load, or events whose efects fall entirely outside the horizon. ElecLoad outperforms ITA-LF across groups; removing event text while retaining load history and covariates increases error when relevant events are present. Its advantage over the direct-only variant is larger for multiple events: PJM WAPE is 3.69% versus 4.31%, compared with 2.63% versus 2.80% without events. The variant uses the same model and inputs to forecast the entire horizon, so this comparison evaluates the full adaptive workflow.

Interaction recovery. To complement forecasting results under real-event co-occurrence, this controlled experiment tests whether

![](images/e2fbb432634328ba0d87e5f36e2a29fbc7a92f7aa2fd683cfcbc1bd78633bc54.jpg)

![](images/51db18030198293ee51861f591010e033275f60c7a484c4e63b68b889e4a1d33.jpg)

ElecLoad w/ direct only ElecLoad (Full)

![](images/7d389fe1464b81a91080aebfe5e8dcc4001dc6177b55e43e702c75c111162d81.jpg)

(d) Generalization IEE  
![](images/503073acf460f700a2c0753cea1f7300344995743906be097d91f9c3afaa8a22.jpg)  
Figure 3: Forecasting on GEFCom2014 synthetic event cases. (a) WAPE for four event conditions; 0 retains the original target without added events. (b) IEE grouped by zero/nonzero target interaction. (c,d) WAPE and IEE across generalization settings.

Table 3: Ablation study (� = 96, � = 96 for GEFCom2014; � = 168, � = 24 for PJM and LCL).
<table><tr><td rowspan="2">Group</td><td rowspan="2">Variant</td><td colspan="2">GEFCom2014</td><td colspan="2">PJM</td><td colspan="2">LCL</td></tr><tr><td>WAPE NRMSE WAPE NRMSE WAPE NRMSE</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>ElecLoad (Full)</td><td>6.68</td><td>8.42</td><td>3.13</td><td>3.86</td><td>41.94</td><td>77.24</td></tr><tr><td rowspan="2">Inputs</td><td>w/o event text</td><td>8.46</td><td>10.63</td><td>3.51</td><td>4.37</td><td>46.27</td><td>83.16</td></tr><tr><td>w/o covariates</td><td>7.39</td><td>9.36</td><td>3.79</td><td>4.71</td><td>44.12</td><td>80.58</td></tr><tr><td rowspan="3">Planning</td><td>w/ direct only</td><td>7.82</td><td>9.91</td><td>3.48</td><td>4.32</td><td>42.91</td><td>78.83</td></tr><tr><td>w/ segmented only</td><td>6.97</td><td>8.86</td><td>3.39</td><td>4.21</td><td>45.36</td><td>82.07</td></tr><tr><td>w/ equal-length segments</td><td>7.51</td><td>9.54</td><td>3.46</td><td>4.29</td><td>43.68</td><td>80.01</td></tr><tr><td rowspan="3">Refinement</td><td>w/o refinement</td><td>7.24</td><td>9.18</td><td>3.37</td><td>4.18</td><td>44.63</td><td>81.62</td></tr><tr><td>w/o feedback</td><td>7.06</td><td>8.97</td><td>3.29</td><td>4.08</td><td>43.57</td><td>79.85</td></tr><tr><td>w/ random refinement</td><td>6.94</td><td>8.81</td><td>3.32</td><td>4.12</td><td>43.06</td><td>79.18</td></tr><tr><td rowspan="2">Reconciliation</td><td>w/o reconciliation</td><td>6.77</td><td>8.56</td><td>3.15</td><td>3.90</td><td>42.08</td><td>77.51</td></tr><tr><td>w/ forced reconciliation</td><td>6.73</td><td>8.51</td><td>3.19</td><td>3.97</td><td>42.31</td><td>77.96</td></tr></table>

Removing feedback retains refinement; random refinement selects refinement targets at random.

Table 4: Forecasting performance under diferent event conditions (� = 96, � = 96 for GEFCom2014; � = 168, � = 24 for PJM and LCL).
<table><tr><td rowspan="2">Event Condition</td><td rowspan="2">Method</td><td colspan="2">GEFCom2014</td><td colspan="2">PJM</td><td colspan="2">LCL</td></tr><tr><td></td><td>WAPE NRMSE WAPE NRMSE WAPE NRMSE</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">No Event</td><td>ITA-LF</td><td>13.50</td><td>16.50</td><td>3.34</td><td>3.80</td><td>56.08</td><td>89.68</td></tr><tr><td>ElecLoad w/o event text</td><td>5.44</td><td>6.83</td><td>2.41</td><td>2.88</td><td>39.96</td><td>69.54</td></tr><tr><td>ElecLoad w/ direct only</td><td>6.00</td><td>7.60</td><td>2.80</td><td>3.40</td><td>38.72</td><td>69.47</td></tr><tr><td>ElecLoad (Full)</td><td>5.44</td><td>6.83</td><td>2.63</td><td>3.19</td><td>38.59</td><td>70.28</td></tr><tr><td rowspan="4">Single Event</td><td>ITA-LF</td><td>15.80</td><td>19.40</td><td>4.12</td><td>5.10</td><td>64.39</td><td>103.40</td></tr><tr><td>ElecLoad w/o event text</td><td>8.20</td><td>10.30</td><td>3.57</td><td>4.43</td><td>47.64</td><td>84.39</td></tr><tr><td>ElecLoad w/ direct only</td><td>7.90</td><td>10.00</td><td>3.42</td><td>4.25</td><td>42.68</td><td>77.34</td></tr><tr><td>ElecLoad (Full)</td><td>6.42</td><td>8.06</td><td>3.12</td><td>3.84</td><td>41.73</td><td>76.38</td></tr><tr><td rowspan="4">Multiple Events</td><td>ITA-LF</td><td>19.00</td><td>23.50</td><td>5.18</td><td>6.63</td><td>81.73</td><td>142.29</td></tr><tr><td>ElecLoad w/o event text</td><td>10.80</td><td>13.80</td><td>4.63</td><td>5.91</td><td>57.08</td><td>107.93</td></tr><tr><td>ElecLoad w/ direct only</td><td>9.50</td><td>12.10</td><td>4.31</td><td>5.43</td><td>51.00</td><td>97.68</td></tr><tr><td>ElecLoad (Full)</td><td>7.61</td><td>9.71</td><td>3.69</td><td>4.62</td><td>48.44</td><td>90.84</td></tr><tr><td rowspan="4">Irrelevant Events</td><td>ITA-LF</td><td>14.00</td><td>17.20</td><td>4.39</td><td>5.48</td><td>58.56</td><td>94.55</td></tr><tr><td>ElecLoad w/o event text</td><td>8.02</td><td>9.08</td><td>3.83</td><td>4.81</td><td>40.40</td><td>70.78</td></tr><tr><td>ElecLoad w/ direct only</td><td>6.90</td><td>8.60</td><td>3.64</td><td>4.53</td><td>39.24</td><td>70.83</td></tr><tr><td>ElecLoad (Full)</td><td>5.49</td><td>6.91</td><td>3.26</td><td>4.04</td><td>39.00</td><td>71.46</td></tr></table>

![](images/fbb7150a12001146f4abc7aef15192bb4f9cb2bd97823b70041d74d7832f43de.jpg)

![](images/b87664080f0701032df688abd2ce228d37ea3d9dd8604ceaab4718b4c0b036c1.jpg)  
Figure 4: WAPE versus normalized token cost on GEF-Com2014: (a) routing; (b) refinement and reconciliation.

forecasts recover specified interaction efects. We construct synthetic event cases with four conditions for each GEFCom2014 base case: 0 retains the original future targets without adding events; � and � each add a diferent relevant event; and �� includes both. We modify future targets according to the events and supply the corresponding event descriptions.

An event’s efect is its load change relative to condition 0. Interaction is the change under �� minus the sum of the changes under � and �. Zero interaction means that the combined efect equals this sum; nonzero interaction means a larger or smaller combined efect. WAPE measures load forecast accuracy under each condition. Interaction estimation error (IEE) compares the interaction computed from the four forecasts, $\widehat { \Delta } _ { A B , k }$ , with its target counterpart:

$$
\begin{array} { r l r } {  { \Delta _ { A B , k } = y _ { A B , k } - y _ { A , k } - y _ { B , k } + y _ { 0 , k } , } } \\ & { } & { \quad \mathrm { I E E } = \frac { \sum _ { k } \vert \widehat { \Delta } _ { A B , k } - \Delta _ { A B , k } \vert } { \sum _ { k } \vert y _ { 0 , k } \vert } , } \end{array}\tag{8}
$$

where $y _ { s , k }$ is the target load under condition � at step �. Both sums cover the forecast horizon; IEE is normalized by the sum of absolute loads under condition 0 and reported as a percentage.

Figure 3(a) reports WAPE under each condition; under ��, ElecLoad achieves 3.20% versus 3.94% for the direct-only variant. Figure 3(b) groups cases by zero or nonzero target interaction; in the nonzero group, ElecLoad’s IEE is 0.57% versus 1.12% for this variant. In this controlled comparison, ElecLoad is more accurate in both load forecasting and interaction estimation.

![](images/c77e276c1a12af5605d4c1c2171410d614d9631f0ae3b0a6fc191a463ba69c78.jpg)

![](images/26ffe17322ba70f8142ed52ff9b0c7a11213e6f69bf8b0dc90f83f64d3b2b7cf.jpg)  
Figure 5: ElecLoad case: (a) T1 planning and T2 initial forecasts; (b) T3-guided refinement and T4 boundary reconciliation.

Table 5: Ablations of routing, refinement, and reconciliation.
<table><tr><td rowspan="2">Group</td><td rowspan="2">Variant</td><td colspan="3">GEFCom2014</td><td colspan="3">PJM</td><td colspan="3">LCL</td></tr><tr><td>E</td><td>C</td><td>G</td><td>E</td><td>C</td><td>G</td><td>E</td><td>C</td><td>G</td></tr><tr><td rowspan="3">Routing</td><td>w/ direct only</td><td>7.82</td><td>7.8</td><td></td><td>3.48</td><td>7.0</td><td></td><td>42.91</td><td>6.5</td><td></td></tr><tr><td>w/ segmented only</td><td>6.97 24.0</td><td></td><td>0.052</td><td>3.39</td><td>22.0</td><td>0.006</td><td>45.36</td><td></td><td>20.0-0.181</td></tr><tr><td>ElecLoad (Full)</td><td>6.6818.7</td><td></td><td>0.105</td><td>3.13</td><td>15.3</td><td>0.042</td><td>41.94</td><td>12.7</td><td>0.156</td></tr><tr><td rowspan="3">Refinement</td><td>w/o refinement</td><td>7.24 15.0</td><td></td><td>1</td><td></td><td>3.3712.2</td><td>一</td><td>44.63</td><td>10.0</td><td></td></tr><tr><td>w/ random refinement</td><td>6.94 18.6</td><td></td><td>0.083</td><td></td><td>3.32 15.2</td><td>0.017</td><td>43.06</td><td>12.8</td><td>0.561</td></tr><tr><td>ElecLoad (Full)</td><td></td><td></td><td></td><td></td><td>6.68 18.7 0.151 3.13 15.3</td><td>0.077</td><td>41.94 12.7</td><td></td><td>0.996</td></tr><tr><td rowspan="3"></td><td>w/o reconciliation</td><td>6.77 17.6</td><td></td><td></td><td></td><td>3.15 14.4</td><td>一</td><td>42.08 12.1</td><td></td><td></td></tr><tr><td>Reconciliation w/ forced reconciliation 6.73 19.5 0.021 3.19 16.2 -0.022 42.31 13.7 -0.144</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ElecLoad (Full)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6.68 18.7 0.082 3.13 15.3 0.022 41.94 12.7 0.233</td></tr></table>

Generalization. Figure 3(c,d) reports WAPE (c) and IEE (d) under Seen (combinations encountered during training), New overlap (seen combinations with unseen temporal overlap), and New pair (unseen combinations). ElecLoad achieves the lowest values of both metrics among the four methods in every setting. Appendix A.4 provides construction details.

Appendix C.4 provides a supplementary analysis of how adding irrelevant event text afects load forecasts.

## 5.5 Planning and Refinement Analysis

Accuracy–cost trade-of. Table 5 evaluates how routing, refinement, and reconciliation afect forecasting accuracy and token cost. � is WAPE (%), and � is the mean total number of input and output tokens per case, in thousands. We define $G = ( E _ { \mathrm { r e f } } - E ) / ( C - C _ { \mathrm { r e f } } )$ using each group’s first row as the reference for each dataset. It measures the WAPE reduction in percentage points per additional 1,000 tokens; “—” denotes an undefined value.

The direct-only variant always uses direct forecasting. The segmented-only variant always partitions the horizon into segments. It retains event-guided boundaries, selective refinement, and boundary reconciliation. Figure 4 plots WAPE against normalized token cost on all GEFCom2014 test cases. For each case, we divide its token count by that of its direct run, then average over cases. Panel (a) compares ElecLoad with the direct-only and segmented-only vari ants. Panel (b) compares refinement and reconciliation with every case on the segmented path. The refinement and reconciliation groups in Table 5 retain adaptive routing.

Adaptive routing achieves lower WAPE with fewer tokens than the segmented-only variant on all three datasets. On GEFCom2014, ElecLoad reduces WAPE from 6.97% to 6.68% while using 22.1% fewer tokens. Relative to direct-only forecasting, it reduces WAPE from 7.82% to 6.68% at a higher cost of 18.7k versus 7.8k tokens. Selective refinement also achieves lower WAPE than random refinement at similar token cost across all three datasets; on GEFCom2014, the comparison is 6.68% versus 6.94% at 18.7k versus 18.6k tokens. Boundary reconciliation achieves lower error with fewer tokens than forced reconciliation. Together, these comparisons support adapting both the forecasting path and the allocation of refinement calls within the evaluated workflow.

Case study. Figure 5(a) shows overlapping heatwave and demand response events. Event analysis guides T1 to partition the horizon into S1–S3; S2 covers their overlap. T2 independently forecasts each segment, forming “T2 Initial”, which overestimates load in S2. In panel (b), T3 provides feedback; T2 revises S2 values, and the main agent reassesses the full forecast to obtain “After T3”. T4’s boundary reconciliation locally adjusts forecast values near segment boundaries, producing “Final” and complementing refinement focused on the overestimated segment.

## 6 Conclusion

We present ElecLoad, which uses multi-event analysis to guide adaptive electricity load forecasting so that the forecasting process follows changes in event efects over time. Experiments on day ahead and multi-day forecasting show that ElecLoad performs best in most metric comparisons; its error reduction over the direct-only variant, which uses the same model and inputs, is larger under multi-event conditions. Ablation analysis shows that the relative performance of direct and segmented forecasting varies across datasets, whereas the adaptive forecasting process achieves lower average error than either fixed forecasting path. Selective refinement yields lower error than random refinement at similar token cost across all three evaluated datasets. These results indicate that event analysis can efectively inform which periods to forecast together and which segments require refinement.

## References

[1] Jillian Ambrose. 2026. Fans for fans: power demand surges as England viewers cope with heatwave. The Guardian. June 24, 2026. Accessed October 14, 2026. https://www.theguardian.com/football/2026/jun/24/fans-power-demandsurges-england-viewers-heatwave-world-cup

[2] Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, et al. 2025. Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821 (2025).

[3] Australian Energy Market Operator. [n. d.]. NEMWeb: Operational Demand. Accessed October 25, 2026. https://www.nemweb.com.au/Reports/CURRENT/ Operational\_Demand/

[4] Yun Bai, Simon Camal, and Andrea Michiorri. 2024. News and load: A quantitative exploration ofnatural language processing applications for forecasting day-ahead electricity system demand. IEEE Transactions on Power Systems 39, 5 (2024), 6222– 6234.

[5] Yun Bai, Simon Camal, and Andrea Michiorri. 2025. News and load: Social and economic drivers of regional multi-horizon electricity demand forecasting. IEEE Transactions on Power Systems 40, 6 (2025), 4839–4851.

[6] Yujie Chen, Yifei Gao, Runyao Yu, Yuhe Wu, Guangyu Wang, Yue Chen, and Tongxin Li. 2026. Counterfactual load forecasting with LLM-structured events and representation learning. Applied Energy 425 (2026), 128554.

[7] Mingyue Cheng, Zhiding Liu, Xiaoyu Tao, Qi Liu, Jintao Zhang, Tingyue Pan, Shilong Zhang, Panjing He, Xiaohan Zhang, Daoyu Wang, et al. 2026. A comprehensive survey of time series forecasting: Concepts, challenges, and future directions. IEEE Transactions on Knowledge and Data Engineering (2026).

[8] Mingyue Cheng, Jiahao Wang, Daoyu Wang, Xiaoyu Tao, Qi Liu, and Enhong Chen. 2026. Can slow-thinking llms reason over time? empirical studies in time series forecasting. In Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining. 99–110.

[9] Mingyue Cheng, Jiqian Yang, Tingyue Pan, Qi Liu, Zhi Li, and Shijin Wang. 2025. Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis. In Companion Proceedings ofthe ACM on Web Conference 2025. 171–180.

[10] DeepSeek-AI. 2026. DeepSeek-V4-Flash. Model card. https://huggingface.co/ deepseek-ai/DeepSeek-V4-Flash

[11] Electric Reliability Council of Texas. [n. d.]. Hourly Load Data Archives. Accessed October 25, 2026. https://www.ercot.com/gridinfo/load/load\_hist

[12] ENTSO-E. [n. d.]. Power Statistics. Accessed October 25, 2026. https://www. entsoe.eu/data/power-stats

[13] Mingyang Gao, Suyang Zhou, Wei Gu, Zhi Wu, Haiquan Liu, Aihua Zhou, and Xinliang Wang. 2025. MMGPT4LF: Leveraging an optimized pre-trained GPT-2 model with multi-modal cross-attention for load forecasting. Applied Energy 392 (2025), 125965.

[14] Yunfeng Ge, Ming Jin, Yiji Zhao, Hongyan Li, Bo Du, Chang Xu, and Shirui Pan. 2026. EventTSF: Event-Aware Non-Stationary Time Series Forecasting. In Proceedings ofthe Thirty-Fifth International Joint Conference on Artificial Intelligence. 4313–4321. doi:10.24963/ijcai.2026/480

[15] Matthias Hofmann and Turid Siebenbrunner. 2023. A rich dataset of hourly resi dential electricity consumption data and survey answers from the iFlex dynamic pricing experiment. doi:10.5281/zenodo.8248802

[16] Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli, and Rob J Hyndman. 2016. Probabilistic energy forecasting: Global energy forecasting competition 2014 and beyond. 896–913 pages.

[17] International Energy Agency. 2025. Energy and AI. Technical Report. International Energy Agency, Paris. https://www.iea.org/reports/energy-and-ai

[18] Ayush Jain and Rajat Sen. 2026. TimesFM-3: A zero-shot foundation model for multivariate forecasting. Google Research Blog 31 (2026).

[19] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly eficient gradient boosting decision tree. Advances in neural information processing systems 30 (2017).

[20] Stephan Kolassa and Wolfgang Schütz. 2007. Advantages of the MAD/MEAN ratio over the MAPE. Foresight: The International Journal of Applied Forecasting 6 (2007), 40–43.

[21] Pekka Koponen, Jussi Ikäheimo, Juha Koskela, Christina Brester, and Harri Niska. 2020. Assessing and comparing short term load forecasting performance. Energies 13, 8 (2020), 2054.

[22] Shuhao Li, Wanfu Zheng, Mingcheng Li, Shanshuo Xing, and Zhe Wang. 2026. AutoLFM: A multi-agent LLM framework for automated building load forecasting. In Building Simulation, Vol. 19. Springer, 545–568.

[23] Zhiding Liu, Jiqian Yang, Mingyue Cheng, Zhi Li, and Enhong Chen. 2026. World2TS: Real-World Event Context for Time-Series Forecasting. arXiv preprint arXiv:2610.14861 (2026).

[24] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[25] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv

preprint arXiv:2211.14730 (2022).

[26] Bokai Pan, Mingyue Cheng, Zhiding Liu, Shuo Yu, Xiaoyu Tao, Yuchong Wu, Qi Liu, Defu Lian, and Enhong Chen. 2026. Castflow: Learning role-specialized agentic workflows for time series forecasting. arXiv preprint arXiv:2604.27840 (2026).

[27] Tingyue Pan, Mingyue Cheng, Shilong Zhang, Zhiding Liu, Xiaoyu Tao, Yucong Luo, Jintao Zhang, and Qi Liu. 2025. OneCast: Structured Decomposition and Modular Generation for Cross-Domain Time Series Forecasting. arXiv preprint arXiv:2510.24028 (2025).

[28] PJM Interconnection. [n. d.]. Data Miner 2: Metered Load. Accessed October 25, 2026. https://dataminer2.pjm.com/feed/hrl\_load\_metered

[29] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen. ai/blog?id=qwen3.5

[30] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

[31] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).

[32] Shreyashi Shukla and Tao Hong. 2024. BigDEAL Challenge 2022: Forecasting peak timing of electricity demand. IET Smart Grid 7, 4 (2024), 442–459.

[33] Xiaoyu Tao, Mingyue Cheng, Ze Guo, Shuo Yu, Yaguo Liu, Qi Liu, and Shijin Wang. 2026. MemCast: Memory-Driven Time Series Forecasting with Experience-Conditioned Reasoning. In Proceedings ofthe 43rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 306). PMLR, 119553–119569. https://proceedings.mlr.press/v306/tao26a.html

[34] Xiaoyu Tao, Mingyue Cheng, Bokai Pan, Chuang Jiang, Huanjian Zhang, Tian Gao, Yaguo Liu, Qi Liu, and Enhong Chen. 2026. CastFSR: A Fast–Slow–Reflect Agentic Reasoning Framework for Context-Aware Time Series Forecasting. arXiv preprint arXiv:2608.03031 (2026).

[35] Xiaoyu Tao, Shilong Zhang, Mingyue Cheng, Daoyu Wang, Tingyue Pan, Bokai Pan, Changqing Zhang, and Shijin Wang. 2026. From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization. In Proceedings ofthe Thirty-Fifth International Joint Conference on Artificial Intelligence. 3043–3051. doi:10.24963/ijcai.2026/338

[36] UK Power Networks. [n. d.]. SmartMeter Energy Consumption Data in London Households. Accessed October 25, 2026. https://data.london.gov.uk/dataset/ smartmeter-energy-consumption-data-in-london-households-vqm0d

[37] U.S. Energy Information Administration. [n. d.]. Hourly Electric Grid Monitor. Accessed October 25, 2026. https://www.eia.gov/electricity/gridmonitor/

[38] Daoyu Wang, Mingyue Cheng, Zhiding Liu, and Qi Liu. 2025. TimeDART: A Difusion Autoregressive Transformer for Self-Supervised Time Series Representation. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 267). PMLR, 62627–62651. https://proceedings.mlr.press/v267/wang25r.html

[39] Xinlei Wang, Maike Feng, Jing Qiu, Jinjin Gu, and Junhua Zhao. 2024. From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection. Advances in Neural Information Processing Systems 37 (2024), 58118–58153.

[40] Xinlei Wang, Jinjin Gu, Jing Qiu, Guolong Liu, Xinlei Cai, Jinzhou Zhu, Yanli Liu, Zhaoyang Dong, and Junhua Zhao. 2026. News-Driven Load Forecasting: Generative Agents and Large Language Models for Unstructured Data and Event Analysis. Engineering (2026).

[41] Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, and Mingsheng Long. 2024. Timexer: Empowering transformers for time series forecasting with exogenous variables. Advances in neural information processing systems 37 (2024), 469–498.

[42] Yamin Wang and Lei Wu. 2017. Improving economic values of day-ahead load forecasts to real-time power system operations. IET Generation, Transmission & Distribution 11, 17 (2017), 4238–4247.

[43] Hanshu Yan, Hong-Quang Pham, Ibrahim Taha Aksu, Jun Hao Liew, Chenghao Liu, Doyen Sahoo, Junnan Li, Caiming Xiong, and Silvio Savarese. 2026. Moira iAgent: An Agentic Framework for Context-Aware Time-Series Forecasting. Salesforce AI Research Blog (2026).

[44] Xiaohan Zhang, Tian Gao, Mingyue Cheng, Bokai Pan, Ze Guo, Yaguo Liu, and Xiaoyu Tao. 2025. Alphacast: A human wisdom-llm intelligence co-reasoning framework for interactive time series forecasting. arXiv e-prints (2025), arXiv– 2511.

[45] Pengfei Zhao, Di Cao, Yanbo Wang, Zhe Chen, and Weihao Hu. 2023. Gaussian process-aided transfer learning for probabilistic load forecasting agains anomalous events. IEEE transactions on power systems 38, 3 (2023), 2962–2965.

## A Evaluation Construction

## A.1 Data Sources

GEFCom2014 load-track series [16] provide original cases without added events and the basis for our synthetic event cases. Nord combines national load from ENTSO-E Power Statistics [12] with Nord Pool Urgent Market Messages. PJM uses metered hourly native load from PJM Data Miner 2 [28]. PJM and SPP (SWPP) use EIA-930 balancing-area demand [37] obtained through Catalyst Cooperative PUDL. Both use National Weather Service (NWS) weather alerts from the Iowa Environmental Mesonet archive; PJM also uses PJM Emergency Procedures alerts.

LCL uses UK Power Networks smart-meter consumption and tarifs from the London Datastore [36], with daily tarif notices as events. iFlex provides household load, experiment days, and research price schedules [15]. AEMO uses NEMWeb operational demand [3] and market notices. ERCOT uses its Hourly Load Data Archives [11], news releases, operating condition notices, and NWS weather alerts.

## A.2 Temporal Splits and Window Selection

ID cases are assigned chronologically by the date of the first predicted hour. Table 6 summarizes these dates across retained cases, combining both horizons where used. The ranges describe selected windows without implying continuous coverage. Dates use UTC except for GEFCom2014, which retains its source time axis; ERCOT includes all 85 cases in Table 1. Nord uses origins before 2023 for training, 2023 for validation, and 2024 for testing; PJM and SPP use origins before 2023, in 2023–2024, and from 2025 onward, respectively. LCL and iFlex split the eligible schedule dates chronologically, keeping all household views of a date in the same split. Their validation and test households also appear in training, so these evaluations concern forecasting later dates for known households.

Synthetic variants inherit the split of their original load window. No GEFCom2014 parent window is shared across training, validation, and testing. Nord, PJM, and SPP additionally keep explicitly linked notices and revisions in the same split. For distinct original windows, selected ID forecast targets do not overlap on the same physical load series, including across splits. For later origins, input histories may reuse earlier cases’ target hours once these are observed.

Window selection. All cases use hourly load with the history and forecast lengths in Table 1. GEFCom2014 windows are drawn from a 96-hour origin grid. Nord, PJM, and SPP select windows for coverage of source entities, calendar quarters, and event-description counts; LCL and iFlex balance household and schedule-date cover age. The retained origins therefore need not have uniform spacing. AEMO and ERCOT are reserved for OOD evaluation, with aggregation defined below.

## A.3 Forecast-Origin Information

Inputs contain observed load history, available covariates, and event announcements available at the forecast origin. Forecast-horizon temperatures in GEFCom2014 are weather forecasts available at that origin. LCL tarif notices and iFlex research price schedules provide next-day planned conditions. An event’s efective interval specifies its alignment with the load horizon, separately from the announcement’s availability time. Label-assisted teacher demonstrations use training cases; held-out load labels are used for evaluation.

Table 6: Calendar date ranges by data source and split.
<table><tr><td>Source</td><td>Split</td><td>Date range</td></tr><tr><td rowspan="3">GEFCom2014</td><td>Train</td><td>2005-01-05-2009-10-31</td></tr><tr><td>Val.</td><td>2009-11-08-2010-11-11</td></tr><tr><td>Test</td><td>2010-11-19-2011-11-26</td></tr><tr><td rowspan="3">Nord</td><td>Train</td><td>2015-06-30-2022-12-30</td></tr><tr><td>Val.</td><td>2023-01-01-2023-12-05</td></tr><tr><td>Test</td><td>2024-01-01-2024-11-19</td></tr><tr><td rowspan="3">PJM</td><td>Train</td><td>2004-05-18-2022-12-10</td></tr><tr><td>Val.</td><td>2023-01-01-2024-11-06</td></tr><tr><td>Test</td><td>2025-01-01-2026-07-18</td></tr><tr><td rowspan="3">SPP</td><td>Train</td><td>2015-07-08-2022-12-26</td></tr><tr><td>Val.</td><td>2023-01-01-2024-06-22</td></tr><tr><td>Test</td><td>2025-01-01-2026-07-14</td></tr><tr><td rowspan="3">LCL</td><td>Train</td><td>2013-01-04-2013-11-20</td></tr><tr><td>Val.</td><td>2013-11-21-2013-12-09</td></tr><tr><td>Test</td><td>2013-12-10-2013-12-26</td></tr><tr><td rowspan="3">iFlex</td><td>Train</td><td>2020-02-11-2021-02-12</td></tr><tr><td>Val.</td><td>2021-02-15-2021-02-24</td></tr><tr><td>Test</td><td>2021-03-05-2021-03-25</td></tr><tr><td>AEMO (OOD)</td><td>Test</td><td>2025-08-08-2026-09-24</td></tr><tr><td>ERCOT (OOD)</td><td>Test</td><td>2021-02-12-2026-08-26</td></tr></table>

## A.4 Synthetic Event Construction

Programmes A–F are industrial pause (−0.12, 0), demand response (−0.20, 0.5), commercial closure (−0.10, 0), commissioning (0.12, 0), deferred demand (−0.16, 1) and restart (0.10, 0), respectively; pairs specify $( b , \rho )$ . Table 7 counts forecast-window efects. With duration $d \in \{ 6 , 1 2 \}$ , intensity $q \in \{ 0 . 7 5 , 1 , 1 . 2 5 \}$ and start �,

$$
\delta _ { t } = \left\{ \begin{array} { l l } { b q z _ { t } , } & { s \leq t < s + d , } \\ { - \rho b q \sum _ { u = s } ^ { s + d - 1 } z _ { u } / d , } & { s + d + 2 \leq t < s + 2 d + 2 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{9}
$$

where � is the signed fraction, $\rho$ the return fraction, and � the curve the programme acts on. Independent composition sums parentcurve responses; shared composition applies events successively by start time, breaking ties alphabetically. History-changing families are evaluated separately. Eight matched conditions define

$$
\Delta _ { A B C } = \sum _ { S \subseteq \{ A , B , C \} } ( - 1 ) ^ { 3 - | S | } y _ { S } ,\tag{10}
$$

$$
\mathrm { I E E } _ { 3 } = 1 0 0 \frac { \| \widehat { \Delta } _ { A B C } - \Delta _ { A B C } \| _ { 1 } } { \| y _ { 0 } \| _ { 1 } } .\tag{11}
$$

where $y _ { \emptyset } = y _ { 0 }$ , hats denote predictions, and $\mathrm { I E E } _ { 3 }$ is a percentage.

GEFCom2014 counts each programme once over the union of its direct and recovery impact windows; overlap percentages in Table 7 use cases with multiple relevant events.

Table 7: GEFCom2014 case composition $( L = 9 6 , H = 9 6 )$
<table><tr><td>Split</td><td>N</td><td>0</td><td>H</td><td>I</td><td>1</td><td>≥2</td><td>Overlap</td></tr><tr><td>Train</td><td>640</td><td>105</td><td>15</td><td>80</td><td>250</td><td>190</td><td>130 (68.4)</td></tr><tr><td>Val.</td><td>100</td><td>15</td><td>2</td><td>20</td><td>36</td><td>27</td><td>20 (74.1)</td></tr><tr><td>ID test</td><td>200</td><td>30</td><td>4</td><td>40</td><td>72</td><td>54</td><td>40 (74.1)</td></tr></table>

0: original; H: history-only programmes; I: irrelevant-only; 1/≥2: one/multiple future-relevant programmes. Overlap: count (%).

## A.5 Event Statistics and Evaluation Groups

Table 8 counts input event descriptions, with each announcement or daily price schedule counted once. Two event descriptions overlap when neither is explicitly marked irrelevant and their specified efective intervals intersect for a positive duration within the forecast window.

Table 8: Test-set event descriptions and temporal overlap.
<table><tr><td></td><td></td><td></td><td colspan="4">Event descriptions</td><td></td><td></td></tr><tr><td>Source</td><td>L/H</td><td>N</td><td>0</td><td>1</td><td>2</td><td>≥3</td><td>Overlap</td><td>Incomp.</td></tr><tr><td>ID test</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GEFCom2014 original</td><td>96/96</td><td>30</td><td>30</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>GEFCom2014 synthetic</td><td>96/96</td><td>170</td><td>0</td><td>102</td><td>58</td><td>10</td><td>40 (23.5%)</td><td>0</td></tr><tr><td>Nord</td><td>168/24</td><td>96</td><td>0</td><td>54</td><td>30</td><td>12</td><td>30 (31.2%)</td><td>16</td></tr><tr><td>Nord</td><td>96/96</td><td>24</td><td>0</td><td>15</td><td>5</td><td>4</td><td>3 (12.5%)</td><td>11</td></tr><tr><td>PJM</td><td>168/24</td><td>112</td><td>35</td><td>40</td><td>14</td><td>23</td><td>21 (18.8%)</td><td>37</td></tr><tr><td>PJM</td><td>96/96</td><td>28</td><td>8</td><td>10</td><td>2</td><td>8</td><td>7 (25.0%)</td><td>6</td></tr><tr><td>SPP</td><td>168/24</td><td>48</td><td>22</td><td>14</td><td>5</td><td>7</td><td>12 (25.0%)</td><td>0</td></tr><tr><td>SPP</td><td>96/96</td><td>12</td><td>5</td><td>3</td><td>1</td><td>3</td><td>4 (33.3%)</td><td>0</td></tr><tr><td>LCL</td><td>168/24</td><td>48</td><td>12</td><td>24</td><td>9</td><td>3</td><td>5 (10.4%)</td><td>0</td></tr><tr><td>iFlex</td><td>168/24</td><td>32</td><td>0</td><td>32</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>OOD test</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AEMO</td><td>96/96</td><td>150</td><td>0</td><td>0</td><td>34</td><td>116</td><td>42 (28.0%)</td><td>120</td></tr><tr><td>ERCOT</td><td>168/24</td><td>85</td><td>0</td><td>20</td><td>5</td><td>60</td><td>63 (74.1%)</td><td>0</td></tr></table>

�: test cases in the row. 0/1/2/≥3: case counts by number of event descriptions. Overlap percentages use all � cases. Incomp.: cases containing at least one event description with an incomplete interval. ERCOT description counts cover all 85 cases; the main performance evaluation uses the 71 system-level cases.

Event-count groups. Table 9 groups cases by the number of distinct events whose windows intersect the forecast horizon for a positive duration. PJM and SPP use recorded efective intervals, counting repeated notices of one event once. Both show larger WAPE reductions over direct-only forecasting with multiple events.

Table 9: WAPE by number of events $( L ~ = ~ 9 6 , H ~ = ~ 9 6$ for GEFCom2014; � = 168, � = 24 for PJM and SPP).
<table><tr><td>Dataset</td><td>Event Condition</td><td>N</td><td>Full</td><td>Direct</td><td>No text</td><td>Δ</td></tr><tr><td rowspan="2">GEFCom2014</td><td>Single Event</td><td>72</td><td>6.42</td><td>7.90</td><td>8.20</td><td>-1.48</td></tr><tr><td>Multiple Events</td><td>54</td><td>7.61</td><td>9.50</td><td>10.80</td><td>-1.89</td></tr><tr><td rowspan="3">PJM</td><td>No Event</td><td>35</td><td>2.63</td><td>2.80</td><td>2.41</td><td>-0.17</td></tr><tr><td>Single Event</td><td>45</td><td>3.12</td><td>3.42</td><td>3.57</td><td>-0.30</td></tr><tr><td>Multiple Events</td><td>32</td><td>3.69</td><td>4.31</td><td>4.63</td><td>-0.62</td></tr><tr><td rowspan="2">SPP</td><td>Single Event</td><td>18</td><td>2.74</td><td>2.91</td><td>3.08</td><td>-0.17</td></tr><tr><td>Multiple Events</td><td>8</td><td>3.11</td><td>3.43</td><td>3.70</td><td>-0.32</td></tr></table>

�: cases per group; Full: ElecLoad (Full); Direct: direct-only; No text: without event text. Δ: Full minus Direct before rounding (percentage points).

Event-relevance groups. Case counts for Table 4 are as follows. GEFCom2014: 30 No Event, 72 Single Event, 54 Multiple Events, 40 Irrelevant Events; four history-only cases fall outside these conditions. PJM: 35 No Event, 33 Single Event, 28 Multiple Events, and 16 Irrelevant Events. LCL: 12 cases per condition.

## A.6 Evaluation Aggregation

Reported metrics use the forecast actually delivered for every case, including the recorded fallback after failure. Dataset results average case errors; for AEMO, regional errors are first averaged within each forecast origin, then across origins. The 85 ERCOT cases comprise 71 system-level cases for the main performance evaluation and 14 regional sensitivity cases. The latter are excluded from the main performance metrics. WAPE and NRMSE retain near-zero targets without denominator ofsets; zero denominators are undefined. LCL’s low-consumption windows can yield large percentage errors with equal weight. We checked the reported results for consistency with the experimental code, configurations, and data.

## B Training and Runtime Definitions

## B.1 Reasoning Training

The teacher model is DeepSeek-v4-flash [10]. Reasoning RL uses group relative policy optimization (GRPO) [31] with the reward below and selects its checkpoint by mean validation reward.

Reasoning uses clipped WAPE � and NRMSE �. Let � be the smaller available historical or HGB reference loss, and $[ x ] _ { 0 } ^ { 1 } \ =$ min(1, max(0, �)):

$$
L = 0 . 9 \operatorname* { m i n } ( W , 4 ) + 0 . 1 \operatorname* { m i n } ( N , 4 ) ,\tag{12}
$$

$$
d _ { * } = b _ { * } - L ,\tag{13}
$$

$$
g = d _ { * } / \operatorname* { m a x } ( b _ { * } , 0 . 1 ) ,\tag{14}
$$

$$
B = 0 . 1 \operatorname* { m i n } \left( \left[ \frac { g - 0 . 0 5 } { 0 . 2 0 } \right] _ { 0 } ^ { 1 } , \left[ \frac { d _ { * } - 0 . 0 2 } { 0 . 0 8 } \right] _ { 0 } ^ { 1 } \right) ,\tag{15}
$$

$$
s = \frac { \operatorname* { m e a n } | \hat { y } | } { \operatorname* { m a x } ( \operatorname* { m e a n } | y | , s \mathrm { d } ( y ) , 1 ) + 1 0 ^ { - 8 } } ,\tag{16}
$$

$$
x = \frac { \operatorname* { m a x } | \hat { y } | } { \operatorname* { m a x } ( \operatorname* { m a x } | y | , \mathrm { s d } ( y ) , 1 ) + 1 0 ^ { - 8 } } ,\tag{17}
$$

$$
P _ { s } = - 0 . 2 1 _ { s > 1 0 } \operatorname* { m i n } ( 1 , \log _ { 1 0 } ( s ) / 3 ) ,\tag{18}
$$

$$
P _ { x } = - 0 . 8 \ : 1 _ { x > 1 0 0 } \ : \mathrm { m i n } ( 1 , \log _ { 1 0 } ( x / 1 0 0 + 1 ) ) ,\tag{19}
$$

$$
R _ { \mathrm { r e a s o n i n g } } = \left\{ { \begin{array} { l l } { - L + B + P _ { s } + P _ { x } , } & { { \mathrm { v a l i d ~ o u t p u t } } , } \\ { - 6 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} } \right.\tag{20}
$$

The bonus is zero without a reference. Historical candidates repeat the last value, recent horizon mean, or an autocorrelation-qualified seasonal pattern (threshold 0.5).

## B.2 Agent SFT

Agent SFT minimizes Eq. 21, supervising each call’s current response in its own context.

For a batch D of demonstration episodes, SFT averages valid response-token losses within episodes and then across episodes:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \frac { 1 } { | \mathcal { D } | } \sum _ { e \in \mathcal { D } } \frac { 1 } { | \mathcal { T } | } \sum _ { t \in \mathcal { T } _ { e } } \log \pi _ { \theta } ( a _ { e , t } \mid h _ { e , t } ) ,\tag{21}
$$

where �� is the shared model’s conditional token probability, $\mathcal { T } _ { e }$ indexes all valid supervised response tokens in �, and $a _ { e , t }$ is the target token. Each $h _ { e , t }$ contains the corresponding call’s input and response prefix before �.

## B.3 Agent RL

Agent RL uses clipped policy ratios [30] and KL regularization toward frozen agent SFT weights. Its checkpoint maximizes the mean selected-path validation score across all attempted cases:

$$
R ( e ) = U ( e ) - F ( e ) ,\tag{22}
$$

where � follows Eq. 5, and �(�) equals one for failure to com plete the selected path normally or for fallback delivery, and zero otherwise. Normal completion requires valid outputs from every required stage after any allowed repairs. Scores use the forecasts actually delivered, including fallbacks, and all incurred call costs, including failed attempts.

Role-specific credit. Training uses two independently sampled T1 executions per case, with paired direct and segmented evaluations. T1 compares the utility of its selected path with the other sample’s selected path and with its own alternative path. Initial T2 forecasts retain the same-path utility comparison between samples and add improvement over a last-day repetition baseline. T2 refinement and T4 receive credit for immediate error reduction minus call cost. T3 receives credit for error reduction through the subsequent refinement loop, minus its own and subsequent call costs:

$$
A _ { \mathrm { T 1 } } = U _ { \mathrm { s e l e c t e d } } - U _ { \mathrm { s e l e c t e d } } ^ { \mathrm { s i b l i n g } }
$$

$$
+ \textstyle \frac { 1 } { 2 } I _ { \mathrm { p a i r } } ( U _ { \mathrm { s e l e c t e d } } - U _ { \mathrm { o t h e r } } ) ,\tag{23}
$$

$$
A _ { \mathrm { i n i t , l o c a l } } = 1 0 \frac { \sum _ { t \in S } \bigl ( | y _ { t } - y _ { t } ^ { \mathrm { r e p } } | - | y _ { t } - \hat { y } _ { t } | \bigr ) } { \| y \| _ { 1 } } ,\tag{24}
$$

$$
\begin{array} { r l } { A _ { \mathrm { r e f i n e / T 4 } } = 1 0 I _ { \mathrm { a c c e p t e d } } ( W _ { \mathrm { b e f o r e } } - W _ { \mathrm { a f t e r } } ) } & { } \\ { - \ : 0 . 0 1 C _ { \mathrm { a t t e m p t } } / C _ { \mathrm { r e f } } , } \end{array}\tag{25}
$$

$$
\begin{array} { r } { A _ { \mathrm { T 3 } } = 1 0 I _ { \mathrm { a c c e p t e d } } ( W _ { \mathrm { b e f o r e } } - W _ { \mathrm { p r e T 4 } } ) } \\ { -  0 . 0 1 ( C _ { \mathrm { o w n } } + C _ { \mathrm { s u f f x } } ) / C _ { \mathrm { r e f } } . } \end{array}\tag{26}
$$

Here, � is the utility in Eq. 5; �<sub>selected</sub> and $U _ { \mathrm { o t h e r } }$ refer to the selected and alternative paths, and $U _ { \mathrm { s e l e c t e d } } ^ { \mathrm { s i b l i n g } }$ to the other independent sample’s selected path. $I _ { \mathrm { p a i r } } = 1$ when the T1 attempt is accepted and both paths complete normally. For initial T2 forecasts, � is the assigned segment (the full horizon on the direct path), $y ^ { \mathrm { r e p } }$ repeats the last 24 observed loads, and $\| y \| _ { 1 }$ sums absolute target loads over the full horizon.

� denotes full-horizon WAPE as a fraction. Subscripts before and after denote predictions immediately before and after an update; preT4 denotes the last retained prediction after the refinement loop and before T4. $I _ { \mathrm { a c c e p t e d } } = 1$ when the attempt passes output validation. $C _ { \mathrm { a t t e m p t } }$ and $C _ { \mathrm { { o w n } } }$ count input and output tokens of the current refinement/T4 and T3 attempts, respectively; $C _ { \mathrm { s u f f i x } }$ counts subsequent T2/T3 attempts before T4. The frozen per-case reference $C _ { \mathrm { r e f } }$ uses Agent SFT with a direct plan and one full-horizon prediction, or an identified same-case Reasoning fallback.

After two refinements, T3 must stop and receives no task credit. An invalid T3 attempt incurs its own cost but receives no quality gain or subsequent-call cost. If a call remains invalid after its repair, both attempts receive a −1 output-validity signal, optimized with a separate clipped policy objective.

## B.4 Runtime Settings and Output Handling

Table 10: Runtime settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Agent RL batch; epochs; learning rate</td><td> $1 6 ; ~ 2 ; ~ 1 0 ^ { - 6 }$ </td></tr><tr><td>Train/validation; PPO clip; KL</td><td>1800/300; 0.2; 0.001</td></tr><tr><td>Prompt/response/context tokens</td><td>32768/4096/36864</td></tr><tr><td>Train/validation temperature</td><td>1/0</td></tr><tr><td>Max segments; minimum steps</td><td>4; 6</td></tr><tr><td>Refinements; repairs/call</td><td>2; 1</td></tr><tr><td>Edit steps per patch; total</td><td>4;8</td></tr></table>

Output validation and fallback. ElecLoad checks response structure, numerical finiteness, and forecast length, allowing one repair per invalid call. If execution still fails, it returns the latest complete forecast when available; otherwise, it repeats the last 24 observed loads to cover the horizon. Evaluation uses the delivered forecast, and token costs include failed calls and repairs.

## B.5 Baseline Implementations

The baselines use the same evaluation cases through their supported input interfaces. TimeReasoner uses DeepSeek-v4-flash. CastFSR uses a training-free API coordinator; MemCast uses a training-free variant without visual analysis. The EventTSF adaptation normalizes load history and uses event text. ITA-LF retains its trained checkpoint and constrains decoding to the required output format and horizon length.

## C Additional Analyses

Unless stated otherwise, tables use $L ~ = ~ 9 6 , H ~ = ~ 9 6$ for GEF-Com2014 and $L ~ = ~ 1 6 8 , H ~ = ~ 2 4$ for PJM/LCL. kTok denotes thousands of tokens per case; component and execution ablations measure cost on GEFCom2014.

## C.1 Model Scale and Training Ablations

The variant without Agent training in Table 11 retains ElecLoad’s execution protocol. In Table 11, 4B/9B WAPEs difer by at most 0.13 percentage points.

Table 11: Model scale and training.
<table><tr><td>Setting</td><td>GEFCom2014</td><td>PJM</td><td>LCL</td></tr><tr><td>ElecLoad 0.8B</td><td>8.14</td><td>3.37</td><td>47.36</td></tr><tr><td>ElecLoad 4B</td><td>6.68</td><td>3.13</td><td>41.94</td></tr><tr><td>ElecLoad 9B</td><td>6.63</td><td>3.16</td><td>41.81</td></tr><tr><td>w/o Reasoning init.</td><td>7.42</td><td>3.46</td><td>45.08</td></tr><tr><td>w/o Agent training</td><td>8.05</td><td>3.83</td><td>47.18</td></tr><tr><td>w/o Agent SFT</td><td>7.19</td><td>3.42</td><td>44.28</td></tr><tr><td>w/o Agent RL</td><td>7.06</td><td>3.33</td><td>43.35</td></tr><tr><td>Reasoning: SFT+RL</td><td>7.37</td><td>3.42</td><td>44.32</td></tr><tr><td>Reasoning w/o SFT</td><td>8.16</td><td>3.87</td><td>48.18</td></tr><tr><td>Reasoning w/o RL</td><td>7.68</td><td>3.63</td><td>45.87</td></tr></table>

## C.2 Training-Component Ablations

Table 12 uses the Reasoning SFT+RL and ElecLoad 4B rows of Table 11 as references (ElecLoad: 18.7k tokens per GEFCom2014 case). Removing the numerical term drops −� while � still uses �; removing the format penalty assigns invalid outputs reward 0. Local quality credit comprises quality terms for initial T2 forecasting, T2 refinement, T3, and T4. On GEFCom2014, cost credit saves 8.9k tokens per case for 0.08 points higher WAPE, as shown in Table 12.

(a) No relevant event  
Table 12: Training-component ablations.
<table><tr><td>Group</td><td>Removed component</td><td>GEFCom2014</td><td>PJM</td><td>LCL</td><td>kTok</td></tr><tr><td rowspan="5">Reasoning</td><td>Numerical term</td><td>8.13</td><td>3.83</td><td>48.01</td><td>一</td></tr><tr><td>Improvement bonus</td><td>7.52</td><td>3.53</td><td>44.97</td><td>1</td></tr><tr><td>Scale penalty</td><td>7.62</td><td>3.66</td><td>47.27</td><td>1</td></tr><tr><td>Extreme penalty</td><td>7.56</td><td>3.52</td><td>46.87</td><td>一</td></tr><tr><td>Format penalty</td><td>7.45</td><td>3.52</td><td>45.03</td><td>一</td></tr><tr><td rowspan="5">Agent</td><td>Final accuracy credit</td><td>7.48</td><td>3.58</td><td>45.72</td><td>17.7</td></tr><tr><td>Cost credit</td><td>6.60</td><td>3.12</td><td>41.87</td><td>27.6</td></tr><tr><td>Contract credit</td><td>7.05</td><td>3.36</td><td>44.12</td><td>21.39</td></tr><tr><td>Local quality credit</td><td>7.15</td><td>3.43</td><td>44.65</td><td>19.11</td></tr><tr><td>Paired route credit</td><td>7.07</td><td>3.39</td><td>43.93</td><td>22.21</td></tr></table>

## C.3 Execution Variants

In Table 13, the default allows four segments, two refinements and eight edited steps. The four forced segments retain event-guided boundaries for $H \ : = \ : 9 6$ and contain six steps each for � = 24; forced refinements disable stopping until the specified count. Firstsegment refinement always selects $S _ { 1 }$ under the same budget. Regeneration repeats all segments from original inputs and existing reports without feedback. On GEFCom2014, the default workflow has lower WAPE and cost than refining all segments once or regenerating them. Four refinements slightly lower WAPE at higher cost.

Table 13: Execution variants.
<table><tr><td>Setting</td><td>GEFCom2014</td><td>PJM</td><td>LCL</td><td>kTok</td></tr><tr><td>Default</td><td>6.68</td><td>3.13</td><td>41.94</td><td>18.7</td></tr><tr><td>At most 2 segments</td><td>6.95</td><td>3.26</td><td>42.05</td><td>15.9</td></tr><tr><td>At most 3 segments</td><td>6.73</td><td>3.14</td><td>41.98</td><td>17.5</td></tr><tr><td>Force 4 segments</td><td>7.06</td><td>3.33</td><td>44.93</td><td>26.1</td></tr><tr><td>Refine first segment</td><td>6.98</td><td>3.35</td><td>43.46</td><td>18.5</td></tr><tr><td>Force 1 refinement</td><td>6.85</td><td>3.21</td><td>42.73</td><td>17.3</td></tr><tr><td>Force 2 refinements</td><td>6.73</td><td>3.22</td><td>42.24</td><td>21.4</td></tr><tr><td>Allow 4 refinements</td><td>6.63</td><td>3.15</td><td>41.93</td><td>22.8</td></tr><tr><td>Refine all segments once</td><td>6.73</td><td>3.18</td><td>42.08</td><td>26.1</td></tr><tr><td>Regenerate all segments</td><td>6.84</td><td>3.29</td><td>42.83</td><td>29.5</td></tr><tr><td>Edit limit: 4 steps</td><td>6.73</td><td>3.17</td><td>42.06</td><td>18.2</td></tr><tr><td>Edit limit: 12 steps</td><td>6.72</td><td>3.18</td><td>42.05</td><td>19.5</td></tr></table>

## C.4 Irrelevant-Text Efects

We measure how irrelevant event text changes forecasting error by adding it to GEFCom2014 cases while keeping the load history, covariates, target load, and any existing relevant event descriptions fixed. For each method, we report its mean WAPE after the addition minus its mean WAPE before, in percentage points, on the same cases; positive values mean higher error. Figure 6(a) adds the text to cases without relevant events; Figure 6(b) adds it alongside existing relevant event descriptions. Among the methods that use event text, ElecLoad has the smallest increase in both settings.

![](images/8ec2bb965ec378c32bbeedab7a7a8bd825f2a7e3fbd7afb3b374c5188adba857.jpg)

(b) Relevant events  
![](images/28b78dfcefae7d6a221bbec46fc933d8a6f6715bb7e4ff7545237043d8f26613.jpg)  
Figure 6: Mean WAPE change after adding irrelevant text on GEFCom2014 (pp: percentage points).

The irrelevant-text analysis in Table 14 uses a separate set of 48 cases per cell.

Table 14: Irrelevant-text efects (� = 96, � = 96).
<table><tr><td></td><td colspan="2">No events</td><td colspan="2">Relevant events</td></tr><tr><td>Type</td><td>E</td><td>D</td><td>E</td><td>D</td></tr><tr><td>Administrative</td><td>+0.012</td><td>-0.023</td><td>+0.045</td><td>-0.015</td></tr><tr><td>Wrong target</td><td>+0.032</td><td>-0.013</td><td>+0.061</td><td>-0.011</td></tr><tr><td>Outside horizon</td><td>-0.005</td><td>+0.027</td><td>-0.005</td><td>-0.007</td></tr><tr><td>Cancelled</td><td>+0.028</td><td>+0.057</td><td>+0.045</td><td>-0.086</td></tr><tr><td>Superseded</td><td>-0.032</td><td>-0.033</td><td>-0.043</td><td>-0.043</td></tr></table>

E: ElecLoad; D: direct-only.  
Values: paired after-minus-before WAPE changes (percentage points).

## C.5 Paired WAPE Diferences

Table 15 reports paired WAPE diferences between ElecLoad and direct-only forecasting; negative values favor ElecLoad. For GEF-Com2014 and PJM, the 95% intervals are below zero; the LCL interval includes zero.

Table 15: Paired WAPE diferences.
<table><tr><td>Source</td><td>∆WAPE</td><td>95% interval</td><td>kTok</td><td>P99</td></tr><tr><td>GEFCom2014</td><td>-1.14</td><td>[-1.31, -0.98]</td><td>18.7</td><td>16.18</td></tr><tr><td>PJM</td><td>-0.35</td><td>[-0.58, -0.12]</td><td>15.3</td><td>8.50</td></tr><tr><td>LCL</td><td>-0.97</td><td>[-1.95, +0.07]</td><td>12.7</td><td>79.88</td></tr></table>

ΔWAPE: ElecLoad minus direct-only (percentage points).  
P99: 99th percentile of ElecLoad case WAPE (%).