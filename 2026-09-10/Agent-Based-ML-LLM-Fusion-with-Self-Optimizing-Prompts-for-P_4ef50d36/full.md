# Agent-Based ML-LLM Fusion with Self-Optimizing Prompts for Plateau Weather Alerts

1<sup>st</sup> Shuai Yan

Chengdu Jincheng College

2<sup>nd</sup> Yang Xu

Chengdu Jincheng College

3<sup>rd</sup> Shan He<sup>∗</sup>

Chengdu Jincheng College

College of Computer and Software

Chengdu, Sichuan, China

College of Computer and Software

yanshuai1@cdjcc.edu.cn

Chengdu, Sichuan, China

xuyang88@cdjcc.edu.cn

College of Computer and Software

Chengdu, Sichuan, China

heshan@cdjcc.edu.cn

Abstract—To address insufficient contextualization, weak generalization, and poor scenario adaptation in tourism meteorological services, we propose SmartWeatherAgent—a unified three-stage architecture integrating intent recognition, hazard prediction, and reasoning-enhanced generation. The system fuses rule-based methods with large language models to parse queries at multiple granularities and employs a LightGBM model enriched with highland-specific features (e.g., wind speed abruptness rate), achieving an F1-Macro score of 0.605 with 1.60 ms latency on high-wind, precipitation, and low-temperature events. A 12-round micro-step prompt self-optimization loop boosts the composite warning quality score S<sub>final</sub> from 4.2 (B01) to 8.9 (B12, +112%). Key improvements include a sharp rise in B08 from data source citation (6.5 → 8.5), sustained high performance in B10 via physical mechanism explanation, and a peak scientific rigor score of 9.2 in B12 through explicit uncertainty statements. The system autonomously generates structured warnings that integrate causal mechanisms, spatiotemporal evolution, quantitative evidence, regulatory references, and confidence statements—enhancing professional depth, logical rigor, and scientific soundness, and advancing meteorological services toward proactive perception, explainable decision-making, and intelligent agency.

Index Terms—Tibetan tourism; Large Language Models; Machine Learning; Prompt Engineering; Iterative Ablation Study

## I. INTRODUCTION

Highland tourism meteorology exhibits high dynamism, strong spatial heterogeneity, and scenario dependence, posing significant challenges to the accuracy and timeliness of existing service systems. Existing methods—such as static rule-based systems or generic large language models—commonly exhibit high response latency, poor scenario adaptation, weak context awareness, and the absence of a self-evolution mechanism, thereby failing to meet the personalized decision-making needs of tourists [1]. To address these limitations, we propose SmartWeatherAgent—a framework that, for the first time, embeds a prompt self-adaptation mechanism into the core of an intelligent meteorological agent, inspired by the “generationas-reasoning” paradigm of large language models, to establish a unified architecture comprising intent recognition, hazard prediction, and reasoning-enhanced generation [2]. Through a closed-loop prompt refinement strategy, the system achieves: (i)

## II. METHODOLOGY

fine-grained deconstruction of user query intents; (ii) efficient short-range nowcasting modeling of highland extreme weather events [3]; and (iii) dynamic, context-aware generation of warning content—where each interaction drives prompt selfevolution.

## A. Intent Recognition Module

Combines regular expression matching with a large language model (Qwen3) to classify user intents into six categories, including “simple inquiry,” “hazard alert,” and “family travel” [1]. Regular expressions are applied for initial filtering, while the LLM resolves contextual ambiguities, thereby enhancing system robustness [2].

## B. Hazard Prediction Module

Targeting three high-impact weather events prevalent in plateau regions—strong winds, precipitation, and low temperatures—this work proposes a short-term nowcasting model based on LightGBM, enhanced with plateau-specific features: Absolute wind speed difference: $\Delta W ( t ) = | W ( t ) - $ $W ( t - 1 ) |$ ; Precipitation burst indicator: $1 \{ { P } ( t ) > { P } _ { 9 5 } \}$ ;Gust ratio: $R _ { g } \dot { ( } t ) \ = \ G ( t ) / W ( t ) ;$ ; Temporal encoding:ϕ<sub>sin</sub>(t) = sin $\begin{array} { r } { \left( \frac { 2 \pi H } { 2 4 } \right) , \ \phi _ { \mathrm { c o s } } ( t ) = \cos \left( \frac { 2 \pi H } { 2 4 } \right) } \end{array}$

Additional features include rolling statistics (e.g., 6-hour moving average of temperature), threshold-based binary features (e.g., “diurnal temperature range $> 1 0 ^ { \circ } \mathrm { C } ^ { \prime \prime } )$ , and quantilebased extremeness markers (e.g., “temperature below the 5th percentile”) [3]. Here, W(t) denotes the hourly mean wind speed, G(t) the gust speed, P(t) the hourly precipitation, and H the local hour of day. Collectively, these features form a multidimensional input representation that supports real-time hazard prediction and provides structured grounding for the subsequent generation module.

## C. Prompt Self-Adaptation Module

This module establishes a closed-loop pipeline of “generation → evaluation → optimization” to drive the large language model through 12 rounds of micro-step iterative refinement for prompt self-adaptation [4]. The overall output quality is quantified by a composite score:

$$
S _ { \mathrm { f i n a l } } = 0 . 3 5 \cdot S _ { \mathrm { s e m a n t i c } } + 0 . 3 0 \cdot S _ { \mathrm { l o g i c a l } } + 0 . 3 5 \cdot S _ { \mathrm { s c i e n t i f i c } } ,\tag{1}
$$

Each component is evaluated as follows:

Assesses whether the output progressively achieves the following sequence: phenomenon description → single-cause attribution → multi-factor coupled mechanisms → regional risk differentiation → defense measures linked to underlying physical processes. Each successful transition yields a 2–3 point increment; scores of 9–10 require explicit support from climatic context and the use of nested, compound causal expressions [5].

Evaluates the completeness of the reasoning chain: “meteorological system trigger → temporal evolution → impact propagation → targeted mitigation advice.” A single complete chain earns 7–8 points; only systems exhibiting parallel, nested reasoning chains with precise and unbroken mapping to actionable measures achieve 9–10 points.

Computed as the sum of five dimensions: clarity of quantitative metrics, data traceability, accuracy of regulatory citations, completeness of uncertainty statements, and terminological rigor. Any missing dimension incurs a penalty; outputs containing scientific inaccuracies are capped at a maximum score of 3.

## III. EXPERIMENTS

## A. Model Selection and Hyperparameter Optimization

This experiment quantifies the performance of high-altitude meteorological hazard prediction models through a twodimensional evaluation framework: inference latency and overall classification performance (F1-Macro, denoted as F1-M). LightGBM was compared against Random Forest, Gradient Boosting, XGBoost, and CatBoost using an hourly meteorological observation dataset from Lhasa as the benchmark.

a) Dataset and Feature Engineering: The experiment uses historical hourly meteorological data for Lhasa provided by the VisualCrossing platform, spanning from January 1, 2024, to May 21, 2025, comprising 12 168 records. Core variables include temperature ${ } ^ { ( \circ } \mathrm { C } )$ , hourly precipitation (mm), mean wind speed and gust speed $( \mathrm { k m h ^ { - 1 } } ) .$ , UV index, and visibility (km). Missing values account for less than 2.1 % of the data and are imputed using linear interpolation, leveraging the temporal continuity of the time series [4]. The dataset is split chronologically into training and test sets in a 7:3 ratio.

Significant class imbalance is observed: in the training set, normal weather accounts for 91.5 %, while strong wind and precipitation constitute only 1.0 % and 2.5 %, respectively [6], highlighting the challenge posed by the low frequency of extreme events on the plateau for minority-class recognition.

To capture the rapid evolution and nonlinear dynamics of meteorological hazards, six categories of derived features are engineered: The engineered features include: 3-hour and 6-hour rolling window statistics (mean and standard deviation); the absolute first-order difference of wind speed; the gust ratio (gust speed divided by mean wind speed); a binary indicator for diurnal temperature range exceeding $1 0 ^ { \circ } \mathrm { C } ;$ sin/cosine-encoded cyclical features for hour-of-day and month; and extreme-event indicator variables (e.g., set to 1 if hourly precipitation exceeds the 95th percentile of the training distribution). The final feature vector has a dimensionality of 38.

![](images/bffc900080a7972e58834fb6ca0dfdc4d438ca518855a59210dd18de353677d1.jpg)  
Fig. 1: Visualization of Hyperparameter Optimization Process

b) Experimental Design: All models undergo hyperparameter tuning via Bayesian optimization, with the objective function defined as the F1-Macro score under 5- fold time-series cross-validation (TimeSeriesSplit) to prevent temporal information leakage caused by random data splitting. The optimization search space covers common hyperparameters: The following hyperparameter ranges are explored: n\_estimators $\in [ 1 0 0 , 5 0 0 ]$ , learning\_rate ∈ [0.01, 0.3], max\_depth ∈ [3, 10], and subsample ∈ [0.5, 1.0] These ranges are aligned with official recommendations from mainstream gradient boosting frameworks and recent literature on meteorological forecasting. The results are shown in the figure1.

This study constructs hazard labels based on hourly meteorological observations from Lhasa spanning 2024–2025. Informed by the climatic characteristics of the Tibetan Plateau and the empirical data distribution, three high-impact weather events are defined as follows: Strong wind is defined as hourly mean wind speed ≥ 20 km $\mathrm { h } ^ { - 1 }$ ; low temperature as hourly temperature $\leq - 5 ^ { \circ } \mathrm { C } ;$ and precipitation as hourly precipitation $> 0 . 1$ mm.

Statistical analysis reveals a pronounced class imbalance in the training set: 95.0 % normal weather, 3.5 % low-temperature events, 1.2 % strong-wind events, and only 0.3 % precipitation events.

Each model undergoes 25 optimization iterations, approximating a local optimum under constrained computational budgets [7]. To holistically evaluate the practical utility of models in plateau meteorological hazard warning, we define a composite performance score S to quantify their recognition capability across critical hazard categories:

$$
S = 0 . 3 \cdot F \mathrm { 1 - M a c r o } + 0 . 2 \cdot F 1 _ { g } + 0 . 2 \cdot F 1 _ { r } + 0 . 3 \cdot F 1 _ { c }\tag{2}
$$

where $F 1 _ { g } , F 1 _ { r } .$ , and $F 1 _ { c }$ denote the F1 scores for strong wind, precipitation, and low temperature, respectively.

Note: The weighting scheme reflects domain considerations—although strong wind and precipitation are sparse in occurrence, they entail high risk and are thus assigned equal weight (0.2). Low-temperature events, while moderately frequent, also pose significant hazards and consequently receive the highest weight (0.3) among the specific hazard categories.

TABLE I: Performance Comparison of Different Models
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>F1-g</td><td rowspan=1 colspan=1>F1-r</td><td rowspan=1 colspan=1>F1-c</td><td rowspan=1 colspan=1>F1-M</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>Lat</td></tr><tr><td rowspan=1 colspan=1>LGBM</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>1.60</td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>1.56</td></tr><tr><td rowspan=1 colspan=1>CB</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>1.58</td></tr><tr><td rowspan=1 colspan=1>XGB</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>10.32</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>63.72</td></tr></table>

c) Experimental Results and Analysis: As shown in Figure 2, the experimental results demonstrate that the proposed LightGBM model significantly outperforms all baseline methods in both overall warning capability (S) and critical hazard event recognition performance (F1 scores). As shown in Table 1, LightGBM achieves the highest composite score $( S = 0 . 5 5 )$ , outperforming XGBoost and Gradient Boosting .

Regarding extreme weather recognition, LightGBM attains an F1 score of 0.50 for the precipitation category (note that this metric’s stability is limited due to sparse samples). For lowtemperature events, the F1 score reaches 0.77, markedly outperforming all baselines. Although its F1 score for strong wind events (0.17) is slightly lower than the best value (0.18 achieved by gradient boosting), the difference is minimal, indicating that LightGBM maintains strong recognition capability for high-risk, sparse events while achieving better class balance. Additionally, LightGBM’s inference latency is merely 1.60 ms, significantly outperforming XGBoost and Random Forest, thereby meeting the stringent millisecond-level real-time warning requirements of plateau tourism scenarios.

This superior performance primarily stems from Light-GBM’s unique combination of histogram-based efficient feature splitting and adaptive learning strategies for sparse hazard samples, effectively mitigating the challenges posed by the dominance of normal weather samples (>91 %) and the sporadic, low-frequency nature of hazardous events (e.g., strong winds account for only 1.0 %) in plateau meteorological data. Overall, the method achieves a high accuracy of 97 % while maintaining relatively optimal class-balanced recognition capability (F1-macro = 0.61), particularly demonstrating robust performance on low-temperature and strong-wind events. By keeping inference latency within 2 ms, it provides a solid technical foundation for a high-reliability, low-latency, endto-end warning system for meteorological hazards in plateau tourism.

## B. Self-Optimization Experiment of Meteorological Warning Prompts via Micro-Step Iteration

To validate the efficacy of fine-grained prompt iterative optimization in approximating provincial warning standards and enhancing multi-dimensional performance, we implement a “generation–evaluation–optimization” closed-loop framework in a high-altitude severe convective scenario. The initial prompt, being unstructured and lacking domain-specific constraints, yielded outputs missing critical elements—namely warning levels, quantified metrics, protective actions, and regulatory justification—thereby deviating markedly from operational norms [8]. Over the course of $1 2 \times 5$ rounds of micro-step refinement, we progressively introduced hierarchical constraints (“temporal → spatial → mechanistic → uncertainty”) to systematically evaluate the prompt’s evolution in terms of semantic depth, logical coherence, and scientific rigor, thereby aligning with the paradigms of dynamic prompt engineering and phased objective scheduling [9].

![](images/109cd3b57d019ee3810936484297ac0edc029485a80bb6107767ec3843987317.jpg)  
Fig. 2: Performance Comparison

a) Experimental Design: A controlled micro-step design was adopted: the B01 baseline used informal prompts, yielding outputs that deviated significantly from the standard template, whereas the adaptive group employed a three-layer closed-loop architecture. The generation layer fed simulated meteorological features into Qwen-Max; the evaluation layer scored outputs (0.0–10.0) across semantic depth, logical coherence, and scientific rigor, with diagnostic feedback; and the optimization layer refined prompts in stages—early (B01–B04) added core warning elements (e.g., alert level, area, precautions); middle (B05–B08) emphasized quantified metrics and data traceabil ity; and late (B09–B12) prioritized mechanistic explanations and decision-support capabilities. This staged prompt selfreconstruction integrates incremental engineering with objective scheduling, using the three-dimensional scores to compute a composite alignment metric against the Technical Regulations on the Issuance of Meteorological Disaster Warning Signals [10].

b) Experimental Results and Performance Analysis: As shown in Figure 3, five-round averages show semantic depth rising from 2.0 (B01) to 8.5 (B12), with a +1.5 jump at B10 (“explain physical mechanisms”); logical coherence steadily increased from 7.5 to 9.0, driven by B11 (“construct temporal evolution chain”); and scientific rigor surged from 3.0 to 9.2, marked by a +2.2 gain at B08 (“cite data sources”) and a peak at B12 (“add uncertainty statements”) after a dip to 7.0 at B11. The composite score $S _ { \mathrm { f i n a l } }$ improved from 4.2 to 8.9 (+112%), first exceeding 8.0 at B08 (6.5 → 8.5).

![](images/27ee8a47d48ba3fe625278573bca34deaf0166ed0f5d5f8091b518cd09455df6.jpg)  
Fig. 3: prompt optimization scores

During the early phase (B01–B06), $S _ { \mathrm { f i n a l } }$ rose gradually from 4.2 to 7.5 as essential warning elements were incorporated. In the mid phase (B07–B09), introducing quantified variables at B07 temporarily reduced semantic depth to 4.7, causing $S _ { \mathrm { f i n a l } }$ to dip to 6.5; B08 recovered performance via enhanced data traceability. In the late phase (B10–B12), coordinated optimization stabilized $S _ { \mathrm { f i n a l } }$ at 8.0–8.9, with B12 achieving a balanced high score (semantic: 8.5, logical: 9.0, scientific: 9.2). This demonstrates that integrating mechanism explanation, temporal evolution chains, and uncertainty statements effectively overcomes model limitations, unifying professional depth, structural rigor, and scientific credibility. All metrics showed standard deviations < 0.5, confirming trajectory stability and reproducibility.

## IV. DISCUSSION AND LIMITATIONS

Although the proposed method performs well in intelligent weather warning tasks for highland tourism cities, several limitations remain. First, the training and validation data are limited to Lhasa and are not representative of other highland cities, limiting the model’s geographic generalization capability. Second, although the prompt optimization over 12 rounds achieves end-to-end autonomous refinement—e.g., B08 automatically incorporating data provenance and B12 generating statements of uncertainty—the evolutionary process remains constrained by predefined evaluation dimensions and a staged framework. It lacks the ability to openly perceive emerging warning needs and to structurally self-reconfigure, thereby limiting the system’s sustained adaptability in dynamic, complex scenarios. Third, the system is trained solely on historical observational data and has not been integrated with real-time operational meteorological data streams; consequently, its robustness and real-world effectiveness under challenging conditions—such as communication outages, sensor noise, or extremely rare events—require validation through field deployment.

## V. CONCLUSION

This paper presents SmartWeatherAgent, an end-to-end intelligent weather service framework for short-range nowcasting in highland tourism, integrating intent recognition, a lightweight high-impact weather predictor, and LLM-based generation. Its three-stage, feedback-driven architecture—generation, evaluation, and optimization—iteratively refines warning messages to enhance professionalism, structural integrity, and alignment with meteorological standards. The highland-optimized predictor accurately detects key hazards while remaining sensitive to sparse precipitation, ensuring reliability and timeliness in complex terrain. The system’s modular, self-evolving, and scenario-adaptive design makes it applicable beyond tourism— to broader public safety and emergency response contexts requiring real-time awareness. Future work will integrate realtime meteorological data, support multimodal interaction, and develop a meta-prompt-driven self-reflective optimizer to boost generalization, interaction naturalness, and output credibility

## REFERENCES

[1] Shao, S. and Xiao, C. (2024) A Data Enhancement Method for Non-Autoregressive Data Models Based on Joint Multi-Intent Detection and Slot Filling. In: International Conference on Electronics and Devices, Computational Science (ICEDCS), Marseille, France. pp. 504–509.

[2] Arumuganainar, A., Kushal and P, A. D. (2025) Beyond Traditional ML: LLM-Powered Rule-Based Hydraulic Pump Fault Diagnosis. In: 1st International Conference on AIML-Applications for Engineering & Technology (ICAET), Pune, India. pp. 1–7.

[3] Samantaray, A. K., Mahapatra, K., Kabi, B. and Routray, A. (2015) A Novel Approach of Speech Emotion Recognition with Prosody, Quality and Derived Features Using SVM Classifier for a Class of North-Eastern Languages. In: IEEE 2nd International Conference on Recent Trends in Information Systems (ReTIS), Kolkata, India. pp. 372–377.

[4] Ifthaker Hamim, A. M. A., Hossen, M. S., Ahamed, F. and Ifty, R. A. (2025) ”AdaptPrompt: A Framework for Adaptive and Efficient Prompt Engineering in Large Language Models.” In: 2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN), Rangpur, Bangladesh. pp. 1–6.

[5] Li, R., Yu, H., Du, K., Xiao, Z., Yan, B. and Yuan, Z. (2023) ”Adaptive Semantic Fusion Framework for Unsupervised Monocular Depth Estima tion.” In: 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece. pp. 1–5.

[6] Basnayake, B. R. P. M. and Chandrasekara, N. V. (2024) Assessing the Performance of Feedforward Neural Network Models with Random Data Split for Time Series Data: A Simulation Study. In: International Research Conference on Smart Computing and Systems Engineering (SCSE), Colombo, Sri Lanka. pp. 1–6.

[7] Xu, H., Li, R. and Chen, Q. (2025) Research on Deep Neural Network Hyperparameter Optimization Method Based on Improved Tree Seed Algorithm. In: IEEE 7th International Conference on Communications, Information System and Computer Engineering (CISCE), Guangzhou, China. pp. 919–922.

[8] Leung, J. and Shen, Z. (2024) Prompt Engineering for Curriculum Design. In: 4th International Conference on Educational Technology (ICET), Wuhan, China. pp. 97–101.

[9] Zhong, J., Tang, D., Gu, M., Xie, M., Tao, Z. and Zhang, Z. (2025) ”GradPromptOpt: An Enhanced Prompt Optimization Method to Improve Performance of LLMs.” In: 2025 8th International Symposium on Big Data and Applied Statistics (ISBDAS), Guangzhou, China. pp. 716–720.

[10] Khan, I. (2024) ”Your Future in Prompt Engineering.” In: The Quick Guide to Prompt Engineering: Generative AI Tips and Tricks for ChatGPT, Bard, Dall-E, and Midjourney, Wiley. pp. 445–460.