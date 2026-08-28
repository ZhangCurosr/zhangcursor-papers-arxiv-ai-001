# A Multi-Modal AI Framework for Real-Time Queue Prediction, Management and Optimisation in Intelligent Border Control Systems

Varvara Mama <sup>†</sup> , Eleni Veroni<sup>†</sup> , Nikolaos Kapsalis <sup>§</sup> , Christos D. Nikolopoulos <sup>†§</sup> , Member, IEEE and Anargyros T. Baklezos <sup>†§</sup> , Member, IEEE.

<sup>†</sup>Department of Electronic Engineering, Hellenic Mediterranean University, Chania, Greece, Crete, Greece

<sup>§</sup>School of Electrical and Computer Engineering, National Technical University of Athens, Athens, Greece, Athens, Greece Emails: ncapsalis@gmail.com, {ddk345, ddk187}@edu.hmu.gr, {abaklezos, cnikolo }@ hmu.gr

Abstract—In the present work an efficient border control management procedure is proposed. Compared to operational queue management systems, whose operations are based on mostly static data, the proposed work takes into account dynamic traffic conditions, thus enabling optimal performance, even in cases of uncertainty. To this end, we are proposing a multimodal Artificial Intelligence (AI) framework, tailored to the needs of border control systems, which enables real-time queue prediction, management and resource optimization. The novel proposed approach integrates heterogeneous data sources and presents them through a unified representation by employing Long Short-Term Memory (LSTM) networks for queue forecasting. Furthermore, it leverages Model Predictive Control (MPC) and scheduling optimization to derive actionable control policies, which in turn can be presented to border control officers. The proposed work has been evaluated using synthetic data simulating realistic traffic. The evaluation results demonstrate that the proposed method reduces queue prediction error by up to 35% and average waiting time by 30%. Accordingly, the average throughput increases by nearly 20%, compared to ARIMA and rule-based methods. The above mentioned results show the effectiveness and efficiency of combining AI architectures with optimization techniques for proactive and adaptive border traffic management.

Index Terms—Data Fusion, AI/ML modelling, real-time systems, Queue prediction, intelligent transportation systems, border control, LSTM, traffic optimisation.

## I. INTRODUCTION

Current border control systems are nowadays faced with an ever-increasing number of people crossing borders. This fact may be attributed to a multitude of reasons, such as globalization, tourism and migration flows. In Europe, in particular, recent reports from European agencies suggest that border traffic remains highly dynamic in nature, on the one hand, and also rapidly fluctuating. This may be attributed to a number of factors, such as geopolitical, economic, and seasonal factors [1]–[3]. As a result, operational border officials are required to handle an ever-increasing number of vehicles and passengers, while at the same time ensuring security protocols and compliance with legal and regulatory requirements are satisfied. The aforementioned requirements result in a trade-off between speed and security. On the one hand, border control officers must minimize waiting times and congestion, while on the other, they must anticipate and be prepared for everincreasing sophisticated threats. Such approaches may include, but are not limited to, biometric verification and risk-based screening [4], which in turn are time-consuming and thus have a negative impact on waiting times. The increase in waiting time is also amplified by the introduction of sophisticated border management systems, such as Entry/Exit System (EES) [3], [4], as suggested by recent work [2]. In general, border queue management is a stochastic and non-stationary problem, with passengers’ and vehicles’ arrival rates showing temporal variability and uncertainty. Traditional approaches to the aforementioned problems strongly rely on statistical models and rule-based decision derived from historical data [5]–[8]. These methods are limited in their capability to adapt to real-time fluctuations. Furthermore, they are not capable of integrating and making use of heterogeneous data sources. As a result, they often fail to address high variability in passenger and vehicle flows. Recent works employing Machine Learning (ML) and Deep Learning (DL) techniques have been employed to predict traffic and intelligent transportation systems (ITS). Recurrent neural networks, particularly Long Short-Term Memory (LSTM) models, have been successfully applied to capture temporal variations in traffic flow data [8]–[10]. More advanced approaches show increased accuracy by employing spatio-temporal modelling and graph-based learning [11]– [13]. However, the majority of the above mentioned works focus on urban traffic networks. Since border environments have unique characteristics compared to urban environments, these works fail to address inherent characteristics of border control systems, such as bursty arrivals, multi-stage inspection processes, and integration with existing security systems.

Actionable decisions are based on efficient and effective prediction, optimization and control strategies. In particular, Model Predictive Control (MPC) has been used to optimize control policies under dynamic conditions [14]–[16]. Similarly, scheduling and resource allocation methods have been applied to improve system efficiency in transportation and logistics [17]. Recent studies have also explored reinforcement learning for adaptive traffic control [17], [18]. Nevertheless, existing approaches typically treat prediction and control as separate problems, limiting their effectiveness in real-time operational settings. Current border control systems, as mentioned above, cannot integrate data from multiple sources. Several data fusion techniques have been proposed in the literature [21]–[24]. These approaches enable the integration of large heterogeneous data, including vehicle and preregistration information, and sensor and environmental data. However, these techniques have not been yet applied to a border control context. Thus, the need for a unified framework capable to combine and pair data from multiple sources with predictive and optimization capabilities is imperative. To address these challenges, this paper proposes a comprehensive framework for intelligent queue prediction, management, and optimization tailored for border control systems. The proposed approach employs data fusion, deep learning-based prediction, and optimization capabilities within a unified architectural framework. The proposed approach employs an LSTM-based model in order to identify temporal patterns in traffic data. Furthermore, a Model Predictive Control (MPC) layer is used to enable real-time efficient resource allocation and traffic flow. The main contributions of this work include the formulation of the border queue management problem as a stochastic system. It is enhanced with prediction and optimization capabilities. Secondly, we develop a data fusion framework that leverages heterogeneous inputs, such as real-time sensor and environmental data. Finally, the proposed approach is compared to baselines, achieving significant improvements in prediction accuracy and throughput by minimizing waiting times. The remainder of this paper is organised as follows. Section II reviews related work in traffic prediction and optimization. Section III presents the proposed system model and methodology. Section IV describes the experimental setup, while Section V reports the results and analysis. Finally, Section VI concludes the paper and outlines future research directions.

## II. INTELLIGENT QUEUE PREDICTION, MANAGEMENTAND OPTIMISATION

## A. Problem Definition and Mathematical Formulation

The proposed system includes a predictive component, responsible for traffic variables estimation. These variables include vehicle arrival times, queue lengths and temporal vehicle density distributions. In order to efficiently and accurately predict the aforementioned variables, ML models are employed. These models are trained taking into account temporal and environmental factors. Moreover, temporal features – including time of day, seasonality and environmental conditions – are also taken into account during the training phases, resulting in enhanced prediction accuracy. Prediction accuracy is further enhanced by employing periodic retraining capabilities and adaptation mechanisms. These factors ensure that the proposed models remain accurate even in cases of unforeseen events or abnormal traffic patterns. The complete system architecture is depicted in Fig. 1

Efficient queue management at border crossing points can be modelled as a dynamic multi-server queuing system. Let $\lambda ( t )$ denote the arrival rate of vehicles at time t, and $\mu _ { k } ( t )$ the service rate of lane k. The queue length $Q ( t )$ evolves as:

$$
Q ( t + 1 ) = \operatorname* { m a x } \left( Q ( t ) + A ( t ) - S ( t ) , 0 \right)\tag{1}
$$

where $A ( t ) \sim \lambda ( t )$ represents arrivals and:

$$
S ( t ) = \sum _ { k = 1 } ^ { K ( t ) } \mu _ { k } ( t )\tag{2}
$$

is the total service capacity.

The optimisation objective is to minimise congestion:

$$
\operatorname* { m i n } _ { \mathbf { u } ( t ) } \mathbb { E } \left[ \sum _ { t = 0 } ^ { T } Q ( t ) \right]\tag{3}
$$

subject to:

$$
K ( t ) \le K _ { \operatorname* { m a x } } , \quad \mu _ { k } ( t ) \le \mu _ { \operatorname* { m a x } }\tag{4}
$$

where ${ \bf \delta u } ( t )$ is a custom control point, i.e., the corresponding lane allocation.

## B. System Architecture and Data Representation

The system operates over a multi-modal input space:

$$
\mathbf { x } _ { t } = \left[ x _ { t } ^ { ( v e h ) } , x _ { t } ^ { ( h i s t ) } , x _ { t } ^ { ( p r e ) } , x _ { t } ^ { ( e n v ) } \right]\tag{5}
$$

where the vector components are the real-time data $\mathrm { r e \mathrm { - } }$ lated to the vehicle information (i.e. the historical data, preregistration information, and environmental context).

The prediction task is defined as:

$$
f _ { \theta } : \mathbf { x } _ { t - T : t }  \hat { Q } ( t + 1 : t + H )\tag{6}
$$

## C. Multi-Source Data Fusion

The key feature of the proposed approach lies in the ability to integrate and exploit a diverse set of data sources. The system interprets in Real-time th vehicle counts and density estimations from computer vision-based systems; various historical traffic datasets (possibly provided by border authorities), Pre-registration data (volunteers who are willing to skip lines and speed up the procedures in customs), including vehicle metadata and estimated arrival locations, Ferry ticketing information, and in case they are available,the Environmental and contextual data (i.e. weather conditions and external traffic indicators).

These inputs are fused using temporal alignment and feature harmonisation techniques to create a unified data ecosystem. This multi-modal data fusion approach enables the system to reconcile discrepancies between expected and observed traffic patterns, significantly improving predictive accuracy compared to isolated data sources. Feature fusion is defined as:

$$
\mathbf { z } _ { t } = \phi ( \mathbf { x } _ { t } )\tag{7}
$$

For heterogeneous inputs:

$$
\mathbf { z } _ { t } = \sum _ { i = 1 } ^ { N } w _ { i } \cdot \phi _ { i } ( x _ { t } ^ { ( i ) } )\tag{8}
$$

where $w _ { i }$ are the estimated weights.

(a) Overall System Architecture  
![](images/b11a60f10f8edaaaea720b798df0c3d94f37a4122ee6859484dfce498aa68b37.jpg)  
(b) Queue Prediction & Optimization Process

![](images/1383a97e71d8cdd478f1df44d32d771c7747515337cf46a05a38a058cca5eba8.jpg)  
Fig. 1. (a) Overall system architecture for intelligent queue prediction and optimisation, including data ingestion, predictive modelling (LSTM), optimisation (MPC and scheduling), and decision support. (b) Detailed queue prediction and optimisation workflow from multi-source inputs to control actions.

## D. LSTM-Based Queue Prediction

The temporal dependencies are evaluated using LSTM networks according to the analysis below:

$$
\mathbf { f } _ { t } = \sigma ( W _ { f } \mathbf { z } _ { t } + U _ { f } \mathbf { h } _ { t - 1 } + b _ { f } )\tag{9}
$$

$$
\mathbf { i } _ { t } = \sigma ( W _ { i } \mathbf { z } _ { t } + U _ { i } \mathbf { h } _ { t - 1 } + b _ { i } )\tag{10}
$$

$$
\tilde { \mathbf { c } } _ { t } = \operatorname { t a n h } ( W _ { c } \mathbf { z } _ { t } + U _ { c } \mathbf { h } _ { t - 1 } + b _ { c } )\tag{11}
$$

$$
\mathbf { c } _ { t } = \mathbf { f } _ { t } \odot \mathbf { c } _ { t - 1 } + \mathbf { i } _ { t } \odot \tilde { \mathbf { c } } _ { t }\tag{12}
$$

$$
\mathbf { o } _ { t } = \sigma ( W _ { o } \mathbf { z } _ { t } + U _ { o } \mathbf { h } _ { t - 1 } + b _ { o } )
$$

$$
\mathbf { h } _ { t } = \mathbf { o } _ { t } \odot \operatorname { t a n h } ( \mathbf { c } _ { t } )\tag{13}
$$

(14)

where in the equations above, the lowercase variables represent vectors. Matrices $W _ { q }$ and $U _ { q }$ contain, respectively, the weights of the input and recurrent connections, where the subscript $q$ can either be the input gate $i ,$ output gate $^ { O , }$ the forget gate $f$ or the memory cell $c ,$ depending on the activation being calculated. In this section, we are thus using a “vector notation”. $\mathrm { S o } ,$ for example, $c _ { t } \in \mathbb { R } ^ { h }$ is not just one unit of one LSTM cell, but contains h LSTM cell units. Moreover, the initial values are $c _ { 0 } = 0$ and $h _ { 0 } = 0$ and the operator $\odot$ denotes the Hadamard product (element-wise product). The subscript t indexes the time step. This way the predicted queue length can be calculated as:

$$
{ \hat { Q } } ( t + 1 ) = W _ { y } \mathbf { h } _ { t } + b _ { y }\tag{15}
$$

The training loss is defined as:

$$
\mathcal { L } = \frac { 1 } { H } \sum _ { i = 1 } ^ { H } \Big ( Q ( t + i ) - \hat { Q } ( t + i ) \Big ) ^ { 2 }\tag{16}
$$

$$
E . ~ P r e d i c t i v e ~ Q u e u e ~ M a n a g e m e n t ~ S y s t e m ~ ( P Q M S )
$$

Based on the predictive outputs of the above approach, the methodology implements a Predictive Queue Management System (PQMS) tailored to border control operations. The PQMS can in this way, provide early detection of congestion hotspots, forecast the lane-specific queue length and propose recommendations for traffic routing and lane allocation. These insights aim to deliver to border operators through an intuitive decision support interface, enabling proactive management of traffic flows. The PQMS is tightly coupled with the resource allocation module, facilitating coordinated actions such as opening additional lanes, reallocating personnel, and adjusting inspection priorities .

By transitioning from reactive to predictive management, the system enhances operational efficiency and reduces waiting times for travellers. Let $\hat { Q } _ { k } ( t )$ denote predicted queue per lane and $u _ { k } ( t )$ a binary decision variable:

$$
\operatorname* { m i n } _ { u _ { k } ( t ) } \sum _ { k = 1 } ^ { K _ { \operatorname* { m a x } } } \hat { Q } _ { k } ( t )\tag{17}
$$

subject to:

$$
\sum _ { k = 1 } ^ { K _ { \operatorname* { m a x } } } u _ { k } ( t ) \leq K _ { \operatorname* { m a x } }\tag{18}
$$

Traffic routing is defined as:

$$
\operatorname* { m i n } _ { r _ { i j } ( t ) } \sum _ { j } \hat { Q } _ { j } ( t )\tag{19}
$$

$$
\sum _ { j } r _ { i j } ( t ) = 1 , \quad r _ { i j } ( t ) \geq 0\tag{20}
$$

## F. Ferry Terminal Optimisation

The optimization component is responsible for enabling the proposed framework to adapt to different border environmnents, namely land and ferry borders. Based on their distinct characteristics, when addressing land borders, the proposed framework focuses on continuous flow, by maximizing lane utilization and minimizing queues. On the contrary, when adapted to ferry terminals, its main focus is handling burst traffic patterns, usually associated with scheduled ferry arrivals or departures. The system employs AI-driven optimisation techniques to: Minimise total waiting time across all vehicles Maximise throughput under resource constraints Maintain compliance with security and inspection requirements Finally, the proposed work specifically addresses ferry border crossing, by employing specific algorithms and methods in order to ensure optimized vehicle embarkation and disembarkation. This enables increased terminal efficiency by reducing congestion on ramps.

Let V (t) be the set of vehicles and $\tau _ { i }$ their arrival times. The objective is:

$$
\operatorname* { m i n } _ { i \in V \left( t \right) } \left( \tau _ { i } ^ { s t a r t } - \tau _ { i } \right)\tag{21}
$$

Scheduling formulation:

$$
\operatorname* { m i n } _ { \pi } \sum _ { i } W _ { i } \cdot C _ { i }\tag{22}
$$

## G. Model Predictive Control (MPC)

The optimisation over horizon H is:

$$
\operatorname* { m i n } _ { \mathbf { u } _ { t : t + H } } \sum _ { k = 0 } ^ { H } \left( Q ( t + k ) ^ { 2 } + \alpha \| \mathbf { u } ( t + k ) \| ^ { 2 } \right)\tag{23}
$$

subject to:

$$
Q ( t + 1 ) = Q ( t ) + \lambda ( t ) - S ( t )\tag{24}
$$

H. Joint Optimisation with Resource Allocation

$$
\operatorname* { m i n } \left( \beta _ { 1 } \sum Q ( t ) + \beta _ { 2 } \sum C _ { r e s o u r c e } ( t ) \right)
$$

## I. Evaluation Metrics

(25)

Mean Squared Error:

$$
\mathrm { M S E } = \frac { 1 } { N } \sum ( Q - \hat { Q } ) ^ { 2 }\tag{26}
$$

Average waiting time:

$$
W = \frac { 1 } { N } \sum _ { i } ( t _ { i } ^ { e x i t } - t _ { i } ^ { a r r i v a l } )\tag{27}
$$

Throughput:

$$
\mathrm { T h r o u g h p u t } = \frac { \mathrm { v e h i c l e s ~ p r o c e s s e d } } { \mathrm { t i m e } }\tag{28}
$$

## III. EXPERIMENTAL SETUP

The authors in order to simulate a border crossing scenario multiple datasets were used as: realistic vehicle arrival patterns, seasonal and peak-hour variations and synthetic pre-registration and weather data. Afterwards, these datasets compared against baseline datasets, i.s. the Historical average model, the ARIMA time-series model, and the Static rulebased queue management approach.

The Evaluation Metrics that were used for this benchmark test are:

$$
M S E = \frac { 1 } { N } \sum ( Q - \hat { Q } ) ^ { 2 }\tag{29}
$$

$$
W = \frac { 1 } { N } \sum ( t _ { e x i t } - t _ { a r r i v a l } )\tag{30}
$$

## A. Experimental Protocol

To ensure rigorous evaluation, we simulate a border control environment using a discrete-event traffic model calibrated with realistic parameters:

• Arrival process: Non-homogeneous Poisson process with time-varying rate

• Peak periods: Simulated using Gaussian demand surges

• Service rates: Lane-dependent stochastic processing times

• Traffic scenarios: Normal, peak, and disruption conditions

Each experiment is repeated over 10 independent runs to ensure statistical reliability. Results are reported as mean values with standard deviation.

## IV. RESULTS AND DISCUSSION

Regarding the Prediction Accuracy, it should be noted that the proposed LSTM model achieves 25–35% lower MSE than ARIMA, and a robust performance under peak conditions. Table I presents the performance comparison between the proposed model and baseline approaches.

The proposed LSTM-based model achieves approximately 35% lower MSE compared to ARIMA, demonstrating its ability to capture temporal dependencies and non-linear traffic patterns. As an operational performance compared to baseline datasets the waiting time reduced by ∼30% throughput increased by ∼20% resource utilisation improved significantly.

TABLE I  
QUEUE PREDICTION PERFORMANCE COMPARISON
<table><tr><td>Model</td><td>MSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Historical Average</td><td>45.2</td><td>5.8</td><td>6.72</td></tr><tr><td>ARIMA</td><td>31.5</td><td>4.6</td><td>5.61</td></tr><tr><td>LSTM (Proposed)</td><td>20.3</td><td>3.2</td><td>4.51</td></tr></table>

Table II shows the operational improvements achieved by integrating prediction with optimisation.

TABLE II  
OPERATIONAL PERFORMANCE EVALUATION
<table><tr><td>Metric</td><td>Baseline</td><td>Proposed</td><td>Improvement</td></tr><tr><td>Avg Waiting Time (min)</td><td>18.5</td><td>12.9</td><td>↓30%</td></tr><tr><td>Throughput (veh/hr)</td><td>820</td><td>980</td><td>↑19.5%</td></tr><tr><td>Queue Length (avg)</td><td>35.2</td><td>24.6</td><td>↓ 30.1%</td></tr></table>

The results confirm that predictive optimisation significantly enhances throughput while reducing congestion. Moreover, and in order to evaluate the contribution of each data modality, authors perform an ablation study as depicted in Table III.

TABLE III  
ABLATION STUDY ON DATA MODALITIES
<table><tr><td>Configuration</td><td>MSE</td><td>Performance Drop</td></tr><tr><td>Full Model</td><td>20.3</td><td></td></tr><tr><td>Without Pre-registration</td><td>24.1</td><td>+18.7%</td></tr><tr><td>Without Environmental Data</td><td>22.8</td><td>+12.3%</td></tr><tr><td>Without Historical Data</td><td>27.5</td><td>+35.5%</td></tr></table>

The results demonstrate that multi-modal data fusion significantly improves prediction accuracy. The proposed framework consistently outperforms baseline methods across all evaluation metrics. The LSTM model effectively captures nonlinear temporal dependencies and abrupt demand fluctuations, particularly during peak periods where traditional models such as ARIMA exhibit delayed response and smoothing effects.

The integration of predictive modelling with MPC-based optimisation further amplifies performance gains. Unlike reactive baselines, the proposed approach anticipates congestion and proactively adjusts system parameters, resulting in reduced queue accumulation and improved throughput.

Notably, performance improvements are more pronounced under high-variance traffic conditions, indicating that the model generalises well to non-stationary environments. This is particularly relevant for border control scenarios, where traffic patterns are inherently unpredictable.

The integration of LSTM-based prediction with MPC optimisation provides a robust framework for handling both steady-state and burst traffic scenarios. The system performs particularly well in high-variability environments such as ferry terminals, where traditional models fail to adapt. A paired ttest confirms that the improvements of the proposed method over ARIMA are statistically significant $( p < 0 . 0 1 )$ , indicating that performance gains are not due to random variation.

![](images/95b57c10069ee430c880e521479e74aa9212787bc2d04b7b51887a457c838d1b.jpg)  
Fig. 2. Queue prediction performance over time.

Algorithm 1 PQMS Control Loop   
1: Input multi-source data $\mathbf { x } _ { t }$   
2: Fuse features $\mathbf { z } _ { t } = \phi ( \mathbf { x } _ { t } )$   
3: Predict queue $\hat { Q } ( t + 1 )$ using LSTM   
4: Solve MPC optimisation problem   
5: Generate control actions ${ \bf \delta u } ( t )$   
6: Apply actions (lane allocation, routing)   
7: Update system state and repeat

To evaluate robustness under extreme conditions, we simulate sudden traffic surges and partial sensor failures. The proposed model maintains stable performance, with less than 10% degradation in prediction accuracy, whereas baseline models degrade by over 25%. Despite its effectiveness, the proposed framework has several limitations. First, the current evaluation is based primarily on synthetic traffic scenarios that emulate realistic border crossing conditions. Although this approach enables controlled experimentation under diverse operating conditions, it cannot fully capture the complexity, uncertainty, and behavioral variability observed in real-world border operations. Consequently, further validation using operational datasets from land and maritime border crossing points is necessary to fully assess the generalization capability of the proposed methodology. Second, the predictive model relies on the availability and quality of multiple heterogeneous data sources, including historical traffic records, pre-registration information, environmental conditions, and real-time sensor measurements. Missing data, communication delays, sensor failures, or inaccurate measurements may reduce prediction accuracy and consequently affect the optimization process. Although the proposed architecture demonstrated robustness under simulated sensor failures, additional fault-tolerant data fusion and uncertainty estimation mechanisms should be incorporated for deployment in operational environments.

## V. CONCLUSION

This paper presented a unified framework for intelligent queue prediction and optimisation in border control systems, combining multi-modal data fusion, deep learning, and control-theoretic optimisation. The proposed approach demonstrates substantial improvements in both predictive accuracy and operational performance, enabling proactive and adaptive traffic management. Future work will focus on reinforcement learning-based policy ooptimization, integration with realworld border systems, and deployment on distributed edge infrastructures.

## ACKNOWLEDGMENT

This research was partially funded by the European Commission under the Horizon Europe Programme through the ’ORION’ project under grant agreement No. 101225611. Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union.

## REFERENCES

[1] European Commission,“Entry/Exit System (EES) - Migration and Home Affairs,“ European Commission, 2026. [Online]. Available: https://home-affairs.ec.europa.eu/policies/schengen/smart-borders/entryexit-system en

[2] Airports Council International Europe, “EES Rollout Impact Report: Border Processing Time Analysis,” ACI Europe, 2025–2026.

[3] IEEE Public Safety Technology, “High-Tech Border Security: Current and Emerging Trends,” IEEE Public Safety, 2022. [Online]. Available: https://publicsafety.ieee.org/topics/high-tech-border-securitycurrent-and-emerging-trends/

[4] M. Papaioannou, G. Mantas, D. K. Lymberopoulos, and J. Rodriguez, “User Authentication and Authorization for Next Generation Mobile Passenger ID Devices for Land and Sea Border Control,” in Proc. IEEE CAMAD, 2020. doi: 10.1109/csndsp49049.2020.9249574.

[5] C. Ma, G. Dai, and J. Zhou, “Short-Term Traffic Flow Prediction for Urban Road Sections Based on Time Series Analysis and LSTM BiLSTM Method,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 6, pp. 5615–5624, 2022.

[6] J. Liu, N. Q. Wu, Y. Qiao, and Z. W. Li, “Short-Term Traffic Flow Forecasting Using Ensemble Approach Based on Deep Belief Networks,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 1, pp. 404–417, 2022.

[7] R. L. Abduljabbar, H. Dia, and P.-W. Tsai, “Development and Evaluation of Bidirectional LSTM Freeway Traffic Forecasting Models Using Simulation Data,” Scientific Reports, vol. 11, p. 23899, 2021.

[8] S. Afandizadeh et al., “Deep Learning Algorithms for Traffic Forecasting: A Comprehensive Review and Comparison with Classical Ones,” Journal ofAdvanced Transportation, 2024. doi: 10.1155/2024/9981657.

[9] C. X. Ma, Y. P. Zhao, G. W. Dai, X. C. Xu, and S. C. Wong, “A Novel STFSA-CNN-GRU Hybrid Model for Short-Term Traffic Speed Prediction,” IEEE Trans. Intell. Transp. Syst., vol. 24, no. 4, pp. 3728– 3737, 2023.

[10] W. Jiang and J. Luo, “Graph Neural Network for Traffic Forecasting: A Survey,” Expert Systems with Applications, vol. 207, 2022.

[11] S. Guo, Y. Lin, H. Wan, X. Li, and G. Cong, “Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting,” IEEE Trans. Knowl. Data Eng., vol. 34, no. 11, pp. 5415–5428, 2022.

[12] T. Liu, A. Jiang, and J. Zhou, “GraphSAGE-Based Dynamic Spatial– Temporal Graph Convolutional Network for Traffic Prediction,” IEEE Trans. Intell. Transp. Syst., 2023.

[13] M. Lablack and Y. Shen, “Spatio-Temporal Graph Mixformer for Traffic Forecasting,” Expert Systems with Applications, vol. 288, p. 120281, 2023.

[14] B.-L. Ye, W. Wu, K. Ruan, L. Li, T. Chen, H. Gao, and Y. Chen, “A Survey of Model Predictive Control Methods for Traffic Signal Control,” IEEE/CAA J. Autom. Sinica, vol. 6, no. 3, pp. 623–640, 2019.

[15] V. H. Pham and H. S. Ahn, “Distributed Stochastic MPC Traffic Signal Control for Urban Networks,” IEEE Trans. Intell. Transp. Syst., vol. 24, pp. 8079–8096, 2023.

[16] C. Zhang, T. Z. Qiu, and A. Kim, “Centralized and Decentralized Signal Control with Short-Term Origin-Destination Demand for Network Traffic,” Journal of Advanced Transportation, 2022. doi: 10.1155/2022/5806160.

[17] J. Hu et al., “A Multi-Agent Deep Reinforcement Learning Approach for Traffic Signal Coordination,” IET Intell. Transp. Syst., 2024. doi: 10.1049/itr2.12521.

[18] M. Noaeen et al., “Reinforcement Learning in Urban Network Traffic Signal Control: A Systematic Literature Review,” Expert Systems with Applications, vol. 199, p. 116830, 2022.

[19] F. Mao, Z. Li, and L. Li, “A Comparison of Deep Reinforcement Learning Models for Isolated Traffic Signal Control,” IEEE Intell. Transp. Syst. Mag., vol. 15, no. 1, pp. 160–180, 2023.

[20] X. Liang, X. Du, G. Wang, and Z. Han, “A Deep Reinforcement Learning Network for Traffic Light Cycle Control,” IEEE Trans. Veh. Technol., vol. 68, no. 2, pp. 1243–1253, 2019.

[21] J. Guerrero-Iba´nez, S. Zeadally, and J. Contreras-Castillo, “Data Fusion˜ for ITS: A Systematic Literature Review,” Information Fusion, 2022. doi: 10.1016/j.inffus.2022.xx.

[22] Y. Zhang, C. Tu, K. Gao et al., “Multisensor Information Fusion: Future of Environmental Perception in Intelligent Vehicles,” J. Intell. Connected Veh., vol. 7, no. 3, pp. 163–176, 2024. doi: 10.26599/JICV.2023.9210049.

[23] W. Jiang et al., “Towards Effective Fusion and Forecasting of Multimodal Spatio-Temporal Data for Smart Mobility,” arXiv, 2024. [Online]. Available: https://arxiv.org/html/2407.16123v1

[24] “Data Fusion of Heterogeneous Data Sources for ITS,” IEEE Xplore, 2024. doi: 10.1109/10788563.

[25] F. Storani, R. Di Pace, and B. De Schutter, “A Traffic Responsive Control Framework for Signalized Junctions Based on Hybrid Traffic Flow Representation,” Journal ofIntelligent Transportation Systems, vol. 27, no. 5, pp. 606–625, 2023.

[26] S. Bouktif, A. Cheniki, A. Ouni, and H. El-Sayed, “Deep Reinforcement Learning for Traffic Signal Control with Consistent State and Reward Design Approach,” Knowledge-Based Systems, vol. 267, p. 110440, 2023.