# Beyond Sunny Skies: Using Large Language Models to Predict Urban Traffic in Adverse Weather

Zhu Xiao, Senior Member, IEEE, Kuan Feng, Tong Li, Member, IEEE, Chenxi Liu, Gang Wang, Zhuo Tang, and Kenli Li

Abstract—Urban traffic flow prediction is a cornerstone of intelligent transportation systems, yet accurate forecasting under adverse weather conditions remains a significant challenge. Existing data-driven models, while effective for routine patterns, often struggle with the sparse data and complex non-linear shifts associated with rare meteorological events. These methods typically treat environmental contexts as shallow numerical features, failing to capture the underlying semantic reasoning—why and how specific adverse conditions alter mobility behaviors across regions with different functional compositions. To address this limitation, we propose Multi-STG, a unified framework that integrates Large Language Model (LLM)-derived semantic priors with graph-based spatiotemporal learning. Multi-STG introduces an LLM-empowered semantic enhancement module that transforms comprehensive context descriptions such as weather, POIs, and holidays into high-level semantic embeddings, effectively providing a knowledge-driven complement to data scarcity. Furthermore, we design a multimodal attentional fusion mechanism to dynamically weigh heterogeneous inputs and a semantic-enhanced prediction head with gated residual calibration to robustly adjust forecasts based on semantic guidance. Extensive experiments on real-world multi-city private car datasets demonstrate that Multi-STG outperforms state-of-theart baselines, reducing MAE by 11.82% on average relative to the strongest baseline across three cities, while maintaining stable performance under sparse and non-stationary conditions.

Index Terms—Prediction, Large Language Models, Adverse Weather, Private Car, Urban Traffic.

## I. INTRODUCTION

Urban traffic flow prediction estimates future traffic conditions, such as volume, speed, and density, across a citywide road network by modeling historical mobility patterns and spatial dependencies [1]–[4]. As a core component of intelligent transportation systems, traffic flow prediction supports congestion management, dynamic route planning, and sustainable infrastructure development [5], [6]. Accurate forecasts enable urban planners and traffic authorities to optimize signal control, allocate road resources more efficiently, and provide travelers with reliable guidance [7]. However, urban traffic dynamics are strongly influenced by external factors, including adverse weather conditions (e.g., heavy precipitation and fog)

and special events (e.g., holidays), which can cause abrupt, nonlinear departures from regular traffic patterns. Accurate, robust forecasting therefore requires capturing intrinsic spatiotemporal dependencies and integrating heterogeneous contextual information to account for traffic variations induced by these external factors [8], [9].

Extensive research has examined spatiotemporal urban traffic forecasting from various perspectives. Models such as LSTM [10] and Transformer [11] capture temporal dependencies, while graph neural networks [12], [13] model spatial relationships among road segments or regions. To account for external influences, multimodal approaches incorporate auxiliary information into forecasting. For example, T-GCN and MegaCRN capture non-periodic traffic shifts through temporal graph convolution and meta-graph learning, respectively [14], [15]. Other methods incorporate meteorological observations, such as rainfall and visibility [16], [17], or use points of interest to characterize regional functions [18]–[20]. Despite these advances, integrating external information does not necessarily capture the mechanisms through which environmental conditions reshape traffic patterns. Existing approaches often incorporate auxiliary signals through direct encoding or attention mechanisms, but the resulting numerical representations may not adequately capture interactions between environmental conditions and regional functions. Such limitations become particularly pronounced under rare conditions, when traffic responses may differ substantially across regions with distinct functional attributes. Moreover, these approaches offer limited support for explicit semantic reasoning about how external conditions influence regional traffic dynamics, leaving contextual relationships insufficiently explored. Under the imbalanced distribution of observed traffic data, adverse scenarios are underrepresented, encouraging models to prioritize routine patterns during training. These models struggle to generalize to nonlinear traffic changes arising from complex external conditions.

Large language models (LLMs) offer a promising complement to conventional historical-data forecasting methods [21]– [23]. Trained on extensive textual corpora, LLMs possess broad world knowledge and can reason about contextual relationships. Given structured descriptions combining a region’s POI composition, weather conditions, and holiday status, LLMs can suggest plausible effects on human mobility, including responses that are sparsely represented in historical observations. These semantic priors may supplement limited empirical evidence and support generalization under rare or extreme conditions [24], [25]. The central methodological challenge lies in integrating semantic guidance from LLMs into quantitative spatiotemporal forecasting while preserving the numerical relationships learned from traffic observations.

In this paper, we aim to improve urban traffic flow prediction under adverse weather by combining semantic reasoning from LLMs with spatiotemporal graph learning. Achieving this objective presents three closely related technical challenges. First, structured semantic representations must capture interactions between external conditions and traffic dynamics. This requires transforming unstructured, heterogeneous contextual descriptions into features that characterize how adverse weather may affect regions with distinct functional compositions. Second, information from POI distributions, weather vectors, holiday labels, and traffic histories must be dynamically integrated, as the relevance and influence of these inputs vary across locations and over time. For example, heavy rainfall may affect travel demand differently in business and residential districts. Third, the contribution of semantic information must be carefully regulated, particularly when supervision for rare events is limited. Addressing this challenge requires a robust, interpretable correction mechanism that uses contextual semantic priors to calibrate backbone forecasts while preserving reliable numerical patterns, balancing predictive stability with adaptability to unusual conditions.

To address these challenges, we propose Multi-STG, a unified framework that integrates LLM semantic priors with a scalable graph Transformer backbone. First, we design an LLM-empowered semantic enhancement module to extract contextual semantic features. This module constructs structured prompts from regional information, including POI types and weather conditions, and queries a pretrained LLM to obtain semantic embeddings. Lightweight adapter networks then align these embeddings with the numerical feature space. Second, we introduce a multimodal attentional fusion module to dynamically integrate heterogeneous information. Conditioned on the current spatiotemporal state, this module adaptively weights traffic, semantic, and POI embeddings to prioritize the most relevant signals. Finally, we develop a semantic-enhanced prediction head with a gated residual correction mechanism to regulate the contribution of semantic information. The gate explicitly controls the extent of semantic correction, allowing contextual priors to refine backbone forecasts while preserving the numerical patterns learned from traffic observations.

Our contributions are summarized as follows:

• We identify key limitations of existing traffic forecasting models under adverse weather, particularly sparse observations and limited semantic reasoning capabilities. We highlight the importance of integrating contextual information, including weather conditions and POI distributions, to capture complex nonlinear traffic variations in rare scenarios.

• We propose Multi-STG, a unified framework that integrates LLM-derived semantic priors with a graph backbone. The framework comprises an LLM-empowered semantic enhancement module to extract contextual semantic features, a multimodal attentional fusion module to adaptively weight heterogeneous inputs, and a semanticenhanced prediction head for gated residual calibration.

• We conduct extensive experiments on datasets of private vehicle traffic from three cities. The results demonstrate that Multi-STG outperforms leading baselines, achieving an average MAE reduction of 11.82% relative to the strongest baseline while maintaining stable performance across all three cities.

## II. RELATED WORK

## A. Urban Traffic Prediction

Urban traffic prediction has been extensively studied through spatiotemporal modeling and multimodal fusion [26], [27]. Early deep learning approaches primarily employed recurrent or convolutional architectures to capture temporal and spatial dependencies [28]. Graph neural networks have enabled more explicit modeling of road network topology through graph convolutions and dynamic graph learning [29]– [31]. Attention mechanisms further capture long-range spatial interactions and provide flexible spatiotemporal receptive fields [32], [33]. Hybrid architectures combining graph learning with recurrent networks have also been explored to model spatial dependencies that evolve with changing traffic conditions [34], [35].

Beyond structural dependencies, recent studies have emphasized the importance of incorporating external conditions into traffic prediction. Models that integrate POI semantics, event information, holiday patterns, or regional functional attributes can better capture heterogeneous urban dynamics [36]–[38]. Weather conditions have received particular attention because adverse weather can induce substantial nonlinear changes in mobility patterns. Multimodal frameworks demonstrate the benefits of integrating meteorological observations, contextual information, and mobility features to improve forecasting robustness [16], [17]. Generative and probabilistic spatiotemporal models have also been developed to characterize predictive uncertainty under adverse weather conditions [39]–[43].

Despite these advances, existing approaches primarily learn statistical relationships from historical observations and incorporate a limited range of modalities. Using semantic and commonsense knowledge to reason about traffic responses under rare or extreme conditions, particularly adverse weather, remains underexplored, motivating the integration of language models with broad contextual knowledge into traffic prediction.

## B. LLMs in Traffic Prediction

Large language models (LLMs) offer rich semantic priors and contextual reasoning capabilities, motivating growing interest in urban computing applications. Recent studies have explored their potential for urban representation learning, multimodal reasoning, and mobility understanding. Some approaches align textual descriptions with mobility data to pretrain representations of urban regions or trajectories [44], [45]. Others use prompting or language-model-guided embedding refinement to incorporate contextual semantics into downstream mobility tasks [46].

Specialized LLMs have emerged to support urban analytics, mobility understanding, and traffic reasoning. Foundation models tailored to urban data demonstrate the value of incorporating textual and commonsense knowledge into urban planning and spatiotemporal perception [47], [48]. In traffic modeling, recent studies have explored mobility analysis using LLMs, external-knowledge-informed traffic flow reasoning, and multimodal traffic generation [21], [49]. Models focusing on weather and disaster contexts have also been proposed to support mobility forecasting under extreme conditions [50].

Existing traffic research using LLMs primarily focuses on semantic understanding and descriptive reasoning, with limited attention to directly improving spatiotemporal prediction. Few studies examine how to balance LLM-derived semantic knowledge with temporal patterns learned from traffic observations, particularly under adverse weather, when semantic priors may exert excessive influence and introduce prediction bias. To address this gap, our framework combines LLM-based semantic enhancement, selective multimodal fusion, and gated residual calibration to improve the robustness and interpretability of traffic forecasting under adverse weather conditions.

## III. PRELIMINARIES

## A. Urban Traffic Flow

Urban traffic flow quantifies vehicle movement within a city and reflects travel demand, urban activity, and transportation system performance. Traffic patterns are shaped by urban structure, weather conditions, and social activities. Accurate forecasts support intelligent transportation systems by enabling congestion management, traffic signal optimization, and efficient road resource allocation. Reliable predictions also support individual route planning, commercial logistics, and urban infrastructure planning. Under adverse weather conditions, timely forecasts can help transportation authorities anticipate disruptions and maintain operational efficiency.

To represent urban traffic flow, a city is typically partitioned into mutually disjoint regions, denoted by $\mathcal { R } = \{ r _ { 1 } , \ldots , r _ { N } \}$ These regions may be defined using geographic grids, administrative boundaries, or clustering methods that group areas with similar mobility characteristics. For each region r , traffic flow during interval t, denoted by $x _ { i } ( t )$ , is defined as the total number of trip departures and arrivals recorded within that region, corresponding to the sum of regional outflow and inflow. Aggregating these observations yields a citywide traffic vector ${ \bf x } ( t ) \in \mathbb { R } ^ { N }$ . The sequence $\{ \mathbf { x } ( 1 ) , \ldots , \mathbf { x } ( T ) \}$ forms a multivariate time series. Traffic flow exhibits temporal dependencies, including recurring commuting peaks and demand changes during holidays, as well as spatial correlations among neighboring regions or regions with similar functions. These spatial relationships are commonly represented by an urban graph $\mathcal { G } = ( \mathcal { R } , \mathbf { A } )$ , where $\mathbf { A } \in \dot { \mathbb { R } } ^ { N \times N }$ encodes geographic proximity, road connectivity, or mobility connections between regions. Under this formulation, traffic flow prediction aims to estimate $\mathbf { x } ( t + \tau )$ at a future horizon τ, conditioned on historical observations x(1:t) and relevant external information, such as weather conditions, POI distributions, and holiday schedules. This formulation captures the spatiotemporal evolution of urban mobility while accounting for contextual influences.

TABLE I  
PRECIPITATION SENSITIVITY BY POI COMPOSITION IN XI’AN, MEASURED AS THE PERCENTAGE CHANGE IN DAILY FLOW PER MILLIMETER AFTER ACCOUNTING FOR WEEKDAY AND HOLIDAY EFFECTS. LOW AND HIGH DENOTE THE BOTTOM AND TOP TERCILES OF EACH POI CATEGORY’S REGIONAL SHARE. DIFF. DENOTES HIGH MINUS LOW.
<table><tr><td>POI Category</td><td>Low</td><td>High</td><td>Diff.</td></tr><tr><td>Commercial Services</td><td>-1.77</td><td>-1.19</td><td>+0.58</td></tr><tr><td>Cultural &amp; Entertainment</td><td>-1.55</td><td>-1.64</td><td>-0.08</td></tr><tr><td>Tourism &amp; Travel</td><td>-1.19</td><td>-1.80</td><td>-0.61</td></tr><tr><td>Transportation Facilities</td><td>-1.19</td><td>-2.22</td><td>-1.04</td></tr><tr><td>Residential &amp; Office</td><td>-1.97</td><td>-1.13</td><td>+0.84</td></tr></table>

## B. Weather-Traffic Sensitivity

We use a large collection of private vehicle trip records obtained from a telematics service provider in China, covering 2022 to 2024. For this study, we select three metropolitan areas and combine trip records with meteorological observations and statutory holiday schedules. Trip departure and arrival locations and timestamps are used to reconstruct regional traffic flows.

Fig. 1 presents Xi’an as a representative example to illustrate associations between traffic volume, weather conditions, and holidays. The weather analysis covers December 2023 to February 2024, while the forecasting experiments use contextual information throughout the full modeling period. We consider precipitation, snowfall, and visibility as key meteorological variables.

Traffic volume generally declines as weather conditions deteriorate. Average daily volume decreases from 10,080 on dry days to 9,856 under light precipitation of up to 1 mm and 9,331 under precipitation above 1 mm. However, the latter category includes only eight days, so interpret it cautiously. Each additional centimeter of snowfall is associated with an approximately 5.6% reduction in daily volume. Lower visibility also coincides with reduced traffic, although this association cannot be separated from accompanying conditions such as heavy rain and fog.

Holiday periods exhibit distinct departures from routine traffic patterns. During the Spring Festival, traffic volume peaks before the holiday, reaches a minimum during the holiday, and rebounds afterward. These fluctuations suggest that holiday schedules provide relevant context beyond regular temporal patterns.

Weather associations also vary across regions. After accounting for weekday and holiday effects, 84% of the 120 regions in Xi’an exhibit a negative association between precipitation and traffic volume. The estimated mean and median reductions are 1.4% and 0.6% per millimeter, respectively. Table I compares precipitation sensitivity across regions with low and high shares of different POI categories, defined by the bottom and top terciles of each category’s share. Transportation facilities show the largest contrast, while other categories exhibit smaller differences. Together, these observations motivate contextual modeling that accounts for weather conditions, holiday patterns, and regional functional composition alongside spatiotemporal dependencies.

Precipitation vs. Trajectory Count  
![](images/0c556393a8de6aa4c6f1fe74f1d9006ac4dd613353fbffc7407aa187fc275386.jpg)

Snow Depth vs. Trajectory Count  
![](images/78e624830bb2e7d70c73cb46837ceb6adac997673db9ce256646b0d8c566e9dd.jpg)  
Visibility vs. Trajectory Count

![](images/1f5d41800ebd28ae820798a6d95edb001b6491e8886758d6e390ca851eda98d5.jpg)  
Fig. 1. Statistical analysis of traffic trajectory volume, weather conditions, and holiday events in Xi’an, illustrating the correlation between traffic flow fluctuations and external environmental factors.

## C. Problem Definition

We consider an urban road network partitioned into N regions. At each time step t, the observed traffic state is represented by $\mathbf { \boldsymbol { X } } _ { t } ~ \in ~ \mathbb { R } ^ { \bar { N } \times { C } }$ , where $C \ = \ 3$ corresponds to regional traffic flow, time of day, and day of the week. Regional traffic flow is calculated as the sum of the inflow and outflow events defined above. In addition, the model incorporates the following contextual information.

Regional POI Distribution: For each region i, we aggregate POIs across five functional categories to construct a feature vector $P _ { i } \in \mathbb { R } ^ { D _ { \mathrm { p o i } } }$ , where $D _ { \mathrm { p o i } } = 5$ . These categories comprise commercial services, culture and entertainment, tourism and travel, transportation facilities, and residential and office functions. The resulting vector captures the region’s functional composition.

External Environmental Information: We represent weather conditions and calendar attributes, including holiday indicators and date types, as $\scriptstyle { E _ { t } }$

LLM Semantic Features: We construct structured prompts from the regional POI distribution $P _ { i }$ and environmental information $\scriptstyle { E _ { t } }$ . A pretrained LLM processes these prompts to obtain a semantic representation $S _ { i , t } \in \mathbb { R } ^ { D _ { \mathrm { s c m } } }$ that characterizes potential contextual influences on traffic in region i at time t.

The input for each region and time step combines $X _ { i , t } , P _ { i }$ and $S _ { i , t }$ . Given a historical window of length T, the model learns a mapping $f _ { \Theta } ( \cdot )$ to predict traffic flows over the next $T ^ { \prime }$ time steps:

$$
[ { \pmb X } _ { t + 1 } ^ { \prime } , \ldots , { \pmb X } _ { t + T ^ { \prime } } ^ { \prime } ] = f _ { \Theta } \left( [ { \pmb X } _ { t - T + 1 } , \ldots , { \pmb X } _ { t } ] , { \pmb P } , { \pmb S } _ { t - T + 1 : t } \right) ,\tag{1}
$$

where $P \in \mathbb { R } ^ { N \times D _ { p o i } }$ contains the POI features of all regions, $S _ { t - T + 1 : t }$ contains the corresponding semantic features within the historical window, and Θ denotes the model parameters. Each output $\boldsymbol X _ { t + \tau } ^ { \prime } \in \mathbb R ^ { N \times 1 }$ contains the predicted regional traffic flows at horizon τ.

## IV. METHODOLOGY

## A. Framework Overview

In this section, we present Multi-STG, a framework for robust urban traffic flow prediction under adverse weather conditions. Multi-STG integrates semantic reasoning from large language models (LLMs) with spatiotemporal graph learning. As illustrated in Fig. 2, the framework comprises four main modules:

Regional Feature Modeling: This module processes raw trajectory data, partitions the city into a regular grid, and constructs initial spatiotemporal representations from historical regional traffic flows.

LLM-Empowered Semantic Enhancement: This module uses a pretrained LLM to process regional context, including POI composition, weather conditions, and holiday information. The resulting semantic representations characterize potential nonlinear influences of external conditions on regional traffic dynamics.

Multimodal Attentional Fusion: This module dynamically integrates static POI features, LLM semantic embeddings, and historical spatiotemporal representations. An attention mechanism with adaptive gating weights assigns each modality’s contribution based on the current context.

Graph Transformer Module: As the predictive backbone, this module propagates information over a learned adaptive graph to capture spatial dependencies, followed by spatiotemporal attention to model global spatial and temporal interactions. A semantic prediction head then applies gated residual corrections to refine the backbone forecasts while preserving numerical patterns learned from traffic observations.

![](images/8de0889f3c312c34ed1721d1038a44bdc1b86a99ad440918c79b06a25599bfa7.jpg)  
Fig. 2. Overview of the proposed Multi-STG framework. The architecture consists of four key components: (1) Regional Feature Modeling for grid-based region partitioning and initial spatiotemporal encoding; (2) LLM-Empowered Semantic Enhancement for inferring traffic impacts from external contexts (weather, holidays); (3) Multimodal Attentional Fusion for dynamically integrating traffic, semantic, and POI features; and (4) a Graph Transformer Module that combines learned graph propagation with spatial and temporal attention, culminating in a semantic-enhanced residual prediction.

## B. Regional Feature Modeling

Regional Feature Modeling provides the foundation for our framework by transforming private vehicle trajectory records into structured regional representations. These representations capture traffic dynamics, including commuting patterns and weekday-weekend variations, and define the spatial units for subsequent semantic enhancement and graph learning. The process comprises two stages: dividing urban regions using a regular grid and modeling urban spatiotemporal dynamics.

1) Urban Region Division: We partition each city into a regular grid covering the geographic extent of the trajectory data. Specifically, we determine a bounding box for each city and divide the enclosed area into cells measuring 2 km×2 km. All cells are retained as regions, including those with low traffic volumes, without filtering or merging based on traffic density. This procedure yields N = 120 regions for Xi’an, N = 153 for Shenyang, and N = 30 for Urumqi.

The grid partition offers three practical advantages over clustering methods. First, once the bounding box and grid alignment are fixed, the partition is deterministic and requires no model fitting. Second, fixed cell boundaries support consistent spatial aggregation as traffic volumes vary across seasons and cities. Third, uniform cell sizes provide a common spatial resolution that facilitates cross-city comparisons. A potential limitation is that individual cells may contain diverse land uses. We therefore explicitly incorporate each cell’s POI composition, as described in Section III-C, to represent regional functional characteristics.

2) Urban Spatial-Temporal Modeling: Urban traffic exhibits spatial dependencies among regions. Using the regional partition ${ \mathcal { R } } ,$ we represent the citywide traffic system as a directed graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where the nodes $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { N } \}$ correspond to the regions in $\mathcal { R }$ . We learn the adjacency matrix from trainable node embeddings:

$$
\mathbf { A } _ { t } = \mathrm { s o f t m a x } \left( \operatorname { R e L U } \left( \mathbf { E } _ { t } \mathbf { E } _ { t } ^ { \top } \right) \right) ,\tag{2}
$$

where $\mathbf { E } _ { t } ~ \in ~ \mathbb { R } ^ { N \times d }$ denotes the node embedding matrix, and softmax normalization is applied over each row. This formulation allows the model to learn spatial relationships jointly with forecasting.

To represent temporal dynamics, we aggregate trip events into hourly intervals. Let $x _ { i } ( t )$ denote the traffic flow in region $r _ { i }$ during interval $t ,$ defined as the total number of trip departures and arrivals recorded in that region. Observations over T intervals form a flow matrix $\mathbf { X } ^ { \mathrm { { r a w } } } \in \mathbb { R } ^ { N \times T }$ , with each row representing the temporal sequence of a region. We augment each flow observation with two temporal covariates, time of day and day of week, yielding an input tensor $\mathbf { X } \in \mathbb { R } ^ { T \times N \times 3 }$ Together, the learned adjacency matrix and the input tensor provide a unified spatiotemporal representation for modeling dependencies across regions and over time.

## C. LLM-Empowered Semantic Enhancement

Urban traffic patterns are shaped by road network structure, historical flow dynamics, and contextual factors such as weather conditions, holidays, and the spatial distribution of Points of Interest (POIs). These contextual factors influence both daily fluctuations and recurring demand patterns. For example, adverse weather can suppress mobility, while public holidays can alter travel demand within and between cities.

Existing traffic prediction models commonly encode contextual variables using numerical embeddings. Although these representations provide useful information, they may inadequately capture interactions among environmental conditions and regional functions. Heavy rain or snow, for example, may affect traffic differently in commercial and residential areas, particularly during holidays. To better represent these interactions, we introduce semantic reasoning from a pretrained LLM. By drawing on contextual knowledge, the LLM provides semantic priors that describe potential traffic responses under different combinations of regional and environmental conditions.

![](images/321ae3cda6019025928a6d1869e440b13d26b5d0f590400715e582fd779b5e94.jpg)  
Fig. 3. Example of inferring traffic flow influence using a pre-trained LLM. The prompt is structured with task instruction, contextual factors (weather and holiday), and regional POI composition. The LLM generates a semantic reasoning response and outputs an 8-dimensional influence vector with entries in [−1, 1] indicating the intensity of traffic flow variation.

For each region $r _ { i }$ , we summarize the POI distribution as $\mathbf { p } _ { i } ~ \in ~ \mathbb { R } ^ { D _ { \mathrm { p o i } } }$ <sup>i</sup> , where each dimension represents a functional category. At time step t, weather features $w _ { t }$ and holiday labels $h _ { t }$ provide additional context. Together, these variables form the contextual feature set $\mathcal { E } _ { i , t } = \{ \mathbf { p } _ { i } , w _ { t } , h _ { t } \}$ . A prompt construction function $\Phi ( \cdot )$ converts these structured features into a natural language description:

$$
\mathrm { P r o m p t } i , t = \Phi ( { \mathcal { E } } i , t ) .\tag{3}
$$

The prompt describes the regional functional composition, weather conditions, and holiday status, and requests a structured assessment of their potential influence on traffic. This formulation enables the LLM to interpret the contextual features jointly.

As illustrated in Fig. 3, the pretrained LLM M processes the prompt and produces a semantic feature vector:

$$
\mathbf { s } _ { i , t } = \mathcal { M } ( \mathrm { P r o m p t } i , t ) \in \mathbb { R } ^ { D \mathrm { s e m } } .\tag{4}
$$

We set $D _ { \mathrm { s e m } } \ = \ 8 .$ , with dimensions corresponding to the morning peak, evening peak, daytime activity, nighttime activity, leisure demand, weather influence, holiday influence, and overall traffic level. This predefined structure organizes the semantic output into interpretable aspects of regional traffic dynamics. The resulting representation supplements the original contextual variables with semantic assessments of their combined influence. For example, heavy snowfall during a holiday may affect leisure demand and commuting activity differently across regions with distinct functional compositions. These semantic priors provide additional contextual guidance for subsequent spatiotemporal prediction.

## D. Multimodal Attentional Fusion

To improve responsiveness to changes in external conditions, we introduce a dynamic fusion module that combines attention with adaptive gating. The module takes traffic features X, semantic features S generated by the LLM, and POI features P as inputs. Traffic features provide attention queries, semantic features provide keys, and POI features provide values. A complementary gating mechanism adjusts each modality’s contribution at every time step and region. Together, we capture cross-region dependencies and adapt modality contributions to the current context.

The input features are represented as $\mathbf { X } \in \mathbb { R } ^ { T \times N \times D _ { x } } , \mathbf { S } \in$ $\mathbb { R } ^ { T \times N \times \dot { D _ { \mathrm { s e m } } } }$ , and $\mathbf { P } \in \mathbb { R } ^ { N \times D _ { \mathrm { p o i } } }$ , where $D _ { x } = C = 3$ . Using column vectors, we construct the attention queries, keys, and values as

$$
{ \bf q } _ { t , n } = { \bf W } _ { Q } { \bf x } _ { t , n } , \qquad { \bf k } _ { t , m } = { \bf W } _ { K } { \bf s } _ { t , m } , \qquad { \bf v } _ { m } = { \bf W } _ { V } { \bf p } _ { m } ,\tag{5}
$$

where the learned projection matrices map all three representations into a common space of dimension d. The attention weights are computed as

$$
\alpha _ { t , n , m } = \mathrm { s o f t m a x } _ { m } \left( \frac { \mathbf { q } _ { t , n } ^ { \top } \mathbf { k } _ { t , m } } { \sqrt { d } } \right) .\tag{6}
$$

The resulting attention representation aggregates POI information across regions:

$$
\mathbf { a } _ { t , n } = \sum _ { m = 1 } ^ { N } \alpha _ { t , n , m } \mathbf { v } _ { m } .\tag{7}
$$

This formulation allows each region to retrieve functional information according to the compatibility between the local traffic state and the semantic context of other regions.

To regulate modality contributions, we additionally project the traffic, semantic, and POI features into aligned representations $\widetilde { \mathbf { x } } _ { t , n } , \widetilde { \mathbf { s } } _ { t , n }$ , and $\widetilde { \mathbf { p } } _ { n }$ , each belonging to $\mathbb { R } ^ { \breve { d } }$ . These representations are concatenated and passed through a lightweight mapping, followed by softmax normalization:

$$
\mathbf { g } _ { t , n } = \mathrm { s o f t m a x } \left( \mathrm { M L P } \left( \left[ \widetilde { \mathbf { x } } _ { t , n } \| \widetilde { \mathbf { s } } _ { t , n } \| \widetilde { \mathbf { p } } _ { n } \right] \right) \right) \in \mathbb { R } ^ { 3 } ,\tag{8}
$$

where ∥ denotes concatenation. The three components correspond to the traffic, semantic, and POI modalities, respectively. The gated representation is then obtained as

$$
\begin{array} { r } { \mathbf { g } _ { t , n } ^ { \mathrm { f u s e } } = g _ { t , n } ^ { ( 1 ) } \widetilde { \mathbf { x } } _ { t , n } + g _ { t , n } ^ { ( 2 ) } \widetilde { \mathbf { s } } _ { t , n } + g _ { t , n } ^ { ( 3 ) } \widetilde { \mathbf { p } } _ { n } . } \end{array}\tag{9}
$$

![](images/200b85ce26bf350958a6a76ca3bf2347cd9ee516f6086af737ff9950a254ecc4.jpg)  
Fig. 4. Graph Transformer architecture with adaptive graph propagation and gated fusion of representations at zero and one hop, spatial and temporal attention, and a semantic residual prediction head for contextual calibration.

The gating weights vary across time and regions, allowing the model to adjust modality contributions under different conditions, including adverse weather.

Finally, we combine the gated representation with the attention output through residual fusion:

$$
\mathbf { F } _ { t , n } = \mathbf { g } _ { t , n } ^ { \mathrm { f u s e } } + \boldsymbol { \gamma } \cdot \mathbf { a } _ { t , n } ,\tag{10}
$$

where $\gamma$ is a learnable scaling parameter. This design combines direct modality weighting with semantic guidance for aggregating regional functional information, producing the fused representation for the downstream predictor.

## E. Graph Transformer Module

Urban traffic systems exhibit complex spatiotemporal dependencies arising from interactions among regions and external disturbances. To model these dependencies, we introduce the Graph Transformer Module shown in Fig. 4. Given the fused input tensor $\mathbf { H } ^ { ( 0 ) } \in \mathbb { R } ^ { T \times N \times D }$ , the module performs gated graph propagation, spatial and temporal attention, and semantic residual calibration.

The module first propagates information through the learned adjacency matrix ${ \bf A } _ { t }$ . At each time step t, the original representation $\mathbf { H } _ { t } ^ { ( 0 ) } \in \mathbb { R } ^ { N \times D }$ preserves the features of individual regions, while one propagation step aggregates information according to the learned graph:

$$
\mathbf { H } _ { t } ^ { ( 1 ) } = \mathbf { A } _ { t } \mathbf { H } _ { t } ^ { ( 0 ) } .\tag{11}
$$

A learnable gate balances the original and propagated representations:

$$
\mathbf { H } _ { t } = ( 1 - \beta ) \odot \mathbf { H } _ { t } ^ { ( 0 ) } + \beta \odot \mathbf { H } _ { t } ^ { ( 1 ) } ,\tag{12}
$$

where $\beta$ contains gating values in [0, 1], broadcast as needed, and ⊙ denotes elementwise multiplication. This operation preserves regional information while incorporating dependencies encoded by the learned adjacency. The connected regions need not be geographically adjacent.

Spatial and temporal attention subsequently capture interactions across regions and historical observations. The propagated features are projected into queries, keys, and values:

$$
\mathbf { Q } = \mathbf { H } \mathbf { W } _ { Q } , \qquad \mathbf { K } = \mathbf { H } \mathbf { W } _ { K } , \qquad \mathbf { V } = \mathbf { H } \mathbf { W } _ { V } ,\tag{13}
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { K }$ , and $\mathbf { W } _ { V }$ are learned projection matrices applied across all regions and time steps.

Spatial attention aggregates information across regions at each time step. The conventional attention formulation is

$$
\mathbf { O } _ { t } ^ { \mathrm { s p a t i a l } } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { t } \mathbf { K } _ { t } ^ { \top } } { \sqrt { d _ { k } } } \right) \mathbf { V } _ { t } ,\tag{14}
$$

where $\mathbf { Q } _ { t } , \mathbf { K } _ { t } \ \in \ \mathbb { R } ^ { N \times d _ { k } }$ , and softmax normalization is applied over the source regions. Each target region aggregates information according to the compatibility between the query and the keys of other regions, allowing interactions beyond geographic proximity to contribute to prediction.

Temporal attention captures dependencies across historical time steps within each region:

$$
\mathbf { O } _ { n } ^ { \mathrm { t e m p o r a l } } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } ^ { ( n ) } ( \mathbf { K } ^ { ( n ) } ) ^ { \top } } { \sqrt { d _ { k } } } \right) \mathbf { V } ^ { ( n ) } ,\tag{15}
$$

where $\mathbf { Q } ^ { ( n ) } , \mathbf { K } ^ { ( n ) } \in \mathbb { R } ^ { T \times d _ { k } }$ contain the query and key representations for region n over the historical window. This operation allows different observations to contribute according to their relevance, supporting the modeling of recurring patterns and transient fluctuations.

The equations above describe the conventional softmax attention formulation. For computational efficiency, the implementation uses linearized attention, which replaces the softmax kernel with a feature-map approximation and aggregates key and value information before applying the queries. For fixed feature dimensions, this changes the computational dependence on the number of regions from quadratic to linear. The temporal counterpart similarly avoids constructing a full attention matrix over time. Although the approximation may affect attention estimates, no noticeable loss in prediction accuracy was observed for the shallow attention layers and city graphs considered in this study. A systematic assessment of the relationship between accuracy and efficiency remains a direction for future work.

The spatial and temporal outputs are concatenated and projected into a unified representation Z. This representation combines interactions across regions with temporal dependencies within each region and provides the input to the forecasting head.

Finally, a semantic prediction head calibrates the backbone forecasts using contextual information from the LLM. Let

$$
\mathbf { B } = f _ { \mathrm { b a c k b o n e } } ( \mathbf { Z } ) \in \mathbb { R } ^ { T ^ { \prime } \times N \times 1 }\tag{16}
$$

denote the backbone predictions for the next $T ^ { \prime }$ time steps. We broadcast the semantic features $\mathbf { S } _ { t }$ across the prediction horizon to obtain $\overline { { \mathbf { S } } } _ { t }$ . A lightweight MLP processes the concatenated backbone predictions and semantic features to produce a residual correction:

$$
\Delta \mathbf { X } = f _ { \mathrm { s e m } } \left( \left[ \mathbf { B } \| \overline { { \mathbf { S } } } _ { t } \right] \right) ,\tag{17}
$$

where ∥ denotes concatenation along the feature dimension, and $\Delta \mathbf { X }$ has the same dimensions as B. The final predictions are

$$
\begin{array} { r } { \mathbf { X } _ { t + 1 : t + T ^ { \prime } } ^ { \prime } = \mathbf { B } + \lambda \Delta \mathbf { X } , } \end{array}\tag{18}
$$

where λ is a learnable residual weight controlling the contribution of semantic correction. This formulation allows contextual priors to refine the backbone forecasts while retaining an explicit prediction component learned from traffic observations.

## V. EXPERIMENTS

In this section, we evaluate Multi-STG using private vehicle trajectory data and compare performance against established forecasting baselines. We also conduct ablation studies to assess individual components, experiments with varying amounts of training data to examine the effect of data availability, evaluations across cities to assess transferability, and hyperparameter sensitivity analyses to examine model stability.

## A. Datasets

We use a private vehicle trajectory dataset [51] collected in three Chinese cities: Xi’an, Shenyang, and Urumqi. The six monthly files contain 5,144,177 deduplicated trips. Among these, 2,838,999 trips fall within the modeling window and have at least one endpoint inside the corresponding city boundary. These trips yield 4,816,823 inflow and outflow events, comprising 1,743,732 events in Xi’an, 2,305,984 in Shenyang, and 767,107 in Urumqi. The modeling window extends from 1 December 2023 through 24 May 2024, covering 4,224 hourly intervals. Each city is partitioned into a regular grid of 2 km × 2 km cells, yielding 120, 153, and 30 regions, respectively. Regional traffic flow is defined as the total number of trip departures and arrivals recorded within each region during an hourly interval. Table II summarizes the dataset statistics.

Before training, we preprocess the data by removing duplicates, filtering anomalous records, aggregating hourly, and denoising sparse regional time series by temporal period. Denoising statistics are estimated exclusively from the training segment, and the same preprocessing pipeline is applied to all models. As summarized in Table III, the raw attributes include vehicle and trip identifiers, departure and arrival timestamps, and origin and destination coordinates.

To represent regional functionality, we construct functional profiles using Points of Interest (POIs) obtained from open map APIs. Each POI is assigned to a region according to the geographic coordinates. To reduce sparsity while preserving functional distinctions, we consolidate the 14 original categories into five classes: Commercial Services, Cultural & Entertainment, Tourism & Travel, Transportation Facilities, and Residential & Office. The category mapping is detailed in Table IV. For each region, we calculate the proportion of POIs in each class, producing a five-dimensional feature vector. This vector characterizes regional functional composition and provides static semantic context for modeling traffic patterns across heterogeneous urban regions.

Data providers and participating users authorized data access and use. All records were anonymized before acquisition, and traffic data used for modeling were aggregated by region to protect personal information.

## B. Metrics and Baselines

We evaluate forecasting performance using MAE, RMSE, WAPE, and $R ^ { 2 }$ . MAE and RMSE are reported in flow units, with RMSE assigning greater weight to large errors. WAPE measures aggregate relative error and is defined as $\begin{array} { r } { \mathrm { W A P E } = \frac { \sum _ { j } | \hat { y } _ { j } - y _ { j } | } { \sum _ { j } | y _ { j } | } \times 1 0 0 \% } \end{array}$ , where $j$ indexes the evaluated prediction points. $R ^ { 2 }$ evaluates predictive fit relative to a baseline that always predicts the mean observed flow. We compute all metrics globally by pooling predictions across test windows, forecast horizons, and regions. All models use the same preprocessed series for training and evaluation.

We compare Multi-STG with eight representative forecasting baselines:

• LSTM [10]. Uses gated recurrence to capture temporal dependencies. The standard architecture provides a sequential baseline without explicitly modeling graph relationships among regions.

• STGCN [52]. Combines graph convolutions with gated temporal convolutions to jointly capture spatial relationships and temporal dynamics.

• GWaveNet [53]. Combines adaptive graph learning with dilated causal convolutions to capture latent spatial connectivity and temporal dependencies over extended historical windows.

• DSTAGNN [54]. Models dynamic spatial dependencies and temporal patterns through learned graph relationships, spatial attention, and gated temporal convolutions.

• TACTiS [55]. Uses a Transformer architecture and an attentional copula decoder to estimate joint predictive distributions for multivariate time series.

• USTD [40]. Provides a unified diffusion framework with a shared spatiotemporal encoder and specialized attention denoisers for forecasting and spatial interpolation.

• DiffSTG [39]. Extends denoising diffusion probabilistic modeling to spatiotemporal graphs, enabling nonautoregressive forecasting and predictive uncertainty estimation.

• SARAF [56]. Augments forecasting with historical retrieval, balancing temporal relevance and retrieval diversity and adjusting aggregation according to series stationarity.

## C. Experiment Settings

We organize traffic observations, POI distributions, and weather data by region and align temporal observations to hourly intervals. The LLM-empowered Semantic Enhancement module derives semantic features from regional contextual information. The grid partitions yield graphs with 120, 153, and 30 nodes for Xi’an, Shenyang, and Urumqi, respectively. Each sample contains 24 hours of historical observations and 12 hours of target flows. Of the 4,189 candidate sliding windows, 2,932 are used for training, 396 for validation, and 839 for testing (approximately 7:1:2). To avoid target-time overlap, we discard 11 windows at the training/validation boundary and 11 at the validation/test boundary, leaving 4,167 used windows; historical input windows may overlap across splits, whereas target intervals are disjoint. We standardize the flow channel using statistics computed exclusively from the training set and transform predictions back to the original scale for evaluation.

The Graph Transformer backbone uses a model dimension of 152, four attention heads, and three layers, with a dropout rate of 0.1. The Multimodal Attentional Fusion module uses two attention heads, a hidden dimension of 64, and softmax gating to regulate modality contributions. We train the model using Adam with a learning rate of $1 \times 1 0 ^ { - 3 }$ and a batch size of 32. Training runs for up to 100 epochs, with an early stopping patience of 20 epochs and an $\ell _ { 1 }$ loss computed on standardized flow values. Unless otherwise specified, all experiments follow the same training configuration and evaluation protocol.

TABLE II  
STATISTICS OF THE PRIVATE VEHICLE TRIP DATASET. #TRIPS DENOTES DEDUPLICATED TRIPS WITH AT LEAST ONE ENDPOINT WITHIN THE CITY AND MODELING PERIOD. #EVENTS COUNTS ELIGIBLE DEPARTURES AND ARRIVALS, WITH TWO EVENTS WHEN BOTH ENDPOINTS QUALIFY. #VEHICLES DENOTES UNIQUE VEHICLES IN THE SELECTED TRIPS.
<table><tr><td>City</td><td>#Trips</td><td>#Vehicles</td><td>#Events</td><td>Modeling Period</td><td>Longitude Range</td><td>Latitude Range</td><td>Grid</td><td>#Regions</td></tr><tr><td>Xi&#x27;an</td><td>1,024,825</td><td>18,695</td><td>1,743,732</td><td>2023/12/01–2024/05/24</td><td> $1 0 8 . 8 5 \mathrm { - } 1 0 9 . 0 5 ^ { \circ } \mathrm { E }$ </td><td> $3 4 . 1 5 { - } 3 4 . 3 5 ^ { \circ } \mathrm { N }$ </td><td> $1 2 \times 1 0$ </td><td>120</td></tr><tr><td>Shenyang</td><td>1,294,948</td><td>13,803</td><td>2,305,984</td><td>2023/12/01–2024/05/24</td><td> $1 2 3 . 3 2 \mathrm { - } 1 2 3 . 5 2 ^ { \circ } \mathrm { E }$ </td><td> $4 1 . 6 4 \mathrm { - } 4 1 . 9 4 ^ { \circ } \mathrm { N }$ </td><td> $1 7 \times 9$ </td><td>153</td></tr><tr><td>Urumqi</td><td>519,226</td><td>6,429</td><td>767,107</td><td>2023/12/01–2024/05/24</td><td>87.55–87.65°E</td><td> $4 3 . 7 5 { - } 4 3 . 8 5 ^ { \circ } \mathrm { N }$ </td><td> $6 \times 5$ </td><td>30</td></tr></table>

TABLE III  
ATTRIBUTES OF THE PRIVATE VEHICLE TRIP DATASET.
<table><tr><td>Attribute</td><td>Description</td><td>Attribute</td><td>Description</td></tr><tr><td>RecID</td><td>Unique trip record identifier</td><td>StartLon</td><td>Longitude of the trip origin</td></tr><tr><td>ObjectID</td><td>Unique vehicle identifier</td><td>StartLat</td><td>Latitude of the trip origin</td></tr><tr><td>StartTime</td><td>Trip start timestamp</td><td>StopLon</td><td>Longitude of the trip destination</td></tr><tr><td>StopTime</td><td>Trip end timestamp</td><td>StopLat</td><td>Latitude of the trip destination</td></tr></table>

## D. Overall Performance Evaluation

Table V compares Multi-STG with eight forecasting baselines across Xi’an, Shenyang, and Urumqi. The best and second-best results are shown in bold and underlined, respectively. Multi-STG achieves the best performance across all 12 metric-city combinations. Relative to the strongest baseline for each city and metric, Multi-STG reduces MAE by 9.90% to 13.15%, averaging 11.82%, RMSE by 7.84% to 11.72%, and WAPE by 9.89% to 13.17%. The corresponding $R ^ { 2 }$ values are 0.8917, 0.9251, and 0.9468 for Xi’an, Shenyang, and Urumqi, respectively.

DSTAGNN, STGCN, and SARAF provide the strongest baseline results in different settings. DSTAGNN leads across all four metrics in Xi’an and achieves the best baseline RMSE and $R ^ { 2 }$ in Shenyang. STGCN achieves the lowest baseline MAE and WAPE in Shenyang and Urumqi, while SARAF performs best among the baselines on RMSE and $R ^ { 2 }$ in Urumqi. LSTM remains competitive on selected metrics but does not match the strongest baseline results. The probabilistic models DiffSTG, USTD, and TACTiS produce higher point forecast errors in this evaluation. These results characterize the accuracy of their point predictions, while the quality of their predictive distributions requires separate assessment.

The consistent gains across all three cities demonstrate Multi-STG’s overall effectiveness. The framework combines adaptive graph propagation, spatial and temporal attention, multimodal fusion, and contextual semantic information to model regional traffic dynamics. The results support this integrated design for urban traffic prediction across heterogeneous city settings.

## E. Ablation Studies

To assess the contributions of individual components and the effectiveness of their integration, we evaluate ablation variants of Multi-STG on the three city datasets under identical experimental settings. The variants are defined as follows:

• STG (w/o Contextual Features): Uses the spatiotemporal forecasting backbone with traffic sequences alone, excluding external contextual and environmental information.

• A-STG (w/o LLM-Empowered Semantic Enhancement): Incorporates POI distributions, weather conditions, and holiday indicators through direct feature encoding, without LLM-based semantic reasoning. This variant evaluates the additional contribution of semantic enhancement beyond directly encoded context.

• B-STG (w/o Multimodal Attentional Fusion): Replaces the Multimodal Attentional Fusion module with simple feature concatenation to assess the contribution of adaptive modality weighting and selection.

• C-STG (w/o Semantic Enhanced Prediction Head): Removes the semantic prediction head while retaining all other components, allowing the contribution of residual calibration to be evaluated.

• Multi-STG (Full Model): Integrates the Graph Transformer Module, LLM-empowered semantic enhancement, Multimodal Attentional Fusion, and the Semantic Enhanced Prediction Head.

Table VI summarizes the results. MAE, RMSE, and WAPE decrease consistently from STG through A-STG, B-STG, and C-STG to Multi-STG, while $R ^ { 2 }$ increases in the same sequence. The full model performs best across all 12 combinations of metrics and cities.

Using the full model as the reference, the MAE increases for C-STG are 3.43%, 3.26%, and 3.70% in Xi’an, Shenyang, and Urumqi, respectively. The corresponding increases are 5.51%, 5.38%, and 6.13% for B-STG, 7.52%, 7.36%, and 8.43% for A-STG, and 10.10%, 10.16%, and 10.95% for STG.

These comparisons support the value of incorporating contextual features and show that LLM-based semantic enhancement provides benefits beyond direct feature encoding. The results also demonstrate the advantages of adaptive multimodal fusion over concatenation and the consistent contribution of semantic residual calibration to prediction accuracy.

## F. Hyperparameter Study

1) Sensitivity of Residual Weights: To assess sensitivity to the residual weight λ in the Semantic Enhanced Prediction Head, we evaluate $\lambda \in \{ 0 . 0 5 , 0 . 1 , 0 . 3 , 0 . 5 \}$ while keeping all other settings fixed. Fig. 5 presents the results.

TABLE V  
TABLE IV  
POI CATEGORY AGGREGATION AND FUNCTIONAL RATIONALE.
<table><tr><td>Merged Category</td><td>Original Categories Included</td><td>Rationale for Merging</td></tr><tr><td>Commercial Services</td><td>Dining, Shopping, Lifestyle Services, Compa- nies, Finance, Automotive, Healthcare Facilities</td><td>Groups destinations associated with commercial and service activities, encompassing routine visits, discretionary travel, and essential medical trips.</td></tr><tr><td>Cultural &amp; Entertainment</td><td>Leisure, Education, Sports, Culture</td><td>Combines destinations for educational, cultural, sporting, and leisure activities, with travel demand shaped by activity schedules, weather conditions, and holidays.</td></tr><tr><td>Tourism &amp; Travel</td><td>Scenic Spots, Hotels, Lodging</td><td>Groups tourist attractions and accommodation facilities associated with visitor activity, holiday travel, and movement between regions.</td></tr><tr><td>Transportation Facilities</td><td>Transportation Hubs, including Stations, Air- ports, and Ports</td><td>Represents mobility terminals connecting local and regional travel, with demand influenced by transport schedules and holiday travel patterns.</td></tr><tr><td>Residential &amp; Office</td><td>Residential, Business, Office Areas</td><td>Captures residential origins and workplace destinations that structure routine commuting and recurring travel between regions.</td></tr></table>

FORECASTING PERFORMANCE ACROSS THREE CITIES. BEST AND SECOND BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. IMPROV. REPORTS PERCENTAGE REDUCTIONS IN MAE, RMSE, AND WAPE AND ABSOLUTE GAINS IN $R ^ { 2 }$ RELATIVE TO THE STRONGEST BASELINE FOR EACH CITY AND METRIC.
<table><tr><td rowspan="2">Model</td><td colspan="4">Xi&#x27;an</td><td colspan="4">Shenyang</td><td colspan="4">Urumqi</td></tr><tr><td>MAE</td><td>RMSE</td><td>WAPE (%)</td><td> ${ \bf R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td>WAPE (%)</td><td> ${ \bf R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td>WAPE (%)</td><td> ${ \bf R } ^ { 2 }$ </td></tr><tr><td>DSTAGNN</td><td>0.8386</td><td>1.3001</td><td>25.58</td><td>0.8725</td><td>0.8962</td><td>1.4237</td><td>25.45</td><td>0.9079</td><td>1.2280</td><td>1.9650</td><td>20.35</td><td>0.9277</td></tr><tr><td>STGCN</td><td>0.8425</td><td>1.3407</td><td>25.70</td><td>0.8644</td><td>0.8936</td><td>1.4763</td><td>25.37</td><td>0.9009</td><td>1.1732</td><td>1.9935</td><td>19.44</td><td>0.9256</td></tr><tr><td>GWaveNet</td><td>0.9131</td><td>1.4512</td><td>27.85</td><td>0.8411</td><td>0.9498</td><td>1.5739</td><td>26.97</td><td>0.8874</td><td>1.2415</td><td>2.0970</td><td>20.57</td><td>0.9176</td></tr><tr><td>SARAF</td><td>0.8798</td><td>1.3426</td><td>26.84</td><td>0.8640</td><td>0.9013</td><td>1.4336</td><td>25.59</td><td>0.9066</td><td>1.1827</td><td>1.9098</td><td>19.60</td><td>0.9317</td></tr><tr><td>LSTM</td><td>0.8936</td><td>1.4197</td><td>27.26</td><td>0.8479</td><td>0.9324</td><td>1.5417</td><td>26.47</td><td>0.8920</td><td>1.2011</td><td>2.0200</td><td>19.90</td><td>0.9236</td></tr><tr><td>TACTiS</td><td>0.9948</td><td>1.5300</td><td>30.34</td><td>0.8234</td><td>1.0760</td><td>1.7342</td><td>30.55</td><td>0.8633</td><td>1.6163</td><td>2.6308</td><td>26.78</td><td>0.8704</td></tr><tr><td>USTD</td><td>1.1301</td><td>1.7630</td><td>34.47</td><td>0.7655</td><td>1.1893</td><td>1.9407</td><td>33.77</td><td>0.8288</td><td>1.6421</td><td>2.7619</td><td>27.21</td><td>0.8571</td></tr><tr><td>DiffSTG</td><td>1.6902</td><td>2.3281</td><td>51.55</td><td>0.5910</td><td>1.8056</td><td>2.7049</td><td>51.27</td><td>0.6675</td><td>1.9059</td><td>3.1326</td><td>31.58</td><td>0.8162</td></tr><tr><td>Multi-STG</td><td>0.7556</td><td>1.1982</td><td>23.05</td><td>0.8917</td><td>0.7826</td><td>1.2836</td><td>22.22</td><td>0.9251</td><td>1.0189</td><td>1.6860</td><td>16.88</td><td>0.9468</td></tr><tr><td>Improv.</td><td>9.90%</td><td>7.84%</td><td>9.89%</td><td>+0.0192</td><td>12.42%</td><td>PABLE VI</td><td>12.42%</td><td>+0.0172</td><td>13.15%</td><td>11.72%</td><td>13.17%</td><td>+0.0151</td></tr></table>

ABLATION RESULTS ACROSS THREE CITIES. STG (W/O CONTEXTUAL FEATURES), A-STG (W/O LLM-EMPOWERED SEMANTIC ENHANCEMENT), B-STG (W/O MULTIMODAL ATTENTIONAL FUSION), C-STG (W/O SEMANTIC ENHANCED PREDICTION HEAD), MULTI-STG (FULL MODEL).
<table><tr><td rowspan="2">Model</td><td colspan="4">Xi&#x27;an</td><td colspan="4">Shenyang</td><td colspan="4">Urumqi</td></tr><tr><td>MAE</td><td>RMSE</td><td>WAPE (%)</td><td> ${ \bf R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td>WAPE (%)</td><td> ${ \bf R } ^ { 2 }$ </td><td>MAE</td><td>RMSE</td><td>WAPE (%)</td><td> ${ \bf R } ^ { 2 }$ </td></tr><tr><td>STG</td><td>0.8319</td><td>1.2952</td><td>25.38</td><td>0.8735</td><td>0.8621</td><td>1.3864</td><td>24.48</td><td>0.9126</td><td>1.1305</td><td>1.8247</td><td>18.73</td><td>0.9377</td></tr><tr><td>A-STG</td><td>0.8124</td><td>1.2735</td><td>24.79</td><td>0.8777</td><td>0.8402</td><td>1.3643</td><td>23.85</td><td>0.9153</td><td>1.1048</td><td>1.7866</td><td>18.30</td><td>0.9402</td></tr><tr><td>B-STG</td><td>0.7972</td><td>1.2528</td><td>24.31</td><td>0.8817</td><td>0.8247</td><td>1.3416</td><td>23.41</td><td>0.9182</td><td>1.0814</td><td>1.7563</td><td>17.91</td><td>0.9422</td></tr><tr><td>C-STG</td><td>0.7815</td><td>1.2307</td><td>23.83</td><td>0.8857</td><td>0.8081</td><td>1.3188</td><td>22.94</td><td>0.9209</td><td>1.0566</td><td>1.7275</td><td>17.50</td><td>0.9442</td></tr><tr><td>Multi-STG</td><td>0.7556</td><td>1.1982</td><td>23.05</td><td>0.8917</td><td>0.7826</td><td>1.2836</td><td>22.22</td><td>0.9251</td><td>1.0189</td><td>1.6860</td><td>16.88</td><td>0.9468</td></tr></table>

Within each city, MAE varies by at most 1.3% across the tested values. No single value performs best across all three cities. Xi’an and Shenyang achieve the lowest MAE at $\lambda =$ 0.5, while Urumqi achieves the lowest MAE at $\lambda = 0 . 0 5$ These findings suggest that performance is relatively stable across the tested residual weights within [0.05, 0.5].

2) Sensitivity to Historical and Prediction Lengths: To assess sensitivity to historical input length (seq\_len) and prediction horizon (pred\_len), we vary each parameter independently over {6, 12, 24} hours, yielding a 3 × 3 experimental grid. All other hyperparameters remain fixed. Fig. 6 summarizes the results.

Longer input sequences generally reduce MAE, while longer prediction horizons generally increase forecasting error. Across all three cities, the lowest MAE occurs with 24 hours of history and a 6-hour prediction horizon, yielding 0.7499, 0.7784, and 1.0143 for Xi’an, Shenyang, and Urumqi, respectively. The main experiments use the same historical window to predict the next 12 hours, yielding MAE values of 0.7556, 0.7826, and 1.0189.

The lower errors at a 6-hour horizon reflect an easier forecasting task. We therefore retain a 12-hour prediction horizon for the main comparisons and report the configuration with 24 hours of history and 6 hours of prediction as the best observed setting within the sensitivity grid.

## G. Sensitivity to Data Volume

To examine the effect of training data scale, we train Multi-STG on the Xi’an dataset using 25%, 50%, 75%, and 100% of the training windows while keeping all other settings fixed. As shown in Fig. 7, performance improves as the training fraction increases. From 25% to 100%, MAE decreases from 0.8144 to 0.7556, RMSE from 1.2864 to 1.1982, and WAPE from 24.84% to 23.05%, while $R ^ { 2 }$ increases from 0.8751 to 0.8917. These moderate but consistent gains suggest that

![](images/7856368e31f53e7c7211519de9d4bf3663aaeae52545ad8b72850a1a8772e339.jpg)  
(a) Xi'an

![](images/3ff980b40979feb5e4083e519e5f13769d83fbce5b2958fa6eb7d02c5a258e88.jpg)  
(b) Shenyang

![](images/2f68202cecc77342a30d532481c3d2959b2f212b057e6cf29f4932f354790688.jpg)  
(c) Urumqi

Fig. 5. Sensitivity of the residual weight λ in the Semantic-Enhanced Prediction Head across three cities.  
![](images/af59e8900f5c92893e66342f634020066ff49e2c1f8cf3a26390a3f25912134c.jpg)

Fig. 6. Sensitivity analysis over historical length (seq len) and prediction length (pred len).  
![](images/bedbb5a5d6c062672465d268e895dc02b6237fcb656a4debdfeb98422d23b791.jpg)

![](images/8e6af27327348e4e4195b997eefebc468edcf6c1996be4dfbc282f846f673c7a.jpg)  
Fig. 7. Sensitivity analysis of training data volume on model performance. Left: MAE and RMSE; right: $R ^ { 2 }$ and WAPE.

Multi-STG benefits from additional training data, with gradual improvements across the evaluated data fractions.

## H. Cross-City Generalization

To assess transferability, we train Multi-STG on each source city and evaluate predictions on the other two target cities. Together with in-city evaluations, these experiments form a $3 \times 3$ matrix. We compare Multi-STG with DSTAGNN, the strongest baseline on Xi’an. Normalization statistics are computed from the training segment of each target city, and both models follow the same data split and evaluation protocol. Fig. 8 reports MAE, RMSE, WAPE, and $R ^ { 2 }$ , using a shared color scale for each metric across both models. Diagonal entries reproduce the results from Table V. For evaluations within Xi’an, Shenyang, and Urumqi, respectively, Multi-STG achieves MAE values of 0.76, 0.78, and 1.02 and RMSE values of 1.20, 1.28, and 1.69. The corresponding DSTAGNN results are 0.84, 0.90, and 1.23 for MAE and 1.30, 1.42, and 1.97 for RMSE. Multi-STG performs better across all four metrics in each city, with MAE reductions of 9.90%, 12.68%, and 17.03%. For transfers between cities, MAE ranges from

1.08 to 1.96 for Multi-STG and from 1.23 to 2.50 for DSTAGNN. Across all nine source-target city combinations, Multi-STG performs better on seven combinations for each metric. DSTAGNN achieves lower MAE and WAPE when transferring from Xi’an to Shenyang, and better RMSE and $R ^ { 2 }$ when transferring from Urumqi to Xi’an. For transfer from Urumqi to Shenyang, DSTAGNN performs better across all four metrics, achieving MAE of 1.36 versus 1.58, RMSE of 2.18 versus 2.64, WAPE of 38.54% versus 44.96%, and $R ^ { 2 }$ of 0.783 versus 0.683. Averaged over all nine combinations, Multi-STG reduces mean MAE from 1.47 to 1.30 and mean RMSE from 2.25 to 2.01, indicating an overall advantage despite variation across transfer settings.

## VI. CONCLUSION

In this paper, we present Multi-STG, a unified framework that integrates LLM-derived semantic priors with spatiotemporal graph learning to improve regional private vehicle flow prediction under adverse weather conditions. The framework combines regional feature modeling, LLM-based semantic enhancement, multimodal attentional fusion, a Graph Transformer backbone, and a semantic prediction head with gated residual calibration. Experiments on private vehicle datasets from three cities demonstrate improved forecasting accuracy, while sensitivity analyses indicate relatively stable performance across the tested hyperparameter settings. Dependence on LLM availability, potential semantic bias, and additional computational costs remain practical considerations. Future work can explore lightweight adaptation, fusion that accounts for semantic uncertainty, and model distillation for efficient deployment. These findings support the potential of combining contextual knowledge from language models with spatiotemporal learning for urban traffic forecasting.

## REFERENCES

[1] R. Li, F. Zhang, T. Li, N. Zhang, and T. Zhang, “Dmgan: Dynamic multihop graph attention network for traffic forecasting,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 9, pp. 9088–9101, 2022.

[2] C. Rong, T. Li, J. Feng, and Y. Li, “Inferring origin-destination flows from population distribution,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 603–613, 2021.

[3] B. Liu, T. Li, Z. Xiao, W. Chen, Y. Hei, J. Ding, and G. Chen, “A multigranularity self-guiding graph diffusion model for predicting private car activity,” IEEE Transactions on Intelligent Transportation Systems, vol. 27, no. 6, pp. 7172–7186, 2026.

![](images/93c1cd550940a8c7669b89e8c129a3cf2f63863837d263df12e267a67ce99021.jpg)

![](images/3bd5c7b593f91876f00ab5045e8593e70bb3f961746767d0dd7a525435d70f2f.jpg)

![](images/34396b36a0cb29a80f70c287dc8893cf9c7ab4ef005cb95ffc6d80a9dbaa2567.jpg)

![](images/aa1ce136deab936df82012a1c7987a6be91ee216890f48a9f96bd1fa0e03131f.jpg)

![](images/09000de54d61da0b7866628838412cdf302262a4764ef48e78b4be97f9ea28f1.jpg)

![](images/3d1bc611ccb111f4231360654fce754ec306fc5347764c5e8b8de2be0401d159.jpg)  
(a) Multi-STG

![](images/b8bce526608bf7819b03f0b85272828ca9d6142238aa75830649afdd6dabbd28.jpg)

(b) DSTAGNN  
![](images/68ef5db2a9dc8c9658b965272c8a4dd987b385d033c2e4954b5c2c5619e9f6a5.jpg)  
Fig. 8. Cross-city generalization. (a) Multi-STG. (b) DSTAGNN. Each subfigure reports MAE, RMSE, WAPE, and R<sup>2</sup>. Within each heatmap, rows denote source cities and columns denote target cities; XA, SY, and UR denote Xi’an, Shenyang, and Urumqi, respectively. Colors are normalized per metric across both methods, and darker indicates better performance. The diagonal corresponds to in-city evaluation.

[4] C. Liu, Z. Xiao, W. Long, T. Li, H. Jiang, and K. Li, “Vehicle trajectory data processing, analytics, and applications: A survey,” ACM Computing Surveys, vol. 57, no. 9, pp. 1–36, 2025.

[5] M. Diao, H. Kong, and J. Zhao, “Impacts of transportation network companies on urban mobility,” Nature Sustainability, vol. 4, no. 6, pp. 494–500, 2021.

[6] D. Hu, L. Chen, H. Fang, Z. Fang, T. Li, and Y. Gao, “Spatio-temporal trajectory similarity measures: A comprehensive survey and quantitative study,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 5, pp. 2191–2212, 2024.

[7] R. Wang, T. Li, Z. Xiao, S. Zhang, Z. Tang, and K. Li, “Joint optimization of mec offloading and traffic signal control via diffusionbased hierarchical drl,” in 2026 IEEE/ACM International Symposium on Quality of Service (IWQoS). IEEE, 2026, pp. 1–10.

[8] H.-W. Wang, Z.-R. Peng, D. Wang, Y. Meng, T. Wu, W. Sun, and Q.- C. Lu, “Evaluation and prediction of transportation resilience under extreme weather events: A diffusion graph convolutional approach,” Transportation Research Part C: Emerging Technologies, vol. 115, p. 102619, 2020.

[9] C. Li, W. Liu, and H. Yang, “Deep causal inference for understanding the impact of meteorological variations on traffic,” Transportation Research Part C: Emerging Technologies, vol. 165, p. 104744, 2024.

[10] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008.

[12] W. Jiang and J. Luo, “Graph neural network for traffic forecasting: A survey,” Expert Systems with Applications, vol. 207, p. 117921, 2022.

[13] M. Zhang, T. Li, Y. Li, and P. Hui, “Multi-view joint graph representation learning for urban region embedding,” in Proceedings ofthe twenty-ninth international conference on international joint conferences on artificial intelligence, 2020, pp. 4431–4437.

[14] L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, and H. Li, “T-gcn: A temporal graph convolutional network for traffic prediction,” IEEE transactions on intelligent transportation systems, vol. 21, no. 9, pp. 3848–3858, 2020.

[15] R. Jiang, Z. Wang, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, S. Fukushima, and T. Suzumura, “Spatio-temporal meta-graph learning for traffic forecasting,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 7, 2023, pp. 8078–8086.

[16] A. Koesdwiady, R. Soua, and F. Karray, “Improving traffic flow prediction with weather information in connected cars: A deep learning approach,” IEEE Transactions on Vehicular Technology, vol. 65, no. 12, pp. 9508–9517, 2016.

[17] H. Zhong, J. Wang, C. Chen, J. Wang, D. Li, and K. Guo, “Weather interaction-aware spatio-temporal attention networks for urban traffic flow prediction,” Buildings, vol. 14, no. 3, p. 647, 2024.

[18] K. Wang, L. Liu, Y. Liu, G. Li, F. Zhou, and L. Lin, “Urban regional function guided traffic flow prediction,” Information Sciences, vol. 634, pp. 308–320, 2023.

[19] T. Li, Y. Xi, H. Wang, Y. Li, S. Tarkoma, and P. Hui, “Learning representations of satellite imagery by leveraging point-of-interests,” ACM Transactions on Intelligent Systems and Technology, vol. 14, no. 4, pp. 1–32, 2023.

[20] Y. Xi, T. Li, H. Wang, Y. Li, S. Tarkoma, and P. Hui, “Beyond the first law of geography: Learning representations of satellite imagery by leveraging point-of-interests,” in Proceedings of the ACM Web Conference 2022, 2022, pp. 3308–3316.

[21] C. Liu, Q. Xu, H. Miao, S. Yang, L. Zhang, C. Long, Z. Li, and R. Zhao, “Timecma: Towards llm-empowered multivariate time series forecasting via cross-modality alignment,” in AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, T. Walsh, J. Shah, and Z. Kolter, Eds. AAAI Press, 2025, pp. 18 780–18 788. [Online]. Available: https://doi.org/10.1609/aaai.v39i18.34067

[22] S. Zhang, T. Li, Z. Xiao, H. Du, and K. Huang, “Lsdm: Llm-enhanced spatio-temporal diffusion model for service-level mobile traffic prediction,” IEEE Transactions on Networking, pp. 1–1, 2026.

[23] Z. Xiao, R. Wang, J. Bai, T. Li, S. Zhang, K. Li, and Z. Han, “An llm-enhanced conditional diffusion model for mobile traffic prediction,” IEEE Communications Magazine, vol. 63, no. 9, pp. 60–67, 2025.

[24] T. B. Brown et al., “Language models are few-shot learners,” NeurIPS, vol. 33, pp. 1877–1901, 2020.

[25] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 22 199–22 213.

[26] J. Gong, Y. Liu, T. Li, H. Chai, X. Wang, J. Feng, C. Deng, D. Jin, and Y. Li, “Empowering spatial knowledge graph for mobile traffic prediction,” in Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems, 2023, pp. 1–11.

[27] J. Gong, T. Li, H. Wang, Y. Liu, X. Wang, Z. Wang, C. Deng, J. Feng, D. Jin, and Y. Li, “KGDA: A knowledge graph driven decomposition approach for cellular traffic prediction,” ACM Transactions on Intelligent Systems and Technology, vol. 15, no. 6, pp. 1–22, 2024.

[28] M. Zhang, T. Li, H. Shi, Y. Li, and P. Hui, “A decomposition approach for urban anomaly detection across spatiotemporal data,” in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 2019, pp. 6043–6049.

[29] C. Song, Y. Lin, S. Guo, and H. Wan, “Spatial-temporal synchronous graph convolutional networks: A new framework for spatial-temporal

network data forecasting,” in The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020. AAAI Press, 2020, pp. 914–921. [Online]. Available: https://doi.org/10.1609/aaai.v34i01.5438

[30] S. Zhang, T. Li, S. Hui, G. Li, Y. Liang, L. Yu, D. Jin, and Y. Li, “Deep transfer learning for city-scale cellular traffic generation through urban knowledge graph,” in Proceedings ofthe 29th ACM SIGKDD conference on knowledge discovery and data mining, 2023, pp. 4842–4851.

[31] Q. Yu, H. Wang, T. Li, D. Jin, X. Wang, L. Zhu, J. Feng, and C. Deng, “Network traffic overload prediction with temporal graph attention convolutional networks,” in 2022 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2022, pp. 885– 890.

[32] C. Zheng, X. Fan, C. Wang, and J. Qi, “GMAN: A graph multi-attention network for traffic prediction,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 01, 2020, pp. 1234–1241.

[33] M. Xu, W. Dai, C. Liu, X. Gao, W. Lin, G.-J. Qi, and H. Xiong, “Spatial-temporal transformer networks for traffic flow forecasting,” arXiv preprint arXiv:2001.02908, 2020.

[34] F. Li, J. Feng, H. Yan, G. Jin, F. Yang, F. Sun, D. Jin, and Y. Li, “Dynamic graph convolutional recurrent network for traffic prediction: Benchmark and solution,” ACM Transactions on Knowledge Discovery from Data, vol. 17, no. 1, pp. 1–21, 2023.

[35] Q. Long, H. Wang, T. Li, L. Huang, K. Wang, Q. Wu, G. Li, Y. Liang, L. Yu, and Y. Li, “Practical synthetic human trajectories generation based on variational point processes,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 4561–4571.

[36] B. Zhang, Z. Li, K. Wang, M. Sun, and N. S. Lam, “Lg-fair: A localglobal fairness framework for spatio-temporal event forecasting,” ACM Transactions on Intelligent Systems and Technology, 2026.

[37] M. Zhang, T. Li, and P. Hui, “Cityneuro: Towards location and time prediction for urban abnormal events,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 12, pp. 12 125–12 138, 2023.

[38] H. Chai, T. Li, F. Jiang, S. Zhang, and Y. Li, “Knowledge guided conditional diffusion model for controllable mobile traffic generation,” in Companion Proceedings of the ACM Web Conference 2024, 2024, pp. 851–854.

[39] H. Wen, Y. Lin, Y. Xia, H. Wan, Q. Wen, R. Zimmermann, and Y. Liang, “Diffstg: Probabilistic spatio-temporal graph forecasting with denoising diffusion models,” in Proceedings ofthe 31st ACM International Conference on Advances in Geographic Information Systems, 2023, pp. 1–12.

[40] J. Hu, X. Liu, Z. Fan, Y. Liang, and R. Zimmermann, “Towards unifying diffusion models for probabilistic spatio-temporal graph learning,” in Proceedings of the 32nd ACM International Conference on Advances in Geographic Information Systems. Atlanta GA USA: ACM, Oct. 2024, pp. 135–146.

[41] Q. Long, C. Rong, T. Li, and Y. Li, “Dynamic population distribution aware human trajectory generation with diffusion model,” ACM Transactions on Intelligent Systems and Technology, vol. 17, no. 3, pp. 1–23, 2026.

[42] S. Hui, H. Wang, T. Li, X. Yang, X. Wang, J. Feng, L. Zhu, C. Deng, P. Hui, D. Jin et al., “Large-scale urban cellular traffic generation via knowledge-enhanced gans with multi-periodic patterns,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 4195–4206.

[43] T. Li, S. Hui, S. Zhang, H. Wang, Y. Zhang, P. Hui, D. Jin, and Y. Li, “Mobile user traffic generation via multi-scale hierarchical gan,” ACM Transactions on Knowledge Discovery from Data, vol. 18, no. 8, pp. 1–19, 2024.

[44] Y. Yan, H. Wen, S. Zhong, W. Chen, H. Chen, Q. Wen, R. Zimmermann, and Y. Liang, “UrbanCLIP: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web,” in Proceedings of the ACM Web Conference 2024, 2024, pp. 4006–4017.

[45] J. Feng, Y. Du, J. Zhao, and Y. Li, “AgentMove: A large language model based agentic framework for zero-shot next location prediction,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 1322–1338.

[46] J. Jin, Y. Song, D. Kan, H. Zhu, X. Sun, Z. Li, X. Sun, and J. Zhang, “Urban region pre-training and prompting: A graph-based approach,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, 2025, pp. 1071–1082.

[47] Z. Li, L. Xia, J. Tang, Y. Xu, L. Shi, L. Xia, D. Yin, and C. Huang, “UrbanGPT: Spatio-temporal large language models,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 5351–5362.

[48] J. Feng, T. Liu, Y. Du, S. Guo, Y. Lin, and Y. Li, “CityGPT: Empowering urban spatial cognition of large language models,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, 2025, pp. 591–602.

[49] S. Zhang, D. Fu, W. Liang, Z. Zhang, B. Yu, P. Cai, and B. Yao, “TrafficGPT: Viewing, processing and interacting with traffic foundation models,” Transport Policy, vol. 150, pp. 95–105, 2024.

[50] J. Shao, S. Li, K. Zhang, Y. Guo, and M. Li, “Leveraging LLMs for cross-city and cross-scenario traffic speed forecasting under abnormal weather conditions,” Transportation Research Part C: Emerging Technologies, vol. 188, p. 105702, 2026.

[51] T. Li, A. Alhilal, A. Zhang, M. A. Hoque, D. Chatzopoulos, Z. Xiao, Y. Li, and P. Hui, “Driving big data: A first look at driving behavior via a large-scale private car dataset,” in 2019 IEEE 35th International Conference on Data Engineering Workshops (ICDEW). IEEE, 2019, pp. 61–68.

[52] B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,” in Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI), 2018, pp. 3634–3640.

[53] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, “Graph wavenet for deep spatial-temporal graph modeling,” in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019, S. Kraus, Ed. ijcai.org, 2019, pp. 1907–1913. [Online]. Available: https://doi.org/10.24963/ijcai.2019/264

[54] S. Lan, Y. Ma, W. Huang, W. Wang, H. Yang, and P. Li, “DSTAGNN: dynamic spatial-temporal aware graph neural network for traffic flow forecasting,” in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds.,´ vol. 162. PMLR, 2022, pp. 11 906–11 917. [Online]. Available: https://proceedings.mlr.press/v162/lan22a.html

[55] A. Drouin, E. Marcotte, and N. Chapados, “TACTiS: Transformer-<sup>´</sup> attentional copulas for time series,” in Proceedings of the 39th International Conference on Machine Learning, vol. 162. PMLR, 2022, pp. 5447–5493.

[56] S. Zhou, H. Schoner, Z. Wu, E. Fouch ¨ e, I. Wilson, and S. Wang,´ “Stationarity-aware retrieval-augmented time series forecasting,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, ser. KDD ’26. Association for Computing Machinery, 2026, pp. 6875–6886. [Online]. Available: https://doi.org/10.1145/3770855.3817813