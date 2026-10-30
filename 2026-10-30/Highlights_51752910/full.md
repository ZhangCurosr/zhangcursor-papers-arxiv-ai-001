Graphical Abstract

Multistep and Multivariate Time Series Explainability with GradientSHAP: A Wind Power Case Study

Robbie Southam, Zhong Fan, Ke Li

## Highlights

Multistep and Multivariate Time Series Explainability with GradientSHAP: A Wind Power Case Study

Robbie Southam, Zhong Fan, Ke Li

• We present a novel application of GradientSHAP to day-ahead wind power forecasting, addressing a critical gap in explainable AI for renewable energy systems.

• We provide a practical guide for deriving meaningful insights from SHAP methods, combining domain knowledge with innovative visualizations tailored for complex time series data.

• Our results reveal critical temporal, spatial, and feature-specific dynamics in wind power forecasting, validating known relationships and uncovering novel interactions through GradientSHAP analysis.

• We advance the limited body of work on multivariate, multitarget time series explainability, which is crucial for improving renewable energy forecasting and operational transparency.

# Multistep and Multivariate Time Series Explainability with GradientSHAP: A Wind Power Case Study

Robbie Southam<sup>a,∗</sup>, Zhong Fan<sup>a</sup>, Ke Li<sup>b</sup>

<sup>a</sup>Department of Engineering, University of Exeter, Exeter, EX4 4QF, United Kingdom <sup>b</sup>Department of Computer Science, University of Exeter, Exeter, EX4 4QF, United Kingdom

## Abstract

As the efects of climate change become increasingly apparent, the demand for renewable energy sources such as wind power continues to grow. However, renewables’ variable and intermittent nature presents significant challenges in their integration into energy grids. Artificial intelligence (AI) has emerged as a powerful tool to address these issues. Still, its adoption in the energy sector has been hindered by the opacity of many AI models, which often function as black boxes. This paper introduces a carefully adapted GradientSHAP approach to provide interpretability for multivariate, multistep wind power forecasting — a largely underexplored area. This capability is critical for day-ahead grid management and, to the best of our knowledge, this is the first study to apply GradientSHAP in the context of wind power forecasting. To address the inherent dimensionality, we developed novel visualizations that deliver both local and global explanations, uncovering intricate relationships between environmental factors, site-specific conditions, and temporal dynamics. While GradientSHAP is not an out-of-the-box solution, our findings demonstrate that, with thoughtful handling of SHAP values and domain expertise, it can ofer valuable insights into model behaviour, supporting more informed and trustworthy decision-making. Nevertheless, its complexity poses challenges for practical implementation. Future research should focus on streamlining these interpretability methods to make them more actionable for routine deployment in critical energy applications.

Keywords: Forecasting, Renewable Energy, Explainable Artificial, Intelligence, Interpretability

## 1. Introduction

Achieving global net-zero goals necessitates urgent, transformative eforts to reduce carbon emissions and accelerate the shift to renewable energy sources (RESs), prompting a fundamental reshaping of the global energy framework [1]. Central to this transition is the rapid deployment of RESs such as wind[1], [2], [3]. Wind power, as one of the fastest-growing RESs, has emerged as a critical driver in reducing greenhouse gas (GHG) emissions, while also presenting substantial economic benefits [4], [5]. However, the variability and intermittency of renewable energy pose significant challenges for integration, impacting grid stability and operational reliability [1], [3], [6].

Accurate short-term wind power forecasting is indispensable for addressing these challenges. It plays a vital role in managing grid load efectively, ensuring reliable energy dispatch, and optimising penetration and connection rate [7], [8], [9], [10]. However, despite substantial advancements in forecasting accuracy, top-performing models often function as “black boxes,” with their decision processes obscured. This lack of transparency hinders stakeholder trust and limits the adoption of AI in high-stakes domains such as energy[11].

Explainable Artificial Intelligence (XAI) approaches, particularly post-hoc feature attribution methods such as Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP)[12], have gained traction as tools to illuminate model logic and improve interpretability [6], [7], [8], [12]. These methods enable grid operators and policymakers to glean insights into model behaviour by highlighting the contribution of each feature to model outputs, enhancing their capacity to manage grid stability and develop informed policies[6], [13], [14].

SHAP, grounded in cooperative game theory, apportions feature importance based on each feature’s marginal contribution to model outputs. This approach satisfies key principles of local accuracy, missingness, and consistency, ensuring fair and reliable attribution [12]. However, traditional SHAP implementations, such as DeepSHAP and KernelSHAP, are limited by computational demands or strict assumptions like linearity or feature independence [15], [16]. GradientSHAP, an extension specifically designed for diferentiable models, addresses these constraints by leveraging expected gradients (EG) [17] to provide more eficient and reliable explanations while maintaining the same desirable principles as SHAP[17].

To the best of our knowledge, this paper represents the first application of GradientSHAP in wind power forecasting and extends the very limited body of work on multivariate, multistep time series explainability. Through a comprehensive analysis of GradientSHAP values, supported by several novel visualizations, we uncover known physical relationships and explore intriguing feature interactions. Additionally, the model’s predictions exhibited a strong reliance on typical temporal patterns, including diurnal and, to a lesser extent, seasonal trends.

While GradientSHAP, combined with domain expertise, demonstrated considerable potential in enhancing interpretability and fostering transparency, its reliance on expert knowledge limits its broader applicability.

Our analysis, which largely exposes the model’s decision process, reveals significant alignment with real-world dynamics in wind power forecasting. This demonstrates the utility of GradientSHAP, when combined with domain knowledge, for enhancing interpretability in short-term forecasting models. By improving transparency, this approach empowers grid operators and policymakers to make more informed decisions, contributing to greater grid stability and operational eficiency. Ultimately, by fostering trust in AI systems, this work supports the broader adoption of explainable AI in the energy sector.

The remainder of this paper is organised as follows. 2 reviews relevant literature on feature attribution in energy forecasting, with a particular focus on SHAP and its applications in multivariate, multistep forecasting tasks. Section 3 outlines the GradientSHAP methodology and describes the feature engineering and model architecture used in our forecasting approach. Section 4 presents the results, starting with global and local SHAP analyses and examining feature interactions, followed by a detailed examination of temporal patterns and the seasonal dynamics of feature importance. Finally, Section 5 concludes with a summary of findings and directions for future work.

## 2. Related Work

Despite SHAP’s prominence as a feature attribution method in energy forecasting, most applications remain focused on demand-side tasks[18], [19], [20], [21], with notably fewer studies on supply-side forecasting[22], [23], [24]. Additionally, these studies largely employ Recurrent Neural Network (RNN)- based models, ensemble methods, and Artificial Neural Networks (ANNs) with limited innovation in the SHAP methodology itself. Significantly, few address the complexity of multivariate or multistep timeseries tasks[23], [24], [25] and even fewer address both[19], underscoring the need for further exploration and novel approaches[26].

For example, on the demand side, van Zyl et al. evaluated the eficacy of Grad-CAM and SHAP for feature selection in a multistep load forecasting scenario using a Convolutional Neural Network (CNN), generating explanations with DeepSHAP[24]. They found both methods to enhance forecasting accuracy and model eficiency by identifying and excluding irrelevant features. Although Grad-CAM was computationally more eficient, SHAP better-captured features that degraded forecast accuracy, though it inaccurately ranked feature importance, likely due to DeepSHAP’s assumptions. The authors highlight the need for refinement in XAI methods within timeseries forecasting to improve interpretability.

To the best of our knowledge, only Białek et al. tackles the problem of both multivariate and multistep time series by applying SHAP for feature attribution in a heat demand forecasting model for the Warsaw District Heating Network[19]. Their use of DeepSHAP aimed to provide both local and global explanations. Due to the high degree of correlation among features, the authors aggregated SHAP values to capture meaningful groups, such as aggregating temperature-related features to observe their collective influence on heat demand. To interpret temporal dependencies, they employed visualisation methods such as SHAP attribution versus cos\_yday for seasonality, aggregated feature group plots, 3D scatter plots for delayed temporal features, and box plots to illustrate distributions across values of temporal features. While this study demonstrates that SHAP can ofer valuable insights for multistep multivariate forecasting, the authors note that interpreting the results still requires significant domain expertise. Moreover, the analysis relied heavily on engineered features without state-of-the-art models, limiting its generalisability and applicability to more advanced model architectures. Importantly, the independence assumption inherent in DeepSHAP further restricts the accuracy of the attributions, given the high correlation between many features.

In the context of energy forecasting Maarif et al. investigates energy consumption predictions using LSTM networks and SHAP explainability[21]. The authors developed single-layer, double-layer, and bidirectional LSTM models, and tested them on a steel industry dataset, finding that the double-layer model ofered the best performance. SHAP summary plots were used for interpretability, specifically layered violin and bar plots, to display the contribution of each feature, ranked in descending order. The bar chart presents the mean absolute SHAP values for each feature, while the layered violin plot provides a more detailed view by showing the distribution of SHAP values for each feature along with corresponding feature values. This analysis highlighted the significant influence of leading current reactive power and time of day, with minimal impact from features like CO\_2 emissions and weekly status.

While GradientSHAP appears highly suitable for the context of this paper and forecasting problems as a whole (see Section 3.4), its application within timeseries forecasting, particularly in the energy domain, remains limited. There are only a handful of studies in time series contexts [24], [27], [28], [29]. Although some studies indicate that GradientSHAP outperforms alternative methods [30], [31], to the best of our knowledge, it is limited to a single paper within the energy domain[24]. Furthermore, many studies employing GradientSHAP omit the number of background samples used, which complicates comparisons between methods.

While [32] present perturbation and sequence analysis techniques to benchmark feature attribution methods in univariate timeseries forecasting[32],

Veerappa et al. adapt this to multivariate classification and incorporates GradientSHAP into the benchmark[33]. Specifically, they evaluate GradientSHAP, DeepSHAP, path-integrated gradients (PIG) and a time-series adapted version of LIME called Limefor-Time in the maritime domain. Additionally, the authors extend the prior evaluation methodology with a technique called “noise” to assess the quality of explanations for both perturbation and sequence analysis; investigating how much the model’s output is afected if noise is added to the input. The authors deem these methods suitable for multivariate time series, showing that DeepSHAP and GradientSHAP work robustly in all cases; while being one of the few papers to indicate the number of background samples (5000). While the performance of DeepSHAP and GradientSHAP were essentially the same, GradientSHAP has significantly less execution time than all other methods and ∼ 13% that of DeepSHAP. Further, they suggest the use of temporal heatmaps for more intuitive explanations.

This paper seeks to address these gaps by applying GradientSHAP in a novel multivariate, multistep energy forecasting context, an area that remains largely unexplored. By leveraging GradientSHAP’s computational eficiency, and it’s ability to operate with reduced assumptions compared to traditional methods, we aim to enhance both the interpretability and operational viability of modern energy forecasting models, as well as provide a tangible example of utilising GradientSHAP for domain experts.

## 3. Methodology

## 3.1. Dataset

We use WIND[34] by The National Renewable Energy Laboratory (NREL), providing seven years of meteorological data in 5 minute granularity from more than 126, 000 sites.

## 3.1.1. Feature Selection

To ensure varied environmental conditions, 4 states have been selected spread out around the US: Washington, New York, Iowa and Texas. Three sites are sampled for each, giving a total of 12 sites, containing five years. The resulting data were split into the train, validation and test sets temporally. Here, the years 2008-2010, 2011 and 2012 were used for each. Additionally, we reduce the granularity of the dataset to an hour by averaging.

To determine the number of lags we use the Autocorrelation Function (ACF), Partial Autocorrelation Function (PACF), and Cross-Correlation Function (CCF)[35] on the training set and determine that a lag of 48 is best for balancing the model learning and insights we can obtain from the explanation. We also test for stationarity using Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests before this, to ensure interpretations of correlation are meaningful.

Similarly, we average the dataset by month, day, year, and hour to observe any seasonality. We find that the most prevalent trends are both hourly and monthly.

## 3.1.2. Preprocessing

To ensure the model was able to make the most of the data, we performed additional encoding on some features. To ensure that daily and seasonal patterns could be captured better, we used cyclical encoding on the day and month features. Similarly, we performed the same on wind direction to retain the circular structure and prevent discontinuity between 0 deg and 360 deg. Finally, since we only use 11 sites from such a big pool, we integer encode the site ID.

To generate time-series samples for training the model, we utilise the rolling window approach. At each time step t, a window W was created that included 48 historical time steps for input and 24 time steps for prediction. An example of this can be seen in Figure A.8.

## 3.2. Model Architectures

Recurrent neural networks (RNNs) and their variants, particularly long short-term memory (LSTM) networks, are widely used for multivariate time series forecasting, especially for multistep problems [36]. LSTMs are favoured due to their ability to capture dependencies across longer sequences by mitigating the vanishing gradient problem that afects traditional RNNs.

To develop an efective forecasting model, we explore several architectures, beginning with a baseline LSTM model and progressing to more intricate encoder-decoder (seq2seq) structures. Two RNNbased networks, LSTMs in this case, make up the encoder/decoder, or sequence to sequence (Seq2Seq) architecture[37]. The encoder first processes and compresses the input sequence into a fixed-sized context vector, which the decoder then uses to iteratively produce each step of the output sequence. However, the fixed context vector can limit the model’s ability to retain detailed information over long input sequences.

To enhance the model’s predictive accuracy and focus on relevant input segments during decoding, we integrate Luong attention [38] into the Seq2Seq architecture. Luong attention allows the model to dynamically adjust the importance of each input time step in generating each output prediction. By actively comparing the current decoder hidden state, $\mathbf { h } _ { t } .$ , with each encoder output, $\mathbf { h } _ { s } ,$ Luong attention assigns weights (alignment scores) that emphasise the most relevant portions of the input sequence. This approach enables a focused alignment on the recent 48-hour context, which is particularly useful in dayahead forecasting where immediate past conditions are strong predictors of near-future outcomes.

All three of Luong’s alignment scoring functions are considered in our model [38]:

$$
\mathrm { s c o r e } ( \mathbf { h } _ { t } , \mathbf { h } _ { s } ) = \left\{ \begin{array} { l l } { \mathbf { h } _ { t } ^ { \top } \mathbf { h } _ { s } } & { \mathrm { d o t } } \\ { \mathbf { h } _ { t } ^ { \top } \mathbf { W } _ { a } \mathbf { h } _ { s } } & { \mathrm { g e n e r a l } } \\ { \mathbf { v } _ { a } ^ { \top } \operatorname { t a n h } \left( \mathbf { W } _ { a } [ \mathbf { h } _ { t } ; \mathbf { h } _ { s } ] \right) } & { \mathrm { c o n c a t } } \end{array} \right.\tag{1}
$$

Each method provides diferent levels of flexibility in learning relationships within the data:

• Dot: Computationally eficient, directly computes the dot product between $\mathbf { h } _ { t }$ and $\mathbf { h } _ { s }$

• General: Uses a learnable weight matrix $\mathbf { W } _ { a }$ to scale $\mathbf { h } _ { s } .$ , capturing adaptable contributions across input steps.

• Concat: Concatenates $\mathbf { h } _ { t }$ with $\mathbf { h } _ { s }$ and applies a non-linear transformation, ofering higher flexibility but at greater computational cost.

While self-attention and Transformer-based crossattention mechanisms excel at capturing long-range dependencies, our temporal encoding of the day and month features should serve as a proxy for these longer-term dependencies. Luong attention on the other hand, focuses on short-term dependencies within recent observations, which are particularly relevant for day-ahead predictions and is less computationally expensive. Further, using explicit alignment scores enhance interpretability and enabling stakeholders to trace forecasted outputs back to specific recent observations.

Finally, an embedding layer processes the site ID feature to capture location-specific factors, such as microclimates or terrain efects, which can vary between sites and influence wind

Finally, an embedding layer processes the site ID feature to capture location-specific factors, such as microclimates or terrain efects, which can vary between sites and influence wind and power patterns.

## 3.3. Optimisation

To achieve more accurate and logically sound explanations and ensure further reliability in generated forecasts, we optimise both hyperparameters and the choice of architecture. Given the dimensionality of the search space and the computational expense of training, the models we employ Bayesian Optimisation (BO) to optimise the choice of architecture and remaining parameters. Additionally, we use early stopping and a learning rate scheduler to ensure that models do not overfit and reduce training time. Fixed hyperparameters and ones that were found to work well overall can be seen in Table A.3.

Generally, the encoder-decoder architecture performed the best with attention. The general scoring function within the attention mechanism performed the best, compared to concat and dot. A detailed optimisation summary can be found in Table A.1 and the final architecture in Figure A.2.

## 3.4. Explainability by Feature Attribution

Feature attribution has become a core post-hoc interpretability method across numerous fields, including energy[22], [24], driven by the increasing complexity of machine learning models. Feature attribution attempts to address the black-box challenge by quantifying the importance of each input feature for a given prediction, thus helping domain experts to gain insights into model decisions [39].

SHAP was originally introduced as a unified framework that combines multiple approaches to additive feature importance, such as LIME, DeepLIFT, and Layer-wise Relevance Propagation (LRP), under a single theoretical foundation based on cooperative game theory[12]. In SHAP, each feature is treated as a player in a game. Players work together to make the model prediction which is viewed as the payout and must be fairly distributed. The marginal contribution of each feature to the prediction is calculated by averaging its impact across all possible combinations of features, or coalitions. This yields the Shapley value for each feature i:

$$
\begin{array} { l } { \displaystyle \phi _ { i , S H A P } = \sum _ { S \subseteq F \setminus \{ i \} } \frac { | S | ! ( | F | - | S | - 1 ) ! } { | F | ! } } \\ { \displaystyle \left[ f _ { S \cup \{ i \} } ( x _ { S \cup \{ i \} } ) - f _ { S } ( x _ { S } ) \right] } \end{array}\tag{2}
$$

where $F$ is the set of all features and S corresponds to a subset of features.

SHAP ensures several desirable properties that make it well-suited for model interpretability:

• Local Accuracy: The SHAP value sum approximates the model’s output at a given instance.

• Missingness: Features not in the original input are assigned zero importance.

• Consistency: If a feature’s contribution increases in the model, its Shapley value does not decrease.

Given these properties, Lundberg and Lee prove that methods not based on shapely violate local accuracy and/or consistency.

GradientSHAP, is a more recent, gradient-based extension of SHAP specifically for diferentiable models and is a synthesis of ideas from Integrated Gradients (IG)[40] and SmoothGrad. IG is a path attribution method corresponding to Aumann-Shapley values[41]; a cost-sharing method that extends Shapley values to infinite-player games. GradientSHAP follows the same approach as IG but is enhanced by sampling across multiple baselines and averaging gradients along paths between each baseline and the input. Many traditional SHAP variants are based on the observational conditional expectation, which leads to importance attributed to irrelevant features, and misleading explanations when features are correlated[16], [42]. By integrating over a background set, GradientSHAP follows a more interventional approach and produces explanations that are “true to the model” rather than “true to the data”[43].

In contrast to SHAP, GradientSHAP extends the original IG calculation, calculating Shapley values by taking the expectation of gradients over interpolated points from each baseline to the target input:

$$
\begin{array} { r l r } {  { \phi _ { i , E G } ( f , x ; D ) = \mathbb { E } _ { x ^ { \prime } \sim D , \alpha \sim U ( 0 , 1 ) } } } \\ & { } & { [ ( x _ { i } - x _ { i } ^ { \prime } ) \times \frac { \partial f ( x ^ { \prime } + \alpha \times ( x - x ^ { \prime } ) ) } { \partial x _ { i } } ] } \end{array}\tag{3}
$$

where $x _ { \prime }$ is a sample from the original data distribution D, and x′ is a reference sample. Similarly, α is a scaling factor, and $\frac { \delta \check { F } } { \delta x _ { i } }$ is the gradient with respect to feature i.

Not only does EG satisfy all axioms proposed by Lundberg and Lee (which are similar to the properties mentioned above) for SHAP, several newer axioms are also satisfied [40]:

• Sensitivity(a): For every input and baseline that difer in one feature but have diferent predictions then the difering feature should be given a non-zero attribution.

• Sensitivity(b) - Dummy: This axiom is satisfied by any interventional. It requires that a feature will get zero attribution if it does not afect the model’s output.

• Implementation Invariance: The attribution for any two, functionally equivalent networks should always be identical.

Adhering to these axioms is ideal for complex multistep and multivariate time series problems, where many feature interactions and dependencies can exist. By leveraging EG, GradientSHAP provides a computationally eficient[33] and theoretically grounded approach to day-ahead forecasting; enabling a thorough exploration of lags, features, prediction steps and their interactions.

## 3.4.1. Utilising GradientSHAP

To ensure a balanced trade-of between computational eficiency and explanation fidelity, we followed the guidance[33] and set the background sample size to 5000. As indicated in [44], the number of background samples can significantly impact GradientSHAP’s stability, where too few samples may introduce noise and too many can incur unnecessary computational costs. For testing and explanation generation, we used 300 samples, calculating SHAP values for each feature across all 48 historical time steps and 24 future predictions.

A notable challenge encountered during implementation was the incompatibility between GradientSHAP and embeddings, which prevented SHAP from attributing any contribution to the site ID feature due to its integer encoding. This issue, documented in the SHAP package, arises because integerencoded embeddings lack gradients, resulting in no attribution to these features in the output. To address this, we implemented forward and backward hooks to sum the gradients of the site ID embedding vector, ensuring proper gradient flow and enabling accurate feature attribution. This solution preserved the integrity of the explanations and allowed GradientSHAP to provide a complete view of feature contributions, crucial for reliable decision-making in wind power forecasting.

## 3.5. Experimental Setup

Model training, inference and explanation generation have been performed on a system with a Nvidia GeForce GTX 1080 Ti, Intel Core i7-14700k and 48GB of RAM. PyTorch[45] has been used for model development, training and inference using the GPU.

## 4. Results & Discussion

This section provides a comprehensive breakdown of the model’s feature importance through various

plots, forming global and local feature explanations. We start with summary plots and progressively uncover interaction efects and temporal dynamics.

## 4.1. Summary Plots

The summary plots in Figures 1a, 1b show the SHAP values across both lags and prediction steps. Here, the absolute SHAP value is used as a summary to show the overall impact. Across all visualisations, the most dominant features are site ID and power. However, it is clear from figure 1a that most of this attribution for power comes from the most recent time step t-0 and is only predictive for the first 7 prediction steps. Beyond this, site ID and therefore site-specific factors captured by site ID (and minorly month) take over, especially in the mid-range prediction steps. Additionally, temperature has negligible importance while the remaining features have a small amount with the most importance attributed to t−0.

In contrast, the heatmaps in Figures 2 and A.7 provide instance-specific insights and local explanations with directional SHAP values. For the instance recorded, there is a peak in power production observed. During this peak, site ID increases in importance, while power decreases. Hour shows a slight peak in importance, while month drops to zero during the peak. Additionally, positive SHAP values for site ID and hour indicate their role in predicting higher outputs, while other features, including power, predominantly predict lower outputs. While the latter may seem counterintuitive, both observations suggest that more stable periods rely on power, wind speed and monthly seasonality, while diurnal patterns and site-specific factors inform predictions during periods of instability.

Additionally, the layered violin SHAP plot in Figure 3 shows the top 10 most important features in order of importance. These plots allow analysis of the most predictive, individual features, their directional impact, feature values, and distributions. The results align with earlier findings, emphasising that power t-0 is the most important feature, with high feature values leading to positive predictions and low values leading to negative predictions. Hour is important at both t − 0 and t − 1, with its distribution seeming to align with diurnal patterns in that early hours reduce prediction. Finally, wind speed at t-0 and wind direction at t-0 are also significant, with wind speed mirroring the relationship of power to its feature values.

![](images/2bf0e459143bb366f650d916cb669a7de77027cd39defc2adcf548ef0c5d3318.jpg)  
(a) Across Lags

![](images/0e4cc5915bda4f06549e5749a7c287eebfd890fc76eba97610a110214c31d84f.jpg)  
(b) Across Prediction Steps  
Figure 1: Heatmaps for feature importance.

## 4.2. Analysing Interactions

In this section, we investigate the dominant role of site ID to determine if its influence can be attributed to site-level variations in other features.

To explore potential interactions we use SHAP dependency plots. These plots depict the SHAP values for each site ID, coloured by various feature values. However, since site ID is a categorical feature, a beeswarm-style plot is more appropriate in this case. This allows for a clearer visualisation of the distribution of SHAP values across all instances, efectively showing the variance and spread of SHAP values at each site ID. Furthermore, because of the negligible importance of earlier time steps, we use the 10 most recent time steps to capture interactions that are most likely to afect prediction outcomes. Moreover, to account for the additional dimension of prediction time steps, the SHAP values are summed across all prediction horizons. This summation ofers an overall view of the feature’s impact on the model output.

The dependency plots in Figure 4 reveal a significant variation in SHAP values between sites. Sites 0 (Washington) and 10 (Texas) consistently exhibit higher SHAP values, indicating stronger feature interactions at these locations. In contrast, sites 4 (New York), 5, and 6 (Iowa) show lower SHAP values, suggesting reduced variability or weaker environmental interactions.

No obvious relationship is seen in Figures 4a and 4c, however, the higher values of power are not present in all sites, specifically, sites 0, 1 and 3. Despite this, they have some of the highest SHAP values out of all the sites, reiterating what was found in the layered violin plot and suggesting confounding factors. As for the wind speed plot, high wind speeds appear to be exceptionally rare and only really present in sites 2, 9, and 10, suggesting that these sites have higher wind variability and overall maximum wind speeds, but this is not directly influencing the model.

Analysis of temperature and wind direction (Figure 4b and 4d) reveals further variation between sites. Generally, higher temperatures seem to occur more centrally, while lower values appear at the extremes in SHAP value. This can mostly be explained by high temperatures generally being associated with atmospheric stability, leading to reduced wind speeds and therefore reducing the influence of temperature as it corresponds with lower variability and weaker winds. This also explains why site ID becomes less important in sites with a majority of high temperatures, as power generation is more predictable by general trends. Additionally, sites 3, 4 5 and 7 have a significantly higher proportion of low values of wind direction and higher variability. Interestingly this also aligns with the temperature values observed previously. While there is no clear link overall, the SHAP extremes have matched feature value magnitude for most sites. This suggests that wind direction can either enhance or reduce Site ID’s importance, depending on local wind patterns and turbine alignment or yaw control[46].

![](images/91e5329c6cc240935bdb96071d2e0ebb1f5aa2ca41a39b3a6adf66df6f9c5fe0.jpg)  
Figure 2: Temporal heatmaps for a single samples features across prediction steps.

![](images/2abbeaba8f3e97f02fdeaed603c7dc419dc524aa1583cb13b1e79505d112d5ee.jpg)  
Figure 3: Layered Violin Plot

Although some useful interpretations can be made from these plots, it can become unclear when comparing the feature values on the same scale. It would be useful to colour the sites by feature value individually to uncover more localised efects that may be masked.

## 4.3. Verifying Relationships

The SHAP Joint plots (Figure 5) provide a detailed overview of SHAP dependency plots alongside KDE plots discretised into feature value ranges, for both the x-axis and y-axis, enabling the identification of high- and low-density regions in the feature space. This allows for visual verification of known relationships in the model while highlighting the impact of secondary features to investigate interactions further. Each row in the figure corresponds to a diferent primary feature, with each plot coloured by a secondary feature.

The first row, coloured by power, demonstrates a largely expected relationship between wind speed and SHAP values (first column): wind speed is strongly and positively correlated with the SHAP value and therefore power output. However, two clear clusters form, one which remains centred around a SHAP value of 0 (1a) and one which follows this positive relationship (1b). The wind direction plot (rightmost column) reveals a potential explanation, forming another two clusters, one centred around zero (2a). In contrast to the wind speed and power plots, there seems to be another cluster following a quadratic relationship with a minimum around 200◦ (2b). The latter indicates that wind direction may be overriding the importance of wind speed and power in these instances. The cluster 1a in the wind speed and power plots may correspond to points that likely reflect suboptimal wind direction for turbine alignment, suppressing the importance of other features. However, given the presence of cluster 2b, there are likely other confounders at play, such as wind shear. Temperature (second column) shows a negative correlation with the SHAP value. As previously mentioned this is expected, yet, this is invalidated when colouring wind speed by temperature (third row, second column) as there is no clear correlation. This may be accounted for by the site specific factors or ineficiencies at higher temperatures.

![](images/8c54212413ba2de23efa6ed55e1e59767db951633003a89593fbef8e5133e17c.jpg)

(a) Power  
![](images/c3e5bbb0bb97f068d131d7210380693fdfaf15abe98c06e38b4960e5ab57bdf3.jpg)

(b) Temperature  
![](images/58c926d03a7aee5b7f8de2b57b111e261626985f23bf4b1c09d5cc34773d0589.jpg)

(c) Wind Speed  
![](images/23bcefb73306b524f20f429750a87a2f18e987f2461517a3186f5c06d7fa87d4.jpg)  
(d) Wind Direction  
Figure 4: SHAP Dependency plot for site ID coloured by all features, using t-10 to t-0. Values have been aggregated over forecast horizons.

![](images/e83999686f016c69b6e4a729b3f6c29d8c49380ba1d15fb356f17008f1c779a9.jpg)

![](images/4a65fa4ea85031c4d972162883b45858669bf6c2914ef69692216a02ad4802d2.jpg)

![](images/60b44aaf3fd7a2332ec50dcf46e4b9e66571c08ae3d8b0fce5378607d6f839bd.jpg)

![](images/fe4366944be8a3a57cfc125237fe2afcbb42f496375da666fa29c2922470761d.jpg)

![](images/b4d00dcb415e8b8963d37a8cec14f8ae04ad5464f7aca61ba5f71be94b7a44d7.jpg)

![](images/e061bf8f2b63c9f5376079024625b5045b0af6991823d1c12717c6719104509f.jpg)

![](images/fe4f97773a9fdf2c0ff47b20381c6b263f264c5a8fbccc7722e11c247f7399d5.jpg)

![](images/0aba632f59c0ad1cda40057721581a2496b2d1fc836d7954a16545fcb93f7a1e.jpg)

![](images/9ea81e93c15c0766c838f057b897a608bdeaa41741099ee495a75be10092ef58.jpg)

![](images/487ef6851cd638735ee4e9254b44d75a061dfc5d9913dc2769b229f14244779f.jpg)

![](images/87600eee2cc8145aa019c4123c2864b403b8398b3697ff9091f4a0983112d816.jpg)

![](images/ac4d46c558b1499184356bb18d33b90e0655c28e32872f31f797d12468b5b9c8.jpg)  
Figure 5: SHAP dependency plots for all features from t − 10 to t − 0, with observational cluster annotations. Clusters 1a and 1b (wind speed) and 2a and 2b (wind direction) are identified in the top row only (coloured by power).

The second row ofers further insight now focusing on wind speed. Generally, points with low wind speeds adhere to the expected pattern (first column): power is strongly and positively correlated with the output power and therefore SHAP value. Furthermore, the same type of clustering appears as in the previous wind speed plot when it comes to SHAP values $( \mathit { 1 a , 1 b } )$ , albeit with more spread at the power extremes for cluster 1a. Most of the wind speed values in the wind direction plot (rightmost column) are evenly distributed, however, higher values seem to cluster around 200◦, like the minimum of cluster 2b. Despite this high wind speed, the ones within this cluster have a low SHAP value. It is unclear from the KDE plot for the y-axis as to which cluster these high wind speed values correspond to, but it is clear that the relationship between wind speed and output breaks down in cluster 2b, with the lowest SHAP values having the highest wind speeds and the samples with the highest SHAP values have the lowest wind speeds. Therefore, it may be the case that cluster 2b corresponds with cluster 1a.

The power plots coloured by temperature (third row) reveal further context for this. Within the wind direction plot (rightmost column) the highest values are temperature are clustered around the minimum of 2b, 200◦, with lower values clustered at larger angles and the KDE shows a negative relationship between the two features. Taking this together, if we assume clusters 2b and 1a are one of the same, given that 1b does not follow the assumed relationship regarding wind speed/power and SHAP value, then temperature may be degrading the attribution of wind speed and power. This can further be seen in the last plot coloured by wind speed (third row, third column) within the distribution of wind direction values. This can generally be explained by higher temperatures being associated with atmospheric stability where warmer air creates pressure gradients that drive winds. Conversely, at lower temperature, there may be greater temperature gradients, which create strong winds and variability making wind direction more influential. Therefore, there could be a pressure gradient causing a suboptimal change in wind direction and reducing the importance of other features in cluster 1a which could also explain the correlation found between SHAP value and temperature..

Given its influence on pressure gradients and the interactions between temperature and wind direction, it may have also been beneficial to include atmospheric pressure data for a more complete explanation. Additionally, further insights could have been gained by confirming the clusters and identifying what sites they belong to.

## 4.4. Temporal Analysis

The temporal SHAP analysis (Figure 6) examines how feature importance changes over time, specifically focusing on seasonal and diurnal patterns. We aggregated SHAP values over the month and day features to show how feature importance evolves with changing meteorological conditions, similar to [19]. Additionally, we choose to use the final feature for simplicity and normalise the SHAP values across all features for comparability.

In the month plot (Figure 6b), we observe that wind speed and power appear to experience very similar changes across time. Both exhibit a seasonal dip between March and June, which coincides with known seasonal trends. Both Wind speed and temperature tend to also be lower during the warmer months leading to an expected reduction in SHAP value. Given the prior relationships found earlier, temperature also dips in these months, potentially leading to lower SHAP values. Both features also peak in October and to a lesser extent in March.

![](images/e63d49f047e25b0744dfea67a45a430c80857e39bb6c9f437b2288b680619504.jpg)  
Figure 6: Temporal aggregation of SHAP values at t-0 by month and hour.

Both power and wind direction, show a peak in July, with the latter a sharply reaching the maximum SHAP value. This can be explained by the prior discussion of temperature and atmospheric stability, whereby more organised and consistent patterns can be found in warmer months. In December, both temperature and wind direction see significant peaks, with the former reaching the maximum. In this month, power and wind speed are moderate, and site ID is at the minimum. temperature dominates. This may suggest that wind direction becomes more predictable or aligned with turbine placement during colder months, while temperature variability has a larger efect on power production.

The final peak within site ID can be seen in February, where site ID and temperature are moderate but power and wind speed are of low importance. This could indicate site-specific efects, such as geographical location or regional climate conditions that drive diferences in turbine performance, even when other meteorological factors are not highly significant. Generally, the importance of site ID seems to be reduced when the importance of other features is high and increases when the importance of other features is low, suggesting that during these periods, site-specific factors like terrain or localised wind patterns become more influential. These periods likely correspond to periods of stability and instability and likely reflect the prior findings on local explanations in Section 4.1 are likely also global patterns.

Temperature shows clear seasonality, peaking in the winter months and gradually decreasing to a minimum in July. This aligns with colder temperatures introducing more variation in turbine operation due to factors like eficiency loss during hotter periods, or more stable wind patterns, demonstrating the model’s sensitivity to temperature efects on turbine performance. Overall, all features show three distinct peaks and troughs, aligning with the seasons or responding to other feature changes afected by the season.

The hourly SHAP analysis (Figure 6a) ofers valuable insights into how feature importance varies throughout the day. SHAP values for wind speed and power exhibit notable increases and decreases around midday and late evening, mirroring typical diurnal wind patterns. In addition, most features have a peak around 6 AM and again at 9 PM, with temperature being the most notable. These times correspond closely with sunset and sunrise, periods during which temperature gradients caused by solar heating and cooling can lead to shifts in atmospheric stability. The only exception is that wind speed dips during the 9 PM peak which could reflect calmer wind conditions due to less thermal gradients at night.

Later in the day, around 2 PM, temperature drops to its minimum and is accompanied by a decrease in wind speed and power. Interestingly, during this same period, site ID experiences a slight increase. This pattern may indicate a phase of temperature stabilization, which could afect conditions in the hours that follow.

Additionally, temperature seems to dominate the earlier hours (up to 2 PM), except for two peaks of site ID. These observations can again be explained by typical diurnal wind patterns. During the day, solar heating creates turbulence, increasing surface winds, leading to more reliance on temperature, while at night, the absence of solar heating leads to a more stable atmosphere with reduced wind speeds. After sunset, cooling reintroduces turbulence, allowing wind speeds to increase again [47], [48].

Site ID tends to have a higher importance in the early hours, dropping around midday before gradually increasing, similarly to temperature and wind speed, but shows three peaks at 6 AM and 8 AM, where other features (except temperature) are substantially lower. This suggests that early morning conditions may depend more on site-specific features, possibly due to terrain efects or microclimates that influence the wind, whereas later in the day, atmospheric stability and general weather patterns take over.

Power and wind speed seem to peak during the later hours. Specifically, power has one main peak at midnight, where wind speed and wind direction also peak. The others occur at 5 AM and 6 PM, both aligning with wind speed. Other than reflecting diurnal variation, these peaks could be caused by increased turbine eficiency when wind speeds are more stable and consistent, either at night or during the early morning and evening transition periods[49].

Generally, the hour plot seems highly sporadic, showing significant changes in importance and power production from hour to hour. Hence, it may have been more informative to sample hourly data from WIND rather than taking the mean. Furthermore, normalising the SHAP values may have been useful to compare relative importance, but may have minimised per-feature trends.

## 4.5. Summary

We conducted a detailed SHAP analysis to interpret the decision-making process of our day-ahead wind power forecasting model, yielding the following key findings:

1. Site ID’s Dominance: Site ID consistently emerges as the most important feature across time steps and prediction horizons. This highlights the significant role of site-specific factors, such as geographical and environmental conditions, which interact with other variables to shape predictions.

2. Importance of Time Steps: t-0 consistently dominates feature importance across most variables, with earlier lags contributing negligible influence except site ID.

3. Temporal Variability: Seasonal and diurnal patterns reflect shifts in feature importance. Power t-0 dominates short-term predictions, while Site ID becomes increasingly influential in mid-range predictions. Temperature peaks in importance during colder months, correlating with operational challenges, while diurnal patterns show a relative increase in reliance on sitespecific factors during early morning hours.

4. Feature Interactions: The analysis reveals complex relationships between features. Wind direction often overrides wind speed when suboptimal alignments or local patterns are present. Clusters in SHAP joint plots confirm that features like temperature and wind direction interact non-linearly, driven by local conditions and turbine-specific factors.

5. Verified Relationships: Known relationships, such as the positive correlation between wind speed and power, are validated. However, clusters of low SHAP values at high wind speeds and suboptimal wind directions suggest suppressed feature importance in certain conditions, likely due to pressure gradients and turbine misalignment.

6. Environmental and Localized Efects: Temperature impacts the model in distinct ways: higher temperatures, associated with atmospheric stability, reduce variability and diminish feature importance, while lower temperatures increase variability in wind direction and enhance its predictive role. Site-specific conditions, such as terrain and local wind patterns, also influence model outputs, particularly during transitional periods like early morning and late evening.

Overall, our SHAP analysis largely aligns the model’s decision-making process with real-world dynamics while further illuminating the inner workings of the model. By leveraging global and local visualisations, we validated known relationships, uncovered temporal patterns, and identified interactions that inform predictions.

## 5. Conclusion

This paper demonstrates the utility of SHAP methods, namely GradientSHAP as a tool for enhancing interpretability in multivariate, multistep time series forecasting. We focus on day-ahead wind power forecasting, a vital task for efective grid management and the global transition toward net-zero energy systems, given the inherent unpredictability of wind power.

Our application of GradientSHAP revealed critical insights into the decision-making process of our forecasting model. Site ID, representing site-specific factors such as terrain, environmental conditions, and turbine configurations, emerged as the most important global feature. Its sustained importance across all time steps, particularly in mid-range prediction horizons, highlights its role in capturing contextual and long-term variability, beyond the immediate influence of other features. These findings strongly suggest that Site ID should be a core component of forecasting models to capture site-specific variability effectively. Power t-0, the most recent historical observation, dominated short-term predictions, reflecting the model’s reliance on immediate past data for accurate forecasting. Temporal SHAP heatmaps revealed dynamic shifts in feature importance, with diurnal and seasonal patterns aligning closely with real-world dynamics. These patterns have emphasised the complementary roles of site-specific and meteorological factors. The analysis also highlighted complex interactions between site ID and other features, particularly wind direction and temperature. These findings uncover subtle dependencies, such as the influence of atmospheric stability and pressure gradients, on feature importance.

In addition to uncovering novel insights, the analysis validated known relationships, and interactions, strengthening confidence in the robustness of the model and its alignment with real-world dynamics.

Despite its strengths, GradientSHAP’s interpretability remains context-dependent, requiring substantial domain expertise to translate SHAP values into valuable insights. Static visualisations provided valuable explanatory power but limited actionable dynamic insights. In addition, out-of-manifold samples and fixed sampling methods may have introduced biases. Integrating on-manifold sampling[50] or counterfactual approaches such as CF-SHAP[51] could improve the explanations.

Future work could improve GradientSHAP’s utility by integrating additional features such as atmospheric pressure to enrich explanations or spatial data, which could further capture site-specific factors. Additionally developing interactive explanation tools that allow dynamic adjustments of lags and prediction steps could provide significantly more explantory power. Furthermore, while our study focused on SHAP values, incorporating analysis of the model’s attention weights could complement these insights, allowing richer comparisons and a more comprehensive understanding of the model’s behaviour. Concept discovery, grouping consistently important features into domain-informed clusters, could provide more intuitive and less demanding explanations.

By advancing these methodologies, we can bridge the gap between model complexity and interpretability, paving the way for a wider adoption of AI systems within the energy sector. Ultimately, our findings contribute to a more reliable and transparent approach to energy forecasting, supporting the global transition to sustainable and resilient energy systems.

Appendix A.1. Supplementary Results

![](images/74af7c2143842482a03ab7e546b8ecb944f9171703157c24518cfb6949c7458d.jpg)  
Figure A.7: Temporal heatmap for a single sample’s features across lags.

## Appendix A.2. Model And Optimisation

<table><tr><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Range</td><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Best Found</td></tr><tr><td rowspan=1 colspan=1>Embeddings</td><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=1>1 - 20</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>9</td></tr><tr><td rowspan=4 colspan=1>Encoder</td><td rowspan=1 colspan=1>Dropout rate</td><td rowspan=1 colspan=1>0 - 0.75</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=1 colspan=1>Post layer dropout rate</td><td rowspan=1 colspan=1>0 - 0.75</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=1 colspan=1>Num. layers</td><td rowspan=1 colspan=1>1 - 4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Hidden size</td><td rowspan=1 colspan=1>4 - 64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=4 colspan=1>Decoder</td><td rowspan=1 colspan=1>Dropout rate</td><td rowspan=1 colspan=1>0 - 0.75</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=1>Num. layers</td><td rowspan=1 colspan=1>1, 4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Hidden size</td><td rowspan=1 colspan=1>4 - 64</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>Attention type</td><td rowspan=1 colspan=1>None, general, concat, dot</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>General</td></tr><tr><td rowspan=2 colspan=1>FC</td><td rowspan=1 colspan=1>Dropout rate</td><td rowspan=1 colspan=1>0 - 0.75</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Num. layers</td><td rowspan=1 colspan=1>1 - 3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3</td></tr></table>

Table A.1: Summary of final optimised model & training hyperparameters

<table><tr><td colspan="1" rowspan="1">Layer (Type)</td><td colspan="1" rowspan="1">Output Shape</td><td colspan="1" rowspan="1">Param #</td></tr><tr><td colspan="1" rowspan="1">EncDecAttention</td><td colspan="1" rowspan="1">[1, 24]</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Embedding (1-1)</td><td colspan="1" rowspan="1">[1, 48, 9]</td><td colspan="1" rowspan="1">99</td></tr><tr><td colspan="1" rowspan="1">Encoder (1-2)LSTM (2-1)</td><td colspan="1" rowspan="1">[1, 48, 30][1, 48, 30]</td><td colspan="1" rowspan="1">13,440</td></tr><tr><td colspan="1" rowspan="1">Dropout (1-3)</td><td colspan="1" rowspan="1">[1, 48, 30]</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Decoder (1-4)LSTM (2-2)</td><td colspan="1" rowspan="1">[1, 1, 30][1, 1, 30]</td><td colspan="1" rowspan="1">29,760</td></tr><tr><td colspan="1" rowspan="1">Attn (1-5)Linear (2-3)</td><td colspan="1" rowspan="1">[48, 1, 1][1, 48, 30]</td><td colspan="1" rowspan="1">930</td></tr><tr><td colspan="1" rowspan="1">Linear (1-6)</td><td colspan="1" rowspan="1">[1, 30]</td><td colspan="1" rowspan="1">1,830</td></tr><tr><td colspan="3" rowspan="1">Total params: 46,090Trainable params: 46,090Non-trainable params: 0Total mult-adds (M): 1.43Input size (MB): 0.00Forward/backward pass size (MB): 0.30Params size (MB): 0.18Estimated Total Size (MB): 0.49</td></tr><tr><td colspan="1" rowspan="1">Category</td><td colspan="1" rowspan="1">Parameter</td><td colspan="1" rowspan="1">Value</td></tr><tr><td colspan="1" rowspan="2">Early Stopping</td><td colspan="1" rowspan="1">Threshold</td><td colspan="1" rowspan="1">0.0001</td></tr><tr><td colspan="1" rowspan="1">Patience</td><td colspan="1" rowspan="1">5</td></tr><tr><td colspan="1" rowspan="6">Learning Rate</td><td colspan="1" rowspan="1">Initial LR</td><td colspan="1" rowspan="1">0.0001</td></tr><tr><td colspan="1" rowspan="1">Scheduler type</td><td colspan="1" rowspan="1">Reduce on plateau</td></tr><tr><td colspan="1" rowspan="1">Factor</td><td colspan="1" rowspan="1">0.1</td></tr><tr><td colspan="1" rowspan="1">Patience</td><td colspan="1" rowspan="1">10</td></tr><tr><td colspan="1" rowspan="1">Threshold</td><td colspan="1" rowspan="1">0.001</td></tr><tr><td colspan="1" rowspan="1">Min. LR</td><td colspan="1" rowspan="1"> $\overline { { 1 \times 1 0 ^ { - 8 } } }$ </td></tr><tr><td colspan="1" rowspan="1">Misc</td><td colspan="1" rowspan="1">Batch size</td><td colspan="1" rowspan="1">32</td></tr></table>

Table A.2: Summary of final forecasting model architecture.

Table A.3: Summary of fixed hyperparameters. LR = learning rate.

Appendix A.3. Preprocessing  
![](images/28f44737483d6ac6854e9c078b854452cd0494e7bb0b37299f05a3f1946fd548.jpg)  
Figure A.8: Illustration of three non-overlapping windows for site 0.

## References

[1] S. Haron, “ADViCE: AI for Decarbonisation Challenges,” Dec. 2023. Accessed: Jan. 16, 2024. [Online]. Available: https : / / www . turing . ac . uk / research / research - projects/advice

[2] F. Dell’Anna, “Green jobs and energy eficiency as strategies for economic growth and the reduction of environmental impacts,” Energy Policy, vol. 149, p. 112 031, 2021, issn: 0301-4215. doi: 10.1016/j.enpol.2020.112031 [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0301421520307424

[3] W. Przychodzen and J. Przychodzen, “Determinants of renewable energy production in transition economies: A panel data approach,” Energy, vol. 191, p. 116 583, 2020, issn: 0360- 5442. doi: 10 . 1016 / j . energy . 2019 . 116583 [Online]. Available: https : / / www . sciencedirect.com/science/article/pii/ S0360544219322789

[4] H. Zhang, Y. Hu, and J. He, “Wind tunnel experiment of multi-mode ARC sail device,” Polish Maritime Research, vol. 28, pp. 20–29, Dec. 2021. doi: 10.2478/pomr-2021-0046

[5] M. Sivaramakrishnaiah et al., “Numerical and experimental investigations on a bladeless turbine: Tesla’s cohesion-type innovation,” International Journal of Renewable Energy Development, vol. 13, no. 1, pp. 110–121, 2024.

[6] V. N. Nguyen et al., “Potential of explainable artificial intelligence in advancing renewable energy: Challenges and prospects,” Energy & Fuels, vol. 38, no. 3, pp. 1692–1712, 2024. doi: 10.1021/acs.energyfuels.3c04343 [Online]. Available: https://doi.org/10.1021/acs. energyfuels.3c04343

[7] R. Machlev et al., “Explainable Artificial Intelligence (XAI) techniques for energy and power systems: Review, challenges and opportunities,” Energy and AI, vol. 9, p. 100 169, 2022, issn: 2666-5468. doi: 10.1016/j.egyai.2022. 100169 [Online]. Available: https : / / www .

sciencedirect.com/science/article/pii/ S2666546822000246

[8] B. Ersöz, Ş. Sağıroğlu, and H. İ. Bülbül, “A short review on explainable artificial intelligence in renewable energy and resources,” in 2022 11th International Conference on Renewable Energy Research and Application (ICR-ERA), IEEE, 2022, pp. 247–252.

[9] H. Li, “Short-term wind power prediction via spatial temporal analysis and deep residual networks,” Frontiers in Energy Research, vol. 10, p. 920 407, May 2022. doi: 10 . 3389 / fenrg . 2022.920407

[10] W.-C. Tsai, C.-M. Hong, C.-S. Tu, W.-M. Lin, and C.-H. Chen, “A review of modern wind power generation forecasting technologies,” Sustainability, vol. 15, no. 14, 2023, issn: 2071-1050. doi: 10 . 3390 / su151410757 [Online]. Available: https : / / www . mdpi . com / 2071-1050/15/14/10757

[11] R. Alsaigh, R. Mehmood, and I. Katib, “AI explainability and governance in smart energy systems: A review,” Frontiers in Energy Research, vol. 11, 2023, issn: 2296-598X. doi: 10 . 3389 / fenrg . 2023 . 1071291 [Online]. Available: https://www.frontiersin.org/ journals/energy- research/articles/10. 3389/fenrg.2023.1071291

[12] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17, Red Hook, NY, USA: Curran Associates Inc., 2017, pp. 4768–4777, isbn: 978-1-5108-6096-4. [Online]. Available: https: / / api . semanticscholar . org / CorpusID : 21889700

[13] N. G. E. S. Operator, Probabilistic machine learning solution for dynamic reserve setting, Energy Networks Association, 2022. [Online]. Available: https : / / smarter . energynetworks . org / projects / nia2 \_ ngeso003/

[14] C. Xu, Z. Liao, C. Li, X. Zhou, and R. Xie, “Review on interpretable machine learning in smart grid,” Energies, vol. 15, p. 4427, Jun. 2022. doi: 10.3390/en15124427

[15] K. Aas, M. Jullum, and A. Løland, “Explaining individual predictions when features are dependent: More accurate approximations to Shapley values,” Artificial Intelligence, vol. 298, p. 103 502, 2021, issn: 0004-3702. doi: 10 . 1016/j.artint.2021.103502 [Online]. Available: https : / / www . sciencedirect . com / science/article/pii/S0004370221000539

[16] D. Janzing, L. Minorics, and P. Blöbaum, “Feature relevance quantification in explainable AI: A causal problem,” in International Conference on Artificial Intelligence and Statistics, PMLR, 2020, pp. 2907–2916.

[17] G. Erion, J. D. Janizek, P. Sturmfels, S. Lundberg, and S.-I. Lee, Improving performance of deep learning models with axiomatic attribution priors and expected gradients, 2020. [Online]. Available: https : / / arxiv . org / abs / 1906 . 10670

[18] L. Gugliermetti, F. Cumo, and S. Agostinelli, “A future direction of machine learning for building energy management: Interpretable models,” Energies, vol. 17, no. 3, 2024, issn: 1996-1073. doi: 10.3390/en17030700 [Online]. Available: https : / / www . mdpi . com / 1996 - 1073/17/3/700

[19] J. Białek, W. Bujalski, K. Wojdan, M. Guzek, and T. Kurek, “Dataset level explanation of heat demand forecasting ANN with SHAP,” Energy, vol. 261, p. 125 075, 2022, issn: 0360- 5442. doi: 10 . 1016 / j . energy . 2022 . 125075 [Online]. Available: https : / / www . sciencedirect.com/science/article/pii/ S0360544222019703

[20] X. Cui, M. Lee, C. Koo, and T. Hong, “Energy consumption prediction and household feature analysis for diferent residential building types using machine learning and SHAP: Toward energy-eficient buildings,” Energy and Buildings, vol. 309, p. 113 997, 2024, issn: 0378-

7788. doi: 10 . 1016 / j . enbuild . 2024 . 113997 [Online]. Available: https : / / www . sciencedirect.com/science/article/pii/ S0378778824001130

[21] M. R. Maarif, A. R. Saleh, M. Habibi, N. L. Fitriyani, and M. Syafrudin, “Energy usage forecasting model based on long shortterm memory (LSTM) and eXplainable artificial intelligence (XAI),” Information-an International Interdisciplinary Journal, vol. 14, no. 5, 2023, issn: 2078-2489. doi: 10 . 3390 / info14050265 [Online]. Available: https : / / www.mdpi.com/2078-2489/14/5/265

[22] Y. Zhang, R. Ma, J. Liu, X. Liu, O. Petrosian, and K. Krinkin, “Comparison and explanation of forecasting algorithms for energy time series,” Mathematics, vol. 9, no. 21, 2021, issn: 2227-7390. doi: 10 . 3390 / math9212794 [Online]. Available: https : / / www . mdpi . com / 2227-7390/9/21/2794

[23] E. Dolgintseva, H. Wu, O. Petrosian, A. Zhadan, A. Allakhverdyan, and A. Martemyanov, “Comparison of multi-step forecasting methods for renewable energy,” Energy Systems, Mar. 7, 2024, issn: 1868-3975. doi: 10. 1007/s12667- 024- 00656- w [Online]. Available: https://doi.org/10.1007/s12667- 024-00656-w

[24] C. van Zyl, X. Ye, and R. Naidoo, “Harnessing eXplainable artificial intelligence for feature selection in time series energy forecasting: A comparative analysis of Grad-CAM and SHAP,” Applied Energy, vol. 353, p. 122 079, 2024, issn: 0306-2619. doi: 10.1016/j.apenergy.2023. 122079 [Online]. Available: https : / / www . sciencedirect.com/science/article/pii/ S0306261923014435

[25] C. Cakiroglu, S. Demir, M. Hakan Ozdemir, B. Latif Aylak, G. Sariisik, and L. Abualigah, “Data-driven interpretable ensemble learning methods for the prediction of wind turbine power incorporating SHAP analysis,” Expert Systems with Applications, vol. 237, p. 121 464, 2024, issn: 0957-4174. doi: 10.1016/j.eswa.

2023 . 121464 [Online]. Available: https : / / www.sciencedirect.com/science/article/ pii/S0957417423019668

[26] I. Simic, V. Sabol, and E. E. Veas, “XAI methods for neural time series classification: A brief review,” CoRR, vol. abs/2108.08009, 2021. [Online]. Available: https://arxiv.org/ abs/2108.08009

[27] A. Cossu, F. Spinnato, R. Guidotti, and D. Bacciu, “Drifting explanations in continual learning,” Neurocomputing, vol. 597, p. 127 960, 2024, issn: 0925-2312. doi: 10 . 1016 / j . neucom . 2024 . 127960 [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0925231224007318

[28] H. Turbé, M. Bjelogrlic, C. Lovis, and G. Mengaldo, “Evaluation of post-hoc interpretability methods in time-series classification,” Nature Machine Intelligence, vol. 5, no. 3, pp. 250– 260, Mar. 1, 2023, issn: 2522-5839. doi: 10 . 1038/s42256- 023- 00620- w [Online]. Available: https://doi.org/10.1038/s42256- 023-00620-w

[29] T. T. Nguyen, T. Le Nguyen, and G. Ifrim, “Robust explainer recommendation for time series classification,” Data Mining and Knowledge Discovery, pp. 1–42, 2024.

[30] D. I. Serramazza, T. T. Nguyen, T. Le Nguyen, and G. Ifrim, “Evaluating explanation methods for multivariate time series classification,” in Advanced Analytics and Learning on Temporal Data, G. Ifrim et al., Eds., Cham: Springer Nature Switzerland, 2023, pp. 159–175, isbn: 978-3-031-49896-1.

[31] H. Najjar et al., “Feature attribution methods for multivariate time-series explainability in remote sensing,” in IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium, IEEE, 2023, pp. 5014–5017.

[32] U. Schlegel, H. Arnout, M. El-Assady, D. Oelke, and D. A. Keim, “Towards a rigorous evaluation of XAI methods on time series,” in 2019 IEEE/CVF International Conference on

Computer Vision Workshop (ICCVW), IEEE, 2019, pp. 4197–4201.

[33] M. Veerappa, M. Anneken, N. Burkart, and M. F. Huber, “Validation of XAI explanations for multivariate time series classification in the maritime domain,” Journal of Computational Science, vol. 58, p. 101 539, 2022, issn: 1877-7503. doi: 10 . 1016 / j . jocs . 2021 . 101539 [Online]. Available: https : / / www . sciencedirect.com/science/article/pii/ S1877750321001976

[34] C. Draxl, B. Hodge, A. Clifton, and J. Mc-Caa, “The wind integration national dataset (WIND) toolkit,” Applied Energy, vol. 151, pp. 355–366, 2015.

[35] C. Feng, M. Cui, B.-M. Hodge, and J. Zhang, “A data-driven multi-model methodology with deep feature selection for short-term wind forecasting,” Applied Energy, vol. 190, pp. 1245– 1257, 2017, issn: 0306-2619. doi: 10.1016/j. apenergy . 2017 . 01 . 043 [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0306261917300508

[36] S. S. Chandel, A. Gupta, R. Chandel, and S. Tajjour, “Review of deep learning techniques for power generation prediction of industrial solar photovoltaic plants,” Solar Compass, vol. 8, p. 100 061, 2023, issn: 2772-9400. doi: 10 . 1016/j.solcom.2023.100061 [Online]. Available: https : / / www . sciencedirect . com / science/article/pii/S2772940023000292

[37] I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” in Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2, ser. NIPS’14, Cambridge, MA, USA: MIT Press, 2014, pp. 3104–3112.

[38] T. Luong, H. Pham, and C. D. Manning, “Efective approaches to attentionbased neural machine translation,” ArXiv, vol. abs/1508.04025, 2015. [Online]. Available: https : / / api . semanticscholar . org / CorpusID:1998416

[39] F. Doshi-Velez and B. Kim, Towards a rigorous science of interpretable machine learning, 2017. arXiv: 1702.08608 [stat.ML]. [Online]. Available: https://arxiv.org/abs/1702.08608

[40] M. Sundararajan, A. Taly, and Q. Yan, Axiomatic attribution for deep networks, 2017. arXiv: 1703.01365 [cs.LG].

[41] R. J. Aumann and L. S. Shapley, Values ofNon-Atomic Games. Princeton, NJ: Princeton University Press, 1974.

[42] M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in International Conference on Machine Learning, PMLR, 2020, pp. 9269–9278.

[43] H. Chen, J. D. Janizek, S. M. Lundberg, and S.-I. Lee, “True to the model or true to the data?” ArXiv, vol. abs/2006.16234, 2020. [Online]. Available: https : / / api . semanticscholar.org/CorpusID:220250235

[44] H. Yuan, M. Liu, L. Kang, C. Miao, and Y. Wu, “An empirical study of the efect of background data size on the stability of SHapley Additive exPlanations (SHAP) for deep learning models,” 2022. arXiv: 2204.11351.

[45] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, Curran Associates, Inc., 2019, pp. 8024–8035. [Online]. Available: http : / / papers.neurips.cc/paper/9015-pytorchan- imperative- style- high- performancedeep-learning-library.pdf

[46] R. He, H. Yang, S. Sun, L. Lu, H. Sun, and X. Gao, “A machine learning-based fatigue loads and power prediction method for wind turbines under yaw control,” Applied Energy, vol. 326, p. 120 013, 2022, issn: 0306-2619. doi: 10 . 1016 / j . apenergy . 2022 . 120013 [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0306261922012703

[47] I. Petenko, G. Casasanta, S. Bucci, M. Kallistratova, R. Sozzi, and S. Argentini, “Turbulence, low-level jets, and waves in the tyrrhenian coastal zone as shown by sodar,” Atmosphere, vol. 11, no. 1, 2020, issn: 2073-4433. doi: 10.3390/atmos11010028 [Online]. Available: https://www.mdpi.com/2073-4433/11/ 1/28

[48] National Weather Service, Flight environment - wind shear and turbulence. [Online]. Available: https : / / www . weather . gov / source / zhu/ZHU\_Training\_Page/winds/Wx\_Terms/ Flight\_Environment.htm

[49] L. Liu and R. J. Stevens, “Efects of atmospheric stability on the performance of a wind turbine located behind a three-dimensional hill,” Renewable Energy, vol. 175, pp. 926– 935, 2021, issn: 0960-1481. doi: 10 . 1016 / j.renene.2021.05.035 [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0960148121007114

[50] C. Frye, D. de Mijolla, T. Begley, L. Cowton, M. Stanley, and I. Feige, “Shapley explainability on the data manifold,” 2020. arXiv: 2006. 01272.

[51] E. Albini, J. Long, D. Dervovic, and D. Magazzeni, “Counterfactual shapley additive explanations,” in 2022 ACM Conference on Fairness, Accountability, and Transparency, ser. FAccT ’22, ACM, Jun. 2022. doi: 10.1145/ 3531146 . 3533168 [Online]. Available: http : //dx.doi.org/10.1145/3531146.3533168