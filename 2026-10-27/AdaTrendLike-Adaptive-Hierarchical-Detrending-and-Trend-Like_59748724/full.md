# AdaTrendLike: Adaptive Hierarchical Detrending and Trend Likelihood Scoring for Time Series Anomaly Detection

Van Kwan Zhi Koh, Songnan Lin, Rongmo, Luo, Zhiping Lin, Life Senior Member, IEEE, and Bihan Wen, Senior Member, IEEE

Abstract—The rapid proliferation of Industrial Internet of Things devices has led to increasingly large-scale and nonstationary time-series data, posing significant challenges for timeseries anomaly detection (TSAD). Existing approaches often focus on reconstructing or forecasting observations while overlooking the role of trend dynamics, which constitute a major source of non-stationarity and contain valuable information for anomaly identification. To address this limitation, we propose AdaTrend-Like, a TSAD framework that directly estimates trend likelihoods for anomaly detection. Specifically, we introduce Standardization Embedding to capture local statistical characteristics as latent representations and AdaDetrending to adaptively decompose input series into trend and detrended components. The detrended component is processed through a Fast Fourier Transform and a complex-valued convolutional neural network to learn normal periodic patterns, while TrendEst models the distribution of the trend component. Based on the estimated distribution, TrendLike computes anomaly scores by measuring the likelihood of observed trends, enabling effective detection of deviations from normal behavior. In addition to improving detection performance, the proposed framework provides feature-level interpretability by identifying anomalies through deviations in the estimated trend distribution. Extensive experiments on five public benchmark datasets demonstrate that AdaTrendLike consistently outperforms state-of-the-art TSAD methods, validating its effectiveness and robustness across diverse application scenarios.

Index Terms—Time series anomaly detection, frequencydomain representation learning

## I. INTRODUCTION

With the widespread deployment of Industrial Internet of Things devices, the volume of time series data has grown rapidly, creating a pressing need for robust Time Series Anomaly Detection (TSAD) tools for large-scale sensory data. In cybersecurity systems, malicious activity is flagged when the network exhibits unusual temporal behavior. In cyber-physical systems, which integrate digital and physical infrastructure, TSAD methods are utilized to detect physical anomalies caused by cyber intrusions in domains such as water treatment plants [1] and unmanned aerial vehicles [2]. In IT infrastructures, monitoring systems track server metrics to detect anomalies indicative of system failures or resource bottlenecks [3], [4]. Despite differences in anomaly types across domains, the core challenge remains detecting deviations from normal behavior in temporally dependent, non-stationary time series.

Classical methods such as Local Outlier Factor (LOF), Isolation Forest (IF), and One-Class Support Vector Machine (OCSVM) [5]–[7] are widely used. However, with the increasing scale and complexity of time series data, these methods struggle to capture temporal dependencies and high-dimensional feature interactions. Consequently, learningbased methods have emerged as a promising alternative by leveraging representation learning to model complex temporal and feature relationships [8]–[10]. These learning-based approaches typically rely on reconstruction errors to identify anomalies, which can be unreliable because normal variations or seasonal trends may be misclassified as anomalies. Moreover, this approach does not consider predictive uncertainties, which are crucial for informed and interpretable anomaly detection, such as identifying the features that contribute to anomalies.

Non-stationarity in real-world time series arises from multiple factors and can obscure the distinction between normal and abnormal patterns [11]. One of these factors is the trend component, which represents the progressive long-term behavior of the data and can be leveraged in two main ways: as a dominant feature for modeling and as a means to detrend the series, promoting local stationarity. Trends can be extracted using traditional statistical models, but their effectiveness is limited because these models rely on the assumption of local stationarity [12]. In contrast, learning-based decomposition methods address this by incorporating mechanisms that extract the trend component within the inner blocks of the models [11], [13], [14]. Although the inner blocks provide rich representations that capture underlying trends, these methods rely on trend separation mainly to improve reconstruction rather than explicitly exploiting the trend component for TSAD.

To address these challenges, we propose AdaTrendLike, an Adaptive Hierarchical Detrending and Trend Likelihood Scoring method for TSAD. Our method first employs a Standardization Embedding (SE) module that encodes the local statistical properties of time series as latent embeddings. To further handle non-stationary behaviors, we propose Adaptive Detrending (AdaDetrending), a hierarchical module that progressively extracts trend components through multiple layers. At each layer, local trends are estimated using moving averages, and the residuals representing the detrended components are processed through a Fast Fourier Transform (FFT), followed by a complex-valued Convolutional Neural Network (CNN). This combination enhances the model’s capability to accurately capture normal behavior in time series exhibiting periodicity via frequency representations and local dependency via convolution. Finally, the trend components extracted from all layers are aggregated and modeled by the Trend Estimator (TrendEst) to estimate a global trend distribution, while the Trend Likelihood Scoring (TrendLike) module computes anomaly scores based on the likelihood of observed trends. By explicitly modeling the distribution of the trend component and performing anomaly detection through likelihood estimation, the proposed framework achieves improved robustness under non-stationary dynamics. Unlike existing methods that primarily exploit trend information to facilitate signal reconstruction, AdaTrendLike directly integrates trend modeling into the anomaly detection process. Specifically, deviations from the estimated trend distribution are treated as anomaly evidence, enabling effective detection of trend-related anomalies while enhancing robustness to non-stationary time-series data. The main contributions are as follows:

• We propose AdaTrendLike, a framework comprising four modules: Standardization Embedding, AdaDetrending, TrendEst, and TrendLike. The framework hierarchically decomposes representations into trend and detrended components to address non-stationarity and estimates the trend distribution to capture deviations with feature-level interpretability.

• The AdaDetrending module performs hierarchical detrending using FFT, followed by a complex-valued CNN that effectively models normal temporal patterns, particularly in time series with periodic behavior.

• The TrendLike module computes anomaly scores for TSAD by leveraging the trend distribution estimated by TrendEst, enabling robust anomaly detection and interpretability through the quantification of deviations from the learned trend distribution.

The remainder of this paper is organized as follows. Section II reviews the related work. Section III presents the proposed methodology. Section IV presents the experimental results. Finally, Section V concludes the paper.

## II. RELATED WORK

We first review two common TSAD paradigms, namely deep one-class and reconstruction-based methods, followed by decomposition methods for handling non-stationarity and frequency-domain representation learning for periodic patterns, both of which are closely related to our method.

## A. Deep One-Class Methods

Classical methods such as IF [5], OCSVM [6], and LOF [7] have been widely used for unsupervised anomaly detection. Building on these foundations, advanced models like Deep-SVDD [15], which jointly train a neural network with Support Vector Data Description (SVDD) [6] to enclose normal data in a minimal hypersphere, and THOC [16], which employs dilated recurrent neural networks with skip connections for multi-resolution features, have emerged. COUTA [17] further introduces uncertainty quantification and anomaly perturbation to enhance confidence in detection. However, detecting anomalies as deviations from the learned data manifold remains challenging in high-dimensional spaces due to the curse of dimensionality.

## B. Reconstruction-based Methods

Deep learning methods often incorporate inductive biases, such as specific training objectives or architectural constraints, that favor the reconstruction of normal patterns. The resulting reconstruction errors are then used to identify anomalous segments. For instance, TranAD [18] employs an adversarial training procedure to amplify reconstruction errors, while EvoAAE [19] optimizes a variational autoencoder–adversarial framework through particle swarm optimization. However, in both cases, the reliance on reconstruction-based objectives can lead to over-generalization, enabling anomalous patterns to be reconstructed too accurately and thereby masking their detection [8]. MoARNN-AM [2] similarly uses a genetic algorithm to optimize the hyperparameters of recurrent neural network variants and attention mechanisms. However, it is designed for multi-class anomaly detection and relies on labeled anomalies, which are typically scarce in real-world settings [17], [20]. Rather than depending on extensive hyperparameter optimization or labeled data, other methods introduce model-specific inductive biases. AT [9] amplifies normal–abnormal scores using the discrepancy between an association vector from a multi-head self-attention mechanism and a prior association vector based on a Gaussian distribution. MEMTO [8] employs a Transformer encoder to extract latent representations, along with a gated memory module that is incrementally updated to capture prototypical normal patterns. Although these models are designed to reconstruct normal patterns effectively, they are trained exclusively on normal data. Consequently, variations arising from aleatoric uncertainty may be mistakenly flagged as anomalies.

Instead of focusing on model design, some approaches improve detection performance by refining anomaly scoring criteria. DAGMM [21] employs a Gaussian Mixture Model for anomaly scoring, NSPR [20] scores anomalies based on pointwise and sequential reconstruction deviations, and TranAD [18] uses adversarial reconstruction errors. Although these strategies achieve strong empirical performance, they lack an explicit mechanism to account for deviations from the normal data distribution, limiting their robustness to non-anomalous variations.

## C. Decomposition of Dominant Components

Learning from dominant decomposed components in time series analysis involves separating signals into key components, such as trend, seasonality, and residuals, to improve representation learning [11], [13], [14], [22]. TransDe [23], SDDformer [24], and Time-Frequency analysis based TSAD (TFAD) model [22] are methods that incorporate time-domain decomposition for TSAD. These methods primarily employ time-domain decomposition to separate temporal components, making the learned representations dependent on predefined decomposition assumptions, limiting their ability to capture complex temporal dynamics. Therefore, recent methods perform decomposition in the latent space. Autoformer [14] extracts trend components directly from latent representations, while D3R [11] decomposes latent representations into trend and stable components before recombination. Although these models effectively learn trends to facilitate representation learning, they do not explicitly exploit trend information as an anomaly cue for TSAD. As a result, they fail to leverage the valuable information encoded in the trend component, where deviations from the underlying temporal trajectory can serve as strong indicators of anomalous behavior.

## D. Frequency-domain Representation Learning

Typically, FFT is applied directly to the input time series to transform time-domain signals into the frequency domain, enabling the capture of frequency-based patterns [22], [25]– [27]. MAGE [27] transforms the input time series into the frequency domain using FFT, followed by a graph neural network to learn the feature topology. Similarly, TFAD method applies FFT to the decomposed components before modeling. Instead of raw or decomposed features, recent studies perform FFT on latent representations [13], [28]. Fedformer [13] combines FFT with transformer self-attention, selecting random frequency components to capture dominant patterns. FreTS [28] applies FFT to latent features and processes them with a complex-valued multilayer perceptron (MLP) to model global spectral components. Furthermore, existing approaches are primarily designed for TSF, with limited exploration in TSAD. In contrast, our method introduces frequency-learning biases to improve anomaly separability and employs a Complex CNN to capture frequency-domain periodicity and local temporal dependencies.

## III. METHODOLOGY

## A. Problem Formulation

Given a multivariate time series with a total length $M ,$ we segment it into a set of sub-series using a rolling window of length L with a stride value s, where each subseries serves as the input and label for the model. Let ${ \bf X } = [ { \bf x } _ { 1 } , { \bf x } _ { 2 } , . . . , { \bf x } _ { L } | { \bf x } _ { t } \in \mathbb { R } ^ { d _ { x } } ]$ denote a subseries of the multivariate time series, where L is the length of the window and $d _ { x }$ is the number of multivariate features. This corresponds to a sequence of labels denoted as $\mathbf { Y } = [ y _ { 1 } , \dots , y _ { L } | y _ { t } \in \mathbb { Z } _ { 2 } ]$ where $y _ { t } = 1$ indicates that the time step is abnormal, and $y _ { t } = 0$ indicates that the time step is normal. The model objective is to predict the abnormal time steps of X by computing the anomaly scores for all time steps $\hat { \mathbf { A } } = [ \hat { a } _ { 1 } , \dotsc , \hat { a } _ { L } | \hat { a } _ { t } \in \mathbb { R } ]$ and using a predefined threshold ζ, such that the predicted labels $\hat { \mathbf { Y } } = [ \hat { y } _ { 1 } , \hdots , \hat { y } _ { L } | \hat { y } _ { t } \in \mathbb { Z } _ { 2 } ]$ are determined by $\hat { y } _ { t } = 0 ,$ , if $\hat { a } _ { t } < \zeta$ and $\hat { y } _ { t } = 1$ , if $\hat { a } _ { t } \geq \zeta$

## B. Overview of AdaTrendLike

As shown in Fig. 1, our method comprises four modules: Standardization embedding (SE), AdaDetrending, TrendEst, and TrendLike. The SE module first standardizes the input time series X and projects the standardized data into a latent feature space $\mathbf { Z } _ { D } ^ { ( 0 ) }$ . At the same time, this module encodes the statistical properties of the input into latent embeddings representing the mean $\mathbf { E } _ { \mu }$ and standard deviation $\mathbf { E } _ { \sigma }$ . AdaDetrending is a hierarchical module that progressively extracts local trend components by using a moving average across multiple layers. At each layer, the detrended residuals are mapped to the frequency domain via a real FFT and processed using complex-valued CNNs, forming the FreqCNN module. After the final AdaDetrending layer, the resulting detrended component $\mathbf { Z } _ { D } ^ { ( n _ { d } ) }$ is used to reconstruct the detrended reconstruction D<sup>ˆ</sup> via a Multilayer Perceptron (MLP). Meanwhile, the trend components $\mathbf { Z } _ { T } ^ { ( l _ { d } ) }$ from each layer are aggregated using a learnable matrix W through a nonlinear weighted sum to produce the overall trend component $\mathbf { Z } _ { T T }$ . This component is passed to the TrendEst module, which estimates the parameters of the trend distribution, including the mean $\hat { \pmb { \mu } } _ { T }$ , covariance $\hat { \Sigma } _ { T }$ , and temporal vector πˆ. Finally, the TrendLike module utilizes these parameters to compute the anomaly scores A<sup>ˆ</sup> , which are used to identify anomalous segments across time steps.

## C. Standardization Embedding

Stationarity is essential for reliable time series analysis [12]. Although global standardization using training statistics is commonly applied, it may remove meaningful local patterns, such as magnitudes and variances, limiting model learning [29], [30]. Inspired by [29], we propose standardizing the input features based on the local input window rather than the entire training dataset and using the standardization parameters as embedding features.

Given an input time series X, we first compute the mean $\pmb { \mu } _ { i n }$ and standard deviation $\pmb { \sigma } _ { i n }$ along the feature dimension at each time step and use them to obtain the standardized input X<sup>¯</sup> . The standardized input X<sup>¯</sup> is then projected by an MLP into the latent representation $\mathbf { Z } _ { D } ^ { ( 0 ) } \ \in \ \overline { { \mathbb { R } ^ { L \times d _ { m } } } }$ , where $d _ { m }$ denotes the latent dimension. Unless otherwise specified, all MLPs in this paper consist of two linear layers with a ReLU activation function in between. To preserve local statistical information lost during standardization, $\pmb { \mu } _ { i n }$ and $\pmb { \sigma } _ { i n }$ are separately encoded using two dedicated MLPs, producing $\mathbf { E } _ { \mu } \in \mathbb { R } ^ { d _ { m } }$ and $\mathbf { E } _ { \sigma } \in \mathbb { R } ^ { d _ { m } }$ , respectively. A Softplus activation is applied to $\mathbf { E } _ { \sigma }$ to ensure positivity, enabling it to serve as a valid scaling factor. Overall, we denote this module as $S E ( \mathbf { X } )$ which outputs ${ \bf Z } _ { D } ^ { ( 0 ) } , { \bf E } _ { \mu } ,$ and $\mathbf { E } _ { \sigma }$ . The embeddings $\mathbf { E } _ { \mu }$ and $\mathbf { E } _ { \sigma }$ are subsequently used in the AdaDetrending module to denormalize the latent representations at each layer.

## D. Adaptive Detrending

Trends represent the long-term behavior of a time series, providing a reference for normal patterns and enabling detrending to mitigate non-stationarity. Unlike conventional approaches that extract trends directly in the time domain, the proposed method performs trend extraction in the projected latent space [14], yielding a more expressive representation. Furthermore, our proposed hierarchical decomposition progressively learns multi-level trend representations and their corresponding detrended components, enabling the model to capture evolving trends across multiple temporal scales. At each detrending layer $l _ { d } ,$ the trend component $\dot { \bf Z } _ { T } ^ { ( l _ { d } ) }$ is obtained by applying a moving average $M A ( \cdot , k _ { a v g } )$ with a window size $k _ { a v g }$ along the time step dimension. This trend is then subtracted from the input $\mathbf { Z } _ { D } ^ { ( l _ { d } - 1 ) }$ , with edge values padded by replicating adjacent time steps to preserve the sequence length. As described,

![](images/e77e30c860d0771f35247659160daf7c5410ec8fff55de27f18cd9e77ca3e046.jpg)  
Fig. 1: Overview of the AdaTrendLike framework. Red circles (#1–#4) indicate the sequential stages of the process. (#1) Standardization Embedding: Maps input X to initial input representation $\mathbf { Z } _ { D } ^ { ( 0 ) }$ and extracts standardized embeddings $\mathbf { E } _ { \sigma } , \mathbf { E } _ { \mu }$ (#2) AdaDetrending: Removes the trend via moving average $\mathbf { Z } _ { T } ^ { ( l _ { d } ) }$ and captures periodic patterns via FreqCNN, yielding the detrended component $\mathbf { Z } _ { D } ^ { ( l _ { d } ) }$ . (#3) TrendEst: Aggregates layer-wise trends $\mathbf { Z } _ { T T }$ to estimate distribution parameters $( \hat { \mu } , \hat { \Sigma } )$ and the temporal vector $\hat { \pi } .$ . (#4) TrendLike computes the step-wise anomaly score $\hat { a } _ { t }$ from the weighted log-likelihood of the predicted trend $\hat { \mathbf { t } } _ { m }$ under the estimated distribution.

$$
\begin{array} { r } { \tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } = \mathbf { Z } _ { D } ^ { ( l _ { d } - 1 ) } - \mathbf { Z } _ { T } ^ { ( l _ { d } ) } , } \end{array}\tag{1}
$$

where $\mathbf { Z } _ { T } ^ { ( l _ { d } ) } \ = \ M A ( \mathbf { Z } _ { D } ^ { ( l _ { d } - 1 ) } , k _ { a v g } ) \ \in \ \mathbb { R } ^ { L \times d _ { m } }$ is the trend component, $\tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } \in \mathbb { R } ^ { L \times d _ { m } }$ is the detrended component, and the layer index $l _ { d }$ ranges from 1 to $n _ { d }$ . Modeling time series in the frequency domain facilitates more effective identification of periodic patterns that may be difficult to detect in the time domain [13], [28], [31]. Motivated by the findings in [28], we employ a complex-valued CNN to process the frequency-domain representations at progressively increasing resolutions, ranging from coarse to fine. The CNN facilitates the learning of local structures inherent in the frequency domain, thereby enabling the model to emphasize dominant periodic components. This hierarchical processing enhances the characterization of normal behavioral patterns in time series exhibiting periodicity. The operation is represented as $F r e q C N N ( \tilde { \mathbf { Z } } _ { D } ^ { ( \tilde { l } _ { d } ) } )$ with a residual connection. For each layer $l _ { d } ,$ the operation is defined as:

$$
\mathbf { Z } _ { D } ^ { ( l _ { d } ) } = F r e q C N N ( \tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } ) + \tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } .\tag{2}
$$

We now formally define the FreqCNN module. Let $\tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } \in$ $\mathbb { R } ^ { L \times d _ { m } }$ denote the detrended latent representation at the $l _ { d ^ { - } }$ th layer. We first transform the input time series into the frequency domain using a real-valued fast Fourier transform (rFFT), which computes only the positive-frequency components of the real-valued input, denoted as $\mathcal { F } _ { d _ { l } }$ with frequency <sup>l</sup><sub>f</sub> resolution $d _ { l _ { f } }$ . This transformation is given by $\begin{array} { r l } { \mathbf { Z } _ { C } ^ { ( l _ { f } ) } } & { { } = } \end{array}$ $\mathcal { F } _ { d _ { l _ { f } } } ( \tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } )$ , where $\begin{array} { r } { d _ { l _ { f } } = \frac { L } { 2 ^ { l _ { f } } } } \end{array}$ denotes the frequency resolution at layer $l _ { f } ,$ for $0 \le l _ { f } < \bar { n _ { f } }$ , with $n _ { f }$ being the total number of frequency resolution layers, and $\mathbf { Z } _ { C } ^ { ( l _ { f } ) } \in \mathbb { C } ^ { ( \frac { d _ { l _ { f } } } { 2 } + 1 ) \times d _ { m } }$ is the frequency-domain representation. After that, we apply a complex-valued CNN and inverse rFFT $\mathcal { F } _ { d _ { l _ { f } } } ^ { - 1 }$ , as described:

$$
\mathbf { Z } _ { F } ^ { ( l _ { f } ) } = \mathcal { F } _ { d _ { l _ { f } } } ^ { - 1 } \big ( \mathrm { C o m p l e x C N N } ( \mathbf { Z } _ { C } ^ { ( l _ { f } ) } ) \big ) ,\tag{3}
$$

where ComplexCNN(·) denotes a complex-valued convolution defined under standard complex multiplication. The real and imaginary components are processed by two bias-free CNN branches with identical architecture, each followed by ReLU activation and configured with a grouping factor $g _ { \mathrm { c n n } }$ to reduce computational and memory overhead. The processed representation $\mathbf { Z } _ { F } ^ { ( l _ { f } ) } \in \mathbb { R } ^ { d _ { l _ { f } } \times d _ { m } }$ from each resolution is then concatenated along the $d _ { l _ { f } }$ dimension, and standardization embeddings are applied to denormalize the latent representation, expressed as follows:

$$
\mathbf { Z } _ { \mathrm { m u l } } = \mathbf { C } \mathbf { a } \mathbf { t } _ { d _ { l _ { f } } } \left( \mathbf { Z } _ { F } ^ { ( 0 ) } , \ldots , \mathbf { Z } _ { F } ^ { ( n _ { f } - 1 ) } \right) \odot \left( \mathbf { 1 } \mathbf { E } _ { \sigma } ^ { \top } \right) + \mathbf { 1 } \mathbf { E } _ { \mu } ^ { \top } ,\tag{4}
$$

where $\mathrm { C a t } _ { d _ { l _ { f } } } ( \cdot )$ denotes the concentration along the feature dimension $\begin{array} { r } { d _ { l _ { f } } , \mathbf { 1 } \in \mathbb { R } ^ { \sum _ { i = 0 } ^ { n _ { f } - 1 } \left\lfloor \frac { L } { 2 ^ { i } } \right\rfloor } , \odot } \end{array}$ denotes the element-wise product, and $\mathbf { Z } _ { \mathrm { m u l } } \in \mathbb { R } ^ { \sum _ { i = 0 } ^ { n _ { f } - 1 } \left\lfloor \frac { L } { 2 ^ { i } } \right\rfloor \times d _ { m } }$ . Finally, we apply an MLP to $\mathbf { Z } _ { \mathrm { m u l } }$ , mapping it back to the original latent shape and producing the output $\mathbf { Z } _ { D } ^ { ( l _ { f } ) } \in \mathbb { R } ^ { L \times d _ { m } }$ . Whereas the trend component $\bar { \bf Z } _ { T } ^ { ( l _ { d } ) }$ , each layer captures temporal patterns at varying scales, progressing from local to global as the layer depth increases.

With the trends extracted from each layer $\mathbf { Z } _ { T } ^ { ( l _ { d } ) }$ , the overall trend representation $\mathbf { Z } _ { T T }$ is obtained by integrating multiscale trends via a nonlinear weighted summation: $\mathbf { Z } _ { T T } =$ $\begin{array} { r } { \sum _ { l _ { d } = 1 } ^ { n _ { d } } f \left( \mathbf { W } ^ { ( l _ { d } ) } \mathbf { Z } _ { T } ^ { ( l _ { d } ) } \right) } \end{array}$ , where $\mathbf { W } ^ { ( l _ { d } ) } \in \mathbb { R } ^ { L \times L }$ is a learnable matrix for the $l _ { d } \mathrm { - t h }$ layer and $f ( \cdot )$ denotes the ReLU function. After the AdaDetrending module, the detrended component $\mathbf { Z } _ { D } ^ { ( n _ { d } ) }$ is passed through an MLP to reconstruct the detrended signal $\hat { \textbf { D } } \in \mathbb { R } ^ { L \times d _ { a } }$ , while the overall trend representation $\mathbf { Z } _ { T T } \in \mathbb { R } ^ { L \times d _ { m } }$ is fed into the TrendEst to estimate the parameters of the trend distribution.

## E. Trend Estimator

TrendEst models the trend distribution as a multivariate Gaussian: $\mathbf { T } \sim \mathcal { N } ( \hat { \pmb { \mu } } _ { T } , \hat { \pmb { \Sigma } } _ { T } )$ . Although this Gaussian assumption does not fully capture the true underlying distribution, it provides a flexible approximation for complex patterns while enabling robust uncertainty estimation. The overall trend $Z _ { T T }$ is used to estimate the mean $\hat { \pmb { \mu } } _ { T }$ via a one-dimensional (1D) CNN, 1DCNNµ, formulated as follows:

$$
\begin{array} { r } { \hat { \pmb { \mu } } _ { T } = 1 \mathrm { D C N N } _ { \mu } ( \mathbf { Z } _ { T T } ) \in \mathbb { R } ^ { L \times d _ { x } } , } \end{array}\tag{5}
$$

where $1 \mathrm { D C N N } _ { \mu }$ denotes a two-layer, one-dimensional (1D) CNN applied along the temporal dimension with a kernel size of 3 and ReLU activation, whose output layer projects to a shape of $\mathbb { R } ^ { L \times d _ { x } }$

The estimation of $\hat { \Sigma } _ { T }$ requires positive definiteness and numerical stability, but incurs substantial memory and computational overhead. To address this, we leverage the overall trend $\mathbf { Z } _ { T T }$ to predict a low-rank factor matrix $\mathbf { Z } _ { \sigma }$ . Then, a softplus transformation, DiagonalSoftplus(·), is applied to the diagonal entries to ensure positivity. The lower-triangular component is then extracted using tril(·), and the resulting matrix is used as a Cholesky factor to obtain a valid covariance matrix. As formulated,

$$
\mathbf { Z } _ { \sigma } = \mathrm { R e s h a p e } \big ( 1 \mathrm { D C N N } _ { \sigma } ( \mathbf { Z } _ { T T } ) \big ) \in \mathbb { R } ^ { L \times d _ { x } \times r } ,\tag{6}
$$

$$
\mathbf { L } _ { \sigma , l } = \mathrm { t r i l } \big ( \mathrm { D i a g o n a l S o f t p l u s } ( \mathbf { Z } _ { \sigma , l } \mathbf { Z } _ { \sigma , l } ^ { \top } ) \big ) ,\tag{7}
$$

$$
\begin{array} { r } { \hat { \Sigma } _ { T , l } = { \bf L } _ { \sigma , l } { \bf L } _ { \sigma , l } ^ { \top } \in \mathbb R ^ { d _ { x } \times d _ { x } } , \quad l = 1 , \ldots , L , } \end{array}\tag{8}
$$

where 1DCNN shares the same architecture as $1 \mathrm { D C N N } _ { \mu }$ but outputs a matrix in $\mathbb { R } ^ { L \times d _ { x } r }$ , which is reshaped into $\mathbb { R } ^ { L \times d _ { x } \times r }$ to obtain a sequence of low-rank factor matrices $\mathbf { Z } _ { \sigma , l } \in \mathbb { R } ^ { d _ { x } \times r }$ We set $r = d _ { m } / 1 6 \ll d _ { m }$ , obtaining a compact low-rank parameterization that reduces memory and computation.

Since time series anomalies often persist or influence neighboring observations, step-wise anomaly scores should account for local temporal context. Therefore, we introduce a temporal vector πˆ to model the relative importance of neighboring time steps within a local window, formulated as:

$$
\hat { \pi } = \frac { 1 \mathrm { D C N N } _ { \pi } ( \mathbf Z _ { T T } ) } { T e m p } \in \mathbb R ^ { L } ,\tag{9}
$$

where $\pmb { \hat { \pi } } = \hat { \pi } _ { 1 : L } = [ \hat { \pi } _ { 1 } , \hat { \pi } _ { 2 } , \dots , \hat { \pi } _ { L } ] ^ { \top }$ , 1DCNN denotes a two-layer, 1D CNN applied along the temporal dimension with a kernel size of 3 and ReLU activation, whose output layer maps to $\mathbb { R } ^ { L }$ . Here, Temp $\in \mathbb { R }$ is a learnable scaling parameter initialized as $\sqrt { d _ { m } }$ , which controls the sharpness of the softmax distribution used in both the loss function and anomaly score computation. Together, $\hat { \mu } _ { T } , \hat { \Sigma } _ { T }$ , and πˆ are used to compute the anomaly scores A<sup>ˆ</sup> for each time step.

## F. Training Process

The objective of the loss function is to jointly minimize the reconstruction loss of the detrended output, denoted as $\mathcal { L } _ { R } ,$ , and maximize the likelihood of the estimated trend distribution, denoted as $\mathcal { L } _ { L L }$ . The reconstruction loss is expressed as:

$$
\mathcal { L } _ { R } = \frac { 1 } { L \cdot d _ { x } } \left\| \mathbf { X } - \mathbf { T } - \hat { \mathbf { D } } \right\| _ { F } ^ { 2 } ,\tag{10}
$$

where $\mathbf { T } = M A ( \mathbf { X } , k _ { a v g } )$ denotes the ground truth trend. The $M A ( \cdot , k _ { a v g } )$ is employed due to its simplicity and effectiveness in estimating underlying trends. The parameters of the multivariate Gaussian distribution are estimated by scaling the trend likelihoods with a time-dependent weight $\hat { w } _ { t } .$ , expressed as:

$$
\mathcal { L } _ { L L } = \log \sum _ { t = 1 } ^ { L } \hat { w } _ { t } p ( \mathbf { t } _ { t } | \hat { \pmb { \mu } } _ { T , t } , \hat { \Sigma } _ { T , t } ) ,\tag{11}
$$

where $\mathbf { t } _ { t }$ denotes the trend component at time step $t ,$ corresponding to the t-th row of T, and $p ( \mathbf { t } _ { t } \mid \hat { \mu } _ { T , t } \hat { \Sigma } _ { T , t } )$ denotes the probability density of a multivariate Gaussian distribution evaluated at $\mathbf { t } _ { t } .$ . The time-dependent weights are defined as $\pmb { \hat { w } } \ = \ \mathrm { s o f t m a x } ( \hat { \pi } _ { 1 : L } )$ , where the softmax is applied along the temporal dimension. The total loss function is defined as: $\mathcal { L } _ { T } = \mathcal { L } _ { R } - \lambda \mathcal { L } _ { L L }$ , where $\lambda > 0$ is a hyperparameter that controls the trade-off between reconstruction loss and the accuracy of the trend distribution. The complete training procedure is outlined in Algorithm 1.

```latex
Algorithm 1 Training process
Input: $\mathbf { X } \in \mathbb { R } ^ { L \times d _ { x } }$
Initialize: $\mathbf { Z } _ { T T } = \mathbf { 0 }$
repeat
${ \bf \dot { Z } } _ { D } ^ { ( 0 ) } , { \bf E } _ { \mu } , { \bf E } _ { \sigma } = \mathrm { S E } ( { \bf X } )$
for l = 1 to n do
$\mathbf { Z } _ { T } ^ { ( l _ { d } ) } = M A ( \mathbf { Z } _ { D } ^ { ( l _ { d } - 1 ) } , k _ { a v g } )$
$\tilde { \mathbf { Z } } _ { D } ^ { ( l _ { d } ) } = \mathbf { Z } _ { D } ^ { ( l _ { d } - 1 ) } - \mathbf { Z } _ { T } ^ { ( l _ { d } ) }$
$\mathbf { Z } _ { D } ^ { ( \lfloor _ { d } ) } = F r e q C N N ( \bar { \mathbf { \tilde { Z } } } _ { D } ^ { ( \lfloor _ { d } ) } ) + \tilde { \mathbf { Z } } _ { D } ^ { ( \lfloor _ { d } ) }$
$\begin{array} { r } { \mathbf { Z } _ { T T } ^ { - } = \mathbf { Z } _ { T T } + f \left( \mathbf { W } ^ { ( \overline { { l } } _ { d } ) } \mathbf { Z } _ { T } ^ { ( l _ { d } ) } \right) ^ { - } } \end{array}$
end for
$\hat { \mathbf { D } } = \overset { \mathbf { \ i } \mathbf { \times } \mathbf { \times } } { M L P } ( \mathbf { Z } _ { D } ^ { ( n _ { d } ) } )$
$\hat { \pmb { \mu } } _ { T } , \hat { \pmb { \Sigma } } _ { T } , \hat { \pmb { \pi } } = \hat { \bf { \tau } }$ TrendEs $( { \bf z } _ { T T } )$
until $\mathcal { L } _ { T }$ converges
```

## G. TrendLike

Unlike reconstruction-based anomaly scoring methods that rely on reconstruction errors to identify anomalies, we propose TrendLike, which leverages the estimated trend distribution. This approach improves sensitivity to subtle anomalies and better distinguishes between normal fluctuations and true abnormal behavior. The anomaly score $a _ { t }$ at time step t is defined as a weighted negative log-likelihood over a local temporal window:

$$
\hat { a } _ { t } = - \log \sum _ { m = \operatorname* { m a x } ( 1 , t - \delta ) } ^ { \operatorname* { m i n } ( M , t + \delta ) } \bar { w } _ { m } p \big ( \hat { \mathbf { t } } _ { m } \mid \hat { \pmb { \mu } } _ { T , m } , \hat { \Sigma } _ { T , m } \big ) ,\tag{12}
$$

where $\hat { \mathbf { t } } _ { m }$ denotes the m-th row of $\begin{array} { r } { \hat { \bf T } = { \bf X } - \hat { \bf D } . } \end{array}$ , corresponding to the trend reconstruction, δ denotes the window size, t the time step, and M the total length of the time series. The timedependent weights $\pmb { \bar { w } } = \mathrm { s o f t m a x } \big ( \hat { \pi } _ { t - \delta : t + \delta } \big )$ define a normalized distribution over the local window $[ t - \delta , t + \delta ]$ along the time axis, thereby capturing the influence of neighboring time steps. This procedure is applied to all time steps up to M, rather than only within the training window length L. The resulting anomaly scores are denoted by $\hat { \mathbf { A } } = [ \hat { a } _ { 1 } , \hat { a } _ { 2 } , \dots , \hat { a } _ { M } ]$ from which anomalous time steps Y<sup>ˆ</sup> are identified using a thresholding criterion.

## IV. EXPERIMENTS

## A. Experiment Setup

1) Datasets: This study utilizes five publicly available datasets from real-world applications: the Server Machine Dataset (SMD) [4], the Soil Moisture Active Passive (SMAP) and Mars Science Laboratory (MSL) datasets from NASA [32], the Secure Water Treatment (SWAT) dataset [1], and the Pooled Server Metrics (PSM) dataset [3]. Missing values are handled via forward filling, where the most recent non-null value is propagated to fill gaps. For SMD, MSL, and SMAP, which consist of multiple entities, the data are preprocessed by concatenating entities along the time-step dimension. Each dataset comes with predefined training and testing splits, with anomalies appearing exclusively in the test sets. Each dataset is provided with predefined training and testing partitions, with anomalous samples confined to the test partition. All datasets are normalized using min–max scaling to the range [−1, 1], with scaling parameters estimated exclusively from the training data.

2) Baseline Models: We evaluate a diverse set of TSAD baselines spanning classical methods, including IF [5], OCSVM [6], and LOF [7]; deep one-class approaches, including THOC [16] and COUTA [17]; and reconstructionbased methods, including DAGMM [21], TranAD [18], AT [9], MEMTO [8], NSPR [20], and D3R [11]. All models are evaluated on the provided datasets, except for D3R, which is designed to operate only on datasets with continuous-valued features. Experiments for THOC, AT, MEMTO, COUTA, NSPR, and D3R are conducted using publicly available implementations, while the remaining models are evaluated using the Multivariate TSAD toolkit [33]. Each dataset has predefined training and testing splits, with details provided in Table I.

TABLE I: Details of the datasets
<table><tr><td>Datasets</td><td>SMD</td><td>MSL</td><td>SMAP</td><td>SWAT</td><td>PSM</td></tr><tr><td>Dimensions</td><td>38</td><td>55</td><td>25</td><td>51</td><td>25</td></tr><tr><td>Entities</td><td>28</td><td>27</td><td>55</td><td>1</td><td>1</td></tr><tr><td>Number of train steps</td><td>708405</td><td>58317</td><td>140825</td><td>495000</td><td>132481</td></tr><tr><td>Number of test steps</td><td>708420</td><td>73729</td><td>444035</td><td>449919</td><td>87841</td></tr><tr><td>Anomaly steps/Test steps</td><td>4.16%</td><td>10.48%</td><td></td><td>12.83%12.14% 27.76%</td><td></td></tr></table>

3) Evaluation Metrics: Anomalous time steps are determined by a predefined threshold ζ. The predicted labels $\hat { \mathbf { Y } } = [ \hat { y } _ { 1 } , \dotsc , \hat { y } _ { L } | \hat { y } _ { t } \in \mathbb { Z } _ { 2 } ]$ are determined by $\hat { y } _ { t } = 0 , \mathrm { i f } \hat { a } _ { t } < \zeta$ and $\hat { y } _ { t } = 1$ , and if $\hat { a } _ { t } \geq \zeta ,$ where $\zeta$ is obtained by iteration via grid search to maximize the F1-score [8], [11], [20], referred to as the Best F1-score. A common evaluation practice, known as point adjustment, treats a predicted anomaly as correct if it falls within any truth anomalous segment [8]. However, several studies question this practice, as it enables even trivial or adversarial algorithms to achieve inflated performance scores [11], [20]. Consequently, we evaluate our model strictly based on its raw predictions, without applying point adjustment. In addition, we report threshold-free performance using the Area Under the Receiver Operating Characteristic curve (AUC) to provide a complementary evaluation.

4) Implementation Details and Hyperparameter Selection: Experiments are conducted using the PyTorch library [34] on a server equipped with an RTX 3090 GPU. The model is trained for 100 epochs with a batch size of 64, employing the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ . Early stopping is applied when the validation loss does not decrease for two consecutive epochs. For all public datasets, we use the training portion as defined by the dataset providers, which is assumed to be free of anomalies. From this training set, we adopt a 90/10 split to create training and validation subsets. We set the input window length equal to the anomaly scoring window, $L \ = \delta .$ , and a stride of $s \ = \ L / 4$ is applied across all datasets. The remaining hyperparameters, along with their search ranges and selected values, are summarized in Table II. For a fair comparison with benchmark models, labeled anomalies are used for hyperparameter tuning, following prior work [8], [11], [20]. A detailed sensitivity analysis is provided in Section IV-C5, offering practical guidance on which hyperparameters are most critical for achieving strong performance.

TABLE II: Search space and selected hyperparameters
<table><tr><td></td><td>Search space</td><td>SMD</td><td>MSL</td><td>SMAP</td><td>SWAT</td><td>PSM</td></tr><tr><td> $L$ </td><td>{64, 128, 256, 512, 1028}</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td> $d _ { m }$ </td><td>{64, 128, 256, 512, 1028}</td><td>256</td><td>512</td><td>256</td><td>256</td><td>64</td></tr><tr><td> $k _ { a v g }$ </td><td>{3, 5, 7, 9, 11}</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td> $k _ { f }$ </td><td>{3, 5, 7, 9, 11}</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td> $n _ { f }$ </td><td>{1, 2, 4, 6, 8}</td><td>4</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td> $n _ { d }$ </td><td>{1, 2, 4, 8, 16}</td><td>4</td><td>4</td><td>1</td><td>4</td><td>4</td></tr><tr><td>gcnn</td><td>{1, 8, 16, 32, 64}</td><td>16</td><td>16</td><td>64</td><td>16</td><td>16</td></tr><tr><td> $\lambda$ </td><td> $\tilde { \{ 1 e ^ { - 3 } , 0 . 1 , 1 , 1 0 , 1 0 0 \} }$ </td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table>

## B. Model Performance

This section presents quantitative evaluations against baseline models, followed by visualization analyses that provide additional insights into the results. For visualization consistency, in Figs. 4–5, 7, and 11, black denotes the trend reconstruction $\hat { \mathbf { T } } = \mathbf { X } - \hat { \mathbf { D } }$ , blue denotes the estimated trend mean $\hat { \pmb { \mu } } _ { T }$ with confidence intervals diag(Σ<sup>ˆ</sup> ), and red shading indicates ground-truth anomaly segments Y.

1) Main results: The main results are summarized in Table III, where bold values indicate the best results. Our proposed AdaTrendLike achieves the best performance on 4 out of 5 datasets, demonstrating its effectiveness in TSAD. While deep learning approaches have been proposed, classical methods such as IF and LOF remain strong baselines. Deep One-Class methods, such as THOC and COUTA, which extend classical architectures, offer only marginal gains over classical methods.

AT and MEMTO employ inductive biases to foster diverse representations; however, this may cause over-generalization that negatively impacts anomaly detection accuracy. DAGMM and NPSR employ unique anomaly scoring mechanisms that yield strong performance, highlighting the crucial role of effective anomaly scoring strategies. NPSR achieves highly competitive results by leveraging sequential pattern comparison to characterize normal behavior. Similarly, our model specifically utilizes trend information. D3R performs decomposition but relies on reconstruction loss, whereas our method employs TrendLike for anomaly scoring, achieving better results.

Notably, AdaTrendLike achieves the best performance on four out of five datasets, demonstrating strong effectiveness for TSAD. On SMD, the performance gain is less pronounced, likely due to its highly imbalanced anomaly distribution, with only 4.16% anomalous time steps (Table I). As shown in Fig. 4, many anomalous observations closely follow the underlying trends of normal data, resulting in weak anomaly evidence. In addition, SMD exhibits substantially shorter anomaly segments, with a mean length of 90, compared to MSL (216), SMAP (826), SWaT (1561), and PSM (339). Since our model focuses on learning trends that capture long-term temporal patterns, short anomaly segments provide limited evidence for reliably estimating trend deviations, thereby reducing the effectiveness of TrendLike-based anomaly scoring. Nevertheless, AdaTrendLike still achieves the second-best AUC on SMD, indicating competitive ranking performance. Overall, these results highlight the advantages of the proposed hierarchical detrending framework, frequency-domain modeling, and TrendLike-based anomaly scoring.

TABLE III: ROC and Best F1-Scores of all models
<table><tr><td rowspan="2">Models</td><td>SMD</td><td>MSL</td><td>SMAP</td><td>SWAT</td><td></td><td>PSM</td></tr><tr><td>|AUC F1</td><td>AUC F1</td><td>AUC</td><td>F1 AUC</td><td>F1</td><td>AUC F1</td></tr><tr><td>IF [5]</td><td>|0.664 0.176|0.603 0.252|0.646 0.312|0.845 0.736|0.685 0.500</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OCSVM [6]</td><td></td><td></td><td></td><td></td><td></td><td>0.623 0.228|0.525 0.241|0.393 0.227|0.881 0.758|0.573 0.465</td></tr><tr><td>LOF [7]</td><td></td><td></td><td></td><td></td><td></td><td>0.645 0.124|0.556 0.216|0.623 0.283|0.888 0.774|0.728 0.583</td></tr><tr><td>THOC [16]</td><td></td><td></td><td></td><td></td><td></td><td>0.653 0.229|0.586 0.218|0.412 0.227|0.893 0.754|0.707 0.501</td></tr><tr><td>COUTA [17]</td><td></td><td></td><td></td><td></td><td></td><td>[0.660 0.199|0.557 0.212|0.526 0.259|0.751 0.502|0.693 0.529</td></tr><tr><td>DAGMM [21]]</td><td></td><td></td><td></td><td></td><td></td><td>0.689 0.180|0.600 0.229|0.513 0.270|0.862 0.749|0.660 0.484</td></tr><tr><td>TranAD [18]</td><td>[0.655 0.171|0.549 0.192|0.572 0.255|0.888 0.819|0.733 0.569</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AT [9]</td><td></td><td></td><td></td><td></td><td></td><td>[0.400 0.044|0.445 0.166|0.606 0.264|0.478 0.097|0.455 0.368</td></tr><tr><td>MEMTO [8]</td><td>[0.503 0.038|0.495 0.030|0.511 0.113|0.767 0.527|0.502 0.061</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>D3R [11]</td><td>0.714 0.201</td><td></td><td></td><td></td><td></td><td>0.343 0.218|0.590 0.443</td></tr><tr><td>NSPR [20]</td><td>0.833 0.386|0.653 0.283|0.534 0.248|0.890 0.847|0.792 0.645</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>0.722 0.203|0.684 0.310|0.658 0.377|0.884 0.850|0.820 0.687</td><td></td><td></td><td></td><td></td><td></td></tr></table>

2) Visualization Analysis: Anomalies may arise from various features and time steps, but identifying which features are responsible is challenging due to the high dimensionality of the input data. Our model addresses this by providing a visual comparison between the estimated trends and trend reconstruction. From the SWAT dataset, four labeled scenarios, with feature names (e.g., P102) following the dataset documentation, are summarized in Table IV. ‘Attacked features indicate features subjected to external changes, while ‘Expected feature changes’ indicate features for which changes are anticipated. As shown in Fig. 2, scenarios 1, 2, and 4 show deviations between estimated trends and expected feature changes. Notably, MV303 in scenario 1 shows an unexpected deviation, suggesting a link to the attacked feature. For scenario 3, which involves gradual changes, our model does not detect this anomaly segment, likely because the trendlearning mechanism treats such changes as normal behavior.

Based on these scenarios, practitioners can assess anomalies from deviations along the estimated trend, highlighting contributing features and detection likelihood. For further analysis, trend estimation on other datasets are presented in Fig. 3. The MSL and SMAP datasets contain frequent short-term spikes, whereas the SWAT and PSM datasets exhibit smoother patterns. Our model captures these spikes within the estimated trend distribution, adapting to such patterns. The temporal vector provides a global representation of the features within a segment, capturing patterns such as sudden drops, mean shifts, and high-frequency fluctuations. During anomalous segments, it highlights noticeable changes in these characteristics, which can serve as indicators of anomaly occurrence.

TABLE IV: Labeled anomaly scenarios for SWAT dataset
<table><tr><td>Scenarios</td><td>Attacked features</td><td>Expected feature changes</td></tr><tr><td>1</td><td>P102</td><td>P102 is expected to switch from off to on.</td></tr><tr><td>2</td><td>AIT202</td><td>AIT202 and P203 are expected changes.</td></tr><tr><td>3</td><td>LIT101</td><td>LIT101 exhibits gradual changes.</td></tr><tr><td>4</td><td>UV401,AIT502,P501</td><td>All listed features are expected to change.</td></tr></table>

![](images/91e05f19d8b11be9f3354d7c8e5f51f9f5f37884b35f9fc01f55ecc5dfb74e18.jpg)  
(a) Scenario 1

![](images/7754e99780b98e68bf31497b4a310331edbe271ffbf5a2acbf051575e2aea6fc.jpg)  
(b) Scenario 2

![](images/a120e2d93cfabc87624c756937563a6e386a750e8f398a72575bef7a03cb2bc0.jpg)  
(c) Scenario 3

![](images/c57d87e9a7433a6eee4cf962149e7ce30dec1607aebc1fc8076e3d25ff971528.jpg)  
(d) Scenario 4  
Fig. 2: Visualization of anomaly scenarios for SWAT dataset

## C. Ablation Study

This section presents the ablation study results using the best F1-scores, unless otherwise specified.

1) Module Analysis: We conducted ablation studies by removing each module individually. Exclude Embedding removes the input standardization as described in Section III-C and the denormalization steps in (4) with $\mathbf { E } _ { \sigma }$ and $\mathbf { E } _ { \mu } .$ Exclude Detrending removes the detrending procedure in (1)

![](images/ccd31f430542697ac847b939b1315eedd7a9981e2ad90cf160442693e9533d49.jpg)  
(a) MSL

![](images/8aae7793829c05509b14f49988ef2a376efe97550830e095c5b2b9e040022176.jpg)

![](images/f2abced787f527301d159f31cdde9723bdf12bf1ad64eebda1072cbbbd60806a.jpg)  
(c) SWAT

(b) SMAP  
![](images/7c62745d50398edc704621a6d2bf4971891b1ad0bfb2a4100de7c483f14fe93b.jpg)  
(d) PSM

Fig. 3: Visualization of trend distribution  
![](images/56f89fdc7349167509f9a02a39a12a57319f91c908feae29306aea4d4383321e.jpg)  
Fig. 4: Trend distribution for selected SMD features

and $\mathbf { Z } _ { T T }$ in (5)-(8) is replaced by $\mathbf { Z } _ { D } ^ { ( n _ { d } ) }$ . Exclude FreqCNN replaces FreqCNN(·) with a linear layer followed by a ReLU activation, projecting from $L \ { \mathrm { t o } \ d _ { m } }$ and back to L. As shown in Table V, each exclusion leads to a drop in TSAD performance, highlighting the benefits of all components. To illustrate the effect of detrending, the green boxes in Fig. 5 highlight the comparison. For the first two features, where trend-pattern anomalies are present, the estimated trend in Fig. 5a deviates more from the trend reconstruction than in the Exclude Detrending variant (Fig. 5a), resulting in clearer anomaly discrimination. For the remaining two features with abrupt spikes, the differences are minor. Overall, detrending improves anomaly clarity, especially for trend-related anoma-

lies.

![](images/af58d30317cc11d49bc4b7fd4c8a910eca69448c725dc0d436d2a70a6516ffb2.jpg)  
(a) Ours

![](images/dce2f259e64beccdf14e26834b42fb6cd828fca1e4b0e319efef0a85297ce595.jpg)  
(b) Exclude Detrending  
Fig. 5: Comparison of PSM features across model variants

2) Effect of FreqCNN Variants: The proposed FreqCNN is a crucial module that learns strong periodic features, making the model more effective in learning normal behavior that exhibits periodicity. In Table VI, the symbol % denotes the change of specific sub-components within FreqCNN. To assess their impact, we have conducted ablation studies by changing the rFFT resolutions of $\frac { L } { 2 ^ { l _ { f } } }$ to a constant resolution of L and substituting the complex-valued CNN with a complex-valued MLP. In Table VI, the use of $\frac { L } { 2 ^ { l _ { f } } }$ resolution rFFT and the CNN architecture improves TSAD performance, demonstrating the importance of FreqCNN’s design choices.

3) Effect of Different Anomaly Criterion: This section analyzes the performance of the three anomaly criteria: MSE computes the anomaly score using the reconstruction loss in (10), where the ground-truth trend T is replaced with ${ \pmb { \mu } } _ { T }$ during both training and anomaly score computation; Exclude πˆ removes the temporal vector from TrendLike in (12) and from the likelihood loss in (11); Exclude Summation omits the summation over adjacent time steps in (12). As illustrated in Fig. 6, our proposed TrendLike method produces distinct and consistent anomaly scores. In contrast, variants that exclude the temporal vector tend to overestimate anomaly durations, yielding anomalous segments that exceed the ground-truth boundaries and thus increase false alarms, whereas those that exclude the summation exhibit random spikes even during normal segments. Table VII shows that TrendLike consistently delivers superior results. While our anomaly criterion achieves the best overall performance, it still fails to detect certain anomalous segments, as illustrated in Fig. 6, corresponding to the trend estimation in Fig. 7. The detected anomalies typically involve multiple features within the same time segment and are accompanied by abrupt changes, whereas undetected segments, highlighted in green, exhibit only minor deviations within the estimated trend distribution, indicating a limitation in capturing subtle anomalies.

4) Threshold selection protocols: For large-scale time series, labeled anomalies are often scarce due to high annotation costs and the rarity of anomalous events [9], [19], thus using labeled anomaly information for threshold selection [9], [20], [21] may be infeasible. Therefore, we conduct an ablation study using three label-free methods: median thresholding (ζ = median score), mean thresholding (ζ = mean score), and a K-means-based approach (ζ = larger cluster centroid minus twice its standard deviation). Table VIII shows that thresholds computed solely from unlabeled data do not consistently achieve optimal performance. Nonetheless, the K-means approach remains competitive, with results on MSL, SWaT, and PSM close to the best F1 scores.

TABLE V: Model variants
<table><tr><td>Model Variants MSL SWAT PSM</td></tr><tr><td>Exclude Embedding 0.218 0.838 0.690 Exclude Detrending 0.262 0.843 0.649 Exclude FreqCNN 0.299 0.823 0.565 Ours 0.310 0.850 0.687</td></tr></table>

TABLE VI: FreqCNN variants
<table><tr><td> $\frac { L } { 2 ^ { l _ { f } } }$ </td><td>resolution rFFT CNN MSL SWAT PSM</td></tr><tr><td>X X</td><td>0.238 0.831 0.662</td></tr><tr><td>X √</td><td>0.312 0.846 0.590</td></tr><tr><td>√</td><td>X 0.278 0.8490.549</td></tr><tr><td>√</td><td>√ 0.3100.8500.687</td></tr></table>

![](images/82e40cba047e762abe00fb57c614b8714f13487c8af50fa6ae93c730756821a5.jpg)  
Fig. 6: Anomaly scores for a segment of the SWAT dataset

![](images/2cc1d3b7aebe9ad32e17c0223feac8ec26b816ebdb70e818c65b93cd7fba264a.jpg)  
Fig. 7: Trend estimation for a segment of the SWAT dataset

TABLE VIII: Threshold values and F1-scores for different threshold selection protocols.
<table><tr><td rowspan="2">Methods</td><td colspan="2">SMD</td><td colspan="2">MSL</td><td colspan="2">SMAP</td><td colspan="2">SWAT</td><td colspan="2">PSM</td></tr><tr><td>ζ</td><td>F1</td><td>ζ</td><td>F1</td><td>ζ</td><td>F1</td><td>ζ</td><td>F1</td><td>ζ</td><td>F1</td></tr><tr><td>Median</td><td></td><td></td><td></td><td>21.040.98830.440.24614.05</td><td></td><td>0.293</td><td>28.18</td><td></td><td>0.217 13.85 0.438</td><td></td></tr><tr><td>Mean</td><td></td><td>21.070.134 30.450.30214.080.299</td><td></td><td></td><td></td><td></td><td>28.18</td><td></td><td>0.21713.86 0.451</td><td></td></tr><tr><td>K-means</td><td></td><td>21.00 0.080 30.46 0.300 13.95 0.252 29.39</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.84013.97 0.660</td><td></td></tr><tr><td>Best F1</td><td></td><td>22.10 0.203 30.46 0.310 14.50 0.378 29.20 0.850 14.00 0.687</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE VII: Anomaly criterion variants
<table><tr><td>Anomaly Criterion Variants MSL SWAT PSM</td><td></td></tr><tr><td>MSE</td><td>0.293 0.836 0.600</td></tr><tr><td>Exclude Summation</td><td>0.261 0.653 0.435</td></tr><tr><td>Exclude π</td><td>0.2960.8280.632</td></tr><tr><td>Exclude π &amp; Summation</td><td>0.232 0.794 0.642</td></tr><tr><td>TrendLike (Ours)</td><td>0.310 0.850 0.687</td></tr></table>

5) Hyperparameter Sensitivity Analysis: Each hyperparameter sensitivity analysis is conducted by varying one parameter over a predefined range while keeping all other parameters fixed, as listed in Table II. Across datasets, the PSM dataset exhibits the greatest sensitivity to hyperparameters, potentially due to its higher anomaly ratio. Accordingly, hyperparameter tuning may improve model fitting and enhance anomaly detection performance in datasets with higher anomaly ratios. As shown in Fig. 8, the sensitive hyperparameters are mainly $L , \ d _ { m }$ , and $n _ { d } .$ For $L ,$ shorter input lengths yield better performance, which declines as the input length increases. Meanwhile, increasing $d _ { m }$ may improve performance, but there is a trade-off between accuracy and model complexity. Varying the number of $n _ { d }$ layers allows the model to capture both local and global trends. For certain datasets, particularly SMAP and PSM, it has a significant impact, suggesting that proper tuning is necessary. The hyperparameter $g _ { \mathrm { c n n } }$ has little effect on most datasets except PSM, suggesting that larger values can be used to reduce computational complexity while maintaining accuracy. The remaining hyperparameter exhibits low sensitivity and can be set to constant values with minimal impact on performance. Overall, tuning the parameters $L ,$ $d _ { m } ,$ and $n _ { d }$ is generally sufficient to identify near-optimal hyperparameter settings for most datasets.

![](images/db1cbf41ddde4dc27ad3d3b620dc5748717a2b57f05add121f59aaf273615955.jpg)  
(a) L

![](images/94fb97c204fe8ddcfa8f80474b6dca1bc0161ac94a682f8b12285e43a94292ab.jpg)  
(b) $d _ { m }$

![](images/82461e75b55ba6187727b076dff64f329fc8e422cd4b1682cba87cd1e24166ef.jpg)

![](images/430ee58527e312e650d06fd5051d4e20f186027134052801397ba020ce66396e.jpg)  
(d) $n _ { f }$

(c) n<sub>d</sub>  
![](images/a0512b4aed56d14daaa6b8308deb8ca73b6fe811e0520e8bcb266509c690fe3a.jpg)

![](images/ff3d5fecdf890dd495144d13eaf996ecb05ac97d9806e78a6bc59489c1c35208.jpg)

(e) g<sub>cnn</sub>  
![](images/05e756d5c0836cad851304a4e19cc24bc536515f5eac584cbb26fb90f4eea04c.jpg)

(f) $k _ { f }$  
![](images/1ab2a581f8831b1fde2b32541882922ebd5946c20b696d91521e27e3c1a27c77.jpg)  
(g) $k _ { a v g }$  
(h) λ  
Fig. 8: Hyperparameters sensitivity

6) Complexity Analysis: Table IX summarizes the training and inference computational complexity and runtime of the four baseline models (IF, LOF, TranAD, and NSPR) in comparison with our proposed model on the SWaT dataset. For consistency and fair comparison, all deep learning (DL) models, including TranAD, NSPR, and the proposed model, are trained with a fixed batch size of 64, whereas IF and LOF do not require batch-wise training. Runtime is reported as training time per epoch, and inference time is measured per sample, where each sample corresponds to a window of 100 time steps. Among the baseline methods, IF incurs the lowest training cost. Although LOF is more computationally expensive, it remains relatively efficient compared to deep learning (DL) approaches, which require multiple training epochs. For per-sample inference, IF achieves the fastest runtime due to its shallow tree traversal. In contrast, LOF is the slowest among the baselines, as it requires computing knearest neighbor distances and estimating local densities for each sample. DL models, including TranAD, NSPR, and our model, entail higher training costs but maintain competitive inference efficiency.

The TrendLike module is the primary computational bottleneck. In particular, the final covariance reconstruction $\mathbf { L } _ { \sigma , l } \mathbf { L } _ { \sigma , l } ^ { \top }$ introduces a computational complexity of $\mathcal { O } ( L d _ { x } ^ { 3 } )$ . This term becomes the primary bottleneck when $d _ { x }$ is large. TranAD [18] employs a standard Transformer architecture with attention complexity $\mathcal { O } ( L ^ { 2 } d _ { m } )$ , while NSPR adopts Performer [35], a linearized Transformer variant with attention complexity $\mathcal { O } ( L r d _ { m } )$ , where $d _ { m } \gg r$ . All DL models include a feedforward network (FFN) with computational complexity $\mathcal { O } ( L d _ { m } ^ { 2 } )$ . When $d _ { m } \gg L , d _ { x }$ , the FFN term dominates across all methods, leading to similar overall computational costs. When $L \gg d _ { m } , d _ { x }$ , TranAD scales quadratically with L and becomes the most expensive, while NSPR remains the most efficient. Our method scales linearly with L but complexity depending on feature dimension $d _ { x }$ . When $d _ { x } \gg L , d _ { m }$ , our model becomes the most computationally expensive. In this regime, NSPR remains the most efficient, while the relative ordering between TranAD and our method depends on the specific values of $d _ { m }$ and $d _ { x }$ . TranAD may incur higher practical runtime due to its two-stage training procedure, while its inference speed remains competitive. NSPR introduces overhead from stacking multiple linearized Transformer blocks, which increases runtime despite its lower theoretical complexity. In contrast, the TrendLike module is computed in a single forward pass, enabling efficient training and inference. Despite its additional computational cost, our approach remains competitive with existing methods.

TABLE IX: Models complexity on the SWAT dataset
<table><tr><td>Models</td><td>Training/ runtime per epoch(s)</td><td>Inference/ runtime per sample(ms)</td><td>Parameters (Million)</td></tr><tr><td>IF [5]</td><td> $\mathcal { O } ( T \cdot \psi \cdot \log \psi ) / 1 . 4$ </td><td> $\mathcal { O } ( T \cdot \log \psi ) / 0 . 0 5$ </td><td>0.011</td></tr><tr><td>LOF [7]</td><td> $\mathcal { O } ( M ^ { 2 } d _ { x } ) / 6 9 . 8 8$ </td><td> $\mathcal { O } ( M L d _ { x } ) / 3 2 . 7$ </td><td>0.5</td></tr><tr><td>TranAD [18]</td><td> $\mathcal { O } ( L ^ { 2 } d _ { \mathrm { m } } ) / 1 9 4 . 6 3$ </td><td>same as training/3.51</td><td>0.16</td></tr><tr><td>NSPR [20]</td><td> $\mathcal { O } ( L r d _ { m } ) / 3 4 . 4 7$ </td><td>same as training/17.9</td><td>0.39</td></tr><tr><td>Ours</td><td> $\mathcal { O } ( L d _ { x } ^ { 3 } ) / 1 9 . 5 2$ </td><td>same as training/8.60</td><td>0.92</td></tr></table>

T = number of trees, ψ = subsample size, M = total length of the dataset

7) Distribution variants: To assess the impact of different distributional assumptions, we conduct an ablation study using four alternatives: an independent and identically distributed (i.i.d.) Gaussian distribution (G-dist), an i.i.d. Student’s tdistribution (t-dist), and quantile prediction. These variants are implemented by replacing the TrendEst(·) module without altering the overall model architecture. For i.i.d. G-dist, each element of the $L \times d _ { x }$ time series is modeled independently by a Gaussian distribution, with the mean $\pmb { \mu } \in \mathbb { R } ^ { L \times d _ { a } }$ and standard deviation $\sigma ~ \in ~ \mathbb { R } ^ { L \times d _ { x } }$ estimated using 1DCNNs, the latter with a softplus activation. For the i.i.d. Student’s t-distribution, an additional 1DCNN with a softplus activation is employed to estimate the degrees of freedom (DOF). Both variants are trained by maximizing the log-likelihood formulation. For quantile predictions, a 1DCNN estimates $q = 1 0 0$ quantiles and is optimized using the pinball loss [36]. The multivariate Student’s t-distribution (Multi t-dist) follows the same estimation procedure as the multivariate Gaussian distribution, with an additional 1DCNN using a softplus activation to estimate a time-dependent DOF parameter, producing an L-dimensional vector. For all variants, the computation of the temporal vector and detrended reconstruction losses remains unchanged. Table X shows that the i.i.d. assumption consistently yields the poorest performance, as it fails to capture the feature dependencies inherent in real-world time series. In contrast, the multivariate and quantile-based distributions achieve comparable and substantially better results, with the multivariate formulation exhibiting greater robustness across datasets. These findings underscore the importance of distributional modeling in TSAD, suggesting the multivariate formulation as a reliable default choice, while the most suitable distribution may still depend on the characteristics of the dataset.

TABLE X: Results for different distribution estimations
<table><tr><td rowspan="2">Distributions</td><td>SMD</td><td></td><td>MSL</td><td></td><td>SMAP</td><td></td><td>SWAT</td><td></td><td>PSM</td></tr><tr><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td><td>F1</td></tr><tr><td>i.i.d G-dist</td><td>|0.750 0.197 0.636 0.272 0.443 0.244 0.876 0.800 0.752 0.553</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>i.i.d t-dist</td><td>0.754 0.190 0.621 0.274 0.359 0.243 0.885 0.813 0.786 0.575</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Quantiles</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7790.2640.6460.296 0.4230.2420.899 0.7960.7330.564</td><td></td></tr><tr><td>Multi t-dist</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7230.184 0.6690.286 0.727 0.398 0.8830.8450.790 0.636</td><td></td></tr><tr><td>Ours</td><td>0.722 0.203 0.684 0.310 0.6580.377 0.884 0.850 0.820 0.687</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

We further assess the validity of the Multi G-dist assumption on the SMD and PSM datasets, which are selected as representative cases of relatively poor and strong anomaly detection performance, respectively. Under a Multi G-dist, the standardized residuals ${ \bf r } _ { i } ~ = ~ \hat { \Sigma } _ { T , i } ^ { - \frac { 1 } { 2 } } \left( \hat { \bf t } _ { i } - \hat { \pmb { \mu } } _ { T , i } \right)$ , where i denotes the time step, are expected to follow a standard normal distribution. Therefore, we compute the standardized residuals, aggregate them across features at each time step, and pool the resulting values over all sliding windows of length 128 for evaluation using Q–Q plots. As shown in Fig. 9, the SMD dataset exhibits greater deviations from normality than the PSM dataset, indicating a poorer fit to the Multi G-dist. This is consistent with its comparatively poorer anomaly detection performance. Conversely, the PSM dataset exhibits good agreement with normality in the central region of the Q–Q plots. Overall, the Multi G-dist provides a reasonable approximation of the underlying trend distribution. Nevertheless, as the estimated trend distributions may deviate from the Multi G-dist assumption, feature-level interpretation is considered a secondary analysis and is applied only to segments identified as anomalous by the anomaly score.

![](images/b1e1e784649ec382a8fc6cc7ee2dc6d841eea7854317b736a6f431ec4d7259f0.jpg)

![](images/bd3e3bf6f6937d0d5950c694b0f39d72915c54b4a74ed3aafa52a507ad0d14d7.jpg)

![](images/56a6108d231bea7edfe2457c10941fc9be6103a92d6640fc1b82808fedfeea1c.jpg)

![](images/2d2da10d267ac74cc756c6eb2771720e6599ab71076a8167ee74cbd7e8a816da.jpg)  
(a) SMD

![](images/f42eedf7901c2ba08ed3259f1c550332ce00b8d73e7b7b8eb9ecf4a462288cc9.jpg)

![](images/9abf83d316fc3a797feb3e0540d31b558f6c03961fcb9bb810d720eedde88686.jpg)

![](images/1be1e6df65bcb5170991b69f743af7d397277229072e0fb370d65bf7ac948cb9.jpg)  
(b) PSM

![](images/5098f3cf3e092b21561055ab89a05621fe39d017f85b019294fc9451e2e00026.jpg)  
Fig. 9: Q-Q plots of 4 features for SMD and PSM

8) Anomaly Contamination Robustness: Real-world datasets often contain undetected anomalies, which can distort the learned normal patterns [17]. To evaluate robustness under such conditions, we simulate anomaly contamination by injecting 10%–50% of abnormal samples from the original test set into the training data. All models are retrained on the contaminated training sets and evaluated under two settings. (i) Fig. 10a reports performance on the complete original test set, where the anomalous samples injected into the training set are also retained in the test set. This setting preserves the original test distribution and provides a comprehensive assessment of anomaly detection performance. (ii) Fig. 10b reports performance on the remaining test samples after a subset of anomalous samples has been incorporated into the training set. This setting isolates the impact of training contamination while avoiding overlap between the training and test sets. Comparing these two settings mitigates bias from different train–test splits, as some anomalies are more difficult to detect. The detailed analysis is provided as follows:

• For the SMD dataset, anomalies are subtle (Fig. 4), and only 4.16% of test time steps are anomalous. As a result, injecting additional anomalies has a limited impact on the training distribution, leading to similar performance across different contamination levels.

• For the MSL (Fig. 3a) and SMAP (Fig. 3b) datasets, anomaly contamination appears to have a limited impact on performance, as anomalies are typically erratic and manifest as spikes or abrupt changes. By explicitly modeling smooth long-term trends, our approach may better capture such deviations, contributing to improved robustness.

• For the SWAT dataset (Fig. 3c), smooth and gradual anomalies may be partially treated as normal variations due to the model’s emphasis on trend-level dynamics, increasing its susceptibility to contamination. Figs. 11a and 11b show that contamination causes overgeneralization and reduced anomaly detectability.

• For the PSM dataset (Fig. 3d), many anomalies fall within the estimated trend distribution. Consequently, training on contaminated data inflates the estimated variance to accommodate these fluctuations, as shown by the comparison between Figs. 11c and 11d.

Since evolving trends are a major source of time-series nonstationarity, we investigate whether sensitivity to imperfect trend estimation increases with dataset non-stationarity. The non-stationarity level is quantified using the average Wasserstein distance between consecutive Gaussian distributions parameterized by the sample mean and covariance of sliding windows with a length of 128 and a stride of 32, yielding scores of 0.198 (SMD), 0.187 (MSL), 0.266 (SMAP), 0.336 (SWAT), and 0.125 (PSM). To evaluate robustness under imperfect trend estimation, synthetic trend perturbations are introduced into the test set by injecting random level shifts and slope changes with varying magnitudes into randomly selected temporal segments, affecting 10%–50% of the time steps and features. As shown in Fig. 10c, although imperfect trend estimation causes performance degradation, the proposed model remains robust, with highly non-stationary datasets such as SWAT and SMAP exhibiting smaller performance degradation.

![](images/c26753aa692c50dea734a633b577c46423085e14682f2489a4ab752eceecca85.jpg)  
(a) Full data

![](images/879d39ffcbe27ad185b1ff862a4bf617f149cbe1e546fc6edaf3b1fa9933124e.jpg)  
(b) Partitioned data

![](images/19a119bceb33c1c799754647b255606a90fda56daf56e3bbf244e466e0e81957.jpg)  
(c) Synthetic anomaly

Fig. 10: Performance under injected anomalies  
![](images/2b9a3a0c8c8d6001f22f12b6ce0422cbe215e7f381b86fc0df6a12ddee0eab01.jpg)

![](images/15c42d2a9527cc4f7673c70e9c3a6ddd316bea2eb964bda5209fafe9c0a3e965.jpg)  
(b) 50% Anomaly (SWAT)

(a) Anomaly-free (SWAT)  
![](images/bea7e234fd94ede92f2ff2e01cfc1e9c144d8848447ba1ec67136fe43fbfc861.jpg)  
(c) Anomaly-free (PSM)

![](images/1b3b97c8c2349fedb1b6150ee0b676ef518eb7beb3c5aaf840113fd431f31cc7.jpg)  
(d) 50% Anomaly (PSM)  
Fig. 11: Trend estimation under anomaly contamination

## V. CONCLUSIONS

In conclusion, AdaTrendLike provides an effective framework for non-stationary time series by explicitly modeling trend and detrended components as distinct dominant elements. Hierarchical detrending facilitates estimation of their distribution and clearer separation from the ground truth. Complementing this, frequency-based convolutional neural networks further process the detrended component, improving anomaly detection performance. Together, these enable direct estimation of trend likelihood and provide feature-wise interpretability with actionable insights into which variables are affected by anomalous behavior. Despite these advances, datasets with subtle anomalies remain challenging, motivating future work to incorporate additional time-series components, such as cyclic and seasonal patterns, alternative distributional assumptions, and feature-wise anomaly scoring.

## REFERENCES

[1] A. P. Mathur and N. O. Tippenhauer, “Swat: a water treatment testbed for research and training on ics security,” in 2016 International Workshop on Cyber-physical Systems for Smart Water Networks (CySWater), 2016, pp. 31–36.

[2] K.-D. Lu, B.-X. Zhang, Y. Xu, Y. Shen, and Z.-G. Wu, “Moarnnam: Multi-objective automated recurrent neural network with attention mechanism for cyber-attack detection of uav,” IEEE Transactions on Consumer Electronics, vol. 72, no. 1, pp. 1738–1749, 2026.

[3] A. Abdulaal, Z. Liu, and T. Lancewicki, “Practical approach to asynchronous multivariate time series anomaly detection and localization,” in Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, 2021, pp. 2485–2494.

[4] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 2828– 2837.

[5] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 eighth ieee international conference on data mining. IEEE, 2008, pp. 413–422.

[6] D. M. Tax and R. P. Duin, “Support vector data description,” Machine learning, vol. 54, pp. 45–66, 2004.

[7] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “Lof: identifying density-based local outliers,” in Proceedings of the 2000 ACM SIGMOD international conference on Management of data, 2000, pp. 93–104.

[8] J. Song, K. Kim, J. Oh, and S. Cho, “Memto: Memory-guided transformer for multivariate time series anomaly detection,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[9] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly transformer: Time series anomaly detection with association discrepancy,” in International Conference on Learning Representations, 2022.

[10] X. Peng, Y. Lin, Q. Cao, Y. Cen, H. Zhuang, and Z. Lin, “Traffic anomaly detection in intelligent transport applications with time series data using informer,” in 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2022, pp. 3309–3314.

[11] C. Wang, Z. Zhuang, Q. Qi, J. Wang, X. Wang, H. Sun, and J. Liao, “Drift doesn’t matter: dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[12] R. J. Hyndman and G. Athanasopoulos, Forecasting: principles and practice, 3rd edition. OTexts, 2021.

[13] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 27 268–27 286.

[14] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” in Neural Information Processing Systems, 2021.

[15] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui, A. Binder, E. Muller, and M. Kloft, “Deep one-class classification,”¨ in International conference on machine learning. PMLR, 2018, pp. 4393–4402.

[16] L. Shen, Z. Li, and J. Kwok, “Timeseries anomaly detection using temporal hierarchical one-class network,” Advances in Neural Information Processing Systems, vol. 33, pp. 13 016–13 026, 2020.

[17] H. Xu, Y. Wang, S. Jian, Q. Liao, Y. Wang, and G. Pang, “Calibrated one-class classification for unsupervised time series anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 11, pp. 5723–5736, 2024.

[18] S. Tuli, G. Casale, and N. R. Jennings, “Tranad: deep transformer networks for anomaly detection in multivariate time series data,” Proc. VLDB Endow., vol. 15, no. 6, p. 1201–1214, Feb. 2022.

[19] G.-Q. Zeng, Y.-W. Yang, K.-D. Lu, G.-G. Geng, and J. Weng, “Evolutionary adversarial autoencoder for unsupervised anomaly detection of industrial internet of things,” IEEE Transactions on Reliability, vol. 74, no. 3, pp. 3454–3468, 2025.

[20] C.-Y. A. Lai, F.-K. Sun, Z. Gao, J. H. Lang, and D. Boning, “Nominality score conditioned time series anomaly detection by point/sequential reconstruction,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 76 637–76 655.

[21] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, “Deep autoencoding gaussian mixture model for unsupervised anomaly detection,” in International Conference on Learning Representations, 2018.

[22] C. Zhang, T. Zhou, Q. Wen, and L. Sun, “TFAD: A decomposition time series anomaly detection architecture with time-frequency analysis,” in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, ser. CIKM ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 2497–2507.

[23] W. Zhang and C. Luo, “Decomposition-based multi-scale transformer framework for time series anomaly detection,” Neural Networks, vol. 187, p. 107399, 2025.

[24] S. Fu, X. Gao, B. Li, F. Zhai, J. Lu, B. Xue, J. Yu, and C. Xiao, “Multivariate time series anomaly detection via separation, decomposition, and dual transformer-based autoencoder,” Applied Soft Computing, vol. 159, p. 111671, 2024.

[25] H. Ren, B. Xu, Y. Wang, C. Yi, C. Huang, X. Kou, T. Xing, M. Yang, J. Tong, and Q. Zhang, “Time-series anomaly detection service at microsoft,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 3009–3017.

[26] S. Li, Z. Wang, X. Wang, Z. Yin, and M. Yao, “Frequency-enhanced and decomposed transformer for multivariate time series anomaly detection,” Applied Intelligence, vol. 55, no. 6, pp. 1–18, 2025.

[27] W. Wang, X. Zhou, T. Qiu, and L. Wang, “MAGE: Multiperiodic adaptive graph evolution guided anomaly detection in industrial iot,” IEEE Transactions on Industrial Informatics, vol. 21, no. 8, pp. 6126– 6136, 2025.

[28] K. Yi, Q. Zhang, W. Fan, S. Wang, P. Wang, H. He, N. An, D. Lian, L. Cao, and Z. Niu, “Frequency-domain mlps are more effective learners in time series forecasting,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[29] Y. Liu, H. Wu, J. Wang, and M. Long, “Non-stationary transformers: Exploring the stationarity in time series forecasting,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 9881–9893.

[30] Q. Wang, Y. Zhu, Z. Sun, D. Li, and Y. Ma, “A multi-scale patch mixer network for time series anomaly detection,” Engineering Applications of Artificial Intelligence, vol. 140, p. 109687, 2025.

[31] T. Zhou, Z. MA, x. wang, Q. Wen, L. Sun, T. Yao, W. Yin, and R. Jin, “Film: Frequency improved legendre memory model for long-term time series forecasting,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 12 677–12 690.

[32] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, “Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding,” in Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 2018, pp. 387–395.

[33] J. Liu, W. Gu, Z. Chen, Y. Li, Y. Su, and M. R. Lyu, “Mtad: Tools and benchmarks for multivariate time series anomaly detection,” arXiv preprint arXiv:2401.06175, 2024.

[34] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[35] K. M. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlos, P. Hawkins, J. Q. Davis, A. Mohiuddin, L. Kaiser, D. B. Belanger, L. J. Colwell, and A. Weller, “Rethinking attention with performers,” in International Conference on Learning Representations, 2021.

[36] R. Wen, K. Torkkola, B. M. Narayanaswamy, and D. Madeka, “A multihorizon quantile recurrent forecaster,” in NeurIPS 2017, 2017.