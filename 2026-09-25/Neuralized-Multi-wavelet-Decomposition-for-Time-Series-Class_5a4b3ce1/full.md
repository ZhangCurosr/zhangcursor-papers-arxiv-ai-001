# Neuralized Multi-wavelet Decomposition for Time Series Classification and Forecasting

Xiaohan Jiang, Jingyuan Wang, Jiahao Ji, Yongyao Wang, Chen Yang, and Junjie Wu

Abstract—Time series analysis is fundamental in domains such as finance, healthcare, and meteorology. Real-world time series often exhibit multiscale characteristics shaped by diverse latent factors, resulting in intricate temporal patterns and rich frequency structures. However, existing approaches typically focus on either frequency-domain decomposition or time-domain pattern extraction in isolation, neglecting their joint structure. This decoupled modeling limits representation expressiveness and undermines performance in tasks requiring simultaneous temporal and spectral reasoning. To address this gap, we propose m-WCN, a novel end-to-end deep learning framework that neuralizes multi-wavelet decomposition for joint extraction of temporal patterns and frequency components. By approximating the classical GHM multi-wavelet transform with trainable convolutional operators and enforcing orthogonality constraints, m-WCN produces interpretable multi-resolution representations. Built on this foundation, we introduce two task-specific architectures: TFBC for time series classification, which boosts discriminative features across frequency scales, and FTB for forecasting, which ensembles frequency-aware predictors. Extensive experiments on 64 UCR datasets and seven public forecasting benchmarks demonstrate the effectiveness of our approach. Built on the neuralized m-WCN, our TFBC and FTB outperform various baseline models across diverse datasets, achieving average improvements of 19.97% in classification and 19.92% in forecasting tasks.

Index Terms—Time series classification, Time series forecasting, Multi-wavelet decomposition, Neuralization, Pattern and frequency analysis

## 1 INTRODUCTION

A <sup>Time</sup> <sup>series</sup> <sup>is</sup> <sup>a</sup> <sup>sequence</sup> <sup>of</sup> <sup>data</sup> <sup>points</sup> <sup>recorded</sup> <sup>in</sup>temporal order. Time series analysis, including time se- temporal order. Time series analysis, including time series classification (TSC) and time series forecasting (TSF) [1], plays a crucial role in understanding dynamic systems and supporting decision-making across diverse domains such as finance, healthcare, and meteorology [2]–[4].

In real-world scenarios, time series are often shaped by multiple latent factors, giving rise to signals with intricate temporal patterns and diverse frequency components. For instance, from a frequency-domain perspective, traffic flow data may contain low-frequency trends reflecting daily commuting patterns and high-frequency fluctuations caused by sudden events. From a temporal perspective, the same data can be segmented into distinct intervals such as morning rush hour, evening rush hour, and off-peak periods. Effectively modeling and disentangling both temporal and frequency characteristics is therefore essential for achieving robust, interpretable, and high-quality time series analysis. To this end, decomposition-based approaches have proven effective for disentangling these complex structures [5], [6]. By decomposing a time series into distinct subcomponents, such methods support more expressive representations, facilitate task-specific learning, and improve overall performance on both classification and forecasting problems.

In the literature, decomposition-inspired time series analysis methods are broadly categorized into two groups: frequency-based and time-based approaches. Frequency-based methods employ spectral decomposition techniques to transform a time series into a frequency domain representation, where each component captures information associated with a specific frequency. Common techniques in this category include the Discrete Fourier Transform (DFT) [7], Discrete Wavelet Transform (DWT) [8], and Z-transform [9]. These methods are grounded in strong mathematical foundations and offer interpretable decompositions that reveal intrinsic properties of the input signals. Despite their theoretical rigor, frequency-based methods often rely on a fixed and limited set of basis functions, which may impose overly restrictive assumptions about the underlying structure of the time series. As a result, they may struggle to effectively capture complex patterns, local variations, and nonstationary behaviors commonly found in real-world data.

The second category comprises time-based approaches, which treat a time series as an ordered sequence of data points and aim to uncover patterns directly through datadriven techniques. These methods typically decompose a series into multiple temporal segments or components, each representing a distinct underlying pattern. Representative techniques include Empirical Mode Decomposition (EMD) [10] and shapelet-based methods [11]–[13]. While time-based approaches offer flexibility and are well-suited to capturing localized or irregular patterns, they often neglect the frequency characteristics of the signal. This omission can limit their capacity to detect meaningful structures that manifest across different frequency scales, potentially compromising performance in tasks where frequency dynamics are critical.

The emergence of deep learning has introduced powerful tools for advancing decomposition-based time series analysis. Early approaches typically use frequency or temporal decomposition as a preprocessing step to extract handcrafted features, which are then fed into deep neural networks [14]–[16]. Although this loosely coupled paradigm can enhance performance compared to using raw time series, it suffers from a lack of end-to-end optimization, as the decomposition and learning stages are treated as separate processes with independently trained parameters. More recently, efforts have been made to incorporate decomposition techniques directly into end-to-end deep learning architectures [1], [17]. These integrated models aim to jointly optimize decomposition and representation learning. However, most of these methods primarily focus on frequency-domain decomposition, often overlooking the rich structural information embedded in temporal patterns. As a result, they fall short of fully capturing the complementary insights offered by both time- and frequency-domain representations. Therefore, there is a critical need for a unified framework that seamlessly integrates both time-domain pattern decomposition and frequency-domain component analysis within a learnable, end-to-end neural architecture. Such a framework should leverage the interpretability and structural rigor of traditional decomposition methods while incorporating the adaptive learning capabilities of deep networks.

To address these challenges, we propose multi-Wavelet Convolution Networks (m-WCN), a novel framework that performs simultaneous temporal pattern and frequency decomposition in a fully trainable, end-to-end manner. Our approach is grounded in the theory of Multi-Wavelet Decomposition (MWD), which we leverage to construct a pattern-frequency joint decomposition architecture. Specifically, MWD first transforms the input time series into multiple temporal pattern components via a pre-filtering step, and then further decomposes each pattern component into hierarchical frequency components using dedicated scaling and wavelet functions. To preserve the mathematical rigor of MWD while enabling data-driven flexibility, we neuralize the two core steps of the GHM multi-wavelet algorithm — an established instance of MWD — using convolutional neural networks. In the temporal pattern decomposition phase, m-WCN enhances the fixed GHM pre-filter with learnable convolutional kernels to adaptively extract representative time-domain patterns. In the frequency decomposition phase, m-WCN enhances the GHM scaling and wavelet functions using trainable two-dimensional convolutional kernels, enabling it to better capture frequency structures tailored to the input data. Unlike traditional multi-wavelet decomposition methods with fixed parameters, all components in m-WCN are fully learnable and can be fine-tuned to fit the training data for various learning tasks. This enables m-WCN to combine the theoretical advantages of multiwavelet signal decomposition with the powerful representation learning capabilities of deep neural networks. Furthermore, we introduce an orthogonal regularization term to promote diversity among learned components, thereby preserving the orthogonality property intrinsic to MWD and ensuring interpretability of the extracted patterns.

Based on m-WCN, we also propose two task-specific deep learning models for the time series classification (TSC) and time series forecasting (TSF), respectively. The key issue in TSC is to extract discriminative features from a time series. Therefore, we propose a m-WCN-based Time-Frequency Boosting Classification (TFBC) model, which leverages frequency components extracted by m-WCN in a coarse-to-fine boosting manner, where higher-frequency features are used to complement the information missed by lower-frequency ones. This design enables the model to exploit complementary time-frequency cues, enhancing its discriminative power for time series classification. For the TSF task, a key challenge lies in accurately modeling future dynamics by capturing latent trends across multiple frequency scales. To tackle this, we propose the Frequency TSMixer [18] Bagging (FTB) model, which processes each frequency component extracted by m-WCN using a dedicated TSMixer network. The outputs from all frequencyspecific TSMixer are then aggregated in a bagging manner to generate the final forecast, effectively leveraging complementary information across different frequency bands. To facilitate more effective model training, we design tailored pre-training strategies for both TFBC and FTB. For TFBC, we introduce a Frequency Contrastive Learning (FCL) objective that encourages consistency among representations across frequency components of the same input. For FTB, we propose a Frequency Representation Pre-training (FRP) strategy, which guides each predictor to predict its corresponding future frequency component, thereby enhancing the frequency-awareness and forecasting capability of the model.

We evaluate the effectiveness of m-WCN and its taskspecific variants through extensive experiments across diverse benchmarks. We evaluate TFBC on 64 UCR time series datasets for TSC, and FTB on seven real-world public datasets for TSF. The results demonstrate the two models superiority to various baselines by an average performance improvement of 19.97% and 19.92% in classification and forecasting tasks, respectively (See Sec. 4 of the Supplementary Materials for the calculation details of average performance improvement). The contributions of our work can be summarized as follows:

• We propose m-WCN, the first end-to-end deep learning framework that jointly integrates frequency and temporalpattern decomposition via a neuralized multi-wavelet design, bridging classic signal processing and modern representation learning.

• We develop two task-specific architectures: TFBC for classification and FTB for forecasting. TFBC employs frequency-wise boosting to capture complementary decision cues across scales, while FTB performs frequencyaligned forecasting by assigning dedicated predictors to each frequency band. Both architectures demonstrate strong adaptability to the structural characteristics of time series in their respective tasks.

• The proposed models achieved the state-of-the-art performance over a large number of real-world datasets.

## 2 RELATED WORK

Pattern Analysis (PA). It is a crucial technique for identifying distinctive data properties [10], [11]. In time series analysis, PA helps extract informative features to support downstream models [12], [19]. Several deep learning methods have successfully integrated PA with neural networks, achieving notable results. Examples include empirical mode decomposition [20], tensor decomposition [21], and shapelet-based methods [22]. However, many of these methods overlook frequency information, which can limit their ability to capture meaningful patterns.

Frequency Analysis. It is a key technique for revealing data characteristics in the frequency domain, using methods such as Discrete Fourier Transform (DFT) [7] and Discrete Wavelet Transform (DWT) [8]. In time series analysis, traditional methods typically incorporate frequency coefficients from discrete analyses as model features but lack deeper integration and refinement [23]. Deep learning has broadened the use of frequency analysis—neural networks can automatically tune key frequency coefficients [1], and neural operator learning has been combined with frequency analysis [24]. Self-attention mechanisms have also been used to integrate DFT and DWT [14]. However, existing approaches often fail to effectively combine frequency-based and pattern-based methods, both of which are critical in time series analysis. The Discrete Multi-Wavelet Transform (DMWT) [25], [26] extends DWT by integrating frequency and pattern information to extract joint features. However, DMWT cannot adapt pattern modes to varying data characteristics, often producing patterns that misalign with the actual data. Our approach addresses this by using deep neural networks to learn and extract combined frequency-pattern features that better reflect the data’s intrinsic regularities.

Time Series Classification (TSC). TSC aims to categorize time series patterns using models trained on labeled data. Traditional methods, including distance-based [27], [28], feature extraction [29], and ensemble approaches [23], often rely on handcrafted features like distance metrics and differences. However, these techniques may struggle with large or complex datasets. Recently, deep neural networks have become a powerful tool for TSC, capable of automatically learning complex features. This includes supervised feature mining [22], unsupervised feature learning [30], and Transformer-based models [31], [32]. While these models effectively generate diverse features through representational learning, they may still overlook unique aspects of time series data, such as mixed Pattern-Frequency features.

Time Series Forecasting (TSF). TSF refers to predicting future values of a time series using past and present data, which is widely adopted in nearly all application domains. A classic model is autoregressive integrated moving average (ARIMA) [33], with a great many variants, such as ARIMA with explanatory variables (ARIMAX) [34] and seasonal ARIMA (SARIMA) [35], to meet the requirements of various applications. In recent years, deep learning has emerged as the leading approach in this field. It includes various methodologies such as neural ordinary differential equations [36], [37], probabilistic forecasting models [38], and transformer-based architectures [39], [40]. Despite these advancements, integrating both pattern and frequency features remains a challenge, with few studies successfully addressing this dual consideration.

TABLE 1  
Notations used in multi-wavelet decomposition and m-WCN.
<table><tr><td rowspan=1 colspan=2>Notations</td><td rowspan=1 colspan=1>|Description</td></tr><tr><td rowspan=1 colspan=2> $\pmb { s } = ( s _ { 0 } , \dots , s _ { T } )$  $\pmb { c } = ( c _ { 1 } , \hdots , c _ { M } )$  $\pmb { s } _ { t } = ( s _ { t - L } , \dots , s _ { t } )$  $\boldsymbol z t = \left( s _ { t + 1 } , \ldots , s _ { t + L ^ { \prime } } \right)$ </td><td rowspan=1 colspan=1>|The input series for m-WCN and TSC.The one-hot category label for TSC.The input series for m-WCN and TSF.|The series to be predicted for TSF.</td></tr><tr><td rowspan=1 colspan=2> $X ^ { h _ { n } } , n \in \{ 1 , \ldots , N \}$  $X ^ { l _ { n } } , n \in \{ 1 , \ldots , N \}$  $\boldsymbol { Z } _ { t } ^ { h _ { n } }$ and $ { \boldsymbol { z } } _ { t } ^ { l _ { N } }$ </td><td rowspan=1 colspan=1>The n-th high frequency component of s.The n-th low frequency component of s.The frequency components of $_ { z t }$ in TSF.</td></tr><tr><td rowspan=1 colspan=2> $\psi ( t )$ and φ(t)</td><td rowspan=5 colspan=1>The wavelet and scaling function ofsingle wavelet decomposition.The wavelet functions of multi-waveletdecomposition.The scaling functions of multi-waveletdecomposition.The wavelet and scaling function of theGHM decomposition.The wavelet and scaling function of them-WCN model.</td></tr><tr><td rowspan=1 colspan=2> $\{ \psi _ { 1 } ( t ) , \ldots , \psi _ { K } ( t ) \}$ </td></tr><tr><td rowspan=1 colspan=2> $\{ \phi _ { 1 } ( t ) , \ldots , \phi _ { K } ( t ) \}$ </td></tr><tr><td rowspan=1 colspan=2>H and L</td></tr><tr><td rowspan=1 colspan=2> and č</td></tr><tr><td rowspan=2 colspan=2> $\pmb { g } ^ { l _ { n } }$ and $\pmb { g } ^ { h _ { n } }$ Θ, w</td><td rowspan=2 colspan=1>The representations of frequency in TSC.The representations of frequency in TSF.|Learnable parameters.</td></tr><tr><td rowspan=1 colspan=1>Etn and Eh</td></tr></table>

## 3 PRELIMINARIES

## 3.1 Notations

In this paper, we use lowercase letters in regular font $( a , b )$ to denote scalars, lowercase bold letters $( a , b )$ to denote vectors, uppercase bold letters $( A , B )$ to denote matrices, and uppercase calligraphic letters $( A , B )$ to denote tensors or sets of matrices. The uppercase letters in regular font (A, B) are used to denote constants.

For a matrix $\textbf { \em A } \in \mathbb { R } ^ { I \times J }$ , its i-th row vector and j-th column vector are denoted as $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \mathbf \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi $ and $\mathbf { \delta } _ { a ; j } ,$ respectively. For a third-order tensor $\pmb { \mathcal { A } } \in \mathbb { R } ^ { I \times J \times K }$ , its horizontal slices, lateral slices, and frontal slices are denoted as $\begin{array} { r } { A _ { i : } , A _ { : j : } , } \end{array}$ , and $A _ { : : k } ,$ respectively. The row, column, and tube fibers (vectors) of the tensor are denoted as $\mathbf { \Delta } a _ { i : k } , \mathbf { \Delta } a _ { : j k }$ , and $\mathbf { } _ { \mathbf { } } \mathbf { } _ { \mathbf { } } \mathbf { } \mathbf { } \mathbf { } \mathbf { } a _ { i j } \mathbf { } \mathbf { : } \mathbf { }$ , respectively.

Tab. 1 summarizes the notations used for multi-wavelet decomposition and our m-WCN framework. In this paper, subscripts $( \mathrm { e . g . } , s _ { t } )$ are used to denote the index of a variable within a series, vector, matrix, or tensor, while superscripts $( \mathbf { e . g . } , \pmb { X } ^ { l _ { n } } )$ are used to indicate the corresponding frequency components. Specifically, $* ^ { l _ { n } }$ denotes the low-frequency component at the n-th layer, and ∗<sup>hn</sup> denotes the high- $* ^ { h _ { n } }$ frequency component at the n-th layer.

## 3.2 Wavelet Decomposition

Given a series $\pmb { \mathscr { s } } = ( \mathscr { s } _ { 0 } , \ldots , \mathscr { s } _ { t } , \ldots , \mathscr { s } _ { T } ) .$ , wavelet decomposition applies a wavelet function $\psi ( t )$ and a scaling function $\phi ( t )$ to extract its high- and low-frequency components:

$$
s _ { t } ^ { h } = \sum _ { i = 0 } ^ { T } \psi ( t ) s _ { i } , \quad \mathrm { a n d } \quad s _ { t } ^ { l } = \sum _ { i = 0 } ^ { T } \phi ( t ) s _ { i } .\tag{1}
$$

Here, $\pmb { s } ^ { h } = ( s _ { 0 } ^ { h } , \ldots , s _ { t } ^ { h } , \ldots , s _ { T } ^ { h } )$ and $\pmb { s } ^ { l } = ( s _ { 0 } ^ { l } , \ldots , s _ { t } ^ { l } , \ldots , s _ { T } ^ { l } )$ denote the high-frequency and low-frequency component series, respectively. The decomposition in Eq. (1) effectively captures frequency information from the input series: high-frequency components reflect short-term variations, while low-frequency components capture long-term trends. Different wavelet decomposition methods employ various choices of wavelet and scaling functions [41].

## 3.3 Multi-wavelet DecompositionA1 A2

In the basic wavelet decomposition described in Sec. 3.2, the input is a scalar time series, and the algorithm appliesSeries a single wavelet function $\psi ( t )$ and a single scaling function $\phi ( t )$ to extract frequency components. This method, known as single wavelet decomposition, assumes that the series can be characterized by a single set of frequency components. However, real-world time series often exhibit complex and rich spectral structures, sometimes requiring decomposition into many frequency components. To better capture such complexity, Multi-Wavelet Decomposition (MWD) [42] provides a more expressive and flexible framework for<sup>Multi-wavelet</sup> <sup>Decomposition</sup> frequency analysis. The MWD process consists of two key steps: a time-domain pre-filtering step and a frequency-Predomain decomposition step.

Pre-filtering (Time-domain Pattern Decomposition). In the pre-filtering step, MWD transforms the scalar time series into a multivariate (vector) time series. Given a sequence $\pmb { \mathscr { s } } = ( \mathscr { s } _ { 1 } , \ldots , \mathscr { s } _ { t } , \ldots , \mathscr { s } _ { T } ) .$ , a linear transformation is applied to construct the vector series:

$$
\pmb { x } _ { t } = \pmb { P } \cdot \left( s _ { t } , s _ { t + 1 } , \ldots , s _ { t + M } \right) ^ { \top } ,\tag{2}
$$

where $P \in \mathbb { R } ^ { K \times ( M + 1 ) }$ is a filter matrix that maps a local segment $\left( s _ { t } , s _ { t + 1 } , \ldots , s _ { t + M } \right) ^ { \top }$ into the vector $\pmb { x } _ { t } \in \mathbb { R } ^ { K }$ at time slice t. The filter matrix P is predefined according to the specific multi-wavelet decomposition algorithm [25]. Over all $T$ time slices, this transformation produces a $K \cdot$ dimensional multivariate series $\pmb { X } = ( \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { t } , \ldots , \pmb { x } _ { T } ) ^ { \ 1 }$ Each subseries ${ \pmb x } _ { k : \pmb { \imath } } = ( x _ { k 1 } , \ldots , x _ { k T } )$ in X represents a projection of the original series onto a distinct subspace. In this way, the pre-filtering process extracts multiple time-domain patterns $\pmb { x } _ { k : } , k \in \{ 1 , \ldots , K \}$ from the original input series $s ,$ capturing diverse temporal structures within the data.

Wavelet Decomposition (Frequency-domain Component Decomposition). In the wavelet decomposition step, MWD utilizes multiple wavelet and scaling functions to process different subseries of X. Given K wavelet functions $\{ \psi _ { 1 } ( t ) , \ldots , \psi _ { K } ( t ) \}$ and scaling functions $\{ \phi _ { 1 } ( t ) , \ldots , \phi _ { K } ( t ) \}$ the vector series X is transformed as follows:

$$
\begin{array} { r }  \left( \begin{array} { c } { \tilde { x } _ { 1 t } ^ { h } } \\ { \vdots } \\  \tilde { x } _ { K t } ^ { \dot { h } } \rule { 0 ex } { 5 ex } \right) = \left( \begin{array} { c c } { \sum _ { i = 0 } ^ { T } \psi _ { 1 } ( t ) x _ { 1 i } } \\ { \vdots } \\ { \vdots } \\ { \sum _ { i = 0 } ^ { T } \psi _ { K } ( t ) x _ { K i } } \end{array} \right) , \quad \left( \begin{array} { c } { \tilde { x } _ { 1 t } ^ { l } } \\ { \vdots } \\  \tilde { x } _ { K t } ^ { \dot { l } } \rule { 0 ex } { 5 ex } \right) = \left( \begin{array} { c } { \sum _ { i = 0 } ^ { T } \phi _ { 1 } ( t ) x _ { 1 i } } \\ { \vdots } \\ { \sum _ { i = 0 } ^ { T } \phi _ { K } ( t ) x _ { K i } \rule { 0 ex } { 5 ex } \right) } \end{array} \end{array} \end{array} \end{array}\tag{3}
$$

Letting $\tilde { \pmb { x } } _ { t } ^ { h } = ( \tilde { x } _ { 1 t } ^ { h } , \ldots , \tilde { x } _ { K t } ^ { h } ) ^ { \top }$ and $\tilde { \mathbf { \mathbf { \em x } } } _ { t } ^ { l } = ( \tilde { x } _ { 1 t } ^ { l } , \ldots , \tilde { x } _ { K t } ^ { l } ) ^ { \top }$ , we define $\tilde { \mathbf { X } } ^ { h } = ( \tilde { \mathbf { x } } _ { 1 } ^ { h } , \dots , \tilde { \mathbf { x } } _ { T } ^ { h } )$ and $\tilde { \pmb { X } } ^ { l } = ( \tilde { \pmb { x } } _ { 1 } ^ { l } , \dots , \tilde { \pmb { x } } _ { T } ^ { l } )$ as the highand low-frequency representations of the vector series $\boldsymbol { x }$

To preserve the total length of the decomposed components relative to the original input sequence ${ \dot { X } } ,$ a downsampling operation is applied to both frequency representations:

$$
{ \cal X } ^ { h } = \tilde { \cal X } ^ { h } \downarrow 2 , { \cal X } ^ { l } = \tilde { \cal X } ^ { l } \downarrow 2 ,\tag{4}
$$

where ↓ 2 denotes a downsampling operation that reduces the sequence length by half. The resulting sequences $X ^ { h } \in \mathbb { R } ^ { K \times ( T / 2 ) }$ and $\boldsymbol { X } ^ { l } \in \mathbb { R } ^ { \breve { K } \times ( T / 2 ) }$ represent the high- and low-frequency components, respectively. Each subseries $\pmb { x } _ { k : } ^ { h }$ and $\boldsymbol { x } _ { k } ^ { l }$ in $X ^ { \tilde { h } }$ and $\dot { \boldsymbol { X } } ^ { l }$ corresponds to the frequency-domain transformation of a distinct time-domain pattern identified during the pre-filtering step.

![](images/9af95337e7f442e168eb841e3cb29b93408c194a4e31f5a5c6ab6cde8e9b2c03.jpg)  
Fig. 1. Comparison of single-wavelet and multi-wavelet decomposition (Pat: Pattern). A1 and A2 denote the temporal patterns associated with the morning peak and the evening peak, respectively. Single-wavelet decomposition processes the mixed patterns A1 and A2 together, whereas multi-wavelet decomposition first separates them through prefiltering and then performs frequency decomposition on the separated pattern components.

The MWD recursively applies the decomposition process defined in Eq. (3) and Eq. (4). Letting $X ^ { \hat { l } _ { 0 } } ~ = ~ X$ denote the input to the first layer, the high- and low-frequency components at the n-th layer are computed as:

$$
\left\{ X ^ { h _ { n } } , X ^ { l _ { n } } \right\} = \mathrm { M W D } \left( X ^ { l _ { n - 1 } } \right) ,\tag{5}
$$

where MWD(·) represents the composite operation of multiwavelet transformation and downsampling.

For an N-layer decomposition, the final output is a set of sequences $\{ X ^ { h _ { 1 } ^ { - } } , X ^ { h _ { 2 } } , \ldots , \bar { X } ^ { h _ { N } } , X ^ { l _ { N } } \}$ , where $\dot { \pmb { X } } ^ { h _ { 1 } } , \ldots , \pmb { X } ^ { h _ { N } }$ are the high-frequency components from each layer, and $X ^ { l _ { N } }$ is the low-frequency component obtained at the final layer. Intermediate low-frequency components are used only for recursive decomposition and are not retained in the final output.

Remark: Compared with single wavelet decomposition, multi-wavelet decomposition offers significant advantages in analyzing temporal compound characteristics within sequential signals. Fig. 1 illustrates this with an example of total traffic volume in a city over a single day. The series exhibits two distinct patterns—morning and evening peaks—representing compound characteristics. In single wavelet decomposition, both peaks must be analyzed simultaneously using the same wavelet basis, which may result in mixed or entangled frequency components. In contrast, multi-wavelet decomposition first applies a pre-filtering step to separate these patterns and then analyzes their frequency components using different wavelet and scaling functions. This leads to more homogeneous subseries, making them easier to model and predict in downstream tasks.

## 4 MULTI-WAVELET CONVOLUTION NETWORK

In this section, we introduce a novel network architecture, m-WCN (multi-Wavelet Convolution Networks). The core idea of m-WCN is to use a convolutional network structure to neuralize the pre-filtering and decomposition steps of multi-wavelet decomposition. Sec. 4.1 presents the backbone multi-wavelet algorithm adopted by m-WCN. Sec. 4.2 details the neuralization of the pre-filtering step, and Sec. 4.3 describes the neuralization of the decomposition step.

## 4.1 GHM Multi-wavelet and m-WCN Framework

The proposed m-WCN adopts the Geronimo–Hardin– Massopust (GHM) multi-wavelet as its backbone decomposition algorithm. GHM is a classical multi-wavelet transform, known for its compact support, orthogonality, and favorable regularity properties of its scaling and wavelet functions [43].

In the time-domain pattern decomposition step, GHM employs a predefined projection matrix to transform the original scalar series into a two-dimensional vector series. This projection matrix $P ,$ used in Eq. (2), is defined as:

$$
P = { \left( \begin{array} { l l l l l l l } { - 0 . 0 0 3 6 } & { 0 . 0 8 4 4 } & { 0 . 9 9 2 8 } & { 0 . 0 8 4 9 } & { - 0 . 0 0 3 6 } & { 0 . 0 0 0 1 5 } \\ { 0 . 0 0 0 1 5 } & { - 0 . 0 0 3 6 } & { - 0 . 0 8 4 9 } & { 0 . 9 9 2 8 } & { - 0 . 0 8 4 4 } & { 0 . 0 0 3 6 } \end{array} \right) } _ { d }\tag{6}
$$

By applying the projection matrix P, the GHM algorithm transforms the original scalar input series into a two-dimensional vector series, preparing it for subsequent frequency-domain decomposition.

For the wavelet decomposition step, GHM defines the scaling functions using four matrices:

$$
\begin{array} { r l } & { \pmb { L } _ { : 1 : } = \left( \begin{array} { c c } { \frac { 3 } { 5 \sqrt { 2 } } } & { - \frac { 1 } { 2 0 } } \\ { 0 } & { \frac { 9 } { 2 0 } } \end{array} \right) , \quad \pmb { L } _ { : 2 : } = \left( \begin{array} { c c } { \frac { 4 } { 5 } } & { - \frac { 3 \sqrt { 2 } } { 1 0 } } \\ { 0 } & { - \frac { 3 \sqrt { 2 } } { 1 0 } } \end{array} \right) , } \\ & { \pmb { L } _ { : 3 : } = \left( \begin{array} { c c } { \frac { 3 } { 5 \sqrt { 2 } } } & { \frac { 9 } { 2 0 } } \\ { 0 } & { - \frac { 1 } { 2 0 } } \end{array} \right) , \quad \pmb { L } _ { : 4 : } = \left( \begin{array} { c c } { 0 } & { \frac { 1 } { \sqrt { 2 } } } \\ { 0 } & { 0 } \end{array} \right) . } \end{array}\tag{7}
$$

These matrices form a third-order tensor $\pmb { \mathcal { L } } = ( \pmb { L } _ { : 1 : } , \pmb { L } _ { : 2 : } , \pmb { L } _ { : 3 : } ,$ $\scriptstyle { L _ { : 4 : } } )$ , which serves as a low-pass filter. The horizontal slices $\pmb { L } _ { 1 : : } \overset { ' } { \in } \mathbb { R } ^ { 4 \times 2 }$ and $\pmb { L } _ { 2 : : } \in \mathbb { R } ^ { 4 \times 2 }$ of $\pmb { \mathcal { L } }$ correspond to the two scaling functions of the GHM algorithm, i.e., $\phi _ { 1 } = { \pmb { L } } _ { 1 : }$ <sub>:</sub> and $\phi _ { 2 } = { \bf L } _ { 2 : : }$ . Similarly, the wavelet functions are defined by the following matrices:

$$
\begin{array} { r l r l } & { H _ { : 1 : } = \left( \begin{array} { c c } { - \frac { 1 } { 2 0 } } & { \frac { 1 } { 1 0 \sqrt { 2 } } } \\ { \frac { 9 } { 2 0 } } & { \frac { 9 } { 1 0 \sqrt { 2 } } } \end{array} \right) , } & & { H _ { : 2 : } = \left( \begin{array} { c c } { - \frac { 3 \sqrt { 2 } } { 1 0 } } & { \frac { 3 } { 1 0 } } \\ { - \frac { 3 \sqrt { 2 } } { 1 0 } } & { - \frac { 3 } { 1 0 } } \end{array} \right) , } \\ & { H _ { : 3 : } = \left( \begin{array} { c c } { \frac { 9 } { 2 0 } } & { - \frac { 9 } { 1 0 \sqrt { 2 } } } \\ { - \frac { 1 } { 2 0 } } & { - \frac { 1 } { 1 0 \sqrt { 2 } } } \end{array} \right) , } & & { H _ { : 4 : } = \left( \begin{array} { c c } { - \frac { 1 } { \sqrt { 2 } } } & { 0 } \\ { 0 } & { 0 } \end{array} \right) . } \end{array}\tag{8}
$$

These matrices form the high-pass filter tensor $\varkappa = \left( H _ { : 1 : } \right.$ $H _ { : 2 : , } \ H _ { : 3 : , } \ H _ { : 4 : } )$ . The horizontal slices $\ b { H } _ { 1 : } ~ \in ~ \mathbb { R } ^ { 4 \times 2 }$ and $H _ { 2 : : } \in \mathbb { R } ^ { 4 \times 2 }$ correspond to the two wavelet functions of the GHM algorithm, denoted as $\psi _ { 1 } = H _ { 1 }$ <sub>::</sub> and $\psi _ { 2 } = H _ { 2 } ;$

In the GHM algorithm, L and H are used as convolution kernels to extract the low- and high-frequency components from the two-dimensional series X generated by the prefiltering step. Theoretically, the functions encoded in $\pmb { \mathcal { L } }$ and H possess compact support, orthogonality, and regularity, making GHM particularly well-suited for multiscale analysis of complex temporal signals [43].

![](images/a8285f4620de25898795f10480ce9a3840de6dc507df1eebc3da6bd640a42fd9.jpg)  
Fig. 2. Illustration of the m-WCN framework. Dec: Decomposition. Conv: Convolution. ↓ 2 denotes the downsampling operation with a sample rate of 2. ∗ is the convolutional operator.

Remark: Although the preset GHM parameters offer strong theoretical guarantees and desirable mathematical properties, their fixed nature limits flexibility when handling complex or heterogeneous data signals. In contrast, neural network-based parameter learning can adaptively adjust model parameters to better fit the data. To leverage the strengths of both approaches, we propose a novel architecture, i.e., m-WCN, that approximates the multi-wavelet decomposition process within a neural network framework. By learning transformation coefficients directly from the input series, m-WCN captures both time-domain patterns and frequency-domain components in an end-to-end trainable manner, enabling more adaptive time series analysis.

The m-WCN consists of two core modules: a Pre-filter Network, corresponding to the pre-filtering step of multiwavelet decomposition, and a Wavelet Decomposition Network, which mirrors the wavelet decomposition step. Fig. 2 illustrates the overall framework of m-WCN. In the following sections, we detail these modules.

## 4.2 Pre-filter Network of m-WCN

The Pre-filter Network is designed to neuralize the prefiltering phase of MWD, enabling the model to adaptively transform the input time series into a multidimensional feature series. Each dimension corresponds to a distinct temporal pattern, allowing the network to capture diverse structural variations embedded within the original signal.

Since the pre-filtering process resembles a convolution operation, we neuralize it using a convolutional neural network, incorporating the MWD kernel matrix P defined in Eq. (6). Specifically, given an input time series $\textbf { \textit { s } } = \mathbf { \kappa } \left( s _ { 1 } , \ldots , s _ { t } , \ldots , s _ { T } \right)$ and a convolution kernel $\begin{array} { r l } { p _ { : 1 } } & { { } = } \end{array}$ $\left( p _ { 1 1 } , \ldots , p _ { k 1 } , \ldots , p _ { K 1 } \right)$ (which corresponds to a row vector in the GHM projection matrix $P ) _ { \ I }$ , we define the convolution operation to compute the output series $\begin{array} { r l } { \pmb { x } } & { { } = } \end{array}$ $( x _ { 1 } , \dots , x _ { t } , \dots , x _ { T } )$ as:

$$
{ \pmb x } = { \pmb p } _ { : 1 } * { \pmb s } ,\tag{9}
$$

where ∗ denotes the convolution operator. Each element $x _ { t }$ in the output series x is computed as:

$$
x _ { t } = \sum _ { k = 1 } ^ { K } p _ { k 1 } \cdot s _ { t + k } .\tag{10}
$$

Recall the pre-filtering procedure in Eq. (2). The GHMbased multi-wavelet pre-filtering step can be reformulated in convolutional form as:

$$
{ \binom { { \pmb x } _ { : 1 } } { { \pmb x } _ { : 2 } } } = { \binom { p _ { : 1 } * s } { p _ { : 2 } * s } } ,\tag{11}
$$

where the convolution kernels $\pmb { p } _ { : 1 }$ and $\pmb { p } _ { : 2 }$ are the row vectors of the GHM pre-filter matrix $_ { P }$ in Eq. (6). This formulation closely mirrors the original pre-filtering step in GHM decomposition and naturally supports implementation via convolutional layers within neural networks.

To enhance adaptability and enable the network to better capture time-domain patterns specific to the input series $s ,$ we introduce a learnable parameter matrix $W ^ { \boldsymbol { p } } \in \mathbb { R } ^ { | \boldsymbol { P } | }$ and define the trainable kernel as:

$$
\tilde { \pmb { W } } = \pmb { P } + \pmb { W } ^ { p } ,\tag{12}
$$

i.e., each element is computed as $\tilde { w } _ { i j } = w _ { i j } ^ { p } + p _ { i j }$ . The output of the pre-filtering network in m-WCN is then given by:

$$
\begin{array} { r } { \pmb { X } = \left( \frac { \pmb { x } _ { : 1 } } { \pmb { x } _ { : 2 } } \right) = \sigma \left( \tilde { \pmb { w } } _ { : 1 } \ast \pmb { s } \right) , } \\ { \pmb { X } = \left( \frac { \pmb { x } _ { : 2 } } { \pmb { x } _ { : 2 } } \right) = \sigma \left( \tilde { \pmb { w } } _ { : 2 } \ast \pmb { s } \right) , } \end{array}\tag{13}
$$

where $\sigma ( \cdot )$ denotes the sigmoid activation function. This non-linear activation is introduced to capture complex dependencies within the time series. The design in Eq. (12) and Eq. (13) preserves the theoretical structure of the GHM formulation while introducing learnable flexibility, enabling m-WCN to adapt effectively through end-to-end training.

## 4.3 Wavelet Decomposition Network of m-WCN

In this module, we neuralize the decomposition process of m-WCN, enabling the multivariate series X in Eq. (13) to be transformed into distinct frequency components in a learnable and data-adaptive manner.

We define a 2-dimensional convolution operator for multivariate time series. Given an input series $\mathbf { \bar { X } } \in \mathbb { R } ^ { N \times T }$ with N features and $T$ time steps, and a kernel matrix $\pmb { K } ~ \in ~ \mathbb { R } ^ { M \times N }$ , the 2-dimensional convolution operation ∗ produces an output series as:

$$
y = K * X ,\tag{14}
$$

where $\pmb { y } = ( y _ { 1 } , \dots , y _ { T } )$ . The t-th item $y _ { t }$ is computed as:

$$
y _ { t } = \sum _ { m = 1 } ^ { M } k _ { m : } \cdot x _ { : , t + m } ,\tag{15}
$$

where $k _ { m } .$ is the m-th row of the kernel matrix K, and $\mathbf { \boldsymbol { x } } _ { : , t + m }$ denotes the $( t + m )$ -th column vector of the input series $\boldsymbol { x }$

Furthermore, we extend the convolution kernel into a third-order tensor $\boldsymbol { \kappa } ~ \in ~ \mathbb { R } ^ { M \times N \times K }$ . The convolution then produces a multivariate output series:

$$
\begin{array} { r } { \pmb { Y } = \left( \begin{array} { c } { \pmb { y } _ { 1 : } } \\ { \vdots } \\ { \vdots } \\ { \pmb { y } _ { K : } } \end{array} \right) = \pmb { K } \ast \pmb { X } = \left( \begin{array} { c } { \pmb { K } _ { 1 : : } \ast \pmb { X } } \\ { \vdots } \\ { \pmb { K } _ { K : : } \ast \pmb { X } } \end{array} \right) , } \end{array}\tag{16}
$$

where $\pmb { K } _ { k : }$ denotes the k-th horizontal slice of the tensor $\kappa ,$ and each $\mathbf { \nabla } \mathbf { \textbf {  { y } } } _ { k : \mathbf { \nabla } }$ is the corresponding output feature series.

For the GHM algorithm, the wavelet decomposition in Eq. (3) can be equivalently expressed as a 2-dimensional

convolution operation with tensor kernels, following the formulation in Eq. (16), as follows:

$$
\begin{array} { r } { \pmb { X } ^ { h } = \binom { \pmb { x } _ { 1 : } ^ { h } } { \pmb { x } _ { 2 : } ^ { h } } = \pmb { \mathcal { H } } \ast \pmb { X } = \left( \pmb { H } _ { 1 : : } \ast \pmb { X } \right) , } \\ { \pmb { X } ^ { l } = \binom { \pmb { x } _ { 1 : } ^ { l } } { \pmb { x } _ { 2 : } ^ { l } } = \pmb { \mathcal { L } } \ast \pmb { X } = \left( \pmb { L } _ { 2 : } \ast \pmb { X } \right) , } \end{array}\tag{17}
$$

where the kernel tensors $\pmb { \mathcal { L } }$ and H are defined in Eq. (7) and Eq. (8), respectively.

In m-WCN, the fixed wavelet and scaling tensors H and L are enhanced with two learnable parameter tensors $w ^ { h } \in$ $\mathbb { R } ^ { | \mathcal { H } | }$ and ${ \boldsymbol w ^ { l } \in \mathbb R ^ { | \boldsymbol L | } }$ as follows:

$$
\tilde { \pmb { \mathscr { H } } } = \pmb { \mathscr { W } } ^ { h } + \pmb { \mathscr { H } } , \qquad \tilde { \pmb { \mathscr { L } } } = \pmb { \mathscr { W } } ^ { l } + \pmb { \mathscr { L } } .\tag{18}
$$

This formulation allows m-WCN to adaptively adjust the wavelet and scaling functions based on the characteristics of the input data. By augmenting the traditional GHM filters with learnable parameters, the model gains greater flexibility and adaptability, leading to improved frequency component extraction tailored to diverse time series.

Using the learnable parameters defined in Eq. (18), m-WCN recursively extracts high- and low-frequency components of the input series as follows:

$$
\begin{array} { r } { { \pmb X } ^ { h _ { n } } = \sigma \left( ( \tilde { \pmb { \mathscr { H } } } * { \pmb X } ^ { l _ { n - 1 } } ) \downarrow 2 \right) , } \\ { { \pmb X } ^ { l _ { n } } = \sigma \left( ( \tilde { \pmb { C } } * { \pmb X } ^ { l _ { n - 1 } } ) \downarrow 2 \right) , } \end{array}\tag{19}
$$

where $\sigma ( \cdot )$ denotes the sigmoid activation function. The use of $\sigma ( \cdot )$ enables the model to capture complex, nonlinear relationships within the frequency domain, thereby enhancing its capacity to model intricate temporal patterns across multiple decomposition layers.

Similar to the multi-wavelet decomposition in Sec. 3.3, the output of m-WCN with N layers is a set of frequency components:

$$
\pmb { \mathcal { X } } = \left\{ \pmb { X } ^ { h _ { 1 } } , \pmb { X } ^ { h _ { 2 } } , \dots , \pmb { X } ^ { h _ { N } } , \pmb { X } ^ { l _ { N } } \right\} ,\tag{20}
$$

where $X ^ { h _ { 1 } } \in \mathbb { R } ^ { 2 \times ( T / 2 ) }$ represents the highest-frequency component, and $\pmb { X } ^ { l _ { N } } \in \mathrm { ~ \partial ~ } \mathbb { R } ^ { 2 \times ( T / 2 ^ { N } ) }$ corresponds to the lowest-frequency component. This set X constitutes the final time-frequency features extracted by m-WCN.

Remark: To sum up, m-WCN uses a deep neural network framework to approximately implement the GHM multi-wavelet decomposition. Achieving the GHM algorithm within such a framework offers several advantages. First, the parameters in GHM can be fine-tuned via backpropagation with task-specific loss functions, enabling the extracted frequency features to carry more relevant information than those obtained from the traditional GHM method. Second, the output series produced by m-WCN can be fed into downstream neural networks for further analysis, facilitating end-to-end training of the entire pipeline (see Sec. 5) and thereby improving overall model performance.

## 4.4 Orthogonality Regularization

In this part, we first introduce the orthogonality property of MWD, and then design an orthogonality regularization term to constrain the convolution kernels of m-WCN to align with the orthogonality property of MWD.

MWD possesses properties such as orthogonality, tight� <sup>Level</sup> <sup>1</sup> support, and regularity. Among these, orthogonality servesConcat � Classiferk as the foundation, ensuring that the input series can bew �(1) decomposed and reconstructed completely, stably, and with-��(�)a<sup>m</sup> out redundancy. The other properties build upon this foun-��(�)N dation to further enhance performance and facilitate practi-<sup>D</sup> cal applications. Therefore, this paper focuses on preserving� (�)M �(3) the most critical property of MWD: orthogonality. Its de-� (�) �(3) tailed introduction is given in Sec. 5 of the Supplementary Materials (SM).

To satisfy the orthogonality requirement in the proposed m-WCN framework, we must ensure that the corresponding convolutional kernels $\tilde { \mathcal { H } }$ and L<sup>˜</sup> are as orthonormal as possible. Specifically, given $\tilde { H } _ { k : : }$ and $\tilde { L } _ { k : : \prime }$ which denote the k-th horizontal slices of the tensors H<sup>˜</sup> and $\tilde { \pmb { { c } } }$ respectively, we expect that

$$
\begin{array} { r l } & { \tilde { H } _ { 1 : : } \cdot \tilde { H } _ { 1 : : } ^ { \top } = \tilde { H } _ { 2 : : } \cdot \tilde { H } _ { 2 : } ^ { \top } = I , \tilde { H } _ { 1 : } \cdot \tilde { H } _ { 2 : } ^ { \top } = \tilde { H } _ { 2 : : } \cdot \tilde { H } _ { 1 : } ^ { \top } = { \bf 0 } , } \\ & { \tilde { L } _ { 1 : : } \cdot \tilde { L } _ { 1 : } ^ { \top } = \tilde { L } _ { 2 : : } \cdot \tilde { L } _ { 2 : } ^ { \top } = I , \tilde { L } _ { 1 : } \cdot \tilde { L } _ { 2 : } ^ { \top } = \tilde { L } _ { 2 : } \cdot \tilde { L } _ { 1 : } ^ { \top } = { \bf 0 } , } \\ & { \tilde { H } _ { 1 : : } \cdot \tilde { L } _ { 1 : } ^ { \top } = \tilde { H } _ { 2 : } \cdot \tilde { L } _ { 2 : } ^ { \top } = { \bf 0 } , \tilde { H } _ { 1 : } \cdot \tilde { L } _ { 2 : } ^ { \top } = \tilde { H } _ { 2 : : } \cdot \tilde { L } _ { 1 : } ^ { \top } = { \bf 0 } , } \\ & { \tilde { L } _ { 1 : } \cdot \tilde { H } _ { 1 : } ^ { \top } = \tilde { L } _ { 2 : } \cdot \tilde { H } _ { 2 : } ^ { \top } = I , \tilde { L } _ { 1 : } \cdot \tilde { H } _ { 2 : } ^ { \top } = \tilde { L } _ { 2 : } \cdot \tilde { H } _ { 1 : } ^ { \top } = { \bf 0 } , } \end{array}\tag{21}
$$

where I is the identity matrix and 0 is a zero matrix. Eq. (21) indicates that: i) each convolutional kernel forms a roworthonormal matrix, and ii) each convolutional kernel is orthogonal to every other convolutional kernel.

Each condition in Eq. (21) is supposed to be satisfied. However, directly optimizing all these terms would introduce numerous optimization objectives, leading to a computationally expensive and inefficient process, and potentially causing unstable training. To address this, this paper adopts the block-Toeplitz (BT) method [44] to efficiently represent and handle large numbers of 2D convolution kernels. Specifically, a Toeplitz matrix is one where elements are constant along each diagonal. A Block-Toeplitz matrix generalizes this concept: instead of scalars, the elements are small matrices (blocks), and these blocks remain constant along each diagonal, similar to the structure of a standard Toeplitz matrix. Given the convolution kernels of m-WCN, they can be organized into a BT matrix B as shown below:

$$
\begin{array} { r } { \pmb { B } ^ { \top } = \left( \begin{array} { c c c c c c c c c c c c } { \tilde { H } _ { 1 : : : } } & { 0 } & { \tilde { H } _ { 2 : : } } & { 0 } & { \tilde { L } _ { 1 : : } } & { 0 } & { \tilde { L } _ { 2 : : } } & { 0 } \\ { 0 } & { \tilde { H } _ { 1 : : } } & { 0 } & { \tilde { H } _ { 2 : : } } & { 0 } & { \tilde { L } _ { 1 : } } & { 0 } & { \tilde { L } _ { 2 : } } \end{array} \right) } \end{array}\tag{22}
$$

Based on $B ,$ the optimization objective for kernel orthogonality in Eq. (21) can be reformulated as

$$
\begin{array} { r } { \mathcal { L } _ { o } = \| \boldsymbol { B } \cdot \boldsymbol { B } ^ { \intercal } - \boldsymbol { I } \| _ { F } ^ { 2 } , } \end{array}\tag{23}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. With the BT method, the orthogonality regularization is formulated as a single matrix multiplication operation. This allows deep learning libraries to leverage parallelization to accelerate computation, thereby ensuring computational efficiency.

Remark: The orthogonality regularization in this part offers several advantages. First, when filters (kernels) are learned to be as orthogonal as possible, they become decorrelated, resulting in filter responses that are significantly less redundant. Moreover, the orthonormal kernel matrix helps gradients back-propagate stably, thus preventing gradient explosion and gradient vanishing. This is because multiplying a matrix by an orthonormal matrix preserves its norm.

![](images/c72471e48bf69bf941905ebcae5b5d22b5f2669f84b438b011d38b47cb9efcff.jpg)  
Fig. 3. The TFBC network for classification: A three-level example.

## 5 M-WCN-BASED TIME SERIES ANALYSIS

In this section, we propose two m-WCN-based network architectures to leverage the capability of m-WCN for different downstream tasks, including time series classification (TSC) and time series forecasting (TSF).

## 5.1 m-WCN-based Time Series Classification

The time series classification (TSC) task aims to predict the category label of a time series based on its temporal features. For instance, diagnosing whether a patient has heart disease can be formulated as a classification problem using electrocardiogram (ECG) signal sequences as input. A key challenge in TSC lies in effectively extracting discriminative features from the input time series. m-WCN provides a powerful solution by jointly capturing both temporal structures and frequency-domain characteristics of time series through its time-frequency decomposition framework. Building on this capability, we propose a Time-Frequency Boosting Classification (TFBC) network, which exploits the rich representations in the frequency components extracted by m-WCN layer by layer. To further enhance the discriminative power of the frequency features, we introduce a Frequency Contrastive Learning (FCL) loss, enabling the network to be trained in a pre-training followed by fine-tuning paradigm. This design encourages the model to learn generalized and class-discriminative representations that are robust across diverse TSC scenarios.

## 5.1.1 Time-Frequency Boosting Classification

In m-WCN, the extracted frequency components represent complementary aspects of the input signal. While each component carries a certain degree of discriminative power, the low-frequency components typically encode more informative features for classification tasks than high-frequency components in wavelet decompositions [1]. Motivated by this insight, the TFBC network adopts a boosting strategy to exploit the hierarchical relationship between low- and high-frequency components. The classification process begins with the lowest-frequency component to generate an initial prediction. Higher-frequency components are then introduced sequentially to refine this prediction by learning to correct the residual errors from earlier stages. This hierarchical boosting mechanism enables the model to integrate both stable global patterns and subtle local variations in the signal. Fig. 3 illustrates the architecture of TFBC.

Given the lowest-frequency component in Eq. (20), i.e., $X ^ { l _ { N } }$ , the TFBC network uses an encoder Enc(·) to transform it into a representation vector:

$$
{ \pmb g } ^ { l _ { N } } = \mathrm { E n c } \left( { \pmb X } ^ { l _ { N } } ; { \pmb \Theta } _ { e } ^ { l _ { N } } \right) ,\tag{24}
$$

where $\Theta _ { e } ^ { N }$ denotes the trainable parameters of the encoder, and $\pmb { g } ^ { l _ { N } }$ is the resulting representation vector. We implement Enc(·) using TimesNet [45], due to its effectiveness in feature extraction and transformation.

Subsequently, the TFBC network applies a Softmax classifier to predict the class label from $\pmb { g } ^ { l _ { N } }$

$$
\hat { \pmb { c } } ^ { l _ { N } } = \mathrm { S o f t M a x } \left( \pmb { g } ^ { l _ { N } } \right) ,\tag{25}
$$

where $\hat { \pmb { c } } ^ { l _ { N } }$ is the probability vector representing the prediction for the one-hot class label. The operations defined in Eq. (24) and Eq. (25) together constitute the N-th layer of the TFBC network.

For the n-th high-frequency components, where $n \in$ $\{ 0 , \ldots , N - 1 \}$ , the TFBC network applies an encoder to generate the representation vector for the high-frequency component $X ^ { h _ { n } ^ { - } }$ extracted from the n-th decomposition layer of m-WCN:

$$
\pmb { g } ^ { h _ { n } } = \operatorname { E n c } \left( \pmb { X } ^ { h _ { n } } ; \pmb { \Theta } _ { e } ^ { h _ { n } } \right) ,\tag{26}
$$

Then, the class label prediction is generated using a residual formulation:

$$
\hat { \pmb { c } } ^ { h _ { n } } = \hat { \pmb { c } } ^ { h _ { n + 1 } } + \mathrm { S o f t M a x } \left( \pmb { g } ^ { h _ { n } } \right) .\tag{27}
$$

Intuitively, Eq. (27) uses $\pmb { g } ^ { h _ { n } }$ to estimate the residual error between the current prediction and the one from the higher layer, $\mathrm { i . e . , ~ } \hat { c } ^ { h _ { n } } - \hat { c } ^ { h _ { n + 1 } } = \mathrm { S o f t M a x } \left( g ^ { h _ { n } } \right)$ . In this way, the representation vector $\pmb { g } ^ { h _ { n } }$ learns to compensate for what the (n + 1)-th layer fails to model, forming a boosting-style learning mechanism that progressively refines the prediction across layers.

Finally, the final class label prediction is given by the first layer. According to the recursive expression in Eq. (27), the final label prediction is calculated as an ensemble form of the predictions based on the lowest frequency component and high-frequency components, i.e.,

$$
\hat { \pmb { c } } = \frac { 1 } { N + 1 } \left( \mathrm { S o f t M a x } \left( \pmb { g } ^ { l _ { N } } \right) + \sum _ { n = 1 } ^ { N } \mathrm { S o f t M a x } \left( \pmb { g } ^ { h _ { n } } \right) \right) .\tag{28}
$$

Remark: In the TFBC model, the frequency components extracted from all levels of m-WCN are utilized to generate an ensemble class prediction. As these components reflect different frequency resolutions [8], TFBC effectively captures diverse perspectives of the input time series by aggregating information across multiple spectral scales. This design enables TFBC to operate as a multiview learning framework, enhancing its ability to achieve high-performance time series classification. Moreover, the classifier at the n-th level refines its prediction $\hat { c } ^ { n }$ based on both its own representation $\pmb { g } ^ { n }$ and the output from the $( n + 1 ) ‐ \mathrm { t h } \mathrm { l e v e l } , \dot { \hat { \mathbf { c } } } ^ { h _ { n + 1 } }$ . This residual formulation allows the model to incrementally integrate knowledge that is not captured by higher-frequency components alone, thereby improving the overall classification accuracy through progressive refinement.

## 5.1.2 Frequency Contrastive Learning

In the TFBC model, each frequency component is encoded into a representation vector $\pmb { g } ^ { n }$ . To enhance the discriminative capacity of these representations, we introduce a selfsupervised Frequency Contrastive Learning (FCL) strategy to pre-train the parameters of TFBC. This approach encourages the model to learn semantically meaningful frequencyrelated embeddings.

In TFBC, the representations $\pmb { g } ^ { n }$ obtained at different decomposition levels describe the same input signal from multiple frequency perspectives. These multi-level representations can thus be regarded as natural augmentations of the original signal. Based on this intuition, we treat pairs of representations $\pmb { g } ^ { n }$ and $\pmb { g } ^ { m }$ extracted from the same time series instance as positive samples, while representations from different time series are treated as negative samples.

Specifically, for an input time series $\scriptstyle { \pmb { s } } _ { i } ,$ we obtain N+1 representation vectors from different frequency components, denoted as $\mathbf { \Delta } _ { \pmb { g } _ { i } ^ { \pi } }$ , where $\pi ~ \in ~ \Pi ~ = ~ \left\{ h _ { 1 } , \dots , h _ { N } , l _ { N } \right\}$ denotes the index of a frequency component. To facilitate contrastive learning, a nonlinear projection head $g$ is used to map each level-specific representation into a latent space: ${ \pmb v } _ { i } ^ { \pi } \ = \ g ( { \pmb g } _ { i } ^ { \pi } )$ . In this latent space, we define pairwise contrastive objectives using the InfoNCE loss. For two representations $\pmb { v } _ { i } ^ { \pi _ { 1 } }$ and $\pmb { v } _ { i } ^ { \pi _ { 2 } }$ derived from different frequency levels of the same input series sample $\boldsymbol { S } _ { i } ,$ the contrastive loss is computed as:

$$
\begin{array} { r l r } {  { \mathcal L _ { i } ( \boldsymbol \pi _ { 1 } , \boldsymbol \pi _ { 2 } ) = - \log \frac { h ( \boldsymbol v _ { i } ^ { \pi _ { 1 } } , \boldsymbol v _ { i } ^ { \pi _ { 2 } } ) } { h ( \boldsymbol v _ { i } ^ { \pi _ { 1 } } , \boldsymbol v _ { i } ^ { \pi _ { 2 } } ) + \mathrm { N P } } , } } \\ & { } & { \mathrm { w h e r e \quad N P } = \sum _ { j \ne i } h ( \boldsymbol v _ { i } ^ { \pi _ { 1 } } , \boldsymbol v _ { j } ^ { \pi _ { 1 } } ) + \sum _ { j \ne i } h ( \boldsymbol v _ { i } ^ { \pi _ { 2 } } , \boldsymbol v _ { j } ^ { \pi _ { 2 } } ) . } \end{array}\tag{29}
$$

Here, $h ( { \pmb v } _ { a } , { \pmb v } _ { b } ) = \mathrm { e x p } ( \mathrm { s i m } ( { \pmb v } _ { a } , { \pmb v } _ { b } ) / \tau )$ , where sim(·, ·) denotes cosine similarity and τ is a temperature parameter controlling the sharpness of similarity scores. The projection head $g$ is implemented as a two-layer MLP.

The final frequency contrastive loss across a batch of I samples is defined as:

$$
\mathcal { L } _ { c } = \frac { 1 } { H } \sum _ { i = 1 } ^ { I } \sum _ { \pi _ { 1 } \in \Pi } \sum _ { \pi _ { 2 } \neq \pi _ { 1 } } \mathcal { L } _ { i } ( \pi _ { 1 } , \pi _ { 2 } ) ,\tag{30}
$$

where $\Pi = \left\{ h _ { 1 } , \ldots , h _ { N } , l _ { N } \right\}$ , and $H = I \times N \times ( N + 1 )$ is the total number of contrastive pairs in the batch. This framework encourages the model to pull together representations of the same input across different frequency components while pushing apart representations from different inputs. As a result, the model learns semantically rich, frequencyrelated embeddings that provide strong initialization for downstream time series classification tasks via fine-tuning.

## 5.1.3 Optimization for TFBC

For the final classification loss of TFBC, we adopt a layerwise supervision strategy to enhance training effectiveness. Specifically, given a set of I input time series samples, we define $\hat { c } _ { i } ^ { \pi }$ as the predicted class label generated by Eq. (27) for the i-th sample, where the superscript π denotes the corresponding frequency component (either $h _ { n }$ or $l _ { N } )$ . The cross-entropy loss for $\hat { c } _ { i } ^ { \pi }$ is defined as:

$$
\tilde { \mathcal { L } } _ { e } ( \pi ) = - \frac { 1 } { I } \sum _ { i = 1 } ^ { I } \left( \pmb { c } _ { i } ^ { \top } \ln \hat { \pmb { c } } _ { i } ^ { \pi } + ( 1 - \pmb { c } _ { i } ) ^ { \top } \ln \left( 1 - \hat { \pmb { c } } _ { i } ^ { \pi } \right) \right) ,\tag{31}
$$

where $\mathbf { c } _ { i }$ denotes the one-hot encoded ground truth label for the i-th sample.

![](images/9ec2b1d1c6dc4c36b44e9fb765a856615d56ec2c98e69ab5907a7657d2b906c4.jpg)  
Fig. 4. The FTB model for forecasting: A three-level example. FRP: Frequency Representation Pre-training.

For an TFBC model with N m-WCN frequency decomposition layers, the overall cross-entropy objective is defined as a weighted sum of all individual layer losses $\tilde { \mathcal { L } } _ { e } ( \pi )$

$$
\mathcal { L } _ { e } = \sum _ { n = 1 } ^ { N } \frac { N - n + 1 } { N } \tilde { \mathcal { L } } _ { e } ( h _ { n } ) + \frac { 1 } { N } \tilde { \mathcal { L } } _ { e } ( l _ { N } ) .\tag{32}
$$

This weighting scheme ensures that the final prediction $\hat { \pmb { c } } = \hat { \pmb { c } } ^ { h _ { 1 } } - \check { \pmb { \mathit { \tau } } }$ which integrates information from all frequency components — receives the highest weight of 1, while the prediction $\hat { \pmb { c } } ^ { l _ { N } }$ — based solely on the lowest-frequency component — receives the smallest weight of $1 / N .$ . This design prioritizes the supervisory signal for the most comprehensive prediction while still guiding the learning process at earlier layers.

The overall objective for TSC is derived by integrating the classification loss in $\operatorname { E q } .$ (32), the contrastive loss in Eq. (30), and the orthogonal regularization in Eq. (23) as

$$
\mathcal { L } _ { \mathrm { T S C } } = \mathcal { L } _ { e } + \gamma _ { 1 } \mathcal { L } _ { c } + \gamma _ { 2 } \mathcal { L } _ { o } ,\tag{33}
$$

where $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are hyperparameters that balance the contributions of different losses.

## 5.2 m-WCN-based Time Series Forecasting

The time series forecasting (TSF) task aims to predict future values of a time series based on its historical observations, typically following an autoregressive paradigm. Unlike time series classification (TSC), where class labels may be associated with any frequency component, TSF tasks often exhibit frequency-aligned dependencies — meaning that each frequency component in the future series tends to correlate most strongly with the corresponding frequency component in the historical series. Motivated by this insight, we propose a m-WCN-based Frequency TSMixer Bagging framework, abbreviated as FTB, for time series forecasting. To support training of FTB, we further introduce a Frequency Representation Pre-training (FRP) strategy to enhance frequencyaware learning. The overall architecture of the proposed forecasting model is illustrated in Fig. 4.

## 5.2.1 Bagging of TSMixer for Forecasting

In the FTB network, we decompose a complex TSF task into multiple sub-problems, each corresponding to forecasting a specific frequency component extracted by m-WCN. Since each frequency component captures a relatively simpler temporal pattern — either long-term trends or short-term fluctuations — these sub-problems are easier to model individually. To handle them effectively, FTB employs separate TSMixer-based predictors for each frequency component.

The predictions from these parallel sub-models are then aggregated to produce the final forecasting result, enabling the model to capture both coarse-grained and fine-grained temporal dynamics in a frequency-aware manner.

Specifically, given an input time series of infinite length, we apply a sliding window of size $T + 1$ over the past to the current time step $t ,$ resulting in the input segment:

$$
\pmb { \mathscr { s } } _ { t } = \left( \mathscr { s } _ { t - T } , \mathscr { . . . } , \mathscr { s } _ { t - 1 } , \mathscr { s } _ { t } \right) .\tag{34}
$$

In the TSF setting, the goal is to predict the future segment of the series starting from time t+1 over a prediction horizon of length $L ,$ defined as:

$$
z _ { t } = ( s _ { t + 1 } , \ldots , s _ { t + l } , \ldots , s _ { t + L } ) .\tag{35}
$$

The FTB model first applies m-WCN to decompose the input series $\mathbf { \Delta } _ { \mathbf { \mathcal { S } } _ { t } }$ into a set of frequency components: $\mathbf { \dot { \mathbf { X } } } _ { t } ^ { h _ { 1 } } , \ldots , \mathbf { X } _ { t } ^ { h _ { N } } , \mathbf { X } _ { t } ^ { l _ { N } } .$ , capturing both high- and lowfrequency patterns. Each component is then processed independently using a TSMixer network to obtain its frequencyspecific representation sequence:

$$
\begin{array} { r l } & { \pmb { { E } } _ { t } ^ { h _ { n } } = \left( e _ { t - \frac { T } { 2 ^ { n + 1 } } } ^ { h _ { n } } , \ldots , e _ { t } ^ { h _ { n } } \right) = \mathrm { T F } \left( \pmb { X } _ { t } ^ { h _ { n } } ; \pmb { \Theta } ^ { h _ { n } } \right) , } \\ & { \pmb { { E } } _ { t } ^ { l _ { N } } = \left( e _ { t - \frac { T } { 2 ^ { N + 1 } } } ^ { l _ { N } } , \ldots , e _ { t } ^ { l _ { N } } \right) = \mathrm { T F } \left( \pmb { X } _ { t } ^ { l _ { N } } ; \pmb { \Theta } ^ { l _ { N } } \right) , } \end{array}\tag{36}
$$

where $\mathrm { T F } ( \cdot )$ denotes the TSMixer encoder, and $\Theta ^ { h _ { r } }$ and $\Theta ^ { l _ { N } }$ are the learnable parameters for the n-th highfrequency and final low-frequency predictors, respectively.

Then, the FTB model applies an attention mechanism to integrate the representation sequences from all frequency components:

$$
U = \mathrm { A T T } \left( E _ { t } ^ { h _ { 1 } } \big | \big | \dots \big | \big | E _ { t } ^ { h _ { N } } \big | \big | E _ { t } ^ { l _ { N } } \right) ,\tag{37}
$$

where $\mathrm { A T T } ( \cdot )$ denotes the attention network and ∥ indicates concatenation along the temporal dimension. This step captures the inter-frequency dependencies and synthesizes a unified context representation $U$

Finally, a multi-layer perceptron (MLP) is used to generate the forecasted future sequence:

$$
{ \hat { z } } _ { t } = \operatorname { M L P } ( U ) ,\tag{38}
$$

where $\hat { z } _ { t } = ( \hat { s } _ { t + 1 } , \ldots , \hat { s } _ { t + l } , \ldots , \hat { s } _ { t + L } )$ is the predicted future segment corresponding to $z _ { t + 1 }$ . Each $\hat { s } _ { t + \cdot }$ <sub>l</sub> represents the predicted value at time step $t + l ,$ , completing the end-toend forecasting pipeline.

Remark: In the TSF task, different frequency components reflect temporal dynamics at varying scales, i.e., low frequency captures long-term trends, while high frequency encodes short-term variations. All these components are equally essential for accurate forecasting. To effectively utilize this multi-scale information, the FTB model adopts a bagging approach: each frequency-specific representation independently contributes to the prediction of the future series. This design enhances the model’s ability to capture rich temporal dependencies across all spectral bands.

## 5.2.2 Frequency Representation Pre-training

To improve the quality and forecasting relevance of the frequency-specific representations in Eq. (36), we propose a Frequency Representation Pre-training (FRP) strategy. This approach aims to enhance each frequency component’s ability to capture predictive information by encouraging its representation to reconstruct the corresponding component in the future series.

Specifically, given the future time series segment ${ \boldsymbol { z } } _ { t }$ defined in Eq. (35), we apply GHM multi-wavelet decomposition to extract its frequency components: $\begin{array} { r l } { \mathcal { \mathrm { ) } } _ { t } } & { { } = } \end{array}$ $\{ Y _ { t } ^ { h _ { 1 } } , \ldots , Y _ { t } ^ { h _ { N } } , Y _ { t } ^ { l _ { N } } \}$ . Each component $Y _ { t } ^ { h _ { n } ^ { * } }$ serves as the prediction label for pre-training the representation $E _ { t } ^ { h _ { n } }$ generated in Eq. (36). Next, we employ an MLP network to predict each future frequency component from its corresponding past representation:

$$
\hat { Y } _ { t } ^ { h _ { n } } = \mathrm { M L P } \left( E _ { t } ^ { h _ { n } } ; \boldsymbol { \Theta } _ { p } ^ { h _ { n } } \right) ,\tag{39}
$$

where $\Theta _ { p } ^ { h _ { n } }$ are learnable parameters for the n-th predictor. The pre-training loss for time step t is formulated as:

$$
\mathcal { L } _ { t } = \sum _ { n = 1 } ^ { N } \frac { 1 } { D ^ { n } } \left. \pmb { Y } _ { t } ^ { h _ { n } } - \hat { \pmb { Y } } _ { t } ^ { h _ { n } } \right. _ { F } ^ { 2 } + \frac { 1 } { D ^ { N } } \left. \pmb { Y } _ { t } ^ { l _ { N } } - \hat { \pmb { Y } } _ { t } ^ { l _ { N } } \right. _ { F } ^ { 2 } ,\tag{40}
$$

where $D ^ { n }$ is the dimensionality of the n-th frequency component, used to normalize losses across different scales.

Over M forecast training samples, the total pre-training objective is defined as:

$$
\mathcal { L } _ { p } = \sum _ { t = 1 } ^ { M } \mathcal { L } _ { t } + \gamma _ { 3 } \mathcal { L } _ { o } ,\tag{41}
$$

where $\gamma _ { 3 }$ is a hyperparameter and $\mathcal { L } _ { o }$ denotes the orthogonality regularization term introduced in Eq. (23).

## 5.2.3 Optimization for FTB

In the fine-tuning stage, the FTB model is initialized with parameters learned during the pre-training phase. The entire model is then trained in an end-to-end manner using the mean square error (MSE) loss to minimize the discrepancy between the predicted and ground-truth future sequences. Specifically, given M time series forecasting samples, the fine-tuning objective is defined as:

$$
\mathcal { L } _ { \mathrm { T S F } } = \frac { 1 } { M } \sum _ { t = 1 } ^ { M } \| z _ { t } - \hat { z } _ { t } \| _ { 2 } ^ { 2 } ,\tag{42}
$$

where $\hat { z } _ { t }$ is the predicted sequence generated by Eq. (38), and ${ \boldsymbol { z } } _ { t }$ is the ground truth future segment.

## 5.3 Complexity Discussion

Complexity Discussion. Let T denote the input length and N denote the decomposition depth. The additional computational cost introduced by m-WCN mainly comes from two modules: the pre-filter network and the wavelet decomposition network. In our GHM-based implementation, both the convolutional kernel sizes and the number of channels are fixed constants. Therefore, the complexity of m-WCN grows linearly with the input length.

• Time complexity. The pre-filter network scans the input sequence once using fixed-size convolutional filters, leading to a time complexity of $O ( T )$ . For the wavelet decomposition network, the sequence length is downsampled by a factor of 2 after each decomposition layer. Thus, the input length to the n-th decomposition layer is $T / 2 ^ { n - 1 }$ . Since each layer applies fixed-size convolutional filters with a fixed number of channels, the total computational cost over all N decomposition layers is $\begin{array} { r } { \sum _ { n = 1 } ^ { N } \dot { O } ( T / 2 ^ { n - 1 } ) = } \end{array}$ $O ( T )$ . Therefore, the overall additional time complexity introduced by m-WCN is linear in T.

• Space complexity. The space complexity is also $O ( T )$ . The final retained frequency components consist of the highfrequency components from all decomposition layers and the final low-frequency component. Their total sequence length is bounded by $\begin{array} { r } { \sum _ { n = 1 } ^ { N } \hat { T } / 2 ^ { n } + T / 2 ^ { N } \le T , } \end{array}$ , up to constant factors determined by the fixed channel number. During training, storing intermediate activations for backpropagation also leads to a geometric sum over sequence lengths and therefore remains linear in T.

In summary, m-WCN introduces only linear additional time and memory overhead with respect to the input length. This indicates that the proposed decomposition module is computationally efficient and can be integrated into downstream classification and forecasting models without causing excessive complexity.

## 6 EXPERIMENTAL RESULTS

## 6.1 Task I: Time Series Classification

## 6.1.1 Datasets

We evaluate our model on the UCR TSC archive 2018 [46], which is a commonly used large-scale univariate time series dataset covering the areas of image contour classification, motion classification, ECG classification, sensor data classification, and others. The UCR archive is designed to provide a standard and comprehensive archive for univariate TSC, which has split the data into training and test set. Due to space limitations, we select 64 representative datasets from the UCR archive in our experiments. The dataset selection follows two principles. First, the selected representative datasets are intended to provide a more challenging and discriminative benchmark for evaluating different methods. Many UCR datasets have been extensively studied, and their performance is nearly saturated. That is, many algorithms achieve close-to-zero error rates on these datasets. We therefore excluded these nearly saturated datasets from the representative subset. Second, the selected subset covers all major UCR data categories used in our experiments, including Device, ECG, Image, Motion, Sensor, Simulated, and Spectro datasets. In addition, we also provide the experimental results on all 128 UCR datasets in Sec. 3 of SM.

## 6.1.2 Evaluation Metric and Baselines

We choose the error rate to evaluate the classification performance. We utilize seven baselines for comparison, including a feature-based approach, two ensemble methods, three deep learning models, and a wavelet-based deep model.

• miniRocket [47]: An advanced model that utilizes convolution kernels to extract features from time series to improve classification outcomes. Using a ridge regression classifier, miniRocket demonstrates superior performance compared to traditional techniques in TSC. We chose it as the exemplar baseline for feature-based approaches.

TABLE 2  
Classification performance comparison on 64 UCR time series datasets regarding error rate. The Dataset column is in the form of “Abbr/Type ID”, where the dataset’s full name and corresponding type are in Sec. 1.1 of SM. The best result is bold, while the second-best result is underlined.
<table><tr><td colspan="10">Dataset|miniRocket Hivecote2 TS-Chief OS-CNN TimeURL MILLET mWDN TFBC|</td><td colspan="10">Dataset|miniRocket Hivecote2 TS-Chief OS-CNN TimeURL MILLET mWDN TFBC</td></tr><tr><td>Comput/1</td><td>0.268</td><td>0.240</td><td>0.300</td><td>0.293</td><td>0.264</td><td>0.240</td><td>0.360</td><td>0.220</td><td></td><td>CrkX/4| 0.179</td><td></td><td>0.172</td><td>0.179</td><td>0.145 0.133</td><td>0.267</td><td>0.154</td><td></td><td>0.216 0.168</td></tr><tr><td>ElecD/1</td><td>0.258</td><td>0.274</td><td>0.284</td><td>0.276</td><td>0.273</td><td>0.271</td><td>0.342</td><td>0.235</td><td>CrkY/4</td><td>0.172</td><td>0.154</td><td>0.200</td><td></td><td></td><td>0.264</td><td>0.149</td><td>0.172</td><td>0.141</td></tr><tr><td>LrgKA/1</td><td>0.125</td><td>0.080</td><td>0.232</td><td>0.104</td><td>0.087</td><td>0.096</td><td>0.152</td><td>0.067</td><td>CrkZ/4</td><td>0.172</td><td>0.141</td><td>0.169</td><td></td><td>0.137</td><td>0.254</td><td>0.136</td><td>0.162</td><td>0.169</td></tr><tr><td>RefrD/1</td><td>0.520</td><td>0.448</td><td>0.429</td><td>0.497</td><td>0.379</td><td>0.491</td><td>0.493</td><td>0.424</td><td>Haptics/4</td><td>0.471</td><td>0.445</td><td>0.471</td><td></td><td>0.490</td><td>0.451</td><td>0.513</td><td>0.461</td><td>0.427</td></tr><tr><td>ScrnT/1</td><td>0.539</td><td>0.429</td><td>0.501</td><td>0.474</td><td>0.467</td><td>0.405</td><td>0.480</td><td>0.461</td><td>ISkat/4</td><td>0.524</td><td>0.456</td><td>0.473</td><td></td><td>0.571</td><td>0.556</td><td>0.533</td><td>0.566</td><td>0.445</td></tr><tr><td>SmKA/1</td><td>0.173</td><td>0.163</td><td>0.176</td><td>0.279</td><td>0.233</td><td>0.227</td><td>0.221</td><td>0.168</td><td>ToeS1/4</td><td>0.039</td><td>0.035</td><td>0.031</td><td></td><td>0.046</td><td>0.066</td><td>0.039</td><td>0.031</td><td>0.026</td></tr><tr><td>ECG200/2</td><td>0.090</td><td>0.140</td><td>0.160</td><td>0.092</td><td>0.070</td><td>0.100</td><td>0.070</td><td>0.050</td><td>ToeS2/4</td><td>0.077</td><td>0.062</td><td></td><td>0.038</td><td>0.054</td><td>0.131</td><td>0.069</td><td>0.138</td><td>0.038</td></tr><tr><td>ECG5000/2</td><td>0.055</td><td>0.053</td><td>0.054</td><td>0.060</td><td>0.057</td><td>0.061</td><td>0.070</td><td>0.033</td><td>UWAll/4</td><td>0.029</td><td>0.025</td><td></td><td>0.030</td><td>0.058</td><td>0.039</td><td>0.046</td><td>0.028</td><td>0.020</td></tr><tr><td>NFET1/2</td><td>0.051</td><td>0.050</td><td>0.083</td><td>0.037</td><td>0.049</td><td>0.055</td><td>0.026</td><td>0.040</td><td>UWX/4</td><td>0.152</td><td>0.142</td><td>0.157</td><td></td><td>0.178</td><td>0.192</td><td>0.184</td><td>0.218</td><td>0.148</td></tr><tr><td>NFET2/2</td><td>0.036</td><td>0.034</td><td>0.052</td><td>0.040</td><td>0.046</td><td>0.042</td><td>0.028</td><td>0.044</td><td>UWY/4</td><td>0.224</td><td>0.219</td><td>0.229</td><td></td><td>0.243</td><td>0.257</td><td>0.250</td><td>0.232</td><td>0.210</td></tr><tr><td>Adiac/3</td><td>0.184</td><td>0.194</td><td>0.202</td><td>0.165</td><td>0.182</td><td>0.174</td><td>0.155</td><td>0.143</td><td>UWZ/4</td><td>0.199</td><td>0.201</td><td>0.214</td><td></td><td>0.236</td><td>0.247</td><td>0.253</td><td>0.265</td><td>0.164</td></tr><tr><td>Arrow/3</td><td>0.137</td><td>0.131</td><td>0.194</td><td>0.162</td><td>0.103</td><td>0.200</td><td>0.181</td><td>0.091</td><td>Worms/4</td><td>0.260</td><td>0.260</td><td>0.182</td><td></td><td>0.235</td><td>0.182</td><td>0.208</td><td>0.195</td><td>0.169</td></tr><tr><td>BChic/3</td><td>0.100</td><td>0.100</td><td>0.050</td><td>0.115</td><td>0.100</td><td>0.050</td><td>0.100</td><td>0</td><td>WormT/4</td><td>0.221</td><td>0.195</td><td>0.169</td><td></td><td>0.343</td><td>0.169</td><td>0.286</td><td>0.208</td><td>0.130</td></tr><tr><td>DPOAG/3</td><td>0.266</td><td>0.237</td><td>0.252</td><td>0.262</td><td>0.216</td><td>0.288</td><td>0.245</td><td>0.216</td><td>Chlor/5</td><td>0.245</td><td>0.241</td><td>0.340</td><td></td><td>0.161</td><td>0.217</td><td>0.131</td><td>0.095</td><td>0.058</td></tr><tr><td>DPOC/3</td><td>0.210</td><td>0.225</td><td>0.243</td><td>0.234</td><td>0.210</td><td>0.264</td><td>0.217</td><td>0.181</td><td>EQ/5</td><td>0.273</td><td>0.252</td><td>0.252</td><td></td><td>0.330</td><td>0.180</td><td>0.288</td><td>0.223</td><td>0.245</td></tr><tr><td>DPTW/3</td><td>0.345</td><td>0.281</td><td>0.324</td><td>0.336</td><td>0.288</td><td>0.302</td><td>0.268</td><td>0.281</td><td>FordA/5</td><td>0.052</td><td>0.044</td><td>0.050</td><td></td><td>0.045</td><td>0.075</td><td>0.042</td><td>0.110</td><td>0.028</td></tr><tr><td>FaceAll/3</td><td>0.193</td><td>0.118</td><td>0.158</td><td>0.155</td><td>0.072</td><td>0.182</td><td>0.098</td><td>0.090</td><td>FordB/5</td><td>0.180</td><td>0.163</td><td>0.177</td><td></td><td>0.162</td><td>0.225</td><td>0.156</td><td>0.222</td><td>0.144</td></tr><tr><td>FaceUCR/3</td><td>0.040</td><td>0.035</td><td>0.033</td><td>0.033</td><td>0.071</td><td>0.038</td><td>0.087</td><td>0.037</td><td>ItaPD/5</td><td>0.037</td><td>0.030</td><td>0.035</td><td></td><td>0.053</td><td>0.030</td><td>0.041</td><td>0.023</td><td>0.030</td></tr><tr><td>FWords/3</td><td>0.163</td><td>0.167</td><td>0.152</td><td>0.184</td><td>0.198</td><td>0.169</td><td>0.281</td><td>0.167</td><td>Lgt2/5</td><td>0.246</td><td>0.213</td><td>0.164</td><td></td><td>0.193</td><td>0.049</td><td>0.115</td><td>0.145</td><td>0.066</td></tr><tr><td>HandO/3</td><td>0.065</td><td>0.059</td><td>0.062</td><td>0.071</td><td>0.051</td><td>0.046</td><td>0.100</td><td>0.024</td><td>Lgt7/5</td><td>0.205</td><td>0.192</td><td>0.233</td><td></td><td>0.207</td><td>0.123</td><td>0.219</td><td>0.091</td><td>0.178</td></tr><tr><td>Herring/3</td><td>0.312</td><td>0.391</td><td>0.359</td><td>0.392</td><td>0.297</td><td>0.484</td><td>0.453</td><td>0.329</td><td>Phonm/5</td><td>0.706</td><td>0.633</td><td>0.639</td><td></td><td>0.695</td><td>0.715</td><td>0.698</td><td>0.800</td><td>0.604</td></tr><tr><td>MedIm/3</td><td>0.209</td><td>0.193</td><td>0.204</td><td>0.231</td><td>0.207</td><td>0.212</td><td>0.164</td><td>0.190</td><td>SAR1/5</td><td>0.113</td><td>0.085</td><td>0.168</td><td></td><td>0.020</td><td>0.065</td><td>0.022</td><td>0.042</td><td>0.065</td></tr><tr><td>MPOAG/3</td><td>0.448</td><td>0.422</td><td>0.429</td><td>0.464 0.186</td><td>0.331 0.144</td><td>0.539 0.189</td><td>0.461 0.192</td><td>0.351 0.172</td><td>SAR2/5 SLCrv/5</td><td>0.079 0.018</td><td>0.077</td><td>0.104</td><td></td><td>0.046</td><td>0.075</td><td>0.072</td><td>0.064</td><td>0.049 0.017</td></tr><tr><td>MPOC/3 MPTW/3</td><td>0.151 0.474</td><td>0.151 0.422</td><td>0.175 0.435</td></table>

• Hivecote2 [23]: A heterogeneous meta ensemble model that achieves state-of-the-art performance in TSC. It is an improved version of Hivecote that forms its ensemble from classifiers of multiple domains.

• TS-Chief [48]: TS-Chief, short for Time Series Combination of Heterogeneous and Ensemble Embedding Forests, leverages the scalability of tree classifiers and decades of research into accurate and specialized TSC techniques.

• OS-CNN [49]: It proposes an omni-scale block (OS-block) for 1D CNN, which uses many kernels of different sizes for multi-scale feature extraction. It is a representative of CNN-based deep learning models.

• TimeURL [50]: A self-supervised model for time series representation learning that employs contrastive learning to capture both segment-level and instance-level information. It is the latest representative of deep learning models utilizing self-supervised techniques.

• MILLET [51]: It is a recent deep learning model that employs multiple instance learning techniques to improve performance and provide local explanations for TSC.

• mWDN [1]: This method focuses on the frequency-based decomposition. It uses a multi-level Wavelet Decomposition Network, which implements an approximate wavelet decomposition using a fully convolutional layer.

## 6.1.3 Implementation Details

Our TFBC model uses the Adam optimizer for training. We configure the batch size to 16. The task balancing coefficients $\gamma _ { 1 } , \gamma _ { 2 }$ are adjusted using a dynamic weightaveraging method [52], beginning with values of 1.0 each. The maximum number of training epochs is 200; during epochs 1 to 100, the learning rate is maintained at 0.001, decreases to 0.0001 between epochs 100 to 150, and further reduces to 0.00001 from epochs 150 to 200. The number of decomposition levels is searched from 1 to 5, while the hidden dimension is searched in the set {16, 24, 32, 48}. These hyperparameters are set based on the optimal performance on the validation dataset, with details in Sec. 1.4 of SM. Our code is available at https://github.com/Beihang-BIGSCity/ mwcn ts.

TABLE 3  
TSC performance summary. Avg. Err. denotes the average classification error rate. Count: winning count. Rank(a)/Rank(g): the average ranking in terms of arithmetic and geometry.
<table><tr><td colspan="11">Metric |mRocket Hivecote2 TS-Chief OS-CNN TimeURL MILLET mWDN TFBC</td></tr><tr><td>Avg. Err.</td><td>0.187</td><td>0.171</td><td>0.186</td><td>0.192</td><td>0.173</td><td>0.186</td><td></td><td>0.184</td><td>0.146</td></tr><tr><td>Count</td><td>3</td><td>7</td><td>6</td><td>7</td><td>12</td><td></td><td>5</td><td>11</td><td>34</td></tr><tr><td>Rank (a)</td><td>5.11</td><td>3.53</td><td>5.00</td><td>5.34</td><td>4.17</td><td>5.00</td><td></td><td>4.63</td><td>1.88</td></tr><tr><td>Rank (g)</td><td>4.58</td><td>3.10</td><td>4.32</td><td>4.61</td><td>3.32</td><td>4.32</td><td></td><td>3.71</td><td>1.59</td></tr></table>

## 6.1.4 Results and Analysis

Tab. 2 shows the detailed experimental results of 64 UCR datasets, with a summary in Tab. 3. Each experiment was run five times with different random seeds, and the average performance is reported. From the results, we can have the following key observations.

First, it is clear that among all the competitors, TFBC achieves the best performance in terms of both the largest number of wins (the best in 34 out of 64 datasets) and the highest average rank with regard to both arithmetic (1.88) and geometry (1.59). In addition, TFBC reduces the average error rate by 19.97% over all baselines on average (See Sec. 4 of SM for the calculation details of average performance improvement). The rank index indicates that even in the cases where our model is not the best, its performance is still very competitive. TFBC’s superior performance demonstrates the effectiveness of our model and underscores the importance of simultaneously capturing frequency and pattern information. Our model’s superiority is further verified by the Nemenyi test at level 5% in Sec. 1.3 of SM.

TABLE 4  
Ablation study on the 64 UCR datasets. Avg. Err. denotes the average classification error rate. Count: winning count. Rank(a)/Rank(g): the average ranking in terms of arithmetic and geometry.
<table><tr><td>Metric</td><td>r/fp</td><td>r/wd</td><td>w/o m-WCN</td><td>w/o OR</td><td>w/o FCL</td><td>TFBC</td></tr><tr><td>Avg. Err.</td><td>0.162</td><td>0.170</td><td>0.294</td><td>0.156</td><td>0.154</td><td>0.146</td></tr><tr><td>Count</td><td>9</td><td>15</td><td>5</td><td>10</td><td>14</td><td>26</td></tr><tr><td>Rank(a)</td><td>3.187</td><td>3.516</td><td>5.125</td><td>2.781</td><td>2.828</td><td>2.281</td></tr><tr><td>Rank(g)</td><td>2.790</td><td>2.943</td><td>4.641</td><td>2.487</td><td>2.471</td><td>1.921</td></tr></table>

Second, our TFBC beats other competitive models across all categories of datasets, highlighting its robustness and adaptability to different data types. Compared with the second-best model, TFBC exhibits a greater improvement on datasets belonging to IMAGE, MOTION, and SENSOR types (refer to the illustration in Sec. 1.2 of SM). This can be attributed to the complex patterns of these datasets, where the performance benefits of multi-wavelet decomposition are more pronounced.

Third, mWDN, as a competitive baseline, can be seen as a degradation of TFBC that removes the prefiltering module that extracts distinct patterns. The comparison between mWDN and TFBC highlights that the pattern-based decomposition is essential to the success of TFBC in TSC.

Finally, deep learning methods, including TimeURL, mWDN, and TFBC, perform better overall, indicating that deep learning’s representation learning ability is suitable for extracting features from large-scale time series data. Besides, TimeURL, which adopts self-supervised learning technology, achieves better results by refining the representation through contrastive learning. This is why we also incorporate contrastive learning into model training.

The results on the full set of 128 UCR datasets lead to consistent observations (See Sec. 3 of SM)..

## 6.1.5 Ablation ofImportant Modules

We conduct an ablation study on the 64 representative UCR datasets to evaluate the contribution of each proposed component. We compare TFBC with five variants: (1) r/fp, which replaces the learnable pre-filtering network with fixed GHM pre-filtering parameters; (2) r/wd, which replaces the learnable multi-wavelet decomposition with standard wavelet decomposition; (3) w/o m-WCN, which removes the m-WCN module; (4) w/o OR, which removes the orthogonality regularization in Eq. (23); and (5) w/o FCL, which removes the frequency contrastive learning loss in Eq. (30). The results are summarized in Table 4.

The results show that each component contributes to the final performance. TFBC achieves the lowest average error rate of 0.146, the largest winning count of 26, and the best arithmetic and geometric average ranks of 2.281 and 1.921, respectively. Removing the whole m-WCN module leads to the largest performance drop. The average error rate increases from 0.146 to 0.294, and the arithmetic average rank worsens from 2.281 to 5.125. This confirms that the neuralized multi-wavelet decomposition module is the core component of TFBC.

The comparison with r/fp and r/wd further verifies the benefit of learnable multi-wavelet decomposition. When the learnable pre-filtering network is replaced with fixed parameters, the average error rate increases to 0.162. When multiwavelet decomposition is replaced by standard wavelet decomposition, the average error rate increases to 0.170. These results show that both learnable pattern decomposition and multi-wavelet frequency decomposition are important. They allow TFBC to adapt the decomposition process to different datasets, rather than using fixed decomposition only as feature engineering.

The results of w/o OR and w/o FCL also demonstrate the usefulness of the two training regularizers. Removing orthogonality regularization increases the average error rate to 0.156 and reduces the winning count to 10. Removing frequency contrastive learning increases the average error rate to 0.154 and reduces the winning count to 14. These results show that orthogonality regularization helps reduce redundancy among decomposed components, while frequency contrastive learning improves the discriminative ability of frequency representations. Overall, the ablation results confirm that m-WCN, learnable decomposition, orthogonality regularization, and frequency contrastive learning all make positive contributions to TFBC.

The ablation results on the full 128 UCR datasets are provided in Sec. 3.2 of SM. The conclusions are consistent with those obtained on the 64 representative datasets.

## 6.2 Task II: Time Series Forecasting

## 6.2.1 Datasets

In the experiments, we compare our model with baseline models over seven real-world TSF datasets. All of the datasets are publicly available:

• ETT (h1, h2, m1, m2): the Electricity Transformer Temperature (ETT) datasets [53] contain 2 years of electricity transformer temperature data from a county in China. They include four subsets: ETTh1 and ETTh2 at the 1-hour level, and ETTm1 and ETTm2 at the 15-minute level.

• Electricity: the electricity dataset from the UCI Machine Learning Repository<sup>2</sup> contains hourly electricity consumption for 370 clients from 2012 to 2014.

• Traffic: the traffic dataset from the California Department of Transportation<sup>3</sup> contains road occupancy rates measured by 862 sensors in the San Francisco Bay area freeways during 2015 and 2016.

• Weather: the weather dataset from NOAA/NCEI Local Climatological Data<sup>4</sup> contains local climatological observations from nearly 1,600 U.S. locations over four years from 2010 to 2013. The data are collected at 1-hour intervals. Each data point contains the target value “wet bulb” and 11 climate features.

In the experiments, we follow common task settings in TSF and use our model and baselines to forecast the future 96, 192, 336, and 720 steps. The input sequence length for each forecasting task is searched over the set {96, 192, 336, 720}. The original series is divided into training, validation, and test sets in a ratio of 6:2:2 for ETT Datasets and in a ratio of 7:1:2 for other datasets. Subsequently, a sliding window approach is employed to produce data samples for each dataset.

## 6.2.2 Evaluation Metric and Baselines

In the experiments, we use two common metrics to evaluate prediction accuracy: Mean Squared Error (MSE) and Mean Absolute Error (MAE). We compared our model with seven competitive baselines.

• Autoformer [39]: The approach introduces a decomposition mechanism grounded in auto-correlation. It adheres to the Transformer encoder-decoder framework but incorporates a decomposition module to capture the complex temporal dynamics of the hidden states.

• Fedformer [14]: This model is a frequency-enhanced decomposed transformer utilizing the seasonal-trend decomposition approach. The decomposition mechanism encapsulates the overall pattern of time series data, while Transformers extract detailed structural aspects.

• Dlinear [54]: The model is a combination of a decomposition scheme used in Autoformer and Fedformer with linear layers. It first decomposes the input series into a trend component and a remainder (seasonal) component, and then two linear layers are applied to each component for the final prediction.

• Basisformer [55]: This model utilizes cross-attention to calculate the similarity coefficients between the time series and learnable bases in the historical view, and then selects the bases in the future view based on the similarity coefficients for accurate prediction.

• Pathformer [56]: It models the multi-scale characteristics of time series with adaptive pathways integrating temporal resolutions and temporal distance information.

• PatchTST [57]: This method segments time series into subseries-level patches, which are served as input tokens to the Transformer.

• TimeLLM [58]: This is a recent large language model based framework for time series forecasting. It reformulates time series forecasting as a sequence modeling task and leverages the representation and reasoning ability of pre-trained language models. We include TimeLLM as a recent strong baseline to evaluate whether FTB remains competitive against LLM-based forecasting methods.

## 6.2.3 Implementation Details

We have implemented the FTB model utilizing PyTorch, and it is trained with the Adam optimizer at a learning rate of 0.001 for a total of 70 epochs. The hidden embedding size is configured to 64, with a batch size of 32. The task balancing coefficient $\gamma _ { 3 }$ is optimized through a dynamic weightaveraging method [52], starting from an initial value of 1.0. We search the number of decomposition levels from 1 to 5, and the hidden dimension in the set {16, 24, 32, 48}. Both hyperparameters are set based on the optimal performance on validation datasets (see Sec. 2.1 of SM). Our code is available at https://github.com/Beihang-BIGSCity/mwcn ts.

TABLE 5  
Performance comparison of TSF with different prediction lengths. The best result is bold, while the second-best result is underlined. Auto.=Autoformer, Fed.=Fedformer, Basis.=Basisformer, and Path.=PathFormer.
<table><tr><td colspan="3">Dataset | Horizon | Metric</td><td>Auto.</td><td>Fed.</td><td></td><td>DLinear Basis.</td><td>Path.</td><td></td><td>TimeLLM PatchTST</td><td>FTB</td></tr><tr><td colspan="3"></td><td>MSE 0.071 MAE</td><td>0.079</td><td>0.056</td><td>0.055</td><td>0.057</td><td>0.058</td><td>0.057</td><td>0.052</td></tr><tr><td rowspan="7">ETITH1</td><td rowspan="7">96 192 336 720</td><td>MSE</td><td>0.206 0.114</td><td>0.215 0.104</td><td>0.180 0.071</td><td>0.178</td><td>0.180</td><td>0.183</td><td>0.179</td><td>0.165</td></tr><tr><td></td><td></td><td></td><td></td><td>0.072</td><td>0.075</td><td>0.072</td><td>0.076</td><td>0.064</td></tr><tr><td>MAE</td><td>0.262</td><td>0.245</td><td>0.204</td><td>0.204</td><td>0.208</td><td>0.202</td><td>0.209</td><td>0.180</td></tr><tr><td>MSE</td><td>0.107</td><td>0.119</td><td>0.098</td><td>0.086</td><td>0.076</td><td>0.082</td><td>0.093</td><td>0.063</td></tr><tr><td></td><td></td><td></td><td>0.244</td><td>0.227</td><td>0.216</td><td>0.231</td><td>0.240</td><td>0.201</td></tr><tr><td>MAE</td><td>0.258</td><td>0.270</td><td>0.189</td><td></td><td>0.090</td><td>0.093</td><td>0.097</td><td>0.077</td></tr><tr><td>MSE MAE</td><td>0.126 0.283</td><td>0.142 0.299</td><td>0.359</td><td>0.080 0.220</td><td>0.238</td><td>0.243</td><td>0.245</td><td>0.213</td></tr><tr><td rowspan="7">ET2</td><td rowspan="8">96 192 336</td><td>MSE</td><td>0.150</td><td>0.132</td><td>0.132</td><td>0.133</td><td>0.137</td><td>0.132</td><td>0.129</td><td>0.130</td></tr><tr><td>MAE</td><td>0.303</td><td>0.287</td><td>0.279</td><td>0.286</td><td>0.291</td><td>0.286</td><td>0.282</td><td>0.284 0.170</td></tr><tr><td>MSE</td><td>0.195</td><td>0.171</td><td>0.175</td><td>0.183</td><td>0.371</td><td>0.177</td><td>0.168</td><td>0.331</td></tr><tr><td>MAE</td><td>0.343</td><td>0.331</td><td>0.334</td><td>0.336</td><td>0.390</td><td>0.336</td><td>0.328</td><td></td></tr><tr><td>MSE</td><td>0.234</td><td>0.193</td><td>0.211</td><td>0.211</td><td>0.331</td><td>0.195</td><td>0.185</td><td>0.183</td></tr><tr><td>MAE</td><td>0.387</td><td>0.366</td><td>0.369</td><td>0.367</td><td>0.373</td><td>0.374</td><td>0.351</td><td>0.346</td></tr><tr><td>MSE</td><td>0.272</td><td>0.233</td><td>0.295</td><td>0.238</td><td>0.417</td><td>0.237</td><td>0.224</td><td>0.219</td></tr><tr><td>720 96</td><td>MAE 0.418</td><td>0.387</td><td>0.442</td><td>0.393</td><td>0.434</td><td>0.397</td><td>0.383</td><td>0.371</td></tr><tr><td rowspan="6">ETTm1</td><td rowspan="8">192 336</td><td>MSE</td><td>0.051</td><td>0.029</td><td>0.027 0.123</td><td>0.029</td><td>0.029</td><td>0.029</td><td>0.026</td><td>0.022</td></tr><tr><td></td><td>0.176</td><td>0.127</td><td></td><td>0.127</td><td>0.125</td><td>0.129</td><td>0.121</td><td>0.116</td></tr><tr><td>MAE</td><td>0.076</td><td>0.042</td><td>0.043 0.044</td><td></td><td>0.040</td><td>0.044</td><td>0.039</td><td>0.031</td></tr><tr><td>MSE</td><td>0.221</td><td></td><td>0.154</td><td>0.160</td><td>0.153</td><td>0.162</td><td>0.150</td><td>0.139</td></tr><tr><td>MAE</td><td>0.081</td><td>0.159 0.059</td><td>0.060 0.059</td><td></td><td>0.057</td><td>0.062</td><td>0.053</td><td>0.042</td></tr><tr><td>MSE MAE</td><td>0.226</td><td>0.185</td><td>0.180</td><td>0.186</td><td>0.182</td><td>0.188</td><td>0.173</td><td>0.162</td></tr><tr><td>MSE</td><td>0.106</td><td>0.079</td><td>0.081 0.211 0.221</td><td>0.083</td><td>0.082 0.220</td><td>0.085 0.227</td><td>0.074 0.207</td><td>0.059 0.189</td></tr><tr><td rowspan="6">ETTm2 336</td><td rowspan="4">96 192</td><td></td><td>0.066</td><td></td><td>0.070</td><td>0.071</td><td>0.064</td><td>0.073</td><td>0.070</td><td>0.062</td></tr><tr><td>MSE 0.086</td><td>0.223</td><td>0.193</td><td>0.191</td><td>0.191</td><td>0.181</td><td>0.196</td><td>0.191</td><td>0.177</td></tr><tr><td>MAE MSE</td><td>0.148 0.113</td><td></td><td>0.104 0.104</td><td></td><td>0.100</td><td>0.106</td><td>0.104</td><td>0.102</td></tr><tr><td>MAE 0.295</td><td>0.262</td><td></td><td>0.238</td><td>0.235</td><td>0.232 0.129</td><td>0.241 0.132</td><td>0.238 0.135</td><td>0.234 0.126</td></tr><tr><td>MSE MAE 0.309</td><td>0.155</td><td>0.158 0.135 0.305</td><td>0.278</td><td>0.130 0.275</td><td>0.270</td><td>0.273</td><td>0.278</td><td>0.260 0.158</td></tr><tr><td rowspan="8">Eleiciy</td><td rowspan="8">720 96</td><td>MSE 0.177</td><td></td><td>0.199 0.348</td><td>0.332</td><td>0.182 0.180 0.332</td><td>0.329</td><td>0.330</td><td>0.188 0.332</td><td>0.315</td></tr><tr><td>MAE 0.329</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.203</td></tr><tr><td>MSE</td><td>0.381</td><td>0.253</td><td>0.374</td><td>0.333</td><td>0.373</td><td>0.221</td><td>0.217 0.326</td><td>0.322</td></tr><tr><td>MAE</td><td>0.458</td><td>0.370</td><td>0.439</td><td>0.408</td><td>0.520 0.306</td><td>0.335 0.269</td><td></td><td>0.256 0.247</td></tr><tr><td>MSE</td><td>0.469</td><td>0.282</td><td>0.351</td><td>0.371</td><td>0.389</td><td></td><td>0.354</td><td>0.351</td></tr><tr><td>MAE</td><td>0.516</td><td>0.386</td><td>0.422 0.379</td><td>0.427</td><td>0.369</td><td>0.384 0.322</td><td></td><td>0.296 0.285</td></tr><tr><td>MSE</td><td>0.591</td><td>0.346</td><td>0.442</td><td>0.413 0.455</td><td>0.427</td><td>0.417</td><td>0.387</td><td>0.378</td></tr><tr><td>MAE 720 MSE 0.658</td><td>0.570</td><td>0.431 0.422</td><td>0.417</td><td>0.471</td><td>0.383</td><td>0.356</td><td>0.339 0.448</td><td>0.323 0.431 0.421</td></tr><tr><td rowspan="6">Tratic</td><td rowspan="8">96 192 336 MSE MAE 720 MSE MAE</td><td>MSE</td><td></td><td>0.484</td><td>0.303</td><td></td><td></td><td></td><td></td><td>0.102</td></tr><tr><td></td><td>0.244</td><td>0.207</td><td></td><td>0.186</td><td>0.144</td><td>0.164</td><td>0.176</td><td>0.172</td></tr><tr><td>MAE</td><td>0.352</td><td>0.312</td><td>0.396</td><td>0.280</td><td>0.211 0.139</td><td>0.246 0.171</td><td>0.253</td><td>0.162 0.122</td></tr><tr><td>MSE MAE</td><td>0.280 0.385 0.312</td><td>0.205</td><td>0.247 0.335</td><td>0.175 0.267</td></table>

## 6.2.4 Results and Analysis

Tab. 5 presents the comparison results of all the methods, from which we make the following three observations. Moreover, Tab. 6 gives the performance summary over seven forecasting datasets of the TSF experiments.

First, according to Tab. 6, our model consistently outperforms all competing baselines across most tasks on the seven datasets, with an average error reduction of 24.45% in MSE and 15.38% in MAE (See Sec. 4 of SM for the calculation details of average performance improvement). In contrast, the second-best model varies across tasks. This shows that our model offers more stable and reliable results, highlighting its robustness and adaptability to various data.

Second, compared with Fedformer, which mainly exploits frequency-domain information, and PathFormer and PatchTST, which focus more on temporal pattern modeling, FTB jointly models pattern and frequency information through neuralized multi-wavelet decomposition. The results show that FTB achieves the best overall average

TABLE 6  
TSF performance summary over seven forecasting datasets. Count denotes the number of wins including ties over all metric-horizon comparisons. Rank(a) and Rank(g) denote the arithmetic and geometric average ranks.
<table><tr><td>Metric</td><td>|Auto. Fed.</td><td></td><td></td><td></td><td>DLinear Basis. Path. TimeLLM PatchTST FTB</td><td></td><td></td><td></td></tr><tr><td>Avg. MSE</td><td>0.192</td><td>0.147</td><td>0.163</td><td>0.146</td><td>0.153</td><td>0.129</td><td>0.126</td><td>0.112</td></tr><tr><td>Avg. MAE</td><td>0.3030.265</td><td></td><td>0.271</td><td>0.2480.242</td><td></td><td>0.238</td><td>0.233</td><td>0.216</td></tr><tr><td>Count</td><td>0</td><td>0</td><td>1</td><td>0</td><td>5</td><td>0</td><td>4</td><td>50</td></tr><tr><td>Rank(a)</td><td>7.42</td><td>5.61</td><td>5.55</td><td>4.87</td><td>3.72</td><td>4.49</td><td>3.17</td><td>1.17</td></tr><tr><td>Rank(g)</td><td>7.26</td><td>5.35</td><td>5.22</td><td>4.65</td><td>3.21</td><td>4.26</td><td>2.86</td><td>1.12</td></tr></table>

TABLE 7  
Ablation study of FTB on the Electricity dataset.
<table><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td></td><td>720</td></tr><tr><td>Metric</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>FTB</td><td>0.203 0.322</td><td>0.247 0.351</td><td>0.285</td><td>0.378</td><td>0.323 0.421</td></tr><tr><td>r/ fp</td><td>0.215 0.352</td><td>0.266 0.386</td><td>0.288</td><td>0.359</td><td>0.342 0.432</td></tr><tr><td>r/wd</td><td>0.247 0.359</td><td>0.319</td><td>0.366 0.340</td><td>0.389</td><td>0.373 0.486</td></tr><tr><td>w/o mwcn</td><td>0.256 0.389</td><td>0.307</td><td>0.395 0.332</td><td>0.425</td><td>0.388 0.502</td></tr><tr><td>w/o or</td><td>0.238 0.333</td><td>0.258</td><td>0.362 0.288</td><td>0.381</td><td>0.359 0.453</td></tr><tr><td>w/o pt</td><td>0.2280.341</td><td>0.268</td><td>0.368</td><td>0.3150.391</td><td>0.345 0.488</td></tr></table>

MSE and MAE across the seven forecasting datasets. This supports the effectiveness of jointly optimizing pattern and frequency decomposition for time series forecasting.

Third, models with explicit structural priors often show more interpretable behavior as the prediction horizon changes. In our results, FTB generally exhibits a smooth increase in prediction error when the horizon becomes longer, which is consistent with the increasing difficulty of long-term forecasting. This behavior suggests that the proposed multi-wavelet decomposition provides a useful inductive bias for capturing changes in data predictability. By contrast, several baselines show more dataset-dependent trends across horizons.

## 6.2.5 Ablation of Important Modules

The ablation study is conducted to analyze how each of the proposed components in FTB affects the final forecasting performance. We prepare five variants for comparison. The first four variants are the same as those of Sec. 6.1.5. The last variant removes the pre-training phase, denoted as w/o pt. We report the experimental results on all forecasting tasks regarding both metrics in Tab. 7. Results on all seven datasets show a similar phenomenon, so we only report the results on the Electricity dataset for simplicity.

We can observe from Tab. 7 that all components contribute to the model’s overall performance. Particularly, variants w/o mwcn and r/ wd show a great performance decrease, indicating that our proposed m-WCN can effectively capture the characteristics of the input series data by using learnable multi-wavelets for pattern and frequency decomposition. In addition, the pre-training phase plays a crucial role in maintaining the model’s performance. This is due to the fact that the pre-training task, which involves recovering the MWD decomposition outcomes, imposes constraints on the model’s learning process and reduces the risk of overfitting.

## 6.2.6 Efficiency Comparison

We further conduct empirical efficiency comparisons on seven forecasting datasets, including ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, and Traffic. All experiments are conducted on the same hardware platform: a single NVIDIA GeForce RTX 3090 GPU. For all methods, the input length and prediction length are both set to 336. Table 8 reports the average training time, training memory, inference time, and inference memory over the seven datasets. The training time is measured as the one-epoch training time, while the inference time is measured on the test set. As shown in Table 8, FTB requires 63.76 seconds per training epoch, which is comparable to Autoformer and PathFormer. Its inference time is 6.24 seconds, also close to PathFormer. In terms of memory usage, FTB requires 464.07 MB during training and 407.62 MB during inference, which remains within a practical range among Transformer-based forecasting models. Overall, these results show that FTB maintains practical computational efficiency while achieving strong forecasting performance.

TABLE 8  
Average computational cost on the forecasting experiments. TT, TM, IT, and IM denote training time, training memory, inference time, and inference memory, respectively.
<table><tr><td>Method</td><td>TT (s/epoch)</td><td>TM (MB)</td><td>IT (s)</td><td>IM (MB)</td></tr><tr><td>Autoformer</td><td>66.27</td><td>3848.13</td><td>22.53</td><td>738.82</td></tr><tr><td>Fedformer</td><td>237.78</td><td>2612.12</td><td>15.32</td><td>677.35</td></tr><tr><td>DLinear</td><td>7.54</td><td>21.14</td><td>4.62</td><td>19.63</td></tr><tr><td>Basisformer</td><td>42.49</td><td>58.47</td><td>3.52</td><td>39.15</td></tr><tr><td>PathFormer</td><td>58.39</td><td>2227.41</td><td>6.82</td><td>367.09</td></tr><tr><td>PatchTST</td><td>8.91</td><td>153.49</td><td>3.00</td><td>68.23</td></tr><tr><td>FTB</td><td>63.76</td><td>464.07</td><td>6.24</td><td>407.62</td></tr></table>

## 6.2.7 Prediction Visualization

In this part, we further visualize the prediction results to explore the effectiveness of our model. As shown in Fig. 5, we depict the prediction results of the 96-step forecasting task on the Electricity dataset. The left and right figures visualize periodic and non-periodic samples, respectively. In order to make the results more convincing, we also include the optimal baseline model, PatchTST, as a comparison.

From the visualization results, we have three main findings. (1) From the left half of Fig. 5, the periodic series mainly consists of two kinds of patterns, i.e., rising and falling patterns. PatchTST accurately identifies the rising pattern, but it struggles with the falling pattern, as indicated by the dotted circle. In contrast, our FTB effectively captures both rising and falling patterns. This capability stems from our pattern decomposition module, which enables the features from different channels to discern unique patterns (refer to Sec. 2.2 of SM for decomposed component visualization), thereby enhancing pattern learning quality. (2) In the more intricate non-periodic sample depicted in the right-hand section of Fig. 5, PatchTST learned the trend shifts but significantly struggles to detect high-frequency details, as shown in dotted circles. However, our FTB, which utilizes frequency decomposition, proficiently captures both high- and low-frequency details, leading to highly accurate predictions. This highlights the efficacy of our model in utilizing frequency data to improve the analysis of complex time series. (3) Integrating the findings from both subfigures reveals that our model delivers superior predictive performance on time series signals regardless of whether their distributions are periodic or aperiodic. This highlights the robustness of our model and its adaptability in diverse situations.

![](images/b6a819870bc35d64925bed8b1cfc868a07cad949803fed7019c9db36fcd8814c.jpg)  
(a)

![](images/545f49a33b7caaaba2a5033341274f01d8cd8f2f53dfb616c101aef5bbae3648.jpg)

![](images/f480acad8e5dfdb2070a48b5b4f839297de9bed831d92d28db83e245ae1fe541.jpg)  
(c)

(b)  
![](images/ccf8673464078a5fb607015d199457e78d39ba3e20382490429d6c6031f291e1.jpg)  
(d)  
Fig. 5. Visualization of forecasting results. Subfigures (a) and (b) show the prediction results of FTB, while subfigures (c) and (d) show the prediction results of PatchTST on the same samples.

## 7 CONCLUSION AND FUTURE WORK

In this paper, we proposed the Multi-Wavelet Decomposition Convolution Network (m-WCN), a novel framework that integrates frequency and pattern decomposition within a trainable, end-to-end architecture for time series analysis. By incorporating neuralized MWD and orthogonality regularization, m-WCN achieves enhanced adaptability while preserving interpretability. Moreover, it can be seamlessly integrated into deep learning pipelines. We developed two m-WCN-based models, TFBC and FTB, specifically designed for classification and forecasting tasks, respectively. Extensive experiments on diverse real-world datasets demonstrated our models’ superiority to state-of-the-art baselines across various settings, underscoring our models’ robustness and versatility in handling time series data.

Due to the characteristics of MWD, our current framework is naturally suited for univariate time series. Extending it to multivariate time series requires handling each variable independently, which can be cumbersome. As part of future work, we aim to extend m-WCN to support multivariate time series more naturally, thereby broadening its applicability and impact within the time series community.

## ACKNOWLEDGMENTS

Prof. Wang’s work is supported by the National Natural Science Foundation of China (No. 72242101, 72625015), and the Science and Technology Development Fund Macau SAR (0052/2023/RIA1). Dr. Junjie Wu’s work was partially supported by the National Natural Science Foundation of China (72595861), the Outstanding Young Scientist Program of Beijing Universities (JWZQ20240201002), and the Shenzhen Science and Technology Program (CJGJZD20230724093201004).

## REFERENCES

[1] J. Wang, Z. Wang, J. Li, and J. Wu, “Multilevel wavelet decomposition network for interpretable time series analysis,” in Proc. of KDD, 2018, pp. 2437–2446.

[2] J. Wang, Y. Zhang, K. Tang, J. Wu, and Z. Xiong, “Alphastock: A buying-winners-and-selling-losers investment strategy using interpretable deep reinforcement attention networks,” in Proc. of KDD, 2019, pp. 1900–1908.

[3] H. Ren, J. Wang, and W. X. Zhao, “Generative adversarial networks enhanced pre-training for insufficient electronic health records modeling,” in Proc. of KDD, 2022, pp. 3810–3818.

[4] C. Duchon and R. Hale, Time series analysis in meteorology and climatology: An introduction, 2012.

[5] L. B. Godfrey and M. S. Gashler, “Neural decomposition of timeseries data for effective generalization,” IEEE TNNLS, vol. 29, no. 7, pp. 2973–2985, 2018.

[6] K. Zhang, R. Genc¸ay, and M. E. Yazgan, “Application of wavelet decomposition in time-series forecasting,” Economics Letters, vol. 158, pp. 41–46, 2017.

[7] L. Bluestein, “A linear filtering approach to the computation of discrete fourier transform,” IEEE Transactions on Audio and Electroacoustics, vol. 18, no. 4, pp. 451–455, 1970.

[8] S. G. Mallat, “A theory for multiresolution signal decomposition: the wavelet representation,” IEEE TPAMI, vol. 11, no. 7, pp. 674– 693, 1989.

[9] S. Palani, “The z-transform analysis of discrete time signals and systems,” in Signals and Systems. Springer, 2021, pp. 921–1055.

[10] G. Rilling, P. Flandrin, P. Goncalves et al., “On empirical mode decomposition and its algorithms,” in IEEE-EURASIP NSIP, vol. 3, no. 3, 2003, pp. 8–11.

[11] L. Ye and E. J. Keogh, “Time series shapelets: a new primitive for data mining,” in Proc. of KDD, 2009, pp. 947–956.

[12] J. Hills, J. Lines, E. Baranauskas, J. Mapp, and A. J. Bagnall, “Classification of time series by shapelet transformation,” DMKD, vol. 28, no. 4, pp. 851–881, 2014.

[13] L. Ye and E. J. Keogh, “Time series shapelets: a novel technique that allows accurate, interpretable and fast classification,” DMKD, vol. 22, no. 1-2, pp. 149–182, 2011.

[14] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in Proc. of ICML, ser. Proceedings of Machine Learning Research, vol. 162, 2022, pp. 27 268–27 286.

[15] Z. Hajiabotorabi, A. Kazemi, F. F. Samavati, and F. M. M. Ghaini, “Improving DWT-RNN model via b-spline wavelet multiresolution to forecast a high-frequency time series,” Expert Systems With Applications, vol. 138, 2019.

[16] H. Liu, H. Tian, D. Pan, and Y. Li, “Forecasting models for wind speed using wavelet, wavelet packet, time series and artificial neural networks,” Applied Energy, vol. 107, pp. 191–208, 2013.

[17] J. Wang, C. Yang, X. Jiang, and J. Wu, “WHEN: A wavelet-dtw hybrid attention network for heterogeneous time series analysis,” in Proc. of KDD, 2023, pp. 2361–2373.

[18] S.-A. Chen, C.-L. Li, N. Yoder, S. O. Arik, and T. Pfister, “Tsmixer: An all-mlp architecture for time series forecasting,” arXiv preprint arXiv:2303.06053, 2023.

[19] J. Wang, Z. Peng, X. Wang, C. Li, and J. Wu, “Deep fuzzy cognitive maps for interpretable multivariate time series prediction,” IEEE transactions on fuzzy systems, vol. 29, no. 9, pp. 2647–2660, 2020.

[20] X. Qiu, Y. Ren, P. N. Suganthan, and G. A. J. Amaratunga, “Empirical mode decomposition based ensemble deep learning for load demand time series forecasting,” Applied Soft Computing, vol. 54, pp. 246–255, 2017.

[21] J. Wang, J. Wu, Z. Wang, F. Gao, and Z. Xiong, “Understanding urban dynamics via context-aware tensor factorization with neighboring regularization,” IEEE TKDE, vol. 32, no. 11, pp. 2269–2283, 2020.

[22] J. Grabocka, N. Schilling, M. Wistuba, and L. Schmidt-Thieme, “Learning time-series shapelets,” in Proc. of KDD, 2014, pp. 392– 401.

[23] M. Middlehurst, J. Large, M. Flynn, J. Lines, A. Bostrom, and A. J. Bagnall, “HIVE-COTE 2.0: A new meta ensemble for time series classification,” Machine Learning, vol. 110, no. 11, pp. 3211–3243, 2021.

[24] G. Gupta, X. Xiao, and P. Bogdan, “Multiwavelet-based operator learning for differential equations,” in Proc. of NeurIPS, 2021, pp. 24 048–24 062.

[25] X. Xia, J. S. Geronimo, D. P. Hardin, and B. W. Suter, “Design of prefilters for discrete multiwavelet transforms,” IEEE TSP, vol. 44, no. 1, pp. 25–35, 1996.

[26] J. Y. Tham, L. Shen, S. L. Lee, and H. H. Tan, “A general approach for analysis and application of discrete multiwavelet transforms,” IEEE TSP, vol. 48, no. 2, pp. 457–464, 2000.

[27] Z. Xing, J. Pei, and E. J. Keogh, “A brief survey on sequence classification,” SIGKDD Explorations, vol. 12, no. 1, pp. 40–48, 2010.

[28] D. J. Berndt and J. Clifford, “Using dynamic time warping to find patterns in time series.” in Proc. of KDD, vol. 10, no. 16, 1994, pp. 359–370.

[29] Y. Zhang, R. Jin, and Z. Zhou, “Understanding bag-of-words model: a statistical framework,” International Journal of Machine Learning and Cybernetics, vol. 1, no. 1-4, pp. 43–52, 2010.

[30] J. Franceschi, A. Dieuleveut, and M. Jaggi, “Unsupervised scalable representation learning for multivariate time series,” in Proc. of NeurIPS, 2019, pp. 4652–4663.

[31] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, and C. Eickhoff, “A transformer-based framework for multivariate time series representation learning,” in Proc. of KDD, 2021, pp. 2114–2124.

[32] R. R. Chowdhury, X. Zhang, J. Shang, R. K. Gupta, and D. Hong, “Tarnet: Task-aware reconstruction for time-series transformer,” in Proc. of KDD, 2022, pp. 212–220.

[33] P. Newbold, “ARIMA model building and the time series analysis approach to forecasting,” Journal of Forecasting, vol. 2, no. 1, pp. 23–35, 1983.

[34] C. Kongcharoen and T. Kruangpradit, “Autoregressive integrated moving average with explanatory variable (ARIMAX) model for thailand export,” in 33rd International Symposium on Forecasting, South Korea, 2013, pp. 1–8.

[35] B. M. Williams, P. K. Durvasula, and D. E. Brown, “Urban freeway traffic flow prediction: application of seasonal autoregressive integrated moving average and exponential smoothing models,” Transportation Research Record, vol. 1644, no. 1, pp. 132–141, 1998.

[36] J. Morrill, C. Salvi, P. Kidger, and J. Foster, “Neural rough differential equations for long time series,” in Proc. of ICML, 2021, pp. 7829–7838.

[37] J. Ji, J. Wang, Z. Jiang, J. Jiang, and H. Zhang, “Stden: Towards physics-guided neural networks for traffic flow prediction,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 4, 2022, pp. 4048–4056.

[38] D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” International Journal of Forecasting, vol. 36, no. 3, pp. 1181–1191, 2020.

[39] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” Proc. of NeurIPS, vol. 34, pp. 22 419–22 430, 2021.

[40] C. Han, J. Wang, Y. Wang, X. Yu, H. Lin, C. Li, and J. Wu, “Bridging traffic state and trajectory for dynamic road network and trajectory representation learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

[41] G. R. Lee, R. Gommers, F. Waselewski, K. Wohlfahrt, and A. O’Leary, “Pywavelets: A python package for wavelet analysis,” Journal of Open Source Software, vol. 4, no. 36, p. 1237, 2019.

[42] F. Keinert, Wavelets and multiwavelets. Chapman and Hall/CRC, 2003.

[43] J. S. Geronimo, D. P. Hardin, and P. R. Massopust, “Fractal functions and wavelet expansions based on several scaling functions,” Journal of Approximation Theory, vol. 78, no. 3, pp. 373–401, 1994.

[44] A. Bottcher and B. Silbermann,¨ Introduction to large truncated Toeplitz matrices, 2012.

[45] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2D-variation modeling for general time series analysis,” in Proc. of ICLR, 2023.

[46] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, “The ucr time series archive,” IEEE/CAA Journal of Automatica Sinica, vol. 6, no. 6, pp. 1293–1305, 2019.

[47] A. Dempster, D. F. Schmidt, and G. I. Webb, “Minirocket: A very fast (almost) deterministic transform for time series classification,” in Proc. of KDD, 2021, pp. 248–257.

[48] A. Shifaz, C. Pelletier, F. Petitjean, and G. I. Webb, “TS-CHIEF: A scalable and accurate forest algorithm for time series classification,” DMKD, vol. 34, no. 3, pp. 742–775, 2020.

[49] W. Tang, G. Long, L. Liu, T. Zhou, M. Blumenstein, and J. Jiang, “Omni-Scale CNNs: A simple and effective kernel size configuration for time series classification,” in Proc. of ICLR, 2022.

[50] J. Liu and S. Chen, “TimesURL: Self-supervised contrastive learning for universal time series representation learning,” in Proc. of AAAI, 2024, pp. 13 918–13 926.

[51] J. Early, G. Cheung, K. Cutajar, H. Xie, J. Kandola, and N. Twomey, “Inherently interpretable time series classification via multiple instance learning,” 2024.

[52] S. Liu, E. Johns, and A. J. Davison, “End-to-end multi-task learning with attention,” in Proc. of CVPR, 2019, pp. 1871–1880.

[53] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proc. of AAAI, 2021, pp. 11 106– 11 115.

[54] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proc. of AAAI, 2023, pp. 11 121–11 128.

[55] Z. Ni, H. Yu, S. Liu, J. Li, and W. Lin, “Basisformer: Attentionbased time series forecasting with learnable and interpretable basis,” in Proc. of NeurIPS, 2024, pp. 71 222–71 241.

[56] P. Chen, Y. ZHANG, Y. Cheng, Y. Shu, Y. Wang, Q. Wen, B. Yang, and C. Guo, “Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting,” in Proc. of ICLR, 2023.

[57] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Proc. of ICLR, 2023.

[58] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P. Chen, Y. Liang, Y. Li, S. Pan, and Q. Wen, “Time-LLM: Time series forecasting by reprogramming large language models,” in Proceedings of the 12th International Conference on Learning Representations (ICLR’24). OpenReview.net, 2024.

![](images/a0308cae52a2828adc2b2bbd1f39f57a8df4866fab3f9dff47cbb5a1e2faefeb.jpg)  
Xiaohan Jiang is a PhD student at the School of Computer Science and Engineering, Beihang University, China. Her research interests include natural language processing, time series analysis, and interpretable machine learning.

![](images/dcd45a6f7ddcc83497bca44d827413f60cf4ba4f8e73614ff16967f7c4ae5668.jpg)

Jingyuan Wang received his Ph.D. degree from the Department of Computer Science and Technology at Tsinghua University. He is currently a Professor at the School of Computer Science and Engineering and the School of Economics and Management, Beihang University. He is also the head of the Beihang Interest Group on SmartCity (BIGSCity). His research interests include data mining and machine learning, with a particular focus on smart cities and spatiotemporal data analytics. He has received several prestigious academic honors and awards, including the NSFC Distinguished Young Scholar, the NSFC for Excellent Young Scholar, the Beijing Young Scholar Award, and the First Prize of the Ministry of Education’s Technological Invention Award.

![](images/26e728236b3b22ab6e57ab3726b8f7abf68a1052b3bc13de6d46a2ea5fa6abe0.jpg)

Jiahao Ji is a Ph.D. candidate at the School of Computer Science and Engineering, Beihang University. He received his B.S from Beihang University in 2019. His research interests include spatio-temporal data mining, interpretable machine learning, and urban computing.

![](images/5226dc8678f09203f7d902ab3e361af0d5d8e39b8a4d08aa0404d12e07df50ca.jpg)

![](images/1bcf4de5e83cb25c5291b552f3d5aee9f640ae9d5de9f7b7476e186a04c5c3be.jpg)

![](images/ed5bbefc8991415f39cf236e4ecf7befca8d6999e3a2dc2db8aeb269ef2d4afc.jpg)  
Beijing Universities.

Yongyao Wang is a master student at the School of Computer Science and Engineering, Beihang University. His research interests spatiotemporal data mining.

Chen Yang is a Ph.D. candidate at the School of Computer Science and Engineering, Beihang University. He received his master’s degree from Peking University in 2020. He received his bachelor’s degree from Sun Yat-sen University in 2017. His research interests include time series analysis and machine learning in economics.

Junjie Wu received his Ph.D. degree in Management Science and Engineering from Tsinghua University, from where he also holds a B.E. degree in Civil Engineering. He is currently a full professor and the Dean of the School of Economics and Management, Beihang University. His general area of research is machine learning, with applications in social, urban and financial computing. He is the recipient of the grant of NSFC Distinguished Young Scholars and the grant of Outstanding Young Scientist Program of