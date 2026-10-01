# A Time-Aware Bag-of-Receptive-Fields for Interpretable Irregular Time Series Classification

Francesco Spinnato<sup>1,2[0000−0002−3203−6716]</sup>

<sup>1</sup> Department of Computer Science, University of Pisa, Italy francesco.spinnato@unipi.it <sup>2</sup> ISTI-CNR, Pisa, Italy

Abstract. Irregular time series, characterized by non-uniform sampling intervals, missing observations, and variable lengths, are ubiquitous in healthcare, mobility, and environmental monitoring, yet efective and interpretable classifiers for this setting are limited. Existing approaches often rely on imputation, which can obscure the temporal structure of the data, or require complex neural architectures that are opaque and dificult to explain. In this work, we extend the Bag-Of-Receptive-Fields (BORF), a fast, deterministic, and interpretable transform for time series, to the irregular setting. Our key contribution is a time-weighted normalization scheme in which each observation is weighted proportionally to its associated time delta, making pattern extraction sensitive to the actual temporal distribution of samples rather than only their index position. This requires deriving an eficient sliding-window recurrence for the time-weighted standard deviation, preserving the linear time complexity of BORF. We benchmark the resulting method against stateof-the-art irregular time series classifiers on datasets from the pyrregular repository, demonstrating competitive classification performance with the added benefit of human-interpretable explanations.

Keywords: irregular time series · time series classification · bag-ofpatterns · explainable AI · symbolic aggregate approximation

## 1 Introduction

Temporal data are central to domains such as healthcare, mobility analytics, environmental monitoring, and industrial sensing [16]. In practice, however, such data are rarely collected on a perfectly regular grid. Sensors may operate at diferent frequencies, measurements may cover diferent time spans, and failures or acquisition policies may produce gaps and missing values [7]. The resulting irregular time series exhibit non-uniform sampling intervals, partial observation, and variable lengths. These properties afect the meaning of local temporal patterns and the validity of explanations produced by a classifier.

Time series classification (TSC) is well established for regularly sampled data, with mature repositories and large empirical comparisons across model families [1,13]. The irregular setting is less standardized: many studies rely on narrow benchmarks or artificially remove observations from regular series [21,14]. A recent benchmark [18] addresses this gap by evaluating irregular TSC methods under a unified representation and experimental protocol. Its results suggest that simple generalist methods originally designed for regular data, including rocket [5] and borf [17], can be competitive with purpose-built irregular classifiers, even when timestamps are not explicitly exploited.

This motivates revisiting symbolic and dictionary-based methods for irregular TSC. Such methods represent a time series through counts of local symbolic patterns rather than through latent neural states or randomized feature maps [2,10]. The Bag-Of-Receptive-Fields transform, borf, follows this line by constructing a deterministic sparse representation from symbolic descriptions of local receptive fields [17]. Its features are explicit, countable, and associated with concrete local patterns, making the representation naturally more inspectable than many black-box alternatives. The original borf formulation, however, is designed for regularly sampled data. Its local summaries and normalizations are based on observation order, implicitly assuming that consecutive observations have equal temporal support. This assumption breaks down under irregular sampling. A dense burst of observations over a short interval can dominate local statistics, while a sparse region spanning a longer interval can be underrepresented. Uniform local averaging therefore summarizes observations rather than elapsed time, discarding information carried by the sampling structure itself.

We present i-borf (Irregular Bag-Of-Receptive-Fields), an extension of borf for irregular time series. The central idea is to preserve the symbolic, deterministic, and interpretable structure of borf, while replacing its index-based local statistics with time-weighted aggregation and normalization. This allows the symbolic representation to reflect temporal support rather than only observation order, without first imputing missing values or resampling the data onto a regular grid. To make this eficient, we derive the sliding-window weighted variance update required by the transform. Online weighted variance updates and removal by negative weights have been studied previously [23], and Welford-style estimators are classical for expanding windows [22]; here, we specialize the recurrence to moving receptive fields with one incoming and one outgoing weighted observation.

Our contributions are as follows. (1) We extend SAX and the Bag-Of-Receptive-Fields transform to irregular time series through time-weighted aggregation and normalization. (2) We derive the sliding-window weighted variance update used by i-borf, enabling eficient normalization over moving receptive fields with non-uniform temporal supports. (3) We evaluate i-borf on irregular TSC datasets, comparing it against state-of-the-art irregular classifiers. (4) We present explainability examples for irregular TSC by tracing influential symbolic features back to timestamped temporal regions.

The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3 recaps the borf framework and irregular TSC setting. Section 4 presents i-borf. Section 5 reports experiments. Section 6 concludes.

## 2 Related Work

Irregular Time Series Classification. Irregular time series classification has been addressed by several specialised modelling paradigms [20]. These include continuous-time models based on neural ODEs and CDEs [8], recurrent architectures with time-decay or missingness mechanisms [4,3], graph-based models for inter-sensor dependencies under missingness [24], and attention-based models with continuous-time positional encodings or imputation-oriented self-attention [15,6]. While efective for specific irregularity patterns, these methods often require substantial model selection and remain dificult to interpret [18]. In contrast, recent TSC bake-ofs have highlighted the strength of generalist models, designed to perform robustly across heterogeneous datasets with limited task-specific tuning [13]. Kernel-based transforms such as rocket exemplify this trend, combining strong empirical performance with simple downstream classifiers [5]. The Bag-of-Receptive-Fields (borf) follows the same generalist philosophy, replacing random convolutional features with a deterministic dictionary-based representation built from local receptive fields [17]. A recent benchmark on irregular TSC [18] showed that these models remain surprisingly competitive even when originally designed for regular data, with rocket emerging as the strongest overall method and borf among the best-performing alternatives. This work builds on that evidence by extending borf to ITS, preserving its generalist and interpretable character while incorporating time-aware operations suited to irregular sampling.

Dictionary-Based Time Series Classification. Dictionary-based methods discretize time series into symbolic words and build a bag-of-patterns representation [2]. The Symbolic Aggregate approXimation (SAX) [10] and its window-wise extension underpin many such approaches, including bop [11], mr-seql [9], and borf [17]. Because these methods extract features over many overlapping windows, their scalability depends on eficient updates of local statistics, from fast Fourier updates in spectral dictionaries to stable online mean and variance estimators such as Welford’s method and weighted extensions thereof [22,23]. All of these methods assume regularly-sampled, fixed-length inputs; to the best of our knowledge, our work is the first to extend the SAX-based dictionary paradigm to ITS by replacing uniform aggregation with time-weighted aggregation.

Explainability for Time Series. Explainability for time series is a vast research area [19]. We focus on the line most directly connected to borf: SHAP-based attribution over interpretable symbolic representations. While SHAP [12] can be applied to any TSC model, explanations over raw timestamps or learned representations are often computationally costly and dificult to relate to recurring temporal patterns. borf [17] addresses this by combining pattern frequency counts with a linear classifier, yielding local saliency maps and global pattern importance scores. For ITS, comparable explanation mechanisms remain largely unexplored, since irregular sampling and missing observations complicate temporal attribution. i-borf inherits the interpretable feature space of borf and extends its SHAP-based explanation framework to irregular time series.

## 3 Background

In this section, we present all the necessary concepts to understand our proposal.   
We begin by defining an irregular time series data.

Definition 1 (Irregular Time Series Signal). An irregular signal is a sequence ofm observations each paired with a timestamp, $\mathbf { x } = [ ( x _ { 1 } , t _ { 1 } ) , \ldots , ( x _ { m } , t _ { m } ) ]$ ， where $t _ { 1 } < \cdots < t _ { m }$ and the intervals $t _ { i + 1 } - t _ { i }$ are not necessarily constant.

Definition 2 (Irregular Time Series). An irregular time series is a collection of c potentially irregular time series signals, $\mathbf X = \{ \mathbf x _ { 1 } , \dots , \mathbf x _ { c } \}$

Definition 3 (Irregular Time Series Dataset). An irregular time series dataset $\mathcal { X } = \{ \dot { \mathbf { X } } _ { 1 } , \dots , \mathbf { X } _ { n } \}$ is a collection of n irregular time series.

Irregularity can manifest in diferent ways, such as uneven sampling, partial observation (missing values), and raggedness (sampling, length or alignment mismatches) [18], which can afect downstream tasks, in our setting, classification.

Definition 4 (Irregular Time Series Classification). Given an irregular dataset X , Irregular TSC is the task of training a model f to predict a categorical label y for each input time series X, i $\mathbf { \mu } . e . , f ( \mathcal { X } ) = [ f ( \mathbf { X } _ { 1 } ) , \ldots , f ( \mathbf { X } _ { n } ) ] = \hat { \mathbf { y } } \in \mathbb { N } ^ { n }$

## 3.1 Bag-Of-Receptive-Fields

borf [17] is a deterministic and interpretable transform for regular time series classification. It extends the Bag-Of-Patterns representation [2] by replacing contiguous sliding windows with receptive $~ f i e l d s ,$ and by combining this more flexible pattern extraction with the Symbolic Aggregate approXimation (SAX) pipeline [10]. As in Bag-of-Words models for text, the final representation does not store the raw subsequences themselves, but the frequency with which symbolic patterns occur in each signal, as shown in Figure 1.

Let $\mathbf { x } = [ x _ { 1 } , \dots , x _ { m } ]$ be a univariate regularly sampled signal. For a window size w, dilation d, and stride s, the i-th receptive field is the ordered subsequence $[ x _ { i } , x _ { i + d } , \ldots , x _ { i + d ( w - 1 ) } ]$ . The dilation controls the spacing between consecutive observations inside the receptive field, allowing the method to capture patterns at diferent temporal resolutions, while the stride controls the distance between consecutive receptive fields.

![](images/83829c5eb8503fbcc013d0d95d235ac88acb63e3dcee581627a3557cc7bf86df.jpg)  
Fig. 1. Bag-of-Receptive-Fields toy representation for two time series.

Each receptive field is then converted into a SAX word. First, the receptive field is partitioned into l non-overlapping segments of equal size $q = w / l$ . Second, Piecewise Aggregate Approximation (PAA)

![](images/ca1017f2de165ea3d37a75b6ecd609b8efcbeddb6ddcece21de0ddf64702503a.jpg)  
Fig. 2. Naive PAA diagram used in the classical Bag-of-Patterns. The time series is divided into sliding windows, each window is segmented, and the average is taken.

is applied by computing the mean of each segment. This produces, for every receptive field, a vector of l segment means (Figure 2). Third, segment means are normalized with respect to the mean and standard deviation of the enclosing receptive field. For the i-th receptive field and its j-th segment mean $\mu _ { i , j } \colon$

$$
\mu _ { i , j } ^ { * } = \frac { \mu _ { i , j } - \mu _ { i } ^ { \mathrm { w i n } } } { \sigma _ { i } ^ { \mathrm { w i n } } } ,
$$

where $\mu _ { i } ^ { \mathrm { w i n } }$ and $\sigma _ { i } ^ { \mathrm { w i n } }$ are the mean and standard deviation of the full receptive field. Finally, the normalized values are discretized using Gaussian breakpoints into an alphabet of size $\alpha ,$ yielding a symbolic word of length l. Thus, each receptive field is mapped from a numerical subsequence to a SAX word. After discretization, each SAX word is hashed and identical words are counted. The final representation is a sparse contains for each time series the frequency of appearance of each symbolic receptive field (Figure 1).

Because each feature corresponds to the count of a specific symbolic pattern in a specific signal, the representation remains interpretable. In the original borf pipeline, a linear classifier such as ridge regression can be trained on the resulting sparse representation, and feature attributions can be mapped back to symbolic words and their temporal locations. This makes it possible to identify which patterns contribute to a prediction and where they occur in the original signal.

A central contribution of borf is the eficient computation of window-wise SAX words. Instead of recomputing each segment mean independently, which can be quadratic in the signal length, borf precomputes moving averages for the segment size q. PAA segment means, window means, and window standard deviations are then obtained by lookup, so the cost depends only on the number of receptive fields and the word length.

The limitation is that SAX normalization assumes regularly sampled observations: receptive fields are indexed by position, segment means are unweighted, and window statistics ignore timestamps. We target this step specifically: the borf pipeline is retained, while regular SAX normalization is replaced with a time-aware normalization for irregular time series.

## 4 Irregular Bag-Of-Receptive-Fields

We introduce i-borf (Irregular Bag-Of-Receptive-Fields), an extension of borf for irregular time series. The method modifies the PAA and normalization stages of the original pipeline by replacing uniform means and standard deviations with time-weighted counterparts, computed eficiently through a sliding-window recurrence for time-weighted statistics. All subsequent steps, namely receptivefield extraction, symbolic discretization, hashing, and counting, follow the borf pipeline described in Section 3 and are therefore not repeated here.

## 4.1 Time-Delta Weights

When observations are non-uniformly spaced, a plain average over a receptive field treats a densely-sampled interval and a sparsely-sampled one identically, ignoring the information carried by the timestamps. A densely-sampled observation covers only a brief moment in time, while a sparsely-sampled one represents a longer interval and should contribute proportionally more to any temporal aggregate. We assign to each observation $x _ { i }$ a weight equal to the time elapsed since the previous observation:

$$
\delta _ { i } = t _ { i } - t _ { i - 1 } , \quad i > 1 .\tag{1}
$$

Intuitively, $\delta _ { i }$ approximates the length of the time interval represented by $x _ { i } .$ making the weighted mean a discretization of a time integral over the signal. Equation (1) is undefined for $i = 1$ , as the first observation has no predecessor. Several boundary conditions are reasonable in practice. In our implementation, we set $\delta _ { 1 } = \delta _ { 2 }$ , assigning the first observation the same temporal support as the second. Investigating the sensitivity to alternative boundary conventions is left for future work. When all $\delta _ { i }$ are equal, the weights cancel and every time-weighted quantity defined below reduces to its uniform counterpart in borf.

## 4.2 Time-Weighted Piecewise Aggregate Approximation

Let x be an irregular signal with weights $[ \delta _ { 1 } , \ldots , \delta _ { m } ]$ defined by Equation (1). For the receptive field starting at index i, with window size w, dilation d, stride s, word length l, and segment size $q = w / l ,$ , let $a _ { i j k } = 1 + ( i - 1 ) s + ( j - 1 ) d q + ( k - 1 ) d$ denote the signal index of the k-th element of segment $j$ in window i. The time-weighted segment mean for segment $j$ is then:

$$
\hat { \mu } _ { i , j } = \frac { \displaystyle \sum _ { k = 1 } ^ { q } \delta _ { a _ { i j k } } x _ { a _ { i j k } } } { \displaystyle \sum _ { k = 1 } ^ { q } \delta _ { a _ { i j k } } } .\tag{2}
$$

This is a time-aware generalization of the PAA mean: if all $\delta _ { i }$ are equal, it reduces to the original uniform mean of borf.

## 4.3 Time-Weighted Standardization

Each segment mean $\hat { \mu } _ { i , j }$ is standardized by the weighted mean and standard deviation of its enclosing window:

$$
\hat { \mu } _ { i , j } ^ { * } = \frac { \hat { \mu } _ { i , j } - \hat { \mu } _ { i } ^ { \mathrm { w i n } } } { \hat { \sigma } _ { i } ^ { \mathrm { w i n } } } ,\tag{3}
$$

where $\hat { \mu } _ { i } ^ { \mathrm { w i n } }$ and $\hat { \ \sigma } _ { i } ^ { \mathrm { w i n } }$ are the weighted mean and standard deviation over all w observations in window i, with possibly dilated indices $i , i + d , \ldots , i + d ( w - 1 )$ Let $\begin{array} { r } { W _ { i } ^ { \mathrm { w i n } } = \sum _ { k = 0 } ^ { w - 1 } \delta _ { i + k d } } \end{array}$ be the sum of the window weights, then:

$$
\hat { \mu } _ { i } ^ { \mathrm { w i n } } = \frac { 1 } { W _ { i } ^ { \mathrm { w i n } } } \sum _ { k = 0 } ^ { w - 1 } \delta _ { i + k d } x _ { i + k d } ,\tag{4}
$$

$$
( \hat { \sigma } _ { i } ^ { \mathrm { w i n } } ) ^ { 2 } = \frac { 1 } { W _ { i } ^ { \mathrm { w i n } } } \sum _ { k = 0 } ^ { w - 1 } \delta _ { i + k d } \big ( x _ { i + k d } - \hat { \mu } _ { i } ^ { \mathrm { w i n } } \big ) ^ { 2 } .\tag{5}
$$

In a regular signal all $\delta _ { i }$ are equal and Equations (3) to (5) reduce exactly to the uniform z-score of the original borf. In the irregular case, the weighting reduces the influence of densely sampled observations within each receptive field, with receptive-field boundaries, dilation, stride, and counts remaining index-based.

## 4.4 Sliding-Window Weighted Standard Deviation

The main computational issue introduced by time-weighted normalization is the repeated computation of weighted means and standard deviations over overlapping windows. Computing these quantities independently for each of the windows takes $O ( m ^ { 2 } )$ in the worst case. A recurrence that updates the weighted mean and variance in O(1) per window slide reduces the total cost to $O ( m )$ ; Welford-style algorithms [22] achieve this while remaining numerically stable, avoiding the catastrophic cancellation that plagues the naïve two-pass formula in an online setting [22]. We therefore derive a sliding-window weighted variance recurrence. Conceptually, the recurrence extends online variance updates in the spirit of [23], but difers from them because the window both receives a new observation and removes an old one at every step. Let $\begin{array} { r } { W _ { i } = \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } } \end{array}$ be the total weight of the window $[ x _ { i } , \ldots , x _ { i + w - 1 } ]$ , Sliding the window by one step removes $x _ { i }$ (weight $\delta _ { i } )$ and adds $x _ { i + w }$ (weight $\delta _ { i + w } )$ . The weighted mean update follows directly from the definition:

$$
\hat { \mu } _ { i + 1 } = \frac { W _ { i } \hat { \mu } _ { i } - \delta _ { i } x _ { i } + \delta _ { i + w } x _ { i + w } } { W _ { i + 1 } } .\tag{6}
$$

Define now the weighted sum of squared deviations from the mean:

$$
S _ { i } = \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } ( x _ { r } - \hat { \mu } _ { i } ) ^ { 2 } = W _ { i } \cdot ( \hat { \sigma } _ { i } ) ^ { 2 } ,\tag{7}
$$

the variance update is the following:

Proposition 1 (Sliding-window weighted variance update).

$$
\begin{array} { r } { { S _ { i + 1 } } = S _ { i } + \underbrace { { \delta _ { i + w } { { \left( { { x _ { i + w } } - { \hat { \mu } } _ { i } } \right) } { { \left( { { x _ { i + w } } - { \hat { \mu } } _ { i + 1 } } \right) } } } } } _ { { i n c o m i n g } } - \underbrace { { { \delta _ { i } } \left( { { x _ { i } } - { \hat { \mu } } _ { i } } \right) } { { \left( { { x _ { i } } - { \hat { \mu } } _ { i + 1 } } \right) } } } _ { o u t g o i n g } , } \end{array}\tag{8}
$$

and thus $\hat { \sigma } _ { i + 1 } = \sqrt { S _ { i + 1 } / W _ { i + 1 } }$

The proof is presented in Appendix A. This procedure assumes dilation $d = 1$ ; dilation $d > 1$ can be handled by applying the recurrences independently to each of the d interleaved sub-sequences $[ x _ { r } , x _ { r + d } , x _ { r + 2 d } , . ~ . ~ . ]$ for $r \in \{ 1 , \ldots , d \}$ Computing all weighted moving averages and sliding-window standard deviations via Equation $( 6 )$ and Proposition 1 requires $O ( m )$ per sub-sequence, hence $O ( m )$ overall for any fixed $d ,$ identical to the original borf. Indeed, the additional weighted updates require constant time per window shift and therefore do not change the $O ( m )$ asymptotic complexity.

## 5 Experiments

Datasets. We evaluate our proposal on 15 datasets provided by the pyrregular framework [18], covering health, mobility, human activity recognition, and sensor domains, using its default preprocessing and settings. We select all the datasets that exhibit missing values, uneven sampling, or ragged sampling, since these irregularity types directly afect the computation of local statistics and can therefore benefit from time-aware normalization. In contrast, datasets characterized only by unequal length or temporal shift are excluded, as these irregularities do not by themselves require modifying the SAX normalization step. We use the default train/test splits and report macro-averaged F1, which is robust to class imbalance. Code is available at https: $/ / { \tt g } \dot { \bf 1 }$ thub.com/fspinna/borf.

Competitors. We benchmark i-borf against reference results of state-ofthe-art models from [18]. i-borf uses the same heuristic as borf [17]: w ∈ $[ 2 ^ { 2 } , \ldots , 2 ^ { \lfloor \log _ { 2 } m \rfloor } ]$ , dilations $d \in [ 2 ^ { 0 } , \dots , 2 ^ { \lfloor \log _ { 2 } \log _ { 2 } m \rfloor } ]$ , word lengths $l \in \{ 2 , 4 , 8 \}$ , alphabet size $\alpha = 3$ , stride $s = 1$ , and threshold $\theta = 0 . 1 5$ . However, diferently from the classical borf implementation, i-borf uses Ridge instead of the classical lgbm. Thus, to isolate the efect of the proposed weighting scheme, we also include the classical borf with a Ridge head (borf+rc).

## 5.1 Results

The CD plots in Figure 3 provide a global comparison across datasets. For macro-F1, i-borf is the clear leading method: it obtains the best average rank, 3.20, and is not statistically tied to any competing approach under a pairwise one-sided Wilcoxon signed-rank test with Holm correction at $\alpha = 0 . 1$ . The closest method is borf+rc, with average rank 4.20, followed by rocket, rifc, and lgbm. For accuracy, i-borf also obtains the best average rank, 3.80, although the separation from the closest competitors is less pronounced. Overall, the rank-based comparison indicates indicates that the proposed transform gives the strongest aggregate performance, with an especially clear advantage in terms of macro-F1.

![](images/b8d84f06b1c8c5d736d90ef4037bffadfa9d1e2dd76d634c5976be8651079a08.jpg)  
Fig. 3. CD plot for the benchmarked models in terms of F1 and Accuracy. Best models to the right. Connected methods are not significantly diferent according to pairwise one-sided Wilcoxon signed-rank tests with Holm correction at $\alpha = 0 . 1$

Table 1. Average F1 score on the test set for each dataset and each classifier. Standard deviation is reported for highly stochastic methods. Missing values are due to exceeded memory or maximum runtime.
<table><tr><td></td><td>|I-BORF</td><td></td><td>BORF BORF+RC</td><td>BRITS</td><td>LGBM</td><td>RAINDROP</td><td>RIFC</td><td></td><td>ROCKET</td></tr><tr><td>ABF</td><td>0.90</td><td>0.17</td><td>0.33</td><td> $0 . 3 3 \pm 0 . 0 1$ </td><td>0.17</td><td> $0 . 2 7 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 0$ </td><td></td><td> $0 . 1 7 \pm 0 . 0 0$ </td></tr><tr><td>AN</td><td>0.84</td><td>0.80</td><td>0.84</td><td> $0 . 6 5 \pm 0 . 0 0$ </td><td>0.80</td><td> $0 . 6 4 \pm 0 . 0 5$ </td><td> $0 . 8 8 \pm 0 . 0 2$ </td><td></td><td> ${ \bf 0 . 9 0 \pm 0 . 0 4 }$ </td></tr><tr><td>DD</td><td>0.55</td><td>0.51</td><td>0.52</td><td> $0 . 5 2 \pm 0 . 0 2$ </td><td>0.52</td><td> $0 . 4 5 \pm 0 . 0 4$ </td><td> $0 . 4 9 \pm 0 . 0 4$ </td><td></td><td> $0 . 5 4 \pm 0 . 0 3$ </td></tr><tr><td>DG</td><td>0.81</td><td>0.34</td><td>0.82</td><td> $0 . 7 2 \pm 0 . 0 7$ </td><td>0.34</td><td> $0 . 6 0 \pm 0 . 2 2$ </td><td> $0 . 3 4 \pm 0 . 0 0$ </td><td></td><td> $0 . 3 4 \pm 0 . 0 0$ </td></tr><tr><td>DW</td><td>0.97</td><td>0.42</td><td>0.96</td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td>0.42</td><td> $0 . 7 8 \pm 0 . 3 1$ </td><td> $0 . 4 2 \pm 0 . 0 0$ </td><td></td><td> $0 . 4 2 \pm 0 . 0 0$ </td></tr><tr><td>GS</td><td>0.40</td><td>0.41</td><td>0.37</td><td></td><td>0.13</td><td></td><td> $0 . 0 7 \pm 0 . 0 2$ </td><td></td><td> $0 . 3 1 \pm 0 . 1 5$ </td></tr><tr><td>LPA</td><td>0.96</td><td>0.73</td><td>0.91</td><td> $0 . 2 8 \pm 0 . 2 0$ </td><td>0.53</td><td> $0 . 3 3 \pm 0 . 0 9$ </td><td> $0 . 3 2 \pm 0 . 2 0$ </td><td></td><td> $0 . 0 2 \pm 0 . 0 1$ </td></tr><tr><td>MI3</td><td>0.35</td><td>0.27</td><td>0.35</td><td> $0 . 4 2 \pm 0 . 1 1$ </td><td>0.41</td><td> $0 . 3 6 \pm 0 . 1 5$ </td><td> ${ \bf 0 . 5 6 \pm 0 . 2 2 }$ </td><td></td><td> $0 . 3 5 \pm 0 . 0 0$ </td></tr><tr><td>P12</td><td>0.57</td><td>0.51</td><td>0.56</td><td> $0 . 4 6 \pm 0 . 0 0$ </td><td>0.55</td><td> $0 . 5 6 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 6 3 \pm 0 . 0 1 }$ </td><td></td><td> $0 . 4 7 \pm 0 . 0 1$ </td></tr><tr><td>P19</td><td>0.66</td><td>0.71</td><td>0.66</td><td> $0 . 4 9 \pm 0 . 0 0$ </td><td>0.75</td><td> $0 . 6 9 \pm 0 . 0 1$ </td><td> $0 . 6 6 \pm 0 . 0 3$ </td><td></td><td> $0 . 7 1 \pm 0 . 0 1$ </td></tr><tr><td>PA2</td><td>0.75</td><td>0.53</td><td>0.75</td><td></td><td>0.33</td><td></td><td> $0 . 3 7 \pm 0 . 3 2$ </td><td></td><td> $0 . 6 6 \pm 0 . 1 0$ </td></tr><tr><td>PGE</td><td>0.78</td><td>0.40</td><td>0.78</td><td> $\mathbf { 0 . 7 8 \pm 0 . 0 0 }$ </td><td>0.40</td><td> $0 . 4 8 \pm 0 . 2 6$ </td><td> $0 . 4 0 \pm 0 . 0 0$ </td><td></td><td> $0 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>SE</td><td>0.63</td><td>0.47</td><td>0.56</td><td> $0 . 4 8 \pm 0 . 1 5$ </td><td>0.42</td><td> $0 . 4 0 \pm 0 . 1 5$ </td><td> ${ \bf 0 . 8 2 \pm 0 . 0 4 }$ </td><td></td><td> $0 . 8 0 \pm 0 . 0 8$ </td></tr><tr><td>TA</td><td>0.44</td><td>0.42</td><td>0.44</td><td> $0 . 2 3 \pm 0 . 0 0$ </td><td>0.77</td><td> $0 . 2 5 \pm 0 . 0 2$ </td><td> $0 . 3 8 \pm 0 . 0 6$ </td><td></td><td> $0 . 5 7 \pm 0 . 0 1$ </td></tr><tr><td>VE</td><td>0.96</td><td>0.97</td><td>0.92</td><td> $0 . 5 0 \pm 0 . 0 8$ </td><td>0.94</td><td> $0 . 6 5 \pm 0 . 0 2$ </td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td></td><td> $0 . 9 4 \pm 0 . 0 2$ </td></tr></table>

The per-dataset results in Table 1 confirm that the improvement is not driven by a single dataset. i-borf obtains the best or tied-best F1 score on 6 out of 15 datasets, and improves over the original borf on 12 datasets. The comparison with borf+rc isolates the role of the classifier. On several datasets, borf+rc already improves substantially over the original borf configuration, indicating that the linear Ridge head is a strong and stable choice for the resulting sparse representation. Nevertheless, i-borf improves over borf+rc on 8 datasets, ties on 6, and is worse on only one. This suggests that the gains are not only due to the classifier, but also to the proposed time-aware normalization. Compared with competitor irregular time-series models, such as rifc, rocket, and lgbm performance is more uniform. Indeed while the latter obtain the best score on some datasets, for instance MI3, SE, TA, and P19 their performance is less uniform across the benchmark, whereas i-borf provides strong average ranks.

![](images/aa780e089b6594d02bf59d5638db40238eb4f17136f6df96e61e26f6ed8828d4.jpg)

Fig. 4. Three examples of instances from the ABF dataset, from left to right, Alembic, Bowl, and Flask.  
![](images/922e4b7d6927182f1f4a55f818a521da064536231b8332603e71f92d259d1a2a.jpg)  
Fig. 5. Local explanations on one Alembic (top) and one Flask (bottom) instance. To the left of each plot is the time series, colored based on the importance of each observation. To the right the medoid shape of the most important not contained pattern.

## 5.2 Explainability

A key advantage of i-borf over all other ITS classifiers considered here is the availability of human-interpretable explanations. We demonstrate this on the Alembics-Bowls-Flasks dataset (ABF) [18]. There are three classes, which are Alembics, Bowls, and Flasks, and difer by how much the temporal axis is skewed, i.e., if it has positive (Alembic), negative (Flask), or no skewness (Bowl), as shown in Figure 4. We compute SHAP values for a ridge classifier trained on the i-borf representation and build a saliency map by projecting pattern importances back to their temporal locations in the original signal, following [17].

Local explanation. Figure 5 shows local explanations for one Alembic instance and one Flask instance. In both cases, the left panel reports the original time series, with observations colored according to their SHAP contribution, while the right panel shows the medoid of the most important pattern that is not contained in the instance. For the Alembic instance, the strongest contribution is concentrated in the early-to-middle part of the series, where the signal follows a fast decreasing trajectory before reaching its minimum. The most important not-contained pattern is also decreasing, but slowly, more characteristic of the opposite skew (associated with Flasks). For the Flask instance, the relevant region is shifted toward the later part of the series, where the curve rises after a pronounced valley. In this case, the most important not-contained pattern is slowly increasing, closer to Alembic instances.

![](images/ee207f842315d3c5ea00c2a21de858f3cfb8c124abdfeacb70e43d333467a30e.jpg)  
Fig. 6. Global explanations on the ABF dataset. Left: all temporal alignments of the receptive fields associated with the most important symbolic patterns. Right: pattern frequency by class, with points colored according to their SHAP value.

Global explanation. Figure 6 provides a dataset-level view of the top-3 most important patterns. The left panels show the receptive fields associated with the selected symbolic patterns, in all their possible alignments in the training set, while the right panels report how often each pattern appears in the three classes, with color indicating its SHAP value. The visualization makes the class structure explicit. Patterns associated with Alembics tend to correspond to slowly decreasing receptive fields, while patterns associated with Flasks exhibit the opposite behavior. Bowls occupy an intermediate regime: their relevant patterns are less skewed and appear between the two extremes represented by Alembics and Flasks. The frequency plots further show that the learned representation is not only separating the classes through raw occurrence counts, but also through the direction and magnitude of the induced classifier contribution. Patterns that occur frequently in one class and receive consistently positive SHAP values provide global evidence for that class, whereas absent or negatively weighted patterns provide evidence against it. This illustrates that i-borf retains the interpretability mechanism of borf while extending it to irregular time series: explanations can still be read both locally, as salient temporal regions in a single instance, and globally, as class-specific receptive-field shapes.

## 6 Conclusions

We presented i-borf, an extension of borf to irregular time series that introduces time-weighted aggregation and normalization into the symbolic transformation. This makes the transform aware of the temporal distribution of samples without imputation, while preserving the linear complexity, determinism, and interpretability of borf. Its main technical contribution is a sliding-window weighted standard deviation recurrence, which extends the moving-average speedup of borf and Welford-style updates to weighted sliding windows. Experiments on the pyrregular benchmark show that time-weighting improves over the unweighted baseline. Moreover, i-borf retains the explanation framework of borf, supporting local saliency maps and global pattern-importance analyses, a capability absent from the competing ITS classifiers considered here.

A limitation is that i-borf introduces time awareness only in segment and window statistics, while receptive-field construction and word counts remain index-based. Moreover, time-delta weighting assumes that each observation represents the preceding interval, which may be inappropriate for point events or clinician-triggered measurements, and does not explicitly model partial observation. Future work will study alternative weighting conventions, the efects of diferent irregularity types, continuous-time receptive fields, and the quantitative quality and stability of explanations.

Acknowledgements. This study has been partially funded by the Italian Project Fondo Italiano per la Scienza FIS00001966 “MIMOSA”, 101120763 “TANGO”, by the European Commission under the NextGeneration EU programme, “SoBig-Data.it – Strengthening the Italian RI for Social Mining and Big Data Analytics” – Prot. IR0000013 – Av. n. 3264 del 28/12/2021.

## References

1. Bagnall, A., Lines, J., Bostrom, A., Large, J., Keogh, E.: The great time series classification bake of: a review and experimental evaluation of recent algorithmic advances. Data mining and knowledge discovery 31, 606–660 (2017)

2. Baydogan, M.G., Runger, G., Tuv, E.: A bag-of-features framework to classify time series. IEEE transactions on pattern analysis and machine intelligence 35(11), 2796–2802 (2013)

3. Cao, W., Wang, D., Li, J., Zhou, H., Li, L., Li, Y.: Brits: Bidirectional recurrent imputation for time series. Advances in neural information processing systems 31 (2018)

4. Che, Z., Purushotham, S., Cho, K., Sontag, D., Liu, Y.: Recurrent neural networks for multivariate time series with missing values. Scientific reports 8(1), 6085 (2018)

5. Dempster, A., Petitjean, F., Webb, G.I.: Rocket: exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery 34(5), 1454–1495 (2020)

6. Du, W., Côté, D., Liu, Y.: Saits: Self-attention-based imputation for time series. Expert Systems with Applications 219, 119619 (2023)

7. Harvey, A., Koopman, S.J., Penzer, J.: Messy time series: a unified approach. Advances in econometrics 13, 103–144 (1998)

8. Kidger, P., Morrill, J., Foster, J., Lyons, T.: Neural controlled diferential equations for irregular time series. Advances in Neural Information Processing Systems 33, 6696–6707 (2020)

9. Le Nguyen, T., Gsponer, S., Ilie, I., O’reilly, M., Ifrim, G.: Interpretable time series classification using linear models and multi-resolution multi-domain symbolic representations. Data mining and knowledge discovery 33, 1183–1222 (2019)

10. Lin, J., Keogh, E., Wei, L., Lonardi, S.: Experiencing sax: a novel symbolic rep resentation of time series. Data Mining and knowledge discovery 15, 107–144 (2007)

11. Lin, J., Khade, R., Li, Y.: Rotation-invariant similarity in time series using bag-ofpatterns representation. Journal of Intelligent Information Systems 39, 287–315 (2012)

12. Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. Advances in neural information processing systems 30 (2017)

13. Middlehurst, M., Schäfer, P., Bagnall, A.: Bake of redux: a review and experimental evaluation of recent time series classification algorithms. Data Mining and Knowledge Discovery (Apr 2024)

14. Mitra, R., McGough, S.F., Chakraborti, T., Holmes, C., Copping, R., Hagenbuch, N., Biedermann, S., Noonan, J., Lehmann, B., Shenvi, A., et al.: Learning from data with structured missingness. Nature Machine Intelligence 5(1), 13–23 (2023)

15. Shukla, S.N., Marlin, B.M.: Multi-time attention networks for irregularly sampled time series. In: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net (2021), https:// openreview.net/forum?id=4c0J6lwQ4\_

16. Shumway, R.H., Stofer, D.S.: Time series analysis and its applications, vol. 3. Springer (2000)

17. Spinnato, F., Guidotti, R., Monreale, A., Nanni, M.: Fast, interpretable and deterministic time series classification with a bag-of-receptive-fields. IEEE Access (2024)

18. Spinnato, F., Landi, C.: PYRREGULAR: A unified framework for irregular time series, with classification benchmarks. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=qetBM8nLkf

19. Theissler, A., Spinnato, F., Schlegel, U., Guidotti, R.: Explainable ai for time series classification: a review, taxonomy and research directions. Ieee Access 10, 100700–100724 (2022)

20. Wang, J., Du, W., Cao, W., Zhang, K., Wang, W., Liang, Y., Wen, Q.: Deep learning for multivariate time series imputation: A survey. arXiv preprint arXiv:2402.04059 (2024)

21. Weerakody, P.B., Wong, K.W., Wang, G., Ela, W.: A review of irregular time series data handling with gated recurrent neural networks. Neurocomputing 441, 161–178 (2021)

22. Welford, B.P.: Note on a method for calculating corrected sums of squares and products. Technometrics 4(3), 419–420 (1962)

23. West, D.: Updating mean and variance estimates: An improved method. Communi cations of the ACM 22(9), 532–535 (1979)

24. Zhang, X., Zeman, M., Tsiligkaridis, T., Zitnik, M.: Graph-guided network for irregularly sampled multivariate time series. In: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net (2022), https://openreview.net/forum?id=Kwm8I7dU-l5

## A Proof of Proposition 1

Throughout, $\begin{array} { r } { W _ { i } = \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } } \end{array}$ denotes the total weight of the window $[ x _ { i } , \ldots , x _ { i + w - 1 } ]$ and $\begin{array} { r } { \hat { \mu } _ { i } = ( 1 / W _ { i } ) \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } x _ { \imath } } \end{array}$ <sub>r</sub> its weighted mean. We write $W _ { i + 1 } = W _ { i } - \delta _ { i } +$ $\delta _ { i + w }$

Proof. Define $S _ { i } = W _ { i } ( \hat { \sigma } _ { i } ) ^ { 2 }$ . Using the identity

$$
( \hat { \sigma } _ { i } ) ^ { 2 } = \frac { 1 } { W _ { i } } \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } ( x _ { r } - \hat { \mu } _ { i } ) ^ { 2 } = \frac { 1 } { W _ { i } } \left[ \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } x _ { r } ^ { 2 } \right] - \hat { \mu } _ { i } ^ { 2 } ,\tag{9}
$$

we obtain:

$$
S _ { i } = \left[ \sum _ { r = i } ^ { i + w - 1 } \delta _ { r } x _ { r } ^ { 2 } \right] - W _ { i } \hat { \mu } _ { i } ^ { 2 } .\tag{10}
$$

Computing $S _ { i + 1 } - S _ { i }$ and substituting Equation (10):

$$
S _ { i + 1 } - S _ { i } = \delta _ { i + w } x _ { i + w } ^ { 2 } - \delta _ { i } x _ { i } ^ { 2 } - W _ { i + 1 } \hat { \mu } _ { i + 1 } ^ { 2 } + W _ { i } \hat { \mu } _ { i } ^ { 2 } .\tag{11}
$$

Substituting $W _ { i } = W _ { i + 1 } - \delta _ { i + w } + \delta _ { i }$ into the last term and grouping:

$$
\begin{array} { r l } & { - W _ { i + 1 } \hat { \mu } _ { i + 1 } ^ { 2 } + W _ { i } \hat { \mu } _ { i } ^ { 2 } } \\ & { = W _ { i + 1 } ( \hat { \mu } _ { i } - \hat { \mu } _ { i + 1 } ) ( \hat { \mu } _ { i } + \hat { \mu } _ { i + 1 } ) + \delta _ { i } \hat { \mu } _ { i } ^ { 2 } - \delta _ { i + w } \hat { \mu } _ { i } ^ { 2 } . } \end{array}\tag{12}
$$

Multiplying $\hat { \mu } _ { i + 1 }$ (Equation (6)) through by $W _ { i + 1 }$ and rearranging, using $W _ { i + 1 } -$ $W _ { i } = \delta _ { i + w } - \delta _ { i } { \mathrm { : } }$

$$
\begin{array} { r l } {  { W _ { i + 1 } ( \hat { \mu } _ { i } - \hat { \mu } _ { i + 1 } ) = W _ { i + 1 } \hat { \mu } _ { i } - W _ { i } \hat { \mu } _ { i } + \delta _ { i } x _ { i } - \delta _ { i + w } x _ { i + w } } } \\ & { = ( \delta _ { i + w } - \delta _ { i } ) \hat { \mu } _ { i } + \delta _ { i } x _ { i } - \delta _ { i + w } x _ { i + w } } \\ & { = \delta _ { i } ( x _ { i } - \hat { \mu } _ { i } ) - \delta _ { i + w } ( x _ { i + w } - \hat { \mu } _ { i } ) . } \end{array}\tag{13}
$$

Substituting Equation (13) into Equation (12) and then into Equation (11):

$$
\begin{array} { r l } & { S _ { i + 1 } - S _ { i } = \delta _ { i + w } x _ { i + w } ^ { 2 } - \delta _ { i } x _ { i } ^ { 2 } } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } + \left[ \delta _ { i } ( x _ { i } - \hat { \mu } _ { i } ) - \delta _ { i + w } ( x _ { i + w } - \hat { \mu } _ { i } ) \right] \left( \hat { \mu } _ { i } + \hat { \mu } _ { i + 1 } \right) } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x } + \delta _ { i } \hat { \mu } _ { i } ^ { 2 } - \delta _ { i + w } \hat { \mu } _ { i } ^ { 2 } . } \end{array}\tag{14}
$$

Collecting terms in $\delta _ { i + w }$ and $\delta _ { i }$

$$
\begin{array} { r l } & { S _ { i + 1 } - S _ { i } = \delta _ { i + w } \big [ x _ { i + w } ^ { 2 } - ( x _ { i + w } - \hat { \mu } _ { i } ) ( \hat { \mu } _ { i } + \hat { \mu } _ { i + 1 } ) - \hat { \mu } _ { i } ^ { 2 } \big ] } \\ & { \qquad + \delta _ { i } \big [ - x _ { i } ^ { 2 } + ( x _ { i } - \hat { \mu } _ { i } ) ( \hat { \mu } _ { i } + \hat { \mu } _ { i + 1 } ) + \hat { \mu } _ { i } ^ { 2 } \big ] . } \end{array}\tag{15}
$$

Using the algebraic identities

$$
x ^ { 2 } - ( x - \mu _ { \mathrm { o l d } } ) ( \mu _ { \mathrm { o l d } } + \mu _ { \mathrm { n e w } } ) - \mu _ { \mathrm { o l d } } ^ { 2 } = ( x - \mu _ { \mathrm { o l d } } ) ( x - \mu _ { \mathrm { n e w } } )
$$

and

$$
- x ^ { 2 } + ( x - \mu _ { \mathrm { o l d } } ) ( \mu _ { \mathrm { o l d } } + \mu _ { \mathrm { n e w } } ) + \mu _ { \mathrm { o l d } } ^ { 2 } = - ( x - \mu _ { \mathrm { o l d } } ) ( x - \mu _ { \mathrm { n e w } } ) ,
$$

for the entering and leaving observations, respectively, gives Equation (8). ⊓⊔