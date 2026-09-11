# AUC MAXIMIZATION FROM BIASED POSITIVE-UNLABELED DATA WITH CONFIDENCE

Atsutoshi Kumagai, Tomoharu Iwata, Hiroshi Takahashi, Taishi Nishiyama, Kazuki Adachi, Yasuhiro Fujiwara   
NTT, Inc.   
atsutoshi.kumagai@ntt.com

## ABSTRACT

Maximizing the area under the receiver operating characteristic curve (AUC) is a standard approach to imbalanced binary classification. Although positive and negative data are required for maximizing the AUC, negative data are often difficult to collect in some real-world applications due to privacy concerns or the need for specialized expertise to annotate them. Thus, AUC maximization from positive and unlabeled (PU) data has been attracting attention. Existing methods assume that labeled positive data are unbiased samples from the true positive distribution. However, this ideal assumption is often violated in practice. In this paper, we propose a method to maximize the AUC from biased PU data. To address the bias, our key idea is to exploit confidence, i.e., the probability that an instance is positive, associated with the small number of labeled positive data. We derive an estimator of the AUC risk using biased PU data with confidence, enabling AUC maximization under such bias. We further show that the rewritten AUC risk induces a Bayes-optimal AUC ranking even when the available confidence is any strictly increasing transformation of the true posterior probability. We experimentally show the effectiveness of our method on eight real-world datasets.

## 1 INTRODUCTION

In many real-world binary classification tasks such as cyber security (Mirsky et al., 2018; Bagui & Li, 2021), medical care (Yang et al., 2021), product inspection (Park et al., 2016), and fraud detection (Su et al., 2021), class-imbalance frequently arises, where the amount of positive data is much smaller than that of negative data (Johnson & Khoshgoftaar, 2019). For such imbalanced data, classification accuracy, which is the standard performance metric in ordinary classification, is not a suitable measure (Yang & Ying, 2022). Instead, the area under the receiver operating characteristic curve (AUC) is widely used (Bradley, 1997; McDermott et al., 2024). The AUC represents the probability that a classifier will rank a positive instance higher than a negative one (Yang & Ying, 2022). Owing to the nature of the ranking, the AUC can adequately measure the classifier’s performance even with imbalanced data. Consequently, maximizing the AUC enables accurate classifiers to be learned from imbalanced data (Liu et al., 2020; Yuan et al., 2021a; Yang & Ying, 2022).

Although labeled positive and negative (PN) data are required for AUC maximization, negative data are often difficult to collect in practice. For example, in cyber security, malicious data (positive data) can be collected from public blocklists, but benign data of legitimate users (negative data) are often unavailable due to privacy concerns, and identifying benign data from given unlabeled data requires a high level of expertise (Mirsky et al., 2018; Kumagai et al., 2024; 2025a). In medical care, although positive patient data can be collected on the basis of confirmed diagnoses, treating all other data as negative is problematic since such data often include false-negatives or pre-symptomatic patients who have already contracted the disease (Bekker & Davis, 2020). Similar situations occur in other applications such as product inspection (Kumagai et al., 2024) and fraud detection (Su et al., 2021).

To address such situations, several studies have proposed methods to maximize the AUC from only positive and unlabeled (PU) data (Sakai et al., 2018; Xie & Li, 2018; Xie et al., 2024). These methods assume that labeled positive data are unbiased samples from the true positive distribution, which is known as the selected completely at random (SCAR) setting (Elkan & Noto, 2008). Although this ideal assumption enables an unbiased estimator of the AUC to be derived with PU data, it is often violated in practice since labeled positive data are often biased. For example, blocklists tend to contain certain (widely known) types of malicious data, and diseases that can be diagnosed are usually limited to well-understood cases.

(a)  
![](images/f561b8efc149501dd60a7921e7ec2435003383f50482a07523685bd4b1da4f47.jpg)  
(b)  
(c)  
(d)  
Figure 1: Illustrations of our problem setting and other related settings. Red, blue, and gray points are positive, negative, and unlabeled data, respectively. The dark/light red colors show high/low confidence values for positive data. The black line is illustrated to visually distinguish positive and negative data. (a) Standard AUC maximization with PN data. (b) AUC Maximization from PU data under the SCAR setting where labeled positive data are unbiased. (c) AUC Maximization from PU data under the SAR setting where labeled positive data are biased. Note that there are no studies for this setting. (d) Our problem setting: AUC maximization from PU data with positive-confidence under the SAR setting. In class-imbalanced settings, the number of labeled positive instances (each equipped with confidence) is small.

To cope with such biased situations, some PU learning studies for ordinary classification have recently considered the selected at random (SAR) setting, where the probability of selecting positive data to be labeled depends on their feature values (Kato et al., 2019; Teisseyre et al., 2025; Gerych et al., 2022). While this setting is more realistic, these methods are not designed for maximizing the AUC. In addition, to make the problem tractable, they require strong assumptions about data distributions or labeling mechanisms, such as the PN distributions being separated (Gerych et al., 2022) or the selection probability of positive data following some restricted function forms (Gerych et al., 2022; He et al., 2018; Teisseyre et al., 2025). When the assumption is violated, the performance of these methods drastically deteriorates.

To address this problem, we propose a method for maximizing the AUC under the SAR setting by using biased positive data equipped with confidence and unlabeled data, without such restrictive assumptions about data distributions or labeling mechanisms. Here, the confidence is formally defined as the probability that a given instance is positive, and we refer to this confidence for labeled positive data as positive-confidence. Figure 1 illustrates our problem setting.

Such confidence need not be collected specifically for the proposed method; in many applications, it is already produced as a by-product of existing labeling or operational processes. For example, when the same instance is labeled by multiple annotators to enhance reliability, the average of these labels can naturally form confidence (Ishida et al., 2023). Such multi-annotator labeling is used in various domains such as medical care, cyber security, and crowdsourcing (Le et al., 2023; Kantchelian et al., 2015; Tang et al., 2025). Confidence-like scores are also routinely available in applications, such as BI-RADS scores representing the likelihood of cancer in medical care (Sato, 2025), alert severity levels in security operations (Jalalvand et al., 2024), and risk scores in financial fraud detection. Moreover, automatic labeling systems, including LLM-based annotators and pre-trained classifiers, can also provide such information (Ishida et al., 2018; Tang et al., 2025). These examples motivate the growing body of confidence-based weakly supervised learning methods (Ishida et al., 2018; Cao et al., 2021; Berthon et al., 2021; Feng et al., 2021; Wang et al., 2023b).

However, to the best of our knowledge, positive-confidence has not been used for PU learning under the SAR setting. The key idea is that positive-confidence provides information that allows us to correct the bias in the labeled positive data. By using such confidence, we derive an estimator of the AUC risk from biased PU data without relying on the strong assumptions about data distributions or labeling mechanisms required by existing SAR methods. This enables us to maximize the AUC from biased PU data. A practical concern is that obtained confidence may not be accurately calibrated as a probability. We further show that the proposed method does not require exact calibration to recover the optimal AUC ranking: when the observed confidence is given by any strictly increasing transformation of the true posterior probability<sup>1</sup>, minimizing the resulting AUC risk still yields the Bayes-optimal AUC ranking. This indicates that the proposed method can also exploit confidence that is systematically over- or under-confident, as long as its ordering is preserved. In addition, our method is model-agnostic: it can use any differentiable classifier such as linear classifier, kernelbased classifier, and neural network, which is preferable in practice.

## 2 RELATED WORK

Various methods for AUC maximization have been proposed (Brefeld et al., 2005; Ying et al., 2016; Liu et al., 2020; Yuan et al., 2021a; Yang & Ying, 2022). Previous studies (Fujino & Ueda, 2016; Yuan et al., 2021a;b; Wang et al., 2023a) have shown that AUC maximization methods often outperform other methods for imbalanced data such as class balanced loss (Charoenphakdee et al., 2019), focal loss (Lin et al., 2017), or sampling-based methods (Menardi & Torelli, 2014; Chawla et al., 2002). However, these AUC maximization methods require both labeled PN data, and thus, they cannot be applied to our problem setting.

PU learning methods train binary classifiers by using only PU data (Bekker & Davis, 2020). A representative approach is empirical risk minimization, which rewrites the empirical risk by using only PU data (Sugiyama et al., 2022). Although most methods are designed for ordinary classification on balanced data (Du Plessis et al., 2015; Kiryo et al., 2017; Jiang et al., 2023), several studies have recently shown the AUC can also be maximized from PU data (Sakai et al., 2018; Xie & Li, 2018; Charoenphakdee et al., 2019; Xie et al., 2024). However, these PU learning and AUC maximization methods assume the SCAR setting, i.e., labeled positive data are unbiased, which is not often satisfied in practice. This paper tackles the more realistic but challenging SAR setting where labeled positive data are biased.

Some PU learning methods for the SAR setting have recently been proposed (Bekker et al., 2019; Kato et al., 2019; Gerych et al., 2022; He et al., 2018; Teisseyre et al., 2025). Although several studies used EM algorithm-based methods (Bekker et al., 2019; Gong et al., 2021), they lack theoretical guarantees for learning the true classifier (Gerych et al., 2022; Teisseyre et al., 2025). To make the problem theoretically tractable, most studies impose restrictive assumptions about data distributions or labeling mechanisms, such as the data separation assumption (Gerych et al., 2022), the invariance of order assumption (Kato et al., 2019), and the probabilistic gap assumption (He et al., 2018; Gerych et al., 2022). However, when the assumption is violated, they cannot work well. In addition, they are not designed for maximizing the AUC. In contrast, the proposed method can maximize the AUC from biased PU data using positive-confidence, without relying on such restrictive assumptions.

Several confidence-based learning methods have recently been proposed (Ishida et al., 2018; Cao et al., 2021; Berthon et al., 2021; Feng et al., 2021; Wang et al., 2023b). Especially, Ishida et al. (2018) and Shinoda et al. (2021) use positive-confidence to learn binary classifiers. Although they do not require unlabeled data, they cannot be applied for maximizing the AUC, and require that labeled positive data are unbiased. One PU learning method uses positive-confidence information to deal with noisy labels (Tang et al., 2025). However, it is not designed for either AUC maximization or the SAR setting. In addition, existing confidence-based learning methods typically formulate confidence as true posterior probabilities. In contrast, the proposed method can maximize the AUC even if the observed confidence is given by any strictly increasing transformation of the true posterior.

## 3 PRELIMINARIES

We briefly introduce AUC maximization. Let input instance $\mathbf { x } \in \mathcal { X }$ and its class label $y \in \{ 0 , 1 \}$ be associated with probability density $p ( \mathbf { x } , y )$ , where 1 and 0 correspond to PN classes, respectively. The conditional probability densities for PN classes are denoted as $p ^ { \mathrm { p } } ( \mathbf { x } ) \mathbf { \Psi } : = p ( \mathbf { x } | y \mathbf { \Psi } = \mathbf { \Phi } 1 )$ and $p ^ { \mathrm { n } } ( \mathbf { x } ) : = p ( \mathbf { x } | y = 0 )$ . Let $s : \mathcal { X } \to$ R be a score function that quantifies how likely an instance is positive. The classifier is defined by the score function with threshold $t \colon \hat { y } = I ( s ( \mathbf x ) \ge t )$ , where $I ( z )$ is the indicator function that outputs 1 if z is true and 0 otherwise.

The AUC is the probability that a randomly drawn positive instance is ranked higher than a randomly drawn negative instance with ties counted as one half (Yang & Ying, 2022). Formally, the AUC with score function s can be represented as

$$
\begin{array} { r l } & { \mathrm { A U C } ( s ) = \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { n } } \sim p ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ \ell _ { 0 1 } \big ( s \big ( \mathbf { x } ^ { \mathrm { n } } \big ) - s \big ( \mathbf { x } ^ { \mathrm { p } } \big ) \big ) \right] } \\ & { \qquad = 1 - \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { n } } \sim p ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ \ell _ { 0 1 } \big ( s \big ( \mathbf { x } ^ { \mathrm { p } } \big ) - s \big ( \mathbf { x } ^ { \mathrm { n } } \big ) \big ) \right] , } \end{array}\tag{1}
$$

where $\ell _ { 0 1 } ( z )$ is the zero-one loss, with $\ell _ { 0 1 } ( z ) = 1 { \mathrm { ~ i f ~ } } z < 0 , 0 { \mathrm { ~ i f ~ } } z > 0 ,$ , and $1 / 2 { \mathrm { i f } } z = 0$ , and E denotes the expectation. Maximizing the AUC is equivalent to minimizing the following AUC risk,

$$
\begin{array} { r } { \mathcal { R } ( s ) : = \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { n } } \sim p ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ \ell _ { 0 1 } \mathopen { } \mathclose \bgroup \left( s \mathopen { } \mathclose \bgroup \left( \mathbf { x } ^ { \mathrm { p } } \aftergroup \egroup \right) - s \mathopen { } \mathclose \bgroup \left( \mathbf { x } ^ { \mathrm { n } } \aftergroup \egroup \right) \aftergroup \egroup \right) \right] . } \end{array}\tag{2}
$$

Since the gradient of the zero-one loss is zero almost everywhere, the AUC risk cannot be directly minimized with gradient descent methods. To address this, a common approach is to use the smoothed AUC risk by replacing $\ell _ { 0 1 } ( z )$ with the sigmoid surrogate $\sigma ( - z )$ , where $\sigma ( z ) =$ $1 / ( 1 + \exp ( - z ) )$ (Charoenphakdee et al., 2019):

$$
\mathcal { R } _ { \sigma } ( s ) : = \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { n } } \sim p ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ \sigma \big ( - s \big ( \mathbf { x } ^ { \mathrm { p } } \big ) + s \big ( \mathbf { x } ^ { \mathrm { n } } \big ) \big ) \right] .\tag{3}
$$

Suppose we have $N ^ { \mathrm { p } }$ positive instances $\{ \mathbf { x } _ { 1 } ^ { \mathrm { p } } , \ldots , \mathbf { x } _ { N ^ { \mathrm { p } } } ^ { \mathrm { p } } \}$ drawn from $p ^ { \mathrm { p } } ( \mathbf { x } )$ and $N ^ { \mathrm { n } }$ negative instances $\left\{ \mathbf { x } _ { 1 } ^ { \mathrm { n } } , \ldots , \mathbf { x } _ { N ^ { \mathrm { n } } } ^ { \mathrm { n } } \right\}$ drawn from $p ^ { \mathrm { n } } ( \mathbf { x } )$ , the empirical estimator of Eq. (3) is given as

$$
\widehat { \mathcal { R } } _ { \sigma } ( s ) = \frac { 1 } { N ^ { \mathrm { p } } N ^ { \mathrm { n } } } \sum _ { n = 1 } ^ { N ^ { \mathrm { p } } } \sum _ { m = 1 } ^ { N ^ { \mathrm { n } } } \sigma ( - s ( \mathbf { x } _ { n } ^ { \mathrm { p } } ) + s ( \mathbf { x } _ { m } ^ { \mathrm { n } } ) ) .\tag{4}
$$

By minimizing this empirical AUC risk w.r.t. the parameters of $^ { \mathrm { ~ ~ } } { } _ { s , \ }$ we can obtain good score functions to maximize the AUC (Yang & Ying, 2022).

## 4 PROPOSED METHOD

## 4.1 PROBLEM FORMULATION

We assume that there exists probability density $p ( \mathbf { x } , y , o )$ , where $\mathbf { x } \in \mathcal { X }$ is an input instance, $y \in$ $\{ 0 , 1 \}$ is its class, and $o \in \{ 0 , 1 \}$ represents whether its class y is observed (labeled). Here, $o = 1$ denotes that its class is labeled, and $o = 0$ denotes that its class is unlabeled. In PU learning setting, only positive data are labeled. This PU assumption can be formally represented as

$$
p ( o = 1 | \mathbf { x } , y = 0 ) = 0 ,\tag{5}
$$

which means that negative data are never labeled (Elkan & Noto, 2008). Propensity score $e ( \mathbf { x } )$ is defined as

$$
e ( \mathbf { x } ) : = p ( o = 1 | \mathbf { x } , y = 1 ) ,\tag{6}
$$

which represents the probability that positive instance x is labeled. Most PU learning methods assume the SCAR setting, i.e., propensity score $e ( \mathbf { x } )$ does not depend on feature values ${ \bf x } , e ( { \bf x } ) =$ $p ( o = 1 | y = 1 )$ (Sakai et al., 2018; Kiryo et al., 2017; Xie & Li, 2018; Xie et al., 2024). Although this assumption simplifies the problem, it is seldom met in practice. Thus, this paper assumes the SAR setting,

$$
e ( \mathbf { x } ) = p ( o = 1 | \mathbf { x } , y = 1 ) \not \equiv p ( o = 1 | y = 1 ) ,\tag{7}
$$

which means that $e ( \mathbf { x } )$ depends on feature values x. This is a more realistic but difficult setting.   
Following previous studies on biased PU learning (Bekker et al., 2019; Bekker & Davis, 2020;   
Gerych et al., 2022), we assume the positivity condition that $e ( \mathbf { x } ) > 0$ for p<sup>p</sup>-almost every x.

For collecting PU data, there are two possible scenarios: the one-sample scenario (also known as the censoring scenario) (Elkan & Noto, 2008) and the two-sample scenario (also known as the casecontrol scenario) (Ward et al., 2009). This paper focuses on the one-sample scenario since it is more commonly used in PU learning studies (Bekker & Davis, 2020). We can easily modify the proposed method for the two-sample scenario, which is described in Section D.2. In the one-sample scenario, a set of unlabeled data is first sampled from true marginal density $p ( \mathbf { x } ) = \pi p ^ { \mathrm { p } } ( \mathbf { x } ) + ( \bar { 1 } - \pi ) p ^ { \mathrm { n } } ( \mathbf { x } )$ where $\pi : = p ( y = 1 )$ is the positive class-prior. Then, if instance x is positive, it can be labeled with probability $e ( \mathbf { x } )$ , and x remains unlabeled with probability $1 - e ( \mathbf { x } ) $ ; if instance x is negative, it is never labeled and thus x remains unlabeled with probability 1. In our problem setting, we additionally assume that each labeled positive instance x is equipped with confidence $r ( \mathbf { x } ) : = p ( y =$ $1 | \mathbf { x } )$ . Note that this equality does not have to strictly hold as shown later. As a result, we are given PU data $X ^ { \mathrm { p } } \cup X$ with positive-confidence $R ^ { \mathrm { p } }$

$$
\begin{array} { r l } & { X ^ { \mathrm { p } } : = \{ \mathbf { x } _ { n } ^ { \mathrm { p } } \} _ { n = 1 } ^ { N ^ { \mathrm { p } } } \sim p ^ { 1 } ( \mathbf { x } ) : = p ( \mathbf { x } | o = 1 ) , X : = \{ \mathbf { x } _ { m } \} _ { m = 1 } ^ { N } \sim p ^ { \mathrm { u } } ( \mathbf { x } ) : = p ( \mathbf { x } | o = 0 ) , } \\ & { R ^ { \mathrm { p } } : = \{ r _ { n } ^ { \mathrm { p } } | r _ { n } ^ { \mathrm { p } } = p ( y = 1 | \mathbf { x } _ { n } ^ { \mathrm { p } } ) \} _ { n = 1 } ^ { N ^ { \mathrm { p } } } , } \end{array}\tag{8}
$$

where $p ^ { 1 } ( \mathbf { x } )$ and $p ^ { \mathbf { u } } ( \mathbf { x } )$ are labeled (positive) and unlabeled densities, respectively. From the data generation process, true marginal density $p ( \mathbf { x } )$ can be also represented as $p ( \mathbf { x } ) = \alpha p ^ { 1 } ( \mathbf { x } ) + ( 1 -$ $\alpha ) p ^ { \mathrm { u } } ( \mathbf { x } )$ , where $c : = p ( o = 1 | y = 1 )$ and ${ \dot { \alpha } } : = p ( o = 1 ) = { \dot { c } } \pi ^ { 2 }$ . Note that $p ^ { 1 } ( \mathbf { x } ) \neq p ^ { \mathrm { p } } ( \mathbf { x } )$ in the SAR setting. Our goal is to learn score function s that can maximize the AUC under the SAR setting by using given PU data with positive-confidence $X ^ { \mathrm { p } } \cup X \cup R ^ { \mathrm { p } }$

## 4.2 AUC RISK ESTIMATOR WITH BIASED PU DATA AND POSITIVE-CONFIDENCE

We derive the AUC risk estimator in the SAR setting from biased PU data with positive-confidence. The objective function to be minimized is the following smoothed AUC risk,

$$
\begin{array} { r } { \mathcal { R } _ { \sigma } ( s ) = \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { n } } \sim p ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ^ { \mathrm { n } } ) \right] , } \end{array}\tag{9}
$$

where we set $f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ^ { \mathrm { n } } ) : = \sigma ( - s ( \mathbf { x } ^ { \mathrm { p } } ) + s ( \mathbf { x } ^ { \mathrm { n } } ) )$ . Although this AUC risk depends on $p ^ { \mathrm { p } } ( \mathbf { x } )$ and $p ^ { \mathrm { n } } ( \mathbf { x } )$ , data from both the distributions are unavailable in our setting; it seems impossible to calculate. However, we show that it is in fact possible below.

First, by the definition $p ( \mathbf { x } ) = \pi p ^ { \mathrm { p } } ( \mathbf { x } ) + ( 1 - \pi ) p ^ { \mathrm { n } } ( \mathbf { x } )$ , the negative density can be represented as

$$
p ^ { \mathrm { n } } ( \mathbf { x } ) = \frac { 1 } { 1 - \pi } \left[ p ( \mathbf { x } ) - \pi p ^ { \mathrm { p } } ( \mathbf { x } ) \right] .\tag{10}
$$

By substituting Eq. (10) into Eq. (9), we obtain

$$
\mathcal { R } _ { \sigma } ( s ) = \frac { 1 } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathsf { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathsf { p } } , \mathbf { x } ) \right] - \frac { \pi } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathsf { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { \bar { x } } ^ { \mathsf { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathsf { p } } , \bar { \mathbf { x } } ^ { \mathsf { p } } ) \right] .\tag{11}
$$

Here, $\mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { \overline { { x } } } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { \overline { { x } } } ^ { \mathrm { p } } ) \right] = 1 / 2$ due to the symmetry of sigmoid function $\sigma$ (Xie & Li, 2018; Charoenphakdee et al., 2019; Xie et al., 2024) (The proof is described in Section C.1). Thus, the AUC risk can be represented as

$$
\mathcal { R } _ { \sigma } ( s ) = \frac { 1 } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{12}
$$

Labeled density $p ^ { 1 } ( \mathbf { x } )$ and positive density $p ^ { \mathrm { p } } ( \mathbf { x } )$ are related as follows,

$$
p ^ { 1 } ( { \bf x } ) = \frac { e ( { \bf x } ) } { c } p ^ { \mathrm { p } } ( { \bf x } ) ,\tag{13}
$$

which means that a positive instance is first sampled from $p ^ { \mathrm { p } } ( \mathbf { x } )$ and then is observed as labeled data with probability in accordance with propensity score $e ( \mathbf { x } )$ (Bekker & Davis, 2020). The derivation of Eq. (13) is described in Section C.2. Under the positivity condition described in Section 4.1, Eq. (13) can be inverted as $p ^ { \mathrm { p } } ( \mathbf { x } ) = c p ^ { \mathrm { l } } ( \mathbf { x } ) / e ( \mathbf { x } )$ for $\bar { p } ^ { \mathrm { p } }$ -almost every $\mathbf { x } .$

Additionally, by the PU assumption, $p ( o = 1 | \mathbf { x } ) = e ( \mathbf { x } ) p ( y = 1 | \mathbf { x } )$ , as shown in Section C.3. Thus, for $p ( y = 1 | \mathbf { x } ) > 0$ , we obtain

$$
e ( \mathbf { x } ) = { \frac { p ( o = 1 | \mathbf { x } ) } { p ( y = 1 | \mathbf { x } ) } } .\tag{14}
$$

Since $p ^ { \mathrm { p } } ( \mathbf { x } ) = p ( y = 1 | \mathbf { x } ) p ( \mathbf { x } ) / \pi$ , we have $p ( y = 1 | \mathbf { x } ) > 0$ for $p ^ { \mathrm { p } }$ -almost every x. Thus, Eq. (14) can be used together with the inverted form of Eq. (13) in the $p ^ { \mathrm { p . } }$ -expectation in $\mathrm { E q . } ( 1 2 ) . ^ { 3 }$ We can therefore rewrite Eq. (12) as

$$
\mathcal { R } _ { \sigma } ( s ) = \frac { c } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{15}
$$

From this equation, we can see that weight $p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) / p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } )$ becomes larger for labeled positive instance $\mathbf { x } ^ { \mathrm { p } }$ with higher $p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) \left( \mathrm { i . e } ^ { } \right.$ ., more likely to be positive) and lower $p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } )$ (i.e., rarer as labeled data). By substituting $\bar { p } ( { \bf x } ) = \alpha p ^ { 1 } ( { \bf x } ) + \bar { ( 1 - \alpha ) } p ^ { \mathrm { u } } ( { \bf x } )$ into Eq. (15), we obtain

$$
\begin{array} { l } { { \displaystyle \mathcal { R } _ { \sigma } ( s ) = \frac { c } { 1 - \pi } \left[ \alpha \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \bar { \mathbf { x } } ^ { \mathrm { p } } ) \right] \right. } } \\ { { \displaystyle \qquad + \left. ( 1 - \alpha ) \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ^ { \mathrm { u } } ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] \right] - \frac { \pi } { 2 ( 1 - \pi ) } . } } \end{array}\tag{16}
$$

Here, we train probabilistic classifier $\hat { u } ( { \bf x } )$ to estimate $u ( \mathbf { x } ) : = p ( o = 1 | \mathbf { x } )$ using given training data $X ^ { \mathrm { p } } \cup X$ . We also estimate labeled probability $\alpha = p ( o = 1 )$ as $\hat { \alpha } : = \dot { N ^ { \mathrm { p } } } / ( N ^ { \bar { \mathrm { p } } } \dot { + } N )$ . In addition, the confidence for labeled positive instances, $r _ { n } ^ { \mathrm { p } } = p ( y = 1 | \mathbf { x } _ { n } ^ { \mathrm { p } } )$ , is given in our setting. Thus, by replacing $u ( \mathbf { x } ) = p ( o = 1 | \mathbf { x } )$ and α with their estimates and each expectation with the sample average in Eq. (16), we obtain the following estimator of the AUC risk:

$$
\hat { \mathcal { R } } _ { \sigma } ( s ) = \frac { c } { 1 - \pi } \left[ \frac { \hat { \alpha } } { N ^ { \mathrm { p } } \big ( N ^ { \mathrm { p } } - 1 \big ) } \sum _ { n \neq m } ^ { N ^ { \mathrm { p } } , N ^ { \mathrm { p } } } \frac { r _ { n } ^ { \mathrm { p } } } { \hat { u } _ { n } ^ { \mathrm { p } } } f \big ( \mathbf { x } _ { n } ^ { \mathrm { p } } , \bar { \mathbf { x } } _ { m } ^ { \mathrm { p } } \big ) + \frac { \big ( 1 - \hat { \alpha } \big ) } { N ^ { \mathrm { p } } N } \sum _ { n , m = 1 } ^ { N ^ { \mathrm { p } } , N } \frac { r _ { n } ^ { \mathrm { p } } } { \hat { u } _ { n } ^ { \mathrm { p } } } f \big ( \mathbf { x } _ { n } ^ { \mathrm { p } } , \mathbf { x } _ { m } \big ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{17}
$$

where $\hat { u } _ { n } ^ { \mathrm { p } } : = \hat { u } ( \mathbf { x } _ { n } ^ { \mathrm { p } } )$ , and we omit the case of $n = m$ in the first term to avoid biases following (Sakai et al., 2018). Since coefficient $c / ( 1 - \pi )$ and constant $\pi / ( 2 ( 1 - \pi ) )$ do not affect the optimization of $s ,$ we can safely ignore them for training. This property is preferable in practice since π and c are difficult or impossible to estimate (Bekker & Davis, 2020). Note that although we use the sigmoid function in the AUC risk in Eq. (9), we can derive the AUC risk estimator of the same form in Eq. (17) whenever we use symmetric functions $( \mathrm { i . e . }$ , function σ satisfying $\sigma ( z ) + \sigma ( - z ) = k$ for any $z \in \mathbb { R }$ and k is a constant). This is because the second term in Eq. (11) also becomes constant. The symmetric functions include many common loss functions such as sigmoid, ramp, and unhinged functions (Charoenphakdee et al., 2019). Even when using non-symmetric functions, we can derive a slightly modified version of the AUC risk estimator, which also enables us to maximize the AUC without requiring $\pi$ and $c .$ The details are described in Section D.1. We further analyze the sensitivity of the rewritten risk to labeling probability estimation errors in Section B.2 and the generalization behavior of its empirical estimator in Section B.3.

Algorithm 1 shows the training procedure of the proposed method with stochastic gradient descent methods. We first set mini-batch sizes $U$ and $\bar { P }$ so as to maintain the original label ratio (Line 1). Then, we train probabilistic binary classifier $\hat { u } ( { \bf x } )$ to estimate $p ( o = 1 | \mathbf { x } )$ with PU data $X ^ { \mathrm { p } } \cup X$ (Line 2). Then, we randomly sample PU data with positive-confidence from given data $X ^ { \mathrm { p } } \cup X \cup R ^ { \mathrm { p } }$ (Lines 4–5). We calculate the loss in Eq. (17) for learning score function s with trained classifier uˆ(x) (Line 6). We update parameters of score function s by using the gradient of the loss (Line 7). In the gradient calculation of the last step, parameters of trained classifier $\hat { u } ( { \bf x } )$ are frozen.

## 4.3 ROBUSTNESS TO STRICTLY INCREASING TRANSFORMATIONS OF POSITIVE-CONFIDENCE

By replacing true posterior $p ( y = 1 | \mathbf { x } )$ in Eq. (15) with observed confidence $r ( \mathbf { x } )$ , we obtain the objective, which is exactly equivalent to the original AUC risk when $r ( \mathbf { x } ) = p ( y = 1 | \mathbf { x } )$ . However, we show below that this exact equality is not necessary to recover the Bayes-optimal ranking for the AUC.

<sup>3</sup>The denominator in the resulting expectation is also nonzero almost surely. In fact, since $p ^ { 1 } ( \mathbf { x } ) = p ( o =$ $1 | \mathbf { x } ) p ( \mathbf { x } ) / p ( o = 1 )$ , we have $p ( o = 1 | \mathbf { x } ) > 0$ for $p ^ { 1 } .$ -almost every x.

Algorithm 1 Training procedure of the proposed method   
Require: Biased PU data with positive-confidence $X ^ { \mathrm { p } } \cup X \cup R ^ { \mathrm { p } }$ and total mini-batch size M   
Ensure: Model parameters of score function s   
1: Partition M into unlabeled and positive mini-batch sizes, U and $P ,$ so that $P / ( P + U ) =$   
$N ^ { \mathrm { p } } / ( N ^ { \mathrm { p } } + N ) = : { \hat { \alpha } }$   
2: Train classifier $\hat { u } ( { \bf x } )$ to estimate $p ( o = 1 | \mathbf { x } )$ with PU data $X ^ { \mathrm { p } } \cup X$   
3: repeat   
4: Sample unlabeled data with size U from X   
5: Sample positive data with size P and their confidence from $X ^ { \mathrm { p } } \cup R ^ { \mathrm { p } }$   
6: Calculate the loss in Eq. (17) on the sampled data with trained classifier $\hat { u } ( { \bf x } )$   
7: Update parameters of s with the gradient of the loss   
8: until End condition is satisfied;

Proposition 4.1. Suppose that the available confidence is $r ^ { \prime } ( \mathbf { x } ) \ = \ h \left( p ( y = 1 | \mathbf { x } ) \right)$ , where $h :$ $[ 0 , \bar { 1 } ]  [ 0 , 1 ]$ is an arbitrary strictly increasing function. Then, the objective obtained by replacing $p ( y = 1 | \mathbf { x } )$ with $r ^ { \prime } ( \mathbf { x } )$ in Eq. (15), i.e.,

$$
\mathcal { R } _ { \sigma } ^ { h } ( s ) = \frac { c } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \frac { r ^ { \prime } ( \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) }\tag{18}
$$

induces the same Bayes-optimal rankingfor the AUC as the original AUC risk in $E q . \ ( 9 )$

The proof is provided in Section B.1. This result shows that the proposed method can recover the Bayes-optimal ranking for the AUC even when the available confidence is given as any strictly increasing transformation of the true posterior probability which includes systematic over- or under confidence. In this sense, the proposed method can treat a broader class of confidence than true posterior probabilities.

## 5 EXPERIMENTS

## 5.1 DATA

We mainly used six real-world datasets: Mnist (LeCun et al., 1998), FashionMnist (Fmnist) (Xiao et al., 2017), Svhn (Netzer et al., 2011), Cifar10 (Krizhevsky et al., 2009), Diabetes (Gardner et al., 2023), and Blood (Gardner et al., 2023). The first four datasets are image datasets, while the remaining two are tabular datasets. These datasets have been commonly used in PU learning studies (Kumagai et al., 2024; 2025a;b; Xie et al., 2024; Kiryo et al., 2017; Jiang et al., 2023). Following the previous studies (Kumagai et al., 2025b; 2024; 2025a; Xie et al., 2024), we constructed binary classification problems for the image datasets by partitioning their original classes into positive and negative classes, while using the original binary labels for Diabetes and Blood. The detailed construction is provided in Section E.1.

For training and validation in each dataset, we first randomly sampled 5, 000 and 1, 000 instances from the original data with positive class-prior π, respectively. To create the class-imbalanced data, we set π to a small value. Specifically, we changed π within $\{ 0 . 0 5 , 0 . 1 , 0 . 1 5 , 0 . 2 \}$ in our experiments. Then, we set positive labeling rate $c = p ( o = 1 | y = 1 )$ to 0.1<sup>4</sup>. We then selected a fraction c of the positive instances within the initially sampled data as labeled positive data according to a biased sampling scheme. Specifically, to induce sampling bias, we made instances from a subset of the positive classes more likely to be labeled for the image datasets and introduced a distance-based selection bias for the tabular datasets. The detailed selection procedure for each dataset and result under other selection biases are provided in Sections E.1 and F.4, respectively. We used 2, 500 positive and 2, 500 negative data as test data for evaluation.

In the main experiments, positive-confidence was estimated using a probabilistic classifier trained on 10, 000 labeled instances with class-prior $\pi ,$ following the standard evaluation protocol in previous confidence-based learning studies (Ishida et al., 2018; Cao et al., 2021; Berthon et al., 2021;

Wang et al., 2023b). Note that these labeled data were used solely to generate confidence values for controlled evaluation. This protocol enables us to quantitatively assess the effectiveness of our AUC risk estimator. Confidence obtained from pre-trained classifiers is also practically relevant, as such classifiers are widely deployed in many real-world systems. Note that results with positiveconfidence obtained from real-world human annotators are also reported in Section 5.3. All labeled positive data for training and validation were equipped with positive-confidence. Training data, validation data, test data, and data for the classifier to estimate positive-confidence did not overlap. We conducted 10 experiments for each positive class-prior π changing the random seeds and evaluated the mean test AUC.

## 5.2 COMPARISON METHODS

We compared the proposed method (Ours) with eight methods: non-traditional classifier-based method (NTC) (Elkan & Noto, 2008), non-negative PU learning method (nnPU) (Kiryo et al., 2017), AUC maximization method with PU data (PUAUC) (Xie & Li, 2018; Xie et al., 2024), PU learning method with selection bias (PUSB) (Kato et al., 2019), probabilistic gap-based method (PG) (Gerych et al., 2022), positive-confidence-based method (Pconf) (Ishida et al., 2018), confidence-based noisy PU learning method (NPU) (Tang et al., 2025), and the proposed method without positive-confidence (w/oConf). We further evaluated LBE, a PU learning method under the SAR setting in Section F.6. All methods including our method used neural networks for modeling classifiers.

NTC learns the binary classifier with PU data by simply treating all unlabeled data as negative data. The proposed method also used it for estimating $p ( o = 1 | \mathbf { x } )$ . nnPU and PUAUC are PU learning methods for the SCAR setting (i.e., they assume that labeled positive data are unbiased). Specifically, nnPU learns the classifier by minimizing the non-negative PU risk with PU data. PUAUC learns the classifier by minimizing the AUC risk that is calculated with PU data. PUSB and PG are PU learning methods for the SAR setting (i.e., they assume that labeled positive data are biased). To deal with the SAR setting, PUSB assumes that $p ( o \mid = 1 | \mathbf { x } )$ and $p ( y = 1 | \mathbf { x } )$ induce the same order on the feature space. PG assumes propensity score e(x) to be a linear function of $p ( y = 1 | \mathbf { x } )$ . PUSB and PG do not use positive-confidence information. In contrast, Pconf and NPU use positive-confidence information as in the proposed method. Pconf learns the classifier by using only positive data with confidence. NPU learns the classifier by using PU data with positive-confidence, which is designed to handle label noise in labeled positive data. w/oConf is the proposed method that assumes $r _ { n } ^ { \mathrm { p } } = 1$ for all labeled positive data in Eq. (17). We compared it to evaluate the effectiveness of using positive-confidence in our framework. For nnPU, PUSB, and NPU, the absolute loss correction was used for preventing overfitting (Kumagai et al., 2025b).

For the proposed method and w/oConf, we rounded up $\hat { u } _ { n } ^ { \mathrm { p } } = \hat { u } ( \mathbf { x } _ { n } ^ { \mathrm { p } } )$ in Eq. (17) less than 0.01 to 0.01 to stabilize the training process following the previous study (Ishida et al., 2018). We empirically confirm the robustness to this clipping value in Section F.9. The empirical risk with validation PU data was used for early-stopping to mitigate overfitting. All methods were implemented using PyTorch (Paszke et al., 2017). The details of settings such as network architectures are described in Section E.3.

## 5.3 RESULTS

Table 1 shows the average test AUCs of each method on the six real-world datasets. Full results including the standard deviations are described in Section F.11. The proposed method performed the best or comparably to it in all cases. NTC tended not to work well since it treats all unlabeled data as negative, which leads to biased results. nnPU and PUAUC did not work well since they are not designed for the SAR setting. In particular, since the main difference between the proposed method and PUAUC is whether the SAR setting is taken into account or not, this result indicates the importance of considering the bias of labeled positive data in AUC maximization framework. PUSB and PG, which are PU learning methods for the SAR setting, tended to perform worse than the proposed method because their assumptions to deal with the SAR setting were invalid in our experiments. By using positive-confidence information, the proposed method was able to perform well without relying on such assumptions. Since Pconf does not use unlabeled data and assumes that labeled positive data are unbiased, it also did not work well even with positive-confidence. NPU, which uses positive-confidence in the noisy PU learning framework, also did not work well since it is not designed for either AUC maximization or the SAR setting. The proposed method often outperformed w/oConf, which is the proposed method without positive-confidence. Thus, this result indicates the effectiveness of using positive-confidence information in our framework. Overall, these results show the effectiveness of the proposed method. The results with each π value are described in Section F.1. The proposed method worked well in each case.

Table 1: Average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2}. We set $c = p ( o = 1 | y = 1 ) = 0 . 1$ . Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Ours</td><td>w/oConf</td><td>NTC</td><td>nnPU</td><td>PUAUC</td><td>PUSB</td><td>PG</td><td>Pconf</td><td>NPU</td></tr><tr><td>Mnist</td><td>0.8979</td><td>0.8769</td><td>0.8575</td><td>0.8196</td><td>0.8828</td><td>0.8841</td><td>0.8575</td><td>0.8193</td><td>0.8285</td></tr><tr><td>Fmnist</td><td>0.9495</td><td>0.9371</td><td>0.9165</td><td>0.9296</td><td>0.9409</td><td>0.9474</td><td>0.9165</td><td>0.7737</td><td>0.9470</td></tr><tr><td>Svhn</td><td>0.6673</td><td>0.6666</td><td>0.5328</td><td>0.4885</td><td>0.6704</td><td>0.5965</td><td>0.5328</td><td>0.4900</td><td>0.4893</td></tr><tr><td>Cifar10</td><td>0.8643</td><td>0.8371</td><td>0.8013</td><td>0.4819</td><td>0.8620</td><td>0.8442</td><td>0.8013</td><td>0.4767</td><td>0.4883</td></tr><tr><td>Diabetes</td><td>0.7342</td><td>0.6926</td><td>0.7357</td><td>0.5234</td><td>0.7171</td><td>0.7230</td><td>0.7357</td><td>0.3974</td><td>0.4626</td></tr><tr><td>Blood</td><td>0.5509</td><td>0.5287</td><td>0.5409</td><td>0.4847</td><td>0.5395</td><td>0.5450</td><td>0.5409</td><td>0.4962</td><td>0.4828</td></tr></table>

![](images/294af64a5b94740540dab1a4b633b441c348b326a61176a7b45158093a7a54c3.jpg)  
(a) Mnist

![](images/24f82e0364c025094ebfab5f6eec440bf6a0585c90ca8f3f6dd9837fb3ca83fd.jpg)  
(b) Cifar10

![](images/0658924033b101e203affb7fe0ee0158cadbd9ee3497baacee241226a43637a9.jpg)  
(c) Diabetes

![](images/0ab839003d72de15ab265ada23996fcec10de22abb2a11ca9e87a46286cb3c3e.jpg)  
(d) Blood  
Figure 2: The average test AUCs and their standard errors over different positive class-priors for different parameters k of the transformation $h ( r ) = r ^ { k }$ . Due to space limitation, we reported the results on two image and two tabular datasets; results on all datasets are provided in Section F.7.

Table 2: Results on the Cifar10-H and Fmnist-H datasets: average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2}. We set $c = p ( o = \bar { 1 } \bar { | } y = 1 ) = 0 . 1$ . Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Ours</td><td>w/oConf</td><td>NTC</td><td>nnPU</td><td>PUAUC</td><td>PUSB</td><td>PG</td><td>Pconf</td><td>NPU</td></tr><tr><td>Cifar10-H</td><td>0.5991</td><td>0.5998</td><td>0.4779</td><td>0.4404</td><td>0.5813</td><td>0.5106</td><td>0.4779</td><td>0.5828</td><td>0.5712</td></tr><tr><td>Fmnist-H</td><td>0.8285</td><td>0.8192</td><td>0.7823</td><td>0.6013</td><td>0.8324</td><td>0.8242</td><td>0.7823</td><td>0.5211</td><td>0.8061</td></tr></table>

We evaluated the performance of the proposed method when the confidence was distorted by strictly increasing transformation h. As described in Section 4.3, minimizing the proposed AUC objective with such transformed confidence still leads to the Bayes-optimal ranking for the AUC. Thus, the proposed method is expected to be empirically robust to such transformations. We considered transformation $h ( r ) = r ^ { k }$ , where $k > 0 ,$ , with $\dot { k } = 1$ corresponding to the original confidence. Figure 2 shows the average test AUCs with their standard errors when changing k of the transformation $h ( r ) = r ^ { k }$ . Overall, the performance of the proposed method remained relatively stable across different transformations and showed no systematic degradation as k moved away from 1. In contrast, the performances of Pconf that also uses positive-confidence substantially degraded for some values of k because it does not have the ranking-invariance property of the proposed method. Results for other transformations are also reported in Section F.7. We also evaluated the robustness of the proposed method to additive Gaussian noise in positive-confidence in Section F.8.

In our experiments so far, we have used classifier-based confidence. However, confidence can also be obtained from labels provided by multiple annotators. Therefore, we evaluated this setting using the Cifar10-H and Fmnist-H datasets, where each instance was labeled by multiple real-world human annotators (Peterson et al., 2019; Ishida et al., 2023), and which have been widely used in confidence-based learning studies (Ishida et al., 2023; Ushio et al., 2026). Confidence for each instance can be obtained by averaging the corresponding annotations (Ishida et al., 2023). Details of the datasets and data construction are provided in Section E.2. Table 2 shows the results. On both datasets, the proposed method achieved performance statistically comparable to the best-performing method, demonstrating its applicability to human-derived confidence. On Cifar10-H, the average positive-confidence value was close to one. Therefore, the proposed method behaved similarly to w/oConf, which sets all positive-confidence values to one, resulting in similar performance.

## 6 CONCLUSION

In this paper, we proposed an AUC maximization method under the SAR setting with biased PU data and positive-confidence. By using positive-confidence, the proposed method can maximize the AUC even in the SAR setting without restrictive assumptions about the data distributions or labeling mechanism. Our experiments on eight real-world datasets demonstrate the effectiveness of the proposed method.

## REFERENCES

Sikha Bagui and Kunqi Li. Resampling imbalanced data for network intrusion detection datasets. Journal ofBig Data, 8(1):6, 2021.

Jessa Bekker and Jesse Davis. Learning from positive and unlabeled data: A survey. Machine Learning, 109(4):719–760, 2020.

Jessa Bekker, Pieter Robberechts, and Jesse Davis. Beyond the selected completely at random assumption for learning from positive and unlabeled data. In ECML, 2019.

Antonin Berthon, Bo Han, Gang Niu, Tongliang Liu, and Masashi Sugiyama. Confidence scores make instance-dependent label-noise learning possible. In ICML, 2021.

Andrew P Bradley. The use of the area under the roc curve in the evaluation of machine learning algorithms. Pattern recognition, 30(7):1145–1159, 1997.

Ulf Brefeld, Tobias Scheffer, et al. Auc maximizing support vector learning. In Proceedings of the ICML 2005 workshop on ROC Analysis in Machine Learning, 2005.

Yuzhou Cao, Lei Feng, Yitian Xu, Bo An, Gang Niu, and Masashi Sugiyama. Learning from similarity-confidence data. In ICML, 2021.

Nontawat Charoenphakdee, Jongyeong Lee, and Masashi Sugiyama. On symmetric losses for learn ing from corrupted labels. In ICML, 2019.

Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority over-sampling technique. Journal ofartificial intelligence research, 16:321–357, 2002.

Marthinus Du Plessis, Gang Niu, and Masashi Sugiyama. Convex formulation for learning from positive and unlabeled data. In ICML, 2015.

Charles Elkan and Keith Noto. Learning classifiers from only positive and unlabeled data. In SIGKDD, 2008.

Lei Feng, Senlin Shu, Nan Lu, Bo Han, Miao Xu, Gang Niu, Bo An, and Masashi Sugiyama. Pointwise binary classification with pairwise confidence comparisons. In ICML, 2021.

Akinori Fujino and Naonori Ueda. A semi-supervised auc optimization method with generative models. In ICDM, 2016.

Josh Gardner, Zoran Popovic, and Ludwig Schmidt. Benchmarking distribution shift in tabular data with tableshift. NeurIPS, 2023.

Walter Gerych, Thomas Hartvigsen, Luke Buquicchio, Emmanuel Agu, and Elke Rundensteiner. Recovering the propensity score from biased positive unlabeled data. In AAAI, 2022.

Chen Gong, Qizhou Wang, Tongliang Liu, Bo Han, Jane You, Jian Yang, and Dacheng Tao. Instance-dependent positive and unlabeled learning with labeling bias estimation. IEEE transactions on pattern analysis and machine intelligence, 44(8):4163–4177, 2021.

Fengxiang He, Tongliang Liu, Geoffrey I Webb, and Dacheng Tao. Instance-dependent pu learning by bayesian optimal relabeling. arXiv preprint arXiv:1808.02180, 2018.

Takashi Ishida, Gang Niu, and Masashi Sugiyama. Binary classification from positive-confidence data. In NeurIPS, 2018.

Takashi Ishida, Ikko Yamane, Nontawat Charoenphakdee, Gang Niu, and Masashi Sugiyama. Is the performance of my deep network too good to be true? a direct approach to estimating the bayes error in binary classification. In ICLR, 2023.

Fatemeh Jalalvand, Mohan Baruwal Chhetri, Surya Nepal, and Cecile Paris. Alert prioritisation in security operations centres: a systematic survey on criteria and methods. ACM Computing Surveys, 57(2):1–36, 2024.

Yangbangyan Jiang, Qianqian Xu, Yunrui Zhao, Zhiyong Yang, Peisong Wen, Xiaochun Cao, and Qingming Huang. Positive-unlabeled learning with label distribution alignment. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

Justin M Johnson and Taghi M Khoshgoftaar. Survey on deep learning with class imbalance. Journal ofBig Data, 6(1):1–54, 2019.

Takafumi Kanamori, Shohei Hido, and Masashi Sugiyama. A least-squares approach to direct importance estimation. The Journal ofMachine Learning Research, 10:1391–1445, 2009.

Alex Kantchelian, Michael Carl Tschantz, Sadia Afroz, Brad Miller, Vaishaal Shankar, Rekha Bachwani, Anthony D Joseph, and J Doug Tygar. Better malware ground truth: Techniques for weighting anti-virus vendor labels. In AISec, 2015.

Masahiro Kato and Takeshi Teshima. Non-negative bregman divergence minimization for deep direct density ratio estimation. In ICML, 2021.

Masahiro Kato, Takeshi Teshima, and Junya Honda. Learning from positive and unlabeled data with a selection bias. In ICLR, 2019.

Diederik P Kingma and Jimmy Ba. Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Ryuichi Kiryo, Gang Niu, Marthinus C Du Plessis, and Masashi Sugiyama. Positive-unlabeled learning with non-negative risk estimator. NeurIPS, 2017.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Atsutoshi Kumagai, Tomoharu Iwata, Hiroshi Takahashi, Taishi Nishiyama, and Yasuhiro Fujiwara. Auc maximization under positive distribution shift. In NeurIPS, 2024.

Atsutoshi Kumagai, Tomoharu Iwata, Hiroshi Takahashi, Taishi Nishiyama, Kazuki Adachi, and Yasuhiro Fujiwara. Positive-unlabeled auc maximization under covariate shift. In ICML, 2025a.

Atsutoshi Kumagai, Tomoharu Iwata, Hiroshi Takahashi, Taishi Nishiyama, and Yasuhiro Fujiwara. Importance-weighted positive-unlabeled learning for distribution shift adaptation. In AISTATS, 2025b.

Khiem H Le, Tuan V Tran, Hieu H Pham, Hieu T Nguyen, Tung T Le, and Ha Q Nguyen. Learning from multiple expert annotators for enhancing anomaly detection in medical image analysis. IEEE Access, 11:14105–14114, 2023.

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to´ document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object ´ detection. In ICCV, 2017.

Mingrui Liu, Zhuoning Yuan, Yiming Ying, and Tianbao Yang. Stochastic auc maximization with deep neural networks. In ICLR, 2020.

Matthew McDermott, Lasse Hyldig Hansen, Haoran Zhang, Giovanni Angelotti, and Jack Gallifant. A closer look at auroc and auprc under class imbalance. In NeurIPS, 2024.

Giovanna Menardi and Nicola Torelli. Training and assessing classification rules with imbalanced data. Data mining and knowledge discovery, 28:92–122, 2014.

Aditya Krishna Menon and Robert C Williamson. Bipartite ranking: a risk-theoretic perspective. Journal ofMachine Learning Research, 17(195):1–102, 2016.

Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai. Kitsune: an ensemble of autoencoders for online network intrusion detection. In NDSS, 2018.

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervisedfeature learning, volume 2011, pp. 7. Granada, Spain, 2011.

Je-Kang Park, Bae-Keun Kwon, Jun-Hyub Park, and Dong-Joong Kang. Machine learning-based imaging system for surface defect inspection. International Journal of Precision Engineering and Manufacturing-Green Technology, 3:303–310, 2016.

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.

Joshua C Peterson, Ruairidh M Battleday, Thomas L Griffiths, and Olga Russakovsky. Human uncertainty makes classification more robust. In ICCV, 2019.

Tomoya Sakai and Nobuyuki Shimizu. Covariate shift adaptation on learning from positive and unlabeled data. In AAAI, 2019.

Tomoya Sakai, Gang Niu, and Masashi Sugiyama. Semi-supervised auc optimization based on positive-unlabeled learning. Machine Learning, 107:767–794, 2018.

Ryoma Sato. Interestingness first classifiers. arXiv preprint arXiv:2508.19780, 2025.

Kazuhiko Shinoda, Hirotaka Kaji, and Masashi Sugiyama. Binary classification from positive data with skewed confidence. In IJCAI, 2021.

Guangxin Su, Weitong Chen, and Miao Xu. Positive-unlabeled learning from imbalanced data. In IJCAI, 2021.

Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density ratio estimation in machine learning. Cambridge University Press, 2012.

Masashi Sugiyama, Han Bao, Takashi Ishida, Nan Lu, and Tomoya Sakai. Machine learning from weak supervision: an empirical risk minimization approach. MIT Press, 2022.

Xijia Tang, Chao Xu, Hong Tao, Xiaoyu Ma, and Chenping Hou. Confidence-based pu learning with instance-dependent label noise. IEEE Transactions on Neural Networks and Learning Systems, 2025.

Paweł Teisseyre, Timo Martens, Jessa Bekker, and Jesse Davis. Learning from biased positiveunlabeled data via threshold calibration. In AISTATS, 2025.

Ryota Ushio, Takashi Ishida, and Masashi Sugiyama. Practical estimation of the optimal classification error with soft labels and calibration. In ICLR, 2026.

Guanjin Wang, Stephen Wai Hang Kwok, Mohammed Yousufuddin, and Ferdous Sohel. A novel auc maximization imbalanced learning approach for predicting composite outcomes in covid-19 hospitalized patients. IEEEjournal ofbiomedical and health informatics, 2023a.

Wei Wang, Lei Feng, Yuchen Jiang, Gang Niu, Min-Ling Zhang, and Masashi Sugiyama. Binary classification with confidence difference. In NeurIPS, 2023b.

Gill Ward, Trevor Hastie, Simon Barry, Jane Elith, and John R Leathwick. Presence-only data and the em algorithm. Biometrics, 65(2):554–563, 2009.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

Zheng Xie and Ming Li. Semi-supervised auc optimization without guessing labels of unlabeled data. In AAAI, 2018.

Zheng Xie, Yu Liu, Hao-Yuan He, Ming Li, and Zhi-Hua Zhou. Weakly supervised auc optimization: a unified partial auc approach. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Chen Yang, Minghan Zhao, Chenyu Zhu, Suiwei Xie, and Yifei Chen. Automated detection of breast cancer metastases. In DSInS, 2021.

Tianbao Yang and Yiming Ying. Auc maximization in the era of big data and ai: A survey. ACM computing surveys, 55(8):1–37, 2022.

Yiming Ying, Longyin Wen, and Siwei Lyu. Stochastic online auc maximization. NeurIPS, 2016.

Zhuoning Yuan, Zhishuai Guo, Nitesh Chawla, and Tianbao Yang. Compositional training for endto-end deep auc maximization. In ICLR, 2021a.

Zhuoning Yuan, Yan Yan, Milan Sonka, and Tianbao Yang. Large-scale robust deep auc maximization: a new surrogate loss and empirical studies on medical image classification. In ICCV, 2021b.

## A EXTENDED RELATED WORK

Kumagai et al. (2025a) have recently proposed a distribution shift adaptation method for AUC maximization. Specifically, as distribution shift, this paper assumes the covariate shift, which refers to the situation in which the distribution of input instance changes while the input-output (class label) relationship remains unchanged between the training and test phases. This method uses not only PU data from the training phase but also unlabeled data from the test phase to address the covariate shift. The PU data in the training phase can be regarded as biased data. Specifically, although the distributions of labeled positive data and unlabeled positive data in the training phase are identical (i.e., the SCAR setting), the distributions of labeled positive data in the training phase and the unlabeled positive data in the test phase may differ due to the covariate shift (i.e., a SARlike setting). While this approach requires a large amount of unlabeled data obtained in the training phase to handle covariate shift, such unlabeled data are not available in our problem setting. Instead, the proposed method requires confidence information for a small amount of positive data.

## B THEORETICAL ANALYSIS

We first show a basic relationship between the supports of the labeling probability and the posterior probability, and derive an expectation identity used in the following analysis.

Lemma B.1. Let $\eta ( \mathbf { x } ) : = p ( y = 1 | \mathbf { x } ) , u ( \mathbf { x } ) : = p ( o = 1 | \mathbf { x } )$ , and α $: = p ( o = 1 )$ . Under the PU assumption and the positivity condition in Section 4.1,

$$
\begin{array} { r } { I ( u ( { \bf x } ) > 0 ) = I ( \eta ( { \bf x } ) > 0 ) \quad p { - } a l m o s t s u r e l y , } \end{array}\tag{19}
$$

and $u ( \mathbf { x } ) > 0 f o r p ^ { \mathrm { l } }$ -almost every x. Moreover,for any measurablefunction $\varphi$ such that thefollowing expectations arefinite,

$$
\mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \frac { \varphi ( \mathbf { x } ) } { u ( \mathbf { x } ) } \right] = \frac { 1 } { \alpha } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \varphi ( \mathbf { x } ) I ( \eta ( \mathbf { x } ) > 0 ) \right] .\tag{20}
$$

Proof. By Eq. (98) in Section C.3,

$$
u ( { \bf x } ) = e ( { \bf x } ) \eta ( { \bf x } ) .\tag{21}
$$

Therefore, $u ( { \bf x } ) > 0$ implies $\eta ( \mathbf { x } ) > 0$

To show the reverse implication, the positivity condition implies

$$
\int _ { \{ { \bf x } : e ( { \bf x } ) = 0 \} } p ^ { \mathrm { p } } ( { \bf x } ) d { \bf x } = 0 .\tag{22}
$$

Therefore, letting $A : = \{ \mathbf { x } : e ( \mathbf { x } ) = 0 , \eta ( \mathbf { x } ) > 0 \}$ , we have

$$
\begin{array} { l } { { \displaystyle 0 = \int _ { A } p ^ { \mathrm { p } } ( { \bf x } ) d { \bf x } } } \\ { { \displaystyle ~ = \frac { 1 } { \pi } \int _ { A } \eta ( { \bf x } ) p ( { \bf x } ) d { \bf x } } , } \end{array}\tag{23}
$$

where we used $p ^ { \mathrm { p } } ( \mathbf { x } ) = \eta ( \mathbf { x } ) p ( \mathbf { x } ) / \pi$ . Since $\eta ( \mathbf { x } ) > 0$ on A, this implies

$$
\int _ { A } p ( \mathbf { x } ) d \mathbf { x } = 0 .\tag{24}
$$

Thus, $e ( \mathbf { x } ) > 0$ for p-almost every x such that $\eta ( \mathbf { x } ) > 0$ . Since $u ( { \bf x } ) = e ( { \bf x } ) \eta ( { \bf x } )$ , it follows that

$$
\eta ( \mathbf { x } ) > 0 \quad \implies \quad u ( \mathbf { x } ) > 0 \qquad p \mathrm { - a l m o s t ~ s u r e l y } .\tag{25}
$$

Combining the two implications, we obtain

$$
\begin{array} { r } { I ( u ( \mathbf { x } ) > 0 ) = I ( \eta ( \mathbf { x } ) > 0 ) \quad p \mathrm { - a l m o s t ~ s u r e l y } . } \end{array}\tag{26}
$$

Next, by Bayes’ rule,

$$
p ^ { 1 } ( \mathbf { x } ) = \frac { u ( \mathbf { x } ) p ( \mathbf { x } ) } { \alpha } .\tag{27}
$$

Therefore,

$$
\begin{array} { l c l } { \displaystyle \int _ { \{ \mathbf { x } : u ( \mathbf { x } ) = 0 \} } p ^ { 1 } ( \mathbf { x } ) d \mathbf { x } = \frac { 1 } { \alpha } \int _ { \{ \mathbf { x } : u ( \mathbf { x } ) = 0 \} } u ( \mathbf { x } ) p ( \mathbf { x } ) d \mathbf { x } } \\ { \displaystyle = 0 . } \end{array}\tag{28}
$$

Thus, $u ( { \bf x } ) > 0$ for $p ^ { 1 } .$ -almost every x. We can therefore restrict the expectation under $p ^ { 1 }$ to the region where $u ( { \bf x } ) > 0$ . Using Eq. (27), we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \frac { \varphi ( \mathbf { x } ) } { u ( \mathbf { x } ) } \right] } \\ & { = \int _ { \{ \mathbf { x } : u ( \mathbf { x } ) > 0 \} } p ^ { 1 } ( \mathbf { x } ) \frac { \varphi ( \mathbf { x } ) } { u ( \mathbf { x } ) } d \mathbf { x } } \\ & { = \frac { 1 } { \alpha } \int _ { \{ \mathbf { x } : u ( \mathbf { x } ) > 0 \} } p ( \mathbf { x } ) \varphi ( \mathbf { x } ) d \mathbf { x } } \\ & { = \frac { 1 } { \alpha } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \varphi ( \mathbf { x } ) I ( u ( \mathbf { x } ) > 0 ) \right] } \\ & { = \frac { 1 } { \alpha } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \varphi ( \mathbf { x } ) I ( \eta ( \mathbf { x } ) > 0 ) \right] , } \end{array}\tag{29}
$$

where the last equality follows from Eq. (19).

## B.1 PROOF OF THE ROBUSTNESS STATEMENT UNDER STRICTLY INCREASING TRANSFORMATIONS OF POSITIVE-CONFIDENCE

Proof. Let $\eta ( \mathbf { x } ) : = p ( y = 1 | \mathbf { x } )$ and $u ( \mathbf { x } ) : = p ( o = 1 | \mathbf { x } )$ . Since the multiplicative coefficient in Eq. (18) is positive and the additive term is independent of s, minimizing $\hat { \mathcal { R } } _ { \sigma } ^ { h } ( s )$ is equivalent to minimizing

$$
\widetilde { \mathcal { R } } _ { \sigma } ^ { h } ( s ) : = \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \frac { h ( \eta ( \mathbf { x } ^ { \mathrm { p } } ) ) } { u ( \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] .\tag{30}
$$

Applying Lemma B.1 with

$$
\varphi ( \mathbf { x } ^ { \mathrm { p } } ) = h ( \eta ( \mathbf { x } ^ { \mathrm { p } } ) ) \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } [ f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) ] ,\tag{31}
$$

we obtain

$$
\widetilde { \mathcal { R } } _ { \sigma } ^ { h } ( s ) = \frac { 1 } { \alpha } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } \sim p ( \mathbf { x } ) } \left[ h ( \eta ( \mathbf { x } ^ { \mathrm { p } } ) ) I ( \eta ( \mathbf { x } ^ { \mathrm { p } } ) > 0 ) f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] .\tag{32}
$$

Define

$$
g _ { h } ( z ) : = h ( z ) I ( z > 0 ) ,\tag{33}
$$

and

$$
Z _ { h } : = \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } [ g _ { h } ( \eta ( \mathbf { x } ) ) ] ,
$$

$$
q _ { h } ( \mathbf { x } ) : = \frac { g _ { h } ( \eta ( \mathbf { x } ) ) p ( \mathbf { x } ) } { Z _ { h } } .\tag{34}
$$

Since $\pi = \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } [ \eta ( \mathbf { x } ) ] > 0$ and $g _ { h } ( z ) > 0$ for $z > 0 ,$ , we have $Z _ { h } > 0$ . Then,

$$
\displaystyle \widetilde { \mathcal { R } } _ { \sigma } ^ { h } ( s ) = \frac { Z _ { h } } { \alpha } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim q _ { h } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } [ f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) ] .\tag{35}
$$

Thus, up to a positive multiplicative constant, $\mathcal { \widetilde { R } } _ { \sigma } ^ { h }$ is the sigmoid AUC surrogate for bipartite ranking between $q _ { h }$ and $p .$ By the AUC consistency of the sigmoid loss (Charoenphakdee et al., 2019) and the Bayes characterization of bipartite ranking (Menon & Williamson, 2016), the Bayes-optimal ranking for this problem is determined by

$$
{ \frac { q _ { h } ( \mathbf { x } ) } { p ( \mathbf { x } ) } } = { \frac { g _ { h } ( \eta ( \mathbf { x } ) ) } { Z _ { h } } } .\tag{36}
$$

Since $h : [ 0 , 1 ] \to [ 0 , 1 ]$ is strictly increasing, $g _ { h }$ is also strictly increasing on $\lbrack 0 , 1 \rbrack :$ : for $z > 0$ $g _ { h } ( z ) = \dot { h ( z ) } > h ( \mathsf { \dot { 0 } } ) \overset { , } { \geq } 0 = g _ { h } \mathsf { \dot { ( } 0 ) }$ , and for $0 ^ { \cdot } < z _ { 1 } < z _ { 2 } , g _ { h } ( z _ { 1 } ) = h ( \bar { z _ { 1 } } ) < h ( \bar { z _ { 2 } } ) = g _ { h } ( z _ { 2 } )$ Thus, the Bayes-optimal ranking induced by $q _ { h } ( \mathbf { x } ) / p ( \mathbf { x } )$ is identical to that induced by $\eta ( \mathbf { x } )$

On the other hand, the Bayes-optimal ranking for the original AUC risk in Eq. (9) is determined by

$$
\frac { p ^ { \mathrm { p } } ( \mathbf { x } ) } { p ^ { \mathrm { n } } ( \mathbf { x } ) } = \frac { 1 - \pi } { \pi } \frac { \eta ( \mathbf { x } ) } { 1 - \eta ( \mathbf { x } ) } ,\tag{37}
$$

which is also strictly increasing in $\eta ( \mathbf { x } )$ . Thus, the transformed objective and the original AUC risk induce the same Bayes-optimal ranking for the AUC. □

## B.2 SENSITIVITY TO LABELING-PROBABILITY ESTIMATION

In practice, the labeling probability $p ( o = 1 | \mathbf { x } )$ in Eq. (15) is unknown and must be estimated from the given PU data. Here, we analyze how its estimation error affects the rewritten AUC risk and the resulting ranking.

Proposition B.2. Let $\eta ( \mathbf { x } ) : = p ( y = 1 | \mathbf { x } ) a n d u ( \mathbf { x } ) : = p ( o = 1 | \mathbf { x } )$ . For an estimate ${ \widehat { u } } ( \mathbf { x } ) o f u ( \mathbf { x } )$ define

$$
\mathcal { R } _ { \sigma } ( s ; \widehat { u } ) : = \frac { c } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \frac { \eta ( \mathbf { x } ^ { \mathrm { p } } ) } { \widehat { u } ( \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{38}
$$

Note that $\mathcal { R } _ { \sigma } ( s ; u ) = \mathcal { R } _ { \sigma } ( s )$ in $E q .$ . (15). Suppose that $\widehat { u } ( \mathbf { x } ) \geq \delta > 0$ for all x and

$$
| \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) | \leq \epsilon\tag{39}
$$

for p-almost every x satisfying $u ( { \bf x } ) > 0$ . Then,

$$
\operatorname* { s u p } _ { s } | \mathcal { R } _ { \sigma } ( s ; \widehat { u } ) - \mathcal { R } _ { \sigma } ( s ; u ) | \leq \frac { \epsilon } { ( 1 - \pi ) \delta } .\tag{40}
$$

Furthermore, suppose that $\widehat { s }$ is an approximate minimizer satisfying

$$
\mathcal { R } _ { \sigma } ( \widehat { s } ; \widehat { u } ) \leq \operatorname* { i n f } _ { s } \mathcal { R } _ { \sigma } ( s ; \widehat { u } ) + \xi _ { \mathrm { o p t } } ,\tag{41}
$$

where $\xi _ { \mathrm { o p t } } \ge 0 .$ . Then,

$$
\mathcal { R } _ { \sigma } ( \widehat { s } ; u ) - \operatorname* { i n f } _ { s } \mathcal { R } _ { \sigma } ( s ; u ) \leq \frac { 2 \epsilon } { ( 1 - \pi ) \delta } + \xi _ { \mathrm { o p t } } .\tag{42}
$$

Proof. Since $0 \leq f \leq 1$

$$
\begin{array} { l } { \displaystyle \lvert \mathcal { R } _ { \sigma } ( s ; \widehat { u } ) - \mathcal { R } _ { \sigma } ( s ; u ) \rvert } \\ { \displaystyle \leq \frac { c } { 1 - \pi } \mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \eta ( \mathbf { x } ) \frac { \displaystyle \lvert \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) \rvert } { \displaystyle u ( \mathbf { x } ) \widehat { u } ( \mathbf { x } ) } \right] . } \end{array}\tag{43}
$$

Applying Lemma B.1 with

$$
\varphi ( \mathbf x ) = \eta ( \mathbf x ) \frac { | \widehat { u } ( \mathbf x ) - u ( \mathbf x ) | } { \widehat { u } ( \mathbf x ) } ,\tag{44}
$$

we obtain

$$
\begin{array} { l } { \displaystyle | \mathcal { R } _ { \sigma } ( s ; \widehat { u } ) - \mathcal { R } _ { \sigma } ( s ; u ) | } \\ { \displaystyle \leq \frac { c } { \alpha ( 1 - \pi ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \eta ( \mathbf { x } ) \frac { | \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) | } { \widehat { u } ( \mathbf { x } ) } I ( \eta ( \mathbf { x } ) > 0 ) \right] } \\ { \displaystyle \leq \frac { c \epsilon } { \alpha ( 1 - \pi ) \delta } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \eta ( \mathbf { x } ) I ( \eta ( \mathbf { x } ) > 0 ) \right] } \\ { \displaystyle = \frac { c \epsilon } { \alpha ( 1 - \pi ) \delta } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } [ \eta ( \mathbf { x } ) ] = \frac { \epsilon } { ( 1 - \pi ) \delta } , } \end{array}\tag{45}
$$

where the second inequality follows from $I ( \eta ( \mathbf { x } ) \ > \ 0 ) \ = \ I ( u ( \mathbf { x } ) \ > \ 0 )$ p-almost surely in Lemma B.1 and the assumptions on $\widehat { u } .$ In the last equality, we used $\bar { \mathbb { E } } _ { p } [ \eta ( \mathbf { x } ) ] = \pi$ and $\alpha = c \pi$ By taking the supremum over s, we obtain Eq. (40).

By Eq. (40),

$$
\begin{array} { r l } & { \mathcal { R } _ { \sigma } ( \widehat { s } ; u ) \leq \mathcal { R } _ { \sigma } ( \widehat { s } ; \widehat { u } ) + \frac { \epsilon } { ( 1 - \pi ) \delta } } \\ & { \qquad \leq \displaystyle \operatorname* { i n f } _ { s } \mathcal { R } _ { \sigma } ( s ; \widehat { u } ) + \xi _ { \mathrm { o p t } } + \frac { \epsilon } { ( 1 - \pi ) \delta } } \\ & { \qquad \leq \displaystyle \operatorname* { i n f } _ { s } \mathcal { R } _ { \sigma } ( s ; u ) + \xi _ { \mathrm { o p t } } + \frac { 2 \epsilon } { ( 1 - \pi ) \delta } , } \end{array}\tag{46}
$$

which proves the second claim.

Eq. (40) shows that, for fixed $\pi$ and $\delta ,$ the deviation of the rewritten AUC risk increases at most linearly with the labeling probability estimation error $\epsilon .$ The factor $1 / \delta$ also indicates that the effect of estimation errors can be amplified when the estimated labeling probability is close to zero. This provides theoretical motivation for the clipping used in our experiments.

We next show how misspecification of the labeling probability affects the resulting ranking. Using $p ^ { 1 } ( \mathbf { x } ) = u ( \mathbf { x } ) p ( \mathbf { x } ) / \alpha$ , the part of $\mathcal { R } _ { \sigma } ( s ; \widehat { u } )$ that depends on s can, up to a positive multiplicative constant, be written as

$$
\mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \eta ( \mathbf { x } ^ { \mathrm { p } } ) \frac { u ( \mathbf { x } ^ { \mathrm { p } } ) } { \widehat { u } ( \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] .\tag{47}
$$

Define

$$
\widetilde { \eta } ( \mathbf { x } ) : = \eta ( \mathbf { x } ) \frac { u ( \mathbf { x } ) } { \widehat { u } ( \mathbf { x } ) } .\tag{48}
$$

By the same Bayes characterization of bipartite ranking used in Section B.1 (Menon & Williamson, $2 0 1 6 )$ , the Bayes-optimal ranking associated with this objective is determined by $\widetilde { \eta } ( \mathbf { x } )$

This result shows that some estimation errors in the labeling probability do not affect the resulting ranking. First, suppose that ${ \widehat { u } } ( \mathbf { x } ) = a u ( \mathbf { x } )$ for some constant $a > 0$ for p-almost every x satisfying $u ( { \bf x } ) > 0$ . On this set,

$$
\widetilde { \eta } ( \mathbf { x } ) = \eta ( \mathbf { x } ) \frac { u ( \mathbf { x } ) } { \widehat { u } ( \mathbf { x } ) } = \frac { \eta ( \mathbf { x } ) } { a } .\tag{49}
$$

On the other hand, by Lemma B.1, $u ( { \bf x } ) = 0$ implies $\eta ( \mathbf { x } ) = 0$ p-almost surely. In addition, since $\widetilde { \eta } ( \mathbf { x } ) = 0$ when $u ( { \bf x } ) = 0$ , we have

$$
{ \widetilde { \eta } } ( \mathbf { x } ) = { \frac { \eta ( \mathbf { x } ) } { a } } \quad p { \mathrm { - a l m o s t ~ s u r e l y } } .\tag{50}
$$

This almost-sure equality is sufficient for the population AUC. Since $p ( \mathbf { x } ) = \pi p ^ { \mathrm { p } } ( \mathbf { x } ) + ( 1 - \pi ) p ^ { \mathrm { n } } ( \mathbf { x } )$ with $0 < \pi < 1$ , any p-null set is also null under both $p ^ { \mathrm { p } }$ and $p ^ { \mathrm { n } }$ . Hence, the pairwise ordering induced by ηe and $\eta / a$ is identical $p ^ { \mathrm { p } } \times p ^ { \mathrm { n } }$ -almost surely. Since $a > 0 , \eta ( \mathbf { x } ) / a$ preserves the ordering induced by $\eta ( \mathbf { x } )$ . Thus, a constant multiplicative error in the estimated labeling probability does not change the Bayes-optimal AUC ranking.

More generally, suppose that the relative estimation error is uniformly bounded as

$$
\left| \frac { \widehat { u } ( \mathbf { x } ) } { u ( \mathbf { x } ) } - 1 \right| \leq \rho < 1\tag{51}
$$

for $p \mathrm { - }$ almost every x satisfying $u ( { \bf x } ) > 0$ . Then,

$$
1 - \rho \le \frac { \widehat { u } ( \mathbf { x } ) } { u ( \mathbf { x } ) } \leq 1 + \rho ,\tag{52}
$$

and thus

$$
\frac { \eta ( \mathbf { x } ) } { 1 + \rho } \leq \widetilde { \eta } ( \mathbf { x } ) \leq \frac { \eta ( \mathbf { x } ) } { 1 - \rho } .\tag{53}
$$

For two instances x and x¯ outside the corresponding p-null set satisfying $\eta ( \mathbf { x } ) > \eta ( \bar { \mathbf { x } } ) > 0$ , their ordering is guaranteed to be preserved if the smallest possible value of $\widetilde { \eta } ( \mathbf { x } )$ is larger than the largest possible value of $\widetilde { \eta } ( \bar { \bf x } )$ , namely, if

$$
\frac { \eta ( \mathbf { x } ) } { 1 + \rho } > \frac { \eta ( \bar { \mathbf { x } } ) } { 1 - \rho } ,\tag{54}
$$

or equivalently,

$$
\frac { \eta ( \mathbf { x } ) } { \eta ( \bar { \mathbf { x } } ) } > \frac { 1 + \rho } { 1 - \rho } .\tag{55}
$$

If $\eta ( \bar { \mathbf { x } } ) = 0 < \eta ( \mathbf { x } )$ , their ordering is also preserved outside the corresponding p-null set. Therefore, under the relative-error bound in Eq. (51), a pair can have its ordering reversed only when their posterior probabilities are sufficiently close.

## B.3 GENERALIZATION ERROR ANALYSIS

We next provide a finite-sample analysis of learning with the proposed AUC risk estimator. We consider finite score function and labeling probability classes to analyze the basic convergence behavior of the estimator.

Let S be a finite class of score functions, and let $\mathcal { U } _ { \delta }$ be a finite class of labeling probability functions such that

$$
v ( \mathbf { x } ) \geq \delta > 0 \quad { \mathrm { f o r ~ a l l ~ } } v \in \mathcal { U } _ { \delta } { \mathrm { ~ a n d ~ } } \mathbf { x } .\tag{56}
$$

Define $K : = | \mathcal { S } | | \mathcal { U } _ { \delta } | . \ \mathrm { L e t } \ \widetilde { r } ( \mathbf { x } ) \in [ 0 , 1 ]$ denote the observed, possibly noisy confidence. To make the dependence on the score function explicit, we write

$$
f _ { s } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) : = \sigma ( - s ( \mathbf { x } ) + s ( \mathbf { x } ^ { \prime } ) ) .\tag{57}
$$

For $s \in S$ and $v \in \mathcal { U } _ { \delta }$ , define

$$
h _ { s , v } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) : = \frac { \widetilde { r } ( \mathbf { x } ) } { v ( \mathbf { x } ) } f _ { s } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) .\tag{58}
$$

Since $0 \leq \widetilde { r } ( \mathbf { x } ) \leq 1 , 0 \leq f _ { s } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \leq 1 , \mathrm { a n d } v ( \mathbf { x } ) \geq \delta _ { }$ , we have

$$
0 \leq h _ { s , v } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \leq \frac { 1 } { \delta } .\tag{59}
$$

Define the empirical positive-positive and positive-unlabeled terms corresponding to Eq. (17) as

$$
\widehat { L } _ { \mathrm { p p } } ( s , v ) : = \frac { 1 } { N ^ { \mathrm { p } } ( N ^ { \mathrm { p } } - 1 ) } \sum _ { n \neq m } ^ { N ^ { \mathrm { p } } , N ^ { \mathrm { p } } } h _ { s , v } ( \mathbf { x } _ { n } ^ { \mathrm { p } } , \mathbf { x } _ { m } ^ { \mathrm { p } } ) ,\tag{60}
$$

$$
\widehat { L } _ { \mathrm { p u } } ( s , v ) : = \frac { 1 } { N ^ { \mathrm { p } } N } \sum _ { n = 1 } ^ { N ^ { \mathrm { p } } } \sum _ { m = 1 } ^ { N } h _ { s , v } ( \mathbf { x } _ { n } ^ { \mathrm { p } } , \mathbf { x } _ { m } ) .\tag{61}
$$

The corresponding population terms are

$$
L _ { \mathrm { p p } } ( s , v ) : = \mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \prime } \sim p ^ { 1 } ( \mathbf { x } ) } [ h _ { s , v } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) ] ,\tag{62}
$$

$$
L _ { \mathrm { p u } } ( s , v ) : = \mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } ^ { \prime } \sim p ^ { \mathrm { u } } ( \mathbf { x } ) } [ h _ { s , v } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) ] .\tag{63}
$$

Let $\alpha : = p ( o = 1 )$ and define

$$
L ( s , v ) : = \alpha L _ { \mathrm { p p } } ( s , v ) + ( 1 - \alpha ) L _ { \mathrm { p u } } ( s , v ) .\tag{64}
$$

We first derive a uniform finite-sample bound. For $\tau \in ( 0 , 1 )$ ), define

$$
B _ { N ^ { \mathrm { p } } , N } ( \tau ) : = \frac 1 \delta \left[ \alpha \sqrt { \frac { 2 \log ( 4 K / \tau ) } { N ^ { \mathrm { p } } } } + ( 1 - \alpha ) \sqrt { \frac { \log ( 4 K / \tau ) } { 2 } \left( \frac { 1 } { N ^ { \mathrm { p } } } + \frac { 1 } { N } \right) } \right] .\tag{65}
$$

Lemma B.3. With probability at least $1 - \tau$

$$
\operatorname* { s u p } _ { s \in \mathcal { S } } \Big | \alpha \widehat { L } _ { \mathrm { p p } } ( s , v ) + ( 1 - \alpha ) \widehat { L } _ { \mathrm { p u } } ( s , v ) - L ( s , v ) \Big | \leq B _ { N ^ { \mathrm { p } } , N } ( \tau ) .\tag{66}
$$

Proof. Consider a fixed pair $( s , v )$ . Replacing one labeled positive instance changes at most $2 ( N ^ { \mathrm { p } } -$ 1) summands in $\widehat { L } _ { \mathrm { p p } } ( s , v )$ . Thus, by Eq. (59), replacing one labeled positive instance changes $\widehat { L } _ { \mathrm { p p } } ( s , v )$ by at most $2 / ( N ^ { \mathrm { p } } \delta )$ ). As a result, McDiarmid’s inequality gives

$$
\operatorname* { P r } \left( \Big | \widehat { L } _ { \mathrm { p p } } ( s , v ) - L _ { \mathrm { p p } } ( s , v ) \Big | \geq t \right) \leq 2 \exp \left( - \frac { N ^ { \mathrm { p } } \delta ^ { 2 } t ^ { 2 } } { 2 } \right) .\tag{67}
$$

For $\widehat { L } _ { \mathrm { p u } } ( s , v )$ , replacing one labeled positive instance changes N summands and thus changes the average by at most $1 / ( \bar { N } ^ { \mathrm { p } } \delta )$ . Similarly, replacing one unlabeled instance changes $N ^ { \mathrm { p } }$ summands and thus changes the average by at most $1 / { \bar { ( } N \delta ) }$ . As a result, McDiarmid’s inequality gives

$$
\operatorname* { P r } \left( \Big | \widehat { L } _ { \mathrm { p u } } ( s , v ) - L _ { \mathrm { p u } } ( s , v ) \Big | \geq t \right) \leq 2 \exp \left( - \frac { 2 \delta ^ { 2 } t ^ { 2 } } { 1 / N ^ { \mathrm { p } } + 1 / N } \right) .\tag{68}
$$

Applying a union bound over the $K = | S | | \mathcal { U } _ { \delta } |$ pairs in $S \times \mathcal { U } _ { \delta }$ and over the two empirical terms yields, with probability at least $1 - \tau .$

$$
\operatorname* { s u p } _ { s \in S \atop v \in \mathcal { U } _ { \delta } } \Big | \widehat { L } _ { \mathrm { p p } } ( s , v ) - L _ { \mathrm { p p } } ( s , v ) \Big | \leq \frac { 1 } { \delta } \sqrt { \frac { 2 \log ( 4 K / \tau ) } { N ^ { \mathrm { p } } } } ,\tag{69}
$$

$$
\operatorname* { s u p } _ { s \in \mathcal { S } } \Big | \widehat { L } _ { \mathrm { p u } } ( s , v ) - L _ { \mathrm { p u } } ( s , v ) \Big | \leq \frac { 1 } { \delta } \sqrt { \frac { \log ( 4 K / \tau ) } { 2 } \left( \frac { 1 } { N ^ { \mathrm { p } } } + \frac { 1 } { N } \right) } .\tag{70}
$$

Eq. (66) follows by combining these two inequalities with Eq. (64).

In practice, α is estimated as

$$
\widehat { \alpha } : = \frac { N ^ { \mathrm { p } } } { N ^ { \mathrm { p } } + N } .\tag{71}
$$

Define its estimation error as

$$
\epsilon _ { \alpha } : = | \widehat { \alpha } - \alpha | .\tag{72}
$$

Since $0 \leq \widehat { L } _ { \mathrm { p p } } ( s , v ) , \widehat { L } _ { \mathrm { p u } } ( s , v ) \leq 1 / \delta ,$ we have

$$
\begin{array} { r l } & { \Big | \widehat { \alpha } \widehat { L } _ { \mathrm { p p } } ( s , v ) + ( 1 - \widehat { \alpha } ) \widehat { L } _ { \mathrm { p u } } ( s , v ) - \alpha \widehat { L } _ { \mathrm { p p } } ( s , v ) - ( 1 - \alpha ) \widehat { L } _ { \mathrm { p u } } ( s , v ) \Big | } \\ & { = | \widehat { \alpha } - \alpha | \Big | \widehat { L } _ { \mathrm { p p } } ( s , v ) - \widehat { L } _ { \mathrm { p u } } ( s , v ) \Big | } \\ & { \leq \frac { \epsilon _ { \alpha } } { \delta } . } \end{array}\tag{73}
$$

Thus, on the event in Eq. (66),

$$
\operatorname* { s u p } _ { s \in \mathcal { S } \atop v \in \mathcal { U } _ { \delta } } \Big | \widehat { \alpha } \widehat { L } _ { \mathrm { p p } } ( s , v ) + ( 1 - \widehat { \alpha } ) \widehat { L } _ { \mathrm { p u } } ( s , v ) - L ( s , v ) \Big |
$$

$$
\leq B _ { N ^ { \mathrm { p } } , N } ( \tau ) + \frac { \epsilon _ { \alpha } } { \delta } .\tag{74}
$$

Thus, Eq. (74) also holds with probability at least $1 - \tau$

For arbitrary confidence $\widetilde { r }$ and $v \in { \mathcal { U } } _ { \delta } .$ define the population and empirical rewritten risks by

$$
\mathcal { R } _ { \sigma } ( s ; \widetilde { r } , v ) : = \frac { c } { 1 - \pi } L ( s , v ) - \frac { \pi } { 2 ( 1 - \pi ) } ,\tag{75}
$$

$$
\widehat { \mathcal { R } } _ { \sigma } ( s ; \widetilde { r } , v ) : = \frac { c } { 1 - \pi } \left[ \widehat { \alpha } \widehat { L } _ { \mathrm { p p } } ( s , v ) + ( 1 - \widehat { \alpha } ) \widehat { L } _ { \mathrm { p u } } ( s , v ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{76}
$$

Define

$$
\Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) : = \frac { c } { 1 - \pi } \left[ B _ { N ^ { \mathrm { p } } , N } ( \tau ) + \frac { \epsilon _ { \alpha } } { \delta } \right] .\tag{77}
$$

Then, Eq. (74) implies that, with probability at least $1 - \tau .$

$$
\operatorname* { s u p } _ { \stackrel { s \in S } { v \in \mathcal { U } _ { \delta } } } \left| \widehat { \mathcal { R } } _ { \sigma } ( s ; \widetilde { r } , v ) - \mathcal { R } _ { \sigma } ( s ; \widetilde { r } , v ) \right| \leq \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) .\tag{78}
$$

We next define the population risk corresponding to the true confidence and labeling probability. For the following analysis, we interpret $\eta ( \mathbf { x } ) / u ( \mathbf { x } )$ as zero when $u ( { \bf x } ) = 0$ . By Lemma B.1, $u ( { \bf x } ) > 0$ for $p ^ { 1 } .$ -almost every $\mathbf { x } .$ Therefore, this convention does not affect expectations with respect to p<sup>l</sup> and thus does not change the rewritten AUC risk in Eq. (15).

We denote this population risk by

$$
\mathcal { R } _ { \sigma } ( s ; \eta , u ) : = \frac { c } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathsf { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \frac { \eta ( \mathbf { x } ^ { \mathsf { p } } ) } { u ( \mathbf { x } ^ { \mathsf { p } } ) } f _ { s } ( \mathbf { x } ^ { \mathsf { p } } , \mathbf { x } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{79}
$$

By Eq. (15), $\mathcal { R } _ { \sigma } ( s ; \eta , u )$ is equal to the original population AUC risk.

We next show the excess-risk bound for the score function learned with the proposed estimator.

Theorem B.4. Suppose that the data-dependent estimate $\widehat { u } \in \mathcal { U } _ { \delta }$ and that ${ \widehat { s } } \in S$ is an approximate empirical minimizer satisfying

$$
\widehat { \mathcal { R } } _ { \sigma } ( \widehat { s } ; \widetilde { r } , \widehat { u } ) \leq \operatorname* { i n f } _ { s \in { \mathcal { S } } } \widehat { \mathcal { R } } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) + \xi _ { \mathrm { o p t } } ,\tag{80}
$$

where $\xi _ { \mathrm { o p t } } \ge 0 .$ . Suppose that the PU assumption and the positivity condition in Section 4.1 hold, and that

$$
\begin{array} { r } { | \widetilde { r } ( \mathbf { x } ) - \eta ( \mathbf { x } ) | \le \epsilon _ { r } , } \\ { | \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) | \le \epsilon _ { u } } \end{array}\tag{81}
$$

for p-almost every x satisfying $u ( { \bf x } ) > 0$ . Then, with probability at least $1 - \tau$

$$
\mathcal { R } _ { \sigma } ( \widehat { s } ; \eta , u ) - \operatorname* { i n f } _ { s \in \mathcal { S } } \mathcal { R } _ { \sigma } ( s ; \eta , u ) \leq 2 \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) + \frac { 2 ( c \epsilon _ { r } + \epsilon _ { u } ) } { ( 1 - \pi ) \delta } + \xi _ { \mathrm { o p t } } .\tag{82}
$$

Proof. On the event in Eq. (78), the uniform bound holds simultaneously for all $( s , v ) \in \mathcal { S } \times \mathcal { U } _ { \delta }$ Therefore, since $\widehat { u } \in \mathcal { U } _ { \delta }$ , it also holds for the data-dependent estimate ub:

$$
\operatorname* { s u p } _ { s \in S } \Big | \widehat { \mathcal { R } } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) - \mathcal { R } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) \Big | \leq \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) .\tag{83}
$$

Since $s$ is finite, let

$$
s ^ { \dagger } \in \mathop { \arg \operatorname* { m i n } } _ { s \in \mathcal { S } } \mathcal { R } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) .\tag{84}
$$

Then, by Eqs. (83) and (80), we have

$$
\begin{array} { r l } & { \mathcal { R } _ { \sigma } ( \widehat { s } ; \widetilde { r } , \widehat { u } ) \leq \widehat { \mathcal { R } } _ { \sigma } ( \widehat { s } ; \widetilde { r } , \widehat { u } ) + \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) } \\ & { \qquad \leq \widehat { \mathcal { R } } _ { \sigma } ( s ^ { \dagger } ; \widetilde { r } , \widehat { u } ) + \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) + \xi _ { \mathrm { o p t } } } \\ & { \qquad \leq \mathcal { R } _ { \sigma } ( s ^ { \dagger } ; \widetilde { r } , \widehat { u } ) + 2 \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) + \xi _ { \mathrm { o p t } } . } \end{array}\tag{85}
$$

By the definition of $s ^ { \dagger }$ , this yields

$$
\mathcal { R } _ { \sigma } ( \widehat { s } ; \widetilde { r } , \widehat { u } ) - \operatorname* { i n f } _ { s \in { \mathcal { S } } } \mathcal { R } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) \leq 2 \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) + \xi _ { \mathrm { o p t } } .\tag{86}
$$

We next analyze the effects of the confidence error and the labeling-probability estimation error. For any $s \in S$ , using $0 \leq f _ { s } \leq 1$ , we have

$$
\begin{array} { r l } & { \displaystyle \lvert \mathcal { R } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) - \mathcal { R } _ { \sigma } ( s ; \eta , u ) \rvert } \\ & { \le \displaystyle \frac { c } { 1 - \pi } \mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \left. \frac { \widetilde { r } ( \mathbf { x } ) } { \widehat { u } ( \mathbf { x } ) } - \frac { \eta ( \mathbf { x } ) } { u ( \mathbf { x } ) } \right. \right] . } \end{array}\tag{87}
$$

Here and below, $\eta ( \mathbf { x } ) / u ( \mathbf { x } )$ is interpreted as zero when $u ( { \bf x } ) = 0$ . By Lemma $\mathbf { B } . 1 , u ( \mathbf { x } ) > 0$ for $p ^ { 1 } .$ -almost every x. Thus, we can restrict the expectation in $\mathrm { E q . } \ ( 8 7 )$ to the set $\{ \mathbf { x } : u ( \mathbf { x } ) > 0 \}$ . On this set,

$$
\left| \frac { \widetilde { r } ( \mathbf { x } ) } { \widehat { u } ( \mathbf { x } ) } - \frac { \eta ( \mathbf { x } ) } { u ( \mathbf { x } ) } \right| \leq \frac { \left| \widetilde { r } ( \mathbf { x } ) - \eta ( \mathbf { x } ) \right| } { \widehat { u } ( \mathbf { x } ) } + \eta ( \mathbf { x } ) \frac { \left| \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) \right| } { u ( \mathbf { x } ) \widehat { u } ( \mathbf { x } ) } .\tag{88}
$$

Since ${ \widehat { u } } ( \mathbf { x } ) \geq \delta$ and $| \widetilde { r } ( \mathbf { x } ) - \eta ( \mathbf { x } ) | \leq \epsilon _ { r }$ for p-almost every x satisfying $u ( { \bf x } ) > 0$ , the expectation of the first term is bounded as

$$
\mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \frac { \lvert \widetilde { r } ( \mathbf { x } ) - \eta ( \mathbf { x } ) \rvert } { \widehat { u } ( \mathbf { x } ) } \right] \leq \frac { \epsilon _ { r } } { \delta } .\tag{89}
$$

For the second term, using $p ^ { 1 } ( \mathbf { x } ) = u ( \mathbf { x } ) p ( \mathbf { x } ) / \alpha$ , we obtain

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ I ( u ( \mathbf { x } ) > 0 ) \eta ( \mathbf { x } ) \frac { \vert \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) \vert } { u ( \mathbf { x } ) \widehat { u } ( \mathbf { x } ) } \right] } \\ { = \displaystyle \frac { 1 } { \alpha } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ I ( u ( \mathbf { x } ) > 0 ) \eta ( \mathbf { x } ) \frac { \vert \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) \vert } { \widehat { u } ( \mathbf { x } ) } \right] } \\ { \leq \displaystyle \frac { \epsilon _ { u } } { \alpha \delta } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \eta ( \mathbf { x } ) I ( u ( \mathbf { x } ) > 0 ) \right] . } \end{array}\tag{90}
$$

By Lemma B.1, $I ( u ( { \bf x } ) > 0 ) = I ( \eta ( { \bf x } ) > 0 )$ p-almost surely. Therefore,

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \eta ( \mathbf { x } ) I ( u ( \mathbf { x } ) > 0 ) \right] = \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } [ \eta ( \mathbf { x } ) ] } \\ { = \pi . } \end{array}\tag{91}
$$

Since $\alpha = c \pi$ , Eq. (90) gives

$$
\mathbb { E } _ { \mathbf { x } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ I ( u ( \mathbf { x } ) > 0 ) \eta ( \mathbf { x } ) \frac { \vert \widehat { u } ( \mathbf { x } ) - u ( \mathbf { x } ) \vert } { u ( \mathbf { x } ) \widehat { u } ( \mathbf { x } ) } \right] \leq \frac { \epsilon _ { u } } { c \delta } .\tag{92}
$$

Combining Eqs. (87), (89), and (92), we obtain

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { s \in S } | \mathcal { R } _ { \sigma } ( s ; \widetilde { r } , \widehat { u } ) - \mathcal { R } _ { \sigma } ( s ; \eta , u ) | \leq \displaystyle \frac { c } { 1 - \pi } \left( \displaystyle \frac { \epsilon _ { r } } { \delta } + \displaystyle \frac { \epsilon _ { u } } { c \delta } \right) } \\ { \displaystyle = \displaystyle \frac { c \epsilon _ { r } + \epsilon _ { u } } { ( 1 - \pi ) \delta } . } \end{array}\tag{93}
$$

Let

$$
s ^ { * } \in \arg \operatorname* { m i n } _ { s \in \mathcal { S } } \mathcal { R } _ { \sigma } ( s ; \eta , u ) .\tag{94}
$$

Using Eq. (93) for both $\widehat { s }$ and $s ^ { * }$ and Eq. (86), we obtain

$$
\begin{array} { r l } & { \mathcal { R } _ { \sigma } ( \widehat { s } ; \eta , u ) \leq \mathcal { R } _ { \sigma } ( \widehat { s } ; \widetilde { r } , \widehat { u } ) + \frac { c \epsilon _ { r } + \epsilon _ { u } } { ( 1 - \pi ) \delta } } \\ & { \qquad \leq \mathcal { R } _ { \sigma } ( s ^ { * } ; \widetilde { r } , \widehat { u } ) + 2 \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) + \xi _ { \mathrm { o p t } } + \frac { c \epsilon _ { r } + \epsilon _ { u } } { ( 1 - \pi ) \delta } } \\ & { \qquad \leq \mathcal { R } _ { \sigma } ( s ^ { * } ; \eta , u ) + 2 \Gamma _ { N ^ { \mathrm { p } } , N } ( \tau ) + \xi _ { \mathrm { o p t } } + \frac { 2 ( c \epsilon _ { r } + \epsilon _ { u } ) } { ( 1 - \pi ) \delta } , } \end{array}\tag{95}
$$

which proves Eq. (82).

The bound in Eq. (82) decomposes the excess smoothed AUC risk into the finite-sample statistical error, the estimation error of $\alpha ,$ , the confidence error, the labeling-probability estimation error, and the optimization error. For fixed finite function classes, $B _ { N ^ { \mathrm { p } } , N } ( \tau )$ converges to zero as $N ^ { \mathrm { p } } , N $ $\infty .$ . Thus, the excess risk converges to zero as the sample sizes increase, provided that the estimation errors of $\alpha ,$ the confidence, and the labeling probability, as well as the optimization error, also vanish. Here, we consider finite function classes for a self-contained analysis. Extending the result to infinite function classes using Rademacher complexity is left for future work.

## C DERIVATIONS FOR THE PROPOSED AUC RISK

## C.1 CONSTANCY OF THE POSITIVE-POSITIVE TERM

Lemma C.1. $\begin{array} { r } { \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathrm { p } } , \bar { \mathbf { x } } ^ { \mathrm { p } } ) \right] = \frac { 1 } { 2 } . } \end{array}$

Proof. Since $f ( { \bf x } ^ { \mathrm { p } } , \bar { \bf x } ^ { \mathrm { p } } ) = \sigma ( - s ( { \bf x } ^ { \mathrm { p } } ) + s ( \bar { \bf x } ^ { \mathrm { p } } ) )$ and $\sigma ( z ) + \sigma ( - z ) = 1$ for all $z \in \mathbb { R }$

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathrm { p } } , \bar { \mathbf { x } } ^ { \mathrm { p } } ) \right] = \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ 1 - f ( \bar { \mathbf { x } } ^ { \mathrm { p } } , \mathbf { x } ^ { \mathrm { p } } ) \right] } \\ & { \qquad = 1 - \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ f ( \bar { \mathbf { x } } ^ { \mathrm { p } } , \mathbf { x } ^ { \mathrm { p } } ) \right] . } \end{array}\tag{96}
$$

It is clear that the lemma follows from this equation.

## C.2 DERIVATION OF THE RELATION BETWEEN THE LABELED AND POSITIVE DENSITIES

Lemma C.2. Labeled density $p ^ { 1 } ( { \bf x } ) = p ( { \bf x } | o = 1 )$ can be represented as $\begin{array} { r } { p ^ { 1 } ( \mathbf { x } ) = \frac { e ( \mathbf { x } ) } { c } p ^ { \mathrm { p } } ( \mathbf { x } ) } \end{array}$ , where $e ( \mathbf { x } ) = p ( o = 1 | \mathbf { x } , y = 1 ) , c = p ( o = 1 | y = 1 )$ , and $p ^ { \mathrm { p } } ( \mathbf { x } ) = p ( \mathbf { x } | y = 1 )$

Proof.

$$
{ \begin{array} { r l } & { p ^ { 1 } ( \mathbf { x } ) = p ( \mathbf { x } | o = 1 ) } \\ & { \qquad = p ( \mathbf { x } | o = 1 , y = 1 ) } \\ & { \qquad = { \frac { p ( \mathbf { x } , o = 1 , y = 1 ) } { p ( o = 1 , y = 1 ) } } } \\ & { \qquad = { \frac { p ( o = 1 | \mathbf { x } , y = 1 ) p ( \mathbf { x } | y = 1 ) p ( y = 1 ) } { p ( o = 1 | y = 1 ) p ( y = 1 ) } } } \\ & { \qquad = { \frac { e ( \mathbf { x } ) } { c } } p ^ { \mathrm { p } } ( \mathbf { x } ) , } \end{array} }\tag{97}
$$

where we used the PU property (labeled data are always positive) in the second equality and the Bayes’ rule in the third equality. □

## C.3 RELATION BETWEEN THE PROPENSITY SCORE AND THE LABELING PROBABILITY

Lemma C.3. Labeling probability $p ( o = 1 | \mathbf { x } )$ and propensity score $e ( \mathbf { x } ) = p ( o = 1 | \mathbf { x } , y = 1 )$ satisfy

$$
p ( o = 1 | \mathbf { x } ) = e ( \mathbf { x } ) p ( y = 1 | \mathbf { x } ) .\tag{98}
$$

Consequently, for any x such that $p ( y = 1 | \mathbf { x } ) > 0$

$$
e ( \mathbf { x } ) = { \frac { p ( o = 1 | \mathbf { x } ) } { p ( y = 1 | \mathbf { x } ) } } .\tag{99}
$$

Proof. By the law of total probability,

$$
\begin{array} { c } { { p ( o = 1 | \mathbf { x } ) = p ( o = 1 | \mathbf { x } , y = 1 ) p ( y = 1 | \mathbf { x } ) + p ( o = 1 | \mathbf { x } , y = 0 ) p ( y = 0 | \mathbf { x } ) } } \\ { { { } } } \\ { { = e ( \mathbf { x } ) p ( y = 1 | \mathbf { x } ) , } } \end{array}\tag{100}
$$

where the second equality follows from the definition of the propensity score and the PU assumption $p ( o \mathrm { ~ = ~ } 1 | \mathbf { x } , y \mathrm { ~ = ~ } 0 ) \mathrm { ~ = ~ } 0$ . Dividing both sides by $p ( y = 1 | \mathbf { x } )$ gives Eq. (99). Moreover, since $p ^ { \mathrm { p } } ( \mathbf { x } ) = p ( y = 1 | \mathbf { x } ) p ( \mathbf { x } ) / \pi$ , we have $p ( y = 1 | \mathbf { x } ) > 0$ for $p ^ { \mathrm { p . } }$ -almost every x. Therefore, Eq. (14) holds for $p ^ { \mathrm { p . } }$ -almost every x. □

## D EXTENSIONS OF THE PROPOSED METHOD

## D.1 EXTENSION TO NON-SYMMETRIC LOSS FUNCTIONS

In this section, we derive the AUC risk estimator when using non-symmetric function σ, which also enables us to maximize the AUC without requiring π and c.

Specifically, since the second term in Eq. (11) does not become a constant with non-symmetric functions, we need to calculate this term with biased positive data. Using Eqs. (13) and (14), as in the derivation of the proposed risk in the main paper, the second term in Eq. (11) can be rewritten as

$$
\begin{array} { l } { \displaystyle \frac { \pi } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathsf { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathsf { p } } \sim p ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ f ( \mathbf { x } ^ { \mathsf { p } } , \bar { \mathbf { x } } ^ { \mathsf { p } } ) \right] } \\ { \displaystyle \qquad = \frac { c \alpha } { 1 - \pi } \mathbb { E } _ { \mathbf { x } ^ { \mathsf { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathsf { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \displaystyle \frac { p ( y = 1 | \mathbf { x } ^ { \mathsf { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathsf { p } } ) } \displaystyle \frac { p ( y = 1 | \bar { \mathbf { x } } ^ { \mathsf { p } } ) } { p ( o = 1 | \bar { \mathbf { x } } ^ { \mathsf { p } } ) } f ( \mathbf { x } ^ { \mathsf { p } } , \bar { \mathbf { x } } ^ { \mathsf { p } } ) \right] , } \end{array}\tag{101}
$$

where we used $\alpha = c \pi$

By replacing the constant term $\frac { \pi } { 2 ( 1 - \pi ) }$ in Eq. (16) with the above-derived term, we can obtain the AUC risk with non-symmetric functions as follows:

$$
\begin{array} { l } { \displaystyle \mathcal { R } ( s ) = \frac { c } { 1 - \pi } \left[ \alpha \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \bar { \mathbf { x } } ^ { \mathrm { p } } ) \right] \right. } \\ { \displaystyle \left. + ( 1 - \alpha ) \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ^ { \mathrm { u } } ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \mathbf { x } ) \right] \right. } \\ { \displaystyle \left. - \alpha \mathbb { E } _ { \mathbf { x } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \bar { \mathbf { x } } ^ { \mathrm { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathrm { p } } ) } { p ( o = 1 | \mathbf { x } ^ { \mathrm { p } } ) } \frac { p ( y = 1 | \bar { \mathbf { x } } ^ { \mathrm { p } } ) } { p ( o = 1 | \bar { \mathbf { x } } ^ { \mathrm { p } } ) } f ( \mathbf { x } ^ { \mathrm { p } } , \bar { \mathbf { x } } ^ { \mathrm { p } } ) \right] \right] . } \end{array}\tag{102}
$$

The corresponding AUC risk estimator can be obtained by approximating the expectation with the sample average. Since the coefficient $c / ( 1 - \pi )$ does not affect the optimization of s, we can also perform training without knowing π and c.

## D.2 EXTENSION TO THE TWO-SAMPLE SCENARIO

In the main paper, we assume the one-sample scenario for collecting PU data. In this section, we consider the proposed method for the two-sample scenario. In this scenario, PU data are sampled

independently: labeled positive data are drawn from $p ^ { 1 } ( \mathbf { x } )$ and unlabeled data are drawn from true marginal density $p ( \mathbf { x } )$ . As a result, we are given the following data for training:

$$
\begin{array} { r l } & { X ^ { \mathrm { p } } : = \{ \mathbf { x } _ { n } ^ { \mathrm { p } } \} _ { n = 1 } ^ { N ^ { \mathrm { p } } } \sim p ^ { \mathrm { l } } ( \mathbf { x } ) : = p ( \mathbf { x } | o = 1 ) , } \\ & { X : = \{ \mathbf { x } _ { m } \} _ { m = 1 } ^ { N } \sim p ( \mathbf { x } ) , } \\ & { R ^ { \mathrm { p } } : = \{ r _ { n } ^ { \mathrm { p } } | \ r _ { n } ^ { \mathrm { p } } = p ( y = 1 | \mathbf { x } _ { n } ^ { \mathrm { p } } ) \} _ { n = 1 } ^ { N ^ { \mathrm { p } } } . } \end{array}\tag{103}
$$

In this scenario, the relationship between $p ^ { 1 } ( \mathbf { x } )$ and $p ^ { \mathrm { p } } ( \mathbf { x } )$ in Eq. (13) is also satisfied. Additionally, propensity score $e ( \mathbf { x } )$ can be rewritten as

$$
\begin{array} { r l } & { e ( \mathbf { x } ) = p ( o = 1 | \mathbf { x } , y = 1 ) } \\ & { \quad = \frac { p ( o = 1 , \mathbf { x } , y = 1 ) } { p ( \mathbf { x } , y = 1 ) } } \\ & { \quad = \frac { p ( y = 1 | \mathbf { x } , o = 1 ) p ( \mathbf { x } | o = 1 ) p ( o = 1 ) } { p ( y = 1 | \mathbf { x } ) p ( \mathbf { x } ) } } \\ & { \quad = \frac { p ( \mathbf { x } | o = 1 ) p ( o = 1 ) } { p ( y = 1 | \mathbf { x } ) p ( \mathbf { x } ) } } \\ & { \quad = \frac { \alpha w ( \mathbf { x } ) } { p ( y = 1 | \mathbf { x } ) } , } \end{array}\tag{104}
$$

where we used the Bayes’ rule in the second equality, the PU property (labeled data are always positive), $p ( y = 1 | \mathbf { x } , o = 1 ) = 1$ , in the fourth equality, and set $\begin{array} { r } { w ( \mathbf { x } ) : = \frac { p ( \mathbf { x } | o = 1 ) } { p ( \mathbf { x } ) } } \end{array}$ and $\alpha : = p ( o =$ 1) in the fifth equality. Here, unlike Eq. (14) of the one-sample scenario, we do not use $p ( o = 1 | \mathbf { x } )$ to represent $e ( \mathbf { x } )$ . This is because the numbers of PU data are arbitrarily determined by the users (data collectors) in the two-sample scenario and thus true label ratio α cannot be estimated from the given data (Elkan & Noto, 2008). As a result, $p ( o = 1 | \mathbf { x } ) = \alpha p ( \mathbf { x } | o = 1 ) / p ( \mathbf { x } )$ cannot also be estimated from the given data (Elkan & Noto, 2008). In contrast, density-ratio $w ( \mathbf { x } ) = p ( \mathbf { x } | o = 1 ) / p ( \mathbf { x } )$ in Eq. (104) can be estimated from given PU data without knowing α by using off-the-shelf density-ratio estimation methods (Kanamori et al., 2009; Kato & Teshima, 2021; Kato et al., 2019; Sugiyama et al., 2012).

By substituting Eqs. (13) and (104) into Eq. (12), AUC risk $R _ { \sigma } ( s )$ can be represented as

$$
\mathcal { R } _ { \sigma } ( s ) = \frac { c } { ( 1 - \pi ) \alpha } \mathbb { E } _ { \mathbf { x } ^ { \mathsf { p } } \sim p ^ { 1 } ( \mathbf { x } ) } \mathbb { E } _ { \mathbf { x } \sim p ( \mathbf { x } ) } \left[ \frac { p ( y = 1 | \mathbf { x } ^ { \mathsf { p } } ) } { w ( \mathbf { x } ^ { \mathsf { p } } ) } f ( \mathbf { x } ^ { \mathsf { p } } , \mathbf { x } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } .\tag{105}
$$

Here, coefficient $c / ( ( 1 - \pi ) \alpha )$ and constant $\pi / ( 2 ( 1 - \pi ) )$ do not affect the optimization for score function s and thus we can safely ignore them. The estimator of the AUC risk in Eq. (105) with training data $X ^ { \mathrm { p } } \cup X \cup R ^ { \mathrm { p } }$ can be represented as

$$
\hat { \mathcal { R } } _ { \sigma } ( s ) = \frac { c } { ( 1 - \pi ) \alpha } \left[ \frac { 1 } { N ^ { \mathrm { p } } N } \sum _ { n , m = 1 } ^ { N ^ { \mathrm { p } } , N } \frac { r _ { n } ^ { \mathrm { p } } } { \hat { w } _ { n } ^ { \mathrm { p } } } f ( \mathbf { x } _ { n } ^ { \mathrm { p } } , \mathbf { x } _ { m } ) \right] - \frac { \pi } { 2 ( 1 - \pi ) } ,\tag{106}
$$

where $\hat { w } _ { n } ^ { \mathrm { p } }$ denotes the estimate of density-ratio $w \bigl ( \mathbf { x } _ { n } ^ { \mathrm { p } } \bigr )$ for instance $\mathbf { x } _ { n } ^ { \mathrm { p } }$

## E ADDITIONAL EXPERIMENTAL DETAILS

## E.1 DATASETS AND DATA CONSTRUCTION

Mnist consists of hand-written images of 10 digits. Fmnist consists of images of 10 fashion cate gories. In Mnist and Fmnist, each image is represented by gray scale with 28 × 28 pixels. Svhn consists of images of 10 printed digits clipped from photographs of house number plates. Cifar10 consists of images of 10 animal and vehicle categories. In Svhn and Cifar10, each image is represented by $3 2 \times \mathrm { \bar { 3 2 } ~ R G B }$ pixels. Diabetes is a tabular dataset whose task is to predict whether the respondent has diabetes or not. Each respondent is represented as 142-dimensional features. Blood is a tabular dataset whose task is hypertension diagnosis for high-risk age people. Each survey subject is represented as 100-dimensional features.

We created the binary classification problems from each dataset following the data creation procedure of previous studies (Kumagai et al., 2025b; 2024; 2025a; Xie et al., 2024). Specifically, for Mnist and Svhn, we used odd digits as positive and even digits as negative. For Fmnist, we used non-upper garment classes (1, 5, 7, 8, and 9) as positive and the others as negative. For Cifar10, we used the vehicle classes (0, 1, 8, and 9) as positive and the animal classes as negative. For Diabetes and Blood, we used the original binary class labels.

To create the sampling bias, for Mnist and Svhn, 90% of data were selected from digits 1, 3, and $^ { 5 , }$ and 10% of data were selected from digits 7 and 9 for labeled positive data. For Fmnist, 90% of data were selected from classes 1, 5, and 7, and 10% of data were selected from classes 8 and 9 for labeled positive data. For Cifar10, 90% of data were selected from classes 0, and 1, and 10% of data were selected from classes 8 and 9 for labeled positive data. The remaining unselected data were used for unlabeled data for training and validation.

To create the selection bias in Diabetes and Blood datasets, we followed the procedure used in previous studies (Sakai & Shimizu, 2019; Kumagai et al., 2025a). Specifically, we first calculated the Euclidean distance between positive instance ${ \bf x } _ { n }$ and the mean vector of all positive data x¯, $\mathbf { c } _ { n } : = \| \mathbf { x } _ { n } - \bar { \mathbf { x } } \|$ . Then, we found the median $\mathbf { c } _ { \mathrm { m e d } }$ from all $\left\{ \mathbf { c } _ { n } \right\}$ . We split all $\left\{ \mathbf { c } _ { n } \right\}$ into the first set whose elements were smaller than $\mathbf { c } _ { \mathrm { m e d } }$ and the second set whose elements were greater than or equal to $\mathbf { c } _ { \mathrm { m e d } }$ . Within labeled positive data, 90% data were selected from the first set and 10% data from the second set. In addition, we provide the results when using other selection biases for the image and tabular datasets in Section F.4.

## E.2 DATASETS WITH HUMAN-DERIVED CONFIDENCE

Cifar10-H and Fmnist-H were real-world image datasets where each instance was labeled by approximately 50 and 67 real-world human annotators, respectively (Peterson et al., 2019; Ishida et al., 2023). In these datasets, confidence information for each instance can be obtained by averaging the corresponding annotations (Ishida et al., 2023). For Cifar10-H, we treated the land-related classes (1, 3, 4, 5, 7, and 9) as positive and the remaining classes (e.g., water- or sky-related classes) as negative following the previous study (Ishida et al., 2023). To create the sampling bias, 90% of the labeled positive data were sampled from classes 1, 3, and 4. For Fmnist-H, we treated the casual classes (0, 1, 2, 6, and 7) as positive and the non-casual classes (3, 4, 5, 8, and 9) as negative. To create the sampling bias, 90% of the labeled positive data were sampled from classes 0, 1, and 2. Since the amount of annotated data was limited, we set the number of initially sampled unlabeled training to 3, 000 for both datasets. The other conditions were the same as those used for Cifar10 and Fmnist.

## E.3 MODEL AND TRAINING DETAILS

For Mnist, Fmnist, Fmnist-H, Diabetes, and Blood, a three-layered feed-forward neural network with ReLU activation was used for the classifier (score function). The number of hidden nodes was 124. For Svhn, Cifar10, and Cifar10-H, a convolutional neural network, which consisted of two convolutional blocks followed by a three-layered feed-forward neural network, was used for the classifier (score function). The first (second) convolutional block comprised a 6 (16) filter $5 \times 5$ convolution, the ReLU activation, and a $2 \times 2$ max-pooling layer. For all comparison methods including the classifier to estimate positive-confidence, the same neural network architecture was used.

For the proposed method, w/oConf, nnPU, PUAUC, Pconf, and NPU, the sigmoid function was used as a surrogate, following the previous studies (Kiryo et al., 2017; Kumagai et al., 2024; 2025a). For PUSB, we used the logarithmic loss function to deal with the selection bias following the original paper (Kato et al., 2019). For NTC and the classifier to estimate $p ( y = 1 | \mathbf { x } )$ , we used the logistic regression with the neural network as probabilistic classifiers. For nnPU, PUAUC, PUSB, and NPU, whose original objectives involve expectations over the marginal density, we exactly reexpressed these expectations in terms of the labeled positive and unlabeled densities under the one-sample sampling scheme (Bekker & Davis, 2020) for fair comparisons.

For all methods, the empirical risk (loss) with validation PU data was used for early-stopping to mitigate overfitting. For NPU, the weighting parameter was also chosen on the basis of the validation empirical risk from $\{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ . Confidence score $\tilde { r } ( \mathbf { x } )$ used in NPU was set to one since the observed positive labels are noise-free in our setting. For the proposed method and w/oConf, we rounded up $\hat { u } _ { n } ^ { \mathrm { p } } = \hat { u } ( \mathbf { x } _ { n } ^ { \mathrm { p } } )$ in Eq. (17) less than 0.01 to 0.01 to stabilize the training process following the previous study (Ishida et al., 2018). Similarly, for Pconf and NPU, we rounded up $r _ { n } ^ { \mathrm { p } } = p ( y = 1 | \mathbf { x } _ { n } ^ { \mathrm { p } } )$ less than 0.01 to 0.01 for their objectives. For all methods, we used the Adam optimizer (Kingma & Ba, 2014). We set the learning rate to $1 0 ^ { - 4 }$ . Total mini-batch size $M = U + { \dot { P } }$ was set to 1, 024 and PU mini-batch sizes P and $\breve { U }$ were set so as to maintain the label ratio α in the given all data. For the classifier for estimating $p ( y = 1 | \mathbf { x } )$ , we set the training epoch to 30. For other methods, the maximum number of epochs was 200 and the early-stopping was used. All methods were implemented using PyTorch (Paszke et al., 2017), and all experiments were conducted on a Linux server with an Intel Xeon CPU and A100 GPU.

![](images/a31d1607312028fb932166392622cbe0502d9eea4b40b628ccdc4d04843dafce.jpg)  
(a) Mnist

![](images/7571390dca38ee534c01fa44f1788290b27a5290ecd9b2540f0bc2700fdcf6db.jpg)  
(b) Fmnist

![](images/dec8be97ffb7d39860d3d2aaca4d945bc3d9e74e9a772d2327e33a8808062141.jpg)  
(c) Svhn

![](images/7ee7948fcf5d323d8e58a7105deefef7d20ce1edd5f8c7898251218c249e162e.jpg)  
(d) Cifar10

![](images/70cea23768c8a20ae4bef24cb52d37e447b5282da9b4d5f1fd6807c11d298736.jpg)  
(e) Diabetes

![](images/4854e076a8acd23d7ed92cbac5c35a3f253480b3d24f118c2657589783cd689c.jpg)  
(f) Blood  
Figure 3: Average test AUCs with their standard errors when changing positive class-prior π within {0.05, 0.1, 0.15, 0.2}.

## F ADDITIONAL EXPERIMENTAL RESULTS

## F.1 RESULTS WITH DIFFERENT POSITIVE CLASS-PRIORS π

Figure 3 shows the average test AUCs with their standard errors when changing the value of positive class-prior π. The proposed method tended to perform well with each π value. Most methods tended to improve the performance as the value of π increased. This is because the number of labeled positive data increased when π increased due to the relationship $\alpha = p ( o = 1 ) = c \pi$ with fixed $c = p ( o = 1 | y = 1 )$

Figure 4 shows the average test AUCs with their standard errors when changing the value of positive class-prior π while maintaining the labeled positive data size. To maintain the size, we adjusted the value of $c = p ( o = 1 | y = 1 )$ such that $p ( o \ : = \ : 1 ) \ : = \ : c \pi \ : = \ : 0 . 0 2$ for each π. As a result, the labeled positive data and unlabeled data sizes were 100 and 4, 900, respectively. The proposed method worked well in most cases and simply increasing π did not lead to performance improving.

## F.2 RESULTS WITH DIFFERENT POSITIVE LABELING RATES $c = p ( o = 1 | y = 1 )$

Tables 3 and 4 show the average test AUCs over different positive class-prior π when changing the value of $c = p ( o = 1 | y = 1 )$ . The proposed method performed well for each c value. As c increases, the number of labeled positive data increases, and thus the performance of the proposed method tends to improve.

![](images/4bda278869bb05b38a66246a28e2a053d50ecc9a18bd71bce90b09e3db42ba89.jpg)  
(a) Mnist

![](images/2a1e8aae1689f1adff45a5fc9f2088c25a28a644a4c215140151b8c7879ceead.jpg)  
(b) Fmnist

![](images/1a071dddb996cad254acc71c21aaf430ec2817db172fe628dd3c77e5b2766f48.jpg)  
(c) Svhn

![](images/1e6a29fbbe5b2cf11b302c5f937d6590a8945e7221088af54dd7dac9e13ab05e.jpg)  
(d) Cifar10

![](images/8c510981ebb313e611da00b0c5c3dbfe0983b44619c9509bf5ac6afb905ad1ec.jpg)  
(e) Diabetes

![](images/de76605b242dc7c85ab11f694114924d7b79d1d4dc2d133fb47b73bc4354898e.jpg)  
(f) Blood  
Figure 4: Average test AUCs with their standard errors when changing positive class-prior π within {0.05, 0.1, 0.15, 0.2} with the fixed number of labeled positive data.

Table 3: Average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2} with $c = 0 . 0 5$
<table><tr><td>Data</td><td>Ours</td><td>w/oConf</td><td>NTC</td><td>nnPU</td><td>PUAUC</td><td>PUSB</td><td>PG</td><td>Pconf</td><td>NPU</td></tr><tr><td>Mnist</td><td>0.8643</td><td>0.8465</td><td>0.8136</td><td>0.8009</td><td>0.8538</td><td>0.8525</td><td>0.8136</td><td>0.7697</td><td>0.8095</td></tr><tr><td>Fmnist</td><td>0.9220</td><td>0.9141</td><td>0.8827</td><td>0.9205</td><td>0.9164</td><td>0.9305</td><td>0.8827</td><td>0.6691</td><td>0.9318</td></tr><tr><td>Svhn</td><td>0.5896</td><td>0.5826</td><td>0.5090</td><td>0.4885</td><td>0.5911</td><td>0.5465</td><td>0.5090</td><td>0.4906</td><td>0.4898</td></tr><tr><td>Cifar10</td><td>0.8235</td><td>0.7974</td><td>0.7312</td><td>0.4751</td><td>0.8151</td><td>0.7979</td><td>0.7312</td><td>0.4976</td><td>0.4838</td></tr><tr><td>Diabetes</td><td>0.7109</td><td>0.6821</td><td>0.6949</td><td>0.5660</td><td>0.6905</td><td>0.7065</td><td>0.6949</td><td>0.4013</td><td>0.4744</td></tr><tr><td>Blood</td><td>0.5311</td><td>0.5170</td><td>0.5144</td><td>0.4905</td><td>0.5196</td><td>0.5269</td><td>0.5144</td><td>0.5041</td><td>0.4780</td></tr><tr><td># Best</td><td>6</td><td>1</td><td>0</td><td>0</td><td>2</td><td>3</td><td>0</td><td>0</td><td>1</td></tr></table>

Table 4: Average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2} with $c = 0 . 1 5$
<table><tr><td>Data</td><td>Ours</td><td>w/oConf</td><td>NTC</td><td>nnPU</td><td>PUAUC</td><td>PUSB</td><td>PG</td><td>Pconf</td><td>NPU</td></tr><tr><td>Mnist</td><td>0.9131</td><td>0.8931</td><td>0.8712</td><td>0.8216</td><td>0.8978</td><td>0.8972</td><td>0.8712</td><td>0.8456</td><td>0.8323</td></tr><tr><td>Fmnist</td><td>0.9589</td><td>0.9475</td><td>0.9333</td><td>0.9317</td><td>0.9512</td><td>0.9544</td><td>0.9333</td><td>0.8156</td><td>0.9529</td></tr><tr><td>Svhn</td><td>0.7095</td><td>0.7185</td><td>0.5524</td><td>0.4882</td><td>0.7167</td><td>0.6493</td><td>0.5524</td><td>0.4891</td><td>0.4884</td></tr><tr><td>Cifar10</td><td>0.8804</td><td>0.8548</td><td>0.8350</td><td>0.4949</td><td>0.8882</td><td>0.8751</td><td>0.8350</td><td>0.4867</td><td>0.4902</td></tr><tr><td>Diabetes</td><td>0.7478</td><td>0.7000</td><td>0.7483</td><td>0.5032</td><td>0.7321</td><td>0.7439</td><td>0.7483</td><td>0.3956</td><td>0.4628</td></tr><tr><td>Blood</td><td>0.5609</td><td>0.5314</td><td>0.5489</td><td>0.4829</td><td>0.5431</td><td>0.5495</td><td>0.5489</td><td>0.4947</td><td>0.4826</td></tr><tr><td># Best</td><td>5</td><td>1</td><td>1</td><td>0</td><td>2</td><td>1</td><td>1</td><td>0</td><td>1</td></tr></table>

## F.3 VISUALIZATION

The proposed method assigns weight $p ( y = 1 | \mathbf { x } ) / p ( o = 1 | \mathbf { x } )$ to the labeled positive instance x in the AUC risk of Eq. (16) for dealing with the bias of given positive data. In this section, we investigated the distribution of the estimated weights for labeled positive data with the Mnist dataset. Figure 5 shows the result. As can be seen, the values of the weights ranged from around 0.05 to about 84.0, with the majority distributed between 0.0 and 10.0. By leveraging these estimated weights, the proposed method performed superiorly.

![](images/062c6960041f1b463eeeee96c9dba2b3141c58260f76873769aac7f5394efd3f.jpg)  
Figure 5: Weight distribution of the proposed method with the Mnist dataset when $\pi = 0 . 2$

Table 5: The results using confidence-dependent selection bias: average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2}. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Ours</td><td>w/oConf</td><td>NTC</td><td>nnPU</td><td>PUAUC</td><td>PUSB</td><td>PG</td><td>Pconf</td><td>NPU</td></tr><tr><td>Mnist</td><td>0.9142</td><td>0.9063</td><td>0.8941</td><td>0.8876</td><td>0.9132</td><td>0.9162</td><td>0.8941</td><td>0.7540</td><td>0.8901</td></tr><tr><td>Fmnist</td><td>0.9748</td><td>0.9728</td><td>0.9488</td><td>0.9237</td><td>0.9738</td><td>0.9709</td><td>0.9488</td><td>0.4847</td><td>0.9494</td></tr><tr><td>Svhn</td><td>0.6546</td><td>0.6594</td><td>0.5290</td><td>0.4890</td><td>0.6650</td><td>0.5831</td><td>0.5290</td><td>0.4899</td><td>0.4894</td></tr><tr><td>Cifar10</td><td>0.8614</td><td>0.8499</td><td>0.8103</td><td>0.5196</td><td>0.8595</td><td>0.8444</td><td>0.8103</td><td>0.5608</td><td>0.5788</td></tr><tr><td>Diabetes</td><td>0.7684</td><td>0.7395</td><td>0.7682</td><td>0.5923</td><td>0.7643</td><td>0.7699</td><td>0.7682</td><td>0.3808</td><td>0.5249</td></tr><tr><td>Blood</td><td>0.6129</td><td>0.5846</td><td>0.5927</td><td>0.5011</td><td>0.6076</td><td>0.6085</td><td>0.5927</td><td>0.4769</td><td>0.4762</td></tr></table>

Table 6: Training time [sec] of each method on the Mnist dataset.
<table><tr><td>Ours</td><td>w/oConf</td><td>NTC</td><td>nnPU</td><td>PUAUC</td><td>PUSB</td><td>PG</td><td>Pconf</td><td>NPU</td></tr><tr><td>84.5424</td><td>84.6170</td><td>84.8195</td><td>84.5650</td><td>83.3238</td><td>85.2143</td><td>0.3280</td><td>80.4764</td><td>84.9054</td></tr></table>

## F.4 RESULTS WITH CONFIDENCE-DEPENDENT SELECTION BIAS

In the main paper, we considered class-based and distance-based selection biases for the image and tabular datasets, respectively. In this section, we evaluated the proposed method using a confidencedependent selection bias, which often arises in real-world applications. Specifically, we fixed the fraction of labeled positive instances at $c = 0 . 1$ , so that $N ^ { \mathrm { p } } = V c \pi$ , and sampled these instances from the positive instances in an initial data pool of size V , using sampling weights proportional to $\sigma ( 4 r ( \mathbf { x } ) - 2 )$ , where $r ( \mathbf { x } ) = p ( y = 1 | \mathbf { x } )$ . We used the probabilistic classifier for estimating r(x). In this setting, positive instances with higher confidence were more likely to be selected as labeled data. Table 5 shows the results. The proposed method performed well under this confidence-dependent selection bias.

## F.5 COMPUTATION COSTS

We investigated the training time of the proposed method on the Mnist dataset. We used a Linux server with a 2.20GHz CPU. Table 6 shows the results. Although the proposed method (Ours), w/oConf, and PG require the training of $p ( o = 1 | \mathbf { x } )$ , we omitted its training time here because we wanted to purely investigate the computation cost of our proposed loss function in Eq. (17). There were no significant differences in training time among all methods except for PG. The training time of PG is short since the classifier of PG can be obtained analytically from trained $p ( o = 1 | \mathbf { x } )$ However, it had lower test AUCs than the proposed method as described in Table 1. The full training times, including the training time of $p ( o = 1 | \mathbf { x } )$ , for the proposed method (Ours), w/oConf, and PG were 169.3620, 169.4360, and 85.1475, respectively. Although the proposed method had a longer training time than other methods due to the additional training of $p ( o = 1 | \mathbf { x } )$ , it had superior test AUCs.

Table 7: Comparison with LBE: average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2}. We set $c = p ( o = 1 | y = 1 ) = 0 . 1$ . Values in bold are not statistically different at the 5% level from the best performing method in each column according to a paired t-test.
<table><tr><td>Method</td><td>Mnist</td><td>Fmnist</td><td>Svhn</td><td>Cifar10</td><td>Diabetes</td><td>Blood</td></tr><tr><td>Ours</td><td>0.8979</td><td>0.9495</td><td> $\overline { { { \bf 0 . 6 6 7 3 } } }$ </td><td>0.8643</td><td>0.7342</td><td>0.5509</td></tr><tr><td>LBE</td><td>0.8631</td><td>0.9245</td><td>0.5668</td><td>0.8271</td><td>0.7292</td><td>0.5399</td></tr></table>

![](images/c8d1cb6f71637f2ef153901e5ab358c857582693f13d8aab2ec99366bdc29b50.jpg)  
(a) Mnist

![](images/ad7eaedc48674bf9f4e9a3c63c9351062356981797f4753ca5c49e5298e34a61.jpg)

![](images/e3e24fd290b6d80f62ef23170f06cd333c6212912ddec04c291fff8c877cc307.jpg)  
(c) Svhn

![](images/3ae2def0d44e12ee7bd015021b2d0e4d8d18654d4cc0a9585cb0becc6fc025ee.jpg)  
(d) Cifar10

(b) Fmnist  
![](images/2915dd2fb21eaf9ab2b9953653b2e6cc86bfab5ac20dc1a8dd413e99c0ba7cb6.jpg)  
(e) Diabetes

![](images/c245ca79c1c29234fb8b07a8f374dfcb7d167e7ef01d0ed1614f8aac96af5cfd.jpg)  
(f) Blood  
Figure 6: The average test AUCs and their standard errors over different positive class-priors for different parameters k of the transformation $h ( r ) = r ^ { k }$

## F.6 COMPARISON WITH AN ADDITIONAL SAR-SPECIFIC METHOD

To further compare the proposed method with methods specifically designed for the SAR setting, we additionally evaluated LBE (LBE-MLP) (Gong et al., 2021), a representative PU learning method for both the SAR setting and the one-sample scenario. This method estimates both $p ( y = 1 | \mathbf { x } )$ and $p ( o = 1 | \mathbf { x } , y = 1 )$ from biased PU data using the EM algorithm. Table 7 shows the results. The proposed method often outperformed LBE. This would be because LBE is not designed for AUC maximization and has difficulty identifying $p ( o = 1 | \mathbf { x } , y = 1 )$ from biased PU data. In contrast, our method is designed to maximize AUC and to avoid the identifiability issue by using positiveconfidence.

## F.7 RESULTS UNDER STRICTLY INCREASING TRANSFORMATIONS OF POSITIVE-CONFIDENCE

Figures 6 and 7 show the average test AUCs with their standard errors when changing k of the transformations $h ( r ) = r ^ { k }$ and $h ( r ) { \overset { \vartriangle } { = } } r ^ { k } / ( r ^ { k } + ( 1 - r ) ^ { k } )$ , respectively. For both transformation forms, the performance of the proposed method remained relatively stable across different k values and showed no large degradation as k moved away from 1. In contrast, the performances of Pconf that also uses positive-confidence substantially degraded for some values of $k ,$ reflecting its sensitivity to transformations of the confidence values.

## F.8 RESULTS WITH GAUSSIAN NOISE ON POSITIVE-CONFIDENCE

As shown in Section 4.3, the proposed method can recover the Bayes-optimal AUC ranking even when the positive-confidence is distorted by a strictly increasing transformation. We further evaluated its robustness when random noise was added to the positive-confidence. For noisy positive-confidence, we added the zero-mean Gaussian noise with standard deviations chosen from {0.01, 0.05, 0.1, 0.15} following the previous study (Ishida et al., 2018). As the standard deviation increases, the noise on positive-confidence becomes bigger. When the modified positive-confidence was less than 0 or greater than 1, we clipped the value to 0 and 1, respectively. Table 8 shows the results. As expected, the performance of the proposed method tended to decrease as the noise increased, because our objective function in Eq. (17) explicitly depends on confidence values. However, the degradation was relatively small on several datasets. Even with a standard deviation of 0.15, the proposed method often outperformed w/oConf, which does not use positive-confidence. These results suggest that using positive-confidence can remain beneficial even when the confidence contains random noise.

![](images/a3b59147f1c3a2fda598530b37513f5288a738b7eec06241590bf1efd9419931.jpg)  
(a) Mnist

![](images/4acba2ca9ad69925e1554afdbbc72da7205e18a5c6b5a3900e56edc16d80b673.jpg)  
(b) Fmnist

![](images/23efd5b4ca920cb24db43cc26be881efce9fb53722a9a513a7e26bdde337339f.jpg)  
(c) Svhn

![](images/49cdcb34fa88bc49d908e211d54898ce26380518d5601d375e98feedb9b87d08.jpg)  
(d) Cifar10

![](images/8e2b2fc7298bbf80107377aeb1746650c86f6a9b7ecd5ea7074e143071acdc72.jpg)  
(e) Diabetes

![](images/3421efef300b0bd18b565c68d40f74bce0a2a90f715fad2d2d8ec4b98545938a.jpg)  
(f) Blood  
Figure 7: The average test AUCs and their standard errors over different positive class-priors for different parameters k of the transformation $h ( r ) = r ^ { k } / ( r ^ { k } + ( 1 - r ) ^ { k } )$

Table 8: The results with noisy positive-confidence: average test AUCs over different positive classprior π within {0.05, 0.1, 0.15, 0.2}. Std represents the standard deviation of Gaussian noise. Std = 0.0 means that there is no noise.
<table><tr><td>Std</td><td>0.0</td><td>0.01</td><td>0.05</td><td>0.1</td><td>0.15</td><td>w/oConf</td></tr><tr><td>Mnist</td><td>0.8979</td><td>0.8966</td><td>0.8932</td><td>0.8905</td><td>0.8883</td><td>0.8769</td></tr><tr><td>Fmnist</td><td>0.9495</td><td>0.9495</td><td>0.9501</td><td>0.9496</td><td>0.9489</td><td>0.9371</td></tr><tr><td>Svhn</td><td>0.6673</td><td>0.6662</td><td>0.6601</td><td>0.6367</td><td>0.6161</td><td>0.6666</td></tr><tr><td>Cifar10</td><td>0.8643</td><td>0.8670</td><td>0.8623</td><td>0.8579</td><td>0.8552</td><td>0.8371</td></tr><tr><td>Diabetes</td><td>0.7342</td><td>0.7333</td><td>0.7277</td><td>0.7189</td><td>0.7013</td><td>0.6926</td></tr><tr><td>Blood</td><td>0.5509</td><td>0.5457</td><td>0.5435</td><td>0.5376</td><td>0.5321</td><td>0.5287</td></tr></table>

## F.9 ROBUSTNESS ANALYSIS FOR THE CLIPPING

As mentioned in Section 5.2, we used the clipping for estimated labeling probability uˆ(x) to stabilize the training process. Specifically, when the estimated labeling probability was smaller than a clipping value, we replaced it with the clipping value. Here, we empirically investigated the robustness of the proposed method to the clipping values. Table 9 shows the average test AUCs by changing clipping value τ within {0.001, 0.005, 0.01, 0.05, 0.1}. These results show that the proposed method is relatively robust against the choice of the clipping values.

Table 9: Average test AUCs when varying the clipping value τ for the estimated labeling probability $p ( o = 1 | \mathbf { x } )$
<table><tr><td>Data</td><td>τ = 0.001</td><td>0.005</td><td>0.01</td><td>0.05</td><td>0.1</td></tr><tr><td>Mnist</td><td>0.8945</td><td>0.8962</td><td>0.8979</td><td>0.8997</td><td>0.8995</td></tr><tr><td>Fmnist</td><td>0.9467</td><td>0.9480</td><td>0.9495</td><td>0.9503</td><td>0.9494</td></tr><tr><td>Svhn</td><td>0.6656</td><td>0.6672</td><td>0.6673</td><td>0.6525</td><td>0.6533</td></tr><tr><td>Cifar10</td><td>0.8605</td><td>0.8624</td><td>0.8643</td><td>0.8642</td><td>0.8631</td></tr><tr><td>Diabetes</td><td>0.7325</td><td>0.7322</td><td>0.7342</td><td>0.7392</td><td>0.7400</td></tr><tr><td>Blood</td><td>0.5480</td><td>0.5493</td><td>0.5509</td><td>0.5564</td><td>0.5564</td></tr></table>

Table 10: The results with noisy estimated labeling probabilities: average test AUCs over different positive class-prior π within {0.05, 0.1, 0.15, 0.2}. Std represents the standard deviation of Gaussian noise. Std = 0.0 means that there is no noise.
<table><tr><td>Std</td><td>0.0</td><td>0.01</td><td>0.05</td><td>0.1</td><td>0.15</td></tr><tr><td>Mnist</td><td>0.8979</td><td>0.8961</td><td>0.8933</td><td>0.8896</td><td>0.8867</td></tr><tr><td>Fmnist</td><td>0.9495</td><td>0.9501</td><td>0.9468</td><td>0.9403</td><td>0.9364</td></tr><tr><td>Svhn</td><td>0.6673</td><td>0.6595</td><td>0.6354</td><td>0.6214</td><td>0.6166</td></tr><tr><td>Cifar10</td><td>0.8643</td><td>0.8689</td><td>0.8621</td><td>0.8567</td><td>0.8547</td></tr><tr><td>Diabetes</td><td>0.7342</td><td>0.7351</td><td>0.7325</td><td>0.7329</td><td>0.7328</td></tr><tr><td>Blood</td><td>0.5509</td><td>0.5453</td><td>0.5421</td><td>0.5412</td><td>0.5401</td></tr></table>

## F.10 RESULTS WITH GAUSSIAN NOISE ON ESTIMATED LABELING PROBABILITY

The proposed method uses the labeling probability estimated from the given PU data, uˆ(x), to construct the rewritten AUC estimator. Here, we evaluated the proposed method when random noise was added to estimated labeling probability uˆ(x). Specifically, we added the zero-mean Gaussian noise with standard deviations chosen from {0.01, 0.05, 0.1, 0.15} to uˆ(x). When the modified labeling probability was less than 0.01 or greater than 1, we clipped the value to 0.01 and 1, respectively. Table 10 shows the results. As expected, the performance of the proposed method tended to decrease as the noise increased. However, the degradation was relatively small on several datasets. Even with a standard deviation of 0.15, the proposed method often worked well. These results empirically demonstrate that the proposed method is reasonably robust to estimation errors in the labeling probability.

## F.11 FULL RESULTS

Table 11 shows the average test AUCs with their standard deviations over different positive classprior π in each dataset. The proposed method performed the best or comparably to it in all cases.

Table 11: Average test AUCs with their standard deviations over different positive class-prior π within {0.05, 0.1, 0.15, 0.2}. We set $c = p ( o = 1 | y = 1 ) = 0 . 1$ . Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td colspan="2">Ours</td><td colspan="2">w/oConf</td><td colspan="2">NTC</td><td colspan="2">nnPU PUAUC</td></tr><tr><td>Mnist</td><td colspan="2">0.8979(0.0158)</td><td colspan="2">0.8769(0.0226)</td><td colspan="2">0.8575(0.0192) 0.8196(0.0531)</td><td colspan="2">0.8828(0.0203)</td></tr><tr><td>Fmnist</td><td colspan="2">0.9495(0.0236)</td><td colspan="2">0.9371(0.0267)</td><td colspan="2">0.9165(0.0219) 0.9296(0.0180)</td><td colspan="2">0.9409(0.0231)</td></tr><tr><td>Svhn</td><td colspan="2">0.6673(0.0688)</td><td colspan="2">0.6666(0.0579)</td><td colspan="2">0.5328(0.0305) 0.4885(0.0069)</td><td colspan="2">0.6704(0.0539)</td></tr><tr><td>Cifar10</td><td colspan="2">0.8643(0.0373)</td><td colspan="2">0.8371(0.0420)</td><td colspan="2">0.8013(0.0491) 0.4819(0.1191)</td><td colspan="2">0.8620(0.0355)</td></tr><tr><td>Diabetes</td><td colspan="2">0.7342(0.0306)</td><td colspan="2">0.6926(0.0268)</td><td colspan="2">0.7357(0.0183) 0.5234(0.0775)</td><td colspan="2">0.7171(0.0258)</td></tr><tr><td>Blood</td><td colspan="2">0.5509(0.0317)</td><td colspan="2">0.5287(0.0253)</td><td colspan="2">0.5409(0.0336) 0.4847(0.0182)</td><td colspan="2">0.5395(0.0273)</td></tr><tr><td rowspan="7"></td><td colspan="2">Data Mnist</td><td colspan="2">PUSB</td><td colspan="2">PG Pconf</td><td colspan="2">NPU</td></tr><tr><td colspan="2">Fmnist</td><td colspan="2">0.8841(0.0197) 0.8575(0.0192) 0.9165(0.0219)</td><td colspan="2">0.8193(0.0807)</td><td colspan="2">0.8285(0.0598)</td></tr><tr><td colspan="2"></td><td colspan="2">0.9474(0.0195)</td><td colspan="2">0.7737(0.1616)</td><td colspan="2">0.9470(0.0276)</td></tr><tr><td colspan="2">Svhn Cifar10</td><td colspan="2">0.5965(0.0763) 0.8442(0.0446)</td><td colspan="2">0.5328(0.0305) 0.4900(0.0070) 0.8013(0.0491) 0.4767(0.1310)</td><td colspan="2">0.4893(0.0068) 0.4883(0.1395)</td></tr><tr><td colspan="2">Diabetes</td><td colspan="2">0.7230(0.0248)</td><td colspan="2">0.7357(0.0183) 0.3974(0.0368)</td><td colspan="2">0.4626(0.0605)</td></tr><tr><td colspan="2">Blood</td><td colspan="2">0.5450(0.0323)</td><td colspan="2">0.5409(0.0336) 0.4962(0.0204)</td><td colspan="2">0.4828(0.0200)</td></tr><tr><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr></table>