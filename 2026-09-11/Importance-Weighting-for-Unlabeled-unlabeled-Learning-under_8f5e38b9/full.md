# Importance Weighting for Unlabeled-unlabeled Learning under Distribution Shift

Atsutoshi Kumagai, Tomoharu Iwata, Hiroshi Takahashi, Taishi Nishiyama, Kazuki Adachi, Yasuhiro Fujiwara NTT, Inc. atsutoshi.kumagai@ntt.com

## Abstract

Unlabeled-unlabeled (UU) learning allows us to learn a binary classifier from two sets of unlabeled data with different class-priors. It is a general framework because it includes a wide variety of supervised learning such as positive-unlabeled (PU) learning, noisy label learning, and similarity-based learning. Existing UU learning assumes that the test and training distributions have the same class-conditional densities. However, this assumption rarely holds in practice due to distribution shifts. This paper proposes a distribution shift adaptation method for UU learning that uses UU data in the training distribution and a few UU data in the test distribution. The proposed method is based on the importance weighting, which minimizes the test risk by using training data with estimated importance weights. Although existing importance weighting methods cannot handle UU data, we show that it can be done in a principled manner. Thanks to the generality of UU learning, our method can handle various learning problems such as PU and noisy label learning under distribution shift within a single framework while existing methods are usually tailored to a specific problem. Moreover, it does not require any assumption of the shift types such as covariate shift. We experimentally demonstrate the effectiveness of the proposed method with real-world datasets.

## 1 Introduction

In supervised learning, a large amount of labeled data is required for learning accurate classifiers. However, in practice, labeled data are often very costly or even infeasible to collect. This fact has motivated us to investigate learning algorithms that work well from data with weak supervision (e.g., noisy labels), which are often easier to collect than clean and complete labeled data [50]. In this paper, we consider unlabeled-unlabeled (UU) learning where the aim is to learn a binary classifier from two unlabeled datasets that share the class-conditional densities but have different class-priors (i.e., the proportion of positives in each unlabeled data) [33, 35, 32]. In this setting, class-priors can be regarded as weak supervision. We focus on this challenging setting because of its high generality: it can represent a wide variety of learning problems, such as positive-negative (PN) learning [54], positive unlabeled (PU) learning [6, 20, 2], noisy label learning [58, 4], pairwise-comparison learning [10], and similarity-based learning [1, 45], by specifying values of the class-priors [5].

Although several UU learning methods have recently been proposed [33, 35, 32, 56], they assume that the test and training distributions have the same class-conditional densities. However, in practice, this assumption is often violated by various factors such as differences in data collection periods [27] and environments [24]. For example, in cyber security, (noisy) labeled malicious and normal data can be regarded as UU data, and the data distribution can vary over time by the emergence of new types of attacks [26]. When such a distribution shift occurs, the performances of existing method drastically deteriorate.

![](images/6757ca04de5cde068ea83c2451a2305f7ffb2f395382987d2476f893fcdead6d.jpg)  
Figure 1: Our problem setting. We are given UU data in the training distribution and a few UU data in the test distribution. UU data in each distribution shares the class-conditional densities but have different class-priors (i.e., $\theta _ { \mathrm { t r } } ^ { \mathrm { a } } \neq \theta _ { \mathrm { t r } } ^ { \mathrm { b } }$ and $\theta _ { \mathrm { t e } } ^ { \mathrm { a } } \neq \theta _ { \mathrm { t e } } ^ { \mathrm { b } } )$ . The class-conditional densities can vary between the training and test distributions. The aim is to learn a binary classifier that can accurately classify test data drawn from the test distribution with true class-prior $\pi _ { \mathrm { t e } }$ by using these UU datasets. More details of our problem setting are provided in subsection 4.1.

In this paper, we propose a distribution shift adaptation method for UU learning that uses UU data in the training distribution and a few UU data in the test distribution<sup>1</sup>. A few UU data are relatively easy to collect even in the test distribution. Figure 1 shows the overview of our problem setting. Our method is based on the importance weighting, which is a well-established and commonly used framework for distribution shift adaptation on ordinary supervised learning [51, 8, 34]. This framework estimates importance weights, i.e., the ratio between training and test densities, and learns classifiers by minimizing the importance-weighted empirical training risk that is approximately equivalent to the test risk. Although many distribution shift adaptation methods have been proposed for settings such as PN learning [7, 8] and PU learning [27], no existing methods can treat UU data. To the best of our knowledge, we are the first to show that importance weighting can be performed from UU data in a principled manner. By minimizing a weighted sum of the importance-weighted empirical risk with UU data in the training distribution and the empirical risk with UU data in the test distribution, the proposed method can train classifiers that fit on the test distribution.

Thanks to the generality of UU learning, the proposed method can handle various types of learning problems under distribution shift within a single framework, including PN, PU, noisy label, pairwisecomparison, and similarity-based learning. This contrasts with existing distribution shift adaptation methods, which are typically tailored to a specific problem [39, 48]. Note that there are no existing distribution shift adaptation methods that assume noisy labels in both distributions nor methods for similarity-based learning or paired-comparison learning. In addition, the proposed method can flexibly handle different types of supervision between the training and test phases, e.g., noisy labeled data in the training distribution and a few PN data in the test distribution. Furthermore, by using UU data in the test distribution, the proposed method does not require assumptions about shift types such as covariate shift [46] and concept shift [37], which are usually required in existing distribution shift adaptation methods that use unlabeled data in the test distribution [46]. Since shift types are generally difficult or impossible to identify from unlabeled data [27], this property is preferable in practice.

## 2 Related Work

Discriminative clustering or unsupervised classification methods attempt to learn a classifier from a set of unlabeled data [59, 21]. However, they are often suboptimal since they rely on a clustering assumption that one cluster corresponds to one class, which is often violated in practice [33, 35]. To overcome this issue, UU learning methods are based on the empirical risk minimization (ERM) that rewrites the risk using two unlabeled datasets with different class-priors [33, 35, 36]. It can learn a binary classifier without the clustering assumption [33]. Some methods can use m $( m \ge 2 )$ sets of unlabeled data with different class-priors to learn a binary classifier [32, 56]. However, all these methods assume that the test and training distributions have the same class-conditional densities, and thus they are inappropriate for our problem settings where the class-conditional densities can vary.

Distribution shift or domain adaptation methods attempt to mitigate the gap between the training and test distributions [39, 55, 53, 48]. Although many methods have been proposed, there are no methods for UU learning. For the adaptation, one representative approach is to learn invariant features between the training and test distributions [43, 38, 11, 23, 29]. Although this approach is promising in some applications, the invariant features often do not improve the performance since they do not explicitly minimize the test risk [63, 27]. Another representative approach is the importance weighting, which can adapt to the shift by explicitly minimizing the test risk, and it has solid theoretical properties such as consistency [34, 52]. Accordingly, we extend it to the UU setting.

As for data used for the adaptation, existing methods often assume unlabeled data in the test distribution [43, 11, 23, 46]. Since there is no supervision in the test distribution, they typically assume a specific shift type such as covariate shift [46]. However, the shift type is generally difficult or impossible to identify from unlabeled data [7, 27]. Some recent importance weighting methods can adapt to the shift without shift type assumptions by using some supervision in the test distribution, such as a few PN data [7, 8] and a few PU data [27]. The proposed method also does not require shift type assumptions because it uses a few UU data in the test distribution. By appropriately setting the class-priors, the proposed method can recover these methods [7, 27] as special cases. In general, since the proposed method must perform importance weighting solely from unlabeled data, it addresses a more challenging problem than these methods that can exploit labeled (positive) data. Although several methods assume noisy labeled data in the training distribution and unlabeled data in the test distribution [47, 62, 61, 14], they cannot treat noisy labeled data in both the training and test distributions, but the proposed method can.

## 3 Preliminary

We explain UU learning based on the ERM [33, 35]. Let $\mathbf { x } \in \mathcal { X }$ and $y \in \{ \pm 1 \}$ be the input and output random variables, where X is the input space, $y = + 1$ and −1 represent positive and negative classes, respectively. Let $p ( \mathbf { x } , y )$ be the joint density, $p ^ { \mathrm { p } } ( \mathbf { x } ) : = p ( \mathbf { x } | y = + 1 )$ ) and $p ^ { \mathrm { n } } ( \mathbf { x } ) : = p ( \mathbf { x } | y = - 1 )$ be the positive- and negative-conditional densities, $p ( \mathbf { x } ) = \pi p ^ { \mathrm { p } } ( \mathbf { x } ) + ( 1 - \pi ) p ^ { \mathrm { n } } ( \mathbf { x } )$ be the marginal density, and $\pi : = p ( y = + 1 )$ be the class-prior. $f : \mathcal { X }  \mathbb { I }$ R is a decision function and the predicted label is obtained by $t = \mathrm { s i g n } ( f ( \mathbf { x } ) )$ , where sign(·) is a sign function. Let $\ell : \mathbb { R } \times \{ \pm 1 \} \to \mathbb { R } _ { > 0 }$ be the loss function, such that value $\ell ( t , y )$ means the loss for ground truth label y by t. The binary classification aims to learn f that minimizes the expected test error, called the risk,

$$
R ( f ) : = \mathbb { E } _ { p ( \mathbf { x } , y ) } [ \ell ( f ( \mathbf { x } ) , y ) ] = \pi \mathbb { E } _ { p ^ { \mathrm { p } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , + 1 ) ] + ( 1 - \pi ) \mathbb { E } _ { p ^ { \mathrm { n } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , - 1 ) ] ,\tag{1}
$$

where $\mathbb { E } _ { p ( z ) }$ is an expectation over $p ( z )$ . In standard supervised learning, we are given labeled positive and negative data drawn from $p ( \mathbf { x } , y )$ . Thus, we can directly train the classifier by minimizing the empirical risk that is calculated by replacing the expectation in Eq. (1) with the sample average.

However, in UU learning, we are given two sets of unlabeled data $X ^ { \mathrm { a } }$ and $X ^ { \mathrm { b } }$ drawn from the following marginal distributions:

$$
\begin{array} { r l } & { X ^ { \mathrm { a } } = \{ \mathbf { x } _ { n } ^ { \mathrm { a } } \} _ { n = 1 } ^ { N ^ { \mathrm { a } } } \sim p ^ { \mathrm { a } } ( \mathbf { x } ) : = \theta ^ { \mathrm { a } } p ^ { \mathrm { p } } ( \mathbf { x } ) + ( 1 - \theta ^ { \mathrm { a } } ) p ^ { \mathrm { n } } ( \mathbf { x } ) , } \\ & { X ^ { \mathrm { b } } = \{ \mathbf { x } _ { n } ^ { \mathrm { b } } \} _ { n = 1 } ^ { N ^ { \mathrm { b } } } \sim p ^ { \mathrm { b } } ( \mathbf { x } ) : = \theta ^ { \mathrm { b } } p ^ { \mathrm { p } } ( \mathbf { x } ) + ( 1 - \theta ^ { \mathrm { b } } ) p ^ { \mathrm { n } } ( \mathbf { x } ) , } \end{array}\tag{2}
$$

where $\theta ^ { \mathrm { a } }$ and $\theta ^ { \mathrm { b } }$ are two class-priors such that $\theta ^ { \mathrm { a } } \neq \theta ^ { \mathrm { b } }$ . Marginal densities $p ( \mathbf { x } ) , p ^ { \mathrm { a } } ( \mathbf { x } )$ , and $p ^ { \mathrm { b } } ( \mathbf { x } )$ share the class-conditional densities $p ^ { \mathrm { p } } ( \mathbf { x } )$ and $p ^ { \mathrm { n } } ( \mathbf { \dot { x } } )$ . To train the classifier from the two unlabeled datasets, UU learning rewrites the risk in Eq. (1) with two marginal densities $p ^ { \mathrm { a } } ( \mathbf { x } )$ and $p ^ { \mathrm { b } } ( \mathbf { x } )$ Specifically, by solving Eq. (2) for $p ^ { \mathrm { p } } ( \mathbf { x } )$ and $p ^ { \mathrm { n } } ( \mathbf { x } )$ , we can obtain the following equations:

$$
\pi p ^ { \mathrm { p } } ( \mathbf { x } ) = a p ^ { \mathrm { a } } ( \mathbf { x } ) - c p ^ { \mathrm { b } } ( \mathbf { x } ) , ( 1 - \pi ) p ^ { \mathrm { n } } ( \mathbf { x } ) = - b p ^ { \mathrm { a } } ( \mathbf { x } ) + d p ^ { \mathrm { b } } ( \mathbf { x } ) ,\tag{3}
$$

where $\begin{array} { r } { a : = \frac { ( 1 - \theta ^ { \mathrm { b } } ) \pi } { \theta ^ { \mathrm { a } } - \theta ^ { \mathrm { b } } } , b : = \frac { \theta ^ { \mathrm { b } } ( 1 - \pi ) } { \theta ^ { \mathrm { a } } - \theta ^ { \mathrm { b } } } , c : = \frac { ( 1 - \theta ^ { \mathrm { a } } ) \pi } { \theta ^ { \mathrm { a } } - \theta ^ { \mathrm { b } } } } \end{array}$ , and $\begin{array} { r } { d : = \frac { \theta ^ { \mathrm { a } } ( 1 - \pi ) } { \theta ^ { \mathrm { a } } - \theta ^ { \mathrm { b } } } } \end{array}$ . By using Eq. (3), we can obtain

$$
\pi \mathbb { E } _ { p ^ { \mathrm { p } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , + 1 ) ] = a \mathbb { E } _ { p ^ { \mathrm { a } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } , + 1 ) ] - c \mathbb { E } _ { p ^ { \mathrm { b } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } , + 1 ) ] ,
$$

$$
( 1 - \pi ) \mathbb { E } _ { p ^ { \mathrm { n } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , - 1 ) ] = - b \mathbb { E } _ { p ^ { \mathrm { a } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } , - 1 ) ] + d \mathbb { E } _ { p ^ { \mathrm { b } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } , - 1 ) ] .\tag{4}
$$

Plugging them into Eq. (1), we obtain

$$
R ( f ) = a \mathbb { E } _ { p ^ { \mathrm { s } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , + 1 ) ] - c \mathbb { E } _ { p ^ { \mathrm { b } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , + 1 ) ] - b \mathbb { E } _ { p ^ { \mathrm { s } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , - 1 ) ] + d \mathbb { E } _ { p ^ { \mathrm { b } } ( \mathbf { x } ) } [ \ell ( f ( \mathbf { x } ) , - 1 ) ] .\tag{5}
$$

Since Eq. (5) is represented with $p ^ { \mathrm { a } } ( \mathbf { x } )$ and $p ^ { \mathrm { b } } ( \mathbf { x } )$ , we can approximate it with $X ^ { \mathrm { a } }$ and $X ^ { \mathrm { b } }$ . We call this rewritten risk the UU risk. Although both equations in (4) are non-negative by definition, when highly expressive models such as neural networks are used for $f ,$ their empirical estimates can take negative values leading to serious overfitting [35]. To alleviate this issue, non-negative correction such as the absolute value correction is commonly used [35, 32]. Specifically, the empirical estimate of Eq. (5) with the absolute value correction is given by

$$
\hat { R } ( f ) = \left| \frac { a } { N ^ { \mathrm { a } } } \sum _ { n = 1 } ^ { N ^ { \mathrm { a } } } \ell ( f ( \mathbf { x } _ { n } ^ { \mathrm { a } } ) , + 1 ) - \frac { c } { N ^ { \mathrm { b } } } \sum _ { n = 1 } ^ { N ^ { \mathrm { b } } } \ell ( f ( \mathbf { x } _ { n } ^ { \mathrm { b } } ) , + 1 ) \right| + \left| \frac { d } { N ^ { \mathrm { b } } } \sum _ { n = 1 } ^ { N ^ { \mathrm { b } } } \ell ( f ( \mathbf { x } _ { n } ^ { \mathrm { b } } ) , - 1 ) - \frac { b } { N ^ { \mathrm { a } } } \sum _ { n = 1 } ^ { N ^ { \mathrm { a } } } \ell ( f ( \mathbf { x } _ { n } ^ { \mathrm { a } } ) , - 1 ) \right| ,\tag{6}
$$

where | · | is the absolute value function to penalize the risk for being negative. When there are many UU data, we can obtain an accurate classifier $f$ by minimizing Eq. (6) w.r.t. parameters of $f .$ . UU learning includes a wide variety of learning problems as special cases. For example, when $\dot { \theta } ^ { \mathrm { a } } = 1$ and $\theta ^ { \mathrm { { b } } } = 0 , X ^ { \mathrm { { a } } }$ and $X ^ { \mathrm { b } }$ are positive and negative data, respectively; Eq. (6) becomes the empirical risk on ordinary PN learning with $( a , b , c , d ) \bar { = } \left( \pi , 0 , 0 , 1 \bar { - } \pi \right) [ 5 4 ]$ . When $\theta ^ { \mathrm { a } } = 1$ and $\theta ^ { \mathrm { b } } = \dot { \pi } , X ^ { \mathrm { a } }$ and $X ^ { \mathrm { b } }$ are positive and unlabeled data, respectively; Eq. (6) becomes the non-negative empirical risk on PU learning with $( a , b , c , d ) = ( \pi , \bar { \pi , } 0 , 1 ) \bar { [ 1 3 ] }$ . When $1 > \theta ^ { \mathrm { a } } > \theta ^ { \mathrm { b } } > 0 , \breve { X } ^ { \mathrm { a } }$ and $\dot { X } ^ { \mathrm { b } }$ can be regarded as noisy positive and negative data, respectively; Eq. (6) becomes the non-negative empirical risk on noisy label learning.

## 4 Proposed Method

We first define our problem setting (subsection 4.1). Then, we derive the importance-weighted UU risk (subsection 4.2) and the estimation method for importance weights from UU data (subsection 4.3). After describing our classifier loss function with the importance-weighted UU risk (subsection 4.4), we explain our training procedure (subsection 4.5).

## 4.1 Problem Setting

Let $p _ { \mathrm { t r } } ( \mathbf { x } , y )$ be the joint density of the training distribution, $p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } ) : = p _ { \mathrm { t r } } ( \mathbf { x } | y = + 1 )$ and $p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } ) : =$ $p _ { \mathrm { t r } } ( \mathbf { x } | y = - 1 )$ be the positive and negative-conditional training densities, respectively, and $\pi _ { \mathrm { t r } } : =$ $p _ { \mathrm { t r } } ( y = + 1 )$ be the positive training class-prior. Similarly, let $p _ { \mathrm { t e } } ( \mathbf { x } , y )$ be the joint density of the test distribution, $p _ { \mathrm { t e } } ^ { \mathrm { p } } ( \mathbf { x } ) \bar { : } = p _ { \mathrm { t e } } ( \mathbf { x } | y = \bar { + } 1 )$ and $p _ { \mathrm { t e } } ^ { \mathrm { n } } ( \mathbf { x } ) : = p _ { \mathrm { t e } } ( \mathbf { x } | y = - 1 )$ be the positive and negativeconditional test densities, respectively, and $\pi _ { \mathrm { t e } } : = p _ { \mathrm { t e } } ( y = + 1 )$ be the positive test class-prior. We assume that the training and test distributions are related but different, $p _ { \mathrm { t r } } ( \mathbf { x } , y ) \neq p _ { \mathrm { t e } } ( \mathbf { x } , y )$ , which is the most general shift form. Note that we do not know the specific shift type such as covariate shift.

Suppose that we are given two sets of unlabeled data $X _ { \mathrm { t r } } ^ { \mathrm { a } }$ and $X _ { \mathrm { t r } } ^ { \mathrm { b } }$ drawn from different marginal densities $p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } )$ and $p _ { \mathrm { t r } } ^ { \mathrm { { \bar { b } } } } ( \mathbf { x } )$ , respectively, in the training phase:

$$
\begin{array} { r l } & { X _ { \mathrm { t r } } ^ { \mathrm { a } } = \bigl \{ \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { a } } \bigr \} _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { a } } } \sim p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } ) { : = } \theta _ { \mathrm { t r } } ^ { \mathrm { a } } p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } ) { + } ( 1 { - } \theta _ { \mathrm { t r } } ^ { \mathrm { a } } ) p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } ) , } \\ & { X _ { \mathrm { t r } } ^ { \mathrm { b } } { = } \bigl \{ \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { b } } \bigr \} _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { b } } } \sim p _ { \mathrm { t r } } ^ { \mathrm { b } } ( \mathbf { x } ) { : = } \theta _ { \mathrm { t r } } ^ { \mathrm { b } } p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } ) { + } ( 1 { - } \theta _ { \mathrm { t r } } ^ { \mathrm { b } } ) p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } ) , } \end{array}\tag{7}
$$

where $\theta _ { \mathrm { t r } } ^ { \mathrm { a } }$ and $\theta _ { \mathrm { t r } } ^ { \mathrm { b } }$ are two class-priors such that $\theta _ { \mathrm { t r } } ^ { \mathrm { a } } \neq \theta _ { \mathrm { t r } } ^ { \mathrm { b } }$ . Similarly, suppose that we are also given two sets of unlabeled data $X _ { \mathrm { t e } } ^ { \mathrm { a } }$ and $X _ { \mathrm { t e } } ^ { \mathrm { b } }$ drawn from different marginal densities $p _ { \mathrm { t e } } ^ { \mathrm { a } } ( \mathbf { x } )$ and $p _ { \mathrm { t e } } ^ { \mathrm { b } } ( \mathbf { x } )$ respectively, in the test phase:

$$
\begin{array} { r } { X _ { \mathrm { t e } } ^ { \mathrm { a } } = \big \{ \mathbf { x } _ { \mathrm { t e } , n } ^ { \mathrm { a } } \big \} _ { n = 1 } ^ { N _ { \mathrm { t e } } ^ { \mathrm { a } } } \sim p _ { \mathrm { t e } } ^ { \mathrm { a } } ( \mathbf { x } ) { : = } \theta _ { \mathrm { t e } } ^ { \mathrm { a } } p _ { \mathrm { t e } } ^ { \mathrm { p } } ( \mathbf { x } ) { + } ( 1 { - } \theta _ { \mathrm { t e } } ^ { \mathrm { a } } ) p _ { \mathrm { t e } } ^ { \mathrm { n } } ( \mathbf { x } ) , } \\ { X _ { \mathrm { t e } } ^ { \mathrm { b } } = \big \{ \mathbf { x } _ { \mathrm { t e } , n } ^ { \mathrm { b } } \big \} _ { n = 1 } ^ { N _ { \mathrm { t e } } ^ { \mathrm { b } } } \sim p _ { \mathrm { t e } } ^ { \mathrm { b } } ( \mathbf { x } ) { : = } \theta _ { \mathrm { t e } } ^ { \mathrm { b } } p _ { \mathrm { t e } } ^ { \mathrm { p } } ( \mathbf { x } ) { + } ( 1 { - } \theta _ { \mathrm { t e } } ^ { \mathrm { b } } ) p _ { \mathrm { t e } } ^ { \mathrm { n } } ( \mathbf { x } ) , } \end{array}\tag{8}
$$

where $\theta _ { \mathrm { t e } } ^ { \mathrm { a } }$ and $\theta _ { \mathrm { t e } } ^ { \mathrm { b } }$ are two class-priors such that $\theta _ { \mathrm { t e } } ^ { \mathrm { a } } \neq \theta _ { \mathrm { t e } } ^ { \mathrm { b } }$ . We assume that UU data in the test distribution are much smaller than those in the training distribution, i.e., $N _ { { \mathrm { t e } } } ^ { \mathrm { a } } + N _ { { \mathrm { t e } } } ^ { \mathrm { b } } \ll N _ { { \mathrm { t r } } } ^ { \mathrm { a } } + N _ { { \mathrm { t r } } } ^ { \mathrm { b } }$

We assume that class-priors $\pi _ { \mathrm { t e } } , \theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } , \theta _ { \mathrm { t e } } ^ { \mathrm { a } }$ , and $\theta _ { \mathrm { t e } } ^ { \mathrm { b } }$ are known as in previous UU learning studies [33, 35, 32]. They can be estimated in some cases [36, 31, 16]. Our goal is to learn a classifier f that accurately classifies test instance x drawn from $p _ { \mathrm { t e } } ( { \bf x } ) = \pi _ { \mathrm { t e } } p _ { \mathrm { t e } } ^ { \mathrm { p } } ( { \bf x } ) + ( 1 - \pi _ { \mathrm { t e } } ) p _ { \mathrm { t e } } ^ { \mathrm { n } } ( { \bf x } )$ by using unlabeled datasets $X _ { \mathrm { t r } } ^ { \mathrm { a } } \cup X _ { \mathrm { t r } } ^ { \mathrm { b } } \cup X _ { \mathrm { t e } } ^ { \mathrm { a } } \cup X _ { \mathrm { t e } } ^ { \mathrm { b } }$ . The proposed method can handle various learning problems, such as PN, PU, and noisy labeled learning, by specifying values of the class-priors as described in Section 3. It can be naturally applied even when types of supervision differ between the training and test phases, e.g., noisy labeled data in the training distribution and a few PN data in the test distribution.

## 4.2 Importance-weighted UU Risk

In this subsection, we derive the importance-weighted empirical risk using UU training data by rewriting the test risk. First, we consider the risk on the test distribution:

$$
R _ { \mathrm { t e } } ( f ) : = \mathbb { E } _ { p _ { \mathrm { t e } } ( \mathbf { x } , y ) } \left[ \ell ( f ( \mathbf { x } ) , y ) \right] ,\tag{9}
$$

which is the objective to be minimized for learning f that fits on the test distribution. By introducing importance weight $w ( \mathbf { x } , y ) : = p _ { \mathrm { t e } } ( \mathbf { x } , y ) / p _ { \mathrm { t r } } ( \mathbf { x } , y ) ^ { 2 }$ , as shown in previous studies [7, 8], this risk can be rewritten as follows,

$$
\begin{array} { r l } & { R _ { \mathrm { t e } } ( f ) = \mathbb { E } _ { p _ { \mathrm { t r } } ( \mathbf { x } , y ) } \left[ w ( \mathbf { x } , y ) \ell ( f ( \mathbf { x } ) , y ) \right] } \\ & { \qquad = \pi _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ \ell _ { w } ( f ( \mathbf { x } ) , + 1 ) \right] + ( 1 - \pi _ { \mathrm { t r } } ) \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ \ell _ { w } ( f ( \mathbf { x } ) , - 1 ) \right] = : R _ { \mathrm { t r } } ^ { \mathrm { w } } ( f ) , } \end{array}\tag{10}
$$

where we set $\ell _ { w } ( \mathbf { x } , y ) : = w ( \mathbf { x } , y ) \ell ( f ( \mathbf { x } ) , y )$ . This reformulation shows that the test risk can be expressed as the importance-weighted risk computed on the training distribution. While this risk depends on class-conditional densities $p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } )$ and $p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } )$ , by using the same procedure described in Section 3, it can be rewritten with UU training densities $p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } )$ and $p _ { \mathrm { t r } } ^ { \mathrm { b } } ( \mathbf { x } )$

$$
\begin{array} { r l r } & { } & { R _ { \mathrm { t r } } ^ { \mathrm { w } } ( f ) = a _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } ) } [ \ell _ { w } ( f ( \mathbf { x } ) , + 1 ) ] - c _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { b } } ( \mathbf { x } ) } [ \ell _ { w } ( f ( \mathbf { x } ) , + 1 ) ] } \\ & { } & { - b _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } ) } [ \ell _ { w } ( f ( \mathbf { x } ) , - 1 ) ] + d _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { b } } ( \mathbf { x } ) } [ \ell _ { w } ( f ( \mathbf { x } ) , - 1 ) ] , } \end{array}\tag{11}
$$

where $\begin{array} { r l r } & { a _ { \mathrm { t r } } : = \frac { ( 1 - \theta _ { \mathrm { t r } } ^ { \mathrm { b } } ) \pi _ { \mathrm { t r } } } { \theta _ { \mathrm { t r } } ^ { \mathrm { a } } - \theta _ { \mathrm { t r } } ^ { \mathrm { b } } } , b _ { \mathrm { t r } } : = } & { \frac { \theta _ { \mathrm { t r } } ^ { \mathrm { b } } ( 1 - \pi _ { \mathrm { t r } } ) } { \theta _ { \mathrm { t r } } ^ { \mathrm { a } } - \theta _ { \mathrm { t r } } ^ { \mathrm { b } } } , c _ { \mathrm { t r } } : = \frac { ( 1 - \theta _ { \mathrm { t r } } ^ { \mathrm { a } } ) \pi _ { \mathrm { t r } } } { \theta _ { \mathrm { t r } } ^ { \mathrm { a } } - \theta _ { \mathrm { t r } } ^ { \mathrm { b } } } } \end{array}$ , and $\begin{array} { r } { d _ { \mathrm { t r } } : = \ : \frac { \theta _ { \mathrm { t r } } ^ { \mathrm { a } } ( 1 - \pi _ { \mathrm { t r } } ) } { \theta _ { \mathrm { t r } } ^ { \mathrm { a } } - \theta _ { \mathrm { t r } } ^ { \mathrm { b } } } } \end{array}$ . Thus, the empirical estimate of Eq. (11) with UU training data $X _ { \mathrm { t r } } ^ { \mathrm { a } } \cup X _ { \mathrm { t r } } ^ { \mathrm { b } }$ is represented as

$$
\hat { R } _ { \mathrm { t r } } ^ { \mathrm { w } } ( f ) = \left| \frac { a _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { a } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { a } } } \ell _ { w } ( f ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { a } } ) , + 1 ) - \frac { c _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { b } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { b } } } \ell _ { w } ( f ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { b } } ) , + 1 ) \right|
$$

$$
+ \left| \frac { d _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { b } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { b } } } \ell _ { w } ( f ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { b } } ) , - 1 ) - \frac { b _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { a } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { a } } } \ell _ { w } ( f ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { a } } ) , - 1 ) \right| ,\tag{12}
$$

where we also used the absolute value correction to prevent overfitting.

## 4.3 Importance Weight Estimation from UU Data

Because importance weight w $( { \bf x } , y ) = p _ { \mathrm { t e } } ( { \bf x } , y ) / p _ { \mathrm { t r } } ( { \bf x } , y )$ in Eq (12) is not observable, we need to estimate it with UU data. A standard approach for this task is to use density-ratio estimation methods that directly estimate the ratio between training and test densities from data without density estimation [52, 17, 3]. However, they are often numerically unstable because $w ( \mathbf { x } , y )$ can diverge in regions where $p _ { \mathrm { t r } } ( \mathbf { x } , y )$ is small [60, 25]. To mitigate this issue, we instead use the relative density-ratio as the importance weight:

$$
w _ { \alpha } ( \mathbf { x } , y ) : = \frac { p _ { \mathrm { t e } } ( \mathbf { x } , y ) } { \alpha p _ { \mathrm { t e } } ( \mathbf { x } , y ) + ( 1 - \alpha ) p _ { \mathrm { t r } } ( \mathbf { x } , y ) } ,\tag{13}
$$

where $\alpha \in [ 0 , 1 ]$ is a hyperparameter $[ 6 0 ] . \ w _ { \alpha } ( \mathbf { x } , y )$ is always bounded above by $1 / \alpha$ , and recovers w $\left( \mathbf { x } , y \right)$ when $\alpha = 0$ . Thus, $w _ { \alpha } ( \mathbf { x } , y )$ is a bounded generalization of $w ( \mathbf { x } , y )$ . The importance weighting with the relative density-ratio has shown to improve stability and performance [60, 44, 27].

To estimate $w _ { \alpha } ( { \bf x } , y )$ , we introduce model $m ( \mathbf { x } , y ) \in [ 0 , 1 / \alpha ]$ , such as a neural network. Following previous studies [60, 17, 27], we determine parameters of $m ( \mathbf { x } , y )$ so that the expected squared error between true importance weight $w _ { \alpha } ( { \bf x } , y )$ and $m ( \mathbf { x } , y )$ is minimized:

$$
\begin{array} { r l } & { J ( m ) : = \mathbb { E } _ { p _ { \alpha } ( \mathbf { x } , y ) } \left[ \left( w _ { \alpha } ( \mathbf { x } , y ) - m ( \mathbf { x } , y ) \right) ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } _ { p _ { \mathrm { t e } } ( \mathbf { x } , y ) } \left[ \alpha m ( \mathbf { x } , y ) ^ { 2 } - 2 m ( \mathbf { x } , y ) \right] + \mathbb { E } _ { p _ { \mathrm { t r } } ( \mathbf { x } , y ) } \left[ ( 1 - \alpha ) m ( \mathbf { x } , y ) ^ { 2 } \right] + C , } \end{array}\tag{14}
$$

where $p _ { \alpha } ( \mathbf { x } , y ) : = \alpha p _ { \mathrm { t e } } ( \mathbf { x } , y ) + ( 1 - \alpha ) p _ { \mathrm { t r } } ( \mathbf { x } , y )$ and C is a constant term that does not depend on m. Letting $M _ { 1 } ( \mathbf { x } , y ) : = \alpha m ( \mathbf { x } , y ) ^ { 2 } - 2 m ( \mathbf { x } , y )$ and $M _ { 2 } ( \mathbf { x } , y ) : = ( 1 - \alpha ) m ( \mathbf { x } , y ) ^ { 2 }$ , we have

$$
J ( m ) { = } \pi _ { \mathrm { t e } } \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , { + } 1 ) \right] { + } ( 1 { - } \pi _ { \mathrm { t e } } ) \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , { - } 1 ) \right]
$$

$$
+ \pi _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , + 1 ) \right] + ( 1 - \pi _ { \mathrm { t r } } ) \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , - 1 ) \right] + C .\tag{15}
$$

By using Eq. (3), each expectation term in Eq. (15) can be rewritten as

$$
\pi _ { \mathrm { t e } } \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , + 1 ) \right] = a _ { \mathrm { t e } } \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { a } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , + 1 ) \right] - c _ { \mathrm { t e } } \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { b } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , + 1 ) \right] ,\tag{16}
$$

$$
( 1 - \pi _ { \mathrm { t e } } ) \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , - 1 ) \right] = - b _ { \mathrm { t e } } \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , - 1 ) \right] + d _ { \mathrm { t e } } \mathbb { E } _ { p _ { \mathrm { t e } } ^ { \mathrm { b } } ( \mathbf { x } ) } \left[ M _ { 1 } ( \mathbf { x } , - 1 ) \right] ,\tag{17}
$$

$$
\pi _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { p } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , + 1 ) \right] = a _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , + 1 ) \right] - c _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { b } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , + 1 ) \right] ,\tag{18}
$$

$$
( 1 - \pi _ { \mathrm { t r } } ) \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { n } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , - 1 ) \right] = - b _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { a } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , - 1 ) \right] + d _ { \mathrm { t r } } \mathbb { E } _ { p _ { \mathrm { t r } } ^ { \mathrm { b } } ( \mathbf { x } ) } \left[ M _ { 2 } ( \mathbf { x } , - 1 ) \right] ,\tag{19}
$$

where $\begin{array} { r l } & { a _ { \mathrm { t e } } : = \frac { ( 1 - \theta _ { \mathrm { t e } } ^ { \mathrm { b } } ) \pi _ { \mathrm { t e } } } { \theta _ { \mathrm { t e } } ^ { \mathrm { a } } - \theta _ { \mathrm { t e } } ^ { \mathrm { b } } } , b _ { \mathrm { t e } } : = \frac { \theta _ { \mathrm { t e } } ^ { \mathrm { b } } ( 1 - \pi _ { \mathrm { t e } } ) } { \theta _ { \mathrm { t e } } ^ { \mathrm { a } } - \theta _ { \mathrm { t e } } ^ { \mathrm { b } } } , c _ { \mathrm { t e } } : = \frac { ( 1 - \theta _ { \mathrm { t e } } ^ { \mathrm { a } } ) \pi _ { \mathrm { t e } } } { \theta _ { \mathrm { t e } } ^ { \mathrm { a } } - \theta _ { \mathrm { t e } } ^ { \mathrm { b } } } } \end{array}$ , and $\begin{array} { r } { d _ { \mathrm { t e } } : = \frac { \theta _ { \mathrm { t e } } ^ { \mathrm { a } } ( 1 - \pi _ { \mathrm { t e } } ) } { \theta _ { \mathrm { t e } } ^ { \mathrm { a } } - \theta _ { \mathrm { t e } } ^ { \mathrm { b } } } } \end{array}$

Since $M _ { 1 } ( \mathbf { x } , y ) = \alpha m ( \mathbf { x } , y ) ^ { 2 } - 2 m ( \mathbf { x } , y ) \geq - 1 / \alpha$ for all x and y, Eqs. (16) and (17) are bounded below $\mathrm { b y } - \pi _ { \mathrm { t e } } / \alpha$ and $- ( 1 - \pi _ { \mathrm { t e } } ) / \alpha$ , respectively. However, as in the risk, their empirical estimates can take smaller values and it may cause overfitting. To prevent this, we apply correction function $F ( z ) : = | z - k | + k ( k , z \in \mathbb { R } )$ where k is the lower bound of the loss [15]. When $z \geq k , F ( z )$ returns z. When $z < k , F ( z ) = 2 k - z > k$ . Thus, this function can penalize the loss for being smaller than k. Additionally, while Eqs. (18) and (19) are non-negative by definition, their empirical estimates can be negative. We correct this by applying the absolute value function $F ( z ) = | z |$ as in the risk. Consequently, the empirical estimate of Eq. (15) with the loss corrections becomes

$$
\begin{array} { r l } { \hat { f } ( m ) = | \frac { \alpha _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { s } } } \displaystyle \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { s } } } M _ { 1 } ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { s } } , + 1 ) - \frac { C _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { s } } } \displaystyle \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { s } } } M _ { 1 } ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { b } } , + 1 ) - k _ { 1 } | } & { } \\ { + | \frac { d _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { s } } } \displaystyle \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { s } } } M _ { 1 } ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { b } } , - 1 ) - \frac { b _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { s } } } \displaystyle \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { s } } } M _ { 1 } ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { a } } , - 1 ) - k _ { 2 } | } & { } \\ { + | \frac { \alpha _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { s } } } \displaystyle \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { s } } } M _ { 2 } ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { a } } , + 1 ) - \frac { C _ { \mathrm { t r } } } { N _ { \mathrm { t r } } ^ { \mathrm { s } } } \displaystyle \sum _ { n = 1 } ^ { N _ { \mathrm { t r } } ^ { \mathrm { s } } } M _ { 2 } ( \mathbf { x } _ { \mathrm { t r } , n } ^ { \mathrm { b } } , + 1 ) | } & { } \\  + | \frac  d  \end{array}\tag{20}
$$

where $k _ { 1 } : = - \pi _ { \mathrm { t e } } / \alpha , k _ { 2 } : = - ( 1 - \pi _ { \mathrm { t e } } ) / \alpha$ , and we omit constant terms that do not depend on m.

## 4.4 Classifier Loss Function

Our loss function for learning classifier f is a weighted sum of the empirical UU risk on the test distribution and the importance-weighted empirical UU risk on the training distribution in Eq. (12):

$$
\begin{array} { r } { \hat { L } ( f , m ) : = \beta \hat { R } _ { \mathrm { t e } } ( f ) + ( 1 - \beta ) \hat { R } _ { \mathrm { t r } } ^ { \mathrm { w } } ( f , m ) , } \end{array}\tag{21}
$$

Algorithm 1 Training procedure of the proposed method   
Require: UU data in the training and test distributions $X _ { \mathrm { t r } } ^ { \mathrm { a } } \cup X _ { \mathrm { t r } } ^ { \mathrm { b } } \cup X _ { \mathrm { t e } } ^ { \mathrm { a } } \cup X _ { \mathrm { t e } } ^ { \mathrm { b } }$ , mini-batch sizes   
for the training and test distributions $( B _ { \mathrm { t r } } , B _ { \mathrm { t e } } )$ , positive class-priors $( \pi _ { \mathrm { t r } } , \pi _ { \mathrm { t e } } , \theta _ { \mathrm { t r } } ^ { a } , \theta _ { \mathrm { t r } } ^ { b } , \theta _ { \mathrm { t e } } ^ { a } , \theta _ { \mathrm { t e } } ^ { b } )$   
relative parameter $\alpha ,$ and weighting parameter $\beta .$   
Ensure: Parameters of neural networks h, u, and v.   
1: repeat   
2: Sample UU data with size $B _ { \mathrm { t r } }$ from $X _ { \mathrm { t r } } ^ { \mathrm { a } } \cup X _ { \mathrm { t r } } ^ { \mathrm { b } }$   
3: Sample UU data with size $B _ { \mathrm { t e } }$ from $X _ { \mathrm { t e } } ^ { \mathrm { a } } \cup X _ { \mathrm { t e } } ^ { \mathrm { b } }$   
4: # Importance weight estimation   
5: Calculate loss in Eq. (20) on the sampled data   
6: Update parameters of v with the gradient of the loss fixing feature extractor h   
7: # Classifier learning   
8: Calculate the loss in Eq. (21) on the sampled data with current importance weights   
9: Update classifier parameters of u and h with the gradient of the loss fixing the importance   
weights   
10: until End condition is satisfied;

where $\beta \in [ 0 , 1 )$ is a weighting hyperparameter, and $\hat { R } _ { \mathrm { t e } } ( f )$ is the empirical test UU risk calculated with $X _ { \mathrm { t e } } ^ { \mathrm { a } } \cup X _ { \mathrm { t e } } ^ { \mathrm { b } }$

$$
\begin{array} { r } { \hat { R } _ { \mathrm { t e } } ( f ) = \left| \frac { a _ { \mathrm { t e } } } { N _ { \mathrm { t e } } ^ { \mathrm { a } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t e } } ^ { \mathrm { a } } } \ell ( f ( \mathbf { x } _ { \mathrm { t e } , n } ^ { \mathrm { a } } ) , + 1 ) - \frac { c _ { \mathrm { t e } } } { N _ { \mathrm { t e } } ^ { \mathrm { b } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t e } } ^ { \mathrm { b } } } \ell ( f ( \mathbf { x } _ { \mathrm { t e } , n } ^ { \mathrm { b } } ) , + 1 ) \right| } \\ { + \left| \frac { d _ { \mathrm { t e } } } { N _ { \mathrm { t e } } ^ { \mathrm { b } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t e } } ^ { \mathrm { b } } } \ell ( f ( \mathbf { x } _ { \mathrm { t e } , n } ^ { \mathrm { b } } ) , - 1 ) - \frac { b _ { \mathrm { t e } } } { N _ { \mathrm { t e } } ^ { \mathrm { a } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t e } } ^ { \mathrm { a } } } \ell ( f ( \mathbf { x } _ { \mathrm { t e } , n } ^ { \mathrm { a } } ) , - 1 ) \right| . } \end{array}\tag{22}
$$

Here, we use Eq. (6) to derive Eq. (22) and explicitly describe the dependency of the importance weight model m in $\hat { R } _ { \mathrm { t r } } ^ { \mathrm { w } }$ for clarity. Although unlabeled data $X _ { \mathrm { t e } } ^ { \mathrm { a } } \cup X _ { \mathrm { t e } } ^ { \mathrm { b } }$ are used for estimating the importance weights, they are not directly used for learning f in $\hat { R } _ { \mathrm { t r } } ^ { \mathrm { w } }$ . By using both empirical risks, we can use all available data $X _ { \mathrm { t r } } ^ { \mathrm { a } } \cup X _ { \mathrm { t r } } ^ { \mathrm { b } } \cup X _ { \mathrm { t e } } ^ { \mathrm { a } } \cup X _ { \mathrm { t e } } ^ { \mathrm { b } }$ for learning f directly.

## 4.5 Training

To compute $\hat { R } _ { \mathrm { t r } } ^ { \mathrm { w } } ( f , m )$ , the conventional importance weighting approach first estimates the importance weights and subsequently uses them to calculate the importance-weighted risk [60, 52, 17, 3]. However, the importance weight is difficult to estimate when using complex models such as neural networks or complex data such as image data [7, 27, 18, 42]. Because the estimation error directly affects the subsequent classifier learning, this two-step approach often does not work well [7, 27]. To alleviate this, recent methods use a dynamic approach that iterates between the importance weight estimation and classifier learning while sharing a neural network for feature extraction [7, 8, 27]. By sharing the feature extractor, they can effectively perform the importance weighting.

The proposed method also follows this dynamic approach. Specifically, we use the following neural networks for modeling classifiers and importance weights,

$$
f ( \mathbf { x } ) : = u ( h ( \mathbf { x } ) ) , m ( \mathbf { x } , y ) : = v ( [ h ( \mathbf { x } ) , y ] ) ,\tag{23}
$$

where $h : \mathcal { X } \to \mathbb { R } ^ { K } , u : \mathbb { R } ^ { K } \to \mathbb { R }$ , and $v : \mathbb { R } ^ { K + 2 }  \mathbb { R }$ are neural networks for feature extraction, classifiers, and importance weights, respectively. Here, we assume that label $y \in \{ - 1 , + 1 \}$ is represented as one-hot encoding and $[ \cdot , \cdot ]$ is a concatenation. Algorithm 1 shows the training procedure of our method with stochastic gradient descent methods. We first randomly sample the mini-batch data (UU data) from the training and test distributions (Lines 2–3). Then, we calculate the loss in Eq. (20) for the importance weight estimation (Line 5) and update parameters of v with the gradient of the loss fixing feature extractor h (Line 6). We fixed h to avoid overfitting as in [7, 27]. Then, by using the estimated importance weights, we calculate the loss in Eq. (21) for classifier learning (Line 8). We update parameters of classifier u and h by using the gradient of the loss (Line 9). In this step, we fixed the importance weights to avoid learning a meaningless model, $m ( \mathbf { x } , y ) = 0$ for all x and y.

## 5 Experiments

## 5.1 Data

We used four real-world datasets in the main paper: MNIST [28], FashionMNIST (FMNIST) [57], CIFAR10 [22], and DIABETES [12]. They have been commonly used in recent distribution shift adaptation studies [27, 26, 7, 41]. MNIST and FMNIST are $2 8 \times 2 8$ grayscale image datasets, while CIFAR10 consists of 32×32 RGB images. DIABETES is a tabular dataset with distribution shift [12], where the task is to predict whether a respondent has diabetes from 142 dimensional features. Data from the training and test distributions are “white non-hispanic” and other racial groups, respectively.

For MNIST, FMNIST, and CIFAR10, following the previous studies [27, 8], we constructed binary classification tasks by partitioning the original classes of each dataset into positive and negative classes, and created two types of distribution shifts: the first one is the support shift, where the support of input x changes between the training and testing phases. This shift is a particularly challenging type of covariate shift [8, 27]. The second one is the input-output relation shift (concept shift), where the inputs’ support does not change but $p ( y | \mathbf { x } )$ varies between the training and testing phases. Note that no methods use the knowledge that such shifts have occurred in our experiments. The detailed construction procedure is provided in Section A.

For UU data in the training distribution $X _ { \mathrm { t r } } ^ { \mathrm { a } }$ and $X _ { \mathrm { t r } } ^ { \mathrm { b } }$ , we set the numbers of the instances $( N _ { \mathrm { t r } } ^ { a } , N _ { \mathrm { t r } } ^ { b } )$ to (2500, 2500). For UU data in the test distribution $X _ { \mathrm { t e } } ^ { \mathrm { a } }$ and $X _ { \mathrm { t e } } ^ { \mathrm { b } }$ , we changed the numbers of the instances $( N _ { \mathrm { t e } } ^ { a } , N _ { \mathrm { t e } } ^ { b } )$ ) within $\{ ( 5 0 , 5 0 ) , ( 1 0 0 , 1 0 0 ) , ( 1 5 0 , 1 5 0 ) \}$ . For these UU datasets, we changed the class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } ) : = ( \theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } ) = ( \theta _ { \mathrm { t e } } ^ { \mathrm { a } } , \theta _ { \mathrm { t e } } ^ { \mathrm { b } } )$ within $\{ ( 0 . 8 , 0 . 2 ) , ( 0 . 7 , 0 . 3 ) , ( 0 . 6 , 0 . 4 ) \}$ as in the previous studies [35]. We also used different UU data in the test distribution where each size is 100 as validation data. We used 2000 positive and 2000 negative data in the test distribution as test data for evaluation $( \mathrm { i . e . , } \pi _ { \mathrm { t e } } = 0 . 5 )$ . Training, validation, and test datasets did not overlap. We conducted 10 experiments for each pair of the number of UU data in the test distribution and the class-priors while changing the random seeds and evaluated the mean test accuracy.

## 5.2 Comparison Methods

We compared our method with five UU learning-based methods: UU learning method on the test distribution (teUU) [35], UU learning method on the training distribution (trUU), multi-task learning method of UU learning (mtUU), multi-task learning method of UU learning with a single neural network (mtsUU), and domain adaptation method for UU learning (daUU). All methods including our method used neural networks for classifiers and the absolute value correction to prevent overfitting.

teUU and trUU learn the classifier by minimizing the non-negative empirical risk (in Eq. (6)) with UU data in the test and training distributions, respectively. Note that teUU and trUU are stateof-the-art UU learning methods. Since no distribution adaptation methods for UU learning exist, we created several adaptation methods (mtUU, mtsUU, and daUU). mtUU and mtsUU learn the classifier by minimizing the weighted sum of the non-negative empirical UU risks on the training and test distributions. mtUU used a two-head neural network for handling the difference between the training and test distributions. mtsUU used the single neural network as in the proposed method. The difference between the proposed method and mtsUU is whether or not the importance weights are used in Eq. (21) (i.e., mtsUU used $m ( \mathbf { x } , y ) = 1$ for all x, y). daUU minimizes the weighted sum of the non-negative empirical UU risks on both distributions (i.e., the loss of mtsUU) while minimizing the feature discrepancy to obtain invariant features as in [49]. We used the maximum mean discrepancy (MMD) loss [30] to minimize the feature discrepancy, which is commonly used for distribution shift adaptation [9, 27]. The details of network architectures and hyperparameters are described in Sections B and C.

## 5.3 Results

Table 1 shows the average test accuracies for each dataset. The full results with the standard deviations are described in subsection D.11. The proposed method performed the best or comparably to it in almost all cases (6 out of 7 cases). teUU and trUU tended to perform worse than the proposed method because teUU used only a small amount of UU data in the test distribution and trUU used only UU data in the training distribution. Although mtsUU and mtUU used UU data in both distributions as in the proposed method, their performance was limited since they do not have an explicit mechanism for transfer. As mtsUU is the proposed method without importance weights, the result highlights the effectiveness of the importance weights. Although daUU aims to transfer by learning invariant features, it does not explicitly target the test risk. In contrast, the proposed method achieved better performance in more cases by directly minimizing the test risk within the importance-weighting framework. The results with each number of UU data in the test distribution and with each class-priors were described in subsections D.1 and D.2, respectively. The proposed method worked well on each case.

Table 1: Average test accuracies over different UU test data sizes $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } )$ within {(50, 50), (100, 100), (150, 150)} and class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. In the Shift column, ‘S’ and ‘IO’ represent the support shift and input-output relation shift, respectively. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Shift</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td>MNIST</td><td>S IO</td><td>0.7482 0.7211</td><td>0.7419 0.7334</td><td>0.5879 0.5412</td><td>0.7388 0.6922</td><td>0.7355 0.6996</td><td>0.7423 0.6929</td></tr><tr><td>FMNIST</td><td>S IO</td><td>0.9620 0.8841</td><td>0.9531 0.8929</td><td>0.9161 0.7278</td><td>0.9551 0.8796</td><td>0.9509 0.8823</td><td>0.9550 0.8769</td></tr><tr><td>CIFAR10</td><td>S IO</td><td>0.8294 0.7125</td><td>0.6929 0.6864</td><td>0.8258 0.6665</td><td>0.8281 0.7084</td><td>0.8166 0.6947</td><td>0.8308 0.7099</td></tr><tr><td>DIABETES</td><td></td><td>0.7090</td><td>0.6300</td><td>0.7115</td><td>0.7117</td><td>0.7011</td><td>0.7118</td></tr></table>

Table 2: Results with a few PN data in the test distribution: Average test accuracies over different UU test data sizes $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } )$ within {(10, 10), (50, 50), (100, 100)} and training class-priors $( \theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. Class-priors of UU test data $( \theta _ { \mathrm { t e } } ^ { \mathrm { a } } , \theta _ { \mathrm { t e } } ^ { \mathrm { b } } )$ were set to (1, 0).
<table><tr><td>Data</td><td>Shift</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td>MNIST</td><td>S IO</td><td>0.8237 0.8075</td><td>0.8184 0.8102</td><td>0.6012 0.5543</td><td>0.8194 0.8012</td><td>0.8309 0.8127</td><td>0.8177 0.7998</td></tr><tr><td>FMNIST</td><td>S IO</td><td>0.9726 0.9367</td><td>0.9655 0.9285</td><td>0.9181 0.7505</td><td>0.9719 0.9364</td><td>0.9705 0.9299</td><td>0.9717 0.9393</td></tr><tr><td>CIFAR10</td><td>S IO</td><td>0.8492 0.7605</td><td>0.7555 0.7455</td><td>0.8286 0.6732</td><td>0.8422 0.7563</td><td>0.8376 0.7492</td><td>0.8466 0.7611</td></tr><tr><td>DIABETES</td><td></td><td>0.7251</td><td>0.6757</td><td>0.7233</td><td>0.7252</td><td>0.7209</td><td>0.7238</td></tr></table>

Figure 2 shows the distribution of estimated importance weights of the proposed method on MNIST dataset with the support shift when $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } ) ~ = ~ ( 5 0 , 5 0 )$ and $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } ) ~ = ~ ( 0 . 7 , 0 . 3 )$ ‘High’ and ‘Low’ represent data in the test distribution and data that are not in the test distribution, respectively. The proposed method was able to correctly estimate importance weights, i.e., the weights of ‘High’ and $\mathrm { \cdot } \mathrm { L o w } ^ { \mathrm { \bullet } }$ are large and small, respectively.

![](images/e630d54f7ee28788250abdd66ce2d6747d6a21fe4d488f2b4e531cf7fe886405.jpg)  
Figure 2: Importance weight distribution of the proposed method on MNIST with the support shift.

Table 2 shows the results when types of supervision were different between the training and testing phases: a few PN data in the test distribution and noisy labeled data in the training distribution. We evaluated this setting because a few data might be accurately labeled by domain experts. Considering the cost of precise labeling, we used fewer instances than in previous setting. The proposed method also performed the best or comparably to it in 6 out of 7 cases. This result shows that the proposed method works well even with different types of supervision.

## 6 Conclusion

In this paper, we proposed a distribution shift adaptation method for UU learning with the importance weighting. Our method can handle various types of learning problems under distribution shift such as PN, PU, noisy label, and similarity-based learning. Moreover, it can perform adaptation without assumptions of the shift types. The experiments showed the effectiveness of our method.

## References

[1] H. Bao, G. Niu, and M. Sugiyama. Classification from pairwise similarity and unlabeled data. In ICML, 2018.

[2] J. Bekker and J. Davis. Learning from positive and unlabeled data: A survey. Machine Learning, 109(4):719–760, 2020.

[3] S. Bickel, M. Brückner, and T. Scheffer. Discriminative learning under covariate shift. Journal of Machine Learning Research, 10(9), 2009.

[4] N. Charoenphakdee, J. Lee, and M. Sugiyama. On symmetric losses for learning from corrupted labels. In ICML, 2019.

[5] C.-K. Chiang and M. Sugiyama. Unified risk analysis for weakly supervised learning. Transactions on Machine Learning Research, 2025.

[6] M. Du Plessis, G. Niu, and M. Sugiyama. Convex formulation for learning from positive and unlabeled data. In ICML, 2015.

[7] T. Fang, N. Lu, G. Niu, and M. Sugiyama. Rethinking importance weighting for deep learning under distribution shift. NeurIPS, 2020.

[8] T. Fang, N. Lu, G. Niu, and M. Sugiyama. Generalizing importance weighting to a universal solver for distribution shift problems. NeurIPS, 2023.

[9] A. Farahani, S. Voghoei, K. Rasheed, and H. R. Arabnia. A brief review of domain adaptation. ICDATA, 2021.

[10] L. Feng, S. Shu, N. Lu, B. Han, M. Xu, G. Niu, B. An, and M. Sugiyama. Pointwise binary classification with pairwise confidence comparisons. In ICML, 2021.

[11] Y. Ganin and V. Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, 2015.

[12] J. Gardner, Z. Popovic, and L. Schmidt. Benchmarking distribution shift in tabular data with tableshift. NeurIPS, 2023.

[13] Z. Hammoudeh and D. Lowd. Learning from positive and unlabeled data with arbitrary positive shift. NeurIPS, 2020.

[14] Z. Han, X.-J. Gui, H. Sun, Y. Yin, and S. Li. Towards accurate and robust domain adaptation under multiple noisy environments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):6460–6479, 2022.

[15] T. Ishida, I. Yamane, T. Sakai, G. Niu, and M. Sugiyama. Do we need zero training loss after achieving zero training error? In ICML, 2020.

[16] S. Jain, M. White, and P. Radivojac. Estimating the class prior and posterior from noisy positives and unlabeled data. NeurIPS, 2016.

[17] T. Kanamori, S. Hido, and M. Sugiyama. A least-squares approach to direct importance estimation. The Journal ofMachine Learning Research, 10:1391–1445, 2009.

[18] M. Kato and T. Teshima. Non-negative bregman divergence minimization for deep direct density ratio estimation. In ICML, 2021.

[19] D. P. Kingma and J. Ba. Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[20] R. Kiryo, G. Niu, M. C. Du Plessis, and M. Sugiyama. Positive-unlabeled learning with non-negative risk estimator. NeurIPS, 2017.

[21] A. Krause, P. Perona, and R. Gomes. Discriminative clustering by regularized information maximization. NeurIPS, 2010.

[22] A. Krizhevsky, G. Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[23] A. Kumagai and T. Iwata. Unsupervised domain adaptation by matching distributions based on the maximum mean discrepancy via unilateral transformations. In AAAI, 2019.

[24] A. Kumagai, T. Iwata, and Y. Fujiwara. Transfer anomaly detection by inferring latent domain representations. NeurIPS, 2019.

[25] A. Kumagai, T. Iwata, and Y. Fujiwara. Meta-learning for relative density-ratio estimation. In NeurIPS, 2021.

[26] A. Kumagai, T. Iwata, H. Takahashi, T. Nishiyama, and Y. Fujiwara. Auc maximization under positive distribution shift. NeurIPS, 2024.

[27] A. Kumagai, T. Iwata, H. Takahashi, T. Nishiyama, and Y. Fujiwara. Importance-weighted positive-unlabeled learning for distribution shift adaptation. In AISTATS, 2025.

[28] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.

[29] C.-Y. Lee, T. Batra, M. H. Baig, and D. Ulbricht. Sliced wasserstein discrepancy for unsupervised domain adaptation. In CVPR, 2019.

[30] Y. Li, K. Swersky, and R. Zemel. Generative moment matching networks. In ICML, 2015.

[31] T. Liu and D. Tao. Classification with noisy labels by importance reweighting. IEEE Transactions on pattern analysis and machine intelligence, 38(3):447–461, 2015.

[32] N. Lu, S. Lei, G. Niu, I. Sato, and M. Sugiyama. Binary classification from multiple unlabeled datasets via surrogate set classification. In ICML, 2021.

[33] N. Lu, G. Niu, A. K. Menon, and M. Sugiyama. On the minimal supervision for training any binary classifier from only unlabeled data. In ICLR, 2019.

[34] N. Lu, T. Zhang, T. Fang, T. Teshima, and M. Sugiyama. Rethinking importance weighting for transfer learning. In Federated and Transfer Learning, pages 185–231. Springer, 2022.

[35] N. Lu, T. Zhang, G. Niu, and M. Sugiyama. Mitigating overfitting in supervised classification from two unlabeled datasets: a consistent risk correction approach. In AISTATS, 2020.

[36] A. Menon, B. Van Rooyen, C. S. Ong, and B. Williamson. Learning from corrupted binary labels via class-probability estimation. In ICML, 2015.

[37] J. G. Moreno-Torres, T. Raeder, R. Alaiz-Rodríguez, N. V. Chawla, and F. Herrera. A unifying view on dataset shift in classification. Pattern recognition, 45(1):521–530, 2012.

[38] S. Motiian, M. Piccirilli, D. A. Adjeroh, and G. Doretto. Unified deep supervised domain adaptation and generalization. In ICCV, 2017.

[39] S. J. Pan and Q. Yang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10):1345–1359, 2009.

[40] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in pytorch. 2017.

[41] Y.-Y. Qian, P. Zhao, Y.-J. Zhang, M. Sugiyama, and Z.-H. Zhou. Efficient non-stationary online learning by wavelets with applications to online distribution shift adaptation. In ICML, 2024.

[42] B. Rhodes, K. Xu, and M. U. Gutmann. Telescoping density-ratio estimation. NeurIPS, 2020.

[43] K. Saito, K. Watanabe, Y. Ushiku, and T. Harada. Maximum classifier discrepancy for unsupervised domain adaptation. In CVPR, 2018.

[44] T. Sakai and N. Shimizu. Covariate shift adaptation on learning from positive and unlabeled data. In AAAI, 2019.

[45] T. Shimada, H. Bao, I. Sato, and M. Sugiyama. Classification from pairwise similarities/dissimilarities and unlabeled data via empirical risk minimization. Neural Computation, 33(5):1234–1268, 2021.

[46] H. Shimodaira. Improving predictive inference under covariate shift by weighting the loglikelihood function. Journal of statistical planning and inference, 90(2):227–244, 2000.

[47] Y. Shu, Z. Cao, M. Long, and J. Wang. Transferable curriculum for weakly-supervised domain adaptation. In AAAI, 2019.

[48] P. Singhal, R. Walambe, S. Ramanna, and K. Kotecha. Domain adaptation: challenges, methods, datasets, and applications. IEEE access, 11:6973–7020, 2023.

[49] J. Sonntag, G. Behrens, and L. Schmidt-Thieme. Positive-unlabeled domain adaptation. arXiv preprint arXiv:2202.05695, 2022.

[50] M. Sugiyama, H. Bao, T. Ishida, N. Lu, and T. Sakai. Machine learning from weak supervision: an empirical risk minimization approach. MIT Press, 2022.

[51] M. Sugiyama, M. Krauledat, and K.-R. Müller. Covariate shift adaptation by importance weighted cross validation. Journal ofMachine Learning Research, 8(5), 2007.

[52] M. Sugiyama, T. Suzuki, and T. Kanamori. Density ratio estimation in machine learning. Cambridge University Press, 2012.

[53] C. Tan, F. Sun, T. Kong, W. Zhang, C. Yang, and C. Liu. A survey on deep transfer learning. In ICANN, 2018.

[54] V. Vapnik. Statistical learning theory. John Wiley & Sons google schola, 2:831–842, 1998.

[55] G. Wilson and D. J. Cook. A survey of unsupervised deep domain adaptation. ACM Transactions on Intelligent Systems and Technology, 11(5):1–46, 2020.

[56] Y. Wu, X. Xia, J. Yu, B. Han, G. Niu, M. Sugiyama, and T. Liu. Making binary classification from multiple unlabeled datasets almost free of supervision. arXiv preprint arXiv:2306.07036, 2023.

[57] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

[58] Z. Xie, Y. Liu, H.-Y. He, M. Li, and Z.-H. Zhou. Weakly supervised auc optimization: a unified partial auc approach. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[59] L. Xu, J. Neufeld, B. Larson, and D. Schuurmans. Maximum margin clustering. NeurIPS, 2004.

[60] M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya, and M. Sugiyama. Relative density-ratio estimation for robust distribution comparison. Neural computation, 25(5):1324–1370, 2013.

[61] S. Yu, Z. Zhu, B. Liu, A. K. Jain, and J. Zhou. Robust unsupervised domain adaptation from a corrupted source. In ICDM, 2022.

[62] X. Yu, T. Liu, M. Gong, K. Zhang, K. Batmanghelich, and D. Tao. Label-noise robust domain adaptation. In ICML, 2020.

[63] H. Zhao, R. T. Des Combes, K. Zhang, and G. Gordon. On learning invariant representations for domain adaptation. In ICML, 2019.

## A Construction Procedure of Support and Input-output Relation Shifts

In this section, we describe the construction procedure for the distribution shifts, which is the same as that used in the previous study [27].

We first describe the procedure to construct the support shift. For MNIST, even and odd digits were treated as negative and positive, respectively. We used digits 0, 2, and 4 (4, 6, and 8) for negative- and digits 1, 3, and 5 (5, 7, and 9) for positive-conditional densities of the training (test) distribution. For FMNIST, the upper garments (0, 2, 3, 4, and 6) and the others were treated as negative and positive, respectively, where numbers in parentheses represent class labels. We used class labels 0, 2, and 3 (3, 4, and 6) for negative- and class labels 1, 5, and 7 (7, 8, and 9) for positive-conditional densities of the training (test) distribution. For CIFAR10, the animal categories (2, 3, 4, 5, 6, and 7) and the others were treated as negative and positive, respectively. We used class labels 2, 3, 4, and 5 (4, 5, 6, and 7) as negative and class labels 0, 1, and 8 (1, 8, and 9) as positive in the training (test) distribution.

We next describe the procedure to construct the input-output shift. For MNIST, we used even digits for negative- and odd digits for positive-conditional densities of the test distribution. For the training distribution, we swapped the digits 0 and 2 with 1 and 3 (e.g., the digits 0, 2, 5, 7, and 9 were used for the positive-conditional density of the training distribution). For FMNIST, we used upper garments (0, 2, 3, 4, and 6) for negative- and the others for positive-conditional densities of the test distribution. We swapped the class labels 0 and 2 with 1 and 5 in the training distribution. For CIFAR10, we used the animal categories (2, 3, 4, 5, 6, and 7) for negative- and the vehicles for positive-conditional densities of the test distribution. We swapped the class labels 2 and 3 with 0 and 1 in the training distribution.

## B Neural Network Architectures

For MNIST, FMNIST, and DIABETES, a three-layered feed-forward neural network with ReLU activation was used for feature extractor h. The number of hidden and output nodes was 128. For CIFAR10, a convolutional neural network, which consisted of two convolutional blocks followed by a two-layered feed-forward neural network, was used for feature extractor h. The first (second) convolutional block comprised a 6 (16) filter $5 \times 5$ convolution, the ReLU activation, and a $2 \times 2$ max-pooling layer. The numbers of the hidden and output nodes were 120 and 84, respectively. One-layered and two-layered feed-forward neural networks were used for u and v, respectively. For the output activation of $v ,$ we used $\textstyle { \frac { 1 } { \alpha } } \sigma ( \cdot )$ , where σ is a sigmoid function, to match the value range of the relative density-ratio (importance weights). For all comparison methods, the same neural network architecture $( \mathrm { i . e . , } u ( h ( \mathbf { x } ) ) )$ ) was used for the classifier. For mtUU, two different classifier heads (i.e., u) were used for the training and test distributions.

## C Hyperparameters

For all methods, the empirical UU risk with validation data on the test distribution was used to select hyperparameters and early-stopping to mitigate overfitting. For all methods, the sigmoid loss was used for loss ℓ as in the previous study [20, 27]. For the proposed method, relative parameter α was selected from {0.1, 0.5, 0.9} and training class-prior $\pi _ { \mathrm { t r } }$ was set to 0.5. For the proposed method, mtUU, mtsUU, and daUU, weighting parameter $\beta$ was selected from {0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9}. Note that the proposed method, mtUU, and mtsUU are equivalent to teUU when $\beta = 1$ . For daUU, the MMD loss was applied to $h ( \mathbf { x } )$ . The number of RBF kernel mixtures was set to 5 as in the previous study [30]. The weighting parameter of the MMD loss was chosen from $\{ 1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { \div } \}$ . The mini-batch sizes for the training and test distributions, $B _ { \mathrm { t r } }$ and $B _ { \mathrm { t e } } ,$ were set to 512. For all methods, we used the Adam optimizer [19]. We set the learning rate to $1 0 ^ { - 4 }$ and $1 0 ^ { - 3 }$ for classifier learning and importance weight estimation, respectively. The maximum number of epochs was 200. All methods were implemented using Pytorch [40], and all experiments were conducted on a Linux server with an Intel Xeon CPU and A100 GPU.

![](images/9e0ec2723357bb2ad8e162fdaec6da9c67b32517191f2722d0576605b8cd9659.jpg)  
(a) MNIST(S)

![](images/2d0582ebf9e6f4aba33086c6bf309e6bab8cbee74bb7514c3d820644351a012d.jpg)  
(b) MNIST(IO)

![](images/c2f417d8bbf0b4fa5d1977eb9eb9670bcc0c450f030db6714e2c1c9ad118eef2.jpg)  
(c) FMNIST(S)

![](images/a012f909c087bb4623824a5d72732bdc6ac2df3c5facabf4e48077740bbc83df.jpg)  
(d) FMNIST(IO)

![](images/4b394e975a5462b472c642f10f8b5993510903ae9b3e33551bce509e07ca855c.jpg)  
(e) CIFAR10(S)

![](images/55c7f889b87e1536e9034c54b56c67a852769a10ca94d95c99bbdef871f124a3.jpg)  
(f) CIFAR10(IO)

![](images/cd361d1467f9c0072743584b8c082e14aa4126a44256dccfb68ba32c41610bcb.jpg)  
(g) DIABETES

Figure 3: The average test accuracies with the standard errors when changing the number of UU data in the test distribution.  
![](images/3ab7597fd9d18f2bc451c3072a845c0d069f466de714aedf205274718270c91d.jpg)  
(a) MNIST(S)

![](images/52a9a687df848973a580bdb870984ca72589d7cb046e217e19fe93ca597d5c3b.jpg)  
(b) MNIST(IO)

![](images/e688e23fae1e5163ca6989842ca4f5408a1499e3a2f5d6b4894e6c4024a50ffc.jpg)

![](images/0101403812e8b3f25b380446c255084e0e18df0c59db3b7088c50d58de06e6f5.jpg)  
(c) FMNIST(S)  
(d) FMNIST(IO)

![](images/5f943ddd9b77a6aece0c298e45b5605ea6928c092172d54adc4cb8062c54807f.jpg)  
(e) CIFAR10(S)

![](images/c6fd46bd57fc4a1bba61261651e1e9d595083a26c298211715c70725cd2d670a.jpg)  
(f) CIFAR10(IO)

![](images/b4bfa5b48fcb840e249c0fa650b5db41bb40f6b03fc5b8999bb39d034eea911d.jpg)  
(g) DIABETES  
Figure 4: The average test accuracies with the standard errors when changing the class-priors of UU data.

## D Additional Experimental Results

## D.1 Results with Different Numbers of UU Data in the Test Distribution

Figure 3 shows the average test accuracies with the standard errors when changing the numbers of UU data in the test distribution in the case where data of both the training and test distributions are noisy. As the number of UU data increased, the performance of the proposed method improved as expected. The proposed method tended to work well in each UU data size.

## D.2 Results with Different Class-priors of Given UU Datasets

Figure 4 shows the average test accuracies with the standard errors when changing the class-priors of UU data in the test distribution in the case where data of both the training and test distributions are noisy. As the class-priors of UU data became closer, the performance of each method decreased. This is because when the class-priors are close, two sets of unlabeled data become similar and thu are less informative to learn a binary classifier. The proposed method tended to work well in each setting of the class-priors.

![](images/3bdf2cf00e68ce0f2221a788cff30f80de8139a3260f9e15b265188b2f85cf65.jpg)  
(a) MNIST(S)

![](images/bdc9c1b41ff187d00203f5e1a3a62408ecfb59d2dbfddefcbcadf0df0d0422ab.jpg)  
(b) MNIST(IO)

![](images/9006d11088989d6c1d9233e992a8273d36c49568d624bd74f4a4b24d042f1563.jpg)  
(c) FMNIST(S)

![](images/0f5b26171e0c53b558daa9ae5f5ef77300ecc9bf65c3d17280bd8458a0f75600.jpg)  
(d) FMNIST(IO)

![](images/d15c95f3231704c6f5e88513d617e983f06357db16a24c7e4b6fcc4d2c5a990e.jpg)  
(e) CIFAR10(S)

![](images/151c2bfc8550f5bfa1862ff035a2c446f0f79bf1804e2a2909214931ab1c092c.jpg)  
(f) CIFAR10(IO)

![](images/0bcf2fa09a7a8d9770f9631f8b8b86fb507e725ad194d509a4934edf112bf176.jpg)  
(g) DIABETES  
Figure 5: The average test accuracies with the standard errors of our method when changing $\beta .$

![](images/f06aeb081845dca02cdd0870895aae19a7712c225bf764c94591447cbbb51529.jpg)  
(a) MNIST(S)

![](images/9f1e5e3324b96366b3be483dcee2ae9abec452dd05689483659174965d19dba2.jpg)

![](images/9973eb12e6a965cbbb1d8f989e07b15f2f40448af80f9ab7aea14b88f72b8ff0.jpg)  
(b) MNIST(IO)

![](images/abc9b83041141dcb63a2d8e1529aaf99e776a87ecb531ee91144549de33eff2a.jpg)  
(c) FMNIST(S)  
(d) FMNIST(IO)

![](images/c00ce727a1c8116857e2af94beb7544bacb4a7b0910bbc352f19cd2069938e82.jpg)  
(e) CIFAR10(S)

![](images/03bb97b60e90ef4495bdf7af4828f8ff405ed8ee9ad4bd83bbe106a305fa8896.jpg)  
(f) CIFAR10(IO)

![](images/56185ed34938d3fde08a30846af1bc494e4a38c7a0d77bcde9e65e20cf7c8901.jpg)  
(g) DIABETES  
Figure 6: The average test accuracies with the standard errors of our method when changing α.

## D.3 Dependency of Weighting Parameter $\beta$

Figure 5 shows the average test accuracies with the standard errors when changing weighting parameter β in the case where data of both the training and test distributions are noisy. The value of β is used in Eq. (21) for controlling the effects of two empirical risks. Note that the proposed method with $\beta = 1$ is equivalent to teUU that uses only UU data in the test distribution. Although the trend of the results varied across datasets, the proposed method was able to select good β by using validation UU data.

## D.4 Dependency of Relative Parameter α

Figure 6 shows the average test accuracies with the standard errors when changing relative parameter α of Eq. (13) in the case where data of both the training and test distributions are noisy. Although the trend varied across datasets, the performance tended not to change significantly even when α was changed. This result shows that the proposed method is relatively robust against the value of α.

Table 3: Comparison with the proposed method that sequentially performs the label estimation and the importance weighting (Ours w/ seq): average test accuracies over different UU data sizes in the test distribution and class-priors. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data Shift</td><td>MNIST</td><td>IO</td><td>FMNIST</td><td></td><td>CIFAR10</td><td>DIABETES IO</td></tr><tr><td>Ours</td><td>S 0.7482</td><td>0.7211</td><td>S 0.9620</td><td>IO 0.8841</td><td>S 0.8294</td><td>0.7090</td></tr><tr><td>Ours w/ seq</td><td>0.7515</td><td>0.7186</td><td>0.9593</td><td>0.8906</td><td>0.7125 0.8170 0.6959</td><td>0.7123</td></tr></table>

## D.5 Comparison with Our Method that Sequentially Performs Label Estimation and Importance Weighting

The proposed method directly performs the importance weighting from given UU datasets. However, one possible approach is to first estimate (pseudo) PN data from the given UU data using existing UU learning methods and then apply the importance weighting method for PN data [8]. We compared this approach (Ours w/ seq) with the proposed method in Table 3. Note that Ours w/ seq is also considered within our proposal since importance weighting methods for UU data do not exist. The proposed method performed comparably to or even better than Ours w/ seq. Since Ours w/ seq sequentially performs the label estimation and the importance weighting, the error on the label estimation propagates the subsequent importance weighting, which may lead to the performance degrading in some cases. In contrast, since the proposed method performs both simultaneously, it might be more robust.

## D.6 Results with Noisy Class-priors

Hitherto, we have assumed that true class-priors of given UU datasets $\theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } , \theta _ { \mathrm { t e } } ^ { \mathrm { a } }$ , and $\theta _ { \mathrm { t e } } ^ { \mathrm { b } }$ are known to train classifiers. However, they might be unavailable in practice. Therefore, we investigated the performance of the proposed method with noisy class-priors.

We first considered independent noise in the class-priors following the previous work [32]. Specifically, we first set true class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } ) : = ( \dot { \theta } _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } ) = ( \theta _ { \mathrm { t e } } ^ { \mathrm { a } } , \theta _ { \mathrm { t e } } ^ { \mathrm { b } } )$ for the data generation to (0.7, 0.3). Then, we replaced each $\theta \in \{ \theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } , \theta _ { \mathrm { t e } } ^ { \mathrm { a } } , \theta _ { \mathrm { t e } } ^ { \mathrm { b } } \}$ by $\theta ^ { \prime } = \theta + \gamma \sigma$ where γ uniform randomly takes a value in $\{ - 1 , + 1 \}$ and $\sigma \in \{ 0 . 0 5 , 0 . 1 , 0 . 1 5 , 0 . 1 9 5 \}$ . We used each noisy $\theta ^ { \prime }$ as the class-prior for training classifiers. Table 4 shows the average test accuracies when changing values of $\sigma .$ Here, $\sigma = 0$ means that there is no noise. As the value of σ (magnitude of noise) increased, the performance of the proposed method tended to decrease. This result is reasonable since information on class-priors is only supervision to train a classifier.

We next considered a simple correlated/systematic noise setting, in which all class-priors of given UU datasets, $\theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } , \theta _ { \mathrm { t e } } ^ { \mathrm { a } } ,$ and $\theta _ { \mathrm { t e } } ^ { \mathrm { b } } ,$ are jointly biased in the same direction rather than perturbed independently. Specifically, we replaced each θ by $\theta ^ { \prime } = \theta + \sigma$ . We chose this setting because, in practice, estimates or prior knowledge may tend to be consistently overestimated or underestimated, rather than perturbed independently. Table 5 shows the average test accuracies when changing values of σ. As the noise magnitude |σ| increased, the performance of the proposed method also tended to decrease. However, compared with the independent-noise result in Table 4, the degradation was overall more moderate (e.g., on CIFAR10 (S) with $\sigma = 0 . 1 9 5$ , the test accuracy is 0.8229 in Table 5, compared with 0.7611 in Table 4). A possible reason is that this noise shifts all class-priors in the same direction and thus tends to preserve their relative relationships, leading to a milder degradation.

## D.7 Results on Larger-scale Image Datasets

Here, we evaluated the proposed method on larger versions of the image datasets used in the main paper (i.e., MNIST, FMNIST, and CIFAR10). For each image dataset, we set the training UU data size to $( N _ { \mathrm { t r } } ^ { \mathrm { a } } , N _ { \mathrm { t r } } ^ { \mathrm { b } } ) = ( 6 , 0 0 0 , 6 , 0 0 0 )$ and the test UU data size to $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } ) = ( 1 0 0 , 1 0 0 )$ . Table 6 shows the results. The proposed method performed the best or comparably to it in almost all cases (5 out of 6 cases). The results show the effectiveness of the proposed method on the image datasets with increased data sizes.

Table 4: Results of the proposed method with independent noise in the class-priors: average test accuracies over different UU data sizes in the test distribution. σ represents the magnitude of noise.
<table><tr><td>Data</td><td>Shift</td><td> $\overline { { \sigma = 0 } }$ </td><td> $\overline { { \sigma = 0 . 0 5 } }$ </td><td> $\overline { { \sigma = 0 . 1 } }$ </td><td> $\overline { { \sigma = 0 . 1 5 } }$ </td><td> $\overline { { \sigma = 0 . 1 9 5 } }$ </td></tr><tr><td rowspan="2">MNIST</td><td>S</td><td>0.7569</td><td>0.7495</td><td>0.7409</td><td>0.7245</td><td>0.6970</td></tr><tr><td>IO</td><td>0.7438</td><td>0.6968</td><td>0.6853</td><td>0.6735</td><td>0.6449</td></tr><tr><td rowspan="2">FMNIST</td><td>S</td><td>0.9653</td><td>0.9477</td><td>0.9391</td><td>0.9210</td><td>0.9072</td></tr><tr><td>IO</td><td>0.9050</td><td>0.8891</td><td>0.8737</td><td>0.8517</td><td>0.8371</td></tr><tr><td rowspan="2">CIFAR10</td><td>S</td><td>0.8404</td><td>0.8345</td><td>0.8080</td><td>0.7915</td><td>0.7611</td></tr><tr><td>IO</td><td>0.7244</td><td>0.7181</td><td>0.7058</td><td>0.7095</td><td>0.6889</td></tr><tr><td>DIABETES</td><td></td><td>0.7172</td><td>0.7205</td><td>0.7233</td><td>0.7214</td><td>0.6916</td></tr></table>

Table 5: Results of the proposed method with correlated noise in the class-priors: average test accuracies over different UU data sizes in the test distribution. |σ| represents the magnitude of noise.
<table><tr><td>Data</td><td>Shift</td><td> $\sigma = - 0 . 1 9 5$ </td><td> $\sigma = - 0 . 1 5$ </td><td> $\sigma = - 0 . 1$ </td><td> $\sigma = - 0 . 0 5$ </td><td> $\sigma = 0$ </td><td> $\sigma = 0 . 0 5$ </td><td> $\sigma = 0 . 1$ </td><td> $\sigma = 0 . 1 5$ </td><td> $\sigma = 0 . 1 9 5$ </td></tr><tr><td rowspan="2">MNIST</td><td>S</td><td>0.7198</td><td>0.7266</td><td>0.7443</td><td>0.7549</td><td>0.7569</td><td>0.7576</td><td>0.7469</td><td>0.7473</td><td>0.7436</td></tr><tr><td>IO</td><td>0.7003</td><td>0.7223</td><td>0.7374</td><td>0.7427</td><td>0.7438</td><td>0.7237</td><td>0.7167</td><td>0.6861</td><td>0.6750</td></tr><tr><td rowspan="3">FMNIST</td><td>S</td><td>0.8966</td><td>0.9106</td><td>0.9238</td><td>0.9481</td><td>0.9653</td><td>0.9562</td><td>0.9332</td><td>0.9221</td><td>0.9058</td></tr><tr><td>IO</td><td>0.8697</td><td>0.8701</td><td>0.8747</td><td>0.8872</td><td>0.9050</td><td>0.8918</td><td>0.8696</td><td>0.8534</td><td>0.8254</td></tr><tr><td></td><td>0.8195</td><td>0.8246</td><td>0.8315</td><td>0.8392</td><td>0.8404</td><td>0.8399</td><td>0.8351</td><td>0.8305</td><td>0.8229</td></tr><tr><td rowspan="2">CIFAR10</td><td>S IO</td><td>0.7045</td><td>0.7053</td><td>0.7142</td><td>0.7167</td><td>0.7244</td><td>0.7220</td><td>0.7176</td><td>0.7105</td><td>0.7116</td></tr><tr><td></td><td>0.7206</td><td>0.7231</td><td>0.7196</td><td>0.7171</td><td>0.7172</td><td>0.7173</td><td>0.7192</td><td>0.7214</td><td>0.7194</td></tr></table>

## D.8 Results on Larger-scale Tabular Datasets with Real Distribution Shift

We evaluated the proposed method with larger-scale tabular datasets with real distribution shift. In this experiments, we newly used a FOODSTAMP, a real-world tabular dataset used for recent distribution adaptation studies [12, 27]. In this dataset, the task is to predict whether an individual is receiving food stamps. The distribution shift occurs due to the difference of the geographic regions in which individuals live. For FOODSTAMP and DIABETES datasets, we set the training UU data size $( N _ { \mathrm { t r } } ^ { \mathrm { a } } , N _ { \mathrm { t r } } ^ { \mathrm { b } } ) = ( 6 , 0 0 0 , 6 , 0 0 0 )$ and the test UU data size $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } ) = ( 1 5 0 , 1 5 0 )$ to evaluate the proposed method with a larger-scale regime. Table 7 shows the results. The proposed method remains competitive in such larger-scale cases.

## D.9 Computation Costs

We evaluated the training time of the proposed method on MNIST(S) with $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } ) = ( 1 5 0 , 1 5 0 )$ We used a Linux server with a 2.20Hz CPU. For comparison, we also evaluated the methods that use both UU data in the training and the test distributions as in the proposed method. Table 8 shows the results. Since the proposed method estimated both importance weights and classifiers, it had slightly longer training time than mtsUU and mtUU. However, the differences were not significant. daUU had longer training time than the proposed method due to the calculation of the MMD loss to mitigate the feature discrepancy. This result indicates that the proposed method is practical in terms of computation costs.

## D.10 F1 Scores

Although we used test accuracies as the primary evaluation metric in the main paper, alternative metrics, such as F1 scores, can also be informative. Table 9 shows the average F1 scores for each dataset under the setting where data of both the training and test distributions are noisy. The proposed method also performed well in terms of the F1 scores.

## D.11 Full Results with Standard Deviations

Tables 10 and 11 show the average test accuracies with their standard deviations of each method.

Table 6: Results on larger-scale image data: average test accuracies over different class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Shift</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td rowspan="2">MNIST</td><td>S</td><td>0.7517</td><td>0.7511</td><td>0.5961</td><td>0.7530</td><td>0.7380</td><td>0.7625</td></tr><tr><td>IO</td><td>0.7284</td><td>0.7394</td><td>0.5429</td><td>0.6994</td><td>0.7086</td><td>0.6982</td></tr><tr><td rowspan="2">FMNIST</td><td>S</td><td>0.9700</td><td>0.9595</td><td>0.9256</td><td>0.9589</td><td>0.9511</td><td>0.9585</td></tr><tr><td>IO</td><td>0.8842</td><td>0.9045</td><td>0.7353</td><td>0.8808</td><td>0.8934</td><td>0.8841</td></tr><tr><td rowspan="2">CIFAR10</td><td>S</td><td>0.8481</td><td>0.7054</td><td>0.8542</td><td>0.8511</td><td>0.8500</td><td>0.8521</td></tr><tr><td>IO</td><td>0.7140</td><td>0.6922</td><td>0.6938</td><td>0.7236</td><td>0.6956</td><td>0.7291</td></tr></table>

Table 7: Results on larger-scale tabular data with distribution shift: average test accuracies over different class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td>DIABETES (large)</td><td>0.7240</td><td>0.6458</td><td>0.7261</td><td>0.7261</td><td>0.7199</td><td>0.7250</td></tr><tr><td>FOODSTAMP</td><td>0.7159</td><td>0.6314</td><td>0.7180</td><td>0.7136</td><td>0.6909</td><td>0.7170</td></tr></table>

Table 8: Training time [s] of the proposed method on MNIST(S).
<table><tr><td>Ours</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td>118.522</td><td>108.0755</td><td>107.771</td><td>143.969</td></tr></table>

Table 9: Average F1 scores over different UU test data sizes $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } )$ within {(50, 50), (100, 100), (150, 150)} and class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. In the Shift column, ‘S’ and ‘IO’ represent the support shift and input-output relation shift, respectively. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Shift</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td>MNIST</td><td>S IO</td><td>0.7328 0.7170</td><td>0.7325 0.7281</td><td>0.5038 0.4747</td><td>0.7253 0.6752</td><td>0.7317 0.6858</td><td>0.7295 0.6676</td></tr><tr><td>FMNIST</td><td>S IO</td><td>0.9610 0.8793</td><td>0.9527 0.8889</td><td>0.9107 0.7327</td><td>0.9537 0.8745</td><td>0.9492 0.8763</td><td>0.9537 0.8711</td></tr><tr><td>CIFAR10</td><td>S IO</td><td>0.8288 0.7174</td><td>0.6912 0.6873</td><td>0.8220 0.6501</td><td>0.8244 0.7058</td><td>0.8076 0.6964</td><td>0.8283 0.7074</td></tr><tr><td>DIABETES</td><td></td><td>0.7186</td><td>0.6368</td><td>0.7203</td><td>0.7210</td><td>0.7019</td><td>0.7219</td></tr></table>

Table 10: Average test accuracies with standard deviations over different UU test data sizes $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } )$ within {(50, 50), (100, 100), (150, 150)} and class-priors $( \theta ^ { \mathrm { a } } , \theta ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. In the Shift column, ‘S’ and ‘IO’ represent the support shift and input-output relation shift, respectively. Values in bold are not statistically different at the 5% level from the best performing method in each row according to a paired t-test.
<table><tr><td>Data</td><td>Shift</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td rowspan="2">MNIST</td><td>S</td><td>0.7482(0.059)</td><td>0.7419(0.083)</td><td>0.5879(0.035)</td><td>0.7388(0.062)</td><td>0.7355(0.094)</td><td>0.7423(0.060)</td></tr><tr><td>IO</td><td>0.7211(0.089)</td><td>0.7334(0.083)</td><td>0.5412(0.025)</td><td>0.6922(0.110)</td><td>0.6996(0.116)</td><td>0.6929(0.113)</td></tr><tr><td rowspan="2">FMNIST</td><td>S</td><td>0.9620(0.019)</td><td>0.9531(0.040)</td><td>0.9161(0.015)</td><td>0.9551(0.022)</td><td>0.9509(0.029)</td><td>0.9550(0.022)</td></tr><tr><td>IO</td><td>0.8841(0.063)</td><td>0.8929(0.055)</td><td>0.7278(0.047)</td><td>0.8796(0.069)</td><td>0.8823(0.064)</td><td>0.8769(0.072)</td></tr><tr><td rowspan="2">CIFAR10</td><td>S</td><td>0.8294(0.042)</td><td>0.6929(0.082)</td><td>0.8258(0.039)</td><td>0.8281(0.038)</td><td>0.8166(0.068)</td><td>0.8308(0.039)</td></tr><tr><td>IO</td><td>0.7125(0.061)</td><td>0.6864(0.077)</td><td>0.6665(0.052)</td><td>0.7084(0.060)</td><td>0.6947(0.072)</td><td>0.7099(0.063)</td></tr><tr><td>DIABETES</td><td></td><td>0.7090(0.033)</td><td>0.6300(0.059)</td><td>0.7115(0.033)</td><td>0.7117(0.032)</td><td>0.7011(0.051)</td><td>0.7118(0.031)</td></tr></table>

Table 11: Results with a few PN data in the test distribution: Average test accuracies with standard deviations over different UU test data sizes $( N _ { \mathrm { t e } } ^ { \mathrm { a } } , N _ { \mathrm { t e } } ^ { \mathrm { b } } )$ within {(10, 10), (50, 50), (100, 100)} and training class-priors $( \theta _ { \mathrm { t r } } ^ { \mathrm { a } } , \theta _ { \mathrm { t r } } ^ { \mathrm { b } } )$ within {(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)}. Class-priors of UU test data $( \theta _ { \mathrm { t e } } ^ { \mathrm { a } } , \theta _ { \mathrm { t e } } ^ { \mathrm { b } } )$ were set to (1, 0).
<table><tr><td>Data</td><td>Shift</td><td>Ours</td><td>teUU</td><td>trUU</td><td>mtsUU</td><td>mtUU</td><td>daUU</td></tr><tr><td rowspan="2">MNIST</td><td>S</td><td>0.8237(0.058)</td><td>0.8184(0.055)</td><td>0.6012(0.031)</td><td>0.8194(0.057)</td><td>0.8309(0.059)</td><td>0.8177(0.057)</td></tr><tr><td>IO</td><td>0.8075(0.057)</td><td>0.8102(0.055)</td><td>0.5543(0.020)</td><td>0.8012(0.064)</td><td>0.8127(0.063)</td><td>0.7998(0.074)</td></tr><tr><td rowspan="2">FMNIST</td><td>S</td><td>0.9726(0.013)</td><td>0.9655(0.024)</td><td>0.9181(0.015)</td><td>0.9719(0.014)</td><td>0.9705(0.018)</td><td>0.9717(0.014)</td></tr><tr><td>IO</td><td>0.9367(0.034)</td><td>0.9285(0.047)</td><td>0.7505(0.033)</td><td>0.9364(0.033)</td><td>0.9299(0.048)</td><td>0.9393(0.033)</td></tr><tr><td rowspan="2">CIFAR10</td><td>S</td><td>0.8492(0.027)</td><td>0.7555(0.083)</td><td>0.8286(0.035)</td><td>0.8422(0.028)</td><td>0.8376(0.029)</td><td>0.8466(0.027)</td></tr><tr><td>IO</td><td>0.7605(0.053)</td><td>0.7455(0.064)</td><td>0.6732(0.045)</td><td>0.7563(0.046)</td><td>0.7492(0.055)</td><td>0.7611(0.047)</td></tr><tr><td>DIABETES</td><td></td><td>0.7251(0.013)</td><td>0.6757(0.042)</td><td>0.7233(0.016)</td><td>0.7252(0.011)</td><td>0.7209(0.015)</td><td>0.7238(0.012)</td></tr></table>