# Eficient Pairwise Diference Learning: A Comparative Study of Encoding Strategies and Training Pair Selection

Bernard Georges<sup>1</sup>, Mohamed Karim Belaid<sup>2</sup>, and Manuel Rodrigues<sup>1</sup>

University of Minho, ALGORITMI CENTER LASI, Braga, Portugal pg53698@uminho.pt, manuel.rodrigues@algoritmi.uminho.pt 2 IDIADA Fahrzeugtechnik GmbH, Munich, Germany karim.belaid@idiada.com

Abstract. Pairwise Diference Learning for Classification (PDC) enhances supervised learning in low-data regimes by training a binary similarity classifier on pairs of instances rather than individual samples, efectively inflating the training set from n to O(n²) examples. However, the PDC framework leaves critical implementation choices underspecified, particularly regarding input representation and pair selection strategies. This work systematically investigates these design decisions through extensive empirical evaluation. We compare four feature encoders—paired features, triplet with relative diference, absolute diference, and a novel triplet with absolute diference—alongside four pair selection strategies ranging from exhaustive to kernel-thinning-based schemes. Experiments across nearly 200 benchmark classification datasets using Decision Tree, Multilayer Perceptron, Gaussian Naïve Bayes, and Support Vector Classifier as base models reveal that the proposed triplet with absolute difference encoder (ϕ<sub>3dabs</sub>) achieves the highest average F1 scores. More notably, restricting training to upper-triangle pairs reduces computational time by approximately 40% with negligible accuracy loss. Kernel thinning-based selection provided no accuracy benefits while substantially increasing computational overhead. Overall, PDC configurations outperformed base models in 94% of tested cases, though no single configuration dominated universally, suggesting dataset-specific optimization remains an open challenge.

Keywords: Pairwise Diference Learning · Classification · Low-data regimes · Feature encoding · Pair selection

## 1 Introduction

Many machine learning applications operate in regimes where labeled examples are limited or expensive to acquire. In such settings, simply increasing model complexity is often less efective than enriching the supervision signal extracted from the available data. Pairwise learning schemes ofer one such avenue: instead of learning from isolated samples, they create additional supervision by forming pairs of instances and encoding their relative relationships. Recent work has shown that pairwise methods can improve accuracy and robustness in regression and classification, especially in low-data regimes [11, 13, 6, 8, 10, 15, 4, 7, 1, 2, 12].

In this work, we focus on Pairwise Diference Learning for classification (PDC) [1]. PDC solves a multiclass classification task by training a binary “similar vs. dissimilar” classifier on pairs of examples and then using its outputs to reconstruct class posteriors. Intuitively, instead of asking the model to predict the label of a single point, PDC asks how likely two points are to share the same class. By pairing the n training points, the method inflates the number of efective training examples to $\scriptstyle { \mathcal { O } } ( n ^ { 2 } )$ , exposing the learner to rich comparative information that is particularly useful when data are scarce.

Formally, let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ with $x _ { i } \in \mathbb { R } ^ { d }$ and $y _ { i } \in \{ 1 , \ldots , K \}$ . PDC first selects a set of index pairs

$$
\mathcal { P } \subseteq \{ 1 , \ldots , n \} ^ { 2 } , \quad i \neq j ,
$$

and applies a feature map

$$
\phi : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { m } , \qquad z _ { i , j } = \phi ( x _ { i } , x _ { j } ) ,
$$

Which encodes joint information about the two inputs (for example, their concatenation and $/ \mathrm { o r }$ their diferences). Each pair receives a binary label.

$$
s _ { i , j } = \mathbf { 1 } \{ y _ { i } = y _ { j } \} ,
$$

Indicating whether the two instances share the same class. A base classifier $g$ is trained on the paired dataset.

$$
\mathcal { D } _ { \operatorname { p a i r } } = \{ ( z _ { i , j } , s _ { i , j } ) : ( i , j ) \in \mathcal { P } \} ,
$$

Learning to distinguish similar from dissimilar pairs. At prediction time, a query point x is compared to a set of training anchors $\{ x _ { j } \}$ , producing scores

$$
{ \hat { s } } _ { j } ( x ) = g { \big ( } \phi ( x , x _ { j } ) { \big ) } ,
$$

which estimate how likely x is to share its class with anchor $x _ { j }$ . Aggregating these scores over anchors of each class k yields an estimate of the posterior class probabilities, from which the final prediction is obtained by selecting the class with the highest estimated probability.

Research gap. While the PDC framework is conceptually simple, it leaves several important implementation choices open. In practice, diferent PDC variants $[ 4 ,$ $7 , 1 ]$ difer at least along two dimensions: (i) the input representation $\phi ( x , x ^ { \prime } )$ used for the similarity model $^ { g , }$ and (ii) the definition of the pair set ${ \mathcal { P } } _ { : }$ , that is, which pairs are generated and used for training and prediction. The original PDC variant of [1] adopts a specific encoding and an almost exhaustive triplet-based pairing scheme, which can be computationally expensive and may include many weakly informative pairs. To the best of our knowledge, there is no systematic comparison of alternative input formats and pairing strategies within a unified experimental protocol.

Contributions. We build on the PDC formulation of [1] and systematically study how these implementation choices afect its behaviour in practice. Our contributions are:

1. We compare three diferent input formats for the similarity model in PDC, varying how the two instances and their diferences are represented in $\phi ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } )$ and study their impact on predictive performance and computational cost.

2. We compare four pairing strategies for PDC, ranging from exhaustive to more selective schemes, and analyze how the choice of P influences accuracy, training time, and efective dataset size.

3. We conduct an empirical evaluation on a suite of benchmark classification datasets, providing practical guidance on which PDC design choices are most efective under diferent data and resource constraints.

## 2 Related Work

Early work by [11] proposed a pairwise scheme in which each instance was represented by its features along with a pairing weight, encoding similarity between vectors through a Gaussian kernel and feature-wise diferences. Independently, [14] introduced the twin-network paradigm. This approach employs two parallel neural networks with shared weights to project instances into a common latent space, where a distance function quantifies the degree of similarity or dissimilarity between inputs.

Subsequent contributions focused on structured and additive modeling. [6, 8] developed pairwise strategies for structured data by applying Radial Basis Function (RBF) transformations to create augmented points, which are then paired with the original instances. In parallel and independently, [10] worked on both the theoretical and practical scope of pairwise learning in chemical data, proposing the method later adopted in PDC, where instances are paired using both their original features and their diferences.

Further advances addressed automation and scalability. [15] demonstrated the feasibility of scaling pairwise models to large, complex datasets, marking a shift toward practical applicability. This direction was reinforced by [4], who emphasized evaluating pairwise models under heterogeneous and distributionally shifted data, using a pairing strategy based solely on absolute diferences.[12] ofered a critical perspective by analyzing the theoretical limitations of pairwise methods, informing more principled model design. Both approaches incorporated point selection algorithms, leveraging random subsampling of the majority class and k-nearest neighbors, respectively.

The most recent work has emphasized refinement and adaptation. [7] proposed adaptive pairwise strategies that allow models to dynamically adjust to evolving input distributions, highlighting the growing importance of flexibility in modern applications. Complementing this, [1, 2] introduced pairwise weighting mechanisms to balance data contributions during training, advancing fairness, generalization, and interpretability in pairwise learning frameworks.

Regarding point selection, [5] proposed a coreset-based approach that reduces the original dataset by iteratively partitioning instances into two balanced subsets using kernel computations, followed by selecting the optimal subset as the resulting dataset.

## 3 Method

## 3.1 Feature representation of a pair of datapoints

A central design choice in PDC is how to represent a pair of inputs before passing it to the similarity model g. Given two datapoints $x _ { i } , x _ { j } \in \mathbb { R } ^ { d }$ , an encoder

$$
\phi : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R } ^ { m }
$$

Maps the pair $( x _ { i } , x _ { j } )$ to a feature vector $z _ { i , j } = \phi ( x _ { i } , x _ { j } )$ , which is then used to predict whether the two instances belong to the same class. Diferent encoders trade of information content, symmetry, and dimensionality, and may therefore lead to diferent accuracy–cost profiles.

In the following, we consider four natural encoders that are all compatible with the same PDC pipeline (same classifier $g$ and same aggregation rule). They difer only in how the pair $( x _ { i } , x _ { j } )$ is represented.

Paired features $\left( \phi _ { \mathrm { 2 d } } \right)$ . The simplest encoder keeps only the original features of the two datapoints, without any explicit diference term, an approach used in several pairwise classification schemes [8, 6, 7]:

$$
\begin{array} { r } { \phi _ { 2 \mathrm { d } } ( x _ { i } , x _ { j } ) = \left( x _ { i } , x _ { j } \right) \in \mathbb { R } ^ { 2 d } . } \end{array}\tag{1}
$$

Here, the model is expected to infer useful similarity or diference patterns internally from the concatenated inputs. The dimensionality is $2 d ,$ and the encoder exposes the full original feature vectors to the classifier without imposing any particular structure on their relationship.

Triplet with relative diference $( \phi _ { \mathrm { 3 d r e l } } )$ . A common PDC-style representation [10, 1] extends this idea by explicitly adding the signed, component-wise diference between the two inputs:

$$
\phi _ { \mathrm { 3 d r e l } } ( x _ { i } , x _ { j } ) = \left( x _ { i } , x _ { j } , x _ { i } - x _ { j } \right) \in \mathbb { R } ^ { 3 d } .\tag{2}
$$

This encoder concatenates both inputs and their relative diference. The representation is directional: swapping i and $j$ changes the sign of the last d components. The additional block provides the classifier with explicit information about how the two instances difer along each feature dimension.

Absolute diference $( \phi _ { \mathrm { a b s } } )$ . A more compact alternative, inspired by SAME-style representations [4], discards the original features and keeps only the componentwise absolute diference:

$$
\phi _ { \mathrm { a b s } } ( x _ { i } , x _ { j } ) = \left| x _ { i } - x _ { j } \right| \in \mathbb { R } ^ { d } ,\tag{3}
$$

where subtraction and absolute value are applied element-wise. This encoder is invariant to the order of the pair, i.e., $\phi _ { \mathrm { a b s } } ( x _ { i } , x _ { j } ) = \phi _ { \mathrm { a b s } } ( x _ { j } , x _ { i } )$ , so symmetric pairs do not need to be generated. It has the lowest dimensionality among the encoders considered here, which may reduce computational cost, but it also removes information about the absolute position of $x _ { i }$ and $x _ { j }$ in feature space.

Triplet with absolute diference $\left( \phi _ { \mathrm { 3 d a b s } } \right)$ . Finally, we propose a non-directional triplet representation that combines the full original features with an orderinvariant diference term:

$$
\phi _ { \mathrm { 3 d a b s } } ( x _ { i } , x _ { j } ) = \left( x _ { i } , x _ { j } , | x _ { i } - x _ { j } | \right) \in \mathbb { R } ^ { 3 d } .\tag{4}
$$

Compared to $\phi _ { \mathrm { 3 d r e l } }$ , this encoder removes the sign of the diference while retaining both original feature vectors. It therefore preserves essentially the same information, but it no longer encodes which of the two points is “larger” along a given dimension. This can simplify the learning task for $g$ by making the representation less sensitive to the ordering of the pair, while still providing a rich description of both absolute and relative structure.

Illustrative example. To make the four encoders concrete, consider two Titanic passengers described by two numeric features: Age (in years) and Fare (ticket price in US dollars). Let

$$
x _ { i } = ( 2 2 , 8 ) , \qquad x _ { j } = ( 3 8 , 7 2 ) ,
$$

so that

$$
x _ { i } - x _ { j } = ( - 1 6 , - 6 4 ) , \qquad | x _ { i } - x _ { j } | = ( 1 6 , 6 4 ) .
$$

Table 1 shows the resulting representations. For the same underlying pair $( x _ { i } , x _ { j } )$ $\phi _ { \mathrm { 2 d } }$ keeps only who the passengers are, $\phi _ { \mathrm { a b s } }$ keeps only how much they difer, $\phi _ { \mathrm { 3 d r e l } }$ augments the concatenation with signed diferences, and $\phi _ { \mathrm { 3 d a b s } }$ combines the concatenation with absolute diferences in a single feature vector.

## 3.2 Input pair selection

Beyond the feature encoder $\phi ,$ , PDC also depends on how many and which pairs are actually generated during training. Given a training set

$$
D = \{ x _ { 1 } , \ldots , x _ { n } \} ,
$$

The naïve construction forms all ordered pairs

$$
D \times D = \{ ( x _ { i } , x _ { j } ) : 1 \leq i , j \leq n \} ,
$$

Table 1: Example of the four pair encoders on two Titanic passengers with features Age (years) and Fare (US\$).
<table><tr><td>Encoder</td><td>Output vector  $\phi ( x _ { i } , x _ { j } )$ </td></tr><tr><td> $\boxed { \phi _ { 2 \mathrm { d } } ( x _ { i } , x _ { j } ) }$   $\phi _ { \mathrm { 3 d r e l } } ( x _ { i } , x _ { j } )$   $\phi _ { \mathrm { a b s } } ( x _ { i } , x _ { j } )$ </td><td> $\overline { { ( 2 2 , 8 , 3 8 , 7 2 ) } }$   $( 2 2 , 8 , 3 8 , 7 2 , - 1 6 , - 6 4 )$   $( 1 6 , ~ 6 4 )$   $\left| \phi _ { 3 \mathrm { d a b s } } ( x _ { i } , x _ { j } ) \right| ( 2 2 , 8 , 3 8 , 7 2 , 1 6 , 6 4 )$ </td></tr></table>

Which can be visualized as an n×n matrix of pairs. This “full grid” includes selfpairs $( x _ { i } , x _ { i } )$ on the diagonal and symmetric duplicates $( x _ { i } , x _ { j } )$ and $( x _ { j } , x _ { i } )$ of the diagonal. While this maximizes the number of training examples $( n ^ { 2 } \ \mathrm { p a i r s } )$ 2 it also introduces redundancy and increases computational cost.

In this work, we model pair selection as an operator.

$$
\delta : \{ 1 , \ldots , n \} ^ { 2 }  \{ 0 , 1 \} ,
$$

which indicates whether a pair $( i , j )$ is kept $( \delta ( i , j ) = 1 )$ or discarded $( \delta ( i , j ) = 0 )$ We consider four patterns, which can be visualized as diferent “shadings” of the $n \times n$ pair matrix.

All pairs $( \delta _ { \mathbf { \delta \mathbf { \delta \mathbf { u } } } } )$ . The baseline pattern keeps all ordered pairs, including both symmetric counterparts and the diagonal:

$$
\mathcal { P } _ { \overline { { \mathbf { u } } } } = \{ ( i , j ) : 1 \leq i , j \leq n \} , \qquad | \mathcal { P } _ { \overline { { \mathbf { u } } } } | = n ^ { 2 } .
$$

This setting maximizes the training set size but also maximizes redundancy between $( i , j )$ and $( j , i )$ , especially when $\phi$ is symmetric (e.g., absolute-diference encoders).

Unordered pairs $( \delta \mathfrak { s } )$ . To remove symmetric duplicates while keeping self-pairs, we restrict to the upper triangle (including the diagonal):

$$
{ \mathcal { P } } _ { \mathbb { q } } = \{ ( i , j ) : 1 \leq i \leq j \leq n \} , \qquad \left| { \mathcal { P } } \right| = { \frac { n ( n + 1 ) } { 2 } } .
$$

Each unordered pair $\{ i , j \}$ appears exactly once, and diagonal elements $( i , i )$ are retained. This preserves self-pairs, which can be interpreted as “perfect matches” and may serve as a useful reference for the similarity model.

non symmetric unordered pairs $( \delta \cdot )$ . A more aggressive reduction removes both symmetric duplicates and self-pairs by keeping only the strict upper triangle:

$$
{ \mathcal { P } } { \mathsf { * } } = \{ ( i , j ) : 1 \leq i < j \leq n \} , \qquad | { \mathcal { P } } { \mathsf { * } } | = { \frac { n ( n - 1 ) } { 2 } } .
$$

This pattern eliminates self-pairs and uses each unordered pair exactly once. It allows us to test whether self-pairs are actually beneficial or merely inflate the training set with easy, potentially uninformative examples.

Kernel thinning $( \delta _ { * } )$ . We additionally consider a data–dependent selection pattern called kernel thinning [5], in contrast to $\delta _ { \overline { { \mathbf { I } } } }$ , δ and $\delta \bullet .$ which keep or discard pairs solely based on their position in the $n \times n { \mathrm { ~ g r i d } }$ . Kernel thinning selects a subset of points that is well spread with respect to a chosen kernel, with the goal of producing a smaller training set that better represents the underlying distribution, thereby reducing imbalance and mitigating overfitting. Formally, it takes the original training set D of size n and returns a thinned subset $D ^ { ( m ) }$ of size approximately $h ( n ) = n / 2 ^ { m }$ , where $m \in \mathbb { N }$ is a splitting parameter: each increment in m doubles the number of thinning steps and halves the dataset size.

In our notation, δ denotes the mask induced by applying kernel thinning either before or after pair construction, and $\left| \mathcal { P } _ { \mathrm { k t } } \right|$ corresponds to the number of pairs that the similarity model receives. If thinning is applied before pairing, we first obtain $D ^ { ( m ) }$ of size $h ( n ) = n / 2 ^ { m }$ data points and then build unordered pairs using $\delta _ { \mathfrak { s } . }$ . The resulting number of pairs is

$$
\left| \mathcal { P } _ { * } ^ { \mathrm { p r e } } \right| = \frac { h ( n ) \cdot \left( h ( n ) + 1 \right) } { 2 } = \frac { n \left( n + 2 ^ { m } \right) } { 2 ^ { 2 m + 1 } } .
$$

If thinning is applied after pairing, we first construct $\mathcal { P } _ { \daleth }$ of size $n ( n + 1 ) / 2$ and then thin the set of pairs by the same factor $2 ^ { m }$ , yielding.

$$
\left| \mathcal { P } _ { \ast } ^ { \mathrm { p o s t } } \right| = h ( \left| \mathcal { P } \right| ) = \frac { n ( n + 1 ) } { 2 ^ { m + 1 } } .
$$

In practice, we restrict m to small values (here $m \in \{ 1 , 2 \} )$ to avoid excessively small training sets, and we primarily use pre-pairing thinning to better respect the i.i.d. assumption on the inputs. This data–adaptive mask $\delta _ { * }$ therefore provides a fourth, complementary way to control the efective number of training pairs seen by PDC, trading of computational cost against distributional coverage.

Illustrative example. To visualize the efect of the diferent selection patterns, consider a toy dataset of four datapoints represented by the four card suits,

$$
D = ( x _ { 1 } , x _ { 2 } , x _ { 3 } , x _ { 4 } ) = ( \bullet , \odot , \diamond , \bullet ) .
$$

Table 2 shows how $D \times D$ is transformed by the masks $\delta _ { \overline { { \mathbf { u } } } } , \delta _ { \overline { { \mathbf { \Theta } } } } , \delta _ { \overline { { \mathbf { \Theta } } } } ,$ and $\delta _ { * }$ (with $m = 2 )$ . The first three patterns are purely index-based: they keep all pairs, only the upper triangle with diagonal, or only the strict upper triangle, respectively. The kernel-thinning pattern δ instead keeps half of the pairs (here illustrated by an arbitrary subset of cells).

## 4 Evaluation

## 4.1 Benchmarking datasets

The datasets used in these experiments are from the OpenML repository, [3]. These datasets were selected and used by [1], and in order to ensure consistency and continuity across all experimental runs, the PDC application contains a parquet file that contains the IDs of the used datasets. This list is then used to retrieve the datasets needed from the OpenML repository. Furthermore, in order to confirm the use of the same datasets across all analysis, in case of errors or other random events during execution, the meta-algorithm extracts key descriptors from the API and dataset, including the number of features, number of instances, class distribution and other 8 descriptor values that are then cross-verified to ensure that the identical datasets will be used across all comparisons and analyses done with the results.

Table 2: Toy example of pair-selection patterns on $D = ( \bullet , \odot , \diamond , \bullet )$ . Each matrix entry denotes the ordered pair $( x _ { i } , x _ { j } )$ ; empty cells indicate discarded pairs. For δ with $m = 1$ , an arbitrary half of the pairs are kept.  
![](images/bc92225428341a4f595eede5f6fc127b969ba293ce0bd474fd7f28206e1c5081.jpg)

PDC also performs some basic pre-processing on the datasets utilising the methods available in the [9] library. Namely, it uses OrdinalEncoder, OneHotEncoder and StandardScaler. After which, it drops columns containing null values. In the wake of the pre-processing, the data is then put through the crossvalidation method that divides it into 5 folds. These steps were not altered, meaning that a faithful reproduction can be obtained through the algorithm presented in the original repository.

## 4.2 Base ML models

PDC is a meta-algorithm, meaning that it works on top of multiple ML models. These models have diferent paradigms, meaning diferent requirements and strategies. Due to this, 4 paradigms were chosen, each represented by a model. Decision Tree is our baseline model, meaning it will be used to test all initial approaches; the best results will then also be used with the other 3 models.

– Decision Tree (DT) - Its interpretability, simplicity, and ability to handle both categorical and numerical data make it one of the most widely used models.

Multilayer Perceptron Classifier (MLP) - MLP is a neural network. These models are capable of approximating highly non-linear decision boundaries

– Gaussian Naïve Bayes Classifier (GNB) - A probabilistic model, meaning it computes the likelihood of each possible class and assigns the instance to the class with the highest probability.

– Support Vector Classifier (SVC) - based on the principles of Support Vector Machines (SVMs), it aims to find an optimal separating hyperplane in the feature space that maximises the margin between classes.

## 4.3 Metric

This study focused primarily on two metrics: accuracy and computational time. For accuracy, the selected metric was the $F _ { 1 }$ score as it provides a balanced and interpretable measure of model performance and is particularly suited for evaluating imbalanced datasets, such as those generated by the PDL algorithm. The analysis further extends beyond individual dataset-specific $F _ { 1 }$ scores to include the mean $F _ { 1 }$ score, $\bar { F } _ { 1 }$ , computed across all datasets within a given experiment. As for computational performance, it is primarily assessed through the total execution time of each run, including training and testing time. To further validate the results, following the identification of the most efective improvements on these metrics, a t-test and Bayesian posterior coeficients are conducted to determine the statistical significance and validity of the results with 95% significance.

## 4.4 Results & Discussion

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\phi _ { \mathrm { 3 d r e l } }$ </td><td rowspan=1 colspan=1> $\phi _ { \mathrm { a b s } }$ </td><td rowspan=1 colspan=1> $\phi _ { \mathrm { 3 d a b s } }$ </td></tr><tr><td rowspan=2 colspan=1>All Pairs $\overline { { ( \delta \mathbf { \mathbf { u } } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { F _ { 1 } } }$ </td><td rowspan=1 colspan=1>0.804 ± 0.1510</td><td rowspan=1 colspan=1>.797 ± 0.154|</td><td rowspan=1 colspan=1>0.810 ± 0.150</td></tr><tr><td rowspan=1 colspan=1>Time [min]</td><td rowspan=1 colspan=1> $\overline { { 8 1 } }$ </td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>81</td></tr><tr><td rowspan=2 colspan=1>UT w/ diagonal (δ)</td><td rowspan=1 colspan=1> $F _ { 1 }$ </td><td rowspan=1 colspan=1>0.806 ± 0.148|</td><td rowspan=1 colspan=1>0.796 ± 0.155|</td><td rowspan=1 colspan=1>0.810 ± 0.148</td></tr><tr><td rowspan=1 colspan=1>Time [min]</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>44</td></tr><tr><td rowspan=2 colspan=1>UT w/o diagonal (δ)</td><td rowspan=1 colspan=1> $\overline { { F _ { 1 } } }$ </td><td rowspan=1 colspan=1>0.806 ± 0.148</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>0.809 ± 0.147</td></tr><tr><td rowspan=1 colspan=1>Time [min]</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>48</td></tr></table>

Table 3: $F _ { 1 }$ score Comparison of diferent input combinations

As reported in Table 3, the $\phi _ { \mathrm { 3 d a b s } }$ method achieved the highest average $F _ { 1 }$ scores. This was achieved both with $\delta _ { \overline { { \mathbf { I } } } }$ and $\delta { \ * }$ yielding 0.810, meaning improvements of 0.006 and 0.004 over the baseline, $\phi _ { \mathrm { 3 d r e l } } .$ , respectively.

![](images/22059b30e277969641654751794a475e038760cc5cc90245ec1f283c82576b89.jpg)  
Fig. 1: Histogram of the $F _ { 1 }$ score diference of between a method and $\phi _ { \mathrm { 3 d r e l } }$ 1

Nonetheless, when closely examined and as presented in figure 1, $\phi _ { \mathrm { a b s } }$ presents higher amounts of big improvements, improvements greater than 0.05, while also having more deterioration, negative diferences. These diferentials demonstrate that even though $\phi _ { \mathrm { 3 d r e l } }$ presents higher averages than $\phi _ { \mathrm { a b s } }$ , the later still presents good functionalities in specific datasets. Furthermore, when examined closely and subjected to a paired statistical test such as Student t-test, no statistically significant efects were detected meaning the improvements may have been too inconsistent.

The most substantial benefit of these representations lies instead within the $\delta { \ * }$ method, which consistently presented the best or a slightly lower $F _ { 1 }$ score with an average of 40% decrease on the time taken to accomplish the experiments.

<table><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>BaselineBefore</td><td rowspan=1 colspan=1>Pairing $\overline { { ( \mathrm { m } - 1 ) } }$ </td><td rowspan=1 colspan=1>After Pairing $\overline { { ( \mathrm { m } { = } 1 ) } }$ </td><td rowspan=1 colspan=1>After Pairing $\overline { { ( \mathrm { m } { = } 2 ) } }$ </td></tr><tr><td rowspan=1 colspan=2>Results (80 DS)</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.811</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.825</td></tr><tr><td rowspan=1 colspan=1>Time [m</td><td rowspan=1 colspan=1>in]</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>209</td><td rowspan=1 colspan=1>147</td></tr></table>

Table 4: Comparison of $F _ { 1 }$ and run time across Diferent Kernel Thinning Settings for 80 Datasets per experiment

For the point selection, the results did not present any improvements, with the best case when selection is done after pairing. Nonetheless, in this case, the computation complexity presents a major increase, with an increase of 290% or 207 minutes, and no improvement in the accuracy.

With these results in mind, the best improvements and chosen methods to continue experimentation on the remaining 3 models were $\phi _ { \mathrm { 3 d a b s } }$ combined with $\delta _ { \overline { { \mathbf { I } } } }$ and $\delta _ { \vec { \mathbf { \nabla } } } .$ , and $\phi _ { \mathrm { 2 d } }$

<table><tr><td rowspan=1 colspan=1>Model (num of datasets)</td><td rowspan=1 colspan=1>No PDL</td><td rowspan=1 colspan=1> $\phi _ { \mathrm { 3 d r e l } } \circ \delta _ { \overline { { \mathbf { \Theta } } } }$ </td><td rowspan=1 colspan=1> $\phi _ { \mathrm { 3 d a b s } } \circ \delta _ { \overline { { \mathbf { u } } } }$ </td><td rowspan=1 colspan=1> $\phi _ { \mathrm { 3 d a b s } } \circ \delta _ { \vec { \mathbf { \imath } } }$ </td><td rowspan=1 colspan=1> $\phi _ { \mathrm { 2 d } } \circ \delta _ { \overline { { \mathbf { { \Pi } } } } }$ </td></tr><tr><td rowspan=1 colspan=1>MLP (187 DS)</td><td rowspan=1 colspan=1> $\overline { { 0 . 7 5 9 \pm 0 . 1 6 8 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 7 9 4 \pm 0 . 1 6 8 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 8 1 0 \pm 0 . 1 5 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 8 0 3 \pm 0 . 1 6 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 7 9 4 \pm 0 . 1 6 9 } }$ </td></tr><tr><td rowspan=1 colspan=1>GaussianNB (195 DS)</td><td rowspan=1 colspan=1> $\overline { { 0 . 7 2 7 \pm 0 . 1 7 9 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 6 0 3 \pm 0 . 2 3 0 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 6 9 1 \pm 0 . 2 1 2 } }$ </td><td rowspan=1 colspan=1> $0 . 6 8 4 \pm 0 . 2 1 5$ </td><td rowspan=1 colspan=1>0.327 ± 0.088</td></tr><tr><td rowspan=1 colspan=1>SVC (70 DS)</td><td rowspan=1 colspan=1> $\overline { { 0 . 8 6 9 \pm 0 . 1 5 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 8 7 1 \pm 0 . 1 5 7 } }$ </td><td rowspan=1 colspan=1> $0 . 8 8 0 \pm 0 . 1 4 1 |</td><td rowspan=1 colspan=1>0 . 8 8 0 \pm 0 . 1 4 4$ </td><td rowspan=1 colspan=1>N/A</td></tr><tr><td rowspan=1 colspan=1>Decision Tree (195 DS)</td><td rowspan=1 colspan=1> $\overline { { 0 . 7 8 2 \pm 0 . 1 5 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 8 0 4 \pm 0 . 1 5 1 } }$ </td><td rowspan=1 colspan=1> $0 . 8 1 0 \pm 0 . 1 5 0 |</td><td rowspan=1 colspan=1>0 . 8 1 0 \pm 0 . 1 4 8$ </td><td rowspan=1 colspan=1> $\overline { { \mathrm { ~ N ~ } / \mathrm { ~ A ~ } } }$ </td></tr></table>

Table 5: $F _ { 1 }$ score of every experiment done in the diferent models

MLP presented the best $F _ { 1 }$ improvement with 0.016 for $\phi _ { \mathrm { 3 d a b s } } \circ \delta _ { \overline { { \mathbf { \theta } } } }$ this was closely followed by $\phi _ { \mathrm { 3 d a b s } } \circ \delta \mathfrak { s }$ with 0.803. These results presented the general trend of all the results with $\phi _ { \mathrm { 3 d a b s } } \circ \delta _ { \overline { { \mathbf { \theta } } } }$ having the highest score, closely followed by $\phi _ { \mathrm { 3 d a b s } } \circ \delta \mathfrak { s }$ with either the same $F _ { 1 }$ or a slight decrease. This came with great improvements in time and complexity. In the case of MLP, there was a 54% decrease in time. These results, similarly to what happened to DT, did not present significance with the t-test. Nevertheless, they did present a Bayesian probability higher than 75% of being better than $\phi _ { \mathrm { 3 d r e l } } \circ \delta _ { \mathbf { \overline { { \mathbf { \Lambda } } } } }$ , which is significant.

As for GNB, the result showed the highest increase with an increase of 0.088 in $F _ { 1 }$ between $\phi _ { \mathrm { 3 d r e l } } \circ \delta _ { \overline { { \mathbf { \Theta } } } }$ and $\phi _ { \mathrm { 3 d a b s } } \circ \delta _ { \overline { { \mathbf { \theta } } } }$ . Nonetheless, this improvement is not enough to beat the base model without PDC, this being the only case where PDC presented a deterioration when compared to the model.

SVC maintains the trend with the highest $F _ { 1 }$ result. This model, however, also showed the highest amount of prediction errors, with 10 datasets not being able to complete their predictions due to the model not predicting any instance as the class.

![](images/5143c900ad879734e20f542aff81d0c3400dca0adf88fba23f1853c1768e2508.jpg)  
Fig. 2: Pie Chart of All Methods with Best Results per Dataset<sup>2</sup>

![](images/9bb0f6c8009e9e1234e975204269fccb68df1f0750cf891cd53c96aadf533245.jpg)  
Fig. 3: Pie Chart of the PDC Results vs Base Model Results per Dataset<sup>2</sup>

For a final analysis, it was comprised all the $F _ { 1 }$ scores across all the common datasets done through the experiment. The chart shows the amount of datasets in which each method presented the best result with a precision of 5 decimal points. As seen in Figure 3 in its entirety PDC generally presented improvements in the $F _ { 1 }$ score, with only 4 out of 70 cases reporting a better score in the base model. However, as presented in Figure 2, no single method consistently presents the best score, with the highest amount being presented by $\phi _ { \mathrm { a b s } } \circ \delta $ , 22 datasets which is 30% of the total datasets. This example further demonstrates the inconsistency of such methods, which achieve great improvements while also having a low averages.

## 5 Conclusion

A closer inspection of the prediction evaluation revealed that a significant improvement is possible on a per-dataset basis, where the PDL, when evaluated collectively in its diferent configurations, was able to reach the best $F _ { 1 }$ score in most tested cases. Nevertheless, no single configuration can be distinguished as each presents some datasets in which it outperforms its peers. To pursue the idea of an optimiser that selects the most adequate configuration for a given field of application or dataset class is once again beyond the scope of this study.

In spite of the absence of statistical significance, the obtained results provided good insights into the working of the PDC algorithm, particularly in the design and evaluation of its training phase. During the study, some of the proposed modifications presented promising results, particularly with the pairing function $\phi _ { \mathrm { 3 d a b s } }$ demonstrating strong performance in three cases, namely when the meta-algorithm used either the DT, MLP, or SVC as its underlying base models. Moreover, the investigation into the training scope has also revealed an interesting trade-of. As training in the upper triangle, $\delta _ { \daleth }$ , has shown a statistically non-significant decrease in accuracy when compared to its full matrix counterpart, $\delta _ { \overline { { \mathbf { I } } } }$ , it significantly decreased the complexity and time spent in training.

On the other hand, the research experimented with the use of point selection during the training phase, which was subsequently discarded as a valid enhancement feature. In fact, the point selection method resulted in a great increase in time with no gratification presented in return for its drawback. Beyond this, the results corroborated the finding mentioned in [1] regarding the inefectiveness of the GNB classifier as a base model for the PDC meta-algorithm. The collected results presented with 95% certainty a significant decrease in $F _ { 1 }$ scores when compared to other base models.

## References

1. Belaid, M.K., Rabus, M., Hüllermeier, E.: Pairwise diference learning for classification. In: International Conference on Discovery Science. pp. 284–299. Springer (2024)

2. Belaid, M.K., Wibiral, T., Rabus, M., Hüllermeier, E.: Weighted Pairwise Diference Learning. In: proc. of LWDA’24. CEUR-WS.org, Würzburg (2024)

3. Bischl, B., Casalicchio, G., Feurer, M., Gijsbers, P., Hutter, F., Lang, M., Mantovani, R.G., van Rijn, J.N., Vanschoren, J.: Openml: A benchmarking layer on top

of openml to quickly create, download, and share systematic benchmarks. NeurIPS (2021), https://datasets-benchmarks-proceedings.neurips.cc/paper\_files/ paper/2021/hash/c7e1249ffc03eb9ded908c236bd1996d-Abstract-round2.html

4. Corbara, S., Moreo, A., Sebastiani, F.: Same or diferent? dif-vectors for authorship analysis. ACM Transactions on Knowledge Discovery from Data 18(1), 1–36 (2023)

5. Dwivedi, R., Mackey, L.: Kernel thinning. Journal of Machine Learning Research 25(152), 1–77 (2024)

6. Izonin, I., Tkachenko, R., Dronyuk, I., Tkachenko, P., Gregus, M., Rashkevych, M.: Predictive modeling based on small data in clinical medicine: Rbf-based additive input-doubling method. Math. Biosci. Eng. (2021)

7. Izonin, I., Tkachenko, R., Havryliuk, M., Gregus, M., Yendyk, P., Tolstyak, Y.: An adaptation of the input doubling method for solving classification tasks in case of small data processing. Procedia Computer Science (2024)

8. Izonin, I., Tkachenko, R., Shakhovska, N., Lotoshynska, N.: The additive inputdoubling method based on the svr with nonlinear kernels. Symmetry (2021)

9. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011)

10. Tynes, M., Gao, W., Burrill, D.J., Batista, E.R., Perez, D., Yang, P., Lubbers, N.: Pairwise diference regression: a machine learning meta-algorithm for improved prediction and uncertainty quantification in chemical search. Journal of chemical information and modeling 61(8), 3846–3857 (2021)

11. Vitynskyi, P., Tkachenko, R.: Neiropodibna struktura dlya zadach prohnozuvannya v umovakh korotkykh vyborok danykh. Sci. Bull. UNFU (2019)

12. Wetzel, S.J.: How to get the most out of Twinned Regression (Jan 2023)

13. Wetzel, S.J., Ryczko, K., Melko, R.G., Tamblyn, I.: Twin neural network regression. Applied AI Letters 3(4), e78 (2020)

14. Wetzel, S.J., Ryczko, K., Melko, R.G., Tamblyn, I.: Twin neural network regression. Applied AI Letters 3(4), e78 (2022)

15. Xie, J., Kaiwei, C., Kong, L., Zhou, J., Qu, W.: Automated essay scoring via pairwise contrastive regression. In: proc. of the 29th COLING (2022)